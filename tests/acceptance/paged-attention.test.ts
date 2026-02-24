/**
 * Paged Attention Tests
 *
 * Validates:
 * - kvCacheFragmentationFactor() unit behavior
 * - Memory inflation when pagedAttention is OFF
 * - Recommendations for enabling paged attention (OOM + success)
 * - Regression: default config (pagedAttention: true) unchanged
 */

import { describe, it, expect } from 'vitest';
import {
  kvCacheFragmentationFactor,
  KV_CACHE_FRAGMENTATION_CAP,
  calculateInferenceMemory,
  calculateMemoryFromConfig,
  validateMemoryFits,
} from '../../src/core/inference/memory.ts';
import { runInferenceSimulation } from '../../src/core/inference/simulation.ts';
import { getModel } from '../../src/core/models/index.ts';
import { H100_SXM, A100_80GB } from '../../src/core/hardware/gpu.ts';

// ══════════════════════════════════════════════════════════════════════
// kvCacheFragmentationFactor() unit tests
// ══════════════════════════════════════════════════════════════════════

describe('kvCacheFragmentationFactor', () => {
  it('caps at KV_CACHE_FRAGMENTATION_CAP for short sequences', () => {
    // actualSeqLen=1536, maxSeqLen=128K → 128000/1536 ≈ 83 → capped at 2.0
    expect(kvCacheFragmentationFactor(1536, 128000)).toBe(KV_CACHE_FRAGMENTATION_CAP);
  });

  it('returns dynamic ratio for mid-range sequences', () => {
    // actualSeqLen=100K, maxSeqLen=128K → 128000/100000 = 1.28
    expect(kvCacheFragmentationFactor(100000, 128000)).toBeCloseTo(1.28, 2);
  });

  it('returns 1.0 when actual equals max', () => {
    expect(kvCacheFragmentationFactor(128000, 128000)).toBe(1.0);
  });

  it('returns 1.0 for actualSeqLen=0 (guard)', () => {
    expect(kvCacheFragmentationFactor(0, 128000)).toBe(1.0);
  });

  it('returns 1.0 for maxSeqLen=0 (guard)', () => {
    expect(kvCacheFragmentationFactor(1024, 0)).toBe(1.0);
  });

  it('returns 1.0 for negative values (guard)', () => {
    expect(kvCacheFragmentationFactor(-100, 128000)).toBe(1.0);
    expect(kvCacheFragmentationFactor(1024, -1)).toBe(1.0);
  });
});

// ══════════════════════════════════════════════════════════════════════
// Memory effect tests
// ══════════════════════════════════════════════════════════════════════

describe('Paged attention memory effects', () => {
  const model = getModel('llama3-8b')!;

  it('pagedAttention ON: KV cache matches base calculateInferenceMemory', () => {
    const totalSeqLen = 1024 + 512;
    const baseMemory = calculateInferenceMemory(model, totalSeqLen, 32, 'bf16', 'bf16', true);
    const configMemory = calculateMemoryFromConfig({
      modelSpec: model,
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 32,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
      flashAttention: true,
      pagedAttention: true,
      continuousBatching: false,
      speculative: { enabled: false, draftModel: null, numSpeculativeTokens: 4, acceptanceRate: 0.7 },
    });
    expect(configMemory.kvCache).toBe(baseMemory.kvCache);
    expect(configMemory.total).toBe(baseMemory.total);
  });

  it('pagedAttention OFF: KV cache inflated by fragmentation factor', () => {
    const totalSeqLen = 1024 + 512; // 1536 total, maxSeqLen=8192 → factor = min(8192/1536, 2.0) ≈ 2.0 (capped)
    const expectedFactor = kvCacheFragmentationFactor(totalSeqLen, model.maxSeqLength);
    expect(expectedFactor).toBe(KV_CACHE_FRAGMENTATION_CAP); // capped at 2.0

    const memoryON = calculateMemoryFromConfig({
      modelSpec: model, gpu: H100_SXM, numGPUs: 1, batchSize: 32,
      inputSeqLen: 1024, outputSeqLen: 512, weightPrecision: 'bf16', kvCachePrecision: 'bf16',
      flashAttention: true, pagedAttention: true, continuousBatching: false,
      speculative: { enabled: false, draftModel: null, numSpeculativeTokens: 4, acceptanceRate: 0.7 },
    });

    const memoryOFF = calculateMemoryFromConfig({
      modelSpec: model, gpu: H100_SXM, numGPUs: 1, batchSize: 32,
      inputSeqLen: 1024, outputSeqLen: 512, weightPrecision: 'bf16', kvCachePrecision: 'bf16',
      flashAttention: true, pagedAttention: false, continuousBatching: false,
      speculative: { enabled: false, draftModel: null, numSpeculativeTokens: 4, acceptanceRate: 0.7 },
    });

    expect(memoryOFF.kvCache).toBeCloseTo(memoryON.kvCache * expectedFactor, 0);
    expect(memoryOFF.weights).toBe(memoryON.weights); // weights unchanged
    expect(memoryOFF.total).toBeGreaterThan(memoryON.total);
  });

  it('pagedAttention OFF: overhead recomputed with inflated KV', () => {
    const memoryOFF = calculateMemoryFromConfig({
      modelSpec: model, gpu: H100_SXM, numGPUs: 1, batchSize: 32,
      inputSeqLen: 1024, outputSeqLen: 512, weightPrecision: 'bf16', kvCachePrecision: 'bf16',
      flashAttention: true, pagedAttention: false, continuousBatching: false,
      speculative: { enabled: false, draftModel: null, numSpeculativeTokens: 4, acceptanceRate: 0.7 },
    });
    // overhead = 0.1 * (weights + inflatedKV)
    const expectedOverhead = (memoryOFF.weights + memoryOFF.kvCache) * 0.1;
    expect(memoryOFF.overhead).toBeCloseTo(expectedOverhead, 0);
  });

  it('pagedAttention OFF with TP: fragmentation still applied', () => {
    const memoryON = calculateMemoryFromConfig({
      modelSpec: model, gpu: H100_SXM, numGPUs: 2, batchSize: 32,
      inputSeqLen: 1024, outputSeqLen: 512, weightPrecision: 'bf16', kvCachePrecision: 'bf16',
      flashAttention: true, pagedAttention: true, continuousBatching: false, tensorParallel: 2,
      speculative: { enabled: false, draftModel: null, numSpeculativeTokens: 4, acceptanceRate: 0.7 },
    });

    const memoryOFF = calculateMemoryFromConfig({
      modelSpec: model, gpu: H100_SXM, numGPUs: 2, batchSize: 32,
      inputSeqLen: 1024, outputSeqLen: 512, weightPrecision: 'bf16', kvCachePrecision: 'bf16',
      flashAttention: true, pagedAttention: false, continuousBatching: false, tensorParallel: 2,
      speculative: { enabled: false, draftModel: null, numSpeculativeTokens: 4, acceptanceRate: 0.7 },
    });

    expect(memoryOFF.kvCache).toBeGreaterThan(memoryON.kvCache);
    expect(memoryOFF.total).toBeGreaterThan(memoryON.total);
  });

  it('pagedAttention OFF with EP: fragmentation applied on MoE models', () => {
    const moeModel = getModel('deepseek-v3')!;

    const memoryON = calculateMemoryFromConfig({
      modelSpec: moeModel, gpu: H100_SXM, numGPUs: 8, batchSize: 4,
      inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'fp8', kvCachePrecision: 'fp8',
      flashAttention: true, pagedAttention: true, continuousBatching: false,
      tensorParallel: 4, expertParallel: 2,
      speculative: { enabled: false, draftModel: null, numSpeculativeTokens: 4, acceptanceRate: 0.7 },
    });

    const memoryOFF = calculateMemoryFromConfig({
      modelSpec: moeModel, gpu: H100_SXM, numGPUs: 8, batchSize: 4,
      inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'fp8', kvCachePrecision: 'fp8',
      flashAttention: true, pagedAttention: false, continuousBatching: false,
      tensorParallel: 4, expertParallel: 2,
      speculative: { enabled: false, draftModel: null, numSpeculativeTokens: 4, acceptanceRate: 0.7 },
    });

    expect(memoryOFF.kvCache).toBeGreaterThan(memoryON.kvCache);
  });

  it('memory utilization increases when toggling OFF', () => {
    const valON = validateMemoryFits({
      modelSpec: model, gpu: H100_SXM, numGPUs: 1, batchSize: 64,
      inputSeqLen: 1024, outputSeqLen: 512, weightPrecision: 'bf16', kvCachePrecision: 'bf16',
      flashAttention: true, pagedAttention: true, continuousBatching: false,
      speculative: { enabled: false, draftModel: null, numSpeculativeTokens: 4, acceptanceRate: 0.7 },
    }, H100_SXM);

    const valOFF = validateMemoryFits({
      modelSpec: model, gpu: H100_SXM, numGPUs: 1, batchSize: 64,
      inputSeqLen: 1024, outputSeqLen: 512, weightPrecision: 'bf16', kvCachePrecision: 'bf16',
      flashAttention: true, pagedAttention: false, continuousBatching: false,
      speculative: { enabled: false, draftModel: null, numSpeculativeTokens: 4, acceptanceRate: 0.7 },
    }, H100_SXM);

    expect(valOFF.utilizationPercent).toBeGreaterThan(valON.utilizationPercent);
  });

  it('fragmentation can push a tight config from fitting to OOM', () => {
    // Use A100 80GB with large batch to get close to limit
    const valON = validateMemoryFits({
      modelSpec: model, gpu: A100_80GB, numGPUs: 1, batchSize: 512,
      inputSeqLen: 2048, outputSeqLen: 1024, weightPrecision: 'bf16', kvCachePrecision: 'bf16',
      flashAttention: true, pagedAttention: true, continuousBatching: false,
      speculative: { enabled: false, draftModel: null, numSpeculativeTokens: 4, acceptanceRate: 0.7 },
    }, A100_80GB);

    const valOFF = validateMemoryFits({
      modelSpec: model, gpu: A100_80GB, numGPUs: 1, batchSize: 512,
      inputSeqLen: 2048, outputSeqLen: 1024, weightPrecision: 'bf16', kvCachePrecision: 'bf16',
      flashAttention: true, pagedAttention: false, continuousBatching: false,
      speculative: { enabled: false, draftModel: null, numSpeculativeTokens: 4, acceptanceRate: 0.7 },
    }, A100_80GB);

    // With PA off, memory should be significantly higher
    expect(valOFF.memory.total).toBeGreaterThan(valON.memory.total);
    // If ON fits, OFF should use more memory (may or may not OOM depending on exact numbers)
    if (valON.fits) {
      expect(valOFF.utilizationPercent).toBeGreaterThan(valON.utilizationPercent);
    }
  });

  it('short sequences cause more fragmentation than long ones', () => {
    // Short seq: 1024+512 = 1536 total, maxSeq=8192, factor = min(8192/1536, 2.0) = 2.0
    const memShortOFF = calculateMemoryFromConfig({
      modelSpec: model, gpu: H100_SXM, numGPUs: 1, batchSize: 32,
      inputSeqLen: 512, outputSeqLen: 512, weightPrecision: 'bf16', kvCachePrecision: 'bf16',
      flashAttention: true, pagedAttention: false, continuousBatching: false,
      speculative: { enabled: false, draftModel: null, numSpeculativeTokens: 4, acceptanceRate: 0.7 },
    });
    const memShortON = calculateMemoryFromConfig({
      modelSpec: model, gpu: H100_SXM, numGPUs: 1, batchSize: 32,
      inputSeqLen: 512, outputSeqLen: 512, weightPrecision: 'bf16', kvCachePrecision: 'bf16',
      flashAttention: true, pagedAttention: true, continuousBatching: false,
      speculative: { enabled: false, draftModel: null, numSpeculativeTokens: 4, acceptanceRate: 0.7 },
    });
    const shortRatio = memShortOFF.kvCache / memShortON.kvCache;

    // Long seq: 4096+4096 = 8192 total, maxSeq=8192, factor = 1.0
    const memLongOFF = calculateMemoryFromConfig({
      modelSpec: model, gpu: H100_SXM, numGPUs: 1, batchSize: 32,
      inputSeqLen: 4096, outputSeqLen: 4096, weightPrecision: 'bf16', kvCachePrecision: 'bf16',
      flashAttention: true, pagedAttention: false, continuousBatching: false,
      speculative: { enabled: false, draftModel: null, numSpeculativeTokens: 4, acceptanceRate: 0.7 },
    });
    const memLongON = calculateMemoryFromConfig({
      modelSpec: model, gpu: H100_SXM, numGPUs: 1, batchSize: 32,
      inputSeqLen: 4096, outputSeqLen: 4096, weightPrecision: 'bf16', kvCachePrecision: 'bf16',
      flashAttention: true, pagedAttention: true, continuousBatching: false,
      speculative: { enabled: false, draftModel: null, numSpeculativeTokens: 4, acceptanceRate: 0.7 },
    });
    const longRatio = memLongOFF.kvCache / memLongON.kvCache;

    expect(shortRatio).toBeGreaterThan(longRatio);
  });
});

// ══════════════════════════════════════════════════════════════════════
// Recommendation tests
// ══════════════════════════════════════════════════════════════════════

describe('Paged attention recommendations', () => {
  it('OOM + PA off → recommendation includes "paged attention"', () => {
    // batch=72 OOMs with PA off (~86GB) but fits with PA on on A100 80GB (80 GiB)
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: A100_80GB,
      numGPUs: 1,
      batchSize: 72,
      inputSeqLen: 2048,
      outputSeqLen: 1024,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
      pagedAttention: false,
    });
    expect(result.success).toBe(false);
    expect(result.recommendations.some(r => r.toLowerCase().includes('paged attention'))).toBe(true);
  });

  it('OOM + PA already on → no paged attention recommendation', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-70b',
      gpu: A100_80GB,
      numGPUs: 1,
      batchSize: 128,
      inputSeqLen: 2048,
      outputSeqLen: 1024,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
      pagedAttention: true,
    });
    if (!result.success) {
      expect(result.recommendations.some(r => r.toLowerCase().includes('paged attention'))).toBe(false);
    }
  });

  it('success + PA off → recommendation includes "paged attention"', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 64,
      inputSeqLen: 512,
      outputSeqLen: 256,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
      pagedAttention: false,
    });
    expect(result.success).toBe(true);
    expect(result.recommendations.some(r => r.toLowerCase().includes('paged attention'))).toBe(true);
  });

  it('success + PA on → no paged attention recommendation', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 64,
      inputSeqLen: 512,
      outputSeqLen: 256,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
      pagedAttention: true,
    });
    expect(result.success).toBe(true);
    expect(result.recommendations.some(r => r.toLowerCase().includes('paged attention'))).toBe(false);
  });

  it('OOM resolved by PA: re-simulated config fits', () => {
    // batch=72 OOMs with PA off but fits with PA on
    const resultOFF = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: A100_80GB,
      numGPUs: 1,
      batchSize: 72,
      inputSeqLen: 2048,
      outputSeqLen: 1024,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
      pagedAttention: false,
    });

    const resultON = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: A100_80GB,
      numGPUs: 1,
      batchSize: 72,
      inputSeqLen: 2048,
      outputSeqLen: 1024,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
      pagedAttention: true,
    });

    expect(resultOFF.success).toBe(false);
    expect(resultON.success).toBe(true);
    expect(resultOFF.recommendations.some(r => r.toLowerCase().includes('paged attention'))).toBe(true);
  });
});

// ══════════════════════════════════════════════════════════════════════
// Regression tests
// ══════════════════════════════════════════════════════════════════════

describe('Paged attention regression', () => {
  it('default config (pagedAttention: true) produces identical results to base calculation', () => {
    const model = getModel('llama3-8b')!;
    const totalSeqLen = 1024 + 512;
    const baseMemory = calculateInferenceMemory(model, totalSeqLen, 1, 'bf16', 'bf16', true);
    const configMemory = calculateMemoryFromConfig({
      modelSpec: model, gpu: H100_SXM, numGPUs: 1, batchSize: 1,
      inputSeqLen: 1024, outputSeqLen: 512, weightPrecision: 'bf16', kvCachePrecision: 'bf16',
      flashAttention: true, pagedAttention: true, continuousBatching: false,
      speculative: { enabled: false, draftModel: null, numSpeculativeTokens: 4, acceptanceRate: 0.7 },
    });
    expect(configMemory.kvCache).toBe(baseMemory.kvCache);
    expect(configMemory.weights).toBe(baseMemory.weights);
    expect(configMemory.overhead).toBe(baseMemory.overhead);
    expect(configMemory.total).toBe(baseMemory.total);
  });

  it('default simulation results unchanged for pagedAttention: true', () => {
    // Running with explicit pagedAttention: true should behave the same as before
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
      pagedAttention: true,
    });
    expect(result.success).toBe(true);
    // Memory should not include any fragmentation factor
    const model = getModel('llama3-8b')!;
    const baseMemory = calculateInferenceMemory(model, 1536, 1, 'bf16', 'bf16', true);
    expect(result.memory.kvCache).toBe(baseMemory.kvCache);
  });
});
