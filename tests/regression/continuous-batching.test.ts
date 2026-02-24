/**
 * Tests for continuous batching inference simulation
 */

import { describe, it, expect } from 'vitest';
import {
  calculateContinuousBatchingMetrics,
  calculateMetricsFromConfig,
  estimateTTFT,
  runInferenceSimulation,
} from '../../src/core/inference/index.ts';
import { getModel } from '../../src/core/models/index.ts';
import { H100_SXM } from '../../src/core/hardware/gpu.ts';
import type { InferenceConfig } from '../../src/types/inference.ts';

// Helper to build a full InferenceConfig for direct calculateContinuousBatchingMetrics calls
function makeConfig(overrides: Partial<InferenceConfig> = {}): InferenceConfig {
  const model = overrides.modelSpec ?? getModel('llama3-8b', 2048)!;
  return {
    modelSpec: model,
    gpu: overrides.gpu ?? H100_SXM,
    numGPUs: overrides.numGPUs ?? 1,
    batchSize: overrides.batchSize ?? 32,
    inputSeqLen: overrides.inputSeqLen ?? 1024,
    outputSeqLen: overrides.outputSeqLen ?? 512,
    weightPrecision: overrides.weightPrecision ?? 'bf16',
    kvCachePrecision: overrides.kvCachePrecision ?? 'bf16',
    flashAttention: overrides.flashAttention ?? true,
    pagedAttention: overrides.pagedAttention ?? true,
    continuousBatching: overrides.continuousBatching ?? true,
    speculative: overrides.speculative ?? { enabled: false, draftModel: null, numSpeculativeTokens: 4, acceptanceRate: 0.7 },
    tensorParallel: overrides.tensorParallel,
    expertParallel: overrides.expertParallel,
  };
}

describe('Continuous Batching — Unit Tests', () => {
  it('slot-cycling formula: throughput = B * outSeqLen * 1000 / (cbTTFT + effectiveTpot * outSeqLen)', () => {
    const config = makeConfig({ batchSize: 32 });
    const baseMetrics = calculateMetricsFromConfig(config);
    const cbMetrics = calculateContinuousBatchingMetrics(config, baseMetrics);

    // Verify the formula holds
    const B = 32;
    const outSeqLen = config.outputSeqLen;
    const cycleTime = cbMetrics.latency.ttft + cbMetrics.latency.tpot * outSeqLen;
    const expectedThroughput = B * outSeqLen * 1000 / cycleTime;

    expect(cbMetrics.throughput.tokensPerSecond).toBeCloseTo(expectedThroughput, 1);
  });

  it('scheduling overhead is 1% base + scales to 2% at B=128+', () => {
    const config1 = makeConfig({ batchSize: 1 });
    const base1 = calculateMetricsFromConfig(config1);
    const cb1 = calculateContinuousBatchingMetrics(config1, base1);
    // At B=1: overhead = 0.01 + 0.01 * min(1, 1/128) ≈ 1.008%
    const expectedTpot1 = base1.latency.tpot * (1 + 0.01 + 0.01 * Math.min(1, 1 / 128));
    expect(cb1.latency.tpot).toBeCloseTo(expectedTpot1, 6);

    const config128 = makeConfig({ batchSize: 128 });
    const base128 = calculateMetricsFromConfig(config128);
    const cb128 = calculateContinuousBatchingMetrics(config128, base128);
    // At B=128: overhead = 0.01 + 0.01 * 1 = 2%
    const expectedTpot128 = base128.latency.tpot * 1.02;
    expect(cb128.latency.tpot).toBeCloseTo(expectedTpot128, 6);
  });

  it('prefill interference is 0 at B=1 and 10% at B≥32', () => {
    const config1 = makeConfig({ batchSize: 1 });
    const base1 = calculateMetricsFromConfig(config1);
    const cb1 = calculateContinuousBatchingMetrics(config1, base1);
    // At B=1: interference = 0.10 * min(1, 0/32) = 0
    const batch1TTFT = estimateTTFT(config1.modelSpec, config1.inputSeqLen, config1.gpu, config1.weightPrecision, 1);
    expect(cb1.latency.ttft).toBeCloseTo(batch1TTFT, 1);

    const config32 = makeConfig({ batchSize: 32 });
    const base32 = calculateMetricsFromConfig(config32);
    const cb32 = calculateContinuousBatchingMetrics(config32, base32);
    // At B=32: interference = 0.10 * min(1, 31/32) ≈ 9.7%
    const batch1TTFT32 = estimateTTFT(config32.modelSpec, config32.inputSeqLen, config32.gpu, config32.weightPrecision, 1);
    const expectedCBTTFT = batch1TTFT32 * (1 + 0.10 * Math.min(1, 31 / 32));
    expect(cb32.latency.ttft).toBeCloseTo(expectedCBTTFT, 1);
  });

  it('utilization metrics unchanged between static and CB', () => {
    const config = makeConfig({ batchSize: 32 });
    const baseMetrics = calculateMetricsFromConfig(config);
    const cbMetrics = calculateContinuousBatchingMetrics(config, baseMetrics);

    expect(cbMetrics.utilization).toEqual(baseMetrics.utilization);
  });
});

describe('Continuous Batching — Integration (Throughput)', () => {
  it('CB throughput > static throughput at batch=32 (Llama 8B, H100)', () => {
    const staticResult = runInferenceSimulation({
      modelId: 'llama3-8b',
      batchSize: 32,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      continuousBatching: false,
    });

    const cbResult = runInferenceSimulation({
      modelId: 'llama3-8b',
      batchSize: 32,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      continuousBatching: true,
    });

    expect(cbResult.success).toBe(true);
    expect(staticResult.success).toBe(true);
    expect(cbResult.throughput.tokensPerSecond).toBeGreaterThan(
      staticResult.throughput.tokensPerSecond
    );
  });

  it('CB benefit larger for prefill-heavy workloads', () => {
    // Prefill-heavy: input=4096, output=128
    const prefillHeavyStatic = runInferenceSimulation({
      modelId: 'llama3-8b',
      batchSize: 32,
      inputSeqLen: 4096,
      outputSeqLen: 128,
      continuousBatching: false,
    });
    const prefillHeavyCB = runInferenceSimulation({
      modelId: 'llama3-8b',
      batchSize: 32,
      inputSeqLen: 4096,
      outputSeqLen: 128,
      continuousBatching: true,
    });
    const prefillHeavyRatio = prefillHeavyCB.throughput.tokensPerSecond /
      prefillHeavyStatic.throughput.tokensPerSecond;

    // Decode-heavy: input=128, output=2048
    const decodeHeavyStatic = runInferenceSimulation({
      modelId: 'llama3-8b',
      batchSize: 32,
      inputSeqLen: 128,
      outputSeqLen: 2048,
      continuousBatching: false,
    });
    const decodeHeavyCB = runInferenceSimulation({
      modelId: 'llama3-8b',
      batchSize: 32,
      inputSeqLen: 128,
      outputSeqLen: 2048,
      continuousBatching: true,
    });
    const decodeHeavyRatio = decodeHeavyCB.throughput.tokensPerSecond /
      decodeHeavyStatic.throughput.tokensPerSecond;

    // Prefill-heavy should benefit more from CB
    expect(prefillHeavyRatio).toBeGreaterThan(decodeHeavyRatio);
  });

  it('CB at batch=1 has modest effect', () => {
    const staticResult = runInferenceSimulation({
      modelId: 'llama3-8b',
      batchSize: 1,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      continuousBatching: false,
    });
    const cbResult = runInferenceSimulation({
      modelId: 'llama3-8b',
      batchSize: 1,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      continuousBatching: true,
    });

    const ratio = cbResult.throughput.tokensPerSecond / staticResult.throughput.tokensPerSecond;
    // At batch=1, CB has minimal effect: ratio should be between 0.9 and 2.0
    expect(ratio).toBeGreaterThan(0.9);
    expect(ratio).toBeLessThan(2.0);
  });
});

describe('Continuous Batching — Integration (Latency)', () => {
  it('CB TTFT < static TTFT at batch=64 (Llama 70B, TP=4)', () => {
    const staticResult = runInferenceSimulation({
      modelId: 'llama2-70b',
      numGPUs: 4,
      tensorParallel: 4,
      batchSize: 64,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      continuousBatching: false,
    });
    const cbResult = runInferenceSimulation({
      modelId: 'llama2-70b',
      numGPUs: 4,
      tensorParallel: 4,
      batchSize: 64,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      continuousBatching: true,
    });

    expect(cbResult.success).toBe(true);
    // CB uses batch=1 TTFT (+ interference), which should be much less than static batch=64 TTFT
    expect(cbResult.latency.ttft).toBeLessThan(staticResult.latency.ttft);
  });

  it('CB TPOT slightly > static TPOT due to scheduling overhead, but < 1.05x', () => {
    const staticResult = runInferenceSimulation({
      modelId: 'llama3-8b',
      batchSize: 32,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      continuousBatching: false,
    });
    const cbResult = runInferenceSimulation({
      modelId: 'llama3-8b',
      batchSize: 32,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      continuousBatching: true,
    });

    // CB TPOT has ~1.5% overhead at batch=32
    expect(cbResult.latency.tpot).toBeGreaterThan(staticResult.latency.tpot);
    expect(cbResult.latency.tpot).toBeLessThan(staticResult.latency.tpot * 1.05);
  });
});

describe('Continuous Batching — Integration (Memory)', () => {
  it('memory identical with and without CB', () => {
    const staticResult = runInferenceSimulation({
      modelId: 'llama3-8b',
      batchSize: 32,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      continuousBatching: false,
    });
    const cbResult = runInferenceSimulation({
      modelId: 'llama3-8b',
      batchSize: 32,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      continuousBatching: true,
    });

    expect(cbResult.memory.total).toBe(staticResult.memory.total);
    expect(cbResult.memory.weights).toBe(staticResult.memory.weights);
    expect(cbResult.memory.kvCache).toBe(staticResult.memory.kvCache);
  });
});

describe('Continuous Batching — Benchmark', () => {
  it('Llama 70B, TP=4, batch=64: CB/static throughput ratio 1.1-3x', () => {
    const staticResult = runInferenceSimulation({
      modelId: 'llama2-70b',
      numGPUs: 4,
      tensorParallel: 4,
      batchSize: 64,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      continuousBatching: false,
    });
    const cbResult = runInferenceSimulation({
      modelId: 'llama2-70b',
      numGPUs: 4,
      tensorParallel: 4,
      batchSize: 64,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      continuousBatching: true,
    });

    const ratio = cbResult.throughput.tokensPerSecond / staticResult.throughput.tokensPerSecond;
    expect(ratio).toBeGreaterThan(1.1);
    expect(ratio).toBeLessThan(3.0);
  });
});

describe('Continuous Batching — Interaction Tests', () => {
  it('CB + speculative decoding both enabled → success, positive throughput', () => {
    const result = runInferenceSimulation({
      modelId: 'llama2-70b',
      numGPUs: 4,
      tensorParallel: 4,
      batchSize: 8,
      inputSeqLen: 512,
      outputSeqLen: 256,
      continuousBatching: true,
      speculativeEnabled: true,
      draftModelId: 'llama3-8b',
      numSpeculativeTokens: 4,
      acceptanceRate: 0.7,
    });

    expect(result.success).toBe(true);
    expect(result.throughput.tokensPerSecond).toBeGreaterThan(0);
  });

  it('CB + speculative: headline TPOT uses speculative only when specTpot < cbTpot', () => {
    // With a good draft model, speculative should beat CB TPOT
    const cbOnlyResult = runInferenceSimulation({
      modelId: 'llama2-70b',
      numGPUs: 4,
      tensorParallel: 4,
      batchSize: 4,
      inputSeqLen: 512,
      outputSeqLen: 256,
      continuousBatching: true,
      speculativeEnabled: false,
    });

    const cbSpecResult = runInferenceSimulation({
      modelId: 'llama2-70b',
      numGPUs: 4,
      tensorParallel: 4,
      batchSize: 4,
      inputSeqLen: 512,
      outputSeqLen: 256,
      continuousBatching: true,
      speculativeEnabled: true,
      draftModelId: 'llama3-8b',
      numSpeculativeTokens: 4,
      acceptanceRate: 0.7,
    });

    // If speculative is applied, headline TPOT = effectiveTpot × (1 + CB scheduling overhead).
    // CB overhead still applies because the scheduler runs every draft-verify cycle.
    if (cbSpecResult.speculative && cbSpecResult.speculative.effectiveTpot < cbOnlyResult.latency.tpot) {
      expect(cbSpecResult.latency.tpot).toBeGreaterThan(cbSpecResult.speculative.effectiveTpot);
      // Overhead is at most 2%, so headline TPOT is within 2% of effectiveTpot
      expect(cbSpecResult.latency.tpot).toBeLessThan(cbSpecResult.speculative.effectiveTpot * 1.025);
    } else {
      // If speculative isn't beneficial, CB TPOT is used
      expect(cbSpecResult.latency.tpot).toBeCloseTo(cbOnlyResult.latency.tpot, 1);
    }
  });

  it('CB + TP → success, TP-aware TTFT', () => {
    const result = runInferenceSimulation({
      modelId: 'llama2-70b',
      numGPUs: 4,
      tensorParallel: 4,
      batchSize: 32,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      continuousBatching: true,
    });

    expect(result.success).toBe(true);
    expect(result.continuousBatching).toBe(true);
    // TP-aware batch=1 TTFT should be much less than static batch=32 TTFT
    const staticResult = runInferenceSimulation({
      modelId: 'llama2-70b',
      numGPUs: 4,
      tensorParallel: 4,
      batchSize: 32,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      continuousBatching: false,
    });
    expect(result.latency.ttft).toBeLessThan(staticResult.latency.ttft);
  });

  it('CB + speculative + multi-replica: throughput scales by numReplicas', () => {
    // 2 replicas (8 GPUs, TP=4)
    const result = runInferenceSimulation({
      modelId: 'llama2-70b',
      numGPUs: 8,
      tensorParallel: 4,
      batchSize: 16,
      inputSeqLen: 512,
      outputSeqLen: 256,
      continuousBatching: true,
      speculativeEnabled: true,
      draftModelId: 'llama3-8b',
      numSpeculativeTokens: 4,
      acceptanceRate: 0.7,
    });

    // 1 replica (4 GPUs, TP=4)
    const singleResult = runInferenceSimulation({
      modelId: 'llama2-70b',
      numGPUs: 4,
      tensorParallel: 4,
      batchSize: 8, // batchPerReplica = 16/2 = 8
      inputSeqLen: 512,
      outputSeqLen: 256,
      continuousBatching: true,
      speculativeEnabled: true,
      draftModelId: 'llama3-8b',
      numSpeculativeTokens: 4,
      acceptanceRate: 0.7,
    });

    expect(result.success).toBe(true);
    expect(singleResult.success).toBe(true);
    // 2 replicas should have ~2x throughput of 1 replica
    const ratio = result.throughput.tokensPerSecond / singleResult.throughput.tokensPerSecond;
    expect(ratio).toBeGreaterThan(1.8);
    expect(ratio).toBeLessThan(2.2);
  });
});

describe('Continuous Batching — Recommendations', () => {
  it('Enable CB fires at batch > 4', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      batchSize: 32,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      continuousBatching: false,
    });

    expect(result.success).toBe(true);
    const hasCBRec = result.recommendations.some(r =>
      r.toLowerCase().includes('continuous batching')
    );
    expect(hasCBRec).toBe(true);
  });

  it('Disable CB does NOT fire at batch > 4', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      batchSize: 32,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      continuousBatching: true,
    });

    expect(result.success).toBe(true);
    const hasDisableRec = result.recommendations.some(r =>
      r.toLowerCase().includes('disabling continuous batching')
    );
    expect(hasDisableRec).toBe(false);
  });

  it('Enable CB does NOT fire when CB already on', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      batchSize: 32,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      continuousBatching: true,
    });

    expect(result.success).toBe(true);
    const hasEnableRec = result.recommendations.some(r =>
      r.toLowerCase().includes('enabling continuous batching')
    );
    expect(hasEnableRec).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// MoE models with CB
// ---------------------------------------------------------------------------

describe('CB — MoE models', () => {
  it('DeepSeek V3 CB > static throughput (INT4)', () => {
    // DSV3 671B needs INT4 quantization to fit on 8 H100s with TP=8
    const staticResult = runInferenceSimulation({
      modelId: 'deepseek-v3',
      numGPUs: 8,
      tensorParallel: 8,
      batchSize: 32,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      weightPrecision: 'int4',
      continuousBatching: false,
    });
    const cbResult = runInferenceSimulation({
      modelId: 'deepseek-v3',
      numGPUs: 8,
      tensorParallel: 8,
      batchSize: 32,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      weightPrecision: 'int4',
      continuousBatching: true,
    });

    expect(cbResult.success).toBe(true);
    expect(cbResult.throughput.tokensPerSecond).toBeGreaterThan(staticResult.throughput.tokensPerSecond);
    expect(cbResult.latency.ttft).toBeLessThan(staticResult.latency.ttft);
    expect(cbResult.memory.total).toBe(staticResult.memory.total);
  });

  it('Grok-1 MoE CB at large batch (INT8)', () => {
    // Grok-1 314B needs INT8 to fit on 8 H100s
    const staticResult = runInferenceSimulation({
      modelId: 'grok-1',
      numGPUs: 8,
      tensorParallel: 8,
      batchSize: 64,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      weightPrecision: 'int8',
      continuousBatching: false,
    });
    const cbResult = runInferenceSimulation({
      modelId: 'grok-1',
      numGPUs: 8,
      tensorParallel: 8,
      batchSize: 64,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      weightPrecision: 'int8',
      continuousBatching: true,
    });

    expect(cbResult.success).toBe(true);
    const ratio = cbResult.throughput.tokensPerSecond / staticResult.throughput.tokensPerSecond;
    expect(ratio).toBeGreaterThan(1.0);
    // TPOT overhead < 5%
    expect(cbResult.latency.tpot).toBeLessThan(staticResult.latency.tpot * 1.05);
  });

  it('LLaMA 4 Maverick MoE CB succeeds (INT8)', () => {
    const result = runInferenceSimulation({
      modelId: 'llama4-maverick',
      numGPUs: 8,
      tensorParallel: 8,
      batchSize: 32,
      inputSeqLen: 2048,
      outputSeqLen: 512,
      weightPrecision: 'int8',
      continuousBatching: true,
    });

    expect(result.success).toBe(true);
    expect(result.throughput.tokensPerSecond).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// Quantized inference with CB
// ---------------------------------------------------------------------------

describe('CB — Quantized inference', () => {
  it('INT4 + CB (LLaMA 70B single GPU)', () => {
    const staticResult = runInferenceSimulation({
      modelId: 'llama2-70b',
      numGPUs: 1,
      batchSize: 16,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      weightPrecision: 'int4',
      kvCachePrecision: 'int4',
      continuousBatching: false,
    });
    const cbResult = runInferenceSimulation({
      modelId: 'llama2-70b',
      numGPUs: 1,
      batchSize: 16,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      weightPrecision: 'int4',
      kvCachePrecision: 'int4',
      continuousBatching: true,
    });

    expect(cbResult.success).toBe(true);
    expect(cbResult.throughput.tokensPerSecond).toBeGreaterThan(staticResult.throughput.tokensPerSecond);
  });

  it('INT8 + CB (LLaMA 70B TP=2)', () => {
    const staticResult = runInferenceSimulation({
      modelId: 'llama2-70b',
      numGPUs: 2,
      tensorParallel: 2,
      batchSize: 32,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      weightPrecision: 'int8',
      kvCachePrecision: 'int8',
      continuousBatching: false,
    });
    const cbResult = runInferenceSimulation({
      modelId: 'llama2-70b',
      numGPUs: 2,
      tensorParallel: 2,
      batchSize: 32,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      weightPrecision: 'int8',
      kvCachePrecision: 'int8',
      continuousBatching: true,
    });

    expect(cbResult.success).toBe(true);
    expect(cbResult.throughput.tokensPerSecond).toBeGreaterThan(staticResult.throughput.tokensPerSecond);
    expect(cbResult.memory.total).toBe(staticResult.memory.total);
  });

  it('FP8 + CB no NaN', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      numGPUs: 1,
      batchSize: 32,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      weightPrecision: 'fp8',
      kvCachePrecision: 'fp8',
      continuousBatching: true,
    });

    expect(result.success).toBe(true);
    expect(Number.isFinite(result.throughput.tokensPerSecond)).toBe(true);
    expect(Number.isFinite(result.latency.ttft)).toBe(true);
    expect(Number.isFinite(result.latency.tpot)).toBe(true);
  });

  it('FP4 + CB no NaN (regression)', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpuId: 'b200',
      numGPUs: 1,
      batchSize: 16,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      weightPrecision: 'fp4',
      kvCachePrecision: 'fp4',
      continuousBatching: true,
    });

    expect(result.success).toBe(true);
    expect(Number.isFinite(result.throughput.tokensPerSecond)).toBe(true);
    expect(Number.isFinite(result.latency.ttft)).toBe(true);
    expect(Number.isFinite(result.latency.tpot)).toBe(true);
    expect(result.throughput.tokensPerSecond).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// Different GPUs with CB
// ---------------------------------------------------------------------------

describe('CB — Different GPUs', () => {
  it('A10G (24GB budget) + CB', () => {
    const staticResult = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpuId: 'a10g',
      numGPUs: 1,
      batchSize: 4,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      weightPrecision: 'int8',
      continuousBatching: false,
    });
    const cbResult = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpuId: 'a10g',
      numGPUs: 1,
      batchSize: 4,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      weightPrecision: 'int8',
      continuousBatching: true,
    });

    expect(cbResult.success).toBe(true);
    expect(cbResult.throughput.tokensPerSecond).toBeGreaterThanOrEqual(
      staticResult.throughput.tokensPerSecond * 0.95
    );
  });

  it('L4 + CB', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpuId: 'l4',
      numGPUs: 1,
      batchSize: 8,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      weightPrecision: 'fp8',
      continuousBatching: true,
    });

    expect(result.success).toBe(true);
    expect(result.throughput.tokensPerSecond).toBeGreaterThan(0);
  });

  it('H200 large batch + CB', () => {
    const staticResult = runInferenceSimulation({
      modelId: 'llama2-70b',
      gpuId: 'h200-sxm',
      numGPUs: 4,
      tensorParallel: 4,
      batchSize: 128,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      continuousBatching: false,
    });
    const cbResult = runInferenceSimulation({
      modelId: 'llama2-70b',
      gpuId: 'h200-sxm',
      numGPUs: 4,
      tensorParallel: 4,
      batchSize: 128,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      continuousBatching: true,
    });

    expect(cbResult.success).toBe(true);
    expect(cbResult.throughput.tokensPerSecond).toBeGreaterThan(staticResult.throughput.tokensPerSecond);
    expect(cbResult.latency.ttft).toBeLessThan(staticResult.latency.ttft);
  });
});

// ---------------------------------------------------------------------------
// MLA model with CB
// ---------------------------------------------------------------------------

describe('CB — MLA model', () => {
  it('DeepSeek V3 MLA TPOT correct with CB (INT4)', () => {
    const result = runInferenceSimulation({
      modelId: 'deepseek-v3',
      numGPUs: 8,
      tensorParallel: 8,
      batchSize: 32,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      weightPrecision: 'int4',
      continuousBatching: true,
    });

    expect(result.success).toBe(true);
    expect(Number.isFinite(result.latency.tpot)).toBe(true);
    expect(result.latency.tpot).toBeGreaterThan(0);

    // CB TTFT < static TTFT
    const staticResult = runInferenceSimulation({
      modelId: 'deepseek-v3',
      numGPUs: 8,
      tensorParallel: 8,
      batchSize: 32,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      weightPrecision: 'int4',
      continuousBatching: false,
    });
    expect(result.latency.ttft).toBeLessThan(staticResult.latency.ttft);
  });
});

// ---------------------------------------------------------------------------
// Large batch sizes
// ---------------------------------------------------------------------------

describe('CB — Large batch sizes', () => {
  it('B=128 scheduling overhead saturates at 2%', () => {
    const config = makeConfig({ batchSize: 128 });
    const base = calculateMetricsFromConfig(config);
    const cb = calculateContinuousBatchingMetrics(config, base);

    // At B=128: overhead = 0.01 + 0.01 * min(1, 128/128) = 0.02
    const tpotRatio = cb.latency.tpot / base.latency.tpot;
    expect(tpotRatio).toBeLessThanOrEqual(1.025);
    expect(tpotRatio).toBeGreaterThan(1.0);
  });

  it('B=256 CB still works without NaN', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      numGPUs: 4,
      tensorParallel: 4,
      batchSize: 256,
      inputSeqLen: 512,
      outputSeqLen: 256,
      continuousBatching: true,
    });

    expect(result.success).toBe(true);
    expect(Number.isFinite(result.throughput.tokensPerSecond)).toBe(true);
    expect(Number.isFinite(result.latency.tpot)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Long sequences with CB
// ---------------------------------------------------------------------------

describe('CB — Long sequences', () => {
  it('inputSeqLen=8192 TTFT drops dramatically with CB', () => {
    const staticResult = runInferenceSimulation({
      modelId: 'llama3-8b',
      numGPUs: 1,
      batchSize: 16,
      inputSeqLen: 8192,
      outputSeqLen: 256,
      continuousBatching: false,
    });
    const cbResult = runInferenceSimulation({
      modelId: 'llama3-8b',
      numGPUs: 1,
      batchSize: 16,
      inputSeqLen: 8192,
      outputSeqLen: 256,
      continuousBatching: true,
    });

    expect(cbResult.success).toBe(true);
    // CB TTFT = batch=1 TTFT with interference, static TTFT = batch=16 TTFT
    // CB TTFT << static TTFT (roughly 1/16 × (1 + interference))
    expect(cbResult.latency.ttft).toBeLessThan(staticResult.latency.ttft * 0.30);
  });
});

// ---------------------------------------------------------------------------
// CB + EP interaction
// ---------------------------------------------------------------------------

describe('CB — EP interaction', () => {
  it('MoE with EP + CB (Mixtral 8x22B, TP=4 EP=2)', () => {
    // TP×EP must ≤ numGPUs, so use TP=4 EP=2 on 8 GPUs
    const result = runInferenceSimulation({
      modelId: 'mixtral-8x22b',
      numGPUs: 8,
      tensorParallel: 4,
      expertParallel: 2,
      batchSize: 32,
      inputSeqLen: 1024,
      outputSeqLen: 512,
      continuousBatching: true,
    });

    expect(result.success).toBe(true);
    expect(result.throughput.tokensPerSecond).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// Throughput monotonicity
// ---------------------------------------------------------------------------

describe('CB — Throughput monotonicity', () => {
  it('CB throughput increases with batch size', () => {
    const batchSizes = [1, 4, 8, 16, 32, 64];
    const throughputs = batchSizes.map(bs => {
      const result = runInferenceSimulation({
        modelId: 'llama3-8b',
        numGPUs: 1,
        batchSize: bs,
        inputSeqLen: 1024,
        outputSeqLen: 512,
        continuousBatching: true,
      });
      expect(result.success).toBe(true);
      return result.throughput.tokensPerSecond;
    });

    // Monotonic within noise (allowing 5% dip)
    for (let i = 0; i < throughputs.length - 1; i++) {
      expect(throughputs[i + 1]).toBeGreaterThanOrEqual(throughputs[i] * 0.95);
    }
  });
});

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

describe('CB — Edge cases', () => {
  it('outputSeqLen=1 (single token generation)', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      numGPUs: 1,
      batchSize: 32,
      inputSeqLen: 1024,
      outputSeqLen: 1,
      continuousBatching: true,
    });

    expect(result.success).toBe(true);
    expect(result.throughput.tokensPerSecond).toBeGreaterThan(0);
    // TTFT dominates when outputSeqLen=1
    expect(result.latency.ttft).toBeGreaterThan(result.latency.tpot);
  });

  it('decode-heavy workload — CB benefit small', () => {
    const staticResult = runInferenceSimulation({
      modelId: 'llama3-8b',
      numGPUs: 1,
      batchSize: 16,
      inputSeqLen: 128,
      outputSeqLen: 2048,
      continuousBatching: false,
    });
    const cbResult = runInferenceSimulation({
      modelId: 'llama3-8b',
      numGPUs: 1,
      batchSize: 16,
      inputSeqLen: 128,
      outputSeqLen: 2048,
      continuousBatching: true,
    });

    expect(cbResult.success).toBe(true);
    // Decode-dominated → small benefit
    const ratio = cbResult.throughput.tokensPerSecond / staticResult.throughput.tokensPerSecond;
    expect(ratio).toBeLessThan(1.5);
  });
});
