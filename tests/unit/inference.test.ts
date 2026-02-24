/**
 * Tests for inference simulation module
 */

import { describe, it, expect } from 'vitest';
import {
  kvCachePerToken,
  totalKVCacheMemory,
  kvCacheReductionFactor,
  getAttentionTypeString,
  calculateInferenceMemory,
  calculateMemoryWithTP,
  estimateTTFT,
  estimateTPOT,
  calculateLatencyMetrics,
  calculateThroughputMetrics,
  calculateUtilizationMetrics,
  expectedAcceptedTokens,
  theoreticalSpeedup,
  InferenceSimulationEngine,
  runInferenceSimulation,
} from '../../src/core/inference/index.ts';
import { getModel } from '../../src/core/models/index.ts';
import { H100_SXM } from '../../src/core/hardware/gpu.ts';

describe('KV Cache Calculations', () => {
  const llama7b = getModel('llama2-7b', 2048)!;
  const llama70b = getModel('llama2-70b', 2048)!;

  it('should calculate KV cache per token correctly', () => {
    // Llama 2 7B: 32 layers, 32 heads (MHA), head_dim=128, bf16
    // Per token: 2 * 32 * 32 * 128 * 2 = 524,288 bytes ≈ 0.5MB
    const perToken = kvCachePerToken(llama7b, 'bf16');
    expect(perToken).toBe(524288); // Deterministic: 2*32*32*128*2
  });

  it('should calculate total KV cache memory', () => {
    const seqLen = 1024;
    const batchSize = 4;
    const perToken = kvCachePerToken(llama7b, 'bf16');
    const totalMemory = totalKVCacheMemory(llama7b, seqLen, batchSize, 'bf16');

    expect(totalMemory).toBe(perToken * seqLen * batchSize);
  });

  it('should calculate KV cache reduction from GQA', () => {
    // Llama 2 7B has MHA (numKvHeads == numAttentionHeads)
    expect(kvCacheReductionFactor(llama7b)).toBe(1);

    // Llama 2 70B has GQA (8 KV heads, 64 attention heads)
    expect(kvCacheReductionFactor(llama70b)).toBeCloseTo(0.125, 3);
  });

  it('should identify attention type correctly', () => {
    expect(getAttentionTypeString(llama7b)).toBe('MHA');
    expect(getAttentionTypeString(llama70b)).toContain('GQA');
  });

  it('MLA: DeepSeek V3 KV cache per token (bf16)', () => {
    const v3 = getModel('deepseek-v3', 2048)!;
    // 61 layers × (512 + 64) × 2 bytes = 70,272
    const perToken = kvCachePerToken(v3, 'bf16');
    expect(perToken).toBe(61 * (512 + 64) * 2);
  });

  it('MLA: KV cache is NOT halved by TP (replicated)', () => {
    const v3 = getModel('deepseek-v3', 2048)!;
    const perToken = kvCachePerToken(v3, 'bf16');
    // totalKVCacheMemory uses kvCachePerToken which doesn't divide by TP
    // At TP=1 and TP=2, per-token KV cache is the same
    expect(perToken).toBe(61 * 576 * 2);
  });

  it('MLA: reduction factor vs MHA ≈ 0.018 (57x reduction)', () => {
    const v3 = getModel('deepseek-v3', 2048)!;
    const factor = kvCacheReductionFactor(v3);
    // (512 + 64) / (2 × 128 × 128) = 576/32768 ≈ 0.0176
    expect(factor).toBeCloseTo(576 / 32768, 3);
  });

  it('MLA: getAttentionTypeString returns MLA', () => {
    const v3 = getModel('deepseek-v3', 2048)!;
    expect(getAttentionTypeString(v3)).toBe('MLA');
  });
});

describe('Memory Calculations', () => {
  const llama7b = getModel('llama2-7b', 2048)!;

  it('should calculate inference memory breakdown', () => {
    const memory = calculateInferenceMemory(llama7b, 1024, 1, 'bf16', 'bf16', true);

    expect(memory.weights).toBeGreaterThan(0);
    expect(memory.kvCache).toBeGreaterThan(0);
    expect(memory.activations).toBeGreaterThan(0);
    expect(memory.overhead).toBeGreaterThan(0);
    expect(memory.total).toBe(
      memory.weights + memory.kvCache + memory.activations + memory.overhead
    );
  });

  it('should calculate weights memory correctly', () => {
    // 7B params in bf16 = 7B * 2 bytes = 14GB
    const memory = calculateInferenceMemory(llama7b, 1024, 1, 'bf16', 'bf16');
    const expectedWeights = llama7b.totalParams * 2; // bf16 = 2 bytes
    expect(memory.weights).toBeCloseTo(expectedWeights, -6); // Within 1MB
  });

  it('should scale KV cache with batch size', () => {
    const memory1 = calculateInferenceMemory(llama7b, 1024, 1, 'bf16', 'bf16');
    const memory4 = calculateInferenceMemory(llama7b, 1024, 4, 'bf16', 'bf16');

    // KV cache should scale linearly with batch size
    expect(memory4.kvCache).toBeCloseTo(memory1.kvCache * 4, -4);
  });
});

describe('Latency Calculations', () => {
  const llama7b = getModel('llama2-7b', 2048)!;

  it('should estimate TTFT (Time to First Token)', () => {
    const ttft = estimateTTFT(llama7b, 512, H100_SXM, 'bf16');

    // TTFT should be positive and reasonable (tens of ms for 7B on H100)
    expect(ttft).toBeGreaterThan(0);
    expect(ttft).toBeLessThan(1000); // Less than 1 second
  });

  it('should estimate TPOT (Time Per Output Token)', () => {
    const tpot = estimateTPOT(llama7b, 512, 1, H100_SXM, 'bf16');

    // TPOT should be positive and reasonable (single-digit ms for 7B on H100)
    expect(tpot).toBeGreaterThan(0);
    expect(tpot).toBeLessThan(100); // Less than 100ms per token
  });

  it('should calculate full latency metrics', () => {
    const metrics = calculateLatencyMetrics(llama7b, 512, 256, 1, H100_SXM, 'bf16');

    expect(metrics.ttft).toBeGreaterThan(0);
    expect(metrics.tpot).toBeGreaterThan(0);
    expect(metrics.prefillTime).toBe(metrics.ttft);
    expect(metrics.totalLatency).toBeGreaterThan(metrics.prefillTime);
  });

  it('should calculate throughput metrics', () => {
    const metrics = calculateThroughputMetrics(llama7b, 512, 256, 1, H100_SXM, 'bf16');

    expect(metrics.tokensPerSecond).toBeGreaterThan(0);
    expect(metrics.requestsPerSecond).toBeGreaterThan(0);
    expect(metrics.prefillTokensPerSecond).toBeGreaterThan(0);
    expect(metrics.decodeTokensPerSecond).toBeGreaterThan(0);
  });

  it('should identify bottleneck correctly', () => {
    const utilization = calculateUtilizationMetrics(llama7b, 512, 256, 1, H100_SXM, 'bf16');

    // Should have valid utilization values
    expect(utilization.computeUtilization).toBeGreaterThanOrEqual(0);
    expect(utilization.computeUtilization).toBeLessThanOrEqual(1);
    expect(utilization.rooflineAttainment).toBeGreaterThanOrEqual(0);
    expect(utilization.rooflineAttainment).toBeLessThanOrEqual(1);

    // Decode phase is typically memory-bound
    expect(['compute', 'memory_bandwidth', 'memory_capacity']).toContain(utilization.bottleneck);
  });
});

describe('Speculative Decoding', () => {
  it('should calculate expected accepted tokens', () => {
    // With 100% acceptance rate, all K tokens should be accepted
    expect(expectedAcceptedTokens(4, 1.0)).toBe(4);

    // With 0% acceptance rate, no tokens accepted (plus 1 from target)
    expect(expectedAcceptedTokens(4, 0)).toBe(1);

    // With 70% acceptance rate
    const expected = expectedAcceptedTokens(4, 0.7);
    expect(expected).toBeGreaterThan(1);
    expect(expected).toBeLessThan(5);
  });

  it('should calculate theoretical speedup', () => {
    const speedup70 = theoreticalSpeedup(4, 0.7);
    const speedup90 = theoreticalSpeedup(4, 0.9);

    // Higher acceptance rate should give higher speedup
    expect(speedup90).toBeGreaterThan(speedup70);

    // Speedup should be bounded
    expect(speedup70).toBeGreaterThan(1);
    expect(speedup90).toBeLessThanOrEqual(5);
  });
});

describe('Inference Simulation Engine', () => {
  it('should run a basic inference simulation', () => {
    const result = runInferenceSimulation({
      modelId: 'llama2-7b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 256,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
      flashAttention: true,
    });

    expect(result.success).toBe(true);
    expect(result.errors).toHaveLength(0);

    // Check memory breakdown
    expect(result.memory.weights).toBeGreaterThan(0);
    expect(result.memory.kvCache).toBeGreaterThan(0);
    expect(result.memory.total).toBeGreaterThan(0);

    // Check latency metrics
    expect(result.latency.ttft).toBeGreaterThan(0);
    expect(result.latency.tpot).toBeGreaterThan(0);

    // Check throughput
    expect(result.throughput.tokensPerSecond).toBeGreaterThan(0);

    // Check events were generated
    expect(result.events.length).toBeGreaterThan(0);
  });

  it('should detect OOM errors', () => {
    // Try to run 70B model on a single 24GB GPU (should fail)
    const result = runInferenceSimulation({
      modelId: 'llama2-70b',
      gpu: {
        ...H100_SXM,
        memoryGB: 24, // Simulate smaller GPU
      },
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 256,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
    });

    expect(result.success).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);
  });

  it('should run with speculative decoding', () => {
    const llama1b = getModel('phi-2', 2048); // Small model as draft

    if (llama1b) {
      const result = runInferenceSimulation({
        modelId: 'llama2-7b',
        gpu: H100_SXM,
        numGPUs: 1,
        batchSize: 1,
        inputSeqLen: 512,
        outputSeqLen: 256,
        speculativeEnabled: true,
        draftModelSpec: llama1b,
        numSpeculativeTokens: 4,
        acceptanceRate: 0.7,
      });

      expect(result.success).toBe(true);
      expect(result.speculative).toBeDefined();
      expect(result.speculative?.speedup).toBeGreaterThan(1);

      // Speculative metrics should feed into headline latency
      expect(result.latency.tpot).toBeCloseTo(result.speculative!.effectiveTpot, 5);
    }
  });

  it('should provide summary information', () => {
    const engine = new InferenceSimulationEngine();
    engine.configure({
      modelId: 'llama2-7b',
      gpu: H100_SXM,
      batchSize: 4,
      inputSeqLen: 512,
      outputSeqLen: 256,
    });

    const summary = engine.getSummary();
    expect(summary).not.toBeNull();
    expect(summary?.model).toContain('7B'); // Model name varies
    expect(summary?.params).toMatch(/7\.?\d*B/); // Allow for variations
    expect(summary?.batchSize).toBe(4);
  });
});

describe('MLA Inference Memory with TP', () => {
  const v3 = getModel('deepseek-v3', 2048)!;

  it('MLA KV cache per rank is constant across TP degrees (replicated)', () => {
    const memTP1 = calculateMemoryWithTP(v3, 1024, 1, 1, 'bf16', 'bf16');
    const memTP2 = calculateMemoryWithTP(v3, 1024, 1, 2, 'bf16', 'bf16');
    const memTP4 = calculateMemoryWithTP(v3, 1024, 1, 4, 'bf16', 'bf16');

    // KV cache should be identical across TP degrees (replicated, not split)
    expect(memTP1.kvCache).toBe(memTP2.kvCache);
    expect(memTP2.kvCache).toBe(memTP4.kvCache);

    // Weights should decrease with TP
    expect(memTP2.weights).toBeCloseTo(memTP1.weights / 2, -6);
    expect(memTP4.weights).toBeCloseTo(memTP1.weights / 4, -6);
  });

  it('MLA KV cache value matches hand calculation', () => {
    // V3: 61 layers × (512 + 64) × 2 bytes × 1024 seq × 1 batch = 72,097,792
    const mem = calculateMemoryWithTP(v3, 1024, 1, 1, 'bf16', 'bf16');
    const expected = 61 * 576 * 2 * 1024 * 1;
    expect(mem.kvCache).toBe(expected);
  });
});
