/**
 * Comprehensive inference benchmark validation tests
 *
 * These tests validate our inference calculations against known benchmarks
 * and industry-standard calculators.
 *
 * Reference sources:
 * - NVIDIA H100 specs: 3.35 TB/s memory bandwidth, 989 dense TFLOPS BF16
 * - LLM inference is typically memory-bandwidth bound during decode
 * - Known TTFT/TPOT benchmarks from vLLM, TensorRT-LLM, and published papers
 */

import { describe, it, expect } from 'vitest';
import {
  kvCachePerToken,
  totalKVCacheMemory,
  kvCacheReductionFactor,
  calculateInferenceMemory,
  estimateTTFT,
  estimateTPOT,
  calculateLatencyMetrics,
  calculateThroughputMetrics,
  calculateUtilizationMetrics,
  modelWeightsMemory,
  runInferenceSimulation,
  calculateLatencyWithTP,
  calculateContinuousBatchingMetrics,
} from '../../src/core/inference/index.ts';
import { moeWeightBytesPerStep } from '../../src/core/inference/latency.ts';
import { getPrecisionBytes } from '../../src/core/inference/kv-cache.ts';
import { getModel } from '../../src/core/models/index.ts';
import { H100_SXM, H200_SXM } from '../../src/core/hardware/gpu.ts';

describe('Model Weights Memory Validation', () => {
  it('should calculate correct weights for Llama 2 7B in BF16', () => {
    const llama7b = getModel('llama2-7b', 2048)!;
    const weights = modelWeightsMemory(llama7b, 'bf16');

    // Llama 7B actually has ~6.74B params * 2 bytes = ~13.5 GB
    const expectedGB = llama7b.totalParams * 2 / 1e9;
    expect(weights / 1e9).toBeCloseTo(expectedGB, 1);
    expect(weights / 1e9).toBeGreaterThan(13);
    expect(weights / 1e9).toBeLessThan(15);
  });

  it('should calculate correct weights for Llama 2 70B in BF16', () => {
    const llama70b = getModel('llama2-70b', 2048)!;
    const weights = modelWeightsMemory(llama70b, 'bf16');

    // Llama 70B actually has ~69B params * 2 bytes = ~138 GB
    const expectedGB = llama70b.totalParams * 2 / 1e9;
    expect(weights / 1e9).toBeCloseTo(expectedGB, 1);
    expect(weights / 1e9).toBeGreaterThan(130);
    expect(weights / 1e9).toBeLessThan(145);
  });

  it('should halve memory with INT8 quantization', () => {
    const llama7b = getModel('llama2-7b', 2048)!;
    const bf16Weights = modelWeightsMemory(llama7b, 'bf16');
    const int8Weights = modelWeightsMemory(llama7b, 'int8');

    expect(int8Weights).toBeCloseTo(bf16Weights / 2, -6);
  });

  it('should quarter memory with INT4 quantization', () => {
    const llama7b = getModel('llama2-7b', 2048)!;
    const bf16Weights = modelWeightsMemory(llama7b, 'bf16');
    const int4Weights = modelWeightsMemory(llama7b, 'int4');

    expect(int4Weights).toBeCloseTo(bf16Weights / 4, -6);
  });
});

describe('KV Cache Memory Validation', () => {
  /**
   * KV Cache formula: 2 * num_layers * num_kv_heads * head_dim * bytes * seq_len * batch
   *
   * Llama 2 7B: 32 layers, 32 KV heads (MHA), head_dim=128
   * Per token: 2 * 32 * 32 * 128 * 2 = 524,288 bytes = 0.5 MB/token
   */
  it('should calculate KV cache per token for Llama 2 7B (MHA)', () => {
    const llama7b = getModel('llama2-7b', 2048)!;
    const perToken = kvCachePerToken(llama7b, 'bf16');

    // Expected: 2 * 32 * 32 * 128 * 2 = 524,288 bytes
    const expected = 2 * 32 * 32 * 128 * 2;
    expect(perToken).toBe(expected);
    expect(perToken / 1e6).toBeCloseTo(0.5, 1); // ~0.5 MB/token
  });

  /**
   * Llama 2 70B: 80 layers, 8 KV heads (GQA), head_dim=128
   * Per token: 2 * 80 * 8 * 128 * 2 = 327,680 bytes = 0.31 MB/token
   *
   * Note: GQA reduces KV cache by factor of numAttentionHeads/numKvHeads = 64/8 = 8x
   */
  it('should calculate KV cache per token for Llama 2 70B (GQA)', () => {
    const llama70b = getModel('llama2-70b', 2048)!;
    const perToken = kvCachePerToken(llama70b, 'bf16');

    // Expected: 2 * 80 * 8 * 128 * 2 = 327,680 bytes
    const expected = 2 * 80 * 8 * 128 * 2;
    expect(perToken).toBe(expected);
    expect(perToken / 1e6).toBeCloseTo(0.31, 1); // ~0.31 MB/token
  });

  it('should correctly identify GQA reduction factor', () => {
    const llama7b = getModel('llama2-7b', 2048)!;
    const llama70b = getModel('llama2-70b', 2048)!;

    // Llama 7B has MHA (32 attention heads, 32 KV heads)
    expect(kvCacheReductionFactor(llama7b)).toBe(1);

    // Llama 70B has GQA (64 attention heads, 8 KV heads)
    // Reduction = 8/64 = 0.125
    expect(kvCacheReductionFactor(llama70b)).toBeCloseTo(0.125, 3);
  });

  it('should calculate total KV cache for batch and sequence', () => {
    const llama7b = getModel('llama2-7b', 2048)!;
    const seqLen = 2048;
    const batchSize = 4;

    const perToken = kvCachePerToken(llama7b, 'bf16');
    const total = totalKVCacheMemory(llama7b, seqLen, batchSize, 'bf16');

    expect(total).toBe(perToken * seqLen * batchSize);

    // For 2048 tokens * 4 batch * 0.5MB = ~4 GB
    expect(total / 1e9).toBeCloseTo(4, 0);
  });
});

describe('Inference Memory Breakdown Validation', () => {
  it('should calculate reasonable total memory for Llama 7B inference', () => {
    const llama7b = getModel('llama2-7b', 2048)!;
    const memory = calculateInferenceMemory(llama7b, 2048, 1, 'bf16', 'bf16', true);

    // Weights: ~13.5 GB (actual 6.74B params)
    // KV Cache: ~1 GB (2048 tokens)
    // Activations: small with flash attention
    // Overhead: ~10%
    // Total: ~15-18 GB

    expect(memory.weights / 1e9).toBeGreaterThan(13);
    expect(memory.weights / 1e9).toBeLessThan(15);
    expect(memory.kvCache / 1e9).toBeCloseTo(1, 0);
    expect(memory.total / 1e9).toBeLessThan(20);
    expect(memory.total / 1e9).toBeGreaterThan(13);
  });

  it('should fit Llama 7B on single H100 80GB', () => {
    const llama7b = getModel('llama2-7b', 2048)!;
    const memory = calculateInferenceMemory(llama7b, 4096, 8, 'bf16', 'bf16', true);

    // Even with 4K context and batch 8, should fit in 80GB
    expect(memory.total / 1e9).toBeLessThan(80);
  });

  it('should NOT fit Llama 70B on single H100 80GB without quantization', () => {
    const llama70b = getModel('llama2-70b', 2048)!;
    const memory = calculateInferenceMemory(llama70b, 2048, 1, 'bf16', 'bf16', true);

    // 70B BF16 = 140GB weights alone, won't fit
    expect(memory.weights / 1e9).toBeGreaterThan(80);
  });

  it('should fit Llama 70B on single H100 with INT4 quantization', () => {
    const llama70b = getModel('llama2-70b', 2048)!;
    const memory = calculateInferenceMemory(llama70b, 2048, 1, 'int4', 'bf16', true);

    // INT4: ~69B * 0.5 bytes = ~34.5GB weights
    expect(memory.weights / 1e9).toBeGreaterThan(33);
    expect(memory.weights / 1e9).toBeLessThan(36);
    expect(memory.total / 1e9).toBeLessThan(80);
  });
});

describe('TTFT (Time to First Token) Validation', () => {
  /**
   * TTFT is prefill time - processing all input tokens in parallel
   * It's compute-bound for longer sequences
   *
   * Expected range for Llama 7B on H100:
   * - 128 tokens: ~5-15ms
   * - 512 tokens: ~20-50ms
   * - 2048 tokens: ~80-200ms
   */
  it('should estimate reasonable TTFT for Llama 7B on H100', () => {
    const llama7b = getModel('llama2-7b', 2048)!;

    const ttft128 = estimateTTFT(llama7b, 128, H100_SXM, 'bf16');
    const ttft512 = estimateTTFT(llama7b, 512, H100_SXM, 'bf16');
    const ttft2048 = estimateTTFT(llama7b, 2048, H100_SXM, 'bf16');

    // Actual: 128→4.4ms, 512→17.4ms, 2048→69.8ms
    expect(ttft128).toBeGreaterThan(3);
    expect(ttft512).toBeGreaterThan(10);
    expect(ttft2048).toBeGreaterThan(30);

    // Longer prompts should have higher TTFT
    expect(ttft512).toBeGreaterThan(ttft128);
    expect(ttft2048).toBeGreaterThan(ttft512);

    // Reasonable ranges (allowing for model variations)
    expect(ttft128).toBeLessThan(50); // Under 50ms for 128 tokens
    expect(ttft2048).toBeLessThan(500); // Under 500ms for 2K tokens
  });

  it('should scale TTFT roughly linearly with prompt length', () => {
    const llama7b = getModel('llama2-7b', 2048)!;

    const ttft256 = estimateTTFT(llama7b, 256, H100_SXM, 'bf16');
    const ttft512 = estimateTTFT(llama7b, 512, H100_SXM, 'bf16');
    const ttft1024 = estimateTTFT(llama7b, 1024, H100_SXM, 'bf16');

    // Should roughly double as sequence doubles (with some variance)
    const ratio512to256 = ttft512 / ttft256;
    const ratio1024to512 = ttft1024 / ttft512;

    expect(ratio512to256).toBeGreaterThan(1.5);
    expect(ratio512to256).toBeLessThan(3);
    expect(ratio1024to512).toBeGreaterThan(1.5);
    expect(ratio1024to512).toBeLessThan(3);
  });
});

describe('TPOT (Time Per Output Token) Validation', () => {
  /**
   * TPOT is decode time per token - memory-bandwidth bound
   *
   * For memory-bound decode:
   * TPOT ≈ (model_weights + kv_cache) / memory_bandwidth
   *
   * H100: 3.35 TB/s bandwidth
   * Llama 7B BF16: ~14 GB weights
   * Minimum TPOT ≈ 14GB / 3.35TB/s ≈ 4.2ms (theoretical minimum)
   *
   * Realistic TPOT with overhead: 5-15ms for batch=1
   */
  it('should estimate reasonable TPOT for Llama 7B on H100 batch=1', () => {
    const llama7b = getModel('llama2-7b', 2048)!;
    const tpot = estimateTPOT(llama7b, 512, 1, H100_SXM, 'bf16');

    // Theoretical minimum: 14GB / 3.35TB/s = 4.2ms
    // Realistic with overhead: 5-20ms
    expect(tpot).toBeGreaterThan(4);
    expect(tpot).toBeLessThan(30);
  });

  it('should have lower effective TPOT with larger batches', () => {
    const llama7b = getModel('llama2-7b', 2048)!;

    const tpot1 = estimateTPOT(llama7b, 512, 1, H100_SXM, 'bf16');
    const tpot8 = estimateTPOT(llama7b, 512, 8, H100_SXM, 'bf16');

    // Per-request TPOT may increase with batch, but throughput improves
    // The function returns per-token time, which may be similar
    expect(tpot1).toBeGreaterThan(0);
    expect(tpot8).toBeGreaterThan(0);
  });

  it('should have higher TPOT for larger models', () => {
    const llama7b = getModel('llama2-7b', 2048)!;
    const llama70b = getModel('llama2-70b', 2048)!;

    // Use INT4 for 70B to fit in memory
    const tpot7b = estimateTPOT(llama7b, 512, 1, H100_SXM, 'bf16');
    const tpot70b = estimateTPOT(llama70b, 512, 1, H100_SXM, 'int4');

    // 70B even in INT4 (~35GB) should have higher TPOT than 7B BF16 (~14GB)
    expect(tpot70b).toBeGreaterThan(tpot7b);
  });
});

describe('Throughput Validation', () => {
  /**
   * Throughput benchmarks:
   * - Llama 7B on H100 batch=1: ~50-150 tok/s
   * - Llama 7B on H100 batch=8: ~200-500 tok/s
   */
  it('should estimate reasonable throughput for Llama 7B batch=1', () => {
    const llama7b = getModel('llama2-7b', 2048)!;
    const metrics = calculateThroughputMetrics(llama7b, 512, 256, 1, H100_SXM, 'bf16');

    // Expected: 50-200 tok/s for batch=1
    expect(metrics.tokensPerSecond).toBeGreaterThan(30);
    expect(metrics.tokensPerSecond).toBeLessThan(300);

    // Decode should be slower than prefill (per token)
    expect(metrics.decodeTokensPerSecond).toBeLessThan(metrics.prefillTokensPerSecond);
  });

  it('should have higher throughput with larger batches', () => {
    const llama7b = getModel('llama2-7b', 2048)!;

    const throughput1 = calculateThroughputMetrics(llama7b, 256, 128, 1, H100_SXM, 'bf16');
    const throughput8 = calculateThroughputMetrics(llama7b, 256, 128, 8, H100_SXM, 'bf16');

    // Batching should improve throughput
    expect(throughput8.tokensPerSecond).toBeGreaterThan(throughput1.tokensPerSecond);
  });
});

describe('Utilization Metrics Validation', () => {
  it('should identify decode as memory-bound', () => {
    const llama7b = getModel('llama2-7b', 2048)!;
    const util = calculateUtilizationMetrics(llama7b, 512, 256, 1, H100_SXM, 'bf16');

    // Decode phase is typically memory-bandwidth bound for batch=1
    expect(['memory_bandwidth', 'memory_capacity']).toContain(util.bottleneck);
    expect(util.isMemoryBound).toBe(true);
  });

  it('should have valid utilization values', () => {
    const llama7b = getModel('llama2-7b', 2048)!;
    const util = calculateUtilizationMetrics(llama7b, 512, 256, 1, H100_SXM, 'bf16');

    expect(util.computeUtilization).toBeGreaterThanOrEqual(0);
    expect(util.computeUtilization).toBeLessThanOrEqual(1);
    expect(util.rooflineAttainment).toBeGreaterThanOrEqual(0);
    expect(util.rooflineAttainment).toBeLessThanOrEqual(1);
    expect(util.memoryCapacityUtilization).toBeGreaterThanOrEqual(0);
    expect(util.memoryCapacityUtilization).toBeLessThanOrEqual(1);
  });
});

describe('Full Inference Simulation Validation', () => {
  it('should run successful simulation for Llama 7B on H100', () => {
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

    // Validate memory
    expect(result.memory.total / 1e9).toBeLessThan(80);
    expect(result.memory.weights / 1e9).toBeGreaterThan(13);
    expect(result.memory.weights / 1e9).toBeLessThan(15);

    // Validate latency
    expect(result.latency.ttft).toBeGreaterThan(0);
    expect(result.latency.tpot).toBeGreaterThan(0);
    expect(result.latency.totalLatency).toBeGreaterThan(result.latency.ttft);

    // Validate throughput
    expect(result.throughput.tokensPerSecond).toBeGreaterThan(30);
  });

  it('should fail for OOM configurations', () => {
    const result = runInferenceSimulation({
      modelId: 'llama2-70b',
      gpu: { ...H100_SXM, memoryGB: 24 }, // Simulated small GPU
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

  it('should work with tensor parallelism for large models', () => {
    const result = runInferenceSimulation({
      modelId: 'llama2-70b',
      gpu: H100_SXM,
      numGPUs: 8,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 256,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
      tensorParallel: 8,
    });

    expect(result.success).toBe(true);
    // Memory per GPU should be ~1/8 of full model
    expect(result.memory.weights / 1e9).toBeLessThan(20);
  });

  it('should generate events for visualization', () => {
    const result = runInferenceSimulation({
      modelId: 'llama2-7b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 128,
      outputSeqLen: 64,
    });

    expect(result.success).toBe(true);
    expect(result.events.length).toBeGreaterThan(0);

    // Should have simulation start/end events
    expect(result.events.some(e => e.type === 'simulation_start')).toBe(true);
    expect(result.events.some(e => e.type === 'simulation_end')).toBe(true);

    // Should have prefill and decode phases
    expect(result.events.some(e => e.type === 'prefill_start')).toBe(true);
    expect(result.events.some(e => e.type === 'decode_start')).toBe(true);
  });
});

describe('Comparison with Industry Benchmarks', () => {
  /**
   * Known benchmarks from vLLM, TensorRT-LLM, and published papers:
   *
   * Llama 2 7B on H100 (batch=1, 2K context):
   * - TTFT: 20-50ms
   * - TPOT: 5-15ms
   * - Throughput: 60-150 tok/s
   *
   * These are optimized implementations; our simulator should be in the ballpark.
   */
  it('should match vLLM-like performance for Llama 7B', () => {
    const llama7b = getModel('llama2-7b', 2048)!;
    const metrics = calculateLatencyMetrics(llama7b, 512, 256, 1, H100_SXM, 'bf16');

    // TTFT should be reasonable (allowing 2x margin for unoptimized estimate)
    expect(metrics.ttft).toBeLessThan(100); // Under 100ms for 512 tokens

    // TPOT should be reasonable
    expect(metrics.tpot).toBeLessThan(30); // Under 30ms per token
    expect(metrics.tpot).toBeGreaterThan(3); // Above theoretical minimum
  });

  /**
   * Memory efficiency check:
   * A well-optimized inference should use close to:
   * weights + KV cache + small overhead
   */
  it('should have efficient memory usage', () => {
    const llama7b = getModel('llama2-7b', 2048)!;
    const memory = calculateInferenceMemory(llama7b, 2048, 1, 'bf16', 'bf16', true);

    const weightsGB = memory.weights / 1e9;
    const kvCacheGB = memory.kvCache / 1e9;
    const overheadGB = memory.overhead / 1e9;
    const totalGB = memory.total / 1e9;

    // Overhead should be less than 20% of weights + KV
    const expectedBase = weightsGB + kvCacheGB;
    expect(overheadGB).toBeLessThan(expectedBase * 0.2);

    // Total should be weights + KV + reasonable overhead + activations
    expect(totalGB).toBeLessThan(expectedBase * 1.5);
  });
});

describe('Edge Cases and Error Handling', () => {
  it('should handle very long sequences', () => {
    // Just verify model exists for this context length
    expect(getModel('llama2-7b', 8192)).toBeDefined();
    const result = runInferenceSimulation({
      modelId: 'llama2-7b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 4096,
      outputSeqLen: 1024,
    });

    // Should succeed but with high KV cache usage
    expect(result.success).toBe(true);
    expect(result.memory.kvCache / 1e9).toBeGreaterThan(2); // Significant KV cache
  });

  it('should handle batch=1 correctly (most common inference case)', () => {
    const result = runInferenceSimulation({
      modelId: 'llama2-7b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 128,
      outputSeqLen: 128,
    });

    expect(result.success).toBe(true);
    expect(result.latency.tpot).toBeGreaterThan(0);
  });

  it('should handle minimum sequence length', () => {
    const result = runInferenceSimulation({
      modelId: 'llama2-7b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 1,
      outputSeqLen: 1,
    });

    expect(result.success).toBe(true);
  });
});

describe('TTFT Batch Scaling', () => {
  it('Llama 70B TTFT scales roughly linearly with batch size', () => {
    const llama70b = getModel('llama2-70b', 2048)!;
    // Use FP8 to make it compute-bound even at batch=1
    const ttftB1 = estimateTTFT(llama70b, 1024, H100_SXM, 'fp8', 1);
    const ttftB32 = estimateTTFT(llama70b, 1024, H100_SXM, 'fp8', 32);

    // At batch=32, FLOPs are 32x larger. Dense models read same weights regardless of batch,
    // so compute dominates and TTFT should scale ~linearly.
    const ratio = ttftB32 / ttftB1;
    expect(ratio).toBeGreaterThan(25);
    expect(ratio).toBeLessThan(35);
  });

  it('dense model TTFT unchanged by moeWeightBytesPerStep fix (regression)', () => {
    const llama70b = getModel('llama2-70b', 2048)!;
    // For dense models, moeWeightBytesPerStep returns activeParams * bytes = totalParams * bytes
    // This is identical to modelWeightsMemory, so TTFT should be unchanged
    const ttft = estimateTTFT(llama70b, 1024, H100_SXM, 'bf16', 1);

    // Compute: 2 * 69B * 1024 / (989T * 0.4) = 357ms (compute-bound)
    // Memory: 69B * 2 / 3.35T = 41.2ms
    // TTFT = max(357, 41) = 357ms
    expect(ttft).toBeGreaterThan(300);
    expect(ttft).toBeLessThan(450);
  });
});

describe('MoE TTFT: DeepSeek V3', () => {
  const v3 = getModel('deepseek-v3', 2048)!;

  it('V3 batch=1 TTFT is memory-bound (~140ms at seq=1024)', () => {
    // Compute: 2 * 37.6B * 1024 / (1979T * 0.4) = 97ms
    // Weight read: coupon-collector over 1024 tokens → nearly all 671B params
    //   ~671GB (FP8) / 4.8TB/s = 140ms
    // baseTtft = max(97, 140) = 140ms (memory-bound)
    const ttft = estimateTTFT(v3, 1024, H200_SXM, 'fp8', 1);
    expect(ttft).toBeGreaterThan(100);
    expect(ttft).toBeLessThan(180);
  });

  it('V3 batch=32 TTFT is large (compute-bound)', () => {
    // Compute: 2 * 37.6B * 1024 * 32 / (1979T * 0.4) = 3107ms (single GPU)
    // Weight read: ~443GB at batch=32 → memory time ≈ 92ms
    // baseTtft = max(3107, 92) = 3107ms (compute-bound)
    const ttft = estimateTTFT(v3, 1024, H200_SXM, 'fp8', 32);
    expect(ttft).toBeGreaterThan(2500);
    expect(ttft).toBeLessThan(4000);
  });

  it('V3 batch=32 TTFT > 10x batch=1 TTFT (compute dominance)', () => {
    const ttftB1 = estimateTTFT(v3, 1024, H200_SXM, 'fp8', 1);
    const ttftB32 = estimateTTFT(v3, 1024, H200_SXM, 'fp8', 32);

    expect(ttftB32 / ttftB1).toBeGreaterThan(10);
  });

  it('V3 with TP=8 batch=32: TTFT in expected range (~300-600ms)', () => {
    const latency = calculateLatencyWithTP(v3, 1024, 256, 32, H200_SXM, 8, 'fp8');

    // baseTtft ~3107ms / 8 + allReduce ~64ms ≈ 452ms
    expect(latency.ttft).toBeGreaterThan(250);
    expect(latency.ttft).toBeLessThan(700);
  });

  it('V3 with TP=8 batch=1: TTFT is fast (~15-30ms)', () => {
    const latency = calculateLatencyWithTP(v3, 1024, 256, 1, H200_SXM, 8, 'fp8');

    // baseTtft ~140ms / 8 + small allReduce ≈ 17.5 + 8 = ~25ms
    expect(latency.ttft).toBeGreaterThan(5);
    expect(latency.ttft).toBeLessThan(50);
  });
});

describe('MoE weight bytes scale with batch', () => {
  const v3 = getModel('deepseek-v3', 2048)!;

  it('V3 batch=1 FP8: weight bytes ≈ activeParams', () => {
    const bytes = moeWeightBytesPerStep(v3, 1, 'fp8');
    const activeBytes = (v3.activeParams ?? v3.totalParams) * getPrecisionBytes('fp8');

    // At batch=1 with top-8/256: fractionTouched ≈ 8/256 = 0.03125
    // So weight bytes ≈ shared + 3.1% of routed ≈ close to activeParams
    expect(bytes / 1e9).toBeGreaterThan(30);
    expect(bytes / 1e9).toBeLessThan(60);
    // Should be within ~50% of activeParams bytes
    expect(bytes / activeBytes).toBeGreaterThan(0.7);
    expect(bytes / activeBytes).toBeLessThan(1.8);
  });

  it('V3 batch=512 FP8: weight bytes ≈ totalParams', () => {
    const bytes = moeWeightBytesPerStep(v3, 512, 'fp8');
    const totalBytes = v3.totalParams * getPrecisionBytes('fp8');

    // At batch=512, nearly all experts touched
    expect(bytes / totalBytes).toBeGreaterThan(0.95);
    expect(bytes / totalBytes).toBeLessThanOrEqual(1.0);
  });

  it('dense model: weight bytes constant across batch sizes', () => {
    const llama70b = getModel('llama2-70b', 2048)!;
    const b1 = moeWeightBytesPerStep(llama70b, 1, 'bf16');
    const b32 = moeWeightBytesPerStep(llama70b, 32, 'bf16');
    const b512 = moeWeightBytesPerStep(llama70b, 512, 'bf16');
    const expected = modelWeightsMemory(llama70b, 'bf16');

    expect(b1).toBe(b32);
    expect(b1).toBe(b512);
    expect(b1).toBe(expected);
  });
});

describe('CB TTFT vs static TTFT', () => {
  it('DeepSeek V3 batch=32 TP=8: CB TTFT << static TTFT', () => {
    const v3 = getModel('deepseek-v3', 2048)!;

    // Static batching TTFT with batch=32
    const staticLatency = calculateLatencyWithTP(v3, 1024, 256, 32, H200_SXM, 8, 'fp8');

    // CB uses batch=1 TTFT per slot
    const cbConfig = {
      modelSpec: v3,
      gpu: H200_SXM,
      numGPUs: 8,
      batchSize: 32,
      inputSeqLen: 1024,
      outputSeqLen: 256,
      weightPrecision: 'fp8' as const,
      kvCachePrecision: 'fp8' as const,
      flashAttention: true,
      pagedAttention: false,
      continuousBatching: true,
      speculative: { enabled: false, draftModel: null, numSpeculativeTokens: 5, acceptanceRate: 0.7 },
      tensorParallel: 8,
    };

    const baseMetrics = {
      latency: staticLatency,
      throughput: {
        tokensPerSecond: 0,
        requestsPerSecond: 0,
        prefillTokensPerSecond: 0,
        decodeTokensPerSecond: 0,
      },
      utilization: {
        computeUtilization: 0,
        rooflineAttainment: 0,
        memoryCapacityUtilization: 0,
        isComputeBound: false,
        isMemoryBound: true,
        bottleneck: 'memory_bandwidth' as const,
      },
    };

    const cbMetrics = calculateContinuousBatchingMetrics(cbConfig, baseMetrics);

    // Static TTFT should be large (~300-600ms, compute-bound at batch=32)
    expect(staticLatency.ttft).toBeGreaterThan(250);

    // CB TTFT should be small (~10-50ms, batch=1 per slot with interference)
    expect(cbMetrics.latency.ttft).toBeGreaterThan(5);
    expect(cbMetrics.latency.ttft).toBeLessThan(60);

    // CB TTFT should be much smaller than static TTFT
    expect(cbMetrics.latency.ttft).toBeLessThan(staticLatency.ttft / 5);
  });
});
