/**
 * Batch Scaling Validation Tests
 *
 * Validates that inference latency and throughput scale correctly with batch size.
 * Key physics:
 *   - TTFT (prefill) scales linearly with batch (compute-bound, FLOPs ∝ batch)
 *   - TPOT (decode) decreases with batch (weight amortization: weights read once per step)
 *   - Decode wall-clock time per step is roughly constant for small batch (weights-dominated)
 *   - Throughput scales ~linearly with batch (not quadratically!)
 *   - Cost per token decreases ~linearly with batch (not quadratically!)
 *
 * These tests guard against the B² scaling bug where:
 *   - TTFT didn't include batch in prefill FLOPs
 *   - decodeTime used per-token TPOT instead of per-step time
 *   - TP AllReduce overhead wasn't normalized to per-token
 */

import { describe, it, expect } from 'vitest';
import {
  estimateTTFT,
  estimateTPOT,
  calculateLatencyMetrics,
  calculateLatencyWithTP,
} from '../../src/core/inference/latency.ts';
import {
  runInferenceSimulation,
  type InferenceSimulationConfig,
} from '../../src/core/inference/simulation.ts';
import { getModel } from '../../src/core/models/index.ts';
import { H100_SXM } from '../../src/core/hardware/gpu.ts';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function infer(cfg: Partial<InferenceSimulationConfig>) {
  return runInferenceSimulation({
    modelId: 'llama3-8b',
    gpuId: 'h100-sxm',
    numGPUs: 1,
    batchSize: 1,
    inputSeqLen: 512,
    outputSeqLen: 256,
    weightPrecision: 'bf16',
    kvCachePrecision: 'bf16',
    ...cfg,
  });
}

const llama8b = getModel('llama3-8b', 512)!;

// ===========================================================================
// Section 1: TTFT scales with batch size
// ===========================================================================

describe('TTFT batch scaling', () => {
  it('TTFT at batch=1 equals base TTFT (no batch overhead)', () => {
    const ttft1 = estimateTTFT(llama8b, 512, H100_SXM, 'bf16', 1);
    const ttftDefault = estimateTTFT(llama8b, 512, H100_SXM, 'bf16');
    expect(ttft1).toBeCloseTo(ttftDefault, 6);
  });

  it('TTFT scales linearly with batch size (compute-bound)', () => {
    const ttft1 = estimateTTFT(llama8b, 512, H100_SXM, 'bf16', 1);
    const ttft4 = estimateTTFT(llama8b, 512, H100_SXM, 'bf16', 4);
    const ttft16 = estimateTTFT(llama8b, 512, H100_SXM, 'bf16', 16);

    // Compute-bound: TTFT should scale roughly linearly
    // May not be exact 4x if memory-bound at batch=1, but at batch=4+ it's compute-bound
    expect(ttft4).toBeGreaterThan(ttft1 * 2); // At least 2x for 4x batch
    expect(ttft16 / ttft4).toBeGreaterThan(3.5); // ~4x for 4x batch increase
    expect(ttft16 / ttft4).toBeLessThan(4.5);
  });

  it('TTFT(batch=128) >> TTFT(batch=1)', () => {
    const ttft1 = estimateTTFT(llama8b, 512, H100_SXM, 'bf16', 1);
    const ttft128 = estimateTTFT(llama8b, 512, H100_SXM, 'bf16', 128);

    // 128x batch should give much larger TTFT
    expect(ttft128).toBeGreaterThan(ttft1 * 50);
  });
});

// ===========================================================================
// Section 2: TPOT = inter-token latency (decode step time)
// ===========================================================================

describe('TPOT batch scaling', () => {
  it('TPOT (step time) increases slightly with batch as KV cache grows', () => {
    const tpot1 = estimateTPOT(llama8b, 512, 1, H100_SXM, 'bf16');
    const tpot4 = estimateTPOT(llama8b, 512, 4, H100_SXM, 'bf16');
    const tpot64 = estimateTPOT(llama8b, 512, 64, H100_SXM, 'bf16');

    // Step time increases with batch (more KV cache to read)
    expect(tpot4).toBeGreaterThan(tpot1);
    expect(tpot64).toBeGreaterThan(tpot4);
  });

  it('TPOT growth accelerates as KV cache dominates at high batch', () => {
    // At small batch, weights dominate → step time barely changes
    // At large batch, KV cache grows → step time increases faster
    const tpot1 = estimateTPOT(llama8b, 512, 1, H100_SXM, 'bf16');
    const tpot2 = estimateTPOT(llama8b, 512, 2, H100_SXM, 'bf16');
    const tpot128 = estimateTPOT(llama8b, 512, 128, H100_SXM, 'bf16');
    const tpot256 = estimateTPOT(llama8b, 512, 256, H100_SXM, 'bf16');

    const growth_1_to_2 = tpot2 / tpot1;
    const growth_128_to_256 = tpot256 / tpot128;

    // Early: weights dominate, doubling batch barely changes step time
    expect(growth_1_to_2).toBeGreaterThan(1.0);
    expect(growth_1_to_2).toBeLessThan(1.1);

    // Late: KV cache dominates, doubling batch increases step time more
    expect(growth_128_to_256).toBeGreaterThan(growth_1_to_2);
  });

  it('TPOT at batch=1 vs batch=4 is similar when weights dominate', () => {
    // TPOT = step time = (weights + KV cache) / bandwidth
    // When weights >> KV cache, step time is roughly constant
    const tpot1 = estimateTPOT(llama8b, 512, 1, H100_SXM, 'bf16');
    const tpot4 = estimateTPOT(llama8b, 512, 4, H100_SXM, 'bf16');

    // Step times should be similar (weights dominate at small batch)
    const ratio = tpot4 / tpot1;
    expect(ratio).toBeGreaterThan(0.9);
    expect(ratio).toBeLessThan(1.3); // Small increase due to KV cache
  });
});

// ===========================================================================
// Section 3: Throughput scales linearly (NOT quadratically) with batch
// ===========================================================================

describe('Throughput batch scaling', () => {
  it('throughput scales roughly linearly with batch (not B²)', () => {
    const result1 = infer({ batchSize: 1 });
    const result8 = infer({ batchSize: 8 });
    const result64 = infer({ batchSize: 64 });

    const tps1 = result1.throughput.tokensPerSecond;
    const tps8 = result8.throughput.tokensPerSecond;
    const tps64 = result64.throughput.tokensPerSecond;

    // 8x batch should give roughly 3-8x throughput (linear, with overhead)
    const ratio_8_1 = tps8 / tps1;
    expect(ratio_8_1).toBeGreaterThan(3);
    expect(ratio_8_1).toBeLessThan(8.5);

    // 64x batch should give roughly 20-64x throughput
    const ratio_64_1 = tps64 / tps1;
    expect(ratio_64_1).toBeGreaterThan(15);
    expect(ratio_64_1).toBeLessThan(65);

    // CRITICAL: ratio should NOT scale quadratically
    // If B² bug existed: ratio_64_1 would be ~4096, not ~50
    expect(ratio_64_1).toBeLessThan(200);
  });

  it('throughput 128x batch gives < 128x improvement (sublinear at high batch)', () => {
    // batch=512 OOMs on single H100 for 8B model, so test at 128
    const result1 = infer({ batchSize: 1 });
    const result128 = infer({ batchSize: 128 });

    expect(result128.success).toBe(true);
    const ratio = result128.throughput.tokensPerSecond / result1.throughput.tokensPerSecond;

    // Should be less than 128 (TTFT overhead + KV cache growth)
    expect(ratio).toBeLessThan(128);
    // But still substantial
    expect(ratio).toBeGreaterThan(20);
  });

  it('doubling batch roughly doubles throughput (when weights dominate decode)', () => {
    // At moderate batch sizes where weights >> KV cache
    const result4 = infer({ batchSize: 4 });
    const result8 = infer({ batchSize: 8 });

    const ratio = result8.throughput.tokensPerSecond / result4.throughput.tokensPerSecond;

    // Should be roughly 2x (1.5-2.5)
    expect(ratio).toBeGreaterThan(1.5);
    expect(ratio).toBeLessThan(2.5);
  });
});

// ===========================================================================
// Section 4: Cost per token scales correctly
// ===========================================================================

describe('Cost scaling with batch', () => {
  it('cost per token decreases with batch but not faster than 1/B', () => {
    // Cost ∝ 1/throughput. If throughput ∝ B, cost ∝ 1/B.
    const result1 = infer({ batchSize: 1 });
    const result32 = infer({ batchSize: 32 });
    const result256 = infer({ batchSize: 256 });

    const tps1 = result1.throughput.tokensPerSecond;
    const tps32 = result32.throughput.tokensPerSecond;
    const tps256 = result256.throughput.tokensPerSecond;

    // Cost ratio = tps_higher_batch / tps_lower_batch
    // Should be < 32 (can't be better than linear)
    expect(tps32 / tps1).toBeLessThan(32);
    expect(tps256 / tps1).toBeLessThan(256);

    // But still significant improvement
    expect(tps32 / tps1).toBeGreaterThan(10);
    expect(tps256 / tps1).toBeGreaterThan(40);
  });
});

// ===========================================================================
// Section 5: Grok-1 batch=512 on 3072 H100s produces realistic numbers
// ===========================================================================

describe('Grok-1 batch=512 validation', () => {
  it('Grok-1 FP8 batch=512 on 3072 H100s: cost is realistic', () => {
    const result = infer({
      modelId: 'grok-1',
      gpuId: 'h100-sxm',
      numGPUs: 3072,
      batchSize: 512,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'fp8',
      kvCachePrecision: 'fp8',
      tensorParallel: 8,
    });

    expect(result.success).toBe(true);

    // 384 replicas × per-replica throughput → cluster-wide throughput
    // Each replica handles batchPerReplica=ceil(512/384)=2 sequences
    const tps = result.throughput.tokensPerSecond;
    expect(tps).toBeGreaterThan(10000);   // 384 replicas, each doing ~100+ tok/s
    expect(tps).toBeLessThan(500000);     // not millions
  });

  it('Grok-1 batch scaling on single replica: 64x batch gives sublinear improvement', () => {
    // Use 8 GPUs with TP=8 → 1 replica, so batch scales directly
    const result1 = infer({
      modelId: 'grok-1',
      gpuId: 'h100-sxm',
      numGPUs: 8,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'fp8',
      kvCachePrecision: 'fp8',
      tensorParallel: 8,
    });

    const result64 = infer({
      modelId: 'grok-1',
      gpuId: 'h100-sxm',
      numGPUs: 8,
      batchSize: 64,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'fp8',
      kvCachePrecision: 'fp8',
      tensorParallel: 8,
    });

    const ratio = result64.throughput.tokensPerSecond / result1.throughput.tokensPerSecond;

    // Must be sublinear in batch
    expect(ratio).toBeLessThan(64);
    // But batch=64 should still be a significant improvement
    expect(ratio).toBeGreaterThan(10);
  });

  it('Grok-1 uses activeParams for prefill FLOPs (MoE)', () => {
    const grok1 = getModel('grok-1', 512)!;
    // Grok-1 is MoE: activeParams << totalParams
    expect(grok1.activeParams).toBeLessThan(grok1.totalParams * 0.5);

    // TTFT should be based on activeParams (reasonable), not totalParams (too slow)
    const ttft = estimateTTFT(grok1, 512, H100_SXM, 'fp8', 1);
    // Actual: ~76ms for Grok-1 batch=1 FP8
    expect(ttft).toBeGreaterThan(40);
    expect(ttft).toBeLessThan(120);
  });
});

// ===========================================================================
// Section 6: TP path batch correctness
// ===========================================================================

describe('TP batch scaling', () => {
  it('TP TPOT (step time) increases with batch due to KV cache and AllReduce', () => {
    const latency1 = calculateLatencyWithTP(llama8b, 512, 256, 1, H100_SXM, 2, 'bf16');
    const latency8 = calculateLatencyWithTP(llama8b, 512, 256, 8, H100_SXM, 2, 'bf16');

    // TPOT is step time — increases with batch (more KV cache + larger AllReduce)
    expect(latency8.tpot).toBeGreaterThan(latency1.tpot);

    // Decode time also increases (more per-step work)
    expect(latency8.decodeTime).toBeGreaterThan(latency1.decodeTime);
  });

  it('TP throughput at batch=64 is realistic', () => {
    const result = infer({
      modelId: 'llama3-70b',
      gpuId: 'h100-sxm',
      numGPUs: 8,
      batchSize: 64,
      inputSeqLen: 512,
      outputSeqLen: 256,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
      tensorParallel: 8,
    });

    expect(result.success).toBe(true);
    // Actual: ~4655 tok/s. Published: ~460 tok/s per GPU (our number is total).
    expect(result.throughput.tokensPerSecond).toBeGreaterThan(3500);
    expect(result.throughput.tokensPerSecond).toBeLessThan(5800);
  });
});

// ===========================================================================
// Section 7: Total latency decomposition at high batch
// ===========================================================================

describe('Latency decomposition', () => {
  it('prefillTime + decodeTime = totalLatency', () => {
    const metrics = calculateLatencyMetrics(llama8b, 512, 256, 64, H100_SXM, 'bf16');
    expect(metrics.totalLatency).toBeCloseTo(metrics.prefillTime + metrics.decodeTime, 3);
  });

  it('at high batch, TTFT (prefill) becomes significant fraction of total', () => {
    const metrics1 = calculateLatencyMetrics(llama8b, 512, 256, 1, H100_SXM, 'bf16');
    const metrics128 = calculateLatencyMetrics(llama8b, 512, 256, 128, H100_SXM, 'bf16');

    const prefillFrac1 = metrics1.prefillTime / metrics1.totalLatency;
    const prefillFrac128 = metrics128.prefillTime / metrics128.totalLatency;

    // At batch=1, prefill is usually a small fraction
    // At batch=128, prefill becomes more significant (but decode still dominates with long output)
    expect(prefillFrac128).toBeGreaterThan(prefillFrac1);
  });

  it('decodeTime includes batch factor', () => {
    const metrics1 = calculateLatencyMetrics(llama8b, 512, 256, 1, H100_SXM, 'bf16');
    const metrics4 = calculateLatencyMetrics(llama8b, 512, 256, 4, H100_SXM, 'bf16');

    // Decode time at batch=4 should be roughly same as batch=1
    // (step time is similar, both have 256 steps)
    // With batch=4: decodeTime = tpot * 256 * 4, but tpot is ~4x smaller
    // So decodeTime is roughly the same
    const ratio = metrics4.decodeTime / metrics1.decodeTime;
    expect(ratio).toBeGreaterThan(0.8);
    expect(ratio).toBeLessThan(1.5);
  });
});

// ===========================================================================
// Section 8: Batch size recommendation
// ===========================================================================

describe('Batch size recommendation', () => {
  it('recommends increasing batch when GPU memory is underutilized', () => {
    const result = infer({ batchSize: 1 });
    expect(result.recommendations.length).toBeGreaterThan(0);
    expect(result.recommendations.some(r => /increasing batch size/i.test(r))).toBe(true);
  });

  it('recommendation describes direction without specific numbers', () => {
    const result = infer({ batchSize: 1 });
    const rec = result.recommendations.find(r => r.includes('increasing batch size'));
    expect(rec).toBeDefined();
    // Should explain why (headroom + throughput benefit)
    expect(rec).toMatch(/headroom/i);
    expect(rec).toMatch(/throughput/);
    // Should NOT contain specific numbers
    expect(rec).not.toMatch(/~\d+/);
    expect(rec).not.toMatch(/\d+%/);
    expect(rec).not.toMatch(/\d+×/);
  });

  it('no batch recommendation when GPU memory is near-full', () => {
    // Use 8 GPUs with TP=8 (1 replica) and high batch so memory is well-utilized
    // Grok-1 on 8xH100 with large batch fills memory, leaving little room
    const result = infer({
      modelId: 'grok-1',
      numGPUs: 8,
      batchSize: 512,
      inputSeqLen: 512,
      outputSeqLen: 256,
      weightPrecision: 'fp8',
      kvCachePrecision: 'bf16',
      tensorParallel: 8,
    });
    const batchRecs = result.recommendations.filter(r => r.includes('increasing batch'));
    expect(batchRecs.length).toBe(0);
  });

  it('Grok-1 batch=1 gets batch increase recommendation at massive scale', () => {
    // With 3072 GPUs / TP=8 = 384 replicas, batch=1 gives batchPerReplica=1.
    // batchSizeIncrease now jumps to numReplicas×2=768 (batchPerReplica=2),
    // which amortizes weight reads and improves throughput.
    const result = infer({
      modelId: 'grok-1',
      numGPUs: 3072,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 256,
      weightPrecision: 'fp8',
      kvCachePrecision: 'bf16',
      tensorParallel: 8,
    });
    expect(result.recommendations.length).toBeGreaterThan(0);
    // Batch increase should now fire (jumps to numReplicas×2 instead of batch×2)
    const batchRec = result.recommendations.find(r => r.includes('batch size') || r.includes('increasing batch'));
    expect(batchRec).toBeDefined();
  });

  it('Grok-1 batch=512 on single replica shows no strong batch recommendation', () => {
    // Use 8 GPUs with TP=8 (1 replica) so batch=512 actually fills memory
    const result = infer({
      modelId: 'grok-1',
      numGPUs: 8,
      batchSize: 512,
      inputSeqLen: 512,
      outputSeqLen: 256,
      weightPrecision: 'fp8',
      kvCachePrecision: 'bf16',
      tensorParallel: 8,
    });
    // At batch=512 on single replica, memory is well-utilized
    // Max batch improvement from 512→higher is < 2×, so no rec
    const batchRecs = result.recommendations.filter(r => r.includes('increasing batch'));
    expect(batchRecs.length).toBe(0);
  });
});

// ===========================================================================
// Section 9: Regression guard — batch=1 unchanged
// ===========================================================================

describe('Batch=1 regression guard', () => {
  it('Llama 3 8B batch=1 TTFT unchanged', () => {
    const ttft = estimateTTFT(llama8b, 512, H100_SXM, 'bf16', 1);
    // This should be the same as before the fix (batch=1 was always correct)
    expect(ttft).toBeGreaterThan(0.01);
    expect(ttft).toBeLessThan(50);
  });

  it('Llama 3 8B batch=1 throughput reasonable', () => {
    const result = infer({ batchSize: 1 });
    // At batch=1, single GPU, BF16: ~80-200 tok/s expected
    expect(result.throughput.tokensPerSecond).toBeGreaterThan(50);
    expect(result.throughput.tokensPerSecond).toBeLessThan(500);
  });
});
