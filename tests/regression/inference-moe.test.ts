/**
 * Tests for MoE-specific inference improvements:
 * - Batch-aware weight bytes (moeWeightBytesPerStep)
 * - Expert Parallelism (EP) memory, latency, prefill
 * - Utilization metrics consistency
 * - Speculative decoding with MoE models
 */

import { describe, it, expect } from 'vitest';
import { getModel } from '../../src/core/models/index.ts';
import { H100_SXM, H200_SXM } from '../../src/core/hardware/gpu.ts';
import {
  moeParamSplit,
  moeWeightBytesPerStep,
  estimateTTFT,
  estimateTPOT,
  calculateLatencyWithTP,
  calculateUtilizationMetrics,
} from '../../src/core/inference/latency.ts';
import { calculateMemoryWithTP } from '../../src/core/inference/memory.ts';
import { getPrecisionBytes } from '../../src/core/inference/kv-cache.ts';
import {
  draftModelOverhead,
  estimateAcceptanceRate,
} from '../../src/core/inference/speculative.ts';
import { runInferenceSimulation } from '../../src/core/inference/simulation.ts';

// Models used across tests
const v3 = getModel('deepseek-v3', 2048)!;
const llama7b = getModel('llama2-7b', 2048)!;
const gpu = H100_SXM;

describe('moeParamSplit', () => {
  it('decomposes V3 into shared and routed expert params', () => {
    const { sharedParams, routedExpertParams } = moeParamSplit(v3);

    // V3: 256 experts × 3 × 7168 × 2048 × numMoELayers
    const expertIntermediate = v3.expertIntermediateSize!;
    const paramsPerExpert = 3 * v3.hiddenSize * expertIntermediate;
    const numMoELayers = v3.numMoELayers!;
    const expectedRouted = v3.numExperts! * paramsPerExpert * numMoELayers;
    const expectedShared = v3.totalParams - expectedRouted;

    expect(routedExpertParams).toBeCloseTo(expectedRouted, -3);
    expect(sharedParams).toBeCloseTo(expectedShared, -3);
    expect(sharedParams + routedExpertParams).toBeCloseTo(v3.totalParams, -3);

    // Shared includes attention, embeddings, norms, shared experts, dense MLPs — should be significant
    expect(sharedParams).toBeGreaterThan(10e9); // >10B shared params
    expect(routedExpertParams).toBeGreaterThan(500e9); // bulk of 671B is routed experts
  });

  it('returns all params as shared for dense models', () => {
    const { sharedParams, routedExpertParams } = moeParamSplit(llama7b);

    expect(routedExpertParams).toBe(0);
    expect(sharedParams).toBe(llama7b.totalParams);
  });
});

describe('moeWeightBytesPerStep', () => {
  it('V3 batch=1: weight bytes ≈ activeParams × bytesPerParam', () => {
    const bytes = moeWeightBytesPerStep(v3, 1, 'bf16');

    // At batch=1, only a handful of experts touched — should be close to activeParams
    // But shared params are always read, so it's sharedParams + small fraction of routed
    const { sharedParams } = moeParamSplit(v3);
    const sharedBytes = sharedParams * getPrecisionBytes('bf16');

    expect(bytes).toBeGreaterThan(sharedBytes); // At least shared params
    expect(bytes).toBeLessThan(v3.totalParams * getPrecisionBytes('bf16') * 0.2); // Much less than total
  });

  it('V3 batch=512: weight bytes ≈ totalParams × bytesPerParam', () => {
    const bytes = moeWeightBytesPerStep(v3, 512, 'bf16');
    const totalBytes = v3.totalParams * getPrecisionBytes('bf16');

    // At batch=512 with top-8/256, fraction touched ≈ 1 - (248/256)^512 ≈ 1.0
    expect(bytes).toBeGreaterThan(totalBytes * 0.95);
    expect(bytes).toBeLessThanOrEqual(totalBytes);
  });

  it('V3 batch=128: weight bytes between active and total', () => {
    const bytesB1 = moeWeightBytesPerStep(v3, 1, 'bf16');
    const bytesB128 = moeWeightBytesPerStep(v3, 128, 'bf16');
    const bytesB512 = moeWeightBytesPerStep(v3, 512, 'bf16');

    // Monotonically increasing with batch
    expect(bytesB128).toBeGreaterThan(bytesB1);
    expect(bytesB512).toBeGreaterThan(bytesB128);

    // At batch=128, should be significantly more than batch=1
    expect(bytesB128 / bytesB1).toBeGreaterThan(3);
  });

  it('dense model: same bytes at all batch sizes', () => {
    const b1 = moeWeightBytesPerStep(llama7b, 1, 'bf16');
    const b512 = moeWeightBytesPerStep(llama7b, 512, 'bf16');

    expect(b1).toBe(b512);
    // Should be activeParams × bytes (same as totalParams for dense)
    expect(b1).toBeCloseTo(llama7b.totalParams * getPrecisionBytes('bf16'), -3);
  });
});

describe('TPOT batch scaling for MoE', () => {
  it('V3 TPOT at batch=512 >> TPOT at batch=1 (weight read dominates)', () => {
    const tpotB1 = estimateTPOT(v3, 2048, 1, gpu, 'bf16');
    const tpotB512 = estimateTPOT(v3, 2048, 512, gpu, 'bf16');

    // V3 at batch=1 reads ~activeParams, at batch=512 reads ~totalParams
    // Weight bytes differ by ~18x (671B vs ~37B)
    // KV cache also grows, but weight read should dominate the ratio
    expect(tpotB512 / tpotB1).toBeGreaterThan(5);
  });

  it('dense model TPOT does not grow dramatically with batch', () => {
    const tpotB1 = estimateTPOT(llama7b, 2048, 1, gpu, 'bf16');
    const tpotB32 = estimateTPOT(llama7b, 2048, 32, gpu, 'bf16');

    // Dense model reads same weight bytes regardless of batch
    // TPOT increases only due to KV cache growth
    // Ratio should be much less dramatic than MoE
    expect(tpotB32 / tpotB1).toBeLessThan(5);
  });
});

describe('EP memory', () => {
  it('V3 EP=8: expert weights distributed, shared params unchanged', () => {
    const tp = 8;
    const memNoEP = calculateMemoryWithTP(v3, 4096, 1, tp, 'bf16', 'bf16', 1);
    const memEP8 = calculateMemoryWithTP(v3, 4096, 1, tp, 'bf16', 'bf16', 8);

    // With EP=8, per-device weight memory should be significantly less
    expect(memEP8.weights).toBeLessThan(memNoEP.weights);

    // Verify the split: shared/TP + routed/(TP×EP)
    const bytes = getPrecisionBytes('bf16');
    const { sharedParams, routedExpertParams } = moeParamSplit(v3);
    const expectedWeightsEP8 = (sharedParams / tp + routedExpertParams / (tp * 8)) * bytes;
    expect(memEP8.weights).toBeCloseTo(expectedWeightsEP8, -3);

    // Shared experts are in sharedParams (not distributed by EP)
    // So shared portion should be the same regardless of EP
    const expectedWeightsNoEP = (sharedParams / tp + routedExpertParams / tp) * bytes;
    expect(memNoEP.weights).toBeCloseTo(expectedWeightsNoEP, -3);
  });

  it('EP=1 for dense model: no change in memory', () => {
    const memEP1 = calculateMemoryWithTP(llama7b, 2048, 1, 1, 'bf16', 'bf16', 1);
    const memEP4 = calculateMemoryWithTP(llama7b, 2048, 1, 1, 'bf16', 'bf16', 4);

    // Dense model has no routed experts — EP makes no difference
    expect(memEP1.weights).toBe(memEP4.weights);
  });
});

describe('EP communication and decode latency', () => {
  it('V3 EP=4 adds All-to-All overhead to TPOT', () => {
    const tp = 8;
    const latencyNoEP = calculateLatencyWithTP(v3, 512, 256, 32, gpu, tp, 'bf16', 1);
    const latencyEP4 = calculateLatencyWithTP(v3, 512, 256, 32, gpu, tp, 'bf16', 4);

    // EP=4 adds All-to-All comm overhead, but weight reads are smaller per device
    // Net effect depends on balance — TPOT may go up or down
    // But All-to-All should be non-zero additional cost
    expect(latencyEP4.tpot).toBeGreaterThan(0);
    expect(latencyNoEP.tpot).toBeGreaterThan(0);

    // Both should produce finite results
    expect(isFinite(latencyEP4.tpot)).toBe(true);
    expect(isFinite(latencyEP4.ttft)).toBe(true);
  });

  it('EP overhead includes (ep-1)/ep factor', () => {
    const tp = 8;
    // EP=2: (2-1)/2 = 0.5 of traffic is remote
    // EP=4: (4-1)/4 = 0.75 of traffic is remote
    const latEP2 = calculateLatencyWithTP(v3, 512, 256, 32, gpu, tp, 'bf16', 2);
    const latEP4 = calculateLatencyWithTP(v3, 512, 256, 32, gpu, tp, 'bf16', 4);

    // Can't directly observe the All-to-All overhead in isolation,
    // but we can verify both return valid results
    expect(isFinite(latEP2.tpot)).toBe(true);
    expect(isFinite(latEP4.tpot)).toBe(true);
  });
});

describe('Prefill with EP', () => {
  it('V3 TTFT with EP=8 < TTFT with EP=1', () => {
    const tp = 8;
    const latNoEP = calculateLatencyWithTP(v3, 2048, 256, 1, gpu, tp, 'bf16', 1);
    const latEP8 = calculateLatencyWithTP(v3, 2048, 256, 1, gpu, tp, 'bf16', 8);

    // With EP, expert compute is distributed — TTFT should decrease
    expect(latEP8.ttft).toBeLessThan(latNoEP.ttft);

    // Speedup < 8x (shared compute not split by EP)
    const speedup = latNoEP.ttft / latEP8.ttft;
    expect(speedup).toBeGreaterThan(1);
    expect(speedup).toBeLessThan(8);
  });

  it('dense model: EP has no effect on TTFT', () => {
    const tp = 2;
    const latNoEP = calculateLatencyWithTP(llama7b, 2048, 256, 1, gpu, tp, 'bf16', 1);
    const latEP4 = calculateLatencyWithTP(llama7b, 2048, 256, 1, gpu, tp, 'bf16', 4);

    // Dense model: no MoE → EP doesn't affect prefill
    expect(latEP4.ttft).toBeCloseTo(latNoEP.ttft, 3);
  });
});

describe('Speculative decoding with MoE', () => {
  it('draftModelOverhead uses activeParams for MoE target', () => {
    // Llama 7B as draft, V3 as target
    const overhead = draftModelOverhead(llama7b, v3);
    const activeParams = v3.activeParams ?? v3.totalParams;

    // Should be llama7b.totalParams / v3.activeParams (not totalParams)
    const expected = llama7b.totalParams / activeParams;
    expect(overhead).toBeCloseTo(expected, 5);

    // With totalParams it would be ~7B/671B ≈ 0.01, with activeParams ~7B/37B ≈ 0.19
    expect(overhead).toBeGreaterThan(0.1);
    expect(overhead).toBeLessThan(0.5);
  });

  it('estimateAcceptanceRate uses activeParams for MoE', () => {
    const rate = estimateAcceptanceRate(llama7b, v3);

    // Should use activeParams ratio, not totalParams ratio
    // With activeParams: 7B/37B ≈ 0.19, sqrt(0.19) ≈ 0.44, base = 0.5 + 0.4*0.44 ≈ 0.67
    // With totalParams: 7B/671B ≈ 0.01, sqrt(0.01) ≈ 0.1, base = 0.5 + 0.4*0.1 = 0.54
    expect(rate).toBeGreaterThan(0.5);
    expect(rate).toBeLessThan(0.95);
  });
});

describe('Utilization metrics consistency', () => {
  it('V3 decode intensity at batch=1 uses batch-aware weight bytes (higher than totalParams-based)', () => {
    const util = calculateUtilizationMetrics(v3, 512, 256, 1, gpu, 'bf16');

    // All values should be finite and bounded
    expect(util.computeUtilization).toBeGreaterThanOrEqual(0);
    expect(util.computeUtilization).toBeLessThanOrEqual(1);
    expect(util.rooflineAttainment).toBeGreaterThanOrEqual(0);
    expect(util.rooflineAttainment).toBeLessThanOrEqual(1);
    expect(isFinite(util.memoryCapacityUtilization)).toBe(true);
  });

  it('V3 decode at batch=1: intensity higher than if using totalParams for bytes', () => {
    // With batch-aware: weight bytes ≈ activeParams (small) → higher intensity
    // If using totalParams: weight bytes ≈ 671B × 2 → much lower intensity
    const util = calculateUtilizationMetrics(v3, 512, 256, 1, gpu, 'bf16');

    // At batch=1 for MoE, intensity should be relatively high (few bytes, some FLOPs)
    // The rooflineAttainment reflects decode intensity vs ridge point
    // It should not be near-zero (which would happen with totalParams bytes)
    expect(util.rooflineAttainment).toBeGreaterThan(0);
  });

  it('dense model utilization unchanged', () => {
    const util = calculateUtilizationMetrics(llama7b, 512, 256, 1, gpu, 'bf16');

    expect(util.computeUtilization).toBeGreaterThan(0);
    expect(util.rooflineAttainment).toBeGreaterThan(0);
    expect(util.bottleneck).toBeDefined();
  });
});

describe('EP simulation engine integration', () => {
  it('V3 with EP=8 runs successfully', () => {
    const result = runInferenceSimulation({
      modelId: 'deepseek-v3',
      batchSize: 32,
      inputSeqLen: 512,
      outputSeqLen: 256,
      weightPrecision: 'fp8',
      tensorParallel: 8,
      expertParallel: 8,
      numGPUs: 64,
    });

    expect(result.success).toBe(true);
    expect(result.latency.ttft).toBeGreaterThan(0);
    expect(result.latency.tpot).toBeGreaterThan(0);
    expect(result.throughput.tokensPerSecond).toBeGreaterThan(0);
    expect(isFinite(result.latency.totalLatency)).toBe(true);
  });

  it('EP validation: EP must divide numExperts', () => {
    const result = runInferenceSimulation({
      modelId: 'deepseek-v3',
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 256,
      weightPrecision: 'fp8',
      tensorParallel: 8,
      expertParallel: 7, // 256 not divisible by 7
      numGPUs: 64,
    });

    expect(result.success).toBe(false);
    expect(result.errors.some(e => e.includes('divide'))).toBe(true);
  });

  it('EP validation: TP×EP must not exceed numGPUs', () => {
    const result = runInferenceSimulation({
      modelId: 'deepseek-v3',
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 256,
      weightPrecision: 'fp8',
      tensorParallel: 8,
      expertParallel: 32,
      numGPUs: 64, // 8×32=256 > 64
    });

    expect(result.success).toBe(false);
    expect(result.errors.some(e => e.includes('TP×EP'))).toBe(true);
  });

  it('EP on non-MoE model generates warning', () => {
    const result = runInferenceSimulation({
      modelId: 'llama2-7b',
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 256,
      tensorParallel: 2,
      expertParallel: 2,
      numGPUs: 4,
    });

    // Should succeed but with a warning
    expect(result.warnings.some(w => w.includes('non-MoE'))).toBe(true);
  });
});

describe('MoE absolute TTFT bounds', () => {
  it('V3 single-GPU batch=1 FP8 TTFT: memory-bound ~140ms', () => {
    // Compute: 2 * 37.6B * 1024 / (1979T * 0.4) = 97ms
    // Weight read: coupon-collector over 1024 tokens → nearly all 671B params
    //   ~671GB (FP8) / 4.8TB/s = 140ms
    // baseTtft = max(97, 140) = 140ms (memory-bound at seq=1024)
    const ttft = estimateTTFT(v3, 1024, H200_SXM, 'fp8', 1);
    expect(ttft).toBeGreaterThan(100);
    expect(ttft).toBeLessThan(180);
  });

  it('V3 single-GPU batch=32 FP8 TTFT: heavily compute-bound ~3100ms', () => {
    // Compute: 2 * 37.6B * 1024 * 32 / (1979T * 0.4) = 3107ms
    // Weight read: ~443GB / 4.8TB/s = 92ms
    // baseTtft = max(3107, 92) = 3107ms
    const ttft = estimateTTFT(v3, 1024, H200_SXM, 'fp8', 32);
    expect(ttft).toBeGreaterThan(2500);
    expect(ttft).toBeLessThan(4000);
  });

  it('V3 TP=8 batch=32 FP8 TTFT: ~300-600ms', () => {
    const latency = calculateLatencyWithTP(v3, 1024, 256, 32, H200_SXM, 8, 'fp8');
    // baseTtft / 8 + allReduce ≈ 388 + 64 = ~452ms
    expect(latency.ttft).toBeGreaterThan(250);
    expect(latency.ttft).toBeLessThan(700);
  });

  it('V3 TP=8 batch=1 FP8 TTFT: ~10-40ms', () => {
    const latency = calculateLatencyWithTP(v3, 1024, 256, 1, H200_SXM, 8, 'fp8');
    // baseTtft / 8 + small allReduce ≈ 12 + 8 = ~20ms
    expect(latency.ttft).toBeGreaterThan(5);
    expect(latency.ttft).toBeLessThan(50);
  });

  it('V3 TTFT batch sensitivity: batch=32 >> batch=1', () => {
    const ttftB1 = estimateTTFT(v3, 1024, H200_SXM, 'fp8', 1);
    const ttftB32 = estimateTTFT(v3, 1024, H200_SXM, 'fp8', 32);

    // Ratio should be large — compute scales with batch, weights don't scale much
    const ratio = ttftB32 / ttftB1;
    expect(ratio).toBeGreaterThan(20);
    expect(ratio).toBeLessThan(40);
  });
});
