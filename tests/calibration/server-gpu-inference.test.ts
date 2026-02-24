/**
 * Server GPU Inference Calibration Tests
 *
 * Validates the simulator against server-class GPU hardware for inference.
 * Complements consumer-gpu-inference.test.ts (RTX 4090/3090) with datacenter
 * GPUs: H100 SXM, H200 SXM, A100 80GB, B200.
 *
 * Each test pins the actual simulator output with tight tolerances to catch
 * regressions. Hardware ratio tests are the strongest — bandwidth ratios are
 * physical constants that cancel systematic modelling errors.
 *
 * Sim throughput is typically 2-10× lower than optimized runtimes (vLLM,
 * TRT-LLM) because the analytical model does not capture CUDA graphs, fused
 * kernels, or continuous batching. Published ranges are documented in comments
 * for reviewer context, not as pass/fail targets.
 *
 * H100 SXM: 80 GB, 3.35 TB/s memBW, 989 BF16 TFLOPS, 1979 FP8 TFLOPS
 * H200 SXM: 141 GB, 4.8 TB/s memBW, 989 BF16 TFLOPS, 1979 FP8 TFLOPS
 * A100 80GB: 80 GB, 2.039 TB/s memBW, 312 BF16 TFLOPS, no native FP8
 * B200 SXM: 180 GB, 7.7 TB/s memBW, 2250 BF16 TFLOPS, 4500 FP8 TFLOPS
 *
 * Expected bandwidth ratios (batch=1, BW-bound):
 *   H200/H100: 4.8 / 3.35  = 1.4328
 *   A100/H100: 2.039 / 3.35 = 0.6087
 *   B200/H100: 7.7 / 3.35  = 2.2985
 *
 * Sections:
 *   1. Server GPU Hardware Ratios (12 tests)
 *   2. Decode Throughput — H100 SXM (3 tests)
 *   3. Decode Throughput — A100 80GB (2 tests)
 *   4. Decode Throughput — H200 SXM (3 tests)
 *   5. FP8/BF16 Throughput Ratio (5 tests)
 *   6. Quantization Speedup Ratios on H100 (6 tests)
 *   7. Model Size Scaling (5 tests)
 *   8. Roofline Transition — Batch Scaling (5 tests)
 */

import { describe, it, expect } from 'vitest';
import { runInferenceSimulation } from '../../src/core/inference/simulation.ts';
import type { InferencePrecision } from '../../src/types/inference.ts';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Standard inference config: batch=1, 512 input + 256 output tokens */
function inf(modelId: string, gpuId: string, weightPrecision: InferencePrecision) {
  return runInferenceSimulation({
    modelId,
    gpuId,
    numGPUs: 1,
    batchSize: 1,
    inputSeqLen: 512,
    outputSeqLen: 256,
    weightPrecision,
    kvCachePrecision: 'bf16',
    tensorParallel: 1,
  });
}

/** Inference with tensor parallelism */
function infTP(
  modelId: string,
  gpuId: string,
  weightPrecision: InferencePrecision,
  numGPUs: number,
  tp: number
) {
  return runInferenceSimulation({
    modelId,
    gpuId,
    numGPUs,
    batchSize: 1,
    inputSeqLen: 512,
    outputSeqLen: 256,
    weightPrecision,
    kvCachePrecision: 'bf16',
    tensorParallel: tp,
  });
}

/** Inference with custom batch size */
function infBatch(
  modelId: string,
  gpuId: string,
  weightPrecision: InferencePrecision,
  batchSize: number
) {
  return runInferenceSimulation({
    modelId,
    gpuId,
    numGPUs: 1,
    batchSize,
    inputSeqLen: 512,
    outputSeqLen: 256,
    weightPrecision,
    kvCachePrecision: 'bf16',
    tensorParallel: 1,
  });
}

/** Assert value is within ±pct of expected */
function expectWithin(actual: number, expected: number, pct: number, label?: string) {
  const lo = expected * (1 - pct);
  const hi = expected * (1 + pct);
  expect(
    actual,
    `${label ?? ''} ${actual} not within ±${pct * 100}% of ${expected} [${lo.toFixed(4)}, ${hi.toFixed(4)}]`
  ).toBeGreaterThanOrEqual(lo);
  expect(
    actual,
    `${label ?? ''} ${actual} not within ±${pct * 100}% of ${expected} [${lo.toFixed(4)}, ${hi.toFixed(4)}]`
  ).toBeLessThanOrEqual(hi);
}

const TOL = 0.20; // ±20% for absolute throughput
const RATIO_TOL = 0.05; // ±5% for hardware ratios
const PREC_TOL = 0.10; // ±10% for precision speedup ratios

// ===========================================================================
// Section 1: Server GPU Hardware Ratios
//
// Strongest tests — bandwidth ratios are physical constants. At batch=1
// decode is memory-bandwidth-bound, so throughput ratio tracks memBW ratio.
// ===========================================================================

describe('Section 1: Server GPU Hardware Ratios', () => {
  // --- BF16 ratios ---
  it('H200/H100 BF16 ratio ≈ 1.43 (BW: 4.8/3.35 = 1.4328)', () => {
    const h200 = inf('llama2-7b', 'h200-sxm', 'bf16');
    const h100 = inf('llama2-7b', 'h100-sxm', 'bf16');
    const ratio = (1000 / h200.latency.tpot) / (1000 / h100.latency.tpot);
    expectWithin(ratio, 1.4296, RATIO_TOL, 'H200/H100 BF16');
  });

  it('A100/H100 BF16 ratio ≈ 0.61 (BW: 2.039/3.35 = 0.6087)', () => {
    const a100 = inf('llama2-7b', 'a100-80gb', 'bf16');
    const h100 = inf('llama2-7b', 'h100-sxm', 'bf16');
    const ratio = (1000 / a100.latency.tpot) / (1000 / h100.latency.tpot);
    expectWithin(ratio, 0.6099, RATIO_TOL, 'A100/H100 BF16');
  });

  it('B200/H100 BF16 ratio ≈ 2.28 (BW: 7.7/3.35 = 2.2985)', () => {
    const b200 = inf('llama2-7b', 'b200', 'bf16');
    const h100 = inf('llama2-7b', 'h100-sxm', 'bf16');
    const ratio = (1000 / b200.latency.tpot) / (1000 / h100.latency.tpot);
    expectWithin(ratio, 2.2832, RATIO_TOL, 'B200/H100 BF16');
  });

  // --- INT4 ratios — should be consistent with BF16 ---
  it('H200/H100 INT4 ratio ≈ 1.43 (±5%)', () => {
    const h200 = inf('llama2-7b', 'h200-sxm', 'int4');
    const h100 = inf('llama2-7b', 'h100-sxm', 'int4');
    const ratio = (1000 / h200.latency.tpot) / (1000 / h100.latency.tpot);
    expectWithin(ratio, 1.4251, RATIO_TOL, 'H200/H100 INT4');
  });

  it('A100/H100 INT4 ratio ≈ 0.61 (±5%)', () => {
    const a100 = inf('llama2-7b', 'a100-80gb', 'int4');
    const h100 = inf('llama2-7b', 'h100-sxm', 'int4');
    const ratio = (1000 / a100.latency.tpot) / (1000 / h100.latency.tpot);
    expectWithin(ratio, 0.6116, RATIO_TOL, 'A100/H100 INT4');
  });

  it('B200/H100 INT4 ratio ≈ 2.26 (±5%)', () => {
    const b200 = inf('llama2-7b', 'b200', 'int4');
    const h100 = inf('llama2-7b', 'h100-sxm', 'int4');
    const ratio = (1000 / b200.latency.tpot) / (1000 / h100.latency.tpot);
    expectWithin(ratio, 2.2618, RATIO_TOL, 'B200/H100 INT4');
  });

  // --- FP8 ratio ---
  it('H200/H100 FP8 ratio ≈ 1.43 (±5%)', () => {
    const h200 = inf('llama2-7b', 'h200-sxm', 'fp8');
    const h100 = inf('llama2-7b', 'h100-sxm', 'fp8');
    const ratio = (1000 / h200.latency.tpot) / (1000 / h100.latency.tpot);
    expectWithin(ratio, 1.4275, RATIO_TOL, 'H200/H100 FP8');
  });

  // --- Cross-precision consistency ---
  it('H200/H100 ratio stable across BF16/INT4/FP8 (±2%)', () => {
    const h200_bf16 = inf('llama2-7b', 'h200-sxm', 'bf16');
    const h100_bf16 = inf('llama2-7b', 'h100-sxm', 'bf16');
    const h200_int4 = inf('llama2-7b', 'h200-sxm', 'int4');
    const h100_int4 = inf('llama2-7b', 'h100-sxm', 'int4');
    const h200_fp8 = inf('llama2-7b', 'h200-sxm', 'fp8');
    const h100_fp8 = inf('llama2-7b', 'h100-sxm', 'fp8');

    const rBf16 = (1000 / h200_bf16.latency.tpot) / (1000 / h100_bf16.latency.tpot);
    const rInt4 = (1000 / h200_int4.latency.tpot) / (1000 / h100_int4.latency.tpot);
    const rFp8 = (1000 / h200_fp8.latency.tpot) / (1000 / h100_fp8.latency.tpot);

    // All three ratios within 2% of each other
    expectWithin(rInt4, rBf16, 0.02, 'INT4 vs BF16 ratio consistency');
    expectWithin(rFp8, rBf16, 0.02, 'FP8 vs BF16 ratio consistency');
  });

  // --- Strict throughput ordering ---
  it('Throughput ordering: B200 > H200 > H100 > A100', () => {
    const b200 = 1000 / inf('llama2-7b', 'b200', 'bf16').latency.tpot;
    const h200 = 1000 / inf('llama2-7b', 'h200-sxm', 'bf16').latency.tpot;
    const h100 = 1000 / inf('llama2-7b', 'h100-sxm', 'bf16').latency.tpot;
    const a100 = 1000 / inf('llama2-7b', 'a100-80gb', 'bf16').latency.tpot;

    expect(b200).toBeGreaterThan(h200);
    expect(h200).toBeGreaterThan(h100);
    expect(h100).toBeGreaterThan(a100);
  });

  // --- Each ratio within 5% of theoretical BW ratio ---
  it('All ratios track theoretical BW ratios within 5%', () => {
    const h100 = 1000 / inf('llama2-7b', 'h100-sxm', 'bf16').latency.tpot;
    const h200 = 1000 / inf('llama2-7b', 'h200-sxm', 'bf16').latency.tpot;
    const a100 = 1000 / inf('llama2-7b', 'a100-80gb', 'bf16').latency.tpot;
    const b200 = 1000 / inf('llama2-7b', 'b200', 'bf16').latency.tpot;

    // Theoretical BW ratios
    expectWithin(h200 / h100, 4.8 / 3.35, RATIO_TOL, 'H200/H100 vs theory');
    expectWithin(a100 / h100, 2.039 / 3.35, RATIO_TOL, 'A100/H100 vs theory');
    expectWithin(b200 / h100, 7.7 / 3.35, RATIO_TOL, 'B200/H100 vs theory');
  });
});

// ===========================================================================
// Section 2: Decode Throughput — H100 SXM
//
// Batch=1 decode tok/s pinned to simulator output ±20%.
// Published vLLM/TRT-LLM ranges in comments for context.
// ===========================================================================

describe('Section 2: Decode Throughput — H100 SXM', () => {
  // Published vLLM: 350-400 tok/s. Sim is analytical, no CUDA graphs.
  it('LLaMA 2 7B BF16: ~172 tok/s', () => {
    const r = inf('llama2-7b', 'h100-sxm', 'bf16');
    expect(r.success).toBe(true);
    const tokps = 1000 / r.latency.tpot;
    expectWithin(tokps, 172.4, TOL, 'H100 LLaMA 2 7B BF16');
  });

  // Published vLLM: 300-350 tok/s.
  it('LLaMA 3 8B BF16: ~151 tok/s', () => {
    const r = inf('llama3-8b', 'h100-sxm', 'bf16');
    expect(r.success).toBe(true);
    const tokps = 1000 / r.latency.tpot;
    expectWithin(tokps, 151.1, TOL, 'H100 LLaMA 3 8B BF16');
  });

  // Published vLLM TP=8: ~900-1200 tok/s. Sim models TP comm overhead.
  it('LLaMA 3 70B BF16 TP=8: ~158 tok/s', () => {
    const r = infTP('llama3-70b', 'h100-sxm', 'bf16', 8, 8);
    expect(r.success).toBe(true);
    const tokps = 1000 / r.latency.tpot;
    expectWithin(tokps, 157.7, TOL, 'H100 LLaMA 3 70B BF16 TP=8');
  });
});

// ===========================================================================
// Section 3: Decode Throughput — A100 80GB
//
// Fills the A100 inference coverage gap.
// ===========================================================================

describe('Section 3: Decode Throughput — A100 80GB', () => {
  // Published vLLM: 105-125 tok/s.
  it('LLaMA 2 7B BF16: ~105 tok/s', () => {
    const r = inf('llama2-7b', 'a100-80gb', 'bf16');
    expect(r.success).toBe(true);
    const tokps = 1000 / r.latency.tpot;
    expectWithin(tokps, 105.2, TOL, 'A100 LLaMA 2 7B BF16');
  });

  // Published: ~90-110 tok/s.
  it('LLaMA 3 8B BF16: ~92 tok/s', () => {
    const r = inf('llama3-8b', 'a100-80gb', 'bf16');
    expect(r.success).toBe(true);
    const tokps = 1000 / r.latency.tpot;
    expectWithin(tokps, 92.1, TOL, 'A100 LLaMA 3 8B BF16');
  });
});

// ===========================================================================
// Section 4: Decode Throughput — H200 SXM
//
// Fills the H200 inference coverage gap.
// ===========================================================================

describe('Section 4: Decode Throughput — H200 SXM', () => {
  it('LLaMA 2 7B BF16: ~247 tok/s', () => {
    const r = inf('llama2-7b', 'h200-sxm', 'bf16');
    expect(r.success).toBe(true);
    const tokps = 1000 / r.latency.tpot;
    expectWithin(tokps, 246.5, TOL, 'H200 LLaMA 2 7B BF16');
  });

  it('LLaMA 3 8B BF16: ~216 tok/s', () => {
    const r = inf('llama3-8b', 'h200-sxm', 'bf16');
    expect(r.success).toBe(true);
    const tokps = 1000 / r.latency.tpot;
    expectWithin(tokps, 216.0, TOL, 'H200 LLaMA 3 8B BF16');
  });

  // 70B FP8 ≈ 70.6 GB weights — fits on H200's 141 GB
  it('LLaMA 3 70B FP8 TP=1: fits on 141 GB, ~53 tok/s', () => {
    const r = inf('llama3-70b', 'h200-sxm', 'fp8');
    expect(r.success).toBe(true);
    expect(r.memory.total / 1e9).toBeLessThan(141);
    const tokps = 1000 / r.latency.tpot;
    expectWithin(tokps, 52.7, TOL, 'H200 LLaMA 3 70B FP8');
  });
});

// ===========================================================================
// Section 5: FP8/BF16 Throughput Ratio
//
// FP8 halves weight bytes, but throughput ratio is <2× because:
//   - getBandwidthEfficiency() drops for smaller weight volumes
//   - Dequant overhead: 1.05× (Transformer Engine) or 1.10× (software)
// A100 lacks native FP8, so uses software dequant (1.10×).
// ===========================================================================

describe('Section 5: FP8/BF16 Throughput Ratio', () => {
  it('H100 LLaMA 2 7B FP8/BF16 ≈ 1.66 (±10%)', () => {
    const bf16 = inf('llama2-7b', 'h100-sxm', 'bf16');
    const fp8 = inf('llama2-7b', 'h100-sxm', 'fp8');
    const ratio = (1000 / fp8.latency.tpot) / (1000 / bf16.latency.tpot);
    expectWithin(ratio, 1.656, PREC_TOL, 'H100 7B FP8/BF16');
  });

  it('H100 LLaMA 3 8B FP8/BF16 ≈ 1.70 (±10%)', () => {
    const bf16 = inf('llama3-8b', 'h100-sxm', 'bf16');
    const fp8 = inf('llama3-8b', 'h100-sxm', 'fp8');
    const ratio = (1000 / fp8.latency.tpot) / (1000 / bf16.latency.tpot);
    expectWithin(ratio, 1.701, PREC_TOL, 'H100 8B FP8/BF16');
  });

  it('H200 LLaMA 2 7B FP8/BF16 ≈ 1.65 (±10%)', () => {
    const bf16 = inf('llama2-7b', 'h200-sxm', 'bf16');
    const fp8 = inf('llama2-7b', 'h200-sxm', 'fp8');
    const ratio = (1000 / fp8.latency.tpot) / (1000 / bf16.latency.tpot);
    expectWithin(ratio, 1.653, PREC_TOL, 'H200 7B FP8/BF16');
  });

  it('FP8/BF16 ratio consistent across H100/H200 (±5%)', () => {
    const h100_bf16 = inf('llama2-7b', 'h100-sxm', 'bf16');
    const h100_fp8 = inf('llama2-7b', 'h100-sxm', 'fp8');
    const h200_bf16 = inf('llama2-7b', 'h200-sxm', 'bf16');
    const h200_fp8 = inf('llama2-7b', 'h200-sxm', 'fp8');

    const h100Ratio = (1000 / h100_fp8.latency.tpot) / (1000 / h100_bf16.latency.tpot);
    const h200Ratio = (1000 / h200_fp8.latency.tpot) / (1000 / h200_bf16.latency.tpot);
    expectWithin(h200Ratio, h100Ratio, RATIO_TOL, 'H200 vs H100 FP8/BF16 consistency');
  });

  // A100 lacks Transformer Engine — uses software dequant (1.10× vs 1.05×)
  it('A100 FP8/BF16 ratio < H100 ratio (no Transformer Engine)', () => {
    const a100_bf16 = inf('llama2-7b', 'a100-80gb', 'bf16');
    const a100_fp8 = inf('llama2-7b', 'a100-80gb', 'fp8');
    const h100_bf16 = inf('llama2-7b', 'h100-sxm', 'bf16');
    const h100_fp8 = inf('llama2-7b', 'h100-sxm', 'fp8');

    const a100Ratio = (1000 / a100_fp8.latency.tpot) / (1000 / a100_bf16.latency.tpot);
    const h100Ratio = (1000 / h100_fp8.latency.tpot) / (1000 / h100_bf16.latency.tpot);
    expect(a100Ratio).toBeLessThan(h100Ratio);
  });
});

// ===========================================================================
// Section 6: Quantization Speedup Ratios on H100
//
// INT4 halves vs FP8 (0.5 vs 1.0 bytes/param), INT8 = FP8 bytes but
// different dequant path. Expected ordering: INT4 > FP8 ≥ INT8 > BF16.
// ===========================================================================

describe('Section 6: Quantization Speedup Ratios on H100', () => {
  it('LLaMA 2 7B INT4/BF16 ≈ 2.42 (±10%)', () => {
    const bf16 = inf('llama2-7b', 'h100-sxm', 'bf16');
    const int4 = inf('llama2-7b', 'h100-sxm', 'int4');
    const ratio = (1000 / int4.latency.tpot) / (1000 / bf16.latency.tpot);
    expectWithin(ratio, 2.415, PREC_TOL, 'H100 7B INT4/BF16');
  });

  it('LLaMA 2 7B INT8/BF16 ≈ 1.58 (±10%)', () => {
    const bf16 = inf('llama2-7b', 'h100-sxm', 'bf16');
    const int8 = inf('llama2-7b', 'h100-sxm', 'int8');
    const ratio = (1000 / int8.latency.tpot) / (1000 / bf16.latency.tpot);
    expectWithin(ratio, 1.584, PREC_TOL, 'H100 7B INT8/BF16');
  });

  it('LLaMA 2 7B FP8/BF16 ≈ 1.66 (±10%)', () => {
    const bf16 = inf('llama2-7b', 'h100-sxm', 'bf16');
    const fp8 = inf('llama2-7b', 'h100-sxm', 'fp8');
    const ratio = (1000 / fp8.latency.tpot) / (1000 / bf16.latency.tpot);
    expectWithin(ratio, 1.656, PREC_TOL, 'H100 7B FP8/BF16');
  });

  it('Ordering: INT4 > FP8 ≥ INT8 > BF16 (tok/s)', () => {
    const bf16 = 1000 / inf('llama2-7b', 'h100-sxm', 'bf16').latency.tpot;
    const int8 = 1000 / inf('llama2-7b', 'h100-sxm', 'int8').latency.tpot;
    const fp8 = 1000 / inf('llama2-7b', 'h100-sxm', 'fp8').latency.tpot;
    const int4 = 1000 / inf('llama2-7b', 'h100-sxm', 'int4').latency.tpot;

    expect(int4).toBeGreaterThan(fp8);
    expect(fp8).toBeGreaterThanOrEqual(int8);
    expect(int8).toBeGreaterThan(bf16);
  });

  // Cross-model consistency
  it('LLaMA 3 8B INT4/BF16 ≈ 2.56 (±10%)', () => {
    const bf16 = inf('llama3-8b', 'h100-sxm', 'bf16');
    const int4 = inf('llama3-8b', 'h100-sxm', 'int4');
    const ratio = (1000 / int4.latency.tpot) / (1000 / bf16.latency.tpot);
    expectWithin(ratio, 2.561, PREC_TOL, 'H100 8B INT4/BF16');
  });

  it('LLaMA 3 8B FP8/BF16 ≈ 1.70 (±10%)', () => {
    const bf16 = inf('llama3-8b', 'h100-sxm', 'bf16');
    const fp8 = inf('llama3-8b', 'h100-sxm', 'fp8');
    const ratio = (1000 / fp8.latency.tpot) / (1000 / bf16.latency.tpot);
    expectWithin(ratio, 1.701, PREC_TOL, 'H100 8B FP8/BF16');
  });
});

// ===========================================================================
// Section 7: Model Size Scaling
//
// Decode throughput scales roughly inversely with model weight size
// (bandwidth-bound), but not perfectly: getBandwidthEfficiency() improves
// for larger transfer sizes, compressing the ratio.
// ===========================================================================

describe('Section 7: Model Size Scaling', () => {
  it('7B/13B throughput ratio on H100 INT4 ≈ 1.65 (±10%)', () => {
    const r7b = inf('llama2-7b', 'h100-sxm', 'int4');
    const r13b = inf('llama2-13b', 'h100-sxm', 'int4');
    const ratio = (1000 / r7b.latency.tpot) / (1000 / r13b.latency.tpot);
    // Weight ratio 13B/7B ≈ 1.9×, throughput ratio ≈ 1.65× (BW eff offset)
    expectWithin(ratio, 1.650, PREC_TOL, '7B/13B INT4 ratio');
  });

  it('7B/70B throughput ratio on H100 INT4 ≈ 6.73 (±10%)', () => {
    const r7b = inf('llama2-7b', 'h100-sxm', 'int4');
    const r70b = inf('llama3-70b', 'h100-sxm', 'int4');
    expect(r70b.success).toBe(true);
    const ratio = (1000 / r7b.latency.tpot) / (1000 / r70b.latency.tpot);
    // Weight ratio 70B/7B ≈ 10.2×, throughput ratio ≈ 6.73× (BW eff offset)
    expectWithin(ratio, 6.725, PREC_TOL, '7B/70B INT4 ratio');
  });

  it('Throughput monotonically decreasing: 7B > 13B > 70B', () => {
    const t7b = 1000 / inf('llama2-7b', 'h100-sxm', 'int4').latency.tpot;
    const t13b = 1000 / inf('llama2-13b', 'h100-sxm', 'int4').latency.tpot;
    const t70b = 1000 / inf('llama3-70b', 'h100-sxm', 'int4').latency.tpot;

    expect(t7b).toBeGreaterThan(t13b);
    expect(t13b).toBeGreaterThan(t70b);
  });

  it('7B/8B BF16 ratio on H100 ≈ 1.14 (±10%)', () => {
    const r7b = inf('llama2-7b', 'h100-sxm', 'bf16');
    const r8b = inf('llama3-8b', 'h100-sxm', 'bf16');
    const ratio = (1000 / r7b.latency.tpot) / (1000 / r8b.latency.tpot);
    expectWithin(ratio, 1.142, PREC_TOL, '7B/8B BF16 ratio');
  });

  it('Throughput ratio < weight ratio (BW efficiency effect)', () => {
    const r7b = inf('llama2-7b', 'h100-sxm', 'int4');
    const r70b = inf('llama3-70b', 'h100-sxm', 'int4');
    expect(r70b.success).toBe(true);

    const throughputRatio = (1000 / r7b.latency.tpot) / (1000 / r70b.latency.tpot);
    // Weight ratio: 70B has ~10.2× more params than 7B
    const weightRatio = 70.55 / 6.738; // approximate total params in billions
    expect(throughputRatio).toBeLessThan(weightRatio);
  });
});

// ===========================================================================
// Section 8: Roofline Transition — Batch Scaling
//
// At batch=1, decode is memory-bandwidth-bound: H200 beats H100 by ~1.43×
// (BW ratio). As batch increases, decode moves toward compute-bound: H200
// and H100 have identical 989 BF16 TFLOPS, so the ratio trends toward 1.0.
//
// Arithmetic intensity = 2 × activeParams × batch / weightBytes.
// At batch=1 this is ~2 (well below ridge point) → BW-bound.
// At batch=128+ this is ~256 (above ridge) → transitioning to compute-bound.
// ===========================================================================

describe('Section 8: Roofline Transition — Batch Scaling', () => {
  it('H200/H100 ratio at batch=1 ≈ 1.42 (BW-bound, tracks bandwidth ratio)', () => {
    const h200 = infBatch('llama3-8b', 'h200-sxm', 'bf16', 1);
    const h100 = infBatch('llama3-8b', 'h100-sxm', 'bf16', 1);
    const ratio = h200.throughput.tokensPerSecond / h100.throughput.tokensPerSecond;
    expectWithin(ratio, 1.4237, RATIO_TOL, 'H200/H100 batch=1');
  });

  it('H200/H100 ratio at batch=128 ≈ 1.21 (transitioning toward compute-bound)', () => {
    const h200 = infBatch('llama3-8b', 'h200-sxm', 'bf16', 128);
    const h100 = infBatch('llama3-8b', 'h100-sxm', 'bf16', 128);
    expect(h200.success).toBe(true);
    expect(h100.success).toBe(true);
    const ratio = h200.throughput.tokensPerSecond / h100.throughput.tokensPerSecond;
    expectWithin(ratio, 1.2147, RATIO_TOL, 'H200/H100 batch=128');
  });

  it('High-batch ratio significantly closer to 1.0 than low-batch ratio', () => {
    const h200_b1 = infBatch('llama3-8b', 'h200-sxm', 'bf16', 1);
    const h100_b1 = infBatch('llama3-8b', 'h100-sxm', 'bf16', 1);
    const h200_b128 = infBatch('llama3-8b', 'h200-sxm', 'bf16', 128);
    const h100_b128 = infBatch('llama3-8b', 'h100-sxm', 'bf16', 128);

    const lowBatchRatio = h200_b1.throughput.tokensPerSecond / h100_b1.throughput.tokensPerSecond;
    const highBatchRatio = h200_b128.throughput.tokensPerSecond / h100_b128.throughput.tokensPerSecond;

    // High-batch ratio closer to 1.0 than low-batch ratio
    expect(Math.abs(highBatchRatio - 1.0)).toBeLessThan(Math.abs(lowBatchRatio - 1.0));
    // At least 0.15 closer (1.42 - 1.21 ≈ 0.21)
    expect(lowBatchRatio - highBatchRatio).toBeGreaterThan(0.15);
  });

  it('A100/H100 ratio at batch=128 moves toward TFLOPS ratio (312/989 ≈ 0.316)', () => {
    const a100_b1 = infBatch('llama3-8b', 'a100-80gb', 'bf16', 1);
    const h100_b1 = infBatch('llama3-8b', 'h100-sxm', 'bf16', 1);
    const a100_b128 = infBatch('llama3-8b', 'a100-80gb', 'bf16', 128);
    const h100_b128 = infBatch('llama3-8b', 'h100-sxm', 'bf16', 128);

    const lowBatchRatio = a100_b1.throughput.tokensPerSecond / h100_b1.throughput.tokensPerSecond;
    const highBatchRatio = a100_b128.throughput.tokensPerSecond / h100_b128.throughput.tokensPerSecond;

    // Low batch ≈ 0.60 (BW ratio), high batch ≈ 0.44 (moving toward 0.316 TFLOPS ratio)
    expect(highBatchRatio).toBeLessThan(lowBatchRatio);
    // High batch should be between TFLOPS ratio and BW ratio
    expect(highBatchRatio).toBeGreaterThan(312 / 989); // above pure TFLOPS ratio
    expect(highBatchRatio).toBeLessThan(2.039 / 3.35); // below pure BW ratio
  });

  it('Batch=128 decode is compute-bound on H100', () => {
    const r = infBatch('llama3-8b', 'h100-sxm', 'bf16', 128);
    expect(r.success).toBe(true);
    expect(r.utilization.isComputeBound).toBe(true);
  });
});
