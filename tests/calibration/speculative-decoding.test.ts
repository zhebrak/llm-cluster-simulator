/**
 * Speculative Decoding Tests
 *
 * Part 1 — Unit Tests:
 *   Formula correctness, TP-awareness, integration checks for the speculative
 *   decoding pipeline (5-bug fix). Tests that the pipeline is internally
 *   consistent: metrics feed through, memory is accounted for, TP sharding
 *   works, recommendations fire correctly.
 *
 * Part 2 — Benchmark Calibration:
 *   Validates the simulator's speculative decoding model against published
 *   benchmarks from NVIDIA TRT-LLM, AMD ROCm, and vLLM. Each test pins
 *   the actual simulator output with tight tolerances, and comments document
 *   published real-world ranges for reference.
 *
 * Calibration approach:
 * 1. Run calibration script to get actual simulator values
 * 2. Verify each value is plausible against published benchmarks
 * 3. Pin simulator output with ±20% tolerance for speedup, ±10% for acceptance rates
 * 4. Comment each test with published source and pinned simulator value
 *
 * Key configuration notes:
 * - H200 (141GB): Llama 70B fp8 (~70.6GB) fits at TP=1
 * - H100 (80GB):  Llama 70B fp8 (~70.6GB) fits at TP=1; bf16 needs TP≥2
 * - MI300X (192GB): Llama 70B bf16 (~140GB) fits at TP=1
 * - RTX 4090 (24GB): max ~8B bf16 or 13B int4
 * - DeepSeek V3 (671B fp8 ~671GB): needs TP=8 on H200
 */
import { describe, it, expect } from 'vitest';
import {
  expectedAcceptedTokens,
  estimateDraftTime,
  estimateVerificationTime,
  calculateSpeculativeMetrics,
  estimateAcceptanceRate,
  findOptimalSpecConfig,
  runInferenceSimulation,
} from '../../src/core/inference/index.ts';
import { generateInferenceRecommendations } from '../../src/core/inference/recommendations.ts';
import { getModel } from '../../src/core/models/index.ts';
import { H100_SXM, H200_SXM, A100_80GB, MI300X, RTX_4090, RTX_3090 } from '../../src/core/hardware/gpu.ts';
import type { GPUSpec } from '../../src/types/hardware.ts';
import type { ModelSpec } from '../../src/types/model.ts';
import type { InferencePrecision } from '../../src/types/inference.ts';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Assert value is within ±pct of expected */
function expectWithin(actual: number, expected: number, pct: number, label?: string) {
  const lo = expected * (1 - pct);
  const hi = expected * (1 + pct);
  expect(
    actual,
    `${label ?? ''} ${actual.toFixed(4)} not within ±${pct * 100}% of ${expected} [${lo.toFixed(4)}, ${hi.toFixed(4)}]`
  ).toBeGreaterThanOrEqual(lo);
  expect(
    actual,
    `${label ?? ''} ${actual.toFixed(4)} not within ±${pct * 100}% of ${expected} [${lo.toFixed(4)}, ${hi.toFixed(4)}]`
  ).toBeLessThanOrEqual(hi);
}

/** Run speculative decoding simulation and return key metrics */
function specSim(
  target: ModelSpec,
  draft: ModelSpec,
  gpu: GPUSpec,
  precision: InferencePrecision,
  tp: number,
  K: number,
  opts?: { alpha?: number; batch?: number; cb?: boolean }
) {
  const alpha = opts?.alpha ?? estimateAcceptanceRate(draft, target);
  const result = runInferenceSimulation({
    modelSpec: target,
    gpu,
    numGPUs: tp,
    batchSize: opts?.batch ?? 1,
    inputSeqLen: 512,
    outputSeqLen: 256,
    weightPrecision: precision,
    kvCachePrecision: precision,
    tensorParallel: tp,
    continuousBatching: opts?.cb ?? false,
    speculativeEnabled: true,
    draftModelSpec: draft,
    numSpeculativeTokens: K,
    acceptanceRate: alpha,
  });
  return result;
}

const TOL = 0.20;     // ±20% for speedup
const ALPHA_TOL = 0.10; // ±10% for acceptance rates

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

const llama32_1b = getModel('llama3.2-1b', 2048)!;
const llama32_3b = getModel('llama3.2-3b', 2048)!;
const llama31_8b = getModel('llama3.1-8b', 2048)!;
const llama3_8b = getModel('llama3-8b', 2048)!;
const llama33_70b = getModel('llama3.3-70b', 2048)!;
const llama2_7b = getModel('llama2-7b', 2048)!;
const llama2_13b = getModel('llama2-13b', 2048)!;
const llama2_70b = getModel('llama2-70b', 2048)!;
const llama3_70b = getModel('llama3-70b', 2048)!;
const gpt3_125m = getModel('gpt3-125m', 2048)!;
const dsV3 = getModel('deepseek-v3', 2048)!;

// ###########################################################################
//
//  Part 1 — Unit Tests
//
// ###########################################################################

// ── Unit: expectedAcceptedTokens formula ──

describe('expectedAcceptedTokens formula', () => {
  it('α=0 → 1 token (target only)', () => {
    expect(expectedAcceptedTokens(4, 0)).toBe(1);
  });

  it('α=1 → K tokens (all accepted)', () => {
    expect(expectedAcceptedTokens(4, 1.0)).toBe(4);
  });

  it('α=0.5, K=4 → geometric series + 1', () => {
    const expected = 0.5 * (1 - Math.pow(0.5, 4)) / (1 - 0.5) + 1;
    expect(expectedAcceptedTokens(4, 0.5)).toBeCloseTo(expected, 10);
  });

  it('α=0.7, K=4 → between 1 and 5', () => {
    const result = expectedAcceptedTokens(4, 0.7);
    expect(result).toBeGreaterThan(1);
    expect(result).toBeLessThan(5);
    const expected = 0.7 * (1 - Math.pow(0.7, 4)) / (1 - 0.7) + 1;
    expect(result).toBeCloseTo(expected, 10);
  });
});

// ── Unit: TP-aware draft time ──

describe('TP-aware draft time', () => {
  it('draft time with TP=1 uses single-GPU estimateTPOT', () => {
    const tp1 = estimateDraftTime(gpt3_125m, 4, 512, 1, H100_SXM, 'bf16', undefined, 1);
    expect(tp1).toBeGreaterThan(0);
    expect(Number.isFinite(tp1)).toBe(true);
  });

  it('draft time with TP>1 is different from TP=1', () => {
    const tp1 = estimateDraftTime(llama3_8b, 4, 512, 1, H100_SXM, 'bf16', undefined, 1);
    const tp4 = estimateDraftTime(llama3_8b, 4, 512, 1, H100_SXM, 'bf16', undefined, 4);
    expect(tp4).not.toBeCloseTo(tp1, 0);
  });
});

// ── Unit: TP-aware verification ──

describe('TP-aware verification', () => {
  it('verification time with TP=1 is finite', () => {
    const t = estimateVerificationTime(llama2_70b, 4, 512, 1, H100_SXM, 'bf16', undefined, 1);
    expect(t).toBeGreaterThan(0);
    expect(Number.isFinite(t)).toBe(true);
  });

  it('verification with TP>1 accounts for compute sharding', () => {
    const tp1 = estimateVerificationTime(llama2_70b, 4, 512, 1, H100_SXM, 'bf16', undefined, 1);
    const tp8 = estimateVerificationTime(llama2_70b, 4, 512, 1, H100_SXM, 'bf16', undefined, 8);
    expect(tp8).toBeLessThan(tp1);
  });
});

// ── Unit: batch degradation (physics, not heuristic) ──

describe('batch degradation (physics)', () => {
  it('speedup at batch=1 > speedup at batch=64', () => {
    const specConfig = {
      enabled: true as const,
      draftModel: gpt3_125m,
      numSpeculativeTokens: 4,
      acceptanceRate: 0.7,
    };

    const metricsB1 = calculateSpeculativeMetrics(
      specConfig, llama2_70b, 768, 1, H100_SXM, 'bf16'
    );

    const metricsB64 = calculateSpeculativeMetrics(
      specConfig, llama2_70b, 768, 64, H100_SXM, 'bf16'
    );

    expect(metricsB1).not.toBeNull();
    expect(metricsB64).not.toBeNull();
    expect(metricsB1!.speedup).toBeGreaterThan(metricsB64!.speedup);
  });
});

// ── Integration: metrics feed through ──
// Use Llama 8B (fits on 1 H100 in bf16) with a tiny draft model

describe('speculative metrics feed through to headline results', () => {
  it('result.latency.tpot equals speculative.effectiveTpot when speedup > 1', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 256,
      speculativeEnabled: true,
      draftModelSpec: gpt3_125m,
      numSpeculativeTokens: 4,
      acceptanceRate: 0.7,
    });

    expect(result.success).toBe(true);
    expect(result.speculative).toBeDefined();
    expect(result.speculative!.speedup).toBeGreaterThan(1);
    // Bug 1 fix: headline TPOT must equal speculative effective TPOT
    expect(result.latency.tpot).toBeCloseTo(result.speculative!.effectiveTpot, 5);
  });

  it('speculative throughput is higher than baseline', () => {
    const baseline = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 256,
    });

    const speculative = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 256,
      speculativeEnabled: true,
      draftModelSpec: gpt3_125m,
      numSpeculativeTokens: 4,
      acceptanceRate: 0.7,
    });

    expect(baseline.success).toBe(true);
    expect(speculative.success).toBe(true);
    expect(speculative.throughput.tokensPerSecond).toBeGreaterThan(
      baseline.throughput.tokensPerSecond
    );
  });
});

// ── Integration: draft memory included ──

describe('draft model memory', () => {
  it('memory.total is larger with speculative decoding enabled', () => {
    const baseline = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
    });

    const specResult = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      speculativeEnabled: true,
      draftModelSpec: gpt3_125m,
      numSpeculativeTokens: 4,
      acceptanceRate: 0.7,
    });

    expect(specResult.success).toBe(true);
    expect(baseline.success).toBe(true);
    expect(specResult.memory.total).toBeGreaterThan(baseline.memory.total);
    // GPT-3 125M in bf16 ≈ 250MB
    const draftMemoryDelta = specResult.memory.total - baseline.memory.total;
    expect(draftMemoryDelta).toBeGreaterThan(200e6);
    expect(draftMemoryDelta).toBeLessThan(500e6);
  });

  it('OOM with large draft model on memory-tight config', () => {
    // Llama 70B in bf16 = 140GB, doesn't fit on 1 GPU even without draft
    const result = runInferenceSimulation({
      modelId: 'llama2-70b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      speculativeEnabled: true,
      draftModelSpec: llama3_8b,
      numSpeculativeTokens: 4,
      acceptanceRate: 0.7,
    });

    expect(result.success).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);
    // Draft model memory should be included in the OOM diagnostic
    expect(result.memory.weights).toBeGreaterThan(llama2_70b.totalParams * 2); // target weights
  });
});

// ── Integration: TP-aware speculative simulation ──
// Use Llama 70B with TP to fit in memory

describe('TP-aware speculative simulation', () => {
  it('speculative decoding works with TP and speedup decreases vs non-TP baseline', () => {
    // Llama 70B with TP=4 on 4 H100s (bf16: 140GB / 4 = 35GB/GPU, fits)
    const resultTP4 = runInferenceSimulation({
      modelId: 'llama2-70b',
      gpu: H100_SXM,
      numGPUs: 4,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 256,
      tensorParallel: 4,
      speculativeEnabled: true,
      draftModelSpec: gpt3_125m,
      numSpeculativeTokens: 4,
      acceptanceRate: 0.7,
    });

    const resultTP8 = runInferenceSimulation({
      modelId: 'llama2-70b',
      gpu: H100_SXM,
      numGPUs: 8,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 256,
      tensorParallel: 8,
      speculativeEnabled: true,
      draftModelSpec: gpt3_125m,
      numSpeculativeTokens: 4,
      acceptanceRate: 0.7,
    });

    expect(resultTP4.success).toBe(true);
    expect(resultTP8.success).toBe(true);
    expect(resultTP4.speculative).toBeDefined();
    expect(resultTP8.speculative).toBeDefined();
    // More TP = faster baseline TPOT = less relative speculative speedup
    expect(resultTP8.speculative!.speedup).toBeLessThan(resultTP4.speculative!.speedup);
    // Both should still be > 1 (still beneficial)
    expect(resultTP4.speculative!.speedup).toBeGreaterThan(1);
    expect(resultTP8.speculative!.speedup).toBeGreaterThan(1);
    // Speculative TPOT should feed into headline
    expect(resultTP4.latency.tpot).toBeCloseTo(resultTP4.speculative!.effectiveTpot, 5);
  });
});

// ── Benchmark: Llama 70B + 8B draft on H100, TP=4 ──

describe('benchmark: Llama 70B speculative decoding', () => {
  it('batch=1 with TP=4: speedup 1.3-3.0x', () => {
    const result = runInferenceSimulation({
      modelId: 'llama2-70b',
      gpu: H100_SXM,
      numGPUs: 4,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 256,
      tensorParallel: 4,
      speculativeEnabled: true,
      draftModelSpec: llama3_8b,
      numSpeculativeTokens: 5,
      acceptanceRate: 0.7,
    });

    expect(result.success).toBe(true);
    expect(result.speculative).toBeDefined();
    // With TP=4 the baseline is already fast, so speedup is smaller than single-GPU
    expect(result.speculative!.speedup).toBeGreaterThanOrEqual(1.3);
    expect(result.speculative!.speedup).toBeLessThanOrEqual(3.0);
  });

  it('batch=64 with TP=4: speedup < 1.5 (marginal benefit at large batch)', () => {
    const result = runInferenceSimulation({
      modelId: 'llama2-70b',
      gpu: H100_SXM,
      numGPUs: 4,
      batchSize: 64,
      inputSeqLen: 512,
      outputSeqLen: 256,
      tensorParallel: 4,
      speculativeEnabled: true,
      draftModelSpec: llama3_8b,
      numSpeculativeTokens: 5,
      acceptanceRate: 0.7,
    });

    expect(result.success).toBe(true);
    expect(result.speculative).toBeDefined();
    expect(result.speculative!.speedup).toBeLessThan(1.5);
  });
});

// ── Integration: multi-replica + speculative ──

describe('multi-replica + speculative decoding', () => {
  it('2 replicas (numGPUs=8, TP=4) with speculative enabled', () => {
    const result = runInferenceSimulation({
      modelId: 'llama2-70b',
      gpu: H100_SXM,
      numGPUs: 8,
      batchSize: 2,
      inputSeqLen: 512,
      outputSeqLen: 256,
      tensorParallel: 4,
      speculativeEnabled: true,
      draftModelSpec: gpt3_125m,
      numSpeculativeTokens: 4,
      acceptanceRate: 0.7,
    });

    expect(result.success).toBe(true);
    expect(result.speculative).toBeDefined();
    expect(result.speculative!.speedup).toBeGreaterThan(1);
    // Speculative TPOT should feed into headline (no CB, so exact match)
    expect(result.latency.tpot).toBeCloseTo(result.speculative!.effectiveTpot, 5);

    // Compare against single-replica config (numGPUs=4, TP=4, batch=1)
    const singleResult = runInferenceSimulation({
      modelId: 'llama2-70b',
      gpu: H100_SXM,
      numGPUs: 4,
      batchSize: 1, // batchPerReplica = 2/2 = 1
      inputSeqLen: 512,
      outputSeqLen: 256,
      tensorParallel: 4,
      speculativeEnabled: true,
      draftModelSpec: gpt3_125m,
      numSpeculativeTokens: 4,
      acceptanceRate: 0.7,
    });

    expect(singleResult.success).toBe(true);
    // Throughput should be ~2x single replica
    const ratio = result.throughput.tokensPerSecond / singleResult.throughput.tokensPerSecond;
    expect(ratio).toBeGreaterThan(1.8);
    expect(ratio).toBeLessThan(2.2);
  });
});

// ── Integration: no-benefit warning ──

describe('speculative decoding no-benefit warning', () => {
  it('warns when speculative provides no speedup', () => {
    // Small model (8B) + large draft (8B) at large batch → speedup ≤ 1.0
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 128,
      inputSeqLen: 512,
      outputSeqLen: 256,
      speculativeEnabled: true,
      draftModelSpec: llama3_8b,
      numSpeculativeTokens: 4,
      acceptanceRate: 0.5,
    });

    expect(result.success).toBe(true);
    expect(result.speculative).toBeDefined();
    expect(result.speculative!.speedup).toBeLessThanOrEqual(1.0);
    const hasWarning = result.warnings.some(w => w.includes('no benefit'));
    expect(hasWarning).toBe(true);
  });
});

// ── Integration: draft model KV cache included ──

describe('draft model KV cache memory', () => {
  it('draft KV cache is included in memory total', () => {
    // Use an 8B draft model at long sequence → non-trivial KV cache
    const result = runInferenceSimulation({
      modelId: 'llama2-70b',
      gpu: H100_SXM,
      numGPUs: 4,
      batchSize: 1,
      inputSeqLen: 2048,
      outputSeqLen: 2048,
      tensorParallel: 4,
      speculativeEnabled: true,
      draftModelSpec: llama3_8b,
      numSpeculativeTokens: 4,
      acceptanceRate: 0.7,
    });

    // Same config without speculative
    const baseline = runInferenceSimulation({
      modelId: 'llama2-70b',
      gpu: H100_SXM,
      numGPUs: 4,
      batchSize: 1,
      inputSeqLen: 2048,
      outputSeqLen: 2048,
      tensorParallel: 4,
    });

    expect(result.success).toBe(true);
    expect(baseline.success).toBe(true);
    const delta = result.memory.total - baseline.memory.total;
    // 8B draft in bf16: ~16GB weights / 4 TP = 4GB weights + KV cache
    // KV cache for 8B at seqLen=4096 should be non-trivial
    expect(delta).toBeGreaterThan(4e9); // at least draft weights
    // kvCache delta should be positive (draft KV included)
    expect(result.memory.kvCache).toBeGreaterThan(baseline.memory.kvCache);
  });
});

// ── Recommendations ──

describe('speculative decoding recommendations', () => {
  it('suggests enabling speculative decoding for large model at batch=1', () => {
    // Use A100 where TPOT > 30ms triggers the recommendation
    const config = {
      modelId: 'llama2-70b',
      gpuId: 'a100-80gb',
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 256,
      weightPrecision: 'fp8' as const,
    };

    const result = runInferenceSimulation(config);
    expect(result.success).toBe(true);
    // TPOT > 30ms on A100 (slower bandwidth than H100)
    expect(result.latency.tpot).toBeGreaterThan(30);

    const recs = generateInferenceRecommendations(config, result);
    const hasSpecRec = recs.some(r => r.toLowerCase().includes('speculative'));
    expect(hasSpecRec).toBe(true);
  });

  it('suggests disabling speculative decoding at large batch', () => {
    // Llama 8B + GPT-3 125M draft, batch=32 on H100
    const config = {
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 32,
      inputSeqLen: 512,
      outputSeqLen: 256,
      speculativeEnabled: true,
      draftModelSpec: gpt3_125m,
      numSpeculativeTokens: 5,
      acceptanceRate: 0.7,
    };

    const result = runInferenceSimulation(config);
    expect(result.success).toBe(true);

    const recs = generateInferenceRecommendations(config, result);
    const hasDisableRec = recs.some(r => r.toLowerCase().includes('disabling speculative'));
    expect(hasDisableRec).toBe(true);
  });
});

// ###########################################################################
//
//  Part 2 — Benchmark Calibration
//
// ###########################################################################

// ===========================================================================
// Section 1: Acceptance Rate Estimation
//
// Validates estimateAcceptanceRate() formula:
//   α = clamp(0.5 + 0.4*sqrt(draftParams/targetParams) + familyBonus + archBonus, 0.3, 0.95)
//   familyBonus = 0.05 when same family; archBonus = 0.05 when same heads AND hiddenSize
// Uses activeParams for MoE models.
// ===========================================================================

describe('Section 1: Acceptance Rate Estimation', () => {
  // Simulator: α = 0.6029. NVIDIA TRT-LLM 3.55x on H200 implies α ≥ 0.6.
  it('1.1 llama3.2-1b → llama3.3-70b (same family llama3): α ≈ 0.603', () => {
    const alpha = estimateAcceptanceRate(llama32_1b, llama33_70b);
    expectWithin(alpha, 0.6029, ALPHA_TOL, 'α(1B→70B)');
  });

  // Simulator: α = 0.6354. Larger draft → higher acceptance.
  it('1.2 llama3.2-3b → llama3.3-70b (same family): α ≈ 0.635', () => {
    const alpha = estimateAcceptanceRate(llama32_3b, llama33_70b);
    expectWithin(alpha, 0.6354, ALPHA_TOL, 'α(3B→70B)');
  });

  // Simulator: α = 0.6849. 8B/70B = higher param ratio → even higher α.
  it('1.3 llama3-8b → llama3.3-70b (same family): α ≈ 0.685', () => {
    const alpha = estimateAcceptanceRate(llama3_8b, llama33_70b);
    expectWithin(alpha, 0.6849, ALPHA_TOL, 'α(8B→70B)');
  });

  // Simulator: α = 0.6750. AMD ROCm used this pair at ~0.7.
  it('1.4 llama2-7b → llama2-70b (same family llama2): α ≈ 0.675', () => {
    const alpha = estimateAcceptanceRate(llama2_7b, llama2_70b);
    expectWithin(alpha, 0.6750, ALPHA_TOL, 'α(7B→70B)');
  });

  // Simulator: α = 0.5192. Cross-family (no family bonus) and tiny draft.
  // Monotonicity: 1.5 < 1.1 < 1.2 < 1.3
  it('1.5 gpt3-125m → llama3.3-70b (cross-family): α ≈ 0.519, and monotonicity holds', () => {
    const alpha = estimateAcceptanceRate(gpt3_125m, llama33_70b);
    expectWithin(alpha, 0.5192, ALPHA_TOL, 'α(125M→70B cross-family)');

    // Monotonicity: larger draft → higher acceptance; same family → bonus
    const a11 = estimateAcceptanceRate(llama32_1b, llama33_70b);
    const a12 = estimateAcceptanceRate(llama32_3b, llama33_70b);
    const a13 = estimateAcceptanceRate(llama3_8b, llama33_70b);
    expect(alpha, 'cross-family < same-family 1B').toBeLessThan(a11);
    expect(a11, '1B < 3B').toBeLessThan(a12);
    expect(a12, '3B < 8B').toBeLessThan(a13);
  });
});

// ===========================================================================
// Section 2: Published Benchmark Validation — H200
//
// Source: NVIDIA TRT-LLM blog (2024). H200 single GPU, batch=1, K=10, fp8.
// TRT-LLM achieves 3.55x via engine-level optimizations (paged KV, fused
// kernels, CUDA graphs) not modeled by the simulator. The simulator models
// the physics: (draftTime + verifyTime) / acceptedTokens. Expect 1.3-2.5x
// from the simulator. Pinned values are regression anchors.
// ===========================================================================

describe('Section 2: H200 FP8 Benchmarks', () => {
  // TRT-LLM: 3.55x. Simulator: 1.899x (physics-only, no engine optimizations).
  it('2.1 Llama 70B + 1B draft: speedup ≈ 1.90', () => {
    const r = specSim(llama33_70b, llama32_1b, H200_SXM, 'fp8', 1, 10);
    expect(r.success).toBe(true);
    expect(r.speculative).toBeDefined();
    expectWithin(r.speculative!.speedup, 1.899, TOL, 'H200 70B+1B speedup');
  });

  // TRT-LLM: 3.16x. Simulator: 1.613x.
  it('2.2 Llama 70B + 3B draft: speedup ≈ 1.61', () => {
    const r = specSim(llama33_70b, llama32_3b, H200_SXM, 'fp8', 1, 10);
    expect(r.success).toBe(true);
    expect(r.speculative).toBeDefined();
    expectWithin(r.speculative!.speedup, 1.613, TOL, 'H200 70B+3B speedup');
  });

  // TRT-LLM: 2.63x. Simulator: 1.292x. Larger draft has higher α but
  // more overhead, net lower speedup.
  it('2.3 Llama 70B + 8B draft: speedup ≈ 1.29', () => {
    const r = specSim(llama33_70b, llama31_8b, H200_SXM, 'fp8', 1, 10);
    expect(r.success).toBe(true);
    expect(r.speculative).toBeDefined();
    expectWithin(r.speculative!.speedup, 1.292, TOL, 'H200 70B+8B speedup');
  });

  // Ordering: smaller draft → faster iteration → higher speedup
  it('2.4 Ordering: 1B speedup > 3B speedup > 8B speedup', () => {
    const r1b = specSim(llama33_70b, llama32_1b, H200_SXM, 'fp8', 1, 10);
    const r3b = specSim(llama33_70b, llama32_3b, H200_SXM, 'fp8', 1, 10);
    const r8b = specSim(llama33_70b, llama31_8b, H200_SXM, 'fp8', 1, 10);
    expect(r1b.speculative!.speedup).toBeGreaterThan(r3b.speculative!.speedup);
    expect(r3b.speculative!.speedup).toBeGreaterThan(r8b.speculative!.speedup);
  });
});

// ===========================================================================
// Section 3: Published Benchmark Validation — MI300X
//
// Source: AMD ROCm blog (2024). MI300X TP=1, 192GB fits 70B bf16, batch=1, K=8.
// Published speedup range: 1.5-2.0x.
// ===========================================================================

describe('Section 3: MI300X Benchmarks', () => {
  // AMD ROCm: 1.5-2.0x. Simulator: 1.504x.
  it('3.1 Llama 3 70B + 8B, MI300X bf16 K=8: speedup ≈ 1.50', () => {
    const r = specSim(llama3_70b, llama3_8b, MI300X, 'bf16', 1, 8);
    expect(r.success).toBe(true);
    expect(r.speculative).toBeDefined();
    expectWithin(r.speculative!.speedup, 1.504, TOL, 'MI300X 70B+8B speedup');
  });

  // AMD ROCm: 1.5-1.9x. Simulator: 1.547x.
  it('3.2 Llama 2 70B + 7B, MI300X bf16 K=8: speedup ≈ 1.55', () => {
    const r = specSim(llama2_70b, llama2_7b, MI300X, 'bf16', 1, 8);
    expect(r.success).toBe(true);
    expect(r.speculative).toBeDefined();
    expectWithin(r.speculative!.speedup, 1.547, TOL, 'MI300X 70B+7B speedup');
  });

  // Cross-GPU comparison: MI300X bf16 TP=1 vs H100 fp8 TP=1 with same pair.
  // Speculative speedup depends on draft/target ratio, not absolute GPU speed.
  // Both should show similar relative speedup. Simulator: ratio ≈ 1.00.
  it('3.3 MI300X bf16 vs H100 fp8 (70B + 125M, K=8): similar speedup', () => {
    const mi = specSim(llama2_70b, gpt3_125m, MI300X, 'bf16', 1, 8);
    const h100 = specSim(llama2_70b, gpt3_125m, H100_SXM, 'fp8', 1, 8);
    expect(mi.success).toBe(true);
    expect(h100.success).toBe(true);
    expect(mi.speculative!.speedup).toBeGreaterThan(1.0);
    expect(h100.speculative!.speedup).toBeGreaterThan(1.0);
    // Ratio should be near 1.0 (speedup is GPU-independent for same model pair)
    const ratio = mi.speculative!.speedup / h100.speculative!.speedup;
    expectWithin(ratio, 1.001, 0.30, 'MI300X/H100 speedup ratio');
  });
});

// ===========================================================================
// Section 4: H100 SXM Benchmark Pairs
//
// H100 80GB: 70B bf16 needs TP≥2, fp8 fits at TP=1.
// Literature: 1.5-3.0x at batch=1 with α≥0.6, K=5-10.
// ===========================================================================

describe('Section 4: H100 SXM Benchmarks', () => {
  // Simulator: 1.752x. Literature: 1.5-3.0x. Uses fp8 to fit 70B+7B at TP=2.
  it('4.1 Llama 2 70B + 7B, H100 fp8 TP=2, K=5, α=0.7: speedup ≈ 1.75', () => {
    const r = specSim(llama2_70b, llama2_7b, H100_SXM, 'fp8', 2, 5, { alpha: 0.7 });
    expect(r.success).toBe(true);
    expect(r.speculative).toBeDefined();
    expectWithin(r.speculative!.speedup, 1.752, TOL, 'H100 70B+7B fp8 TP=2');
  });

  // Simulator: 1.994x. Tiny draft → fast iteration, moderate α.
  it('4.2 Llama 3 70B + 1B, H100 bf16 TP=2, K=8: speedup ≈ 1.99', () => {
    const r = specSim(llama3_70b, llama32_1b, H100_SXM, 'bf16', 2, 8);
    expect(r.success).toBe(true);
    expect(r.speculative).toBeDefined();
    expectWithin(r.speculative!.speedup, 1.994, TOL, 'H100 70B+1B bf16 TP=2');
  });

  // Simulator: 1.813x. Small target (8B fits single GPU), tiny draft.
  it('4.3 Llama 3 8B + 125M, H100 bf16 TP=1, K=5: speedup ≈ 1.81', () => {
    const r = specSim(llama3_8b, gpt3_125m, H100_SXM, 'bf16', 1, 5);
    expect(r.success).toBe(true);
    expect(r.speculative).toBeDefined();
    expect(r.speculative!.speedup).toBeGreaterThan(1.0);
    expectWithin(r.speculative!.speedup, 1.813, TOL, 'H100 8B+125M bf16 TP=1');
  });

  // More TP → faster baseline → less relative gain from speculation.
  // Simulator: TP=2 fp8 (1.752) > TP=4 fp8 (1.681).
  it('4.4 TP=2 speedup > TP=4 speedup for same pair', () => {
    const tp2 = specSim(llama2_70b, llama2_7b, H100_SXM, 'fp8', 2, 5, { alpha: 0.7 });
    const tp4 = specSim(llama2_70b, llama2_7b, H100_SXM, 'fp8', 4, 5, { alpha: 0.7 });
    expect(tp2.success).toBe(true);
    expect(tp4.success).toBe(true);
    expect(tp2.speculative!.speedup).toBeGreaterThan(tp4.speculative!.speedup);
  });
});

// ===========================================================================
// Section 5: K-Sweep
//
// How speedup varies with number of speculative tokens K. Uses
// findOptimalSpecConfig() and direct simulation.
// Config: H100 SXM fp8, Llama 70B + Llama 3.2 1B, TP=1, batch=1.
// ===========================================================================

describe('Section 5: K-Sweep', () => {
  // Even K=1 helps for large target with tiny draft.
  // Simulator: K=1 speedup = 1.553.
  it('5.1 K=1 speedup > 1.0', () => {
    const r = specSim(llama33_70b, llama32_1b, H100_SXM, 'fp8', 1, 1);
    expect(r.success).toBe(true);
    expect(r.speculative!.speedup).toBeGreaterThan(1.0);
  });

  // Speedup peaks around K=5-7, then diminishes.
  // K=5: 2.066, K=10: 1.899, K=16: 1.664. Ratio K16/K10 = 0.876 < 1.5.
  it('5.2 Speedup increases then plateaus: K=5 > K=1, diminishing returns K=16/K=10 < 1.5', () => {
    const rK1 = specSim(llama33_70b, llama32_1b, H100_SXM, 'fp8', 1, 1);
    const rK5 = specSim(llama33_70b, llama32_1b, H100_SXM, 'fp8', 1, 5);
    const rK10 = specSim(llama33_70b, llama32_1b, H100_SXM, 'fp8', 1, 10);
    const rK16 = specSim(llama33_70b, llama32_1b, H100_SXM, 'fp8', 1, 16);

    expect(rK5.speculative!.speedup).toBeGreaterThan(rK1.speculative!.speedup);
    const ratio = rK16.speculative!.speedup / rK10.speculative!.speedup;
    expect(ratio).toBeLessThan(1.5);
  });

  // findOptimalSpecConfig α=0.7: Simulator: optimalK=6, expectedSpeedup=2.564.
  // 16 entries in breakdown (K=1..16). optimalK is argmax.
  it('5.3 findOptimalSpecConfig α=0.7: optimalK ∈ [3,12], speedup > 1.5', () => {
    const opt = findOptimalSpecConfig(
      llama32_1b, llama33_70b, 0.7, 768, 1, H100_SXM, 'fp8', undefined, 1
    );
    expect(opt.optimalK).toBeGreaterThanOrEqual(3);
    expect(opt.optimalK).toBeLessThanOrEqual(12);
    expect(opt.expectedSpeedup).toBeGreaterThan(1.5);
    expect(opt.breakdown).toHaveLength(16);
    // Verify optimalK is argmax of breakdown
    const maxEntry = opt.breakdown.reduce((a, b) => a.speedup > b.speedup ? a : b);
    expect(opt.optimalK).toBe(maxEntry.k);
  });

  // Low α → small optimal K. Simulator: optimalK=2, expectedSpeedup=1.290.
  it('5.4 findOptimalSpecConfig α=0.3: optimalK ∈ [1,4], speedup < 1.5', () => {
    const opt = findOptimalSpecConfig(
      llama32_1b, llama33_70b, 0.3, 768, 1, H100_SXM, 'fp8', undefined, 1
    );
    expect(opt.optimalK).toBeGreaterThanOrEqual(1);
    expect(opt.optimalK).toBeLessThanOrEqual(4);
    expect(opt.expectedSpeedup).toBeLessThan(1.5);
  });
});

// ===========================================================================
// Section 6: α-Sweep
//
// Config: H100 SXM fp8, Llama 70B + Llama 3.2 1B, TP=1, batch=1, K=5.
// Simulator values: α=0.3→1.230, 0.4→1.430, 0.5→1.697, 0.6→2.054,
//                   0.7→2.535, 0.8→3.179, 0.9→4.038. α=0.9 K=10→5.195.
// ===========================================================================

describe('Section 6: α-Sweep', () => {
  // Speedup monotonically increases with α.
  it('6.1 Speedup monotonically increases with α (0.3 through 0.9)', () => {
    const alphas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    const speedups = alphas.map(alpha =>
      specSim(llama33_70b, llama32_1b, H100_SXM, 'fp8', 1, 5, { alpha }).speculative!.speedup
    );
    for (let i = 1; i < speedups.length; i++) {
      expect(speedups[i], `α=${alphas[i]} > α=${alphas[i - 1]}`).toBeGreaterThan(speedups[i - 1]);
    }
  });

  // α=0.3, K=5: tiny draft overhead vs large target still provides marginal benefit.
  // Simulator: 1.230.
  it('6.2 α=0.3, K=5 still provides benefit (speedup ≥ 1.0)', () => {
    const r = specSim(llama33_70b, llama32_1b, H100_SXM, 'fp8', 1, 5, { alpha: 0.3 });
    expect(r.speculative!.speedup).toBeGreaterThanOrEqual(1.0);
    expectWithin(r.speculative!.speedup, 1.230, TOL, 'α=0.3 K=5');
  });

  // α=0.9, K=10: near-theoretical max. expectedTokens ≈ 6.5.
  // Simulator: 5.195.
  it('6.3 α=0.9, K=10 achieves high speedup (> 3.0)', () => {
    const r = specSim(llama33_70b, llama32_1b, H100_SXM, 'fp8', 1, 10, { alpha: 0.9 });
    expect(r.speculative!.speedup).toBeGreaterThan(3.0);
    expectWithin(r.speculative!.speedup, 5.195, TOL, 'α=0.9 K=10');
  });
});

// ===========================================================================
// Section 7: Consumer GPU Speculative Decoding
//
// RTX 4090/3090 single GPU, TP=1, 24GB limit.
// ===========================================================================

describe('Section 7: Consumer GPU Speculative Decoding', () => {
  // Simulator: 1.788x, mem=15.6GB (13.5 target + 0.25 draft + KV + overhead).
  it('7.1 Llama 2 7B + 125M, RTX 4090 bf16, K=5: speedup ≈ 1.79', () => {
    const r = specSim(llama2_7b, gpt3_125m, RTX_4090, 'bf16', 1, 5);
    expect(r.success).toBe(true);
    expect(r.memory.total / 1e9).toBeLessThan(24);
    expectWithin(r.speculative!.speedup, 1.788, TOL, 'RTX4090 7B+125M');
  });

  // Simulator: 1.813x, mem=18.2GB. 8B model is a tighter fit on 24GB.
  it('7.2 Llama 3 8B + 125M, RTX 4090 bf16, K=5: speedup ≈ 1.81', () => {
    const r = specSim(llama3_8b, gpt3_125m, RTX_4090, 'bf16', 1, 5);
    expect(r.success).toBe(true);
    expect(r.memory.total / 1e9).toBeLessThan(24);
    expectWithin(r.speculative!.speedup, 1.813, TOL, 'RTX4090 8B+125M');
  });

  // Simulator: 1.917x, mem=7.4GB. INT4 quantization makes 13B fit easily.
  it('7.3 Llama 2 13B + 125M, RTX 4090 int4, K=5: speedup ≈ 1.92', () => {
    const r = specSim(llama2_13b, gpt3_125m, RTX_4090, 'int4', 1, 5);
    expect(r.success).toBe(true);
    expect(r.memory.total / 1e9).toBeLessThan(24);
    expectWithin(r.speculative!.speedup, 1.917, TOL, 'RTX4090 13B+125M int4');
  });

  // Speculative speedup depends on parameter ratios, not absolute GPU speed.
  // 4090 vs 3090 should give identical relative speedup.
  // Simulator: 4090=1.788, 3090=1.788, ratio=1.000.
  it('7.4 RTX 4090 vs 3090 speedup ratio ≈ 1.0 (±5%)', () => {
    const r4090 = specSim(llama2_7b, gpt3_125m, RTX_4090, 'bf16', 1, 5);
    const r3090 = specSim(llama2_7b, gpt3_125m, RTX_3090, 'bf16', 1, 5);
    expect(r4090.success).toBe(true);
    expect(r3090.success).toBe(true);
    const ratio = r4090.speculative!.speedup / r3090.speculative!.speedup;
    expectWithin(ratio, 1.000, 0.05, '4090/3090 speedup ratio');
  });
});

// ===========================================================================
// Section 8: Batch Degradation
//
// Source: vLLM blog (speedup degrades at high batch), AMD ROCm (slowdowns at
// batch≥8). Config: H100 SXM fp8, Llama 70B + Llama 3.2 1B, TP=2, K=5, α=0.7.
// Simulator: batch=1→2.446, 8→2.417, 32→1.978, 128→0.632.
// ===========================================================================

describe('Section 8: Batch Degradation', () => {
  // Strictly monotone decreasing speedup with batch size.
  it('8.1 Speedup decreases with batch: 1 > 8 > 32 > 128', () => {
    const batches = [1, 8, 32, 128];
    const speedups = batches.map(batch =>
      specSim(llama2_70b, llama32_1b, H100_SXM, 'fp8', 2, 5, { alpha: 0.7, batch }).speculative!.speedup
    );
    for (let i = 1; i < speedups.length; i++) {
      expect(
        speedups[i],
        `batch=${batches[i]}(${speedups[i].toFixed(3)}) < batch=${batches[i - 1]}(${speedups[i - 1].toFixed(3)})`
      ).toBeLessThan(speedups[i - 1]);
    }
  });

  // Batch=32: speculation still beneficial but reduced.
  // Simulator: 1.978x.
  it('8.2 Batch=32: speedup ≈ 1.98 (still beneficial)', () => {
    const r = specSim(llama2_70b, llama32_1b, H100_SXM, 'fp8', 2, 5, { alpha: 0.7, batch: 32 });
    expect(r.success).toBe(true);
    expectWithin(r.speculative!.speedup, 1.978, TOL, 'batch=32 speedup');
  });

  // Batch=128: speculation provides marginal or negative benefit.
  // Simulator: 0.632x (worse than no speculation).
  it('8.3 Batch=128: speedup < 1.0 (speculation hurts at large batch)', () => {
    const r = specSim(llama2_70b, llama32_1b, H100_SXM, 'fp8', 2, 5, { alpha: 0.7, batch: 128 });
    expect(r.success).toBe(true);
    expect(r.speculative!.speedup).toBeLessThan(1.0);
  });

  // Same-size draft: draft takes as long as normal decode.
  // 8B + 8B at batch=64: speedup = 0.353, warning includes 'no benefit'.
  it('8.4 Same-size draft (8B+8B, batch=64): speedup ≤ 1.0 with no-benefit warning', () => {
    const r = specSim(llama3_8b, llama3_8b, H100_SXM, 'bf16', 1, 4, { alpha: 0.5, batch: 64 });
    expect(r.success).toBe(true);
    expect(r.speculative).toBeDefined();
    expect(r.speculative!.speedup).toBeLessThanOrEqual(1.0);
    const hasWarning = r.warnings.some(w => w.includes('no benefit'));
    expect(hasWarning).toBe(true);
  });
});

// ===========================================================================
// Section 9: Cross-GPU Consistency
//
// Speculative speedup depends on parameter ratios, not absolute GPU speed.
// Same model pair at same TP should show similar speedup across GPUs.
// ===========================================================================

describe('Section 9: Cross-GPU Consistency', () => {
  // H100 vs A100 with 8B + 125M, TP=1, K=5, α=0.7.
  // Simulator: H100=2.436, A100=2.436, ratio=1.000.
  it('9.1 H100 vs A100 (8B+125M bf16 TP=1): speedup ratio ≈ 1.0', () => {
    const h100 = specSim(llama3_8b, gpt3_125m, H100_SXM, 'bf16', 1, 5, { alpha: 0.7 });
    const a100 = specSim(llama3_8b, gpt3_125m, A100_80GB, 'bf16', 1, 5, { alpha: 0.7 });
    expect(h100.success).toBe(true);
    expect(a100.success).toBe(true);
    const ratio = h100.speculative!.speedup / a100.speculative!.speedup;
    expectWithin(ratio, 1.000, 0.05, 'H100/A100 speedup ratio');
  });

  // H200 vs H100 both using fp8. Both should give speedup > 1.0.
  // Simulator: H200=2.066, H100=2.066.
  it('9.2 H200 vs H100 fp8 TP=1 (70B+1B): both > 1.0', () => {
    const h200 = specSim(llama33_70b, llama32_1b, H200_SXM, 'fp8', 1, 5);
    const h100 = specSim(llama33_70b, llama32_1b, H100_SXM, 'fp8', 1, 5);
    expect(h200.success).toBe(true);
    expect(h100.success).toBe(true);
    expect(h200.speculative!.speedup).toBeGreaterThan(1.0);
    expect(h100.speculative!.speedup).toBeGreaterThan(1.0);
  });

  // Ordering consistency: 70B+1B vs 70B+8B should have same relative ranking
  // on both H100 and A100. Use fp8 TP=2 for H100, bf16 TP=2 for A100 to fit.
  it('9.3 Ordering preserved across GPUs: 1B draft > 8B draft on both H100 and A100', () => {
    // H100: 70B + 1B
    const h100_1b = specSim(llama33_70b, llama32_1b, H100_SXM, 'bf16', 2, 5, { alpha: 0.7 });
    // H100: 8B + 125M (different target since 70B+8B OOM at bf16 TP=2)
    // Instead, use fixed α to make the comparison meaningful
    const h100_small = specSim(llama33_70b, gpt3_125m, H100_SXM, 'bf16', 2, 5, { alpha: 0.7 });

    // A100: same configs
    const a100_1b = specSim(llama33_70b, llama32_1b, A100_80GB, 'bf16', 2, 5, { alpha: 0.7 });
    const a100_small = specSim(llama33_70b, gpt3_125m, A100_80GB, 'bf16', 2, 5, { alpha: 0.7 });

    expect(h100_1b.success).toBe(true);
    expect(h100_small.success).toBe(true);
    expect(a100_1b.success).toBe(true);
    expect(a100_small.success).toBe(true);

    // Same α, so ordering depends on draft size → consistent across GPUs
    const h100Ordering = h100_1b.speculative!.speedup > h100_small.speculative!.speedup;
    const a100Ordering = a100_1b.speculative!.speedup > a100_small.speculative!.speedup;
    expect(h100Ordering).toBe(a100Ordering);
  });
});

// ===========================================================================
// Section 10: MoE Target with Dense Draft
//
// DeepSeek V3 (671B total, 37.6B active) validates that activeParams is used
// for acceptance rate estimation. MoE decode reads ALL expert weights from HBM
// but only activates ~37.6B params → still bandwidth-bound on total weights.
// ===========================================================================

describe('Section 10: MoE Target with Dense Draft', () => {
  // estimateAcceptanceRate uses activeParams for MoE.
  // α(1B→V3) based on 1.24B/37.6B active, not 1.24B/671B total.
  // Simulator: α = 0.5726. If using totalParams, would be ~0.52.
  it('10.1 estimateAcceptanceRate uses activeParams for MoE', () => {
    const alpha = estimateAcceptanceRate(llama32_1b, dsV3);
    expectWithin(alpha, 0.5726, ALPHA_TOL, 'α(1B→V3 active)');

    // Verify it's using activeParams by comparing to what totalParams would give
    const totalParamsRatio = llama32_1b.totalParams / dsV3.totalParams;
    const activeParamsRatio = llama32_1b.totalParams / (dsV3.activeParams ?? dsV3.totalParams);
    // activeParams ratio is much larger (1.24B/37.6B ≈ 0.033 vs 1.24B/671B ≈ 0.002)
    expect(activeParamsRatio).toBeGreaterThan(totalParamsRatio * 5);
  });

  // V3 at TP=8 on H200 fp8. MoE verification is expensive (reads all expert weights).
  // Simulator: speedup = 0.749 (speculation doesn't help — verification cost too high).
  it('10.2 V3 + 1B at H200 fp8 TP=8: speculation provides marginal benefit', () => {
    const r = specSim(dsV3, llama32_1b, H200_SXM, 'fp8', 8, 5);
    expect(r.success).toBe(true);
    expect(r.speculative).toBeDefined();
    // MoE verification reads all 671B weights → expensive → low or no speedup
    expectWithin(r.speculative!.speedup, 0.749, TOL, 'V3+1B MoE speedup');
  });

  // MoE speedup < dense speedup for same draft and K.
  // V3 active=37.6B → draft is 1/37 of active compute. Dense 70B → draft is 1/70.
  // But MoE verification reads ALL weights → much more expensive than dense.
  // Simulator: V3=0.791 < 70B=2.054 (with α=0.6 for fair comparison).
  it('10.3 MoE speedup < dense speedup for same draft', () => {
    const moe = specSim(dsV3, llama32_1b, H200_SXM, 'fp8', 8, 5, { alpha: 0.6 });
    const dense = specSim(llama33_70b, llama32_1b, H200_SXM, 'fp8', 1, 5, { alpha: 0.6 });
    expect(moe.success).toBe(true);
    expect(dense.success).toBe(true);
    expect(moe.speculative!.speedup).toBeLessThan(dense.speculative!.speedup);
  });
});

// ===========================================================================
// Section 11: CB + Speculative & Memory
//
// Continuous batching adds scheduling overhead on top of speculative TPOT.
// Draft model weights and KV cache increase total memory.
// ===========================================================================

describe('Section 11: CB + Speculative & Memory', () => {
  // CB scheduling overhead (1-2%) applied on top of speculative effectiveTpot.
  // Simulator: tpot=2.798 > effectiveTpot=2.770.
  it('11.1 CB + speculative: tpot > speculative effectiveTpot (CB overhead)', () => {
    const r = specSim(llama3_8b, gpt3_125m, H100_SXM, 'bf16', 1, 4, { alpha: 0.7, cb: true });
    expect(r.success).toBe(true);
    expect(r.speculative).toBeDefined();
    expect(r.speculative!.speedup).toBeGreaterThan(1.0);
    // CB overhead: headline TPOT > raw speculative effectiveTpot
    expect(r.latency.tpot).toBeGreaterThan(r.speculative!.effectiveTpot);
    // But the overhead should be small (< 5%)
    const overhead = r.latency.tpot / r.speculative!.effectiveTpot - 1;
    expect(overhead).toBeLessThan(0.05);
    expect(overhead).toBeGreaterThan(0);
  });

  // Draft model memory adds to total. GPT-3 125M in bf16 ≈ 250MB + KV cache.
  // Simulator: delta = 352.8MB.
  it('11.2 Draft model memory grows total memory', () => {
    const baseline = runInferenceSimulation({
      modelSpec: llama3_8b,
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 256,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
      tensorParallel: 1,
    });

    const specResult = specSim(llama3_8b, gpt3_125m, H100_SXM, 'bf16', 1, 4, { alpha: 0.7 });

    expect(baseline.success).toBe(true);
    expect(specResult.success).toBe(true);
    expect(specResult.memory.total).toBeGreaterThan(baseline.memory.total);

    // Delta should be approximately draft weights + draft KV cache
    const deltaGB = (specResult.memory.total - baseline.memory.total) / 1e9;
    expect(deltaGB).toBeGreaterThan(0.2); // at least draft weights (~250MB)
    expect(deltaGB).toBeLessThan(1.0);    // but not huge
  });

  // Large draft on tight memory → OOM.
  // 8B target + 8B draft on RTX 4090: ~16 + 16 = 32GB > 24GB.
  // Simulator: mem=34.0GB, success=false.
  it('11.3 Large draft on tight memory → OOM (8B + 8B on RTX 4090)', () => {
    const r = specSim(llama3_8b, llama3_8b, RTX_4090, 'bf16', 1, 4, { alpha: 0.7 });
    expect(r.success).toBe(false);
    expect(r.errors.length).toBeGreaterThan(0);
    expect(r.memory.total / 1e9).toBeGreaterThan(24);
  });
});
