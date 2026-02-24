/**
 * Pareto frontier validation tests.
 *
 * Verifies mathematical correctness, precision ordering, TP tradeoffs,
 * batch size effects, CB dominance, MoE EP, and sweep sanity.
 */

import { describe, it, expect } from 'vitest';
import { getModel } from '../../src/core/models/index.ts';
import { ALL_GPUS } from '../../src/core/hardware/gpu.ts';
import { runInferenceSimulationRaw } from '../../src/core/inference/simulation.ts';
import {
  computeCostPerMToken,
  buildSweepKey,
  extractParetoFrontier,
  type ParetoPoint,
} from '../../src/core/inference/pareto.ts';
import { generateInferenceCandidates } from '../../src/core/inference/exploration.ts';
import type { InferenceSimulationConfig } from '../../src/core/inference/simulation.ts';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Run a single config and return a ParetoPoint (or null if OOM). */
function runPoint(config: InferenceSimulationConfig, numGPUs: number, rate: number): ParetoPoint | null {
  const result = runInferenceSimulationRaw(config);
  if (!result.success || result.throughput.tokensPerSecond <= 0) return null;
  const cost = computeCostPerMToken(result.throughput.tokensPerSecond, numGPUs, rate);
  if (!isFinite(cost) || cost <= 0) return null;
  const gpu = ALL_GPUS[config.gpuId ?? ''] ?? ALL_GPUS['h100-sxm'];
  return {
    config: {
      tp: config.tensorParallel ?? 1,
      ep: config.expertParallel,
      batchSize: config.batchSize ?? 1,
      weightPrecision: (config.weightPrecision ?? 'bf16') as ParetoPoint['config']['weightPrecision'],
      kvCachePrecision: (config.kvCachePrecision ?? 'bf16') as ParetoPoint['config']['kvCachePrecision'],
      continuousBatching: config.continuousBatching ?? false,
    },
    costPerMToken: cost,
    ttft: result.latency.ttft,
    tpot: result.latency.tpot,
    throughput: result.throughput.tokensPerSecond,
    memoryUtil: result.memory.total / (gpu.memoryGB * 1e9),
  };
}

const RATE_H100 = 3.50; // $/GPU-hour
const RATE_H200 = 4.50;

// ---------------------------------------------------------------------------
// 1. Pareto frontier mathematical correctness
// ---------------------------------------------------------------------------

describe('Pareto frontier mathematical correctness', () => {
  // Build a moderate set of real points: Llama 70B on 4× H100
  const model = getModel('llama3-70b', 512)!;
  const gpu = ALL_GPUS['h100-sxm'];
  const baseConfig: InferenceSimulationConfig = {
    modelId: 'llama3-70b',
    gpuId: 'h100-sxm',
    numGPUs: 4,
    inputSeqLen: 512,
    outputSeqLen: 128,
    flashAttention: true,
    pagedAttention: true,
  };
  const candidates = generateInferenceCandidates(baseConfig, model, gpu);

  // Run all candidates (synchronously — no abort needed in tests)
  const points: ParetoPoint[] = [];
  for (const c of candidates) {
    const p = runPoint(c, 4, RATE_H100);
    if (p) points.push(p);
  }

  const frontier = extractParetoFrontier(points, 'ttft');
  const frontierTpot = extractParetoFrontier(points, 'tpot');

  it('has meaningful data — at least 10 successful configs', () => {
    expect(points.length).toBeGreaterThanOrEqual(10);
  });

  it('frontier has >= 2 points (meaningful tradeoff)', () => {
    expect(frontier.length).toBeGreaterThanOrEqual(2);
    expect(frontierTpot.length).toBeGreaterThanOrEqual(2);
  });

  it('frontier is sorted by cost ascending', () => {
    for (let i = 1; i < frontier.length; i++) {
      expect(frontier[i].costPerMToken).toBeGreaterThanOrEqual(frontier[i - 1].costPerMToken);
    }
  });

  it('frontier latency is monotonically decreasing', () => {
    for (let i = 1; i < frontier.length; i++) {
      expect(frontier[i].ttft).toBeLessThanOrEqual(frontier[i - 1].ttft);
    }
  });

  it('no non-frontier point dominates a frontier point', () => {
    for (const p of points) {
      for (const f of frontier) {
        // p dominates f if p has BOTH lower cost AND lower TTFT
        const dominates = p.costPerMToken < f.costPerMToken && p.ttft < f.ttft;
        expect(dominates).toBe(false);
      }
    }
  });

  it('every non-frontier point is dominated by at least one frontier point', () => {
    const frontierSet = new Set(frontier);
    for (const p of points) {
      if (frontierSet.has(p)) continue;
      // At least one frontier point should dominate p (lower/equal cost AND lower/equal latency, with strict in at least one)
      const dominated = frontier.some(
        f => f.costPerMToken <= p.costPerMToken && f.ttft <= p.ttft &&
             (f.costPerMToken < p.costPerMToken || f.ttft < p.ttft)
      );
      expect(dominated).toBe(true);
    }
  });
});

// ---------------------------------------------------------------------------
// 2. Precision ordering
// ---------------------------------------------------------------------------

describe('Precision ordering (Llama 70B, 8× H100)', () => {
  const base: InferenceSimulationConfig = {
    modelId: 'llama3-70b',
    gpuId: 'h100-sxm',
    numGPUs: 8,
    inputSeqLen: 512,
    outputSeqLen: 128,
    flashAttention: true,
    pagedAttention: true,
    tensorParallel: 4,
    batchSize: 32,
    continuousBatching: false,
  };

  const bf16 = runPoint({ ...base, weightPrecision: 'bf16', kvCachePrecision: 'bf16' }, 8, RATE_H100);
  const fp8 = runPoint({ ...base, weightPrecision: 'fp8', kvCachePrecision: 'fp8' }, 8, RATE_H100);
  const int4 = runPoint({ ...base, weightPrecision: 'int4', kvCachePrecision: 'int4' }, 8, RATE_H100);

  it('INT4 is cheaper than BF16 at matched config', () => {
    expect(bf16).not.toBeNull();
    expect(int4).not.toBeNull();
    expect(int4!.costPerMToken).toBeLessThan(bf16!.costPerMToken);
  });

  it('FP8 is cheaper than BF16 at matched config', () => {
    expect(fp8).not.toBeNull();
    expect(fp8!.costPerMToken).toBeLessThan(bf16!.costPerMToken);
  });
});

// ---------------------------------------------------------------------------
// 3. TP vs latency tradeoff
// ---------------------------------------------------------------------------

describe('TP vs latency tradeoff (Llama 70B, 8× H100)', () => {
  // Use batch=1 so per-replica batch is always 1 regardless of replicas,
  // isolating the pure TP effect on per-replica latency.
  const base: InferenceSimulationConfig = {
    modelId: 'llama3-70b',
    gpuId: 'h100-sxm',
    numGPUs: 8,
    inputSeqLen: 512,
    outputSeqLen: 128,
    flashAttention: true,
    pagedAttention: true,
    weightPrecision: 'fp8',
    kvCachePrecision: 'fp8',
    batchSize: 1,
    continuousBatching: false,
  };

  const tp1 = runPoint({ ...base, tensorParallel: 1 }, 8, RATE_H100);
  const tp4 = runPoint({ ...base, tensorParallel: 4 }, 8, RATE_H100);

  it('higher TP reduces per-replica TTFT (batch=1)', () => {
    // TP=1 may OOM on 70B — skip if so
    if (!tp1 || !tp4) return;
    // With batch=1, per-replica batch is always 1, so TP=4 should have lower TTFT
    expect(tp4.ttft).toBeLessThan(tp1.ttft);
  });

  it('higher TP increases cost at batch=1 (fewer replicas, compute-bound)', () => {
    if (!tp1 || !tp4) return;
    // At batch=1, compute is tiny → increasing TP just wastes GPUs
    // TP=1 gets 8 replicas, TP=4 gets 2 replicas → TP=1 has 4× more aggregate throughput
    expect(tp4.costPerMToken).toBeGreaterThan(tp1.costPerMToken);
  });
});

// ---------------------------------------------------------------------------
// 4. Batch size vs cost/latency
// ---------------------------------------------------------------------------

describe('Batch size vs cost/latency (Llama 70B, 4× H100, TP=4)', () => {
  const base: InferenceSimulationConfig = {
    modelId: 'llama3-70b',
    gpuId: 'h100-sxm',
    numGPUs: 4,
    inputSeqLen: 512,
    outputSeqLen: 128,
    flashAttention: true,
    pagedAttention: true,
    weightPrecision: 'fp8',
    kvCachePrecision: 'fp8',
    tensorParallel: 4,
    continuousBatching: false,
  };

  const b1 = runPoint({ ...base, batchSize: 1 }, 4, RATE_H100);
  const b64 = runPoint({ ...base, batchSize: 64 }, 4, RATE_H100);

  it('larger batch reduces cost per token', () => {
    expect(b1).not.toBeNull();
    expect(b64).not.toBeNull();
    expect(b64!.costPerMToken).toBeLessThan(b1!.costPerMToken);
  });

  it('larger batch increases TTFT', () => {
    expect(b64!.ttft).toBeGreaterThan(b1!.ttft);
  });
});

// ---------------------------------------------------------------------------
// 5. CB vs static batching dominance
// ---------------------------------------------------------------------------

describe('CB vs static batching', () => {
  const base: InferenceSimulationConfig = {
    modelId: 'llama3-70b',
    gpuId: 'h100-sxm',
    numGPUs: 4,
    inputSeqLen: 512,
    outputSeqLen: 128,
    flashAttention: true,
    pagedAttention: true,
    weightPrecision: 'fp8',
    kvCachePrecision: 'fp8',
    tensorParallel: 4,
  };

  it('CB dominates at high batch', () => {
    const cb = runPoint({ ...base, batchSize: 64, continuousBatching: true }, 4, RATE_H100);
    const st = runPoint({ ...base, batchSize: 64, continuousBatching: false }, 4, RATE_H100);
    expect(cb).not.toBeNull();
    expect(st).not.toBeNull();
    // CB should have lower cost (higher throughput due to better scheduling)
    expect(cb!.costPerMToken).toBeLessThan(st!.costPerMToken);
  });

  it('static competitive at batch=1', () => {
    const cb = runPoint({ ...base, batchSize: 1, continuousBatching: true }, 4, RATE_H100);
    const st = runPoint({ ...base, batchSize: 1, continuousBatching: false }, 4, RATE_H100);
    expect(cb).not.toBeNull();
    expect(st).not.toBeNull();
    // At batch=1, difference should be small (< 50%)
    const ratio = cb!.costPerMToken / st!.costPerMToken;
    expect(ratio).toBeGreaterThan(0.5);
    expect(ratio).toBeLessThan(1.5);
  });
});

// ---------------------------------------------------------------------------
// 6. MoE model — DeepSeek V3 (8× H200)
// ---------------------------------------------------------------------------

describe('MoE: DeepSeek V3 (8× H200)', () => {
  const model = getModel('deepseek-v3', 512)!;
  const gpu = ALL_GPUS['h200-sxm'];
  const baseConfig: InferenceSimulationConfig = {
    modelId: 'deepseek-v3',
    gpuId: 'h200-sxm',
    numGPUs: 8,
    inputSeqLen: 512,
    outputSeqLen: 128,
    flashAttention: true,
    pagedAttention: true,
  };

  // Build a focused set of candidates
  const candidates = generateInferenceCandidates(baseConfig, model, gpu);
  const points: ParetoPoint[] = [];
  for (const c of candidates) {
    const p = runPoint(c, 8, RATE_H200);
    if (p) points.push(p);
  }

  const frontier = extractParetoFrontier(points, 'ttft');

  it('has successful configs', () => {
    expect(points.length).toBeGreaterThanOrEqual(5);
  });

  it('frontier has >= 3 points (rich tradeoff space)', () => {
    expect(frontier.length).toBeGreaterThanOrEqual(3);
  });

  it('EP reduces cost for MoE at matched TP/batch/precision', () => {
    const base: InferenceSimulationConfig = {
      modelId: 'deepseek-v3',
      gpuId: 'h200-sxm',
      numGPUs: 8,
      inputSeqLen: 512,
      outputSeqLen: 128,
      flashAttention: true,
      pagedAttention: true,
      weightPrecision: 'fp8',
      kvCachePrecision: 'fp8',
      tensorParallel: 2,
      batchSize: 16,
      continuousBatching: false,
    };

    const ep1 = runPoint({ ...base, expertParallel: 1 }, 8, RATE_H200);
    const ep4 = runPoint({ ...base, expertParallel: 4 }, 8, RATE_H200);

    // Either may OOM — only assert if both succeed
    if (ep1 && ep4) {
      expect(ep4.costPerMToken).toBeLessThan(ep1.costPerMToken);
    }
  });
});

// ---------------------------------------------------------------------------
// 7. Sweep sanity
// ---------------------------------------------------------------------------

describe('Sweep sanity', () => {
  it('all points have finite, positive values', () => {
    const model = getModel('llama3-70b', 512)!;
    const gpu = ALL_GPUS['h100-sxm'];
    const baseConfig: InferenceSimulationConfig = {
      modelId: 'llama3-70b',
      gpuId: 'h100-sxm',
      numGPUs: 4,
      inputSeqLen: 512,
      outputSeqLen: 128,
      flashAttention: true,
      pagedAttention: true,
    };
    const candidates = generateInferenceCandidates(baseConfig, model, gpu);
    const points: ParetoPoint[] = [];
    for (const c of candidates) {
      const p = runPoint(c, 4, RATE_H100);
      if (p) points.push(p);
    }
    for (const p of points) {
      expect(p.costPerMToken).toBeGreaterThan(0);
      expect(p.ttft).toBeGreaterThan(0);
      expect(p.tpot).toBeGreaterThan(0);
      expect(p.throughput).toBeGreaterThan(0);
      expect(isFinite(p.costPerMToken)).toBe(true);
      expect(isFinite(p.ttft)).toBe(true);
      expect(isFinite(p.tpot)).toBe(true);
      expect(isFinite(p.throughput)).toBe(true);
    }
  });

  it('sweep key changes with model/GPU/numGPUs/seqLens', () => {
    const k1 = buildSweepKey('llama3-70b', 'h100-sxm', 4, 512, 128);
    const k2 = buildSweepKey('llama3-70b', 'h100-sxm', 8, 512, 128);
    const k3 = buildSweepKey('llama3-8b', 'h100-sxm', 4, 512, 128);
    const k4 = buildSweepKey('llama3-70b', 'a100-80gb', 4, 512, 128);
    const k5 = buildSweepKey('llama3-70b', 'h100-sxm', 4, 1024, 128);
    const k6 = buildSweepKey('llama3-70b', 'h100-sxm', 4, 512, 256);
    expect(k1).not.toBe(k2);
    expect(k1).not.toBe(k3);
    expect(k1).not.toBe(k4);
    expect(k1).not.toBe(k5);
    expect(k1).not.toBe(k6);
    // Same inputs → same key
    expect(buildSweepKey('llama3-70b', 'h100-sxm', 4, 512, 128)).toBe(k1);
  });

  it('computeCostPerMToken handles edge cases', () => {
    expect(computeCostPerMToken(0, 4, 3.5)).toBe(Infinity);
    expect(computeCostPerMToken(1000, 4, 0)).toBe(Infinity);
    const cost = computeCostPerMToken(10000, 8, 3.5);
    expect(cost).toBeGreaterThan(0);
    expect(isFinite(cost)).toBe(true);
  });

  it('extractParetoFrontier on empty array returns empty', () => {
    expect(extractParetoFrontier([], 'ttft')).toEqual([]);
  });
});
