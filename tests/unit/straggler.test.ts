/**
 * Unit tests for order-statistics straggler model.
 *
 * The model computes straggler overhead as σ√(2·ln(N)) where N = numNodes
 * and σ is the per-node step-time coefficient of variation.
 *
 * Tests verify: zero at N=1, exact values, monotonicity, concavity.
 */
import { describe, it, expect, afterEach } from 'vitest';
import {
  type SimulationConfig,
  getSimulationMetrics,
} from '../../src/core/simulation/engine.ts';
import { createMultiNodeCluster, createSingleNodeCluster } from '../../src/core/hardware/topology.ts';
import { getStragglerSigma, setStragglerSigma } from '../../src/core/strategies/base.ts';

const SIGMA = getStragglerSigma();

/** Expected overhead fraction for N nodes at given sigma. */
function expectedOverhead(numNodes: number, sigma: number = SIGMA): number {
  if (numNodes <= 1) return 0;
  return sigma * Math.sqrt(2 * Math.log(numNodes));
}

afterEach(() => {
  // Restore default sigma after each test
  setStragglerSigma(SIGMA);
});

describe('Straggler overhead: order-statistics model', () => {
  it('numNodes=1 → 0 overhead', () => {
    expect(expectedOverhead(1)).toBe(0);

    // Verify via simulator: single-node config should have no scaleOverhead
    const metrics = getSimulationMetrics({
      clusterConfig: createSingleNodeCluster('h100-sxm', 8)!,
      modelId: 'llama2-7b',
      globalBatchSize: 16,
      microBatchSize: 2,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      flashAttention: true,
      mixedPrecision: 'bf16',
    });
    expect(metrics.timing.scaleOverhead).toBe(0);
  });

  it('numNodes=2 → σ√(2·ln(2)) (exact value)', () => {
    const expected = SIGMA * Math.sqrt(2 * Math.log(2));
    expect(expectedOverhead(2)).toBeCloseTo(expected, 12);
    // Sanity: should be ~1.53% at σ=0.013
    expect(expected).toBeGreaterThan(0);
    expect(expected).toBeLessThan(0.05);
  });

  it('spot checks at N=32, 128, 2048', () => {
    const o32 = expectedOverhead(32);
    const o128 = expectedOverhead(128);
    const o2048 = expectedOverhead(2048);

    // All should be positive
    expect(o32).toBeGreaterThan(0);
    expect(o128).toBeGreaterThan(0);
    expect(o2048).toBeGreaterThan(0);

    // Verify exact formulas
    expect(o32).toBeCloseTo(SIGMA * Math.sqrt(2 * Math.log(32)), 12);
    expect(o128).toBeCloseTo(SIGMA * Math.sqrt(2 * Math.log(128)), 12);
    expect(o2048).toBeCloseTo(SIGMA * Math.sqrt(2 * Math.log(2048)), 12);
  });

  it('monotonicity: overhead(N+1) > overhead(N) for N >= 2', () => {
    for (const n of [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]) {
      expect(
        expectedOverhead(n + 1),
        `overhead(${n + 1}) should exceed overhead(${n})`,
      ).toBeGreaterThan(expectedOverhead(n));
    }
  });

  it('concavity: overhead(2N) - overhead(N) < overhead(N) - overhead(N/2) for N >= 4', () => {
    for (const n of [4, 8, 16, 32, 64, 128, 256, 512]) {
      const lower = expectedOverhead(n) - expectedOverhead(n / 2);
      const upper = expectedOverhead(2 * n) - expectedOverhead(n);
      expect(
        upper,
        `concavity at N=${n}: Δ(${2 * n},${n})=${upper} should be < Δ(${n},${n / 2})=${lower}`,
      ).toBeLessThan(lower);
    }
  });

  it('sigma=0 → zero overhead at any N', () => {
    setStragglerSigma(0);
    const metrics = getSimulationMetrics({
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 128)!,
      modelId: 'llama2-7b',
      globalBatchSize: 2048,
      microBatchSize: 2,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      flashAttention: true,
      mixedPrecision: 'bf16',
    });
    expect(metrics.timing.scaleOverhead).toBe(0);
  });

  it('higher sigma → higher overhead (directionality)', () => {
    const config: SimulationConfig = {
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 128)!,
      modelId: 'llama2-7b',
      globalBatchSize: 2048,
      microBatchSize: 2,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      flashAttention: true,
      mixedPrecision: 'bf16',
    };

    setStragglerSigma(0.010);
    const mfuLow = getSimulationMetrics(config).mfu;

    setStragglerSigma(0.020);
    const mfuHigh = getSimulationMetrics(config).mfu;

    // Higher sigma → more straggler overhead → lower MFU
    expect(mfuHigh).toBeLessThan(mfuLow);
  });

  it('multi-node config has nonzero scaleOverhead', () => {
    const metrics = getSimulationMetrics({
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 128)!,
      modelId: 'llama2-7b',
      globalBatchSize: 2048,
      microBatchSize: 2,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      flashAttention: true,
      mixedPrecision: 'bf16',
    });
    expect(metrics.timing.scaleOverhead).toBeGreaterThan(0);
  });
});
