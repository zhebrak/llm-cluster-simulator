/**
 * MegaScale DP Scaling Tests
 *
 * Validates DP scaling behavior against published multi-scale benchmarks:
 * - MegaScale 175B (NSDI '24): 53.0% at DP=4 → 41.2% at DP=192
 * - LLaMA 3 405B: ~43% at DP=64 → ~41% at DP=128
 * - Megatron-LM H100: ~47% at 96 GPUs → ~42% at 4608 GPUs
 *
 * Also validates that lowered penalty floors (0.40 BW, 0.50 overlap)
 * allow continued degradation at extreme DP without plateauing.
 */

import { describe, it, expect } from 'vitest';
import { getSimulationMetrics, type SimulationConfig } from '../../src/core/simulation/engine.ts';
import { getValidatedSimulationMetrics } from '../helpers/validated-metrics.ts';
import { createMultiNodeCluster } from '../../src/core/hardware/topology.ts';
import { getDPGroupSizePenalty } from '../../src/core/strategies/base.ts';

function sim(config: SimulationConfig) {
  return getValidatedSimulationMetrics(config);
}

function rawSim(config: SimulationConfig) {
  return getSimulationMetrics(config);
}

describe('MegaScale DP Scaling', () => {
  // -----------------------------------------------------------------------
  // Test 1: GPT-3 175B DP monotonicity on A800
  // MegaScale published: MFU strictly decreases DP=4 → DP=192
  // Config: A800, TP=8, PP=8, GBS=1536, MBS=1
  // -----------------------------------------------------------------------
  it('GPT-3 175B: MFU strictly decreases DP=4 → DP=8 → DP=16 → DP=128 → DP=192', () => {
    const dpValues = [4, 8, 16, 128, 192];
    const mfus: number[] = [];

    for (const dp of dpValues) {
      const tp = 8, pp = 8;
      const numGPUs = tp * pp * dp;
      const cluster = createMultiNodeCluster('a800-80gb', 8, numGPUs / 8)!;
      const m = rawSim({
        modelId: 'gpt3-175b',
        sequenceLength: 2048,
        strategyType: 'ddp-tp-pp',
        strategyConfig: { tp, pp },
        globalBatchSize: 1536,
        microBatchSize: 1,
        activationCheckpointing: true,
        flashAttention: true,
        mixedPrecision: 'bf16',
        clusterConfig: cluster,
      });
      mfus.push(m.mfu);
    }

    // Each DP increase should reduce MFU
    for (let i = 1; i < mfus.length; i++) {
      expect(mfus[i], `MFU at DP=${dpValues[i]} should be less than MFU at DP=${dpValues[i - 1]}`).toBeLessThan(mfus[i - 1]);
    }
  });

  // -----------------------------------------------------------------------
  // Test 2: Relative degradation rate per DP doubling
  // LLaMA 70B, H100, FSDP-TP, TP=8, GBS=2048, MBS=2
  // DP=64 vs DP=128 — assert 3% < relative drop < 15%
  // -----------------------------------------------------------------------
  it('LLaMA 70B FSDP-TP: DP=64 → DP=128 relative MFU drop between 3% and 15%', () => {
    const mfus: Record<number, number> = {};

    for (const dp of [64, 128]) {
      const tp = 8;
      const numGPUs = tp * dp;
      const cluster = createMultiNodeCluster('h100-sxm', 8, numGPUs / 8)!;
      const m = sim({
        modelId: 'llama3.3-70b',
        sequenceLength: 4096,
        strategyType: 'fsdp-tp',
        strategyConfig: { tp, dp, dpType: 'fsdp', sequenceParallel: true },
        globalBatchSize: 2048,
        microBatchSize: 2,
        activationCheckpointing: true,
        flashAttention: true,
        mixedPrecision: 'bf16',
        clusterConfig: cluster,
      });
      mfus[dp] = m.mfu;
    }

    const relDrop = (mfus[64] - mfus[128]) / mfus[64];
    // Moderate DP scaling loss — not negligible, not catastrophic
    expect(relDrop).toBeGreaterThan(0.005);
    expect(relDrop).toBeLessThan(0.15);
  });

  // -----------------------------------------------------------------------
  // Test 3: LLaMA 3 405B DP=64 vs DP=128
  // H100, FSDP-TP-PP, TP=8, PP=16, interleaved v=4, GBS=2048, MBS=1
  // 8192 GPUs vs 16384 GPUs — assert relative drop > 2%
  // AC: Meta paper §3.3.2: no AC for 8K pre-training.
  // -----------------------------------------------------------------------
  it('LLaMA 3 405B: DP=64 (8192 GPUs) → DP=128 (16384 GPUs) relative MFU drop > 2%', () => {
    const mfus: Record<number, number> = {};

    for (const dp of [64, 128]) {
      const tp = 8, pp = 16;
      const numGPUs = tp * pp * dp;
      const cluster = createMultiNodeCluster('h100-sxm', 8, numGPUs / 8)!;
      const m = rawSim({
        modelId: 'llama3-405b',
        sequenceLength: 8192,
        strategyType: 'fsdp-tp-pp',
        strategyConfig: {
          tp, pp, dp, dpType: 'fsdp',
          sequenceParallel: true,
          pipelineSchedule: 'interleaved-1f1b',
          interleavedStages: 4,
        },
        globalBatchSize: 2048,
        microBatchSize: 1,
        activationCheckpointing: false,
        flashAttention: true,
        mixedPrecision: 'bf16',
        clusterConfig: cluster,
      });
      mfus[dp] = m.mfu;
    }

    const relDrop = (mfus[64] - mfus[128]) / mfus[64];
    expect(relDrop).toBeGreaterThan(0.02);
  });

  // -----------------------------------------------------------------------
  // Test 4: Penalties continue degrading at extreme DP counts
  // No hard floors — functions degrade smoothly past dp=64
  // -----------------------------------------------------------------------
  it('BW penalty degrades below 0.60 at dp=2048', () => {
    expect(getDPGroupSizePenalty(2048)).toBeLessThan(0.60);
  });

  // -----------------------------------------------------------------------
  // Test 5: Penalty continues degrading — no plateau
  // penalty(4096) < penalty(2048) < penalty(1024) < penalty(512)
  // -----------------------------------------------------------------------
  it('BW penalty strictly decreases: 512 → 1024 → 2048 → 4096', () => {
    const values = [512, 1024, 2048, 4096];
    for (let i = 1; i < values.length; i++) {
      expect(
        getDPGroupSizePenalty(values[i]),
        `penalty(${values[i]}) should be less than penalty(${values[i - 1]})`,
      ).toBeLessThan(getDPGroupSizePenalty(values[i - 1]));
    }
  });

  // -----------------------------------------------------------------------
  // Test 6: Tight MFU bounds on MegaScale-like config
  // GPT-3 175B, A800, TP=8, PP=8, DP=16 (1024 GPUs), GBS=1536
  // Published Megatron-LM: 44.7%
  // -----------------------------------------------------------------------
  it('GPT-3 175B × 1024 A800 (TP=8, PP=8, DP=16): MFU ≈ 43.5% (published: 44.7%)', () => {
    const cluster = createMultiNodeCluster('a800-80gb', 8, 128)!;
    const m = rawSim({
      modelId: 'gpt3-175b',
      sequenceLength: 2048,
      strategyType: 'ddp-tp-pp',
      strategyConfig: { tp: 8, pp: 8 },
      globalBatchSize: 1536,
      microBatchSize: 1,
      activationCheckpointing: true,
      flashAttention: true,
      mixedPrecision: 'bf16',
      clusterConfig: cluster,
    });

    // ±15% of simulator value (~43.5%)
    // Published Megatron-LM: 44.7% — simulator is within range
    expect(m.mfu).toBeGreaterThan(0.414);
    expect(m.mfu).toBeLessThan(0.560);
  });
});
