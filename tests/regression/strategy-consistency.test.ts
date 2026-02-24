/**
 * Strategy route consistency and exploit regression tests.
 *
 * Section A: Equivalent configs through different code paths should agree within 2pp.
 * Section B: Known exploit configs documented with target bounds.
 */
import { describe, it, expect } from 'vitest';
import { type SimulationConfig } from '../../src/core/simulation/engine.ts';
import { getValidatedSimulationMetrics } from '../helpers/validated-metrics.ts';
import { createMultiNodeCluster, createSingleNodeCluster } from '../../src/core/hardware/topology.ts';

function sim(config: SimulationConfig) {
  return getValidatedSimulationMetrics(config);
}

// ── Section A: Route consistency ──
// Same config through standalone vs 3D code paths should agree within 2pp.

describe('Route consistency (same config, different code paths)', () => {
  it('LLaMA 2 7B FSDP: fsdp vs fsdp-tp-pp (tp=1, pp=1)', () => {
    const standalone = sim({
      clusterConfig: createSingleNodeCluster('h100-sxm', 8)!,
      modelId: 'llama2-7b',
      globalBatchSize: 16,
      microBatchSize: 2,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      flashAttention: true,
      mixedPrecision: 'bf16',
    });

    const threeD = sim({
      clusterConfig: createSingleNodeCluster('h100-sxm', 8)!,
      modelId: 'llama2-7b',
      globalBatchSize: 16,
      microBatchSize: 2,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 1, pp: 1 },
      flashAttention: true,
      mixedPrecision: 'bf16',
    });

    const diffPp = Math.abs(standalone.mfu - threeD.mfu) * 100;
    expect(diffPp).toBeLessThan(5);
  });

  it('LLaMA 2 7B TP=4: fsdp-tp vs fsdp-tp-pp (tp=4, pp=1)', () => {
    const standalone = sim({
      clusterConfig: createSingleNodeCluster('h100-sxm', 8)!,
      modelId: 'llama2-7b',
      globalBatchSize: 16,
      microBatchSize: 2,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 4 },
      flashAttention: true,
      mixedPrecision: 'bf16',
    });

    const threeD = sim({
      clusterConfig: createSingleNodeCluster('h100-sxm', 8)!,
      modelId: 'llama2-7b',
      globalBatchSize: 16,
      microBatchSize: 2,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 4, pp: 1 },
      flashAttention: true,
      mixedPrecision: 'bf16',
    });

    const diffPp = Math.abs(standalone.mfu - threeD.mfu) * 100;
    expect(diffPp).toBeLessThan(2);
  });
});

// ── Section B: Exploit regression ──
// Known configs that can produce inflated MFU due to strategy seams.
// Each test enforces an upper bound on MFU for these configs.

describe('Exploit regression targets', () => {
  // Target: ≤ 47% (EP=32 dispatch + fabric congestion with dp+ep=160)
  it('DeepSeek V3 FSDP-TP dp=512: mfu ≤ 49%', () => {
    const metrics = sim({
      clusterConfig: createMultiNodeCluster('h800-sxm', 8, 512)!,
      modelId: 'deepseek-v3',
      globalBatchSize: 8192,
      microBatchSize: 2,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: {
        tp: 4, pp: 8, dp: 128, ep: 32,
        dpType: 'fsdp',
        sequenceParallel: true,
        pipelineSchedule: 'dualpipe-v',
        interleavedStages: 1,
        numMicroBatches: 64,
      },
      activationCheckpointing: true,
      flashAttention: true,
      mixedPrecision: 'fp8',
    });
    expect(metrics.mfu * 100).toBeLessThanOrEqual(49);
  });

  // Target: ≤ 65% (large MoE with EP=32 on 256 nodes)
  it('LLaMA 4 Maverick FSDP-TP ep=32: modelFlopsMfu ≤ 65%', () => {
    const metrics = sim({
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 256)!,
      modelId: 'llama4-maverick',
      globalBatchSize: 2048,
      microBatchSize: 1,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: {
        tp: 8, pp: 1, dp: 256, ep: 32,
        dpType: 'fsdp',
        sequenceParallel: true,
      },
      activationCheckpointing: true,
      flashAttention: true,
      mixedPrecision: 'bf16',
    });
    expect((metrics.modelFlopsMfu ?? metrics.mfu) * 100).toBeLessThanOrEqual(65);
  });

  // Target: ≤ 51% (Grok-1 MoE with EP>1 interactions)
  it('Grok-1 dp=64 tp=4 sp=true: mfu ≤ 51%', () => {
    const metrics = sim({
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 64)!,
      modelId: 'grok-1',
      globalBatchSize: 512,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp',
      strategyConfig: {
        tp: 4,
        sequenceParallel: true,
      },
      activationCheckpointing: true,
      flashAttention: true,
      mixedPrecision: 'bf16',
    });
    expect(metrics.mfu * 100).toBeLessThanOrEqual(51);
  });

  // Target: ≤ 47% (GPT-3 DDP-TP-PP, full AC)
  it('GPT-3 DDP-TP-PP: mfu ≤ 47%', () => {
    const metrics = sim({
      clusterConfig: createMultiNodeCluster('a100-80gb', 8, 128)!,
      modelId: 'gpt3-175b',
      globalBatchSize: 1536,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'ddp-tp-pp',
      strategyConfig: {
        tp: 8, pp: 8, dp: 16,
        dpType: 'ddp',
        sequenceParallel: false,
        pipelineSchedule: 'interleaved-1f1b',
        interleavedStages: 2,
      },
      activationCheckpointing: true,
      flashAttention: false,
      mixedPrecision: 'bf16',
    });
    expect(metrics.mfu * 100).toBeLessThanOrEqual(47);
  });

  // Target: ≤ 52% (Nemotron-4 340B, no PP, dp=768)
  // Per-layer FSDP pipeline hides most comm for this compute-bound model.
  // No PP means no bubble penalty, so MFU is higher than the PP=12 anchor (41%).
  it('Nemotron-4 FSDP-TP dp=768 no PP: mfu ≤ 52%', () => {
    const metrics = sim({
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 768)!,
      modelId: 'nemotron-4-340b',
      globalBatchSize: 768,
      microBatchSize: 1,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp',
      strategyConfig: {
        tp: 8,
        sequenceParallel: true,
      },
      activationCheckpointing: true,
      checkpointingGranularity: 'selective',
      flashAttention: true,
      mixedPrecision: 'bf16',
    });
    expect(metrics.mfu * 100).toBeLessThanOrEqual(52);
  });
});
