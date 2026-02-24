/**
 * PP P2P Bandwidth Tests
 *
 * Verifies that pipeline-parallel P2P communication uses per-GPU bandwidth
 * (not aggregate node bandwidth) for cross-node transfers.
 *
 * P2P send/recv uses one NIC per GPU, unlike collectives (AllReduce, AllGather)
 * which use all NICs via multiple NCCL rings.
 */

import { describe, it, expect } from 'vitest';
import { getValidatedSimulationMetrics } from '../helpers/validated-metrics.ts';
import { createMultiNodeCluster } from '../../src/core/hardware/index.ts';
import type { SimulationConfig } from '../../src/core/simulation/engine.ts';

function sim(config: SimulationConfig) {
  return getValidatedSimulationMetrics(config);
}

describe('PP P2P Bandwidth', () => {
  // Cross-node PP should be significantly slower than intra-node PP
  // because P2P uses one NIC per GPU (aggregate/gpusPerNode).
  it('cross-node PP has lower MFU than intra-node PP for same model', () => {
    const model = 'llama3.1-8b';

    // Intra-node: PP=2, TP=4 on 8 GPUs (1 node) → stagesPerNode=2 >= pp, uses NVLink
    const intraNode = sim({
      modelId: model,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: {
        tp: 4, pp: 2, dp: 1, dpType: 'fsdp',
        sequenceParallel: true,
      },
      globalBatchSize: 16,
      microBatchSize: 2,
      activationCheckpointing: true,
      flashAttention: true,
      mixedPrecision: 'bf16',
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 1)!,
    });

    // Cross-node: PP=2, TP=8 on 16 GPUs (2 nodes) → stagesPerNode=1, uses IB per-GPU
    const crossNode = sim({
      modelId: model,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: {
        tp: 8, pp: 2, dp: 1, dpType: 'fsdp',
        sequenceParallel: true,
      },
      globalBatchSize: 16,
      microBatchSize: 2,
      activationCheckpointing: true,
      flashAttention: true,
      mixedPrecision: 'bf16',
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 2)!,
    });

    // Cross-node PP should produce lower MFU due to IB P2P bandwidth limit
    expect(crossNode.mfu).toBeLessThan(intraNode.mfu);
  });

  // PP=4 with TP=8 on 4 nodes → stagesPerNode=1 (all cross-node).
  // With per-GPU BW = 400/8 = 50 GB/s (NDR), P2P comm is non-trivial.
  // A 70B model should still achieve reasonable MFU thanks to C/(C+T) overlap.
  it('70B PP=4 all-cross-node: MFU still reasonable due to overlap', () => {
    const cluster = createMultiNodeCluster('h100-sxm', 8, 4)!;  // 32 GPUs
    const m = sim({
      modelId: 'llama3.3-70b',
      sequenceLength: 4096,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: {
        tp: 8, pp: 4, dp: 1, dpType: 'fsdp',
        sequenceParallel: true,
      },
      globalBatchSize: 32,
      microBatchSize: 2,
      activationCheckpointing: true,
      flashAttention: true,
      mixedPrecision: 'bf16',
      clusterConfig: cluster,
    });

    // Should still be > 25% MFU — overlap absorbs most PP comm for large models
    expect(m.mfu).toBeGreaterThan(0.25);
    // But not unreasonably high
    expect(m.mfu).toBeLessThan(0.55);
  });

  // Standalone pipeline-parallel strategy should also use per-GPU bandwidth
  it('standalone PP strategy uses per-GPU bandwidth for cross-node', () => {
    // Single-node PP (NVLink)
    const singleNode = sim({
      modelId: 'llama3.1-8b',
      sequenceLength: 4096,
      strategyType: 'ddp-tp-pp',
      strategyConfig: {
        tp: 1, pp: 4, dp: 2, dpType: 'ddp',
      },
      globalBatchSize: 16,
      microBatchSize: 2,
      activationCheckpointing: true,
      flashAttention: true,
      mixedPrecision: 'bf16',
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 1)!,
    });

    // Multi-node PP (IB per-GPU)
    const multiNode = sim({
      modelId: 'llama3.1-8b',
      sequenceLength: 4096,
      strategyType: 'ddp-tp-pp',
      strategyConfig: {
        tp: 1, pp: 4, dp: 4, dpType: 'ddp',
      },
      globalBatchSize: 32,
      microBatchSize: 2,
      activationCheckpointing: true,
      flashAttention: true,
      mixedPrecision: 'bf16',
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 2)!,
    });

    // Multi-node should have lower MFU due to PP cross-node P2P + DP comm
    expect(multiNode.mfu).toBeLessThan(singleNode.mfu);
  });
});
