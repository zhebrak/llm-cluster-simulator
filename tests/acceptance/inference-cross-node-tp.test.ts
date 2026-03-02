/**
 * Inference Cross-Node TP Tests
 *
 * Validates that TP > gpusPerNode correctly models hierarchical AllReduce
 * latency for inference, preventing the "precision cheat" where reducing
 * weight precision alone could bypass the bandwidth wall.
 *
 * Physics: When TP spans multiple nodes, AllReduce uses a 3-phase
 * hierarchical algorithm (RS-NVLink → AR-IB → AG-NVLink) with higher
 * alpha (latency per collective) and lower effective bandwidth than
 * flat intra-node NVLink.
 */

import { describe, it, expect } from 'vitest';
import { getModel } from '../../src/core/models/index.ts';
import { H100_SXM } from '../../src/core/hardware/gpu.ts';
import { createMultiNodeCluster } from '../../src/core/hardware/topology.ts';
import {
  calculateLatencyWithTP,
} from '../../src/core/inference/latency.ts';
import { runInferenceSimulation } from '../../src/core/inference/simulation.ts';
import type { InferencePrecision } from '../../src/types/inference.ts';

const llama70b = getModel('llama2-70b', 2048)!;
const gpu = H100_SXM;
const cluster2x8 = createMultiNodeCluster('h100-sxm', 8, 2)!;

describe('Inference cross-node TP', () => {
  it('TP=16 cross-node: throughput < 80 tok/s at any precision', () => {
    const precisions: InferencePrecision[] = ['bf16', 'int8', 'fp8', 'int4'];

    for (const prec of precisions) {
      const result = runInferenceSimulation({
        modelSpec: llama70b,
        gpu,
        numGPUs: 16,
        clusterConfig: cluster2x8,
        batchSize: 1,
        inputSeqLen: 512,
        outputSeqLen: 128,
        weightPrecision: prec,
        tensorParallel: 16,
      });

      expect(result.success).toBe(true);
      // Cross-node TP=16 should be slow enough that no precision saves it
      expect(result.throughput.tokensPerSecond).toBeLessThan(80);
    }
  });

  it('TP=8 intra-node with 2 replicas: throughput > 80 tok/s', () => {
    const result = runInferenceSimulation({
      modelSpec: llama70b,
      gpu,
      numGPUs: 16,
      clusterConfig: cluster2x8,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'int8',
      tensorParallel: 8,
    });

    expect(result.success).toBe(true);
    // 2 replicas × TP=8 intra-node should comfortably exceed 80 tok/s
    expect(result.numReplicas).toBe(2);
    expect(result.throughput.tokensPerSecond).toBeGreaterThan(80);
  });

  it('intra-node TP unchanged (no clusterConfig)', () => {
    // Without clusterConfig, TP=8 should use flat NVLink model
    const withCluster = calculateLatencyWithTP(
      llama70b, 512, 128, 1, gpu, 8, 'bf16', 1, undefined, true,
      8, cluster2x8.interNodeBandwidthGBps,
    );
    const withoutCluster = calculateLatencyWithTP(
      llama70b, 512, 128, 1, gpu, 8, 'bf16',
    );

    // TP=8 fits in a single node (gpusPerNode=8), so clusterConfig shouldn't matter
    expect(Math.abs(withCluster.tpot - withoutCluster.tpot) / withoutCluster.tpot).toBeLessThan(0.001);
  });

  it('intra-node TP unchanged (TP <= gpusPerNode)', () => {
    // TP=4 with gpusPerNode=8 → no cross-node penalty
    const withCluster = calculateLatencyWithTP(
      llama70b, 512, 128, 1, gpu, 4, 'bf16', 1, undefined, true,
      8, cluster2x8.interNodeBandwidthGBps,
    );
    const withoutCluster = calculateLatencyWithTP(
      llama70b, 512, 128, 1, gpu, 4, 'bf16',
    );

    expect(Math.abs(withCluster.tpot - withoutCluster.tpot) / withoutCluster.tpot).toBeLessThan(0.001);
  });

  it('non-divisible TP falls back to intra-node model', () => {
    // TP=12, gpusPerNode=8 → 12 % 8 != 0 → crossesNodes = false
    const withCluster = calculateLatencyWithTP(
      llama70b, 512, 128, 1, gpu, 12, 'bf16', 1, undefined, true,
      8, cluster2x8.interNodeBandwidthGBps,
    );
    const withoutCluster = calculateLatencyWithTP(
      llama70b, 512, 128, 1, gpu, 12, 'bf16',
    );

    // Should be identical — non-divisible TP uses flat model
    expect(Math.abs(withCluster.tpot - withoutCluster.tpot) / withoutCluster.tpot).toBeLessThan(0.001);
  });

  it('cross-node TP has higher TPOT than intra-node TP', () => {
    // TP=16 across 2 nodes should be slower than TP=8 within 1 node
    const crossNode = calculateLatencyWithTP(
      llama70b, 512, 128, 1, gpu, 16, 'bf16', 1, undefined, true,
      8, cluster2x8.interNodeBandwidthGBps,
    );
    const intraNode = calculateLatencyWithTP(
      llama70b, 512, 128, 1, gpu, 8, 'bf16',
    );

    // Cross-node TP=16 should be notably slower despite 2x more GPUs
    expect(crossNode.tpot).toBeGreaterThan(intraNode.tpot);
  });
});
