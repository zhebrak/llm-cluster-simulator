/**
 * Centralized benchmark config factory.
 *
 * Shared by test files and scripts that build SimulationConfigs
 * for published benchmark configurations.
 */

import { type SimulationConfig } from '../../src/core/simulation/engine.ts';
import { createMultiNodeCluster } from '../../src/core/hardware/topology.ts';

export function benchmarkConfig(
  gpuId: string,
  gpusPerNode: number,
  numNodes: number,
  modelId: string,
  strategy: SimulationConfig['strategyType'],
  globalBatchSize: number,
  microBatchSize: number,
  sequenceLength: number,
  strategyConfig?: SimulationConfig['strategyConfig'],
  opts?: {
    maxSteps?: number;
    mixedPrecision?: SimulationConfig['mixedPrecision'];
    activationCheckpointing?: boolean;
    checkpointingGranularity?: 'full' | 'selective';
    flashAttention?: boolean;
  },
): SimulationConfig {
  return {
    clusterConfig: createMultiNodeCluster(gpuId, gpusPerNode, numNodes)!,
    modelId,
    globalBatchSize,
    microBatchSize,
    sequenceLength,
    strategyType: strategy,
    strategyConfig,
    maxSteps: opts?.maxSteps,
    mixedPrecision: opts?.mixedPrecision,
    activationCheckpointing: opts?.activationCheckpointing,
    checkpointingGranularity: opts?.checkpointingGranularity,
    flashAttention: opts?.flashAttention,
  };
}
