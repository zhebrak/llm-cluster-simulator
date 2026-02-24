/**
 * Shared helper for building a StrategyContext in tests.
 */

import { getModel } from '../../src/core/models/index.ts';
import { getPresetCluster } from '../../src/core/hardware/index.ts';
import { DTYPE_PRESETS, DEFAULT_DTYPE_CONFIG, DEFAULT_ADAMW_CONFIG, DEFAULT_LR_SCHEDULE } from '../../src/types/index.ts';
import type { StrategyContext } from '../../src/core/strategies/base.ts';
import type { ClusterConfig } from '../../src/types/index.ts';

export function makeStrategyContext(
  modelId: string,
  clusterId: string,
  overrides?: {
    globalBatchSize?: number;
    microBatchSize?: number;
    seqLength?: number;
    gradientAccumulationSteps?: number;
    activationCheckpointing?: boolean;
    flashAttention?: boolean;
    dpDegree?: number;
    mixedPrecision?: string;
    clusterConfig?: ClusterConfig;
  },
): StrategyContext {
  const seqLength = overrides?.seqLength ?? 2048;
  const model = getModel(modelId, seqLength);
  const cluster = overrides?.clusterConfig ?? getPresetCluster(clusterId);

  if (!model || !cluster) {
    throw new Error(`Invalid model or cluster: ${modelId}, ${clusterId}`);
  }

  const globalBatchSize = overrides?.globalBatchSize ?? 512;
  const microBatchSize = overrides?.microBatchSize ?? 4;
  const effectiveDP = overrides?.dpDegree ?? cluster.totalGPUs;
  const gradientAccumulationSteps = overrides?.gradientAccumulationSteps ??
    Math.ceil(globalBatchSize / (microBatchSize * effectiveDP));
  const dtypes = overrides?.mixedPrecision
    ? (DTYPE_PRESETS[overrides.mixedPrecision] ?? DEFAULT_DTYPE_CONFIG)
    : DEFAULT_DTYPE_CONFIG;

  return {
    model,
    cluster,
    training: {
      globalBatchSize,
      microBatchSize,
      sequenceLength: seqLength,
      maxSteps: 1000,
      optimizer: DEFAULT_ADAMW_CONFIG,
      lrSchedule: DEFAULT_LR_SCHEDULE,
      dtypes,
      gradientClipping: 1.0,
      gradientAccumulationSteps,
    },
    seqLength,
    microBatchSize,
    globalBatchSize,
    gradientAccumulationSteps,
    activationCheckpointing: overrides?.activationCheckpointing ?? true,
    flashAttention: overrides?.flashAttention ?? true,
  };
}
