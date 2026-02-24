/**
 * Mixture of Experts (MoE) specific utilities
 */

import type { ModelSpec, DType } from '../../types/index.ts';
import { DTYPE_BYTES } from '../../types/index.ts';

/**
 * Calculate MoE memory breakdown
 */
export function calculateMoEMemory(
  model: ModelSpec,
  dtype: DType,
  expertParallelDegree: number = 1
): {
  expertParams: number;
  routerParams: number;
  sharedParams: number;
  totalParams: number;
  expertMemoryPerDevice: number;
  routerMemory: number;
} {
  if (!model.isMoE || !model.numExperts) {
    return {
      expertParams: 0,
      routerParams: 0,
      sharedParams: model.totalParams,
      totalParams: model.totalParams,
      expertMemoryPerDevice: 0,
      routerMemory: 0,
    };
  }

  const bytesPerParam = DTYPE_BYTES[dtype];
  const numExperts = model.numExperts;
  const numMoELayers = model.numMoELayers ?? model.numLayers;
  const hiddenSize = model.hiddenSize;
  const expertIntermediate = model.expertIntermediateSize ?? model.intermediateSize;

  // Expert parameters per layer — routed experts only (EP-distributed)
  // gated: 3 matrices (gate+up+down), non-gated: 2 matrices (up+down)
  const mlpMatrices = model.gatedMLP ? 3 : 2;
  const paramsPerExpert = mlpMatrices * hiddenSize * expertIntermediate;
  const routedExpertParams = numExperts * paramsPerExpert * numMoELayers;
  const expertParams = routedExpertParams;

  // Router parameters per layer
  const routerParamsPerLayer = hiddenSize * numExperts;
  const routerParams = routerParamsPerLayer * numMoELayers;

  // Shared parameters (embeddings, attention, norms, output, shared experts, dense layer MLPs)
  const sharedParams = model.totalParams - expertParams - routerParams;

  // Expert memory per device with expert parallelism
  const expertsPerDevice = Math.ceil(numExperts / expertParallelDegree);
  const expertMemoryPerDevice = expertsPerDevice * paramsPerExpert * numMoELayers * bytesPerParam;

  const routerMemory = routerParams * bytesPerParam;

  return {
    expertParams,
    routerParams,
    sharedParams,
    totalParams: model.totalParams,
    expertMemoryPerDevice,
    routerMemory,
  };
}
