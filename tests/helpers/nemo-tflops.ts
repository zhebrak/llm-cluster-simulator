import { getModel } from '../../src/core/models/index.ts';

/**
 * Derive the step time implied by NeMo published TFLOPS.
 *
 * NeMo published TFLOPS use the Megatron-LM formula:
 *   TFLOPS/GPU = 4 × flopsPerToken × tokens / (stepTime_s × gpus × 1e12)
 *
 * Factor 4 = forward + backward_wgrad + backward_dgrad + recompute_forward.
 * flopsPerToken is architecture-specific (GQA, SwiGLU, seq-dependent attention).
 * Confirmed empirically: 8B pure-FSDP implied step time matches sim within 1%.
 */
export function nemoImpliedStepTimeMs(
  modelId: string,
  seqLength: number,
  globalBatchSize: number,
  totalGPUs: number,
  publishedTflopsPerGPU: number,
): number {
  const model = getModel(modelId, seqLength)!;
  const tokensPerStep = globalBatchSize * seqLength;
  const nemoFlopsPerStep = 4 * model.flopsPerToken * tokensPerStep;
  const stepTimeSec = nemoFlopsPerStep / (publishedTflopsPerGPU * totalGPUs * 1e12);
  return stepTimeSec * 1000;
}
