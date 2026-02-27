/**
 * Base defaults for game mode tasks.
 *
 * Training base = "naive" state: FP32, no AC, no FA.
 * Beginner tasks 01-03 need zero overrides; intermediate/advanced tasks
 * add prerequisites (BF16, AC, FA) via their task setup overrides.
 *
 * Inference base = reasonable baseline: BF16, FA/PA on, no batching or
 * advanced features.
 */

export const TRAINING_DEFAULTS = {
  mixedPrecision: 'fp32' as const,
  activationCheckpointing: false,
  checkpointingGranularity: 'full' as const,
  flashAttention: false,
  globalBatchSize: 64,
  microBatchSize: 1,
  sequenceLength: 4096,
  sequenceParallel: false,
  finetuningMethod: 'full' as const,
  loraRank: 16,
  loraTargetModules: 'q_k_v_o' as const,
  tpDegree: 1,
  ppDegree: 1,
  epDegree: 1,
  cpDegree: 1,
  pipelineSchedule: '1f1b' as const,
  interleavedStages: 2,
} as const;

export const INFERENCE_DEFAULTS = {
  weightPrecision: 'bf16' as const,
  kvCachePrecision: 'bf16' as const,
  batchSize: 1,
  inputSeqLen: 1024,
  outputSeqLen: 256,
  flashAttention: true,
  pagedAttention: true,
  continuousBatching: false,
  tensorParallel: 1,
  expertParallel: 1,
  speculativeDecoding: false,
  draftModelId: null as string | null,
  numSpeculativeTokens: 4,
  acceptanceRate: 0.7,
} as const;

/** Common prerequisites for intermediate/advanced training tasks */
export const TRAINING_PREREQUISITES = {
  mixedPrecision: 'bf16' as const,
  flashAttention: true,
  activationCheckpointing: true,
} as const;
