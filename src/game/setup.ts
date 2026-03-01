/**
 * Config-application helpers for game/RPG modes.
 *
 * Extracted from game.ts so both useGameStore and useRPGStore can
 * configure the simulator from a TaskSetup without duplication.
 */

import type { TaskSetup } from './types.ts';
import { TRAINING_DEFAULTS, INFERENCE_DEFAULTS } from './defaults.ts';
import { useConfigStore } from '../stores/config.ts';
import type { TrainingConfig } from '../stores/config.ts';

export const VALID_STRATEGY_TYPES: Set<string> = new Set([
  'ddp', 'fsdp', 'zero-1', 'zero-3', 'auto',
  'fsdp-tp', 'zero1-tp',
  'ddp-tp-pp', 'zero1-tp-pp', 'fsdp-tp-pp',
]);

/**
 * Reset config store to naive training defaults.
 * FP32, no AC, no FA — beginner tasks need zero overrides.
 */
export function resetToTrainingDefaults(configStore: ReturnType<typeof useConfigStore.getState>): void {
  configStore.setPrecision(TRAINING_DEFAULTS.mixedPrecision);
  configStore.setSequenceLength(TRAINING_DEFAULTS.sequenceLength);
  configStore.setTrainingParams({
    activationCheckpointing: TRAINING_DEFAULTS.activationCheckpointing,
    checkpointingGranularity: TRAINING_DEFAULTS.checkpointingGranularity,
    flashAttention: TRAINING_DEFAULTS.flashAttention,
    globalBatchSize: TRAINING_DEFAULTS.globalBatchSize,
    microBatchSize: TRAINING_DEFAULTS.microBatchSize,
    sequenceParallel: TRAINING_DEFAULTS.sequenceParallel,
    finetuningMethod: TRAINING_DEFAULTS.finetuningMethod,
    loraRank: TRAINING_DEFAULTS.loraRank,
    loraTargetModules: TRAINING_DEFAULTS.loraTargetModules,
  });
  configStore.setStrategyParams({
    tpDegree: TRAINING_DEFAULTS.tpDegree,
    ppDegree: TRAINING_DEFAULTS.ppDegree,
    epDegree: TRAINING_DEFAULTS.epDegree,
    cpDegree: TRAINING_DEFAULTS.cpDegree,
    pipelineSchedule: TRAINING_DEFAULTS.pipelineSchedule,
    interleavedStages: TRAINING_DEFAULTS.interleavedStages,
  });
}

/**
 * Reset config store to reasonable inference defaults.
 * BF16, FA/PA on, no batching or advanced features.
 */
export function resetToInferenceDefaults(configStore: ReturnType<typeof useConfigStore.getState>): void {
  configStore.setInferenceParams({
    batchSize: INFERENCE_DEFAULTS.batchSize,
    inputSeqLen: INFERENCE_DEFAULTS.inputSeqLen,
    outputSeqLen: INFERENCE_DEFAULTS.outputSeqLen,
    weightPrecision: INFERENCE_DEFAULTS.weightPrecision,
    kvCachePrecision: INFERENCE_DEFAULTS.kvCachePrecision,
    flashAttention: INFERENCE_DEFAULTS.flashAttention,
    pagedAttention: INFERENCE_DEFAULTS.pagedAttention,
    continuousBatching: INFERENCE_DEFAULTS.continuousBatching,
    tensorParallel: INFERENCE_DEFAULTS.tensorParallel,
    expertParallel: INFERENCE_DEFAULTS.expertParallel,
    speculativeDecoding: INFERENCE_DEFAULTS.speculativeDecoding,
  });
}

/**
 * Apply task-specific overrides from TaskSetup onto config store.
 * Only fields explicitly set in the setup (not undefined) are applied.
 */
export function applyTaskOverrides(setup: TaskSetup, configStore: ReturnType<typeof useConfigStore.getState>, mode: 'training' | 'inference'): void {
  if (mode === 'training') {
    if (setup.mixedPrecision !== undefined) configStore.setPrecision(setup.mixedPrecision);
    if (setup.sequenceLength !== undefined) configStore.setSequenceLength(setup.sequenceLength);

    const trainingOverrides: Partial<TrainingConfig> = {};
    if (setup.activationCheckpointing !== undefined) trainingOverrides.activationCheckpointing = setup.activationCheckpointing;
    if (setup.checkpointingGranularity !== undefined) trainingOverrides.checkpointingGranularity = setup.checkpointingGranularity;
    if (setup.flashAttention !== undefined) trainingOverrides.flashAttention = setup.flashAttention;
    if (setup.globalBatchSize !== undefined) trainingOverrides.globalBatchSize = setup.globalBatchSize;
    if (setup.microBatchSize !== undefined) trainingOverrides.microBatchSize = setup.microBatchSize;
    if (setup.sequenceParallel !== undefined) trainingOverrides.sequenceParallel = setup.sequenceParallel;
    if (setup.finetuningMethod !== undefined) trainingOverrides.finetuningMethod = setup.finetuningMethod;
    if (setup.loraRank !== undefined) trainingOverrides.loraRank = setup.loraRank;
    if (setup.loraTargetModules !== undefined) trainingOverrides.loraTargetModules = setup.loraTargetModules;
    if (Object.keys(trainingOverrides).length > 0) configStore.setTrainingParams(trainingOverrides);

    const strategyOverrides: Partial<TrainingConfig> = {};
    if (setup.tpDegree !== undefined) strategyOverrides.tpDegree = setup.tpDegree;
    if (setup.ppDegree !== undefined) strategyOverrides.ppDegree = setup.ppDegree;
    if (setup.epDegree !== undefined) strategyOverrides.epDegree = setup.epDegree;
    if (setup.cpDegree !== undefined) strategyOverrides.cpDegree = setup.cpDegree;
    if (setup.pipelineSchedule !== undefined) strategyOverrides.pipelineSchedule = setup.pipelineSchedule;
    if (setup.interleavedStages !== undefined) strategyOverrides.interleavedStages = setup.interleavedStages;
    if (Object.keys(strategyOverrides).length > 0) configStore.setStrategyParams(strategyOverrides);
  } else {
    const inferenceOverrides: Record<string, unknown> = {};
    if (setup.weightPrecision !== undefined) inferenceOverrides.weightPrecision = setup.weightPrecision;
    if (setup.kvCachePrecision !== undefined) inferenceOverrides.kvCachePrecision = setup.kvCachePrecision;
    if (setup.batchSize !== undefined) inferenceOverrides.batchSize = setup.batchSize;
    if (setup.inputSeqLen !== undefined) inferenceOverrides.inputSeqLen = setup.inputSeqLen;
    if (setup.outputSeqLen !== undefined) inferenceOverrides.outputSeqLen = setup.outputSeqLen;
    if (setup.tensorParallel !== undefined) inferenceOverrides.tensorParallel = setup.tensorParallel;
    if (setup.expertParallel !== undefined) inferenceOverrides.expertParallel = setup.expertParallel;
    if (setup.pagedAttention !== undefined) inferenceOverrides.pagedAttention = setup.pagedAttention;
    if (setup.continuousBatching !== undefined) inferenceOverrides.continuousBatching = setup.continuousBatching;
    if (setup.speculativeDecoding !== undefined) inferenceOverrides.speculativeDecoding = setup.speculativeDecoding;
    if (setup.numSpeculativeTokens !== undefined) inferenceOverrides.numSpeculativeTokens = setup.numSpeculativeTokens;
    if (setup.acceptanceRate !== undefined) inferenceOverrides.acceptanceRate = setup.acceptanceRate;
    if (setup.flashAttention !== undefined) inferenceOverrides.flashAttention = setup.flashAttention;
    if (Object.keys(inferenceOverrides).length > 0) configStore.setInferenceParams(inferenceOverrides);

    // Draft model requires separate store action (resolves ModelSpec from ID)
    if (setup.draftModelId !== undefined) {
      configStore.setDraftModel(setup.draftModelId);
    }
  }
}

/**
 * Full pipeline: apply a TaskSetup to the config store.
 * Switches mode, sets model/cluster, resets to defaults, applies overrides.
 */
export function applySetupToConfig(setup: TaskSetup, mode: 'training' | 'inference'): void {
  const configStore = useConfigStore.getState();

  // Step 1: Switch mode if needed
  if (configStore.mode !== mode) {
    configStore.setMode(mode);
  }

  // Step 2: Set model and cluster
  configStore.setModel(setup.modelId);
  const numGPUs = setup.numGPUs ?? 1;
  const gpusPerNode = setup.gpusPerNode ?? Math.min(numGPUs, 8);
  configStore.setCustomCluster(setup.gpuId, numGPUs, gpusPerNode);
  configStore.setPricePerGPUHour(null);

  // Step 3: Set strategy (training only), then reset to base defaults
  if (mode === 'training') {
    if (setup.strategyType && VALID_STRATEGY_TYPES.has(setup.strategyType)) {
      configStore.setStrategy(setup.strategyType as TrainingConfig['strategyType']);
    }
    resetToTrainingDefaults(configStore);
  } else {
    resetToInferenceDefaults(configStore);
  }

  // Step 4: Apply task-specific overrides
  applyTaskOverrides(setup, configStore, mode);
}
