/**
 * Shared simulation helpers for game/RPG test files.
 *
 * Single source of truth for building configs, running simulations,
 * and checking criteria across calibration, uniqueness, and mission tests.
 */

import {
  SimulationEngine,
  type SimulationConfig,
} from '../../src/core/simulation/engine.ts';
import {
  runInferenceSimulation,
} from '../../src/core/inference/simulation.ts';
import {
  createMultiNodeCluster,
  createSingleNodeCluster,
} from '../../src/core/hardware/index.ts';
import { evaluateCriterion, validateExpectedChanges } from '../../src/game/validation.ts';
import type { TaskConfigSnapshot } from '../../src/game/validation.ts';
import { getGPUHourlyRate, calculateCostPerMillionTokens } from '../../src/core/cost/cloud.ts';
import { TRAINING_DEFAULTS, INFERENCE_DEFAULTS } from '../../src/game/defaults.ts';
import type { TaskSetup, WinningCriterion } from '../../src/game/types.ts';

// ── Cluster ─────────────────────────────────────────────────────────

export function makeCluster(gpuId: string, numGPUs: number, gpusPerNode: number) {
  const numNodes = Math.ceil(numGPUs / gpusPerNode);
  if (numNodes === 1) return createSingleNodeCluster(gpuId, numGPUs);
  return createMultiNodeCluster(gpuId, gpusPerNode, numNodes);
}

// ── Config builders ─────────────────────────────────────────────────

export function buildEffectiveTrainingConfig(setup: TaskSetup) {
  return {
    modelId: setup.modelId,
    gpuId: setup.gpuId,
    numGPUs: setup.numGPUs ?? 1,
    gpusPerNode: setup.gpusPerNode ?? Math.min(setup.numGPUs ?? 1, 8),
    strategyType: setup.strategyType ?? 'ddp',
    mixedPrecision: setup.mixedPrecision ?? TRAINING_DEFAULTS.mixedPrecision,
    sequenceLength: setup.sequenceLength ?? TRAINING_DEFAULTS.sequenceLength,
    activationCheckpointing: setup.activationCheckpointing ?? TRAINING_DEFAULTS.activationCheckpointing,
    checkpointingGranularity: setup.checkpointingGranularity ?? TRAINING_DEFAULTS.checkpointingGranularity,
    flashAttention: setup.flashAttention ?? TRAINING_DEFAULTS.flashAttention,
    globalBatchSize: setup.globalBatchSize ?? TRAINING_DEFAULTS.globalBatchSize,
    microBatchSize: setup.microBatchSize ?? TRAINING_DEFAULTS.microBatchSize,
    sequenceParallel: setup.sequenceParallel ?? TRAINING_DEFAULTS.sequenceParallel,
    finetuningMethod: setup.finetuningMethod ?? TRAINING_DEFAULTS.finetuningMethod,
    loraRank: setup.loraRank ?? TRAINING_DEFAULTS.loraRank,
    loraTargetModules: setup.loraTargetModules ?? TRAINING_DEFAULTS.loraTargetModules,
    tpDegree: setup.tpDegree ?? TRAINING_DEFAULTS.tpDegree,
    ppDegree: setup.ppDegree ?? TRAINING_DEFAULTS.ppDegree,
    epDegree: setup.epDegree ?? TRAINING_DEFAULTS.epDegree,
    cpDegree: setup.cpDegree ?? TRAINING_DEFAULTS.cpDegree,
    pipelineSchedule: setup.pipelineSchedule ?? TRAINING_DEFAULTS.pipelineSchedule,
    interleavedStages: setup.interleavedStages ?? TRAINING_DEFAULTS.interleavedStages,
  };
}

export function buildEffectiveInferenceConfig(setup: TaskSetup) {
  return {
    modelId: setup.modelId,
    gpuId: setup.gpuId,
    numGPUs: setup.numGPUs ?? 1,
    gpusPerNode: setup.gpusPerNode ?? Math.min(setup.numGPUs ?? 1, 8),
    weightPrecision: setup.weightPrecision ?? INFERENCE_DEFAULTS.weightPrecision,
    kvCachePrecision: setup.kvCachePrecision ?? INFERENCE_DEFAULTS.kvCachePrecision,
    batchSize: setup.batchSize ?? INFERENCE_DEFAULTS.batchSize,
    inputSeqLen: setup.inputSeqLen ?? INFERENCE_DEFAULTS.inputSeqLen,
    outputSeqLen: setup.outputSeqLen ?? INFERENCE_DEFAULTS.outputSeqLen,
    flashAttention: setup.flashAttention ?? INFERENCE_DEFAULTS.flashAttention,
    pagedAttention: setup.pagedAttention ?? INFERENCE_DEFAULTS.pagedAttention,
    continuousBatching: setup.continuousBatching ?? INFERENCE_DEFAULTS.continuousBatching,
    tensorParallel: setup.tensorParallel ?? INFERENCE_DEFAULTS.tensorParallel,
    expertParallel: setup.expertParallel ?? INFERENCE_DEFAULTS.expertParallel,
    speculativeDecoding: setup.speculativeDecoding ?? INFERENCE_DEFAULTS.speculativeDecoding,
    draftModelId: setup.draftModelId ?? INFERENCE_DEFAULTS.draftModelId,
    numSpeculativeTokens: setup.numSpeculativeTokens ?? INFERENCE_DEFAULTS.numSpeculativeTokens,
    acceptanceRate: setup.acceptanceRate ?? INFERENCE_DEFAULTS.acceptanceRate,
  };
}

// ── Simulation runners ──────────────────────────────────────────────

export function runTraining(cfg: ReturnType<typeof buildEffectiveTrainingConfig>) {
  const cluster = makeCluster(cfg.gpuId, cfg.numGPUs, cfg.gpusPerNode);
  const engine = new SimulationEngine();

  const strategyConfig: Record<string, unknown> = {};
  strategyConfig.tp = cfg.tpDegree;
  strategyConfig.pp = cfg.ppDegree;
  strategyConfig.ep = cfg.epDegree;
  strategyConfig.cp = cfg.cpDegree;
  strategyConfig.sequenceParallel = cfg.sequenceParallel;
  if (cfg.pipelineSchedule !== '1f1b') strategyConfig.pipelineSchedule = cfg.pipelineSchedule;
  if (cfg.interleavedStages !== 2) strategyConfig.interleavedStages = cfg.interleavedStages;

  const config: SimulationConfig = {
    modelId: cfg.modelId,
    clusterConfig: cluster,
    sequenceLength: cfg.sequenceLength,
    globalBatchSize: cfg.globalBatchSize,
    microBatchSize: cfg.microBatchSize,
    strategyType: cfg.strategyType as SimulationConfig['strategyType'],
    strategyConfig: Object.keys(strategyConfig).length > 0 ? strategyConfig : undefined,
    activationCheckpointing: cfg.activationCheckpointing,
    checkpointingGranularity: cfg.checkpointingGranularity,
    flashAttention: cfg.flashAttention,
    mixedPrecision: cfg.mixedPrecision as SimulationConfig['mixedPrecision'],
    finetuningMethod: cfg.finetuningMethod as SimulationConfig['finetuningMethod'],
    loraRank: cfg.loraRank,
    loraTargetModules: cfg.loraTargetModules as SimulationConfig['loraTargetModules'],
  };
  engine.configure(config);
  return engine.simulate();
}

export function runInference(cfg: ReturnType<typeof buildEffectiveInferenceConfig>) {
  return runInferenceSimulation({
    modelId: cfg.modelId,
    gpuId: cfg.gpuId,
    numGPUs: cfg.numGPUs,
    batchSize: cfg.batchSize,
    inputSeqLen: cfg.inputSeqLen,
    outputSeqLen: cfg.outputSeqLen,
    weightPrecision: cfg.weightPrecision,
    kvCachePrecision: cfg.kvCachePrecision,
    flashAttention: cfg.flashAttention,
    pagedAttention: cfg.pagedAttention,
    continuousBatching: cfg.continuousBatching,
    tensorParallel: cfg.tensorParallel,
    expertParallel: cfg.expertParallel,
    speculativeEnabled: cfg.speculativeDecoding,
    draftModelId: cfg.draftModelId ?? undefined,
    numSpeculativeTokens: cfg.numSpeculativeTokens,
    acceptanceRate: cfg.acceptanceRate,
  });
}

// ── Context builders ────────────────────────────────────────────────

export function buildTrainingContext(setup: TaskSetup) {
  const cfg = buildEffectiveTrainingConfig(setup);
  const metrics = runTraining(cfg);
  return { success: metrics.memoryUtilization <= 1.0, gpuId: cfg.gpuId, ...metrics };
}

export function buildTrainingContextWith(setup: TaskSetup, overrides: Partial<TaskSetup>) {
  return buildTrainingContext({ ...setup, ...overrides });
}

export function buildInferenceContext(setup: TaskSetup) {
  const cfg = buildEffectiveInferenceConfig(setup);
  const result = runInference(cfg);
  if (!result) return null;
  const rate = getGPUHourlyRate(cfg.gpuId).rate;
  const memUtil = result.utilization?.memoryCapacityUtilization ?? 0;
  return {
    success: memUtil <= 1.0,
    ...result,
    numGPUs: cfg.numGPUs,
    memoryUtilization: memUtil,
    costPerMillionTokens: calculateCostPerMillionTokens(
      rate, cfg.numGPUs, result.throughput?.tokensPerSecond ?? 0,
    ),
  };
}

export function buildInferenceContextWith(setup: TaskSetup, overrides: Partial<TaskSetup>) {
  return buildInferenceContext({ ...setup, ...overrides });
}

// ── Snapshot builder ────────────────────────────────────────────────

/** Build TaskConfigSnapshot from setup + defaults (mirrors captureTaskConfig() in the app).
 *  Uses Required<> so missing fields cause a compile error if TaskConfigSnapshot grows. */
export function buildSnapshot(setup: TaskSetup, mode: 'training' | 'inference'): Required<TaskConfigSnapshot> {
  const t = buildEffectiveTrainingConfig(setup);
  const i = buildEffectiveInferenceConfig(setup);
  return {
    modelId: setup.modelId,
    gpuId: setup.gpuId,
    numGPUs: setup.numGPUs ?? 1,
    precision: t.mixedPrecision,
    activationCheckpointing: t.activationCheckpointing,
    checkpointingGranularity: t.checkpointingGranularity,
    flashAttention: mode === 'training' ? t.flashAttention : i.flashAttention,
    globalBatchSize: t.globalBatchSize,
    microBatchSize: t.microBatchSize,
    sequenceLength: t.sequenceLength,
    sequenceParallel: t.sequenceParallel,
    strategyType: t.strategyType,
    tpDegree: t.tpDegree,
    ppDegree: t.ppDegree,
    epDegree: t.epDegree,
    cpDegree: t.cpDegree,
    pipelineSchedule: t.pipelineSchedule,
    interleavedStages: t.interleavedStages,
    finetuningMethod: t.finetuningMethod,
    loraRank: t.loraRank,
    loraTargetModules: t.loraTargetModules,
    weightPrecision: i.weightPrecision,
    kvCachePrecision: i.kvCachePrecision,
    batchSize: i.batchSize,
    inputSeqLen: i.inputSeqLen,
    outputSeqLen: i.outputSeqLen,
    tensorParallel: i.tensorParallel,
    expertParallel: i.expertParallel,
    pagedAttention: i.pagedAttention,
    continuousBatching: i.continuousBatching,
    speculativeDecoding: i.speculativeDecoding,
    pricePerGPUHour: null,
  };
}

// ── Criteria helpers ────────────────────────────────────────────────

export function checkAllCriteria(ctx: object | null, criteria: WinningCriterion[]): boolean {
  if (!ctx) return false;
  return criteria.every(c => evaluateCriterion(ctx, c));
}

export function checkAnyCriterionFails(ctx: object | null, criteria: WinningCriterion[]): boolean {
  if (!ctx) return true;
  return criteria.some(c => !evaluateCriterion(ctx, c));
}
