/**
 * Shared expected-changes snapshot helpers for game/RPG test files.
 *
 * Single source of truth for building TaskConfigSnapshot objects
 * and validating expectedChanges across all expected-changes tests.
 */

import { validateExpectedChanges } from '../../src/game/validation.ts';
import type { TaskConfigSnapshot } from '../../src/game/validation.ts';
import { getTaskById } from '../../src/game/tasks/index.ts';
import { getMissionById } from '../../src/rpg/missions/index.ts';
import { INFERENCE_DEFAULTS, TRAINING_DEFAULTS } from '../../src/game/defaults.ts';
import type { TaskSetup } from '../../src/game/types.ts';

// ── Base snapshot ───────────────────────────────────────────────────

/**
 * Create a full TaskConfigSnapshot with mode-appropriate defaults.
 *
 * @param mode - 'training' defaults to fp32 precision (game task convention),
 *               'inference' defaults to bf16 precision (RPG mission convention)
 */
export function makeSnapshot(
  mode: 'training' | 'inference',
  overrides: Partial<TaskConfigSnapshot> = {},
): TaskConfigSnapshot {
  return {
    modelId: 'llama3.1-8b',
    gpuId: 'a100-sxm-80gb',
    numGPUs: 1,
    precision: mode === 'training' ? 'fp32' : 'bf16',
    activationCheckpointing: false,
    checkpointingGranularity: 'full',
    flashAttention: mode === 'training' ? false : INFERENCE_DEFAULTS.flashAttention,
    globalBatchSize: 64,
    microBatchSize: mode === 'training' ? 64 : 1,
    sequenceLength: mode === 'training' ? 2048 : 1024,
    sequenceParallel: false,
    strategyType: 'ddp',
    tpDegree: 1,
    ppDegree: 1,
    epDegree: 1,
    cpDegree: 1,
    pipelineSchedule: '1f1b',
    interleavedStages: 1,
    finetuningMethod: 'full',
    loraRank: 16,
    loraTargetModules: 'q_v',
    weightPrecision: mode === 'training' ? 'fp32' : INFERENCE_DEFAULTS.weightPrecision,
    kvCachePrecision: mode === 'training' ? 'fp16' : INFERENCE_DEFAULTS.kvCachePrecision,
    batchSize: mode === 'training' ? 1 : INFERENCE_DEFAULTS.batchSize,
    inputSeqLen: mode === 'training' ? 512 : INFERENCE_DEFAULTS.inputSeqLen,
    outputSeqLen: mode === 'training' ? 128 : INFERENCE_DEFAULTS.outputSeqLen,
    tensorParallel: mode === 'training' ? 1 : INFERENCE_DEFAULTS.tensorParallel,
    expertParallel: mode === 'training' ? 1 : INFERENCE_DEFAULTS.expertParallel,
    pagedAttention: mode === 'training' ? false : INFERENCE_DEFAULTS.pagedAttention,
    continuousBatching: mode === 'training' ? false : INFERENCE_DEFAULTS.continuousBatching,
    speculativeDecoding: mode === 'training' ? false : INFERENCE_DEFAULTS.speculativeDecoding,
    pricePerGPUHour: null,
    ...overrides,
  };
}

// ── Task snapshots (game) ───────────────────────────────────────────

export function makeSnapshotForTask(taskId: string, overrides: Partial<TaskConfigSnapshot> = {}): TaskConfigSnapshot {
  const task = getTaskById(taskId);
  if (!task) throw new Error(`Task ${taskId} not found`);
  // Game tasks always use 'training' mode base defaults
  return makeSnapshot('training', overrides);
}

export function validateTask(taskId: string, snapshot: TaskConfigSnapshot, current: TaskConfigSnapshot) {
  const task = getTaskById(taskId);
  if (!task) throw new Error(`Task ${taskId} not found`);
  return validateExpectedChanges(snapshot, current, task.expectedChanges);
}

// ── Mission snapshots (RPG) ─────────────────────────────────────────

function applySetupToSnapshot(
  setup: TaskSetup,
  primaryMode: 'training' | 'inference',
  overrides: Partial<TaskConfigSnapshot>,
): TaskConfigSnapshot {
  const s = setup;

  if (primaryMode === 'training') {
    return makeSnapshot('inference', {
      modelId: s.modelId,
      gpuId: s.gpuId,
      numGPUs: s.numGPUs ?? 1,
      precision: s.mixedPrecision ?? TRAINING_DEFAULTS.mixedPrecision,
      activationCheckpointing: s.activationCheckpointing ?? TRAINING_DEFAULTS.activationCheckpointing,
      checkpointingGranularity: s.checkpointingGranularity ?? TRAINING_DEFAULTS.checkpointingGranularity,
      flashAttention: s.flashAttention ?? TRAINING_DEFAULTS.flashAttention,
      globalBatchSize: s.globalBatchSize ?? TRAINING_DEFAULTS.globalBatchSize,
      microBatchSize: s.microBatchSize ?? TRAINING_DEFAULTS.microBatchSize,
      sequenceLength: s.sequenceLength ?? TRAINING_DEFAULTS.sequenceLength,
      sequenceParallel: s.sequenceParallel ?? TRAINING_DEFAULTS.sequenceParallel,
      strategyType: s.strategyType ?? 'ddp',
      tpDegree: s.tpDegree ?? TRAINING_DEFAULTS.tpDegree,
      ppDegree: s.ppDegree ?? TRAINING_DEFAULTS.ppDegree,
      epDegree: s.epDegree ?? TRAINING_DEFAULTS.epDegree,
      cpDegree: s.cpDegree ?? TRAINING_DEFAULTS.cpDegree,
      pipelineSchedule: s.pipelineSchedule ?? TRAINING_DEFAULTS.pipelineSchedule,
      interleavedStages: s.interleavedStages ?? TRAINING_DEFAULTS.interleavedStages,
      finetuningMethod: s.finetuningMethod ?? TRAINING_DEFAULTS.finetuningMethod,
      ...overrides,
    });
  }

  // Inference mode
  return makeSnapshot('inference', {
    modelId: s.modelId,
    gpuId: s.gpuId,
    numGPUs: s.numGPUs ?? 1,
    weightPrecision: s.weightPrecision ?? INFERENCE_DEFAULTS.weightPrecision,
    kvCachePrecision: s.kvCachePrecision ?? INFERENCE_DEFAULTS.kvCachePrecision,
    batchSize: s.batchSize ?? INFERENCE_DEFAULTS.batchSize,
    inputSeqLen: s.inputSeqLen ?? INFERENCE_DEFAULTS.inputSeqLen,
    outputSeqLen: s.outputSeqLen ?? INFERENCE_DEFAULTS.outputSeqLen,
    flashAttention: s.flashAttention ?? INFERENCE_DEFAULTS.flashAttention,
    pagedAttention: s.pagedAttention ?? INFERENCE_DEFAULTS.pagedAttention,
    continuousBatching: s.continuousBatching ?? INFERENCE_DEFAULTS.continuousBatching,
    tensorParallel: s.tensorParallel ?? INFERENCE_DEFAULTS.tensorParallel,
    expertParallel: s.expertParallel ?? INFERENCE_DEFAULTS.expertParallel,
    speculativeDecoding: s.speculativeDecoding ?? INFERENCE_DEFAULTS.speculativeDecoding,
    ...overrides,
  });
}

export function makeSnapshotForMission(missionId: string, overrides: Partial<TaskConfigSnapshot> = {}): TaskConfigSnapshot {
  const mission = getMissionById(missionId);
  if (!mission) throw new Error(`Mission ${missionId} not found`);
  return applySetupToSnapshot(mission.setup, mission.primaryMode, overrides);
}

export function makeSnapshotForObjective(
  missionId: string,
  objectiveId: string,
  overrides: Partial<TaskConfigSnapshot> = {},
): TaskConfigSnapshot {
  const mission = getMissionById(missionId);
  if (!mission || !mission.objectives) throw new Error(`Mission ${missionId} has no objectives`);
  const obj = mission.objectives.find(o => o.id === objectiveId);
  if (!obj) throw new Error(`Objective ${objectiveId} not found`);
  return applySetupToSnapshot(obj.setup, obj.primaryMode, overrides);
}

export function validateMission(missionId: string, snapshot: TaskConfigSnapshot, current: TaskConfigSnapshot) {
  const mission = getMissionById(missionId);
  if (!mission) throw new Error(`Mission ${missionId} not found`);
  return validateExpectedChanges(snapshot, current, mission.expectedChanges);
}

export function validateObjective(
  missionId: string,
  objectiveId: string,
  snapshot: TaskConfigSnapshot,
  current: TaskConfigSnapshot,
) {
  const mission = getMissionById(missionId);
  if (!mission || !mission.objectives) throw new Error(`Mission ${missionId} has no objectives`);
  const obj = mission.objectives.find(o => o.id === objectiveId);
  if (!obj) throw new Error(`Objective ${objectiveId} not found`);
  return validateExpectedChanges(snapshot, current, obj.expectedChanges ?? []);
}
