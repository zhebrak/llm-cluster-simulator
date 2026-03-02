/**
 * Expected-changes tests for Arc 2 missions.
 *
 * Validates that expectedChanges correctly detect approach (lesson learned)
 * vs bypass (cheating). Philosophy: be generous — as long as the user
 * addresses the core lesson, complementary tinkering is fine.
 */

import { describe, it, expect } from 'vitest';
import { validateExpectedChanges } from '../../src/game/validation.ts';
import type { TaskConfigSnapshot } from '../../src/game/validation.ts';
import { getMissionById } from '../../src/rpg/missions/index.ts';
import { INFERENCE_DEFAULTS, TRAINING_DEFAULTS } from '../../src/game/defaults.ts';

// ── Helpers ──────────────────────────────────────────────────────────

/** Build a full TaskConfigSnapshot with defaults. */
function makeSnapshot(overrides: Partial<TaskConfigSnapshot> = {}): TaskConfigSnapshot {
  return {
    modelId: 'llama3.1-8b',
    gpuId: 'a100-sxm-80gb',
    numGPUs: 1,
    precision: 'bf16',
    activationCheckpointing: false,
    checkpointingGranularity: 'full',
    flashAttention: INFERENCE_DEFAULTS.flashAttention,
    globalBatchSize: 64,
    microBatchSize: 1,
    sequenceLength: 1024,
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
    weightPrecision: INFERENCE_DEFAULTS.weightPrecision,
    kvCachePrecision: INFERENCE_DEFAULTS.kvCachePrecision,
    batchSize: INFERENCE_DEFAULTS.batchSize,
    inputSeqLen: INFERENCE_DEFAULTS.inputSeqLen,
    outputSeqLen: INFERENCE_DEFAULTS.outputSeqLen,
    tensorParallel: INFERENCE_DEFAULTS.tensorParallel,
    expertParallel: INFERENCE_DEFAULTS.expertParallel,
    pagedAttention: INFERENCE_DEFAULTS.pagedAttention,
    continuousBatching: INFERENCE_DEFAULTS.continuousBatching,
    speculativeDecoding: INFERENCE_DEFAULTS.speculativeDecoding,
    pricePerGPUHour: null,
    ...overrides,
  };
}

/**
 * Build snapshot matching what captureTaskConfig() produces after applySetupToConfig.
 * For inference missions: merge inference defaults + setup.
 * For training missions: merge training defaults + setup.
 */
function makeSnapshotForMission(missionId: string, overrides: Partial<TaskConfigSnapshot> = {}): TaskConfigSnapshot {
  const mission = getMissionById(missionId);
  if (!mission) throw new Error(`Mission ${missionId} not found`);
  const s = mission.setup;

  if (mission.primaryMode === 'training') {
    return makeSnapshot({
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
      finetuningMethod: s.finetuningMethod ?? TRAINING_DEFAULTS.finetuningMethod,
      ...overrides,
    });
  }

  // Inference mode
  return makeSnapshot({
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

function makeSnapshotForObjective(
  missionId: string,
  objectiveId: string,
  overrides: Partial<TaskConfigSnapshot> = {},
): TaskConfigSnapshot {
  const mission = getMissionById(missionId);
  if (!mission || !mission.objectives) throw new Error(`Mission ${missionId} has no objectives`);
  const obj = mission.objectives.find(o => o.id === objectiveId);
  if (!obj) throw new Error(`Objective ${objectiveId} not found`);
  const s = obj.setup;

  if (obj.primaryMode === 'training') {
    return makeSnapshot({
      modelId: s.modelId,
      gpuId: s.gpuId,
      numGPUs: s.numGPUs ?? 1,
      precision: s.mixedPrecision ?? TRAINING_DEFAULTS.mixedPrecision,
      activationCheckpointing: s.activationCheckpointing ?? TRAINING_DEFAULTS.activationCheckpointing,
      flashAttention: s.flashAttention ?? TRAINING_DEFAULTS.flashAttention,
      strategyType: s.strategyType ?? 'ddp',
      finetuningMethod: s.finetuningMethod ?? TRAINING_DEFAULTS.finetuningMethod,
      ...overrides,
    });
  }

  return makeSnapshot({
    modelId: s.modelId,
    gpuId: s.gpuId,
    numGPUs: s.numGPUs ?? 1,
    weightPrecision: s.weightPrecision ?? INFERENCE_DEFAULTS.weightPrecision,
    tensorParallel: s.tensorParallel ?? INFERENCE_DEFAULTS.tensorParallel,
    speculativeDecoding: s.speculativeDecoding ?? INFERENCE_DEFAULTS.speculativeDecoding,
    ...overrides,
  });
}

function validate(missionId: string, snapshot: TaskConfigSnapshot, current: TaskConfigSnapshot) {
  const mission = getMissionById(missionId);
  if (!mission) throw new Error(`Mission ${missionId} not found`);
  return validateExpectedChanges(snapshot, current, mission.expectedChanges);
}

function validateObjective(
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

// ═══════════════════════════════════════════════════════════════════════
// Mission 2-1: First Light (lesson: TP for latency)
// expectedChanges: tensorParallel=increased, modelId/gpuId/numGPUs/weightPrecision=unchanged
// ═══════════════════════════════════════════════════════════════════════

describe('Mission 2-1 — First Light (TP for latency)', () => {
  const snapshot = makeSnapshotForMission('mission-2-1');

  it('correct: TP 2→4', () => {
    expect(validate('mission-2-1', snapshot,
      makeSnapshotForMission('mission-2-1', { tensorParallel: 4 })).valid).toBe(true);
  });
  it('bypass: change model', () => {
    expect(validate('mission-2-1', snapshot,
      makeSnapshotForMission('mission-2-1', { modelId: 'llama3.1-8b' })).valid).toBe(false);
  });
  it('bypass: no changes', () => {
    expect(validate('mission-2-1', snapshot, snapshot).valid).toBe(false);
  });
  it('bypass: change GPU', () => {
    expect(validate('mission-2-1', snapshot,
      makeSnapshotForMission('mission-2-1', { gpuId: 'h100-sxm' })).valid).toBe(false);
  });
  it('bypass: change precision', () => {
    expect(validate('mission-2-1', snapshot,
      makeSnapshotForMission('mission-2-1', { weightPrecision: 'int4' })).valid).toBe(false);
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Mission 2-2: The Derelict (lesson: reduce TP for replicas)
// expectedChanges: tensorParallel=decreased, modelId/gpuId/numGPUs=unchanged
// ═══════════════════════════════════════════════════════════════════════

describe('Mission 2-2 — The Derelict (replica scaling)', () => {
  const snapshot = makeSnapshotForMission('mission-2-2');

  it('correct: TP 8→2', () => {
    expect(validate('mission-2-2', snapshot,
      makeSnapshotForMission('mission-2-2', { tensorParallel: 2 })).valid).toBe(true);
  });
  it('correct: TP 8→4', () => {
    expect(validate('mission-2-2', snapshot,
      makeSnapshotForMission('mission-2-2', { tensorParallel: 4 })).valid).toBe(true);
  });
  it('bypass: no changes (still TP=8)', () => {
    expect(validate('mission-2-2', snapshot, snapshot).valid).toBe(false);
  });
  it('bypass: increase TP further', () => {
    expect(validate('mission-2-2', snapshot,
      makeSnapshotForMission('mission-2-2', { tensorParallel: 16 })).valid).toBe(false);
  });
  it('bypass: change model', () => {
    expect(validate('mission-2-2', snapshot,
      makeSnapshotForMission('mission-2-2', { modelId: 'llama3.1-8b' })).valid).toBe(false);
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Mission 2-3: Ghost Writer (lesson: speculative decoding)
// expectedChanges: speculativeDecoding=enabled, modelId/gpuId/numGPUs/tensorParallel=unchanged
// ═══════════════════════════════════════════════════════════════════════

describe('Mission 2-3 — Ghost Writer (speculative decoding)', () => {
  const snapshot = makeSnapshotForMission('mission-2-3');

  it('correct: enable speculative decoding', () => {
    expect(validate('mission-2-3', snapshot,
      makeSnapshotForMission('mission-2-3', { speculativeDecoding: true })).valid).toBe(true);
  });
  it('bypass: no changes', () => {
    expect(validate('mission-2-3', snapshot, snapshot).valid).toBe(false);
  });
  it('bypass: change model', () => {
    expect(validate('mission-2-3', snapshot,
      makeSnapshotForMission('mission-2-3', { modelId: 'llama3.1-8b' })).valid).toBe(false);
  });
  it('bypass: change GPU', () => {
    expect(validate('mission-2-3', snapshot,
      makeSnapshotForMission('mission-2-3', { gpuId: 'h100-sxm' })).valid).toBe(false);
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Mission 2-4: The Weight of Memory (lesson: precision + strategy)
// expectedChanges: precision=changed, strategyType=changed, modelId/gpuId/numGPUs=unchanged
// ═══════════════════════════════════════════════════════════════════════

describe('Mission 2-4 — The Weight of Memory (BF16 + FSDP)', () => {
  const snapshot = makeSnapshotForMission('mission-2-4');

  it('correct: BF16 + FSDP', () => {
    expect(validate('mission-2-4', snapshot,
      makeSnapshotForMission('mission-2-4', { precision: 'bf16', strategyType: 'fsdp' })).valid).toBe(true);
  });
  it('correct: FP8 + FSDP', () => {
    expect(validate('mission-2-4', snapshot,
      makeSnapshotForMission('mission-2-4', { precision: 'fp8', strategyType: 'fsdp' })).valid).toBe(true);
  });
  it('bypass: only change precision (still DDP)', () => {
    expect(validate('mission-2-4', snapshot,
      makeSnapshotForMission('mission-2-4', { precision: 'bf16' })).valid).toBe(false);
  });
  it('bypass: only change strategy (still FP32)', () => {
    expect(validate('mission-2-4', snapshot,
      makeSnapshotForMission('mission-2-4', { strategyType: 'fsdp' })).valid).toBe(false);
  });
  it('bypass: no changes', () => {
    expect(validate('mission-2-4', snapshot, snapshot).valid).toBe(false);
  });
  it('bypass: change model', () => {
    expect(validate('mission-2-4', snapshot,
      makeSnapshotForMission('mission-2-4', { modelId: 'qwen3-4b' })).valid).toBe(false);
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Mission 2-5: Activation Avalanche (lesson: activation checkpointing)
// expectedChanges: activationCheckpointing=enabled, modelId/gpuId/numGPUs/strategyType/precision=unchanged
// ═══════════════════════════════════════════════════════════════════════

describe('Mission 2-5 — Activation Avalanche (activation checkpointing)', () => {
  const snapshot = makeSnapshotForMission('mission-2-5');

  it('correct: enable AC', () => {
    expect(validate('mission-2-5', snapshot,
      makeSnapshotForMission('mission-2-5', { activationCheckpointing: true })).valid).toBe(true);
  });
  it('bypass: no changes', () => {
    expect(validate('mission-2-5', snapshot, snapshot).valid).toBe(false);
  });
  it('bypass: change strategy instead of AC', () => {
    expect(validate('mission-2-5', snapshot,
      makeSnapshotForMission('mission-2-5', { strategyType: 'ddp' })).valid).toBe(false);
  });
  it('bypass: change precision instead of AC', () => {
    expect(validate('mission-2-5', snapshot,
      makeSnapshotForMission('mission-2-5', { precision: 'fp8' })).valid).toBe(false);
  });
  it('bypass: change model', () => {
    expect(validate('mission-2-5', snapshot,
      makeSnapshotForMission('mission-2-5', { modelId: 'qwen3-4b' })).valid).toBe(false);
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Mission 2-6: The Adapter (lesson: LoRA / QLoRA)
// expectedChanges: finetuningMethod=changed, modelId/gpuId/numGPUs=unchanged
// ═══════════════════════════════════════════════════════════════════════

describe('Mission 2-6 — The Adapter (LoRA)', () => {
  const snapshot = makeSnapshotForMission('mission-2-6');

  it('correct: LoRA', () => {
    expect(validate('mission-2-6', snapshot,
      makeSnapshotForMission('mission-2-6', { finetuningMethod: 'lora' })).valid).toBe(true);
  });
  it('correct: QLoRA', () => {
    expect(validate('mission-2-6', snapshot,
      makeSnapshotForMission('mission-2-6', { finetuningMethod: 'qlora' })).valid).toBe(true);
  });
  it('bypass: no changes', () => {
    expect(validate('mission-2-6', snapshot, snapshot).valid).toBe(false);
  });
  it('bypass: change model', () => {
    expect(validate('mission-2-6', snapshot,
      makeSnapshotForMission('mission-2-6', { modelId: 'llama3.1-8b' })).valid).toBe(false);
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Mission 2-7: The Shipment (lesson: FP8)
// expectedChanges: weightPrecision=changed, modelId/gpuId/numGPUs/tensorParallel=unchanged
// ═══════════════════════════════════════════════════════════════════════

describe('Mission 2-7 — The Shipment (FP8)', () => {
  const snapshot = makeSnapshotForMission('mission-2-7');

  it('correct: FP8', () => {
    expect(validate('mission-2-7', snapshot,
      makeSnapshotForMission('mission-2-7', { weightPrecision: 'fp8' })).valid).toBe(true);
  });
  it('correct: INT8', () => {
    expect(validate('mission-2-7', snapshot,
      makeSnapshotForMission('mission-2-7', { weightPrecision: 'int8' })).valid).toBe(true);
  });
  it('bypass: no changes', () => {
    expect(validate('mission-2-7', snapshot, snapshot).valid).toBe(false);
  });
  it('bypass: change model', () => {
    expect(validate('mission-2-7', snapshot,
      makeSnapshotForMission('mission-2-7', { modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Mission 2-8: Bandwidth Wall (lesson: reduce TP + FP8 for replicas)
// expectedChanges: tensorParallel=decreased, modelId/gpuId=unchanged
// ═══════════════════════════════════════════════════════════════════════

describe('Mission 2-8 — Bandwidth Wall (network topology)', () => {
  const snapshot = makeSnapshotForMission('mission-2-8');

  it('correct: TP 16→8', () => {
    expect(validate('mission-2-8', snapshot,
      makeSnapshotForMission('mission-2-8', { tensorParallel: 8 })).valid).toBe(true);
  });
  it('correct: TP 16→8 + FP8', () => {
    expect(validate('mission-2-8', snapshot,
      makeSnapshotForMission('mission-2-8', { tensorParallel: 8, weightPrecision: 'fp8' })).valid).toBe(true);
  });
  it('bypass: no changes', () => {
    expect(validate('mission-2-8', snapshot, snapshot).valid).toBe(false);
  });
  it('bypass: change model', () => {
    expect(validate('mission-2-8', snapshot,
      makeSnapshotForMission('mission-2-8', { modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Mission 2-9: The Pipeline (lesson: pipeline parallelism)
// expectedChanges: ppDegree=increased, modelId/gpuId/numGPUs/tpDegree=unchanged
// ═══════════════════════════════════════════════════════════════════════

describe('Mission 2-9 — The Pipeline (pipeline parallelism)', () => {
  const snapshot = makeSnapshotForMission('mission-2-9');

  it('correct: PP 1→2', () => {
    expect(validate('mission-2-9', snapshot,
      makeSnapshotForMission('mission-2-9', { ppDegree: 2 })).valid).toBe(true);
  });
  it('bypass: no changes', () => {
    expect(validate('mission-2-9', snapshot, snapshot).valid).toBe(false);
  });
  it('bypass: change TP instead', () => {
    expect(validate('mission-2-9', snapshot,
      makeSnapshotForMission('mission-2-9', { tpDegree: 4 })).valid).toBe(false);
  });
  it('bypass: change model', () => {
    expect(validate('mission-2-9', snapshot,
      makeSnapshotForMission('mission-2-9', { modelId: 'llama3.1-8b' })).valid).toBe(false);
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Mission 2-10: The Protein Problem (multi-objective)
// Top-level: no expectedChanges. Per-objective: guards only.
// ═══════════════════════════════════════════════════════════════════════

describe('Mission 2-10 — The Protein Problem (multi-objective)', () => {
  describe('Training objective', () => {
    const snapshot = makeSnapshotForObjective('mission-2-10', 'obj-train');

    it('correct: change precision + enable AC', () => {
      expect(validateObjective('mission-2-10', 'obj-train', snapshot,
        makeSnapshotForObjective('mission-2-10', 'obj-train', {
          precision: 'fp8', activationCheckpointing: true,
        })).valid).toBe(true);
    });
    it('bypass: change model', () => {
      expect(validateObjective('mission-2-10', 'obj-train', snapshot,
        makeSnapshotForObjective('mission-2-10', 'obj-train', {
          modelId: 'qwen3-4b',
        })).valid).toBe(false);
    });
  });

  describe('Inference objective', () => {
    const snapshot = makeSnapshotForObjective('mission-2-10', 'obj-infer');

    it('correct: change precision + increase TP', () => {
      expect(validateObjective('mission-2-10', 'obj-infer', snapshot,
        makeSnapshotForObjective('mission-2-10', 'obj-infer', {
          weightPrecision: 'int8', tensorParallel: 4,
        })).valid).toBe(true);
    });
    it('bypass: change model', () => {
      expect(validateObjective('mission-2-10', 'obj-infer', snapshot,
        makeSnapshotForObjective('mission-2-10', 'obj-infer', {
          modelId: 'llama3.1-8b',
        })).valid).toBe(false);
    });
  });
});
