/**
 * Expected-changes tests for Arc 3 missions.
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
      epDegree: s.epDegree ?? TRAINING_DEFAULTS.epDegree,
      cpDegree: s.cpDegree ?? TRAINING_DEFAULTS.cpDegree,
      pipelineSchedule: s.pipelineSchedule ?? TRAINING_DEFAULTS.pipelineSchedule,
      interleavedStages: s.interleavedStages ?? TRAINING_DEFAULTS.interleavedStages,
      finetuningMethod: s.finetuningMethod ?? TRAINING_DEFAULTS.finetuningMethod,
      ...overrides,
    });
  }

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
      checkpointingGranularity: s.checkpointingGranularity ?? TRAINING_DEFAULTS.checkpointingGranularity,
      flashAttention: s.flashAttention ?? TRAINING_DEFAULTS.flashAttention,
      strategyType: s.strategyType ?? 'ddp',
      finetuningMethod: s.finetuningMethod ?? TRAINING_DEFAULTS.finetuningMethod,
      globalBatchSize: s.globalBatchSize ?? TRAINING_DEFAULTS.globalBatchSize,
      microBatchSize: s.microBatchSize ?? TRAINING_DEFAULTS.microBatchSize,
      sequenceLength: s.sequenceLength ?? TRAINING_DEFAULTS.sequenceLength,
      sequenceParallel: s.sequenceParallel ?? TRAINING_DEFAULTS.sequenceParallel,
      tpDegree: s.tpDegree ?? TRAINING_DEFAULTS.tpDegree,
      ppDegree: s.ppDegree ?? TRAINING_DEFAULTS.ppDegree,
      epDegree: s.epDegree ?? TRAINING_DEFAULTS.epDegree,
      cpDegree: s.cpDegree ?? TRAINING_DEFAULTS.cpDegree,
      pipelineSchedule: s.pipelineSchedule ?? TRAINING_DEFAULTS.pipelineSchedule,
      interleavedStages: s.interleavedStages ?? TRAINING_DEFAULTS.interleavedStages,
      ...overrides,
    });
  }

  return makeSnapshot({
    modelId: s.modelId,
    gpuId: s.gpuId,
    numGPUs: s.numGPUs ?? 1,
    weightPrecision: s.weightPrecision ?? INFERENCE_DEFAULTS.weightPrecision,
    kvCachePrecision: s.kvCachePrecision ?? INFERENCE_DEFAULTS.kvCachePrecision,
    batchSize: s.batchSize ?? INFERENCE_DEFAULTS.batchSize,
    inputSeqLen: s.inputSeqLen ?? INFERENCE_DEFAULTS.inputSeqLen,
    outputSeqLen: s.outputSeqLen ?? INFERENCE_DEFAULTS.outputSeqLen,
    tensorParallel: s.tensorParallel ?? INFERENCE_DEFAULTS.tensorParallel,
    expertParallel: s.expertParallel ?? INFERENCE_DEFAULTS.expertParallel,
    continuousBatching: s.continuousBatching ?? INFERENCE_DEFAULTS.continuousBatching,
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
// Mission 3-1: Landfall (lesson: continuous batching)
// expectedChanges: continuousBatching=enabled, model/gpu/numGPUs/TP/precision/batchSize/seqLens=unchanged
// ═══════════════════════════════════════════════════════════════════════

describe('Mission 3-1 — Landfall (continuous batching)', () => {
  const snapshot = makeSnapshotForMission('mission-3-1');

  it('correct: enable continuous batching', () => {
    expect(validate('mission-3-1', snapshot,
      makeSnapshotForMission('mission-3-1', { continuousBatching: true })).valid).toBe(true);
  });
  it('bypass: no changes', () => {
    expect(validate('mission-3-1', snapshot, snapshot).valid).toBe(false);
  });
  it('bypass: increase batch size without CB', () => {
    expect(validate('mission-3-1', snapshot,
      makeSnapshotForMission('mission-3-1', { batchSize: 64 })).valid).toBe(false);
  });
  it('bypass: change model', () => {
    expect(validate('mission-3-1', snapshot,
      makeSnapshotForMission('mission-3-1', { modelId: 'llama3.1-8b' })).valid).toBe(false);
  });
  it('bypass: change GPU', () => {
    expect(validate('mission-3-1', snapshot,
      makeSnapshotForMission('mission-3-1', { gpuId: 'a100-80gb' })).valid).toBe(false);
  });
  it('bypass: change precision', () => {
    expect(validate('mission-3-1', snapshot,
      makeSnapshotForMission('mission-3-1', { weightPrecision: 'fp8' })).valid).toBe(false);
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Mission 3-2: Deep Signal (lesson: context parallelism)
// expectedChanges: cpDegree=increased, model/gpu/numGPUs/seqLen/precision/TP/PP=unchanged
// ═══════════════════════════════════════════════════════════════════════

describe('Mission 3-2 — Deep Signal (context parallelism)', () => {
  const snapshot = makeSnapshotForMission('mission-3-2');

  it('correct: increase CP to 4', () => {
    expect(validate('mission-3-2', snapshot,
      makeSnapshotForMission('mission-3-2', { cpDegree: 4 })).valid).toBe(true);
  });
  it('bypass: no changes (CP=1)', () => {
    expect(validate('mission-3-2', snapshot, snapshot).valid).toBe(false);
  });
  it('bypass: change precision without CP', () => {
    expect(validate('mission-3-2', snapshot,
      makeSnapshotForMission('mission-3-2', { precision: 'bf16' })).valid).toBe(false);
  });
  it('bypass: change TP without CP', () => {
    expect(validate('mission-3-2', snapshot,
      makeSnapshotForMission('mission-3-2', { tpDegree: 4 })).valid).toBe(false);
  });
  it('bypass: change model', () => {
    expect(validate('mission-3-2', snapshot,
      makeSnapshotForMission('mission-3-2', { modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('bypass: change sequence length', () => {
    expect(validate('mission-3-2', snapshot,
      makeSnapshotForMission('mission-3-2', { sequenceLength: 8192 })).valid).toBe(false);
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Mission 3-3: Alien Model (lesson: expert parallelism)
// expectedChanges: epDegree=increased, model/gpu/numGPUs/seqLen/precision=unchanged
// TP, MBS, GBS are unlocked — the epDegree=increased requirement already forces EP.
// ═══════════════════════════════════════════════════════════════════════

describe('Mission 3-3 — Alien Model (expert parallelism)', () => {
  const snapshot = makeSnapshotForMission('mission-3-3');

  it('correct: increase EP to 8', () => {
    expect(validate('mission-3-3', snapshot,
      makeSnapshotForMission('mission-3-3', { epDegree: 8 })).valid).toBe(true);
  });
  it('correct: increase EP to 4', () => {
    expect(validate('mission-3-3', snapshot,
      makeSnapshotForMission('mission-3-3', { epDegree: 4 })).valid).toBe(true);
  });
  it('correct: EP + MBS change (MBS unlocked, EP satisfies required change)', () => {
    expect(validate('mission-3-3', snapshot,
      makeSnapshotForMission('mission-3-3', { epDegree: 8, microBatchSize: 4 })).valid).toBe(true);
  });
  it('correct: EP + TP change (TP unlocked, EP satisfies required change)', () => {
    expect(validate('mission-3-3', snapshot,
      makeSnapshotForMission('mission-3-3', { epDegree: 8, tpDegree: 4 })).valid).toBe(true);
  });
  it('bypass: no changes (EP=1)', () => {
    expect(validate('mission-3-3', snapshot, snapshot).valid).toBe(false);
  });
  it('bypass: change precision without EP', () => {
    expect(validate('mission-3-3', snapshot,
      makeSnapshotForMission('mission-3-3', { precision: 'bf16' })).valid).toBe(false);
  });
  it('bypass: increase MBS without EP (still fails — EP required)', () => {
    expect(validate('mission-3-3', snapshot,
      makeSnapshotForMission('mission-3-3', { microBatchSize: 4 })).valid).toBe(false);
  });
  it('bypass: change model', () => {
    expect(validate('mission-3-3', snapshot,
      makeSnapshotForMission('mission-3-3', { modelId: 'llama3.1-8b' })).valid).toBe(false);
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Mission 3-4: Big Train (lesson: pipeline schedule + interleaving)
// expectedChanges: pipelineSchedule=changed, interleavedStages=increased,
//   model/gpu/numGPUs/PP/seqLen=unchanged
// ═══════════════════════════════════════════════════════════════════════

describe('Mission 3-4 — Big Train (pipeline schedule)', () => {
  const snapshot = makeSnapshotForMission('mission-3-4');

  it('correct: interleaved-1f1b v=4', () => {
    expect(validate('mission-3-4', snapshot,
      makeSnapshotForMission('mission-3-4', {
        pipelineSchedule: 'interleaved-1f1b',
        interleavedStages: 4,
      })).valid).toBe(true);
  });
  it('bypass: no changes', () => {
    expect(validate('mission-3-4', snapshot, snapshot).valid).toBe(false);
  });
  it('bypass: increase GBS without changing schedule', () => {
    expect(validate('mission-3-4', snapshot,
      makeSnapshotForMission('mission-3-4', { globalBatchSize: 128 })).valid).toBe(false);
  });
  it('bypass: change PP', () => {
    expect(validate('mission-3-4', snapshot,
      makeSnapshotForMission('mission-3-4', { ppDegree: 4 })).valid).toBe(false);
  });
  it('bypass: change model', () => {
    expect(validate('mission-3-4', snapshot,
      makeSnapshotForMission('mission-3-4', { modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Mission 3-5: Resource War (multi-objective)
// Per-objective expectedChanges only.
// ═══════════════════════════════════════════════════════════════════════

describe('Mission 3-5 — Resource War (multi-objective)', () => {
  describe('Objective: Biosignature training', () => {
    const snapshot = makeSnapshotForObjective('mission-3-5', 'obj-biosig-train');

    it('correct: change precision to FP8', () => {
      expect(validateObjective('mission-3-5', 'obj-biosig-train', snapshot,
        makeSnapshotForObjective('mission-3-5', 'obj-biosig-train', { precision: 'fp8' })).valid).toBe(true);
    });
    it('bypass: no changes', () => {
      expect(validateObjective('mission-3-5', 'obj-biosig-train', snapshot, snapshot).valid).toBe(false);
    });
    it('bypass: change model', () => {
      expect(validateObjective('mission-3-5', 'obj-biosig-train', snapshot,
        makeSnapshotForObjective('mission-3-5', 'obj-biosig-train', { modelId: 'qwen3-4b' })).valid).toBe(false);
    });
  });

  describe('Objective: Fine-tune', () => {
    const snapshot = makeSnapshotForObjective('mission-3-5', 'obj-finetune');

    it('correct: change to LoRA', () => {
      expect(validateObjective('mission-3-5', 'obj-finetune', snapshot,
        makeSnapshotForObjective('mission-3-5', 'obj-finetune', { finetuningMethod: 'lora' })).valid).toBe(true);
    });
    it('correct: change to QLoRA', () => {
      expect(validateObjective('mission-3-5', 'obj-finetune', snapshot,
        makeSnapshotForObjective('mission-3-5', 'obj-finetune', { finetuningMethod: 'qlora' })).valid).toBe(true);
    });
    it('bypass: no changes', () => {
      expect(validateObjective('mission-3-5', 'obj-finetune', snapshot, snapshot).valid).toBe(false);
    });
    it('bypass: change model', () => {
      expect(validateObjective('mission-3-5', 'obj-finetune', snapshot,
        makeSnapshotForObjective('mission-3-5', 'obj-finetune', { modelId: 'llama3.1-8b' })).valid).toBe(false);
    });
  });

  describe('Objective: Probe inference', () => {
    const snapshot = makeSnapshotForObjective('mission-3-5', 'obj-probe-infer');

    it('correct: increase batch + enable CB', () => {
      expect(validateObjective('mission-3-5', 'obj-probe-infer', snapshot,
        makeSnapshotForObjective('mission-3-5', 'obj-probe-infer', {
          batchSize: 32, continuousBatching: true,
        })).valid).toBe(true);
    });
    it('bypass: enable CB without increasing batch', () => {
      expect(validateObjective('mission-3-5', 'obj-probe-infer', snapshot,
        makeSnapshotForObjective('mission-3-5', 'obj-probe-infer', {
          continuousBatching: true,
        })).valid).toBe(false);
    });
    it('bypass: increase batch without CB', () => {
      expect(validateObjective('mission-3-5', 'obj-probe-infer', snapshot,
        makeSnapshotForObjective('mission-3-5', 'obj-probe-infer', {
          batchSize: 32,
        })).valid).toBe(false);
    });
    it('bypass: change model', () => {
      expect(validateObjective('mission-3-5', 'obj-probe-infer', snapshot,
        makeSnapshotForObjective('mission-3-5', 'obj-probe-infer', {
          modelId: 'llama3.1-8b',
        })).valid).toBe(false);
    });
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Mission 3-6: All Systems Nominal (capstone, multi-objective)
// Per-objective expectedChanges.
// ═══════════════════════════════════════════════════════════════════════

describe('Mission 3-6 — All Systems Nominal (capstone)', () => {
  describe('Objective: Long-context training', () => {
    const snapshot = makeSnapshotForObjective('mission-3-6', 'obj-longctx-train');

    it('correct: increase CP + change schedule', () => {
      expect(validateObjective('mission-3-6', 'obj-longctx-train', snapshot,
        makeSnapshotForObjective('mission-3-6', 'obj-longctx-train', {
          cpDegree: 4, pipelineSchedule: 'interleaved-1f1b',
        })).valid).toBe(true);
    });
    it('bypass: only increase CP (schedule unchanged)', () => {
      expect(validateObjective('mission-3-6', 'obj-longctx-train', snapshot,
        makeSnapshotForObjective('mission-3-6', 'obj-longctx-train', {
          cpDegree: 4,
        })).valid).toBe(false);
    });
    it('bypass: only change schedule (CP unchanged)', () => {
      expect(validateObjective('mission-3-6', 'obj-longctx-train', snapshot,
        makeSnapshotForObjective('mission-3-6', 'obj-longctx-train', {
          pipelineSchedule: 'interleaved-1f1b',
        })).valid).toBe(false);
    });
    it('bypass: change model', () => {
      expect(validateObjective('mission-3-6', 'obj-longctx-train', snapshot,
        makeSnapshotForObjective('mission-3-6', 'obj-longctx-train', {
          modelId: 'llama3.3-70b',
        })).valid).toBe(false);
    });
    it('bypass: change TP', () => {
      expect(validateObjective('mission-3-6', 'obj-longctx-train', snapshot,
        makeSnapshotForObjective('mission-3-6', 'obj-longctx-train', {
          tpDegree: 4,
        })).valid).toBe(false);
    });
  });

  describe('Objective: MoE training', () => {
    const snapshot = makeSnapshotForObjective('mission-3-6', 'obj-moe-train');

    it('correct: increase EP', () => {
      expect(validateObjective('mission-3-6', 'obj-moe-train', snapshot,
        makeSnapshotForObjective('mission-3-6', 'obj-moe-train', {
          epDegree: 8,
        })).valid).toBe(true);
    });
    it('correct: increase EP + change precision', () => {
      expect(validateObjective('mission-3-6', 'obj-moe-train', snapshot,
        makeSnapshotForObjective('mission-3-6', 'obj-moe-train', {
          epDegree: 8, precision: 'fp8',
        })).valid).toBe(true);
    });
    it('bypass: no changes', () => {
      expect(validateObjective('mission-3-6', 'obj-moe-train', snapshot, snapshot).valid).toBe(false);
    });
    it('bypass: change model', () => {
      expect(validateObjective('mission-3-6', 'obj-moe-train', snapshot,
        makeSnapshotForObjective('mission-3-6', 'obj-moe-train', {
          modelId: 'llama3.1-8b',
        })).valid).toBe(false);
    });
  });

  describe('Objective: Feed inference', () => {
    const snapshot = makeSnapshotForObjective('mission-3-6', 'obj-feed-infer');

    it('correct: change precision + enable CB', () => {
      expect(validateObjective('mission-3-6', 'obj-feed-infer', snapshot,
        makeSnapshotForObjective('mission-3-6', 'obj-feed-infer', {
          weightPrecision: 'fp8', continuousBatching: true,
        })).valid).toBe(true);
    });
    it('bypass: only change precision', () => {
      expect(validateObjective('mission-3-6', 'obj-feed-infer', snapshot,
        makeSnapshotForObjective('mission-3-6', 'obj-feed-infer', {
          weightPrecision: 'fp8',
        })).valid).toBe(false);
    });
    it('bypass: only enable CB', () => {
      expect(validateObjective('mission-3-6', 'obj-feed-infer', snapshot,
        makeSnapshotForObjective('mission-3-6', 'obj-feed-infer', {
          continuousBatching: true,
        })).valid).toBe(false);
    });
    it('bypass: change model', () => {
      expect(validateObjective('mission-3-6', 'obj-feed-infer', snapshot,
        makeSnapshotForObjective('mission-3-6', 'obj-feed-infer', {
          modelId: 'llama3.1-8b',
        })).valid).toBe(false);
    });
  });

  describe('Objective: Latency inference', () => {
    const snapshot = makeSnapshotForObjective('mission-3-6', 'obj-latency-infer');

    it('correct: decrease TP to 8', () => {
      expect(validateObjective('mission-3-6', 'obj-latency-infer', snapshot,
        makeSnapshotForObjective('mission-3-6', 'obj-latency-infer', {
          tensorParallel: 8,
        })).valid).toBe(true);
    });
    it('bypass: no changes (TP=16)', () => {
      expect(validateObjective('mission-3-6', 'obj-latency-infer', snapshot, snapshot).valid).toBe(false);
    });
    it('bypass: increase TP', () => {
      expect(validateObjective('mission-3-6', 'obj-latency-infer', snapshot,
        makeSnapshotForObjective('mission-3-6', 'obj-latency-infer', {
          tensorParallel: 32,
        })).valid).toBe(false);
    });
    it('bypass: change model', () => {
      expect(validateObjective('mission-3-6', 'obj-latency-infer', snapshot,
        makeSnapshotForObjective('mission-3-6', 'obj-latency-infer', {
          modelId: 'llama3.3-70b',
        })).valid).toBe(false);
    });
  });
});
