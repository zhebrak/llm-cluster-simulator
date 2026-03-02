/**
 * Expected-changes tests for RPG missions 1-1 through 1-6.
 *
 * Validates that expectedChanges correctly detect approach (lesson learned)
 * vs bypass (cheating). Philosophy: be generous — as long as the user
 * addresses the core lesson, complementary tinkering is fine.
 */

import { describe, it, expect } from 'vitest';
import { validateExpectedChanges } from '../../src/game/validation.ts';
import type { TaskConfigSnapshot } from '../../src/game/validation.ts';
import { getMissionById } from '../../src/rpg/missions/index.ts';
import { INFERENCE_DEFAULTS } from '../../src/game/defaults.ts';
import type { RPGMission } from '../../src/rpg/types.ts';

// ── Helpers ──────────────────────────────────────────────────────────

/** Build a full TaskConfigSnapshot with inference-mode defaults. */
function makeSnapshot(overrides: Partial<TaskConfigSnapshot> = {}): TaskConfigSnapshot {
  return {
    // Hardware
    modelId: 'llama3.1-8b',
    gpuId: 'a100-sxm-80gb',
    numGPUs: 1,
    // Training (type-required but irrelevant for inference missions)
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
    // Inference (from INFERENCE_DEFAULTS)
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
 * Merges inference defaults with the mission's setup, then applies overrides.
 */
function makeSnapshotForMission(missionId: string, overrides: Partial<TaskConfigSnapshot> = {}): TaskConfigSnapshot {
  const mission = getMissionById(missionId);
  if (!mission) throw new Error(`Mission ${missionId} not found`);
  const s = mission.setup;
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

/** Validate expectedChanges for a mission. Returns { valid, failedChecks }. */
function validate(missionId: string, snapshot: TaskConfigSnapshot, current: TaskConfigSnapshot) {
  const mission = getMissionById(missionId);
  if (!mission) throw new Error(`Mission ${missionId} not found`);
  return validateExpectedChanges(snapshot, current, mission.expectedChanges);
}

// ═════════════════════════════════════════════════════════════════════
// Mission 1-1: Wake-Up Call (lesson: quantization)
// expectedChanges: weightPrecision=changed, modelId=unchanged, gpuId=unchanged
// ═════════════════════════════════════════════════════════════════════

describe('Mission 1-1 — Wake-Up Call (quantization)', () => {
  const snapshot = makeSnapshotForMission('mission-1-1');

  it('correct: BF16→INT8', () => {
    expect(validate('mission-1-1', snapshot,
      makeSnapshotForMission('mission-1-1', { weightPrecision: 'int8' })).valid).toBe(true);
  });
  it('correct: BF16→INT4', () => {
    expect(validate('mission-1-1', snapshot,
      makeSnapshotForMission('mission-1-1', { weightPrecision: 'int4' })).valid).toBe(true);
  });
  it('correct: BF16→FP8', () => {
    expect(validate('mission-1-1', snapshot,
      makeSnapshotForMission('mission-1-1', { weightPrecision: 'fp8' })).valid).toBe(true);
  });
  it('correct + tinkering: change precision AND tweak batch size', () => {
    expect(validate('mission-1-1', snapshot,
      makeSnapshotForMission('mission-1-1', { weightPrecision: 'int8', batchSize: 4 })).valid).toBe(true);
  });
  it('correct + tinkering: change precision AND change KV cache', () => {
    expect(validate('mission-1-1', snapshot,
      makeSnapshotForMission('mission-1-1', { weightPrecision: 'int4', kvCachePrecision: 'fp8' })).valid).toBe(true);
  });
  it('bypass: only change model (did not learn quantization)', () => {
    expect(validate('mission-1-1', snapshot,
      makeSnapshotForMission('mission-1-1', { modelId: 'qwen3-4b' })).valid).toBe(false);
  });
  it('bypass: only change GPU', () => {
    expect(validate('mission-1-1', snapshot,
      makeSnapshotForMission('mission-1-1', { gpuId: 'a100-sxm-80gb' })).valid).toBe(false);
  });
  it('bypass: no changes at all', () => {
    expect(validate('mission-1-1', snapshot, snapshot).valid).toBe(false);
  });
  it('bypass: increase numGPUs instead of learning quantization', () => {
    expect(validate('mission-1-1', snapshot,
      makeSnapshotForMission('mission-1-1', { numGPUs: 4 })).valid).toBe(false);
  });
});

// ═════════════════════════════════════════════════════════════════════
// Mission 1-2: The Upgrade (lesson: model selection)
// expectedChanges: modelId=changed, gpuId=unchanged
// ═════════════════════════════════════════════════════════════════════

describe('Mission 1-2 — The Upgrade (model selection)', () => {
  const snapshot = makeSnapshotForMission('mission-1-2');

  it('correct: switch to 8B model', () => {
    expect(validate('mission-1-2', snapshot,
      makeSnapshotForMission('mission-1-2', { modelId: 'llama3.1-8b' })).valid).toBe(true);
  });
  it('correct: switch to any smaller model', () => {
    expect(validate('mission-1-2', snapshot,
      makeSnapshotForMission('mission-1-2', { modelId: 'qwen3-14b' })).valid).toBe(true);
  });
  it('correct + tinkering: switch model AND also quantize', () => {
    expect(validate('mission-1-2', snapshot,
      makeSnapshotForMission('mission-1-2', { modelId: 'llama3.1-8b', weightPrecision: 'int4' })).valid).toBe(true);
  });
  it('correct + tinkering: switch model AND change batch size', () => {
    expect(validate('mission-1-2', snapshot,
      makeSnapshotForMission('mission-1-2', { modelId: 'llama3.1-8b', batchSize: 8 })).valid).toBe(true);
  });
  it('bypass: only change GPU (did not learn model selection)', () => {
    expect(validate('mission-1-2', snapshot,
      makeSnapshotForMission('mission-1-2', { gpuId: 'h100-sxm' })).valid).toBe(false);
  });
  it('bypass: only quantize without model change', () => {
    expect(validate('mission-1-2', snapshot,
      makeSnapshotForMission('mission-1-2', { weightPrecision: 'int4' })).valid).toBe(false);
  });
  it('bypass: no changes at all', () => {
    expect(validate('mission-1-2', snapshot, snapshot).valid).toBe(false);
  });
  it('bypass: increase numGPUs instead of learning model selection', () => {
    expect(validate('mission-1-2', snapshot,
      makeSnapshotForMission('mission-1-2', { numGPUs: 4 })).valid).toBe(false);
  });
});

// ═════════════════════════════════════════════════════════════════════
// Mission 1-3: Slow Reflexes (lesson: GPU bandwidth)
// expectedChanges: gpuId=changed, modelId=unchanged
// ═════════════════════════════════════════════════════════════════════

describe('Mission 1-3 — Slow Reflexes (GPU bandwidth)', () => {
  const snapshot = makeSnapshotForMission('mission-1-3');

  it('correct: switch to RTX 4090', () => {
    expect(validate('mission-1-3', snapshot,
      makeSnapshotForMission('mission-1-3', { gpuId: 'rtx-4090' })).valid).toBe(true);
  });
  it('correct: switch to A100', () => {
    expect(validate('mission-1-3', snapshot,
      makeSnapshotForMission('mission-1-3', { gpuId: 'a100-sxm-80gb' })).valid).toBe(true);
  });
  it('correct: switch to H100', () => {
    expect(validate('mission-1-3', snapshot,
      makeSnapshotForMission('mission-1-3', { gpuId: 'h100-sxm' })).valid).toBe(true);
  });
  it('correct + tinkering: change GPU AND also change precision', () => {
    expect(validate('mission-1-3', snapshot,
      makeSnapshotForMission('mission-1-3', { gpuId: 'rtx-4090', weightPrecision: 'int4' })).valid).toBe(true);
  });
  it('bypass: only change model (did not learn bandwidth matters)', () => {
    expect(validate('mission-1-3', snapshot,
      makeSnapshotForMission('mission-1-3', { modelId: 'qwen3-4b' })).valid).toBe(false);
  });
  it('bypass: only quantize without GPU change', () => {
    expect(validate('mission-1-3', snapshot,
      makeSnapshotForMission('mission-1-3', { weightPrecision: 'int4' })).valid).toBe(false);
  });
  it('bypass: no changes at all', () => {
    expect(validate('mission-1-3', snapshot, snapshot).valid).toBe(false);
  });
  it('bypass: increase numGPUs instead of learning bandwidth', () => {
    expect(validate('mission-1-3', snapshot,
      makeSnapshotForMission('mission-1-3', { numGPUs: 4 })).valid).toBe(false);
  });
});

// ═════════════════════════════════════════════════════════════════════
// Mission 1-4: Cryo-Pod Monitoring (lesson: batching)
// expectedChanges: batchSize=increased, modelId=unchanged, gpuId=unchanged
// ═════════════════════════════════════════════════════════════════════

describe('Mission 1-4 — Cryo-Pod Monitoring (batching)', () => {
  const snapshot = makeSnapshotForMission('mission-1-4');

  it('correct: batch=2 (minimum increase)', () => {
    expect(validate('mission-1-4', snapshot,
      makeSnapshotForMission('mission-1-4', { batchSize: 2 })).valid).toBe(true);
  });
  it('correct: batch=16', () => {
    expect(validate('mission-1-4', snapshot,
      makeSnapshotForMission('mission-1-4', { batchSize: 16 })).valid).toBe(true);
  });
  it('correct: batch=64', () => {
    expect(validate('mission-1-4', snapshot,
      makeSnapshotForMission('mission-1-4', { batchSize: 64 })).valid).toBe(true);
  });
  it('correct + tinkering: increase batch AND enable continuous batching', () => {
    expect(validate('mission-1-4', snapshot,
      makeSnapshotForMission('mission-1-4', { batchSize: 16, continuousBatching: true })).valid).toBe(true);
  });
  it('correct + tinkering: increase batch AND change precision', () => {
    expect(validate('mission-1-4', snapshot,
      makeSnapshotForMission('mission-1-4', { batchSize: 8, weightPrecision: 'int4' })).valid).toBe(true);
  });
  it('bypass: only change GPU (did not learn batching)', () => {
    expect(validate('mission-1-4', snapshot,
      makeSnapshotForMission('mission-1-4', { gpuId: 'h100-sxm' })).valid).toBe(false);
  });
  it('bypass: only change model', () => {
    expect(validate('mission-1-4', snapshot,
      makeSnapshotForMission('mission-1-4', { modelId: 'qwen3-4b' })).valid).toBe(false);
  });
  it('bypass: no changes at all', () => {
    expect(validate('mission-1-4', snapshot, snapshot).valid).toBe(false);
  });
  it('bypass: increase numGPUs instead of learning batching', () => {
    expect(validate('mission-1-4', snapshot,
      makeSnapshotForMission('mission-1-4', { numGPUs: 4 })).valid).toBe(false);
  });
});

// ═════════════════════════════════════════════════════════════════════
// Mission 1-5: Memory Leak (lesson: FA/KV cache — deliberately permissive)
//
// expectedChanges: modelId=unchanged, gpuId=unchanged (guards only)
// Three valid levers (FA, KV quant, paged attention) — mandating one
// would be unfair, so only hardware/model changes are blocked.
// ═════════════════════════════════════════════════════════════════════

describe('Mission 1-5 — Memory Leak (FA/KV cache, permissive)', () => {
  const snapshot = makeSnapshotForMission('mission-1-5');

  it('correct: enable Flash Attention', () => {
    expect(validate('mission-1-5', snapshot,
      makeSnapshotForMission('mission-1-5', { flashAttention: true })).valid).toBe(true);
  });
  it('correct: quantize KV cache', () => {
    expect(validate('mission-1-5', snapshot,
      makeSnapshotForMission('mission-1-5', { kvCachePrecision: 'int8' })).valid).toBe(true);
  });
  it('correct: combine FA + KV quant', () => {
    expect(validate('mission-1-5', snapshot,
      makeSnapshotForMission('mission-1-5', {
        flashAttention: true, kvCachePrecision: 'fp8',
      })).valid).toBe(true);
  });
  it('correct: even no technique changes passes expectedChanges (winning criteria rejects OOM separately)', () => {
    // expectedChanges only guard model/GPU — no required technique
    expect(validate('mission-1-5', snapshot, snapshot).valid).toBe(true);
  });
  it('bypass: only change model', () => {
    expect(validate('mission-1-5', snapshot,
      makeSnapshotForMission('mission-1-5', { modelId: 'qwen3-4b' })).valid).toBe(false);
  });
  it('bypass: only change GPU', () => {
    expect(validate('mission-1-5', snapshot,
      makeSnapshotForMission('mission-1-5', { gpuId: 'h100-sxm' })).valid).toBe(false);
  });
  it('bypass: increase numGPUs instead of learning FA/KV cache', () => {
    expect(validate('mission-1-5', snapshot,
      makeSnapshotForMission('mission-1-5', { numGPUs: 4 })).valid).toBe(false);
  });
});

// ═════════════════════════════════════════════════════════════════════
// Mission 1-6: The Archive Vault (lesson: tensor parallelism)
// expectedChanges: tensorParallel=increased, modelId=unchanged, gpuId=unchanged
// ═════════════════════════════════════════════════════════════════════

describe('Mission 1-6 — The Archive Vault (tensor parallelism)', () => {
  const snapshot = makeSnapshotForMission('mission-1-6');

  it('correct: TP=2', () => {
    expect(validate('mission-1-6', snapshot,
      makeSnapshotForMission('mission-1-6', { tensorParallel: 2 })).valid).toBe(true);
  });
  it('correct: TP=4', () => {
    expect(validate('mission-1-6', snapshot,
      makeSnapshotForMission('mission-1-6', { tensorParallel: 4 })).valid).toBe(true);
  });
  it('correct + tinkering: TP=4 AND also change precision', () => {
    expect(validate('mission-1-6', snapshot,
      makeSnapshotForMission('mission-1-6', { tensorParallel: 4, weightPrecision: 'int8' })).valid).toBe(true);
  });
  it('correct + tinkering: TP=2 AND change batch size', () => {
    expect(validate('mission-1-6', snapshot,
      makeSnapshotForMission('mission-1-6', { tensorParallel: 2, batchSize: 4 })).valid).toBe(true);
  });
  it('bypass: only change model (did not learn TP)', () => {
    expect(validate('mission-1-6', snapshot,
      makeSnapshotForMission('mission-1-6', { modelId: 'llama3.1-8b' })).valid).toBe(false);
  });
  it('bypass: only quantize without TP', () => {
    expect(validate('mission-1-6', snapshot,
      makeSnapshotForMission('mission-1-6', { weightPrecision: 'int4' })).valid).toBe(false);
  });
  it('bypass: no changes at all', () => {
    expect(validate('mission-1-6', snapshot, snapshot).valid).toBe(false);
  });
  it('bypass: increase numGPUs instead of learning TP', () => {
    expect(validate('mission-1-6', snapshot,
      makeSnapshotForMission('mission-1-6', { numGPUs: 8 })).valid).toBe(false);
  });
});

// ═════════════════════════════════════════════════════════════════════
// Mission 1-7: Fuel Budget (lesson: cost optimization via quantization + fewer GPUs)
// expectedChanges: numGPUs=decreased, modelId=unchanged, gpuId=unchanged, pricePerGPUHour=unchanged
// ═════════════════════════════════════════════════════════════════════

describe('Mission 1-7 — Fuel Budget (cost optimization)', () => {
  const snapshot = makeSnapshotForMission('mission-1-7');

  it('correct: INT8 + TP=2 + 2 GPUs', () => {
    expect(validate('mission-1-7', snapshot,
      makeSnapshotForMission('mission-1-7', { numGPUs: 2, tensorParallel: 2, weightPrecision: 'int8' })).valid).toBe(true);
  });
  it('correct: INT4 + single GPU', () => {
    expect(validate('mission-1-7', snapshot,
      makeSnapshotForMission('mission-1-7', { numGPUs: 1, tensorParallel: 1, weightPrecision: 'int4' })).valid).toBe(true);
  });
  it('bypass: increase numGPUs', () => {
    expect(validate('mission-1-7', snapshot,
      makeSnapshotForMission('mission-1-7', { numGPUs: 8 })).valid).toBe(false);
  });
  it('bypass: change model', () => {
    expect(validate('mission-1-7', snapshot,
      makeSnapshotForMission('mission-1-7', { modelId: 'llama3.1-8b' })).valid).toBe(false);
  });
  it('bypass: change GPU', () => {
    expect(validate('mission-1-7', snapshot,
      makeSnapshotForMission('mission-1-7', { gpuId: 'h100-sxm' })).valid).toBe(false);
  });
  it('bypass: no changes at all', () => {
    expect(validate('mission-1-7', snapshot, snapshot).valid).toBe(false);
  });
});
