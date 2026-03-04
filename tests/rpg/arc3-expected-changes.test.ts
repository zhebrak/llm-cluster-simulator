/**
 * Expected-changes tests for Arc 3 missions.
 *
 * Validates that expectedChanges correctly detect approach (lesson learned)
 * vs bypass (cheating). Philosophy: be generous — as long as the user
 * addresses the core lesson, complementary tinkering is fine.
 */

import { describe, it, expect } from 'vitest';
import {
  makeSnapshotForMission,
  makeSnapshotForObjective,
  validateMission as validate,
  validateObjective,
} from '../helpers/snapshots.ts';

// ═══════════════════════════════════════════════════════════════════════
// Mission 3-1: Landfall (lesson: continuous batching)
// expectedChanges: model/gpu/numGPUs/seqLens=unchanged (scenario locks only)
// Precision, batch size, TP, CB are unlocked — physics enforces the lesson
// (no non-CB config can reach 12,500 tok/s threshold)
// ═══════════════════════════════════════════════════════════════════════

describe('Mission 3-1 — Landfall (continuous batching)', () => {
  const snapshot = makeSnapshotForMission('mission-3-1');

  it('correct: enable continuous batching', () => {
    expect(validate('mission-3-1', snapshot,
      makeSnapshotForMission('mission-3-1', { continuousBatching: true })).valid).toBe(true);
  });
  it('correct: enable CB + change precision (unlocked, physics enforces)', () => {
    expect(validate('mission-3-1', snapshot,
      makeSnapshotForMission('mission-3-1', { continuousBatching: true, weightPrecision: 'fp4' })).valid).toBe(true);
  });
  it('correct: enable CB + change batch size (unlocked)', () => {
    expect(validate('mission-3-1', snapshot,
      makeSnapshotForMission('mission-3-1', { continuousBatching: true, batchSize: 256 })).valid).toBe(true);
  });
  it('bypass: change model', () => {
    expect(validate('mission-3-1', snapshot,
      makeSnapshotForMission('mission-3-1', { modelId: 'llama3.1-8b' })).valid).toBe(false);
  });
  it('bypass: change GPU', () => {
    expect(validate('mission-3-1', snapshot,
      makeSnapshotForMission('mission-3-1', { gpuId: 'a100-80gb' })).valid).toBe(false);
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
  it('bypass: switch to full AC without CP (checkpointing strategy locked)', () => {
    expect(validate('mission-3-2', snapshot,
      makeSnapshotForMission('mission-3-2', { checkpointingGranularity: 'full' })).valid).toBe(false);
  });
  it('bypass: switch to LoRA without CP (training method locked)', () => {
    expect(validate('mission-3-2', snapshot,
      makeSnapshotForMission('mission-3-2', { finetuningMethod: 'lora' })).valid).toBe(false);
  });
  it('bypass: switch to QLoRA without CP (training method locked)', () => {
    expect(validate('mission-3-2', snapshot,
      makeSnapshotForMission('mission-3-2', { finetuningMethod: 'qlora' })).valid).toBe(false);
  });
  it('bypass: change strategy type without CP (strategy locked)', () => {
    expect(validate('mission-3-2', snapshot,
      makeSnapshotForMission('mission-3-2', { strategyType: 'fsdp' })).valid).toBe(false);
  });
  it('correct: CP=4 + GBS change (GBS unlocked, CP satisfies required change)', () => {
    expect(validate('mission-3-2', snapshot,
      makeSnapshotForMission('mission-3-2', { cpDegree: 4, globalBatchSize: 256 })).valid).toBe(true);
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
//   model/gpu/numGPUs/PP/precision/GBS=unchanged
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
  it('bypass: change precision to FP8', () => {
    expect(validate('mission-3-4', snapshot,
      makeSnapshotForMission('mission-3-4', {
        pipelineSchedule: 'interleaved-1f1b',
        interleavedStages: 2,
        precision: 'fp8',
      })).valid).toBe(false);
  });
  it('bypass: increase GBS', () => {
    expect(validate('mission-3-4', snapshot,
      makeSnapshotForMission('mission-3-4', {
        pipelineSchedule: 'interleaved-1f1b',
        interleavedStages: 2,
        globalBatchSize: 256,
      })).valid).toBe(false);
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Mission 3-5: Resource War (multi-objective)
// Per-objective expectedChanges only.
// ═══════════════════════════════════════════════════════════════════════

describe('Mission 3-5 — Resource War (multi-objective)', () => {
  describe('Objective: Biosignature training', () => {
    const snapshot = makeSnapshotForObjective('mission-3-5', 'obj-biosig-train');

    it('correct: change precision + enable AC', () => {
      expect(validateObjective('mission-3-5', 'obj-biosig-train', snapshot,
        makeSnapshotForObjective('mission-3-5', 'obj-biosig-train', { precision: 'fp8', activationCheckpointing: true })).valid).toBe(true);
    });
    it('correct: change precision only (AC not required)', () => {
      expect(validateObjective('mission-3-5', 'obj-biosig-train', snapshot,
        makeSnapshotForObjective('mission-3-5', 'obj-biosig-train', { precision: 'fp8' })).valid).toBe(true);
    });
    it('bypass: only enable AC (precision still BF16)', () => {
      expect(validateObjective('mission-3-5', 'obj-biosig-train', snapshot,
        makeSnapshotForObjective('mission-3-5', 'obj-biosig-train', { activationCheckpointing: true })).valid).toBe(false);
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
    it('correct: increase batch without CB (passes expected changes; TTFT criterion enforces CB)', () => {
      expect(validateObjective('mission-3-5', 'obj-probe-infer', snapshot,
        makeSnapshotForObjective('mission-3-5', 'obj-probe-infer', {
          batchSize: 32,
        })).valid).toBe(true);
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
    it('bypass: CP=4 + 1F1B + GBS bump (GBS locked)', () => {
      expect(validateObjective('mission-3-6', 'obj-longctx-train', snapshot,
        makeSnapshotForObjective('mission-3-6', 'obj-longctx-train', {
          cpDegree: 4, globalBatchSize: 256,
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
    it('bypass: only change precision without EP', () => {
      expect(validateObjective('mission-3-6', 'obj-moe-train', snapshot,
        makeSnapshotForObjective('mission-3-6', 'obj-moe-train', {
          precision: 'fp8',
        })).valid).toBe(false);
    });
    it('bypass: change model', () => {
      expect(validateObjective('mission-3-6', 'obj-moe-train', snapshot,
        makeSnapshotForObjective('mission-3-6', 'obj-moe-train', {
          modelId: 'llama3.1-8b',
        })).valid).toBe(false);
    });
    it('bypass: EP + GBS bump (GBS locked)', () => {
      expect(validateObjective('mission-3-6', 'obj-moe-train', snapshot,
        makeSnapshotForObjective('mission-3-6', 'obj-moe-train', {
          epDegree: 2, globalBatchSize: 512,
        })).valid).toBe(false);
    });
    it('bypass: EP + TP reduction (TP locked)', () => {
      expect(validateObjective('mission-3-6', 'obj-moe-train', snapshot,
        makeSnapshotForObjective('mission-3-6', 'obj-moe-train', {
          epDegree: 2, tpDegree: 4,
        })).valid).toBe(false);
    });
    it('bypass: EP + seqLen increase (seqLen locked)', () => {
      expect(validateObjective('mission-3-6', 'obj-moe-train', snapshot,
        makeSnapshotForObjective('mission-3-6', 'obj-moe-train', {
          epDegree: 2, sequenceLength: 8192,
        })).valid).toBe(false);
    });
  });

  describe('Objective: Feed inference', () => {
    const snapshot = makeSnapshotForObjective('mission-3-6', 'obj-feed-infer');

    it('correct: increase batch + enable CB', () => {
      expect(validateObjective('mission-3-6', 'obj-feed-infer', snapshot,
        makeSnapshotForObjective('mission-3-6', 'obj-feed-infer', {
          batchSize: 32, continuousBatching: true,
        })).valid).toBe(true);
    });
    it('bypass: only increase batch (no CB)', () => {
      expect(validateObjective('mission-3-6', 'obj-feed-infer', snapshot,
        makeSnapshotForObjective('mission-3-6', 'obj-feed-infer', {
          batchSize: 32,
        })).valid).toBe(false);
    });
    it('bypass: only enable CB (batch still 1)', () => {
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
