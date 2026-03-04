/**
 * Expected-changes tests for Arc 2 missions.
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
// Mission 2-7: The Shipment (lesson: FP8 as compute accelerator)
// expectedChanges: precision=changed, modelId/gpuId/numGPUs/strategyType=unchanged
// ═══════════════════════════════════════════════════════════════════════

describe('Mission 2-7 — The Shipment (FP8 training)', () => {
  const snapshot = makeSnapshotForMission('mission-2-7');

  it('correct: change precision to FP8', () => {
    expect(validate('mission-2-7', snapshot,
      makeSnapshotForMission('mission-2-7', { precision: 'fp8' })).valid).toBe(true);
  });
  it('bypass: no changes', () => {
    expect(validate('mission-2-7', snapshot, snapshot).valid).toBe(false);
  });
  it('bypass: change model', () => {
    expect(validate('mission-2-7', snapshot,
      makeSnapshotForMission('mission-2-7', { modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('bypass: change strategy', () => {
    expect(validate('mission-2-7', snapshot,
      makeSnapshotForMission('mission-2-7', { strategyType: 'ddp' })).valid).toBe(false);
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Mission 2-8: Bandwidth Wall (lesson: reduce TP for replicas)
// expectedChanges: tensorParallel=decreased, modelId/gpuId=unchanged
// ═══════════════════════════════════════════════════════════════════════

describe('Mission 2-8 — Bandwidth Wall (network topology)', () => {
  const snapshot = makeSnapshotForMission('mission-2-8');

  it('correct: TP 16→8 + batch increase', () => {
    expect(validate('mission-2-8', snapshot,
      makeSnapshotForMission('mission-2-8', { tensorParallel: 8, batchSize: 64 })).valid).toBe(true);
  });
  it('correct: TP 16→4 + batch increase', () => {
    expect(validate('mission-2-8', snapshot,
      makeSnapshotForMission('mission-2-8', { tensorParallel: 4, batchSize: 64 })).valid).toBe(true);
  });
  it('correct: TP 16→8 without batch change (batch increase optional)', () => {
    expect(validate('mission-2-8', snapshot,
      makeSnapshotForMission('mission-2-8', { tensorParallel: 8 })).valid).toBe(true);
  });
  it('bypass: no changes', () => {
    expect(validate('mission-2-8', snapshot, snapshot).valid).toBe(false);
  });
  it('bypass: batch increase only (TP unchanged)', () => {
    expect(validate('mission-2-8', snapshot,
      makeSnapshotForMission('mission-2-8', { batchSize: 128 })).valid).toBe(false);
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
  it('bypass: change TP and PP together', () => {
    expect(validate('mission-2-9', snapshot,
      makeSnapshotForMission('mission-2-9', { tpDegree: 4, ppDegree: 2 })).valid).toBe(false);
  });
  it('bypass: change model', () => {
    expect(validate('mission-2-9', snapshot,
      makeSnapshotForMission('mission-2-9', { modelId: 'llama3.1-8b' })).valid).toBe(false);
  });
  it('bypass: only increase GBS', () => {
    expect(validate('mission-2-9', snapshot,
      makeSnapshotForMission('mission-2-9', { globalBatchSize: 128 })).valid).toBe(false);
  });
  it('bypass: only change precision to FP8', () => {
    expect(validate('mission-2-9', snapshot,
      makeSnapshotForMission('mission-2-9', { precision: 'fp8' })).valid).toBe(false);
  });
  it('bypass: only increase MBS', () => {
    expect(validate('mission-2-9', snapshot,
      makeSnapshotForMission('mission-2-9', { microBatchSize: 2 })).valid).toBe(false);
  });
  it('bypass: only change sequence length', () => {
    expect(validate('mission-2-9', snapshot,
      makeSnapshotForMission('mission-2-9', { sequenceLength: 8192 })).valid).toBe(false);
  });
  it('correct + tinkering: PP increase AND also change precision', () => {
    // PP is the core change; tinkering with other fields is blocked by locks
    expect(validate('mission-2-9', snapshot,
      makeSnapshotForMission('mission-2-9', { ppDegree: 2, precision: 'fp8' })).valid).toBe(false);
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
    it('bypass: only change MBS without precision change', () => {
      expect(validateObjective('mission-2-10', 'obj-train', snapshot,
        makeSnapshotForObjective('mission-2-10', 'obj-train', {
          microBatchSize: 2,
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
