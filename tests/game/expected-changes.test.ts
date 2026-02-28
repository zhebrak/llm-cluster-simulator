/**
 * Integration tests for expectedChanges — verifies generosity and correctness
 * of approach validation for each task category.
 *
 * Design principle: "Be generous." Only block when the user clearly bypassed the
 * intended technique. Legitimate complementary changes should always be allowed.
 */

import { describe, it, expect } from 'vitest';
import { validateExpectedChanges } from '../../src/game/validation.ts';
import type { TaskConfigSnapshot } from '../../src/game/validation.ts';
import { getTaskById } from '../../src/game/tasks/index.ts';

/** Create a base snapshot with typical training defaults */
function makeSnapshot(overrides: Partial<TaskConfigSnapshot> = {}): TaskConfigSnapshot {
  return {
    modelId: 'llama3.1-8b', gpuId: 'a100-sxm-80gb', numGPUs: 1,
    precision: 'fp32', activationCheckpointing: false, checkpointingGranularity: 'full',
    flashAttention: false, globalBatchSize: 64, microBatchSize: 64,
    sequenceLength: 2048, sequenceParallel: false, strategyType: 'ddp',
    tpDegree: 1, ppDegree: 1, epDegree: 1, cpDegree: 1,
    pipelineSchedule: '1f1b', interleavedStages: 1, finetuningMethod: 'full',
    loraRank: 16, loraTargetModules: 'q_v',
    weightPrecision: 'fp32', kvCachePrecision: 'fp16', batchSize: 1,
    inputSeqLen: 512, outputSeqLen: 128, tensorParallel: 1, expertParallel: 1,
    pagedAttention: false, continuousBatching: false, speculativeDecoding: false,
    ...overrides,
  };
}

/** Shorthand: validate a task's expectedChanges with snapshot→current */
function validate(taskId: string, snapshot: TaskConfigSnapshot, current: TaskConfigSnapshot) {
  const task = getTaskById(taskId);
  if (!task) throw new Error(`Task ${taskId} not found`);
  return validateExpectedChanges(snapshot, current, task.expectedChanges);
}

// ═════════════════════════════════════════════════════════════════════════
// Training Beginner
// ═════════════════════════════════════════════════════════════════════════

describe('TB01 — Tensor Cores', () => {
  const snapshot = makeSnapshot({ precision: 'fp32' });

  it('correct: FP32→BF16', () => {
    expect(validate('training-beginner-01', snapshot, makeSnapshot({ precision: 'bf16' })).valid).toBe(true);
  });
  it('correct: FP32→FP16', () => {
    expect(validate('training-beginner-01', snapshot, makeSnapshot({ precision: 'fp16' })).valid).toBe(true);
  });
  it('correct: FP32→FP8', () => {
    expect(validate('training-beginner-01', snapshot, makeSnapshot({ precision: 'fp8' })).valid).toBe(true);
  });
  it('correct: change precision AND increase MBS (complementary)', () => {
    expect(validate('training-beginner-01', snapshot,
      makeSnapshot({ precision: 'bf16', microBatchSize: 128 })).valid).toBe(true);
  });
  it('wrong: change model only', () => {
    expect(validate('training-beginner-01', snapshot,
      makeSnapshot({ modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: increase numGPUs without changing precision', () => {
    expect(validate('training-beginner-01', snapshot,
      makeSnapshot({ numGPUs: 8 })).valid).toBe(false);
  });
  it('wrong: no changes at all', () => {
    expect(validate('training-beginner-01', snapshot, makeSnapshot()).valid).toBe(false);
  });
});

describe('TB02 — GPU Memory', () => {
  const snapshot = makeSnapshot({ precision: 'fp32' });

  it('correct: FP32→BF16', () => {
    expect(validate('training-beginner-02', snapshot, makeSnapshot({ precision: 'bf16' })).valid).toBe(true);
  });
  it('correct: FP32→INT4', () => {
    expect(validate('training-beginner-02', snapshot, makeSnapshot({ precision: 'fp4' })).valid).toBe(true);
  });
  it('correct: change precision AND enable AC (complementary)', () => {
    expect(validate('training-beginner-02', snapshot,
      makeSnapshot({ precision: 'bf16', activationCheckpointing: true })).valid).toBe(true);
  });
  it('wrong: only reduce MBS (precision unchanged)', () => {
    expect(validate('training-beginner-02', snapshot,
      makeSnapshot({ microBatchSize: 4 })).valid).toBe(false);
  });
  it('wrong: change model', () => {
    expect(validate('training-beginner-02', snapshot,
      makeSnapshot({ modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
});

describe('TB03 — Activation Memory', () => {
  const snapshot = makeSnapshot({ microBatchSize: 64, globalBatchSize: 64 });

  it('correct: MBS 64→4', () => {
    expect(validate('training-beginner-03', snapshot,
      makeSnapshot({ microBatchSize: 4, globalBatchSize: 64 })).valid).toBe(true);
  });
  it('correct: MBS 64→32 (any decrease)', () => {
    expect(validate('training-beginner-03', snapshot,
      makeSnapshot({ microBatchSize: 32, globalBatchSize: 64 })).valid).toBe(true);
  });
  it('correct: MBS 64→1', () => {
    expect(validate('training-beginner-03', snapshot,
      makeSnapshot({ microBatchSize: 1, globalBatchSize: 64 })).valid).toBe(true);
  });
  it('correct: decrease MBS AND enable AC (complementary)', () => {
    expect(validate('training-beginner-03', snapshot,
      makeSnapshot({ microBatchSize: 4, globalBatchSize: 64, activationCheckpointing: true })).valid).toBe(true);
  });
  it('wrong: decrease GBS instead (GBS protected)', () => {
    expect(validate('training-beginner-03', snapshot,
      makeSnapshot({ microBatchSize: 64, globalBatchSize: 32 })).valid).toBe(false);
  });
  it('wrong: change model', () => {
    expect(validate('training-beginner-03', snapshot,
      makeSnapshot({ microBatchSize: 4, globalBatchSize: 64, modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: enable AC without reducing MBS', () => {
    expect(validate('training-beginner-03', snapshot,
      makeSnapshot({ microBatchSize: 64, globalBatchSize: 64, activationCheckpointing: true })).valid).toBe(false);
  });
});

describe('TB04 — Activation Checkpointing', () => {
  const snapshot = makeSnapshot({ activationCheckpointing: false });

  it('correct: enable AC', () => {
    expect(validate('training-beginner-04', snapshot,
      makeSnapshot({ activationCheckpointing: true })).valid).toBe(true);
  });
  it('correct: enable AC AND change precision (complementary)', () => {
    expect(validate('training-beginner-04', snapshot,
      makeSnapshot({ activationCheckpointing: true, precision: 'bf16' })).valid).toBe(true);
  });
  it('wrong: reduce MBS without enabling AC', () => {
    expect(validate('training-beginner-04', snapshot,
      makeSnapshot({ microBatchSize: 4 })).valid).toBe(false);
  });
  it('wrong: change model', () => {
    expect(validate('training-beginner-04', snapshot,
      makeSnapshot({ modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
});

describe('TB05 — Flash Attention', () => {
  const snapshot = makeSnapshot({ flashAttention: false });

  it('correct: enable FA', () => {
    expect(validate('training-beginner-05', snapshot,
      makeSnapshot({ flashAttention: true })).valid).toBe(true);
  });
  it('correct: enable FA AND change precision (complementary)', () => {
    expect(validate('training-beginner-05', snapshot,
      makeSnapshot({ flashAttention: true, precision: 'bf16' })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('training-beginner-05', snapshot,
      makeSnapshot({ modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
});

describe('TB06 — Scaling to Multiple GPUs', () => {
  const snapshot = makeSnapshot({ numGPUs: 1 });

  it('correct: 1→8 GPUs', () => {
    expect(validate('training-beginner-06', snapshot,
      makeSnapshot({ numGPUs: 8 })).valid).toBe(true);
  });
  it('correct: 1→4 GPUs', () => {
    expect(validate('training-beginner-06', snapshot,
      makeSnapshot({ numGPUs: 4 })).valid).toBe(true);
  });
  it('correct: increase GPUs AND change GBS (complementary)', () => {
    expect(validate('training-beginner-06', snapshot,
      makeSnapshot({ numGPUs: 8, globalBatchSize: 128 })).valid).toBe(true);
  });
  it('wrong: change GPU type', () => {
    expect(validate('training-beginner-06', snapshot,
      makeSnapshot({ gpuId: 'h100-sxm' })).valid).toBe(false);
  });
  it('wrong: change precision on 1 GPU (numGPUs not increased)', () => {
    expect(validate('training-beginner-06', snapshot,
      makeSnapshot({ precision: 'fp8' })).valid).toBe(false);
  });
});

describe('TB07 — When Models Don\'t Fit', () => {
  const snapshot = makeSnapshot({ strategyType: 'ddp' });

  it('correct: DDP→FSDP', () => {
    expect(validate('training-beginner-07', snapshot,
      makeSnapshot({ strategyType: 'fsdp' })).valid).toBe(true);
  });
  it('correct: DDP→FSDP-TP', () => {
    expect(validate('training-beginner-07', snapshot,
      makeSnapshot({ strategyType: 'fsdp-tp' })).valid).toBe(true);
  });
  it('correct: DDP→ZeRO-1', () => {
    expect(validate('training-beginner-07', snapshot,
      makeSnapshot({ strategyType: 'zero-1' })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('training-beginner-07', snapshot,
      makeSnapshot({ modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('training-beginner-07', snapshot,
      makeSnapshot({ gpuId: 'h100-sxm' })).valid).toBe(false);
  });
});

describe('TB08 — Micro-Batch Size & Memory', () => {
  const snapshot = makeSnapshot({ microBatchSize: 4, gpuId: 'a100-sxm-80gb' });

  it('correct: MBS 4→1', () => {
    expect(validate('training-beginner-08', snapshot,
      makeSnapshot({ microBatchSize: 1, gpuId: 'a100-sxm-80gb' })).valid).toBe(true);
  });
  it('correct: MBS 4→2', () => {
    expect(validate('training-beginner-08', snapshot,
      makeSnapshot({ microBatchSize: 2, gpuId: 'a100-sxm-80gb' })).valid).toBe(true);
  });
  it('correct: decrease MBS AND enable AC (complementary)', () => {
    expect(validate('training-beginner-08', snapshot,
      makeSnapshot({ microBatchSize: 1, gpuId: 'a100-sxm-80gb', activationCheckpointing: true })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('training-beginner-08', snapshot,
      makeSnapshot({ microBatchSize: 1, gpuId: 'a100-sxm-80gb', modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: enable AC without reducing MBS', () => {
    expect(validate('training-beginner-08', snapshot,
      makeSnapshot({ microBatchSize: 4, gpuId: 'a100-sxm-80gb', activationCheckpointing: true })).valid).toBe(false);
  });
});

describe('TB09 — Going Multi-Node', () => {
  const snapshot = makeSnapshot({ numGPUs: 8, gpuId: 'a100-sxm-80gb' });

  it('correct: 8→16 GPUs', () => {
    expect(validate('training-beginner-09', snapshot,
      makeSnapshot({ numGPUs: 16, gpuId: 'a100-sxm-80gb' })).valid).toBe(true);
  });
  it('correct: 8→32 GPUs', () => {
    expect(validate('training-beginner-09', snapshot,
      makeSnapshot({ numGPUs: 32, gpuId: 'a100-sxm-80gb' })).valid).toBe(true);
  });
  it('correct: increase GPUs AND increase GBS (complementary)', () => {
    expect(validate('training-beginner-09', snapshot,
      makeSnapshot({ numGPUs: 16, gpuId: 'a100-sxm-80gb', globalBatchSize: 128 })).valid).toBe(true);
  });
  it('wrong: change GPU type', () => {
    expect(validate('training-beginner-09', snapshot,
      makeSnapshot({ numGPUs: 16, gpuId: 'h100-sxm' })).valid).toBe(false);
  });
  it('wrong: change model', () => {
    expect(validate('training-beginner-09', snapshot,
      makeSnapshot({ numGPUs: 16, gpuId: 'a100-sxm-80gb', modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
});

describe('TB10 — Efficiency Win', () => {
  const snapshot = makeSnapshot({ activationCheckpointing: true, gpuId: 'a100-sxm-80gb' });

  it('correct: disable AC', () => {
    expect(validate('training-beginner-10', snapshot,
      makeSnapshot({ activationCheckpointing: false, gpuId: 'a100-sxm-80gb' })).valid).toBe(true);
  });
  it('correct: disable AC AND increase MBS (complementary)', () => {
    expect(validate('training-beginner-10', snapshot,
      makeSnapshot({ activationCheckpointing: false, gpuId: 'a100-sxm-80gb', microBatchSize: 128 })).valid).toBe(true);
  });
  it('correct: disable AC AND change precision (complementary)', () => {
    expect(validate('training-beginner-10', snapshot,
      makeSnapshot({ activationCheckpointing: false, gpuId: 'a100-sxm-80gb', precision: 'bf16' })).valid).toBe(true);
  });
  it('wrong: switch to FP8 without disabling AC', () => {
    expect(validate('training-beginner-10', snapshot,
      makeSnapshot({ activationCheckpointing: true, gpuId: 'a100-sxm-80gb', precision: 'fp8' })).valid).toBe(false);
  });
  it('wrong: change model', () => {
    expect(validate('training-beginner-10', snapshot,
      makeSnapshot({ activationCheckpointing: false, gpuId: 'a100-sxm-80gb', modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('training-beginner-10', snapshot,
      makeSnapshot({ activationCheckpointing: false, gpuId: 'h100-sxm' })).valid).toBe(false);
  });
});

// ═════════════════════════════════════════════════════════════════════════
// Inference Beginner (representative tests)
// ═════════════════════════════════════════════════════════════════════════

describe('IB01 — First Inference', () => {
  const snapshot = makeSnapshot({ weightPrecision: 'fp32' });

  it('correct: FP32→BF16', () => {
    expect(validate('inference-beginner-01', snapshot,
      makeSnapshot({ weightPrecision: 'bf16' })).valid).toBe(true);
  });
  it('correct: FP32→INT4', () => {
    expect(validate('inference-beginner-01', snapshot,
      makeSnapshot({ weightPrecision: 'int4' })).valid).toBe(true);
  });
  it('wrong: change GPU', () => {
    expect(validate('inference-beginner-01', snapshot,
      makeSnapshot({ gpuId: 'h100-sxm' })).valid).toBe(false);
  });
});

describe('IB02 — Right-Sizing Your Model', () => {
  const snapshot = makeSnapshot({ modelId: 'llama3.3-70b', gpuId: 'rtx-4090' });

  it('correct: change model to 8B', () => {
    expect(validate('inference-beginner-02', snapshot,
      makeSnapshot({ modelId: 'llama3.1-8b', gpuId: 'rtx-4090' })).valid).toBe(true);
  });
  it('correct: change model to different smaller model', () => {
    expect(validate('inference-beginner-02', snapshot,
      makeSnapshot({ modelId: 'qwen3-14b', gpuId: 'rtx-4090' })).valid).toBe(true);
  });
  it('correct: change model AND quantize (complementary)', () => {
    expect(validate('inference-beginner-02', snapshot,
      makeSnapshot({ modelId: 'llama3.1-8b', gpuId: 'rtx-4090', weightPrecision: 'int4' })).valid).toBe(true);
  });
  it('wrong: change GPU instead of model', () => {
    expect(validate('inference-beginner-02', snapshot,
      makeSnapshot({ modelId: 'llama3.3-70b', gpuId: 'h100-sxm' })).valid).toBe(false);
  });
  it('wrong: only quantize without changing model', () => {
    expect(validate('inference-beginner-02', snapshot,
      makeSnapshot({ modelId: 'llama3.3-70b', gpuId: 'rtx-4090', weightPrecision: 'int4' })).valid).toBe(false);
  });
});

describe('IB03 — KV Cache (multi-solution, no required changes)', () => {
  const snapshot = makeSnapshot();

  it('correct: reduce batchSize (no required changes, only hardware protected)', () => {
    expect(validate('inference-beginner-03', snapshot,
      makeSnapshot({ batchSize: 1 })).valid).toBe(true);
  });
  it('correct: change kvCachePrecision', () => {
    expect(validate('inference-beginner-03', snapshot,
      makeSnapshot({ kvCachePrecision: 'fp8' })).valid).toBe(true);
  });
  it('correct: do both', () => {
    expect(validate('inference-beginner-03', snapshot,
      makeSnapshot({ batchSize: 1, kvCachePrecision: 'fp8' })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('inference-beginner-03', snapshot,
      makeSnapshot({ modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('inference-beginner-03', snapshot,
      makeSnapshot({ gpuId: 'h100-sxm' })).valid).toBe(false);
  });
});

describe('IB04 — The Bandwidth Bottleneck', () => {
  const snapshot = makeSnapshot({ modelId: 'llama3.1-8b', gpuId: 't4', weightPrecision: 'int8' });

  it('correct: change GPU to RTX 4090', () => {
    expect(validate('inference-beginner-04', snapshot,
      makeSnapshot({ modelId: 'llama3.1-8b', gpuId: 'rtx-4090', weightPrecision: 'int8' })).valid).toBe(true);
  });
  it('correct: change GPU to H100', () => {
    expect(validate('inference-beginner-04', snapshot,
      makeSnapshot({ modelId: 'llama3.1-8b', gpuId: 'h100-sxm', weightPrecision: 'int8' })).valid).toBe(true);
  });
  it('correct: change GPU AND batch size (complementary)', () => {
    expect(validate('inference-beginner-04', snapshot,
      makeSnapshot({ modelId: 'llama3.1-8b', gpuId: 'rtx-4090', weightPrecision: 'int8', batchSize: 4 })).valid).toBe(true);
  });
  it('wrong: change model instead of GPU', () => {
    expect(validate('inference-beginner-04', snapshot,
      makeSnapshot({ modelId: 'llama3.3-70b', gpuId: 't4', weightPrecision: 'int8' })).valid).toBe(false);
  });
  it('wrong: change weight precision instead of GPU', () => {
    expect(validate('inference-beginner-04', snapshot,
      makeSnapshot({ modelId: 'llama3.1-8b', gpuId: 't4', weightPrecision: 'int4' })).valid).toBe(false);
  });
  it('wrong: no changes', () => {
    expect(validate('inference-beginner-04', snapshot, snapshot).valid).toBe(false);
  });
});

describe('IB07 — Flash Attention', () => {
  const snapshot = makeSnapshot({ flashAttention: false, inputSeqLen: 4096 });

  it('correct: enable flashAttention', () => {
    expect(validate('inference-beginner-07', snapshot,
      makeSnapshot({ flashAttention: true, inputSeqLen: 4096 })).valid).toBe(true);
  });
  it('correct: enable FA AND reduce batch (complementary)', () => {
    expect(validate('inference-beginner-07', snapshot,
      makeSnapshot({ flashAttention: true, inputSeqLen: 4096, batchSize: 1 })).valid).toBe(true);
  });
  it('wrong: reduce inputSeqLen without enabling FA', () => {
    expect(validate('inference-beginner-07', snapshot,
      makeSnapshot({ flashAttention: false, inputSeqLen: 512 })).valid).toBe(false);
  });
});

// ═════════════════════════════════════════════════════════════════════════
// Inference Intermediate (representative)
// ═════════════════════════════════════════════════════════════════════════

describe('II01 — TP for Large Models', () => {
  const snapshot = makeSnapshot({ tensorParallel: 1 });

  it('correct: increase TP', () => {
    expect(validate('inference-intermediate-01', snapshot,
      makeSnapshot({ tensorParallel: 4 })).valid).toBe(true);
  });
  it('wrong: change GPU', () => {
    expect(validate('inference-intermediate-01', snapshot,
      makeSnapshot({ tensorParallel: 4, gpuId: 'h100-sxm' })).valid).toBe(false);
  });
});

describe('II08 — Choosing the Right GPU', () => {
  const snapshot = makeSnapshot({ gpuId: 'a100-sxm-80gb' });

  it('correct: change GPU (task is about choosing GPU)', () => {
    expect(validate('inference-intermediate-08', snapshot,
      makeSnapshot({ gpuId: 'h100-sxm' })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('inference-intermediate-08', snapshot,
      makeSnapshot({ modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
});

// ═════════════════════════════════════════════════════════════════════════
// Training Intermediate (representative)
// ═════════════════════════════════════════════════════════════════════════

describe('TI01 — Why TP', () => {
  const snapshot = makeSnapshot({ strategyType: 'ddp' });

  it('correct: change strategy', () => {
    expect(validate('training-intermediate-01', snapshot,
      makeSnapshot({ strategyType: 'fsdp-tp' })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('training-intermediate-01', snapshot,
      makeSnapshot({ modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
});

describe('TI03 — NVLink Boundary', () => {
  const snapshot = makeSnapshot({ tpDegree: 16, gpuId: 'a100-sxm-80gb' });

  it('correct: decrease TP', () => {
    expect(validate('training-intermediate-03', snapshot,
      makeSnapshot({ tpDegree: 8, gpuId: 'a100-sxm-80gb' })).valid).toBe(true);
  });
  it('wrong: change GPU', () => {
    expect(validate('training-intermediate-03', snapshot,
      makeSnapshot({ tpDegree: 8, gpuId: 'h100-sxm' })).valid).toBe(false);
  });
});

// ═════════════════════════════════════════════════════════════════════════
// Training Advanced (representative)
// ═════════════════════════════════════════════════════════════════════════

describe('TA09 — GPU Architecture', () => {
  const snapshot = makeSnapshot({ gpuId: 'a100-sxm-80gb' });

  it('correct: change GPU (task IS about changing GPU)', () => {
    expect(validate('training-advanced-09', snapshot,
      makeSnapshot({ gpuId: 'h100-sxm' })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('training-advanced-09', snapshot,
      makeSnapshot({ modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
});

describe('TA10 — Optimal Configuration (capstone, no required changes)', () => {
  const snapshot = makeSnapshot({ gpuId: 'a100-sxm-80gb' });

  it('correct: any change that keeps hardware', () => {
    expect(validate('training-advanced-10', snapshot,
      makeSnapshot({ gpuId: 'a100-sxm-80gb', precision: 'bf16', strategyType: 'fsdp-tp-pp' })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('training-advanced-10', snapshot,
      makeSnapshot({ gpuId: 'a100-sxm-80gb', modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('training-advanced-10', snapshot,
      makeSnapshot({ gpuId: 'h100-sxm' })).valid).toBe(false);
  });
});

// ═════════════════════════════════════════════════════════════════════════
// Inference Advanced (representative)
// ═════════════════════════════════════════════════════════════════════════

describe('IA04 — Optimizing Time to First Token', () => {
  const snapshot = makeSnapshot({ tensorParallel: 2 });

  it('correct: increase TP to 8', () => {
    expect(validate('inference-advanced-04', snapshot,
      makeSnapshot({ tensorParallel: 8 })).valid).toBe(true);
  });
  it('correct: increase TP to 4 with FP8', () => {
    expect(validate('inference-advanced-04', snapshot,
      makeSnapshot({ tensorParallel: 4, weightPrecision: 'fp8' })).valid).toBe(true);
  });
  it('wrong: change GPU', () => {
    expect(validate('inference-advanced-04', snapshot,
      makeSnapshot({ tensorParallel: 8, gpuId: 'h100-sxm' })).valid).toBe(false);
  });
});

describe('IA10 — Full Serving Stack (capstone, no required changes)', () => {
  const snapshot = makeSnapshot({ gpuId: 'a100-sxm-80gb' });

  it('correct: any changes that keep hardware', () => {
    expect(validate('inference-advanced-10', snapshot,
      makeSnapshot({
        gpuId: 'a100-sxm-80gb',
        weightPrecision: 'fp8',
        tensorParallel: 8,
        speculativeDecoding: true,
      })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('inference-advanced-10', snapshot,
      makeSnapshot({ gpuId: 'a100-sxm-80gb', modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
});
