/**
 * Integration tests for expectedChanges — verifies generosity and correctness
 * of approach validation for each task category.
 *
 * Design principle: "Be generous." Only block when the user clearly bypassed the
 * intended technique. Legitimate complementary changes should always be allowed.
 */

import { describe, it, expect } from 'vitest';
import type { TaskConfigSnapshot } from '../../src/game/validation.ts';
import { makeSnapshot as _makeSnapshot, validateTask as validate } from '../helpers/snapshots.ts';

const makeSnapshot = (overrides?: Partial<TaskConfigSnapshot>) => _makeSnapshot('training', overrides);

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

describe('TB02 — Activation Memory', () => {
  const snapshot = makeSnapshot({ microBatchSize: 64, globalBatchSize: 64 });

  it('correct: MBS 64→4', () => {
    expect(validate('training-beginner-02', snapshot,
      makeSnapshot({ microBatchSize: 4, globalBatchSize: 64 })).valid).toBe(true);
  });
  it('correct: MBS 64→32 (any decrease)', () => {
    expect(validate('training-beginner-02', snapshot,
      makeSnapshot({ microBatchSize: 32, globalBatchSize: 64 })).valid).toBe(true);
  });
  it('correct: MBS 64→1', () => {
    expect(validate('training-beginner-02', snapshot,
      makeSnapshot({ microBatchSize: 1, globalBatchSize: 64 })).valid).toBe(true);
  });
  it('correct: decrease MBS AND enable AC (complementary)', () => {
    expect(validate('training-beginner-02', snapshot,
      makeSnapshot({ microBatchSize: 4, globalBatchSize: 64, activationCheckpointing: true })).valid).toBe(true);
  });
  it('wrong: decrease GBS instead (GBS protected)', () => {
    expect(validate('training-beginner-02', snapshot,
      makeSnapshot({ microBatchSize: 64, globalBatchSize: 32 })).valid).toBe(false);
  });
  it('wrong: change model', () => {
    expect(validate('training-beginner-02', snapshot,
      makeSnapshot({ microBatchSize: 4, globalBatchSize: 64, modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: enable AC without reducing MBS', () => {
    expect(validate('training-beginner-02', snapshot,
      makeSnapshot({ microBatchSize: 64, globalBatchSize: 64, activationCheckpointing: true })).valid).toBe(false);
  });
});

describe('TB03 — Sequence Length and Memory', () => {
  const snapshot = makeSnapshot({ sequenceLength: 8192 });

  it('correct: seqLen 8192→4096', () => {
    expect(validate('training-beginner-03', snapshot, makeSnapshot({ sequenceLength: 4096 })).valid).toBe(true);
  });
  it('correct: seqLen 8192→2048', () => {
    expect(validate('training-beginner-03', snapshot, makeSnapshot({ sequenceLength: 2048 })).valid).toBe(true);
  });
  it('correct: reduce seqLen AND enable AC (complementary)', () => {
    expect(validate('training-beginner-03', snapshot,
      makeSnapshot({ sequenceLength: 4096, activationCheckpointing: true })).valid).toBe(true);
  });
  it('wrong: only change precision (seqLen unchanged)', () => {
    expect(validate('training-beginner-03', snapshot,
      makeSnapshot({ precision: 'bf16', sequenceLength: 8192 })).valid).toBe(false);
  });
  it('wrong: change model', () => {
    expect(validate('training-beginner-03', snapshot,
      makeSnapshot({ modelId: 'llama3.3-70b' })).valid).toBe(false);
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
  const snapshot = makeSnapshot({ precision: 'bf16', gpuId: 'h100-sxm' });

  it('correct: disable AC + enable FA', () => {
    expect(validate('training-beginner-10', snapshot,
      makeSnapshot({ activationCheckpointing: false, flashAttention: true, precision: 'bf16', gpuId: 'h100-sxm' })).valid).toBe(true);
  });
  it('correct: disable AC + enable FA + increase MBS (complementary)', () => {
    expect(validate('training-beginner-10', snapshot,
      makeSnapshot({ activationCheckpointing: false, flashAttention: true, precision: 'bf16', gpuId: 'h100-sxm', microBatchSize: 128 })).valid).toBe(true);
  });
  it('correct: only disable AC (still valid — no technique enforcement)', () => {
    expect(validate('training-beginner-10', snapshot,
      makeSnapshot({ activationCheckpointing: false, precision: 'bf16', gpuId: 'h100-sxm' })).valid).toBe(true);
  });
  it('wrong: switch to FP8', () => {
    expect(validate('training-beginner-10', snapshot,
      makeSnapshot({ activationCheckpointing: false, flashAttention: true, precision: 'fp8', gpuId: 'h100-sxm' })).valid).toBe(false);
  });
  it('wrong: change model', () => {
    expect(validate('training-beginner-10', snapshot,
      makeSnapshot({ activationCheckpointing: false, precision: 'bf16', gpuId: 'h100-sxm', modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('training-beginner-10', snapshot,
      makeSnapshot({ activationCheckpointing: false, precision: 'bf16', gpuId: 'a100-sxm-80gb' })).valid).toBe(false);
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

describe('IB05 — Batch Size and Throughput', () => {
  const snapshot = makeSnapshot({ batchSize: 1 });

  it('correct: increase batch', () => {
    expect(validate('inference-beginner-05', snapshot,
      makeSnapshot({ batchSize: 8 })).valid).toBe(true);
  });
  it('correct: increase batch AND quantize (complementary)', () => {
    expect(validate('inference-beginner-05', snapshot,
      makeSnapshot({ batchSize: 8, weightPrecision: 'fp8' })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('inference-beginner-05', snapshot,
      makeSnapshot({ batchSize: 8, modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('inference-beginner-05', snapshot,
      makeSnapshot({ batchSize: 8, gpuId: 'h100-sxm' })).valid).toBe(false);
  });
});

describe('IB06 — TTFT vs TPOT', () => {
  const snapshot = makeSnapshot({ batchSize: 32 });

  it('correct: decrease batch', () => {
    expect(validate('inference-beginner-06', snapshot,
      makeSnapshot({ batchSize: 1 })).valid).toBe(true);
  });
  it('correct: decrease batch to 8', () => {
    expect(validate('inference-beginner-06', snapshot,
      makeSnapshot({ batchSize: 8 })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('inference-beginner-06', snapshot,
      makeSnapshot({ modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('inference-beginner-06', snapshot,
      makeSnapshot({ batchSize: 1, gpuId: 'h100-sxm' })).valid).toBe(false);
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

describe('IB08 — KV Cache Precision', () => {
  const snapshot = makeSnapshot({ kvCachePrecision: 'bf16' });

  it('correct: change KV precision to FP8', () => {
    expect(validate('inference-beginner-08', snapshot,
      makeSnapshot({ kvCachePrecision: 'fp8' })).valid).toBe(true);
  });
  it('correct: change KV precision to INT8', () => {
    expect(validate('inference-beginner-08', snapshot,
      makeSnapshot({ kvCachePrecision: 'int8' })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('inference-beginner-08', snapshot,
      makeSnapshot({ kvCachePrecision: 'fp8', modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('inference-beginner-08', snapshot,
      makeSnapshot({ kvCachePrecision: 'fp8', gpuId: 'h100-sxm' })).valid).toBe(false);
  });
});

describe('IB09 — Latency vs Throughput', () => {
  const snapshot = makeSnapshot({ batchSize: 32 });

  it('correct: change batch to 16', () => {
    expect(validate('inference-beginner-09', snapshot,
      makeSnapshot({ batchSize: 16 })).valid).toBe(true);
  });
  it('correct: change batch to 8', () => {
    expect(validate('inference-beginner-09', snapshot,
      makeSnapshot({ batchSize: 8 })).valid).toBe(true);
  });
  it('correct: change batch AND quantize (complementary)', () => {
    expect(validate('inference-beginner-09', snapshot,
      makeSnapshot({ batchSize: 16, weightPrecision: 'fp8' })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('inference-beginner-09', snapshot,
      makeSnapshot({ modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('inference-beginner-09', snapshot,
      makeSnapshot({ batchSize: 16, gpuId: 'h100-sxm' })).valid).toBe(false);
  });
});

describe('IB10 — Continuous Batching', () => {
  const snapshot = makeSnapshot({ continuousBatching: false });

  it('correct: enable CB', () => {
    expect(validate('inference-beginner-10', snapshot,
      makeSnapshot({ continuousBatching: true })).valid).toBe(true);
  });
  it('correct: enable CB AND increase batch (complementary)', () => {
    expect(validate('inference-beginner-10', snapshot,
      makeSnapshot({ continuousBatching: true, batchSize: 16 })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('inference-beginner-10', snapshot,
      makeSnapshot({ continuousBatching: true, modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('inference-beginner-10', snapshot,
      makeSnapshot({ continuousBatching: true, gpuId: 'h100-sxm' })).valid).toBe(false);
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

describe('II02 — TP Degree vs Latency', () => {
  const snapshot = makeSnapshot({ tensorParallel: 1 });

  it('correct: increase TP to 8', () => {
    expect(validate('inference-intermediate-02', snapshot,
      makeSnapshot({ tensorParallel: 8 })).valid).toBe(true);
  });
  it('correct: increase TP to 4', () => {
    expect(validate('inference-intermediate-02', snapshot,
      makeSnapshot({ tensorParallel: 4 })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('inference-intermediate-02', snapshot,
      makeSnapshot({ tensorParallel: 8, modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('inference-intermediate-02', snapshot,
      makeSnapshot({ tensorParallel: 8, gpuId: 'rtx-4090' })).valid).toBe(false);
  });
});

describe('II03 — Serving 405B', () => {
  const snapshot = makeSnapshot({ weightPrecision: 'bf16', tensorParallel: 1 });

  it('correct: quantize + increase TP', () => {
    expect(validate('inference-intermediate-03', snapshot,
      makeSnapshot({ weightPrecision: 'fp8', tensorParallel: 8 })).valid).toBe(true);
  });
  it('correct: INT4 + TP=8', () => {
    expect(validate('inference-intermediate-03', snapshot,
      makeSnapshot({ weightPrecision: 'int4', tensorParallel: 8 })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('inference-intermediate-03', snapshot,
      makeSnapshot({ weightPrecision: 'fp8', tensorParallel: 8, modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('inference-intermediate-03', snapshot,
      makeSnapshot({ weightPrecision: 'fp8', tensorParallel: 8, gpuId: 'rtx-4090' })).valid).toBe(false);
  });
});

describe('II04 — When TP Wastes GPUs', () => {
  const snapshot = makeSnapshot({ tensorParallel: 8 });

  it('correct: decrease TP to 2', () => {
    expect(validate('inference-intermediate-04', snapshot,
      makeSnapshot({ tensorParallel: 2 })).valid).toBe(true);
  });
  it('correct: decrease TP to 1', () => {
    expect(validate('inference-intermediate-04', snapshot,
      makeSnapshot({ tensorParallel: 1 })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('inference-intermediate-04', snapshot,
      makeSnapshot({ tensorParallel: 2, modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('inference-intermediate-04', snapshot,
      makeSnapshot({ tensorParallel: 2, gpuId: 'rtx-4090' })).valid).toBe(false);
  });
});

describe('II05 — KV Cache at Long Contexts (open-ended)', () => {
  const snapshot = makeSnapshot();

  it('correct: change kvCachePrecision', () => {
    expect(validate('inference-intermediate-05', snapshot,
      makeSnapshot({ kvCachePrecision: 'fp8' })).valid).toBe(true);
  });
  it('correct: reduce batch', () => {
    expect(validate('inference-intermediate-05', snapshot,
      makeSnapshot({ batchSize: 8 })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('inference-intermediate-05', snapshot,
      makeSnapshot({ modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('inference-intermediate-05', snapshot,
      makeSnapshot({ gpuId: 'h100-sxm' })).valid).toBe(false);
  });
});

describe('II06 — Maximizing Batch Throughput', () => {
  const snapshot = makeSnapshot({ batchSize: 16, continuousBatching: false });

  it('correct: increase batch', () => {
    expect(validate('inference-intermediate-06', snapshot,
      makeSnapshot({ batchSize: 32, continuousBatching: false })).valid).toBe(true);
  });
  it('correct: decrease batch', () => {
    expect(validate('inference-intermediate-06', snapshot,
      makeSnapshot({ batchSize: 8, continuousBatching: false })).valid).toBe(true);
  });
  it('wrong: enable CB (protected)', () => {
    expect(validate('inference-intermediate-06', snapshot,
      makeSnapshot({ batchSize: 32, continuousBatching: true })).valid).toBe(false);
  });
  it('wrong: change model', () => {
    expect(validate('inference-intermediate-06', snapshot,
      makeSnapshot({ batchSize: 32, modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('inference-intermediate-06', snapshot,
      makeSnapshot({ batchSize: 32, gpuId: 'h100-sxm' })).valid).toBe(false);
  });
});

describe('II07 — Minimizing Per-Token Latency', () => {
  const snapshot = makeSnapshot({ weightPrecision: 'bf16', tensorParallel: 1, batchSize: 1 });

  it('correct: quantize + increase TP + increase batch', () => {
    expect(validate('inference-intermediate-07', snapshot,
      makeSnapshot({ weightPrecision: 'int4', tensorParallel: 4, batchSize: 4 })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('inference-intermediate-07', snapshot,
      makeSnapshot({ weightPrecision: 'int4', tensorParallel: 4, batchSize: 4, modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('inference-intermediate-07', snapshot,
      makeSnapshot({ weightPrecision: 'int4', tensorParallel: 4, batchSize: 4, gpuId: 'h100-sxm' })).valid).toBe(false);
  });
  it('wrong: only quantize without TP or batch', () => {
    expect(validate('inference-intermediate-07', snapshot,
      makeSnapshot({ weightPrecision: 'int4', tensorParallel: 1, batchSize: 1 })).valid).toBe(false);
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
  it('wrong: only increase batch size (no GPU change)', () => {
    expect(validate('inference-intermediate-08', snapshot,
      makeSnapshot({ batchSize: 32 })).valid).toBe(false);
  });
});

describe('II09 — Speculative Decoding', () => {
  const snapshot = makeSnapshot({ speculativeDecoding: false });

  it('correct: enable speculative decoding', () => {
    expect(validate('inference-intermediate-09', snapshot,
      makeSnapshot({ speculativeDecoding: true })).valid).toBe(true);
  });
  it('correct: enable spec dec AND increase TP (complementary)', () => {
    expect(validate('inference-intermediate-09', snapshot,
      makeSnapshot({ speculativeDecoding: true, tensorParallel: 4 })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('inference-intermediate-09', snapshot,
      makeSnapshot({ speculativeDecoding: true, modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('inference-intermediate-09', snapshot,
      makeSnapshot({ speculativeDecoding: true, gpuId: 'rtx-4090' })).valid).toBe(false);
  });
});

describe('II10 — FP8 Quantized Inference', () => {
  const snapshot = makeSnapshot({ weightPrecision: 'bf16' });

  it('correct: change to FP8', () => {
    expect(validate('inference-intermediate-10', snapshot,
      makeSnapshot({ weightPrecision: 'fp8' })).valid).toBe(true);
  });
  it('correct: change to INT8', () => {
    expect(validate('inference-intermediate-10', snapshot,
      makeSnapshot({ weightPrecision: 'int8' })).valid).toBe(true);
  });
  it('correct: change precision AND adjust TP (complementary)', () => {
    expect(validate('inference-intermediate-10', snapshot,
      makeSnapshot({ weightPrecision: 'fp8', tensorParallel: 4 })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('inference-intermediate-10', snapshot,
      makeSnapshot({ weightPrecision: 'fp8', modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('inference-intermediate-10', snapshot,
      makeSnapshot({ weightPrecision: 'fp8', gpuId: 'rtx-4090' })).valid).toBe(false);
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

describe('TI02 — TP Degree Selection', () => {
  const snapshot = makeSnapshot({ tpDegree: 1 });

  it('correct: change TP to 4', () => {
    expect(validate('training-intermediate-02', snapshot,
      makeSnapshot({ tpDegree: 4 })).valid).toBe(true);
  });
  it('correct: change TP to 8', () => {
    expect(validate('training-intermediate-02', snapshot,
      makeSnapshot({ tpDegree: 8 })).valid).toBe(true);
  });
  it('correct: change TP AND adjust GBS (complementary)', () => {
    expect(validate('training-intermediate-02', snapshot,
      makeSnapshot({ tpDegree: 4, globalBatchSize: 128 })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('training-intermediate-02', snapshot,
      makeSnapshot({ tpDegree: 4, modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('training-intermediate-02', snapshot,
      makeSnapshot({ tpDegree: 4, gpuId: 'rtx-4090' })).valid).toBe(false);
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

describe('TI04 — Pipeline Parallelism Intro', () => {
  const snapshot = makeSnapshot({ ppDegree: 1 });

  it('correct: increase PP to 4', () => {
    expect(validate('training-intermediate-04', snapshot,
      makeSnapshot({ ppDegree: 4 })).valid).toBe(true);
  });
  it('correct: increase PP to 8', () => {
    expect(validate('training-intermediate-04', snapshot,
      makeSnapshot({ ppDegree: 8 })).valid).toBe(true);
  });
  it('correct: increase PP AND adjust TP (complementary)', () => {
    expect(validate('training-intermediate-04', snapshot,
      makeSnapshot({ ppDegree: 4, tpDegree: 8 })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('training-intermediate-04', snapshot,
      makeSnapshot({ ppDegree: 4, modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('training-intermediate-04', snapshot,
      makeSnapshot({ ppDegree: 4, gpuId: 'h100-sxm' })).valid).toBe(false);
  });
});

describe('TI05 — The Pipeline Bubble', () => {
  const snapshot = makeSnapshot({ globalBatchSize: 16 });

  it('correct: increase GBS to 256', () => {
    expect(validate('training-intermediate-05', snapshot,
      makeSnapshot({ globalBatchSize: 256 })).valid).toBe(true);
  });
  it('correct: increase GBS to 128', () => {
    expect(validate('training-intermediate-05', snapshot,
      makeSnapshot({ globalBatchSize: 128 })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('training-intermediate-05', snapshot,
      makeSnapshot({ globalBatchSize: 256, modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('training-intermediate-05', snapshot,
      makeSnapshot({ globalBatchSize: 256, gpuId: 'h100-sxm' })).valid).toBe(false);
  });
});

describe('TI06 — Interleaved Scheduling', () => {
  const snapshot = makeSnapshot({ pipelineSchedule: '1f1b' });

  it('correct: switch to interleaved', () => {
    expect(validate('training-intermediate-06', snapshot,
      makeSnapshot({ pipelineSchedule: 'interleaved-1f1b' })).valid).toBe(true);
  });
  it('correct: switch to interleaved AND increase GBS (complementary)', () => {
    expect(validate('training-intermediate-06', snapshot,
      makeSnapshot({ pipelineSchedule: 'interleaved-1f1b', globalBatchSize: 128 })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('training-intermediate-06', snapshot,
      makeSnapshot({ pipelineSchedule: 'interleaved-1f1b', modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('training-intermediate-06', snapshot,
      makeSnapshot({ pipelineSchedule: 'interleaved-1f1b', gpuId: 'h100-sxm' })).valid).toBe(false);
  });
});

describe('TI07 — Sequence Parallelism', () => {
  const snapshot = makeSnapshot({ sequenceParallel: false });

  it('correct: enable SP', () => {
    expect(validate('training-intermediate-07', snapshot,
      makeSnapshot({ sequenceParallel: true })).valid).toBe(true);
  });
  it('correct: enable SP AND adjust GBS (complementary)', () => {
    expect(validate('training-intermediate-07', snapshot,
      makeSnapshot({ sequenceParallel: true, globalBatchSize: 256 })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('training-intermediate-07', snapshot,
      makeSnapshot({ sequenceParallel: true, modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('training-intermediate-07', snapshot,
      makeSnapshot({ sequenceParallel: true, gpuId: 'h100-sxm' })).valid).toBe(false);
  });
});

describe('TI08 — Scaling to a Cluster', () => {
  const snapshot = makeSnapshot({ tpDegree: 1, precision: 'bf16' });

  it('correct: change TP to 4', () => {
    expect(validate('training-intermediate-08', snapshot,
      makeSnapshot({ tpDegree: 4, precision: 'bf16' })).valid).toBe(true);
  });
  it('correct: change TP to 8', () => {
    expect(validate('training-intermediate-08', snapshot,
      makeSnapshot({ tpDegree: 8, precision: 'bf16' })).valid).toBe(true);
  });
  it('wrong: change precision instead (protected)', () => {
    expect(validate('training-intermediate-08', snapshot,
      makeSnapshot({ tpDegree: 1, precision: 'fp8' })).valid).toBe(false);
  });
  it('wrong: change model', () => {
    expect(validate('training-intermediate-08', snapshot,
      makeSnapshot({ tpDegree: 4, precision: 'bf16', modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('training-intermediate-08', snapshot,
      makeSnapshot({ tpDegree: 4, precision: 'bf16', gpuId: 'rtx-4090' })).valid).toBe(false);
  });
});

describe('TI09 — MoE: Mixture of Experts', () => {
  const snapshot = makeSnapshot({ strategyType: 'ddp' });

  it('correct: switch to FSDP', () => {
    expect(validate('training-intermediate-09', snapshot,
      makeSnapshot({ strategyType: 'fsdp' })).valid).toBe(true);
  });
  it('correct: switch to FSDP-TP', () => {
    expect(validate('training-intermediate-09', snapshot,
      makeSnapshot({ strategyType: 'fsdp-tp' })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('training-intermediate-09', snapshot,
      makeSnapshot({ strategyType: 'fsdp', modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('training-intermediate-09', snapshot,
      makeSnapshot({ strategyType: 'fsdp', gpuId: 'rtx-4090' })).valid).toBe(false);
  });
});

describe('TI10 — Precision Frontier: FP8', () => {
  const snapshot = makeSnapshot({ precision: 'bf16' });

  it('correct: change to FP8', () => {
    expect(validate('training-intermediate-10', snapshot,
      makeSnapshot({ precision: 'fp8' })).valid).toBe(true);
  });
  it('correct: change to FP8 AND adjust TP (complementary)', () => {
    expect(validate('training-intermediate-10', snapshot,
      makeSnapshot({ precision: 'fp8', tpDegree: 4 })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('training-intermediate-10', snapshot,
      makeSnapshot({ precision: 'fp8', modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('training-intermediate-10', snapshot,
      makeSnapshot({ precision: 'fp8', gpuId: 'rtx-4090' })).valid).toBe(false);
  });
});

// ═════════════════════════════════════════════════════════════════════════
// Training Advanced (representative)
// ═════════════════════════════════════════════════════════════════════════

describe('TA01 — 3D Parallelism (capstone, no required changes)', () => {
  const snapshot = makeSnapshot();

  it('correct: any config change that keeps hardware', () => {
    expect(validate('training-advanced-01', snapshot,
      makeSnapshot({ tpDegree: 8, ppDegree: 4, pipelineSchedule: 'interleaved-1f1b' })).valid).toBe(true);
  });
  it('correct: different valid approach', () => {
    expect(validate('training-advanced-01', snapshot,
      makeSnapshot({ tpDegree: 8, ppDegree: 8, globalBatchSize: 256 })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('training-advanced-01', snapshot,
      makeSnapshot({ modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('training-advanced-01', snapshot,
      makeSnapshot({ gpuId: 'rtx-4090' })).valid).toBe(false);
  });
});

describe('TA02 — Expert Parallelism', () => {
  const snapshot = makeSnapshot({ epDegree: 1 });

  it('correct: increase EP to 8', () => {
    expect(validate('training-advanced-02', snapshot,
      makeSnapshot({ epDegree: 8 })).valid).toBe(true);
  });
  it('correct: increase EP to 32', () => {
    expect(validate('training-advanced-02', snapshot,
      makeSnapshot({ epDegree: 32 })).valid).toBe(true);
  });
  it('correct: increase EP AND change precision (complementary)', () => {
    expect(validate('training-advanced-02', snapshot,
      makeSnapshot({ epDegree: 8, precision: 'fp8' })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('training-advanced-02', snapshot,
      makeSnapshot({ epDegree: 8, modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('training-advanced-02', snapshot,
      makeSnapshot({ epDegree: 8, gpuId: 'rtx-4090' })).valid).toBe(false);
  });
});

describe('TA03 — Context Parallelism', () => {
  const snapshot = makeSnapshot({ cpDegree: 1 });

  it('correct: increase CP to 4', () => {
    expect(validate('training-advanced-03', snapshot,
      makeSnapshot({ cpDegree: 4 })).valid).toBe(true);
  });
  it('correct: increase CP to 2', () => {
    expect(validate('training-advanced-03', snapshot,
      makeSnapshot({ cpDegree: 2 })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('training-advanced-03', snapshot,
      makeSnapshot({ cpDegree: 4, modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('training-advanced-03', snapshot,
      makeSnapshot({ cpDegree: 4, gpuId: 'rtx-4090' })).valid).toBe(false);
  });
});

describe('TA04 — The Communication Budget', () => {
  const snapshot = makeSnapshot({ tpDegree: 4 });

  it('correct: change TP to 1', () => {
    expect(validate('training-advanced-04', snapshot,
      makeSnapshot({ tpDegree: 1 })).valid).toBe(true);
  });
  it('correct: change TP to 2', () => {
    expect(validate('training-advanced-04', snapshot,
      makeSnapshot({ tpDegree: 2 })).valid).toBe(true);
  });
  it('correct: change TP to 8', () => {
    expect(validate('training-advanced-04', snapshot,
      makeSnapshot({ tpDegree: 8 })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('training-advanced-04', snapshot,
      makeSnapshot({ tpDegree: 1, modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('training-advanced-04', snapshot,
      makeSnapshot({ tpDegree: 1, gpuId: 'rtx-4090' })).valid).toBe(false);
  });
});

describe('TA05 — LoRA at Scale', () => {
  const snapshot = makeSnapshot({ finetuningMethod: 'full' });

  it('correct: switch to LoRA', () => {
    expect(validate('training-advanced-05', snapshot,
      makeSnapshot({ finetuningMethod: 'lora' })).valid).toBe(true);
  });
  it('correct: switch to QLoRA', () => {
    expect(validate('training-advanced-05', snapshot,
      makeSnapshot({ finetuningMethod: 'qlora' })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('training-advanced-05', snapshot,
      makeSnapshot({ finetuningMethod: 'lora', modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('training-advanced-05', snapshot,
      makeSnapshot({ finetuningMethod: 'lora', gpuId: 'rtx-4090' })).valid).toBe(false);
  });
});

describe('TA06 — QLoRA on Budget GPUs', () => {
  const snapshot = makeSnapshot({ finetuningMethod: 'full' });

  it('correct: switch to QLoRA', () => {
    expect(validate('training-advanced-06', snapshot,
      makeSnapshot({ finetuningMethod: 'qlora' })).valid).toBe(true);
  });
  it('correct: switch to LoRA', () => {
    expect(validate('training-advanced-06', snapshot,
      makeSnapshot({ finetuningMethod: 'lora' })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('training-advanced-06', snapshot,
      makeSnapshot({ finetuningMethod: 'qlora', modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('training-advanced-06', snapshot,
      makeSnapshot({ finetuningMethod: 'qlora', gpuId: 'h100-sxm' })).valid).toBe(false);
  });
});

describe('TA07 — Reproducing DeepSeek V3', () => {
  const snapshot = makeSnapshot({ precision: 'bf16' });

  it('correct: change to FP8', () => {
    expect(validate('training-advanced-07', snapshot,
      makeSnapshot({ precision: 'fp8' })).valid).toBe(true);
  });
  it('correct: change to FP8 AND set EP (complementary)', () => {
    expect(validate('training-advanced-07', snapshot,
      makeSnapshot({ precision: 'fp8', epDegree: 32 })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('training-advanced-07', snapshot,
      makeSnapshot({ precision: 'fp8', modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('training-advanced-07', snapshot,
      makeSnapshot({ precision: 'fp8', gpuId: 'rtx-4090' })).valid).toBe(false);
  });
});

describe('TA08 — Nemotron 340B', () => {
  const snapshot = makeSnapshot({ checkpointingGranularity: 'full' });

  it('correct: change to selective', () => {
    expect(validate('training-advanced-08', snapshot,
      makeSnapshot({ checkpointingGranularity: 'selective' })).valid).toBe(true);
  });
  it('correct: change to selective AND adjust PP (complementary)', () => {
    expect(validate('training-advanced-08', snapshot,
      makeSnapshot({ checkpointingGranularity: 'selective', ppDegree: 12 })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('training-advanced-08', snapshot,
      makeSnapshot({ checkpointingGranularity: 'selective', modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('training-advanced-08', snapshot,
      makeSnapshot({ checkpointingGranularity: 'selective', gpuId: 'rtx-4090' })).valid).toBe(false);
  });
});

describe('TA09 — GPU Architecture', () => {
  const snapshot = makeSnapshot({ gpuId: 'a100-sxm-80gb', tpDegree: 8 });

  it('correct: change GPU to H100', () => {
    expect(validate('training-advanced-09', snapshot,
      makeSnapshot({ gpuId: 'h100-sxm', tpDegree: 8 })).valid).toBe(true);
  });
  it('correct: change GPU AND adjust batch (complementary)', () => {
    expect(validate('training-advanced-09', snapshot,
      makeSnapshot({ gpuId: 'h100-sxm', tpDegree: 8, microBatchSize: 2, globalBatchSize: 128 })).valid).toBe(true);
  });
  it('correct: change GPU AND change TP (complementary)', () => {
    expect(validate('training-advanced-09', snapshot,
      makeSnapshot({ gpuId: 'h100-sxm', tpDegree: 4 })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('training-advanced-09', snapshot,
      makeSnapshot({ modelId: 'llama3.3-70b', gpuId: 'h100-sxm', tpDegree: 8 })).valid).toBe(false);
  });
  it('wrong: no GPU change (only tune TP/GBS on A100)', () => {
    expect(validate('training-advanced-09', snapshot,
      makeSnapshot({ gpuId: 'a100-sxm-80gb', tpDegree: 4, globalBatchSize: 128 })).valid).toBe(false);
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

describe('IA01 — DeepSeek V3 Inference', () => {
  const snapshot = makeSnapshot({ weightPrecision: 'bf16', tensorParallel: 1 });

  it('correct: quantize + increase TP', () => {
    expect(validate('inference-advanced-01', snapshot,
      makeSnapshot({ weightPrecision: 'int4', tensorParallel: 8 })).valid).toBe(true);
  });
  it('correct: FP8 + TP=8', () => {
    expect(validate('inference-advanced-01', snapshot,
      makeSnapshot({ weightPrecision: 'fp8', tensorParallel: 8 })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('inference-advanced-01', snapshot,
      makeSnapshot({ weightPrecision: 'int4', tensorParallel: 8, modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('inference-advanced-01', snapshot,
      makeSnapshot({ weightPrecision: 'int4', tensorParallel: 8, gpuId: 'rtx-4090' })).valid).toBe(false);
  });
  it('wrong: only quantize without TP', () => {
    expect(validate('inference-advanced-01', snapshot,
      makeSnapshot({ weightPrecision: 'int4', tensorParallel: 1 })).valid).toBe(false);
  });
});

describe('IA02 — Expert Parallel Inference', () => {
  const snapshot = makeSnapshot({ expertParallel: 1 });

  it('correct: increase EP to 2', () => {
    expect(validate('inference-advanced-02', snapshot,
      makeSnapshot({ expertParallel: 2 })).valid).toBe(true);
  });
  it('correct: increase EP AND adjust TP (complementary)', () => {
    expect(validate('inference-advanced-02', snapshot,
      makeSnapshot({ expertParallel: 2, tensorParallel: 8 })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('inference-advanced-02', snapshot,
      makeSnapshot({ expertParallel: 2, modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('inference-advanced-02', snapshot,
      makeSnapshot({ expertParallel: 2, gpuId: 'rtx-4090' })).valid).toBe(false);
  });
});

describe('IA03 — Speculative Decoding Mastery', () => {
  const snapshot = makeSnapshot({ speculativeDecoding: false });

  it('correct: enable speculative decoding', () => {
    expect(validate('inference-advanced-03', snapshot,
      makeSnapshot({ speculativeDecoding: true })).valid).toBe(true);
  });
  it('correct: enable spec dec AND increase TP (complementary)', () => {
    expect(validate('inference-advanced-03', snapshot,
      makeSnapshot({ speculativeDecoding: true, tensorParallel: 8 })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('inference-advanced-03', snapshot,
      makeSnapshot({ speculativeDecoding: true, modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('inference-advanced-03', snapshot,
      makeSnapshot({ speculativeDecoding: true, gpuId: 'rtx-4090' })).valid).toBe(false);
  });
});

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

describe('IA05 — Multi-Replica Serving', () => {
  const snapshot = makeSnapshot({ weightPrecision: 'bf16' });

  it('correct: change batchSize only', () => {
    expect(validate('inference-advanced-05', snapshot,
      makeSnapshot({ weightPrecision: 'bf16', batchSize: 4 })).valid).toBe(true);
  });
  it('correct: change precision and TP', () => {
    expect(validate('inference-advanced-05', snapshot,
      makeSnapshot({ weightPrecision: 'fp8', tensorParallel: 2 })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('inference-advanced-05', snapshot,
      makeSnapshot({ weightPrecision: 'fp8', modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('inference-advanced-05', snapshot,
      makeSnapshot({ weightPrecision: 'fp8', gpuId: 'rtx-4090' })).valid).toBe(false);
  });
});

describe('IA06 — Cost-per-Token Optimization (open-ended)', () => {
  const snapshot = makeSnapshot();

  it('correct: any config change that keeps hardware', () => {
    expect(validate('inference-advanced-06', snapshot,
      makeSnapshot({ tensorParallel: 2, batchSize: 32 })).valid).toBe(true);
  });
  it('correct: different valid approach', () => {
    expect(validate('inference-advanced-06', snapshot,
      makeSnapshot({ weightPrecision: 'fp8', batchSize: 16 })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('inference-advanced-06', snapshot,
      makeSnapshot({ modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('inference-advanced-06', snapshot,
      makeSnapshot({ gpuId: 'h100-sxm' })).valid).toBe(false);
  });
});

describe('IA07 — Long Context Inference', () => {
  const snapshot = makeSnapshot({ inputSeqLen: 128000, kvCachePrecision: 'bf16' });

  it('correct: change kvCachePrecision', () => {
    expect(validate('inference-advanced-07', snapshot,
      makeSnapshot({ inputSeqLen: 128000, kvCachePrecision: 'fp8' })).valid).toBe(true);
  });
  it('correct: change TP AND KV precision (complementary)', () => {
    expect(validate('inference-advanced-07', snapshot,
      makeSnapshot({ inputSeqLen: 128000, tensorParallel: 16, kvCachePrecision: 'fp8' })).valid).toBe(true);
  });
  it('wrong: only change TP without KV quant', () => {
    expect(validate('inference-advanced-07', snapshot,
      makeSnapshot({ inputSeqLen: 128000, tensorParallel: 16, kvCachePrecision: 'bf16' })).valid).toBe(false);
  });
  it('wrong: change model', () => {
    expect(validate('inference-advanced-07', snapshot,
      makeSnapshot({ inputSeqLen: 128000, modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('inference-advanced-07', snapshot,
      makeSnapshot({ inputSeqLen: 128000, gpuId: 'rtx-4090' })).valid).toBe(false);
  });
  it('wrong: change inputSeqLen (protected)', () => {
    expect(validate('inference-advanced-07', snapshot,
      makeSnapshot({ inputSeqLen: 4096 })).valid).toBe(false);
  });
});

describe('IA08 — Latency SLA Optimization (open-ended)', () => {
  const snapshot = makeSnapshot();

  it('correct: adjust batch', () => {
    expect(validate('inference-advanced-08', snapshot,
      makeSnapshot({ batchSize: 4 })).valid).toBe(true);
  });
  it('correct: adjust batch AND quantize (complementary)', () => {
    expect(validate('inference-advanced-08', snapshot,
      makeSnapshot({ batchSize: 4, weightPrecision: 'fp8' })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('inference-advanced-08', snapshot,
      makeSnapshot({ modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('inference-advanced-08', snapshot,
      makeSnapshot({ gpuId: 'rtx-4090' })).valid).toBe(false);
  });
});

describe('IA09 — Running MoE on Consumer Hardware', () => {
  const snapshot = makeSnapshot({ weightPrecision: 'bf16', tensorParallel: 1, expertParallel: 1 });

  it('correct: quantize + increase TP', () => {
    expect(validate('inference-advanced-09', snapshot,
      makeSnapshot({ weightPrecision: 'int8', tensorParallel: 4 })).valid).toBe(true);
  });
  it('correct: INT4 + TP=4', () => {
    expect(validate('inference-advanced-09', snapshot,
      makeSnapshot({ weightPrecision: 'int4', tensorParallel: 4 })).valid).toBe(true);
  });
  it('correct: quantize + EP=4 (expert parallelism)', () => {
    expect(validate('inference-advanced-09', snapshot,
      makeSnapshot({ weightPrecision: 'int8', expertParallel: 4 })).valid).toBe(true);
  });
  it('correct: INT4 + EP=4', () => {
    expect(validate('inference-advanced-09', snapshot,
      makeSnapshot({ weightPrecision: 'int4', expertParallel: 4 })).valid).toBe(true);
  });
  it('wrong: change model', () => {
    expect(validate('inference-advanced-09', snapshot,
      makeSnapshot({ weightPrecision: 'int8', tensorParallel: 4, modelId: 'llama3.3-70b' })).valid).toBe(false);
  });
  it('wrong: change GPU', () => {
    expect(validate('inference-advanced-09', snapshot,
      makeSnapshot({ weightPrecision: 'int8', tensorParallel: 4, gpuId: 'h100-sxm' })).valid).toBe(false);
  });
  // Note: "only quantize without TP or EP" now passes expectedChanges (tensorParallel: 'increased' removed).
  // That's fine — winning criteria handle it: INT8/TP=1 → OOM, INT4/TP=1 → memUtil > 90%.
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
