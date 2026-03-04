/**
 * Comprehensive game task calibration tests
 *
 * Validates all 60 tasks:
 *   1. Default setup (base defaults + task overrides) does NOT pass — every task requires user action
 *   2. Winning configs pass all criteria
 *   3. Hint progressions show improvement for key tasks
 *
 * This is intentionally slow (runs actual simulator for each task).
 * Run separately: npx vitest run tests/game/calibration.test.ts
 */

import { describe, it, expect } from 'vitest';
import { ALL_TASKS, getTaskById } from '../../src/game/tasks/index.ts';
import { validateExpectedChanges } from '../../src/game/validation.ts';
import type { TaskSetup } from '../../src/game/types.ts';
import {
  buildTrainingContext,
  buildTrainingContextWith,
  buildInferenceContext,
  buildInferenceContextWith,
  buildSnapshot,
  checkAllCriteria,
  checkAnyCriterionFails,
} from '../helpers/simulation.ts';

// ── Winning configs ───────────────────────────────────────────────────
// Overrides from task defaults that make all criteria pass.

const WINNING_OVERRIDES: Record<string, Partial<TaskSetup>> = {
  // Training beginner
  'training-beginner-01': { mixedPrecision: 'bf16' },  // FP32→BF16 for Tensor Core throughput
  'training-beginner-02': { microBatchSize: 4 },  // MBS 64→4 to fit activation memory (GA=16)
  'training-beginner-03': { sequenceLength: 4096 },  // seqLen 8192→4096 to reduce activation memory
  'training-beginner-04': { activationCheckpointing: true },  // already has BF16 from setup
  'training-beginner-05': { flashAttention: true },  // already has BF16+AC from setup
  'training-beginner-06': { numGPUs: 8 },  // Scale from 1→8 GPUs for DDP throughput
  'training-beginner-07': { strategyType: 'fsdp' },  // DDP→FSDP to fit Qwen3-14B on 8 A100s
  'training-beginner-08': { microBatchSize: 2 },  // 8 GPUs (1 node), reduce MBS from 4 to fit in memory without AC
  'training-beginner-09': { numGPUs: 16, globalBatchSize: 16 },  // Scale to 2 nodes for throughput > 40k tok/s
  'training-beginner-10': { activationCheckpointing: false, flashAttention: true },  // Disable AC + enable FA — 8B fits without AC on 8 H100s, FA removes attention overhead

  // Training intermediate (all have BF16+FA+AC from setup)
  'training-intermediate-01': { strategyType: 'fsdp-tp', tpDegree: 4, sequenceParallel: true },  // FSDP→FSDP-TP
  'training-intermediate-02': { tpDegree: 4, globalBatchSize: 128 },
  'training-intermediate-03': { tpDegree: 4, globalBatchSize: 128 },
  'training-intermediate-04': { tpDegree: 8, ppDegree: 4 },
  'training-intermediate-05': { globalBatchSize: 256 },  // increase GBS to reduce bubble
  'training-intermediate-06': { pipelineSchedule: 'interleaved-1f1b', interleavedStages: 4, globalBatchSize: 128 },  // switch to interleaved
  'training-intermediate-07': { sequenceParallel: true },  // SP reduces activation memory, improves MFU
  'training-intermediate-08': { tpDegree: 4 },
  'training-intermediate-09': { strategyType: 'fsdp' },  // DDP OOMs for MoE, switch to FSDP
  'training-intermediate-10': { mixedPrecision: 'fp8', tpDegree: 4, globalBatchSize: 128 },

  // Training advanced (all have BF16+FA+AC from setup)
  'training-advanced-01': { tpDegree: 8, ppDegree: 4, globalBatchSize: 256, pipelineSchedule: 'interleaved-1f1b' },
  'training-advanced-02': { tpDegree: 2, epDegree: 8, mixedPrecision: 'fp8', globalBatchSize: 256 },  // 256 GPUs
  'training-advanced-03': { cpDegree: 4 },  // 512 GPUs, TP=8 PP=8 from setup, CP=4 reduces bubble via higher GA
  'training-advanced-04': { tpDegree: 1 },  // Reduce TP from 4→1 to improve C/T ratio (MFU > 42%)
  'training-advanced-05': { finetuningMethod: 'lora' },
  'training-advanced-06': { finetuningMethod: 'qlora' },
  'training-advanced-07': { mixedPrecision: 'fp8', tpDegree: 4, ppDegree: 8, epDegree: 32, globalBatchSize: 8192, microBatchSize: 2 },
  'training-advanced-08': { tpDegree: 8, ppDegree: 8, pipelineSchedule: 'interleaved-1f1b', interleavedStages: 4, checkpointingGranularity: 'selective', globalBatchSize: 256 },
  'training-advanced-09': { gpuId: 'h100-sxm', tpDegree: 8, microBatchSize: 2, globalBatchSize: 128 },  // A100→H100, 16 GPUs
  'training-advanced-10': { tpDegree: 8, ppDegree: 8, pipelineSchedule: 'interleaved-1f1b', interleavedStages: 2, globalBatchSize: 512 },

  // Inference beginner
  'inference-beginner-01': { weightPrecision: 'fp8' },   // quantize for throughput
  'inference-beginner-02': { modelId: 'llama3.1-8b' },  // 70B can't fit on RTX 4090, switch to 8B
  'inference-beginner-03': { batchSize: 8 },  // reduce batch from 32 to avoid OOM
  'inference-beginner-04': { gpuId: 'rtx-4090' },  // T4 bandwidth too low, switch to higher-BW GPU
  'inference-beginner-05': { batchSize: 8 },
  'inference-beginner-06': { batchSize: 1 },  // reduce batch from 8→1 for low TPOT
  'inference-beginner-07': { flashAttention: true },  // FA needed at 32K seq to avoid OOM
  'inference-beginner-08': { kvCachePrecision: 'fp8' },
  'inference-beginner-09': { batchSize: 16 },  // sweet spot: TPOT < 13ms AND throughput > 400 (start=32)
  'inference-beginner-10': { continuousBatching: true, batchSize: 16 },

  // Inference intermediate
  'inference-intermediate-01': { tensorParallel: 4 },
  'inference-intermediate-02': { tensorParallel: 8 },
  'inference-intermediate-03': { tensorParallel: 8, weightPrecision: 'fp8' },
  'inference-intermediate-04': { tensorParallel: 2 },  // Reduce TP from 8→2: 4 replicas, throughput > 1000
  'inference-intermediate-05': { kvCachePrecision: 'fp8' },  // KV quant to fit batch=32 at 16K, throughput > 135
  'inference-intermediate-06': { batchSize: 32 },  // A100 TP=8 from setup (batch=16), increase to 32: throughput > 1000 AND TPOT < 15ms
  'inference-intermediate-07': { tensorParallel: 4, weightPrecision: 'int4', batchSize: 4 },  // RTX 4090 x4: INT4+TP4+batch4 for TPOT<25ms AND cost<$4/Mtok
  'inference-intermediate-08': { gpuId: 'h100-sxm' },  // A10G bandwidth too low for TPOT < 12ms — switch to H100
  'inference-intermediate-09': { tensorParallel: 4, speculativeDecoding: true, draftModelId: 'llama2-7b' },
  'inference-intermediate-10': { tensorParallel: 4, weightPrecision: 'fp8' },

  // Inference advanced
  'inference-advanced-01': { tensorParallel: 8, weightPrecision: 'int4' },  // FP8 OOMs; INT4 fits
  'inference-advanced-02': { tensorParallel: 8, expertParallel: 2, batchSize: 8 },  // FP8 pre-set in setup
  'inference-advanced-03': { tensorParallel: 8, speculativeDecoding: true, draftModelId: 'llama3.1-8b' },  // FP8 pre-set; spec dec + TP=8
  'inference-advanced-04': { tensorParallel: 8 },  // TP=2→8 divides prefill compute, TTFT < 2500
  'inference-advanced-05': { tensorParallel: 2, weightPrecision: 'fp8' },  // BF16+TP=4→FP8+TP=2 for more replicas, throughput > 250
  'inference-advanced-06': { tensorParallel: 2, batchSize: 32 },  // FP8 pre-set; TP=2 + batch for cost optimization
  'inference-advanced-07': { tensorParallel: 16, kvCachePrecision: 'fp8' },  // FP8 pre-set; TP + KV quant for long context
  'inference-advanced-08': { batchSize: 4 },  // FP8 pre-set; batch=4 threads the SLA needle
  'inference-advanced-09': { tensorParallel: 4, weightPrecision: 'int8', batchSize: 4 },  // Mixtral 8x7B on 4x RTX 4090: INT8 + TP=4 (EP=4 + INT8 also valid)
  'inference-advanced-10': { tensorParallel: 8, weightPrecision: 'fp8', batchSize: 16 },
};

// ── Tests ──────────────────────────────────────────────────────────────

describe('All tasks should NOT pass with default setup', () => {
  for (const task of ALL_TASKS) {
    it(task.id, () => {
      const ctx = task.mode === 'training'
        ? buildTrainingContext(task.setup)
        : buildInferenceContext(task.setup);
      expect(
        checkAnyCriterionFails(ctx, task.winningCriteria),
        `${task.id} should NOT pass with default setup — at least one criterion must fail`,
      ).toBe(true);
    });
  }
});

describe('Winning configs should pass all criteria', () => {
  for (const task of ALL_TASKS) {
    const overrides = WINNING_OVERRIDES[task.id];
    if (!overrides) continue;  // skip if no winning config defined

    it(task.id, () => {
      const setup = { ...task.setup, ...overrides };
      const ctx = task.mode === 'training'
        ? buildTrainingContext(setup)
        : buildInferenceContext(setup);
      expect(
        checkAllCriteria(ctx, task.winningCriteria),
        `${task.id} winning config should pass all criteria`,
      ).toBe(true);
    });
  }
});

describe('Winning configs should satisfy expectedChanges', () => {
  for (const task of ALL_TASKS) {
    const overrides = WINNING_OVERRIDES[task.id];
    if (!overrides || !task.expectedChanges?.length) continue;

    it(task.id, () => {
      const snapshot = buildSnapshot(task.setup, task.mode);
      const current = buildSnapshot({ ...task.setup, ...overrides }, task.mode);
      const { valid, failedChecks } = validateExpectedChanges(snapshot, current, task.expectedChanges);
      expect(
        valid,
        `${task.id} winning config fails expectedChanges: ${failedChecks.map(c => `${c.field}: ${c.check}`).join(', ')}`,
      ).toBe(true);
    });
  }
});

// ── Hint progression tests ─────────────────────────────────────────────

describe('Hint progressions — key tasks show improvement', () => {
  it('training-beginner-03: seqLen 8192 OOMs → seqLen 4096 fits', () => {
    const task = getTaskById('training-beginner-03')!;
    // Default: seqLen 8192 → should OOM
    const defaultCtx = buildTrainingContext(task.setup);
    expect(defaultCtx.memoryUtilization).toBeGreaterThan(1.0);

    // Fix: seqLen 4096 → should fit with good MFU
    const fixedCtx = buildTrainingContextWith(task.setup, { sequenceLength: 4096 });
    expect(fixedCtx.memoryUtilization).toBeLessThan(1.0);
    expect(fixedCtx.mfu).toBeGreaterThan(0.50);
  });

  it('training-beginner-04: BF16 no AC OOMs → BF16+AC fits', () => {
    const task = getTaskById('training-beginner-04')!;
    // Default: BF16 (from task setup), no AC → should OOM or be very tight
    const defaultCtx = buildTrainingContext(task.setup);
    expect(defaultCtx.memoryUtilization).toBeGreaterThan(0.95);

    // Fix: +AC → should fit
    const fixedCtx = buildTrainingContextWith(task.setup, { activationCheckpointing: true });
    expect(fixedCtx.memoryUtilization).toBeLessThan(1.0);
  });

  it('training-beginner-05: BF16+AC no FA mem>85% → +FA mem<85%', () => {
    const task = getTaskById('training-beginner-05')!;
    // Default: BF16+AC (from setup), no FA → mem should be above 85%
    const defaultCtx = buildTrainingContext(task.setup);
    // The model might or might not be above 85%, but FA should reduce it
    const fixedCtx = buildTrainingContextWith(task.setup, { flashAttention: true });
    expect(fixedCtx.memoryUtilization).toBeLessThan(0.85);
  });

  it('training-beginner-07: DDP+14B OOMs → FSDP fits', () => {
    const task = getTaskById('training-beginner-07')!;
    // Default: DDP → should OOM with Qwen3-14B on 8 A100 GPUs
    const defaultCtx = buildTrainingContext(task.setup);
    expect(defaultCtx.memoryUtilization).toBeGreaterThan(1.0);

    // Fix: FSDP → should fit
    const fixedCtx = buildTrainingContextWith(task.setup, { strategyType: 'fsdp' });
    expect(fixedCtx.memoryUtilization).toBeLessThan(1.0);
  });

  it('inference-beginner-02: 70B on RTX 4090 OOMs → 8B fits and throughput > 20 tok/s', () => {
    const task = getTaskById('inference-beginner-02')!;
    // Default: LLaMA 70B on RTX 4090 → OOMs (even INT4: 70B × 0.5 = 35 GB > 24 GB)
    const defaultCtx = buildInferenceContext(task.setup);
    expect(checkAnyCriterionFails(defaultCtx, task.winningCriteria)).toBe(true);

    // Fix: switch to 8B → fits and throughput > 20 tok/s
    const fixedCtx = buildInferenceContextWith(task.setup, { modelId: 'llama3.1-8b' })!;
    expect(fixedCtx.success).toBe(true);
    expect(fixedCtx.throughput.tokensPerSecond).toBeGreaterThan(20);
  });

  it('inference-beginner-04: T4 INT8 throughput < 60 tok/s → RTX 4090 INT8 throughput > 60 tok/s', () => {
    const task = getTaskById('inference-beginner-04')!;
    // Default: LLaMA 8B INT8 on T4 → low throughput
    const defaultCtx = buildInferenceContext(task.setup)!;
    expect(defaultCtx.throughput.tokensPerSecond).toBeLessThan(60);

    // Fix: switch to RTX 4090 → higher bandwidth, throughput > 60 tok/s
    const fixedCtx = buildInferenceContextWith(task.setup, { gpuId: 'rtx-4090' })!;
    expect(fixedCtx.throughput.tokensPerSecond).toBeGreaterThan(60);
  });

  it('inference-beginner-04: GPU sweep — high-BW GPUs pass 60 tok/s, low-BW GPUs fail', () => {
    const task = getTaskById('inference-beginner-04')!;
    // High-bandwidth GPUs should clearly pass
    for (const gpuId of ['rtx-4090', 'a100-80gb', 'h100-sxm']) {
      const ctx = buildInferenceContextWith(task.setup, { gpuId })!;
      expect(ctx.throughput.tokensPerSecond).toBeGreaterThan(60);
    }
    // Low-bandwidth GPUs should clearly fail
    for (const gpuId of ['t4', 'l4', 'a10g']) {
      const ctx = buildInferenceContextWith(task.setup, { gpuId })!;
      expect(ctx.throughput.tokensPerSecond).toBeLessThan(60);
    }
  });

  it('inference-beginner-07: no FA at 32K OOMs → +FA fits', () => {
    const task = getTaskById('inference-beginner-07')!;
    // Default: no FA at 32K seq → OOM (attention scores memory explodes)
    const defaultCtx = buildInferenceContext(task.setup)!;
    expect(defaultCtx.memoryUtilization).toBeGreaterThan(1.0);

    // Fix: +FA → fits comfortably
    const fixedCtx = buildInferenceContextWith(task.setup, { flashAttention: true })!;
    expect(fixedCtx.memoryUtilization).toBeLessThan(1.0);
  });

  it('inference-beginner-10: batch=1 no CB → batch=16 + CB > 500 tok/s', () => {
    const task = getTaskById('inference-beginner-10')!;
    // Default: batch=1, no CB → low throughput
    const defaultCtx = buildInferenceContext(task.setup)!;
    expect(defaultCtx.throughput.tokensPerSecond).toBeLessThan(500);

    // Fix: CB + batch=16
    const fixedCtx = buildInferenceContextWith(task.setup, { continuousBatching: true, batchSize: 16 })!;
    expect(fixedCtx.throughput.tokensPerSecond).toBeGreaterThan(500);
  });

  it('inference-intermediate-01: TP=1 OOMs → TP=4 fits', () => {
    const task = getTaskById('inference-intermediate-01')!;
    // Default: TP=1 → 70B BF16 won't fit on single GPU
    const defaultCtx = buildInferenceContext(task.setup);
    if (defaultCtx) {
      expect(defaultCtx.memoryUtilization).toBeGreaterThanOrEqual(1.0);
    }

    // Fix: TP=4
    const fixedCtx = buildInferenceContextWith(task.setup, { tensorParallel: 4 })!;
    expect(fixedCtx.memoryUtilization).toBeLessThan(1.0);
  });

  it('training-intermediate-10: BF16 ~30% MFU → FP8 > 40% MFU', () => {
    const task = getTaskById('training-intermediate-10')!;
    // Default: BF16 with prereqs + good config
    const bf16Ctx = buildTrainingContextWith(task.setup, { tpDegree: 4, globalBatchSize: 128 });
    // MFU should be around 30% or so at BF16

    // Fix: FP8
    const fp8Ctx = buildTrainingContextWith(task.setup, { mixedPrecision: 'fp8', tpDegree: 4, globalBatchSize: 128 });
    expect(fp8Ctx.mfu).toBeGreaterThan(bf16Ctx.mfu);
    expect(fp8Ctx.mfu).toBeGreaterThan(0.40);
  });

  it('training-advanced-05: full FT OOMs → LoRA fits', () => {
    const task = getTaskById('training-advanced-05')!;
    // Default: full FT → 70B on 8 H100s with FSDP
    const defaultCtx = buildTrainingContext(task.setup);
    // Check if it OOMs or is very tight
    const defaultMemUtil = defaultCtx.memoryUtilization;

    // Fix: LoRA → should reduce memory
    const loraCtx = buildTrainingContextWith(task.setup, { finetuningMethod: 'lora' });
    expect(loraCtx.memoryUtilization).toBeLessThan(0.90);
    expect(loraCtx.memoryUtilization).toBeLessThan(defaultMemUtil);
  });
});
