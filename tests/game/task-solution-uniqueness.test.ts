/**
 * Task solution uniqueness tests
 *
 * For each flagged task, tests whether alternative approaches (shortcuts)
 * can satisfy winning criteria without the expected solution.
 *
 * Three outcomes per test:
 *   UNIQUE   — shortcut does NOT satisfy criteria → expected solution is the only way
 *   GUARDED  — shortcut DOES satisfy criteria but expectedChanges blocks it (justified)
 *   PROBLEM  — shortcut works AND expectedChanges doesn't block (needs task fix)
 *
 * Run: npx vitest run tests/game/task-solution-uniqueness.test.ts
 */

import { describe, it, expect } from 'vitest';
import { getTaskById } from '../../src/game/tasks/index.ts';
import {
  buildTrainingContextWith,
  buildInferenceContext,
  buildInferenceContextWith,
  checkAllCriteria,
} from '../helpers/simulation.ts';

// ── Training task uniqueness tests ────────────────────────────────────

describe('Training task solution uniqueness', () => {

  describe('TI-06: Interleaved Scheduling — GUARDED', () => {
    // Setup: GPT-3 175B, 64 A100s, zero1-tp-pp, TP=8, PP=8, GBS=16, SP=true
    // Winning: pipelineBubble < 0.12
    // Expected: change pipelineSchedule (to interleaved)
    // Shortcut: increase GBS with 1F1B passes criteria but expectedChanges blocks it.

    const task = getTaskById('training-intermediate-06')!;

    it('GBS=128 with 1F1B achieves bubble<12% — GUARDED by expectedChanges[pipelineSchedule]', () => {
      const ctx = buildTrainingContextWith(task.setup, { globalBatchSize: 128 });
      expect(ctx.success).toBe(true);
      expect(ctx.pipelineBubble).toBeLessThan(0.12);
      // Passes criteria, but expectedChanges requires pipelineSchedule changed.
      // Justified: task specifically teaches interleaved scheduling.
      expect(task.expectedChanges!.some(c => c.field === 'pipelineSchedule')).toBe(true);
    });
  });

  describe('TI-07: Sequence Parallelism — UNIQUE', () => {
    // Setup: 70B, 16 H100s, fsdp-tp, TP=4, SP=false, GBS=128, BF16, FA, AC
    // Winning: MFU > 0.41, success=true
    // Expected: enable SP
    // TP=2 gets MFU>41% but OOMs. TP=4 gets ~40.4%. TP=8 gets ~35.6%.
    // No valid (non-OOM) config without SP achieves MFU>41%.

    const task = getTaskById('training-intermediate-07')!;

    it('no TP/GBS combo without SP achieves both success=true and MFU>41%', () => {
      const alternatives: Partial<TaskSetup>[] = [
        { tpDegree: 2, globalBatchSize: 256 },
        { tpDegree: 4, globalBatchSize: 256 },
        { tpDegree: 8, globalBatchSize: 128 },
        { tpDegree: 8, globalBatchSize: 256 },
        { tpDegree: 4, globalBatchSize: 512 },
      ];

      for (const alt of alternatives) {
        const ctx = buildTrainingContextWith(task.setup, { ...alt, sequenceParallel: false });
        const passes = checkAllCriteria(ctx, task.winningCriteria);
        // None should pass all criteria (success AND MFU>41%) without SP
        expect(passes).toBe(false);
      }
    });
  });

  describe('TI-10: Precision Frontier: FP8 — GUARDED', () => {
    // Setup: 70B, 16 H100s, fsdp-tp, BF16, FA, AC, SP
    // Winning: MFU > 0.40
    // Expected: change precision (to FP8)
    // Shortcut: BF16 TP=8 GBS=128 achieves MFU>40% but expectedChanges blocks it.

    const task = getTaskById('training-intermediate-10')!;

    it('BF16 TP=8 achieves MFU>40% — GUARDED by expectedChanges[precision]', () => {
      // Find any BF16 combo that passes
      let found = false;
      for (const tp of [2, 4, 8]) {
        for (const gbs of [128, 256, 512]) {
          const ctx = buildTrainingContextWith(task.setup, { tpDegree: tp, globalBatchSize: gbs });
          if (ctx.success && ctx.mfu > 0.40) {
            found = true;
            break;
          }
        }
        if (found) break;
      }
      // At least one BF16 combo passes winning criteria
      expect(found).toBe(true);
      // But expectedChanges requires precision changed
      expect(task.expectedChanges!.some(c => c.field === 'precision' && c.check === 'changed')).toBe(true);
    });
  });

  describe('TA-02: Expert Parallelism — UNIQUE', () => {
    // Setup: V3, 256 H100s, fsdp-tp, BF16, FA, AC, SP
    // Winning: MFU > 0.20, success=true
    // Expected: increase EP degree
    // No shortcut without EP achieves the criteria.

    const task = getTaskById('training-advanced-02')!;

    it('fsdp-tp without EP cannot achieve MFU>20% for V3', () => {
      const alternatives: Partial<TaskSetup>[] = [
        { tpDegree: 2, globalBatchSize: 256 },
        { tpDegree: 4, globalBatchSize: 256 },
        { tpDegree: 8, globalBatchSize: 256 },
        { tpDegree: 4, mixedPrecision: 'fp8' as const, globalBatchSize: 256 },
        { tpDegree: 8, mixedPrecision: 'fp8' as const, globalBatchSize: 256 },
      ];

      for (const alt of alternatives) {
        const ctx = buildTrainingContextWith(task.setup, { ...alt, epDegree: 1 });
        expect(!ctx.success || ctx.mfu <= 0.20).toBe(true);
      }
    });
  });

  describe('TA-07: Reproducing DeepSeek V3 — UNIQUE', () => {
    // Setup: V3, 2048 H800s, fsdp-tp-pp, BF16, FA, AC, SP
    // Winning: MFU > 0.35
    // Expected: change precision (to FP8)
    // No BF16 parallelism config achieves the criteria.

    const task = getTaskById('training-advanced-07')!;

    it('BF16 with published parallelism config cannot achieve MFU>35%', () => {
      const alternatives: Partial<TaskSetup>[] = [
        { tpDegree: 4, ppDegree: 8, epDegree: 32, globalBatchSize: 8192, microBatchSize: 2 },
        { tpDegree: 8, ppDegree: 4, epDegree: 16, globalBatchSize: 8192, microBatchSize: 2 },
        { tpDegree: 4, ppDegree: 4, epDegree: 16, globalBatchSize: 4096, microBatchSize: 2 },
      ];

      for (const alt of alternatives) {
        const ctx = buildTrainingContextWith(task.setup, alt);
        expect(!ctx.success || ctx.mfu <= 0.35).toBe(true);
      }
    });
  });

  describe('TA-09: GPU Architecture Comparison — UNIQUE', () => {
    // Setup: 70B, 16 A100s, fsdp-tp, BF16, FA, AC, SP, TP=8
    // Winning: gpuId == 'h100-sxm', MFU > 0.40, success=true
    // expectedChanges: gpuId changed, modelId unchanged.
    // The gpuId criterion ensures A100 configs always fail regardless of MFU.

    const task = getTaskById('training-advanced-09')!;

    it('A100 always fails gpuId criterion — UNIQUE by design', () => {
      // A100 can achieve MFU>40% but the gpuId=='h100-sxm' criterion blocks it
      const ctx = buildTrainingContextWith(task.setup, { tpDegree: 8, globalBatchSize: 128 });
      expect(ctx.gpuId).toBe('a100-80gb');
      expect(checkAllCriteria(ctx, task.winningCriteria)).toBe(false);
    });

    it('H100 achieves all criteria — GPU change IS required', () => {
      const ctx = buildTrainingContextWith(task.setup, {
        gpuId: 'h100-sxm', tpDegree: 8, microBatchSize: 2, globalBatchSize: 128,
      });
      expect(ctx.success).toBe(true);
      expect(ctx.gpuId).toBe('h100-sxm');
      expect(ctx.mfu).toBeGreaterThan(0.40);
      expect(checkAllCriteria(ctx, task.winningCriteria)).toBe(true);
    });
  });
});

// ── Inference task uniqueness tests ───────────────────────────────────

describe('Inference task solution uniqueness', () => {

  describe('IB-04: Bandwidth Bottleneck — UNIQUE', () => {
    // Setup: 8B, T4, INT8, batch=1
    // Winning: throughput > 60 tok/s
    // Expected: gpuId changed
    // INT4 on T4 cannot reach the threshold.

    const task = getTaskById('inference-beginner-04')!;

    it('INT4 on T4 cannot achieve >60 tok/s for 8B', () => {
      const ctx = buildInferenceContextWith(task.setup, { weightPrecision: 'int4' });
      const passes = ctx ? checkAllCriteria(ctx, task.winningCriteria) : false;
      expect(passes).toBe(false);
    });
  });

  describe('IB-10: Continuous Batching — GUARDED', () => {
    // Setup: 8B, H100, 1 GPU, defaults
    // Winning: throughput > 500 tok/s
    // Expected: continuousBatching enabled
    // Static batch=4 achieves >500 tok/s but expectedChanges blocks it.

    const task = getTaskById('inference-beginner-10')!;

    it('static batch=4 achieves >500 tok/s — GUARDED by expectedChanges[continuousBatching]', () => {
      const ctx = buildInferenceContextWith(task.setup, { batchSize: 4, continuousBatching: false });
      expect(ctx).not.toBeNull();
      expect(checkAllCriteria(ctx, task.winningCriteria)).toBe(true);
      // Passes criteria, but expectedChanges requires continuousBatching enabled
      expect(task.expectedChanges!.some(c => c.field === 'continuousBatching' && c.check === 'enabled')).toBe(true);
    });
  });

  describe('II-09: Speculative Decoding — GUARDED', () => {
    // Setup: 70B, 4 H100s (gpusPerNode=4)
    // Winning: TPOT < 9ms
    // Expected: speculativeDecoding enabled
    // FP8 TP=4 achieves TPOT<9ms but expectedChanges blocks it.

    const task = getTaskById('inference-intermediate-09')!;

    it('FP8 TP=4 achieves TPOT<9ms — GUARDED by expectedChanges[speculativeDecoding]', () => {
      const ctx = buildInferenceContextWith(task.setup, {
        weightPrecision: 'fp8', tensorParallel: 4,
      });
      expect(ctx).not.toBeNull();
      expect(checkAllCriteria(ctx, task.winningCriteria)).toBe(true);
      // Passes criteria, but expectedChanges requires speculativeDecoding enabled
      expect(task.expectedChanges!.some(c => c.field === 'speculativeDecoding' && c.check === 'enabled')).toBe(true);
    });
  });

  describe('II-10: FP8 Inference — GUARDED', () => {
    // Setup: 70B, 4 H100s (gpusPerNode=4)
    // Winning: throughput > 120 tok/s
    // Expected: weightPrecision changed
    // BF16 batch=2 TP=4 achieves >120 tok/s but expectedChanges blocks it.

    const task = getTaskById('inference-intermediate-10')!;

    it('BF16 batch=2 TP=4 achieves >120 tok/s — GUARDED by expectedChanges[weightPrecision]', () => {
      const ctx = buildInferenceContextWith(task.setup, {
        batchSize: 2, tensorParallel: 4,
      });
      expect(ctx).not.toBeNull();
      expect(checkAllCriteria(ctx, task.winningCriteria)).toBe(true);
      // Passes criteria, but expectedChanges requires weightPrecision changed
      expect(task.expectedChanges!.some(c => c.field === 'weightPrecision' && c.check === 'changed')).toBe(true);
    });
  });

  describe('IA-02: Expert Parallel Inference — GUARDED', () => {
    // Setup: V3, 16 H100s (gpusPerNode=8)
    // Winning: throughput > 100 tok/s, success=true
    // Expected: expertParallel increased
    // TP=8 INT4 without EP achieves >100 tok/s but expectedChanges blocks it.

    const task = getTaskById('inference-advanced-02')!;

    it('TP=8 INT4 without EP achieves >100 tok/s — GUARDED by expectedChanges[expertParallel]', () => {
      const ctx = buildInferenceContextWith(task.setup, {
        tensorParallel: 8, weightPrecision: 'int4', batchSize: 8,
      });
      expect(ctx).not.toBeNull();
      expect(checkAllCriteria(ctx, task.winningCriteria)).toBe(true);
      // Passes criteria, but expectedChanges requires expertParallel increased
      expect(task.expectedChanges!.some(c => c.field === 'expertParallel' && c.check === 'increased')).toBe(true);
    });
  });

  describe('IA-03: Speculative Decoding Mastery — UNIQUE (by threshold)', () => {
    // Setup: 405B, 16 H100s (gpusPerNode=8), FP8 pre-set
    // Winning: TPOT < 12ms, throughput > 120 tok/s
    // Expected: speculativeDecoding enabled
    // FP8+TP=16 gets TPOT=10.5 (<12) but throughput=92.9 (<120) — fails throughput.
    // FP8+TP=8 gets TPOT=19.9 (>12) — fails TPOT.
    // Only spec dec configs achieve both.

    const task = getTaskById('inference-advanced-03')!;

    it('FP8 TP=16 without spec dec fails throughput>120', () => {
      const ctx = buildInferenceContextWith(task.setup, {
        tensorParallel: 16,
      });
      expect(ctx).not.toBeNull();
      expect(checkAllCriteria(ctx, task.winningCriteria)).toBe(false);
    });

    it('FP8 TP=8 without spec dec fails TPOT<12', () => {
      const ctx = buildInferenceContextWith(task.setup, {
        tensorParallel: 8,
      });
      expect(ctx).not.toBeNull();
      expect(checkAllCriteria(ctx, task.winningCriteria)).toBe(false);
    });
  });

  describe('IA-04: Optimizing TTFT — UNIQUE', () => {
    // Setup: 70B, 8 H100s (gpusPerNode=8), TP=2, inputSeqLen=32768
    // Winning: TTFT < 2500ms
    // Expected: tensorParallel increased
    // FP8 at TP=2 cannot achieve the threshold.

    const task = getTaskById('inference-advanced-04')!;

    it('FP8 at TP=2 cannot achieve TTFT<2500ms for 70B at 32K input', () => {
      const ctx = buildInferenceContextWith(task.setup, { weightPrecision: 'fp8' });
      const passes = ctx ? checkAllCriteria(ctx, task.winningCriteria) : false;
      expect(passes).toBe(false);
    });
  });

  describe('IA-05: Multi-Replica Serving — MULTIPLE PATHS', () => {
    // Setup: 70B, 8 H100s (gpusPerNode=8), TP=4, BF16
    // Winning: throughput > 260 tok/s
    // No weightPrecision constraint — batch tuning and quantization+TP both valid.

    const task = getTaskById('inference-advanced-05')!;

    it('BF16 batch=4 achieves >260 tok/s (batch tuning path)', () => {
      const ctx = buildInferenceContextWith(task.setup, { batchSize: 4 });
      expect(ctx).not.toBeNull();
      expect(checkAllCriteria(ctx, task.winningCriteria)).toBe(true);
    });

    it('FP8 TP=2 achieves >260 tok/s (quantization+replica path)', () => {
      const ctx = buildInferenceContextWith(task.setup, { weightPrecision: 'fp8', tensorParallel: 2 });
      expect(ctx).not.toBeNull();
      expect(checkAllCriteria(ctx, task.winningCriteria)).toBe(true);
    });

    it('FP8 alone at TP=4 BS=1 does NOT achieve >260 tok/s', () => {
      const ctx = buildInferenceContextWith(task.setup, { weightPrecision: 'fp8' });
      expect(ctx).not.toBeNull();
      expect(checkAllCriteria(ctx, task.winningCriteria)).toBe(false);
    });
  });

  describe('IA-06: Cost Optimization — UNIQUE (by threshold)', () => {
    // Setup: 70B, 8 L40S (gpusPerNode=8), FP8 pre-set
    // Winning: cost < $5/M tokens
    // FP8 pre-set, so precision change is not the lesson.
    // Naive config (TP=8, batch=1) has cost ~$38 — far above threshold.
    // Player must discover TP/batch tradeoff on PCIe GPUs.

    const task = getTaskById('inference-advanced-06')!;

    it('FP8 TP=8 batch=1 (naive fix for OOM) does NOT achieve cost<$5/M', () => {
      const ctx = buildInferenceContextWith(task.setup, {
        tensorParallel: 8, batchSize: 1,
      });
      expect(ctx).not.toBeNull();
      expect(checkAllCriteria(ctx, task.winningCriteria)).toBe(false);
    });
  });

  describe('IA-08: Latency SLA — UNIQUE (by threshold)', () => {
    // Setup: 70B, 8 H100s, TP=8, batch=24, inputSeqLen=4096, FP8 pre-set
    // Winning: TTFT<500ms, TPOT<10ms, throughput>500
    // Starting config (batch=24): TTFT=~2282 (fails).
    // batch=1: throughput ~192 (fails >500).
    // batch=8: TTFT ~761 (fails <500).
    // Only batch=3-6 range satisfies all three constraints.

    const task = getTaskById('inference-advanced-08')!;

    it('starting config (FP8 batch=24) fails TTFT SLA', () => {
      const ctx = buildInferenceContext(task.setup);
      expect(ctx).not.toBeNull();
      expect(checkAllCriteria(ctx, task.winningCriteria)).toBe(false);
    });

    it('batch=1 fails throughput threshold', () => {
      const ctx = buildInferenceContextWith(task.setup, { batchSize: 1 });
      expect(ctx).not.toBeNull();
      expect(checkAllCriteria(ctx, task.winningCriteria)).toBe(false);
    });

    it('batch=8 fails TTFT threshold', () => {
      const ctx = buildInferenceContextWith(task.setup, { batchSize: 8 });
      expect(ctx).not.toBeNull();
      expect(checkAllCriteria(ctx, task.winningCriteria)).toBe(false);
    });
  });
});
