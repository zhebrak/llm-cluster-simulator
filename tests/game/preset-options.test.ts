/**
 * Verifies all task setup values are representable in the UI dropdowns.
 *
 * Prevents regressions where a task sets a value (e.g., weightPrecision: 'fp32')
 * that doesn't appear in the corresponding UI selector.
 */

import { describe, it, expect } from 'vitest';
import { ALL_TASKS } from '../../src/game/tasks/index.ts';

// ── UI option values (from Sidebar.tsx) ──────────────────────────────────

/** Weight precision values available in the inference quantization dropdown */
const QUANTIZATION_OPTIONS = new Set([
  'bf16', 'fp16', 'fp8', 'fp4', 'int8', 'int4',
  'q8_0', 'q6_k', 'q5_k_m', 'q4_k_m', 'q3_k_m', 'q2_k',
]);

/** Mixed precision values available in the training precision dropdown */
const PRECISION_OPTIONS = new Set([
  'fp32', 'tf32', 'fp16', 'bf16', 'fp8',
]);

/** Strategy types available in the training strategy dropdown */
const STRATEGY_OPTIONS = new Set([
  'ddp', 'fsdp', 'zero-1', 'fsdp-tp', 'zero1-tp', 'ddp-tp-pp', 'zero1-tp-pp', 'fsdp-tp-pp',
]);

/** Pipeline schedule values available in the pipeline schedule dropdown */
const PIPELINE_SCHEDULE_OPTIONS = new Set([
  '1f1b', 'interleaved-1f1b', 'dualpipe-v',
]);

/** Fine-tuning method values available in the fine-tuning dropdown */
const FINETUNING_OPTIONS = new Set([
  'full', 'lora', 'qlora',
]);

/** LoRA target module values available in the LoRA targets dropdown */
const LORA_TARGET_OPTIONS = new Set([
  'q_v', 'q_k_v_o', 'all_linear',
]);

/** Checkpointing granularity values available in the AC dropdown */
const CHECKPOINTING_GRANULARITY_OPTIONS = new Set([
  'full', 'selective',
]);

/** KV cache precision values available in the inference KV cache dropdown */
const KV_CACHE_PRECISION_OPTIONS = new Set([
  'bf16', 'fp16', 'fp8', 'fp4', 'int8', 'int4',
]);

// ── Tests ────────────────────────────────────────────────────────────────

describe('Task setup values must be representable in UI dropdowns', () => {
  const inferenceTasks = ALL_TASKS.filter(t => t.mode === 'inference');
  const trainingTasks = ALL_TASKS.filter(t => t.mode === 'training');

  for (const task of inferenceTasks) {
    if (task.setup.weightPrecision) {
      it(`${task.id}: weightPrecision '${task.setup.weightPrecision}' is in quantization dropdown`, () => {
        expect(
          QUANTIZATION_OPTIONS.has(task.setup.weightPrecision!),
          `${task.id} sets weightPrecision='${task.setup.weightPrecision}' which is not in the UI dropdown. ` +
          `Available: ${[...QUANTIZATION_OPTIONS].join(', ')}`,
        ).toBe(true);
      });
    }
    if (task.setup.kvCachePrecision) {
      it(`${task.id}: kvCachePrecision '${task.setup.kvCachePrecision}' is in KV cache dropdown`, () => {
        expect(
          KV_CACHE_PRECISION_OPTIONS.has(task.setup.kvCachePrecision!),
          `${task.id} sets kvCachePrecision='${task.setup.kvCachePrecision}' which is not in the UI dropdown. ` +
          `Available: ${[...KV_CACHE_PRECISION_OPTIONS].join(', ')}`,
        ).toBe(true);
      });
    }
  }

  for (const task of trainingTasks) {
    if (task.setup.mixedPrecision) {
      it(`${task.id}: mixedPrecision '${task.setup.mixedPrecision}' is in precision dropdown`, () => {
        expect(
          PRECISION_OPTIONS.has(task.setup.mixedPrecision!),
          `${task.id} sets mixedPrecision='${task.setup.mixedPrecision}' which is not in the UI dropdown. ` +
          `Available: ${[...PRECISION_OPTIONS].join(', ')}`,
        ).toBe(true);
      });
    }
    if (task.setup.strategyType) {
      it(`${task.id}: strategyType '${task.setup.strategyType}' is in strategy dropdown`, () => {
        expect(
          STRATEGY_OPTIONS.has(task.setup.strategyType!),
          `${task.id} sets strategyType='${task.setup.strategyType}' which is not in the UI dropdown. ` +
          `Available: ${[...STRATEGY_OPTIONS].join(', ')}`,
        ).toBe(true);
      });
    }
    if (task.setup.pipelineSchedule) {
      it(`${task.id}: pipelineSchedule '${task.setup.pipelineSchedule}' is in pipeline schedule dropdown`, () => {
        expect(
          PIPELINE_SCHEDULE_OPTIONS.has(task.setup.pipelineSchedule!),
          `${task.id} sets pipelineSchedule='${task.setup.pipelineSchedule}' which is not in the UI dropdown. ` +
          `Available: ${[...PIPELINE_SCHEDULE_OPTIONS].join(', ')}`,
        ).toBe(true);
      });
    }
    if (task.setup.finetuningMethod) {
      it(`${task.id}: finetuningMethod '${task.setup.finetuningMethod}' is in fine-tuning dropdown`, () => {
        expect(
          FINETUNING_OPTIONS.has(task.setup.finetuningMethod!),
          `${task.id} sets finetuningMethod='${task.setup.finetuningMethod}' which is not in the UI dropdown. ` +
          `Available: ${[...FINETUNING_OPTIONS].join(', ')}`,
        ).toBe(true);
      });
    }
    if (task.setup.loraTargetModules) {
      it(`${task.id}: loraTargetModules '${task.setup.loraTargetModules}' is in LoRA targets dropdown`, () => {
        expect(
          LORA_TARGET_OPTIONS.has(task.setup.loraTargetModules!),
          `${task.id} sets loraTargetModules='${task.setup.loraTargetModules}' which is not in the UI dropdown. ` +
          `Available: ${[...LORA_TARGET_OPTIONS].join(', ')}`,
        ).toBe(true);
      });
    }
    if (task.setup.checkpointingGranularity) {
      it(`${task.id}: checkpointingGranularity '${task.setup.checkpointingGranularity}' is in AC dropdown`, () => {
        expect(
          CHECKPOINTING_GRANULARITY_OPTIONS.has(task.setup.checkpointingGranularity!),
          `${task.id} sets checkpointingGranularity='${task.setup.checkpointingGranularity}' which is not in the UI dropdown. ` +
          `Available: ${[...CHECKPOINTING_GRANULARITY_OPTIONS].join(', ')}`,
        ).toBe(true);
      });
    }
  }
});
