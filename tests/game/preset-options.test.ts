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
  }
});
