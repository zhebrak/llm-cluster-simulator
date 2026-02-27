/**
 * Tests for game task definitions — validates all 60 tasks
 */

import { describe, it, expect } from 'vitest';
import { ALL_TASKS, getTaskById, getTasksForLevel } from '../../src/game/tasks/index.ts';
import type { GameMode, GameDifficulty } from '../../src/game/types.ts';
import { getModel } from '../../src/core/models/registry.ts';
import { getGPU } from '../../src/core/hardware/gpu.ts';
import type { SimulationMetrics } from '../../src/core/simulation/engine.ts';
import type { InferenceSimulationResult } from '../../src/types/inference.ts';

/**
 * Derive valid fields from the actual context shape.
 * Training context = { success: boolean, ...SimulationMetrics }
 * Inference context = { ...InferenceSimulationResult, memoryUtilization, costPerMillionTokens }
 */

// Build a mock training context with all possible fields
const mockTrainingMetrics: SimulationMetrics = {
  tokensPerSecond: 0, samplesPerSecond: 0, tflopsPerGPU: 0,
  mfu: 0, hfu: 0, communicationOverhead: 0, pipelineBubble: 0,
  memoryPerGPU: { parameters: 0, gradients: 0, optimizerStates: 0, activations: 0, peakActivations: 0, temporary: 0, reserved: 0, total: 0 },
  memoryUtilization: 0,
  stepTimeMs: 0,
  timing: { forward: 0, backward: 0, optimizer: 0, communication: 0, overlap: 0, scaleOverhead: 0, total: 0 },
  communicationGrossMs: 0, communicationExposedMs: 0, overlapHiddenFraction: 0,
  // Optional fields that may appear
  timeToTrainHours: 0, totalCost: 0, modelFlopsMfu: 0, fp8HwUtil: 0, resolvedStoredLayers: 0,
};
const MOCK_TRAINING_CONTEXT = { success: true, ...mockTrainingMetrics };

// Build a mock inference context
const mockInferenceResult: InferenceSimulationResult = {
  success: true,
  memory: { weights: 0, kvCache: 0, activations: 0, overhead: 0, total: 0 },
  kvCacheState: { currentSeqLen: 0, batchSize: 0, memoryUsed: 0, memoryPerToken: 0, utilizationPercent: 0 },
  latency: { ttft: 0, tpot: 0, totalLatency: 0, prefillTime: 0, decodeTime: 0 },
  throughput: { tokensPerSecond: 0, requestsPerSecond: 0, prefillTokensPerSecond: 0, decodeTokensPerSecond: 0 },
  utilization: { computeUtilization: 0, rooflineAttainment: 0, memoryCapacityUtilization: 0, isComputeBound: false, isMemoryBound: false, bottleneck: 'memory' as const },
  speculative: { expectedAcceptedTokens: 0, speedup: 0, draftModelOverhead: 0, verificationTime: 0, effectiveTpot: 0 },
  maxConcurrentRequests: 0,
  continuousBatching: false,
  errors: [], warnings: [], recommendations: [], events: [],
};
// Mirror buildValidationContext's explicit field picks (no errors/warnings/recommendations/events)
const MOCK_INFERENCE_CONTEXT = {
  success: mockInferenceResult.success,
  memory: mockInferenceResult.memory,
  kvCacheState: mockInferenceResult.kvCacheState,
  latency: mockInferenceResult.latency,
  throughput: mockInferenceResult.throughput,
  utilization: mockInferenceResult.utilization,
  speculative: mockInferenceResult.speculative,
  continuousBatching: mockInferenceResult.continuousBatching,
  maxConcurrentRequests: mockInferenceResult.maxConcurrentRequests,
  memoryUtilization: 0,
  costPerMillionTokens: 0,
};

/**
 * Extract all valid dot-paths from a context object (up to 2 levels deep).
 */
function extractValidPaths(obj: Record<string, unknown>, prefix = ''): Set<string> {
  const paths = new Set<string>();
  for (const key of Object.keys(obj)) {
    const fullKey = prefix ? `${prefix}.${key}` : key;
    paths.add(fullKey);
    const val = obj[key];
    if (val && typeof val === 'object' && !Array.isArray(val)) {
      for (const subKey of Object.keys(val as Record<string, unknown>)) {
        paths.add(`${fullKey}.${subKey}`);
      }
    }
  }
  return paths;
}

const VALID_TRAINING_PATHS = extractValidPaths(MOCK_TRAINING_CONTEXT as Record<string, unknown>);
const VALID_INFERENCE_PATHS = extractValidPaths(MOCK_INFERENCE_CONTEXT as Record<string, unknown>);

function validateFieldPath(field: string, mode: GameMode): string | null {
  const validPaths = mode === 'training' ? VALID_TRAINING_PATHS : VALID_INFERENCE_PATHS;

  if (validPaths.has(field)) return null;

  // Check if the top-level part is valid (for better error messages)
  const topLevel = field.split('.')[0];
  const topLevelValid = [...validPaths].some(p => p === topLevel || p.startsWith(`${topLevel}.`));

  if (!topLevelValid) {
    return `Unknown ${mode} field: "${topLevel}"`;
  }
  return `Unknown nested ${mode} field: "${field}"`;
}

describe('Game tasks — structure validation', () => {
  it('has exactly 60 tasks', () => {
    expect(ALL_TASKS.length).toBe(60);
  });

  it('has 10 tasks per level (6 levels)', () => {
    const modes: GameMode[] = ['training', 'inference'];
    const diffs: GameDifficulty[] = ['beginner', 'intermediate', 'advanced'];

    for (const mode of modes) {
      for (const diff of diffs) {
        const tasks = getTasksForLevel(mode, diff);
        expect(tasks.length, `${mode}-${diff} should have 10 tasks`).toBe(10);
      }
    }
  });

  it('has unique task IDs', () => {
    const ids = ALL_TASKS.map(t => t.id);
    const uniqueIds = new Set(ids);
    expect(uniqueIds.size, `Duplicate IDs found: ${ids.filter((id, i) => ids.indexOf(id) !== i)}`).toBe(ids.length);
  });

  it('has sequential order 0-9 within each level', () => {
    const modes: GameMode[] = ['training', 'inference'];
    const diffs: GameDifficulty[] = ['beginner', 'intermediate', 'advanced'];

    for (const mode of modes) {
      for (const diff of diffs) {
        const tasks = getTasksForLevel(mode, diff);
        const orders = tasks.map(t => t.order).sort((a, b) => a - b);
        expect(orders, `${mode}-${diff} should have orders 0-9`).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      }
    }
  });

  it('all tasks are retrievable by ID', () => {
    for (const task of ALL_TASKS) {
      const found = getTaskById(task.id);
      expect(found, `Task ${task.id} should be retrievable`).toBeDefined();
      expect(found?.id).toBe(task.id);
    }
  });
});

describe('Game tasks — required fields', () => {
  for (const task of ALL_TASKS) {
    it(`${task.id} has all required fields`, () => {
      expect(task.title.length).toBeGreaterThan(0);
      expect(task.briefing.length).toBeGreaterThan(10);
      expect(task.concept.length).toBeGreaterThan(0);
      expect(task.successExplanation.length).toBeGreaterThan(10);
      expect(task.hints.length).toBeGreaterThanOrEqual(2);
      expect(task.winningCriteria.length).toBeGreaterThanOrEqual(1);
      expect(task.setup.modelId.length).toBeGreaterThan(0);
      expect(task.setup.gpuId.length).toBeGreaterThan(0);
    });
  }
});

describe('Game tasks — setup references valid IDs', () => {
  for (const task of ALL_TASKS) {
    it(`${task.id} references valid model: ${task.setup.modelId}`, () => {
      const model = getModel(task.setup.modelId);
      expect(model, `Model "${task.setup.modelId}" not found in registry`).toBeDefined();
    });

    it(`${task.id} references valid GPU: ${task.setup.gpuId}`, () => {
      const gpu = getGPU(task.setup.gpuId);
      expect(gpu, `GPU "${task.setup.gpuId}" not found in registry`).toBeDefined();
    });
  }
});

describe('Game tasks — criterion field paths are valid', () => {
  for (const task of ALL_TASKS) {
    for (const criterion of task.winningCriteria) {
      it(`${task.id}: field "${criterion.field}" is valid for ${task.mode} mode`, () => {
        const error = validateFieldPath(criterion.field, task.mode);
        expect(error, error ?? '').toBeNull();
      });
    }
  }
});

describe('Game tasks — criteria have valid operators and values', () => {
  const VALID_OPERATORS = new Set(['>', '>=', '<', '<=', '==', '!=']);

  for (const task of ALL_TASKS) {
    for (const criterion of task.winningCriteria) {
      it(`${task.id}: criterion "${criterion.label}" is well-formed`, () => {
        expect(VALID_OPERATORS.has(criterion.operator), `Invalid operator: ${criterion.operator}`).toBe(true);
        expect(criterion.label.length).toBeGreaterThan(0);
        expect(criterion.value !== undefined && criterion.value !== null, 'Value must be defined').toBe(true);
      });
    }
  }
});

describe('Game tasks — training tasks have training-only fields', () => {
  const trainingTasks = ALL_TASKS.filter(t => t.mode === 'training');

  for (const task of trainingTasks) {
    it(`${task.id} may have strategyType in setup`, () => {
      // Training tasks can optionally specify strategy
      if (task.setup.strategyType) {
        const validStrategies = ['ddp', 'fsdp', 'zero-1', 'zero-3', 'fsdp-tp', 'zero1-tp', 'ddp-tp-pp', 'zero1-tp-pp', 'fsdp-tp-pp'];
        expect(validStrategies, `Invalid strategy: ${task.setup.strategyType}`).toContain(task.setup.strategyType);
      }
    });
  }
});

describe('Game tasks — inference tasks have no strategyType', () => {
  const inferenceTasks = ALL_TASKS.filter(t => t.mode === 'inference');

  for (const task of inferenceTasks) {
    it(`${task.id} has no strategyType in setup`, () => {
      expect(task.setup.strategyType, 'Inference tasks should not have strategyType').toBeUndefined();
    });
  }
});
