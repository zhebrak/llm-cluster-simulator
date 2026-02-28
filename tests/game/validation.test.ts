/**
 * Tests for game mode validation — criteria evaluation + expected-change validation
 */

import { describe, it, expect } from 'vitest';
import {
  resolvePath,
  evaluateCriterion,
  validateTask,
  buildValidationContext,
  validateExpectedChanges,
} from '../../src/game/validation.ts';
import type { TaskConfigSnapshot } from '../../src/game/validation.ts';
import type { WinningCriterion, ExpectedChange } from '../../src/game/types.ts';

describe('resolvePath', () => {
  it('resolves top-level fields', () => {
    expect(resolvePath({ mfu: 0.42 }, 'mfu')).toBe(0.42);
  });

  it('resolves nested dot-paths', () => {
    const obj = { latency: { ttft: 12.5, tpot: 3.2 } };
    expect(resolvePath(obj, 'latency.ttft')).toBe(12.5);
    expect(resolvePath(obj, 'latency.tpot')).toBe(3.2);
  });

  it('resolves deeply nested paths', () => {
    const obj = { a: { b: { c: { d: 99 } } } };
    expect(resolvePath(obj, 'a.b.c.d')).toBe(99);
  });

  it('returns undefined for missing paths', () => {
    expect(resolvePath({ mfu: 0.5 }, 'hfu')).toBeUndefined();
    expect(resolvePath({ a: { b: 1 } }, 'a.c')).toBeUndefined();
    expect(resolvePath({}, 'x.y.z')).toBeUndefined();
  });

  it('returns undefined for null/undefined intermediates', () => {
    expect(resolvePath({ a: null }, 'a.b')).toBeUndefined();
    expect(resolvePath(null, 'a')).toBeUndefined();
  });
});

describe('evaluateCriterion', () => {
  const make = (field: string, operator: WinningCriterion['operator'], value: number | boolean): WinningCriterion => ({
    field, operator, value, label: 'test',
  });

  it('evaluates > operator', () => {
    expect(evaluateCriterion({ mfu: 0.45 }, make('mfu', '>', 0.40))).toBe(true);
    expect(evaluateCriterion({ mfu: 0.40 }, make('mfu', '>', 0.40))).toBe(false);
    expect(evaluateCriterion({ mfu: 0.35 }, make('mfu', '>', 0.40))).toBe(false);
  });

  it('evaluates >= operator', () => {
    expect(evaluateCriterion({ mfu: 0.40 }, make('mfu', '>=', 0.40))).toBe(true);
    expect(evaluateCriterion({ mfu: 0.39 }, make('mfu', '>=', 0.40))).toBe(false);
  });

  it('evaluates < operator', () => {
    expect(evaluateCriterion({ memoryUtilization: 0.85 }, make('memoryUtilization', '<', 1.0))).toBe(true);
    expect(evaluateCriterion({ memoryUtilization: 1.2 }, make('memoryUtilization', '<', 1.0))).toBe(false);
  });

  it('evaluates <= operator', () => {
    expect(evaluateCriterion({ x: 5 }, make('x', '<=', 5))).toBe(true);
    expect(evaluateCriterion({ x: 6 }, make('x', '<=', 5))).toBe(false);
  });

  it('evaluates == operator with boolean', () => {
    expect(evaluateCriterion({ success: true }, make('success', '==', true))).toBe(true);
    expect(evaluateCriterion({ success: false }, make('success', '==', true))).toBe(false);
  });

  it('evaluates != operator', () => {
    expect(evaluateCriterion({ success: false }, make('success', '!=', true))).toBe(true);
    expect(evaluateCriterion({ success: true }, make('success', '!=', true))).toBe(false);
  });

  it('returns false for missing field', () => {
    expect(evaluateCriterion({}, make('mfu', '>', 0.40))).toBe(false);
  });

  it('evaluates nested field paths', () => {
    const ctx = { throughput: { tokensPerSecond: 5000 } };
    expect(evaluateCriterion(ctx, make('throughput.tokensPerSecond', '>', 4000))).toBe(true);
    expect(evaluateCriterion(ctx, make('throughput.tokensPerSecond', '>', 6000))).toBe(false);
  });
});

describe('validateTask', () => {
  const criteria: WinningCriterion[] = [
    { field: 'success', operator: '==', value: true, label: 'Success' },
    { field: 'mfu', operator: '>', value: 0.30, label: 'MFU > 30%' },
  ];

  it('returns all passed when all criteria met', () => {
    const result = validateTask({ success: true, mfu: 0.45 }, criteria);
    expect(result.passed).toBe(true);
    expect(result.results).toHaveLength(2);
    expect(result.results[0].met).toBe(true);
    expect(result.results[1].met).toBe(true);
  });

  it('returns failed when any criterion not met', () => {
    const result = validateTask({ success: true, mfu: 0.20 }, criteria);
    expect(result.passed).toBe(false);
    expect(result.results[0].met).toBe(true);
    expect(result.results[1].met).toBe(false);
  });

  it('returns all unmet for null context', () => {
    const result = validateTask(null, criteria);
    expect(result.passed).toBe(false);
    expect(result.results).toHaveLength(2);
    expect(result.results.every(r => !r.met)).toBe(true);
  });
});

// ── validateExpectedChanges ──────────────────────────────────────────

/** Helper: create a base snapshot with defaults */
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

function makeChange(field: string, check: ExpectedChange['check']): ExpectedChange {
  return { field, check, label: `test: ${field} ${check}` };
}

describe('validateExpectedChanges — check semantics', () => {
  const snapshot = makeSnapshot();

  it('changed: passes when field differs from snapshot', () => {
    const current = makeSnapshot({ precision: 'bf16' });
    const result = validateExpectedChanges(snapshot, current, [makeChange('precision', 'changed')]);
    expect(result.valid).toBe(true);
    expect(result.failedChecks).toHaveLength(0);
  });

  it('changed: fails when field same as snapshot', () => {
    const current = makeSnapshot(); // identical
    const result = validateExpectedChanges(snapshot, current, [makeChange('precision', 'changed')]);
    expect(result.valid).toBe(false);
    expect(result.failedChecks).toHaveLength(1);
  });

  it('unchanged: passes when field same as snapshot', () => {
    const current = makeSnapshot();
    const result = validateExpectedChanges(snapshot, current, [makeChange('modelId', 'unchanged')]);
    expect(result.valid).toBe(true);
  });

  it('unchanged: fails when field differs', () => {
    const current = makeSnapshot({ modelId: 'llama3.3-70b' });
    const result = validateExpectedChanges(snapshot, current, [makeChange('modelId', 'unchanged')]);
    expect(result.valid).toBe(false);
  });

  it('increased: passes when current > snapshot', () => {
    const current = makeSnapshot({ numGPUs: 8 });
    const result = validateExpectedChanges(snapshot, current, [makeChange('numGPUs', 'increased')]);
    expect(result.valid).toBe(true);
  });

  it('increased: fails when current <= snapshot', () => {
    const current = makeSnapshot({ numGPUs: 1 });
    const result = validateExpectedChanges(snapshot, current, [makeChange('numGPUs', 'increased')]);
    expect(result.valid).toBe(false);
  });

  it('increased: fails when current equals snapshot', () => {
    const current = makeSnapshot();
    const result = validateExpectedChanges(snapshot, current, [makeChange('numGPUs', 'increased')]);
    expect(result.valid).toBe(false);
  });

  it('decreased: passes when current < snapshot', () => {
    const current = makeSnapshot({ microBatchSize: 4 });
    const result = validateExpectedChanges(snapshot, current, [makeChange('microBatchSize', 'decreased')]);
    expect(result.valid).toBe(true);
  });

  it('decreased: fails when current >= snapshot', () => {
    const current = makeSnapshot({ microBatchSize: 64 });
    const result = validateExpectedChanges(snapshot, current, [makeChange('microBatchSize', 'decreased')]);
    expect(result.valid).toBe(false);
  });

  it('enabled: passes when current === true', () => {
    const current = makeSnapshot({ activationCheckpointing: true });
    const result = validateExpectedChanges(snapshot, current, [makeChange('activationCheckpointing', 'enabled')]);
    expect(result.valid).toBe(true);
  });

  it('enabled: fails when current !== true', () => {
    const current = makeSnapshot({ activationCheckpointing: false });
    const result = validateExpectedChanges(snapshot, current, [makeChange('activationCheckpointing', 'enabled')]);
    expect(result.valid).toBe(false);
  });

  it('disabled: passes when current === false', () => {
    const current = makeSnapshot({ activationCheckpointing: false });
    const result = validateExpectedChanges(snapshot, current, [makeChange('activationCheckpointing', 'disabled')]);
    expect(result.valid).toBe(true);
  });

  it('disabled: fails when current !== false', () => {
    const current = makeSnapshot({ activationCheckpointing: true });
    const result = validateExpectedChanges(snapshot, current, [makeChange('activationCheckpointing', 'disabled')]);
    expect(result.valid).toBe(false);
  });
});

describe('validateExpectedChanges — composite', () => {
  const snapshot = makeSnapshot();

  it('all checks pass → valid with empty failedChecks', () => {
    const current = makeSnapshot({ precision: 'bf16' }); // changed precision, unchanged modelId
    const changes = [
      makeChange('precision', 'changed'),
      makeChange('modelId', 'unchanged'),
    ];
    const result = validateExpectedChanges(snapshot, current, changes);
    expect(result.valid).toBe(true);
    expect(result.failedChecks).toHaveLength(0);
  });

  it('any check fails → invalid with failing items', () => {
    const current = makeSnapshot({ precision: 'bf16', modelId: 'llama3.3-70b' }); // changed both
    const changes = [
      makeChange('precision', 'changed'),
      makeChange('modelId', 'unchanged'), // this will fail
    ];
    const result = validateExpectedChanges(snapshot, current, changes);
    expect(result.valid).toBe(false);
    expect(result.failedChecks).toHaveLength(1);
    expect(result.failedChecks[0].field).toBe('modelId');
  });

  it('empty expectedChanges → valid', () => {
    const result = validateExpectedChanges(snapshot, snapshot, []);
    expect(result.valid).toBe(true);
    expect(result.failedChecks).toHaveLength(0);
  });

  it('undefined expectedChanges → valid (backward compat)', () => {
    const result = validateExpectedChanges(snapshot, snapshot, undefined);
    expect(result.valid).toBe(true);
    expect(result.failedChecks).toHaveLength(0);
  });

  it('multiple failures are all reported', () => {
    const current = makeSnapshot(); // no changes at all
    const changes = [
      makeChange('precision', 'changed'),
      makeChange('numGPUs', 'increased'),
      makeChange('modelId', 'unchanged'), // this passes
    ];
    const result = validateExpectedChanges(snapshot, current, changes);
    expect(result.valid).toBe(false);
    expect(result.failedChecks).toHaveLength(2);
    const failedFields = result.failedChecks.map(c => c.field);
    expect(failedFields).toContain('precision');
    expect(failedFields).toContain('numGPUs');
  });
});

describe('buildValidationContext', () => {
  it('returns null for null mode', () => {
    const ctx = buildValidationContext({} as any, null);
    expect(ctx).toBeNull();
  });

  it('returns null when no training result exists', () => {
    const simState = { training: { result: null, metrics: null } } as any;
    expect(buildValidationContext(simState, 'training')).toBeNull();
  });

  it('builds training context from result and metrics', () => {
    const simState = {
      training: {
        result: { success: true },
        metrics: {
          mfu: 0.42,
          hfu: 0.55,
          memoryUtilization: 0.75,
          tokensPerSecond: 50000,
          samplesPerSecond: 100,
          tflopsPerGPU: 150,
          communicationOverhead: 0.15,
          pipelineBubble: 0.05,
          stepTimeMs: 500,
          overlapHiddenFraction: 0.60,
          communicationGrossMs: 50,
          communicationExposedMs: 20,
          memoryPerGPU: { parameters: 14e9, total: 60e9 },
          timing: { forward: 100, backward: 200, total: 500 },
        },
      },
    } as any;

    const ctx = buildValidationContext(simState, 'training') as any;
    expect(ctx).not.toBeNull();
    expect(ctx.success).toBe(true);
    expect(ctx.mfu).toBe(0.42);
    expect(ctx.memoryUtilization).toBe(0.75);
    expect(ctx.timing.forward).toBe(100);
  });

  it('returns null when no inference result exists', () => {
    const simState = { inference: { result: null } } as any;
    expect(buildValidationContext(simState, 'inference')).toBeNull();
  });

  it('builds inference context from result', () => {
    const simState = {
      inference: {
        result: {
          success: true,
          memory: { weights: 14e9, kvCache: 2e9, total: 18e9 },
          latency: { ttft: 50, tpot: 10, totalLatency: 500 },
          throughput: { tokensPerSecond: 100 },
          utilization: { memoryCapacityUtilization: 0.85 },
          kvCacheState: { memoryUsed: 2e9 },
          maxConcurrentRequests: 10,
          continuousBatching: true,
        },
      },
    } as any;

    const ctx = buildValidationContext(simState, 'inference') as any;
    expect(ctx).not.toBeNull();
    expect(ctx.success).toBe(true);
    expect(ctx.latency.tpot).toBe(10);
    expect(ctx.throughput.tokensPerSecond).toBe(100);
    expect(ctx.memoryUtilization).toBe(0.85);
    // costPerMillionTokens is computed from config store pricing, not from result
    expect(typeof ctx.costPerMillionTokens).toBe('number');
    expect(ctx.costPerMillionTokens).toBeGreaterThan(0);
  });
});
