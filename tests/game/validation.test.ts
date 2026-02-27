/**
 * Tests for game mode validation — criteria evaluation
 */

import { describe, it, expect } from 'vitest';
import {
  resolvePath,
  evaluateCriterion,
  validateTask,
  buildValidationContext,
} from '../../src/game/validation.ts';
import type { WinningCriterion } from '../../src/game/types.ts';

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
