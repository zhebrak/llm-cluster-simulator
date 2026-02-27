/**
 * Game mode validation — evaluates winning criteria against simulation results
 */

import type { WinningCriterion, ValidationResult, GameMode } from './types.ts';
import type { SimulationState } from '../stores/simulation.ts';
import { useConfigStore } from '../stores/config.ts';
import { getGPUHourlyRate, calculateCostPerMillionTokens } from '../core/cost/cloud.ts';

/**
 * Resolve a dot-path on an object. Returns undefined if any segment is missing.
 */
export function resolvePath(obj: unknown, path: string): unknown {
  const segments = path.split('.');
  let current: unknown = obj;
  for (const seg of segments) {
    if (current == null || typeof current !== 'object') return undefined;
    current = (current as Record<string, unknown>)[seg];
  }
  return current;
}

/**
 * Evaluate a single criterion against a context object.
 */
export function evaluateCriterion(ctx: object, criterion: WinningCriterion): boolean {
  const actual = resolvePath(ctx, criterion.field);
  if (actual === undefined || actual === null) return false;

  const { operator, value } = criterion;
  switch (operator) {
    case '>':  return (actual as number) > (value as number);
    case '>=': return (actual as number) >= (value as number);
    case '<':  return (actual as number) < (value as number);
    case '<=': return (actual as number) <= (value as number);
    case '==': return actual === value;
    case '!=': return actual !== value;
    default:   return false;
  }
}

/**
 * Validate all criteria for a task. Returns { passed, results[] }.
 * If ctx is null (no valid simulation), all criteria are marked unmet.
 */
export function validateTask(
  ctx: object | null,
  criteria: WinningCriterion[],
): ValidationResult {
  if (!ctx) {
    return {
      passed: false,
      results: criteria.map(c => ({ criterion: c, met: false })),
    };
  }

  const results = criteria.map(criterion => ({
    criterion,
    met: evaluateCriterion(ctx, criterion),
  }));

  return {
    passed: results.every(r => r.met),
    results,
  };
}

/**
 * Build a validation context from simulation state for the given mode.
 * Returns null if no valid result exists for that mode.
 *
 * Training context: flat-ish merge of SimulationResult + SimulationMetrics fields.
 * Inference context: InferenceSimulationResult fields.
 */
export function buildValidationContext(
  simState: SimulationState,
  mode: GameMode | null,
): object | null {
  if (!mode) return null;

  if (mode === 'training') {
    const { result, metrics } = simState.training;
    if (!result || !metrics) return null;

    return { success: result.success, ...metrics };
  }

  if (mode === 'inference') {
    const { result } = simState.inference;
    if (!result) return null;

    // Pricing reads from config store at simulation-complete time — correct because
    // the user just clicked Run with this config. If async pricing lookups or deferred
    // config updates are ever added, this coupling would need revisiting.
    const { gpuId, numGPUs, pricePerGPUHour } = useConfigStore.getState();
    const rate = pricePerGPUHour ?? getGPUHourlyRate(gpuId).rate;

    // Explicitly pick metric fields — mirrors training pattern.
    // Excludes engine artifacts (errors, warnings, recommendations, events).
    return {
      success: result.success,
      memory: result.memory,
      kvCacheState: result.kvCacheState,
      latency: result.latency,
      throughput: result.throughput,
      utilization: result.utilization,
      speculative: result.speculative,
      continuousBatching: result.continuousBatching,
      maxConcurrentRequests: result.maxConcurrentRequests,
      // Derived fields not in the engine output
      memoryUtilization: result.utilization?.memoryCapacityUtilization,
      costPerMillionTokens: calculateCostPerMillionTokens(
        rate, numGPUs, result.throughput?.tokensPerSecond ?? 0,
      ),
    };
  }

  return null;
}
