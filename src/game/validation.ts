/**
 * Game mode validation — evaluates winning criteria against simulation results
 */

import type { WinningCriterion, ValidationResult, GameMode, ExpectedChange } from './types.ts';
import type { SimulationState } from '../stores/simulation.ts';
import { useConfigStore } from '../stores/config.ts';
import { getGPUHourlyRate, calculateCostPerMillionTokens } from '../core/cost/cloud.ts';

/**
 * Flat snapshot of config state for expected-change validation.
 * Captured after applyTaskSetup(), compared against current state on each sim run.
 */
export interface TaskConfigSnapshot {
  // Hardware
  modelId: string;
  gpuId: string;
  numGPUs: number;
  // Training
  precision: string;
  activationCheckpointing: boolean;
  checkpointingGranularity: string;
  flashAttention: boolean;
  globalBatchSize: number;
  microBatchSize: number;
  sequenceLength: number;
  sequenceParallel: boolean;
  strategyType: string;
  tpDegree: number;
  ppDegree: number;
  epDegree: number;
  cpDegree: number;
  pipelineSchedule: string;
  interleavedStages: number;
  finetuningMethod: string;
  loraRank: number;
  loraTargetModules: string;
  // Inference
  weightPrecision: string;
  kvCachePrecision: string;
  batchSize: number;
  inputSeqLen: number;
  outputSeqLen: number;
  tensorParallel: number;
  expertParallel: number;
  pagedAttention: boolean;
  continuousBatching: boolean;
  speculativeDecoding: boolean;
  pricePerGPUHour: number | null;
}

/**
 * Capture a flat config snapshot from the config store.
 */
export function captureTaskConfig(): TaskConfigSnapshot {
  const state = useConfigStore.getState();
  return {
    // Hardware
    modelId: state.modelId,
    gpuId: state.gpuId,
    numGPUs: state.numGPUs,
    // Training
    precision: state.precision,
    activationCheckpointing: state.training.activationCheckpointing,
    checkpointingGranularity: state.training.checkpointingGranularity,
    flashAttention: state.mode === 'training' ? state.training.flashAttention : state.inference.flashAttention,
    globalBatchSize: state.training.globalBatchSize,
    microBatchSize: state.training.microBatchSize,
    sequenceLength: state.sequenceLength,
    sequenceParallel: state.training.sequenceParallel,
    strategyType: state.training.strategyType,
    tpDegree: state.training.tpDegree,
    ppDegree: state.training.ppDegree,
    epDegree: state.training.epDegree,
    cpDegree: state.training.cpDegree,
    pipelineSchedule: state.training.pipelineSchedule,
    interleavedStages: state.training.interleavedStages,
    finetuningMethod: state.training.finetuningMethod,
    loraRank: state.training.loraRank,
    loraTargetModules: state.training.loraTargetModules,
    // Inference
    weightPrecision: state.inference.weightPrecision,
    kvCachePrecision: state.inference.kvCachePrecision,
    batchSize: state.inference.batchSize,
    inputSeqLen: state.inference.inputSeqLen,
    outputSeqLen: state.inference.outputSeqLen,
    tensorParallel: state.inference.tensorParallel,
    expertParallel: state.inference.expertParallel,
    pagedAttention: state.inference.pagedAttention,
    continuousBatching: state.inference.continuousBatching,
    speculativeDecoding: state.inference.speculativeDecoding,
    pricePerGPUHour: state.pricePerGPUHour,
  };
}

/**
 * Evaluate a single expected-change check.
 */
function evaluateExpectedChange(
  snapshot: TaskConfigSnapshot,
  current: TaskConfigSnapshot,
  change: ExpectedChange,
): boolean {
  const field = change.field as keyof TaskConfigSnapshot;
  const snapshotVal = snapshot[field];
  const currentVal = current[field];

  switch (change.check) {
    case 'changed':   return currentVal !== snapshotVal;
    case 'unchanged': return currentVal === snapshotVal;
    case 'increased': return (currentVal as number) > (snapshotVal as number);
    case 'decreased': return (currentVal as number) < (snapshotVal as number);
    case 'enabled':   return currentVal === true;
    case 'disabled':  return currentVal === false;
    default:          return false;
  }
}

/**
 * Validate all expected changes. Returns { valid, failedChecks }.
 * Empty or undefined expectedChanges → always valid.
 */
export function validateExpectedChanges(
  snapshot: TaskConfigSnapshot,
  current: TaskConfigSnapshot,
  changes: ExpectedChange[] | undefined,
): { valid: boolean; failedChecks: ExpectedChange[] } {
  if (!changes || changes.length === 0) {
    return { valid: true, failedChecks: [] };
  }

  const failedChecks = changes.filter(
    change => !evaluateExpectedChange(snapshot, current, change),
  );

  return { valid: failedChecks.length === 0, failedChecks };
}

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
      // Hardware context
      numGPUs,
      // Derived fields not in the engine output
      memoryUtilization: result.utilization?.memoryCapacityUtilization,
      costPerMillionTokens: calculateCostPerMillionTokens(
        rate, numGPUs, result.throughput?.tokensPerSecond ?? 0,
      ),
    };
  }

  return null;
}
