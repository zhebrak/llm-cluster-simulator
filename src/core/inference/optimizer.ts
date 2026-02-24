/**
 * Inference Auto-Optimizer
 *
 * Pure function: takes an InferenceSimulationConfig + target ('throughput' | 'latency'),
 * returns an OptimizationResult with the best config found.
 *
 * Three phases:
 *   1. Fix — resolve OOM (max 10 iterations)
 *   2. Greedy — iteratively apply best single-mutation improvement (max 100 iterations)
 *   3. Explore — brute-force grid search over the config space
 */

import type { InferenceSimulationConfig } from './simulation.ts';
import type { InferenceSimulationResult } from '../../types/inference.ts';
import { runInferenceSimulationRaw } from './simulation.ts';
import { OOM_GENERATORS, SUCCESS_GENERATORS } from './recommendations.ts';
import { generateInferenceCandidates } from './exploration.ts';
import { getModel } from '../models/index.ts';
import { ALL_GPUS, H100_SXM } from '../hardware/gpu.ts';
import type { ModelSpec } from '../../types/model.ts';
import type { GPUSpec } from '../../types/hardware.ts';

// ── Result types ─────────────────────────────────────────────────────

export interface InferenceChangelogEntry {
  field: string;
  from: string;
  to: string;
}

export interface InferenceOptimizationResult {
  success: boolean;
  originalConfig: InferenceSimulationConfig;
  optimizedConfig: InferenceSimulationConfig;
  changelog: InferenceChangelogEntry[];
  beforeMetric: number;
  afterMetric: number;
  totalSimulations: number;
  phases: { fix: number; greedy: number; explore: number };
  target: 'throughput' | 'latency';
}

// ── Helpers ──────────────────────────────────────────────────────────

function simulateRaw(config: InferenceSimulationConfig): InferenceSimulationResult | null {
  try {
    return runInferenceSimulationRaw(config);
  } catch {
    return null;
  }
}

function getMetric(
  result: InferenceSimulationResult,
  target: 'throughput' | 'latency',
): number {
  if (!result.success) return target === 'throughput' ? 0 : Infinity;
  return target === 'throughput'
    ? result.throughput.tokensPerSecond
    : result.latency.tpot;
}

function isBetter(
  newMetric: number,
  oldMetric: number,
  target: 'throughput' | 'latency',
  threshold = 0.995,
): boolean {
  if (target === 'throughput') {
    return newMetric > oldMetric / threshold; // ≥0.5% improvement
  }
  return newMetric < oldMetric * threshold; // ≥0.5% improvement (lower is better)
}

function buildChangelog(
  original: InferenceSimulationConfig,
  optimized: InferenceSimulationConfig,
): InferenceChangelogEntry[] {
  const entries: InferenceChangelogEntry[] = [];

  const check = (field: string, from: unknown, to: unknown) => {
    if (JSON.stringify(from) !== JSON.stringify(to)) {
      entries.push({ field, from: String(from), to: String(to) });
    }
  };

  check('tensorParallel', original.tensorParallel ?? 1, optimized.tensorParallel ?? 1);
  check('expertParallel', original.expertParallel ?? 1, optimized.expertParallel ?? 1);
  check('batchSize', original.batchSize ?? 1, optimized.batchSize ?? 1);
  check('weightPrecision', original.weightPrecision ?? 'bf16', optimized.weightPrecision ?? 'bf16');
  check('kvCachePrecision', original.kvCachePrecision ?? 'bf16', optimized.kvCachePrecision ?? 'bf16');
  check('continuousBatching', original.continuousBatching ?? false, optimized.continuousBatching ?? false);
  check('flashAttention', original.flashAttention ?? true, optimized.flashAttention ?? true);

  return entries;
}

function resolveModel(config: InferenceSimulationConfig): ModelSpec | null {
  if (config.modelSpec) return config.modelSpec;
  if (config.modelId) return getModel(config.modelId, config.inputSeqLen ?? 512) ?? null;
  return null;
}

function resolveGPU(config: InferenceSimulationConfig): GPUSpec {
  if (config.gpu) return config.gpu;
  if (config.gpuId) return ALL_GPUS[config.gpuId] ?? H100_SXM;
  return H100_SXM;
}

// ── Main optimizer ───────────────────────────────────────────────────

export function optimizeInference(
  config: InferenceSimulationConfig,
  target: 'throughput' | 'latency',
): InferenceOptimizationResult {
  const originalConfig = { ...config };
  let currentConfig = { ...config };
  let totalSims = 0;
  const phaseSims = { fix: 0, greedy: 0, explore: 0 };

  let currentResult = simulateRaw(currentConfig);
  totalSims++;
  phaseSims.fix++;

  if (!currentResult) {
    return {
      success: false,
      originalConfig,
      optimizedConfig: currentConfig,
      changelog: [],
      beforeMetric: target === 'throughput' ? 0 : Infinity,
      afterMetric: target === 'throughput' ? 0 : Infinity,
      totalSimulations: totalSims,
      phases: phaseSims,
      target,
    };
  }

  // ── Phase 1: Fix OOM ───────────────────────────────────────────────

  for (let i = 0; i < 10 && !currentResult.success; i++) {
    let fixed = false;

    for (const gen of OOM_GENERATORS) {
      const candidate = gen(currentConfig, currentResult);
      if (!candidate?.configMutation) continue;

      const mutated = candidate.configMutation(currentConfig);
      const mutatedResult = simulateRaw(mutated);
      totalSims++;
      phaseSims.fix++;

      if (mutatedResult?.success) {
        currentConfig = mutated;
        currentResult = mutatedResult;
        fixed = true;
        break;
      }
    }

    if (!fixed) break;
  }

  // ── Phase 2: Greedy improvement ────────────────────────────────────

  let currentMetric = getMetric(currentResult, target);

  for (let iter = 0; iter < 100; iter++) {
    if (!currentResult.success) break;

    let bestMutation: InferenceSimulationConfig | null = null;
    let bestMetric = currentMetric;

    for (const gen of SUCCESS_GENERATORS) {
      const candidate = gen(currentConfig, currentResult);
      if (!candidate?.configMutation) continue;

      const mutated = candidate.configMutation(currentConfig);
      const mutatedResult = simulateRaw(mutated);
      totalSims++;
      phaseSims.greedy++;

      if (!mutatedResult?.success) continue;

      const mutatedMetric = getMetric(mutatedResult, target);
      if (isBetter(mutatedMetric, bestMetric, target)) {
        bestMetric = mutatedMetric;
        bestMutation = mutated;
      }
    }

    if (!bestMutation) break;

    currentConfig = bestMutation;
    currentResult = simulateRaw(currentConfig)!;
    totalSims++;
    phaseSims.greedy++;
    currentMetric = bestMetric;
  }

  // ── Phase 3: Explore grid ──────────────────────────────────────────

  const model = resolveModel(currentConfig);
  const gpu = resolveGPU(currentConfig);

  if (model) {
    const candidates = generateInferenceCandidates(currentConfig, model, gpu);

    for (const candidate of candidates) {
      const result = simulateRaw(candidate);
      totalSims++;
      phaseSims.explore++;

      if (!result?.success) continue;

      const metric = getMetric(result, target);
      if (isBetter(metric, currentMetric, target)) {
        currentMetric = metric;
        currentConfig = candidate;
        currentResult = result;
      }
    }
  }

  // ── Post-explore greedy pass ───────────────────────────────────────

  for (let iter = 0; iter < 20; iter++) {
    if (!currentResult?.success) break;

    let bestMutation: InferenceSimulationConfig | null = null;
    let bestMetric = currentMetric;

    for (const gen of SUCCESS_GENERATORS) {
      const candidate = gen(currentConfig, currentResult);
      if (!candidate?.configMutation) continue;

      const mutated = candidate.configMutation(currentConfig);
      const mutatedResult = simulateRaw(mutated);
      totalSims++;
      phaseSims.greedy++;

      if (!mutatedResult?.success) continue;

      const mutatedMetric = getMetric(mutatedResult, target);
      if (isBetter(mutatedMetric, bestMetric, target)) {
        bestMetric = mutatedMetric;
        bestMutation = mutated;
      }
    }

    if (!bestMutation) break;

    currentConfig = bestMutation;
    currentResult = simulateRaw(currentConfig)!;
    totalSims++;
    phaseSims.greedy++;
    currentMetric = bestMetric;
  }

  // ── Build result ───────────────────────────────────────────────────

  const originalResult = simulateRaw(originalConfig);
  totalSims++;
  const beforeMetric = originalResult ? getMetric(originalResult, target) : (target === 'throughput' ? 0 : Infinity);

  return {
    success: currentResult !== null && currentResult.success,
    originalConfig,
    optimizedConfig: currentConfig,
    changelog: buildChangelog(originalConfig, currentConfig),
    beforeMetric,
    afterMetric: currentMetric,
    totalSimulations: totalSims,
    phases: phaseSims,
    target,
  };
}
