/**
 * Pareto frontier computation for inference configuration exploration.
 *
 * Sweeps the candidate grid from exploration.ts, runs each through
 * runInferenceSimulationRaw(), and extracts the non-dominated frontier
 * on cost-vs-latency axes.
 */

import type { InferenceSimulationConfig } from './simulation.ts';
import type { InferencePrecision } from '../../types/inference.ts';
import type { ModelSpec } from '../../types/model.ts';
import type { GPUSpec } from '../../types/hardware.ts';
import { generateInferenceCandidates } from './exploration.ts';
import { runInferenceSimulationRaw } from './simulation.ts';
import { gpuCapacityBytes } from '../strategies/base.ts';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ParetoPoint {
  config: {
    tp: number;
    ep: number | undefined;
    batchSize: number;
    weightPrecision: InferencePrecision;
    kvCachePrecision: InferencePrecision;
    continuousBatching: boolean;
  };
  costPerMToken: number;  // $/M tokens
  ttft: number;           // ms
  tpot: number;           // ms
  throughput: number;      // tok/s
  memoryUtil: number;      // 0-1
}

export interface ParetoSweepResult {
  points: ParetoPoint[];         // All successful configs
  frontier: ParetoPoint[];       // Non-dominated (cost vs TTFT)
  frontierTpot: ParetoPoint[];   // Non-dominated (cost vs TPOT)
  totalCandidates: number;
  totalSuccessful: number;
  sweepKey: string;              // modelId|gpuId|numGPUs for invalidation
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Cost per million tokens given throughput, GPU count, and hourly rate.
 * Extracted from ResultsDashboard.tsx:420.
 */
export function computeCostPerMToken(
  tokensPerSecond: number,
  numGPUs: number,
  gpuHourlyRate: number,
): number {
  if (tokensPerSecond <= 0 || gpuHourlyRate <= 0) return Infinity;
  return (gpuHourlyRate * numGPUs / 3600) / tokensPerSecond * 1e6;
}

/**
 * Build a cache key that changes when the hardware/model envelope changes
 * (but NOT when batch/TP/precision change — those are swept).
 * kvCachePrecision is NOT swept (set in user config), so it must be in the key.
 */
export function buildSweepKey(
  modelId: string,
  gpuId: string,
  numGPUs: number,
  inputSeqLen?: number,
  outputSeqLen?: number,
  kvCachePrecision?: string,
): string {
  return `${modelId}|${gpuId}|${numGPUs}|${inputSeqLen ?? 0}|${outputSeqLen ?? 0}|${kvCachePrecision ?? 'bf16'}`;
}

/**
 * Extract the Pareto-optimal frontier from a set of points.
 *
 * A point is on the frontier if no other point has both lower cost AND
 * lower latency. The frontier is sorted by cost ascending (and therefore
 * latency descending).
 */
export function extractParetoFrontier(
  points: ParetoPoint[],
  latencyKey: 'ttft' | 'tpot',
): ParetoPoint[] {
  if (points.length === 0) return [];

  // Sort by cost ascending
  const sorted = [...points].sort((a, b) => a.costPerMToken - b.costPerMToken);

  const frontier: ParetoPoint[] = [];
  let minLatency = Infinity;

  // Walk from lowest cost to highest. A point is on the frontier if its
  // latency is lower than any point we've already included (which all
  // have lower cost).
  for (const p of sorted) {
    if (p[latencyKey] < minLatency) {
      frontier.push(p);
      minLatency = p[latencyKey];
    }
  }

  return frontier;
}

// ---------------------------------------------------------------------------
// Sweep
// ---------------------------------------------------------------------------

const CHUNK_SIZE = 50;

/**
 * Run the full Pareto sweep. Async — yields between chunks to keep the
 * UI responsive. Returns null if aborted.
 */
export async function runParetoSweep(
  baseConfig: InferenceSimulationConfig,
  model: ModelSpec,
  gpu: GPUSpec,
  gpuHourlyRate: number,
  numGPUs: number,
  onProgress?: (fraction: number) => void,
  abortSignal?: AbortSignal,
): Promise<ParetoSweepResult | null> {
  const candidates = generateInferenceCandidates(baseConfig, model, gpu);
  const totalCandidates = candidates.length;
  const points: ParetoPoint[] = [];

  for (let i = 0; i < totalCandidates; i += CHUNK_SIZE) {
    // Check abort between chunks
    if (abortSignal?.aborted) return null;

    const end = Math.min(i + CHUNK_SIZE, totalCandidates);
    for (let j = i; j < end; j++) {
      const candidate = candidates[j];
      try {
        const result = runInferenceSimulationRaw(candidate);
        if (!result.success) continue;
        if (result.throughput.tokensPerSecond <= 0) continue;

        const cost = computeCostPerMToken(
          result.throughput.tokensPerSecond,
          numGPUs,
          gpuHourlyRate,
        );

        // Skip non-finite values
        if (!isFinite(cost) || !isFinite(result.latency.ttft) || !isFinite(result.latency.tpot)) continue;
        if (cost <= 0 || result.latency.ttft <= 0 || result.latency.tpot <= 0) continue;

        const gpuMemBytes = gpuCapacityBytes(gpu.memoryGB);
        points.push({
          config: {
            tp: candidate.tensorParallel ?? 1,
            ep: candidate.expertParallel,
            batchSize: candidate.batchSize ?? 1,
            weightPrecision: (candidate.weightPrecision ?? 'bf16') as InferencePrecision,
            kvCachePrecision: (candidate.kvCachePrecision ?? 'bf16') as InferencePrecision,
            continuousBatching: candidate.continuousBatching ?? false,
          },
          costPerMToken: cost,
          ttft: result.latency.ttft,
          tpot: result.latency.tpot,
          throughput: result.throughput.tokensPerSecond,
          memoryUtil: result.memory.total / gpuMemBytes,
        });
      } catch {
        // Skip configs that throw
      }
    }

    // Report progress + yield to event loop
    onProgress?.((end / totalCandidates));
    await new Promise(r => setTimeout(r, 0));
  }

  // Check abort one more time before building frontiers
  if (abortSignal?.aborted) return null;

  const sweepKey = buildSweepKey(
    baseConfig.modelId ?? '',
    baseConfig.gpuId ?? '',
    numGPUs,
    baseConfig.inputSeqLen,
    baseConfig.outputSeqLen,
    baseConfig.kvCachePrecision,
  );

  return {
    points,
    frontier: extractParetoFrontier(points, 'ttft'),
    frontierTpot: extractParetoFrontier(points, 'tpot'),
    totalCandidates,
    totalSuccessful: points.length,
    sweepKey,
  };
}
