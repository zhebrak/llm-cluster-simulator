/**
 * Inference configuration exploration grid
 *
 * Generates candidate InferenceSimulationConfigs from a constrained Cartesian product.
 * Used by the optimizer's Phase 3 (explore) to find globally better configs.
 */

import type { InferenceSimulationConfig } from './simulation.ts';
import type { ModelSpec } from '../../types/model.ts';
import type { GPUSpec } from '../../types/hardware.ts';

export const MAX_EXPLORE_SIMS = 20000;

/** Mulberry32 PRNG — deterministic, fast, 32-bit state. */
function createSeededRandom(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function generateInferenceCandidates(
  baseConfig: InferenceSimulationConfig,
  model: ModelSpec,
  gpu: GPUSpec,
  seed?: number,
): InferenceSimulationConfig[] {
  const numGPUs = baseConfig.numGPUs ?? 1;
  const isMoE = model.isMoE ?? false;
  const numExperts = model.numExperts ?? 0;

  const hasFP8 = gpu.fp8TFLOPS > 0;
  const hasINT4 = true; // INT4 is always software-emulable

  // Build dimension arrays — let OOM be the filter, not artificial caps
  const tpPowers = [1, 2, 4, 8, 16, 32, 64];
  const tpDegrees = tpPowers.filter(tp => tp <= numGPUs);

  const epPowers = [1, 2, 4, 8, 16, 32, 64, 128, 256];
  const epDegrees = isMoE && numExperts >= 2
    ? epPowers.filter(ep => ep === 1 || (ep <= numExperts && numExperts % ep === 0))
    : [1];

  const weightPrecisions: InferenceSimulationConfig['weightPrecision'][] = ['bf16'];
  if (hasFP8) weightPrecisions.push('fp8');
  if (hasINT4) weightPrecisions.push('int4');
  // Two most popular GGUF quants — fill gaps between int4/fp8/bf16
  weightPrecisions.push('q4_k_m', 'q8_0');
  // Always include user's current precision so it appears on the chart
  if (baseConfig.weightPrecision && !weightPrecisions.includes(baseConfig.weightPrecision)) {
    weightPrecisions.push(baseConfig.weightPrecision);
  }

  // Batch sizes up to 16384 — OOM filters naturally
  const absoluteBatchSizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384];
  const perReplicaTargets = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];
  // Always include user's current batch size so it appears on the chart
  if (baseConfig.batchSize && !absoluteBatchSizes.includes(baseConfig.batchSize)) {
    absoluteBatchSizes.push(baseConfig.batchSize);
  }
  const cbValues = [true, false];

  const candidates: InferenceSimulationConfig[] = [];

  for (const tp of tpDegrees) {
    for (const ep of epDegrees) {
      // tp × ep must fit in available GPUs
      if (tp * ep > numGPUs) continue;

      // numReplicas depends on TP and EP — compute batch sizes per combo
      const numReplicas = Math.max(1, Math.floor(numGPUs / (tp * Math.max(1, ep))));
      const replicaBatchSizes = perReplicaTargets.map(t => t * numReplicas);
      const batchSet = new Set([...absoluteBatchSizes, ...replicaBatchSizes]);
      const batchSizes = Array.from(batchSet).sort((a, b) => a - b);

      for (const weightPrec of weightPrecisions) {
        for (const batch of batchSizes) {
          for (const cb of cbValues) {
            candidates.push({
              ...baseConfig,
              tensorParallel: tp,
              expertParallel: isMoE ? ep : undefined,
              weightPrecision: weightPrec,
              batchSize: batch,
              continuousBatching: cb,
              flashAttention: true,
            });
          }
        }
      }
    }
  }

  // Random sample if exceeding budget
  if (candidates.length > MAX_EXPLORE_SIMS) {
    const random = createSeededRandom(seed ?? 42);
    for (let i = candidates.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [candidates[i], candidates[j]] = [candidates[j], candidates[i]];
    }
    return candidates.slice(0, MAX_EXPLORE_SIMS);
  }

  return candidates;
}
