/**
 * Sequence Length Sweep
 *
 * Sweeps input sequence length (powers of 2) across precisions,
 * recording TTFT, TPOT, and memory utilization at each point.
 * Detects OOM cutoffs where KV cache exhausts GPU memory.
 */

import type { ModelSpec } from '../../types/model.ts';
import type { GPUSpec } from '../../types/hardware.ts';
import type { InferencePrecision } from '../../types/inference.ts';
import type { InferenceSimulationConfig } from './simulation.ts';
import { runInferenceSimulationRaw } from './simulation.ts';
import { gpuCapacityBytes } from '../strategies/base.ts';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SeqLenSweepPoint {
  seqLen: number;
  ttft: number;
  tpot: number;
  memoryUtil: number;
  tokensPerSecond: number;
}

export interface SeqLenSweepResult {
  /** precision group ('bf16'|'fp8'|'int4'|'q4_k_m'|'q8_0') -> points */
  groups: Record<string, SeqLenSweepPoint[]>;
  /** precision group -> first OOM seqLen (0 = never OOM'd) */
  oomCutoffs: Record<string, number>;
}

// ---------------------------------------------------------------------------
// Precision helpers
// ---------------------------------------------------------------------------

/** Weight precision for each sweep group. KV cache precision comes from user config. */
const WEIGHT_PRECISIONS: Record<string, InferencePrecision> = {
  bf16:   'bf16',
  fp8:    'fp8',
  int4:   'int4',
  q4_k_m: 'q4_k_m',
  q8_0:   'q8_0',
};

function availablePrecisions(gpu: GPUSpec): string[] {
  const precs = ['bf16'];
  if (gpu.fp8TFLOPS && gpu.fp8TFLOPS > 0) precs.push('fp8');
  precs.push('int4');
  precs.push('q4_k_m', 'q8_0');
  return precs;
}

// ---------------------------------------------------------------------------
// Sweep
// ---------------------------------------------------------------------------

const MIN_SEQ_LEN = 128;
const MAX_SEQ_LEN = 1_048_576;

export function runSeqLenSweep(
  baseConfig: InferenceSimulationConfig,
  _model: ModelSpec,
  gpu: GPUSpec,
): SeqLenSweepResult {
  const precisions = availablePrecisions(gpu);
  const groups: Record<string, SeqLenSweepPoint[]> = {};
  const oomCutoffs: Record<string, number> = {};

  for (const prec of precisions) {
    groups[prec] = [];
    oomCutoffs[prec] = 0;
  }

  // Track which precisions have OOM'd
  const oomedAt = new Map<string, number>();

  for (let seqLen = MIN_SEQ_LEN; seqLen <= MAX_SEQ_LEN; seqLen *= 2) {
    // Stop when all precisions have OOM'd and we've swept one past the last
    if (oomedAt.size >= precisions.length) {
      const maxOom = Math.max(...oomedAt.values());
      if (seqLen > maxOom) break;
    }

    for (const prec of precisions) {
      // Skip precisions that already OOM'd
      if (oomedAt.has(prec)) continue;

      const config: InferenceSimulationConfig = {
        ...baseConfig,
        inputSeqLen: seqLen,
        weightPrecision: WEIGHT_PRECISIONS[prec],
      };

      const result = runInferenceSimulationRaw(config);

      if (!result.success) {
        // OOM — record cutoff
        oomCutoffs[prec] = seqLen;
        oomedAt.set(prec, seqLen);
        continue;
      }

      const gpuMemBytes = gpuCapacityBytes(gpu.memoryGB);
      const totalMem = result.memory.total;
      const memUtil = totalMem / gpuMemBytes;

      // If memory > 100%, treat as OOM even if sim "succeeded"
      if (memUtil > 1.0) {
        oomCutoffs[prec] = seqLen;
        oomedAt.set(prec, seqLen);
        continue;
      }

      const ttft = result.latency.ttft;
      const tpot = result.latency.tpot;

      if (!isFinite(ttft) || !isFinite(tpot) || ttft <= 0 || tpot <= 0) continue;

      groups[prec].push({ seqLen, ttft, tpot, memoryUtil: memUtil, tokensPerSecond: result.throughput.tokensPerSecond });
    }
  }

  // For each precision that OOM'd, binary-search between the last valid
  // power-of-2 point and the OOM cutoff to find the max non-OOM seqLen.
  const gpuMemBytes = gpuCapacityBytes(gpu.memoryGB);
  for (const prec of precisions) {
    const cutoff = oomCutoffs[prec];
    if (cutoff <= 0) continue;

    const pts = groups[prec];
    const lastValid = pts.length > 0 ? pts[pts.length - 1].seqLen : MIN_SEQ_LEN / 2;
    if (cutoff - lastValid <= 1) continue;

    let lo = lastValid;
    let hi = cutoff;
    let bestPoint: SeqLenSweepPoint | null = null;

    // ~10 iterations gives precision within ~0.1% of the boundary
    for (let i = 0; i < 10; i++) {
      const mid = Math.round((lo + hi) / 2);
      if (mid <= lo || mid >= hi) break;

      const config: InferenceSimulationConfig = {
        ...baseConfig,
        inputSeqLen: mid,
        weightPrecision: WEIGHT_PRECISIONS[prec],
      };
      const result = runInferenceSimulationRaw(config);
      const fits = result.success && result.memory.total / gpuMemBytes <= 1.0;

      if (fits) {
        const ttft = result.latency.ttft;
        const tpot = result.latency.tpot;
        if (isFinite(ttft) && isFinite(tpot) && ttft > 0 && tpot > 0) {
          bestPoint = {
            seqLen: mid,
            ttft,
            tpot,
            memoryUtil: result.memory.total / gpuMemBytes,
            tokensPerSecond: result.throughput.tokensPerSecond,
          };
        }
        lo = mid;
      } else {
        hi = mid;
      }
    }

    // Move OOM line to the binary-searched boundary (tighter than power-of-2)
    oomCutoffs[prec] = hi;

    if (bestPoint && bestPoint.seqLen > lastValid) {
      pts.push(bestPoint);
    }
  }

  return { groups, oomCutoffs };
}
