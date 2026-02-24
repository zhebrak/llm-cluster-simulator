/**
 * Inference recommendation generators
 *
 * 26 generators in priority order (8 OOM + 18 success); returns first 3 non-null results.
 * OOM generators pre-validate candidates internally (re-simulate to confirm OOM resolved).
 * Fallback message is appended only when no actionable OOM recommendation resolves the issue.
 * Validated generators provide a configMutation for "what-if" validation:
 * the engine re-simulates with the mutated config and only keeps the
 * recommendation if the result meaningfully improves (throughput, memory, or latency).
 *
 * Heuristic-only generators (no configMutation) are used only when the simulation
 * can't model the effect (paged attention fragmentation, speculative decoding draft
 * model selection) or for pure warnings (FP32, context length).
 *
 * Each recommendation suggests a direction (increase/decrease/try option)
 * rather than prescribing specific values. Numbers are already on the results
 * page — recommendations just say what to change.
 */

import type { InferenceSimulationResult } from '../../types/inference.ts';
import type { InferenceSimulationConfig } from './simulation.ts';
import { runInferenceSimulationRaw } from './simulation.ts';
import { getPrecisionBytes } from './kv-cache.ts';
import { moeParamSplit } from './latency.ts';
import { getModel } from '../models/index.ts';
import { H100_SXM, supportsFlashAttention } from '../hardware/gpu.ts';
import { gpuCapacityBytes } from '../strategies/base.ts';
import type { GPUSpec } from '../../types/hardware.ts';
import type { ModelSpec } from '../../types/model.ts';

// ── types ────────────────────────────────────────────────────────────

export interface InferenceRecommendationCandidate {
  message: string;
  configMutation?: (config: InferenceSimulationConfig) => InferenceSimulationConfig;
  /** Which metric the mutation targets. Validation checks this dimension for ≥1% improvement. */
  validationMode?: 'throughput' | 'memory' | 'latency';
}

type InferenceCandidateGenerator = (
  config: InferenceSimulationConfig,
  result: InferenceSimulationResult,
) => InferenceRecommendationCandidate | null;

type InferenceGenerator = (
  config: InferenceSimulationConfig,
  result: InferenceSimulationResult,
) => string | null;

function wrapGenerator(gen: InferenceGenerator): InferenceCandidateGenerator {
  return (config, result) => {
    const msg = gen(config, result);
    return msg !== null ? { message: msg } : null;
  };
}

// ── helpers ──────────────────────────────────────────────────────────

function resolveGPU(config: InferenceSimulationConfig): GPUSpec {
  return config.gpu ?? H100_SXM;
}

function resolveModel(config: InferenceSimulationConfig): ModelSpec {
  if (config.modelSpec) return config.modelSpec;
  if (config.modelId) {
    const model = getModel(config.modelId, config.inputSeqLen ?? 512);
    if (model) return model;
  }
  throw new Error('Cannot resolve model from config');
}

function computeNumReplicas(config: InferenceSimulationConfig): number {
  const numGPUs = config.numGPUs ?? 1;
  const tp = config.tensorParallel ?? 1;
  const ep = config.expertParallel ?? 1;
  return Math.max(1, Math.floor(numGPUs / (tp * Math.max(1, ep))));
}

// ── validation ───────────────────────────────────────────────────────

export function validateInferenceCandidate(
  candidate: InferenceRecommendationCandidate,
  config: InferenceSimulationConfig,
  currentResult: InferenceSimulationResult,
): boolean {
  if (!candidate.configMutation) return true; // heuristic-only always passes

  try {
    const mutatedConfig = candidate.configMutation(config);
    const mutatedResult = runInferenceSimulationRaw(mutatedConfig);

    // Reject if mutated config OOMs
    if (!mutatedResult.success) return false;

    // If current config is OOM, accept any mutation that fits in memory
    if (!currentResult.success) return true;

    // Check the dimension this generator targets
    const mode = candidate.validationMode ?? 'throughput';
    switch (mode) {
      case 'throughput':
        return mutatedResult.throughput.tokensPerSecond > currentResult.throughput.tokensPerSecond * 1.01;
      case 'memory':
        return mutatedResult.memory.total < currentResult.memory.total * 0.99;
      case 'latency':
        return currentResult.latency.tpot > 0
          && mutatedResult.latency.tpot < currentResult.latency.tpot * 0.99;
    }
  } catch (e) {
    if (import.meta.env?.DEV) console.warn('Recommendation validation failed:', e);
    return false;
  }
}

// ══════════════════════════════════════════════════════════════════════
// OOM GENERATORS (positions 1–5) — fire only when !result.success
// ══════════════════════════════════════════════════════════════════════

// ── 1. OOM: Paged Attention ───────────────────────────────────────────

const oomPagedAttention: InferenceCandidateGenerator = (config, result) => {
  if (result.success) return null;
  if (config.pagedAttention !== false) return null;
  if (result.memory.kvCache <= result.memory.total * 0.10) return null;

  return {
    message: 'Try enabling paged attention — it eliminates KV cache memory fragmentation from contiguous pre-allocation, often recovering enough memory to fit.',
    configMutation: (cfg) => ({ ...cfg, pagedAttention: true }),
  };
};

// ── 2. OOM: Increase TP ──────────────────────────────────────────────

// OOM generators pre-validate candidates internally by re-simulating each one.
// This costs ~10-20 extra simulations per OOM pass (3-4 generators × 3-5 candidates),
// but inference simulation is cheap (~1ms), so this is fine at browser speeds.
// Don't extend candidate lists to 100+ values without reconsidering this trade-off.

const oomTPIncrease: InferenceCandidateGenerator = (config, result) => {
  if (result.success) return null;

  const gpu = resolveGPU(config);
  const gpuMemoryBytes = gpuCapacityBytes(gpu.memoryGB);
  const tp = config.tensorParallel ?? 1;
  const numGPUs = config.numGPUs ?? 1;

  if (tp >= numGPUs) return null;
  if (result.memory.weights <= gpuMemoryBytes * 0.5) return null;

  for (const candidateTP of [2, 4, 8]) {
    if (candidateTP <= tp || candidateTP > numGPUs) continue;

    try {
      const mutated = runInferenceSimulationRaw({ ...config, tensorParallel: candidateTP });
      if (mutated.success) {
        return {
          message: 'Try increasing tensor parallelism — this shards model weights and KV cache across GPUs to reduce per-GPU memory.',
          configMutation: (cfg) => ({ ...cfg, tensorParallel: candidateTP }),
        };
      }
    } catch { continue; }
  }

  return null;
};

// ── 2. OOM: Increase EP ──────────────────────────────────────────────

const oomEPIncrease: InferenceCandidateGenerator = (config, result) => {
  if (result.success) return null;

  const model = resolveModel(config);
  if (!model.isMoE || !model.numExperts) return null;

  const { routedExpertParams } = moeParamSplit(model);
  if (routedExpertParams <= model.totalParams * 0.30) return null;

  const tp = config.tensorParallel ?? 1;
  const currentEP = config.expertParallel ?? 1;
  const numGPUs = config.numGPUs ?? 1;

  // Dynamic candidate list: powers of 2 up to numExperts
  const epCandidates: number[] = [];
  for (let ep = 2; ep <= model.numExperts; ep *= 2) epCandidates.push(ep);

  for (const candidateEP of epCandidates) {
    if (candidateEP <= currentEP) continue;
    if (candidateEP * tp > numGPUs) continue;
    if (model.numExperts % candidateEP !== 0) continue;

    try {
      const mutated = runInferenceSimulationRaw({ ...config, expertParallel: candidateEP });
      if (mutated.success) {
        return {
          message: 'Try enabling expert parallelism — this distributes MoE expert weights across GPUs, reducing per-GPU memory for expert-heavy models.',
          configMutation: (cfg) => ({ ...cfg, expertParallel: candidateEP }),
        };
      }
    } catch { continue; }
  }

  return null;
};

// ── 3. OOM: Combined TP + EP ─────────────────────────────────────────

const oomTPAndEPCombined: InferenceCandidateGenerator = (config, result) => {
  if (result.success) return null;

  const model = resolveModel(config);
  if (!model.isMoE || !model.numExperts) return null;

  const tp = config.tensorParallel ?? 1;
  const currentEP = config.expertParallel ?? 1;
  const numGPUs = config.numGPUs ?? 1;

  // Only fire when there are enough GPUs for a meaningful combination
  if (numGPUs < 4) return null;
  // Skip if TP is already high and EP is already set
  if (tp >= 8 && currentEP >= 2) return null;

  // Try max TP within power-of-2 up to 8, then increasing EP
  const maxTP = Math.min(8, numGPUs);
  if (maxTP <= tp && currentEP > 1) return null;

  const epCandidates: number[] = [];
  for (let ep = 2; ep <= model.numExperts; ep *= 2) epCandidates.push(ep);

  for (const candidateEP of epCandidates) {
    if (candidateEP * maxTP > numGPUs) continue;
    if (model.numExperts % candidateEP !== 0) continue;

    try {
      const mutated = runInferenceSimulationRaw({
        ...config,
        tensorParallel: maxTP,
        expertParallel: candidateEP,
      });
      if (mutated.success) {
        return {
          message: 'Try combining tensor parallelism with expert parallelism — sharding both shared weights and expert weights across GPUs can resolve OOM when neither alone suffices.',
          configMutation: (cfg) => ({
            ...cfg,
            tensorParallel: maxTP,
            expertParallel: candidateEP,
          }),
        };
      }
    } catch { continue; }
  }

  return null;
};

// ── 4. OOM: Quantization ─────────────────────────────────────────────

const oomQuantization: InferenceCandidateGenerator = (config, result) => {
  if (result.success) return null;

  const currentWeightBytes = getPrecisionBytes(config.weightPrecision ?? 'bf16');
  if (currentWeightBytes <= 1) return null;

  // Text-only: inform the user that reducing precision would help, but don't auto-apply
  // (respects the user's intentional precision choice)
  return {
    message: 'Model doesn\'t fit — consider reducing weight precision to FP8 or INT4.',
  };
};

// ── 5. OOM: Reduce batch ─────────────────────────────────────────────

const oomBatchReduction: InferenceCandidateGenerator = (config, result) => {
  if (result.success) return null;

  const batchSize = config.batchSize ?? 1;
  if (batchSize <= 1) return null;
  if (result.memory.kvCache <= result.memory.total * 0.2) return null;

  // Try progressively halving batch size
  for (let candidate = Math.ceil(batchSize / 2); candidate >= 1; candidate = Math.ceil(candidate / 2)) {
    try {
      const mutated = runInferenceSimulationRaw({ ...config, batchSize: candidate });
      if (mutated.success) {
        return {
          message: 'Try reducing batch size — fewer concurrent sequences means less KV cache memory.',
          configMutation: (cfg) => ({ ...cfg, batchSize: candidate }),
        };
      }
    } catch { /* continue */ }
    if (candidate === 1) break;
  }

  return null;
};

// ── 6. OOM: Reduce context ───────────────────────────────────────────

const oomContextReduction: InferenceGenerator = (config, result) => {
  if (result.success) return null;

  const inputSeqLen = config.inputSeqLen ?? 512;
  const outputSeqLen = config.outputSeqLen ?? 256;
  const totalSeqLen = inputSeqLen + outputSeqLen;

  if (totalSeqLen <= 1024) return null;
  if (result.memory.kvCache <= result.memory.total * 0.2) return null;

  return 'Try reducing input or output sequence length — shorter context reduces KV cache memory.';
};

// ── OOM: Fallback (used as last-resort only, not in generator list) ──

const oomFallback = (config: InferenceSimulationConfig, _result: InferenceSimulationResult): string => {
  const numGPUs = config.numGPUs ?? 1;
  if (numGPUs > 1) {
    return 'Model does not fit in GPU memory — try increasing tensor parallelism or expert parallelism to shard the model across GPUs.';
  }
  return 'Model weights exceed GPU memory — consider using a GPU with more memory, adding more GPUs, or a smaller model.';
};

// ══════════════════════════════════════════════════════════════════════
// SUCCESS GENERATORS (positions 6–17) — fire only when result.success
// ══════════════════════════════════════════════════════════════════════

// ── 6. FP32 warning ──────────────────────────────────────────────────

const fp32Warning: InferenceGenerator = (config, result) => {
  if (!result.success) return null;

  if ((config.weightPrecision ?? 'bf16') !== 'fp32') return null;

  return 'FP32 weights are rarely needed for inference — BF16 or FP16 halve memory with negligible quality difference.';
};

// ── 7. Flash Attention (validated) ───────────────────────────────────

const flashAttention: InferenceCandidateGenerator = (config, result) => {
  if (!result.success) return null;

  // flashAttention defaults to true in the engine when not specified
  if (config.flashAttention !== false) return null;

  const gpu = resolveGPU(config);
  if (!supportsFlashAttention(gpu)) return null;

  return {
    message: 'Consider enabling Flash Attention to reduce activation memory.',
    configMutation: (cfg) => ({ ...cfg, flashAttention: true }),
    validationMode: 'memory',
  };
};

// ── 8. Paged Attention (validated) ────────────────────────────────────

const pagedAttentionEnable: InferenceCandidateGenerator = (config, result) => {
  if (!result.success) return null;
  if (config.pagedAttention !== false) return null;
  if (result.memory.kvCache <= result.memory.total * 0.10) return null;

  return {
    message: 'Consider enabling paged attention — it eliminates KV cache fragmentation from contiguous pre-allocation, reducing memory waste.',
    configMutation: (cfg) => ({ ...cfg, pagedAttention: true }),
    validationMode: 'memory',
  };
};

// ── 9. KV cache precision match (validated) ──────────────────────────

const kvCachePrecisionMatch: InferenceCandidateGenerator = (config, result) => {
  if (!result.success) return null;

  const weightPrecision = config.weightPrecision ?? 'bf16';
  const weightBytes = getPrecisionBytes(weightPrecision);
  const kvBytes = getPrecisionBytes(config.kvCachePrecision ?? 'bf16');

  if (kvBytes <= weightBytes) return null;
  if (result.memory.kvCache / result.memory.total <= 0.05) return null;

  return {
    message: 'Consider reducing KV cache precision to match weight precision — KV values are computed from quantized weights, so higher KV precision wastes memory.',
    configMutation: (cfg) => ({ ...cfg, kvCachePrecision: weightPrecision }),
    validationMode: 'memory',
  };
};

// ── 10. GPU generation upgrade (heuristic) ───────────────────────────

const gpuUpgrade: InferenceGenerator = (config, result) => {
  if (!result.success) return null;
  const gpu = resolveGPU(config);
  const currentGen = new Set(['hopper', 'blackwell', 'ada', 'cdna3', 'cdna4']);
  if (currentGen.has(gpu.architecture)) return null;
  if (result.latency.tpot <= 20) return null;
  return 'Consider a newer-generation GPU — Hopper and Blackwell offer higher memory bandwidth and FP8 support.';
};

// ── 11. TP increase (validated) ──────────────────────────────────────

const tpIncrease: InferenceCandidateGenerator = (config, result) => {
  if (!result.success) return null;

  const tp = config.tensorParallel ?? 1;
  const numGPUs = config.numGPUs ?? 1;
  const gpu = resolveGPU(config);
  const gpuMemoryBytes = gpuCapacityBytes(gpu.memoryGB);

  if (tp !== 1) return null;
  if (numGPUs < 2) return null;
  if (result.memory.weights <= gpuMemoryBytes * 0.5) return null;

  return {
    message: 'Consider enabling tensor parallelism — sharding weights across GPUs increases effective memory bandwidth, improving decode throughput.',
    configMutation: (cfg) => ({ ...cfg, tensorParallel: 2 }),
  };
};

// ── 12. EP increase (validated) ───────────────────────────────────────

const epIncrease: InferenceCandidateGenerator = (config, result) => {
  if (!result.success) return null;

  const model = resolveModel(config);
  if (!model.isMoE || !model.numExperts) return null;

  const { routedExpertParams } = moeParamSplit(model);
  if (routedExpertParams <= model.totalParams * 0.30) return null;

  const tp = config.tensorParallel ?? 1;
  const currentEP = config.expertParallel ?? 1;
  const numGPUs = config.numGPUs ?? 1;

  if (numGPUs < tp * 2) return null;

  for (const candidateEP of [2, 4, 8, 16]) {
    if (candidateEP <= currentEP) continue;
    if (candidateEP * tp > numGPUs) continue;
    if (model.numExperts % candidateEP !== 0) continue;

    return {
      message: 'Consider enabling expert parallelism — distributing MoE experts across GPUs reduces per-GPU expert weight reads and can improve decode throughput.',
      configMutation: (cfg) => ({ ...cfg, expertParallel: candidateEP }),
    };
  }

  return null;
};

// ── 13. EP reduction (validated) ──────────────────────────────────────

const epReduction: InferenceCandidateGenerator = (config, result) => {
  if (!result.success) return null;

  const model = resolveModel(config);
  if (!model.isMoE || !model.numExperts) return null;

  const currentEP = config.expertParallel ?? 1;
  if (currentEP < 2) return null;

  const newEP = Math.max(1, Math.floor(currentEP / 2));

  return {
    message: 'Consider reducing expert parallelism — fewer GPUs per replica means more independent serving replicas, which can increase total cluster throughput.',
    configMutation: (cfg) => ({ ...cfg, expertParallel: newEP }),
  };
};

// ── 13. TP reduction (validated) ─────────────────────────────────────

const tpReduction: InferenceCandidateGenerator = (config, result) => {
  if (!result.success) return null;

  const tp = config.tensorParallel ?? 1;
  if (tp < 2) return null;

  return {
    message: 'Consider reducing tensor parallelism — each GPU can serve independently, increasing total replicas and cluster throughput.',
    configMutation: (cfg) => ({ ...cfg, tensorParallel: Math.floor(tp / 2) }),
  };
};

// ── 14. Context length headroom (validated) ──────────────────────────

const contextLengthHeadroom: InferenceCandidateGenerator = (config, result) => {
  if (!result.success) return null;

  if (result.utilization.memoryCapacityUtilization >= 0.50) return null;

  const model = resolveModel(config);
  const inputSeqLen = config.inputSeqLen ?? 512;
  const outputSeqLen = config.outputSeqLen ?? 256;
  const totalSeqLen = inputSeqLen + outputSeqLen;

  // Only suggest if there's meaningful room to grow
  if (totalSeqLen >= model.maxSeqLength * 0.5) return null;

  return {
    message: 'GPU memory has headroom — you could increase sequence length for longer context.',
    configMutation: (cfg) => ({ ...cfg, inputSeqLen: inputSeqLen * 2 }),
    validationMode: 'memory',
  };
};

// ── 15. Batch size decrease (validated) ───────────────────────────────

const batchSizeDecrease: InferenceCandidateGenerator = (config, result) => {
  if (!result.success) return null;

  const batchSize = config.batchSize ?? 1;
  if (batchSize <= 1) return null;

  // Suggest when TPOT is high (latency-sensitive workloads)
  if (result.latency.tpot <= 100) return null;

  return {
    message: 'Consider reducing batch size if per-request latency is critical — smaller batches reduce time-per-output-token at the cost of total throughput.',
    configMutation: (cfg) => ({ ...cfg, batchSize: Math.ceil(batchSize / 2) }),
    validationMode: 'latency',
  };
};

// ── 17. Context length warning ───────────────────────────────────────

const contextLengthWarning: InferenceGenerator = (config, result) => {
  if (!result.success) return null;

  const model = resolveModel(config);
  const inputSeqLen = config.inputSeqLen ?? 512;
  const outputSeqLen = config.outputSeqLen ?? 256;
  const totalSeqLen = inputSeqLen + outputSeqLen;

  if (totalSeqLen <= 0.75 * model.maxSeqLength) return null;

  return "Sequence length is approaching the model's maximum context — KV cache memory grows linearly and may limit batch size.";
};

// ── 20. Speculative decoding disable (validated) ─────────────────────

const speculativeDecodingDisable: InferenceGenerator = (config, result) => {
  if (!result.success) return null;
  if (!config.speculativeEnabled) return null;

  const batchSize = config.batchSize ?? 1;
  const speedup = result.speculative?.speedup ?? 1.0;

  // Suggest disabling when batch is large or speedup is marginal
  if (batchSize <= 16 && speedup >= 1.2) return null;

  return 'Consider disabling speculative decoding — at higher batch sizes, the draft model overhead outweighs the benefit.';
};

// ── 21. Speculative decoding enable (heuristic) ─────────────────────

const speculativeDecodingEnable: InferenceGenerator = (config, result) => {
  if (!result.success) return null;
  if (config.speculativeEnabled) return null;

  const batchSize = config.batchSize ?? 1;
  if (batchSize > 8) return null;
  if (result.latency.tpot <= 30) return null;

  const model = resolveModel(config);
  const activeParams = model.activeParams ?? model.totalParams;
  if (activeParams <= 10e9) return null;

  return 'Consider enabling speculative decoding — with a smaller draft model, it can significantly reduce decode latency at low batch sizes.';
};

// ── 21b. Prefill bottleneck (heuristic) ───────────────────────────────

const prefillBottleneck: InferenceGenerator = (config, result) => {
  if (!result.success) return null;
  if (result.latency.prefillTime <= result.latency.totalLatency * 0.50) return null;
  const batchSize = config.batchSize ?? 1;
  if (batchSize <= 1) return null;
  if (config.continuousBatching) return null;
  return 'Prefill dominates total request latency at this batch size — consider enabling continuous batching to reduce per-request TTFT by prefilling one sequence at a time.';
};

// ── 22. Continuous batching enable (validated) ───────────────────────

const continuousBatchingEnable: InferenceCandidateGenerator = (config, result) => {
  if (!result.success) return null;
  if (config.continuousBatching) return null;

  const batchSize = config.batchSize ?? 1;
  const numReplicas = computeNumReplicas(config);
  const batchPerReplica = Math.ceil(batchSize / numReplicas);
  if (batchPerReplica <= 4) return null;

  return {
    message: 'Consider enabling continuous batching — it keeps GPU slots occupied and eliminates inter-batch idle time.',
    configMutation: (cfg) => ({ ...cfg, continuousBatching: true }),
  };
};

// ── 23. Continuous batching disable (heuristic) ──────────────────────

const continuousBatchingDisable: InferenceGenerator = (config, result) => {
  if (!result.success) return null;
  if (!config.continuousBatching) return null;

  const batchSize = config.batchSize ?? 1;
  if (batchSize > 2) return null;

  return 'Consider disabling continuous batching — at very low batch sizes, the scheduling overhead outweighs the benefit.';
};

// ── 24. Batch size search (validated) ─────────────────────────────────

const batchSizeSearch: InferenceCandidateGenerator = (config, result) => {
  if (!result.success) return null;

  if (result.utilization.memoryCapacityUtilization >= 0.80) return null;

  const batchSize = config.batchSize ?? 1;
  const maxConcurrent = result.maxConcurrentRequests;
  if (batchSize >= maxConcurrent) return null;

  // Try batch×2, batch×4, and batch×8
  const multiplierCandidates = [batchSize * 2, batchSize * 4, batchSize * 8];

  // Also try per-replica targets for large clusters where batch << numReplicas
  const numReplicas = computeNumReplicas(config);
  const perReplicaTargets = [2, 4, 8, 16, 32, 64];
  const replicaCandidates = perReplicaTargets.map(t => t * numReplicas);

  const candidateSet = new Set([...multiplierCandidates, ...replicaCandidates]);
  const candidates = Array.from(candidateSet)
    .filter(b => b > batchSize && b <= maxConcurrent)
    .sort((a, b) => a - b);
  if (candidates.length === 0) return null;

  // Pick the best candidate by throughput
  let bestBatch = batchSize;
  let bestThroughput = result.throughput.tokensPerSecond;

  for (const candidate of candidates) {
    try {
      const mutated = runInferenceSimulationRaw({ ...config, batchSize: candidate });
      if (!mutated.success) continue;
      if (mutated.throughput.tokensPerSecond > bestThroughput * 1.01) {
        bestThroughput = mutated.throughput.tokensPerSecond;
        bestBatch = candidate;
      }
    } catch { continue; }
  }

  if (bestBatch === batchSize) return null;

  return {
    message: 'Memory headroom available — consider increasing batch size to improve throughput, keeping latency SLAs in mind.',
    configMutation: (cfg) => ({ ...cfg, batchSize: bestBatch }),
  };
};

// ── Positive confirmation (fallback) ─────────────────────────────────

const positiveConfirmation = (_config: InferenceSimulationConfig, _result: InferenceSimulationResult, hasActionable: boolean): string => {
  if (!hasActionable) return 'Well-configured for this model and hardware setup.';
  return 'Solid baseline — some optimizations may help.';
};

// ── public API ───────────────────────────────────────────────────────

export const OOM_GENERATORS: InferenceCandidateGenerator[] = [
  /* 1 */ oomPagedAttention,
  /* 2 */ oomTPIncrease,
  /* 3 */ oomEPIncrease,
  /* 4 */ oomTPAndEPCombined,
  /* 5 */ oomQuantization,
  /* 6 */ oomBatchReduction,
  /* 7 */ wrapGenerator(oomContextReduction),
  // oomFallback is NOT here — it's appended as last-resort after the loop
];

export const SUCCESS_GENERATORS: InferenceCandidateGenerator[] = [
  // Warnings first (always relevant, non-actionable)
  /* 7  */ wrapGenerator(fp32Warning),
  /* 8  */ flashAttention,
  /* 9  */ pagedAttentionEnable,
  /* 10 */ wrapGenerator(contextLengthWarning),
  // Speculative decoding disable (heuristic, catches bad configs early)
  /* 10 */ wrapGenerator(speculativeDecodingDisable),
  // Optimization suggestions (actionable, validated where possible)
  /* 11 */ kvCachePrecisionMatch,
  /* 12 */ wrapGenerator(gpuUpgrade),
  /* 14 */ tpIncrease,
  /* 15 */ epIncrease,
  /* 16 */ epReduction,
  /* 17 */ wrapGenerator(prefillBottleneck),
  /* 18 */ continuousBatchingEnable,
  /* 19 */ contextLengthHeadroom,
  /* 20 */ tpReduction,
  /* 21 */ batchSizeDecrease,
  /* 22 */ wrapGenerator(continuousBatchingDisable),
  // Heuristic suggestions (later in priority)
  /* 23 */ wrapGenerator(speculativeDecodingEnable),
  /* 24 */ batchSizeSearch,
];

export function generateInferenceRecommendations(
  config: InferenceSimulationConfig,
  result: InferenceSimulationResult,
): string[] {
  const generators = result.success ? SUCCESS_GENERATORS : OOM_GENERATORS;
  const results: string[] = [];

  for (const gen of generators) {
    if (results.length >= 2) break;
    const candidate = gen(config, result);
    if (candidate === null) continue;

    if (validateInferenceCandidate(candidate, config, result)) {
      results.push(candidate.message);
    }
  }

  if (result.success) {
    results.unshift(positiveConfirmation(config, result, results.length > 0));
  } else if (results.length === 0) {
    // Last-resort fallback: no actionable OOM recommendation resolved the issue
    results.push(oomFallback(config, result));
  }

  return results;
}
