/**
 * Post-simulation recommendation generators
 *
 * 15 generators in priority order; returns first 2 non-null results
 * (plus 1 status message at position 0, up to 3 total).
 * Validated generators provide a configMutation for "what-if" validation:
 * the engine re-simulates with the mutated config and checks the targeted
 * dimension (throughput or memory) for ≥1% improvement.
 *
 * Heuristic-only generators (no configMutation) are used for compound warnings
 * (memoryCritical) and diagnostic observations (scaleEfficiency) where the
 * solution involves changing training semantics (GBS, cluster size).
 *
 * Each recommendation suggests a direction (increase/decrease/try strategy)
 * rather than prescribing specific values. Numbers are already on the results
 * page — recommendations just say what to change.
 */
// See docs/OPTIMIZER.md for recommendation engine architecture.

import { SimulationEngine, type SimulationConfig, type SimulationMetrics } from './engine.ts';
import type { StrategyContext } from '../strategies/base.ts';
import { supportsFlashAttention } from '../hardware/gpu.ts';
// ── helpers ──────────────────────────────────────────────────────────

type StrategyType = SimulationConfig['strategyType'];

const ONE_D_STRATEGIES: StrategyType[] = ['ddp', 'fsdp', 'zero-1', 'zero-3'];
const TWO_D_STRATEGIES: StrategyType[] = ['fsdp-tp', 'zero1-tp'];
const THREE_D_STRATEGIES: StrategyType[] = ['ddp-tp-pp', 'zero1-tp-pp', 'fsdp-tp-pp'];

function is1D(s: StrategyType): boolean { return ONE_D_STRATEGIES.includes(s); }
function is2D(s: StrategyType): boolean { return TWO_D_STRATEGIES.includes(s); }
function is3D(s: StrategyType): boolean { return THREE_D_STRATEGIES.includes(s); }

function getEffectiveTP(config: SimulationConfig, gpusPerNode: number): number {
  if (config.strategyConfig?.tp) return config.strategyConfig.tp;
  if (is2D(config.strategyType) || is3D(config.strategyType)) {
    return gpusPerNode;
  }
  return 1;
}

function getEffectivePP(config: SimulationConfig, totalGPUs: number, tp: number): number {
  if (config.strategyConfig?.pp) return config.strategyConfig.pp;
  if (is3D(config.strategyType)) {
    return Math.min(8, Math.floor(totalGPUs / tp));
  }
  return 1;
}

function getEffectiveCP(config: SimulationConfig): number {
  return config.strategyConfig?.cp ?? 1;
}


function getEffectiveDP(config: SimulationConfig, ctx: StrategyContext): number {
  const tp = getEffectiveTP(config, ctx.cluster.gpusPerNode);
  const pp = getEffectivePP(config, ctx.cluster.totalGPUs, tp);
  const cp = getEffectiveCP(config);
  return Math.max(1, Math.floor(ctx.cluster.totalGPUs / (tp * pp * cp)));
}

function samplesPerGPU(config: SimulationConfig, ctx: StrategyContext): number {
  return config.globalBatchSize / getEffectiveDP(config, ctx);
}

// ── candidate / validation types ────────────────────────────────────

export interface RecommendationCandidate {
  message: string;
  configMutation?: (config: SimulationConfig) => SimulationConfig;
  /** Which metric the mutation targets. Validation checks this dimension for ≥1% improvement. */
  validationMode?: 'throughput' | 'memory';
  /** If set, reject mutations whose resulting memory utilization exceeds this ceiling. */
  maxMemoryUtilization?: number;
}

type CandidateGenerator = (
  config: SimulationConfig,
  ctx: StrategyContext,
  metrics: SimulationMetrics,
) => RecommendationCandidate | null;

// Simple generators return plain string|null — wrap them into CandidateGenerator
type Generator = (
  config: SimulationConfig,
  ctx: StrategyContext,
  metrics: SimulationMetrics,
) => string | null;

function wrapGenerator(gen: Generator): CandidateGenerator {
  return (config, ctx, metrics) => {
    const msg = gen(config, ctx, metrics);
    return msg !== null ? { message: msg } : null;
  };
}

// ── validation ──────────────────────────────────────────────────────

export function validateCandidate(
  candidate: RecommendationCandidate,
  config: SimulationConfig,
  currentMetrics: SimulationMetrics,
): boolean {
  if (!candidate.configMutation) return true; // no mutation = heuristic-only, always pass

  // Clear pre-computed GA so buildContext recomputes it for the new parallelism
  // layout. Mutations that change TP/PP/EP alter DP, which invalidates GA.
  const mutatedConfig = {
    ...candidate.configMutation(config),
    gradientAccumulationSteps: undefined,
  };
  try {
    const engine = new SimulationEngine();
    engine.configure(mutatedConfig);
    const mutatedMetrics = engine.simulate();

    // Reject if mutated config OOMs or exceeds the candidate's memory ceiling
    const memCeiling = candidate.maxMemoryUtilization ?? 1.0;
    if (mutatedMetrics.memoryUtilization > memCeiling) return false;

    // If current config is already OOM, accept any mutation that fits in memory.
    // Throughput comparison is meaningless when the current config can't run.
    if (currentMetrics.memoryUtilization > 1.0) return true;

    // Check the dimension this generator targets
    const mode = candidate.validationMode ?? 'throughput';
    switch (mode) {
      case 'throughput':
        // Use tokensPerSecond (not MFU) because MFU's denominator changes with
        // precision — FP8 doubles peak TFLOPS, making MFU drop even when throughput
        // genuinely increases.
        return mutatedMetrics.tokensPerSecond > currentMetrics.tokensPerSecond * 1.01;
      case 'memory': {
        // Memory must improve. Throughput guard depends on how tight memory is:
        // ≥85%: allow up to 1% throughput regression (memory headroom matters)
        // <85%: require no throughput regression (memory is comfortable)
        // OOM configs already bypass this entire check at line 124.
        const tightMemory = currentMetrics.memoryUtilization >= 0.85;
        const tpsFloor = tightMemory ? 0.99 : 1.0;
        return mutatedMetrics.memoryUtilization < currentMetrics.memoryUtilization * 0.99
          && mutatedMetrics.tokensPerSecond >= currentMetrics.tokensPerSecond * tpsFloor;
      }
    }
  } catch {
    return false; // config threw (invalid parallelism, etc.)
  }
}

// ── 1. Strategy upgrade path (validated) ──────────────────────────────

const strategyUpgrade: CandidateGenerator = (config, ctx, metrics) => {
  const { strategyType } = config;
  const modelParams = ctx.model.totalParams;
  const memUtil = metrics.memoryUtilization;
  const commOverhead = metrics.communicationOverhead;
  const multiNode = ctx.cluster.numNodes > 1;
  const gpusPerNode = ctx.cluster.gpusPerNode;

  // DDP → FSDP (memory claim)
  if (strategyType === 'ddp' && modelParams > 7e9 && memUtil > 0.70) {
    return {
      message: 'Consider switching to FSDP — DDP replicates all parameters on every GPU. Sharding optimizer states would significantly reduce memory pressure.',
      configMutation: (cfg) => ({ ...cfg, strategyType: 'fsdp' as const }),
      validationMode: 'memory',
    };
  }

  // Any 1D → FSDP + TP (throughput claim)
  if (is1D(strategyType)) {
    if (modelParams > 30e9 && multiNode) {
      return {
        message: 'Consider adding Tensor Parallelism (e.g. FSDP + TP) — large models across multiple nodes benefit from keeping heavy communication on fast intra-node links.',
        configMutation: (cfg) => ({
          ...cfg,
          strategyType: 'fsdp-tp' as const,
          strategyConfig: { ...cfg.strategyConfig, tp: gpusPerNode },
        }),
      };
    }
    if (modelParams > 13e9 && commOverhead > 0.30) {
      return {
        message: 'Consider adding Tensor Parallelism (e.g. FSDP + TP) — communication overhead is high with single-dimension parallelism. Intra-node Tensor Parallelism would reduce cross-node traffic.',
        configMutation: (cfg) => ({
          ...cfg,
          strategyType: 'fsdp-tp' as const,
          strategyConfig: { ...cfg.strategyConfig, tp: gpusPerNode },
        }),
      };
    }
  }

  // Any 2D → FSDP + TP + PP (memory claim)
  if (is2D(strategyType) && modelParams > 70e9 && memUtil > 0.85) {
    return {
      message: 'Consider adding Pipeline Parallelism — splitting layers into pipeline stages would reduce per-GPU memory for this large model.',
      configMutation: (cfg) => ({
        ...cfg,
        strategyType: 'fsdp-tp-pp' as const,
        strategyConfig: { ...cfg.strategyConfig, pp: 2 },
      }),
      validationMode: 'memory',
    };
  }

  // DDP-TP-PP → FSDP-TP-PP (memory claim)
  if (strategyType === 'ddp-tp-pp' && memUtil > 0.75) {
    return {
      message: 'Consider replacing DDP with FSDP in the data-parallel dimension — sharding optimizer states would reduce memory pressure.',
      configMutation: (cfg) => ({ ...cfg, strategyType: 'fsdp-tp-pp' as const }),
      validationMode: 'memory',
    };
  }

  return null;
};

// ── 2. Memory critical ───────────────────────────────────────────────

const memoryCritical: Generator = (config, ctx, metrics) => {
  if (metrics.memoryUtilization > 0.95) {
    const isOOM = metrics.memoryUtilization > 1.0;
    const prefix = isOOM
      ? 'Not enough GPU memory for this configuration.'
      : 'Memory is dangerously close to OOM.';
    const ac = config.activationCheckpointing ?? true;
    const acg = config.checkpointingGranularity ?? 'full';
    const fa = (config.flashAttention ?? true) || !supportsFlashAttention(ctx.cluster.node.gpu);
    const canReduceMBS = config.microBatchSize > 1;
    if (ac && fa) {
      const canUpgradeAC = acg === 'selective';
      if (canUpgradeAC) {
        if (canReduceMBS) {
          return `${prefix} Try switching to full activation checkpointing, reducing micro-batch size, or shortening sequence length.`;
        }
        return `${prefix} Try switching to full activation checkpointing or reducing sequence length.`;
      }
      if (canReduceMBS) {
        return `${prefix} Try reducing micro-batch size or sequence length.`;
      }
      return `${prefix} Try reducing sequence length or switching to a strategy that shards more aggressively.`;
    }
    const tips: string[] = [];
    if (!ac) tips.push('activation checkpointing');
    if (!fa) tips.push('Flash Attention');
    if (canReduceMBS) {
      return `${prefix} Try enabling ${tips.join(' and ')} or reducing micro-batch size.`;
    }
    return `${prefix} Try enabling ${tips.join(' and ')}.`;
  }
  return null;
};

// ── 3. Pipeline reduction (NEW — validated) ──────────────────────────
//
// When memory is tight on a 3D strategy with high PP, reducing pipeline stages
// is often better than reducing MBS: total FSDP sharding (TP×PP×DP = totalGPUs)
// stays the same, so memory is similar, but more DP means less gradient
// accumulation and better throughput.

const pipelineReductionSuggestion: CandidateGenerator = (config, ctx, metrics) => {
  if (!is3D(config.strategyType)) return null;

  if (metrics.memoryUtilization <= 0.90) return null;

  const gpusPerNode = ctx.cluster.gpusPerNode;
  const totalGPUs = ctx.cluster.totalGPUs;
  const tp = getEffectiveTP(config, gpusPerNode);
  const pp = getEffectivePP(config, totalGPUs, tp);

  if (pp <= 2) return null;

  const newPP = Math.floor(pp / 2);
  if (ctx.model.numLayers % newPP !== 0) return null;

  return {
    message: 'Consider reducing Pipeline Parallelism — fewer pipeline stages would increase data parallelism and improve throughput while FSDP sharding keeps memory manageable.',
    configMutation: (cfg) => ({
      ...cfg,
      strategyConfig: {
        ...cfg.strategyConfig,
        pp: newPP,
      },
    }),
  };
};

// ── 4. Scale efficiency ──────────────────────────────────────────────

const scaleEfficiency: Generator = (config, ctx, metrics) => {
  const modelParams = ctx.model.totalParams;
  const totalGPUs = ctx.cluster.totalGPUs;
  const mfu = metrics.mfu;
  const memUtil = metrics.memoryUtilization;

  // OOM — memoryCritical already handles this, MFU is meaningless
  if (memUtil > 1.0) return null;

  // MBS exceeds samples-per-GPU → wasted compute
  const spg = samplesPerGPU(config, ctx);
  if (config.microBatchSize > spg && mfu < 0.35) {
    return 'Micro-batch size exceeds samples per GPU — try increasing global batch size so each GPU processes at least one full micro-batch.';
  }

  if (mfu >= 0.25) return null;

  const paramsPerGPU = modelParams / totalGPUs;

  // Cluster is massively over-provisioned
  if (paramsPerGPU < 50e6 && totalGPUs > 32) {
    return 'Cluster is over-provisioned for this model — consider training a larger model to utilize this many GPUs.';
  }

  // MFU critically low
  if (mfu < 0.15) {
    const tokensPerStep = config.globalBatchSize * ctx.seqLength;
    const commOverhead = metrics.communicationOverhead;

    if (tokensPerStep > 10e6) {
      // If MBS is small and communication dominates, the real fix is MBS — defer
      // to communicationOverhead / microBatchSizing generators downstream.
      if (config.microBatchSize < 4 && commOverhead > 0.5 && memUtil < 0.75) {
        return null;
      }
      return 'MFU is critically low — batch size is already large. Consider fewer GPUs or a larger model.';
    }
    if (memUtil > 0.85) {
      return 'MFU is critically low and memory is nearly full — try increasing global batch size to amortize communication overhead.';
    }
    return 'MFU is critically low — try increasing global batch size to give each GPU more work per step.';
  }

  return null;
};

// ── 5. Data parallelism recovery (validated) ─────────────────────────
//
// When DP=1 on a multi-GPU config, all GPUs are consumed by TP/PP/EP
// with no data-parallel replicas. Reducing TP (or PP if TP=1) frees
// GPUs for DP, which typically improves throughput for smaller models.

const dataParallelismSuggestion: CandidateGenerator = (config, ctx, _metrics) => {
  if (is1D(config.strategyType)) return null;
  if (ctx.cluster.totalGPUs <= 1) return null;

  const gpusPerNode = ctx.cluster.gpusPerNode;
  const totalGPUs = ctx.cluster.totalGPUs;
  const tp = getEffectiveTP(config, gpusPerNode);
  const pp = getEffectivePP(config, totalGPUs, tp);
  const cp = getEffectiveCP(config);
  const dp = Math.floor(totalGPUs / (tp * pp * cp));

  if (dp > 1) return null;

  // Try halving TP first (higher comm overhead per degree)
  if (tp > 1) {
    const newTP = Math.floor(tp / 2) || 1;
    if (ctx.model.numAttentionHeads % newTP !== 0) return null;

    return {
      message: 'There is no data parallelism — consider reducing Tensor Parallelism to allow data-parallel replicas for better throughput.',
      configMutation: (cfg) => ({
        ...cfg,
        strategyConfig: {
          ...cfg.strategyConfig,
          tp: newTP,
        },
      }),
    };
  }

  // TP is already 1 — try halving PP
  if (pp > 1) {
    const newPP = Math.floor(pp / 2) || 1;
    if (newPP > 1 && ctx.model.numLayers % newPP !== 0) return null;

    return {
      message: 'There is no data parallelism — consider reducing Pipeline Parallelism to allow data-parallel replicas for better throughput.',
      configMutation: (cfg) => ({
        ...cfg,
        strategyConfig: {
          ...cfg.strategyConfig,
          pp: newPP,
        },
      }),
    };
  }

  return null;
};

// ── 6. TP topology (validated) ────────────────────────────────────────

const tpTopology: CandidateGenerator = (config, ctx, metrics) => {
  if (is1D(config.strategyType)) return null;

  const gpusPerNode = ctx.cluster.gpusPerNode;
  const tp = getEffectiveTP(config, gpusPerNode);
  const commOverhead = metrics.communicationOverhead;
  const modelParams = ctx.model.totalParams;
  const heads = ctx.model.numAttentionHeads;

  // Cross-node TP is very slow — reduce to stay within a single node
  if (tp > gpusPerNode) {
    return {
      message: 'Tensor Parallelism crosses node boundaries — consider reducing it to stay within a single node for faster communication.',
      configMutation: (cfg) => ({
        ...cfg,
        strategyConfig: { ...cfg.strategyConfig, tp: gpusPerNode },
      }),
    };
  }

  // Under-utilizing intra-node bandwidth — increase TP to fill node
  if (tp < gpusPerNode && modelParams > 13e9 && (heads % gpusPerNode === 0) && commOverhead > 0.20) {
    return {
      message: 'Consider increasing Tensor Parallelism to fill the node — attention heads divide evenly and it may reduce communication overhead.',
      configMutation: (cfg) => ({
        ...cfg,
        strategyConfig: { ...cfg.strategyConfig, tp: gpusPerNode },
      }),
    };
  }

  return null;
};

// ── 6. Activation checkpointing (validated) ──────────────────────────

const activationCheckpointingSuggestion: CandidateGenerator = (config, _ctx, metrics) => {
  const acEnabled = config.activationCheckpointing ?? true;
  const acg = config.checkpointingGranularity ?? 'full';
  const memUtil = metrics.memoryUtilization;

  // A) Off + high memory → suggest selective (less compute hit than full)
  if (!acEnabled && memUtil > 0.80) {
    return {
      message: 'Consider enabling selective activation checkpointing — it discards attention activations with minimal compute overhead.',
      configMutation: (cfg) => ({ ...cfg, activationCheckpointing: true, checkpointingGranularity: 'selective' as const }),
    };
  }

  // B) Selective + OOM → suggest full (more memory savings)
  if (acEnabled && acg === 'selective' && memUtil > 1.0) {
    return {
      message: 'Consider switching to full activation checkpointing for deeper memory savings.',
      configMutation: (cfg) => ({ ...cfg, checkpointingGranularity: 'full' as const }),
    };
  }

  // C) Full + low memory → suggest selective (less compute overhead)
  if (acEnabled && acg === 'full' && memUtil < 0.70) {
    return {
      message: 'Consider switching to selective activation checkpointing — it keeps MLP activations and only recomputes attention, reducing compute overhead.',
      configMutation: (cfg) => ({ ...cfg, checkpointingGranularity: 'selective' as const }),
    };
  }

  return null;
};

// ── 7. Flash Attention (validated by memory) ──────────────────────────

const flashAttentionSuggestion: CandidateGenerator = (config, ctx, metrics) => {
  const faEnabled = config.flashAttention ?? true;
  if (faEnabled) return null;

  const gpu = ctx.cluster.node.gpu;
  if (!supportsFlashAttention(gpu)) return null;

  // Only suggest when memory is moderately used or sequence length is long
  if (metrics.memoryUtilization <= 0.60 && ctx.seqLength < 4096) return null;

  return {
    message: 'Consider enabling Flash Attention — it reduces activation memory with no quality impact.',
    configMutation: (cfg) => ({ ...cfg, flashAttention: true }),
    validationMode: 'memory',
  };
};

// ── 8. Pipeline bubble + schedule upgrade (expanded — validated) ─────
//
// Two triggers for suggesting interleaved 1F1B:
// A) High bubble — interleaved reduces bubble by overlapping virtual stages
// B) Tight memory — interleaved reduces peak in-flight microbatches from pp to
//    ceil(pp/v), significantly cutting activation memory

const pipelineBubbleAndSchedule: CandidateGenerator = (config, ctx, metrics) => {
  if (!is3D(config.strategyType)) return null;

  const gpusPerNode = ctx.cluster.gpusPerNode;
  const tp = getEffectiveTP(config, gpusPerNode);
  const pp = getEffectivePP(config, ctx.cluster.totalGPUs, tp);
  if (pp <= 1) return null;

  const sc = config.strategyConfig ?? {};
  const currentSchedule = sc.pipelineSchedule ?? '1f1b';
  const numLayers = ctx.model.numLayers;
  const bubble = metrics.pipelineBubble;
  const memUtil = metrics.memoryUtilization;

  const ga = sc.numMicroBatches ?? ctx.gradientAccumulationSteps;

  // DualPipeV degraded mode: GA < 2*PP means insufficient micro-batches for
  // bidirectional overlap. Suggest increasing GBS to reach the threshold.
  if (currentSchedule === 'dualpipe-v' && ga < 2 * pp) {
    const dp = getEffectiveDP(config, ctx);
    const mbs = config.microBatchSize;
    const targetGBS = 2 * pp * mbs * dp;
    const targetGA = Math.ceil(targetGBS / (mbs * dp));
    if (targetGA <= 256) {
      return {
        message: 'DualPipeV is in degraded mode — consider increasing global batch size for sufficient gradient accumulation.',
        configMutation: (cfg) => ({
          ...cfg,
          globalBatchSize: targetGBS,
          gradientAccumulationSteps: undefined,
        }),
      };
    }
  }

  // Suggest DualPipeV when m >= 2*PP and schedule is 1F1B or interleaved
  if (currentSchedule !== 'dualpipe-v' && bubble > 0.10 && ga >= 2 * pp) {
    return {
      message: 'Consider DualPipeV pipeline schedule — it achieves near-zero bubble by overlapping forward and backward passes.',
      configMutation: (cfg) => ({
        ...cfg,
        strategyConfig: {
          ...cfg.strategyConfig,
          pipelineSchedule: 'dualpipe-v' as const,
        },
      }),
    };
  }

  // Suggest interleaved when not already using it and layers fit
  if (currentSchedule === '1f1b') {
    // Find the best valid v (largest = most bubble reduction)
    const validVs = [2, 3, 4, 6, 8].filter(v => numLayers % (pp * v) === 0);
    if (validVs.length > 0) {
      const bestV = validVs[validVs.length - 1];
      // Trigger on high bubble OR tight memory
      if (bubble > 0.10 || memUtil > 0.85) {
        const message = memUtil > 0.85
          ? 'Consider switching to interleaved 1F1B pipeline schedule — it reduces peak activation memory by processing smaller virtual stages.'
          : 'Consider switching to interleaved 1F1B pipeline schedule — it reduces the pipeline bubble by overlapping virtual stages.';
        return {
          message,
          configMutation: (cfg) => ({
            ...cfg,
            strategyConfig: {
              ...cfg.strategyConfig,
              pipelineSchedule: 'interleaved-1f1b' as const,
              interleavedStages: bestV,
            },
          }),
        };
      }
    }
  }

  // Suggest increasing v when already on interleaved but bubble is still high
  if (currentSchedule === 'interleaved-1f1b' && bubble > 0.10) {
    const currentV = sc.interleavedStages ?? 2;
    const validVs = [2, 3, 4, 6, 8].filter(v => v > currentV && numLayers % (pp * v) === 0);
    if (validVs.length > 0) {
      const nextV = validVs[0];
      return {
        message: 'Consider increasing virtual stages — it further reduces the pipeline bubble.',
        configMutation: (cfg) => ({
          ...cfg,
          strategyConfig: {
            ...cfg.strategyConfig,
            interleavedStages: nextV,
          },
        }),
      };
    }
  }

  // Fallback: generic bubble advice when bubble is high (no mutation — heuristic only)
  if (bubble > 0.10) {
    return {
      message: 'Pipeline bubble is significant — consider increasing global batch size or reducing pipeline stages.',
    };
  }

  return null;
};

// ── 9. Sequence Parallelism (NEW — validated) ────────────────────────

const sequenceParallelismSuggestion: CandidateGenerator = (config, ctx, metrics) => {
  // Only for 2D or 3D strategies
  if (is1D(config.strategyType)) return null;

  const sc = config.strategyConfig ?? {};
  const spEnabled = sc.sequenceParallel ?? true; // defaults ON for combined strategies
  if (spEnabled) return null;

  const tp = getEffectiveTP(config, ctx.cluster.gpusPerNode);
  if (tp <= 1) return null;

  // Only suggest when memory or comm overhead is a concern
  if (metrics.memoryUtilization <= 0.70 && metrics.communicationOverhead <= 0.20) return null;

  return {
    message: 'Consider enabling Sequence Parallelism — it distributes activation memory across the TP group.',
    configMutation: (cfg) => ({
      ...cfg,
      strategyConfig: {
        ...cfg.strategyConfig,
        sequenceParallel: true,
      },
    }),
    validationMode: 'memory',
  };
};

// ── 10. Context Parallelism (validated) ─────────────────────────────

const contextParallelismSuggestion: CandidateGenerator = (config, ctx, metrics) => {
  if (metrics.memoryUtilization < 0.70) return null;
  if (ctx.seqLength < 8192) return null;

  const cp = getEffectiveCP(config);
  if (cp > 1) return null;

  // Don't suggest CP if it would split chunks below 8192 tokens
  const newCP = 2;
  if (ctx.seqLength / newCP < 8192) return null;

  const gpusPerNode = ctx.cluster.gpusPerNode;
  const totalGPUs = ctx.cluster.totalGPUs;
  const tp = getEffectiveTP(config, gpusPerNode);
  const pp = getEffectivePP(config, totalGPUs, tp);
  const newDP = Math.floor(totalGPUs / (tp * pp * newCP));
  if (newDP < 1) return null;
  if (ctx.seqLength % newCP !== 0) return null;

  return {
    message: 'Consider enabling Context Parallelism — it splits the sequence across GPUs, reducing activation memory for long sequences.',
    configMutation: (cfg) => ({
      ...cfg,
      strategyConfig: { ...cfg.strategyConfig, cp: newCP },
    }),
    validationMode: 'memory',
  };
};

// ── 11. Expert Parallelism (validated) ─────────────────────────────

const expertParallelismSuggestion: CandidateGenerator = (config, ctx, metrics) => {
  const model = ctx.model;
  if (!model.isMoE || !model.numExperts || model.numExperts < 8) return null;

  const sc = config.strategyConfig ?? {};
  const currentEP = sc.ep ?? 1;
  const gpusPerNode = ctx.cluster.gpusPerNode;
  const totalGPUs = ctx.cluster.totalGPUs;
  const tp = getEffectiveTP(config, gpusPerNode);
  const pp = getEffectivePP(config, totalGPUs, tp);
  const cp = getEffectiveCP(config);
  const dp = Math.floor(totalGPUs / (tp * pp * cp));
  const heads = ctx.model.numAttentionHeads;
  const memUtil = metrics.memoryUtilization;

  // Find largest valid EP > currentEP that divides both numExperts and DP
  function findBestEP(maxEP: number, dpDeg: number): number {
    for (let ep = maxEP; ep >= 2; ep--) {
      if (model.numExperts! % ep === 0 && dpDeg % ep === 0 && ep > currentEP) {
        return ep;
      }
    }
    return 0;
  }

  const validationMode = memUtil > 0.85 ? 'memory' as const : 'throughput' as const;

  // Strategy A: increase EP within current TP layout
  const maxEP_A = Math.floor(gpusPerNode / tp);
  const candidateA = findBestEP(maxEP_A, dp);

  if (candidateA > 0) {
    const message = currentEP === 1
      ? 'Consider enabling Expert Parallelism — distributing experts across GPUs would reduce per-GPU expert memory and improve throughput for this MoE model.'
      : 'Consider increasing Expert Parallelism — distributing experts across more GPUs would reduce per-GPU expert memory and communication volume.';
    return {
      message,
      configMutation: (cfg) => ({
        ...cfg,
        strategyConfig: {
          ...cfg.strategyConfig,
          ep: candidateA,
        },
      }),
      validationMode,
    };
  }

  // Strategy B: reduce TP to increase DP and maxEP, then set both in a compound mutation
  for (let newTP = Math.floor(tp / 2); newTP >= 1; newTP = Math.floor(newTP / 2)) {
    if (heads % newTP !== 0) continue;

    const newDP = Math.floor(totalGPUs / (newTP * pp));
    const newMaxEP = Math.floor(gpusPerNode / newTP);
    const candidateB = findBestEP(newMaxEP, newDP);

    if (candidateB > 0) {
      return {
        message: currentEP > 1
          ? 'Consider reducing Tensor Parallelism and increasing Expert Parallelism — trading TP for EP can improve throughput for MoE models by distributing experts more efficiently.'
          : 'Consider reducing Tensor Parallelism and enabling Expert Parallelism — trading TP for EP can improve throughput for MoE models by distributing experts more efficiently.',
        configMutation: (cfg) => ({
          ...cfg,
          strategyConfig: {
            ...cfg.strategyConfig,
            tp: newTP,
            ep: candidateB,
          },
        }),
        validationMode,
      };
    }

    if (newTP <= 1) break;
  }

  return null;
};

// ── 11. Communication overhead (partially validated) ──────────────────

const communicationOverhead: CandidateGenerator = (config, ctx, metrics) => {
  const commOverhead = metrics.communicationOverhead;
  const memUtil = metrics.memoryUtilization;
  const currentMBS = config.microBatchSize;

  if (commOverhead <= 0.25) return null;

  // Small micro-batch with room for memory — validate by throughput
  if (currentMBS < 4 && memUtil < 0.65 && currentMBS < samplesPerGPU(config, ctx)) {
    return {
      message: 'Consider increasing micro-batch size — larger batches amortize communication overhead.',
      configMutation: (cfg) => ({ ...cfg, microBatchSize: currentMBS * 2 }),
    };
  }

  return null;
};

// ── 12. Micro-batch sizing (validated) ─────────────────────────────────

const microBatchSizing: CandidateGenerator = (config, ctx, metrics) => {
  const mfu = metrics.mfu;
  const commOverhead = metrics.communicationOverhead;
  const currentMBS = config.microBatchSize;
  const spg = samplesPerGPU(config, ctx);
  const ga = ctx.gradientAccumulationSteps;

  if (mfu > 0.40) return null;
  if (currentMBS >= spg) return null;
  if (ga <= 1) return null;
  if (metrics.memoryUtilization > 0.75) return null;  // no room to grow MBS

  // Only recommend MBS increase when communication overhead is actually high —
  // that's the one scenario where fewer GA steps measurably helps.
  // Memory headroom alone says nothing about whether MFU will improve.
  if (commOverhead > 0.15 && ga >= 8) {
    return {
      message: 'Consider increasing micro-batch size — fewer gradient accumulation steps reduce communication overhead.',
      configMutation: (cfg) => ({ ...cfg, microBatchSize: currentMBS * 2 }),
    };
  }

  return null;
};

// ── 14. GBS scaling (validated) ───────────────────────────────────────
//
// Tries GBS×2 (fewer steps, potentially worse step time) and GBS÷2
// (more steps, potentially better step time). The optimizer uses
// timeToTrainHours directly; here we use 'throughput' validationMode
// since higher throughput correlates with faster training.

const gbsScaling: CandidateGenerator = (config, ctx, metrics) => {
  const gbs = config.globalBatchSize;
  const mbs = config.microBatchSize;
  const dp = getEffectiveDP(config, ctx);

  const candidates: number[] = [];

  // Try GBS×2 — guard: GA must not exceed 256
  const gaUp = Math.ceil((gbs * 2) / (mbs * dp));
  if (gaUp <= 256) candidates.push(gbs * 2);

  // Try GBS÷2 — guard: GA must be ≥ 1
  const gaDown = Math.ceil(Math.floor(gbs / 2) / (mbs * dp));
  if (gaDown >= 1 && Math.floor(gbs / 2) >= mbs) candidates.push(Math.floor(gbs / 2));

  if (candidates.length === 0) return null;

  // Pick the candidate with the best timeToTrainHours
  let bestGBS = gbs;
  let bestTime = Infinity;

  const seqLen = config.sequenceLength;
  const currentStepTime = metrics.stepTimeMs;
  const currentMaxSteps = config.maxSteps ?? 1000;
  const currentTime = currentStepTime * currentMaxSteps / 3.6e6;
  bestTime = currentTime;

  for (const candidateGBS of candidates) {
    try {
      const mutatedConfig = {
        ...config,
        globalBatchSize: candidateGBS,
        gradientAccumulationSteps: undefined,
      };
      const engine = new SimulationEngine();
      engine.configure(mutatedConfig);
      const mutatedMetrics = engine.simulate();
      if (mutatedMetrics.memoryUtilization > 1.0) continue;

      const targetTokens = (config.maxSteps ?? 1000) * gbs * seqLen;
      const candidateTokensPerStep = candidateGBS * seqLen;
      const candidateBaseSteps = Math.ceil(targetTokens / candidateTokensPerStep);
      const candidateTime = mutatedMetrics.stepTimeMs * candidateBaseSteps / 3.6e6;

      if (candidateTime < bestTime * 0.995) {
        bestTime = candidateTime;
        bestGBS = candidateGBS;
      }
    } catch { continue; }
  }

  if (bestGBS === gbs) return null;

  const direction = bestGBS > gbs ? 'increasing' : 'decreasing';
  return {
    message: `Consider ${direction} global batch size — it would change the compute-to-communication ratio and number of training steps, potentially reducing total training time.`,
    configMutation: (cfg) => ({
      ...cfg,
      globalBatchSize: bestGBS,
      gradientAccumulationSteps: undefined,
    }),
    validationMode: 'throughput' as const,
  };
};

// ── Positive confirmation ─────────────────────────────────────────────

const positiveConfirmation = (
  _config: SimulationConfig, _ctx: StrategyContext,
  metrics: SimulationMetrics, hasActionable: boolean,
): string | null => {
  if (metrics.memoryUtilization > 1.0) return null; // OOM — memoryCritical is the status
  if (!hasActionable) return 'Well-optimized for this model and cluster.';
  return 'Solid baseline — some optimizations may help.';
};

// ══════════════════════════════════════════════════════════════════════
// CONFIG-FIX GENERATORS — fire when validation errors exist
// ══════════════════════════════════════════════════════════════════════

// ── TP must divide attention heads ───────────────────────────────────

const tpHeadsFix: CandidateGenerator = (config, ctx, _metrics) => {
  const tp = getEffectiveTP(config, ctx.cluster.gpusPerNode);
  if (tp <= 1) return null;
  if (ctx.model.numAttentionHeads % tp === 0) return null;

  // Find largest valid TP < current
  const heads = ctx.model.numAttentionHeads;
  let bestTP = 1;
  for (let t = tp - 1; t >= 2; t--) {
    if (heads % t === 0 && t <= ctx.cluster.gpusPerNode) {
      bestTP = t;
      break;
    }
  }

  return {
    message: `Tensor Parallelism degree must evenly divide the model's attention heads — try reducing TP to a divisor of ${heads}.`,
    configMutation: (cfg) => ({
      ...cfg,
      strategyConfig: { ...cfg.strategyConfig, tp: bestTP },
    }),
  };
};

// ── PP×v must divide model layers ────────────────────────────────────

const pipelineLayersFix: CandidateGenerator = (config, ctx, _metrics) => {
  if (!is3D(config.strategyType)) return null;

  const tp = getEffectiveTP(config, ctx.cluster.gpusPerNode);
  const pp = getEffectivePP(config, ctx.cluster.totalGPUs, tp);
  if (pp <= 1) return null;

  const sc = config.strategyConfig ?? {};
  const schedule = sc.pipelineSchedule ?? '1f1b';
  const v = schedule === 'interleaved-1f1b' ? (sc.interleavedStages ?? 2) : 1;
  const numLayers = ctx.model.numLayers;

  if (numLayers % (pp * v) === 0) return null;

  // If interleaved is the problem, suggest disabling it first
  if (v > 1 && numLayers % pp === 0) {
    return {
      message: "Model layers aren't divisible by PP×v for interleaved scheduling — try switching back to standard 1F1B pipeline schedule.",
      configMutation: (cfg) => ({
        ...cfg,
        strategyConfig: { ...cfg.strategyConfig, pipelineSchedule: '1f1b' as const },
      }),
    };
  }

  // Otherwise suggest reducing PP to a valid divisor
  let bestPP = 1;
  for (let p = pp - 1; p >= 2; p--) {
    if (numLayers % p === 0) {
      bestPP = p;
      break;
    }
  }

  return {
    message: `Model layers (${numLayers}) must be evenly divisible by pipeline stages — try reducing PP to a divisor of ${numLayers}.`,
    configMutation: (cfg) => ({
      ...cfg,
      strategyConfig: { ...cfg.strategyConfig, pp: bestPP },
    }),
  };
};

const CONFIG_FIX_GENERATORS: CandidateGenerator[] = [
  tpHeadsFix,
  pipelineLayersFix,
];

// ── public API ───────────────────────────────────────────────────────

export const GENERATORS: CandidateGenerator[] = [
  /* 1  */ strategyUpgrade,
  /* 2  */ wrapGenerator(memoryCritical),
  /* 3  */ pipelineReductionSuggestion,
  /* 4  */ wrapGenerator(scaleEfficiency),
  /* 5  */ dataParallelismSuggestion,
  /* 6  */ tpTopology,
  /* 7  */ activationCheckpointingSuggestion,
  /* 8  */ flashAttentionSuggestion,
  /* 9  */ pipelineBubbleAndSchedule,
  /* 10 */ sequenceParallelismSuggestion,
  /* 11 */ contextParallelismSuggestion,
  /* 12 */ expertParallelismSuggestion,
  /* 13 */ communicationOverhead,
  /* 14 */ microBatchSizing,
  /* 15 */ gbsScaling,
];

export function generateRecommendations(
  config: SimulationConfig,
  ctx: StrategyContext,
  metrics: SimulationMetrics,
  errors: string[] = [],
): string[] {
  // When config has validation errors (not just OOM), prioritize config-fix generators.
  // OOM is handled by memoryCritical + validated generators in GENERATORS.
  const hasConfigErrors = errors.length > 0 && metrics.memoryUtilization <= 1.0;

  const generators = hasConfigErrors ? CONFIG_FIX_GENERATORS : GENERATORS;
  const results: string[] = [];

  for (const gen of generators) {
    if (results.length >= 2) break;
    const candidate = gen(config, ctx, metrics);
    if (candidate === null) continue;

    if (validateCandidate(candidate, config, metrics)) {
      results.push(candidate.message);
    }
  }

  // Always show positive confirmation on top when config is valid
  if (!hasConfigErrors) {
    const confirmation = positiveConfirmation(config, ctx, metrics, results.length > 0);
    if (confirmation) results.unshift(confirmation);
  }

  return results;
}
