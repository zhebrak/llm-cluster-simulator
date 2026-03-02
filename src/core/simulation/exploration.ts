/**
 * Training configuration exploration grid
 *
 * Generates candidate SimulationConfigs from a constrained Cartesian product.
 * Used by the optimizer's Phase 3 (explore) to find globally better configs
 * that greedy single-mutation search can't reach.
 */
// See docs/OPTIMIZER.md for exploration grid search details.

import type { SimulationConfig } from './engine.ts';
import type { ModelSpec } from '../../types/model.ts';
import type { ClusterConfig } from '../../types/index.ts';
import { gpuSupportsPrecision } from '../hardware/index.ts';

export const MAX_EXPLORE_SIMS = 5000;
export const MAX_EXPLORE_SIMS_MOE = 10000;

/** Optimizer cap: DP groups >512 are unvalidated by published benchmarks. */
export const MAX_OPTIMIZER_DP = 512;

/** Optimizer cap: no published pipeline config uses >64 micro-batches. */
export const MAX_PIPELINE_MICROBATCHES = 64;

// ── GBS convergence ceiling ──────────────────────────────────────────
// Critical batch size scaling (McCandlish et al. 2018): B_crit ∝ sqrt(params).
// Beyond this, gradient noise dominates and larger batches waste compute.
const B_CRIT_COEFFICIENT = 20;          // tokens / sqrt(param)
const B_CRIT_FLOOR_TOKENS = 4_000_000;  // 4M minimum

/** Maximum useful global batch size in tokens, from critical batch size scaling. */
export function getCriticalBatchTokens(totalParams: number): number {
  return Math.max(B_CRIT_FLOOR_TOKENS, B_CRIT_COEFFICIENT * Math.sqrt(totalParams));
}

type StrategyType = SimulationConfig['strategyType'];

const ONE_D: StrategyType[] = ['ddp', 'fsdp', 'zero-1'];
const TWO_D: StrategyType[] = ['fsdp-tp', 'zero1-tp'];
const THREE_D: StrategyType[] = ['ddp-tp-pp', 'zero1-tp-pp', 'fsdp-tp-pp'];
const ALL_STRATEGIES: StrategyType[] = [...ONE_D, ...TWO_D, ...THREE_D];

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

/** In-place Fisher-Yates shuffle using seeded PRNG. */
function shuffle<T>(arr: T[], random: () => number): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

/**
 * Generate candidate training configs from a constrained Cartesian product.
 * Returns at most MAX_EXPLORE_SIMS (or MAX_EXPLORE_SIMS_MOE for MoE) candidates,
 * using stratified sampling by (strategy, AC, EP) when the grid exceeds the budget.
 */
export function generateTrainingCandidates(
  baseConfig: SimulationConfig,
  model: ModelSpec,
  cluster: ClusterConfig,
  seed?: number,
): SimulationConfig[] {
  const totalGPUs = cluster.totalGPUs;
  const gpusPerNode = cluster.gpusPerNode;
  const numLayers = model.numLayers;
  const numHeads = model.numAttentionHeads;
  const isMoE = model.isMoE ?? false;
  const numExperts = model.numExperts ?? 0;

  const currentGBS = baseConfig.globalBatchSize;
  const currentPrecision = baseConfig.mixedPrecision ?? 'bf16';
  const seqLength = baseConfig.sequenceLength;

  // Build dimension arrays
  const tpDegrees = [1, 2, 4, 8].filter(tp => tp <= gpusPerNode && numHeads % tp === 0);
  const ppDegrees = [1, 2, 4, 8, 16].filter(pp => pp <= numLayers);
  const epDegrees = isMoE && numExperts >= 2
    ? [1, 2, 4, 8, 16, 32].filter(ep => ep === 1 || (numExperts % ep === 0))
    : [1];
  const cpDegrees = seqLength >= 8192
    ? [1, 2, 4, 8].filter(cp => seqLength / cp >= 8192)
    : [1];
  // Ceiling: max of current GBS and scaling-law ceiling.
  // If user already chose a GBS above the scaling law, trust their judgment —
  // but don't inflate further. If user is below, allow up to the ceiling.
  const ceilingGBS = Math.floor(getCriticalBatchTokens(model.totalParams) / seqLength);
  const maxGBS = Math.max(currentGBS, ceilingGBS);
  const gbsValues = [
    Math.max(64, Math.floor(currentGBS / 4)),
    Math.max(64, Math.floor(currentGBS / 2)),
    currentGBS,
    Math.min(maxGBS, currentGBS * 2),
    Math.min(maxGBS, currentGBS * 4),
  ].filter((v, i, a) => a.indexOf(v) === i); // dedupe
  const mbsValues = [1, 2, 4, 8];
  const acOptions: Array<{ ac: boolean; acg: 'full' | 'selective' }> = [
    { ac: true, acg: 'full' },
    { ac: true, acg: 'selective' },
  ];
  // Never switch to FP8 during auto-optimize — FP8 affects MFU calculation
  // and should be an explicit user choice. But do allow downgrading from fp32
  // to bf16/fp16, which is always beneficial for training.
  const gpu = cluster.node.gpu;
  const precisionValues: Array<SimulationConfig['mixedPrecision']> = currentPrecision === 'fp32'
    ? (['bf16', 'fp16'] as const).filter(p => gpuSupportsPrecision(gpu, p))
    : [currentPrecision];

  const candidates: SimulationConfig[] = [];

  for (const strategy of ALL_STRATEGIES) {
    for (const tp of tpDegrees) {
      for (const pp of ppDegrees) {
        // Constraint: match strategy family
        const is1D = ONE_D.includes(strategy);
        const is2D = TWO_D.includes(strategy);
        const is3D = THREE_D.includes(strategy);

        if (is1D && (tp > 1 || pp > 1)) continue;
        if (is2D && (tp <= 1 || pp > 1)) continue;
        if (is3D && pp <= 1 && tp <= 1) continue;
        // 3D allows tp=1,pp>1 (ddp-tp-pp) but requires pp>1 or tp>1
        if (is3D && pp <= 1 && tp <= 1) continue;

        for (const cp of cpDegrees) {
          // tp × pp × cp must leave room for DP
          if (tp * pp * cp > totalGPUs) continue;
          const dp = Math.floor(totalGPUs / (tp * pp * cp));
          if (dp < 1) continue;
          if (dp > MAX_OPTIMIZER_DP) continue;

          // 1D strategies with CP > 1 route through 3D internally, which is fine,
          // but skip CP > 1 for strategies that don't benefit (only if seq is short)
          if (is1D && cp > 1 && (tp > 1 || pp > 1)) continue;

          for (const ep of epDegrees) {
            // EP must divide DP
            if (ep > 1 && dp % ep !== 0) continue;
            // EP only for MoE
            if (ep > 1 && !isMoE) continue;
            // EP must fit with TP in a node
            if (ep > 1 && tp * ep > totalGPUs) continue;

            for (const rawGBS of gbsValues) {
              for (const mbs of mbsValues) {
                // Snap GBS to nearest multiple of MBS × DP so GA is exact
                const unit = mbs * dp;
                const gbs = Math.max(unit, Math.floor(rawGBS / unit) * unit);
                const ga = gbs / unit;
                if (pp > 1 && ga > MAX_PIPELINE_MICROBATCHES) continue;
                if (ga > 256) continue;

                for (const { ac, acg } of acOptions) {
                  for (const precision of precisionValues) {
                    const sp = !is1D; // SP on for hybrid/3D

                    // Standard 1f1b candidate
                    candidates.push({
                      ...baseConfig,
                      globalBatchSize: gbs,
                      microBatchSize: mbs,
                      gradientAccumulationSteps: undefined,
                      strategyType: strategy,
                      strategyConfig: {
                        tp,
                        pp,
                        dp,
                        ep: isMoE ? ep : undefined,
                        cp,
                        sequenceParallel: sp,
                        pipelineSchedule: '1f1b' as const,
                      },
                      activationCheckpointing: ac,
                      checkpointingGranularity: acg,
                      mixedPrecision: precision,
                    });

                    // Interleaved-1f1b variants for PP>1
                    if (pp > 1) {
                      for (const v of [2, 3, 4, 6, 8]) {
                        if (numLayers % (pp * v) === 0 && numLayers / (pp * v) >= 2) {
                          candidates.push({
                            ...baseConfig,
                            globalBatchSize: gbs,
                            microBatchSize: mbs,
                            gradientAccumulationSteps: undefined,
                            strategyType: strategy,
                            strategyConfig: {
                              tp,
                              pp,
                              dp,
                              ep: isMoE ? ep : undefined,
                              cp,
                              sequenceParallel: sp,
                              pipelineSchedule: 'interleaved-1f1b' as const,
                              interleavedStages: v,
                            },
                            activationCheckpointing: ac,
                            checkpointingGranularity: acg,
                            mixedPrecision: precision,
                          });
                        }
                      }

                      // DualPipeV for PP>1 when GA >= 2*PP (non-degraded mode)
                      if (ga >= 2 * pp) {
                        candidates.push({
                          ...baseConfig,
                          globalBatchSize: gbs,
                          microBatchSize: mbs,
                          gradientAccumulationSteps: undefined,
                          strategyType: strategy,
                          strategyConfig: {
                            tp,
                            pp,
                            dp,
                            ep: isMoE ? ep : undefined,
                            cp,
                            sequenceParallel: sp,
                            pipelineSchedule: 'dualpipe-v' as const,
                          },
                          activationCheckpointing: ac,
                          checkpointingGranularity: acg,
                          mixedPrecision: precision,
                        });
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // EP dimension inflates the MoE grid ~6×; use a larger budget to compensate
  const budget = isMoE ? MAX_EXPLORE_SIMS_MOE : MAX_EXPLORE_SIMS;

  if (candidates.length <= budget) return candidates;

  // ── Stratified sampling by (strategy, AC granularity, EP) ──────────
  // Naive uniform random drops rare-but-important corners (e.g. a single
  // optimal {zero1-tp-pp, selective, EP=8} config out of ~50K candidates).
  // Stratify on the dimensions that most strongly differentiate MFU, then
  // sample diversely within each stratum across TP/PP/MBS/pipeline.
  const random = createSeededRandom(seed ?? 42);

  const buckets = new Map<string, SimulationConfig[]>();
  for (const c of candidates) {
    const key = `${c.strategyType}|${c.checkpointingGranularity}|${c.strategyConfig?.ep ?? 1}`;
    let bucket = buckets.get(key);
    if (!bucket) { bucket = []; buckets.set(key, bucket); }
    bucket.push(c);
  }

  // Guarantee coverage: allocate 70% of budget evenly across strata,
  // 30% proportional to stratum size (larger strata get more representation)
  const guaranteedBudget = Math.floor(budget * 0.7);
  const perTriple = Math.max(2, Math.floor(guaranteedBudget / buckets.size));

  const sampled: SimulationConfig[] = [];
  const overflow: SimulationConfig[] = [];

  for (const [, bucket] of buckets) {
    shuffle(bucket, random);
    const take = Math.min(perTriple, bucket.length);
    sampled.push(...bucket.slice(0, take));
    if (bucket.length > take) {
      overflow.push(...bucket.slice(take));
    }
  }

  // Fill remaining budget proportionally from unsampled candidates
  const remaining = budget - sampled.length;
  if (remaining > 0 && overflow.length > 0) {
    shuffle(overflow, random);
    sampled.push(...overflow.slice(0, Math.min(remaining, overflow.length)));
  }

  return sampled;
}
