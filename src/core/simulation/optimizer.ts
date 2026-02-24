/**
 * Training Auto-Optimizer
 *
 * Pure function: takes a SimulationConfig + training target, returns an
 * OptimizationResult with the best config found.
 *
 * Three phases:
 *   1. Fix — resolve OOM (max 10 iterations)
 *   2. Greedy — iteratively apply best single-mutation improvement (max 200 iterations)
 *   3. Explore — brute-force grid search over the config space
 */

import { SimulationEngine, type SimulationConfig, type SimulationMetrics } from './engine.ts';
import { GENERATORS } from './recommendations.ts';
import { generateTrainingCandidates, getCriticalBatchTokens, MAX_OPTIMIZER_DP, MAX_PIPELINE_MICROBATCHES } from './exploration.ts';
import type { ModelSpec } from '../../types/model.ts';
import type { ClusterConfig } from '../../types/index.ts';

// ── Result types ─────────────────────────────────────────────────────

export interface ChangelogEntry {
  field: string;
  from: string;
  to: string;
}

export interface OptimizationResult {
  success: boolean;
  originalConfig: SimulationConfig;
  optimizedConfig: SimulationConfig;
  changelog: ChangelogEntry[];
  beforeMetric: number;
  afterMetric: number;
  totalSimulations: number;
  phases: { fix: number; greedy: number; explore: number };
}

// ── Helpers ──────────────────────────────────────────────────────────

/**
 * Scale-aware memory safety threshold.
 *
 * Larger multi-rack clusters face higher OOM risk due to NCCL memory
 * fragmentation, straggler GPU memory spikes, and reduced per-node
 * memory headroom from collective buffers. We use a tighter threshold
 * for ≥256 GPUs (typically 32+ nodes / multi-rack).
 */
function maxSafeMemoryUtil(config: SimulationConfig): number {
  const totalGPUs = config.clusterConfig?.totalGPUs ?? 8;
  return totalGPUs >= 256 ? 0.85 : 0.87;
}

function simulate(config: SimulationConfig): SimulationMetrics | null {
  try {
    const engine = new SimulationEngine();
    engine.configure({ ...config, gradientAccumulationSteps: undefined });
    return engine.simulate();
  } catch {
    return null;
  }
}

function computeTimeToTrain(
  metrics: SimulationMetrics,
  config: SimulationConfig,
  targetTokens: number,
  seqLength: number,
): number {
  const tokensPerStep = config.globalBatchSize * seqLength;
  const baseSteps = Math.ceil(targetTokens / tokensPerStep);
  return metrics.stepTimeMs * baseSteps / 3.6e6; // hours
}

function buildChangelog(original: SimulationConfig, optimized: SimulationConfig): ChangelogEntry[] {
  const entries: ChangelogEntry[] = [];

  const check = (field: string, from: unknown, to: unknown) => {
    if (JSON.stringify(from) !== JSON.stringify(to)) {
      entries.push({ field, from: String(from), to: String(to) });
    }
  };

  check('strategyType', original.strategyType, optimized.strategyType);
  check('globalBatchSize', original.globalBatchSize, optimized.globalBatchSize);
  check('microBatchSize', original.microBatchSize, optimized.microBatchSize);
  check('mixedPrecision', original.mixedPrecision ?? 'bf16', optimized.mixedPrecision ?? 'bf16');
  // Combine AC + granularity into one change entry: "off" / "selective" / "full"
  const acLabel = (ac: boolean | undefined, acg: string | undefined) =>
    (ac ?? true) ? (acg ?? 'full') : 'off';
  check('activationCheckpointing',
    acLabel(original.activationCheckpointing, original.checkpointingGranularity),
    acLabel(optimized.activationCheckpointing, optimized.checkpointingGranularity));
  check('flashAttention', original.flashAttention ?? true, optimized.flashAttention ?? true);

  const osc = original.strategyConfig ?? {};
  const nsc = optimized.strategyConfig ?? {};
  check('tp', osc.tp ?? 1, nsc.tp ?? 1);
  check('pp', osc.pp ?? 1, nsc.pp ?? 1);
  check('ep', osc.ep ?? 1, nsc.ep ?? 1);
  check('cp', osc.cp ?? 1, nsc.cp ?? 1);
  check('sequenceParallel', osc.sequenceParallel ?? false, nsc.sequenceParallel ?? false);
  check('pipelineSchedule', osc.pipelineSchedule ?? '1f1b', nsc.pipelineSchedule ?? '1f1b');
  if (nsc.pipelineSchedule === 'interleaved-1f1b') {
    check('interleavedStages', osc.interleavedStages ?? 1, nsc.interleavedStages ?? 1);
  }

  return entries;
}

/** Effective data-parallel degree from a config's parallelism dimensions. */
function getEffectiveDPFromConfig(config: SimulationConfig): number {
  const sc = config.strategyConfig ?? {};
  const tp = sc.tp ?? 1;
  const pp = sc.pp ?? 1;
  const cp = sc.cp ?? 1;
  const totalGPUs = config.clusterConfig?.totalGPUs ?? 1;
  return Math.max(1, Math.floor(totalGPUs / (tp * pp * cp)));
}

/** Check whether a candidate config exceeds optimizer caps (DP or pipeline micro-batches). */
function exceedsOptimizerCaps(config: SimulationConfig): boolean {
  const dp = getEffectiveDPFromConfig(config);
  if (dp > MAX_OPTIMIZER_DP) return true;
  const pp = config.strategyConfig?.pp ?? 1;
  if (pp > 1) {
    const ga = Math.ceil(config.globalBatchSize / (config.microBatchSize * dp));
    if (ga > MAX_PIPELINE_MICROBATCHES) return true;
  }
  return false;
}

// ── Main optimizer ───────────────────────────────────────────────────

export function optimizeTraining(
  config: SimulationConfig,
  targetTokens: number,
  seqLength: number,
): OptimizationResult {
  const originalConfig = { ...config };
  let currentConfig = { ...config };
  let totalSims = 0;
  const phaseSims = { fix: 0, greedy: 0, explore: 0 };

  let currentMetrics = simulate(currentConfig);
  totalSims++;
  phaseSims.fix++;

  // If initial config is invalid (e.g. TP > totalGPUs after loading inference preset),
  // fall back to safe 1D FSDP config before giving up
  if (!currentMetrics) {
    const totalGPUs = currentConfig.clusterConfig?.totalGPUs ?? 1;
    currentConfig = {
      ...currentConfig,
      strategyType: 'fsdp',
      strategyConfig: { tp: 1, pp: 1, dp: totalGPUs, ep: 1, cp: 1 },
    };
    currentMetrics = simulate(currentConfig);
    totalSims++;
    phaseSims.fix++;
  }

  if (!currentMetrics) {
    return {
      success: false,
      originalConfig,
      optimizedConfig: currentConfig,
      changelog: [],
      beforeMetric: Infinity,
      afterMetric: Infinity,
      totalSimulations: totalSims,
      phases: phaseSims,
    };
  }

  // ── Phase 1: Fix OOM ───────────────────────────────────────────────
  // Accept any mutation that reduces memory, even if still OOM.
  // Multiple iterations compound (e.g. DDP→FSDP→FSDP-TP).

  for (let i = 0; i < 10 && currentMetrics.memoryUtilization > 1.0; i++) {
    let bestMutated: SimulationConfig | null = null;
    let bestMetrics: SimulationMetrics | null = null;
    let bestMemUtil = currentMetrics.memoryUtilization;

    const ctx = buildContext(currentConfig);
    for (const gen of GENERATORS) {
      const candidate = gen(currentConfig, ctx, currentMetrics);
      if (!candidate?.configMutation) continue;

      const mutated = {
        ...candidate.configMutation(currentConfig),
        gradientAccumulationSteps: undefined,
      };
      if (exceedsOptimizerCaps(mutated)) continue;
      const mutatedMetrics = simulate(mutated);
      totalSims++;
      phaseSims.fix++;

      if (mutatedMetrics && mutatedMetrics.memoryUtilization < bestMemUtil) {
        bestMemUtil = mutatedMetrics.memoryUtilization;
        bestMutated = mutated;
        bestMetrics = mutatedMetrics;
      }
    }

    if (!bestMutated || !bestMetrics) break;
    currentConfig = bestMutated;
    currentMetrics = bestMetrics;
  }

  // ── GBS convergence ceiling ────────────────────────────────────────
  // Resolve model early so all phases can enforce the critical batch size cap.
  const model = resolveModel(currentConfig);
  const maxGBSTokens = model ? getCriticalBatchTokens(model.totalParams) : Infinity;
  const maxGBS = Math.max(currentConfig.globalBatchSize, Math.floor(maxGBSTokens / seqLength));

  // ── Phase 2: Greedy improvement ────────────────────────────────────

  let currentTime = currentMetrics.memoryUtilization <= 1.0
    ? computeTimeToTrain(currentMetrics, currentConfig, targetTokens, seqLength)
    : Infinity;

  for (let iter = 0; iter < 200; iter++) {
    if (currentMetrics.memoryUtilization > 1.0) break;

    let bestMutation: SimulationConfig | null = null;
    let bestTime = currentTime;
    const ctx = buildContext(currentConfig);

    for (const gen of GENERATORS) {
      const candidate = gen(currentConfig, ctx, currentMetrics);
      if (!candidate?.configMutation) continue;

      const mutated = {
        ...candidate.configMutation(currentConfig),
        gradientAccumulationSteps: undefined,
      };
      if (exceedsOptimizerCaps(mutated)) continue;
      const mutatedMetrics = simulate(mutated);
      totalSims++;
      phaseSims.greedy++;

      // Reject configs above safe memory threshold — they'll OOM in practice
      if (!mutatedMetrics || mutatedMetrics.memoryUtilization > maxSafeMemoryUtil(mutated)) continue;
      // Reject GBS above convergence ceiling
      if (mutated.globalBatchSize > maxGBS) continue;

      const mutatedTime = computeTimeToTrain(mutatedMetrics, mutated, targetTokens, seqLength);
      if (mutatedTime < bestTime * 0.995) { // ≥0.5% improvement
        bestTime = mutatedTime;
        bestMutation = mutated;
      }
    }

    if (!bestMutation) break;

    currentConfig = bestMutation;
    currentMetrics = simulate(currentConfig)!;
    totalSims++;
    phaseSims.greedy++;
    currentTime = bestTime;
  }

  // ── Snap GBS to DP alignment after greedy mutations ────────────────
  {
    const greedyUnit = currentConfig.microBatchSize * getEffectiveDPFromConfig(currentConfig);
    if (currentConfig.globalBatchSize % greedyUnit !== 0) {
      currentConfig = {
        ...currentConfig,
        globalBatchSize: Math.max(greedyUnit, Math.floor(currentConfig.globalBatchSize / greedyUnit) * greedyUnit),
        gradientAccumulationSteps: undefined,
      };
      currentMetrics = simulate(currentConfig)!;
      currentTime = computeTimeToTrain(currentMetrics, currentConfig, targetTokens, seqLength);
    }
  }

  // ── Phase 3: Explore grid ──────────────────────────────────────────

  // Resolve cluster for exploration grid (model already resolved above for GBS ceiling)
  const cluster = resolveCluster(currentConfig);

  if (model && cluster) {
    const candidates = generateTrainingCandidates(currentConfig, model, cluster);

    // Track global best across all candidates rather than progressive greedy.
    // Progressive greedy raises the bar with each improvement, which can reject
    // a much better config found later that doesn't clear the 0.5% delta over
    // the latest intermediate winner. Instead, find the absolute best, then
    // compare once against the Phase 2 baseline.
    let exploreBestTime = Infinity;
    let exploreBestMemUtil = 0;
    let exploreBestConfig: SimulationConfig | null = null;
    let exploreBestMetrics: SimulationMetrics | null = null;

    for (const candidate of candidates) {
      const metrics = simulate(candidate);
      totalSims++;
      phaseSims.explore++;

      // Reject configs above safe memory threshold — they'll OOM in practice
      if (!metrics || metrics.memoryUtilization > maxSafeMemoryUtil(candidate)) continue;
      // Reject GBS above convergence ceiling
      if (candidate.globalBatchSize > maxGBS) continue;

      const time = computeTimeToTrain(metrics, candidate, targetTokens, seqLength);

      // Primary: faster training time. Secondary: prefer higher memory utilization
      // when within 1% — higher util means better resource usage, less over-sharding
      // or under-batching.
      const dominated = time < exploreBestTime * 0.99
        || (time < exploreBestTime * 1.01 && metrics.memoryUtilization > exploreBestMemUtil);
      if (dominated) {
        exploreBestTime = time;
        exploreBestMemUtil = metrics.memoryUtilization;
        exploreBestConfig = candidate;
        exploreBestMetrics = metrics;
      }
    }

    // Require ≥0.5% improvement over Phase 2 baseline to accept
    if (exploreBestConfig && exploreBestMetrics && exploreBestTime < currentTime * 0.995) {
      currentTime = exploreBestTime;
      currentConfig = exploreBestConfig;
      currentMetrics = exploreBestMetrics;
    }
  }

  // ── Run Phase 2 again on the best explored config ──────────────────
  // One more greedy pass to locally optimize the explored config

  for (let iter = 0; iter < 50; iter++) {
    if (!currentMetrics || currentMetrics.memoryUtilization > 1.0) break;

    let bestMutation: SimulationConfig | null = null;
    let bestTime = currentTime;
    const ctx = buildContext(currentConfig);

    for (const gen of GENERATORS) {
      const candidate = gen(currentConfig, ctx, currentMetrics);
      if (!candidate?.configMutation) continue;

      const mutated = {
        ...candidate.configMutation(currentConfig),
        gradientAccumulationSteps: undefined,
      };
      if (exceedsOptimizerCaps(mutated)) continue;
      const mutatedMetrics = simulate(mutated);
      totalSims++;
      phaseSims.greedy++;

      // Reject configs above safe memory threshold — they'll OOM in practice
      if (!mutatedMetrics || mutatedMetrics.memoryUtilization > maxSafeMemoryUtil(mutated)) continue;
      // Reject GBS above convergence ceiling
      if (mutated.globalBatchSize > maxGBS) continue;

      const mutatedTime = computeTimeToTrain(mutatedMetrics, mutated, targetTokens, seqLength);
      if (mutatedTime < bestTime * 0.995) {
        bestTime = mutatedTime;
        bestMutation = mutated;
      }
    }

    if (!bestMutation) break;

    currentConfig = bestMutation;
    currentMetrics = simulate(currentConfig)!;
    totalSims++;
    phaseSims.greedy++;
    currentTime = bestTime;
  }

  // ── Final GBS alignment snap ───────────────────────────────────────
  {
    const finalUnit = currentConfig.microBatchSize * getEffectiveDPFromConfig(currentConfig);
    if (currentConfig.globalBatchSize % finalUnit !== 0) {
      currentConfig = {
        ...currentConfig,
        globalBatchSize: Math.max(finalUnit, Math.floor(currentConfig.globalBatchSize / finalUnit) * finalUnit),
        gradientAccumulationSteps: undefined,
      };
      const snappedMetrics = simulate(currentConfig);
      if (snappedMetrics) {
        currentMetrics = snappedMetrics;
        currentTime = computeTimeToTrain(currentMetrics, currentConfig, targetTokens, seqLength);
      }
    }
  }

  // ── Build result ───────────────────────────────────────────────────

  const originalMetrics = simulate(originalConfig);
  totalSims++;
  const beforeTime = originalMetrics && originalMetrics.memoryUtilization <= 1.0
    ? computeTimeToTrain(originalMetrics, originalConfig, targetTokens, seqLength)
    : Infinity;

  return {
    success: currentMetrics !== null && currentMetrics.memoryUtilization <= 1.0,
    originalConfig,
    optimizedConfig: currentConfig,
    changelog: buildChangelog(originalConfig, currentConfig),
    beforeMetric: beforeTime,
    afterMetric: currentTime,
    totalSimulations: totalSims,
    phases: phaseSims,
  };
}

// ── Private helpers to build strategy context / resolve model+cluster ─

import type { StrategyContext } from '../strategies/base.ts';
import { getModel } from '../models/index.ts';
import { getPresetCluster } from '../hardware/index.ts';
import { DTYPE_PRESETS, DEFAULT_ADAMW_CONFIG, DEFAULT_LR_SCHEDULE } from '../../types/index.ts';
import { supportsFlashAttention } from '../hardware/gpu.ts';

function resolveModel(config: SimulationConfig): ModelSpec | null {
  if (config.modelSpec) return config.modelSpec;
  if (config.modelId) return getModel(config.modelId, config.sequenceLength) ?? null;
  return null;
}

function resolveCluster(config: SimulationConfig): ClusterConfig | null {
  if (config.clusterConfig) return config.clusterConfig;
  if (config.clusterId) return getPresetCluster(config.clusterId) ?? null;
  return null;
}

function buildContext(config: SimulationConfig): StrategyContext {
  const engine = new SimulationEngine();
  engine.configure({ ...config, gradientAccumulationSteps: undefined });
  // We need to build context — use the engine's internal method indirectly
  // by simulating and extracting what we need
  const model = resolveModel(config);
  const cluster = resolveCluster(config);

  if (!model || !cluster) {
    throw new Error('Cannot resolve model or cluster');
  }

  // Replicate engine's getEffectiveParallelism logic
  const sc = config.strategyConfig ?? {};
  const gpusPerNode = cluster.gpusPerNode;
  const totalGPUs = cluster.totalGPUs;

  let tp: number, pp: number;
  switch (config.strategyType) {
    case 'fsdp-tp':
    case 'zero1-tp':
      tp = sc.tp ?? Math.min(8, gpusPerNode);
      pp = 1;
      break;
    case 'ddp-tp-pp':
    case 'zero1-tp-pp':
      tp = sc.tp ?? Math.min(8, gpusPerNode);
      pp = sc.pp ?? Math.min(8, Math.floor(totalGPUs / tp));
      break;
    case 'fsdp-tp-pp':
      tp = sc.tp ?? Math.min(8, gpusPerNode);
      pp = sc.pp ?? Math.min(4, Math.floor(totalGPUs / tp));
      break;
    default:
      tp = sc.tp ?? 1;
      pp = sc.pp ?? 1;
      break;
  }

  const cp = sc.cp ?? 1;
  const effectiveDP = Math.max(1, Math.floor(totalGPUs / (tp * pp * cp)));
  const ga = Math.ceil(config.globalBatchSize / (config.microBatchSize * effectiveDP));

  const dtypePreset = config.mixedPrecision ?? 'bf16';
  const dtypes = DTYPE_PRESETS[dtypePreset] ?? DTYPE_PRESETS.bf16;

  return {
    model,
    cluster,
    training: {
      globalBatchSize: config.globalBatchSize,
      microBatchSize: config.microBatchSize,
      sequenceLength: config.sequenceLength,
      maxSteps: config.maxSteps ?? 1000,
      optimizer: DEFAULT_ADAMW_CONFIG,
      lrSchedule: DEFAULT_LR_SCHEDULE,
      dtypes,
      gradientClipping: 1.0,
      gradientAccumulationSteps: ga,
    },
    seqLength: config.sequenceLength,
    microBatchSize: config.microBatchSize,
    globalBatchSize: config.globalBatchSize,
    gradientAccumulationSteps: ga,
    activationCheckpointing: config.activationCheckpointing ?? true,
    checkpointingGranularity: config.checkpointingGranularity ?? 'full',
    flashAttention: (config.flashAttention ?? true) && supportsFlashAttention(cluster.node.gpu),
  };
}
