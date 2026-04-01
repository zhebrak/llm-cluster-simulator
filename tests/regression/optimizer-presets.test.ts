/**
 * Optimizer Preset Regression Tests
 *
 * Runs the auto-optimizer on all 8 training DEMO_PRESETS and pins MFU
 * ranges. Ensures the optimizer produces sane results (no extreme MFU
 * from degenerate interleaved PP configs, no OOM).
 */

import { describe, it, expect } from 'vitest';
import { optimizeTraining } from '../../src/core/simulation/optimizer.ts';
import { SimulationEngine, type SimulationConfig, type SimulationMetrics } from '../../src/core/simulation/engine.ts';
import { DEMO_PRESETS } from '../../src/stores/config.ts';
import { getModel } from '../../src/core/models/index.ts';
import {
  createMultiNodeCluster,
  createSingleNodeCluster,
  getPresetCluster,
} from '../../src/core/hardware/index.ts';
import type { ClusterConfig } from '../../src/types/index.ts';

// ── Helpers ──────────────────────────────────────────────────────────

function makeCluster(preset: (typeof DEMO_PRESETS)[number]): ClusterConfig | undefined {
  if (preset.clusterId && preset.clusterId !== 'custom') {
    return getPresetCluster(preset.clusterId) ?? undefined;
  }
  const gpuId = preset.gpuId ?? 'h100-sxm';
  const numGPUs = preset.numGPUs ?? 8;
  const gpusPerNode = preset.gpusPerNode ?? 8;
  const numNodes = Math.ceil(numGPUs / gpusPerNode);
  if (numNodes === 1) {
    return createSingleNodeCluster(gpuId, numGPUs);
  }
  return createMultiNodeCluster(gpuId, gpusPerNode, numNodes);
}

function presetToSimConfig(preset: (typeof DEMO_PRESETS)[number]): SimulationConfig {
  const cluster = makeCluster(preset);
  return {
    modelId: preset.modelId,
    clusterConfig: cluster,
    sequenceLength: preset.sequenceLength,
    globalBatchSize: preset.globalBatchSize,
    microBatchSize: preset.microBatchSize,
    strategyType: preset.strategyType,
    strategyConfig: {
      tp: preset.tpDegree,
      pp: preset.ppDegree,
      dp: preset.dpDegree,
      ep: preset.epDegree,
      dpType: preset.dpType,
      sequenceParallel: preset.sequenceParallel,
      pipelineSchedule: preset.pipelineSchedule,
      interleavedStages: preset.interleavedStages,
    },
    activationCheckpointing: preset.activationCheckpointing,
    checkpointingGranularity: preset.checkpointingGranularity,
    flashAttention: preset.flashAttention,
    mixedPrecision: preset.precision,
  };
}

function getMetrics(config: SimulationConfig) {
  const engine = new SimulationEngine();
  engine.configure(config);
  return engine.simulate();
}

function configRow(config: SimulationConfig, metrics: SimulationMetrics) {
  const sc = config.strategyConfig ?? {};
  return {
    strat: config.strategyType,
    TP: sc.tp ?? 1,
    PP: sc.pp ?? 1,
    DP: sc.dp ?? '?',
    EP: sc.ep ?? 1,
    sched: sc.pipelineSchedule ?? '1f1b',
    v: sc.interleavedStages ?? 1,
    SP: sc.sequenceParallel ?? false,
    AC: (config.activationCheckpointing ?? true)
      ? (config.checkpointingGranularity ?? 'full') : 'off',
    MBS: config.microBatchSize,
    GBS: config.globalBatchSize,
    'MFU%': (metrics.mfu * 100).toFixed(1),
    'HFU%': (metrics.hfu * 100).toFixed(1),
    'mem%': (metrics.memoryUtilization * 100).toFixed(1),
    'tok/s': Math.round(metrics.tokensPerSecond),
  };
}

// ── Expected ranges per preset ────────────────────────────────────────
// Tight bands (observed ± 3pp) around optimizer output. The optimizer
// minimizes time-to-train, so higher MFU is a side effect of faster
// training. MFU is computed against BF16 peak (industry convention), so
// FP8 presets can legitimately exceed 50%.
//
// The relative improvement cap (MAX_RELATIVE_IMPROVEMENT) catches physics
// changes that inflate optimizer gains. Per-preset overrides are available
// for models with known large optimizer deltas.

interface PresetExpectation {
  slug: string;
  mfuMin: number;
  mfuMax: number;
  maxRelativeImprovement?: number;  // per-preset override (default MAX_RELATIVE_IMPROVEMENT)
}

const MAX_RELATIVE_IMPROVEMENT = 1.35;

const EXPECTATIONS: PresetExpectation[] = [
  { slug: 'llama3-405b',     mfuMin: 0.38, mfuMax: 0.44 },  // 41.1% — DP capped at 512, optimizer retains PP=16
  { slug: 'nemotron-4-340b', mfuMin: 0.49, mfuMax: 0.55 },  // 52.0%
  { slug: 'gpt3-175b',       mfuMin: 0.54, mfuMax: 0.60, maxRelativeImprovement: 1.40 },  // 57.0% — FA=false baseline has attention HBM penalty; optimizer enables FA, widening gain
  { slug: 'deepseek-v3-r1',  mfuMin: 0.60, mfuMax: 0.67, maxRelativeImprovement: 1.50 },  // 66.2% — drops PP=8→1, MoE baseline MFU is low
  { slug: 'llama4-maverick', mfuMin: 0.79, mfuMax: 0.85 },  // 82.3%
  { slug: 'grok-2.5',        mfuMin: 0.96, mfuMax: 1.02 },  // 99.2%
  { slug: 'qwen3-32b',       mfuMin: 0.44, mfuMax: 0.52 },  // 45.4% — 2048 H800, FSDP-TP
  { slug: 'olmo3-32b',       mfuMin: 0.42, mfuMax: 0.50 },  // 1024 H100, seq=8192
];

// ── Tests ─────────────────────────────────────────────────────────────

describe('Optimizer on all training presets', () => {
  for (const preset of DEMO_PRESETS) {
    const expectation = EXPECTATIONS.find(e => e.slug === preset.slug);

    it(`${preset.modelLabel} (${preset.clusterLabel}): optimizer MFU in sane range`, () => {
      expect(expectation, `Missing expectation for ${preset.slug}`).toBeDefined();

      const config = presetToSimConfig(preset);
      const model = getModel(preset.modelId, preset.sequenceLength);
      expect(model, `Model ${preset.modelId} should exist`).toBeDefined();

      // Baseline MFU
      const baselineMetrics = getMetrics(config);
      expect(baselineMetrics.memoryUtilization, 'Baseline should not OOM').toBeLessThanOrEqual(1.0);

      // Run optimizer
      const targetTokens = preset.targetTokens ?? model!.activeParams * 20;
      const result = optimizeTraining(config, targetTokens, preset.sequenceLength);
      expect(result.success, 'Optimizer should succeed').toBe(true);

      // Optimized MFU
      const optimizedMetrics = getMetrics(result.optimizedConfig);
      expect(optimizedMetrics.memoryUtilization, 'Optimized should not OOM').toBeLessThanOrEqual(1.0);

      // ── Diagnostic table ──
      console.log(`\n── ${preset.modelLabel} (${preset.clusterLabel}) ──`);
      console.table([
        { ' ': 'baseline', ...configRow(config, baselineMetrics) },
        { ' ': 'optimized', ...configRow(result.optimizedConfig, optimizedMetrics) },
      ]);
      if (result.changelog.length > 0) {
        console.log('  Changes:', result.changelog.map(c => `${c.field}: ${c.from} → ${c.to}`).join(', '));
      }

      // ── Metrics are finite ──
      expect(Number.isFinite(optimizedMetrics.mfu)).toBe(true);
      expect(Number.isFinite(optimizedMetrics.stepTimeMs)).toBe(true);

      // ── MFU in expected range ──
      expect(
        optimizedMetrics.mfu,
        `Optimized MFU ${(optimizedMetrics.mfu * 100).toFixed(1)}% should be ≥ ${(expectation!.mfuMin * 100).toFixed(0)}%`,
      ).toBeGreaterThanOrEqual(expectation!.mfuMin);
      expect(
        optimizedMetrics.mfu,
        `Optimized MFU ${(optimizedMetrics.mfu * 100).toFixed(1)}% should be ≤ ${(expectation!.mfuMax * 100).toFixed(0)}%`,
      ).toBeLessThanOrEqual(expectation!.mfuMax);

      // ── No regression — optimized MFU ≥ baseline ──
      expect(optimizedMetrics.mfu).toBeGreaterThanOrEqual(baselineMetrics.mfu);

      // ── Time-to-train improves (or stays equal) ──
      expect(result.afterMetric).toBeLessThanOrEqual(result.beforeMetric);

      // ── Relative improvement cap ──
      // Catches physics changes that inflate optimizer gains beyond plausible limits.
      const relativeImprovement = optimizedMetrics.mfu / baselineMetrics.mfu;
      const maxRelative = expectation!.maxRelativeImprovement ?? MAX_RELATIVE_IMPROVEMENT;
      expect(
        relativeImprovement,
        `Relative MFU improvement ${relativeImprovement.toFixed(2)}x exceeds ${maxRelative}x cap`,
      ).toBeLessThanOrEqual(maxRelative);

      // ── Phase sims sum correctly ──
      // totalSimulations includes a final baseline re-sim not attributed to any phase
      const phaseSum = result.phases.fix + result.phases.greedy + result.phases.explore;
      expect(phaseSum).toBeGreaterThanOrEqual(result.totalSimulations - 1);
      expect(phaseSum).toBeLessThanOrEqual(result.totalSimulations);
    });
  }
});
