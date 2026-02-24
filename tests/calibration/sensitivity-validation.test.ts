/**
 * Sensitivity validation: perturb each model parameter and verify simulation
 * outputs move in physically expected directions.
 *
 * Covers all 8 training presets and 6 inference presets × core/GQA/MoE/MLA
 * parameter perturbations ≈ 140 test cases.
 */

import { describe, it, expect } from 'vitest';
import { getModelConfig } from '../../src/core/models/architectures.ts';
import { buildModelSpec } from '../../src/core/models/primitives.ts';
import { getSimulationMetrics, type SimulationConfig } from '../../src/core/simulation/engine.ts';
import { runInferenceSimulation, type InferenceSimulationConfig } from '../../src/core/inference/simulation.ts';
import { toSimConfig, PUBLISHED } from '../../src/data/published-training-configs.ts';
import { createMultiNodeCluster, createSingleNodeCluster } from '../../src/core/hardware/topology.ts';
import type { ModelConfig } from '../../src/types/index.ts';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Deep-clone a ModelConfig and apply a single field override. */
function perturb(config: ModelConfig, overrides: Partial<ModelConfig>): ModelConfig {
  return { ...config, ...overrides };
}

/** Aligned hiddenSize delta: must stay divisible by numAttentionHeads. */
function alignedHiddenDelta(config: ModelConfig, target: number = 512): number {
  return Math.ceil(target / config.numAttentionHeads) * config.numAttentionHeads;
}

// ---------------------------------------------------------------------------
// Perturbation definitions
// ---------------------------------------------------------------------------

type Direction = 'up' | 'ge' | 'eq' | 'eq_pct';

interface Perturbation {
  name: string;
  /** Which models this perturbation applies to (ModelSpec physics). */
  appliesTo: (c: ModelConfig) => boolean;
  /** Stricter filter for simulation tests — defaults to appliesTo. */
  affectsSimulation?: (c: ModelConfig) => boolean;
  /** Return modified config (or undefined to skip). */
  apply: (c: ModelConfig) => ModelConfig | undefined;
  /** Expected directions for ModelSpec fields. */
  spec: {
    totalParams: Direction;
    activeParams: Direction;
    flopsPerToken: Direction;
  };
  /** Expected direction for training memory. */
  trainingMemory: Direction;
  /** Expected direction for training stepTime (skip if either OOMs). */
  trainingStepTime?: Direction;
  /** Expected direction for inference memory. */
  inferenceMemory: Direction;
  /** Expected directions for inference latency. */
  inferenceTTFT: Direction;
  inferenceTPOT: Direction;
}

/** True if a MoE model has some dense layers (intermediateSize affects sim). */
function hasDenseLayers(c: ModelConfig): boolean {
  if ((c.numExperts ?? 0) === 0) return true; // fully dense model
  return (c.firstKDenseLayers ?? 0) > 0 || (c.moeLayerFrequency ?? 1) > 1;
}

const PERTURBATIONS: Perturbation[] = [
  // --- Core parameters (all models) ---
  {
    name: 'numLayers +4',
    appliesTo: () => true,
    apply: c => perturb(c, { numLayers: c.numLayers + 4 }),
    spec: { totalParams: 'up', activeParams: 'up', flopsPerToken: 'up' },
    trainingMemory: 'up',
    trainingStepTime: 'up',
    inferenceMemory: 'up',
    inferenceTTFT: 'up',
    inferenceTPOT: 'up',
  },
  {
    name: 'hiddenSize +aligned(512)',
    appliesTo: () => true,
    apply: c => {
      const delta = alignedHiddenDelta(c);
      return perturb(c, { hiddenSize: c.hiddenSize + delta });
    },
    spec: { totalParams: 'up', activeParams: 'up', flopsPerToken: 'up' },
    trainingMemory: 'up',
    trainingStepTime: 'up',
    inferenceMemory: 'up',
    inferenceTTFT: 'up',
    inferenceTPOT: 'up',
  },
  {
    name: 'intermediateSize +1024',
    // For fully-MoE models (no dense layers), intermediateSize is unused
    appliesTo: hasDenseLayers,
    apply: c => perturb(c, { intermediateSize: c.intermediateSize + 1024 }),
    spec: { totalParams: 'up', activeParams: 'up', flopsPerToken: 'up' },
    trainingMemory: 'up',
    trainingStepTime: 'up',
    inferenceMemory: 'up',
    inferenceTTFT: 'up',
    inferenceTPOT: 'up',
  },
  {
    name: 'vocabSize +10000',
    appliesTo: () => true,
    apply: c => perturb(c, { vocabSize: c.vocabSize + 10000 }),
    spec: { totalParams: 'up', activeParams: 'up', flopsPerToken: 'ge' },
    trainingMemory: 'up',
    inferenceMemory: 'up',
    inferenceTTFT: 'ge',
    inferenceTPOT: 'eq_pct', // vocab adds <1% to decode weight reads
  },

  // --- GQA-only ---
  {
    name: 'numKvHeads ×2',
    appliesTo: c => (c.numKvHeads ?? c.numAttentionHeads) < c.numAttentionHeads,
    apply: c => {
      const current = c.numKvHeads ?? c.numAttentionHeads;
      const doubled = current * 2;
      if (doubled > c.numAttentionHeads) return undefined; // can't exceed numHeads
      return perturb(c, { numKvHeads: doubled });
    },
    spec: { totalParams: 'up', activeParams: 'up', flopsPerToken: 'up' },
    trainingMemory: 'up',
    inferenceMemory: 'up',
    inferenceTTFT: 'ge',
    inferenceTPOT: 'up',
  },

  // --- MoE-only ---
  {
    name: 'numExperts ×2',
    appliesTo: c => (c.numExperts ?? 0) > 0,
    apply: c => perturb(c, { numExperts: c.numExperts! * 2 }),
    // activeParams: router gate is hiddenSize × numExperts, so doubling experts
    // adds a small router delta (<1%) — active expert weights unchanged
    spec: { totalParams: 'up', activeParams: 'eq_pct', flopsPerToken: 'ge' },
    trainingMemory: 'up',
    inferenceMemory: 'up',
    inferenceTTFT: 'ge',
    inferenceTPOT: 'up',
  },
  {
    name: 'numActiveExperts +2',
    appliesTo: c => (c.numExperts ?? 0) > 0,
    apply: c => {
      const newActive = (c.numActiveExperts ?? 1) + 2;
      if (newActive > (c.numExperts ?? 1)) return undefined;
      return perturb(c, { numActiveExperts: newActive });
    },
    // totalParams: all expert weights counted regardless of active count; gate
    // is hiddenSize × numExperts (unchanged). So totalParams stays the same.
    spec: { totalParams: 'eq', activeParams: 'up', flopsPerToken: 'up' },
    trainingMemory: 'ge',
    // Inference loads all expert weights regardless — memory unchanged
    inferenceMemory: 'ge',
    inferenceTTFT: 'up',
    inferenceTPOT: 'up',
  },
  {
    name: 'expertIntermediateSize +512',
    appliesTo: c => (c.expertIntermediateSize ?? 0) > 0,
    apply: c => perturb(c, { expertIntermediateSize: c.expertIntermediateSize! + 512 }),
    spec: { totalParams: 'up', activeParams: 'up', flopsPerToken: 'up' },
    trainingMemory: 'up',
    trainingStepTime: 'up',
    inferenceMemory: 'up',
    inferenceTTFT: 'up',
    inferenceTPOT: 'up',
  },

  // --- MLA-only (DeepSeek V3) ---
  {
    name: 'kvLoraRank +128',
    appliesTo: c => (c.kvLoraRank ?? 0) > 0,
    apply: c => perturb(c, { kvLoraRank: c.kvLoraRank! + 128 }),
    spec: { totalParams: 'up', activeParams: 'up', flopsPerToken: 'ge' },
    trainingMemory: 'up',
    inferenceMemory: 'up',
    inferenceTTFT: 'ge',
    inferenceTPOT: 'up',
  },
  {
    name: 'qkRopeHeadDim +32',
    appliesTo: c => (c.qkRopeHeadDim ?? 0) > 0,
    apply: c => perturb(c, { qkRopeHeadDim: c.qkRopeHeadDim! + 32 }),
    spec: { totalParams: 'up', activeParams: 'up', flopsPerToken: 'ge' },
    trainingMemory: 'up',
    inferenceMemory: 'up',
    inferenceTTFT: 'ge',
    inferenceTPOT: 'up',
  },
];

// ---------------------------------------------------------------------------
// Training preset configs
// ---------------------------------------------------------------------------

interface TrainingPreset {
  name: string;
  modelId: string;
  simConfig: SimulationConfig;
}

function makeTrainingPresets(): TrainingPreset[] {
  // Presets 1-4: from PUBLISHED
  const fromPublished: TrainingPreset[] = [
    { name: 'LLaMA 3 405B', modelId: 'llama3-405b', simConfig: toSimConfig(PUBLISHED.llama3_405b_8k) },
    { name: 'Nemotron-4 340B', modelId: 'nemotron-4-340b', simConfig: toSimConfig(PUBLISHED.nemotron_4_340b) },
    { name: 'GPT-3 175B', modelId: 'gpt3-175b', simConfig: toSimConfig(PUBLISHED.gpt3_175b) },
    { name: 'DeepSeek V3', modelId: 'deepseek-v3', simConfig: toSimConfig(PUBLISHED.deepseek_v3_fp8_h800) },
  ];

  // Presets 5-8: manually constructed from DemoPreset fields
  const manual: TrainingPreset[] = [
    {
      name: 'LLaMA 4 Maverick',
      modelId: 'llama4-maverick',
      simConfig: {
        modelId: 'llama4-maverick',
        clusterConfig: createSingleNodeCluster('h100-sxm', 512)
          ?? createMultiNodeCluster('h100-sxm', 8, 64)!,
        sequenceLength: 8192,
        strategyType: 'fsdp-tp',
        strategyConfig: { tp: 4, ep: 32, sequenceParallel: true },
        globalBatchSize: 4096,
        microBatchSize: 2,
        activationCheckpointing: true,
        flashAttention: true,
        mixedPrecision: 'fp8',
      },
    },
    {
      name: 'Grok 2.5',
      modelId: 'grok-2.5',
      simConfig: {
        modelId: 'grok-2.5',
        clusterConfig: createMultiNodeCluster('h100-sxm', 8, 64)!,
        sequenceLength: 8192,
        strategyType: 'fsdp-tp-pp',
        strategyConfig: {
          tp: 4, pp: 2, ep: 8, sequenceParallel: true,
          pipelineSchedule: 'interleaved-1f1b', interleavedStages: 2,
        },
        globalBatchSize: 4096,
        microBatchSize: 2,
        activationCheckpointing: true,
        checkpointingGranularity: 'selective',
        flashAttention: true,
        mixedPrecision: 'fp8',
      },
    },
    {
      name: 'Qwen3 32B',
      modelId: 'qwen3-32b',
      simConfig: {
        modelId: 'qwen3-32b',
        clusterConfig: createMultiNodeCluster('h800-sxm', 8, 256)!,
        sequenceLength: 4096,
        strategyType: 'fsdp-tp',
        strategyConfig: { tp: 4, sequenceParallel: true },
        globalBatchSize: 2048,
        microBatchSize: 2,
        activationCheckpointing: true,
        checkpointingGranularity: 'selective',
        flashAttention: true,
        mixedPrecision: 'bf16',
      },
    },
    {
      name: 'OLMo 3 32B',
      modelId: 'olmo3-32b',
      simConfig: {
        modelId: 'olmo3-32b',
        clusterConfig: createMultiNodeCluster('h100-sxm', 8, 128)!,
        sequenceLength: 8192,
        strategyType: 'fsdp',
        strategyConfig: {},
        globalBatchSize: 1024,
        microBatchSize: 1,
        activationCheckpointing: true,
        checkpointingGranularity: 'selective',
        flashAttention: true,
        mixedPrecision: 'bf16',
      },
    },
  ];

  return [...fromPublished, ...manual];
}

// ---------------------------------------------------------------------------
// Inference preset configs
// ---------------------------------------------------------------------------

interface InferencePreset {
  name: string;
  modelId: string;
  config: InferenceSimulationConfig;
}

const INFERENCE_PRESETS: InferencePreset[] = [
  {
    name: 'DeepSeek V3 8×H200 FP8',
    modelId: 'deepseek-v3',
    config: {
      modelId: 'deepseek-v3', gpuId: 'h200-sxm', numGPUs: 8,
      batchSize: 32, inputSeqLen: 1024, outputSeqLen: 512,
      weightPrecision: 'fp8', kvCachePrecision: 'fp8',
      flashAttention: true, tensorParallel: 8,
    },
  },
  {
    name: 'LLaMA 3.3 70B 4×H100 BF16',
    modelId: 'llama3.3-70b',
    config: {
      modelId: 'llama3.3-70b', gpuId: 'h100-sxm', numGPUs: 4,
      batchSize: 16, inputSeqLen: 1024, outputSeqLen: 512,
      weightPrecision: 'bf16', kvCachePrecision: 'bf16',
      flashAttention: true, tensorParallel: 4,
    },
  },
  {
    name: 'LLaMA 3.3 70B 1×H200 INT4',
    modelId: 'llama3.3-70b',
    config: {
      modelId: 'llama3.3-70b', gpuId: 'h200-sxm', numGPUs: 1,
      batchSize: 8, inputSeqLen: 1024, outputSeqLen: 512,
      weightPrecision: 'int4', kvCachePrecision: 'fp8',
      flashAttention: true, tensorParallel: 1,
    },
  },
  {
    name: 'Qwen3 235B-A22B 4×H200 FP8',
    modelId: 'qwen3-235b-a22b',
    config: {
      modelId: 'qwen3-235b-a22b', gpuId: 'h200-sxm', numGPUs: 4,
      batchSize: 32, inputSeqLen: 1024, outputSeqLen: 512,
      weightPrecision: 'fp8', kvCachePrecision: 'fp8',
      flashAttention: true, tensorParallel: 4,
    },
  },
  {
    name: 'LLaMA 3.1 8B 1×L4 FP8',
    modelId: 'llama3.1-8b',
    config: {
      modelId: 'llama3.1-8b', gpuId: 'l4',  numGPUs: 1,
      batchSize: 8, inputSeqLen: 1024, outputSeqLen: 512,
      weightPrecision: 'fp8', kvCachePrecision: 'fp8',
      flashAttention: true, tensorParallel: 1,
    },
  },
  {
    name: 'LLaMA 3.1 8B 1×A10G INT8',
    modelId: 'llama3.1-8b',
    config: {
      modelId: 'llama3.1-8b', gpuId: 'a10g', numGPUs: 1,
      batchSize: 4, inputSeqLen: 1024, outputSeqLen: 256,
      weightPrecision: 'int8', kvCachePrecision: 'int8',
      flashAttention: true, tensorParallel: 1,
    },
  },
];

// ---------------------------------------------------------------------------
// Build baseline + perturbed ModelSpecs
// ---------------------------------------------------------------------------

function getBaselineAndPerturbed(
  modelId: string,
  seqLen: number,
  perturbation: Perturbation,
): { baseline: ReturnType<typeof buildModelSpec>; perturbed: ReturnType<typeof buildModelSpec> } | null {
  const rawConfig = getModelConfig(modelId);
  if (!rawConfig) throw new Error(`Unknown model: ${modelId}`);
  if (!perturbation.appliesTo(rawConfig)) return null;
  const modified = perturbation.apply(rawConfig);
  if (!modified) return null;

  const baseline = buildModelSpec(rawConfig, seqLen);
  const pert = buildModelSpec(modified, seqLen);
  return { baseline, perturbed: pert };
}

// ---------------------------------------------------------------------------
// Assertion helpers
// ---------------------------------------------------------------------------

function assertDirection(actual: number, baseline: number, dir: Direction, label: string) {
  switch (dir) {
    case 'up':
      expect(actual, `${label}: expected ${actual} > ${baseline}`).toBeGreaterThan(baseline);
      break;
    case 'ge':
      expect(actual, `${label}: expected ${actual} >= ${baseline}`).toBeGreaterThanOrEqual(baseline);
      break;
    case 'eq':
      expect(actual, `${label}: expected ${actual} === ${baseline}`).toBe(baseline);
      break;
    case 'eq_pct': {
      // Within 1% of baseline
      const delta = Math.abs(actual - baseline) / Math.max(baseline, 1e-12);
      expect(delta, `${label}: expected <1% change, got ${(delta * 100).toFixed(2)}%`).toBeLessThan(0.01);
      break;
    }
  }
}

// ===========================================================================
// Tests
// ===========================================================================

const TRAINING_PRESETS = makeTrainingPresets();

// Collect all unique modelIds across both training and inference presets
const ALL_MODEL_IDS = [...new Set([
  ...TRAINING_PRESETS.map(p => p.modelId),
  ...INFERENCE_PRESETS.map(p => p.modelId),
])];

describe('Sensitivity Validation', () => {
  // -----------------------------------------------------------------------
  // 1. ModelSpec physics (no simulation — just param/flop checks)
  // -----------------------------------------------------------------------
  describe('ModelSpec physics', () => {
    for (const pert of PERTURBATIONS) {
      describe(pert.name, () => {
        const applicableModels = ALL_MODEL_IDS.filter(id => {
          const cfg = getModelConfig(id);
          return cfg && pert.appliesTo(cfg) && pert.apply(cfg) !== undefined;
        });

        it.each(applicableModels)('%s — totalParams direction', (modelId) => {
          const pair = getBaselineAndPerturbed(modelId, 2048, pert)!;
          assertDirection(pair.perturbed.totalParams, pair.baseline.totalParams, pert.spec.totalParams, 'totalParams');
        });

        it.each(applicableModels)('%s — activeParams direction', (modelId) => {
          const pair = getBaselineAndPerturbed(modelId, 2048, pert)!;
          assertDirection(pair.perturbed.activeParams, pair.baseline.activeParams, pert.spec.activeParams, 'activeParams');
        });

        it.each(applicableModels)('%s — flopsPerToken direction', (modelId) => {
          const pair = getBaselineAndPerturbed(modelId, 2048, pert)!;
          assertDirection(pair.perturbed.flopsPerToken, pair.baseline.flopsPerToken, pert.spec.flopsPerToken, 'flopsPerToken');
        });
      });
    }
  });

  // -----------------------------------------------------------------------
  // 2. Training sensitivity
  // -----------------------------------------------------------------------
  describe('Training sensitivity', () => {
    for (const preset of TRAINING_PRESETS) {
      describe(preset.name, () => {
        const rawConfig = getModelConfig(preset.modelId)!;
        const seqLen = preset.simConfig.sequenceLength ?? 8192;

        for (const pert of PERTURBATIONS) {
          const simFilter = pert.affectsSimulation ?? pert.appliesTo;
          if (!simFilter(rawConfig)) continue;
          const modified = pert.apply(rawConfig);
          if (!modified) continue;

          describe(pert.name, () => {
            // Build perturbed ModelSpec once for this describe block
            const perturbedSpec = buildModelSpec(modified, seqLen);

            // Run baseline and perturbed simulations
            const baselineMetrics = getSimulationMetrics(preset.simConfig);
            const perturbedMetrics = getSimulationMetrics({
              ...preset.simConfig,
              modelId: undefined,
              modelSpec: perturbedSpec,
            });

            const baseOOM = baselineMetrics.memoryUtilization > 1.0;
            const pertOOM = perturbedMetrics.memoryUtilization > 1.0;

            it('memoryPerGPU.total increases', () => {
              assertDirection(
                perturbedMetrics.memoryPerGPU.total,
                baselineMetrics.memoryPerGPU.total,
                pert.trainingMemory,
                'training memory',
              );
            });

            if (pert.trainingStepTime && !baseOOM && !pertOOM) {
              it('stepTimeMs direction correct', () => {
                assertDirection(
                  perturbedMetrics.stepTimeMs,
                  baselineMetrics.stepTimeMs,
                  pert.trainingStepTime!,
                  'training stepTime',
                );
              });
            }
          });
        }
      });
    }
  });

  // -----------------------------------------------------------------------
  // 3. Inference sensitivity
  // -----------------------------------------------------------------------
  describe('Inference sensitivity', () => {
    for (const preset of INFERENCE_PRESETS) {
      describe(preset.name, () => {
        const rawConfig = getModelConfig(preset.modelId)!;
        const seqLen = preset.config.inputSeqLen ?? 1024;

        for (const pert of PERTURBATIONS) {
          const simFilter = pert.affectsSimulation ?? pert.appliesTo;
          if (!simFilter(rawConfig)) continue;
          const modified = pert.apply(rawConfig);
          if (!modified) continue;

          describe(pert.name, () => {
            const perturbedSpec = buildModelSpec(modified, seqLen);

            const baselineResult = runInferenceSimulation(preset.config);
            const perturbedResult = runInferenceSimulation({
              ...preset.config,
              modelId: undefined,
              modelSpec: perturbedSpec,
            });

            // If baseline fails, skip all assertions (can't compare)
            if (!baselineResult.success) return;

            it('memory.total direction correct', () => {
              if (!perturbedResult.success) {
                // Perturbed OOM means memory increased — pass
                return;
              }
              assertDirection(
                perturbedResult.memory.total,
                baselineResult.memory.total,
                pert.inferenceMemory,
                'inference memory',
              );
            });

            // Skip latency assertions if perturbed OOM
            if (perturbedResult.success) {
              it('latency.ttft direction correct', () => {
                assertDirection(
                  perturbedResult.latency.ttft,
                  baselineResult.latency.ttft,
                  pert.inferenceTTFT,
                  'inference TTFT',
                );
              });

              it('latency.tpot direction correct', () => {
                assertDirection(
                  perturbedResult.latency.tpot,
                  baselineResult.latency.tpot,
                  pert.inferenceTPOT,
                  'inference TPOT',
                );
              });
            }
          });
        }
      });
    }
  });
});
