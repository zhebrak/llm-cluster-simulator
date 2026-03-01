/**
 * Configuration store for simulation settings
 * Supports both training and inference modes with shared model/hardware config
 */

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import type { ModelSpec, ModelConfig, ClusterConfig, InferencePrecision } from '../types/index.ts';
import { PRECISION_SPECS } from '../types/index.ts';
import { getModel, modelRegistry } from '../core/models/index.ts';
import { getPresetCluster, createMultiNodeCluster, supportsFlashAttention, getGPU } from '../core/hardware/index.ts';
import { AVAILABLE_STRATEGIES } from '../core/strategies/index.ts';
import type { FinetuningMethod, LoraTargetModules } from '../core/strategies/lora.ts';
import { useSimulationStore } from './simulation.ts';
import { getGPUHourlyRate } from '../core/cost/index.ts';
import type { ShareConfig } from '../utils/share.ts';
import { type PublishedTrainingConfig, PUBLISHED } from '../data/published-training-configs.ts';

/**
 * Application mode - training or inference
 */
export type AppMode = 'training' | 'inference';

/**
 * Training goal presets
 */
export type TrainingGoal = 'chinchilla' | 'heavy-overtrain' | 'finetune' | 'custom';

/**
 * DP backend type for hybrid parallel strategies
 */
export type DPType = 'ddp' | 'fsdp' | 'zero-1' | 'zero-2' | 'zero-3';

/** Maps strategy name to its DP backend type */
const DP_TYPE_MAP: Record<string, DPType> = {
  'ddp': 'ddp', 'fsdp': 'fsdp', 'zero-1': 'zero-1', 'zero-3': 'zero-3',
  'fsdp-tp': 'fsdp', 'zero1-tp': 'zero-1',
  'ddp-tp-pp': 'ddp', 'zero1-tp-pp': 'zero-1', 'fsdp-tp-pp': 'fsdp',
};

/**
 * Snapshot of top-level hardware/model fields, saved per mode.
 * When switching modes, the outgoing mode's snapshot is saved and the incoming mode's is restored.
 */
interface HardwareSnapshot {
  modelId: string;
  clusterId: string;
  gpuId: string;
  numGPUs: number;
  gpusPerNode: number;
  sequenceLength: number;
  precision: 'fp32' | 'tf32' | 'fp16' | 'bf16' | 'fp8' | 'fp4';
  pricePerGPUHour: number | null;
}

/**
 * Training-specific configuration
 */
export type PipelineScheduleType = '1f1b' | 'interleaved-1f1b' | 'dualpipe-v';
export type CPImplementationType = 'ring' | 'all-gather';

export interface TrainingConfig {
  globalBatchSize: number;
  microBatchSize: number;
  gradientAccumulationSteps: number;
  // Token-based training target (more meaningful than arbitrary steps)
  trainingGoal: TrainingGoal;
  targetTokens: number; // Total tokens to train on
  strategyType: 'ddp' | 'fsdp' | 'zero-1' | 'zero-3' | 'auto'
    | 'fsdp-tp' | 'zero1-tp'
    | 'ddp-tp-pp' | 'zero1-tp-pp' | 'fsdp-tp-pp';
  tpDegree: number;
  ppDegree: number;
  dpDegree: number;
  epDegree: number;
  cpDegree: number;
  cpImplementation: CPImplementationType;
  numMicroBatches: number;
  activationCheckpointing: boolean;
  checkpointingGranularity: 'full' | 'selective';
  selectiveStoredLayers: number | 'auto';  // default: 'auto'
  flashAttention: boolean;
  sequenceParallel: boolean;
  // DP backend for hybrid strategies (3D parallel)
  dpType: DPType;
  // Pipeline schedule (PP strategies only)
  pipelineSchedule: PipelineScheduleType;
  interleavedStages: number;  // Virtual stages per device (v), default 2, range 2-8
  // Fine-tuning
  finetuningMethod: FinetuningMethod;
  loraRank: number;
  loraTargetModules: LoraTargetModules;
  // Snapshot of shared hardware fields preserved while this mode is inactive
  savedHardware?: HardwareSnapshot;
}

/**
 * Inference-specific configuration
 */
export interface InferenceConfigState {
  batchSize: number;
  inputSeqLen: number;
  outputSeqLen: number;
  weightPrecision: InferencePrecision;
  kvCachePrecision: InferencePrecision;
  flashAttention: boolean;
  pagedAttention: boolean;
  continuousBatching: boolean;
  tensorParallel: number;
  expertParallel: number;
  // Speculative decoding
  speculativeDecoding: boolean;
  draftModelId: string | null;
  draftModelSpec: ModelSpec | null;
  numSpeculativeTokens: number;
  acceptanceRate: number;
  // Pareto chart UI: which precision groups are visible
  paretoVisiblePrecisions: Record<string, boolean>;
  // Snapshot of shared hardware fields preserved while this mode is inactive
  savedHardware?: HardwareSnapshot;
}

/**
 * Combined configuration state
 */
export interface ConfigState {
  // Application mode
  mode: AppMode;

  // Shared: Model configuration
  modelId: string;
  modelSpec: ModelSpec | null;
  customModels: Record<string, ModelConfig>;
  nextCustomModelId: number;

  // Shared: Cluster/Hardware configuration
  clusterId: string;
  clusterConfig: ClusterConfig | null;
  gpuId: string;
  numGPUs: number;
  gpusPerNode: number;

  // Shared: Sequence length (used by both modes)
  sequenceLength: number;

  // Shared: Precision (main precision for both modes)
  precision: 'fp32' | 'tf32' | 'fp16' | 'bf16' | 'fp8' | 'fp4';

  // Shared: Custom price per GPU-hour (null = use default from GPU_HOURLY_RATES)
  pricePerGPUHour: number | null;

  // Training-specific configuration
  training: TrainingConfig;

  // Inference-specific configuration
  inference: InferenceConfigState;

  // Actions - Mode
  setMode: (mode: AppMode) => void;

  // Actions - Shared
  setModel: (modelId: string) => void;
  setCluster: (clusterId: string) => void;
  setCustomCluster: (gpuId: string, numGPUs: number, gpusPerNode: number) => void;
  setSequenceLength: (length: number) => void;
  setPrecision: (precision: 'fp32' | 'tf32' | 'fp16' | 'bf16' | 'fp8' | 'fp4') => void;

  // Actions - Training
  setTrainingParams: (params: Partial<TrainingConfig>) => void;
  setStrategy: (strategyType: TrainingConfig['strategyType']) => void;
  setStrategyParams: (params: Partial<TrainingConfig>) => void;

  // Actions - Inference
  setInferenceParams: (params: Partial<InferenceConfigState>) => void;
  setDraftModel: (modelId: string | null) => void;
  setSpeculativeDecoding: (enabled: boolean) => void;

  // Actions - Pricing
  setPricePerGPUHour: (price: number | null) => void;

  // Actions - Custom Model
  addCustomModel: (config: ModelConfig) => void;
  deleteCustomModel: (id: string) => void;

  // Actions - General
  reset: () => void;
  resetPrev: () => void;
  loadPresetBySlug: (slug: string) => void;
  loadShareConfig: (config: ShareConfig) => void;

  // Actions - Training Goal
  setTrainingGoal: (goal: TrainingGoal) => void;
  setTargetTokens: (tokens: number) => void;

  // Actions - Snapshot restore (for game/RPG mode exit)
  restoreFromSnapshot: (json: string) => void;

}

/**
 * Calculate target tokens based on training goal and model size.
 * Uses activeParams (not totalParams) because Chinchilla's compute-optimal
 * scaling D = 20N applies to per-token FLOPs, which scale with active params.
 * For dense models, activeParams === totalParams so this is unchanged.
 */
function calculateTargetTokens(goal: TrainingGoal, activeParams: number): number {
  switch (goal) {
    case 'chinchilla':
      // Chinchilla optimal: ~20 tokens per active parameter
      return Math.round(activeParams * 20);
    case 'heavy-overtrain':
      // Heavy overtrain: 200x active params (LLaMA-style inference-optimized)
      return Math.round(activeParams * 200);
    case 'finetune':
      // Fine-tuning: typically 1-10B tokens
      return 1e9;
    case 'custom':
    default:
      // Default to Chinchilla
      return Math.round(activeParams * 20);
  }
}

/**
 * Demo preset: a fully-specified model + cluster + strategy combo
 */
interface DemoPreset {
  slug: string;         // URL-safe identifier for ?preset= links
  modelLabel: string;   // e.g. "Mistral 7B"
  clusterLabel: string; // e.g. "8× A100"
  modelId: string;
  clusterId: string;
  // Custom cluster fields (used when clusterId='custom')
  gpuId?: string;
  numGPUs?: number;
  gpusPerNode?: number;
  sequenceLength: number;
  precision: 'fp32' | 'tf32' | 'fp16' | 'bf16' | 'fp8' | 'fp4';
  strategyType: TrainingConfig['strategyType'];
  tpDegree: number;
  ppDegree: number;
  dpDegree: number;
  epDegree: number;
  globalBatchSize: number;
  microBatchSize: number;
  gradientAccumulationSteps: number;
  activationCheckpointing: boolean;
  checkpointingGranularity?: 'full' | 'selective';
  selectiveStoredLayers?: number | 'auto';
  flashAttention: boolean;
  sequenceParallel: boolean;
  trainingGoal: TrainingGoal;
  targetTokens?: number; // Required when trainingGoal='custom', otherwise computed
  dpType: DPType;
  pipelineSchedule: PipelineScheduleType;
  interleavedStages: number;
}

interface DemoPresetOverrides {
  slug: string;
  modelLabel: string;
  clusterLabel: string;
  trainingGoal: TrainingGoal;
  targetTokens?: number;
}

function publishedToDemoPreset(
  pc: PublishedTrainingConfig,
  overrides: DemoPresetOverrides,
): DemoPreset {
  const totalGPUs = pc.gpusPerNode * pc.numNodes;
  const sc = pc.strategyConfig;
  const tp = sc?.tp ?? 1;
  const pp = sc?.pp ?? 1;
  const cp = sc?.cp ?? 1;
  const ep = sc?.ep ?? 1;
  // EP subdivides DP (not orthogonal axis) — see 3d-parallel.ts:468
  // Same formula as config-parity.test.ts:49
  const dp = Math.floor(totalGPUs / (tp * pp * cp));

  return {
    ...overrides,
    modelId: pc.modelId,
    clusterId: 'custom',
    gpuId: pc.gpuId,
    numGPUs: totalGPUs,
    gpusPerNode: pc.gpusPerNode,
    sequenceLength: pc.sequenceLength,
    precision: (pc.mixedPrecision ?? 'bf16') as DemoPreset['precision'],
    strategyType: pc.strategyType,
    tpDegree: tp,
    ppDegree: pp,
    dpDegree: dp,
    epDegree: ep,
    globalBatchSize: pc.globalBatchSize,
    microBatchSize: pc.microBatchSize,
    gradientAccumulationSteps: Math.ceil(pc.globalBatchSize / (pc.microBatchSize * dp)),
    activationCheckpointing: pc.activationCheckpointing,
    checkpointingGranularity: pc.checkpointingGranularity,
    flashAttention: pc.flashAttention,
    sequenceParallel: sc?.sequenceParallel ?? false,
    dpType: DP_TYPE_MAP[pc.strategyType] ?? 'fsdp',
    pipelineSchedule: sc?.pipelineSchedule ?? '1f1b',
    interleavedStages: sc?.interleavedStages ?? 2,
  };
}

export const DEMO_PRESETS: DemoPreset[] = [
  // 1. LLaMA 3 405B — derived from published config (Meta §3.3.2)
  publishedToDemoPreset(PUBLISHED.llama3_405b_8k, {
    slug: 'llama3-405b', modelLabel: 'LLaMA 3 405B', clusterLabel: '16384× H100',
    trainingGoal: 'custom', targetTokens: 15e12,
  }),
  // 2. Nemotron-4 340B — derived from published config (Megatron-Core blog)
  publishedToDemoPreset(PUBLISHED.nemotron_4_340b, {
    slug: 'nemotron-4-340b', modelLabel: 'Nemotron-4 340B', clusterLabel: '6144× H100',
    trainingGoal: 'custom', targetTokens: 9e12,
  }),
  // 3. GPT-3 175B — derived from published config (Narayanan et al. 2021)
  publishedToDemoPreset(PUBLISHED.gpt3_175b, {
    slug: 'gpt3-175b', modelLabel: 'GPT-3 175B', clusterLabel: '1024× A100',
    trainingGoal: 'custom', targetTokens: 300e9,
  }),
  // 4. DeepSeek V3/R1 — derived from published config
  publishedToDemoPreset(PUBLISHED.deepseek_v3_fp8_h800, {
    slug: 'deepseek-v3-r1', modelLabel: 'DeepSeek V3/R1', clusterLabel: '2048× H800',
    trainingGoal: 'custom', targetTokens: 15e12,
  }),
  // 5. LLaMA 4 Maverick — Speculative parallelism, FP8 per Meta LLaMA 4 report
  {
    slug: 'llama4-maverick',
    modelLabel: 'LLaMA 4 Maverick', clusterLabel: '512× H100',
    modelId: 'llama4-maverick', clusterId: '512x-h100', sequenceLength: 8192, precision: 'fp8',
    strategyType: 'fsdp-tp', tpDegree: 4, ppDegree: 1, dpDegree: 128, epDegree: 32,
    globalBatchSize: 4096, microBatchSize: 2, gradientAccumulationSteps: 16,
    activationCheckpointing: true, flashAttention: true, sequenceParallel: true,
    trainingGoal: 'custom', targetTokens: 22e12, dpType: 'fsdp', pipelineSchedule: '1f1b', interleavedStages: 2,
  },
  // 6. Grok 2.5 — Speculative config, xAI has not published training details
  {
    slug: 'grok-2.5',
    modelLabel: 'Grok 2.5', clusterLabel: '512× H100',
    modelId: 'grok-2.5', clusterId: '512x-h100', sequenceLength: 8192, precision: 'fp8',
    strategyType: 'fsdp-tp-pp', tpDegree: 4, ppDegree: 2, dpDegree: 64, epDegree: 8,
    globalBatchSize: 4096, microBatchSize: 2, gradientAccumulationSteps: 32,
    activationCheckpointing: true, checkpointingGranularity: 'selective',
    flashAttention: true, sequenceParallel: true,
    trainingGoal: 'custom', targetTokens: 6e12, dpType: 'fsdp',
    pipelineSchedule: 'interleaved-1f1b', interleavedStages: 2,
  },
  // 7. Qwen3 32B — 2048 H800 (estimated), FSDP+TP
  {
    slug: 'qwen3-32b',
    modelLabel: 'Qwen3 32B', clusterLabel: '2048× H800',
    modelId: 'qwen3-32b', clusterId: 'custom', gpuId: 'h800-sxm', numGPUs: 2048, gpusPerNode: 8,
    sequenceLength: 4096, precision: 'bf16',
    strategyType: 'fsdp-tp', tpDegree: 4, ppDegree: 1, dpDegree: 512, epDegree: 1,
    globalBatchSize: 2048, microBatchSize: 2, gradientAccumulationSteps: 2,
    activationCheckpointing: true, checkpointingGranularity: 'selective',
    flashAttention: true, sequenceParallel: true,
    trainingGoal: 'custom', targetTokens: 36e12, dpType: 'fsdp', pipelineSchedule: '1f1b', interleavedStages: 2,
  },
  // 8. OLMo 3 32B — 1024 H100s, FSDP (AI2, 2025; published ~41% MFU)
  {
    slug: 'olmo3-32b',
    modelLabel: 'OLMo 3 32B', clusterLabel: '1024× H100',
    modelId: 'olmo3-32b', clusterId: 'custom', gpuId: 'h100-sxm', numGPUs: 1024, gpusPerNode: 8,
    sequenceLength: 8192, precision: 'bf16',
    strategyType: 'fsdp', tpDegree: 1, ppDegree: 1, dpDegree: 1024, epDegree: 1,
    globalBatchSize: 1024, microBatchSize: 1, gradientAccumulationSteps: 1,
    activationCheckpointing: true, checkpointingGranularity: 'selective', selectiveStoredLayers: 35,
    flashAttention: true, sequenceParallel: false,
    trainingGoal: 'custom', targetTokens: 5.5e12, dpType: 'fsdp', pipelineSchedule: '1f1b', interleavedStages: 2,
  },
];

/**
 * Inference demo preset: model + cluster + inference config
 */
interface InferenceDemoPreset {
  slug: string;         // URL-safe identifier for ?preset= links
  modelLabel: string;
  clusterLabel: string;
  modelId: string;
  clusterId: string;
  // Custom cluster fields (used when clusterId='custom')
  gpuId?: string;
  numGPUs?: number;
  gpusPerNode?: number;
  sequenceLength: number;
  batchSize: number;
  inputSeqLen: number;
  outputSeqLen: number;
  weightPrecision: InferencePrecision;
  kvCachePrecision: InferencePrecision;
  flashAttention: boolean;
  tensorParallel: number;
}

const INFERENCE_DEMO_PRESETS: InferenceDemoPreset[] = [
  // 1. DeepSeek V3/R1 — MoE FP8 on 8× H200, TP=8
  {
    slug: 'deepseek-v3-r1-inference',
    modelLabel: 'DeepSeek V3/R1', clusterLabel: '8× H200',
    modelId: 'deepseek-v3', clusterId: 'custom', gpuId: 'h200-sxm', numGPUs: 8, gpusPerNode: 8,
    sequenceLength: 4096,
    batchSize: 32, inputSeqLen: 1024, outputSeqLen: 512,
    weightPrecision: 'fp8', kvCachePrecision: 'fp8',
    flashAttention: true, tensorParallel: 8,
  },
  // 2. LLaMA 3.3 70B — BF16 on 4× H100, TP=4 (matches training dtype)
  {
    slug: 'llama3.3-70b-inference',
    modelLabel: 'LLaMA 3.3 70B', clusterLabel: '4× H100',
    modelId: 'llama3.3-70b', clusterId: 'custom', gpuId: 'h100-sxm', numGPUs: 4, gpusPerNode: 4,
    sequenceLength: 8192,
    batchSize: 16, inputSeqLen: 1024, outputSeqLen: 512,
    weightPrecision: 'bf16', kvCachePrecision: 'bf16',
    flashAttention: true, tensorParallel: 4,
  },
  // 3. LLaMA 3.3 70B INT4 — single H200, INT4 weights + FP8 KV cache (standard serving config)
  {
    slug: 'llama3.3-70b-int4-inference',
    modelLabel: 'LLaMA 3.3 70B INT4', clusterLabel: '1× H200',
    modelId: 'llama3.3-70b', clusterId: 'custom', gpuId: 'h200-sxm', numGPUs: 1, gpusPerNode: 1,
    sequenceLength: 8192,
    batchSize: 8, inputSeqLen: 1024, outputSeqLen: 512,
    weightPrecision: 'int4', kvCachePrecision: 'fp8',
    flashAttention: true, tensorParallel: 1,
  },
  // 4. Qwen3 235B-A22B — MoE FP8 on 4× H200, TP=4
  {
    slug: 'qwen3-235b-inference',
    modelLabel: 'Qwen3 235B-A22B', clusterLabel: '4× H200',
    modelId: 'qwen3-235b-a22b', clusterId: 'custom', gpuId: 'h200-sxm', numGPUs: 4, gpusPerNode: 4,
    sequenceLength: 4096,
    batchSize: 32, inputSeqLen: 1024, outputSeqLen: 512,
    weightPrecision: 'fp8', kvCachePrecision: 'fp8',
    flashAttention: true, tensorParallel: 4,
  },
  // 5. LLaMA 3.1 8B — FP8 on 1× L4, budget edge
  {
    slug: 'llama3.1-8b-inference',
    modelLabel: 'LLaMA 3.1 8B', clusterLabel: '1× L4',
    modelId: 'llama3.1-8b', clusterId: 'inf-1x-l4', sequenceLength: 8192,
    batchSize: 8, inputSeqLen: 1024, outputSeqLen: 512,
    weightPrecision: 'fp8', kvCachePrecision: 'fp8',
    flashAttention: true, tensorParallel: 1,
  },
  // 6. LLaMA 3.1 8B INT8 — budget cloud inference on AWS g5 (A10G 24GB)
  {
    slug: 'llama3.1-8b-a10g-inference',
    modelLabel: 'LLaMA 3.1 8B INT8', clusterLabel: '1× A10G',
    modelId: 'llama3.1-8b', clusterId: 'inf-1x-a10g', sequenceLength: 8192,
    batchSize: 4, inputSeqLen: 1024, outputSeqLen: 256,
    weightPrecision: 'int8', kvCachePrecision: 'int8',
    flashAttention: true, tensorParallel: 1,
  },
];

// Separate indices so training and inference cycling don't interfere.
// Index means "currently loaded preset". Preset 0 is loaded on startup.
let trainingPresetIndex = 0;
let inferencePresetIndex = 0;

/** Returns the model and cluster labels of the next preset for the current mode. */
export function getNextDemoPreset(mode: AppMode): { modelLabel: string; clusterLabel: string } {
  if (mode === 'inference') {
    const { modelLabel, clusterLabel } = INFERENCE_DEMO_PRESETS[(inferencePresetIndex + 1) % INFERENCE_DEMO_PRESETS.length];
    return { modelLabel, clusterLabel };
  }
  const { modelLabel, clusterLabel } = DEMO_PRESETS[(trainingPresetIndex + 1) % DEMO_PRESETS.length];
  return { modelLabel, clusterLabel };
}

/** Returns the model and cluster labels of the previous preset for the current mode. */
export function getPrevDemoPreset(mode: AppMode): { modelLabel: string; clusterLabel: string } {
  if (mode === 'inference') {
    const idx = (inferencePresetIndex - 1 + INFERENCE_DEMO_PRESETS.length) % INFERENCE_DEMO_PRESETS.length;
    const { modelLabel, clusterLabel } = INFERENCE_DEMO_PRESETS[idx];
    return { modelLabel, clusterLabel };
  }
  const idx = (trainingPresetIndex - 1 + DEMO_PRESETS.length) % DEMO_PRESETS.length;
  const { modelLabel, clusterLabel } = DEMO_PRESETS[idx];
  return { modelLabel, clusterLabel };
}

/** Looks up a preset by its URL slug (case-insensitive). */
export function findPresetBySlug(slug: string):
  | { preset: DemoPreset; mode: 'training' }
  | { preset: InferenceDemoPreset; mode: 'inference' }
  | null {
  const s = slug.toLowerCase();
  const training = DEMO_PRESETS.find(p => p.slug === s);
  if (training) return { preset: training, mode: 'training' };
  const inference = INFERENCE_DEMO_PRESETS.find(p => p.slug === s);
  if (inference) return { preset: inference, mode: 'inference' };
  return null;
}

/** Extract HardwareSnapshot from a training or inference demo preset. */
function snapshotFromPreset(p: DemoPreset | InferenceDemoPreset): HardwareSnapshot {
  const gpuId = p.gpuId ?? (p.clusterId !== 'custom' ? (getPresetCluster(p.clusterId)?.node.gpu.id ?? 'h100-sxm') : 'h100-sxm');
  const numGPUs = p.numGPUs ?? (p.clusterId !== 'custom' ? (getPresetCluster(p.clusterId)?.totalGPUs ?? 8) : 8);
  const gpusPerNode = p.gpusPerNode ?? (p.clusterId !== 'custom' ? (getPresetCluster(p.clusterId)?.gpusPerNode ?? 8) : 8);
  const precision = 'precision' in p ? p.precision : ('weightPrecision' in p ? p.weightPrecision as HardwareSnapshot['precision'] : 'bf16');
  return {
    modelId: p.modelId, clusterId: p.clusterId, gpuId, numGPUs, gpusPerNode,
    sequenceLength: p.sequenceLength, precision, pricePerGPUHour: null,
  };
}

// Derive initial configs from first presets so they stay in sync automatically
const _tp0 = DEMO_PRESETS[0];
const _ip0 = INFERENCE_DEMO_PRESETS[0];

const initialTrainingConfig: TrainingConfig = {
  globalBatchSize: _tp0.globalBatchSize,
  microBatchSize: _tp0.microBatchSize,
  gradientAccumulationSteps: _tp0.gradientAccumulationSteps,
  trainingGoal: _tp0.trainingGoal,
  targetTokens: _tp0.targetTokens ?? calculateTargetTokens(_tp0.trainingGoal, (getModel(_tp0.modelId, _tp0.sequenceLength)?.activeParams ?? 7e9)),
  strategyType: _tp0.strategyType,
  tpDegree: _tp0.tpDegree,
  ppDegree: _tp0.ppDegree,
  dpDegree: _tp0.dpDegree,
  epDegree: _tp0.epDegree,
  cpDegree: 1,
  cpImplementation: 'ring' as CPImplementationType,
  numMicroBatches: _tp0.gradientAccumulationSteps,
  activationCheckpointing: _tp0.activationCheckpointing,
  checkpointingGranularity: 'full' as const,
  selectiveStoredLayers: 'auto' as number | 'auto',
  flashAttention: _tp0.flashAttention,
  sequenceParallel: _tp0.sequenceParallel,
  dpType: _tp0.dpType,
  pipelineSchedule: _tp0.pipelineSchedule,
  interleavedStages: _tp0.interleavedStages,
  finetuningMethod: 'full' as FinetuningMethod,
  loraRank: 16,
  loraTargetModules: 'q_k_v_o' as LoraTargetModules,
  savedHardware: snapshotFromPreset(_tp0),
};

const initialInferenceConfig: InferenceConfigState = {
  batchSize: _ip0.batchSize,
  inputSeqLen: _ip0.inputSeqLen,
  outputSeqLen: _ip0.outputSeqLen,
  weightPrecision: _ip0.weightPrecision,
  kvCachePrecision: _ip0.kvCachePrecision,
  flashAttention: _ip0.flashAttention,
  pagedAttention: true,
  continuousBatching: true,
  tensorParallel: _ip0.tensorParallel,
  expertParallel: 1,
  speculativeDecoding: false,
  draftModelId: null,
  draftModelSpec: null,
  numSpeculativeTokens: 4,
  acceptanceRate: 0.7,
  paretoVisiblePrecisions: { bf16: true, fp8: true, int4: true, q4_k_m: false, q8_0: false },
  savedHardware: snapshotFromPreset(_ip0),
};

// Default state derived from first training preset
const _tp0hw = initialTrainingConfig.savedHardware!;
const initialState = {
  mode: 'training' as AppMode,

  modelId: _tp0hw.modelId,
  modelSpec: getModel(_tp0hw.modelId, _tp0hw.sequenceLength) as ModelSpec | null,
  customModels: {} as Record<string, ModelConfig>,
  nextCustomModelId: 1,

  clusterId: _tp0hw.clusterId,
  clusterConfig: _tp0hw.clusterId === 'custom'
    ? createMultiNodeCluster(_tp0hw.gpuId, _tp0hw.gpusPerNode, Math.ceil(_tp0hw.numGPUs / _tp0hw.gpusPerNode)) ?? null as ClusterConfig | null
    : getPresetCluster(_tp0hw.clusterId) ?? null as ClusterConfig | null,
  gpuId: _tp0hw.gpuId,
  numGPUs: _tp0hw.numGPUs,
  gpusPerNode: _tp0hw.gpusPerNode,

  sequenceLength: _tp0hw.sequenceLength,
  precision: _tp0hw.precision,
  pricePerGPUHour: null as number | null,

  training: initialTrainingConfig,
  inference: initialInferenceConfig,
};

const STORAGE_KEY = 'llm-sim-config';
const STORAGE_VERSION = 1;

/**
 * Load persisted config from localStorage synchronously.
 * Rehydrates derived objects (modelSpec, clusterConfig, draftModelSpec) from IDs.
 */
const validStrategyIds = new Set(AVAILABLE_STRATEGIES.map(s => s.id));
const validModes = new Set<string>(['training', 'inference']);
const validPrecisions = new Set<string>(['fp32', 'tf32', 'fp16', 'bf16', 'fp8', 'fp4']);
const validInferencePrecisions = new Set<string>(Object.keys(PRECISION_SPECS));
const validTrainingGoals = new Set<string>(['chinchilla', 'heavy-overtrain', 'finetune', 'custom']);

/**
 * Rehydrate config state from a raw JSON string (same format saveState writes).
 * Returns initialState on any parse/validation error.
 */
function rehydrateFromJSON(raw: string): typeof initialState {
  try {
    const { state: ps, version } = JSON.parse(raw);
    if (version !== STORAGE_VERSION || !ps) return initialState;

    // Migrate legacy single custom model → multi-model map
    let customModels: Record<string, ModelConfig> = {};
    let nextCustomModelId = 1;
    if (ps.customModels && typeof ps.customModels === 'object' && !Array.isArray(ps.customModels)) {
      customModels = ps.customModels;
      nextCustomModelId = ps.nextCustomModelId ?? (Math.max(0, ...Object.keys(customModels)
        .map(k => parseInt(k.replace('custom-', '')) || 0)) + 1);
    }
    // Re-register all custom models before ID validation
    for (const [id, cfg] of Object.entries(customModels)) {
      if ((cfg as unknown as Record<string, unknown>)?.hiddenSize && (cfg as unknown as Record<string, unknown>)?.numLayers) {
        try { modelRegistry.registerCustom(id, cfg as ModelConfig); } catch { /* ignore — invalid custom model in localStorage */ }
      }
    }

    // Validate persisted IDs against current registries — fall back to
    // defaults for unresolvable entries (e.g. unavailable or renamed models).
    const modelId = (ps.modelId && getModel(ps.modelId)) ? ps.modelId : initialState.modelId;
    const sequenceLength = ps.sequenceLength ?? initialState.sequenceLength;
    const gpuId = (ps.gpuId && getGPU(ps.gpuId)) ? ps.gpuId : initialState.gpuId;
    const clusterId = (ps.clusterId === 'custom' || (ps.clusterId && getPresetCluster(ps.clusterId)))
      ? ps.clusterId : initialState.clusterId;
    const numGPUs = ps.numGPUs ?? initialState.numGPUs;
    const gpusPerNode = ps.gpusPerNode ?? initialState.gpusPerNode;
    const mode = validModes.has(ps.mode) ? ps.mode : initialState.mode;
    const precision = validPrecisions.has(ps.precision) ? ps.precision : initialState.precision;

    // Merge persisted values onto defaults (so new fields get defaults)
    const training = { ...initialTrainingConfig, ...(ps.training ?? {}) };

    // Validate strategy type
    if (!validStrategyIds.has(training.strategyType)) {
      training.strategyType = initialTrainingConfig.strategyType;
      training.tpDegree = initialTrainingConfig.tpDegree;
      training.ppDegree = initialTrainingConfig.ppDegree;
      training.dpDegree = initialTrainingConfig.dpDegree;
    }

    // Fall back for any unknown training goal
    if (!validTrainingGoals.has(training.trainingGoal)) {
      training.trainingGoal = initialTrainingConfig.trainingGoal;
    }

    const inference: InferenceConfigState = {
      ...initialInferenceConfig,
      ...(ps.inference ?? {}),
      // Rehydrate draftModelSpec from ID (null if model is unavailable)
      draftModelSpec: ps.inference?.draftModelId
        ? getModel(ps.inference.draftModelId, ps.inference?.inputSeqLen ?? initialInferenceConfig.inputSeqLen) ?? null
        : null,
    };

    // Validate inference precision enums
    if (inference.weightPrecision && !validInferencePrecisions.has(inference.weightPrecision)) {
      inference.weightPrecision = initialInferenceConfig.weightPrecision;
    }
    if (inference.kvCachePrecision && !validInferencePrecisions.has(inference.kvCachePrecision)) {
      inference.kvCachePrecision = initialInferenceConfig.kvCachePrecision;
    }

    return {
      mode,
      modelId,
      modelSpec: getModel(modelId, sequenceLength) ?? null,
      customModels,
      nextCustomModelId,
      clusterId,
      clusterConfig: clusterId === 'custom'
        ? createMultiNodeCluster(gpuId, gpusPerNode, Math.ceil(numGPUs / gpusPerNode)) ?? null
        : getPresetCluster(clusterId) ?? null,
      gpuId,
      numGPUs,
      gpusPerNode,
      sequenceLength,
      precision,
      pricePerGPUHour: ps.pricePerGPUHour ?? null,
      training,
      inference,
    };
  } catch {
    return initialState;
  }
}

function loadPersistedState(): typeof initialState {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return initialState;
  return rehydrateFromJSON(raw);
}

/**
 * Save config state to localStorage (strips derived objects and functions).
 */
function saveState(state: ConfigState): void {
  try {
    const { modelSpec: _modelSpec, clusterConfig: _clusterConfig, ...rest } = state as unknown as Record<string, unknown>;
    // Strip functions (JSON.stringify drops them) and derived objects
    const toSave = {
      ...rest,
      inference: { ...state.inference, draftModelSpec: undefined },
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ state: toSave, version: STORAGE_VERSION }));
  } catch { /* localStorage full or unavailable — silently ignore */ }
}

/** Build full state from a training preset. */
function applyTrainingPreset(preset: DemoPreset): Partial<ConfigState> {
  const modelSpec = getModel(preset.modelId, preset.sequenceLength) ?? null;
  const clusterConfig = preset.clusterId === 'custom' && preset.gpuId
    ? createMultiNodeCluster(preset.gpuId, preset.gpusPerNode ?? 8, Math.ceil((preset.numGPUs ?? 8) / (preset.gpusPerNode ?? 8))) ?? null
    : getPresetCluster(preset.clusterId) ?? null;
  const gpuId = preset.gpuId ?? clusterConfig?.node.gpu.id ?? 'h100-sxm';
  const numGPUs = preset.numGPUs ?? clusterConfig?.totalGPUs ?? 8;
  const gpusPerNode = preset.gpusPerNode ?? clusterConfig?.gpusPerNode ?? 8;
  const targetTokens = preset.targetTokens ?? calculateTargetTokens(
    preset.trainingGoal,
    modelSpec?.activeParams ?? modelSpec?.totalParams ?? 7e9
  );
  const ga = preset.gradientAccumulationSteps;

  const hwSnapshot: HardwareSnapshot = {
    modelId: preset.modelId,
    clusterId: preset.clusterId,
    gpuId,
    numGPUs,
    gpusPerNode,
    sequenceLength: preset.sequenceLength,
    precision: preset.precision,
    pricePerGPUHour: null,
  };

  const training: TrainingConfig = {
    globalBatchSize: preset.globalBatchSize,
    microBatchSize: preset.microBatchSize,
    gradientAccumulationSteps: ga,
    trainingGoal: preset.trainingGoal,
    targetTokens,
    strategyType: preset.strategyType,
    tpDegree: preset.tpDegree,
    ppDegree: preset.ppDegree,
    dpDegree: preset.dpDegree,
    epDegree: preset.epDegree,
    cpDegree: 1,
    cpImplementation: 'ring' as CPImplementationType,
    numMicroBatches: ga,
    activationCheckpointing: preset.activationCheckpointing,
    checkpointingGranularity: preset.checkpointingGranularity ?? 'full',
    selectiveStoredLayers: preset.selectiveStoredLayers ?? 'auto' as number | 'auto',
    flashAttention: preset.flashAttention,
    sequenceParallel: preset.sequenceParallel,
    dpType: preset.dpType,
    pipelineSchedule: preset.pipelineSchedule,
    interleavedStages: preset.interleavedStages,
    finetuningMethod: 'full' as FinetuningMethod,
    loraRank: 16,
    loraTargetModules: 'q_k_v_o' as LoraTargetModules,
    savedHardware: hwSnapshot,
  };

  return {
    mode: 'training' as AppMode,
    // Shared fields
    modelId: preset.modelId,
    modelSpec,
    clusterId: preset.clusterId,
    clusterConfig,
    gpuId,
    numGPUs,
    gpusPerNode,
    sequenceLength: preset.sequenceLength,
    precision: preset.precision,
    pricePerGPUHour: null as number | null,
    // Training config
    training,
    // Inference config intentionally omitted — preserved from current state
  };
}

/** Build full state from an inference preset. */
function applyInferencePreset(ip: InferenceDemoPreset): Partial<ConfigState> {
  const modelSpec = getModel(ip.modelId, ip.sequenceLength) ?? null;
  const clusterConfig = ip.clusterId === 'custom' && ip.gpuId
    ? createMultiNodeCluster(ip.gpuId, ip.gpusPerNode ?? 8, Math.ceil((ip.numGPUs ?? 1) / (ip.gpusPerNode ?? 8))) ?? null
    : getPresetCluster(ip.clusterId) ?? null;
  const gpuId = ip.gpuId ?? clusterConfig?.node.gpu.id ?? 'h100-sxm';
  const numGPUs = ip.numGPUs ?? clusterConfig?.totalGPUs ?? 1;
  const gpusPerNode = ip.gpusPerNode ?? clusterConfig?.gpusPerNode ?? 1;

  const hwSnapshot: HardwareSnapshot = {
    modelId: ip.modelId,
    clusterId: ip.clusterId,
    gpuId,
    numGPUs,
    gpusPerNode,
    sequenceLength: ip.sequenceLength,
    precision: ip.weightPrecision as typeof initialState.precision,
    pricePerGPUHour: null,
  };

  return {
    mode: 'inference' as AppMode,
    // Shared fields
    modelId: ip.modelId,
    modelSpec,
    clusterId: ip.clusterId,
    clusterConfig,
    gpuId,
    numGPUs,
    gpusPerNode,
    sequenceLength: ip.sequenceLength,
    precision: ip.weightPrecision as typeof initialState.precision,
    pricePerGPUHour: null as number | null,
    // Inference config
    inference: {
      ...initialInferenceConfig,
      batchSize: ip.batchSize,
      inputSeqLen: ip.inputSeqLen,
      outputSeqLen: ip.outputSeqLen,
      weightPrecision: ip.weightPrecision,
      kvCachePrecision: ip.kvCachePrecision,
      flashAttention: ip.flashAttention,
      tensorParallel: ip.tensorParallel,
      savedHardware: hwSnapshot,
    },
    // Training config and legacy mirrors intentionally omitted — preserved from current state
  };
}

/**
 * Helper to preserve custom models across state resets (presets, reset, share config).
 * Custom models are a user library, not per-session config.
 */
function withPreservedCustomModels(
  get: () => ConfigState,
  set: (fn: (draft: ConfigState) => void) => void,
  fn: () => void
) {
  const preserved = get().customModels;
  const nextId = get().nextCustomModelId;
  fn();
  set(draft => {
    draft.customModels = preserved;
    draft.nextCustomModelId = nextId;
  });
  // Re-register in model registry (preset loading may have cleared them)
  for (const [id, cfg] of Object.entries(preserved)) {
    try { modelRegistry.registerCustom(id, cfg); } catch { /* ignore — re-registration failure is non-fatal */ }
  }
}

/**
 * Copy current top-level hardware fields into the active mode's snapshot.
 * Called inside immer producers after any shared setter mutates top-level state.
 */
function syncSnapshot(draft: ConfigState): void {
  const target = draft.mode === 'training' ? draft.training : draft.inference;
  target.savedHardware = {
    modelId: draft.modelId,
    clusterId: draft.clusterId,
    gpuId: draft.gpuId,
    numGPUs: draft.numGPUs,
    gpusPerNode: draft.gpusPerNode,
    sequenceLength: draft.sequenceLength,
    precision: draft.precision,
    pricePerGPUHour: draft.pricePerGPUHour,
  };
}

export const useConfigStore = create<ConfigState>()(
  immer((set, get) => ({
    ...loadPersistedState(),

    // Mode switching — swap hardware state between modes
    setMode: (mode: AppMode) => {
      if (get().mode === mode) return;
      set(state => {
        // Save current top-level hardware → outgoing mode's snapshot
        syncSnapshot(state);

        // Restore incoming mode's snapshot → top-level
        const incoming = mode === 'training' ? state.training : state.inference;
        if (incoming.savedHardware) {
          const hw = incoming.savedHardware;
          state.modelId = hw.modelId;
          state.clusterId = hw.clusterId;
          state.gpuId = hw.gpuId;
          state.numGPUs = hw.numGPUs;
          state.gpusPerNode = hw.gpusPerNode;
          state.sequenceLength = hw.sequenceLength;
          state.precision = hw.precision;
          state.pricePerGPUHour = hw.pricePerGPUHour;
        }
        // else: first switch to this mode — keep current top-level values

        state.mode = mode;
      });
      // Rehydrate derived objects (modelSpec, clusterConfig) from restored IDs
      const s = get();
      const modelSpec = getModel(s.modelId, s.sequenceLength);
      const clusterConfig = s.clusterId === 'custom'
        ? createMultiNodeCluster(s.gpuId, s.gpusPerNode, Math.ceil(s.numGPUs / s.gpusPerNode))
        : getPresetCluster(s.clusterId);
      set(state => {
        state.modelSpec = modelSpec ?? null;
        state.clusterConfig = clusterConfig ?? null;
      });
      // Clear stale simulation results from previous mode
      useSimulationStore.getState().reset();
    },

    // Shared model configuration
    setModel: (modelId: string) => {
      const state = get();
      const modelSpec = getModel(modelId, state.sequenceLength);
      set(draft => {
        draft.modelId = modelId;
        draft.modelSpec = modelSpec ?? null;
        // Recalculate target tokens based on new model size (unless custom)
        if (modelSpec && draft.training.trainingGoal !== 'custom') {
          draft.training.targetTokens = calculateTargetTokens(
            draft.training.trainingGoal,
            modelSpec.activeParams ?? modelSpec.totalParams
          );
        }
        // Fall back to 1F1B if interleaved schedule is incompatible with new model's layer count
        if (
          modelSpec &&
          draft.training.pipelineSchedule === 'interleaved-1f1b' &&
          draft.training.ppDegree > 1 &&
          modelSpec.numLayers % (draft.training.ppDegree * draft.training.interleavedStages) !== 0
        ) {
          draft.training.pipelineSchedule = '1f1b';
        }
        syncSnapshot(draft);
      });
      // Clear stale simulation results
      useSimulationStore.getState().reset();
    },

    // Custom model
    addCustomModel: (customConfig: ModelConfig) => {
      const n = get().nextCustomModelId;
      const id = `custom-${n}`;
      modelRegistry.registerCustom(id, customConfig);
      const state = get();
      const modelSpec = getModel(id, state.sequenceLength) ?? null;
      set(draft => {
        draft.customModels[id] = customConfig;
        draft.nextCustomModelId = n + 1;
        draft.modelId = id;
        draft.modelSpec = modelSpec;
        if (modelSpec && draft.training.trainingGoal !== 'custom') {
          draft.training.targetTokens = calculateTargetTokens(
            draft.training.trainingGoal,
            modelSpec.activeParams ?? modelSpec.totalParams
          );
        }
        syncSnapshot(draft);
      });
      useSimulationStore.getState().reset();
    },

    deleteCustomModel: (id: string) => {
      modelRegistry.remove(id);
      set(draft => {
        delete draft.customModels[id];
        if (draft.modelId === id) {
          draft.modelId = initialState.modelId;
          draft.modelSpec = getModel(initialState.modelId, draft.sequenceLength) ?? null;
        }
        if (draft.inference.draftModelId === id) {
          draft.inference.draftModelId = null;
          draft.inference.draftModelSpec = null;
        }
      });
      useSimulationStore.getState().reset();
    },

    // Shared cluster configuration
    setCluster: (clusterId: string) => {
      const clusterConfig = getPresetCluster(clusterId);
      if (clusterConfig) {
        set(state => {
          // Freeze current rate when GPU changes so price doesn't jump
          if (state.pricePerGPUHour === null && clusterConfig.node.gpu.id !== state.gpuId) {
            state.pricePerGPUHour = getGPUHourlyRate(state.gpuId).rate;
          }
          state.clusterId = clusterId;
          state.clusterConfig = clusterConfig;
          state.numGPUs = clusterConfig.totalGPUs;
          state.gpusPerNode = clusterConfig.gpusPerNode;
          state.gpuId = clusterConfig.node.gpu.id;
          state.training.dpDegree = Math.floor(clusterConfig.totalGPUs / (state.training.tpDegree * state.training.ppDegree * state.training.cpDegree));
          // Recalculate gradient accumulation steps based on DP degree (not total GPUs)
          // GA = globalBatchSize / (microBatchSize * dpDegree)
          const newGA = Math.ceil(
            state.training.globalBatchSize / (state.training.microBatchSize * state.training.dpDegree)
          );
          state.training.gradientAccumulationSteps = Math.max(1, newGA);
          state.training.numMicroBatches = state.training.gradientAccumulationSteps;
          // Auto-disable Flash Attention on incompatible GPUs
          if (!supportsFlashAttention(clusterConfig.node.gpu)) {
            state.training.flashAttention = false;
            state.inference.flashAttention = false;
          }
          syncSnapshot(state);
        });
        // Clear stale simulation results
        useSimulationStore.getState().reset();
      }
    },

    setCustomCluster: (gpuId: string, numGPUs: number, gpusPerNode: number) => {
      if (numGPUs <= 0 || gpusPerNode <= 0) return; // Silently ignore invalid
      gpusPerNode = Math.min(gpusPerNode, numGPUs); // Can't have more GPUs/node than total
      const numNodes = Math.ceil(numGPUs / gpusPerNode);
      const clusterConfig = createMultiNodeCluster(gpuId, gpusPerNode, numNodes);
      if (clusterConfig) {
        set(state => {
          // Freeze current rate when GPU changes so price doesn't jump
          if (state.pricePerGPUHour === null && gpuId !== state.gpuId) {
            state.pricePerGPUHour = getGPUHourlyRate(state.gpuId).rate;
          }
          state.clusterId = 'custom';
          state.clusterConfig = clusterConfig;
          state.gpuId = gpuId;
          state.numGPUs = numGPUs;
          state.gpusPerNode = gpusPerNode;
          state.training.dpDegree = Math.floor(numGPUs / (state.training.tpDegree * state.training.ppDegree * state.training.cpDegree));
          // Recalculate gradient accumulation steps based on DP degree (not total GPUs)
          // GA = globalBatchSize / (microBatchSize * dpDegree)
          const newGA = Math.ceil(
            state.training.globalBatchSize / (state.training.microBatchSize * state.training.dpDegree)
          );
          state.training.gradientAccumulationSteps = Math.max(1, newGA);
          state.training.numMicroBatches = state.training.gradientAccumulationSteps;
          // Auto-disable Flash Attention on incompatible GPUs
          if (!supportsFlashAttention(clusterConfig.node.gpu)) {
            state.training.flashAttention = false;
            state.inference.flashAttention = false;
          }
          syncSnapshot(state);
        });
        // Clear stale simulation results
        useSimulationStore.getState().reset();
      }
    },

    setSequenceLength: (length: number) => {
      if (length <= 0) return; // Silently ignore invalid values from UI
      set(state => {
        state.sequenceLength = length;
        // Update model spec with new sequence length
        if (state.modelId) {
          state.modelSpec = getModel(state.modelId, length) ?? null;
        }
        syncSnapshot(state);
      });
    },

    setPrecision: (precision: 'fp32' | 'tf32' | 'fp16' | 'bf16' | 'fp8' | 'fp4') => {
      set(state => {
        state.precision = precision;
        // Sync to inference config
        state.inference.weightPrecision = precision as InferencePrecision;
        state.inference.kvCachePrecision = precision as InferencePrecision;
        syncSnapshot(state);
      });
    },

    // Training configuration
    setTrainingParams: (params: Partial<TrainingConfig>) => {
      // Silently ignore invalid values from UI
      if (params.globalBatchSize !== undefined && params.globalBatchSize <= 0) return;
      if (params.microBatchSize !== undefined && params.microBatchSize <= 0) return;
      set(state => {
        if (params.globalBatchSize !== undefined) {
          state.training.globalBatchSize = params.globalBatchSize;
        }
        if (params.microBatchSize !== undefined) {
          state.training.microBatchSize = params.microBatchSize;
        }
        if (params.gradientAccumulationSteps !== undefined) {
          state.training.gradientAccumulationSteps = params.gradientAccumulationSteps;
        }
        if (params.activationCheckpointing !== undefined) {
          state.training.activationCheckpointing = params.activationCheckpointing;
        }
        if (params.checkpointingGranularity !== undefined) {
          state.training.checkpointingGranularity = params.checkpointingGranularity;
        }
        if (params.selectiveStoredLayers !== undefined) {
          state.training.selectiveStoredLayers = params.selectiveStoredLayers;
        }
        if (params.flashAttention !== undefined) {
          state.training.flashAttention = params.flashAttention;
        }
        if (params.sequenceParallel !== undefined) {
          state.training.sequenceParallel = params.sequenceParallel;
        }
        if (params.finetuningMethod !== undefined) {
          state.training.finetuningMethod = params.finetuningMethod;
        }
        if (params.loraRank !== undefined) {
          state.training.loraRank = params.loraRank;
        }
        if (params.loraTargetModules !== undefined) {
          state.training.loraTargetModules = params.loraTargetModules;
        }
        // Recalculate gradient accumulation if batch sizes changed and GA wasn't explicitly set
        // Use DP degree (not total GPUs) to account for TP/PP parallelism
        if ((params.globalBatchSize !== undefined || params.microBatchSize !== undefined) &&
            params.gradientAccumulationSteps === undefined) {
          const newGA = Math.ceil(
            state.training.globalBatchSize / (state.training.microBatchSize * state.training.dpDegree)
          );
          state.training.gradientAccumulationSteps = Math.max(1, newGA);
          state.training.numMicroBatches = state.training.gradientAccumulationSteps;
        }
      });
    },

    setStrategy: (strategyType: TrainingConfig['strategyType']) => {
      set(state => {
        const prevStrategy = state.training.strategyType;
        state.training.strategyType = strategyType;

        // Helper to recalculate GA based on DP degree
        const recalcGA = () => {
          const newGA = Math.ceil(state.training.globalBatchSize / (state.training.microBatchSize * state.training.dpDegree));
          state.training.gradientAccumulationSteps = Math.max(1, newGA);
          state.training.numMicroBatches = state.training.gradientAccumulationSteps;
        };

        // Helper: auto-select EP for MoE models in hybrid strategies
        // Picks the largest divisor of numExperts that fits in gpusPerNode / tp
        const autoSelectEP = (tp: number, pp: number): number => {
          const model = state.modelSpec;
          if (!model || !model.isMoE || !model.numExperts) return 1;
          const numExperts = model.numExperts;
          const maxEP = Math.floor(state.gpusPerNode / tp);
          const dp = Math.floor(state.numGPUs / (tp * pp));
          let bestEP = 1;
          for (let ep = maxEP; ep >= 1; ep--) {
            if (numExperts % ep === 0 && dp % ep === 0) { bestEP = ep; break; }
          }
          return bestEP;
        };

        // Determine strategy families
        const isHybrid = (st: string) =>
          ['fsdp-tp', 'zero1-tp', 'ddp-tp-pp', 'zero1-tp-pp', 'fsdp-tp-pp'].includes(st);
        const is3D = (st: string) =>
          ['ddp-tp-pp', 'zero1-tp-pp', 'fsdp-tp-pp'].includes(st);

        // When switching between hybrid strategies, preserve user-set TP/PP/EP.
        // Only auto-select when coming from a 1D strategy (where TP=1, PP=1 are invalid defaults).
        const preserve = isHybrid(prevStrategy) && isHybrid(strategyType);

        // Adjust parallelism degrees based on strategy
        if (strategyType === 'ddp' || strategyType === 'fsdp' || strategyType === 'zero-1' || strategyType === 'zero-3') {
          // 1D strategies: reset all parallelism (keep CP if set)
          state.training.tpDegree = 1;
          state.training.ppDegree = 1;
          state.training.epDegree = 1;
          state.training.cpDegree = 1;
          state.training.dpDegree = state.numGPUs;
          state.training.dpType = DP_TYPE_MAP[strategyType] ?? 'ddp';
          recalcGA();
        } else {
          // Hybrid strategy (2D or 3D)
          let tp: number, pp: number, ep: number;

          if (preserve) {
            // Preserve user-set TP
            tp = state.training.tpDegree;
            if (tp < 1 || tp > state.numGPUs) tp = state.gpusPerNode;

            // Preserve PP if new strategy is 3D; reset to 1 for 2D
            if (is3D(strategyType)) {
              pp = state.training.ppDegree;
              if (pp < 1 || tp * pp > state.numGPUs) pp = Math.min(4, Math.floor(state.numGPUs / tp));
            } else {
              pp = 1;
            }

            // Preserve EP, validate it still divides DP
            ep = state.training.epDegree;
            const dp = Math.floor(state.numGPUs / (tp * pp));
            const model = state.modelSpec;
            const numExperts = model?.numExperts ?? 0;
            if (ep < 1 || dp % ep !== 0 || (numExperts > 0 && numExperts % ep !== 0)) {
              ep = autoSelectEP(tp, pp);
            }
          } else {
            // Coming from 1D: auto-select sensible defaults
            tp = state.gpusPerNode;
            pp = is3D(strategyType) ? Math.min(4, Math.floor(state.numGPUs / tp)) : 1;
            ep = autoSelectEP(tp, pp);
          }

          state.training.tpDegree = tp;
          state.training.ppDegree = pp;
          state.training.epDegree = ep;
          state.training.cpDegree = 1;
          state.training.dpDegree = Math.max(1, Math.floor(state.numGPUs / (tp * pp)));
          state.training.dpType = DP_TYPE_MAP[strategyType] ?? 'fsdp';
          state.training.sequenceParallel = true;
          recalcGA();
        }
      });
      // Clear stale simulation results
      useSimulationStore.getState().reset();
    },

    setStrategyParams: (params: Partial<TrainingConfig>) => {
      // Silently ignore invalid values from UI
      if (params.tpDegree !== undefined && params.tpDegree <= 0) return;
      if (params.ppDegree !== undefined && params.ppDegree <= 0) return;
      if (params.cpDegree !== undefined && params.cpDegree <= 0) return;
      if (params.epDegree !== undefined && params.epDegree <= 0) return;
      set(state => {
        if (params.tpDegree !== undefined) {
          state.training.tpDegree = params.tpDegree;
        }
        if (params.ppDegree !== undefined) {
          state.training.ppDegree = params.ppDegree;
        }
        if (params.dpDegree !== undefined) {
          state.training.dpDegree = params.dpDegree;
        }
        if (params.epDegree !== undefined) {
          state.training.epDegree = params.epDegree;
        }
        if (params.cpDegree !== undefined) {
          state.training.cpDegree = params.cpDegree;
        }
        if (params.cpImplementation !== undefined) {
          state.training.cpImplementation = params.cpImplementation;
        }
        if (params.numMicroBatches !== undefined) {
          state.training.numMicroBatches = params.numMicroBatches;
        }
        if (params.sequenceParallel !== undefined) {
          state.training.sequenceParallel = params.sequenceParallel;
        }
        if (params.dpType !== undefined) {
          state.training.dpType = params.dpType;
        }
        if (params.pipelineSchedule !== undefined) {
          state.training.pipelineSchedule = params.pipelineSchedule;
        }
        if (params.interleavedStages !== undefined) {
          state.training.interleavedStages = params.interleavedStages;
        }
        // Recalculate DP and GA when parallelism degrees change
        if (params.cpDegree !== undefined || params.epDegree !== undefined || params.tpDegree !== undefined || params.ppDegree !== undefined) {
          const tp = state.training.tpDegree;
          const pp = state.training.ppDegree;
          const cp = state.training.cpDegree;
          state.training.dpDegree = Math.floor(state.numGPUs / (tp * pp * cp));
          // Clamp DP to 1 minimum — UI allows incremental TP/PP changes that
          // temporarily create invalid states; validate() catches real errors.
          if (state.training.dpDegree <= 0) state.training.dpDegree = 1;
          // Don't clamp EP here — let the user set any value freely.
          // Strategy validation reports "EP must divide DP" if invalid.
          // Recalc GA and sync numMicroBatches (defensive Math.max on dp for belt-and-suspenders)
          const dp = Math.max(1, state.training.dpDegree);
          const newGA = Math.max(1, Math.ceil(state.training.globalBatchSize / (state.training.microBatchSize * dp)));
          state.training.gradientAccumulationSteps = newGA;
          state.training.numMicroBatches = newGA;
        }
      });
    },

    // Training goal and token target
    setTrainingGoal: (goal: TrainingGoal) => {
      set(state => {
        state.training.trainingGoal = goal;
        // Recalculate target tokens based on model size
        if (state.modelSpec) {
          state.training.targetTokens = calculateTargetTokens(goal, state.modelSpec.activeParams ?? state.modelSpec.totalParams);
        }
      });
    },

    setTargetTokens: (tokens: number) => {
      set(state => {
        state.training.targetTokens = tokens;
        state.training.trainingGoal = 'custom'; // Switch to custom when manually set
      });
    },

    // Inference configuration
    setInferenceParams: (params: Partial<InferenceConfigState>) => {
      set(state => {
        const clean = { ...params };
        if (clean.batchSize !== undefined && clean.batchSize < 1) delete clean.batchSize;
        if (clean.inputSeqLen !== undefined && clean.inputSeqLen < 1) delete clean.inputSeqLen;
        if (clean.outputSeqLen !== undefined && clean.outputSeqLen < 1) delete clean.outputSeqLen;
        if (clean.tensorParallel !== undefined && clean.tensorParallel < 1) delete clean.tensorParallel;
        if (clean.expertParallel !== undefined && clean.expertParallel < 1) delete clean.expertParallel;
        if (clean.numSpeculativeTokens !== undefined && clean.numSpeculativeTokens < 1) delete clean.numSpeculativeTokens;
        Object.assign(state.inference, clean);
      });
    },

    setDraftModel: (modelId: string | null) => {
      if (modelId === null) {
        set(state => {
          state.inference.draftModelId = null;
          state.inference.draftModelSpec = null;
        });
      } else {
        const seqLen = get().inference.inputSeqLen;
        const draftModelSpec = getModel(modelId, seqLen);
        set(state => {
          state.inference.draftModelId = modelId;
          state.inference.draftModelSpec = draftModelSpec ?? null;
        });
      }
    },

    setSpeculativeDecoding: (enabled: boolean) => {
      set(state => {
        state.inference.speculativeDecoding = enabled;
      });
    },

    setPricePerGPUHour: (price: number | null) => {
      set(state => {
        state.pricePerGPUHour = price;
        syncSnapshot(state);
      });
    },

    reset: () => {
      const currentMode = get().mode;
      withPreservedCustomModels(get, set, () => {
        if (currentMode === 'inference') {
          inferencePresetIndex = (inferencePresetIndex + 1) % INFERENCE_DEMO_PRESETS.length;
          set(() => applyInferencePreset(INFERENCE_DEMO_PRESETS[inferencePresetIndex]));
        } else {
          trainingPresetIndex = (trainingPresetIndex + 1) % DEMO_PRESETS.length;
          set(() => applyTrainingPreset(DEMO_PRESETS[trainingPresetIndex]));
        }
      });
      useSimulationStore.getState().reset();
    },

    resetPrev: () => {
      const currentMode = get().mode;
      withPreservedCustomModels(get, set, () => {
        if (currentMode === 'inference') {
          inferencePresetIndex = (inferencePresetIndex - 1 + INFERENCE_DEMO_PRESETS.length) % INFERENCE_DEMO_PRESETS.length;
          set(() => applyInferencePreset(INFERENCE_DEMO_PRESETS[inferencePresetIndex]));
        } else {
          trainingPresetIndex = (trainingPresetIndex - 1 + DEMO_PRESETS.length) % DEMO_PRESETS.length;
          set(() => applyTrainingPreset(DEMO_PRESETS[trainingPresetIndex]));
        }
      });
      useSimulationStore.getState().reset();
    },

    loadPresetBySlug: (slug: string) => {
      const match = findPresetBySlug(slug);
      if (!match) return;
      withPreservedCustomModels(get, set, () => {
        if (match.mode === 'inference') {
          set(() => applyInferencePreset(match.preset as InferenceDemoPreset));
        } else {
          set(() => applyTrainingPreset(match.preset as DemoPreset));
        }
      });
      useSimulationStore.getState().reset();
    },

    loadShareConfig: (config: ShareConfig) => {
      withPreservedCustomModels(get, set, () => {
      if (config.mode === 't') {
        // Training mode — atomic state load
        const modelId = (config.model && getModel(config.model)) ? config.model : initialState.modelId;
        const gpuId = (config.gpu && getGPU(config.gpu)) ? config.gpu : initialState.gpuId;
        const sequenceLength = config.seq || initialState.sequenceLength;
        const numGPUs = config.n || initialState.numGPUs;
        const gpusPerNode = config.gpn || initialState.gpusPerNode;
        const precision = validPrecisions.has(config.pr) ? config.pr as typeof initialState.precision : initialState.precision;
        const strategyType = validStrategyIds.has(config.st)
          ? config.st as TrainingConfig['strategyType']
          : initialTrainingConfig.strategyType;
        const trainingGoal = validTrainingGoals.has(config.goal)
          ? config.goal as TrainingGoal
          : initialTrainingConfig.trainingGoal;
        const tok = Number(config.tok);
        const targetTokens = (Number.isFinite(tok) && tok > 0) ? tok : initialTrainingConfig.targetTokens;

        const modelSpec = getModel(modelId, sequenceLength) ?? null;
        const numNodes = Math.ceil(numGPUs / gpusPerNode);
        const clusterConfig = createMultiNodeCluster(gpuId, gpusPerNode, numNodes) ?? null;

        // Share URLs come from external sources — degrade gracefully with Math.max
        const tp = Math.max(1, config.tp || 1);
        const pp = Math.max(1, config.pp || 1);
        const cp = Math.max(1, config.cp || 1);
        const dp = Math.max(1, config.dp || Math.floor(numGPUs / (tp * pp * cp)));
        const ep = Math.max(1, config.ep || 1);
        const gbs = Math.max(1, config.gbs || initialTrainingConfig.globalBatchSize);
        const mbs = Math.max(1, config.mbs || initialTrainingConfig.microBatchSize);
        const ga = Math.max(1, Math.ceil(gbs / (mbs * dp)));

        // Determine dpType from strategy

        const training: TrainingConfig = {
          globalBatchSize: gbs,
          microBatchSize: mbs,
          gradientAccumulationSteps: ga,
          trainingGoal,
          targetTokens,
          strategyType,
          tpDegree: tp,
          ppDegree: pp,
          dpDegree: dp,
          epDegree: ep,
          cpDegree: cp,
          cpImplementation: (config.cpi === 'a' ? 'all-gather' : 'ring') as CPImplementationType,
          numMicroBatches: ga,
          activationCheckpointing: config.ckpt ?? true,
          checkpointingGranularity: (config.acg === 's' ? 'selective' : 'full') as 'full' | 'selective',
          selectiveStoredLayers: config.sl != null ? config.sl : 'auto',
          flashAttention: config.fa ?? true,
          sequenceParallel: config.sp ?? false,
          dpType: DP_TYPE_MAP[strategyType] ?? 'fsdp',
          pipelineSchedule: (config.ps === 'interleaved-1f1b' ? 'interleaved-1f1b' : config.ps === 'dualpipe-v' ? 'dualpipe-v' : '1f1b') as PipelineScheduleType,
          interleavedStages: config.is || 2,
          finetuningMethod: (config.ft === 'lora' || config.ft === 'qlora' ? config.ft : 'full') as FinetuningMethod,
          loraRank: [4, 8, 16, 32, 64].includes(config.lr ?? 0) ? config.lr! : 16,
          loraTargetModules: (config.ltm === 'q_v' || config.ltm === 'q_k_v_o' || config.ltm === 'all_linear' ? config.ltm : 'q_k_v_o') as LoraTargetModules,
          savedHardware: {
            modelId, clusterId: 'custom', gpuId, numGPUs, gpusPerNode,
            sequenceLength, precision, pricePerGPUHour: config.price,
          },
        };

        set(() => ({
          ...initialState,
          mode: 'training' as AppMode,
          modelId,
          modelSpec,
          clusterId: 'custom',
          clusterConfig,
          gpuId,
          numGPUs,
          gpusPerNode,
          sequenceLength,
          precision,
          pricePerGPUHour: config.price,
          training,
          inference: initialInferenceConfig,
        }));
      } else {
        // Inference mode — atomic state load
        const modelId = (config.model && getModel(config.model)) ? config.model : initialState.modelId;
        const gpuId = (config.gpu && getGPU(config.gpu)) ? config.gpu : initialState.gpuId;
        const numGPUs = config.n || 1;
        const gpusPerNode = config.gpn || 8;

        const modelSpec = getModel(modelId) ?? null;
        const numNodes = Math.ceil(numGPUs / gpusPerNode);
        const clusterConfig = createMultiNodeCluster(gpuId, gpusPerNode, numNodes) ?? null;

        const weightPrecision = validInferencePrecisions.has(config.wpr)
          ? config.wpr as InferencePrecision
          : initialInferenceConfig.weightPrecision;
        const kvCachePrecision = validInferencePrecisions.has(config.kvpr)
          ? config.kvpr as InferencePrecision
          : initialInferenceConfig.kvCachePrecision;

        // Rehydrate draft model spec if speculative decoding
        const draftModelSpec = config.sd && config.dm
          ? getModel(config.dm, config.iseq || initialInferenceConfig.inputSeqLen) ?? null
          : null;

        const inference: InferenceConfigState = {
          ...initialInferenceConfig,
          batchSize: config.bs || initialInferenceConfig.batchSize,
          inputSeqLen: config.iseq || initialInferenceConfig.inputSeqLen,
          outputSeqLen: config.oseq || initialInferenceConfig.outputSeqLen,
          weightPrecision,
          kvCachePrecision,
          flashAttention: config.fa ?? true,
          pagedAttention: config.pa ?? true,
          continuousBatching: config.cb ?? false,
          tensorParallel: config.tp || 1,
          expertParallel: config.ep || 1,
          speculativeDecoding: config.sd ?? false,
          draftModelId: config.dm ?? null,
          draftModelSpec,
          numSpeculativeTokens: config.nst || initialInferenceConfig.numSpeculativeTokens,
          acceptanceRate: config.ar || initialInferenceConfig.acceptanceRate,
          savedHardware: {
            modelId, clusterId: 'custom', gpuId, numGPUs, gpusPerNode,
            sequenceLength: initialState.sequenceLength,
            precision: weightPrecision as typeof initialState.precision,
            pricePerGPUHour: config.price,
          },
        };

        set(() => ({
          ...initialState,
          mode: 'inference' as AppMode,
          modelId,
          modelSpec,
          clusterId: 'custom',
          clusterConfig,
          gpuId,
          numGPUs,
          gpusPerNode,
          sequenceLength: initialState.sequenceLength,
          precision: weightPrecision as typeof initialState.precision,
          pricePerGPUHour: config.price,
          training: initialTrainingConfig,
          inference,
        }));
      }
      }); // end withPreservedCustomModels
      useSimulationStore.getState().reset();
    },

    restoreFromSnapshot: (json: string) => {
      const rehydrated = rehydrateFromJSON(json);
      set(() => rehydrated);
      // Persist so localStorage stays in sync
      saveState(useConfigStore.getState());
      useSimulationStore.getState().reset();
    },
  }))
);

// Persist every state change to localStorage
useConfigStore.subscribe(saveState);
