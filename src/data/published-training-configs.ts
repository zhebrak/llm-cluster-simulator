/**
 * Canonical source of truth for published training benchmark configurations.
 *
 * Every test file and script that needs a published benchmark config should
 * import from here. This prevents drift between duplicate inline definitions.
 *
 * Tier 1: authoritative first-party published MFU numbers
 * Tier 2: third-party or indirect MFU numbers
 */

import { type SimulationConfig } from '../core/simulation/engine.ts';
import { createMultiNodeCluster } from '../core/hardware/topology.ts';

// ---------------------------------------------------------------------------
// Interface
// ---------------------------------------------------------------------------

export interface PublishedTrainingConfig {
  id: string;                // e.g. 'gpt3-175b'
  label: string;             // e.g. 'GPT-3 175B'
  tier: 1 | 2;
  // Cluster
  gpuId: string;
  gpusPerNode: number;
  numNodes: number;
  // Training
  modelId: string;
  sequenceLength: number;
  strategyType: SimulationConfig['strategyType'];
  strategyConfig: SimulationConfig['strategyConfig'];
  globalBatchSize: number;
  microBatchSize: number;
  activationCheckpointing: boolean;
  checkpointingGranularity?: 'full' | 'selective';
  flashAttention: boolean;
  mixedPrecision: SimulationConfig['mixedPrecision'];
  // Published reference
  published: {
    mfu?: number;           // numeric, e.g. 0.442
    mfuLabel: string;       // display, e.g. '44.2%' or '~40%' or '41-42%'
    source: string;         // citation text
    sourceUrl?: string;     // arxiv/blog URL
  };
}

// ---------------------------------------------------------------------------
// Helper: convert to SimulationConfig
// ---------------------------------------------------------------------------

export function toSimConfig(pc: PublishedTrainingConfig): SimulationConfig {
  return {
    clusterConfig: createMultiNodeCluster(pc.gpuId, pc.gpusPerNode, pc.numNodes)!,
    modelId: pc.modelId,
    sequenceLength: pc.sequenceLength,
    strategyType: pc.strategyType,
    strategyConfig: pc.strategyConfig,
    globalBatchSize: pc.globalBatchSize,
    microBatchSize: pc.microBatchSize,
    activationCheckpointing: pc.activationCheckpointing,
    checkpointingGranularity: pc.checkpointingGranularity,
    flashAttention: pc.flashAttention,
    mixedPrecision: pc.mixedPrecision,
  };
}

// ---------------------------------------------------------------------------
// Tier 1 — authoritative first-party published MFU
// ---------------------------------------------------------------------------

/** LLaMA 3.1 405B 8K — Meta (2024), 16384 H100, ~40% MFU */
export const LLAMA3_405B_8K: PublishedTrainingConfig = {
  id: 'llama3-405b-8k',
  label: 'LLaMA 3.1 405B 8K',
  tier: 1,
  gpuId: 'h100-sxm', gpusPerNode: 8, numNodes: 2048,
  modelId: 'llama3-405b', sequenceLength: 8192,
  strategyType: 'fsdp-tp-pp',
  strategyConfig: {
    tp: 8, pp: 16, sequenceParallel: true,
    pipelineSchedule: 'interleaved-1f1b', interleavedStages: 4,
  },
  globalBatchSize: 2048, microBatchSize: 1,
  activationCheckpointing: false,
  flashAttention: true,
  mixedPrecision: 'bf16',
  published: {
    mfu: 0.40,
    mfuLabel: '~40%',
    source: 'Meta LLaMA 3 paper (2024) §3.3.2, Table 4',
    sourceUrl: 'https://arxiv.org/abs/2407.21783',
  },
};

/** LLaMA 3.1 405B 131K — Meta (2024), 16384 H100, CP=16, 38% Model FLOPs MFU */
export const LLAMA3_405B_131K: PublishedTrainingConfig = {
  id: 'llama3-405b-131k',
  label: 'LLaMA 3.1 405B 131K',
  tier: 1,
  gpuId: 'h100-sxm', gpusPerNode: 8, numNodes: 2048,
  modelId: 'llama3-405b', sequenceLength: 131072,
  strategyType: 'fsdp-tp-pp',
  strategyConfig: {
    tp: 8, pp: 16, cp: 16, cpImplementation: 'all-gather',
    sequenceParallel: true,
    pipelineSchedule: 'interleaved-1f1b', interleavedStages: 4,
  },
  globalBatchSize: 2048, microBatchSize: 1,
  activationCheckpointing: true,
  flashAttention: true,
  mixedPrecision: 'bf16',
  published: {
    mfu: 0.38,
    mfuLabel: '38%\\*',
    source: 'Meta LLaMA 3 paper (2024) §3.3.2, Table 4',
    sourceUrl: 'https://arxiv.org/abs/2407.21783',
  },
};

/** DeepSeek V3 671B FP8 — DeepSeek (2024), 2048 H800, 43.7% MFU */
export const DEEPSEEK_V3_FP8_H800: PublishedTrainingConfig = {
  id: 'deepseek-v3-fp8-h800',
  label: 'DeepSeek V3 671B FP8',
  tier: 1,
  gpuId: 'h800-sxm', gpusPerNode: 8, numNodes: 256,
  modelId: 'deepseek-v3', sequenceLength: 4096,
  strategyType: 'fsdp-tp-pp',
  strategyConfig: {
    tp: 4, pp: 8, ep: 32, sequenceParallel: true,
    pipelineSchedule: 'dualpipe-v', numMicroBatches: 64,
  },
  globalBatchSize: 8192, microBatchSize: 2,
  activationCheckpointing: true,
  flashAttention: true,
  mixedPrecision: 'fp8',
  published: {
    mfu: 0.437,
    mfuLabel: '43.7%',
    source: 'DeepSeek V3 paper (2024) §3.1; MFU from ISCA 2025 Table 4',
    sourceUrl: 'https://arxiv.org/abs/2412.19437',
  },
};

/** Nemotron-4 340B — NVIDIA (2024), 6144 H100, 41-42% MFU */
export const NEMOTRON_4_340B: PublishedTrainingConfig = {
  id: 'nemotron-4-340b',
  label: 'Nemotron-4 340B',
  tier: 1,
  gpuId: 'h100-sxm', gpusPerNode: 8, numNodes: 768,
  modelId: 'nemotron-4-340b', sequenceLength: 4096,
  strategyType: 'fsdp-tp-pp',
  strategyConfig: {
    tp: 8, pp: 12, sequenceParallel: true,
    pipelineSchedule: 'interleaved-1f1b', interleavedStages: 8,
  },
  globalBatchSize: 768, microBatchSize: 1,
  activationCheckpointing: true,
  checkpointingGranularity: 'selective',
  flashAttention: true,
  mixedPrecision: 'bf16',
  published: {
    mfu: 0.41,
    mfuLabel: '41-42%',
    source: 'Megatron-Core blog (2024) Table 2',
    sourceUrl: 'https://developer.nvidia.com/blog/train-generative-ai-models-more-efficiently-with-new-nvidia-megatron-core-functionalities/',
  },
};

/**
 * GPT-3 175B — Narayanan et al. (2021), 1024 A100, 44.2% MFU
 *
 * Predates Flash Attention (2022) and Sequence Parallelism (2022).
 * Uses interleaved 1F1B with v=2 virtual stages per device.
 */
export const GPT3_175B: PublishedTrainingConfig = {
  id: 'gpt3-175b',
  label: 'GPT-3 175B',
  tier: 1,
  gpuId: 'a100-80gb', gpusPerNode: 8, numNodes: 128,
  modelId: 'gpt3-175b', sequenceLength: 2048,
  strategyType: 'ddp-tp-pp',
  strategyConfig: {
    tp: 8, pp: 8,
    sequenceParallel: false,
    pipelineSchedule: 'interleaved-1f1b', interleavedStages: 2,
  },
  globalBatchSize: 1536, microBatchSize: 1,
  activationCheckpointing: true,
  flashAttention: false,
  mixedPrecision: 'bf16',
  published: {
    mfu: 0.442,
    mfuLabel: '44.2%',
    source: 'Narayanan et al. (2021) §5.1',
    sourceUrl: 'https://arxiv.org/abs/2104.04473',
  },
};

/** IBM FSDP LLaMA 2 7B — IBM Research (2023), 128 A100, 57% MFU */
export const IBM_LLAMA2_7B: PublishedTrainingConfig = {
  id: 'ibm-llama2-7b',
  label: 'IBM LLaMA 2 7B',
  tier: 1,
  gpuId: 'a100-80gb', gpusPerNode: 8, numNodes: 16,
  modelId: 'llama2-7b', sequenceLength: 4096,
  strategyType: 'fsdp',
  strategyConfig: undefined,
  globalBatchSize: 256, microBatchSize: 2,
  activationCheckpointing: false,
  flashAttention: true,
  mixedPrecision: 'bf16',
  published: {
    mfu: 0.57,
    mfuLabel: '57%',
    source: 'IBM blog (2023)',
    sourceUrl: 'https://research.ibm.com/blog/pytorch-fsdp',
  },
};

/** OLMo 2 32B — AI2 (2025), 1280 H100, ~38% MFU */
export const OLMO2_32B: PublishedTrainingConfig = {
  id: 'olmo2-32b',
  label: 'OLMo 2 32B',
  tier: 2,
  gpuId: 'h100-sxm', gpusPerNode: 8, numNodes: 160,
  modelId: 'olmo2-32b', sequenceLength: 4096,
  strategyType: 'fsdp',
  strategyConfig: undefined,
  globalBatchSize: 2048, microBatchSize: 1,
  activationCheckpointing: true,
  checkpointingGranularity: 'selective',
  flashAttention: true,
  mixedPrecision: 'bf16',
  published: {
    mfu: 0.38,
    mfuLabel: '~38%',
    source: 'AI2 OLMo 2 blog (2025)',
    sourceUrl: 'https://allenai.org/blog/olmo2-32B',
  },
};

/** OLMo 3 32B — AI2 (2025), 1024 H100, ~41% MFU */
export const OLMO3_32B: PublishedTrainingConfig = {
  id: 'olmo3-32b',
  label: 'OLMo 3 32B',
  tier: 1,
  gpuId: 'h100-sxm', gpusPerNode: 8, numNodes: 128,
  modelId: 'olmo3-32b', sequenceLength: 8192,
  strategyType: 'fsdp',
  strategyConfig: undefined,
  globalBatchSize: 1024, microBatchSize: 1,
  activationCheckpointing: true,
  checkpointingGranularity: 'selective',
  flashAttention: true,
  mixedPrecision: 'bf16',
  published: {
    mfu: 0.41,
    mfuLabel: '~41%',
    source: 'OLMo 3 paper (2025)',
    sourceUrl: 'https://arxiv.org/abs/2512.13961',
  },
};

/**
 * BLOOM 176B — BigScience (2022), 384 A100-80GB, ~48% MFU (8PD convention)
 * First zero1-tp-pp anchor; ALiBi positional encoding, non-gated MLP
 * PP=12 on 70 layers: uneven division (5.83 layers/stage)
 * 150 TFLOPS from Megatron-LM 8PD formula / 312 peak = 48.1%
 */
export const BLOOM_176B: PublishedTrainingConfig = {
  id: 'bloom-176b',
  label: 'BLOOM 176B',
  tier: 1,
  gpuId: 'a100-80gb', gpusPerNode: 8, numNodes: 48,
  modelId: 'bloom-176b', sequenceLength: 2048,
  strategyType: 'zero1-tp-pp',
  strategyConfig: { tp: 4, pp: 12 },
  globalBatchSize: 2048, microBatchSize: 2,
  activationCheckpointing: true,
  flashAttention: false,
  mixedPrecision: 'bf16',
  published: {
    mfu: 0.48,
    mfuLabel: '~48%',
    source: 'BigScience (2022)',
    sourceUrl: 'https://arxiv.org/abs/2211.05100',
  },
};

/**
 * MT-NLG 530B — Microsoft/NVIDIA (2022), 2240 A100-80GB, ~40% MFU (8PD convention)
 * Largest model anchor (530B); deep PP=35; first-party MS/NVIDIA
 * PP=35 on 105 layers: 3 layers/stage (clean division)
 * 126 TFLOPS from Megatron-LM 8PD formula / 312 peak = 40.4%
 */
export const MT_NLG_530B: PublishedTrainingConfig = {
  id: 'mt-nlg-530b',
  label: 'MT-NLG 530B',
  tier: 2,
  gpuId: 'a100-80gb', gpusPerNode: 8, numNodes: 280,
  modelId: 'megatron-turing-530b', sequenceLength: 2048,
  strategyType: 'zero1-tp-pp',
  strategyConfig: { tp: 8, pp: 35 },
  globalBatchSize: 1920, microBatchSize: 1,
  activationCheckpointing: true,
  flashAttention: false,
  mixedPrecision: 'bf16',
  published: {
    mfu: 0.40,
    mfuLabel: '~40%',
    source: 'Smith et al. (2022)',
    sourceUrl: 'https://arxiv.org/abs/2201.11990',
  },
};

/**
 * Nemotron-4 15B DGXC — NVIDIA (2024), 64 H100, 56% MFU (genuine 6PD)
 * First fsdp-tp anchor; highest MFU target; NVIDIA first-party
 * TP=2, DP=32 (not TP=8 like the paper's production training)
 */
export const NEMOTRON_4_15B_DGXC: PublishedTrainingConfig = {
  id: 'nemotron-4-15b-dgxc',
  label: 'Nemotron-4 15B DGXC',
  tier: 1,
  gpuId: 'h100-sxm', gpusPerNode: 8, numNodes: 8,
  modelId: 'nemotron-4-15b', sequenceLength: 4096,
  strategyType: 'fsdp-tp',
  strategyConfig: { tp: 2 },
  globalBatchSize: 256, microBatchSize: 4,
  activationCheckpointing: true,
  checkpointingGranularity: 'selective',
  flashAttention: true,
  mixedPrecision: 'bf16',
  published: {
    mfu: 0.56,
    mfuLabel: '~56%',
    source: 'NVIDIA DGXC benchmarking (2024)',
    sourceUrl: 'https://github.com/NVIDIA/dgxc-benchmarking',
  },
};

// ---------------------------------------------------------------------------
// Tier 2 — third-party or indirect MFU numbers
// ---------------------------------------------------------------------------

/**
 * NeMo LLaMA 3.1 405B CP=2 — NeMo Framework, 1024 H100, 763 TFLOPS HFU
 *
 * Interleaved 1F1B with VP=2 per NVIDIA's Megatron Bridge recipe (PP=8, VP=2).
 * 126 layers / (8×2) = 7.875 — uneven but Megatron-Core handles unbalanced
 * virtual pipeline stages. Sim uses Math.ceil; validator emits a warning.
 */
export const NEMO_405B_CP2: PublishedTrainingConfig = {
  id: 'nemo-405b-cp2',
  label: 'NeMo 405B CP=2',
  tier: 2,
  gpuId: 'h100-sxm', gpusPerNode: 8, numNodes: 128,
  modelId: 'llama3-405b', sequenceLength: 8192,
  strategyType: 'fsdp-tp-pp',
  strategyConfig: {
    tp: 8, pp: 8, cp: 2, cpImplementation: 'all-gather',
    sequenceParallel: true,
    pipelineSchedule: 'interleaved-1f1b', interleavedStages: 2,
  },
  globalBatchSize: 512, microBatchSize: 1,
  activationCheckpointing: true,
  checkpointingGranularity: 'selective',
  flashAttention: true,
  mixedPrecision: 'bf16',
  published: {
    mfuLabel: '763 TFLOPS HFU',
    source: 'NeMo Framework published benchmarks',
  },
};

/**
 * Mixtral 8x22B EP=4 — MoE Parallel Folding (2025), 128 H100, 46.3% MFU
 * Megatron-Core baseline config.
 */
export const MIXTRAL_8X22B_EP4: PublishedTrainingConfig = {
  id: 'mixtral-8x22b-ep4',
  label: 'Mixtral 8x22B MoE',
  tier: 1,
  gpuId: 'h100-sxm', gpusPerNode: 8, numNodes: 16,
  modelId: 'mixtral-8x22b', sequenceLength: 4096,
  strategyType: 'fsdp-tp-pp',
  strategyConfig: {
    tp: 2, pp: 8, ep: 4, sequenceParallel: true,
  },
  globalBatchSize: 256, microBatchSize: 1,
  activationCheckpointing: true,
  flashAttention: true,
  mixedPrecision: 'bf16',
  published: {
    mfu: 0.463,
    mfuLabel: '46.3%',
    source: 'MoE Parallel Folding (2025)',
    sourceUrl: 'https://arxiv.org/abs/2504.14960',
  },
};

/**
 * Qwen2 MoE 57B-A14B EP=4 — MoE Parallel Folding (2025), 64 H100, 35.3% MFU
 * Megatron-Core baseline config.
 */
export const QWEN2_57B_A14B_EP4: PublishedTrainingConfig = {
  id: 'qwen2-57b-a14b-ep4',
  label: 'Qwen2 MoE 57B-A14B',
  tier: 2,
  gpuId: 'h100-sxm', gpusPerNode: 8, numNodes: 8,
  modelId: 'qwen2-57b-a14b', sequenceLength: 4096,
  strategyType: 'fsdp-tp-pp',
  strategyConfig: {
    tp: 2, pp: 4, ep: 4, sequenceParallel: true,
  },
  globalBatchSize: 256, microBatchSize: 1,
  activationCheckpointing: true,
  flashAttention: true,
  mixedPrecision: 'bf16',
  published: {
    mfu: 0.353,
    mfuLabel: '35.3%',
    source: 'MoE Parallel Folding (2025)',
    sourceUrl: 'https://arxiv.org/abs/2504.14960',
  },
};

/**
 * MT-NLG 530B at 350 nodes (DP=10) — ~39% MFU
 * DP scaling point for monotonicity test
 */
export const MT_NLG_530B_350: PublishedTrainingConfig = {
  id: 'mt-nlg-530b-350',
  label: 'MT-NLG 530B (350n)',
  tier: 2,
  gpuId: 'a100-80gb', gpusPerNode: 8, numNodes: 350,
  modelId: 'megatron-turing-530b', sequenceLength: 2048,
  strategyType: 'zero1-tp-pp',
  strategyConfig: { tp: 8, pp: 35 },
  globalBatchSize: 1920, microBatchSize: 1,
  activationCheckpointing: true,
  flashAttention: false,
  mixedPrecision: 'bf16',
  published: {
    mfu: 0.39,
    mfuLabel: '~39%',
    source: 'Smith et al. (2022)',
    sourceUrl: 'https://arxiv.org/abs/2201.11990',
  },
};

/**
 * MT-NLG 530B at 420 nodes (DP=12) — ~36% MFU
 * DP scaling point for monotonicity test
 */
export const MT_NLG_530B_420: PublishedTrainingConfig = {
  id: 'mt-nlg-530b-420',
  label: 'MT-NLG 530B (420n)',
  tier: 2,
  gpuId: 'a100-80gb', gpusPerNode: 8, numNodes: 420,
  modelId: 'megatron-turing-530b', sequenceLength: 2048,
  strategyType: 'zero1-tp-pp',
  strategyConfig: { tp: 8, pp: 35 },
  globalBatchSize: 1920, microBatchSize: 1,
  activationCheckpointing: true,
  flashAttention: false,
  mixedPrecision: 'bf16',
  published: {
    mfu: 0.36,
    mfuLabel: '~36%',
    source: 'Smith et al. (2022)',
    sourceUrl: 'https://arxiv.org/abs/2201.11990',
  },
};

/**
 * Mosaic LLaMA 70B — Databricks/MosaicML (2024), 512 H100, 41.25% MFU
 * FSDP-TP with TP=8, llm-foundry benchmark
 */
export const MOSAIC_70B: PublishedTrainingConfig = {
  id: 'mosaic-70b',
  label: 'Mosaic LLaMA 70B',
  tier: 2,
  gpuId: 'h100-sxm', gpusPerNode: 8, numNodes: 64,
  modelId: 'llama2-70b', sequenceLength: 2048,
  strategyType: 'fsdp-tp',
  strategyConfig: { tp: 8 },
  globalBatchSize: 1024, microBatchSize: 2,
  activationCheckpointing: true,
  flashAttention: true,
  mixedPrecision: 'bf16',
  published: {
    mfu: 0.4125,
    mfuLabel: '41.25%',
    source: 'MosaicML llm-foundry (2024)',
  },
};

/**
 * Qwen3 30B-A3B MoE EP=8 — NeMo Framework, 16 H100, 241 HFU TFLOPS
 * Exercises EP scaling (effectiveEp=8), small expert GEMM penalty
 * (expertIntermediateSize=768 < 1536), and expert count scaling
 * (128 experts / 8 EP = 16 experts/GPU > 8 threshold).
 */
export const QWEN3_30B_A3B_EP8: PublishedTrainingConfig = {
  id: 'qwen3-30b-a3b-ep8',
  label: 'Qwen3 30B-A3B MoE',
  tier: 2,
  gpuId: 'h100-sxm', gpusPerNode: 8, numNodes: 2,
  modelId: 'qwen3-30b-a3b', sequenceLength: 4096,
  strategyType: 'fsdp-tp-pp',
  strategyConfig: { tp: 1, pp: 2, ep: 8 },
  globalBatchSize: 512, microBatchSize: 2,
  activationCheckpointing: true,
  flashAttention: true,
  mixedPrecision: 'bf16',
  published: {
    mfuLabel: '241 HFU TFLOPS',
    source: 'NeMo Framework',
    sourceUrl: 'https://docs.nvidia.com/nemo/automodel',
  },
};

// ---------------------------------------------------------------------------
// All configs as an array and lookup maps
// ---------------------------------------------------------------------------

export const ALL_PUBLISHED_CONFIGS: PublishedTrainingConfig[] = [
  // Tier 1
  LLAMA3_405B_8K,
  LLAMA3_405B_131K,
  DEEPSEEK_V3_FP8_H800,
  NEMOTRON_4_340B,
  GPT3_175B,
  IBM_LLAMA2_7B,
  OLMO2_32B,
  OLMO3_32B,
  BLOOM_176B,
  MT_NLG_530B,
  NEMOTRON_4_15B_DGXC,
  // Tier 2
  NEMO_405B_CP2,
  MIXTRAL_8X22B_EP4,
  QWEN2_57B_A14B_EP4,
  MT_NLG_530B_350,
  MT_NLG_530B_420,
  MOSAIC_70B,
  QWEN3_30B_A3B_EP8,
];

/** Map from config ID to config object */
export const PUBLISHED_TRAINING_CONFIGS = new Map<string, PublishedTrainingConfig>(
  ALL_PUBLISHED_CONFIGS.map(c => [c.id, c]),
);

/** Convenience object for dot-access: PUBLISHED.gpt3_175b etc. */
export const PUBLISHED = {
  llama3_405b_8k: LLAMA3_405B_8K,
  llama3_405b_131k: LLAMA3_405B_131K,
  deepseek_v3_fp8_h800: DEEPSEEK_V3_FP8_H800,
  nemotron_4_340b: NEMOTRON_4_340B,
  gpt3_175b: GPT3_175B,
  ibm_llama2_7b: IBM_LLAMA2_7B,
  olmo2_32b: OLMO2_32B,
  olmo3_32b: OLMO3_32B,
  bloom_176b: BLOOM_176B,
  mt_nlg_530b: MT_NLG_530B,
  nemotron_4_15b_dgxc: NEMOTRON_4_15B_DGXC,
  nemo_405b_cp2: NEMO_405B_CP2,
  mixtral_8x22b_ep4: MIXTRAL_8X22B_EP4,
  qwen2_57b_a14b_ep4: QWEN2_57B_A14B_EP4,
  mt_nlg_530b_350: MT_NLG_530B_350,
  mt_nlg_530b_420: MT_NLG_530B_420,
  mosaic_70b: MOSAIC_70B,
  qwen3_30b_a3b_ep8: QWEN3_30B_A3B_EP8,
} as const;
