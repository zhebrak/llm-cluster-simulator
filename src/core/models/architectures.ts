/**
 * Built-in model architectures
 * Accurate specifications for popular LLM architectures
 */

import type { ModelConfig } from '../../types/index.ts';

/**
 * GPT-3 family models
 * Architecture similar to GPT-2 but larger scale
 */
export const GPT3_CONFIGS: Record<string, ModelConfig> = {
  'gpt3-125m': {
    name: 'GPT-3 (125M)',
    family: 'gpt3',
    numLayers: 12,
    hiddenSize: 768,
    intermediateSize: 3072,
    numAttentionHeads: 12,
    vocabSize: 50257,
    maxSeqLength: 2048,
    attentionType: 'mha',
    normType: 'layernorm',
    activation: 'gelu',
    gatedMLP: false,
    useBias: true,
    tiedEmbeddings: false,
    useRotaryEmbed: false,
  },
  'gpt3-1.3b': {
    name: 'GPT-3 (1.3B)',
    family: 'gpt3',
    numLayers: 24,
    hiddenSize: 2048,
    intermediateSize: 8192,
    numAttentionHeads: 16,
    vocabSize: 50257,
    maxSeqLength: 2048,
    attentionType: 'mha',
    normType: 'layernorm',
    activation: 'gelu',
    gatedMLP: false,
    useBias: true,
    tiedEmbeddings: false,
    useRotaryEmbed: false,
  },
  'gpt3-6.7b': {
    name: 'GPT-3 (6.7B)',
    family: 'gpt3',
    numLayers: 32,
    hiddenSize: 4096,
    intermediateSize: 16384,
    numAttentionHeads: 32,
    vocabSize: 50257,
    maxSeqLength: 2048,
    attentionType: 'mha',
    normType: 'layernorm',
    activation: 'gelu',
    gatedMLP: false,
    useBias: true,
    tiedEmbeddings: false,
    useRotaryEmbed: false,
  },
  'gpt3-13b': {
    name: 'GPT-3 (13B)',
    family: 'gpt3',
    numLayers: 40,
    hiddenSize: 5140,
    intermediateSize: 20560,
    numAttentionHeads: 40,
    vocabSize: 50257,
    maxSeqLength: 2048,
    attentionType: 'mha',
    normType: 'layernorm',
    activation: 'gelu',
    gatedMLP: false,
    useBias: true,
    tiedEmbeddings: false,
    useRotaryEmbed: false,
  },
  'gpt3-175b': {
    name: 'GPT-3 (175B)',
    family: 'gpt3',
    numLayers: 96,
    hiddenSize: 12288,
    intermediateSize: 49152,
    numAttentionHeads: 96,
    vocabSize: 50257,
    maxSeqLength: 2048,
    attentionType: 'mha',
    normType: 'layernorm',
    activation: 'gelu',
    gatedMLP: false,
    useBias: true,
    tiedEmbeddings: false,
    useRotaryEmbed: false,
  },
};

/**
 * OpenAI GPT-OSS family (August 2025)
 * Architecture: SwiGLU MoE, GQA(kv=8), RoPE+YaRN, RMSNorm, attention bias.
 * Non-standard headDim=64 (natural=2880/64=45): "wider attention" — Q projection
 * is 2880→4096, attention dim exceeds residual stream. Same pattern as MiniMax M2.5, GLM-4.5-Air.
 * intermediateSize = hiddenSize = expertIntermediateSize = 2880 — NOT a copy-paste error.
 * All layers are MoE (no firstKDenseLayers). Expert MLPs are intentionally narrow (2880)
 * because there are 128/32 experts; total MLP params come from expert count, not width.
 * Training SWA NOT modeled: alternating sliding (128-token window) and full attention.
 * Impact negligible with Flash Attention at typical training seqLength.
 * MXFP4-quantized MoE weights (4.25 bits) for inference; 60.8GB (120B), 12.8GB (20B).
 * Paper: 116.83B/5.13B active (120B), 20.91B/3.61B active (20B).
 * Source: arxiv.org/abs/2508.10925, HuggingFace config.json
 */
export const GPTOSS_CONFIGS: Record<string, ModelConfig> = {
  'gpt-oss-120b': {
    name: 'GPT-OSS (120B)',
    family: 'gpt-oss',
    numLayers: 36,
    hiddenSize: 2880,
    intermediateSize: 2880,        // = expertIntermediateSize (all layers MoE, no dense fallback)
    numAttentionHeads: 64,
    numKvHeads: 8,                 // GQA group size 8
    headDim: 64,                   // Non-standard: 2880/64=45, actual is 64 (wider attention)
    vocabSize: 201088,
    maxSeqLength: 131072,          // 128K context (YaRN RoPE extension from 4K base)
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,                // SwiGLU
    useBias: true,                 // attention_bias: true in HF config
    tiedEmbeddings: false,
    useRotaryEmbed: true,
    numExperts: 128,
    numActiveExperts: 4,
    expertIntermediateSize: 2880,  // Intentionally narrow: width × 128 experts = total MLP params
  },
  'gpt-oss-20b': {
    name: 'GPT-OSS (20B)',
    family: 'gpt-oss',
    numLayers: 24,
    hiddenSize: 2880,
    intermediateSize: 2880,        // = expertIntermediateSize (all layers MoE)
    numAttentionHeads: 64,
    numKvHeads: 8,
    headDim: 64,                   // Non-standard: wider attention (same as 120B)
    vocabSize: 201088,
    maxSeqLength: 131072,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: true,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
    numExperts: 32,
    numActiveExperts: 4,
    expertIntermediateSize: 2880,  // Intentionally narrow: width × 32 experts = total MLP params
  },
};

/**
 * LLaMA 2 family models
 * Same architecture as LLaMA 1 but with extended context
 */
export const LLAMA2_CONFIGS: Record<string, ModelConfig> = {
  'llama2-7b': {
    name: 'LLaMA-2 (7B)',
    family: 'llama2',
    numLayers: 32,
    hiddenSize: 4096,
    intermediateSize: 11008,
    numAttentionHeads: 32,
    vocabSize: 32000,
    maxSeqLength: 4096,
    attentionType: 'mha',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'llama2-13b': {
    name: 'LLaMA-2 (13B)',
    family: 'llama2',
    numLayers: 40,
    hiddenSize: 5120,
    intermediateSize: 13824,
    numAttentionHeads: 40,
    vocabSize: 32000,
    maxSeqLength: 4096,
    attentionType: 'mha',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'llama2-70b': {
    name: 'LLaMA-2 (70B)',
    family: 'llama2',
    numLayers: 80,
    hiddenSize: 8192,
    intermediateSize: 28672,
    numAttentionHeads: 64,
    numKvHeads: 8,  // GQA
    vocabSize: 32000,
    maxSeqLength: 4096,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
};

/**
 * LLaMA 3 family models
 * Features GQA for all sizes, larger vocab, longer context
 */
export const LLAMA3_CONFIGS: Record<string, ModelConfig> = {
  'llama3.1-8b': {
    name: 'LLaMA-3.1 (8B)',
    family: 'llama3',
    numLayers: 32,
    hiddenSize: 4096,
    intermediateSize: 14336,
    numAttentionHeads: 32,
    numKvHeads: 8,
    vocabSize: 128256,
    maxSeqLength: 131072,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'llama3-405b': {
    name: 'LLaMA-3.1 (405B)',
    family: 'llama3',
    numLayers: 126,
    hiddenSize: 16384,
    intermediateSize: 53248,
    numAttentionHeads: 128,
    numKvHeads: 8,
    vocabSize: 128256,
    maxSeqLength: 131072,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'llama3.2-1b': {
    name: 'LLaMA-3.2 (1B)',
    family: 'llama3',
    numLayers: 16,
    hiddenSize: 2048,
    intermediateSize: 8192,
    numAttentionHeads: 32,
    numKvHeads: 8,
    vocabSize: 128256,
    maxSeqLength: 131072,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: true,
    useRotaryEmbed: true,
  },
  'llama3.2-3b': {
    name: 'LLaMA-3.2 (3B)',
    family: 'llama3',
    numLayers: 28,
    hiddenSize: 3072,
    intermediateSize: 8192,
    numAttentionHeads: 24,
    numKvHeads: 8,
    vocabSize: 128256,
    maxSeqLength: 131072,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: true,
    useRotaryEmbed: true,
  },
  'llama3.3-70b': {
    name: 'LLaMA-3.3 (70B)',
    family: 'llama3',
    numLayers: 80,
    hiddenSize: 8192,
    intermediateSize: 28672,
    numAttentionHeads: 64,
    numKvHeads: 8,
    vocabSize: 128256,
    maxSeqLength: 131072,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
};

/**
 * Mistral / Mixtral family models
 * Dense models: GQA, RoPE. Configs represent v0.3+ releases which use full attention
 * (sliding_window: null). SWA is not modeled (only used in Mistral 7B v0.1).
 * MoE models (Mixtral): Sparse mixture of experts
 */
export const MISTRAL_CONFIGS: Record<string, ModelConfig> = {
  // Devstral 2 — Ministral 3 architecture with 131072-token tokenizer
  // Same arch as Mistral Large but 4x larger vocab (131072 vs 32768) → ~2.4B more embedding params
  'devstral-2': {
    name: 'Devstral 2 (123B)',
    family: 'mistral',
    numLayers: 88,
    hiddenSize: 12288,
    intermediateSize: 28672,
    numAttentionHeads: 96,
    numKvHeads: 8,
    vocabSize: 131072,          // Ministral 3 tokenizer (4x larger than Mistral Large)
    maxSeqLength: 262144,       // 256K context
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  // Devstral Small 2 — identical arch to Mistral Small 24B but 12x longer context
  'devstral-small-2': {
    name: 'Devstral Small 2 (24B)',
    family: 'mistral',
    numLayers: 40,
    hiddenSize: 5120,
    intermediateSize: 32768,
    numAttentionHeads: 32,
    numKvHeads: 8,
    headDim: 128,               // Non-standard: 5120/32=160, actual is 128
    vocabSize: 131072,
    maxSeqLength: 393216,       // 384K context
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  // Mistral Large 3 — DeepSeek V3 architecture clone with MLA + granular MoE
  // 61 layers (prime) — only standard 1F1B works with PP>1
  // n_heads=128 verified from HF params.json (same as DeepSeek V3)
  // Identical MLA dims as DeepSeek V3 (kvLoraRank, qLoraRank, qk dims, v_head_dim)
  // MoE: 128 experts / 4 active (vs V3's 256/8), expert dim 4096 (vs 2048) — fewer but fatter
  'mistral-large-3-675b': {
    name: 'Mistral Large 3 (675B)',
    family: 'mistral',
    numLayers: 61,
    hiddenSize: 7168,
    intermediateSize: 16384,
    numAttentionHeads: 128,
    numKvHeads: 128,
    headDim: 128,
    vocabSize: 131072,
    maxSeqLength: 262144,
    attentionType: 'mla',
    kvLoraRank: 512,
    qLoraRank: 1536,
    qkNopeHeadDim: 128,
    qkRopeHeadDim: 64,
    vHeadDim: 128,
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
    numExperts: 128,
    numActiveExperts: 4,
    expertIntermediateSize: 4096,
    numSharedExperts: 1,
    firstKDenseLayers: 3,
  },
  // Ministral 3 dense models — GQA, SwiGLU, 256K context
  'ministral-3-3b': {
    name: 'Ministral 3 (3B)',
    family: 'mistral',
    numLayers: 26,
    hiddenSize: 3072,
    intermediateSize: 9216,
    numAttentionHeads: 32,
    numKvHeads: 8,
    headDim: 128,
    vocabSize: 131072,
    maxSeqLength: 262144,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: true,
    useRotaryEmbed: true,
  },
  'ministral-3-8b': {
    name: 'Ministral 3 (8B)',
    family: 'mistral',
    numLayers: 34,
    hiddenSize: 4096,
    intermediateSize: 14336,
    numAttentionHeads: 32,
    numKvHeads: 8,
    vocabSize: 131072,
    maxSeqLength: 262144,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'ministral-3-14b': {
    name: 'Ministral 3 (14B)',
    family: 'mistral',
    numLayers: 40,
    hiddenSize: 5120,
    intermediateSize: 16384,
    numAttentionHeads: 32,
    numKvHeads: 8,
    headDim: 128,
    vocabSize: 131072,
    maxSeqLength: 262144,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'mixtral-8x22b': {
    name: 'Mixtral 8x22B (141B)',
    family: 'mistral',
    numLayers: 56,
    hiddenSize: 6144,
    intermediateSize: 16384,
    numAttentionHeads: 48,
    numKvHeads: 8,
    vocabSize: 32768,
    maxSeqLength: 65536,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
    numExperts: 8,
    numActiveExperts: 2,
    expertIntermediateSize: 16384,
  },
};

/**
 * DeepSeek MoE models
 * Features: Fine-grained experts, shared experts
 */
export const DEEPSEEK_CONFIGS: Record<string, ModelConfig> = {
  'deepseek-moe-16b': {
    name: 'DeepSeek-MoE (16B)',
    family: 'deepseek',
    numLayers: 28,
    hiddenSize: 2048,
    intermediateSize: 10944,
    numAttentionHeads: 16,
    vocabSize: 102400,
    maxSeqLength: 4096,
    attentionType: 'mha',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
    numExperts: 64,
    numActiveExperts: 6,
    expertIntermediateSize: 1408,
    numSharedExperts: 2,
    firstKDenseLayers: 1,
  },
  'deepseek-v2': {
    name: 'DeepSeek-V2 (236B)',
    family: 'deepseek',
    numLayers: 60,
    hiddenSize: 5120,
    intermediateSize: 12288,
    numAttentionHeads: 128,
    numKvHeads: 128,          // MLA decompresses to all heads
    headDim: 128,             // v_head_dim
    vocabSize: 102400,
    maxSeqLength: 128000,
    attentionType: 'mla',
    kvLoraRank: 512,
    qLoraRank: 1536,
    qkNopeHeadDim: 128,
    qkRopeHeadDim: 64,
    vHeadDim: 128,
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
    numExperts: 160,
    numActiveExperts: 6,
    expertIntermediateSize: 1536,
    numSharedExperts: 2,
    firstKDenseLayers: 1,
    routingDeviceLimit: 6,    // Each token contacts at most M=6 EP groups (paper §2.1.1)
  },
  // 61 layers (prime) — interleaved PP scheduling requires layers divisible by pp*v,
  // so only standard 1F1B works with PP>1. DeepSeek used 1F1B in their training.
  'deepseek-v3': {
    name: 'DeepSeek V3/R1 (671B)',
    family: 'deepseek',
    numLayers: 61,
    hiddenSize: 7168,
    intermediateSize: 18432,
    numAttentionHeads: 128,
    numKvHeads: 128,          // MLA decompresses to all heads
    headDim: 128,             // v_head_dim
    vocabSize: 129280,
    maxSeqLength: 163840,
    attentionType: 'mla',
    kvLoraRank: 512,
    qLoraRank: 1536,
    qkNopeHeadDim: 128,
    qkRopeHeadDim: 64,
    vHeadDim: 128,
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
    numExperts: 256,
    numActiveExperts: 8,
    expertIntermediateSize: 2048,
    numSharedExperts: 1,
    firstKDenseLayers: 3,
    routingDeviceLimit: 4,    // Each token contacts at most M=4 EP groups (paper §3.4)
  },
};

/**
 * Kimi K2 / K2.5 (Moonshot AI, 2025)
 * K2, K2-Instruct, K2-Thinking, and K2.5 (multimodal) all share the same 1T LLM backbone.
 * K2.5 adds a ~400M vision encoder (negligible vs 1T, not modeled). We use the K2.5 name
 * since it's the latest release and what people will search for.
 *
 * Architecture: DeepSeek V3 variant — same MLA dims, hidden size, expert intermediate size.
 * Key differences: 384 experts (vs 256), 64 attention heads (vs 128), 1 dense layer (vs 3),
 * vocabSize 163840 (vs 129280). Uses DeepseekV3ForCausalLM on HuggingFace.
 * Source: moonshotai/Kimi-K2-Instruct config.json, arXiv 2507.20534
 */
export const KIMI_CONFIGS: Record<string, ModelConfig> = {
  'kimi-k2.5': {
    name: 'Kimi K2.5 (1T)',
    family: 'kimi',

    // Architecture — same hidden/intermediate as DeepSeek V3
    numLayers: 61,                    // Same as V3 — prime number
    hiddenSize: 7168,
    intermediateSize: 18432,          // Dense layer FFN (1 dense layer only)
    numAttentionHeads: 64,            // Half of V3's 128 (inference efficiency)
    numKvHeads: 64,                   // MLA decompresses to all heads
    headDim: 128,                     // Override (cosmetic for MLA, matches vHeadDim)
    vocabSize: 163840,
    maxSeqLength: 131072,             // K2-Base; K2.5 extends to 262K via YaRN

    // MLA — identical dimensions to DeepSeek V3
    attentionType: 'mla',
    kvLoraRank: 512,
    qLoraRank: 1536,
    qkNopeHeadDim: 128,
    qkRopeHeadDim: 64,
    vHeadDim: 128,

    // MoE — 384 experts (50% more than V3's 256)
    numExperts: 384,
    numActiveExperts: 8,
    numSharedExperts: 1,
    expertIntermediateSize: 2048,
    firstKDenseLayers: 1,             // Only layer 0 is dense (V3 has 3)

    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,                   // attention_bias: false
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
};

/**
 * Qwen 2.5 family models
 * Architecture: RMSNorm, SiLU, gated MLP, GQA, RoPE, useBias: true
 */
export const QWEN_CONFIGS: Record<string, ModelConfig> = {
  'qwen2.5-0.5b': {
    name: 'Qwen2.5 (0.5B)',
    family: 'qwen',
    numLayers: 24,
    hiddenSize: 896,
    intermediateSize: 4864,
    numAttentionHeads: 14,
    numKvHeads: 2,
    vocabSize: 151936,
    maxSeqLength: 32768,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: true,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'qwen2.5-1.5b': {
    name: 'Qwen2.5 (1.5B)',
    family: 'qwen',
    numLayers: 28,
    hiddenSize: 1536,
    intermediateSize: 8960,
    numAttentionHeads: 12,
    numKvHeads: 2,
    vocabSize: 151936,
    maxSeqLength: 32768,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: true,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'qwen2.5-3b': {
    name: 'Qwen2.5 (3B)',
    family: 'qwen',
    numLayers: 36,
    hiddenSize: 2048,
    intermediateSize: 11008,
    numAttentionHeads: 16,
    numKvHeads: 2,
    vocabSize: 151936,
    maxSeqLength: 32768,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: true,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'qwen2.5-7b': {
    name: 'Qwen2.5 (7B)',
    family: 'qwen',
    numLayers: 28,
    hiddenSize: 3584,
    intermediateSize: 18944,
    numAttentionHeads: 28,
    numKvHeads: 4,
    vocabSize: 152064,
    maxSeqLength: 32768,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: true,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'qwen2.5-14b': {
    name: 'Qwen2.5 (14B)',
    family: 'qwen',
    numLayers: 48,
    hiddenSize: 5120,
    intermediateSize: 13824,
    numAttentionHeads: 40,
    numKvHeads: 8,
    vocabSize: 152064,
    maxSeqLength: 131072,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: true,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'qwen2.5-32b': {
    name: 'Qwen2.5 (32B)',
    family: 'qwen',
    numLayers: 64,
    hiddenSize: 5120,
    intermediateSize: 27648,
    numAttentionHeads: 40,
    numKvHeads: 8,
    vocabSize: 152064,
    maxSeqLength: 32768,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: true,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'qwen2.5-72b': {
    name: 'Qwen2.5 (72B)',
    family: 'qwen',
    numLayers: 80,
    hiddenSize: 8192,
    intermediateSize: 29568,
    numAttentionHeads: 64,
    numKvHeads: 8,
    vocabSize: 152064,
    maxSeqLength: 32768,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: true,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
};

/**
 * Gemma 2 family models (Google, 2024)
 * Architecture: RMSNorm, GeGLU (gelu gated MLP), GQA, RoPE, tied embeddings
 * Training SWA NOT modeled: alternating local (window=4096) / global attention layers.
 * Impact negligible with Flash Attention at typical training seqLength (<=8192).
 */
export const GEMMA2_CONFIGS: Record<string, ModelConfig> = {
  'gemma2-2b': {
    name: 'Gemma 2 (2B)',
    family: 'gemma',
    numLayers: 26,
    hiddenSize: 2304,
    intermediateSize: 9216,
    numAttentionHeads: 8,
    numKvHeads: 4,
    vocabSize: 256000,
    maxSeqLength: 8192,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'gelu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: true,
    useRotaryEmbed: true,
  },
  'gemma2-9b': {
    name: 'Gemma 2 (9B)',
    family: 'gemma',
    numLayers: 42,
    hiddenSize: 3584,
    intermediateSize: 14336,
    numAttentionHeads: 16,
    numKvHeads: 8,
    vocabSize: 256000,
    maxSeqLength: 8192,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'gelu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: true,
    useRotaryEmbed: true,
  },
  'gemma2-27b': {
    name: 'Gemma 2 (27B)',
    family: 'gemma',
    numLayers: 46,
    hiddenSize: 4608,
    intermediateSize: 36864,
    numAttentionHeads: 32,
    numKvHeads: 16,
    vocabSize: 256000,
    maxSeqLength: 8192,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'gelu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: true,
    useRotaryEmbed: true,
  },
};

/**
 * Phi-3 family models (Microsoft, 2024)
 * Architecture: RMSNorm, gated MLP, RoPE, no bias
 */
export const PHI3_CONFIGS: Record<string, ModelConfig> = {
  'phi3-mini': {
    name: 'Phi-3 Mini (3.8B)',
    family: 'phi3',
    numLayers: 32,
    hiddenSize: 3072,
    intermediateSize: 8192,
    numAttentionHeads: 32,
    vocabSize: 32064,
    maxSeqLength: 131072,
    attentionType: 'mha',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'phi3-small': {
    name: 'Phi-3 Small (7B)',
    family: 'phi3',
    numLayers: 32,
    hiddenSize: 4096,
    intermediateSize: 14336,
    numAttentionHeads: 32,
    numKvHeads: 8,
    vocabSize: 100352,
    maxSeqLength: 131072,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'gelu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'phi3-medium': {
    name: 'Phi-3 Medium (14B)',
    family: 'phi3',
    numLayers: 40,
    hiddenSize: 5120,
    intermediateSize: 17920,
    numAttentionHeads: 40,
    numKvHeads: 10,
    vocabSize: 32064,
    maxSeqLength: 131072,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
};

/**
 * Phi-4 family models (Microsoft, 2024-2025)
 * Architecture: GQA for all sizes, SiLU, gated MLP, no bias, RoPE (except mini-flash)
 * Larger tokenizer than Phi-3 (100K-200K vocab)
 */
export const PHI4_CONFIGS: Record<string, ModelConfig> = {
  'phi4': {
    name: 'Phi-4 (14B)',
    family: 'phi4',
    numLayers: 40,
    hiddenSize: 5120,
    intermediateSize: 17920,
    numAttentionHeads: 40,
    numKvHeads: 10,
    vocabSize: 100352,
    maxSeqLength: 16384,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'phi4-mini': {
    name: 'Phi-4 Mini (3.8B)',
    family: 'phi4',
    numLayers: 32,
    hiddenSize: 3072,
    intermediateSize: 8192,
    numAttentionHeads: 24,
    numKvHeads: 8,
    vocabSize: 200064,
    maxSeqLength: 131072,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: true,
    useRotaryEmbed: true,
  },
  'phi4-mini-flash': {
    name: 'Phi-4 Mini Flash (2.5B)',
    family: 'phi4',
    numLayers: 32,
    hiddenSize: 2560,
    intermediateSize: 10240,
    numAttentionHeads: 40,
    numKvHeads: 20,
    vocabSize: 200064,
    maxSeqLength: 262144,
    attentionType: 'gqa',
    normType: 'layernorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: true,
    useRotaryEmbed: false,
  },
};


/**
 * OLMo 2 family models (AI2, 2024)
 * Architecture: MHA (not GQA), RMSNorm, SwiGLU, RoPE, no bias
 * Fully open training data and code
 */
export const OLMO2_CONFIGS: Record<string, ModelConfig> = {
  'olmo2-7b': {
    name: 'OLMo 2 (7B)',
    family: 'olmo2',
    numLayers: 32,
    hiddenSize: 4096,
    intermediateSize: 11008,
    numAttentionHeads: 32,
    vocabSize: 100278,
    maxSeqLength: 4096,
    attentionType: 'mha',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'olmo2-13b': {
    name: 'OLMo 2 (13B)',
    family: 'olmo2',
    numLayers: 40,
    hiddenSize: 5120,
    intermediateSize: 13824,
    numAttentionHeads: 40,
    vocabSize: 100278,
    maxSeqLength: 4096,
    attentionType: 'mha',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
};

/**
 * OLMo 3 family models (AI2, Dec 2025)
 * Architecture: RMSNorm, SwiGLU, RoPE, no bias — successor to OLMo 2
 * 7B: identical dimensions to OLMo 2 7B except maxSeqLength (4096→65536)
 * 32B: completely different model — 64 layers, 27648 FFN, GQA (8 KV heads)
 * Training SWA NOT modeled: 3/4 layers use 4K window, every 4th full attention.
 * QK-Norm NOT modeled: negligible params (2 × headDim scaling vectors per layer).
 * Source: allenai/Olmo-3-7B-Instruct, allenai/Olmo-3.1-32B-Instruct config.json
 */
export const OLMO3_CONFIGS: Record<string, ModelConfig> = {
  'olmo3-7b': {
    name: 'OLMo 3 (7B)',
    family: 'olmo3',
    numLayers: 32,
    hiddenSize: 4096,
    intermediateSize: 11008,
    numAttentionHeads: 32,
    // numKvHeads omitted — defaults to numAttentionHeads (MHA), same as OLMo 2 7B
    vocabSize: 100278,
    maxSeqLength: 65536,
    attentionType: 'mha',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'olmo3-32b': {
    name: 'OLMo 3 (32B)',
    family: 'olmo3',
    numLayers: 64,
    hiddenSize: 5120,
    // intermediateSize 5.4× hidden — deliberately oversized FFN to hit 32B target
    // for direct comparison with Qwen3-32B despite smaller vocab (100K vs 152K)
    intermediateSize: 27648,
    numAttentionHeads: 40,
    numKvHeads: 8,              // GQA group ratio = 5 (non-power-of-2)
    vocabSize: 100278,
    maxSeqLength: 65536,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
};


/**
 * Nemotron-4 family models (NVIDIA, 2024)
 * Architecture: ReLU² (approx as relu), non-gated 4x MLP, LayerNorm, GQA, RoPE
 * Published MFU: 34.3% on 3072 H100s
 */
export const NEMOTRON_4_CONFIGS: Record<string, ModelConfig> = {
  'nemotron-4-15b': {
    name: 'Nemotron-4 (15B)',
    family: 'nemotron-4',
    numLayers: 32,
    hiddenSize: 6144,
    intermediateSize: 24576,
    numAttentionHeads: 48,
    numKvHeads: 8,
    vocabSize: 256000,
    maxSeqLength: 4096,
    attentionType: 'gqa',
    normType: 'layernorm',
    activation: 'relu',
    gatedMLP: false,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'nemotron-4-340b': {
    name: 'Nemotron-4 (340B)',
    family: 'nemotron-4',
    numLayers: 96,
    hiddenSize: 18432,
    intermediateSize: 73728,
    numAttentionHeads: 96,
    numKvHeads: 8,
    vocabSize: 256000,
    maxSeqLength: 4096,
    attentionType: 'gqa',
    normType: 'layernorm',
    activation: 'relu',
    gatedMLP: false,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
};

/**
 * Grok-1 (xAI, 2024)
 * Architecture: MoE with 8 experts (2 active), GQA, RMSNorm, GeGLU (gated GELU), RoPE
 * Source: https://github.com/xai-org/grok-1/blob/main/model.py
 *   - MLP is gated: output = linear_1(GELU(linear(x)) * linear_v(x))  (3 weight matrices)
 *   - intermediateSize = ffn_size(6144, widening_factor=8) = int(8*6144)*2//3 = 32768
 *     (2/3 reduction compensates for the extra gate matrix)
 *   - Published: 314B total, 86B active (25% utilization)
 *   - Simulator: ~316.5B total, ~84.6B active (within 1-2% of published)
 */
export const GROK_CONFIGS: Record<string, ModelConfig> = {
  'grok-1': {
    name: 'Grok-1 (314B)',
    family: 'grok',
    numLayers: 64,
    hiddenSize: 6144,
    intermediateSize: 32768,
    numAttentionHeads: 48,
    numKvHeads: 8,
    vocabSize: 131072,
    maxSeqLength: 8192,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'gelu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
    numExperts: 8,
    numActiveExperts: 2,
    expertIntermediateSize: 32768,
  },

  /**
   * Grok 2.5 (xAI, 2025)
   * Open-weight release of Grok 2 on HuggingFace (xai-org/grok-2)
   * Architecture: Residual MoE — every layer has both a shared dense FFN (intermediate=32768)
   *   AND 8 sparse MoE experts (intermediate=16384 each, 2 active)
   * The shared dense FFN (3×8192×32768 = 805.3M params) equals exactly 2 experts
   *   (2 × 3×8192×16384 = 805.3M), so modeled as numSharedExperts=2
   * GeGLU activation (gated GELU), GQA with 8 KV heads, 128K context via RoPE
   * Published: ~270B total, ~115B active
   * Simulator: ~269.5B total, ~114.9B active (within 1% of published)
   */
  'grok-2.5': {
    name: 'Grok 2.5 (270B)',
    family: 'grok',
    numLayers: 64,
    hiddenSize: 8192,
    intermediateSize: 32768,      // shared FFN size; unused — modeled via 2 shared experts × 16384
    numAttentionHeads: 64,
    numKvHeads: 8,
    vocabSize: 131072,
    maxSeqLength: 131072,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'gelu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
    numExperts: 8,
    numActiveExperts: 2,
    expertIntermediateSize: 16384,
    numSharedExperts: 2,
  },
};

/**
 * MiniMax M2.5 (MiniMax, 2026)
 * Architecture: MoE with 256 experts (8 active, 0 shared), GQA, RMSNorm, SiLU, gated MLP, no bias, untied, RoPE
 * headDim=128 (override: natural 3072/48=64)
 * All 62 layers are MoE — no dense-first layers
 * Has MTP modules (3 modules, 1 layer each) — auxiliary training technique, not modeled
 * Published: ~230B total, ~10B active (simulator: ~228.7B / ~11.03B)
 */
export const MINIMAX_CONFIGS: Record<string, ModelConfig> = {
  'minimax-m2.5': {
    name: 'MiniMax-M2.5 (230B)',
    family: 'minimax',
    numLayers: 62,
    hiddenSize: 3072,
    intermediateSize: 1536,      // = expertIntermediateSize (no shared experts, all layers MoE)
    numAttentionHeads: 48,
    numKvHeads: 8,
    headDim: 128,                // Natural=64, override to 128
    vocabSize: 200064,
    maxSeqLength: 196608,        // 192K context
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
    numExperts: 256,
    numActiveExperts: 8,
    expertIntermediateSize: 1536,
  },
};

/**
 * GLM family models (Zhipu AI / THUDM, 2024-2025)
 * Dense: GQA, RMSNorm, SiLU, gated MLP, RoPE
 * MoE: Fine-grained experts, shared experts, firstKDenseLayers
 * GLM-4.7-Flash and GLM-5 use MLA attention
 */
export const GLM_CONFIGS: Record<string, ModelConfig> = {
  'glm4-9b': {
    name: 'GLM-4 (9B)',
    family: 'GLM',
    numLayers: 40,
    hiddenSize: 4096,
    intermediateSize: 13696,
    numAttentionHeads: 32,
    numKvHeads: 2,
    vocabSize: 151552,
    maxSeqLength: 131072,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: true,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'glm4-32b': {
    name: 'GLM-4 (32B)',
    family: 'GLM',
    numLayers: 61,
    hiddenSize: 6144,
    intermediateSize: 23040,
    numAttentionHeads: 48,
    numKvHeads: 2,
    vocabSize: 151552,
    maxSeqLength: 32768,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'glm4.5-air': {
    name: 'GLM-4.5 Air (107B)',
    family: 'GLM',
    numLayers: 46,
    hiddenSize: 4096,
    intermediateSize: 10944,
    numAttentionHeads: 96,
    numKvHeads: 8,
    headDim: 128,                   // override: 96×128 ≠ 4096
    vocabSize: 151552,
    maxSeqLength: 131072,
    attentionType: 'gqa',
    numExperts: 128,
    numActiveExperts: 8,
    numSharedExperts: 1,
    expertIntermediateSize: 1408,
    firstKDenseLayers: 1,
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: true,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'glm4.7': {
    name: 'GLM-4.7 (353B)',
    family: 'GLM',
    numLayers: 92,
    hiddenSize: 5120,
    intermediateSize: 12288,
    numAttentionHeads: 96,
    numKvHeads: 8,
    headDim: 128,                   // override: 96×128=12288 ≠ 5120
    vocabSize: 151552,
    maxSeqLength: 202752,
    attentionType: 'gqa',
    numExperts: 160,
    numActiveExperts: 8,
    numSharedExperts: 1,
    expertIntermediateSize: 1536,
    firstKDenseLayers: 3,
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: true,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'glm4.7-flash': {
    name: 'GLM-4.7 Flash (30B)',
    family: 'GLM',
    numLayers: 47,
    hiddenSize: 2048,
    intermediateSize: 10240,
    numAttentionHeads: 20,
    numKvHeads: 20,                 // MLA decompresses to all heads
    vocabSize: 154880,
    maxSeqLength: 202752,
    attentionType: 'mla',
    kvLoraRank: 512,
    qLoraRank: 768,
    qkNopeHeadDim: 192,
    qkRopeHeadDim: 64,
    vHeadDim: 256,
    numExperts: 64,
    numActiveExperts: 4,
    numSharedExperts: 1,
    expertIntermediateSize: 1536,
    firstKDenseLayers: 1,
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'glm5': {
    name: 'GLM-5 (743B)',
    family: 'GLM',
    numLayers: 78,
    hiddenSize: 6144,
    intermediateSize: 12288,
    numAttentionHeads: 64,
    numKvHeads: 64,                 // MLA decompresses to all heads
    headDim: 64,                    // override (MLA uses its own dims)
    vocabSize: 154880,
    maxSeqLength: 202752,
    attentionType: 'mla',
    kvLoraRank: 512,
    qLoraRank: 2048,
    qkNopeHeadDim: 192,
    qkRopeHeadDim: 64,
    vHeadDim: 256,
    numExperts: 256,
    numActiveExperts: 8,
    numSharedExperts: 1,
    expertIntermediateSize: 2048,
    firstKDenseLayers: 3,
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
};

/**
 * LLaMA 4 family models (Meta, 2025)
 * Architecture: MoE, GQA (q=40, kv=8), head_dim=128, RMSNorm, SiLU, gated MLP, no bias, untied, RoPE
 * Both have shared experts (1 per MoE layer) + routed experts
 * Maverick: alternating dense/MoE layers (moeLayerFrequency=2, 24 dense + 24 MoE)
 */
export const LLAMA4_CONFIGS: Record<string, ModelConfig> = {
  'llama4-scout': {
    name: 'LLaMA-4 Scout (109B)',
    family: 'llama4',
    numLayers: 48,
    hiddenSize: 5120,
    intermediateSize: 16384,
    numAttentionHeads: 40,
    numKvHeads: 8,
    vocabSize: 202048,
    maxSeqLength: 10485760,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
    numExperts: 16,
    numActiveExperts: 1,
    expertIntermediateSize: 8192,
    numSharedExperts: 1,
  },
  'llama4-maverick': {
    name: 'LLaMA-4 Maverick (400B)',
    family: 'llama4',
    numLayers: 48,
    hiddenSize: 5120,
    intermediateSize: 16384,
    numAttentionHeads: 40,
    numKvHeads: 8,
    vocabSize: 202048,
    maxSeqLength: 1048576,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
    numExperts: 128,
    numActiveExperts: 1,
    expertIntermediateSize: 8192,
    numSharedExperts: 1,
    moeLayerFrequency: 2,
  },
};

/**
 * Qwen 3 family models (Alibaba, 2025)
 * Dense: vocab=151936, GQA(kv=8), RMSNorm, SiLU, gated MLP, no bias, RoPE, maxSeqLen=40960
 * All set head_dim=128 in config.json (decoupled from hidden_size/num_heads)
 * MoE: vocab=151936, GQA(kv=4), 128 experts, 8 active, no shared experts
 */
export const QWEN3_CONFIGS: Record<string, ModelConfig> = {
  'qwen3-0.6b': {
    name: 'Qwen3 (0.6B)',
    family: 'qwen3',
    numLayers: 28,
    hiddenSize: 1024,
    intermediateSize: 3072,
    numAttentionHeads: 16,
    numKvHeads: 8,
    headDim: 128,             // Natural=64, override to 128
    vocabSize: 151936,
    maxSeqLength: 40960,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: true,
    useRotaryEmbed: true,
  },
  'qwen3-1.7b': {
    name: 'Qwen3 (1.7B)',
    family: 'qwen3',
    numLayers: 28,
    hiddenSize: 2048,
    intermediateSize: 6144,
    numAttentionHeads: 16,
    numKvHeads: 8,
    vocabSize: 151936,
    maxSeqLength: 40960,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: true,
    useRotaryEmbed: true,
  },
  'qwen3-4b': {
    name: 'Qwen3 (4B)',
    family: 'qwen3',
    numLayers: 36,
    hiddenSize: 2560,
    intermediateSize: 9728,
    numAttentionHeads: 32,
    numKvHeads: 8,
    headDim: 128,             // Natural=80, override to 128
    vocabSize: 151936,
    maxSeqLength: 40960,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: true,
    useRotaryEmbed: true,
  },
  'qwen3-8b': {
    name: 'Qwen3 (8B)',
    family: 'qwen3',
    numLayers: 36,
    hiddenSize: 4096,
    intermediateSize: 12288,
    numAttentionHeads: 32,
    numKvHeads: 8,
    vocabSize: 151936,
    maxSeqLength: 40960,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'qwen3-14b': {
    name: 'Qwen3 (14B)',
    family: 'qwen3',
    numLayers: 40,
    hiddenSize: 5120,
    intermediateSize: 17408,
    numAttentionHeads: 40,
    numKvHeads: 8,
    vocabSize: 151936,
    maxSeqLength: 40960,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'qwen3-32b': {
    name: 'Qwen3 (32B)',
    family: 'qwen3',
    numLayers: 64,
    hiddenSize: 5120,
    intermediateSize: 25600,
    numAttentionHeads: 64,
    numKvHeads: 8,
    headDim: 128,             // Natural=80, override to 128
    vocabSize: 151936,
    maxSeqLength: 40960,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  // Qwen 3 MoE models — 128 experts, 8 active, GQA(kv=4), no shared experts
  'qwen3-30b-a3b': {
    name: 'Qwen3 MoE (30B-A3B)',
    family: 'qwen3',
    numLayers: 48,
    hiddenSize: 2048,
    intermediateSize: 768,    // Set to expertIntermediateSize (no shared experts)
    numAttentionHeads: 32,
    numKvHeads: 4,
    headDim: 128,             // Natural=64, override to 128
    vocabSize: 151936,
    maxSeqLength: 40960,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
    numExperts: 128,
    numActiveExperts: 8,
    expertIntermediateSize: 768,
  },
  'qwen3-235b-a22b': {
    name: 'Qwen3 MoE (235B-A22B)',
    family: 'qwen3',
    numLayers: 94,
    hiddenSize: 4096,
    intermediateSize: 1536,   // Set to expertIntermediateSize (no shared experts)
    numAttentionHeads: 64,
    numKvHeads: 4,
    headDim: 128,             // Natural=64, override to 128
    vocabSize: 151936,
    maxSeqLength: 40960,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
    numExperts: 128,
    numActiveExperts: 8,
    expertIntermediateSize: 1536,
  },
};

/**
 * Gemma 3 family models (Google, 2025)
 * Architecture: GELU, gated MLP, no bias, tied embeddings, RoPE
 * Head dims decoupled from hidden_size/num_heads (all need override)
 * Training SWA NOT modeled: interleaved 5 local (window=1024) + 1 global attention layers.
 * Worst-case unmodeled impact: ~10% per-layer FLOPs at seq=8192 (attention is ~13% of
 * layer FLOPs, SWA saves ~73% of that). Not worth per-layer window tracking complexity.
 */
export const GEMMA3_CONFIGS: Record<string, ModelConfig> = {
  'gemma3-1b': {
    name: 'Gemma 3 (1B)',
    family: 'gemma',
    numLayers: 26,
    hiddenSize: 1152,
    intermediateSize: 6912,
    numAttentionHeads: 4,
    numKvHeads: 1,
    headDim: 256,             // Natural=288, override to 256
    vocabSize: 262144,
    maxSeqLength: 32768,
    attentionType: 'mqa',
    normType: 'rmsnorm',
    activation: 'gelu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: true,
    useRotaryEmbed: true,
  },
  'gemma3-4b': {
    name: 'Gemma 3 (4B)',
    family: 'gemma',
    numLayers: 34,
    hiddenSize: 2560,
    intermediateSize: 10240,
    numAttentionHeads: 8,
    numKvHeads: 4,
    headDim: 256,             // Natural=320, override to 256
    vocabSize: 262208,
    maxSeqLength: 131072,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'gelu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: true,
    useRotaryEmbed: true,
  },
  'gemma3-12b': {
    name: 'Gemma 3 (12B)',
    family: 'gemma',
    numLayers: 48,
    hiddenSize: 3840,
    intermediateSize: 15360,
    numAttentionHeads: 16,
    numKvHeads: 8,
    headDim: 256,             // Natural=240, override to 256
    vocabSize: 262208,
    maxSeqLength: 131072,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'gelu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: true,
    useRotaryEmbed: true,
  },
  'gemma3-27b': {
    name: 'Gemma 3 (27B)',
    family: 'gemma',
    numLayers: 62,
    hiddenSize: 5376,
    intermediateSize: 21504,
    numAttentionHeads: 32,
    numKvHeads: 16,
    headDim: 128,             // Natural=168, override to 128
    vocabSize: 262208,
    maxSeqLength: 131072,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'gelu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: true,
    useRotaryEmbed: true,
  },
};

/**
 * All built-in model configurations
 */
// Hidden from UI selectors but still resolvable by ID (superseded or pruned from dropdown)
const HIDDEN_MODELS: Record<string, ModelConfig> = {
  // Qwen2 MoE — 64 routed experts (8 active) + 1 shared expert
  // Shared expert intermediate_size=20480 ≡ 8 × expertIntermediateSize(2560)
  // All layers are MoE (decoder_sparse_step=1)
  // Source: huggingface.co/Qwen/Qwen2-57B-A14B config.json
  'qwen2-57b-a14b': {
    name: 'Qwen2 MoE (57B-A14B)',
    family: 'qwen',
    numLayers: 28,
    hiddenSize: 3584,
    intermediateSize: 2560,         // = expertIntermediateSize (all layers MoE)
    numAttentionHeads: 28,
    numKvHeads: 4,
    vocabSize: 151936,
    maxSeqLength: 131072,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: true,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
    numExperts: 64,
    numActiveExperts: 8,
    numSharedExperts: 1,
    sharedExpertIntermediateSize: 20480, // 1 shared expert × 20480 (better saturated than 8 × 2560)
    expertIntermediateSize: 2560,
  },
  // BLOOM 176B — BigScience (2022), ALiBi positional encoding (no RoPE)
  // 70 layers, h=14336, 112 heads, vocab=250880, non-gated MLP, LayerNorm
  // Source: https://arxiv.org/abs/2211.05100
  'bloom-176b': {
    name: 'BLOOM (176B)',
    family: 'bloom',
    numLayers: 70,
    hiddenSize: 14336,
    intermediateSize: 57344,    // 4 * 14336
    numAttentionHeads: 112,
    vocabSize: 250880,
    maxSeqLength: 2048,
    attentionType: 'mha',
    normType: 'layernorm',
    activation: 'gelu',
    gatedMLP: false,
    useBias: true,
    tiedEmbeddings: false,
    useRotaryEmbed: false,      // ALiBi positional encoding
  },
  // Megatron-Turing NLG 530B — GPT-3 architecture scaled to 530B
  'megatron-turing-530b': {
    name: 'Megatron-Turing NLG (530B)',
    family: 'gpt3',
    numLayers: 105,
    hiddenSize: 20480,
    intermediateSize: 81920,   // 4× hidden
    numAttentionHeads: 128,    // head_dim=160
    vocabSize: 50257,
    maxSeqLength: 2048,
    attentionType: 'mha',
    normType: 'layernorm',
    activation: 'gelu',
    gatedMLP: false,
    useBias: true,
    tiedEmbeddings: false,
    useRotaryEmbed: false,
  },
  // OLMo 2 32B — benchmark calibration
  // Source: HuggingFace allenai/OLMo-2-0325-32B config.json
  'olmo2-32b': {
    name: 'OLMo 2 (32B)',
    family: 'olmo2',
    numLayers: 64,
    hiddenSize: 5120,
    intermediateSize: 27648,
    numAttentionHeads: 40,
    numKvHeads: 8,
    vocabSize: 100352,          // padded to 128 (vs 100278 in OLMo 3)
    maxSeqLength: 4096,         // native 4K context (vs 65K in OLMo 3)
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  // Mistral — superseded by Ministral 3 / Devstral 2 / Large 3
  'mistral-7b': {
    name: 'Mistral (7B)',
    family: 'mistral',
    numLayers: 32,
    hiddenSize: 4096,
    intermediateSize: 14336,
    numAttentionHeads: 32,
    numKvHeads: 8,
    vocabSize: 32768,
    maxSeqLength: 32768,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'mistral-nemo-12b': {
    name: 'Mistral Nemo (12B)',
    family: 'mistral',
    numLayers: 40,
    hiddenSize: 5120,
    intermediateSize: 14336,
    numAttentionHeads: 32,
    numKvHeads: 8,
    headDim: 128,
    vocabSize: 131072,
    maxSeqLength: 131072,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'codestral-22b': {
    name: 'Codestral (22B)',
    family: 'mistral',
    numLayers: 56,
    hiddenSize: 6144,
    intermediateSize: 16384,
    numAttentionHeads: 48,
    numKvHeads: 8,
    vocabSize: 32768,
    maxSeqLength: 32768,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'mistral-small-24b': {
    name: 'Mistral Small (24B)',
    family: 'mistral',
    numLayers: 40,
    hiddenSize: 5120,
    intermediateSize: 32768,
    numAttentionHeads: 32,
    numKvHeads: 8,
    headDim: 128,
    vocabSize: 131072,
    maxSeqLength: 32768,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'mistral-large-123b': {
    name: 'Mistral Large (123B)',
    family: 'mistral',
    numLayers: 88,
    hiddenSize: 12288,
    intermediateSize: 28672,
    numAttentionHeads: 96,
    numKvHeads: 8,
    vocabSize: 32768,
    maxSeqLength: 131072,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'mixtral-8x7b': {
    name: 'Mixtral 8x7B (47B)',
    family: 'mistral',
    numLayers: 32,
    hiddenSize: 4096,
    intermediateSize: 14336,
    numAttentionHeads: 32,
    numKvHeads: 8,
    vocabSize: 32000,
    maxSeqLength: 32768,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
    numExperts: 8,
    numActiveExperts: 2,
    expertIntermediateSize: 14336,
  },
  // DBRX (Databricks, 2024) — MoE with 16 experts (4 active)
  'dbrx': {
    name: 'DBRX (132B)',
    family: 'dbrx',
    numLayers: 40,
    hiddenSize: 6144,
    intermediateSize: 10752,
    numAttentionHeads: 48,
    numKvHeads: 8,
    vocabSize: 100352,
    maxSeqLength: 32768,
    attentionType: 'gqa',
    normType: 'layernorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
    numExperts: 16,
    numActiveExperts: 4,
    expertIntermediateSize: 10752,
  },
  // Yi (01.AI, 2024) — LLaMA-like
  'yi-6b': {
    name: 'Yi (6B)',
    family: 'yi',
    numLayers: 32,
    hiddenSize: 4096,
    intermediateSize: 11008,
    numAttentionHeads: 32,
    numKvHeads: 4,
    vocabSize: 64000,
    maxSeqLength: 4096,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'yi-34b': {
    name: 'Yi (34B)',
    family: 'yi',
    numLayers: 60,
    hiddenSize: 7168,
    intermediateSize: 20480,
    numAttentionHeads: 56,
    numKvHeads: 8,
    vocabSize: 64000,
    maxSeqLength: 4096,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  // Command R (Cohere, 2024) — LayerNorm, SwiGLU, tied embeddings
  'command-r': {
    name: 'Command R (35B)',
    family: 'command-r',
    numLayers: 40,
    hiddenSize: 8192,
    intermediateSize: 22528,
    numAttentionHeads: 64,
    vocabSize: 256000,
    maxSeqLength: 131072,
    attentionType: 'mha',
    normType: 'layernorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: true,
    useRotaryEmbed: true,
  },
  'command-r-plus': {
    name: 'Command R+ (104B)',
    family: 'command-r',
    numLayers: 64,
    hiddenSize: 12288,
    intermediateSize: 33792,
    numAttentionHeads: 96,
    numKvHeads: 8,
    vocabSize: 256000,
    maxSeqLength: 131072,
    attentionType: 'gqa',
    normType: 'layernorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: true,
    useRotaryEmbed: true,
  },
  // LLaMA 3 — superseded by 3.1
  'llama3-8b': {
    name: 'LLaMA-3 (8B)',
    family: 'llama3',
    numLayers: 32,
    hiddenSize: 4096,
    intermediateSize: 14336,
    numAttentionHeads: 32,
    numKvHeads: 8,
    vocabSize: 128256,
    maxSeqLength: 8192,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  'llama3-70b': {
    name: 'LLaMA-3 (70B)',
    family: 'llama3',
    numLayers: 80,
    hiddenSize: 8192,
    intermediateSize: 28672,
    numAttentionHeads: 64,
    numKvHeads: 8,
    vocabSize: 128256,
    maxSeqLength: 8192,
    attentionType: 'gqa',
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
  },
  // DeepSeek-R1 — identical architecture to V3, only maxSeqLength differs
  'deepseek-r1': {
    name: 'DeepSeek-R1 (671B)',
    family: 'deepseek',
    numLayers: 61,
    hiddenSize: 7168,
    intermediateSize: 18432,
    numAttentionHeads: 128,
    numKvHeads: 128,          // MLA decompresses to all heads
    headDim: 128,             // v_head_dim
    vocabSize: 129280,
    maxSeqLength: 163840,
    attentionType: 'mla',
    kvLoraRank: 512,
    qLoraRank: 1536,
    qkNopeHeadDim: 128,
    qkRopeHeadDim: 64,
    vHeadDim: 128,
    normType: 'rmsnorm',
    activation: 'silu',
    gatedMLP: true,
    useBias: false,
    tiedEmbeddings: false,
    useRotaryEmbed: true,
    numExperts: 256,
    numActiveExperts: 8,
    expertIntermediateSize: 2048,
    numSharedExperts: 1,
    firstKDenseLayers: 3,
    routingDeviceLimit: 4,    // Same as V3 — M=4 device-limited routing
  },
};

export const ALL_MODEL_CONFIGS: Record<string, ModelConfig> = {
  ...GPT3_CONFIGS,
  ...GPTOSS_CONFIGS,
  ...LLAMA2_CONFIGS,
  ...LLAMA3_CONFIGS,
  ...LLAMA4_CONFIGS,
  ...HIDDEN_MODELS,
  ...MISTRAL_CONFIGS,
  ...DEEPSEEK_CONFIGS,
  ...QWEN_CONFIGS,
  ...QWEN3_CONFIGS,
  ...GEMMA2_CONFIGS,
  ...GEMMA3_CONFIGS,
  ...PHI3_CONFIGS,
  ...PHI4_CONFIGS,
  ...OLMO2_CONFIGS,
  ...OLMO3_CONFIGS,
  ...NEMOTRON_4_CONFIGS,
  ...GROK_CONFIGS,
  ...GLM_CONFIGS,
  ...KIMI_CONFIGS,
  ...MINIMAX_CONFIGS,
};

/**
 * Model families for grouping in UI
 */
export const MODEL_FAMILIES = [
  // Alphabetical by display name; biggest models first within each family
  { id: 'deepseek', name: 'DeepSeek', models: ['deepseek-v3', 'deepseek-v2', 'deepseek-moe-16b'] },
  { id: 'gemma', name: 'Gemma', models: [
    'gemma3-27b', 'gemma3-12b', 'gemma3-4b', 'gemma3-1b',
    'gemma2-27b', 'gemma2-9b', 'gemma2-2b',
  ]},
  { id: 'glm', name: 'GLM', models: ['glm5', 'glm4.7', 'glm4.7-flash', 'glm4.5-air', 'glm4-32b', 'glm4-9b'] },
  { id: 'gpt3', name: 'GPT', models: ['gpt-oss-120b', 'gpt-oss-20b', 'gpt3-175b', 'gpt3-13b', 'gpt3-6.7b', 'gpt3-1.3b', 'gpt3-125m'] },
  { id: 'grok', name: 'Grok', models: ['grok-2.5', 'grok-1'] },
  { id: 'kimi', name: 'Kimi', models: ['kimi-k2.5'] },
  { id: 'llama4', name: 'LLaMA-4', models: ['llama4-maverick', 'llama4-scout'] },
  { id: 'llama3', name: 'LLaMA-3', models: [
    'llama3.3-70b',
    'llama3.2-3b', 'llama3.2-1b',
    'llama3-405b', 'llama3.1-8b',
  ]},
  { id: 'llama2', name: 'LLaMA-2', models: ['llama2-70b', 'llama2-13b', 'llama2-7b'] },
  { id: 'minimax', name: 'MiniMax', models: ['minimax-m2.5'] },
  { id: 'mistral', name: 'Mistral', models: [
    'mistral-large-3-675b', 'ministral-3-14b', 'ministral-3-8b', 'ministral-3-3b',
    'devstral-2', 'devstral-small-2', 'mixtral-8x22b',
  ]},
  { id: 'nemotron-4', name: 'Nemotron-4', models: ['nemotron-4-340b', 'nemotron-4-15b'] },
  { id: 'olmo', name: 'OLMo', models: ['olmo3-32b', 'olmo3-7b', 'olmo2-13b', 'olmo2-7b'] },
  { id: 'phi', name: 'Phi', models: [
    'phi4', 'phi4-mini', 'phi4-mini-flash',
    'phi3-medium', 'phi3-small', 'phi3-mini',
  ]},
  { id: 'qwen3', name: 'Qwen 3', models: [
    'qwen3-235b-a22b', 'qwen3-32b', 'qwen3-30b-a3b', 'qwen3-14b', 'qwen3-8b', 'qwen3-4b', 'qwen3-1.7b', 'qwen3-0.6b',
  ]},
  { id: 'qwen', name: 'Qwen 2.5', models: [
    'qwen2.5-72b', 'qwen2.5-32b', 'qwen2.5-14b', 'qwen2.5-7b', 'qwen2.5-3b', 'qwen2.5-1.5b', 'qwen2.5-0.5b',
  ]},
];

/**
 * Get model config by ID
 */
export function getModelConfig(modelId: string): ModelConfig | undefined {
  return ALL_MODEL_CONFIGS[modelId];
}

/**
 * Get popular/recommended models for quick selection
 */
export const POPULAR_MODELS = [
  'gpt3-125m',
  'llama2-7b',
  'llama2-70b',
  'llama3.1-8b',
  'llama3.3-70b',
  'llama3-405b',
  'llama4-scout',
  'mistral-7b',
  'mistral-nemo-12b',
  'mistral-small-24b',
  'mistral-large-3-675b',
  'mistral-large-123b',
  'mixtral-8x7b',
  'gpt3-175b',
  'gemma2-27b',
  'gemma3-27b',
  'phi3-mini',
  'phi4',
  'deepseek-v3',
  'qwen2.5-7b',
  'qwen2.5-72b',
  'qwen3-8b',
  'qwen3-32b',
  'grok-2.5',
  'grok-1',
  'minimax-m2.5',
];
