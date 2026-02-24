/**
 * Model architecture types
 */

import type { DType } from './base.ts';

// Attention mechanism types
export type AttentionType = 'mha' | 'mqa' | 'gqa' | 'mla';

// Layer types
export type LayerType =
  | 'embedding'
  | 'attention'
  | 'mlp'
  | 'layernorm'
  | 'rmsnorm'
  | 'linear'
  | 'moe'
  | 'output';

// Base layer specification
export interface LayerSpec {
  type: LayerType;
  name: string;
  params: number;           // Number of parameters
  flops: number;            // FLOPs for forward pass
  matmulFlops: number;      // Subset of flops from matmul operations (GEMM/batched GEMM)
  activationSize: number;   // Activation memory per token in bytes
  inputShape: number[];
  outputShape: number[];
}

// Embedding layer
export interface EmbeddingLayerSpec extends LayerSpec {
  type: 'embedding';
  vocabSize: number;
  embeddingDim: number;
}

// Attention layer
export interface AttentionLayerSpec extends LayerSpec {
  type: 'attention';
  attentionType: AttentionType;
  numHeads: number;
  numKvHeads: number;       // For GQA/MQA
  headDim: number;
  hiddenSize: number;
  useFlashAttention: boolean;
  useRotaryEmbed: boolean;
}

// MLP/FFN layer
export interface MLPLayerSpec extends LayerSpec {
  type: 'mlp';
  hiddenSize: number;
  intermediateSize: number;
  activation: 'gelu' | 'silu' | 'relu';
  useBias: boolean;
  gatedMLP: boolean;        // LLaMA-style SwiGLU
}

// Normalization layer
export interface NormLayerSpec extends LayerSpec {
  type: 'layernorm' | 'rmsnorm';
  hiddenSize: number;
  eps: number;
}

// MoE layer
export interface MoELayerSpec extends LayerSpec {
  type: 'moe';
  numExperts: number;
  numActiveExperts: number;  // Top-k experts per token
  numSharedExperts: number;  // Always-active experts per MoE layer
  expertSize: number;
  sharedExpertSize?: number; // Total shared expert params
  hiddenSize: number;
  intermediateSize: number;
  capacityFactor: number;    // Load balancing capacity
}

// Output/LM head layer
export interface OutputLayerSpec extends LayerSpec {
  type: 'output';
  hiddenSize: number;
  vocabSize: number;
  tiedEmbeddings: boolean;
}

// Union of all layer specs
export type AnyLayerSpec =
  | EmbeddingLayerSpec
  | AttentionLayerSpec
  | MLPLayerSpec
  | NormLayerSpec
  | MoELayerSpec
  | OutputLayerSpec
  | LayerSpec;

// Transformer block (attention + MLP + norms)
export interface TransformerBlockSpec {
  index: number;
  preNorm: NormLayerSpec;
  attention: AttentionLayerSpec;
  postAttentionNorm?: NormLayerSpec;
  mlp: MLPLayerSpec | MoELayerSpec;
  postNorm?: NormLayerSpec;
  totalParams: number;
  totalFlops: number;
  totalMatmulFlops: number;
}

// Full model specification
export interface ModelSpec {
  name: string;
  family: string;

  // Architecture
  numLayers: number;
  hiddenSize: number;
  intermediateSize: number;
  numAttentionHeads: number;
  numKvHeads: number;
  headDim: number;
  vocabSize: number;
  maxSeqLength: number;

  // Attention config
  attentionType: AttentionType;
  useRotaryEmbed: boolean;
  rotaryBase?: number;

  // MLA (Multi-head Latent Attention) — DeepSeek V2/V3/R1
  kvLoraRank?: number;
  qLoraRank?: number;
  qkNopeHeadDim?: number;
  qkRopeHeadDim?: number;
  vHeadDim?: number;

  // Normalization
  normType: 'layernorm' | 'rmsnorm';
  normEps: number;

  // MLP config
  activation: 'gelu' | 'silu' | 'relu';
  gatedMLP: boolean;
  useBias: boolean;

  // MoE config (optional)
  isMoE: boolean;
  numExperts?: number;
  numActiveExperts?: number;
  numSharedExperts?: number;
  sharedExpertIntermediateSize?: number; // Shared expert MLP intermediate dim (default = expertIntermediateSize)
  numMoELayers?: number;         // Count of MoE layers (for mixed architectures)
  expertIntermediateSize?: number;
  routingDeviceLimit?: number;   // Max EP groups each token can contact (DeepSeek V3: M=4)

  // Embedding
  tiedEmbeddings: boolean;

  // Computed totals
  totalParams: number;
  activeParams: number;      // For MoE models
  embeddingParams: number;
  attentionParams: number;
  mlpParams: number;

  // FLOPs per token
  flopsPerToken: number;
  matmulFlopsPerToken: number;       // Matmul-only FLOPs per token
  matmulFraction: number;            // matmulFlopsPerToken / flopsPerToken

  // Layer breakdown
  layers: AnyLayerSpec[];
  blocks: TransformerBlockSpec[];
}

// Model configuration (input to create ModelSpec)
export interface ModelConfig {
  name: string;
  family?: string;

  numLayers: number;
  hiddenSize: number;
  intermediateSize: number;
  numAttentionHeads: number;
  numKvHeads?: number;
  headDim?: number;           // Override head dimension (default: hiddenSize / numAttentionHeads)
  vocabSize: number;
  maxSeqLength: number;

  attentionType?: AttentionType;
  normType?: 'layernorm' | 'rmsnorm';
  activation?: 'gelu' | 'silu' | 'relu';
  gatedMLP?: boolean;
  useBias?: boolean;
  tiedEmbeddings?: boolean;
  useRotaryEmbed?: boolean;

  // MLA (Multi-head Latent Attention) — DeepSeek V2/V3/R1
  kvLoraRank?: number;
  qLoraRank?: number;
  qkNopeHeadDim?: number;
  qkRopeHeadDim?: number;
  vHeadDim?: number;

  // MoE
  numExperts?: number;
  numActiveExperts?: number;
  expertIntermediateSize?: number;
  moeLayerFrequency?: number;    // Every Nth layer is MoE (default 1 = all MoE)
  numSharedExperts?: number;     // Always-active experts per MoE layer (default 0)
  sharedExpertIntermediateSize?: number; // Shared expert MLP intermediate dim (default = expertIntermediateSize)
  firstKDenseLayers?: number;    // First K layers are always dense regardless of moeLayerFrequency (default 0)
  lastKDenseLayers?: number;     // Last K layers are always dense regardless of moeLayerFrequency (default 0; no built-in models use this yet)
  routingDeviceLimit?: number;   // Max EP groups each token can contact (DeepSeek V3: M=4, V2: M=6)
}

// Training dtype configuration
export interface DTypeConfig {
  params: DType;           // Parameter storage dtype
  compute: DType;          // Compute dtype
  gradients: DType;        // Gradient storage dtype
  optimizer: DType;        // Optimizer state dtype (usually fp32)
  activation: DType;       // Activation checkpointing dtype
}

// Default training dtypes
export const DEFAULT_DTYPE_CONFIG: DTypeConfig = {
  params: 'bf16',
  compute: 'bf16',
  gradients: 'fp32',
  optimizer: 'fp32',
  activation: 'bf16',
};

// Mixed precision presets
export const DTYPE_PRESETS: Record<string, DTypeConfig> = {
  fp32: {
    params: 'fp32',
    compute: 'fp32',
    gradients: 'fp32',
    optimizer: 'fp32',
    activation: 'fp32',
  },
  fp16: {
    params: 'fp16',
    compute: 'fp16',
    gradients: 'fp16',
    optimizer: 'fp32',
    activation: 'fp16',
  },
  bf16: {
    params: 'bf16',
    compute: 'bf16',
    gradients: 'bf16',
    optimizer: 'fp32',
    activation: 'bf16',
  },
  fp8: {
    params: 'fp8',
    compute: 'fp8',
    gradients: 'bf16',
    optimizer: 'fp32',
    activation: 'fp8',
  },
  tf32: {
    params: 'tf32',
    compute: 'tf32',
    gradients: 'fp32',
    optimizer: 'fp32',
    activation: 'tf32',
  },
  fp4: {
    params: 'fp4',
    compute: 'fp4',
    gradients: 'bf16',
    optimizer: 'fp32',
    activation: 'bf16',
  },
};
