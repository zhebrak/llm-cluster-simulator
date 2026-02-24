/**
 * Models module exports
 */

// Primitives
export {
  paramMemory,
  matmulFlops,
  batchedMatmulFlops,
  createEmbeddingLayer,
  createAttentionLayer,
  createMLPLayer,
  createNormLayer,
  createMoELayer,
  createOutputLayer,
  createTransformerBlock,
  buildModelSpec,
} from './primitives.ts';

// Architectures
export {
  GPT3_CONFIGS,
  LLAMA2_CONFIGS,
  LLAMA3_CONFIGS,
  LLAMA4_CONFIGS,
  MISTRAL_CONFIGS,
  DEEPSEEK_CONFIGS,
  QWEN_CONFIGS,
  QWEN3_CONFIGS,
  GEMMA3_CONFIGS,
  GROK_CONFIGS,
  PHI4_CONFIGS,
  ALL_MODEL_CONFIGS,
  MODEL_FAMILIES,
  POPULAR_MODELS,
  getModelConfig,
} from './architectures.ts';

// Registry
export type { ModelMetadata } from './registry.ts';
export {
  modelRegistry,
  getModel,
} from './registry.ts';

// MoE utilities
export {
  calculateMoEMemory,
} from './moe.ts';
