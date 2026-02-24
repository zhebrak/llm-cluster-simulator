/**
 * KV Cache calculations for LLM inference
 *
 * The KV cache stores Key and Value matrices from previous tokens
 * to avoid recomputation during autoregressive generation.
 *
 * Formula: Memory = 2 * num_layers * num_kv_heads * head_dim * bytes_per_element * seq_len * batch_size
 * The "2" accounts for both K and V matrices.
 */

import type { ModelSpec } from '../../types/model.ts';
import type {
  InferencePrecision,
  KVCacheConfig,
  KVCacheState,
} from '../../types/inference.ts';

/**
 * Get bytes per element for a precision type
 */
export function getPrecisionBytes(precision: InferencePrecision): number {
  const specs: Record<InferencePrecision, number> = {
    fp32: 4,
    fp16: 2,
    bf16: 2,
    fp8: 1,
    int8: 1,
    int4: 0.5,
    fp4: 0.5,
    // GGUF quantization — bpw / 8
    q2_k:   3.00 / 8,
    q3_k_m: 3.89 / 8,
    q4_k_m: 4.83 / 8,
    q5_k_m: 5.67 / 8,
    q6_k:   6.57 / 8,
    q8_0:   8.50 / 8,
  };
  return specs[precision];
}

/**
 * Calculate KV cache memory per token for a single sequence
 * This is a key formula for understanding inference memory requirements.
 *
 * @param model - Model specification
 * @param precision - KV cache precision
 * @returns Bytes per token for KV cache
 */
export function kvCachePerToken(
  model: ModelSpec,
  precision: InferencePrecision = 'bf16'
): number {
  const bytesPerElement = getPrecisionBytes(precision);

  if (model.attentionType === 'mla' && model.kvLoraRank && model.qkRopeHeadDim) {
    // MLA: store compressed KV latent + decoupled RoPE key per layer
    // NOT split by TP (replicated across ranks — all heads need full latent)
    // No factor of 2: single compressed latent, not separate K/V
    const result = model.numLayers * (model.kvLoraRank + model.qkRopeHeadDim) * bytesPerElement;
    if (result <= 0) throw new Error('KV cache per token must be positive');
    return result;
  }

  // Standard: 2 * layers * kv_heads * head_dim * bytes
  if (model.numKvHeads <= 0) throw new Error(`numKvHeads must be positive, got ${model.numKvHeads}`);
  if (model.headDim <= 0) throw new Error(`headDim must be positive, got ${model.headDim}`);
  return 2 * model.numLayers * model.numKvHeads * model.headDim * bytesPerElement;
}

/**
 * Calculate total KV cache memory for a batch
 *
 * @param model - Model specification
 * @param seqLen - Current sequence length (prompt + generated tokens)
 * @param batchSize - Number of concurrent sequences
 * @param precision - KV cache precision
 * @returns Total KV cache memory in bytes
 */
export function totalKVCacheMemory(
  model: ModelSpec,
  seqLen: number,
  batchSize: number,
  precision: InferencePrecision = 'bf16'
): number {
  return kvCachePerToken(model, precision) * seqLen * batchSize;
}

/**
 * Calculate KV cache reduction factor from GQA/MQA
 *
 * MHA: numKvHeads == numAttentionHeads → factor = 1
 * GQA: numKvHeads < numAttentionHeads → factor = numKvHeads/numAttentionHeads
 * MQA: numKvHeads == 1 → factor = 1/numAttentionHeads
 *
 * @param model - Model specification
 * @returns Reduction factor (e.g., 0.25 for 4x reduction)
 */
export function kvCacheReductionFactor(model: ModelSpec): number {
  if (model.attentionType === 'mla' && model.kvLoraRank && model.qkRopeHeadDim) {
    // MLA reduction vs full MHA: (kv_lora_rank + rope_dim) vs 2 * nH * head_dim
    const vHeadDim = model.vHeadDim ?? model.headDim;
    const mhaPerLayer = 2 * model.numAttentionHeads * vHeadDim;
    return (model.kvLoraRank + model.qkRopeHeadDim) / mhaPerLayer;
  }
  return model.numKvHeads / model.numAttentionHeads;
}

/**
 * Get attention type string for display
 */
export function getAttentionTypeString(model: ModelSpec): string {
  if (model.attentionType === 'mla') return 'MLA';
  if (model.numKvHeads === model.numAttentionHeads) {
    return 'MHA';
  } else if (model.numKvHeads === 1) {
    return 'MQA';
  } else {
    return `GQA (${model.numKvHeads} KV heads)`;
  }
}

/**
 * Calculate maximum sequence length that fits in available memory
 * given model weights and other memory requirements
 *
 * @param model - Model specification
 * @param availableMemoryBytes - Available GPU memory for KV cache
 * @param batchSize - Number of concurrent sequences
 * @param precision - KV cache precision
 * @returns Maximum sequence length
 */
export function maxSequenceLengthForMemory(
  model: ModelSpec,
  availableMemoryBytes: number,
  batchSize: number,
  precision: InferencePrecision = 'bf16'
): number {
  const bytesPerToken = kvCachePerToken(model, precision);
  return Math.floor(availableMemoryBytes / (bytesPerToken * batchSize));
}

/**
 * Calculate maximum batch size that fits for a given sequence length
 *
 * @param model - Model specification
 * @param availableMemoryBytes - Available GPU memory for KV cache
 * @param seqLen - Target sequence length
 * @param precision - KV cache precision
 * @returns Maximum batch size
 */
export function maxBatchSizeForSequence(
  model: ModelSpec,
  availableMemoryBytes: number,
  seqLen: number,
  precision: InferencePrecision = 'bf16'
): number {
  const bytesPerToken = kvCachePerToken(model, precision);
  return Math.floor(availableMemoryBytes / (bytesPerToken * seqLen));
}

/**
 * Create KV cache configuration from model spec
 */
export function createKVCacheConfig(
  model: ModelSpec,
  precision: InferencePrecision = 'bf16',
  options: {
    pagedAttention?: boolean;
    blockSize?: number;
    slidingWindow?: number;
  } = {}
): KVCacheConfig {
  return {
    numLayers: model.numLayers,
    numKvHeads: model.numKvHeads,
    headDim: model.headDim,
    maxSeqLen: model.maxSeqLength,
    precision,
    pagedAttention: options.pagedAttention ?? false,
    blockSize: options.blockSize ?? 16,
    slidingWindow: options.slidingWindow,
  };
}

/**
 * Calculate KV cache state at a given point in generation
 */
export function calculateKVCacheState(
  model: ModelSpec,
  currentSeqLen: number,
  batchSize: number,
  precision: InferencePrecision = 'bf16',
  gpuMemoryBytes?: number
): KVCacheState {
  const memoryPerToken = kvCachePerToken(model, precision);
  const memoryUsed = memoryPerToken * currentSeqLen * batchSize;

  // Calculate utilization if GPU memory is provided
  let utilizationPercent = 0;
  if (gpuMemoryBytes) {
    utilizationPercent = (memoryUsed / gpuMemoryBytes) * 100;
  }

  return {
    currentSeqLen,
    batchSize,
    memoryUsed,
    memoryPerToken,
    utilizationPercent,
  };
}

/**
 * Validate KV cache fits in GPU memory
 */
export function validateKVCacheFits(
  model: ModelSpec,
  seqLen: number,
  batchSize: number,
  gpuMemoryBytes: number,
  weightsMemoryBytes: number,
  precision: InferencePrecision = 'bf16'
): {
  fits: boolean;
  kvCacheMemory: number;
  totalRequired: number;
  available: number;
  deficit: number;
} {
  const kvCacheMemory = totalKVCacheMemory(model, seqLen, batchSize, precision);
  const overheadFactor = 1.1; // 10% overhead for activations, etc.
  const totalRequired = weightsMemoryBytes + kvCacheMemory * overheadFactor;
  const available = gpuMemoryBytes;
  const deficit = Math.max(0, totalRequired - available);

  return {
    fits: totalRequired <= available,
    kvCacheMemory,
    totalRequired,
    available,
    deficit,
  };
}
