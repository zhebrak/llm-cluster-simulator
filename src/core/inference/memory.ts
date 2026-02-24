/**
 * Inference Memory Calculations
 *
 * Memory breakdown for LLM inference differs from training:
 * - No gradient storage
 * - No optimizer states
 * - KV cache instead (grows with sequence length)
 * - Smaller activation memory (only forward pass)
 */

import type { ModelSpec } from '../../types/model.ts';
import type { GPUSpec } from '../../types/hardware.ts';
import { gpuCapacityBytes } from '../strategies/base.ts';
import type {
  InferencePrecision,
  InferenceMemoryBreakdown,
  InferenceConfig,
} from '../../types/inference.ts';
import { totalKVCacheMemory, getPrecisionBytes } from './kv-cache.ts';
import { moeParamSplit } from './latency.ts';

/**
 * Calculate model weights memory
 *
 * @param model - Model specification
 * @param precision - Weight precision
 * @returns Memory in bytes
 */
export function modelWeightsMemory(
  model: ModelSpec,
  precision: InferencePrecision = 'bf16'
): number {
  const bytesPerParam = getPrecisionBytes(precision);
  return model.totalParams * bytesPerParam;
}

/**
 * Calculate activation memory during inference
 * During inference, we only need forward pass activations (no gradients)
 * With modern frameworks, only one layer's activations are in memory at a time
 *
 * @param model - Model specification
 * @param batchSize - Batch size
 * @param seqLen - Current sequence length
 * @param precision - Activation precision
 * @returns Peak activation memory in bytes
 */
export function activationMemory(
  model: ModelSpec,
  batchSize: number,
  seqLen: number,
  precision: InferencePrecision = 'bf16'
): number {
  const bytesPerElement = getPrecisionBytes(precision);

  // For inference, we keep activations for one layer at a time
  // Main contributors:
  // 1. Hidden states: batch * seq * hidden
  // 2. Attention scores: batch * heads * seq * seq (for prefill)
  // 3. MLP intermediate: batch * seq * intermediate

  // Hidden states (input + output of layer)
  const hiddenStates = 2 * batchSize * seqLen * model.hiddenSize * bytesPerElement;

  // Attention scores (during prefill, this can be large)
  // With FlashAttention, this is much smaller
  const attentionScores = batchSize * model.numAttentionHeads *
    seqLen * seqLen * bytesPerElement;

  // MLP intermediate activations
  const mlpActivations = batchSize * seqLen * model.intermediateSize * bytesPerElement;

  // For inference without FlashAttention, use full attention scores
  // For inference with FlashAttention, this is reduced significantly
  // Return conservative estimate (without FlashAttention)
  return Math.max(hiddenStates, attentionScores, mlpActivations);
}

/**
 * Calculate activation memory with FlashAttention
 * FlashAttention tiles the computation and doesn't materialize full attention matrix
 */
export function activationMemoryWithFlashAttention(
  model: ModelSpec,
  batchSize: number,
  seqLen: number,
  precision: InferencePrecision = 'bf16'
): number {
  const bytesPerElement = getPrecisionBytes(precision);

  // With FlashAttention, we don't store full attention matrix
  // Only hidden states and MLP activations
  const hiddenStates = 2 * batchSize * seqLen * model.hiddenSize * bytesPerElement;
  const mlpActivations = batchSize * seqLen * model.intermediateSize * bytesPerElement;

  // Small tiling buffer for FlashAttention
  const blockSize = 128; // Typical block size
  const flashBuffer = batchSize * model.numAttentionHeads *
    blockSize * blockSize * bytesPerElement;

  return Math.max(hiddenStates, mlpActivations) + flashBuffer;
}

/**
 * Maximum KV cache fragmentation factor when paged attention is disabled.
 *
 * Without paged attention, serving systems pre-allocate contiguous KV cache
 * buffers to the model's max sequence length for each request, even if the
 * actual sequences are shorter. The waste ratio is maxSeqLen / actualSeqLen,
 * capped at 2.0×.
 *
 * Examples (Llama 3, maxSeqLen=128K):
 *   actualSeqLen=1536  → factor=2.0× (capped)
 *   actualSeqLen=64K   → factor=2.0× (capped)
 *   actualSeqLen=100K  → factor=1.28×
 *   actualSeqLen=128K  → factor=1.0× (no waste — already at max)
 *
 * This models intra-request pre-allocation waste only. Real systems also
 * suffer external fragmentation (scattered gaps across requests), which
 * paged attention eliminates via non-contiguous block mapping. The cap of
 * 2.0× is conservative — literature reports 60-80% total waste without
 * paging (Kwon et al., 2023).
 */
export const KV_CACHE_FRAGMENTATION_CAP = 2.0;

export function kvCacheFragmentationFactor(actualSeqLen: number, maxSeqLen: number): number {
  if (actualSeqLen <= 0 || maxSeqLen <= 0) return 1.0;
  return Math.min(maxSeqLen / actualSeqLen, KV_CACHE_FRAGMENTATION_CAP);
}

/**
 * Calculate CUDA/framework memory overhead
 * Typically 5-15% of model + KV cache memory
 */
export function calculateOverhead(
  weightsMemory: number,
  kvCacheMemory: number,
  overheadFactor: number = 0.1
): number {
  return (weightsMemory + kvCacheMemory) * overheadFactor;
}

/**
 * Calculate complete inference memory breakdown
 */
export function calculateInferenceMemory(
  model: ModelSpec,
  seqLen: number,
  batchSize: number,
  weightPrecision: InferencePrecision = 'bf16',
  kvCachePrecision: InferencePrecision = 'bf16',
  flashAttention: boolean = true
): InferenceMemoryBreakdown {
  const weights = modelWeightsMemory(model, weightPrecision);
  const kvCache = totalKVCacheMemory(model, seqLen, batchSize, kvCachePrecision);

  const activations = flashAttention
    ? activationMemoryWithFlashAttention(model, batchSize, seqLen, kvCachePrecision)
    : activationMemory(model, batchSize, seqLen, kvCachePrecision);

  const overhead = calculateOverhead(weights, kvCache);

  if (weights < 0 || kvCache < 0 || activations < 0) {
    throw new Error('Memory component cannot be negative');
  }

  return {
    weights,
    kvCache,
    activations,
    overhead,
    total: weights + kvCache + activations + overhead,
  };
}

/**
 * Calculate memory breakdown at different points during generation
 */
export function memoryTimeline(
  model: ModelSpec,
  inputSeqLen: number,
  outputSeqLen: number,
  batchSize: number,
  weightPrecision: InferencePrecision = 'bf16',
  kvCachePrecision: InferencePrecision = 'bf16',
  flashAttention: boolean = true
): {
  phase: 'prefill' | 'decode';
  seqLen: number;
  memory: InferenceMemoryBreakdown;
}[] {
  const timeline: {
    phase: 'prefill' | 'decode';
    seqLen: number;
    memory: InferenceMemoryBreakdown;
  }[] = [];

  // Prefill phase - peak memory with full input sequence
  timeline.push({
    phase: 'prefill',
    seqLen: inputSeqLen,
    memory: calculateInferenceMemory(
      model, inputSeqLen, batchSize, weightPrecision, kvCachePrecision, flashAttention
    ),
  });

  // Decode phase - memory grows as we generate tokens
  // Sample at a few points to avoid too many data points
  const samplePoints = [
    Math.floor(outputSeqLen * 0.25),
    Math.floor(outputSeqLen * 0.5),
    Math.floor(outputSeqLen * 0.75),
    outputSeqLen,
  ].filter((v, i, arr) => v > 0 && arr.indexOf(v) === i); // Remove duplicates and zeros

  for (const outputTokens of samplePoints) {
    const totalSeqLen = inputSeqLen + outputTokens;
    timeline.push({
      phase: 'decode',
      seqLen: totalSeqLen,
      memory: calculateInferenceMemory(
        model, totalSeqLen, batchSize, weightPrecision, kvCachePrecision, flashAttention
      ),
    });
  }

  return timeline;
}

/**
 * Calculate memory from inference config
 */
export function calculateMemoryFromConfig(config: InferenceConfig): InferenceMemoryBreakdown {
  const totalSeqLen = config.inputSeqLen + config.outputSeqLen;

  // If tensor parallelism or expert parallelism is enabled, use TP/EP-aware memory calculation
  const tp = config.tensorParallel ?? 1;
  const ep = config.expertParallel ?? 1;
  let result: InferenceMemoryBreakdown;
  if (tp > 1 || ep > 1) {
    result = calculateMemoryWithTP(
      config.modelSpec,
      totalSeqLen,
      config.batchSize,
      tp,
      config.weightPrecision,
      config.kvCachePrecision,
      ep,
      config.flashAttention
    );
  } else {
    result = calculateInferenceMemory(
      config.modelSpec,
      totalSeqLen,
      config.batchSize,
      config.weightPrecision,
      config.kvCachePrecision,
      config.flashAttention
    );
  }

  // Without paged attention, inflate KV cache by fragmentation factor
  if (!config.pagedAttention) {
    const factor = kvCacheFragmentationFactor(totalSeqLen, config.modelSpec.maxSeqLength);
    const inflatedKV = result.kvCache * factor;
    const overhead = calculateOverhead(result.weights, inflatedKV);
    result = { ...result, kvCache: inflatedKV, overhead,
      total: result.weights + inflatedKV + result.activations + overhead };
  }

  return result;
}

/**
 * Check if configuration fits in GPU memory
 */
export function validateMemoryFits(
  config: InferenceConfig,
  gpu: GPUSpec
): {
  fits: boolean;
  memory: InferenceMemoryBreakdown;
  gpuMemoryBytes: number;
  utilizationPercent: number;
  headroom: number;
} {
  const memory = calculateMemoryFromConfig(config);
  const gpuMemoryBytes = gpuCapacityBytes(gpu.memoryGB);
  const utilizationPercent = (memory.total / gpuMemoryBytes) * 100;
  const headroom = gpuMemoryBytes - memory.total;

  return {
    fits: memory.total <= gpuMemoryBytes,
    memory,
    gpuMemoryBytes,
    utilizationPercent,
    headroom,
  };
}

/**
 * Calculate maximum batch size that fits in GPU memory
 */
export function maxBatchSizeForGPU(
  model: ModelSpec,
  totalSeqLen: number,
  gpu: GPUSpec,
  weightPrecision: InferencePrecision = 'bf16',
  kvCachePrecision: InferencePrecision = 'bf16'
): number {
  const gpuMemoryBytes = gpuCapacityBytes(gpu.memoryGB);
  const weightsMemory = modelWeightsMemory(model, weightPrecision);

  // Reserve 15% for overhead and activations
  const availableForKV = (gpuMemoryBytes - weightsMemory) * 0.85;

  if (availableForKV <= 0) {
    return 0; // Model weights don't fit
  }

  // KV cache per sequence
  const kvCachePerSeq = totalKVCacheMemory(model, totalSeqLen, 1, kvCachePrecision);

  return Math.floor(availableForKV / kvCachePerSeq);
}

/**
 * Calculate memory with tensor parallelism and optional expert parallelism.
 * With TP, weights are split but KV cache is replicated (or replicated for MLA).
 * With EP, routed expert weights are additionally split across EP ranks,
 * while shared params (attention, embeddings, norms, shared experts, dense MLPs)
 * are only split by TP.
 */
export function calculateMemoryWithTP(
  model: ModelSpec,
  seqLen: number,
  batchSize: number,
  tpDegree: number,
  weightPrecision: InferencePrecision = 'bf16',
  kvCachePrecision: InferencePrecision = 'bf16',
  epDegree: number = 1,
  flashAttention: boolean = true
): InferenceMemoryBreakdown {
  // With EP, shared params split by TP only; routed expert params split by TP×EP
  const bytesPerParam = getPrecisionBytes(weightPrecision);
  const { sharedParams, routedExpertParams } = moeParamSplit(model);
  const weights = (sharedParams / tpDegree + routedExpertParams / (tpDegree * epDegree)) * bytesPerParam;

  // KV cache per TP rank
  let kvCachePerRank: number;
  if (model.attentionType === 'mla' && model.kvLoraRank && model.qkRopeHeadDim) {
    // MLA: compressed KV latent replicated across TP ranks (not split)
    kvCachePerRank = model.numLayers * (model.kvLoraRank + model.qkRopeHeadDim) *
      getPrecisionBytes(kvCachePrecision) * seqLen * batchSize;
  } else {
    // Standard: KV heads split across TP ranks
    const kvHeadsPerRank = Math.ceil(model.numKvHeads / tpDegree);
    kvCachePerRank = 2 * model.numLayers * kvHeadsPerRank *
      model.headDim * getPrecisionBytes(kvCachePrecision) * seqLen * batchSize;
  }

  // Activations are also sharded
  const activations = (flashAttention
    ? activationMemoryWithFlashAttention(model, batchSize, seqLen, kvCachePrecision)
    : activationMemory(model, batchSize, seqLen, kvCachePrecision)
  ) / tpDegree;

  const overhead = calculateOverhead(weights, kvCachePerRank);

  return {
    weights,
    kvCache: kvCachePerRank,
    activations,
    overhead,
    total: weights + kvCachePerRank + activations + overhead,
  };
}

