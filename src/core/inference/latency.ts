/**
 * Latency Model for LLM Inference
 *
 * Key metrics:
 * - TTFT (Time to First Token): Time from prompt submission to first generated token
 *   - Dominated by prefill phase (compute-bound for most sequences)
 * - TPOT (Time Per Output Token): Time to generate each subsequent token
 *   - Memory-bandwidth bound (reading weights + KV cache)
 *
 * The fundamental insight: prefill is compute-bound, decode is memory-bound.
 */
// See docs/INFERENCE.md for latency model formulas and calibration.

import type { ModelSpec } from '../../types/model.ts';
import type { GPUSpec } from '../../types/hardware.ts';
import { gpuCapacityBytes } from '../strategies/base.ts';
import type {
  InferencePrecision,
  LatencyMetrics,
  ThroughputMetrics,
  UtilizationMetrics,
  InferenceConfig,
} from '../../types/inference.ts';
import { getPrecisionBytes, totalKVCacheMemory } from './kv-cache.ts';
import { modelWeightsMemory, calculateMemoryFromConfig } from './memory.ts';
import { getMatmulSaturationFactor, getMemoryBandwidthScaling } from '../hardware/gpu.ts';

let _prefillResidual = 0.40;
export function getPrefillResidual(): number { return _prefillResidual; }
export function setPrefillResidual(v: number) { _prefillResidual = v; }

let _decodeSamplingOverheadMs = 0.03;
export function getDecodeSamplingOverheadMs(): number { return _decodeSamplingOverheadMs; }
export function setDecodeSamplingOverheadMs(v: number) { _decodeSamplingOverheadMs = v; }

/** Per-round AllReduce latency for tree algorithm (ms).
 * Tree AllReduce uses 2*ceil(log2(tp)) rounds (reduce + broadcast phases).
 * Each round: kernel dispatch, NVLink/PCIe hop, barrier synchronization.
 * Based on NCCL small-message benchmarks: TP=2~6us, TP=4~22us, TP=8~35us. */
let _nvlinkPerRoundMs = 0.005;  // 5us per tree round on NVLink
export function getNvlinkPerRoundMs(): number { return _nvlinkPerRoundMs; }
export function setNvlinkPerRoundMs(v: number) { _nvlinkPerRoundMs = v; }

let _pciePerRoundMs = 0.015;    // 15us per tree round on PCIe
export function getPciePerRoundMs(): number { return _pciePerRoundMs; }
export function setPciePerRoundMs(v: number) { _pciePerRoundMs = v; }

// TP AllReduce BW utilization (0.80 from NCCL profiling)
let _tpCommEfficiency = 0.80;
export function getTpCommEfficiency(): number { return _tpCommEfficiency; }
export function setTpCommEfficiency(v: number) { _tpCommEfficiency = v; }

// EP all-to-all overlap fraction in prefill (heavily overlapped with compute)
let _epPrefillOverlap = 0.15;
export function getEpPrefillOverlap(): number { return _epPrefillOverlap; }
export function setEpPrefillOverlap(v: number) { _epPrefillOverlap = v; }

// Decode bandwidth efficiency sigmoid parameters
let _bwEffFloor = 0.35;
export function getBwEffFloor(): number { return _bwEffFloor; }
export function setBwEffFloor(v: number) { _bwEffFloor = v; }

let _bwEffScale = 5.0;
export function getBwEffScale(): number { return _bwEffScale; }
export function setBwEffScale(v: number) { _bwEffScale = v; }

// CB scheduling overhead base fraction
let _cbSchedulingBase = 0.01;
export function getCbSchedulingBase(): number { return _cbSchedulingBase; }
export function setCbSchedulingBase(v: number) { _cbSchedulingBase = v; }

// CB max prefill interference from concurrent decode slots
let _cbPrefillInterferenceMax = 0.10;
export function getCbPrefillInterferenceMax(): number { return _cbPrefillInterferenceMax; }
export function setCbPrefillInterferenceMax(v: number) { _cbPrefillInterferenceMax = v; }

/** Prefill MFU derived from GPU architecture. H100 SXM bf16 = 0.40 (reference). */
export function getPrefillEfficiency(gpu: GPUSpec, _precision: InferencePrecision = 'bf16'): number {
  // Always use 'bf16' for memBWScaling to avoid FP8 Amdahl double-counting
  // (FP8 speedup is already captured by getGPUTFLOPS)
  return getMemoryBandwidthScaling(gpu, 'bf16') * _prefillResidual;
}

/**
 * Decompose model params into shared (replicated) and routed expert (EP-distributable).
 * Shared includes: attention, embeddings, norms, output head, shared experts, dense layer MLPs.
 * Routed expert params = numExperts × paramsPerExpert × numMoELayers.
 */
export function moeParamSplit(model: ModelSpec): { sharedParams: number; routedExpertParams: number } {
  if (!model.isMoE || !model.numExperts) {
    return { sharedParams: model.totalParams, routedExpertParams: 0 };
  }
  const expertIntermediate = model.expertIntermediateSize ?? model.intermediateSize;
  // gated: gate + up + down projections (3 matrices); non-gated: up + down (2 matrices)
  const mlpMatrices = model.gatedMLP ? 3 : 2;
  const paramsPerExpert = mlpMatrices * model.hiddenSize * expertIntermediate;
  const numMoELayers = model.numMoELayers ?? model.numLayers;
  const routedExpertParams = model.numExperts * paramsPerExpert * numMoELayers;
  // Shared = everything else (attention, embeddings, norms, output head, shared experts, dense MLPs)
  const sharedParams = model.totalParams - routedExpertParams;
  if (sharedParams < 0) {
    throw new Error(`Routed expert params (${routedExpertParams}) exceed totalParams (${model.totalParams})`);
  }
  return { sharedParams, routedExpertParams };
}

/**
 * Calculate weight bytes that must be read per decode step, accounting for MoE batch effects.
 *
 * For MoE models, the number of expert weights read depends on how many unique experts
 * are activated across the batch. With top-k routing and uniform expert selection:
 *   fractionExperts = 1 - (1 - numActive/numExperts)^batchSize
 *
 * At batch=1: ≈ activeParams × bytes (only active experts read)
 * At large batch: → totalParams × bytes (nearly all experts touched)
 */
export function moeWeightBytesPerStep(
  model: ModelSpec,
  batchSize: number,
  precision: InferencePrecision
): number {
  const bytes = getPrecisionBytes(precision);
  const { sharedParams, routedExpertParams } = moeParamSplit(model);

  if (routedExpertParams === 0) {
    return (model.activeParams ?? model.totalParams) * bytes;
  }

  const numActive = model.numActiveExperts ?? 2;
  const numExperts = model.numExperts!;
  // Fraction of experts touched by at least one token in the batch (uniform routing assumption)
  const fractionTouched = 1 - Math.pow(1 - numActive / numExperts, batchSize);
  return (sharedParams + fractionTouched * routedExpertParams) * bytes;
}

/**
 * Effective memory bandwidth utilization depends on weight volume per step.
 * Small models: kernel launch overhead dominates → low utilization (40-50%)
 * Large models: weight reads dominate → high utilization (75-85%)
 * Floor at 0.35 for tiny draft models / embeddings-only passes.
 */
export function getBandwidthEfficiency(totalBytes: number): number {
  const totalGB = totalBytes / (1024 ** 3);
  return Math.max(_bwEffFloor, Math.min(0.85, _bwEffFloor + 0.50 * (1 - 1 / (1 + totalGB / _bwEffScale))));
}

/**
 * Dequantization overhead for quantized weight/KV cache reads.
 * Quantized formats require on-the-fly conversion to BF16/FP16 before matmul:
 * group metadata reads (scales/zeros), unpacking ALU ops, reduced kernel
 * occupancy. Overhead scales with quantization aggressiveness.
 */
export function getQuantizationOverhead(precision: InferencePrecision, gpu: GPUSpec): number {
  switch (precision) {
    // Sub-4-bit GGUF: k-quants with fused GGML kernels
    case 'q2_k':
    case 'q3_k_m':
      return 1.15;
    // 4-bit generic (GPTQ/AWQ): separate dequant kernels
    case 'int4':
      return 1.20;
    // 4-bit GGUF: highly optimized fused GGML dequant
    case 'q4_k_m':
      return 1.10;
    // 5-6 bit GGUF: simple dequant, well-optimized GGML kernels
    case 'q5_k_m':
    case 'q6_k':
      return 1.08;
    // 8-bit: simple byte-level dequant
    case 'int8':
    case 'q8_0':
      return 1.10;
    // FP8: near-native on Transformer Engine GPUs, software dequant otherwise
    case 'fp8':
      return gpu.hasTransformerEngine ? 1.05 : 1.10;
    // FP4: Blackwell-only, native tensor core support
    case 'fp4':
      return 1.05;
    // Native precision: no dequant
    default:
      return 1.0;  // bf16, fp16, fp32
  }
}

/**
 * Get effective TFLOPS for a precision type
 */
export function getGPUTFLOPS(
  gpu: GPUSpec,
  precision: InferencePrecision
): number {
  let result: number;
  switch (precision) {
    case 'fp32':
      result = gpu.fp32TFLOPS;
      break;
    case 'fp16':
      result = gpu.fp16TFLOPS;
      break;
    case 'bf16':
      result = gpu.bf16TFLOPS || gpu.fp16TFLOPS;
      break;
    case 'fp8':
      result = gpu.fp8TFLOPS || gpu.bf16TFLOPS || gpu.fp16TFLOPS;
      break;
    case 'int8':
      result = gpu.int8TOPS || gpu.bf16TFLOPS || gpu.fp16TFLOPS;
      break;
    case 'int4':   // W4A16: GPTQ, AWQ — dequant to FP16, compute at FP16
      result = gpu.bf16TFLOPS || gpu.fp16TFLOPS;
      break;
    case 'fp4':
      result = gpu.fp4TFLOPS || gpu.int4TOPS || gpu.int8TOPS || gpu.bf16TFLOPS || gpu.fp16TFLOPS;
      break;
    // GGUF quantization — all W*A16: weights dequantized to FP16, compute at FP16
    case 'q2_k':
    case 'q3_k_m':
    case 'q4_k_m':
    case 'q5_k_m':
    case 'q6_k':
    case 'q8_0':
      result = gpu.bf16TFLOPS || gpu.fp16TFLOPS;
      break;
    default:
      result = gpu.bf16TFLOPS || gpu.fp16TFLOPS;
  }
  if (!result || result <= 0) {
    throw new Error(`GPU ${gpu.name} has no usable compute capability for ${precision} (all TFLOPS values are zero)`);
  }
  return result;
}

/**
 * Calculate FLOPs for prefill (processing the prompt)
 *
 * FLOPs ≈ 2 * model_params * input_tokens
 * The "2" comes from multiply-accumulate operations
 */
export function prefillFLOPs(model: ModelSpec, inputTokens: number): number {
  // Each token requires ~2 * params FLOPs for forward pass (linear projections)
  // For MoE models, only active experts compute per token
  const activeParams = model.activeParams ?? model.totalParams;
  const linearFLOPs = 2 * activeParams * inputTokens;

  // Attention score FLOPs: QK^T + scores*V (not included in 2*activeParams)
  // Causal mask: token i attends to (i+1) positions, sum = S*(S+1)/2
  let attnPerLayerPerPosition: number;
  if (model.attentionType === 'mla' && model.kvLoraRank && model.qkRopeHeadDim) {
    attnPerLayerPerPosition = 2 * model.numAttentionHeads * (2 * model.kvLoraRank + model.qkRopeHeadDim);
  } else {
    attnPerLayerPerPosition = 4 * model.numAttentionHeads * model.headDim;
  }
  const attnFLOPs = attnPerLayerPerPosition * inputTokens * (inputTokens + 1) / 2 * model.numLayers;
  return linearFLOPs + attnFLOPs;
}

/**
 * Calculate FLOPs for a single decode step
 * Similar to prefill but for a single token
 */
export function decodeFLOPs(model: ModelSpec, seqLen: number = 0): number {
  // Single token forward pass (linear projections)
  // For MoE models, only active experts compute per token
  const activeParams = model.activeParams ?? model.totalParams;
  const linearFLOPs = 2 * activeParams;
  if (seqLen <= 0) return linearFLOPs;

  // Attention score FLOPs: QK^T + scores*V (not included in 2*activeParams)
  // Decode: single token attending to seqLen previous positions
  let attnPerLayer: number;
  if (model.attentionType === 'mla' && model.kvLoraRank && model.qkRopeHeadDim) {
    attnPerLayer = 2 * model.numAttentionHeads * (2 * model.kvLoraRank + model.qkRopeHeadDim) * seqLen;
  } else {
    attnPerLayer = 4 * model.numAttentionHeads * model.headDim * seqLen;
  }
  return linearFLOPs + attnPerLayer * model.numLayers;
}

/**
 * Estimate Time to First Token (TTFT)
 *
 * TTFT = max(compute_time, memory_load_time)
 * - Compute time: FLOPs / (GPU TFLOPS * MFU)
 * - Memory time: weights_bytes / bandwidth
 *
 * Prefill is typically compute-bound for reasonable sequence lengths.
 *
 * @param model - Model specification
 * @param inputTokens - Number of prompt tokens
 * @param gpu - GPU specification
 * @param precision - Compute precision
 * @param batchSize - Number of sequences in batch (prefill FLOPs scale with batch)
 * @returns TTFT in milliseconds
 */
export function estimateTTFT(
  model: ModelSpec,
  inputTokens: number,
  gpu: GPUSpec,
  precision: InferencePrecision = 'bf16',
  batchSize: number = 1,
  flashAttention: boolean = true,
): number {
  // FLOPs for processing entire prompt batch
  // All sequences in the batch are prefilled together
  const flops = prefillFLOPs(model, inputTokens) * batchSize;

  // GPU compute capacity in FLOPS, adjusted for matmul saturation
  const gpuTFLOPS = getGPUTFLOPS(gpu, precision);
  const tokensInPrefill = inputTokens * batchSize;
  const saturationFactor = getMatmulSaturationFactor(tokensInPrefill, model.hiddenSize, gpu);
  const gpuFLOPS = gpuTFLOPS * 1e12 * saturationFactor;
  if (gpuFLOPS <= 0) throw new Error('Effective GPU FLOPS is zero — cannot compute TTFT');

  // Compute time (accounting for real-world efficiency)
  const computeTimeSeconds = flops / (gpuFLOPS * getPrefillEfficiency(gpu, precision));

  // Memory load time for weights
  // Prefill processes batchSize × inputTokens tokens through expert routing —
  // nearly all experts are touched (coupon-collector over S×B tokens).
  // CB already passes batchSize=1 here (single-request prefill), so this
  // correctly uses inputTokens for CB and batchSize×inputTokens for static.
  const weightsBytes = moeWeightBytesPerStep(model, batchSize * inputTokens, precision);
  const bandwidthBytesPerSec = gpu.memoryBandwidthTBps * 1e12;
  // No bandwidth efficiency penalty for prefill: per-layer weight reads are large (MBs)
  // and pipelined with compute kernels, achieving near-peak HBM utilization regardless
  // of total model size. The getBandwidthEfficiency() sigmoid is for decode physics only.
  const memoryTimeSeconds = weightsBytes / bandwidthBytesPerSec;

  // Prefill is compute-bound, but we take max to be safe
  let ttftMs = Math.max(computeTimeSeconds, memoryTimeSeconds) * 1000;

  // Without FA, attention materializes full seq×seq matrix in HBM per layer.
  // Additive because attention HBM occurs during attention phase, not overlapped with weight reads.
  if (!flashAttention) {
    const attentionHBMBytes = model.numLayers * model.numAttentionHeads
      * inputTokens * inputTokens
      * getPrecisionBytes(precision) * batchSize * 2.5;
    const attentionHBMTimeMs = (attentionHBMBytes / bandwidthBytesPerSec) * 1000;
    ttftMs += attentionHBMTimeMs;
  }

  return ttftMs;
}

/**
 * Estimate Time Per Output Token (TPOT)
 *
 * TPOT is memory-bandwidth bound because each token generation requires:
 * 1. Reading model weights (amortized across batch)
 * 2. Reading KV cache for attention
 * 3. Writing new KV entries
 *
 * @param model - Model specification
 * @param currentSeqLen - Current sequence length (prompt + generated so far)
 * @param batchSize - Batch size
 * @param gpu - GPU specification
 * @param precision - Compute precision
 * @returns TPOT in milliseconds
 */
export function estimateTPOT(
  model: ModelSpec,
  currentSeqLen: number,
  batchSize: number,
  gpu: GPUSpec,
  precision: InferencePrecision = 'bf16',
  kvCachePrecision?: InferencePrecision
): number {
  // Bytes that must be read per token generation
  // For MoE: weight bytes depend on how many experts are touched by the batch
  const weightsBytes = moeWeightBytesPerStep(model, batchSize, precision);
  const kvCacheBytes = totalKVCacheMemory(model, currentSeqLen, batchSize, kvCachePrecision ?? precision);

  // Weights are read once per step, amortized across batch
  // KV cache must be read for each sequence
  // Quantized formats inflate effective bytes due to dequantization overhead
  const weightDequantOverhead = getQuantizationOverhead(precision, gpu);
  const kvDequantOverhead = getQuantizationOverhead(kvCachePrecision ?? precision, gpu);
  const totalBytesPerStep = weightsBytes * weightDequantOverhead
                          + kvCacheBytes * kvDequantOverhead;

  // Memory bandwidth
  if (gpu.memoryBandwidthTBps <= 0) throw new Error(`GPU ${gpu.name} has zero memory bandwidth`);
  const bandwidthBytesPerSec = gpu.memoryBandwidthTBps * 1e12;

  // Memory-bound: time = bytes / bandwidth
  // Effective bandwidth utilization depends on total HBM streaming volume (weights + KV)
  const bandwidthEfficiency = getBandwidthEfficiency(weightsBytes + kvCacheBytes);
  const timePerStepSeconds = totalBytesPerStep / (bandwidthBytesPerSec * bandwidthEfficiency);

  // TPOT = inter-token latency = time per decode step
  // Each step reads weights once + KV cache for all sequences, producing one token per sequence
  const memoryTimeMs = timePerStepSeconds * 1000;

  // Compute floor — decode becomes compute-bound at large batch sizes.
  // Matmul saturation: batchSize tokens × hiddenSize per-GPU matmul
  const flopsPerStep = decodeFLOPs(model, currentSeqLen) * batchSize;
  const decodeSaturation = getMatmulSaturationFactor(batchSize, model.hiddenSize, gpu);
  const gpuFLOPS = getGPUTFLOPS(gpu, precision) * 1e12 * decodeSaturation;
  const computeTimeMs = (flopsPerStep / (gpuFLOPS * getPrefillEfficiency(gpu, precision))) * 1000;

  return Math.max(memoryTimeMs, computeTimeMs) + _decodeSamplingOverheadMs;
}

/**
 * Estimate average TPOT over the entire generation
 * TPOT increases as sequence grows (more KV cache to read)
 */
export function estimateAverageTPOT(
  model: ModelSpec,
  inputSeqLen: number,
  outputSeqLen: number,
  batchSize: number,
  gpu: GPUSpec,
  precision: InferencePrecision = 'bf16',
  kvCachePrecision?: InferencePrecision
): number {
  // Average sequence length during decode
  const avgSeqLen = inputSeqLen + outputSeqLen / 2;
  return estimateTPOT(model, avgSeqLen, batchSize, gpu, precision, kvCachePrecision);
}

/**
 * Calculate full latency metrics
 */
export function calculateLatencyMetrics(
  model: ModelSpec,
  inputSeqLen: number,
  outputSeqLen: number,
  batchSize: number,
  gpu: GPUSpec,
  precision: InferencePrecision = 'bf16',
  kvCachePrecision?: InferencePrecision,
  flashAttention: boolean = true,
): LatencyMetrics {
  const ttft = estimateTTFT(model, inputSeqLen, gpu, precision, batchSize, flashAttention);

  // TPOT = inter-token latency (decode step time)
  const initialTpot = estimateTPOT(model, inputSeqLen, batchSize, gpu, precision, kvCachePrecision);
  const finalTpot = estimateTPOT(model, inputSeqLen + outputSeqLen, batchSize, gpu, precision, kvCachePrecision);
  const avgTpot = (initialTpot + finalTpot) / 2;

  const prefillTime = ttft;
  // Decode: outputSeqLen steps, each taking avgTpot wall-clock time
  const decodeTime = avgTpot * outputSeqLen;
  const totalLatency = prefillTime + decodeTime;

  return {
    ttft,
    tpot: avgTpot,
    totalLatency,
    prefillTime,
    decodeTime,
  };
}

/**
 * Calculate throughput metrics
 */
export function calculateThroughputMetrics(
  model: ModelSpec,
  inputSeqLen: number,
  outputSeqLen: number,
  batchSize: number,
  gpu: GPUSpec,
  precision: InferencePrecision = 'bf16',
  kvCachePrecision?: InferencePrecision,
  flashAttention: boolean = true,
): ThroughputMetrics {
  const latency = calculateLatencyMetrics(
    model, inputSeqLen, outputSeqLen, batchSize, gpu, precision, kvCachePrecision, flashAttention
  );

  // Tokens generated per second (output tokens)
  const tokensPerSecond = (outputSeqLen * batchSize) / (latency.totalLatency / 1000);

  // Requests completed per second
  const requestsPerSecond = batchSize / (latency.totalLatency / 1000);

  // Prefill throughput (input tokens processed per second)
  const prefillTokensPerSecond = (inputSeqLen * batchSize) / (latency.prefillTime / 1000);

  // Decode throughput (output tokens per second)
  const decodeTokensPerSecond = (outputSeqLen * batchSize) / (latency.decodeTime / 1000);

  return {
    tokensPerSecond,
    requestsPerSecond,
    prefillTokensPerSecond,
    decodeTokensPerSecond,
  };
}

/**
 * Calculate utilization metrics using roofline model.
 *
 * Arithmetic intensity (ops/byte) determines which resource is the bottleneck:
 * - Prefill: intensity = 2*P*S / weightsBytes ≈ S/bytesPerParam (scales with seq len)
 * - Decode:  intensity = 2*P*B / (weightsBytes + kvCacheBytes)  (scales with batch)
 *
 * GPU ridge point = peakFLOPS / peakBandwidth (ops/byte).
 * If intensity > ridge → compute-bound (attainment = 1.0)
 * If intensity < ridge → memory-bound (attainment < 1.0, proportional to intensity/ridge)
 *
 * Roofline attainment = min(intensity / ridge, 1.0)
 *   1.0 = fully compute-bound; low values = memory-bandwidth-bound
 */
export function calculateUtilizationMetrics(
  model: ModelSpec,
  inputSeqLen: number,
  outputSeqLen: number,
  batchSize: number,
  gpu: GPUSpec,
  precision: InferencePrecision = 'bf16',
  totalMemoryOverride?: number,
  kvCachePrecision?: InferencePrecision
): UtilizationMetrics {
  const kvPrec = kvCachePrecision ?? precision;
  const gpuTFLOPS = getGPUTFLOPS(gpu, precision);
  const peakFLOPS = gpuTFLOPS * 1e12;
  const peakBandwidth = gpu.memoryBandwidthTBps * 1e12;
  const ridgePoint = peakFLOPS / peakBandwidth; // ops/byte

  // ── Prefill: compute utilization ──
  // Prefill reads all weights (all experts touched during batched prefill), does 2*activeP*S*B FLOPs
  const prefillWeightsBytes = modelWeightsMemory(model, precision);
  const prefillFlops = prefillFLOPs(model, inputSeqLen) * batchSize;
  const prefillIntensity = prefillWeightsBytes > 0 ? prefillFlops / prefillWeightsBytes : 0;
  const computeUtilization = ridgePoint > 0
    ? Math.min(prefillIntensity / ridgePoint, 1.0) : 0;

  // ── Decode: compute utilization (how far below ridge) ──
  // Decode reads batch-aware weight bytes + KV cache per step, does 2*activeP*B FLOPs
  const avgSeqLen = inputSeqLen + outputSeqLen / 2;
  const kvCacheBytes = totalKVCacheMemory(model, avgSeqLen, batchSize, kvPrec);
  const decodeWeightsBytes = moeWeightBytesPerStep(model, batchSize, precision);
  const decodeFlops = decodeFLOPs(model, avgSeqLen) * batchSize;
  const decodeBytes = decodeWeightsBytes + kvCacheBytes;
  const decodeIntensity = decodeBytes > 0 ? decodeFlops / decodeBytes : 0;
  // Fraction of peak compute achieved during decode (intensity/ridge)
  // When memory-bound (intensity < ridge): this value is low, showing compute underutilization
  const decodeComputeFraction = ridgePoint > 0
    ? Math.min(decodeIntensity / ridgePoint, 1.0) : 0;

  // Memory capacity utilization
  const gpuMemoryBytes = gpuCapacityBytes(gpu.memoryGB);
  const totalMemory = totalMemoryOverride ?? (prefillWeightsBytes + totalKVCacheMemory(
    model, inputSeqLen + outputSeqLen, batchSize, kvPrec
  ));
  const memoryCapacityUtilization = totalMemory / gpuMemoryBytes;

  // Determine bottleneck
  const prefillComputeBound = prefillIntensity >= ridgePoint;
  const decodeMemoryBound = decodeIntensity < ridgePoint;

  let bottleneck: 'compute' | 'memory_bandwidth' | 'memory_capacity';
  if (memoryCapacityUtilization > 0.9) {
    bottleneck = 'memory_capacity';
  } else if (decodeMemoryBound) {
    bottleneck = 'memory_bandwidth';
  } else {
    bottleneck = 'compute';
  }

  return {
    computeUtilization,
    rooflineAttainment: decodeComputeFraction,
    memoryCapacityUtilization,
    isComputeBound: prefillComputeBound,
    isMemoryBound: decodeMemoryBound,
    bottleneck,
  };
}

/**
 * Calculate latency breakdown by phase (for visualization)
 */
export function latencyBreakdown(
  model: ModelSpec,
  inputSeqLen: number,
  outputSeqLen: number,
  batchSize: number,
  gpu: GPUSpec,
  precision: InferencePrecision = 'bf16',
  flashAttention: boolean = true,
): {
  phase: 'prefill' | 'decode';
  startMs: number;
  endMs: number;
  tokensProcessed: number;
  bottleneck: 'compute' | 'memory';
}[] {
  const ttft = estimateTTFT(model, inputSeqLen, gpu, precision, batchSize, flashAttention);
  const avgTpot = estimateAverageTPOT(model, inputSeqLen, outputSeqLen, batchSize, gpu, precision);

  const prefillEnd = ttft;
  const decodeEnd = prefillEnd + avgTpot * outputSeqLen;

  return [
    {
      phase: 'prefill',
      startMs: 0,
      endMs: prefillEnd,
      tokensProcessed: inputSeqLen * batchSize,
      bottleneck: 'compute',
    },
    {
      phase: 'decode',
      startMs: prefillEnd,
      endMs: decodeEnd,
      tokensProcessed: outputSeqLen * batchSize,
      bottleneck: 'memory',
    },
  ];
}

/** Compute exposed AllReduce overhead after layer-pipelining overlap.
 * Each AllReduce hides behind the next sub-layer's compute+memory work
 * (different HW resources: NVLink/PCIe vs HBM). Last collective is always
 * fully exposed (nothing to overlap with). */
function computeExposedAllReduce(
  arTimePerCollective: number,
  numCollectives: number,
  baseExecTime: number
): number {
  if (numCollectives <= 0 || arTimePerCollective <= 0) return 0;
  const gapBetweenCollectives = baseExecTime / numCollectives;
  const exposedPerCollective = Math.max(0, arTimePerCollective - gapBetweenCollectives);
  return (numCollectives - 1) * exposedPerCollective + arTimePerCollective;
}

/**
 * Calculate latency with tensor parallelism and optional expert parallelism.
 * TP reduces per-GPU memory bandwidth requirements.
 * EP distributes expert compute and reduces expert weight reads per device.
 */
export function calculateLatencyWithTP(
  model: ModelSpec,
  inputSeqLen: number,
  outputSeqLen: number,
  batchSize: number,
  gpu: GPUSpec,
  tpDegree: number,
  precision: InferencePrecision = 'bf16',
  epDegree: number = 1,
  kvCachePrecision?: InferencePrecision,
  flashAttention: boolean = true,
): LatencyMetrics {
  const bytes = getPrecisionBytes(precision);
  const commBytes = Math.max(2, bytes); // activations always communicated in ≥bf16
  const commBandwidthGBps = gpu.nvlinkBandwidthGBps || gpu.pcieBandwidthGBps;
  const commBandwidthBps = commBandwidthGBps * 1e9;
  const ep = Math.max(1, epDegree);

  // Tree-round alpha: per-collective latency scales with TP degree
  const isNvlink = gpu.nvlinkBandwidthGBps > 0;
  const numRounds = tpDegree > 1 ? 2 * Math.ceil(Math.log2(tpDegree)) : 0;
  const alphaPerCollective = isNvlink
    ? _nvlinkPerRoundMs * numRounds
    : _pciePerRoundMs * numRounds;

  // 2 AllReduces per dense layer (attention + MLP output), 1 per MoE layer with EP
  const numMoELayers = model.numMoELayers ?? 0;
  const numDenseLayers = model.numLayers - numMoELayers;
  const arPerMoELayer = ep > 1 ? 1 : 2;
  const numTPCollectives = 2 * numDenseLayers + arPerMoELayer * numMoELayers;

  // ── Prefill ──
  // Always pass flashAttention=true to estimateTTFT; this function adds TP-aware penalty below
  const baseTtft = estimateTTFT(model, inputSeqLen, gpu, precision, batchSize, true);
  const bandwidthBytesPerSec = gpu.memoryBandwidthTBps * 1e12;

  // TP AllReduce per-collective bandwidth (ring factor stays as-is)
  const prefillHiddenBytes = batchSize * inputSeqLen * model.hiddenSize * commBytes;
  const prefillBwPerCollective = tpDegree > 1
    ? (prefillHiddenBytes / (commBandwidthBps * _tpCommEfficiency)) * 1000 * 2 * ((tpDegree - 1) / tpDegree)
    : 0;
  const prefillArPerCollective = alphaPerCollective + prefillBwPerCollective;

  // Without FA, attention materializes full seq×seq matrix in HBM per layer.
  // TP-aware: each GPU materializes scores for numHeads/tp heads only.
  let faHBMPenaltyMs = 0;
  if (!flashAttention) {
    const headsPerGPU = Math.ceil(model.numAttentionHeads / tpDegree);
    const attentionHBMBytes = model.numLayers * headsPerGPU
      * inputSeqLen * inputSeqLen
      * getPrecisionBytes(precision) * batchSize * 2.5;
    const bwEfficiency = getBandwidthEfficiency(attentionHBMBytes);
    faHBMPenaltyMs = (attentionHBMBytes / (bandwidthBytesPerSec * bwEfficiency)) * 1000;
  }

  let ttft: number;
  if (model.isMoE && model.numExperts && ep > 1) {
    // Split FLOPs: shared part scales with 1/TP, expert part scales with 1/(TP×EP)
    const numActive = model.numActiveExperts ?? 2;
    const expertIntermediate = model.expertIntermediateSize ?? model.intermediateSize;
    // Active expert FLOPs per MoE layer (routed + shared experts)
    const mlpMatrices = model.gatedMLP ? 3 : 2;
    const sharedIntermediate = model.sharedExpertIntermediateSize ?? expertIntermediate;
    const activeExpertFlopsPerLayer = numActive * mlpMatrices * model.hiddenSize * expertIntermediate * 2
      + (model.numSharedExperts ?? 0) * mlpMatrices * model.hiddenSize * sharedIntermediate * 2;
    const numMoELayersLocal = model.numMoELayers ?? model.numLayers;
    const totalFlopsPerToken = model.flopsPerToken;
    const expertFlopsFraction = Math.min(0.95,
      (activeExpertFlopsPerLayer * numMoELayersLocal) / totalFlopsPerToken);
    const sharedFlopsFraction = 1 - expertFlopsFraction;
    // Effective parallelism: shared compute divided by TP, expert compute divided by TP×EP
    const effectiveParallelism = 1 / (sharedFlopsFraction / tpDegree + expertFlopsFraction / (tpDegree * ep));

    // EP All-to-All for prefill (heavily overlapped with compute — 85% hidden)
    const allToAllPerLayer = 2 * batchSize * inputSeqLen * model.hiddenSize * numActive * commBytes * (ep - 1) / ep;
    const epAllToAllPrefillBytes = allToAllPerLayer * numMoELayersLocal;
    const epAllToAllPrefillMs = (epAllToAllPrefillBytes / (commBandwidthBps * _tpCommEfficiency)) * 1000 * _epPrefillOverlap;

    const prefillBaseTime = baseTtft / effectiveParallelism;
    const allReduceLatencyMs = tpDegree > 1
      ? computeExposedAllReduce(prefillArPerCollective, numTPCollectives, prefillBaseTime)
      : 0;
    ttft = prefillBaseTime + allReduceLatencyMs + epAllToAllPrefillMs + faHBMPenaltyMs;
  } else {
    const prefillBaseTime = baseTtft / tpDegree;
    const allReduceLatencyMs = tpDegree > 1
      ? computeExposedAllReduce(prefillArPerCollective, numTPCollectives, prefillBaseTime)
      : 0;
    ttft = prefillBaseTime + allReduceLatencyMs + faHBMPenaltyMs;
  }

  // ── Decode ──
  const avgSeqLen = inputSeqLen + outputSeqLen / 2;
  const baseTpot = estimateTPOT(model, avgSeqLen, batchSize, gpu, precision, kvCachePrecision);

  // Per-collective decode AllReduce: alpha + bandwidth
  const decodeBwPerCollective = tpDegree > 1
    ? (batchSize * model.hiddenSize * commBytes / (commBandwidthBps * _tpCommEfficiency))
      * 1000 * 2 * ((tpDegree - 1) / tpDegree)
    : 0;
  const decodeArPerCollective = alphaPerCollective + decodeBwPerCollective;

  // EP All-to-All overhead for decode (per step)
  let epAllToAllDecodeMs = 0;
  if (model.isMoE && model.numExperts && ep > 1) {
    const numActive = model.numActiveExperts ?? 2;
    const numMoELayersLocal = model.numMoELayers ?? model.numLayers;
    // Per MoE layer: dispatch + combine = 2 × batchSize × hiddenSize × numActive × commBytes × (ep-1)/ep
    const allToAllPerLayer = 2 * batchSize * model.hiddenSize * numActive * commBytes * (ep - 1) / ep;
    const allToAllBytes = allToAllPerLayer * numMoELayersLocal;
    epAllToAllDecodeMs = (allToAllBytes / (commBandwidthBps * _tpCommEfficiency)) * 1000;
  }

  let tpot: number;
  if (model.attentionType === 'mla' && model.kvLoraRank && model.qkRopeHeadDim) {
    // MLA: weights scale with TP (and EP for experts), KV cache is replicated (not split)
    const totalWeightBytes = moeWeightBytesPerStep(model, batchSize, precision);
    let epAdjustedWeights: number;
    if (ep > 1) {
      // EP further distributes routed expert weights across EP ranks
      const { sharedParams, routedExpertParams } = moeParamSplit(model);
      const numActive = model.numActiveExperts ?? 2;
      const numExperts = model.numExperts!;
      const fractionTouched = 1 - Math.pow(1 - numActive / numExperts, batchSize);
      epAdjustedWeights = (sharedParams + fractionTouched * routedExpertParams / ep) * bytes / tpDegree;
    } else {
      epAdjustedWeights = totalWeightBytes / tpDegree;
    }
    const kvCachePerRank = model.numLayers * (model.kvLoraRank + model.qkRopeHeadDim) *
      getPrecisionBytes(kvCachePrecision ?? precision) * avgSeqLen * batchSize;
    const bytesPerStep = epAdjustedWeights + kvCachePerRank;
    const bandwidthBytesPerSec = gpu.memoryBandwidthTBps * 1e12;
    const memoryTpot = (bytesPerStep / (bandwidthBytesPerSec * getBandwidthEfficiency(epAdjustedWeights + kvCachePerRank))) * 1000;

    // Compute floor for MLA decode — same as non-MLA path
    const decodeFlopsPerStep = decodeFLOPs(model, avgSeqLen) * batchSize;
    let effectiveComputeParallelism = tpDegree;
    if (model.isMoE && model.numExperts && ep > 1) {
      // Split: shared compute scales with 1/TP, expert compute scales with 1/(TP×EP)
      const numActive = model.numActiveExperts ?? 2;
      const expertIntermediate = model.expertIntermediateSize ?? model.intermediateSize;
      const mlpMatrices = model.gatedMLP ? 3 : 2;
      const sharedIntermediate = model.sharedExpertIntermediateSize ?? expertIntermediate;
      const activeExpertFlopsPerLayer = numActive * mlpMatrices * model.hiddenSize * expertIntermediate * 2
        + (model.numSharedExperts ?? 0) * mlpMatrices * model.hiddenSize * sharedIntermediate * 2;
      const numMoELayersLocal = model.numMoELayers ?? model.numLayers;
      const expertFlopsFraction = Math.min(0.95,
        (activeExpertFlopsPerLayer * numMoELayersLocal) / model.flopsPerToken);
      const sharedFlopsFraction = 1 - expertFlopsFraction;
      effectiveComputeParallelism = 1 / (sharedFlopsFraction / tpDegree + expertFlopsFraction / (tpDegree * ep));
    }
    const mlaSaturation = getMatmulSaturationFactor(batchSize, model.hiddenSize / tpDegree, gpu);
    const computeTpot = (decodeFlopsPerStep / (effectiveComputeParallelism * getGPUTFLOPS(gpu, precision) * 1e12 * mlaSaturation * getPrefillEfficiency(gpu, precision))) * 1000;

    const decodeBaseTime = Math.max(memoryTpot, computeTpot);
    const perStepAllReduce = tpDegree > 1
      ? computeExposedAllReduce(decodeArPerCollective, numTPCollectives, decodeBaseTime)
      : 0;
    tpot = decodeBaseTime + perStepAllReduce + epAllToAllDecodeMs;
  } else {
    const decodeBaseTime = baseTpot / tpDegree;
    const perStepAllReduce = tpDegree > 1
      ? computeExposedAllReduce(decodeArPerCollective, numTPCollectives, decodeBaseTime)
      : 0;
    tpot = decodeBaseTime + perStepAllReduce + epAllToAllDecodeMs;
  }

  const prefillTime = ttft;
  const decodeTime = tpot * outputSeqLen;
  const totalLatency = prefillTime + decodeTime;

  return {
    ttft,
    tpot,
    totalLatency,
    prefillTime,
    decodeTime,
  };
}

/**
 * Calculate all metrics from inference config
 */
export function calculateMetricsFromConfig(config: InferenceConfig): {
  latency: LatencyMetrics;
  throughput: ThroughputMetrics;
  utilization: UtilizationMetrics;
} {
  const { modelSpec, gpu, batchSize, inputSeqLen, outputSeqLen, weightPrecision } = config;
  const kvPrec = config.kvCachePrecision ?? weightPrecision;
  const ep = config.expertParallel ?? 1;

  if ((config.tensorParallel && config.tensorParallel > 1) || ep > 1) {
    const tp = config.tensorParallel ?? 1;
    const latency = calculateLatencyWithTP(
      modelSpec, inputSeqLen, outputSeqLen, batchSize, gpu, tp, weightPrecision, ep, kvPrec,
      config.flashAttention,
    );

    // Recalculate throughput and utilization with TP
    const tokensPerSecond = (outputSeqLen * batchSize) / (latency.totalLatency / 1000);
    const throughput: ThroughputMetrics = {
      tokensPerSecond,
      requestsPerSecond: batchSize / (latency.totalLatency / 1000),
      prefillTokensPerSecond: (inputSeqLen * batchSize) / (latency.prefillTime / 1000),
      decodeTokensPerSecond: (outputSeqLen * batchSize) / (latency.decodeTime / 1000),
    };

    // Use TP-aware memory for capacity utilization
    const memory = calculateMemoryFromConfig(config);
    const utilization = calculateUtilizationMetrics(
      modelSpec, inputSeqLen, outputSeqLen, batchSize, gpu, weightPrecision, memory.total, kvPrec
    );

    return { latency, throughput, utilization };
  }

  const memory = calculateMemoryFromConfig(config);
  return {
    latency: calculateLatencyMetrics(modelSpec, inputSeqLen, outputSeqLen, batchSize, gpu, weightPrecision, kvPrec, config.flashAttention),
    throughput: calculateThroughputMetrics(modelSpec, inputSeqLen, outputSeqLen, batchSize, gpu, weightPrecision, kvPrec, config.flashAttention),
    utilization: calculateUtilizationMetrics(modelSpec, inputSeqLen, outputSeqLen, batchSize, gpu, weightPrecision, memory.total, kvPrec),
  };
}

/**
 * CB scheduling overhead as a fraction of TPOT.
 * 1% base + scales to 2% at batch=128+.
 */
export function cbSchedulingOverhead(batchSize: number): number {
  return _cbSchedulingBase + _cbSchedulingBase * Math.min(1, batchSize / 128);
}

/**
 * Calculate continuous batching metrics.
 *
 * Steady-state slot-cycling model: all B slots occupied, each cycling through
 * prefill (single-request) then decode. The throughput formula is structurally
 * identical to static batching (B × outSeqLen × 1000 / (TTFT + TPOT × outSeqLen)).
 * The CB difference is which TTFT and TPOT go in: batch=1 TTFT (with interference)
 * instead of batch=B TTFT, and TPOT with scheduling overhead.
 */
export function calculateContinuousBatchingMetrics(
  config: InferenceConfig,
  baseMetrics: { latency: LatencyMetrics; throughput: ThroughputMetrics; utilization: UtilizationMetrics }
): { latency: LatencyMetrics; throughput: ThroughputMetrics; utilization: UtilizationMetrics } {
  const { modelSpec, gpu, inputSeqLen, outputSeqLen, weightPrecision } = config;
  const B = config.batchSize;
  const tp = config.tensorParallel ?? 1;
  const ep = config.expertParallel ?? 1;

  const schedulingOverhead = cbSchedulingOverhead(B);

  // Prefill interference from concurrent decode slots: 0 at B=1, up to 10% at B≥32
  const prefillInterference = _cbPrefillInterferenceMax * Math.min(1, (B - 1) / 32);

  // Effective TPOT with scheduling overhead
  const baseTpot = baseMetrics.latency.tpot;
  const effectiveTpot = baseTpot * (1 + schedulingOverhead);

  // Batch=1 TTFT (true single-request prefill time)
  let batch1TTFT: number;
  if (tp > 1 || ep > 1) {
    batch1TTFT = calculateLatencyWithTP(modelSpec, inputSeqLen, outputSeqLen, 1, gpu, tp, weightPrecision, ep, undefined, config.flashAttention).ttft;
  } else {
    batch1TTFT = estimateTTFT(modelSpec, inputSeqLen, gpu, weightPrecision, 1, config.flashAttention);
  }

  // CB TTFT with interference from concurrent decode slots
  const cbTTFT = batch1TTFT * (1.0 + prefillInterference);

  // Slot cycle time and throughput
  const decodeTime = effectiveTpot * outputSeqLen;
  const cycleTime = cbTTFT + decodeTime;
  if (cycleTime <= 0) throw new Error('Continuous batching cycle time must be positive');
  const cbThroughput = B * outputSeqLen * 1000 / cycleTime;

  // Per-request latency
  const cbTotalLatency = cbTTFT + decodeTime;

  const latency: LatencyMetrics = {
    ttft: cbTTFT,
    tpot: effectiveTpot,
    totalLatency: cbTotalLatency,
    prefillTime: cbTTFT,
    decodeTime,
  };

  const throughput: ThroughputMetrics = {
    tokensPerSecond: cbThroughput,
    requestsPerSecond: B * 1000 / cycleTime,
    prefillTokensPerSecond: inputSeqLen * B * 1000 / cycleTime,
    decodeTokensPerSecond: cbThroughput,
  };

  // Utilization unchanged (same memory, same roofline)
  return { latency, throughput, utilization: baseMetrics.utilization };
}

/**
 * Reset all perturbable inference latency parameters to their defaults.
 */
export function resetInferenceLatencyParams(): void {
  _prefillResidual = 0.40;
  _decodeSamplingOverheadMs = 0.03;
  _nvlinkPerRoundMs = 0.005;
  _pciePerRoundMs = 0.015;
  _tpCommEfficiency = 0.80;
  _epPrefillOverlap = 0.15;
  _bwEffFloor = 0.35;
  _bwEffScale = 5.0;
  _cbSchedulingBase = 0.01;
  _cbPrefillInterferenceMax = 0.10;
}

