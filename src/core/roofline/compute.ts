/**
 * Roofline Model Computation
 *
 * Computes arithmetic intensity (FLOPs/byte) and attained TFLOPS for
 * per-operation and aggregate workload points on a roofline chart.
 *
 * References:
 * - Williams, Waterman & Patterson, "Roofline: An Insightful Visual
 *   Performance Model for Multicore Architectures", 2009
 * - Dao et al., "FlashAttention: Fast and Memory-Efficient Exact
 *   Attention with IO-Awareness", 2022
 */
// See docs/PHYSICS.md#memory-model for activation checkpointing formulas.

import type { ModelSpec } from '../../types/model.ts';
import type { GPUSpec } from '../../types/hardware.ts';
import type { DType } from '../../types/base.ts';
import { DTYPE_BYTES } from '../../types/base.ts';
import { getEffectiveTFLOPS } from '../hardware/gpu.ts';
import type { SimulationMetrics } from '../simulation/engine.ts';
import type { InferenceSimulationResult, InferencePrecision } from '../../types/inference.ts';
import { prefillFLOPs, decodeFLOPs, moeWeightBytesPerStep } from '../inference/latency.ts';
import { calculateMoEMemory } from '../models/moe.ts';
import { totalKVCacheMemory } from '../inference/kv-cache.ts';
import { NF4_BYTES_PER_PARAM, computeLoraTrainableParams } from '../strategies/lora.ts';
import type { LoraTargetModules } from '../strategies/lora.ts';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface RooflinePoint {
  label: string;
  arithmeticIntensity: number;  // FLOPs/byte (X-axis)
  attainedTFLOPS: number;       // TFLOPS (Y-axis)
  flopsFraction: number;        // Fraction of total FLOPs (0..1), controls dot size
  category: 'forward-matmul' | 'forward-other' | 'optimizer' | 'aggregate' | 'prefill' | 'decode';
  isAggregate: boolean;
}

export interface RooflineCeiling {
  peakComputeTFLOPS: number;   // Horizontal ceiling
  peakBandwidthTBps: number;   // Slope of diagonal ceiling
  ridgePoint: number;          // FLOPs/byte where ceilings meet
}

export interface RooflineData {
  ceiling: RooflineCeiling;
  points: RooflinePoint[];
}

// ---------------------------------------------------------------------------
// Ceiling
// ---------------------------------------------------------------------------

export function computeCeiling(gpu: GPUSpec, dtype: DType | InferencePrecision): RooflineCeiling {
  const peakTFLOPS = getEffectiveTFLOPS(gpu, dtype);
  const peakBW = gpu.memoryBandwidthTBps;
  return {
    peakComputeTFLOPS: peakTFLOPS,
    peakBandwidthTBps: peakBW,
    ridgePoint: peakBW > 0 ? peakTFLOPS / peakBW : 0,
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Attained TFLOPS on the roofline ceiling at a given arithmetic intensity. */
function ceilingTFLOPS(intensity: number, ceiling: RooflineCeiling): number {
  return Math.min(ceiling.peakComputeTFLOPS, intensity * ceiling.peakBandwidthTBps);
}

/**
 * Arithmetic intensity for a GEMM: (M, K) × (K, N).
 * AI = 2·M·K·N / ((M·K + K·N + M·N) · bytesPerElem)
 */
export function gemmIntensity(M: number, K: number, N: number, bytesPerElem: number): number {
  const flops = 2 * M * K * N;
  const bytes = (M * K + K * N + M * N) * bytesPerElem;
  return bytes > 0 ? flops / bytes : 0;
}

// ---------------------------------------------------------------------------
// Training Roofline
// ---------------------------------------------------------------------------

export interface TrainingRooflineConfig {
  tp: number;
  pp: number;
  dp: number;
  ep?: number;
  seqLength: number;
  microBatchSize: number;
  globalBatchSize: number;
  activationCheckpointing: boolean;
  checkpointingGranularity?: 'full' | 'selective';
  flashAttention: boolean;
  dpType?: 'ddp' | 'fsdp' | 'zero-1' | 'zero-3';
  sequenceParallel?: boolean;
  finetuningMethod?: 'full' | 'lora' | 'qlora';
  loraRank?: number;
  loraTargetModules?: LoraTargetModules;
}

export function computeTrainingRoofline(
  model: ModelSpec,
  gpu: GPUSpec,
  dtype: DType,
  metrics: SimulationMetrics,
  config: TrainingRooflineConfig,
): RooflineData {
  const ceiling = computeCeiling(gpu, dtype);
  const bytesPerElem = DTYPE_BYTES[dtype as DType] ?? 2;
  const { tp, pp, dp, seqLength: S, microBatchSize: mbs } = config;
  const tokens = S * mbs; // tokens per micro-batch per GPU

  // ── Per-operation points ──

  const points: RooflinePoint[] = [];
  const H = model.hiddenSize;
  const numHeads = model.numAttentionHeads;
  const numKvHeads = model.numKvHeads;
  const headDim = model.headDim;
  const numLayers = model.numLayers;

  // Accumulate total forward FLOPs for fraction calculation
  let totalForwardFlops = 0;
  const opFlops: { label: string; flops: number; intensity: number; category: RooflinePoint['category'] }[] = [];

  // --- a) QKV Projection ---
  {
    const M = tokens;
    const K = H;
    // Q output columns (sharded by TP)
    const N_q = (numHeads * headDim) / tp;
    // K,V output columns (sharded by TP), 2 projections
    const N_kv = (numKvHeads * headDim) / tp;

    // Handle MLA: different projection dimensions
    let flops: number;
    let intensity: number;
    if (model.attentionType === 'mla' && model.qLoraRank && model.kvLoraRank) {
      // MLA: Q path = down_proj(H→qLoraRank) + up_proj(qLoraRank→numHeads*headDim)
      // KV path = down_proj(H→kvLoraRank)
      const qLoraRank = model.qLoraRank;
      const kvLoraRank = model.kvLoraRank;
      const qDown = 2 * M * H * qLoraRank / tp;
      const qUp = 2 * M * qLoraRank * (numHeads * headDim) / tp;
      const kvDown = 2 * M * H * kvLoraRank / tp;
      flops = (qDown + qUp + kvDown) * numLayers;
      // Intensity: use the dominant matmul (qUp is largest)
      intensity = gemmIntensity(M, qLoraRank, (numHeads * headDim) / tp, bytesPerElem);
    } else {
      // Standard QKV
      const qkvFlops = 2 * M * K * N_q + 2 * 2 * M * K * N_kv;
      flops = qkvFlops * numLayers;
      intensity = gemmIntensity(M, K, N_q + 2 * N_kv, bytesPerElem);
    }
    totalForwardFlops += flops;
    opFlops.push({ label: 'QKV Projection', flops, intensity, category: 'forward-matmul' });
  }

  // --- b) Attention Scores (Q @ K^T) ---
  {
    const numHeadsPerGPU = numHeads / tp;
    const M_attn = S;
    const K_attn = headDim;
    const N_attn = S;
    const flops = numHeadsPerGPU * 2 * M_attn * K_attn * N_attn * mbs * numLayers;

    // Without Flash Attention: full S×S matrix materialized
    let attnBytes = numHeadsPerGPU * (M_attn * K_attn + K_attn * N_attn + M_attn * N_attn) * bytesPerElem * mbs * numLayers;

    // Flash Attention reduces HBM traffic
    if (config.flashAttention) {
      const faFactor = Math.min(2 * headDim * headDim / S, 8);
      if (faFactor > 1) {
        attnBytes = attnBytes / faFactor;
      }
    }

    const intensity = attnBytes > 0 ? flops / attnBytes : 0;
    totalForwardFlops += flops;
    opFlops.push({ label: 'Attention Scores', flops, intensity, category: 'forward-matmul' });
  }

  // --- c) Attention Output (scores @ V + output projection) ---
  {
    const numHeadsPerGPU = numHeads / tp;
    // scores @ V: batched (S, S) × (S, headDim)
    const svFlops = numHeadsPerGPU * 2 * S * S * headDim * mbs * numLayers;
    // output projection: (tokens, numHeads*headDim/tp) × (numHeads*headDim/tp, H)
    const projN = (numHeads * headDim) / tp;
    const outProjFlops = 2 * tokens * projN * H * numLayers;
    const flops = svFlops + outProjFlops;

    // Intensity dominated by output projection (has weight reuse)
    const intensity = gemmIntensity(tokens, projN, H, bytesPerElem);
    totalForwardFlops += flops;
    opFlops.push({ label: 'Attention Output', flops, intensity, category: 'forward-matmul' });
  }

  // --- d) MLP ---
  {
    let mlpFlops = 0;
    let mlpIntensity = 0;
    let label = 'MLP';

    // Check if model has MoE layers
    if (model.isMoE && model.numExperts && model.numActiveExperts) {
      const expertIntermediate = model.expertIntermediateSize ?? model.intermediateSize;
      const numMoELayers = model.numMoELayers ?? numLayers;
      const numDenseLayers = numLayers - numMoELayers;
      const mlpMatrices = model.gatedMLP ? 3 : 2;
      const numActive = model.numActiveExperts;
      const numShared = model.numSharedExperts ?? 0;

      // Dense MLP layers
      if (numDenseLayers > 0) {
        const N_dense = model.intermediateSize / tp;
        const denseMlpFlops = mlpMatrices * 2 * tokens * H * N_dense * numDenseLayers;
        mlpFlops += denseMlpFlops;
        mlpIntensity = gemmIntensity(tokens, H, N_dense, bytesPerElem);
      }

      // Expert MLP layers (active experts only — that's what runs per token)
      const N_expert = expertIntermediate / tp;
      const sharedIntermediate = model.sharedExpertIntermediateSize ?? expertIntermediate;
      const N_shared = sharedIntermediate / tp;
      const expertMlpFlops = (numActive * mlpMatrices * 2 * tokens * H * N_expert
        + numShared * mlpMatrices * 2 * tokens * H * N_shared) * numMoELayers;
      mlpFlops += expertMlpFlops;

      // Expert matmuls have smaller dimensions → potentially lower intensity
      const expertIntensity = gemmIntensity(tokens, H, N_expert, bytesPerElem);

      // Use the MoE expert intensity for the point (it's the more interesting one)
      if (numDenseLayers > 0) {
        // Add dense MLP as separate point
        const denseN = model.intermediateSize / tp;
        const denseFlops = mlpMatrices * 2 * tokens * H * denseN * numDenseLayers;
        opFlops.push({
          label: 'Dense MLP',
          flops: denseFlops,
          intensity: gemmIntensity(tokens, H, denseN, bytesPerElem),
          category: 'forward-matmul',
        });
        opFlops.push({
          label: 'Expert MLP',
          flops: expertMlpFlops,
          intensity: expertIntensity,
          category: 'forward-matmul',
        });
      } else {
        mlpIntensity = expertIntensity;
        opFlops.push({ label: 'Expert MLP', flops: mlpFlops, intensity: mlpIntensity, category: 'forward-matmul' });
      }
      totalForwardFlops += mlpFlops;
    } else {
      const N_mlp = model.intermediateSize / tp;
      const mlpMatrices = model.gatedMLP ? 3 : 2;
      mlpFlops = mlpMatrices * 2 * tokens * H * N_mlp * numLayers;
      mlpIntensity = gemmIntensity(tokens, H, N_mlp, bytesPerElem);
      label = model.gatedMLP ? 'MLP (SwiGLU)' : 'MLP';
      totalForwardFlops += mlpFlops;
      opFlops.push({ label, flops: mlpFlops, intensity: mlpIntensity, category: 'forward-matmul' });
    }
  }

  // --- e) Norms (RMSNorm/LayerNorm) ---
  {
    // ~4 ops/element, read+write
    const normsPerLayer = 2; // pre-attn + post-attn
    const flops = tokens * H * 4 * numLayers * normsPerLayer;
    const bytes = tokens * H * 2 * bytesPerElem * numLayers * normsPerLayer; // read + write
    const intensity = bytes > 0 ? flops / bytes : 0;
    totalForwardFlops += flops;
    opFlops.push({ label: model.normType === 'rmsnorm' ? 'RMSNorm' : 'LayerNorm', flops, intensity, category: 'forward-other' });
  }

  // --- f) Softmax (only shown when Flash Attention is OFF) ---
  if (!config.flashAttention) {
    const numHeadsPerGPU = numHeads / tp;
    const flops = 5 * numHeadsPerGPU * S * S * mbs * numLayers;
    const bytes = numHeadsPerGPU * S * S * 2 * bytesPerElem * mbs * numLayers;
    const intensity = bytes > 0 ? flops / bytes : 0;
    totalForwardFlops += flops;
    opFlops.push({ label: 'Softmax', flops, intensity, category: 'forward-other' });
  }

  // --- g) Optimizer (AdamW) ---
  {
    const ep = config.ep ?? 1;
    const dpShardFactor = (config.dpType === 'fsdp' || config.dpType === 'zero-3') ? dp : 1;
    const isMoE = model.isMoE && model.numExperts && ep >= 1;
    const moeInfo = calculateMoEMemory(model, dtype, ep);
    const sharedP = isMoE ? moeInfo.sharedParams + moeInfo.routerParams : model.activeParams;
    const expertP = isMoE ? moeInfo.expertParams : 0;

    const expertDPReplicas = ep > 1 ? Math.max(1, Math.floor(dp / ep)) : dp;
    const sharedOptSharding = tp * pp * dpShardFactor;
    const expertOptSharding = tp * pp * ep * (dpShardFactor > 1 ? expertDPReplicas : 1);
    const paramsPerGPU = sharedP / sharedOptSharding + expertP / expertOptSharding;

    const flops = 10 * paramsPerGPU;
    const bytes = 30 * paramsPerGPU;
    const intensity = bytes > 0 ? flops / bytes : 0;
    opFlops.push({ label: 'Optimizer (AdamW)', flops, intensity, category: 'optimizer' });
  }

  // Compute FLOPs fractions and build points
  // Forward * 3 (fwd + bwd) gives full training FLOPs; optimizer is separate
  const totalTrainingFlops = totalForwardFlops * 3; // forward + backward (2x) ≈ 3x forward
  const optimizerEntry = opFlops.find(o => o.category === 'optimizer');
  const fullTotalFlops = totalTrainingFlops + (optimizerEntry?.flops ?? 0);

  for (const op of opFlops) {
    const fraction = fullTotalFlops > 0 ? (
      op.category === 'optimizer'
        ? op.flops / fullTotalFlops
        : (op.flops * 3) / fullTotalFlops  // forward ops count 3x (fwd + bwd)
    ) : 0;

    points.push({
      label: op.label,
      arithmeticIntensity: op.intensity,
      attainedTFLOPS: ceilingTFLOPS(op.intensity, ceiling),
      flopsFraction: fraction,
      category: op.category,
      isAggregate: false,
    });
  }

  // --- Aggregate "Overall Workload" point ---
  {
    const ep = config.ep ?? 1;
    const totalGPUs = tp * pp * dp;
    const ga = Math.ceil(config.globalBatchSize / (mbs * dp));

    // Total FLOPs per GPU per step (useful work: 6PD full, 4PD LoRA/QLoRA)
    const flopMultiplier = config.finetuningMethod && config.finetuningMethod !== 'full' ? 4 : 6;
    const totalFlopsPerGPU = flopMultiplier * model.activeParams * S * config.globalBatchSize / totalGPUs;

    // Split params into shared (attention/norms/embed) and expert (routed MoE)
    const isMoE = model.isMoE && model.numExperts && ep >= 1;
    const moeInfo = calculateMoEMemory(model, dtype, ep);
    const sharedParams = isMoE ? moeInfo.sharedParams + moeInfo.routerParams : model.activeParams;
    const expertParams = isMoE ? moeInfo.expertParams : 0;

    // HBM weight traffic per microbatch: shared layers all-gathered from FSDP peers (TP×PP),
    // expert layers only local subset (TP×PP×EP)
    const isFSDP = config.dpType === 'fsdp' || config.dpType === 'zero-3';
    const sharedComputeParams = sharedParams / (tp * pp);
    const expertComputeParams = expertParams / (tp * pp * ep);
    const computeParams = sharedComputeParams + expertComputeParams;

    // Per-microbatch HBM passes over weights:
    //   DDP:  fwd 1R + bwd 1R (no ckpt) | fwd 1R + bwd 2R (ckpt)  → 2 | 3
    //   FSDP: fwd 1W+1R + bwd 1W+1R     | fwd 1W+1R + bwd 1W+2R   → 4 | 5
    // Selective ckpt: only QKV weights re-read (~10% of total), rounds to same as no-ckpt
    const fullCkpt = config.activationCheckpointing && config.checkpointingGranularity !== 'selective';
    const weightMultiplier = isFSDP
      ? (fullCkpt ? 5 : 4)
      : (fullCkpt ? 3 : 2);
    // QLoRA base weights stored at NF4 (0.515 bytes/param), not dtype
    const effectiveBytesPerElem = config.finetuningMethod === 'qlora'
      ? NF4_BYTES_PER_PARAM
      : bytesPerElem;
    const weightBytes = computeParams * effectiveBytesPerElem * weightMultiplier * ga;

    // LoRA: only adapter params have gradients and optimizer states
    const isLoRA = config.finetuningMethod === 'lora' || config.finetuningMethod === 'qlora';
    const trainableParams = isLoRA
      ? computeLoraTrainableParams(model, config.loraRank ?? 16, config.loraTargetModules ?? 'q_k_v_o')
      : null;

    // Gradients: production write + collective read (AllReduce or ReduceScatter) = 2 passes
    // LoRA: only adapter grads (BF16), full: all param grads (fp32)
    const gradParamsPerGPU = isLoRA ? (trainableParams! / (tp * pp)) : computeParams;
    const gradBytesPerParam = isLoRA ? 2 : 4; // adapter grads BF16, full grads fp32
    const gradBytes = gradParamsPerGPU * gradBytesPerParam * 2;

    // Optimizer: operates on FSDP-sharded slice
    // LoRA: only adapter params have optimizer states
    const dpShardFactor = (config.dpType === 'fsdp' || config.dpType === 'zero-3') ? dp : 1;
    const expertDPReplicas = ep > 1 ? Math.max(1, Math.floor(dp / ep)) : dp;
    const sharedOptSharding = tp * pp * dpShardFactor;
    const expertOptSharding = tp * pp * ep * (dpShardFactor > 1 ? expertDPReplicas : 1);
    const optParamsPerGPU = isLoRA
      ? (trainableParams! / (tp * pp * dpShardFactor))
      : (sharedParams / sharedOptSharding + expertParams / expertOptSharding);
    const optBytes = optParamsPerGPU * 30;

    const layersPerGPU = numLayers / pp;
    const actTrafficPerLayerPerMB = 2 * tokens * H * bytesPerElem;
    const actBytes = actTrafficPerLayerPerMB * layersPerGPU * ga;
    const ppBytes = pp > 1 ? 2 * tokens * H * bytesPerElem * ga : 0;

    const totalBytes = weightBytes + gradBytes + optBytes + actBytes + ppBytes;
    const aggregateIntensity = totalBytes > 0 ? totalFlopsPerGPU / totalBytes : 0;

    points.push({
      label: 'Overall Workload',
      arithmeticIntensity: aggregateIntensity,
      attainedTFLOPS: metrics.tflopsPerGPU,
      flopsFraction: 1.0,
      category: 'aggregate',
      isAggregate: true,
    });
  }

  return { ceiling, points };
}

// ---------------------------------------------------------------------------
// Inference Roofline
// ---------------------------------------------------------------------------

export interface InferenceRooflineConfig {
  inputSeqLen: number;
  outputSeqLen: number;
  batchSize: number;
  tp: number;
  continuousBatching?: boolean;
}

export function computeInferenceRoofline(
  model: ModelSpec,
  gpu: GPUSpec,
  precision: InferencePrecision,
  result: InferenceSimulationResult,
  config: InferenceRooflineConfig,
): RooflineData {
  const ceiling = computeCeiling(gpu, precision);
  const { inputSeqLen, outputSeqLen, batchSize, tp } = config;
  const points: RooflinePoint[] = [];

  // Compute FLOPs share: total prefill vs total decode FLOPs across full request batch
  const prefillBatch = config.continuousBatching ? 1 : batchSize;
  const totalPrefillFlops = prefillFLOPs(model, inputSeqLen) * batchSize; // all requests
  const totalDecodeFlops = decodeFLOPs(model) * batchSize * outputSeqLen; // all tokens
  const totalFlops = totalPrefillFlops + totalDecodeFlops;
  const prefillFrac = totalFlops > 0 ? totalPrefillFlops / totalFlops : 0.5;
  const decodeFrac = totalFlops > 0 ? totalDecodeFlops / totalFlops : 0.5;

  // --- a) Prefill point (actual measured) ---
  // CB prefills one request at a time (TTFT is batch=1); static batching prefills the full batch.
  {
    const flops = prefillFLOPs(model, inputSeqLen) * prefillBatch;
    const weightsBytes = moeWeightBytesPerStep(model, prefillBatch, precision);
    const intensity = weightsBytes > 0 ? flops / weightsBytes : 0;
    const ttftSeconds = result.latency.ttft / 1000;
    const attainedTFLOPS = ttftSeconds > 0 ? flops / ttftSeconds / 1e12 / tp : 0;

    points.push({
      label: config.continuousBatching ? 'Prefill (per request)' : `Prefill (B=${batchSize})`,
      arithmeticIntensity: intensity,
      attainedTFLOPS,
      flopsFraction: prefillFrac,
      category: 'prefill',
      isAggregate: false,
    });
  }

  // --- b) Decode point (actual measured) ---
  {
    const flops = decodeFLOPs(model) * batchSize;
    const weightsBytes = moeWeightBytesPerStep(model, batchSize, precision);
    const avgSeqLen = inputSeqLen + outputSeqLen / 2;
    const kvCacheBytes = totalKVCacheMemory(model, avgSeqLen, batchSize, precision);
    const totalBytes = weightsBytes + kvCacheBytes;
    const intensity = totalBytes > 0 ? flops / totalBytes : 0;
    const tpotSeconds = result.latency.tpot / 1000;
    const attainedTFLOPS = tpotSeconds > 0 ? flops / tpotSeconds / 1e12 / tp : 0;

    points.push({
      label: `Decode (B=${batchSize})`,
      arithmeticIntensity: intensity,
      attainedTFLOPS,
      flopsFraction: decodeFrac,
      category: 'decode',
      isAggregate: false,
    });
  }

  return { ceiling, points };
}
