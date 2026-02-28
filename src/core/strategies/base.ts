/**
 * Base strategy interface and abstract class for parallelism strategies
 */
// See docs/PHYSICS.md and docs/STRATEGIES.md for formula derivations and calibration anchors.

import type {
  ModelSpec,
  ClusterConfig,
  TrainingHyperparams,
  MemoryBreakdown,
  CommunicationBreakdown,
  TimingBreakdown,
  StrategyValidation,
  StrategyAnalysis,
  DType,
  SimulationEvent,
  GPUSpec,
  OptimizerType,
} from '../../types/index.ts';
import { DTYPE_BYTES } from '../../types/index.ts';
import { getEffectiveTFLOPS, getTrainingTFLOPS, getMemoryBandwidthScaling, getComputeSaturationFactor } from '../hardware/gpu.ts';
import { calculateMoEMemory } from '../models/moe.ts';
import { gemmIntensity } from '../roofline/compute.ts';
import type { LoraConfig } from './lora.ts';
import { computeLoraTrainableParams } from './lora.ts';

/**
 * Configuration passed to strategy methods
 */
export interface StrategyContext {
  model: ModelSpec;
  cluster: ClusterConfig;
  training: TrainingHyperparams;
  seqLength: number;
  microBatchSize: number;
  globalBatchSize: number;
  gradientAccumulationSteps: number;
  activationCheckpointing: boolean;
  checkpointingGranularity?: 'full' | 'selective';
  selectiveStoredLayers?: number;  // resolved value (never 'auto'); undefined = all layers
  flashAttention: boolean;
  lora?: LoraConfig;
}

/**
 * Abstract base class for parallelism strategies
 */
export abstract class ParallelismStrategy {
  abstract readonly name: string;
  abstract readonly shortName: string;
  abstract readonly description: string;

  /** Resolved stored-layers count for selective AC (set during computeMemoryPerGPU). */
  protected _resolvedStoredLayers?: number;

  /**
   * Calculate memory usage per GPU for this strategy
   */
  abstract computeMemoryPerGPU(ctx: StrategyContext): MemoryBreakdown;

  /**
   * Calculate communication volume and breakdown
   */
  abstract computeCommunication(ctx: StrategyContext): CommunicationBreakdown;

  /**
   * Estimate timing for one training step
   */
  abstract computeTiming(ctx: StrategyContext): TimingBreakdown;

  /**
   * Validate that the strategy can be applied to the given configuration
   */
  abstract validate(ctx: StrategyContext): StrategyValidation;

  /**
   * Generate simulation events for visualization
   */
  abstract generateEvents(ctx: StrategyContext): SimulationEvent[];

  /**
   * Compute full analysis including efficiency metrics
   */
  computeAnalysis(ctx: StrategyContext): StrategyAnalysis {
    const memory = this.computeMemoryPerGPU(ctx);
    const communication = this.computeCommunication(ctx);
    const timing = this.computeTiming(ctx);
    const validation = this.validate(ctx);

    // Straggler overhead: order-statistics model
    // BSP sync means slowest node gates the step. Expected maximum of N
    // nodes with step-time CV=σ grows as σ√(2·ln(N)).
    const numNodes = ctx.cluster.numNodes;
    if (numNodes > 1) {
      const overhead = _stragglerSigma * Math.sqrt(2 * Math.log(numNodes));
      timing.scaleOverhead = timing.total * overhead;
      timing.total += timing.scaleOverhead;
    }

    // MFU (Model FLOPS Utilization) - PaLM paper definition
    // Always uses 6 * P * D regardless of activation checkpointing:
    //   Forward: 2*P*D, Backward: 4*P*D → 6*P*D total useful work
    //
    // HFU (Hardware FLOPS Utilization) - includes recompute overhead:
    //   With activation checkpointing: 8*P*D (extra forward pass during backward)
    //   Without: same as MFU (6*P*D)
    //
    // For MoE models, use activeParams (only active experts contribute per token)
    const totalTokens = ctx.seqLength * ctx.globalBatchSize;
    const activeParams = ctx.model.activeParams ?? ctx.model.totalParams;
    const adapterParams = ctx.lora
      ? computeLoraTrainableParams(ctx.model, ctx.lora.rank, ctx.lora.targetModules)
      : 0;

    // Full: 6PD (forward 2PD + act grads 2PD + weight grads 2PD)
    // LoRA: 4PD (frozen forward + act grads) + 6AD (adapter forward + act grads + weight grads)
    const modelFlops = ctx.lora
      ? (4 * activeParams + 6 * adapterParams) * totalTokens
      : 6 * activeParams * totalTokens;
    // HFU includes recompute as a continuous fraction:
    //   No AC:        recomputeFraction = 0     → HFU = MFU
    //   Selective AC:  recomputeFraction = f     → HFU/MFU ≈ 1.04-1.11 (GQA)
    //   Full AC:      recomputeFraction = 1     → HFU/MFU = 1.33
    // LoRA recompute = full forward (frozen model still runs).
    // When selective AC has storedLayers < N, blend selective and full fractions.
    const selectiveFrac = getSelectiveRecomputeFraction(ctx.model);
    const storedFrac = ctx.activationCheckpointing && ctx.checkpointingGranularity === 'selective'
      ? Math.min(this._resolvedStoredLayers ?? ctx.selectiveStoredLayers ?? ctx.model.numLayers, ctx.model.numLayers) / ctx.model.numLayers
      : 1;
    const recomputeFraction = ctx.activationCheckpointing
      ? (ctx.checkpointingGranularity === 'selective'
        ? storedFrac * selectiveFrac + (1 - storedFrac) * 1.0
        : 1.0)
      : 0;
    // HFU: recompute re-runs forward including adapters
    const hardwareFlops = ctx.lora
      ? ((4 + 2 * recomputeFraction) * activeParams
        + (6 + 2 * recomputeFraction) * adapterParams) * totalTokens
      : (6 + 2 * recomputeFraction) * activeParams * totalTokens;

    // MFU reference: always BF16 peak (industry convention).
    // FP8/FP4 accelerate matmul but MFU is reported against the standard
    // training peak. Matches DeepSeek V3, Meta, NVIDIA published numbers.
    // NOTE: Step time uses getTrainingTFLOPS(gpu, computeDtype) in each
    // strategy's computeTiming() — that path is NOT changed here.
    const gpu = ctx.cluster.node.gpu;
    const gpuPeakTFLOPS = getEffectiveTFLOPS(gpu, 'bf16');
    const totalGPUTFLOPS = ctx.cluster.totalGPUs * gpuPeakTFLOPS;

    const stepTimeSeconds = timing.total / 1000;
    const achievedTFLOPS = modelFlops / stepTimeSeconds / 1e12;
    const rawMfu = achievedTFLOPS / totalGPUTFLOPS;
    const hfu = (hardwareFlops / stepTimeSeconds / 1e12) / totalGPUTFLOPS;

    const mfu = rawMfu;

    // Communication reporting — gross, exposed, hidden fraction
    const communicationGrossMs = timing.communication + (timing.epCommunication ?? 0);
    const communicationExposedMs = communicationGrossMs - timing.overlap;
    const overlapHiddenFraction = communicationGrossMs > 0 ? timing.overlap / communicationGrossMs : 0;

    // Communication overhead — exposed (non-overlapped) fraction only
    const communicationOverhead = communicationExposedMs / timing.total;

    // Pipeline bubble (0 for non-pipeline strategies)
    const pipelineBubble = 0;

    // Memory efficiency
    const gpuMemoryGB = ctx.cluster.node.gpu.memoryGB;
    const memoryEfficiency = memory.total / gpuCapacityBytes(gpuMemoryGB);

    // Throughput
    const tokensPerSecond = (ctx.globalBatchSize * ctx.seqLength) / stepTimeSeconds;
    const samplesPerSecond = ctx.globalBatchSize / stepTimeSeconds;
    const tflopsPerGPU = achievedTFLOPS / ctx.cluster.totalGPUs;

    return {
      memory,
      communication,
      timing,
      validation,
      mfu,
      hfu,
      communicationOverhead,
      pipelineBubble,
      memoryEfficiency,
      communicationGrossMs,
      communicationExposedMs,
      overlapHiddenFraction,
      tokensPerSecond,
      samplesPerSecond,
      tflopsPerGPU,
      resolvedStoredLayers: this._resolvedStoredLayers,
    };
  }
}

/**
 * DP bandwidth degradation: fabric congestion, multi-hop routing, NCCL ring/tree inefficiency.
 * At large DP (>64 inter-node participants), AllReduce/AllGather/ReduceScatter performance
 * degrades due to multi-hop routing through IB fat-tree fabric, network congestion from
 * concurrent collectives, and NCCL scheduling overhead growing with ring/tree size.
 *
 * For standalone DP strategies, the input should be numNodes (inter-node participants).
 * NCCL hierarchical collectives run intra-node rings/trees first, then inter-node
 * reduction — fabric congestion scales with inter-node hops, not total GPUs.
 * For 3D strategies, DP degree ≈ numNodes (TP fills the node), so using dp directly
 * gives the correct inter-node participant count.
 *
 * dp₀=64: within-pod DP where topology is well-mapped. Above 64, log degradation.
 * Floor at 0.40: MegaScale (NSDI '24) shows continued degradation through DP=192 with no plateau.
 * Floor binds at dp≈51000 — effectively never for real configs.
 */
// Order-statistics straggler model
// In BSP, the slowest of N nodes gates the step.
// E[max(X₁...Xₙ)] ≈ μ + σ√(2·ln(N))
// σ = per-node step-time coefficient of variation
//
// Calibrated from MegaScale (NSDI '24) and Meta LLaMA 3.1 on purpose-built
// DGX/InfiniBand fabrics. Not topology-aware — same σ for all cluster types.
// Infrastructure for per-topology σ exists (get/setStragglerSigma) but is
// not yet calibrated for heterogeneous or Ethernet-based clusters.
let _stragglerSigma = 0.014;
/** Get/set STRAGGLER_SIGMA for sensitivity analysis and retune. */
export function getStragglerSigma(): number { return _stragglerSigma; }
export function setStragglerSigma(v: number): void { _stragglerSigma = v; }

export const DP_BW_REF = 64;
let _dpBwAlpha = 0.15;
/** Get/set DP_BW_ALPHA for sensitivity analysis. */
export function getDpBwAlpha(): number { return _dpBwAlpha; }
export function setDpBwAlpha(v: number): void { _dpBwAlpha = v; }
let _dpBwFloor = 0.40;
/** Get/set DP_BW_FLOOR for sensitivity analysis. */
export function getDpBwFloor(): number { return _dpBwFloor; }
export function setDpBwFloor(v: number): void { _dpBwFloor = v; }

/**
 * Second-tier DP bandwidth penalty for extreme DP groups (dp > 256).
 * Captures additional congestion from multi-rail saturation, ECMP collisions,
 * and adaptive routing overhead in large IB fabrics.
 * Zero impact on calibrated benchmarks (all dp ≤ 128).
 */
const DP_BW_REF_2 = 256;
let _dpBwAlpha2 = 0.08;
export function getDPGroupSizePenalty(dp: number, options?: { ref?: number }): number {
  const ref = options?.ref ?? DP_BW_REF;
  let penalty = dp <= ref ? 1.0
    : Math.max(_dpBwFloor, 1 / (1 + _dpBwAlpha * Math.log2(dp / ref)));
  if (dp > DP_BW_REF_2) {
    penalty *= 1 / (1 + _dpBwAlpha2 * Math.log2(dp / DP_BW_REF_2));
  }
  return Math.max(_dpBwFloor, penalty);
}

/**
 * Per-collective latency overhead for large DP groups.
 * NCCL tree latency scales as ~log2(N) with ~3-5µs per inter-node hop.
 * 30µs per log2 step is conservative-realistic (2×log2(N) hops + sync jitter).
 * Zero below dp=64 (within-pod, topology well-mapped).
 */
const DP_LATENCY_REF = 64;
const DP_LATENCY_PER_LOG2_MS = 0.030; // 30µs per log2 step

export function getDPCollectiveLatencyMs(dp: number): number {
  if (dp <= DP_LATENCY_REF) return 0;
  return DP_LATENCY_PER_LOG2_MS * Math.log2(dp / DP_LATENCY_REF);
}


/**
 * Helper functions for memory calculations
 */

/**
 * Calculate parameter memory in bytes
 */
export function calculateParamMemory(
  params: number,
  dtype: DType
): number {
  return params * DTYPE_BYTES[dtype];
}

/**
 * Calculate gradient memory in bytes
 */
export function calculateGradientMemory(
  params: number,
  dtype: DType
): number {
  return params * DTYPE_BYTES[dtype];
}

/**
 * Calculate optimizer state memory for AdamW
 *
 * FP32/TF32: working weights are already fp32, no master copy needed
 *   AdamW: m1(4) + m2(4) = 8 bytes/param
 * BF16/FP16: need fp32 master copy for optimizer step
 *   AdamW: master(4) + m1(4) + m2(4) = 12 bytes/param
 * FP8/FP4: need bf16 master copy (Transformer Engine convention)
 *   AdamW: master(2) + m1(4) + m2(4) = 10 bytes/param
 */
export function calculateOptimizerMemory(
  params: number,
  optimizerType: 'adamw' | 'adam' | 'sgd' | 'adafactor' | 'lion' = 'adamw',
  paramDtype: string = 'bf16'
): number {
  const needsMasterCopy = paramDtype !== 'fp32' && paramDtype !== 'tf32';
  // FP8/FP4: BF16 master (2 bytes) per Transformer Engine convention.
  // BF16/FP16: FP32 master (4 bytes) for loss scaling stability.
  const masterBytes = needsMasterCopy
    ? (paramDtype === 'fp8' || paramDtype === 'fp4' ? 2 : 4)
    : 0;

  switch (optimizerType) {
    case 'adamw':
    case 'adam':
      // 2 states (momentum, variance) in fp32 + optional master weights
      return params * (8 + masterBytes);
    case 'sgd':
      // 1 state (momentum) in fp32 + optional master weights
      return params * (4 + masterBytes);
    case 'adafactor':
      // Row/column factored states, approximately 4 bytes/param (no master copy)
      return params * 4;
    case 'lion':
      // 1 state (momentum) in fp32 + optional master weights
      return params * (4 + masterBytes);
    default:
      return params * (8 + masterBytes);
  }
}

/**
 * Compute the number of parameters each GPU updates during the optimizer step.
 * Accounts for TP/PP/DP/EP sharding and MoE expert distribution.
 * Used by both memory calculations and optimizer step timing.
 */
export function getOptimizerParamsPerGPU(
  model: ModelSpec,
  paramDtype: string,
  sharding: { tp: number; pp: number; dp: number; ep: number; dpType: string },
): number {
  const isMoE = model.isMoE && model.numExperts && sharding.ep >= 1;
  const moeInfo = calculateMoEMemory(model, paramDtype as DType, sharding.ep);
  const shared = isMoE ? moeInfo.sharedParams + moeInfo.routerParams : model.totalParams;
  const expert = isMoE ? moeInfo.expertParams : 0;

  let sharedShard = sharding.tp * sharding.pp;
  let expertShard = sharding.tp * sharding.pp * sharding.ep;
  const expertDPReps = sharding.ep > 1
    ? Math.max(1, Math.floor(sharding.dp / sharding.ep))
    : sharding.dp;

  // All dpTypes except DDP shard the optimizer step across DP
  if (sharding.dpType !== 'ddp') {
    sharedShard *= sharding.dp;
    expertShard *= expertDPReps;
  }

  return shared / sharedShard + expert / expertShard;
}

/**
 * Compute optimizer step wall-clock time in milliseconds (memory-bandwidth-bound).
 * AdamW streams all states through HBM: grads + m + v + master + params.
 */
export function computeOptimizerStepTime(
  paramsPerGPU: number,
  gpu: GPUSpec,
  optimizerType: OptimizerType = 'adamw',
  paramDtype: string = 'bf16',
): number {
  const needsMasterCopy = paramDtype !== 'fp32' && paramDtype !== 'tf32';
  let bytesPerParam: number;
  switch (optimizerType) {
    case 'adamw':
    case 'adam':
      bytesPerParam = needsMasterCopy ? 30 : 28;
      break;
    case 'sgd':
    case 'lion':
      bytesPerParam = needsMasterCopy ? 22 : 20;
      break;
    case 'adafactor':
      bytesPerParam = needsMasterCopy ? 22 : 20;
      break;
    default:
      bytesPerParam = needsMasterCopy ? 30 : 28;
  }
  const BW_EFFICIENCY = 0.75;
  const totalBytes = paramsPerGPU * bytesPerParam;
  const effectiveBW = gpu.memoryBandwidthTBps * 1e12 * BW_EFFICIENCY;
  return (totalBytes / effectiveBW) * 1000; // ms
}

/**
 * Fraction of per-layer FLOPs that selective checkpointing recomputes.
 *
 * Selective AC discards Q/K/V/O projection outputs (the dominant memory savings).
 * During backward, these four linear projections are recomputed from the saved LN1 input.
 * This function returns attnLinearFLOPs / (attnLinearFLOPs + mlpFLOPs) per layer,
 * weighted across dense and MoE layers for mixed architectures.
 *
 * Typical values: GQA models 13-22%, MHA models 25-33%.
 */
export function getSelectiveRecomputeFraction(model: ModelSpec): number {
  const h = model.hiddenSize;
  const qDim = model.numAttentionHeads * model.headDim;
  const kvDim = model.numKvHeads * model.headDim;
  // Q, K, V, O projections: each is a matmul with 2*h*dim FLOPs per token
  const attnLinearFLOPs = 2 * h * (2 * qDim + 2 * kvDim);

  if (model.isMoE && model.numMoELayers && model.numMoELayers < model.numLayers) {
    // Mixed architecture: weighted average of dense and MoE layer fractions
    const I = model.intermediateSize;
    const mlpFLOPsDense = model.gatedMLP ? 6 * h * I : 4 * h * I;
    const denseFrac = attnLinearFLOPs / (attnLinearFLOPs + mlpFLOPsDense);

    const eI = model.expertIntermediateSize ?? I;
    const nActive = model.numActiveExperts ?? 1;
    const nShared = model.numSharedExperts ?? 0;
    const sI = model.sharedExpertIntermediateSize ?? eI;
    const mlpFLOPsMoE = (model.gatedMLP ? 6 : 4) * h * (nActive * eI + nShared * sI);
    const moeFrac = attnLinearFLOPs / (attnLinearFLOPs + mlpFLOPsMoE);

    const numDense = model.numLayers - model.numMoELayers;
    return (numDense * denseFrac + model.numMoELayers * moeFrac) / model.numLayers;
  }

  // Pure dense or pure MoE
  const I = model.isMoE
    ? (model.expertIntermediateSize ?? model.intermediateSize) * (model.numActiveExperts ?? 1)
      + (model.numSharedExperts ?? 0) * (model.sharedExpertIntermediateSize ?? model.expertIntermediateSize ?? model.intermediateSize)
    : model.intermediateSize;
  const mlpFLOPs = model.gatedMLP ? 6 * h * I : 4 * h * I;
  return attnLinearFLOPs / (attnLinearFLOPs + mlpFLOPs);
}

/**
 * Effective backward multiplier blending selective and full AC fractions.
 *
 * When storedLayers < totalLayers, layers 1..K use selective recompute
 * (2.0 + f where f = getSelectiveRecomputeFraction) and layers K+1..N
 * use full recompute (2.85). The overall multiplier is the weighted blend.
 */
export function getEffectiveBackwardMultiplier(
  model: ModelSpec,
  activationCheckpointing: boolean,
  granularity: 'full' | 'selective' | undefined,
  storedLayers?: number,
  totalLayers?: number,
): number {
  if (!activationCheckpointing) return 2.0;
  if (granularity !== 'selective') return 2.85;
  const N = totalLayers ?? model.numLayers;
  const effectiveStored = Math.min(storedLayers ?? N, N);
  const storedFrac = N > 0 ? effectiveStored / N : 1;
  const selectiveMult = 2.0 + getSelectiveRecomputeFraction(model);
  return storedFrac * selectiveMult + (1 - storedFrac) * 2.85;
}

/**
 * Solve for max K (stored layers) that fits the available activation budget.
 *
 * For each candidate K from N down to 0, checks if
 *   K × selectivePerLayer + sqrt(N-K) × fullPerLayer + transient ≤ available.
 * O(N) is intentional — for N≤200 layers the loop body is pure arithmetic.
 */
export function solveMaxStoredLayers(
  numLayers: number,
  selectivePerLayer: number,
  fullPerLayer: number,
  transient: number,
  available: number,
): number {
  for (let k = numLayers; k >= 0; k--) {
    const nonStored = numLayers - k;
    const mem = k * selectivePerLayer
      + (nonStored > 0 ? Math.sqrt(nonStored) * fullPerLayer : 0)
      + transient;
    if (mem <= available) return k;
  }
  return 0;  // even full AC doesn't fit
}

/**
 * Estimate activation memory for forward pass
 * This is a simplified model based on common patterns
 *
 * Attention memory:
 * - Standard attention: O(seq²) - full attention matrix (batch, heads, seq, seq) is materialized
 * - Flash Attention: O(seq) - only one tile at a time is in memory
 */
export function estimateActivationMemory(
  model: ModelSpec,
  seqLength: number,
  microBatchSize: number,
  dtype: DType,
  checkpointing: boolean = true,
  flashAttention: boolean = false,
  ep: number = 1,
  granularity: 'full' | 'selective' = 'full',
  storedLayers?: number,
  totalLayers?: number,
): number {
  const bytesPerElement = DTYPE_BYTES[dtype];
  const tokens = seqLength * microBatchSize;
  const h = model.hiddenSize;

  // Per-layer activation estimate: all tensors saved during forward for backward pass.
  //
  // Attention score memory has shape (batch, heads, seq, seq)
  // - Standard attention stores: pre-softmax scores (2B) + post-softmax probs (2B) + dropout mask (1B)
  //   = 2.5× the single-tensor estimate. Per Korthikanti et al. 2022, Table 1.
  // - Flash Attention handles all internally → O(seq) tile memory only.
  const attentionScoreMemory = flashAttention
    ? model.numAttentionHeads * seqLength * bytesPerElement * microBatchSize
    : model.numAttentionHeads * seqLength * seqLength * bytesPerElement * microBatchSize * 2.5;

  // GQA/MQA: K and V projections are smaller when numKvHeads < numAttentionHeads
  // Use actual headDim (may differ from hiddenSize / numAttentionHeads for models like Qwen3)
  const qDim = model.numAttentionHeads * model.headDim;
  const kvDim = model.numKvHeads * model.headDim;

  // MLP intermediate coefficient: gated MLP (SwiGLU) saves gate + up + product (3I),
  // standard MLP (GeLU/ReLU) saves up + activation output (2I).
  const mlpIntermCoeff = model.gatedMLP ? 3 : 2;

  // ── Per-layer activation decomposition ──
  // Split into shared (always kept), attention (discarded in selective), MLP (always kept).
  //
  // Data flow through one transformer layer:
  //   x (input) → LN1(x) → Q,K,V = QKV(LN1_out) → attn_out = FlashAttn(Q,K,V)
  //   → O = OutProj(attn_out) → y = x + Dropout(O)
  //   → LN2(y) → mlp_out = MLP(LN2_out) → z = y + Dropout(mlp_out)
  //
  // Shared (always kept): x (LN1 input) + y (post-attn residual = LN2 input) + LN2 output = 3h
  // Note: y serves double duty — residual for the add AND LN2 input. Stored once.
  const sharedActivations = 3 * h * tokens * bytesPerElement;

  // Attention (discarded in selective, recomputed from x during backward):
  //   LN1 output (h) + attn output (h) + QKV + attn scores + attn dropout mask
  const attentionActivations = (2 * h + qDim + 2 * kvDim) * tokens * bytesPerElement
    + tokens  // attention dropout mask (1 byte each)
    + attentionScoreMemory;

  // MLP (always kept): MLP intermediates + MLP dropout mask
  // MLP activation per layer — dense vs MoE have different intermediate widths.
  // MoE layers: each token flows through (numActive + numShared) expert MLPs simultaneously.
  const numMoELayers = model.numMoELayers ?? 0;
  const numDenseLayers = model.numLayers - numMoELayers;

  const denseMlpIntermediate = model.intermediateSize * mlpIntermCoeff;
  // EP activation memory reduction: with EP, each GPU holds numExperts/EP local experts.
  // Per-rank intermediates = (numActive/ep) worth of expert intermediates + shared experts.
  // Derivation: With balanced routing, each of numExperts/ep local experts processes
  // tokensPerMB × numActive/numExperts tokens. Total per-rank intermediates:
  //   (numExperts/ep) × (tokensPerMB × numActive/numExperts) × expertIntermediate × mlpCoeff
  //   = (numActive/ep) × tokensPerMB × expertIntermediate × mlpCoeff
  // Shared expert intermediates remain full size on every GPU (not distributed by EP).
  const sharedExpertIntermediate = model.sharedExpertIntermediateSize ?? model.expertIntermediateSize ?? model.intermediateSize;
  const moeMlpIntermediate = model.isMoE && model.expertIntermediateSize
    ? ((model.numActiveExperts ?? 2) / ep * model.expertIntermediateSize
      + (model.numSharedExperts ?? 0) * sharedExpertIntermediate) * mlpIntermCoeff
    : denseMlpIntermediate;

  const denseMlpAct = denseMlpIntermediate * tokens * bytesPerElement + tokens;
  const moeMlpAct = moeMlpIntermediate * tokens * bytesPerElement + tokens;

  // Per-layer activation = shared + attention + MLP
  const densePerLayer = sharedActivations + attentionActivations + denseMlpAct;
  const moePerLayer = sharedActivations + attentionActivations + moeMlpAct;

  // Selective per-layer: discard attention, keep shared + MLP
  const denseSelectivePerLayer = sharedActivations + denseMlpAct;
  const moeSelectivePerLayer = sharedActivations + moeMlpAct;

  // Weighted sum across layer types
  const totalLayerActivations = numDenseLayers > 0 && numMoELayers > 0
    ? numDenseLayers * densePerLayer + numMoELayers * moePerLayer
    : model.numLayers * (numMoELayers > 0 ? moePerLayer : densePerLayer);

  if (checkpointing) {
    if (granularity === 'selective') {
      // Selective AC with stored-layers budget:
      // Layers 1..K store MLP+shared activations (selective checkpointing).
      // Layers K+1..N use full checkpointing (sqrt(N-K) segments).
      // When storedLayers is undefined, all layers use selective (backward compat).
      const N = totalLayers ?? model.numLayers;
      const effectiveStored = Math.min(storedLayers ?? N, N);
      const nonStored = N - effectiveStored;

      const selectivePerLayer = numDenseLayers > 0 && numMoELayers > 0
        ? (numDenseLayers * denseSelectivePerLayer + numMoELayers * moeSelectivePerLayer) / model.numLayers
        : (numMoELayers > 0 ? moeSelectivePerLayer : denseSelectivePerLayer);
      // Stored layers: selective checkpointing (shared + MLP, no attention)
      const storedTotal = effectiveStored * selectivePerLayer;

      // Non-stored layers: full checkpointing (sqrt(K) segments)
      const fullSegment = nonStored > 0
        ? Math.sqrt(nonStored) * Math.max(densePerLayer, moePerLayer)
        : 0;

      // During backward, one layer's attention activations are recomputed transiently.
      // Without FA, this includes the O(seq²) attention score matrix.
      return storedTotal + fullSegment + attentionActivations;
    } else {
      // Full: sqrt(N) checkpointing — peak is the most expensive segment of ~sqrt(N) layers.
      // Use max(densePerLayer, moePerLayer) — the worst-case segment is almost always
      // all one type (e.g., V3: 58/61 MoE, worst segment is all-MoE).
      const N = totalLayers ?? model.numLayers;
      const checkpointedLayers = Math.sqrt(N);
      return checkpointedLayers * Math.max(densePerLayer, moePerLayer);
    }
  } else {
    return totalLayerActivations;
  }
}

/**
 * Convert GPU memory spec to physical byte capacity.
 *
 * NVIDIA markets HBM capacity in GiB but labels it "GB" (e.g., A100 "80 GB"
 * is 80 GiB = 85,899,345,920 bytes). This function applies the GiB→bytes
 * conversion so capacity checks use actual physical HBM size.
 */
export function gpuCapacityBytes(memoryGB: number): number {
  return memoryGB * (1024 ** 3);
}

/** Auto-resolve targets 95% of GPU capacity — leaves 5% headroom for
 * runtime allocations (NCCL workspace, CUDA graphs, peak transient buffers)
 * that the analytical model does not track. Matches OLMo-core/Megatron-LM
 * practice of leaving a memory margin for selective AC budget mode. */
export const STORED_LAYERS_CAPACITY_FRACTION = 0.95;

/**
 * Calculate reserved/framework memory
 * CUDA context, cuDNN workspace, etc.
 *
 * Reserved covers:
 * - CUDA context + cuBLAS/cuDNN workspaces: ~1 GB
 * - PyTorch caching allocator fragmentation: ~7% of physical HBM
 *   (internal fragmentation from variable tensor sizes, gradient accumulation
 *   patterns, and allocation/free interleaving during training)
 *
 * On A100 80 GiB: 1.0 + 6.0 = 7.0 GB reserved, leaving ~78.9 GB available.
 * Real training jobs (Megatron-LM, FSDP) routinely use 90-95% of GPU RAM.
 */
export function calculateReservedMemory(gpuMemoryGB: number): number {
  const capacity = gpuCapacityBytes(gpuMemoryGB);
  const baseReserved = 1.0e9; // ~1 GB CUDA context + framework overhead
  const fragmentationReserve = capacity * 0.07; // 7% allocator fragmentation
  return baseReserved + fragmentationReserve;
}

/**
 * Calculate temporary buffer memory
 * For collective operations, gradient accumulation, etc.
 */
export function calculateTemporaryMemory(
  params: number,
  dtype: DType,
  bucketSizeMB: number = 25
): number {
  // Gradient bucketing buffer
  const bucketSize = bucketSizeMB * 1e6;
  // Additional buffers for communication
  const commBuffers = Math.min(params * DTYPE_BYTES[dtype] * 0.1, bucketSize * 4);
  return bucketSize + commBuffers;
}

// Mutable runtime residuals for retune grid search.
// Dense: framework-level optimizations (kernel fusion, memory planning, communication
// scheduling) that the analytical model doesn't credit.
// MoE: higher than dense because expert-specific overhead (roofline, saturation, grouped
// GEMM efficiency, load imbalance) is modeled explicitly per-expert in 3d-parallel.ts.
// The dense residual absorbs attention-path overhead (softmax, masking, variable memory
// access patterns) that experts don't have.
let _runtimeResidual = 0.655;
let _moeRuntimeResidual = 0.97;

export function getRuntimeResidual(isMoE?: boolean): number {
  return isMoE ? _moeRuntimeResidual : _runtimeResidual;
}
export function setRuntimeResidual(v: number, isMoE?: boolean): void {
  if (isMoE) _moeRuntimeResidual = v;
  else _runtimeResidual = v;
}

/**
 * Shared compute efficiency formula used by all strategy files.
 * Computes GPU kernel utilization (excluding explicit communication overhead).
 *
 * Multiplicative model: efficiency = saturation × memBW × runtimeResidual.
 * Bounded [0,1] by construction: saturation ∈ [0,1], memBW ∈ [0.9,1.1], residual ∈ [0,1].
 */
export function computeComputeEfficiency(
  model: ModelSpec,
  tokensPerMicroBatch: number,
  gpu: GPUSpec,
  computeDtype: string,
  options?: {
    tp?: number;              // hiddenSize divisor for saturation (default 1)
    isMoE?: boolean;          // use MoE runtime residual (default false)
  }
): { efficiency: number; effectiveTFLOPS: number } {
  const tp = options?.tp ?? 1;
  const residual = options?.isMoE ? _moeRuntimeResidual : _runtimeResidual;

  const saturationFactor = getComputeSaturationFactor(tokensPerMicroBatch, model.hiddenSize / tp, gpu);
  const memBWScaling = getMemoryBandwidthScaling(gpu, computeDtype);
  const gpuTFLOPS = getTrainingTFLOPS(gpu, computeDtype);
  const efficiency = saturationFactor * memBWScaling * residual;
  const effectiveTFLOPS = gpuTFLOPS * efficiency;

  return { efficiency, effectiveTFLOPS };
}

/**
 * Non-matmul compute overhead per micro-batch forward pass (milliseconds).
 *
 * Each transformer layer has memory-bandwidth-bound ops between matmuls:
 *   2× RMSNorm (read+write): 4 × tokens × h × bytesPerElem
 *   Activation + gated multiply: (gated ? 5 : 2) × tokens × I × bytesPerElem
 *   2× residual add (read two + write one): 6 × tokens × h × bytesPerElem
 *
 * These ops are <0.2% of FLOPs but ~10-15% of wall-clock time for single-node
 * dense models. With TP+SP, non-matmul ops are split across GPUs, reducing
 * the overhead to ~3-5% for large multi-node runs.
 *
 * Returns forward-only time; callers multiply for backward/recompute.
 */
export function computeNonMatmulTimeMs(
  model: ModelSpec,
  tokensPerMicroBatch: number,
  gpu: GPUSpec,
  options?: { tp?: number; sp?: boolean; pp?: number; ep?: number; flashAttention?: boolean; seqLength?: number; microBatchSize?: number },
): number {
  const tp = options?.tp ?? 1;
  const sp = options?.sp ?? false;
  const pp = options?.pp ?? 1;
  const ep = options?.ep ?? 1;
  const h = model.hiddenSize;
  const I = model.intermediateSize;
  const bytesPerElem = 2; // bf16

  // Norm and residual traffic: replicated across TP ranks (full tokens × h),
  // unless SP is enabled which splits the sequence dimension across TP.
  const seqDivisor = (sp && tp > 1) ? tp : 1;
  const normBytes = 4 * tokensPerMicroBatch * h * bytesPerElem / seqDivisor;
  const residualBytes = 6 * tokensPerMicroBatch * h * bytesPerElem / seqDivisor;

  // MLP activation traffic (SiLU + gated multiply): always TP-sharded because
  // these ops run on column-parallel MLP intermediates (tokens × I/tp).
  const activFactor = model.gatedMLP ? 5 : 2;

  // MoE layers have different MLP dimensions
  const numMoELayers = model.numMoELayers ?? 0;
  const numDenseLayers = model.numLayers - numMoELayers;

  const denseActivBytes = activFactor * tokensPerMicroBatch * (I / tp) * bytesPerElem;

  let moeActivBytes = denseActivBytes;
  if (numMoELayers > 0 && model.isMoE) {
    const expertI = model.expertIntermediateSize ?? I;
    const numActive = model.numActiveExperts ?? 2;
    // With EP, each GPU processes numActive/ep experts' tokens
    const effectiveActive = ep > 1 ? numActive / ep : numActive;
    moeActivBytes = activFactor * tokensPerMicroBatch * (expertI / tp) * effectiveActive * bytesPerElem;
  }

  const densePerLayer = normBytes + denseActivBytes + residualBytes;
  const moePerLayer = normBytes + moeActivBytes + residualBytes;

  // Weighted sum across layer types
  let totalBytes: number;
  if (numMoELayers > 0 && numDenseLayers > 0) {
    totalBytes = numDenseLayers * densePerLayer + numMoELayers * moePerLayer;
  } else if (numMoELayers > 0) {
    totalBytes = model.numLayers * moePerLayer;
  } else {
    totalBytes = model.numLayers * densePerLayer;
  }

  // PP: each GPU stage processes numLayers/pp layers
  const layersPerStage = pp > 1 ? Math.ceil(model.numLayers / pp) : model.numLayers;
  totalBytes = totalBytes * layersPerStage / model.numLayers;

  // Without Flash Attention, standard attention materializes the full seq×seq score matrix
  // in HBM per layer: pre-softmax scores (2B) + post-softmax probs (2B) + dropout mask (1B)
  // = 2.5× the single-tensor estimate (Korthikanti et al. 2022, Table 1).
  // With FA2, all done in SRAM tiles — no intermediate HBM traffic.
  if (options?.flashAttention === false && options?.seqLength && options?.microBatchSize) {
    const headsPerGPU = model.numAttentionHeads / tp;
    const seq = options.seqLength;
    const mbs = options.microBatchSize;
    const attentionHBMPerLayer = headsPerGPU * seq * seq * bytesPerElem * mbs * 2.5;
    totalBytes += attentionHBMPerLayer * layersPerStage;
  }

  // Effective memory BW for elementwise kernels (~65% of peak)
  const ELEMENTWISE_BW_EFFICIENCY = 0.65;
  const effectiveBW = gpu.memoryBandwidthTBps * 1e12 * ELEMENTWISE_BW_EFFICIENCY;

  // Kernel dispatch + autograd bookkeeping: ~40µs per layer
  const KERNEL_LAUNCH_OVERHEAD_MS = 0.04;

  return (totalBytes / effectiveBW) * 1000 + layersPerStage * KERNEL_LAUNCH_OVERHEAD_MS;
}

/**
 * Roofline scaling factor for expert GEMMs.
 * Returns 1.0 when compute-bound, <1.0 when memory-bandwidth-bound.
 *
 * Compares BW ceiling (AI × peakBW) against effectiveTFLOPS.
 * Using effectiveTFLOPS (not raw peak) avoids double-counting with
 * efficiency penalties already in effectiveTFLOPS. When BW-bound,
 * the ceiling dominates regardless of compute efficiency.
 */
export function getExpertGemmRooflineFactor(
  tokPerExpert: number,
  hiddenPerTP: number,
  intermediatePerTP: number,
  bytesPerElem: number,
  gpu: GPUSpec,
  effectiveTFLOPS: number,
): number {
  if (gpu.memoryBandwidthTBps <= 0 || effectiveTFLOPS <= 0) return 1.0;
  const ai = gemmIntensity(tokPerExpert, hiddenPerTP, intermediatePerTP, bytesPerElem);
  const bwCeiling = ai * gpu.memoryBandwidthTBps;  // attainable TFLOPS from BW
  return Math.min(1.0, bwCeiling / effectiveTFLOPS);
}
