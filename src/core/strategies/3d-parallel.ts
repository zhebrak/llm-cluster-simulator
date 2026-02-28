/**
 * 3D Parallelism Strategy
 * Combines Tensor Parallel (TP) + Pipeline Parallel (PP) + Data Parallel (DP)
 *
 * Process group organization:
 * - TP groups: GPUs within same PP stage and DP rank
 * - PP groups: GPUs across stages with same TP rank and DP rank
 * - DP groups: GPUs with same TP rank and PP stage
 *
 * Example for 64 GPUs with TP=4, PP=4, DP=4:
 * - 4 TP groups of 4 GPUs each (within node, NVLink)
 * - 4 PP groups of 4 GPUs each (can span nodes)
 * - 16 DP groups of 4 GPUs each (typically across nodes)
 *
 * Memory per GPU:
 * - 1/(TP*PP) of model parameters
 * - Activations for DP micro-batch
 *
 * Communication:
 * - TP: AllReduce within TP groups (NVLink)
 * - PP: Point-to-point between stages
 * - DP: AllReduce gradients across DP groups
 */
// See docs/PHYSICS.md and docs/STRATEGIES.md for formula derivations and calibration anchors.

import type {
  MemoryBreakdown,
  CommunicationBreakdown,
  TimingBreakdown,
  StrategyValidation,
  SimulationEvent,
  PipelineSchedule,
} from '../../types/index.ts';
import { DTYPE_BYTES } from '../../types/index.ts';
import {
  ParallelismStrategy,
  type StrategyContext,
  calculateOptimizerMemory,
  estimateActivationMemory,
  calculateReservedMemory,
  getOptimizerParamsPerGPU,
  computeOptimizerStepTime,
  getDPGroupSizePenalty,
  getDPCollectiveLatencyMs,
  computeComputeEfficiency,
  getExpertGemmRooflineFactor,
  getEffectiveBackwardMultiplier,
  solveMaxStoredLayers,
  computeNonMatmulTimeMs,
  gpuCapacityBytes,
  STORED_LAYERS_CAPACITY_FRACTION,
} from './base.ts';
import {
  computeFSDPExposedComm,
  computeDDPOverlap,
  computeZeROGradOverlap,
  computeTPOverlap,
  computePPOverlap,
  computeEPSlackOverlap,
  applyProtocolOverhead,
  BUCKET_SIZE_BYTES,
} from './overlap.ts';
import {
  validateHybridCombination,
  validateTPTopology,
  type DPType,
} from '../validation/hybrid-validator.ts';
import { calculateMoEMemory } from '../models/moe.ts';
import {
  computeLoraParamsPerRank,
  getQloraDequantTimeMs,
  getLoraBackwardMultiplier,
  NF4_BYTES_PER_PARAM,
} from './lora.ts';
import { getMatmulSaturationFactor, getGroupedGemmEfficiency } from '../hardware/gpu.ts';
import { getPerNicBandwidthGBps } from '../hardware/interconnect.ts';
import { cpCausalWorkDistribution } from '../physics/derived.ts';

// =========================================================================
// Perturbable fitted parameters (get/set accessors for sensitivity analysis)
// =========================================================================

/** CP AllGather overlap fraction with interleaved PP or DualPipeV. */
let _cpAllGatherOverlapInterleaved = 0.15;
export function getCpAllGatherOverlapInterleaved(): number { return _cpAllGatherOverlapInterleaved; }
export function setCpAllGatherOverlapInterleaved(v: number): void { _cpAllGatherOverlapInterleaved = v; }

/** Per-virtual-stage transition latency: NCCL P2P setup + stream sync + kernel launch. */
let _ppStageTransitionMs = 0.020;
export function getPpStageTransitionMs(): number { return _ppStageTransitionMs; }
export function setPpStageTransitionMs(v: number): void { _ppStageTransitionMs = v; }


export interface ThreeDParallelConfig {
  tp: number;                    // Tensor parallel degree
  pp: number;                    // Pipeline parallel degree
  dp: number;                    // Data parallel degree
  ep: number;                    // Expert parallel degree (MoE only, default 1)
  cp: number;                    // Context parallel degree (ring attention, default 1)
  cpImplementation: 'ring' | 'all-gather';  // Ring attention (P2P overlap) vs AllGather (Megatron-LM)
  numMicroBatches: number;       // For pipeline
  schedule: PipelineSchedule;
  interleavedStages: number;     // Virtual stages per device (v). 1 = standard, ≥2 = interleaved
  sequenceParallel: boolean;     // SP with TP
  dpType: 'ddp' | 'fsdp' | 'zero-1' | 'zero-2' | 'zero-3';
  activationCheckpointing: boolean;
  checkpointingGranularity: 'full' | 'selective';
}

export const DEFAULT_3D_CONFIG: ThreeDParallelConfig = {
  tp: 1,
  pp: 1,
  dp: 1,
  ep: 1,
  cp: 1,
  cpImplementation: 'ring' as const,
  numMicroBatches: 8,
  schedule: '1f1b',
  interleavedStages: 1,
  sequenceParallel: true,
  dpType: 'fsdp',
  activationCheckpointing: true,
  checkpointingGranularity: 'full',
};

export class ThreeDParallelStrategy extends ParallelismStrategy {
  readonly name = '3D Parallelism';
  readonly shortName = '3D';
  readonly description = 'Combines tensor, pipeline, and data parallelism for maximum scalability. Used for training the largest models.';

  private config: ThreeDParallelConfig;
  private _resolvedStoredPerStage?: number;

  constructor(config: Partial<ThreeDParallelConfig> = {}) {
    super();
    this.config = { ...DEFAULT_3D_CONFIG, ...config };
  }

  get totalGPUs(): number {
    return this.config.tp * this.config.pp * this.config.cp * this.config.dp;
  }

  /**
   * Calculate pipeline bubble
   */
  private calculateBubble(): number {
    const pp = this.config.pp;
    const m = this.config.numMicroBatches;

    if (pp <= 1) return 0;

    switch (this.config.schedule) {
      case 'gpipe':
        return (pp - 1) / (pp - 1 + m);
      case '1f1b':
        return (pp - 1) / (pp - 1 + m);
      case 'interleaved-1f1b': {
        const v = this.config.interleavedStages;
        return (pp - 1) / (pp - 1 + m * v);
      }
      case 'dualpipe-v': {
        // DualPipeV: paper says (PP/2-1)(F&B+B-3W) where PP = virtual stages = 2×pp.
        // With balanced F≈B_input≈W and F&B=2F, total useful work per step ≈ 6m*F.
        // Practical heuristic: bubble_frac = (pp-1) / (pp-1 + 6*m).
        // Gives ~1.7% for V3 (pp=16, m=144). Paper claims ~0 when perfectly balanced.
        // Minimum: m >= 2*pp (physical). For m < 2*pp, fall back to standard bubble.
        if (m < 2 * pp) return (pp - 1) / (pp - 1 + m);
        return (pp - 1) / (pp - 1 + 6 * m);
      }
      case 'zero-bubble':
        return 0.05;
      default:
        return (pp - 1) / (pp - 1 + m);
    }
  }

  computeMemoryPerGPU(ctx: StrategyContext): MemoryBreakdown {
    const { model, training, seqLength, microBatchSize, cluster } = ctx;
    const paramDtype = training.dtypes.params;
    const gradDtype = training.dtypes.gradients;
    const activationDtype = training.dtypes.activation;
    const optimizerType = training.optimizer.type;

    const { tp, pp, dp, ep, cp } = this.config;
    const effectiveSeqLength = seqLength / cp;

    // MoE: split params into shared (attention/norms/embeddings) and expert portions
    const isMoE = model.isMoE && model.numExperts && ep >= 1;
    const moeInfo = calculateMoEMemory(model, paramDtype, ep);
    const sharedParams = isMoE ? moeInfo.sharedParams + moeInfo.routerParams : model.totalParams;
    const expertParams = isMoE ? moeInfo.expertParams : 0;

    // Shared params: sharded by TP × PP × (DP if FSDP/ZeRO)
    let sharedParamSharding = tp * pp;
    // Expert params: sharded by TP × PP × EP × (DP if FSDP/ZeRO)
    // TP shards expert FFN weights, PP distributes layers across stages, EP distributes experts
    let expertParamSharding = tp * pp * ep;

    // EP subdivides DP: shared params use full DP, expert params use DP/EP replicas
    // Guard: if EP > DP (invalid config), floor(dp/ep)=0 would cause div-by-zero → Infinity
    const expertDPReplicas = ep > 1 ? Math.max(1, Math.floor(dp / ep)) : dp;

    if (this.config.dpType === 'fsdp' || this.config.dpType === 'zero-3') {
      sharedParamSharding *= dp;
      expertParamSharding *= expertDPReplicas;
    }

    const sharedParamsPerGPU = sharedParams / sharedParamSharding;
    const expertParamsPerGPU = expertParams / expertParamSharding;
    const paramsPerGPU = sharedParamsPerGPU + expertParamsPerGPU;

    let parameters: number;
    let gradients: number;
    let optimizerStates: number;

    if (ctx.lora) {
      // LoRA/QLoRA: base weights at storage bytes + adapter weights partially TP-sharded
      const storageBytes = ctx.lora.method === 'qlora' ? NF4_BYTES_PER_PARAM : DTYPE_BYTES[paramDtype];
      const baseWeightMem = paramsPerGPU * storageBytes;
      // Adapter params: partially TP-sharded, PP distributes layers
      const adapterParamsPerRank = computeLoraParamsPerRank(model, ctx.lora.rank, ctx.lora.targetModules, tp, pp);
      // FSDP further shards adapters across DP
      const dpSharding = (this.config.dpType === 'fsdp' || this.config.dpType === 'zero-3') ? dp : 1;
      const adapterWeightMem = adapterParamsPerRank * DTYPE_BYTES[paramDtype] / dpSharding;
      parameters = baseWeightMem + adapterWeightMem;

      // Gradients: adapter params only, BF16, sharded by DP if FSDP/ZeRO-2/3
      const gradDPSharding = (this.config.dpType === 'fsdp' || this.config.dpType === 'zero-3' || this.config.dpType === 'zero-2') ? dp : 1;
      gradients = adapterParamsPerRank * DTYPE_BYTES['bf16'] / gradDPSharding;

      // Optimizer: adapter params only
      const optParamsPerGPU = adapterParamsPerRank / (this.config.dpType !== 'ddp' ? dp : 1);
      optimizerStates = calculateOptimizerMemory(optParamsPerGPU, optimizerType, paramDtype);
    } else {
      parameters = paramsPerGPU * DTYPE_BYTES[paramDtype];

      // Gradients: sharded by TP, PP, and potentially DP (shared); TP, PP, EP and potentially DP (expert)
      let sharedGradSharding = tp * pp;
      let expertGradSharding = tp * pp * ep;
      if (this.config.dpType === 'fsdp' || this.config.dpType === 'zero-3' || this.config.dpType === 'zero-2') {
        sharedGradSharding *= dp;
        expertGradSharding *= expertDPReplicas;
      }
      gradients = (sharedParams / sharedGradSharding + expertParams / expertGradSharding) * DTYPE_BYTES[gradDtype];

      // Optimizer states (uses shared helper for consistent sharding logic)
      const optimizerParamsPerGPU = getOptimizerParamsPerGPU(
        model, paramDtype, { tp, pp, dp, ep, dpType: this.config.dpType }
      );
      optimizerStates = calculateOptimizerMemory(
        optimizerParamsPerGPU,
        optimizerType,
        paramDtype
      );
    }

    // Activations: affected by TP (with SP), PP (in-flight micro-batches), and CP
    // Each CP rank holds seq_len/cp tokens — activations scale linearly with 1/cp
    // Interleaved 1F1B has the same peak in-flight micro-batches as standard 1F1B:
    // each device holds v virtual stages, each with its own in-flight activation,
    // so peak is still pp (Narayanan 2021 §2.3: "same memory requirements").
    const tokensPerMicroBatch = effectiveSeqLength * microBatchSize;
    const inFlightBase = this.config.schedule === 'dualpipe-v' ? 2 * pp + 1 : pp;
    const inFlightMicroBatches = pp > 1 ? Math.min(inFlightBase, this.config.numMicroBatches) : 1;

    // Get per-layer activation from the standard model (no checkpointing — we apply our own)
    // Use effectiveSeqLength (seqLength/cp) — each CP rank holds seq/cp tokens
    const isSelective = this.config.activationCheckpointing && this.config.checkpointingGranularity === 'selective';
    const fullModelActivations = estimateActivationMemory(
      model, effectiveSeqLength, microBatchSize, activationDtype, false, ctx.flashAttention, ep
    );
    const perLayerActivation = fullModelActivations / model.numLayers;

    // SP and checkpointing applied on top
    // With SP: all tensors effectively 1/tp — TP-sharded tensors (Q,K,V,attn_out,MLP intermediates)
    // are split along hidden dim, SP-sharded tensors (LN inputs, residuals) split along sequence dim.
    // Attention scores use numHeads/tp heads per rank. All reduce to 1/tp per rank.
    // Without SP: only TP-sharded tensors get 1/tp, replicated tensors (4h) stay full size.
    let activationMultiplier: number;
    if (this.config.sequenceParallel && tp > 1) {
      activationMultiplier = 1 / tp;
    } else if (tp > 1) {
      // TP without SP: TP-sharded (qDim + 2*kvDim + h + mlpInterm) vs replicated (4h for LN/residuals)
      const mlpIntermCoeff = model.gatedMLP ? 3 : 2;
      const sharded = (model.numAttentionHeads * model.headDim) + 2 * (model.numKvHeads * model.headDim)
        + model.hiddenSize + mlpIntermCoeff * model.intermediateSize;
      const replicated = 4 * model.hiddenSize;
      activationMultiplier = (sharded / tp + replicated) / (sharded + replicated);
    } else {
      activationMultiplier = 1.0;
    }
    const layersPerStage = pp > 1 ? Math.ceil(model.numLayers / pp) : model.numLayers;

    // Resolve stored layers per-stage for selective AC auto mode.
    // Per-stage resolution (not global K/pp): the per-stage memory budget accounts for
    // inFlightMicroBatches, TP/SP activation multipliers, and MoE overhead — all specific
    // to the 3D parallel decomposition. A global K divided by PP ignores these.
    let resolvedStoredPerStage = layersPerStage;
    if (isSelective) {
      if (ctx.selectiveStoredLayers != null) {
        // Manual: distribute global K across stages with floor (conservative)
        resolvedStoredPerStage = Math.min(Math.floor(ctx.selectiveStoredLayers / pp), layersPerStage);
      } else {
        // Auto: solve per-stage — estimate non-activation memory and find max K that fits.
        // Comm buffers and reserved are not yet computed (they follow activations in the code),
        // so we estimate them here without introducing circular dependencies.
        const gpu = cluster.node.gpu;
        const commActivationBytesEst = Math.max(DTYPE_BYTES[activationDtype], DTYPE_BYTES['bf16']);
        const canUseFP8Est = activationDtype === 'fp8' && gpu.hasTransformerEngine;
        const tpCommBytesEst = canUseFP8Est ? DTYPE_BYTES['fp8'] : commActivationBytesEst;
        const tpBufferEst = model.hiddenSize * tokensPerMicroBatch * tpCommBytesEst * 2;
        const ppBufferEst = model.hiddenSize * tokensPerMicroBatch * commActivationBytesEst * 2;
        const MOE_CF = 1.15;
        const epBufferEst = ep > 1
          ? 2 * tokensPerMicroBatch * model.hiddenSize * DTYPE_BYTES[activationDtype] * 2 * MOE_CF : 0;
        const cpBufferEst = cp > 1 ? (() => {
          const kvDimEst = model.attentionType === 'mla'
            ? (model.kvLoraRank! + model.qkRopeHeadDim!)
            : model.numKvHeads * model.headDim * 2;
          const bufMult = this.config.activationCheckpointing ? 3 : 2;
          return bufMult * kvDimEst * effectiveSeqLength * microBatchSize * commActivationBytesEst;
        })() : 0;
        // FSDP gather buffers / DDP gradient comm buffers
        const dpBufferEst = (this.config.dpType === 'fsdp' || this.config.dpType === 'zero-3')
          ? (() => {
              const ppl = model.totalParams / model.numLayers;
              return (ppl / tp) * DTYPE_BYTES[paramDtype] * 2; // double buffering
            })()
          : (model.totalParams / (tp * pp)) * DTYPE_BYTES[gradDtype] * 0.1;
        const temporaryEst = tpBufferEst + ppBufferEst + dpBufferEst + epBufferEst + cpBufferEst;
        const reservedEst = calculateReservedMemory(gpu.memoryGB);
        // FSDP peak overhead: gather + grad + prefetch buffers
        const fsdpGatherEst = (this.config.dpType === 'fsdp' || this.config.dpType === 'zero-3')
          ? (() => {
              const ppl = model.totalParams / model.numLayers;
              const gp = (ppl / tp) * DTYPE_BYTES[paramDtype];
              const gg = (ppl / tp) * DTYPE_BYTES[gradDtype];
              return gp + gg + gp * 2; // gather + grad + prefetch
            })()
          : 0;
        const nonActivationEst = parameters + gradients + optimizerStates
          + temporaryEst + reservedEst + fsdpGatherEst;
        const available = gpuCapacityBytes(gpu.memoryGB) * STORED_LAYERS_CAPACITY_FRACTION - nonActivationEst;
        // Compute per-layer selective and full activation sizes (with TP/SP multiplier)
        const selectiveForStage = estimateActivationMemory(
          model, effectiveSeqLength, microBatchSize, activationDtype, true, ctx.flashAttention, ep,
          'selective', layersPerStage, layersPerStage,
        ) * activationMultiplier;
        const selectivePerLayerVal = layersPerStage > 0 ? selectiveForStage / layersPerStage : 0;
        const fullPerLayerVal = perLayerActivation * activationMultiplier;
        // transient = attention activations (recomputed transiently during backward)
        // estimateActivationMemory with selective includes one layer of transient attention
        const transient = selectiveForStage - layersPerStage * selectivePerLayerVal;
        // Available per in-flight micro-batch (each micro-batch holds its own activations)
        const availablePerMB = available / inFlightMicroBatches;
        resolvedStoredPerStage = solveMaxStoredLayers(
          layersPerStage, selectivePerLayerVal, fullPerLayerVal,
          transient, availablePerMB,
        );
      }
      // Report model-wide resolved value for UI
      this._resolvedStoredPerStage = resolvedStoredPerStage;
      this._resolvedStoredLayers = Math.min(resolvedStoredPerStage * pp, model.numLayers);
    }

    let baseActivations: number;
    if (this.config.activationCheckpointing) {
      if (isSelective) {
        // Selective AC with per-stage stored-layers resolution.
        // Pass totalLayers = layersPerStage so sqrt operates over per-stage layer counts.
        const selectiveActivations = estimateActivationMemory(
          model, effectiveSeqLength, microBatchSize, activationDtype, true, ctx.flashAttention, ep,
          'selective', resolvedStoredPerStage, layersPerStage,
        );
        baseActivations = selectiveActivations * activationMultiplier;
      } else {
        baseActivations = Math.sqrt(layersPerStage) * perLayerActivation * activationMultiplier;
      }
    } else {
      baseActivations = layersPerStage * perLayerActivation * activationMultiplier;
    }

    // MoE per-layer activation overhead: router logits + dispatch buffers
    // Capacity factor 1.15: headroom for imperfect router load balancing (range 1.0–1.25)
    const MOE_CAPACITY_FACTOR = 1.15;
    let moeActivationOverhead = 0;
    if (isMoE) {
      const numExperts = model.numExperts!;
      const numActiveExperts = model.numActiveExperts ?? 2;
      const numMoELayers = model.numMoELayers ?? model.numLayers;
      const moeLayersPerStage = pp > 1 ? Math.ceil(numMoELayers / pp) : numMoELayers;
      // Router logits: tokens × numExperts × bytes
      const routerLogits = tokensPerMicroBatch * numExperts * DTYPE_BYTES[activationDtype];
      // Dispatch buffers: tokens × numActiveExperts × 4 (int32 indices)
      const dispatchBuffers = tokensPerMicroBatch * numActiveExperts * 4;
      moeActivationOverhead = (routerLogits + dispatchBuffers) * moeLayersPerStage * MOE_CAPACITY_FACTOR;
    }

    // Scale by in-flight micro-batches for PP
    const activations = (baseActivations + moeActivationOverhead) * inFlightMicroBatches;

    // Peak activations and DP buffer depend on dpType
    let peakActivations: number;
    let dpBuffer: number;

    if (this.config.dpType === 'fsdp' || this.config.dpType === 'zero-3') {
      // FSDP/ZeRO-3: during forward/backward, must all-gather full TP-sliced layer from DP peers
      const paramsPerLayer = model.totalParams / model.numLayers;
      // Each TP rank holds 1/tp of each layer; FSDP gathers from DP peers to reconstruct the TP slice
      const gatheredParamBuffer = (paramsPerLayer / tp) * DTYPE_BYTES[paramDtype];
      const gatheredGradBuffer = (paramsPerLayer / tp) * DTYPE_BYTES[gradDtype];
      // Prefetch: 2 layers ahead (matching FSDP default prefetchCount=2)
      const prefetchBuffers = gatheredParamBuffer * 2;
      peakActivations = activations + gatheredParamBuffer + gatheredGradBuffer + prefetchBuffers;
      // Double buffering for FSDP all-gather overlap
      dpBuffer = gatheredParamBuffer * 2;
    } else {
      // DDP/ZeRO-1/ZeRO-2: no gather needed for params, just gradient comm buffers
      peakActivations = activations;
      dpBuffer = (model.totalParams / (tp * pp)) * DTYPE_BYTES[gradDtype] * 0.1;
    }

    // Temporary buffers for all communication types
    // FP8 training with Transformer Engine (Hopper+): TP all-reduce uses FP8 quantized collectives.
    // PP/CP comm stays BF16 (full activations / KV blocks need precision).
    const gpu = cluster.node.gpu;
    const canUseFP8Comm = activationDtype === 'fp8' && gpu.hasTransformerEngine;
    const tpCommBytes = canUseFP8Comm
      ? DTYPE_BYTES['fp8']
      : Math.max(DTYPE_BYTES[activationDtype], DTYPE_BYTES['bf16']);
    const commActivationBytes = Math.max(DTYPE_BYTES[activationDtype], DTYPE_BYTES['bf16']);
    const tpBuffer = model.hiddenSize * tokensPerMicroBatch * tpCommBytes * 2;
    const ppBuffer = model.hiddenSize * tokensPerMicroBatch * commActivationBytes * 2;
    // EP temporary buffers: double-buffered All-to-All (with capacity factor for routing imbalance)
    const epBuffer = ep > 1
      ? 2 * tokensPerMicroBatch * model.hiddenSize * DTYPE_BYTES[activationDtype] * 2 * MOE_CAPACITY_FACTOR
      : 0;
    // CP ring buffer: KV blocks exchanged during ring attention
    // With activation checkpointing: backward recomputes the forward ring while
    // simultaneously running the backward ring → 3× buffers (fwd recompute + bwd + recv).
    // Without checkpointing: standard double-buffer (compute + recv).
    const cpBuffer = cp > 1 ? (() => {
      const kvDimPerLayer = model.attentionType === 'mla'
        ? (model.kvLoraRank! + model.qkRopeHeadDim!)
        : model.numKvHeads * model.headDim * 2;  // K + V
      const bufferMultiplier = this.config.activationCheckpointing ? 3 : 2;
      return bufferMultiplier * kvDimPerLayer * effectiveSeqLength * microBatchSize
        * commActivationBytes;
    })() : 0;
    const temporary = tpBuffer + ppBuffer + dpBuffer + epBuffer + cpBuffer;

    const reserved = calculateReservedMemory(cluster.node.gpu.memoryGB);

    const total = parameters + gradients + optimizerStates + peakActivations + temporary + reserved;

    return {
      parameters,
      gradients,
      optimizerStates,
      activations,
      peakActivations,
      temporary,
      reserved,
      total,
    };
  }

  computeCommunication(ctx: StrategyContext): CommunicationBreakdown {
    const { model, training, seqLength, microBatchSize } = ctx;
    const paramDtype = training.dtypes.params;
    const gradDtype = training.dtypes.gradients;
    const activationDtype = training.dtypes.activation;

    const { tp, pp, dp, ep, cp, numMicroBatches } = this.config;
    const isMoE = model.isMoE && model.numExperts && ep > 1;

    // FP8 training with Transformer Engine (Hopper+): TP all-reduce uses FP8 quantized collectives.
    // PP/CP comm stays BF16 (full activations / KV blocks need precision).
    // EP all-to-all (token dispatch) already uses activationDtype directly below.
    const gpu = ctx.cluster.node.gpu;
    const canUseFP8Comm = activationDtype === 'fp8' && gpu.hasTransformerEngine;
    const tpCommBytes = canUseFP8Comm
      ? DTYPE_BYTES['fp8']
      : Math.max(DTYPE_BYTES[activationDtype], DTYPE_BYTES['bf16']);
    const commActivationBytes = Math.max(DTYPE_BYTES[activationDtype], DTYPE_BYTES['bf16']);

    // Each CP rank holds seq/cp tokens — all per-rank communication uses effectiveSeqLength
    const effectiveSeqLength = seqLength / cp;

    // Tensor Parallel communication (within TP groups)
    // When EP>1, MoE layers only TP-shard attention (expert FFN uses EP, not TP),
    // so MoE layers have half the TP collectives of dense layers.
    let tensorParallel = 0;
    if (tp > 1) {
      const hiddenBytes = effectiveSeqLength * microBatchSize * model.hiddenSize * tpCommBytes;
      const numMoELayers = isMoE ? (model.numMoELayers ?? model.numLayers) : 0;
      const numDenseLayers = model.numLayers - numMoELayers;
      if (this.config.sequenceParallel) {
        const allGatherVolume = hiddenBytes * (tp - 1) / tp;
        const reduceScatterVolume = hiddenBytes * (tp - 1) / tp;
        // Dense: 4 collectives/layer (attn AG+RS + MLP AG+RS, fwd+bwd)
        // MoE:   2 collectives/layer (attn AG+RS only; expert FFN uses EP, not TP)
        tensorParallel = (4 * numDenseLayers + 2 * numMoELayers)
          * (allGatherVolume + reduceScatterVolume);
      } else {
        const allReduceVolume = 2 * hiddenBytes * (tp - 1) / tp;
        // Dense: 2 AllReduces/layer (attn + MLP), MoE: 1 (attn only)
        tensorParallel = (2 * numDenseLayers + 1 * numMoELayers) * allReduceVolume;
      }
    }

    // Pipeline Parallel communication (between stages)
    // Interleaved PP has pp×v - 1 stage boundaries per micro-batch (vs pp - 1)
    let pipelineParallel = 0;
    if (pp > 1) {
      const activationSize = effectiveSeqLength * microBatchSize * model.hiddenSize * commActivationBytes;
      const v = this.config.schedule === 'interleaved-1f1b' ? this.config.interleavedStages : 1;
      // DualPipeV: V-shape doubles PP comm (forward + reverse direction)
      const boundaries = this.config.schedule === 'dualpipe-v' ? 2 * (pp - 1) : pp * v - 1;
      pipelineParallel = 2 * activationSize * numMicroBatches * boundaries;
    }

    // Expert Parallel communication (All-to-All for MoE)
    // 4× per MoE layer per step: fwd dispatch + fwd combine + bwd dispatch + bwd combine
    // EP dispatches local tokens (seq/cp per CP rank)
    let expertParallel = 0;
    if (isMoE) {
      const tokens = effectiveSeqLength * microBatchSize;
      const numMoELayers = model.numMoELayers ?? model.numLayers;
      const numActive = model.numActiveExperts ?? 2;
      // Routing locality: fraction of worst-case uniform-random traffic that actually crosses
      // EP boundaries. Two complementary effects:
      //
      // 1. Density-based locality: 1/(1+density) where density = expertsPerRank / numActive.
      //    Learned routers preferentially activate local experts proportional to supply/demand:
      //      density=1 (supply=demand) → 50% cross-rank
      //      density=4 (large local pool) → 20% cross-rank
      //      density→∞ (all experts local) → 0% cross-rank
      //
      // 2. Device-limited routing (optional): some architectures hard-cap the number of EP
      //    groups each token can contact. DeepSeek V3 uses M=4 (each token contacts at most
      //    4 of 32 EP groups → 12.5% cross-rank). When set, this is a tighter bound than
      //    density alone. Use min(densityLocality, M/ep) to take the tighter constraint.
      const numExperts = model.numExperts ?? 8;
      const expertsPerRank = numExperts / ep;
      const densityLocality = 1 / (1 + expertsPerRank / numActive);
      const deviceLimit = model.routingDeviceLimit;
      const routingLocality = deviceLimit != null
        ? Math.min(densityLocality, deviceLimit / ep)
        : densityLocality;
      const allToAllPerLayer = 4 * tokens * model.hiddenSize * numActive * DTYPE_BYTES[activationDtype]
        * (ep - 1) / ep * routingLocality;
      expertParallel = allToAllPerLayer * numMoELayers;
    }

    // Context Parallel communication (ring attention KV exchange)
    // Each ring step exchanges one KV chunk, (cp-1) steps per layer, fwd + bwd
    let contextParallel = 0;
    if (cp > 1) {
      const kvDim = model.attentionType === 'mla'
        ? (model.kvLoraRank! + model.qkRopeHeadDim!)
        : model.numKvHeads * model.headDim * 2;
      const kvChunkBytes = kvDim * (seqLength / cp) * microBatchSize * commActivationBytes;
      contextParallel = kvChunkBytes * (cp - 1) * model.numLayers * 2;  // fwd + bwd
    }

    // Data Parallel communication (across DP groups)
    // EP subdivides DP: shared params AllReduce across full DP,
    // expert params AllReduce across DP/EP replicas within each EP subgroup
    let dataParallel = 0;
    if (dp > 1) {
      const moeInfo = calculateMoEMemory(model, gradDtype, ep);
      const sharedParamsPerGPU = isMoE
        ? (moeInfo.sharedParams + moeInfo.routerParams) / (tp * pp)
        : model.totalParams / (tp * pp);
      const expertParamsPerGPU = isMoE
        ? moeInfo.expertParams / (ep * tp * pp)  // experts sharded across EP, TP, and PP
        : 0;

      // Shared params: AllReduce across full DP dimension
      const sharedFactor = (dp - 1) / dp;
      // Expert params: AllReduce across DP/EP replicas (smaller group)
      const expertDPGroup = ep > 1 ? Math.floor(dp / ep) : dp;
      const expertFactor = expertDPGroup > 1 ? (expertDPGroup - 1) / expertDPGroup : 0;

      if (ctx.lora) {
        // LoRA: only adapter gradients need DP sync (always BF16, even with FP8 compute).
        // FSDP still AllGathers full base weights (frozen but needed for forward pass).
        const adapterParamsPerGPU = computeLoraParamsPerRank(model, ctx.lora.rank, ctx.lora.targetModules, tp, pp);
        const storageBytes = ctx.lora.method === 'qlora' ? NF4_BYTES_PER_PARAM : DTYPE_BYTES[paramDtype];

        switch (this.config.dpType) {
          case 'ddp': {
            // AllReduce adapter grads only at BF16
            dataParallel = 2 * adapterParamsPerGPU * DTYPE_BYTES['bf16'] * sharedFactor;
            break;
          }
          case 'fsdp':
          case 'zero-3': {
            // Forward AllGather: full base weights at storage bytes (NF4 for QLoRA)
            // Backward ReduceScatter: adapter grads only at BF16
            dataParallel = (2 * sharedParamsPerGPU * storageBytes + adapterParamsPerGPU * DTYPE_BYTES['bf16']) * sharedFactor +
                           (2 * expertParamsPerGPU * storageBytes) * expertFactor;
            break;
          }
          case 'zero-2': {
            // ReduceScatter adapter grads + AllGather adapter params after optimizer
            dataParallel = (adapterParamsPerGPU * DTYPE_BYTES['bf16'] + adapterParamsPerGPU * DTYPE_BYTES[paramDtype]) * sharedFactor;
            break;
          }
          case 'zero-1': {
            // AllReduce adapter grads + AllGather adapter params after optimizer
            dataParallel = (2 * adapterParamsPerGPU * DTYPE_BYTES['bf16'] + adapterParamsPerGPU * DTYPE_BYTES[paramDtype]) * sharedFactor;
            break;
          }
        }
      } else {
        switch (this.config.dpType) {
          case 'ddp': {
            const sharedGradBytes = sharedParamsPerGPU * DTYPE_BYTES[gradDtype];
            const expertGradBytes = expertParamsPerGPU * DTYPE_BYTES[gradDtype];
            dataParallel = 2 * sharedGradBytes * sharedFactor + 2 * expertGradBytes * expertFactor;
            break;
          }
          case 'fsdp':
          case 'zero-3': {
            const sharedParamBytes = sharedParamsPerGPU * DTYPE_BYTES[paramDtype];
            const sharedGradBytes = sharedParamsPerGPU * DTYPE_BYTES[gradDtype];
            const expertParamBytes = expertParamsPerGPU * DTYPE_BYTES[paramDtype];
            const expertGradBytes = expertParamsPerGPU * DTYPE_BYTES[gradDtype];
            dataParallel = (2 * sharedParamBytes + sharedGradBytes) * sharedFactor +
                           (2 * expertParamBytes + expertGradBytes) * expertFactor;
            break;
          }
          case 'zero-2': {
            const sharedParamBytes = sharedParamsPerGPU * DTYPE_BYTES[paramDtype];
            const sharedGradBytes = sharedParamsPerGPU * DTYPE_BYTES[gradDtype];
            const expertParamBytes = expertParamsPerGPU * DTYPE_BYTES[paramDtype];
            const expertGradBytes = expertParamsPerGPU * DTYPE_BYTES[gradDtype];
            dataParallel = (sharedGradBytes + sharedParamBytes) * sharedFactor +
                           (expertGradBytes + expertParamBytes) * expertFactor;
            break;
          }
          case 'zero-1': {
            const sharedParamBytes = sharedParamsPerGPU * DTYPE_BYTES[paramDtype];
            const sharedGradBytes = sharedParamsPerGPU * DTYPE_BYTES[gradDtype];
            const expertParamBytes = expertParamsPerGPU * DTYPE_BYTES[paramDtype];
            const expertGradBytes = expertParamsPerGPU * DTYPE_BYTES[gradDtype];
            dataParallel = (2 * sharedGradBytes + sharedParamBytes) * sharedFactor +
                           (2 * expertGradBytes + expertParamBytes) * expertFactor;
            break;
          }
        }
      }
    }

    return {
      dataParallel,
      tensorParallel,
      pipelineParallel,
      expertParallel,
      contextParallel,
      total: dataParallel + tensorParallel + pipelineParallel + expertParallel + contextParallel,
    };
  }

  computeTiming(ctx: StrategyContext): TimingBreakdown {
    const { model, training, seqLength, microBatchSize, cluster, gradientAccumulationSteps } = ctx;
    const gpu = cluster.node.gpu;

    const { tp, pp, dp, ep, cp } = this.config;
    const layersPerStage = pp > 1 ? Math.ceil(model.numLayers / pp) : model.numLayers;

    // Compute time per GPU — per-layer model for MoE with EP
    // CP reduces per-rank compute: each rank processes seq_len/cp tokens
    const effectiveSeqLength = seqLength / cp;
    const tokensPerMicroBatch = effectiveSeqLength * microBatchSize;
    const flopsPerToken = model.flopsPerToken;

    const computeDtype = training.dtypes.compute;

    // Compute efficiency — uses MoE residual for MoE models (lower due to unmodeled
    // per-layer overhead: router backward, permutation backward, aux loss, expert scheduling).
    // Expert kernel efficiency is modeled per-layer below via getMatmulSaturationFactor().
    const { effectiveTFLOPS } = computeComputeEfficiency(
      model, tokensPerMicroBatch, gpu, computeDtype, { tp, isMoE: model.isMoE }
    );

    // Non-matmul overhead (memory-bandwidth-bound ops: norms, activations, residual adds)
    const nonMatmulPerMB = computeNonMatmulTimeMs(model, tokensPerMicroBatch, gpu,
      { tp, sp: this.config.sequenceParallel, pp, ep,
        flashAttention: ctx.flashAttention, seqLength: ctx.seqLength, microBatchSize: ctx.microBatchSize });

    // Per-layer forward timing model
    // For MoE models, each MoE layer has router, permutation, and load imbalance overhead
    // plus (when EP>1) all-to-all communication:
    //   router + permute + dispatch_alltoall → expert_compute/ep → combine_alltoall + unpermute
    // When EP=1, all-to-all volume naturally goes to zero via (ep-1)/ep = 0, but router,
    // permutation, and load imbalance overhead still apply (they occur locally regardless of EP).
    // For dense layers, each layer is just compute/tp.
    // PP distributes layers across pipeline stages — each GPU runs numLayers/pp layers.
    // TP shards matrices within a layer. So per-layer compute divides by TP only,
    // and we sum over numLayers/pp layers.
    const isMoE = model.isMoE && !!model.numExperts;
    let forwardTimePerMicrobatch: number;
    let alltoallTimePerLayer = 0;
    let moeLayersPerGPU = 0;
    let epCommForward = 0;        // Total EP comm per microbatch forward (gross)
    let epCommExposedFwd = 0;     // EP comm not overlapped with compute slack
    let totalDispatchOverheadFwd = 0;  // EP coord overhead per microbatch forward
    let computeOnlyForward = 0;       // Forward compute time excluding EP comm
    // Per-layer timing for FSDP pipeline overlap (hoisted from MoE block)
    let _perLayerDenseFwd = 0;    // dense layer fwd time (ms)
    let _perLayerMoeFwd = 0;      // MoE layer fwd time (ms, includes EP comm)
    let _perLayerDenseCount = 0;  // dense layers per PP stage
    let physicsFloor = 0;             // Minimum forward time from peak FLOPS
    let totalDispatchOverhead = 0;    // EP dispatch overhead per microbatch
    // MoE diagnostic intermediates (per-MB, scaled to per-step after backwardMultiplier is known)
    let _diag_expertComputePerMB = 0;     // Expert compute × loadImbalance × moeLayersPerGPU
    let _diag_denseComputePerMB = 0;      // Dense + MoE-layer attention per MB
    let _diag_overheadPerMB = 0;          // Router + permutation + coord + physics floor excess
    let _diag_groupedGemmFactor = 1.0;
    let _diag_routedFlopFraction = 0;

    if (isMoE) {
      const numMoELayers = model.numMoELayers ?? model.numLayers;
      const numDenseLayers = model.numLayers - numMoELayers;
      const numActive = model.numActiveExperts ?? 2;
      const numShared = model.numSharedExperts ?? 0;
      const expertIntermediate = model.expertIntermediateSize ?? model.intermediateSize;
      const sharedIntermediate = model.sharedExpertIntermediateSize ?? expertIntermediate;
      const mlpMatrices = model.gatedMLP ? 3 : 2; // SwiGLU: 3 (gate+up+down); standard: 2 (up+down)

      // ── Per-layer FLOPs breakdown ──
      // Expert FLOPs: routed experts use expertIntermediate, shared experts use sharedIntermediate
      const routedFlopsPerMoELayer = numActive * mlpMatrices * model.hiddenSize * expertIntermediate * 2;
      const sharedFlopsPerMoELayer = numShared * mlpMatrices * model.hiddenSize * sharedIntermediate * 2;
      const expertFlopsPerMoELayer = routedFlopsPerMoELayer + sharedFlopsPerMoELayer;
      // Dense MLP FLOPs per dense layer: mlpMatrices × H × intermediateSize × 2
      const denseMlpFlopsPerDenseLayer = mlpMatrices * model.hiddenSize * model.intermediateSize * 2;
      // Attention + norms FLOPs per layer (same for every layer):
      // total = attention + norms + dense MLP (on dense layers) + expert MLP (on MoE layers)
      const totalExpertFlops = expertFlopsPerMoELayer * numMoELayers;
      const totalDenseMlpFlops = denseMlpFlopsPerDenseLayer * numDenseLayers;
      const totalAttnNormFlops = flopsPerToken - totalExpertFlops - totalDenseMlpFlops;
      const attnNormFlopsPerLayer = totalAttnNormFlops / model.numLayers;

      // Convert FLOPs to time: per-layer compute divided by TP (not PP — PP is layer distribution)
      const flopsToMs = (flops: number) => tokensPerMicroBatch * flops / tp / (effectiveTFLOPS * 1e12) * 1000;

      // Dense layer: attention + norms + dense MLP, all sharded by TP
      const denseLayerTime = flopsToMs(attnNormFlopsPerLayer + denseMlpFlopsPerDenseLayer);
      // MoE layer dense part: attention + norms only (no dense MLP)
      const moeLayerAttnTime = flopsToMs(attnNormFlopsPerLayer);
      // Routed expert compute per GPU (compute-bound estimate)
      // EP distributes experts across ranks; each expert receives tokens from ALL
      // ep ranks via all-to-all. Per-GPU expert FLOPs are EP-invariant:
      //   experts_per_GPU = numExperts/ep, tokens_per_expert ∝ ep/numExperts → ep cancels.
      // TP and EP are orthogonal: TP shards each expert's weights (column/row parallel)
      // regardless of EP. flopsToMs includes /tp, which is correct for both branches.
      const routedComputeTimeCB = flopsToMs(routedFlopsPerMoELayer);
      // Shared expert compute per GPU (not distributed by EP — runs on every rank)
      const sharedComputeTimeCB = flopsToMs(sharedFlopsPerMoELayer);
      // Roofline: small expert GEMMs (low M, small K or N after TP sharding) can be
      // memory-bandwidth-bound rather than compute-bound.
      const numExperts = model.numExperts ?? 8;
      const tokPerExpert = tokensPerMicroBatch * numActive * ep / numExperts;
      // Expert GEMM dimensions for saturation/roofline analysis.
      // EP>1: each EP rank holds full expert weight matrices for its local experts.
      // The TP AllReduce happens after the expert block, not within individual expert GEMMs.
      // EP=1: experts are TP-sharded like dense layers, so GEMM dims use /tp.
      const expertK = ep > 1 ? model.hiddenSize : model.hiddenSize / tp;
      const expertN = ep > 1 ? expertIntermediate : expertIntermediate / tp;
      const rooflineFactor = getExpertGemmRooflineFactor(
        tokPerExpert, expertK, expertN,
        DTYPE_BYTES[computeDtype], gpu, effectiveTFLOPS,
      );
      // Expert GEMM saturation: per-expert matrices may not fill tensor cores.
      // Uses dominant expert GEMM dimensions (M=tokPerExpert, N=expertN).
      // Floor at 0.50: extremely small expert GEMMs (e.g., V2's 160 experts with EP=1)
      // can't degrade worse than 2×. Below this, the power-law overshoots because
      // getMatmulSaturationFactor is calibrated for dense GEMMs, not micro-expert workloads
      // where kernel launch overhead dominates over occupancy loss.
      const routedSaturation = Math.max(0.50, getMatmulSaturationFactor(tokPerExpert, expertN, gpu));
      // Shared experts process ALL tokens with larger intermediates — better saturated
      const sharedSaturation = numShared > 0
        ? Math.max(0.50, getMatmulSaturationFactor(tokensPerMicroBatch, sharedIntermediate / tp, gpu))
        : 1.0;
      // Grouped GEMM efficiency: fine-grained MoE (many small experts) suffers from
      // L2 cache thrashing between expert weight matrices and wave quantization in
      // grouped GEMM kernels. Uses RAW expertIntermediate (not /tp) — architectural
      // property, not deployment property.
      // Modulated by roofline: penalty diminishes when GEMM is already memory-bound
      // (L2 data reuse is irrelevant when the GEMM doesn't benefit from cache).
      const expertsPerGPU = numExperts / ep;
      const rawGemmFactor = getGroupedGemmEfficiency(expertIntermediate, expertsPerGPU, gpu);
      const groupedGemmFactor = 1.0 - Math.min(rooflineFactor, 1.0) * (1.0 - rawGemmFactor);
      const expertComputeTime = routedComputeTimeCB / rooflineFactor / routedSaturation / groupedGemmFactor
        + sharedComputeTimeCB / sharedSaturation;

      // Per-layer all-to-all: dispatch + combine. Backward EP comm uses separate
      // multiplier (2× with AC, 1× without) — see backwardEPCommMultiplier below.
      // Volume per direction = tokens × H × numActive × dtype × (ep-1)/ep × routingLocality
      const activationDtype = training.dtypes.activation;
      const epWithinNode = ep * tp <= cluster.gpusPerNode;
      const epBandwidth = epWithinNode
        ? cluster.node.intraNodeInterconnect.bandwidthGBps
        : cluster.interNodeBandwidthGBps;
      // Routing locality: same model as computeCommunication() — see detailed comment there.
      // min(densityLocality, deviceLimit/ep) when device-limited routing is available.
      const expertsPerRank = numExperts / ep;
      const densityLocality = 1 / (1 + expertsPerRank / numActive);
      const deviceLimit = model.routingDeviceLimit;
      const routingLocality = deviceLimit != null
        ? Math.min(densityLocality, deviceLimit / ep)
        : densityLocality;
      // Device-limited routing bounds the effective EP group size for latency and
      // coordination scaling. With M=4 on EP=32, each rank exchanges data with
      // at most M peers — NCCL sync tree depth and scheduling overhead scale with
      // actual peer count. BW group-size penalty uses raw ep: NCCL buffer
      // allocation and message scheduling operate over the full communicator.
      const alltoallVolumePerDirection = tokensPerMicroBatch * model.hiddenSize * numActive
        * DTYPE_BYTES[activationDtype] * (ep - 1) / ep * routingLocality;
      // dispatch + combine = 2 directions per forward pass
      // All-to-all achieves lower bandwidth efficiency than point-to-point:
      // NCCL all-to-all on NVLink: ~50-60% of peak; on IB: ~40-50% (multi-hop, congestion)
      // All-to-all BW efficiency degrades with group size — uses unified getDPGroupSizePenalty
      // with ref=8 (EP reference point). All-to-all is O(n²) message pairs vs O(n) ring
      // steps for AllReduce, so EP=8 is already a large group. The DP_BW_ALPHA=0.15 curve
      // shape correctly penalizes EP above 8.
      const baseAlltoallEfficiency = epWithinNode ? 0.55 : 0.45;
      const groupSizePenalty = getDPGroupSizePenalty(ep, { ref: 8 });
      const alltoallBWEfficiency = baseAlltoallEfficiency * groupSizePenalty;
      const effectiveEpBW = epBandwidth * alltoallBWEfficiency;
      // Latency per collective: NCCL synchronization + kernel launch overhead.
      // EP latency scaling zeroed pending high-EP anchor (EP>=16)
      const baseLatency = epWithinNode ? 0.05 : 0.1; // NVLink ~50μs, IB ~100μs
      const alltoallLatencyMs = baseLatency;
      const alltoallBWTime = 2 * alltoallVolumePerDirection / (effectiveEpBW * 1e9) * 1000;
      // EP=1: no all-to-all occurs at all — skip both BW time and latency
      alltoallTimePerLayer = ep > 1 ? alltoallBWTime + 2 * alltoallLatencyMs : 0;

      // Load imbalance: with EP>1, the max-loaded expert among EP ranks determines
      // wall-clock time (cross-GPU sync barrier). Derived from multinomial order
      // statistics: E[max] ≈ μ + σ√(2 ln E), damped by aux-loss routing correction.
      // EP=1: no sync barrier, just minor grouped GEMM inefficiency from variable batch sizes.
      let loadImbalanceFactor: number;
      if (ep > 1) {
        // Load imbalance damping zeroed pending 3rd MoE anchor — raw multinomial
        // prediction suppressed. Reintroduce via loadImbalanceFactor() from
        // physics/derived.ts with damping once validated.
        loadImbalanceFactor = 1;
      } else {
        loadImbalanceFactor = 1.02;  // minor grouped GEMM inefficiency from variable token counts
      }
      // Router overhead: gating network forward + top-k selection + permutation/unpermutation
      // Router FLOPs: tokens × hiddenSize × numExperts × 2 (matmul) — small but non-zero
      // Permutation: reorder tokens for dispatch, un-reorder after combine — memory-bandwidth bound
      const routerFlops = tokensPerMicroBatch * model.hiddenSize * numExperts * 2;
      const routerTime = routerFlops / (effectiveTFLOPS * 1e12) * 1000;
      // Permutation is memory-bandwidth bound: read + write tokens × hiddenSize × dtype, twice
      if (gpu.memoryBandwidthTBps <= 0) throw new Error(`GPU ${gpu.name} has zero memory bandwidth`);
      const memBWGBps = gpu.memoryBandwidthTBps * 1000; // Convert TB/s to GB/s
      const permutationBytes = 4 * tokensPerMicroBatch * model.hiddenSize * DTYPE_BYTES[training.dtypes.activation];
      const permutationTime = permutationBytes / (memBWGBps * 1e9) * 1000;

      // EP coordination overhead zeroed pending high-EP anchor (EP>=16)
      const coordOverheadPerLayer = 0;

      // FP8 quantized collectives (Hopper Transformer Engine): EP dispatch+combine
      // transfer FP8 activations (volume already uses FP8 bytes). Quantization
      // (BF16→FP8) before dispatch and dequantization (FP8→BF16) after combine
      // add memory-bandwidth-bound overhead on the critical path. The reverse
      // conversions (dispatch dequant at expert, combine quant at expert) overlap
      // with expert compute. CUDA multi-stream pipelining hides a fraction of
      // the remaining quant/dequant behind adjacent NCCL transfers: TE userbuffers
      // issue cast kernels on separate streams concurrent with NCCL collectives
      // (see Transformer Engine release notes on wgrad/dgrad GEMM–NCCL overlap).
      // No published paper precisely quantifies this ratio for EP cast operations.
      const FP8_QUANT_STREAM_OVERLAP = 0.5; // ~50% of cast wall time hidden by multi-stream pipelining
      const canUseFP8Comm = activationDtype === 'fp8' && gpu.hasTransformerEngine;
      let fp8QuantOverheadPerLayer = 0;
      if (canUseFP8Comm && ep > 1) {
        const elementsPerDirection = tokensPerMicroBatch * model.hiddenSize * numActive;
        // Each conversion: read src (2B or 1B) + write dst (1B or 2B) = 3B/element
        const bytesPerConversion = 3 * elementsPerDirection;
        // 2 critical-path conversions (dispatch quant, combine dequant)
        const criticalPathConversions = 2;
        fp8QuantOverheadPerLayer = criticalPathConversions * bytesPerConversion
          * (1 - FP8_QUANT_STREAM_OVERLAP) / (memBWGBps * 1e9) * 1000;
      }

      const expertBlockTimeBase = alltoallTimePerLayer + expertComputeTime * loadImbalanceFactor
        + routerTime + permutationTime;
      const dispatchOverheadPerLayer = coordOverheadPerLayer + fp8QuantOverheadPerLayer;
      const expertBlockTime = expertBlockTimeBase;
      const moeLayerTime = moeLayerAttnTime + expertBlockTime;

      // PP distributes layers: each GPU runs numLayers/pp layers
      moeLayersPerGPU = numMoELayers / pp;
      const denseLayersPerGPU = numDenseLayers / pp;

      // Per-layer model: optimistic forward time reflecting EP tradeoff
      const perLayerForward = denseLayerTime * denseLayersPerGPU + moeLayerTime * moeLayersPerGPU;

      // EP comm time embedded in MoE layer timing — must not be hidden by physics floor
      epCommForward = alltoallTimePerLayer * moeLayersPerGPU;
      computeOnlyForward = perLayerForward - epCommForward;

      // Physics floor: can't exceed peak throughput for total per-GPU FLOPs.
      // Expert compute is EP-invariant (ep cancels) and TP-sharded (÷tp) like dense layers.
      // Both branches use ÷(tp×pp). Expert portion adds grouped GEMM penalty.
      const routedExpertFlopsTotal = routedFlopsPerMoELayer * numMoELayers;
      const routedFlopFraction = flopsPerToken > 0
        ? routedExpertFlopsTotal / flopsPerToken
        : 0;
      if (ep > 1) {
        const nonExpertFlopsPerToken = flopsPerToken - routedExpertFlopsTotal;
        const nonExpertFloorMs = tokensPerMicroBatch * nonExpertFlopsPerToken / (tp * pp)
          / (effectiveTFLOPS * 1e12) * 1000;
        const expertFloorMs = tokensPerMicroBatch * routedExpertFlopsTotal / (tp * pp)
          / (effectiveTFLOPS * groupedGemmFactor * 1e12) * 1000;
        physicsFloor = nonExpertFloorMs + expertFloorMs;
      } else {
        const basePhysicsFloor = tokensPerMicroBatch * flopsPerToken / (tp * pp)
          / (effectiveTFLOPS * 1e12) * 1000;
        physicsFloor = basePhysicsFloor * (1 + routedFlopFraction * (1 / groupedGemmFactor - 1));
      }

      // When physics floor > computeOnlyForward, the GPU has "slack" compute cycles
      // (the floor forces time = floor, but actual compute finishes sooner). EP dispatch
      // can overlap with this slack — the network transfers tokens while the GPU computes.
      const floorSlack = Math.max(0, physicsFloor - computeOnlyForward);
      const epOverlapWithSlack = computeEPSlackOverlap(epCommForward, floorSlack);
      epCommExposedFwd = epCommForward - epOverlapWithSlack;
      // Dispatch overhead is non-compute framework time — applied after the physics floor,
      // not absorbed by it. Only affects MoE layers.
      totalDispatchOverhead = dispatchOverheadPerLayer * moeLayersPerGPU;
      totalDispatchOverheadFwd = totalDispatchOverhead;
      forwardTimePerMicrobatch = Math.max(computeOnlyForward, physicsFloor) + epCommExposedFwd + totalDispatchOverhead;
      // Populate MoE diagnostic intermediates (per-MB, forward only)
      _diag_expertComputePerMB = expertComputeTime * loadImbalanceFactor * moeLayersPerGPU;
      _diag_denseComputePerMB = denseLayerTime * denseLayersPerGPU + moeLayerAttnTime * moeLayersPerGPU;
      const floorExcess = Math.max(0, physicsFloor - computeOnlyForward);
      _diag_overheadPerMB = (routerTime + permutationTime + coordOverheadPerLayer) * moeLayersPerGPU + floorExcess;
      _diag_groupedGemmFactor = groupedGemmFactor;
      _diag_routedFlopFraction = routedFlopFraction;
      // Hoist per-layer timing for FSDP pipeline overlap
      _perLayerDenseFwd = denseLayerTime;
      _perLayerMoeFwd = moeLayerTime;
      _perLayerDenseCount = denseLayersPerGPU;
    } else {
      // Dense model (MoE models enter the isMoE path above, including EP=1)
      const totalFlops = tokensPerMicroBatch * flopsPerToken / (tp * pp);
      forwardTimePerMicrobatch = (totalFlops / (effectiveTFLOPS * 1e12)) * 1000;
    }
    // QLoRA dequantization: bandwidth-bound (reads NF4 + writes BF16 = 2.5 bytes/param).
    // Applies to forward time per microbatch — dequant is on the critical path before compute.
    if (ctx.lora?.method === 'qlora') {
      forwardTimePerMicrobatch += getQloraDequantTimeMs(model, gpu, tp, pp);
    }
    // Backward multiplier: recompute cost is physically identical regardless of DP type.
    // FSDP AllGather/backward overlap is credited solely via computeFSDPExposedComm() —
    // using a reduced multiplier here would double-count the same overlap window.
    // LoRA: backward skips frozen weight grads → ~1.05× (no ckpt) or ~2.05× (with ckpt).
    const backwardMultiplier = ctx.lora
      ? getLoraBackwardMultiplier(model, ctx.lora, ctx.activationCheckpointing, ctx.checkpointingGranularity, this._resolvedStoredPerStage, layersPerStage)
      : getEffectiveBackwardMultiplier(
          model, ctx.activationCheckpointing, ctx.checkpointingGranularity,
          this._resolvedStoredPerStage,
          layersPerStage,
        );
    // EP communication scales differently from compute in backward pass.
    // Compute: full AC recomputes entire forward (1×) + gradient backward (~1.85×) = 2.85×.
    // EP comm: recompute re-runs dispatch+combine (1× if AC) + gradient backward
    // reverses dispatch+combine (1×, same tensor shape). Total: 2× with AC, 1× without.
    // The backward multiplier applies only to the compute portion.
    const epCommFwd = epCommExposedFwd + totalDispatchOverheadFwd;
    const computeFwd = forwardTimePerMicrobatch - epCommFwd;
    const backwardEPCommMultiplier = ctx.activationCheckpointing ? 2.0 : 1.0;
    const backwardTimePerMicrobatch = computeFwd * backwardMultiplier
      + epCommFwd * backwardEPCommMultiplier;

    // Communication times (non-EP comm; EP is folded into per-layer timing above for MoE)
    const communication = this.computeCommunication(ctx);

    // TP typically uses NVLink (within node), but cross-node TP uses hierarchical all-reduce
    let tpBandwidth: number;
    if (tp <= cluster.gpusPerNode) {
      tpBandwidth = cluster.node.intraNodeInterconnect.bandwidthGBps;
    } else {
      // Hierarchical all-reduce: RS(NVLink) + AR(IB) + AG(NVLink)
      // Total time = 2*(G-1)/G * D/nvBW + 2*(N-1)/N * D/ibBW
      // Volume already uses (tp-1)/tp ring factor, so effective BW =
      //   (tp-1)/tp / ((G-1)/G / nvBW + (N-1)/N / ibBW)
      const G = cluster.gpusPerNode;
      const N = Math.ceil(tp / G);
      const nvBW = cluster.node.intraNodeInterconnect.bandwidthGBps;
      const ibBW = cluster.interNodeBandwidthGBps;
      tpBandwidth = ((tp - 1) / tp) / ((G - 1) / G / nvBW + (N - 1) / N / ibBW);
    }
    const tpCommTime = tp > 1 ? (communication.tensorParallel / (tpBandwidth * 1e9)) * 1000 : 0;

    // PP can use NVLink or IB depending on placement.
    // Pipeline critical path is limited by the slowest link — use min() for mixed configs.
    const stagesPerNode = Math.floor(cluster.gpusPerNode / tp);
    const perNicBW = getPerNicBandwidthGBps(
      cluster.node.interNodeInterconnect, cluster.node.numNICs
    );
    const concurrentP2PStreams = Math.min(pp - 1, cluster.node.numNICs);
    const ppPerGPUBW = Math.min(perNicBW, cluster.interNodeBandwidthGBps / concurrentP2PStreams);
    const ppBandwidth = stagesPerNode >= pp
      ? cluster.node.intraNodeInterconnect.bandwidthGBps       // all intra-node
      : stagesPerNode <= 1
        ? ppPerGPUBW                                            // all cross-node
        : Math.min(                                             // mixed: cross-node bottleneck
            cluster.node.intraNodeInterconnect.bandwidthGBps,
            ppPerGPUBW
          );

    // DP bandwidth: NVLink if all DP ranks fit within a node, else IB.
    // stagesPerNode already computed for PP (each stage = tp GPUs).
    // dpRanksPerNode = how many DP ranks fit in one node after TP and PP.
    const dpRanksPerNode = stagesPerNode >= pp ? Math.floor(stagesPerNode / pp) : 0;
    const baseDPBW = dpRanksPerNode >= dp
      ? cluster.node.intraNodeInterconnect.bandwidthGBps   // all DP intra-node
      : cluster.interNodeBandwidthGBps;                      // DP crosses nodes

    // DP uses IB across nodes — bandwidth degrades at large DP group sizes.
    // When EP all-to-all crosses nodes, it shares the same IB fabric as DP
    // allreduce — model total fabric participants for congestion penalty.
    // For MoE with EP, expert params AllReduce across smaller DP/EP group
    // (less congestion), while shared params use full DP. Blend penalties
    // by volume fraction (harmonic mean gives correct total time).
    let dpCommTimePerStep = 0;
    if (dp > 1 && communication.dataParallel > 0) {
      const epCrossNode = isMoE && ep > 1 && ep * tp > cluster.gpusPerNode;
      const fabricParticipants = epCrossNode ? dp + ep : dp;
      const fullDPPenalty = getDPGroupSizePenalty(fabricParticipants);

      if (isMoE && ep > 1) {
        const expertDPGroup = Math.floor(dp / ep);
        const expertDPPenalty = getDPGroupSizePenalty(expertDPGroup);
        const moeInfo = calculateMoEMemory(model, training.dtypes.gradients, ep);
        const sharedPG = (moeInfo.sharedParams + moeInfo.routerParams) / (tp * pp);
        const expertPG = moeInfo.expertParams / (ep * tp * pp);
        const sf = (dp - 1) / dp;
        const ef = expertDPGroup > 1 ? (expertDPGroup - 1) / expertDPGroup : 0;
        const sw = sharedPG * sf;
        const ew = expertPG * ef;
        const tw = sw + ew;
        // Harmonic mean: 1 / (frac_s/P_s + frac_e/P_e) — correct because times add
        const blended = tw > 0
          ? 1 / ((sw / tw) / fullDPPenalty + (ew / tw) / expertDPPenalty)
          : fullDPPenalty;
        dpCommTimePerStep = (communication.dataParallel / (baseDPBW * blended * 1e9)) * 1000;
      } else {
        dpCommTimePerStep = (communication.dataParallel / (baseDPBW * fullDPPenalty * 1e9)) * 1000;
      }
    }

    // EP communication is modeled per-layer inside forwardTimePerMicrobatch above.
    // The step-level epCommTime is zero — it's already baked into forward/backward times.
    const epCommTime = 0;

    // Pipeline timing with bubble
    // The bubble fraction is based on numMicroBatches (pipeline fill/drain overhead)
    // But total time scales with gradientAccumulationSteps (actual micro-batches processed)
    const bubble = this.calculateBubble();
    // Layer imbalance: when numLayers doesn't divide evenly by pp, the bottleneck
    // stage has ceil(N/pp) layers while timing uses N/pp (the average). Pipeline
    // throughput is gated by the slowest stage.
    const layerImbalancePenalty = pp > 1
      ? Math.ceil(model.numLayers / pp) / (model.numLayers / pp)
      : 1;
    // Clamp: if numMicroBatches < pp, bubble > 1 which would make efficiency negative.
    // Floor at 5% efficiency (20x slowdown) — validation catches the misconfiguration.
    const pipelineEfficiency = Math.max(0.05, (1 - bubble) / layerImbalancePenalty);

    // Forward/backward times accounting for pipeline
    // Total micro-batches = gradientAccumulationSteps (NOT numMicroBatches * GA)
    // numMicroBatches only affects bubble calculation, not total work
    // Non-matmul time is added separately with correct backward multiplier (2.0 with ckpt,
    // 1.0 without) rather than using the matmul backward multiplier (2.5/2.85).
    const bwdNonMatmulMult = ctx.activationCheckpointing ? 2.0 : 1.0;
    const forwardTime = (forwardTimePerMicrobatch + nonMatmulPerMB) * gradientAccumulationSteps / pipelineEfficiency;
    const backwardTime = (backwardTimePerMicrobatch + nonMatmulPerMB * bwdNonMatmulMult) * gradientAccumulationSteps / pipelineEfficiency;
    const optimizerParamsPerGPU = ctx.lora
      ? computeLoraParamsPerRank(model, ctx.lora.rank, ctx.lora.targetModules, tp, pp)
        / (this.config.dpType !== 'ddp' ? dp : 1)
      : getOptimizerParamsPerGPU(
          model, training.dtypes.params, { tp, pp, dp, ep, dpType: this.config.dpType }
        );
    const optimizerTime = computeOptimizerStepTime(
      optimizerParamsPerGPU, gpu, training.optimizer.type, training.dtypes.params
    );

    // DP comm: FSDP/ZeRO-3 comm scales with GA (per micro-batch), others once per step
    let dpGrossTime: number;
    let dpOverlap: number;
    let dpNumCollectives: number;
    const dpTypeScalesWithGA = this.config.dpType === 'fsdp' || this.config.dpType === 'zero-3';
    const dpIntraNode = cluster.numNodes === 1;

    if (dpTypeScalesWithGA && dp > 1) {
      // Add per-collective latency for FSDP/ZeRO-3 path (per-MB collectives)
      // 3 collectives per layer per MB: AllGather(fwd) + AllGather(bwd) + ReduceScatter(bwd).
      // Real FSDP with no_sync() skips RS on non-final MBs (GA>1); modeled as full per-MB
      // communication since the savings are negligible in the compute-bound regime.
      const dpCollectivesPerMB = layersPerStage * 3;
      dpCommTimePerStep += dpCollectivesPerMB * getDPCollectiveLatencyMs(dp);
      dpNumCollectives = dpCollectivesPerMB * gradientAccumulationSteps;

      // Per-layer FSDP pipeline: decompose dpCommTimePerStep into per-layer AG and RS
      // dpCommTimePerStep = 2×AG_total (fwd+bwd) + RS_total (bwd only).
      // The overlap model treats agPerLayer as a SINGLE AllGather (applied once per
      // direction), so we split by single-AG fraction = param / (2*param + grad).
      const paramBytesPerElem = DTYPE_BYTES[training.dtypes.params];
      const gradBytesPerElem = DTYPE_BYTES[training.dtypes.gradients];
      const singleAgFraction = paramBytesPerElem / (2 * paramBytesPerElem + gradBytesPerElem);
      const rsFraction = gradBytesPerElem / (2 * paramBytesPerElem + gradBytesPerElem);
      const agPerLayer = dpCommTimePerStep * singleAgFraction / layersPerStage;
      const rsPerLayer = dpCommTimePerStep * rsFraction / layersPerStage;

      // Build per-layer compute arrays for heterogeneous MoE layers
      let fwdPerLayer: number[];
      let bwdPerLayer: number[];

      if (isMoE && (_perLayerDenseCount > 0 || moeLayersPerGPU > 0)) {
        // MoE model: dense attention layers + MoE expert layers have different compute
        // Include per-layer share of nonMatmulPerMB in each layer's compute budget
        const nonMatmulPerLayer = nonMatmulPerMB / layersPerStage;
        const bwdNonMatmulMultLayer = ctx.activationCheckpointing ? 2.0 : 1.0;
        // Ensure dense + MoE counts sum to exactly layersPerStage (fractional PP splits)
        const denseCount = Math.round(_perLayerDenseCount);
        const moeCount = layersPerStage - denseCount;
        fwdPerLayer = [];
        bwdPerLayer = [];
        for (let i = 0; i < denseCount; i++) {
          fwdPerLayer.push(_perLayerDenseFwd + nonMatmulPerLayer);
          bwdPerLayer.push(_perLayerDenseFwd * backwardMultiplier + nonMatmulPerLayer * bwdNonMatmulMultLayer);
        }
        for (let i = 0; i < moeCount; i++) {
          // MoE backward: compute scales by backwardMultiplier, EP comm by backwardEPCommMultiplier
          const moeCompute = _perLayerMoeFwd - alltoallTimePerLayer;
          fwdPerLayer.push(_perLayerMoeFwd + nonMatmulPerLayer);
          bwdPerLayer.push(moeCompute * backwardMultiplier + alltoallTimePerLayer * backwardEPCommMultiplier + nonMatmulPerLayer * bwdNonMatmulMultLayer);
        }
      } else {
        // Dense model or uniform layers: single-element closed-form
        const fwdPerLayerVal = (forwardTimePerMicrobatch + nonMatmulPerMB) / layersPerStage;
        const bwdNonMatmulMultLayer = ctx.activationCheckpointing ? 2.0 : 1.0;
        const bwdPerLayerVal = (backwardTimePerMicrobatch + nonMatmulPerMB * bwdNonMatmulMultLayer) / layersPerStage;
        fwdPerLayer = [fwdPerLayerVal];
        bwdPerLayer = [bwdPerLayerVal];
      }

      const exposedPerMB = computeFSDPExposedComm({
        fwdComputePerLayer: fwdPerLayer,
        bwdComputePerLayer: bwdPerLayer,
        allGatherPerLayer: agPerLayer,
        reduceScatterPerLayer: rsPerLayer,
        numLayers: layersPerStage,
        backwardPrefetch: true,
      });

      dpGrossTime = dpCommTimePerStep * gradientAccumulationSteps;
      dpOverlap = dpGrossTime - exposedPerMB * gradientAccumulationSteps;
    } else {
      // EP-aware per-GPU param count: expert params are distributed across EP groups
      const perGPUParamsForBuckets = isMoE && ep > 1
        ? (() => {
            const moeInfo = calculateMoEMemory(model, training.dtypes.gradients, ep);
            return (moeInfo.sharedParams + moeInfo.routerParams) / (tp * pp)
                 + moeInfo.expertParams / (ep * tp * pp);
          })()
        : model.totalParams / (tp * pp);

      dpGrossTime = dpCommTimePerStep;
      dpGrossTime += getDPCollectiveLatencyMs(dp); // Single AllReduce per step
      // DDP/ZeRO-1/2: bucketed AllReduce — numBuckets collectives
      const perGPUGradientBytesForBuckets = perGPUParamsForBuckets * DTYPE_BYTES[training.dtypes.gradients];
      dpNumCollectives = Math.max(1, Math.ceil(perGPUGradientBytesForBuckets / BUCKET_SIZE_BYTES));
      if ((this.config.dpType === 'zero-1' || this.config.dpType === 'zero-2') && dp > 1) {
        // Param AllGather after optimizer is sequential — only grad sync overlaps.
        // Compute fraction from dtype sizes: paramGather / (gradSync + paramGather)
        const paramBytesPerElem = DTYPE_BYTES[training.dtypes.params];
        const gradBytesPerElem = DTYPE_BYTES[training.dtypes.gradients];
        const paramGatherFraction = this.config.dpType === 'zero-1'
          ? paramBytesPerElem / (2 * gradBytesPerElem + paramBytesPerElem)   // AllReduce + AllGather
          : paramBytesPerElem / (gradBytesPerElem + paramBytesPerElem);      // ReduceScatter + AllGather
        const gradSyncTime = dpCommTimePerStep * (1 - paramGatherFraction);
        const perGPUGradientBytes = perGPUParamsForBuckets * DTYPE_BYTES[training.dtypes.gradients];
        dpOverlap = computeZeROGradOverlap({
          stage: this.config.dpType === 'zero-1' ? 1 : 2,
          overlapComm: true,
          gradSyncTime,
          backwardTime: backwardTimePerMicrobatch + nonMatmulPerMB * bwdNonMatmulMult,
          gradientBytes: perGPUGradientBytes,
        });
      } else {
        // DDP: AllReduce can overlap with backward (bucketed timeline model)
        const perGPUGradientBytes = perGPUParamsForBuckets * DTYPE_BYTES[training.dtypes.gradients];
        dpOverlap = computeDDPOverlap({
          commTime: dpCommTimePerStep,
          backwardTime: backwardTimePerMicrobatch + nonMatmulPerMB * bwdNonMatmulMult,
          gradientBytes: perGPUGradientBytes,
        });
      }
    }

    // TP comm is per-microbatch: all-reduces happen layer-by-layer within each
    // microbatch's forward/backward, pipelined with compute. The overlap efficiency
    // depends on the compute-to-comm ratio (large models → high overlap, most TP
    // comm hidden behind next layer's compute; small models → less overlap).
    // TP comm volume from computeCommunication() covers all numLayers, but each PP stage
    // only processes numLayers/pp layers. Scale to per-stage for the layer-level overlap model.
    const crossNodeTP = tp > cluster.gpusPerNode;
    // Dense: 2 TP collectives/layer (attention + MLP), MoE with EP>1: 1 (attention only)
    const moeLayersPerStageTP = (isMoE && ep > 1)
      ? Math.ceil((model.numMoELayers ?? model.numLayers) / pp)
      : 0;
    const denseLayersPerStageTP = layersPerStage - moeLayersPerStageTP;
    const tpCollectivesPerStage = 2 * denseLayersPerStageTP + 1 * moeLayersPerStageTP;
    const tpCommWithOverhead = applyProtocolOverhead(tpCommTime / pp, crossNodeTP ? 'tp_crossnode' : 'tp_nvlink', tpCollectivesPerStage, !crossNodeTP);
    const computePerMB = forwardTimePerMicrobatch + backwardTimePerMicrobatch;
    const tpOverlapEff = computeTPOverlap({
      computePerMB,
      tpCommWithOverhead,
    });
    // Exposed TP per step = non-overlapped residual per microbatch × GA
    const tpExposedPerStep = tpCommWithOverhead * (1 - tpOverlapEff) * gradientAccumulationSteps;

    // Context Parallel timing model — two implementations:
    // - Ring attention: per-step P2P pipeline where KV transfer overlaps with chunk attention.
    // - All-gather (Megatron-LM): AllGather KV before attention, fully exposed.
    let cpExposedPerStep = 0;
    if (cp > 1) {
      const chunkSeq = seqLength / cp;

      // KV block size — shared by both implementations
      const kvDim = model.attentionType === 'mla'
        ? (model.kvLoraRank! + model.qkRopeHeadDim!)
        : model.numKvHeads * model.headDim * 2;
      // BF16 floor: CP comm sends KV at BF16 minimum (same convention
      // as PP comm and computeCommunication DP/TP paths).
      const cpCommBytes = Math.max(DTYPE_BYTES[training.dtypes.activation], DTYPE_BYTES['bf16']);
      const kvBlockBytes = kvDim * chunkSeq * microBatchSize * cpCommBytes;

      // CP placement and bandwidth
      const cpWithinNode = tp * cp <= cluster.gpusPerNode;
      const cpBandwidth = cpWithinNode
        ? cluster.node.intraNodeInterconnect.bandwidthGBps
        : cluster.interNodeBandwidthGBps / cluster.gpusPerNode;

      const fwdPasses = ctx.activationCheckpointing ? 2 : 1;
      const layersPerGPU = model.numLayers / pp;

      if (this.config.cpImplementation === 'all-gather') {
        // All-gather CP (Meta, Megatron-LM): AllGather KV before attention, fully exposed.
        // Forward: AllGather(KV) per layer. Backward: local dK/dV for own chunk (no ReduceScatter).
        // With AC: +1 AllGather for recompute.
        const allGatherVolume = kvBlockBytes * (cp - 1);
        const cpCollectiveEff = cpWithinNode ? 0.90 : 0.82;  // NCCL ring all-gather
        const allGatherTimeMs = allGatherVolume / (cpBandwidth * cpCollectiveEff * 1e9) * 1000;

        // Partial overlap: with interleaved PP, AllGather overlaps with non-attention
        // compute from previous microbatch (layernorm, projections). At PP=1 or
        // non-interleaved, overlap is minimal — only CUDA scheduling slack.
        const isInterleaved = pp > 1 && (this.config.schedule === 'interleaved-1f1b' || this.config.schedule === 'dualpipe-v');
        const cpAllGatherOverlap = isInterleaved ? _cpAllGatherOverlapInterleaved : 0.05;
        const exposedAllGatherTime = allGatherTimeMs * (1 - cpAllGatherOverlap);

        // fwdPasses AllGathers only — backward dK/dV is local (no ReduceScatter needed)
        const commPerLayerPerMB = fwdPasses * exposedAllGatherTime;
        cpExposedPerStep = commPerLayerPerMB * layersPerGPU * gradientAccumulationSteps;
      } else {
        // Ring attention: per-step P2P pipeline within the attention layer. At each step,
        // KV P2P transfer overlaps with blockwise attention compute (QK^T + softmax +
        // scores×V) on one KV chunk. Step 0 uses local KV (no transfer), so only CP-1
        // transfers overlap.
        const cpBWEfficiency = cpWithinNode ? 0.90 : 0.85;  // P2P higher BW than collectives
        const transferTime = kvBlockBytes / (cpBandwidth * cpBWEfficiency * 1e9) * 1000;

        // Per-chunk attention FLOPs: QK^T + scores×V (what overlaps with KV transfer)
        const qkDim = model.attentionType === 'mla'
          ? (model.qkNopeHeadDim! + model.qkRopeHeadDim!)
          : model.headDim;
        const vDim = model.attentionType === 'mla' ? model.vHeadDim! : model.headDim;
        const attnFLOPsPerChunk = 2 * model.numAttentionHeads * chunkSeq * chunkSeq
          * (qkDim + vDim) * microBatchSize / tp;
        const computeTimeFwd = attnFLOPsPerChunk / (effectiveTFLOPS * 1e12) * 1000;
        const computeTimeBwd = computeTimeFwd * 2;  // backward attention ~2× FLOPs

        const { diagonalComputeFraction, diagonalSteps, normalSteps } = cpCausalWorkDistribution(cp);

        // Scheduling jitter scales with CP degree: more peers = more coordination.
        // CP=2 (robust overlap, C/T=3-5×): 3.5%. CP=16 (marginal, fragile): 10.5%.
        const jitterFloor = 0.03 + 0.005 * (cp - 1);

        // Per-step exposed transfer: diagonal step has reduced compute (causal mask),
        // normal steps have full cross-attention compute.
        const exposedDiagonalFwd = Math.max(transferTime * jitterFloor,
          transferTime - computeTimeFwd * diagonalComputeFraction);
        const exposedNormalFwd = Math.max(transferTime * jitterFloor,
          transferTime - computeTimeFwd);
        const exposedFwd = diagonalSteps * exposedDiagonalFwd + normalSteps * exposedNormalFwd;

        const exposedDiagonalBwd = Math.max(transferTime * jitterFloor,
          transferTime - computeTimeBwd * diagonalComputeFraction);
        const exposedNormalBwd = Math.max(transferTime * jitterFloor,
          transferTime - computeTimeBwd);
        const exposedBwd = diagonalSteps * exposedDiagonalBwd + normalSteps * exposedNormalBwd;

        const exposedPerLayerPerMB = fwdPasses * exposedFwd + exposedBwd;
        cpExposedPerStep = exposedPerLayerPerMB * layersPerGPU * gradientAccumulationSteps;
      }
    }

    // PP wall-clock from first principles (matching TP/CP exposed pattern).
    // Per-MB: each GPU does 2 P2P transfers (fwd activation + bwd gradient).
    // Interleaved: 2×v (one pair per virtual stage boundary).
    // DualPipeV: v=2 — during F&B blocks, chunk 0 sends fwd right + chunk 1 sends grad right
    // simultaneously, competing for the same link → 2× per-GPU P2P wall clock.
    let ppExposedPerStep = 0;
    if (pp > 1 && gradientAccumulationSteps > 0) {
      const commActivationBytes = Math.max(DTYPE_BYTES[training.dtypes.activation], DTYPE_BYTES['bf16']);
      const activationBytes = effectiveSeqLength * microBatchSize * model.hiddenSize * commActivationBytes;
      const ppCommPerTransferMs = (activationBytes / (ppBandwidth * 1e9)) * 1000;

      const v = this.config.schedule === 'interleaved-1f1b' ? this.config.interleavedStages
              : this.config.schedule === 'dualpipe-v' ? 2
              : 1;
      const ppCommPerMB = 2 * v * ppCommPerTransferMs;

      const ppOverlapEff = computePPOverlap({ computePerMB, ppCommPerMB });
      // Per-virtual-stage transition overhead (NCCL P2P setup + stream sync + kernel launch ≈ 20µs).
      // Not overlappable — sequential sync on critical path.
      // Interleaved-1F1B: each of v virtual stage boundaries is a serial transition.
      // DualPipe-V: fwd/bwd directions proceed concurrently from both ends, so serial
      // transitions = 2×(pp-1) — the V-shape's bandwidth doubling is already in ppCommPerMB.
      const ppTransitionsPerMB = this.config.schedule === 'dualpipe-v'
        ? 2 * (pp - 1)
        : 2 * v * (pp - 1);
      const ppTransitionOverheadMs = ppTransitionsPerMB * _ppStageTransitionMs;
      ppExposedPerStep = ppCommPerMB * (1 - ppOverlapEff) * gradientAccumulationSteps
                       + ppTransitionOverheadMs * gradientAccumulationSteps;
    }

    const dpGrossWithOverhead = applyProtocolOverhead(
      dpGrossTime, dpTypeScalesWithGA ? 'dp_fsdp' : 'dp_ddp', dpNumCollectives, dpIntraNode,
    );
    const epGrossWithOverhead = applyProtocolOverhead(epCommTime, 'dp_ep');

    const totalCommTime = tpExposedPerStep            // TP: layer-overlap already applied
      + ppExposedPerStep                               // PP: per-GPU wall-clock, overlap already applied
      + dpGrossWithOverhead                            // DP: proportional + per-collective
      + epGrossWithOverhead                            // All-to-All: moderate (epCommTime=0, no per-collective needed)
      + cpExposedPerStep;                              // CP: overhead baked into jitter floor (ring) or collective eff (all-gather)

    // Clamp: overlap can't exceed total comm (would make step time negative)
    const overlap = Math.min(totalCommTime, dpOverlap);

    // EP comm is embedded in forward/backward times — track exposed portion for reporting.
    // Forward: 1× EP comm. Backward: 2× with AC (recompute + gradient), 1× without AC.
    // Physics floor slack overlap ratio from forward carries over proportionally.
    const epCommPerStep = isMoE
      ? epCommExposedFwd * (1 + backwardEPCommMultiplier) * gradientAccumulationSteps / pipelineEfficiency
      : 0;

    const total = forwardTime + backwardTime + optimizerTime + totalCommTime - overlap;
    if (total <= 0) throw new Error(`Step time must be positive, got ${total}ms`);

    // Forward/backward sub-breakdown (all models)
    const fwdScale = gradientAccumulationSteps / pipelineEfficiency;
    const _forwardComputeMs = forwardTimePerMicrobatch * fwdScale;
    const _forwardNonMatmulMs = nonMatmulPerMB * fwdScale;
    const _backwardComputeMs = backwardTimePerMicrobatch * fwdScale;
    const _backwardNonMatmulMs = nonMatmulPerMB * bwdNonMatmulMult * fwdScale;

    // MoE per-pass decomposition (replaces old full-step fields)
    let moeExpertFwdMs: number | undefined;
    let moeDenseFwdMs: number | undefined;
    let moeOverheadFwdMs: number | undefined;
    let epCommFwdMs: number | undefined;
    let moeExpertBwdMs: number | undefined;
    let moeDenseBwdMs: number | undefined;
    let moeOverheadBwdMs: number | undefined;
    let epCommBwdMs: number | undefined;
    if (isMoE) {
      const bwdScale = backwardMultiplier * fwdScale;
      moeExpertFwdMs = _diag_expertComputePerMB * fwdScale;
      moeExpertBwdMs = _diag_expertComputePerMB * bwdScale;
      moeDenseFwdMs = _diag_denseComputePerMB * fwdScale + nonMatmulPerMB * fwdScale;
      moeDenseBwdMs = _diag_denseComputePerMB * bwdScale + nonMatmulPerMB * bwdNonMatmulMult * fwdScale;
      moeOverheadFwdMs = _diag_overheadPerMB * fwdScale;
      moeOverheadBwdMs = _diag_overheadPerMB * bwdScale;
      epCommFwdMs = epCommExposedFwd * fwdScale;
      epCommBwdMs = epCommExposedFwd * backwardEPCommMultiplier * fwdScale;
    }

    return {
      forward: forwardTime,
      backward: backwardTime,
      optimizer: optimizerTime,
      communication: totalCommTime,
      epCommunication: epCommPerStep,
      overlap,
      scaleOverhead: 0,
      total,
      forwardComputeMs: _forwardComputeMs,
      forwardNonMatmulMs: _forwardNonMatmulMs,
      backwardComputeMs: _backwardComputeMs,
      backwardNonMatmulMs: _backwardNonMatmulMs,
      tpExposed: tpExposedPerStep,
      ppExposed: ppExposedPerStep,
      dpGross: dpGrossWithOverhead,
      cpExposed: cpExposedPerStep,
      pipelineBubbleFraction: 1 - (1 - bubble) / layerImbalancePenalty,
      moeExpertFwdMs,
      moeDenseFwdMs,
      moeOverheadFwdMs,
      epCommFwdMs,
      moeExpertBwdMs,
      moeDenseBwdMs,
      moeOverheadBwdMs,
      epCommBwdMs,
      groupedGemmFactor: isMoE ? _diag_groupedGemmFactor : undefined,
      routedFlopFraction: isMoE ? _diag_routedFlopFraction : undefined,
    };
  }

  validate(ctx: StrategyContext): StrategyValidation {
    const errors: string[] = [];
    const warnings: string[] = [];
    const { tp, pp, dp, ep, cp } = this.config;

    // Check total GPU count (TP × PP × CP × DP = totalGPUs; EP subdivides DP)
    const requiredGPUs = tp * pp * cp * dp;
    if (requiredGPUs !== ctx.cluster.totalGPUs) {
      errors.push(
        cp > 1
          ? `3D config (TP=${tp} × PP=${pp} × CP=${cp} × DP=${dp} = ${requiredGPUs}) doesn't match cluster size (${ctx.cluster.totalGPUs} GPUs).`
          : `3D config (TP=${tp} × PP=${pp} × DP=${dp} = ${requiredGPUs}) doesn't match cluster size (${ctx.cluster.totalGPUs} GPUs).`
      );
    }

    // CP validation
    if (cp > 1) {
      if (ctx.seqLength % cp !== 0) {
        errors.push(`Sequence length (${ctx.seqLength}) must be divisible by CP degree (${cp}).`);
      }
      if (ctx.seqLength / cp < 1024) {
        warnings.push(`Sequence chunk (${ctx.seqLength / cp} tokens) is very small — CP overhead may dominate.`);
      }
    }

    // EP validation
    if (ep > 1 && ep > dp) {
      errors.push(`EP=${ep} exceeds DP=${dp}. EP must be ≤ DP since it subdivides the data parallel dimension.`);
    }
    if (ep > 1 && dp % ep !== 0) {
      errors.push(`EP=${ep} must divide DP degree (${dp}).`);
    }
    if (ep > 1 && !ctx.model.isMoE) {
      errors.push(`EP=${ep} is set but model is not a Mixture of Experts.`);
    }
    if (ep > 1 && ctx.model.numExperts && ctx.model.numExperts % ep !== 0) {
      errors.push(`EP=${ep} must divide number of experts (${ctx.model.numExperts}).`);
    }
    if (ep > 1 && ctx.model.numExperts && ep > ctx.model.numExperts) {
      errors.push(`EP=${ep} exceeds numExperts (${ctx.model.numExperts}). Cannot have more EP ranks than experts.`);
    }
    if (ep > 1 && ep * tp > ctx.cluster.gpusPerNode) {
      warnings.push(`EP × TP (${ep} × ${tp} = ${ep * tp}) exceeds GPUs per node (${ctx.cluster.gpusPerNode}). EP All-to-All will use slower inter-node links.`);
    }

    // TP constraints
    if (tp > 1) {
      if (ctx.model.numAttentionHeads % tp !== 0) {
        errors.push(`TP=${tp} must divide attention heads (${ctx.model.numAttentionHeads}).`);
      }
      if (tp > ctx.cluster.gpusPerNode) {
        warnings.push(`TP=${tp} exceeds GPUs per node (${ctx.cluster.gpusPerNode}). Cross-node TP will use slower inter-node interconnect instead of NVLink.`);
        if (tp % ctx.cluster.gpusPerNode !== 0) {
          warnings.push(
            `TP=${tp} doesn't evenly divide GPUs per node (${ctx.cluster.gpusPerNode}). ` +
            `Cross-node TP performance estimate may be inaccurate.`
          );
        }
      }
    }

    // PP constraints
    if (pp > ctx.model.numLayers) {
      errors.push(`PP=${pp} exceeds model layers (${ctx.model.numLayers}). Cannot have more pipeline stages than layers.`);
    } else if (pp > 1 && ctx.model.numLayers % pp !== 0) {
      warnings.push(`Model layers (${ctx.model.numLayers}) don't divide evenly by PP=${pp}.`);
    }

    // DualPipeV constraints
    if (this.config.schedule === 'dualpipe-v' && pp > 1) {
      if (this.config.numMicroBatches < 2 * pp) {
        warnings.push(
          `DualPipeV requires at least 2×PP micro-batches (${2 * pp}), but only ${this.config.numMicroBatches} available. Bubble reduction is degraded.`
        );
      }
    }

    // Interleaved schedule constraints
    if (this.config.schedule === 'interleaved-1f1b' && pp > 1) {
      const v = this.config.interleavedStages;
      if (v < 2) {
        warnings.push(`Interleaved 1F1B requires at least 2 virtual stages (currently ${v}).`);
      }
      if (ctx.model.numLayers % (pp * v) !== 0) {
        warnings.push(`Model layers (${ctx.model.numLayers}) not divisible by PP×v (${pp}×${v}=${pp * v}). Some stages will have uneven layer counts.`);
      }
    }

    // Memory check
    const memory = this.computeMemoryPerGPU(ctx);
    const gpuMemoryBytes = gpuCapacityBytes(ctx.cluster.node.gpu.memoryGB);

    if (memory.total > gpuMemoryBytes) {
      const gpuMemGB = ctx.cluster.node.gpu.memoryGB;
      const reqGB = (memory.total / (1024 ** 3)).toFixed(2);
      errors.push(
        `OOM: requires ${reqGB} GB per GPU but only ${gpuMemGB} GB available.`
      );
      // OOM suggestions handled by unified recommendation engine (generateRecommendations)
    }

    // Pipeline efficiency
    if (pp > 1 && this.config.numMicroBatches < pp) {
      errors.push(
        `Pipeline cannot fill — not enough gradient accumulation steps for the number of pipeline stages. ` +
        `Increase global batch size or reduce pipeline stages.`
      );
    }
    const bubble = this.calculateBubble();
    if (bubble > 0.2) {
      warnings.push(
        `Pipeline bubble is high. ` +
        `Increase global batch size or reduce pipeline stages.`
      );
    }

    // Process group placement recommendations
    if (tp > 1 && pp > 1) {
      const tpPPGroupSize = tp * pp;
      if (tpPPGroupSize > ctx.cluster.gpusPerNode) {
        warnings.push(
          'TP x PP group spans multiple nodes. Ensure PP stages are placed to minimize cross-node communication.'
        );
      }
    }

    // Validate hybrid parallelism combinations (ZeRO-2/3 + PP, etc.)
    const hybridWarning = validateHybridCombination({
      dpType: this.config.dpType as DPType,
      tp,
      pp,
      gpusPerNode: ctx.cluster.gpusPerNode,
    });
    if (hybridWarning) {
      warnings.push(hybridWarning.message);
    }

    // Validate TP topology (cross-node TP warning)
    const tpTopologyWarning = validateTPTopology(tp, ctx.cluster.gpusPerNode);
    if (tpTopologyWarning) {
      warnings.push(tpTopologyWarning.message);
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings,
      suggestions: [],
    };
  }

  generateEvents(ctx: StrategyContext): SimulationEvent[] {
    const events: SimulationEvent[] = [];
    const timing = this.computeTiming(ctx);
    const { cluster, model } = ctx;
    const { tp, pp, dp, ep, cp, numMicroBatches } = this.config;

    events.push({
      id: 'sim-start',
      type: 'simulation-start',
      category: 'simulation',
      timestamp: 0,
      duration: 0,
      gpuId: -1,
      config: {
        totalGPUs: cluster.totalGPUs,
        totalSteps: 1,
        modelName: model.name,
        strategyName: `${this.name} (TP=${tp}, PP=${pp}, DP=${dp}${ep > 1 ? `, EP=${ep}` : ''}${cp > 1 ? `, CP=${cp}` : ''})`,
      },
    });

    // Generate simplified events showing the parallel structure
    // Cap to 1 DP rank — all DP ranks are symmetric, no need to emit millions of events
    const microBatchTime = timing.forward / numMicroBatches;
    const maxMBEvents = Math.min(numMicroBatches, 32); // Cap micro-batches too for very large pipelines
    let maxTimestamp = 0;

    const dpSample = Math.min(dp, 1);
    for (let dpRank = 0; dpRank < dpSample; dpRank++) {
      for (let ppStage = 0; ppStage < pp; ppStage++) {
        for (let tpRank = 0; tpRank < tp; tpRank++) {
          const gpuId = dpRank * pp * tp + ppStage * tp + tpRank;
          let gpuTimestamp = ppStage * microBatchTime; // Stagger by PP stage

          // Forward phases
          for (let mb = 0; mb < maxMBEvents; mb++) {
            events.push({
              id: `gpu${gpuId}-fwd-${mb}`,
              type: 'phase-start',
              category: 'phase',
              timestamp: gpuTimestamp,
              duration: microBatchTime,
              gpuId,
              phase: 'forward',
              stepNumber: 0,
              microBatchId: mb,
            });
            gpuTimestamp += microBatchTime * 1.1; // Small gap
          }

          // Backward phases
          for (let mb = maxMBEvents - 1; mb >= 0; mb--) {
            events.push({
              id: `gpu${gpuId}-bwd-${mb}`,
              type: 'phase-start',
              category: 'phase',
              timestamp: gpuTimestamp,
              duration: microBatchTime * 2,
              gpuId,
              phase: 'backward',
              stepNumber: 0,
              microBatchId: mb,
            });
            gpuTimestamp += microBatchTime * 2.1;
          }

          // DP gradient sync (at end)
          if (dp > 1) {
            events.push({
              id: `gpu${gpuId}-dp-sync`,
              type: 'collective-start',
              category: 'collective',
              timestamp: gpuTimestamp,
              duration: timing.communication * 0.3,
              gpuId,
              operation: this.config.dpType === 'ddp' ? 'all-reduce' : 'reduce-scatter',
              sizeBytes: ctx.model.totalParams * 2 / (tp * pp),
              numRanks: dp,
              algorithm: 'ring',
              isIntraNode: false,
            });
            gpuTimestamp += timing.communication * 0.3;
          }

          // Optimizer
          events.push({
            id: `gpu${gpuId}-opt`,
            type: 'phase-start',
            category: 'phase',
            timestamp: gpuTimestamp,
            duration: timing.optimizer,
            gpuId,
            phase: 'optimizer',
            stepNumber: 0,
          });

          maxTimestamp = Math.max(maxTimestamp, gpuTimestamp + timing.optimizer);
        }
      }
    }

    events.push({
      id: 'sim-end',
      type: 'simulation-end',
      category: 'simulation',
      timestamp: maxTimestamp,
      duration: 0,
      gpuId: -1,
      metrics: {
        totalTimeMs: timing.total,
        avgStepTimeMs: timing.total,
        tokensPerSecond: (ctx.globalBatchSize * ctx.seqLength) / (timing.total / 1000),
        mfu: this.computeAnalysis(ctx).mfu,
      },
    });

    return events;
  }

  computeAnalysis(ctx: StrategyContext) {
    const analysis = super.computeAnalysis(ctx);
    const bubble = this.calculateBubble();
    const pp = this.config.pp;
    const imbalance = pp > 1
      ? Math.ceil(ctx.model.numLayers / pp) / (ctx.model.numLayers / pp)
      : 1;
    analysis.pipelineBubble = 1 - (1 - bubble) / imbalance;
    return analysis;
  }
}

// Factory function
export function create3DParallelStrategy(
  tp: number,
  pp: number,
  dp: number,
  config?: Partial<ThreeDParallelConfig>
): ThreeDParallelStrategy {
  const numMicroBatches = config?.numMicroBatches ?? Math.max(8, pp * 2);
  const ep = config?.ep ?? 1;
  const cp = config?.cp ?? 1;
  return new ThreeDParallelStrategy({ ...config, tp, pp, dp, ep, cp, numMicroBatches });
}

// Auto-configure 3D parallel based on model and cluster
export function autoConfig3DParallel(
  modelParams: number,
  gpuMemoryGB: number,
  totalGPUs: number,
  gpusPerNode: number
): ThreeDParallelConfig {
  // Estimate memory requirement
  const estimatedMemoryPerGPU = modelParams * 18; // Rough estimate: 18 bytes/param for training

  // Start with max TP that fits in a node
  let tp = 1;
  let pp = 1;
  let dp = totalGPUs;

  // If model is too large for single GPU, increase TP
  if (estimatedMemoryPerGPU / (gpuCapacityBytes(gpuMemoryGB)) > 0.8) {
    tp = Math.min(gpusPerNode, 8);
  }

  // If still too large, add PP
  if (estimatedMemoryPerGPU / tp / (gpuCapacityBytes(gpuMemoryGB)) > 0.8) {
    pp = Math.min(Math.ceil(estimatedMemoryPerGPU / tp / (gpuCapacityBytes(gpuMemoryGB)) / 0.5), totalGPUs / tp);
  }

  // Remaining GPUs go to DP
  dp = Math.floor(totalGPUs / (tp * pp));

  // Ensure product matches total
  while (tp * pp * dp !== totalGPUs && dp > 1) {
    dp--;
  }

  return {
    tp,
    pp,
    dp,
    ep: 1,
    cp: 1,
    cpImplementation: 'ring',
    numMicroBatches: Math.max(8, pp * 2),
    schedule: pp > 4 ? 'interleaved-1f1b' : '1f1b',
    interleavedStages: pp > 4 ? 2 : 1,
    sequenceParallel: tp > 1,
    dpType: dp > 4 ? 'fsdp' : 'ddp',
    activationCheckpointing: true,
    checkpointingGranularity: 'full',
  };
}
