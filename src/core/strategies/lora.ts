/**
 * LoRA/QLoRA Fine-tuning Support
 *
 * LoRA (Low-Rank Adaptation) freezes base model weights and adds small trainable
 * low-rank adapter matrices to selected layers. QLoRA further compresses base
 * weights to NF4 (4-bit) to reduce parameter memory.
 *
 * Key design:
 * - Forward pass runs through full model (frozen weights + adapters)
 * - Backward skips weight gradients for frozen params (only activation grads + adapter grads)
 * - Memory: base weights (frozen) + small adapter weights/grads/optimizer
 * - Communication: only adapter gradients need AllReduce
 */
// See docs/PHYSICS.md and docs/STRATEGIES.md for formula derivations and calibration anchors.

import type { ModelSpec, GPUSpec } from '../../types/index.ts';
import { getSelectiveRecomputeFraction } from './base.ts';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type FinetuningMethod = 'full' | 'lora' | 'qlora';
export type LoraTargetModules = 'q_v' | 'q_k_v_o' | 'all_linear';

export interface LoraConfig {
  method: FinetuningMethod;
  rank: number;               // r: 4, 8, 16, 32, 64
  targetModules: LoraTargetModules;
}

// ---------------------------------------------------------------------------
// Trainable parameter calculation
// ---------------------------------------------------------------------------

/**
 * Compute global LoRA trainable params (for MFU numerator and test validation).
 *
 * Per-layer adapter params depend on target modules:
 * - q_v:        Q + V adapters
 * - q_k_v_o:    Q + K + V + O adapters
 * - all_linear: Q + K + V + O + MLP adapters (dense + shared expert only, NOT routed experts)
 *
 * For GQA models, K/V dimensions use numKvHeads (smaller than Q/O).
 * For MLA models (DeepSeek V2/V3/R1), use kvLoraRank / qLoraRank.
 */
export function computeLoraTrainableParams(
  model: ModelSpec,
  rank: number,
  targetModules: LoraTargetModules,
): number {
  const d = model.hiddenSize;

  // Attention projection dimensions (GQA-aware)
  let d_q: number, d_k: number, d_v: number;
  if (model.attentionType === 'mla' && model.kvLoraRank != null) {
    // MLA: compressed latent dimensions
    d_k = model.kvLoraRank;
    d_v = model.kvLoraRank;
    d_q = model.qLoraRank ?? (model.numAttentionHeads * model.headDim);
  } else {
    d_q = model.numAttentionHeads * model.headDim;
    d_k = model.numKvHeads * model.headDim;
    d_v = model.numKvHeads * model.headDim;
  }

  // O projection input = attention output = numAttentionHeads × vHeadDim (MLA) or headDim
  const d_o_in = model.attentionType === 'mla' && model.vHeadDim
    ? model.numAttentionHeads * model.vHeadDim
    : model.numAttentionHeads * model.headDim;

  // Per-layer attention adapter params
  let perLayerAttnParams: number;
  switch (targetModules) {
    case 'q_v':
      // A[d,r] + B[r,d_q] for Q, A[d,r] + B[r,d_v] for V
      perLayerAttnParams = rank * (d + d_q) + rank * (d + d_v);
      break;
    case 'q_k_v_o':
    case 'all_linear':
      // Q(col) + K(col) + V(col) + O(row)
      perLayerAttnParams =
        rank * (d + d_q) +   // Q
        rank * (d + d_k) +   // K
        rank * (d + d_v) +   // V
        rank * (d_o_in + d); // O (row-parallel: input=nH×headDim, output=d)
      break;
  }

  // Per-layer MLP adapter params (only for all_linear)
  let perLayerDenseMlpParams = 0;
  if (targetModules === 'all_linear') {
    const I = model.intermediateSize;
    if (model.gatedMLP) {
      // GatedMLP (SwiGLU): gate(col) + up(col) + down(row)
      perLayerDenseMlpParams =
        rank * (d + I) +   // gate (column-parallel)
        rank * (d + I) +   // up (column-parallel)
        rank * (I + d);    // down (row-parallel)
    } else {
      // Standard MLP: up(col) + down(row)
      perLayerDenseMlpParams =
        rank * (d + I) +   // up
        rank * (I + d);    // down
    }
  }

  // MoE handling: all_linear adapts dense layers + shared expert MLPs, NOT routed experts
  const numMoELayers = model.numMoELayers ?? 0;
  const numDenseLayers = model.numLayers - numMoELayers;

  // Dense layers: attention + dense MLP
  let totalParams = numDenseLayers * (perLayerAttnParams + perLayerDenseMlpParams);

  // MoE layers: attention + shared expert MLP (not routed experts)
  if (numMoELayers > 0) {
    let perMoeLayerSharedMlpParams = 0;
    if (targetModules === 'all_linear' && model.numSharedExperts && model.numSharedExperts > 0) {
      const sI = model.sharedExpertIntermediateSize ?? model.expertIntermediateSize ?? model.intermediateSize;
      const perSharedExpertMlp = model.gatedMLP
        ? rank * (d + sI) + rank * (d + sI) + rank * (sI + d)
        : rank * (d + sI) + rank * (sI + d);
      perMoeLayerSharedMlpParams = model.numSharedExperts * perSharedExpertMlp;
    }
    totalParams += numMoELayers * (perLayerAttnParams + perMoeLayerSharedMlpParams);
  }

  // If there are no MoE layers, all layers are dense
  if (numMoELayers === 0) {
    totalParams = model.numLayers * (perLayerAttnParams + perLayerDenseMlpParams);
  }

  return totalParams;
}

/**
 * Compute per-GPU LoRA trainable params accounting for TP sharding and PP layer distribution.
 *
 * In Megatron-style TP:
 * - Column-parallel (Q, K, V, gate, up): A [d, r] replicated, B [r, out/tp] sharded
 *   → per rank: r * (d + out/tp)
 * - Row-parallel (O, down): A [in/tp, r] sharded, B [r, d] replicated
 *   → per rank: r * (in/tp + d)
 */
export function computeLoraParamsPerRank(
  model: ModelSpec,
  rank: number,
  targetModules: LoraTargetModules,
  tp: number,
  pp: number,
): number {
  const d = model.hiddenSize;

  // Attention dimensions (GQA-aware)
  let d_q: number, d_k: number, d_v: number;
  if (model.attentionType === 'mla' && model.kvLoraRank != null) {
    d_k = model.kvLoraRank;
    d_v = model.kvLoraRank;
    d_q = model.qLoraRank ?? (model.numAttentionHeads * model.headDim);
  } else {
    d_q = model.numAttentionHeads * model.headDim;
    d_k = model.numKvHeads * model.headDim;
    d_v = model.numKvHeads * model.headDim;
  }

  // O projection input = attention output = numAttentionHeads × vHeadDim (MLA) or headDim
  const d_o_in = model.attentionType === 'mla' && model.vHeadDim
    ? model.numAttentionHeads * model.vHeadDim
    : model.numAttentionHeads * model.headDim;

  // Per-layer per-rank attention adapter params
  let perLayerAttnPerRank: number;
  switch (targetModules) {
    case 'q_v':
      // Q(col): A replicated, B sharded → r*(d + d_q/tp)
      // V(col): A replicated, B sharded → r*(d + d_v/tp)
      perLayerAttnPerRank =
        rank * (d + d_q / tp) +
        rank * (d + d_v / tp);
      break;
    case 'q_k_v_o':
    case 'all_linear':
      // Q(col) + K(col) + V(col): output dim sharded
      // O(row): input dim sharded
      perLayerAttnPerRank =
        rank * (d + d_q / tp) +     // Q (column-parallel)
        rank * (d + d_k / tp) +     // K (column-parallel)
        rank * (d + d_v / tp) +     // V (column-parallel)
        rank * (d_o_in / tp + d);   // O (row-parallel: nH×headDim sharded by TP)
      break;
  }

  // Per-layer per-rank MLP adapter params
  let perLayerDenseMlpPerRank = 0;
  if (targetModules === 'all_linear') {
    const I = model.intermediateSize;
    if (model.gatedMLP) {
      perLayerDenseMlpPerRank =
        rank * (d + I / tp) +   // gate (column-parallel)
        rank * (d + I / tp) +   // up (column-parallel)
        rank * (I / tp + d);    // down (row-parallel)
    } else {
      perLayerDenseMlpPerRank =
        rank * (d + I / tp) +   // up
        rank * (I / tp + d);    // down
    }
  }

  // MoE: shared expert MLP per rank
  const numMoELayers = model.numMoELayers ?? 0;
  const numDenseLayers = model.numLayers - numMoELayers;

  const perLayerDense = perLayerAttnPerRank + perLayerDenseMlpPerRank;
  let perLayerMoE = perLayerAttnPerRank; // MoE layers: attention only from dense part

  if (targetModules === 'all_linear' && numMoELayers > 0 && model.numSharedExperts && model.numSharedExperts > 0) {
    const sI = model.sharedExpertIntermediateSize ?? model.expertIntermediateSize ?? model.intermediateSize;
    const perSharedExpertMlpPerRank = model.gatedMLP
      ? rank * (d + sI / tp) + rank * (d + sI / tp) + rank * (sI / tp + d)
      : rank * (d + sI / tp) + rank * (sI / tp + d);
    perLayerMoE += model.numSharedExperts * perSharedExpertMlpPerRank;
  }

  // Total params across layers for this GPU's PP stage
  const layersPerStage = Math.ceil(model.numLayers / pp);

  // Approximate distribution: layers are assigned to stages sequentially
  // For simplicity, use proportional distribution
  let totalPerRank: number;
  if (numMoELayers === 0) {
    totalPerRank = layersPerStage * perLayerDense;
  } else {
    // Proportionally split dense and MoE layers across PP stages
    const denseFraction = numDenseLayers / model.numLayers;
    const moeFraction = numMoELayers / model.numLayers;
    totalPerRank = layersPerStage * (denseFraction * perLayerDense + moeFraction * perLayerMoE);
  }

  return totalPerRank;
}

// ---------------------------------------------------------------------------
// QLoRA dequantization time
// ---------------------------------------------------------------------------

/**
 * QLoRA dequant is bandwidth-bound (not FLOP-bound).
 * Per layer: reads 0.5 bytes (NF4) + writes 2 bytes (BF16) = 2.5 bytes per param.
 * Added to forward time per microbatch (happens on critical path).
 *
 * 7B: ~0.15ms/layer × 32 = 4.8ms total (negligible)
 * 70B: ~1.3ms/layer × 80/pp (significant at pp=1)
 */
export function getQloraDequantTimeMs(
  model: ModelSpec,
  gpu: GPUSpec,
  tp: number,
  pp: number,
): number {
  const paramsPerLayer = model.totalParams / model.numLayers;
  const paramsPerLayerPerGPU = paramsPerLayer / tp;
  const layersPerStage = Math.ceil(model.numLayers / pp);
  const dequantTimePerLayer = paramsPerLayerPerGPU * 2.5 / (gpu.memoryBandwidthTBps * 1e12) * 1000;
  return dequantTimePerLayer * layersPerStage;
}

// ---------------------------------------------------------------------------
// LoRA backward FLOPs
// ---------------------------------------------------------------------------

/**
 * Compute LoRA backward FLOPs. Decomposes the backward pass:
 *
 * Standard training:
 * - Without ckpt: backward = 2× forward (actgrads + wgrads)
 * - With ckpt: backward = 3× forward (recompute + actgrads + wgrads)
 *
 * LoRA training (frozen weight grads skipped):
 * - Without ckpt: backward = actgrads × overhead = ~1.05× forward
 * - With ckpt: backward = recompute (full) + actgrads × overhead = ~2.05× forward
 *
 * Overhead formula: 1.0 + max(0.05, 2 * trainableParams / totalParams).
 * The 0.05 floor represents fixed PEFT framework overhead (hooks, adapter fwd/bwd,
 * gradient routing). The scaling term (2 * trainable/total) only exceeds the floor
 * when trainable > 2.5% of total params — which effectively never happens with LoRA
 * on models ≥7B (even r=64 all_linear on 7B is ~2.3%). In practice, the overhead is
 * a flat 1.05 for all real-world configs. The scaling term exists for correctness on
 * sub-1B models where adapter fraction can be meaningful.
 */
// The 0.05 floor also subsumes adapter forward/backward matmul time, which is
// < 0.5% of base model compute for all practical LoRA configs (≥ 7B models).
// MFU accounts for adapter FLOPs explicitly in base.ts via computeLoraTrainableParams.
export function getLoraBackwardMultiplier(
  model: ModelSpec,
  loraConfig: LoraConfig,
  checkpointing: boolean,
  granularity: 'full' | 'selective' = 'full',
  storedLayers?: number,
  totalLayers?: number,
): number {
  const trainableParams = computeLoraTrainableParams(model, loraConfig.rank, loraConfig.targetModules);
  const overhead = 1.0 + Math.max(0.05, 2 * trainableParams / model.totalParams);

  if (checkpointing && granularity === 'selective') {
    // Selective: recompute attention linear projections (model-dependent fraction)
    const selectiveMult = getSelectiveRecomputeFraction(model) + 1 * overhead;
    // When storedLayers < totalLayers, blend selective and full recompute fractions.
    // Layers 1..K use selective recompute, layers K+1..N use full recompute.
    const N = totalLayers ?? model.numLayers;
    const effectiveStored = Math.min(storedLayers ?? N, N);
    const storedFrac = N > 0 ? effectiveStored / N : 1;
    const fullMult = 1 + 1 * overhead;
    return storedFrac * selectiveMult + (1 - storedFrac) * fullMult;
  } else if (checkpointing) {
    // Full: recompute (1× forward, full model) + actgrads (1× forward) × overhead
    return 1 + 1 * overhead;
  } else {
    // No checkpointing: actgrads only (1× forward) × overhead
    return 1 * overhead;
  }
}

/**
 * NF4 storage bytes per parameter (QLoRA base weights).
 * 0.5 bytes per param + 3% overhead for quantization scales (1 fp16 scale per 64 values).
 */
export const NF4_BYTES_PER_PARAM = 0.5 * 1.03;
