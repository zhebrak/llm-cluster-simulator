/**
 * Tensor Parallelism (TP) Strategy
 * Based on Megatron-LM style tensor parallelism
 *
 * TP splits individual layers across GPUs:
 * - Attention: Split heads across GPUs
 * - MLP: Column-parallel first linear, row-parallel second linear
 *
 * Communication:
 * - Forward: AllReduce after each transformer block
 * - Backward: AllReduce after each transformer block (gradient sync)
 *
 * Memory per GPU:
 * - 1/TP of attention and MLP weights
 * - Full embedding and output layers (can be parallelized separately)
 * - Activations for micro-batch
 */
// See docs/PHYSICS.md and docs/STRATEGIES.md for formula derivations and calibration anchors.

import type {
  MemoryBreakdown,
  CommunicationBreakdown,
  TimingBreakdown,
  StrategyValidation,
  SimulationEvent,
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
  computeComputeEfficiency,
  getEffectiveBackwardMultiplier,
  gpuCapacityBytes,
  STORED_LAYERS_CAPACITY_FRACTION,
} from './base.ts';
import { computeTPOverlap, applyProtocolOverhead } from './overlap.ts';
import {
  computeLoraParamsPerRank,
  getQloraDequantTimeMs,
  getLoraBackwardMultiplier,
  NF4_BYTES_PER_PARAM,
} from './lora.ts';

export interface TensorParallelConfig {
  degree: number;                    // TP world size
  sequenceParallel: boolean;         // Enable sequence parallelism
  paralllelizeEmbedding: boolean;    // Parallelize embedding layer
  parallelizeOutput: boolean;        // Parallelize output layer
}

export const DEFAULT_TP_CONFIG: TensorParallelConfig = {
  degree: 1,
  sequenceParallel: true,
  paralllelizeEmbedding: true,
  parallelizeOutput: true,
};

export class TensorParallelStrategy extends ParallelismStrategy {
  readonly name = 'Tensor Parallelism';
  readonly shortName = 'TP';
  readonly description = 'Splits individual layers across GPUs. Efficient for large layers, requires fast interconnect (NVLink).';

  private config: TensorParallelConfig;

  constructor(config: Partial<TensorParallelConfig> = {}) {
    super();
    this.config = { ...DEFAULT_TP_CONFIG, ...config };
  }

  computeMemoryPerGPU(ctx: StrategyContext): MemoryBreakdown {
    const { model, training, seqLength, microBatchSize, cluster } = ctx;
    const paramDtype = training.dtypes.params;
    const gradDtype = training.dtypes.gradients;
    const activationDtype = training.dtypes.activation;
    const optimizerType = training.optimizer.type;

    const tp = this.config.degree;

    // Calculate per-component parameters
    const embeddingParams = model.embeddingParams;
    const attentionParams = model.attentionParams;
    const mlpParams = model.mlpParams;
    const outputParams = model.tiedEmbeddings ? 0 : model.vocabSize * model.hiddenSize;
    const normParams = model.totalParams - embeddingParams - attentionParams - mlpParams - outputParams;

    // TP splits attention and MLP
    // Attention: QKV projections split by heads, output projection row-parallel
    // MLP: First linear column-parallel, second linear row-parallel
    const shardedAttention = attentionParams / tp;
    const shardedMLP = mlpParams / tp;

    // Embedding: can be column-parallel (vocab split) or replicated
    const shardedEmbedding = this.config.paralllelizeEmbedding ?
      embeddingParams / tp : embeddingParams;

    // Output: row-parallel or replicated
    const shardedOutput = this.config.parallelizeOutput ?
      outputParams / tp : outputParams;

    // Norms are replicated (small)
    const shardedNorm = normParams;

    const totalShardedParams = shardedAttention + shardedMLP + shardedEmbedding + shardedOutput + shardedNorm;

    // Memory calculations
    let parameters: number;
    let gradients: number;
    let optimizerStates: number;

    if (ctx.lora) {
      const storageBytes = ctx.lora.method === 'qlora' ? NF4_BYTES_PER_PARAM : DTYPE_BYTES[paramDtype];
      parameters = totalShardedParams * storageBytes;
      // Add adapter weight memory (partially TP-sharded)
      const adapterParamsPerRank = computeLoraParamsPerRank(model, ctx.lora.rank, ctx.lora.targetModules, tp, 1);
      parameters += adapterParamsPerRank * DTYPE_BYTES[paramDtype];
      // Gradients: adapter only, BF16
      gradients = adapterParamsPerRank * DTYPE_BYTES['bf16'];
      // Optimizer: adapter only
      optimizerStates = calculateOptimizerMemory(adapterParamsPerRank, optimizerType, paramDtype);
    } else {
      parameters = totalShardedParams * DTYPE_BYTES[paramDtype];
      gradients = totalShardedParams * DTYPE_BYTES[gradDtype];
      optimizerStates = calculateOptimizerMemory(totalShardedParams, optimizerType, paramDtype);
    }

    // Temporary buffers for AllReduce
    const hiddenBytes = seqLength * microBatchSize * model.hiddenSize * DTYPE_BYTES[activationDtype];
    const temporary = hiddenBytes * 2; // Double buffer

    const reserved = calculateReservedMemory(cluster.node.gpu.memoryGB);

    // Non-activation memory (needed for stored-layers auto-resolve budget)
    const nonActivation = parameters + gradients + optimizerStates + temporary + reserved;

    // Activations: split by sequence parallel if enabled
    // With SP: all activation tensors effectively 1/tp per rank.
    // Without SP: only TP-sharded tensors (Q,K,V,attn_out,MLP) get 1/tp;
    // replicated tensors (4h: LN inputs, residuals) stay full size.
    let activationMultiplier: number;
    if (this.config.sequenceParallel && tp > 1) {
      activationMultiplier = 1 / tp;
    } else if (tp > 1) {
      const mlpIntermCoeff = model.gatedMLP ? 3 : 2;
      const sharded = (model.numAttentionHeads * model.headDim) + 2 * (model.numKvHeads * model.headDim)
        + model.hiddenSize + mlpIntermCoeff * model.intermediateSize;
      const replicated = 4 * model.hiddenSize;
      activationMultiplier = (sharded / tp + replicated) / (sharded + replicated);
    } else {
      activationMultiplier = 1;
    }

    // Resolve stored-layers count for selective AC
    let resolvedStoredLayers = model.numLayers;
    if (ctx.activationCheckpointing && ctx.checkpointingGranularity === 'selective') {
      if (ctx.selectiveStoredLayers != null) {
        resolvedStoredLayers = ctx.selectiveStoredLayers;
      } else {
        // Auto: find max K that fits GPU memory
        const available = gpuCapacityBytes(cluster.node.gpu.memoryGB) * STORED_LAYERS_CAPACITY_FRACTION - nonActivation;
        resolvedStoredLayers = 0;
        for (let k = model.numLayers; k >= 0; k--) {
          const mem = estimateActivationMemory(
            model, seqLength, microBatchSize, activationDtype,
            true, ctx.flashAttention, 1, 'selective', k,
          );
          if (mem * activationMultiplier <= available) { resolvedStoredLayers = k; break; }
        }
      }
    }
    this._resolvedStoredLayers = resolvedStoredLayers;

    const baseActivations = estimateActivationMemory(
      model, seqLength, microBatchSize, activationDtype,
      ctx.activationCheckpointing, ctx.flashAttention, 1,
      ctx.checkpointingGranularity, resolvedStoredLayers,
    );
    const activations = baseActivations * activationMultiplier;
    const peakActivations = activations * 1.3; // Some overhead for comm buffers

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
    const { model, training, seqLength, microBatchSize, cluster } = ctx;
    const activationDtype = training.dtypes.activation;

    const tp = this.config.degree;

    // TP communication per transformer layer:
    // Forward: 2 AllReduce (after attention, after MLP) or 1 AllReduce if fused
    // Backward: 2 AllReduce (same positions)
    // With sequence parallel: AllGather/ReduceScatter instead

    // FP8 training with Transformer Engine (Hopper+): TP all-reduce uses FP8 quantized collectives.
    const gpu = cluster.node.gpu;
    const canUseFP8Comm = activationDtype === 'fp8' && gpu.hasTransformerEngine;
    const tpCommBytes = canUseFP8Comm
      ? DTYPE_BYTES['fp8']
      : Math.max(DTYPE_BYTES[activationDtype], DTYPE_BYTES['bf16']);
    const hiddenBytes = seqLength * microBatchSize * model.hiddenSize * tpCommBytes;

    let perLayerComm: number;

    if (this.config.sequenceParallel) {
      // Sequence parallel: AllGather + ReduceScatter per block
      // More communication but smaller messages
      const allGatherVolume = hiddenBytes * (tp - 1) / tp;
      const reduceScatterVolume = hiddenBytes * (tp - 1) / tp;
      // Forward: 2 AllGather (before attention input, before MLP input)
      // Backward: 2 ReduceScatter + 2 AllGather
      perLayerComm = 2 * (allGatherVolume + reduceScatterVolume);
    } else {
      // Standard TP: AllReduce after each parallel region
      // Forward: 2 AllReduce (after attention output, after MLP output)
      // Backward: 2 AllReduce
      const allReduceVolume = 2 * hiddenBytes * (tp - 1) / tp;
      perLayerComm = 2 * allReduceVolume; // fwd + bwd
    }

    const tensorParallel = perLayerComm * model.numLayers;

    return {
      dataParallel: 0,
      tensorParallel,
      pipelineParallel: 0,
      expertParallel: 0,
      total: tensorParallel,
    };
  }

  computeTiming(ctx: StrategyContext): TimingBreakdown {
    const { model, training, seqLength, microBatchSize, cluster, gradientAccumulationSteps } = ctx;
    const gpu = cluster.node.gpu;

    const tp = this.config.degree;

    // Compute time (reduced by TP)
    const tokensPerMicroBatch = seqLength * microBatchSize;
    const flopsPerToken = model.flopsPerToken;

    // TP ideally provides perfect scaling for compute
    // But some layers (norms, embeddings) may not scale
    const totalFlops = tokensPerMicroBatch * flopsPerToken / tp;

    const computeDtype = training.dtypes.compute;
    const { effectiveTFLOPS } = computeComputeEfficiency(model, tokensPerMicroBatch, gpu, computeDtype, { tp });

    // QLoRA dequantization time
    const dequantTime = ctx.lora?.method === 'qlora'
      ? getQloraDequantTimeMs(model, gpu, tp, 1) : 0;

    const forwardTime = (totalFlops / (effectiveTFLOPS * 1e12)) * 1000 + dequantTime;
    // Backward: LoRA skips frozen weight gradients
    let backwardMultiplier: number;
    if (ctx.lora) {
      backwardMultiplier = getLoraBackwardMultiplier(model, ctx.lora, ctx.activationCheckpointing, ctx.checkpointingGranularity, this._resolvedStoredLayers);
    } else {
      backwardMultiplier = getEffectiveBackwardMultiplier(
        model, ctx.activationCheckpointing, ctx.checkpointingGranularity,
        this._resolvedStoredLayers,
      );
    }
    const backwardTime = (totalFlops / (effectiveTFLOPS * 1e12)) * 1000 * backwardMultiplier
      + (ctx.lora?.method === 'qlora' && ctx.activationCheckpointing ? dequantTime : 0);

    // Optimizer: LoRA only updates adapter params per rank
    let optimizerParamsPerGPU: number;
    if (ctx.lora) {
      optimizerParamsPerGPU = computeLoraParamsPerRank(model, ctx.lora.rank, ctx.lora.targetModules, tp, 1);
    } else {
      optimizerParamsPerGPU = getOptimizerParamsPerGPU(
        model, training.dtypes.params,
        { tp: cluster.totalGPUs, pp: 1, dp: 1, ep: 1, dpType: 'ddp' }
      );
    }
    const optimizerTime = computeOptimizerStepTime(
      optimizerParamsPerGPU, gpu, training.optimizer.type, training.dtypes.params
    );

    // Communication: TP requires NVLink for efficiency
    const communication = this.computeCommunication(ctx);

    // TP typically uses NVLink (within node), but cross-node TP uses hierarchical all-reduce
    let bandwidth: number;
    if (tp <= cluster.gpusPerNode) {
      bandwidth = cluster.node.intraNodeInterconnect.bandwidthGBps;
    } else {
      const G = cluster.gpusPerNode;
      const N = Math.ceil(tp / G);
      const nvBW = cluster.node.intraNodeInterconnect.bandwidthGBps;
      const ibBW = cluster.interNodeBandwidthGBps;
      bandwidth = ((tp - 1) / tp) / ((G - 1) / G / nvBW + (N - 1) / N / ibBW);
    }

    const crossNodeTP = tp > cluster.gpusPerNode;
    const rawCommTime = (communication.total / (bandwidth * 1e9)) * 1000;
    // 2 collectives per layer (attention AllReduce + MLP AllReduce)
    const tpCollectives = 2 * model.numLayers;
    const tpCommWithOverhead = applyProtocolOverhead(rawCommTime, crossNodeTP ? 'tp_crossnode' : 'tp_nvlink', tpCollectives, !crossNodeTP);

    // TP comm is per-microbatch: all-reduces happen layer-by-layer within each
    // microbatch's forward/backward, pipelined with compute. Overlap efficiency
    // depends on compute-to-comm ratio (large models → high overlap).
    const computePerMB = forwardTime + backwardTime;
    const tpOverlapEff = computeTPOverlap({
      computePerMB,
      tpCommWithOverhead,
    });
    // Exposed TP per step = non-overlapped residual per microbatch × GA
    const communicationTime = tpCommWithOverhead * (1 - tpOverlapEff) * gradientAccumulationSteps;
    const overlap = 0; // TP layer-overlap already applied above

    const totalForwardTime = forwardTime * gradientAccumulationSteps;
    const totalBackwardTime = backwardTime * gradientAccumulationSteps;

    const total = totalForwardTime + totalBackwardTime + optimizerTime + communicationTime - overlap;

    return {
      forward: totalForwardTime,
      backward: totalBackwardTime,
      optimizer: optimizerTime,
      communication: communicationTime,
      overlap,
      scaleOverhead: 0,
      total,
      forwardComputeMs: totalForwardTime,
      backwardComputeMs: totalBackwardTime,
      tpExposed: communicationTime,
      ppExposed: 0,
      dpGross: 0,
      cpExposed: 0,
    };
  }

  validate(ctx: StrategyContext): StrategyValidation {
    const errors: string[] = [];
    const warnings: string[] = [];
    const tp = this.config.degree;

    // TP degree must divide number of attention heads
    if (ctx.model.numAttentionHeads % tp !== 0) {
      errors.push(
        `TP degree ${tp} must divide number of attention heads (${ctx.model.numAttentionHeads}).`
      );
    }

    // TP degree should be <= GPUs per node (NVLink)
    if (tp > ctx.cluster.gpusPerNode) {
      warnings.push(
        `TP degree ${tp} exceeds GPUs per node (${ctx.cluster.gpusPerNode}). ` +
        `Cross-node TP will use slower inter-node interconnect instead of NVLink.`
      );
      if (tp % ctx.cluster.gpusPerNode !== 0) {
        warnings.push(
          `TP degree ${tp} doesn't evenly divide GPUs per node (${ctx.cluster.gpusPerNode}). ` +
          `Cross-node TP performance estimate may be inaccurate.`
        );
      }
    }

    // Check for power of 2
    if (tp > 1 && !Number.isInteger(Math.log2(tp))) {
      warnings.push(
        `TP degree ${tp} is not a power of 2. This may reduce communication efficiency.`
      );
    }

    // Memory check
    const memory = this.computeMemoryPerGPU(ctx);
    const gpuMemoryBytes = gpuCapacityBytes(ctx.cluster.node.gpu.memoryGB);

    if (memory.total > gpuMemoryBytes) {
      errors.push(
        `OOM: requires ${(memory.total / (1024 ** 3)).toFixed(2)} GB per GPU but only ${ctx.cluster.node.gpu.memoryGB} GB available.`
      );
    }

    // NVLink requirement
    if (tp > 1 && ctx.cluster.node.gpu.nvlinkBandwidthGBps === 0) {
      warnings.push(
        'GPU does not have NVLink. TP over PCIe will have significantly lower performance.'
      );
    }

    // TP efficiency for small models
    if (tp > 4 && ctx.model.totalParams < 10e9) {
      warnings.push(
        `High TP degree (${tp}) for relatively small model may have poor efficiency.`
      );
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
    const communication = this.computeCommunication(ctx);
    const { cluster, gradientAccumulationSteps, model } = ctx;

    const tp = this.config.degree;

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
        strategyName: `${this.name} (TP=${tp})`,
      },
    });

    const layerTime = timing.forward / model.numLayers / gradientAccumulationSteps;
    const commPerLayer = communication.tensorParallel / model.numLayers / 2;

    let maxTimestamp = 0;

    // For each TP group (assuming cluster is all TP for now)
    for (let gpuId = 0; gpuId < tp; gpuId++) {
      let gpuTimestamp = 0;

      for (let microStep = 0; microStep < gradientAccumulationSteps; microStep++) {
        // Forward pass
        events.push({
          id: `gpu${gpuId}-fwd-${microStep}`,
          type: 'phase-start',
          category: 'phase',
          timestamp: gpuTimestamp,
          duration: timing.forward / gradientAccumulationSteps,
          gpuId,
          phase: 'forward',
          stepNumber: 0,
          microBatchId: microStep,
        });

        // Per-layer AllReduce
        for (let layer = 0; layer < model.numLayers; layer++) {
          const layerStart = gpuTimestamp + layer * layerTime;
          events.push({
            id: `gpu${gpuId}-tp-allreduce-fwd-${microStep}-${layer}`,
            type: 'collective-start',
            category: 'collective',
            timestamp: layerStart + layerTime * 0.8,
            duration: layerTime * 0.15,
            gpuId,
            operation: this.config.sequenceParallel ? 'all-gather' : 'all-reduce',
            sizeBytes: commPerLayer / model.numLayers,
            numRanks: tp,
            algorithm: 'ring',
            isIntraNode: true,
          });
        }

        gpuTimestamp += timing.forward / gradientAccumulationSteps;

        // Backward pass
        events.push({
          id: `gpu${gpuId}-bwd-${microStep}`,
          type: 'phase-start',
          category: 'phase',
          timestamp: gpuTimestamp,
          duration: timing.backward / gradientAccumulationSteps,
          gpuId,
          phase: 'backward',
          stepNumber: 0,
          microBatchId: microStep,
        });

        gpuTimestamp += timing.backward / gradientAccumulationSteps;
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
}

// Export factory
export function createTensorParallelStrategy(
  degree: number,
  config?: Partial<TensorParallelConfig>
): TensorParallelStrategy {
  return new TensorParallelStrategy({ ...config, degree });
}
