/**
 * Sequence Parallelism Strategy
 * Based on Megatron-LM Sequence Parallelism
 *
 * Sequence parallelism extends tensor parallelism by also distributing:
 * - LayerNorm computations across the sequence dimension
 * - Dropout computations across the sequence dimension
 *
 * This reduces activation memory by the TP degree for these operations.
 *
 * Combined with TP:
 * - Sequence split for LayerNorm/Dropout
 * - Head split for Attention
 * - Column/Row split for MLP
 *
 * Communication:
 * - AllGather before attention/MLP (gather sequence)
 * - ReduceScatter after attention/MLP (scatter + reduce)
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
import { getCollectiveBandwidth } from '../hardware/interconnect.ts';
import {
  computeLoraParamsPerRank,
  getQloraDequantTimeMs,
  getLoraBackwardMultiplier,
  NF4_BYTES_PER_PARAM,
} from './lora.ts';

export interface SequenceParallelConfig {
  tpDegree: number;  // Tensor parallel degree (SP uses same process group)
}

export const DEFAULT_SP_CONFIG: SequenceParallelConfig = {
  tpDegree: 1,
};

export class SequenceParallelStrategy extends ParallelismStrategy {
  readonly name = 'Sequence Parallelism';
  readonly shortName = 'SP';
  readonly description = 'Extends tensor parallelism to distribute LayerNorm and Dropout across sequence dimension. Reduces activation memory by TP factor.';

  private config: SequenceParallelConfig;

  constructor(config: Partial<SequenceParallelConfig> = {}) {
    super();
    this.config = { ...DEFAULT_SP_CONFIG, ...config };
  }

  computeMemoryPerGPU(ctx: StrategyContext): MemoryBreakdown {
    const { model, training, seqLength, microBatchSize, cluster } = ctx;
    const paramDtype = training.dtypes.params;
    const gradDtype = training.dtypes.gradients;
    const activationDtype = training.dtypes.activation;
    const optimizerType = training.optimizer.type;

    const tp = this.config.tpDegree;

    // Parameters: same as TP (split attention and MLP)
    const attentionParams = model.attentionParams;
    const mlpParams = model.mlpParams;
    const embeddingParams = model.embeddingParams;
    const outputParams = model.tiedEmbeddings ? 0 : model.vocabSize * model.hiddenSize;
    const normParams = model.totalParams - attentionParams - mlpParams - embeddingParams - outputParams;

    const shardedParams = (attentionParams + mlpParams) / tp + embeddingParams / tp + outputParams / tp + normParams;

    let parameters: number;
    let gradients: number;
    let optimizerStates: number;

    if (ctx.lora) {
      const storageBytes = ctx.lora.method === 'qlora' ? NF4_BYTES_PER_PARAM : DTYPE_BYTES[paramDtype];
      parameters = shardedParams * storageBytes;
      const adapterParamsPerRank = computeLoraParamsPerRank(model, ctx.lora.rank, ctx.lora.targetModules, tp, 1);
      parameters += adapterParamsPerRank * DTYPE_BYTES[paramDtype];
      gradients = adapterParamsPerRank * DTYPE_BYTES['bf16'];
      optimizerStates = calculateOptimizerMemory(adapterParamsPerRank, optimizerType, paramDtype);
    } else {
      parameters = shardedParams * DTYPE_BYTES[paramDtype];
      gradients = shardedParams * DTYPE_BYTES[gradDtype];
      optimizerStates = calculateOptimizerMemory(shardedParams, optimizerType, paramDtype);
    }

    // Temporary: AllGather/ReduceScatter buffers
    const temporary = model.hiddenSize * seqLength * microBatchSize * DTYPE_BYTES[activationDtype] * 2;

    const reserved = calculateReservedMemory(cluster.node.gpu.memoryGB);

    // Non-activation memory (needed for stored-layers auto-resolve budget)
    const nonActivation = parameters + gradients + optimizerStates + temporary + reserved;

    // SP reduces all activation tensors by 1/tp: TP-sharded tensors (Q,K,V,MLP) split along
    // hidden dim, SP-sharded tensors (LN inputs, residuals) split along sequence dim.
    const activationMultiplier = tp > 1 ? 1 / tp : 1;

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
      ctx.activationCheckpointing, ctx.flashAttention,
      1, ctx.checkpointingGranularity, resolvedStoredLayers,
    );
    const activations = baseActivations * activationMultiplier;

    const peakActivations = activations * 1.4;

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

    const tp = this.config.tpDegree;

    // SP communication per layer:
    // Forward: AllGather before attention, AllGather before MLP (gather sequence)
    //          ReduceScatter after attention, ReduceScatter after MLP
    // Backward: Reverse operations

    // FP8 training with Transformer Engine (Hopper+): TP/SP comm uses FP8 quantized collectives.
    const gpu = cluster.node.gpu;
    const canUseFP8Comm = activationDtype === 'fp8' && gpu.hasTransformerEngine;
    const tpCommBytes = canUseFP8Comm
      ? DTYPE_BYTES['fp8']
      : Math.max(DTYPE_BYTES[activationDtype], DTYPE_BYTES['bf16']);
    const fullActivationSize = seqLength * microBatchSize * model.hiddenSize * tpCommBytes;

    // AllGather: each rank sends 1/tp and receives full
    const allGatherVolume = fullActivationSize * (tp - 1) / tp;
    // ReduceScatter: each rank sends full and receives 1/tp (plus reduction)
    const reduceScatterVolume = fullActivationSize * (tp - 1) / tp;

    // Per layer: 2 AllGather + 2 ReduceScatter for forward
    // Same for backward
    const perLayerComm = 2 * (allGatherVolume + reduceScatterVolume);
    const tensorParallel = perLayerComm * model.numLayers * 2; // fwd + bwd

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

    const tp = this.config.tpDegree;

    // Compute time (reduced by TP)
    const tokensPerMicroBatch = seqLength * microBatchSize;
    const flopsPerToken = model.flopsPerToken;
    const totalFlops = tokensPerMicroBatch * flopsPerToken / tp;

    const computeDtype = training.dtypes.compute;
    const { effectiveTFLOPS } = computeComputeEfficiency(model, tokensPerMicroBatch, gpu, computeDtype, { tp });

    // QLoRA dequantization time
    const dequantTime = ctx.lora?.method === 'qlora'
      ? getQloraDequantTimeMs(model, gpu, tp, 1) : 0;

    const forwardTime = (totalFlops / (effectiveTFLOPS * 1e12)) * 1000 + dequantTime;
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

    // Communication
    const communication = this.computeCommunication(ctx);
    // TP/SP typically uses NVLink (within node), but cross-node TP uses hierarchical all-reduce
    let bandwidth: number;
    if (tp <= cluster.gpusPerNode) {
      bandwidth = getCollectiveBandwidth(cluster.node.intraNodeInterconnect);
    } else {
      const G = cluster.gpusPerNode;
      const N = Math.ceil(tp / G);
      const nvBW = getCollectiveBandwidth(cluster.node.intraNodeInterconnect);
      const ibBW = cluster.interNodeBandwidthGBps;
      bandwidth = ((tp - 1) / tp) / ((G - 1) / G / nvBW + (N - 1) / N / ibBW);
    }

    const crossNodeTP = tp > cluster.gpusPerNode;
    const rawCommTime = (communication.total / (bandwidth * 1e9)) * 1000;
    // SP: 2 collectives per layer (AllGather + ReduceScatter per layer)
    const spCollectives = 2 * model.numLayers;
    const tpCommWithOverhead = applyProtocolOverhead(rawCommTime, crossNodeTP ? 'tp_crossnode' : 'tp_nvlink', spCollectives, !crossNodeTP);

    // SP comm (AllGather/ReduceScatter) is per-microbatch, layer-by-layer,
    // pipelined with compute. SP ops can be chunked and streamed.
    const computePerMB = forwardTime + backwardTime;
    // SP always enabled here — AllGather/ReduceScatter pipeline better than AllReduce
    const spOverlap = computeTPOverlap({
      computePerMB,
      tpCommWithOverhead,
    });
    // Exposed SP per step = non-overlapped residual per microbatch × GA
    const communicationTime = tpCommWithOverhead * (1 - spOverlap) * gradientAccumulationSteps;
    const overlap = 0; // Layer-overlap already applied above

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
    const tp = this.config.tpDegree;

    // Same constraints as TP
    if (ctx.model.numAttentionHeads % tp !== 0) {
      errors.push(
        `TP/SP degree ${tp} must divide number of attention heads (${ctx.model.numAttentionHeads}).`
      );
    }

    if (tp > ctx.cluster.gpusPerNode) {
      warnings.push(
        `TP/SP degree ${tp} exceeds GPUs per node (${ctx.cluster.gpusPerNode}). ` +
        `Cross-node TP will use slower inter-node interconnect instead of NVLink.`
      );
      if (tp % ctx.cluster.gpusPerNode !== 0) {
        warnings.push(
          `TP/SP degree ${tp} doesn't evenly divide GPUs per node (${ctx.cluster.gpusPerNode}). ` +
          `Cross-node TP performance estimate may be inaccurate.`
        );
      }
    }

    // Sequence length should be divisible by TP for clean SP
    if (ctx.seqLength % tp !== 0) {
      warnings.push(
        `Sequence length ${ctx.seqLength} not divisible by SP degree ${tp}. ` +
        `May require padding.`
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
    const { cluster, gradientAccumulationSteps, model } = ctx;

    const tp = this.config.tpDegree;

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

    let maxTimestamp = 0;

    for (let gpuId = 0; gpuId < tp; gpuId++) {
      let gpuTimestamp = 0;

      for (let microStep = 0; microStep < gradientAccumulationSteps; microStep++) {
        // Forward
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

        // AllGather events (interleaved with compute)
        const layerTime = timing.forward / model.numLayers / gradientAccumulationSteps;
        for (let layer = 0; layer < model.numLayers; layer++) {
          events.push({
            id: `gpu${gpuId}-ag-${microStep}-${layer}`,
            type: 'collective-start',
            category: 'collective',
            timestamp: gpuTimestamp + layer * layerTime,
            duration: layerTime * 0.2,
            gpuId,
            operation: 'all-gather',
            sizeBytes: timing.communication / model.numLayers / 4,
            numRanks: tp,
            algorithm: 'ring',
            isIntraNode: true,
          });
          events.push({
            id: `gpu${gpuId}-rs-${microStep}-${layer}`,
            type: 'collective-start',
            category: 'collective',
            timestamp: gpuTimestamp + layer * layerTime + layerTime * 0.5,
            duration: layerTime * 0.2,
            gpuId,
            operation: 'reduce-scatter',
            sizeBytes: timing.communication / model.numLayers / 4,
            numRanks: tp,
            algorithm: 'ring',
            isIntraNode: true,
          });
        }

        gpuTimestamp += timing.forward / gradientAccumulationSteps;

        // Backward
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
export function createSequenceParallelStrategy(
  tpDegree: number
): SequenceParallelStrategy {
  return new SequenceParallelStrategy({ tpDegree });
}

