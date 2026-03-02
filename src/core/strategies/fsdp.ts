/**
 * Fully Sharded Data Parallel (FSDP) Strategy
 * Also known as ZeRO-3
 *
 * FSDP shards model parameters, gradients, and optimizer states across GPUs.
 * Parameters are gathered on-demand for computation and released after.
 *
 * Memory per GPU:
 * - 1/N of parameters (gathered when needed)
 * - 1/N of gradients (reduced and scattered)
 * - 1/N of optimizer states
 * - Activations for micro-batch
 *
 * Communication:
 * - AllGather of parameters before each forward/backward layer
 * - ReduceScatter of gradients after each backward layer
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
  calculateParamMemory,
  calculateGradientMemory,
  calculateOptimizerMemory,
  estimateActivationMemory,
  calculateReservedMemory,
  getOptimizerParamsPerGPU,
  computeOptimizerStepTime,
  getDPGroupSizePenalty,
  getDPCollectiveLatencyMs,
  computeComputeEfficiency,
  getEffectiveBackwardMultiplier,
  computeNonMatmulTimeMs,
  gpuCapacityBytes,
  STORED_LAYERS_CAPACITY_FRACTION,
} from './base.ts';
import { computeFSDPExposedComm, applyProtocolOverhead, PROTOCOL_OVERHEAD, PER_COLLECTIVE_OVERHEAD_MS } from './overlap.ts';
import { getCollectiveBandwidth } from '../hardware/interconnect.ts';
import {
  computeLoraTrainableParams,
  getQloraDequantTimeMs,
  getLoraBackwardMultiplier,
  NF4_BYTES_PER_PARAM,
} from './lora.ts';

export interface FSDPConfig {
  shardingDegree: number;      // How many GPUs to shard across
  cpuOffload: boolean;         // Offload params/optimizer to CPU
  prefetchCount: number;       // Number of layers to prefetch
  backwardPrefetch: boolean;   // Prefetch during backward pass
  limitAllGatherBw: boolean;   // Limit AllGather to prevent memory spikes
}

export const DEFAULT_FSDP_CONFIG: FSDPConfig = {
  shardingDegree: -1, // -1 means shard across all GPUs
  cpuOffload: false,
  prefetchCount: 2,
  backwardPrefetch: true,
  limitAllGatherBw: false,
};

export class FSDPStrategy extends ParallelismStrategy {
  readonly name = 'Fully Sharded Data Parallel';
  readonly shortName = 'FSDP';
  readonly description = 'Shards parameters, gradients, and optimizer states across GPUs. Enables training of models larger than GPU memory.';

  private config: FSDPConfig;

  constructor(config: Partial<FSDPConfig> = {}) {
    super();
    this.config = { ...DEFAULT_FSDP_CONFIG, ...config };
  }

  private getShardingDegree(ctx: StrategyContext): number {
    if (this.config.shardingDegree === -1) {
      return ctx.cluster.totalGPUs;
    }
    return Math.min(this.config.shardingDegree, ctx.cluster.totalGPUs);
  }

  computeMemoryPerGPU(ctx: StrategyContext): MemoryBreakdown {
    const { model, training, seqLength, microBatchSize, cluster } = ctx;
    const paramDtype = training.dtypes.params;
    const gradDtype = training.dtypes.gradients;
    const activationDtype = training.dtypes.activation;
    const optimizerType = training.optimizer.type;

    const shardingDegree = this.getShardingDegree(ctx);

    let parameters: number;
    let gradients: number;
    let optimizerStates: number;

    if (ctx.lora) {
      // FSDP + LoRA/QLoRA: shards both base weights AND adapters
      const storageBytes = ctx.lora.method === 'qlora' ? NF4_BYTES_PER_PARAM : DTYPE_BYTES[paramDtype];
      const baseWeightMem = model.totalParams * storageBytes / shardingDegree;
      const trainableParams = computeLoraTrainableParams(model, ctx.lora.rank, ctx.lora.targetModules);
      const adapterWeightMem = trainableParams * DTYPE_BYTES[paramDtype] / shardingDegree;
      parameters = baseWeightMem + adapterWeightMem;
      // Gradients: trainable params only, BF16, sharded
      gradients = trainableParams * DTYPE_BYTES['bf16'] / shardingDegree;
      // Optimizer: trainable params only, sharded
      optimizerStates = calculateOptimizerMemory(trainableParams, optimizerType, paramDtype) / shardingDegree;
    } else {
      // FSDP: sharded across GPUs
      // Each GPU holds 1/N of params, grads, and optimizer states
      const totalParamMemory = calculateParamMemory(model.totalParams, paramDtype);
      const totalGradMemory = calculateGradientMemory(model.totalParams, gradDtype);
      const totalOptimizerMemory = calculateOptimizerMemory(model.totalParams, optimizerType, paramDtype);

      // Sharded portions
      parameters = totalParamMemory / shardingDegree;
      gradients = totalGradMemory / shardingDegree;
      optimizerStates = totalOptimizerMemory / shardingDegree;
    }

    // Peak activations: during forward/backward, we need to gather full layer params
    // Temporary buffer for one FSDP unit (typically one transformer layer)
    const paramsPerLayer = model.totalParams / model.numLayers;
    const gatheredParamBuffer = paramsPerLayer * DTYPE_BYTES[paramDtype];
    const gatheredGradBuffer = paramsPerLayer * DTYPE_BYTES[gradDtype];

    // With prefetching, we may have multiple layers' params in memory
    const prefetchBuffers = gatheredParamBuffer * this.config.prefetchCount;

    // Temporary buffers for communication
    const temporary = gatheredParamBuffer * 2; // Double buffering for overlap

    // CPU offload reduces GPU memory
    let cpuOffloadSavings = 0;
    if (this.config.cpuOffload) {
      // Offload optimizer states to CPU
      cpuOffloadSavings = optimizerStates * 0.9; // Keep 10% for active state
    }

    // Framework reserved memory
    const reserved = calculateReservedMemory(cluster.node.gpu.memoryGB);

    // Resolve stored layers for selective AC auto-sizing.
    // For each candidate K from N down to 0, find max K whose activation
    // memory fits the remaining GPU budget after non-activation costs.
    const nonActivation = parameters + gradients + optimizerStates
      + gatheredParamBuffer + gatheredGradBuffer + prefetchBuffers
      + temporary + reserved - cpuOffloadSavings;
    let resolvedStoredLayers: number | undefined;
    if (ctx.activationCheckpointing && ctx.checkpointingGranularity === 'selective') {
      if (ctx.selectiveStoredLayers != null) {
        resolvedStoredLayers = ctx.selectiveStoredLayers;
      } else {
        const available = gpuCapacityBytes(cluster.node.gpu.memoryGB) * STORED_LAYERS_CAPACITY_FRACTION - nonActivation;
        resolvedStoredLayers = 0;
        for (let k = model.numLayers; k >= 0; k--) {
          const mem = estimateActivationMemory(
            model, seqLength, microBatchSize, activationDtype,
            true, ctx.flashAttention, 1, 'selective', k,
          );
          if (mem <= available) { resolvedStoredLayers = k; break; }
        }
      }
    }
    this._resolvedStoredLayers = resolvedStoredLayers;

    // Activations are NOT sharded - each GPU processes its own micro-batch
    const activations = estimateActivationMemory(
      model,
      seqLength,
      microBatchSize,
      activationDtype,
      ctx.activationCheckpointing,
      ctx.flashAttention,
      1,
      ctx.checkpointingGranularity,
      resolvedStoredLayers,
    );

    const peakActivations = activations + gatheredParamBuffer + gatheredGradBuffer + prefetchBuffers;

    const total = parameters + gradients + optimizerStates + peakActivations + temporary + reserved - cpuOffloadSavings;

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
    const { model, training } = ctx;
    const paramDtype = training.dtypes.params;

    const shardingDegree = this.getShardingDegree(ctx);
    const n = shardingDegree;

    // FSDP communication per step:
    // Forward: AllGather params for each layer
    // Backward: AllGather params for each layer + ReduceScatter grads for each layer
    //
    // LoRA: AllGather still gathers full base weights (frozen but needed for forward).
    // Uses storage bytes: NF4=0.5 for QLoRA, paramDtype for LoRA.
    // ReduceScatter only for trainable adapter grads (BF16, tiny volume).

    let allGatherBytes: number;
    let reduceScatterBytes: number;

    if (ctx.lora) {
      const storageBytes = ctx.lora.method === 'qlora' ? NF4_BYTES_PER_PARAM : DTYPE_BYTES[paramDtype];
      const trainableParams = computeLoraTrainableParams(model, ctx.lora.rank, ctx.lora.targetModules);
      // AllGather: full base weights at storage bytes + adapter weights
      allGatherBytes = model.totalParams * storageBytes + trainableParams * DTYPE_BYTES[paramDtype];
      // ReduceScatter: only adapter grads at BF16
      reduceScatterBytes = trainableParams * DTYPE_BYTES['bf16'];
    } else {
      const gradDtype = training.dtypes.gradients;
      allGatherBytes = model.totalParams * DTYPE_BYTES[paramDtype];
      reduceScatterBytes = model.totalParams * DTYPE_BYTES[gradDtype];
    }

    const allGatherVolume = allGatherBytes * (n - 1) / n;
    const reduceScatterVolume = reduceScatterBytes * (n - 1) / n;

    // Forward: 1 AllGather per layer
    // Backward: 1 AllGather + 1 ReduceScatter per layer
    const forwardComm = allGatherVolume;
    const backwardComm = allGatherVolume + reduceScatterVolume;

    const dataParallel = forwardComm + backwardComm;

    return {
      dataParallel,
      tensorParallel: 0,
      pipelineParallel: 0,
      expertParallel: 0,
      total: dataParallel,
    };
  }

  computeTiming(ctx: StrategyContext): TimingBreakdown {
    const { model, training, seqLength, microBatchSize, cluster, gradientAccumulationSteps } = ctx;
    const gpu = cluster.node.gpu;

    // Calculate compute time (same as DDP)
    const tokensPerMicroBatch = seqLength * microBatchSize;
    const flopsPerToken = model.flopsPerToken;
    const totalFlops = tokensPerMicroBatch * flopsPerToken;

    const forwardFlops = totalFlops;
    // Backward FLOPs: LoRA skips frozen weight gradients
    let backwardFlops: number;
    if (ctx.lora) {
      backwardFlops = totalFlops * getLoraBackwardMultiplier(model, ctx.lora, ctx.activationCheckpointing, ctx.checkpointingGranularity, this._resolvedStoredLayers);
    } else {
      // Backward multiplier: recompute cost is physically identical regardless of DP type.
      // FSDP AllGather/backward overlap is credited solely via computeFSDPExposedComm() —
      // using a reduced multiplier here would double-count the same overlap window.
      backwardFlops = totalFlops * getEffectiveBackwardMultiplier(
        model, ctx.activationCheckpointing, ctx.checkpointingGranularity,
        this._resolvedStoredLayers,
      );
    }

    const computeDtype = training.dtypes.compute;
    const { effectiveTFLOPS } = computeComputeEfficiency(model, tokensPerMicroBatch, gpu, computeDtype);

    // Non-matmul overhead (memory-bandwidth-bound ops: norms, activations, residual adds)
    const nonMatmulPerMB = computeNonMatmulTimeMs(model, tokensPerMicroBatch, gpu,
      { flashAttention: ctx.flashAttention, seqLength, microBatchSize });

    // QLoRA dequantization time (bandwidth-bound, added to forward)
    const dequantTime = ctx.lora?.method === 'qlora'
      ? getQloraDequantTimeMs(model, gpu, 1, 1) : 0;

    const forwardTime = (forwardFlops / (effectiveTFLOPS * 1e12)) * 1000 + dequantTime + nonMatmulPerMB;
    // Backward: non-matmul ops run during gradient computation (+1×),
    // and again during recompute if checkpointing (+1×)
    const bwdNonMatmulMult = ctx.activationCheckpointing ? 2.0 : 1.0;
    const backwardTime = (backwardFlops / (effectiveTFLOPS * 1e12)) * 1000
      + (ctx.lora?.method === 'qlora' && ctx.activationCheckpointing ? dequantTime : 0)
      + nonMatmulPerMB * bwdNonMatmulMult;

    // Optimizer: LoRA only updates trainable params
    let optimizerParamsPerGPU: number;
    if (ctx.lora) {
      const trainableParams = computeLoraTrainableParams(model, ctx.lora.rank, ctx.lora.targetModules);
      optimizerParamsPerGPU = trainableParams / cluster.totalGPUs; // FSDP shards across DP
    } else {
      optimizerParamsPerGPU = getOptimizerParamsPerGPU(
        model, training.dtypes.params,
        { tp: 1, pp: 1, dp: cluster.totalGPUs, ep: 1, dpType: 'fsdp' }
      );
    }
    const optimizerTime = computeOptimizerStepTime(
      optimizerParamsPerGPU, gpu, training.optimizer.type, training.dtypes.params
    );

    // Communication time
    const communication = this.computeCommunication(ctx);
    const shardingDegree = this.getShardingDegree(ctx);
    let bandwidth: number;

    if (cluster.numNodes === 1) {
      bandwidth = getCollectiveBandwidth(cluster.node.intraNodeInterconnect);
    } else {
      // FSDP within nodes uses NVLink, across nodes uses IB
      // Weighted average based on sharding topology
      const gpusPerNode = cluster.gpusPerNode;

      if (shardingDegree <= gpusPerNode) {
        bandwidth = getCollectiveBandwidth(cluster.node.intraNodeInterconnect);
      } else {
        // Cross-node communication
        bandwidth = Math.min(
          getCollectiveBandwidth(cluster.node.intraNodeInterconnect),
          cluster.interNodeBandwidthGBps
        );
      }
    }

    const commBytes = communication.total;
    // DP scaling penalties use numNodes (= inter-node participants for standalone FSDP).
    // NCCL hierarchical collectives (ring within node, tree across nodes) have inter-node
    // hops as the bottleneck. Penalties are calibrated for 3D parallel where dp ≈ numNodes.
    const dpGroupSize = cluster.numNodes;
    const dpPenalizedBandwidth = bandwidth * getDPGroupSizePenalty(dpGroupSize);
    const rawCommTime = (commBytes / (dpPenalizedBandwidth * 1e9)) * 1000;
    const dpCollectivesPerMB = model.numLayers * 3; // AllGather fwd + AllGather bwd + ReduceScatter bwd
    const dpLatencyPerMB = dpCollectivesPerMB * getDPCollectiveLatencyMs(dpGroupSize);
    const intraNode = cluster.numNodes === 1;
    const commTimePerMB = applyProtocolOverhead(rawCommTime, 'dp_fsdp', dpCollectivesPerMB, intraNode) + dpLatencyPerMB;

    // Per-layer FSDP pipeline: decompose comm into per-layer AG and RS times
    const allGatherBytes = model.totalParams * DTYPE_BYTES[training.dtypes.params];
    const reduceScatterBytes = model.totalParams * DTYPE_BYTES[training.dtypes.gradients];
    const n = shardingDegree;
    const agVolPerLayer = allGatherBytes * (n - 1) / n / model.numLayers;
    const rsVolPerLayer = reduceScatterBytes * (n - 1) / n / model.numLayers;

    // Convert per-layer volumes to time using same BW/overhead as commTimePerMB
    const dpCollectiveLatency = getDPCollectiveLatencyMs(dpGroupSize);
    const perCollOverhead = intraNode
      ? PER_COLLECTIVE_OVERHEAD_MS * 0.1   // NVLink: ~5μs
      : PER_COLLECTIVE_OVERHEAD_MS;         // IB: ~50μs
    const agPerLayer = (agVolPerLayer / (dpPenalizedBandwidth * 1e9)) * 1000
      * (1 + PROTOCOL_OVERHEAD.dp_fsdp) + perCollOverhead + dpCollectiveLatency;
    const rsPerLayer = (rsVolPerLayer / (dpPenalizedBandwidth * 1e9)) * 1000
      * (1 + PROTOCOL_OVERHEAD.dp_fsdp) + perCollOverhead + dpCollectiveLatency;

    const exposedPerMB = computeFSDPExposedComm({
      fwdComputePerLayer: [forwardTime / model.numLayers],
      bwdComputePerLayer: [backwardTime / model.numLayers],
      allGatherPerLayer: agPerLayer,
      reduceScatterPerLayer: rsPerLayer,
      numLayers: model.numLayers,
      backwardPrefetch: this.config.backwardPrefetch,
    });

    // Communication per micro-batch: AllGather(fwd) + AllGather(bwd) + ReduceScatter(bwd).
    // Real FSDP with no_sync() skips ReduceScatter on non-final micro-batches (GA>1),
    // but the savings are negligible in the compute-bound regime — per-layer pipelining
    // already hides ~95% of communication, and the RS cold-start delta is < 0.2% of step time.
    const communicationTime = commTimePerMB * gradientAccumulationSteps;
    const overlap = communicationTime - exposedPerMB * gradientAccumulationSteps;

    const totalForwardTime = forwardTime * gradientAccumulationSteps;
    const totalBackwardTime = backwardTime * gradientAccumulationSteps;

    const total = totalForwardTime + totalBackwardTime + optimizerTime + communicationTime - overlap;

    // Forward/backward sub-breakdown
    const ga = gradientAccumulationSteps;
    const fwdComputeMs = ((forwardFlops / (effectiveTFLOPS * 1e12)) * 1000 + dequantTime) * ga;
    const fwdNonMatmulMs = nonMatmulPerMB * ga;
    const bwdComputeMs = ((backwardFlops / (effectiveTFLOPS * 1e12)) * 1000
      + (ctx.lora?.method === 'qlora' && ctx.activationCheckpointing ? dequantTime : 0)) * ga;
    const bwdNonMatmulMs = nonMatmulPerMB * bwdNonMatmulMult * ga;

    return {
      forward: totalForwardTime,
      backward: totalBackwardTime,
      optimizer: optimizerTime,
      communication: communicationTime,
      overlap,
      scaleOverhead: 0,
      total,
      forwardComputeMs: fwdComputeMs,
      forwardNonMatmulMs: fwdNonMatmulMs,
      backwardComputeMs: bwdComputeMs,
      backwardNonMatmulMs: bwdNonMatmulMs,
      tpExposed: 0,
      ppExposed: 0,
      dpGross: communicationTime,
      cpExposed: 0,
    };
  }

  validate(ctx: StrategyContext): StrategyValidation {
    const errors: string[] = [];
    const warnings: string[] = [];
    const memory = this.computeMemoryPerGPU(ctx);
    const gpuMemoryBytes = gpuCapacityBytes(ctx.cluster.node.gpu.memoryGB);

    // Check if sharded model fits
    if (memory.total > gpuMemoryBytes) {
      errors.push(
        `OOM: even with FSDP sharding, requires ${(memory.total / (1024 ** 3)).toFixed(2)} GB per GPU but only ${ctx.cluster.node.gpu.memoryGB} GB available.`
      );
    } else if (memory.total > gpuMemoryBytes * 0.85) {
      warnings.push(
        `Memory usage is ${((memory.total / gpuMemoryBytes) * 100).toFixed(1)}% of GPU capacity. ` +
        `Consider reducing micro-batch size for stability.`
      );
    }

    // Check sharding efficiency
    const shardingDegree = this.getShardingDegree(ctx);
    if (shardingDegree > 64 && ctx.model.totalParams < 10e9) {
      warnings.push(
        `High sharding degree (${shardingDegree}) for relatively small model (${(ctx.model.totalParams / 1e9).toFixed(1)}B). ` +
        `Communication overhead may be significant.`
      );
    }

    // Check for cross-node sharding efficiency
    if (ctx.cluster.numNodes > 1) {
      const gpusPerNode = ctx.cluster.gpusPerNode;
      if (shardingDegree > gpusPerNode) {
        warnings.push(
          'FSDP sharding spans multiple nodes. ' +
          'Consider using hybrid FSDP (shard within node, DDP across nodes) for better performance.'
        );
      }
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

    let timestamp = 0;

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
        strategyName: this.name,
      },
    });

    const layerTime = timing.forward / model.numLayers / gradientAccumulationSteps;
    const commPerLayer = communication.dataParallel / model.numLayers / 2; // Forward share

    const maxEventGPUs = Math.min(cluster.totalGPUs, 16);
    for (let gpuId = 0; gpuId < maxEventGPUs; gpuId++) {
      let gpuTimestamp = 0;

      for (let microStep = 0; microStep < gradientAccumulationSteps; microStep++) {
        // Forward pass with per-layer AllGather
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

        // Interleaved AllGather events
        for (let layer = 0; layer < model.numLayers; layer++) {
          if (layer < model.numLayers - 1) { // Prefetch next layer
            events.push({
              id: `gpu${gpuId}-allgather-fwd-${microStep}-${layer}`,
              type: 'collective-start',
              category: 'collective',
              timestamp: gpuTimestamp + layer * layerTime,
              duration: layerTime * 0.3, // Overlapped
              gpuId,
              operation: 'all-gather',
              sizeBytes: commPerLayer,
              numRanks: this.getShardingDegree(ctx),
              algorithm: 'ring',
              isIntraNode: cluster.numNodes === 1,
            });
          }
        }

        gpuTimestamp += timing.forward / gradientAccumulationSteps;

        // Backward pass with AllGather + ReduceScatter
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

      // Optimizer step
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

      timestamp = Math.max(timestamp, gpuTimestamp + timing.optimizer);
    }

    events.push({
      id: 'sim-end',
      type: 'simulation-end',
      category: 'simulation',
      timestamp,
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

// Export singleton instance with default config
export const fsdpStrategy = new FSDPStrategy();
