/**
 * Distributed Data Parallel (DDP) Strategy
 *
 * DDP replicates the model on each GPU and synchronizes gradients via AllReduce.
 * Each GPU processes a different batch of data.
 *
 * Memory per GPU:
 * - Full model parameters
 * - Full gradients
 * - Full optimizer states
 * - Activations for micro-batch
 *
 * Communication:
 * - AllReduce of gradients after each backward pass
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
  calculateTemporaryMemory,
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
import { computeDDPOverlap, applyProtocolOverhead, BUCKET_SIZE_BYTES } from './overlap.ts';
import {
  computeLoraTrainableParams,
  getQloraDequantTimeMs,
  getLoraBackwardMultiplier,
  NF4_BYTES_PER_PARAM,
} from './lora.ts';

export class DDPStrategy extends ParallelismStrategy {
  readonly name = 'Distributed Data Parallel';
  readonly shortName = 'DDP';
  readonly description = 'Replicates model on each GPU, synchronizes gradients via AllReduce. Simple and efficient for models that fit in GPU memory.';

  computeMemoryPerGPU(ctx: StrategyContext): MemoryBreakdown {
    const { model, training, seqLength, microBatchSize, cluster } = ctx;
    const paramDtype = training.dtypes.params;
    const gradDtype = training.dtypes.gradients;
    const activationDtype = training.dtypes.activation;
    const optimizerType = training.optimizer.type;

    let parameters: number;
    let gradients: number;
    let optimizerStates: number;

    if (ctx.lora) {
      // LoRA/QLoRA: base weights (frozen) + adapter weights
      const storageBytes = ctx.lora.method === 'qlora' ? NF4_BYTES_PER_PARAM : DTYPE_BYTES[paramDtype];
      const baseWeightMem = model.totalParams * storageBytes;
      const trainableParams = computeLoraTrainableParams(model, ctx.lora.rank, ctx.lora.targetModules);
      const adapterWeightMem = trainableParams * DTYPE_BYTES[paramDtype];
      parameters = baseWeightMem + adapterWeightMem;
      // Gradients: trainable params only, always BF16
      gradients = trainableParams * DTYPE_BYTES['bf16'];
      // Optimizer: trainable params only
      optimizerStates = calculateOptimizerMemory(trainableParams, optimizerType, paramDtype);
    } else {
      // DDP: full model on each GPU
      parameters = calculateParamMemory(model.totalParams, paramDtype);
      gradients = calculateGradientMemory(model.totalParams, gradDtype);
      optimizerStates = calculateOptimizerMemory(model.totalParams, optimizerType, paramDtype);
    }

    // Temporary buffers for gradient bucketing
    const temporary = calculateTemporaryMemory(
      model.totalParams,
      gradDtype,
      25 // Default 25MB buckets
    );

    // Framework reserved memory
    const reserved = calculateReservedMemory(cluster.node.gpu.memoryGB);

    // Resolve stored layers for selective AC auto-sizing.
    // For each candidate K from N down to 0, find max K whose activation
    // memory fits the remaining GPU budget after non-activation costs.
    const nonActivation = parameters + gradients + optimizerStates + temporary + reserved;
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

    // Activations for forward pass
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

    // Peak activations during backward: base activations + one layer's gradient workspace
    const effectiveLayers = ctx.activationCheckpointing ? Math.sqrt(model.numLayers) : model.numLayers;
    const peakActivations = activations + activations / effectiveLayers;

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
    const { model, training, cluster } = ctx;

    // DDP: AllReduce gradients across all GPUs
    // LoRA: only adapter gradients, always BF16
    let gradientBytes: number;
    if (ctx.lora) {
      const trainableParams = computeLoraTrainableParams(model, ctx.lora.rank, ctx.lora.targetModules);
      gradientBytes = trainableParams * DTYPE_BYTES['bf16'];
    } else {
      const gradDtype = training.dtypes.gradients;
      gradientBytes = model.totalParams * DTYPE_BYTES[gradDtype];
    }
    // AllReduce volume = 2 * gradients (reduce-scatter + all-gather)
    const allReduceVolume = 2 * gradientBytes;

    // For ring AllReduce: (n-1)/n of data sent per GPU
    const n = cluster.totalGPUs;
    const effectiveVolume = allReduceVolume * (n - 1) / n;

    return {
      dataParallel: effectiveVolume,
      tensorParallel: 0,
      pipelineParallel: 0,
      expertParallel: 0,
      total: effectiveVolume,
    };
  }

  computeTiming(ctx: StrategyContext): TimingBreakdown {
    const { model, training, seqLength, microBatchSize, cluster, gradientAccumulationSteps } = ctx;
    const gpu = cluster.node.gpu;

    // Calculate compute time
    const tokensPerMicroBatch = seqLength * microBatchSize;
    const flopsPerToken = model.flopsPerToken;
    const totalFlops = tokensPerMicroBatch * flopsPerToken;

    // Forward pass FLOPs (1x)
    const forwardFlops = totalFlops;
    // Backward FLOPs: LoRA skips frozen weight gradients
    let backwardFlops: number;
    if (ctx.lora) {
      backwardFlops = totalFlops * getLoraBackwardMultiplier(model, ctx.lora, ctx.activationCheckpointing, ctx.checkpointingGranularity, this._resolvedStoredLayers);
    } else {
      // Backward = 2× forward (2.85× with checkpointing). DDP lacks AllGather
      // overlap but gets minor savings from backward kernel scheduling.
      // Selective: recompute attention linear projections (model-dependent fraction).
      // When storedLayers < N, blends selective and full recompute fractions.
      backwardFlops = totalFlops * getEffectiveBackwardMultiplier(
        model, ctx.activationCheckpointing, ctx.checkpointingGranularity,
        this._resolvedStoredLayers,
      );
    }

    const computeDtype = training.dtypes.compute;
    const { effectiveTFLOPS } = computeComputeEfficiency(model, tokensPerMicroBatch, gpu, computeDtype);

    // Non-matmul overhead (memory-bandwidth-bound ops: norms, activations, residual adds)
    const nonMatmulPerMB = computeNonMatmulTimeMs(model, tokensPerMicroBatch, gpu);

    // QLoRA dequantization time (bandwidth-bound, added to forward)
    const dequantTime = ctx.lora?.method === 'qlora'
      ? getQloraDequantTimeMs(model, gpu, 1, 1) : 0;

    // Time in milliseconds
    const forwardTime = (forwardFlops / (effectiveTFLOPS * 1e12)) * 1000 + dequantTime + nonMatmulPerMB;
    // Backward: non-matmul ops run during gradient computation (+1×),
    // and again during recompute if checkpointing (+1×)
    const bwdNonMatmulMult = ctx.activationCheckpointing ? 2.0 : 1.0;
    // With ckpt, backward recomputes forward (including dequant)
    const backwardTime = (backwardFlops / (effectiveTFLOPS * 1e12)) * 1000
      + (ctx.lora?.method === 'qlora' && ctx.activationCheckpointing ? dequantTime : 0)
      + nonMatmulPerMB * bwdNonMatmulMult;

    // Optimizer step time (memory-bandwidth-bound, independent of batch/seq)
    // LoRA: optimizer only updates trainable params
    let optimizerParamsPerGPU: number;
    if (ctx.lora) {
      optimizerParamsPerGPU = computeLoraTrainableParams(model, ctx.lora.rank, ctx.lora.targetModules);
    } else {
      optimizerParamsPerGPU = getOptimizerParamsPerGPU(
        model, training.dtypes.params,
        { tp: 1, pp: 1, dp: cluster.totalGPUs, ep: 1, dpType: 'ddp' }
      );
    }
    const optimizerTime = computeOptimizerStepTime(
      optimizerParamsPerGPU, gpu, training.optimizer.type, training.dtypes.params
    );

    // Communication time for AllReduce
    const communication = this.computeCommunication(ctx);
    let bandwidth: number;

    if (cluster.numNodes === 1) {
      // Single node: use NVLink/NVSwitch bandwidth
      bandwidth = cluster.node.intraNodeInterconnect.bandwidthGBps;
    } else {
      // Multi-node: limited by inter-node bandwidth
      bandwidth = Math.min(
        cluster.node.intraNodeInterconnect.bandwidthGBps,
        cluster.interNodeBandwidthGBps
      );
    }

    // AllReduce time (with some overhead) — bandwidth degrades at large DP group sizes
    // DP scaling penalties use numNodes (= inter-node participants for standalone DDP).
    // NCCL hierarchical collectives (ring within node, tree across nodes) have inter-node
    // hops as the bottleneck. Penalties are calibrated for 3D parallel where dp ≈ numNodes.
    const commBytes = communication.total;
    const dpGroupSize = cluster.numNodes;
    const dpPenalizedBandwidth = bandwidth * getDPGroupSizePenalty(dpGroupSize);
    const rawCommTime = (commBytes / (dpPenalizedBandwidth * 1e9)) * 1000;
    const gradientBytes = model.totalParams * DTYPE_BYTES[training.dtypes.gradients];
    const numBuckets = Math.max(1, Math.ceil(gradientBytes / BUCKET_SIZE_BYTES));
    const intraNode = cluster.numNodes === 1;
    let communicationTime = applyProtocolOverhead(rawCommTime, 'dp_ddp', numBuckets, intraNode);
    communicationTime += getDPCollectiveLatencyMs(dpGroupSize); // single AllReduce

    // With gradient accumulation, scale compute time
    const totalForwardTime = forwardTime * gradientAccumulationSteps;
    const totalBackwardTime = backwardTime * gradientAccumulationSteps;

    // Communication can overlap with backward pass (bucketed AllReduce timeline model)
    const overlap = computeDDPOverlap({
      commTime: communicationTime,
      backwardTime,
      gradientBytes,
    });

    // Total step time
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

    // Check if model fits in GPU memory - this is a hard constraint
    if (memory.total > gpuMemoryBytes) {
      errors.push(
        `OOM: requires ${(memory.total / (1024 ** 3)).toFixed(2)} GB per GPU but only ${ctx.cluster.node.gpu.memoryGB} GB available.`
      );
    } else if (memory.total > gpuMemoryBytes * 0.9) {
      const tip = ctx.activationCheckpointing
        ? 'Consider reducing micro-batch size or sequence length.'
        : 'Consider reducing micro-batch size or enabling activation checkpointing.';
      warnings.push(
        `Memory usage (${(memory.total / (1024 ** 3)).toFixed(2)} GB) is close to GPU capacity. ${tip}`
      );
    }

    // Check for efficient GPU utilization
    if (ctx.cluster.totalGPUs > 1 && ctx.microBatchSize < 4) {
      warnings.push(
        'Small micro-batch size may lead to low GPU utilization. ' +
        'Consider increasing batch size or using gradient accumulation.'
      );
    }

    // Check for communication efficiency
    if (ctx.cluster.numNodes > 1) {
      const timingBreakdown = this.computeTiming(ctx);
      const commOverhead = timingBreakdown.communication / timingBreakdown.total;

      if (commOverhead > 0.3) {
        warnings.push(
          `Communication overhead is ${(commOverhead * 100).toFixed(1)}%. ` +
          `Consider using gradient compression or reducing synchronization frequency.`
        );
      }
    }

    // MoE warning: DDP replicates all experts on every GPU
    if (ctx.model.isMoE && ctx.model.numExperts) {
      warnings.push(
        `DDP replicates all ${ctx.model.numExperts} experts on every GPU. Consider a hybrid strategy (e.g. FSDP + TP) with expert parallelism to distribute experts.`
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
    const { cluster, gradientAccumulationSteps } = ctx;

    let timestamp = 0;

    // Simulation start
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
        modelName: ctx.model.name,
        strategyName: this.name,
      },
    });

    // For each GPU, generate parallel events (cap to avoid millions of events)
    const maxEventGPUs = Math.min(cluster.totalGPUs, 16);
    for (let gpuId = 0; gpuId < maxEventGPUs; gpuId++) {
      let gpuTimestamp = 0;

      // Gradient accumulation loop
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

        gpuTimestamp += timing.forward / gradientAccumulationSteps;

        // Backward pass (overlapped with communication on last micro-step)
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

      // Communication (AllReduce) - overlapped with last backward
      const commStart = gpuTimestamp - timing.overlap;
      events.push({
        id: `gpu${gpuId}-comm`,
        type: 'collective-start',
        category: 'collective',
        timestamp: commStart,
        duration: timing.communication,
        gpuId,
        operation: 'all-reduce',
        sizeBytes: this.computeCommunication(ctx).total,
        numRanks: cluster.totalGPUs,
        algorithm: cluster.totalGPUs <= 8 ? 'ring' : 'tree',
        isIntraNode: cluster.numNodes === 1,
      });

      // Optimizer step
      events.push({
        id: `gpu${gpuId}-opt`,
        type: 'phase-start',
        category: 'phase',
        timestamp: gpuTimestamp + timing.communication - timing.overlap,
        duration: timing.optimizer,
        gpuId,
        phase: 'optimizer',
        stepNumber: 0,
      });

      timestamp = Math.max(timestamp, gpuTimestamp + timing.communication - timing.overlap + timing.optimizer);
    }

    // Simulation end
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

// Export singleton instance
export const ddpStrategy = new DDPStrategy();
