/**
 * ZeRO (Zero Redundancy Optimizer) Strategies
 *
 * Based on the DeepSpeed ZeRO paper:
 * - ZeRO-1: Partition optimizer states
 * - ZeRO-2: + Partition gradients
 * - ZeRO-3: + Partition parameters (equivalent to FSDP)
 *
 * Memory formulas (from DeepSpeed paper):
 * Let Ψ = parameters, N = data parallel degree
 *
 * Baseline DDP: 2Ψ (params) + 2Ψ (grads) + 12Ψ (optimizer) = 16Ψ
 *
 * ZeRO-1: 2Ψ + 2Ψ + 12Ψ/N = 4Ψ + 12Ψ/N
 * ZeRO-2: 2Ψ + 2Ψ/N + 12Ψ/N = 2Ψ + 14Ψ/N
 * ZeRO-3: 2Ψ/N + 2Ψ/N + 12Ψ/N = 16Ψ/N
 *
 * (Assuming bf16 params/grads and fp32 optimizer states)
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
  getDPGroupSizePenalty,
  getDPCollectiveLatencyMs,
  computeComputeEfficiency,
  getEffectiveBackwardMultiplier,
  computeNonMatmulTimeMs,
  gpuCapacityBytes,
  STORED_LAYERS_CAPACITY_FRACTION,
} from './base.ts';
import { computeFSDPExposedComm, computeZeROGradOverlap, applyProtocolOverhead, BUCKET_SIZE_BYTES, PROTOCOL_OVERHEAD, PER_COLLECTIVE_OVERHEAD_MS } from './overlap.ts';
import {
  computeLoraTrainableParams,
  getQloraDequantTimeMs,
  getLoraBackwardMultiplier,
  NF4_BYTES_PER_PARAM,
} from './lora.ts';

export type ZeROStage = 1 | 2 | 3;

export interface ZeROConfig {
  stage: ZeROStage;
  contiguousGradients: boolean;
  overlapComm: boolean;
  reduceScatterBucketSize: number;  // bytes
  allGatherBucketSize: number;      // bytes
  prefetchBucketSize: number;       // bytes for ZeRO-3
}

export const DEFAULT_ZERO_CONFIG: ZeROConfig = {
  stage: 3,
  contiguousGradients: true,
  overlapComm: true,
  reduceScatterBucketSize: 25 * 1024 * 1024, // 25MB
  allGatherBucketSize: 25 * 1024 * 1024,
  prefetchBucketSize: 50 * 1024 * 1024,      // 50MB
};

export class ZeROStrategy extends ParallelismStrategy {
  readonly shortName: string;
  readonly description: string;

  private config: ZeROConfig;

  constructor(config: Partial<ZeROConfig> = {}) {
    super();
    this.config = { ...DEFAULT_ZERO_CONFIG, ...config };
    this.shortName = `ZeRO-${this.config.stage}`;
    this.description = this.getStageDescription();
  }

  get name(): string {
    return `ZeRO Stage ${this.config.stage}`;
  }

  private getStageDescription(): string {
    switch (this.config.stage) {
      case 1:
        return 'Partitions optimizer states across GPUs. Parameters and gradients remain replicated.';
      case 2:
        return 'Partitions optimizer states and gradients. Parameters remain replicated.';
      case 3:
        return 'Partitions optimizer states, gradients, and parameters. Maximum memory efficiency.';
      default:
        return 'ZeRO optimizer partitioning strategy.';
    }
  }

  computeMemoryPerGPU(ctx: StrategyContext): MemoryBreakdown {
    const { model, training, seqLength, microBatchSize, cluster } = ctx;
    const paramDtype = training.dtypes.params;
    const gradDtype = training.dtypes.gradients;
    const activationDtype = training.dtypes.activation;
    const optimizerType = training.optimizer.type;

    const N = cluster.totalGPUs; // Data parallel degree
    const Ψ = model.totalParams;

    const paramBytes = DTYPE_BYTES[paramDtype];

    let parameters: number;
    let gradients: number;
    let optimizerStates: number;

    if (ctx.lora) {
      // LoRA/QLoRA with ZeRO: base weights frozen, only adapters are trainable
      const storageBytes = ctx.lora.method === 'qlora' ? NF4_BYTES_PER_PARAM : DTYPE_BYTES[paramDtype];
      const trainableParams = computeLoraTrainableParams(model, ctx.lora.rank, ctx.lora.targetModules);
      const totalBaseWeightMem = Ψ * storageBytes;
      const totalAdapterWeightMem = trainableParams * DTYPE_BYTES[paramDtype];
      const totalGradMem = trainableParams * DTYPE_BYTES['bf16']; // adapters always BF16
      const totalOptMem = calculateOptimizerMemory(trainableParams, optimizerType, paramDtype);

      switch (this.config.stage) {
        case 1:
          // ZeRO-1: Only optimizer sharded. Params + grads full.
          parameters = totalBaseWeightMem + totalAdapterWeightMem;
          gradients = totalGradMem;
          optimizerStates = totalOptMem / N;
          break;
        case 2:
          parameters = totalBaseWeightMem + totalAdapterWeightMem;
          gradients = totalGradMem / N;
          optimizerStates = totalOptMem / N;
          break;
        case 3:
          parameters = totalBaseWeightMem / N + totalAdapterWeightMem / N;
          gradients = totalGradMem / N;
          optimizerStates = totalOptMem / N;
          break;
        default:
          parameters = totalBaseWeightMem + totalAdapterWeightMem;
          gradients = totalGradMem;
          optimizerStates = totalOptMem / N;
      }
    } else {
      // Calculate base sizes
      const gradBytes = DTYPE_BYTES[gradDtype];
      const totalParamMemory = Ψ * paramBytes;
      const totalGradMemory = Ψ * gradBytes;
      const totalOptimizerMemory = calculateOptimizerMemory(Ψ, optimizerType, paramDtype);

      switch (this.config.stage) {
        case 1:
          // ZeRO-1: Only optimizer states partitioned
          parameters = totalParamMemory;
          gradients = totalGradMemory;
          optimizerStates = totalOptimizerMemory / N;
          break;

        case 2:
          // ZeRO-2: Optimizer states + gradients partitioned
          parameters = totalParamMemory;
          gradients = totalGradMemory / N;
          optimizerStates = totalOptimizerMemory / N;
          break;

        case 3:
          // ZeRO-3: Everything partitioned
          parameters = totalParamMemory / N;
          gradients = totalGradMemory / N;
          optimizerStates = totalOptimizerMemory / N;
          break;

        default:
          parameters = totalParamMemory;
          gradients = totalGradMemory;
          optimizerStates = totalOptimizerMemory / N;
      }
    }

    // For ZeRO-3, use dynamic buffers matching FSDP; stages 1/2 use fixed buckets
    let temporary: number;
    let gatherBufferOverhead = 0;
    if (this.config.stage === 3) {
      const paramsPerLayer = Ψ / model.numLayers;
      const gatheredParamBuffer = paramsPerLayer * paramBytes;
      temporary = gatheredParamBuffer * 2; // Double buffering for overlap
      const gatheredGradBuffer = paramsPerLayer * DTYPE_BYTES[gradDtype];
      const prefetchBuffers = gatheredParamBuffer * 2; // prefetchCount=2, matching FSDP
      gatherBufferOverhead = gatheredParamBuffer + gatheredGradBuffer + prefetchBuffers;
    } else {
      temporary = this.config.reduceScatterBucketSize + this.config.allGatherBucketSize;
    }

    const reserved = calculateReservedMemory(cluster.node.gpu.memoryGB);

    // Resolve stored layers for selective AC auto-sizing.
    // For each candidate K from N down to 0, find max K whose activation
    // memory fits the remaining GPU budget after non-activation costs.
    const nonActivation = parameters + gradients + optimizerStates
      + gatherBufferOverhead + temporary + reserved;
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

    // Activations (not partitioned by ZeRO)
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

    // Peak activations — for ZeRO-3, include gather + grad + prefetch buffers (matching FSDP)
    const peakActivations = activations + gatherBufferOverhead;

    const total = parameters + gradients + optimizerStates + peakActivations +
                  temporary + reserved;

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
    const paramDtype = training.dtypes.params;

    const N = cluster.totalGPUs;

    // LoRA: only adapter grads need AllReduce/ReduceScatter (BF16).
    // AllGather still needs full params at storage bytes.
    let paramBytesTotal: number;
    let gradBytesTotal: number;

    if (ctx.lora) {
      const storageBytes = ctx.lora.method === 'qlora' ? NF4_BYTES_PER_PARAM : DTYPE_BYTES[paramDtype];
      const trainableParams = computeLoraTrainableParams(model, ctx.lora.rank, ctx.lora.targetModules);
      paramBytesTotal = model.totalParams * storageBytes + trainableParams * DTYPE_BYTES[paramDtype];
      gradBytesTotal = trainableParams * DTYPE_BYTES['bf16'];
    } else {
      const gradDtype = training.dtypes.gradients;
      paramBytesTotal = model.totalParams * DTYPE_BYTES[paramDtype];
      gradBytesTotal = model.totalParams * DTYPE_BYTES[gradDtype];
    }

    let dataParallel = 0;

    switch (this.config.stage) {
      case 1:
        // ZeRO-1: AllReduce gradients + AllGather updated params
        dataParallel = 2 * gradBytesTotal * (N - 1) / N;
        dataParallel += paramBytesTotal * (N - 1) / N;
        break;

      case 2:
        // ZeRO-2: ReduceScatter gradients + AllGather updated params
        dataParallel = gradBytesTotal * (N - 1) / N;
        dataParallel += paramBytesTotal * (N - 1) / N;
        break;

      case 3: {
        // ZeRO-3: AllGather params (2x: fwd + bwd) + ReduceScatter grads
        const allGatherFwd = paramBytesTotal * (N - 1) / N;
        const allGatherBwd = paramBytesTotal * (N - 1) / N;
        const reduceScatter = gradBytesTotal * (N - 1) / N;
        dataParallel = allGatherFwd + allGatherBwd + reduceScatter;
        break;
      }
    }

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

    // Compute time calculation
    const tokensPerMicroBatch = seqLength * microBatchSize;
    const totalFlops = tokensPerMicroBatch * model.flopsPerToken;

    const computeDtype = training.dtypes.compute;
    const { effectiveTFLOPS } = computeComputeEfficiency(model, tokensPerMicroBatch, gpu, computeDtype);

    // Non-matmul overhead (memory-bandwidth-bound ops: norms, activations, residual adds)
    const nonMatmulPerMB = computeNonMatmulTimeMs(model, tokensPerMicroBatch, gpu,
      { flashAttention: ctx.flashAttention, seqLength, microBatchSize });

    // QLoRA dequantization time
    const dequantTime = ctx.lora?.method === 'qlora'
      ? getQloraDequantTimeMs(model, gpu, 1, 1) : 0;

    const forwardTime = (totalFlops / (effectiveTFLOPS * 1e12)) * 1000 + dequantTime + nonMatmulPerMB;
    // Backward FLOPs: LoRA skips frozen weight gradients
    let backwardFlopsMultiplier: number;
    if (ctx.lora) {
      backwardFlopsMultiplier = getLoraBackwardMultiplier(model, ctx.lora, ctx.activationCheckpointing, ctx.checkpointingGranularity, this._resolvedStoredLayers);
    } else {
      // Backward multiplier: recompute cost is physically identical regardless of DP type.
      // FSDP/ZeRO-3 AllGather/backward overlap is credited solely via computeFSDPExposedComm() —
      // using a reduced multiplier here would double-count the same overlap window.
      // When storedLayers < N, blends selective and full recompute fractions.
      backwardFlopsMultiplier = getEffectiveBackwardMultiplier(
        model, ctx.activationCheckpointing, ctx.checkpointingGranularity,
        this._resolvedStoredLayers,
      );
    }
    // Backward: non-matmul ops run during gradient computation (+1×),
    // and again during recompute if checkpointing (+1×)
    const bwdNonMatmulMult = ctx.activationCheckpointing ? 2.0 : 1.0;
    const backwardTime = (totalFlops * backwardFlopsMultiplier / (effectiveTFLOPS * 1e12)) * 1000
      + (ctx.lora?.method === 'qlora' && ctx.activationCheckpointing ? dequantTime : 0)
      + nonMatmulPerMB * bwdNonMatmulMult;

    // Optimizer: LoRA only updates trainable params
    let optimizerParamsPerGPU: number;
    if (ctx.lora) {
      const trainableParams = computeLoraTrainableParams(model, ctx.lora.rank, ctx.lora.targetModules);
      // ZeRO shards optimizer across DP
      optimizerParamsPerGPU = trainableParams / cluster.totalGPUs;
    } else {
      optimizerParamsPerGPU = getOptimizerParamsPerGPU(
        model, training.dtypes.params,
        { tp: 1, pp: 1, dp: cluster.totalGPUs, ep: 1, dpType: `zero-${this.config.stage}` }
      );
    }
    const optimizerTime = computeOptimizerStepTime(
      optimizerParamsPerGPU, gpu, training.optimizer.type, training.dtypes.params
    );

    // Communication
    const communication = this.computeCommunication(ctx);
    let bandwidth: number;

    if (cluster.numNodes === 1) {
      bandwidth = cluster.node.intraNodeInterconnect.bandwidthGBps;
    } else {
      // Structural alignment with FSDP: use intra-node BW when sharding fits in one node
      const gpusPerNode = cluster.gpusPerNode;
      if (cluster.totalGPUs <= gpusPerNode) {
        bandwidth = cluster.node.intraNodeInterconnect.bandwidthGBps;
      } else {
        bandwidth = Math.min(
          cluster.node.intraNodeInterconnect.bandwidthGBps,
          cluster.interNodeBandwidthGBps
        );
      }
    }

    // DP scaling penalties use numNodes (= inter-node participants for standalone ZeRO).
    // NCCL hierarchical collectives (ring within node, tree across nodes) have inter-node
    // hops as the bottleneck. Penalties are calibrated for 3D parallel where dp ≈ numNodes.
    const dpGroupSize = cluster.numNodes;
    const dpPenalizedBandwidth = bandwidth * getDPGroupSizePenalty(dpGroupSize);
    const rawCommTime = (communication.total / (dpPenalizedBandwidth * 1e9)) * 1000;
    const commOverheadType = this.config.stage === 3 ? 'dp_fsdp' as const : 'dp_ddp' as const;
    const intraNode = cluster.numNodes === 1;
    // ZeRO-3: per-layer collectives; ZeRO-1/2: commTimePerMB unused (separate grad/param paths below)
    const z3CollectivesPerMB = model.numLayers * 3;
    let commTimePerMB = applyProtocolOverhead(
      rawCommTime, commOverheadType,
      this.config.stage === 3 ? z3CollectivesPerMB : 0, intraNode,
    );

    const totalForwardTime = forwardTime * gradientAccumulationSteps;
    const totalBackwardTime = backwardTime * gradientAccumulationSteps;

    let communicationTime: number;
    let overlap: number;

    if (this.config.stage === 3) {
      // ZeRO-3 does AllGather per layer per micro-batch — scales with GA
      // Add per-collective latency BEFORE overlap calc (matching FSDP)
      const dpCollectivesPerMB = z3CollectivesPerMB;
      commTimePerMB += dpCollectivesPerMB * getDPCollectiveLatencyMs(dpGroupSize);

      // Per-layer FSDP pipeline: decompose comm into per-layer AG and RS times
      const N = cluster.totalGPUs;
      const allGatherBytes = model.totalParams * DTYPE_BYTES[training.dtypes.params];
      const reduceScatterBytes = model.totalParams * DTYPE_BYTES[training.dtypes.gradients];
      const agVolPerLayer = allGatherBytes * (N - 1) / N / model.numLayers;
      const rsVolPerLayer = reduceScatterBytes * (N - 1) / N / model.numLayers;

      const dpCollectiveLatency = getDPCollectiveLatencyMs(dpGroupSize);
      const perCollOverhead = intraNode
        ? PER_COLLECTIVE_OVERHEAD_MS * 0.1
        : PER_COLLECTIVE_OVERHEAD_MS;
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
        backwardPrefetch: this.config.overlapComm,
      });

      communicationTime = commTimePerMB * gradientAccumulationSteps;
      overlap = communicationTime - exposedPerMB * gradientAccumulationSteps;
    } else {
      // ZeRO-1/2: two distinct comm phases per step:
      // 1) Grad sync (AllReduce for stage 1, ReduceScatter for stage 2) — during backward, can overlap
      // 2) Param AllGather after optimizer step — sequential, cannot overlap with compute
      const N = cluster.totalGPUs;
      const gradBytesTotal = model.totalParams * DTYPE_BYTES[training.dtypes.gradients];
      const paramBytesTotal = model.totalParams * DTYPE_BYTES[training.dtypes.params];

      const gradSyncVolume = this.config.stage === 1
        ? 2 * gradBytesTotal * (N - 1) / N   // AllReduce
        : gradBytesTotal * (N - 1) / N;       // ReduceScatter

      const paramGatherVolume = paramBytesTotal * (N - 1) / N; // AllGather

      // Grad sync is bucketed (like DDP); param gather is a single AllGather
      const gradientBytes = model.totalParams * DTYPE_BYTES[training.dtypes.gradients];
      const gradSyncBuckets = Math.max(1, Math.ceil(gradientBytes / BUCKET_SIZE_BYTES));
      const gradSyncTime = applyProtocolOverhead((gradSyncVolume / (dpPenalizedBandwidth * 1e9)) * 1000, commOverheadType, gradSyncBuckets, intraNode);
      const paramGatherTime = applyProtocolOverhead((paramGatherVolume / (dpPenalizedBandwidth * 1e9)) * 1000, commOverheadType, 1, intraNode);

      communicationTime = gradSyncTime + paramGatherTime;
      communicationTime += 2 * getDPCollectiveLatencyMs(dpGroupSize); // gradSync + paramGather

      // Only grad sync can overlap with backward (bucketed AllReduce/ReduceScatter)
      // Param AllGather is fully sequential — zero overlap
      overlap = computeZeROGradOverlap({
        stage: this.config.stage as 1 | 2,
        overlapComm: this.config.overlapComm,
        gradSyncTime,
        backwardTime,
        gradientBytes,
      });
    }

    const total = totalForwardTime + totalBackwardTime + optimizerTime + communicationTime - overlap;

    // Forward/backward sub-breakdown
    const ga = gradientAccumulationSteps;
    const fwdComputeMs = ((totalFlops / (effectiveTFLOPS * 1e12)) * 1000 + dequantTime) * ga;
    const fwdNonMatmulMs = nonMatmulPerMB * ga;
    const bwdComputeMs = ((totalFlops * backwardFlopsMultiplier / (effectiveTFLOPS * 1e12)) * 1000
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

    // Memory check
    if (memory.total > gpuMemoryBytes) {
      errors.push(
        `OOM: requires ${(memory.total / (1024 ** 3)).toFixed(2)} GB per GPU but only ${ctx.cluster.node.gpu.memoryGB} GB available.`
      );
    }

    // Stage-specific warnings
    if (this.config.stage === 1 && ctx.model.totalParams > 10e9) {
      warnings.push(
        'ZeRO-1 only shards optimizer states. ' +
        'For models >10B parameters, consider ZeRO-2 or ZeRO-3.'
      );
    }

    if (this.config.stage === 3 && ctx.cluster.totalGPUs > 64 && ctx.model.totalParams < 5e9) {
      warnings.push(
        'ZeRO-3 with many GPUs for a small model may have high communication overhead. ' +
        'Consider ZeRO-2 or sharding across fewer GPUs.'
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

    let timestamp = 0;

    const maxEventGPUs = Math.min(cluster.totalGPUs, 16);
    for (let gpuId = 0; gpuId < maxEventGPUs; gpuId++) {
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

        if (this.config.stage === 3) {
          events.push({
            id: `gpu${gpuId}-allgather-fwd-${microStep}`,
            type: 'collective-start',
            category: 'collective',
            timestamp: gpuTimestamp,
            duration: timing.forward / gradientAccumulationSteps * 0.2,
            gpuId,
            operation: 'all-gather',
            sizeBytes: communication.dataParallel / 3,
            numRanks: cluster.totalGPUs,
            algorithm: 'ring',
            isIntraNode: cluster.numNodes === 1,
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

      // Gradient sync
      const collectiveOp = this.config.stage === 1 ? 'all-reduce' :
                          this.config.stage >= 2 ? 'reduce-scatter' : 'all-reduce';
      events.push({
        id: `gpu${gpuId}-grad-sync`,
        type: 'collective-start',
        category: 'collective',
        timestamp: gpuTimestamp - timing.overlap,
        duration: timing.communication * 0.6,
        gpuId,
        operation: collectiveOp,
        sizeBytes: communication.dataParallel / 2,
        numRanks: cluster.totalGPUs,
        algorithm: 'ring',
        isIntraNode: cluster.numNodes === 1,
      });

      // Optimizer
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

      // Param sync after optimizer (for ZeRO-1 and ZeRO-2)
      if (this.config.stage < 3) {
        events.push({
          id: `gpu${gpuId}-param-sync`,
          type: 'collective-start',
          category: 'collective',
          timestamp: gpuTimestamp + timing.communication - timing.overlap + timing.optimizer,
          duration: timing.communication * 0.4,
          gpuId,
          operation: 'all-gather',
          sizeBytes: communication.dataParallel / 2,
          numRanks: cluster.totalGPUs,
          algorithm: 'ring',
          isIntraNode: cluster.numNodes === 1,
        });
      }

      timestamp = Math.max(timestamp, gpuTimestamp + timing.communication - timing.overlap + timing.optimizer);
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

// Export instances for each stage
export const zeroStage1Strategy = new ZeROStrategy({ stage: 1 });

// Factory function
export function createZeROStrategy(stage: ZeROStage, config?: Partial<ZeROConfig>): ZeROStrategy {
  return new ZeROStrategy({ ...config, stage });
}
