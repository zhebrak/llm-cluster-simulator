/**
 * Pipeline Parallelism (PP) Strategy
 * Based on GPipe and PipeDream scheduling
 *
 * PP splits the model vertically across stages (GPUs):
 * - Each stage processes a contiguous set of layers
 * - Micro-batches flow through the pipeline
 *
 * Schedules:
 * - GPipe: All forward, then all backward (simple but large bubble)
 * - 1F1B: One Forward One Backward interleaved (smaller bubble)
 * - Interleaved 1F1B: Virtual stages for even smaller bubble
 *
 * Memory per GPU:
 * - 1/PP of model parameters
 * - Activations for in-flight micro-batches
 *
 * Communication:
 * - Point-to-point: Send activations to next stage
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
  computeComputeEfficiency,
  getEffectiveBackwardMultiplier,
  gpuCapacityBytes,
  STORED_LAYERS_CAPACITY_FRACTION,
} from './base.ts';
import { computePPOverlap, applyProtocolOverhead } from './overlap.ts';
import { computeLoraTrainableParams, getQloraDequantTimeMs, getLoraBackwardMultiplier, NF4_BYTES_PER_PARAM } from './lora.ts';
import { getPerNicBandwidthGBps } from '../hardware/interconnect.ts';

export interface PipelineParallelConfig {
  degree: number;                    // PP stages
  numMicroBatches: number;           // Micro-batches in pipeline
  schedule: PipelineSchedule;        // Scheduling algorithm
  interleavedStages?: number;        // For interleaved 1F1B
  activationCheckpointing: boolean;
  checkpointingGranularity: 'full' | 'selective';
}

export const DEFAULT_PP_CONFIG: PipelineParallelConfig = {
  degree: 1,
  numMicroBatches: 8,
  schedule: '1f1b',
  activationCheckpointing: true,
  checkpointingGranularity: 'full',
};

export class PipelineParallelStrategy extends ParallelismStrategy {
  readonly name = 'Pipeline Parallelism';
  readonly shortName = 'PP';
  readonly description = 'Splits model into stages across GPUs. Suitable for deep models, works over slower interconnects.';

  private config: PipelineParallelConfig;

  constructor(config: Partial<PipelineParallelConfig> = {}) {
    super();
    this.config = { ...DEFAULT_PP_CONFIG, ...config };
  }

  /**
   * Calculate pipeline bubble fraction
   */
  calculateBubble(): number {
    const pp = this.config.degree;
    const m = this.config.numMicroBatches;

    switch (this.config.schedule) {
      case 'gpipe':
        // GPipe bubble: fraction of wall-clock time that is idle
        return (pp - 1) / (pp - 1 + m);

      case '1f1b':
        // 1F1B bubble: (pp-1) warmup slots out of (pp-1+m) total slots
        return (pp - 1) / (pp - 1 + m);

      case 'interleaved-1f1b': {
        // Interleaved reduces bubble by number of virtual stages
        const v = this.config.interleavedStages ?? 2;
        return (pp - 1) / (pp - 1 + m * v);
      }

      case 'zero-bubble':
        // Zero bubble scheduling (theoretical)
        return 0.05; // Small overhead

      default:
        return (pp - 1) / (pp - 1 + m);
    }
  }

  /**
   * Calculate number of in-flight micro-batches per stage
   */
  private getInFlightMicroBatches(): number {
    const pp = this.config.degree;
    const m = this.config.numMicroBatches;

    switch (this.config.schedule) {
      case 'gpipe':
        // GPipe: all micro-batches in flight during forward
        return m;

      case '1f1b':
        // 1F1B: at most pp micro-batches per stage
        return pp;

      case 'interleaved-1f1b': {
        // Interleaved: fewer per virtual stage
        const v = this.config.interleavedStages ?? 2;
        return Math.ceil(pp / v);
      }

      default:
        return pp;
    }
  }

  computeMemoryPerGPU(ctx: StrategyContext): MemoryBreakdown {
    const { model, training, seqLength, microBatchSize, cluster } = ctx;
    const paramDtype = training.dtypes.params;
    const gradDtype = training.dtypes.gradients;
    const activationDtype = training.dtypes.activation;
    const optimizerType = training.optimizer.type;

    const pp = this.config.degree;

    // Parameters: split evenly across stages
    const layersPerStage = Math.ceil(model.numLayers / pp);

    // First stage has embedding, last stage has output head
    // Middle stages have transformer layers only
    // Approximate as even split
    const paramsPerStage = model.totalParams / pp;

    // Parameters, gradients, optimizer — per stage
    let parameters: number;
    let gradients: number;
    let optimizerStates: number;

    if (ctx.lora) {
      // LoRA/QLoRA: base weights (frozen, split across PP stages) + adapter weights (split across PP stages)
      const storageBytes = ctx.lora.method === 'qlora' ? NF4_BYTES_PER_PARAM : DTYPE_BYTES[paramDtype];
      parameters = paramsPerStage * storageBytes;
      const trainablePerStage = computeLoraTrainableParams(model, ctx.lora.rank, ctx.lora.targetModules) / pp;
      parameters += trainablePerStage * DTYPE_BYTES[paramDtype];
      // Gradients: trainable params only, always BF16
      gradients = trainablePerStage * DTYPE_BYTES['bf16'];
      // Optimizer: trainable params only
      optimizerStates = calculateOptimizerMemory(trainablePerStage, optimizerType, paramDtype);
    } else {
      parameters = paramsPerStage * DTYPE_BYTES[paramDtype];
      gradients = paramsPerStage * DTYPE_BYTES[gradDtype];
      optimizerStates = calculateOptimizerMemory(paramsPerStage, optimizerType, paramDtype);
    }

    // Activations: need to store for in-flight micro-batches
    const inFlightMicroBatches = this.getInFlightMicroBatches();
    const tokensPerMicroBatch = seqLength * microBatchSize;

    // Per-layer activation size
    const activationPerLayer = tokensPerMicroBatch * model.hiddenSize * DTYPE_BYTES[activationDtype];

    let activations: number;
    if (this.config.activationCheckpointing) {
      // With checkpointing: store only layer boundaries per micro-batch
      activations = activationPerLayer * inFlightMicroBatches * 2; // Input + output
    } else {
      // Without checkpointing: store all layer activations
      activations = activationPerLayer * layersPerStage * inFlightMicroBatches;
    }

    const peakActivations = activations * 1.5; // Peak during backward with gradients

    // Temporary: communication buffers for send/recv
    const temporary = activationPerLayer * 4; // Double buffer for send and recv

    const reserved = calculateReservedMemory(cluster.node.gpu.memoryGB);

    // Resolve stored-layers count for selective AC
    const stageLayerCount = layersPerStage;
    let resolvedStoredLayers = model.numLayers;
    if (ctx.activationCheckpointing && ctx.checkpointingGranularity === 'selective') {
      if (ctx.selectiveStoredLayers != null) {
        resolvedStoredLayers = ctx.selectiveStoredLayers;
      } else {
        // Auto: find max K that fits GPU memory (per-stage budget)
        const nonActivation = parameters + gradients + optimizerStates + temporary + reserved;
        const available = gpuCapacityBytes(cluster.node.gpu.memoryGB) * STORED_LAYERS_CAPACITY_FRACTION - nonActivation;
        resolvedStoredLayers = 0;
        for (let k = model.numLayers; k >= 0; k--) {
          const mem = estimateActivationMemory(
            model, seqLength, microBatchSize, activationDtype,
            true, ctx.flashAttention, 1, 'selective', k,
          );
          // PP splits activations across stages; scale by stageLayerCount / numLayers
          if (mem * stageLayerCount / model.numLayers <= available) { resolvedStoredLayers = k; break; }
        }
      }
    }
    this._resolvedStoredLayers = resolvedStoredLayers;

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
    const activationDtype = training.dtypes.activation;

    const pp = this.config.degree;
    const m = this.config.numMicroBatches;

    // PP communication: point-to-point between adjacent stages
    // Each micro-batch activation is sent/received once per stage boundary
    // FP8 training: PP comm stays BF16 (activations transferred at higher precision)
    const commActivationBytes = Math.max(DTYPE_BYTES[activationDtype], DTYPE_BYTES['bf16']);
    const activationSize = seqLength * microBatchSize * model.hiddenSize * commActivationBytes;

    // Forward: send activation to next stage (pp-1 boundaries, m micro-batches)
    // Backward: send gradients to previous stage (same volume)
    const pipelineParallel = 2 * activationSize * m * (pp - 1);

    return {
      dataParallel: 0,
      tensorParallel: 0,
      pipelineParallel,
      expertParallel: 0,
      total: pipelineParallel,
    };
  }

  computeTiming(ctx: StrategyContext): TimingBreakdown {
    const { model, training, seqLength, microBatchSize, cluster, gradientAccumulationSteps } = ctx;
    const gpu = cluster.node.gpu;

    const pp = this.config.degree;
    const m = this.config.numMicroBatches;

    // Compute time per stage
    const tokensPerMicroBatch = seqLength * microBatchSize;
    const flopsPerToken = model.flopsPerToken;
    const totalFlops = tokensPerMicroBatch * flopsPerToken / pp; // Split by PP

    const computeDtype = training.dtypes.compute;
    const { effectiveTFLOPS } = computeComputeEfficiency(
      model, tokensPerMicroBatch, gpu, computeDtype
    );

    // Time per micro-batch per stage
    let forwardTimePerMicroBatch = (totalFlops / (effectiveTFLOPS * 1e12)) * 1000;
    // QLoRA dequantization: bandwidth-bound, per-stage (pp divides the cost)
    if (ctx.lora?.method === 'qlora') {
      forwardTimePerMicroBatch += getQloraDequantTimeMs(model, gpu, 1, pp);
    }
    // Backward = 2× forward (2.85× with checkpointing). PP lacks FSDP-style
    // AllGather overlap but gets minor savings from backward kernel scheduling.
    const backwardTimePerMicroBatch = forwardTimePerMicroBatch * (
      ctx.lora
        ? getLoraBackwardMultiplier(model, ctx.lora, ctx.activationCheckpointing, ctx.checkpointingGranularity, this._resolvedStoredLayers)
        : getEffectiveBackwardMultiplier(
            model, ctx.activationCheckpointing, ctx.checkpointingGranularity,
            this._resolvedStoredLayers,
          )
    );

    // Communication time per micro-batch
    const communication = this.computeCommunication(ctx);
    const commPerMicroBatch = communication.pipelineParallel / m / 2 / (pp - 1 || 1);

    // PP can use inter-node links since it's point-to-point
    const perNicBW = getPerNicBandwidthGBps(
      cluster.node.interNodeInterconnect, cluster.node.numNICs
    );
    const concurrentP2PStreams = Math.min(pp - 1, cluster.node.numNICs);
    const bandwidth = ctx.cluster.numNodes > 1
      ? Math.min(perNicBW, cluster.interNodeBandwidthGBps / concurrentP2PStreams)
      : cluster.node.intraNodeInterconnect.bandwidthGBps;

    // 1 P2P transfer per direction; the caller doubles for fwd+bwd
    const ppIntraNode = ctx.cluster.numNodes === 1;
    const commTimePerMicroBatch = applyProtocolOverhead((commPerMicroBatch / (bandwidth * 1e9)) * 1000, 'pp', 1, ppIntraNode);

    // Total pipeline time with bubble
    // Clamp bubble to prevent nonsensical timing when numMicroBatches < pp
    const bubble = Math.min(this.calculateBubble(), 0.95);

    // Ideal time: all micro-batches processed without bubble
    const idealForward = forwardTimePerMicroBatch * m;
    const idealBackward = backwardTimePerMicroBatch * m;

    // Actual time includes bubble overhead
    const bubbleTime = (idealForward + idealBackward) * bubble;

    // Communication overlaps with compute in steady state (C/(C+T) overlap model)
    const totalCommTime = commTimePerMicroBatch * m * 2;
    const computePerMB = forwardTimePerMicroBatch + backwardTimePerMicroBatch;
    const commOverlapEff = computePPOverlap({ computePerMB, ppCommPerMB: commTimePerMicroBatch * 2 });
    const commOverlap = totalCommTime * commOverlapEff;

    // Scale by gradient accumulation
    const forwardTime = idealForward * gradientAccumulationSteps;
    const backwardTime = idealBackward * gradientAccumulationSteps;
    const communicationTime = totalCommTime * gradientAccumulationSteps;

    // Optimizer step (memory-bandwidth-bound, once per gradient accumulation)
    let optimizerParamsPerGPU: number;
    if (ctx.lora) {
      optimizerParamsPerGPU = computeLoraTrainableParams(model, ctx.lora.rank, ctx.lora.targetModules) / pp;
    } else {
      optimizerParamsPerGPU = getOptimizerParamsPerGPU(
        model, training.dtypes.params,
        { tp: 1, pp: cluster.totalGPUs, dp: 1, ep: 1, dpType: 'ddp' }
      );
    }
    const optimizerTime = computeOptimizerStepTime(
      optimizerParamsPerGPU, gpu, training.optimizer.type, training.dtypes.params
    );

    const overlap = commOverlap * gradientAccumulationSteps;
    const total = forwardTime + backwardTime + bubbleTime * gradientAccumulationSteps +
                  communicationTime - overlap + optimizerTime;

    return {
      forward: forwardTime,
      backward: backwardTime + bubbleTime * gradientAccumulationSteps,
      optimizer: optimizerTime,
      communication: communicationTime,
      overlap,
      scaleOverhead: 0,
      total,
    };
  }

  validate(ctx: StrategyContext): StrategyValidation {
    const errors: string[] = [];
    const warnings: string[] = [];
    const pp = this.config.degree;
    const m = this.config.numMicroBatches;

    // PP degree must divide number of layers reasonably
    if (ctx.model.numLayers % pp !== 0) {
      warnings.push(
        `Model has ${ctx.model.numLayers} layers which doesn't divide evenly by PP=${pp}. ` +
        `Some stages will have unequal workloads.`
      );
    }

    // Interleaved schedule: layers must divide evenly by PP × v
    if (this.config.schedule === 'interleaved-1f1b') {
      const v = this.config.interleavedStages ?? 2;
      if (v < 2) {
        warnings.push(`Interleaved 1F1B requires at least 2 virtual stages (currently ${v}).`);
      }
      if (ctx.model.numLayers % (pp * v) !== 0) {
        warnings.push(`Model layers (${ctx.model.numLayers}) not divisible by PP×v (${pp}×${v}=${pp * v}). Some stages will have uneven layer counts.`);
      }
    }

    // Micro-batches must be >= pipeline stages for the pipeline to fill
    if (m < pp) {
      errors.push(
        `Pipeline cannot fill — not enough gradient accumulation steps for the number of pipeline stages. ` +
        `Increase global batch size or reduce pipeline stages.`
      );
    }

    // Check micro-batch count for efficiency
    const bubble = this.calculateBubble();
    if (bubble > 0.3) {
      warnings.push(
        `Pipeline bubble is high. ` +
        `Increase global batch size or reduce pipeline stages.`
      );
    }

    // Memory check
    const memory = this.computeMemoryPerGPU(ctx);
    const gpuMemoryBytes = gpuCapacityBytes(ctx.cluster.node.gpu.memoryGB);

    if (memory.total > gpuMemoryBytes) {
      errors.push(
        `OOM: requires ${(memory.total / (1024 ** 3)).toFixed(2)} GB per GPU but only ${ctx.cluster.node.gpu.memoryGB} GB available.`
      );
      // OOM suggestions handled by unified recommendation engine (generateRecommendations)
    }

    // Activation memory warning
    if (!this.config.activationCheckpointing && memory.activations > memory.parameters * 2) {
      warnings.push(
        'Activation memory is high. Consider enabling activation checkpointing.'
      );
    }

    // PP over slow interconnect warning
    if (ctx.cluster.numNodes > 1 && ctx.cluster.interNodeBandwidthGBps < 25) {
      warnings.push(
        'Pipeline parallelism over low-bandwidth interconnect may cause communication bottleneck.'
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
    const { cluster, model } = ctx;

    const pp = this.config.degree;
    const m = this.config.numMicroBatches;
    const bubble = this.calculateBubble();

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
        strategyName: `${this.name} (PP=${pp}, ${this.config.schedule})`,
      },
    });

    const microBatchFwdTime = timing.forward / m;
    const microBatchBwdTime = (timing.backward - timing.backward * bubble) / m;

    let maxTimestamp = 0;

    // Generate events based on schedule
    if (this.config.schedule === '1f1b' || this.config.schedule === 'interleaved-1f1b') {
      // 1F1B schedule
      for (let stage = 0; stage < pp; stage++) {
        let stageTime = 0;

        // Warmup: forward-only until pipeline fills
        const warmupMicrobatches = pp - stage - 1;
        for (let i = 0; i < Math.min(warmupMicrobatches, m); i++) {
          events.push({
            id: `stage${stage}-warmup-fwd-${i}`,
            type: 'phase-start',
            category: 'phase',
            timestamp: stageTime + stage * microBatchFwdTime,
            duration: microBatchFwdTime,
            gpuId: stage,
            phase: 'forward',
            stepNumber: 0,
            microBatchId: i,
          });
          stageTime += microBatchFwdTime;
        }

        // Steady state: interleaved 1F1B
        const steadyStateMicrobatches = m - warmupMicrobatches;
        for (let i = 0; i < steadyStateMicrobatches; i++) {
          const mbId = warmupMicrobatches + i;

          // Forward
          events.push({
            id: `stage${stage}-steady-fwd-${mbId}`,
            type: 'phase-start',
            category: 'phase',
            timestamp: stageTime + stage * microBatchFwdTime,
            duration: microBatchFwdTime,
            gpuId: stage,
            phase: 'forward',
            stepNumber: 0,
            microBatchId: mbId,
          });
          stageTime += microBatchFwdTime;

          // Backward for earlier microbatch
          events.push({
            id: `stage${stage}-steady-bwd-${i}`,
            type: 'phase-start',
            category: 'phase',
            timestamp: stageTime + stage * microBatchFwdTime,
            duration: microBatchBwdTime,
            gpuId: stage,
            phase: 'backward',
            stepNumber: 0,
            microBatchId: i,
          });
          stageTime += microBatchBwdTime;
        }

        // Cooldown: backward-only
        for (let i = 0; i < warmupMicrobatches && steadyStateMicrobatches + i < m; i++) {
          events.push({
            id: `stage${stage}-cooldown-bwd-${steadyStateMicrobatches + i}`,
            type: 'phase-start',
            category: 'phase',
            timestamp: stageTime + stage * microBatchFwdTime,
            duration: microBatchBwdTime,
            gpuId: stage,
            phase: 'backward',
            stepNumber: 0,
            microBatchId: steadyStateMicrobatches + i,
          });
          stageTime += microBatchBwdTime;
        }

        // Optimizer
        events.push({
          id: `stage${stage}-opt`,
          type: 'phase-start',
          category: 'phase',
          timestamp: stageTime + stage * microBatchFwdTime,
          duration: timing.optimizer,
          gpuId: stage,
          phase: 'optimizer',
          stepNumber: 0,
        });

        maxTimestamp = Math.max(maxTimestamp, stageTime + stage * microBatchFwdTime + timing.optimizer);
      }
    } else {
      // GPipe schedule (simple)
      for (let stage = 0; stage < pp; stage++) {
        // All forwards
        for (let mb = 0; mb < m; mb++) {
          events.push({
            id: `stage${stage}-fwd-${mb}`,
            type: 'phase-start',
            category: 'phase',
            timestamp: stage * microBatchFwdTime + mb * microBatchFwdTime / pp,
            duration: microBatchFwdTime,
            gpuId: stage,
            phase: 'forward',
            stepNumber: 0,
            microBatchId: mb,
          });
        }

        // All backwards
        const bwdStart = m * microBatchFwdTime + (pp - 1 - stage) * microBatchBwdTime;
        for (let mb = m - 1; mb >= 0; mb--) {
          events.push({
            id: `stage${stage}-bwd-${mb}`,
            type: 'phase-start',
            category: 'phase',
            timestamp: bwdStart + (m - 1 - mb) * microBatchBwdTime / pp,
            duration: microBatchBwdTime,
            gpuId: stage,
            phase: 'backward',
            stepNumber: 0,
            microBatchId: mb,
          });
        }

        maxTimestamp = Math.max(maxTimestamp, bwdStart + m * microBatchBwdTime + timing.optimizer);
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
  // Override to add pipeline bubble to analysis
  computeAnalysis(ctx: StrategyContext) {
    const analysis = super.computeAnalysis(ctx);
    analysis.pipelineBubble = this.calculateBubble();
    return analysis;
  }
}

// Export factory
export function createPipelineParallelStrategy(
  degree: number,
  numMicroBatches: number,
  schedule: PipelineSchedule = '1f1b',
  config?: Partial<PipelineParallelConfig>
): PipelineParallelStrategy {
  return new PipelineParallelStrategy({ ...config, degree, numMicroBatches, schedule });
}

