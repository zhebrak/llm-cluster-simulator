/**
 * Simulation Engine
 * Orchestrates the distributed training simulation
 */
// See docs/OVERVIEW.md for engine lifecycle and architecture.

import type {
  ModelSpec,
  ClusterConfig,
  TrainingHyperparams,
  SimulationResult,
  SimulationState,
  SimulationEvent,
  MemoryBreakdown,
  TimingBreakdown,
} from '../../types/index.ts';
import { getModel } from '../models/index.ts';
import { getPresetCluster, supportsFlashAttention, getPrecisionFallbackWarning, getEffectiveTFLOPS } from '../hardware/index.ts';
import {
  type StrategyContext,
  ParallelismStrategy,
  ddpStrategy,
  fsdpStrategy,
  zeroStage1Strategy,
  create3DParallelStrategy,
  createTensorParallelStrategy,
  createPipelineParallelStrategy,
  gpuCapacityBytes,
} from '../strategies/index.ts';
import { type DTypeConfig, DTYPE_PRESETS, DEFAULT_ADAMW_CONFIG, DEFAULT_LR_SCHEDULE } from '../../types/index.ts';
import { generateRecommendations } from './recommendations.ts';
import type { FinetuningMethod, LoraTargetModules } from '../strategies/lora.ts';
import { computeLoraTrainableParams } from '../strategies/lora.ts';

/**
 * Simulation configuration input
 */
export interface SimulationConfig {
  // Model
  modelId?: string;
  modelSpec?: ModelSpec;

  // Cluster
  clusterId?: string;
  clusterConfig?: ClusterConfig;

  // Training
  globalBatchSize: number;
  microBatchSize: number;
  sequenceLength: number;
  gradientAccumulationSteps?: number;
  maxSteps?: number;

  // Strategy
  strategyType: 'ddp' | 'fsdp' | 'zero-1' | 'zero-3' | 'auto'
    | 'fsdp-tp' | 'zero1-tp'
    | 'ddp-tp-pp' | 'zero1-tp-pp' | 'fsdp-tp-pp';
  strategyConfig?: {
    tp?: number;
    pp?: number;
    dp?: number;
    ep?: number;
    cp?: number;
    cpImplementation?: 'ring' | 'all-gather';
    numMicroBatches?: number;
    dpType?: 'ddp' | 'fsdp' | 'zero-1' | 'zero-2' | 'zero-3';
    sequenceParallel?: boolean;
    pipelineSchedule?: '1f1b' | 'interleaved-1f1b' | 'dualpipe-v';
    interleavedStages?: number;
  };

  // Options
  activationCheckpointing?: boolean;
  checkpointingGranularity?: 'full' | 'selective';
  selectiveStoredLayers?: number;  // undefined = auto (strategies resolve internally)
  flashAttention?: boolean;
  mixedPrecision?: 'fp32' | 'tf32' | 'fp16' | 'bf16' | 'fp8' | 'fp4';

  // Fine-tuning
  finetuningMethod?: FinetuningMethod;
  loraRank?: number;
  loraTargetModules?: LoraTargetModules;
}

/**
 * Simulation metrics
 */
export interface SimulationMetrics {
  // Throughput
  tokensPerSecond: number;
  samplesPerSecond: number;
  tflopsPerGPU: number;

  // Efficiency
  mfu: number;
  hfu: number;
  communicationOverhead: number;
  pipelineBubble: number;

  // Memory
  memoryPerGPU: MemoryBreakdown;
  memoryUtilization: number;

  // Timing
  stepTimeMs: number;
  timing: TimingBreakdown;

  // Training projection
  timeToTrainHours?: number;
  totalCost?: number;

  // Model FLOPs MFU: uses actual flopsPerToken (incl. quadratic attention)
  // instead of 6P approximation. Only populated when divergence > 10%.
  modelFlopsMfu?: number;

  // Communication reporting
  communicationGrossMs: number;
  communicationExposedMs: number;
  overlapHiddenFraction: number;

  // FP8 hardware utilization: MFU recomputed against FP8 peak TFLOPS.
  // Only populated when training in FP8 on a GPU with FP8 support.
  fp8HwUtil?: number;

  // Resolved stored-layers count for selective AC
  resolvedStoredLayers?: number;
}

/**
 * Main simulation engine
 */
export class SimulationEngine {
  private config: SimulationConfig | null = null;
  private model: ModelSpec | null = null;
  private cluster: ClusterConfig | null = null;
  private strategy: ParallelismStrategy | null = null;
  private context: StrategyContext | null = null;

  /**
   * Configure the simulation
   */
  configure(config: SimulationConfig): void {
    this.config = config;

    // Validate core training inputs
    if (config.globalBatchSize <= 0) throw new Error(`globalBatchSize must be positive, got ${config.globalBatchSize}`);
    if (config.microBatchSize <= 0) throw new Error(`microBatchSize must be positive, got ${config.microBatchSize}`);
    if (config.sequenceLength <= 0) throw new Error(`sequenceLength must be positive, got ${config.sequenceLength}`);

    // Resolve model
    if (config.modelSpec) {
      this.model = config.modelSpec;
    } else if (config.modelId) {
      const model = getModel(config.modelId, config.sequenceLength);
      if (!model) {
        throw new Error(`Model not found: ${config.modelId}`);
      }
      this.model = model;
    } else {
      throw new Error('Either modelId or modelSpec must be provided');
    }

    // Resolve cluster
    if (config.clusterConfig) {
      this.cluster = config.clusterConfig;
    } else if (config.clusterId) {
      const cluster = getPresetCluster(config.clusterId);
      if (!cluster) {
        throw new Error(`Cluster preset not found: ${config.clusterId}`);
      }
      this.cluster = cluster;
    } else {
      throw new Error('Either clusterId or clusterConfig must be provided');
    }

    // Select strategy
    this.strategy = this.selectStrategy(config);

    // Build context
    this.context = this.buildContext(config);
  }

  /**
   * Select appropriate strategy based on config
   */
  private selectStrategy(config: SimulationConfig): ParallelismStrategy {
    const strategyConfig = config.strategyConfig ?? {};
    const ac = config.activationCheckpointing ?? true;
    const acg = config.checkpointingGranularity ?? 'full';
    const cp = strategyConfig.cp ?? 1;
    const cpImpl = strategyConfig.cpImplementation ?? 'ring';

    switch (config.strategyType) {
      case 'ddp':
        // 1D strategies with CP > 1: route through 3D to get CP support
        if (cp > 1) {
          const dp = Math.floor(this.cluster!.totalGPUs / cp);
          return create3DParallelStrategy(1, 1, dp, {
            cp,
            cpImplementation: cpImpl,
            dpType: 'ddp',
            activationCheckpointing: ac,
            checkpointingGranularity: acg,
          });
        }
        return ddpStrategy;

      case 'fsdp':
        if (cp > 1) {
          const dp = Math.floor(this.cluster!.totalGPUs / cp);
          return create3DParallelStrategy(1, 1, dp, {
            cp,
            cpImplementation: cpImpl,
            dpType: 'fsdp',
            activationCheckpointing: ac,
            checkpointingGranularity: acg,
          });
        }
        return fsdpStrategy;

      case 'zero-1':
        if (cp > 1) {
          const dp = Math.floor(this.cluster!.totalGPUs / cp);
          return create3DParallelStrategy(1, 1, dp, {
            cp,
            cpImplementation: cpImpl,
            dpType: 'zero-1',
            activationCheckpointing: ac,
            checkpointingGranularity: acg,
          });
        }
        return zeroStage1Strategy;

      case 'zero-3':
        if (cp > 1) {
          const dp = Math.floor(this.cluster!.totalGPUs / cp);
          return create3DParallelStrategy(1, 1, dp, {
            cp,
            cpImplementation: cpImpl,
            dpType: 'fsdp',
            activationCheckpointing: ac,
            checkpointingGranularity: acg,
          });
        }
        return fsdpStrategy;  // ZeRO-3 ≡ FSDP (same full-sharding semantics)

      // Hybrid Strategies
      case 'fsdp-tp': {
        const tp = strategyConfig.tp ?? Math.min(8, this.cluster!.gpusPerNode);
        if (tp <= 0) throw new Error(`TP must be positive, got ${tp}`);
        const ep = (this.model!.isMoE && strategyConfig.ep) ? strategyConfig.ep : 1;
        const dp = Math.floor(this.cluster!.totalGPUs / (tp * cp));
        if (dp <= 0) throw new Error(`DP must be positive (TP=${tp} × CP=${cp} exceeds ${this.cluster!.totalGPUs} GPUs)`);
        return create3DParallelStrategy(tp, 1, dp, {
          ep,
          cp,
          cpImplementation: cpImpl,
          dpType: 'fsdp',
          sequenceParallel: strategyConfig.sequenceParallel ?? true,
          activationCheckpointing: ac,
          checkpointingGranularity: acg,
        });
      }

      case 'zero1-tp': {
        const tp = strategyConfig.tp ?? Math.min(8, this.cluster!.gpusPerNode);
        if (tp <= 0) throw new Error(`TP must be positive, got ${tp}`);
        const ep = (this.model!.isMoE && strategyConfig.ep) ? strategyConfig.ep : 1;
        const dp = Math.floor(this.cluster!.totalGPUs / (tp * cp));
        if (dp <= 0) throw new Error(`DP must be positive (TP=${tp} × CP=${cp} exceeds ${this.cluster!.totalGPUs} GPUs)`);
        return create3DParallelStrategy(tp, 1, dp, {
          ep,
          cp,
          cpImplementation: cpImpl,
          dpType: 'zero-1',
          sequenceParallel: strategyConfig.sequenceParallel ?? true,
          activationCheckpointing: ac,
          checkpointingGranularity: acg,
        });
      }

      case 'ddp-tp-pp': {
        const tp = strategyConfig.tp ?? Math.min(8, this.cluster!.gpusPerNode);
        if (tp <= 0) throw new Error(`TP must be positive, got ${tp}`);
        const pp = strategyConfig.pp ?? Math.min(8, Math.floor(this.cluster!.totalGPUs / tp));
        if (pp <= 0) throw new Error(`PP must be positive, got ${pp}`);
        const ep = (this.model!.isMoE && strategyConfig.ep) ? strategyConfig.ep : 1;
        const dp = Math.floor(this.cluster!.totalGPUs / (tp * pp * cp));
        if (dp <= 0) throw new Error(`DP must be positive (TP=${tp} × PP=${pp} × CP=${cp} exceeds ${this.cluster!.totalGPUs} GPUs)`);
        const ga = Math.max(1, Math.ceil(config.globalBatchSize / (config.microBatchSize * dp)));
        const schedule = strategyConfig.pipelineSchedule ?? '1f1b';
        return create3DParallelStrategy(tp, pp, dp, {
          ep,
          cp,
          cpImplementation: cpImpl,
          dpType: 'ddp',
          numMicroBatches: strategyConfig.numMicroBatches ?? ga,
          schedule,
          interleavedStages: schedule === 'interleaved-1f1b' ? (strategyConfig.interleavedStages ?? 2) : 1,
          sequenceParallel: strategyConfig.sequenceParallel ?? true,
          activationCheckpointing: ac,
          checkpointingGranularity: acg,
        });
      }

      case 'zero1-tp-pp': {
        const tp = strategyConfig.tp ?? Math.min(8, this.cluster!.gpusPerNode);
        if (tp <= 0) throw new Error(`TP must be positive, got ${tp}`);
        const pp = strategyConfig.pp ?? Math.min(8, Math.floor(this.cluster!.totalGPUs / tp));
        if (pp <= 0) throw new Error(`PP must be positive, got ${pp}`);
        const ep = (this.model!.isMoE && strategyConfig.ep) ? strategyConfig.ep : 1;
        const dp = Math.floor(this.cluster!.totalGPUs / (tp * pp * cp));
        if (dp <= 0) throw new Error(`DP must be positive (TP=${tp} × PP=${pp} × CP=${cp} exceeds ${this.cluster!.totalGPUs} GPUs)`);
        const ga = Math.max(1, Math.ceil(config.globalBatchSize / (config.microBatchSize * dp)));
        const schedule = strategyConfig.pipelineSchedule ?? '1f1b';
        return create3DParallelStrategy(tp, pp, dp, {
          ep,
          cp,
          cpImplementation: cpImpl,
          dpType: 'zero-1',
          numMicroBatches: strategyConfig.numMicroBatches ?? ga,
          schedule,
          interleavedStages: schedule === 'interleaved-1f1b' ? (strategyConfig.interleavedStages ?? 2) : 1,
          sequenceParallel: strategyConfig.sequenceParallel ?? true,
          activationCheckpointing: ac,
          checkpointingGranularity: acg,
        });
      }

      case 'fsdp-tp-pp': {
        const tp = strategyConfig.tp ?? Math.min(8, this.cluster!.gpusPerNode);
        if (tp <= 0) throw new Error(`TP must be positive, got ${tp}`);
        const pp = strategyConfig.pp ?? Math.min(4, Math.floor(this.cluster!.totalGPUs / tp));
        if (pp <= 0) throw new Error(`PP must be positive, got ${pp}`);
        const ep = (this.model!.isMoE && strategyConfig.ep) ? strategyConfig.ep : 1;
        const dp = Math.floor(this.cluster!.totalGPUs / (tp * pp * cp));
        if (dp <= 0) throw new Error(`DP must be positive (TP=${tp} × PP=${pp} × CP=${cp} exceeds ${this.cluster!.totalGPUs} GPUs)`);
        const ga = Math.max(1, Math.ceil(config.globalBatchSize / (config.microBatchSize * dp)));
        const schedule = strategyConfig.pipelineSchedule ?? '1f1b';
        return create3DParallelStrategy(tp, pp, dp, {
          ep,
          cp,
          cpImplementation: cpImpl,
          dpType: 'fsdp',
          numMicroBatches: strategyConfig.numMicroBatches ?? ga,
          schedule,
          interleavedStages: schedule === 'interleaved-1f1b' ? (strategyConfig.interleavedStages ?? 2) : 1,
          sequenceParallel: strategyConfig.sequenceParallel ?? true,
          activationCheckpointing: ac,
          checkpointingGranularity: acg,
        });
      }

      case 'auto':
        return this.autoSelectStrategy();

      default:
        return ddpStrategy;
    }
  }

  /**
   * Auto-select best strategy based on model and cluster
   */
  private autoSelectStrategy(): ParallelismStrategy {
    if (!this.model || !this.cluster) {
      return ddpStrategy;
    }

    const modelParams = this.model.totalParams;
    const gpuMemoryGB = this.cluster.node.gpu.memoryGB;
    const totalGPUs = this.cluster.totalGPUs;
    const gpusPerNode = this.cluster.gpusPerNode;

    // Estimate memory per GPU with DDP
    const estimatedMemoryGB = (modelParams * 18) / (1024 ** 3); // ~18 bytes/param for training

    // MoE-specific auto-selection: prioritize EP for expert distribution, then TP
    if (this.model.isMoE && this.model.numExperts) {
      // For MoE, start with modest TP (experts don't need TP), maximize EP
      const numExperts = this.model.numExperts;
      // Try TP = 1, 2, 4 (prefer smaller TP for MoE since experts are split by EP)
      for (const tp of [1, 2, 4]) {
        if (tp > gpusPerNode) continue;
        const maxEP = Math.floor(gpusPerNode / tp);
        // Find largest EP that divides numExperts and fits in node
        let ep = 1;
        for (let e = maxEP; e >= 1; e--) {
          if (numExperts % e === 0) { ep = e; break; }
        }
        const dp = Math.floor(totalGPUs / tp);
        if (dp >= 1 && dp % ep === 0) {
          return create3DParallelStrategy(tp, 1, dp, {
            ep,
            dpType: 'fsdp',
            sequenceParallel: tp > 1,
          });
        }
      }
      // Fallback: TP=1, EP=1, all DP
      return create3DParallelStrategy(1, 1, totalGPUs, {
        ep: 1,
        dpType: 'fsdp',
      });
    }

    // If model fits in single GPU, use DDP
    if (estimatedMemoryGB < gpuMemoryGB * 0.7) {
      return ddpStrategy;
    }

    // If model fits with FSDP, use FSDP
    if (estimatedMemoryGB / totalGPUs < gpuMemoryGB * 0.7) {
      return fsdpStrategy;
    }

    // Need model parallelism
    // Start with TP within node
    const tp = Math.min(gpusPerNode, 8);

    // If still doesn't fit, add PP
    if (estimatedMemoryGB / tp < gpuMemoryGB * 0.7) {
      return createTensorParallelStrategy(tp, { sequenceParallel: true });
    }

    let pp = Math.ceil(estimatedMemoryGB / (tp * gpuMemoryGB * 0.5));
    pp = Math.min(pp, Math.floor(totalGPUs / tp));

    const dp = Math.floor(totalGPUs / (tp * pp));

    if (dp >= 1) {
      return create3DParallelStrategy(tp, pp, dp, {
        numMicroBatches: Math.max(8, pp * 2),
      });
    }

    // Fallback to max PP
    return createPipelineParallelStrategy(
      Math.floor(totalGPUs / tp),
      16,
      'interleaved-1f1b'
    );
  }

  /**
   * Get effective parallelism degrees, mirroring selectStrategy's auto-selection
   * defaults. Without this, buildContext would default to tp=1/pp=1 when
   * strategyConfig omits them, while selectStrategy auto-selects tp=gpusPerNode.
   */
  private getEffectiveParallelism(config: SimulationConfig): { tp: number; pp: number; ep: number; cp: number } {
    const strategyConfig = config.strategyConfig ?? {};
    const gpusPerNode = this.cluster!.gpusPerNode;
    const totalGPUs = this.cluster!.totalGPUs;
    const isMoE = this.model!.isMoE;
    const ep = (isMoE && strategyConfig.ep) ? strategyConfig.ep : 1;
    const cp = strategyConfig.cp ?? 1;

    switch (config.strategyType) {
      case 'fsdp-tp':
      case 'zero1-tp': {
        const tp = strategyConfig.tp ?? Math.min(8, gpusPerNode);
        return { tp, pp: 1, ep, cp };
      }
      case 'ddp-tp-pp':
      case 'zero1-tp-pp': {
        const tp = strategyConfig.tp ?? Math.min(8, gpusPerNode);
        const pp = strategyConfig.pp ?? Math.min(8, Math.floor(totalGPUs / tp));
        return { tp, pp, ep, cp };
      }
      case 'fsdp-tp-pp': {
        const tp = strategyConfig.tp ?? Math.min(8, gpusPerNode);
        const pp = strategyConfig.pp ?? Math.min(4, Math.floor(totalGPUs / tp));
        return { tp, pp, ep, cp };
      }
      default: {
        return { tp: strategyConfig.tp ?? 1, pp: strategyConfig.pp ?? 1, ep, cp };
      }
    }
  }

  /**
   * Build strategy context
   */
  private buildContext(config: SimulationConfig): StrategyContext {
    if (!this.model || !this.cluster) {
      throw new Error('Model and cluster must be configured');
    }

    // Calculate effective data parallelism accounting for TP, PP, and CP (EP subdivides DP, not the GPU product)
    const { tp, pp, cp } = this.getEffectiveParallelism(config);
    const effectiveDP = Math.max(1, Math.floor(this.cluster.totalGPUs / (tp * pp * cp)));

    const gradientAccumulationSteps = config.gradientAccumulationSteps ??
      Math.ceil(config.globalBatchSize / (config.microBatchSize * effectiveDP));

    const dtypePreset = config.mixedPrecision ?? 'bf16';
    const dtypes: DTypeConfig = DTYPE_PRESETS[dtypePreset] ?? DTYPE_PRESETS.bf16;

    const training: TrainingHyperparams = {
      globalBatchSize: config.globalBatchSize,
      microBatchSize: config.microBatchSize,
      sequenceLength: config.sequenceLength,
      maxSteps: config.maxSteps ?? 1000,
      optimizer: DEFAULT_ADAMW_CONFIG,
      lrSchedule: DEFAULT_LR_SCHEDULE,
      dtypes,
      gradientClipping: 1.0,
      gradientAccumulationSteps,
    };

    // Build LoRA config if fine-tuning method is LoRA/QLoRA
    const lora = config.finetuningMethod && config.finetuningMethod !== 'full'
      ? {
          method: config.finetuningMethod,
          rank: config.loraRank ?? 16,
          targetModules: config.loraTargetModules ?? 'q_k_v_o' as LoraTargetModules,
        }
      : undefined;

    return {
      model: this.model,
      cluster: this.cluster,
      training,
      seqLength: config.sequenceLength,
      microBatchSize: config.microBatchSize,
      globalBatchSize: config.globalBatchSize,
      gradientAccumulationSteps,
      activationCheckpointing: config.activationCheckpointing ?? true,
      checkpointingGranularity: config.checkpointingGranularity ?? 'full',
      selectiveStoredLayers: config.selectiveStoredLayers,
      flashAttention: (config.flashAttention ?? true) && supportsFlashAttention(this.cluster.node.gpu),
      lora,
    };
  }

  /**
   * Validate configuration
   */
  validate(): { valid: boolean; errors: string[]; warnings: string[]; suggestions: string[] } {
    if (!this.strategy || !this.context) {
      return { valid: false, errors: ['Simulation not configured'], warnings: [], suggestions: [] };
    }

    const validation = this.strategy.validate(this.context);
    const warnings = [...validation.warnings];

    // Add precision fallback warning if GPU doesn't natively support selected precision
    const computeDtype = this.context.training.dtypes.compute;
    const gpu = this.context.cluster.node.gpu;
    const fallbackWarning = getPrecisionFallbackWarning(gpu, computeDtype);
    if (fallbackWarning) {
      warnings.push(fallbackWarning);
    }

    return {
      valid: validation.valid,
      errors: validation.errors,
      warnings,
      suggestions: validation.suggestions,
    };
  }

  /**
   * Run simulation and get metrics
   */
  simulate(): SimulationMetrics {
    if (!this.strategy || !this.context || !this.model || !this.cluster) {
      throw new Error('Simulation not configured');
    }

    const analysis = this.strategy.computeAnalysis(this.context);

    // Calculate step time
    const stepTimeMs = analysis.timing.total;
    const stepTimeSeconds = stepTimeMs / 1000;

    // Throughput metrics
    const tokensPerStep = this.context.globalBatchSize * this.context.seqLength;
    const tokensPerSecond = tokensPerStep / stepTimeSeconds;
    const samplesPerSecond = this.context.globalBatchSize / stepTimeSeconds;

    // TFLOPS calculation using standard 6*P*T formula
    // Training FLOPS = 6 * activeParams * tokens (industry standard)
    const activeParams = this.model.activeParams ?? this.model.totalParams;
    const flopsPerStep = 6 * activeParams * tokensPerStep;
    const totalTFLOPS = flopsPerStep / stepTimeSeconds / 1e12;
    const tflopsPerGPU = totalTFLOPS / this.cluster.totalGPUs;

    // Memory utilization
    const gpuMemoryBytes = gpuCapacityBytes(this.cluster.node.gpu.memoryGB);
    const memoryUtilization = analysis.memory.total / gpuMemoryBytes;

    // Training time projection
    let timeToTrainHours: number | undefined;
    if (this.config?.maxSteps) {
      const totalSeconds = this.config.maxSteps * stepTimeSeconds;
      timeToTrainHours = totalSeconds / 3600;
    }

    // Model FLOPs MFU: uses actual flopsPerToken (includes quadratic attention).
    // For LoRA, adapter FLOPs use the 2A parameter-count approximation (exact for
    // adapter matmuls: two linear projections with no attention scaling or FFN gating).
    // The base model uses architecture-aware flopsPerToken — this asymmetry is intentional
    // since adapters are architecturally simpler than the base model's attention/MLP layers.
    const baseFlopsPerStep = this.context.lora
      ? 2 * this.model.flopsPerToken * tokensPerStep
      : 3 * this.model.flopsPerToken * tokensPerStep;

    const adapterFlopsPerStep = this.context.lora
      ? 6 * computeLoraTrainableParams(this.model, this.context.lora.rank, this.context.lora.targetModules) * tokensPerStep
      : 0;

    const modelFlopsPerStep = baseFlopsPerStep + adapterFlopsPerStep;
    const totalGPUTFLOPS = this.cluster.totalGPUs * getEffectiveTFLOPS(this.cluster.node.gpu, 'bf16');
    const modelFlopsMfu = modelFlopsPerStep / (stepTimeSeconds * totalGPUTFLOPS * 1e12);
    const showModelFlopsMfu = analysis.mfu > 0 && modelFlopsMfu / analysis.mfu > 1.10;

    // FP8 hardware utilization: MFU scaled by BF16/FP8 peak ratio
    const computeDtype = this.config!.mixedPrecision ?? 'bf16';
    const fp8Peak = this.cluster.node.gpu.fp8TFLOPS;
    const fp8HwUtil = (computeDtype === 'fp8' && fp8Peak > 0)
      ? analysis.mfu * getEffectiveTFLOPS(this.cluster.node.gpu, 'bf16') / fp8Peak
      : undefined;

    // Safety net: catch any NaN/Infinity that slipped through upstream
    if (!Number.isFinite(stepTimeMs)) throw new Error(`stepTimeMs is not finite: ${stepTimeMs}`);
    if (!Number.isFinite(tokensPerSecond)) throw new Error(`tokensPerSecond is not finite: ${tokensPerSecond}`);
    if (!Number.isFinite(memoryUtilization)) throw new Error(`memoryUtilization is not finite: ${memoryUtilization}`);
    if (!Number.isFinite(analysis.mfu)) throw new Error(`MFU is not finite: ${analysis.mfu}`);

    return {
      tokensPerSecond,
      samplesPerSecond,
      tflopsPerGPU,
      mfu: analysis.mfu,
      hfu: analysis.hfu,
      communicationOverhead: analysis.communicationOverhead,
      pipelineBubble: analysis.pipelineBubble,
      communicationGrossMs: analysis.communicationGrossMs,
      communicationExposedMs: analysis.communicationExposedMs,
      overlapHiddenFraction: analysis.overlapHiddenFraction,
      memoryPerGPU: analysis.memory,
      memoryUtilization,
      stepTimeMs,
      timing: analysis.timing,
      timeToTrainHours,
      ...(showModelFlopsMfu ? { modelFlopsMfu } : {}),
      ...(fp8HwUtil != null ? { fp8HwUtil } : {}),
      ...(analysis.resolvedStoredLayers != null ? { resolvedStoredLayers: analysis.resolvedStoredLayers } : {}),
    };
  }

  /**
   * Generate simulation events for visualization
   */
  generateEvents(): SimulationEvent[] {
    if (!this.strategy || !this.context) {
      throw new Error('Simulation not configured');
    }

    return this.strategy.generateEvents(this.context);
  }

  /**
   * Get full simulation result
   */
  run(): SimulationResult {
    const validation = this.validate();

    // Always simulate and generate recommendations — even for OOM configs.
    // simulate() produces theoretical metrics (with memoryUtilization > 1.0 for OOM),
    // and the recommendation engine handles OOM via validated suggestions that
    // confirm the fix actually resolves the memory issue.
    const metrics = this.simulate();
    const ctx = this.context!;
    const recommendations = generateRecommendations(this.config!, ctx, metrics, validation.errors);

    if (!validation.valid) {
      return {
        success: false,
        state: {
          status: 'error',
          progress: 0,
          currentStep: 0,
          totalSteps: 0,
          currentTimeMs: 0,
          totalTimeMs: 0,
          playbackSpeed: 1,
          events: { events: [], currentIndex: 0, isComplete: false, totalDurationMs: 0 },
          timelines: [],
          error: validation.errors.join('; '),
        },
        events: [],
        metrics: {
          avgStepTimeMs: 0,
          minStepTimeMs: 0,
          maxStepTimeMs: 0,
          totalTimeMs: 0,
          tokensPerSecond: 0,
          samplesPerSecond: 0,
          mfu: 0,
          hfu: 0,
          communicationOverhead: 0,
          pipelineBubble: 0,
          peakMemoryGB: +(metrics.memoryPerGPU.total / (1024 ** 3)).toFixed(2),
        },
        analysis: {
          bottleneck: 'memory',
          recommendations,
        },
      };
    }

    const events = this.generateEvents();

    // Determine bottleneck
    let bottleneck: 'compute' | 'memory' | 'communication' = 'compute';
    if (metrics.memoryUtilization > 0.9) {
      bottleneck = 'memory';
    } else if (metrics.communicationOverhead > 0.3) {
      bottleneck = 'communication';
    }

    const state: SimulationState = {
      status: 'complete',
      progress: 1,
      currentStep: this.config?.maxSteps ?? 1,
      totalSteps: this.config?.maxSteps ?? 1,
      currentTimeMs: metrics.stepTimeMs,
      totalTimeMs: metrics.stepTimeMs,
      playbackSpeed: 1,
      events: {
        events,
        currentIndex: events.length,
        isComplete: true,
        totalDurationMs: metrics.stepTimeMs,
      },
      timelines: [],
    };

    return {
      success: true,
      state,
      events,
      metrics: {
        avgStepTimeMs: Math.round(metrics.stepTimeMs),
        minStepTimeMs: Math.round(metrics.stepTimeMs),
        maxStepTimeMs: Math.round(metrics.stepTimeMs),
        totalTimeMs: Math.round(metrics.stepTimeMs * (this.config?.maxSteps ?? 1)),
        tokensPerSecond: Math.round(metrics.tokensPerSecond),
        samplesPerSecond: Math.round(metrics.samplesPerSecond),
        mfu: +metrics.mfu.toFixed(2),
        hfu: +metrics.hfu.toFixed(2),
        communicationOverhead: +metrics.communicationOverhead.toFixed(2),
        pipelineBubble: +metrics.pipelineBubble.toFixed(2),
        peakMemoryGB: +(metrics.memoryPerGPU.total / (1024 ** 3)).toFixed(2),
      },
      analysis: {
        bottleneck,
        recommendations,
      },
    };
  }

  /**
   * Get current configuration summary
   */
  getSummary(): {
    model: string;
    params: string;
    cluster: string;
    gpus: number;
    strategy: string;
  } | null {
    if (!this.model || !this.cluster || !this.strategy) {
      return null;
    }

    const formatParams = (n: number) => {
      if (n >= 1e12) return `${(n / 1e12).toFixed(1)}T`;
      if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
      if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
      return `${(n / 1e3).toFixed(1)}K`;
    };

    return {
      model: this.model.name,
      params: formatParams(this.model.totalParams),
      cluster: this.cluster.name,
      gpus: this.cluster.totalGPUs,
      strategy: this.strategy.name,
    };
  }
}

// Singleton instance for simple usage
export const simulationEngine = new SimulationEngine();

/**
 * Quick simulation function
 */
export function runSimulation(config: SimulationConfig): SimulationResult {
  const engine = new SimulationEngine();
  engine.configure(config);
  return engine.run();
}

/**
 * Quick metrics function
 */
export function getSimulationMetrics(config: SimulationConfig): SimulationMetrics {
  const engine = new SimulationEngine();
  engine.configure(config);
  return engine.simulate();
}
