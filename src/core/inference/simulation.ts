/**
 * Inference Simulation Engine
 *
 * Orchestrates the LLM inference simulation, combining:
 * - Memory calculations
 * - Latency modeling
 * - Speculative decoding
 * - Event generation for visualization
 */

import type { ModelSpec } from '../../types/model.ts';
import { bytesToGB } from '../../types/base.ts';
import { gpuCapacityBytes } from '../strategies/base.ts';
import type { GPUSpec } from '../../types/hardware.ts';
import type {
  InferencePrecision,
  InferenceConfig,
  InferenceSimulationResult,
  SpeculativeMetrics,
  InferenceEvent,
  TokenGeneratedEvent,
  KVCacheUpdateEvent,
  KVCacheState,
  InferenceMemoryEvent,
  PhaseEvent,
  SpeculativeDraftEvent,
  SpeculativeVerifyEvent,
  SpeculativeResultEvent,
} from '../../types/inference.ts';
import { DEFAULT_INFERENCE_CONFIG } from '../../types/inference.ts';
import { calculateMemoryFromConfig, validateMemoryFits, modelWeightsMemory } from './memory.ts';
import { totalKVCacheMemory } from './kv-cache.ts';
import {
  calculateLatencyMetrics,
  calculateMetricsFromConfig,
  calculateContinuousBatchingMetrics,
  cbSchedulingOverhead,
  estimateTPOT,
} from './latency.ts';
import { calculateSpeculativeMetrics, simulateSpeculativeDecoding } from './speculative.ts';
import { getModel } from '../models/index.ts';
import { ALL_GPUS, H100_SXM, supportsFlashAttention } from '../hardware/gpu.ts';
import { generateInferenceRecommendations } from './recommendations.ts';

/**
 * Inference simulation configuration input
 */
export interface InferenceSimulationConfig {
  // Model
  modelId?: string;
  modelSpec?: ModelSpec;

  // Hardware
  gpuId?: string;
  gpu?: GPUSpec;
  numGPUs?: number;

  // Inference parameters
  batchSize?: number;
  inputSeqLen?: number;
  outputSeqLen?: number;

  // Precision
  weightPrecision?: InferencePrecision;
  kvCachePrecision?: InferencePrecision;

  // Optimizations
  flashAttention?: boolean;
  pagedAttention?: boolean;
  tensorParallel?: number;
  expertParallel?: number;

  // Batching
  continuousBatching?: boolean;

  // Speculative decoding
  speculativeEnabled?: boolean;
  draftModelId?: string;
  draftModelSpec?: ModelSpec;
  numSpeculativeTokens?: number;
  acceptanceRate?: number;
}

/**
 * Main Inference Simulation Engine
 */
export class InferenceSimulationEngine {
  private config: InferenceConfig | null = null;
  private inputConfig: InferenceSimulationConfig | null = null;
  private numReplicas = 1;
  private batchPerReplica = 1;

  /**
   * Configure the simulation
   */
  configure(input: InferenceSimulationConfig): void {
    this.inputConfig = input;

    // Resolve model
    let modelSpec: ModelSpec;
    if (input.modelSpec) {
      modelSpec = input.modelSpec;
    } else if (input.modelId) {
      const model = getModel(input.modelId, input.inputSeqLen ?? 512);
      if (!model) {
        throw new Error(`Model not found: ${input.modelId}`);
      }
      modelSpec = model;
    } else {
      throw new Error('Either modelId or modelSpec must be provided');
    }

    // Resolve GPU
    let gpu: GPUSpec;
    if (input.gpu) {
      gpu = input.gpu;
    } else if (input.gpuId) {
      const resolved = ALL_GPUS[input.gpuId];
      if (!resolved) throw new Error(`GPU not found: ${input.gpuId}`);
      gpu = resolved;
    } else {
      // Default to H100 SXM
      gpu = H100_SXM;
    }

    // Resolve draft model for speculative decoding
    let draftModel: ModelSpec | null = null;
    if (input.speculativeEnabled) {
      if (input.draftModelSpec) {
        draftModel = input.draftModelSpec;
      } else if (input.draftModelId) {
        draftModel = getModel(input.draftModelId, input.inputSeqLen ?? 512) ?? null;
      }
    }

    // Validate GPU specs
    if (gpu.memoryGB <= 0) throw new Error(`GPU ${gpu.name} has invalid memory: ${gpu.memoryGB}GB`);
    if (gpu.memoryBandwidthTBps <= 0) throw new Error(`GPU ${gpu.name} has invalid bandwidth: ${gpu.memoryBandwidthTBps} TB/s`);

    // Build config
    const numGPUs = input.numGPUs ?? 1;
    const batchSize = input.batchSize ?? DEFAULT_INFERENCE_CONFIG.batchSize!;
    const inputSeqLen = input.inputSeqLen ?? DEFAULT_INFERENCE_CONFIG.inputSeqLen!;
    const outputSeqLen = input.outputSeqLen ?? DEFAULT_INFERENCE_CONFIG.outputSeqLen!;

    // Validate inputs
    if (numGPUs <= 0) throw new Error(`numGPUs must be positive, got ${numGPUs}`);
    if (batchSize <= 0) throw new Error(`batchSize must be positive, got ${batchSize}`);
    if (inputSeqLen <= 0) throw new Error(`inputSeqLen must be positive, got ${inputSeqLen}`);
    if (outputSeqLen <= 0) throw new Error(`outputSeqLen must be positive, got ${outputSeqLen}`);

    this.config = {
      modelSpec,
      gpu,
      numGPUs,
      batchSize,
      inputSeqLen,
      outputSeqLen,
      weightPrecision: input.weightPrecision ?? DEFAULT_INFERENCE_CONFIG.weightPrecision!,
      kvCachePrecision: input.kvCachePrecision ?? DEFAULT_INFERENCE_CONFIG.kvCachePrecision!,
      flashAttention: (input.flashAttention ?? DEFAULT_INFERENCE_CONFIG.flashAttention!) && supportsFlashAttention(gpu),
      pagedAttention: input.pagedAttention ?? DEFAULT_INFERENCE_CONFIG.pagedAttention!,
      continuousBatching: input.continuousBatching ?? false,
      tensorParallel: input.tensorParallel,
      expertParallel: input.expertParallel,
      speculative: {
        enabled: input.speculativeEnabled ?? false,
        draftModel,
        numSpeculativeTokens: input.numSpeculativeTokens ?? DEFAULT_INFERENCE_CONFIG.speculative!.numSpeculativeTokens,
        acceptanceRate: input.acceptanceRate ?? DEFAULT_INFERENCE_CONFIG.speculative!.acceptanceRate,
      },
    };

    // Compute replicas: numGPUs / (TP × EP) gives independent serving replicas.
    // Unlike training where EP subdivides DP, in inference EP and TP are independent
    // dimensions — each replica needs tp GPUs for tensor sharding AND ep GPUs for
    // expert distribution, with no overlap between the two.
    const tp = this.config.tensorParallel ?? 1;
    const ep = this.config.expertParallel ?? 1;
    const gpusPerReplica = Math.max(1, tp) * Math.max(1, ep);
    this.numReplicas = Math.max(1, Math.floor(this.config.numGPUs / gpusPerReplica));
    this.batchPerReplica = Math.max(1, Math.ceil(this.config.batchSize / this.numReplicas));
  }

  /**
   * Validate configuration
   */
  validate(): { valid: boolean; errors: string[]; warnings: string[] } {
    if (!this.config) {
      return { valid: false, errors: ['Simulation not configured'], warnings: [] };
    }

    const errors: string[] = [];
    const warnings: string[] = [];

    // Check memory fit (per-replica batch size), including draft model if speculative decoding is enabled
    const replicaConfig = { ...this.config, batchSize: this.batchPerReplica };
    const memoryValidation = validateMemoryFits(replicaConfig, this.config.gpu);
    let totalMemory = memoryValidation.memory.total;
    if (this.config.speculative.enabled && this.config.speculative.draftModel) {
      const draftTP = this.config.tensorParallel ?? 1;
      totalMemory += modelWeightsMemory(this.config.speculative.draftModel, this.config.weightPrecision) / draftTP;
      const totalSeqLen = this.config.inputSeqLen + this.config.outputSeqLen;
      totalMemory += totalKVCacheMemory(this.config.speculative.draftModel, totalSeqLen, this.batchPerReplica, this.config.kvCachePrecision) / draftTP;
    }
    const gpuMemBytes = gpuCapacityBytes(this.config.gpu.memoryGB);
    if (totalMemory > gpuMemBytes) {
      errors.push(
        `Configuration requires ${bytesToGB(totalMemory).toFixed(1)}GB but GPU only has ${this.config.gpu.memoryGB}GB`
      );
    } else if ((totalMemory / gpuMemBytes) * 100 > 90) {
      warnings.push(
        `Memory utilization is ${((totalMemory / gpuMemBytes) * 100).toFixed(0)}%, which may cause OOM`
      );
    }

    // Check tensor parallelism
    const tp = this.config.tensorParallel ?? 1;
    if (tp > 1) {
      if (this.config.numGPUs < tp) {
        errors.push(
          `Tensor parallel degree ${tp} requires at least ${tp} GPUs`
        );
      }
    }

    // Check expert parallelism
    const ep = this.config.expertParallel ?? 1;
    if (ep > 1) {
      if (!this.config.modelSpec.isMoE || !this.config.modelSpec.numExperts) {
        warnings.push('Expert parallelism has no effect on non-MoE models');
      } else if (this.config.modelSpec.numExperts % ep !== 0) {
        errors.push(
          `EP degree ${ep} must evenly divide numExperts (${this.config.modelSpec.numExperts})`
        );
      }
      if (tp * ep > this.config.numGPUs) {
        errors.push(
          `TP×EP (${tp}×${ep}=${tp * ep}) exceeds numGPUs (${this.config.numGPUs})`
        );
      }
    }

    // Check speculative decoding
    if (this.config.speculative.enabled && !this.config.speculative.draftModel) {
      errors.push('Speculative decoding enabled but no draft model specified');
    }

    // Check sequence length
    if (this.config.inputSeqLen + this.config.outputSeqLen > this.config.modelSpec.maxSeqLength) {
      warnings.push(
        `Total sequence length ${this.config.inputSeqLen + this.config.outputSeqLen} exceeds model max ${this.config.modelSpec.maxSeqLength}`
      );
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings,
    };
  }

  /**
   * Run full simulation
   */
  run(skipRecommendations = false): InferenceSimulationResult {
    if (!this.config) {
      throw new Error('Simulation not configured');
    }

    const replicaConfig = { ...this.config, batchSize: this.batchPerReplica };

    const validation = this.validate();
    if (!validation.valid) {
      // Still compute actual memory breakdown for error diagnostics
      let memory = calculateMemoryFromConfig(replicaConfig);
      // Include draft model weights and KV cache in OOM diagnostics
      if (this.config.speculative.enabled && this.config.speculative.draftModel) {
        const draftTP = this.config.tensorParallel ?? 1;
        const draftWeights = modelWeightsMemory(this.config.speculative.draftModel, this.config.weightPrecision) / draftTP;
        const totalSeqLen = this.config.inputSeqLen + this.config.outputSeqLen;
        const draftKVCache = totalKVCacheMemory(this.config.speculative.draftModel, totalSeqLen, this.batchPerReplica, this.config.kvCachePrecision) / draftTP;
        memory = { ...memory, weights: memory.weights + draftWeights, kvCache: memory.kvCache + draftKVCache, total: memory.total + draftWeights + draftKVCache };
      }
      const gpuMemoryBytes = gpuCapacityBytes(this.config.gpu.memoryGB);

      const oomResult: InferenceSimulationResult = {
        success: false,
        memory,
        kvCacheState: {
          currentSeqLen: 0,
          batchSize: this.config.batchSize,
          memoryUsed: 0,
          memoryPerToken: 0,
          utilizationPercent: (memory.total / gpuMemoryBytes) * 100,
        },
        latency: {
          ttft: 0,
          tpot: 0,
          totalLatency: 0,
          prefillTime: 0,
          decodeTime: 0,
        },
        throughput: {
          tokensPerSecond: 0,
          requestsPerSecond: 0,
          prefillTokensPerSecond: 0,
          decodeTokensPerSecond: 0,
        },
        utilization: {
          computeUtilization: 0,
          rooflineAttainment: 0,
          memoryCapacityUtilization: memory.total / gpuMemoryBytes,
          isComputeBound: false,
          isMemoryBound: true,
          bottleneck: 'memory_capacity',
        },
        maxConcurrentRequests: 0,
        errors: validation.errors,
        warnings: validation.warnings,
        recommendations: [],
        events: [],
      };

      if (!skipRecommendations) {
        oomResult.recommendations = generateInferenceRecommendations(this.inputConfig!, oomResult);
      }
      return oomResult;
    }

    // Calculate memory breakdown (per-replica batch)
    let memory = calculateMemoryFromConfig(replicaConfig);

    // Add draft model weights and KV cache when speculative decoding is enabled
    const tp = this.config.tensorParallel ?? 1;
    const ep = this.config.expertParallel ?? 1;
    if (this.config.speculative.enabled && this.config.speculative.draftModel) {
      const draftWeights = modelWeightsMemory(this.config.speculative.draftModel, this.config.weightPrecision) / tp;
      const draftKVCache = totalKVCacheMemory(this.config.speculative.draftModel, this.config.inputSeqLen + this.config.outputSeqLen, this.batchPerReplica, this.config.kvCachePrecision) / tp;
      memory = {
        ...memory,
        weights: memory.weights + draftWeights,
        kvCache: memory.kvCache + draftKVCache,
        total: memory.total + draftWeights + draftKVCache,
      };
    }

    // Calculate KV cache state at end of generation
    const totalSeqLen = this.config.inputSeqLen + this.config.outputSeqLen;
    // Use the actual per-GPU KV cache from the TP-aware memory breakdown
    const kvCachePerGPU = memory.kvCache;
    const gpuMemoryBytes = gpuCapacityBytes(this.config.gpu.memoryGB);
    const availableForKV = gpuMemoryBytes - memory.weights - memory.activations - memory.overhead;
    const kvCacheState: KVCacheState = {
      currentSeqLen: totalSeqLen,
      batchSize: this.batchPerReplica,
      memoryUsed: kvCachePerGPU,
      memoryPerToken: this.batchPerReplica > 0 ? kvCachePerGPU / (totalSeqLen * this.batchPerReplica) : 0,
      utilizationPercent: availableForKV > 0 ? Math.min((kvCachePerGPU / availableForKV) * 100, 100) : 100,
    };

    // Calculate performance metrics (per-replica)
    const metrics = calculateMetricsFromConfig(replicaConfig);

    // Apply continuous batching adjustment
    let effectiveMetrics = metrics;
    if (this.config.continuousBatching) {
      effectiveMetrics = calculateContinuousBatchingMetrics(replicaConfig, metrics);
    }

    // Calculate speculative decoding metrics if enabled.
    // Pass effectiveMetrics.latency.tpot as baseline override so speedup reflects
    // the actual alternative (CB-adjusted when CB is on, static otherwise).
    let speculative: SpeculativeMetrics | undefined;
    if (this.config.speculative.enabled && this.config.speculative.draftModel) {
      speculative = calculateSpeculativeMetrics(
        this.config.speculative,
        this.config.modelSpec,
        this.config.inputSeqLen + this.config.outputSeqLen / 2,
        this.batchPerReplica,
        this.config.gpu,
        this.config.weightPrecision,
        this.config.kvCachePrecision,
        tp,
        ep,
        effectiveMetrics.latency.tpot
      ) ?? undefined;
    }

    // Apply speculative metrics to headline latency/throughput when beneficial.
    // When CB is enabled, compare speculative TPOT against CB TPOT (not static baseline).
    // CB scheduling overhead still applies — the scheduler runs every draft-verify cycle.
    let latency = effectiveMetrics.latency;
    let perReplicaThroughput = effectiveMetrics.throughput;
    if (speculative && speculative.effectiveTpot < effectiveMetrics.latency.tpot) {
      const overhead = this.config.continuousBatching
        ? cbSchedulingOverhead(this.batchPerReplica)
        : 0;
      const specTpot = speculative.effectiveTpot * (1 + overhead);
      const specDecodeTime = specTpot * this.config.outputSeqLen;
      const specTotalLatency = latency.ttft + specDecodeTime;

      latency = {
        ...latency,
        tpot: specTpot,
        decodeTime: specDecodeTime,
        totalLatency: specTotalLatency,
      };

      // Recompute throughput from speculative-adjusted latency
      const totalLatencySec = specTotalLatency / 1000;
      perReplicaThroughput = {
        tokensPerSecond: (this.config.outputSeqLen * this.batchPerReplica) / totalLatencySec,
        requestsPerSecond: this.batchPerReplica / totalLatencySec,
        prefillTokensPerSecond: effectiveMetrics.throughput.prefillTokensPerSecond,
        decodeTokensPerSecond: (this.config.outputSeqLen * this.batchPerReplica) / (specDecodeTime / 1000),
      };
    }

    // Warn when speculative decoding is enabled but provides no benefit
    if (speculative && speculative.speedup <= 1.0) {
      validation.warnings.push(
        'Speculative decoding is enabled but provides no benefit for this configuration — draft model overhead exceeds the verification savings.'
      );
    }

    // Calculate max concurrent requests (per-replica, then scale)
    const maxConcurrentPerReplica = this.calculateMaxConcurrentRequests();
    const maxConcurrentRequests = maxConcurrentPerReplica * this.numReplicas;

    // Generate events for visualization
    const events = this.generateEvents();

    // Scale throughput across all replicas
    const successResult: InferenceSimulationResult = {
      success: true,
      memory,
      kvCacheState,
      latency,
      throughput: {
        tokensPerSecond: perReplicaThroughput.tokensPerSecond * this.numReplicas,
        requestsPerSecond: perReplicaThroughput.requestsPerSecond * this.numReplicas,
        prefillTokensPerSecond: perReplicaThroughput.prefillTokensPerSecond * this.numReplicas,
        decodeTokensPerSecond: perReplicaThroughput.decodeTokensPerSecond * this.numReplicas,
      },
      utilization: effectiveMetrics.utilization,
      speculative,
      continuousBatching: this.config.continuousBatching || undefined,
      maxConcurrentRequests,
      errors: [],
      warnings: validation.warnings,
      recommendations: [],
      events,
    };

    if (!skipRecommendations) {
      successResult.recommendations = generateInferenceRecommendations(this.inputConfig!, successResult);
    }
    return successResult;
  }

  /**
   * Calculate maximum concurrent requests that fit in GPU memory
   */
  private calculateMaxConcurrentRequests(): number {
    if (!this.config) return 0;

    const gpuMemoryBytes = gpuCapacityBytes(this.config.gpu.memoryGB);

    // Compute memory at batch=1 and batch=2 to get marginal cost per request
    // Use replicaConfig (TP-aware) for per-GPU memory
    const replicaBase = { ...this.config, batchSize: this.batchPerReplica };
    const memAt1 = calculateMemoryFromConfig({ ...replicaBase, batchSize: 1 });
    const memAt2 = calculateMemoryFromConfig({ ...replicaBase, batchSize: 2 });
    const perRequestCost = memAt2.total - memAt1.total;

    if (perRequestCost <= 0) return 1;

    // Base memory (weights + overhead + 1 request) minus the per-request cost = fixed cost
    const fixedCost = memAt1.total - perRequestCost;
    const availableForRequests = gpuMemoryBytes - fixedCost;

    if (availableForRequests <= 0) return 0;

    return Math.max(0, Math.floor(availableForRequests / perRequestCost));
  }

  /**
   * Generate events for visualization timeline
   */
  generateEvents(): InferenceEvent[] {
    if (!this.config) return [];

    const events: InferenceEvent[] = [];
    let currentTime = 0;

    const { modelSpec, gpu, inputSeqLen, outputSeqLen, weightPrecision, kvCachePrecision } = this.config;
    const batchSize = this.batchPerReplica;

    // TP/EP-aware memory helper: computes per-GPU memory at a given sequence length
    const getEventMemory = (seqLen: number, bs: number) =>
      calculateMemoryFromConfig({
        ...this.config!,
        batchSize: bs,
        inputSeqLen: seqLen,
        outputSeqLen: 0,
      });

    // Simulation start
    events.push({
      type: 'simulation_start',
      timestamp: currentTime,
    });

    // Prefill phase
    const ttft = calculateLatencyMetrics(
      modelSpec, inputSeqLen, outputSeqLen, batchSize, gpu, weightPrecision, kvCachePrecision
    ).ttft;

    events.push({
      type: 'prefill_start',
      timestamp: currentTime,
      phase: 'prefill',
    } as PhaseEvent);

    // Memory snapshot at prefill start
    const prefillMemory = getEventMemory(inputSeqLen, batchSize);
    events.push({
      type: 'memory_snapshot',
      timestamp: currentTime,
      breakdown: prefillMemory,
    } as InferenceMemoryEvent);

    // KV cache state after prefill
    events.push({
      type: 'kv_cache_update',
      timestamp: currentTime + ttft * 0.9,
      currentSeqLen: inputSeqLen,
      memoryUsedBytes: prefillMemory.kvCache,
      utilizationPercent: (prefillMemory.total / gpuCapacityBytes(gpu.memoryGB)) * 100,
    } as KVCacheUpdateEvent);

    currentTime += ttft;

    events.push({
      type: 'prefill_end',
      timestamp: currentTime,
      phase: 'prefill',
      tokensProcessed: inputSeqLen * batchSize,
      durationMs: ttft,
    } as PhaseEvent);

    // Decode phase
    events.push({
      type: 'decode_start',
      timestamp: currentTime,
      phase: 'decode',
    } as PhaseEvent);

    // Generate token events
    if (this.config.speculative.enabled && this.config.speculative.draftModel) {
      // Speculative decoding events
      const specResult = simulateSpeculativeDecoding(
        this.config.speculative,
        modelSpec,
        inputSeqLen,
        outputSeqLen,
        batchSize,
        gpu,
        weightPrecision,
        kvCachePrecision,
      );

      let tokenCount = 0;
      for (const iteration of specResult.iterations) {
        // Draft event
        events.push({
          type: 'speculative_draft',
          timestamp: currentTime,
          tokens: iteration.draftTokens,
          draftTimeMs: iteration.draftTime,
        } as SpeculativeDraftEvent);

        currentTime += iteration.draftTime;

        // Verify event
        events.push({
          type: 'speculative_verify',
          timestamp: currentTime,
          verificationTimeMs: iteration.verifyTime,
        } as SpeculativeVerifyEvent);

        currentTime += iteration.verifyTime;

        // Accept/reject events for each token
        for (const token of iteration.draftTokens) {
          events.push({
            type: token.status === 'accepted' ? 'speculative_accept' : 'speculative_reject',
            timestamp: currentTime,
            tokenIndex: tokenCount++,
            token: token.token,
          } as SpeculativeResultEvent);
        }

        // KV cache update
        const seqLen = inputSeqLen + tokenCount;
        const kvMemory = getEventMemory(seqLen, batchSize).kvCache;

        events.push({
          type: 'kv_cache_update',
          timestamp: currentTime,
          currentSeqLen: seqLen,
          memoryUsedBytes: kvMemory,
          utilizationPercent: (kvMemory / gpuCapacityBytes(gpu.memoryGB)) * 100,
        } as KVCacheUpdateEvent);
      }
    } else {
      // Standard decode events
      const avgSeqLen = inputSeqLen + outputSeqLen / 2;
      const avgTpot = estimateTPOT(modelSpec, avgSeqLen, batchSize, gpu, weightPrecision, kvCachePrecision);

      // Sample token events (not every token to avoid too many events)
      const sampleRate = Math.max(1, Math.floor(outputSeqLen / 50));

      for (let i = 0; i < outputSeqLen; i++) {
        currentTime += avgTpot;

        if (i % sampleRate === 0 || i === outputSeqLen - 1) {
          events.push({
            type: 'token_generated',
            timestamp: currentTime,
            tokenIndex: i,
            phase: 'decode',
            latencyMs: avgTpot,
          } as TokenGeneratedEvent);

          // KV cache update
          const seqLen = inputSeqLen + i + 1;
          const kvMemory = getEventMemory(seqLen, batchSize).kvCache;

          events.push({
            type: 'kv_cache_update',
            timestamp: currentTime,
            currentSeqLen: seqLen,
            memoryUsedBytes: kvMemory,
            utilizationPercent: (kvMemory / gpuCapacityBytes(gpu.memoryGB)) * 100,
          } as KVCacheUpdateEvent);
        }
      }
    }

    const decodeEndTime = currentTime;

    events.push({
      type: 'decode_end',
      timestamp: decodeEndTime,
      phase: 'decode',
      tokensProcessed: outputSeqLen * batchSize,
      durationMs: decodeEndTime - ttft,
    } as PhaseEvent);

    // Final memory snapshot
    const finalMemory = getEventMemory(inputSeqLen + outputSeqLen, batchSize);
    events.push({
      type: 'memory_snapshot',
      timestamp: decodeEndTime,
      breakdown: finalMemory,
    } as InferenceMemoryEvent);

    // Simulation end
    events.push({
      type: 'simulation_end',
      timestamp: decodeEndTime,
    });

    return events;
  }

  /**
   * Get summary for display
   */
  getSummary(): {
    model: string;
    params: string;
    gpu: string;
    precision: string;
    batchSize: number;
    seqLength: string;
  } | null {
    if (!this.config) return null;

    const formatParams = (n: number) => {
      if (n >= 1e12) return `${(n / 1e12).toFixed(1)}T`;
      if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
      if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
      return `${(n / 1e3).toFixed(1)}K`;
    };

    return {
      model: this.config.modelSpec.name,
      params: formatParams(this.config.modelSpec.totalParams),
      gpu: `${this.config.numGPUs}x ${this.config.gpu.name}`,
      precision: this.config.weightPrecision.toUpperCase(),
      batchSize: this.config.batchSize,
      seqLength: `${this.config.inputSeqLen} + ${this.config.outputSeqLen}`,
    };
  }

  /**
   * Get current config
   */
  getConfig(): InferenceConfig | null {
    return this.config;
  }
}

/**
 * Singleton instance
 */
export const inferenceEngine = new InferenceSimulationEngine();

/**
 * Quick simulation function
 */
export function runInferenceSimulation(config: InferenceSimulationConfig): InferenceSimulationResult {
  const engine = new InferenceSimulationEngine();
  engine.configure(config);
  return engine.run();
}

/**
 * Raw simulation without recommendation generation.
 * Used by the recommendation validator to avoid infinite recursion
 * (run → generateRecommendations → validate → runInferenceSimulation → run → ...).
 */
export function runInferenceSimulationRaw(config: InferenceSimulationConfig): InferenceSimulationResult {
  const engine = new InferenceSimulationEngine();
  engine.configure(config);
  return engine.run(true);
}
