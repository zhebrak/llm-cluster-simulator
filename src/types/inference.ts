/**
 * Inference-specific types for LLM inference simulation
 */

import type { ModelSpec } from './model.ts';
import type { GPUSpec, ClusterConfig } from './hardware.ts';

/**
 * Inference precision options
 */
export type InferencePrecision =
  | 'fp32' | 'fp16' | 'bf16' | 'fp8' | 'fp4' | 'int8' | 'int4'
  | 'q2_k' | 'q3_k_m' | 'q4_k_m' | 'q5_k_m' | 'q6_k' | 'q8_0';

/**
 * Precision specification with relative quality
 */
export interface PrecisionSpec {
  bytes: number;
  relativeQuality: number;
  name: string;
  supported: boolean;
}

export const PRECISION_SPECS: Record<InferencePrecision, PrecisionSpec> = {
  fp32: { bytes: 4, relativeQuality: 1.0, name: 'FP32', supported: true },
  fp16: { bytes: 2, relativeQuality: 0.999, name: 'FP16', supported: true },
  bf16: { bytes: 2, relativeQuality: 0.999, name: 'BF16', supported: true },
  fp8: { bytes: 1, relativeQuality: 0.995, name: 'FP8', supported: true },
  fp4: { bytes: 0.5, relativeQuality: 0.92, name: 'FP4', supported: true },
  int8: { bytes: 1, relativeQuality: 0.99, name: 'INT8', supported: true },
  int4: { bytes: 0.5, relativeQuality: 0.95, name: 'INT4', supported: true },
  // GGUF quantization (llama.cpp / Ollama) — bpw from Artefact2's measurements
  q2_k:   { bytes: 3.00 / 8, relativeQuality: 0.70, name: 'Q2_K',   supported: true },
  q3_k_m: { bytes: 3.89 / 8, relativeQuality: 0.82, name: 'Q3_K_M', supported: true },
  q4_k_m: { bytes: 4.83 / 8, relativeQuality: 0.92, name: 'Q4_K_M', supported: true },
  q5_k_m: { bytes: 5.67 / 8, relativeQuality: 0.96, name: 'Q5_K_M', supported: true },
  q6_k:   { bytes: 6.57 / 8, relativeQuality: 0.98, name: 'Q6_K',   supported: true },
  q8_0:   { bytes: 8.50 / 8, relativeQuality: 0.995, name: 'Q8_0',  supported: true },
};

/**
 * Inference phase
 */
export type InferencePhase = 'prefill' | 'decode';

/**
 * Inference memory breakdown
 */
export interface InferenceMemoryBreakdown {
  weights: number;           // Model weights in bytes
  kvCache: number;           // KV cache memory in bytes
  activations: number;       // Activation memory (peak)
  overhead: number;          // CUDA/framework overhead
  total: number;             // Total memory requirement
}

/**
 * KV Cache configuration
 */
export interface KVCacheConfig {
  numLayers: number;
  numKvHeads: number;
  headDim: number;
  maxSeqLen: number;
  precision: InferencePrecision;
  pagedAttention: boolean;
  blockSize?: number;        // For paged attention (default: 16)
  slidingWindow?: number;    // Sliding window attention size
}

/**
 * KV Cache state during inference
 */
export interface KVCacheState {
  currentSeqLen: number;
  batchSize: number;
  memoryUsed: number;        // Current KV cache memory
  memoryPerToken: number;    // Memory per token
  utilizationPercent: number;
}

/**
 * Latency metrics
 */
export interface LatencyMetrics {
  ttft: number;              // Time to First Token (ms)
  tpot: number;              // Time Per Output Token (ms)
  totalLatency: number;      // Total generation latency (ms)
  prefillTime: number;       // Prefill phase time (ms)
  decodeTime: number;        // Decode phase time (ms)
}

/**
 * Throughput metrics
 */
export interface ThroughputMetrics {
  tokensPerSecond: number;   // Generation throughput
  requestsPerSecond: number; // Request throughput (for batch)
  prefillTokensPerSecond: number;  // Prefill throughput
  decodeTokensPerSecond: number;   // Decode throughput
}

/**
 * Utilization metrics
 */
export interface UtilizationMetrics {
  computeUtilization: number;      // 0-1, roofline: prefill arithmetic intensity / ridge point (1.0 = fully compute-bound)
  rooflineAttainment: number; // 0-1, roofline attainment: min(intensity/ridge, 1.0). Low = memory-bound, 1.0 = compute-bound
  memoryCapacityUtilization: number;  // 0-1, fraction of GPU memory used
  isComputeBound: boolean;
  isMemoryBound: boolean;
  bottleneck: 'compute' | 'memory_bandwidth' | 'memory_capacity';
}

/**
 * Speculative decoding configuration
 */
export interface SpeculativeDecodingConfig {
  enabled: boolean;
  draftModel: ModelSpec | null;    // Smaller, faster draft model
  numSpeculativeTokens: number;    // K tokens to speculate (typically 4-8)
  acceptanceRate: number;          // Expected acceptance rate (0-1)
}

/**
 * Speculative decoding metrics
 */
export interface SpeculativeMetrics {
  expectedAcceptedTokens: number;
  speedup: number;                 // Compared to standard decoding
  draftModelOverhead: number;      // Time for draft model (ms)
  verificationTime: number;        // Time for target verification (ms)
  effectiveTpot: number;           // Effective TPOT with speculation (ms)
}

/**
 * Speculative token state for visualization
 */
export interface SpeculativeTokenState {
  tokenId: number;
  token: string;
  position: number;
  status: 'pending' | 'accepted' | 'rejected';
  draftProbability: number;
  targetProbability: number;
}

/**
 * Inference configuration
 */
export interface InferenceConfig {
  // Model
  modelSpec: ModelSpec;

  // Hardware
  gpu: GPUSpec;
  numGPUs: number;
  clusterConfig?: ClusterConfig;

  // Inference parameters
  batchSize: number;
  inputSeqLen: number;       // Prompt length
  outputSeqLen: number;      // Max tokens to generate

  // Precision
  weightPrecision: InferencePrecision;
  kvCachePrecision: InferencePrecision;

  // Optimizations
  flashAttention: boolean;
  pagedAttention: boolean;
  continuousBatching: boolean;

  // Speculative decoding
  speculative: SpeculativeDecodingConfig;

  // Advanced
  tensorParallel?: number;   // TP degree for multi-GPU
  expertParallel?: number;   // EP degree for MoE models
  kvCacheQuantization?: boolean;
  slidingWindow?: number;
}

/**
 * Full inference simulation result
 */
export interface InferenceSimulationResult {
  success: boolean;

  // Memory analysis
  memory: InferenceMemoryBreakdown;
  kvCacheState: KVCacheState;

  // Performance metrics
  latency: LatencyMetrics;
  throughput: ThroughputMetrics;
  utilization: UtilizationMetrics;

  // Speculative decoding (if enabled)
  speculative?: SpeculativeMetrics;

  // Continuous batching (if enabled)
  continuousBatching?: boolean;

  // Derived metrics
  costPerMillionTokens?: number;
  maxConcurrentRequests: number;
  numReplicas: number;

  // Validation
  errors: string[];
  warnings: string[];
  recommendations: string[];

  // Events for visualization
  events: InferenceEvent[];
}

/**
 * Inference event types for visualization
 */
export type InferenceEventType =
  | 'simulation_start'
  | 'simulation_end'
  | 'prefill_start'
  | 'prefill_end'
  | 'decode_start'
  | 'decode_end'
  | 'token_generated'
  | 'kv_cache_update'
  | 'memory_snapshot'
  | 'speculative_draft'
  | 'speculative_verify'
  | 'speculative_accept'
  | 'speculative_reject';

/**
 * Base inference event
 */
export interface InferenceEventBase {
  type: InferenceEventType;
  timestamp: number;         // ms from start
  gpuId?: number;
}

/**
 * Token generated event
 */
export interface TokenGeneratedEvent extends InferenceEventBase {
  type: 'token_generated';
  tokenIndex: number;
  phase: InferencePhase;
  latencyMs: number;
}

/**
 * KV cache update event
 */
export interface KVCacheUpdateEvent extends InferenceEventBase {
  type: 'kv_cache_update';
  currentSeqLen: number;
  memoryUsedBytes: number;
  utilizationPercent: number;
}

/**
 * Memory snapshot event
 */
export interface InferenceMemoryEvent extends InferenceEventBase {
  type: 'memory_snapshot';
  breakdown: InferenceMemoryBreakdown;
}

/**
 * Speculative draft event
 */
export interface SpeculativeDraftEvent extends InferenceEventBase {
  type: 'speculative_draft';
  tokens: SpeculativeTokenState[];
  draftTimeMs: number;
}

/**
 * Speculative verification event
 */
export interface SpeculativeVerifyEvent extends InferenceEventBase {
  type: 'speculative_verify';
  verificationTimeMs: number;
}

/**
 * Speculative accept/reject event
 */
export interface SpeculativeResultEvent extends InferenceEventBase {
  type: 'speculative_accept' | 'speculative_reject';
  tokenIndex: number;
  token: string;
}

/**
 * Phase start/end events
 */
export interface PhaseEvent extends InferenceEventBase {
  type: 'prefill_start' | 'prefill_end' | 'decode_start' | 'decode_end';
  phase: InferencePhase;
  tokensProcessed?: number;
  durationMs?: number;
}

/**
 * Union of all inference events
 */
export type InferenceEvent =
  | InferenceEventBase
  | TokenGeneratedEvent
  | KVCacheUpdateEvent
  | InferenceMemoryEvent
  | SpeculativeDraftEvent
  | SpeculativeVerifyEvent
  | SpeculativeResultEvent
  | PhaseEvent;

/**
 * Default inference configuration
 */
export const DEFAULT_INFERENCE_CONFIG: Partial<InferenceConfig> = {
  batchSize: 1,
  inputSeqLen: 1024,
  outputSeqLen: 512,
  weightPrecision: 'bf16',
  kvCachePrecision: 'bf16',
  flashAttention: true,
  pagedAttention: true,
  continuousBatching: false,
  speculative: {
    enabled: false,
    draftModel: null,
    numSpeculativeTokens: 4,
    acceptanceRate: 0.7,
  },
};
