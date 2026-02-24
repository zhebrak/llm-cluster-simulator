/**
 * Inference Module - LLM Inference Simulation
 *
 * This module provides comprehensive inference simulation including:
 * - KV cache calculations and management
 * - Memory breakdown for inference workloads
 * - Latency modeling (TTFT, TPOT)
 * - Speculative decoding support
 * - Event generation for visualization
 */

// KV Cache
export {
  getPrecisionBytes,
  kvCachePerToken,
  totalKVCacheMemory,
  kvCacheReductionFactor,
  getAttentionTypeString,
  maxSequenceLengthForMemory,
  maxBatchSizeForSequence,
  createKVCacheConfig,
  calculateKVCacheState,
  validateKVCacheFits,
} from './kv-cache.ts';

// Memory
export {
  KV_CACHE_FRAGMENTATION_CAP,
  kvCacheFragmentationFactor,
  modelWeightsMemory,
  activationMemory,
  activationMemoryWithFlashAttention,
  calculateOverhead,
  calculateInferenceMemory,
  memoryTimeline,
  calculateMemoryFromConfig,
  validateMemoryFits,
  maxBatchSizeForGPU,
  calculateMemoryWithTP,
} from './memory.ts';

// Latency
export {
  getGPUTFLOPS,
  prefillFLOPs,
  decodeFLOPs,
  estimateTTFT,
  estimateTPOT,
  estimateAverageTPOT,
  calculateLatencyMetrics,
  calculateThroughputMetrics,
  calculateUtilizationMetrics,
  latencyBreakdown,
  calculateLatencyWithTP,
  calculateMetricsFromConfig,
  calculateContinuousBatchingMetrics,
  cbSchedulingOverhead,
} from './latency.ts';

// Speculative Decoding
export {
  expectedAcceptedTokens,
  theoreticalSpeedup,
  draftModelOverhead,
  estimateDraftTime,
  estimateVerificationTime,
  effectiveTPOTWithSpeculation,
  calculateSpeculativeMetrics,
  simulateSpeculativeDecoding,
  findOptimalSpecConfig,
  estimateAcceptanceRate,
  isSpeculativeDecodingBeneficial,
} from './speculative.ts';
export type { SpeculativeIteration } from './speculative.ts';

// Simulation Engine
export {
  InferenceSimulationEngine,
  inferenceEngine,
  runInferenceSimulation,
  runInferenceSimulationRaw,
} from './simulation.ts';
export type { InferenceSimulationConfig } from './simulation.ts';

// Recommendations
export { generateInferenceRecommendations } from './recommendations.ts';

// Optimizer
export type { InferenceOptimizationResult, InferenceChangelogEntry } from './optimizer.ts';
export { optimizeInference } from './optimizer.ts';

// Pareto Frontier
export type { ParetoPoint, ParetoSweepResult } from './pareto.ts';
export {
  computeCostPerMToken,
  buildSweepKey,
  extractParetoFrontier,
  runParetoSweep,
} from './pareto.ts';

// Sequence Length Sweep
export type { SeqLenSweepPoint, SeqLenSweepResult } from './seq-len-sweep.ts';
export { runSeqLenSweep } from './seq-len-sweep.ts';

