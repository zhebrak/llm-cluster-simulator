/**
 * Parallelism strategy types
 */

import type { MemoryBreakdown, CommunicationBreakdown, TimingBreakdown } from './base.ts';

// Basic parallelism dimensions
export type ParallelismDimension = 'data' | 'tensor' | 'pipeline' | 'sequence' | 'expert' | 'context';

// Strategy types
export type StrategyType =
  | 'ddp' | 'fsdp' | 'zero-1' | 'zero-3'
  | 'fsdp-tp' | 'zero1-tp'
  | 'ddp-tp-pp' | 'zero1-tp-pp' | 'fsdp-tp-pp'
  | 'auto';

// Pipeline schedule types
export type PipelineSchedule =
  | 'gpipe'         // GPipe: all forward then all backward
  | '1f1b'          // One Forward One Backward
  | 'interleaved-1f1b'  // Virtual stages interleaved
  | 'dualpipe-v'    // DualPipeV: bidirectional V-shape with F/B overlap
  | 'zero-bubble';  // Zero bubble scheduling

// Data parallelism configuration
export interface DataParallelConfig {
  type: 'ddp' | 'fsdp' | 'zero-1' | 'zero-2' | 'zero-3';
  worldSize: number;           // Number of DP ranks
  gradientAccumulationSteps: number;
  gradientClipping: number | null;
  overlapCommCompute: boolean;
  bucketSizeMB: number;        // For gradient bucketing

}

// Tensor parallelism configuration
export interface TensorParallelConfig {
  enabled: boolean;
  degree: number;              // TP world size
  sequenceParallel: boolean;   // Enable sequence parallelism with TP
  communicationBackend: 'nccl' | 'gloo';
}

// Pipeline parallelism configuration
export interface PipelineParallelConfig {
  enabled: boolean;
  degree: number;              // PP world size (number of stages)
  numMicroBatches: number;
  schedule: PipelineSchedule;
  interleavedStages?: number;  // For interleaved 1F1B
  activationCheckpointing: boolean;
}

// Context parallelism (Ring Attention)
export interface ContextParallelConfig {
  enabled: boolean;
  degree: number;
  chunkSize: number;           // Sequence chunk size
}

// Expert parallelism configuration
export interface ExpertParallelConfig {
  enabled: boolean;
  degree: number;              // EP world size
  expertSlicing: boolean;      // Whether to slice experts across ranks
  capacityFactor: number;
  dropTokens: boolean;
  loadBalancingLoss: boolean;
}

// Activation checkpointing configuration
export interface ActivationCheckpointConfig {
  enabled: boolean;
  granularity: 'full' | 'selective' | 'offload';
  checkpointEveryNLayers: number;
  selectiveLayers: string[];   // Which layer types to checkpoint
}

// Complete strategy configuration
export interface StrategyConfig {
  name: string;
  type: StrategyType;

  // Parallelism dimensions
  dataParallel: DataParallelConfig;
  tensorParallel: TensorParallelConfig;
  pipelineParallel: PipelineParallelConfig;
  contextParallel: ContextParallelConfig;
  expertParallel: ExpertParallelConfig;

  // Memory optimization
  activationCheckpointing: ActivationCheckpointConfig;
  mixedPrecision: boolean;

  // Computed
  totalGPUs: number;
  dpRanks: number;
  tpRanks: number;
  ppRanks: number;
  epRanks: number;
  cpRanks: number;
}

// Strategy validation result
export interface StrategyValidation {
  valid: boolean;
  errors: string[];
  warnings: string[];
  suggestions: string[];
}

// Strategy analysis result
export interface StrategyAnalysis {
  memory: MemoryBreakdown;
  communication: CommunicationBreakdown;
  timing: TimingBreakdown;
  validation: StrategyValidation;

  // Efficiency metrics
  mfu: number;                 // Model FLOPS Utilization (0-1)
  hfu: number;                 // Hardware FLOPS Utilization (0-1)
  communicationOverhead: number; // Fraction of step time in exposed (non-overlapped) communication
  pipelineBubble: number;      // Pipeline bubble fraction (0-1)
  memoryEfficiency: number;    // Actually used / allocated memory

  // Communication reporting
  communicationGrossMs: number;     // timing.communication + epCommunication
  communicationExposedMs: number;   // gross - timing.overlap
  overlapHiddenFraction: number;    // timing.overlap / gross (0 to 1)

  // Throughput
  tokensPerSecond: number;
  samplesPerSecond: number;
  tflopsPerGPU: number;

  // Selective AC: resolved stored-layers count (undefined when not selective)
  resolvedStoredLayers?: number;
}

// Comparison between strategies
export interface StrategyComparison {
  strategies: StrategyConfig[];
  analyses: StrategyAnalysis[];
  winner: {
    memory: number;            // Index of strategy with lowest memory
    throughput: number;        // Index with highest throughput
    efficiency: number;        // Index with highest MFU
    cost: number;              // Index with lowest cost
  };
  recommendations: string[];
}

// Default configurations
export const DEFAULT_DP_CONFIG: DataParallelConfig = {
  type: 'ddp',
  worldSize: 1,
  gradientAccumulationSteps: 1,
  gradientClipping: 1.0,
  overlapCommCompute: true,
  bucketSizeMB: 25,
};

export const DEFAULT_TP_CONFIG: TensorParallelConfig = {
  enabled: false,
  degree: 1,
  sequenceParallel: false,
  communicationBackend: 'nccl',
};

export const DEFAULT_PP_CONFIG: PipelineParallelConfig = {
  enabled: false,
  degree: 1,
  numMicroBatches: 1,
  schedule: '1f1b',
  activationCheckpointing: true,
};

export const DEFAULT_CP_CONFIG: ContextParallelConfig = {
  enabled: false,
  degree: 1,
  chunkSize: 8192,
};

export const DEFAULT_EP_CONFIG: ExpertParallelConfig = {
  enabled: false,
  degree: 1,
  expertSlicing: false,
  capacityFactor: 1.25,
  dropTokens: false,
  loadBalancingLoss: true,
};

export const DEFAULT_ACTIVATION_CHECKPOINT_CONFIG: ActivationCheckpointConfig = {
  enabled: true,
  granularity: 'selective',
  checkpointEveryNLayers: 1,
  selectiveLayers: ['attention', 'mlp'],
};
