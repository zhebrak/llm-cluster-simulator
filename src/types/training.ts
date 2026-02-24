/**
 * Training configuration types
 */

import type { DTypeConfig } from './model.ts';
import type { StrategyConfig } from './strategy.ts';
import type { ClusterConfig } from './hardware.ts';

// Optimizer type
export type OptimizerType = 'adamw' | 'adam' | 'sgd' | 'adafactor' | 'lion';

// Learning rate scheduler
export type LRSchedulerType =
  | 'constant'
  | 'linear'
  | 'cosine'
  | 'cosine-warmup'
  | 'warmup-stable-decay';

// Optimizer configuration
export interface OptimizerConfig {
  type: OptimizerType;
  learningRate: number;
  weightDecay: number;
  beta1: number;
  beta2: number;
  eps: number;
  fusedOptimizer: boolean;    // Use fused CUDA kernel
}

// Learning rate schedule configuration
export interface LRScheduleConfig {
  type: LRSchedulerType;
  warmupSteps: number;
  warmupRatio: number;
  minLR: number;
  decaySteps?: number;
}

// Training hyperparameters
export interface TrainingHyperparams {
  globalBatchSize: number;    // Total batch size across all GPUs
  microBatchSize: number;     // Batch size per forward pass per GPU
  sequenceLength: number;
  maxSteps: number;
  maxTokens?: number;         // Alternative to maxSteps

  optimizer: OptimizerConfig;
  lrSchedule: LRScheduleConfig;
  dtypes: DTypeConfig;

  gradientClipping: number | null;
  gradientAccumulationSteps: number;
}

// Complete training configuration
export interface TrainingConfig {
  id: string;
  name: string;

  hyperparams: TrainingHyperparams;
  strategy: StrategyConfig;
  cluster: ClusterConfig;

  // Computed values
  effectiveBatchSize: number;     // globalBatchSize after accumulation
  tokensPerStep: number;          // effectiveBatchSize * seqLength
  stepsPerEpoch?: number;         // If dataset size known
  totalTokens: number;            // maxSteps * tokensPerStep

  // Optimizer state size per parameter
  optimizerStateMultiplier: number;  // AdamW = 12 (2 fp32 states + fp32 master)
}

// Training run statistics
export interface TrainingRunStats {
  currentStep: number;
  totalSteps: number;
  elapsedTimeMs: number;
  remainingTimeMs: number;

  // Throughput
  tokensPerSecond: number;
  samplesPerSecond: number;
  stepsPerSecond: number;

  // Efficiency
  mfu: number;
  hfu: number;
  gpuUtilization: number;

  // Memory
  peakMemoryGB: number;
  currentMemoryGB: number;

  // Loss (simulated/estimated)
  estimatedLoss: number;
}

// Chinchilla optimal training configuration
export interface ChinchillaOptimal {
  activeParams: number;  // For MoE, use activeParams (not totalParams)
  optimalTokens: number;
  optimalSteps: number;
  computeOptimalFlops: number;
  // C = 6 * N * D where N = active params, D = tokens
}

// Training cost projection
export interface TrainingCostProjection {
  totalGPUHours: number;
  estimatedCost: number;
  costPerToken: number;
  costPerMFLOP: number;

  // Breakdown
  computeCost: number;
  storageCost: number;
  networkCost: number;

  // Comparison
  cloudProvider: string;
  instanceType: string;
  pricePerGPUHour: number;
}

// Default optimizer configs
export const DEFAULT_ADAMW_CONFIG: OptimizerConfig = {
  type: 'adamw',
  learningRate: 1e-4,
  weightDecay: 0.1,
  beta1: 0.9,
  beta2: 0.95,
  eps: 1e-8,
  fusedOptimizer: true,
};

export const DEFAULT_LR_SCHEDULE: LRScheduleConfig = {
  type: 'cosine-warmup',
  warmupSteps: 2000,
  warmupRatio: 0.01,
  minLR: 1e-5,
};

// Calculate optimizer state multiplier
export function getOptimizerStateMultiplier(optimizer: OptimizerType): number {
  switch (optimizer) {
    case 'adamw':
    case 'adam':
      // 2 FP32 states (momentum, variance) + FP32 master weights = 12 bytes/param
      return 12;
    case 'sgd':
      // 1 FP32 momentum state + FP32 master = 8 bytes/param
      return 8;
    case 'adafactor':
      // Row/column factored = ~4 bytes/param average
      return 4;
    case 'lion':
      // 1 FP32 momentum + FP32 master = 8 bytes/param
      return 8;
    default:
      return 12;
  }
}

/** Chinchilla optimal tokens. For MoE, pass activeParams (not totalParams). */
export function getChinchillaOptimalTokens(activeParams: number): number {
  // Chinchilla formula: D_opt ≈ 20 * N (active params)
  return Math.round(20 * activeParams);
}

/** Compute-optimal FLOPs. For MoE, pass activeParams (not totalParams). */
export function getComputeOptimalFlops(activeParams: number, tokens: number): number {
  // C ≈ 6 * N * D for transformer training
  return 6 * activeParams * tokens;
}
