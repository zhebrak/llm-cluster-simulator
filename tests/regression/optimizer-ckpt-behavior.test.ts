/**
 * Auto-optimizer checkpointing behavior verification
 *
 * Verifies that the optimizer correctly evaluates activation checkpointing
 * based on model size and configuration.
 */
import { describe, it, expect } from 'vitest';
import { optimizeTraining } from '../../src/core/simulation/optimizer.ts';
import {
  type SimulationConfig,
  getSimulationMetrics,
} from '../../src/core/simulation/engine.ts';
import { createMultiNodeCluster } from '../../src/core/hardware/topology.ts';
import { getModel } from '../../src/core/models/index.ts';

function optimize(config: SimulationConfig) {
  const model = getModel(config.modelId!, config.sequenceLength)!;
  const targetTokens = 20 * (model.activeParams ?? model.totalParams);
  return optimizeTraining(config, targetTokens, config.sequenceLength);
}

describe('Auto-optimizer checkpointing behavior', () => {
  it('405B TP=4 PP=8 MBS=4 ckpt=OFF correctly OOMs', () => {
    // Without activation checkpointing, 405B at MBS=4 exceeds H100 80GB memory.
    // The activation memory model correctly captures this.
    const result = getSimulationMetrics({
      modelId: 'llama3-405b',
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 256)!,
      globalBatchSize: 4096,
      microBatchSize: 4,
      sequenceLength: 8192,
      strategyType: 'fsdp-tp-pp',
      mixedPrecision: 'bf16',
      activationCheckpointing: false,
      flashAttention: true,
      strategyConfig: {
        tp: 4, pp: 8, sequenceParallel: true,
        pipelineSchedule: 'interleaved-1f1b', interleavedStages: 8,
      },
    });
    expect(result.memoryUtilization).toBeGreaterThan(1.0);
  });

  it('405B optimizer produces valid config', () => {
    const result = optimize({
      modelId: 'llama3-405b',
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 256)!,
      globalBatchSize: 4096,
      microBatchSize: 4,
      sequenceLength: 8192,
      strategyType: 'fsdp-tp-pp',
      activationCheckpointing: true,
      flashAttention: true,
      strategyConfig: { tp: 4, pp: 8, sequenceParallel: true },
    });

    expect(result.success).toBe(true);
    // Verify the optimized config actually fits in memory
    const metrics = getSimulationMetrics(result.optimizedConfig);
    expect(metrics.memoryUtilization).toBeLessThan(0.87);
    console.log('405B optimized:', {
      ckpt: result.optimizedConfig.activationCheckpointing,
      mbs: result.optimizedConfig.microBatchSize,
      strategy: result.optimizedConfig.strategyType,
      tp: result.optimizedConfig.strategyConfig?.tp,
      pp: result.optimizedConfig.strategyConfig?.pp,
      memUtil: metrics.memoryUtilization.toFixed(3),
    });
  });

  it('LLaMA 2 7B: optimizer fits within memory (ckpt=OFF eligible)', () => {
    const result = optimize({
      modelId: 'llama2-7b',
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 1)!,
      globalBatchSize: 32,
      microBatchSize: 2,
      sequenceLength: 4096,
      strategyType: 'fsdp',
      activationCheckpointing: true,
      flashAttention: true,
    });

    expect(result.success).toBe(true);
    // 7B is small enough that ckpt=OFF should be a valid option
    // (optimizer may or may not pick it depending on throughput tradeoff)
    const metrics = getSimulationMetrics(result.optimizedConfig);
    expect(metrics.memoryUtilization).toBeLessThan(1.0);
    console.log('7B optimized:', {
      ckpt: result.optimizedConfig.activationCheckpointing,
      mbs: result.optimizedConfig.microBatchSize,
      strategy: result.optimizedConfig.strategyType,
      memUtil: metrics.memoryUtilization.toFixed(3),
    });
  });

  it('LLaMA 3 8B: optimizer produces reasonable config', () => {
    const result = optimize({
      modelId: 'llama3-8b',
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 1)!,
      globalBatchSize: 32,
      microBatchSize: 2,
      sequenceLength: 8192,
      strategyType: 'fsdp',
      activationCheckpointing: true,
      flashAttention: true,
    });

    expect(result.success).toBe(true);
    const metrics = getSimulationMetrics(result.optimizedConfig);
    expect(metrics.memoryUtilization).toBeLessThan(1.0);
    console.log('8B optimized:', {
      ckpt: result.optimizedConfig.activationCheckpointing,
      mbs: result.optimizedConfig.microBatchSize,
      strategy: result.optimizedConfig.strategyType,
      memUtil: metrics.memoryUtilization.toFixed(3),
    });
  });

  it('LLaMA 2 70B: optimizer produces valid config', () => {
    const result = optimize({
      modelId: 'llama2-70b',
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 8)!,
      globalBatchSize: 256,
      microBatchSize: 2,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp',
      activationCheckpointing: true,
      flashAttention: true,
      strategyConfig: { tp: 8, sequenceParallel: true },
    });

    expect(result.success).toBe(true);
    const metrics = getSimulationMetrics(result.optimizedConfig);
    expect(metrics.memoryUtilization).toBeLessThan(1.0);
    console.log('70B optimized:', {
      ckpt: result.optimizedConfig.activationCheckpointing,
      mbs: result.optimizedConfig.microBatchSize,
      strategy: result.optimizedConfig.strategyType,
      memUtil: metrics.memoryUtilization.toFixed(3),
    });
  });

  it('GPT-3 175B: optimizer produces valid config', () => {
    const result = optimize({
      modelId: 'gpt3-175b',
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 128)!,
      globalBatchSize: 1024,
      microBatchSize: 2,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp-pp',
      activationCheckpointing: true,
      flashAttention: true,
      strategyConfig: { tp: 8, pp: 8, sequenceParallel: true },
    });

    expect(result.success).toBe(true);
    const metrics = getSimulationMetrics(result.optimizedConfig);
    expect(metrics.memoryUtilization).toBeLessThan(1.0);
    console.log('175B optimized:', {
      ckpt: result.optimizedConfig.activationCheckpointing,
      mbs: result.optimizedConfig.microBatchSize,
      strategy: result.optimizedConfig.strategyType,
      memUtil: metrics.memoryUtilization.toFixed(3),
    });
  });
});
