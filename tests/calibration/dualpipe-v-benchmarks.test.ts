/**
 * DualPipeV Benchmark Validation Tests
 *
 * Validates DualPipeV scheduling results against published data and
 * compares with other pipeline schedules.
 *
 * IMPORTANT: The existing DeepSeek V3 test in published-benchmarks.test.ts
 * stays UNCHANGED. It validates "given their published parallelism
 * (FSDP-TP PP=1 EP=64), what do we predict?" This file validates
 * "with DualPipeV scheduling, how close do we get?"
 */

import { describe, it, expect } from 'vitest';
import {
  type SimulationConfig,
  type SimulationMetrics,
  getSimulationMetrics,
} from '../../src/core/simulation/engine.ts';
import { createMultiNodeCluster } from '../../src/core/hardware/topology.ts';

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

function benchmarkConfig(
  gpuId: string,
  gpusPerNode: number,
  numNodes: number,
  modelId: string,
  strategy: SimulationConfig['strategyType'],
  globalBatchSize: number,
  microBatchSize: number,
  sequenceLength: number,
  strategyConfig?: SimulationConfig['strategyConfig'],
): SimulationConfig {
  return {
    clusterConfig: createMultiNodeCluster(gpuId, gpusPerNode, numNodes)!,
    modelId,
    globalBatchSize,
    microBatchSize,
    sequenceLength,
    strategyType: strategy,
    strategyConfig,
  };
}

function sim(config: SimulationConfig): SimulationMetrics {
  return getSimulationMetrics(config);
}

// ===========================================================================
// Section 1: DeepSeek V3 with DualPipeV PP=16
// ===========================================================================

describe('DeepSeek V3 with DualPipeV PP=16', () => {
  it('MFU ≈ 32% with BF16 compute (published: 42.9% with FP8+DualPipe)', () => {
    // DeepSeek V3: 2048 H100, TP=4, PP=16, DP=32, EP=64
    // Published 42.9% is BF16-equivalent MFU using FP8 compute (~2× peak).
    // Our simulator uses BF16 peak only, so ~37% is expected.
    const metrics = sim(benchmarkConfig(
      'h100-sxm', 8, 256, 'deepseek-v3',
      'fsdp-tp-pp', 4608, 1, 4096,
      { tp: 4, pp: 16, ep: 32, sequenceParallel: true, pipelineSchedule: 'dualpipe-v' },
    ));

    // Device-limited routing (M=4): routingLocality = min(densityLocality, 4/64) = 0.0625.
    // Load imbalance scales with expert count: log2(256/8)=5 → factor 1.25.
    // Actual simulator value: ~0.305 (device-limited routing effectiveEp=4).
    // Bounds: ±15% of 0.305 → [0.259, 0.351]
    expect(metrics.mfu).toBeGreaterThan(0.259);
    expect(metrics.mfu).toBeLessThan(0.351);

    // Bubble should be low (~6.3% including layer imbalance penalty for 61 layers / PP=16)
    expect(metrics.pipelineBubble).toBeLessThan(0.08);
    expect(metrics.pipelineBubble).toBeGreaterThan(0.03);
  });

  it('DualPipeV PP=16 has lower bubble than 1F1B PP=16', () => {
    const dualPipe = sim(benchmarkConfig(
      'h100-sxm', 8, 256, 'deepseek-v3',
      'fsdp-tp-pp', 4608, 1, 4096,
      { tp: 4, pp: 16, ep: 32, sequenceParallel: true, pipelineSchedule: 'dualpipe-v' },
    ));

    const onef1b = sim(benchmarkConfig(
      'h100-sxm', 8, 256, 'deepseek-v3',
      'fsdp-tp-pp', 4608, 1, 4096,
      { tp: 4, pp: 16, ep: 32, sequenceParallel: true, pipelineSchedule: '1f1b' },
    ));

    expect(dualPipe.pipelineBubble).toBeLessThan(onef1b.pipelineBubble);
    // DualPipeV bubble should be at least 2× lower than 1F1B
    // (layer imbalance penalty shifts both schedules by a constant, compressing the ratio)
    expect(dualPipe.pipelineBubble).toBeLessThan(onef1b.pipelineBubble / 2);
  });

  it('DualPipeV PP=16 has lower bubble than interleaved PP=16', () => {
    const dualPipe = sim(benchmarkConfig(
      'h100-sxm', 8, 256, 'deepseek-v3',
      'fsdp-tp-pp', 4608, 1, 4096,
      { tp: 4, pp: 16, ep: 32, sequenceParallel: true, pipelineSchedule: 'dualpipe-v' },
    ));

    const interleaved = sim(benchmarkConfig(
      'h100-sxm', 8, 256, 'deepseek-v3',
      'fsdp-tp-pp', 4608, 1, 4096,
      { tp: 4, pp: 16, ep: 32, sequenceParallel: true, pipelineSchedule: 'interleaved-1f1b', interleavedStages: 2 },
    ));

    expect(dualPipe.pipelineBubble).toBeLessThan(interleaved.pipelineBubble);
  });
});

// ===========================================================================
// Section 2: Degraded mode (m < 2*PP)
// ===========================================================================

describe('DualPipeV Degraded Mode', () => {
  it('m < 2*PP degrades to standard bubble formula', () => {
    // Use small GBS to force low GA (m < 2*PP=32)
    const degraded = sim(benchmarkConfig(
      'h100-sxm', 8, 256, 'deepseek-v3',
      'fsdp-tp-pp', 128, 4, 4096,
      { tp: 4, pp: 16, ep: 32, sequenceParallel: true, pipelineSchedule: 'dualpipe-v' },
    ));

    const standard = sim(benchmarkConfig(
      'h100-sxm', 8, 256, 'deepseek-v3',
      'fsdp-tp-pp', 128, 4, 4096,
      { tp: 4, pp: 16, ep: 32, sequenceParallel: true, pipelineSchedule: '1f1b' },
    ));

    // In degraded mode, bubble should be the same as 1F1B
    expect(Math.abs(degraded.pipelineBubble - standard.pipelineBubble)).toBeLessThan(0.001);
  });
});

// ===========================================================================
// Section 3: Activation memory (PP+1 in-flight)
// ===========================================================================

describe('DualPipeV Activation Memory', () => {
  it('DualPipeV uses more activation memory than 1F1B (2*pp+1 vs pp in-flight)', () => {
    // Nemotron-4 340B on 6144 H100s — dense model, easier to compare memory
    const dualPipe = sim(benchmarkConfig(
      'h100-sxm', 8, 768, 'nemotron-4-340b',
      'fsdp-tp-pp', 8192, 4, 4096,
      { tp: 4, pp: 8, sequenceParallel: true, pipelineSchedule: 'dualpipe-v' },
    ));

    const onef1b = sim(benchmarkConfig(
      'h100-sxm', 8, 768, 'nemotron-4-340b',
      'fsdp-tp-pp', 8192, 4, 4096,
      { tp: 4, pp: 8, sequenceParallel: true, pipelineSchedule: '1f1b' },
    ));

    // DualPipeV: 2*pp+1 = 17 in-flight vs 1F1B: pp = 8
    // But capped by min(inFlightBase, numMicroBatches), so actual ratio depends on GA.
    // For this config: GBS=8192, MBS=4, DP=768*8/(4*8)=192, GA=ceil(8192/(4*192))=11
    // DualPipeV: min(17, 11) = 11, 1F1B: min(8, 11) = 8 → ratio = 11/8 = 1.375
    const ratio = dualPipe.memoryPerGPU.peakActivations / onef1b.memoryPerGPU.peakActivations;
    expect(dualPipe.memoryPerGPU.peakActivations).toBeGreaterThan(onef1b.memoryPerGPU.peakActivations);
    expect(ratio).toBeGreaterThan(1.2);
    expect(ratio).toBeLessThan(1.6); // ~1.375 expected (11/8)
  });
});
