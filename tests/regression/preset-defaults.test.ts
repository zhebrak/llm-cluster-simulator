/**
 * Preset Defaults Regression Tests
 *
 * Pins all training and inference demo preset simulations to their current
 * metric values within ±2.5% relative tolerance. Any change to the simulation
 * engine, strategies, models, or presets that shifts metrics beyond this band
 * will break these tests — forcing explicit acknowledgement.
 */

import { describe, it, expect } from 'vitest';
import { getValidatedSimulationMetrics } from '../helpers/validated-metrics.ts';
import { runInferenceSimulation } from '../../src/core/inference/index.ts';
import { createMultiNodeCluster, createSingleNodeCluster, getPresetCluster, getGPU } from '../../src/core/hardware/index.ts';
import { type SimulationConfig } from '../../src/core/simulation/engine.ts';
import { type ClusterConfig } from '../../src/types/index.ts';

const TOLERANCE = 0.025; // ±2.5%

function makeCluster(opts: {
  clusterId?: string;
  gpuId?: string;
  numGPUs?: number;
  gpusPerNode?: number;
}): ClusterConfig | undefined {
  if (opts.clusterId && opts.clusterId !== 'custom') {
    return getPresetCluster(opts.clusterId) ?? undefined;
  }
  const gpuId = opts.gpuId ?? 'h100-sxm';
  const numGPUs = opts.numGPUs ?? 8;
  const gpusPerNode = opts.gpusPerNode ?? 8;
  const numNodes = Math.ceil(numGPUs / gpusPerNode);
  if (numNodes === 1) {
    return createSingleNodeCluster(gpuId, numGPUs);
  }
  return createMultiNodeCluster(gpuId, gpusPerNode, numNodes);
}

/** Assert value is within ±tolerance of expected (relative). */
function expectClose(actual: number, expected: number, label: string) {
  const rel = Math.abs(actual - expected) / expected;
  expect(
    rel,
    `${label}: expected ~${expected.toPrecision(4)}, got ${actual.toPrecision(4)} (${(rel * 100).toFixed(2)}% off, limit ${TOLERANCE * 100}%)`
  ).toBeLessThanOrEqual(TOLERANCE);
}

// ─── Training Presets ─────────────────────────────────────────────────

interface TrainingPresetCase {
  name: string;
  clusterId?: string;
  gpuId?: string;
  numGPUs?: number;
  gpusPerNode?: number;
  config: SimulationConfig;
  pinned: {
    mfu: number;
    hfu: number;
    memUtil: number;
    tokPerSec: number;
    stepTimeMs: number;
  };
}

const TRAINING_PRESETS: TrainingPresetCase[] = [
  {
    name: 'LLaMA 3 405B (16384× H100)',
    gpuId: 'h100-sxm', numGPUs: 16384, gpusPerNode: 8,
    config: {
      modelId: 'llama3-405b',
      sequenceLength: 8192,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: {
        tp: 8, pp: 16, dp: 128, ep: 1,
        dpType: 'fsdp',
        sequenceParallel: true,
        pipelineSchedule: 'interleaved-1f1b',
        interleavedStages: 4,
      },
      globalBatchSize: 2048,
      microBatchSize: 1,
      activationCheckpointing: false,
      flashAttention: true,
      mixedPrecision: 'bf16',
    },
    pinned: { mfu: 0.4150, hfu: 0.4150, memUtil: 0.9491, tokPerSec: 2761753, stepTimeMs: 6075 },
  },
  {
    name: 'Nemotron-4 340B (6144× H100)',
    gpuId: 'h100-sxm', numGPUs: 6144, gpusPerNode: 8,
    config: {
      modelId: 'nemotron-4-340b',
      sequenceLength: 4096,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: {
        tp: 8, pp: 12, dp: 64, ep: 1,
        dpType: 'fsdp',
        sequenceParallel: true,
        pipelineSchedule: 'interleaved-1f1b',
        interleavedStages: 8,
      },
      globalBatchSize: 768,
      microBatchSize: 1,
      activationCheckpointing: true,
      checkpointingGranularity: 'selective',
      flashAttention: true,
      mixedPrecision: 'bf16',
    },
    pinned: { mfu: 0.4208, hfu: 0.4507, memUtil: 0.3938, tokPerSec: 1249664, stepTimeMs: 2517 },
  },
  {
    name: 'GPT-3 175B (1024× A100)',
    gpuId: 'a100-80gb', numGPUs: 1024, gpusPerNode: 8,
    config: {
      modelId: 'gpt3-175b',
      sequenceLength: 2048,
      strategyType: 'ddp-tp-pp',
      strategyConfig: {
        tp: 8, pp: 8, dp: 16, ep: 1,
        dpType: 'ddp',
        sequenceParallel: false,
        pipelineSchedule: 'interleaved-1f1b',
        interleavedStages: 2,
      },
      globalBatchSize: 1536,
      microBatchSize: 1,
      activationCheckpointing: true,
      flashAttention: false,
      mixedPrecision: 'bf16',
    },
    pinned: { mfu: 0.4199, hfu: 0.5599, memUtil: 0.9128, tokPerSec: 127632, stepTimeMs: 24647 },
  },
  {
    name: 'DeepSeek V3 (2048× H800)',
    gpuId: 'h800-sxm', numGPUs: 2048, gpusPerNode: 8,
    config: {
      modelId: 'deepseek-v3',
      sequenceLength: 4096,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: {
        tp: 4, pp: 8, dp: 64, ep: 32,
        dpType: 'fsdp',
        sequenceParallel: true,
        pipelineSchedule: 'dualpipe-v',
        interleavedStages: 1,
        numMicroBatches: 64,
      },
      globalBatchSize: 8192,
      microBatchSize: 2,
      activationCheckpointing: true,
      flashAttention: true,
      mixedPrecision: 'fp8',
    },
    // H800: identical compute to H100 but NVLink 200 GB/s (vs 450).
    // Published: 43.7% non-causal MFU (ISCA 2025).
    // MoE overhead is modeled per-layer in 3d-parallel.ts with separate MoE residual.
    pinned: { mfu: 0.4460, hfu: 0.5947, memUtil: 0.4959, tokPerSec: 4009248, stepTimeMs: 8369 },
  },
  {
    name: 'LLaMA 4 Maverick (512× H100)',
    clusterId: '512x-h100',
    config: {
      modelId: 'llama4-maverick',
      sequenceLength: 8192,
      strategyType: 'fsdp-tp',
      strategyConfig: {
        tp: 4, pp: 1, dp: 128, ep: 32,
        dpType: 'fsdp',
        sequenceParallel: true,
        pipelineSchedule: '1f1b',
        interleavedStages: 2,
      },
      globalBatchSize: 4096,
      microBatchSize: 2,
      activationCheckpointing: true,
      flashAttention: true,
      mixedPrecision: 'fp8',
    },
    pinned: { mfu: 0.6528, hfu: 0.8704, memUtil: 0.4044, tokPerSec: 3205775, stepTimeMs: 10467 },
  },
  {
    name: 'Grok 2.5 (512× H100)',
    clusterId: '512x-h100',
    config: {
      modelId: 'grok-2.5',
      sequenceLength: 8192,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: {
        tp: 4, pp: 2, dp: 64, ep: 8,
        dpType: 'fsdp',
        sequenceParallel: true,
        pipelineSchedule: 'interleaved-1f1b',
        interleavedStages: 2,
      },
      globalBatchSize: 4096,
      microBatchSize: 2,
      activationCheckpointing: true,
      checkpointingGranularity: 'selective',
      flashAttention: true,
      mixedPrecision: 'fp8',
    },
    pinned: { mfu: 0.8920, hfu: 0.9175, memUtil: 0.6777, tokPerSec: 655202, stepTimeMs: 51212 },
  },
  {
    name: 'Qwen3 32B (64× H100)',
    clusterId: '64x-h100',
    config: {
      modelId: 'qwen3-32b',
      sequenceLength: 4096,
      strategyType: 'fsdp-tp',
      strategyConfig: {
        tp: 4, pp: 1, dp: 16, ep: 1,
        dpType: 'fsdp',
        sequenceParallel: true,
        pipelineSchedule: '1f1b',
        interleavedStages: 2,
      },
      globalBatchSize: 2048,
      microBatchSize: 8,
      activationCheckpointing: true,
      flashAttention: true,
      mixedPrecision: 'bf16',
    },
    pinned: { mfu: 0.4091, hfu: 0.5455, memUtil: 0.3825, tokPerSec: 131738, stepTimeMs: 63676 },
  },
  {
    name: 'OLMo 2 7B (512× H100)',
    gpuId: 'h100-sxm', numGPUs: 512, gpusPerNode: 8,
    config: {
      modelId: 'olmo2-7b',
      sequenceLength: 4096,
      strategyType: 'fsdp',
      strategyConfig: {
        tp: 1, pp: 1, dp: 512, ep: 1,
        dpType: 'fsdp',
        sequenceParallel: false,
        pipelineSchedule: '1f1b',
        interleavedStages: 2,
      },
      globalBatchSize: 1024,
      microBatchSize: 2,
      activationCheckpointing: true,
      flashAttention: true,
      mixedPrecision: 'bf16',
    },
    pinned: { mfu: 0.4055, hfu: 0.5407, memUtil: 0.1872, tokPerSec: 4689608, stepTimeMs: 894 },
  },
];

// ─── Inference Presets ────────────────────────────────────────────────

interface InferencePresetCase {
  name: string;
  clusterId?: string;
  gpuId?: string;
  numGPUs?: number;
  modelId: string;
  batchSize: number;
  inputSeqLen: number;
  outputSeqLen: number;
  weightPrecision: 'fp32' | 'fp16' | 'bf16' | 'fp8' | 'fp4' | 'int8' | 'int4';
  kvCachePrecision: 'fp32' | 'fp16' | 'bf16' | 'fp8' | 'fp4' | 'int8' | 'int4';
  flashAttention: boolean;
  tensorParallel: number;
  pinned: {
    ttft: number;
    tpot: number;
    throughputTps: number;
  };
}

const INFERENCE_PRESETS: InferencePresetCase[] = [
  {
    name: 'DeepSeek V3 (8× H200)',
    gpuId: 'h200-sxm', numGPUs: 8,
    modelId: 'deepseek-v3',
    batchSize: 32, inputSeqLen: 1024, outputSeqLen: 512,
    weightPrecision: 'fp8', kvCachePrecision: 'fp8',
    flashAttention: true, tensorParallel: 8,
    pinned: { ttft: 386.0, tpot: 14.434, throughputTps: 2098.7 },
  },
  {
    name: 'LLaMA 3.3 70B (4× H100)',
    gpuId: 'h100-sxm', numGPUs: 4,
    modelId: 'llama3.3-70b',
    batchSize: 16, inputSeqLen: 1024, outputSeqLen: 512,
    weightPrecision: 'bf16', kvCachePrecision: 'bf16',
    flashAttention: true, tensorParallel: 4,
    pinned: { ttft: 1476.06, tpot: 14.075, throughputTps: 943.5 },
  },
  {
    name: 'Qwen3 235B-A22B (4× H200)',
    gpuId: 'h200-sxm', numGPUs: 4,
    modelId: 'qwen3-235b-a22b',
    batchSize: 32, inputSeqLen: 1024, outputSeqLen: 512,
    weightPrecision: 'fp8', kvCachePrecision: 'fp8',
    flashAttention: true, tensorParallel: 4,
    pinned: { ttft: 455.21, tpot: 14.358, throughputTps: 2098.8 },
  },
  {
    name: 'LLaMA 3.3 70B INT4 (1× H200)',
    gpuId: 'h200-sxm', numGPUs: 1,
    modelId: 'llama3.3-70b',
    batchSize: 8, inputSeqLen: 1024, outputSeqLen: 512,
    weightPrecision: 'int4', kvCachePrecision: 'fp8',
    flashAttention: true, tensorParallel: 1,
    pinned: { ttft: 2789.65, tpot: 11.659, throughputTps: 460.7 },
  },
  {
    name: 'LLaMA 3.1 8B (1× L4)',
    clusterId: 'inf-1x-l4',
    modelId: 'llama3.1-8b',
    batchSize: 8, inputSeqLen: 1024, outputSeqLen: 512,
    weightPrecision: 'fp8', kvCachePrecision: 'fp8',
    flashAttention: true, tensorParallel: 1,
    pinned: { ttft: 166.20, tpot: 4.1439, throughputTps: 1790.3 },
  },
  {
    name: 'LLaMA 3.1 8B INT8 (1× A10G)',
    gpuId: 'a10g', numGPUs: 1,
    modelId: 'llama3.1-8b',
    batchSize: 4, inputSeqLen: 1024, outputSeqLen: 256,
    weightPrecision: 'int8', kvCachePrecision: 'int8',
    flashAttention: true, tensorParallel: 1,
    pinned: { ttft: 2136.26, tpot: 23.211, throughputTps: 125.5 },
  },
];

// ─── Training Tests ───────────────────────────────────────────────────

describe('Training Preset Defaults (pinned ±2.5%)', () => {
  for (const preset of TRAINING_PRESETS) {
    it(preset.name, () => {
      const cluster = makeCluster({
        clusterId: preset.clusterId,
        gpuId: preset.gpuId,
        numGPUs: preset.numGPUs,
        gpusPerNode: preset.gpusPerNode,
      });
      expect(cluster, 'Cluster should be created').toBeDefined();

      const metrics = getValidatedSimulationMetrics({ ...preset.config, clusterConfig: cluster! });

      // Sanity: no OOM, valid MFU range, no NaN
      expect(metrics.memoryUtilization, 'Should not OOM').toBeLessThan(1.0);
      expect(metrics.mfu, 'MFU should not exceed 100%').toBeLessThanOrEqual(1.0);
      expect(metrics.mfu, 'MFU should be positive').toBeGreaterThan(0);

      // Pinned values ±2.5%
      expectClose(metrics.mfu, preset.pinned.mfu, 'MFU');
      expectClose(metrics.hfu, preset.pinned.hfu, 'HFU');
      expectClose(metrics.memoryUtilization, preset.pinned.memUtil, 'Memory utilization');
      expectClose(metrics.tokensPerSecond, preset.pinned.tokPerSec, 'Tokens/sec');
      expectClose(metrics.stepTimeMs, preset.pinned.stepTimeMs, 'Step time');
    });
  }
});

// ─── Inference Tests ──────────────────────────────────────────────────

describe('Inference Preset Defaults (pinned ±2.5%)', () => {
  for (const preset of INFERENCE_PRESETS) {
    it(preset.name, () => {
      const gpu = preset.gpuId ? getGPU(preset.gpuId) : undefined;
      const result = runInferenceSimulation({
        modelId: preset.modelId,
        gpu,
        numGPUs: preset.numGPUs,
        batchSize: preset.batchSize,
        inputSeqLen: preset.inputSeqLen,
        outputSeqLen: preset.outputSeqLen,
        weightPrecision: preset.weightPrecision,
        kvCachePrecision: preset.kvCachePrecision,
        flashAttention: preset.flashAttention,
        tensorParallel: preset.tensorParallel,
      });

      // Sanity: should succeed
      expect(result.success, 'Inference should succeed').toBe(true);
      expect(result.errors.length, 'Should have no errors').toBe(0);

      // Pinned values ±2.5%
      expectClose(result.latency.ttft, preset.pinned.ttft, 'TTFT');
      expectClose(result.latency.tpot, preset.pinned.tpot, 'TPOT');
      expectClose(result.throughput.tokensPerSecond, preset.pinned.throughputTps, 'Throughput');
    });
  }
});
