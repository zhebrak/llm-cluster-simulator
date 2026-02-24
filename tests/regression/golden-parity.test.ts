/**
 * Golden parity gate for overlap model.
 *
 * Loads the baseline snapshot (tests/golden/overlap-baseline.json) and verifies
 * that current simulator output matches exactly. Tolerance: 1e-12 relative error
 * to accommodate float reordering from function extraction.
 *
 * Regenerate baseline: npx tsx scripts/dump-golden.ts
 */
import { describe, it, expect } from 'vitest';
import {
  type SimulationConfig,
  getSimulationMetrics,
} from '../../src/core/simulation/engine.ts';
import { createMultiNodeCluster, createSingleNodeCluster } from '../../src/core/hardware/topology.ts';
import goldenData from '../golden/overlap-baseline.json';

interface GoldenEntry {
  name: string;
  mfu: number;
  totalTimeMs: number;
  communication: number;
  overlap: number;
}

const golden: GoldenEntry[] = goldenData;

// Tolerance: 1e-12 relative error (accommodates float reordering, not formula changes)
const REL_TOL = 1e-12;

function assertClose(actual: number, expected: number, label: string) {
  if (expected === 0) {
    expect(Math.abs(actual)).toBeLessThan(REL_TOL);
    return;
  }
  const relErr = Math.abs(actual - expected) / Math.abs(expected);
  expect(relErr, `${label}: actual=${actual}, expected=${expected}, relErr=${relErr}`).toBeLessThan(REL_TOL);
}

// Build configs matching the dump script — must stay in sync.
function getConfig(name: string): SimulationConfig {
  const configs: Record<string, SimulationConfig> = {
    'LLaMA 3.1 405B × 16384 H100': {
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 2048)!,
      modelId: 'llama3-405b',
      globalBatchSize: 2048, microBatchSize: 1, sequenceLength: 8192,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: {
        tp: 8, pp: 16, dp: 16, dpType: 'fsdp', sequenceParallel: true,
        pipelineSchedule: 'interleaved-1f1b', interleavedStages: 4,
      },
      activationCheckpointing: false, flashAttention: true, mixedPrecision: 'bf16',
    },
    'GPT-3 175B × 1024 A100': {
      clusterConfig: createMultiNodeCluster('a100-80gb', 8, 128)!,
      modelId: 'gpt3-175b',
      globalBatchSize: 1536, microBatchSize: 1, sequenceLength: 2048,
      strategyType: 'ddp-tp-pp',
      strategyConfig: {
        tp: 8, pp: 8, dp: 16, dpType: 'ddp', sequenceParallel: false,
        pipelineSchedule: 'interleaved-1f1b', interleavedStages: 2,
      },
      activationCheckpointing: true, flashAttention: false, mixedPrecision: 'bf16',
    },
    'IBM FSDP 7B × 128 A100': {
      clusterConfig: createMultiNodeCluster('a100-80gb', 8, 16)!,
      modelId: 'llama2-7b',
      globalBatchSize: 256, microBatchSize: 2, sequenceLength: 4096,
      strategyType: 'fsdp',
      activationCheckpointing: false, flashAttention: true, mixedPrecision: 'bf16',
    },
    'DeepSeek V3 × 2048 H800': {
      clusterConfig: createMultiNodeCluster('h800-sxm', 8, 256)!,
      modelId: 'deepseek-v3',
      globalBatchSize: 8192, microBatchSize: 2, sequenceLength: 4096,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: {
        tp: 4, pp: 8, dp: 64, ep: 32, dpType: 'fsdp', sequenceParallel: true,
        pipelineSchedule: 'dualpipe-v', interleavedStages: 1, numMicroBatches: 64,
      },
      activationCheckpointing: true, flashAttention: true, mixedPrecision: 'fp8',
    },
    'Nemotron-4 340B × 6144 H100': {
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 768)!,
      modelId: 'nemotron-4-340b',
      globalBatchSize: 768, microBatchSize: 1, sequenceLength: 4096,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: {
        tp: 8, pp: 12, dp: 64, dpType: 'fsdp', sequenceParallel: true,
        pipelineSchedule: 'interleaved-1f1b', interleavedStages: 8,
      },
      activationCheckpointing: true, checkpointingGranularity: 'selective',
      flashAttention: true, mixedPrecision: 'bf16',
    },
    'LLaMA2 7B FSDP single-node H100': {
      clusterConfig: createSingleNodeCluster('h100-sxm', 8)!,
      modelId: 'llama2-7b',
      globalBatchSize: 16, microBatchSize: 2, sequenceLength: 2048,
      strategyType: 'fsdp',
      flashAttention: true, mixedPrecision: 'bf16',
    },
    'LLaMA2 7B DDP single-node H100': {
      clusterConfig: createSingleNodeCluster('h100-sxm', 8)!,
      modelId: 'llama2-7b',
      globalBatchSize: 16, microBatchSize: 2, sequenceLength: 2048,
      strategyType: 'ddp',
      flashAttention: true, mixedPrecision: 'bf16',
    },
    'LLaMA3 8B ZeRO-1 single-node H100': {
      clusterConfig: createSingleNodeCluster('h100-sxm', 8)!,
      modelId: 'llama3-8b',
      globalBatchSize: 16, microBatchSize: 2, sequenceLength: 2048,
      strategyType: 'zero-1',
      flashAttention: true, mixedPrecision: 'bf16',
    },
    'LLaMA3 8B ZeRO-3 single-node H100': {
      clusterConfig: createSingleNodeCluster('h100-sxm', 8)!,
      modelId: 'llama3-8b',
      globalBatchSize: 16, microBatchSize: 2, sequenceLength: 2048,
      strategyType: 'zero-3',
      flashAttention: true, mixedPrecision: 'bf16',
    },
    'LLaMA2 7B TP=4 single-node H100': {
      clusterConfig: createSingleNodeCluster('h100-sxm', 8)!,
      modelId: 'llama2-7b',
      globalBatchSize: 16, microBatchSize: 2, sequenceLength: 2048,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 4 },
      flashAttention: true, mixedPrecision: 'bf16',
    },
    'LLaMA2 7B FSDP-TP-PP(1,1) single-node H100': {
      clusterConfig: createSingleNodeCluster('h100-sxm', 8)!,
      modelId: 'llama2-7b',
      globalBatchSize: 16, microBatchSize: 2, sequenceLength: 2048,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 1, pp: 1 },
      flashAttention: true, mixedPrecision: 'bf16',
    },
    'LLaMA2 7B FSDP-TP-PP(4,1) single-node H100': {
      clusterConfig: createSingleNodeCluster('h100-sxm', 8)!,
      modelId: 'llama2-7b',
      globalBatchSize: 16, microBatchSize: 2, sequenceLength: 2048,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 4, pp: 1 },
      flashAttention: true, mixedPrecision: 'bf16',
    },
    'LLaMA2 70B × 2048 A100 FSDP-TP': {
      clusterConfig: createMultiNodeCluster('a100-80gb', 8, 256)!,
      modelId: 'llama2-70b',
      globalBatchSize: 512, microBatchSize: 1, sequenceLength: 4096,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8 },
      flashAttention: true, mixedPrecision: 'bf16',
    },
    'LLaMA3 8B FSDP 4-node H100': {
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 4)!,
      modelId: 'llama3-8b',
      globalBatchSize: 64, microBatchSize: 2, sequenceLength: 2048,
      strategyType: 'fsdp',
      flashAttention: true, mixedPrecision: 'bf16',
    },
    'LLaMA2 7B FSDP 8-node A100': {
      clusterConfig: createMultiNodeCluster('a100-80gb', 8, 8)!,
      modelId: 'llama2-7b',
      globalBatchSize: 128, microBatchSize: 2, sequenceLength: 4096,
      strategyType: 'fsdp',
      flashAttention: true, mixedPrecision: 'bf16',
    },
    'Mixtral 8x7B FSDP single-node H100': {
      clusterConfig: createSingleNodeCluster('h100-sxm', 8)!,
      modelId: 'mixtral-8x7b',
      globalBatchSize: 16, microBatchSize: 2, sequenceLength: 2048,
      strategyType: 'fsdp',
      flashAttention: true, mixedPrecision: 'bf16',
    },
    'Mixtral 8x22B EP=4 × 128 H100': {
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 16)!,
      modelId: 'mixtral-8x22b',
      globalBatchSize: 256, microBatchSize: 1, sequenceLength: 4096,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 2, pp: 8, ep: 4, sequenceParallel: true },
      activationCheckpointing: true, flashAttention: true, mixedPrecision: 'bf16',
    },
    'LLaMA2 70B PP=8 FSDP-TP-PP × 32 H100': {
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 4)!,
      modelId: 'llama2-70b',
      globalBatchSize: 1024, microBatchSize: 2, sequenceLength: 2048,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 1, pp: 8 },
      flashAttention: true, mixedPrecision: 'bf16',
    },
  };

  const config = configs[name];
  if (!config) throw new Error(`Unknown golden config: ${name}`);
  return config;
}

describe('Golden parity gate (overlap model)', () => {
  for (const entry of golden) {
    it(entry.name, () => {
      const config = getConfig(entry.name);
      const metrics = getSimulationMetrics(config);

      assertClose(metrics.mfu, entry.mfu, 'mfu');
      assertClose(metrics.timing.total, entry.totalTimeMs, 'totalTimeMs');
      assertClose(metrics.timing.communication, entry.communication, 'communication');
      assertClose(metrics.timing.overlap, entry.overlap, 'overlap');
    });
  }
});
