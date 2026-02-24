/**
 * Dump golden baseline MFU snapshot for overlap refactor parity gate.
 *
 * Run to regenerate golden overlap baselines.
 * Output: tests/golden/overlap-baseline.json
 *
 * Usage: npx tsx scripts/dump-golden.ts
 */
import { writeFileSync, mkdirSync } from 'fs';
import { resolve, dirname } from 'path';
import {
  type SimulationConfig,
  getSimulationMetrics,
} from '../src/core/simulation/engine.ts';
import { createMultiNodeCluster, createSingleNodeCluster } from '../src/core/hardware/topology.ts';

interface GoldenEntry {
  name: string;
  mfu: number;
  totalTimeMs: number;
  communication: number;
  overlap: number;
  communicationExposedMs: number;
  overlapHiddenFraction: number;
}

function run(name: string, config: SimulationConfig): GoldenEntry {
  const metrics = getSimulationMetrics(config);
  return {
    name,
    mfu: metrics.mfu,
    totalTimeMs: metrics.timing.total,
    communication: metrics.timing.communication,
    overlap: metrics.timing.overlap,
    communicationExposedMs: metrics.communicationExposedMs,
    overlapHiddenFraction: metrics.overlapHiddenFraction,
  };
}

const entries: GoldenEntry[] = [];

// ── Calibration benchmarks (from retune.ts train set) ──

entries.push(run('LLaMA 3.1 405B × 16384 H100', {
  clusterConfig: createMultiNodeCluster('h100-sxm', 8, 2048)!,
  modelId: 'llama3-405b',
  globalBatchSize: 2048, microBatchSize: 1, sequenceLength: 8192,
  strategyType: 'fsdp-tp-pp',
  strategyConfig: {
    tp: 8, pp: 16, dp: 16, dpType: 'fsdp', sequenceParallel: true,
    pipelineSchedule: 'interleaved-1f1b', interleavedStages: 4,
  },
  activationCheckpointing: false, flashAttention: true, mixedPrecision: 'bf16',
}));

entries.push(run('GPT-3 175B × 1024 A100', {
  clusterConfig: createMultiNodeCluster('a100-80gb', 8, 128)!,
  modelId: 'gpt3-175b',
  globalBatchSize: 1536, microBatchSize: 1, sequenceLength: 2048,
  strategyType: 'ddp-tp-pp',
  strategyConfig: {
    tp: 8, pp: 8, dp: 16, dpType: 'ddp', sequenceParallel: false,
    pipelineSchedule: 'interleaved-1f1b', interleavedStages: 2,
  },
  activationCheckpointing: true, flashAttention: false, mixedPrecision: 'bf16',
}));

entries.push(run('IBM FSDP 7B × 128 A100', {
  clusterConfig: createMultiNodeCluster('a100-80gb', 8, 16)!,
  modelId: 'llama2-7b',
  globalBatchSize: 256, microBatchSize: 2, sequenceLength: 4096,
  strategyType: 'fsdp',
  activationCheckpointing: false, flashAttention: true, mixedPrecision: 'bf16',
}));

entries.push(run('DeepSeek V3 × 2048 H800', {
  clusterConfig: createMultiNodeCluster('h800-sxm', 8, 256)!,
  modelId: 'deepseek-v3',
  globalBatchSize: 8192, microBatchSize: 2, sequenceLength: 4096,
  strategyType: 'fsdp-tp-pp',
  strategyConfig: {
    tp: 4, pp: 8, dp: 64, ep: 32, dpType: 'fsdp', sequenceParallel: true,
    pipelineSchedule: 'dualpipe-v', interleavedStages: 1, numMicroBatches: 64,
  },
  activationCheckpointing: true, flashAttention: true, mixedPrecision: 'fp8',
}));

entries.push(run('Nemotron-4 340B × 6144 H100', {
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
}));

// ── Standalone strategy paths ──

entries.push(run('LLaMA2 7B FSDP single-node H100', {
  clusterConfig: createSingleNodeCluster('h100-sxm', 8)!,
  modelId: 'llama2-7b',
  globalBatchSize: 16, microBatchSize: 2, sequenceLength: 2048,
  strategyType: 'fsdp',
  flashAttention: true, mixedPrecision: 'bf16',
}));

entries.push(run('LLaMA2 7B DDP single-node H100', {
  clusterConfig: createSingleNodeCluster('h100-sxm', 8)!,
  modelId: 'llama2-7b',
  globalBatchSize: 16, microBatchSize: 2, sequenceLength: 2048,
  strategyType: 'ddp',
  flashAttention: true, mixedPrecision: 'bf16',
}));

entries.push(run('LLaMA3 8B ZeRO-1 single-node H100', {
  clusterConfig: createSingleNodeCluster('h100-sxm', 8)!,
  modelId: 'llama3-8b',
  globalBatchSize: 16, microBatchSize: 2, sequenceLength: 2048,
  strategyType: 'zero-1',
  flashAttention: true, mixedPrecision: 'bf16',
}));

entries.push(run('LLaMA3 8B ZeRO-3 single-node H100', {
  clusterConfig: createSingleNodeCluster('h100-sxm', 8)!,
  modelId: 'llama3-8b',
  globalBatchSize: 16, microBatchSize: 2, sequenceLength: 2048,
  strategyType: 'zero-3',
  flashAttention: true, mixedPrecision: 'bf16',
}));

entries.push(run('LLaMA2 7B TP=4 single-node H100', {
  clusterConfig: createSingleNodeCluster('h100-sxm', 8)!,
  modelId: 'llama2-7b',
  globalBatchSize: 16, microBatchSize: 2, sequenceLength: 2048,
  strategyType: 'fsdp-tp',
  strategyConfig: { tp: 4 },
  flashAttention: true, mixedPrecision: 'bf16',
}));

// ── Route consistency pairs ──

entries.push(run('LLaMA2 7B FSDP-TP-PP(1,1) single-node H100', {
  clusterConfig: createSingleNodeCluster('h100-sxm', 8)!,
  modelId: 'llama2-7b',
  globalBatchSize: 16, microBatchSize: 2, sequenceLength: 2048,
  strategyType: 'fsdp-tp-pp',
  strategyConfig: { tp: 1, pp: 1 },
  flashAttention: true, mixedPrecision: 'bf16',
}));

entries.push(run('LLaMA2 7B FSDP-TP-PP(4,1) single-node H100', {
  clusterConfig: createSingleNodeCluster('h100-sxm', 8)!,
  modelId: 'llama2-7b',
  globalBatchSize: 16, microBatchSize: 2, sequenceLength: 2048,
  strategyType: 'fsdp-tp-pp',
  strategyConfig: { tp: 4, pp: 1 },
  flashAttention: true, mixedPrecision: 'bf16',
}));

// ── Additional multi-node / multi-strategy configs ──

entries.push(run('LLaMA2 70B × 2048 A100 FSDP-TP', {
  clusterConfig: createMultiNodeCluster('a100-80gb', 8, 256)!,
  modelId: 'llama2-70b',
  globalBatchSize: 512, microBatchSize: 1, sequenceLength: 4096,
  strategyType: 'fsdp-tp',
  strategyConfig: { tp: 8 },
  flashAttention: true, mixedPrecision: 'bf16',
}));

entries.push(run('LLaMA3 8B FSDP 4-node H100', {
  clusterConfig: createMultiNodeCluster('h100-sxm', 8, 4)!,
  modelId: 'llama3-8b',
  globalBatchSize: 64, microBatchSize: 2, sequenceLength: 2048,
  strategyType: 'fsdp',
  flashAttention: true, mixedPrecision: 'bf16',
}));

entries.push(run('LLaMA2 7B FSDP 8-node A100', {
  clusterConfig: createMultiNodeCluster('a100-80gb', 8, 8)!,
  modelId: 'llama2-7b',
  globalBatchSize: 128, microBatchSize: 2, sequenceLength: 4096,
  strategyType: 'fsdp',
  flashAttention: true, mixedPrecision: 'bf16',
}));

entries.push(run('Mixtral 8x7B FSDP single-node H100', {
  clusterConfig: createSingleNodeCluster('h100-sxm', 8)!,
  modelId: 'mixtral-8x7b',
  globalBatchSize: 16, microBatchSize: 2, sequenceLength: 2048,
  strategyType: 'fsdp',
  flashAttention: true, mixedPrecision: 'bf16',
}));

entries.push(run('Mixtral 8x22B EP=4 × 128 H100', {
  clusterConfig: createMultiNodeCluster('h100-sxm', 8, 16)!,
  modelId: 'mixtral-8x22b',
  globalBatchSize: 256, microBatchSize: 1, sequenceLength: 4096,
  strategyType: 'fsdp-tp-pp',
  strategyConfig: { tp: 2, pp: 8, ep: 4, sequenceParallel: true },
  activationCheckpointing: true, flashAttention: true, mixedPrecision: 'bf16',
}));

// ── PP standalone ──

entries.push(run('LLaMA2 70B PP=8 FSDP-TP-PP × 32 H100', {
  clusterConfig: createMultiNodeCluster('h100-sxm', 8, 4)!,
  modelId: 'llama2-70b',
  globalBatchSize: 1024, microBatchSize: 2, sequenceLength: 2048,
  strategyType: 'fsdp-tp-pp',
  strategyConfig: { tp: 1, pp: 8 },
  flashAttention: true, mixedPrecision: 'bf16',
}));

// ── Write output ──

const outPath = resolve(import.meta.dirname!, '../tests/golden/overlap-baseline.json');
mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, JSON.stringify(entries, null, 2) + '\n');
console.log(`Wrote ${entries.length} golden entries to ${outPath}`);

for (const e of entries) {
  console.log(`  ${e.name}: mfu=${(e.mfu * 100).toFixed(2)}% total=${e.totalTimeMs.toFixed(2)}ms exposed=${e.communicationExposedMs.toFixed(2)}ms hidden=${(e.overlapHiddenFraction * 100).toFixed(1)}%`);
}
