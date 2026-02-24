/**
 * Auto-optimizer baseline check
 *
 * Runs the training optimizer against calibrated baseline configs and verifies
 * that the optimized MFU does not exceed the baseline MFU by more than 2pp.
 */

import { getSimulationMetrics, type SimulationConfig } from '../src/core/simulation/engine.ts';
import { optimizeTraining } from '../src/core/simulation/optimizer.ts';
import { createMultiNodeCluster } from '../src/core/hardware/topology.ts';

interface Baseline {
  name: string;
  config: SimulationConfig;
  targetTokens: number;
  seqLength: number;
  expectedMfu: number; // stated baseline MFU (fraction, e.g. 0.398)
}

const baselines: Baseline[] = [
  // 1. LLaMA 3.1 405B — 16384 H100 SXM
  {
    name: 'LLaMA 3.1 405B × 16384 H100',
    config: {
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 2048)!,
      modelId: 'llama3-405b',
      globalBatchSize: 2048,
      microBatchSize: 1,
      sequenceLength: 8192,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: {
        tp: 8,
        pp: 16,
        sequenceParallel: true,
        pipelineSchedule: 'interleaved-1f1b',
        interleavedStages: 4,
      },
      activationCheckpointing: false,
      flashAttention: true,
      mixedPrecision: 'bf16',
    },
    targetTokens: 15e12,
    seqLength: 8192,
    expectedMfu: 0.398,
  },

  // 2. Nemotron-4 340B — 6144 H100 SXM
  {
    name: 'Nemotron-4 340B × 6144 H100',
    config: {
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 768)!,
      modelId: 'nemotron-4-340b',
      globalBatchSize: 768,
      microBatchSize: 1,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: {
        tp: 8,
        pp: 12,
        sequenceParallel: true,
        pipelineSchedule: 'interleaved-1f1b',
        interleavedStages: 8,
      },
      activationCheckpointing: true,
      checkpointingGranularity: 'selective',
      flashAttention: true,
      mixedPrecision: 'bf16',
    },
    targetTokens: 9e12,
    seqLength: 4096,
    expectedMfu: 0.406,
  },

  // 3. GPT-3 175B — 1024 A100 80GB
  {
    name: 'GPT-3 175B × 1024 A100',
    config: {
      clusterConfig: createMultiNodeCluster('a100-80gb', 8, 128)!,
      modelId: 'gpt3-175b',
      globalBatchSize: 1536,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'ddp-tp-pp',
      strategyConfig: {
        tp: 8,
        pp: 8,
        pipelineSchedule: 'interleaved-1f1b',
        interleavedStages: 2,
      },
      activationCheckpointing: true,
      checkpointingGranularity: 'full',
      flashAttention: false,
      mixedPrecision: 'bf16',
    },
    targetTokens: 300e9,
    seqLength: 2048,
    expectedMfu: 0.440,
  },

  // 4. DeepSeek V3 — 2048 H800 SXM
  {
    name: 'DeepSeek V3 × 2048 H800',
    config: {
      clusterConfig: createMultiNodeCluster('h800-sxm', 8, 256)!,
      modelId: 'deepseek-v3',
      globalBatchSize: 8192,
      microBatchSize: 2,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: {
        tp: 4,
        pp: 8,
        ep: 32,
        sequenceParallel: true,
        pipelineSchedule: 'dualpipe-v',
        numMicroBatches: 64, // GBS=8192, MBS=2, DP=64 → GA=64
      },
      activationCheckpointing: true,
      checkpointingGranularity: 'selective',
      flashAttention: true,
      mixedPrecision: 'fp8',
    },
    targetTokens: 15e12,
    seqLength: 4096,
    expectedMfu: 0.480,
  },

  // 5. Grok-1 — 256 H100 SXM
  {
    name: 'Grok-1 × 256 H100',
    config: {
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 32)!,
      modelId: 'grok-1',
      globalBatchSize: 2048,
      microBatchSize: 2,
      sequenceLength: 8192,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: {
        tp: 8,
        pp: 2,
        ep: 4,
        pipelineSchedule: 'interleaved-1f1b',
        interleavedStages: 2,
      },
      activationCheckpointing: true,
      checkpointingGranularity: 'full',
      flashAttention: true,
      mixedPrecision: 'bf16',
    },
    targetTokens: 3e12,
    seqLength: 8192,
    expectedMfu: 0.422,
  },

  // 6. Qwen3 32B — 64 H100 SXM
  {
    name: 'Qwen3 32B × 64 H100',
    config: {
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 8)!,
      modelId: 'qwen3-32b',
      globalBatchSize: 2048,
      microBatchSize: 8,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp',
      strategyConfig: {
        tp: 4,
        sequenceParallel: true,
      },
      activationCheckpointing: true,
      checkpointingGranularity: 'full',
      flashAttention: true,
      mixedPrecision: 'bf16',
    },
    targetTokens: 36e12,
    seqLength: 4096,
    expectedMfu: 0.424,
  },
];

// ── Run ──────────────────────────────────────────────────────────────

console.log('Auto-Optimizer Baseline Check');
console.log('═'.repeat(80));
console.log('Pass criterion: optimized MFU must not exceed baseline MFU by > 2pp\n');

let allPassed = true;

for (const b of baselines) {
  // 1. Simulate baseline to confirm MFU
  const baselineMetrics = getSimulationMetrics(b.config);
  const baselineMfu = baselineMetrics.mfu;

  // 2. Run optimizer
  const result = optimizeTraining(b.config, b.targetTokens, b.seqLength);

  // 3. Simulate optimized config to get its MFU
  let optimizedMfu = baselineMfu;
  if (result.success) {
    const optimizedMetrics = getSimulationMetrics(result.optimizedConfig);
    optimizedMfu = optimizedMetrics.mfu;
  }

  const deltaPp = (optimizedMfu - baselineMfu) * 100;
  const pass = deltaPp <= 2.0;
  if (!pass) allPassed = false;

  const status = pass ? 'PASS' : 'FAIL';
  const icon = pass ? '✓' : '✗';

  console.log(`${icon} ${status}  ${b.name}`);
  console.log(`  Baseline MFU:  ${(baselineMfu * 100).toFixed(1)}%  (stated: ${(b.expectedMfu * 100).toFixed(1)}%)`);
  console.log(`  Optimized MFU: ${(optimizedMfu * 100).toFixed(1)}%`);
  console.log(`  Delta:         ${deltaPp >= 0 ? '+' : ''}${deltaPp.toFixed(1)}pp  ${pass ? '(<= 2pp)' : '(> 2pp!)'}`);
  console.log(`  Sims: ${result.totalSimulations} (fix=${result.phases.fix} greedy=${result.phases.greedy} explore=${result.phases.explore})`);

  if (result.changelog.length > 0) {
    console.log(`  Changes: ${result.changelog.map(c => `${c.field}: ${c.from}→${c.to}`).join(', ')}`);
  } else {
    console.log(`  Changes: none (baseline already optimal)`);
  }
  console.log();
}

console.log('═'.repeat(80));
console.log(allPassed ? 'ALL PASSED — no optimized config exceeds baseline by > 2pp' : 'SOME FAILED');
process.exit(allPassed ? 0 : 1);
