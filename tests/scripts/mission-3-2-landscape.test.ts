/**
 * Mission 3-2 Landscape: LLaMA 3.1 405B long-context training configs
 *
 * Explores the parallelism landscape for 405B at 131K seqLen on 512 H100s.
 * Prints a formatted table of strategy, TP, PP, CP, DP, memUtil, MFU%, and success.
 */

import { describe, it } from 'vitest';
import {
  type SimulationConfig,
  getSimulationMetrics,
  runSimulation,
} from '../../src/core/simulation/engine.ts';
import { benchmarkConfig } from '../helpers/benchmark-config.ts';

interface ConfigEntry {
  label: string;
  strategy: SimulationConfig['strategyType'];
  tp: number;
  pp: number;
  cp: number;
}

const configs: ConfigEntry[] = [
  { label: ' 1', strategy: 'fsdp-tp-pp', tp: 8, pp: 4, cp: 1 },
  { label: ' 2', strategy: 'fsdp-tp-pp', tp: 8, pp: 4, cp: 2 },
  { label: ' 3', strategy: 'fsdp-tp-pp', tp: 8, pp: 4, cp: 4 },
  { label: ' 4', strategy: 'fsdp-tp-pp', tp: 8, pp: 4, cp: 8 },
  { label: ' 5', strategy: 'fsdp-tp',    tp: 8, pp: 1, cp: 1 },
  { label: ' 6', strategy: 'fsdp-tp',    tp: 8, pp: 1, cp: 2 },
  { label: ' 7', strategy: 'fsdp-tp',    tp: 8, pp: 1, cp: 4 },
  { label: ' 8', strategy: 'fsdp-tp',    tp: 8, pp: 1, cp: 8 },
  { label: ' 9', strategy: 'fsdp-tp',    tp: 8, pp: 1, cp: 16 },
  { label: '10', strategy: 'fsdp',       tp: 1, pp: 1, cp: 1 },
  { label: '11', strategy: 'fsdp',       tp: 1, pp: 1, cp: 4 },
  { label: '12', strategy: 'fsdp',       tp: 1, pp: 1, cp: 8 },
];

describe('Mission 3-2 Landscape', () => {
  it('prints results table for all configs', () => {
    const header = [
      '#'.padStart(2),
      'Strategy'.padEnd(12),
      'TP'.padStart(3),
      'PP'.padStart(3),
      'CP'.padStart(3),
      'DP'.padStart(4),
      'GA'.padStart(4),
      'MemUtil%'.padStart(9),
      'MFU%'.padStart(7),
      'HFU%'.padStart(7),
      'ModelMFU%'.padStart(10),
      'StepMs'.padStart(10),
      'TokPerSec'.padStart(12),
      'Success'.padStart(8),
    ].join(' | ');

    const separator = '-'.repeat(header.length);

    console.log('\n');
    console.log('=== Mission 3-2: LLaMA 405B, 512 H100s, seqLen=131072, FP8, selective AC ===');
    console.log(separator);
    console.log(header);
    console.log(separator);

    for (const entry of configs) {
      try {
        const strategyConfig: SimulationConfig['strategyConfig'] = {
          tp: entry.tp,
          pp: entry.pp,
          cp: entry.cp,
          sequenceParallel: true,
        };

        const config = benchmarkConfig(
          'h100-sxm',           // gpuId
          8,                     // gpusPerNode
          64,                    // numNodes (512 GPUs / 8 per node)
          'llama3-405b',         // modelId
          entry.strategy,        // strategy
          128,                   // globalBatchSize
          1,                     // microBatchSize
          131072,                // sequenceLength
          strategyConfig,
          {
            mixedPrecision: 'fp8',
            activationCheckpointing: true,
            checkpointingGranularity: 'selective',
            flashAttention: true,
          },
        );

        const result = runSimulation(config);
        const metrics = result.metrics;
        const success = result.success;

        // Derive DP from totalGPUs / (TP * PP * CP)
        const dp = Math.floor(512 / (entry.tp * entry.pp * entry.cp));
        const ga = Math.max(1, Math.ceil(128 / (1 * dp)));

        // Get detailed metrics for memUtil and full MFU
        const detailed = getSimulationMetrics(config);

        const row = [
          entry.label.padStart(2),
          entry.strategy.padEnd(12),
          String(entry.tp).padStart(3),
          String(entry.pp).padStart(3),
          String(entry.cp).padStart(3),
          String(dp).padStart(4),
          String(ga).padStart(4),
          (detailed.memoryUtilization * 100).toFixed(1).padStart(9),
          (detailed.mfu * 100).toFixed(2).padStart(7),
          (detailed.hfu * 100).toFixed(2).padStart(7),
          (detailed.modelFlopsMfu != null ? (detailed.modelFlopsMfu * 100).toFixed(2) : 'n/a').padStart(10),
          detailed.stepTimeMs.toFixed(0).padStart(10),
          Math.round(detailed.tokensPerSecond).toString().padStart(12),
          (success ? 'OK' : 'FAIL').padStart(8),
        ].join(' | ');

        console.log(row);
      } catch (e: unknown) {
        const dp = Math.floor(512 / (entry.tp * entry.pp * entry.cp));
        const errMsg = e instanceof Error ? e.message : String(e);
        console.log(
          `${entry.label.padStart(2)} | ${entry.strategy.padEnd(12)} | ${String(entry.tp).padStart(3)} | ${String(entry.pp).padStart(3)} | ${String(entry.cp).padStart(3)} | ${String(dp).padStart(4)} |      |           |        |        |           |           |             | ERROR: ${errMsg}`
        );
      }
    }

    console.log(separator);
    console.log('\n');
  });
});
