/**
 * Mission 3-5 Activation Checkpointing Necessity Sweep
 *
 * Sweeps seqLen x config matrix to find where AC becomes physically necessary
 * regardless of whether the player uses TP.
 *
 * Base: LLaMA 3.1 8B, 8x H100-SXM (1 node), GBS=64, FA=true
 */

import { describe, it } from 'vitest';
import {
  type SimulationConfig,
  getSimulationMetrics,
} from '../../src/core/simulation/engine.ts';
import { benchmarkConfig } from '../helpers/benchmark-config.ts';

interface ConfigEntry {
  label: string;
  strategy: SimulationConfig['strategyType'];
  tp: number;
  precision: SimulationConfig['mixedPrecision'];
  ac: boolean;
  acGranularity?: 'full' | 'selective';
  mbs: number;
  sequenceParallel?: boolean;
}

const configs: ConfigEntry[] = [
  { label: 'A', strategy: 'fsdp',    tp: 1, precision: 'bf16', ac: false,                          mbs: 4 },
  { label: 'B', strategy: 'fsdp',    tp: 1, precision: 'fp8',  ac: false,                          mbs: 4 },
  { label: 'C', strategy: 'fsdp',    tp: 1, precision: 'fp8',  ac: true, acGranularity: 'selective', mbs: 4 },
  { label: 'D', strategy: 'fsdp-tp', tp: 8, precision: 'fp8',  ac: false,                          mbs: 4, sequenceParallel: true },
  { label: 'E', strategy: 'fsdp-tp', tp: 8, precision: 'fp8',  ac: true, acGranularity: 'selective', mbs: 4, sequenceParallel: true },
  { label: 'F', strategy: 'fsdp',    tp: 1, precision: 'bf16', ac: true, acGranularity: 'selective', mbs: 4 },
];

const seqLens = [8192, 16384, 32768, 65536];

describe('Mission 3-5 AC Sweep: seqLen x config matrix', () => {
  it('prints full sweep table', () => {
    const header = [
      'SeqLen'.padEnd(7),
      'Cfg'.padEnd(4),
      'Strategy'.padEnd(9),
      'TP'.padStart(3),
      'Prec'.padEnd(5),
      'AC'.padEnd(10),
      'MBS'.padStart(4),
      'MemUtil%'.padStart(9),
      'MFU%'.padStart(7),
      'Result'.padStart(7),
    ].join(' | ');
    const sep = '-'.repeat(header.length);

    console.log('\n');
    console.log('=== Mission 3-5 AC Sweep: LLaMA 3.1 8B, 8x H100-SXM, GBS=64, FA=true ===');
    console.log(sep);
    console.log(header);
    console.log(sep);

    for (const seqLen of seqLens) {
      for (const e of configs) {
        const strategyConfig: SimulationConfig['strategyConfig'] =
          e.strategy === 'fsdp-tp'
            ? { tp: e.tp, sequenceParallel: e.sequenceParallel ?? true }
            : undefined;

        const config = benchmarkConfig(
          'h100-sxm', 8, 1,
          'llama3.1-8b',
          e.strategy,
          64,       // GBS
          e.mbs,
          seqLen,
          strategyConfig,
          {
            mixedPrecision: e.precision,
            activationCheckpointing: e.ac,
            checkpointingGranularity: e.acGranularity,
            flashAttention: true,
          },
        );

        const metrics = getSimulationMetrics(config);
        const memUtil = metrics.memoryUtilization;
        const mfu = metrics.mfu;
        const success = memUtil <= 1.0;

        const acStr = e.ac ? (e.acGranularity ?? 'full') : 'off';
        const row = [
          String(seqLen).padEnd(7),
          e.label.padEnd(4),
          e.strategy.padEnd(9),
          String(e.tp).padStart(3),
          (e.precision ?? 'bf16').padEnd(5),
          acStr.padEnd(10),
          String(e.mbs).padStart(4),
          (memUtil * 100).toFixed(1).padStart(9),
          (mfu * 100).toFixed(2).padStart(7),
          (success ? 'OK' : 'OOM').padStart(7),
        ].join(' | ');
        console.log(row);
      }
      console.log(sep);
    }

    console.log('\n');
    console.log('Legend:');
    console.log('  A = FSDP, TP=1, BF16, AC=off (starting config)');
    console.log('  B = FSDP, TP=1, FP8,  AC=off');
    console.log('  C = FSDP, TP=1, FP8,  AC=selective (intended winner)');
    console.log('  D = FSDP-TP, TP=8, FP8, AC=off (TP escape attempt)');
    console.log('  E = FSDP-TP, TP=8, FP8, AC=selective (TP + AC)');
    console.log('  F = FSDP, TP=1, BF16, AC=selective');
    console.log('\n');

    // Summary: for each seqLen, show which configs fit
    console.log('=== Summary: which configs fit at each seqLen ===');
    for (const seqLen of seqLens) {
      const fits: string[] = [];
      const ooms: string[] = [];
      for (const e of configs) {
        const strategyConfig: SimulationConfig['strategyConfig'] =
          e.strategy === 'fsdp-tp'
            ? { tp: e.tp, sequenceParallel: e.sequenceParallel ?? true }
            : undefined;

        const config = benchmarkConfig(
          'h100-sxm', 8, 1,
          'llama3.1-8b',
          e.strategy,
          64,
          e.mbs,
          seqLen,
          strategyConfig,
          {
            mixedPrecision: e.precision,
            activationCheckpointing: e.ac,
            checkpointingGranularity: e.acGranularity,
            flashAttention: true,
          },
        );

        const metrics = getSimulationMetrics(config);
        if (metrics.memoryUtilization <= 1.0) {
          fits.push(e.label);
        } else {
          ooms.push(e.label);
        }
      }
      console.log(`  seqLen=${seqLen}: FIT=[${fits.join(',')}]  OOM=[${ooms.join(',')}]`);
    }
    console.log('\n');
  });
});
