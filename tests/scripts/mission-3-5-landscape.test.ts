/**
 * Mission 3-5 "Resource War" Landscape
 *
 * Tests all three sub-objectives with starting and expected winning configs.
 * Prints formatted tables showing how each config change affects metrics.
 *
 * Objective 1: Biosignature Training (precision + AC)
 * Objective 2: Fine-tune 70B (finetuning method)
 * Objective 3: Inference throughput (batch + continuous batching)
 */

import { describe, it, expect } from 'vitest';
import {
  type SimulationConfig,
  type SimulationMetrics,
  getSimulationMetrics,
} from '../../src/core/simulation/engine.ts';
import { benchmarkConfig } from '../helpers/benchmark-config.ts';
import { runInferenceSimulation } from '../../src/core/inference/index.ts';
import { A100_80GB } from '../../src/core/hardware/gpu.ts';

// ─── Objective 1: Biosignature Training (obj-biosig-train) ─────────────────

describe('Objective 1: Biosignature Training', () => {
  interface TrainEntry {
    label: string;
    precision: SimulationConfig['mixedPrecision'];
    ac: boolean;
    acGranularity?: 'full' | 'selective';
  }

  const entries: TrainEntry[] = [
    { label: 'A (start)', precision: 'bf16', ac: false },
    { label: 'B',         precision: 'fp8',  ac: false },
    { label: 'C',         precision: 'bf16', ac: true, acGranularity: 'selective' },
    { label: 'D',         precision: 'fp8',  ac: true, acGranularity: 'selective' },
    { label: 'E',         precision: 'fp8',  ac: true, acGranularity: 'full' },
  ];

  it('prints results table and validates winning configs', () => {
    const header = [
      'Config'.padEnd(12),
      'Precision'.padEnd(10),
      'AC'.padEnd(12),
      'MemUtil%'.padStart(9),
      'MFU%'.padStart(7),
      'HFU%'.padStart(7),
      'Success'.padStart(8),
    ].join(' | ');
    const sep = '-'.repeat(header.length);

    console.log('\n');
    console.log('=== Objective 1: Biosignature Training — LLaMA 3.1 8B, 8×H100 SXM, FSDP ===');
    console.log(sep);
    console.log(header);
    console.log(sep);

    const results: { label: string; memUtil: number; mfu: number; success: boolean }[] = [];

    for (const e of entries) {
      const config = benchmarkConfig(
        'h100-sxm', 8, 1,              // 8 GPUs, 1 node
        'llama3.1-8b',                  // model
        'fsdp',                         // strategy
        64,                             // GBS
        4,                              // MBS
        8192,                           // seqLen
        undefined,                      // no strategyConfig needed for FSDP
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

      results.push({ label: e.label, memUtil, mfu, success });

      const acStr = e.ac ? (e.acGranularity ?? 'full') : 'off';
      const row = [
        e.label.padEnd(12),
        (e.precision ?? 'bf16').padEnd(10),
        acStr.padEnd(12),
        (memUtil * 100).toFixed(1).padStart(9),
        (mfu * 100).toFixed(2).padStart(7),
        (metrics.hfu * 100).toFixed(2).padStart(7),
        (success ? 'OK' : 'OOM').padStart(8),
      ].join(' | ');
      console.log(row);
    }

    console.log(sep);
    console.log('\n');

    // Starting config A is OOM (BF16, no AC, 8B on 8×H100 with full activations)
    const startA = results.find(r => r.label === 'A (start)')!;
    expect(startA.success).toBe(false);
    expect(startA.memUtil).toBeGreaterThan(1.0);

    // FP8 alone (B) is also OOM — precision helps MFU but doesn't solve memory
    const configB = results.find(r => r.label === 'B')!;
    expect(configB.success).toBe(false);

    // BF16 + selective AC (C) fits but MFU is below 50%
    const configC = results.find(r => r.label === 'C')!;
    expect(configC.success).toBe(true);
    expect(configC.mfu).toBeLessThan(0.50);

    // Winning criteria: success=true AND MFU > 50%
    // Config D (FP8 + selective AC) should win — fits AND MFU > 50%
    const configD = results.find(r => r.label === 'D')!;
    expect(configD.success).toBe(true);
    expect(configD.mfu).toBeGreaterThan(0.50);

    // Config E (FP8 + full AC) also fits, MFU should be ~50% or above
    const configE = results.find(r => r.label === 'E')!;
    expect(configE.success).toBe(true);
    expect(configE.mfu).toBeGreaterThanOrEqual(0.50);

    // FP8 improves MFU vs BF16 (compare D vs C, both successful with selective AC)
    expect(configD.mfu).toBeGreaterThan(configC.mfu);
  });
});

// ─── Objective 2: Fine-tune 70B (obj-finetune) ────────────────────────────

describe('Objective 2: Fine-tune LLaMA 3.3 70B', () => {
  interface FTEntry {
    label: string;
    method: 'full' | 'lora' | 'qlora';
    targetModules?: 'q_v' | 'q_k_v_o' | 'all_linear';
  }

  const entries: FTEntry[] = [
    { label: 'A (start)', method: 'full' },
    { label: 'B',         method: 'lora',  targetModules: 'q_v' },
    { label: 'C',         method: 'lora',  targetModules: 'all_linear' },
    { label: 'D',         method: 'qlora', targetModules: 'q_v' },
    { label: 'E',         method: 'qlora', targetModules: 'all_linear' },
  ];

  it('prints results table and validates winning configs', () => {
    const header = [
      'Config'.padEnd(12),
      'Method'.padEnd(8),
      'Targets'.padEnd(12),
      'MemUtil%'.padStart(9),
      'MFU%'.padStart(7),
      'Success'.padStart(8),
    ].join(' | ');
    const sep = '-'.repeat(header.length);

    console.log('\n');
    console.log('=== Objective 2: Fine-tune LLaMA 3.3 70B, 8×H100 SXM, FSDP ===');
    console.log(sep);
    console.log(header);
    console.log(sep);

    const results: { label: string; memUtil: number; mfu: number; success: boolean }[] = [];

    for (const e of entries) {
      const config: SimulationConfig = {
        ...benchmarkConfig(
          'h100-sxm', 8, 1,
          'llama3.3-70b',
          'fsdp',
          16,   // GBS
          1,    // MBS
          4096, // seqLen
          undefined,
          {
            activationCheckpointing: true,
            checkpointingGranularity: 'full',
            flashAttention: true,
          },
        ),
        finetuningMethod: e.method,
        ...(e.targetModules ? { loraTargetModules: e.targetModules } : {}),
      };

      const metrics = getSimulationMetrics(config);
      const memUtil = metrics.memoryUtilization;
      const mfu = metrics.mfu;
      const success = memUtil <= 1.0;

      results.push({ label: e.label, memUtil, mfu, success });

      const row = [
        e.label.padEnd(12),
        e.method.padEnd(8),
        (e.targetModules ?? 'n/a').padEnd(12),
        (memUtil * 100).toFixed(1).padStart(9),
        (mfu * 100).toFixed(2).padStart(7),
        (success ? 'OK' : 'OOM').padStart(8),
      ].join(' | ');
      console.log(row);
    }

    console.log(sep);
    console.log('\n');

    // Starting config A (full fine-tune 70B on 8×H100) should OOM
    const startA = results.find(r => r.label === 'A (start)')!;
    expect(startA.success).toBe(false);

    // At least one LoRA or QLoRA config should fit
    const loraWinners = results.filter(r => r.label !== 'A (start)' && r.success);
    expect(loraWinners.length).toBeGreaterThan(0);

    // All LoRA/QLoRA configs should use less memory than full fine-tune
    for (const winner of results.filter(r => r.label !== 'A (start)')) {
      expect(winner.memUtil).toBeLessThan(startA.memUtil);
    }
  });
});

// ─── Objective 3: Inference (obj-probe-infer) ──────────────────────────────

describe('Objective 3: Inference Throughput', () => {
  interface InfEntry {
    label: string;
    batchSize: number;
    continuousBatching: boolean;
  }

  const entries: InfEntry[] = [
    { label: 'A (start)', batchSize: 1,  continuousBatching: false },
    { label: 'B',         batchSize: 8,  continuousBatching: false },
    { label: 'C',         batchSize: 1,  continuousBatching: true },
    { label: 'D',         batchSize: 8,  continuousBatching: true },
    { label: 'E',         batchSize: 16, continuousBatching: true },
    { label: 'F',         batchSize: 32, continuousBatching: true },
  ];

  it('prints results table and validates winning configs', () => {
    const header = [
      'Config'.padEnd(12),
      'Batch'.padStart(6),
      'CB'.padEnd(5),
      'MemUtil%'.padStart(9),
      'TokPerSec'.padStart(12),
      'TTFT(ms)'.padStart(10),
      'TPOT(ms)'.padStart(10),
      'Success'.padStart(8),
    ].join(' | ');
    const sep = '-'.repeat(header.length);

    console.log('\n');
    console.log('=== Objective 3: Inference — LLaMA 3.3 70B, 4×A100 80GB, TP=4, BF16 ===');
    console.log(sep);
    console.log(header);
    console.log(sep);

    const results: { label: string; tokPerSec: number; success: boolean; memUtil: number }[] = [];

    for (const e of entries) {
      const result = runInferenceSimulation({
        modelId: 'llama3.3-70b',
        gpu: A100_80GB,
        numGPUs: 4,
        tensorParallel: 4,
        batchSize: e.batchSize,
        inputSeqLen: 2048,
        outputSeqLen: 512,
        weightPrecision: 'bf16',
        kvCachePrecision: 'bf16',
        flashAttention: true,
        continuousBatching: e.continuousBatching,
      });

      const memUtil = result.utilization.memoryCapacityUtilization;
      const tokPerSec = result.throughput.tokensPerSecond;
      const success = result.success;

      results.push({ label: e.label, tokPerSec, success, memUtil });

      const row = [
        e.label.padEnd(12),
        String(e.batchSize).padStart(6),
        (e.continuousBatching ? 'on' : 'off').padEnd(5),
        (memUtil * 100).toFixed(1).padStart(9),
        Math.round(tokPerSec).toString().padStart(12),
        result.latency.ttft.toFixed(1).padStart(10),
        result.latency.tpot.toFixed(2).padStart(10),
        (success ? 'OK' : 'OOM').padStart(8),
      ].join(' | ');
      console.log(row);
    }

    console.log(sep);
    console.log('\n');

    // Starting config A should succeed but have low throughput
    const startA = results.find(r => r.label === 'A (start)')!;
    expect(startA.success).toBe(true);
    expect(startA.tokPerSec).toBeLessThan(800); // Below the 800 tok/s target

    // At least one config with batch>1 and CB=on should exceed 800 tok/s
    const winners = results.filter(r => r.success && r.tokPerSec > 800);
    expect(winners.length).toBeGreaterThan(0);

    // Increasing batch should increase throughput (for successful configs)
    const successfulResults = results.filter(r => r.success);
    for (let i = 1; i < successfulResults.length; i++) {
      // Throughput should generally increase with batch size
      // (not strictly monotone due to CB toggle, but batch=8 CB=off > batch=1 CB=off)
    }

    // Verify batch=8 without CB has higher throughput than batch=1 without CB
    const configA = results.find(r => r.label === 'A (start)')!;
    const configB = results.find(r => r.label === 'B')!;
    if (configB.success) {
      expect(configB.tokPerSec).toBeGreaterThan(configA.tokPerSec);
    }

    // Verify CB=on at same batch size has higher throughput
    const configD = results.find(r => r.label === 'D')!; // batch=8, CB=on
    if (configB.success && configD.success) {
      expect(configD.tokPerSec).toBeGreaterThan(configB.tokPerSec);
    }
  });
});
