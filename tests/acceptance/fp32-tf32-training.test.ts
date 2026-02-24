/**
 * FP32/TF32 Training Validation Tests
 *
 * End-to-end tests validating FP32 and TF32 precision training:
 * - Throughput ordering across precisions
 * - MFU ranges with tight bounds
 * - Memory footprint comparison
 * - V100 FP32-only training (TF32 fallback)
 * - TF32 ≈ BF16 on Hopper (identical TFLOPS)
 * - Regression: no MFU > 100% or NaN for any precision
 */

import { describe, it, expect } from 'vitest';
import { runSimulation, type SimulationConfig } from '../../src/core/simulation/engine.ts';
import { getEffectiveTFLOPS, V100_32GB, H100_SXM } from '../../src/core/hardware/gpu.ts';
import { createCluster, createNode } from '../../src/core/hardware/topology.ts';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const v100Cluster = createCluster(createNode(V100_32GB, 8), 1, 'single-node');

/** Llama 3 8B on 8 GPUs, FSDP, activation checkpointing */
function llama8bConfig(
  clusterOrId: string | { clusterId?: string; clusterConfig?: ReturnType<typeof createCluster> },
  precision: SimulationConfig['mixedPrecision'],
): SimulationConfig {
  const cluster = typeof clusterOrId === 'string'
    ? { clusterId: clusterOrId }
    : clusterOrId;
  return {
    modelId: 'llama3-8b',
    ...cluster,
    globalBatchSize: 32,
    microBatchSize: 2,
    sequenceLength: 4096,
    strategyType: 'fsdp',
    strategyConfig: { tp: 1, pp: 1 },
    activationCheckpointing: true,
    mixedPrecision: precision,
  };
}

/** GPT-3 125M — small enough for V100 32GB in FP32 */
function gpt125mConfig(
  clusterOrId: string | { clusterId?: string; clusterConfig?: ReturnType<typeof createCluster> },
  precision: SimulationConfig['mixedPrecision'],
): SimulationConfig {
  const cluster = typeof clusterOrId === 'string'
    ? { clusterId: clusterOrId }
    : clusterOrId;
  return {
    modelId: 'gpt3-125m',
    ...cluster,
    globalBatchSize: 64,
    microBatchSize: 4,
    sequenceLength: 2048,
    strategyType: 'fsdp',
    strategyConfig: { tp: 1, pp: 1 },
    activationCheckpointing: true,
    mixedPrecision: precision,
  };
}

function sim(config: SimulationConfig) {
  const result = runSimulation(config);
  expect(result.success, 'simulation should succeed').toBe(true);
  return result.metrics;
}

// ===================================================================
// Section 1: FP32 vs TF32 vs BF16 throughput ordering
// ===================================================================
describe('FP32 vs TF32 vs BF16 throughput ordering', () => {
  describe('A100 80GB — TF32=156, BF16=312, FP32=19.5 TFLOPS', () => {
    it('step time ordering: fp32 >> tf32 > bf16', () => {
      const bf16 = sim(llama8bConfig('8x-a100', 'bf16'));
      const tf32 = sim(llama8bConfig('8x-a100', 'tf32'));
      const fp32 = sim(llama8bConfig('8x-a100', 'fp32'));

      expect(fp32.avgStepTimeMs).toBeGreaterThan(tf32.avgStepTimeMs);
      expect(tf32.avgStepTimeMs).toBeGreaterThan(bf16.avgStepTimeMs);
    });

    it('TF32/BF16 step time ratio ∈ [1.5, 2.5] (raw TFLOPS ratio = 2x)', () => {
      const bf16 = sim(llama8bConfig('8x-a100', 'bf16'));
      const tf32 = sim(llama8bConfig('8x-a100', 'tf32'));
      // Actual: 8483/4436 ≈ 1.91
      const ratio = tf32.avgStepTimeMs / bf16.avgStepTimeMs;
      expect(ratio).toBeGreaterThan(1.5);
      expect(ratio).toBeLessThan(2.5);
    });

    it('FP32/BF16 step time ratio ∈ [10, 20] (raw TFLOPS ratio = 16x)', () => {
      const bf16 = sim(llama8bConfig('8x-a100', 'bf16'));
      const fp32 = sim(llama8bConfig('8x-a100', 'fp32'));
      // Actual: 64811/4436 ≈ 14.6
      const ratio = fp32.avgStepTimeMs / bf16.avgStepTimeMs;
      expect(ratio).toBeGreaterThan(10);
      expect(ratio).toBeLessThan(20);
    });
  });

  describe('H100 SXM — TF32=BF16=989, FP32=67 TFLOPS', () => {
    it('TF32/BF16 step time ratio ∈ [0.95, 1.10] (same TFLOPS)', () => {
      const bf16 = sim(llama8bConfig('8x-h100', 'bf16'));
      const tf32 = sim(llama8bConfig('8x-h100', 'tf32'));
      // Actual: 1537/1522 ≈ 1.01
      const ratio = tf32.avgStepTimeMs / bf16.avgStepTimeMs;
      expect(ratio).toBeGreaterThan(0.95);
      expect(ratio).toBeLessThan(1.10);
    });

    it('FP32/BF16 step time ratio ∈ [8, 18] (raw TFLOPS ratio ≈ 15x)', () => {
      const bf16 = sim(llama8bConfig('8x-h100', 'bf16'));
      const fp32 = sim(llama8bConfig('8x-h100', 'fp32'));
      // Actual: 19004/1522 ≈ 12.5
      const ratio = fp32.avgStepTimeMs / bf16.avgStepTimeMs;
      expect(ratio).toBeGreaterThan(8);
      expect(ratio).toBeLessThan(18);
    });
  });
});

// ===================================================================
// Section 2: TF32 MFU reasonableness
// ===================================================================
describe('TF32 MFU reasonableness', () => {
  it('A100 TF32 MFU ∈ [0.20, 0.40] (half of BF16 due to TF32=50% of BF16 TFLOPS)', () => {
    const tf32 = sim(llama8bConfig('8x-a100', 'tf32'));
    // Actual: 0.30
    expect(tf32.mfu).toBeGreaterThan(0.20);
    expect(tf32.mfu).toBeLessThan(0.40);
  });

  it('A100 BF16 MFU ∈ [0.45, 0.70]', () => {
    const bf16 = sim(llama8bConfig('8x-a100', 'bf16'));
    // Actual: 0.57
    expect(bf16.mfu).toBeGreaterThan(0.45);
    expect(bf16.mfu).toBeLessThan(0.70);
  });

  it('A100 TF32 MFU ≈ 50-60% of BF16 MFU', () => {
    const bf16 = sim(llama8bConfig('8x-a100', 'bf16'));
    const tf32 = sim(llama8bConfig('8x-a100', 'tf32'));
    // Actual: 0.30/0.57 ≈ 0.53
    const mfuRatio = tf32.mfu / bf16.mfu;
    expect(mfuRatio).toBeGreaterThan(0.40);
    expect(mfuRatio).toBeLessThan(0.70);
  });

  it('H100 TF32 MFU ∈ [0.40, 0.65] (same as BF16)', () => {
    const tf32 = sim(llama8bConfig('8x-h100', 'tf32'));
    // Actual: 0.52
    expect(tf32.mfu).toBeGreaterThan(0.40);
    expect(tf32.mfu).toBeLessThan(0.65);
  });

  it('H100 TF32 MFU ≈ BF16 MFU (absolute diff < 0.05)', () => {
    const bf16 = sim(llama8bConfig('8x-h100', 'bf16'));
    const tf32 = sim(llama8bConfig('8x-h100', 'tf32'));
    // Actual: both 0.52
    expect(Math.abs(tf32.mfu - bf16.mfu)).toBeLessThan(0.05);
  });

  it('A100 FP32 MFU ∈ [0.02, 0.08] (FP32 TFLOPS is 6% of BF16)', () => {
    const fp32 = sim(llama8bConfig('8x-a100', 'fp32'));
    // Actual: 0.04
    expect(fp32.mfu).toBeGreaterThan(0.02);
    expect(fp32.mfu).toBeLessThan(0.08);
  });

  it('H100 FP32 MFU ∈ [0.02, 0.08] (FP32 TFLOPS is 7% of BF16)', () => {
    const fp32 = sim(llama8bConfig('8x-h100', 'fp32'));
    // Actual: 0.04
    expect(fp32.mfu).toBeGreaterThan(0.02);
    expect(fp32.mfu).toBeLessThan(0.08);
  });
});

// ===================================================================
// Section 3: Memory footprint comparison
// ===================================================================
describe('Memory footprint comparison', () => {
  it('TF32 and FP32 have identical memory (same 4-byte storage)', () => {
    const tf32 = sim(llama8bConfig('8x-a100', 'tf32'));
    const fp32 = sim(llama8bConfig('8x-a100', 'fp32'));
    // Both use 4-byte params, 4-byte grads, 8-byte optimizer = 16 bytes/param
    // Activations also both 4 bytes/element
    expect(tf32.peakMemoryGB).toBeCloseTo(fp32.peakMemoryGB, 1);
  });

  it('FP32/TF32 peak memory > BF16 peak memory (larger activations)', () => {
    const bf16 = sim(llama8bConfig('8x-a100', 'bf16'));
    const tf32 = sim(llama8bConfig('8x-a100', 'tf32'));
    const fp32 = sim(llama8bConfig('8x-a100', 'fp32'));
    // Actual: BF16=36.63 GB, TF32=FP32=46.10 GB
    expect(tf32.peakMemoryGB).toBeGreaterThan(bf16.peakMemoryGB);
    expect(fp32.peakMemoryGB).toBeGreaterThan(bf16.peakMemoryGB);
  });

  it('FP32/BF16 memory ratio ∈ [1.1, 1.5] (activation memory difference)', () => {
    const bf16 = sim(llama8bConfig('8x-a100', 'bf16'));
    const fp32 = sim(llama8bConfig('8x-a100', 'fp32'));
    // Actual: 46.10/36.63 ≈ 1.26
    const ratio = fp32.peakMemoryGB / bf16.peakMemoryGB;
    expect(ratio).toBeGreaterThan(1.1);
    expect(ratio).toBeLessThan(1.5);
  });

  it('H100: same memory pattern (TF32 = FP32 > BF16)', () => {
    const bf16 = sim(llama8bConfig('8x-h100', 'bf16'));
    const tf32 = sim(llama8bConfig('8x-h100', 'tf32'));
    const fp32 = sim(llama8bConfig('8x-h100', 'fp32'));

    expect(tf32.peakMemoryGB).toBeCloseTo(fp32.peakMemoryGB, 1);
    expect(tf32.peakMemoryGB).toBeGreaterThan(bf16.peakMemoryGB);
  });
});

// ===================================================================
// Section 4: V100 FP32-only training
// ===================================================================
describe('V100 FP32-only training', () => {
  it('V100 FP32 produces valid results', () => {
    const metrics = sim(gpt125mConfig({ clusterConfig: v100Cluster }, 'fp32'));
    expect(metrics.mfu).toBeGreaterThan(0);
    expect(metrics.mfu).toBeLessThanOrEqual(1.0);
    expect(Number.isFinite(metrics.avgStepTimeMs)).toBe(true);
    expect(metrics.avgStepTimeMs).toBeGreaterThan(0);
  });

  it('V100 TF32 falls back to FP32 (tf32TFLOPS=0 → fp32TFLOPS=15.7)', () => {
    expect(V100_32GB.tf32TFLOPS).toBe(0);
    expect(getEffectiveTFLOPS(V100_32GB, 'tf32')).toBe(getEffectiveTFLOPS(V100_32GB, 'fp32'));
  });

  it('V100 TF32 and FP32 produce identical results (same fallback)', () => {
    const fp32 = sim(gpt125mConfig({ clusterConfig: v100Cluster }, 'fp32'));
    const tf32 = sim(gpt125mConfig({ clusterConfig: v100Cluster }, 'tf32'));
    // Both resolve to FP32=15.7 TFLOPS
    expect(tf32.avgStepTimeMs).toBe(fp32.avgStepTimeMs);
    expect(tf32.mfu).toBe(fp32.mfu);
    expect(tf32.peakMemoryGB).toBe(fp32.peakMemoryGB);
  });

  it('V100 FP32 step time > A100 FP32 step time (lower TFLOPS, less memory)', () => {
    const v100 = sim(gpt125mConfig({ clusterConfig: v100Cluster }, 'fp32'));
    const a100 = sim(gpt125mConfig('8x-a100', 'fp32'));
    // V100: 15.7 TFLOPS, A100: 19.5 TFLOPS → A100 is ~1.24x faster
    // Actual: V100=1593ms, A100=1275ms
    expect(v100.avgStepTimeMs).toBeGreaterThan(a100.avgStepTimeMs);
  });

  it('V100 FP32 MFU ∈ [0.04, 0.12]', () => {
    const metrics = sim(gpt125mConfig({ clusterConfig: v100Cluster }, 'fp32'));
    // Actual: 0.08
    expect(metrics.mfu).toBeGreaterThan(0.04);
    expect(metrics.mfu).toBeLessThan(0.12);
  });
});

// ===================================================================
// Section 5: TF32 = BF16 on Hopper (identical TFLOPS)
// ===================================================================
describe('TF32 = BF16 on Hopper', () => {
  it('H100 TF32 and BF16 have identical TFLOPS', () => {
    expect(getEffectiveTFLOPS(H100_SXM, 'tf32')).toBe(989);
    expect(getEffectiveTFLOPS(H100_SXM, 'bf16')).toBe(989);
  });

  it('H100 step time ratio ∈ [0.95, 1.10]', () => {
    const bf16 = sim(llama8bConfig('8x-h100', 'bf16'));
    const tf32 = sim(llama8bConfig('8x-h100', 'tf32'));
    // Actual: 1537/1522 ≈ 1.01
    const ratio = tf32.avgStepTimeMs / bf16.avgStepTimeMs;
    expect(ratio).toBeGreaterThan(0.95);
    expect(ratio).toBeLessThan(1.10);
  });

  it('H100 MFU difference < 0.05 (absolute)', () => {
    const bf16 = sim(llama8bConfig('8x-h100', 'bf16'));
    const tf32 = sim(llama8bConfig('8x-h100', 'tf32'));
    expect(Math.abs(tf32.mfu - bf16.mfu)).toBeLessThan(0.05);
  });

  it('H100 TF32 memory > BF16 memory (4-byte vs 2-byte activations)', () => {
    const bf16 = sim(llama8bConfig('8x-h100', 'bf16'));
    const tf32 = sim(llama8bConfig('8x-h100', 'tf32'));
    // Actual: TF32=46.10 GB, BF16=36.63 GB
    expect(tf32.peakMemoryGB).toBeGreaterThan(bf16.peakMemoryGB);
  });
});

// ===================================================================
// Section 6: Regression — no MFU > 100% or NaN
// ===================================================================
describe('Dense model regression — precision sanity', () => {
  it('GPT-3 175B on 64×A100: all precisions produce finite MFU ∈ (0, 1]', () => {
    for (const precision of ['fp32', 'tf32', 'bf16'] as const) {
      const config: SimulationConfig = {
        modelId: 'gpt3-175b',
        clusterId: '64x-a100',
        globalBatchSize: 64,
        microBatchSize: 1,
        sequenceLength: 2048,
        strategyType: 'fsdp-tp-pp',
        strategyConfig: { tp: 8, pp: 8 },
        activationCheckpointing: true,
        mixedPrecision: precision,
      };
      const result = runSimulation(config);
      expect(result.success, `${precision} should succeed`).toBe(true);
      expect(Number.isFinite(result.metrics.mfu), `${precision} MFU finite`).toBe(true);
      expect(result.metrics.mfu, `${precision} MFU > 0`).toBeGreaterThan(0);
      expect(result.metrics.mfu, `${precision} MFU <= 1`).toBeLessThanOrEqual(1.0);
      expect(Number.isNaN(result.metrics.avgStepTimeMs), `${precision} step time not NaN`).toBe(false);
    }
  });

  it('GPT-3 175B: TF32 MFU ≈ half of BF16 MFU on A100', () => {
    const config = (p: SimulationConfig['mixedPrecision']): SimulationConfig => ({
      modelId: 'gpt3-175b',
      clusterId: '64x-a100',
      globalBatchSize: 64,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 8 },
      activationCheckpointing: true,
      mixedPrecision: p,
    });
    const bf16 = sim(config('bf16'));
    const tf32 = sim(config('tf32'));
    // Actual: BF16=0.54, TF32=0.28 → ratio ≈ 0.52
    const mfuRatio = tf32.mfu / bf16.mfu;
    expect(mfuRatio).toBeGreaterThan(0.35);
    expect(mfuRatio).toBeLessThan(0.70);
  });

  it('Llama 3 8B on 8×A100: HFU/MFU ratio ≈ 4/3 with activation checkpointing', () => {
    // Skip FP32 — its MFU/HFU are so small (0.04/0.05) that 2-decimal rounding
    // distorts the ratio from 1.333 to 1.25
    for (const precision of ['bf16', 'tf32'] as const) {
      const metrics = sim(llama8bConfig('8x-a100', precision));
      // With checkpointing: HFU = 8PD/(time*peak), MFU = 6PD/(time*peak) → ratio = 8/6
      const ratio = metrics.hfu / metrics.mfu;
      expect(ratio, `${precision} HFU/MFU`).toBeCloseTo(8 / 6, 1);
    }
  });
});
