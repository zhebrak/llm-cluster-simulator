/**
 * H800 / A800 Benchmark Validation Tests
 *
 * Tests for China-export GPU variants (H800 SXM, A800 80GB).
 * These GPUs have identical compute to H100/A100 but reduced NVLink bandwidth.
 *
 * Published reference: DeepSeek V3 on 2048 H800 SXM → 43.7% non-causal MFU
 * (ISCA 2025, arXiv 2505.09343, Table 4, BF16 peak denominator)
 *
 * Our simulator models FP8 TP quantized collectives (Hopper Transformer
 * Engine), giving ~44.9% MFU on H800 (published: 43.7%, +1.2pp delta).
 * EP coordination penalty scales with log2(ep) for cross-node overhead.
 * Tests pin to actual sim output ±15%.
 */

import { describe, it, expect } from 'vitest';
import { getValidatedSimulationMetrics } from '../helpers/validated-metrics.ts';
import { SimulationEngine, type SimulationConfig } from '../../src/core/simulation/engine.ts';
import { createMultiNodeCluster, createSingleNodeCluster } from '../../src/core/hardware/topology.ts';

const TOLERANCE = 0.15; // ±15%

function expectClose(actual: number, expected: number, label: string) {
  const rel = Math.abs(actual - expected) / expected;
  expect(
    rel,
    `${label}: expected ~${expected.toPrecision(4)}, got ${actual.toPrecision(4)} (${(rel * 100).toFixed(2)}% off, limit ${TOLERANCE * 100}%)`
  ).toBeLessThanOrEqual(TOLERANCE);
}

// ─── Test 1: DeepSeek V3 FP8 on 2048 H800 SXM ──────────────────────
// AC: DeepSeek V3 uses a custom selective recomputation (RMSNorm + MLA up-proj +
// SwiGLU output) which differs from Megatron-LM's selective (attention-only recompute).
// Their approach has similar total recompute overhead to our full AC model, so we keep
// full AC here. Sim gives 45.9% vs published 43.7% (+2.2pp analytical ceiling overshoot).

describe('DeepSeek V3 FP8 on 2048 H800 SXM', () => {
  const cluster = createMultiNodeCluster('h800-sxm', 8, 256)!;
  const config: SimulationConfig = {
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
    clusterConfig: cluster,
  };

  it('produces valid metrics (no OOM)', () => {
    const m = getValidatedSimulationMetrics(config);
    expect(m.memoryUtilization, 'Should not OOM').toBeLessThan(1.0);
    expect(m.mfu, 'MFU should be positive').toBeGreaterThan(0);
    expect(m.mfu, 'MFU should not exceed 100%').toBeLessThanOrEqual(1.0);
  });

  it('MFU pinned ±15%', () => {
    const m = getValidatedSimulationMetrics(config);
    // Published 43.7% (ISCA 2025). Sim output ~0.449 (BW penalty at raw ep=32,
    // latency at effectiveEp=4, FP8 quant/dequant overhead on EP dispatch+combine).
    // MoE residual (0.635) captures additional per-layer overhead.
    expectClose(m.mfu, 0.4491, 'MFU');
  });

  it('throughput and step time pinned ±15%', () => {
    const m = getValidatedSimulationMetrics(config);
    expectClose(m.tokensPerSecond, 4036963, 'Tokens/sec');
    expectClose(m.stepTimeMs, 8312, 'Step time');
  });
});

// ─── Test 2: DeepSeek V3 BF16 on 2048 H800 SXM ─────────────────────

describe('DeepSeek V3 BF16 on 2048 H800 SXM', () => {
  it('fits in memory with EP=32 (EP activation memory reduction)', () => {
    // EP=32 distributes expert intermediate activations across 32 EP groups,
    // reducing per-GPU activation memory enough for BF16 to fit (~0.87 memUtil).
    // DualPipeV has 2*pp+1=17 in-flight microbatches — needs MBS=2 to fit in BF16.
    // MBS=4 OOMs (~1.13) due to the high in-flight count.
    const cluster = createMultiNodeCluster('h800-sxm', 8, 256)!;
    const engine = new SimulationEngine();
    engine.configure({
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
      mixedPrecision: 'bf16',
      clusterConfig: cluster,
    });
    const m = engine.simulate();
    // EP=32 activation memory reduction: expert intermediates are distributed
    // across EP groups, so each GPU only stores 1/EP of the MoE activation tensors.
    // This is physically correct — each EP rank only computes its local experts.
    expect(m.memoryUtilization, 'BF16 with EP=32 should fit').toBeLessThan(1.0);
    expect(m.memoryUtilization, 'memUtil should be ~0.87').toBeGreaterThan(0.75);
    expect(m.memoryUtilization).toBeLessThan(0.95);
  });
});

// ─── Test 3: H800 vs H100 NVLink delta ──────────────────────────────

describe('H800 vs H100 NVLink delta (same DeepSeek V3 FP8 config)', () => {
  const makeV3Config = (gpuId: string): SimulationConfig => ({
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
    clusterConfig: createMultiNodeCluster(gpuId, 8, 256)!,
  });

  it('H800 MFU < H100 MFU (reduced NVLink)', () => {
    const h800 = getValidatedSimulationMetrics(makeV3Config('h800-sxm'));
    const h100 = getValidatedSimulationMetrics(makeV3Config('h100-sxm'));
    expect(h800.mfu, 'H800 MFU should be lower than H100').toBeLessThan(h100.mfu);
  });

  it('H800 MFU > H100 MFU × 0.50 (NVLink is not the sole bottleneck)', () => {
    const h800 = getValidatedSimulationMetrics(makeV3Config('h800-sxm'));
    const h100 = getValidatedSimulationMetrics(makeV3Config('h100-sxm'));
    expect(h800.mfu, 'H800 should retain >50% of H100 MFU').toBeGreaterThan(h100.mfu * 0.50);
  });
});

// ─── Test 4: A800 vs A100 single-node FSDP ──────────────────────────

describe('A800 vs A100 single-node FSDP (Mistral 7B)', () => {
  const makeFsdpConfig = (gpuId: string): SimulationConfig => ({
    modelId: 'mistral-7b',
    sequenceLength: 4096,
    strategyType: 'fsdp',
    strategyConfig: {
      tp: 1, pp: 1, dp: 8, ep: 1,
      dpType: 'fsdp',
      sequenceParallel: false,
    },
    globalBatchSize: 1024,
    microBatchSize: 8,
    activationCheckpointing: true,
    flashAttention: true,
    mixedPrecision: 'bf16',
    clusterConfig: createSingleNodeCluster(gpuId, 8)!,
  });

  it('A800 MFU ≤ A100 MFU (reduced NVLink)', () => {
    const a800 = getValidatedSimulationMetrics(makeFsdpConfig('a800-80gb'));
    const a100 = getValidatedSimulationMetrics(makeFsdpConfig('a100-80gb'));
    expect(a800.mfu).toBeLessThanOrEqual(a100.mfu);
  });

  it('MFU delta is modest (<5% absolute — single-node FSDP comm is small)', () => {
    const a800 = getValidatedSimulationMetrics(makeFsdpConfig('a800-80gb'));
    const a100 = getValidatedSimulationMetrics(makeFsdpConfig('a100-80gb'));
    const delta = Math.abs(a100.mfu - a800.mfu);
    expect(delta, 'MFU delta should be < 5 percentage points').toBeLessThan(0.05);
  });
});
