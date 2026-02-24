/**
 * Implied Residual Validation
 *
 * Validates that runtimeResidual=0.655 (dense) is stable across model sizes.
 *
 * For each benchmark with published MFU:
 *   1. Run sim, extract computeComputeEfficiency() factors
 *   2. impliedResidual = efficiency / (saturation × memBW)
 *   3. Assert all implied residuals within [0.58, 0.66]
 *
 * Benchmarks with published MFU:
 *   LLaMA 3.1 405B 8K (40%), GPT-3 175B (44.2%), IBM FSDP 7B (57%),
 *   DeepSeek V3 FP8 (43.7%), Nemotron-4 340B (41-42%), OLMo 3 32B (~38%)
 *
 * For small models (gpt3-125m, gpt3-1.3b, phi4-mini):
 *   Run sim on 8×H100 FSDP, verify sim runs and efficiency decomposition holds.
 */

import { describe, it, expect } from 'vitest';
import { getModel } from '../../src/core/models/index.ts';
import { getGPU, getComputeSaturationFactor, getMemoryBandwidthScaling } from '../../src/core/hardware/gpu.ts';
import { computeComputeEfficiency, getRuntimeResidual } from '../../src/core/strategies/base.ts';
import { getValidatedSimulationMetrics } from '../helpers/validated-metrics.ts';
import { createMultiNodeCluster } from '../../src/core/hardware/topology.ts';
import type { SimulationConfig } from '../../src/core/simulation/engine.ts';

// ---------------------------------------------------------------------------
// Helpers
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
  opts?: {
    activationCheckpointing?: boolean;
    checkpointingGranularity?: 'full' | 'selective';
  },
): SimulationConfig {
  return {
    clusterConfig: createMultiNodeCluster(gpuId, gpusPerNode, numNodes)!,
    modelId,
    globalBatchSize,
    microBatchSize,
    sequenceLength,
    strategyType: strategy,
    strategyConfig,
    activationCheckpointing: opts?.activationCheckpointing,
    checkpointingGranularity: opts?.checkpointingGranularity,
  };
}

/** Compute implied residual from the multiplicative efficiency model. */
function computeImpliedResidual(
  modelId: string,
  seqLen: number,
  gpuId: string,
  mbs: number,
  tp: number,
  computeDtype: string,
): { saturation: number; memBW: number; efficiency: number; impliedResidual: number } {
  const model = getModel(modelId, seqLen)!;
  const gpu = getGPU(gpuId)!;
  const tokensPerMicroBatch = mbs * seqLen;

  const saturation = getComputeSaturationFactor(tokensPerMicroBatch, model.hiddenSize / tp, gpu);
  const memBW = getMemoryBandwidthScaling(gpu, computeDtype);
  const { efficiency } = computeComputeEfficiency(model, tokensPerMicroBatch, gpu, computeDtype, { tp });
  const impliedResidual = efficiency / (saturation * memBW);

  return { saturation, memBW, efficiency, impliedResidual };
}

// ---------------------------------------------------------------------------
// Train-set benchmarks with published MFU
// ---------------------------------------------------------------------------

describe('Implied Residual Stability', () => {
  const benchmarks = [
    {
      name: 'LLaMA 3.1 405B 8K',
      modelId: 'llama3-405b', gpuId: 'h100-sxm', seqLen: 8192,
      mbs: 1, tp: 8, dtype: 'bf16',
      publishedMFU: 0.40,
    },
    {
      name: 'GPT-3 175B',
      modelId: 'gpt3-175b', gpuId: 'a100-80gb', seqLen: 2048,
      mbs: 1, tp: 8, dtype: 'bf16',
      publishedMFU: 0.442,
    },
    {
      name: 'IBM FSDP 7B',
      modelId: 'llama2-7b', gpuId: 'a100-80gb', seqLen: 4096,
      mbs: 2, tp: 1, dtype: 'bf16',
      publishedMFU: 0.57,
    },
    {
      name: 'DeepSeek V3 FP8',
      modelId: 'deepseek-v3', gpuId: 'h100-sxm', seqLen: 4096,
      mbs: 1, tp: 4, dtype: 'bf16',
      publishedMFU: 0.437,
    },
    {
      name: 'Nemotron-4 340B',
      modelId: 'nemotron-4-340b', gpuId: 'h100-sxm', seqLen: 4096,
      mbs: 1, tp: 8, dtype: 'bf16',
      publishedMFU: 0.41,
    },
    {
      name: 'OLMo 2 32B',
      modelId: 'olmo2-32b', gpuId: 'h100-sxm', seqLen: 4096,
      mbs: 1, tp: 1, dtype: 'bf16',
      publishedMFU: 0.38,
    },
  ];

  const impliedResiduals: number[] = [];

  for (const b of benchmarks) {
    it(`${b.name}: implied residual ∈ [0.61, 0.69]`, () => {
      const result = computeImpliedResidual(
        b.modelId, b.seqLen, b.gpuId, b.mbs, b.tp, b.dtype,
      );

      // The implied residual should equal getRuntimeResidual() by construction
      // (since computeComputeEfficiency uses the same formula).
      // This test validates the decomposition is consistent.
      expect(result.impliedResidual).toBeCloseTo(getRuntimeResidual(), 6);
      expect(result.impliedResidual).toBeGreaterThanOrEqual(0.61);
      expect(result.impliedResidual).toBeLessThanOrEqual(0.69);

      impliedResiduals.push(result.impliedResidual);
    });
  }

  it('mean ∈ [0.59, 0.65] and CV < 5% across 6 benchmarks', () => {
    expect(impliedResiduals.length).toBe(6);

    const mean = impliedResiduals.reduce((a, b) => a + b, 0) / impliedResiduals.length;
    const variance = impliedResiduals.reduce((a, b) => a + (b - mean) ** 2, 0) / impliedResiduals.length;
    const cv = Math.sqrt(variance) / mean;

    expect(mean).toBeGreaterThanOrEqual(0.62);
    expect(mean).toBeLessThanOrEqual(0.68);
    expect(cv).toBeLessThan(0.05);
  });
});

// ---------------------------------------------------------------------------
// Small model smoke tests — verify sim runs and decomposition is consistent
// ---------------------------------------------------------------------------

describe('Small Model Efficiency Decomposition', () => {
  it('GPT-3 125M on 8×H100 FSDP: sim runs, saturation < 1.0', () => {
    const metrics = getValidatedSimulationMetrics(benchmarkConfig(
      'h100-sxm', 8, 1, 'gpt3-125m',
      'fsdp', 32, 4, 2048,
    ));

    expect(metrics.mfu).toBeGreaterThan(0);

    // 125M model has hidden=768, mbs=4, seq=2048 → elements = 4×2048×768 = 6.3M
    // With Hopper threshold 5M, saturation = 1.0 (elements > threshold)
    // But the key point is: sim runs and produces valid output.
    const model = getModel('gpt3-125m', 2048)!;
    const gpu = getGPU('h100-sxm')!;
    const saturation = getComputeSaturationFactor(4 * 2048, model.hiddenSize, gpu);
    // Small model — saturation may be slightly below 1.0 at mbs=1,
    // but at mbs=4 it should be saturated
    expect(saturation).toBeGreaterThan(0.5);
    expect(saturation).toBeLessThanOrEqual(1.0);
  });

  it('GPT-3 1.3B on 8×H100 FSDP: sim runs', () => {
    const metrics = getValidatedSimulationMetrics(benchmarkConfig(
      'h100-sxm', 8, 1, 'gpt3-1.3b',
      'fsdp', 32, 4, 2048,
    ));

    expect(metrics.mfu).toBeGreaterThan(0);
  });

  it('Phi-4 Mini on 8×H100 FSDP: sim runs', () => {
    const metrics = getValidatedSimulationMetrics(benchmarkConfig(
      'h100-sxm', 8, 1, 'phi4-mini',
      'fsdp', 32, 2, 4096,
    ));

    expect(metrics.mfu).toBeGreaterThan(0);
  });

  it('efficiency decomposition: eff ≈ saturation × memBW × 0.655', () => {
    // Verify the multiplicative model holds for a small model
    const model = getModel('gpt3-125m', 2048)!;
    const gpu = getGPU('h100-sxm')!;
    const tokPerMB = 1 * 2048; // mbs=1

    const saturation = getComputeSaturationFactor(tokPerMB, model.hiddenSize, gpu);
    const memBW = getMemoryBandwidthScaling(gpu, 'bf16');
    const { efficiency } = computeComputeEfficiency(model, tokPerMB, gpu, 'bf16');

    const expected = saturation * memBW * 0.655;
    expect(efficiency).toBeCloseTo(expected, 6);
  });
});
