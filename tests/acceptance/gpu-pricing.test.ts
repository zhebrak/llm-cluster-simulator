/**
 * GPU Pricing Coverage Tests
 *
 * Ensures every GPU in the hardware registry has explicit pricing,
 * the simulation config snapshot stores the correct gpuId,
 * and cost computations produce differentiated results across GPUs.
 */

import { describe, it, expect } from 'vitest';
import { ALL_GPUS } from '../../src/core/hardware/gpu.ts';
import {
  GPU_HOURLY_RATES,
  DEFAULT_GPU_HOURLY_RATE,
  getGPUHourlyRate,
} from '../../src/core/cost/cloud.ts';
import { getSimulationMetrics, type SimulationConfig } from '../../src/core/simulation/engine.ts';
import { createMultiNodeCluster } from '../../src/core/hardware/topology.ts';
import {
  CLUSTER_PRESETS,
  getPresetCluster,
} from '../../src/core/hardware/presets.ts';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function simulateGPU(gpuId: string): {
  stepTimeMs: number;
  mfu: number;
  tokensPerSecond: number;
  timeToTrainHours: number;
} {
  const cluster = createMultiNodeCluster(gpuId, 8, 1);
  if (!cluster) throw new Error(`Cannot create cluster for ${gpuId}`);
  const config: SimulationConfig = {
    clusterConfig: cluster,
    modelId: 'gpt3-125m',
    globalBatchSize: 64,
    microBatchSize: 8,
    sequenceLength: 2048,
    strategyType: 'fsdp',
    maxSteps: 1000,
  };
  const metrics = getSimulationMetrics(config);
  return {
    stepTimeMs: metrics.stepTimeMs,
    mfu: metrics.mfu,
    tokensPerSecond: metrics.tokensPerSecond,
    timeToTrainHours: metrics.timeToTrainHours!,
  };
}

function estimatedCost(gpuId: string, numGPUs: number = 8): number {
  const sim = simulateGPU(gpuId);
  const rate = getGPUHourlyRate(gpuId).rate;
  const gpuHours = sim.timeToTrainHours * numGPUs;
  return gpuHours * rate;
}

// ---------------------------------------------------------------------------
// Section 1: Pricing map coverage
// ---------------------------------------------------------------------------

describe('GPU Pricing Coverage', () => {
  it('every GPU in the registry has an explicit hourly rate', () => {
    const missing: string[] = [];
    for (const gpuId of Object.keys(ALL_GPUS)) {
      if (!(gpuId in GPU_HOURLY_RATES)) {
        missing.push(gpuId);
      }
    }
    expect(missing, `GPUs missing from GPU_HOURLY_RATES: ${missing.join(', ')}`).toEqual([]);
  });

  it('GPU_HOURLY_RATES has no stale entries absent from the registry', () => {
    const registryIds = new Set(Object.keys(ALL_GPUS));
    const stale: string[] = [];
    for (const key of Object.keys(GPU_HOURLY_RATES)) {
      if (!registryIds.has(key)) {
        stale.push(key);
      }
    }
    expect(stale, `Stale entries in GPU_HOURLY_RATES: ${stale.join(', ')}`).toEqual([]);
  });

  it('all rates are positive numbers', () => {
    for (const [id, entry] of Object.entries(GPU_HOURLY_RATES)) {
      expect(entry.rate, `${id} rate`).toBeGreaterThan(0);
      expect(entry.name, `${id} name`).toBeTruthy();
    }
  });

  it('getGPUHourlyRate returns specific rate for known GPUs', () => {
    expect(getGPUHourlyRate('h100-sxm').rate).toBe(2.50);
    expect(getGPUHourlyRate('a100-80gb').rate).toBe(1.75);
    expect(getGPUHourlyRate('b200').rate).toBe(5.00);
    expect(getGPUHourlyRate('h200-sxm').rate).toBe(3.50);
    expect(getGPUHourlyRate('mi325x').rate).toBe(2.50);
    expect(getGPUHourlyRate('mi350x').rate).toBe(4.50);
  });

  it('getGPUHourlyRate returns default for unknown GPU IDs', () => {
    expect(getGPUHourlyRate('unknown-gpu')).toEqual(DEFAULT_GPU_HOURLY_RATE);
    expect(getGPUHourlyRate('')).toEqual(DEFAULT_GPU_HOURLY_RATE);
  });

  it('cluster IDs do NOT match any pricing entry', () => {
    const clusterIdPatterns = [
      '128x-a100', '8x-h100', '64x-h200', '256x-b200', '1x-a100',
      '8x-mi325x', '8x-mi350x', '16x-mi300x',
    ];
    for (const clusterId of clusterIdPatterns) {
      expect(
        GPU_HOURLY_RATES[clusterId],
        `cluster ID "${clusterId}" should not be a key in GPU_HOURLY_RATES`
      ).toBeUndefined();
    }
  });

  it('higher-end GPUs have higher rates than lower-end', () => {
    const h100 = getGPUHourlyRate('h100-sxm').rate;
    const h200 = getGPUHourlyRate('h200-sxm').rate;
    const b200 = getGPUHourlyRate('b200').rate;
    const a100 = getGPUHourlyRate('a100-80gb').rate;
    const t4 = getGPUHourlyRate('t4').rate;

    expect(b200).toBeGreaterThan(h200);
    expect(h200).toBeGreaterThan(h100);
    expect(h100).toBeGreaterThan(a100);
    expect(a100).toBeGreaterThan(t4);
  });

  it('MI350X rate is higher than MI325X rate', () => {
    expect(getGPUHourlyRate('mi350x').rate).toBeGreaterThan(getGPUHourlyRate('mi325x').rate);
  });
});

// ---------------------------------------------------------------------------
// Section 2: Cost differentiation across GPUs
// ---------------------------------------------------------------------------

describe('Cost Differentiation', () => {
  // GPUs with enough memory for GPT-3 125M on 8 GPUs with FSDP
  const trainableGPUs = Object.keys(ALL_GPUS).filter(id => {
    const gpu = ALL_GPUS[id];
    return gpu.memoryGB >= 16 && (gpu.bf16TFLOPS > 0 || gpu.fp16TFLOPS > 0);
  });

  it('every trainable GPU produces a unique estimated cost', () => {
    const costs = new Map<string, string>(); // cost_key → gpuId
    const results: { id: string; cost: number; rate: number; hours: number }[] = [];
    const skipped: string[] = [];

    for (const gpuId of trainableGPUs) {
      try {
        const sim = simulateGPU(gpuId);
        const rate = getGPUHourlyRate(gpuId).rate;
        const gpuHours = sim.timeToTrainHours * 8;
        const cost = gpuHours * rate;
        results.push({ id: gpuId, cost, rate, hours: sim.timeToTrainHours });
      } catch {
        skipped.push(gpuId);
      }
    }

    expect(skipped.length, `Too many GPUs skipped: ${skipped.join(', ')}`).toBeLessThan(5);

    for (const { id, cost } of results) {
      // Use 6 decimal places — full float precision, costs must be numerically distinct
      const key = cost.toFixed(6);
      if (costs.has(key)) {
        const other = costs.get(key)!;
        const r1 = results.find(r => r.id === id)!;
        const r2 = results.find(r => r.id === other)!;
        throw new Error(
          `${id} (rate=$${r1.rate}/hr, ${r1.hours.toFixed(4)}h) and ${other} (rate=$${r2.rate}/hr, ${r2.hours.toFixed(4)}h) ` +
          `produce identical cost: $${cost.toFixed(6)}`
        );
      }
      costs.set(key, id);
    }

    expect(results.length).toBeGreaterThan(15);
  });

  it('MI325X and MI350X produce different costs', () => {
    const cost325 = estimatedCost('mi325x');
    const cost350 = estimatedCost('mi350x');
    // Costs are small (< $1) for 1000 steps of GPT-3 125M, so compare with enough precision
    expect(Math.abs(cost325 - cost350)).toBeGreaterThan(0.001);
    // MI350X is faster (lower hours) but more expensive (higher rate)
    const sim325 = simulateGPU('mi325x');
    const sim350 = simulateGPU('mi350x');
    expect(sim350.stepTimeMs).toBeLessThan(sim325.stepTimeMs);
    expect(getGPUHourlyRate('mi350x').rate).toBeGreaterThan(getGPUHourlyRate('mi325x').rate);
  });

  it('H100 and A100 produce different costs', () => {
    const costH100 = estimatedCost('h100-sxm');
    const costA100 = estimatedCost('a100-80gb');
    expect(Math.abs(costH100 - costA100)).toBeGreaterThan(0.001);
  });

  it('H100 and H200 produce different costs', () => {
    const costH100 = estimatedCost('h100-sxm');
    const costH200 = estimatedCost('h200-sxm');
    expect(Math.abs(costH100 - costH200)).toBeGreaterThan(0.001);
  });

  it('B200 has different cost from H200', () => {
    const costB200 = estimatedCost('b200');
    const costH200 = estimatedCost('h200-sxm');
    expect(Math.abs(costB200 - costH200)).toBeGreaterThan(0.001);
  });
});

// ---------------------------------------------------------------------------
// Section 3: Cluster preset validation
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Section 4: Every preset cluster has correct gpuId for pricing lookup
// ---------------------------------------------------------------------------

describe('All cluster presets map to valid GPU pricing', () => {
  it('every cluster preset node.gpu.id exists in GPU_HOURLY_RATES', () => {
    const missing: string[] = [];
    for (const presetId of Object.keys(CLUSTER_PRESETS)) {
      const cluster = getPresetCluster(presetId);
      if (!cluster) {
        missing.push(`${presetId}: failed to create`);
        continue;
      }
      const gpuId = cluster.node.gpu.id;
      if (!(gpuId in GPU_HOURLY_RATES)) {
        missing.push(`${presetId}: gpu.id="${gpuId}" not in GPU_HOURLY_RATES`);
      }
    }
    expect(missing, `Presets with missing pricing:\n${missing.join('\n')}`).toEqual([]);
  });
});
