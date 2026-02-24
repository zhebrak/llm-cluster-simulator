/**
 * GPU Differentiation Tests
 *
 * Ensures that every GPU produces unique simulation results for both
 * training and inference. Specifically targets same-TFLOPS GPU pairs
 * where memory bandwidth scaling differentiates otherwise identical
 * peak TFLOPS.
 *
 * Same-TFLOPS pairs:
 * - H100 SXM / H200 SXM (989 bf16 TFLOPS, different memBW)
 * - A100 40GB / A100 80GB (312 bf16 TFLOPS, different memBW)
 * - MI300X / MI325X (1307 bf16 TFLOPS, different memBW)
 * - V100 / A10 (125 bf16/fp16 TFLOPS, different memBW)
 *
 * Note: H100 NVL has 835 bf16 TFLOPS (lower clocks at 400W TDP), not same as SXM.
 * MI210 has 1.6 TB/s memBW (not 1.638 like MI250X).
 */

import { describe, it, expect } from 'vitest';
import {
  ALL_GPUS,
  A100_40GB,
  A100_80GB,
  H100_SXM,
  H100_NVL,
  H200_SXM,
  MI300X,
  MI325X,
  MI250X,
  MI210,
  getEffectiveTFLOPS,
  getMemoryBandwidthScaling,
} from '../../src/core/hardware/gpu.ts';
import { createMultiNodeCluster } from '../../src/core/hardware/topology.ts';
import {
  type SimulationConfig,
} from '../../src/core/simulation/engine.ts';
import { getValidatedSimulationMetrics } from '../helpers/validated-metrics.ts';
import {
  estimateTTFT,
  estimateTPOT,
  calculateLatencyMetrics,
  calculateLatencyWithTP,
} from '../../src/core/inference/latency.ts';
import { runInferenceSimulation } from '../../src/core/inference/simulation.ts';
import { getModel } from '../../src/core/models/index.ts';
import type { GPUSpec } from '../../src/types/index.ts';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function trainingConfig(
  gpuId: string,
  gpusPerNode: number,
  numNodes: number,
  modelId: string,
  strategy: SimulationConfig['strategyType'],
  gbs: number,
  mbs: number,
  seqLen: number,
  strategyConfig?: SimulationConfig['strategyConfig'],
): SimulationConfig {
  return {
    clusterConfig: createMultiNodeCluster(gpuId, gpusPerNode, numNodes)!,
    modelId,
    globalBatchSize: gbs,
    microBatchSize: mbs,
    sequenceLength: seqLen,
    strategyType: strategy,
    strategyConfig,
  };
}

function getTrainingMFU(gpuId: string, strategy: SimulationConfig['strategyType'] = 'fsdp'): number {
  const config = trainingConfig(gpuId, 8, 1, 'llama2-7b', strategy, 16, 2, 4096);
  const metrics = getValidatedSimulationMetrics(config);
  return metrics.mfu;
}

function getTrainingStepTime(gpuId: string, strategy: SimulationConfig['strategyType'] = 'fsdp'): number {
  const config = trainingConfig(gpuId, 8, 1, 'llama2-7b', strategy, 16, 2, 4096);
  const metrics = getValidatedSimulationMetrics(config);
  return metrics.stepTimeMs;
}

const smallModel = getModel('gpt3-125m')!;
const mediumModel = getModel('llama3-8b')!;

// ---------------------------------------------------------------------------
// Section 1: Memory bandwidth scaling produces distinct values per GPU
// ---------------------------------------------------------------------------

describe('Memory bandwidth scaling: distinct per GPU', () => {
  const gpuEntries = Object.entries(ALL_GPUS);

  it('every GPU has a unique memBW scaling factor', () => {
    // China-export variants (H800, A800) have identical compute + memBW as originals.
    // They're differentiated by NVLink bandwidth, not single-GPU compute metrics.
    const exportVariantPairs = new Set(['a800-80gb:a100-80gb', 'h800-sxm:h100-sxm']);
    const seen = new Map<string, string>();
    for (const [id, gpu] of gpuEntries) {
      const tflops = getEffectiveTFLOPS(gpu, 'bf16') || getEffectiveTFLOPS(gpu, 'fp16');
      const scaling = getMemoryBandwidthScaling(gpu, tflops > 0 ? 'bf16' : 'fp16');
      // Key: TFLOPS + scaling (same TFLOPS but different scaling = different results)
      const key = `${tflops.toFixed(2)}_${scaling.toFixed(6)}`;
      if (seen.has(key)) {
        const pair = `${id}:${seen.get(key)}`;
        const reversePair = `${seen.get(key)}:${id}`;
        if (!exportVariantPairs.has(pair) && !exportVariantPairs.has(reversePair)) {
          throw new Error(
            `${id} and ${seen.get(key)} have identical TFLOPS (${tflops}) AND memBW scaling (${scaling.toFixed(6)}) — training results would be identical`
          );
        }
      }
      seen.set(key, id);
    }
  });

  it('same-TFLOPS pairs have different scaling', () => {
    // Enumerate all known same-TFLOPS pairs
    const pairs: [string, GPUSpec, string, GPUSpec][] = [
      ['H100 SXM', H100_SXM, 'H200 SXM', H200_SXM],
      ['A100 40GB', A100_40GB, 'A100 80GB', A100_80GB],
      ['MI300X', MI300X, 'MI325X', MI325X],
    ];

    for (const [name1, gpu1, name2, gpu2] of pairs) {
      const tflops1 = gpu1.bf16TFLOPS || gpu1.fp16TFLOPS;
      const tflops2 = gpu2.bf16TFLOPS || gpu2.fp16TFLOPS;
      expect(tflops1, `${name1} vs ${name2} should have same TFLOPS`).toBe(tflops2);

      const s1 = getMemoryBandwidthScaling(gpu1, 'bf16');
      const s2 = getMemoryBandwidthScaling(gpu2, 'bf16');
      expect(s1, `${name1} vs ${name2} must have different memBW scaling`).not.toBe(s2);
    }
  });

  it('H100 NVL has different TFLOPS than H100 SXM (835 vs 989)', () => {
    expect(H100_NVL.bf16TFLOPS).toBe(835);
    expect(H100_SXM.bf16TFLOPS).toBe(989);
    expect(H100_NVL.bf16TFLOPS).toBeLessThan(H100_SXM.bf16TFLOPS);
  });

  it('H200 SXM has higher scaling than H100 SXM (more memBW)', () => {
    const h200 = getMemoryBandwidthScaling(H200_SXM, 'bf16');
    const h100 = getMemoryBandwidthScaling(H100_SXM, 'bf16');
    expect(h200).toBeGreaterThan(h100);
    // H200 should be ~5% faster
    expect(h200 / h100).toBeGreaterThan(1.03);
    expect(h200 / h100).toBeLessThan(1.10);
  });

  it('MI325X has higher scaling than MI300X (more memBW)', () => {
    const mi325 = getMemoryBandwidthScaling(MI325X, 'bf16');
    const mi300 = getMemoryBandwidthScaling(MI300X, 'bf16');
    expect(mi325).toBeGreaterThan(mi300);
  });

  it('H100 SXM is the reference point (scaling = 1.0)', () => {
    const scaling = getMemoryBandwidthScaling(H100_SXM, 'bf16');
    expect(scaling).toBeCloseTo(1.0, 4);
  });
});

// ---------------------------------------------------------------------------
// Section 2: Training — same-TFLOPS pairs produce different MFU
// ---------------------------------------------------------------------------

describe('Training: same-TFLOPS GPU pairs produce different results', () => {
  it('H100 SXM vs H200 SXM: H200 has higher MFU', () => {
    const mfuH100 = getTrainingMFU('h100-sxm');
    const mfuH200 = getTrainingMFU('h200-sxm');
    expect(mfuH200).toBeGreaterThan(mfuH100);
    // Should be 3-8% higher (memBW scaling)
    expect(mfuH200 / mfuH100).toBeGreaterThan(1.03);
    expect(mfuH200 / mfuH100).toBeLessThan(1.10);
  });

  it('H100 SXM vs H100 NVL: different TFLOPS produce different MFU', () => {
    const mfuSXM = getTrainingMFU('h100-sxm');
    const mfuNVL = getTrainingMFU('h100-nvl');
    // NVL has 835 bf16 (vs 989 SXM) but higher memBW (3.9 vs 3.35)
    expect(mfuNVL).not.toBe(mfuSXM);
    expect(mfuNVL).toBeGreaterThan(0);
    expect(mfuSXM).toBeGreaterThan(0);
  });

  it('H200 SXM has lower step time than H100 SXM (same TFLOPS, more memBW)', () => {
    const stepH100 = getTrainingStepTime('h100-sxm');
    const stepH200 = getTrainingStepTime('h200-sxm');
    expect(stepH200).toBeLessThan(stepH100);
  });

  it('MI300X vs MI325X: MI325X has higher MFU', () => {
    const mfuMI300 = getTrainingMFU('mi300x');
    const mfuMI325 = getTrainingMFU('mi325x');
    expect(mfuMI325).toBeGreaterThan(mfuMI300);
    // MI325X has 6.0 vs 5.3 TB/s — ~1-2% difference
    expect(mfuMI325 / mfuMI300).toBeGreaterThan(1.005);
  });

  it('A100 40GB vs A100 80GB: 80GB has higher MFU (more memBW)', () => {
    const mfu40 = getTrainingMFU('a100-40gb');
    const mfu80 = getTrainingMFU('a100-80gb');
    expect(mfu80).toBeGreaterThan(mfu40);
  });

  it('Hopper variants produce distinct step times', () => {
    // H100 SXM (989 bf16, 3.35 TB/s), H100 NVL (835 bf16, 3.9 TB/s), H200 SXM (989 bf16, 4.8 TB/s)
    const stepH100 = getTrainingStepTime('h100-sxm');
    const stepNVL = getTrainingStepTime('h100-nvl');
    const stepH200 = getTrainingStepTime('h200-sxm');
    const steps = [stepH100, stepNVL, stepH200];
    // All must be distinct
    expect(new Set(steps).size).toBe(3);
    // H200 should be fastest (highest TFLOPS + most memBW)
    expect(stepH200).toBeLessThan(stepH100);
  });
});

// ---------------------------------------------------------------------------
// Section 3: Training — every GPU produces unique results
// ---------------------------------------------------------------------------

describe('Training: every GPU produces unique step time', () => {
  // Only test GPUs that can fit LLaMA 7B in bf16 on 8 GPUs with FSDP
  const fsdpCapableGPUs = Object.entries(ALL_GPUS).filter(([_, gpu]) => {
    // Need enough memory for FSDP (sharded params + activations)
    // ~7B * 18 bytes / 8 GPUs = ~16 GB min, plus activations
    return gpu.memoryGB >= 24 && (gpu.bf16TFLOPS > 0 || gpu.fp16TFLOPS > 0);
  });

  it(`${fsdpCapableGPUs.length} GPUs produce ${fsdpCapableGPUs.length} distinct step times`, () => {
    const stepTimes: { id: string; stepTime: number }[] = [];

    for (const [gpuId] of fsdpCapableGPUs) {
      try {
        const config = trainingConfig(gpuId, 8, 1, 'llama2-7b', 'fsdp', 16, 2, 4096);
        const metrics = getValidatedSimulationMetrics(config);
        stepTimes.push({ id: gpuId, stepTime: metrics.stepTimeMs });
      } catch {
        // Some GPUs may not support bf16 training
      }
    }

    // Check for duplicates
    const seen = new Map<string, string>();
    for (const { id, stepTime } of stepTimes) {
      const key = stepTime.toFixed(2);
      if (seen.has(key)) {
        throw new Error(
          `${id} and ${seen.get(key)} produce identical step time: ${stepTime.toFixed(2)}ms`
        );
      }
      seen.set(key, id);
    }

    expect(stepTimes.length).toBeGreaterThan(10);
  });
});

// ---------------------------------------------------------------------------
// Section 4: Inference — same-TFLOPS pairs produce different TPOT
// ---------------------------------------------------------------------------

describe('Inference: same-TFLOPS GPU pairs produce different TPOT', () => {
  it('H100 SXM vs H200 SXM: H200 has lower TPOT (more memBW)', () => {
    const tpotH100 = estimateTPOT(mediumModel, 512, 1, H100_SXM, 'bf16');
    const tpotH200 = estimateTPOT(mediumModel, 512, 1, H200_SXM, 'bf16');
    expect(tpotH200).toBeLessThan(tpotH100);
    // H200 has 4.8/3.35 = 1.43x more bandwidth → ~30% lower TPOT
    expect(tpotH100 / tpotH200).toBeGreaterThan(1.2);
    expect(tpotH100 / tpotH200).toBeLessThan(1.5);
  });

  it('H100 SXM vs H100 NVL: NVL has lower TPOT', () => {
    const tpotSXM = estimateTPOT(mediumModel, 512, 1, H100_SXM, 'bf16');
    const tpotNVL = estimateTPOT(mediumModel, 512, 1, H100_NVL, 'bf16');
    expect(tpotNVL).toBeLessThan(tpotSXM);
  });

  it('MI300X vs MI325X: MI325X has lower TPOT', () => {
    const tpotMI300 = estimateTPOT(mediumModel, 512, 1, MI300X, 'bf16');
    const tpotMI325 = estimateTPOT(mediumModel, 512, 1, MI325X, 'bf16');
    expect(tpotMI325).toBeLessThan(tpotMI300);
    // MI325X has 6.0/5.3 = 1.13x more bandwidth → ~10% lower TPOT
    expect(tpotMI300 / tpotMI325).toBeGreaterThan(1.05);
    expect(tpotMI300 / tpotMI325).toBeLessThan(1.20);
  });

  it('A100 40GB vs A100 80GB: 80GB has lower TPOT (more memBW)', () => {
    const tpot40 = estimateTPOT(mediumModel, 512, 1, A100_40GB, 'bf16');
    const tpot80 = estimateTPOT(mediumModel, 512, 1, A100_80GB, 'bf16');
    expect(tpot80).toBeLessThan(tpot40);
  });
});

// ---------------------------------------------------------------------------
// Section 5: Inference full simulation — same-TFLOPS pairs produce different throughput
// ---------------------------------------------------------------------------

describe('Inference simulation: same-TFLOPS pairs produce different throughput', () => {
  it('MI300X vs MI325X: MI325X has higher throughput', () => {
    const mi300 = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: MI300X,
      numGPUs: 1,
      batchSize: 32,
      inputSeqLen: 512,
      outputSeqLen: 256,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
    });
    const mi325 = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: MI325X,
      numGPUs: 1,
      batchSize: 32,
      inputSeqLen: 512,
      outputSeqLen: 256,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
    });
    expect(mi300.success).toBe(true);
    expect(mi325.success).toBe(true);
    expect(mi325.throughput.tokensPerSecond).toBeGreaterThan(mi300.throughput.tokensPerSecond);
    expect(mi325.latency.tpot).toBeLessThan(mi300.latency.tpot);
  });

  it('H100 SXM vs H200 SXM: H200 has higher throughput', () => {
    const h100 = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 32,
      inputSeqLen: 512,
      outputSeqLen: 256,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
    });
    const h200 = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H200_SXM,
      numGPUs: 1,
      batchSize: 32,
      inputSeqLen: 512,
      outputSeqLen: 256,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
    });
    expect(h100.success).toBe(true);
    expect(h200.success).toBe(true);
    expect(h200.throughput.tokensPerSecond).toBeGreaterThan(h100.throughput.tokensPerSecond);
    expect(h200.latency.tpot).toBeLessThan(h100.latency.tpot);
  });

  it('Hopper trio: all produce valid distinct throughput', () => {
    const configs = [
      { name: 'H100 SXM', gpu: H100_SXM },
      { name: 'H100 NVL', gpu: H100_NVL },
      { name: 'H200 SXM', gpu: H200_SXM },
    ];
    const results = configs.map(({ name, gpu }) => ({
      name,
      result: runInferenceSimulation({
        modelId: 'llama3-8b',
        gpu,
        numGPUs: 1,
        batchSize: 32,
        inputSeqLen: 512,
        outputSeqLen: 256,
        weightPrecision: 'bf16',
        kvCachePrecision: 'bf16',
      }),
    }));

    for (const { name, result } of results) {
      expect(result.success, `${name} should succeed`).toBe(true);
    }

    const throughputs = results.map(r => r.result.throughput.tokensPerSecond);
    // All must be distinct
    expect(new Set(throughputs.map(t => t.toFixed(2))).size).toBe(3);
    // H200 has most memBW → highest throughput (inference is memory-bound)
    const [h100sxm, _h100nvl, h200] = throughputs;
    expect(h200, 'H200 > H100 SXM throughput').toBeGreaterThan(h100sxm);
  });
});

// ---------------------------------------------------------------------------
// Section 6: Inference — every GPU produces unique TPOT
// ---------------------------------------------------------------------------

describe('Inference: every GPU produces unique total latency', () => {
  // Use GPT-3 125M which fits on all GPUs
  // TPOT depends only on memory bandwidth (decode is memory-bound), so GPUs
  // with same memBW (MI210/MI250X) correctly have the same TPOT.
  // But total latency = TTFT + decode, and TTFT depends on TFLOPS (compute-bound).
  it('all GPUs produce distinct total latency for same model', () => {
    // China-export variants (H800, A800) have identical compute + memBW as originals,
    // so single-GPU inference latency is identical. They differ only in NVLink bandwidth.
    const exportVariantPairs = new Set(['a800-80gb:a100-80gb', 'h800-sxm:h100-sxm']);
    const seen = new Map<string, string>();
    const results: { id: string; totalLatency: number }[] = [];

    for (const [id, gpu] of Object.entries(ALL_GPUS)) {
      const precision = (gpu.bf16TFLOPS > 0 ? 'bf16' : 'fp16') as 'bf16' | 'fp16';
      const latency = calculateLatencyMetrics(smallModel, 512, 128, 1, gpu, precision);
      results.push({ id, totalLatency: latency.totalLatency });
    }

    for (const { id, totalLatency } of results) {
      const key = totalLatency.toFixed(4);
      if (seen.has(key)) {
        const pair = `${id}:${seen.get(key)}`;
        const reversePair = `${seen.get(key)}:${id}`;
        if (!exportVariantPairs.has(pair) && !exportVariantPairs.has(reversePair)) {
          throw new Error(
            `${id} and ${seen.get(key)} produce identical total latency: ${totalLatency.toFixed(4)}ms`
          );
        }
      }
      seen.set(key, id);
    }

    expect(results.length).toBe(Object.keys(ALL_GPUS).length);
  });

  it('MI210 vs MI250X: MI250X has higher memBW and TFLOPS', () => {
    // MI250X: 1.638 TB/s, 191.5 TFLOPS. MI210: 1.6 TB/s, 181 TFLOPS.
    const precision = 'fp16' as const;
    const tpotMI250 = estimateTPOT(smallModel, 512, 1, MI250X, precision);
    const tpotMI210 = estimateTPOT(smallModel, 512, 1, MI210, precision);
    // MI250X has slightly more memBW (1.638 vs 1.6) → lower TPOT
    expect(tpotMI250).toBeLessThan(tpotMI210);

    // TTFT also differs (different TFLOPS: 191.5 vs 181)
    const ttftMI250 = estimateTTFT(smallModel, 512, MI250X, precision);
    const ttftMI210 = estimateTTFT(smallModel, 512, MI210, precision);
    expect(ttftMI250).not.toBe(ttftMI210);
    expect(ttftMI250).toBeLessThan(ttftMI210); // MI250X is faster (more TFLOPS)
  });
});

// ---------------------------------------------------------------------------
// Section 7: Inference with TP — same-TFLOPS pairs still differentiated
// ---------------------------------------------------------------------------

describe('Inference with TP: same-TFLOPS pairs still differentiated', () => {
  it('H100 SXM vs H200 SXM with TP=2: H200 has lower TPOT', () => {
    const h100 = calculateLatencyWithTP(mediumModel, 512, 256, 8, H100_SXM, 2, 'bf16');
    const h200 = calculateLatencyWithTP(mediumModel, 512, 256, 8, H200_SXM, 2, 'bf16');
    expect(h200.tpot).toBeLessThan(h100.tpot);
    expect(h200.totalLatency).toBeLessThan(h100.totalLatency);
  });

  it('MI300X vs MI325X with TP=2: MI325X has lower TPOT', () => {
    const mi300 = calculateLatencyWithTP(mediumModel, 512, 256, 8, MI300X, 2, 'bf16');
    const mi325 = calculateLatencyWithTP(mediumModel, 512, 256, 8, MI325X, 2, 'bf16');
    expect(mi325.tpot).toBeLessThan(mi300.tpot);
  });
});

// ---------------------------------------------------------------------------
// Section 8: Memory bandwidth scaling sanity
// ---------------------------------------------------------------------------

describe('Memory bandwidth scaling: sanity checks', () => {
  it('scaling is always positive and bounded', () => {
    for (const [id, gpu] of Object.entries(ALL_GPUS)) {
      const dtype = gpu.bf16TFLOPS > 0 ? 'bf16' : 'fp16';
      const scaling = getMemoryBandwidthScaling(gpu, dtype);
      expect(scaling, `${id} scaling should be > 0.8`).toBeGreaterThan(0.8);
      expect(scaling, `${id} scaling should be < 1.2`).toBeLessThan(1.2);
    }
  });

  it('higher memBW relative to TFLOPS → higher scaling', () => {
    // Sort GPUs by operational intensity (TFLOPS/memBW)
    // Lower OI = more balanced = higher scaling
    const gpuList = Object.entries(ALL_GPUS)
      .filter(([_, g]) => g.bf16TFLOPS > 0)
      .map(([id, gpu]) => ({
        id,
        oi: gpu.bf16TFLOPS / gpu.memoryBandwidthTBps,
        scaling: getMemoryBandwidthScaling(gpu, 'bf16'),
      }))
      .sort((a, b) => a.oi - b.oi);

    // Generally: lower OI → higher scaling (correlation should be negative)
    // Check first vs last
    expect(gpuList[0].scaling).toBeGreaterThan(gpuList[gpuList.length - 1].scaling);
  });
});
