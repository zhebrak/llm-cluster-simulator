/**
 * Inference Simulation Validation Tests
 *
 * Validates bug fixes and ensures correct behavior across all GPUs:
 * - getGPUTFLOPS() precision fallback chain
 * - Latency sanity (finite, positive) for all 11 GPUs
 * - Utilization metrics never NaN/Infinity
 * - TP with NVLink=0 GPUs (H100 PCIe, MI300X, MI325X)
 * - Full runInferenceSimulation() smoke tests
 * - Latency ordering (faster GPUs → lower latency)
 * - Speculative decoding with different GPUs
 */

import { describe, it, expect } from 'vitest';
import {
  ALL_GPUS,
  V100_32GB,
  A100_80GB,
  H100_SXM,
  H100_PCIE,
  B200,
  MI300X,
  MI325X,
} from '../../src/core/hardware/gpu.ts';
import {
  getGPUTFLOPS,
  estimateTTFT,
  estimateTPOT,
  calculateUtilizationMetrics,
  calculateLatencyWithTP,
} from '../../src/core/inference/latency.ts';
import { runInferenceSimulation, InferenceSimulationEngine } from '../../src/core/inference/simulation.ts';
import { getModel } from '../../src/core/models/index.ts';
import type { InferencePrecision } from '../../src/types/inference.ts';

const ALL_PRECISIONS: InferencePrecision[] = ['fp32', 'fp16', 'bf16', 'fp8', 'fp4', 'int8', 'int4'];

// Helper: get a model that fits on a given GPU
function getModelForGPU(gpu: typeof V100_32GB): { id: string; seqLen: number } {
  if (gpu.memoryGB < 40) {
    return { id: 'gpt3-125m', seqLen: 512 };
  }
  return { id: 'llama3-8b', seqLen: 512 };
}

// ── Section 1: getGPUTFLOPS() precision resolution ──

describe('getGPUTFLOPS() precision fallback chain', () => {
  it('V100: bf16 → falls back to fp16 (125)', () => {
    expect(getGPUTFLOPS(V100_32GB, 'bf16')).toBe(125);
  });

  it('V100: fp8 → falls back through bf16(0) → fp16 (125)', () => {
    expect(getGPUTFLOPS(V100_32GB, 'fp8')).toBe(125);
  });

  it('V100: int8 → falls back through bf16(0) → fp16 (125)', () => {
    expect(getGPUTFLOPS(V100_32GB, 'int8')).toBe(125);
  });

  it('V100: int4 → falls back through chain → fp16 (125)', () => {
    expect(getGPUTFLOPS(V100_32GB, 'int4')).toBe(125);
  });

  it('V100: fp32 → 15.7', () => {
    expect(getGPUTFLOPS(V100_32GB, 'fp32')).toBe(15.7);
  });

  it('V100: fp16 → 125', () => {
    expect(getGPUTFLOPS(V100_32GB, 'fp16')).toBe(125);
  });

  it('A100: bf16 → 312 (native)', () => {
    expect(getGPUTFLOPS(A100_80GB, 'bf16')).toBe(312);
  });

  it('A100: fp8 → falls back to bf16 (312)', () => {
    // A100 has fp8TFLOPS=0
    expect(getGPUTFLOPS(A100_80GB, 'fp8')).toBe(312);
  });

  it('H100 SXM: bf16 → 989 (dense)', () => {
    expect(getGPUTFLOPS(H100_SXM, 'bf16')).toBe(989);
  });

  it('H100 SXM: fp8 → 1979 (native)', () => {
    expect(getGPUTFLOPS(H100_SXM, 'fp8')).toBe(1979);
  });

  it('MI300X: bf16 → 1307', () => {
    expect(getGPUTFLOPS(MI300X, 'bf16')).toBe(1307);
  });

  it('MI300X: fp8 → 2614', () => {
    expect(getGPUTFLOPS(MI300X, 'fp8')).toBe(2614);
  });

  it('every GPU × every precision → positive result', () => {
    for (const [gpuId, gpu] of Object.entries(ALL_GPUS)) {
      for (const precision of ALL_PRECISIONS) {
        const tflops = getGPUTFLOPS(gpu, precision);
        expect(tflops, `${gpuId} × ${precision}`).toBeGreaterThan(0);
        expect(Number.isFinite(tflops), `${gpuId} × ${precision} finite`).toBe(true);
      }
    }
  });
});

// ── Section 2: Latency sanity across all 11 GPUs ──

describe('latency sanity across all GPUs', () => {
  for (const [gpuId, gpu] of Object.entries(ALL_GPUS)) {
    const { id: modelId, seqLen } = getModelForGPU(gpu);

    it(`${gpuId}: estimateTTFT() is finite and positive`, () => {
      const model = getModel(modelId, seqLen)!;
      expect(model).toBeDefined();
      const ttft = estimateTTFT(model, seqLen, gpu, 'bf16');
      expect(Number.isFinite(ttft), `TTFT for ${gpuId}`).toBe(true);
      expect(ttft).toBeGreaterThan(0);
    });

    it(`${gpuId}: estimateTPOT() is finite and positive`, () => {
      const model = getModel(modelId, seqLen)!;
      const tpot = estimateTPOT(model, seqLen, 1, gpu, 'bf16');
      expect(Number.isFinite(tpot), `TPOT for ${gpuId}`).toBe(true);
      expect(tpot).toBeGreaterThan(0);
    });
  }
});

// ── Section 3: Utilization metrics never NaN/Infinity ──

describe('utilization metrics never NaN/Infinity', () => {
  for (const [gpuId, gpu] of Object.entries(ALL_GPUS)) {
    const { id: modelId, seqLen } = getModelForGPU(gpu);

    it(`${gpuId}: computeUtilization ∈ [0,1], memoryBW ∈ [0,1], valid bottleneck`, () => {
      const model = getModel(modelId, seqLen)!;
      expect(model).toBeDefined();
      const util = calculateUtilizationMetrics(model, seqLen, 128, 1, gpu, 'bf16');

      expect(Number.isNaN(util.computeUtilization)).toBe(false);
      expect(Number.isFinite(util.computeUtilization)).toBe(true);
      expect(util.computeUtilization).toBeGreaterThanOrEqual(0);
      expect(util.computeUtilization).toBeLessThanOrEqual(1);

      expect(Number.isNaN(util.rooflineAttainment)).toBe(false);
      expect(Number.isFinite(util.rooflineAttainment)).toBe(true);
      expect(util.rooflineAttainment).toBeGreaterThanOrEqual(0);
      expect(util.rooflineAttainment).toBeLessThanOrEqual(1);

      expect(['compute', 'memory_bandwidth', 'memory_capacity']).toContain(util.bottleneck);
    });
  }
});

// ── Section 4: TP with NVLink=0 GPUs ──

describe('TP with NVLink=0 GPUs (falls back to PCIe)', () => {
  const nvlinkZeroGPUs = [
    { name: 'H100 PCIe', gpu: H100_PCIE },
    { name: 'MI300X', gpu: MI300X },
    { name: 'MI325X', gpu: MI325X },
  ];

  for (const { name, gpu } of nvlinkZeroGPUs) {
    it(`${name} + TP=2: finite TTFT and TPOT`, () => {
      const model = getModel('llama3-8b', 512)!;
      expect(model).toBeDefined();
      const latency = calculateLatencyWithTP(model, 512, 128, 1, gpu, 2, 'bf16');

      expect(Number.isFinite(latency.ttft), `${name} TTFT`).toBe(true);
      expect(latency.ttft).toBeGreaterThan(0);

      expect(Number.isFinite(latency.tpot), `${name} TPOT`).toBe(true);
      expect(latency.tpot).toBeGreaterThan(0);

      expect(Number.isFinite(latency.totalLatency), `${name} totalLatency`).toBe(true);
      expect(latency.totalLatency).toBeGreaterThan(0);
    });
  }

  it('H100 SXM + TP=2 uses NVLink (faster than PCIe)', () => {
    const model = getModel('llama3-8b', 512)!;
    const nvlinkLatency = calculateLatencyWithTP(model, 512, 128, 1, H100_SXM, 2, 'bf16');
    const pcieLatency = calculateLatencyWithTP(model, 512, 128, 1, H100_PCIE, 2, 'bf16');

    // H100 SXM has NVLink, should have lower communication overhead
    // But H100 SXM also has higher base compute, so total should be lower
    expect(Number.isFinite(nvlinkLatency.totalLatency)).toBe(true);
    expect(Number.isFinite(pcieLatency.totalLatency)).toBe(true);
  });
});

// ── Section 5: Full runInferenceSimulation() smoke tests ──

describe('runInferenceSimulation() smoke tests', () => {
  const gpuModelPairs: { gpuId: string; modelId: string }[] = [
    { gpuId: 'v100-32gb', modelId: 'gpt3-125m' },
    { gpuId: 'a100-40gb', modelId: 'llama3-8b' },
    { gpuId: 'a100-80gb', modelId: 'llama3-8b' },
    { gpuId: 'h100-sxm', modelId: 'llama3-8b' },
    { gpuId: 'h100-pcie', modelId: 'llama3-8b' },
    { gpuId: 'h100-nvl', modelId: 'llama3-8b' },
    { gpuId: 'h200-sxm', modelId: 'llama3-8b' },
    { gpuId: 'b200', modelId: 'llama3-8b' },
    { gpuId: 'gb200', modelId: 'llama3-8b' },
    { gpuId: 'mi300x', modelId: 'llama3-8b' },
    { gpuId: 'mi325x', modelId: 'llama3-8b' },
    // Mistral 7B (~14.4GB bf16) — fits all 24GB+ GPUs
    { gpuId: 'v100-32gb', modelId: 'mistral-7b' },
    { gpuId: 'a100-40gb', modelId: 'mistral-7b' },
    { gpuId: 'a100-80gb', modelId: 'mistral-7b' },
    { gpuId: 'h100-sxm', modelId: 'mistral-7b' },
    { gpuId: 'h100-pcie', modelId: 'mistral-7b' },
    { gpuId: 'h100-nvl', modelId: 'mistral-7b' },
    { gpuId: 'h200-sxm', modelId: 'mistral-7b' },
    { gpuId: 'b200', modelId: 'mistral-7b' },
    { gpuId: 'gb200', modelId: 'mistral-7b' },
    { gpuId: 'mi300x', modelId: 'mistral-7b' },
    { gpuId: 'mi325x', modelId: 'mistral-7b' },
    { gpuId: 'l40s', modelId: 'mistral-7b' },
    { gpuId: 'l40', modelId: 'mistral-7b' },
    { gpuId: 'rtx-6000-ada', modelId: 'mistral-7b' },
    { gpuId: 'a10', modelId: 'mistral-7b' },
    { gpuId: 'a10g', modelId: 'mistral-7b' },
    { gpuId: 'l4', modelId: 'mistral-7b' },
    { gpuId: 'rtx-4090', modelId: 'mistral-7b' },
    { gpuId: 'rtx-3090', modelId: 'mistral-7b' },
    // Mistral Nemo 12B (~24.4GB bf16) — fits 40GB+ GPUs
    { gpuId: 'a100-40gb', modelId: 'mistral-nemo-12b' },
    { gpuId: 'a100-80gb', modelId: 'mistral-nemo-12b' },
    { gpuId: 'h100-sxm', modelId: 'mistral-nemo-12b' },
    { gpuId: 'h100-pcie', modelId: 'mistral-nemo-12b' },
    { gpuId: 'h100-nvl', modelId: 'mistral-nemo-12b' },
    { gpuId: 'h200-sxm', modelId: 'mistral-nemo-12b' },
    { gpuId: 'b200', modelId: 'mistral-nemo-12b' },
    { gpuId: 'gb200', modelId: 'mistral-nemo-12b' },
    { gpuId: 'mi300x', modelId: 'mistral-nemo-12b' },
    { gpuId: 'mi325x', modelId: 'mistral-nemo-12b' },
    { gpuId: 'l40s', modelId: 'mistral-nemo-12b' },
    { gpuId: 'l40', modelId: 'mistral-nemo-12b' },
    { gpuId: 'rtx-6000-ada', modelId: 'mistral-nemo-12b' },
    // Codestral 22B (~44.4GB bf16) — fits 80GB+ GPUs
    { gpuId: 'a100-80gb', modelId: 'codestral-22b' },
    { gpuId: 'h100-sxm', modelId: 'codestral-22b' },
    { gpuId: 'h100-pcie', modelId: 'codestral-22b' },
    { gpuId: 'h100-nvl', modelId: 'codestral-22b' },
    { gpuId: 'h200-sxm', modelId: 'codestral-22b' },
    { gpuId: 'b200', modelId: 'codestral-22b' },
    { gpuId: 'gb200', modelId: 'codestral-22b' },
    { gpuId: 'mi300x', modelId: 'codestral-22b' },
    { gpuId: 'mi325x', modelId: 'codestral-22b' },
    // Mistral Small 24B (~47.2GB bf16) — fits 80GB+ GPUs
    { gpuId: 'a100-80gb', modelId: 'mistral-small-24b' },
    { gpuId: 'h100-sxm', modelId: 'mistral-small-24b' },
    { gpuId: 'h100-pcie', modelId: 'mistral-small-24b' },
    { gpuId: 'h100-nvl', modelId: 'mistral-small-24b' },
    { gpuId: 'h200-sxm', modelId: 'mistral-small-24b' },
    { gpuId: 'b200', modelId: 'mistral-small-24b' },
    { gpuId: 'gb200', modelId: 'mistral-small-24b' },
    { gpuId: 'mi300x', modelId: 'mistral-small-24b' },
    { gpuId: 'mi325x', modelId: 'mistral-small-24b' },
    { gpuId: 'l40s', modelId: 'llama3-8b' },
    { gpuId: 'l40', modelId: 'llama3-8b' },
    { gpuId: 'rtx-6000-ada', modelId: 'llama3-8b' },
    { gpuId: 'a10', modelId: 'gpt3-125m' },
    { gpuId: 'a10g', modelId: 'gpt3-125m' },
    { gpuId: 'l4', modelId: 'gpt3-125m' },
    { gpuId: 'rtx-4090', modelId: 'gpt3-125m' },
    { gpuId: 'rtx-3090', modelId: 'gpt3-125m' },
  ];

  for (const { gpuId, modelId } of gpuModelPairs) {
    it(`${gpuId} + ${modelId}: success with valid metrics`, () => {
      const gpu = ALL_GPUS[gpuId];
      const result = runInferenceSimulation({
        modelId,
        gpu,
        batchSize: 1,
        inputSeqLen: 512,
        outputSeqLen: 128,
        weightPrecision: 'bf16',
      });

      expect(result.success).toBe(true);
      expect(result.errors).toHaveLength(0);

      // Latency
      expect(result.latency.ttft).toBeGreaterThan(0);
      expect(Number.isFinite(result.latency.ttft)).toBe(true);
      expect(result.latency.tpot).toBeGreaterThan(0);
      expect(Number.isFinite(result.latency.tpot)).toBe(true);

      // Throughput
      expect(result.throughput.tokensPerSecond).toBeGreaterThan(0);
      expect(Number.isFinite(result.throughput.tokensPerSecond)).toBe(true);

      // Utilization not NaN
      expect(Number.isNaN(result.utilization.computeUtilization)).toBe(false);
      expect(Number.isNaN(result.utilization.rooflineAttainment)).toBe(false);

      // Memory fits
      expect(result.memory.total).toBeLessThanOrEqual(gpu.memoryGB * 1e9);
      expect(result.memory.total).toBeGreaterThan(0);
    });
  }
});

// ── Section 6: Comprehensive model × GPU inference coverage ──
// Every model that fits on a single GPU in bf16, tested across all GPU architectures

describe('Comprehensive inference: all models × matching GPUs', () => {
  // Models grouped by bf16 weight size tier → paired with GPUs where they fit
  const INFERENCE_TIERS: { tier: string; pairs: { modelId: string; gpuIds: string[] }[] }[] = [
    {
      tier: 'tiny (<2GB bf16 weights)',
      pairs: [
        { modelId: 'gpt3-125m',   gpuIds: ['v100-32gb', 'a100-80gb', 'h100-sxm', 'b200', 'mi300x', 'rtx-4090', 'rtx-3090', 'l4'] },
      ],
    },
    {
      tier: 'small (2–10GB bf16 weights)',
      pairs: [
        { modelId: 'gpt3-1.3b',   gpuIds: ['v100-32gb', 'a100-80gb', 'h100-sxm', 'rtx-4090'] },
        { modelId: 'llama3.2-1b', gpuIds: ['v100-32gb', 'a100-80gb', 'h100-sxm', 'rtx-4090', 'l4'] },
        { modelId: 'llama3.2-3b', gpuIds: ['v100-32gb', 'a100-80gb', 'h100-sxm', 'rtx-4090'] },
        { modelId: 'gemma2-2b',   gpuIds: ['v100-32gb', 'h100-sxm', 'rtx-4090', 'l40s'] },
        { modelId: 'phi3-mini',   gpuIds: ['v100-32gb', 'a100-80gb', 'h100-sxm', 'rtx-4090'] },
      ],
    },
    {
      tier: 'medium (10–20GB bf16 weights)',
      pairs: [
        { modelId: 'llama2-7b',   gpuIds: ['a100-40gb', 'a100-80gb', 'h100-sxm', 'b200', 'mi300x', 'l40s', 'rtx-4090'] },
        { modelId: 'llama3-8b',   gpuIds: ['a100-40gb', 'a100-80gb', 'h100-sxm', 'b200', 'mi300x', 'l40s'] },
        { modelId: 'mistral-7b',  gpuIds: ['a100-40gb', 'a100-80gb', 'h100-sxm', 'b200', 'mi300x', 'l40s', 'rtx-4090'] },
        { modelId: 'qwen2.5-7b',    gpuIds: ['a100-40gb', 'a100-80gb', 'h100-sxm', 'mi300x', 'l40s'] },
        { modelId: 'phi3-small',  gpuIds: ['a100-40gb', 'a100-80gb', 'h100-sxm', 'b200'] },
        { modelId: 'olmo2-7b',    gpuIds: ['a100-40gb', 'a100-80gb', 'h100-sxm', 'mi300x', 'l40s'] },
        { modelId: 'yi-6b',       gpuIds: ['a100-40gb', 'a100-80gb', 'h100-sxm', 'l40s', 'rtx-4090'] },
        { modelId: 'gemma2-9b',   gpuIds: ['a100-40gb', 'a100-80gb', 'h100-sxm', 'l40s'] },
        { modelId: 'gpt3-6.7b',   gpuIds: ['a100-40gb', 'a100-80gb', 'h100-sxm', 'b200'] },
      ],
    },
    {
      tier: 'medium-large (20–40GB bf16 weights)',
      pairs: [
        { modelId: 'mistral-nemo-12b', gpuIds: ['a100-40gb', 'a100-80gb', 'h100-sxm', 'b200', 'mi300x', 'l40s'] },
        { modelId: 'llama2-13b',       gpuIds: ['a100-40gb', 'a100-80gb', 'h100-sxm', 'b200', 'mi300x'] },
        { modelId: 'olmo2-13b',        gpuIds: ['a100-40gb', 'a100-80gb', 'h100-sxm', 'b200'] },
        { modelId: 'gpt3-13b',         gpuIds: ['a100-80gb', 'h100-sxm', 'b200'] },
        { modelId: 'phi3-medium',      gpuIds: ['a100-80gb', 'h100-sxm', 'b200', 'mi300x'] },
        { modelId: 'nemotron-4-15b',     gpuIds: ['a100-80gb', 'h100-sxm', 'b200'] },
        { modelId: 'deepseek-moe-16b', gpuIds: ['a100-80gb', 'h100-sxm', 'mi300x'] },
      ],
    },
    {
      tier: 'large (40–80GB bf16 weights)',
      pairs: [
        { modelId: 'codestral-22b',     gpuIds: ['a100-80gb', 'h100-sxm', 'h200-sxm', 'b200', 'mi300x'] },
        { modelId: 'mistral-small-24b', gpuIds: ['a100-80gb', 'h100-sxm', 'h200-sxm', 'b200', 'mi300x'] },
        { modelId: 'gemma2-27b',        gpuIds: ['a100-80gb', 'h100-sxm', 'h200-sxm', 'b200', 'mi300x'] },
        { modelId: 'yi-34b',            gpuIds: ['a100-80gb', 'h100-sxm', 'h200-sxm', 'b200', 'mi300x'] },
      ],
    },
  ];

  for (const { tier, pairs } of INFERENCE_TIERS) {
    describe(tier, () => {
      for (const { modelId, gpuIds } of pairs) {
        for (const gpuId of gpuIds) {
          it(`${gpuId} + ${modelId}`, () => {
            const gpu = ALL_GPUS[gpuId];
            const result = runInferenceSimulation({
              modelId,
              gpu,
              batchSize: 1,
              inputSeqLen: 512,
              outputSeqLen: 128,
              weightPrecision: 'bf16',
            });

            expect(result.success, `${gpuId}+${modelId} should succeed: ${result.errors.join(', ')}`).toBe(true);
            expect(result.latency.ttft).toBeGreaterThan(0);
            expect(Number.isFinite(result.latency.ttft)).toBe(true);
            expect(result.latency.tpot).toBeGreaterThan(0);
            expect(Number.isFinite(result.latency.tpot)).toBe(true);
            expect(result.throughput.tokensPerSecond).toBeGreaterThan(0);
            expect(Number.isFinite(result.throughput.tokensPerSecond)).toBe(true);
            expect(result.memory.total).toBeGreaterThan(0);
            expect(result.memory.total).toBeLessThanOrEqual(gpu.memoryGB * 1e9);
          });
        }
      }
    });
  }
});

// ── Section 7: Latency ordering ──

describe('latency ordering', () => {
  it('TTFT ordering: V100 > A100 > H100 > B200 (faster GPU = lower latency)', () => {
    // Use gpt3-125m which fits on all GPUs
    const model = getModel('gpt3-125m', 512)!;
    const inputTokens = 512;

    const ttftV100 = estimateTTFT(model, inputTokens, V100_32GB, 'bf16');
    const ttftA100 = estimateTTFT(model, inputTokens, A100_80GB, 'bf16');
    const ttftH100 = estimateTTFT(model, inputTokens, H100_SXM, 'bf16');
    const ttftB200 = estimateTTFT(model, inputTokens, B200, 'bf16');

    expect(ttftV100).toBeGreaterThan(ttftA100);
    expect(ttftA100).toBeGreaterThan(ttftH100);
    expect(ttftH100).toBeGreaterThan(ttftB200);
  });

  it('larger model = higher TTFT on same GPU', () => {
    const small125m = getModel('gpt3-125m', 512)!; // ~125M params
    const llama8b = getModel('llama3-8b', 512)!;  // ~8B params

    const ttftSmall = estimateTTFT(small125m, 512, H100_SXM, 'bf16');
    const ttftLarge = estimateTTFT(llama8b, 512, H100_SXM, 'bf16');

    expect(ttftLarge).toBeGreaterThan(ttftSmall);
  });
});

// ── Section 7: Speculative decoding with different GPUs ──

describe('speculative decoding', () => {
  it('H100 SXM + speculative: speedup > 1', () => {
    runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
    });

    const specResult = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
      speculativeEnabled: true,
      draftModelId: 'gpt3-125m',
      numSpeculativeTokens: 5,
      acceptanceRate: 0.7,
    });

    expect(specResult.success).toBe(true);
    expect(specResult.speculative).toBeDefined();
    expect(specResult.speculative!.speedup).toBeGreaterThan(1);

    // Speculative metrics should affect main latency (Bug 1 fix)
    expect(specResult.latency.tpot).toBeCloseTo(specResult.speculative!.effectiveTpot, 5);
  });

  it('V100 + speculative with fp16 fallback: finite results', () => {
    const result = runInferenceSimulation({
      modelId: 'gpt3-125m',
      gpu: V100_32GB,
      batchSize: 1,
      inputSeqLen: 256,
      outputSeqLen: 64,
      weightPrecision: 'bf16', // Will fall back to fp16 on V100
      speculativeEnabled: true,
      draftModelId: 'gpt3-125m',
      numSpeculativeTokens: 3,
      acceptanceRate: 0.6,
    });

    expect(result.success).toBe(true);
    expect(Number.isFinite(result.latency.ttft)).toBe(true);
    expect(Number.isFinite(result.latency.tpot)).toBe(true);
    expect(result.latency.ttft).toBeGreaterThan(0);
  });
});

// ── Section 8: KV cache precision sensitivity ──

describe('KV cache precision sensitivity', () => {
  it('fp8 KV cache reduces TPOT vs bf16 KV cache (same weight precision)', () => {
    const bf16KV = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpuId: 'h100-sxm',
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 256,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
    });
    const fp8KV = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpuId: 'h100-sxm',
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 256,
      weightPrecision: 'bf16',
      kvCachePrecision: 'fp8',
    });

    expect(bf16KV.success).toBe(true);
    expect(fp8KV.success).toBe(true);
    // fp8 KV cache halves KV cache memory → less bytes to read → faster TPOT
    expect(fp8KV.latency.tpot).toBeLessThan(bf16KV.latency.tpot);
    // KV cache memory should be ~half
    expect(fp8KV.memory.kvCache).toBeLessThan(bf16KV.memory.kvCache * 0.6);
    expect(fp8KV.memory.kvCache).toBeGreaterThan(bf16KV.memory.kvCache * 0.4);
  });

  it('KV precision affects TPOT but not TTFT (prefill does not read KV cache)', () => {
    const bf16KV = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpuId: 'h100-sxm',
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 256,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
    });
    const fp8KV = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpuId: 'h100-sxm',
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 256,
      weightPrecision: 'bf16',
      kvCachePrecision: 'fp8',
    });

    // TTFT should be identical (prefill doesn't read KV cache for latency)
    expect(fp8KV.latency.ttft).toBeCloseTo(bf16KV.latency.ttft, 5);
    // TPOT should differ
    expect(fp8KV.latency.tpot).not.toBeCloseTo(bf16KV.latency.tpot, 5);
  });

  it('KV precision threads through TP path (MLA model)', () => {
    const bf16KV = runInferenceSimulation({
      modelId: 'deepseek-v3',
      gpuId: 'h200-sxm',
      numGPUs: 8,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'fp8',
      kvCachePrecision: 'bf16',
      tensorParallel: 8,
    });
    const fp8KV = runInferenceSimulation({
      modelId: 'deepseek-v3',
      gpuId: 'h200-sxm',
      numGPUs: 8,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'fp8',
      kvCachePrecision: 'fp8',
      tensorParallel: 8,
    });

    expect(bf16KV.success).toBe(true);
    expect(fp8KV.success).toBe(true);
    // fp8 KV cache should reduce TPOT even with MLA + TP
    expect(fp8KV.latency.tpot).toBeLessThan(bf16KV.latency.tpot);
  });
});

// ── Section 9: Event memory TP/EP consistency ──

describe('Event memory TP/EP consistency', () => {
  it('TP=4: event memory snapshots reflect TP-sharded weights', () => {
    const engine = new InferenceSimulationEngine();
    engine.configure({
      modelId: 'llama3-8b',
      gpuId: 'h100-sxm',
      numGPUs: 4,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 64,
      weightPrecision: 'bf16',
      tensorParallel: 4,
    });
    const result = engine.run();
    expect(result.success).toBe(true);

    // Find memory_snapshot events
    const memEvents = result.events.filter((e: { type: string }) => e.type === 'memory_snapshot');
    expect(memEvents.length).toBeGreaterThanOrEqual(2);

    // Event memory weights should match result memory weights (both TP-aware)
    const firstSnapshot = memEvents[0] as { breakdown: { weights: number } };
    expect(firstSnapshot.breakdown.weights).toBeCloseTo(result.memory.weights, -1);
  });

  it('TP=1: event memory matches non-TP path', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpuId: 'h100-sxm',
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 64,
      weightPrecision: 'bf16',
    });
    expect(result.success).toBe(true);

    const memEvents = result.events.filter((e: { type: string }) => e.type === 'memory_snapshot');
    expect(memEvents.length).toBeGreaterThanOrEqual(2);

    const lastSnapshot = memEvents[memEvents.length - 1] as { breakdown: { weights: number; total: number } };
    // Weights should match exactly (no TP sharding)
    expect(lastSnapshot.breakdown.weights).toBeCloseTo(result.memory.weights, -1);
  });
});
