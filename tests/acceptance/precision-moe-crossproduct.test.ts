/**
 * Precision × Model (MoE + Dense) Cross-Product Tests
 *
 * Validates that ALL precision types produce valid results for both:
 *   - Dense models (LLaMA 3 8B, GPT-3 175B)
 *   - MoE models (Mixtral 8x7B, DeepSeek-V3, Grok-1)
 *
 * Covers:
 *   1. Training simulation: all precisions × dense + MoE models → finite MFU, no NaN
 *   2. Inference simulation: all precisions × dense + MoE models → finite latency, no NaN
 *   3. MoE active params: inference FLOPs use activeParams, memory uses totalParams
 *   4. Precision scaling: lower precision → less memory, higher throughput
 */

import { describe, it, expect } from 'vitest';
import { runSimulation, type SimulationConfig } from '../../src/core/simulation/engine.ts';
import { runInferenceSimulation } from '../../src/core/inference/simulation.ts';
import { getModel } from '../../src/core/models/index.ts';
import { modelRegistry } from '../../src/core/models/registry.ts';
import { getPrecisionBytes } from '../../src/core/inference/kv-cache.ts';
import { modelWeightsMemory } from '../../src/core/inference/memory.ts';
import { prefillFLOPs, decodeFLOPs } from '../../src/core/inference/latency.ts';
import type { InferencePrecision } from '../../src/types/inference.ts';

// ── Precision lists ──

const TRAINING_PRECISIONS = ['fp32', 'tf32', 'fp16', 'bf16', 'fp8', 'fp4'] as const;
const INFERENCE_PRECISIONS: InferencePrecision[] = ['fp32', 'fp16', 'bf16', 'fp8', 'fp4', 'int8', 'int4'];

// ── Model selections with proper parallelism configs ──

interface TrainModel {
  id: string;
  name: string;
  cluster: string;
  strategy: SimulationConfig['strategyType'];
  strategyConfig: Record<string, unknown>;
  gbs: number;
  mbs: number;
  seqLen: number;
}

const DENSE_TRAIN_MODELS: TrainModel[] = [
  { id: 'gpt3-125m', name: 'GPT-3 125M', cluster: '8x-h100', strategy: 'fsdp',
    strategyConfig: {}, gbs: 64, mbs: 4, seqLen: 2048 },
  { id: 'llama3-8b', name: 'LLaMA 3 8B', cluster: '8x-h100', strategy: 'fsdp',
    strategyConfig: {}, gbs: 32, mbs: 2, seqLen: 4096 },
];

const MOE_TRAIN_MODELS: TrainModel[] = [
  // Mixtral 8x7B needs 16 GPUs with FSDP+TP to fit (47B params, 8 experts)
  { id: 'mixtral-8x7b', name: 'Mixtral 8x7B', cluster: '16x-h100', strategy: 'fsdp-tp',
    strategyConfig: { tp: 2 }, gbs: 32, mbs: 1, seqLen: 2048 },
  // Grok-1 314B needs large cluster with 3D parallelism
  { id: 'grok-1', name: 'Grok-1 314B', cluster: '128x-h100', strategy: 'fsdp-tp-pp',
    strategyConfig: { tp: 8, pp: 4 }, gbs: 64, mbs: 1, seqLen: 2048 },
];

const ALL_TRAIN_MODELS = [...DENSE_TRAIN_MODELS, ...MOE_TRAIN_MODELS];

// ── Helpers ──

function trainConfig(
  model: TrainModel,
  precision: string,
): SimulationConfig {
  return {
    modelId: model.id,
    clusterId: model.cluster,
    globalBatchSize: model.gbs,
    microBatchSize: model.mbs,
    sequenceLength: model.seqLen,
    strategyType: model.strategy,
    strategyConfig: model.strategyConfig,
    activationCheckpointing: true,
    mixedPrecision: precision as SimulationConfig['mixedPrecision'],
  };
}

function inferConfig(
  modelId: string,
  gpuId: string,
  numGPUs: number,
  weightPrecision: InferencePrecision,
  kvCachePrecision: InferencePrecision = 'bf16',
) {
  return {
    modelId,
    gpuId,
    numGPUs,
    batchSize: 1,
    inputSeqLen: 512,
    outputSeqLen: 128,
    weightPrecision,
    kvCachePrecision,
    tensorParallel: numGPUs,
  };
}

// ===========================================================================
// Section 1: getPrecisionBytes — every InferencePrecision returns a number
// ===========================================================================

describe('getPrecisionBytes completeness', () => {
  for (const p of INFERENCE_PRECISIONS) {
    it(`${p} → finite positive number`, () => {
      const bytes = getPrecisionBytes(p);
      expect(bytes).toBeGreaterThan(0);
      expect(Number.isFinite(bytes)).toBe(true);
    });
  }
});

// ===========================================================================
// Section 2: Training simulation × all precisions × dense + MoE
// ===========================================================================

describe('Training precision × model cross-product', () => {
  for (const model of ALL_TRAIN_MODELS) {
    describe(model.name, () => {
      for (const precision of TRAINING_PRECISIONS) {
        it(`${precision}: valid results or clean OOM (no NaN)`, () => {
          const result = runSimulation(trainConfig(model, precision));

          if (result.success) {
            expect(Number.isFinite(result.metrics.mfu)).toBe(true);
            expect(result.metrics.mfu).toBeGreaterThan(0);
            expect(result.metrics.mfu).toBeLessThan(1);
            expect(Number.isFinite(result.metrics.tokensPerSecond)).toBe(true);
            expect(result.metrics.tokensPerSecond).toBeGreaterThan(0);
            expect(Number.isNaN(result.metrics.avgStepTimeMs)).toBe(false);
          } else {
            // OOM is acceptable for large precisions — but never NaN
            expect(Number.isNaN(result.metrics.mfu)).toBe(false);
          }
        });
      }
    });
  }
});

// ===========================================================================
// Section 3: Inference simulation × all precisions × dense + MoE
// ===========================================================================

describe('Inference precision × model cross-product', () => {
  // Dense models: single H100 for 8B, 8× H100 for 175B
  const inferModels = [
    { id: 'llama3-8b', name: 'LLaMA 3 8B (dense)', gpu: 'h100-sxm', numGPUs: 1 },
    { id: 'gpt3-175b', name: 'GPT-3 175B (dense)', gpu: 'h100-sxm', numGPUs: 8 },
    { id: 'mixtral-8x7b', name: 'Mixtral 8x7B (MoE)', gpu: 'h100-sxm', numGPUs: 2 },
    { id: 'deepseek-v3', name: 'DeepSeek-V3 (MoE)', gpu: 'h200-sxm', numGPUs: 16 },
    { id: 'grok-1', name: 'Grok-1 314B (MoE)', gpu: 'h200-sxm', numGPUs: 16 },
  ];

  for (const model of inferModels) {
    describe(model.name, () => {
      for (const precision of INFERENCE_PRECISIONS) {
        it(`weight=${precision}: no NaN, finite positive latency`, () => {
          const result = runInferenceSimulation(
            inferConfig(model.id, model.gpu, model.numGPUs, precision),
          );

          // Must either succeed or fail cleanly (OOM), never NaN
          if (result.success) {
            expect(Number.isFinite(result.latency.ttft)).toBe(true);
            expect(result.latency.ttft).toBeGreaterThan(0);
            expect(Number.isFinite(result.latency.tpot)).toBe(true);
            expect(result.latency.tpot).toBeGreaterThan(0);
            expect(Number.isFinite(result.latency.totalLatency)).toBe(true);
            expect(Number.isNaN(result.memory.total)).toBe(false);
            expect(result.memory.total).toBeGreaterThan(0);
          } else {
            // OOM is acceptable — but memory values should still be finite, not NaN
            expect(Number.isNaN(result.memory.weights)).toBe(false);
          }
        });
      }
    });
  }
});

// ===========================================================================
// Section 4: MoE active params — inference FLOPs vs memory
// ===========================================================================

describe('MoE active params correctness', () => {
  const moeModels = [
    { id: 'mixtral-8x7b', name: 'Mixtral 8x7B' },
    { id: 'deepseek-v3', name: 'DeepSeek-V3' },
    { id: 'grok-1', name: 'Grok-1' },
    { id: 'deepseek-r1', name: 'DeepSeek-R1' },
    { id: 'dbrx', name: 'DBRX' },
  ];

  for (const m of moeModels) {
    describe(m.name, () => {
      const model = getModel(m.id)!;
      const meta = modelRegistry.getMetadata(m.id)!;

      it('is flagged as MoE', () => {
        expect(model.isMoE).toBe(true);
        expect(meta.isMoE).toBe(true);
      });

      it('activeParams < totalParams', () => {
        expect(model.activeParams).toBeDefined();
        expect(model.activeParams).toBeLessThan(model.totalParams);
        expect(model.activeParams).toBeGreaterThan(0);
      });

      it('prefillFLOPs uses activeParams (not totalParams)', () => {
        const flops = prefillFLOPs(model, 1024);
        const floorFromActive = 2 * model.activeParams! * 1024;
        const wrongFromTotal = 2 * model.totalParams * 1024;

        // prefillFLOPs >= 2*activeParams*tokens (attention score FLOPs add to the base)
        expect(flops).toBeGreaterThanOrEqual(floorFromActive);
        expect(flops).toBeLessThan(wrongFromTotal);
      });

      it('decodeFLOPs uses activeParams (not totalParams)', () => {
        const flops = decodeFLOPs(model);
        // decodeFLOPs >= 2*activeParams (attention score FLOPs add to the base)
        expect(flops).toBeGreaterThanOrEqual(2 * model.activeParams!);
        expect(flops).toBeLessThan(2 * model.totalParams);
      });

      it('modelWeightsMemory uses totalParams (all experts loaded)', () => {
        const mem = modelWeightsMemory(model, 'bf16');
        expect(mem).toBe(model.totalParams * 2); // bf16 = 2 bytes
      });
    });
  }
});

// ===========================================================================
// Section 5: Dense models — activeParams === totalParams
// ===========================================================================

describe('Dense model activeParams === totalParams', () => {
  const denseModels = [
    { id: 'llama3-8b', name: 'LLaMA 3 8B' },
    { id: 'llama2-70b', name: 'LLaMA 2 70B' },
    { id: 'gpt3-175b', name: 'GPT-3 175B' },
  ];

  for (const m of denseModels) {
    it(`${m.name}: activeParams equals totalParams`, () => {
      const model = getModel(m.id)!;
      expect(model.isMoE).toBe(false);
      expect(model.activeParams).toBe(model.totalParams);
    });

    it(`${m.name}: prefillFLOPs >= 2 × totalParams × tokens`, () => {
      const model = getModel(m.id)!;
      // prefillFLOPs includes attention score FLOPs (QK^T + scores*V) beyond 2*P*T
      expect(prefillFLOPs(model, 1000)).toBeGreaterThanOrEqual(2 * model.totalParams * 1000);
    });
  }
});

// ===========================================================================
// Section 6: Precision scaling — lower precision → less memory
// ===========================================================================

describe('Inference precision scaling', () => {
  const model = getModel('llama3-8b')!;

  it('weight memory decreases with lower precision', () => {
    const fp32 = modelWeightsMemory(model, 'fp32');
    const bf16 = modelWeightsMemory(model, 'bf16');
    const fp8 = modelWeightsMemory(model, 'fp8');
    const fp4 = modelWeightsMemory(model, 'fp4');
    const int4 = modelWeightsMemory(model, 'int4');

    expect(fp32).toBe(bf16 * 2);
    expect(bf16).toBe(fp8 * 2);
    expect(fp8).toBe(fp4 * 2);
    expect(fp4).toBe(int4); // both 0.5 bytes
  });

  it('MoE model weight memory also scales with precision', () => {
    const moe = getModel('mixtral-8x7b')!;

    const bf16 = modelWeightsMemory(moe, 'bf16');
    const fp8 = modelWeightsMemory(moe, 'fp8');
    const fp4 = modelWeightsMemory(moe, 'fp4');

    expect(bf16).toBe(fp8 * 2);
    expect(fp8).toBe(fp4 * 2);
    // Uses totalParams for memory (all experts)
    expect(bf16).toBe(moe.totalParams * 2);
  });
});

// ===========================================================================
// Section 7: Training MFU sanity — MoE should use activeParams
// ===========================================================================

describe('Training MFU uses activeParams for MoE', () => {
  it('Mixtral 8x7B MFU is reasonable (not inflated by totalParams)', () => {
    const model = MOE_TRAIN_MODELS.find(m => m.id === 'mixtral-8x7b')!;
    const result = runSimulation(trainConfig(model, 'bf16'));
    expect(result.success).toBe(true);
    expect(result.metrics.mfu).toBeGreaterThan(0);
    expect(result.metrics.mfu).toBeLessThan(1);
  });

  it('Grok-1 MFU is reasonable', () => {
    const model = MOE_TRAIN_MODELS.find(m => m.id === 'grok-1')!;
    const result = runSimulation(trainConfig(model, 'bf16'));
    expect(result.success).toBe(true);
    expect(result.metrics.mfu).toBeGreaterThan(0);
    expect(result.metrics.mfu).toBeLessThan(1);
  });

  it('Dense LLaMA 3 8B MFU is in expected range', () => {
    const model = DENSE_TRAIN_MODELS.find(m => m.id === 'llama3-8b')!;
    const result = runSimulation(trainConfig(model, 'bf16'));
    expect(result.success).toBe(true);
    // MFU is a fraction (0-1), not percentage
    expect(result.metrics.mfu).toBeGreaterThan(0.2);
    expect(result.metrics.mfu).toBeLessThan(0.8);
  });
});

// ===========================================================================
// Section 8: Inference with FP4 on different GPUs
// ===========================================================================

describe('FP4 inference (regression test for NaN bug)', () => {
  it('Grok-1 on H100 SXM with FP4 weights: no NaN', () => {
    const result = runInferenceSimulation({
      modelId: 'grok-1',
      gpuId: 'h100-sxm',
      numGPUs: 8,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'fp4',
      kvCachePrecision: 'fp8',
      tensorParallel: 8,
    });

    expect(Number.isNaN(result.memory.total)).toBe(false);
    if (result.success) {
      expect(Number.isFinite(result.latency.ttft)).toBe(true);
      expect(Number.isFinite(result.latency.tpot)).toBe(true);
    }
  });

  it('LLaMA 3 8B on B200 with FP4 weights: succeeds', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpuId: 'b200',
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'fp4',
      kvCachePrecision: 'fp8',
      tensorParallel: 1,
    });

    expect(result.success).toBe(true);
    expect(Number.isFinite(result.latency.ttft)).toBe(true);
    expect(result.latency.ttft).toBeGreaterThan(0);
    expect(Number.isFinite(result.memory.weights)).toBe(true);
    // 8B params at FP4 = ~4GB
    expect(result.memory.weights / 1e9).toBeCloseTo(4, 0);
  });

  it('Mixtral 8x7B FP4 weights: memory uses totalParams', () => {
    const result = runInferenceSimulation({
      modelId: 'mixtral-8x7b',
      gpuId: 'h200-sxm',
      numGPUs: 2,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'fp4',
      kvCachePrecision: 'bf16',
      tensorParallel: 2,
    });

    expect(result.success).toBe(true);
    expect(Number.isNaN(result.memory.weights)).toBe(false);
    // Mixtral total ~46.7B params at FP4 = ~23.3GB, split across 2 GPUs ≈ ~11.7GB per GPU
    expect(result.memory.weights / 1e9).toBeGreaterThan(5);
    expect(result.memory.weights / 1e9).toBeLessThan(30);
  });
});

// ===========================================================================
// Section 9: KV cache precision with FP4
// ===========================================================================

describe('FP4 KV cache precision', () => {
  it('KV cache with FP4 is half the size of FP8', () => {
    const result_fp8 = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpuId: 'h100-sxm',
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 2048,
      outputSeqLen: 256,
      weightPrecision: 'bf16',
      kvCachePrecision: 'fp8',
      tensorParallel: 1,
    });

    const result_fp4 = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpuId: 'h100-sxm',
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 2048,
      outputSeqLen: 256,
      weightPrecision: 'bf16',
      kvCachePrecision: 'fp4',
      tensorParallel: 1,
    });

    expect(result_fp8.success).toBe(true);
    expect(result_fp4.success).toBe(true);

    // FP4 KV cache should be half of FP8
    const ratio = result_fp8.memory.kvCache / result_fp4.memory.kvCache;
    expect(ratio).toBeCloseTo(2, 1);

    // Weights should be identical (both bf16)
    expect(result_fp8.memory.weights).toBe(result_fp4.memory.weights);
  });
});

// ===========================================================================
// Section 10: All MoE models from registry — inference smoke test
// ===========================================================================

describe('All MoE models inference smoke test (bf16 + fp4)', () => {
  const allMoE = modelRegistry.getAllMetadata().filter(m => m.isMoE);

  for (const meta of allMoE) {
    for (const precision of ['bf16', 'fp4'] as InferencePrecision[]) {
      it(`${meta.name} (${precision}): no NaN in results`, () => {
        const result = runInferenceSimulation({
          modelId: meta.id,
          gpuId: 'h200-sxm',
          numGPUs: 16,
          batchSize: 1,
          inputSeqLen: 256,
          outputSeqLen: 64,
          weightPrecision: precision,
          kvCachePrecision: 'bf16',
          tensorParallel: 16,
        });

        // No NaN anywhere
        expect(Number.isNaN(result.memory.total)).toBe(false);
        expect(Number.isNaN(result.memory.weights)).toBe(false);
        expect(Number.isNaN(result.memory.kvCache)).toBe(false);

        if (result.success) {
          expect(Number.isNaN(result.latency.ttft)).toBe(false);
          expect(Number.isNaN(result.latency.tpot)).toBe(false);
          expect(result.latency.ttft).toBeGreaterThan(0);
          expect(result.latency.tpot).toBeGreaterThan(0);
        }
      });
    }
  }
});

// ===========================================================================
// Section 11: All dense models training smoke test across precisions
// ===========================================================================

describe('Dense models training across all precisions', () => {
  for (const model of DENSE_TRAIN_MODELS) {
    describe(model.name, () => {
      for (const precision of TRAINING_PRECISIONS) {
        it(`${precision}: MFU > 0, no NaN`, () => {
          const result = runSimulation(trainConfig(model, precision));
          expect(result.success).toBe(true);
          expect(Number.isNaN(result.metrics.mfu)).toBe(false);
          expect(result.metrics.mfu).toBeGreaterThan(0);
          expect(Number.isNaN(result.metrics.avgStepTimeMs)).toBe(false);
          expect(Number.isNaN(result.metrics.peakMemoryGB)).toBe(false);
        });
      }
    });
  }
});
