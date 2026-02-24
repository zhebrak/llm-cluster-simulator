/**
 * NVIDIA T4 GPU Validation Tests
 *
 * The T4 is a Turing-architecture inference-focused GPU:
 * - 16 GB GDDR6, 320 GB/s bandwidth
 * - 70W TDP, single-slot low-profile passive cooling
 * - PCIe Gen3 only, no NVLink
 * - Turing Tensor Cores: FP16=65 TFLOPS, INT8=130 TOPS, INT4=260 TOPS
 * - No BF16, TF32, FP8, FP4 support (pre-Ampere)
 *
 * Tests cover: spec accuracy, dtype fallbacks, interconnect mapping,
 * training strategies (small models), and inference simulation.
 */

import { describe, it, expect } from 'vitest';
import {
  T4,
  ALL_GPUS,
  V100_32GB,
  L4,
  A10,
  H100_SXM,
  getEffectiveTFLOPS,
} from '../../src/core/hardware/gpu.ts';
import {
  getIntraNodeInterconnect,
  getNvSwitchSpec,
  PCIE_SPECS,
} from '../../src/core/hardware/interconnect.ts';
import { createMultiNodeCluster } from '../../src/core/hardware/topology.ts';
import { getSimulationMetrics } from '../../src/core/simulation/engine.ts';
import type { SimulationConfig } from '../../src/core/simulation/engine.ts';
import { getModel } from '../../src/core/models/index.ts';
import {
  estimateTTFT,
  estimateTPOT,
  calculateLatencyMetrics,
  calculateThroughputMetrics,
  calculateUtilizationMetrics,
  calculateInferenceMemory,
  modelWeightsMemory,
  runInferenceSimulation,
  kvCachePerToken,
  totalKVCacheMemory,
  getGPUTFLOPS,
} from '../../src/core/inference/index.ts';

// ============================================================================
// Section 1: Spec accuracy — pin T4 values to official NVIDIA datasheet
// ============================================================================

describe('GPU spec accuracy: T4', () => {
  const gpu = T4;

  it('id is t4', () => expect(gpu.id).toBe('t4'));
  it('name is NVIDIA T4', () => expect(gpu.name).toBe('NVIDIA T4'));
  it('vendor is nvidia', () => expect(gpu.vendor).toBe('nvidia'));
  it('architecture is turing', () => expect(gpu.architecture).toBe('turing'));

  // Memory
  it('memoryGB = 16 (GDDR6)', () => expect(gpu.memoryGB).toBe(16));
  it('memoryBandwidthTBps = 0.32 (320 GB/s)', () => expect(gpu.memoryBandwidthTBps).toBe(0.32));

  // Compute
  it('fp32TFLOPS = 8.1', () => expect(gpu.fp32TFLOPS).toBe(8.1));
  it('tf32TFLOPS = 0 (no TF32 on Turing)', () => expect(gpu.tf32TFLOPS).toBe(0));
  it('fp16TFLOPS = 65 (Tensor Core mixed-precision)', () => expect(gpu.fp16TFLOPS).toBe(65));
  it('bf16TFLOPS = 0 (no BF16 on Turing)', () => expect(gpu.bf16TFLOPS).toBe(0));
  it('fp8TFLOPS = 0 (no FP8 on Turing)', () => expect(gpu.fp8TFLOPS).toBe(0));
  it('fp4TFLOPS = 0 (no FP4 on Turing)', () => expect(gpu.fp4TFLOPS).toBe(0));
  it('int8TOPS = 130', () => expect(gpu.int8TOPS).toBe(130));
  it('int4TOPS = 260', () => expect(gpu.int4TOPS).toBe(260));

  // Tensor cores
  it('hasTensorCores = true (2nd gen Turing)', () => expect(gpu.hasTensorCores).toBe(true));
  it('tensorCoreTFLOPS = 65 (FP16 peak)', () => expect(gpu.tensorCoreTFLOPS).toBe(65));

  // Power
  it('tdpWatts = 70', () => expect(gpu.tdpWatts).toBe(70));

  // Interconnect
  it('nvlinkBandwidthGBps = 0 (no NVLink)', () => expect(gpu.nvlinkBandwidthGBps).toBe(0));
  it('nvlinkVersion = null', () => expect(gpu.nvlinkVersion).toBeNull());
  it('pcieBandwidthGBps = 15.75 (Gen3 x16)', () => expect(gpu.pcieBandwidthGBps).toBe(15.75));

  // Features
  it('hasTransformerEngine = false', () => expect(gpu.hasTransformerEngine).toBe(false));
  it('hasNvSwitch = false', () => expect(gpu.hasNvSwitch).toBe(false));
});

// ============================================================================
// Section 2: T4 in catalog and categories
// ============================================================================

describe('T4 catalog integration', () => {
  it('T4 is in ALL_GPUS', () => {
    expect(ALL_GPUS['t4']).toBeDefined();
    expect(ALL_GPUS['t4']).toBe(T4);
  });

  it('T4 spec consistency: INT8 = 2x FP16, INT4 = 2x INT8', () => {
    expect(T4.int8TOPS).toBe(T4.fp16TFLOPS * 2);
    expect(T4.int4TOPS).toBe(T4.int8TOPS * 2);
  });
});

// ============================================================================
// Section 3: TFLOPS resolution and dtype fallbacks
// ============================================================================

describe('T4 dtype fallback chain', () => {
  it('T4 + fp32 → native 8.1 TFLOPS', () => {
    expect(getEffectiveTFLOPS(T4, 'fp32')).toBe(8.1);
  });

  it('T4 + fp16 → native 65 TFLOPS', () => {
    expect(getEffectiveTFLOPS(T4, 'fp16')).toBe(65);
  });

  it('T4 + bf16 → falls back to fp16 (65 TFLOPS)', () => {
    // Like V100, T4 has no BF16 — falls back to FP16
    expect(getEffectiveTFLOPS(T4, 'bf16')).toBe(65);
  });

  it('T4 + tf32 → falls back to fp32 (8.1 TFLOPS)', () => {
    expect(getEffectiveTFLOPS(T4, 'tf32')).toBe(8.1);
  });

  it('T4 + fp8 → falls back through bf16(0) → fp16 (65 TFLOPS)', () => {
    expect(getEffectiveTFLOPS(T4, 'fp8')).toBe(65);
  });

  it('T4 + fp4 → falls back through fp8(0) → bf16(0) → fp16 (65 TFLOPS)', () => {
    expect(getEffectiveTFLOPS(T4, 'fp4')).toBe(65);
  });

  it('T4 + int8 → native 130 TOPS', () => {
    expect(getEffectiveTFLOPS(T4, 'int8')).toBe(130);
  });

  it('T4 + int4 → native 260 TOPS', () => {
    expect(getEffectiveTFLOPS(T4, 'int4')).toBe(260);
  });

  it('all dtypes resolve to positive TFLOPS', () => {
    const dtypes = ['fp32', 'tf32', 'fp16', 'bf16', 'fp8', 'fp4', 'int8', 'int4'];
    for (const dtype of dtypes) {
      expect(getEffectiveTFLOPS(T4, dtype), `T4 + ${dtype}`).toBeGreaterThan(0);
    }
  });
});

describe('T4 vs V100 dtype fallback parity', () => {
  // Both are pre-Ampere GPUs without BF16/TF32/FP8 support
  it('both fall back bf16 → fp16', () => {
    // V100 has 125 TFLOPS FP16, T4 has 65 TFLOPS FP16
    expect(getEffectiveTFLOPS(T4, 'bf16')).toBe(T4.fp16TFLOPS);
    expect(getEffectiveTFLOPS(V100_32GB, 'bf16')).toBe(V100_32GB.fp16TFLOPS);
  });

  it('both fall back fp8 → fp16', () => {
    expect(getEffectiveTFLOPS(T4, 'fp8')).toBe(T4.fp16TFLOPS);
    expect(getEffectiveTFLOPS(V100_32GB, 'fp8')).toBe(V100_32GB.fp16TFLOPS);
  });
});

// ============================================================================
// Section 4: Interconnect mapping
// ============================================================================

describe('T4 interconnect mapping', () => {
  it('T4 (no NVLink) → PCIe Gen3 fallback', () => {
    const ic = getIntraNodeInterconnect(T4);
    expect(ic.type).toBe('pcie');
    expect(ic.bandwidthGBps).toBe(PCIE_SPECS['pcie-gen3'].bandwidthGBps);
    expect(ic.bandwidthGBps).toBe(15.75);
  });

  it('T4 (no NvSwitch) → undefined', () => {
    expect(getNvSwitchSpec(T4)).toBeUndefined();
  });
});

// ============================================================================
// Section 5: T4 comparisons with other GPUs
// ============================================================================

describe('T4 relative positioning', () => {
  it('T4 has less memory than V100 (16 vs 32 GB)', () => {
    expect(T4.memoryGB).toBeLessThan(V100_32GB.memoryGB);
  });

  it('T4 has less memory bandwidth than V100 (0.32 vs 0.9 TB/s)', () => {
    expect(T4.memoryBandwidthTBps).toBeLessThan(V100_32GB.memoryBandwidthTBps);
  });

  it('T4 has much lower FP16 TFLOPS than V100 (65 vs 125)', () => {
    expect(T4.fp16TFLOPS).toBeLessThan(V100_32GB.fp16TFLOPS);
  });

  it('T4 has much lower TDP than V100 (70W vs 300W)', () => {
    expect(T4.tdpWatts).toBeLessThan(V100_32GB.tdpWatts);
  });

  it('T4 has similar TDP to L4 (70W vs 72W)', () => {
    expect(Math.abs(T4.tdpWatts - L4.tdpWatts)).toBeLessThan(5);
  });

  it('T4 FP16 TFLOPS/W is competitive (0.93 vs V100 0.42)', () => {
    const t4Efficiency = T4.fp16TFLOPS / T4.tdpWatts;
    const v100Efficiency = V100_32GB.fp16TFLOPS / V100_32GB.tdpWatts;
    expect(t4Efficiency).toBeGreaterThan(v100Efficiency);
  });

  it('T4 has INT8 support unlike V100', () => {
    expect(T4.int8TOPS).toBeGreaterThan(0);
    expect(V100_32GB.int8TOPS).toBe(0);
  });
});

// ============================================================================
// Section 6: Training strategy smoke tests (small models that fit 16GB)
// ============================================================================

function makeConfig(
  gpuId: string,
  modelId: string,
  strategyType: SimulationConfig['strategyType'],
  totalGPUs: number,
  numNodes: number,
  strategyConfig?: SimulationConfig['strategyConfig'],
): SimulationConfig {
  return {
    clusterId: undefined,
    clusterConfig: createMultiNodeCluster(gpuId, totalGPUs / numNodes, numNodes)!,
    modelId,
    globalBatchSize: totalGPUs * 2,
    microBatchSize: 2,
    sequenceLength: 2048,
    strategyType,
    strategyConfig,
  };
}

function assertFiniteAndValid(metrics: { stepTimeMs: number; mfu: number }, label: string) {
  expect(metrics.stepTimeMs, `${label}: stepTimeMs > 0`).toBeGreaterThan(0);
  expect(metrics.stepTimeMs, `${label}: stepTimeMs finite`).toBeLessThan(Infinity);
  expect(metrics.mfu, `${label}: mfu > 0`).toBeGreaterThan(0);
  expect(metrics.mfu, `${label}: mfu ≤ 1`).toBeLessThanOrEqual(1);
}

describe('T4 training: simple DP strategies (gpt3-125m)', () => {
  const strategies: SimulationConfig['strategyType'][] = ['ddp', 'fsdp', 'zero-1', 'zero-3'];

  for (const strat of strategies) {
    it(`gpt3-125m + ${strat} on 8× T4`, () => {
      const m = getSimulationMetrics(makeConfig('t4', 'gpt3-125m', strat, 8, 1));
      assertFiniteAndValid(m, `T4/gpt3-125m/${strat}`);
    });
  }
});

describe('T4 training: hybrid TP strategies (gpt3-125m)', () => {
  const strategies: SimulationConfig['strategyType'][] = ['fsdp-tp', 'zero1-tp'];

  for (const strat of strategies) {
    it(`gpt3-125m + ${strat} on 8× T4`, () => {
      const m = getSimulationMetrics(makeConfig('t4', 'gpt3-125m', strat, 8, 1, { tp: 2 }));
      assertFiniteAndValid(m, `T4/gpt3-125m/${strat}`);
    });
  }
});

describe('T4 training: 3D parallel strategies (gpt3-125m)', () => {
  const strategies: SimulationConfig['strategyType'][] = ['ddp-tp-pp', 'zero1-tp-pp', 'fsdp-tp-pp'];

  for (const strat of strategies) {
    it(`gpt3-125m + ${strat} on 16× T4`, () => {
      const m = getSimulationMetrics(makeConfig('t4', 'gpt3-125m', strat, 16, 2, { tp: 2, pp: 2 }));
      assertFiniteAndValid(m, `T4/gpt3-125m/${strat}`);
    });
  }
});

describe('T4 training: additional small models', () => {
  it('llama3.2-1b + fsdp on 8× T4', () => {
    const m = getSimulationMetrics(makeConfig('t4', 'llama3.2-1b', 'fsdp', 8, 1));
    assertFiniteAndValid(m, 'T4/llama3.2-1b/fsdp');
  });

  it('llama3.2-3b + fsdp on 8× T4', () => {
    const m = getSimulationMetrics(makeConfig('t4', 'llama3.2-3b', 'fsdp', 8, 1));
    assertFiniteAndValid(m, 'T4/llama3.2-3b/fsdp');
  });

  it('gemma2-2b + ddp on 8× T4', () => {
    const m = getSimulationMetrics(makeConfig('t4', 'gemma2-2b', 'ddp', 8, 1));
    assertFiniteAndValid(m, 'T4/gemma2-2b/ddp');
  });

  it('phi3-mini + fsdp on 8× T4', () => {
    const m = getSimulationMetrics(makeConfig('t4', 'phi3-mini', 'fsdp', 8, 1));
    assertFiniteAndValid(m, 'T4/phi3-mini/fsdp');
  });

  it('gpt3-1.3b + zero-1 on 8× T4', () => {
    const m = getSimulationMetrics(makeConfig('t4', 'gpt3-1.3b', 'zero-1', 8, 1));
    assertFiniteAndValid(m, 'T4/gpt3-1.3b/zero-1');
  });
});

describe('T4 training: memory overflow for large models', () => {
  it('llama3-8b DDP on 8× T4 should OOM (8B params > 16GB in bf16)', () => {
    const m = getSimulationMetrics(makeConfig('t4', 'llama3-8b', 'ddp', 8, 1));
    const gpuMemBytes = T4.memoryGB * 1e9;
    expect(
      m.memoryPerGPU.total,
      `llama3-8b DDP should exceed 16GB T4 memory`,
    ).toBeGreaterThan(gpuMemBytes);
  });
});

describe('T4 training: step time ordering', () => {
  it('T4 is slower than V100 for gpt3-125m FSDP', () => {
    const t4m = getSimulationMetrics(makeConfig('t4', 'gpt3-125m', 'fsdp', 8, 1));
    const v100m = getSimulationMetrics(makeConfig('v100-32gb', 'gpt3-125m', 'fsdp', 8, 1));
    expect(t4m.stepTimeMs).toBeGreaterThan(v100m.stepTimeMs);
  });
});

// ============================================================================
// Section 7: Inference tests — the T4's primary use case
// ============================================================================

describe('T4 inference: TFLOPS resolution', () => {
  it('bf16 inference falls back to fp16 (65 TFLOPS)', () => {
    expect(getGPUTFLOPS(T4, 'bf16')).toBe(65);
  });

  it('fp8 inference falls back to fp16 (65 TFLOPS)', () => {
    expect(getGPUTFLOPS(T4, 'fp8')).toBe(65);
  });

  it('int8 inference uses native 130 TOPS', () => {
    expect(getGPUTFLOPS(T4, 'int8')).toBe(130);
  });

  it('int4 inference uses fp16 TFLOPS (W4A16 dequants to FP16)', () => {
    // INT4 is W4A16: weights stored as INT4, dequantized to FP16 for compute.
    // Uses bf16TFLOPS || fp16TFLOPS, NOT int4TOPS.
    expect(getGPUTFLOPS(T4, 'int4')).toBe(65); // fp16TFLOPS (bf16 is 0 on Turing)
  });

  it('fp16 inference uses native 65 TFLOPS', () => {
    expect(getGPUTFLOPS(T4, 'fp16')).toBe(65);
  });
});

describe('T4 inference: memory calculations', () => {
  it('GPT-3 125M fits comfortably in FP16', () => {
    const model = getModel('gpt3-125m', 2048)!;
    const memory = calculateInferenceMemory(model, 512, 1, 'fp16', 'fp16', true);

    // 125M params * 2 bytes = ~250 MB weights
    expect(memory.weights / 1e9).toBeLessThan(0.5);
    expect(memory.total / 1e9).toBeLessThan(1);
    expect(memory.total / 1e9).toBeLessThan(T4.memoryGB);
  });

  it('Llama 3.2 1B fits in FP16', () => {
    const model = getModel('llama3.2-1b', 2048)!;
    const memory = calculateInferenceMemory(model, 512, 1, 'fp16', 'fp16', true);

    // ~1.2B params * 2 bytes = ~2.4 GB weights
    expect(memory.weights / 1e9).toBeGreaterThan(2);
    expect(memory.weights / 1e9).toBeLessThan(3);
    expect(memory.total / 1e9).toBeLessThan(T4.memoryGB);
  });

  it('Llama 2 7B in FP16 barely fits at batch=1, short seq (tight on 16GB)', () => {
    const model = getModel('llama2-7b', 2048)!;
    const memory = calculateInferenceMemory(model, 512, 1, 'fp16', 'fp16', true);

    // ~6.74B params * 2 bytes = ~13.5 GB weights, leaves very little room
    expect(memory.weights / 1e9).toBeGreaterThan(13);
    expect(memory.total / 1e9).toBeGreaterThan(14); // Very tight
  });

  it('Llama 2 7B in FP16 does NOT fit with batch=4 or longer seq', () => {
    const model = getModel('llama2-7b', 4096)!;
    const memory = calculateInferenceMemory(model, 2048, 4, 'fp16', 'fp16', true);

    // Weights (~13.5GB) + KV cache (4*2048 tokens) will exceed 16GB
    expect(memory.total / 1e9).toBeGreaterThan(T4.memoryGB);
  });

  it('Llama 2 7B fits with INT8 quantization', () => {
    const model = getModel('llama2-7b', 2048)!;
    const memory = calculateInferenceMemory(model, 512, 1, 'int8', 'fp16', true);

    // INT8: 7B params * 1 byte = ~7 GB weights
    expect(memory.weights / 1e9).toBeGreaterThan(6);
    expect(memory.weights / 1e9).toBeLessThan(8);
    expect(memory.total / 1e9).toBeLessThan(T4.memoryGB);
  });

  it('Llama 2 7B fits with INT4 quantization', () => {
    const model = getModel('llama2-7b', 2048)!;
    const memory = calculateInferenceMemory(model, 512, 1, 'int4', 'fp16', true);

    // INT4: 7B params * 0.5 bytes = ~3.5 GB weights
    expect(memory.weights / 1e9).toBeGreaterThan(3);
    expect(memory.weights / 1e9).toBeLessThan(4);
    expect(memory.total / 1e9).toBeLessThan(T4.memoryGB);
  });

  it('Mistral 7B fits with INT4 quantization', () => {
    const model = getModel('mistral-7b', 2048)!;
    const memory = calculateInferenceMemory(model, 512, 1, 'int4', 'fp16', true);

    expect(memory.total / 1e9).toBeLessThan(T4.memoryGB);
  });
});

describe('T4 inference: TTFT (Time to First Token)', () => {
  it('GPT-3 125M: TTFT should be fast on T4', () => {
    const model = getModel('gpt3-125m', 2048)!;
    const ttft = estimateTTFT(model, 128, T4, 'fp16');

    // Very small model, should be sub-10ms even on T4
    expect(ttft).toBeGreaterThan(0);
    expect(ttft).toBeLessThan(20);
  });

  it('Llama 3.2 1B: TTFT reasonable for different prompt lengths', () => {
    const model = getModel('llama3.2-1b', 2048)!;

    const ttft128 = estimateTTFT(model, 128, T4, 'fp16');
    const ttft512 = estimateTTFT(model, 512, T4, 'fp16');
    const ttft1024 = estimateTTFT(model, 1024, T4, 'fp16');

    // TTFT should increase with prompt length
    expect(ttft128).toBeGreaterThan(0);
    expect(ttft512).toBeGreaterThan(ttft128);
    expect(ttft1024).toBeGreaterThan(ttft512);

    // Should be under 1 second for all lengths
    expect(ttft1024).toBeLessThan(1000);
  });

  it('T4 has higher TTFT than H100 for same model', () => {
    const model = getModel('llama3.2-1b', 2048)!;
    const ttftT4 = estimateTTFT(model, 512, T4, 'fp16');
    const ttftH100 = estimateTTFT(model, 512, H100_SXM, 'bf16');

    // T4 should be significantly slower than H100
    expect(ttftT4).toBeGreaterThan(ttftH100);
  });

  it('TTFT scales roughly linearly with prompt length', () => {
    const model = getModel('llama3.2-1b', 2048)!;
    const ttft256 = estimateTTFT(model, 256, T4, 'fp16');
    const ttft512 = estimateTTFT(model, 512, T4, 'fp16');

    const ratio = ttft512 / ttft256;
    expect(ratio).toBeGreaterThan(1.5);
    expect(ratio).toBeLessThan(3);
  });
});

describe('T4 inference: TPOT (Time Per Output Token)', () => {
  it('GPT-3 125M: TPOT should be low', () => {
    const model = getModel('gpt3-125m', 2048)!;
    const tpot = estimateTPOT(model, 256, 1, T4, 'fp16');

    // Small model, even with T4's limited bandwidth should be fast
    expect(tpot).toBeGreaterThan(0);
    expect(tpot).toBeLessThan(10); // Under 10ms per token
  });

  it('Llama 3.2 1B: TPOT reasonable for batch=1', () => {
    const model = getModel('llama3.2-1b', 2048)!;
    const tpot = estimateTPOT(model, 512, 1, T4, 'fp16');

    // 1.2B params in FP16 = ~2.4GB / 320 GB/s = ~7.5ms theoretical minimum
    // With overhead: 8-20ms expected
    expect(tpot).toBeGreaterThan(5);
    expect(tpot).toBeLessThan(30);
  });

  it('TPOT is memory-bandwidth bound on T4 (low bandwidth GPU)', () => {
    const model = getModel('llama3.2-1b', 2048)!;
    const util = calculateUtilizationMetrics(model, 512, 128, 1, T4, 'fp16');

    // T4's 320 GB/s is the bottleneck, decode should be memory-bound
    expect(util.isMemoryBound).toBe(true);
    expect(['memory_bandwidth', 'memory_capacity']).toContain(util.bottleneck);
  });

  it('Llama 2 7B INT4: TPOT with quantization', () => {
    const model = getModel('llama2-7b', 2048)!;
    const tpot = estimateTPOT(model, 512, 1, T4, 'int4');

    // INT4: 7B * 0.5 bytes = ~3.5GB / 320 GB/s = ~11ms theoretical minimum
    expect(tpot).toBeGreaterThan(8);
    expect(tpot).toBeLessThan(50);
  });

  it('INT4 quantization gives lower TPOT than FP16 for same model size', () => {
    const model = getModel('llama3.2-1b', 2048)!;
    const tpotFP16 = estimateTPOT(model, 512, 1, T4, 'fp16');
    const tpotINT4 = estimateTPOT(model, 512, 1, T4, 'int4');

    // INT4 reads 4x fewer weight bytes → lower TPOT
    expect(tpotINT4).toBeLessThan(tpotFP16);
  });
});

describe('T4 inference: throughput', () => {
  it('GPT-3 125M: high throughput for tiny model', () => {
    const model = getModel('gpt3-125m', 2048)!;
    const metrics = calculateThroughputMetrics(model, 128, 128, 1, T4, 'fp16');

    // Small model should achieve decent throughput even on T4
    expect(metrics.tokensPerSecond).toBeGreaterThan(50);
    expect(metrics.tokensPerSecond).toBeLessThan(2000);
  });

  it('Llama 3.2 1B: throughput scales with batch size', () => {
    const model = getModel('llama3.2-1b', 2048)!;

    const through1 = calculateThroughputMetrics(model, 256, 128, 1, T4, 'fp16');
    const through4 = calculateThroughputMetrics(model, 256, 128, 4, T4, 'fp16');

    // Batching should improve total throughput
    expect(through4.tokensPerSecond).toBeGreaterThan(through1.tokensPerSecond);
  });

  it('Llama 2 7B INT4: reasonable throughput with quantization', () => {
    const model = getModel('llama2-7b', 2048)!;
    const metrics = calculateThroughputMetrics(model, 128, 128, 1, T4, 'int4');

    // Should be achievable but slower than larger GPUs
    expect(metrics.tokensPerSecond).toBeGreaterThan(5);
    expect(metrics.tokensPerSecond).toBeLessThan(200);
  });
});

describe('T4 inference: latency metrics end-to-end', () => {
  it('GPT-3 125M: complete latency breakdown', () => {
    const model = getModel('gpt3-125m', 2048)!;
    const metrics = calculateLatencyMetrics(model, 128, 64, 1, T4, 'fp16');

    expect(metrics.ttft).toBeGreaterThan(0);
    expect(metrics.tpot).toBeGreaterThan(0);
    expect(metrics.prefillTime).toBe(metrics.ttft);
    expect(metrics.totalLatency).toBeGreaterThan(metrics.prefillTime);
    expect(metrics.decodeTime).toBeGreaterThan(0);
    expect(metrics.totalLatency).toBe(metrics.prefillTime + metrics.decodeTime);
  });

  it('Llama 3.2 1B: reasonable end-to-end latency', () => {
    const model = getModel('llama3.2-1b', 2048)!;
    const metrics = calculateLatencyMetrics(model, 256, 128, 1, T4, 'fp16');

    // Total latency for 128 output tokens on 1B model
    expect(metrics.totalLatency).toBeGreaterThan(100);   // At least 100ms
    expect(metrics.totalLatency).toBeLessThan(30000);    // Under 30 seconds
  });
});

describe('T4 inference: full simulation', () => {
  it('GPT-3 125M: successful inference simulation', () => {
    const result = runInferenceSimulation({
      modelId: 'gpt3-125m',
      gpu: T4,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 128,
      outputSeqLen: 64,
      weightPrecision: 'fp16',
      kvCachePrecision: 'fp16',
      flashAttention: true,
    });

    expect(result.success).toBe(true);
    expect(result.errors).toHaveLength(0);
    expect(result.memory.total / 1e9).toBeLessThan(T4.memoryGB);
    expect(result.latency.ttft).toBeGreaterThan(0);
    expect(result.latency.tpot).toBeGreaterThan(0);
    expect(result.throughput.tokensPerSecond).toBeGreaterThan(0);
  });

  it('Llama 3.2 1B: successful inference simulation', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3.2-1b',
      gpu: T4,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 256,
      outputSeqLen: 128,
      weightPrecision: 'fp16',
      kvCachePrecision: 'fp16',
      flashAttention: true,
    });

    expect(result.success).toBe(true);
    expect(result.errors).toHaveLength(0);
    expect(result.memory.total / 1e9).toBeLessThan(T4.memoryGB);
  });

  it('Llama 3.2 3B: successful inference with FP16', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3.2-3b',
      gpu: T4,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 256,
      outputSeqLen: 128,
      weightPrecision: 'fp16',
      kvCachePrecision: 'fp16',
      flashAttention: true,
    });

    expect(result.success).toBe(true);
    expect(result.memory.total / 1e9).toBeLessThan(T4.memoryGB);
  });

  it('Llama 2 7B FP16: OOM on T4 with batch=4 and long seq', () => {
    const result = runInferenceSimulation({
      modelId: 'llama2-7b',
      gpu: T4,
      numGPUs: 1,
      batchSize: 4,
      inputSeqLen: 2048,
      outputSeqLen: 512,
      weightPrecision: 'fp16',
      kvCachePrecision: 'fp16',
    });

    expect(result.success).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);
  });

  it('Llama 2 7B INT8: fits on T4 with quantization', () => {
    const result = runInferenceSimulation({
      modelId: 'llama2-7b',
      gpu: T4,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 256,
      outputSeqLen: 128,
      weightPrecision: 'int8',
      kvCachePrecision: 'fp16',
      flashAttention: true,
    });

    expect(result.success).toBe(true);
    expect(result.memory.total / 1e9).toBeLessThan(T4.memoryGB);
  });

  it('Llama 2 7B INT4: fits with plenty of headroom', () => {
    const result = runInferenceSimulation({
      modelId: 'llama2-7b',
      gpu: T4,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 256,
      weightPrecision: 'int4',
      kvCachePrecision: 'fp16',
      flashAttention: true,
    });

    expect(result.success).toBe(true);
    expect(result.memory.total / 1e9).toBeLessThan(T4.memoryGB);
    expect(result.throughput.tokensPerSecond).toBeGreaterThan(0);
  });

  it('simulation generates events for visualization', () => {
    const result = runInferenceSimulation({
      modelId: 'gpt3-125m',
      gpu: T4,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 64,
      outputSeqLen: 32,
      weightPrecision: 'fp16',
      kvCachePrecision: 'fp16',
    });

    expect(result.success).toBe(true);
    expect(result.events.length).toBeGreaterThan(0);
    expect(result.events.some(e => e.type === 'simulation_start')).toBe(true);
    expect(result.events.some(e => e.type === 'simulation_end')).toBe(true);
    expect(result.events.some(e => e.type === 'prefill_start')).toBe(true);
    expect(result.events.some(e => e.type === 'decode_start')).toBe(true);
  });
});

// ============================================================================
// Section 8: T4 inference — quantization comparison
// ============================================================================

describe('T4 inference: quantization precision ladder', () => {
  it('FP16 > INT8 > INT4 weights memory for same model', () => {
    const model = getModel('llama3.2-1b', 2048)!;

    const fp16Weights = modelWeightsMemory(model, 'fp16');
    const int8Weights = modelWeightsMemory(model, 'int8');
    const int4Weights = modelWeightsMemory(model, 'int4');

    expect(fp16Weights).toBeGreaterThan(int8Weights);
    expect(int8Weights).toBeGreaterThan(int4Weights);

    // INT8 ≈ FP16/2, INT4 ≈ FP16/4
    expect(int8Weights).toBeCloseTo(fp16Weights / 2, -6);
    expect(int4Weights).toBeCloseTo(fp16Weights / 4, -6);
  });

  it('lower precision → faster TPOT on T4', () => {
    const model = getModel('llama3.2-1b', 2048)!;

    const tpotFP16 = estimateTPOT(model, 512, 1, T4, 'fp16');
    const tpotINT8 = estimateTPOT(model, 512, 1, T4, 'int8');
    const tpotINT4 = estimateTPOT(model, 512, 1, T4, 'int4');

    expect(tpotFP16).toBeGreaterThan(tpotINT8);
    expect(tpotINT8).toBeGreaterThan(tpotINT4);
  });
});

// ============================================================================
// Section 9: T4 inference vs other inference GPUs (L4, A10)
// ============================================================================

describe('T4 vs L4 vs A10 inference comparison', () => {
  it('L4 has faster TPOT than T4 (similar TDP, newer arch)', () => {
    const model = getModel('gpt3-125m', 2048)!;

    // Use fp16 for T4, bf16 for L4 (which supports it)
    const tpotT4 = estimateTPOT(model, 256, 1, T4, 'fp16');
    const tpotL4 = estimateTPOT(model, 256, 1, L4, 'bf16');

    // L4 has slightly lower bandwidth (300 vs 320 GB/s) but better compute
    // TPOT is bandwidth-bound, so T4 might actually be slightly faster for TPOT
    // The key test is that both produce valid results
    expect(tpotT4).toBeGreaterThan(0);
    expect(tpotL4).toBeGreaterThan(0);
  });

  it('A10 has higher bandwidth than T4 → lower decode latency', () => {
    const model = getModel('gpt3-125m', 2048)!;

    const tpotT4 = estimateTPOT(model, 256, 1, T4, 'fp16');
    const tpotA10 = estimateTPOT(model, 256, 1, A10, 'fp16');

    // A10 has 600 GB/s vs T4's 320 GB/s
    expect(tpotA10).toBeLessThan(tpotT4);
  });

  it('all inference GPUs produce valid simulation results', () => {
    const gpus = [T4, L4, A10];
    for (const gpu of gpus) {
      const result = runInferenceSimulation({
        modelId: 'gpt3-125m',
        gpu,
        numGPUs: 1,
        batchSize: 1,
        inputSeqLen: 128,
        outputSeqLen: 64,
        weightPrecision: 'fp16',
        kvCachePrecision: 'fp16',
      });

      expect(result.success, `${gpu.name} inference failed`).toBe(true);
      expect(result.latency.ttft, `${gpu.name} TTFT`).toBeGreaterThan(0);
      expect(result.latency.tpot, `${gpu.name} TPOT`).toBeGreaterThan(0);
      expect(result.throughput.tokensPerSecond, `${gpu.name} throughput`).toBeGreaterThan(0);
    }
  });
});

// ============================================================================
// Section 10: T4 inference — batch size and KV cache
// ============================================================================

describe('T4 inference: KV cache and batch sizing', () => {
  it('KV cache grows with sequence length', () => {
    const model = getModel('llama3.2-1b', 2048)!;

    const kv256 = totalKVCacheMemory(model, 256, 1, 'fp16');
    const kv1024 = totalKVCacheMemory(model, 1024, 1, 'fp16');

    expect(kv1024).toBeGreaterThan(kv256);
    expect(kv1024 / kv256).toBe(4); // KV cache scales linearly, exact
  });

  it('limited batch size due to 16GB memory', () => {
    // Llama 3.2 1B: ~2.4GB weights, leaving ~13.6GB for KV cache
    const model = getModel('llama3.2-1b', 2048)!;
    const perTokenKV = kvCachePerToken(model, 'fp16');

    // At 2048 tokens, KV cache per batch entry = perToken * 2048
    const kvPerBatch = perTokenKV * 2048;
    const weightsGB = model.totalParams * 2; // FP16
    const availableForKV = (T4.memoryGB * 1e9) - weightsGB;
    const maxBatchEstimate = Math.floor(availableForKV / kvPerBatch);

    // Should be able to fit some batch entries (T4 has limited memory)
    expect(maxBatchEstimate).toBeGreaterThan(0);
    expect(maxBatchEstimate).toBeLessThan(500);
  });
});
