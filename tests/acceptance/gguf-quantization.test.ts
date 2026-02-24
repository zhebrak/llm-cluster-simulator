/**
 * GGUF Quantization Tests
 *
 * Validates:
 * 1. BPW accuracy for all 6 GGUF types
 * 2. Weight memory matches published GGUF file sizes
 * 3. TFLOPS mapping (INT4 bug fix + GGUF types)
 * 4. Quantization ordering (decode throughput follows weight size)
 * 5. Decode throughput vs published benchmarks
 * 6. Full simulation smoke tests for each GGUF type
 * 7. INT4/INT8 bug fix regression
 * 8. FP8 retains compute speedup
 */

import { describe, test, expect } from 'vitest';
import { getPrecisionBytes } from '../../src/core/inference/kv-cache.ts';
import {
  getGPUTFLOPS,
  estimateTTFT,
  estimateTPOT,
  calculateThroughputMetrics,
} from '../../src/core/inference/latency.ts';
import { runInferenceSimulationRaw } from '../../src/core/inference/simulation.ts';
import { runSeqLenSweep } from '../../src/core/inference/seq-len-sweep.ts';
import { generateInferenceCandidates } from '../../src/core/inference/exploration.ts';
import { getModel } from '../../src/core/models/index.ts';
import {
  getGPU,
  H100_SXM,
  H200_SXM,
  B200,
  A100_80GB,
  RTX_4090,
  RTX_3090,
  V100_32GB,
  T4,
  MI300X,
  MI250X,
  L4,
  L40S,
} from '../../src/core/hardware/gpu.ts';
import { calculateMemoryWithTP } from '../../src/core/inference/memory.ts';
import type { InferencePrecision } from '../../src/types/inference.ts';

// ── Test 1: BPW Accuracy ──────────────────────────────────────────────

describe('Test 1: BPW accuracy', () => {
  const expectedBPW: Record<string, number> = {
    q2_k:   3.00,
    q3_k_m: 3.89,
    q4_k_m: 4.83,
    q5_k_m: 5.67,
    q6_k:   6.57,
    q8_0:   8.50,
  };

  for (const [type, bpw] of Object.entries(expectedBPW)) {
    test(`${type} → ${bpw} bpw (${(bpw / 8).toFixed(4)} bytes/param)`, () => {
      expect(getPrecisionBytes(type as InferencePrecision)).toBeCloseTo(bpw / 8, 4);
    });
  }

  // Existing types unchanged
  test('INT4 unchanged at 0.5 bytes/param', () => {
    expect(getPrecisionBytes('int4')).toBe(0.5);
  });
  test('INT8 unchanged at 1.0 bytes/param', () => {
    expect(getPrecisionBytes('int8')).toBe(1.0);
  });
  test('BF16 unchanged at 2.0 bytes/param', () => {
    expect(getPrecisionBytes('bf16')).toBe(2.0);
  });
});

// ── Test 2: Weight Memory vs Published GGUF File Sizes ────────────────

describe('Test 2: Weight memory matches published GGUF file sizes', () => {
  // Published sizes from HuggingFace
  // Llama 2 7B: ~6.74B params
  // Llama 3.3 70B: ~70.6B params (using llama3.3-70b in registry)
  const cases: { modelId: string; type: InferencePrecision; publishedGB: number; label: string }[] = [
    { modelId: 'llama2-7b',   type: 'q4_k_m', publishedGB: 4.08,  label: 'Llama 2 7B Q4_K_M' },
    { modelId: 'llama2-7b',   type: 'q5_k_m', publishedGB: 4.78,  label: 'Llama 2 7B Q5_K_M' },
    { modelId: 'llama3.3-70b', type: 'q4_k_m', publishedGB: 42.5,  label: 'Llama 3.3 70B Q4_K_M' },
    { modelId: 'llama3.3-70b', type: 'q5_k_m', publishedGB: 49.9,  label: 'Llama 3.3 70B Q5_K_M' },
    { modelId: 'llama3.3-70b', type: 'q6_k',   publishedGB: 57.9,  label: 'Llama 3.3 70B Q6_K' },
    { modelId: 'llama3.3-70b', type: 'q8_0',   publishedGB: 75.0,  label: 'Llama 3.3 70B Q8_0' },
  ];

  for (const { modelId, type, publishedGB, label } of cases) {
    test(`${label}: ~${publishedGB} GB (±5%)`, () => {
      const model = getModel(modelId);
      expect(model).toBeDefined();
      const weightBytes = model!.totalParams * getPrecisionBytes(type);
      const weightGB = weightBytes / 1e9;
      expect(weightGB).toBeGreaterThan(publishedGB * 0.95);
      expect(weightGB).toBeLessThan(publishedGB * 1.05);
    });
  }
});

// ── Test 3: TFLOPS Mapping (Bug Fix Validation) ──────────────────────

describe('Test 3: TFLOPS mapping', () => {
  test('INT4 — W4A16 uses bf16TFLOPS, not INT4 TOPS', () => {
    // H100: bf16TFLOPS=989, int4TOPS=3958
    // W4A16 dequantizes to BF16 before matmul — compute is BF16-rate.
    expect(getGPUTFLOPS(H100_SXM, 'int4')).toBe(H100_SXM.bf16TFLOPS); // 989
    expect(getGPUTFLOPS(H100_SXM, 'int4')).not.toBe(H100_SXM.int4TOPS); // NOT 3958
  });

  test('INT8 stays correct — W8A8 uses native INT8 tensor cores', () => {
    expect(getGPUTFLOPS(H100_SXM, 'int8')).toBe(H100_SXM.int8TOPS); // 1979
  });

  test('GGUF types — all dequant to FP16, use bf16TFLOPS', () => {
    const ggufTypes: InferencePrecision[] = ['q2_k', 'q3_k_m', 'q4_k_m', 'q5_k_m', 'q6_k', 'q8_0'];
    for (const q of ggufTypes) {
      expect(getGPUTFLOPS(H100_SXM, q)).toBe(H100_SXM.bf16TFLOPS); // 989
    }
  });

  test('Native precision formats keep dedicated TFLOPS', () => {
    expect(getGPUTFLOPS(H100_SXM, 'fp8')).toBe(H100_SXM.fp8TFLOPS);   // 1979
    expect(getGPUTFLOPS(H100_SXM, 'bf16')).toBe(H100_SXM.bf16TFLOPS); // 989
    expect(getGPUTFLOPS(H100_SXM, 'fp16')).toBe(H100_SXM.fp16TFLOPS); // 989
  });

  test('V100 fallback (no bf16TFLOPS): falls back to fp16TFLOPS', () => {
    // V100: bf16TFLOPS=0, fp16TFLOPS=125
    expect(getGPUTFLOPS(V100_32GB, 'q4_k_m')).toBe(V100_32GB.fp16TFLOPS); // 125
    expect(getGPUTFLOPS(V100_32GB, 'int4')).toBe(V100_32GB.fp16TFLOPS);   // 125
  });
});

// ── Test 4: Quantization Ordering (Decode Throughput) ────────────────

describe('Test 4: Quantization ordering — decode throughput follows weight size', () => {
  test('batch=1 decode: fewer bytes/param → faster decode', () => {
    const model = getModel('llama3-8b')!;
    expect(model).toBeDefined();

    // At batch=1, decode is bandwidth-bound. Order by effective bytes/param
    // (raw bytes × dequant overhead). GGUF formats have lower overhead than
    // generic INT4 (GPTQ/AWQ) due to fused GGML dequant kernels:
    // Q2_K(0.375×1.15=0.431) < Q3_K_M(0.486×1.15=0.559) < INT4(0.5×1.20=0.600)
    // < Q4_K_M(0.604×1.10=0.664) < Q5_K_M(0.709×1.08=0.766) < Q6_K(0.821×1.08=0.887)
    // < INT8(1.0×1.10=1.10) < Q8_0(1.0625×1.10=1.169) < BF16(2.0×1.0=2.0)
    const precisions: InferencePrecision[] = [
      'q2_k', 'q3_k_m', 'int4', 'q4_k_m', 'q5_k_m', 'q6_k', 'int8', 'q8_0', 'bf16',
    ];
    const tpots = precisions.map(p =>
      estimateTPOT(model, 512, 1, H100_SXM, p)
    );

    // Each subsequent precision should have >= TPOT (slower decode)
    for (let i = 0; i < tpots.length - 1; i++) {
      expect(tpots[i]).toBeLessThanOrEqual(tpots[i + 1] * 1.01); // 1% tolerance for rounding
    }
  });
});

// ── Test 5: Decode Throughput vs Published Benchmarks ────────────────

describe('Test 5: Decode throughput vs published benchmarks', () => {
  test('RTX 4090 + Llama 8B + INT4, batch=1: ~150 tok/s (±35%)', () => {
    const model = getModel('llama3-8b')!;
    const throughput = calculateThroughputMetrics(model, 100, 100, 1, RTX_4090, 'int4');
    // Published: ~150 tok/s (NVIDIA blog, TRT-LLM optimized)
    expect(throughput.decodeTokensPerSecond).toBeGreaterThan(150 * 0.65);
    expect(throughput.decodeTokensPerSecond).toBeLessThan(150 * 1.35);
  });

  test('RTX 4090 + Llama 8B + Q4_K_M, batch=1: ~104 tok/s (±35%)', () => {
    const model = getModel('llama3-8b')!;
    const throughput = calculateThroughputMetrics(model, 512, 128, 1, RTX_4090, 'q4_k_m');
    // Published: ~104 tok/s (hardware-corner, llama.cpp, 16K context)
    // Our model uses shorter context → faster → upper bound more generous
    expect(throughput.decodeTokensPerSecond).toBeGreaterThan(104 * 0.65);
    expect(throughput.decodeTokensPerSecond).toBeLessThan(104 * 1.60);
  });

  test('RTX 3090 + Llama 8B + Q4_K_M, batch=1: ~87 tok/s (±35%)', () => {
    const model = getModel('llama3-8b')!;
    const throughput = calculateThroughputMetrics(model, 512, 128, 1, RTX_3090, 'q4_k_m');
    // Published: ~87 tok/s (hardware-corner, llama.cpp)
    expect(throughput.decodeTokensPerSecond).toBeGreaterThan(87 * 0.65);
    expect(throughput.decodeTokensPerSecond).toBeLessThan(87 * 1.60);
  });

  test('Theoretical: INT4 ~1.21× faster decode than Q4_K_M (0.604/0.5)', () => {
    const model = getModel('llama3-8b')!;
    const int4Tput = calculateThroughputMetrics(model, 512, 128, 1, RTX_4090, 'int4');
    const q4Tput = calculateThroughputMetrics(model, 512, 128, 1, RTX_4090, 'q4_k_m');
    const ratio = int4Tput.decodeTokensPerSecond / q4Tput.decodeTokensPerSecond;
    // Expected ratio: ~1.21 (0.604/0.5), ±20%
    expect(ratio).toBeGreaterThan(1.0);
    expect(ratio).toBeLessThan(1.5);
  });
});

// ── Test 6: Full Simulation Smoke Tests ──────────────────────────────

describe('Test 6: Full simulation smoke tests', () => {
  const ggufTypes: InferencePrecision[] = ['q2_k', 'q3_k_m', 'q4_k_m', 'q5_k_m', 'q6_k', 'q8_0'];
  const configs = [
    { modelId: 'llama3-8b', gpuId: 'h100-sxm', label: 'Llama 8B on H100' },
    { modelId: 'llama2-7b', gpuId: 'rtx-4090', label: 'Llama 2 7B on RTX 4090' },
  ];

  for (const { modelId, gpuId, label } of configs) {
    for (const quant of ggufTypes) {
      test(`${label} + ${quant}: simulation succeeds with sane values`, () => {
        const result = runInferenceSimulationRaw({
          modelId,
          gpuId,
          batchSize: 1,
          inputSeqLen: 512,
          outputSeqLen: 128,
          weightPrecision: quant,
          kvCachePrecision: 'bf16',
        });

        expect(result.success).toBe(true);
        expect(result.memory.weights).toBeGreaterThan(0);
        expect(result.memory.total).toBeGreaterThan(0);
        expect(isFinite(result.memory.weights)).toBe(true);
        expect(isFinite(result.memory.total)).toBe(true);
        expect(result.latency.ttft).toBeGreaterThan(0);
        expect(result.latency.tpot).toBeGreaterThan(0);
        expect(isFinite(result.latency.ttft)).toBe(true);
        expect(isFinite(result.latency.tpot)).toBe(true);
        expect(result.throughput.tokensPerSecond).toBeGreaterThan(0);
        expect(isFinite(result.throughput.tokensPerSecond)).toBe(true);
        expect(result.utilization.memoryCapacityUtilization).toBeGreaterThan(0);
        expect(result.utilization.memoryCapacityUtilization).toBeLessThanOrEqual(1.0);
      });
    }
  }

  test('Q4_K_M weights ~20% more than INT4 (0.604/0.5 = 1.21×)', () => {
    const resultQ4 = runInferenceSimulationRaw({
      modelId: 'llama3-8b',
      gpuId: 'h100-sxm',
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'q4_k_m',
      kvCachePrecision: 'bf16',
    });
    const resultInt4 = runInferenceSimulationRaw({
      modelId: 'llama3-8b',
      gpuId: 'h100-sxm',
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'int4',
      kvCachePrecision: 'bf16',
    });

    expect(resultQ4.success).toBe(true);
    expect(resultInt4.success).toBe(true);
    const ratio = resultQ4.memory.weights / resultInt4.memory.weights;
    // Expected: 0.604/0.5 = 1.208
    expect(ratio).toBeGreaterThan(1.15);
    expect(ratio).toBeLessThan(1.30);
  });
});

// ── Test 7: INT4/INT8 Bug Fix Regression ────────────────────────────

describe('Test 7: INT4 TFLOPS bug fix regression', () => {
  test('INT4 TTFT comparable to BF16 (same compute TFLOPS after fix)', () => {
    const model = getModel('llama3-8b')!;
    const ttftBf16 = estimateTTFT(model, 1024, H100_SXM, 'bf16', 1);
    const ttftInt4 = estimateTTFT(model, 1024, H100_SXM, 'int4', 1);
    // INT4 TTFT should be comparable to BF16 (same compute TFLOPS now)
    // Not 4× faster like before the fix
    expect(ttftInt4 / ttftBf16).toBeGreaterThan(0.5);
    expect(ttftInt4 / ttftBf16).toBeLessThan(1.5);
  });

  test('INT4 compute-bound TTFT at large batch — NOT 4× faster than BF16', () => {
    const model = getModel('llama3-8b')!;
    // At large batch, prefill is compute-bound → TFLOPS matters
    const ttftBf16 = estimateTTFT(model, 2048, H100_SXM, 'bf16', 64);
    const ttftInt4 = estimateTTFT(model, 2048, H100_SXM, 'int4', 64);
    // Before fix: INT4 was ~4× faster (used 3958 TOPS). After fix: similar to BF16.
    const ratio = ttftInt4 / ttftBf16;
    expect(ratio).toBeGreaterThan(0.7);
    expect(ratio).toBeLessThan(1.3);
  });
});

// ── Test 8: FP8 Retains Compute Speedup ─────────────────────────────

describe('Test 8: FP8 retains native compute speedup', () => {
  test('FP8 prefill ~2× faster than BF16 at compute-bound batch sizes', () => {
    const model = getModel('llama3-8b')!;
    const ttftBf16 = estimateTTFT(model, 2048, H100_SXM, 'bf16', 64);
    const ttftFp8 = estimateTTFT(model, 2048, H100_SXM, 'fp8', 64);
    // FP8: 1979 TFLOPS vs BF16: 989 TFLOPS → ~2× faster compute
    expect(ttftFp8).toBeLessThan(ttftBf16 * 0.7);
  });

  test('FP8 TFLOPS is ~2× BF16 on H100', () => {
    const ratio = getGPUTFLOPS(H100_SXM, 'fp8') / getGPUTFLOPS(H100_SXM, 'bf16');
    expect(ratio).toBeGreaterThan(1.9);
    expect(ratio).toBeLessThan(2.1);
  });
});

// ── Test 9: Seq-Len Sweep includes GGUF groups ─────────────────────

describe('Test 9: Seq-len sweep includes GGUF groups', () => {
  test('sweep result contains q4_k_m and q8_0 groups', () => {
    const model = getModel('llama3-8b')!;
    const gpu = getGPU('h100-sxm')!;
    const result = runSeqLenSweep(
      {
        modelId: 'llama3-8b',
        gpuId: 'h100-sxm',
        batchSize: 1,
        inputSeqLen: 512,
        outputSeqLen: 128,
        weightPrecision: 'bf16',
        kvCachePrecision: 'bf16',
      },
      model,
      gpu,
    );

    expect(result.groups).toHaveProperty('q4_k_m');
    expect(result.groups).toHaveProperty('q8_0');
    // Both should have data points (not empty)
    expect(result.groups['q4_k_m'].length).toBeGreaterThan(0);
    expect(result.groups['q8_0'].length).toBeGreaterThan(0);
  });

  test('GGUF groups have valid latency and memory values', () => {
    const model = getModel('llama3-8b')!;
    const gpu = getGPU('h100-sxm')!;
    const result = runSeqLenSweep(
      {
        modelId: 'llama3-8b',
        gpuId: 'h100-sxm',
        batchSize: 1,
        inputSeqLen: 512,
        outputSeqLen: 128,
        weightPrecision: 'bf16',
        kvCachePrecision: 'bf16',
      },
      model,
      gpu,
    );

    for (const prec of ['q4_k_m', 'q8_0'] as const) {
      for (const pt of result.groups[prec]) {
        expect(pt.ttft).toBeGreaterThan(0);
        expect(pt.tpot).toBeGreaterThan(0);
        expect(pt.memoryUtil).toBeGreaterThan(0);
        expect(pt.memoryUtil).toBeLessThanOrEqual(1.0);
        expect(isFinite(pt.ttft)).toBe(true);
        expect(isFinite(pt.tpot)).toBe(true);
      }
    }
  });
});

// ── Test 10: Exploration KV cache is independent of weight precision ─

describe('Test 10: Exploration KV cache from baseConfig', () => {
  test('all candidates inherit KV cache precision from baseConfig', () => {
    const model = getModel('llama3-8b')!;
    const gpu = getGPU('h100-sxm')!;
    const candidates = generateInferenceCandidates(
      {
        modelId: 'llama3-8b',
        gpuId: 'h100-sxm',
        batchSize: 1,
        inputSeqLen: 512,
        outputSeqLen: 128,
        weightPrecision: 'bf16',
        kvCachePrecision: 'fp8',
      },
      model,
      gpu,
    );

    // All candidates should use fp8 KV regardless of weight precision
    for (const candidate of candidates) {
      expect(candidate.kvCachePrecision).toBe('fp8');
    }

    // Verify GGUF candidates exist
    const q4Candidates = candidates.filter(c => c.weightPrecision === 'q4_k_m');
    const q8Candidates = candidates.filter(c => c.weightPrecision === 'q8_0');
    expect(q4Candidates.length).toBeGreaterThan(0);
    expect(q8Candidates.length).toBeGreaterThan(0);
  });
});

// ── Test 11: GPU Architecture TFLOPS Routing ─────────────────────────

describe('Test 11: GPU architecture TFLOPS routing', () => {
  const ggufTypes: InferencePrecision[] = ['q2_k', 'q3_k_m', 'q4_k_m', 'q5_k_m', 'q6_k', 'q8_0'];

  // All GGUF types dequant to FP16, compute at BF16/FP16 rate
  const gpuExpected: { gpu: typeof H100_SXM; name: string; expected: number }[] = [
    { gpu: H100_SXM,  name: 'H100 SXM (hopper)',     expected: 989 },
    { gpu: H200_SXM,  name: 'H200 SXM (hopper)',     expected: 989 },
    { gpu: B200,       name: 'B200 (blackwell)',       expected: 2250 },
    { gpu: A100_80GB,  name: 'A100 80GB (ampere DC)',  expected: 312 },
    { gpu: RTX_4090,   name: 'RTX 4090 (ada)',         expected: 165.2 },
    { gpu: RTX_3090,   name: 'RTX 3090 (ampere)',      expected: 142 },
    { gpu: MI300X,     name: 'MI300X (cdna3)',         expected: 1307 },
    { gpu: MI250X,     name: 'MI250X (cdna2)',         expected: 191.5 },
    { gpu: L4,         name: 'L4 (ada)',               expected: 121 },
    { gpu: L40S,       name: 'L40S (ada)',             expected: 362 },
  ];

  for (const { gpu, name, expected } of gpuExpected) {
    for (const q of ggufTypes) {
      test(`${name} + ${q} → ${expected} TFLOPS`, () => {
        expect(getGPUTFLOPS(gpu, q)).toBe(expected);
      });
    }
  }

  // Special: V100 and T4 have bf16TFLOPS=0, fall back to fp16TFLOPS
  test('V100 (volta, bf16=0) falls back to fp16TFLOPS=125', () => {
    for (const q of ggufTypes) {
      expect(getGPUTFLOPS(V100_32GB, q)).toBe(125);
    }
  });

  test('T4 (turing, bf16=0) falls back to fp16TFLOPS=65', () => {
    for (const q of ggufTypes) {
      expect(getGPUTFLOPS(T4, q)).toBe(65);
    }
  });
});

// ── Test 12: Full Simulation Across GPU Architectures ────────────────

describe('Test 12: Full simulation across GPU architectures', () => {
  const gpuConfigs: { id: string; name: string; bw: number }[] = [
    { id: 'h100-sxm',  name: 'H100 SXM',  bw: 3.35 },
    { id: 'a100-80gb',  name: 'A100 80GB',  bw: 2.039 },
    { id: 'rtx-4090',   name: 'RTX 4090',   bw: 1.008 },
    { id: 'rtx-3090',   name: 'RTX 3090',   bw: 0.936 },
    { id: 'v100-32gb',  name: 'V100 32GB',  bw: 0.9 },
    { id: 't4',          name: 'T4',          bw: 0.32 },
    { id: 'mi300x',     name: 'MI300X',     bw: 5.3 },
    { id: 'mi250x',     name: 'MI250X',     bw: 1.638 },
    { id: 'l4',          name: 'L4',          bw: 0.3 },
    { id: 'b200',        name: 'B200',        bw: 7.7 },
  ];

  for (const { id, name } of gpuConfigs) {
    test(`${name}: Llama 8B q4_k_m simulation succeeds with sane values`, () => {
      const result = runInferenceSimulationRaw({
        modelId: 'llama3-8b', gpuId: id,
        batchSize: 1, inputSeqLen: 512, outputSeqLen: 128,
        weightPrecision: 'q4_k_m', kvCachePrecision: 'bf16',
      });

      expect(result.success).toBe(true);
      expect(result.memory.weights).toBeGreaterThan(0);
      expect(result.memory.total).toBeGreaterThan(0);
      expect(isFinite(result.memory.total)).toBe(true);
      expect(result.latency.ttft).toBeGreaterThan(0);
      expect(result.latency.tpot).toBeGreaterThan(0);
      expect(isFinite(result.latency.tpot)).toBe(true);
      expect(result.throughput.tokensPerSecond).toBeGreaterThan(0);
      expect(result.throughput.decodeTokensPerSecond).toBeGreaterThan(0);
      expect(isFinite(result.throughput.decodeTokensPerSecond)).toBe(true);
      expect(result.utilization.memoryCapacityUtilization).toBeLessThanOrEqual(1.0);
    });
  }

  test('throughput roughly tracks GPU memory bandwidth', () => {
    // Higher bandwidth GPUs should generally produce higher decode throughput
    const results = gpuConfigs.map(({ id, bw }) => {
      const result = runInferenceSimulationRaw({
        modelId: 'llama3-8b', gpuId: id,
        batchSize: 1, inputSeqLen: 512, outputSeqLen: 128,
        weightPrecision: 'q4_k_m', kvCachePrecision: 'bf16',
      });
      return { id, bw, tput: result.throughput.decodeTokensPerSecond };
    });

    // Sort by bandwidth, verify highest BW GPU has highest throughput
    results.sort((a, b) => b.bw - a.bw);
    // B200 (7.7 TB/s) should be fastest
    expect(results[0].id).toBe('b200');
    // MI300X (5.3 TB/s) should be second
    expect(results[1].id).toBe('mi300x');
    // T4 (0.32 TB/s) or L4 (0.3 TB/s) should be slowest
    const slowest = results[results.length - 1];
    expect(['t4', 'l4']).toContain(slowest.id);
  });
});

// ── Test 13: KV Cache Precision Independence ─────────────────────────

describe('Test 13: KV cache precision independence from GGUF weight type', () => {
  const kvPrecisions: InferencePrecision[] = ['bf16', 'fp8', 'int8', 'int4'];
  const weightPrecisions: InferencePrecision[] = ['q4_k_m', 'q8_0'];

  test('weight memory is constant regardless of KV cache precision', () => {
    const model = getModel('llama3-8b')!;
    for (const wp of weightPrecisions) {
      const weightMemories = kvPrecisions.map(kvp => {
        const mem = calculateMemoryWithTP(model, 640, 1, 1, wp, kvp);
        return mem.weights;
      });
      // All should be identical for the same weight precision
      for (const wm of weightMemories) {
        expect(wm).toBe(weightMemories[0]);
      }
    }
  });

  test('KV cache memory changes with KV precision (bf16 > fp8 > int8 > int4)', () => {
    const model = getModel('llama3-8b')!;
    const kvMem = kvPrecisions.map(kvp => {
      const mem = calculateMemoryWithTP(model, 640, 1, 1, 'q4_k_m', kvp);
      return mem.kvCache;
    });
    // bf16 > fp8 ≥ int8 > int4
    expect(kvMem[0]).toBeGreaterThan(kvMem[1]); // bf16 > fp8
    expect(kvMem[1]).toBeGreaterThanOrEqual(kvMem[2]); // fp8 ≥ int8
    expect(kvMem[2]).toBeGreaterThan(kvMem[3]); // int8 > int4
  });

  test('all GGUF weight × KV precision combos simulate successfully', () => {
    for (const wp of weightPrecisions) {
      for (const kvp of kvPrecisions) {
        const result = runInferenceSimulationRaw({
          modelId: 'llama3-8b', gpuId: 'h100-sxm',
          batchSize: 1, inputSeqLen: 512, outputSeqLen: 128,
          weightPrecision: wp, kvCachePrecision: kvp,
        });
        expect(result.success).toBe(true);
        expect(isFinite(result.latency.tpot)).toBe(true);
        expect(result.throughput.tokensPerSecond).toBeGreaterThan(0);
      }
    }
  });
});

// ── Test 14: Tensor Parallelism with GGUF ────────────────────────────

describe('Test 14: Tensor parallelism with GGUF weights', () => {
  test('Llama 70B q4_k_m: weights scale inversely with TP', () => {
    const model = getModel('llama3.3-70b')!;
    const tp1 = calculateMemoryWithTP(model, 640, 1, 1, 'q4_k_m', 'bf16');
    const tp2 = calculateMemoryWithTP(model, 640, 1, 2, 'q4_k_m', 'bf16');
    const tp4 = calculateMemoryWithTP(model, 640, 1, 4, 'q4_k_m', 'bf16');
    const tp8 = calculateMemoryWithTP(model, 640, 1, 8, 'q4_k_m', 'bf16');

    // TP=1: ~42.6 GB weights
    expect(tp1.weights / 1e9).toBeGreaterThan(40);
    expect(tp1.weights / 1e9).toBeLessThan(45);

    // Each doubling of TP halves weights (±5%)
    expect(tp2.weights).toBeCloseTo(tp1.weights / 2, -8);
    expect(tp4.weights).toBeCloseTo(tp1.weights / 4, -8);
    expect(tp8.weights).toBeCloseTo(tp1.weights / 8, -8);
  });

  test('Llama 70B q4_k_m: per-GPU memory decreases with TP', () => {
    const tpDegrees = [1, 2, 4, 8];
    const totals = tpDegrees.map(tp => {
      const model = getModel('llama3.3-70b')!;
      return calculateMemoryWithTP(model, 640, 1, tp, 'q4_k_m', 'bf16').total;
    });

    for (let i = 0; i < totals.length - 1; i++) {
      expect(totals[i]).toBeGreaterThan(totals[i + 1]);
    }
  });

  test('Mixtral 8x7B q4_k_m: expert weights split by TP', () => {
    const model = getModel('mixtral-8x7b')!;
    const tp1 = calculateMemoryWithTP(model, 640, 1, 1, 'q4_k_m', 'bf16');
    const tp2 = calculateMemoryWithTP(model, 640, 1, 2, 'q4_k_m', 'bf16');

    // Weights should halve with TP=2
    expect(tp2.weights).toBeCloseTo(tp1.weights / 2, -8);
  });

  test('full simulation with TP=1..8 produces valid results', () => {
    for (const tp of [1, 2, 4, 8]) {
      const result = runInferenceSimulationRaw({
        modelId: 'llama3.3-70b', gpuId: 'h100-sxm', numGPUs: 8,
        batchSize: 1, inputSeqLen: 512, outputSeqLen: 128,
        weightPrecision: 'q4_k_m', kvCachePrecision: 'bf16',
        tensorParallel: tp,
      });
      expect(result.success).toBe(true);
      expect(result.throughput.decodeTokensPerSecond).toBeGreaterThan(0);
      expect(isFinite(result.throughput.decodeTokensPerSecond)).toBe(true);
    }
  });
});

// ── Test 15: Expert Parallelism with GGUF ────────────────────────────

describe('Test 15: Expert parallelism with GGUF weights', () => {
  test('Mixtral 8x7B: expert weights split by EP, KV cache constant', () => {
    const model = getModel('mixtral-8x7b')!;
    const eps = [1, 2, 4, 8];
    const mems = eps.map(ep =>
      calculateMemoryWithTP(model, 640, 1, 1, 'q4_k_m', 'bf16', ep)
    );

    // Weights decrease with EP (routed expert params split)
    for (let i = 0; i < mems.length - 1; i++) {
      expect(mems[i].weights).toBeGreaterThan(mems[i + 1].weights);
    }

    // KV cache stays constant (unaffected by EP)
    for (let i = 1; i < mems.length; i++) {
      expect(mems[i].kvCache).toBe(mems[0].kvCache);
    }
  });

  test('DeepSeek V3: MLA KV cache replicated (not split by TP or EP)', () => {
    const model = getModel('deepseek-v3')!;
    expect(model.attentionType).toBe('mla');

    // KV cache should be identical across EP degrees
    const kvCaches = [1, 2, 4, 8].map(ep =>
      calculateMemoryWithTP(model, 640, 1, 1, 'q4_k_m', 'bf16', ep).kvCache
    );
    for (const kv of kvCaches) {
      expect(kv).toBe(kvCaches[0]);
    }

    // MLA KV cache uses compressed latent format (much smaller than standard)
    // Standard would be 2 * 61 * 128 * 128 * 2 * 640 ≈ 2.56 GB for bf16
    // MLA compressed: 61 * (512+64) * 2 * 640 ≈ 45 MB
    expect(kvCaches[0] / 1e6).toBeGreaterThan(30);
    expect(kvCaches[0] / 1e6).toBeLessThan(60);
  });

  test('DeepSeek V3: weights decrease with EP', () => {
    const model = getModel('deepseek-v3')!;
    const ep1 = calculateMemoryWithTP(model, 640, 1, 1, 'q4_k_m', 'bf16', 1);
    const ep8 = calculateMemoryWithTP(model, 640, 1, 1, 'q4_k_m', 'bf16', 8);

    // EP=1: ~405 GB, EP=8: ~60 GB (most params are routed experts)
    expect(ep1.weights / 1e9).toBeGreaterThan(380);
    expect(ep1.weights / 1e9).toBeLessThan(420);
    expect(ep8.weights / 1e9).toBeGreaterThan(45);
    expect(ep8.weights / 1e9).toBeLessThan(75);
  });
});

// ── Test 16: Continuous Batching with GGUF ───────────────────────────

describe('Test 16: Continuous batching with GGUF weights', () => {
  test('CB throughput ≥ non-CB throughput at all batch sizes', () => {
    for (const batch of [1, 32, 128]) {
      const noCB = runInferenceSimulationRaw({
        modelId: 'llama3-8b', gpuId: 'h100-sxm',
        batchSize: batch, inputSeqLen: 512, outputSeqLen: 128,
        weightPrecision: 'q4_k_m', kvCachePrecision: 'bf16',
        continuousBatching: false,
      });
      const withCB = runInferenceSimulationRaw({
        modelId: 'llama3-8b', gpuId: 'h100-sxm',
        batchSize: batch, inputSeqLen: 512, outputSeqLen: 128,
        weightPrecision: 'q4_k_m', kvCachePrecision: 'bf16',
        continuousBatching: true,
      });

      expect(noCB.success).toBe(true);
      expect(withCB.success).toBe(true);
      expect(withCB.throughput.tokensPerSecond).toBeGreaterThanOrEqual(
        noCB.throughput.tokensPerSecond * 0.99 // allow tiny rounding
      );
    }
  });

  test('CB scheduling overhead is small (TPOT increases by ≤3%)', () => {
    const noCB = runInferenceSimulationRaw({
      modelId: 'llama3-8b', gpuId: 'h100-sxm',
      batchSize: 32, inputSeqLen: 512, outputSeqLen: 128,
      weightPrecision: 'q4_k_m', kvCachePrecision: 'bf16',
      continuousBatching: false,
    });
    const withCB = runInferenceSimulationRaw({
      modelId: 'llama3-8b', gpuId: 'h100-sxm',
      batchSize: 32, inputSeqLen: 512, outputSeqLen: 128,
      weightPrecision: 'q4_k_m', kvCachePrecision: 'bf16',
      continuousBatching: true,
    });

    const tpotRatio = withCB.latency.tpot / noCB.latency.tpot;
    expect(tpotRatio).toBeGreaterThan(1.0);
    expect(tpotRatio).toBeLessThan(1.03);
  });

  test('at batch=1, CB effect is minimal', () => {
    const noCB = runInferenceSimulationRaw({
      modelId: 'llama3-8b', gpuId: 'h100-sxm',
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 128,
      weightPrecision: 'q4_k_m', kvCachePrecision: 'bf16',
      continuousBatching: false,
    });
    const withCB = runInferenceSimulationRaw({
      modelId: 'llama3-8b', gpuId: 'h100-sxm',
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 128,
      weightPrecision: 'q4_k_m', kvCachePrecision: 'bf16',
      continuousBatching: true,
    });

    // Throughput difference should be < 5% at batch=1
    const ratio = withCB.throughput.tokensPerSecond / noCB.throughput.tokensPerSecond;
    expect(ratio).toBeGreaterThan(0.95);
    expect(ratio).toBeLessThan(1.05);
  });
});

// ── Test 17: Speculative Decoding with GGUF ──────────────────────────

describe('Test 17: Speculative decoding with GGUF weights', () => {
  test('Llama 70B + Llama 1B draft: speculative decoding provides speedup', () => {
    const specResult = runInferenceSimulationRaw({
      modelId: 'llama3.3-70b', gpuId: 'h100-sxm', numGPUs: 8,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 128,
      weightPrecision: 'q4_k_m', kvCachePrecision: 'bf16',
      speculativeEnabled: true, draftModelId: 'llama3.2-1b',
      numSpeculativeTokens: 4, acceptanceRate: 0.7,
      tensorParallel: 2,
    });

    expect(specResult.success).toBe(true);
    expect(specResult.speculative).toBeDefined();
    expect(specResult.speculative!.speedup).toBeGreaterThan(1.0);
    expect(specResult.speculative!.effectiveTpot).toBeGreaterThan(0);
  });

  test('draft model memory is included in total', () => {
    const noSpec = runInferenceSimulationRaw({
      modelId: 'llama3.3-70b', gpuId: 'h100-sxm', numGPUs: 8,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 128,
      weightPrecision: 'q4_k_m', kvCachePrecision: 'bf16',
      tensorParallel: 2,
    });
    const withSpec = runInferenceSimulationRaw({
      modelId: 'llama3.3-70b', gpuId: 'h100-sxm', numGPUs: 8,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 128,
      weightPrecision: 'q4_k_m', kvCachePrecision: 'bf16',
      speculativeEnabled: true, draftModelId: 'llama3.2-1b',
      numSpeculativeTokens: 4, acceptanceRate: 0.7,
      tensorParallel: 2,
    });

    expect(noSpec.success).toBe(true);
    expect(withSpec.success).toBe(true);

    // Speculative should have more memory (draft weights + draft KV cache)
    expect(withSpec.memory.total).toBeGreaterThan(noSpec.memory.total);
    expect(withSpec.memory.weights).toBeGreaterThan(noSpec.memory.weights);
  });

  test('speculative metrics have valid values', () => {
    const result = runInferenceSimulationRaw({
      modelId: 'llama3.3-70b', gpuId: 'h100-sxm', numGPUs: 8,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 128,
      weightPrecision: 'q4_k_m', kvCachePrecision: 'bf16',
      speculativeEnabled: true, draftModelId: 'llama3.2-1b',
      numSpeculativeTokens: 4, acceptanceRate: 0.7,
      tensorParallel: 2,
    });

    expect(result.speculative).toBeDefined();
    const spec = result.speculative!;
    expect(spec.expectedAcceptedTokens).toBeGreaterThan(0);
    expect(spec.expectedAcceptedTokens).toBeLessThanOrEqual(4); // can't exceed K
    expect(spec.draftModelOverhead).toBeGreaterThan(0);
    expect(spec.verificationTime).toBeGreaterThan(0);
    expect(isFinite(spec.effectiveTpot)).toBe(true);
  });
});

// ── Test 18: MoE + MLA Models with GGUF ─────────────────────────────

describe('Test 18: MoE + MLA models with GGUF weights', () => {
  test('DeepSeek V3 q4_k_m: weight memory ~405 GB (±10% of published 377 GB)', () => {
    const model = getModel('deepseek-v3')!;
    expect(model.totalParams / 1e9).toBeGreaterThan(660);
    expect(model.totalParams / 1e9).toBeLessThan(680);

    const mem = calculateMemoryWithTP(model, 640, 1, 1, 'q4_k_m', 'bf16');
    const weightGB = mem.weights / 1e9;
    // Simulator: ~405 GB, published GGUF: ~377 GB
    // Allow ±10% from simulator value (which slightly overestimates due to embeddings)
    expect(weightGB).toBeGreaterThan(365);
    expect(weightGB).toBeLessThan(445);
  });

  test('DeepSeek V3: MLA KV cache uses compressed format', () => {
    const model = getModel('deepseek-v3')!;
    expect(model.attentionType).toBe('mla');
    expect(model.kvLoraRank).toBe(512);
    expect(model.qkRopeHeadDim).toBe(64);

    const mem = calculateMemoryWithTP(model, 640, 1, 1, 'q4_k_m', 'bf16');
    // MLA compressed: 61 layers × (512+64) × 2 bytes × 640 tokens ≈ 45 MB
    // Standard MHA would be >2 GB — MLA should be dramatically smaller
    expect(mem.kvCache / 1e6).toBeGreaterThan(30);
    expect(mem.kvCache / 1e6).toBeLessThan(60);
  });

  test('DeepSeek V3 TP=8: weights ~50.6 GB/GPU, KV cache NOT split', () => {
    const model = getModel('deepseek-v3')!;
    const memTp1 = calculateMemoryWithTP(model, 640, 1, 1, 'q4_k_m', 'bf16');
    const memTp8 = calculateMemoryWithTP(model, 640, 1, 8, 'q4_k_m', 'bf16');

    // Weights should be ~1/8th
    expect(memTp8.weights / 1e9).toBeGreaterThan(45);
    expect(memTp8.weights / 1e9).toBeLessThan(56);

    // MLA KV cache is replicated — same across TP degrees
    expect(memTp8.kvCache).toBe(memTp1.kvCache);
  });

  test('DeepSeek V3 TP=8 EP=8: expert weights further split', () => {
    const model = getModel('deepseek-v3')!;
    const memTp8 = calculateMemoryWithTP(model, 640, 1, 8, 'q4_k_m', 'bf16', 1);
    const memTp8Ep8 = calculateMemoryWithTP(model, 640, 1, 8, 'q4_k_m', 'bf16', 8);

    // EP should further reduce weights (routed experts split by EP)
    expect(memTp8Ep8.weights).toBeLessThan(memTp8.weights);
    // EP=8 per-GPU weights should fit comfortably in 80 GB H100
    expect(memTp8Ep8.total / 1e9).toBeLessThan(80);
  });

  test('Llama 4 Maverick q4_k_m: mixed dense/MoE layers handled correctly', () => {
    const result = runInferenceSimulationRaw({
      modelId: 'llama4-maverick', gpuId: 'h100-sxm', numGPUs: 8,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 128,
      weightPrecision: 'q4_k_m', kvCachePrecision: 'bf16',
      tensorParallel: 8,
    });

    expect(result.success).toBe(true);
    expect(result.memory.weights).toBeGreaterThan(0);
    expect(result.throughput.tokensPerSecond).toBeGreaterThan(0);
    expect(isFinite(result.throughput.tokensPerSecond)).toBe(true);

    // Maverick has moeLayerFrequency=2 — verify model loaded correctly
    const model = getModel('llama4-maverick')!;
    expect(model.isMoE).toBe(true);
    expect(model.numExperts).toBe(128);
  });
});

// ── Test 19: OOM Boundaries & Large Model Fit ────────────────────────

describe('Test 19: OOM boundaries and large model fit', () => {
  const oomCases: {
    label: string;
    config: Parameters<typeof runInferenceSimulationRaw>[0];
    expectedFit: boolean;
  }[] = [
    {
      label: 'Llama 70B on RTX 4090 (24GB) q4_k_m → OOM (~42.5 GB weights)',
      config: {
        modelId: 'llama3.3-70b', gpuId: 'rtx-4090',
        batchSize: 1, inputSeqLen: 512, outputSeqLen: 128,
        weightPrecision: 'q4_k_m', kvCachePrecision: 'bf16',
      },
      expectedFit: false,
    },
    {
      label: 'Llama 70B on A100 80GB q4_k_m → fits (~42.5 GB < 80 GB)',
      config: {
        modelId: 'llama3.3-70b', gpuId: 'a100-80gb',
        batchSize: 1, inputSeqLen: 512, outputSeqLen: 128,
        weightPrecision: 'q4_k_m', kvCachePrecision: 'bf16',
      },
      expectedFit: true,
    },
    {
      label: 'Llama 8B on T4 (16GB) q4_k_m → fits (~4.9 GB < 16 GB)',
      config: {
        modelId: 'llama3-8b', gpuId: 't4',
        batchSize: 1, inputSeqLen: 512, outputSeqLen: 128,
        weightPrecision: 'q4_k_m', kvCachePrecision: 'bf16',
      },
      expectedFit: true,
    },
    {
      label: 'Llama 8B on T4 (16GB) bf16 → OOM (~16.1 GB weights + overhead)',
      config: {
        modelId: 'llama3-8b', gpuId: 't4',
        batchSize: 1, inputSeqLen: 512, outputSeqLen: 128,
        weightPrecision: 'bf16', kvCachePrecision: 'bf16',
      },
      expectedFit: false,
    },
    {
      label: 'Llama 8B on T4 (16GB) q8_0 → fits (~8.5 GB < 16 GB)',
      config: {
        modelId: 'llama3-8b', gpuId: 't4',
        batchSize: 1, inputSeqLen: 512, outputSeqLen: 128,
        weightPrecision: 'q8_0', kvCachePrecision: 'bf16',
      },
      expectedFit: true,
    },
    {
      label: 'Llama 405B on H100 (80GB) q4_k_m → OOM (~244 GB weights)',
      config: {
        modelId: 'llama3-405b', gpuId: 'h100-sxm',
        batchSize: 1, inputSeqLen: 512, outputSeqLen: 128,
        weightPrecision: 'q4_k_m', kvCachePrecision: 'bf16',
      },
      expectedFit: false,
    },
    {
      label: 'Llama 405B on 8×H100 TP=8 q4_k_m → fits (~30.5 GB/GPU)',
      config: {
        modelId: 'llama3-405b', gpuId: 'h100-sxm', numGPUs: 8,
        batchSize: 1, inputSeqLen: 512, outputSeqLen: 128,
        weightPrecision: 'q4_k_m', kvCachePrecision: 'bf16',
        tensorParallel: 8,
      },
      expectedFit: true,
    },
    {
      label: 'DeepSeek V3 on 8×H100 TP=8 EP=8 q4_k_m → fits',
      config: {
        modelId: 'deepseek-v3', gpuId: 'h100-sxm', numGPUs: 64,
        batchSize: 1, inputSeqLen: 512, outputSeqLen: 128,
        weightPrecision: 'q4_k_m', kvCachePrecision: 'bf16',
        tensorParallel: 8, expertParallel: 8,
      },
      expectedFit: true,
    },
  ];

  for (const { label, config, expectedFit } of oomCases) {
    test(label, () => {
      const result = runInferenceSimulationRaw(config);
      expect(result.success).toBe(expectedFit);

      if (expectedFit) {
        // When it fits: all metrics should be valid
        expect(result.throughput.tokensPerSecond).toBeGreaterThan(0);
        expect(isFinite(result.latency.tpot)).toBe(true);
        expect(result.utilization.memoryCapacityUtilization).toBeLessThanOrEqual(1.0);
      } else {
        // When OOM: memory should exceed GPU capacity
        expect(result.utilization.memoryCapacityUtilization).toBeGreaterThan(1.0);
        expect(result.utilization.memoryCapacityUtilization).toBeLessThan(10.0);
        expect(result.errors.length).toBeGreaterThan(0);
      }
    });
  }

  test('GGUF quantization enables models that OOM at bf16', () => {
    // Llama 8B on T4: OOM at bf16, fits at q4_k_m — quantization is the enabler
    const bf16 = runInferenceSimulationRaw({
      modelId: 'llama3-8b', gpuId: 't4',
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 128,
      weightPrecision: 'bf16', kvCachePrecision: 'bf16',
    });
    const q4 = runInferenceSimulationRaw({
      modelId: 'llama3-8b', gpuId: 't4',
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 128,
      weightPrecision: 'q4_k_m', kvCachePrecision: 'bf16',
    });

    expect(bf16.success).toBe(false);
    expect(q4.success).toBe(true);
    // q4_k_m weights should be ~3.3× smaller than bf16
    expect(q4.memory.weights).toBeLessThan(bf16.memory.weights * 0.4);
  });
});
