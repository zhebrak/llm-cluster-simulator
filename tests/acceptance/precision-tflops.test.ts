/**
 * Precision-Dependent TFLOPS Validation Tests
 *
 * Validates that different precisions (fp32, bf16, fp8, fp4, etc.) correctly
 * resolve to the right TFLOPS values across all GPU types, and that this
 * flows through training step time, MFU/HFU, and inference latency.
 */

import { describe, it, expect } from 'vitest';
import {
  ALL_GPUS,
  getEffectiveTFLOPS,
  getMemoryBandwidthScaling,
  H100_SXM,
  A100_80GB,
  B200,
  V100_32GB,
} from '../../src/core/hardware/gpu.ts';
import { getGPUTFLOPS, estimateTTFT, estimateTPOT, calculateThroughputMetrics } from '../../src/core/inference/latency.ts';
import { getModel } from '../../src/core/models/index.ts';
import { createCluster, createNode } from '../../src/core/hardware/topology.ts';
import { SimulationEngine } from '../../src/core/simulation/index.ts';
import { assertValidEngine } from '../helpers/validated-metrics.ts';
import type { GPUSpec } from '../../src/types/hardware.ts';
import type { InferencePrecision } from '../../src/types/inference.ts';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Pick a model that fits on the given GPU (small model for small GPUs). */
function getModelForGPU(gpu: GPUSpec): string {
  return gpu.memoryGB < 40 ? 'gpt3-125m' : 'llama3-8b';
}

/** Run a training simulation and return metrics. */
function runTraining(
  gpuId: string,
  precision: 'fp32' | 'tf32' | 'fp16' | 'bf16' | 'fp8' | 'fp4',
) {
  const gpu = ALL_GPUS[gpuId];
  if (!gpu) throw new Error(`Unknown GPU: ${gpuId}`);

  const modelId = getModelForGPU(gpu);
  const model = getModel(modelId, 2048)!;

  // Build a single-node cluster from the GPU
  const node = createNode(gpu, 8);
  const cluster = createCluster(node, 1, 'single-node');

  const engine = new SimulationEngine();
  engine.configure({
    modelSpec: model,
    clusterConfig: cluster,
    globalBatchSize: 64,
    microBatchSize: 2,
    sequenceLength: 2048,
    strategyType: 'fsdp',
    strategyConfig: { tp: 1, pp: 1 },
    activationCheckpointing: true,
    mixedPrecision: precision,
  });
  assertValidEngine(engine);
  return engine.simulate();
}

const ALL_GPU_IDS = Object.keys(ALL_GPUS);
const TRAINING_PRECISIONS = ['fp32', 'tf32', 'fp16', 'bf16', 'fp8', 'fp4'] as const;
const INFERENCE_PRECISIONS: InferencePrecision[] = ['fp32', 'fp16', 'bf16', 'fp8', 'fp4', 'int8', 'int4'];

// ===================================================================
// Section 1: Training TFLOPS resolution
// ===================================================================
describe('Training TFLOPS resolution', () => {
  it('all GPUs × all training precisions → positive, finite values', () => {
    for (const gpuId of ALL_GPU_IDS) {
      const gpu = ALL_GPUS[gpuId];
      for (const prec of TRAINING_PRECISIONS) {
        const tflops = getEffectiveTFLOPS(gpu, prec);
        expect(tflops, `${gpuId}/${prec}`).toBeGreaterThan(0);
        expect(Number.isFinite(tflops), `${gpuId}/${prec} not finite`).toBe(true);
      }
    }
  });

  it('FP8 on A100 falls back to BF16 (312 TFLOPS)', () => {
    expect(A100_80GB.fp8TFLOPS).toBe(0); // no native FP8
    expect(getEffectiveTFLOPS(A100_80GB, 'fp8')).toBe(312); // falls back to bf16
  });

  it('FP4 on H100 falls back to FP8 (1979 TFLOPS)', () => {
    expect(H100_SXM.fp4TFLOPS).toBe(0); // no native FP4
    expect(getEffectiveTFLOPS(H100_SXM, 'fp4')).toBe(1979); // fp4→fp8
  });

  it('BF16 on V100 falls back to FP16 (125 TFLOPS)', () => {
    expect(V100_32GB.bf16TFLOPS).toBe(0); // no native BF16
    expect(getEffectiveTFLOPS(V100_32GB, 'bf16')).toBe(125); // falls back to fp16
  });

  it('FP4 on B200 uses native FP4 (9000 TFLOPS)', () => {
    expect(B200.fp4TFLOPS).toBe(9000);
    expect(getEffectiveTFLOPS(B200, 'fp4')).toBe(9000);
  });
});

// ===================================================================
// Section 2: Inference TFLOPS resolution
// ===================================================================
describe('Inference TFLOPS resolution', () => {
  it('all GPUs × all inference precisions → positive, finite values', () => {
    for (const gpuId of ALL_GPU_IDS) {
      const gpu = ALL_GPUS[gpuId];
      for (const prec of INFERENCE_PRECISIONS) {
        const tflops = getGPUTFLOPS(gpu, prec);
        expect(tflops, `${gpuId}/${prec}`).toBeGreaterThan(0);
        expect(Number.isFinite(tflops), `${gpuId}/${prec} not finite`).toBe(true);
      }
    }
  });

  it('FP4 on H100: inference uses INT4 TOPS (3958), training uses FP8 (1979)', () => {
    // Inference fallback chain: fp4→int4→int8→bf16→fp16
    const infTFLOPS = getGPUTFLOPS(H100_SXM, 'fp4');
    // Training fallback chain: fp4→fp8→bf16→fp16
    const trainTFLOPS = getEffectiveTFLOPS(H100_SXM, 'fp4');
    expect(infTFLOPS).toBe(3958);   // int4 TOPS
    expect(trainTFLOPS).toBe(1979); // fp8 TFLOPS
    expect(infTFLOPS).not.toBe(trainTFLOPS); // asymmetry
  });

  it('INT8 on V100 falls back to FP16 (125)', () => {
    expect(V100_32GB.int8TOPS).toBe(0);
    expect(getGPUTFLOPS(V100_32GB, 'int8')).toBe(125); // → bf16(0) → fp16(125)
  });

  it('getGPUTFLOPS vs getEffectiveTFLOPS for FP4 on H100', () => {
    // These two functions have different fallback chains for FP4
    const inferenceVal = getGPUTFLOPS(H100_SXM, 'fp4');
    const trainingVal = getEffectiveTFLOPS(H100_SXM, 'fp4');
    // Inference: fp4→int4(3958), Training: fp4→fp8(1979)
    expect(inferenceVal).toBeGreaterThan(trainingVal);
  });
});

// ===================================================================
// Section 3: Training step time varies with precision
// ===================================================================
describe('Training step time varies with precision', () => {
  it('H100 SXM: FP8 step time < BF16 step time', () => {
    const fp8 = runTraining('h100-sxm', 'fp8');
    const bf16 = runTraining('h100-sxm', 'bf16');
    expect(fp8.stepTimeMs).toBeLessThan(bf16.stepTimeMs);
    // FP8 is ~2x peak TFLOPS, but comm overhead means ratio is 1.3x-1.9x
    const ratio = bf16.stepTimeMs / fp8.stepTimeMs;
    expect(ratio).toBeGreaterThan(1.1);
    expect(ratio).toBeLessThan(2.2);
  });

  it('H100 SXM: FP32 step time >> BF16 step time', () => {
    const fp32 = runTraining('h100-sxm', 'fp32');
    const bf16 = runTraining('h100-sxm', 'bf16');
    // FP32=67 vs BF16=989 TFLOPS → massive difference
    expect(fp32.stepTimeMs).toBeGreaterThan(bf16.stepTimeMs * 3);
  });

  it('A100 80GB: FP8 and BF16 produce identical step times (fallback)', () => {
    const fp8 = runTraining('a100-80gb', 'fp8');
    const bf16 = runTraining('a100-80gb', 'bf16');
    // A100 has no FP8 hardware, so FP8 falls back to BF16=312 TFLOPS
    // Step times should be very close (within 1% due to floating point)
    const ratio = fp8.stepTimeMs / bf16.stepTimeMs;
    expect(ratio).toBeCloseTo(1.0, 1);
  });

  it('B200: FP4 < FP8 < BF16 step time ordering', () => {
    const fp4 = runTraining('b200', 'fp4');
    const fp8 = runTraining('b200', 'fp8');
    const bf16 = runTraining('b200', 'bf16');
    expect(fp4.stepTimeMs).toBeLessThan(fp8.stepTimeMs);
    expect(fp8.stepTimeMs).toBeLessThan(bf16.stepTimeMs);
  });

  it('H100 SXM: FP8 MFU higher than BF16 MFU (denominator is BF16 peak, compute is faster)', () => {
    const fp8 = runTraining('h100-sxm', 'fp8');
    const bf16 = runTraining('h100-sxm', 'bf16');
    // MFU denominator is always BF16 peak (industry convention).
    // FP8 compute is ~1.67× faster (Amdahl's law), so MFU is higher.
    const ratio = fp8.mfu / bf16.mfu;
    expect(ratio).toBeGreaterThan(1.1);
    expect(ratio).toBeLessThan(1.7);
  });

  it('all GPUs × BF16: positive step time, MFU ∈ (0, 1]', () => {
    for (const gpuId of ALL_GPU_IDS) {
      const metrics = runTraining(gpuId, 'bf16');
      expect(metrics.stepTimeMs, `${gpuId} stepTime`).toBeGreaterThan(0);
      expect(Number.isFinite(metrics.stepTimeMs), `${gpuId} stepTime not finite`).toBe(true);
      expect(metrics.mfu, `${gpuId} MFU`).toBeGreaterThan(0);
      expect(metrics.mfu, `${gpuId} MFU`).toBeLessThanOrEqual(1.0);
    }
  });

  it('all GPUs × FP8: positive step time, no NaN (graceful fallback)', () => {
    for (const gpuId of ALL_GPU_IDS) {
      const metrics = runTraining(gpuId, 'fp8');
      expect(metrics.stepTimeMs, `${gpuId} stepTime`).toBeGreaterThan(0);
      expect(Number.isFinite(metrics.stepTimeMs), `${gpuId} stepTime not finite`).toBe(true);
      expect(metrics.mfu, `${gpuId} MFU`).toBeGreaterThan(0);
      expect(metrics.mfu, `${gpuId} MFU`).not.toBeNaN();
    }
  });
});

// ===================================================================
// Section 4: Inference latency varies with precision
// ===================================================================
describe('Inference latency varies with precision', () => {
  const model8b = getModel('llama3-8b', 2048)!;

  it('H100 SXM: FP8 TTFT < BF16 TTFT (prefill is compute-bound)', () => {
    const fp8Ttft = estimateTTFT(model8b, 1024, H100_SXM, 'fp8');
    const bf16Ttft = estimateTTFT(model8b, 1024, H100_SXM, 'bf16');
    expect(fp8Ttft).toBeLessThan(bf16Ttft);
  });

  it('H100 SXM: FP8 TPOT < BF16 TPOT (fewer bytes to read)', () => {
    const fp8Tpot = estimateTPOT(model8b, 1024, 1, H100_SXM, 'fp8');
    const bf16Tpot = estimateTPOT(model8b, 1024, 1, H100_SXM, 'bf16');
    expect(fp8Tpot).toBeLessThan(bf16Tpot);
  });

  it('A100 80GB: FP8 TTFT ≈ BF16 TTFT (fallback), but FP8 TPOT < BF16 TPOT', () => {
    // TTFT uses getGPUTFLOPS which for A100 FP8 falls back to BF16 (312)
    const fp8Ttft = estimateTTFT(model8b, 1024, A100_80GB, 'fp8');
    const bf16Ttft = estimateTTFT(model8b, 1024, A100_80GB, 'bf16');
    // TTFT should be similar since compute TFLOPS are the same after fallback
    const ttftRatio = fp8Ttft / bf16Ttft;
    expect(ttftRatio).toBeCloseTo(1.0, 1); // within ±5%

    // TPOT uses getPrecisionBytes: FP8=1 byte vs BF16=2 bytes → fewer bytes to read
    const fp8Tpot = estimateTPOT(model8b, 1024, 1, A100_80GB, 'fp8');
    const bf16Tpot = estimateTPOT(model8b, 1024, 1, A100_80GB, 'bf16');
    expect(fp8Tpot).toBeLessThan(bf16Tpot);
  });

  it('INT4 inference on H100: TTFT comparable to BF16 (W4A16 dequants to FP16)', () => {
    // INT4 is W4A16: weights stored as INT4, dequantized to FP16 for compute.
    // Uses bf16TFLOPS (989), NOT int4TOPS (3958). TTFT should be similar to BF16.
    const int4Ttft = estimateTTFT(model8b, 1024, H100_SXM, 'int4');
    const bf16Ttft = estimateTTFT(model8b, 1024, H100_SXM, 'bf16');
    // Same compute TFLOPS → similar TTFT (small difference from weight loading)
    const ratio = int4Ttft / bf16Ttft;
    expect(ratio).toBeGreaterThan(0.5);
    expect(ratio).toBeLessThan(1.5);
  });

  it('all GPUs × all precisions: positive, finite TTFT and TPOT', () => {
    const model125m = getModel('gpt3-125m', 2048)!;
    for (const gpuId of ALL_GPU_IDS) {
      const gpu = ALL_GPUS[gpuId];
      for (const prec of INFERENCE_PRECISIONS) {
        const ttft = estimateTTFT(model125m, 512, gpu, prec);
        const tpot = estimateTPOT(model125m, 512, 1, gpu, prec);
        expect(ttft, `TTFT ${gpuId}/${prec}`).toBeGreaterThan(0);
        expect(Number.isFinite(ttft), `TTFT ${gpuId}/${prec} not finite`).toBe(true);
        expect(tpot, `TPOT ${gpuId}/${prec}`).toBeGreaterThan(0);
        expect(Number.isFinite(tpot), `TPOT ${gpuId}/${prec} not finite`).toBe(true);
      }
    }
  });

  it('full inference simulation: FP8 throughput > BF16 throughput on H100', () => {
    const fp8 = calculateThroughputMetrics(model8b, 1024, 256, 8, H100_SXM, 'fp8');
    const bf16 = calculateThroughputMetrics(model8b, 1024, 256, 8, H100_SXM, 'bf16');
    expect(fp8.tokensPerSecond).toBeGreaterThan(bf16.tokensPerSecond);
  });
});

// ===================================================================
// Section 5: Memory bandwidth scaling with precision
// ===================================================================
describe('Memory bandwidth scaling with precision', () => {
  it('H100 SXM BF16 → scaling ≈ 1.0 (reference)', () => {
    const scaling = getMemoryBandwidthScaling(H100_SXM, 'bf16');
    // H100 SXM BF16 is the reference point (OI ≈ 295)
    expect(scaling).toBeCloseTo(1.0, 1);
  });

  it('H100 SXM FP8 → scaling < 1.0 (higher OI)', () => {
    const scaling = getMemoryBandwidthScaling(H100_SXM, 'fp8');
    // FP8 = 1979 TFLOPS / 3.35 TB/s → OI ≈ 591, higher than reference
    // More compute-heavy → more time waiting on memory-bound ops → scaling < 1.0
    expect(scaling).toBeLessThan(1.0);
    expect(scaling).toBeGreaterThan(0.8);
  });

  it('V100 BF16 → scaling > 1.0 (lower OI)', () => {
    const scaling = getMemoryBandwidthScaling(V100_32GB, 'bf16');
    // V100: bf16 falls back to fp16=125 TFLOPS, BW=0.9 TB/s → OI ≈ 139
    // Lower OI than reference → relatively more bandwidth per FLOP → scaling > 1.0
    expect(scaling).toBeGreaterThan(1.0);
  });
});

// ===================================================================
// Section 6: Cross-precision MFU consistency
// ===================================================================
describe('Cross-precision MFU consistency', () => {
  it('MFU stable across BF16/FP16, higher for FP8 (BF16 denominator convention)', () => {
    // Test on H100 with bf16, fp8, and fp16
    const bf16 = runTraining('h100-sxm', 'bf16');
    const fp8 = runTraining('h100-sxm', 'fp8');
    const fp16 = runTraining('h100-sxm', 'fp16');

    // BF16 and FP16 should be close (same TFLOPS on H100)
    const bf16fp16Ratio = bf16.mfu / fp16.mfu;
    expect(bf16fp16Ratio).toBeGreaterThan(0.9);
    expect(bf16fp16Ratio).toBeLessThan(1.1);

    // FP8 MFU is higher: denominator is BF16 peak, compute is ~1.67× faster
    expect(fp8.mfu).toBeGreaterThan(bf16.mfu);
    expect(fp8.mfu).toBeLessThan(bf16.mfu * 1.7);
  });

  it('HFU/MFU ratio ≈ 1.33 with activation checkpointing', () => {
    // With checkpointing: HFU uses 8PD, MFU uses 6PD → ratio = 8/6 ≈ 1.333
    const metrics = runTraining('h100-sxm', 'bf16');
    const ratio = metrics.hfu / metrics.mfu;
    expect(ratio).toBeCloseTo(8 / 6, 1); // 1.333 ± 0.05
  });

  it('FP32 training on H100: MFU still valid ∈ (0, 1]', () => {
    // FP32 = 67 TFLOPS — much lower peak, but MFU should still be valid
    const metrics = runTraining('h100-sxm', 'fp32');
    expect(metrics.mfu).toBeGreaterThan(0);
    expect(metrics.mfu).toBeLessThanOrEqual(1.0);
    expect(Number.isFinite(metrics.mfu)).toBe(true);
  });
});
