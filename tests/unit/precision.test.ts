/**
 * Comprehensive precision tests
 * Tests that precision calculations are accurate across all GPU types and precisions
 */

import { describe, it, expect } from 'vitest';
import { DTYPE_BYTES } from '../../src/types/base.ts';
import {
  ALL_GPUS,
  getEffectiveTFLOPS,
  getTrainingTFLOPS,
  gpuSupportsPrecision,
  getPrecisionFallbackWarning,
  H100_SXM,
  A100_80GB,
  B200,
} from '../../src/core/hardware/gpu.ts';
import { calculateOptimizerMemory } from '../../src/core/strategies/base.ts';

describe('Precision Byte Sizes', () => {
  it('should have correct byte sizes for all precisions', () => {
    expect(DTYPE_BYTES.fp32).toBe(4);
    expect(DTYPE_BYTES.tf32).toBe(4);  // Stored in FP32 containers
    expect(DTYPE_BYTES.fp16).toBe(2);
    expect(DTYPE_BYTES.bf16).toBe(2);
    expect(DTYPE_BYTES.fp8).toBe(1);
    expect(DTYPE_BYTES.fp4).toBe(0.5);
  });

  it('should calculate memory correctly for each precision', () => {
    const numParams = 1e9; // 1B params

    expect(numParams * DTYPE_BYTES.fp32).toBe(4e9);   // 4GB
    expect(numParams * DTYPE_BYTES.bf16).toBe(2e9);   // 2GB
    expect(numParams * DTYPE_BYTES.fp8).toBe(1e9);    // 1GB
    expect(numParams * DTYPE_BYTES.fp4).toBe(0.5e9);  // 0.5GB
  });
});

describe('GPU Precision Support', () => {
  describe('A100 (Ampere)', () => {
    const a100 = ALL_GPUS['a100-80gb'];

    it('should support FP32, TF32, FP16, BF16', () => {
      expect(gpuSupportsPrecision(a100, 'fp32')).toBe(true);
      expect(gpuSupportsPrecision(a100, 'tf32')).toBe(true);
      expect(gpuSupportsPrecision(a100, 'fp16')).toBe(true);
      expect(gpuSupportsPrecision(a100, 'bf16')).toBe(true);
    });

    it('should NOT support FP8 or FP4', () => {
      expect(gpuSupportsPrecision(a100, 'fp8')).toBe(false);
      expect(gpuSupportsPrecision(a100, 'fp4')).toBe(false);
    });

    it('should support INT8 and INT4', () => {
      expect(gpuSupportsPrecision(a100, 'int8')).toBe(true);
      expect(gpuSupportsPrecision(a100, 'int4')).toBe(true);
    });

    it('should have correct TFLOPS values', () => {
      expect(a100.fp32TFLOPS).toBe(19.5);
      expect(a100.tf32TFLOPS).toBe(156);
      expect(a100.fp16TFLOPS).toBe(312);
      expect(a100.bf16TFLOPS).toBe(312);
      expect(a100.fp8TFLOPS).toBe(0);
      expect(a100.int8TOPS).toBe(624);
      expect(a100.int4TOPS).toBe(1248);
    });
  });

  describe('H100 (Hopper)', () => {
    const h100 = ALL_GPUS['h100-sxm'];

    it('should support all precisions except FP4', () => {
      expect(gpuSupportsPrecision(h100, 'fp32')).toBe(true);
      expect(gpuSupportsPrecision(h100, 'tf32')).toBe(true);
      expect(gpuSupportsPrecision(h100, 'fp16')).toBe(true);
      expect(gpuSupportsPrecision(h100, 'bf16')).toBe(true);
      expect(gpuSupportsPrecision(h100, 'fp8')).toBe(true);
      expect(gpuSupportsPrecision(h100, 'fp4')).toBe(false);
    });

    it('should have correct dense TFLOPS values', () => {
      expect(h100.fp32TFLOPS).toBe(67);
      expect(h100.tf32TFLOPS).toBe(989);
      expect(h100.bf16TFLOPS).toBe(989);   // Dense BF16 (sparse: 1979)
      expect(h100.fp8TFLOPS).toBe(1979);   // Dense FP8 (sparse: 3958)
      expect(h100.int8TOPS).toBe(1979);    // Dense INT8 (sparse: 3958)
    });

    it('should have FP8 ≈ 2x BF16 TFLOPS', () => {
      // Real dense specs are ~989.4 and ~1978.9, integer rounding causes off-by-one
      expect(h100.fp8TFLOPS / h100.bf16TFLOPS).toBeCloseTo(2, 2);
    });
  });

  describe('B200 (Blackwell)', () => {
    const b200 = ALL_GPUS['b200'];

    it('should support ALL precisions including FP4', () => {
      expect(gpuSupportsPrecision(b200, 'fp32')).toBe(true);
      expect(gpuSupportsPrecision(b200, 'tf32')).toBe(true);
      expect(gpuSupportsPrecision(b200, 'fp16')).toBe(true);
      expect(gpuSupportsPrecision(b200, 'bf16')).toBe(true);
      expect(gpuSupportsPrecision(b200, 'fp8')).toBe(true);
      expect(gpuSupportsPrecision(b200, 'fp4')).toBe(true);
      expect(gpuSupportsPrecision(b200, 'int8')).toBe(true);
      expect(gpuSupportsPrecision(b200, 'int4')).toBe(true);
    });

    it('should have correct dense TFLOPS values (HGX 1000W)', () => {
      expect(b200.fp32TFLOPS).toBe(75);
      expect(b200.tf32TFLOPS).toBe(1100);    // Dense TF32 (sparse: 2200)
      expect(b200.bf16TFLOPS).toBe(2250);    // Dense BF16 (sparse: 4500)
      expect(b200.fp8TFLOPS).toBe(4500);     // Dense FP8 (sparse: 9000)
      expect(b200.fp4TFLOPS).toBe(9000);     // Dense FP4 (sparse: 18000)
    });

    it('should have FP4 = 2x FP8 TFLOPS', () => {
      expect(b200.fp4TFLOPS).toBe(b200.fp8TFLOPS * 2);
    });

    it('should have FP8 = 2x BF16 TFLOPS', () => {
      expect(b200.fp8TFLOPS).toBe(b200.bf16TFLOPS * 2);
    });
  });

  describe('V100 (Volta)', () => {
    const v100 = ALL_GPUS['v100-32gb'];

    it('should only support FP32 and FP16', () => {
      expect(gpuSupportsPrecision(v100, 'fp32')).toBe(true);
      expect(gpuSupportsPrecision(v100, 'fp16')).toBe(true);
      expect(gpuSupportsPrecision(v100, 'bf16')).toBe(false);
      expect(gpuSupportsPrecision(v100, 'tf32')).toBe(false);
      expect(gpuSupportsPrecision(v100, 'fp8')).toBe(false);
      expect(gpuSupportsPrecision(v100, 'fp4')).toBe(false);
    });
  });
});

describe('getEffectiveTFLOPS', () => {
  describe('Returns correct values for native precision', () => {
    it('should return exact values when precision is natively supported', () => {
      const h100 = ALL_GPUS['h100-sxm'];

      expect(getEffectiveTFLOPS(h100, 'fp32')).toBe(67);
      expect(getEffectiveTFLOPS(h100, 'tf32')).toBe(989);
      expect(getEffectiveTFLOPS(h100, 'bf16')).toBe(989);
      expect(getEffectiveTFLOPS(h100, 'fp8')).toBe(1979);
    });
  });

  describe('Falls back correctly for unsupported precisions', () => {
    it('should fallback FP8 to BF16 on A100', () => {
      const a100 = ALL_GPUS['a100-80gb'];
      // A100 doesn't support FP8, should fallback to BF16
      expect(getEffectiveTFLOPS(a100, 'fp8')).toBe(a100.bf16TFLOPS);
    });

    it('should fallback FP4 to FP8 on H100', () => {
      const h100 = ALL_GPUS['h100-sxm'];
      // H100 doesn't support FP4, should fallback to FP8
      expect(getEffectiveTFLOPS(h100, 'fp4')).toBe(h100.fp8TFLOPS);
    });

    it('should fallback BF16 to FP16 on V100', () => {
      const v100 = ALL_GPUS['v100-32gb'];
      // V100 doesn't support BF16, should fallback to FP16
      expect(getEffectiveTFLOPS(v100, 'bf16')).toBe(v100.fp16TFLOPS);
    });

    it('should fallback TF32 to FP32 on V100', () => {
      const v100 = ALL_GPUS['v100-32gb'];
      // V100 doesn't support TF32, should fallback to FP32
      expect(getEffectiveTFLOPS(v100, 'tf32')).toBe(v100.fp32TFLOPS);
    });
  });
});

describe('Precision TFLOPS Scaling', () => {
  it('should follow expected scaling pattern: FP32 < TF32 <= BF16 < FP8 < FP4', () => {
    const b200 = ALL_GPUS['b200'];

    expect(b200.fp32TFLOPS).toBeLessThan(b200.tf32TFLOPS);
    expect(b200.tf32TFLOPS).toBeLessThanOrEqual(b200.bf16TFLOPS);
    expect(b200.bf16TFLOPS).toBeLessThan(b200.fp8TFLOPS);
    expect(b200.fp8TFLOPS).toBeLessThan(b200.fp4TFLOPS);
  });

  it('should have 2x scaling between precision tiers (dense)', () => {
    const h100 = ALL_GPUS['h100-sxm'];

    // Dense: TF32 = BF16 for Hopper (both use same tensor core throughput)
    expect(h100.tf32TFLOPS).toBe(h100.bf16TFLOPS);

    // FP8 ≈ 2x BF16 (dense, integer rounding causes off-by-one)
    expect(h100.fp8TFLOPS / h100.bf16TFLOPS).toBeCloseTo(2, 2);
  });
});

describe('Memory Calculation Accuracy', () => {
  it('should calculate 70B model memory correctly at different precisions', () => {
    const params = 70e9; // 70B parameters

    const fp32Memory = params * DTYPE_BYTES.fp32;
    const bf16Memory = params * DTYPE_BYTES.bf16;
    const fp8Memory = params * DTYPE_BYTES.fp8;
    const fp4Memory = params * DTYPE_BYTES.fp4;

    // 70B params
    expect(fp32Memory / 1e9).toBe(280);  // 280 GB
    expect(bf16Memory / 1e9).toBe(140);  // 140 GB
    expect(fp8Memory / 1e9).toBe(70);    // 70 GB
    expect(fp4Memory / 1e9).toBe(35);    // 35 GB
  });

  it('should calculate optimizer memory correctly for BF16 (AdamW: master + m1 + m2 = 12)', () => {
    const params = 7e9; // 7B parameters

    // BF16: needs fp32 master copy + 2 optimizer states = 12 bytes/param
    const optimizerMemory = calculateOptimizerMemory(params, 'adamw', 'bf16');
    expect(optimizerMemory / 1e9).toBe(84); // 84 GB
  });

  it('should calculate optimizer memory correctly for FP32 (AdamW: m1 + m2 = 8, no master)', () => {
    const params = 7e9; // 7B parameters

    // FP32: working weights ARE fp32, no master copy needed = 8 bytes/param
    const optimizerMemory = calculateOptimizerMemory(params, 'adamw', 'fp32');
    expect(optimizerMemory / 1e9).toBe(56); // 56 GB
  });
});

describe('All GPUs have valid TFLOPS values', () => {
  Object.entries(ALL_GPUS).forEach(([_gpuId, gpu]) => {
    describe(gpu.name, () => {
      it('should have positive FP32 TFLOPS', () => {
        expect(gpu.fp32TFLOPS).toBeGreaterThan(0);
      });

      it('should have FP16 >= FP32 TFLOPS', () => {
        expect(gpu.fp16TFLOPS).toBeGreaterThanOrEqual(gpu.fp32TFLOPS);
      });

      it('should have consistent BF16/FP16 values when both supported', () => {
        if (gpu.bf16TFLOPS > 0 && gpu.fp16TFLOPS > 0) {
          // BF16 and FP16 typically have same TFLOPS on tensor cores
          expect(gpu.bf16TFLOPS).toBe(gpu.fp16TFLOPS);
        }
      });

      it('should have FP8 >= BF16 when FP8 is supported', () => {
        if (gpu.fp8TFLOPS > 0) {
          expect(gpu.fp8TFLOPS).toBeGreaterThanOrEqual(gpu.bf16TFLOPS);
        }
      });

      it('should have FP4 >= FP8 when FP4 is supported', () => {
        if (gpu.fp4TFLOPS > 0) {
          expect(gpu.fp4TFLOPS).toBeGreaterThanOrEqual(gpu.fp8TFLOPS);
        }
      });

      it('should have INT4 >= INT8 when both are supported', () => {
        if (gpu.int4TOPS > 0 && gpu.int8TOPS > 0) {
          expect(gpu.int4TOPS).toBeGreaterThanOrEqual(gpu.int8TOPS);
        }
      });
    });
  });
});

describe('getTrainingTFLOPS — matmul/non-matmul model', () => {
  it('H100 BF16: returns getEffectiveTFLOPS unchanged (989)', () => {
    expect(getTrainingTFLOPS(H100_SXM, 'bf16')).toBe(989);
    expect(getTrainingTFLOPS(H100_SXM, 'bf16')).toBe(getEffectiveTFLOPS(H100_SXM, 'bf16'));
  });

  it('H100 FP8: ~1533 TFLOPS (not raw 1979)', () => {
    const training = getTrainingTFLOPS(H100_SXM, 'fp8');
    const raw = getEffectiveTFLOPS(H100_SXM, 'fp8');
    expect(raw).toBe(1979);
    // Amdahl: 989 / (0.80/2 + 0.20) = 989 / 0.60 ≈ 1648, then ×0.93 TE overhead ≈ 1533
    expect(training).toBeCloseTo(1533, -1);
    expect(training).toBeLessThan(raw);
    expect(training).toBeGreaterThan(H100_SXM.bf16TFLOPS);
  });

  it('A100 FP8: falls back to BF16 rate (no native FP8)', () => {
    const training = getTrainingTFLOPS(A100_80GB, 'fp8');
    // A100 has no native FP8, falls back to BF16 (312 TFLOPS)
    // matmulSpeedup = 1.0 → no adjustment
    expect(training).toBe(312);
    expect(training).toBe(getEffectiveTFLOPS(A100_80GB, 'bf16'));
  });

  it('B200 FP8: applies matmul fraction model', () => {
    const training = getTrainingTFLOPS(B200, 'fp8');
    const raw = getEffectiveTFLOPS(B200, 'fp8');
    expect(raw).toBe(4500);
    // Amdahl: 2250 / (0.80/2 + 0.20) = 2250 / 0.60 = 3750, then ×0.93 TE overhead = 3487.5
    expect(training).toBeCloseTo(3487.5, -1);
  });

  it('FP32/TF32/FP16: returns getEffectiveTFLOPS unchanged', () => {
    expect(getTrainingTFLOPS(H100_SXM, 'fp32')).toBe(getEffectiveTFLOPS(H100_SXM, 'fp32'));
    expect(getTrainingTFLOPS(H100_SXM, 'tf32')).toBe(getEffectiveTFLOPS(H100_SXM, 'tf32'));
    expect(getTrainingTFLOPS(H100_SXM, 'fp16')).toBe(getEffectiveTFLOPS(H100_SXM, 'fp16'));
  });
});

describe('Precision-aware optimizer memory', () => {
  it('BF16 AdamW: 12 bytes/param (master + m1 + m2)', () => {
    expect(calculateOptimizerMemory(1e9, 'adamw', 'bf16')).toBe(12e9);
  });

  it('FP32 AdamW: 8 bytes/param (m1 + m2, no master copy)', () => {
    expect(calculateOptimizerMemory(1e9, 'adamw', 'fp32')).toBe(8e9);
  });

  it('TF32 AdamW: 8 bytes/param (stored in fp32 containers, no master)', () => {
    expect(calculateOptimizerMemory(1e9, 'adamw', 'tf32')).toBe(8e9);
  });

  it('FP8 AdamW: 10 bytes/param (BF16 master + 2 momentums)', () => {
    expect(calculateOptimizerMemory(1e9, 'adamw', 'fp8')).toBe(10e9);
  });

  it('FP16 AdamW: 12 bytes/param (master copy needed)', () => {
    expect(calculateOptimizerMemory(1e9, 'adamw', 'fp16')).toBe(12e9);
  });

  it('FP32 SGD: 4 bytes/param (momentum only, no master)', () => {
    expect(calculateOptimizerMemory(1e9, 'sgd', 'fp32')).toBe(4e9);
  });

  it('BF16 SGD: 8 bytes/param (master + momentum)', () => {
    expect(calculateOptimizerMemory(1e9, 'sgd', 'bf16')).toBe(8e9);
  });

  it('default paramDtype is bf16', () => {
    // Backward compat: no paramDtype arg → bf16 → 12 bytes/param
    expect(calculateOptimizerMemory(1e9, 'adamw')).toBe(12e9);
  });
});

describe('Precision fallback warnings', () => {
  it('A100 + FP8: returns fallback warning', () => {
    const warning = getPrecisionFallbackWarning(A100_80GB, 'fp8');
    expect(warning).not.toBeNull();
    expect(warning).toContain('A100');
    expect(warning).toContain('FP8');
    expect(warning).toContain('BF16');
  });

  it('H100 + FP8: no warning (native support)', () => {
    expect(getPrecisionFallbackWarning(H100_SXM, 'fp8')).toBeNull();
  });

  it('H100 + BF16: no warning (native support)', () => {
    expect(getPrecisionFallbackWarning(H100_SXM, 'bf16')).toBeNull();
  });

  it('B200 + FP4: no warning (native support)', () => {
    expect(getPrecisionFallbackWarning(B200, 'fp4')).toBeNull();
  });

  it('H100 + FP4: returns fallback warning', () => {
    const warning = getPrecisionFallbackWarning(H100_SXM, 'fp4');
    expect(warning).not.toBeNull();
    expect(warning).toContain('FP4');
  });
});

describe('Precision memory vs TFLOPS tradeoff', () => {
  it('should show memory reduction proportional to precision reduction', () => {
    // Memory scales linearly with bytes per element
    const fp32Bytes = DTYPE_BYTES.fp32;
    const bf16Bytes = DTYPE_BYTES.bf16;
    const fp8Bytes = DTYPE_BYTES.fp8;
    const fp4Bytes = DTYPE_BYTES.fp4;

    // FP32 -> BF16 = 2x reduction
    expect(fp32Bytes / bf16Bytes).toBe(2);

    // BF16 -> FP8 = 2x reduction
    expect(bf16Bytes / fp8Bytes).toBe(2);

    // FP8 -> FP4 = 2x reduction
    expect(fp8Bytes / fp4Bytes).toBe(2);

    // FP32 -> FP4 = 8x total reduction
    expect(fp32Bytes / fp4Bytes).toBe(8);
  });

  it('should show TFLOPS increase roughly matches memory reduction', () => {
    const b200 = ALL_GPUS['b200'];

    // BF16 -> FP8: 2x memory reduction, exactly 2x TFLOPS increase (4500/2250)
    const fp8ToBf16Ratio = b200.fp8TFLOPS / b200.bf16TFLOPS;
    expect(fp8ToBf16Ratio).toBe(2);

    // FP8 -> FP4: 2x memory reduction, exactly 2x TFLOPS increase (9000/4500)
    const fp4ToFp8Ratio = b200.fp4TFLOPS / b200.fp8TFLOPS;
    expect(fp4ToFp8Ratio).toBe(2);
  });
});
