/**
 * Benchmark Validation Tests
 *
 * These tests validate the simulator against published benchmarks
 * from papers like Megatron-LM, DeepSpeed, and LLaMA.
 */

import { describe, it, expect } from 'vitest';
import {
  MEGATRON_BENCHMARKS,
  DEEPSPEED_BENCHMARKS,
  LLAMA_BENCHMARKS,
  ZERO_MEMORY_FACTORS,
  PIPELINE_BUBBLE_FORMULAS,
} from '../../src/core/validation/benchmarks.ts';
import {
  validateZeROMemory,
  validatePipelineBubble,
} from '../../src/core/validation/validator.ts';

describe('Benchmark Data', () => {
  it('should have valid Megatron benchmarks', () => {
    expect(MEGATRON_BENCHMARKS.length).toBeGreaterThan(0);

    for (const benchmark of MEGATRON_BENCHMARKS) {
      expect(benchmark.params).toBeGreaterThan(0);
      expect(benchmark.numGPUs).toBeGreaterThan(0);
      expect(benchmark.tflopsPerGPU).toBeGreaterThan(0);
      expect(benchmark.mfu).toBeGreaterThan(0);
      expect(benchmark.mfu).toBeLessThanOrEqual(1);
    }
  });

  it('should have valid DeepSpeed benchmarks', () => {
    expect(DEEPSPEED_BENCHMARKS.length).toBeGreaterThan(0);

    for (const benchmark of DEEPSPEED_BENCHMARKS) {
      expect(benchmark.params).toBeGreaterThan(0);
      expect(benchmark.strategy).toMatch(/ZeRO/i);
    }
  });

  it('should have valid LLaMA benchmarks', () => {
    expect(LLAMA_BENCHMARKS.length).toBeGreaterThan(0);

    for (const benchmark of LLAMA_BENCHMARKS) {
      expect(benchmark.model).toBe('LLaMA');
      expect(benchmark.params).toBeGreaterThanOrEqual(7);
    }
  });
});

describe('ZeRO Memory Formulas', () => {
  it('should calculate correct baseline memory', () => {
    // Baseline: 16Ψ (2 param + 2 grad + 12 optimizer)
    expect(ZERO_MEMORY_FACTORS.baseline).toBe(16);
  });

  it('should calculate correct ZeRO-1 memory reduction', () => {
    // ZeRO-1 with 64 GPUs: 4 + 12/64 = 4.1875
    const reduction = ZERO_MEMORY_FACTORS.zero1(64);
    expect(reduction).toBeCloseTo(4.1875, 4);
  });

  it('should calculate correct ZeRO-2 memory reduction', () => {
    // ZeRO-2 with 64 GPUs: 2 + 14/64 = 2.21875
    const reduction = ZERO_MEMORY_FACTORS.zero2(64);
    expect(reduction).toBeCloseTo(2.21875, 4);
  });

  it('should calculate correct ZeRO-3 memory reduction', () => {
    // ZeRO-3 with 64 GPUs: 16/64 = 0.25
    const reduction = ZERO_MEMORY_FACTORS.zero3(64);
    expect(reduction).toBeCloseTo(0.25, 4);
  });

  it('should show increasing memory reduction with each stage', () => {
    const n = 64;
    const baseline = ZERO_MEMORY_FACTORS.baseline;
    const zero1 = ZERO_MEMORY_FACTORS.zero1(n);
    const zero2 = ZERO_MEMORY_FACTORS.zero2(n);
    const zero3 = ZERO_MEMORY_FACTORS.zero3(n);

    expect(baseline).toBeGreaterThan(zero1);
    expect(zero1).toBeGreaterThan(zero2);
    expect(zero2).toBeGreaterThan(zero3);
  });

  it('should validate ZeRO memory factors', () => {
    const validation = validateZeROMemory(64);
    expect(validation).toHaveLength(4);
    expect(validation[0].stage).toBe(0);
    expect(validation[3].stage).toBe(3);
  });
});

describe('Pipeline Bubble Formulas', () => {
  it('should calculate correct GPipe bubble', () => {
    // GPipe with 8 stages, 16 microbatches: (8-1)/(8-1+16) = 7/23 ≈ 0.3043
    const bubble = PIPELINE_BUBBLE_FORMULAS.gpipe(8, 16);
    expect(bubble).toBeCloseTo(7 / 23, 4);
  });

  it('should calculate correct 1F1B bubble', () => {
    // 1F1B has same formula: (8-1)/(8-1+16) = 7/23 ≈ 0.3043
    const bubble = PIPELINE_BUBBLE_FORMULAS['1f1b'](8, 16);
    expect(bubble).toBeCloseTo(7 / 23, 4);
  });

  it('should calculate correct interleaved bubble', () => {
    // Interleaved with v=2: (8-1)/(8-1+16*2) = 7/39 ≈ 0.1795
    const bubble = PIPELINE_BUBBLE_FORMULAS.interleaved(8, 16, 2);
    expect(bubble).toBeCloseTo(7 / 39, 4);
  });

  it('should show bubble decreases with more microbatches', () => {
    const stages = 8;
    const bubble8 = PIPELINE_BUBBLE_FORMULAS.gpipe(stages, 8);
    const bubble16 = PIPELINE_BUBBLE_FORMULAS.gpipe(stages, 16);
    const bubble32 = PIPELINE_BUBBLE_FORMULAS.gpipe(stages, 32);

    expect(bubble8).toBeGreaterThan(bubble16);
    expect(bubble16).toBeGreaterThan(bubble32);
  });

  it('should validate pipeline bubble calculations', () => {
    const validation = validatePipelineBubble(8, 16);
    expect(validation).toHaveLength(3);
    expect(validation[0].schedule).toBe('GPipe');
    expect(validation[1].schedule).toBe('1F1B');
    expect(validation[2].schedule).toBe('Interleaved');
  });

  it('interleaved bubble < 1F1B bubble for same pp, m', () => {
    const stages = 8, m = 16;
    const standard = PIPELINE_BUBBLE_FORMULAS['1f1b'](stages, m);
    const interleaved = PIPELINE_BUBBLE_FORMULAS.interleaved(stages, m, 2);
    expect(interleaved).toBeLessThan(standard);
  });

  it('interleaved bubble decreases with higher v', () => {
    const stages = 8, m = 16;
    const v2 = PIPELINE_BUBBLE_FORMULAS.interleaved(stages, m, 2);
    const v3 = PIPELINE_BUBBLE_FORMULAS.interleaved(stages, m, 3);
    const v4 = PIPELINE_BUBBLE_FORMULAS.interleaved(stages, m, 4);
    expect(v3).toBeLessThan(v2);
    expect(v4).toBeLessThan(v3);
  });
});

describe('Scaling Laws', () => {
  it('should show MFU typically ranges from 0.3-0.6', () => {
    const allBenchmarks = [
      ...MEGATRON_BENCHMARKS,
      ...DEEPSPEED_BENCHMARKS,
      ...LLAMA_BENCHMARKS,
    ];

    for (const benchmark of allBenchmarks) {
      expect(benchmark.mfu).toBeGreaterThanOrEqual(0.2);
      expect(benchmark.mfu).toBeLessThanOrEqual(0.7);
    }
  });

  it('should show larger models require more GPUs', () => {
    // LLaMA models increase GPU count with size
    const llama7b = LLAMA_BENCHMARKS.find(b => b.params === 7);
    const llama65b = LLAMA_BENCHMARKS.find(b => b.params === 65);

    expect(llama7b).toBeDefined();
    expect(llama65b).toBeDefined();
    expect(llama65b!.numGPUs).toBeGreaterThan(llama7b!.numGPUs);
  });
});
