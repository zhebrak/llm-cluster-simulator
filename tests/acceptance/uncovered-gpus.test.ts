/**
 * Benchmark Tests for Previously Uncovered GPUs
 *
 * Covers 11 GPU presets that had no dedicated benchmark tests:
 *   H100 PCIe, H100 NVL, A100 40GB, L4, L40, L40S, A10, A10G,
 *   RTX 6000 Ada, RTX 4090, RTX 3090
 *
 * Each test pins MFU/step-time values (±8%) and verifies performance
 * ordering invariants. This catches:
 *   - GPU spec typos (wrong TFLOPS, memBW)
 *   - Strategy formula regressions
 *   - Memory calculation errors
 *   - FP8 fallback chain bugs
 *
 * Sections:
 *   7. Cross-Family TFLOPS Ratios (spec validation, run first)
 *   1. H100 Variants — MFU + Ordering
 *   2. A100 40GB vs 80GB — Discrimination
 *   3. L-Series — Ordering + Memory Boundaries
 *   4. A10 vs A10G — Discrimination + Ordering
 *   5. RTX GPUs — Ordering + Memory
 *   6. Inference — GPU-Specific Latency
 *   8. FP8 Precision Fallback
 */

import { describe, it, expect } from 'vitest';
import {
  type SimulationConfig,
} from '../../src/core/simulation/engine.ts';
import { getValidatedSimulationMetrics } from '../helpers/validated-metrics.ts';
import { getSimulationMetrics } from '../../src/core/simulation/engine.ts';
import { createMultiNodeCluster } from '../../src/core/hardware/topology.ts';
import { runInferenceSimulation } from '../../src/core/inference/simulation.ts';
import {
  getEffectiveTFLOPS,
  H100_SXM, H100_PCIE, H100_NVL,
  A100_40GB, A100_80GB,
  L4, L40, L40S,
  A10, A10G,
  RTX_4090, RTX_3090,
} from '../../src/core/hardware/gpu.ts';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function trainConfig(
  gpuId: string, modelId: string, strategy: SimulationConfig['strategyType'],
  gpusPerNode: number, numNodes: number, gbs: number, mbs: number, seqLen: number,
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

function sim(config: SimulationConfig) {
  return getValidatedSimulationMetrics(config);
}

/** For intentional OOM tests — skips validation that throws on OOM */
function simOom(config: SimulationConfig) {
  return getSimulationMetrics(config);
}

/** Assert value is within ±pct of expected */
function expectWithin(actual: number, expected: number, pct: number, label?: string) {
  const lo = expected * (1 - pct);
  const hi = expected * (1 + pct);
  expect(actual, `${label ?? ''} ${actual} not within ±${pct * 100}% of ${expected} [${lo.toFixed(4)}, ${hi.toFixed(4)}]`)
    .toBeGreaterThanOrEqual(lo);
  expect(actual, `${label ?? ''} ${actual} not within ±${pct * 100}% of ${expected} [${lo.toFixed(4)}, ${hi.toFixed(4)}]`)
    .toBeLessThanOrEqual(hi);
}

const TOL = 0.08; // ±8% tolerance for pinned values

// ===========================================================================
// Section 7: Cross-Family TFLOPS Ratios — Spec Validation
//
// Pure spec validation: no simulation logic. Catches data entry errors.
// Uses gpt3-125m DDP x8 (compute-bound) to verify TFLOPS ratios track
// through to step time ratios.
// ===========================================================================

describe('Section 7: Cross-Family TFLOPS Ratios', () => {
  it('H100 PCIe TFLOPS ≈ 76% of H100 SXM', () => {
    const ratio = H100_PCIE.bf16TFLOPS / H100_SXM.bf16TFLOPS;
    expectWithin(ratio, 756 / 989, 0.01);
  });

  it('H100 NVL TFLOPS ≈ 84% of H100 SXM', () => {
    const ratio = H100_NVL.bf16TFLOPS / H100_SXM.bf16TFLOPS;
    expectWithin(ratio, 835 / 989, 0.01);
  });

  it('L40S has ≈ 2× TFLOPS of L40', () => {
    const ratio = L40S.bf16TFLOPS / L40.bf16TFLOPS;
    expectWithin(ratio, 362 / 181, 0.01);
  });

  it('A10 has ≈ 1.8× TFLOPS of A10G', () => {
    const ratio = A10.bf16TFLOPS / A10G.bf16TFLOPS;
    expectWithin(ratio, 125 / 70, 0.01);
  });

  it('RTX 4090 has ≈ 1.16× TFLOPS of RTX 3090', () => {
    const ratio = RTX_4090.bf16TFLOPS / RTX_3090.bf16TFLOPS;
    expectWithin(ratio, 165.2 / 142, 0.01);
  });

  it('A100 40GB == A100 80GB TFLOPS', () => {
    expect(A100_40GB.bf16TFLOPS).toBe(A100_80GB.bf16TFLOPS);
  });

  it('A100 40GB memBW < A100 80GB memBW', () => {
    expect(A100_40GB.memoryBandwidthTBps).toBeLessThan(A100_80GB.memoryBandwidthTBps);
  });

  // Step time ratios on gpt3-125m DDP x8 should track TFLOPS ratios
  it('gpt3-125m DDP x8: step time ratios track TFLOPS (H100 SXM vs PCIe)', () => {
    const sxm = sim(trainConfig('h100-sxm', 'gpt3-125m', 'ddp', 8, 1, 16, 2, 2048));
    const pcie = sim(trainConfig('h100-pcie', 'gpt3-125m', 'ddp', 8, 1, 16, 2, 2048));
    // PCIe step time should be roughly (SXM TFLOPS / PCIe TFLOPS) × SXM step time
    // Ratio = 989/756 ≈ 1.31; actual ratio includes memBW + interconnect diffs
    const stepRatio = pcie.stepTimeMs / sxm.stepTimeMs;
    expect(stepRatio).toBeGreaterThan(1.2); // PCIe is slower
    expect(stepRatio).toBeLessThan(2.5);
  });

  it('gpt3-125m DDP x8: A100 40GB vs 80GB step time tracks memBW diff', () => {
    const a40 = sim(trainConfig('a100-40gb', 'gpt3-125m', 'ddp', 8, 1, 16, 2, 2048));
    const a80 = sim(trainConfig('a100-80gb', 'gpt3-125m', 'ddp', 8, 1, 16, 2, 2048));
    // Same TFLOPS, different memBW → small but real step time difference
    expect(a40.stepTimeMs).toBeGreaterThan(a80.stepTimeMs);
  });
});

// ===========================================================================
// Section 1: H100 Variants — MFU + Ordering
// ===========================================================================

describe('Section 1: H100 Variants — llama2-7b FSDP x8', () => {
  const sxm = () => sim(trainConfig('h100-sxm', 'llama2-7b', 'fsdp', 8, 1, 16, 2, 2048));
  const pcie = () => sim(trainConfig('h100-pcie', 'llama2-7b', 'fsdp', 8, 1, 16, 2, 2048));
  const nvl = () => sim(trainConfig('h100-nvl', 'llama2-7b', 'fsdp', 8, 1, 16, 2, 2048));

  it('H100 PCIe: MFU ≈ 32.4%', () => {
    expectWithin(pcie().mfu, 0.324, TOL, 'H100 PCIe MFU');
  });

  it('H100 NVL: MFU ≈ 46.3%', () => {
    expectWithin(nvl().mfu, 0.463, TOL, 'H100 NVL MFU');
  });

  it('Ordering: SXM step time < NVL step time < PCIe step time', () => {
    const s = sxm(), n = nvl(), p = pcie();
    expect(s.stepTimeMs).toBeLessThan(n.stepTimeMs);
    expect(n.stepTimeMs).toBeLessThan(p.stepTimeMs);
  });

  it('TFLOPS/GPU ordering: SXM > NVL > PCIe', () => {
    expect(H100_SXM.bf16TFLOPS).toBeGreaterThan(H100_NVL.bf16TFLOPS);
    expect(H100_NVL.bf16TFLOPS).toBeGreaterThan(H100_PCIE.bf16TFLOPS);
  });
});

// ===========================================================================
// Section 2: A100 40GB vs 80GB — Discrimination
// ===========================================================================

describe('Section 2: A100 40GB vs 80GB — llama2-7b FSDP x8', () => {
  const a40 = () => sim(trainConfig('a100-40gb', 'llama2-7b', 'fsdp', 8, 1, 16, 2, 2048));
  const a80 = () => sim(trainConfig('a100-80gb', 'llama2-7b', 'fsdp', 8, 1, 16, 2, 2048));

  it('A100 40GB: MFU ≈ 48.6%', () => {
    expectWithin(a40().mfu, 0.486, TOL, 'A100 40GB MFU');
  });

  it('A100 40GB step time > A100 80GB step time (lower memBW)', () => {
    expect(a40().stepTimeMs).toBeGreaterThan(a80().stepTimeMs);
  });

  it('MFU ratio within 0.92-1.03 (same TFLOPS, close MFU)', () => {
    const ratio = a40().mfu / a80().mfu;
    expect(ratio).toBeGreaterThanOrEqual(0.92);
    expect(ratio).toBeLessThanOrEqual(1.03);
  });
});

// ===========================================================================
// Section 3: L-Series — Ordering + Memory Boundaries
// ===========================================================================

describe('Section 3: L-Series GPUs', () => {
  it('L40: llama2-7b FSDP x8 MFU ≈ 44.4%', () => {
    const m = sim(trainConfig('l40', 'llama2-7b', 'fsdp', 8, 1, 16, 2, 2048));
    expectWithin(m.mfu, 0.444, TOL, 'L40 MFU');
  });

  it('L40S: llama2-7b FSDP x8 MFU ≈ 33.5%', () => {
    const m = sim(trainConfig('l40s', 'llama2-7b', 'fsdp', 8, 1, 16, 2, 2048));
    expectWithin(m.mfu, 0.335, TOL, 'L40S MFU');
  });

  it('L4: gpt3-125m DDP x8 MFU ≈ 28.6%', () => {
    const m = sim(trainConfig('l4', 'gpt3-125m', 'ddp', 8, 1, 16, 2, 2048));
    expectWithin(m.mfu, 0.286, TOL, 'L4 MFU');
  });

  it('Ordering: L40S step time < L40 step time (362 vs 181 TFLOPS)', () => {
    const l40 = sim(trainConfig('l40', 'llama2-7b', 'fsdp', 8, 1, 16, 2, 2048));
    const l40s = sim(trainConfig('l40s', 'llama2-7b', 'fsdp', 8, 1, 16, 2, 2048));
    expect(l40s.stepTimeMs).toBeLessThan(l40.stepTimeMs);
  });

  it('llama2-7b DDP x8 OOMs on L40 (48GB)', () => {
    const m = simOom(trainConfig('l40', 'llama2-7b', 'ddp', 8, 1, 16, 2, 2048));
    expect(m.memoryUtilization).toBeGreaterThan(1.0);
  });

  it('llama2-7b DDP x8 OOMs on L40S (48GB)', () => {
    const m = simOom(trainConfig('l40s', 'llama2-7b', 'ddp', 8, 1, 16, 2, 2048));
    expect(m.memoryUtilization).toBeGreaterThan(1.0);
  });

  it('llama3-8b DDP x8 OOMs on L4 (24GB)', () => {
    const m = simOom(trainConfig('l4', 'llama3-8b', 'ddp', 8, 1, 16, 2, 2048));
    expect(m.memoryUtilization).toBeGreaterThan(1.0);
  });
});

// ===========================================================================
// Section 4: A10 vs A10G — Discrimination + Ordering
// ===========================================================================

describe('Section 4: A10 vs A10G — gpt3-125m DDP x4', () => {
  const a10 = () => sim(trainConfig('a10', 'gpt3-125m', 'ddp', 4, 1, 8, 2, 2048));
  const a10g = () => sim(trainConfig('a10g', 'gpt3-125m', 'ddp', 4, 1, 8, 2, 2048));

  it('A10: MFU ≈ 35.1%', () => {
    expectWithin(a10().mfu, 0.351, TOL, 'A10 MFU');
  });

  it('A10G: MFU ≈ 40.9%', () => {
    expectWithin(a10g().mfu, 0.409, TOL, 'A10G MFU');
  });

  it('A10 step time < A10G step time (125 vs 70 TFLOPS)', () => {
    expect(a10().stepTimeMs).toBeLessThan(a10g().stepTimeMs);
  });

  it('llama3-8b DDP x4 OOMs on A10 (24GB)', () => {
    const m = simOom(trainConfig('a10', 'llama3-8b', 'ddp', 4, 1, 8, 2, 2048));
    expect(m.memoryUtilization).toBeGreaterThan(1.0);
  });

  it('llama3-8b DDP x4 OOMs on A10G (24GB)', () => {
    const m = simOom(trainConfig('a10g', 'llama3-8b', 'ddp', 4, 1, 8, 2, 2048));
    expect(m.memoryUtilization).toBeGreaterThan(1.0);
  });
});

// ===========================================================================
// Section 5: RTX GPUs — Ordering + Memory
// ===========================================================================

describe('Section 5: RTX GPUs — llama3.2-3b FSDP x8', () => {
  const rtx4090 = () => sim(trainConfig('rtx-4090', 'llama3.2-3b', 'fsdp', 8, 1, 16, 2, 2048));
  const rtx3090 = () => sim(trainConfig('rtx-3090', 'llama3.2-3b', 'fsdp', 8, 1, 16, 2, 2048));
  const rtx6000 = () => sim(trainConfig('rtx-6000-ada', 'llama3.2-3b', 'fsdp', 8, 1, 16, 2, 2048));

  it('RTX 4090: MFU ≈ 43.8%', () => {
    expectWithin(rtx4090().mfu, 0.438, TOL, 'RTX 4090 MFU');
  });

  it('RTX 3090: MFU ≈ 44.8%', () => {
    expectWithin(rtx3090().mfu, 0.448, TOL, 'RTX 3090 MFU');
  });

  it('RTX 6000 Ada: MFU ≈ 42.3%', () => {
    expectWithin(rtx6000().mfu, 0.423, TOL, 'RTX 6000 Ada MFU');
  });

  it('Ordering: RTX 4090 step time < RTX 3090 step time (TFLOPS-driven)', () => {
    // 165.2 vs 142 TFLOPS; memBW is similar (1.008 vs 0.936 TB/s)
    expect(rtx4090().stepTimeMs).toBeLessThan(rtx3090().stepTimeMs);
  });

  it('RTX 6000 Ada fits llama2-7b FSDP x8 (48GB)', () => {
    const m = sim(trainConfig('rtx-6000-ada', 'llama2-7b', 'fsdp', 8, 1, 16, 2, 2048));
    expect(m.memoryUtilization).toBeLessThanOrEqual(1.0);
  });

  it('RTX 6000 Ada llama2-7b DDP x8 OOMs (48GB)', () => {
    const m = simOom(trainConfig('rtx-6000-ada', 'llama2-7b', 'ddp', 8, 1, 16, 2, 2048));
    expect(m.memoryUtilization).toBeGreaterThan(1.0);
  });

  it('llama3-8b DDP x8 OOMs on RTX 4090 (24GB)', () => {
    const m = simOom(trainConfig('rtx-4090', 'llama3-8b', 'ddp', 8, 1, 16, 2, 2048));
    expect(m.memoryUtilization).toBeGreaterThan(1.0);
  });

  it('llama3-8b DDP x8 OOMs on RTX 3090 (24GB)', () => {
    const m = simOom(trainConfig('rtx-3090', 'llama3-8b', 'ddp', 8, 1, 16, 2, 2048));
    expect(m.memoryUtilization).toBeGreaterThan(1.0);
  });
});

// ===========================================================================
// Section 6: Inference — GPU-Specific Latency
//
// Tests that the gpuId resolution fix (simulation.ts) correctly routes
// inference to the specified GPU, producing distinct per-GPU latencies.
// ===========================================================================

describe('Section 6: Inference — GPU-Specific Latency', () => {
  const UNCOVERED_GPU_IDS = [
    'h100-pcie', 'h100-nvl', 'a100-40gb', 'l4', 'l40', 'l40s',
    'a10', 'a10g', 'rtx-6000-ada', 'rtx-4090', 'rtx-3090',
  ];

  // Pin TPOT for llama3.2-1b BF16 on each GPU
  const expectedTPOT: Record<string, number> = {
    'h100-pcie':     2.4181,
    'h100-nvl':      1.2401,
    'a100-40gb':     3.1101,
    'l4':            16.1208,
    'l40':           5.5975,
    'l40s':          5.5975,
    'a10':           8.0604,
    'a10g':          8.0604,
    'rtx-6000-ada':  5.0377,
    'rtx-4090':      4.7979,
    'rtx-3090':      5.1669,
  };

  for (const gpuId of UNCOVERED_GPU_IDS) {
    it(`llama3.2-1b BF16 on ${gpuId}: TPOT ≈ ${expectedTPOT[gpuId]}ms`, () => {
      const r = runInferenceSimulation({
        modelId: 'llama3.2-1b', gpuId, numGPUs: 1,
        batchSize: 1, inputSeqLen: 512, outputSeqLen: 256,
        weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1,
      });
      expect(r.success).toBe(true);
      expectWithin(r.latency.tpot, expectedTPOT[gpuId], TOL, `${gpuId} TPOT`);
    });
  }

  it('Each GPU produces distinct TPOT (no silent H100 SXM fallback)', () => {
    const tpots = new Set<string>();
    for (const gpuId of UNCOVERED_GPU_IDS) {
      const r = runInferenceSimulation({
        modelId: 'llama3.2-1b', gpuId, numGPUs: 1,
        batchSize: 1, inputSeqLen: 512, outputSeqLen: 256,
        weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1,
      });
      tpots.add(r.latency.tpot.toFixed(6));
    }
    // L40 and L40S have same memBW (0.864 TB/s) so same TPOT; A10 and A10G also same memBW
    // At least 8 distinct values out of 11
    expect(tpots.size).toBeGreaterThanOrEqual(8);
  });

  it('Higher-memBW GPUs have lower TPOT (decode is memory-bound)', () => {
    // H100 NVL (3.9 TB/s) < H100 PCIe (2.0 TB/s) < A100 40GB (1.555 TB/s)
    const nvl = runInferenceSimulation({
      modelId: 'llama3.2-1b', gpuId: 'h100-nvl', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256,
      weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1,
    });
    const pcie = runInferenceSimulation({
      modelId: 'llama3.2-1b', gpuId: 'h100-pcie', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256,
      weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1,
    });
    const a100 = runInferenceSimulation({
      modelId: 'llama3.2-1b', gpuId: 'a100-40gb', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256,
      weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1,
    });

    expect(nvl.latency.tpot).toBeLessThan(pcie.latency.tpot);
    expect(pcie.latency.tpot).toBeLessThan(a100.latency.tpot);
  });

  it('llama3-8b BF16 inference fits on 48GB+ GPUs', () => {
    for (const gpuId of ['h100-pcie', 'h100-nvl', 'l40', 'l40s', 'rtx-6000-ada']) {
      const r = runInferenceSimulation({
        modelId: 'llama3-8b', gpuId, numGPUs: 1,
        batchSize: 1, inputSeqLen: 512, outputSeqLen: 256,
        weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1,
      });
      expect(r.success, `${gpuId} should fit llama3-8b BF16`).toBe(true);
    }
  });
});

// ===========================================================================
// Section 8: FP8 Precision Fallback
//
// Ada GPUs (L4, L40S, RTX 4090) have native FP8 → higher TFLOPS.
// Ampere GPUs (A100, A10G, RTX 3090) lack FP8 → fall back to BF16.
// ===========================================================================

describe('Section 8: FP8 Precision Fallback', () => {
  // Ada GPUs: native FP8
  it('L4 (Ada): FP8 = 242 TFLOPS (native)', () => {
    expect(getEffectiveTFLOPS(L4, 'fp8')).toBe(242);
  });

  it('L40S (Ada): FP8 = 733 TFLOPS (native)', () => {
    expect(getEffectiveTFLOPS(L40S, 'fp8')).toBe(733);
  });

  it('RTX 4090 (Ada): FP8 = 330.3 TFLOPS (native)', () => {
    expect(getEffectiveTFLOPS(RTX_4090, 'fp8')).toBe(330.3);
  });

  // Ampere GPUs: FP8 falls back to BF16
  it('A100 40GB (Ampere): FP8 falls back to BF16 = 312 TFLOPS', () => {
    expect(getEffectiveTFLOPS(A100_40GB, 'fp8')).toBe(312);
    expect(A100_40GB.fp8TFLOPS).toBe(0); // no native FP8
  });

  it('A10G (Ampere): FP8 falls back to BF16 = 70 TFLOPS', () => {
    expect(getEffectiveTFLOPS(A10G, 'fp8')).toBe(70);
    expect(A10G.fp8TFLOPS).toBe(0);
  });

  it('RTX 3090 (Ampere): FP8 falls back to BF16 = 142 TFLOPS', () => {
    expect(getEffectiveTFLOPS(RTX_3090, 'fp8')).toBe(142);
    expect(RTX_3090.fp8TFLOPS).toBe(0);
  });

  // Ada FP8 > BF16 (the whole point of FP8)
  it('Ada GPUs: FP8 TFLOPS > BF16 TFLOPS', () => {
    for (const gpu of [L4, L40S, RTX_4090]) {
      expect(
        getEffectiveTFLOPS(gpu, 'fp8'),
        `${gpu.name} FP8 should exceed BF16`,
      ).toBeGreaterThan(getEffectiveTFLOPS(gpu, 'bf16'));
    }
  });

  // Ampere FP8 == BF16 (no hardware support)
  it('Ampere GPUs: FP8 falls back to BF16 TFLOPS', () => {
    for (const gpu of [A100_40GB, A10G, RTX_3090]) {
      expect(
        getEffectiveTFLOPS(gpu, 'fp8'),
        `${gpu.name} FP8 should equal BF16 (no native FP8)`,
      ).toBe(getEffectiveTFLOPS(gpu, 'bf16'));
    }
  });
});
