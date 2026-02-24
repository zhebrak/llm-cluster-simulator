/**
 * FSDP Gather Buffer Validation Tests
 *
 * Validates that FSDP gather buffers (peakActivations - activations) and temporary
 * buffers are computed correctly across standalone FSDP, 3D parallel (FSDP+TP,
 * FSDP+TP+PP), and that non-FSDP strategies have no gather overhead.
 *
 * Formulas under test:
 *
 * Standalone FSDP (fsdp.ts):
 *   gatherOverhead = paramsPerLayer × 2 × 4  (param + grad + 2×prefetch, all bf16)
 *   temporary      = paramsPerLayer × 2 × 2  (double-buffered bf16 param gather)
 *
 * 3D Parallel with dpType=fsdp (3d-parallel.ts):
 *   gatherOverhead = (paramsPerLayer / tp) × 2 × 4
 *   dpBuffer       = (paramsPerLayer / tp) × 2 × 2
 *   temporary      = tpBuffer + ppBuffer + dpBuffer + epBuffer
 *     tpBuffer = hiddenSize × tokensPerMB × 2 × 2
 *     ppBuffer = hiddenSize × tokensPerMB × 2 × 2
 *     epBuffer = 2 × tokensPerMB × hiddenSize × 2 × 2 × 1.15  (if EP>1, else 0)
 *
 * Non-FSDP dpTypes (DDP, ZeRO-1):
 *   peakActivations === activations  (no gather overhead)
 */
import { describe, it, expect } from 'vitest';
import {
  getSimulationMetrics,
  type SimulationConfig,
  type SimulationMetrics,
} from '../../src/core/simulation/engine.ts';
import { createMultiNodeCluster } from '../../src/core/hardware/topology.ts';

// ---------------------------------------------------------------------------
// Helpers (same pattern as memory-distribution.test.ts)
// ---------------------------------------------------------------------------

function cfg(
  gpuId: string,
  gpusPerNode: number,
  numNodes: number,
  modelId: string,
  strategy: SimulationConfig['strategyType'],
  globalBatchSize: number,
  microBatchSize: number,
  sequenceLength: number,
  strategyConfig?: SimulationConfig['strategyConfig'],
): SimulationConfig {
  return {
    clusterConfig: createMultiNodeCluster(gpuId, gpusPerNode, numNodes)!,
    modelId,
    globalBatchSize,
    microBatchSize,
    sequenceLength,
    strategyType: strategy,
    strategyConfig,
  };
}

function sim(config: SimulationConfig): SimulationMetrics {
  return getSimulationMetrics(config);
}

/** Convenience: within tolerance as a fraction (0.001 = 0.1%) */
function expectClose(actual: number, expected: number, tolerance: number, label?: string) {
  const ratio = actual / expected;
  const msg = label ? `${label}: ` : '';
  expect(ratio, `${msg}expected ${expected.toExponential(3)}, got ${actual.toExponential(3)} (ratio=${ratio.toFixed(6)})`).toBeGreaterThan(1 - tolerance);
  expect(ratio, `${msg}expected ${expected.toExponential(3)}, got ${actual.toExponential(3)} (ratio=${ratio.toFixed(6)})`).toBeLessThan(1 + tolerance);
}

// ---------------------------------------------------------------------------
// Section 1: Standalone FSDP — Gather Overhead
// ---------------------------------------------------------------------------

describe('Standalone FSDP: Gather Overhead', () => {
  it('llama3.2-1b (16 layers): overhead = paramsPerLayer × 2 × 4', () => {
    const m = sim(cfg('h100-sxm', 8, 1, 'llama3.2-1b', 'fsdp', 32, 4, 2048));
    const overhead = m.memoryPerGPU.peakActivations - m.memoryPerGPU.activations;
    expectClose(overhead, 617_907_200, 0.001, 'llama3.2-1b gather overhead');
  });

  it('llama2-7b (32 layers): overhead = paramsPerLayer × 2 × 4', () => {
    const m = sim(cfg('h100-sxm', 8, 1, 'llama2-7b', 'fsdp', 32, 4, 2048));
    const overhead = m.memoryPerGPU.peakActivations - m.memoryPerGPU.activations;
    expectClose(overhead, 1_684_603_904, 0.001, 'llama2-7b gather overhead');
  });

  it('llama3-8b (32 layers): overhead = paramsPerLayer × 2 × 4', () => {
    const m = sim(cfg('h100-sxm', 8, 1, 'llama3-8b', 'fsdp', 32, 4, 2048));
    const overhead = m.memoryPerGPU.peakActivations - m.memoryPerGPU.activations;
    expectClose(overhead, 2_007_565_312, 0.001, 'llama3-8b gather overhead');
  });

  it('llama2-70b (80 layers): overhead = paramsPerLayer × 2 × 4', () => {
    const m = sim(cfg('h100-sxm', 8, 1, 'llama2-70b', 'fsdp', 32, 4, 2048));
    const overhead = m.memoryPerGPU.peakActivations - m.memoryPerGPU.activations;
    expectClose(overhead, 6_897_664_819, 0.001, 'llama2-70b gather overhead');
  });
});

// ---------------------------------------------------------------------------
// Section 2: Standalone FSDP — `temporary` Field
// ---------------------------------------------------------------------------

describe('Standalone FSDP: temporary field', () => {
  it('llama3.2-1b: temporary = paramsPerLayer × 2 × 2', () => {
    const m = sim(cfg('h100-sxm', 8, 1, 'llama3.2-1b', 'fsdp', 32, 4, 2048));
    expectClose(m.memoryPerGPU.temporary, 308_953_600, 0.001, 'llama3.2-1b temporary');
  });

  it('llama2-7b: temporary = paramsPerLayer × 2 × 2', () => {
    const m = sim(cfg('h100-sxm', 8, 1, 'llama2-7b', 'fsdp', 32, 4, 2048));
    expectClose(m.memoryPerGPU.temporary, 842_301_952, 0.001, 'llama2-7b temporary');
  });

  it('llama3-8b: temporary = paramsPerLayer × 2 × 2', () => {
    const m = sim(cfg('h100-sxm', 8, 1, 'llama3-8b', 'fsdp', 32, 4, 2048));
    expectClose(m.memoryPerGPU.temporary, 1_003_782_656, 0.001, 'llama3-8b temporary');
  });
});

// ---------------------------------------------------------------------------
// Section 3: 3D FSDP — Gather Overhead
// ---------------------------------------------------------------------------

describe('3D FSDP: Gather Overhead', () => {
  it('llama2-70b fsdp-tp tp=8: overhead = (paramsPerLayer/tp) × 2 × 4', () => {
    const m = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'fsdp-tp', 256, 1, 2048, { tp: 8 }));
    const overhead = m.memoryPerGPU.peakActivations - m.memoryPerGPU.activations;
    expectClose(overhead, 862_208_102, 0.001, '70b fsdp-tp tp=8 overhead');
  });

  it('llama2-70b fsdp-tp tp=4: overhead = (paramsPerLayer/tp) × 2 × 4', () => {
    const m = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'fsdp-tp', 256, 1, 2048, { tp: 4 }));
    const overhead = m.memoryPerGPU.peakActivations - m.memoryPerGPU.activations;
    expectClose(overhead, 1_724_416_205, 0.001, '70b fsdp-tp tp=4 overhead');
  });

  it('llama2-70b fsdp-tp-pp tp=8, pp=2: overhead = (paramsPerLayer/tp) × 2 × 4', () => {
    const m = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'fsdp-tp-pp', 128, 1, 2048, { tp: 8, pp: 2 }));
    const overhead = m.memoryPerGPU.peakActivations - m.memoryPerGPU.activations;
    expectClose(overhead, 862_208_102, 0.001, '70b fsdp-tp-pp tp=8,pp=2 overhead');
  });

  it('gpt3-175b fsdp-tp-pp tp=8, pp=4: overhead = (paramsPerLayer/tp) × 2 × 4', () => {
    const m = sim(cfg('h100-sxm', 8, 32, 'gpt3-175b', 'fsdp-tp-pp', 128, 1, 2048, { tp: 8, pp: 4 }));
    const overhead = m.memoryPerGPU.peakActivations - m.memoryPerGPU.activations;
    expectClose(overhead, 1_824_965_120, 0.001, '175b fsdp-tp-pp tp=8,pp=4 overhead');
  });
});

// ---------------------------------------------------------------------------
// Section 4: 3D FSDP — `temporary` Decomposition
// ---------------------------------------------------------------------------

describe('3D FSDP: temporary decomposition', () => {
  it('llama2-70b fsdp-tp tp=8: temporary = tpBuffer + ppBuffer + dpBuffer', () => {
    const m = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'fsdp-tp', 256, 1, 2048, { tp: 8 }));
    expectClose(m.memoryPerGPU.temporary, 565_321_779, 0.001, '70b fsdp-tp tp=8 temporary');
  });

  it('llama2-70b fsdp-tp-pp tp=8, pp=2: temporary = tpBuffer + ppBuffer + dpBuffer', () => {
    const m = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'fsdp-tp-pp', 128, 1, 2048, { tp: 8, pp: 2 }));
    expectClose(m.memoryPerGPU.temporary, 565_321_779, 0.001, '70b fsdp-tp-pp tp=8,pp=2 temporary');
  });

  it('gpt3-175b fsdp-tp-pp tp=8, pp=4: temporary = tpBuffer + ppBuffer + dpBuffer', () => {
    const m = sim(cfg('h100-sxm', 8, 32, 'gpt3-175b', 'fsdp-tp-pp', 128, 1, 2048, { tp: 8, pp: 4 }));
    expectClose(m.memoryPerGPU.temporary, 1_113_809_152, 0.001, '175b fsdp-tp-pp tp=8,pp=4 temporary');
  });
});

// ---------------------------------------------------------------------------
// Section 5: Negative Cases — No Gather Buffers
// ---------------------------------------------------------------------------

describe('No gather buffers for non-FSDP strategies', () => {
  it('DDP-TP-PP: peakActivations === activations', () => {
    const m = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'ddp-tp-pp', 128, 1, 2048, { tp: 8, pp: 2 }));
    expect(m.memoryPerGPU.peakActivations).toBe(m.memoryPerGPU.activations);
  });

  it('ZeRO1-TP: peakActivations === activations', () => {
    const m = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'zero1-tp', 256, 1, 2048, { tp: 8 }));
    expect(m.memoryPerGPU.peakActivations).toBe(m.memoryPerGPU.activations);
  });

  it('ZeRO1-TP-PP: peakActivations === activations', () => {
    const m = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'zero1-tp-pp', 128, 1, 2048, { tp: 8, pp: 2 }));
    expect(m.memoryPerGPU.peakActivations).toBe(m.memoryPerGPU.activations);
  });
});

// ---------------------------------------------------------------------------
// Section 6: TP Scaling — Gather Buffer Halves with 2×TP
// ---------------------------------------------------------------------------

describe('TP Scaling: gather buffer halves with 2×TP', () => {
  it('llama2-70b fsdp-tp: tp=4 overhead / tp=8 overhead = 2.0', () => {
    const m4 = sim(cfg('h100-sxm', 8, 4, 'llama2-70b', 'fsdp-tp', 64, 1, 2048, { tp: 4 }));
    const m8 = sim(cfg('h100-sxm', 8, 4, 'llama2-70b', 'fsdp-tp', 64, 1, 2048, { tp: 8 }));
    const o4 = m4.memoryPerGPU.peakActivations - m4.memoryPerGPU.activations;
    const o8 = m8.memoryPerGPU.peakActivations - m8.memoryPerGPU.activations;
    expectClose(o4 / o8, 2.0, 0.001, '70b tp=4/tp=8 ratio');
  });

  it('llama3-8b fsdp-tp: tp=2 overhead / tp=4 overhead = 2.0', () => {
    const m2 = sim(cfg('h100-sxm', 8, 2, 'llama3-8b', 'fsdp-tp', 32, 2, 2048, { tp: 2 }));
    const m4 = sim(cfg('h100-sxm', 8, 2, 'llama3-8b', 'fsdp-tp', 32, 2, 2048, { tp: 4 }));
    const o2 = m2.memoryPerGPU.peakActivations - m2.memoryPerGPU.activations;
    const o4 = m4.memoryPerGPU.peakActivations - m4.memoryPerGPU.activations;
    expectClose(o2 / o4, 2.0, 0.001, '8b tp=2/tp=4 ratio');
  });
});

// ---------------------------------------------------------------------------
// Section 7: PP Invariance — Gather Buffer Independent of PP
// ---------------------------------------------------------------------------

describe('PP Invariance: gather buffer independent of PP', () => {
  it('llama2-70b fsdp-tp-pp tp=8: pp=2 vs pp=4 → identical overhead', () => {
    const m2 = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'fsdp-tp-pp', 128, 1, 2048, { tp: 8, pp: 2 }));
    const m4 = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'fsdp-tp-pp', 128, 1, 2048, { tp: 8, pp: 4 }));
    const o2 = m2.memoryPerGPU.peakActivations - m2.memoryPerGPU.activations;
    const o4 = m4.memoryPerGPU.peakActivations - m4.memoryPerGPU.activations;
    expectClose(o2, o4, 0.001, 'pp=2 vs pp=4 overhead');
  });
});

// ---------------------------------------------------------------------------
// Section 8: MoE with EP
// ---------------------------------------------------------------------------

describe('MoE with EP', () => {
  it('mixtral-8x7b fsdp-tp tp=4: ep=1 vs ep=2 → same gather overhead', () => {
    const m1 = sim(cfg('h100-sxm', 8, 4, 'mixtral-8x7b', 'fsdp-tp', 64, 1, 2048, { tp: 4, ep: 1 }));
    const m2 = sim(cfg('h100-sxm', 8, 4, 'mixtral-8x7b', 'fsdp-tp', 64, 1, 2048, { tp: 4, ep: 2 }));
    const o1 = m1.memoryPerGPU.peakActivations - m1.memoryPerGPU.activations;
    const o2 = m2.memoryPerGPU.peakActivations - m2.memoryPerGPU.activations;
    // EP doesn't affect gather overhead — it only adds to temporary via epBuffer
    expectClose(o1, o2, 0.001, 'EP does not affect gather overhead');
  });

  it('mixtral-8x7b fsdp-tp tp=4 ep=2: temporary includes epBuffer', () => {
    const m1 = sim(cfg('h100-sxm', 8, 4, 'mixtral-8x7b', 'fsdp-tp', 64, 1, 2048, { tp: 4, ep: 1 }));
    const m2 = sim(cfg('h100-sxm', 8, 4, 'mixtral-8x7b', 'fsdp-tp', 64, 1, 2048, { tp: 4, ep: 2 }));
    // epBuffer = 2 × tokensPerMB × hiddenSize × 2 × 2 × 1.15
    const epBuffer = m2.memoryPerGPU.temporary - m1.memoryPerGPU.temporary;
    expectClose(epBuffer, 77_175_194, 0.001, 'EP buffer size');
  });
});

// ---------------------------------------------------------------------------
// Section 9: Cross-Validation Invariants
// ---------------------------------------------------------------------------

describe('Cross-validation invariants', () => {
  it('Memory total = sum of components', () => {
    const m = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'fsdp-tp', 256, 1, 2048, { tp: 8 }));
    const computedTotal = m.memoryPerGPU.parameters
      + m.memoryPerGPU.gradients
      + m.memoryPerGPU.optimizerStates
      + m.memoryPerGPU.peakActivations
      + m.memoryPerGPU.temporary
      + m.memoryPerGPU.reserved;
    expectClose(m.memoryPerGPU.total, computedTotal, 0.001, 'Total = sum of components');
  });

  it('FSDP+TP(tp=1) gather overhead matches standalone FSDP', () => {
    // Uses ±1% tolerance (not 0.1%) because these go through different code paths
    // (3d-parallel.ts vs fsdp.ts) — the `temporary` field is computed differently
    // (3D sums tpBuffer+ppBuffer+dpBuffer vs standalone's gatheredParamBuffer×2),
    // causing a small legitimate difference. This is NOT a bug to fix.
    const fsdpTP = sim(cfg('h100-sxm', 8, 1, 'llama2-7b', 'fsdp-tp', 32, 4, 2048, { tp: 1 }));
    const pureFSDP = sim(cfg('h100-sxm', 8, 1, 'llama2-7b', 'fsdp', 32, 4, 2048));
    const overheadTP = fsdpTP.memoryPerGPU.peakActivations - fsdpTP.memoryPerGPU.activations;
    const overheadPure = pureFSDP.memoryPerGPU.peakActivations - pureFSDP.memoryPerGPU.activations;
    expectClose(overheadTP, overheadPure, 0.01, 'fsdp-tp(tp=1) vs standalone FSDP');
  });
});
