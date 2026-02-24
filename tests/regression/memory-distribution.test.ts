/**
 * Memory Distribution Tests
 *
 * Validates that memory is correctly sharded across GPUs for every strategy.
 * Covers: parameter memory, gradient memory, optimizer states, FSDP gather buffers,
 * scaling laws, and edge cases.
 */
import { describe, it, expect } from 'vitest';
import {
  getSimulationMetrics,
  type SimulationConfig,
  type SimulationMetrics,
} from '../../src/core/simulation/engine.ts';
import { createMultiNodeCluster } from '../../src/core/hardware/topology.ts';
import { getModel } from '../../src/core/models/index.ts';

// ---------------------------------------------------------------------------
// Helpers
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

/** Convenience: within tolerance as a fraction (0.01 = 1%) */
function expectClose(actual: number, expected: number, tolerance: number, label?: string) {
  const ratio = actual / expected;
  const msg = label ? `${label}: ` : '';
  expect(ratio, `${msg}expected ${expected.toExponential(3)}, got ${actual.toExponential(3)} (ratio=${ratio.toFixed(4)})`).toBeGreaterThan(1 - tolerance);
  expect(ratio, `${msg}expected ${expected.toExponential(3)}, got ${actual.toExponential(3)} (ratio=${ratio.toFixed(4)})`).toBeLessThan(1 + tolerance);
}

// ---------------------------------------------------------------------------
// Section 1: Parameter Memory Formulas
// ---------------------------------------------------------------------------

describe('Parameter Memory Distribution', () => {
  it('DDP: full replication — params = totalParams × 2', () => {
    const model = getModel('llama2-7b')!;
    const m = sim(cfg('h100-sxm', 8, 1, 'llama2-7b', 'ddp', 32, 4, 2048));
    const expected = model.totalParams * 2; // bf16
    expectClose(m.memoryPerGPU.parameters, expected, 0.01, 'DDP params');
  });

  it('FSDP: sharded — params = totalParams × 2 / N', () => {
    const model = getModel('llama2-7b')!;
    const N = 8;
    const m = sim(cfg('h100-sxm', 8, 1, 'llama2-7b', 'fsdp', 32, 4, 2048));
    const expected = model.totalParams * 2 / N;
    expectClose(m.memoryPerGPU.parameters, expected, 0.01, 'FSDP params');
  });

  it('ZeRO-1: full replication — params = totalParams × 2', () => {
    const model = getModel('llama2-7b')!;
    const m = sim(cfg('h100-sxm', 8, 1, 'llama2-7b', 'zero-1', 32, 4, 2048));
    const expected = model.totalParams * 2;
    expectClose(m.memoryPerGPU.parameters, expected, 0.01, 'ZeRO-1 params');
  });

  it('ZeRO-3: sharded — params = totalParams × 2 / N', () => {
    const model = getModel('llama2-7b')!;
    const N = 8;
    const m = sim(cfg('h100-sxm', 8, 1, 'llama2-7b', 'zero-3', 32, 4, 2048));
    const expected = model.totalParams * 2 / N;
    expectClose(m.memoryPerGPU.parameters, expected, 0.01, 'ZeRO-3 params');
  });

  it('FSDP+TP (tp=8, dp=16): params = totalParams × 2 / (tp × dp)', () => {
    const model = getModel('llama2-70b')!;
    const tp = 8, dp = 16;
    const m = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'fsdp-tp', 256, 1, 2048, { tp }));
    const expected = model.totalParams * 2 / (tp * dp);
    expectClose(m.memoryPerGPU.parameters, expected, 0.01, 'FSDP+TP params');
  });

  it('ZeRO1+TP (tp=8, dp=16): params = totalParams × 2 / tp (ZeRO-1 doesn\'t shard params)', () => {
    const model = getModel('llama2-70b')!;
    const tp = 8;
    const m = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'zero1-tp', 256, 1, 2048, { tp }));
    const expected = model.totalParams * 2 / tp;
    expectClose(m.memoryPerGPU.parameters, expected, 0.01, 'ZeRO1+TP params');
  });

  it('DDP+TP+PP (tp=8, pp=2, dp=8): params = totalParams × 2 / (tp × pp)', () => {
    const model = getModel('llama2-70b')!;
    const tp = 8, pp = 2;
    const m = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'ddp-tp-pp', 128, 1, 2048, { tp, pp }));
    const expected = model.totalParams * 2 / (tp * pp);
    expectClose(m.memoryPerGPU.parameters, expected, 0.01, 'DDP+TP+PP params');
  });

  it('FSDP+TP+PP (tp=8, pp=2, dp=8): params = totalParams × 2 / (tp × pp × dp)', () => {
    const model = getModel('llama2-70b')!;
    const tp = 8, pp = 2, dp = 8;
    const m = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'fsdp-tp-pp', 128, 1, 2048, { tp, pp }));
    const expected = model.totalParams * 2 / (tp * pp * dp);
    expectClose(m.memoryPerGPU.parameters, expected, 0.01, 'FSDP+TP+PP params');
  });

  it('GPT-3 175B on 256x H100 with FSDP+TP+PP (tp=8, pp=4, dp=8)', () => {
    const model = getModel('gpt3-175b')!;
    const tp = 8, pp = 4, dp = 8;
    const m = sim(cfg('h100-sxm', 8, 32, 'gpt3-175b', 'fsdp-tp-pp', 128, 1, 2048, { tp, pp }));
    const expected = model.totalParams * 2 / (tp * pp * dp);
    expectClose(m.memoryPerGPU.parameters, expected, 0.01, 'GPT-3 175B params');
  });
});

// ---------------------------------------------------------------------------
// Section 2: Gradient Memory Formulas
// ---------------------------------------------------------------------------

describe('Gradient Memory Distribution', () => {
  it('DDP: full gradients — grads = totalParams × 2 (bf16)', () => {
    const model = getModel('llama2-7b')!;
    const m = sim(cfg('h100-sxm', 8, 1, 'llama2-7b', 'ddp', 32, 4, 2048));
    const expected = model.totalParams * 2; // bf16 gradients
    expectClose(m.memoryPerGPU.gradients, expected, 0.01, 'DDP grads');
  });

  it('FSDP: sharded grads = totalParams × 2 / N', () => {
    const model = getModel('llama2-7b')!;
    const N = 8;
    const m = sim(cfg('h100-sxm', 8, 1, 'llama2-7b', 'fsdp', 32, 4, 2048));
    const expected = model.totalParams * 2 / N; // bf16 gradients
    expectClose(m.memoryPerGPU.gradients, expected, 0.01, 'FSDP grads');
  });

  it('ZeRO-1: full grads = totalParams × 2', () => {
    const model = getModel('llama2-7b')!;
    const m = sim(cfg('h100-sxm', 8, 1, 'llama2-7b', 'zero-1', 32, 4, 2048));
    const expected = model.totalParams * 2; // bf16 gradients
    expectClose(m.memoryPerGPU.gradients, expected, 0.01, 'ZeRO-1 grads');
  });

  it('ZeRO-3: sharded grads = totalParams × 2 / N', () => {
    const model = getModel('llama2-7b')!;
    const N = 8;
    const m = sim(cfg('h100-sxm', 8, 1, 'llama2-7b', 'zero-3', 32, 4, 2048));
    const expected = model.totalParams * 2 / N; // bf16 gradients
    expectClose(m.memoryPerGPU.gradients, expected, 0.01, 'ZeRO-3 grads');
  });

  it('FSDP+TP (tp=8, dp=16): grads = totalParams × 2 / (tp × dp)', () => {
    const model = getModel('llama2-70b')!;
    const tp = 8, dp = 16;
    const m = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'fsdp-tp', 256, 1, 2048, { tp }));
    const expected = model.totalParams * 2 / (tp * dp); // bf16 gradients
    expectClose(m.memoryPerGPU.gradients, expected, 0.01, 'FSDP+TP grads');
  });

  it('DDP+TP+PP (tp=8, pp=2, dp=8): grads = totalParams × 2 / (tp × pp)', () => {
    const model = getModel('llama2-70b')!;
    const tp = 8, pp = 2;
    const m = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'ddp-tp-pp', 128, 1, 2048, { tp, pp }));
    const expected = model.totalParams * 2 / (tp * pp); // bf16 gradients
    expectClose(m.memoryPerGPU.gradients, expected, 0.01, 'DDP+TP+PP grads');
  });

  it('FSDP+TP+PP (tp=8, pp=2, dp=8): grads = totalParams × 2 / (tp × pp × dp)', () => {
    const model = getModel('llama2-70b')!;
    const tp = 8, pp = 2, dp = 8;
    const m = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'fsdp-tp-pp', 128, 1, 2048, { tp, pp }));
    const expected = model.totalParams * 2 / (tp * pp * dp); // bf16 gradients
    expectClose(m.memoryPerGPU.gradients, expected, 0.01, 'FSDP+TP+PP grads');
  });

  it('ZeRO1+TP (tp=8, dp=16): full grads / tp (ZeRO-1 doesn\'t shard grads)', () => {
    const model = getModel('llama2-70b')!;
    const tp = 8;
    const m = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'zero1-tp', 256, 1, 2048, { tp }));
    const expected = model.totalParams * 2 / tp; // bf16 gradients
    expectClose(m.memoryPerGPU.gradients, expected, 0.01, 'ZeRO1+TP grads');
  });
});

// ---------------------------------------------------------------------------
// Section 3: Optimizer State Formulas (AdamW = 12 bytes/param)
// ---------------------------------------------------------------------------

describe('Optimizer State Distribution', () => {
  it('DDP: full optimizer = totalParams × 12', () => {
    const model = getModel('llama2-7b')!;
    const m = sim(cfg('h100-sxm', 8, 1, 'llama2-7b', 'ddp', 32, 4, 2048));
    const expected = model.totalParams * 12;
    expectClose(m.memoryPerGPU.optimizerStates, expected, 0.01, 'DDP optimizer');
  });

  it('FSDP: sharded optimizer = totalParams × 12 / N', () => {
    const model = getModel('llama2-7b')!;
    const N = 8;
    const m = sim(cfg('h100-sxm', 8, 1, 'llama2-7b', 'fsdp', 32, 4, 2048));
    const expected = model.totalParams * 12 / N;
    expectClose(m.memoryPerGPU.optimizerStates, expected, 0.01, 'FSDP optimizer');
  });

  it('ZeRO-1: sharded optimizer = totalParams × 12 / N', () => {
    const model = getModel('llama2-7b')!;
    const N = 8;
    const m = sim(cfg('h100-sxm', 8, 1, 'llama2-7b', 'zero-1', 32, 4, 2048));
    const expected = model.totalParams * 12 / N;
    expectClose(m.memoryPerGPU.optimizerStates, expected, 0.01, 'ZeRO-1 optimizer');
  });

  it('ZeRO-3: sharded optimizer = totalParams × 12 / N', () => {
    const model = getModel('llama2-7b')!;
    const N = 8;
    const m = sim(cfg('h100-sxm', 8, 1, 'llama2-7b', 'zero-3', 32, 4, 2048));
    const expected = model.totalParams * 12 / N;
    expectClose(m.memoryPerGPU.optimizerStates, expected, 0.01, 'ZeRO-3 optimizer');
  });

  it('DDP+TP+PP (tp=8, pp=2): optimizer = totalParams × 12 / (tp × pp)', () => {
    const model = getModel('llama2-70b')!;
    const tp = 8, pp = 2;
    const m = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'ddp-tp-pp', 128, 1, 2048, { tp, pp }));
    const expected = model.totalParams * 12 / (tp * pp);
    expectClose(m.memoryPerGPU.optimizerStates, expected, 0.01, 'DDP+TP+PP optimizer');
  });

  it('FSDP+TP (tp=8, dp=16): optimizer = totalParams × 12 / (tp × dp)', () => {
    const model = getModel('llama2-70b')!;
    const tp = 8, dp = 16;
    const m = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'fsdp-tp', 256, 1, 2048, { tp }));
    const expected = model.totalParams * 12 / (tp * dp);
    expectClose(m.memoryPerGPU.optimizerStates, expected, 0.01, 'FSDP+TP optimizer');
  });

  it('FSDP+TP+PP (tp=8, pp=2, dp=8): optimizer = totalParams × 12 / (tp × pp × dp)', () => {
    const model = getModel('llama2-70b')!;
    const tp = 8, pp = 2, dp = 8;
    const m = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'fsdp-tp-pp', 128, 1, 2048, { tp, pp }));
    const expected = model.totalParams * 12 / (tp * pp * dp);
    expectClose(m.memoryPerGPU.optimizerStates, expected, 0.01, 'FSDP+TP+PP optimizer');
  });

  it('ZeRO1+TP (tp=8, dp=16): optimizer sharded by dp only = totalParams × 12 / (tp × dp)', () => {
    const model = getModel('llama2-70b')!;
    const tp = 8, dp = 16;
    const m = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'zero1-tp', 256, 1, 2048, { tp }));
    const expected = model.totalParams * 12 / (tp * dp);
    expectClose(m.memoryPerGPU.optimizerStates, expected, 0.01, 'ZeRO1+TP optimizer');
  });
});

// ---------------------------------------------------------------------------
// Section 4: FSDP Gather Buffers (the key section)
// ---------------------------------------------------------------------------

describe('FSDP Gather Buffers in 3D Parallel', () => {
  it('FSDP+TP: peakActivations includes gather buffers (70B, tp=8, dp=16)', () => {
    const model = getModel('llama2-70b')!;
    const tp = 8;
    const m = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'fsdp-tp', 256, 1, 2048, { tp }));

    // paramsPerLayer / tp × 2 bytes = gathered param buffer
    const paramsPerLayer = model.totalParams / model.numLayers;
    const gatheredParamBuffer = (paramsPerLayer / tp) * 2; // bf16

    // peakActivations should exceed activations by at least the param gather buffer
    expect(m.memoryPerGPU.peakActivations).toBeGreaterThan(
      m.memoryPerGPU.activations + gatheredParamBuffer * 0.9
    );
  });

  it('FSDP+TP: concrete gather buffer size (70B, tp=8)', () => {
    const model = getModel('llama2-70b')!;
    const tp = 8;
    const m = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'fsdp-tp', 256, 1, 2048, { tp }));

    // 70B / 80 layers = 875M params/layer
    // 875M / 8 (tp) × 2 (bf16) = 218.75 MB gather param buffer
    // 875M / 8 (tp) × 2 (bf16 grad) = 218.75 MB gather grad buffer
    // prefetch = 218.75 MB × 2 = 437.5 MB
    // total gather overhead = 218.75 + 218.75 + 437.5 = 875 MB
    const paramsPerLayer = model.totalParams / model.numLayers;
    const expectedGatherParam = (paramsPerLayer / tp) * 2;     // ~218.75 MB
    const expectedGatherGrad = (paramsPerLayer / tp) * 2;      // ~218.75 MB (bf16 grads)
    const expectedPrefetch = expectedGatherParam * 2;          // ~437.5 MB
    const expectedOverhead = expectedGatherParam + expectedGatherGrad + expectedPrefetch;

    const actualOverhead = m.memoryPerGPU.peakActivations - m.memoryPerGPU.activations;
    expectClose(actualOverhead, expectedOverhead, 0.02, 'FSDP+TP gather overhead');
  });

  it('FSDP+TP+PP: gather buffers present (70B, tp=8, pp=2)', () => {
    const model = getModel('llama2-70b')!;
    const tp = 8, pp = 2;
    const m = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'fsdp-tp-pp', 128, 1, 2048, { tp, pp }));

    const paramsPerLayer = model.totalParams / model.numLayers;
    const gatheredParamBuffer = (paramsPerLayer / tp) * 2;

    expect(m.memoryPerGPU.peakActivations).toBeGreaterThan(
      m.memoryPerGPU.activations + gatheredParamBuffer * 0.9
    );
  });

  it('DDP+TP+PP: NO gather buffers (peakActivations ≈ activations)', () => {
    const tp = 8, pp = 2;
    const m = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'ddp-tp-pp', 128, 1, 2048, { tp, pp }));

    // DDP doesn't need gather buffers — peakActivations should equal activations
    expect(m.memoryPerGPU.peakActivations).toBe(m.memoryPerGPU.activations);
  });

  it('ZeRO1+TP: NO gather buffers (peakActivations ≈ activations)', () => {
    const tp = 8;
    const m = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'zero1-tp', 256, 1, 2048, { tp }));

    // ZeRO-1 doesn't shard params — no gather needed
    expect(m.memoryPerGPU.peakActivations).toBe(m.memoryPerGPU.activations);
  });

  it('FSDP gather buffer scales with 1/tp — more TP → smaller gather', () => {
    // Compare tp=4 vs tp=8 on a smaller cluster
    const m4 = sim(cfg('h100-sxm', 8, 4, 'llama2-70b', 'fsdp-tp', 64, 1, 2048, { tp: 4 }));
    const m8 = sim(cfg('h100-sxm', 8, 4, 'llama2-70b', 'fsdp-tp', 64, 1, 2048, { tp: 8 }));

    const overhead4 = m4.memoryPerGPU.peakActivations - m4.memoryPerGPU.activations;
    const overhead8 = m8.memoryPerGPU.peakActivations - m8.memoryPerGPU.activations;

    // tp=8 should have ~half the gather overhead of tp=4
    const ratio = overhead8 / overhead4;
    expect(ratio).toBeGreaterThan(0.4);
    expect(ratio).toBeLessThan(0.6);
  });
});

// ---------------------------------------------------------------------------
// Section 5: Scaling Laws
// ---------------------------------------------------------------------------

describe('Memory Scaling Laws', () => {
  it('Doubling DP with FSDP halves static param/grad/optimizer memory', () => {
    // 8 GPUs (dp=1) vs 16 GPUs (dp=2) with FSDP+TP (tp=8)
    const m1 = sim(cfg('h100-sxm', 8, 1, 'llama2-70b', 'fsdp-tp', 32, 1, 2048, { tp: 8 }));
    const m2 = sim(cfg('h100-sxm', 8, 2, 'llama2-70b', 'fsdp-tp', 64, 1, 2048, { tp: 8 }));

    expectClose(m2.memoryPerGPU.parameters, m1.memoryPerGPU.parameters / 2, 0.01, 'Params halve with 2x DP');
    expectClose(m2.memoryPerGPU.gradients, m1.memoryPerGPU.gradients / 2, 0.01, 'Grads halve with 2x DP');
    expectClose(m2.memoryPerGPU.optimizerStates, m1.memoryPerGPU.optimizerStates / 2, 0.01, 'Optimizer halves with 2x DP');
  });

  it('Doubling TP halves static param/grad/optimizer memory', () => {
    // Use DDP-based strategy so DP doesn't affect param sharding
    // tp=4,pp=2,dp=4 on 32 GPUs vs tp=8,pp=2,dp=2 on 32 GPUs
    // DDP params sharded by tp×pp only → 8 vs 16 → should halve
    const m4 = sim(cfg('h100-sxm', 8, 4, 'llama2-70b', 'ddp-tp-pp', 64, 1, 2048, { tp: 4, pp: 2 }));
    const m8 = sim(cfg('h100-sxm', 8, 4, 'llama2-70b', 'ddp-tp-pp', 64, 1, 2048, { tp: 8, pp: 2 }));

    expectClose(m8.memoryPerGPU.parameters, m4.memoryPerGPU.parameters / 2, 0.01, 'Params halve with 2x TP');
    expectClose(m8.memoryPerGPU.gradients, m4.memoryPerGPU.gradients / 2, 0.01, 'Grads halve with 2x TP');
    expectClose(m8.memoryPerGPU.optimizerStates, m4.memoryPerGPU.optimizerStates / 2, 0.01, 'Optimizer halves with 2x TP');
  });

  it('Doubling PP halves static param/grad/optimizer memory', () => {
    // Use DDP-based strategy so DP doesn't affect param sharding
    // tp=8,pp=2,dp=8 on 128 GPUs vs tp=8,pp=4,dp=4 on 128 GPUs
    // DDP params sharded by tp×pp only → 16 vs 32 → should halve
    const m2 = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'ddp-tp-pp', 128, 1, 2048, { tp: 8, pp: 2 }));
    const m4 = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'ddp-tp-pp', 128, 1, 2048, { tp: 8, pp: 4 }));

    expectClose(m4.memoryPerGPU.parameters, m2.memoryPerGPU.parameters / 2, 0.01, 'Params halve with 2x PP');
    expectClose(m4.memoryPerGPU.gradients, m2.memoryPerGPU.gradients / 2, 0.01, 'Grads halve with 2x PP');
    expectClose(m4.memoryPerGPU.optimizerStates, m2.memoryPerGPU.optimizerStates / 2, 0.01, 'Optimizer halves with 2x PP');
  });

  it('FSDP+TP uses less total memory than DDP+TP+PP for same cluster', () => {
    const fsdp = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'fsdp-tp', 256, 1, 2048, { tp: 8 }));
    const ddp = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'ddp-tp-pp', 256, 1, 2048, { tp: 8, pp: 2 }));

    // FSDP shards params/grads/optimizer across DP — should use less static memory
    expect(fsdp.memoryPerGPU.parameters).toBeLessThan(ddp.memoryPerGPU.parameters);
    expect(fsdp.memoryPerGPU.optimizerStates).toBeLessThan(ddp.memoryPerGPU.optimizerStates);
  });

  it('FSDP gather buffer scales with 1/tp', () => {
    // Already tested in Section 4 — additional check with different model
    const m2 = sim(cfg('h100-sxm', 8, 2, 'llama3-8b', 'fsdp-tp', 32, 2, 2048, { tp: 2 }));
    const m4 = sim(cfg('h100-sxm', 8, 2, 'llama3-8b', 'fsdp-tp', 32, 2, 2048, { tp: 4 }));

    const overhead2 = m2.memoryPerGPU.peakActivations - m2.memoryPerGPU.activations;
    const overhead4 = m4.memoryPerGPU.peakActivations - m4.memoryPerGPU.activations;

    const ratio = overhead4 / overhead2;
    expect(ratio).toBeGreaterThan(0.4);
    expect(ratio).toBeLessThan(0.6);
  });
});

// ---------------------------------------------------------------------------
// Section 6: Edge Cases
// ---------------------------------------------------------------------------

describe('Memory Edge Cases', () => {
  it('Single GPU DDP: all memory = full model', () => {
    const model = getModel('llama3.2-1b')!;
    const m = sim(cfg('h100-sxm', 1, 1, 'llama3.2-1b', 'ddp', 8, 8, 2048));

    // Full replication on 1 GPU (bf16: 2B params, 2B grads, 12B optimizer)
    expectClose(m.memoryPerGPU.parameters, model.totalParams * 2, 0.01, 'Single GPU params');
    expectClose(m.memoryPerGPU.gradients, model.totalParams * 2, 0.01, 'Single GPU grads');
    expectClose(m.memoryPerGPU.optimizerStates, model.totalParams * 12, 0.01, 'Single GPU optimizer');
  });

  it('FSDP+TP with tp=1 dp=N should approximate pure FSDP values', () => {
    const fsdpTP = sim(cfg('h100-sxm', 8, 1, 'llama2-7b', 'fsdp-tp', 32, 4, 2048, { tp: 1 }));
    const pureFSDP = sim(cfg('h100-sxm', 8, 1, 'llama2-7b', 'fsdp', 32, 4, 2048));

    // Parameters should match within 5%
    expectClose(fsdpTP.memoryPerGPU.parameters, pureFSDP.memoryPerGPU.parameters, 0.05, 'tp=1 params vs pure FSDP');
    expectClose(fsdpTP.memoryPerGPU.gradients, pureFSDP.memoryPerGPU.gradients, 0.05, 'tp=1 grads vs pure FSDP');
    expectClose(fsdpTP.memoryPerGPU.optimizerStates, pureFSDP.memoryPerGPU.optimizerStates, 0.05, 'tp=1 optimizer vs pure FSDP');
  });

  it('MoE model: expert params sharded additionally by EP', () => {
    // Use DDP-based strategy so DP doesn't shard params (isolates EP effect)
    // tp=4, ep=1, dp=8 vs tp=4, ep=2, dp=4 — both 32 GPUs
    // With DDP, params sharded by tp×pp only for shared, tp×pp×ep for experts
    // So EP=2 should reduce expert param memory by half
    const mNoEP = sim(cfg('h100-sxm', 8, 4, 'mixtral-8x7b', 'ddp-tp-pp', 64, 1, 2048, { tp: 4, pp: 1, ep: 1 }));
    const mEP = sim(cfg('h100-sxm', 8, 4, 'mixtral-8x7b', 'ddp-tp-pp', 64, 1, 2048, { tp: 4, pp: 1, ep: 2 }));

    // With EP, expert params are further sharded — should use less parameter memory
    expect(mEP.memoryPerGPU.parameters).toBeLessThan(mNoEP.memoryPerGPU.parameters);
  });

  it('Very large model (175B+): no component negative or absurdly large', () => {
    const m = sim(cfg('h100-sxm', 8, 32, 'gpt3-175b', 'fsdp-tp-pp', 128, 1, 2048, { tp: 8, pp: 4 }));

    expect(m.memoryPerGPU.parameters).toBeGreaterThan(0);
    expect(m.memoryPerGPU.gradients).toBeGreaterThan(0);
    expect(m.memoryPerGPU.optimizerStates).toBeGreaterThan(0);
    expect(m.memoryPerGPU.activations).toBeGreaterThan(0);
    expect(m.memoryPerGPU.peakActivations).toBeGreaterThanOrEqual(m.memoryPerGPU.activations);
    expect(m.memoryPerGPU.temporary).toBeGreaterThan(0);
    expect(m.memoryPerGPU.reserved).toBeGreaterThan(0);

    // No single component should exceed 100 GB per GPU
    const maxPerComponent = 100e9;
    expect(m.memoryPerGPU.parameters).toBeLessThan(maxPerComponent);
    expect(m.memoryPerGPU.gradients).toBeLessThan(maxPerComponent);
    expect(m.memoryPerGPU.optimizerStates).toBeLessThan(maxPerComponent);
    expect(m.memoryPerGPU.peakActivations).toBeLessThan(maxPerComponent);
  });

  it('Memory total equals sum of components', () => {
    const m = sim(cfg('h100-sxm', 8, 16, 'llama2-70b', 'fsdp-tp', 256, 1, 2048, { tp: 8 }));

    const computedTotal = m.memoryPerGPU.parameters
      + m.memoryPerGPU.gradients
      + m.memoryPerGPU.optimizerStates
      + m.memoryPerGPU.peakActivations
      + m.memoryPerGPU.temporary
      + m.memoryPerGPU.reserved;

    expectClose(m.memoryPerGPU.total, computedTotal, 0.001, 'Total = sum of components');
  });
});
