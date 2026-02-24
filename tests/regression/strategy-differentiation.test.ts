/**
 * Strategy Differentiation Tests
 *
 * Ensures every strategy produces distinct, correct results.
 * Catches bugs where two strategies accidentally produce identical output
 * (e.g. fsdp-tp defaulting sequenceParallel=true when it shouldn't).
 *
 * Tests verify:
 * 1. Every pair of strategies on the same config produces different metrics
 * 2. Each strategy's unique properties are reflected in the output
 * 3. Strategy-specific invariants hold (sharding, memory ordering, comm patterns)
 */
import { describe, it, expect } from 'vitest';
import {
  type SimulationConfig,
  type SimulationMetrics,
} from '../../src/core/simulation/engine.ts';
import { getSimulationMetrics } from '../../src/core/simulation/engine.ts';
import { createMultiNodeCluster } from '../../src/core/hardware/topology.ts';
import { getModel } from '../../src/core/models/index.ts';
import { getModelConfig } from '../../src/core/models/architectures.ts';
import { buildModelSpec } from '../../src/core/models/primitives.ts';
import { FSDPStrategy } from '../../src/core/strategies/fsdp.ts';
import { ZeROStrategy } from '../../src/core/strategies/zero.ts';
import { create3DParallelStrategy } from '../../src/core/strategies/index.ts';
import { type StrategyContext } from '../../src/core/strategies/base.ts';
import { DEFAULT_ADAMW_CONFIG, DEFAULT_LR_SCHEDULE, DEFAULT_DTYPE_CONFIG } from '../../src/types/index.ts';
import { makeStrategyContext } from '../helpers/strategy-context.ts';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function cfg(
  modelId: string,
  strategy: SimulationConfig['strategyType'],
  numNodes: number,
  strategyConfig?: SimulationConfig['strategyConfig'],
): SimulationConfig {
  return {
    clusterConfig: createMultiNodeCluster('h100-sxm', 8, numNodes)!,
    modelId,
    globalBatchSize: 128,
    microBatchSize: 2,
    sequenceLength: 2048,
    strategyType: strategy,
    strategyConfig,
  };
}

function sim(config: SimulationConfig): SimulationMetrics {
  return getSimulationMetrics(config);
}

// ---------------------------------------------------------------------------
// Section 1: Pairwise uniqueness — no two strategies produce the same output
// ---------------------------------------------------------------------------

describe('Pairwise Strategy Uniqueness', () => {
  // 1D strategies on 8 GPUs (1 node)
  describe('1D strategies (8 GPUs, Llama 7B)', () => {
    const strats: SimulationConfig['strategyType'][] = ['ddp', 'zero-1', 'fsdp'];
    const results = new Map<string, SimulationMetrics>();

    // Pre-compute all results
    for (const s of strats) {
      results.set(s, sim(cfg('llama2-7b', s, 1)));
    }

    for (let i = 0; i < strats.length; i++) {
      for (let j = i + 1; j < strats.length; j++) {
        const a = strats[i], b = strats[j];
        it(`${a} differs from ${b} in at least memory or timing`, () => {
          const ma = results.get(a)!;
          const mb = results.get(b)!;

          const memDiffers =
            ma.memoryPerGPU.parameters !== mb.memoryPerGPU.parameters ||
            ma.memoryPerGPU.gradients !== mb.memoryPerGPU.gradients ||
            ma.memoryPerGPU.optimizerStates !== mb.memoryPerGPU.optimizerStates;
          const timeDiffers = ma.stepTimeMs !== mb.stepTimeMs;
          const mfuDiffers = ma.mfu !== mb.mfu;

          expect(
            memDiffers || timeDiffers || mfuDiffers,
            `${a} and ${b} produced identical results — they should differ`
          ).toBe(true);
        });
      }
    }
  });

  // 2D+3D strategies on 128 GPUs (16 nodes), 70B model
  describe('Hybrid strategies (128 GPUs, Llama 70B)', () => {
    const stratConfigs: [SimulationConfig['strategyType'], SimulationConfig['strategyConfig']][] = [
      ['fsdp-tp', { tp: 8 }],
      ['zero1-tp', { tp: 8 }],
      ['ddp-tp-pp', { tp: 8, pp: 2 }],
      ['zero1-tp-pp', { tp: 8, pp: 2 }],
      ['fsdp-tp-pp', { tp: 8, pp: 2 }],
    ];

    const results = new Map<string, SimulationMetrics>();
    for (const [s, sc] of stratConfigs) {
      results.set(s, sim(cfg('llama2-70b', s, 16, sc)));
    }

    for (let i = 0; i < stratConfigs.length; i++) {
      for (let j = i + 1; j < stratConfigs.length; j++) {
        const [a] = stratConfigs[i], [b] = stratConfigs[j];
        it(`${a} differs from ${b}`, () => {
          const ma = results.get(a)!;
          const mb = results.get(b)!;

          // Check multiple dimensions of difference
          const paramsDiff = ma.memoryPerGPU.parameters !== mb.memoryPerGPU.parameters;
          const gradsDiff = ma.memoryPerGPU.gradients !== mb.memoryPerGPU.gradients;
          const optDiff = ma.memoryPerGPU.optimizerStates !== mb.memoryPerGPU.optimizerStates;
          const actDiff = ma.memoryPerGPU.activations !== mb.memoryPerGPU.activations;
          const peakActDiff = ma.memoryPerGPU.peakActivations !== mb.memoryPerGPU.peakActivations;
          const timeDiff = ma.stepTimeMs !== mb.stepTimeMs;
          const mfuDiff = ma.mfu !== mb.mfu;

          expect(
            paramsDiff || gradsDiff || optDiff || actDiff || peakActDiff || timeDiff || mfuDiff,
            `${a} and ${b} produced identical results — they should differ`
          ).toBe(true);
        });
      }
    }
  });
});

// ---------------------------------------------------------------------------
// Section 2: Strategy-specific properties
// ---------------------------------------------------------------------------

describe('DDP-specific properties', () => {
  const model = getModel('llama2-7b')!;
  const m = sim(cfg('llama2-7b', 'ddp', 1));

  it('full parameter replication', () => {
    expect(m.memoryPerGPU.parameters).toBeCloseTo(model.totalParams * 2, -6); // bf16
  });

  it('full gradient replication (bf16)', () => {
    expect(m.memoryPerGPU.gradients).toBeCloseTo(model.totalParams * 2, -6);
  });

  it('full optimizer on each GPU', () => {
    expect(m.memoryPerGPU.optimizerStates).toBeCloseTo(model.totalParams * 12, -6);
  });

  it('no pipeline bubble', () => {
    expect(m.pipelineBubble).toBe(0);
  });
});

describe('FSDP-specific properties', () => {
  const model = getModel('llama2-7b')!;
  const N = 8;
  const m = sim(cfg('llama2-7b', 'fsdp', 1));

  it('params sharded across N GPUs', () => {
    expect(m.memoryPerGPU.parameters).toBeCloseTo(model.totalParams * 2 / N, -6);
  });

  it('grads sharded across N GPUs', () => {
    expect(m.memoryPerGPU.gradients).toBeCloseTo(model.totalParams * 2 / N, -6);
  });

  it('optimizer sharded across N GPUs', () => {
    expect(m.memoryPerGPU.optimizerStates).toBeCloseTo(model.totalParams * 12 / N, -6);
  });

  it('peakActivations > activations (gather buffers)', () => {
    expect(m.memoryPerGPU.peakActivations).toBeGreaterThan(m.memoryPerGPU.activations);
  });
});

describe('ZeRO-1-specific properties', () => {
  const model = getModel('llama2-7b')!;
  const N = 8;
  const m = sim(cfg('llama2-7b', 'zero-1', 1));

  it('full parameter replication (like DDP)', () => {
    expect(m.memoryPerGPU.parameters).toBeCloseTo(model.totalParams * 2, -6);
  });

  it('full gradient replication (like DDP)', () => {
    expect(m.memoryPerGPU.gradients).toBeCloseTo(model.totalParams * 2, -6);
  });

  it('optimizer sharded (unlike DDP)', () => {
    expect(m.memoryPerGPU.optimizerStates).toBeCloseTo(model.totalParams * 12 / N, -6);
  });
});

describe('ZeRO-3-specific properties', () => {
  const model = getModel('llama2-7b')!;
  const N = 8;
  const m = sim(cfg('llama2-7b', 'zero-3', 1));

  it('params sharded (like FSDP)', () => {
    expect(m.memoryPerGPU.parameters).toBeCloseTo(model.totalParams * 2 / N, -6);
  });

  it('grads sharded', () => {
    expect(m.memoryPerGPU.gradients).toBeCloseTo(model.totalParams * 2 / N, -6);
  });

  it('optimizer sharded', () => {
    expect(m.memoryPerGPU.optimizerStates).toBeCloseTo(model.totalParams * 12 / N, -6);
  });
});

// ---------------------------------------------------------------------------
// Section 3: Sequence Parallelism toggle must produce different results
// ---------------------------------------------------------------------------

describe('Sequence Parallelism on vs off', () => {
  const noSP = sim(cfg('llama2-70b', 'fsdp-tp', 16, { tp: 8, sequenceParallel: false }));
  const withSP = sim(cfg('llama2-70b', 'fsdp-tp', 16, { tp: 8, sequenceParallel: true }));

  it('SP=true has lower activation memory', () => {
    expect(withSP.memoryPerGPU.activations).toBeLessThan(noSP.memoryPerGPU.activations);
  });

  it('different step time', () => {
    expect(withSP.stepTimeMs).not.toEqual(noSP.stepTimeMs);
  });

  it('different MFU', () => {
    expect(withSP.mfu).not.toEqual(noSP.mfu);
  });

  it('different communication overhead', () => {
    expect(withSP.communicationOverhead).not.toEqual(noSP.communicationOverhead);
  });

  it('same parameter memory (SP does not change param sharding)', () => {
    expect(withSP.memoryPerGPU.parameters).toEqual(noSP.memoryPerGPU.parameters);
  });

  it('same gradient memory (SP does not change grad sharding)', () => {
    expect(withSP.memoryPerGPU.gradients).toEqual(noSP.memoryPerGPU.gradients);
  });

  it('same optimizer memory (SP does not change optimizer sharding)', () => {
    expect(withSP.memoryPerGPU.optimizerStates).toEqual(noSP.memoryPerGPU.optimizerStates);
  });

  it('SP activation reduction scales with TP degree', () => {
    const tp4noSP = sim(cfg('llama2-70b', 'fsdp-tp', 4, { tp: 4, sequenceParallel: false }));
    const tp4SP = sim(cfg('llama2-70b', 'fsdp-tp', 4, { tp: 4, sequenceParallel: true }));
    const tp2noSP = sim(cfg('llama2-70b', 'fsdp-tp', 2, { tp: 2, sequenceParallel: false }));
    const tp2SP = sim(cfg('llama2-70b', 'fsdp-tp', 2, { tp: 2, sequenceParallel: true }));

    const ratio8 = withSP.memoryPerGPU.activations / noSP.memoryPerGPU.activations;
    const ratio4 = tp4SP.memoryPerGPU.activations / tp4noSP.memoryPerGPU.activations;
    const ratio2 = tp2SP.memoryPerGPU.activations / tp2noSP.memoryPerGPU.activations;

    // Higher TP → lower ratio (more SP benefit)
    expect(ratio8).toBeLessThan(ratio4);
    expect(ratio4).toBeLessThan(ratio2);
    // All should be less than 1
    expect(ratio8).toBeLessThan(1);
    expect(ratio4).toBeLessThan(1);
    expect(ratio2).toBeLessThan(1);
  });
});

// ---------------------------------------------------------------------------
// Section 4: DDP-based vs FSDP-based must show sharding difference
// ---------------------------------------------------------------------------

describe('DDP-based vs FSDP-based 3D strategies', () => {
  const ddp3d = sim(cfg('llama2-70b', 'ddp-tp-pp', 16, { tp: 8, pp: 2 }));
  const fsdp3d = sim(cfg('llama2-70b', 'fsdp-tp-pp', 16, { tp: 8, pp: 2 }));

  it('FSDP shards params, DDP does not — FSDP params < DDP params', () => {
    expect(fsdp3d.memoryPerGPU.parameters).toBeLessThan(ddp3d.memoryPerGPU.parameters);
  });

  it('FSDP shards grads, DDP does not — FSDP grads < DDP grads', () => {
    expect(fsdp3d.memoryPerGPU.gradients).toBeLessThan(ddp3d.memoryPerGPU.gradients);
  });

  it('FSDP shards optimizer, DDP does not — FSDP optimizer < DDP optimizer', () => {
    expect(fsdp3d.memoryPerGPU.optimizerStates).toBeLessThan(ddp3d.memoryPerGPU.optimizerStates);
  });

  it('FSDP has gather buffers (peakActivations > activations), DDP does not', () => {
    expect(fsdp3d.memoryPerGPU.peakActivations).toBeGreaterThan(fsdp3d.memoryPerGPU.activations);
    expect(ddp3d.memoryPerGPU.peakActivations).toEqual(ddp3d.memoryPerGPU.activations);
  });

  it('FSDP total memory < DDP total memory (sharding outweighs gather overhead)', () => {
    expect(fsdp3d.memoryPerGPU.total).toBeLessThan(ddp3d.memoryPerGPU.total);
  });
});

describe('ZeRO-1-based vs DDP-based 3D strategies', () => {
  const ddp3d = sim(cfg('llama2-70b', 'ddp-tp-pp', 16, { tp: 8, pp: 2 }));
  const z1_3d = sim(cfg('llama2-70b', 'zero1-tp-pp', 16, { tp: 8, pp: 2 }));

  it('same param memory (ZeRO-1 does not shard params)', () => {
    expect(z1_3d.memoryPerGPU.parameters).toEqual(ddp3d.memoryPerGPU.parameters);
  });

  it('same gradient memory (ZeRO-1 does not shard grads)', () => {
    expect(z1_3d.memoryPerGPU.gradients).toEqual(ddp3d.memoryPerGPU.gradients);
  });

  it('ZeRO-1 shards optimizer — lower than DDP', () => {
    expect(z1_3d.memoryPerGPU.optimizerStates).toBeLessThan(ddp3d.memoryPerGPU.optimizerStates);
  });

  it('neither has gather buffers', () => {
    expect(ddp3d.memoryPerGPU.peakActivations).toEqual(ddp3d.memoryPerGPU.activations);
    expect(z1_3d.memoryPerGPU.peakActivations).toEqual(z1_3d.memoryPerGPU.activations);
  });

  it('ZeRO-1 lower total memory (optimizer sharding)', () => {
    expect(z1_3d.memoryPerGPU.total).toBeLessThan(ddp3d.memoryPerGPU.total);
  });
});

// ---------------------------------------------------------------------------
// Section 5: 2D vs 3D — adding PP changes pipeline bubble
// ---------------------------------------------------------------------------

describe('2D vs 3D (adding Pipeline Parallelism)', () => {
  const fsdpTP = sim(cfg('llama2-70b', 'fsdp-tp', 16, { tp: 8 }));
  const fsdpTPPP = sim(cfg('llama2-70b', 'fsdp-tp-pp', 16, { tp: 8, pp: 2 }));

  it('3D strategy has pipeline bubble > 0', () => {
    expect(fsdpTPPP.pipelineBubble).toBeGreaterThan(0);
  });

  it('2D strategy has no pipeline bubble', () => {
    expect(fsdpTP.pipelineBubble).toBe(0);
  });

  it('same param memory when total sharding is equal', () => {
    // FSDP+TP: sharding = tp*dp = 8*16 = 128
    // FSDP+TP+PP pp=2: sharding = tp*pp*dp = 8*2*8 = 128
    // Same total sharding on 128 GPUs, so param memory is the same
    expect(fsdpTPPP.memoryPerGPU.parameters).toEqual(fsdpTP.memoryPerGPU.parameters);
  });

  it('different step times', () => {
    expect(fsdpTPPP.stepTimeMs).not.toEqual(fsdpTP.stepTimeMs);
  });

  it('different activation memory (PP has in-flight micro-batches)', () => {
    expect(fsdpTPPP.memoryPerGPU.activations).not.toEqual(fsdpTP.memoryPerGPU.activations);
  });
});

// ---------------------------------------------------------------------------
// Section 6: Cross-strategy memory ordering invariants
// ---------------------------------------------------------------------------

describe('Memory ordering invariants (8 GPUs, Llama 7B)', () => {
  const ddp = sim(cfg('llama2-7b', 'ddp', 1));
  const fsdp = sim(cfg('llama2-7b', 'fsdp', 1));
  const z1 = sim(cfg('llama2-7b', 'zero-1', 1));
  const z3 = sim(cfg('llama2-7b', 'zero-3', 1));

  it('FSDP params < DDP params (sharding)', () => {
    expect(fsdp.memoryPerGPU.parameters).toBeLessThan(ddp.memoryPerGPU.parameters);
  });

  it('ZeRO-3 params < DDP params (sharding)', () => {
    expect(z3.memoryPerGPU.parameters).toBeLessThan(ddp.memoryPerGPU.parameters);
  });

  it('ZeRO-1 params == DDP params (no param sharding)', () => {
    expect(z1.memoryPerGPU.parameters).toEqual(ddp.memoryPerGPU.parameters);
  });

  it('ZeRO-1 optimizer < DDP optimizer (optimizer sharding)', () => {
    expect(z1.memoryPerGPU.optimizerStates).toBeLessThan(ddp.memoryPerGPU.optimizerStates);
  });

  it('ZeRO-1 grads == DDP grads (no grad sharding)', () => {
    expect(z1.memoryPerGPU.gradients).toEqual(ddp.memoryPerGPU.gradients);
  });

  it('FSDP total < DDP total (full sharding wins)', () => {
    expect(fsdp.memoryPerGPU.total).toBeLessThan(ddp.memoryPerGPU.total);
  });

  it('ZeRO-3 total < DDP total', () => {
    expect(z3.memoryPerGPU.total).toBeLessThan(ddp.memoryPerGPU.total);
  });

  it('ZeRO-1 total < DDP total (optimizer sharding)', () => {
    expect(z1.memoryPerGPU.total).toBeLessThan(ddp.memoryPerGPU.total);
  });

  it('DDP has the highest total memory of all strategies', () => {
    expect(ddp.memoryPerGPU.total).toBeGreaterThan(fsdp.memoryPerGPU.total);
    expect(ddp.memoryPerGPU.total).toBeGreaterThan(z1.memoryPerGPU.total);
    expect(ddp.memoryPerGPU.total).toBeGreaterThan(z3.memoryPerGPU.total);
  });
});

describe('Memory ordering invariants (128 GPUs, Llama 70B)', () => {
  const ddp3d = sim(cfg('llama2-70b', 'ddp-tp-pp', 16, { tp: 8, pp: 2 }));
  const z1_3d = sim(cfg('llama2-70b', 'zero1-tp-pp', 16, { tp: 8, pp: 2 }));
  const fsdp3d = sim(cfg('llama2-70b', 'fsdp-tp-pp', 16, { tp: 8, pp: 2 }));
  const fsdpTP = sim(cfg('llama2-70b', 'fsdp-tp', 16, { tp: 8 }));
  const z1TP = sim(cfg('llama2-70b', 'zero1-tp', 16, { tp: 8 }));

  it('FSDP-based 3D has lowest static memory (params + grads + optimizer)', () => {
    const static3d = fsdp3d.memoryPerGPU.parameters + fsdp3d.memoryPerGPU.gradients + fsdp3d.memoryPerGPU.optimizerStates;
    const staticDDP = ddp3d.memoryPerGPU.parameters + ddp3d.memoryPerGPU.gradients + ddp3d.memoryPerGPU.optimizerStates;
    const staticZ1 = z1_3d.memoryPerGPU.parameters + z1_3d.memoryPerGPU.gradients + z1_3d.memoryPerGPU.optimizerStates;
    expect(static3d).toBeLessThan(staticDDP);
    expect(static3d).toBeLessThan(staticZ1);
  });

  it('ZeRO-1+TP params > FSDP+TP params (ZeRO-1 does not shard params)', () => {
    expect(z1TP.memoryPerGPU.parameters).toBeGreaterThan(fsdpTP.memoryPerGPU.parameters);
  });

  it('ZeRO-1+TP grads > FSDP+TP grads', () => {
    expect(z1TP.memoryPerGPU.gradients).toBeGreaterThan(fsdpTP.memoryPerGPU.gradients);
  });
});

// ---------------------------------------------------------------------------
// Section 7: Sequence parallelism effect on all TP-based strategies
// ---------------------------------------------------------------------------

describe('Sequence Parallelism explicitly changes behavior', () => {
  it('fsdp-tp with explicit SP=false differs from SP=true', () => {
    const noSP = sim(cfg('llama2-70b', 'fsdp-tp', 16, { tp: 8, sequenceParallel: false }));
    const withSP = sim(cfg('llama2-70b', 'fsdp-tp', 16, { tp: 8, sequenceParallel: true }));

    expect(noSP.memoryPerGPU.activations).not.toEqual(withSP.memoryPerGPU.activations);
    expect(noSP.stepTimeMs).not.toEqual(withSP.stepTimeMs);
  });

  it('3D strategies with explicit SP=false differ from SP=true', () => {
    const noSP = sim(cfg('llama2-70b', 'fsdp-tp-pp', 16, { tp: 8, pp: 2, sequenceParallel: false }));
    const withSP = sim(cfg('llama2-70b', 'fsdp-tp-pp', 16, { tp: 8, pp: 2, sequenceParallel: true }));

    expect(noSP.memoryPerGPU.activations).not.toEqual(withSP.memoryPerGPU.activations);
    expect(withSP.memoryPerGPU.activations).toBeLessThan(noSP.memoryPerGPU.activations);
  });
});

// ---------------------------------------------------------------------------
// Section 8: Each strategy has correct comm patterns
// ---------------------------------------------------------------------------

describe('Communication pattern correctness', () => {
  it('DDP has only data-parallel communication', () => {
    const m = sim(cfg('llama2-7b', 'ddp', 1));
    // DDP should have comm overhead > 0 (gradient allreduce)
    expect(m.timing.communication).toBeGreaterThan(0);
  });

  it('FSDP has higher communication than DDP (allgather + reduce-scatter)', () => {
    const ddp = sim(cfg('llama2-7b', 'ddp', 1));
    const fsdp = sim(cfg('llama2-7b', 'fsdp', 1));
    expect(fsdp.timing.communication).toBeGreaterThan(ddp.timing.communication);
  });

  it('3D strategies have non-zero communication', () => {
    const strats: [SimulationConfig['strategyType'], SimulationConfig['strategyConfig']][] = [
      ['fsdp-tp', { tp: 8 }],
      ['zero1-tp', { tp: 8 }],
      ['ddp-tp-pp', { tp: 8, pp: 2 }],
      ['zero1-tp-pp', { tp: 8, pp: 2 }],
      ['fsdp-tp-pp', { tp: 8, pp: 2 }],
    ];

    for (const [s, sc] of strats) {
      const m = sim(cfg('llama2-70b', s, 16, sc));
      expect(m.timing.communication, `${s} should have communication > 0`).toBeGreaterThan(0);
    }
  });
});

// ---------------------------------------------------------------------------
// Section 9: MFU sanity for all strategies
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Section 9.5: FSDP ≡ ZeRO-3 numerical equivalence
// ---------------------------------------------------------------------------

describe('FSDP ≡ ZeRO-3 numerical equivalence', () => {
  const fsdp = new FSDPStrategy();
  const zero3 = new ZeROStrategy({ stage: 3 });

  function makeCtx(modelId: string, numNodes: number) {
    return makeStrategyContext(modelId, '', {
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, numNodes)!,
      globalBatchSize: 128,
      microBatchSize: 2,
    });
  }

  function pctDiff(a: number, b: number): number {
    return Math.abs(a - b) / Math.max(a, b);
  }

  const configs: [string, string, number][] = [
    ['Llama 7B, 8 GPUs (1 node)', 'llama2-7b', 1],
    ['Llama 70B, 64 GPUs (8 nodes)', 'llama2-70b', 8],
  ];

  for (const [label, modelId, numNodes] of configs) {
    describe(label, () => {
      const ctx = makeCtx(modelId, numNodes);
      const fsdpMem = fsdp.computeMemoryPerGPU(ctx);
      const z3Mem = zero3.computeMemoryPerGPU(ctx);
      const fsdpTime = fsdp.computeTiming(ctx);
      const z3Time = zero3.computeTiming(ctx);
      const fsdpAnalysis = fsdp.computeAnalysis(ctx);
      const z3Analysis = zero3.computeAnalysis(ctx);

      it('total memory within ±5%', () => {
        expect(pctDiff(fsdpMem.total, z3Mem.total)).toBeLessThan(0.05);
      });

      it('peak activations within ±5%', () => {
        expect(pctDiff(fsdpMem.peakActivations, z3Mem.peakActivations)).toBeLessThan(0.05);
      });

      it('temporary buffers within ±5%', () => {
        expect(pctDiff(fsdpMem.temporary, z3Mem.temporary)).toBeLessThan(0.05);
      });

      it('step time within ±5%', () => {
        expect(pctDiff(fsdpTime.total, z3Time.total)).toBeLessThan(0.05);
      });

      it('MFU within ±5%', () => {
        expect(pctDiff(fsdpAnalysis.mfu, z3Analysis.mfu)).toBeLessThan(0.05);
      });

      it('communication time within ±5%', () => {
        expect(pctDiff(fsdpTime.communication, z3Time.communication)).toBeLessThan(0.05);
      });

      it('overlap within ±5%', () => {
        expect(pctDiff(fsdpTime.overlap, z3Time.overlap)).toBeLessThan(0.05);
      });
    });
  }
});

describe('MFU sanity across all strategies', () => {
  const allStrats: [SimulationConfig['strategyType'], string, number, SimulationConfig['strategyConfig']?][] = [
    ['ddp', 'llama2-7b', 1],
    ['fsdp', 'llama2-7b', 1],
    ['zero-1', 'llama2-7b', 1],
    ['zero-3', 'llama2-7b', 1],
    ['fsdp-tp', 'llama2-70b', 16, { tp: 8 }],
    ['zero1-tp', 'llama2-70b', 16, { tp: 8 }],
    ['ddp-tp-pp', 'llama2-70b', 16, { tp: 8, pp: 2 }],
    ['zero1-tp-pp', 'llama2-70b', 16, { tp: 8, pp: 2 }],
    ['fsdp-tp-pp', 'llama2-70b', 16, { tp: 8, pp: 2 }],
  ];

  for (const [strat, model, nodes, sc] of allStrats) {
    it(`${strat}: MFU in (0, 1) range`, () => {
      const m = sim(cfg(model, strat, nodes, sc));
      expect(m.mfu, `${strat} MFU should be > 0`).toBeGreaterThan(0);
      expect(m.mfu, `${strat} MFU should be <= 1`).toBeLessThanOrEqual(1);
    });

    it(`${strat}: HFU >= MFU (checkpointing adds recompute)`, () => {
      const m = sim(cfg(model, strat, nodes, sc));
      expect(m.hfu).toBeGreaterThanOrEqual(m.mfu);
    });

    it(`${strat}: total memory = sum of components`, () => {
      const m = sim(cfg(model, strat, nodes, sc));
      const computed = m.memoryPerGPU.parameters
        + m.memoryPerGPU.gradients
        + m.memoryPerGPU.optimizerStates
        + m.memoryPerGPU.peakActivations
        + m.memoryPerGPU.temporary
        + m.memoryPerGPU.reserved;
      const ratio = m.memoryPerGPU.total / computed;
      expect(ratio).toBeGreaterThan(0.999);
      expect(ratio).toBeLessThan(1.001);
    });
  }
});

// ---------------------------------------------------------------------------
// Section: MoE numActiveExperts effects
// ---------------------------------------------------------------------------

describe('MoE numActiveExperts effects', () => {
  // Build Scout variants with different numActiveExperts
  const scoutConfig = getModelConfig('llama4-scout')!;
  const seqLen = 4096;

  function scoutVariant(numActive: number) {
    return buildModelSpec({ ...scoutConfig, numActiveExperts: numActive }, seqLen);
  }

  function makeCtx(numActive: number, dp: number): StrategyContext {
    const model = scoutVariant(numActive);
    const cluster = createMultiNodeCluster('h100-sxm', 8, dp)!;
    const gbs = 128;
    const mbs = 1;
    const ga = Math.ceil(gbs / (mbs * dp));
    return {
      model, cluster,
      training: {
        globalBatchSize: gbs, microBatchSize: mbs, sequenceLength: seqLen,
        maxSteps: 1000, optimizer: DEFAULT_ADAMW_CONFIG, lrSchedule: DEFAULT_LR_SCHEDULE,
        dtypes: DEFAULT_DTYPE_CONFIG, gradientClipping: 1.0, gradientAccumulationSteps: ga,
      },
      seqLength: seqLen, microBatchSize: mbs, globalBatchSize: gbs, gradientAccumulationSteps: ga,
      activationCheckpointing: true, flashAttention: true,
    };
  }

  it('All-to-all scales with numActiveExperts (EP>1)', () => {
    // Scout with numActive=1 vs numActive=2, EP=4
    const ctx1 = makeCtx(1, 4);
    const ctx2 = makeCtx(2, 4);
    const strat = create3DParallelStrategy(8, 1, 4, { ep: 4, dpType: 'fsdp' });

    const comm1 = strat.computeCommunication(ctx1);
    const comm2 = strat.computeCommunication(ctx2);

    // Routing locality model: volume ∝ numActive × routingLocality, where
    //   routingLocality = 1 / (1 + expertsPerRank / numActive)
    // This models learned MoE routers preferentially selecting local experts.
    // Higher density (expertsPerRank/numActive) → more local selections → less cross-rank traffic.
    //
    // Scout: 16 experts, EP=4 → expertsPerRank = 16/4 = 4
    //   numActive=1: routingLocality = 1/(1 + 4/1) = 0.20  → effective = 1 × 0.20 = 0.20
    //   numActive=2: routingLocality = 1/(1 + 4/2) = 0.333 → effective = 2 × 0.333 = 0.667
    //   Ratio = 0.667 / 0.20 = 3.333
    //
    // The ratio exceeds 2× because increasing numActive has two compounding effects:
    // (1) more experts selected per token (linear), and (2) higher routingLocality because
    // local supply (4 experts/rank) satisfies a smaller fraction of the increased demand,
    // so more selections must cross EP boundaries.
    const ratio = comm2.expertParallel / comm1.expertParallel;
    expect(ratio).toBeGreaterThan(3.2);
    expect(ratio).toBeLessThan(3.5);
  });

  it('EP=1 expertParallel always zero regardless of numActiveExperts', () => {
    const ctx1 = makeCtx(1, 1);
    const ctx2 = makeCtx(2, 1);
    const strat = create3DParallelStrategy(8, 1, 1, { ep: 1, dpType: 'fsdp' });

    expect(strat.computeCommunication(ctx1).expertParallel).toBe(0);
    expect(strat.computeCommunication(ctx2).expertParallel).toBe(0);
  });

  it('MFU increases with numActiveExperts (more compute amortizes comm)', () => {
    // With EP=1 (no all-to-all), more active experts = more compute = better MFU
    const m1 = sim({
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 1)!,
      modelSpec: scoutVariant(1),
      globalBatchSize: 32, microBatchSize: 1, sequenceLength: seqLen,
      strategyType: 'fsdp',
    });
    const m2 = sim({
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 1)!,
      modelSpec: scoutVariant(2),
      globalBatchSize: 32, microBatchSize: 1, sequenceLength: seqLen,
      strategyType: 'fsdp',
    });
    // More active experts → more useful FLOPs per step → higher MFU
    expect(m2.mfu).toBeGreaterThan(m1.mfu);
  });

  it('activeParams ratio reflects numActiveExperts change', () => {
    // Scout has 1 shared expert. numActive=1 → 2 active MLPs, numActive=2 → 3 active MLPs
    // Ratio of active params should be > 1 (more active experts = more active params)
    const model1 = scoutVariant(1);
    const model2 = scoutVariant(2);
    const ratio = model2.activeParams / model1.activeParams;
    // With 1 shared + 1 routed vs 1 shared + 2 routed, MoE MLP portion grows
    expect(ratio).toBeGreaterThan(1.1);
    expect(ratio).toBeLessThan(1.8);
  });
});
