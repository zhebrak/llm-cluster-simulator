/**
 * Sequence Parallelism (SP) Validation Tests
 *
 * Verifies that the SP checkbox produces correct, measurable effects
 * on every 2D and 3D strategy. Tests validate actual numbers against
 * the known SP model:
 *
 * Activation memory:
 *   - With SP (tp>1): multiplier = 1/tp (all tensors sharded across TP ranks)
 *   - Without SP (tp>1): multiplier = (sharded/tp + replicated) / (sharded + replicated)
 *     where sharded = TP-sharded tensors, replicated = 4h for LN/residuals
 *   - SP_ON / SP_OFF ratio = (1/tp) / multiplier_without_SP
 * Communication: SP doubles TP comm volume (AllGather+ReduceScatter per layer × 4
 *   vs AllReduce per layer × 2) but gets better overlap (0.35 vs 0.20)
 * Net timing: SP is slightly slower (2× comm volume outweighs better overlap)
 * Sharding: SP does not affect parameter/gradient/optimizer sharding
 */
import { describe, it, expect } from 'vitest';
import {
  type SimulationConfig,
  type SimulationMetrics,
} from '../../src/core/simulation/engine.ts';
import { getSimulationMetrics } from '../../src/core/simulation/engine.ts';
import { createMultiNodeCluster } from '../../src/core/hardware/topology.ts';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Compute the expected SP_ON / SP_OFF activation ratio for a given model and TP degree.
 *
 * With SP: activationMultiplier = 1/tp
 * Without SP (tp>1): activationMultiplier = (sharded/tp + replicated) / (sharded + replicated)
 *   where sharded = qDim + 2*kvDim + h + mlpIntermCoeff*I, replicated = 4*h
 *
 * Ratio = (1/tp) / multiplier_without_SP
 *
 * Note: the multiplier is applied uniformly to the full per-layer activation (including
 * attention score memory), so the ratio is exact as computed here. Attention scores are
 * inherently TP-sharded (heads split), but since both SP_ON and SP_OFF apply their
 * multiplier to the same base, the attention score contribution cancels in the ratio.
 */
function expectedSpRatio(
  model: { numAttentionHeads: number; numKvHeads: number; headDim: number;
           hiddenSize: number; intermediateSize: number; gatedMLP: boolean },
  tp: number,
): number {
  const mlpIntermCoeff = model.gatedMLP ? 3 : 2;
  const sharded = (model.numAttentionHeads * model.headDim)
    + 2 * (model.numKvHeads * model.headDim)
    + model.hiddenSize
    + mlpIntermCoeff * model.intermediateSize;
  const replicated = 4 * model.hiddenSize;
  const multiplierNoSp = (sharded / tp + replicated) / (sharded + replicated);
  return (1 / tp) / multiplierNoSp;
}

// Model dimensions used in tests (must match src/core/models/architectures.ts)
const LLAMA2_70B = {
  numAttentionHeads: 64, numKvHeads: 8, headDim: 128,
  hiddenSize: 8192, intermediateSize: 28672, gatedMLP: true,
};

const LLAMA2_7B = {
  numAttentionHeads: 32, numKvHeads: 32, headDim: 128,
  hiddenSize: 4096, intermediateSize: 11008, gatedMLP: true,
};

function run(
  strategy: SimulationConfig['strategyType'],
  nodes: number,
  sc: SimulationConfig['strategyConfig'],
): SimulationMetrics {
  return getSimulationMetrics({
    clusterConfig: createMultiNodeCluster('h100-sxm', 8, nodes)!,
    modelId: 'llama2-70b',
    globalBatchSize: 256,
    microBatchSize: 2,
    sequenceLength: 2048,
    strategyType: strategy,
    strategyConfig: sc,
  });
}

type StrategyDef = {
  name: string;
  type: SimulationConfig['strategyType'];
  nodes: number;
  base: SimulationConfig['strategyConfig'];
};

const TWO_D: StrategyDef[] = [
  { name: 'FSDP+TP',   type: 'fsdp-tp',  nodes: 16, base: { tp: 8 } },
  { name: 'ZeRO-1+TP', type: 'zero1-tp', nodes: 16, base: { tp: 8 } },
];

const THREE_D: StrategyDef[] = [
  { name: 'DDP+TP+PP',    type: 'ddp-tp-pp',   nodes: 16, base: { tp: 8, pp: 2, numMicroBatches: 8 } },
  { name: 'ZeRO-1+TP+PP', type: 'zero1-tp-pp', nodes: 16, base: { tp: 8, pp: 2, numMicroBatches: 8 } },
  { name: 'FSDP+TP+PP',   type: 'fsdp-tp-pp',  nodes: 16, base: { tp: 8, pp: 2, numMicroBatches: 8 } },
];

const ALL_STRATEGIES = [...TWO_D, ...THREE_D];

// ---------------------------------------------------------------------------
// Section 1: SP activation multiplier is exact for every strategy
//
// SP_ON multiplier = 1/tp. SP_OFF multiplier = (sharded/tp + replicated) / total.
// Ratio = (1/tp) / SP_OFF_multiplier. For tp=8 on Llama 2 70B this is ~0.374.
// ---------------------------------------------------------------------------

describe('SP activation multiplier — all strategies, tp=8', () => {
  const expected = expectedSpRatio(LLAMA2_70B, 8);

  for (const s of ALL_STRATEGIES) {
    it(`${s.name}: activation ratio = ${expected.toFixed(4)}`, () => {
      const spOn = run(s.type, s.nodes, { ...s.base, sequenceParallel: true });
      const spOff = run(s.type, s.nodes, { ...s.base, sequenceParallel: false });

      const ratio = spOn.memoryPerGPU.activations / spOff.memoryPerGPU.activations;
      expect(ratio).toBeCloseTo(expected, 2);
    });
  }
});

// ---------------------------------------------------------------------------
// Section 2: SP activation multiplier scales with TP degree
//
// Test tp=2,4,8 to verify ratio = (1/tp) / multiplier_without_SP.
// Higher TP → more activation savings from SP (ratio decreases).
// ---------------------------------------------------------------------------

describe('SP activation multiplier scales with TP degree', () => {
  const tpConfigs: { tp: number; nodes: number }[] = [
    { tp: 2, nodes: 2 },
    { tp: 4, nodes: 4 },
    { tp: 8, nodes: 16 },
  ];

  for (const { tp, nodes } of tpConfigs) {
    const expected = expectedSpRatio(LLAMA2_70B, tp);

    it(`FSDP+TP tp=${tp}: ratio = ${expected.toFixed(4)}`, () => {
      const on = run('fsdp-tp', nodes, { tp, sequenceParallel: true });
      const off = run('fsdp-tp', nodes, { tp, sequenceParallel: false });

      const ratio = on.memoryPerGPU.activations / off.memoryPerGPU.activations;
      expect(ratio).toBeCloseTo(expected, 2);
    });

    it(`DDP+TP+PP tp=${tp}: ratio = ${expected.toFixed(4)}`, () => {
      const pp = 2;
      const on = run('ddp-tp-pp', nodes, { tp, pp, numMicroBatches: 8, sequenceParallel: true });
      const off = run('ddp-tp-pp', nodes, { tp, pp, numMicroBatches: 8, sequenceParallel: false });

      const ratio = on.memoryPerGPU.activations / off.memoryPerGPU.activations;
      expect(ratio).toBeCloseTo(expected, 2);
    });
  }
});

// ---------------------------------------------------------------------------
// Section 3: SP does NOT change sharding (params, grads, optimizer)
//
// SP only affects activations and communication. Static memory must be
// byte-identical between SP=true and SP=false.
// ---------------------------------------------------------------------------

describe('SP does not affect static memory — all strategies', () => {
  for (const s of ALL_STRATEGIES) {
    const spOn = run(s.type, s.nodes, { ...s.base, sequenceParallel: true });
    const spOff = run(s.type, s.nodes, { ...s.base, sequenceParallel: false });

    it(`${s.name}: identical parameter memory`, () => {
      expect(spOn.memoryPerGPU.parameters).toEqual(spOff.memoryPerGPU.parameters);
    });

    it(`${s.name}: identical gradient memory`, () => {
      expect(spOn.memoryPerGPU.gradients).toEqual(spOff.memoryPerGPU.gradients);
    });

    it(`${s.name}: identical optimizer memory`, () => {
      expect(spOn.memoryPerGPU.optimizerStates).toEqual(spOff.memoryPerGPU.optimizerStates);
    });
  }
});

// ---------------------------------------------------------------------------
// Section 4: SP changes timing for every strategy
//
// SP doubles TP comm volume (4 ops/layer vs 2) but gets better overlap
// (0.35 vs 0.20). Net effect: step time and MFU both differ.
// ---------------------------------------------------------------------------

describe('SP changes timing — all strategies', () => {
  for (const s of ALL_STRATEGIES) {
    const spOn = run(s.type, s.nodes, { ...s.base, sequenceParallel: true });
    const spOff = run(s.type, s.nodes, { ...s.base, sequenceParallel: false });

    it(`${s.name}: different step time`, () => {
      expect(spOn.stepTimeMs).not.toEqual(spOff.stepTimeMs);
    });

    it(`${s.name}: different MFU`, () => {
      expect(spOn.mfu).not.toEqual(spOff.mfu);
    });

    it(`${s.name}: different communication overhead`, () => {
      expect(spOn.communicationOverhead).not.toEqual(spOff.communicationOverhead);
    });
  }
});

// ---------------------------------------------------------------------------
// Section 5: SP timing delta scales with TP degree
//
// The overlap delta = tpCommTime × (0.35 - 0.20) = 0.15 × tpCommTime.
// Since tpCommTime grows with TP (more data exchanged), the absolute
// timing difference between SP on/off must increase with TP.
// ---------------------------------------------------------------------------

describe('SP timing delta increases with TP degree', () => {
  it('FSDP+TP: |delta(tp=8)| > |delta(tp=4)| > |delta(tp=2)|', () => {
    const configs = [
      { tp: 2, nodes: 2 },
      { tp: 4, nodes: 4 },
      { tp: 8, nodes: 8 },   // nodes=8 (not 16) to keep DP=8, GA=16 consistent across configs
    ];

    const deltas = configs.map(({ tp, nodes }) => {
      const on = run('fsdp-tp', nodes, { tp, sequenceParallel: true });
      const off = run('fsdp-tp', nodes, { tp, sequenceParallel: false });
      return Math.abs(on.stepTimeMs - off.stepTimeMs);
    });

    expect(deltas[2]).toBeGreaterThan(deltas[1]); // tp=8 > tp=4
    expect(deltas[1]).toBeGreaterThan(deltas[0]); // tp=4 > tp=2
  });

  it('DDP+TP+PP: |delta(tp=8)| > |delta(tp=4)|', () => {
    const configs = [
      { tp: 4, pp: 4, nodes: 16 },
      { tp: 8, pp: 2, nodes: 16 },
    ];

    const deltas = configs.map(({ tp, pp, nodes }) => {
      const on = run('ddp-tp-pp', nodes, { tp, pp, numMicroBatches: 8, sequenceParallel: true });
      const off = run('ddp-tp-pp', nodes, { tp, pp, numMicroBatches: 8, sequenceParallel: false });
      return Math.abs(on.stepTimeMs - off.stepTimeMs);
    });

    expect(deltas[1]).toBeGreaterThan(deltas[0]);
  });
});

// ---------------------------------------------------------------------------
// Section 6: SP activation savings are consistent across DP variants
//
// For the same TP degree, the activation ratio must be the same regardless
// of the DP strategy (DDP vs ZeRO-1 vs FSDP). SP is a TP-local operation.
// ---------------------------------------------------------------------------

describe('SP activation ratio is independent of DP type', () => {
  it('all 3D strategies at tp=8 pp=2 have the same activation ratio', () => {
    const ratios = THREE_D.map(s => {
      const on = run(s.type, s.nodes, { ...s.base, sequenceParallel: true });
      const off = run(s.type, s.nodes, { ...s.base, sequenceParallel: false });
      return on.memoryPerGPU.activations / off.memoryPerGPU.activations;
    });

    // All ratios must be identical (same TP degree)
    for (let i = 1; i < ratios.length; i++) {
      expect(ratios[i]).toEqual(ratios[0]);
    }
  });

  it('all 2D strategies at tp=8 have the same activation ratio', () => {
    const ratios = TWO_D.map(s => {
      const on = run(s.type, s.nodes, { ...s.base, sequenceParallel: true });
      const off = run(s.type, s.nodes, { ...s.base, sequenceParallel: false });
      return on.memoryPerGPU.activations / off.memoryPerGPU.activations;
    });

    for (let i = 1; i < ratios.length; i++) {
      expect(ratios[i]).toEqual(ratios[0]);
    }
  });

  it('2D and 3D at same TP degree have the same activation ratio', () => {
    // FSDP+TP tp=8 vs FSDP+TP+PP tp=8 pp=2 — same TP, same multiplier
    const r2d = (() => {
      const on = run('fsdp-tp', 16, { tp: 8, sequenceParallel: true });
      const off = run('fsdp-tp', 16, { tp: 8, sequenceParallel: false });
      return on.memoryPerGPU.activations / off.memoryPerGPU.activations;
    })();

    const r3d = (() => {
      const on = run('fsdp-tp-pp', 16, { tp: 8, pp: 2, numMicroBatches: 8, sequenceParallel: true });
      const off = run('fsdp-tp-pp', 16, { tp: 8, pp: 2, numMicroBatches: 8, sequenceParallel: false });
      return on.memoryPerGPU.activations / off.memoryPerGPU.activations;
    })();

    expect(r2d).toBeCloseTo(r3d, 10);
  });
});

// ---------------------------------------------------------------------------
// Section 7: SP lower total memory despite slower timing
//
// SP trades slightly more communication time for significantly less
// activation memory. Total GPU memory should be lower with SP.
// ---------------------------------------------------------------------------

describe('SP reduces total GPU memory — all strategies', () => {
  for (const s of ALL_STRATEGIES) {
    it(`${s.name}: total memory with SP < without SP`, () => {
      const spOn = run(s.type, s.nodes, { ...s.base, sequenceParallel: true });
      const spOff = run(s.type, s.nodes, { ...s.base, sequenceParallel: false });

      expect(spOn.memoryPerGPU.total).toBeLessThan(spOff.memoryPerGPU.total);
    });
  }
});

// ---------------------------------------------------------------------------
// Section 8: FSDP gather buffers present with and without SP
//
// FSDP-based strategies include gather buffers in peakActivations.
// SP should not remove gather buffers — it only reduces base activations.
// ---------------------------------------------------------------------------

describe('FSDP gather buffers present regardless of SP', () => {
  const fsdpStrategies: StrategyDef[] = [
    { name: 'FSDP+TP',   type: 'fsdp-tp',   nodes: 16, base: { tp: 8 } },
    { name: 'FSDP+TP+PP', type: 'fsdp-tp-pp', nodes: 16, base: { tp: 8, pp: 2, numMicroBatches: 8 } },
  ];

  for (const s of fsdpStrategies) {
    it(`${s.name} SP=true: peakActivations > activations (gather buffers)`, () => {
      const m = run(s.type, s.nodes, { ...s.base, sequenceParallel: true });
      expect(m.memoryPerGPU.peakActivations).toBeGreaterThan(m.memoryPerGPU.activations * 1.01);
    });

    it(`${s.name} SP=false: peakActivations > activations (gather buffers)`, () => {
      const m = run(s.type, s.nodes, { ...s.base, sequenceParallel: false });
      expect(m.memoryPerGPU.peakActivations).toBeGreaterThan(m.memoryPerGPU.activations * 1.01);
    });
  }

  // Non-FSDP strategies should NOT have gather buffers
  it('DDP+TP+PP: peakActivations = activations (no gather buffers)', () => {
    const m = run('ddp-tp-pp', 16, { tp: 8, pp: 2, numMicroBatches: 8, sequenceParallel: true });
    expect(m.memoryPerGPU.peakActivations).toEqual(m.memoryPerGPU.activations);
  });
});

// ---------------------------------------------------------------------------
// Section 9: SP with smaller model (7B) — verify it still works
//
// Guards against any model-size-dependent bugs.
// ---------------------------------------------------------------------------

describe('SP works on smaller model (Llama 7B)', () => {
  function run7b(strategy: SimulationConfig['strategyType'], sc: SimulationConfig['strategyConfig']) {
    return getSimulationMetrics({
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 2)!,
      modelId: 'llama2-7b',
      globalBatchSize: 128,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: strategy,
      strategyConfig: sc,
    });
  }

  const expectedRatio = expectedSpRatio(LLAMA2_7B, 4); // tp=4

  it(`FSDP+TP tp=4: activation ratio = ${expectedRatio.toFixed(4)}`, () => {
    const on = run7b('fsdp-tp', { tp: 4, sequenceParallel: true });
    const off = run7b('fsdp-tp', { tp: 4, sequenceParallel: false });
    expect(on.memoryPerGPU.activations / off.memoryPerGPU.activations).toBeCloseTo(expectedRatio, 2);
  });

  it(`ZeRO-1+TP tp=4: activation ratio = ${expectedRatio.toFixed(4)}`, () => {
    const on = run7b('zero1-tp', { tp: 4, sequenceParallel: true });
    const off = run7b('zero1-tp', { tp: 4, sequenceParallel: false });
    expect(on.memoryPerGPU.activations / off.memoryPerGPU.activations).toBeCloseTo(expectedRatio, 2);
  });

  it(`DDP+TP+PP tp=4 pp=2: activation ratio = ${expectedRatio.toFixed(4)}`, () => {
    const on = run7b('ddp-tp-pp', { tp: 4, pp: 2, numMicroBatches: 8, sequenceParallel: true });
    const off = run7b('ddp-tp-pp', { tp: 4, pp: 2, numMicroBatches: 8, sequenceParallel: false });
    expect(on.memoryPerGPU.activations / off.memoryPerGPU.activations).toBeCloseTo(expectedRatio, 2);
  });

  it('SP reduces total memory on 7B model', () => {
    const on = run7b('fsdp-tp', { tp: 4, sequenceParallel: true });
    const off = run7b('fsdp-tp', { tp: 4, sequenceParallel: false });
    expect(on.memoryPerGPU.total).toBeLessThan(off.memoryPerGPU.total);
  });
});
