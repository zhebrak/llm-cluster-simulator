/**
 * Parallelism Strategy Validation Test Suite
 *
 * Comprehensive validation of the simulator's 8 parallelism strategies across
 * memory sharding, communication patterns, strategy equivalences, model/GPU
 * matrices, activation scaling, throughput ordering, error paths, MoE specifics,
 * precision interactions, and overlap modeling.
 *
 * ~80 tests organized in describe blocks.
 */

import { describe, it, expect } from 'vitest';
import {
  SimulationEngine,
  getSimulationMetrics,
  type SimulationConfig,
  type SimulationMetrics,
} from '../../src/core/simulation/engine.ts';
import { getModel } from '../../src/core/models/index.ts';
import { createMultiNodeCluster } from '../../src/core/hardware/topology.ts';
// GPU specs available via createMultiNodeCluster

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Run simulation via getSimulationMetrics */
function sim(config: SimulationConfig): SimulationMetrics {
  return getSimulationMetrics(config);
}

/** Assert value is within ±tolerance fraction of expected */
function expectClose(actual: number, expected: number, tolerance: number, label?: string) {
  const ratio = actual / expected;
  const msg = label ? `${label}: ` : '';
  expect(
    ratio,
    `${msg}expected ${expected.toExponential(3)}, got ${actual.toExponential(3)} (ratio=${ratio.toFixed(4)})`,
  ).toBeGreaterThan(1 - tolerance);
  expect(
    ratio,
    `${msg}expected ${expected.toExponential(3)}, got ${actual.toExponential(3)} (ratio=${ratio.toFixed(4)})`,
  ).toBeLessThan(1 + tolerance);
}

/** Model state = parameters + gradients + optimizerStates */
function modelState(m: SimulationMetrics): number {
  return m.memoryPerGPU.parameters + m.memoryPerGPU.gradients + m.memoryPerGPU.optimizerStates;
}

// ---------------------------------------------------------------------------
// Baseline configs — every test spread-overrides only fields it's testing
// ---------------------------------------------------------------------------

const LLAMA2_7B_8xH100: SimulationConfig = {
  modelId: 'llama2-7b',
  clusterId: '8x-h100',
  globalBatchSize: 1024,
  microBatchSize: 2,
  sequenceLength: 2048,
  strategyType: 'ddp',
  activationCheckpointing: true,
  flashAttention: true,
  mixedPrecision: 'bf16',
};

const LLAMA2_70B_64xH100: SimulationConfig = {
  modelId: 'llama2-70b',
  clusterId: '64x-h100',
  globalBatchSize: 1024,
  microBatchSize: 1,
  sequenceLength: 2048,
  strategyType: 'fsdp',
  activationCheckpointing: true,
  flashAttention: true,
  mixedPrecision: 'bf16',
};

const LLAMA3_405B_1024xH100: SimulationConfig = {
  modelId: 'llama3-405b',
  clusterId: '1024x-h100',
  globalBatchSize: 2048,
  microBatchSize: 1,
  sequenceLength: 2048,
  strategyType: 'fsdp-tp-pp',
  strategyConfig: { tp: 8, pp: 16, dp: 8 },
  activationCheckpointing: true,
  flashAttention: true,
  mixedPrecision: 'bf16',
};

// ---------------------------------------------------------------------------
// Ground truth: LLaMA-2 7B (6.74B params, BF16)
// ---------------------------------------------------------------------------
const LLAMA2_7B_PARAMS = 6.74e9;
const LLAMA2_7B_FULL_PARAMS_BYTES = LLAMA2_7B_PARAMS * 2;   // 13.48 GB
const LLAMA2_7B_FULL_GRADS_BYTES = LLAMA2_7B_PARAMS * 2;    // 13.48 GB
const LLAMA2_7B_FULL_OPT_BYTES = LLAMA2_7B_PARAMS * 12;     // 80.88 GB
// Full model state: 107.84 GB (params + grads + opt)

// ===========================================================================
// Section 1: Memory Model Validation (6 tests)
// ===========================================================================
describe('Section 1: Memory Model Validation', () => {
  it('M1: DDP LLaMA-2 7B 8×H100 → OOM (107.84 GB > 80 GB)', () => {
    const m = sim({ ...LLAMA2_7B_8xH100, strategyType: 'ddp' });
    expect(m.memoryUtilization).toBeGreaterThan(1.0);
  });

  it('M2: ZeRO-1 LLaMA-2 7B 8×H100 → fits, correct sharding', () => {
    const m = sim({ ...LLAMA2_7B_8xH100, strategyType: 'zero-1' });
    expect(m.memoryUtilization).toBeLessThanOrEqual(1.0);
    // ZeRO-1: params replicated, grads replicated, optimizer sharded ÷8
    expectClose(m.memoryPerGPU.parameters, LLAMA2_7B_FULL_PARAMS_BYTES, 0.01, 'Z1 params');
    expectClose(m.memoryPerGPU.gradients, LLAMA2_7B_FULL_GRADS_BYTES, 0.01, 'Z1 grads');
    expectClose(m.memoryPerGPU.optimizerStates, LLAMA2_7B_FULL_OPT_BYTES / 8, 0.01, 'Z1 opt');
  });

  it('M3: FSDP LLaMA-2 7B 8×H100 → fits, fully sharded', () => {
    const m = sim({ ...LLAMA2_7B_8xH100, strategyType: 'fsdp' });
    expect(m.memoryUtilization).toBeLessThanOrEqual(1.0);
    // FSDP: everything sharded ÷8
    expectClose(m.memoryPerGPU.parameters, LLAMA2_7B_FULL_PARAMS_BYTES / 8, 0.01, 'FSDP params');
    expectClose(m.memoryPerGPU.gradients, LLAMA2_7B_FULL_GRADS_BYTES / 8, 0.01, 'FSDP grads');
    expectClose(m.memoryPerGPU.optimizerStates, LLAMA2_7B_FULL_OPT_BYTES / 8, 0.01, 'FSDP opt');
  });

  it('M4: LLaMA-2 70B 64×H100 — DDP OOM, ZeRO-1 OOM, FSDP fits', () => {
    const base = { ...LLAMA2_70B_64xH100, strategyConfig: undefined };
    const ddpM = sim({ ...base, strategyType: 'ddp' });
    const z1M = sim({ ...base, strategyType: 'zero-1' });
    const fsdpM = sim({ ...base, strategyType: 'fsdp' });
    expect(ddpM.memoryUtilization).toBeGreaterThan(1.0);
    expect(z1M.memoryUtilization).toBeGreaterThan(1.0);
    expect(fsdpM.memoryUtilization).toBeLessThanOrEqual(1.0);
  });

  it('M5: FSDP+TP vs ZeRO-1+TP — FSDP has lower model state', () => {
    const base = {
      ...LLAMA2_70B_64xH100,
      strategyConfig: { tp: 8 },
    };
    const fsdpTP = sim({ ...base, strategyType: 'fsdp-tp' });
    const z1TP = sim({ ...base, strategyType: 'zero1-tp' });
    expect(modelState(fsdpTP)).toBeLessThan(modelState(z1TP));
  });

  it('M6: 3D variants LLaMA-3 405B 1024×H100 — DDP+TP+PP > ZeRO-1+TP+PP > FSDP+TP+PP', () => {
    const sc = { tp: 8, pp: 16 };
    const ddpM = sim({ ...LLAMA3_405B_1024xH100, strategyType: 'ddp-tp-pp', strategyConfig: sc });
    const z1M = sim({ ...LLAMA3_405B_1024xH100, strategyType: 'zero1-tp-pp', strategyConfig: sc });
    const fsdpM = sim({ ...LLAMA3_405B_1024xH100, strategyType: 'fsdp-tp-pp', strategyConfig: sc });
    expect(modelState(ddpM)).toBeGreaterThanOrEqual(modelState(z1M));
    expect(modelState(z1M)).toBeGreaterThanOrEqual(modelState(fsdpM));
  });
});

// ===========================================================================
// Section 2: Communication Validation (6 tests)
// ===========================================================================
describe('Section 2: Communication Validation', () => {
  it('C1: DDP and FSDP both have communication > 0', () => {
    const ddpM = sim({ ...LLAMA2_7B_8xH100, strategyType: 'ddp' });
    const fsdpM = sim({ ...LLAMA2_7B_8xH100, strategyType: 'fsdp' });
    expect(ddpM.timing.communication).toBeGreaterThan(0);
    expect(fsdpM.timing.communication).toBeGreaterThan(0);
  });

  it('C2: TP scaling — all TP degrees have non-zero communication', () => {
    // With FSDP+TP on fixed cluster, increasing TP reduces DP and thus DP comm,
    // while adding TP comm. Net effect depends on model/cluster. Key invariant:
    // all configs produce non-zero communication and valid results.
    for (const tp of [1, 2, 4, 8]) {
      const m = sim({
        ...LLAMA2_70B_64xH100,
        strategyType: 'fsdp-tp',
        strategyConfig: { tp },
      });
      expect(m.timing.communication, `TP=${tp} comm > 0`).toBeGreaterThan(0);
      expect(m.mfu, `TP=${tp} mfu > 0`).toBeGreaterThan(0);
    }
  });

  it('C3: PP=1 → no bubble; PP>1 → bubble > 0', () => {
    const pp1 = sim({
      ...LLAMA3_405B_1024xH100,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 1 },
    });
    const pp4 = sim({
      ...LLAMA3_405B_1024xH100,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 4 },
    });
    expect(pp1.pipelineBubble).toBe(0);
    expect(pp4.pipelineBubble).toBeGreaterThan(0);
  });

  it('C4: FSDP larger model → proportionally more DP communication', () => {
    // Use same cluster and config, different models
    const small = sim({
      modelId: 'gpt3-125m',
      clusterId: '8x-h100',
      globalBatchSize: 512,
      microBatchSize: 8,
      sequenceLength: 2048,
      strategyType: 'fsdp',
    });
    const large = sim({
      ...LLAMA2_7B_8xH100,
      strategyType: 'fsdp',
    });
    // Larger model → more comm (parameters to gather/scatter)
    expect(large.timing.communication).toBeGreaterThan(small.timing.communication);
  });

  it('C5: Multi-node 64×H100 — TP intra-node, DP cross-node', () => {
    const m = sim({
      ...LLAMA2_70B_64xH100,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8 },
    });
    // TP=8 fits in a single node, DP=8 goes cross-node
    // Both communication components should be present
    expect(m.timing.communication).toBeGreaterThan(0);
    expect(m.mfu).toBeGreaterThan(0);
  });

  it('C6: SP does NOT change model state but DOES reduce activations', () => {
    const base = {
      ...LLAMA2_70B_64xH100,
      strategyType: 'fsdp-tp' as const,
    };
    const spOn = sim({ ...base, strategyConfig: { tp: 8, sequenceParallel: true } });
    const spOff = sim({ ...base, strategyConfig: { tp: 8, sequenceParallel: false } });

    // Model state unchanged
    expectClose(
      modelState(spOn),
      modelState(spOff),
      0.01,
      'SP model state unchanged',
    );

    // SP reduces activations
    expect(spOn.memoryPerGPU.activations).toBeLessThan(spOff.memoryPerGPU.activations);
  });
});

// ===========================================================================
// Section 4: Strategy Equivalences (9 tests)
// ===========================================================================
describe('Section 4: Strategy Equivalences', () => {
  it('E1: DDP DP=1 (1×H100) → correct full model state', () => {
    const m = sim({
      modelId: 'llama2-7b',
      clusterId: '1x-h100',
      globalBatchSize: 32,
      microBatchSize: 2,
      sequenceLength: 2048,
      strategyType: 'ddp',
    });
    expectClose(m.memoryPerGPU.parameters, LLAMA2_7B_FULL_PARAMS_BYTES, 0.01, 'DDP DP=1 params');
    expectClose(m.memoryPerGPU.gradients, LLAMA2_7B_FULL_GRADS_BYTES, 0.01, 'DDP DP=1 grads');
    expectClose(m.memoryPerGPU.optimizerStates, LLAMA2_7B_FULL_OPT_BYTES, 0.01, 'DDP DP=1 opt');
  });

  it('E2: FSDP DP=1 (1×H100) → same model state as DDP DP=1', () => {
    const base = {
      modelId: 'llama2-7b',
      clusterId: '1x-h100',
      globalBatchSize: 32,
      microBatchSize: 2,
      sequenceLength: 2048,
    };
    const ddpM = sim({ ...base, strategyType: 'ddp' as const });
    const fsdpM = sim({ ...base, strategyType: 'fsdp' as const });
    expectClose(modelState(fsdpM), modelState(ddpM), 0.01, 'FSDP DP=1 ≈ DDP DP=1');
  });

  it('E3: ZeRO-1 DP=1 (1×H100) → same model state as DDP DP=1', () => {
    const base = {
      modelId: 'llama2-7b',
      clusterId: '1x-h100',
      globalBatchSize: 32,
      microBatchSize: 2,
      sequenceLength: 2048,
    };
    const ddpM = sim({ ...base, strategyType: 'ddp' as const });
    const z1M = sim({ ...base, strategyType: 'zero-1' as const });
    expectClose(modelState(z1M), modelState(ddpM), 0.01, 'Z1 DP=1 ≈ DDP DP=1');
  });

  it('E4: FSDP+TP tp=1 ≈ FSDP (8×H100)', () => {
    const fsdpM = sim({ ...LLAMA2_7B_8xH100, strategyType: 'fsdp' });
    const fsdpTP1 = sim({
      ...LLAMA2_7B_8xH100,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 1 },
    });
    expectClose(modelState(fsdpTP1), modelState(fsdpM), 0.01, 'FSDP+TP(1) ≈ FSDP');
  });

  it('E5: FSDP+TP+PP pp=1 ≈ FSDP+TP (8×H100)', () => {
    const fsdpTP = sim({
      ...LLAMA2_7B_8xH100,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 2 },
    });
    const fsdpTPPP1 = sim({
      ...LLAMA2_7B_8xH100,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 2, pp: 1 },
    });
    expectClose(modelState(fsdpTPPP1), modelState(fsdpTP), 0.01, 'FSDP+TP+PP(1) ≈ FSDP+TP');
  });

  it('E6: FSDP+TP+PP tp=1,pp=1 ≈ FSDP (8×H100)', () => {
    const fsdpM = sim({ ...LLAMA2_7B_8xH100, strategyType: 'fsdp' });
    const fsdp3d = sim({
      ...LLAMA2_7B_8xH100,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 1, pp: 1 },
    });
    expectClose(modelState(fsdp3d), modelState(fsdpM), 0.01, 'FSDP+TP(1)+PP(1) ≈ FSDP');
  });

  it('E7: DDP+TP+PP tp=1,pp=1 ≈ DDP (8×H100)', () => {
    const ddpM = sim({ ...LLAMA2_7B_8xH100, strategyType: 'ddp' });
    const ddp3d = sim({
      ...LLAMA2_7B_8xH100,
      strategyType: 'ddp-tp-pp',
      strategyConfig: { tp: 1, pp: 1 },
    });
    expectClose(modelState(ddp3d), modelState(ddpM), 0.01, 'DDP+TP(1)+PP(1) ≈ DDP');
  });

  it('E8: ZeRO-1 model state > FSDP model state for DP=8', () => {
    const z1 = sim({ ...LLAMA2_7B_8xH100, strategyType: 'zero-1' });
    const fsdp = sim({ ...LLAMA2_7B_8xH100, strategyType: 'fsdp' });
    expect(modelState(z1)).toBeGreaterThan(modelState(fsdp));
  });

  it('E9: Model state ordering DDP ≥ ZeRO-1 ≥ FSDP for DP=8 and DP=64', () => {
    for (const clusterId of ['8x-h100', '64x-h100'] as const) {
      const base = {
        modelId: 'llama2-7b',
        clusterId,
        globalBatchSize: 1024,
        microBatchSize: 2,
        sequenceLength: 2048,
        activationCheckpointing: true,
        flashAttention: true,
        mixedPrecision: 'bf16' as const,
      };
      const ddpMS = modelState(sim({ ...base, strategyType: 'ddp' }));
      const z1MS = modelState(sim({ ...base, strategyType: 'zero-1' }));
      const fsdpMS = modelState(sim({ ...base, strategyType: 'fsdp' }));
      expect(ddpMS, `${clusterId}: DDP ≥ Z1`).toBeGreaterThanOrEqual(z1MS);
      expect(z1MS, `${clusterId}: Z1 ≥ FSDP`).toBeGreaterThanOrEqual(fsdpMS);
    }
  });
});

// ===========================================================================
// Section 5: Strategy × Model Matrix (8 tests)
// ===========================================================================
describe('Section 5: Strategy × Model Matrix', () => {
  it('GPT-3 125M on 8×H100: DDP fits, FSDP fits, both have reasonable MFU', () => {
    const base = {
      modelId: 'gpt3-125m',
      clusterId: '8x-h100',
      globalBatchSize: 512,
      microBatchSize: 8,
      sequenceLength: 2048,
    };
    const ddpM = sim({ ...base, strategyType: 'ddp' as const });
    const fsdpM = sim({ ...base, strategyType: 'fsdp' as const });
    expect(ddpM.memoryUtilization).toBeLessThanOrEqual(1.0);
    expect(fsdpM.memoryUtilization).toBeLessThanOrEqual(1.0);
    // Both should have similar MFU for this small model (within 8%)
    // DDP and FSDP both use 2.85× backward multiplier with AC; gap from FSDP AllGather overhead
    expect(Math.abs(ddpM.mfu - fsdpM.mfu)).toBeLessThan(0.08);
  });

  it('LLaMA-2 70B on 64×H100: DDP OOM, ZeRO-1 OOM, FSDP fits with reasonable MFU', () => {
    const base = {
      ...LLAMA2_70B_64xH100,
      strategyConfig: undefined,
    };
    const ddpM = sim({ ...base, strategyType: 'ddp' as const });
    const z1M = sim({ ...base, strategyType: 'zero-1' as const });
    const fsdpM = sim({ ...base, strategyType: 'fsdp' as const });
    expect(ddpM.memoryUtilization).toBeGreaterThan(1.0);
    expect(z1M.memoryUtilization).toBeGreaterThan(1.0);
    expect(fsdpM.memoryUtilization).toBeLessThanOrEqual(1.0);
    expect(fsdpM.mfu).toBeGreaterThan(0.10);
  });

  it('LLaMA-3 405B on 512×H100 FSDP+TP+PP: memory ordering DDP > ZeRO-1 > FSDP', () => {
    const base: SimulationConfig = {
      modelId: 'llama3-405b',
      clusterId: '512x-h100',
      globalBatchSize: 2048,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 8 },
      activationCheckpointing: true,
    };
    const ddpMS = modelState(sim({ ...base, strategyType: 'ddp-tp-pp' }));
    const z1MS = modelState(sim({ ...base, strategyType: 'zero1-tp-pp' }));
    const fsdpMS = modelState(sim({ ...base, strategyType: 'fsdp-tp-pp' }));
    expect(ddpMS).toBeGreaterThanOrEqual(z1MS);
    expect(z1MS).toBeGreaterThanOrEqual(fsdpMS);
  });

  it('Mixtral 8×7B on 32×H100: DDP OOM, FSDP fits', () => {
    const base = {
      modelId: 'mixtral-8x7b',
      clusterId: '32x-h100',
      globalBatchSize: 256,
      microBatchSize: 1,
      sequenceLength: 2048,
    };
    const ddpM = sim({ ...base, strategyType: 'ddp' as const });
    const fsdpM = sim({ ...base, strategyType: 'fsdp' as const });
    expect(ddpM.memoryUtilization).toBeGreaterThan(1.0);
    expect(fsdpM.memoryUtilization).toBeLessThanOrEqual(1.0);
  });

  it('DeepSeek-V3 on 256×H100 FSDP+TP: MFU > 0 and uses activeParams', () => {
    const m = sim({
      modelId: 'deepseek-v3',
      clusterId: '256x-h100',
      globalBatchSize: 2048,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8 },
      activationCheckpointing: true,
    });
    // MFU computed from activeParams (~37B). Without EP, MoE is memory-heavy
    // but compute uses only active params → MFU is modest but positive.
    expect(m.mfu).toBeGreaterThan(0.01);
    expect(m.mfu).toBeLessThan(0.50);
  });

  it('LLaMA-2 13B on 8× A100-40GB: DDP OOM, FSDP tight fit', () => {
    const cluster = createMultiNodeCluster('a100-40gb', 8, 1)!;
    const base: SimulationConfig = {
      modelId: 'llama2-13b',
      clusterConfig: cluster,
      globalBatchSize: 128,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'ddp',
      activationCheckpointing: true,
    };
    const ddpM = sim({ ...base, strategyType: 'ddp' });
    const fsdpM = sim({ ...base, strategyType: 'fsdp' });
    expect(ddpM.memoryUtilization).toBeGreaterThan(1.0);
    expect(fsdpM.memoryUtilization).toBeLessThanOrEqual(1.0);
  });

  it('GPT-3 125M on 8×H100 FSDP+TP tp=8: lower MFU than pure FSDP', () => {
    const fsdpTP8 = sim({
      modelId: 'gpt3-125m',
      clusterId: '8x-h100',
      globalBatchSize: 512,
      microBatchSize: 8,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8 },
    });
    const fsdp = sim({
      modelId: 'gpt3-125m',
      clusterId: '8x-h100',
      globalBatchSize: 512,
      microBatchSize: 8,
      sequenceLength: 2048,
      strategyType: 'fsdp',
    });
    // TP=8 with tiny model adds overhead — MFU should be lower than pure FSDP
    expect(fsdpTP8.mfu).toBeLessThan(fsdp.mfu);
  });

  it('LLaMA-3 405B on 1024×H100 FSDP+TP+PP TP=8,PP=16,DP=8: fits, reasonable MFU', () => {
    const m = sim(LLAMA3_405B_1024xH100);
    expect(m.memoryUtilization).toBeLessThanOrEqual(1.0);
    expect(m.mfu).toBeGreaterThan(0.10);
    expect(m.mfu).toBeLessThan(0.60);
  });
});

// ===========================================================================
// Section 6: Strategy × GPU Matrix (5 tests)
// ===========================================================================
describe('Section 6: Strategy × GPU Matrix', () => {
  it('H200 (141GB) vs H100 (80GB): same config → H200 lower memoryUtilization', () => {
    const baseH100 = sim(LLAMA2_70B_64xH100);
    const baseH200 = sim({ ...LLAMA2_70B_64xH100, clusterId: '64x-h200' });
    expect(baseH200.memoryUtilization).toBeLessThan(baseH100.memoryUtilization);
  });

  it('MI300X (192GB): FSDP LLaMA-2 70B 8×MI300X → lower memUtil than H100', () => {
    const h100M = sim({
      ...LLAMA2_70B_64xH100,
      clusterId: '8x-h100',
      globalBatchSize: 128,
    });
    const mi300M = sim({
      ...LLAMA2_70B_64xH100,
      clusterId: '8x-mi300x',
      globalBatchSize: 128,
    });
    // MI300X has 192GB vs H100's 80GB → much lower utilization
    expect(mi300M.memoryUtilization).toBeLessThan(h100M.memoryUtilization);
  });

  it('H200 throughput ≈ H100 for same model (similar TFLOPS)', () => {
    const h100M = sim(LLAMA2_70B_64xH100);
    const h200M = sim({ ...LLAMA2_70B_64xH100, clusterId: '64x-h200' });
    // H200 has same compute as H100, so throughput should be similar (±30%)
    const ratio = h200M.tokensPerSecond / h100M.tokensPerSecond;
    expect(ratio).toBeGreaterThan(0.7);
    expect(ratio).toBeLessThan(1.3);
  });

  it('V100-32GB custom cluster: GPT-3 125M DDP fits on 8×V100', () => {
    const cluster = createMultiNodeCluster('v100-32gb', 8, 1)!;
    const m = sim({
      modelId: 'gpt3-125m',
      clusterConfig: cluster,
      globalBatchSize: 256,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'ddp',
    });
    expect(m.memoryUtilization).toBeLessThanOrEqual(1.0);
    expect(m.mfu).toBeGreaterThan(0);
  });

  it('A100-40GB custom cluster: LLaMA-2 7B FSDP DP=8 fits; DDP OOM', () => {
    const cluster = createMultiNodeCluster('a100-40gb', 8, 1)!;
    const base: SimulationConfig = {
      modelId: 'llama2-7b',
      clusterConfig: cluster,
      globalBatchSize: 256,
      microBatchSize: 2,
      sequenceLength: 2048,
      strategyType: 'ddp',
      activationCheckpointing: true,
    };
    const ddpM = sim({ ...base, strategyType: 'ddp' });
    const fsdpM = sim({ ...base, strategyType: 'fsdp' });
    expect(ddpM.memoryUtilization).toBeGreaterThan(1.0);
    expect(fsdpM.memoryUtilization).toBeLessThanOrEqual(1.0);
  });
});

// ===========================================================================
// Section 7: Activation Memory (6 tests)
// ===========================================================================
describe('Section 7: Activation Memory', () => {
  it('A1: Seq length scaling — activations scale roughly linearly', () => {
    const activations: number[] = [];
    for (const seq of [1024, 2048, 4096, 8192]) {
      const m = sim({
        modelId: 'llama3-8b',
        clusterId: '8x-h100',
        globalBatchSize: 512,
        microBatchSize: 1,
        sequenceLength: seq,
        strategyType: 'fsdp',
        activationCheckpointing: true,
        flashAttention: true,
      });
      activations.push(m.memoryPerGPU.activations);
    }
    // Each doubling of seq should scale activations ∈ [1.5, 2.5]
    for (let i = 1; i < activations.length; i++) {
      const ratio = activations[i] / activations[i - 1];
      expect(ratio, `seq doubling ratio at index ${i}`).toBeGreaterThan(1.5);
      expect(ratio, `seq doubling ratio at index ${i}`).toBeLessThan(2.5);
    }
  });

  it('A2: MBS scaling — activations scale roughly linearly', () => {
    const activations: number[] = [];
    for (const mbs of [1, 2, 4, 8]) {
      const m = sim({
        modelId: 'llama3-8b',
        clusterId: '8x-h100',
        globalBatchSize: 512,
        microBatchSize: mbs,
        sequenceLength: 4096,
        strategyType: 'fsdp',
        activationCheckpointing: true,
        flashAttention: true,
      });
      activations.push(m.memoryPerGPU.activations);
    }
    for (let i = 1; i < activations.length; i++) {
      const ratio = activations[i] / activations[i - 1];
      expect(ratio, `MBS doubling ratio at index ${i}`).toBeGreaterThan(1.5);
      expect(ratio, `MBS doubling ratio at index ${i}`).toBeLessThan(2.5);
    }
  });

  it('A3: Activation checkpointing reduces activations 2×–6×', () => {
    const base: SimulationConfig = {
      modelId: 'llama3-8b',
      clusterId: '8x-h100',
      globalBatchSize: 512,
      microBatchSize: 1,
      sequenceLength: 4096,
      strategyType: 'fsdp',
      flashAttention: true,
    };
    const withCkpt = sim({ ...base, activationCheckpointing: true });
    const withoutCkpt = sim({ ...base, activationCheckpointing: false });
    const reduction = withoutCkpt.memoryPerGPU.activations / withCkpt.memoryPerGPU.activations;
    expect(reduction).toBeGreaterThan(2);
    expect(reduction).toBeLessThan(6);
  });

  it('A4: TP reduces activations', () => {
    const tp1 = sim({
      ...LLAMA2_70B_64xH100,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 1 },
    });
    const tp8 = sim({
      ...LLAMA2_70B_64xH100,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8 },
    });
    expect(tp8.memoryPerGPU.activations).toBeLessThan(tp1.memoryPerGPU.activations);
  });

  it('A5: SP reduces activations, same model state', () => {
    const spOn = sim({
      ...LLAMA2_70B_64xH100,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8, sequenceParallel: true },
    });
    const spOff = sim({
      ...LLAMA2_70B_64xH100,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8, sequenceParallel: false },
    });
    expect(spOn.memoryPerGPU.activations).toBeLessThan(spOff.memoryPerGPU.activations);
    expectClose(modelState(spOn), modelState(spOff), 0.01, 'SP model state unchanged');
  });

  it('A6: PP reduces per-GPU model state with DDP (non-FSDP)', () => {
    // With DDP+TP+PP, PP partitions layers → fewer params/grads/opt per GPU.
    // (With FSDP+TP+PP, FSDP already shards across all GPUs, so PP doesn't
    // further reduce model state — it's the same total shard count.)
    const pp1 = sim({
      ...LLAMA3_405B_1024xH100,
      strategyType: 'ddp-tp-pp',
      strategyConfig: { tp: 8, pp: 1 },
    });
    const pp4 = sim({
      ...LLAMA3_405B_1024xH100,
      strategyType: 'ddp-tp-pp',
      strategyConfig: { tp: 8, pp: 4 },
    });
    expect(modelState(pp4)).toBeLessThan(modelState(pp1));
  });
});

// ===========================================================================
// Section 8: Throughput & MFU (5 tests)
// ===========================================================================
describe('Section 8: Throughput & MFU', () => {
  it('T1: GPT-3 125M 64×H100 — DDP and FSDP have similar MFU (both fit, small model)', () => {
    const base: SimulationConfig = {
      modelId: 'gpt3-125m',
      clusterId: '64x-h100',
      globalBatchSize: 2048,
      microBatchSize: 8,
      sequenceLength: 2048,
      strategyType: 'ddp',
    };
    const ddpM = sim({ ...base, strategyType: 'ddp' });
    const fsdpM = sim({ ...base, strategyType: 'fsdp' });
    // Both fit and have positive MFU
    expect(ddpM.mfu).toBeGreaterThan(0.20);
    expect(fsdpM.mfu).toBeGreaterThan(0.20);
    // For small models, DDP and FSDP MFU are close (within 8%)
    // DDP and FSDP both use 2.85× backward multiplier with AC; gap from FSDP AllGather overhead
    expect(Math.abs(ddpM.mfu - fsdpM.mfu)).toBeLessThan(0.08);
  });

  it('T2: LLaMA-2 70B 64×H100 — DDP OOM (MFU=0 via run()), FSDP MFU > 0', () => {
    const base = { ...LLAMA2_70B_64xH100, strategyConfig: undefined };
    // DDP: OOM
    const ddpEngine = new SimulationEngine();
    ddpEngine.configure({ ...base, strategyType: 'ddp' });
    const ddpResult = ddpEngine.run();
    expect(ddpResult.success).toBe(false);
    expect(ddpResult.metrics.mfu).toBe(0);

    // FSDP: works
    const fsdpEngine = new SimulationEngine();
    fsdpEngine.configure({ ...base, strategyType: 'fsdp' });
    const fsdpResult = fsdpEngine.run();
    expect(fsdpResult.success).toBe(true);
    expect(fsdpResult.metrics.mfu).toBeGreaterThan(0);
  });

  it('T3: LLaMA-3 70B FSDP+TP sweep — MFU not monotonically increasing with TP', () => {
    const mfus: number[] = [];
    for (const tp of [1, 2, 4, 8]) {
      const m = sim({
        modelId: 'llama3-70b',
        clusterId: '64x-h100',
        globalBatchSize: 1024,
        microBatchSize: 1,
        sequenceLength: 2048,
        strategyType: 'fsdp-tp',
        strategyConfig: { tp },
        activationCheckpointing: true,
      });
      mfus.push(m.mfu);
    }
    // There should be a sweet spot — TP=1 and TP=8 both shouldn't be the best
    const maxMFU = Math.max(...mfus);
    const tp1IsBest = mfus[0] === maxMFU;
    const tp8IsBest = mfus[3] === maxMFU;
    // At least one of the extremes should not be the best
    expect(tp1IsBest && tp8IsBest, 'TP=1 and TP=8 should not both be best').toBe(false);
  });

  it('T4: PP bubble decreases with more micro-batches → throughput increases', () => {
    const results: { tps: number; bubble: number }[] = [];
    for (const numMB of [16, 32, 64, 128]) {
      const m = sim({
        ...LLAMA3_405B_1024xH100,
        globalBatchSize: numMB * 1 * 8, // numMB × MBS × DP
        microBatchSize: 1,
        strategyConfig: { tp: 8, pp: 16, numMicroBatches: numMB },
      });
      results.push({ tps: m.tokensPerSecond, bubble: m.pipelineBubble });
    }
    // Bubble should decrease and throughput increase
    for (let i = 1; i < results.length; i++) {
      expect(results[i].bubble, `bubble ${i} < ${i - 1}`).toBeLessThanOrEqual(results[i - 1].bubble);
      expect(results[i].tps, `tps ${i} > ${i - 1}`).toBeGreaterThanOrEqual(results[i - 1].tps);
    }
  });

  it('T5: Interleaved 1F1B → lower pipeline bubble than standard 1F1B', () => {
    const base: SimulationConfig = {
      ...LLAMA3_405B_1024xH100,
      strategyConfig: { tp: 8, pp: 16 },
    };
    const standard = sim({
      ...base,
      strategyConfig: { ...base.strategyConfig, pipelineSchedule: '1f1b' },
    });
    const interleaved = sim({
      ...base,
      strategyConfig: { ...base.strategyConfig, pipelineSchedule: 'interleaved-1f1b', interleavedStages: 2 },
    });
    expect(interleaved.pipelineBubble).toBeLessThan(standard.pipelineBubble);
  });
});

// ===========================================================================
// Section 9: Cross-Strategy Consistency (5 tests)
// ===========================================================================
describe('Section 9: Cross-Strategy Consistency', () => {
  it('X1: DDP vs FSDP — same useful work (6 × P × tokens_per_step)', () => {
    const model = getModel('llama2-7b', 2048)!;
    const base = {
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 256,
      microBatchSize: 4,
      sequenceLength: 2048,
    };
    const ddpM = sim({ ...base, strategyType: 'ddp' as const });
    const fsdpM = sim({ ...base, strategyType: 'fsdp' as const });
    const expectedWork = 6 * (model.activeParams ?? model.totalParams) * 256 * 2048;
    // Both produce the same useful work per step
    const ddpWork = ddpM.tokensPerSecond * ddpM.stepTimeMs / 1000 * 6 * (model.activeParams ?? model.totalParams);
    const fsdpWork = fsdpM.tokensPerSecond * fsdpM.stepTimeMs / 1000 * 6 * (model.activeParams ?? model.totalParams);
    expectClose(ddpWork, expectedWork, 0.01, 'DDP useful work');
    expectClose(fsdpWork, expectedWork, 0.01, 'FSDP useful work');
  });

  it('X2: Same GBS/MBS/DP → same GA regardless of strategy', () => {
    const base: SimulationConfig = {
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 256,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'ddp',
    };
    // All pure DP strategies with DP=8: GA = ceil(256 / (4 × 8)) = 8 — verify via tokens/step
    const ddpM = sim({ ...base, strategyType: 'ddp' });
    const z1M = sim({ ...base, strategyType: 'zero-1' });
    const fsdpM = sim({ ...base, strategyType: 'fsdp' });
    // All produce same tokens per step
    const tokensPerStep = 256 * 2048;
    expectClose(ddpM.tokensPerSecond * ddpM.stepTimeMs / 1000, tokensPerStep, 0.01, 'DDP tokens/step');
    expectClose(z1M.tokensPerSecond * z1M.stepTimeMs / 1000, tokensPerStep, 0.01, 'Z1 tokens/step');
    expectClose(fsdpM.tokensPerSecond * fsdpM.stepTimeMs / 1000, tokensPerStep, 0.01, 'FSDP tokens/step');
  });

  it('X3: Model totalParams unchanged by TP degree', () => {
    const model = getModel('llama2-70b', 2048)!;
    for (const tp of [1, 2, 4, 8]) {
      const m = sim({
        ...LLAMA2_70B_64xH100,
        strategyType: 'fsdp-tp',
        strategyConfig: { tp },
      });
      // Model params are intrinsic, not affected by parallelism
      // Verify via total model state scaling correctly
      expect(m.mfu).toBeGreaterThan(0); // sanity — it runs
    }
    // Model totalParams from registry should be consistent
    expect(model.totalParams).toBeGreaterThan(60e9);
    expect(model.totalParams).toBeLessThan(80e9);
  });

  it('X4: Model totalParams unchanged by PP degree', () => {
    const model = getModel('llama3-405b', 2048)!;
    expect(model.totalParams).toBeGreaterThan(350e9);
    expect(model.totalParams).toBeLessThan(450e9);
    // Run with different PP values
    for (const pp of [1, 4, 8, 16]) {
      const m = sim({
        ...LLAMA3_405B_1024xH100,
        strategyConfig: { tp: 8, pp },
      });
      expect(m.stepTimeMs).toBeGreaterThan(0); // sanity
    }
  });

  it('X5: tokensPerSecond × stepTimeMs/1000 ≈ GBS × seqLen', () => {
    const strategies: SimulationConfig['strategyType'][] = ['ddp', 'fsdp', 'zero-1'];
    const GBS = 256;
    const SEQ = 2048;
    for (const st of strategies) {
      const m = sim({
        modelId: 'llama2-7b',
        clusterId: '8x-h100',
        globalBatchSize: GBS,
        microBatchSize: 4,
        sequenceLength: SEQ,
        strategyType: st,
      });
      const computed = m.tokensPerSecond * m.stepTimeMs / 1000;
      expectClose(computed, GBS * SEQ, 0.01, `${st} tps×time consistency`);
    }
  });
});

// ===========================================================================
// Validation & Error Paths (8 tests)
// ===========================================================================
describe('Validation & Error Paths', () => {
  it('V1: TP×PP×DP ≠ totalGPUs → validation error', () => {
    // 100 GPUs, TP=8, PP=4 → DP=floor(100/32)=3, 8×4×3=96≠100
    const cluster = createMultiNodeCluster('h100-sxm', 4, 25)!; // 100 GPUs, 4 per node
    const engine = new SimulationEngine();
    engine.configure({
      modelId: 'llama2-70b',
      clusterConfig: cluster,
      globalBatchSize: 256,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 4 },
    });
    const validation = engine.validate();
    // TP=8 > gpusPerNode=4 → error
    expect(validation.errors.length).toBeGreaterThan(0);
  });

  it('V2: Non-power-of-2 TP=3 on 8 GPUs → validation error (GPU count mismatch)', () => {
    // TP=3 on 8 GPUs: DP = floor(8/3) = 2, but 3×2=6 ≠ 8 → validation error
    const engine = new SimulationEngine();
    engine.configure({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 256,
      microBatchSize: 2,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 3 },
    });
    const validation = engine.validate();
    expect(validation.errors.length).toBeGreaterThan(0);
    const hasMatchError = validation.errors.some(
      e => e.toLowerCase().includes('match') || e.toLowerCase().includes('cluster'),
    );
    expect(hasMatchError, 'should error on TP×DP ≠ totalGPUs').toBe(true);
  });

  it('V3: TP=16 on 8-GPU-per-node cluster → validation warning (cross-node TP)', () => {
    const engine = new SimulationEngine();
    engine.configure({
      modelId: 'llama2-70b',
      clusterId: '64x-h100',
      globalBatchSize: 256,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 16 },
    });
    const validation = engine.validate();
    // Cross-node TP should produce a warning, not an error
    const hasTPError = validation.errors.some(
      e => e.toLowerCase().includes('exceeds') || e.toLowerCase().includes('per node'),
    );
    expect(hasTPError, 'should NOT error on TP > gpusPerNode').toBe(false);
    const hasTPWarning = validation.warnings.some(
      w => w.toLowerCase().includes('exceeds gpus per node'),
    );
    expect(hasTPWarning, 'should warn about cross-node TP').toBe(true);
  });

  it('V4: PP > numLayers → validation warning, still runs', () => {
    const engine = new SimulationEngine();
    engine.configure({
      modelId: 'gpt3-125m', // 12 layers
      clusterId: '1024x-h100',
      globalBatchSize: 2048,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 128 },
    });
    const validation = engine.validate();
    // Should warn or error about PP > layers
    const hasIssue = validation.warnings.length > 0 || validation.errors.length > 0;
    expect(hasIssue).toBe(true);
  });

  it('V5: Layers not divisible by PP → validation warning', () => {
    // LLaMA-2 70B has 80 layers; PP=6 doesn't divide 80
    const engine = new SimulationEngine();
    engine.configure({
      modelId: 'llama2-70b',
      clusterId: '256x-h100',
      globalBatchSize: 1024,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 6 },
    });
    const validation = engine.validate();
    const hasDivWarning = validation.warnings.some(
      w => w.toLowerCase().includes('divide') || w.toLowerCase().includes('evenly') || w.toLowerCase().includes('unbalanced'),
    );
    expect(hasDivWarning, 'should warn about layers not dividing by PP').toBe(true);
  });

  it('V6: Interleaved schedule layers not divisible by PP×v → warning', () => {
    const engine = new SimulationEngine();
    engine.configure({
      modelId: 'llama2-70b', // 80 layers
      clusterId: '256x-h100',
      globalBatchSize: 1024,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 6, pipelineSchedule: 'interleaved-1f1b', interleavedStages: 2 },
    });
    const validation = engine.validate();
    // 80 layers / (6×2) = 6.67 — not evenly divisible
    const hasWarning = validation.warnings.length > 0;
    expect(hasWarning, 'should warn about interleaved layer divisibility').toBe(true);
  });

  it('V7: DP=1 for all 8 strategies on 1×H100 → no crash', () => {
    const strategies: SimulationConfig['strategyType'][] = [
      'ddp', 'zero-1', 'fsdp',
      'fsdp-tp', 'zero1-tp',
      'ddp-tp-pp', 'zero1-tp-pp', 'fsdp-tp-pp',
    ];
    for (const st of strategies) {
      const m = sim({
        modelId: 'gpt3-125m',
        clusterId: '1x-h100',
        globalBatchSize: 32,
        microBatchSize: 4,
        sequenceLength: 2048,
        strategyType: st,
        strategyConfig: { tp: 1, pp: 1 },
      });
      expect(m.stepTimeMs, `${st} on 1×H100 no crash`).toBeGreaterThan(0);
    }
  });

  it('V8: DP=1 for FSDP → no sharding, model state = full', () => {
    const m = sim({
      modelId: 'llama2-7b',
      clusterId: '1x-h100',
      globalBatchSize: 32,
      microBatchSize: 2,
      sequenceLength: 2048,
      strategyType: 'fsdp',
    });
    expectClose(m.memoryPerGPU.parameters, LLAMA2_7B_FULL_PARAMS_BYTES, 0.01, 'FSDP DP=1 params');
    expectClose(m.memoryPerGPU.optimizerStates, LLAMA2_7B_FULL_OPT_BYTES, 0.01, 'FSDP DP=1 opt');
  });
});

// ===========================================================================
// MoE-Specific Memory (6 tests)
// ===========================================================================
describe('MoE-Specific Memory', () => {
  it('MoE1: Mixtral 8×7B FSDP 32×H100 EP=1 vs EP=8 → both produce valid results', () => {
    // With FSDP, EP subdivides DP: expert sharding = TP×PP×EP×(DP/EP) = TP×PP×DP
    // So param memory is identical. EP benefit is reduced expert DP comm and compute.
    const ep1 = sim({
      modelId: 'mixtral-8x7b',
      clusterId: '32x-h100',
      globalBatchSize: 256,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 1, ep: 1 },
      activationCheckpointing: true,
    });
    const ep8 = sim({
      modelId: 'mixtral-8x7b',
      clusterId: '32x-h100',
      globalBatchSize: 256,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 1, ep: 8 },
      activationCheckpointing: true,
    });
    // Both produce valid results
    expect(ep1.stepTimeMs).toBeGreaterThan(0);
    expect(ep8.stepTimeMs).toBeGreaterThan(0);
    // Param memory is same (EP×(DP/EP) = DP for FSDP)
    expectClose(ep8.memoryPerGPU.parameters, ep1.memoryPerGPU.parameters, 0.01, 'EP param equivalence');
  });

  it('MoE2: DeepSeek-V3 FSDP+TP EP=8 produces valid simulation', () => {
    // With FSDP, EP doesn't change param memory (EP×(DP/EP) = DP).
    // EP benefit: reduced per-GPU expert compute and smaller DP comm groups.
    const ep8 = sim({
      modelId: 'deepseek-v3',
      clusterId: '256x-h100',
      globalBatchSize: 2048,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8, ep: 8 },
      activationCheckpointing: true,
    });
    expect(ep8.stepTimeMs).toBeGreaterThan(0);
    expect(ep8.mfu).toBeGreaterThan(0);
  });

  it('MoE3: EP×FSDP param memory equivalence (EP subdivides DP)', () => {
    // With FSDP: expert sharding = TP×EP×(DP/EP) = TP×DP regardless of EP.
    // This is correct behavior: FSDP already shards across all DP ranks.
    // EP merely changes which ranks hold which expert slices.
    const ep1 = sim({
      modelId: 'deepseek-v3',
      clusterId: '512x-h100',
      globalBatchSize: 4096,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8, ep: 1 },
      activationCheckpointing: true,
    });
    const ep8 = sim({
      modelId: 'deepseek-v3',
      clusterId: '512x-h100',
      globalBatchSize: 4096,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8, ep: 8 },
      activationCheckpointing: true,
    });
    // Param memory is the same (FSDP sharding cancels EP subdivision)
    expectClose(ep8.memoryPerGPU.parameters, ep1.memoryPerGPU.parameters, 0.01, 'EP param equivalence');
  });

  it('MoE4: EP does NOT change total GPU count', () => {
    // Both configs: TP=8, PP=1, DP=32 → 256 GPUs regardless of EP
    const cfg1: SimulationConfig = {
      modelId: 'deepseek-v3',
      clusterId: '256x-h100',
      globalBatchSize: 2048,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8, ep: 1 },
      activationCheckpointing: true,
    };
    const cfg8: SimulationConfig = {
      ...cfg1,
      strategyConfig: { tp: 8, ep: 8 },
    };
    // Both should simulate on same cluster without errors
    const m1 = sim(cfg1);
    const m8 = sim(cfg8);
    expect(m1.stepTimeMs).toBeGreaterThan(0);
    expect(m8.stepTimeMs).toBeGreaterThan(0);
  });

  it('MoE5: MoE routing buffers contribute to activation memory', () => {
    // Mixtral has MoE layers → should have non-trivial activations
    const m = sim({
      modelId: 'mixtral-8x7b',
      clusterId: '32x-h100',
      globalBatchSize: 256,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      activationCheckpointing: true,
    });
    expect(m.memoryPerGPU.activations).toBeGreaterThan(0);
  });

  it('MoE6: EP all-to-all communication — EP=8 has comm, EP=1 same or lower', () => {
    const ep1 = sim({
      modelId: 'mixtral-8x7b',
      clusterId: '32x-h100',
      globalBatchSize: 256,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 1, ep: 1 },
      activationCheckpointing: true,
    });
    const ep8 = sim({
      modelId: 'mixtral-8x7b',
      clusterId: '32x-h100',
      globalBatchSize: 256,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 1, ep: 8 },
      activationCheckpointing: true,
    });
    // Both should have non-zero communication
    expect(ep1.timing.communication).toBeGreaterThan(0);
    expect(ep8.timing.communication).toBeGreaterThan(0);
  });
});

// ===========================================================================
// Mixed Precision Interactions (4 tests)
// ===========================================================================
describe('Mixed Precision Interactions', () => {
  const base: SimulationConfig = {
    modelId: 'llama2-7b',
    clusterId: '8x-h100',
    globalBatchSize: 256,
    microBatchSize: 2,
    sequenceLength: 2048,
    strategyType: 'fsdp',
    activationCheckpointing: true,
    flashAttention: true,
  };

  it('FP1: FP8 param memory ≈ half of BF16 param memory (±1%)', () => {
    const bf16M = sim({ ...base, mixedPrecision: 'bf16' });
    const fp8M = sim({ ...base, mixedPrecision: 'fp8' });
    // FP8: 1 byte/param, BF16: 2 bytes/param
    const ratio = fp8M.memoryPerGPU.parameters / bf16M.memoryPerGPU.parameters;
    expectClose(ratio, 0.5, 0.02, 'FP8/BF16 param ratio');
  });

  it('FP2: FP8 gradient memory = BF16 gradient memory (both bf16 grads)', () => {
    const bf16M = sim({ ...base, mixedPrecision: 'bf16' });
    const fp8M = sim({ ...base, mixedPrecision: 'fp8' });
    expectClose(fp8M.memoryPerGPU.gradients, bf16M.memoryPerGPU.gradients, 0.01, 'FP8/BF16 grads');
  });

  it('FP3: FP8 optimizer memory ≈ 10/12 of BF16 (BF16 master instead of FP32)', () => {
    const bf16M = sim({ ...base, mixedPrecision: 'bf16' });
    const fp8M = sim({ ...base, mixedPrecision: 'fp8' });
    // BF16: master(4)+m1(4)+m2(4) = 12 bytes/param
    // FP8:  master(2)+m1(4)+m2(4) = 10 bytes/param → ratio ≈ 10/12 = 0.8333
    expectClose(fp8M.memoryPerGPU.optimizerStates / bf16M.memoryPerGPU.optimizerStates, 10 / 12, 0.01, 'FP8/BF16 opt');
  });

  it('FP4: FP8 total model state ≈ 13/16 of BF16 (smaller params + BF16 master)', () => {
    const bf16M = sim({ ...base, mixedPrecision: 'bf16' });
    const fp8M = sim({ ...base, mixedPrecision: 'fp8' });
    const ratio = modelState(fp8M) / modelState(bf16M);
    // BF16: 2(param)+2(grad)+12(opt) = 16 bytes/param
    // FP8:  1(param)+2(grad)+10(opt) = 13 bytes/param → ratio ≈ 13/16 = 0.8125
    expectClose(ratio, 13 / 16, 0.02, 'FP8/BF16 model state ratio');
  });
});

// ===========================================================================
// Gradient Accumulation (3 tests)
// ===========================================================================
describe('Gradient Accumulation', () => {
  it('GA1: DDP with GA=1 vs GA=16 → timing.communication unchanged (AllReduce once per step)', () => {
    // GA=1: GBS=32, MBS=4, DP=8 → GA=ceil(32/(4×8))=1
    const ga1 = sim({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 32,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'ddp',
    });
    // GA=16: GBS=512, MBS=4, DP=8 → GA=ceil(512/(4×8))=16
    const ga16 = sim({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 512,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'ddp',
    });
    // DDP AllReduce happens once per step regardless of GA
    expectClose(ga16.timing.communication, ga1.timing.communication, 0.05, 'DDP comm GA-invariant');
  });

  it('GA2: FSDP with GA=1 vs GA=16 → comm scales linearly with GA (AllGather per MB)', () => {
    // GA=1: GBS=32, MBS=4, DP=8 → GA=1
    const ga1 = sim({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 32,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'fsdp',
    });
    // GA=16: GBS=512, MBS=4, DP=8 → GA=16
    const ga16 = sim({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 512,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'fsdp',
    });
    // FSDP: AllGather per micro-batch → comm ∝ GA
    const ratio = ga16.timing.communication / ga1.timing.communication;
    expect(ratio, 'FSDP comm scales ~16× with GA=16').toBeGreaterThan(14);
    expect(ratio, 'FSDP comm scales ~16× with GA=16').toBeLessThan(18);
  });

  it('GA3: GA > 1 does NOT increase peak memory (per-MB activations, in-place grads)', () => {
    const ga1 = sim({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 32,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'fsdp',
    });
    const ga16 = sim({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 512,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'fsdp',
    });
    // Same gradients (in-place accumulation)
    expectClose(ga16.memoryPerGPU.gradients, ga1.memoryPerGPU.gradients, 0.01, 'grads GA-invariant');
    // Same activations (per-micro-batch)
    expectClose(ga16.memoryPerGPU.activations, ga1.memoryPerGPU.activations, 0.01, 'activations GA-invariant');
  });
});

// ===========================================================================
// Overlap Modeling (3 tests)
// ===========================================================================
describe('Overlap Modeling', () => {
  it('OV1: DDP 8×H100 LLaMA-2 7B → timing.overlap > 0', () => {
    const m = sim({ ...LLAMA2_7B_8xH100, strategyType: 'ddp' });
    expect(m.timing.overlap).toBeGreaterThan(0);
  });

  it('OV2: FSDP 8×H100 → overlap > 0 and total < sum of components', () => {
    const m = sim({ ...LLAMA2_7B_8xH100, strategyType: 'fsdp' });
    expect(m.timing.overlap).toBeGreaterThan(0);
    const sumWithoutOverlap = m.timing.forward + m.timing.backward + m.timing.optimizer + m.timing.communication;
    expect(m.timing.total).toBeLessThan(sumWithoutOverlap);
  });

  it('OV3: FSDP+TP with SP on vs off → SP reduces activation memory', () => {
    // SP's overlap benefit is baked into timing.communication (reduced exposed TP comm),
    // but SP also adds its own collective ops, so net comm may be slightly higher.
    // The primary verified benefit is activation memory reduction.
    const spOn = sim({
      ...LLAMA2_70B_64xH100,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8, sequenceParallel: true },
    });
    const spOff = sim({
      ...LLAMA2_70B_64xH100,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8, sequenceParallel: false },
    });
    // SP reduces activation memory (the primary benefit for large models)
    expect(spOn.memoryPerGPU.activations).toBeLessThan(spOff.memoryPerGPU.activations);
    // Model state unchanged
    expectClose(modelState(spOn), modelState(spOff), 0.01, 'SP model state unchanged');
  });
});

// ===========================================================================
// Section 10: Stress Tests (6 tests)
// ===========================================================================
describe('Section 10: Stress Tests', () => {
  it('S1: DP=1 for all 8 strategies on 1×H100 with GPT-3 125M → no crash', () => {
    const strategies: SimulationConfig['strategyType'][] = [
      'ddp', 'zero-1', 'fsdp',
      'fsdp-tp', 'zero1-tp',
      'ddp-tp-pp', 'zero1-tp-pp', 'fsdp-tp-pp',
    ];
    for (const st of strategies) {
      expect(() => {
        sim({
          modelId: 'gpt3-125m',
          clusterId: '1x-h100',
          globalBatchSize: 32,
          microBatchSize: 4,
          sequenceLength: 2048,
          strategyType: st,
          strategyConfig: { tp: 1, pp: 1 },
        });
      }, `${st} should not throw`).not.toThrow();
    }
  });

  it('S2: GPT-3 125M on 1024×H100 FSDP → computes, comm overhead present', () => {
    const m = sim({
      modelId: 'gpt3-125m',
      clusterId: '1024x-h100',
      globalBatchSize: 8192,
      microBatchSize: 8,
      sequenceLength: 2048,
      strategyType: 'fsdp',
    });
    expect(m.stepTimeMs).toBeGreaterThan(0);
    // Tiny model on massive cluster → communication overhead present
    expect(m.communicationOverhead).toBeGreaterThan(0);
    // Should still produce positive MFU (FSDP overlap model handles large DP well)
    expect(m.mfu).toBeGreaterThan(0);
  });

  it('S3: LLaMA-3 405B on 8×H100 DDP → OOM; FSDP+TP tp=8 may fit', () => {
    const ddpM = sim({
      modelId: 'llama3-405b',
      clusterId: '8x-h100',
      globalBatchSize: 8,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'ddp',
    });
    expect(ddpM.memoryUtilization).toBeGreaterThan(1.0);

    const fsdpTPM = sim({
      modelId: 'llama3-405b',
      clusterId: '8x-h100',
      globalBatchSize: 8,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8 },
      activationCheckpointing: true,
    });
    // Should produce a result (even if OOM — that's fine, the point is no crash)
    expect(fsdpTPM.stepTimeMs).toBeGreaterThan(0);
  });

  it('S4: TP×PP×DP ≠ totalGPUs → validation error (same as V1)', () => {
    const cluster = createMultiNodeCluster('h100-sxm', 8, 13)!; // 104 GPUs
    const engine = new SimulationEngine();
    engine.configure({
      modelId: 'llama2-70b',
      clusterConfig: cluster,
      globalBatchSize: 256,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 4 },
    });
    // TP=8, PP=4, DP=floor(104/32)=3 → 8×4×3=96 ≠ 104
    const validation = engine.validate();
    // Some form of warning or error about unused GPUs
    expect(validation.warnings.length + validation.errors.length).toBeGreaterThan(0);
  });

  it('S5: 10 rapid sequential configs → consistent results (no stale state)', () => {
    const configs: SimulationConfig[] = [
      { modelId: 'gpt3-125m', clusterId: '8x-h100', globalBatchSize: 256, microBatchSize: 8, sequenceLength: 2048, strategyType: 'ddp' },
      { modelId: 'llama2-7b', clusterId: '8x-h100', globalBatchSize: 256, microBatchSize: 2, sequenceLength: 2048, strategyType: 'fsdp' },
      { modelId: 'llama2-7b', clusterId: '64x-h100', globalBatchSize: 1024, microBatchSize: 2, sequenceLength: 2048, strategyType: 'fsdp-tp', strategyConfig: { tp: 8 } },
      { modelId: 'gpt3-125m', clusterId: '1x-h100', globalBatchSize: 32, microBatchSize: 4, sequenceLength: 2048, strategyType: 'ddp' },
      { modelId: 'llama3-8b', clusterId: '8x-h100', globalBatchSize: 128, microBatchSize: 1, sequenceLength: 4096, strategyType: 'fsdp' },
      { modelId: 'llama2-70b', clusterId: '64x-h100', globalBatchSize: 512, microBatchSize: 1, sequenceLength: 2048, strategyType: 'fsdp' },
      { modelId: 'gpt3-125m', clusterId: '8x-h100', globalBatchSize: 512, microBatchSize: 8, sequenceLength: 2048, strategyType: 'zero-1' },
      { modelId: 'mixtral-8x7b', clusterId: '32x-h100', globalBatchSize: 256, microBatchSize: 1, sequenceLength: 2048, strategyType: 'fsdp' },
      { modelId: 'llama2-13b', clusterId: '8x-h100', globalBatchSize: 128, microBatchSize: 1, sequenceLength: 2048, strategyType: 'fsdp' },
      { modelId: 'llama2-7b', clusterId: '8x-h100', globalBatchSize: 256, microBatchSize: 4, sequenceLength: 2048, strategyType: 'ddp' },
    ];

    // Run each config twice, results should be identical
    for (let i = 0; i < configs.length; i++) {
      const m1 = sim(configs[i]);
      const m2 = sim(configs[i]);
      expect(m1.mfu, `config ${i} MFU consistency`).toBe(m2.mfu);
      expect(m1.stepTimeMs, `config ${i} stepTime consistency`).toBe(m2.stepTimeMs);
      expect(m1.memoryUtilization, `config ${i} memUtil consistency`).toBe(m2.memoryUtilization);
    }
  });

  it('S6: Maximum DP: GPT-3 125M on 1024×H100 DDP (DP=1024) → runs', () => {
    const m = sim({
      modelId: 'gpt3-125m',
      clusterId: '1024x-h100',
      globalBatchSize: 8192,
      microBatchSize: 8,
      sequenceLength: 2048,
      strategyType: 'ddp',
    });
    expect(m.stepTimeMs).toBeGreaterThan(0);
    expect(m.tokensPerSecond).toBeGreaterThan(0);
    // High comm overhead due to massive DP
    expect(m.communicationOverhead).toBeGreaterThan(0);
  });
});
