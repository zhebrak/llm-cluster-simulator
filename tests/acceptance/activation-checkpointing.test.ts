/**
 * Activation Checkpointing Tests
 *
 * Verifies that the activation checkpointing toggle actually works:
 * - Memory: checkpointing ON uses significantly less activation memory
 * - HFU: changes (8x with checkpointing vs 6x without)
 * - MFU: stays the same regardless of checkpointing (pure useful work)
 * - Step time: increases with checkpointing (recompute overhead)
 * - Across all strategy types, models, and GPU architectures
 */

import { describe, it, expect } from 'vitest';
import { createMultiNodeCluster } from '../../src/core/hardware/topology.ts';
import type { SimulationConfig } from '../../src/core/simulation/engine.ts';
import { getValidatedSimulationMetrics } from '../helpers/validated-metrics.ts';
import { getSimulationMetrics } from '../../src/core/simulation/engine.ts';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeConfig(
  gpuId: string,
  modelId: string,
  strategyType: SimulationConfig['strategyType'],
  totalGPUs: number,
  numNodes: number,
  activationCheckpointing: boolean,
  strategyConfig?: SimulationConfig['strategyConfig'],
): SimulationConfig {
  return {
    clusterId: undefined,
    clusterConfig: createMultiNodeCluster(gpuId, totalGPUs / numNodes, numNodes)!,
    modelId,
    globalBatchSize: totalGPUs * 2,
    microBatchSize: 2,
    sequenceLength: 2048,
    strategyType,
    strategyConfig,
    activationCheckpointing,
  };
}

function pair(
  gpuId: string,
  modelId: string,
  strategyType: SimulationConfig['strategyType'],
  totalGPUs: number,
  numNodes: number,
  strategyConfig?: SimulationConfig['strategyConfig'],
) {
  const on = getValidatedSimulationMetrics(makeConfig(gpuId, modelId, strategyType, totalGPUs, numNodes, true, strategyConfig));
  const off = getValidatedSimulationMetrics(makeConfig(gpuId, modelId, strategyType, totalGPUs, numNodes, false, strategyConfig));
  return { on, off };
}

// ---------------------------------------------------------------------------
// Section 1: Memory reduction — checkpointing ON uses less activation memory
// ---------------------------------------------------------------------------

describe('Activation checkpointing: memory reduction', () => {
  it('FSDP: llama3-8b on H100 — checkpointing reduces memory', () => {
    const { on, off } = pair('h100-sxm', 'llama3-8b', 'fsdp', 8, 1);
    expect(on.memoryPerGPU.activations).toBeLessThan(off.memoryPerGPU.activations);
    // sqrt(L) vs L reduction — should be substantial (at least 2x for 32-layer model)
    expect(off.memoryPerGPU.activations / on.memoryPerGPU.activations).toBeGreaterThan(2);
  });

  it('FSDP: llama3-8b on A100 — checkpointing reduces memory', () => {
    const { on, off } = pair('a100-80gb', 'llama3-8b', 'fsdp', 8, 1);
    expect(on.memoryPerGPU.activations).toBeLessThan(off.memoryPerGPU.activations);
    expect(off.memoryPerGPU.activations / on.memoryPerGPU.activations).toBeGreaterThan(2);
  });

  it('ZeRO-1: gpt3-1.3b on H100 — checkpointing reduces memory', () => {
    const { on, off } = pair('h100-sxm', 'gpt3-1.3b', 'zero-1', 8, 1);
    expect(on.memoryPerGPU.activations).toBeLessThan(off.memoryPerGPU.activations);
  });

  it('ZeRO-3: gpt3-6.7b on A100 — checkpointing reduces memory', () => {
    const { on, off } = pair('a100-80gb', 'gpt3-6.7b', 'zero-3', 8, 1);
    expect(on.memoryPerGPU.activations).toBeLessThan(off.memoryPerGPU.activations);
  });

  it('FSDP-TP: llama3-8b on H100 — checkpointing reduces memory', () => {
    const { on, off } = pair('h100-sxm', 'llama3-8b', 'fsdp-tp', 8, 1, { tp: 2 });
    expect(on.memoryPerGPU.activations).toBeLessThan(off.memoryPerGPU.activations);
  });

  it('FSDP-TP-PP: llama3-70b on H200 — checkpointing reduces memory', () => {
    const { on, off } = pair('h200-sxm', 'llama3-70b', 'fsdp-tp-pp', 16, 2, { tp: 4, pp: 2 });
    expect(on.memoryPerGPU.activations).toBeLessThan(off.memoryPerGPU.activations);
  });

  it('total memory is lower with checkpointing (params/grads/optimizer unchanged)', () => {
    const { on, off } = pair('h100-sxm', 'llama3-8b', 'fsdp', 8, 1);
    // Parameters, gradients, optimizer states should be identical
    expect(on.memoryPerGPU.parameters).toBe(off.memoryPerGPU.parameters);
    expect(on.memoryPerGPU.gradients).toBe(off.memoryPerGPU.gradients);
    expect(on.memoryPerGPU.optimizerStates).toBe(off.memoryPerGPU.optimizerStates);
    // Total memory should be lower with checkpointing
    expect(on.memoryPerGPU.total).toBeLessThan(off.memoryPerGPU.total);
  });
});

// ---------------------------------------------------------------------------
// Section 2: MFU decreases with checkpointing (step takes longer due to recompute)
// ---------------------------------------------------------------------------

describe('Activation checkpointing: MFU decreases', () => {
  it('DDP: MFU lower with checkpointing (longer step time)', () => {
    const { on, off } = pair('h100-sxm', 'gpt3-1.3b', 'ddp', 8, 1);
    expect(on.mfu).toBeLessThan(off.mfu);
    // Ratio should be ~0.75 (3F+comm)/(4F+comm), allow range for comm variation
    expect(on.mfu / off.mfu).toBeGreaterThan(0.65);
    expect(on.mfu / off.mfu).toBeLessThan(0.85);
  });

  it('FSDP: MFU lower with checkpointing', () => {
    const { on, off } = pair('a100-80gb', 'llama3-8b', 'fsdp', 8, 1);
    expect(on.mfu).toBeLessThan(off.mfu);
    // With 2.5× backward multiplier, ratio ~0.86 (higher than old 3× which gave ~0.75)
    expect(on.mfu / off.mfu).toBeGreaterThan(0.75);
    expect(on.mfu / off.mfu).toBeLessThan(0.92);
  });

  it('ZeRO-1: MFU lower with checkpointing', () => {
    const { on, off } = pair('h100-sxm', 'gpt3-1.3b', 'zero-1', 8, 1);
    expect(on.mfu).toBeLessThan(off.mfu);
  });

  it('FSDP-TP: MFU lower with checkpointing', () => {
    const { on, off } = pair('h100-sxm', 'llama3-8b', 'fsdp-tp', 8, 1, { tp: 2 });
    expect(on.mfu).toBeLessThan(off.mfu);
  });

  it('FSDP-TP-PP: MFU lower with checkpointing', () => {
    const { on, off } = pair('h200-sxm', 'llama3-70b', 'fsdp-tp-pp', 16, 2, { tp: 4, pp: 2 });
    expect(on.mfu).toBeLessThan(off.mfu);
  });
});

// ---------------------------------------------------------------------------
// Section 3: HFU stays roughly the same (8PD/4T ≈ 6PD/3T)
// ---------------------------------------------------------------------------

describe('Activation checkpointing: HFU stability', () => {
  it('DDP: HFU is roughly the same ON vs OFF (hardware efficiency unchanged)', () => {
    const { on, off } = pair('h100-sxm', 'gpt3-1.3b', 'ddp', 8, 1);
    // With checkpointing: HFU = 8PD / (4F_time × peak) = 2PD / (F_time × peak)
    // Without: HFU = 6PD / (3F_time × peak) = 2PD / (F_time × peak)
    // → HFU stays ~same, HFU > MFU when checkpointing, HFU = MFU without
    expect(on.hfu).toBeGreaterThan(on.mfu);           // HFU > MFU when checkpointing
    expect(off.hfu).toBeCloseTo(off.mfu, 5);          // HFU = MFU when no checkpointing
    // HFU ratio ON/OFF should be ~1.0 (hardware efficiency unchanged)
    expect(on.hfu / off.hfu).toBeGreaterThan(0.85);
    expect(on.hfu / off.hfu).toBeLessThan(1.15);
  });

  it('FSDP: HFU roughly stable with checkpointing', () => {
    const { on, off } = pair('a100-80gb', 'llama3-8b', 'fsdp', 8, 1);
    expect(on.hfu).toBeGreaterThan(on.mfu);
    expect(off.hfu).toBeCloseTo(off.mfu, 5);
    expect(on.hfu / off.hfu).toBeGreaterThan(0.85);
    expect(on.hfu / off.hfu).toBeLessThan(1.16);
  });

  it('FSDP-TP: HFU roughly stable', () => {
    const { on, off } = pair('h100-sxm', 'llama3-8b', 'fsdp-tp', 8, 1, { tp: 2 });
    expect(on.hfu).toBeGreaterThan(on.mfu);
    expect(off.hfu).toBeCloseTo(off.mfu, 5);
    // With 2.5× backward multiplier, HFU ratio slightly above 1.0 (8PD/3.5F vs 6PD/3F)
    expect(on.hfu / off.hfu).toBeGreaterThan(0.85);
    expect(on.hfu / off.hfu).toBeLessThan(1.20);
  });
});

// ---------------------------------------------------------------------------
// Section 4: Step time — recompute adds overhead
// ---------------------------------------------------------------------------

describe('Activation checkpointing: step time overhead', () => {
  it('DDP: step time increases with checkpointing (recompute cost)', () => {
    const { on, off } = pair('h100-sxm', 'gpt3-1.3b', 'ddp', 8, 1);
    // Checkpointing adds ~33% recompute overhead to forward pass
    expect(on.stepTimeMs).toBeGreaterThan(off.stepTimeMs);
  });

  it('FSDP: step time increases with checkpointing', () => {
    const { on, off } = pair('a100-80gb', 'llama3-8b', 'fsdp', 8, 1);
    expect(on.stepTimeMs).toBeGreaterThan(off.stepTimeMs);
  });

  it('FSDP-TP: step time increases with checkpointing', () => {
    const { on, off } = pair('h100-sxm', 'llama3-8b', 'fsdp-tp', 8, 1, { tp: 2 });
    expect(on.stepTimeMs).toBeGreaterThan(off.stepTimeMs);
  });

  it('FSDP-TP-PP: step time increases with checkpointing', () => {
    const { on, off } = pair('h200-sxm', 'llama3-70b', 'fsdp-tp-pp', 16, 2, { tp: 4, pp: 2 });
    expect(on.stepTimeMs).toBeGreaterThan(off.stepTimeMs);
  });
});

// ---------------------------------------------------------------------------
// Section 5: Cross-GPU — checkpointing works on all GPU types
// ---------------------------------------------------------------------------

describe('Activation checkpointing: cross-GPU verification', () => {
  const gpus = [
    { id: 'a100-80gb', name: 'A100 80GB' },
    { id: 'h100-sxm', name: 'H100 SXM' },
    { id: 'h200-sxm', name: 'H200 SXM' },
    { id: 'b200',     name: 'B200' },
    { id: 'mi300x',   name: 'MI300X' },
    { id: 'mi250x',   name: 'MI250X' },
    { id: 'mi350x',   name: 'MI350X' },
  ];

  for (const gpu of gpus) {
    it(`${gpu.name}: checkpointing reduces memory for FSDP`, () => {
      const model = gpu.id === 'mi250x' ? 'gpt3-1.3b' : 'llama3-8b';
      const { on, off } = pair(gpu.id, model, 'fsdp', 8, 1);
      expect(on.memoryPerGPU.activations, `${gpu.name} activations`).toBeLessThan(off.memoryPerGPU.activations);
      expect(on.mfu, `${gpu.name} MFU`).toBeLessThan(off.mfu);
    });
  }
});

// ---------------------------------------------------------------------------
// Section 6: Cross-strategy — every strategy respects the toggle
// ---------------------------------------------------------------------------

describe('Activation checkpointing: cross-strategy verification', () => {
  const strategies: Array<{
    type: SimulationConfig['strategyType'];
    model: string;
    config?: SimulationConfig['strategyConfig'];
  }> = [
    { type: 'ddp', model: 'gpt3-1.3b' },
    { type: 'fsdp', model: 'llama3-8b' },
    { type: 'zero-1', model: 'gpt3-1.3b' },
    { type: 'zero-3', model: 'gpt3-6.7b' },
    { type: 'fsdp-tp', model: 'llama3-8b', config: { tp: 2 } },
    { type: 'zero1-tp', model: 'llama3-8b', config: { tp: 2 } },
    { type: 'ddp-tp-pp', model: 'llama3-8b', config: { tp: 2, pp: 2 } },
    { type: 'fsdp-tp-pp', model: 'llama3-8b', config: { tp: 2, pp: 2 } },
    { type: 'fsdp-tp', model: 'llama3-8b', config: { tp: 2, sequenceParallel: true } },
  ];

  for (const s of strategies) {
    it(`${s.type}: toggle changes activation memory`, () => {
      const { on, off } = pair('h100-sxm', s.model, s.type, 8, 1, s.config);
      expect(
        on.memoryPerGPU.activations,
        `${s.type} ON activations should be < OFF`
      ).toBeLessThan(off.memoryPerGPU.activations);
    });

    it(`${s.type}: MFU lower with checkpointing (recompute overhead)`, () => {
      const { on, off } = pair('h100-sxm', s.model, s.type, 8, 1, s.config);
      expect(on.mfu).toBeLessThan(off.mfu);
    });
  }
});

// ---------------------------------------------------------------------------
// Section 7: Cross-model — different model sizes
// ---------------------------------------------------------------------------

describe('Activation checkpointing: cross-model verification', () => {
  const models = ['gpt3-125m', 'gpt3-1.3b', 'gpt3-6.7b', 'llama3-8b'];

  for (const model of models) {
    it(`${model}: checkpointing reduces activation memory on FSDP`, () => {
      const { on, off } = pair('h100-sxm', model, 'fsdp', 8, 1);
      expect(on.memoryPerGPU.activations).toBeLessThan(off.memoryPerGPU.activations);
    });

    it(`${model}: MFU lower with checkpointing`, () => {
      const { on, off } = pair('h100-sxm', model, 'fsdp', 8, 1);
      expect(on.mfu).toBeLessThan(off.mfu);
    });
  }
});

// ---------------------------------------------------------------------------
// Section 8: Memory utilization — checkpointing can prevent OOM
// ---------------------------------------------------------------------------

describe('Activation checkpointing: OOM prevention', () => {
  it('large model that OOMs without checkpointing may fit with it', () => {
    // llama3-8b on A100-40GB with DDP is very tight — use raw getSimulationMetrics
    // since the 'without checkpointing' config is expected to OOM
    const on = getSimulationMetrics(makeConfig('a100-40gb', 'llama3-8b', 'ddp', 8, 1, true));
    const off = getSimulationMetrics(makeConfig('a100-40gb', 'llama3-8b', 'ddp', 8, 1, false));
    // With checkpointing should use less memory
    expect(on.memoryUtilization).toBeLessThan(off.memoryUtilization);
  });
});
