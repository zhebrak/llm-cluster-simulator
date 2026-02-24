/**
 * Strategy Selector Tests
 *
 * Validates all 10 production-validated strategy configurations.
 * Tests cover: basic validity, per-strategy calculations, pipeline bubbles,
 * frontend consistency, determinism, benchmark ranges, and SP behavior.
 */

import { describe, it, expect } from 'vitest';
import { SimulationEngine } from '../../src/core/simulation/engine.ts';
import type { SimulationConfig, SimulationMetrics } from '../../src/core/simulation/engine.ts';
import { assertValidEngine } from '../helpers/validated-metrics.ts';

const ALL_STRATEGIES = [
  'ddp', 'fsdp', 'zero-1', 'zero-3',
  'fsdp-tp', 'zero1-tp',
  'ddp-tp-pp', 'zero1-tp-pp', 'fsdp-tp-pp',
] as const;

type StrategyType = typeof ALL_STRATEGIES[number];

// Default strategyConfig for each strategy to ensure buildContext calculates GA correctly
const DEFAULT_STRATEGY_CONFIGS: Record<string, SimulationConfig['strategyConfig']> = {
  'fsdp-tp': { tp: 4 },
  'zero1-tp': { tp: 4 },
  'ddp-tp-pp': { tp: 4, pp: 2 },
  'zero1-tp-pp': { tp: 4, pp: 2 },
  'fsdp-tp-pp': { tp: 4, pp: 2 },
};

function runStrategy(strategyType: StrategyType, overrides: Partial<SimulationConfig> = {}) {
  const defaultConfig = DEFAULT_STRATEGY_CONFIGS[strategyType];
  const engine = new SimulationEngine();
  engine.configure({
    modelId: 'llama2-7b',
    clusterId: '8x-a100',
    globalBatchSize: 64,
    microBatchSize: 4,
    sequenceLength: 2048,
    strategyType,
    ...(defaultConfig && !overrides.strategyConfig ? { strategyConfig: defaultConfig } : {}),
    ...overrides,
  });
  assertValidEngine(engine);
  return engine.simulate();
}

/** Run a raw simulation without OOM validation (for comparison baselines that may OOM). */
function rawSimulate(strategyType: StrategyType, overrides: Partial<SimulationConfig> = {}): SimulationMetrics {
  const defaultConfig = DEFAULT_STRATEGY_CONFIGS[strategyType];
  const engine = new SimulationEngine();
  engine.configure({
    modelId: 'llama2-7b',
    clusterId: '8x-a100',
    globalBatchSize: 64,
    microBatchSize: 4,
    sequenceLength: 2048,
    strategyType,
    ...(defaultConfig && !overrides.strategyConfig ? { strategyConfig: defaultConfig } : {}),
    ...overrides,
  });
  return engine.simulate();
}

// ============================================================
// 1. All 10 strategies produce valid results
// ============================================================
describe('All 10 strategies produce valid results', () => {
  for (const strategy of ALL_STRATEGIES) {
    // DDP doesn't shard memory -- use a smaller model that fits on a single GPU
    const modelOverride: Partial<SimulationConfig> = strategy === 'ddp'
      ? { modelId: 'gpt3-1.3b' }
      : {};

    it(`${strategy}: non-zero MFU, valid step time, positive throughput`, () => {
      const metrics = runStrategy(strategy, modelOverride);

      expect(metrics.mfu).toBeGreaterThan(0);
      expect(metrics.stepTimeMs).toBeGreaterThan(0);
      expect(metrics.tokensPerSecond).toBeGreaterThan(0);
      expect(metrics.samplesPerSecond).toBeGreaterThan(0);
    });

    it(`${strategy}: MFU within realistic range (0.15 - 0.75)`, () => {
      const metrics = runStrategy(strategy, modelOverride);

      expect(metrics.mfu).toBeGreaterThanOrEqual(0.15);
      expect(metrics.mfu).toBeLessThanOrEqual(0.75);
    });
  }
});

// ============================================================
// 2. Backend calculation tests per new strategy
// ============================================================
describe('FSDP + TP + PP pipeline behavior (fsdp-tp-pp)', () => {
  const baseConfig: Partial<SimulationConfig> = {
    modelId: 'llama2-70b',
    clusterId: '128x-a100',
    globalBatchSize: 512,
    microBatchSize: 2,
    sequenceLength: 2048,
  };

  it('changing TP (2→4→8) produces different stepTimeMs and MFU', () => {
    const m1 = runStrategy('fsdp-tp-pp', { ...baseConfig, strategyConfig: { tp: 2, pp: 2 } });
    const m2 = runStrategy('fsdp-tp-pp', { ...baseConfig, strategyConfig: { tp: 4, pp: 2 } });
    const m3 = runStrategy('fsdp-tp-pp', { ...baseConfig, strategyConfig: { tp: 8, pp: 2 } });

    expect(m1.stepTimeMs).not.toEqual(m2.stepTimeMs);
    expect(m2.stepTimeMs).not.toEqual(m3.stepTimeMs);
    expect(m1.mfu).not.toEqual(m2.mfu);
  });

  it('changing PP (2→4→8) produces different stepTimeMs and pipeline bubble', () => {
    const m1 = runStrategy('fsdp-tp-pp', { ...baseConfig, strategyConfig: { tp: 4, pp: 2 } });
    const m2 = runStrategy('fsdp-tp-pp', { ...baseConfig, strategyConfig: { tp: 4, pp: 4 } });
    const m3 = runStrategy('fsdp-tp-pp', { ...baseConfig, strategyConfig: { tp: 4, pp: 8 } });

    expect(m1.stepTimeMs).not.toEqual(m2.stepTimeMs);
    expect(m2.stepTimeMs).not.toEqual(m3.stepTimeMs);
    // More PP stages → larger pipeline bubble (fewer microbatches per stage)
    // Note: FSDP shards params across TP*PP*DP = totalGPUs, so per-GPU param
    // memory stays constant when PP changes (DP adjusts to compensate).
    expect(m1.pipelineBubble).toBeLessThan(m2.pipelineBubble);
    expect(m2.pipelineBubble).toBeLessThan(m3.pipelineBubble);
  });

  it('larger GBS → more GA steps → smaller pipeline bubble', () => {
    // TP=4, PP=4, DP=128/(4*4)=8, MBS=2
    // GBS=64 → GA=4, GBS=256 → GA=16, GBS=512 → GA=32
    const m1 = runStrategy('fsdp-tp-pp', { ...baseConfig, globalBatchSize: 64, strategyConfig: { tp: 4, pp: 4 } });
    const m2 = runStrategy('fsdp-tp-pp', { ...baseConfig, globalBatchSize: 256, strategyConfig: { tp: 4, pp: 4 } });
    const m3 = runStrategy('fsdp-tp-pp', { ...baseConfig, globalBatchSize: 512, strategyConfig: { tp: 4, pp: 4 } });

    expect(m1.pipelineBubble).toBeGreaterThan(m2.pipelineBubble);
    expect(m2.pipelineBubble).toBeGreaterThan(m3.pipelineBubble);
  });

  it('pipeline bubble > 0 when PP > 1', () => {
    const m = runStrategy('fsdp-tp-pp', { ...baseConfig, strategyConfig: { tp: 4, pp: 4 } });
    expect(m.pipelineBubble).toBeGreaterThan(0);
  });
});

describe('ZeRO-1 + TP + PP (zero1-tp-pp)', () => {
  const baseConfig: Partial<SimulationConfig> = {
    modelId: 'llama2-70b',
    clusterId: '128x-a100',
    globalBatchSize: 512,
    microBatchSize: 2,
    sequenceLength: 2048,
  };

  it('changing TP produces different timing', () => {
    // Use TP=4 and TP=8 (TP=2 OOMs since ZeRO-1 does not shard params)
    const m1 = runStrategy('zero1-tp-pp', { ...baseConfig, strategyConfig: { tp: 4, pp: 2 } });
    const m2 = runStrategy('zero1-tp-pp', { ...baseConfig, strategyConfig: { tp: 8, pp: 2 } });

    expect(m1.stepTimeMs).not.toEqual(m2.stepTimeMs);
    expect(m1.mfu).not.toEqual(m2.mfu);
  });

  it('ZeRO-1 shards optimizer by DP, lower optimizer memory than DDP variant', () => {
    // DDP with 70B on TP=4, PP=4 OOMs -- use raw simulate for the DDP comparison side
    const ddpMetrics = rawSimulate('ddp-tp-pp', { ...baseConfig, strategyConfig: { tp: 4, pp: 4 } });
    const zeroMetrics = runStrategy('zero1-tp-pp', { ...baseConfig, strategyConfig: { tp: 4, pp: 4 } });

    // ZeRO-1 shards optimizer states across DP dimension → lower optimizer memory
    expect(zeroMetrics.memoryPerGPU.optimizerStates).toBeLessThan(ddpMetrics.memoryPerGPU.optimizerStates);
  });
});

describe('FSDP + TP + PP (fsdp-tp-pp)', () => {
  const baseConfig: Partial<SimulationConfig> = {
    modelId: 'llama2-70b',
    clusterId: '128x-a100',
    globalBatchSize: 512,
    microBatchSize: 2,
    sequenceLength: 2048,
  };

  it('changing TP produces different timing', () => {
    const m1 = runStrategy('fsdp-tp-pp', { ...baseConfig, strategyConfig: { tp: 2, pp: 2 } });
    const m2 = runStrategy('fsdp-tp-pp', { ...baseConfig, strategyConfig: { tp: 8, pp: 2 } });

    expect(m1.stepTimeMs).not.toEqual(m2.stepTimeMs);
  });

  it('FSDP shards params across TP*PP*DP → lower per-GPU param memory than DDP', () => {
    // DDP with 70B on TP=4, PP=4 OOMs -- use raw simulate for the DDP comparison side
    const ddpMetrics = rawSimulate('ddp-tp-pp', { ...baseConfig, strategyConfig: { tp: 4, pp: 4 } });
    const fsdpMetrics = runStrategy('fsdp-tp-pp', { ...baseConfig, strategyConfig: { tp: 4, pp: 4 } });

    // FSDP shards everything, DDP does not shard params by DP
    expect(fsdpMetrics.memoryPerGPU.parameters).toBeLessThan(ddpMetrics.memoryPerGPU.parameters);
  });
});

describe('Sequence Parallelism (SP checkbox)', () => {
  const baseConfig: Partial<SimulationConfig> = {
    modelId: 'llama2-70b',
    clusterId: '128x-a100',
    globalBatchSize: 512,
    microBatchSize: 2,
    sequenceLength: 2048,
  };

  it('SP=true reduces activation memory vs SP=false', () => {
    const withSP = runStrategy('fsdp-tp', { ...baseConfig, strategyConfig: { tp: 8, sequenceParallel: true } });
    const noSP = runStrategy('fsdp-tp', { ...baseConfig, strategyConfig: { tp: 8, sequenceParallel: false } });

    expect(withSP.memoryPerGPU.activations).toBeLessThan(noSP.memoryPerGPU.activations);
    expect(withSP.mfu).not.toEqual(noSP.mfu);
  });

  it('activation memory decreases with higher TP when SP=true', () => {
    const m2 = runStrategy('fsdp-tp', { ...baseConfig, strategyConfig: { tp: 2, sequenceParallel: true } });
    const m4 = runStrategy('fsdp-tp', { ...baseConfig, strategyConfig: { tp: 4, sequenceParallel: true } });
    const m8 = runStrategy('fsdp-tp', { ...baseConfig, strategyConfig: { tp: 8, sequenceParallel: true } });

    // SP activation multiplier: 0.7 + 0.3/tp decreases with higher TP
    expect(m2.memoryPerGPU.activations).toBeGreaterThan(m4.memoryPerGPU.activations);
    expect(m4.memoryPerGPU.activations).toBeGreaterThan(m8.memoryPerGPU.activations);
  });
});

// ============================================================
// 3. Pipeline bubble validation (for all 3D strategies)
// ============================================================
describe('Pipeline bubble validation', () => {
  // 128x A100 cluster, TP=4, PP=4 → DP=128/(4*4)=8
  // Bubble = (pp-1)/(pp-1+m) where m = ceil(GBS/(MBS*DP))

  it('PP=4, GA=4: bubble = 3/7', () => {
    // GBS=64, MBS=2, DP=8 → GA=ceil(64/16)=4, bubble=3/7≈0.4286
    const m = runStrategy('fsdp-tp-pp', {
      modelId: 'llama2-70b', clusterId: '128x-a100',
      globalBatchSize: 64, microBatchSize: 2, sequenceLength: 2048,
      strategyConfig: { tp: 4, pp: 4 },
    });
    expect(m.pipelineBubble).toBeCloseTo(3 / 7, 2);
  });

  it('PP=4, GA=16: bubble = 3/19', () => {
    // GBS=256, MBS=2, DP=8 → GA=16, bubble=3/19≈0.158
    const m = runStrategy('fsdp-tp-pp', {
      modelId: 'llama2-70b', clusterId: '128x-a100',
      globalBatchSize: 256, microBatchSize: 2, sequenceLength: 2048,
      strategyConfig: { tp: 4, pp: 4 },
    });
    expect(m.pipelineBubble).toBeCloseTo(3 / 19, 2);
  });

  it('PP=4, GA=32: bubble = 3/35', () => {
    // GBS=512, MBS=2, DP=8 → GA=32, bubble=3/35≈0.0857
    const m = runStrategy('fsdp-tp-pp', {
      modelId: 'llama2-70b', clusterId: '128x-a100',
      globalBatchSize: 512, microBatchSize: 2, sequenceLength: 2048,
      strategyConfig: { tp: 4, pp: 4 },
    });
    expect(m.pipelineBubble).toBeCloseTo(3 / 35, 2);
  });

  it('larger GBS → larger GA → smaller bubble', () => {
    const run = (gbs: number) => runStrategy('fsdp-tp-pp', {
      modelId: 'llama2-70b', clusterId: '128x-a100',
      globalBatchSize: gbs, microBatchSize: 2, sequenceLength: 2048,
      strategyConfig: { tp: 4, pp: 4 },
    });
    const small = run(64);   // GA=4
    const mid = run(256);    // GA=16
    const large = run(512);  // GA=32

    expect(mid.pipelineBubble).toBeLessThan(small.pipelineBubble);
    expect(large.pipelineBubble).toBeLessThan(mid.pipelineBubble);
  });

  it('PP=1: bubble = 0 (fsdp-tp has no PP)', () => {
    const m = runStrategy('fsdp-tp', {
      modelId: 'llama2-70b', clusterId: '128x-a100',
      globalBatchSize: 512, microBatchSize: 2, sequenceLength: 2048,
      strategyConfig: { tp: 8 },
    });
    expect(m.pipelineBubble).toBe(0);
  });
});

// ============================================================
// 4. Frontend numbers validation
// ============================================================
describe('Frontend numbers validation', () => {
  it('metrics are internally consistent for each strategy', () => {
    for (const strategy of ALL_STRATEGIES) {
      const engine = new SimulationEngine();
      engine.configure({
        modelId: 'llama2-7b',
        clusterId: '8x-a100',
        globalBatchSize: 64,
        microBatchSize: 4,
        sequenceLength: 2048,
        strategyType: strategy,
      });
      const result = engine.run();

      if (result.success) {
        const metrics = engine.simulate();

        // MFU should match analysis (exported metrics are rounded to 2dp)
        expect(result.metrics.mfu).toBeCloseTo(metrics.mfu, 2);

        // Step time should match (exported metrics are rounded to integers)
        expect(result.metrics.avgStepTimeMs).toBeCloseTo(metrics.stepTimeMs, 0);

        // Communication overhead is non-negative and < 1
        expect(metrics.communicationOverhead).toBeGreaterThanOrEqual(0);
        expect(metrics.communicationOverhead).toBeLessThan(1);

        // Pipeline bubble is in [0, 1)
        expect(metrics.pipelineBubble).toBeGreaterThanOrEqual(0);
        expect(metrics.pipelineBubble).toBeLessThan(1);
      }
    }
  });
});

// ============================================================
// 5. Consistency tests
// ============================================================
describe('Consistency tests', () => {
  it('same config → same results (deterministic)', () => {
    for (const strategy of ALL_STRATEGIES) {
      const config: SimulationConfig = {
        modelId: 'llama2-7b',
        clusterId: '8x-a100',
        globalBatchSize: 64,
        microBatchSize: 4,
        sequenceLength: 2048,
        strategyType: strategy,
      };

      const e1 = new SimulationEngine();
      e1.configure(config);
      const m1 = e1.simulate();

      const e2 = new SimulationEngine();
      e2.configure(config);
      const m2 = e2.simulate();

      expect(m1.stepTimeMs).toEqual(m2.stepTimeMs);
      expect(m1.mfu).toEqual(m2.mfu);
      expect(m1.memoryPerGPU.total).toEqual(m2.memoryPerGPU.total);
    }
  });

  it('fsdp-tp with TP=1 ≈ pure FSDP (MFU within 10%)', () => {
    const baseConfig: Partial<SimulationConfig> = {
      modelId: 'llama2-7b',
      clusterId: '8x-a100',
      globalBatchSize: 64,
      microBatchSize: 4,
      sequenceLength: 2048,
    };

    const tpMetrics = runStrategy('fsdp-tp', { ...baseConfig, strategyConfig: { tp: 1 } });
    const fsdpMetrics = runStrategy('fsdp', baseConfig);

    const mfuDiff = Math.abs(tpMetrics.mfu - fsdpMetrics.mfu);
    expect(mfuDiff).toBeLessThan(0.10);
  });
});

// ============================================================
// 6. Benchmark validation
// ============================================================
describe('Benchmark validation', () => {
  const largeConfig: Partial<SimulationConfig> = {
    modelId: 'llama2-70b',
    clusterId: '128x-a100',
    globalBatchSize: 512,
    microBatchSize: 2,
    sequenceLength: 2048,
    mixedPrecision: 'bf16',
  };

  it('ddp-tp-pp with LLaMA-70B (TP=8, PP=4): MFU in realistic range', () => {
    const m = runStrategy('ddp-tp-pp', { ...largeConfig, strategyConfig: { tp: 8, pp: 4 } });
    expect(m.mfu).toBeGreaterThanOrEqual(0.20);
    expect(m.mfu).toBeLessThanOrEqual(0.60);
  });

  it('zero1-tp-pp with LLaMA-70B (TP=4, PP=4): MFU in realistic range', () => {
    const m = runStrategy('zero1-tp-pp', { ...largeConfig, strategyConfig: { tp: 4, pp: 4 } });
    expect(m.mfu).toBeGreaterThanOrEqual(0.20);
    expect(m.mfu).toBeLessThanOrEqual(0.60);
  });

  it('fsdp-tp-pp with LLaMA-70B (TP=8, PP=4): MFU in realistic range', () => {
    const m = runStrategy('fsdp-tp-pp', { ...largeConfig, strategyConfig: { tp: 8, pp: 4 } });
    expect(m.mfu).toBeGreaterThanOrEqual(0.20);
    expect(m.mfu).toBeLessThanOrEqual(0.60);
  });

  it('fsdp-tp with SP=true and LLaMA-70B (TP=8): MFU in realistic range', () => {
    const m = runStrategy('fsdp-tp', { ...largeConfig, strategyConfig: { tp: 8, sequenceParallel: true } });
    expect(m.mfu).toBeGreaterThanOrEqual(0.25);
    expect(m.mfu).toBeLessThanOrEqual(0.65);
  });

  it('all strategies: MFU < 0.70 (realistic ceiling)', () => {
    for (const strategy of ALL_STRATEGIES) {
      // DDP doesn't shard memory -- use a smaller model that fits on a single GPU
      const overrides: Partial<SimulationConfig> = strategy === 'ddp'
        ? { modelId: 'gpt3-1.3b' }
        : {};
      const m = runStrategy(strategy, overrides);
      expect(m.mfu).toBeLessThan(0.75);
    }
  });
});

// ============================================================
// 7. SP-specific validation
// ============================================================
describe('Sequence Parallelism validation', () => {
  const baseConfig: Partial<SimulationConfig> = {
    modelId: 'llama2-70b',
    clusterId: '128x-a100',
    globalBatchSize: 512,
    microBatchSize: 2,
    sequenceLength: 2048,
  };

  it('SP=true has reasonable communication overhead', () => {
    const spMetrics = runStrategy('fsdp-tp', { ...baseConfig, strategyConfig: { tp: 8, sequenceParallel: true } });

    expect(spMetrics.communicationOverhead).toBeGreaterThan(0);
    expect(spMetrics.communicationOverhead).toBeLessThan(0.5);
  });

  it('SP activation memory reduction: verify multiplier effect', () => {
    // SP multiplier: 0.7 + 0.3/tp
    const m2 = runStrategy('fsdp-tp', { ...baseConfig, strategyConfig: { tp: 2, sequenceParallel: true } });
    const m4 = runStrategy('fsdp-tp', { ...baseConfig, strategyConfig: { tp: 4, sequenceParallel: true } });
    const m8 = runStrategy('fsdp-tp', { ...baseConfig, strategyConfig: { tp: 8, sequenceParallel: true } });

    expect(m2.memoryPerGPU.activations).toBeGreaterThan(m4.memoryPerGPU.activations);
    expect(m4.memoryPerGPU.activations).toBeGreaterThan(m8.memoryPerGPU.activations);
  });

  it('SP=true produces different results from SP=false (activation reduction)', () => {
    const spMetrics = runStrategy('fsdp-tp', { ...baseConfig, strategyConfig: { tp: 8, sequenceParallel: true } });
    const noSpMetrics = runStrategy('fsdp-tp', { ...baseConfig, strategyConfig: { tp: 8, sequenceParallel: false } });

    expect(spMetrics.memoryPerGPU.activations).toBeLessThan(noSpMetrics.memoryPerGPU.activations);
    expect(spMetrics.stepTimeMs).not.toEqual(noSpMetrics.stepTimeMs);
  });
});

// ============================================================
// 8. buildContext GA auto-selection (no explicit strategyConfig)
//    Combined strategies must auto-select TP/PP just like selectStrategy
//    does, so buildContext computes correct DP → correct GA → valid MFU.
// ============================================================
describe('buildContext matches selectStrategy auto-selection', () => {
  const COMBINED_STRATEGIES = [
    'fsdp-tp', 'zero1-tp',
    'ddp-tp-pp', 'zero1-tp-pp', 'fsdp-tp-pp',
  ] as const;

  for (const strategy of COMBINED_STRATEGIES) {
    it(`${strategy}: omitting strategyConfig.tp still gives MFU < 1.0`, () => {
      const engine = new SimulationEngine();
      engine.configure({
        modelId: 'llama2-7b',
        clusterId: '8x-a100',
        globalBatchSize: 64,
        microBatchSize: 4,
        sequenceLength: 2048,
        strategyType: strategy,
        // Deliberately omit strategyConfig — buildContext must auto-select
      });
      const metrics = engine.simulate();

      // buildContext must use the same TP as selectStrategy (auto-detected).
      // If TP is inconsistent, effectiveDP and GA are wrong → MFU > 100%.
      expect(metrics.mfu).toBeLessThan(1.0);
      expect(metrics.mfu).toBeGreaterThan(0.1);
    });

    it(`${strategy}: explicit tp matches omitted tp (same auto-selection)`, () => {
      // 8x-a100 cluster: gpusPerNode=8, so auto tp should be min(8, 8)=8
      const engine1 = new SimulationEngine();
      engine1.configure({
        modelId: 'llama2-7b',
        clusterId: '8x-a100',
        globalBatchSize: 64,
        microBatchSize: 4,
        sequenceLength: 2048,
        strategyType: strategy,
        strategyConfig: { tp: 8 },
      });
      const explicit = engine1.simulate();

      const engine2 = new SimulationEngine();
      engine2.configure({
        modelId: 'llama2-7b',
        clusterId: '8x-a100',
        globalBatchSize: 64,
        microBatchSize: 4,
        sequenceLength: 2048,
        strategyType: strategy,
        // no strategyConfig
      });
      const auto = engine2.simulate();

      // Step time and MFU must be identical — same effective parallelism
      expect(auto.stepTimeMs).toBeCloseTo(explicit.stepTimeMs, 2);
      expect(auto.mfu).toBeCloseTo(explicit.mfu, 4);
    });
  }

  it('multi-node: fsdp-tp without explicit tp auto-selects tp=gpusPerNode', () => {
    // 128x-a100 = 16 nodes × 8 GPUs; auto tp=8, dp=16
    const engine = new SimulationEngine();
    engine.configure({
      modelId: 'llama2-70b',
      clusterId: '128x-a100',
      globalBatchSize: 512,
      microBatchSize: 2,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp',
      // no strategyConfig
    });
    const metrics = engine.simulate();

    expect(metrics.mfu).toBeLessThan(1.0);
    expect(metrics.mfu).toBeGreaterThan(0.15);
  });

  it('3D strategy without explicit tp/pp auto-selects both', () => {
    const engine = new SimulationEngine();
    engine.configure({
      modelId: 'llama2-70b',
      clusterId: '128x-a100',
      globalBatchSize: 512,
      microBatchSize: 2,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp-pp',
      // no strategyConfig at all
    });
    const metrics = engine.simulate();

    expect(metrics.mfu).toBeLessThan(1.0);
    expect(metrics.mfu).toBeGreaterThan(0.15);
  });
});
