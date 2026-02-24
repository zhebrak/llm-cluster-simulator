/**
 * Pipeline Parallelism & Timing Invariant Tests
 *
 * Validates:
 * 1. Pipeline bubble never exceeds 100% (or produces negative efficiency)
 * 2. All timing components are always non-negative
 * 3. MFU is always in [0, 1] for any valid/invalid config
 * 4. numMicroBatches < pp is caught by validation
 * 5. Large-scale 3D-parallel configs (e.g. 405B on 2048 H100s)
 * 6. Pathological edge cases
 */

import { describe, it, expect } from 'vitest';
import { type SimulationConfig } from '../../src/core/simulation/engine.ts';
import { getValidatedSimulationMetrics } from '../helpers/validated-metrics.ts';
import { createMultiNodeCluster } from '../../src/core/hardware/topology.ts';
import {
  create3DParallelStrategy,
  ThreeDParallelStrategy,
} from '../../src/core/strategies/3d-parallel.ts';
import { createPipelineParallelStrategy } from '../../src/core/strategies/pipeline-parallel.ts';
import { getModel } from '../../src/core/models/index.ts';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeConfig(
  gpuId: string,
  modelId: string,
  strategyType: SimulationConfig['strategyType'],
  totalGPUs: number,
  numNodes: number,
  strategyConfig?: SimulationConfig['strategyConfig'],
  overrides?: Partial<SimulationConfig>,
): SimulationConfig {
  return {
    clusterConfig: createMultiNodeCluster(gpuId, totalGPUs / numNodes, numNodes)!,
    modelId,
    globalBatchSize: overrides?.globalBatchSize ?? totalGPUs * 2,
    microBatchSize: overrides?.microBatchSize ?? 2,
    sequenceLength: overrides?.sequenceLength ?? 2048,
    strategyType,
    strategyConfig,
    ...overrides,
  };
}

function assertTimingInvariants(metrics: ReturnType<typeof getValidatedSimulationMetrics>, label: string) {
  // All timing components must be non-negative
  expect(metrics.timing.forward, `${label}: forward >= 0`).toBeGreaterThanOrEqual(0);
  expect(metrics.timing.backward, `${label}: backward >= 0`).toBeGreaterThanOrEqual(0);
  expect(metrics.timing.optimizer, `${label}: optimizer >= 0`).toBeGreaterThanOrEqual(0);
  expect(metrics.timing.communication, `${label}: communication >= 0`).toBeGreaterThanOrEqual(0);
  expect(metrics.timing.overlap, `${label}: overlap >= 0`).toBeGreaterThanOrEqual(0);
  expect(metrics.timing.total, `${label}: total > 0`).toBeGreaterThan(0);

  // Total must be finite
  expect(metrics.timing.total, `${label}: total finite`).toBeLessThan(Infinity);

  // Step time must match total
  expect(metrics.stepTimeMs, `${label}: stepTimeMs > 0`).toBeGreaterThan(0);
  expect(metrics.stepTimeMs, `${label}: stepTimeMs finite`).toBeLessThan(Infinity);

  // MFU must be in [0, 1]
  expect(metrics.mfu, `${label}: mfu >= 0`).toBeGreaterThanOrEqual(0);
  expect(metrics.mfu, `${label}: mfu <= 1`).toBeLessThanOrEqual(1);

  // HFU must be non-negative
  expect(metrics.hfu, `${label}: hfu >= 0`).toBeGreaterThanOrEqual(0);

  // Pipeline bubble must be in [0, 1]
  expect(metrics.pipelineBubble, `${label}: bubble >= 0`).toBeGreaterThanOrEqual(0);
  expect(metrics.pipelineBubble, `${label}: bubble <= 1`).toBeLessThanOrEqual(1);

  // Communication overhead must be in [0, 1]
  expect(metrics.communicationOverhead, `${label}: commOverhead >= 0`).toBeGreaterThanOrEqual(0);
  expect(metrics.communicationOverhead, `${label}: commOverhead <= 1`).toBeLessThanOrEqual(1);
}

// ---------------------------------------------------------------------------
// Section 1: Bubble formula is bounded — no clamping needed
// ---------------------------------------------------------------------------

describe('Bubble formula: (pp-1)/(pp-1+m) is naturally bounded', () => {
  it('create3DParallelStrategy with numMicroBatches < pp produces valid bubble', () => {
    // numMicroBatches=2 with pp=4 → bubble = 3/5 = 0.6 (bounded, no clamping)
    const strategy = create3DParallelStrategy(2, 4, 2, { numMicroBatches: 2 });
    const model = getModel('gpt3-125m', 2048)!;
    const cluster = createMultiNodeCluster('h100-sxm', 4, 4)!;
    const ctx = {
      model,
      cluster,
      training: {
        globalBatchSize: 32,
        microBatchSize: 2,
        sequenceLength: 2048,
        maxSteps: 100,
        optimizer: { type: 'adamw' as const, learningRate: 3e-4, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0.01, fusedOptimizer: false },
        lrSchedule: { type: 'cosine' as const, minLR: 3e-5, warmupSteps: 100, warmupRatio: 0.1 },
        dtypes: { params: 'bf16' as const, compute: 'bf16' as const, gradients: 'bf16' as const, activation: 'bf16' as const, optimizer: 'fp32' as const },
        gradientClipping: 1.0,
        gradientAccumulationSteps: 4,
      },
      seqLength: 2048,
      microBatchSize: 2,
      globalBatchSize: 32,
      gradientAccumulationSteps: 4,
      activationCheckpointing: true,
      flashAttention: true,
    };
    const analysis = strategy.computeAnalysis(ctx);
    expect(analysis.pipelineBubble, 'bubble = 3/5 = 0.6').toBeCloseTo(0.6, 2);
    expect(analysis.pipelineBubble, 'bubble < 1').toBeLessThan(1);
    expect(analysis.timing.total, 'total > 0').toBeGreaterThan(0);
    expect(analysis.mfu, 'mfu >= 0').toBeGreaterThanOrEqual(0);
  });

  it('createPipelineParallelStrategy with numMicroBatches=1 pp=8 produces valid bubble', () => {
    // bubble = 7/8 = 0.875 (still < 1)
    const strategy = createPipelineParallelStrategy(8, 1);
    const model = getModel('gpt3-125m', 2048)!;
    const cluster = createMultiNodeCluster('h100-sxm', 8, 1)!;
    const ctx = {
      model,
      cluster,
      training: {
        globalBatchSize: 16,
        microBatchSize: 2,
        sequenceLength: 2048,
        maxSteps: 100,
        optimizer: { type: 'adamw' as const, learningRate: 3e-4, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0.01, fusedOptimizer: false },
        lrSchedule: { type: 'cosine' as const, minLR: 3e-5, warmupSteps: 100, warmupRatio: 0.1 },
        dtypes: { params: 'bf16' as const, compute: 'bf16' as const, gradients: 'bf16' as const, activation: 'bf16' as const, optimizer: 'fp32' as const },
        gradientClipping: 1.0,
        gradientAccumulationSteps: 2,
      },
      seqLength: 2048,
      microBatchSize: 2,
      globalBatchSize: 16,
      gradientAccumulationSteps: 2,
      activationCheckpointing: true,
      flashAttention: true,
    };
    const timing = strategy.computeTiming(ctx);
    expect(timing.total, 'total > 0').toBeGreaterThan(0);
    expect(timing.forward, 'forward >= 0').toBeGreaterThanOrEqual(0);
  });
});

// ---------------------------------------------------------------------------
// Section 2: Validation catches numMicroBatches < pp
// ---------------------------------------------------------------------------

describe('Validation: numMicroBatches < pp is an error', () => {
  it('3D parallel strategy with numMicroBatches < pp produces validation error', () => {
    // Bypass factory by constructing directly
    const strategy = new ThreeDParallelStrategy({
      tp: 2, pp: 8, dp: 1, ep: 1,
      numMicroBatches: 2,  // < pp=8
      schedule: '1f1b',
      sequenceParallel: true,
      dpType: 'fsdp',
      activationCheckpointing: true,
    });
    const model = getModel('gpt3-125m', 2048)!;
    const cluster = createMultiNodeCluster('h100-sxm', 8, 2)!;
    const ctx = {
      model,
      cluster,
      training: {
        globalBatchSize: 32,
        microBatchSize: 2,
        sequenceLength: 2048,
        maxSteps: 100,
        optimizer: { type: 'adamw' as const, learningRate: 3e-4, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0.01, fusedOptimizer: false },
        lrSchedule: { type: 'cosine' as const, minLR: 3e-5, warmupSteps: 100, warmupRatio: 0.1 },
        dtypes: { params: 'bf16' as const, compute: 'bf16' as const, gradients: 'bf16' as const, activation: 'bf16' as const, optimizer: 'fp32' as const },
        gradientClipping: 1.0,
        gradientAccumulationSteps: 4,
      },
      seqLength: 2048,
      microBatchSize: 2,
      globalBatchSize: 32,
      gradientAccumulationSteps: 4,
      activationCheckpointing: true,
      flashAttention: true,
    };
    const validation = strategy.validate(ctx);
    expect(validation.errors.some(e => /gradient accumulation steps|pipeline cannot fill/i.test(e))).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Section 3: Timing invariants hold for ALL strategy × GPU combos
// ---------------------------------------------------------------------------

describe('Timing invariants: all strategies produce valid metrics', () => {
  const configs: { label: string; config: SimulationConfig }[] = [
    // Simple DP strategies
    { label: 'ddp/h100/gpt3-125m', config: makeConfig('h100-sxm', 'gpt3-125m', 'ddp', 8, 1) },
    { label: 'fsdp/h100/gpt3-125m', config: makeConfig('h100-sxm', 'gpt3-125m', 'fsdp', 8, 1) },
    { label: 'zero-1/h100/gpt3-125m', config: makeConfig('h100-sxm', 'gpt3-125m', 'zero-1', 8, 1) },
    { label: 'zero-3/h100/gpt3-125m', config: makeConfig('h100-sxm', 'gpt3-125m', 'zero-3', 8, 1) },

    // Hybrid TP strategies
    { label: 'fsdp-tp/h100/llama3-8b', config: makeConfig('h100-sxm', 'llama3-8b', 'fsdp-tp', 8, 1, { tp: 2 }) },
    { label: 'zero1-tp/h100/llama3-8b', config: makeConfig('h100-sxm', 'llama3-8b', 'zero1-tp', 8, 1, { tp: 2 }) },
    { label: 'fsdp-tp+sp/h100/llama3-8b', config: makeConfig('h100-sxm', 'llama3-8b', 'fsdp-tp', 8, 1, { tp: 2, sequenceParallel: true }) },

    // 3D parallel strategies (with PP)
    { label: 'fsdp-tp-pp/h100/llama3-8b', config: makeConfig('h100-sxm', 'llama3-8b', 'fsdp-tp-pp', 16, 2, { tp: 4, pp: 2 }) },
    { label: 'ddp-tp-pp/h100/llama3-8b', config: makeConfig('h100-sxm', 'llama3-8b', 'ddp-tp-pp', 16, 2, { tp: 4, pp: 2 }) },
    { label: 'zero1-tp-pp/h100/llama3-8b', config: makeConfig('h100-sxm', 'llama3-8b', 'zero1-tp-pp', 16, 2, { tp: 4, pp: 2 }) },

    // PCIe-only GPUs (no NVLink)
    { label: 'fsdp-tp/rtx4090/gpt3-125m', config: makeConfig('rtx-4090', 'gpt3-125m', 'fsdp-tp', 8, 1, { tp: 2 }) },
    { label: 'ddp/l40s/gpt3-125m', config: makeConfig('l40s', 'gpt3-125m', 'ddp', 8, 1) },

    // Large-scale configs
    { label: 'fsdp-tp/h100/llama3-70b/64gpu', config: makeConfig('h100-sxm', 'llama3-70b', 'fsdp-tp', 64, 8, { tp: 8 }) },
  ];

  for (const { label, config } of configs) {
    it(`${label}: timing invariants hold`, () => {
      const metrics = getValidatedSimulationMetrics(config);
      assertTimingInvariants(metrics, label);
    });
  }
});

// ---------------------------------------------------------------------------
// Section 4: Large-scale PP configs (the original bug scenario)
// ---------------------------------------------------------------------------

describe('Large-scale pipeline configs: 70B on 2048 H200s', () => {
  it('fsdp-tp-pp tp=8 pp=4 default numMicroBatches: valid metrics', () => {
    // 70B on 2048 H100s: tp=8, pp=4, DP=2048/32=64
    // GBS=8192, MBS=16, GA=ceil(8192/(16*64))=8 >= pp=4
    const config = makeConfig('h200-sxm', 'llama3-70b', 'fsdp-tp-pp', 2048, 256, { tp: 8, pp: 4 }, {
      globalBatchSize: 8192,
      microBatchSize: 16,
      sequenceLength: 2048,
    });
    const metrics = getValidatedSimulationMetrics(config);
    assertTimingInvariants(metrics, '70B/2048xH200/fsdp-tp-pp');
  });

  it('fsdp-tp-pp tp=8 pp=4: numMicroBatches derived from GA', () => {
    // GBS=8192, MBS=16, DP=2048/(8*4)=64, GA=ceil(8192/(16*64))=8
    const config = makeConfig('h200-sxm', 'llama3-70b', 'fsdp-tp-pp', 2048, 256,
      { tp: 8, pp: 4 },
      { globalBatchSize: 8192, microBatchSize: 16, sequenceLength: 2048 },
    );
    const metrics = getValidatedSimulationMetrics(config);
    assertTimingInvariants(metrics, '70B/2048xH200/fsdp-tp-pp/ga-derived');
    // bubble = (pp-1)/(pp-1+m), bounded < 1
    expect(metrics.pipelineBubble, 'bubble < 1').toBeLessThan(1);
  });

  it('fsdp-tp-pp tp=8 pp=4 large GBS: low bubble', () => {
    // GBS=16384, MBS=16, DP=64, GA=ceil(16384/(16*64))=16
    // bubble = 3/(3+16) = 3/19 ~ 15.8%
    const config = makeConfig('h200-sxm', 'llama3-70b', 'fsdp-tp-pp', 2048, 256,
      { tp: 8, pp: 4 },
      { globalBatchSize: 16384, microBatchSize: 16, sequenceLength: 2048 },
    );
    const metrics = getValidatedSimulationMetrics(config);
    assertTimingInvariants(metrics, '70B/2048xH200/fsdp-tp-pp/ga16');
    // bubble = 3/19 ~ 0.158
    expect(metrics.pipelineBubble).toBeCloseTo(3 / 19, 2);
    expect(metrics.mfu, 'MFU should be reasonable').toBeGreaterThan(0.1);
  });

  it('fsdp-tp-pp tp=8 pp=8: valid metrics', () => {
    // 70B on 2048 H100s: tp=8, pp=8, DP=2048/64=32
    // GBS=8192, MBS=4, GA=ceil(8192/(4*32))=64 >= pp=8
    const config = makeConfig('h200-sxm', 'llama3-70b', 'fsdp-tp-pp', 2048, 256, { tp: 8, pp: 8 }, {
      globalBatchSize: 8192,
      microBatchSize: 4,
      sequenceLength: 2048,
    });
    const metrics = getValidatedSimulationMetrics(config);
    assertTimingInvariants(metrics, '70B/2048xH200/fsdp-tp-pp/pp8');
  });
});

// ---------------------------------------------------------------------------
// Section 5: Edge cases
// ---------------------------------------------------------------------------

describe('Pipeline edge cases', () => {
  it('pp=1 (no pipeline): bubble = 0', () => {
    const config = makeConfig('h100-sxm', 'gpt3-125m', 'fsdp-tp', 8, 1, { tp: 2 });
    const metrics = getValidatedSimulationMetrics(config);
    expect(metrics.pipelineBubble).toBe(0);
    assertTimingInvariants(metrics, 'pp=1');
  });

  it('pp=2: bubble = (pp-1)/(pp-1+m)', () => {
    // GBS=8*2=16, MBS=2, DP=8/(2*2)=2 → GA=ceil(16/(2*2))=4
    // bubble = 1/(1+4) = 0.2
    const config = makeConfig('h100-sxm', 'llama3-8b', 'fsdp-tp-pp', 8, 1,
      { tp: 2, pp: 2 });
    const metrics = getValidatedSimulationMetrics(config);
    expect(metrics.pipelineBubble).toBeCloseTo(1 / (1 + 4), 2);
    assertTimingInvariants(metrics, 'pp=2');
  });

  it('pp=4: bubble = (pp-1)/(pp-1+m)', () => {
    // GBS=16*2=32, MBS=2, DP=16/(4*4)=1 → GA=ceil(32/(2*1))=16
    // bubble = 3/(3+16) = 3/19
    const config = makeConfig('h100-sxm', 'llama3-8b', 'fsdp-tp-pp', 16, 2,
      { tp: 4, pp: 4 });
    const metrics = getValidatedSimulationMetrics(config);
    expect(metrics.pipelineBubble).toBeCloseTo(3 / 19, 2);
    assertTimingInvariants(metrics, 'pp=4');
  });

  it('large GA: very small bubble', () => {
    // GBS=1024, MBS=2, DP=16/(4*2)=2 → GA=ceil(1024/(2*2))=256
    // bubble = 1/(1+256) ≈ 0.4%
    const config = makeConfig('h100-sxm', 'llama3-8b', 'fsdp-tp-pp', 16, 2,
      { tp: 4, pp: 2 }, { globalBatchSize: 1024 });
    const metrics = getValidatedSimulationMetrics(config);
    expect(metrics.pipelineBubble).toBeCloseTo(1 / (1 + 256), 3);
    assertTimingInvariants(metrics, 'pp=2/ga=256');
  });

  it('single node 8 GPUs with pp=4 tp=2: valid metrics', () => {
    const config = makeConfig('h100-sxm', 'gpt3-125m', 'fsdp-tp-pp', 8, 1, { tp: 2, pp: 4 });
    const metrics = getValidatedSimulationMetrics(config);
    assertTimingInvariants(metrics, 'single-node/pp=4');
  });
});

// ---------------------------------------------------------------------------
// Section 6: Cross-strategy MFU non-negativity sweep
// ---------------------------------------------------------------------------

describe('MFU non-negativity sweep: all strategy types', () => {
  const strategies: SimulationConfig['strategyType'][] = [
    'ddp', 'fsdp', 'zero-1', 'zero-3',
    'fsdp-tp', 'zero1-tp',
    'ddp-tp-pp', 'zero1-tp-pp', 'fsdp-tp-pp',
  ];
  const gpus = ['h100-sxm', 'a100-80gb', 'rtx-4090', 'l40s'];
  const models: { id: string; minMem: number }[] = [
    { id: 'gpt3-125m', minMem: 0 },
    { id: 'gpt3-1.3b', minMem: 26 },
  ];

  const gpuMemGB: Record<string, number> = {
    'h100-sxm': 80, 'a100-80gb': 80, 'rtx-4090': 24, 'l40s': 48,
  };

  for (const gpuId of gpus) {
    for (const model of models) {
      for (const strat of strategies) {
        // Skip combinations that need more memory than available
        const isPP = strat.includes('pp');
        const isTP = strat.includes('tp');
        const totalGPUs = isPP ? 16 : 8;
        const numNodes = isPP ? 2 : 1;
        const strategyConfig: SimulationConfig['strategyConfig'] = {};
        if (isTP) strategyConfig.tp = 2;
        if (isPP) strategyConfig.pp = 2;

        // Skip models that need more memory than this GPU has
        if (model.minMem > 0 && (gpuMemGB[gpuId] ?? 80) < model.minMem) continue;


        it(`${strat}/${gpuId}/${model.id}: MFU in [0, 1] and timing positive`, () => {
          const config = makeConfig(gpuId, model.id, strat, totalGPUs, numNodes, strategyConfig);
          const metrics = getValidatedSimulationMetrics(config);
          expect(metrics.mfu, `${strat}/${gpuId}/${model.id}: mfu >= 0`).toBeGreaterThanOrEqual(0);
          expect(metrics.mfu, `${strat}/${gpuId}/${model.id}: mfu <= 1`).toBeLessThanOrEqual(1);
          expect(metrics.stepTimeMs, `${strat}/${gpuId}/${model.id}: stepTimeMs > 0`).toBeGreaterThan(0);
          expect(metrics.timing.total, `${strat}/${gpuId}/${model.id}: total > 0`).toBeGreaterThan(0);
        });
      }
    }
  }
});

// ---------------------------------------------------------------------------
// Section 7: Interleaved 1F1B Pipeline Schedule
// ---------------------------------------------------------------------------

describe('Interleaved 1F1B', () => {
  it('bubble = (pp-1)/(pp-1+m*v) for v=2', () => {
    // PP=4, m=8, v=2 → bubble = 3/(3+16) = 3/19
    const strategy = new ThreeDParallelStrategy({
      tp: 2, pp: 4, dp: 2, ep: 1,
      numMicroBatches: 8,
      schedule: 'interleaved-1f1b',
      interleavedStages: 2,
      sequenceParallel: true,
      dpType: 'fsdp',
      activationCheckpointing: true,
    });
    const model = getModel('llama3-8b', 2048)!;
    const cluster = createMultiNodeCluster('h100-sxm', 8, 2)!;
    const ctx = {
      model, cluster,
      training: {
        globalBatchSize: 64, microBatchSize: 2, sequenceLength: 2048, maxSteps: 100,
        optimizer: { type: 'adamw' as const, learningRate: 3e-4, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0.01, fusedOptimizer: false },
        lrSchedule: { type: 'cosine' as const, minLR: 3e-5, warmupSteps: 100, warmupRatio: 0.1 },
        dtypes: { params: 'bf16' as const, compute: 'bf16' as const, gradients: 'bf16' as const, activation: 'bf16' as const, optimizer: 'fp32' as const },
        gradientClipping: 1.0, gradientAccumulationSteps: 8,
      },
      seqLength: 2048, microBatchSize: 2, globalBatchSize: 64, gradientAccumulationSteps: 8,
      activationCheckpointing: true, flashAttention: true,
    };
    const analysis = strategy.computeAnalysis(ctx);
    expect(analysis.pipelineBubble).toBeCloseTo(3 / 19, 3);
  });

  it('bubble = (pp-1)/(pp-1+m*v) for v=3', () => {
    // PP=4, m=8, v=3 → bubble = 3/(3+24) = 3/27
    const strategy = new ThreeDParallelStrategy({
      tp: 2, pp: 4, dp: 2, ep: 1,
      numMicroBatches: 8,
      schedule: 'interleaved-1f1b',
      interleavedStages: 3,
      sequenceParallel: true,
      dpType: 'fsdp',
      activationCheckpointing: true,
    });
    const model = getModel('llama3-8b', 2048)!;
    const cluster = createMultiNodeCluster('h100-sxm', 8, 2)!;
    const ctx = {
      model, cluster,
      training: {
        globalBatchSize: 64, microBatchSize: 2, sequenceLength: 2048, maxSteps: 100,
        optimizer: { type: 'adamw' as const, learningRate: 3e-4, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0.01, fusedOptimizer: false },
        lrSchedule: { type: 'cosine' as const, minLR: 3e-5, warmupSteps: 100, warmupRatio: 0.1 },
        dtypes: { params: 'bf16' as const, compute: 'bf16' as const, gradients: 'bf16' as const, activation: 'bf16' as const, optimizer: 'fp32' as const },
        gradientClipping: 1.0, gradientAccumulationSteps: 8,
      },
      seqLength: 2048, microBatchSize: 2, globalBatchSize: 64, gradientAccumulationSteps: 8,
      activationCheckpointing: true, flashAttention: true,
    };
    const analysis = strategy.computeAnalysis(ctx);
    expect(analysis.pipelineBubble).toBeCloseTo(3 / 27, 3);
  });

  it('bubble = (pp-1)/(pp-1+m*v) for v=4', () => {
    // PP=4, m=8, v=4 → bubble = 3/(3+32) = 3/35
    const strategy = new ThreeDParallelStrategy({
      tp: 2, pp: 4, dp: 2, ep: 1,
      numMicroBatches: 8,
      schedule: 'interleaved-1f1b',
      interleavedStages: 4,
      sequenceParallel: true,
      dpType: 'fsdp',
      activationCheckpointing: true,
    });
    const model = getModel('llama3-8b', 2048)!;
    const cluster = createMultiNodeCluster('h100-sxm', 8, 2)!;
    const ctx = {
      model, cluster,
      training: {
        globalBatchSize: 64, microBatchSize: 2, sequenceLength: 2048, maxSteps: 100,
        optimizer: { type: 'adamw' as const, learningRate: 3e-4, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0.01, fusedOptimizer: false },
        lrSchedule: { type: 'cosine' as const, minLR: 3e-5, warmupSteps: 100, warmupRatio: 0.1 },
        dtypes: { params: 'bf16' as const, compute: 'bf16' as const, gradients: 'bf16' as const, activation: 'bf16' as const, optimizer: 'fp32' as const },
        gradientClipping: 1.0, gradientAccumulationSteps: 8,
      },
      seqLength: 2048, microBatchSize: 2, globalBatchSize: 64, gradientAccumulationSteps: 8,
      activationCheckpointing: true, flashAttention: true,
    };
    const analysis = strategy.computeAnalysis(ctx);
    expect(analysis.pipelineBubble).toBeCloseTo(3 / 35, 3);
  });

  it('interleaved bubble < 1F1B bubble for same pp, m', () => {
    const makeStrat = (schedule: '1f1b' | 'interleaved-1f1b', v: number) =>
      new ThreeDParallelStrategy({
        tp: 2, pp: 4, dp: 2, ep: 1,
        numMicroBatches: 8,
        schedule,
        interleavedStages: v,
        sequenceParallel: true,
        dpType: 'fsdp',
        activationCheckpointing: true,
      });
    const model = getModel('llama3-8b', 2048)!;
    const cluster = createMultiNodeCluster('h100-sxm', 8, 2)!;
    const ctx = {
      model, cluster,
      training: {
        globalBatchSize: 64, microBatchSize: 2, sequenceLength: 2048, maxSteps: 100,
        optimizer: { type: 'adamw' as const, learningRate: 3e-4, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0.01, fusedOptimizer: false },
        lrSchedule: { type: 'cosine' as const, minLR: 3e-5, warmupSteps: 100, warmupRatio: 0.1 },
        dtypes: { params: 'bf16' as const, compute: 'bf16' as const, gradients: 'bf16' as const, activation: 'bf16' as const, optimizer: 'fp32' as const },
        gradientClipping: 1.0, gradientAccumulationSteps: 8,
      },
      seqLength: 2048, microBatchSize: 2, globalBatchSize: 64, gradientAccumulationSteps: 8,
      activationCheckpointing: true, flashAttention: true,
    };

    const standard = makeStrat('1f1b', 1).computeAnalysis(ctx);
    const interleaved = makeStrat('interleaved-1f1b', 2).computeAnalysis(ctx);
    expect(interleaved.pipelineBubble).toBeLessThan(standard.pipelineBubble);
  });

  it('interleaved MFU > 1F1B MFU (same config)', () => {
    const model = getModel('llama3-8b', 2048)!;
    const cluster = createMultiNodeCluster('h100-sxm', 8, 2)!;
    const ctx = {
      model, cluster,
      training: {
        globalBatchSize: 64, microBatchSize: 2, sequenceLength: 2048, maxSteps: 100,
        optimizer: { type: 'adamw' as const, learningRate: 3e-4, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0.01, fusedOptimizer: false },
        lrSchedule: { type: 'cosine' as const, minLR: 3e-5, warmupSteps: 100, warmupRatio: 0.1 },
        dtypes: { params: 'bf16' as const, compute: 'bf16' as const, gradients: 'bf16' as const, activation: 'bf16' as const, optimizer: 'fp32' as const },
        gradientClipping: 1.0, gradientAccumulationSteps: 8,
      },
      seqLength: 2048, microBatchSize: 2, globalBatchSize: 64, gradientAccumulationSteps: 8,
      activationCheckpointing: true, flashAttention: true,
    };

    const standard = new ThreeDParallelStrategy({
      tp: 2, pp: 4, dp: 2, ep: 1, numMicroBatches: 8,
      schedule: '1f1b', interleavedStages: 1,
      sequenceParallel: true, dpType: 'fsdp', activationCheckpointing: true,
    }).computeAnalysis(ctx);

    const interleaved = new ThreeDParallelStrategy({
      tp: 2, pp: 4, dp: 2, ep: 1, numMicroBatches: 8,
      schedule: 'interleaved-1f1b', interleavedStages: 2,
      sequenceParallel: true, dpType: 'fsdp', activationCheckpointing: true,
    }).computeAnalysis(ctx);

    expect(interleaved.mfu).toBeGreaterThan(standard.mfu);
  });

  it('communication volume increases with v (more P2P sends)', () => {
    const model = getModel('llama3-8b', 2048)!;
    const cluster = createMultiNodeCluster('h100-sxm', 8, 2)!;
    const ctx = {
      model, cluster,
      training: {
        globalBatchSize: 64, microBatchSize: 2, sequenceLength: 2048, maxSteps: 100,
        optimizer: { type: 'adamw' as const, learningRate: 3e-4, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0.01, fusedOptimizer: false },
        lrSchedule: { type: 'cosine' as const, minLR: 3e-5, warmupSteps: 100, warmupRatio: 0.1 },
        dtypes: { params: 'bf16' as const, compute: 'bf16' as const, gradients: 'bf16' as const, activation: 'bf16' as const, optimizer: 'fp32' as const },
        gradientClipping: 1.0, gradientAccumulationSteps: 8,
      },
      seqLength: 2048, microBatchSize: 2, globalBatchSize: 64, gradientAccumulationSteps: 8,
      activationCheckpointing: true, flashAttention: true,
    };

    const v1 = new ThreeDParallelStrategy({
      tp: 2, pp: 4, dp: 2, ep: 1, numMicroBatches: 8,
      schedule: '1f1b', interleavedStages: 1,
      sequenceParallel: true, dpType: 'fsdp', activationCheckpointing: true,
    }).computeCommunication(ctx);

    const v2 = new ThreeDParallelStrategy({
      tp: 2, pp: 4, dp: 2, ep: 1, numMicroBatches: 8,
      schedule: 'interleaved-1f1b', interleavedStages: 2,
      sequenceParallel: true, dpType: 'fsdp', activationCheckpointing: true,
    }).computeCommunication(ctx);

    expect(v2.pipelineParallel).toBeGreaterThan(v1.pipelineParallel);
  });

  it('activation memory is same for interleaved and standard 1F1B', () => {
    // Narayanan 2021 §2.3: interleaved schedule has the same peak memory as standard
    const model = getModel('llama3-8b', 2048)!;
    const cluster = createMultiNodeCluster('h100-sxm', 8, 2)!;
    const ctx = {
      model, cluster,
      training: {
        globalBatchSize: 64, microBatchSize: 2, sequenceLength: 2048, maxSteps: 100,
        optimizer: { type: 'adamw' as const, learningRate: 3e-4, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0.01, fusedOptimizer: false },
        lrSchedule: { type: 'cosine' as const, minLR: 3e-5, warmupSteps: 100, warmupRatio: 0.1 },
        dtypes: { params: 'bf16' as const, compute: 'bf16' as const, gradients: 'bf16' as const, activation: 'bf16' as const, optimizer: 'fp32' as const },
        gradientClipping: 1.0, gradientAccumulationSteps: 8,
      },
      seqLength: 2048, microBatchSize: 2, globalBatchSize: 64, gradientAccumulationSteps: 8,
      activationCheckpointing: true, flashAttention: true,
    };

    const v1Mem = new ThreeDParallelStrategy({
      tp: 2, pp: 4, dp: 2, ep: 1, numMicroBatches: 8,
      schedule: '1f1b', interleavedStages: 1,
      sequenceParallel: true, dpType: 'fsdp', activationCheckpointing: true,
    }).computeMemoryPerGPU(ctx);

    const v2Mem = new ThreeDParallelStrategy({
      tp: 2, pp: 4, dp: 2, ep: 1, numMicroBatches: 8,
      schedule: 'interleaved-1f1b', interleavedStages: 2,
      sequenceParallel: true, dpType: 'fsdp', activationCheckpointing: true,
    }).computeMemoryPerGPU(ctx);

    expect(v2Mem.activations).toBe(v1Mem.activations);
  });

  it('interleavedStages=1 + interleaved schedule ≈ same as 1F1B (degenerate case)', () => {
    const model = getModel('llama3-8b', 2048)!;
    const cluster = createMultiNodeCluster('h100-sxm', 8, 2)!;
    const ctx = {
      model, cluster,
      training: {
        globalBatchSize: 64, microBatchSize: 2, sequenceLength: 2048, maxSteps: 100,
        optimizer: { type: 'adamw' as const, learningRate: 3e-4, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0.01, fusedOptimizer: false },
        lrSchedule: { type: 'cosine' as const, minLR: 3e-5, warmupSteps: 100, warmupRatio: 0.1 },
        dtypes: { params: 'bf16' as const, compute: 'bf16' as const, gradients: 'bf16' as const, activation: 'bf16' as const, optimizer: 'fp32' as const },
        gradientClipping: 1.0, gradientAccumulationSteps: 8,
      },
      seqLength: 2048, microBatchSize: 2, globalBatchSize: 64, gradientAccumulationSteps: 8,
      activationCheckpointing: true, flashAttention: true,
    };

    const standard = new ThreeDParallelStrategy({
      tp: 2, pp: 4, dp: 2, ep: 1, numMicroBatches: 8,
      schedule: '1f1b', interleavedStages: 1,
      sequenceParallel: true, dpType: 'fsdp', activationCheckpointing: true,
    }).computeAnalysis(ctx);

    const degenerate = new ThreeDParallelStrategy({
      tp: 2, pp: 4, dp: 2, ep: 1, numMicroBatches: 8,
      schedule: 'interleaved-1f1b', interleavedStages: 1,
      sequenceParallel: true, dpType: 'fsdp', activationCheckpointing: true,
    }).computeAnalysis(ctx);

    // With v=1, interleaved formula reduces to standard: (pp-1)/(pp-1+m*1) = (pp-1)/(pp-1+m)
    expect(degenerate.pipelineBubble).toBeCloseTo(standard.pipelineBubble, 4);
  });

  it('end-to-end: interleaved-1f1b via getSimulationMetrics', () => {
    // Use engine path to verify full integration
    const config = makeConfig('h100-sxm', 'llama3-8b', 'fsdp-tp-pp', 16, 2,
      { tp: 2, pp: 4, pipelineSchedule: 'interleaved-1f1b', interleavedStages: 2 });
    const metrics = getValidatedSimulationMetrics(config);
    assertTimingInvariants(metrics, 'interleaved-1f1b/e2e');
    // Interleaved bubble should be less than 1F1B for same config
    const config1f1b = makeConfig('h100-sxm', 'llama3-8b', 'fsdp-tp-pp', 16, 2,
      { tp: 2, pp: 4 });
    const metrics1f1b = getValidatedSimulationMetrics(config1f1b);
    expect(metrics.pipelineBubble).toBeLessThan(metrics1f1b.pipelineBubble);
    expect(metrics.mfu).toBeGreaterThan(0);
    expect(metrics.mfu).toBeLessThanOrEqual(1);
  });
});
