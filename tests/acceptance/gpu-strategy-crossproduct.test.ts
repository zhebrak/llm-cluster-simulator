/**
 * GPU × Strategy Cross-Product Tests
 *
 * Ensures every GPU + dtype combination resolves to positive TFLOPS,
 * every strategy produces finite step times and valid MFU for all GPU families,
 * and edge cases (V100 bf16, fp8 fallback chains) are handled correctly.
 */

import { describe, it, expect } from 'vitest';
import {
  ALL_GPUS,
  getEffectiveTFLOPS,
  V100_32GB,
  A100_80GB,
} from '../../src/core/hardware/gpu.ts';
import { createMultiNodeCluster } from '../../src/core/hardware/topology.ts';
import { getSimulationMetrics, type SimulationConfig } from '../../src/core/simulation/engine.ts';
import { getValidatedSimulationMetrics } from '../helpers/validated-metrics.ts';

// ---------------------------------------------------------------------------
// Section 1: TFLOPS Resolution — every GPU × every dtype
// ---------------------------------------------------------------------------

const ALL_DTYPES = ['fp32', 'tf32', 'fp16', 'bf16', 'fp8', 'fp4', 'int8', 'int4'] as const;

describe('TFLOPS resolution: every GPU × every dtype > 0', () => {
  for (const [gpuId, gpu] of Object.entries(ALL_GPUS)) {
    for (const dtype of ALL_DTYPES) {
      it(`${gpuId} + ${dtype} → positive TFLOPS`, () => {
        const tflops = getEffectiveTFLOPS(gpu, dtype);
        expect(tflops).toBeGreaterThan(0);
      });
    }
  }
});

describe('V100 specific fallback values', () => {
  it('V100 + bf16 falls back to fp16 (125 TFLOPS)', () => {
    expect(getEffectiveTFLOPS(V100_32GB, 'bf16')).toBe(125);
  });

  it('V100 + fp8 falls back through bf16(0) → fp16 (125 TFLOPS)', () => {
    expect(getEffectiveTFLOPS(V100_32GB, 'fp8')).toBe(125);
  });

  it('V100 + fp4 falls back through fp8(0) → bf16(0) → fp16 (125 TFLOPS)', () => {
    expect(getEffectiveTFLOPS(V100_32GB, 'fp4')).toBe(125);
  });

  it('V100 + tf32 falls back to fp32 (15.7 TFLOPS)', () => {
    expect(getEffectiveTFLOPS(V100_32GB, 'tf32')).toBe(15.7);
  });

  it('V100 + int8 falls back through int8(0) → bf16(0) → fp16 (125 TFLOPS)', () => {
    expect(getEffectiveTFLOPS(V100_32GB, 'int8')).toBe(125);
  });

  it('V100 + int4 falls back through int4(0) → int8(0) → bf16(0) → fp16 (125 TFLOPS)', () => {
    expect(getEffectiveTFLOPS(V100_32GB, 'int4')).toBe(125);
  });

  it('V100 + fp32 returns native fp32 (15.7 TFLOPS)', () => {
    expect(getEffectiveTFLOPS(V100_32GB, 'fp32')).toBe(15.7);
  });

  it('V100 + fp16 returns native fp16 (125 TFLOPS)', () => {
    expect(getEffectiveTFLOPS(V100_32GB, 'fp16')).toBe(125);
  });
});

describe('A100 fp8 fallback', () => {
  it('A100 + fp8 falls back to bf16 (312 TFLOPS)', () => {
    expect(getEffectiveTFLOPS(A100_80GB, 'fp8')).toBe(312);
  });
});

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

// Helper: builds a SimulationConfig for the given gpu + strategy combo
function makeConfig(
  gpuId: string,
  modelId: string,
  strategyType: SimulationConfig['strategyType'],
  totalGPUs: number,
  numNodes: number,
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
  };
}

// Strategies grouped by topology requirements
const SIMPLE_DP_STRATEGIES: SimulationConfig['strategyType'][] = ['ddp', 'fsdp', 'zero-1', 'zero-3'];
const HYBRID_TP_STRATEGIES: SimulationConfig['strategyType'][] = ['fsdp-tp', 'zero1-tp'];
const THREE_D_STRATEGIES: SimulationConfig['strategyType'][] = ['ddp-tp-pp', 'zero1-tp-pp', 'fsdp-tp-pp'];
const ALL_STRATEGIES: SimulationConfig['strategyType'][] = [
  ...SIMPLE_DP_STRATEGIES, ...HYBRID_TP_STRATEGIES, ...THREE_D_STRATEGIES,
];

// ---------------------------------------------------------------------------
// Section 2: Comprehensive crash detection (aggregated)
// Replaces ~1,700 individual smoke tests with a single test that reports failures.
// ---------------------------------------------------------------------------

// Models grouped by parameter count tier, each paired with representative GPUs
const MODEL_GPU_TIERS: { tier: string; pairs: { model: string; gpus: string[] }[] }[] = [
  {
    tier: 'tiny (<2B)',
    pairs: [
      { model: 'gpt3-125m',    gpus: ['t4', 'v100-32gb', 'h100-sxm', 'rtx-3090', 'rtx-4090', 'b200', 'mi300x'] },
      { model: 'llama3.2-1b',  gpus: ['t4', 'v100-32gb', 'a100-80gb', 'h100-sxm', 'rtx-4090'] },
    ],
  },
  {
    tier: 'small (2–5B)',
    pairs: [
      { model: 'gpt3-1.3b',     gpus: ['t4', 'v100-32gb', 'a100-80gb', 'h100-sxm'] },
      { model: 'llama3.2-3b',   gpus: ['t4', 'v100-32gb', 'a100-80gb', 'h100-sxm', 'l4'] },
      { model: 'gemma2-2b',     gpus: ['t4', 'v100-32gb', 'h100-sxm', 'rtx-4090'] },
      { model: 'phi3-mini',     gpus: ['t4', 'v100-32gb', 'a100-80gb', 'h100-sxm', 'rtx-4090'] },
    ],
  },
  {
    tier: 'medium (5–15B)',
    pairs: [
      { model: 'gpt3-6.7b',        gpus: ['a100-80gb', 'h100-sxm', 'b200'] },
      { model: 'llama2-7b',         gpus: ['a100-80gb', 'h100-sxm', 'mi300x', 'l40s'] },
      { model: 'mistral-7b',        gpus: ['a100-80gb', 'h100-sxm', 'b200', 'l40s'] },
      { model: 'qwen2.5-7b',        gpus: ['a100-80gb', 'h100-sxm', 'mi300x', 'l40s'] },
      { model: 'phi3-small',        gpus: ['a100-80gb', 'h100-sxm', 'b200'] },
      { model: 'olmo2-7b',          gpus: ['a100-80gb', 'h100-sxm', 'mi300x', 'l40s'] },
      { model: 'yi-6b',             gpus: ['a100-80gb', 'h100-sxm', 'l40s', 'rtx-4090'] },
      { model: 'gemma2-9b',         gpus: ['a100-80gb', 'h100-sxm', 'l40s'] },
      { model: 'mistral-nemo-12b',  gpus: ['a100-80gb', 'h100-sxm', 'b200', 'mi300x'] },
      { model: 'llama2-13b',        gpus: ['a100-80gb', 'h100-sxm', 'b200', 'mi300x'] },
      { model: 'olmo2-13b',         gpus: ['a100-80gb', 'h100-sxm', 'b200'] },
      { model: 'gpt3-13b',          gpus: ['a100-80gb', 'h100-sxm', 'b200'] },
      { model: 'phi3-medium',       gpus: ['a100-80gb', 'h100-sxm', 'b200'] },
      { model: 'nemotron-4-15b',      gpus: ['a100-80gb', 'h100-sxm', 'b200'] },
      { model: 'deepseek-moe-16b',  gpus: ['a100-80gb', 'h100-sxm', 'mi300x'] },
    ],
  },
  {
    tier: 'large (15–70B)',
    pairs: [
      { model: 'codestral-22b',     gpus: ['a100-80gb', 'h100-sxm', 'b200', 'mi300x'] },
      { model: 'mistral-small-24b', gpus: ['a100-80gb', 'h100-sxm', 'b200', 'mi300x'] },
      { model: 'gemma2-27b',        gpus: ['a100-80gb', 'h100-sxm', 'b200'] },
      { model: 'yi-34b',            gpus: ['a100-80gb', 'h100-sxm', 'b200'] },
      { model: 'command-r',         gpus: ['a100-80gb', 'h100-sxm', 'b200', 'mi300x'] },
      { model: 'mixtral-8x7b',      gpus: ['a100-80gb', 'h100-sxm', 'b200'] },
      { model: 'llama2-70b',        gpus: ['h100-sxm', 'b200', 'mi300x'] },
      { model: 'llama3-70b',        gpus: ['h100-sxm', 'b200', 'mi300x'] },
      { model: 'qwen2.5-72b',       gpus: ['h100-sxm', 'b200', 'mi300x'] },
    ],
  },
  {
    tier: 'xl (70B+)',
    pairs: [
      { model: 'command-r-plus',    gpus: ['h100-sxm', 'b200', 'mi300x'] },
      { model: 'mistral-large-123b', gpus: ['h100-sxm', 'b200', 'mi300x'] },
      { model: 'dbrx',              gpus: ['h100-sxm', 'b200', 'mi300x'] },
      { model: 'mixtral-8x22b',     gpus: ['h100-sxm', 'b200', 'mi300x'] },
      { model: 'gpt3-175b',         gpus: ['h100-sxm', 'b200'] },
      { model: 'deepseek-v2',       gpus: ['h100-sxm', 'b200'] },
      { model: 'nemotron-4-340b',     gpus: ['h100-sxm', 'b200'] },
      { model: 'llama3-405b',       gpus: ['h100-sxm', 'b200'] },
      { model: 'deepseek-v3',       gpus: ['h100-sxm', 'b200'] },
    ],
  },
];

describe('Comprehensive crash detection (slow)', () => {
  it('all model × GPU × strategy combos produce finite results', () => {
    const failures: string[] = [];
    for (const { pairs } of MODEL_GPU_TIERS) {
      for (const { model, gpus } of pairs) {
        for (const gpuId of gpus) {
          for (const strat of ALL_STRATEGIES) {
            try {
              const cfg = strat.includes('tp-pp')
                ? makeConfig(gpuId, model, strat, 16, 2, { tp: 2, pp: 2 })
                : strat.includes('tp')
                ? makeConfig(gpuId, model, strat, 8, 1, { tp: 2 })
                : makeConfig(gpuId, model, strat, 8, 1);
              const m = getSimulationMetrics(cfg);
              if (!isFinite(m.stepTimeMs) || m.stepTimeMs <= 0 || m.mfu <= 0 || m.mfu > 1) {
                failures.push(`${model}/${gpuId}/${strat}: step=${m.stepTimeMs} mfu=${m.mfu}`);
              }
            } catch (e) {
              failures.push(`${model}/${gpuId}/${strat}: THREW ${e}`);
            }
          }
        }
      }
    }
    expect(failures, `${failures.length} combos failed:\n${failures.join('\n')}`).toEqual([]);
  }, 120_000);
});

// ---------------------------------------------------------------------------
// Section 3: Edge cases
// ---------------------------------------------------------------------------

function assertFiniteAndValid(metrics: { stepTimeMs: number; mfu: number }, label: string) {
  expect(metrics.stepTimeMs, `${label}: stepTimeMs`).toBeGreaterThan(0);
  expect(metrics.stepTimeMs, `${label}: stepTimeMs finite`).toBeLessThan(Infinity);
  expect(metrics.mfu, `${label}: mfu > 0`).toBeGreaterThan(0);
  expect(metrics.mfu, `${label}: mfu ≤ 1`).toBeLessThanOrEqual(1);
}

describe('Edge cases', () => {
  // The original bug: V100 + bf16 default dtype
  it('V100 + bf16 dtype (the original bug): finite step time and positive MFU', () => {
    const config: SimulationConfig = {
      clusterConfig: createMultiNodeCluster('v100-32gb', 8, 1)!,
      modelId: 'gpt3-125m',
      globalBatchSize: 16,
      microBatchSize: 2,
      sequenceLength: 2048,
      strategyType: 'ddp',
    };
    const metrics = getValidatedSimulationMetrics(config);
    assertFiniteAndValid(metrics, 'V100/bf16/ddp');
  });

  // V100 + fp8 dtype (double fallback: fp8→bf16→fp16)
  it('V100 + fp8 dtype (double fallback)', () => {
    const config: SimulationConfig = {
      clusterConfig: createMultiNodeCluster('v100-32gb', 8, 1)!,
      modelId: 'gpt3-125m',
      globalBatchSize: 16,
      microBatchSize: 2,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      mixedPrecision: 'fp8',
    };
    const metrics = getValidatedSimulationMetrics(config);
    assertFiniteAndValid(metrics, 'V100/fp8/fsdp');
  });

  // V100 with all 10 strategies
  describe('V100 with all 10 strategies', () => {
    for (const strat of SIMPLE_DP_STRATEGIES) {
      it(`V100 + ${strat}`, () => {
        const config = makeConfig('v100-32gb', 'gpt3-125m', strat, 8, 1);
        const metrics = getValidatedSimulationMetrics(config);
        assertFiniteAndValid(metrics, `V100/${strat}`);
      });
    }

    for (const strat of HYBRID_TP_STRATEGIES) {
      it(`V100 + ${strat}`, () => {
        const config = makeConfig('v100-32gb', 'gpt3-125m', strat, 8, 1, { tp: 2 });
        const metrics = getValidatedSimulationMetrics(config);
        assertFiniteAndValid(metrics, `V100/${strat}`);
      });
    }

    for (const strat of THREE_D_STRATEGIES) {
      it(`V100 + ${strat}`, () => {
        const config = makeConfig('v100-32gb', 'gpt3-125m', strat, 16, 2, { tp: 2, pp: 2 });
        const metrics = getValidatedSimulationMetrics(config);
        assertFiniteAndValid(metrics, `V100/${strat}`);
      });
    }
  });

  // A100 + fp8 dtype (fp8=0, falls back to bf16=312)
  it('A100 + fp8 dtype (fp8 fallback to bf16)', () => {
    const config: SimulationConfig = {
      clusterConfig: createMultiNodeCluster('a100-80gb', 8, 1)!,
      modelId: 'llama3-8b',
      globalBatchSize: 16,
      microBatchSize: 2,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      mixedPrecision: 'fp8',
    };
    const metrics = getValidatedSimulationMetrics(config);
    assertFiniteAndValid(metrics, 'A100/fp8/fsdp');
  });

  // H100-PCIe (lower intra-node bandwidth, no NVLink between GPUs in most configs)
  it('H100-PCIe with fsdp-tp (reduced NVLink BW)', () => {
    const config = makeConfig('h100-pcie', 'llama3-8b', 'fsdp-tp', 8, 1, { tp: 2 });
    const metrics = getValidatedSimulationMetrics(config);
    assertFiniteAndValid(metrics, 'H100-PCIe/fsdp-tp');
  });
});

// ---------------------------------------------------------------------------
// Section 4: Real validation — memory, ordering, and sanity invariants
// These tests catch actual bugs, not just "does it crash"
// ---------------------------------------------------------------------------

function getMetrics(
  gpuId: string, modelId: string,
  strategy: SimulationConfig['strategyType'],
  totalGPUs: number, numNodes: number,
  strategyConfig?: SimulationConfig['strategyConfig'],
) {
  return getValidatedSimulationMetrics(makeConfig(gpuId, modelId, strategy, totalGPUs, numNodes, strategyConfig));
}

/** Raw metrics without validation -- use for intentional OOM configs */
function getRawMetrics(
  gpuId: string, modelId: string,
  strategy: SimulationConfig['strategyType'],
  totalGPUs: number, numNodes: number,
  strategyConfig?: SimulationConfig['strategyConfig'],
) {
  return getSimulationMetrics(makeConfig(gpuId, modelId, strategy, totalGPUs, numNodes, strategyConfig));
}

describe('Memory fits on GPU (FSDP configs that should actually work)', () => {
  // These are realistic configs where memory should genuinely fit
  const feasibleConfigs: { label: string; gpuId: string; modelId: string; strategy: SimulationConfig['strategyType']; gpus: number; nodes: number; cfg?: SimulationConfig['strategyConfig'] }[] = [
    { label: 'GPT-3 125M DDP 8x V100',        gpuId: 'v100-32gb', modelId: 'gpt3-125m',    strategy: 'ddp',     gpus: 8,  nodes: 1 },
    { label: 'GPT-3 125M FSDP 8x V100',       gpuId: 'v100-32gb', modelId: 'gpt3-125m',    strategy: 'fsdp',    gpus: 8,  nodes: 1 },
    { label: 'GPT-3 125M DDP 8x H100',        gpuId: 'h100-sxm',  modelId: 'gpt3-125m',    strategy: 'ddp',     gpus: 8,  nodes: 1 },
    { label: 'LLaMA3 8B FSDP 8x A100-80',     gpuId: 'a100-80gb', modelId: 'llama3-8b',    strategy: 'fsdp',    gpus: 8,  nodes: 1 },
    { label: 'LLaMA3 8B FSDP 8x H100',        gpuId: 'h100-sxm',  modelId: 'llama3-8b',    strategy: 'fsdp',    gpus: 8,  nodes: 1 },
    { label: 'LLaMA2 7B FSDP 8x V100',        gpuId: 'v100-32gb', modelId: 'llama2-7b',    strategy: 'fsdp',    gpus: 8,  nodes: 1 },
    { label: 'Mistral Nemo 12B FSDP 8x H100',  gpuId: 'h100-sxm', modelId: 'mistral-nemo-12b', strategy: 'fsdp', gpus: 8, nodes: 1 },
    { label: 'Mistral 7B FSDP 8x A100',       gpuId: 'a100-80gb', modelId: 'mistral-7b',   strategy: 'fsdp',    gpus: 8,  nodes: 1 },
  ];

  for (const c of feasibleConfigs) {
    it(`${c.label}: memory ≤ GPU capacity`, () => {
      const metrics = getMetrics(c.gpuId, c.modelId, c.strategy, c.gpus, c.nodes, c.cfg);
      const gpuMemBytes = ALL_GPUS[c.gpuId].memoryGB * 1e9;
      expect(
        metrics.memoryPerGPU.total,
        `${c.label}: ${(metrics.memoryPerGPU.total / 1e9).toFixed(1)}GB > ${ALL_GPUS[c.gpuId].memoryGB}GB GPU`,
      ).toBeLessThanOrEqual(gpuMemBytes);
    });
  }
});

describe('Memory overflow detected for impossible configs', () => {
  // These configs can NOT fit — memory should exceed GPU capacity
  const impossibleConfigs: { label: string; gpuId: string; modelId: string; strategy: SimulationConfig['strategyType']; gpus: number; nodes: number }[] = [
    { label: 'LLaMA3 8B DDP 8x V100-32',        gpuId: 'v100-32gb', modelId: 'llama3-8b',        strategy: 'ddp',  gpus: 8,  nodes: 1 },
    { label: 'LLaMA3 405B DDP 8x RTX-4090',     gpuId: 'rtx-4090',  modelId: 'llama3-405b',      strategy: 'ddp',  gpus: 8,  nodes: 1 },
    { label: 'LLaMA3 405B FSDP 8x RTX-4090',    gpuId: 'rtx-4090',  modelId: 'llama3-405b',      strategy: 'fsdp', gpus: 8,  nodes: 1 },
    { label: 'DeepSeek V3 DDP 8x L4',           gpuId: 'l4',        modelId: 'deepseek-v3',      strategy: 'ddp',  gpus: 8,  nodes: 1 },
    { label: 'Mistral 123B FSDP 8x V100',       gpuId: 'v100-32gb', modelId: 'mistral-large-123b', strategy: 'fsdp', gpus: 8, nodes: 1 },
  ];

  for (const c of impossibleConfigs) {
    it(`${c.label}: memory exceeds GPU capacity`, () => {
      const metrics = getRawMetrics(c.gpuId, c.modelId, c.strategy, c.gpus, c.nodes);
      const gpuMemBytes = ALL_GPUS[c.gpuId].memoryGB * 1e9;
      expect(
        metrics.memoryPerGPU.total,
        `${c.label}: expected OOM but got ${(metrics.memoryPerGPU.total / 1e9).toFixed(1)}GB ≤ ${ALL_GPUS[c.gpuId].memoryGB}GB`,
      ).toBeGreaterThan(gpuMemBytes);
    });
  }
});

describe('FSDP uses less memory per GPU than DDP', () => {
  const models = ['gpt3-125m', 'llama3-8b', 'mistral-7b', 'mistral-nemo-12b'];
  for (const modelId of models) {
    it(`${modelId}: FSDP memoryPerGPU < DDP memoryPerGPU on H100`, () => {
      const ddp  = getRawMetrics('h100-sxm', modelId, 'ddp',  8, 1);
      const fsdp = getRawMetrics('h100-sxm', modelId, 'fsdp', 8, 1);
      expect(
        fsdp.memoryPerGPU.total,
        `${modelId}: FSDP ${(fsdp.memoryPerGPU.total/1e9).toFixed(1)}GB should be < DDP ${(ddp.memoryPerGPU.total/1e9).toFixed(1)}GB`,
      ).toBeLessThan(ddp.memoryPerGPU.total);
    });
  }
});

describe('Faster GPU → lower step time (same model, same strategy)', () => {
  // V100 < A100 < H100 < B200 in compute power → step time should decrease
  it('LLaMA2 7B FSDP: V100 > A100 > H100 > B200 step time', () => {
    const v100 = getMetrics('v100-32gb', 'llama2-7b', 'fsdp', 8, 1);
    const a100 = getMetrics('a100-80gb', 'llama2-7b', 'fsdp', 8, 1);
    const h100 = getMetrics('h100-sxm',  'llama2-7b', 'fsdp', 8, 1);
    const b200 = getMetrics('b200',       'llama2-7b', 'fsdp', 8, 1);

    expect(v100.stepTimeMs).toBeGreaterThan(a100.stepTimeMs);
    expect(a100.stepTimeMs).toBeGreaterThan(h100.stepTimeMs);
    expect(h100.stepTimeMs).toBeGreaterThan(b200.stepTimeMs);
  });

  it('GPT-3 125M DDP: V100 > H100 > B200 step time', () => {
    const v100 = getMetrics('v100-32gb', 'gpt3-125m', 'ddp', 8, 1);
    const h100 = getMetrics('h100-sxm',  'gpt3-125m', 'ddp', 8, 1);
    const b200 = getMetrics('b200',       'gpt3-125m', 'ddp', 8, 1);

    expect(v100.stepTimeMs).toBeGreaterThan(h100.stepTimeMs);
    expect(h100.stepTimeMs).toBeGreaterThan(b200.stepTimeMs);
  });

  it('Mistral 7B FSDP: A100 > H100 > B200 step time', () => {
    const a100 = getMetrics('a100-80gb', 'mistral-7b', 'fsdp', 8, 1);
    const h100 = getMetrics('h100-sxm',  'mistral-7b', 'fsdp', 8, 1);
    const b200 = getMetrics('b200',       'mistral-7b', 'fsdp', 8, 1);

    expect(a100.stepTimeMs).toBeGreaterThan(h100.stepTimeMs);
    expect(h100.stepTimeMs).toBeGreaterThan(b200.stepTimeMs);
  });
});

describe('Bigger model → longer step time (same GPU, same strategy)', () => {
  it('H100 FSDP: GPT-3 125M < LLaMA2-7B < Mistral-Nemo-12B < Codestral-22B', () => {
    const gpt3  = getMetrics('h100-sxm', 'gpt3-125m',         'fsdp', 8, 1);
    const l7b   = getMetrics('h100-sxm', 'llama2-7b',        'fsdp', 8, 1);
    const m12b  = getMetrics('h100-sxm', 'mistral-nemo-12b', 'fsdp', 8, 1);
    const c22b  = getMetrics('h100-sxm', 'codestral-22b',    'fsdp', 8, 1);

    expect(gpt3.stepTimeMs).toBeLessThan(l7b.stepTimeMs);
    expect(l7b.stepTimeMs).toBeLessThan(m12b.stepTimeMs);
    expect(m12b.stepTimeMs).toBeLessThan(c22b.stepTimeMs);
  });

  it('A100 FSDP: GPT-3 125M < Mistral-7B < Codestral-22B', () => {
    const gpt3 = getMetrics('a100-80gb', 'gpt3-125m',    'fsdp', 8, 1);
    const m7b  = getMetrics('a100-80gb', 'mistral-7b',    'fsdp', 8, 1);
    const c22b = getMetrics('a100-80gb', 'codestral-22b', 'fsdp', 8, 1);

    expect(gpt3.stepTimeMs).toBeLessThan(m7b.stepTimeMs);
    expect(m7b.stepTimeMs).toBeLessThan(c22b.stepTimeMs);
  });
});

describe('MFU sanity bounds', () => {
  // Real-world MFU is typically 30-55% for well-optimized training
  // Above 65% is suspicious; above 80% almost certainly a bug
  const sanityConfigs = [
    { gpuId: 'h100-sxm', modelId: 'llama3-8b',         strategy: 'fsdp' as const },
    { gpuId: 'h100-sxm', modelId: 'llama2-7b',          strategy: 'fsdp' as const },
    { gpuId: 'h100-sxm', modelId: 'mistral-nemo-12b',   strategy: 'fsdp' as const },
    { gpuId: 'h100-sxm', modelId: 'mistral-small-24b',  strategy: 'fsdp' as const },
    { gpuId: 'a100-80gb', modelId: 'llama3-8b',         strategy: 'fsdp' as const },
    { gpuId: 'b200',      modelId: 'llama3-8b',         strategy: 'fsdp' as const },
  ];

  for (const c of sanityConfigs) {
    it(`${c.modelId} on ${c.gpuId} ${c.strategy}: MFU ∈ [5%, 65%]`, () => {
      const metrics = getMetrics(c.gpuId, c.modelId, c.strategy, 8, 1);
      expect(metrics.mfu, `MFU too low: ${(metrics.mfu * 100).toFixed(1)}%`).toBeGreaterThanOrEqual(0.05);
      expect(metrics.mfu, `MFU too high: ${(metrics.mfu * 100).toFixed(1)}%`).toBeLessThanOrEqual(0.65);
    });
  }
});
