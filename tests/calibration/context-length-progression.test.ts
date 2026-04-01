/**
 * Context Length Progression Tests
 *
 * Validates that the simulator's training physics are correct across the full
 * 4K–200K context range. Checks scaling laws (quadratic attention, linear
 * memory with FA, CP memory reduction), re-validates all anchors/presets,
 * and tests the published 131K benchmark.
 */

import { describe, it, expect } from 'vitest';
import {
  type SimulationConfig,
  type SimulationMetrics,
  getSimulationMetrics,
} from '../../src/core/simulation/engine.ts';
import { getValidatedSimulationMetrics } from '../helpers/validated-metrics.ts';
import { getModel } from '../../src/core/models/index.ts';
import { PUBLISHED, toSimConfig } from '../../src/data/published-training-configs.ts';
import { createMultiNodeCluster, createSingleNodeCluster, getPresetCluster } from '../../src/core/hardware/index.ts';
import type { ClusterConfig } from '../../src/types/index.ts';

// ─── Helpers ──────────────────────────────────────────────────────────

function sim(config: SimulationConfig): SimulationMetrics {
  return getValidatedSimulationMetrics(config);
}

function rawSim(config: SimulationConfig): SimulationMetrics {
  return getSimulationMetrics(config);
}

const TOLERANCE = 0.025; // ±2.5%

function expectClose(actual: number, expected: number, label: string) {
  const rel = Math.abs(actual - expected) / expected;
  expect(
    rel,
    `${label}: expected ~${expected.toPrecision(4)}, got ${actual.toPrecision(4)} (${(rel * 100).toFixed(2)}% off, limit ${TOLERANCE * 100}%)`
  ).toBeLessThanOrEqual(TOLERANCE);
}

function makeCluster(gpuId: string, gpusPerNode: number, numNodes: number): ClusterConfig {
  const cluster = numNodes === 1
    ? createSingleNodeCluster(gpuId, gpusPerNode)
    : createMultiNodeCluster(gpuId, gpusPerNode, numNodes);
  if (!cluster) throw new Error(`Failed to create cluster: ${gpuId} ${gpusPerNode}×${numNodes}`);
  return cluster;
}

// =========================================================================
// §1 — Anchor re-validation
//
// Re-run all 10 Tier-1 anchors + the 131K config using the same ±3pp bounds
// from anchor-benchmarks.test.ts.
// =========================================================================

describe('§1 — Anchor re-validation', () => {
  const anchorBounds: {
    key: keyof typeof PUBLISHED;
    label: string;
    lo: number;
    hi: number;
    useModelFlopsMfu?: boolean;
    useRawSim?: boolean;
  }[] = [
    { key: 'llama3_405b_8k',      label: 'LLaMA 3.1 405B 8K',    lo: 0.37,  hi: 0.43,  useRawSim: true },
    { key: 'gpt3_175b',           label: 'GPT-3 175B',            lo: 0.412, hi: 0.472 },
    { key: 'ibm_llama2_7b',       label: 'IBM FSDP LLaMA 7B',    lo: 0.54,  hi: 0.60 },
    { key: 'deepseek_v3_fp8_h800', label: 'DeepSeek V3 FP8',     lo: 0.407, hi: 0.467 },
    { key: 'nemotron_4_340b',       label: 'Nemotron-4 340B',          lo: 0.39,  hi: 0.45 },
    { key: 'olmo2_32b',           label: 'OLMo 2 32B',            lo: 0.35,  hi: 0.41 },
    { key: 'olmo3_32b',           label: 'OLMo 3 32B',            lo: 0.38,  hi: 0.48 },
    { key: 'qwen2_57b_a14b_ep4',  label: 'Qwen2 57B-A14B MoE',   lo: 0.35, hi: 0.50 },
    { key: 'bloom_176b',          label: 'BLOOM 176B',             lo: 0.45,  hi: 0.51, useRawSim: true },
    { key: 'mt_nlg_530b',         label: 'MT-NLG 530B',           lo: 0.37,  hi: 0.45 },
    { key: 'nemotron_4_15b_dgxc',   label: 'Nemotron-4 15B DGXC',   lo: 0.53,  hi: 0.59 },
  ];

  for (const { key, label, lo, hi, useModelFlopsMfu, useRawSim } of anchorBounds) {
    it(`${label}: MFU in [${(lo * 100).toFixed(1)}%, ${(hi * 100).toFixed(1)}%]`, () => {
      const config = toSimConfig(PUBLISHED[key]);
      const m = useRawSim ? rawSim(config) : sim(config);
      const mfu = useModelFlopsMfu ? (m.modelFlopsMfu ?? m.mfu) : m.mfu;
      expect(mfu, `MFU ${(mfu * 100).toFixed(1)}% outside [${lo * 100}%, ${hi * 100}%]`)
        .toBeGreaterThanOrEqual(lo);
      expect(mfu, `MFU ${(mfu * 100).toFixed(1)}% outside [${lo * 100}%, ${hi * 100}%]`)
        .toBeLessThanOrEqual(hi);
    });
  }

  it('LLaMA 3.1 405B 131K: modelFlopsMfu in [0.35, 0.41]', () => {
    const config = toSimConfig(PUBLISHED.llama3_405b_131k);
    const m = rawSim(config);
    const mfu = m.modelFlopsMfu ?? m.mfu;
    expect(mfu, `Model FLOPs MFU ${(mfu * 100).toFixed(1)}% outside [35%, 41%]`)
      .toBeGreaterThanOrEqual(0.35);
    expect(mfu, `Model FLOPs MFU ${(mfu * 100).toFixed(1)}% outside [35%, 41%]`)
      .toBeLessThanOrEqual(0.41);
  });
});

// =========================================================================
// §2 — Preset re-validation
//
// Re-run all 8 training demo presets with ±2.5% pinned values.
// =========================================================================

describe('§2 — Preset re-validation', () => {
  const presets: {
    name: string;
    clusterId?: string;
    gpuId?: string; numGPUs?: number; gpusPerNode?: number;
    config: SimulationConfig;
    pinned: { mfu: number; hfu: number; memUtil: number; tokPerSec: number; stepTimeMs: number };
  }[] = [
    {
      name: 'LLaMA 3 405B (16384× H100)',
      gpuId: 'h100-sxm', numGPUs: 16384, gpusPerNode: 8,
      config: {
        modelId: 'llama3-405b', sequenceLength: 8192, strategyType: 'fsdp-tp-pp',
        strategyConfig: { tp: 8, pp: 16, dp: 128, ep: 1, dpType: 'fsdp', sequenceParallel: true, pipelineSchedule: 'interleaved-1f1b', interleavedStages: 4 },
        globalBatchSize: 2048, microBatchSize: 1, activationCheckpointing: false, flashAttention: true, mixedPrecision: 'bf16',
      },
      pinned: { mfu: 0.4150, hfu: 0.4150, memUtil: 0.9491, tokPerSec: 2761753, stepTimeMs: 6075 },
    },
    {
      name: 'Nemotron-4 340B (6144× H100)',
      gpuId: 'h100-sxm', numGPUs: 6144, gpusPerNode: 8,
      config: {
        modelId: 'nemotron-4-340b', sequenceLength: 4096, strategyType: 'fsdp-tp-pp',
        strategyConfig: { tp: 8, pp: 12, dp: 64, ep: 1, dpType: 'fsdp', sequenceParallel: true, pipelineSchedule: 'interleaved-1f1b', interleavedStages: 8 },
        globalBatchSize: 768, microBatchSize: 1, activationCheckpointing: true, checkpointingGranularity: 'selective', flashAttention: true, mixedPrecision: 'bf16',
      },
      pinned: { mfu: 0.4208, hfu: 0.4507, memUtil: 0.3938, tokPerSec: 1249664, stepTimeMs: 2517 },
    },
    {
      name: 'GPT-3 175B (1024× A100)',
      gpuId: 'a100-80gb', numGPUs: 1024, gpusPerNode: 8,
      config: {
        modelId: 'gpt3-175b', sequenceLength: 2048, strategyType: 'ddp-tp-pp',
        strategyConfig: { tp: 8, pp: 8, dp: 16, ep: 1, dpType: 'ddp', sequenceParallel: false, pipelineSchedule: 'interleaved-1f1b', interleavedStages: 2 },
        globalBatchSize: 1536, microBatchSize: 1, activationCheckpointing: true, flashAttention: false, mixedPrecision: 'bf16',
      },
      pinned: { mfu: 0.4199, hfu: 0.5599, memUtil: 0.9128, tokPerSec: 127632, stepTimeMs: 24647 },
    },
    {
      name: 'DeepSeek V3 (2048× H800)',
      gpuId: 'h800-sxm', numGPUs: 2048, gpusPerNode: 8,
      config: {
        modelId: 'deepseek-v3', sequenceLength: 4096, strategyType: 'fsdp-tp-pp',
        strategyConfig: { tp: 4, pp: 8, dp: 64, ep: 32, dpType: 'fsdp', sequenceParallel: true, pipelineSchedule: 'dualpipe-v', interleavedStages: 1, numMicroBatches: 64 },
        globalBatchSize: 8192, microBatchSize: 2, activationCheckpointing: true, flashAttention: true, mixedPrecision: 'fp8',
      },
      pinned: { mfu: 0.4460, hfu: 0.5947, memUtil: 0.4959, tokPerSec: 4009248, stepTimeMs: 8369 },
    },
    {
      name: 'LLaMA 4 Maverick (512× H100)',
      clusterId: '512x-h100',
      config: {
        modelId: 'llama4-maverick', sequenceLength: 8192, strategyType: 'fsdp-tp',
        strategyConfig: { tp: 4, pp: 1, dp: 128, ep: 32, dpType: 'fsdp', sequenceParallel: true, pipelineSchedule: '1f1b', interleavedStages: 2 },
        globalBatchSize: 4096, microBatchSize: 2, activationCheckpointing: true, flashAttention: true, mixedPrecision: 'fp8',
      },
      pinned: { mfu: 0.6528, hfu: 0.8704, memUtil: 0.4044, tokPerSec: 3205775, stepTimeMs: 10467 },
    },
    {
      name: 'Grok 2.5 (512× H100)',
      clusterId: '512x-h100',
      config: {
        modelId: 'grok-2.5', sequenceLength: 8192, strategyType: 'fsdp-tp-pp',
        strategyConfig: { tp: 4, pp: 2, dp: 64, ep: 8, dpType: 'fsdp', sequenceParallel: true, pipelineSchedule: 'interleaved-1f1b', interleavedStages: 2 },
        globalBatchSize: 4096, microBatchSize: 2, activationCheckpointing: true, checkpointingGranularity: 'selective', flashAttention: true, mixedPrecision: 'fp8',
      },
      pinned: { mfu: 0.8920, hfu: 0.9175, memUtil: 0.6777, tokPerSec: 655202, stepTimeMs: 51212 },
    },
    {
      name: 'Qwen3 32B (64× H100)',
      clusterId: '64x-h100',
      config: {
        modelId: 'qwen3-32b', sequenceLength: 4096, strategyType: 'fsdp-tp',
        strategyConfig: { tp: 4, pp: 1, dp: 16, ep: 1, dpType: 'fsdp', sequenceParallel: true, pipelineSchedule: '1f1b', interleavedStages: 2 },
        globalBatchSize: 2048, microBatchSize: 8, activationCheckpointing: true, flashAttention: true, mixedPrecision: 'bf16',
      },
      pinned: { mfu: 0.4091, hfu: 0.5455, memUtil: 0.3825, tokPerSec: 131738, stepTimeMs: 63676 },
    },
    {
      name: 'OLMo 2 7B (512× H100)',
      gpuId: 'h100-sxm', numGPUs: 512, gpusPerNode: 8,
      config: {
        modelId: 'olmo2-7b', sequenceLength: 4096, strategyType: 'fsdp',
        strategyConfig: { tp: 1, pp: 1, dp: 512, ep: 1, dpType: 'fsdp', sequenceParallel: false, pipelineSchedule: '1f1b', interleavedStages: 2 },
        globalBatchSize: 1024, microBatchSize: 2, activationCheckpointing: true, flashAttention: true, mixedPrecision: 'bf16',
      },
      pinned: { mfu: 0.4055, hfu: 0.5407, memUtil: 0.1872, tokPerSec: 4689608, stepTimeMs: 894 },
    },
  ];

  for (const preset of presets) {
    it(preset.name, () => {
      let cluster: ClusterConfig | undefined;
      if (preset.clusterId) {
        cluster = getPresetCluster(preset.clusterId) ?? undefined;
      } else {
        const gpuId = preset.gpuId ?? 'h100-sxm';
        const numGPUs = preset.numGPUs ?? 8;
        const gpusPerNode = preset.gpusPerNode ?? 8;
        const numNodes = Math.ceil(numGPUs / gpusPerNode);
        cluster = numNodes === 1
          ? createSingleNodeCluster(gpuId, numGPUs) ?? undefined
          : createMultiNodeCluster(gpuId, gpusPerNode, numNodes) ?? undefined;
      }
      expect(cluster).toBeDefined();

      const metrics = sim({ ...preset.config, clusterConfig: cluster! });

      expect(metrics.memoryUtilization).toBeLessThan(1.0);
      expect(metrics.mfu).toBeGreaterThan(0);
      expect(metrics.mfu).toBeLessThanOrEqual(1.0);

      expectClose(metrics.mfu, preset.pinned.mfu, 'MFU');
      expectClose(metrics.hfu, preset.pinned.hfu, 'HFU');
      expectClose(metrics.memoryUtilization, preset.pinned.memUtil, 'Memory utilization');
      expectClose(metrics.tokensPerSecond, preset.pinned.tokPerSec, 'Tokens/sec');
      expectClose(metrics.stepTimeMs, preset.pinned.stepTimeMs, 'Step time');
    });
  }
});

// =========================================================================
// §3 — Quadratic compute scaling (flopsPerToken progression)
//
// getModel(id, seqLength) returns a ModelSpec with seq-dependent flopsPerToken.
// At short seq, flopsPerToken ≈ 2 × activeParams (attention negligible).
// At long seq, quadratic attention inflates flopsPerToken significantly.
// =========================================================================

describe('§3 — Quadratic compute scaling', () => {
  const modelId = 'llama3-405b';
  const seqs = [4096, 8192, 16384, 32768, 65536, 131072];

  const specs = seqs.map(seq => ({
    seq,
    spec: getModel(modelId, seq)!,
  }));

  it('flopsPerToken increases monotonically with seq length', () => {
    for (let i = 1; i < specs.length; i++) {
      expect(
        specs[i].spec.flopsPerToken,
        `flopsPerToken at ${specs[i].seq} should exceed ${specs[i - 1].seq}`
      ).toBeGreaterThan(specs[i - 1].spec.flopsPerToken);
    }
  });

  it('at 4K, flopsPerToken ≈ 2 × activeParams (attention negligible)', () => {
    const spec4k = specs[0].spec;
    const ratio = spec4k.flopsPerToken / (2 * spec4k.activeParams);
    // Should be close to 1.0 (within 10%) — small overhead from attention
    expect(ratio).toBeGreaterThan(0.9);
    expect(ratio).toBeLessThan(1.1);
  });

  it('at 131K, flopsPerToken > 1.5× the 4K value', () => {
    const fpt4k = specs[0].spec.flopsPerToken;
    const fpt131k = specs[specs.length - 1].spec.flopsPerToken;
    expect(fpt131k / fpt4k).toBeGreaterThan(1.5);
  });

  it('doubling ratio grows with seq (superlinear attention scaling)', () => {
    const ratios: number[] = [];
    for (let i = 1; i < specs.length; i++) {
      ratios.push(specs[i].spec.flopsPerToken / specs[i - 1].spec.flopsPerToken);
    }
    // Each doubling ratio should be >= the previous (superlinear)
    for (let i = 1; i < ratios.length; i++) {
      expect(
        ratios[i],
        `Doubling ratio at ${specs[i + 1].seq} should >= ratio at ${specs[i].seq}`
      ).toBeGreaterThanOrEqual(ratios[i - 1] * 0.99); // 1% tolerance for rounding
    }
  });
});

// =========================================================================
// §4 — Memory scaling with FlashAttention ON (linear)
//
// Activation memory should scale roughly linearly with seq when FA is on.
// =========================================================================

describe('§4 — Memory scaling with FA on (linear)', () => {
  const seqs = [4096, 8192, 16384, 32768];

  it('activation memory scales roughly linearly (ratio ≈ 2.0 per doubling)', () => {
    const cluster = makeCluster('h100-sxm', 8, 1);
    const activations = seqs.map(seq =>
      rawSim({
        clusterConfig: cluster,
        modelId: 'llama3.1-8b',
        sequenceLength: seq,
        strategyType: 'fsdp',
        globalBatchSize: 16,
        microBatchSize: 1,
        flashAttention: true,
        activationCheckpointing: false,
        mixedPrecision: 'bf16',
      }).memoryPerGPU.activations,
    );

    for (let i = 1; i < activations.length; i++) {
      const ratio = activations[i] / activations[i - 1];
      expect(
        ratio,
        `Activation ratio ${seqs[i]}/${seqs[i - 1]} = ${ratio.toFixed(2)}, expected ~2.0`
      ).toBeGreaterThan(1.7);
      expect(
        ratio,
        `Activation ratio ${seqs[i]}/${seqs[i - 1]} = ${ratio.toFixed(2)}, expected ~2.0`
      ).toBeLessThan(2.3);
    }
  });
});

// =========================================================================
// §5 — Memory scaling with FlashAttention OFF (quadratic)
//
// Without FA, the attention score matrix is O(seq²), inflating the ratio
// beyond the linear ~2.0. MLP and layernorm still scale linearly, pulling
// the aggregate ratio below 4.0 for small models.
// =========================================================================

describe('§5 — Memory scaling with FA off (quadratic)', () => {
  const cluster = makeCluster('h100-sxm', 8, 1);

  function simAtSeq(seq: number, fa: boolean): SimulationMetrics {
    return rawSim({
      clusterConfig: cluster,
      modelId: 'llama3.1-8b',
      sequenceLength: seq,
      strategyType: 'fsdp',
      globalBatchSize: 16,
      microBatchSize: 1,
      flashAttention: fa,
      activationCheckpointing: false,
      mixedPrecision: 'bf16',
    });
  }

  it('FA-off activation memory > FA-on at same seq', () => {
    const on2048 = simAtSeq(2048, true);
    const off2048 = simAtSeq(2048, false);
    expect(off2048.memoryPerGPU.activations).toBeGreaterThan(on2048.memoryPerGPU.activations);

    const on4096 = simAtSeq(4096, true);
    const off4096 = simAtSeq(4096, false);
    expect(off4096.memoryPerGPU.activations).toBeGreaterThan(on4096.memoryPerGPU.activations);
  });

  it('FA-off ratio 4096/2048 is in [2.5, 4.5] (quadratic-ish)', () => {
    const off2048 = simAtSeq(2048, false);
    const off4096 = simAtSeq(4096, false);
    const ratio = off4096.memoryPerGPU.activations / off2048.memoryPerGPU.activations;
    expect(ratio, `FA-off ratio = ${ratio.toFixed(2)}, expected [2.5, 4.5]`).toBeGreaterThan(2.5);
    expect(ratio, `FA-off ratio = ${ratio.toFixed(2)}, expected [2.5, 4.5]`).toBeLessThan(4.5);
  });

  it('FA-on ratio 4096/2048 is in [1.7, 2.3] (linear)', () => {
    const on2048 = simAtSeq(2048, true);
    const on4096 = simAtSeq(4096, true);
    const ratio = on4096.memoryPerGPU.activations / on2048.memoryPerGPU.activations;
    expect(ratio, `FA-on ratio = ${ratio.toFixed(2)}, expected [1.7, 2.3]`).toBeGreaterThan(1.7);
    expect(ratio, `FA-on ratio = ${ratio.toFixed(2)}, expected [1.7, 2.3]`).toBeLessThan(2.3);
  });
});

// =========================================================================
// §6 — CP memory progression
//
// 405B at 131K: CP divides activation memory. CP=1 OOMs, CP≥8 fits.
// =========================================================================

describe('§6 — CP memory progression', () => {
  const cps = [1, 2, 4, 8, 16];

  function cpSim(cp: number) {
    const cluster = makeCluster('h100-sxm', 8, 2048);
    return rawSim({
      clusterConfig: cluster,
      modelId: 'llama3-405b',
      sequenceLength: 131072,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: {
        tp: 8, pp: 16, cp, cpImplementation: 'all-gather' as const,
        sequenceParallel: true,
        pipelineSchedule: 'interleaved-1f1b' as const,
        interleavedStages: 4,
      },
      globalBatchSize: 2048,
      microBatchSize: 1,
      activationCheckpointing: true,
      flashAttention: true,
      mixedPrecision: 'bf16',
    });
  }

  it('activation memory scales as ~1/cp relative to CP=1', () => {
    const results = cps.map(cp => ({ cp, metrics: cpSim(cp) }));
    const actBase = results[0].metrics.memoryPerGPU.activations;
    for (const r of results.slice(1)) {
      const expected = 1 / r.cp;
      const actual = r.metrics.memoryPerGPU.activations / actBase;
      expect(
        Math.abs(actual - expected),
        `CP=${r.cp}: act ratio ${actual.toFixed(3)} vs expected ${expected.toFixed(3)}`
      ).toBeLessThan(0.05);
    }
  });

  it('CP=1 at 131K OOMs (memoryUtilization > 1.0)', () => {
    expect(cpSim(1).memoryUtilization).toBeGreaterThan(1.0);
  });

  it('CP ≥ 8 fits in 80 GB (memoryUtilization < 1.0)', () => {
    for (const cp of [8, 16]) {
      const m = cpSim(cp);
      expect(
        m.memoryUtilization,
        `CP=${cp} should fit (memUtil=${m.memoryUtilization.toFixed(3)})`
      ).toBeLessThan(1.0);
    }
  });

  it('MFU at CP=16 > 50% of MFU at CP=2 (bounded throughput degradation)', () => {
    const mfuCp2 = cpSim(2).mfu;
    const mfuCp16 = cpSim(16).mfu;
    expect(mfuCp16).toBeGreaterThan(mfuCp2 * 0.50);
  });
});

// =========================================================================
// §7 — Communication fraction vs sequence length
//
// At longer seq, quadratic compute grows faster than linear comm, so
// communicationOverhead (as fraction of total) should decrease.
// =========================================================================

describe('§7 — Communication fraction vs sequence length', () => {
  const cluster = makeCluster('h100-sxm', 8, 2048);

  it('communicationOverhead at 8K < communicationOverhead at 4K (same strategy)', () => {
    const base = {
      clusterConfig: cluster,
      modelId: 'llama3-405b',
      strategyType: 'fsdp-tp-pp' as const,
      strategyConfig: {
        tp: 8, pp: 16, cp: 1,
        sequenceParallel: true,
        pipelineSchedule: 'interleaved-1f1b' as const,
        interleavedStages: 4,
      },
      globalBatchSize: 2048,
      microBatchSize: 1,
      activationCheckpointing: true,
      flashAttention: true,
      mixedPrecision: 'bf16' as const,
    };

    const m4k = rawSim({ ...base, sequenceLength: 4096 });
    const m8k = rawSim({ ...base, sequenceLength: 8192 });

    expect(
      m8k.communicationOverhead,
      `Comm overhead at 8K (${m8k.communicationOverhead.toFixed(4)}) should be < 4K (${m4k.communicationOverhead.toFixed(4)})`
    ).toBeLessThan(m4k.communicationOverhead);
  });

  it('compute fraction at 8K > 4K when CP is held constant', () => {
    // With the same CP=1, doubling seq should increase compute fraction
    // (quadratic attention inflates compute while comm stays ~constant).
    // Varying CP across the sweep confounds this: CP=16 adds substantial
    // ring-comm overhead that offsets the quadratic compute gain.
    const base = {
      clusterConfig: cluster,
      modelId: 'llama3-405b',
      strategyType: 'fsdp-tp-pp' as const,
      strategyConfig: {
        tp: 8, pp: 16, cp: 1,
        sequenceParallel: true,
        pipelineSchedule: 'interleaved-1f1b' as const,
        interleavedStages: 4,
      },
      globalBatchSize: 2048,
      microBatchSize: 1,
      activationCheckpointing: true,
      flashAttention: true,
      mixedPrecision: 'bf16' as const,
    };

    const m4k = rawSim({ ...base, sequenceLength: 4096 });
    const m8k = rawSim({ ...base, sequenceLength: 8192 });

    const frac4k = (m4k.timing.forward + m4k.timing.backward) / m4k.stepTimeMs;
    const frac8k = (m8k.timing.forward + m8k.timing.backward) / m8k.stepTimeMs;

    expect(
      frac8k,
      `Compute fraction at 8K (${frac8k.toFixed(4)}) should exceed 4K (${frac4k.toFixed(4)})`
    ).toBeGreaterThan(frac4k);
  });
});

// =========================================================================
// §8 — Step time, throughput, and timing decomposition
//
// Small model on single node, no CP confound. Step time increases with seq,
// throughput decreases, and the superlinear ratio grows.
// =========================================================================

describe('§8 — Step time, throughput, and timing decomposition', () => {
  const seqs = [4096, 8192, 16384];

  function simAtSeq(seq: number): SimulationMetrics {
    const cluster = makeCluster('h100-sxm', 8, 1);
    return rawSim({
      clusterConfig: cluster,
      modelId: 'llama3.1-8b',
      sequenceLength: seq,
      strategyType: 'fsdp',
      globalBatchSize: 16,
      microBatchSize: 1,
      flashAttention: true,
      activationCheckpointing: false,
      mixedPrecision: 'bf16',
    });
  }

  it('stepTimeMs monotonically increases', () => {
    const results = seqs.map(simAtSeq);
    for (let i = 1; i < results.length; i++) {
      expect(results[i].stepTimeMs).toBeGreaterThan(results[i - 1].stepTimeMs);
    }
  });

  it('tokensPerSecond monotonically decreases', () => {
    const results = seqs.map(simAtSeq);
    for (let i = 1; i < results.length; i++) {
      expect(results[i].tokensPerSecond).toBeLessThan(results[i - 1].tokensPerSecond);
    }
  });

  it('step time ratio 8K/4K < 2.5 (mostly linear — MLP dominates)', () => {
    const m4k = simAtSeq(4096);
    const m8k = simAtSeq(8192);
    expect(m8k.stepTimeMs / m4k.stepTimeMs).toBeLessThan(2.5);
  });

  it('step time ratio 16K/8K > ratio 8K/4K (superlinear attention growth)', () => {
    const results = seqs.map(simAtSeq);
    const ratio8k4k = results[1].stepTimeMs / results[0].stepTimeMs;
    const ratio16k8k = results[2].stepTimeMs / results[1].stepTimeMs;
    expect(ratio16k8k).toBeGreaterThan(ratio8k4k);
  });

  it('compute fraction increases with seq length', () => {
    const results = seqs.map(seq => ({ seq, m: simAtSeq(seq) }));
    const fractions = results.map(r => (r.m.timing.forward + r.m.timing.backward) / r.m.stepTimeMs);

    for (let i = 1; i < fractions.length; i++) {
      expect(
        fractions[i],
        `Compute fraction at ${results[i].seq} should exceed ${results[i - 1].seq}`
      ).toBeGreaterThan(fractions[i - 1]);
    }
  });

  it('at 4K compute fraction is lower than at 16K', () => {
    const m4k = simAtSeq(4096);
    const m16k = simAtSeq(16384);
    const frac4k = (m4k.timing.forward + m4k.timing.backward) / m4k.stepTimeMs;
    const frac16k = (m16k.timing.forward + m16k.timing.backward) / m16k.stepTimeMs;
    expect(frac16k).toBeGreaterThan(frac4k);
  });

  it('comm time fraction decreases with seq length (if diagnostic fields available)', () => {
    const results = seqs.map(seq => ({ seq, m: simAtSeq(seq) }));
    const commFractions = results.map(r => {
      const { tpExposed, dpGross } = r.m.timing;
      if (tpExposed == null && dpGross == null) return null;
      const comm = (tpExposed ?? 0) + (dpGross ?? 0);
      return comm / r.m.stepTimeMs;
    });

    // Only assert if we have diagnostic fields
    if (commFractions.every(f => f != null)) {
      for (let i = 1; i < commFractions.length; i++) {
        expect(
          commFractions[i]!,
          `Comm fraction at ${results[i].seq} should be <= ${results[i - 1].seq}`
        ).toBeLessThanOrEqual(commFractions[i - 1]!);
      }
    }
  });
});

// =========================================================================
// §9 — Cross-model validation
//
// Validate physics properties hold across different architectures.
// =========================================================================

describe('§9 — Cross-model validation', () => {
  const models: {
    name: string;
    modelId: string;
    gpuId: string;
    gpusPerNode: number;
    numNodes: number;
    strategyType: SimulationConfig['strategyType'];
    strategyConfig?: SimulationConfig['strategyConfig'];
    gbs: number;
    mbs: number;
  }[] = [
    {
      name: 'LLaMA 3.3 70B',
      modelId: 'llama3.3-70b',
      gpuId: 'h100-sxm', gpusPerNode: 8, numNodes: 4,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8 },
      gbs: 64, mbs: 1,
    },
    {
      name: 'LLaMA 3.1 8B',
      modelId: 'llama3.1-8b',
      gpuId: 'h100-sxm', gpusPerNode: 8, numNodes: 1,
      strategyType: 'fsdp',
      gbs: 16, mbs: 1,
    },
    {
      name: 'DeepSeek V3',
      modelId: 'deepseek-v3',
      gpuId: 'h800-sxm', gpusPerNode: 8, numNodes: 256,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 4, pp: 8, ep: 32, sequenceParallel: true, pipelineSchedule: 'dualpipe-v', numMicroBatches: 64 },
      gbs: 8192, mbs: 2,
    },
  ];

  for (const m of models) {
    describe(m.name, () => {
      const spec4k = getModel(m.modelId, 4096)!;
      const spec32k = getModel(m.modelId, 32768)!;

      it('flopsPerToken at 32K > flopsPerToken at 4K', () => {
        expect(spec32k.flopsPerToken).toBeGreaterThan(spec4k.flopsPerToken);
      });

      it('activation memory ratio 32K/4K with FA=on is in [6.0, 10.0]', () => {
        const cluster = makeCluster(m.gpuId, m.gpusPerNode, m.numNodes);
        const m4k = rawSim({
          clusterConfig: cluster,
          modelId: m.modelId,
          sequenceLength: 4096,
          strategyType: m.strategyType,
          strategyConfig: m.strategyConfig,
          globalBatchSize: m.gbs,
          microBatchSize: m.mbs,
          flashAttention: true,
          activationCheckpointing: true,
          mixedPrecision: 'bf16',
        });
        const m32k = rawSim({
          clusterConfig: cluster,
          modelId: m.modelId,
          sequenceLength: 32768,
          strategyType: m.strategyType,
          strategyConfig: m.strategyConfig,
          globalBatchSize: m.gbs,
          microBatchSize: m.mbs,
          flashAttention: true,
          activationCheckpointing: true,
          mixedPrecision: 'bf16',
        });
        const ratio = m32k.memoryPerGPU.activations / m4k.memoryPerGPU.activations;
        expect(ratio, `Activation ratio 32K/4K = ${ratio.toFixed(2)}, expected [6.0, 10.0]`).toBeGreaterThan(6.0);
        expect(ratio, `Activation ratio 32K/4K = ${ratio.toFixed(2)}, expected [6.0, 10.0]`).toBeLessThan(10.0);
      });

      it('step time at 32K > step time at 4K', () => {
        const cluster = makeCluster(m.gpuId, m.gpusPerNode, m.numNodes);
        const m4k = rawSim({
          clusterConfig: cluster,
          modelId: m.modelId,
          sequenceLength: 4096,
          strategyType: m.strategyType,
          strategyConfig: m.strategyConfig,
          globalBatchSize: m.gbs,
          microBatchSize: m.mbs,
          flashAttention: true,
          activationCheckpointing: true,
          mixedPrecision: 'bf16',
        });
        const m32k = rawSim({
          clusterConfig: cluster,
          modelId: m.modelId,
          sequenceLength: 32768,
          strategyType: m.strategyType,
          strategyConfig: m.strategyConfig,
          globalBatchSize: m.gbs,
          microBatchSize: m.mbs,
          flashAttention: true,
          activationCheckpointing: true,
          mixedPrecision: 'bf16',
        });
        expect(m32k.stepTimeMs).toBeGreaterThan(m4k.stepTimeMs);
      });
    });
  }
});
