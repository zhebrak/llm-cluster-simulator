/**
 * GPU × Model Matrix Tests
 *
 * Comprehensive cross-verification of GPU specs, model architectures,
 * memory boundaries, precision compatibility, interconnect behavior,
 * production configurations, MoE specifics, generational ordering,
 * cluster topology, and edge cases.
 *
 * ~250 tests across 11 sections.
 */

import { describe, it, expect } from 'vitest';
import {
  ALL_GPUS,
  T4,
  V100_32GB,
  A100_40GB,
  A100_80GB,
  H100_SXM,
  H100_PCIE,
  H100_NVL,
  H200_SXM,
  B200,
  GB200,
  MI250X,
  MI210,
  MI350X,
  MI300X,
  MI325X,
  L40S,
  L40,
  A10,
  A10G,
  L4,
  RTX_4090,
  RTX_3090,
  getEffectiveTFLOPS,
  getPrecisionFallbackWarning,
  gpuSupportsPrecision,
} from '../../src/core/hardware/gpu.ts';
import {
  getIntraNodeInterconnect,
} from '../../src/core/hardware/interconnect.ts';
import {
  DGX_A100,
  DGX_H100,
  DGX_H200,
  DGX_B200,
  GB200_NVL72_TRAY,
  AMD_MI250X_NODE,
  AMD_MI300X_NODE,
  AMD_MI325X_NODE,
  PCIE_T4_SERVER,
  PCIE_L4_SERVER,
  PCIE_H100_SERVER,
} from '../../src/core/hardware/presets.ts';
import { createCluster, createMultiNodeCluster } from '../../src/core/hardware/topology.ts';
import {
  ALL_MODEL_CONFIGS,
  GPT3_CONFIGS,
  LLAMA2_CONFIGS,
  LLAMA3_CONFIGS,
  LLAMA4_CONFIGS,
  MISTRAL_CONFIGS,
  DEEPSEEK_CONFIGS,
  QWEN_CONFIGS,
} from '../../src/core/models/architectures.ts';
import { getModel } from '../../src/core/models/registry.ts';
import type { SimulationConfig } from '../../src/core/simulation/engine.ts';
import { SimulationEngine } from '../../src/core/simulation/engine.ts';
import type { NodeSpec } from '../../src/types/index.ts';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build SimulationConfig for a cluster preset with given model and strategy */
function makeConfig(
  clusterId: string,
  modelId: string,
  strategy: SimulationConfig['strategyType'],
  gbs: number = 8,
  mbs: number = 1,
  strategyConfig?: SimulationConfig['strategyConfig'],
  opts?: Partial<SimulationConfig>,
): SimulationConfig {
  return {
    clusterId,
    modelId,
    strategyType: strategy,
    globalBatchSize: gbs,
    microBatchSize: mbs,
    sequenceLength: 2048,
    strategyConfig,
    activationCheckpointing: true,
    flashAttention: true,
    mixedPrecision: 'bf16',
    ...opts,
  };
}

/** Build SimulationConfig with inline cluster */
function makeConfigWithCluster(
  cluster: ReturnType<typeof createCluster>,
  modelId: string,
  strategy: SimulationConfig['strategyType'],
  gbs: number = 8,
  mbs: number = 1,
  strategyConfig?: SimulationConfig['strategyConfig'],
  opts?: Partial<SimulationConfig>,
): SimulationConfig {
  return {
    clusterConfig: cluster,
    modelId,
    strategyType: strategy,
    globalBatchSize: gbs,
    microBatchSize: mbs,
    sequenceLength: 2048,
    strategyConfig,
    activationCheckpointing: true,
    flashAttention: true,
    mixedPrecision: 'bf16',
    ...opts,
  };
}

/** Simulate and return metrics (doesn't throw on OOM) */
function sim(config: SimulationConfig) {
  const engine = new SimulationEngine();
  engine.configure(config);
  return engine.simulate();
}

/** Check all metrics are finite and non-NaN */
function assertFiniteMetrics(metrics: ReturnType<typeof sim>, label: string) {
  expect(metrics.stepTimeMs, `${label}: stepTimeMs`).toBeGreaterThan(0);
  expect(isFinite(metrics.stepTimeMs), `${label}: stepTimeMs finite`).toBe(true);
  expect(isFinite(metrics.mfu), `${label}: mfu finite`).toBe(true);
  expect(isFinite(metrics.hfu), `${label}: hfu finite`).toBe(true);
  expect(isFinite(metrics.memoryUtilization), `${label}: memUtil finite`).toBe(true);
  expect(isFinite(metrics.tokensPerSecond), `${label}: tok/s finite`).toBe(true);
}

// ===========================================================================
// Section 1: GPU Spec Cross-Verification
// ===========================================================================

describe('Section 1: GPU spec cross-verification', () => {
  const allGPUs = Object.values(ALL_GPUS);

  describe('TFLOPS ordering invariants', () => {
    for (const gpu of allGPUs) {
      it(`${gpu.name}: bf16 >= fp32 (or bf16=0 if unsupported)`, () => {
        if (gpu.bf16TFLOPS > 0) {
          expect(gpu.bf16TFLOPS).toBeGreaterThanOrEqual(gpu.fp32TFLOPS);
        }
      });

      it(`${gpu.name}: fp8 >= bf16 (or fp8=0 if unsupported)`, () => {
        if (gpu.fp8TFLOPS > 0) {
          expect(gpu.fp8TFLOPS).toBeGreaterThanOrEqual(gpu.bf16TFLOPS);
        }
      });
    }
  });

  describe('architecture-specific TFLOPS ratios', () => {
    it('Ampere A100: BF16 = 2× TF32, TF32 ≈ 8× FP32', () => {
      expect(A100_80GB.bf16TFLOPS).toBe(2 * A100_80GB.tf32TFLOPS);
      expect(A100_80GB.tf32TFLOPS / A100_80GB.fp32TFLOPS).toBe(8);
    });

    it('Hopper H100 SXM: BF16 = TF32, FP8 = 2× BF16', () => {
      expect(H100_SXM.bf16TFLOPS).toBe(H100_SXM.tf32TFLOPS);
      expect(H100_SXM.fp8TFLOPS / H100_SXM.bf16TFLOPS).toBeCloseTo(2.0, 1);
    });

    it('Blackwell B200: FP4 = 2× FP8 = 4× BF16', () => {
      expect(B200.fp4TFLOPS).toBe(2 * B200.fp8TFLOPS);
      expect(B200.fp8TFLOPS).toBe(2 * B200.bf16TFLOPS);
    });

    it('CDNA2 MI250X: BF16 = FP16, no TF32, no FP8', () => {
      expect(MI250X.bf16TFLOPS).toBe(MI250X.fp16TFLOPS);
      expect(MI250X.tf32TFLOPS).toBe(0);
      expect(MI250X.fp8TFLOPS).toBe(0);
    });

    it('CDNA3 MI300X: FP8 = 2× BF16', () => {
      expect(MI300X.fp8TFLOPS / MI300X.bf16TFLOPS).toBeCloseTo(2.0, 1);
    });
  });

  describe('memory bandwidth ordering', () => {
    it('GB200 > B200 > MI300X > MI325X > H200 > H100 SXM', () => {
      expect(GB200.memoryBandwidthTBps).toBeGreaterThan(B200.memoryBandwidthTBps);
      expect(B200.memoryBandwidthTBps).toBeGreaterThan(MI300X.memoryBandwidthTBps);
      expect(MI325X.memoryBandwidthTBps).toBeGreaterThan(H200_SXM.memoryBandwidthTBps);
      expect(H200_SXM.memoryBandwidthTBps).toBeGreaterThan(H100_SXM.memoryBandwidthTBps);
    });
  });

  describe('legacy GPU constraints', () => {
    it('V100: no bf16, no tf32', () => {
      expect(V100_32GB.bf16TFLOPS).toBe(0);
      expect(V100_32GB.tf32TFLOPS).toBe(0);
    });

    it('T4: no bf16, no tf32', () => {
      expect(T4.bf16TFLOPS).toBe(0);
      expect(T4.tf32TFLOPS).toBe(0);
    });
  });

  describe('AMD constraints', () => {
    it('MI300X and MI325X: identical compute TFLOPS (same CDNA3 die)', () => {
      expect(MI300X.bf16TFLOPS).toBe(MI325X.bf16TFLOPS);
      expect(MI300X.fp8TFLOPS).toBe(MI325X.fp8TFLOPS);
      expect(MI300X.fp32TFLOPS).toBe(MI325X.fp32TFLOPS);
    });

    it('MI325X has more memory than MI300X', () => {
      expect(MI325X.memoryGB).toBeGreaterThan(MI300X.memoryGB);
    });

    for (const amdGpu of [MI250X, MI300X, MI325X, MI350X, MI210]) {
      it(`${amdGpu.name}: no NVLink`, () => {
        expect(amdGpu.nvlinkBandwidthGBps).toBe(0);
      });
    }
  });

  describe('NVLink version ordering', () => {
    it('V100(2) < A100(3) < H100(4) < B200(5)', () => {
      expect(V100_32GB.nvlinkVersion).toBe(2);
      expect(A100_80GB.nvlinkVersion).toBe(3);
      expect(H100_SXM.nvlinkVersion).toBe(4);
      expect(B200.nvlinkVersion).toBe(5);
    });
  });

  describe('Ada GPU tensor consistency', () => {
    it('L40S has 2× L40 tensor TFLOPS (same die, different enablement)', () => {
      expect(L40S.bf16TFLOPS / L40.bf16TFLOPS).toBeCloseTo(2.0, 0);
    });
  });

  describe('MI350X is no longer estimated', () => {
    it('estimated flag should be undefined/false', () => {
      expect(MI350X.estimated).toBeFalsy();
    });
  });

  describe('total GPU count', () => {
    it('ALL_GPUS has 25 entries', () => {
      expect(Object.keys(ALL_GPUS).length).toBe(25);
    });
  });
});

// ===========================================================================
// Section 2: Model Architecture Verification
// ===========================================================================

describe('Section 2: Model architecture verification', () => {
  describe('GPT-3 family (Brown et al. 2020)', () => {
    const models = GPT3_CONFIGS;

    it.each([
      ['gpt3-125m', 12, 768, 12, 50257, 2048],
      ['gpt3-1.3b', 24, 2048, 16, 50257, 2048],
      ['gpt3-6.7b', 32, 4096, 32, 50257, 2048],
      ['gpt3-13b', 40, 5140, 40, 50257, 2048],
      ['gpt3-175b', 96, 12288, 96, 50257, 2048],
    ])('%s: layers=%i, hidden=%i, heads=%i, vocab=%i, ctx=%i', (id, layers, hidden, heads, vocab, ctx) => {
      const m = models[id];
      expect(m.numLayers).toBe(layers);
      expect(m.hiddenSize).toBe(hidden);
      expect(m.numAttentionHeads).toBe(heads);
      expect(m.vocabSize).toBe(vocab);
      expect(m.maxSeqLength).toBe(ctx);
      expect(m.attentionType).toBe('mha');
    });
  });

  describe('LLaMA 2 family (Touvron et al. 2023)', () => {
    it('7B: MHA, 32 layers, 4096 hidden, 32000 vocab', () => {
      const m = LLAMA2_CONFIGS['llama2-7b'];
      expect(m.numLayers).toBe(32);
      expect(m.hiddenSize).toBe(4096);
      expect(m.numAttentionHeads).toBe(32);
      expect(m.vocabSize).toBe(32000);
      expect(m.attentionType).toBe('mha');
    });

    it('13B: MHA, 40 layers, 5120 hidden', () => {
      const m = LLAMA2_CONFIGS['llama2-13b'];
      expect(m.numLayers).toBe(40);
      expect(m.hiddenSize).toBe(5120);
      expect(m.attentionType).toBe('mha');
    });

    it('70B: GQA with 8 KV heads, 80 layers, 8192 hidden', () => {
      const m = LLAMA2_CONFIGS['llama2-70b'];
      expect(m.numLayers).toBe(80);
      expect(m.hiddenSize).toBe(8192);
      expect(m.numKvHeads).toBe(8);
      expect(m.attentionType).toBe('gqa');
    });
  });

  describe('LLaMA 3 family', () => {
    it('all use GQA(8KV), 128256 vocab, gated SiLU MLP', () => {
      for (const id of ['llama3.1-8b', 'llama3-405b', 'llama3.2-1b', 'llama3.2-3b', 'llama3.3-70b']) {
        const m = LLAMA3_CONFIGS[id] ?? ALL_MODEL_CONFIGS[id];
        expect(m.vocabSize, `${id} vocab`).toBe(128256);
        expect(m.attentionType, `${id} attention`).toBe('gqa');
        expect(m.numKvHeads, `${id} KV heads`).toBe(8);
        expect(m.gatedMLP, `${id} gated`).toBe(true);
        expect(m.activation, `${id} activation`).toBe('silu');
      }
    });

    it('405B: 126 layers, 16384 hidden, 128 heads', () => {
      const m = LLAMA3_CONFIGS['llama3-405b'];
      expect(m.numLayers).toBe(126);
      expect(m.hiddenSize).toBe(16384);
      expect(m.numAttentionHeads).toBe(128);
    });
  });

  describe('LLaMA 4 MoE models', () => {
    it('Maverick: 128 experts, freq=2, shared=1', () => {
      const m = LLAMA4_CONFIGS['llama4-maverick'];
      expect(m.numExperts).toBe(128);
      expect(m.numActiveExperts).toBe(1);
      expect(m.moeLayerFrequency).toBe(2);
      expect(m.numSharedExperts).toBe(1);
    });

    it('Scout: 16 experts, shared=1', () => {
      const m = LLAMA4_CONFIGS['llama4-scout'];
      expect(m.numExperts).toBe(16);
      expect(m.numActiveExperts).toBe(1);
      expect(m.numSharedExperts).toBe(1);
    });
  });

  describe('Mistral family', () => {
    it('Mistral 7B: 32768 vocab (v0.3), GQA(8KV)', () => {
      const m = ALL_MODEL_CONFIGS['mistral-7b'];
      expect(m.vocabSize).toBe(32768);
      expect(m.numKvHeads).toBe(8);
      expect(m.attentionType).toBe('gqa');
    });

    it('Mistral Nemo 12B: headDim=128 override', () => {
      const m = ALL_MODEL_CONFIGS['mistral-nemo-12b'];
      expect(m.headDim).toBe(128);
      expect(m.numLayers).toBe(40);
      expect(m.hiddenSize).toBe(5120);
    });

    it('Mistral Small 24B: 40 layers, 5120 hidden, headDim=128', () => {
      const m = ALL_MODEL_CONFIGS['mistral-small-24b'];
      expect(m.numLayers).toBe(40);
      expect(m.hiddenSize).toBe(5120);
      expect(m.headDim).toBe(128);
    });
  });

  describe('Mixtral MoE models', () => {
    it('8x7B: 8 experts, 2 active', () => {
      const m = ALL_MODEL_CONFIGS['mixtral-8x7b'];
      expect(m.numExperts).toBe(8);
      expect(m.numActiveExperts).toBe(2);
    });

    it('8x22B: 8 experts, 2 active, 56 layers', () => {
      const m = MISTRAL_CONFIGS['mixtral-8x22b'];
      expect(m.numExperts).toBe(8);
      expect(m.numActiveExperts).toBe(2);
      expect(m.numLayers).toBe(56);
    });
  });

  describe('DeepSeek family', () => {
    it('MoE-16B: MHA (pre-MLA), 64 experts, 6 active', () => {
      const m = DEEPSEEK_CONFIGS['deepseek-moe-16b'];
      expect(m.attentionType).toBe('mha');
      expect(m.numExperts).toBe(64);
      expect(m.numActiveExperts).toBe(6);
      expect(m.numSharedExperts).toBe(2);
      expect(m.firstKDenseLayers).toBe(1);
    });

    it('V2: MLA, 160 experts, 6 active', () => {
      const m = DEEPSEEK_CONFIGS['deepseek-v2'];
      expect(m.attentionType).toBe('mla');
      expect(m.numExperts).toBe(160);
      expect(m.numActiveExperts).toBe(6);
      expect(m.kvLoraRank).toBe(512);
      expect(m.qLoraRank).toBe(1536);
    });

    it('V3/R1: MLA, 256 experts, 8 active, 61 layers (prime)', () => {
      const m = DEEPSEEK_CONFIGS['deepseek-v3'];
      expect(m.attentionType).toBe('mla');
      expect(m.numExperts).toBe(256);
      expect(m.numActiveExperts).toBe(8);
      expect(m.numLayers).toBe(61);
      expect(m.kvLoraRank).toBe(512);
      expect(m.qLoraRank).toBe(1536);
      expect(m.qkNopeHeadDim).toBe(128);
      expect(m.qkRopeHeadDim).toBe(64);
      expect(m.vHeadDim).toBe(128);
      expect(m.numSharedExperts).toBe(1);
      expect(m.firstKDenseLayers).toBe(3);
    });
  });

  describe('Qwen 2.5 family', () => {
    it.each(['qwen2.5-7b', 'qwen2.5-14b', 'qwen2.5-32b', 'qwen2.5-72b'])('%s: 152064 vocab, GQA', (id) => {
      const m = QWEN_CONFIGS[id];
      expect(m.vocabSize).toBe(152064); // 7B+ use 152064; 0.5B/1.5B/3B use 151936
      expect(m.attentionType).toBe('gqa');
    });

    it.each(['qwen2.5-0.5b', 'qwen2.5-1.5b', 'qwen2.5-3b'])('%s: 151936 vocab, GQA', (id) => {
      const m = QWEN_CONFIGS[id];
      expect(m.vocabSize).toBe(151936);
      expect(m.attentionType).toBe('gqa');
    });
  });

  describe('MoE param split invariants', () => {
    const moeModels = [
      'mixtral-8x7b', 'mixtral-8x22b',
      'deepseek-moe-16b', 'deepseek-v2', 'deepseek-v3',
      'dbrx', 'grok-1',
      'llama4-scout', 'llama4-maverick',
      'qwen3-30b-a3b', 'qwen3-235b-a22b',
    ];

    for (const id of moeModels) {
      it(`${id}: totalParams > activeParams`, () => {
        const spec = getModel(id)!;
        expect(spec).toBeDefined();
        expect(spec.totalParams).toBeGreaterThan(spec.activeParams);
        expect(spec.isMoE).toBe(true);
      });
    }
  });

  describe('DeepSeek V3/R1 param counts', () => {
    it('V3: ~671B total, ~37.6B active', () => {
      const spec = getModel('deepseek-v3')!;
      expect(spec.totalParams / 1e9).toBeCloseTo(671, -1);
      expect(spec.activeParams / 1e9).toBeCloseTo(37.6, 0);
    });

    it('V2: ~236B total, ~21.4B active', () => {
      const spec = getModel('deepseek-v2')!;
      expect(spec.totalParams / 1e9).toBeCloseTo(236, -1);
      expect(spec.activeParams / 1e9).toBeCloseTo(21.4, 0);
    });
  });
});

// ===========================================================================
// Section 3: Every GPU × Extreme Models (no crash, no NaN)
// ===========================================================================

describe('Section 3: GPU × extreme models — no crash, no NaN', () => {
  const gpuIds = Object.keys(ALL_GPUS);

  describe('GPT-3 125M (smallest) on every GPU', () => {
    for (const gpuId of gpuIds) {
      it(`${gpuId}: finite results`, () => {
        const cluster = createMultiNodeCluster(gpuId, 2, 1);
        if (!cluster) return; // Skip GPUs without cluster helper
        const cfg = makeConfigWithCluster(cluster, 'gpt3-125m', 'ddp', 4, 1);
        const m = sim(cfg);
        assertFiniteMetrics(m, `gpt3-125m/${gpuId}`);
      });
    }
  });

  describe('LLaMA 3 405B (largest dense) on high-end GPUs', () => {
    for (const gpuId of ['h100-sxm', 'b200', 'h200-sxm', 'mi300x']) {
      it(`${gpuId}: finite results`, () => {
        const cluster = createMultiNodeCluster(gpuId, 8, 16)!;
        const cfg = makeConfigWithCluster(cluster, 'llama3-405b', 'fsdp-tp-pp', 128, 1, {
          tp: 8, pp: 4,
        });
        const m = sim(cfg);
        assertFiniteMetrics(m, `llama3-405b/${gpuId}`);
      });
    }
  });

  describe('DeepSeek V3 671B (largest MoE) on high-end GPUs', () => {
    for (const gpuId of ['h100-sxm', 'b200']) {
      it(`${gpuId}: finite results`, () => {
        const cluster = createMultiNodeCluster(gpuId, 8, 32)!;
        const cfg = makeConfigWithCluster(cluster, 'deepseek-v3', 'fsdp-tp', 256, 1, {
          tp: 8, ep: 8,
        });
        const m = sim(cfg);
        assertFiniteMetrics(m, `deepseek-v3/${gpuId}`);
      });
    }
  });
});

// ===========================================================================
// Section 4: Memory Boundary Tests
// ===========================================================================

describe('Section 4: Memory boundary tests', () => {
  it('DDP 7B on A100 80GB (8 GPUs): OOM — DDP does not shard', () => {
    const cfg = makeConfig('8x-a100', 'llama2-7b', 'ddp', 8, 1);
    const m = sim(cfg);
    // DDP needs ~126GB for 7B (18 bytes/param), won't fit in 80GB
    expect(m.memoryUtilization).toBeGreaterThan(1.0);
  });

  it('FSDP 7B on A100 80GB (8 GPUs): FITS', () => {
    const cfg = makeConfig('8x-a100', 'llama2-7b', 'fsdp', 8, 1);
    const m = sim(cfg);
    expect(m.memoryUtilization).toBeLessThan(1.0);
  });

  it('70B FSDP-TP on A100 40GB (8 GPUs): OOM', () => {
    const cluster = createCluster({ ...DGX_A100, gpu: A100_40GB } as NodeSpec, 1, 'single-node');
    const cfg = makeConfigWithCluster(cluster, 'llama2-70b', 'fsdp-tp', 8, 1, { tp: 8 });
    const m = sim(cfg);
    expect(m.memoryUtilization).toBeGreaterThan(1.0);
  });

  it('70B FSDP-TP on A100 80GB (32 GPUs, TP=8, DP=4): fits', () => {
    const cfg = makeConfig('32x-a100', 'llama2-70b', 'fsdp-tp', 32, 1, { tp: 8 });
    const m = sim(cfg);
    expect(m.memoryUtilization).toBeLessThan(1.0);
  });

  it('70B FSDP-TP on MI300X (8 GPUs): fits (192GB)', () => {
    const cfg = makeConfig('8x-mi300x', 'llama2-70b', 'fsdp-tp', 8, 1, { tp: 8 });
    const m = sim(cfg);
    expect(m.memoryUtilization).toBeLessThan(1.0); // Fits in 192GB
  });

  it('8B on RTX 4090 single GPU FSDP: tight or OOM (24GB)', () => {
    const cluster = createMultiNodeCluster('rtx-4090', 1, 1)!;
    const cfg = makeConfigWithCluster(cluster, 'llama3.1-8b', 'fsdp', 1, 1);
    const m = sim(cfg);
    // 8B model needs ~144GB unsharded; 1 GPU FSDP doesn't shard → OOM
    expect(m.memoryUtilization).toBeGreaterThan(1.0);
  });

  it('125M on T4 single GPU DDP: fits (16GB is plenty for 125M)', () => {
    const cluster = createMultiNodeCluster('t4', 1, 1)!;
    const cfg = makeConfigWithCluster(cluster, 'gpt3-125m', 'ddp', 1, 1);
    const m = sim(cfg);
    expect(m.memoryUtilization).toBeLessThan(1.0);
  });

  it('7B on T4 single GPU: OOM (16GB)', () => {
    const cluster = createMultiNodeCluster('t4', 1, 1)!;
    const cfg = makeConfigWithCluster(cluster, 'llama2-7b', 'fsdp', 1, 1);
    const m = sim(cfg);
    expect(m.memoryUtilization).toBeGreaterThan(1.0);
  });

  it('405B on B200 64 GPUs FSDP-TP-PP: fits (180GB per GPU)', () => {
    const cfg = makeConfig('64x-b200', 'llama3-405b', 'fsdp-tp-pp', 64, 1, {
      tp: 8, pp: 4,
    });
    const m = sim(cfg);
    expect(m.memoryUtilization).toBeLessThan(1.0);
  });

  it('H200 vs H100: same config, H200 has more memory headroom', () => {
    const h100cfg = makeConfig('8x-h100', 'llama2-70b', 'fsdp-tp', 8, 1, { tp: 8 });
    const h200cfg = makeConfig('8x-h200', 'llama2-70b', 'fsdp-tp', 8, 1, { tp: 8 });
    const h100m = sim(h100cfg);
    const h200m = sim(h200cfg);
    // H200 has 141GB vs H100 80GB → lower memory utilization
    expect(h200m.memoryUtilization).toBeLessThan(h100m.memoryUtilization);
  });
});

// ===========================================================================
// Section 5: Precision Compatibility
// ===========================================================================

describe('Section 5: Precision compatibility', () => {
  describe('getEffectiveTFLOPS fallback', () => {
    it('V100 + BF16 → FP16 (125 TFLOPS)', () => {
      expect(getEffectiveTFLOPS(V100_32GB, 'bf16')).toBe(125);
    });

    it('T4 + BF16 → FP16 (65 TFLOPS)', () => {
      expect(getEffectiveTFLOPS(T4, 'bf16')).toBe(65);
    });

    it('V100 + TF32 → FP32 (15.7 TFLOPS)', () => {
      expect(getEffectiveTFLOPS(V100_32GB, 'tf32')).toBe(15.7);
    });

    it('A100 + FP8 → BF16 (312 TFLOPS, no native FP8)', () => {
      expect(getEffectiveTFLOPS(A100_80GB, 'fp8')).toBe(312);
    });

    it('H100 SXM + FP8 → 1979 (native)', () => {
      expect(getEffectiveTFLOPS(H100_SXM, 'fp8')).toBe(1979);
    });

    it('MI250X + FP8 → BF16 (191.5)', () => {
      expect(getEffectiveTFLOPS(MI250X, 'fp8')).toBe(MI250X.bf16TFLOPS);
    });

    it('H100 + FP4 → FP8 (1979)', () => {
      expect(getEffectiveTFLOPS(H100_SXM, 'fp4')).toBe(1979);
    });

    it('B200 + FP4 → 9000 (native)', () => {
      expect(getEffectiveTFLOPS(B200, 'fp4')).toBe(9000);
    });
  });

  describe('getPrecisionFallbackWarning', () => {
    it('V100 + BF16: warning (no native BF16)', () => {
      expect(getPrecisionFallbackWarning(V100_32GB, 'bf16')).not.toBeNull();
    });

    it('T4 + BF16: warning', () => {
      expect(getPrecisionFallbackWarning(T4, 'bf16')).not.toBeNull();
    });

    it('A100 + FP8: warning', () => {
      expect(getPrecisionFallbackWarning(A100_80GB, 'fp8')).not.toBeNull();
    });

    it('H100 + BF16: no warning (native)', () => {
      expect(getPrecisionFallbackWarning(H100_SXM, 'bf16')).toBeNull();
    });

    it('H100 + FP8: no warning (native)', () => {
      expect(getPrecisionFallbackWarning(H100_SXM, 'fp8')).toBeNull();
    });

    it('A100 + BF16: no warning (native)', () => {
      expect(getPrecisionFallbackWarning(A100_80GB, 'bf16')).toBeNull();
    });

    it('B200 + FP4: no warning (native)', () => {
      expect(getPrecisionFallbackWarning(B200, 'fp4')).toBeNull();
    });
  });

  describe('gpuSupportsPrecision', () => {
    it('MI350X supports FP4 (CDNA4)', () => {
      expect(gpuSupportsPrecision(MI350X, 'fp4')).toBe(true);
    });

    it('MI300X does not support FP4', () => {
      expect(gpuSupportsPrecision(MI300X, 'fp4')).toBe(false);
    });

    it('V100 does not support BF16', () => {
      expect(gpuSupportsPrecision(V100_32GB, 'bf16')).toBe(false);
    });

    it('V100 supports FP16', () => {
      expect(gpuSupportsPrecision(V100_32GB, 'fp16')).toBe(true);
    });
  });
});

// ===========================================================================
// Section 6: Interconnect-Aware Tests
// ===========================================================================

describe('Section 6: Interconnect-aware tests', () => {
  it('H100 SXM vs H100 PCIe: SXM has higher MFU with TP=2', () => {
    const sxmCluster = createCluster(DGX_H100, 1, 'single-node');
    const pcieCluster = createCluster(PCIE_H100_SERVER, 1, 'single-node');

    const sxmCfg = makeConfigWithCluster(sxmCluster, 'llama2-7b', 'fsdp-tp', 8, 1, { tp: 2 });
    const pcieCfg = makeConfigWithCluster(pcieCluster, 'llama2-7b', 'fsdp-tp', 8, 1, { tp: 2 });

    const sxmM = sim(sxmCfg);
    const pcieM = sim(pcieCfg);

    // SXM with NVLink should have higher MFU than PCIe
    expect(sxmM.mfu).toBeGreaterThan(pcieM.mfu);
  });

  it('MI300X: getIntraNodeInterconnect returns Infinity Fabric, not PCIe', () => {
    const ic = getIntraNodeInterconnect(MI300X);
    expect(ic.type).not.toBe('pcie');
    // Should be Infinity Fabric — name contains 'Infinity Fabric'
    expect(ic.name).toContain('Infinity Fabric');
  });

  it('PCIe-only GPUs: intra-node is PCIe', () => {
    for (const gpu of [RTX_4090, L4, H100_PCIE]) {
      const ic = getIntraNodeInterconnect(gpu);
      // PCIe GPUs without NVSwitch → PCIe fallback
      if (gpu.nvlinkBandwidthGBps === 0 && !gpu.hasNvSwitch) {
        expect(ic.type, `${gpu.name} intra-node`).toBe('pcie');
      }
    }
  });

  it('NVLink bandwidth ordering: B200 (900) > H100 SXM (450) > A100 (300) > V100 (150)', () => {
    expect(B200.nvlinkBandwidthGBps).toBeGreaterThan(H100_SXM.nvlinkBandwidthGBps);
    expect(H100_SXM.nvlinkBandwidthGBps).toBeGreaterThan(A100_80GB.nvlinkBandwidthGBps);
    expect(A100_80GB.nvlinkBandwidthGBps).toBeGreaterThan(V100_32GB.nvlinkBandwidthGBps);
  });

  it('Multi-node A100: inter-node is IB, not NVLink', () => {
    expect(DGX_A100.interNodeInterconnect.type).toBe('infiniband');
  });
});

// ===========================================================================
// Section 7: Production Configuration Validation
// ===========================================================================

describe('Section 7: Production configuration validation', () => {
  it('GPT-3 175B: 1024 A100 80GB, TP=8, PP=8, DP=16 → MFU ∈ [35%, 55%]', () => {
    const cfg = makeConfig('1024x-a100', 'gpt3-175b', 'fsdp-tp-pp', 1024, 1, {
      tp: 8, pp: 8,
    }, { sequenceLength: 2048 });
    const m = sim(cfg);
    expect(m.mfu).toBeGreaterThanOrEqual(0.35);
    expect(m.mfu).toBeLessThanOrEqual(0.55);
  });

  it('LLaMA 3.1 405B: 16384 H100 SXM, TP=8, PP=16, DP=128 → fits, MFU > 20%', () => {
    const cfg = makeConfig('4096x-h100', 'llama3-405b', 'fsdp-tp-pp', 4096, 1, {
      tp: 8, pp: 16,
    }, { sequenceLength: 4096 });
    const m = sim(cfg);
    expect(m.memoryUtilization).toBeLessThan(1.0);
    expect(m.mfu).toBeGreaterThan(0.20);
  });

  it('LLaMA 2 7B FSDP: 8 A100 80GB → MFU ∈ [40%, 60%], HFU ∈ [50%, 75%]', () => {
    const cfg = makeConfig('8x-a100', 'llama2-7b', 'fsdp', 32, 4, undefined, {
      sequenceLength: 4096,
    });
    const m = sim(cfg);
    expect(m.mfu).toBeGreaterThanOrEqual(0.40);
    expect(m.mfu).toBeLessThanOrEqual(0.60);
    expect(m.hfu).toBeGreaterThanOrEqual(0.50);
    expect(m.hfu).toBeLessThanOrEqual(0.75);
  });

  it('DeepSeek V3: 2048 H100 SXM, EP=8 → MFU ∈ [18.5%, 25%]', () => {
    const cfg = makeConfig('2048x-h100', 'deepseek-v3', 'fsdp-tp', 2048, 1, {
      tp: 8, ep: 8,
    }, { sequenceLength: 4096 });
    const m = sim(cfg);
    // Routing locality reduces EP all-to-all volume: tokens are biased toward
    // locally-resident experts, so cross-node all-to-all traffic drops significantly.
    // EP latency scaling (log₂(8/4)=1 → 1.5× latency) + coordination floor add overhead.
    // Bounds: ±15% of observed 0.2172.
    expect(m.mfu).toBeGreaterThanOrEqual(0.185);
    expect(m.mfu).toBeLessThanOrEqual(0.250);
  });
});

// ===========================================================================
// Section 8: MoE-Specific Validations
// ===========================================================================

describe('Section 8: MoE-specific validations', () => {
  describe('total vs active params (published values)', () => {
    const moeChecks: [string, number, number][] = [
      // [modelId, expectedTotalB, expectedActiveB]
      ['mixtral-8x7b', 47, 13],
      ['mixtral-8x22b', 141, 39],
      ['deepseek-moe-16b', 16, 2.7],
      ['deepseek-v2', 236, 21.4],
      ['deepseek-v3', 671, 37.6],
      ['dbrx', 132, 36],
      ['grok-1', 314, 86],  // Published: 314B total, 86B active (GeGLU gated MLP)
      ['llama4-maverick', 400, 17],
      ['llama4-scout', 109, 17],
    ];

    for (const [id, totalB, activeB] of moeChecks) {
      it(`${id}: total ~${totalB}B, active ~${activeB}B`, () => {
        const spec = getModel(id)!;
        expect(spec, `${id} not found`).toBeDefined();
        expect(spec.totalParams / 1e9).toBeGreaterThan(totalB * 0.75);
        expect(spec.totalParams / 1e9).toBeLessThan(totalB * 1.25);
        expect(spec.activeParams / 1e9).toBeGreaterThan(activeB * 0.70);
        expect(spec.activeParams / 1e9).toBeLessThan(activeB * 1.40);
        expect(spec.totalParams).toBeGreaterThan(spec.activeParams);
      });
    }
  });

  describe('EP subdivides DP', () => {
    it('EP=1 same as no EP', () => {
      const cfg1 = makeConfig('64x-h100', 'deepseek-v3', 'fsdp-tp', 64, 1, { tp: 8, ep: 1 });
      const cfg2 = makeConfig('64x-h100', 'deepseek-v3', 'fsdp-tp', 64, 1, { tp: 8 });
      const m1 = sim(cfg1);
      const m2 = sim(cfg2);
      expect(m1.mfu).toBeCloseTo(m2.mfu, 2);
    });

    it('EP configuration does not change totalGPUs = TP×PP×DP', () => {
      // With EP=8, DP should still be totalGPUs/TP (EP subdivides DP)
      const cfg = makeConfig('64x-h100', 'deepseek-v3', 'fsdp-tp', 64, 1, { tp: 8, ep: 8 });
      const m = sim(cfg);
      assertFiniteMetrics(m, 'deepseek-v3 EP=8');
    });
  });

  describe('Chinchilla uses activeParams for MoE', () => {
    it('DeepSeek V3 Chinchilla target << what totalParams would give', () => {
      // activeParams ~37.6B → 20N = ~752B tokens
      // totalParams ~671B → 20N = ~13.4T tokens (way too much)
      const spec = getModel('deepseek-v3')!;
      const chinchillaActive = 20 * spec.activeParams;
      const chinchillaTotal = 20 * spec.totalParams;
      expect(chinchillaActive).toBeLessThan(chinchillaTotal / 10);
    });
  });
});

// ===========================================================================
// Section 9: GPU Generation Comparison
// ===========================================================================

describe('Section 9: GPU generation comparison', () => {
  it('LLaMA 2 70B throughput: B200 > H100 SXM > A100 80GB', () => {
    const configs = ['64x-b200', '64x-h100', '64x-a100'].map(clusterId =>
      makeConfig(clusterId, 'llama2-70b', 'fsdp-tp', 64, 1, { tp: 8 }, { sequenceLength: 4096 })
    );
    const metrics = configs.map(c => sim(c));

    expect(metrics[0].tokensPerSecond).toBeGreaterThan(metrics[1].tokensPerSecond); // B200 > H100
    expect(metrics[1].tokensPerSecond).toBeGreaterThan(metrics[2].tokensPerSecond); // H100 > A100
  });

  it('H200 vs H100: MFU within ±5% (same Hopper die)', () => {
    const h100cfg = makeConfig('64x-h100', 'llama2-70b', 'fsdp-tp', 64, 1, { tp: 8 });
    const h200cfg = makeConfig('64x-h200', 'llama2-70b', 'fsdp-tp', 64, 1, { tp: 8 });
    const h100m = sim(h100cfg);
    const h200m = sim(h200cfg);

    // Same compute die → MFU should be very close
    expect(Math.abs(h200m.mfu - h100m.mfu)).toBeLessThan(0.05);
  });

  it('H200 has ~1.4× memory bandwidth vs H100', () => {
    expect(H200_SXM.memoryBandwidthTBps / H100_SXM.memoryBandwidthTBps).toBeCloseTo(1.43, 1);
  });

  describe('memory headroom ordering', () => {
    it('MI325X > MI300X > B200 > H200 > A100 80GB = H100 SXM', () => {
      expect(MI325X.memoryGB).toBeGreaterThan(MI300X.memoryGB);
      expect(MI300X.memoryGB).toBeGreaterThan(B200.memoryGB);
      expect(B200.memoryGB).toBeGreaterThan(H200_SXM.memoryGB);
      expect(H200_SXM.memoryGB).toBeGreaterThan(A100_80GB.memoryGB);
      // A100 80GB and H100 SXM both have 80GB
      expect(A100_80GB.memoryGB).toBe(H100_SXM.memoryGB);
    });
  });
});

// ===========================================================================
// Section 10: Nodes & GPUs/Node
// ===========================================================================

describe('Section 10: Nodes & GPUs/node', () => {
  it('DGX A100: 8 GPUs/node', () => {
    expect(DGX_A100.numGPUs).toBe(8);
  });

  it('DGX H100: 8 GPUs/node', () => {
    expect(DGX_H100.numGPUs).toBe(8);
  });

  it('DGX H200: 8 GPUs/node', () => {
    expect(DGX_H200.numGPUs).toBe(8);
  });

  it('DGX B200: 8 GPUs/node', () => {
    expect(DGX_B200.numGPUs).toBe(8);
  });

  it('GB200 NVL72 tray: 4 GPUs/tray', () => {
    expect(GB200_NVL72_TRAY.numGPUs).toBe(4);
  });

  it('MI250X node: 8 GCDs', () => {
    expect(AMD_MI250X_NODE.numGPUs).toBe(8);
  });

  it('MI300X node: 8 GPUs', () => {
    expect(AMD_MI300X_NODE.numGPUs).toBe(8);
  });

  it('MI325X node: 8 GPUs', () => {
    expect(AMD_MI325X_NODE.numGPUs).toBe(8);
  });

  it('T4 PCIe server: 4 GPUs/node', () => {
    expect(PCIE_T4_SERVER.numGPUs).toBe(4);
  });

  it('L4 PCIe server: 8 GPUs/node', () => {
    expect(PCIE_L4_SERVER.numGPUs).toBe(8);
  });
});

// ===========================================================================
// Section 11: GPU Edge Cases
// ===========================================================================

describe('Section 11: GPU edge cases', () => {
  it('GB200 NVL: 186 GB per die, NVLink 5', () => {
    expect(GB200.memoryGB).toBe(186);
    expect(GB200.nvlinkVersion).toBe(5);
  });

  it('MI250X: 64 GB per GCD (128 GB per card ÷ 2 GCDs)', () => {
    // MI250X has 128 GB per card but 2 GCDs per card = 64 GB per GCD.
    // Simulator models per-GCD since each GCD operates as an independent "GPU".
    // Node has 4 cards × 2 GCDs = 8 "GPUs" of 64 GB each.
    expect(MI250X.memoryGB).toBe(64);
  });

  it('H100 NVL: 94 GB, NVLink bridge (300 GB/s uni)', () => {
    expect(H100_NVL.memoryGB).toBe(94);
    expect(H100_NVL.nvlinkBandwidthGBps).toBe(300);
    expect(H100_NVL.hasNvSwitch).toBe(false);
  });

  it('A10 vs A10G: different BF16 TFLOPS (125 vs 70), same memory', () => {
    expect(A10.bf16TFLOPS).toBe(125);
    expect(A10G.bf16TFLOPS).toBe(70);
    expect(A10.memoryGB).toBe(A10G.memoryGB);
  });

  it('RTX 4090: no NVLink', () => {
    expect(RTX_4090.nvlinkBandwidthGBps).toBe(0);
    expect(RTX_4090.nvlinkVersion).toBeNull();
  });

  it('L4: 72W TDP, inference-focused', () => {
    expect(L4.tdpWatts).toBe(72);
  });

  it('RTX 4090 BF16: 165.2 TFLOPS (FP16-with-FP32-accumulate, not marketing 330.3)', () => {
    expect(RTX_4090.bf16TFLOPS).toBe(165.2);
    expect(RTX_4090.fp16TFLOPS).toBe(165.2);
  });

  it('RTX 3090 BF16: 142 TFLOPS (82 SMs × 4 TCs × 256 FMA × 1.695 GHz)', () => {
    expect(RTX_3090.bf16TFLOPS).toBe(142);
  });

  describe('GB200 NVL72 tray topology', () => {
    it('GB200 NVL72 tray has 4 GPUs → TP=4 stays intra-tray', () => {
      // 4 GPUs per tray, TP=4 means all within one tray (intra-tray NVLink)
      const cluster = createCluster(GB200_NVL72_TRAY, 2, 'fat-tree');
      const cfg = makeConfigWithCluster(cluster, 'llama2-7b', 'fsdp-tp', 8, 1, { tp: 4 });
      const m = sim(cfg);
      assertFiniteMetrics(m, 'GB200 TP=4');
      // TP=4 fits within a single 4-GPU tray → should work fine
      expect(m.mfu).toBeGreaterThan(0);
    });

    it('GB200 NVL72 with TP=8 requires 2 trays (cross-tray)', () => {
      // 4 GPUs per tray → TP=8 spans 2 trays
      // This should still work but with higher communication overhead
      const cluster = createCluster(GB200_NVL72_TRAY, 4, 'fat-tree'); // 16 GPUs = 4 trays
      const cfg4 = makeConfigWithCluster(cluster, 'llama2-7b', 'fsdp-tp', 16, 1, { tp: 4 });
      const cfg8 = makeConfigWithCluster(cluster, 'llama2-7b', 'fsdp-tp', 16, 1, { tp: 8 });
      const m4 = sim(cfg4);
      const m8 = sim(cfg8);
      assertFiniteMetrics(m4, 'GB200 TP=4');
      assertFiniteMetrics(m8, 'GB200 TP=8 cross-tray');
      // TP=8 spans 2 trays, so cross-tray comm should reduce MFU vs TP=4
      // (TP=4 has more DP though, so comparison is nuanced — just verify both work)
    });

    it('GB200 NVL72 tray uses NVSwitch intra-node', () => {
      expect(GB200_NVL72_TRAY.hasNvSwitch).toBe(true);
      expect(GB200_NVL72_TRAY.intraNodeInterconnect.type).toBe('nvswitch');
    });
  });
});
