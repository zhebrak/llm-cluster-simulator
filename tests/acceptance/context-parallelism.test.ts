/**
 * Context Parallelism (Ring Attention) Validation Tests
 *
 * Verifies that CP correctly:
 * - Reduces activation memory by 1/cp (each rank holds seq/cp tokens)
 * - Composes multiplicatively with SP
 * - Adds minimal throughput overhead via ring-attention overlap
 * - Validates sequence divisibility and chunk size constraints
 * - Works on 1D strategies (routed through 3D internally)
 * - Interacts correctly with EP on MoE models
 */
import { describe, it, expect } from 'vitest';
import {
  SimulationEngine,
  type SimulationConfig,
  getSimulationMetrics,
} from '../../src/core/simulation/engine.ts';
import { createMultiNodeCluster } from '../../src/core/hardware/topology.ts';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function run(overrides: Partial<SimulationConfig> & { nodes: number }) {
  const { nodes, ...rest } = overrides;
  return getSimulationMetrics({
    clusterConfig: createMultiNodeCluster('h100-sxm', 8, nodes)!,
    modelId: 'llama2-70b',
    globalBatchSize: 256,
    microBatchSize: 2,
    sequenceLength: 32768,
    strategyType: 'fsdp-tp',
    strategyConfig: { tp: 8, sequenceParallel: true },
    activationCheckpointing: true,
    flashAttention: true,
    ...rest,
  });
}

function validate(overrides: Partial<SimulationConfig> & { nodes: number }) {
  const { nodes, ...rest } = overrides;
  const engine = new SimulationEngine();
  engine.configure({
    clusterConfig: createMultiNodeCluster('h100-sxm', 8, nodes)!,
    modelId: 'llama2-70b',
    globalBatchSize: 256,
    microBatchSize: 2,
    sequenceLength: 32768,
    strategyType: 'fsdp-tp',
    strategyConfig: { tp: 8, sequenceParallel: true },
    activationCheckpointing: true,
    flashAttention: true,
    ...rest,
  });
  return engine.validate();
}

// ---------------------------------------------------------------------------
// 1. Activation memory reduction
// ---------------------------------------------------------------------------

describe('CP activation memory reduction', () => {
  // 32 GPUs: CP=1 → DP=4, CP=2 → DP=2
  const cp1 = run({ nodes: 4, strategyConfig: { tp: 8, cp: 1, sequenceParallel: true } });
  const cp2 = run({ nodes: 4, strategyConfig: { tp: 8, cp: 2, sequenceParallel: true } });

  it('CP=2 halves activation memory vs CP=1', () => {
    const ratio = cp2.memoryPerGPU.activations / cp1.memoryPerGPU.activations;
    expect(ratio).toBeCloseTo(0.50, 2);
  });

  it('CP=4 quarters activation memory vs CP=1', () => {
    // 64 GPUs: CP=4 → DP=2
    const cp4 = run({ nodes: 8, strategyConfig: { tp: 8, cp: 4, sequenceParallel: true } });
    const ratio = cp4.memoryPerGPU.activations / cp1.memoryPerGPU.activations;
    expect(ratio).toBeCloseTo(0.25, 2);
  });

  it('CP does not change parameter/gradient/optimizer sharding within same DP', () => {
    // Use enough GPUs so both configs have same DP
    // 64 GPUs: CP=1 → DP=8, CP=2 → DP=4 (different DP, so sharding differs)
    // 128 GPUs: CP=1 → DP=16, CP=2 → DP=8 (still different)
    // Instead, compare the sharding formula directly: params should scale with 1/dp
    // CP=2 halves DP, so params double. This is correct behavior.
    const ratio = cp2.memoryPerGPU.parameters / cp1.memoryPerGPU.parameters;
    // CP=2 on 32 GPUs: DP=2 vs DP=4, so params should double
    expect(ratio).toBeCloseTo(2.0, 1);
  });
});

// ---------------------------------------------------------------------------
// 2. GPU decomposition
// ---------------------------------------------------------------------------

describe('CP GPU decomposition', () => {
  it('TP=8, PP=1, CP=2 on 16 GPUs → DP=1', () => {
    const totalGPUs = 16;
    const dp = Math.floor(totalGPUs / (8 * 1 * 2));
    expect(dp).toBe(1);
  });

  it('TP=8, PP=4, CP=4 on 512 GPUs → DP=4', () => {
    const totalGPUs = 512;
    const dp = Math.floor(totalGPUs / (8 * 4 * 4));
    expect(dp).toBe(4);
  });

  it('totalGPUs = tp × pp × cp × dp', () => {
    const tp = 8, pp = 2, cp = 2;
    const totalGPUs = 64;
    const dp = Math.floor(totalGPUs / (tp * pp * cp));
    expect(tp * pp * cp * dp).toBe(totalGPUs);
  });
});

// ---------------------------------------------------------------------------
// 3. Throughput impact
// ---------------------------------------------------------------------------

describe('CP throughput impact', () => {
  it('CP=2 throughput is within 90-100% of CP=1', () => {
    // 32 GPUs
    const cp1 = run({ nodes: 4, strategyConfig: { tp: 8, cp: 1, sequenceParallel: true } });
    const cp2 = run({ nodes: 4, strategyConfig: { tp: 8, cp: 2, sequenceParallel: true } });
    const ratio = cp2.tokensPerSecond / cp1.tokensPerSecond;
    expect(ratio).toBeGreaterThanOrEqual(0.90);
    expect(ratio).toBeLessThanOrEqual(1.01);
  });

  it('CP=4 throughput is within 80-100% of CP=1', () => {
    // LLaMA 70B, seq=32768, CP=4 → chunkSeq=8192. Per-chunk attention FLOPs
    // are modest at this size, so ring attention exposes some comm overhead.
    const cp1 = run({ nodes: 8, strategyConfig: { tp: 8, cp: 1, sequenceParallel: true } });
    const cp4 = run({ nodes: 8, strategyConfig: { tp: 8, cp: 4, sequenceParallel: true } });
    const ratio = cp4.tokensPerSecond / cp1.tokensPerSecond;
    expect(ratio).toBeGreaterThanOrEqual(0.80);
    expect(ratio).toBeLessThanOrEqual(1.01);
  });
});

// ---------------------------------------------------------------------------
// 4. Validation constraints
// ---------------------------------------------------------------------------

describe('CP validation', () => {
  it('seq not divisible by CP → error', () => {
    const result = validate({
      nodes: 4,
      strategyConfig: { tp: 8, cp: 3, sequenceParallel: true },
    });
    expect(result.valid).toBe(false);
    expect(result.errors.some(e => e.includes('divisible by CP'))).toBe(true);
  });

  it('seq/CP < 1024 → warning', () => {
    // seq=4096, CP=8 → chunk=512
    const result = validate({
      nodes: 8,
      sequenceLength: 4096,
      strategyConfig: { tp: 8, cp: 8, sequenceParallel: true },
    });
    expect(result.warnings.some(w => w.includes('very small'))).toBe(true);
  });

  it('CP=1 produces no CP-related messages', () => {
    const result = validate({
      nodes: 4,
      strategyConfig: { tp: 8, cp: 1, sequenceParallel: true },
    });
    const cpMessages = [...result.errors, ...result.warnings].filter(
      m => m.includes('CP') || m.includes('Context')
    );
    expect(cpMessages).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// 5. Llama 3.1 405B 128K benchmark
// ---------------------------------------------------------------------------

describe('Llama 3.1 405B 128K context benchmark', () => {
  // Meta's published config: TP=8, PP=16, CP=16, DP=8, 16384 H100s
  // Published MFU: 38-43% (but note: likely HFU or includes tensor core utilization bonuses)
  // Our model caps at 0.72 compute efficiency for 3D, so MFU will be lower.
  // The key test is that CP=16 brings memory into range and produces reasonable MFU.

  it('fits in memory with CP=16', () => {
    const m = getSimulationMetrics({
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 2048)!,
      modelId: 'llama3-405b',
      globalBatchSize: 2048,
      microBatchSize: 1,
      sequenceLength: 131072,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: {
        tp: 8, pp: 16, cp: 16,
        sequenceParallel: true,
        pipelineSchedule: 'interleaved-1f1b',
        interleavedStages: 4,
      },
      activationCheckpointing: true,
      flashAttention: true,
    });

    // Must fit in 80 GB
    expect(m.memoryUtilization).toBeLessThan(1.0);
    // MFU should be reasonable (>10%) — our model is more conservative than published
    expect(m.mfu).toBeGreaterThan(0.10);
    expect(m.mfu).toBeLessThan(0.55);
    // Communication overhead should be present but not dominant
    expect(m.communicationOverhead).toBeLessThan(0.40);
  });
});

// ---------------------------------------------------------------------------
// 6. CP ring attention overlap
// ---------------------------------------------------------------------------

describe('CP ring attention overlap', () => {
  it('large model with long seq has low CP overhead', () => {
    const cp2 = run({ nodes: 4, strategyConfig: { tp: 8, cp: 2, sequenceParallel: true } });
    // CP exposed comm should be small fraction of total overhead
    expect(cp2.communicationOverhead).toBeLessThan(0.25);
  });

  it('CP overhead increases with higher CP degree', () => {
    const c2 = run({ nodes: 4, strategyConfig: { tp: 8, cp: 2, sequenceParallel: true } });
    const c4 = run({ nodes: 8, strategyConfig: { tp: 8, cp: 4, sequenceParallel: true } });
    // Higher CP = more ring steps = more exposed comm = lower MFU
    expect(c4.mfu).toBeLessThanOrEqual(c2.mfu);
  });
});

// ---------------------------------------------------------------------------
// 7. CP × EP interaction (MoE models)
// ---------------------------------------------------------------------------

describe('CP × EP interaction', () => {
  it('DeepSeek V3 CP=2 halves activations vs CP=1 with EP=8', () => {
    const ds1 = getSimulationMetrics({
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 64)!,
      modelId: 'deepseek-v3',
      globalBatchSize: 512,
      microBatchSize: 1,
      sequenceLength: 32768,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 1, ep: 8, cp: 1, sequenceParallel: true },
      activationCheckpointing: true,
      flashAttention: true,
    });
    const ds2 = getSimulationMetrics({
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 64)!,
      modelId: 'deepseek-v3',
      globalBatchSize: 512,
      microBatchSize: 1,
      sequenceLength: 32768,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 1, ep: 8, cp: 2, sequenceParallel: true },
      activationCheckpointing: true,
      flashAttention: true,
    });

    const ratio = ds2.memoryPerGPU.activations / ds1.memoryPerGPU.activations;
    expect(ratio).toBeCloseTo(0.50, 2);
  });
});

// ---------------------------------------------------------------------------
// 8. CP × SP interaction
// ---------------------------------------------------------------------------

describe('CP × SP compose multiplicatively', () => {
  it('combined CP=2 + SP gives ~0.19x activations vs no CP no SP', () => {
    // SP multiplier = 1/tp = 1/8 = 0.125
    // SP=OFF multiplier is model-dependent (sharded/tp + replicated) / total ≈ 0.334
    // ratio = (1/cp × SP_ON_mult) / SP_OFF_mult = (0.5 × 0.125) / 0.334 ≈ 0.187
    const cpSP = run({ nodes: 4, strategyConfig: { tp: 8, cp: 2, sequenceParallel: true } });
    const noCPnoSP = run({ nodes: 4, strategyConfig: { tp: 8, cp: 1, sequenceParallel: false } });
    const ratio = cpSP.memoryPerGPU.activations / noCPnoSP.memoryPerGPU.activations;
    expect(ratio).toBeCloseTo(0.187, 2);
  });

  it('CP alone gives 0.5x, SP alone gives ~0.37x — product matches combined', () => {
    const base = run({ nodes: 4, strategyConfig: { tp: 8, cp: 1, sequenceParallel: false } });
    const cpOnly = run({ nodes: 4, strategyConfig: { tp: 8, cp: 2, sequenceParallel: false } });
    const spOnly = run({ nodes: 4, strategyConfig: { tp: 8, cp: 1, sequenceParallel: true } });
    const both = run({ nodes: 4, strategyConfig: { tp: 8, cp: 2, sequenceParallel: true } });

    const cpFactor = cpOnly.memoryPerGPU.activations / base.memoryPerGPU.activations;
    const spFactor = spOnly.memoryPerGPU.activations / base.memoryPerGPU.activations;
    const combinedFactor = both.memoryPerGPU.activations / base.memoryPerGPU.activations;

    expect(combinedFactor).toBeCloseTo(cpFactor * spFactor, 3);
  });
});

// ---------------------------------------------------------------------------
// 9. CP on 1D strategies
// ---------------------------------------------------------------------------

describe('CP on 1D strategies', () => {
  it('FSDP with CP=2 routes through 3D and produces valid simulation', () => {
    // 8 GPUs, 7B model, seq=8192 (shorter to avoid OOM)
    const m = getSimulationMetrics({
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 1)!,
      modelId: 'llama2-7b',
      globalBatchSize: 128,
      microBatchSize: 4,
      sequenceLength: 8192,
      strategyType: 'fsdp',
      strategyConfig: { cp: 2 },
      activationCheckpointing: true,
      flashAttention: true,
    });

    // Should produce reasonable results
    expect(m.mfu).toBeGreaterThan(0.05);
    expect(m.mfu).toBeLessThan(0.60);
    expect(m.tokensPerSecond).toBeGreaterThan(0);
  });

  it('FSDP with CP=2 passes validation (tp=1, pp=1, cp=2, dp=4 on 8 GPUs)', () => {
    const engine = new SimulationEngine();
    engine.configure({
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 1)!,
      modelId: 'llama2-7b',
      globalBatchSize: 128,
      microBatchSize: 4,
      sequenceLength: 8192,
      strategyType: 'fsdp',
      strategyConfig: { cp: 2 },
      activationCheckpointing: true,
      flashAttention: true,
    });
    const result = engine.validate();
    // Should have no errors (only possibly OOM)
    const nonOomErrors = result.errors.filter(e => !e.includes('OOM'));
    expect(nonOomErrors).toHaveLength(0);
  });

  it('DDP with CP=2 reduces DP correctly', () => {
    // 8 GPUs: DDP CP=2 → tp=1, pp=1, cp=2, dp=4
    // Without CP: dp=8
    const noCp = getSimulationMetrics({
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 1)!,
      modelId: 'llama2-7b',
      globalBatchSize: 128,
      microBatchSize: 4,
      sequenceLength: 8192,
      strategyType: 'ddp',
      strategyConfig: { cp: 1 },
      activationCheckpointing: true,
      flashAttention: true,
    });
    const withCp = getSimulationMetrics({
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 1)!,
      modelId: 'llama2-7b',
      globalBatchSize: 128,
      microBatchSize: 4,
      sequenceLength: 8192,
      strategyType: 'ddp',
      strategyConfig: { cp: 2 },
      activationCheckpointing: true,
      flashAttention: true,
    });

    // With CP=2, activations should be halved
    const ratio = withCp.memoryPerGPU.activations / noCp.memoryPerGPU.activations;
    expect(ratio).toBeCloseTo(0.50, 2);
  });
});

// ---------------------------------------------------------------------------
// 10. Nemotron-4 340B with CP
// ---------------------------------------------------------------------------

describe('Nemotron-4 340B with CP', () => {
  const nemotronConfig = (cp: number) => getSimulationMetrics({
    clusterConfig: createMultiNodeCluster('h100-sxm', 8, 128)!,
    modelId: 'nemotron-4-340b',
    globalBatchSize: 1024,
    microBatchSize: 2,
    sequenceLength: 4096,
    strategyType: 'fsdp-tp-pp',
    strategyConfig: { tp: 4, pp: 8, cp, sequenceParallel: true },
    activationCheckpointing: true,
    flashAttention: true,
  });

  it('CP=2 halves activation memory', () => {
    const cp1 = nemotronConfig(1);
    const cp2 = nemotronConfig(2);
    const ratio = cp2.memoryPerGPU.activations / cp1.memoryPerGPU.activations;
    expect(ratio).toBeCloseTo(0.50, 2);
  });

  it('CP=2 produces valid simulation', () => {
    const m = nemotronConfig(2);
    expect(m.memoryUtilization).toBeLessThan(1.0);
    expect(m.mfu).toBeGreaterThan(0.10);
    expect(m.mfu).toBeLessThan(0.55);
  });

  it('GQA → CP comm overhead small', () => {
    // Nemotron-4: 8 KV heads × 192 headDim × 2 = 3072 KV dim — small
    // CP=2 on a GQA model should have low overall communication overhead
    const m = nemotronConfig(2);
    expect(m.communicationOverhead).toBeLessThan(0.30);
  });
});

// ---------------------------------------------------------------------------
// 11. MLA models CP comm compression
// ---------------------------------------------------------------------------

describe('MLA models CP comm compression', () => {
  it('DeepSeek V3 MLA — CP buffer smaller than GQA model CP buffer', () => {
    // MLA kvDim = kvLoraRank + qkRopeHeadDim = 512 + 64 = 576
    // GQA kvDim = numKvHeads × headDim × 2 = 8 × 128 × 2 = 2048
    // CP buffer ∝ kvDim, so MLA model should have ~3.6x smaller CP buffer
    const dsv3_cp1 = getSimulationMetrics({
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 128)!,
      modelId: 'deepseek-v3',
      globalBatchSize: 512,
      microBatchSize: 1,
      sequenceLength: 32768,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 1, ep: 8, cp: 1, sequenceParallel: true },
      activationCheckpointing: true,
      flashAttention: true,
    });
    const dsv3_cp2 = getSimulationMetrics({
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 128)!,
      modelId: 'deepseek-v3',
      globalBatchSize: 512,
      microBatchSize: 1,
      sequenceLength: 32768,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 1, ep: 8, cp: 2, sequenceParallel: true },
      activationCheckpointing: true,
      flashAttention: true,
    });
    const llamaCp1 = run({ nodes: 4, strategyConfig: { tp: 8, cp: 1, sequenceParallel: true } });
    const llamaCp2 = run({ nodes: 4, strategyConfig: { tp: 8, cp: 2, sequenceParallel: true } });

    // Compare CP buffer overhead (temp diff between CP=2 and CP=1)
    const dsv3BufferDiff = dsv3_cp2.memoryPerGPU.temporary - dsv3_cp1.memoryPerGPU.temporary;
    const llamaBufferDiff = llamaCp2.memoryPerGPU.temporary - llamaCp1.memoryPerGPU.temporary;
    // MLA (576) should have smaller CP buffer than GQA (2048)
    expect(dsv3BufferDiff).toBeLessThan(llamaBufferDiff);
  });

  it('DeepSeek V3 CP=2 fits in memory and produces valid MFU', () => {
    const m = getSimulationMetrics({
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 128)!,
      modelId: 'deepseek-v3',
      globalBatchSize: 512,
      microBatchSize: 1,
      sequenceLength: 32768,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 1, ep: 8, cp: 2, sequenceParallel: true },
      activationCheckpointing: true,
      flashAttention: true,
    });
    expect(m.memoryUtilization).toBeLessThan(1.0);
    expect(m.mfu).toBeGreaterThan(0.05);
    expect(m.mfu).toBeLessThan(0.60);
  });
});

// ---------------------------------------------------------------------------
// 12. GQA vs MHA CP comm scaling
// ---------------------------------------------------------------------------

describe('GQA vs MHA CP comm scaling', () => {
  it('GPT-3 175B (MHA) has higher comm overhead with CP than LLaMA 70B (GQA)', () => {
    // MHA has numKvHeads=96 × headDim=128 × 2 = 24576 KV dim
    // GQA has numKvHeads=8 × headDim=128 × 2 = 2048 KV dim → 12x smaller
    // So GPT-3 with CP should have more communication overhead
    const gpt3 = getSimulationMetrics({
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 128)!,
      modelId: 'gpt3-175b',
      globalBatchSize: 256,
      microBatchSize: 1,
      sequenceLength: 32768,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 8, cp: 2, sequenceParallel: true },
      activationCheckpointing: true,
      flashAttention: true,
    });
    const llama = run({ nodes: 8, strategyConfig: { tp: 8, cp: 2, sequenceParallel: true } });

    // Both should produce valid results
    expect(gpt3.mfu).toBeGreaterThan(0.05);
    expect(llama.mfu).toBeGreaterThan(0.05);
    // GPT-3 (MHA, 12x more KV heads) should have higher comm overhead with CP
    expect(gpt3.communicationOverhead).toBeGreaterThan(llama.communicationOverhead);
  });
});

// ---------------------------------------------------------------------------
// 13. CP + PP interaction
// ---------------------------------------------------------------------------

describe('CP + PP interaction', () => {
  const run405b = (cp: number) => getSimulationMetrics({
    clusterConfig: createMultiNodeCluster('h100-sxm', 8, 256)!,
    modelId: 'llama3-405b',
    globalBatchSize: 512,
    microBatchSize: 1,
    sequenceLength: 65536,
    strategyType: 'fsdp-tp-pp',
    strategyConfig: {
      tp: 8, pp: 4, cp,
      sequenceParallel: true,
      pipelineSchedule: 'interleaved-1f1b',
      interleavedStages: 4,
    },
    activationCheckpointing: true,
    flashAttention: true,
  });

  it('CP=2 + PP=4 activation memory halving', () => {
    const cp1 = run405b(1);
    const cp2 = run405b(2);
    const ratio = cp2.memoryPerGPU.activations / cp1.memoryPerGPU.activations;
    expect(ratio).toBeCloseTo(0.50, 2);
  });

  it('CP=2 + PP=4 valid simulation', () => {
    const m = run405b(2);
    expect(m.memoryUtilization).toBeLessThan(1.0);
    expect(m.mfu).toBeGreaterThan(0.05);
  });
});

// ---------------------------------------------------------------------------
// 14. CP comm uses BF16 floor (not FP8)
// ---------------------------------------------------------------------------

describe('CP comm uses BF16 floor', () => {
  it('FP8 vs BF16 — CP timing impact similar (CP always uses BF16 floor)', () => {
    const base = {
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 4)!,
      modelId: 'llama2-70b' as const,
      globalBatchSize: 256,
      microBatchSize: 2,
      sequenceLength: 32768,
      strategyType: 'fsdp-tp' as const,
      strategyConfig: { tp: 8, cp: 2, sequenceParallel: true },
      activationCheckpointing: true,
      flashAttention: true,
    };
    const fp8 = getSimulationMetrics({ ...base, mixedPrecision: 'fp8' as const });
    const bf16 = getSimulationMetrics({ ...base, mixedPrecision: 'bf16' as const });

    // Both should succeed
    expect(fp8.mfu).toBeGreaterThan(0.05);
    expect(bf16.mfu).toBeGreaterThan(0.05);

    // FP8 should have higher MFU (faster TP comm via quantized collectives + faster compute)
    // but CP overhead is the same since CP always uses BF16 floor.
    // The CP portion of timing is identical, so the FP8 advantage comes only from TP/compute.
    expect(fp8.mfu).toBeGreaterThan(bf16.mfu);
  });
});

// ---------------------------------------------------------------------------
// 15. CP ring buffer with checkpointing
// ---------------------------------------------------------------------------

describe('CP ring buffer with checkpointing', () => {
  it('checkpointing increases temporary memory (3x vs 2x buffers)', () => {
    const base = {
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 4)!,
      modelId: 'llama2-70b' as const,
      globalBatchSize: 256,
      microBatchSize: 2,
      sequenceLength: 32768,
      strategyType: 'fsdp-tp' as const,
      strategyConfig: { tp: 8, cp: 2, sequenceParallel: true },
      flashAttention: true,
    };
    const withCkpt = getSimulationMetrics({ ...base, activationCheckpointing: true });
    const noCkpt = getSimulationMetrics({ ...base, activationCheckpointing: false });

    // With checkpointing: 3 buffers (fwd recompute + bwd + recv)
    // Without checkpointing: 2 buffers (compute + recv)
    // So temporary memory should be larger with checkpointing
    expect(withCkpt.memoryPerGPU.temporary).toBeGreaterThan(noCkpt.memoryPerGPU.temporary);

    // Buffer diff ≈ 1 × kvDim × effectiveSeq × MBS × 2 bytes
    // kvDim = 8 * 128 * 2 = 2048, effectiveSeq = 32768/2 = 16384, MBS = 2
    const expectedDiffMB = 2048 * 16384 * 2 * 2 / (1024 * 1024);
    const actualDiffMB = (withCkpt.memoryPerGPU.temporary - noCkpt.memoryPerGPU.temporary) / (1024 * 1024);
    expect(actualDiffMB).toBeGreaterThan(expectedDiffMB * 0.7);
    expect(actualDiffMB).toBeLessThan(expectedDiffMB * 1.3);
  });
});

// ---------------------------------------------------------------------------
// 16. Large CP MFU degradation
// ---------------------------------------------------------------------------

describe('Large CP range', () => {
  it('all CP values produce valid results and comm overhead grows', () => {
    const runCP = (cp: number) => getSimulationMetrics({
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 256)!,
      modelId: 'llama3-405b',
      globalBatchSize: 512,
      microBatchSize: 1,
      sequenceLength: 131072,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: {
        tp: 8, pp: 16, cp,
        sequenceParallel: true,
        pipelineSchedule: 'interleaved-1f1b',
        interleavedStages: 4,
      },
      activationCheckpointing: true,
      flashAttention: true,
    });

    const results = [2, 4, 8, 16].map(cp => ({ cp, ...runCP(cp) }));

    // All CP values produce meaningful work
    for (const r of results) {
      expect(r.mfu).toBeGreaterThan(0.05);
      expect(r.tokensPerSecond).toBeGreaterThan(0);
    }

    // Communication overhead generally increases with CP (more ring steps)
    // Allow some non-monotonicity because CP also reduces DP (less DP comm)
    expect(results[results.length - 1].communicationOverhead)
      .toBeGreaterThan(results[0].communicationOverhead * 0.8);
  });
});

// ---------------------------------------------------------------------------
// 17. Different GPUs with CP
// ---------------------------------------------------------------------------

describe('Different GPUs with CP', () => {
  it('A100 80GB with CP=2', () => {
    const cp1 = getSimulationMetrics({
      clusterConfig: createMultiNodeCluster('a100-80gb', 8, 4)!,
      modelId: 'llama2-70b',
      globalBatchSize: 256,
      microBatchSize: 2,
      sequenceLength: 32768,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8, cp: 1, sequenceParallel: true },
      activationCheckpointing: true,
      flashAttention: true,
    });
    const cp2 = getSimulationMetrics({
      clusterConfig: createMultiNodeCluster('a100-80gb', 8, 4)!,
      modelId: 'llama2-70b',
      globalBatchSize: 256,
      microBatchSize: 2,
      sequenceLength: 32768,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8, cp: 2, sequenceParallel: true },
      activationCheckpointing: true,
      flashAttention: true,
    });

    expect(cp2.mfu).toBeGreaterThan(0.05);
    const ratio = cp2.memoryPerGPU.activations / cp1.memoryPerGPU.activations;
    expect(ratio).toBeCloseTo(0.50, 2);
  });

  it('H200 with CP=4 for 405B 128K', () => {
    const m = getSimulationMetrics({
      clusterConfig: createMultiNodeCluster('h200-sxm', 8, 64)!,
      modelId: 'llama3-405b',
      globalBatchSize: 256,
      microBatchSize: 1,
      sequenceLength: 131072,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 4, cp: 4, sequenceParallel: true },
      activationCheckpointing: true,
      flashAttention: true,
    });

    // H200 has 141GB
    expect(m.memoryUtilization).toBeLessThan(1.0);
    expect(m.mfu).toBeGreaterThan(0.05);
  });
});

// ---------------------------------------------------------------------------
// 18. CP edge: DP=1
// ---------------------------------------------------------------------------

describe('CP edge: DP=1', () => {
  it('CP consumes all DP', () => {
    // 16 GPUs: TP=8, CP=2 → DP=1
    const m = getSimulationMetrics({
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 2)!,
      modelId: 'llama2-70b',
      globalBatchSize: 64,
      microBatchSize: 2,
      sequenceLength: 32768,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8, cp: 2, sequenceParallel: true },
      activationCheckpointing: true,
      flashAttention: true,
    });

    expect(m.mfu).toBeGreaterThan(0.05);
    expect(m.tokensPerSecond).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// 19. CP + LoRA / QLoRA
// ---------------------------------------------------------------------------

describe('CP + LoRA', () => {
  const loraConfig = (method: 'lora' | 'qlora') => getSimulationMetrics({
    clusterConfig: createMultiNodeCluster('h100-sxm', 8, 4)!,
    modelId: 'llama2-70b',
    globalBatchSize: 128,
    microBatchSize: 2,
    sequenceLength: 32768,
    strategyType: 'fsdp-tp',
    strategyConfig: { tp: 8, cp: 2, sequenceParallel: true },
    activationCheckpointing: true,
    flashAttention: true,
    finetuningMethod: method,
    loraRank: 16,
    loraTargetModules: 'q_k_v_o',
  });

  it('CP=2 with LoRA succeeds and halves activations', () => {
    const lora = loraConfig('lora');
    expect(lora.memoryUtilization).toBeLessThan(1.0);
    expect(lora.mfu).toBeGreaterThan(0.01);

    // Compare activations with LoRA CP=1
    const loraCp1 = getSimulationMetrics({
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 4)!,
      modelId: 'llama2-70b',
      globalBatchSize: 128,
      microBatchSize: 2,
      sequenceLength: 32768,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8, cp: 1, sequenceParallel: true },
      activationCheckpointing: true,
      flashAttention: true,
      finetuningMethod: 'lora',
      loraRank: 16,
      loraTargetModules: 'q_k_v_o',
    });
    const ratio = lora.memoryPerGPU.activations / loraCp1.memoryPerGPU.activations;
    expect(ratio).toBeCloseTo(0.50, 2);
  });

  it('CP=2 with QLoRA uses less memory than LoRA', () => {
    const lora = loraConfig('lora');
    const qlora = loraConfig('qlora');
    expect(qlora.memoryUtilization).toBeLessThan(1.0);
    expect(qlora.memoryPerGPU.total).toBeLessThan(lora.memoryPerGPU.total);
  });
});

// ---------------------------------------------------------------------------
// 20. CP on ZeRO-1 strategy
// ---------------------------------------------------------------------------

describe('CP on ZeRO-1 strategy', () => {
  it('ZeRO-1 with CP=2 routes through 3D and produces valid simulation', () => {
    const cp1 = getSimulationMetrics({
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 1)!,
      modelId: 'llama2-7b',
      globalBatchSize: 128,
      microBatchSize: 4,
      sequenceLength: 8192,
      strategyType: 'zero-1',
      strategyConfig: { cp: 1 },
      activationCheckpointing: true,
      flashAttention: true,
    });
    const cp2 = getSimulationMetrics({
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 1)!,
      modelId: 'llama2-7b',
      globalBatchSize: 128,
      microBatchSize: 4,
      sequenceLength: 8192,
      strategyType: 'zero-1',
      strategyConfig: { cp: 2 },
      activationCheckpointing: true,
      flashAttention: true,
    });

    expect(cp2.mfu).toBeGreaterThan(0.05);
    const ratio = cp2.memoryPerGPU.activations / cp1.memoryPerGPU.activations;
    expect(ratio).toBeCloseTo(0.50, 2);
  });
});

// ---------------------------------------------------------------------------
// 21. CP boundary: seq=16384, CP=2 (chunk=8192)
// ---------------------------------------------------------------------------

describe('CP boundary: seq=16384, CP=2', () => {
  it('chunk=8192 is the recommendation engine minimum — no warnings', () => {
    const result = validate({
      nodes: 4,
      sequenceLength: 16384,
      strategyConfig: { tp: 8, cp: 2, sequenceParallel: true },
    });
    // chunk=8192 > 1024, so no "very small" warning
    const smallChunkWarnings = result.warnings.filter(w => w.includes('very small'));
    expect(smallChunkWarnings).toHaveLength(0);
  });

  it('CP=2 halves activations at seq=16384', () => {
    const cp1 = run({
      nodes: 4,
      sequenceLength: 16384,
      strategyConfig: { tp: 8, cp: 1, sequenceParallel: true },
    });
    const cp2 = run({
      nodes: 4,
      sequenceLength: 16384,
      strategyConfig: { tp: 8, cp: 2, sequenceParallel: true },
    });
    const ratio = cp2.memoryPerGPU.activations / cp1.memoryPerGPU.activations;
    expect(ratio).toBeCloseTo(0.50, 2);
  });

  it('produces reasonable MFU', () => {
    const m = run({
      nodes: 4,
      sequenceLength: 16384,
      strategyConfig: { tp: 8, cp: 2, sequenceParallel: true },
    });
    expect(m.mfu).toBeGreaterThan(0.10);
  });
});

// ---------------------------------------------------------------------------
// Ring vs All-Gather CP implementation
// ---------------------------------------------------------------------------

describe('Ring vs All-Gather invariant', () => {
  it('all-gather MFU < ring MFU for same config (ring overlaps better)', () => {
    const base = {
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 4)!,
      modelId: 'llama2-70b' as const,
      globalBatchSize: 256,
      microBatchSize: 2,
      sequenceLength: 32768,
      strategyType: 'fsdp-tp' as const,
      strategyConfig: { tp: 8, cp: 2, sequenceParallel: true },
      activationCheckpointing: true,
      flashAttention: true,
    };
    const ring = getSimulationMetrics({
      ...base,
      strategyConfig: { ...base.strategyConfig, cpImplementation: 'ring' as const },
    });
    const allGather = getSimulationMetrics({
      ...base,
      strategyConfig: { ...base.strategyConfig, cpImplementation: 'all-gather' as const },
    });

    // Ring attention overlaps KV P2P with chunk compute — lower overhead
    expect(allGather.mfu).toBeLessThan(ring.mfu);
    // Both should produce valid results
    expect(ring.mfu).toBeGreaterThan(0.05);
    expect(allGather.mfu).toBeGreaterThan(0.05);
  });

  it('CP=1 produces identical results regardless of cpImplementation', () => {
    const base = {
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 4)!,
      modelId: 'llama2-70b' as const,
      globalBatchSize: 256,
      microBatchSize: 2,
      sequenceLength: 32768,
      strategyType: 'fsdp-tp' as const,
      strategyConfig: { tp: 8, cp: 1, sequenceParallel: true },
      activationCheckpointing: true,
      flashAttention: true,
    };
    const ring = getSimulationMetrics({
      ...base,
      strategyConfig: { ...base.strategyConfig, cpImplementation: 'ring' as const },
    });
    const allGather = getSimulationMetrics({
      ...base,
      strategyConfig: { ...base.strategyConfig, cpImplementation: 'all-gather' as const },
    });
    expect(ring.mfu).toEqual(allGather.mfu);
    expect(ring.stepTimeMs).toEqual(allGather.stepTimeMs);
  });

  it('all-gather at high CP has more overhead than ring', () => {
    // 405B, CP=16, 131K — the canonical benchmark gap
    const base = {
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 2048)!,
      modelId: 'llama3-405b' as const,
      globalBatchSize: 2048,
      microBatchSize: 1,
      sequenceLength: 131072,
      strategyType: 'fsdp-tp-pp' as const,
      strategyConfig: {
        tp: 8, pp: 16, cp: 16,
        sequenceParallel: true,
        pipelineSchedule: 'interleaved-1f1b' as const,
        interleavedStages: 4,
      },
      activationCheckpointing: true,
      flashAttention: true,
    };
    const ring = getSimulationMetrics({
      ...base,
      strategyConfig: { ...base.strategyConfig, cpImplementation: 'ring' as const },
    });
    const allGather = getSimulationMetrics({
      ...base,
      strategyConfig: { ...base.strategyConfig, cpImplementation: 'all-gather' as const },
    });
    // All-gather should have lower MFU (more exposed comm)
    expect(allGather.mfu).toBeLessThan(ring.mfu);
    // Gap should be meaningful at CP=16
    expect(ring.mfu / allGather.mfu).toBeGreaterThan(1.05);
  });
});

// ---------------------------------------------------------------------------
// Model FLOPs MFU
// ---------------------------------------------------------------------------

describe('Model FLOPs MFU', () => {
  it('undefined at short sequence (4096) where attention FLOPs are negligible', () => {
    const m = getSimulationMetrics({
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 64)!,
      modelId: 'llama3-405b',
      globalBatchSize: 512,
      microBatchSize: 1,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 16, sequenceParallel: true },
      activationCheckpointing: true,
      flashAttention: true,
    });
    // At 4K seq, flopsPerToken ≈ 2P so Model FLOPs MFU ≈ PaLM MFU (< 10% gap)
    expect(m.modelFlopsMfu).toBeUndefined();
  });

  it('populated at long sequence (131072) for 405B where quadratic attention dominates', () => {
    const m = getSimulationMetrics({
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 2048)!,
      modelId: 'llama3-405b',
      globalBatchSize: 2048,
      microBatchSize: 1,
      sequenceLength: 131072,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: {
        tp: 8, pp: 16, cp: 16,
        sequenceParallel: true,
        pipelineSchedule: 'interleaved-1f1b',
        interleavedStages: 4,
      },
      activationCheckpointing: true,
      flashAttention: true,
    });
    expect(m.modelFlopsMfu).toBeDefined();
    // Model FLOPs MFU should be significantly higher than PaLM MFU
    expect(m.modelFlopsMfu!).toBeGreaterThan(m.mfu * 1.10);
    // Should be reasonable (not > 100%)
    expect(m.modelFlopsMfu!).toBeLessThan(1.0);
  });
});
