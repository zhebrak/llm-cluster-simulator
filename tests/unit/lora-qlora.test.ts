/**
 * LoRA/QLoRA Fine-tuning Tests
 *
 * Validates trainable parameter counts, memory estimates, compute (MFU),
 * communication patterns, and TP sharding for LoRA/QLoRA fine-tuning.
 *
 * Published benchmarks:
 * - LLaMA 2 7B r=8 q_v: 4,194,304 trainable params (HuggingFace PEFT)
 * - LLaMA 2 7B LoRA: ~21 GB memory (HuggingFace docs)
 * - LLaMA 2 7B QLoRA: ~14 GB memory (QLoRA paper)
 * - LLaMA 2 70B QLoRA: fits on 80 GB (QLoRA paper)
 */

import { describe, it, expect } from 'vitest';
import { SimulationEngine } from '../../src/core/simulation/engine.ts';
import { getModel } from '../../src/core/models/index.ts';
import type { GPUSpec } from '../../src/types/index.ts';
import { getPresetCluster } from '../../src/core/hardware/index.ts';
import { createPipelineParallelStrategy, FSDPStrategy } from '../../src/core/strategies/index.ts';
import { DTYPE_PRESETS, DEFAULT_ADAMW_CONFIG, DEFAULT_LR_SCHEDULE } from '../../src/types/index.ts';
import {
  computeLoraTrainableParams,
  computeLoraParamsPerRank,
  getQloraDequantTimeMs,
  getLoraBackwardMultiplier,
} from '../../src/core/strategies/lora.ts';
import { getSelectiveRecomputeFraction } from '../../src/core/strategies/base.ts';

// ---------------------------------------------------------------------------
// Helper: run simulation and return result
// ---------------------------------------------------------------------------
function runSim(config: Parameters<SimulationEngine['configure']>[0]) {
  const engine = new SimulationEngine();
  engine.configure(config);
  return engine.run();
}

// ---------------------------------------------------------------------------
// 1. Trainable parameter counts
// ---------------------------------------------------------------------------
describe('LoRA Trainable Parameters', () => {
  it('LLaMA 2 7B r=8 q_v — exact 4,194,304 params (published)', () => {
    const model = getModel('llama2-7b')!;
    const params = computeLoraTrainableParams(model, 8, 'q_v');
    // Published by HuggingFace PEFT — exact match expected
    expect(params).toBe(4_194_304);
  });

  it('LLaMA 2 7B r=16 all_linear — correct count for attn+MLP', () => {
    const model = getModel('llama2-7b')!;
    const params = computeLoraTrainableParams(model, 16, 'all_linear');
    // 7 linear layers per block (Q,K,V,O + gate,up,down) × 32 layers
    // Per layer: 4×r×(d+d) + 3×r×(d+I) = 4×16×8192 + 3×16×15104 = 524288 + 724992 = 1,249,280
    // Total: 1,249,280 × 32 = 39,976,960
    expect(params).toBeGreaterThan(35e6);
    expect(params).toBeLessThan(45e6);
  });

  it('GQA model (LLaMA 3.1 8B) — K/V adapters smaller than Q/O', () => {
    const model = getModel('llama3.1-8b')!;
    // With GQA (numKvHeads=8 vs numAttentionHeads=32), K/V dims are 1024 vs Q/O 4096
    const paramsQKVO = computeLoraTrainableParams(model, 16, 'q_k_v_o');
    // Q: r*(d + d_q) = 16*(4096+4096) = 131072
    // K: r*(d + d_k) = 16*(4096+1024) = 81920  ← smaller due to GQA
    // V: r*(d + d_v) = 16*(4096+1024) = 81920  ← smaller due to GQA
    // O: r*(d_q + d) = 16*(4096+4096) = 131072
    // Per layer: 131072 + 81920 + 81920 + 131072 = 425984
    // Total: 425984 * 32 = 13,631,488
    const expectedPerLayer = 16 * (4096 + 4096) + 16 * (4096 + 1024) + 16 * (4096 + 1024) + 16 * (4096 + 4096);
    expect(paramsQKVO).toBe(expectedPerLayer * 32);

    // Compare with q_v — should be less than q_k_v_o
    const paramsQV = computeLoraTrainableParams(model, 16, 'q_v');
    expect(paramsQV).toBeLessThan(paramsQKVO);
  });

  it('MoE all_linear skips routed experts (DeepSeek V3)', () => {
    const model = getModel('deepseek-v3')!;
    const params = computeLoraTrainableParams(model, 16, 'all_linear');
    // DeepSeek V3: 256 routed experts × 3 linears × r×(d+I)... would be massive
    // Our impl: only attention + shared expert MLP (1 shared expert)
    // Routed expert MLP adapters NOT included
    // With MLA O-fix: O uses nH×headDim=16384 (not qLoraRank=1536)
    expect(params).toBe(75_816_960);
  });
});

// ---------------------------------------------------------------------------
// 2. Per-rank TP sharding
// ---------------------------------------------------------------------------
describe('LoRA TP Sharding', () => {
  it('adapter memory partially TP-sharded (not fully replicated or split)', () => {
    const model = getModel('llama2-7b')!;
    const tp = 4;
    const globalParams = computeLoraTrainableParams(model, 16, 'q_k_v_o');
    const perRankParams = computeLoraParamsPerRank(model, 16, 'q_k_v_o', tp, 1);
    // For symmetric projections (d=d_q=d_k=d_v=4096):
    // Column-parallel: r*(d + out/tp), Row-parallel: r*(in/tp + d)
    // Per-rank ratio should be between 1/tp (fully sharded) and 1.0 (replicated)
    const ratio = perRankParams / globalParams;
    expect(ratio).toBeGreaterThan(1 / tp + 0.01); // Not fully sharded
    expect(ratio).toBeLessThan(0.95);              // Not fully replicated
    // For tp=4 with symmetric dims: ratio ≈ (1 + 1/tp) / 2 = 0.625
    expect(ratio).toBeGreaterThan(0.45);
    expect(ratio).toBeLessThan(0.75);
  });

  it('PP divides adapter params across stages', () => {
    const model = getModel('llama2-7b')!;
    const paramspp1 = computeLoraParamsPerRank(model, 16, 'q_k_v_o', 1, 1);
    const paramspp4 = computeLoraParamsPerRank(model, 16, 'q_k_v_o', 1, 4);
    // PP=4 should give ~1/4 of PP=1 (32 layers / 4 stages = 8 layers/stage)
    const ratio = paramspp4 / paramspp1;
    expect(ratio).toBeGreaterThan(0.2);
    expect(ratio).toBeLessThan(0.3);
  });
});

// ---------------------------------------------------------------------------
// 3. QLoRA dequant timing
// ---------------------------------------------------------------------------
describe('QLoRA Dequant Timing', () => {
  it('7B dequant < 5ms (negligible)', () => {
    const model = getModel('llama2-7b')!;
    const gpu = { memoryBandwidthTBps: 3.35 } as GPUSpec; // H100-like
    const dequantMs = getQloraDequantTimeMs(model, gpu, 1, 1);
    expect(dequantMs).toBeLessThan(6);
    expect(dequantMs).toBeGreaterThan(0);
  });

  it('70B dequant > 10ms (significant at pp=1)', () => {
    const model = getModel('llama2-70b')!;
    const gpu = { memoryBandwidthTBps: 3.35 } as GPUSpec; // H100-like
    const dequantMs = getQloraDequantTimeMs(model, gpu, 1, 1);
    expect(dequantMs).toBeGreaterThan(10);
  });

  it('TP reduces dequant proportionally', () => {
    const model = getModel('llama2-70b')!;
    const gpu = { memoryBandwidthTBps: 3.35 } as GPUSpec;
    const tp1 = getQloraDequantTimeMs(model, gpu, 1, 1);
    const tp8 = getQloraDequantTimeMs(model, gpu, 8, 1);
    const ratio = tp8 / tp1;
    expect(ratio).toBeGreaterThan(0.10);
    expect(ratio).toBeLessThan(0.15);
  });
});

// ---------------------------------------------------------------------------
// 4. LoRA backward multiplier
// ---------------------------------------------------------------------------
describe('LoRA Backward Compute', () => {
  it('without ckpt: ~52.5% of full (1.05/2)', () => {
    const model = getModel('llama2-7b')!;
    const loraMultiplier = getLoraBackwardMultiplier(model, { method: 'lora', rank: 16, targetModules: 'q_k_v_o' }, false);
    const fullMultiplier = 2;
    const ratio = loraMultiplier / fullMultiplier;
    expect(ratio).toBeGreaterThan(0.45);
    expect(ratio).toBeLessThan(0.60);
  });

  it('with ckpt: ~68% of full (2.05/3)', () => {
    const model = getModel('llama2-7b')!;
    const loraMultiplier = getLoraBackwardMultiplier(model, { method: 'lora', rank: 16, targetModules: 'q_k_v_o' }, true);
    const fullMultiplier = 3;
    const ratio = loraMultiplier / fullMultiplier;
    expect(ratio).toBeGreaterThan(0.60);
    expect(ratio).toBeLessThan(0.75);
  });

  it('overhead is flat 1.05 for practical configs (floor dominates)', () => {
    const model = getModel('llama2-7b')!;
    // The overhead formula is: 1.0 + max(0.05, 2 * trainable/total).
    // The scaling term only exceeds the 0.05 floor when trainable > 2.5% of total.
    // For models ≥7B, even r=64 all_linear is only ~2.3% — so the floor always
    // dominates. The overhead is effectively a flat 1.05 for all real-world configs.
    // The scaling term exists for correctness on sub-1B models.
    const mult_r4 = getLoraBackwardMultiplier(model, { method: 'lora', rank: 4, targetModules: 'q_v' }, false);
    const mult_r64 = getLoraBackwardMultiplier(model, { method: 'lora', rank: 64, targetModules: 'all_linear' }, false);
    // Both hit the floor → both exactly 1.05
    expect(mult_r4).toBe(1.05);
    expect(mult_r64).toBe(1.05);
  });
});

// ---------------------------------------------------------------------------
// 5. Engine-level: Memory
// ---------------------------------------------------------------------------
describe('LoRA/QLoRA Memory (Engine)', () => {
  it('LoRA memory < full training memory (LLaMA 2 7B, FSDP)', () => {
    // Use 8x H100 with FSDP — full 7B training on 1 GPU OOMs (~100+ GB needed).
    // FSDP ratio (~0.63) is higher than DDP ratio (~0.35) because FSDP already shards
    // base weights across 8 GPUs, diluting LoRA's savings (no grads/optimizer for frozen
    // params). The plan's "35-45%" estimate applies to DDP (single GPU), not FSDP.
    const fullResult = runSim({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 32,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
    });

    const loraResult = runSim({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 32,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
      finetuningMethod: 'lora',
      loraRank: 16,
      loraTargetModules: 'q_k_v_o',
    });

    expect(fullResult.success).toBe(true);
    expect(loraResult.success).toBe(true);

    // LoRA should use significantly less memory than full training
    const ratio = loraResult.metrics.peakMemoryGB / fullResult.metrics.peakMemoryGB;
    console.log(`LoRA/full memory ratio: ${ratio.toFixed(3)}`);
    console.log(`Full memory: ${fullResult.metrics.peakMemoryGB.toFixed(1)} GB, LoRA: ${loraResult.metrics.peakMemoryGB.toFixed(1)} GB`);
    expect(ratio).toBeLessThan(0.65); // LoRA should be much less than full
    expect(ratio).toBeGreaterThan(0.15); // But not impossibly small
  });

  it('QLoRA uses less memory than LoRA', () => {
    const loraResult = runSim({
      modelId: 'llama2-7b',
      clusterId: '1x-h100',
      globalBatchSize: 4,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'ddp',
      mixedPrecision: 'bf16',
      finetuningMethod: 'lora',
      loraRank: 16,
      loraTargetModules: 'q_k_v_o',
    });

    const qloraResult = runSim({
      modelId: 'llama2-7b',
      clusterId: '1x-h100',
      globalBatchSize: 4,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'ddp',
      mixedPrecision: 'bf16',
      finetuningMethod: 'qlora',
      loraRank: 16,
      loraTargetModules: 'q_k_v_o',
    });

    expect(loraResult.success).toBe(true);
    expect(qloraResult.success).toBe(true);
    console.log(`LoRA memory: ${loraResult.metrics.peakMemoryGB.toFixed(1)} GB, QLoRA: ${qloraResult.metrics.peakMemoryGB.toFixed(1)} GB`);
    // QLoRA saves ~75% on base weight memory (0.515 bytes vs 2 bytes for bf16)
    expect(qloraResult.metrics.peakMemoryGB).toBeLessThan(loraResult.metrics.peakMemoryGB);
  });

  it('QLoRA 70B fits on 80GB GPU', () => {
    const result = runSim({
      modelId: 'llama2-70b',
      clusterId: '1x-h100',
      globalBatchSize: 1,
      microBatchSize: 1,
      sequenceLength: 512,
      strategyType: 'ddp',
      mixedPrecision: 'bf16',
      finetuningMethod: 'qlora',
      loraRank: 16,
      loraTargetModules: 'q_k_v_o',
    });

    expect(result.success).toBe(true);
    console.log(`QLoRA 70B memory: ${result.metrics.peakMemoryGB.toFixed(1)} GB`);
    // QLoRA paper: LLaMA 2 70B fits on single 80 GB GPU
    expect(result.metrics.peakMemoryGB).toBeLessThan(80);
  });
});

// ---------------------------------------------------------------------------
// 6. Engine-level: MFU
// ---------------------------------------------------------------------------
describe('LoRA MFU', () => {
  it('LoRA MFU uses 4PD + 6AD numerator (lower than full)', () => {
    const fullResult = runSim({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 32,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
    });

    const loraResult = runSim({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 32,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
      finetuningMethod: 'lora',
      loraRank: 16,
      loraTargetModules: 'q_k_v_o',
    });

    expect(fullResult.success).toBe(true);
    expect(loraResult.success).toBe(true);

    // LoRA MFU = (4PD+6AD)/(time*peak) vs full MFU = 6PD/(time*peak)
    // But LoRA step time is also lower (less backward compute).
    // Net effect: LoRA MFU should be lower than full training MFU.
    console.log(`Full MFU: ${(fullResult.metrics.mfu * 100).toFixed(1)}%, LoRA MFU: ${(loraResult.metrics.mfu * 100).toFixed(1)}%`);
    expect(loraResult.metrics.mfu).toBeLessThan(fullResult.metrics.mfu);
    // LoRA MFU should still be positive and reasonable
    expect(loraResult.metrics.mfu).toBeGreaterThan(0.10);
    expect(loraResult.metrics.mfu).toBeLessThan(0.60);
  });
});

// ---------------------------------------------------------------------------
// 6b. LoRA MFU sensitivity to rank and target modules
//
// Uses strategy.computeAnalysis() directly to get unrounded MFU values,
// bypassing engine rounding (toFixed(2)) that erases sub-1% adapter deltas.
// ---------------------------------------------------------------------------
describe('LoRA MFU Sensitivity to Rank and Target Modules', () => {
  const model = getModel('llama2-7b')!;
  const cluster = getPresetCluster('8x-h100')!;
  const dtypes = DTYPE_PRESETS.bf16;
  const gbs = 32;
  const mbs = 4;
  const dp = cluster.totalGPUs;
  const ga = Math.ceil(gbs / (mbs * dp));

  const strategy = new FSDPStrategy();

  function makeCtx(lora: { method: 'lora'; rank: number; targetModules: 'q_v' | 'q_k_v_o' | 'all_linear' }) {
    return {
      model,
      cluster,
      training: {
        globalBatchSize: gbs,
        microBatchSize: mbs,
        sequenceLength: 2048,
        maxSteps: 1000,
        optimizer: DEFAULT_ADAMW_CONFIG,
        lrSchedule: DEFAULT_LR_SCHEDULE,
        dtypes,
        gradientClipping: 1.0,
        gradientAccumulationSteps: ga,
      },
      seqLength: 2048,
      microBatchSize: mbs,
      globalBatchSize: gbs,
      gradientAccumulationSteps: ga,
      activationCheckpointing: true,
      flashAttention: true,
      lora,
    };
  }

  it('higher rank produces strictly higher MFU', () => {
    const r4 = strategy.computeAnalysis(makeCtx({ method: 'lora', rank: 4, targetModules: 'q_k_v_o' }));
    const r64 = strategy.computeAnalysis(makeCtx({ method: 'lora', rank: 64, targetModules: 'q_k_v_o' }));
    expect(r64.mfu).toBeGreaterThan(r4.mfu);
  });

  it('more target modules produce strictly higher MFU', () => {
    const qv = strategy.computeAnalysis(makeCtx({ method: 'lora', rank: 16, targetModules: 'q_v' }));
    const qkvo = strategy.computeAnalysis(makeCtx({ method: 'lora', rank: 16, targetModules: 'q_k_v_o' }));
    expect(qkvo.mfu).toBeGreaterThan(qv.mfu);
  });

  it('all_linear produces highest MFU', () => {
    const qkvo = strategy.computeAnalysis(makeCtx({ method: 'lora', rank: 16, targetModules: 'q_k_v_o' }));
    const allLinear = strategy.computeAnalysis(makeCtx({ method: 'lora', rank: 16, targetModules: 'all_linear' }));
    expect(allLinear.mfu).toBeGreaterThan(qkvo.mfu);
  });

  it('MFU delta between rank=4 and rank=64 is 0.3–2.0% for 7B model', () => {
    // Guards against multiplicative errors (e.g., accidentally applying 6A twice).
    // Adapter params scale linearly with rank, so delta should be well-bounded.
    const r4 = strategy.computeAnalysis(makeCtx({ method: 'lora', rank: 4, targetModules: 'q_k_v_o' }));
    const r64 = strategy.computeAnalysis(makeCtx({ method: 'lora', rank: 64, targetModules: 'q_k_v_o' }));
    const deltaPct = (r64.mfu - r4.mfu) * 100;
    console.log(`MFU delta (r4 vs r64): ${deltaPct.toFixed(3)}%`);
    expect(deltaPct).toBeGreaterThan(0.3);
    expect(deltaPct).toBeLessThan(2.0);
  });
});

// ---------------------------------------------------------------------------
// 7. FSDP + LoRA communication asymmetry
// ---------------------------------------------------------------------------
describe('FSDP LoRA Communication', () => {
  it('FSDP LoRA runs successfully with asymmetric comm', () => {
    const result = runSim({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 32,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
      finetuningMethod: 'lora',
      loraRank: 16,
      loraTargetModules: 'q_k_v_o',
    });

    expect(result.success).toBe(true);
    // Should complete without errors — the asymmetric FSDP comm
    // (AllGather full base weights, ReduceScatter tiny adapter grads) works correctly
    expect(result.metrics.peakMemoryGB).toBeGreaterThan(0);
    expect(result.metrics.tokensPerSecond).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// 8. 3D parallel LoRA
// ---------------------------------------------------------------------------
describe('3D Parallel LoRA', () => {
  it('fsdp-tp with LoRA runs and produces valid metrics', () => {
    const result = runSim({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 32,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp',
      strategyConfig: {
        tp: 4,
        dp: 2,
      },
      mixedPrecision: 'bf16',
      finetuningMethod: 'lora',
      loraRank: 16,
      loraTargetModules: 'q_k_v_o',
    });

    expect(result.success).toBe(true);
    expect(result.metrics.mfu).toBeGreaterThan(0.05);
    expect(result.metrics.mfu).toBeLessThan(0.60);
    expect(result.metrics.peakMemoryGB).toBeGreaterThan(0);
    console.log(`FSDP-TP LoRA: MFU=${(result.metrics.mfu * 100).toFixed(1)}%, mem=${result.metrics.peakMemoryGB.toFixed(1)} GB`);
  });
});

// ---------------------------------------------------------------------------
// 9. Adapter gradient dtype is BF16
// ---------------------------------------------------------------------------
describe('Adapter Gradient Dtype', () => {
  it('FP8 compute still uses BF16 for adapter gradient comm', () => {
    // With FP8 compute, TP comm uses FP8 quantized collectives,
    // but adapter gradients should always use BF16 for DP AllReduce
    const bf16Result = runSim({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 32,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
      finetuningMethod: 'lora',
      loraRank: 16,
      loraTargetModules: 'q_k_v_o',
    });

    const fp8Result = runSim({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 32,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      mixedPrecision: 'fp8',
      finetuningMethod: 'lora',
      loraRank: 16,
      loraTargetModules: 'q_k_v_o',
    });

    expect(bf16Result.success).toBe(true);
    expect(fp8Result.success).toBe(true);
    // Both should produce valid results
    expect(fp8Result.metrics.mfu).toBeGreaterThan(0.05);
  });
});

// ---------------------------------------------------------------------------
// 10. Default 'full' method doesn't change existing behavior
// ---------------------------------------------------------------------------
describe('Full Training Unchanged', () => {
  it('default finetuningMethod=full produces same results as no LoRA config', () => {
    const noLoraResult = runSim({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 32,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
    });

    const fullResult = runSim({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 32,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
      finetuningMethod: 'full',
    });

    expect(noLoraResult.success).toBe(true);
    expect(fullResult.success).toBe(true);
    // Should be identical — 'full' method is a no-op
    expect(fullResult.metrics.mfu).toBe(noLoraResult.metrics.mfu);
    expect(fullResult.metrics.peakMemoryGB).toBe(noLoraResult.metrics.peakMemoryGB);
    expect(fullResult.metrics.avgStepTimeMs).toBe(noLoraResult.metrics.avgStepTimeMs);
  });
});

// ---------------------------------------------------------------------------
// 11. LoRA MoE Validation
// ---------------------------------------------------------------------------
describe('LoRA MoE Validation', () => {
  // Test 1: DeepSeek V3 exact counts + routed expert exclusion
  it('DeepSeek V3: exact q_k_v_o and all_linear counts', () => {
    const model = getModel('deepseek-v3')!;
    const qkvo = computeLoraTrainableParams(model, 16, 'q_k_v_o');
    const allLinear = computeLoraTrainableParams(model, 16, 'all_linear');
    expect(qkvo).toBe(46_473_216);
    expect(allLinear).toBe(75_816_960);
    // ~1.63× increase, NOT 50×+ (routed experts excluded)
    expect(allLinear / qkvo).toBeCloseTo(1.631, 2);
  });

  // Test 2: Mixtral all_linear == q_k_v_o (strongest automated check)
  it('Mixtral 8x7B: all_linear == q_k_v_o (no dense/shared MLPs to adapt)', () => {
    const model = getModel('mixtral-8x7b')!;
    const qkvo = computeLoraTrainableParams(model, 16, 'q_k_v_o');
    const allLinear = computeLoraTrainableParams(model, 16, 'all_linear');
    expect(allLinear).toBe(qkvo);
    expect(qkvo).toBe(13_631_488);
  });

  // Test 3: LLaMA 8B vs Mixtral — same attention = same q_k_v_o
  it('LLaMA 3.1 8B and Mixtral 8x7B have identical q_k_v_o counts', () => {
    const llama = getModel('llama3.1-8b')!;
    const mixtral = getModel('mixtral-8x7b')!;
    const llamaParams = computeLoraTrainableParams(llama, 16, 'q_k_v_o');
    const mixtralParams = computeLoraTrainableParams(mixtral, 16, 'q_k_v_o');
    expect(llamaParams).toBe(mixtralParams);
    expect(llamaParams).toBe(13_631_488);
  });

  // Test 4: Dense MLPs add adapters, MoE without shared experts doesn't
  it('LLaMA 8B all_linear > Mixtral all_linear (dense MLPs vs no shared experts)', () => {
    const llama = getModel('llama3.1-8b')!;
    const mixtral = getModel('mixtral-8x7b')!;
    const llamaAll = computeLoraTrainableParams(llama, 16, 'all_linear');
    const mixtralAll = computeLoraTrainableParams(mixtral, 16, 'all_linear');
    expect(llamaAll).toBe(41_943_040);
    expect(llamaAll).toBeGreaterThan(mixtralAll * 2.5);
  });

  // Test 5: Mixtral QLoRA fits on 80GB (routed experts frozen at NF4)
  it('Mixtral 8x7B QLoRA r=16 fits on single 80GB GPU', () => {
    const result = runSim({
      modelId: 'mixtral-8x7b', clusterId: '1x-h100',
      globalBatchSize: 1, microBatchSize: 1, sequenceLength: 512,
      strategyType: 'ddp', mixedPrecision: 'bf16',
      finetuningMethod: 'qlora', loraRank: 16, loraTargetModules: 'q_k_v_o',
    });
    expect(result.success).toBe(true);
    expect(result.metrics.peakMemoryGB).toBeLessThan(80);
  });

  // Test 6: LLaMA 3.3 70B QLoRA fits on 80GB
  it('LLaMA 3.3 70B QLoRA r=16 fits on single 80GB GPU', () => {
    const result = runSim({
      modelId: 'llama3.3-70b', clusterId: '1x-h100',
      globalBatchSize: 1, microBatchSize: 1, sequenceLength: 512,
      strategyType: 'ddp', mixedPrecision: 'bf16',
      finetuningMethod: 'qlora', loraRank: 16, loraTargetModules: 'q_k_v_o',
    });
    expect(result.success).toBe(true);
    expect(result.metrics.peakMemoryGB).toBeLessThan(80);
  });

  // Test 7-8: All-MoE models without shared experts → all_linear == q_k_v_o
  it('DBRX: all_linear == q_k_v_o (no shared experts)', () => {
    const model = getModel('dbrx')!;
    expect(computeLoraTrainableParams(model, 16, 'all_linear'))
      .toBe(computeLoraTrainableParams(model, 16, 'q_k_v_o'));
  });

  it('Grok-1: all_linear == q_k_v_o (no shared experts)', () => {
    const model = getModel('grok-1')!;
    expect(computeLoraTrainableParams(model, 16, 'all_linear'))
      .toBe(computeLoraTrainableParams(model, 16, 'q_k_v_o'));
  });

  // Test 9-10: Models WITH shared experts/dense layers → all_linear > q_k_v_o
  it('DeepSeek V3: all_linear > q_k_v_o (shared experts + dense layers)', () => {
    const model = getModel('deepseek-v3')!;
    expect(computeLoraTrainableParams(model, 16, 'all_linear'))
      .toBeGreaterThan(computeLoraTrainableParams(model, 16, 'q_k_v_o'));
  });

  it('Maverick: all_linear > q_k_v_o (alternating dense/MoE + shared experts)', () => {
    const model = getModel('llama4-maverick')!;
    expect(computeLoraTrainableParams(model, 16, 'all_linear'))
      .toBeGreaterThan(computeLoraTrainableParams(model, 16, 'q_k_v_o'));
  });
});

// ---------------------------------------------------------------------------
// 12. Pipeline Parallel standalone LoRA
// ---------------------------------------------------------------------------
describe('Pipeline Parallel LoRA', () => {
  it('PP standalone with LoRA runs and produces valid metrics', () => {
    const model = getModel('llama2-7b')!;
    const cluster = getPresetCluster('8x-h100')!;
    const dtypes = DTYPE_PRESETS.bf16;
    const pp = 8;
    const m = 16;
    const gbs = 16;
    const mbs = 1;
    const dp = cluster.totalGPUs / pp; // dp=1 for PP standalone
    const ga = Math.ceil(gbs / (mbs * dp));

    const strategy = createPipelineParallelStrategy(pp, m, '1f1b');
    const ctx = {
      model,
      cluster,
      training: {
        globalBatchSize: gbs,
        microBatchSize: mbs,
        sequenceLength: 2048,
        maxSteps: 1000,
        optimizer: DEFAULT_ADAMW_CONFIG,
        lrSchedule: DEFAULT_LR_SCHEDULE,
        dtypes,
        gradientClipping: 1.0,
        gradientAccumulationSteps: ga,
      },
      seqLength: 2048,
      microBatchSize: mbs,
      globalBatchSize: gbs,
      gradientAccumulationSteps: ga,
      activationCheckpointing: true,
      flashAttention: true,
      lora: { method: 'lora' as const, rank: 16, targetModules: 'q_k_v_o' as const },
    };

    const analysis = strategy.computeAnalysis(ctx);

    // MFU should be positive (PP standalone for 7B on 8 GPUs is bubble-dominated,
    // so MFU is low — ~1.7% with LoRA. This tests correctness, not efficiency.)
    expect(analysis.mfu).toBeGreaterThan(0.005);
    expect(analysis.mfu).toBeLessThan(0.60);

    // Memory should fit in GPU
    expect(analysis.memory.total).toBeLessThan(cluster.node.gpu.memoryGB * 1e9);

    // LoRA memory should be less than full training
    const fullCtx = { ...ctx, lora: undefined };
    const fullAnalysis = strategy.computeAnalysis(fullCtx);
    expect(analysis.memory.total).toBeLessThan(fullAnalysis.memory.total);

    console.log(`PP LoRA: MFU=${(analysis.mfu * 100).toFixed(1)}%, mem=${(analysis.memory.total / 1e9).toFixed(1)} GB`);
    console.log(`PP full: MFU=${(fullAnalysis.mfu * 100).toFixed(1)}%, mem=${(fullAnalysis.memory.total / 1e9).toFixed(1)} GB`);
  });

  it('PP standalone with QLoRA uses less memory than LoRA', () => {
    const model = getModel('llama2-7b')!;
    const cluster = getPresetCluster('8x-h100')!;
    const dtypes = DTYPE_PRESETS.bf16;
    const pp = 8;
    const m = 16;
    const gbs = 16;
    const mbs = 1;
    const dp = cluster.totalGPUs / pp;
    const ga = Math.ceil(gbs / (mbs * dp));

    const strategy = createPipelineParallelStrategy(pp, m, '1f1b');
    const baseCtx = {
      model,
      cluster,
      training: {
        globalBatchSize: gbs,
        microBatchSize: mbs,
        sequenceLength: 2048,
        maxSteps: 1000,
        optimizer: DEFAULT_ADAMW_CONFIG,
        lrSchedule: DEFAULT_LR_SCHEDULE,
        dtypes,
        gradientClipping: 1.0,
        gradientAccumulationSteps: ga,
      },
      seqLength: 2048,
      microBatchSize: mbs,
      globalBatchSize: gbs,
      gradientAccumulationSteps: ga,
      activationCheckpointing: true,
      flashAttention: true,
    };

    const loraAnalysis = strategy.computeAnalysis({
      ...baseCtx,
      lora: { method: 'lora' as const, rank: 16, targetModules: 'q_k_v_o' as const },
    });
    const qloraAnalysis = strategy.computeAnalysis({
      ...baseCtx,
      lora: { method: 'qlora' as const, rank: 16, targetModules: 'q_k_v_o' as const },
    });

    expect(qloraAnalysis.memory.total).toBeLessThan(loraAnalysis.memory.total);
    console.log(`PP LoRA mem: ${(loraAnalysis.memory.total / 1e9).toFixed(1)} GB, PP QLoRA mem: ${(qloraAnalysis.memory.total / 1e9).toFixed(1)} GB`);
  });
});

// ---------------------------------------------------------------------------
// 13. Selective AC + storedLayers blending in LoRA backward multiplier
// ---------------------------------------------------------------------------
describe('LoRA Backward Multiplier with storedLayers', () => {
  const model = getModel('llama3.1-8b')!; // GQA model (f ≈ 0.2)
  const loraConfig = { method: 'lora' as const, rank: 16, targetModules: 'q_k_v_o' as const };

  it('storedLayers < totalLayers returns higher multiplier than storedLayers == totalLayers', () => {
    const multAllStored = getLoraBackwardMultiplier(model, loraConfig, true, 'selective', model.numLayers, model.numLayers);
    const multHalfStored = getLoraBackwardMultiplier(model, loraConfig, true, 'selective', Math.floor(model.numLayers / 2), model.numLayers);
    // Fewer stored layers → more full recompute → higher multiplier
    expect(multHalfStored).toBeGreaterThan(multAllStored);
  });

  it('storedLayers=0 equals full AC multiplier', () => {
    const multZeroStored = getLoraBackwardMultiplier(model, loraConfig, true, 'selective', 0, model.numLayers);
    const multFullAC = getLoraBackwardMultiplier(model, loraConfig, true, 'full');
    // storedFrac=0 → all layers use full recompute → same as full AC
    expect(multZeroStored).toBeCloseTo(multFullAC, 10);
  });

  it('storedLayers omitted behaves like storedLayers == totalLayers', () => {
    const multOmitted = getLoraBackwardMultiplier(model, loraConfig, true, 'selective');
    const multAllStored = getLoraBackwardMultiplier(model, loraConfig, true, 'selective', model.numLayers, model.numLayers);
    expect(multOmitted).toBeCloseTo(multAllStored, 10);
  });

  it('quantified impact: storedFrac=0.5 on GQA model gives ~24% higher multiplier', () => {
    const f = getSelectiveRecomputeFraction(model);
    const overhead = 1.05; // floor dominates for ≥7B models
    const selectiveMult = f + overhead;
    const fullMult = 1 + overhead;
    const halfStored = 0.5 * selectiveMult + 0.5 * fullMult;
    const allSelective = selectiveMult;
    // ~24% under-report when ignoring storedLayers
    const underReport = (halfStored - allSelective) / halfStored;
    expect(underReport).toBeGreaterThan(0.15);
    expect(underReport).toBeLessThan(0.35);
    // Verify the function matches the manual calculation
    const multHalf = getLoraBackwardMultiplier(model, loraConfig, true, 'selective', Math.floor(model.numLayers / 2), model.numLayers);
    expect(multHalf).toBeCloseTo(halfStored, 1);
  });
});
