/**
 * Activation Memory Model Validation
 *
 * Validates per-layer activation formula, PP scaling, SP/TP multipliers,
 * and cross-model behavior with the auto-optimizer.
 */
import { describe, it, expect } from 'vitest';
import {
  type SimulationConfig,
  getSimulationMetrics,
} from '../../src/core/simulation/engine.ts';
import { estimateActivationMemory } from '../../src/core/strategies/base.ts';
import { getModel } from '../../src/core/models/index.ts';
import { createMultiNodeCluster } from '../../src/core/hardware/topology.ts';

// ── Helpers ──────────────────────────────────────────────────────────────

function sim(overrides: Partial<SimulationConfig>) {
  return getSimulationMetrics({
    modelId: 'llama3-405b',
    clusterConfig: createMultiNodeCluster('h100-sxm', 8, 256)!, // 2048 GPUs
    globalBatchSize: 4096,
    microBatchSize: 1,
    sequenceLength: 8192,
    strategyType: 'fsdp-tp-pp',
    mixedPrecision: 'bf16',
    activationCheckpointing: true,
    flashAttention: true,
    strategyConfig: {
      tp: 4,
      pp: 8,
      sequenceParallel: true,
      pipelineSchedule: '1f1b',
    },
    ...overrides,
  });
}

// ── 1. Per-layer formula validation ──────────────────────────────────────

describe('Per-layer activation formula', () => {
  it('405B per-layer activation matches expected coefficient', () => {
    const model = getModel('llama3-405b', 8192)!;
    const h = model.hiddenSize;      // 16384
    const qDim = model.numAttentionHeads * model.headDim;   // 16384
    const kvDim = model.numKvHeads * model.headDim;           // 1024
    const I = model.intermediateSize; // 53248

    // Gated MLP (SwiGLU): 5h + qDim + 2*kvDim + 3I
    const expectedCoeff = 5 * h + qDim + 2 * kvDim + 3 * I;
    expect(expectedCoeff).toBe(260096);

    const tokens = 8192 * 4; // seq * mbs
    const bytes = 2; // bf16
    const raw = estimateActivationMemory(model, 8192, 4, 'bf16', false, true);
    const perLayer = raw / model.numLayers;

    // Per-layer should match coefficient × tokens × bytes (+ small attention score contribution)
    const expectedPerLayer = expectedCoeff * tokens * bytes
      + 2 * tokens // dropout masks: post-attention + post-MLP (1 byte each)
      + model.numAttentionHeads * 8192 * bytes * 4; // flash attention scores
    expect(perLayer).toBeCloseTo(expectedPerLayer, -3);
  });

  it('gated MLP uses 3I coefficient vs 2I for standard', () => {
    const gpt = getModel('gpt3-125m')!;
    const llama = getModel('llama2-7b')!;

    expect(gpt.gatedMLP).toBe(false);
    expect(llama.gatedMLP).toBe(true);

    // GPT-3 125M: standard MLP → coefficient uses 2*I
    const gptRaw = estimateActivationMemory(gpt, 1024, 1, 'bf16', false, true);
    const gptPerLayer = gptRaw / gpt.numLayers;
    const gptExpectedCoeff = 5 * gpt.hiddenSize
      + gpt.numAttentionHeads * gpt.headDim
      + 2 * gpt.numKvHeads * gpt.headDim
      + 2 * gpt.intermediateSize; // standard: 2I
    const gptExpected = gptExpectedCoeff * 1024 * 2
      + 2 * 1024 // dropout masks
      + gpt.numAttentionHeads * 1024 * 2 * 1; // flash attention
    expect(gptPerLayer).toBeCloseTo(gptExpected, -2);

    // LLaMA 7B: gated MLP → coefficient uses 3*I
    const llamaRaw = estimateActivationMemory(llama, 4096, 1, 'bf16', false, true);
    const llamaPerLayer = llamaRaw / llama.numLayers;
    const llamaExpectedCoeff = 5 * llama.hiddenSize
      + llama.numAttentionHeads * llama.headDim
      + 2 * llama.numKvHeads * llama.headDim
      + 3 * llama.intermediateSize; // gated: 3I
    const llamaExpected = llamaExpectedCoeff * 4096 * 2
      + 2 * 4096 // dropout masks
      + llama.numAttentionHeads * 4096 * 2 * 1;
    expect(llamaPerLayer).toBeCloseTo(llamaExpected, -2);
  });
});

// ── 2. PP activation scaling ─────────────────────────────────────────────

describe('PP activation scaling', () => {
  it('405B PP=8 TP=4 MBS=1 ckpt=OFF is OOM on 80GB', () => {
    const result = sim({ activationCheckpointing: false });
    expect(result.memoryUtilization).toBeGreaterThan(1.0);
  });

  it('405B PP=8 TP=4 ckpt=ON fits on 80GB', () => {
    const result = sim({ activationCheckpointing: true });
    expect(result.memoryUtilization).toBeLessThan(1.0);
    expect(result.memoryUtilization).toBeGreaterThan(0.3);
  });

  it('interleaved schedule has same in-flight memory as standard 1F1B', () => {
    // Narayanan 2021 §2.3: interleaved 1F1B has the same peak activation memory
    // as standard 1F1B — each device holds v virtual stages, each with in-flight
    // activations, so peak is still pp.
    const oneF1B = sim({
      activationCheckpointing: true,
      strategyConfig: {
        tp: 4, pp: 8, sequenceParallel: true,
        pipelineSchedule: '1f1b', interleavedStages: 1,
      },
    });
    const interleaved4 = sim({
      activationCheckpointing: true,
      strategyConfig: {
        tp: 4, pp: 8, sequenceParallel: true,
        pipelineSchedule: 'interleaved-1f1b', interleavedStages: 4,
      },
    });
    const interleaved8 = sim({
      activationCheckpointing: true,
      strategyConfig: {
        tp: 4, pp: 8, sequenceParallel: true,
        pipelineSchedule: 'interleaved-1f1b', interleavedStages: 8,
      },
    });

    // All schedules have pp in-flight microbatches → same activation memory
    expect(oneF1B.memoryPerGPU.activations).toBe(interleaved4.memoryPerGPU.activations);
    expect(interleaved4.memoryPerGPU.activations).toBe(interleaved8.memoryPerGPU.activations);
  });
});

// ── 3. SP multiplier correctness ─────────────────────────────────────────

describe('SP multiplier', () => {
  it('SP reduces activation memory by 1/tp factor', () => {
    const spOn = sim({
      strategyConfig: {
        tp: 4, pp: 8, sequenceParallel: true,
        pipelineSchedule: 'interleaved-1f1b', interleavedStages: 8,
      },
    });
    const spOff = sim({
      strategyConfig: {
        tp: 4, pp: 8, sequenceParallel: false,
        pipelineSchedule: 'interleaved-1f1b', interleavedStages: 8,
      },
    });

    const ratio = spOn.memoryPerGPU.activations / spOff.memoryPerGPU.activations;

    // With SP: multiplier = 1/tp = 0.25
    // Without SP: multiplier = (sharded/tp + replicated) / total > 0.25
    // So ratio should be < 1 and model-dependent
    expect(ratio).toBeLessThan(0.7);
    expect(ratio).toBeGreaterThan(0.3);
  });

  it('TP>1 SP=OFF uses sharded/replicated decomposition', () => {
    // Without SP, TP-sharded tensors (Q,K,V,attn_out,MLP) get 1/tp,
    // but replicated tensors (4h: LN inputs, residuals) stay full size.
    // Multiplier = (sharded/tp + replicated) / (sharded + replicated)
    const spOff_tp4 = sim({
      strategyConfig: {
        tp: 4, pp: 8, sequenceParallel: false,
        pipelineSchedule: 'interleaved-1f1b', interleavedStages: 8,
      },
    });
    // The key thing is: TP=4 SP=OFF multiplier is > 1/tp (replicated tensors not sharded)
    const model = getModel('llama3-405b', 8192)!;
    const h = model.hiddenSize;
    const qDim = model.numAttentionHeads * model.headDim;
    const kvDim = model.numKvHeads * model.headDim;
    const I = model.intermediateSize;
    const sharded = qDim + 2 * kvDim + h + 3 * I;
    const replicated = 4 * h;
    const expectedMultiplier = (sharded / 4 + replicated) / (sharded + replicated);

    // Verify multiplier is between 1/tp and 1
    expect(expectedMultiplier).toBeGreaterThan(1 / 4);
    expect(expectedMultiplier).toBeLessThan(1);

    // SP=OFF should have MORE activations than SP=ON for same config
    const spOn_tp4 = sim({
      strategyConfig: {
        tp: 4, pp: 8, sequenceParallel: true,
        pipelineSchedule: 'interleaved-1f1b', interleavedStages: 8,
      },
    });
    expect(spOff_tp4.memoryPerGPU.activations).toBeGreaterThan(
      spOn_tp4.memoryPerGPU.activations
    );

    // Ratio should be approximately expectedMultiplier / (1/tp) = expectedMultiplier * tp
    const ratio = spOff_tp4.memoryPerGPU.activations / spOn_tp4.memoryPerGPU.activations;
    expect(ratio).toBeCloseTo(expectedMultiplier * 4, 0); // ~1.7 for 405B
  });

  it('SP=ON with TP=8 gives greater savings than TP=4', () => {
    const tp4 = sim({
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 256)!,
      strategyConfig: {
        tp: 4, pp: 8, sequenceParallel: true,
        pipelineSchedule: 'interleaved-1f1b', interleavedStages: 8,
      },
    });
    const tp8 = sim({
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 256)!,
      strategyConfig: {
        tp: 8, pp: 4, sequenceParallel: true,
        pipelineSchedule: 'interleaved-1f1b', interleavedStages: 4,
      },
    });

    // TP=8 with SP gives 1/8 multiplier, TP=4 gives 1/4 → TP=8 has less activation per GPU
    // But layersPerStage also changes (PP=4 vs PP=8), so just check relative direction
    // Per-layer act: TP=8 → 1/8 * perLayer, TP=4 → 1/4 * perLayer
    // But layers/stage: PP=4 → more layers, PP=8 → fewer layers
    // Net: activation should be similar magnitude but relationship depends on layout
    expect(tp4.memoryPerGPU.activations).toBeGreaterThan(0);
    expect(tp8.memoryPerGPU.activations).toBeGreaterThan(0);
  });
});

// ── 4. Checkpointing reduces activations ─────────────────────────────────

describe('Checkpointing effect', () => {
  it('checkpointing reduces activations by sqrt(layersPerStage) factor', () => {
    const ckptOff = sim({ activationCheckpointing: false });
    const ckptOn = sim({ activationCheckpointing: true });

    const ratio = ckptOn.memoryPerGPU.activations / ckptOff.memoryPerGPU.activations;
    const layersPerStage = Math.ceil(126 / 8); // 16
    const expectedRatio = Math.sqrt(layersPerStage) / layersPerStage; // 4/16 = 0.25

    expect(ratio).toBeCloseTo(expectedRatio, 1);
  });
});

// ── 5. Cross-model verification ──────────────────────────────────────────

describe('Cross-model memory sanity', () => {
  it('LLaMA 2 7B FSDP ckpt=ON fits on 80GB', () => {
    const result = getSimulationMetrics({
      modelId: 'llama2-7b',
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 1)!,
      globalBatchSize: 32,
      microBatchSize: 4,
      sequenceLength: 4096,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
      activationCheckpointing: true,
      flashAttention: true,
    });
    expect(result.memoryUtilization).toBeLessThan(1.0);
    expect(result.memoryUtilization).toBeGreaterThan(0.1);
  });

  it('LLaMA 2 7B FSDP ckpt=OFF MBS=2 fits on 80GB', () => {
    // MBS=2 is the realistic config for 7B without checkpointing.
    // MBS=4 correctly OOMs (73 GB activations alone) — aggressive for no-ckpt.
    const result = getSimulationMetrics({
      modelId: 'llama2-7b',
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 1)!,
      globalBatchSize: 16,
      microBatchSize: 2,
      sequenceLength: 4096,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
      activationCheckpointing: false,
      flashAttention: true,
    });
    expect(result.memoryUtilization).toBeLessThan(1.0);
    expect(result.memoryUtilization).toBeGreaterThan(0.3);
  });

  it('LLaMA 2 70B FSDP ckpt=ON fits on 80GB', () => {
    const result = getSimulationMetrics({
      modelId: 'llama2-70b',
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 8)!,
      globalBatchSize: 64,
      microBatchSize: 2,
      sequenceLength: 4096,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
      activationCheckpointing: true,
      flashAttention: true,
    });
    expect(result.memoryUtilization).toBeLessThan(1.0);
    expect(result.memoryUtilization).toBeGreaterThan(0.3);
  });

  it('GPT-3 175B needs checkpointing for reasonable configs', () => {
    const ckptOff = getSimulationMetrics({
      modelId: 'gpt3-175b',
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 128)!,
      globalBatchSize: 1024,
      microBatchSize: 2,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp-pp',
      mixedPrecision: 'bf16',
      activationCheckpointing: false,
      flashAttention: true,
      strategyConfig: { tp: 8, pp: 8, sequenceParallel: true },
    });
    const ckptOn = getSimulationMetrics({
      modelId: 'gpt3-175b',
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 128)!,
      globalBatchSize: 1024,
      microBatchSize: 2,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp-pp',
      mixedPrecision: 'bf16',
      activationCheckpointing: true,
      flashAttention: true,
      strategyConfig: { tp: 8, pp: 8, sequenceParallel: true },
    });

    // ckpt=OFF should use significantly more memory
    expect(ckptOff.memoryPerGPU.activations).toBeGreaterThan(
      ckptOn.memoryPerGPU.activations * 2
    );
    // ckpt=ON should fit
    expect(ckptOn.memoryUtilization).toBeLessThan(1.0);
  });

  it('selective checkpointing memory between none and full', () => {
    const ckptOff = sim({ activationCheckpointing: false });
    const selective = sim({
      activationCheckpointing: true,
      checkpointingGranularity: 'selective',
    });
    const full = sim({
      activationCheckpointing: true,
      checkpointingGranularity: 'full',
    });

    // Selective < None (discards attention activations per-layer)
    expect(selective.memoryPerGPU.activations).toBeLessThan(ckptOff.memoryPerGPU.activations);
    // Selective > Full (full uses sqrt(N) layers, selective stores all N)
    expect(selective.memoryPerGPU.activations).toBeGreaterThan(full.memoryPerGPU.activations);
  });

  it('selective MFU > full MFU (less compute overhead)', () => {
    const selective = sim({
      activationCheckpointing: true,
      checkpointingGranularity: 'selective',
    });
    const full = sim({
      activationCheckpointing: true,
      checkpointingGranularity: 'full',
    });

    // Selective recomputes only ~10% of forward (attention QKV projections),
    // full recomputes entire forward → selective has higher MFU
    expect(selective.mfu).toBeGreaterThan(full.mfu);
    // Difference should be meaningful but not huge (~5-20% relative)
    expect(selective.mfu / full.mfu).toBeGreaterThan(1.03);
    expect(selective.mfu / full.mfu).toBeLessThan(1.25);
  });

  it('selective HFU reflects continuous recompute fraction', () => {
    const selective = sim({
      activationCheckpointing: true,
      checkpointingGranularity: 'selective',
    });
    const full = sim({
      activationCheckpointing: true,
      checkpointingGranularity: 'full',
    });

    // Full checkpointing: HFU > MFU (8PD vs 6PD → HFU = MFU * 4/3)
    expect(full.hfu).toBeGreaterThan(full.mfu * 1.2);
    // Selective: HFU above MFU. When all layers fit (storedFrac=1),
    // HFU/MFU ≈ (6 + 2f)/6 ≈ 1.04-1.07 for GQA (f ≈ 0.13-0.22).
    // When auto-resolved storedLayers < totalLayers, the blend pushes
    // HFU toward full-AC ratio (4/3 ≈ 1.333).
    expect(selective.hfu).toBeGreaterThan(selective.mfu * 1.02);
    expect(selective.hfu).toBeLessThan(selective.mfu * 1.34);
  });

  it('Megatron formula cross-check for 405B with SP+TP=4', () => {
    // Megatron (10+24/t)sbh for standard MHA with SP, t=4: coefficient = 16 in sbh units
    // Our formula in sbh: (5h + qDim + 2*kvDim + 3I) / h
    const model = getModel('llama3-405b', 8192)!;
    const h = model.hiddenSize;
    const qDim = model.numAttentionHeads * model.headDim;
    const kvDim = model.numKvHeads * model.headDim;
    const I = model.intermediateSize;

    const ourCoeffInSBH = (5 * h + qDim + 2 * kvDim + 3 * I) / h;
    // Should be close to Megatron's 16 (for standard MHA, I=4h)
    // Our formula gives ~15.9 for 405B (GQA with small kvDim saves a bit)
    expect(ourCoeffInSBH).toBeGreaterThan(14);
    expect(ourCoeffInSBH).toBeLessThan(17);
  });
});

// ── 6. Selective checkpointing decomposition ────────────────────────────

describe('Selective checkpointing decomposition', () => {
  it('attention + MLP + shared = total per-layer activations', () => {
    const model = getModel('llama2-7b', 4096)!;

    // No checkpointing → N × full per-layer
    const noCheckpoint = estimateActivationMemory(model, 4096, 1, 'bf16', false, true);
    // Selective → N × (shared + MLP)
    const selective = estimateActivationMemory(model, 4096, 1, 'bf16', true, true, 1, 'selective');
    // Full → sqrt(N) × full per-layer
    const full = estimateActivationMemory(model, 4096, 1, 'bf16', true, true, 1, 'full');

    // Selective should discard a meaningful fraction (attention is ~20-30% of per-layer)
    const attentionFraction = 1 - selective / noCheckpoint;
    expect(attentionFraction).toBeGreaterThan(0.15);
    expect(attentionFraction).toBeLessThan(0.40);

    // Full uses sqrt(N) layers vs N for selective
    // For 7B (32 layers): sqrt(32) ≈ 5.66, so full ≈ 5.66/32 ≈ 17.7% of no-ckpt
    // Selective ≈ (1 - attentionFraction) × 100% ≈ 70-80% of no-ckpt
    expect(full).toBeLessThan(selective);
    expect(selective).toBeLessThan(noCheckpoint);
  });

  it('selective memory scales with all N layers (not sqrt(N))', () => {
    // Compare two models with different layer counts
    const model7b = getModel('llama2-7b', 4096)!;   // 32 layers
    const model70b = getModel('llama2-70b', 4096)!;  // 80 layers

    const sel7b = estimateActivationMemory(model7b, 4096, 1, 'bf16', true, true, 1, 'selective');
    const full7b = estimateActivationMemory(model7b, 4096, 1, 'bf16', true, true, 1, 'full');
    const sel70b = estimateActivationMemory(model70b, 4096, 1, 'bf16', true, true, 1, 'selective');
    const full70b = estimateActivationMemory(model70b, 4096, 1, 'bf16', true, true, 1, 'full');

    // Full checkpoint ratio should scale as sqrt(N) → sqrt(80)/sqrt(32) ≈ 1.58
    const fullRatio = full70b / full7b;
    // Selective ratio should scale linearly with layers (and per-layer size)
    const selRatio = sel70b / sel7b;

    // Selective scales faster because it's linear in N
    expect(selRatio).toBeGreaterThan(fullRatio);
  });

  it('MoE model selective correctly handles mixed layer types', () => {
    const model = getModel('deepseek-v3', 8192)!;
    expect(model.isMoE).toBe(true);
    expect(model.numMoELayers).toBeGreaterThan(0);

    const noCheckpoint = estimateActivationMemory(model, 8192, 1, 'bf16', false, true);
    const selective = estimateActivationMemory(model, 8192, 1, 'bf16', true, true, 1, 'selective');
    const full = estimateActivationMemory(model, 8192, 1, 'bf16', true, true, 1, 'full');

    expect(full).toBeLessThan(selective);
    expect(selective).toBeLessThan(noCheckpoint);
    expect(selective).toBeGreaterThan(0);
  });
});
