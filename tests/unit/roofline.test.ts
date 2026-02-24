/**
 * Roofline Model tests
 *
 * Validates ceiling computation, per-operation arithmetic intensity,
 * TP effects, MoE expert intensity, inference prefill/decode, aggregate
 * training point, FLOPs fractions, and Flash Attention effects.
 */

import { describe, it, expect } from 'vitest';
import { buildModelSpec } from '../../src/core/models/primitives.ts';
import { ALL_MODEL_CONFIGS } from '../../src/core/models/architectures.ts';
import { ALL_GPUS } from '../../src/core/hardware/gpu.ts';
import { SimulationEngine } from '../../src/core/simulation/engine.ts';
import {
  computeCeiling,
  computeTrainingRoofline,
  computeInferenceRoofline,
} from '../../src/core/roofline/compute.ts';
import { calculateMetricsFromConfig } from '../../src/core/inference/latency.ts';
import type { InferenceSimulationResult } from '../../src/types/inference.ts';

// Helper: build a minimal InferenceSimulationResult from latency metrics
function makeInferenceResult(
  model: Parameters<typeof computeInferenceRoofline>[0],
  gpu: Parameters<typeof computeInferenceRoofline>[1],
  batchSize: number,
  inputSeqLen = 1024,
  outputSeqLen = 512,
  precision: 'bf16' | 'fp16' = 'bf16',
): InferenceSimulationResult {
  const r = calculateMetricsFromConfig({
    modelSpec: model,
    gpu,
    numGPUs: 1,
    batchSize,
    inputSeqLen,
    outputSeqLen,
    weightPrecision: precision,
    kvCachePrecision: precision,
    flashAttention: true,
    pagedAttention: false,
    continuousBatching: false,
    speculative: { enabled: false, draftModel: null, numSpeculativeTokens: 0, acceptanceRate: 0 },
  });
  return {
    success: true,
    memory: { weights: 0, kvCache: 0, activations: 0, overhead: 0, total: 0 },
    kvCacheState: { currentSeqLen: 0, batchSize: 0, memoryUsed: 0, memoryPerToken: 0, utilizationPercent: 0 },
    latency: r.latency,
    throughput: r.throughput,
    utilization: r.utilization,
    maxConcurrentRequests: 0,
    errors: [],
    warnings: [],
    recommendations: [],
    events: [],
  };
}

// ---------------------------------------------------------------------------
// 5a. Ridge point tests
// ---------------------------------------------------------------------------

describe('Ridge point computation', () => {
  it('H100 SXM bf16: ridge ≈ 295', () => {
    const c = computeCeiling(ALL_GPUS['h100-sxm'], 'bf16');
    expect(c.peakComputeTFLOPS).toBe(989);
    expect(c.peakBandwidthTBps).toBe(3.35);
    expect(c.ridgePoint).toBeCloseTo(295.22, 0);
  });

  it('A100 80GB bf16: ridge ≈ 153', () => {
    const c = computeCeiling(ALL_GPUS['a100-80gb'], 'bf16');
    expect(c.peakComputeTFLOPS).toBe(312);
    expect(c.peakBandwidthTBps).toBe(2.039);
    expect(c.ridgePoint).toBeCloseTo(153.02, 0);
  });

  it('B200 bf16: ridge ≈ 292', () => {
    const c = computeCeiling(ALL_GPUS['b200'], 'bf16');
    expect(c.peakComputeTFLOPS).toBe(2250);
    expect(c.peakBandwidthTBps).toBe(7.7);
    expect(c.ridgePoint).toBeCloseTo(292.21, 0);
  });

  it('MI300X bf16: ridge ≈ 247', () => {
    const c = computeCeiling(ALL_GPUS['mi300x'], 'bf16');
    expect(c.peakComputeTFLOPS).toBe(1307);
    expect(c.peakBandwidthTBps).toBe(5.3);
    expect(c.ridgePoint).toBeCloseTo(246.60, 0);
  });
});

// ---------------------------------------------------------------------------
// 5b. Per-operation intensity (LLaMA 7B, H100, bf16)
// ---------------------------------------------------------------------------

describe('Per-operation intensity (LLaMA 7B)', () => {
  const h100 = ALL_GPUS['h100-sxm'];
  const model = buildModelSpec(ALL_MODEL_CONFIGS['llama2-7b'], 2048);

  // Run simulation once for aggregate point
  const engine = new SimulationEngine();
  engine.configure({
    modelId: 'llama2-7b', clusterId: '8x-h100',
    globalBatchSize: 32, microBatchSize: 4, sequenceLength: 2048,
    strategyType: 'fsdp', mixedPrecision: 'bf16',
  });
  const metrics = engine.simulate();
  const roofline = computeTrainingRoofline(model, h100, 'bf16', metrics, {
    tp: 1, pp: 1, dp: 8,
    seqLength: 2048, microBatchSize: 4, globalBatchSize: 32,
    activationCheckpointing: true, flashAttention: true, dpType: 'fsdp',
  });

  const findPoint = (label: string) => roofline.points.find(p => p.label === label);

  it('QKV Projection: deeply compute-bound (AI > 2000)', () => {
    const p = findPoint('QKV Projection')!;
    expect(p).toBeDefined();
    expect(p.arithmeticIntensity).toBeGreaterThan(2000);
    // Actual: 2234
    expect(p.arithmeticIntensity).toBeLessThan(3000);
  });

  it('MLP (SwiGLU): deeply compute-bound (AI > 2000)', () => {
    const p = findPoint('MLP (SwiGLU)')!;
    expect(p).toBeDefined();
    expect(p.arithmeticIntensity).toBeGreaterThan(2000);
    expect(p.arithmeticIntensity).toBeLessThan(3000);
  });

  it('Attention Scores: compute-bound with FA (AI > 800)', () => {
    const p = findPoint('Attention Scores')!;
    expect(p).toBeDefined();
    // With FA at S=2048, headDim=128: faFactor = min(2*128²/2048, 8) = 8
    // AI ~910 (above H100 ridge of 295)
    expect(p.arithmeticIntensity).toBeGreaterThan(800);
    expect(p.arithmeticIntensity).toBeLessThan(1200);
  });

  it('RMSNorm: bandwidth-bound (AI ≈ 1.0)', () => {
    const p = findPoint('RMSNorm')!;
    expect(p).toBeDefined();
    // AI = 4 / (2 * 2) = 1.0 for bf16
    expect(p.arithmeticIntensity).toBeGreaterThan(0.8);
    expect(p.arithmeticIntensity).toBeLessThan(1.2);
  });

  it('Optimizer (AdamW): bandwidth-bound (AI ≈ 0.33)', () => {
    const p = findPoint('Optimizer (AdamW)')!;
    expect(p).toBeDefined();
    // AI = 10/30 ≈ 0.33
    expect(p.arithmeticIntensity).toBeGreaterThan(0.25);
    expect(p.arithmeticIntensity).toBeLessThan(0.45);
  });
});

// ---------------------------------------------------------------------------
// 5c. TP effect on matmul intensity
// ---------------------------------------------------------------------------

describe('TP effect on matmul intensity', () => {
  const h100 = ALL_GPUS['h100-sxm'];
  const model = buildModelSpec(ALL_MODEL_CONFIGS['llama2-70b'], 4096);

  // Dummy metrics for TP=1
  const eng1 = new SimulationEngine();
  eng1.configure({
    modelId: 'llama2-70b', clusterId: '64x-h100',
    globalBatchSize: 64, microBatchSize: 1, sequenceLength: 4096,
    strategyType: 'fsdp-tp', mixedPrecision: 'bf16', strategyConfig: { tp: 1 },
  });
  const m1 = eng1.simulate();

  const eng8 = new SimulationEngine();
  eng8.configure({
    modelId: 'llama2-70b', clusterId: '64x-h100',
    globalBatchSize: 64, microBatchSize: 1, sequenceLength: 4096,
    strategyType: 'fsdp-tp', mixedPrecision: 'bf16', strategyConfig: { tp: 8 },
  });
  const m8 = eng8.simulate();

  const r1 = computeTrainingRoofline(model, h100, 'bf16', m1, {
    tp: 1, pp: 1, dp: 64, seqLength: 4096, microBatchSize: 1, globalBatchSize: 64,
    activationCheckpointing: true, flashAttention: true, dpType: 'fsdp',
  });
  const r8 = computeTrainingRoofline(model, h100, 'bf16', m8, {
    tp: 8, pp: 1, dp: 8, seqLength: 4096, microBatchSize: 1, globalBatchSize: 64,
    activationCheckpointing: true, flashAttention: true, dpType: 'fsdp',
  });

  it('TP=1 QKV: AI > 2000; TP=8 QKV: AI 700-1200', () => {
    const qkv1 = r1.points.find(p => p.label === 'QKV Projection')!;
    const qkv8 = r8.points.find(p => p.label === 'QKV Projection')!;
    // TP=1: 2156, TP=8: 872
    expect(qkv1.arithmeticIntensity).toBeGreaterThan(2000);
    expect(qkv8.arithmeticIntensity).toBeGreaterThan(700);
    expect(qkv8.arithmeticIntensity).toBeLessThan(1200);
  });

  it('TP=1 MLP: AI > 2000; TP=8 MLP: AI 1200-2000', () => {
    const mlp1 = r1.points.find(p => p.label === 'MLP (SwiGLU)')!;
    const mlp8 = r8.points.find(p => p.label === 'MLP (SwiGLU)')!;
    // TP=1: 2493, TP=8: 1550
    expect(mlp1.arithmeticIntensity).toBeGreaterThan(2000);
    expect(mlp8.arithmeticIntensity).toBeGreaterThan(1200);
    expect(mlp8.arithmeticIntensity).toBeLessThan(2000);
  });

  it('TP=8 matmul intensities are lower than TP=1', () => {
    const qkv1 = r1.points.find(p => p.label === 'QKV Projection')!;
    const qkv8 = r8.points.find(p => p.label === 'QKV Projection')!;
    expect(qkv8.arithmeticIntensity).toBeLessThan(qkv1.arithmeticIntensity);

    const mlp1 = r1.points.find(p => p.label === 'MLP (SwiGLU)')!;
    const mlp8 = r8.points.find(p => p.label === 'MLP (SwiGLU)')!;
    expect(mlp8.arithmeticIntensity).toBeLessThan(mlp1.arithmeticIntensity);
  });
});

// ---------------------------------------------------------------------------
// 5d. MoE expert matmul intensity
// ---------------------------------------------------------------------------

describe('MoE expert matmul intensity', () => {
  it('DeepSeek V3: Dense MLP AI >> Expert MLP AI', () => {
    const h100 = ALL_GPUS['h100-sxm'];
    const model = buildModelSpec(ALL_MODEL_CONFIGS['deepseek-v3'], 4096);

    // Use dummy metrics (we only care about intensity)
    const eng = new SimulationEngine();
    eng.configure({
      modelId: 'llama2-70b', clusterId: '64x-h100',
      globalBatchSize: 64, microBatchSize: 1, sequenceLength: 4096,
      strategyType: 'fsdp-tp', mixedPrecision: 'bf16', strategyConfig: { tp: 8 },
    });
    const dummyMetrics = eng.simulate();

    const r = computeTrainingRoofline(model, h100, 'bf16', dummyMetrics, {
      tp: 8, pp: 1, dp: 16, seqLength: 4096, microBatchSize: 1, globalBatchSize: 128,
      activationCheckpointing: true, flashAttention: true, dpType: 'fsdp',
    });

    const dense = r.points.find(p => p.label === 'Dense MLP');
    const expert = r.points.find(p => p.label === 'Expert MLP');
    expect(dense).toBeDefined();
    expect(expert).toBeDefined();

    // Dense MLP AI: ~1223, Expert MLP AI: ~233 — at least 2× difference
    expect(dense!.arithmeticIntensity).toBeGreaterThan(1000);
    expect(expert!.arithmeticIntensity).toBeGreaterThan(150);
    expect(expert!.arithmeticIntensity).toBeLessThan(400);
    expect(dense!.arithmeticIntensity / expert!.arithmeticIntensity).toBeGreaterThan(2);
  });
});

// ---------------------------------------------------------------------------
// 5e. Inference: prefill vs decode regime
// ---------------------------------------------------------------------------

describe('Inference: prefill vs decode', () => {
  const h100 = ALL_GPUS['h100-sxm'];
  const model = buildModelSpec(ALL_MODEL_CONFIGS['llama2-7b'], 2048);

  it('Prefill B=1: high intensity, high attained TFLOPS', () => {
    const ir = makeInferenceResult(model, h100, 1);
    const r = computeInferenceRoofline(model, h100, 'bf16', ir, {
      inputSeqLen: 1024, outputSeqLen: 512, batchSize: 1, tp: 1,
    });
    const prefill = r.points.find(p => p.category === 'prefill')!;
    // AI = 1024 (= inputSeqLen since FLOPs=2P*S, bytes=P*2 → AI=S)
    expect(prefill.arithmeticIntensity).toBeGreaterThan(800);
    expect(prefill.arithmeticIntensity).toBeLessThan(1300);
    expect(prefill.attainedTFLOPS).toBeGreaterThan(300);
  });

  it('Decode B=1: low intensity, low attained TFLOPS (bandwidth-limited)', () => {
    const ir = makeInferenceResult(model, h100, 1);
    const r = computeInferenceRoofline(model, h100, 'bf16', ir, {
      inputSeqLen: 1024, outputSeqLen: 512, batchSize: 1, tp: 1,
    });
    const decode = r.points.find(p => p.category === 'decode')!;
    // AI ≈ 0.95 (deeply bandwidth-bound)
    expect(decode.arithmeticIntensity).toBeGreaterThan(0.5);
    expect(decode.arithmeticIntensity).toBeLessThan(2.0);
    expect(decode.attainedTFLOPS).toBeLessThan(10);
    expect(decode.attainedTFLOPS).toBeGreaterThan(0.5);
  });

  it('Decode B=32: intensity slides right by ~13×', () => {
    const ir = makeInferenceResult(model, h100, 32);
    const r = computeInferenceRoofline(model, h100, 'bf16', ir, {
      inputSeqLen: 1024, outputSeqLen: 512, batchSize: 32, tp: 1,
    });
    const decode = r.points.find(p => p.category === 'decode')!;
    // AI ≈ 12.3 (slides right with batch, but KV cache also grows)
    expect(decode.arithmeticIntensity).toBeGreaterThan(8);
    expect(decode.arithmeticIntensity).toBeLessThan(20);
    expect(decode.attainedTFLOPS).toBeGreaterThan(15);
  });

  it('Decode B=256: intensity approaching ridge', () => {
    const ir = makeInferenceResult(model, h100, 256);
    const r = computeInferenceRoofline(model, h100, 'bf16', ir, {
      inputSeqLen: 1024, outputSeqLen: 512, batchSize: 256, tp: 1,
    });
    const decode = r.points.find(p => p.category === 'decode')!;
    // AI ≈ 18.6 (KV cache dominates at large batch)
    expect(decode.arithmeticIntensity).toBeGreaterThan(12);
    expect(decode.arithmeticIntensity).toBeLessThan(30);
    expect(decode.attainedTFLOPS).toBeGreaterThan(30);
  });
});

// ---------------------------------------------------------------------------
// 5f. Aggregate training point
// ---------------------------------------------------------------------------

describe('Aggregate training point', () => {
  it('matches metrics.tflopsPerGPU and is below peak', () => {
    const h100 = ALL_GPUS['h100-sxm'];
    const model = buildModelSpec(ALL_MODEL_CONFIGS['llama2-7b'], 2048);
    const engine = new SimulationEngine();
    engine.configure({
      modelId: 'llama2-7b', clusterId: '8x-h100',
      globalBatchSize: 32, microBatchSize: 4, sequenceLength: 2048,
      strategyType: 'fsdp', mixedPrecision: 'bf16',
    });
    const metrics = engine.simulate();
    const roofline = computeTrainingRoofline(model, h100, 'bf16', metrics, {
      tp: 1, pp: 1, dp: 8,
      seqLength: 2048, microBatchSize: 4, globalBatchSize: 32,
      activationCheckpointing: true, flashAttention: true, dpType: 'fsdp',
    });

    const agg = roofline.points.find(p => p.isAggregate)!;
    expect(agg).toBeDefined();
    expect(agg.attainedTFLOPS).toBe(metrics.tflopsPerGPU);
    expect(agg.attainedTFLOPS).toBeLessThan(989); // below H100 peak
    expect(agg.attainedTFLOPS).toBeGreaterThan(0);
    expect(agg.arithmeticIntensity).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// 5g. FLOPs fractions sum to ~1.0
// ---------------------------------------------------------------------------

describe('FLOPs fractions', () => {
  it('sum to approximately 1.0', () => {
    const h100 = ALL_GPUS['h100-sxm'];
    const model = buildModelSpec(ALL_MODEL_CONFIGS['llama2-7b'], 2048);
    const engine = new SimulationEngine();
    engine.configure({
      modelId: 'llama2-7b', clusterId: '8x-h100',
      globalBatchSize: 32, microBatchSize: 4, sequenceLength: 2048,
      strategyType: 'fsdp', mixedPrecision: 'bf16',
    });
    const metrics = engine.simulate();
    const roofline = computeTrainingRoofline(model, h100, 'bf16', metrics, {
      tp: 1, pp: 1, dp: 8,
      seqLength: 2048, microBatchSize: 4, globalBatchSize: 32,
      activationCheckpointing: true, flashAttention: true, dpType: 'fsdp',
    });

    const sum = roofline.points
      .filter(p => !p.isAggregate)
      .reduce((s, p) => s + p.flopsFraction, 0);
    // Should sum to ~1.0 (small ops like embedding/output head not modeled)
    expect(sum).toBeGreaterThan(0.95);
    expect(sum).toBeLessThan(1.05);
  });
});

// ---------------------------------------------------------------------------
// 5h. Flash Attention effect on attention score intensity
// ---------------------------------------------------------------------------

describe('Flash Attention effect', () => {
  const h100 = ALL_GPUS['h100-sxm'];
  const dummyEngine = new SimulationEngine();
  dummyEngine.configure({
    modelId: 'llama2-7b', clusterId: '8x-h100',
    globalBatchSize: 8, microBatchSize: 1, sequenceLength: 4096,
    strategyType: 'fsdp', mixedPrecision: 'bf16',
  });
  const dummyMetrics = dummyEngine.simulate();

  it('S=4096: FA increases attention score AI by ~8× (headDim=128)', () => {
    const model = buildModelSpec(ALL_MODEL_CONFIGS['llama2-7b'], 4096);
    const baseConfig = {
      tp: 1, pp: 1, dp: 8,
      seqLength: 4096, microBatchSize: 1, globalBatchSize: 8,
      activationCheckpointing: true, dpType: 'fsdp' as const,
    };

    const noFA = computeTrainingRoofline(model, h100, 'bf16', dummyMetrics, {
      ...baseConfig, flashAttention: false,
    });
    const withFA = computeTrainingRoofline(model, h100, 'bf16', dummyMetrics, {
      ...baseConfig, flashAttention: true,
    });

    const noFA_attn = noFA.points.find(p => p.label === 'Attention Scores')!;
    const FA_attn = withFA.points.find(p => p.label === 'Attention Scores')!;

    // Without FA: AI ≈ 120 (below H100 ridge of 295)
    expect(noFA_attn.arithmeticIntensity).toBeGreaterThan(80);
    expect(noFA_attn.arithmeticIntensity).toBeLessThan(200);

    // With FA: AI ≈ 964 (above ridge — now compute-bound)
    expect(FA_attn.arithmeticIntensity).toBeGreaterThan(700);
    expect(FA_attn.arithmeticIntensity).toBeLessThan(1300);

    // FA factor should be ~8× for headDim=128, S=4096
    const factor = FA_attn.arithmeticIntensity / noFA_attn.arithmeticIntensity;
    expect(factor).toBeGreaterThan(5);
    expect(factor).toBeLessThan(10);
  });

  it('without FA, softmax is shown as a separate point', () => {
    const model = buildModelSpec(ALL_MODEL_CONFIGS['llama2-7b'], 4096);
    const noFA = computeTrainingRoofline(model, h100, 'bf16', dummyMetrics, {
      tp: 1, pp: 1, dp: 8,
      seqLength: 4096, microBatchSize: 1, globalBatchSize: 8,
      activationCheckpointing: true, flashAttention: false, dpType: 'fsdp',
    });
    const withFA = computeTrainingRoofline(model, h100, 'bf16', dummyMetrics, {
      tp: 1, pp: 1, dp: 8,
      seqLength: 4096, microBatchSize: 1, globalBatchSize: 8,
      activationCheckpointing: true, flashAttention: true, dpType: 'fsdp',
    });

    expect(noFA.points.find(p => p.label === 'Softmax')).toBeDefined();
    expect(withFA.points.find(p => p.label === 'Softmax')).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// 5i. LoRA/QLoRA aggregate point
// ---------------------------------------------------------------------------

describe('LoRA/QLoRA aggregate roofline point', () => {
  const h100 = ALL_GPUS['h100-sxm'];
  const model = buildModelSpec(ALL_MODEL_CONFIGS['llama2-7b'], 2048);

  const engine = new SimulationEngine();
  engine.configure({
    modelId: 'llama2-7b', clusterId: '8x-h100',
    globalBatchSize: 32, microBatchSize: 4, sequenceLength: 2048,
    strategyType: 'fsdp', mixedPrecision: 'bf16',
  });
  const metrics = engine.simulate();

  const baseConfig = {
    tp: 1, pp: 1, dp: 8,
    seqLength: 2048, microBatchSize: 4, globalBatchSize: 32,
    activationCheckpointing: true, flashAttention: true, dpType: 'fsdp' as const,
  };

  const fullRoofline = computeTrainingRoofline(model, h100, 'bf16', metrics, baseConfig);
  const loraRoofline = computeTrainingRoofline(model, h100, 'bf16', metrics, {
    ...baseConfig, finetuningMethod: 'lora',
  });
  const qloraRoofline = computeTrainingRoofline(model, h100, 'bf16', metrics, {
    ...baseConfig, finetuningMethod: 'qlora',
  });

  const fullAgg = fullRoofline.points.find(p => p.isAggregate)!;
  const loraAgg = loraRoofline.points.find(p => p.isAggregate)!;
  const qloraAgg = qloraRoofline.points.find(p => p.isAggregate)!;

  it('ordering: full < LoRA < QLoRA (adapter-only grad/opt dominates)', () => {
    // LoRA eliminates ~99.7% of grad+opt bytes (adapter params are tiny).
    // This byte reduction outweighs the 4/6 FLOPs reduction → LoRA AI > full.
    // QLoRA additionally uses NF4 weights (~4× smaller) → highest AI.
    expect(fullAgg.arithmeticIntensity).toBeLessThan(loraAgg.arithmeticIntensity);
    expect(loraAgg.arithmeticIntensity).toBeLessThan(qloraAgg.arithmeticIntensity);
  });

  it('LoRA AI ~1.3-1.5× full AI on FSDP (grad/opt savings > 4/6 FLOPs loss)', () => {
    const ratio = loraAgg.arithmeticIntensity / fullAgg.arithmeticIntensity;
    expect(ratio).toBeGreaterThan(1.3);
    expect(ratio).toBeLessThan(1.5);
  });

  it('QLoRA AI ~4-5× full AI on FSDP (NF4 weights + tiny grad/opt)', () => {
    const ratio = qloraAgg.arithmeticIntensity / fullAgg.arithmeticIntensity;
    expect(ratio).toBeGreaterThan(4.0);
    expect(ratio).toBeLessThan(5.5);
  });

  it('full finetuningMethod=undefined behaves same as full', () => {
    const defaultRoofline = computeTrainingRoofline(model, h100, 'bf16', metrics, {
      ...baseConfig, finetuningMethod: undefined,
    });
    const defaultAgg = defaultRoofline.points.find(p => p.isAggregate)!;
    expect(defaultAgg.arithmeticIntensity).toBe(fullAgg.arithmeticIntensity);
  });

  it('DDP (unsharded): full < LoRA < QLoRA', () => {
    // Without FSDP sharding, weight bytes dominate the denominator even more.
    // Same ordering holds: adapter-only grad/opt shrinks denominator for LoRA,
    // QLoRA additionally compresses weights with NF4.
    const ddpConfig = {
      tp: 1, pp: 1, dp: 1,
      seqLength: 2048, microBatchSize: 4, globalBatchSize: 4,
      activationCheckpointing: true, flashAttention: true, dpType: 'ddp' as const,
    };

    const ddpFull = computeTrainingRoofline(model, h100, 'bf16', metrics, ddpConfig);
    const ddpLoRA = computeTrainingRoofline(model, h100, 'bf16', metrics, {
      ...ddpConfig, finetuningMethod: 'lora',
    });
    const ddpQLoRA = computeTrainingRoofline(model, h100, 'bf16', metrics, {
      ...ddpConfig, finetuningMethod: 'qlora',
    });

    const ddpFullAgg = ddpFull.points.find(p => p.isAggregate)!;
    const ddpLoRAAgg = ddpLoRA.points.find(p => p.isAggregate)!;
    const ddpQLoRAAgg = ddpQLoRA.points.find(p => p.isAggregate)!;

    // Ordering: full < LoRA < QLoRA
    expect(ddpFullAgg.arithmeticIntensity).toBeLessThan(ddpLoRAAgg.arithmeticIntensity);
    expect(ddpLoRAAgg.arithmeticIntensity).toBeLessThan(ddpQLoRAAgg.arithmeticIntensity);

    // DDP QLoRA AI > FSDP QLoRA AI (unsharded weights are larger fraction of bytes)
    expect(ddpQLoRAAgg.arithmeticIntensity).toBeGreaterThan(qloraAgg.arithmeticIntensity);
  });
});

// ---------------------------------------------------------------------------
// 5j. Cross-GPU validation
// ---------------------------------------------------------------------------

describe('Cross-GPU ridge points', () => {
  const gpuIds = ['a100-40gb', 'a100-80gb', 'h100-sxm', 'h200-sxm', 'b200', 'mi300x', 'mi250x', 't4'];

  for (const id of gpuIds) {
    it(`${id}: valid ceiling (ridgePoint > 0, peakTFLOPS > 0, peakBW > 0)`, () => {
      const gpu = ALL_GPUS[id];
      expect(gpu).toBeDefined();
      const dtype = gpu.bf16TFLOPS > 0 ? 'bf16' : 'fp16';
      const c = computeCeiling(gpu, dtype);
      expect(c.peakComputeTFLOPS).toBeGreaterThan(0);
      expect(c.peakBandwidthTBps).toBeGreaterThan(0);
      expect(c.ridgePoint).toBeGreaterThan(0);
    });
  }
});
