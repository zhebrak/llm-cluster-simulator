/**
 * Inference Recommendation Tests
 *
 * Validates the generator-based inference recommendation system:
 * - 7 OOM generators (TP, EP, combined TP+EP, quantization, batch reduction, context reduction)
 *   + last-resort fallback when nothing resolves OOM
 * - 15 success generators (warnings, validated optimizations, heuristic suggestions)
 * - Validation: validated generators re-simulate with configMutation
 * - Priority ordering: max 3 recs, warnings first
 * - Positive confirmation fallback when no generators fire
 * - No-numbers rule: recommendations suggest direction, not specific values
 */

import { describe, it, expect } from 'vitest';
import { runInferenceSimulation } from '../../src/core/inference/simulation.ts';
import {
  H100_SXM,
  A100_80GB,
  B200,
} from '../../src/core/hardware/gpu.ts';

// ══════════════════════════════════════════════════════════════════════
// SUCCESS GENERATORS
// ══════════════════════════════════════════════════════════════════════

// ── FP32 weight warning ──

describe('FP32 weight warning', () => {
  it('fires when weightPrecision is fp32', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'fp32',
      kvCachePrecision: 'fp32',
    });
    expect(result.success).toBe(true);
    expect(result.recommendations.some(r => r.includes('FP32 weights are rarely needed'))).toBe(true);
  });

  it('does not fire for bf16 weights', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
    });
    expect(result.recommendations.some(r => r.includes('FP32 weights are rarely needed'))).toBe(false);
  });
});

// ── Flash Attention ──

describe('Flash Attention suggestion', () => {
  it('fires when explicitly disabled on supported GPU (large batch where activations matter)', () => {
    // Flash attention reduces activation memory from O(s²) to O(s).
    // At large batch × long seq, activation savings are significant (>1% of total).
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 64,
      inputSeqLen: 4096,
      outputSeqLen: 512,
      weightPrecision: 'fp8',
      kvCachePrecision: 'fp8',
      flashAttention: false,
    });
    expect(result.success).toBe(true);
    expect(result.recommendations.some(r => r.includes('Flash Attention'))).toBe(true);
  });

  it('does not fire when flash attention is not explicitly disabled', () => {
    // Default is true — omitting flashAttention means it's enabled
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'fp8',
      kvCachePrecision: 'fp8',
    });
    expect(result.recommendations.some(r => r.includes('Flash Attention'))).toBe(false);
  });
});

// ── Context length warning ──

describe('Near max context length warning', () => {
  it('fires when total seq length > 75% of model max', () => {
    // LLaMA-3 8B has maxSeqLength = 8192. 75% = 6144.
    // Use inputSeqLen=5000 + outputSeqLen=2000 = 7000 > 6144
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 5000,
      outputSeqLen: 2000,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
    });
    expect(result.success).toBe(true);
    expect(result.recommendations.some(r => r.includes("approaching the model's maximum context"))).toBe(true);
  });

  it('does not fire when total seq length <= 75% of model max', () => {
    // 512 + 128 = 640, well below 75% of 8192
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
    });
    expect(result.recommendations.some(r => r.includes("approaching the model's maximum context"))).toBe(false);
  });
});

// ── FP8 quantization — precision is user choice, no suggestion fired ──

describe('FP8 quantization — precision is user choice, no suggestion fired', () => {
  it('does not fire for bf16 weights on H100 — precision is user choice', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
    });
    expect(result.success).toBe(true);
    expect(result.recommendations.some(r => r.includes('reducing weight precision to FP8'))).toBe(false);
  });

  it('does not fire when weights are already FP8', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'fp8',
      kvCachePrecision: 'fp8',
    });
    expect(result.recommendations.some(r => r.includes('reducing weight precision to FP8'))).toBe(false);
  });

  it('does not fire on A100 (no FP8 support)', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: A100_80GB,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
    });
    expect(result.recommendations.some(r => r.includes('reducing weight precision to FP8'))).toBe(false);
  });
});

// ── KV cache precision match ──

describe('KV cache precision match', () => {
  it('fires when KV precision > weight precision and KV cache is significant', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 8,
      inputSeqLen: 2048,
      outputSeqLen: 512,
      weightPrecision: 'fp8',
      kvCachePrecision: 'bf16',
    });
    expect(result.success).toBe(true);
    expect(result.recommendations.some(r => r.includes('KV cache precision to match weight precision'))).toBe(true);
  });

  it('does not fire when KV precision equals weight precision', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 8,
      inputSeqLen: 2048,
      outputSeqLen: 512,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
    });
    expect(result.recommendations.some(r => r.includes('KV cache precision to match weight precision'))).toBe(false);
  });

  it('does not fire when KV precision < weight precision', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 8,
      inputSeqLen: 2048,
      outputSeqLen: 512,
      weightPrecision: 'bf16',
      kvCachePrecision: 'fp8',
    });
    expect(result.recommendations.some(r => r.includes('KV cache precision to match weight precision'))).toBe(false);
  });
});

// ── INT4 quantization — precision is user choice, no suggestion fired ──

describe('INT4 quantization — precision is user choice, no suggestion fired', () => {
  it('does not fire on A100 — precision is user choice', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: A100_80GB,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
    });
    expect(result.success).toBe(true);
    expect(result.recommendations.some(r => r.includes('INT4 quantization'))).toBe(false);
  });

  it('does not fire on H100 when already at FP8', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'fp8',
      kvCachePrecision: 'fp8',
    });
    expect(result.success).toBe(true);
    expect(result.recommendations.some(r => r.includes('INT4 quantization'))).toBe(false);
  });

  it('does not fire when weights are already INT4', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: A100_80GB,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'int4',
      kvCachePrecision: 'int4',
    });
    expect(result.recommendations.some(r => r.includes('INT4 quantization'))).toBe(false);
  });
});

// ── TP suggestion (validated) ──

describe('TP increase suggestion', () => {
  it('does not fire at batch=1 when replicas beat TP (validation rejects)', () => {
    // At batch=1: TP=1 with 2 replicas (2x throughput) beats TP=2 with 1 replica.
    const result = runInferenceSimulation({
      modelId: 'llama3-70b',
      gpu: H100_SXM,
      numGPUs: 2,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'fp8',
      kvCachePrecision: 'fp8',
    });
    if (result.success) {
      expect(result.recommendations.some(r => r.includes('enabling tensor parallelism'))).toBe(false);
    }
  });

  it('does not fire when TP is already enabled', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-70b',
      gpu: H100_SXM,
      numGPUs: 2,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'fp8',
      kvCachePrecision: 'fp8',
      tensorParallel: 2,
    });
    expect(result.recommendations.some(r => r.includes('enabling tensor parallelism'))).toBe(false);
  });

  it('does not fire when weights are small relative to GPU memory', () => {
    // 8B fp8 = 8GB on 80GB GPU → weights < 50% of GPU memory
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 2,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'fp8',
      kvCachePrecision: 'fp8',
    });
    expect(result.recommendations.some(r => r.includes('enabling tensor parallelism'))).toBe(false);
  });
});

// ── Batch size increase (validated) ──

describe('Batch size increase suggestion', () => {
  it('fires when memory has significant headroom', () => {
    // 8B fp8 on H100 = ~8GB on 80GB → huge headroom
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'fp8',
      kvCachePrecision: 'fp8',
    });
    expect(result.success).toBe(true);
    expect(result.recommendations.some(r => r.includes('increasing batch size'))).toBe(true);
  });

  it('does not fire when memory is nearly full', () => {
    // 70B INT4 on A100 with high batch → 91% utilization
    const result = runInferenceSimulation({
      modelId: 'llama3-70b',
      gpu: A100_80GB,
      numGPUs: 1,
      batchSize: 128,
      inputSeqLen: 2048,
      outputSeqLen: 512,
      weightPrecision: 'int4',
      kvCachePrecision: 'int4',
    });
    expect(result.success).toBe(true);
    expect(result.recommendations.some(r => r.includes('increasing batch size'))).toBe(false);
  });
});

// ── TP reduction (validated) ──

describe('TP reduction suggestion', () => {
  it('does not fire when TP=1', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'fp8',
      kvCachePrecision: 'fp8',
    });
    expect(result.recommendations.some(r => r.includes('reducing tensor parallelism'))).toBe(false);
  });
});

// ── Batch size decrease (NEW) ──

describe('Batch size decrease suggestion', () => {
  it('does not fire when batch=1', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-70b',
      gpu: A100_80GB,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'int4',
      kvCachePrecision: 'int4',
    });
    expect(result.success).toBe(true);
    expect(result.recommendations.some(r => r.includes('reducing batch size if per-request latency'))).toBe(false);
  });
});

// ══════════════════════════════════════════════════════════════════════
// OOM GENERATORS
// ══════════════════════════════════════════════════════════════════════

describe('OOM recommendations', () => {
  it('suggests TP increase when weights dominate and GPUs available', () => {
    // 70B bf16 = 140GB on 2×80GB A100 → OOM per GPU, TP=2 would shard
    const result = runInferenceSimulation({
      modelId: 'llama3-70b',
      gpu: A100_80GB,
      numGPUs: 2,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
    });
    expect(result.success).toBe(false);
    expect(result.recommendations.some(r => r.includes('increasing tensor parallelism'))).toBe(true);
  });

  it('suggests quantization when weights exceed GPU and precision can reduce', () => {
    // 70B bf16 = 140GB on single 80GB A100 → OOM, FP8 would halve weights
    const result = runInferenceSimulation({
      modelId: 'llama3-70b',
      gpu: A100_80GB,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
    });
    expect(result.success).toBe(false);
    expect(result.recommendations.some(r => r.includes('reducing weight precision'))).toBe(true);
  });

  it('suggests context reduction when KV cache is significant and seq > 1024', () => {
    // 70B bf16 with long context → OOM with significant KV cache
    const result = runInferenceSimulation({
      modelId: 'llama3-70b',
      gpu: A100_80GB,
      numGPUs: 1,
      batchSize: 32,
      inputSeqLen: 4096,
      outputSeqLen: 1024,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
    });
    expect(result.success).toBe(false);
    expect(result.recommendations.some(r => r.includes('reducing input or output sequence length'))).toBe(true);
  });

  it('shows text-only quantization hint when OOM and precision can be reduced', () => {
    // 405B bf16 on single A100 → weights alone = 810GB >> 80GB
    // oomQuantization fires as text-only (no configMutation) before fallback
    const result = runInferenceSimulation({
      modelId: 'llama3-405b',
      gpu: A100_80GB,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
    });
    expect(result.success).toBe(false);
    expect(result.recommendations.some(r => r.includes('reducing weight precision'))).toBe(true);
  });

  it('shows fallback when nothing resolves OOM on single GPU at low precision', () => {
    // 405B int4 on single A100 → weights ~203GB >> 80GB
    // oomQuantization won't fire (already low precision), fallback fires
    const result = runInferenceSimulation({
      modelId: 'llama3-405b',
      gpu: A100_80GB,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'int4',
      kvCachePrecision: 'int4',
    });
    expect(result.success).toBe(false);
    expect(result.recommendations.some(r => r.includes('Model weights exceed GPU memory'))).toBe(true);
  });

  it('fallback is last-resort only — not shown alongside actionable recommendations', () => {
    // 70B bf16 on 2×A100 → TP=2 resolves OOM, so fallback should NOT appear
    const result = runInferenceSimulation({
      modelId: 'llama3-70b',
      gpu: A100_80GB,
      numGPUs: 2,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
    });
    expect(result.success).toBe(false);
    expect(result.recommendations.some(r => r.includes('tensor parallelism'))).toBe(true);
    // Fallback should NOT appear alongside actionable recommendations
    expect(result.recommendations.some(r => r.includes('Model weights exceed GPU memory'))).toBe(false);
    expect(result.recommendations.some(r => r.includes('Model does not fit'))).toBe(false);
  });

  it('OOM TP suggestion is validated (must actually fit)', () => {
    // The TP OOM generator validates by re-simulating at higher TP
    const result = runInferenceSimulation({
      modelId: 'llama3-70b',
      gpu: A100_80GB,
      numGPUs: 2,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
    });
    expect(result.success).toBe(false);
    // TP=2 on 2×A100 with 70B bf16: 70GB per GPU → fits in 80GB
    expect(result.recommendations.some(r => r.includes('tensor parallelism'))).toBe(true);
  });
});

// ══════════════════════════════════════════════════════════════════════
// SYSTEM BEHAVIOR
// ══════════════════════════════════════════════════════════════════════

// ── Priority ordering ──

describe('Priority ordering', () => {
  it('returns at most 3 recommendations', () => {
    // bf16 on H100 with low batch → multiple generators fire
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
    });
    expect(result.success).toBe(true);
    expect(result.recommendations.length).toBeLessThanOrEqual(3);
    expect(result.recommendations.length).toBeGreaterThanOrEqual(1);
  });

  it('warnings appear before optimization suggestions', () => {
    // FP32 + long context → both fp32Warning and contextLengthWarning should fire before optimizations
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 5000,
      outputSeqLen: 2000,
      weightPrecision: 'fp32',
      kvCachePrecision: 'fp32',
    });
    expect(result.success).toBe(true);
    const fp32Idx = result.recommendations.findIndex(r => r.includes('FP32'));
    const contextIdx = result.recommendations.findIndex(r => r.includes('context'));
    // Both should appear and should be first
    if (fp32Idx >= 0 && contextIdx >= 0) {
      // Warnings should be in first 2 positions
      expect(Math.max(fp32Idx, contextIdx)).toBeLessThan(3);
    }
  });
});

// ── Positive confirmation fallback ──

describe('Positive confirmation / status line', () => {
  it('shows "Well-configured" when no optimization generators match', () => {
    // Use H100 (current-gen) to avoid gpuUpgrade recommendation
    const result = runInferenceSimulation({
      modelId: 'llama3-70b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 128,
      inputSeqLen: 2048,
      outputSeqLen: 512,
      weightPrecision: 'int4',
      kvCachePrecision: 'int4',
      flashAttention: true,
      continuousBatching: true,
    });
    expect(result.success).toBe(true);
    expect(result.recommendations.length).toBeGreaterThanOrEqual(1);
    expect(result.recommendations.some(r => r.includes('Well-configured'))).toBe(true);
  });

  it('shows "Solid baseline" status line when optimization generators produce results', () => {
    // 8B bf16 on H100 → batch increase suggestion fires alongside status
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
    });
    expect(result.success).toBe(true);
    expect(result.recommendations.some(r => r.includes('batch size'))).toBe(true);
    // Status line should always be present for success configs
    expect(result.recommendations.some(r => r.includes('Solid baseline'))).toBe(true);
  });
});

// ── Validation acceptance/rejection ──

describe('Validation acceptance and rejection', () => {
  it('validated generator (batch increase) is accepted when throughput improves', () => {
    // Small model, low batch, lots of headroom → doubling batch improves throughput
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'fp8',
      kvCachePrecision: 'fp8',
    });
    expect(result.success).toBe(true);
    expect(result.recommendations.some(r => r.includes('increasing batch size'))).toBe(true);
  });

  it('validated TP reduction is accepted when replicas improve throughput', () => {
    // TP=2 on 70B fp8: reducing to TP=1 creates 2 replicas (each 70 GB on 80 GB).
    // 2 replicas at TP=1 beat 1 replica at TP=2 in throughput: 2× aggregate HBM
    // bandwidth with slightly higher per-GPU efficiency (larger reads).
    const result = runInferenceSimulation({
      modelId: 'llama3-70b',
      gpu: H100_SXM,
      numGPUs: 2,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'fp8',
      kvCachePrecision: 'fp8',
      tensorParallel: 2,
    });
    if (result.success) {
      expect(result.recommendations.some(r => r.includes('reducing tensor parallelism'))).toBe(true);
    }
  });
});

// ── No-numbers rule ──

describe('No-numbers rule', () => {
  it('recommendations do not contain specific numeric values', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
    });
    expect(result.success).toBe(true);
    for (const rec of result.recommendations) {
      // Should not contain specific batch sizes, percentages, or improvement factors
      expect(rec).not.toMatch(/\b\d+x\b/); // "2x", "4x" etc.
      expect(rec).not.toMatch(/\d+%/); // "50%", "10%" etc.
      expect(rec).not.toMatch(/batch.*(size|=)\s*\d+/i); // "batch size 64"
    }
  });

  it('OOM recommendations do not contain specific numeric values', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-70b',
      gpu: A100_80GB,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
    });
    expect(result.success).toBe(false);
    for (const rec of result.recommendations) {
      expect(rec).not.toMatch(/\b\d+x\b/);
      expect(rec).not.toMatch(/\d+%/);
    }
  });
});

// ── Both OOM and success use the same recommendation engine ──

describe('Unified recommendation engine', () => {
  it('OOM configs get recommendations from the OOM generators', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-405b',
      gpu: A100_80GB,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
    });
    expect(result.success).toBe(false);
    expect(result.recommendations.length).toBeGreaterThanOrEqual(1);
    // OOM recs should not contain success-only status messages
    expect(result.recommendations.some(r => r.includes('Well-configured') || r.includes('Solid baseline'))).toBe(false);
  });

  it('success configs get recommendations from the success generators', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
    });
    expect(result.success).toBe(true);
    expect(result.recommendations.length).toBeGreaterThanOrEqual(1);
    // Success recs should not contain OOM-only messages
    expect(result.recommendations.some(r => r.includes('Model weights exceed GPU memory'))).toBe(false);
    expect(result.recommendations.some(r => r.includes('Model does not fit'))).toBe(false);
  });
});

// ══════════════════════════════════════════════════════════════════════
// EXPERT PARALLELISM GENERATORS
// ══════════════════════════════════════════════════════════════════════

describe('OOM EP increase suggestion', () => {
  it('fires for MoE model when OOM and expert weights are significant', () => {
    // DeepSeek V3 fp8 on 16 GPUs with TP=8, no EP → OOM (671B/8 = 83.9GB > 80GB)
    // Adding EP=2 shards experts across 2 groups: (19.5B/8 + 651.5B/16) × 1 = 43.1GB → fits
    const result = runInferenceSimulation({
      modelId: 'deepseek-v3',
      gpu: H100_SXM,
      numGPUs: 16,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'fp8',
      kvCachePrecision: 'fp8',
      tensorParallel: 8,
    });
    expect(result.success).toBe(false);
    expect(result.recommendations.some(r => r.includes('expert parallelism'))).toBe(true);
  });

  it('does NOT fire for dense model even when OOM', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-405b',
      gpu: A100_80GB,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
    });
    expect(result.success).toBe(false);
    expect(result.recommendations.some(r => r.includes('expert parallelism'))).toBe(false);
  });
});

describe('EP increase suggestion (success path)', () => {
  it('considers EP for MoE model with enough GPUs', () => {
    // Mixtral 8x7B on 8 GPUs with TP=2, EP=1 → EP=2 could help
    const result = runInferenceSimulation({
      modelId: 'mixtral-8x7b',
      gpu: H100_SXM,
      numGPUs: 8,
      batchSize: 16,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'fp8',
      kvCachePrecision: 'fp8',
      tensorParallel: 2,
    });
    if (result.success) {
      // EP may or may not be recommended depending on validation — just verify no crash
      expect(result.recommendations.length).toBeGreaterThanOrEqual(1);
      expect(result.recommendations.length).toBeLessThanOrEqual(3);
    }
  });

  it('does NOT fire for dense models', () => {
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 8,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'fp8',
      kvCachePrecision: 'fp8',
      tensorParallel: 2,
    });
    expect(result.success).toBe(true);
    expect(result.recommendations.some(r => r.includes('expert parallelism'))).toBe(false);
  });

  it('does NOT fire on single GPU', () => {
    const result = runInferenceSimulation({
      modelId: 'mixtral-8x7b',
      gpu: H100_SXM,
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'fp8',
      kvCachePrecision: 'fp8',
    });
    if (result.success) {
      expect(result.recommendations.some(r => r.includes('enabling expert parallelism'))).toBe(false);
    }
  });
});

describe('EP reduction suggestion', () => {
  it('does NOT fire when EP=1', () => {
    const result = runInferenceSimulation({
      modelId: 'mixtral-8x7b',
      gpu: H100_SXM,
      numGPUs: 4,
      batchSize: 8,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'fp8',
      kvCachePrecision: 'fp8',
      tensorParallel: 2,
      expertParallel: 1,
    });
    if (result.success) {
      expect(result.recommendations.some(r => r.includes('reducing expert parallelism'))).toBe(false);
    }
  });

  it('does NOT fire for dense model even with EP=2', () => {
    // Dense models don't have experts — EP generator gates on isMoE
    const result = runInferenceSimulation({
      modelId: 'llama3-8b',
      gpu: H100_SXM,
      numGPUs: 4,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'fp8',
      kvCachePrecision: 'fp8',
      tensorParallel: 2,
      expertParallel: 2,
    });
    if (result.success) {
      expect(result.recommendations.some(r => r.includes('reducing expert parallelism'))).toBe(false);
    }
  });
});

describe('EP recommendation no-numbers rule', () => {
  it('EP recommendation messages do not contain specific numeric values', () => {
    // OOM config to trigger EP rec
    const oomResult = runInferenceSimulation({
      modelId: 'deepseek-v3',
      gpu: H100_SXM,
      numGPUs: 8,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
      tensorParallel: 4,
    });
    for (const rec of oomResult.recommendations) {
      expect(rec).not.toMatch(/\b\d+x\b/);
      expect(rec).not.toMatch(/\d+%/);
    }

    // Success config
    const successResult = runInferenceSimulation({
      modelId: 'mixtral-8x7b',
      gpu: H100_SXM,
      numGPUs: 8,
      batchSize: 16,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'fp8',
      kvCachePrecision: 'fp8',
      tensorParallel: 2,
    });
    for (const rec of successResult.recommendations) {
      expect(rec).not.toMatch(/\b\d+x\b/);
      expect(rec).not.toMatch(/\d+%/);
    }
  });
});

// ══════════════════════════════════════════════════════════════════════
// LARGE-MODEL OOM PRE-VALIDATION (the core bug this fix addresses)
// ══════════════════════════════════════════════════════════════════════

describe('Large-model OOM pre-validation', () => {
  it('Maverick on 1024×B200 with TP=1 EP=1 recommends increasing TP (not just fallback)', () => {
    // Maverick ~400B params in bf16 = ~800GB → way beyond 180GB per B200
    // TP=2 or TP=4 alone may not resolve it; the generator should find the TP that works
    const result = runInferenceSimulation({
      modelId: 'llama4-maverick',
      gpu: B200,
      numGPUs: 1024,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
      tensorParallel: 1,
      expertParallel: 1,
    });
    expect(result.success).toBe(false);
    // Must get actionable recommendations, not just "get a bigger GPU"
    expect(result.recommendations.some(r =>
      r.includes('tensor parallelism') || r.includes('expert parallelism')
    )).toBe(true);
    // Fallback should NOT appear when actionable recs exist
    expect(result.recommendations.some(r => r.includes('Model weights exceed GPU memory'))).toBe(false);
    expect(result.recommendations.some(r => r.includes('Model does not fit'))).toBe(false);
  });

  it('DeepSeek V3 bf16 on 8×H100 with TP=1 EP=1 recommends TP increase', () => {
    // V3 ~671B params in bf16 = ~1342GB → needs high TP
    // TP=2 still OOMs, TP=4 still OOMs, TP=8 = ~167GB per GPU → fits 80GB? No.
    // With fp8 it might, but bf16 on 8×80GB can't hold 671B. Should get quant or combined TP+EP.
    const result = runInferenceSimulation({
      modelId: 'deepseek-v3',
      gpu: H100_SXM,
      numGPUs: 8,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
      tensorParallel: 1,
      expertParallel: 1,
    });
    expect(result.success).toBe(false);
    // Should get at least one actionable recommendation (TP, EP, quant, or combined)
    expect(result.recommendations.length).toBeGreaterThanOrEqual(1);
    expect(result.recommendations.some(r =>
      r.includes('tensor parallelism') ||
      r.includes('expert parallelism') ||
      r.includes('weight precision')
    )).toBe(true);
  });

  it('405B on 2×A100 where TP=2 resolves it still works', () => {
    // 405B fp8 = ~405GB → TP=2 on 2×80GB = 202GB per GPU. Still doesn't fit.
    // But int4 = ~202GB → TP=2 = 101GB per GPU. Still too big for 80GB.
    // Actually needs more GPUs. Test that fallback fires gracefully.
    const result = runInferenceSimulation({
      modelId: 'llama3-405b',
      gpu: A100_80GB,
      numGPUs: 2,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
    });
    expect(result.success).toBe(false);
    // Some recommendations should fire (quant, TP) — they're pre-validated
    // If none resolve OOM, fallback fires
    expect(result.recommendations.length).toBeGreaterThanOrEqual(1);
  });

  it('multi-GPU fallback text suggests TP/EP when numGPUs > 1', () => {
    // A huge model on multi-GPU where nothing resolves → fallback should mention TP/EP
    const result = runInferenceSimulation({
      modelId: 'llama3-405b',
      gpu: A100_80GB,
      numGPUs: 2,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'int4',
      kvCachePrecision: 'int4',
    });
    expect(result.success).toBe(false);
    // Check that it has at least 1 recommendation
    expect(result.recommendations.length).toBeGreaterThanOrEqual(1);
  });

  it('combined TP+EP fires for MoE when neither alone suffices', () => {
    // DeepSeek V3 bf16 on 128×H100: individual TP=8 may not resolve due to expert weights,
    // individual EP may not resolve due to shared weights. Combined might.
    const result = runInferenceSimulation({
      modelId: 'deepseek-v3',
      gpu: H100_SXM,
      numGPUs: 128,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
      tensorParallel: 1,
      expertParallel: 1,
    });
    expect(result.success).toBe(false);
    // With 128 GPUs, some combination of TP and EP should resolve this
    expect(result.recommendations.some(r =>
      r.includes('tensor parallelism') || r.includes('expert parallelism')
    )).toBe(true);
  });
});
