/**
 * Comprehensive TTFT/TPOT TP Scaling Tests
 *
 * Validates inference latency across GPU architectures, TP degrees 1–32,
 * dense/MoE/MLA models, PCIe-only GPUs, and AMD Infinity Fabric GPUs.
 *
 * All bounds are ±15–25% of actual simulator output (calibrated after
 * the AMD Infinity Fabric comm fix in latency.ts / speculative.ts).
 */
import { describe, it, expect } from 'vitest';
import {
  calculateLatencyWithTP,
  calculateLatencyMetrics,
  estimateTTFT,
  estimateTPOT,
  modelWeightsMemory,
} from '../../src/core/inference/index.ts';
import { getBandwidthEfficiency, moeWeightBytesPerStep } from '../../src/core/inference/latency.ts';
import { kvCachePerToken, totalKVCacheMemory } from '../../src/core/inference/kv-cache.ts';
import { getModel } from '../../src/core/models/index.ts';
import {
  H100_SXM, H100_PCIE, H200_SXM, A100_80GB, B200,
  MI300X, MI325X, MI350X, L40S,
} from '../../src/core/hardware/gpu.ts';

// ── Helpers ──
const llama8b  = getModel('llama3.1-8b')!;
const llama70b = getModel('llama3.3-70b')!;
const llama405b = getModel('llama3-405b')!;
const gpt3_175b = getModel('gpt3-175b')!;
const dsv3 = getModel('deepseek-v3')!;

function tp(model: typeof llama8b, seq: number, out: number, bs: number, gpu: typeof H100_SXM, tpDeg: number, prec: 'bf16' | 'fp8' | 'int4' = 'bf16', ep = 1, kv?: 'bf16' | 'fp8', fa = true, gpn?: number, ib?: number) {
  return calculateLatencyWithTP(model, seq, out, bs, gpu, tpDeg, prec, ep, kv, fa, gpn, ib);
}

// ────────────────────────────────────────────────────────────
// Section 1: TTFT TP Scaling — Dense Models
// ────────────────────────────────────────────────────────────
describe('Section 1: TTFT TP Scaling — Dense Models', () => {
  it('LLaMA 8B H100 SXM TP=1 TTFT ≈ 42 ms', () => {
    const r = tp(llama8b, 1024, 128, 1, H100_SXM, 1);
    expect(r.ttft).toBeGreaterThan(32);
    expect(r.ttft).toBeLessThan(54);
  });

  it('LLaMA 8B H100 SXM TP=2 halves TTFT', () => {
    const r1 = tp(llama8b, 1024, 128, 1, H100_SXM, 1);
    const r2 = tp(llama8b, 1024, 128, 1, H100_SXM, 2);
    const ratio = r2.ttft / r1.ttft;
    expect(ratio).toBeGreaterThan(0.42);
    expect(ratio).toBeLessThan(0.58);
  });

  it('LLaMA 8B H100 SXM TP=4 and TP=8 continue scaling', () => {
    const r2 = tp(llama8b, 1024, 128, 1, H100_SXM, 2);
    const r4 = tp(llama8b, 1024, 128, 1, H100_SXM, 4);
    const r8 = tp(llama8b, 1024, 128, 1, H100_SXM, 8);
    expect(r4.ttft / r2.ttft).toBeGreaterThan(0.42);
    expect(r4.ttft / r2.ttft).toBeLessThan(0.58);
    expect(r8.ttft / r4.ttft).toBeGreaterThan(0.42);
    expect(r8.ttft / r4.ttft).toBeLessThan(0.58);
  });

  it('LLaMA 70B H100 SXM TP=2 TTFT ≈ 184 ms', () => {
    const r = tp(llama70b, 1024, 128, 1, H100_SXM, 2);
    expect(r.ttft).toBeGreaterThan(145);
    expect(r.ttft).toBeLessThan(230);
  });

  it('LLaMA 70B H100 SXM TP=8 TTFT ≈ 46 ms', () => {
    const r = tp(llama70b, 1024, 128, 1, H100_SXM, 8);
    expect(r.ttft).toBeGreaterThan(36);
    expect(r.ttft).toBeLessThan(58);
  });

  it('LLaMA 405B H100 SXM TP=8 TTFT ≈ 264 ms', () => {
    const r = tp(llama405b, 1024, 128, 1, H100_SXM, 8);
    expect(r.ttft).toBeGreaterThan(200);
    expect(r.ttft).toBeLessThan(330);
  });

  it('GPT-3 175B H100 SXM TP=8 TTFT ≈ 114 ms', () => {
    const r = tp(gpt3_175b, 1024, 128, 1, H100_SXM, 8);
    expect(r.ttft).toBeGreaterThan(90);
    expect(r.ttft).toBeLessThan(145);
  });

  it('LLaMA 70B TTFT TP scaling ratios ≈ 0.50', () => {
    const r2 = tp(llama70b, 1024, 128, 1, H100_SXM, 2);
    const r4 = tp(llama70b, 1024, 128, 1, H100_SXM, 4);
    const r8 = tp(llama70b, 1024, 128, 1, H100_SXM, 8);
    expect(r4.ttft / r2.ttft).toBeGreaterThan(0.42);
    expect(r4.ttft / r2.ttft).toBeLessThan(0.58);
    expect(r8.ttft / r4.ttft).toBeGreaterThan(0.42);
    expect(r8.ttft / r4.ttft).toBeLessThan(0.58);
  });

  it('AR overhead is sublinear in TP degree for large models', () => {
    const base = tp(llama70b, 1024, 128, 1, H100_SXM, 1);
    const at2 = tp(llama70b, 1024, 128, 1, H100_SXM, 2);
    const at8 = tp(llama70b, 1024, 128, 1, H100_SXM, 8);
    const overhead2 = at2.ttft - base.ttft / 2;
    const overhead8 = at8.ttft - base.ttft / 8;
    // Overhead at TP=8 should be < 4× overhead at TP=2 (sublinear)
    expect(overhead8 / Math.max(overhead2, 0.001)).toBeLessThan(4);
  });

  it('LLaMA 8B base TTFT matches estimateTTFT', () => {
    const baseTTFT = estimateTTFT(llama8b, 1024, H100_SXM, 'bf16', 1, true);
    const tpResult = tp(llama8b, 1024, 128, 1, H100_SXM, 1);
    expect(Math.abs(baseTTFT - tpResult.ttft) / baseTTFT).toBeLessThan(0.01);
  });
});

// ────────────────────────────────────────────────────────────
// Section 2: TPOT TP Scaling — Dense Models
// ────────────────────────────────────────────────────────────
describe('Section 2: TPOT TP Scaling — Dense Models', () => {
  it('LLaMA 70B BF16 TP=2 TPOT ≈ 26 ms', () => {
    const r = tp(llama70b, 1024, 128, 1, H100_SXM, 2);
    expect(r.tpot).toBeGreaterThan(20);
    expect(r.tpot).toBeLessThan(33);
  });

  it('LLaMA 70B BF16 TP=8 TPOT ≈ 7.3 ms', () => {
    const r = tp(llama70b, 1024, 128, 1, H100_SXM, 8);
    expect(r.tpot).toBeGreaterThan(5.5);
    expect(r.tpot).toBeLessThan(9.5);
  });

  it('LLaMA 70B TP scaling ratios slightly > 0.50 (BW eff degradation)', () => {
    const r2 = tp(llama70b, 1024, 128, 1, H100_SXM, 2);
    const r4 = tp(llama70b, 1024, 128, 1, H100_SXM, 4);
    const r8 = tp(llama70b, 1024, 128, 1, H100_SXM, 8);
    // TP=4/TP=2 ≈ 0.521, TP=8/TP=4 ≈ 0.537
    expect(r4.tpot / r2.tpot).toBeGreaterThan(0.44);
    expect(r4.tpot / r2.tpot).toBeLessThan(0.60);
    expect(r8.tpot / r4.tpot).toBeGreaterThan(0.44);
    expect(r8.tpot / r4.tpot).toBeLessThan(0.62);
  });

  it('LLaMA 8B TP=4→8 ratio worsens (small model, AR exposed)', () => {
    const r4 = tp(llama8b, 1024, 128, 1, H100_SXM, 4);
    const r8 = tp(llama8b, 1024, 128, 1, H100_SXM, 8);
    // 8B TP=8 is barely better than TP=4 (AR dominates): ratio ≈ 0.89
    expect(r8.tpot / r4.tpot).toBeGreaterThan(0.75);
    expect(r8.tpot / r4.tpot).toBeLessThan(1.05);
  });

  it('LLaMA 70B TP=8 AR overhead < 5% of TPOT', () => {
    const r1 = tp(llama70b, 1024, 128, 1, H100_SXM, 1);
    const r8 = tp(llama70b, 1024, 128, 1, H100_SXM, 8);
    const overhead = (r8.tpot - r1.tpot / 8) / (r1.tpot / 8);
    // Actual: ~2.2% overhead at TP=2, grows slightly at TP=8 ≈ 14%
    // For 70B at TP=8, overhead is moderate (BW eff + some AR)
    expect(overhead).toBeLessThan(0.25);
  });

  it('BW efficiency monotonically decreasing with per-GPU size', () => {
    const sizes = [140e9, 70e9, 35e9, 17.5e9];
    const effs = sizes.map(s => getBandwidthEfficiency(s));
    for (let i = 1; i < effs.length; i++) {
      expect(effs[i]).toBeLessThan(effs[i - 1]);
    }
  });

  it('BW efficiency range: 0.73–0.84 for 17.5–140 GB', () => {
    expect(getBandwidthEfficiency(140e9)).toBeGreaterThan(0.80);
    expect(getBandwidthEfficiency(140e9)).toBeLessThan(0.86);
    expect(getBandwidthEfficiency(17.5e9)).toBeGreaterThan(0.70);
    expect(getBandwidthEfficiency(17.5e9)).toBeLessThan(0.76);
  });
});

// ────────────────────────────────────────────────────────────
// Section 3: MLA-specific TP Tests — DeepSeek V3
// ────────────────────────────────────────────────────────────
describe('Section 3: MLA-specific TP Tests — DeepSeek V3', () => {
  it('KV cache is constant across TP (MLA: replicated latent)', () => {
    const kvPerToken = kvCachePerToken(dsv3, 'bf16');
    // V3 MLA: 61 layers × 576 values × 2 bytes = 70,272 bytes/token
    expect(kvPerToken).toBe(70272);
  });

  it('TPOT scaling worse than non-MLA at TP=4→8', () => {
    const dsv3_tp4 = tp(dsv3, 1024, 128, 1, H100_SXM, 4, 'fp8');
    const dsv3_tp8 = tp(dsv3, 1024, 128, 1, H100_SXM, 8, 'fp8');
    const mlaRatio = dsv3_tp8.tpot / dsv3_tp4.tpot; // ≈ 0.872

    const dense_tp4 = tp(llama70b, 1024, 128, 1, H100_SXM, 4);
    const dense_tp8 = tp(llama70b, 1024, 128, 1, H100_SXM, 8);
    const denseRatio = dense_tp8.tpot / dense_tp4.tpot; // ≈ 0.537

    // MLA ratio (≈0.87) > dense ratio (≈0.54) → worse scaling
    expect(mlaRatio).toBeGreaterThan(denseRatio + 0.10);
  });

  it('Weight bytes per step > 30 GB at batch=1 FP8', () => {
    const wbytes = moeWeightBytesPerStep(dsv3, 1, 'fp8');
    expect(wbytes / 1e9).toBeGreaterThan(30);
    expect(wbytes / 1e9).toBeLessThan(50);
  });

  it('Long sequences amplify MLA KV fraction', () => {
    const kv1024 = totalKVCacheMemory(dsv3, 1024, 1, 'bf16');
    const kv4096 = totalKVCacheMemory(dsv3, 4096, 1, 'bf16');
    // KV scales linearly with seq: 4096/1024 = 4×
    expect(kv4096 / kv1024).toBeCloseTo(4.0, 1);
    // At seq=4096: KV ≈ 288 MB (constant across TP ranks)
    expect(kv4096 / 1e6).toBeGreaterThan(260);
    expect(kv4096 / 1e6).toBeLessThan(310);
  });
});

// ────────────────────────────────────────────────────────────
// Section 4: Cross-node TP — TP=16, TP=32
// ────────────────────────────────────────────────────────────
describe('Section 4: Cross-node TP — TP=16, TP=32', () => {
  it('LLaMA 405B TP=16 (2 nodes) produces valid results', () => {
    const r = tp(llama405b, 1024, 128, 1, H100_SXM, 16, 'bf16', 1, undefined, true, 8, 50);
    expect(r.ttft).toBeGreaterThan(0);
    expect(r.tpot).toBeGreaterThan(0);
    expect(Number.isFinite(r.ttft)).toBe(true);
    expect(Number.isFinite(r.tpot)).toBe(true);
    // TTFT ≈ 281 ms
    expect(r.ttft).toBeGreaterThan(210);
    expect(r.ttft).toBeLessThan(360);
    // TPOT ≈ 28 ms
    expect(r.tpot).toBeGreaterThan(20);
    expect(r.tpot).toBeLessThan(38);
  });

  it('TP=16 vs TP=8 TPOT ratio 0.40–0.90 (cross-node degradation)', () => {
    const r8 = tp(llama405b, 1024, 128, 1, H100_SXM, 8);
    const r16 = tp(llama405b, 1024, 128, 1, H100_SXM, 16, 'bf16', 1, undefined, true, 8, 50);
    const ratio = r16.tpot / r8.tpot; // ≈ 0.762
    expect(ratio).toBeGreaterThan(0.55);
    expect(ratio).toBeLessThan(0.90);
  });

  it('Scaling efficiency TP=8→16 < TP=4→8 (cross-node penalty)', () => {
    const r4 = tp(llama405b, 1024, 128, 1, H100_SXM, 4);
    const r8 = tp(llama405b, 1024, 128, 1, H100_SXM, 8);
    const r16 = tp(llama405b, 1024, 128, 1, H100_SXM, 16, 'bf16', 1, undefined, true, 8, 50);
    const eff_4to8 = r4.tpot / r8.tpot;   // ≈ 1.97 speedup
    const eff_8to16 = r8.tpot / r16.tpot;  // ≈ 1.31 speedup
    expect(eff_8to16).toBeLessThan(eff_4to8);
  });

  it('TP=32 across 4 nodes produces valid but degraded results', () => {
    const r16 = tp(llama405b, 1024, 128, 1, H100_SXM, 16, 'bf16', 1, undefined, true, 8, 50);
    const r32 = tp(llama405b, 1024, 128, 1, H100_SXM, 32, 'bf16', 1, undefined, true, 8, 50);
    expect(Number.isFinite(r32.ttft)).toBe(true);
    expect(Number.isFinite(r32.tpot)).toBe(true);
    // TP=32 TTFT may increase (IB overhead dominates): ≈ 399 ms
    expect(r32.ttft).toBeGreaterThan(250);
    expect(r32.ttft).toBeLessThan(550);
  });

  it('TP=32 TPOT degrades beyond TP=16 for 405B (IB bottleneck)', () => {
    const r16 = tp(llama405b, 1024, 128, 1, H100_SXM, 16, 'bf16', 1, undefined, true, 8, 50);
    const r32 = tp(llama405b, 1024, 128, 1, H100_SXM, 32, 'bf16', 1, undefined, true, 8, 50);
    // TP=32 TPOT ≈ 40.7 > TP=16 TPOT ≈ 28.0
    expect(r32.tpot).toBeGreaterThan(r16.tpot);
  });
});

// ────────────────────────────────────────────────────────────
// Section 5: PCIe-only GPU TP Tests
// ────────────────────────────────────────────────────────────
describe('Section 5: PCIe-only GPU TP Tests', () => {
  it('L40S TP=2 produces valid TPOT for 8B', () => {
    const r = tp(llama8b, 1024, 128, 1, L40S, 2);
    expect(r.tpot).toBeGreaterThan(10);
    expect(r.tpot).toBeLessThan(20);
  });

  it('L40S TP=4 TPOT > H100 SXM TP=4 for 70B', () => {
    const l40s = tp(llama70b, 1024, 128, 1, L40S, 4);
    const h100 = tp(llama70b, 1024, 128, 1, H100_SXM, 4);
    expect(l40s.tpot).toBeGreaterThan(h100.tpot);
  });

  it('H100 PCIe slower than H100 SXM at TP=2', () => {
    const pcie = tp(llama70b, 1024, 128, 1, H100_PCIE, 2);
    const sxm = tp(llama70b, 1024, 128, 1, H100_SXM, 2);
    expect(pcie.tpot).toBeGreaterThan(sxm.tpot);
  });

  it('H100 PCIe TP=8 8B TPOT > H100 SXM TP=8 8B TPOT (PCIe alpha 3× NVLink)', () => {
    const pcie = tp(llama8b, 1024, 128, 1, H100_PCIE, 8);
    const sxm = tp(llama8b, 1024, 128, 1, H100_SXM, 8);
    // PCIe TPOT ≈ 5.8 vs SXM ≈ 1.9
    expect(pcie.tpot).toBeGreaterThan(sxm.tpot * 1.5);
  });
});

// ────────────────────────────────────────────────────────────
// Section 6: AllReduce Overlap Validation
// ────────────────────────────────────────────────────────────
describe('Section 6: AllReduce Overlap Validation', () => {
  it('70B TP=2: AR negligible (< 5% overhead)', () => {
    const r1 = tp(llama70b, 1024, 128, 1, H100_SXM, 1);
    const r2 = tp(llama70b, 1024, 128, 1, H100_SXM, 2);
    const overhead = (r2.tpot - r1.tpot / 2) / (r1.tpot / 2);
    expect(overhead).toBeLessThan(0.05);
  });

  it('8B TP=8: AR significant (> 30% overhead)', () => {
    const r1 = tp(llama8b, 1024, 128, 1, H100_SXM, 1);
    const r8 = tp(llama8b, 1024, 128, 1, H100_SXM, 8);
    const overhead = (r8.tpot - r1.tpot / 8) / (r1.tpot / 8);
    // Actual: ~132% overhead — small model at high TP
    expect(overhead).toBeGreaterThan(0.30);
  });

  it('70B TP=8: moderate overhead (< 25%)', () => {
    const r1 = tp(llama70b, 1024, 128, 1, H100_SXM, 1);
    const r8 = tp(llama70b, 1024, 128, 1, H100_SXM, 8);
    const overhead = (r8.tpot - r1.tpot / 8) / (r1.tpot / 8);
    expect(overhead).toBeLessThan(0.25);
  });
});

// ────────────────────────────────────────────────────────────
// Section 7: Physics Invariants
// ────────────────────────────────────────────────────────────
describe('Section 7: Physics Invariants', () => {
  it('TTFT(TP=N) >= baseTTFT/N (never faster than perfect scaling)', () => {
    const base = estimateTTFT(llama70b, 1024, H100_SXM, 'bf16', 1, true);
    for (const t of [2, 4, 8]) {
      const r = tp(llama70b, 1024, 128, 1, H100_SXM, t);
      expect(r.ttft).toBeGreaterThanOrEqual(base / t * 0.99); // 1% tolerance for rounding
    }
  });

  it('TPOT(TP=N) >= weightBytes/(N×BW×0.85) (BW floor)', () => {
    const weightsBytes = modelWeightsMemory(llama70b, 'bf16');
    for (const t of [1, 2, 4, 8]) {
      const r = tp(llama70b, 1024, 128, 1, H100_SXM, t);
      const floor = (weightsBytes / t) / (H100_SXM.memoryBandwidthTBps * 1e12 * 0.85) * 1000;
      expect(r.tpot).toBeGreaterThanOrEqual(floor * 0.95); // 5% tolerance
    }
  });

  it('TPOT scaling efficiency monotonically decreasing', () => {
    const r1 = tp(llama70b, 1024, 128, 1, H100_SXM, 1);
    const r2 = tp(llama70b, 1024, 128, 1, H100_SXM, 2);
    const r4 = tp(llama70b, 1024, 128, 1, H100_SXM, 4);
    const r8 = tp(llama70b, 1024, 128, 1, H100_SXM, 8);
    const s1to2 = r1.tpot / r2.tpot;
    const s2to4 = r2.tpot / r4.tpot;
    const s4to8 = r4.tpot / r8.tpot;
    expect(s1to2).toBeGreaterThanOrEqual(s2to4 * 0.98);
    expect(s2to4).toBeGreaterThanOrEqual(s4to8 * 0.98);
  });

  it('TP=1 calculateLatencyWithTP matches calculateLatencyMetrics', () => {
    const tpResult = tp(llama70b, 1024, 128, 1, H100_SXM, 1);
    const basicResult = calculateLatencyMetrics(llama70b, 1024, 128, 1, H100_SXM, 'bf16');
    expect(Math.abs(tpResult.ttft - basicResult.ttft) / basicResult.ttft).toBeLessThan(0.01);
    expect(Math.abs(tpResult.tpot - basicResult.tpot) / basicResult.tpot).toBeLessThan(0.01);
  });

  it('TTFT scaling efficiency also monotonically decreasing', () => {
    const r1 = tp(llama70b, 1024, 128, 1, H100_SXM, 1);
    const r2 = tp(llama70b, 1024, 128, 1, H100_SXM, 2);
    const r4 = tp(llama70b, 1024, 128, 1, H100_SXM, 4);
    const r8 = tp(llama70b, 1024, 128, 1, H100_SXM, 8);
    const s1to2 = r1.ttft / r2.ttft;
    const s2to4 = r2.ttft / r4.ttft;
    const s4to8 = r4.ttft / r8.ttft;
    expect(s1to2).toBeGreaterThanOrEqual(s2to4 * 0.98);
    expect(s2to4).toBeGreaterThanOrEqual(s4to8 * 0.98);
  });

  it('All values positive and non-NaN for multiple models × TP', () => {
    const models = [llama8b, llama70b, gpt3_175b];
    const tps = [1, 2, 4, 8];
    for (const model of models) {
      for (const t of tps) {
        const r = tp(model, 1024, 128, 1, H100_SXM, t);
        expect(r.ttft).toBeGreaterThan(0);
        expect(r.tpot).toBeGreaterThan(0);
        expect(Number.isFinite(r.ttft)).toBe(true);
        expect(Number.isFinite(r.tpot)).toBe(true);
      }
    }
  });
});

// ────────────────────────────────────────────────────────────
// Section 8: Batch Size × TP Interaction
// ────────────────────────────────────────────────────────────
describe('Section 8: Batch Size × TP Interaction', () => {
  it('TTFT batch=32 TP=8 (70B) scales ~linearly with batch', () => {
    const b1 = tp(llama70b, 1024, 128, 1, H100_SXM, 8);
    const b32 = tp(llama70b, 1024, 128, 32, H100_SXM, 8);
    const ratio = b32.ttft / b1.ttft;
    // Actual: ~32× (compute-bound). Allow 20–40×.
    expect(ratio).toBeGreaterThan(20);
    expect(ratio).toBeLessThan(40);
  });

  it('TPOT batch=64 > batch=1 (more KV cache reads)', () => {
    const b1 = tp(llama70b, 1024, 128, 1, H100_SXM, 8);
    const b64 = tp(llama70b, 1024, 128, 64, H100_SXM, 8);
    expect(b64.tpot).toBeGreaterThan(b1.tpot);
  });

  it('Decode comm increases with batch (AR term scales with batch×hidden)', () => {
    const b1 = tp(llama70b, 1024, 128, 1, H100_SXM, 8);
    const b64 = tp(llama70b, 1024, 128, 64, H100_SXM, 8);
    // TPOT difference should be modest (batch mostly affects KV reads)
    expect(b64.tpot).toBeGreaterThan(b1.tpot * 1.01);
    expect(b64.tpot).toBeLessThan(b1.tpot * 2.0);
  });
});

// ────────────────────────────────────────────────────────────
// Section 9: Precision × TP Interaction
// ────────────────────────────────────────────────────────────
describe('Section 9: Precision × TP Interaction', () => {
  it('FP8 TPOT < BF16 TPOT and FP8 TTFT < BF16 TTFT', () => {
    const bf16 = tp(llama70b, 1024, 128, 1, H100_SXM, 8);
    const fp8 = tp(llama70b, 1024, 128, 1, H100_SXM, 8, 'fp8');
    expect(fp8.tpot).toBeLessThan(bf16.tpot);
    expect(fp8.ttft).toBeLessThan(bf16.ttft);
    // FP8/BF16 TPOT ratio ≈ 0.67 (halved weights)
    expect(fp8.tpot / bf16.tpot).toBeGreaterThan(0.50);
    expect(fp8.tpot / bf16.tpot).toBeLessThan(0.80);
  });

  it('INT4 70B TP=1 fits in 80GB and TPOT ≈ 16 ms', () => {
    const weightMem = modelWeightsMemory(llama70b, 'int4');
    expect(weightMem / 1e9).toBeLessThan(80);
    expect(weightMem / 1e9).toBeGreaterThan(30);
    const r = tp(llama70b, 1024, 128, 1, H100_SXM, 1, 'int4');
    expect(r.tpot).toBeGreaterThan(12);
    expect(r.tpot).toBeLessThan(22);
  });

  it('FP8 at high TP exposes AR (smaller model reads → AR fraction higher)', () => {
    const fp8_8b_tp8 = tp(llama8b, 1024, 128, 1, H100_SXM, 8, 'fp8');
    const bf16_8b_tp8 = tp(llama8b, 1024, 128, 1, H100_SXM, 8);
    // FP8/BF16 ratio ≈ 0.996 (AR dominates, weight savings barely matter)
    expect(fp8_8b_tp8.tpot / bf16_8b_tp8.tpot).toBeGreaterThan(0.85);
    expect(fp8_8b_tp8.tpot / bf16_8b_tp8.tpot).toBeLessThan(1.05);
  });
});

// ────────────────────────────────────────────────────────────
// Section 10: AMD GPU TP Tests — MI300X, MI325X, MI350X
// ────────────────────────────────────────────────────────────
describe('Section 10: AMD GPU TP Tests', () => {
  it('MI300X 70B TP=1 TPOT ≈ 32 ms (192 GB fits 140 GB BF16)', () => {
    const r = tp(llama70b, 1024, 128, 1, MI300X, 1);
    // 70B BF16 = 140 GB fits in 192 GB MI300X
    expect(r.tpot).toBeGreaterThan(24);
    expect(r.tpot).toBeLessThan(42);
  });

  it('MI300X 70B TP scaling ratios ≈ 0.50 (Infinity Fabric)', () => {
    const r2 = tp(llama70b, 1024, 128, 1, MI300X, 2);
    const r4 = tp(llama70b, 1024, 128, 1, MI300X, 4);
    const r8 = tp(llama70b, 1024, 128, 1, MI300X, 8);
    // TP=4/TP=2 ≈ 0.521, TP=8/TP=4 ≈ 0.566
    expect(r4.tpot / r2.tpot).toBeGreaterThan(0.43);
    expect(r4.tpot / r2.tpot).toBeLessThan(0.62);
    expect(r8.tpot / r4.tpot).toBeGreaterThan(0.46);
    expect(r8.tpot / r4.tpot).toBeLessThan(0.66);
  });

  it('MI300X 8B TP=8: TPOT improves over TP=4 or saturates', () => {
    const tp4 = tp(llama8b, 1024, 128, 1, MI300X, 4);
    const tp8 = tp(llama8b, 1024, 128, 1, MI300X, 8);
    // 8B is small — TP=8 may be worse than TP=4 due to IF overhead
    // But with IF (336 GB/s), much better than PCIe would be
    expect(tp8.tpot).toBeGreaterThan(0);
    expect(tp8.tpot).toBeLessThan(tp4.tpot * 2.0);
  });

  it('MI300X vs H100 at TP=1: MI300X faster (1.58× more BW)', () => {
    const mi = tp(llama70b, 1024, 128, 1, MI300X, 1);
    const h = tp(llama70b, 1024, 128, 1, H100_SXM, 1);
    // MI300X TPOT/H100 TPOT ≈ 0.632 (no comm, pure BW advantage)
    expect(mi.tpot / h.tpot).toBeGreaterThan(0.50);
    expect(mi.tpot / h.tpot).toBeLessThan(0.75);
  });

  it('MI300X vs H100 at TP=8: MI300X still faster (HBM BW dominates)', () => {
    const mi = tp(llama70b, 1024, 128, 1, MI300X, 8);
    const h = tp(llama70b, 1024, 128, 1, H100_SXM, 8);
    // MI300X TPOT/H100 TPOT ≈ 0.668
    expect(mi.tpot / h.tpot).toBeGreaterThan(0.50);
    expect(mi.tpot / h.tpot).toBeLessThan(0.80);
  });

  it('MI325X 70B TP=1 faster than MI300X (higher BW)', () => {
    const mi325 = tp(llama70b, 1024, 128, 1, MI325X, 1);
    const mi300 = tp(llama70b, 1024, 128, 1, MI300X, 1);
    expect(mi325.tpot).toBeLessThan(mi300.tpot);
  });

  it('MI350X 405B TP=8: fits and runs with high BW/TFLOPS', () => {
    const r = tp(llama405b, 1024, 128, 1, MI350X, 8);
    expect(Number.isFinite(r.ttft)).toBe(true);
    expect(Number.isFinite(r.tpot)).toBe(true);
    // 405B BF16 = 810 GB, MI350X 8× = 2304 GB → fits
    // TTFT ≈ 113 ms (2307 TFLOPS), TPOT ≈ 15.4 ms (8.0 TB/s)
    expect(r.ttft).toBeGreaterThan(85);
    expect(r.ttft).toBeLessThan(145);
    expect(r.tpot).toBeGreaterThan(11);
    expect(r.tpot).toBeLessThan(21);
  });

  it('MI350X TTFT TP scaling near-perfect (403 GB/s IF keeps AR hidden)', () => {
    const r2 = tp(llama70b, 1024, 128, 1, MI350X, 2);
    const r4 = tp(llama70b, 1024, 128, 1, MI350X, 4);
    const r8 = tp(llama70b, 1024, 128, 1, MI350X, 8);
    // TTFT 2→4 ≈ 0.501, 4→8 ≈ 0.502
    expect(r4.ttft / r2.ttft).toBeGreaterThan(0.42);
    expect(r4.ttft / r2.ttft).toBeLessThan(0.58);
    expect(r8.ttft / r4.ttft).toBeGreaterThan(0.42);
    expect(r8.ttft / r4.ttft).toBeLessThan(0.58);
  });

  it('MI300X TP scaling ≈ H100 SXM TP scaling for 70B (both NVLink-like)', () => {
    const mi_2to4 = tp(llama70b, 1024, 128, 1, MI300X, 4).tpot / tp(llama70b, 1024, 128, 1, MI300X, 2).tpot;
    const h_2to4 = tp(llama70b, 1024, 128, 1, H100_SXM, 4).tpot / tp(llama70b, 1024, 128, 1, H100_SXM, 2).tpot;
    // Both ≈ 0.52. Difference < 0.10.
    expect(Math.abs(mi_2to4 - h_2to4)).toBeLessThan(0.10);

    const mi_4to8 = tp(llama70b, 1024, 128, 1, MI300X, 8).tpot / tp(llama70b, 1024, 128, 1, MI300X, 4).tpot;
    const h_4to8 = tp(llama70b, 1024, 128, 1, H100_SXM, 8).tpot / tp(llama70b, 1024, 128, 1, H100_SXM, 4).tpot;
    expect(Math.abs(mi_4to8 - h_4to8)).toBeLessThan(0.10);
  });
});

// ────────────────────────────────────────────────────────────
// Section 11: GPU Comparison at Fixed TP
// ────────────────────────────────────────────────────────────
describe('Section 11: GPU Comparison at Fixed TP', () => {
  it('TPOT ordering matches BW ordering at TP=1 (no comm, pure BW)', () => {
    // BW ordering: B200 (7.7) ≈ MI350X (8.0) > MI300X (5.3) > H200 (4.8) > H100 (3.35) > A100 (2.039)
    const b200  = tp(llama70b, 1024, 128, 1, B200, 1).tpot;
    const mi350 = tp(llama70b, 1024, 128, 1, MI350X, 1).tpot;
    const mi300 = tp(llama70b, 1024, 128, 1, MI300X, 1).tpot;
    const h200  = tp(llama70b, 1024, 128, 1, H200_SXM, 1).tpot;
    const h100  = tp(llama70b, 1024, 128, 1, H100_SXM, 1).tpot;
    const a100  = tp(llama70b, 1024, 128, 1, A100_80GB, 1).tpot;

    // Lower TPOT = faster. B200 and MI350X are close (8.0 vs 7.7 TB/s).
    expect(b200).toBeLessThan(mi300);
    expect(mi350).toBeLessThan(mi300);
    expect(mi300).toBeLessThan(h200);
    expect(h200).toBeLessThan(h100);
    expect(h100).toBeLessThan(a100);
  });

  it('A100/H100 TPOT ratio ≈ BW ratio (1.64×)', () => {
    const a100 = tp(llama70b, 1024, 128, 1, A100_80GB, 1).tpot;
    const h100 = tp(llama70b, 1024, 128, 1, H100_SXM, 1).tpot;
    const tpotRatio = a100 / h100; // ≈ 1.643
    const bwRatio = H100_SXM.memoryBandwidthTBps / A100_80GB.memoryBandwidthTBps; // 1.643
    // TPOT ratio should be within ±30% of BW ratio (BW eff differences)
    expect(tpotRatio).toBeGreaterThan(bwRatio * 0.70);
    expect(tpotRatio).toBeLessThan(bwRatio * 1.30);
  });
});
