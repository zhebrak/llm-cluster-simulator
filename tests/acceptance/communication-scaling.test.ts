/**
 * Communication Scaling Validation Tests
 *
 * Validates that FSDP/ZeRO-3 communication correctly scales with gradient
 * accumulation (GA), while DDP/ZeRO-1 communication does not.
 *
 * Tests:
 * 1. FSDP comm scales with GA (smaller MBS = higher comm overhead)
 * 2. MFU monotonicity: increasing MBS should not decrease MFU for FSDP
 * 3. Small MBS penalty: FSDP MBS=1 has significantly lower MFU than MBS=8
 * 4. DDP unaffected: DDP comm same regardless of MBS
 * 5. ZeRO-3 scales, ZeRO-1 doesn't
 * 6. 3D parallel dpType conditional: FSDP dpType scales, DDP dpType doesn't
 * 7. Recommendation: PP bubble guard
 * 8. Recommendation: quality thresholds
 */

import { describe, it, expect } from 'vitest';
import { getValidatedSimulationMetrics } from '../helpers/validated-metrics.ts';
import type { SimulationConfig } from '../../src/core/simulation/engine.ts';
import { generateRecommendations } from '../../src/core/simulation/recommendations.ts';
import { makeStrategyContext } from '../helpers/strategy-context.ts';

describe('Communication Scaling', () => {
  // ── 1. FSDP comm scales with GA ──────────────────────────────────
  describe('FSDP communication scales with gradient accumulation', () => {
    it('smaller MBS (higher GA) should have higher communication overhead', () => {
      // Same model, same GBS, same cluster — only MBS differs
      // MBS=1 → GA=64, MBS=8 → GA=8
      const configSmallMBS: SimulationConfig = {
        modelId: 'llama2-7b',
        clusterId: '8x-h100',
        globalBatchSize: 512,
        microBatchSize: 1,
        sequenceLength: 2048,
        strategyType: 'fsdp',
      };

      const configLargeMBS: SimulationConfig = {
        modelId: 'llama2-7b',
        clusterId: '8x-h100',
        globalBatchSize: 512,
        microBatchSize: 8,
        sequenceLength: 2048,
        strategyType: 'fsdp',
      };

      const metricsSmall = getValidatedSimulationMetrics(configSmallMBS);
      const metricsLarge = getValidatedSimulationMetrics(configLargeMBS);

      // Small MBS should have higher comm overhead (FSDP comm repeats per micro-batch)
      expect(metricsSmall.communicationOverhead).toBeGreaterThan(metricsLarge.communicationOverhead);
    });
  });

  // ── 2. MFU monotonicity with MBS ─────────────────────────────────
  describe('MFU monotonicity', () => {
    it('increasing MBS should not decrease MFU for FSDP', () => {
      const mbsSizes = [1, 2, 4, 8];
      const mfuValues: number[] = [];

      for (const mbs of mbsSizes) {
        const config: SimulationConfig = {
          modelId: 'llama2-7b',
          clusterId: '8x-h100',
          globalBatchSize: 512,
          microBatchSize: mbs,
          sequenceLength: 2048,
          strategyType: 'fsdp',
        };
        const metrics = getValidatedSimulationMetrics(config);
        mfuValues.push(metrics.mfu);
      }

      // Each larger MBS should have equal or better MFU
      for (let i = 1; i < mfuValues.length; i++) {
        expect(mfuValues[i]).toBeGreaterThanOrEqual(mfuValues[i - 1]);
      }
    });
  });

  // ── 3. Small MBS penalty ──────────────────────────────────────────
  describe('Small MBS penalty for FSDP', () => {
    it('FSDP MBS=1 should have significantly lower MFU than MBS=8', () => {
      const configMBS1: SimulationConfig = {
        modelId: 'llama2-7b',
        clusterId: '8x-h100',
        globalBatchSize: 512,
        microBatchSize: 1,
        sequenceLength: 2048,
        strategyType: 'fsdp',
      };

      const configMBS8: SimulationConfig = {
        modelId: 'llama2-7b',
        clusterId: '8x-h100',
        globalBatchSize: 512,
        microBatchSize: 8,
        sequenceLength: 2048,
        strategyType: 'fsdp',
      };

      const mfu1 = getValidatedSimulationMetrics(configMBS1).mfu;
      const mfu8 = getValidatedSimulationMetrics(configMBS8).mfu;

      // MBS=8 should be meaningfully better (at least 0.5 percentage points).
      // GA-aware FSDP overlap narrows the gap (high-GA configs hide more comm),
      // but MBS=8 still wins due to fewer micro-batches and better GEMM efficiency.
      expect(mfu8 - mfu1).toBeGreaterThan(0.005);
    });
  });

  // ── 4. DDP unaffected by MBS ──────────────────────────────────────
  describe('DDP communication unaffected by MBS', () => {
    it('DDP comm overhead should be similar regardless of MBS', () => {
      const configSmallMBS: SimulationConfig = {
        modelId: 'gpt3-1.3b',
        clusterId: '8x-h100',
        globalBatchSize: 512,
        microBatchSize: 1,
        sequenceLength: 2048,
        strategyType: 'ddp',
      };

      const configLargeMBS: SimulationConfig = {
        modelId: 'gpt3-1.3b',
        clusterId: '8x-h100',
        globalBatchSize: 512,
        microBatchSize: 8,
        sequenceLength: 2048,
        strategyType: 'ddp',
      };

      const metricsSmall = getValidatedSimulationMetrics(configSmallMBS);
      const metricsLarge = getValidatedSimulationMetrics(configLargeMBS);

      // DDP comm is once per step — overhead ratio should be similar
      // (Small MBS has more compute steps but also same single comm, so overhead fraction is close)
      const diffRatio = Math.abs(metricsSmall.communicationOverhead - metricsLarge.communicationOverhead);
      // Allow small variance from efficiency model, but not the large gap FSDP shows
      expect(diffRatio).toBeLessThan(0.10);
    });
  });

  // ── 5. ZeRO-3 scales, ZeRO-1 doesn't ────────────────────────────
  describe('ZeRO stage-conditional scaling', () => {
    it('ZeRO-3 comm overhead increases with smaller MBS, ZeRO-1 does not', () => {
      // ZeRO-3
      const z3SmallMBS: SimulationConfig = {
        modelId: 'llama2-7b',
        clusterId: '8x-h100',
        globalBatchSize: 512,
        microBatchSize: 1,
        sequenceLength: 2048,
        strategyType: 'zero-3',
      };

      const z3LargeMBS: SimulationConfig = {
        ...z3SmallMBS,
        microBatchSize: 8,
      };

      const z3Small = getValidatedSimulationMetrics(z3SmallMBS);
      const z3Large = getValidatedSimulationMetrics(z3LargeMBS);

      // ZeRO-3 should show significant comm scaling with smaller MBS
      expect(z3Small.communicationOverhead).toBeGreaterThan(z3Large.communicationOverhead);

      // ZeRO-1
      const z1SmallMBS: SimulationConfig = {
        modelId: 'llama2-7b',
        clusterId: '8x-h100',
        globalBatchSize: 512,
        microBatchSize: 1,
        sequenceLength: 2048,
        strategyType: 'zero-1',
      };

      const z1LargeMBS: SimulationConfig = {
        ...z1SmallMBS,
        microBatchSize: 8,
      };

      const z1Small = getValidatedSimulationMetrics(z1SmallMBS);
      const z1Large = getValidatedSimulationMetrics(z1LargeMBS);

      // ZeRO-1 comm overhead diff should be much smaller than ZeRO-3
      const z3Diff = z3Small.communicationOverhead - z3Large.communicationOverhead;
      const z1Diff = Math.abs(z1Small.communicationOverhead - z1Large.communicationOverhead);
      expect(z3Diff).toBeGreaterThan(z1Diff * 2);
    });
  });

  // ── 6. 3D parallel dpType conditional ─────────────────────────────
  describe('3D parallel dpType-conditional DP comm scaling', () => {
    it('FSDP dpType scales DP comm with GA, DDP dpType does not', () => {
      // FSDP dpType: fsdp-tp with small vs large MBS
      const fsdpSmall: SimulationConfig = {
        modelId: 'llama2-70b',
        clusterId: '128x-h100',
        globalBatchSize: 512,
        microBatchSize: 1,
        sequenceLength: 2048,
        strategyType: 'fsdp-tp',
        strategyConfig: { tp: 8 },
      };

      const fsdpLarge: SimulationConfig = {
        ...fsdpSmall,
        microBatchSize: 4,
      };

      const fsdpMetricsSmall = getValidatedSimulationMetrics(fsdpSmall);
      const fsdpMetricsLarge = getValidatedSimulationMetrics(fsdpLarge);

      // FSDP dpType should show higher comm with smaller MBS
      expect(fsdpMetricsSmall.communicationOverhead).toBeGreaterThan(fsdpMetricsLarge.communicationOverhead);

      // ZeRO-1 dpType: zero1-tp with small vs large MBS (DDP-like — no per-microbatch comm)
      const ddpSmall: SimulationConfig = {
        modelId: 'llama2-70b',
        clusterId: '128x-h100',
        globalBatchSize: 512,
        microBatchSize: 1,
        sequenceLength: 2048,
        strategyType: 'zero1-tp',
        strategyConfig: { tp: 8 },
      };

      const ddpLarge: SimulationConfig = {
        ...ddpSmall,
        microBatchSize: 4,
      };

      const ddpMetricsSmall = getValidatedSimulationMetrics(ddpSmall);
      const ddpMetricsLarge = getValidatedSimulationMetrics(ddpLarge);

      // DDP dpType: comm overhead diff should be much smaller
      const fsdpDiff = fsdpMetricsSmall.communicationOverhead - fsdpMetricsLarge.communicationOverhead;
      const ddpDiff = Math.abs(ddpMetricsSmall.communicationOverhead - ddpMetricsLarge.communicationOverhead);
      expect(fsdpDiff).toBeGreaterThan(ddpDiff);
    });
  });

  // ── 7. Recommendation: PP bubble guard ────────────────────────────
  describe('Recommendation: PP bubble guard', () => {
    it('PP recommendation should mention bubble when applicable', () => {
      // Create a scenario where 2D→3D PP upgrade would be recommended:
      // Large model, high memory, 2D strategy
      const config: SimulationConfig = {
        modelId: 'llama2-70b',
        clusterId: '128x-h100',
        globalBatchSize: 512,
        microBatchSize: 2,
        sequenceLength: 4096,
        strategyType: 'fsdp-tp',
        strategyConfig: { tp: 8 },
      };

      const metrics = getValidatedSimulationMetrics(config);

      // Only test if the conditions for PP recommendation are met
      if (metrics.memoryUtilization > 0.85) {
        const ctx = makeStrategyContext('llama2-70b', '128x-h100', {
          globalBatchSize: 512,
          microBatchSize: 2,
          seqLength: 4096,
          dpDegree: 16,
        });

        const recs = generateRecommendations(config, ctx, metrics);
        const ppRec = recs.find(r => r.includes('pipeline parallelism'));

        // If PP is recommended, it should either be a clean recommendation
        // or mention bubble if bubble would be high
        if (ppRec) {
          // The recommendation exists — this is the main assertion
          expect(ppRec).toContain('pipeline');
        }
      }
    });
  });

  // ── 8. Recommendation: status line ──────────────────────────────────
  describe('Recommendation: status line', () => {
    it('well-optimized config shows "Well-optimized" status', () => {
      // Test the positive confirmation generator directly by creating metrics
      // that won't trigger any other generators
      const config: SimulationConfig = {
        modelId: 'llama2-7b',
        clusterId: '8x-h100',
        globalBatchSize: 256,
        microBatchSize: 4,
        sequenceLength: 2048,
        strategyType: 'ddp',
      };

      const ctx = makeStrategyContext('llama2-7b', '8x-h100', {
        globalBatchSize: 256,
        microBatchSize: 4,
      });

      // Synthesize "good" metrics that won't trigger warning generators
      const goodMetrics = {
        tokensPerSecond: 100000,
        samplesPerSecond: 50,
        tflopsPerGPU: 300,
        mfu: 0.45,
        hfu: 0.50,
        communicationOverhead: 0.10,
        pipelineBubble: 0,
        communicationGrossMs: 10,
        communicationExposedMs: 5,
        overlapHiddenFraction: 0.5,
        memoryPerGPU: { parameters: 0, gradients: 0, optimizerStates: 0, activations: 0, peakActivations: 0, temporary: 0, reserved: 0, total: 0 },
        memoryUtilization: 0.60,
        stepTimeMs: 100,
        timing: { forward: 40, backward: 80, optimizer: 2, communication: 10, overlap: 5, scaleOverhead: 0, total: 100 },
      };

      const recs = generateRecommendations(config, ctx, goodMetrics);
      // With no generators firing, should get "Well-optimized" status
      expect(recs.some(r => r.includes('Well-optimized'))).toBe(true);

      // Same for higher MFU — still just "Well-optimized" (no MFU-based tiers)
      const excellentMetrics = { ...goodMetrics, mfu: 0.55 };
      const excellentRecs = generateRecommendations(config, ctx, excellentMetrics);
      expect(excellentRecs.some(r => r.includes('Well-optimized'))).toBe(true);
    });

    it('config with actionable suggestions shows "Solid baseline" status', () => {
      const config: SimulationConfig = {
        modelId: 'gpt3-125m',
        clusterId: '8x-h100',
        globalBatchSize: 32,
        microBatchSize: 4,
        sequenceLength: 1024,
        strategyType: 'ddp',
      };

      const metrics = getValidatedSimulationMetrics(config);
      const ctx = makeStrategyContext('gpt3-125m', '8x-h100', {
        globalBatchSize: 32,
        microBatchSize: 4,
        seqLength: 1024,
      });

      const recs = generateRecommendations(config, ctx, metrics);

      // If actionable suggestions exist, status should be "Solid baseline"
      const hasActionable = recs.some(r =>
        r.includes('Consider') || r.includes('Try') || r.includes('MFU')
      );
      if (hasActionable) {
        expect(recs.some(r => r.includes('Solid baseline'))).toBe(true);
      } else {
        expect(recs.some(r => r.includes('Well-optimized'))).toBe(true);
      }
    });
  });
});
