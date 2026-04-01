import { describe, it, expect } from 'vitest';
import { getModel } from '../../src/core/models/index.ts';
import {
  estimateActivationMemory,
  getEffectiveBackwardMultiplier,
  getSelectiveRecomputeFraction,
  solveMaxStoredLayers,
} from '../../src/core/strategies/base.ts';
import { getValidatedSimulationMetrics } from '../helpers/validated-metrics.ts';
import { benchmarkConfig } from '../helpers/benchmark-config.ts';

describe('Stored-layers selective AC', () => {
  // Use a well-known model for predictable test values
  const model = getModel('llama3-8b', 2048)!;

  describe('estimateActivationMemory', () => {
    it('storedLayers=N is identical to current selective (no totalLayers)', () => {
      const baseline = estimateActivationMemory(
        model, 2048, 1, 'bf16', true, true, 1, 'selective',
      );
      const withN = estimateActivationMemory(
        model, 2048, 1, 'bf16', true, true, 1, 'selective',
        model.numLayers, model.numLayers,
      );
      expect(withN).toBeCloseTo(baseline, 0);
    });

    it('storedLayers=0 produces less memory than storedLayers=N', () => {
      const allStored = estimateActivationMemory(
        model, 2048, 1, 'bf16', true, true, 1, 'selective',
        model.numLayers, model.numLayers,
      );
      const noneStored = estimateActivationMemory(
        model, 2048, 1, 'bf16', true, true, 1, 'selective',
        0, model.numLayers,
      );
      expect(noneStored).toBeLessThan(allStored);
    });

    it('monotonically decreasing as storedLayers decreases', () => {
      const N = model.numLayers;
      let prev = Infinity;
      for (let k = N; k >= 0; k -= Math.max(1, Math.floor(N / 8))) {
        const mem = estimateActivationMemory(
          model, 2048, 1, 'bf16', true, true, 1, 'selective', k, N,
        );
        expect(mem).toBeLessThanOrEqual(prev);
        prev = mem;
      }
    });
  });

  describe('getEffectiveBackwardMultiplier', () => {
    it('storedLayers=N returns selective multiplier (2.0 + f)', () => {
      const mult = getEffectiveBackwardMultiplier(
        model, true, 'selective', model.numLayers, model.numLayers,
      );
      // Should be > 2.0 (selective has some recompute)
      expect(mult).toBeGreaterThan(2.0);
      expect(mult).toBeLessThan(2.5);
    });

    it('storedLayers=0 returns full AC multiplier (2.85)', () => {
      const mult = getEffectiveBackwardMultiplier(
        model, true, 'selective', 0, model.numLayers,
      );
      expect(mult).toBeCloseTo(2.85, 1);
    });

    it('no AC returns 2.0', () => {
      const mult = getEffectiveBackwardMultiplier(model, false, undefined);
      expect(mult).toBe(2.0);
    });

    it('full AC returns 2.85', () => {
      const mult = getEffectiveBackwardMultiplier(model, true, 'full');
      expect(mult).toBe(2.85);
    });
  });

  describe('solveMaxStoredLayers', () => {
    it('returns N when budget is generous', () => {
      const k = solveMaxStoredLayers(64, 100, 200, 50, 1e12);
      expect(k).toBe(64);
    });

    it('returns 0 when budget is very tight', () => {
      const k = solveMaxStoredLayers(64, 100, 200, 50, 10);
      expect(k).toBe(0);
    });

    it('returns intermediate value for moderate budget', () => {
      // Budget that fits ~32 selective + sqrt(32) full
      // 32 * 100 + sqrt(32) * 200 + 50 = 3200 + 1131 + 50 = 4381
      const k = solveMaxStoredLayers(64, 100, 200, 50, 4400);
      expect(k).toBeGreaterThanOrEqual(30);
      expect(k).toBeLessThanOrEqual(35);
    });
  });

  describe('OLMo 3 integration', () => {
    it('OLMo 3 32B FSDP + selective AC + auto fits 80 GB H100', () => {
      const metrics = getValidatedSimulationMetrics(benchmarkConfig(
        'h100-sxm', 8, 128, 'olmo3-32b', 'fsdp', 1024, 1, 8192,
        undefined,
        { activationCheckpointing: true, checkpointingGranularity: 'selective' },
      ));
      // Fits in 80 GB (stored-layers auto-resolves to reduce activation memory)
      expect(metrics.memoryUtilization).toBeLessThanOrEqual(1.0);
      // MFU in sane range (published ~41%, sim ~43.4%)
      expect(metrics.mfu).toBeGreaterThan(0.38);
      expect(metrics.mfu).toBeLessThan(0.50);
      // Stored layers resolved < total layers
      expect(metrics.resolvedStoredLayers).toBeDefined();
      expect(metrics.resolvedStoredLayers).toBeLessThan(64);
      expect(metrics.resolvedStoredLayers).toBeGreaterThan(0);
    });

    it('OLMo 2 32B FSDP + selective AC: all layers fit (no budget pressure)', () => {
      const metrics = getValidatedSimulationMetrics(benchmarkConfig(
        'h100-sxm', 8, 160, 'olmo2-32b', 'fsdp', 1024, 1, 4096,
        undefined,
        { activationCheckpointing: true, checkpointingGranularity: 'selective' },
      ));
      // All 64 layers should fit (no budget pressure)
      expect(metrics.resolvedStoredLayers).toBe(64);
    });
  });

  describe('MLA models', () => {
    const v3 = getModel('deepseek-v3', 2048)!;
    const glm5 = getModel('glm5', 2048)!;
    const glm47f = getModel('glm4.7-flash', 2048)!;

    it('DeepSeek V3 selective recompute fraction ~32%', () => {
      const f = getSelectiveRecomputeFraction(v3);
      expect(f).toBeGreaterThan(0.28);
      expect(f).toBeLessThan(0.36);
    });

    it('GLM-5 selective recompute fraction ~33%', () => {
      const f = getSelectiveRecomputeFraction(glm5);
      expect(f).toBeGreaterThan(0.28);
      expect(f).toBeLessThan(0.38);
    });

    it('GLM-4.7-Flash selective recompute fraction ~32%', () => {
      const f = getSelectiveRecomputeFraction(glm47f);
      expect(f).toBeGreaterThan(0.25);
      expect(f).toBeLessThan(0.40);
    });

    it('V3 backward multiplier with selective AC ~2.32', () => {
      const mult = getEffectiveBackwardMultiplier(v3, true, 'selective');
      expect(mult).toBeGreaterThan(2.28);
      expect(mult).toBeLessThan(2.36);
    });

    it('GLM-5 activation memory uses MLA dimensions', () => {
      const mlaMem = estimateActivationMemory(glm5, 2048, 1, 'bf16', true, true, 1, 'full');
      // MLA decompressed dims produce ~5.1 GB; GQA fallback with headDim=64 would give ~3.5 GB.
      // Bound set between the two to catch regressions where MLA branch is not taken.
      expect(mlaMem).toBeGreaterThan(4.0e9);
    });

    it('wider-attention models use nH*headDim for attnOutDim', () => {
      // GLM-4.5-Air: nH=96, headDim=128, h=4096. nH*headDim=12288 (3x hiddenSize).
      // With nH*headDim: ~2.25 GB. With old hiddenSize formula: ~2.03 GB.
      const air = getModel('glm4.5-air', 2048)!;
      const airMem = estimateActivationMemory(air, 2048, 1, 'bf16', true, true, 1, 'full');
      expect(airMem).toBeGreaterThan(2.1e9);
    });
  });
});
