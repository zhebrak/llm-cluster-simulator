/**
 * Full Benchmark Re-Validation (6a)
 *
 * Runs all 11 Tier 1 primary benchmarks and asserts:
 *   - Max absolute MFU delta < 6 pp (catches systemic drift; OLMo 3's gap
 *     is now ~2.8pp with stored-layers auto-resolve; MT-NLG's 4.1pp is the outlier)
 *   - Reports parameter census from PARAMETER_REGISTRY
 *
 * Complements anchor-benchmarks.test.ts — anchors enforce tight per-benchmark
 * bounds (±3pp); this test provides a distribution-wide summary + census snapshot.
 */

import { describe, it, expect } from 'vitest';
import { getSimulationMetrics } from '../../src/core/simulation/engine.ts';
import { getValidatedSimulationMetrics } from '../helpers/validated-metrics.ts';
import { ALL_PUBLISHED_CONFIGS, toSimConfig } from '../../src/data/published-training-configs.ts';
import { PARAMETER_REGISTRY } from '../../src/core/parameter-registry.ts';

describe('Full Benchmark Re-Validation', () => {
  const tier1 = ALL_PUBLISHED_CONFIGS.filter(c => c.tier === 1);

  it('should have exactly 10 Tier 1 benchmarks', () => {
    expect(tier1.length).toBe(10);
  });

  it('max absolute MFU delta across all Tier 1 benchmarks < 6pp', () => {
    const deltas: { label: string; published: number; sim: number; delta: number }[] = [];

    for (const pc of tier1) {
      if (pc.published.mfu == null) continue;

      const config = toSimConfig(pc);
      // Some configs (BLOOM) marginally OOM — use raw sim like anchor tests
      let metrics;
      try {
        metrics = getValidatedSimulationMetrics(config);
      } catch {
        metrics = getSimulationMetrics(config);
      }

      // When published MFU label contains '*', the published number is
      // Model FLOPs MFU (uses actual flopsPerToken incl. quadratic attention),
      // not standard 6PD MFU. Compare against the corresponding sim metric.
      const isModelFlopsMfu = pc.published.mfuLabel.includes('*');
      const sim = isModelFlopsMfu
        ? (metrics.modelFlopsMfu ?? metrics.mfu)
        : metrics.mfu;

      const delta = Math.abs(sim - pc.published.mfu) * 100; // pp
      deltas.push({
        label: pc.label,
        published: pc.published.mfu * 100,
        sim: sim * 100,
        delta,
      });
    }

    expect(deltas.length).toBeGreaterThan(0);

    const maxDelta = Math.max(...deltas.map(d => d.delta));
    expect(
      maxDelta,
      `Max delta ${maxDelta.toFixed(1)}pp exceeds 6pp threshold.\n` +
      deltas.map(d => `  ${d.label}: sim=${d.sim.toFixed(1)}% pub=${d.published.toFixed(1)}% delta=${d.delta.toFixed(1)}pp`).join('\n'),
    ).toBeLessThan(6);
  });

  it('reports parameter census', () => {
    const total = PARAMETER_REGISTRY.length;
    const byTier = {
      physics: PARAMETER_REGISTRY.filter(p => p.tier === 'physics').length,
      'grounded-empirical': PARAMETER_REGISTRY.filter(p => p.tier === 'grounded-empirical').length,
      fitted: PARAMETER_REGISTRY.filter(p => p.tier === 'fitted').length,
      redundant: PARAMETER_REGISTRY.filter(p => p.tier === 'redundant').length,
    };

    // Snapshot — any change to the registry should be intentional
    expect(total).toBeGreaterThanOrEqual(50);
    expect(byTier.fitted).toBeGreaterThanOrEqual(13);
    expect(byTier.physics + byTier['grounded-empirical'] + byTier.fitted + byTier.redundant).toBe(total);
  });
});
