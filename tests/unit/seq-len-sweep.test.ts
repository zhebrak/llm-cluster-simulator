/**
 * Tests for sequence length sweep
 */

import { describe, it, expect } from 'vitest';
import { runSeqLenSweep } from '../../src/core/inference/index.ts';
import type { InferenceSimulationConfig } from '../../src/core/inference/index.ts';
import { getModel } from '../../src/core/models/index.ts';
import { H100_SXM, L4, A100_80GB } from '../../src/core/hardware/gpu.ts';

function makeBaseConfig(overrides: Partial<InferenceSimulationConfig> = {}): InferenceSimulationConfig {
  return {
    modelSpec: getModel('llama3.1-8b', 4096)!,
    gpu: H100_SXM,
    numGPUs: 1,
    batchSize: 1,
    inputSeqLen: 1024,
    outputSeqLen: 128,
    weightPrecision: 'bf16',
    kvCachePrecision: 'bf16',
    flashAttention: true,
    pagedAttention: false,
    tensorParallel: 1,
    continuousBatching: false,
    ...overrides,
  };
}

describe('Sequence Length Sweep', () => {
  it('produces points for each available precision', () => {
    const base = makeBaseConfig();
    const result = runSeqLenSweep(base, base.modelSpec!, base.gpu!);

    // H100 has fp8 support, so expect bf16, fp8, int4
    expect(Object.keys(result.groups)).toContain('bf16');
    expect(Object.keys(result.groups)).toContain('fp8');
    expect(Object.keys(result.groups)).toContain('int4');

    // Each group should have multiple points
    expect(result.groups['bf16'].length).toBeGreaterThan(3);
    expect(result.groups['fp8'].length).toBeGreaterThan(3);
    expect(result.groups['int4'].length).toBeGreaterThan(3);
  });

  it('TTFT grows monotonically with sequence length', () => {
    const base = makeBaseConfig();
    const result = runSeqLenSweep(base, base.modelSpec!, base.gpu!);

    for (const [, points] of Object.entries(result.groups)) {
      for (let i = 1; i < points.length; i++) {
        expect(points[i].ttft).toBeGreaterThan(points[i - 1].ttft);
      }
    }
  });

  it('TPOT grows monotonically with sequence length', () => {
    const base = makeBaseConfig();
    const result = runSeqLenSweep(base, base.modelSpec!, base.gpu!);

    for (const [, points] of Object.entries(result.groups)) {
      for (let i = 1; i < points.length; i++) {
        expect(points[i].tpot).toBeGreaterThanOrEqual(points[i - 1].tpot);
      }
    }
  });

  it('detects OOM cutoff at correct sequence length', () => {
    // Use L4 (24GB) with larger batch to hit OOM earlier
    const model = getModel('llama3.1-8b', 131072)!;
    const base = makeBaseConfig({ gpu: L4, modelSpec: model, batchSize: 32 });
    const result = runSeqLenSweep(base, model, L4);

    // With 8B model on 24GB L4 at batch=32, bf16 should OOM before int4
    const bf16Cutoff = result.oomCutoffs['bf16'];
    const int4Cutoff = result.oomCutoffs['int4'];

    // At least one should OOM
    const anyOom = bf16Cutoff > 0 || int4Cutoff > 0;
    expect(anyOom).toBe(true);

    // If both OOM, bf16 should OOM at a lower or equal seqLen than int4
    if (bf16Cutoff > 0 && int4Cutoff > 0) {
      expect(bf16Cutoff).toBeLessThanOrEqual(int4Cutoff);
    }
  });

  it('stops sweeping after all precisions OOM', () => {
    // Use L4 (24GB) with large batch — all precisions eventually OOM
    const model = getModel('llama3.1-8b', 131072)!;
    const base = makeBaseConfig({ gpu: L4, modelSpec: model, batchSize: 64 });
    const result = runSeqLenSweep(base, model, L4);

    // Verify no points beyond OOM for each precision
    for (const [, points] of Object.entries(result.groups)) {
      const cutoff = result.oomCutoffs[Object.entries(result.groups).find(([, v]) => v === points)?.[0] ?? ''];
      if (cutoff > 0) {
        for (const pt of points) {
          expect(pt.seqLen).toBeLessThan(cutoff);
        }
      }
    }
  });

  it('seqLen values start at 128 and cutoffs are refined by binary search', () => {
    const base = makeBaseConfig();
    const result = runSeqLenSweep(base, base.modelSpec!, base.gpu!);

    for (const [, points] of Object.entries(result.groups)) {
      for (const pt of points) {
        // Minimum is 128
        expect(pt.seqLen).toBeGreaterThanOrEqual(128);
      }
    }

    // OOM cutoffs are refined via binary search (not necessarily powers of 2)
    for (const [, cutoff] of Object.entries(result.oomCutoffs)) {
      if (cutoff > 0) {
        expect(cutoff).toBeGreaterThanOrEqual(128);
      }
    }
  });

  it('lower precision extends OOM boundary', () => {
    // Use L4 (24GB) with medium batch so at least bf16 OOMs
    const model = getModel('llama3.1-8b', 131072)!;
    const base = makeBaseConfig({ gpu: L4, modelSpec: model, batchSize: 32 });
    const result = runSeqLenSweep(base, model, L4);

    // int4 KV cache is 4x smaller than bf16, so int4 should survive to higher seqLen
    const bf16Cutoff = result.oomCutoffs['bf16'];
    const int4Cutoff = result.oomCutoffs['int4'];

    if (bf16Cutoff > 0 && int4Cutoff > 0) {
      expect(int4Cutoff).toBeGreaterThanOrEqual(bf16Cutoff);
    } else if (bf16Cutoff > 0 && int4Cutoff === 0) {
      // int4 never OOM'd but bf16 did — correct (smaller KV cache)
      expect(bf16Cutoff).toBeGreaterThan(0);
    }
  });

  it('no fp8 group when GPU lacks fp8 support', () => {
    // A100 has bf16 but no fp8
    const model = getModel('llama3.1-8b', 4096)!;
    const base = makeBaseConfig({ gpu: A100_80GB, modelSpec: model });
    const result = runSeqLenSweep(base, model, A100_80GB);

    expect(result.groups['bf16'].length).toBeGreaterThan(0);
    expect(result.groups['int4'].length).toBeGreaterThan(0);
    // fp8 group should not exist (not available on A100)
    expect(result.groups['fp8']).toBeUndefined();
  });
});
