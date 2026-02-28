/**
 * Full inference sensitivity regression test.
 * All 11 params × 5 benchmarks. Gated by RUN_SLOW_TESTS=1.
 */
import { describe, it, expect, afterEach } from 'vitest';
import { runInferenceSensitivityAnalysis } from '../../src/core/analysis/inference-sensitivity.ts';
import {
  getAllPerturbableInferenceParams,
  resetAllPerturbableInferenceParams,
} from '../../src/core/analysis/inference-perturbable-params.ts';
import type { InferenceSimulationConfig } from '../../src/core/inference/simulation.ts';

const SLOW = process.env.RUN_SLOW_TESTS === '1';

afterEach(() => {
  resetAllPerturbableInferenceParams();
});

function makeFullBenchmarks(): { name: string; config: InferenceSimulationConfig }[] {
  return [
    {
      name: 'LLaMA 8B 1×H100 BF16 B=32',
      config: {
        modelId: 'llama3.1-8b',
        gpuId: 'h100-sxm',
        numGPUs: 1,
        batchSize: 32,
        inputSeqLen: 512,
        outputSeqLen: 128,
        weightPrecision: 'bf16',
      },
    },
    {
      name: 'LLaMA 70B 4×H100 TP=4 BF16 B=16',
      config: {
        modelId: 'llama3.3-70b',
        gpuId: 'h100-sxm',
        numGPUs: 4,
        tensorParallel: 4,
        batchSize: 16,
        inputSeqLen: 512,
        outputSeqLen: 128,
        weightPrecision: 'bf16',
      },
    },
    {
      name: 'DeepSeek V3 8×H200 TP=4 EP=2 FP8 B=32',
      config: {
        modelId: 'deepseek-v3',
        gpuId: 'h200-sxm',
        numGPUs: 8,
        tensorParallel: 4,
        expertParallel: 2,
        batchSize: 32,
        inputSeqLen: 512,
        outputSeqLen: 128,
        weightPrecision: 'fp8',
      },
    },
    {
      name: 'LLaMA 70B 1×H200 INT4 B=8',
      config: {
        modelId: 'llama3.3-70b',
        gpuId: 'h200-sxm',
        numGPUs: 1,
        batchSize: 8,
        inputSeqLen: 512,
        outputSeqLen: 128,
        weightPrecision: 'int4',
      },
    },
    {
      name: 'LLaMA 8B 1×H100 CB B=64',
      config: {
        modelId: 'llama3.1-8b',
        gpuId: 'h100-sxm',
        numGPUs: 1,
        batchSize: 64,
        inputSeqLen: 512,
        outputSeqLen: 128,
        weightPrecision: 'bf16',
        continuousBatching: true,
      },
    },
  ];
}

describe.skipIf(!SLOW)('Full Inference Sensitivity Analysis', () => {
  it('all 11 params × 5 benchmarks: every non-MEMORY_OVERHEAD param has non-zero sensitivity on ≥1 benchmark', () => {
    const params = getAllPerturbableInferenceParams();
    const benchmarks = makeFullBenchmarks();

    const report = runInferenceSensitivityAnalysis(params, benchmarks, 0.05);

    expect(report.results).toHaveLength(11);

    // Every param except structural zeros should have non-zero sensitivity somewhere.
    // MEMORY_OVERHEAD_FACTOR: memory-only, no latency effect.
    // PCIE_PER_ROUND_MS: all benchmarks use NVLink GPUs (H100/H200), PCIe alpha never fires.
    const STRUCTURAL_ZEROS = new Set(['MEMORY_OVERHEAD_FACTOR', 'PCIE_PER_ROUND_MS']);
    for (const result of report.results) {
      if (STRUCTURAL_ZEROS.has(result.paramName)) continue;
      const hasNonZero = result.deltas.some(d =>
        d.throughputMinus !== d.throughputPlus ||
        d.tpotMinus !== d.tpotPlus ||
        d.ttftMinus !== d.ttftPlus
      );
      expect(hasNonZero).toBe(true);
    }
  });

  it('PREFILL_RESIDUAL is classified as high-sensitivity', () => {
    const params = getAllPerturbableInferenceParams();
    const benchmarks = makeFullBenchmarks();

    const report = runInferenceSensitivityAnalysis(params, benchmarks, 0.05);

    expect(report.highSensitivityParams).toContain('PREFILL_RESIDUAL');
  });

  it('MEMORY_OVERHEAD_FACTOR is dead-weight for latency metrics', () => {
    const params = getAllPerturbableInferenceParams().filter(
      p => p.name === 'MEMORY_OVERHEAD_FACTOR'
    );
    const benchmarks = makeFullBenchmarks();

    const report = runInferenceSensitivityAnalysis(params, benchmarks, 0.05);

    // MEMORY_OVERHEAD_FACTOR only affects memory sizing, not throughput/TPOT/TTFT
    for (const delta of report.results[0].deltas) {
      expect(delta.throughputMinus).toBe(delta.throughputPlus);
      expect(delta.tpotMinus).toBe(delta.tpotPlus);
      expect(delta.ttftMinus).toBe(delta.ttftPlus);
    }
  });

  it('zero-sensitivity patterns: TP params zero on TP=1, CB params zero on non-CB', () => {
    const allParams = getAllPerturbableInferenceParams();
    const benchmarks = makeFullBenchmarks();

    // TP params
    const tpParams = allParams.filter(p =>
      ['NVLINK_PER_ROUND_MS', 'PCIE_PER_ROUND_MS', 'TP_COMM_EFFICIENCY'].includes(p.name)
    );
    const tpReport = runInferenceSensitivityAnalysis(tpParams, benchmarks, 0.05);

    // TP params should have zero sensitivity on TP=1 benchmarks
    const tp1Benchmarks = ['LLaMA 8B 1×H100 BF16 B=32', 'LLaMA 70B 1×H200 INT4 B=8', 'LLaMA 8B 1×H100 CB B=64'];
    for (const result of tpReport.results) {
      for (const delta of result.deltas) {
        if (tp1Benchmarks.includes(delta.benchmarkName)) {
          expect(delta.throughputMinus).toBe(delta.throughputPlus);
        }
      }
    }

    // CB params should have zero sensitivity on non-CB benchmarks
    const cbParams = allParams.filter(p =>
      ['CB_SCHEDULING_BASE', 'CB_PREFILL_INTERFERENCE_MAX'].includes(p.name)
    );
    const cbReport = runInferenceSensitivityAnalysis(cbParams, benchmarks, 0.05);

    const nonCbBenchmarks = benchmarks.filter(b => !b.config.continuousBatching).map(b => b.name);
    for (const result of cbReport.results) {
      for (const delta of result.deltas) {
        if (nonCbBenchmarks.includes(delta.benchmarkName)) {
          expect(delta.throughputMinus).toBe(delta.throughputPlus);
        }
      }
    }
  });

  it('≥7 of 11 params have non-zero sensitivity on the full benchmark set', () => {
    const params = getAllPerturbableInferenceParams();
    const benchmarks = makeFullBenchmarks();

    const report = runInferenceSensitivityAnalysis(params, benchmarks, 0.05);

    const activeParams = report.results.filter(r =>
      r.deltas.some(d =>
        d.throughputMinus !== d.throughputPlus ||
        d.tpotMinus !== d.tpotPlus ||
        d.ttftMinus !== d.ttftPlus
      )
    );

    expect(activeParams.length).toBeGreaterThanOrEqual(7);
  });
});
