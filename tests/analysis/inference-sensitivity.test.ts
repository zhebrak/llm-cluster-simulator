/**
 * Thin CI test for inference sensitivity analysis engine.
 *
 * Runs 3 params × 2 benchmarks to verify the inference sensitivity engine
 * works and produces expected directional results.
 */
import { describe, it, expect, afterEach } from 'vitest';
import { runInferenceSensitivityAnalysis } from '../../src/core/analysis/inference-sensitivity.ts';
import {
  getAllPerturbableInferenceParams,
  resetAllPerturbableInferenceParams,
} from '../../src/core/analysis/inference-perturbable-params.ts';
import type { InferenceSimulationConfig } from '../../src/core/inference/simulation.ts';
import type { PerturbableParam } from '../../src/core/analysis/sensitivity.ts';

afterEach(() => {
  resetAllPerturbableInferenceParams();
});

function makeBenchmarks(): { name: string; config: InferenceSimulationConfig }[] {
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
  ];
}

describe('Inference Sensitivity Analysis Engine', () => {
  it('should run and produce results for PREFILL_RESIDUAL', () => {
    const allParams = getAllPerturbableInferenceParams();
    const prefillParam = allParams.filter(p => p.name === 'PREFILL_RESIDUAL');
    const benchmarks = makeBenchmarks();

    const report = runInferenceSensitivityAnalysis(prefillParam, benchmarks, 0.10);

    expect(report.results).toHaveLength(1);
    expect(report.results[0].paramName).toBe('PREFILL_RESIDUAL');
    expect(report.results[0].deltas).toHaveLength(2);

    // Each benchmark delta should have valid metrics
    for (const delta of report.results[0].deltas) {
      expect(delta.throughputOriginal).toBeGreaterThan(0);
      expect(delta.tpotOriginal).toBeGreaterThan(0);
      expect(delta.ttftOriginal).toBeGreaterThan(0);
    }

    // Sensitivities should be finite
    expect(Number.isFinite(report.results[0].maxThroughputSensitivity)).toBe(true);
    expect(Number.isFinite(report.results[0].maxTpotSensitivity)).toBe(true);
    expect(Number.isFinite(report.results[0].maxTtftSensitivity)).toBe(true);
  });

  it('PREFILL_RESIDUAL should have non-zero TTFT sensitivity', () => {
    const allParams = getAllPerturbableInferenceParams();
    const prefillParam = allParams.filter(p => p.name === 'PREFILL_RESIDUAL');
    const benchmarks = makeBenchmarks();

    const report = runInferenceSensitivityAnalysis(prefillParam, benchmarks, 0.10);

    // PREFILL_RESIDUAL directly affects prefill efficiency → TTFT changes
    for (const delta of report.results[0].deltas) {
      expect(delta.ttftMinus).not.toBe(delta.ttftPlus);
    }
  });

  it('TP_COMM_EFFICIENCY should have zero sensitivity on TP=1, non-zero on TP=4', () => {
    const allParams = getAllPerturbableInferenceParams();
    const tpParam = allParams.filter(p => p.name === 'TP_COMM_EFFICIENCY');
    const benchmarks = makeBenchmarks();

    const report = runInferenceSensitivityAnalysis(tpParam, benchmarks, 0.10);

    // TP=1 benchmark (LLaMA 8B): zero sensitivity
    const tp1Delta = report.results[0].deltas.find(d => d.benchmarkName.includes('8B'))!;
    expect(tp1Delta.throughputMinus).toBe(tp1Delta.throughputPlus);

    // TP=4 benchmark (LLaMA 70B): non-zero sensitivity
    const tp4Delta = report.results[0].deltas.find(d => d.benchmarkName.includes('70B'))!;
    expect(tp4Delta.throughputMinus).not.toBe(tp4Delta.throughputPlus);
  });

  it('BW_EFF_FLOOR should affect small model more than large model for TPOT', () => {
    const allParams = getAllPerturbableInferenceParams();
    const bwParam = allParams.filter(p => p.name === 'BW_EFF_FLOOR');
    const benchmarks = makeBenchmarks();

    const report = runInferenceSensitivityAnalysis(bwParam, benchmarks, 0.10);

    // Small model (8B) has lower weight bytes → closer to floor → more sensitive
    const smallDelta = report.results[0].deltas.find(d => d.benchmarkName.includes('8B'))!;
    const largeDelta = report.results[0].deltas.find(d => d.benchmarkName.includes('70B'))!;

    const smallSens = Math.abs(smallDelta.tpotPlus - smallDelta.tpotMinus) / smallDelta.tpotOriginal;
    const largeSens = Math.abs(largeDelta.tpotPlus - largeDelta.tpotMinus) / largeDelta.tpotOriginal;

    // Small model should be more sensitive to BW floor than large model
    expect(smallSens).toBeGreaterThan(largeSens);
  });

  it('should produce consistent baselines (no state leakage)', () => {
    const allParams = getAllPerturbableInferenceParams();
    const params = allParams.filter(p =>
      p.name === 'PREFILL_RESIDUAL' || p.name === 'BW_EFF_FLOOR'
    );
    const benchmarks = makeBenchmarks();

    const report1 = runInferenceSensitivityAnalysis(params, benchmarks, 0.10);
    const report2 = runInferenceSensitivityAnalysis(params, benchmarks, 0.10);

    // Same baselines across runs
    for (let i = 0; i < report1.results.length; i++) {
      for (let j = 0; j < report1.results[i].deltas.length; j++) {
        expect(report1.results[i].deltas[j].throughputOriginal)
          .toBe(report2.results[i].deltas[j].throughputOriginal);
      }
    }
  });
});
