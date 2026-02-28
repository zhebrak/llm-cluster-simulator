/**
 * Inference Sensitivity Analysis Engine
 *
 * For each perturbable inference parameter, perturbs ±N% and measures
 * throughput, TPOT, and TTFT deltas across benchmark configs.
 *
 * Mirrors the training sensitivity engine (sensitivity.ts) but targets
 * inference metrics instead of MFU.
 */

import type { PerturbableParam } from './sensitivity.ts';
import {
  runInferenceSimulationRaw,
  type InferenceSimulationConfig,
} from '../inference/simulation.ts';

export interface InferenceSensitivityBenchmarkDelta {
  benchmarkName: string;
  throughputMinus: number;
  throughputOriginal: number;
  throughputPlus: number;
  throughputSensitivity: number;
  tpotMinus: number;
  tpotOriginal: number;
  tpotPlus: number;
  tpotSensitivity: number;
  ttftMinus: number;
  ttftOriginal: number;
  ttftPlus: number;
  ttftSensitivity: number;
}

export interface InferenceSensitivityResult {
  paramName: string;
  originalValue: number;
  deltas: InferenceSensitivityBenchmarkDelta[];
  maxThroughputSensitivity: number;
  maxTpotSensitivity: number;
  maxTtftSensitivity: number;
}

export interface InferenceSensitivityReport {
  results: InferenceSensitivityResult[];
  deadWeightParams: string[];       // zero sensitivity on all metrics/benchmarks
  highSensitivityParams: string[];  // large effect on ≥1 metric
}

interface BenchmarkMetrics {
  throughput: number;
  tpot: number;
  ttft: number;
}

function runBenchmark(config: InferenceSimulationConfig): BenchmarkMetrics {
  const result = runInferenceSimulationRaw(config);
  return {
    throughput: result.throughput.tokensPerSecond,
    tpot: result.latency.tpot,
    ttft: result.latency.ttft,
  };
}

/**
 * Run inference sensitivity analysis for a set of perturbable parameters
 * against benchmark configs.
 *
 * @param params - Parameters to perturb (with get/set accessors)
 * @param benchmarks - Inference benchmark configs
 * @param perturbFraction - Fraction to perturb (default 0.10 = ±10%)
 */
export function runInferenceSensitivityAnalysis(
  params: PerturbableParam[],
  benchmarks: { name: string; config: InferenceSimulationConfig }[],
  perturbFraction: number = 0.10,
): InferenceSensitivityReport {
  // Baseline metrics for each benchmark
  const baselines = benchmarks.map(b => ({
    name: b.name,
    metrics: runBenchmark(b.config),
  }));

  const results: InferenceSensitivityResult[] = [];

  for (const param of params) {
    const original = param.getValue();
    if (original === 0) {
      results.push({
        paramName: param.name,
        originalValue: 0,
        deltas: baselines.map(b => ({
          benchmarkName: b.name,
          throughputMinus: b.metrics.throughput,
          throughputOriginal: b.metrics.throughput,
          throughputPlus: b.metrics.throughput,
          throughputSensitivity: 0,
          tpotMinus: b.metrics.tpot,
          tpotOriginal: b.metrics.tpot,
          tpotPlus: b.metrics.tpot,
          tpotSensitivity: 0,
          ttftMinus: b.metrics.ttft,
          ttftOriginal: b.metrics.ttft,
          ttftPlus: b.metrics.ttft,
          ttftSensitivity: 0,
        })),
        maxThroughputSensitivity: 0,
        maxTpotSensitivity: 0,
        maxTtftSensitivity: 0,
      });
      continue;
    }

    const minusVal = original * (1 - perturbFraction);
    const plusVal = original * (1 + perturbFraction);

    // Perturb down
    param.setValue(minusVal);
    const minusMetrics = benchmarks.map(b => runBenchmark(b.config));

    // Perturb up
    param.setValue(plusVal);
    const plusMetrics = benchmarks.map(b => runBenchmark(b.config));

    // Restore original
    param.setValue(original);

    const deltas: InferenceSensitivityBenchmarkDelta[] = baselines.map((b, i) => {
      const denom = 2 * perturbFraction * original;
      return {
        benchmarkName: b.name,
        throughputMinus: minusMetrics[i].throughput,
        throughputOriginal: b.metrics.throughput,
        throughputPlus: plusMetrics[i].throughput,
        throughputSensitivity: (plusMetrics[i].throughput - minusMetrics[i].throughput) / denom,
        tpotMinus: minusMetrics[i].tpot,
        tpotOriginal: b.metrics.tpot,
        tpotPlus: plusMetrics[i].tpot,
        tpotSensitivity: (plusMetrics[i].tpot - minusMetrics[i].tpot) / denom,
        ttftMinus: minusMetrics[i].ttft,
        ttftOriginal: b.metrics.ttft,
        ttftPlus: plusMetrics[i].ttft,
        ttftSensitivity: (plusMetrics[i].ttft - minusMetrics[i].ttft) / denom,
      };
    });

    const throughputSensitivities = deltas.map(d => Math.abs(d.throughputSensitivity));
    const tpotSensitivities = deltas.map(d => Math.abs(d.tpotSensitivity));
    const ttftSensitivities = deltas.map(d => Math.abs(d.ttftSensitivity));

    results.push({
      paramName: param.name,
      originalValue: original,
      deltas,
      maxThroughputSensitivity: Math.max(...throughputSensitivities),
      maxTpotSensitivity: Math.max(...tpotSensitivities),
      maxTtftSensitivity: Math.max(...ttftSensitivities),
    });
  }

  // Classify
  const deadWeightParams = results
    .filter(r => r.maxThroughputSensitivity < 0.001 && r.maxTpotSensitivity < 0.001 && r.maxTtftSensitivity < 0.001)
    .map(r => r.paramName);

  const highSensitivityParams = results
    .filter(r => {
      // High-sensitivity: any benchmark where throughput changes >1% per perturbFraction change
      return r.deltas.some(d => {
        if (d.throughputOriginal <= 0) return false;
        const throughputDelta = Math.abs(d.throughputPlus - d.throughputMinus) / d.throughputOriginal;
        return throughputDelta > 0.01;
      });
    })
    .map(r => r.paramName);

  return { results, deadWeightParams, highSensitivityParams };
}

/**
 * Format inference sensitivity report as a human-readable table.
 */
export function formatInferenceSensitivityReport(report: InferenceSensitivityReport): string {
  const lines: string[] = [];
  lines.push('Inference Parameter Sensitivity Analysis');
  lines.push('='.repeat(100));
  lines.push('');

  // Sort by max throughput sensitivity descending
  const sorted = [...report.results].sort((a, b) => {
    const aMax = Math.max(a.maxThroughputSensitivity, a.maxTpotSensitivity, a.maxTtftSensitivity);
    const bMax = Math.max(b.maxThroughputSensitivity, b.maxTpotSensitivity, b.maxTtftSensitivity);
    return bMax - aMax;
  });

  lines.push(
    `${'Parameter'.padEnd(35)} ${'Value'.padStart(8)} ${'MaxΔTput'.padStart(12)} ${'MaxΔTPOT'.padStart(12)} ${'MaxΔTTFT'.padStart(12)}`
  );
  lines.push('-'.repeat(85));

  for (const r of sorted) {
    // Compute max absolute deltas as percentage of original
    const maxTputDelta = Math.max(...r.deltas.map(d =>
      d.throughputOriginal > 0 ? Math.abs(d.throughputPlus - d.throughputMinus) / d.throughputOriginal : 0
    ));
    const maxTpotDelta = Math.max(...r.deltas.map(d =>
      d.tpotOriginal > 0 ? Math.abs(d.tpotPlus - d.tpotMinus) / d.tpotOriginal : 0
    ));
    const maxTtftDelta = Math.max(...r.deltas.map(d =>
      d.ttftOriginal > 0 ? Math.abs(d.ttftPlus - d.ttftMinus) / d.ttftOriginal : 0
    ));

    const isHigh = report.highSensitivityParams.includes(r.paramName);
    const isDead = report.deadWeightParams.includes(r.paramName);
    const flag = isHigh ? ' **HIGH**' : isDead ? ' (dead)' : '';

    lines.push(
      `${r.paramName.padEnd(35)} ${r.originalValue.toPrecision(3).padStart(8)} ${(maxTputDelta * 100).toFixed(2).padStart(11)}% ${(maxTpotDelta * 100).toFixed(2).padStart(11)}% ${(maxTtftDelta * 100).toFixed(2).padStart(11)}%${flag}`
    );
  }

  lines.push('');
  if (report.deadWeightParams.length > 0) {
    lines.push(`Dead-weight parameters: ${report.deadWeightParams.join(', ')}`);
  }
  if (report.highSensitivityParams.length > 0) {
    lines.push(`High-sensitivity parameters: ${report.highSensitivityParams.join(', ')}`);
  }

  return lines.join('\n');
}
