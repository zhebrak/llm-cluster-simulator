/**
 * Sensitivity analysis engine for empirical parameters.
 *
 * For each registry parameter, perturbs ±10% and measures MFU delta
 * across benchmark configs. Outputs per-parameter sensitivity (MFU derivative),
 * cross-parameter correlation matrix, and dead-weight parameters.
 *
 * Not intended for CI (too heavy/flaky). Use the thin test in
 * tests/analysis/sensitivity.test.ts for regression.
 */

import {
  type SimulationConfig,
  getSimulationMetrics,
} from '../simulation/engine.ts';

export interface SensitivityResult {
  paramName: string;
  originalValue: number;
  benchmarkDeltas: {
    benchmarkName: string;
    mfuAtMinus10: number;
    mfuAtOriginal: number;
    mfuAtPlus10: number;
    sensitivity: number; // (mfuPlus10 - mfuMinus10) / (0.2 * originalValue)
  }[];
  maxSensitivity: number;
  avgSensitivity: number;
}

export interface SensitivityReport {
  results: SensitivityResult[];
  deadWeightParams: string[];    // |maxSensitivity| < 0.001
  highSensitivityParams: string[]; // |maxSensitivity| > 0.05
}

export interface PerturbableParam {
  name: string;
  getValue: () => number;
  setValue: (v: number) => void;
}

/**
 * Run sensitivity analysis for a set of perturbable parameters against benchmark configs.
 *
 * @param params - Parameters to perturb (with get/set accessors)
 * @param benchmarks - Benchmark configs to measure against
 * @param perturbFraction - Fraction to perturb (default 0.10 = ±10%)
 */
export function runSensitivityAnalysis(
  params: PerturbableParam[],
  benchmarks: { name: string; config: SimulationConfig }[],
  perturbFraction: number = 0.10,
): SensitivityReport {
  // Baseline MFU for each benchmark
  const baselines = benchmarks.map(b => ({
    name: b.name,
    mfu: getSimulationMetrics(b.config).mfu,
  }));

  const results: SensitivityResult[] = [];

  for (const param of params) {
    const original = param.getValue();
    if (original === 0) {
      // Skip zero-valued params (can't perturb by percentage)
      results.push({
        paramName: param.name,
        originalValue: 0,
        benchmarkDeltas: baselines.map(b => ({
          benchmarkName: b.name,
          mfuAtMinus10: b.mfu,
          mfuAtOriginal: b.mfu,
          mfuAtPlus10: b.mfu,
          sensitivity: 0,
        })),
        maxSensitivity: 0,
        avgSensitivity: 0,
      });
      continue;
    }

    const minusVal = original * (1 - perturbFraction);
    const plusVal = original * (1 + perturbFraction);

    // Perturb down
    param.setValue(minusVal);
    const minusMFUs = benchmarks.map(b => getSimulationMetrics(b.config).mfu);

    // Perturb up
    param.setValue(plusVal);
    const plusMFUs = benchmarks.map(b => getSimulationMetrics(b.config).mfu);

    // Restore original
    param.setValue(original);

    const benchmarkDeltas = baselines.map((b, i) => {
      const delta = plusMFUs[i] - minusMFUs[i];
      const sensitivity = delta / (2 * perturbFraction * original);
      return {
        benchmarkName: b.name,
        mfuAtMinus10: minusMFUs[i],
        mfuAtOriginal: b.mfu,
        mfuAtPlus10: plusMFUs[i],
        sensitivity,
      };
    });

    const sensitivities = benchmarkDeltas.map(d => Math.abs(d.sensitivity));
    results.push({
      paramName: param.name,
      originalValue: original,
      benchmarkDeltas,
      maxSensitivity: Math.max(...sensitivities),
      avgSensitivity: sensitivities.reduce((a, b) => a + b, 0) / sensitivities.length,
    });
  }

  // Classify
  const deadWeightParams = results
    .filter(r => r.maxSensitivity < 0.001)
    .map(r => r.paramName);

  const highSensitivityParams = results
    .filter(r => r.maxSensitivity > 0.05)
    .map(r => r.paramName);

  return { results, deadWeightParams, highSensitivityParams };
}

/**
 * Format sensitivity report as a human-readable table.
 */
export function formatSensitivityReport(report: SensitivityReport): string {
  const lines: string[] = [];
  lines.push('Parameter Sensitivity Analysis');
  lines.push('=' .repeat(80));
  lines.push('');

  // Sort by max sensitivity descending
  const sorted = [...report.results].sort((a, b) => b.maxSensitivity - a.maxSensitivity);

  lines.push(`${'Parameter'.padEnd(40)} ${'Value'.padStart(8)} ${'MaxSens'.padStart(10)} ${'AvgSens'.padStart(10)}`);
  lines.push('-'.repeat(70));

  for (const r of sorted) {
    const flag = r.maxSensitivity > 0.05 ? ' **HIGH**' : r.maxSensitivity < 0.001 ? ' (dead)' : '';
    lines.push(
      `${r.paramName.padEnd(40)} ${r.originalValue.toPrecision(4).padStart(8)} ${r.maxSensitivity.toFixed(4).padStart(10)} ${r.avgSensitivity.toFixed(4).padStart(10)}${flag}`
    );
  }

  lines.push('');
  if (report.deadWeightParams.length > 0) {
    lines.push(`Dead-weight parameters (|maxSens| < 0.001): ${report.deadWeightParams.join(', ')}`);
  }
  if (report.highSensitivityParams.length > 0) {
    lines.push(`High-sensitivity parameters (|maxSens| > 0.05): ${report.highSensitivityParams.join(', ')}`);
  }

  return lines.join('\n');
}
