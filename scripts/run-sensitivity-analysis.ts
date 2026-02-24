#!/usr/bin/env npx tsx
/**
 * Sensitivity Analysis Script
 *
 * Perturbs each perturbable fitted parameter by ±5% and measures
 * MFU delta across Tier 1+2 benchmarks.
 *
 * Usage:
 *   npx tsx scripts/run-sensitivity-analysis.ts
 */

import { ALL_PUBLISHED_CONFIGS, toSimConfig } from '../src/data/published-training-configs.ts';
import { getSimulationMetrics } from '../src/core/simulation/engine.ts';
import { runSensitivityAnalysis, formatSensitivityReport } from '../src/core/analysis/sensitivity.ts';
import { getAllPerturbableFittedParams, resetAllPerturbableParams } from '../src/core/analysis/perturbable-params.ts';

// ---------------------------------------------------------------------------
// Build benchmark configs from Tier 1+2 published configs
// ---------------------------------------------------------------------------

const configs = ALL_PUBLISHED_CONFIGS.filter(c => c.tier <= 2);
const benchmarks = configs.map(pc => ({
  name: pc.label,
  config: toSimConfig(pc),
}));

console.log(`Running sensitivity analysis with ${benchmarks.length} benchmarks...`);
console.log(`Benchmarks: ${benchmarks.map(b => b.name).join(', ')}`);
console.log();

// ---------------------------------------------------------------------------
// Run sensitivity analysis
// ---------------------------------------------------------------------------

const params = getAllPerturbableFittedParams();
console.log(`Perturbable parameters: ${params.length}`);
console.log();

const PERTURB_FRACTION = 0.05; // ±5%
const report = runSensitivityAnalysis(params, benchmarks, PERTURB_FRACTION);

// Reset all params to defaults after analysis
resetAllPerturbableParams();

// ---------------------------------------------------------------------------
// Output formatted report
// ---------------------------------------------------------------------------

console.log(formatSensitivityReport(report));
console.log();

// ---------------------------------------------------------------------------
// Classification summary
// ---------------------------------------------------------------------------

console.log('Classification Summary');
console.log('='.repeat(80));
console.log();

const sorted = [...report.results].sort((a, b) => b.maxSensitivity - a.maxSensitivity);

// Structural guards: extrapolation safety nets that are unreachable in any practical config
const STRUCTURAL_GUARDS = new Set(['DP_BW_FLOOR']);

function classify(paramName: string, maxDelta: number): string {
  if (maxDelta > 0.01) return 'high-sensitivity';
  if (maxDelta >= 0.00005) return 'low-sensitivity';
  if (STRUCTURAL_GUARDS.has(paramName)) return 'structural guard';
  return 'dead-weight';
}

console.log('### High-sensitivity (>1% MFU per 5% change)');
console.log('These parameters are well-constrained — rounding would cause visible MFU shifts.');
console.log();
for (const r of sorted) {
  const maxDelta = Math.max(...r.benchmarkDeltas.map(d => Math.abs(d.mfuAtPlus10 - d.mfuAtMinus10)));
  if (classify(r.paramName, maxDelta) !== 'high-sensitivity') continue;
  console.log(`  ${r.paramName}: value=${r.originalValue}, max |ΔMFU|=${(maxDelta * 100).toFixed(2)}pp`);
}

console.log();
console.log('### Low-sensitivity (<1% MFU per 5% change)');
console.log('These parameters could be rounded to clean values without loss.');
console.log();
for (const r of sorted) {
  const maxDelta = Math.max(...r.benchmarkDeltas.map(d => Math.abs(d.mfuAtPlus10 - d.mfuAtMinus10)));
  if (classify(r.paramName, maxDelta) !== 'low-sensitivity') continue;
  console.log(`  ${r.paramName}: value=${r.originalValue}, max |ΔMFU|=${(maxDelta * 100).toFixed(2)}pp`);
}

console.log();
console.log('### Structural guards (extrapolation safety nets, unreachable in practice)');
console.log('These exist to prevent unbounded extrapolation at extreme DP scales.');
console.log();
for (const r of sorted) {
  const maxDelta = Math.max(...r.benchmarkDeltas.map(d => Math.abs(d.mfuAtPlus10 - d.mfuAtMinus10)));
  if (classify(r.paramName, maxDelta) !== 'structural guard') continue;
  console.log(`  ${r.paramName}: value=${r.originalValue}, max |ΔMFU|=${(maxDelta * 100).toFixed(2)}pp`);
}

console.log();
console.log('### Dead-weight (zero sensitivity across all benchmarks)');
console.log('These parameters have no measurable effect on any benchmark config.');
console.log();
for (const r of sorted) {
  const maxDelta = Math.max(...r.benchmarkDeltas.map(d => Math.abs(d.mfuAtPlus10 - d.mfuAtMinus10)));
  if (classify(r.paramName, maxDelta) !== 'dead-weight') continue;
  console.log(`  ${r.paramName}: value=${r.originalValue}, max |ΔMFU|=${(maxDelta * 100).toFixed(2)}pp`);
}

// ---------------------------------------------------------------------------
// Markdown table for PHYSICS.md
// ---------------------------------------------------------------------------

console.log();
console.log('### Sensitivity Results Table (for PHYSICS.md)');
console.log();
console.log('| Parameter | Value | Max \\|ΔMFU\\| (±5%) | Classification |');
console.log('|-----------|-------|---------------------|----------------|');
for (const r of sorted) {
  const maxDelta = Math.max(...r.benchmarkDeltas.map(d => Math.abs(d.mfuAtPlus10 - d.mfuAtMinus10)));
  const classification = classify(r.paramName, maxDelta);
  const valStr = typeof r.originalValue === 'number' && r.originalValue < 0.001
    ? r.originalValue.toExponential(2)
    : r.originalValue.toPrecision(3);
  console.log(`| ${r.paramName} | ${valStr} | ${(maxDelta * 100).toFixed(2)}pp | ${classification} |`);
}
