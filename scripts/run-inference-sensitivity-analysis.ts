#!/usr/bin/env npx tsx
/**
 * Inference Sensitivity Analysis Script
 *
 * Perturbs each perturbable inference parameter by ±5% and measures
 * throughput/TPOT/TTFT deltas across 5 inference benchmarks.
 *
 * Usage:
 *   npx tsx scripts/run-inference-sensitivity-analysis.ts
 */

import {
  runInferenceSensitivityAnalysis,
  formatInferenceSensitivityReport,
} from '../src/core/analysis/inference-sensitivity.ts';
import {
  getAllPerturbableInferenceParams,
  resetAllPerturbableInferenceParams,
} from '../src/core/analysis/inference-perturbable-params.ts';
import type { InferenceSimulationConfig } from '../src/core/inference/simulation.ts';

// ---------------------------------------------------------------------------
// 5 Inference Benchmarks
// ---------------------------------------------------------------------------

const benchmarks: { name: string; config: InferenceSimulationConfig }[] = [
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

console.log(`Running inference sensitivity analysis with ${benchmarks.length} benchmarks...`);
console.log(`Benchmarks: ${benchmarks.map(b => b.name).join(', ')}`);
console.log();

// ---------------------------------------------------------------------------
// Run sensitivity analysis
// ---------------------------------------------------------------------------

const params = getAllPerturbableInferenceParams();
console.log(`Perturbable inference parameters: ${params.length}`);
console.log();

const PERTURB_FRACTION = 0.05; // ±5%
const report = runInferenceSensitivityAnalysis(params, benchmarks, PERTURB_FRACTION);

// Reset all params to defaults after analysis
resetAllPerturbableInferenceParams();

// ---------------------------------------------------------------------------
// Output formatted report
// ---------------------------------------------------------------------------

console.log(formatInferenceSensitivityReport(report));
console.log();

// ---------------------------------------------------------------------------
// Classification summary
// ---------------------------------------------------------------------------

console.log('Classification Summary');
console.log('='.repeat(100));
console.log();

// Sort by max overall sensitivity
const sorted = [...report.results].sort((a, b) => {
  const aMax = Math.max(
    ...a.deltas.map(d => d.throughputOriginal > 0 ? Math.abs(d.throughputPlus - d.throughputMinus) / d.throughputOriginal : 0)
  );
  const bMax = Math.max(
    ...b.deltas.map(d => d.throughputOriginal > 0 ? Math.abs(d.throughputPlus - d.throughputMinus) / d.throughputOriginal : 0)
  );
  return bMax - aMax;
});

// Conditional params: zero on some configs, non-zero on others
const CONDITIONAL_PARAMS = new Set([
  'NVLINK_PER_ROUND_MS', 'PCIE_PER_ROUND_MS', 'TP_COMM_EFFICIENCY',
  'EP_PREFILL_OVERLAP',
  'CB_SCHEDULING_BASE', 'CB_PREFILL_INTERFERENCE_MAX',
]);

function classify(paramName: string, maxTputDelta: number, isAllZero: boolean): string {
  if (isAllZero && paramName === 'MEMORY_OVERHEAD_FACTOR') return 'structural zero';
  if (isAllZero) return 'dead-weight';
  if (maxTputDelta > 0.01) return 'high-sensitivity';
  if (CONDITIONAL_PARAMS.has(paramName)) return 'conditional';
  if (maxTputDelta >= 0.0001) return 'low-sensitivity';
  return 'dead-weight';
}

for (const category of ['high-sensitivity', 'low-sensitivity', 'conditional', 'structural zero', 'dead-weight'] as const) {
  const categoryLabel: Record<string, string> = {
    'high-sensitivity': '### High-sensitivity (>1% throughput per 5% change)',
    'low-sensitivity': '### Low-sensitivity (<1% throughput per 5% change)',
    'conditional': '### Conditional (zero on some configs, active on others)',
    'structural zero': '### Structural zero (memory-only, no latency effect)',
    'dead-weight': '### Dead-weight (zero sensitivity across all benchmarks)',
  };
  console.log(categoryLabel[category]);
  console.log();

  for (const r of sorted) {
    const maxTputDelta = Math.max(
      ...r.deltas.map(d => d.throughputOriginal > 0 ? Math.abs(d.throughputPlus - d.throughputMinus) / d.throughputOriginal : 0)
    );
    const isAllZero = r.deltas.every(d =>
      d.throughputMinus === d.throughputPlus &&
      d.tpotMinus === d.tpotPlus &&
      d.ttftMinus === d.ttftPlus
    );

    if (classify(r.paramName, maxTputDelta, isAllZero) !== category) continue;
    console.log(`  ${r.paramName}: value=${r.originalValue}, max |ΔThroughput|=${(maxTputDelta * 100).toFixed(2)}%`);
  }
  console.log();
}

// ---------------------------------------------------------------------------
// Markdown table for PHYSICS.md
// ---------------------------------------------------------------------------

console.log('### Sensitivity Results Table (for PHYSICS.md)');
console.log();
console.log('| Parameter | Value | Tier | Max |ΔThroughput| | Max |ΔTPOT| | Max |ΔTTFT| | Classification |');
console.log('|-----------|-------|------|---------------------|-------------|-------------|----------------|');

const TIER_MAP: Record<string, string> = {
  'PREFILL_RESIDUAL': 'fitted',
  'BW_EFF_FLOOR': 'fitted',
  'BW_EFF_SCALE': 'fitted',
  'EP_PREFILL_OVERLAP': 'fitted',
  'CB_PREFILL_INTERFERENCE_MAX': 'fitted',
  'DECODE_SAMPLING_OVERHEAD_MS': 'grounded-empirical',
  'NVLINK_PER_ROUND_MS': 'grounded-empirical',
  'PCIE_PER_ROUND_MS': 'grounded-empirical',
  'TP_COMM_EFFICIENCY': 'grounded-empirical',
  'CB_SCHEDULING_BASE': 'grounded-empirical',
  'MEMORY_OVERHEAD_FACTOR': 'grounded-empirical',
};

for (const r of sorted) {
  const maxTputDelta = Math.max(
    ...r.deltas.map(d => d.throughputOriginal > 0 ? Math.abs(d.throughputPlus - d.throughputMinus) / d.throughputOriginal : 0)
  );
  const maxTpotDelta = Math.max(
    ...r.deltas.map(d => d.tpotOriginal > 0 ? Math.abs(d.tpotPlus - d.tpotMinus) / d.tpotOriginal : 0)
  );
  const maxTtftDelta = Math.max(
    ...r.deltas.map(d => d.ttftOriginal > 0 ? Math.abs(d.ttftPlus - d.ttftMinus) / d.ttftOriginal : 0)
  );
  const isAllZero = r.deltas.every(d =>
    d.throughputMinus === d.throughputPlus &&
    d.tpotMinus === d.tpotPlus &&
    d.ttftMinus === d.ttftPlus
  );
  const classification = classify(r.paramName, maxTputDelta, isAllZero);
  const tier = TIER_MAP[r.paramName] ?? 'unknown';
  const valStr = r.originalValue < 0.001 ? r.originalValue.toExponential(2) : r.originalValue.toPrecision(3);

  console.log(`| ${r.paramName} | ${valStr} | ${tier} | ${(maxTputDelta * 100).toFixed(2)}% | ${(maxTpotDelta * 100).toFixed(2)}% | ${(maxTtftDelta * 100).toFixed(2)}% | ${classification} |`);
}
