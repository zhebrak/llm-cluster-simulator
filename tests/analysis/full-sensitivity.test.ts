/**
 * Full Sensitivity Regression Test
 *
 * Guarded by RUN_SLOW_TESTS — perturbs all 11 perturbable parameters
 * across 3 representative benchmarks (single-node dense, multi-node 3D, MoE).
 *
 * Asserts:
 *   - Every perturbable param has non-zero sensitivity on at least 1 benchmark
 *   - runtimeResidual is classified high-sensitivity
 *
 * Run: RUN_SLOW_TESTS=1 npx vitest run tests/analysis/full-sensitivity.test.ts
 */

import { describe, it, expect, afterEach } from 'vitest';
import { runSensitivityAnalysis } from '../../src/core/analysis/sensitivity.ts';
import { getAllPerturbableFittedParams, resetAllPerturbableParams } from '../../src/core/analysis/perturbable-params.ts';
import { PUBLISHED, toSimConfig } from '../../src/data/published-training-configs.ts';

describe.skipIf(!process.env.RUN_SLOW_TESTS)('Full Sensitivity Analysis', () => {
  afterEach(() => {
    resetAllPerturbableParams();
  });

  // 3 representative benchmarks:
  // 1. IBM LLaMA 7B on 128 A100 — pure FSDP, single-node dense
  // 2. LLaMA 3.1 405B on 16384 H100 — multi-node 3D parallel
  // 3. DeepSeek V3 on 2048 H800 — MoE with EP
  const benchmarks = [
    { name: 'IBM LLaMA 7B (FSDP)', config: toSimConfig(PUBLISHED.ibm_llama2_7b) },
    { name: 'LLaMA 405B (3D)', config: toSimConfig(PUBLISHED.llama3_405b_8k) },
    { name: 'DeepSeek V3 (MoE)', config: toSimConfig(PUBLISHED.deepseek_v3_fp8_h800) },
  ];

  // Parameters expected to have zero/negligible sensitivity on these 3 benchmarks:
  // - DP_BW_FLOOR: binds at dp~51000, all benchmarks dp < 200
  // - CP all-gather overlap (interleaved): no CP in any benchmark
  // - EXPERT_GEMM_EXPONENT: only affects fine-grained MoE (>64 experts); no such model in this set
  // - EXPERT_COUNT_SCALE: only affects fine-grained MoE (>64 experts); no such model in this set
  const expectedZeroSensitivity = new Set([
    'DP_BW_FLOOR',
    'CP all-gather overlap (interleaved)',
    'EXPERT_GEMM_EXPONENT',
    'EXPERT_COUNT_SCALE',
  ]);

  it('perturbable fitted params have non-zero sensitivity where expected', () => {
    const params = getAllPerturbableFittedParams();
    expect(params.length).toBe(11);

    const report = runSensitivityAnalysis(params, benchmarks, 0.05);

    for (const result of report.results) {
      if (expectedZeroSensitivity.has(result.paramName)) continue;

      const hasNonZero = result.benchmarkDeltas.some(d =>
        Math.abs(d.mfuAtPlus10 - d.mfuAtMinus10) > 1e-6,
      );
      expect(
        hasNonZero,
        `Parameter '${result.paramName}' has zero sensitivity on all benchmarks — ` +
        `may be dead code or not connected to the training MFU path`,
      ).toBe(true);
    }

    // At least 5 of 11 params should have non-zero sensitivity on these 3 benchmarks
    const nonZero = report.results.filter(r =>
      r.benchmarkDeltas.some(d => Math.abs(d.mfuAtPlus10 - d.mfuAtMinus10) > 1e-6),
    );
    expect(nonZero.length).toBeGreaterThanOrEqual(5);
  });

  it('runtimeResidual is high-sensitivity', () => {
    const params = getAllPerturbableFittedParams();
    const report = runSensitivityAnalysis(params, benchmarks, 0.05);

    const rr = report.results.find(r => r.paramName === 'runtimeResidual');
    expect(rr).toBeDefined();

    // runtimeResidual directly multiplies efficiency — 5% perturbation should cause >1pp MFU shift
    const maxDelta = Math.max(
      ...rr!.benchmarkDeltas.map(d => Math.abs(d.mfuAtPlus10 - d.mfuAtMinus10)),
    );
    expect(
      maxDelta,
      `runtimeResidual max |ΔMFU| = ${(maxDelta * 100).toFixed(2)}pp, expected > 1pp`,
    ).toBeGreaterThan(0.01);
  });
});
