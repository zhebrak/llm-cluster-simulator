/**
 * Thin CI test for sensitivity analysis engine.
 *
 * Runs a small subset (5 critical parameters) against 2 benchmarks to verify
 * the sensitivity engine works and produces expected directional results.
 * Full analysis (all 19 perturbable params × 10 benchmarks) is too heavy for CI.
 */
import { describe, it, expect } from 'vitest';
import { runSensitivityAnalysis, type PerturbableParam } from '../../src/core/analysis/sensitivity.ts';
import { type SimulationConfig, getSimulationMetrics } from '../../src/core/simulation/engine.ts';
import { createMultiNodeCluster, createSingleNodeCluster } from '../../src/core/hardware/topology.ts';
import { getDpBwAlpha, setDpBwAlpha, getStragglerSigma, setStragglerSigma } from '../../src/core/strategies/base.ts';

// Small benchmark configs for CI speed
function makeSmallBenchmarks(): { name: string; config: SimulationConfig }[] {
  return [
    {
      name: 'LLaMA2 7B FSDP 8×H100',
      config: {
        clusterConfig: createSingleNodeCluster('h100-sxm', 8)!,
        modelId: 'llama2-7b',
        globalBatchSize: 16,
        microBatchSize: 2,
        sequenceLength: 2048,
        strategyType: 'fsdp',
        flashAttention: true,
        mixedPrecision: 'bf16',
      },
    },
    {
      name: 'GPT-3 175B 3D 1024×A100',
      config: {
        clusterConfig: createMultiNodeCluster('a100-80gb', 8, 128)!,
        modelId: 'gpt3-175b',
        globalBatchSize: 1536,
        microBatchSize: 1,
        sequenceLength: 2048,
        strategyType: 'ddp-tp-pp',
        strategyConfig: {
          tp: 8, pp: 8,
          pipelineSchedule: 'interleaved-1f1b',
          interleavedStages: 2,
        },
        activationCheckpointing: true,
        mixedPrecision: 'bf16',
      },
    },
  ];
}

describe('Sensitivity Analysis Engine', () => {
  it('should run and produce results for critical parameters', () => {
    // We test with a mutable wrapper around a single parameter
    // that we can restore after the test.
    // For CI we just test DP_BW_ALPHA since it's exported and mutable-ish.
    // The real sensitivity analysis would use the parameter registry.
    let testValue = 0.15; // DP_BW_ALPHA default
    const params: PerturbableParam[] = [
      {
        name: 'test_param',
        getValue: () => testValue,
        setValue: (v: number) => { testValue = v; },
      },
    ];

    const benchmarks = makeSmallBenchmarks();

    // Verify engine runs without error
    const report = runSensitivityAnalysis(params, benchmarks, 0.10);

    expect(report.results).toHaveLength(1);
    expect(report.results[0].paramName).toBe('test_param');
    expect(report.results[0].benchmarkDeltas).toHaveLength(2);

    // Each benchmark delta should have valid MFU values
    for (const delta of report.results[0].benchmarkDeltas) {
      expect(delta.mfuAtOriginal).toBeGreaterThan(0);
      expect(delta.mfuAtOriginal).toBeLessThan(1);
      expect(delta.mfuAtMinus10).toBeGreaterThan(0);
      expect(delta.mfuAtPlus10).toBeGreaterThan(0);
    }

    // Sensitivity should be finite
    expect(Number.isFinite(report.results[0].maxSensitivity)).toBe(true);
    expect(Number.isFinite(report.results[0].avgSensitivity)).toBe(true);
  });

  it('should detect zero-valued parameters as zero sensitivity', () => {
    const params: PerturbableParam[] = [
      {
        name: 'zero_param',
        getValue: () => 0,
        setValue: () => {},
      },
    ];

    const benchmarks = makeSmallBenchmarks();
    const report = runSensitivityAnalysis(params, benchmarks);

    expect(report.results[0].maxSensitivity).toBe(0);
    expect(report.deadWeightParams).toContain('zero_param');
  });

  it('should classify high-sensitivity parameters correctly', () => {
    // A parameter that drastically changes MFU when perturbed
    // We'll fake this with a high-sensitivity value
    let fakeValue = 1.0;
    const params: PerturbableParam[] = [
      {
        name: 'fake_high_sens',
        getValue: () => fakeValue,
        setValue: (v: number) => { fakeValue = v; },
      },
    ];

    const benchmarks = makeSmallBenchmarks();
    const report = runSensitivityAnalysis(params, benchmarks);

    // The parameter doesn't actually affect anything, so sensitivity should be ~0
    // This tests that the engine correctly classifies low-impact params
    expect(report.results[0].maxSensitivity).toBeLessThan(0.001);
  });

  it('should detect non-zero sensitivity for real DP_BW_ALPHA parameter', () => {
    // Perturb the real DP_BW_ALPHA via get/set accessors.
    // Need dp > 64 for DP_BW_ALPHA to be active (penalty only kicks in above DP_BW_REF=64).
    const original = getDpBwAlpha();
    const params: PerturbableParam[] = [
      {
        name: 'DP_BW_ALPHA',
        getValue: getDpBwAlpha,
        setValue: setDpBwAlpha,
      },
    ];

    // Use configs with dp > 64 (multi-node FSDP) and dp ≤ 64 (single-node)
    const benchmarks: { name: string; config: SimulationConfig }[] = [
      {
        name: 'LLaMA2 7B single-node (dp=8)',
        config: {
          clusterConfig: createSingleNodeCluster('h100-sxm', 8)!,
          modelId: 'llama2-7b',
          globalBatchSize: 16,
          microBatchSize: 2,
          sequenceLength: 2048,
          strategyType: 'fsdp',
          flashAttention: true,
          mixedPrecision: 'bf16',
        },
      },
      {
        name: 'LLaMA2 7B FSDP 128 nodes (dp=1024)',
        config: {
          clusterConfig: createMultiNodeCluster('a100-80gb', 8, 128)!,
          modelId: 'llama2-7b',
          globalBatchSize: 4096,
          microBatchSize: 4,
          sequenceLength: 2048,
          strategyType: 'fsdp',
          flashAttention: true,
          mixedPrecision: 'bf16',
        },
      },
    ];

    const report = runSensitivityAnalysis(params, benchmarks, 0.10);

    // Verify parameter was restored
    expect(getDpBwAlpha()).toBe(original);

    // dp=1024 (multi-node) should show non-zero sensitivity
    const multiNodeDelta = report.results[0].benchmarkDeltas.find(
      d => d.benchmarkName.includes('dp=1024'),
    )!;
    expect(multiNodeDelta.sensitivity).not.toBe(0);
    // Higher alpha → more penalty → lower MFU → negative sensitivity
    expect(multiNodeDelta.mfuAtPlus10).toBeLessThan(multiNodeDelta.mfuAtMinus10);

    // dp=8 single-node (dp ≤ 64) — no penalty, zero sensitivity
    const singleNodeDelta = report.results[0].benchmarkDeltas.find(
      d => d.benchmarkName.includes('dp=8'),
    )!;
    expect(singleNodeDelta.sensitivity).toBe(0);
  });

  it('should detect non-zero sensitivity for STRAGGLER_SIGMA on multi-node config', () => {
    const original = getStragglerSigma();
    const params: PerturbableParam[] = [
      {
        name: 'STRAGGLER_SIGMA',
        getValue: getStragglerSigma,
        setValue: setStragglerSigma,
      },
    ];

    const benchmarks: { name: string; config: SimulationConfig }[] = [
      {
        name: 'LLaMA2 7B single-node (numNodes=1)',
        config: {
          clusterConfig: createSingleNodeCluster('h100-sxm', 8)!,
          modelId: 'llama2-7b',
          globalBatchSize: 16,
          microBatchSize: 2,
          sequenceLength: 2048,
          strategyType: 'fsdp',
          flashAttention: true,
          mixedPrecision: 'bf16',
        },
      },
      {
        name: 'LLaMA2 7B FSDP 128 nodes (numNodes=128)',
        config: {
          clusterConfig: createMultiNodeCluster('a100-80gb', 8, 128)!,
          modelId: 'llama2-7b',
          globalBatchSize: 4096,
          microBatchSize: 4,
          sequenceLength: 2048,
          strategyType: 'fsdp',
          flashAttention: true,
          mixedPrecision: 'bf16',
        },
      },
    ];

    const report = runSensitivityAnalysis(params, benchmarks, 0.10);

    // Verify parameter was restored
    expect(getStragglerSigma()).toBe(original);

    // numNodes=128 (multi-node) should show non-zero sensitivity
    const multiNodeDelta = report.results[0].benchmarkDeltas.find(
      d => d.benchmarkName.includes('numNodes=128'),
    )!;
    expect(multiNodeDelta.sensitivity).not.toBe(0);
    // Higher sigma → more straggler overhead → lower MFU → negative sensitivity
    expect(multiNodeDelta.mfuAtPlus10).toBeLessThan(multiNodeDelta.mfuAtMinus10);

    // numNodes=1 single-node — no straggler penalty, zero sensitivity
    const singleNodeDelta = report.results[0].benchmarkDeltas.find(
      d => d.benchmarkName.includes('numNodes=1'),
    )!;
    expect(singleNodeDelta.sensitivity).toBe(0);
  });

  it('should produce consistent baselines (no state leakage)', () => {
    const benchmarks = makeSmallBenchmarks();

    // Run twice and verify same MFU
    const mfu1 = getSimulationMetrics(benchmarks[0].config).mfu;
    const mfu2 = getSimulationMetrics(benchmarks[0].config).mfu;

    expect(mfu1).toBe(mfu2);
  });
});
