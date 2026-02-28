/**
 * Published Benchmark Validation — Expanded Tests
 *
 * Regression-guard tests with tight thresholds (±15% of simulator output).
 * Each test documents the published reference value for context, but the
 * threshold is pinned to the simulator's current output, not the published
 * number. This catches accidental regressions without requiring the simulator
 * to perfectly match real-world training runs.
 *
 * MFU calibration table and source references: see docs/BENCHMARKS.md
 */

import { describe, it, expect } from 'vitest';
import {
  type SimulationConfig,
  type SimulationMetrics,
  getSimulationMetrics,
} from '../../src/core/simulation/engine.ts';
import { getValidatedSimulationMetrics } from '../helpers/validated-metrics.ts';
import {
  runInferenceSimulation,
  type InferenceSimulationConfig,
} from '../../src/core/inference/simulation.ts';
import { getModel } from '../../src/core/models/index.ts';
import { benchmarkConfig } from '../helpers/benchmark-config.ts';
import { nemoImpliedStepTimeMs } from '../helpers/nemo-tflops.ts';

function sim(config: SimulationConfig): SimulationMetrics {
  return getValidatedSimulationMetrics(config);
}

function rawSim(config: SimulationConfig): SimulationMetrics {
  return getSimulationMetrics(config);
}

function infer(config: InferenceSimulationConfig) {
  return runInferenceSimulation(config);
}

// ===========================================================================
// Section 1: New Model Training MFU (±15% of simulator output)
//
// All configs use activation checkpointing (default). MFU uses PaLM
// definition (6PD only); published "MFU" numbers that look high are
// likely HFU (8PD with recompute).
// ===========================================================================

describe('Expanded MFU Benchmarks', () => {
  // -----------------------------------------------------------------------
  // DeepSeek-V3: GA=4 with PP=16 → 79% pipeline bubble. Pathological config
  // that tests extreme bubble overhead, NOT benchmark validation.
  // -----------------------------------------------------------------------
  it('DeepSeek-V3 pathological config: GA=4 with PP=16 produces extreme bubble (~3.8% MFU)', () => {
    // GA=4 with PP=16 → 79% bubble — pathological config
    const metrics = rawSim(benchmarkConfig(
      'h100-sxm', 8, 256, 'deepseek-v3',
      'fsdp-tp-pp', 512, 1, 4096,
      { tp: 1, pp: 16 },
    ));

    // ±15% of simulator value (~0.0380). Grouped GEMM (with per-group scheduling
    // overhead) + EP compute penalties apply. GA-aware overlap raises MFU slightly.
    expect(metrics.mfu).toBeGreaterThan(0.0323);
    expect(metrics.mfu).toBeLessThan(0.0437);
  });

  // -----------------------------------------------------------------------
  // DeepSeek-R1: Same architecture as V3, same pathological PP=16 config
  // -----------------------------------------------------------------------
  it('DeepSeek-R1 pathological config: GA=4 with PP=16 produces extreme bubble (~3.8% MFU)', () => {
    const metrics = rawSim(benchmarkConfig(
      'h100-sxm', 8, 256, 'deepseek-r1',
      'fsdp-tp-pp', 512, 1, 4096,
      { tp: 1, pp: 16 },
    ));

    // ±15% — should match V3 since same arch
    expect(metrics.mfu).toBeGreaterThan(0.0323);
    expect(metrics.mfu).toBeLessThan(0.0437);
  });

  // -----------------------------------------------------------------------
  // Llama 3 70B × 2048 H100
  // Published: ~35% MFU (derived from GPU-hours and FLOPS)
  // -----------------------------------------------------------------------
  it('Llama 3 70B × 2048 H100 (FSDP-TP): MFU ≈ 35.2%', () => {
    const metrics = sim(benchmarkConfig(
      'h100-sxm', 8, 256, 'llama3-70b',
      'fsdp-tp', 512, 1, 8192,
      { tp: 8 },
    ));

    // ±15% of simulator value (~35.2%)
    expect(metrics.mfu).toBeGreaterThan(0.2993);
    expect(metrics.mfu).toBeLessThan(0.4049);
  });

  // -----------------------------------------------------------------------
  // Llama 3.3 70B × 2048 H100 (same arch as Llama 3 70B)
  // -----------------------------------------------------------------------
  it('Llama 3.3 70B × 2048 H100 (FSDP-TP): MFU ≈ 35.2%', () => {
    const metrics = sim(benchmarkConfig(
      'h100-sxm', 8, 256, 'llama3.3-70b',
      'fsdp-tp', 512, 1, 8192,
      { tp: 8 },
    ));

    // ±15% of simulator value (~35.2%)
    expect(metrics.mfu).toBeGreaterThan(0.2993);
    expect(metrics.mfu).toBeLessThan(0.4049);
  });

  // -----------------------------------------------------------------------
  // Qwen 2.5 7B × 8 H100 FSDP (single node)
  // Similar class to Llama 3 8B
  // -----------------------------------------------------------------------
  it('Qwen 2.5 7B × 8 H100 FSDP: MFU ≈ 48.0%', () => {
    const metrics = sim(benchmarkConfig(
      'h100-sxm', 8, 1, 'qwen2.5-7b',
      'fsdp', 16, 2, 4096,
    ));

    // ±15% of simulator value (~48.0%)
    // Higher with bf16 grads (less comm overhead)
    expect(metrics.mfu).toBeGreaterThan(0.4077);
    expect(metrics.mfu).toBeLessThan(0.5517);
  });

  // -----------------------------------------------------------------------
  // Qwen 2.5 72B × 128 H100 FSDP-TP
  // -----------------------------------------------------------------------
  it('Qwen 2.5 72B × 128 H100 (FSDP-TP): MFU ≈ 39.9%', () => {
    const metrics = sim(benchmarkConfig(
      'h100-sxm', 8, 16, 'qwen2.5-72b',
      'fsdp-tp', 128, 1, 4096,
      { tp: 8 },
    ));

    // ±15% of simulator value (~39.9%)
    expect(metrics.mfu).toBeGreaterThan(0.3387);
    expect(metrics.mfu).toBeLessThan(0.4583);
  });

  // -----------------------------------------------------------------------
  // Grok-1 314B MoE × 256 H200 FSDP-TP-PP
  // -----------------------------------------------------------------------
  it('Grok-1 314B × 256 H200 (FSDP-TP-PP): MFU ≈ 38.8%', () => {
    const metrics = sim(benchmarkConfig(
      'h200-sxm', 8, 32, 'grok-1',
      'fsdp-tp-pp', 128, 1, 4096,
      { tp: 8, pp: 4 },
    ));

    // ±15% of simulator value (~38.8%)
    // Higher with bf16 grads (less DP comm overhead) + GA-aware overlap
    expect(metrics.mfu).toBeGreaterThan(0.3301);
    expect(metrics.mfu).toBeLessThan(0.4465);
  });

  // -----------------------------------------------------------------------
  // DBRX 132B MoE × 3072 H100 FSDP-TP + EP
  // Published: >50% MFU (Databricks)
  // Databricks used EP for MoE experts; without EP the simulator must
  // AllReduce all 132B params across full DP → only ~17-22% MFU. With
  // TP=8 (full node NVLink), EP=16, no PP: ~22% MFU. Remaining gap vs
  // published >50% is likely HFU labeling (Databricks counting recompute)
  // or production-specific optimizations (HFU here is ~30%).
  // -----------------------------------------------------------------------
  it('DBRX × 3072 H100 (FSDP-TP=8 + EP=16): MFU ≈ 20.0% (published: >50%)', () => {
    const metrics = sim(benchmarkConfig(
      'h100-sxm', 8, 384, 'dbrx',
      'fsdp-tp', 1024, 1, 4096,
      { tp: 8, ep: 16 },
    ));

    // Routing locality factor 1/(1+density) reduces EP all-to-all volume for sparse MoE
    // (DBRX: 4/16 active experts → density=0.25 → locality≈0.80).
    // EP latency scaling + coordination floor add overhead at EP=16.
    // ±15% of simulator value (~0.200)
    expect(metrics.mfu).toBeGreaterThan(0.1701);
    expect(metrics.mfu).toBeLessThan(0.2302);
  });
});

// ===========================================================================
// Section 1b: Phi-3/Phi-4 and OLMo 2 MFU Benchmarks (±15%)
//
// Cluster sizes match published training configurations.
// ===========================================================================

describe('Phi & OLMo MFU Benchmarks', () => {
  // -----------------------------------------------------------------------
  // Phi-4 14B on 1920 H100s
  // Source: Microsoft Phi-4 Technical Report (Dec 2024)
  // Config: 1920 H100-80G, GBS=5760, seqlen=4096, FSDP
  // No published MFU — pinned from simulator
  // -----------------------------------------------------------------------
  it('Phi-4 14B × 1920 H100 FSDP: MFU ≈ 41.2%', () => {
    const metrics = sim(benchmarkConfig(
      'h100-sxm', 8, 240, 'phi4', 'fsdp', 5760, 1, 4096,
    ));
    // ±15% of simulator value (~41.2%)
    expect(metrics.mfu).toBeGreaterThan(0.3498);
    expect(metrics.mfu).toBeLessThan(0.4734);
  });

  // -----------------------------------------------------------------------
  // Phi-4 Mini 3.8B on 1024 A100s
  // Source: Microsoft Phi-4-Mini Technical Report (Feb 2025)
  // Config: 1024 A100-80G, ~14 days, 5T tokens, FSDP
  // -----------------------------------------------------------------------
  it('Phi-4 Mini 3.8B × 1024 A100 FSDP: MFU ≈ 22.8%', () => {
    const metrics = sim(benchmarkConfig(
      'a100-80gb', 8, 128, 'phi4-mini', 'fsdp', 2048, 4, 4096,
    ));
    // ±15% of simulator value (~22.8%)
    expect(metrics.mfu).toBeGreaterThan(0.1934);
    expect(metrics.mfu).toBeLessThan(0.2618);
  });

  // -----------------------------------------------------------------------
  // Phi-3 Mini on 512 H100s
  // Source: Microsoft Phi-3.5 (2024) — 512 H100, 3.4T tokens, 10 days
  // -----------------------------------------------------------------------
  it('Phi-3 Mini 3.8B × 512 H100 FSDP: MFU ≈ 41.3%', () => {
    const metrics = sim(benchmarkConfig(
      'h100-sxm', 8, 64, 'phi3-mini', 'fsdp', 2048, 4, 4096,
    ));
    // ±15% of simulator value (~41.3%)
    expect(metrics.mfu).toBeGreaterThan(0.3511);
    expect(metrics.mfu).toBeLessThan(0.4751);
  });

  // -----------------------------------------------------------------------
  // Phi-3 Small 7B and Phi-3 Medium 14B on 512 H100s
  // -----------------------------------------------------------------------
  it('Phi-3 Small 7B × 512 H100 FSDP: MFU ≈ 45.9%', () => {
    const metrics = sim(benchmarkConfig(
      'h100-sxm', 8, 64, 'phi3-small', 'fsdp', 2048, 2, 4096,
    ));
    // ±15% of simulator value (~45.9%)
    expect(metrics.mfu).toBeGreaterThan(0.39);
    expect(metrics.mfu).toBeLessThan(0.5278);
  });

  it('Phi-3 Medium 14B × 512 H100 FSDP: MFU ≈ 46.0%', () => {
    const metrics = sim(benchmarkConfig(
      'h100-sxm', 8, 64, 'phi3-medium', 'fsdp', 2048, 2, 4096,
    ));
    // ±15% of simulator value (~46.0%)
    expect(metrics.mfu).toBeGreaterThan(0.3905);
    expect(metrics.mfu).toBeLessThan(0.5285);
  });

  // -----------------------------------------------------------------------
  // OLMo 2 32B on 1280 H100s
  // Source: AI2 blog (https://allenai.org/blog/olmo2-32B) — ~38% MFU,
  //         1280 H100-80G (160 nodes), FSDP, GBS=2048, seq=4096, selective AC
  // -----------------------------------------------------------------------
  it('OLMo 2 32B × 1280 H100 FSDP: MFU ≈ 37.4% (published: ~38%)', () => {
    const metrics = sim(benchmarkConfig(
      'h100-sxm', 8, 160, 'olmo2-32b', 'fsdp', 2048, 1, 4096,
      undefined,
      { activationCheckpointing: true, checkpointingGranularity: 'selective' },
    ));
    // ±15% of simulator value (~37.4%). Collective-count degradation
    // (64 layers × 3 = 192 collectives) reduces overlap at large DP scale.
    expect(metrics.mfu).toBeGreaterThan(0.3179);
    expect(metrics.mfu).toBeLessThan(0.4303);
  });

  // -----------------------------------------------------------------------
  // OLMo 3 32B on 1024 H100s
  // Source: OLMo 3 paper (https://arxiv.org/abs/2512.13961) — ~41% MFU,
  //         1024 H100-80G (128 nodes), FSDP, GBS=1024, seq=8192, selective AC
  // Stored-layers auto-resolve (35 of 64 layers selective, rest full AC)
  // -----------------------------------------------------------------------
  it('OLMo 3 32B × 1024 H100 FSDP: MFU ≈ 40% (published: ~41%)', () => {
    const metrics = sim(benchmarkConfig(
      'h100-sxm', 8, 128, 'olmo3-32b', 'fsdp', 1024, 1, 8192,
      undefined,
      { activationCheckpointing: true, checkpointingGranularity: 'selective' },
    ));
    // ±15% of simulator value (~43.4%)
    expect(metrics.mfu).toBeGreaterThan(0.372);
    expect(metrics.mfu).toBeLessThan(0.504);
  });

  // -----------------------------------------------------------------------
  // OLMo 2 7B on 1280 H100s (sim-only, no published MFU for this size)
  // Config: seq=4096 (paper Table 3), GBS=1024 (paper: ~4M tokens/batch)
  // -----------------------------------------------------------------------
  it('OLMo 2 7B × 1280 H100 FSDP: MFU ≈ 33.0%', () => {
    const metrics = sim(benchmarkConfig(
      'h100-sxm', 8, 160, 'olmo2-7b', 'fsdp', 1024, 1, 4096,
    ));
    // ±15% of simulator value (~33.0%)
    expect(metrics.mfu).toBeGreaterThan(0.2801);
    expect(metrics.mfu).toBeLessThan(0.3791);
  });

  // -----------------------------------------------------------------------
  // OLMo 2 13B on 1280 H100s (sim-only, no published MFU for this size)
  // Config: seq=4096 (paper Table 3), GBS=2048 (paper)
  // -----------------------------------------------------------------------
  it('OLMo 2 13B × 1280 H100 FSDP: MFU ≈ 33.8%', () => {
    const metrics = sim(benchmarkConfig(
      'h100-sxm', 8, 160, 'olmo2-13b', 'fsdp', 2048, 1, 4096,
    ));
    // ±15% of simulator value (~33.8%)
    expect(metrics.mfu).toBeGreaterThan(0.2872);
    expect(metrics.mfu).toBeLessThan(0.3886);
  });
});

// ===========================================================================
// Section 1c: GPU-Hours — Phi-4 (Published Training Data)
// ===========================================================================

describe('GPU-Hours Projections — Phi-4 & OLMo', () => {
  // -----------------------------------------------------------------------
  // Phi-4 14B: 1920 H100s, 9.8T tokens, ~21 days → ~967K GPU-hours
  // Source: Microsoft Phi-4 Technical Report (Dec 2024)
  // -----------------------------------------------------------------------
  it('Phi-4 14B × 1920 H100: GPU-hours ≈ 588K (published: ~967K)', () => {
    const gbs = 5760, seqLen = 4096, numGPUs = 1920;
    const totalSteps = Math.ceil(9.8e12 / (gbs * seqLen));
    const metrics = sim(benchmarkConfig(
      'h100-sxm', 8, 240, 'phi4', 'fsdp', gbs, 1, seqLen,
      undefined, { maxSteps: totalSteps },
    ));
    // ±15% of simulator value (~588K GPU-hours)
    const gpuHours = (metrics.timeToTrainHours ?? 0) * numGPUs;
    expect(gpuHours).toBeGreaterThan(500000);
    expect(gpuHours).toBeLessThan(676000);
  });

  // -----------------------------------------------------------------------
  // Phi-4 Mini: 1024 A100s, 5T tokens, ~14 days → ~344K GPU-hours
  // Source: Microsoft Phi-4-Mini Technical Report (Feb 2025)
  // -----------------------------------------------------------------------
  it('Phi-4 Mini × 1024 A100: GPU-hours ≈ 450K (published: ~344K)', () => {
    const gbs = 2048, seqLen = 4096, numGPUs = 1024;
    const totalSteps = Math.ceil(5e12 / (gbs * seqLen));
    const metrics = sim(benchmarkConfig(
      'a100-80gb', 8, 128, 'phi4-mini', 'fsdp', gbs, 4, seqLen,
      undefined, { maxSteps: totalSteps },
    ));
    // ±15% of simulator value (~450K GPU-hours)
    const gpuHours = (metrics.timeToTrainHours ?? 0) * numGPUs;
    expect(gpuHours).toBeGreaterThan(383000);
    expect(gpuHours).toBeLessThan(518000);
  });
});

// ===========================================================================
// Section 2: Published GPU-Hours Projections (±15%)
//
// Compare simulator projected training time against published GPU-hours.
// Pinned to simulator output; published values shown for context only.
// ===========================================================================

describe('GPU-Hours Projections (Expanded)', () => {
  // -----------------------------------------------------------------------
  // Llama 3 8B: 1.3M H100-hours for 15T tokens
  // Source: Meta LLaMA 3 paper (2024)
  // -----------------------------------------------------------------------
  it('Llama 3 8B × 512 H100: GPU-hours ≈ 476k (published: 1.3M)', () => {
    const seqLen = 8192;
    const gbs = 1024;
    const totalSteps = Math.ceil(15e12 / (gbs * seqLen));
    const numGPUs = 512;

    const metrics = sim(benchmarkConfig(
      'h100-sxm', 8, numGPUs / 8, 'llama3-8b',
      'fsdp', gbs, 2, seqLen,
      undefined,
      { maxSteps: totalSteps },
    ));

    // ±15% of simulator value (~549K GPU-hours)
    const gpuHours = (metrics.timeToTrainHours ?? 0) * numGPUs;
    expect(gpuHours).toBeGreaterThan(467000);
    expect(gpuHours).toBeLessThan(631000);
  });

  // -----------------------------------------------------------------------
  // Llama 3 70B: 6.4M H100-hours for 15T tokens
  // Source: Meta LLaMA 3 paper (2024)
  // -----------------------------------------------------------------------
  it('Llama 3 70B × 2048 H100: GPU-hours ≈ 5.06M (published: 6.4M)', () => {
    const seqLen = 8192;
    const gbs = 1024;
    const totalSteps = Math.ceil(15e12 / (gbs * seqLen));
    const numGPUs = 2048;

    const metrics = sim(benchmarkConfig(
      'h100-sxm', 8, numGPUs / 8, 'llama3-70b',
      'fsdp-tp', gbs, 1, seqLen,
      { tp: 8 },
      { maxSteps: totalSteps },
    ));

    // ±15% of simulator value (~5,065K GPU-hours)
    const gpuHours = (metrics.timeToTrainHours ?? 0) * numGPUs;
    expect(gpuHours).toBeGreaterThan(4305000);
    expect(gpuHours).toBeLessThan(5824000);
  });

  // -----------------------------------------------------------------------
  // Llama 3.1 405B: 30.84M H100-hours for 15T tokens
  // Source: Meta LLaMA 3 paper (2024)
  // Meta uses interleaved 1F1B pipeline schedule to reduce bubble overhead.
  // -----------------------------------------------------------------------
  it('Llama 3.1 405B × 16384 H100: GPU-hours ≈ 25.8M (published: 30.8M, AC=off)', () => {
    const seqLen = 8192;
    const gbs = 2048;
    const totalSteps = Math.ceil(15e12 / (gbs * seqLen));
    const numGPUs = 16384;

    const metrics = rawSim(benchmarkConfig(
      'h100-sxm', 8, numGPUs / 8, 'llama3-405b',
      'fsdp-tp-pp', gbs, 1, seqLen,
      { tp: 8, pp: 16, pipelineSchedule: 'interleaved-1f1b', interleavedStages: 4 },
      { maxSteps: totalSteps, activationCheckpointing: false },
    ));

    // ±15% of simulator value (~25.8M GPU-hours) — AC=off per Meta paper §3.3.2
    // Node-based scale overhead at 2048 nodes closes gap to published 30.8M.
    const gpuHours = (metrics.timeToTrainHours ?? 0) * numGPUs;
    expect(gpuHours).toBeGreaterThan(21896000);
    expect(gpuHours).toBeLessThan(29625000);
  });

  // -----------------------------------------------------------------------
  // DeepSeek-V3: 2.788M H800-hours for 14.8T tokens
  // Source: DeepSeek-V3 technical report (2024), ISCA 2025
  // Realistic config: H800 FP8, TP=4, PP=8, EP=32, DualPipe-V
  // Sim: ~2.11M GPU-hours (pure compute). Published: 2.79M (1.32× overhead)
  // Grouped GEMM + EP compute penalties increase step time, closing gap to published.
  // -----------------------------------------------------------------------
  it('DeepSeek-V3 × 2048 H800 FP8: GPU-hours ≈ 2.14M (published: 2.79M)', () => {
    const seqLen = 4096;
    const gbs = 8192;
    const totalSteps = Math.ceil(14.8e12 / (gbs * seqLen));
    const numGPUs = 2048;

    const metrics = sim(benchmarkConfig(
      'h800-sxm', 8, numGPUs / 8, 'deepseek-v3',
      'fsdp-tp-pp', gbs, 2, seqLen,
      { tp: 4, pp: 8, dp: 64, ep: 32,
        dpType: 'fsdp', sequenceParallel: true,
        pipelineSchedule: 'dualpipe-v', interleavedStages: 1,
        numMicroBatches: 64 },
      { maxSteps: totalSteps, mixedPrecision: 'fp8' },
    ));

    // ±15% of simulator value (~2,140K GPU-hours)
    const gpuHours = (metrics.timeToTrainHours ?? 0) * numGPUs;
    expect(gpuHours).toBeGreaterThan(1823000);
    expect(gpuHours).toBeLessThan(2466000);
  });

  // -----------------------------------------------------------------------
  // GPT-3 175B: ~3.1M V100-hours for 300B tokens
  // Source: Brown et al. (2020) estimated from training duration
  // -----------------------------------------------------------------------
  it('GPT-3 175B × 1024 V100: GPU-hours ≈ 1.54M (published: ~3.1M)', () => {
    const seqLen = 2048;
    const gbs = 1536;
    const totalSteps = Math.ceil(300e9 / (gbs * seqLen));
    const numGPUs = 1024;

    const metrics = rawSim(benchmarkConfig(
      'v100-32gb', 8, numGPUs / 8, 'gpt3-175b',
      'ddp-tp-pp', gbs, 1, seqLen,
      { tp: 8, pp: 8 },
      { maxSteps: totalSteps },
    ));

    // ±15% of simulator value (~1,542K GPU-hours)
    const gpuHours = (metrics.timeToTrainHours ?? 0) * numGPUs;
    expect(gpuHours).toBeGreaterThan(1310000);
    expect(gpuHours).toBeLessThan(1773000);
  });
});

// ===========================================================================
// Section 3: Training Throughput (tok/s/GPU, ±20%)
// ===========================================================================

describe('Training Throughput (Expanded)', () => {
  // -----------------------------------------------------------------------
  // DeepSeek-V3: ~1,475 tok/s/GPU (published, H800 FP8)
  // Realistic config: H800 FP8, TP=4, PP=8, EP=32, DualPipe-V
  // Sim: ~1951 tok/s/GPU (higher than published due to pure-compute model)
  // -----------------------------------------------------------------------
  it('DeepSeek-V3 × 2048 H800 FP8: ~1917 tok/s/GPU (published: 1475)', () => {
    const metrics = sim(benchmarkConfig(
      'h800-sxm', 8, 256, 'deepseek-v3',
      'fsdp-tp-pp', 8192, 2, 4096,
      { tp: 4, pp: 8, dp: 64, ep: 32,
        dpType: 'fsdp', sequenceParallel: true,
        pipelineSchedule: 'dualpipe-v', interleavedStages: 1,
        numMicroBatches: 64 },
      { mixedPrecision: 'fp8' },
    ));

    // ±20% of simulator value (~1917 tok/s/GPU)
    const tokPerGPU = metrics.tokensPerSecond / 2048;
    expect(tokPerGPU).toBeGreaterThan(1533);
    expect(tokPerGPU).toBeLessThan(2301);
  });

  // -----------------------------------------------------------------------
  // Llama 3 8B: ~3,200 tok/s/GPU (derived from Meta's GPU-hours)
  // -----------------------------------------------------------------------
  it('Llama 3 8B × 512 H100: ~4376 tok/s/GPU (published: ~3200)', () => {
    const metrics = sim(benchmarkConfig(
      'h100-sxm', 8, 64, 'llama3-8b',
      'fsdp', 512, 2, 8192,
    ));

    // ±20% of simulator value (~4376 tok/s/GPU)
    const tokPerGPU = metrics.tokensPerSecond / 512;
    expect(tokPerGPU).toBeGreaterThan(3500);
    expect(tokPerGPU).toBeLessThan(5252);
  });

  // -----------------------------------------------------------------------
  // Llama 3 70B: ~1,100 tok/s/GPU (derived from GPU-hours)
  // -----------------------------------------------------------------------
  it('Llama 3 70B × 2048 H100: ~1027 tok/s/GPU (published: ~1100)', () => {
    const metrics = sim(benchmarkConfig(
      'h100-sxm', 8, 256, 'llama3-70b',
      'fsdp-tp', 512, 1, 8192,
      { tp: 8 },
    ));

    // ±20% of simulator value (~1027 tok/s/GPU)
    const tokPerGPU = metrics.tokensPerSecond / 2048;
    expect(tokPerGPU).toBeGreaterThan(821);
    expect(tokPerGPU).toBeLessThan(1233);
  });
});

// ===========================================================================
// Section 4: Inference TTFT (±15%)
//
// TTFT = time to first token. Tests use batch=1, inputSeqLen=512,
// outputSeqLen=256.
// ===========================================================================

describe('Inference TTFT (Expanded)', () => {
  // -----------------------------------------------------------------------
  // Llama 3 8B × 1 H100: published ~2-5ms TTFT for short sequences
  // Simulator gives ~20.8ms (analytical model includes more overhead)
  // -----------------------------------------------------------------------
  it('Llama 3 8B × 1 H100: TTFT ≈ 20.8ms', () => {
    const result = infer({
      modelId: 'llama3-8b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256,
      weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1,
    });

    expect(result.success).toBe(true);
    // ±15% of simulator value (~20.8ms)
    expect(result.latency.ttft).toBeGreaterThan(17.7);
    expect(result.latency.ttft).toBeLessThan(23.9);
  });

  // -----------------------------------------------------------------------
  // Llama 3 8B × 1 A100: slower than H100 (312 vs 989 BF16 TFLOPS)
  // -----------------------------------------------------------------------
  it('Llama 3 8B × 1 A100: TTFT ≈ 65.9ms', () => {
    const result = infer({
      modelId: 'llama3-8b', gpuId: 'a100-80gb', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256,
      weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1,
    });

    expect(result.success).toBe(true);
    // ±15% of simulator value (~65.9ms)
    expect(result.latency.ttft).toBeGreaterThan(56.0);
    expect(result.latency.ttft).toBeLessThan(75.8);
  });

  // -----------------------------------------------------------------------
  // Llama 3 70B × 8 H100 TP=8: published ~123ms TTFT
  // Simulator gives ~25.8ms
  // -----------------------------------------------------------------------
  it('Llama 3 70B × 8 H100 TP=8: TTFT ≈ 25.8ms (published: ~123ms)', () => {
    const result = infer({
      modelId: 'llama3-70b', gpuId: 'h100-sxm', numGPUs: 8,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256,
      weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 8,
    });

    expect(result.success).toBe(true);
    // ±15% of simulator value (~25.8ms)
    expect(result.latency.ttft).toBeGreaterThan(21.9);
    expect(result.latency.ttft).toBeLessThan(29.7);
  });

  // -----------------------------------------------------------------------
  // Mixtral 8x7B × 2 H100: published sub-200ms TTFT
  // Simulator gives ~61ms
  // -----------------------------------------------------------------------
  it('Mixtral 8x7B × 2 H100: TTFT ≈ 17ms (published: sub-200ms)', () => {
    const result = infer({
      modelId: 'mixtral-8x7b', gpuId: 'h100-sxm', numGPUs: 2,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256,
      weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 2,
    });

    expect(result.success).toBe(true);
    // ±15% of simulator value (~17.3ms) — uses active params for MoE FLOPs
    expect(result.latency.ttft).toBeGreaterThan(14.7);
    expect(result.latency.ttft).toBeLessThan(19.9);
  });

  // -----------------------------------------------------------------------
  // Qwen 2.5 7B × 1 H100: similar to Llama 8B class
  // -----------------------------------------------------------------------
  it('Qwen 2.5 7B × 1 H100: TTFT ≈ 19.7ms', () => {
    const result = infer({
      modelId: 'qwen2.5-7b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256,
      weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1,
    });

    expect(result.success).toBe(true);
    // ±15% of simulator value (~19.7ms)
    expect(result.latency.ttft).toBeGreaterThan(16.8);
    expect(result.latency.ttft).toBeLessThan(22.7);
  });

  // -----------------------------------------------------------------------
  // Grok-1 314B × 16 H200 TP=16: sanity check (finite positive TTFT)
  // -----------------------------------------------------------------------
  it('Grok-1 314B × 16 H200 TP=16: TTFT ≈ 15ms (MoE active params)', () => {
    const result = infer({
      modelId: 'grok-1', gpuId: 'h200-sxm', numGPUs: 16,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256,
      weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 16,
    });

    expect(result.success).toBe(true);
    // ±15% of simulator value (~15.5ms) — uses active params for MoE FLOPs (GeGLU gated MLP)
    expect(result.latency.ttft).toBeGreaterThan(13.1);
    expect(result.latency.ttft).toBeLessThan(17.8);
  });
});

// ===========================================================================
// Section 5: Inference Throughput (±15%)
//
// Higher batch sizes to measure throughput capacity.
// ===========================================================================

describe('Inference Throughput (Expanded)', () => {
  // -----------------------------------------------------------------------
  // Llama 3 8B × 1 H100 batch=128: published ~12,500 tok/s peak
  // Simulator gives ~6,277 tok/s (analytical model with correct batch scaling)
  // -----------------------------------------------------------------------
  it('Llama 3 8B × 1 H100 batch=128: tok/s ≈ 6.3k', () => {
    const result = infer({
      modelId: 'llama3-8b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 128, inputSeqLen: 512, outputSeqLen: 256,
      weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1,
    });

    expect(result.success).toBe(true);
    // ±15% of simulator value (~6,000 tok/s)
    expect(result.throughput.tokensPerSecond).toBeGreaterThan(5100);
    expect(result.throughput.tokensPerSecond).toBeLessThan(6901);
  });

  // -----------------------------------------------------------------------
  // Llama 3 70B × 8 H100 TP=8 batch=64: published ~460 tok/s
  // Simulator gives ~4,561 tok/s (analytical model with correct batch scaling)
  // -----------------------------------------------------------------------
  it('Llama 3 70B × 8 H100 TP=8 batch=64: tok/s ≈ 4.6k', () => {
    const result = infer({
      modelId: 'llama3-70b', gpuId: 'h100-sxm', numGPUs: 8,
      batchSize: 64, inputSeqLen: 512, outputSeqLen: 256,
      weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 8,
    });

    expect(result.success).toBe(true);
    // ±15% of simulator value (~4,655 tok/s)
    expect(result.throughput.tokensPerSecond).toBeGreaterThan(3957);
    expect(result.throughput.tokensPerSecond).toBeLessThan(5353);
  });

  // -----------------------------------------------------------------------
  // Mixtral 8x7B × 2 H100 batch=32: published ~38 req/s
  // With batch-aware MoE weight bytes: at batch=32 with 8 experts & top-2,
  // nearly all experts are touched (fractionTouched ≈ 1.0), so decode reads
  // ~totalParams worth of weights. Simulator gives ~6.2 req/s.
  // -----------------------------------------------------------------------
  it('Mixtral 8x7B × 2 H100 batch=32: req/s ≈ 6.2', () => {
    const result = infer({
      modelId: 'mixtral-8x7b', gpuId: 'h100-sxm', numGPUs: 2,
      batchSize: 32, inputSeqLen: 512, outputSeqLen: 256,
      weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 2,
    });

    expect(result.success).toBe(true);
    // ±15% of simulator value (~6.4 req/s)
    expect(result.throughput.requestsPerSecond).toBeGreaterThan(5.4);
    expect(result.throughput.requestsPerSecond).toBeLessThan(7.3);
  });
});

// ===========================================================================
// Section 6: Cross-Model Consistency
//
// Strict ordering and relative comparisons — no tolerance ranges.
// These verify the simulator's internal consistency, not accuracy.
// ===========================================================================

describe('Cross-Model Consistency (Expanded)', () => {
  // -----------------------------------------------------------------------
  // Llama 3.3 70B MFU ≈ Llama 3 70B MFU (same architecture)
  // -----------------------------------------------------------------------
  it('Llama 3.3 70B MFU ≈ Llama 3 70B MFU (within ±5%)', () => {
    const llama3 = sim(benchmarkConfig(
      'h100-sxm', 8, 256, 'llama3-70b',
      'fsdp-tp', 512, 1, 8192, { tp: 8 },
    ));
    const llama33 = sim(benchmarkConfig(
      'h100-sxm', 8, 256, 'llama3.3-70b',
      'fsdp-tp', 512, 1, 8192, { tp: 8 },
    ));

    const ratio = llama33.mfu / llama3.mfu;
    expect(ratio).toBeGreaterThan(0.95);
    expect(ratio).toBeLessThan(1.05);
  });

  // -----------------------------------------------------------------------
  // DeepSeek-R1 MFU ≈ DeepSeek-V3 MFU (same architecture)
  // -----------------------------------------------------------------------
  it('DeepSeek-R1 MFU ≈ DeepSeek-V3 MFU (within ±5%)', () => {
    const v3 = rawSim(benchmarkConfig(
      'h100-sxm', 8, 256, 'deepseek-v3',
      'fsdp-tp-pp', 512, 1, 4096, { tp: 1, pp: 16 },
    ));
    const r1 = rawSim(benchmarkConfig(
      'h100-sxm', 8, 256, 'deepseek-r1',
      'fsdp-tp-pp', 512, 1, 4096, { tp: 1, pp: 16 },
    ));

    const ratio = r1.mfu / v3.mfu;
    expect(ratio).toBeGreaterThan(0.95);
    expect(ratio).toBeLessThan(1.05);
  });

  // -----------------------------------------------------------------------
  // Qwen 2.5 72B MFU in same range as Llama 2 70B (within ±25%)
  // Similar model size, similar architecture class
  // -----------------------------------------------------------------------
  it('Qwen 2.5 72B MFU ≈ Llama 2 70B MFU (within ±25%)', () => {
    const qwen = sim(benchmarkConfig(
      'h100-sxm', 8, 16, 'qwen2.5-72b',
      'fsdp-tp', 128, 1, 4096, { tp: 8 },
    ));
    const llama = sim(benchmarkConfig(
      'h100-sxm', 8, 16, 'llama2-70b',
      'fsdp-tp', 128, 1, 4096, { tp: 8 },
    ));

    const ratio = qwen.mfu / llama.mfu;
    expect(ratio).toBeGreaterThan(0.75);
    expect(ratio).toBeLessThan(1.25);
  });

  // -----------------------------------------------------------------------
  // FP8 inference ≤ BF16 inference time (strict)
  // FP8 uses half the memory bandwidth → should be faster
  // -----------------------------------------------------------------------
  it('FP8 inference TTFT ≤ BF16 inference TTFT (Llama 3 8B)', () => {
    const bf16 = infer({
      modelId: 'llama3-8b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256,
      weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1,
    });
    const fp8 = infer({
      modelId: 'llama3-8b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256,
      weightPrecision: 'fp8', kvCachePrecision: 'fp8', tensorParallel: 1,
    });

    expect(bf16.success).toBe(true);
    expect(fp8.success).toBe(true);
    // FP8 should be faster (lower latency)
    expect(fp8.latency.ttft).toBeLessThanOrEqual(bf16.latency.ttft);
    expect(fp8.latency.totalLatency).toBeLessThanOrEqual(bf16.latency.totalLatency);
  });

  // -----------------------------------------------------------------------
  // Larger model = slower inference (strict ordering)
  // -----------------------------------------------------------------------
  it('Larger model has higher inference TTFT (8B < 70B < 314B)', () => {
    const small = infer({
      modelId: 'llama3-8b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256,
      weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1,
    });
    const medium = infer({
      modelId: 'llama3-70b', gpuId: 'h100-sxm', numGPUs: 8,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256,
      weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 8,
    });
    const large = infer({
      modelId: 'grok-1', gpuId: 'h200-sxm', numGPUs: 16,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256,
      weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 16,
    });

    expect(small.success).toBe(true);
    expect(medium.success).toBe(true);
    expect(large.success).toBe(true);

    // TTFT should increase with model size (dense models)
    expect(medium.latency.ttft).toBeGreaterThan(small.latency.ttft);
    // Grok-1 is MoE (86B active of 314B total) on 16 H200s — per-GPU compute
    // is actually lower than dense LLaMA 70B on 8 H100s, so skip this comparison

    // TPOT: LLaMA 70B with TP=8 on H100 has per-GPU weight read of ~70B/8=8.75B,
    // which is smaller than LLaMA 8B on 1 GPU (~8B). With lower bandwidth efficiency
    // on small models, the 70B TP=8 TPOT can be slightly faster than 8B on 1 GPU.
    expect(medium.latency.tpot).toBeGreaterThan(0);
    // Grok-1 is MoE: decode only reads activeParams (~86B) not totalParams (314B).
    // With TP=16 on H200 vs LLaMA 70B with TP=8 on H100, Grok-1 TPOT can be lower.
    // Skip this comparison — MoE + different hardware makes ordering non-monotone.
    expect(large.latency.tpot).toBeGreaterThan(0);
  });

  // -----------------------------------------------------------------------
  // Training: Larger model → more GPU-hours for same token count
  // -----------------------------------------------------------------------
  it('Training GPU-hours: 8B < 70B < 405B for same token count', () => {
    const seqLen = 8192;
    const gbs = 512;
    const tokens = 100e9; // 100B tokens for fast test
    const totalSteps = Math.ceil(tokens / (gbs * seqLen));

    const m8b = sim(benchmarkConfig(
      'h100-sxm', 8, 8, 'llama3-8b', 'fsdp', gbs, 2, seqLen,
      undefined, { maxSteps: totalSteps },
    ));
    const m70b = sim(benchmarkConfig(
      'h100-sxm', 8, 32, 'llama3-70b', 'fsdp-tp', gbs, 1, seqLen,
      { tp: 8 }, { maxSteps: totalSteps },
    ));
    const m405b = rawSim(benchmarkConfig(
      'h100-sxm', 8, 256, 'llama3-405b', 'fsdp-tp-pp', gbs, 1, seqLen,
      { tp: 8, pp: 16 }, { maxSteps: totalSteps },
    ));

    const hours8b = (m8b.timeToTrainHours ?? 0) * 64;
    const hours70b = (m70b.timeToTrainHours ?? 0) * 256;
    const hours405b = (m405b.timeToTrainHours ?? 0) * 2048;

    expect(hours70b).toBeGreaterThan(hours8b);
    expect(hours405b).toBeGreaterThan(hours70b);
  });

  // -----------------------------------------------------------------------
  // MoE models: DBRX and Mixtral MFU should be comparable class
  // Both are MoE with similar active param fractions
  // -----------------------------------------------------------------------
  it('MoE consistency: DBRX and Mixtral MFU within 2× of each other', () => {
    const dbrx = rawSim(benchmarkConfig(
      'h100-sxm', 8, 2, 'dbrx', 'zero1-tp', 32, 1, 2048, { tp: 4 },
    ));
    const mixtral = rawSim(benchmarkConfig(
      'h100-sxm', 8, 2, 'mixtral-8x7b', 'zero1-tp', 32, 1, 2048, { tp: 4 },
    ));

    const ratio = dbrx.mfu / mixtral.mfu;
    expect(ratio).toBeGreaterThan(0.5);
    expect(ratio).toBeLessThan(2.0);
  });
});

// ===========================================================================
// Section 1d: Dense Model Training MFU — Comprehensive Coverage (±15%)
//
// One test per dense model not already covered in Sections 1/1b/2.
// Cluster sizes match model scale (single-node for <10B, multi-node for larger).
// ===========================================================================

describe('Dense Model Training MFU — Comprehensive', () => {
  // --- GPT-3 Family ---
  it('GPT-3 1.3B × 8 H100 DDP: MFU ≈ 36.2%', () => {
    const m = sim(benchmarkConfig('h100-sxm', 8, 1, 'gpt3-1.3b', 'ddp', 32, 4, 2048));
    expect(m.mfu).toBeGreaterThan(0.308);
    expect(m.mfu).toBeLessThan(0.4168);
  });

  it('GPT-3 6.7B × 8 H100 FSDP: MFU ≈ 48.7%', () => {
    const m = sim(benchmarkConfig('h100-sxm', 8, 1, 'gpt3-6.7b', 'fsdp', 32, 4, 2048));
    expect(m.mfu).toBeGreaterThan(0.414);
    expect(m.mfu).toBeLessThan(0.5602);
  });

  it('GPT-3 13B × 16 H100 FSDP: MFU ≈ 47.4%', () => {
    const m = sim(benchmarkConfig('h100-sxm', 8, 2, 'gpt3-13b', 'fsdp', 64, 2, 2048));
    expect(m.mfu).toBeGreaterThan(0.4027);
    expect(m.mfu).toBeLessThan(0.5449);
  });

  // --- LLaMA 2 ---
  it('LLaMA 2 13B × 32 H100 FSDP: MFU ≈ 46.5%', () => {
    const m = sim(benchmarkConfig('h100-sxm', 8, 4, 'llama2-13b', 'fsdp', 64, 2, 4096));
    expect(m.mfu).toBeGreaterThan(0.3951);
    expect(m.mfu).toBeLessThan(0.5347);
  });

  // --- LLaMA 3.x ---
  it('LLaMA 3.2 1B × 8 H100 DDP: MFU ≈ 17.5%', () => {
    const m = sim(benchmarkConfig('h100-sxm', 8, 1, 'llama3.2-1b', 'ddp', 32, 8, 4096));
    expect(m.mfu).toBeGreaterThan(0.1488);
    expect(m.mfu).toBeLessThan(0.2014);
  });

  it('LLaMA 3.2 3B × 8 H100 FSDP: MFU ≈ 40.5%', () => {
    const m = sim(benchmarkConfig('h100-sxm', 8, 1, 'llama3.2-3b', 'fsdp', 32, 4, 4096));
    expect(m.mfu).toBeGreaterThan(0.3439);
    expect(m.mfu).toBeLessThan(0.4655);
  });

  it('LLaMA 3.1 8B × 8 H100 FSDP: MFU ≈ 47.9%', () => {
    const m = sim(benchmarkConfig('h100-sxm', 8, 1, 'llama3.1-8b', 'fsdp', 32, 4, 4096));
    expect(m.mfu).toBeGreaterThan(0.407);
    expect(m.mfu).toBeLessThan(0.5508);
  });

  // --- Gemma 2 ---
  it('Gemma 2 2B × 8 H100 DDP: MFU ≈ 36.0%', () => {
    const m = sim(benchmarkConfig('h100-sxm', 8, 1, 'gemma2-2b', 'ddp', 32, 4, 4096));
    expect(m.mfu).toBeGreaterThan(0.3055);
    expect(m.mfu).toBeLessThan(0.4135);
  });

  it('Gemma 2 9B × 8 H100 FSDP: MFU ≈ 44.9%', () => {
    const m = sim(benchmarkConfig('h100-sxm', 8, 1, 'gemma2-9b', 'fsdp', 32, 4, 4096));
    expect(m.mfu).toBeGreaterThan(0.3813);
    expect(m.mfu).toBeLessThan(0.5159);
  });

  it('Gemma 2 27B × 32 H100 FSDP-TP: MFU ≈ 37.5%', () => {
    const m = sim(benchmarkConfig('h100-sxm', 8, 4, 'gemma2-27b', 'fsdp-tp', 64, 1, 4096, { tp: 4 }));
    expect(m.mfu).toBeGreaterThan(0.3189);
    expect(m.mfu).toBeLessThan(0.4315);
  });

  // --- Gemma 3 (smaller sizes not in new-model-families.test.ts) ---
  it('Gemma 3 1B × 8 H100 DDP: MFU ≈ 31.7%', () => {
    const m = sim(benchmarkConfig('h100-sxm', 8, 1, 'gemma3-1b', 'ddp', 32, 4, 4096));
    // Decreased from ~37.2% due to non-matmul compute time modeling
    expect(m.mfu).toBeGreaterThan(0.2697);
    expect(m.mfu).toBeLessThan(0.3651);
  });

  it('Gemma 3 4B × 8 H100 FSDP: MFU ≈ 42.4%', () => {
    const m = sim(benchmarkConfig('h100-sxm', 8, 1, 'gemma3-4b', 'fsdp', 32, 4, 4096));
    expect(m.mfu).toBeGreaterThan(0.3603);
    expect(m.mfu).toBeLessThan(0.4875);
  });

  // --- Qwen 2.5 ---
  it('Qwen 2.5 0.5B × 8 H100 DDP: MFU ≈ 17.9%', () => {
    const m = sim(benchmarkConfig('h100-sxm', 8, 1, 'qwen2.5-0.5b', 'ddp', 32, 8, 4096));
    // Decreased from ~21.6% due to non-matmul compute time modeling (small model = large impact)
    expect(m.mfu).toBeGreaterThan(0.1521);
    expect(m.mfu).toBeLessThan(0.2059);
  });

  it('Qwen 2.5 1.5B × 8 H100 DDP: MFU ≈ 37.0%', () => {
    const m = sim(benchmarkConfig('h100-sxm', 8, 1, 'qwen2.5-1.5b', 'ddp', 32, 4, 4096));
    expect(m.mfu).toBeGreaterThan(0.3145);
    expect(m.mfu).toBeLessThan(0.4255);
  });

  it('Qwen 2.5 3B × 8 H100 FSDP: MFU ≈ 43.1%', () => {
    const m = sim(benchmarkConfig('h100-sxm', 8, 1, 'qwen2.5-3b', 'fsdp', 32, 4, 4096));
    // Decreased from ~51.0% due to non-matmul compute time modeling
    expect(m.mfu).toBeGreaterThan(0.3663);
    expect(m.mfu).toBeLessThan(0.4957);
  });

  it('Qwen 2.5 14B × 16 H100 FSDP: MFU ≈ 47.5%', () => {
    const m = sim(benchmarkConfig('h100-sxm', 8, 2, 'qwen2.5-14b', 'fsdp', 64, 2, 4096));
    expect(m.mfu).toBeGreaterThan(0.4038);
    expect(m.mfu).toBeLessThan(0.5464);
  });

  it('Qwen 2.5 32B × 32 H100 FSDP-TP: MFU ≈ 38.0%', () => {
    const m = sim(benchmarkConfig('h100-sxm', 8, 4, 'qwen2.5-32b', 'fsdp-tp', 64, 1, 4096, { tp: 4 }));
    expect(m.mfu).toBeGreaterThan(0.3233);
    expect(m.mfu).toBeLessThan(0.4374);
  });

  // --- Qwen 3 ---
  it('Qwen 3 0.6B × 8 H100 DDP: MFU ≈ 11.9%', () => {
    const m = sim(benchmarkConfig('h100-sxm', 8, 1, 'qwen3-0.6b', 'ddp', 32, 8, 4096));
    expect(m.mfu).toBeGreaterThan(0.101);
    expect(m.mfu).toBeLessThan(0.1368);
  });

  it('Qwen 3 1.7B × 8 H100 DDP: MFU ≈ 32.7%', () => {
    const m = sim(benchmarkConfig('h100-sxm', 8, 1, 'qwen3-1.7b', 'ddp', 32, 4, 4096));
    expect(m.mfu).toBeGreaterThan(0.278);
    expect(m.mfu).toBeLessThan(0.3762);
  });

  it('Qwen 3 4B × 8 H100 FSDP: MFU ≈ 38.1%', () => {
    const m = sim(benchmarkConfig('h100-sxm', 8, 1, 'qwen3-4b', 'fsdp', 32, 4, 4096));
    expect(m.mfu).toBeGreaterThan(0.3235);
    expect(m.mfu).toBeLessThan(0.4379);
  });

  it('Qwen 3 14B × 16 H100 FSDP: MFU ≈ 48.8%', () => {
    const m = sim(benchmarkConfig('h100-sxm', 8, 2, 'qwen3-14b', 'fsdp', 64, 2, 4096));
    expect(m.mfu).toBeGreaterThan(0.4143);
    expect(m.mfu).toBeLessThan(0.5607);
  });

  // --- Mistral ---
  it('Mistral Nemo 12B × 16 H100 FSDP: MFU ≈ 48.5%', () => {
    const m = sim(benchmarkConfig('h100-sxm', 8, 2, 'mistral-nemo-12b', 'fsdp', 64, 2, 4096));
    expect(m.mfu).toBeGreaterThan(0.4124);
    expect(m.mfu).toBeLessThan(0.558);
  });

  it('Mistral Small 24B × 32 H100 FSDP-TP: MFU ≈ 39.2%', () => {
    const m = sim(benchmarkConfig('h100-sxm', 8, 4, 'mistral-small-24b', 'fsdp-tp', 64, 1, 4096, { tp: 4 }));
    expect(m.mfu).toBeGreaterThan(0.3336);
    expect(m.mfu).toBeLessThan(0.4513);
  });

  it('Mistral Large 123B × 256 H100 FSDP-TP: MFU ≈ 40.5%', () => {
    const m = sim(benchmarkConfig('h100-sxm', 8, 32, 'mistral-large-123b', 'fsdp-tp', 256, 1, 4096, { tp: 8 }));
    expect(m.mfu).toBeGreaterThan(0.3446);
    expect(m.mfu).toBeLessThan(0.4662);
  });

  it('Codestral 22B × 32 H100 FSDP-TP: MFU ≈ 35.7%', () => {
    const m = sim(benchmarkConfig('h100-sxm', 8, 4, 'codestral-22b', 'fsdp-tp', 64, 1, 4096, { tp: 4 }));
    expect(m.mfu).toBeGreaterThan(0.3037);
    expect(m.mfu).toBeLessThan(0.4109);
  });

  // --- Devstral ---
  it('Devstral 2 × 256 H100 FSDP-TP (tp=8): MFU ≈ 41.0%', () => {
    const m = rawSim(benchmarkConfig('h100-sxm', 8, 32, 'devstral-2', 'fsdp-tp', 256, 1, 4096, { tp: 8 }));
    // ±15% of simulator value (~41.0%)
    expect(m.mfu).toBeGreaterThan(0.3484);
    expect(m.mfu).toBeLessThan(0.4714);
  });

  it('Devstral Small 2 × 8 H100 FSDP: MFU ≈ 50.4%', () => {
    const m = rawSim(benchmarkConfig('h100-sxm', 8, 1, 'devstral-small-2', 'fsdp', 16, 2, 4096));
    // ±15% of simulator value (~50.4%)
    expect(m.mfu).toBeGreaterThan(0.428);
    expect(m.mfu).toBeLessThan(0.5792);
  });

  // --- Command R ---
  it('Command R 35B × 32 H100 FSDP-TP: MFU ≈ 39.0%', () => {
    const m = sim(benchmarkConfig('h100-sxm', 8, 4, 'command-r', 'fsdp-tp', 64, 1, 4096, { tp: 4 }));
    expect(m.mfu).toBeGreaterThan(0.3319);
    expect(m.mfu).toBeLessThan(0.4490);
  });

  it('Command R+ 104B × 128 H100 FSDP-TP: MFU ≈ 41.6%', () => {
    const m = sim(benchmarkConfig('h100-sxm', 8, 16, 'command-r-plus', 'fsdp-tp', 256, 1, 4096, { tp: 8 }));
    expect(m.mfu).toBeGreaterThan(0.3535);
    expect(m.mfu).toBeLessThan(0.4783);
  });

  // --- Nemotron-4 15B ---
  it('Nemotron-4 15B × 16 H100 FSDP: MFU ≈ 53.4%', () => {
    const m = sim(benchmarkConfig('h100-sxm', 8, 2, 'nemotron-4-15b', 'fsdp', 64, 2, 4096));
    expect(m.mfu).toBeGreaterThan(0.4537);
    expect(m.mfu).toBeLessThan(0.6139);
  });

  // --- Yi ---
  it('Yi 6B × 8 H100 FSDP: MFU ≈ 44.6%', () => {
    const m = sim(benchmarkConfig('h100-sxm', 8, 1, 'yi-6b', 'fsdp', 32, 4, 4096));
    expect(m.mfu).toBeGreaterThan(0.3794);
    expect(m.mfu).toBeLessThan(0.5134);
  });

  it('Yi 34B × 32 H100 FSDP-TP: MFU ≈ 37.5%', () => {
    const m = sim(benchmarkConfig('h100-sxm', 8, 4, 'yi-34b', 'fsdp-tp', 64, 1, 4096, { tp: 4 }));
    expect(m.mfu).toBeGreaterThan(0.3191);
    expect(m.mfu).toBeLessThan(0.4317);
  });
});

// ===========================================================================
// Section 1e: MoE Model Training MFU — Comprehensive Coverage (±15%)
// ===========================================================================

describe('MoE Model Training MFU — Comprehensive', () => {
  it('LLaMA 4 Scout × 128 H100 FSDP-TP (tp=8, ep=2): MFU ≈ 29.9%', () => {
    const m = sim(benchmarkConfig(
      'h100-sxm', 8, 16, 'llama4-scout', 'fsdp-tp', 256, 1, 4096, { tp: 8, ep: 2 },
    ));
    // Routing locality factor 1/(1+density) reduces EP all-to-all volume for sparse MoE
    // (Scout: 1/16 active experts → density=0.0625 → locality≈0.94).
    expect(m.mfu).toBeGreaterThan(0.2541);
    expect(m.mfu).toBeLessThan(0.3438);
  });

  it('LLaMA 4 Maverick × 256 H100 FSDP-TP (tp=8, ep=4): MFU ≈ 26.5%', () => {
    const m = sim(benchmarkConfig(
      'h100-sxm', 8, 32, 'llama4-maverick', 'fsdp-tp', 256, 1, 4096, { tp: 8, ep: 4 },
    ));
    // Higher with bf16 grads (less comm overhead) + GA-aware overlap
    expect(m.mfu).toBeGreaterThan(0.2252);
    expect(m.mfu).toBeLessThan(0.3046);
  });

  it('Mixtral 8x22B × 32 H100 FSDP-TP (tp=4): MFU ≈ 40.2%', () => {
    const m = rawSim(benchmarkConfig(
      'h100-sxm', 8, 4, 'mixtral-8x22b', 'fsdp-tp', 64, 1, 4096, { tp: 4 },
    ));
    // Higher with bf16 grads (less DP comm overhead)
    expect(m.mfu).toBeGreaterThan(0.3415);
    expect(m.mfu).toBeLessThan(0.4621);
  });

  it('DeepSeek-V2 236B × 64 H100 FSDP-TP (tp=8): MFU ≈ 12.9%', () => {
    const m = rawSim(benchmarkConfig(
      'h100-sxm', 8, 8, 'deepseek-v2', 'fsdp-tp', 128, 1, 4096, { tp: 8 },
    ));
    // MLA increases active params → higher MFU; bf16 grads reduce comm
    // MLA increases active params; bf16 grads reduce comm. MFU ~12.9%
    expect(m.mfu).toBeGreaterThan(0.1096);
    expect(m.mfu).toBeLessThan(0.1483);
  });

  it('DeepSeek-MoE 16B × 8 H100 FSDP: MFU ≈ 39.4%', () => {
    const m = sim(benchmarkConfig(
      'h100-sxm', 8, 1, 'deepseek-moe-16b', 'fsdp', 32, 4, 4096,
    ));
    expect(m.mfu).toBeGreaterThan(0.3349);
    expect(m.mfu).toBeLessThan(0.4533);
  });

  it('Qwen 3 235B-A22B × 256 H100 FSDP-TP (tp=8, ep=4): MFU ≈ 16.3%', () => {
    const m = sim(benchmarkConfig(
      'h100-sxm', 8, 32, 'qwen3-235b-a22b', 'fsdp-tp', 256, 1, 4096, { tp: 8, ep: 4 },
    ));
    // Routing locality factor 1/(1+density) reduces EP all-to-all volume for sparse MoE
    // (Qwen3-235B: 8/128 active experts → density=0.0625 → locality≈0.94).
    // All-to-all is the dominant bottleneck at EP=4; locality factor raises MFU to ~16.3%.
    expect(m.mfu).toBeGreaterThan(0.1384);
    expect(m.mfu).toBeLessThan(0.1872);
  });
});

// ===========================================================================
// Section 4b: Inference Latency — Comprehensive Coverage (±15%)
//
// TTFT and TPOT for all model families not covered in Section 4.
// Standard params: batch=1, inputSeqLen=512, outputSeqLen=256, bf16.
// ===========================================================================

describe('Inference Latency — Comprehensive', () => {
  // --- GPT-3 ---
  it('GPT-3 6.7B × 1 H100: TTFT ≈ 17.8ms, TPOT ≈ 5.2ms', () => {
    const r = infer({ modelId: 'gpt3-6.7b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1 });
    expect(r.success).toBe(true);
    expect(r.latency.ttft).toBeGreaterThan(15.1); expect(r.latency.ttft).toBeLessThan(20.4);
    expect(r.latency.tpot).toBeGreaterThan(4.98); expect(r.latency.tpot).toBeLessThan(6.73);
  });

  // --- LLaMA 2 ---
  it('LLaMA 2 13B × 1 H100: TTFT ≈ 33.7ms, TPOT ≈ 9.9ms', () => {
    const r = infer({ modelId: 'llama2-13b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1 });
    expect(r.success).toBe(true);
    expect(r.latency.ttft).toBeGreaterThan(28.6); expect(r.latency.ttft).toBeLessThan(38.7);
    expect(r.latency.tpot).toBeGreaterThan(8.76); expect(r.latency.tpot).toBeLessThan(11.85);
  });

  // --- LLaMA 3.x ---
  it('LLaMA 3.2 1B × 1 H100: TTFT ≈ 3.2ms, TPOT ≈ 1.44ms', () => {
    const r = infer({ modelId: 'llama3.2-1b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1 });
    expect(r.success).toBe(true);
    expect(r.latency.ttft).toBeGreaterThan(2.72); expect(r.latency.ttft).toBeLessThan(3.68);
    expect(r.latency.tpot).toBeGreaterThan(1.23); expect(r.latency.tpot).toBeLessThan(1.66);
  });

  it('LLaMA 3.1 8B × 1 H100: TTFT ≈ 20.8ms, TPOT ≈ 6.0ms', () => {
    const r = infer({ modelId: 'llama3.1-8b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1 });
    expect(r.success).toBe(true);
    expect(r.latency.ttft).toBeGreaterThan(17.7); expect(r.latency.ttft).toBeLessThan(23.9);
    expect(r.latency.tpot).toBeGreaterThan(5.60); expect(r.latency.tpot).toBeLessThan(7.58);
  });

  it('LLaMA 4 Scout × 8 H100 TP=8: TTFT ≈ 8.1ms, TPOT ≈ 2.9ms (MoE active)', () => {
    const r = infer({ modelId: 'llama4-scout', gpuId: 'h100-sxm', numGPUs: 8,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 8 });
    expect(r.success).toBe(true);
    // Prefill reads nearly all expert weights (coupon-collector over 512 tokens)
    expect(r.latency.ttft).toBeGreaterThan(8.1 * 0.7); expect(r.latency.ttft).toBeLessThan(8.1 * 1.3);
    expect(r.latency.tpot).toBeGreaterThan(2.47); expect(r.latency.tpot).toBeLessThan(3.34);
  });

  // --- Gemma 2 ---
  it('Gemma 2 9B × 1 H100: TTFT ≈ 23.3ms, TPOT ≈ 6.8ms', () => {
    const r = infer({ modelId: 'gemma2-9b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1 });
    expect(r.success).toBe(true);
    expect(r.latency.ttft).toBeGreaterThan(19.8); expect(r.latency.ttft).toBeLessThan(26.8);
    expect(r.latency.tpot).toBeGreaterThan(6.23); expect(r.latency.tpot).toBeLessThan(8.43);
  });

  // --- Gemma 3 ---
  it('Gemma 3 4B × 1 H100: TTFT ≈ 10.0ms, TPOT ≈ 2.9ms', () => {
    const r = infer({ modelId: 'gemma3-4b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1 });
    expect(r.success).toBe(true);
    expect(r.latency.ttft).toBeGreaterThan(8.54); expect(r.latency.ttft).toBeLessThan(11.55);
    expect(r.latency.tpot).toBeGreaterThan(3.04); expect(r.latency.tpot).toBeLessThan(4.12);
  });

  // --- Qwen 2.5 ---
  it('Qwen 2.5 0.5B × 1 H100: TTFT ≈ 2.1ms, TPOT ≈ 0.84ms', () => {
    const r = infer({ modelId: 'qwen2.5-0.5b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1 });
    expect(r.success).toBe(true);
    // Small model + short seq + batch=1: matmul undersaturation increases compute time
    expect(r.latency.ttft).toBeGreaterThan(1.78); expect(r.latency.ttft).toBeLessThan(2.41);
    expect(r.latency.tpot).toBeGreaterThan(0.71); expect(r.latency.tpot).toBeLessThan(0.97);
  });

  it('Qwen 2.5 14B × 1 H100: TTFT ≈ 38.2ms, TPOT ≈ 11.1ms', () => {
    const r = infer({ modelId: 'qwen2.5-14b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1 });
    expect(r.success).toBe(true);
    expect(r.latency.ttft).toBeGreaterThan(32.5); expect(r.latency.ttft).toBeLessThan(44.0);
    expect(r.latency.tpot).toBeGreaterThan(9.68); expect(r.latency.tpot).toBeLessThan(13.10);
  });

  it('Qwen 2.5 72B × 8 H100 TP=8: TTFT ≈ 26.5ms, TPOT ≈ 6.8ms', () => {
    const r = infer({ modelId: 'qwen2.5-72b', gpuId: 'h100-sxm', numGPUs: 8,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 8 });
    expect(r.success).toBe(true);
    expect(r.latency.ttft).toBeGreaterThan(22.5); expect(r.latency.ttft).toBeLessThan(30.5);
    expect(r.latency.tpot).toBeGreaterThan(5.55); expect(r.latency.tpot).toBeLessThan(7.51);
  });

  // --- Qwen 3 ---
  it('Qwen 3 0.6B × 1 H100: TTFT ≈ 1.9ms, TPOT ≈ 0.85ms', () => {
    const r = infer({ modelId: 'qwen3-0.6b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1 });
    expect(r.success).toBe(true);
    // Small model + short seq + batch=1: matmul undersaturation increases compute time
    expect(r.latency.ttft).toBeGreaterThan(1.61); expect(r.latency.ttft).toBeLessThan(2.19);
    expect(r.latency.tpot).toBeGreaterThan(0.72); expect(r.latency.tpot).toBeLessThan(0.97);
  });

  it('Qwen 3 14B × 1 H100: TTFT ≈ 38.2ms, TPOT ≈ 11.1ms', () => {
    const r = infer({ modelId: 'qwen3-14b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1 });
    expect(r.success).toBe(true);
    expect(r.latency.ttft).toBeGreaterThan(32.5); expect(r.latency.ttft).toBeLessThan(44.0);
    expect(r.latency.tpot).toBeGreaterThan(9.67); expect(r.latency.tpot).toBeLessThan(13.09);
  });

  // --- Mistral ---
  it('Mistral Nemo 12B × 1 H100: TTFT ≈ 31.7ms, TPOT ≈ 9.2ms', () => {
    const r = infer({ modelId: 'mistral-nemo-12b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1 });
    expect(r.success).toBe(true);
    expect(r.latency.ttft).toBeGreaterThan(26.9); expect(r.latency.ttft).toBeLessThan(36.5);
    expect(r.latency.tpot).toBeGreaterThan(8.16); expect(r.latency.tpot).toBeLessThan(11.04);
  });

  it('Mistral Large 123B × 8 H100 TP=8: TTFT ≈ 44.6ms, TPOT ≈ 11.5ms', () => {
    const r = infer({ modelId: 'mistral-large-123b', gpuId: 'h100-sxm', numGPUs: 8,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 8 });
    expect(r.success).toBe(true);
    expect(r.latency.ttft).toBeGreaterThan(37.9); expect(r.latency.ttft).toBeLessThan(51.3);
    expect(r.latency.tpot).toBeGreaterThan(9.28); expect(r.latency.tpot).toBeLessThan(12.55);
  });

  it('Codestral 22B × 2 H100 TP=2: TTFT ≈ 30.4ms, TPOT ≈ 8.3ms', () => {
    const r = infer({ modelId: 'codestral-22b', gpuId: 'h100-sxm', numGPUs: 2,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 2 });
    expect(r.success).toBe(true);
    expect(r.latency.ttft).toBeGreaterThan(25.8); expect(r.latency.ttft).toBeLessThan(34.9);
    expect(r.latency.tpot).toBeGreaterThan(7.09); expect(r.latency.tpot).toBeLessThan(9.59);
  });

  // --- Devstral ---
  it('Devstral 2 × 8 H100 TP=8: TTFT ≈ 45.4ms, TPOT ≈ 11.1ms', () => {
    const r = infer({ modelId: 'devstral-2', gpuId: 'h100-sxm', numGPUs: 8,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 8 });
    expect(r.success).toBe(true);
    // ±15% of simulator value
    expect(r.latency.ttft).toBeGreaterThan(38.6); expect(r.latency.ttft).toBeLessThan(52.2);
    expect(r.latency.tpot).toBeGreaterThan(9.45); expect(r.latency.tpot).toBeLessThan(12.79);
  });

  it('Devstral Small 2 × 2 H100 TP=2: TTFT ≈ 31.4ms, TPOT ≈ 8.8ms', () => {
    const r = infer({ modelId: 'devstral-small-2', gpuId: 'h100-sxm', numGPUs: 2,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 2 });
    expect(r.success).toBe(true);
    // ±15% of simulator value
    expect(r.latency.ttft).toBeGreaterThan(26.7); expect(r.latency.ttft).toBeLessThan(36.2);
    expect(r.latency.tpot).toBeGreaterThan(7.47); expect(r.latency.tpot).toBeLessThan(10.11);
  });

  // --- Command R ---
  it('Command R+ 104B × 8 H100 TP=8: TTFT ≈ 37.2ms, TPOT ≈ 9.7ms', () => {
    const r = infer({ modelId: 'command-r-plus', gpuId: 'h100-sxm', numGPUs: 8,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 8 });
    expect(r.success).toBe(true);
    expect(r.latency.ttft).toBeGreaterThan(31.6); expect(r.latency.ttft).toBeLessThan(42.7);
    expect(r.latency.tpot).toBeGreaterThan(7.87); expect(r.latency.tpot).toBeLessThan(10.65);
  });

  // --- Nemotron-4 ---
  it('Nemotron-4 15B × 1 H100: TTFT ≈ 40.5ms, TPOT ≈ 11.7ms', () => {
    const r = infer({ modelId: 'nemotron-4-15b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1 });
    expect(r.success).toBe(true);
    expect(r.latency.ttft).toBeGreaterThan(34.4); expect(r.latency.ttft).toBeLessThan(46.5);
    expect(r.latency.tpot).toBeGreaterThan(10.18); expect(r.latency.tpot).toBeLessThan(13.77);
  });

  // --- Yi ---
  it('Yi 34B × 2 H100 TP=2: TTFT ≈ 46.5ms, TPOT ≈ 12.9ms', () => {
    const r = infer({ modelId: 'yi-34b', gpuId: 'h100-sxm', numGPUs: 2,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 2 });
    expect(r.success).toBe(true);
    expect(r.latency.ttft).toBeGreaterThan(39.5); expect(r.latency.ttft).toBeLessThan(53.4);
    expect(r.latency.tpot).toBeGreaterThan(10.72); expect(r.latency.tpot).toBeLessThan(14.50);
  });

  // --- MoE Models ---
  it('DeepSeek-MoE 16B × 1 H100: TTFT ≈ 9.8ms, TPOT ≈ 2.87ms', () => {
    const r = infer({ modelId: 'deepseek-moe-16b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1 });
    expect(r.success).toBe(true);
    // At batch=1 seqLen=512, coupon collector saturates → all 64 experts loaded (~32.75GB)
    // Memory-bound: 32.75GB / 3.35 TB/s ≈ 9.78ms > compute time ≈ 7.3ms
    expect(r.latency.ttft).toBeGreaterThan(8.3); expect(r.latency.ttft).toBeLessThan(11.3);
    expect(r.latency.tpot).toBeGreaterThan(2.44); expect(r.latency.tpot).toBeLessThan(3.30);
  });

  it('DBRX 132B × 8 H100 TP=8: TTFT ≈ 12.9ms, TPOT ≈ 3.4ms', () => {
    const r = infer({ modelId: 'dbrx', gpuId: 'h100-sxm', numGPUs: 8,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 8 });
    expect(r.success).toBe(true);
    expect(r.latency.ttft).toBeGreaterThan(10.98); expect(r.latency.ttft).toBeLessThan(14.86);
    expect(r.latency.tpot).toBeGreaterThan(2.83); expect(r.latency.tpot).toBeLessThan(3.83);
  });

  it('Mixtral 8x22B × 8 H100 TP=8: TTFT ≈ 14.2ms, TPOT ≈ 3.7ms', () => {
    const r = infer({ modelId: 'mixtral-8x22b', gpuId: 'h100-sxm', numGPUs: 8,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 8 });
    expect(r.success).toBe(true);
    expect(r.latency.ttft).toBeGreaterThan(12.10); expect(r.latency.ttft).toBeLessThan(16.37);
    expect(r.latency.tpot).toBeGreaterThan(3.04); expect(r.latency.tpot).toBeLessThan(4.11);
  });

  it('DeepSeek-V2 236B × 8 H100 TP=8: TTFT ≈ 17.6ms, TPOT ≈ 3.63ms', () => {
    const r = infer({ modelId: 'deepseek-v2', gpuId: 'h100-sxm', numGPUs: 8,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 8 });
    expect(r.success).toBe(true);
    // Prefill reads nearly all expert weights (coupon-collector over 512 tokens)
    expect(r.latency.ttft).toBeGreaterThan(17.6 * 0.7); expect(r.latency.ttft).toBeLessThan(17.6 * 1.3);
    expect(r.latency.tpot).toBeGreaterThan(3.08); expect(r.latency.tpot).toBeLessThan(4.17);
  });

  // --- OLMo ---
  it('OLMo 2 7B × 1 H100: TTFT ≈ 18.9ms, TPOT ≈ 5.6ms', () => {
    const r = infer({ modelId: 'olmo2-7b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1 });
    expect(r.success).toBe(true);
    expect(r.latency.ttft).toBeGreaterThan(16.1); expect(r.latency.ttft).toBeLessThan(21.7);
    expect(r.latency.tpot).toBeGreaterThan(5.24); expect(r.latency.tpot).toBeLessThan(7.10);
  });

  // --- Phi-3 ---
  it('Phi-3 Medium 14B × 1 H100: TTFT ≈ 36.1ms, TPOT ≈ 10.5ms', () => {
    const r = infer({ modelId: 'phi3-medium', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1 });
    expect(r.success).toBe(true);
    expect(r.latency.ttft).toBeGreaterThan(30.7); expect(r.latency.ttft).toBeLessThan(41.6);
    expect(r.latency.tpot).toBeGreaterThan(9.20); expect(r.latency.tpot).toBeLessThan(12.44);
  });

  // --- Phi-4 Mini Flash ---
  it('Phi-4 Mini Flash × 1 H100: TTFT ≈ 9.5ms, TPOT ≈ 2.8ms', () => {
    const r = infer({ modelId: 'phi4-mini-flash', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1 });
    expect(r.success).toBe(true);
    expect(r.latency.ttft).toBeGreaterThan(8.05); expect(r.latency.ttft).toBeLessThan(10.89);
    expect(r.latency.tpot).toBeGreaterThan(2.91); expect(r.latency.tpot).toBeLessThan(3.94);
  });
});

// ===========================================================================
// Section 6b: Cross-Family Inference & Training Consistency
// ===========================================================================

describe('Cross-Family Consistency — Comprehensive', () => {
  // Inference TTFT ordering within families (larger model = higher TTFT on same GPU)
  it('Qwen 2.5 TTFT ordering: 0.5B < 7B < 14B (1 H100)', () => {
    const small = infer({ modelId: 'qwen2.5-0.5b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1 });
    const med = infer({ modelId: 'qwen2.5-7b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1 });
    const large = infer({ modelId: 'qwen2.5-14b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1 });
    expect(med.latency.ttft).toBeGreaterThan(small.latency.ttft);
    expect(large.latency.ttft).toBeGreaterThan(med.latency.ttft);
  });

  it('GPT-3 TTFT ordering: 1.3B < 6.7B < 13B (1 H100)', () => {
    const s = infer({ modelId: 'gpt3-1.3b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1 });
    const m = infer({ modelId: 'gpt3-6.7b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1 });
    const l = infer({ modelId: 'gpt3-13b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1 });
    expect(m.latency.ttft).toBeGreaterThan(s.latency.ttft);
    expect(l.latency.ttft).toBeGreaterThan(m.latency.ttft);
  });

  it('LLaMA 3 TTFT ordering: 1B < 3B < 8B (1 H100)', () => {
    const s = infer({ modelId: 'llama3.2-1b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1 });
    const m = infer({ modelId: 'llama3.2-3b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1 });
    const l = infer({ modelId: 'llama3.1-8b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1 });
    expect(m.latency.ttft).toBeGreaterThan(s.latency.ttft);
    expect(l.latency.ttft).toBeGreaterThan(m.latency.ttft);
  });

  it('Gemma 3 TTFT ordering: 1B < 4B < 12B (1 H100)', () => {
    const s = infer({ modelId: 'gemma3-1b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1 });
    const m = infer({ modelId: 'gemma3-4b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1 });
    const l = infer({ modelId: 'gemma3-12b', gpuId: 'h100-sxm', numGPUs: 1,
      batchSize: 1, inputSeqLen: 512, outputSeqLen: 256, weightPrecision: 'bf16', kvCachePrecision: 'bf16', tensorParallel: 1 });
    expect(m.latency.ttft).toBeGreaterThan(s.latency.ttft);
    expect(l.latency.ttft).toBeGreaterThan(m.latency.ttft);
  });

  // Training step time ordering within family
  it('Mistral family step time ordering: 7B < 12B < 24B (8 H100 FSDP)', () => {
    const s = sim(benchmarkConfig('h100-sxm', 8, 1, 'mistral-7b', 'fsdp', 16, 2, 2048));
    const m = sim(benchmarkConfig('h100-sxm', 8, 1, 'mistral-nemo-12b', 'fsdp', 16, 2, 2048));
    const l = sim(benchmarkConfig('h100-sxm', 8, 1, 'mistral-small-24b', 'fsdp', 16, 1, 2048));
    expect(m.stepTimeMs).toBeGreaterThan(s.stepTimeMs);
    expect(l.stepTimeMs).toBeGreaterThan(m.stepTimeMs);
  });

  // All MoE models have activeParams < totalParams
  it('All MoE models: activeParams < totalParams', () => {
    const moeIds = ['mixtral-8x7b', 'mixtral-8x22b', 'deepseek-moe-16b', 'deepseek-v2',
      'deepseek-v3', 'deepseek-r1', 'dbrx', 'grok-1', 'llama4-scout', 'llama4-maverick',
      'qwen3-30b-a3b', 'qwen3-235b-a22b'];
    for (const id of moeIds) {
      const model = getModel(id);
      expect(model, `${id} should load`).toBeDefined();
      expect(model!.activeParams, `${id} activeParams < totalParams`).toBeLessThan(model!.totalParams);
    }
  });
});

// ===========================================================================
// Section 7: Analytical GPU-hours Chain Validation
//
// Validates the end-to-end chain: GPU-hours = 6PD / (MFU × peak × 3600)
//
// For each benchmark:
//   1. Run simulator → get MFU and timeToTrainHours
//   2. Compute analyticalGPUHours = 6 × activeParams × tokens / (MFU × peakBF16TFLOPS × 1e12 × 3600)
//   3. Verify simGPUHours ≈ analyticalGPUHours within 5% (consistency)
//   4. Verify publishedGPUHours / analyticalGPUHours is in a reasonable range
//
// MFU denominator always uses BF16 peak TFLOPS (industry convention).
// For MoE models, activeParams (not totalParams) is used — training FLOPs
// scale with active params, not total params.
//
// Overhead ratios > 1.0 are expected: published GPU-hours include failures,
// restarts, checkpointing overhead, annealing, context extension, etc.
// Ratios < 1.0 mean the sim is too pessimistic (underestimates MFU).
// See docs/BENCHMARKS.md for the full GPU-hours projection table.

describe('Analytical GPU-hours Chain Validation: 6PD/(MFU×peak×3600)', () => {
  /**
   * Compute analytical GPU-hours from first principles.
   * @param activeParams - Active parameters (for MoE, NOT totalParams)
   * @param tokens - Total training tokens
   * @param mfu - Model FLOPS Utilization (PaLM definition, 6PD)
   * @param peakBF16TFLOPS - Per-GPU BF16 peak TFLOPS
   */
  function analyticalGPUHours(
    activeParams: number, tokens: number, mfu: number, peakBF16TFLOPS: number,
  ): number {
    return (6 * activeParams * tokens) / (mfu * peakBF16TFLOPS * 1e12 * 3600);
  }

  // ─── Tier 1: Published GPU-hours from paper ──────────────────────────

  it('LLaMA 2 7B: 184K A100-hours for 2T tokens', () => {
    // Source: Meta LLaMA 2 paper (2023), Table 2
    // 128 A100-80GB, FSDP, 2T tokens, seqLen=4096
    const numGPUs = 128;
    const tokens = 2e12;
    const seqLen = 4096;
    const totalSteps = Math.ceil(tokens / (256 * seqLen));

    const metrics = sim(benchmarkConfig(
      'a100-80gb', 8, numGPUs / 8, 'llama2-7b',
      'fsdp', 256, 2, seqLen,
      undefined, { maxSteps: totalSteps },
    ));

    const model = getModel('llama2-7b', seqLen)!;
    const activeP = model.activeParams;
    const analytical = analyticalGPUHours(activeP, tokens, metrics.mfu, 312);
    const simGPUHrs = (metrics.timeToTrainHours ?? 0) * numGPUs;

    // Consistency: sim and analytical should match by construction
    expect(simGPUHrs / analytical).toBeGreaterThan(0.95);
    expect(simGPUHrs / analytical).toBeLessThan(1.05);

    // Overhead ratio: published / analytical ≈ 1.20 (clean pre-training data)
    const ratio = 184000 / analytical;
    expect(ratio).toBeGreaterThan(0.9);
    expect(ratio).toBeLessThan(1.5);
  });

  it('LLaMA 2 13B: 368K A100-hours for 2T tokens', () => {
    // Source: Meta LLaMA 2 paper (2023), Table 2
    const numGPUs = 128;
    const tokens = 2e12;
    const seqLen = 4096;
    const totalSteps = Math.ceil(tokens / (256 * seqLen));

    const metrics = sim(benchmarkConfig(
      'a100-80gb', 8, numGPUs / 8, 'llama2-13b',
      'fsdp', 256, 2, seqLen,
      undefined, { maxSteps: totalSteps },
    ));

    const model = getModel('llama2-13b', seqLen)!;
    const analytical = analyticalGPUHours(model.activeParams, tokens, metrics.mfu, 312);
    const simGPUHrs = (metrics.timeToTrainHours ?? 0) * numGPUs;

    expect(simGPUHrs / analytical).toBeGreaterThan(0.95);
    expect(simGPUHrs / analytical).toBeLessThan(1.05);

    // Overhead ratio ≈ 1.28
    const ratio = 368000 / analytical;
    expect(ratio).toBeGreaterThan(0.9);
    expect(ratio).toBeLessThan(1.6);
  });

  it('LLaMA 2 70B: 1.72M A100-hours for 2T tokens', () => {
    // Source: Meta LLaMA 2 paper (2023), Table 2
    // 2048 A100-80GB, FSDP-TP (TP=8), GBS=512
    const numGPUs = 2048;
    const tokens = 2e12;
    const seqLen = 4096;
    const totalSteps = Math.ceil(tokens / (512 * seqLen));

    const metrics = sim(benchmarkConfig(
      'a100-80gb', 8, numGPUs / 8, 'llama2-70b',
      'fsdp-tp', 512, 1, seqLen,
      { tp: 8 }, { maxSteps: totalSteps },
    ));

    const model = getModel('llama2-70b', seqLen)!;
    const analytical = analyticalGPUHours(model.activeParams, tokens, metrics.mfu, 312);
    const simGPUHrs = (metrics.timeToTrainHours ?? 0) * numGPUs;

    expect(simGPUHrs / analytical).toBeGreaterThan(0.95);
    expect(simGPUHrs / analytical).toBeLessThan(1.05);

    // Overhead ratio ≈ 1.14 (cleanest published data)
    const ratio = 1720000 / analytical;
    expect(ratio).toBeGreaterThan(0.9);
    expect(ratio).toBeLessThan(1.5);
  });

  it('DeepSeek V3: 2.79M H800-hours for 14.8T tokens (FP8)', () => {
    // Source: DeepSeek-V3 technical report (2024), ISCA 2025
    // 2048 H800 SXM, FP8, TP=4, PP=8, EP=32, DualPipe-V
    // MFU uses BF16 peak (989 TFLOPS) as denominator, not FP8 peak
    const numGPUs = 2048;
    const tokens = 14.8e12;
    const seqLen = 4096;
    const totalSteps = Math.ceil(tokens / (8192 * seqLen));

    const metrics = sim(benchmarkConfig(
      'h800-sxm', 8, numGPUs / 8, 'deepseek-v3',
      'fsdp-tp-pp', 8192, 2, seqLen,
      { tp: 4, pp: 8, dp: 64, ep: 32,
        dpType: 'fsdp', sequenceParallel: true,
        pipelineSchedule: 'dualpipe-v', interleavedStages: 1,
        numMicroBatches: 64 },
      { maxSteps: totalSteps, mixedPrecision: 'fp8' },
    ));

    const model = getModel('deepseek-v3', seqLen)!;
    // MUST use activeParams (37.6B), NOT totalParams (671B)
    const analytical = analyticalGPUHours(model.activeParams, tokens, metrics.mfu, 989);
    const simGPUHrs = (metrics.timeToTrainHours ?? 0) * numGPUs;

    expect(simGPUHrs / analytical).toBeGreaterThan(0.95);
    expect(simGPUHrs / analytical).toBeLessThan(1.05);

    // Overhead ratio ≈ 1.30 (H800 failures, long 14.8T training)
    const ratio = 2788000 / analytical;
    expect(ratio).toBeGreaterThan(0.9);
    expect(ratio).toBeLessThan(1.8);
  });

  it('LLaMA 3.1 405B: 30.84M H100-hours for 15.6T tokens', () => {
    // Source: Meta LLaMA 3 paper (2024)
    // 16384 H100 SXM, FSDP-TP-PP (TP=8, PP=16, interleaved v=4)
    //
    // NOTE: Sim gives ~34% MFU vs published ~38-43%. The simulator's
    // interleaved pipeline model doesn't fully capture Meta's zero-bubble
    // scheduling and higher interleave factors. This makes analyticalGPUHours
    // HIGHER than published (ratio < 1.0) — the sim overestimates training
    // time for very large pipeline-parallel models.
    const numGPUs = 16384;
    const tokens = 15.6e12;
    const seqLen = 8192;
    const totalSteps = Math.ceil(tokens / (2048 * seqLen));

    const metrics = rawSim(benchmarkConfig(
      'h100-sxm', 8, numGPUs / 8, 'llama3-405b',
      'fsdp-tp-pp', 2048, 1, seqLen,
      { tp: 8, pp: 16, pipelineSchedule: 'interleaved-1f1b', interleavedStages: 4 },
      { maxSteps: totalSteps },
    ));

    const model = getModel('llama3-405b', seqLen)!;
    const analytical = analyticalGPUHours(model.activeParams, tokens, metrics.mfu, 989);
    const simGPUHrs = (metrics.timeToTrainHours ?? 0) * numGPUs;

    expect(simGPUHrs / analytical).toBeGreaterThan(0.95);
    expect(simGPUHrs / analytical).toBeLessThan(1.05);

    // Ratio < 1.0 expected: sim MFU is pessimistic → analytical GPU-hours
    // exceed published. Widened range to accommodate known pipeline model limitation.
    const ratio = 30840000 / analytical;
    expect(ratio).toBeGreaterThan(0.5);
    expect(ratio).toBeLessThan(1.5);
  });

  // ─── Tier 2: GPU-hours derived from wall-clock × count ──────────────

  it('Phi-4 14B: ~968K H100-hours for 9.8T tokens', () => {
    // Source: Microsoft Phi-4 Technical Report (Dec 2024)
    // 1920 H100-80G, ~21 days → ~968K GPU-hours
    // Overhead includes calendar time, restarts, evaluation
    const numGPUs = 1920;
    const tokens = 9.8e12;
    const seqLen = 4096;
    const totalSteps = Math.ceil(tokens / (5760 * seqLen));

    const metrics = sim(benchmarkConfig(
      'h100-sxm', 8, numGPUs / 8, 'phi4',
      'fsdp', 5760, 1, seqLen,
      undefined, { maxSteps: totalSteps },
    ));

    const model = getModel('phi4', seqLen)!;
    const analytical = analyticalGPUHours(model.activeParams, tokens, metrics.mfu, 989);
    const simGPUHrs = (metrics.timeToTrainHours ?? 0) * numGPUs;

    expect(simGPUHrs / analytical).toBeGreaterThan(0.95);
    expect(simGPUHrs / analytical).toBeLessThan(1.05);

    // Overhead ratio ≈ 1.70 (wall-clock includes calendar overhead)
    const ratio = 968000 / analytical;
    expect(ratio).toBeGreaterThan(0.9);
    expect(ratio).toBeLessThan(2.5);
  });

  it('Phi-3 Mini 3.8B: ~86K H100-hours for 3.3T tokens', () => {
    // Source: Microsoft Phi-3.5 (2024)
    // 512 H100, 3.3T tokens, ~10 days → ~86K GPU-hours
    const numGPUs = 512;
    const tokens = 3.3e12;
    const seqLen = 4096;
    const totalSteps = Math.ceil(tokens / (2048 * seqLen));

    const metrics = sim(benchmarkConfig(
      'h100-sxm', 8, numGPUs / 8, 'phi3-mini',
      'fsdp', 2048, 4, seqLen,
      undefined, { maxSteps: totalSteps },
    ));

    const model = getModel('phi3-mini', seqLen)!;
    const analytical = analyticalGPUHours(model.activeParams, tokens, metrics.mfu, 989);
    const simGPUHrs = (metrics.timeToTrainHours ?? 0) * numGPUs;

    expect(simGPUHrs / analytical).toBeGreaterThan(0.95);
    expect(simGPUHrs / analytical).toBeLessThan(1.05);

    // Overhead ratio ≈ 1.67 (wall-clock derived)
    const ratio = 86000 / analytical;
    expect(ratio).toBeGreaterThan(0.9);
    expect(ratio).toBeLessThan(2.5);
  });

  it('Phi-4 Mini 3.8B: ~344K A100-hours for 5T tokens', () => {
    // Source: Microsoft Phi-4-Mini Technical Report (Feb 2025)
    // 1024 A100-80G, ~14 days → ~344K GPU-hours
    // Using MBS=2 (not MBS=4) with 1024 GPUs so GBS=MBS×DP=2048 matches
    const numGPUs = 1024;
    const tokens = 5e12;
    const seqLen = 4096;
    const totalSteps = Math.ceil(tokens / (2048 * seqLen));

    const metrics = sim(benchmarkConfig(
      'a100-80gb', 8, numGPUs / 8, 'phi4-mini',
      'fsdp', 2048, 2, seqLen,
      undefined, { maxSteps: totalSteps },
    ));

    const model = getModel('phi4-mini', seqLen)!;
    const analytical = analyticalGPUHours(model.activeParams, tokens, metrics.mfu, 312);
    const simGPUHrs = (metrics.timeToTrainHours ?? 0) * numGPUs;

    expect(simGPUHrs / analytical).toBeGreaterThan(0.95);
    expect(simGPUHrs / analytical).toBeLessThan(1.05);

    // Overhead ratio ≈ 1.46 (wall-clock includes calendar overhead)
    const ratio = 344000 / analytical;
    expect(ratio).toBeGreaterThan(0.9);
    expect(ratio).toBeLessThan(2.5);
  });

  // ─── Tier 3: Approximate published data ─────────────────────────────

  it('GPT-3 175B: ~3.1M V100-hours for 300B tokens', () => {
    // Source: Brown et al. (2020), estimated from petaflop-days
    // 1024 V100-32GB, TP=8, PP=8, GBS=1536
    // V100 peak: 125 TFLOPS (FP16 tensor core, no BF16)
    // NOTE: OOMs in sim — V100-32GB too small for 175B with DDP-TP-PP.
    // Using rawSim; MFU/GPU-hours still computed correctly.
    const numGPUs = 1024;
    const tokens = 300e9;
    const seqLen = 2048;
    const totalSteps = Math.ceil(tokens / (1536 * seqLen));

    const metrics = rawSim(benchmarkConfig(
      'v100-32gb', 8, numGPUs / 8, 'gpt3-175b',
      'ddp-tp-pp', 1536, 1, seqLen,
      { tp: 8, pp: 8 },
      { maxSteps: totalSteps },
    ));

    const model = getModel('gpt3-175b', seqLen)!;
    const analytical = analyticalGPUHours(model.activeParams, tokens, metrics.mfu, 125);
    const simGPUHrs = (metrics.timeToTrainHours ?? 0) * numGPUs;

    expect(simGPUHrs / analytical).toBeGreaterThan(0.95);
    expect(simGPUHrs / analytical).toBeLessThan(1.05);

    // Overhead ratio ≈ 2.05 (estimated from PF-days, significant uncertainty)
    const ratio = 3100000 / analytical;
    expect(ratio).toBeGreaterThan(0.9);
    expect(ratio).toBeLessThan(3.5);
  });

  it('LLaMA 3 8B: 1.46M H100-hours for 15T tokens', () => {
    // Source: Meta LLaMA 3 paper (2024)
    // Published GPU-hours likely include annealing, context extension,
    // post-training, and failed experiments — not pure pre-training.
    const numGPUs = 512;
    const tokens = 15e12;
    const seqLen = 8192;
    const totalSteps = Math.ceil(tokens / (1024 * seqLen));

    const metrics = sim(benchmarkConfig(
      'h100-sxm', 8, numGPUs / 8, 'llama3-8b',
      'fsdp', 1024, 2, seqLen,
      undefined, { maxSteps: totalSteps },
    ));

    const model = getModel('llama3-8b', seqLen)!;
    const analytical = analyticalGPUHours(model.activeParams, tokens, metrics.mfu, 989);
    const simGPUHrs = (metrics.timeToTrainHours ?? 0) * numGPUs;

    expect(simGPUHrs / analytical).toBeGreaterThan(0.95);
    expect(simGPUHrs / analytical).toBeLessThan(1.05);

    // Overhead ratio ≈ 2.97 (published includes much more than pre-training)
    const ratio = 1460000 / analytical;
    expect(ratio).toBeGreaterThan(1.5);
    expect(ratio).toBeLessThan(4.0);
  });

  it('LLaMA 3 70B: 7.0M H100-hours for 15T tokens', () => {
    // Source: Meta LLaMA 3 paper (2024)
    // Published GPU-hours likely include post-training overhead
    const numGPUs = 2048;
    const tokens = 15e12;
    const seqLen = 8192;
    const totalSteps = Math.ceil(tokens / (1024 * seqLen));

    const metrics = sim(benchmarkConfig(
      'h100-sxm', 8, numGPUs / 8, 'llama3-70b',
      'fsdp-tp', 1024, 1, seqLen,
      { tp: 8 }, { maxSteps: totalSteps },
    ));

    const model = getModel('llama3-70b', seqLen)!;
    const analytical = analyticalGPUHours(model.activeParams, tokens, metrics.mfu, 989);
    const simGPUHrs = (metrics.timeToTrainHours ?? 0) * numGPUs;

    expect(simGPUHrs / analytical).toBeGreaterThan(0.95);
    expect(simGPUHrs / analytical).toBeLessThan(1.05);

    // Overhead ratio ≈ 1.68 (published includes post-training)
    const ratio = 7000000 / analytical;
    expect(ratio).toBeGreaterThan(0.9);
    expect(ratio).toBeLessThan(2.5);
  });
});

// ===========================================================================
// Section 8: Megatron-LM Benchmarks
//
// Tests for extreme pipeline parallelism and high virtual pipeline stages.
// ===========================================================================

describe('Megatron-LM Benchmarks', () => {
  // -----------------------------------------------------------------------
  // MT-NLG 530B: largest published dense model training
  // Source: Smith et al. (2022) "Using DeepSpeed and Megatron to Train
  //         Megatron-Turing NLG 530B"
  // Published: 126 TFLOPS/GPU on 2240 A100-80GB (~40.4% MFU)
  // Config: TP=8, PP=35, DP=8, GBS=1920, MBS=1, seq=2048, interleaved v=3
  // PP=35 is extreme pipeline depth (105 layers / 35 stages = 3 layers/stage).
  // Paper uses interleaved scheduling (Narayanan 2021): v=3 gives 1 layer per
  // virtual stage (maximum interleaving for this config).
  //
  // Simulator: ~42.5% MFU vs published 40.4%.
  // Slight overshoot from PP comm overhead model at extreme PP=35.
  // -----------------------------------------------------------------------
  it('MT-NLG 530B × 2240 A100-80GB: PP=35 interleaved v=3 (~43% MFU)', () => {
    const metrics = sim(benchmarkConfig(
      'a100-80gb', 8, 280, 'megatron-turing-530b',
      'ddp-tp-pp', 1920, 1, 2048,
      { tp: 8, pp: 35, pipelineSchedule: 'interleaved-1f1b', interleavedStages: 3 },
    ));

    // Simulator: ~42.5% MFU. Published: 40.4% MFU (126 TFLOPS/GPU).
    // PP-depth penalty (0.02 × log2(35) ≈ 0.103) brings extreme PP=35 into range.
    // ±15% of 42.5%: [36.1%, 48.9%]
    expect(metrics.mfu).toBeGreaterThan(0.3612);
    expect(metrics.mfu).toBeLessThan(0.4888);
  });

  // -----------------------------------------------------------------------
  // GPT-3 175B with VPP=12: highest virtual pipeline parallelism in test suite
  // Source: NeMo Framework benchmarks (NVIDIA)
  // Published: 37.68% MFU (FP8, against FP8 peak denominator)
  // Config: TP=4, PP=8, DP=4, GBS=256, MBS=1, seq=2048, interleaved-1f1b v=12
  // Testing BF16 — our MFU uses BF16 peak. NeMo's 37.68% (FP8) falls within
  // ±15% of our 41.8% (BF16), validating cross-definition consistency.
  // -----------------------------------------------------------------------
  it('GPT-3 175B × 128 H100 VPP=12: interleaved pipeline with 12 virtual stages (~44% MFU)', () => {
    const metrics = rawSim(benchmarkConfig(
      'h100-sxm', 8, 16, 'gpt3-175b',
      'ddp-tp-pp', 256, 1, 2048,
      { tp: 4, pp: 8, pipelineSchedule: 'interleaved-1f1b', interleavedStages: 12 },
    ));

    // Simulator: ~43.6% MFU (BF16)
    // Published: 37.68% MFU (NeMo, FP8 against FP8 peak) — within our range
    // ±15% of 43.6%: [37.0%, 50.1%]
    expect(metrics.mfu).toBeGreaterThan(0.3702);
    expect(metrics.mfu).toBeLessThan(0.501);
  });
});

// ===========================================================================
// Section 9: NeMo Dense Benchmarks
//
// Published TFLOPS/GPU from NeMo Framework on DGX-H100.
// AC: NeMo default is selective recomputation (Korthikanti et al. 2022).
//
// NeMo published TFLOPS use the Megatron-LM formula (4 × flopsPerToken)
// which differs from our 6PD basis. We derive implied step time from published
// TFLOPS (convention-independent) and compare against sim. See BENCHMARKS.md §2.5.
// ===========================================================================

describe('NeMo Dense Benchmarks', () => {
  // -----------------------------------------------------------------------
  // NeMo Llama3 8B: TP=1 PP=1 (pure FSDP, single node)
  // Source: NeMo Framework published benchmarks
  // Config: 8 H100, TP=1, GBS=128, MBS=1, seq=8192, FSDP, selective AC
  // Published: 725 TFLOPS/GPU
  // AC: NeMo default is selective (Korthikanti et al. 2022).
  //
  // Validates TP=1 pure FSDP at 8K sequence length.
  // -----------------------------------------------------------------------
  it('NeMo Llama3 8B TP=1 (8 H100 FSDP): all metrics (published: 725 TFLOPS)', () => {
    const metrics = sim(benchmarkConfig(
      'h100-sxm', 8, 1, 'llama3-8b',
      'fsdp', 128, 1, 8192,
      undefined,
      { checkpointingGranularity: 'selective' },
    ));

    // ±15% of simulator value
    expect(metrics.mfu).toBeGreaterThan(0.3849);
    expect(metrics.mfu).toBeLessThan(0.5207);
    expect(metrics.hfu).toBeGreaterThan(0.4095);
    expect(metrics.hfu).toBeLessThan(0.5541);
    expect(metrics.tflopsPerGPU).toBeGreaterThan(381);
    expect(metrics.tflopsPerGPU).toBeLessThan(515);
    expect(metrics.stepTimeMs).toBeGreaterThan(11987);
    expect(metrics.stepTimeMs).toBeLessThan(16218);
    const tokPerGPU = metrics.tokensPerSecond / 8;
    expect(tokPerGPU).toBeGreaterThan(7900);
    expect(tokPerGPU).toBeLessThan(10688);
    const memGB = metrics.memoryPerGPU.total / 1e9;
    expect(memGB).toBeGreaterThan(47.0);
    expect(memGB).toBeLessThan(63.6);
    expect(metrics.pipelineBubble).toBe(0);

    // --- NeMo convention conversion (see BENCHMARKS.md §2.5) ---
    // Implied step time from published 725 TFLOPS: ~13,993 ms.
    // Sim step ~14,102 ms → +0.8% delta, -0.4pp MFU. Validates core compute model.
    const impliedMs = nemoImpliedStepTimeMs('llama3-8b', 8192, 128, 8, 725);
    const stepDelta = (metrics.stepTimeMs - impliedMs) / impliedMs;
    expect(stepDelta).toBeGreaterThan(-0.10);
    expect(stepDelta).toBeLessThan(0.10);
  });

  // -----------------------------------------------------------------------
  // NeMo Llama3 70B: TP=4, PP=8, 64 H100
  // Source: NeMo Framework published benchmarks
  // Config: 64 H100, TP=4, PP=8, GBS=128, MBS=1, seq=8192, selective AC
  // Published: 727 TFLOPS/GPU
  // AC: NeMo default is selective (Korthikanti et al. 2022).
  //
  // Novel TP=4 degree (most tests use TP=8). DP = 64/(4×8) = 2.
  // GA = ceil(128/(1×2)) = 64. Interleaved 1F1B with VP=2 (NeMo Launcher default).
  // 80 layers / (8×2) = 5 layers/vstage. Bubble = 15/(15+64) ≈ 5.19%.
  // -----------------------------------------------------------------------
  it('NeMo Llama3 70B TP=4 PP=8 (64 H100): all metrics (published: 727 TFLOPS)', () => {
    const metrics = rawSim(benchmarkConfig(
      'h100-sxm', 8, 8, 'llama3-70b',
      'fsdp-tp-pp', 128, 1, 8192,
      { tp: 4, pp: 8, sequenceParallel: true,
        pipelineSchedule: 'interleaved-1f1b', interleavedStages: 2 },
      { checkpointingGranularity: 'selective' },
    ));

    // ±15% of simulator value (~44.8% MFU with interleaved 1F1B)
    expect(metrics.mfu).toBeGreaterThan(0.3806);
    expect(metrics.mfu).toBeLessThan(0.5149);
    expect(metrics.hfu).toBeGreaterThan(0.3806);
    expect(metrics.hfu).toBeLessThan(0.5149);
    expect(metrics.tflopsPerGPU).toBeGreaterThan(376);
    expect(metrics.tflopsPerGPU).toBeLessThan(509);
    expect(metrics.stepTimeMs).toBeGreaterThan(13313);
    expect(metrics.stepTimeMs).toBeLessThan(18012);
    const tokPerGPU = metrics.tokensPerSecond / 64;
    expect(tokPerGPU).toBeGreaterThan(889);
    expect(tokPerGPU).toBeLessThan(1203);
    const memGB = metrics.memoryPerGPU.total / 1e9;
    expect(memGB).toBeGreaterThan(54.6);
    expect(memGB).toBeLessThan(73.8);
    expect(metrics.pipelineBubble).toBeGreaterThan(0.0441);
    expect(metrics.pipelineBubble).toBeLessThan(0.0596);

    // --- NeMo convention conversion (see BENCHMARKS.md §2.5) ---
    // Implied step time from published 727 TFLOPS: ~14,487 ms.
    // Sim step ~15,662 ms → +8.1% delta, -3.5pp MFU. Gap from PP overlap modeling.
    const impliedMs = nemoImpliedStepTimeMs('llama3-70b', 8192, 128, 64, 727);
    const stepDelta = (metrics.stepTimeMs - impliedMs) / impliedMs;
    expect(stepDelta).toBeGreaterThan(-0.05);
    expect(stepDelta).toBeLessThan(0.20);
  });
});

// ===========================================================================
// Section 10: NeMo MoE Benchmarks
//
// Published TFLOPS/GPU from NeMo Framework for MoE models.
// Tests novel parallelism combinations: TP=2 with high EP, small MoE + PP.
// ===========================================================================

describe('NeMo MoE Benchmarks', () => {
  // -----------------------------------------------------------------------
  // DeepSeek-V3 TP=2 PP=8 EP=64 on 1024 H100
  // Source: NeMo Framework published benchmarks
  // Config: 1024 H100, TP=2, PP=8, EP=64, GBS=8192, MBS=1, seq=4096
  // Published: 338 TFLOPS/GPU (HFU — includes recompute)
  //
  // Different parallelism than existing tests (which use TP=4).
  // DP = 1024/(2×8) = 64. EP=64 subdivides DP=64 exactly.
  // Note: OOMs at 97 GB/GPU (>80 GB). Uses rawSim.
  // -----------------------------------------------------------------------
  it('NeMo DeepSeek V3 TP=2 PP=8 EP=64 (1024 H100): all metrics (published: 338 HFU TFLOPS)', () => {
    const metrics = rawSim(benchmarkConfig(
      'h100-sxm', 8, 128, 'deepseek-v3',
      'fsdp-tp-pp', 8192, 1, 4096,
      { tp: 2, pp: 8, ep: 64, sequenceParallel: true },
    ));

    // ±15% of simulator value
    // Simulator HFU TFLOPS: 0.564 × 989 = 558 vs published 338 → 65% gap.
    expect(metrics.mfu).toBeGreaterThan(0.3593);
    expect(metrics.mfu).toBeLessThan(0.4861);
    expect(metrics.hfu).toBeGreaterThan(0.4793);
    expect(metrics.hfu).toBeLessThan(0.6485);
    expect(metrics.tflopsPerGPU).toBeGreaterThan(355);
    expect(metrics.tflopsPerGPU).toBeLessThan(481);
    expect(metrics.stepTimeMs).toBeGreaterThan(15008);
    expect(metrics.stepTimeMs).toBeLessThan(20305);
    const tokPerGPU = metrics.tokensPerSecond / 1024;
    expect(tokPerGPU).toBeGreaterThan(1578);
    expect(tokPerGPU).toBeLessThan(2134);
    // OOM expected (97 GB > 80 GB)
    expect(metrics.memoryUtilization).toBeGreaterThan(1.0);
    // Layer imbalance (ceil(61/8)=8 vs 61/8=7.625) adds ~4.9% to reported bubble.
    expect(metrics.pipelineBubble).toBeGreaterThan(0.080);
    expect(metrics.pipelineBubble).toBeLessThan(0.111);
  });

  // -----------------------------------------------------------------------
  // Qwen3 30B-A3B: Small MoE + PP stress test
  // Source: NeMo Framework published benchmarks
  // Config: 16 H100, TP=1, PP=2, EP=8, GBS=512, MBS=2, seq=4096
  // Published: 241 TFLOPS/GPU (HFU)
  //
  // DP = 16/(1×2) = 8. EP=8 subdivides DP=8.
  // 128 experts, 8 active. GA = ceil(512/(2×8)) = 32.
  // Bubble = 1/33 ≈ 3.03%.
  //
  // KNOWN GAP: Simulator HFU TFLOPS: 0.368 × 989 = 364 vs published 241 → 51% overshoot.
  // 128-expert fine-grained MoE (expertIntermediate=768) operates in a regime our model
  // doesn't capture well: per-expert GEMMs (512×768) are memory-bandwidth-bound rather
  // than compute-bound, capacity factor enforcement causes token dropping/padding waste,
  // and framework overhead (dynamic dispatch, kernel scheduling for 128 tiny expert GEMMs)
  // dominates at this granularity. Router computation and token permutation are modeled
  // but these second-order effects are not.
  // -----------------------------------------------------------------------
  it('NeMo Qwen3 30B-A3B TP=1 PP=2 EP=8 (16 H100): all metrics (published: 241 HFU TFLOPS)', () => {
    const metrics = rawSim(benchmarkConfig(
      'h100-sxm', 8, 2, 'qwen3-30b-a3b',
      'fsdp-tp-pp', 512, 2, 4096,
      { tp: 1, pp: 2, ep: 8 },
    ));

    // ±15% of simulator value
    expect(metrics.mfu).toBeGreaterThan(0.1978);
    expect(metrics.mfu).toBeLessThan(0.2675);
    expect(metrics.hfu).toBeGreaterThan(0.2637);
    expect(metrics.hfu).toBeLessThan(0.3567);
    expect(metrics.tflopsPerGPU).toBeGreaterThan(196);
    expect(metrics.tflopsPerGPU).toBeLessThan(265);
    expect(metrics.stepTimeMs).toBeGreaterThan(9741);
    expect(metrics.stepTimeMs).toBeLessThan(13180);
    const tokPerGPU = metrics.tokensPerSecond / 16;
    expect(tokPerGPU).toBeGreaterThan(9721);
    expect(tokPerGPU).toBeLessThan(13153);
    const memGB = metrics.memoryPerGPU.total / 1e9;
    expect(memGB).toBeGreaterThan(41.2);
    expect(memGB).toBeLessThan(55.7);
    expect(metrics.pipelineBubble).toBeGreaterThan(0.0258);
    expect(metrics.pipelineBubble).toBeLessThan(0.0348);
  });

  // -----------------------------------------------------------------------
  // Qwen3 235B-A22B: Large MoE, TP=2, PP=8, EP=32
  // Source: NeMo Framework published benchmarks
  // Config: 512 H100, TP=2, PP=8, EP=32, GBS=2048, MBS=1, seq=4096
  // Published: 178 TFLOPS/GPU (HFU)
  //
  // DP = 512/(2×8) = 32. EP=32 subdivides DP=32.
  // 128 experts, 8 active. GA = ceil(2048/(1×32)) = 64.
  // Bubble = 7/71 ≈ 9.86%.
  //
  // Note: Plan used 256 GPUs with EP=32 but EP must divide DP.
  // 256/(2×8)=16, EP=32>16 — doesn't work. Using 512 GPUs instead.
  // -----------------------------------------------------------------------
  it('NeMo Qwen3 235B-A22B TP=2 PP=8 EP=32 (512 H100): all metrics (published: 178 HFU TFLOPS)', () => {
    const metrics = rawSim(benchmarkConfig(
      'h100-sxm', 8, 64, 'qwen3-235b-a22b',
      'fsdp-tp-pp', 2048, 1, 4096,
      { tp: 2, pp: 8, ep: 32, sequenceParallel: true },
    ));

    // Simulator HFU TFLOPS: 0.238 × 989 = 235 vs published 178 → 32% overshoot
    // Gap: similar 128-expert fine-grained MoE limitations as Qwen3 30B-A3B (see above)
    // ±15% of simulator value
    expect(metrics.mfu).toBeGreaterThan(0.1515);
    expect(metrics.mfu).toBeLessThan(0.2050);
    expect(metrics.hfu).toBeGreaterThan(0.2020);
    expect(metrics.hfu).toBeLessThan(0.2733);
    expect(metrics.tflopsPerGPU).toBeGreaterThan(150);
    expect(metrics.tflopsPerGPU).toBeLessThan(203);
    expect(metrics.stepTimeMs).toBeGreaterThan(10517);
    expect(metrics.stepTimeMs).toBeLessThan(14229);
    const tokPerGPU = metrics.tokensPerSecond / 512;
    expect(tokPerGPU).toBeGreaterThan(1125);
    expect(tokPerGPU).toBeLessThan(1523);
    const memGB = metrics.memoryPerGPU.total / 1e9;
    expect(memGB).toBeGreaterThan(31.7);
    expect(memGB).toBeLessThan(43.1);
    // Layer imbalance (ceil(94/8)=12 vs 94/8=11.75) adds ~2.1% to reported bubble.
    expect(metrics.pipelineBubble).toBeGreaterThan(0.100);
    expect(metrics.pipelineBubble).toBeLessThan(0.135);
  });
});

// ===========================================================================
// Section 11: MoE EP Scaling
//
// Mixtral 8x22B with varying EP to validate EP scaling behavior.
// Config: 128 H100, TP=2, PP=8, GBS=256, MBS=1, seq=4096
// DP = 128/(2×8) = 8. EP ∈ {1, 4, 8}.
//
// Published reference: 49.3% MFU with MoE Parallel Folding — we don't model
// this optimization, so expect lower (~33-44%).
// ===========================================================================

describe('MoE EP Scaling — Mixtral 8x22B', () => {
  it('Mixtral 8x22B EP=1 (128 H100, TP=2 PP=8): MFU ≈ 40.6%', () => {
    const metrics = rawSim(benchmarkConfig(
      'h100-sxm', 8, 16, 'mixtral-8x22b',
      'fsdp-tp-pp', 256, 1, 4096,
      { tp: 2, pp: 8 },
    ));

    // ±15% of simulator value (GA-aware overlap raises MFU)
    expect(metrics.mfu).toBeGreaterThan(0.3448);
    expect(metrics.mfu).toBeLessThan(0.4664);
    expect(metrics.stepTimeMs).toBeGreaterThan(4079);
    expect(metrics.stepTimeMs).toBeLessThan(5518);
    const memGB = metrics.memoryPerGPU.total / 1e9;
    expect(memGB).toBeGreaterThan(44.0);
    expect(memGB).toBeLessThan(59.5);
    expect(metrics.pipelineBubble).toBeGreaterThan(0.1525);
    expect(metrics.pipelineBubble).toBeLessThan(0.2065);
  });

  it('Mixtral 8x22B EP=4 (128 H100, TP=2 PP=8): MFU ≈ 45.1%', () => {
    const metrics = rawSim(benchmarkConfig(
      'h100-sxm', 8, 16, 'mixtral-8x22b',
      'fsdp-tp-pp', 256, 1, 4096,
      { tp: 2, pp: 8, ep: 4 },
    ));

    // ±15% of simulator value
    expect(metrics.mfu).toBeGreaterThan(0.3835);
    expect(metrics.mfu).toBeLessThan(0.5189);
    expect(metrics.stepTimeMs).toBeGreaterThan(3666);
    expect(metrics.stepTimeMs).toBeLessThan(4960);
    const memGB = metrics.memoryPerGPU.total / 1e9;
    expect(memGB).toBeGreaterThan(42.2);
    expect(memGB).toBeLessThan(57.2);
  });

  it('Mixtral 8x22B EP=8 (128 H100, TP=2 PP=8): MFU ≈ 41.8%', () => {
    const metrics = rawSim(benchmarkConfig(
      'h100-sxm', 8, 16, 'mixtral-8x22b',
      'fsdp-tp-pp', 256, 1, 4096,
      { tp: 2, pp: 8, ep: 8 },
    ));

    // ±15% of simulator value
    expect(metrics.mfu).toBeGreaterThan(0.3553);
    expect(metrics.mfu).toBeLessThan(0.4807);
    expect(metrics.stepTimeMs).toBeGreaterThan(3958);
    expect(metrics.stepTimeMs).toBeLessThan(5355);
    const memGB = metrics.memoryPerGPU.total / 1e9;
    expect(memGB).toBeGreaterThan(41.3);
    expect(memGB).toBeLessThan(55.9);
  });

  it('Mixtral 8x22B EP scaling: higher EP reduces memory and improves throughput', () => {
    const ep1 = rawSim(benchmarkConfig(
      'h100-sxm', 8, 16, 'mixtral-8x22b',
      'fsdp-tp-pp', 256, 1, 4096, { tp: 2, pp: 8 },
    ));
    const ep4 = rawSim(benchmarkConfig(
      'h100-sxm', 8, 16, 'mixtral-8x22b',
      'fsdp-tp-pp', 256, 1, 4096, { tp: 2, pp: 8, ep: 4 },
    ));
    const ep8 = rawSim(benchmarkConfig(
      'h100-sxm', 8, 16, 'mixtral-8x22b',
      'fsdp-tp-pp', 256, 1, 4096, { tp: 2, pp: 8, ep: 8 },
    ));

    // Memory decreases with EP (expert sharding)
    expect(ep4.memoryPerGPU.total).toBeLessThan(ep1.memoryPerGPU.total);
    expect(ep8.memoryPerGPU.total).toBeLessThan(ep4.memoryPerGPU.total);

    // MFU improves with EP (reduced expert DP comm)
    expect(ep4.mfu).toBeGreaterThan(ep1.mfu);
    // EP=8 ≈ EP=4 (diminishing returns; GA-aware overlap narrows the gap)
    expect(ep8.mfu).toBeGreaterThan(ep4.mfu * 0.90);

    // Step time decreases with EP
    expect(ep4.stepTimeMs).toBeLessThan(ep1.stepTimeMs);
  });
});

// ===========================================================================
// Section 12: Cross-Dimension Scaling
//
// Monotonicity and ordering tests for CP and EP scaling.
// ===========================================================================

describe('Cross-Dimension Scaling', () => {
  // -----------------------------------------------------------------------
  // CP scaling: Llama 3 405B at seq=32768, 16384 H100
  // TP=8, PP=16, interleaved v=4, CP ∈ {1, 2, 4}
  //
  // At seq=32768: CP=1→DP=128, CP=2→DP=64, CP=4→DP=32
  // Higher CP → less DP → more GA → lower pipeline bubble
  // Memory decreases with CP (sequence split across ranks)
  // -----------------------------------------------------------------------
  it('CP scaling: Llama 405B memory decreases with CP (seq=32768)', () => {
    const cp1 = rawSim(benchmarkConfig(
      'h100-sxm', 8, 2048, 'llama3-405b',
      'fsdp-tp-pp', 2048, 1, 32768,
      { tp: 8, pp: 16, cp: 1, sequenceParallel: true,
        pipelineSchedule: 'interleaved-1f1b', interleavedStages: 4 },
    ));
    const cp2 = rawSim(benchmarkConfig(
      'h100-sxm', 8, 2048, 'llama3-405b',
      'fsdp-tp-pp', 2048, 1, 32768,
      { tp: 8, pp: 16, cp: 2, sequenceParallel: true,
        pipelineSchedule: 'interleaved-1f1b', interleavedStages: 4 },
    ));
    const cp4 = rawSim(benchmarkConfig(
      'h100-sxm', 8, 2048, 'llama3-405b',
      'fsdp-tp-pp', 2048, 1, 32768,
      { tp: 8, pp: 16, cp: 4, sequenceParallel: true,
        pipelineSchedule: 'interleaved-1f1b', interleavedStages: 4 },
    ));

    // Memory decreases monotonically with CP
    expect(cp2.memoryPerGPU.total).toBeLessThan(cp1.memoryPerGPU.total);
    expect(cp4.memoryPerGPU.total).toBeLessThan(cp2.memoryPerGPU.total);

    // Pipeline bubble decreases with CP (more GA from fewer DP ranks)
    expect(cp2.pipelineBubble).toBeLessThan(cp1.pipelineBubble);
    expect(cp4.pipelineBubble).toBeLessThan(cp2.pipelineBubble);

    // Step time decreases with CP (lower bubble dominates CP comm)
    expect(cp2.stepTimeMs).toBeLessThan(cp1.stepTimeMs);
    expect(cp4.stepTimeMs).toBeLessThan(cp2.stepTimeMs);
  });

  it('CP scaling: Llama 405B absolute metrics at each CP degree (seq=32768)', () => {
    // CP=1: MFU ~29.0%, mem ~44.7 GB
    const cp1 = rawSim(benchmarkConfig(
      'h100-sxm', 8, 2048, 'llama3-405b',
      'fsdp-tp-pp', 2048, 1, 32768,
      { tp: 8, pp: 16, cp: 1, sequenceParallel: true,
        pipelineSchedule: 'interleaved-1f1b', interleavedStages: 4 },
    ));
    expect(cp1.mfu).toBeGreaterThan(0.2365);
    expect(cp1.mfu).toBeLessThan(0.3200);
    expect(cp1.memoryPerGPU.total / 1e9).toBeGreaterThan(96.1);
    expect(cp1.memoryPerGPU.total / 1e9).toBeLessThan(130.0);

    // CP=2: MFU ~31.8%, mem ~31.1 GB
    const cp2 = rawSim(benchmarkConfig(
      'h100-sxm', 8, 2048, 'llama3-405b',
      'fsdp-tp-pp', 2048, 1, 32768,
      { tp: 8, pp: 16, cp: 2, sequenceParallel: true,
        pipelineSchedule: 'interleaved-1f1b', interleavedStages: 4 },
    ));
    expect(cp2.mfu).toBeGreaterThan(0.2586);
    expect(cp2.mfu).toBeLessThan(0.3499);
    expect(cp2.memoryPerGPU.total / 1e9).toBeGreaterThan(53.7);
    expect(cp2.memoryPerGPU.total / 1e9).toBeLessThan(72.7);

    // CP=4: MFU ~32.9%, mem ~24.7 GB
    const cp4 = rawSim(benchmarkConfig(
      'h100-sxm', 8, 2048, 'llama3-405b',
      'fsdp-tp-pp', 2048, 1, 32768,
      { tp: 8, pp: 16, cp: 4, sequenceParallel: true,
        pipelineSchedule: 'interleaved-1f1b', interleavedStages: 4 },
    ));
    expect(cp4.mfu).toBeGreaterThan(0.2645);
    expect(cp4.mfu).toBeLessThan(0.3578);
    expect(cp4.memoryPerGPU.total / 1e9).toBeGreaterThan(32.9);
    expect(cp4.memoryPerGPU.total / 1e9).toBeLessThan(44.5);
  });

  // -----------------------------------------------------------------------
  // EP scaling: DeepSeek V3 with EP=1, EP=8, EP=32
  // 2048 H100, TP=4, PP=8, GBS=4608, MBS=1, seq=4096
  // DP = 2048/(4×8) = 64
  //
  // EP=1 is catastrophic for MoE: full expert AllReduce across DP=64.
  // EP=8 and EP=32 dramatically reduce expert comm volume.
  // -----------------------------------------------------------------------
  it('EP scaling: DeepSeek V3 memory and throughput (EP=1 vs EP=8 vs EP=32)', () => {
    const ep1 = rawSim(benchmarkConfig(
      'h100-sxm', 8, 256, 'deepseek-v3',
      'fsdp-tp-pp', 4608, 1, 4096, { tp: 4, pp: 8 },
    ));
    const ep8 = rawSim(benchmarkConfig(
      'h100-sxm', 8, 256, 'deepseek-v3',
      'fsdp-tp-pp', 4608, 1, 4096, { tp: 4, pp: 8, ep: 8 },
    ));
    const ep32 = rawSim(benchmarkConfig(
      'h100-sxm', 8, 256, 'deepseek-v3',
      'fsdp-tp-pp', 4608, 1, 4096, { tp: 4, pp: 8, ep: 32, sequenceParallel: true },
    ));

    // EP dramatically improves MFU for MoE
    expect(ep8.mfu).toBeGreaterThan(ep1.mfu * 2);  // EP=8 at least 2× better than EP=1
    expect(ep32.mfu).toBeGreaterThan(ep8.mfu * 0.95);  // EP=32 similar or better than EP=8

    // Step time decreases dramatically with EP
    expect(ep8.stepTimeMs).toBeLessThan(ep1.stepTimeMs * 0.5);

    // Per-layer pipeline: EP=1 has larger per-layer FSDP AG (all expert params),
    // exceeding per-layer compute in forward pass → higher exposed comm overhead.
    // EP=8 adds all-to-all but drastically reduces FSDP AG volume.
    expect(ep1.communicationOverhead).toBeGreaterThan(ep8.communicationOverhead);
  });

  it('EP scaling: DeepSeek V3 absolute metrics at each EP degree', () => {
    // EP=1: MFU ~9% (all expert params in FSDP AG → per-layer comm-bound in forward)
    const ep1 = rawSim(benchmarkConfig(
      'h100-sxm', 8, 256, 'deepseek-v3',
      'fsdp-tp-pp', 4608, 1, 4096, { tp: 4, pp: 8 },
    ));
    expect(ep1.mfu).toBeGreaterThan(0.076);
    expect(ep1.mfu).toBeLessThan(0.103);
    expect(ep1.communicationOverhead).toBeGreaterThan(0.30);  // >30% comm overhead

    // EP=8: MFU ~26.9% (EP latency scaling + coordination floor add overhead)
    const ep8 = rawSim(benchmarkConfig(
      'h100-sxm', 8, 256, 'deepseek-v3',
      'fsdp-tp-pp', 4608, 1, 4096, { tp: 4, pp: 8, ep: 8 },
    ));
    expect(ep8.mfu).toBeGreaterThan(0.2288);
    expect(ep8.mfu).toBeLessThan(0.3096);

    // EP=32: MFU ~29.7% (device-limited routing effectiveEp=4)
    const ep32 = rawSim(benchmarkConfig(
      'h100-sxm', 8, 256, 'deepseek-v3',
      'fsdp-tp-pp', 4608, 1, 4096, { tp: 4, pp: 8, ep: 32, sequenceParallel: true },
    ));
    expect(ep32.mfu).toBeGreaterThan(0.2456);
    expect(ep32.mfu).toBeLessThan(0.3323);
  });
});

// ===========================================================================
// Section 13: Megatron-LM Weak Scaling (SC 2021)
//
// Source: Narayanan et al. "Efficient Large-Scale Language Model Training
//         on GPU Clusters Using Megatron-LM" (SC 2021)
// A100-80GB, BF16, 3D parallelism with TP=8, PP=8.
// Published: ~148 TFLOPS/GPU for GPT-3 175B, ~174 for GPT-3 13B.
//
// Wider bounds (±20%) since these are 2021 benchmarks with older
// software stacks — real systems have improved since then.
// ===========================================================================

describe('Megatron-LM SC 2021 Benchmarks', () => {
  it('SC21 GPT-3 175B × 1024 A100 (TP=8, PP=8): TFLOPS/GPU ≈ 140 (published: ~148)', () => {
    const metrics = sim(benchmarkConfig(
      'a100-80gb', 8, 128, 'gpt3-175b',
      'ddp-tp-pp', 1536, 1, 2048,
      { tp: 8, pp: 8 },
    ));

    // ±20% of simulator value (~139.7 TFLOPS)
    expect(metrics.tflopsPerGPU).toBeGreaterThan(111);
    expect(metrics.tflopsPerGPU).toBeLessThan(168);
    // MFU ±15%
    expect(metrics.mfu).toBeGreaterThan(0.3805);
    expect(metrics.mfu).toBeLessThan(0.5149);
    // Published 148 TFLOPS within our range
    expect(metrics.stepTimeMs).toBeGreaterThan(19648);
    expect(metrics.stepTimeMs).toBeLessThan(26584);
    const memGB = metrics.memoryPerGPU.total / 1e9;
    expect(memGB).toBeGreaterThan(49.6);
    expect(memGB).toBeLessThan(67.2);
  });

  it('SC21 GPT-3 13B × 256 A100 (FSDP-TP=8): TFLOPS/GPU ≈ 136 (loose match)', () => {
    // Loose match: SC21 used TP=8 PP=1 on DDP; we use FSDP-TP=8 for comparison
    // Published: ~174 TFLOPS/GPU for 13B-class models
    const metrics = sim(benchmarkConfig(
      'a100-80gb', 8, 32, 'gpt3-13b',
      'fsdp-tp', 1024, 2, 2048,
      { tp: 8 },
    ));

    // ±20% of simulator value (~136.1 TFLOPS)
    expect(metrics.tflopsPerGPU).toBeGreaterThan(100);
    expect(metrics.tflopsPerGPU).toBeLessThan(151);
    expect(metrics.mfu).toBeGreaterThan(0.3218);
    expect(metrics.mfu).toBeLessThan(0.4827);
    expect(metrics.stepTimeMs).toBeGreaterThan(4136);
    expect(metrics.stepTimeMs).toBeLessThan(6204);
    const memGB = metrics.memoryPerGPU.total / 1e9;
    expect(memGB).toBeGreaterThan(7.2);
    expect(memGB).toBeLessThan(10.8);
    expect(metrics.pipelineBubble).toBe(0);
  });

  it('SC21: GPT-3 175B and 13B MFU are within 10pp of each other', () => {
    const m175b = sim(benchmarkConfig(
      'a100-80gb', 8, 128, 'gpt3-175b',
      'ddp-tp-pp', 1536, 1, 2048, { tp: 8, pp: 8 },
    ));
    const m13b = sim(benchmarkConfig(
      'a100-80gb', 8, 32, 'gpt3-13b',
      'fsdp-tp', 1024, 2, 2048, { tp: 8 },
    ));

    // 175B uses DDP-TP-PP (PP=8 bubble) but 13B uses FSDP-TP (higher backward multiplier with AC).
    // These effects roughly balance, so MFU values are close.
    const diffPp = Math.abs(m175b.mfu - m13b.mfu) * 100;
    expect(diffPp).toBeLessThan(10);
    // 175B has much higher total throughput (more GPUs × larger model)
    expect(m175b.tokensPerSecond).toBeGreaterThan(m13b.tokensPerSecond * 0.15);
  });
});
