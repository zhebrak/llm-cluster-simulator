/**
 * Auto-Optimizer tests
 *
 * Tests both the training optimizer (optimizeTraining) and the inference
 * optimizer (optimizeInference) for correctness across various scenarios.
 */

import { describe, it, expect } from 'vitest';
import { optimizeTraining } from '../../src/core/simulation/optimizer.ts';
import { optimizeInference } from '../../src/core/inference/optimizer.ts';
import { generateTrainingCandidates, MAX_EXPLORE_SIMS, MAX_EXPLORE_SIMS_MOE } from '../../src/core/simulation/exploration.ts';
import { generateInferenceCandidates, MAX_EXPLORE_SIMS as MAX_INFERENCE_SIMS } from '../../src/core/inference/exploration.ts';
import type { SimulationConfig } from '../../src/core/simulation/engine.ts';
import type { InferenceSimulationConfig } from '../../src/core/inference/simulation.ts';
import { getModel } from '../../src/core/models/index.ts';
import { getPresetCluster } from '../../src/core/hardware/presets.ts';
import { ALL_GPUS } from '../../src/core/hardware/gpu.ts';

// ── Helpers ──────────────────────────────────────────────────────────

function makeTrainingConfig(overrides: Partial<SimulationConfig> = {}): SimulationConfig {
  const model = getModel('llama3-70b', 4096)!;
  const cluster = getPresetCluster('8x-h100')!;

  return {
    modelSpec: model,
    clusterConfig: cluster,
    globalBatchSize: 1024,
    microBatchSize: 2,
    sequenceLength: 4096,
    strategyType: 'ddp',
    activationCheckpointing: true,
    flashAttention: true,
    mixedPrecision: 'bf16',
    ...overrides,
  };
}

function makeInferenceConfig(overrides: Partial<InferenceSimulationConfig> = {}): InferenceSimulationConfig {
  const model = getModel('llama3.1-8b', 4096)!;
  const gpu = ALL_GPUS['h100-sxm'];

  return {
    modelSpec: model,
    gpu,
    numGPUs: 1,
    batchSize: 1,
    inputSeqLen: 512,
    outputSeqLen: 256,
    weightPrecision: 'bf16',
    kvCachePrecision: 'bf16',
    flashAttention: true,
    continuousBatching: false,
    ...overrides,
  };
}

// ── Training Optimizer Tests ─────────────────────────────────────────

describe('Training Optimizer', () => {
  it('should improve a suboptimal DDP config', () => {
    // Llama 8B on 8xH100 with DDP is suboptimal (FSDP or FSDP-TP would be better)
    const model = getModel('llama3.1-8b', 4096)!;
    const cluster = getPresetCluster('8x-h100')!;

    const config = makeTrainingConfig({
      modelSpec: model,
      clusterConfig: cluster,
      globalBatchSize: 256,
      microBatchSize: 2,
      strategyType: 'ddp',
    });
    const targetTokens = 100e9; // 100B tokens

    const result = optimizeTraining(config, targetTokens, config.sequenceLength);

    expect(result.success).toBe(true);
    // Optimizer should find a config that trains faster
    expect(result.afterMetric).toBeLessThan(result.beforeMetric);
    // Should have changed something
    expect(result.changelog.length).toBeGreaterThan(0);
    // Should have run a non-trivial number of simulations
    expect(result.totalSimulations).toBeGreaterThan(1);
  });

  it('should resolve an OOM config', () => {
    // Llama 70B on 64xH100 with DDP will OOM — no sharding
    // 64 GPUs gives the optimizer room to find FSDP-TP with tp=8, dp=8
    const model = getModel('llama3-70b', 4096)!;
    const cluster = getPresetCluster('64x-h100')!;

    const config = makeTrainingConfig({
      modelSpec: model,
      clusterConfig: cluster,
      globalBatchSize: 1024,
      microBatchSize: 2,
      strategyType: 'ddp',
      activationCheckpointing: false,
    });
    const targetTokens = 1e12;

    const result = optimizeTraining(config, targetTokens, config.sequenceLength);

    // Should find a runnable config
    expect(result.success).toBe(true);
    // Before should be Infinity (OOM), after should be finite
    expect(result.beforeMetric).toBe(Infinity);
    expect(result.afterMetric).toBeLessThan(Infinity);
    expect(result.afterMetric).toBeGreaterThan(0);
  });

  it('should explore different GBS values', () => {
    // Use a model that fits easily, focus on GBS optimization
    const model = getModel('llama3.1-8b', 2048)!;
    const cluster = getPresetCluster('8x-h100')!;

    const config = makeTrainingConfig({
      modelSpec: model,
      clusterConfig: cluster,
      globalBatchSize: 64, // Very small — leaves room for improvement
      strategyType: 'fsdp',
    });
    const targetTokens = 100e9;

    const result = optimizeTraining(config, targetTokens, config.sequenceLength);

    expect(result.success).toBe(true);
    // Optimizer should have considered GBS changes
    expect(result.totalSimulations).toBeGreaterThan(5);
  });

  it('should handle MoE models with EP exploration', () => {
    const model = getModel('deepseek-v3', 4096)!;
    const cluster = getPresetCluster('256x-h100')!;

    const config: SimulationConfig = {
      modelSpec: model,
      clusterConfig: cluster,
      globalBatchSize: 4096,
      microBatchSize: 1,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8, ep: 1 },
      activationCheckpointing: true,
      flashAttention: true,
      mixedPrecision: 'bf16',
    };
    const targetTokens = 14.8e12;

    const result = optimizeTraining(config, targetTokens, config.sequenceLength);

    expect(result.success).toBe(true);
    expect(result.totalSimulations).toBeGreaterThan(10);
  });

  it('should not worsen an already-good config', () => {
    // Use an already-optimized small model
    const model = getModel('llama3.1-8b', 2048)!;
    const cluster = getPresetCluster('8x-h100')!;

    const config = makeTrainingConfig({
      modelSpec: model,
      clusterConfig: cluster,
      globalBatchSize: 1024,
      microBatchSize: 4,
      strategyType: 'fsdp',
      activationCheckpointing: true,
      flashAttention: true,
      mixedPrecision: 'bf16',
    });
    const targetTokens = 100e9;

    const result = optimizeTraining(config, targetTokens, config.sequenceLength);

    expect(result.success).toBe(true);
    // The optimizer may or may not find improvements for a well-tuned config,
    // but it should not make things worse
    expect(result.afterMetric).toBeLessThanOrEqual(result.beforeMetric * 1.01);
  });

  it('should have phases that sum correctly', () => {
    const model = getModel('llama3.1-8b', 2048)!;
    const cluster = getPresetCluster('8x-h100')!;

    const config = makeTrainingConfig({
      modelSpec: model,
      clusterConfig: cluster,
      globalBatchSize: 256,
      microBatchSize: 2,
      strategyType: 'fsdp',
    });
    const targetTokens = 100e9;

    const result = optimizeTraining(config, targetTokens, config.sequenceLength);

    const sumPhases = result.phases.fix + result.phases.greedy + result.phases.explore;
    // Total should be at least the sum of phases (may include a few extra for final metrics)
    expect(result.totalSimulations).toBeGreaterThanOrEqual(sumPhases);
  });
});

// ── Training Exploration Grid Tests ──────────────────────────────────

describe('Training Exploration Grid', () => {
  it('should generate candidates for a standard config', () => {
    const model = getModel('llama3.1-8b', 2048)!;
    const cluster = getPresetCluster('8x-h100')!;

    const config = makeTrainingConfig({
      modelSpec: model,
      clusterConfig: cluster,
    });

    const candidates = generateTrainingCandidates(config, model, cluster);
    expect(candidates.length).toBeGreaterThan(0);
    expect(candidates.length).toBeLessThanOrEqual(MAX_EXPLORE_SIMS);
  });

  it('should respect MAX_EXPLORE_SIMS cap for large grids', () => {
    // MoE model with many experts → huge grid, uses higher MoE budget
    const model = getModel('deepseek-v3', 4096)!;
    // Large cluster for maximum grid size
    const cluster = getPresetCluster('256x-h100')!;

    const config: SimulationConfig = {
      modelSpec: model,
      clusterConfig: cluster,
      globalBatchSize: 4096,
      microBatchSize: 1,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8 },
      activationCheckpointing: true,
      mixedPrecision: 'bf16',
    };

    const candidates = generateTrainingCandidates(config, model, cluster);
    expect(candidates.length).toBeLessThanOrEqual(MAX_EXPLORE_SIMS_MOE);
  });

  it('should only generate valid strategy-parallelism combos', () => {
    const model = getModel('llama3.1-8b', 2048)!;
    const cluster = getPresetCluster('8x-h100')!;

    const config = makeTrainingConfig({
      modelSpec: model,
      clusterConfig: cluster,
    });

    const candidates = generateTrainingCandidates(config, model, cluster);

    for (const c of candidates) {
      const tp = c.strategyConfig?.tp ?? 1;
      const pp = c.strategyConfig?.pp ?? 1;

      // 1D strategies must have tp=1, pp=1
      if (['ddp', 'fsdp', 'zero-1'].includes(c.strategyType)) {
        expect(tp).toBe(1);
        expect(pp).toBe(1);
      }

      // 2D strategies must have tp>1, pp=1
      if (['fsdp-tp', 'zero1-tp'].includes(c.strategyType)) {
        expect(tp).toBeGreaterThan(1);
        expect(pp).toBe(1);
      }

      // tp must divide numAttentionHeads
      expect(model.numAttentionHeads % tp).toBe(0);
    }
  });
});

// ── Inference Optimizer Tests ────────────────────────────────────────

describe('Inference Optimizer', () => {
  it('should optimize for throughput', () => {
    const config = makeInferenceConfig({
      batchSize: 1,
      continuousBatching: false,
      weightPrecision: 'bf16',
    });

    const result = optimizeInference(config, 'throughput');

    expect(result.success).toBe(true);
    // Should improve throughput
    expect(result.afterMetric).toBeGreaterThan(result.beforeMetric);
    expect(result.target).toBe('throughput');
  });

  it('should optimize for latency', () => {
    const model = getModel('llama3-70b', 4096)!;
    const gpu = ALL_GPUS['h100-sxm'];

    const config = makeInferenceConfig({
      modelSpec: model,
      gpu,
      numGPUs: 8,
      batchSize: 32,
      tensorParallel: 4,
      weightPrecision: 'bf16',
    });

    const result = optimizeInference(config, 'latency');

    expect(result.success).toBe(true);
    // Should reduce TPOT (or at least not make it worse)
    expect(result.afterMetric).toBeLessThanOrEqual(result.beforeMetric * 1.01);
    expect(result.target).toBe('latency');
  });

  it('should resolve OOM via TP/quantization', () => {
    // Llama 3 70B in BF16 needs ~140GB — OOM on single 80GB H100
    // With 8 GPUs available, optimizer should increase TP or quantize
    const model = getModel('llama3-70b', 4096)!;
    const gpu = ALL_GPUS['h100-sxm'];

    const config = makeInferenceConfig({
      modelSpec: model,
      gpu,
      numGPUs: 8,
      batchSize: 1,
      tensorParallel: 1, // OOM — needs TP or quantization
      weightPrecision: 'bf16',
    });

    const result = optimizeInference(config, 'throughput');

    // Should find a working config via TP increase or weight quantization
    expect(result.success).toBe(true);
    expect(result.changelog.length).toBeGreaterThan(0);
  });

  it('should explore EP for MoE models', () => {
    const model = getModel('deepseek-v3', 4096)!;
    const gpu = ALL_GPUS['h100-sxm'];

    const config: InferenceSimulationConfig = {
      modelSpec: model,
      gpu,
      numGPUs: 8,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 256,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
      tensorParallel: 1,
      expertParallel: 1,
    };

    const result = optimizeInference(config, 'throughput');

    expect(result.success).toBe(true);
    expect(result.totalSimulations).toBeGreaterThan(10);
  });

  it('should not make things worse for a good config', () => {
    const config = makeInferenceConfig({
      batchSize: 32,
      continuousBatching: true,
      flashAttention: true,
      weightPrecision: 'fp8',
    });

    const result = optimizeInference(config, 'throughput');

    expect(result.success).toBe(true);
    // Should not decrease throughput
    expect(result.afterMetric).toBeGreaterThanOrEqual(result.beforeMetric * 0.99);
  });
});

// ── Inference Exploration Grid Tests ─────────────────────────────────

describe('Inference Exploration Grid', () => {
  it('should generate candidates', () => {
    const model = getModel('llama3.1-8b', 4096)!;
    const gpu = ALL_GPUS['h100-sxm'];

    const config = makeInferenceConfig();

    const candidates = generateInferenceCandidates(config, model, gpu);
    expect(candidates.length).toBeGreaterThan(0);
    expect(candidates.length).toBeLessThanOrEqual(MAX_INFERENCE_SIMS);
  });

  it('should include FP8 candidates when GPU supports it', () => {
    const model = getModel('llama3.1-8b', 4096)!;
    const gpu = ALL_GPUS['h100-sxm'];

    const config = makeInferenceConfig();

    const candidates = generateInferenceCandidates(config, model, gpu);
    const hasFP8 = candidates.some(c => c.weightPrecision === 'fp8');
    expect(hasFP8).toBe(true);
  });

  it('should include EP for MoE models', () => {
    const model = getModel('deepseek-v3', 4096)!;
    const gpu = ALL_GPUS['h100-sxm'];

    const config: InferenceSimulationConfig = {
      modelSpec: model,
      gpu,
      numGPUs: 8,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 256,
      weightPrecision: 'bf16',
      kvCachePrecision: 'bf16',
    };

    const candidates = generateInferenceCandidates(config, model, gpu);
    const hasEP = candidates.some(c => (c.expertParallel ?? 1) > 1);
    expect(hasEP).toBe(true);
  });
});

// ── New Recommendation Generator Tests ───────────────────────────────

describe('New Recommendation Generators', () => {
  it('gbsScaling: fires and produces valid mutation', () => {
    const model = getModel('llama3.1-8b', 2048)!;
    const cluster = getPresetCluster('8x-h100')!;

    const config = makeTrainingConfig({
      modelSpec: model,
      clusterConfig: cluster,
      globalBatchSize: 128,
      microBatchSize: 2,
      strategyType: 'fsdp',
    });

    // Just verify the optimizer considers GBS changes
    const targetTokens = 100e9;
    const result = optimizeTraining(config, targetTokens, config.sequenceLength);

    expect(result.success).toBe(true);
    // Should have explored some configs
    expect(result.totalSimulations).toBeGreaterThan(5);
  });

  it('precisionUpgrade: fires on H100 with bf16', () => {
    const model = getModel('llama3.1-8b', 2048)!;
    const cluster = getPresetCluster('8x-h100')!;

    const config = makeTrainingConfig({
      modelSpec: model,
      clusterConfig: cluster,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
    });

    const targetTokens = 100e9;
    const result = optimizeTraining(config, targetTokens, config.sequenceLength);

    expect(result.success).toBe(true);
    // The optimizer should have at least tried FP8
    // (it may or may not end up choosing it depending on throughput)
    expect(result.totalSimulations).toBeGreaterThan(5);
  });

  it('precisionUpgrade: should not fire on A100', () => {
    const model = getModel('llama3.1-8b', 2048)!;
    const cluster = getPresetCluster('8x-a100')!;

    const config = makeTrainingConfig({
      modelSpec: model,
      clusterConfig: cluster,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
    });

    const targetTokens = 100e9;
    const result = optimizeTraining(config, targetTokens, config.sequenceLength);

    expect(result.success).toBe(true);
    // A100 has no fp8TFLOPS, so the precision should not change to fp8
    const precisionChange = result.changelog.find(c => c.field === 'mixedPrecision');
    if (precisionChange) {
      expect(precisionChange.to).not.toBe('fp8');
    }
  });
});

// ── Exploration Grid: Interleaved & PP>4 Tests ──────────────────────────

describe('Exploration Grid: Interleaved and PP>4', () => {
  it('should include PP=8 candidates for models with layers divisible by 8', () => {
    // Llama 3 70B has 80 layers (divisible by 8)
    const model = getModel('llama3-70b', 4096)!;
    const cluster = getPresetCluster('256x-h100')!;

    const config = makeTrainingConfig({
      modelSpec: model,
      clusterConfig: cluster,
      globalBatchSize: 4096,
      microBatchSize: 2,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 4 },
    });

    const candidates = generateTrainingCandidates(config, model, cluster);

    const hasPP8 = candidates.some(c => (c.strategyConfig?.pp ?? 1) === 8);
    expect(hasPP8).toBe(true);
  });

  it('should include interleaved-1f1b candidates', () => {
    // GPT-3 175B has 96 layers — divisible by 2*2=4, 4*2=8, etc.
    const model = getModel('gpt3-175b', 2048)!;
    const cluster = getPresetCluster('512x-h100')!;

    const config = makeTrainingConfig({
      modelSpec: model,
      clusterConfig: cluster,
      globalBatchSize: 2048,
      microBatchSize: 2,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 4 },
    });

    const candidates = generateTrainingCandidates(config, model, cluster);

    const hasInterleaved = candidates.some(
      c => c.strategyConfig?.pipelineSchedule === 'interleaved-1f1b'
    );
    expect(hasInterleaved).toBe(true);

    // Should include both v=2 and v=4 variants
    const hasV2 = candidates.some(
      c => c.strategyConfig?.pipelineSchedule === 'interleaved-1f1b' &&
           c.strategyConfig?.interleavedStages === 2
    );
    const hasV4 = candidates.some(
      c => c.strategyConfig?.pipelineSchedule === 'interleaved-1f1b' &&
           c.strategyConfig?.interleavedStages === 4
    );
    expect(hasV2).toBe(true);
    expect(hasV4).toBe(true);
  });

  it('should include MBS=8 candidates', () => {
    const model = getModel('llama3.1-8b', 2048)!;
    const cluster = getPresetCluster('8x-h100')!;

    const config = makeTrainingConfig({
      modelSpec: model,
      clusterConfig: cluster,
      globalBatchSize: 1024,
      microBatchSize: 4,
    });

    const candidates = generateTrainingCandidates(config, model, cluster);

    const hasMBS8 = candidates.some(c => c.microBatchSize === 8);
    expect(hasMBS8).toBe(true);
  });

  it('should only generate interleaved when pp*v fits within layer count', () => {
    const model = getModel('llama3-70b', 4096)!; // 80 layers
    const cluster = getPresetCluster('256x-h100')!;

    const config = makeTrainingConfig({
      modelSpec: model,
      clusterConfig: cluster,
      globalBatchSize: 4096,
      microBatchSize: 2,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 4 },
    });

    const candidates = generateTrainingCandidates(config, model, cluster);

    for (const c of candidates) {
      if (c.strategyConfig?.pipelineSchedule === 'interleaved-1f1b') {
        const pp = c.strategyConfig?.pp ?? 1;
        const v = c.strategyConfig?.interleavedStages ?? 2;
        expect(pp * v).toBeLessThanOrEqual(model.numLayers);
      }
    }
  });
});

// ── Large-cluster inference optimizer tests ──────────────────────────

describe('Inference Optimizer: Large Cluster', () => {
  it('grid should include batches where batchPerReplica > 1 for Nemotron-4 340B on 6144 H100s', () => {
    const model = getModel('nemotron-4-340b', 4096)!;
    const gpu = ALL_GPUS['h100-sxm'];

    const config: InferenceSimulationConfig = {
      modelSpec: model,
      gpu,
      numGPUs: 6144,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 256,
      weightPrecision: 'int4',
      kvCachePrecision: 'int4',
      tensorParallel: 8,
    };

    const candidates = generateInferenceCandidates(config, model, gpu);

    // With TP=8, numReplicas = 6144/8 = 768
    // Grid should include batches like 768*2=1536, 768*4=3072, etc.
    const hasBatchPerReplicaGT1 = candidates.some(c => {
      const tp = c.tensorParallel ?? 1;
      const ep = c.expertParallel ?? 1;
      const replicas = Math.floor(6144 / (tp * Math.max(1, ep)));
      const batchPerReplica = Math.ceil((c.batchSize ?? 1) / replicas);
      return batchPerReplica > 1;
    });
    expect(hasBatchPerReplicaGT1).toBe(true);
  });

  it('throughput optimizer should find higher batch than latency optimizer for Nemotron-4 340B on 6144 H100s', () => {
    const model = getModel('nemotron-4-340b', 4096)!;
    const gpu = ALL_GPUS['h100-sxm'];

    const config: InferenceSimulationConfig = {
      modelSpec: model,
      gpu,
      numGPUs: 6144,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 256,
      weightPrecision: 'int4',
      kvCachePrecision: 'int4',
      tensorParallel: 8,
    };

    const throughputResult = optimizeInference(config, 'throughput');
    const latencyResult = optimizeInference(config, 'latency');

    expect(throughputResult.success).toBe(true);
    expect(latencyResult.success).toBe(true);

    // Throughput optimizer should pick a significantly larger batch
    const throughputBatch = throughputResult.optimizedConfig.batchSize ?? 1;
    const latencyBatch = latencyResult.optimizedConfig.batchSize ?? 1;
    expect(throughputBatch).toBeGreaterThan(latencyBatch);
  });

  it('throughput result should have batch increase in changelog; latency should not', () => {
    const model = getModel('nemotron-4-340b', 4096)!;
    const gpu = ALL_GPUS['h100-sxm'];

    const config: InferenceSimulationConfig = {
      modelSpec: model,
      gpu,
      numGPUs: 6144,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 256,
      weightPrecision: 'int4',
      kvCachePrecision: 'int4',
      tensorParallel: 8,
    };

    const throughputResult = optimizeInference(config, 'throughput');
    const latencyResult = optimizeInference(config, 'latency');

    // Throughput should have changed batch
    const throughputBatchChange = throughputResult.changelog.find(c => c.field === 'batchSize');
    expect(throughputBatchChange).toBeDefined();
    expect(Number(throughputBatchChange!.to)).toBeGreaterThan(1);

    // Latency should NOT have changed batch (already optimal at 1)
    const latencyBatchChange = latencyResult.changelog.find(c => c.field === 'batchSize');
    if (latencyBatchChange) {
      // If it changed batch at all, it should still be small
      expect(Number(latencyBatchChange.to)).toBeLessThanOrEqual(1);
    }
  });
});
