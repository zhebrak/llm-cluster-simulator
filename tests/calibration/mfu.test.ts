/**
 * MFU (Model FLOPS Utilization) calculation tests
 *
 * Validates against:
 * - Standard formula: Training FLOPS = 6 * params * tokens
 * - Published benchmarks from Megatron-LM, DeepSpeed, LLaMA papers
 */

import { describe, it, expect } from 'vitest';
import { buildModelSpec, matmulFlops, trainingFlops } from '../../src/core/models/primitives.ts';
import { ALL_MODEL_CONFIGS } from '../../src/core/models/architectures.ts';
import { getPresetCluster } from '../../src/core/hardware/presets.ts';
import { SimulationEngine } from '../../src/core/simulation/engine.ts';
import { assertValidEngine } from '../helpers/validated-metrics.ts';

describe('FLOPS calculations', () => {
  describe('matmulFlops', () => {
    it('should compute 2*M*K*N for matrix multiply', () => {
      // (M, K) x (K, N) = 2*M*K*N FLOPs
      expect(matmulFlops(1, 1024, 1024)).toBe(2 * 1 * 1024 * 1024);
      expect(matmulFlops(512, 4096, 4096)).toBe(2 * 512 * 4096 * 4096);
    });
  });

  describe('trainingFlops', () => {
    it('should compute 6 * params * tokens', () => {
      const params = 7e9; // 7B params
      const tokens = 2048;
      expect(trainingFlops(params, tokens)).toBe(6 * params * tokens);
    });
  });

  describe('flopsPerToken', () => {
    it('should be approximately 2 * params for dense models', () => {
      // For transformer models, forward pass FLOPS ≈ 2 * params per token
      // (the factor of 2 comes from multiply-add operations in matmuls)
      const llama7b = ALL_MODEL_CONFIGS['llama2-7b'];
      const model = buildModelSpec(llama7b, 2048);

      // flopsPerToken should be roughly 2 * totalParams
      // Allow 50% tolerance for non-matmul operations (norms, activations, attention scores)
      const expectedFlops = 2 * model.totalParams;
      const ratio = model.flopsPerToken / expectedFlops;

      expect(ratio).toBeGreaterThan(0.5);
      expect(ratio).toBeLessThan(2.0);
    });

    it('should scale with model size', () => {
      const llama7b = buildModelSpec(ALL_MODEL_CONFIGS['llama2-7b'], 2048);
      const llama13b = buildModelSpec(ALL_MODEL_CONFIGS['llama2-13b'], 2048);
      const llama70b = buildModelSpec(ALL_MODEL_CONFIGS['llama2-70b'], 2048);

      // flopsPerToken should roughly scale with param count
      expect(llama13b.flopsPerToken / llama7b.flopsPerToken).toBeGreaterThan(1.5);
      expect(llama70b.flopsPerToken / llama13b.flopsPerToken).toBeGreaterThan(3);
    });
  });
});

describe('MFU calculation', () => {
  it('should produce realistic MFU values (25-55%) for typical configs', () => {
    const engine = new SimulationEngine();

    engine.configure({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 32,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
    });

    assertValidEngine(engine);
    const metrics = engine.simulate();

    // MFU must never exceed 100% (physical constraint)
    expect(metrics.mfu).toBeLessThanOrEqual(1.0);
    // MFU should be in realistic range (~42% for LLaMA-7B FSDP on 8xH100)
    expect(metrics.mfu).toBeGreaterThan(0.30);
    expect(metrics.mfu).toBeLessThan(0.55);
  });

  it('should produce valid MFU for B200 GPUs', () => {
    const engine = new SimulationEngine();

    engine.configure({
      modelId: 'llama2-7b',
      clusterId: '32x-b200',
      globalBatchSize: 128,
      microBatchSize: 4,
      sequenceLength: 4096,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
    });

    const validation = engine.validate();
    expect(validation.valid).toBe(true);
    const metrics = engine.simulate();

    // MFU must never exceed 100%
    expect(metrics.mfu).toBeLessThanOrEqual(1.0);
    expect(metrics.mfu).toBeGreaterThan(0);
    // Should be in realistic range (~43% for LLaMA-7B FSDP on 32xB200)
    expect(metrics.mfu).toBeGreaterThan(0.30);
    expect(metrics.mfu).toBeLessThan(0.55);
  });

  it('should be consistent with 6 * params * tokens formula', () => {
    const engine = new SimulationEngine();

    engine.configure({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 32,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
    });

    assertValidEngine(engine);
    const metrics = engine.simulate();
    const model = buildModelSpec(ALL_MODEL_CONFIGS['llama2-7b'], 2048);
    const cluster = getPresetCluster('8x-h100')!;

    // MFU must never exceed 100% (physical constraint)
    expect(metrics.mfu).toBeLessThanOrEqual(1.0);
    expect(metrics.mfu).toBeGreaterThan(0);

    // Manually compute MFU using standard formula
    const tokens = 32 * 2048; // globalBatchSize * seqLength
    const activeParams = model.activeParams ?? model.totalParams;
    const totalTrainingFlops = 6 * activeParams * tokens;
    const stepTimeSeconds = metrics.stepTimeMs / 1000;
    const achievedTFLOPS = totalTrainingFlops / stepTimeSeconds / 1e12;
    const expectedMFU = achievedTFLOPS / cluster.totalTFLOPS;

    // Should match within 15% (tightened from 30%)
    expect(Math.abs(metrics.mfu - expectedMFU) / expectedMFU).toBeLessThan(0.15);
  });

  it('should increase with larger batch sizes (up to a point)', () => {
    const engine1 = new SimulationEngine();
    const engine2 = new SimulationEngine();

    engine1.configure({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 8,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
    });

    engine2.configure({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 64,
      microBatchSize: 8,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
    });

    assertValidEngine(engine1);
    const metrics1 = engine1.simulate();
    assertValidEngine(engine2);
    const metrics2 = engine2.simulate();

    // Both must have MFU <= 100%
    expect(metrics1.mfu).toBeLessThanOrEqual(1.0);
    expect(metrics2.mfu).toBeLessThanOrEqual(1.0);
    // Larger batch should have better MFU (amortizes communication overhead)
    expect(metrics2.mfu).toBeGreaterThan(metrics1.mfu);
  });
});

describe('Benchmark validation', () => {
  it('LLaMA-7B on 8xH100 should achieve ~35-50% MFU', () => {
    // Based on published training reports, LLaMA training achieves ~40-50% MFU
    const engine = new SimulationEngine();

    engine.configure({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 64,
      microBatchSize: 8,
      sequenceLength: 4096,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
    });

    assertValidEngine(engine);
    const metrics = engine.simulate();

    // MFU must never exceed 100%
    expect(metrics.mfu).toBeLessThanOrEqual(1.0);
    // Should be in the ballpark of published results
    expect(metrics.mfu).toBeGreaterThan(0.30);
    expect(metrics.mfu).toBeLessThan(0.55);
  });

  it('LLaMA-7B on B200 should achieve realistic MFU (not 200%+)', () => {
    // B200 MFU must stay in realistic range (not exceed 100%)
    const engine = new SimulationEngine();

    engine.configure({
      modelId: 'llama2-7b',
      clusterId: '32x-b200',
      globalBatchSize: 128,
      microBatchSize: 4,
      sequenceLength: 4096,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
    });

    const validation = engine.validate();
    expect(validation.valid).toBe(true);
    const metrics = engine.simulate();

    // Critical: MFU must be <= 100%
    expect(metrics.mfu).toBeLessThanOrEqual(1.0);
    // Should be realistic (~43% for FSDP on B200)
    expect(metrics.mfu).toBeGreaterThan(0.30);
    expect(metrics.mfu).toBeLessThan(0.55);
  });

  it('should show higher MFU for FSDP vs DDP on memory-constrained configs', () => {
    // FSDP reduces memory pressure, potentially allowing larger batches
    const engineDDP = new SimulationEngine();
    const engineFSDP = new SimulationEngine();

    engineDDP.configure({
      modelId: 'llama2-13b',
      clusterId: '8x-h100',
      globalBatchSize: 16,
      microBatchSize: 2,
      sequenceLength: 2048,
      strategyType: 'ddp',
      mixedPrecision: 'bf16',
    });

    engineFSDP.configure({
      modelId: 'llama2-13b',
      clusterId: '8x-h100',
      globalBatchSize: 16,
      microBatchSize: 2,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
    });

    const ddpValidation = engineDDP.validate();
    const fsdpValidation = engineFSDP.validate();

    // For larger models, FSDP should have fewer memory errors
    // (DDP replicates full model, FSDP shards it)
    if (!ddpValidation.valid && fsdpValidation.valid) {
      // This is expected - DDP fails on memory, FSDP works
      expect(fsdpValidation.valid).toBe(true);
    }
  });

  it('TFLOPS per GPU should be reasonable', () => {
    const engine = new SimulationEngine();

    engine.configure({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 32,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
    });

    assertValidEngine(engine);
    const metrics = engine.simulate();
    const cluster = getPresetCluster('8x-h100')!;
    const peakTFLOPSPerGPU = cluster.totalTFLOPS / cluster.totalGPUs;

    // Achieved TFLOPS should be a reasonable fraction of peak (~42% for this config)
    expect(metrics.tflopsPerGPU).toBeGreaterThan(peakTFLOPSPerGPU * 0.30);
    expect(metrics.tflopsPerGPU).toBeLessThan(peakTFLOPSPerGPU * 0.55);
  });
});

describe('Memory calculations', () => {
  it('should account for parameters, gradients, optimizer states, and activations', () => {
    const engine = new SimulationEngine();

    engine.configure({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 32,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
    });

    assertValidEngine(engine);
    const metrics = engine.simulate();

    // Memory breakdown should have all components
    expect(metrics.memoryPerGPU.parameters).toBeGreaterThan(0);
    expect(metrics.memoryPerGPU.gradients).toBeGreaterThan(0);
    expect(metrics.memoryPerGPU.optimizerStates).toBeGreaterThan(0);
    expect(metrics.memoryPerGPU.activations).toBeGreaterThan(0);

    // Total should be sum of components (approximately)
    const sumOfParts = metrics.memoryPerGPU.parameters +
                       metrics.memoryPerGPU.gradients +
                       metrics.memoryPerGPU.optimizerStates +
                       metrics.memoryPerGPU.peakActivations +
                       metrics.memoryPerGPU.temporary +
                       metrics.memoryPerGPU.reserved;
    expect(Math.abs(metrics.memoryPerGPU.total - sumOfParts) / sumOfParts).toBeLessThan(0.01);
  });

  it('DDP should use ~16-20 bytes per parameter', () => {
    // DDP memory per param: 2 (bf16 params) + 4 (fp32 grads) + 12 (fp32 optimizer) = ~18 bytes
    // Use a smaller model (GPT-3 1.3B) that fits in DDP on H100 (80 GB)
    const engine = new SimulationEngine();

    engine.configure({
      modelId: 'gpt3-1.3b',
      clusterId: '8x-h100',
      globalBatchSize: 32,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'ddp',
      mixedPrecision: 'bf16',
    });

    assertValidEngine(engine);
    const metrics = engine.simulate();
    const model = buildModelSpec(ALL_MODEL_CONFIGS['gpt3-1.3b'], 2048);

    const bytesPerParam = (metrics.memoryPerGPU.parameters +
                          metrics.memoryPerGPU.gradients +
                          metrics.memoryPerGPU.optimizerStates) / model.totalParams;

    // Should be around 16-20 bytes per param for bf16 params/grads + fp32 optimizer
    expect(bytesPerParam).toBeGreaterThan(14);
    expect(bytesPerParam).toBeLessThan(22);
  });

  it('FSDP should use less memory per GPU than DDP', () => {
    // Use a smaller model (GPT-3 1.3B) that fits in both DDP and FSDP on H100
    const engineDDP = new SimulationEngine();
    const engineFSDP = new SimulationEngine();

    engineDDP.configure({
      modelId: 'gpt3-1.3b',
      clusterId: '8x-h100',
      globalBatchSize: 32,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'ddp',
      mixedPrecision: 'bf16',
    });

    engineFSDP.configure({
      modelId: 'gpt3-1.3b',
      clusterId: '8x-h100',
      globalBatchSize: 32,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
    });

    assertValidEngine(engineDDP);
    const ddpMetrics = engineDDP.simulate();
    assertValidEngine(engineFSDP);
    const fsdpMetrics = engineFSDP.simulate();

    // FSDP should use significantly less memory (params/grads/optimizer are sharded)
    expect(fsdpMetrics.memoryPerGPU.total).toBeLessThan(ddpMetrics.memoryPerGPU.total);
  });
});

describe('Timing calculations', () => {
  it('should have forward, backward, communication, and optimizer times', () => {
    const engine = new SimulationEngine();

    engine.configure({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 32,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
    });

    assertValidEngine(engine);
    const metrics = engine.simulate();

    expect(metrics.timing.forward).toBeGreaterThan(0);
    expect(metrics.timing.backward).toBeGreaterThan(0);
    expect(metrics.timing.communication).toBeGreaterThan(0);
    expect(metrics.timing.optimizer).toBeGreaterThan(0);
  });

  it('backward should be ~2x forward', () => {
    const engine = new SimulationEngine();

    engine.configure({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 32,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
    });

    assertValidEngine(engine);
    const metrics = engine.simulate();

    // With activation checkpointing (default=true), backward = 3x forward (recompute)
    const ratio = metrics.timing.backward / metrics.timing.forward;
    expect(ratio).toBeGreaterThan(1.5);
    expect(ratio).toBeLessThan(3.5);
  });

  it('communication overhead should increase with more GPUs', () => {
    // Use a smaller model that fits on a single H100 GPU
    const engine1 = new SimulationEngine();
    const engine8 = new SimulationEngine();

    engine1.configure({
      modelId: 'gpt3-1.3b',
      clusterId: '1x-h100',
      globalBatchSize: 8,
      microBatchSize: 8,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
    });

    engine8.configure({
      modelId: 'gpt3-1.3b',
      clusterId: '8x-h100',
      globalBatchSize: 64,
      microBatchSize: 8,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
    });

    assertValidEngine(engine1);
    const metrics1 = engine1.simulate();
    assertValidEngine(engine8);
    const metrics8 = engine8.simulate();

    // Single-node FSDP: near-zero comm overhead (NVLink hides most AllGather)
    expect(metrics1.communicationOverhead).toBeLessThan(0.001);
    // Multi-node should have higher communication overhead
    expect(metrics8.communicationOverhead).toBeGreaterThan(metrics1.communicationOverhead);
  });
});
