/**
 * Test that memory changes with precision, sequence length, and micro batch size
 */
import { describe, it, expect } from 'vitest';
import { SimulationEngine } from '../../src/core/simulation/index.ts';
import { getModel } from '../../src/core/models/index.ts';
import { createCluster } from '../../src/core/hardware/topology.ts';
import { DGX_H100 } from '../../src/core/hardware/presets.ts';

describe('Memory Parameter Sensitivity', () => {
  const model = getModel('llama2-7b', 2048)!;
  const cluster = createCluster(DGX_H100, 1, 'single-node');

  const runSimulation = (microBatch: number, seqLen: number, precision: 'fp32' | 'bf16') => {
    const engine = new SimulationEngine();
    engine.configure({
      modelSpec: model,
      clusterConfig: cluster,
      globalBatchSize: 64,
      microBatchSize: microBatch,
      sequenceLength: seqLen,
      strategyType: 'fsdp',
      activationCheckpointing: true,
      mixedPrecision: precision,
    });
    return engine.simulate();
  };

  it('should increase activations with larger micro batch size', () => {
    const metrics1 = runSimulation(1, 1024, 'bf16');
    const metrics4 = runSimulation(4, 1024, 'bf16');

    // 4x micro batch should give ~4x activations
    const ratio = metrics4.memoryPerGPU.activations / metrics1.memoryPerGPU.activations;
    expect(ratio).toBeGreaterThan(3.5);
    expect(ratio).toBeLessThan(4.5);
  });

  it('should increase activations with longer sequence length', () => {
    const metrics1024 = runSimulation(4, 1024, 'bf16');
    const metrics2048 = runSimulation(4, 2048, 'bf16');

    // With flash attention (default), attention memory is O(seq) not O(seq²)
    // So 2x sequence length gives ~2x activations (linear scaling)
    const ratio = metrics2048.memoryPerGPU.activations / metrics1024.memoryPerGPU.activations;
    expect(ratio).toBeGreaterThanOrEqual(2);
  });

  it('should halve parameter memory with bf16 vs fp32', () => {
    const metricsFp32 = runSimulation(4, 2048, 'fp32');
    const metricsBf16 = runSimulation(4, 2048, 'bf16');

    // bf16 should use half the parameter memory
    const ratio = metricsBf16.memoryPerGPU.parameters / metricsFp32.memoryPerGPU.parameters;
    expect(ratio).toBeCloseTo(0.5, 1);
  });

  it('should show activation memory increases with larger micro-batch and seq length', () => {
    const metricsSmall = runSimulation(1, 1024, 'bf16');
    const metricsLarge = runSimulation(8, 4096, 'bf16');

    // Activation memory should scale with micro-batch × seq-length
    // 8×4096 vs 1×1024 = 32× more tokens → much larger activations
    expect(metricsLarge.memoryPerGPU.activations).toBeGreaterThan(metricsSmall.memoryPerGPU.activations * 10);
  });
});
