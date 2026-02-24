/**
 * Hybrid Strategy Parallelism Parameter Tests
 *
 * Validates that TP and PP degree changes actually affect computation
 * for hybrid strategies (fsdp-tp, zero1-tp, ddp-tp-pp, fsdp-tp-pp, etc.)
 */

import { describe, it, expect } from 'vitest';
import { SimulationEngine } from '../../src/core/simulation/engine.ts';
import { assertValidEngine } from '../helpers/validated-metrics.ts';

describe('Hybrid Strategy Parallelism Parameters', () => {
  describe('FSDP + TP: TP degree affects computation', () => {
    it('should produce different timing with different TP degrees', () => {
      const baseConfig = {
        modelId: 'llama2-70b',
        clusterId: '128x-a100',
        globalBatchSize: 512,
        microBatchSize: 4,
        sequenceLength: 2048,
        strategyType: 'fsdp-tp' as const,
      };

      // TP=2
      const engine1 = new SimulationEngine();
      engine1.configure({ ...baseConfig, strategyConfig: { tp: 2 } });
      assertValidEngine(engine1);
      const metrics1 = engine1.simulate();

      // TP=4
      const engine2 = new SimulationEngine();
      engine2.configure({ ...baseConfig, strategyConfig: { tp: 4 } });
      assertValidEngine(engine2);
      const metrics2 = engine2.simulate();

      // TP=8
      const engine3 = new SimulationEngine();
      engine3.configure({ ...baseConfig, strategyConfig: { tp: 8 } });
      assertValidEngine(engine3);
      const metrics3 = engine3.simulate();

      // Step times should differ with different TP degrees
      expect(metrics1.stepTimeMs).not.toEqual(metrics2.stepTimeMs);
      expect(metrics2.stepTimeMs).not.toEqual(metrics3.stepTimeMs);

      // MFU should differ (TP communication overhead changes)
      expect(metrics1.mfu).not.toEqual(metrics2.mfu);
      expect(metrics2.mfu).not.toEqual(metrics3.mfu);

      // Communication overhead should change with TP degree
      expect(metrics1.communicationOverhead).not.toEqual(metrics2.communicationOverhead);
    });

    it('FSDP+TP should have same total parameter memory regardless of TP (FSDP shards all)', () => {
      // With FSDP, params are sharded across ALL GPUs (tp * dp = total)
      // So changing TP doesn't change per-GPU parameter memory
      const baseConfig = {
        modelId: 'llama2-70b',
        clusterId: '128x-a100',
        globalBatchSize: 512,
        microBatchSize: 4,
        sequenceLength: 2048,
        strategyType: 'fsdp-tp' as const,
      };

      const engine1 = new SimulationEngine();
      engine1.configure({ ...baseConfig, strategyConfig: { tp: 2 } });
      assertValidEngine(engine1);
      const metrics1 = engine1.simulate();

      const engine2 = new SimulationEngine();
      engine2.configure({ ...baseConfig, strategyConfig: { tp: 8 } });
      assertValidEngine(engine2);
      const metrics2 = engine2.simulate();

      // Parameter memory should be the same (FSDP shards across all GPUs)
      expect(metrics1.memoryPerGPU.parameters).toEqual(metrics2.memoryPerGPU.parameters);

      // But timing/MFU should differ
      expect(metrics1.mfu).not.toEqual(metrics2.mfu);
    });
  });

  describe('ZeRO-1 + TP: TP degree affects computation', () => {
    it('should produce different results with different TP degrees', () => {
      const baseConfig = {
        modelId: 'llama2-7b',
        clusterId: '128x-a100',
        globalBatchSize: 512,
        microBatchSize: 4,
        sequenceLength: 2048,
        strategyType: 'zero1-tp' as const,
      };

      // TP=2
      const engine1 = new SimulationEngine();
      engine1.configure({ ...baseConfig, strategyConfig: { tp: 2 } });
      assertValidEngine(engine1);
      const metrics1 = engine1.simulate();

      // TP=4
      const engine2 = new SimulationEngine();
      engine2.configure({ ...baseConfig, strategyConfig: { tp: 4 } });
      assertValidEngine(engine2);
      const metrics2 = engine2.simulate();

      // TP=8
      const engine3 = new SimulationEngine();
      engine3.configure({ ...baseConfig, strategyConfig: { tp: 8 } });
      assertValidEngine(engine3);
      const metrics3 = engine3.simulate();

      // Step times should differ
      expect(metrics1.stepTimeMs).not.toEqual(metrics2.stepTimeMs);
      expect(metrics2.stepTimeMs).not.toEqual(metrics3.stepTimeMs);

      // MFU should differ
      expect(metrics1.mfu).not.toEqual(metrics2.mfu);

      // ZeRO-1 only shards optimizer, not params - so TP reduces param memory
      // Each GPU has params sharded by TP only (not DP)
      expect(metrics1.memoryPerGPU.parameters).toBeGreaterThan(metrics2.memoryPerGPU.parameters);
      expect(metrics2.memoryPerGPU.parameters).toBeGreaterThan(metrics3.memoryPerGPU.parameters);
    });
  });

  describe('FSDP + TP + PP: PP degree affects computation', () => {
    it('should produce different timing with different PP degrees', () => {
      const baseConfig = {
        modelId: 'llama2-70b',
        clusterId: '128x-a100',
        globalBatchSize: 512,
        microBatchSize: 4,
        sequenceLength: 2048,
        strategyType: 'fsdp-tp-pp' as const,
      };

      // PP=2
      const engine1 = new SimulationEngine();
      engine1.configure({ ...baseConfig, strategyConfig: { tp: 8, pp: 2, } });
      assertValidEngine(engine1);
      const metrics1 = engine1.simulate();

      // PP=4
      const engine2 = new SimulationEngine();
      engine2.configure({ ...baseConfig, strategyConfig: { tp: 8, pp: 4, } });
      assertValidEngine(engine2);
      const metrics2 = engine2.simulate();

      // PP=8
      const engine3 = new SimulationEngine();
      engine3.configure({ ...baseConfig, strategyConfig: { tp: 8, pp: 8, } });
      assertValidEngine(engine3);
      const metrics3 = engine3.simulate();

      // Step times should differ with different PP degrees
      expect(metrics1.stepTimeMs).not.toEqual(metrics2.stepTimeMs);
      expect(metrics2.stepTimeMs).not.toEqual(metrics3.stepTimeMs);

      // Pipeline bubble should be present (PP > 1)
      expect(metrics1.pipelineBubble).toBeGreaterThan(0);
      expect(metrics2.pipelineBubble).toBeGreaterThan(0);
      expect(metrics3.pipelineBubble).toBeGreaterThan(0);
    });

    it('larger GBS → more GA steps → smaller pipeline bubble', () => {
      // TP=8, PP=4, 128 GPUs → DP=128/(8*4)=4, MBS=4
      // GBS=64 → GA=ceil(64/(4*4))=4, GBS=256 → GA=16, GBS=1024 → GA=64
      const makeEngine = (gbs: number) => {
        const engine = new SimulationEngine();
        engine.configure({
          modelId: 'llama2-70b',
          clusterId: '128x-a100',
          globalBatchSize: gbs,
          microBatchSize: 4,
          sequenceLength: 2048,
          strategyType: 'fsdp-tp-pp' as const,
          strategyConfig: { tp: 8, pp: 4 },
        });
        assertValidEngine(engine);
        return engine.simulate();
      };

      const metrics1 = makeEngine(64);   // GA=4, bubble=3/7
      const metrics2 = makeEngine(256);  // GA=16, bubble=3/19
      const metrics3 = makeEngine(1024); // GA=64, bubble=3/67

      // Pipeline bubble should decrease with more GA steps
      expect(metrics1.pipelineBubble).toBeGreaterThan(metrics2.pipelineBubble);
      expect(metrics2.pipelineBubble).toBeGreaterThan(metrics3.pipelineBubble);
    });
  });

  describe('DDP + TP + PP: PP degree affects computation', () => {
    it('should produce different results with different PP degrees', () => {
      const baseConfig = {
        modelId: 'llama2-7b',
        clusterId: '128x-a100',
        globalBatchSize: 512,
        microBatchSize: 4,
        sequenceLength: 2048,
        strategyType: 'ddp-tp-pp' as const,
      };

      // PP=2
      const engine1 = new SimulationEngine();
      engine1.configure({ ...baseConfig, strategyConfig: { tp: 2, pp: 2, } });
      assertValidEngine(engine1);
      const metrics1 = engine1.simulate();

      // PP=4
      const engine2 = new SimulationEngine();
      engine2.configure({ ...baseConfig, strategyConfig: { tp: 2, pp: 4, } });
      assertValidEngine(engine2);
      const metrics2 = engine2.simulate();

      // PP=8
      const engine3 = new SimulationEngine();
      engine3.configure({ ...baseConfig, strategyConfig: { tp: 2, pp: 8, } });
      assertValidEngine(engine3);
      const metrics3 = engine3.simulate();

      // Step times should differ
      expect(metrics1.stepTimeMs).not.toEqual(metrics2.stepTimeMs);
      expect(metrics2.stepTimeMs).not.toEqual(metrics3.stepTimeMs);

      // DDP+TP+PP: PP shards params across stages, so memory decreases with higher PP
      expect(metrics1.memoryPerGPU.parameters).toBeGreaterThan(
        metrics2.memoryPerGPU.parameters
      );
      expect(metrics2.memoryPerGPU.parameters).toBeGreaterThan(
        metrics3.memoryPerGPU.parameters
      );
    });
  });

  describe('Consistency: Same config produces same results', () => {
    it('fsdp-tp with same TP should produce identical results', () => {
      const config = {
        modelId: 'llama2-70b',
        clusterId: '128x-a100',
        globalBatchSize: 512,
        microBatchSize: 4,
        sequenceLength: 2048,
        strategyType: 'fsdp-tp' as const,
        strategyConfig: { tp: 4 },
      };

      const engine1 = new SimulationEngine();
      engine1.configure(config);
      assertValidEngine(engine1);
      const metrics1 = engine1.simulate();

      const engine2 = new SimulationEngine();
      engine2.configure(config);
      assertValidEngine(engine2);
      const metrics2 = engine2.simulate();

      expect(metrics1.stepTimeMs).toEqual(metrics2.stepTimeMs);
      expect(metrics1.mfu).toEqual(metrics2.mfu);
      expect(metrics1.memoryPerGPU.total).toEqual(metrics2.memoryPerGPU.total);
    });

    it('fsdp-tp-pp with same config should produce identical results', () => {
      const config = {
        modelId: 'llama2-70b',
        clusterId: '128x-a100',
        globalBatchSize: 512,
        microBatchSize: 4,
        sequenceLength: 2048,
        strategyType: 'fsdp-tp-pp' as const,
        strategyConfig: { tp: 8, pp: 4, },
      };

      const engine1 = new SimulationEngine();
      engine1.configure(config);
      assertValidEngine(engine1);
      const metrics1 = engine1.simulate();

      const engine2 = new SimulationEngine();
      engine2.configure(config);
      assertValidEngine(engine2);
      const metrics2 = engine2.simulate();

      expect(metrics1.stepTimeMs).toEqual(metrics2.stepTimeMs);
      expect(metrics1.mfu).toEqual(metrics2.mfu);
      expect(metrics1.pipelineBubble).toEqual(metrics2.pipelineBubble);
    });
  });

  describe('Edge cases', () => {
    it('TP=1 should be equivalent to no TP (pure FSDP)', () => {
      const baseConfig = {
        modelId: 'llama2-7b',
        clusterId: '8x-a100',
        globalBatchSize: 64,
        microBatchSize: 4,
        sequenceLength: 2048,
      };

      // fsdp-tp with TP=1
      const engine1 = new SimulationEngine();
      engine1.configure({
        ...baseConfig,
        strategyType: 'fsdp-tp',
        strategyConfig: { tp: 1 },
      });
      assertValidEngine(engine1);
      const metrics1 = engine1.simulate();

      // Pure FSDP
      const engine2 = new SimulationEngine();
      engine2.configure({
        ...baseConfig,
        strategyType: 'fsdp',
      });
      assertValidEngine(engine2);
      const metrics2 = engine2.simulate();

      // MFU should be very similar (may differ slightly due to 3D strategy overhead)
      const mfuDiff = Math.abs(metrics1.mfu - metrics2.mfu);
      expect(mfuDiff).toBeLessThan(0.10); // Within 10%
    });

    it('PP=1 should have zero pipeline bubble', () => {
      const baseConfig = {
        modelId: 'llama2-7b',
        clusterId: '8x-a100',
        globalBatchSize: 64,
        microBatchSize: 4,
        sequenceLength: 2048,
      };

      // fsdp-tp-pp with PP=1
      const engine1 = new SimulationEngine();
      engine1.configure({
        ...baseConfig,
        strategyType: 'fsdp-tp-pp',
        strategyConfig: { tp: 2, pp: 1 },
      });
      assertValidEngine(engine1);
      const metrics1 = engine1.simulate();

      // Pipeline bubble should be 0 with PP=1
      expect(metrics1.pipelineBubble).toBe(0);
    });
  });
});
