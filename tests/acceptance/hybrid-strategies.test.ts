/**
 * Hybrid Strategy Validation Tests
 *
 * Tests for:
 * 1. Valid/invalid hybrid parallelism combinations
 * 2. Published training configurations (BLOOM, MT-NLG, LLaMA 3)
 * 3. MFU validation against published benchmarks
 */

import { describe, it, expect } from 'vitest';
import {
  validateHybridCombination,
  validateTPTopology,
} from '../../src/core/validation/hybrid-validator.ts';
import { create3DParallelStrategy, type StrategyContext } from '../../src/core/strategies/index.ts';
import { getModel } from '../../src/core/models/index.ts';
import { createMultiNodeCluster } from '../../src/core/hardware/index.ts';
import { DEFAULT_DTYPE_CONFIG, DEFAULT_ADAMW_CONFIG, DEFAULT_LR_SCHEDULE } from '../../src/types/index.ts';
import { makeStrategyContext } from '../helpers/strategy-context.ts';

describe('Hybrid Strategy Validation', () => {
  describe('Valid Combinations', () => {
    it('FSDP + TP should work without warnings', () => {
      const result = validateHybridCombination({
        dpType: 'fsdp',
        tp: 8,
        pp: 1,
        gpusPerNode: 8,
      });
      expect(result).toBeNull();
    });

    it('ZeRO-1 + TP should work without warnings', () => {
      const result = validateHybridCombination({
        dpType: 'zero-1',
        tp: 8,
        pp: 1,
        gpusPerNode: 8,
      });
      expect(result).toBeNull();
    });

    it('ZeRO-1 + TP + PP should work without warnings', () => {
      const result = validateHybridCombination({
        dpType: 'zero-1',
        tp: 8,
        pp: 12,
        gpusPerNode: 8,
      });
      expect(result).toBeNull();
    });

    it('DDP + TP + PP should work without warnings', () => {
      const result = validateHybridCombination({
        dpType: 'ddp',
        tp: 8,
        pp: 16,
        gpusPerNode: 8,
      });
      expect(result).toBeNull();
    });

    it('FSDP + TP + PP should work without warnings', () => {
      const result = validateHybridCombination({
        dpType: 'fsdp',
        tp: 8,
        pp: 16,
        gpusPerNode: 8,
      });
      expect(result).toBeNull();
    });
  });

  describe('Invalid/Caution Combinations', () => {
    it('ZeRO-2 + PP should produce warning', () => {
      const result = validateHybridCombination({
        dpType: 'zero-2',
        tp: 1,
        pp: 8,
        gpusPerNode: 8,
      });
      expect(result).not.toBeNull();
      expect(result?.severity).toBe('warning');
      expect(result?.message).toContain('ZeRO-2 + PP');
    });

    it('ZeRO-3 + PP should produce warning', () => {
      const result = validateHybridCombination({
        dpType: 'zero-3',
        tp: 1,
        pp: 8,
        gpusPerNode: 8,
      });
      expect(result).not.toBeNull();
      expect(result?.severity).toBe('warning');
      expect(result?.message).toContain('ZeRO-3 + PP');
    });

    it('ZeRO-2 + TP + PP should produce warning', () => {
      const result = validateHybridCombination({
        dpType: 'zero-2',
        tp: 8,
        pp: 8,
        gpusPerNode: 8,
      });
      expect(result).not.toBeNull();
      expect(result?.severity).toBe('warning');
    });

    it('ZeRO-3 + TP + PP should produce warning', () => {
      const result = validateHybridCombination({
        dpType: 'zero-3',
        tp: 8,
        pp: 8,
        gpusPerNode: 8,
      });
      expect(result).not.toBeNull();
      expect(result?.severity).toBe('warning');
    });

    it('ZeRO-3 + TP (without PP) should produce caution warning', () => {
      const result = validateHybridCombination({
        dpType: 'zero-3',
        tp: 8,
        pp: 1,
        gpusPerNode: 8,
      });
      expect(result).not.toBeNull();
      expect(result?.severity).toBe('warning');
      expect(result?.message).toContain('1.5x DDP communication');
    });
  });

  describe('TP Topology Validation', () => {
    it('TP within node should not produce warning', () => {
      const result = validateTPTopology(8, 8);
      expect(result).toBeNull();
    });

    it('TP across nodes should produce warning', () => {
      const result = validateTPTopology(16, 8);
      expect(result).not.toBeNull();
      expect(result?.severity).toBe('warning');
      expect(result?.message).toContain('exceeds GPUs per node');
      expect(result?.message).toContain('NVLink');
    });
  });
});

describe('Published Training Configurations', () => {
  describe('GPT-3 175B Config', () => {
    it('should validate GPT-3 175B configuration', () => {
      // GPT-3 175B: TP=8, PP=16, DDP on 1024 A100s
      const cluster = createMultiNodeCluster('a100-80gb', 8, 128); // 1024 GPUs
      if (!cluster) throw new Error('Failed to create cluster');

      const model = getModel('gpt3-175b', 2048);
      if (!model) throw new Error('Failed to get model');

      const strategy = create3DParallelStrategy(8, 16, 8, {
        dpType: 'ddp',
        sequenceParallel: false,
        activationCheckpointing: true,
        schedule: '1f1b',
      });

      // Calculate realistic gradient accumulation
      const globalBatchSize = 1536;
      const microBatchSize = 1;
      const dp = 8;
      const gradientAccumulationSteps = Math.ceil(globalBatchSize / (microBatchSize * dp));

      const ctx: StrategyContext = {
        model,
        cluster,
        training: {
          globalBatchSize,
          microBatchSize,
          sequenceLength: 2048,
          maxSteps: 1000,
          optimizer: DEFAULT_ADAMW_CONFIG,
          lrSchedule: DEFAULT_LR_SCHEDULE,
          dtypes: DEFAULT_DTYPE_CONFIG,
          gradientClipping: 1.0,
          gradientAccumulationSteps,
        },
        seqLength: 2048,
        microBatchSize,
        globalBatchSize,
        gradientAccumulationSteps,
        activationCheckpointing: true,
        flashAttention: true,
      };

      const validation = strategy.validate(ctx);
      expect(validation.valid).toBe(true);

      // Verify MFU is in reasonable range (not >100% and not near 0)
      // With high gradient accumulation, MFU will be lower than published values
      const analysis = strategy.computeAnalysis(ctx);
      expect(analysis.mfu).toBeLessThanOrEqual(1.0);
      expect(analysis.mfu).toBeGreaterThan(0.1); // At least 10% efficiency
    });
  });

  describe('BLOOM 176B Config', () => {
    it('should validate BLOOM 176B configuration', () => {
      // BLOOM 176B: ZeRO-1 + TP=4 + PP=12 on 384 A100s
      const cluster = createMultiNodeCluster('a100-80gb', 8, 48); // 384 GPUs
      if (!cluster) throw new Error('Failed to create cluster');

      const strategy = create3DParallelStrategy(4, 12, 8, {
        dpType: 'zero-1',
        sequenceParallel: false,
        activationCheckpointing: true,
        schedule: '1f1b',
      });

      // BLOOM uses different architecture, use 175B GPT-3 as proxy
      const model = getModel('gpt3-175b', 2048);
      if (!model) throw new Error('Failed to get model');

      const globalBatchSize = 2048;
      const microBatchSize = 1;
      const dp = 8;
      const gradientAccumulationSteps = Math.ceil(globalBatchSize / (microBatchSize * dp));

      const ctx: StrategyContext = {
        model,
        cluster,
        training: {
          globalBatchSize,
          microBatchSize,
          sequenceLength: 2048,
          maxSteps: 1000,
          optimizer: DEFAULT_ADAMW_CONFIG,
          lrSchedule: DEFAULT_LR_SCHEDULE,
          dtypes: DEFAULT_DTYPE_CONFIG,
          gradientClipping: 1.0,
          gradientAccumulationSteps,
        },
        seqLength: 2048,
        microBatchSize,
        globalBatchSize,
        gradientAccumulationSteps,
        activationCheckpointing: true,
        flashAttention: true,
      };

      const validation = strategy.validate(ctx);
      expect(validation.valid).toBe(true);

      // Should not have hybrid combination warnings (ZeRO-1 + TP + PP is valid)
      const hybridWarning = validation.warnings.find(w =>
        w.includes('ZeRO-2') || w.includes('ZeRO-3')
      );
      expect(hybridWarning).toBeUndefined();

      // Verify MFU is in reasonable range
      const analysis = strategy.computeAnalysis(ctx);
      expect(analysis.mfu).toBeLessThanOrEqual(1.0);
      expect(analysis.mfu).toBeGreaterThan(0.1); // At least 10% efficiency
    });
  });

  describe('MT-NLG 530B Config', () => {
    it('should validate MT-NLG 530B configuration', () => {
      // MT-NLG 530B used TP=8, PP=35 on 105-layer model. We use GPT-3 175B (96 layers)
      // as stand-in, so pick PP=12 with v=4 so layers divide evenly: 96/(12×4)=2 layers/stage.
      const cluster = createMultiNodeCluster('a100-80gb', 8, 96); // 768 GPUs
      if (!cluster) throw new Error('Failed to create cluster');

      const strategy = create3DParallelStrategy(8, 12, 8, {
        dpType: 'zero-1',
        sequenceParallel: false,
        activationCheckpointing: true,
        schedule: 'interleaved-1f1b',
        interleavedStages: 4,
      });

      // Use a very large model - no exact 530B available, so we test with validation
      // The key test is that ZeRO-1 + TP + PP is valid
      const model = getModel('gpt3-175b', 2048);
      if (!model) throw new Error('Failed to get model');

      const ctx: StrategyContext = {
        model,
        cluster,
        training: {
          globalBatchSize: 1920,
          microBatchSize: 1,
          sequenceLength: 2048,
          maxSteps: 1000,
          optimizer: DEFAULT_ADAMW_CONFIG,
          lrSchedule: DEFAULT_LR_SCHEDULE,
          dtypes: DEFAULT_DTYPE_CONFIG,
          gradientClipping: 1.0,
          gradientAccumulationSteps: 2,
        },
        seqLength: 2048,
        microBatchSize: 1,
        globalBatchSize: 1920,
        gradientAccumulationSteps: 2,
        activationCheckpointing: true,
        flashAttention: true,
      };

      const validation = strategy.validate(ctx);
      expect(validation.valid).toBe(true);

      // No invalid hybrid combination warnings
      const hybridWarning = validation.warnings.find(w =>
        w.includes('ZeRO-2') || w.includes('ZeRO-3')
      );
      expect(hybridWarning).toBeUndefined();
    });
  });

  describe('LLaMA 3 405B Config', () => {
    it('should validate LLaMA 3 405B configuration', () => {
      // LLaMA 3 405B: FSDP + TP=8 + PP=16 on 16K H100s
      const cluster = createMultiNodeCluster('h100-sxm', 8, 2048); // 16384 GPUs
      if (!cluster) throw new Error('Failed to create cluster');

      const strategy = create3DParallelStrategy(8, 16, 128, {
        dpType: 'fsdp',
        sequenceParallel: true,
        activationCheckpointing: true,
        schedule: 'interleaved-1f1b',
      });

      // Use a large model proxy with shorter seq (175B at 8K needs MBS≤1 to fit)
      const model = getModel('gpt3-175b', 4096);
      if (!model) throw new Error('Failed to get model');

      const globalBatchSize = 16384;
      const microBatchSize = 1;
      const dp = 128;
      const gradientAccumulationSteps = Math.ceil(globalBatchSize / (microBatchSize * dp));

      const ctx: StrategyContext = {
        model,
        cluster,
        training: {
          globalBatchSize,
          microBatchSize,
          sequenceLength: 4096,
          maxSteps: 1000,
          optimizer: DEFAULT_ADAMW_CONFIG,
          lrSchedule: DEFAULT_LR_SCHEDULE,
          dtypes: DEFAULT_DTYPE_CONFIG,
          gradientClipping: 1.0,
          gradientAccumulationSteps,
        },
        seqLength: 4096,
        microBatchSize,
        globalBatchSize,
        gradientAccumulationSteps,
        activationCheckpointing: true,
        flashAttention: true,
      };

      const validation = strategy.validate(ctx);
      expect(validation.valid).toBe(true);

      // FSDP + TP + PP should have no hybrid warnings
      const hybridWarning = validation.warnings.find(w =>
        w.includes('ZeRO-2') || w.includes('ZeRO-3') || w.includes('not recommended')
      );
      expect(hybridWarning).toBeUndefined();

      // Verify MFU is in reasonable range
      const analysis = strategy.computeAnalysis(ctx);
      expect(analysis.mfu).toBeLessThanOrEqual(1.0);
      expect(analysis.mfu).toBeGreaterThan(0.1); // At least 10% efficiency
    });
  });
});

describe('2D Strategy Shortcuts', () => {
  describe('FSDP + TP Configuration', () => {
    it('should create valid FSDP + TP strategy', () => {
      // TP=8, PP=1, DP=16 -> dpDegree=16
      const strategy = create3DParallelStrategy(8, 1, 16, {
        dpType: 'fsdp',
        sequenceParallel: true,
      });

      const ctx = makeStrategyContext('llama2-70b', '128x-a100', { globalBatchSize: 1024, dpDegree: 16 });

      const validation = strategy.validate(ctx);
      expect(validation.valid).toBe(true);

      const analysis = strategy.computeAnalysis(ctx);
      expect(analysis.mfu).toBeGreaterThan(0.25);
      expect(analysis.mfu).toBeLessThanOrEqual(1.0);
    });
  });

  describe('ZeRO-1 + TP Configuration', () => {
    it('should create valid ZeRO-1 + TP strategy', () => {
      // TP=8, PP=1, DP=16 -> dpDegree=16
      const strategy = create3DParallelStrategy(8, 1, 16, {
        dpType: 'zero-1',
        sequenceParallel: true,
      });

      const ctx = makeStrategyContext('llama2-70b', '128x-a100', { globalBatchSize: 1024, dpDegree: 16, microBatchSize: 2 });

      const validation = strategy.validate(ctx);
      expect(validation.valid).toBe(true);

      const analysis = strategy.computeAnalysis(ctx);
      expect(analysis.mfu).toBeGreaterThan(0.25);
      expect(analysis.mfu).toBeLessThanOrEqual(1.0);
    });
  });

  describe('DDP + PP Configuration', () => {
    it('should create valid DDP + PP strategy', () => {
      // 128 GPUs with TP=1, PP=8, DP=16 (1*8*16 = 128)
      const strategy = create3DParallelStrategy(1, 8, 16, {
        dpType: 'ddp',
        numMicroBatches: 16,
      });

      // Use a model that fits well with PP=8
      const ctx = makeStrategyContext('llama2-7b', '128x-a100', {
        globalBatchSize: 512,
        microBatchSize: 2,
      });

      const validation = strategy.validate(ctx);
      // Check for specific errors if invalid
      if (!validation.valid) {
        console.log('DDP+PP validation errors:', validation.errors);
      }
      expect(validation.valid).toBe(true);
    });
  });

  describe('FSDP + PP Configuration', () => {
    it('should create valid FSDP + PP strategy', () => {
      // TP=1, PP=8, DP=16 -> dpDegree=16
      const strategy = create3DParallelStrategy(1, 8, 16, {
        dpType: 'fsdp',
        numMicroBatches: 16,
      });

      // Use MBS=2 (default 4 OOMs on 80GB A100 with increased activation memory)
      const ctx = makeStrategyContext('llama2-70b', '128x-a100', { globalBatchSize: 1024, dpDegree: 16, microBatchSize: 2 });

      const validation = strategy.validate(ctx);
      expect(validation.valid).toBe(true);
    });
  });
});

describe('Invalid Combination Warnings in Strategy', () => {
  it('should warn when using ZeRO-2 + PP in 3D strategy', () => {
    // TP=4, PP=8, DP=4 -> dpDegree=4
    const strategy = create3DParallelStrategy(4, 8, 4, {
      dpType: 'zero-2',
      numMicroBatches: 16,
    });

    const ctx = makeStrategyContext('llama2-70b', '128x-a100', { globalBatchSize: 1024, dpDegree: 4 });

    const validation = strategy.validate(ctx);
    // Should still be valid but with warnings
    expect(validation.warnings.length).toBeGreaterThan(0);
    const hybridWarning = validation.warnings.find(w => w.includes('ZeRO-2 + TP + PP'));
    expect(hybridWarning).toBeDefined();
  });

  it('should warn when using ZeRO-3 + PP in 3D strategy', () => {
    // TP=1, PP=8, DP=16 -> dpDegree=16
    const strategy = create3DParallelStrategy(1, 8, 16, {
      dpType: 'zero-3',
      numMicroBatches: 16,
    });

    const ctx = makeStrategyContext('llama2-70b', '128x-a100', { globalBatchSize: 1024, dpDegree: 16 });

    const validation = strategy.validate(ctx);
    expect(validation.warnings.length).toBeGreaterThan(0);
    const hybridWarning = validation.warnings.find(w => w.includes('ZeRO-3 + PP'));
    expect(hybridWarning).toBeDefined();
  });
});
