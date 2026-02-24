/**
 * Cross-Node TP Tests
 *
 * Validates that TP > gpusPerNode (cross-node TP) works correctly:
 * - Simulation runs without errors (validation produces warning, not error)
 * - Uses inter-node bandwidth instead of NVLink for TP comm
 * - MFU is lower than intra-node TP due to bandwidth penalty
 */

import { describe, it, expect } from 'vitest';
import { SimulationEngine } from '../../src/core/simulation/engine.ts';
import { createMultiNodeCluster } from '../../src/core/hardware/topology.ts';
import { getModel } from '../../src/core/models/index.ts';

function makeEngine(config: Parameters<SimulationEngine['configure']>[0]): SimulationEngine {
  const engine = new SimulationEngine();
  engine.configure(config);
  return engine;
}

describe('Cross-Node TP', () => {

  describe('Validation', () => {
    it('should produce warning (not error) for TP > gpusPerNode', () => {
      const cluster = createMultiNodeCluster('h100-sxm', 8, 2)!; // 16 GPUs, 2 nodes
      const engine = makeEngine({
        modelSpec: getModel('llama3.1-8b', 4096)!,
        clusterConfig: cluster,
        globalBatchSize: 16,
        microBatchSize: 1,
        sequenceLength: 4096,
        strategyType: 'fsdp-tp',
        strategyConfig: { tp: 16 },
        activationCheckpointing: true,
        flashAttention: true,
        mixedPrecision: 'bf16',
      });

      const validation = engine.validate();
      expect(validation.errors).toHaveLength(0);
      const warningText = validation.warnings.join(' ');
      expect(warningText).toMatch(/exceeds GPUs per node/i);
    });

    it('should still error for genuinely invalid configs (heads not divisible)', () => {
      // Llama 3 70B has 64 heads — TP=3 doesn't divide evenly
      const cluster = createMultiNodeCluster('h100-sxm', 8, 2)!;
      const engine = makeEngine({
        modelSpec: getModel('llama3-70b', 4096)!,
        clusterConfig: cluster,
        globalBatchSize: 16,
        microBatchSize: 1,
        sequenceLength: 4096,
        strategyType: 'fsdp-tp',
        strategyConfig: { tp: 3 },
        activationCheckpointing: true,
        flashAttention: true,
        mixedPrecision: 'bf16',
      });

      const validation = engine.validate();
      expect(validation.errors.length).toBeGreaterThan(0);
      expect(validation.errors.join(' ')).toMatch(/attention heads/i);
    });
  });

  describe('Bandwidth penalty', () => {
    it('TP=16 cross-node should have lower MFU than TP=8 intra-node', () => {
      const model = getModel('llama3.1-8b', 4096)!;

      // TP=8 intra-node (1 node, 8 GPUs) — uses NVLink
      const cluster8 = createMultiNodeCluster('h100-sxm', 8, 1)!;
      const engine8 = makeEngine({
        modelSpec: model,
        clusterConfig: cluster8,
        globalBatchSize: 8,
        microBatchSize: 1,
        sequenceLength: 4096,
        strategyType: 'fsdp-tp',
        strategyConfig: { tp: 8 },
        activationCheckpointing: true,
        flashAttention: true,
        mixedPrecision: 'bf16',
      });

      const val8 = engine8.validate();
      expect(val8.errors).toHaveLength(0);
      const metrics8 = engine8.simulate();

      // TP=16 cross-node (2 nodes, 16 GPUs) — uses IB
      const cluster16 = createMultiNodeCluster('h100-sxm', 8, 2)!;
      const engine16 = makeEngine({
        modelSpec: model,
        clusterConfig: cluster16,
        globalBatchSize: 16,
        microBatchSize: 1,
        sequenceLength: 4096,
        strategyType: 'fsdp-tp',
        strategyConfig: { tp: 16 },
        activationCheckpointing: true,
        flashAttention: true,
        mixedPrecision: 'bf16',
      });

      const val16 = engine16.validate();
      expect(val16.errors).toHaveLength(0);
      const metrics16 = engine16.simulate();

      // Cross-node TP should have lower MFU due to hierarchical all-reduce
      // (NVLink RS + IB AR + NVLink AG) vs pure NVLink ring all-reduce
      // Actual: TP=8 ~28.9%, TP=16 ~19.2%
      expect(metrics16.mfu).toBeLessThan(metrics8.mfu);
      expect(metrics8.mfu).toBeGreaterThan(0.245);
      expect(metrics8.mfu).toBeLessThan(0.333);
      expect(metrics16.mfu).toBeGreaterThan(0.15);
      expect(metrics16.mfu).toBeLessThan(0.26);
    });

    it('70B model: TP=16 cross-node vs TP=8 intra-node with FSDP', () => {
      const model = getModel('llama3-70b', 4096)!;
      const cluster = createMultiNodeCluster('h100-sxm', 8, 8)!; // 64 GPUs

      // TP=8 intra-node, DP=8
      const engine8 = makeEngine({
        modelSpec: model,
        clusterConfig: cluster,
        globalBatchSize: 64,
        microBatchSize: 1,
        sequenceLength: 4096,
        strategyType: 'fsdp-tp',
        strategyConfig: { tp: 8 },
        activationCheckpointing: true,
        flashAttention: true,
        mixedPrecision: 'bf16',
      });
      expect(engine8.validate().errors).toHaveLength(0);
      const metrics8 = engine8.simulate();

      // TP=16 cross-node, DP=4
      const engine16 = makeEngine({
        modelSpec: model,
        clusterConfig: cluster,
        globalBatchSize: 64,
        microBatchSize: 1,
        sequenceLength: 4096,
        strategyType: 'fsdp-tp',
        strategyConfig: { tp: 16 },
        activationCheckpointing: true,
        flashAttention: true,
        mixedPrecision: 'bf16',
      });
      expect(engine16.validate().errors).toHaveLength(0);
      const metrics16 = engine16.simulate();

      // TP=16 should be worse: hierarchical all-reduce is slower + DP halved
      // Actual: TP=8 ~35.9%, TP=16 ~22.7%
      expect(metrics16.mfu).toBeLessThan(metrics8.mfu);
      expect(metrics8.mfu).toBeGreaterThan(0.305);
      expect(metrics8.mfu).toBeLessThan(0.413);
      expect(metrics16.mfu).toBeGreaterThan(0.19);
      expect(metrics16.mfu).toBeLessThan(0.27);
    });
  });

  describe('3D parallel with cross-node TP', () => {
    it('should simulate fsdp-tp-pp with TP=16 across nodes', () => {
      const model = getModel('llama3-70b', 4096)!;
      const cluster = createMultiNodeCluster('h100-sxm', 8, 16)!; // 128 GPUs
      const engine = makeEngine({
        modelSpec: model,
        clusterConfig: cluster,
        globalBatchSize: 128,
        microBatchSize: 1,
        sequenceLength: 4096,
        strategyType: 'fsdp-tp-pp',
        strategyConfig: { tp: 16, pp: 2 },
        activationCheckpointing: true,
        flashAttention: true,
        mixedPrecision: 'bf16',
      });

      const validation = engine.validate();
      expect(validation.errors).toHaveLength(0);
      expect(validation.warnings.join(' ')).toMatch(/exceeds GPUs per node/i);

      const metrics = engine.simulate();
      // Actual: ~26.3%
      expect(metrics.mfu).toBeGreaterThan(0.23);
      expect(metrics.mfu).toBeLessThan(0.30);
    });
  });

  describe('Regression: intra-node TP unchanged', () => {
    it('TP=8 within node should not produce cross-node warning', () => {
      const model = getModel('llama3.1-8b', 4096)!;
      const cluster = createMultiNodeCluster('h100-sxm', 8, 1)!;
      const engine = makeEngine({
        modelSpec: model,
        clusterConfig: cluster,
        globalBatchSize: 8,
        microBatchSize: 1,
        sequenceLength: 4096,
        strategyType: 'fsdp-tp',
        strategyConfig: { tp: 8 },
        activationCheckpointing: true,
        flashAttention: true,
        mixedPrecision: 'bf16',
      });

      const validation = engine.validate();
      expect(validation.errors).toHaveLength(0);
      // No cross-node warning for TP=8 with 8 GPUs/node
      const crossNodeWarning = validation.warnings.find(w => /exceeds GPUs per node/i.test(w));
      expect(crossNodeWarning).toBeUndefined();

      const metrics = engine.simulate();
      // Actual: ~31.2% — standard intra-node TP MFU
      expect(metrics.mfu).toBeGreaterThan(0.28);
      expect(metrics.mfu).toBeLessThan(0.36);
    });
  });
});
