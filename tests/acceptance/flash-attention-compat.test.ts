/**
 * Flash Attention GPU Compatibility Tests
 *
 * Flash Attention requires Ampere+ (NVIDIA sm_80+) or CDNA2+ (AMD ROCm).
 * Pre-Ampere GPUs (Turing T4, Volta V100) do NOT support Flash Attention.
 *
 * Tests cover:
 * 1. supportsFlashAttention() returns correct values for all GPU architectures
 * 2. Engine forces flashAttention=false on incompatible GPUs
 * 3. OOM suggestions don't recommend Flash Attention on incompatible GPUs
 * 4. OOM suggestions DO recommend Flash Attention on compatible GPUs
 * 5. Inference engine guards Flash Attention on incompatible GPUs
 */

import { describe, it, expect } from 'vitest';
import {
  T4,
  V100_32GB,
  A100_80GB,
  A100_40GB,
  H100_SXM,
  H100_PCIE,
  H200_SXM,
  B200,
  GB200,
  MI250X,
  MI300X,
  MI325X,
  MI350X,
  L40S,
  L40,
  L4,
  A10,
  A10G,
  RTX_6000_ADA,
  RTX_4090,
  RTX_3090,
  ALL_GPUS,
  supportsFlashAttention,
} from '../../src/core/hardware/gpu.ts';
import { SimulationEngine } from '../../src/core/simulation/engine.ts';
import { assertValidEngine } from '../helpers/validated-metrics.ts';
import type { SimulationConfig } from '../../src/core/simulation/engine.ts';
import { getModel } from '../../src/core/models/index.ts';
import { createMultiNodeCluster } from '../../src/core/hardware/topology.ts';

// ============================================================================
// Section 1: supportsFlashAttention() for all GPU architectures
// ============================================================================

describe('supportsFlashAttention', () => {
  describe('incompatible GPUs (pre-Ampere)', () => {
    it('T4 (Turing) does NOT support Flash Attention', () => {
      expect(supportsFlashAttention(T4)).toBe(false);
    });

    it('V100 (Volta) does NOT support Flash Attention', () => {
      expect(supportsFlashAttention(V100_32GB)).toBe(false);
    });
  });

  describe('compatible NVIDIA GPUs (Ampere+)', () => {
    it('A100 40GB (Ampere) supports Flash Attention', () => {
      expect(supportsFlashAttention(A100_40GB)).toBe(true);
    });

    it('A100 80GB (Ampere) supports Flash Attention', () => {
      expect(supportsFlashAttention(A100_80GB)).toBe(true);
    });

    it('A10 (Ampere) supports Flash Attention', () => {
      expect(supportsFlashAttention(A10)).toBe(true);
    });

    it('A10G (Ampere) supports Flash Attention', () => {
      expect(supportsFlashAttention(A10G)).toBe(true);
    });

    it('RTX 3090 (Ampere) supports Flash Attention', () => {
      expect(supportsFlashAttention(RTX_3090)).toBe(true);
    });

    it('H100 SXM (Hopper) supports Flash Attention', () => {
      expect(supportsFlashAttention(H100_SXM)).toBe(true);
    });

    it('H100 PCIe (Hopper) supports Flash Attention', () => {
      expect(supportsFlashAttention(H100_PCIE)).toBe(true);
    });

    it('H200 (Hopper) supports Flash Attention', () => {
      expect(supportsFlashAttention(H200_SXM)).toBe(true);
    });

    it('B200 (Blackwell) supports Flash Attention', () => {
      expect(supportsFlashAttention(B200)).toBe(true);
    });

    it('GB200 (Blackwell) supports Flash Attention', () => {
      expect(supportsFlashAttention(GB200)).toBe(true);
    });

    it('L40S (Ada) supports Flash Attention', () => {
      expect(supportsFlashAttention(L40S)).toBe(true);
    });

    it('L40 (Ada) supports Flash Attention', () => {
      expect(supportsFlashAttention(L40)).toBe(true);
    });

    it('L4 (Ada) supports Flash Attention', () => {
      expect(supportsFlashAttention(L4)).toBe(true);
    });

    it('RTX 4090 (Ada) supports Flash Attention', () => {
      expect(supportsFlashAttention(RTX_4090)).toBe(true);
    });

    it('RTX 6000 Ada supports Flash Attention', () => {
      expect(supportsFlashAttention(RTX_6000_ADA)).toBe(true);
    });
  });

  describe('compatible AMD GPUs (CDNA2+)', () => {
    it('MI250X (CDNA2) supports Flash Attention', () => {
      expect(supportsFlashAttention(MI250X)).toBe(true);
    });

    it('MI300X (CDNA3) supports Flash Attention', () => {
      expect(supportsFlashAttention(MI300X)).toBe(true);
    });

    it('MI325X (CDNA4) supports Flash Attention', () => {
      expect(supportsFlashAttention(MI325X)).toBe(true);
    });

    it('MI350X (CDNA4) supports Flash Attention', () => {
      expect(supportsFlashAttention(MI350X)).toBe(true);
    });
  });

  it('all GPUs in catalog have defined flash attention support', () => {
    for (const [_id, gpu] of Object.entries(ALL_GPUS)) {
      const result = supportsFlashAttention(gpu);
      expect(typeof result).toBe('boolean');
      // T4 and V100 should be false, everything else true
      if (gpu.architecture === 'turing' || gpu.architecture === 'volta') {
        expect(result).toBe(false);
      } else {
        expect(result).toBe(true);
      }
    }
  });
});

// ============================================================================
// Section 2: Engine guards Flash Attention on incompatible GPUs
// ============================================================================

describe('Engine Flash Attention guard', () => {
  function makeConfig(gpuId: string, flashAttention: boolean): SimulationConfig {
    const model = getModel('gpt3-125m', 1024)!;
    const cluster = createMultiNodeCluster(gpuId, 1, 1)!;
    return {
      modelSpec: model,
      clusterConfig: cluster,
      globalBatchSize: 8,
      microBatchSize: 1,
      sequenceLength: 1024,
      maxSteps: 1,
      strategyType: 'ddp',
      flashAttention,
      activationCheckpointing: true,
      mixedPrecision: 'fp16',
    };
  }

  it('T4 with flashAttention=true still computes without FA benefit', () => {
    const configWithFA = makeConfig('t4', true);
    const configWithoutFA = makeConfig('t4', false);

    const engineWith = new SimulationEngine();
    engineWith.configure(configWithFA);
    assertValidEngine(engineWith);
    const metricsWith = engineWith.simulate();

    const engineWithout = new SimulationEngine();
    engineWithout.configure(configWithoutFA);
    assertValidEngine(engineWithout);
    const metricsWithout = engineWithout.simulate();

    // Both should produce same memory (FA forced off on T4)
    expect(metricsWith.memoryPerGPU.total).toBe(metricsWithout.memoryPerGPU.total);
  });

  it('V100 with flashAttention=true still computes without FA benefit', () => {
    const configWithFA = makeConfig('v100-32gb', true);
    const configWithoutFA = makeConfig('v100-32gb', false);

    const engineWith = new SimulationEngine();
    engineWith.configure(configWithFA);
    assertValidEngine(engineWith);
    const metricsWith = engineWith.simulate();

    const engineWithout = new SimulationEngine();
    engineWithout.configure(configWithoutFA);
    assertValidEngine(engineWithout);
    const metricsWithout = engineWithout.simulate();

    // Both should produce same memory (FA forced off on V100)
    expect(metricsWith.memoryPerGPU.total).toBe(metricsWithout.memoryPerGPU.total);
  });

  it('A100 with flashAttention=true vs false shows memory difference', () => {
    const configWithFA = makeConfig('a100-80gb', true);
    const configWithoutFA = makeConfig('a100-80gb', false);

    const engineWith = new SimulationEngine();
    engineWith.configure(configWithFA);
    assertValidEngine(engineWith);
    const metricsWith = engineWith.simulate();

    const engineWithout = new SimulationEngine();
    engineWithout.configure(configWithoutFA);
    assertValidEngine(engineWithout);
    const metricsWithout = engineWithout.simulate();

    // FA on A100 should reduce memory (O(seq²) → O(seq) for attention)
    expect(metricsWith.memoryPerGPU.total).toBeLessThan(metricsWithout.memoryPerGPU.total);
  });

  it('H100 with flashAttention=true vs false shows memory difference', () => {
    const configWithFA = makeConfig('h100-sxm', true);
    const configWithoutFA = makeConfig('h100-sxm', false);

    const engineWith = new SimulationEngine();
    engineWith.configure(configWithFA);
    assertValidEngine(engineWith);
    const metricsWith = engineWith.simulate();

    const engineWithout = new SimulationEngine();
    engineWithout.configure(configWithoutFA);
    assertValidEngine(engineWithout);
    const metricsWithout = engineWithout.simulate();

    expect(metricsWith.memoryPerGPU.total).toBeLessThan(metricsWithout.memoryPerGPU.total);
  });
});

// ============================================================================
// Section 3: OOM suggestions respect GPU compatibility
// ============================================================================

describe('OOM suggestions and Flash Attention', () => {
  function runOOMSimulation(gpuId: string): string[] {
    // Use a large model that will OOM on a single small GPU
    const model = getModel('llama2-7b', 4096)!;
    const cluster = createMultiNodeCluster(gpuId, 1, 1)!;

    const config: SimulationConfig = {
      modelSpec: model,
      clusterConfig: cluster,
      globalBatchSize: 32,
      microBatchSize: 4,
      sequenceLength: 4096,
      maxSteps: 1,
      strategyType: 'ddp',
      flashAttention: false,
      activationCheckpointing: false,
      mixedPrecision: 'bf16',
    };

    const engine = new SimulationEngine();
    engine.configure(config);
    const result = engine.run();

    return result.analysis?.recommendations ?? [];
  }

  it('T4 OOM does NOT suggest Flash Attention', () => {
    const suggestions = runOOMSimulation('t4');
    const faRecommendation = suggestions.find(s => s.includes('Flash Attention'));
    expect(faRecommendation).toBeUndefined();
  });

  it('V100 OOM does NOT suggest Flash Attention', () => {
    const suggestions = runOOMSimulation('v100-32gb');
    const faRecommendation = suggestions.find(s => s.includes('Flash Attention'));
    expect(faRecommendation).toBeUndefined();
  });

  it('A100 40GB OOM DOES suggest Flash Attention', () => {
    const suggestions = runOOMSimulation('a100-40gb');
    const faRecommendation = suggestions.find(s => s.includes('Flash Attention'));
    expect(faRecommendation).toBeDefined();
  });

  it('L4 OOM DOES suggest Flash Attention', () => {
    const suggestions = runOOMSimulation('l4');
    const faRecommendation = suggestions.find(s => s.includes('Flash Attention'));
    expect(faRecommendation).toBeDefined();
  });
});

// ============================================================================
// Section 4: All strategies gate FA suggestions on GPU compatibility
// ============================================================================

describe('All strategies gate FA suggestions', () => {
  // Test with a config that causes OOM, for strategies that have OOM paths
  const incompatibleGPUs = ['t4', 'v100-32gb'];
  const compatibleGPUs = ['a100-40gb', 'l4'];

  for (const gpuId of incompatibleGPUs) {
    it(`${gpuId}: no FA suggestion in any strategy OOM path`, () => {
      const strategies: SimulationConfig['strategyType'][] = ['ddp', 'fsdp', 'zero-1'];
      const model = getModel('llama2-7b', 4096)!;
      const cluster = createMultiNodeCluster(gpuId, 1, 1)!;

      for (const strategyType of strategies) {
        const config: SimulationConfig = {
          modelSpec: model,
          clusterConfig: cluster,
          globalBatchSize: 32,
          microBatchSize: 4,
          sequenceLength: 4096,
          maxSteps: 1,
          strategyType,
          flashAttention: false,
          activationCheckpointing: false,
          mixedPrecision: 'fp16',
        };

        const engine = new SimulationEngine();
        engine.configure(config);
        const result = engine.run();

        const suggestions = result.analysis?.recommendations ?? [];
        const faRec = suggestions.find(s => s.includes('Flash Attention'));
        expect(faRec).toBeUndefined();
      }
    });
  }

  for (const gpuId of compatibleGPUs) {
    it(`${gpuId}: FA suggestion present when flashAttention=false and OOM`, () => {
      const model = getModel('llama2-7b', 4096)!;
      const cluster = createMultiNodeCluster(gpuId, 1, 1)!;

      const config: SimulationConfig = {
        modelSpec: model,
        clusterConfig: cluster,
        globalBatchSize: 32,
        microBatchSize: 4,
        sequenceLength: 4096,
        maxSteps: 1,
        strategyType: 'ddp',
        flashAttention: false,
        activationCheckpointing: false,
        mixedPrecision: 'bf16',
      };

      const engine = new SimulationEngine();
      engine.configure(config);
      const result = engine.run();

      const suggestions = result.analysis?.recommendations ?? [];
      const faRec = suggestions.find(s => s.includes('Flash Attention'));
      expect(faRec).toBeDefined();
    });
  }
});
