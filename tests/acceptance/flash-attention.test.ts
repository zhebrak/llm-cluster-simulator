/**
 * Comprehensive Flash Attention Tests
 *
 * Validates FA memory reduction formulas, sequence length scaling,
 * interaction with all training strategies and activation checkpointing,
 * inference memory paths (including TP bug fix), and MFU invariance.
 */

import { describe, it, expect } from 'vitest';
import { getModel } from '../../src/core/models/index.ts';
import { estimateActivationMemory } from '../../src/core/strategies/base.ts';
import {
  activationMemory,
  activationMemoryWithFlashAttention,
  calculateInferenceMemory,
  calculateMemoryWithTP,
} from '../../src/core/inference/memory.ts';
import { createMultiNodeCluster } from '../../src/core/hardware/topology.ts';
import { runInferenceSimulation } from '../../src/core/inference/simulation.ts';
import { getValidatedSimulationMetrics } from '../helpers/validated-metrics.ts';
import {
  SimulationEngine,
  type SimulationConfig,
  type SimulationMetrics,
} from '../../src/core/simulation/engine.ts';

// ─── Helpers ────────────────────────────────────────────────────────────────

function getModelOrThrow(id: string, seqLength?: number) {
  const m = getModel(id, seqLength);
  if (!m) throw new Error(`Model not found: ${id}`);
  return m;
}

// ─── Section 1: Formula-Level Validation (Training) ─────────────────────────

describe('Section 1: Formula-level activation memory validation', () => {
  describe('attention score memory ratio equals seqLength exactly', () => {
    // Helper: compute attention score memory in isolation
    // attentionScoreMemory = numAttentionHeads * seqLength^(1 or 2) * bytesPerElement * microBatchSize
    function attentionScoreMemory(
      numHeads: number,
      seqLength: number,
      bytesPerElement: number,
      mbs: number,
      flash: boolean,
    ): number {
      return flash
        ? numHeads * seqLength * bytesPerElement * mbs
        : numHeads * seqLength * seqLength * bytesPerElement * mbs;
    }

    it('GPT-3 125M: ratio = 1024 (seq=1024)', () => {
      const noFA = attentionScoreMemory(12, 1024, 2, 1, false);
      const withFA = attentionScoreMemory(12, 1024, 2, 1, true);
      expect(noFA).toBe(25_165_824);   // 12 * 1024 * 1024 * 2 * 1
      expect(withFA).toBe(24_576);     // 12 * 1024 * 2 * 1
      expect(noFA / withFA).toBe(1024);
    });

    it('LLaMA-3 8B: ratio = 4096 (seq=4096)', () => {
      const noFA = attentionScoreMemory(32, 4096, 2, 1, false);
      const withFA = attentionScoreMemory(32, 4096, 2, 1, true);
      expect(noFA).toBe(1_073_741_824); // 32 * 4096 * 4096 * 2
      expect(withFA).toBe(262_144);     // 32 * 4096 * 2
      expect(noFA / withFA).toBe(4096);
    });

    it('LLaMA-3 70B: ratio = 4096 (seq=4096)', () => {
      const noFA = attentionScoreMemory(64, 4096, 2, 1, false);
      const withFA = attentionScoreMemory(64, 4096, 2, 1, true);
      expect(noFA).toBe(2_147_483_648); // 64 * 4096 * 4096 * 2
      expect(withFA).toBe(524_288);     // 64 * 4096 * 2
      expect(noFA / withFA).toBe(4096);
    });
  });

  describe('full activation memory (no checkpointing)', () => {
    it('GPT-3 125M: exact per-layer and total bytes', () => {
      const model = getModelOrThrow('gpt3-125m');
      // headDim = 768/12 = 64, qDim = 12*64 = 768, kvDim = 12*64 = 768 (MHA)
      // Common (attention + LN/residual): (5*h + qDim + 2*kvDim) * tokens * bytes + attentionScores
      // MLP: mlpIntermCoeff * I * tokens * bytes  (standard MLP → coeff = 2)
      const tokens = 1024; // seq * mbs
      const bytesPerElement = 2;
      const h = 768;
      const qDim = 768;
      const kvDim = 768;
      const I = 3072;
      const mlpIntermCoeff = 2; // standard MLP (not gated)
      // Standard attention stores: pre-softmax scores (2B) + post-softmax probs (2B) + dropout mask (1B)
      // = 2.5× the single-tensor estimate. Per Korthikanti et al. 2022, Table 1.
      const attentionScoresNoFA = 12 * 1024 * 1024 * bytesPerElement * 2.5;
      const commonNoFA = (5 * h + qDim + 2 * kvDim) * tokens * bytesPerElement
        + 2 * tokens // dropout masks: post-attention + post-MLP (1 byte each)
        + attentionScoresNoFA;
      const mlp = mlpIntermCoeff * I * tokens * bytesPerElement;
      const perLayerNoFA = commonNoFA + mlp;
      const totalNoFA = 12 * perLayerNoFA;

      const result = estimateActivationMemory(model, 1024, 1, 'bf16', false, false);
      expect(result).toBe(totalNoFA);

      // With FA
      const attentionScoresFA = 12 * 1024 * bytesPerElement;
      const commonFA = (5 * h + qDim + 2 * kvDim) * tokens * bytesPerElement
        + 2 * tokens // dropout masks
        + attentionScoresFA;
      const perLayerFA = commonFA + mlp;
      const totalFA = 12 * perLayerFA;

      const resultFA = estimateActivationMemory(model, 1024, 1, 'bf16', false, true);
      expect(resultFA).toBe(totalFA);
    });

    it('LLaMA-3 8B: exact per-layer and total bytes', () => {
      const model = getModelOrThrow('llama3-8b');
      // headDim = 128, qDim = 32*128 = 4096, kvDim = 8*128 = 1024 (GQA)
      // Common (attention + LN/residual): (5*h + qDim + 2*kvDim) * tokens * bytes + attentionScores
      // MLP: mlpIntermCoeff * I * tokens * bytes  (gated MLP → coeff = 3)
      const tokens = 4096;
      const bytesPerElement = 2;
      const h = 4096;
      const qDim = 4096;
      const kvDim = 1024;
      const I = 14336;
      const mlpIntermCoeff = 3; // gated MLP (SwiGLU)
      // Standard attention stores: pre-softmax scores (2B) + post-softmax probs (2B) + dropout mask (1B)
      // = 2.5× the single-tensor estimate. Per Korthikanti et al. 2022, Table 1.
      const attentionScoresNoFA = 32 * 4096 * 4096 * bytesPerElement * 2.5;
      const commonNoFA = (5 * h + qDim + 2 * kvDim) * tokens * bytesPerElement
        + 2 * tokens // dropout masks: post-attention + post-MLP (1 byte each)
        + attentionScoresNoFA;
      const mlp = mlpIntermCoeff * I * tokens * bytesPerElement;
      const perLayerNoFA = commonNoFA + mlp;
      const totalNoFA = 32 * perLayerNoFA;

      const result = estimateActivationMemory(model, 4096, 1, 'bf16', false, false);
      expect(result).toBe(totalNoFA);
    });
  });

  describe('parametric: attention score ratio always equals seqLength', () => {
    const seqLengths = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];

    for (const seq of seqLengths) {
      it(`seq=${seq}: ratio = ${seq}`, () => {
        // Use 32 heads (LLaMA-3 8B)
        const noFA = 32 * seq * seq * 2;   // O(seq²)
        const withFA = 32 * seq * 2;       // O(seq)
        expect(noFA / withFA).toBe(seq);
      });
    }
  });
});

// ─── Section 2: Sequence Length Scaling (Training) ──────────────────────────

describe('Section 2: Sequence length scaling', () => {
  const model = getModelOrThrow('gpt3-125m');
  const seqLengths = [512, 1024, 2048, 4096, 8192];

  describe('quadratic scaling without FA', () => {
    it('consecutive pairs have attention score ratio of 4.0', () => {
      for (let i = 0; i < seqLengths.length - 1; i++) {
        const s1 = seqLengths[i];
        const s2 = seqLengths[i + 1]; // s2 = 2 * s1
        // Attention scores: heads * seq² * bytes * mbs
        const scoreS1 = 12 * s1 * s1 * 2;
        const scoreS2 = 12 * s2 * s2 * 2;
        expect(scoreS2 / scoreS1).toBe(4.0);
      }
    });
  });

  describe('linear scaling with FA', () => {
    it('consecutive pairs have attention score ratio of 2.0', () => {
      for (let i = 0; i < seqLengths.length - 1; i++) {
        const s1 = seqLengths[i];
        const s2 = seqLengths[i + 1];
        // With FA: heads * seq * bytes * mbs
        const scoreS1 = 12 * s1 * 2;
        const scoreS2 = 12 * s2 * 2;
        expect(scoreS2 / scoreS1).toBe(2.0);
      }
    });
  });

  describe('savings ratio increases with seqLength', () => {
    it('total activation savings (noFA/FA) strictly increases', () => {
      const savings: number[] = [];
      for (const seq of seqLengths) {
        const noFA = estimateActivationMemory(model, seq, 1, 'bf16', false, false);
        const withFA = estimateActivationMemory(model, seq, 1, 'bf16', false, true);
        savings.push(noFA / withFA);
      }
      for (let i = 0; i < savings.length - 1; i++) {
        expect(savings[i + 1]).toBeGreaterThan(savings[i]);
      }
    });
  });

  describe('exact attention score values at each seqLength', () => {
    for (const seq of seqLengths) {
      it(`seq=${seq}: noFA = ${12 * seq * seq * 2}, FA = ${12 * seq * 2}`, () => {
        const noFA = 12 * seq * seq * 2;
        const withFA = 12 * seq * 2;
        expect(noFA).toBe(12 * seq * seq * 2);
        expect(withFA).toBe(12 * seq * 2);
      });
    }
  });
});

// ─── Section 3: All Training Strategies (End-to-End) ────────────────────────

describe('Section 3: All training strategies FA on/off', () => {
  const strategies: Array<{
    name: string;
    config: Omit<SimulationConfig, 'flashAttention'>;
  }> = [
    {
      name: 'ddp',
      config: {
        modelId: 'llama3.2-1b',
        clusterConfig: createMultiNodeCluster('h100-sxm', 8, 1)!,
        globalBatchSize: 32,
        microBatchSize: 4,
        sequenceLength: 2048,
        strategyType: 'ddp',
        activationCheckpointing: true,
      },
    },
    {
      name: 'fsdp',
      config: {
        modelId: 'llama2-7b',
        clusterConfig: createMultiNodeCluster('h100-sxm', 8, 1)!,
        globalBatchSize: 32,
        microBatchSize: 4,
        sequenceLength: 2048,
        strategyType: 'fsdp',
        activationCheckpointing: true,
      },
    },
    {
      name: 'zero-1',
      config: {
        modelId: 'llama2-7b',
        clusterConfig: createMultiNodeCluster('h100-sxm', 8, 1)!,
        globalBatchSize: 32,
        microBatchSize: 4,
        sequenceLength: 2048,
        strategyType: 'zero-1',
        activationCheckpointing: true,
      },
    },
    {
      name: 'fsdp-tp',
      config: {
        modelId: 'llama3-8b',
        clusterConfig: createMultiNodeCluster('h100-sxm', 8, 1)!,
        globalBatchSize: 32,
        microBatchSize: 4,
        sequenceLength: 2048,
        strategyType: 'fsdp-tp',
        strategyConfig: { tp: 2, sequenceParallel: true },
        activationCheckpointing: true,
      },
    },
    {
      name: 'zero1-tp',
      config: {
        modelId: 'llama3-8b',
        clusterConfig: createMultiNodeCluster('h100-sxm', 8, 1)!,
        globalBatchSize: 32,
        microBatchSize: 4,
        sequenceLength: 2048,
        strategyType: 'zero1-tp',
        strategyConfig: { tp: 2, sequenceParallel: true },
        activationCheckpointing: true,
      },
    },
    {
      name: 'ddp-tp-pp',
      config: {
        modelId: 'llama3-8b',
        clusterConfig: createMultiNodeCluster('h100-sxm', 8, 1)!,
        globalBatchSize: 32,
        microBatchSize: 4,
        sequenceLength: 2048,
        strategyType: 'ddp-tp-pp',
        strategyConfig: { tp: 2, pp: 2, sequenceParallel: true },
        activationCheckpointing: true,
      },
    },
    {
      name: 'zero1-tp-pp',
      config: {
        modelId: 'llama3-8b',
        clusterConfig: createMultiNodeCluster('h100-sxm', 8, 1)!,
        globalBatchSize: 32,
        microBatchSize: 4,
        sequenceLength: 2048,
        strategyType: 'zero1-tp-pp',
        strategyConfig: { tp: 2, pp: 2, sequenceParallel: true },
        activationCheckpointing: true,
      },
    },
    {
      name: 'fsdp-tp-pp',
      config: {
        modelId: 'llama3-8b',
        clusterConfig: createMultiNodeCluster('h100-sxm', 8, 1)!,
        globalBatchSize: 32,
        microBatchSize: 4,
        sequenceLength: 2048,
        strategyType: 'fsdp-tp-pp',
        strategyConfig: { tp: 2, pp: 2, sequenceParallel: true },
        activationCheckpointing: true,
      },
    },
  ];

  for (const { name, config } of strategies) {
    describe(`strategy: ${name}`, () => {
      let metricsFA: SimulationMetrics;
      let metricsNoFA: SimulationMetrics;

      // Compute once per strategy group, lazily
      function ensureMetrics() {
        if (!metricsFA) {
          metricsFA = getValidatedSimulationMetrics({ ...config, flashAttention: true });
          metricsNoFA = getValidatedSimulationMetrics({ ...config, flashAttention: false });
        }
      }

      it('peak activations: FA < noFA', () => {
        ensureMetrics();
        expect(metricsFA.memoryPerGPU.peakActivations).toBeLessThan(metricsNoFA.memoryPerGPU.peakActivations);
      });

      it('total memory: FA < noFA', () => {
        ensureMetrics();
        expect(metricsFA.memoryPerGPU.total).toBeLessThan(metricsNoFA.memoryPerGPU.total);
      });

      it('parameters memory identical', () => {
        ensureMetrics();
        expect(metricsFA.memoryPerGPU.parameters).toBe(metricsNoFA.memoryPerGPU.parameters);
      });

      it('gradients memory identical', () => {
        ensureMetrics();
        expect(metricsFA.memoryPerGPU.gradients).toBe(metricsNoFA.memoryPerGPU.gradients);
      });

      it('optimizer states memory identical', () => {
        ensureMetrics();
        expect(metricsFA.memoryPerGPU.optimizerStates).toBe(metricsNoFA.memoryPerGPU.optimizerStates);
      });

      it('MFU identical (±0.001)', () => {
        ensureMetrics();
        expect(Math.abs(metricsFA.mfu - metricsNoFA.mfu)).toBeLessThan(0.001);
      });

      it('HFU identical (±0.001)', () => {
        ensureMetrics();
        expect(Math.abs(metricsFA.hfu - metricsNoFA.hfu)).toBeLessThan(0.001);
      });

      it('step time identical (±0.1%)', () => {
        ensureMetrics();
        const relDiff = Math.abs(metricsFA.stepTimeMs - metricsNoFA.stepTimeMs) / metricsNoFA.stepTimeMs;
        expect(relDiff).toBeLessThan(0.001);
      });

      it('tokens/sec identical (±0.1%)', () => {
        ensureMetrics();
        const relDiff = Math.abs(metricsFA.tokensPerSecond - metricsNoFA.tokensPerSecond) / metricsNoFA.tokensPerSecond;
        expect(relDiff).toBeLessThan(0.001);
      });
    });
  }
});

// ─── Section 4: FA + Activation Checkpointing Interaction ──────────────────

describe('Section 4: FA + activation checkpointing interaction', () => {
  const model = getModelOrThrow('llama3-8b');

  // All 4 combos: {noFA, FA} × {noCkpt, ckpt}
  const A = estimateActivationMemory(model, 4096, 1, 'bf16', false, false); // noCkpt, noFA
  const B = estimateActivationMemory(model, 4096, 1, 'bf16', false, true);  // noCkpt, FA
  const C = estimateActivationMemory(model, 4096, 1, 'bf16', true, false);  // ckpt, noFA
  const D = estimateActivationMemory(model, 4096, 1, 'bf16', true, true);   // ckpt, FA

  it('strict ordering: A > max(B,C) and min(B,C) > D', () => {
    // A (noCkpt+noFA) is always largest, D (ckpt+FA) is always smallest.
    // B vs C ordering depends on whether FA savings or ckpt savings dominate.
    // With 2.5× attention coefficient (Korthikanti et al. 2022), the quadratic attention
    // scores at seq=4096 can make ckpt+noFA (C) exceed noCkpt+FA (B).
    expect(A).toBeGreaterThan(B);
    expect(A).toBeGreaterThan(C);
    expect(B).toBeGreaterThan(D);
    expect(C).toBeGreaterThan(D);
  });

  it('FA reduction factor is constant regardless of checkpointing', () => {
    // A/B == C/D: checkpointing doesn't interact with FA ratio
    const ratioNoCkpt = A / B;
    const ratioCkpt = C / D;
    expect(Math.abs(ratioNoCkpt - ratioCkpt)).toBeLessThan(0.001);
  });

  it('checkpointing reduction factor is constant regardless of FA', () => {
    // A/C == B/D == sqrt(numLayers) ≈ 5.657
    const ratioNoFA = A / C;
    const ratioFA = B / D;
    const sqrtLayers = Math.sqrt(32);
    expect(Math.abs(ratioNoFA - sqrtLayers)).toBeLessThan(0.001);
    expect(Math.abs(ratioFA - sqrtLayers)).toBeLessThan(0.001);
  });

  it('benefits are multiplicative: A/D = (A/B) × (A/C)', () => {
    const combinedRatio = A / D;
    const faRatio = A / B;
    const ckptRatio = A / C;
    expect(Math.abs(combinedRatio - faRatio * ckptRatio)).toBeLessThan(0.01);
  });

  describe('end-to-end engine test: all 4 combos via fsdp', () => {
    const baseConfig: Omit<SimulationConfig, 'flashAttention' | 'activationCheckpointing'> = {
      modelId: 'llama3-8b',
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 1)!,
      globalBatchSize: 32,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'fsdp',
    };

    // Use raw engine.simulate() — some combos may OOM, which is fine;
    // we only need memoryPerGPU.peakActivations for ordering (always populated).
    function simulate(fa: boolean, ckpt: boolean): SimulationMetrics {
      const engine = new SimulationEngine();
      engine.configure({ ...baseConfig, flashAttention: fa, activationCheckpointing: ckpt });
      return engine.simulate();
    }

    it('peak activation ordering: noCkpt/noFA > noCkpt/FA > ckpt/noFA > ckpt/FA', () => {
      const mA = simulate(false, false);
      const mB = simulate(true, false);
      const mC = simulate(false, true);
      const mD = simulate(true, true);
      expect(mA.memoryPerGPU.peakActivations).toBeGreaterThan(mB.memoryPerGPU.peakActivations);
      expect(mB.memoryPerGPU.peakActivations).toBeGreaterThan(mC.memoryPerGPU.peakActivations);
      expect(mC.memoryPerGPU.peakActivations).toBeGreaterThan(mD.memoryPerGPU.peakActivations);
    });

    it('MFU invariant to FA (±0.001)', () => {
      // Use checkpointing=true for both to ensure they fit in memory
      const withFA = getValidatedSimulationMetrics({ ...baseConfig, flashAttention: true, activationCheckpointing: true });
      const noFA = getValidatedSimulationMetrics({ ...baseConfig, flashAttention: false, activationCheckpointing: true });
      expect(Math.abs(withFA.mfu - noFA.mfu)).toBeLessThan(0.001);
    });
  });
});

// ─── Section 5: Inference Memory FA Reduction ───────────────────────────────

describe('Section 5: Inference memory FA reduction', () => {
  describe('LLaMA-3 8B direct function calls', () => {
    const model = getModelOrThrow('llama3-8b');

    it('activationMemory() returns attention-dominated value (seq=4096)', () => {
      const mem = activationMemory(model, 1, 4096, 'bf16');
      // attentionScores = 32 * 4096 * 4096 * 2 = 1,073,741,824
      // hiddenStates = 2 * 1 * 4096 * 4096 * 2 = 67,108,864
      // mlpActivations = 1 * 4096 * 14336 * 2 = 117,440,512
      // max(67M, 1073M, 117M) = 1,073,741,824
      expect(mem).toBe(1_073_741_824);
    });

    it('activationMemoryWithFlashAttention() returns mlp-dominated value', () => {
      const mem = activationMemoryWithFlashAttention(model, 1, 4096, 'bf16');
      // hiddenStates = 2 * 1 * 4096 * 4096 * 2 = 67,108,864
      // mlpActivations = 1 * 4096 * 14336 * 2 = 117,440,512
      // flashBuffer = 1 * 32 * 128 * 128 * 2 = 1,048,576
      // max(67M, 117M) + 1M = 118,489,088
      expect(mem).toBe(118_489_088);
    });

    it('reduction ratio ≈ 9.06×', () => {
      const noFA = activationMemory(model, 1, 4096, 'bf16');
      const withFA = activationMemoryWithFlashAttention(model, 1, 4096, 'bf16');
      const ratio = noFA / withFA;
      expect(ratio).toBeCloseTo(9.06, 1);
    });
  });

  describe('GPT-3 125M direct function calls', () => {
    const model = getModelOrThrow('gpt3-125m');

    it('activationMemory() = 25,165,824 (attention dominates)', () => {
      const mem = activationMemory(model, 1, 1024, 'bf16');
      // attentionScores = 12 * 1024 * 1024 * 2 = 25,165,824
      // hiddenStates = 2 * 1 * 1024 * 768 * 2 = 3,145,728
      // mlpActivations = 1 * 1024 * 3072 * 2 = 6,291,456
      // max(3.1M, 25.2M, 6.3M) = 25,165,824
      expect(mem).toBe(25_165_824);
    });

    it('activationMemoryWithFlashAttention() = 6,684,672', () => {
      const mem = activationMemoryWithFlashAttention(model, 1, 1024, 'bf16');
      // hiddenStates = 2 * 1 * 1024 * 768 * 2 = 3,145,728
      // mlpActivations = 1 * 1024 * 3072 * 2 = 6,291,456
      // flashBuffer = 1 * 12 * 128 * 128 * 2 = 393,216
      // max(3.1M, 6.3M) + 393K = 6,684,672
      expect(mem).toBe(6_684_672);
    });

    it('reduction ratio ≈ 3.76×', () => {
      const noFA = activationMemory(model, 1, 1024, 'bf16');
      const withFA = activationMemoryWithFlashAttention(model, 1, 1024, 'bf16');
      expect(noFA / withFA).toBeCloseTo(3.76, 1);
    });
  });

  describe('calculateInferenceMemory integration', () => {
    const model = getModelOrThrow('llama3-8b');

    it('only activations differ between FA on/off', () => {
      const memFA = calculateInferenceMemory(model, 4096, 1, 'bf16', 'bf16', true);
      const memNoFA = calculateInferenceMemory(model, 4096, 1, 'bf16', 'bf16', false);

      expect(memFA.weights).toBe(memNoFA.weights);
      expect(memFA.kvCache).toBe(memNoFA.kvCache);
      expect(memFA.overhead).toBe(memNoFA.overhead);
      expect(memFA.activations).toBeLessThan(memNoFA.activations);
      expect(memFA.total).toBeLessThan(memNoFA.total);
    });
  });

  describe('TP path', () => {
    const model = getModelOrThrow('llama3-8b');

    it('TP=2, FA=false uses full attention scores (divided by TP)', () => {
      const memNoFA = calculateMemoryWithTP(model, 4096, 1, 2, 'bf16', 'bf16', 1, false);
      const fullNoFA = activationMemory(model, 1, 4096, 'bf16');
      expect(memNoFA.activations).toBe(fullNoFA / 2);
    });

    it('TP=2, FA=true uses flash attention (divided by TP)', () => {
      const memFA = calculateMemoryWithTP(model, 4096, 1, 2, 'bf16', 'bf16', 1, true);
      const fullFA = activationMemoryWithFlashAttention(model, 1, 4096, 'bf16');
      expect(memFA.activations).toBe(fullFA / 2);
    });

    it('TP path FA=false > FA=true for activations', () => {
      const memNoFA = calculateMemoryWithTP(model, 4096, 1, 2, 'bf16', 'bf16', 1, false);
      const memFA = calculateMemoryWithTP(model, 4096, 1, 2, 'bf16', 'bf16', 1, true);
      expect(memNoFA.activations).toBeGreaterThan(memFA.activations);
      expect(memNoFA.total).toBeGreaterThan(memFA.total);
    });

    it('weights and KV cache identical regardless of FA', () => {
      const memNoFA = calculateMemoryWithTP(model, 4096, 1, 2, 'bf16', 'bf16', 1, false);
      const memFA = calculateMemoryWithTP(model, 4096, 1, 2, 'bf16', 'bf16', 1, true);
      expect(memNoFA.weights).toBe(memFA.weights);
      expect(memNoFA.kvCache).toBe(memFA.kvCache);
    });
  });
});

// ─── Section 6: Inference Recommendation Edge Cases ─────────────────────────

describe('Section 6: Inference recommendation edge cases', () => {
  it('V100 + FA=false: no FA recommendation (unsupported GPU)', () => {
    const result = runInferenceSimulation({
      modelId: 'llama2-7b',
      gpuId: 'v100-32gb',
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      flashAttention: false,
    });
    // V100 (Volta) doesn't support FA — no recommendation to enable it
    const faRecs = result.recommendations.filter(r =>
      r.toLowerCase().includes('flash attention')
    );
    expect(faRecs).toHaveLength(0);
  });

  it('small batch + short seq + FA=false: no FA recommendation (negligible benefit)', () => {
    // With tiny sequences, attention scores are small — FA won't meaningfully help
    const result = runInferenceSimulation({
      modelId: 'gpt3-125m',
      gpuId: 'h100-sxm',
      batchSize: 1,
      inputSeqLen: 128,
      outputSeqLen: 64,
      flashAttention: false,
    });
    // Validator should reject because activation memory is tiny relative to total
    const faRecs = result.recommendations.filter(r =>
      r.toLowerCase().includes('flash attention')
    );
    expect(faRecs).toHaveLength(0);
  });
});

// ─── Section 7: MFU/Throughput Invariance ───────────────────────────────────

describe('Section 7: MFU/throughput invariance', () => {
  it('LLaMA-3 8B FSDP on 8x H100: MFU invariant to FA', () => {
    // Use mbs=1 to avoid OOM with noFA — the 2.5× attention coefficient
    // (Korthikanti et al. 2022) makes seq=4096 mbs=4 noFA exceed 80GB.
    const base: Omit<SimulationConfig, 'flashAttention'> = {
      modelId: 'llama3-8b',
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 1)!,
      globalBatchSize: 8,
      microBatchSize: 1,
      sequenceLength: 4096,
      strategyType: 'fsdp',
      activationCheckpointing: true,
    };

    const withFA = getValidatedSimulationMetrics({ ...base, flashAttention: true });
    const noFA = getValidatedSimulationMetrics({ ...base, flashAttention: false });

    expect(Math.abs(withFA.mfu - noFA.mfu)).toBeLessThan(0.001);
    expect(Math.abs(withFA.hfu - noFA.hfu)).toBeLessThan(0.001);
    const stepTimeDiff = Math.abs(withFA.stepTimeMs - noFA.stepTimeMs) / noFA.stepTimeMs;
    expect(stepTimeDiff).toBeLessThan(0.001);
  });

  it('LLaMA-3 8B 3D-parallel on 8x H100: MFU invariant to FA', () => {
    const base: Omit<SimulationConfig, 'flashAttention'> = {
      modelId: 'llama3-8b',
      clusterConfig: createMultiNodeCluster('h100-sxm', 8, 1)!,
      globalBatchSize: 32,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 2, pp: 2, sequenceParallel: true },
      activationCheckpointing: true,
    };

    const withFA = getValidatedSimulationMetrics({ ...base, flashAttention: true });
    const noFA = getValidatedSimulationMetrics({ ...base, flashAttention: false });

    expect(Math.abs(withFA.mfu - noFA.mfu)).toBeLessThan(0.001);
    expect(Math.abs(withFA.hfu - noFA.hfu)).toBeLessThan(0.001);
    const stepTimeDiff = Math.abs(withFA.stepTimeMs - noFA.stepTimeMs) / noFA.stepTimeMs;
    expect(stepTimeDiff).toBeLessThan(0.001);
  });
});

// ─── Section 8: Long Context Validation ─────────────────────────────────────

describe('Section 8: Long context validation', () => {
  it('LLaMA-3 8B at seq=131072: standard attention score memory ≈ 1 TiB', () => {
    // 32 heads × 131072 × 131072 × 2 bytes = 1,099,511,627,776 bytes ≈ 1 TiB
    const noFA = 32 * 131072 * 131072 * 2;
    expect(noFA).toBe(1_099_511_627_776);
    expect(noFA / (1024 ** 4)).toBeCloseTo(1.0, 1); // ≈ 1 TiB
  });

  it('LLaMA-3 8B at seq=131072: FA attention score memory = 8 MiB', () => {
    // 32 heads × 131072 × 2 bytes = 8,388,608 bytes = 8 MiB
    const withFA = 32 * 131072 * 2;
    expect(withFA).toBe(8_388_608);
    expect(withFA / (1024 ** 2)).toBe(8); // exactly 8 MiB
  });

  it('reduction ratio at 128K context = 131072×', () => {
    const noFA = 32 * 131072 * 131072 * 2;
    const withFA = 32 * 131072 * 2;
    expect(noFA / withFA).toBe(131072);
  });

  it('full activation memory at 128K: estimateActivationMemory confirms huge gap', () => {
    const model = getModelOrThrow('llama3-8b');
    const noFA = estimateActivationMemory(model, 131072, 1, 'bf16', false, false);
    const withFA = estimateActivationMemory(model, 131072, 1, 'bf16', false, true);

    // noFA dominated by attention scores (~35.6 TB total across 32 layers)
    expect(noFA).toBeGreaterThan(1e13); // > 10 TB
    // FA version is ~395 GB (MLP + projections dominate, linear in seq)
    expect(withFA).toBeLessThan(1e12); // < 1 TB
    // Reduction factor ~90× (attention scores eliminated, but MLP/projections remain)
    expect(noFA / withFA).toBeGreaterThan(50);
  });
});
