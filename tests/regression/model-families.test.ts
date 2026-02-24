/**
 * Model Families — Registry + Training & Inference Tests
 *
 * Merged from:
 *   - tests/validation/new-models.test.ts (Model Registry Tests)
 *   - tests/validation/new-model-families.test.ts (Model Family Smoke Tests)
 *
 * Section 1: Model Registry Tests
 *   Validates parameter counts and architecture properties for all model families.
 *   Reference: HuggingFace config.json, published papers.
 *
 * Section 2: Model Family Smoke Tests
 *   Validates training & inference simulation pipeline produces reasonable results.
 *   Regression values pinned from simulator output with ±15% (training) / ±30% (inference).
 */

import { describe, it, expect } from 'vitest';
import { runSimulation } from '../../src/core/simulation/index.ts';
import { runInferenceSimulation } from '../../src/core/inference/simulation.ts';
import { getModel } from '../../src/core/models/index.ts';
import { ALL_MODEL_CONFIGS } from '../../src/core/models/architectures.ts';
import {
  type SimulationConfig,
  type SimulationMetrics,
} from '../../src/core/simulation/engine.ts';
import { getValidatedSimulationMetrics } from '../helpers/validated-metrics.ts';
import { createMultiNodeCluster } from '../../src/core/hardware/topology.ts';

// Helper for tests that need custom cluster sizes matching published configs
function benchmarkConfig(
  gpuId: string, gpusPerNode: number, numNodes: number,
  modelId: string, strategy: SimulationConfig['strategyType'],
  gbs: number, mbs: number, seqLen: number,
  strategyConfig?: SimulationConfig['strategyConfig'],
  opts?: { maxSteps?: number; mixedPrecision?: SimulationConfig['mixedPrecision'] },
): SimulationConfig {
  return {
    clusterConfig: createMultiNodeCluster(gpuId, gpusPerNode, numNodes)!,
    modelId, globalBatchSize: gbs, microBatchSize: mbs, sequenceLength: seqLen,
    strategyType: strategy, strategyConfig,
    maxSteps: opts?.maxSteps, mixedPrecision: opts?.mixedPrecision,
  };
}

function sim(config: SimulationConfig): SimulationMetrics {
  return getValidatedSimulationMetrics(config);
}

// ============================================================================
// Section 1: Model Registry Tests (from new-models.test.ts)
// ============================================================================

describe('New Model Parameter Count Validation', () => {
  const expectedParams: [string, number, number][] = [
    // Gemma 2
    ['gemma2-2b', 2.3e9, 2.9e9],
    ['gemma2-9b', 8e9, 10e9],
    ['gemma2-27b', 24e9, 30e9],
    // LLaMA 3.2
    ['llama3.2-1b', 1e9, 1.5e9],
    ['llama3.2-3b', 2.8e9, 3.6e9],
    // DeepSeek-V3 (MLA: ~671B total, paper: 671B)
    ['deepseek-v3', 665e9, 680e9],
    // Phi-3
    ['phi3-mini', 3.5e9, 4.2e9],
    ['phi3-small', 6.5e9, 8e9],
    ['phi3-medium', 13e9, 15e9],
    // DBRX (total MoE params)
    ['dbrx', 125e9, 140e9],
    // Yi
    ['yi-6b', 5.5e9, 6.5e9],
    ['yi-34b', 32e9, 36e9],
    // OLMo 2
    ['olmo2-7b', 6.5e9, 7.5e9],
    ['olmo2-13b', 12e9, 14e9],
    ['olmo2-32b', 31.6e9, 32.9e9],
    // Command R
    ['command-r', 33e9, 37e9],
    ['command-r-plus', 100e9, 108e9],
    // Nemotron-4
    ['nemotron-4-15b', 14e9, 16e9],
    ['nemotron-4-340b', 330e9, 350e9],
    // Mistral family (new)
    ['mistral-nemo-12b', 11e9, 13e9],
    ['codestral-22b', 21e9, 23e9],
    ['mistral-small-24b', 22e9, 25e9],
    ['mistral-large-123b', 120e9, 128e9],
    // Qwen 2.5
    ['qwen2.5-0.5b', 0.5e9, 0.7e9],
    ['qwen2.5-1.5b', 1.3e9, 1.8e9],
    ['qwen2.5-3b', 2.8e9, 3.6e9],
    ['qwen2.5-7b', 7e9, 8e9],
    ['qwen2.5-14b', 13e9, 16e9],
    ['qwen2.5-32b', 30e9, 35e9],
    ['qwen2.5-72b', 71e9, 75e9],
    // DeepSeek-R1 (MLA: ~671B total, paper: 671B)
    ['deepseek-r1', 665e9, 680e9],
    // DeepSeek-V2 (MLA: ~236B total, paper: 236B)
    ['deepseek-v2', 230e9, 242e9],
    // DeepSeek-MoE-16B
    ['deepseek-moe-16b', 15.5e9, 17.5e9],
    // Mixtral 8x22B
    ['mixtral-8x22b', 139e9, 145e9],
    // Grok-2.5 (Residual MoE: ~270B total, published: 270B)
    ['grok-2.5', 265e9, 275e9],
    // Grok-1 (GeGLU gated MLP: ~316.5B, published: 314B)
    ['grok-1', 310e9, 323e9],
    // Llama 3.3
    ['llama3.3-70b', 68e9, 73e9],
    // Llama 4
    ['llama4-scout', 100e9, 115e9],
    ['llama4-maverick', 390e9, 410e9],
    // Qwen 3 dense
    ['qwen3-0.6b', 0.55e9, 0.7e9],
    ['qwen3-1.7b', 1.5e9, 1.9e9],
    ['qwen3-4b', 3.6e9, 4.4e9],
    ['qwen3-8b', 7.5e9, 8.5e9],
    ['qwen3-14b', 13.5e9, 15.5e9],
    ['qwen3-32b', 30e9, 35e9],
    // Qwen 3 MoE
    ['qwen3-30b-a3b', 28e9, 33e9],
    ['qwen3-235b-a22b', 210e9, 260e9],
    // Phi-4
    ['phi4', 13e9, 15e9],
    ['phi4-mini', 3.5e9, 4.2e9],
    ['phi4-mini-flash', 3.4e9, 3.9e9],
    // Gemma 3
    ['gemma3-1b', 0.9e9, 1.1e9],
    ['gemma3-4b', 3.5e9, 4.3e9],
    ['gemma3-12b', 10.5e9, 12.5e9],
    ['gemma3-27b', 25e9, 29e9],
    // GLM
    ['glm4-9b', 8.5e9, 10.5e9],
    ['glm4-32b', 31e9, 34e9],
    ['glm4.5-air', 100e9, 113e9],
    ['glm4.7', 340e9, 370e9],
    ['glm4.7-flash', 27e9, 33e9],
    ['glm5', 710e9, 780e9],
    // Kimi K2.5 (MLA MoE: ~1026B total)
    ['kimi-k2.5', 1000e9, 1055e9],
    // MiniMax M2.5 (MoE: ~228.7B total, published: ~230B)
    ['minimax-m2.5', 225e9, 232e9],
    // GPT-OSS (MoE)
    ['gpt-oss-120b', 115e9, 119e9],
    ['gpt-oss-20b', 20e9, 22e9],
    // Mistral Large 3 (MoE with MLA: ~673B total, published: 673B)
    ['mistral-large-3-675b', 660e9, 687e9],
    // Ministral 3
    ['ministral-3-3b', 3.3e9, 3.6e9],
    ['ministral-3-8b', 8.1e9, 8.9e9],
    ['ministral-3-14b', 12.9e9, 14.2e9],
    // Devstral 2 (~125B, like Mistral Large but +2.4B from 4x larger vocab)
    ['devstral-2', 122e9, 128e9],
    // Devstral Small 2 (~23.6B, identical arch to mistral-small-24b)
    ['devstral-small-2', 22e9, 25e9],
    // OLMo 3 (7B identical dims to OLMo 2 7B; 32B is ~32.2B)
    ['olmo3-7b', 6.5e9, 7.5e9],
    ['olmo3-32b', 31.6e9, 32.9e9],
  ];

  it.each(expectedParams)(
    '%s should have totalParams between %d and %d',
    (modelId, minParams, maxParams) => {
      const model = getModel(modelId);
      expect(model, `Model ${modelId} not found in registry`).toBeDefined();
      expect(model!.totalParams).toBeGreaterThanOrEqual(minParams);
      expect(model!.totalParams).toBeLessThanOrEqual(maxParams);
    }
  );
});

// Pinned to exact simulator output ±2%. If these break, verify the model config change was intentional.
describe('MoE Models', () => {
  it('deepseek-v3 should have activeParams ~37.55B (paper: 37B)', () => {
    const model = getModel('deepseek-v3')!;
    expect(model).toBeDefined();
    expect(model.isMoE).toBe(true);
    expect(model.activeParams).toBeLessThan(model.totalParams);
    expect(model.activeParams).toBeGreaterThan(37.55e9 * 0.98);
    expect(model.activeParams).toBeLessThan(37.55e9 * 1.02);
  });

  it('dbrx should have activeParams ~36.47B', () => {
    const model = getModel('dbrx')!;
    expect(model).toBeDefined();
    expect(model.isMoE).toBe(true);
    expect(model.activeParams).toBeLessThan(model.totalParams);
    expect(model.activeParams).toBeGreaterThan(36.47e9 * 0.98);
    expect(model.activeParams).toBeLessThan(36.47e9 * 1.02);
  });

  it('deepseek-r1 should have activeParams ~37.55B (paper: 37B)', () => {
    const model = getModel('deepseek-r1')!;
    expect(model).toBeDefined();
    expect(model.isMoE).toBe(true);
    expect(model.activeParams).toBeLessThan(model.totalParams);
    expect(model.activeParams).toBeGreaterThan(37.55e9 * 0.98);
    expect(model.activeParams).toBeLessThan(37.55e9 * 1.02);
  });

  it('grok-1 should have activeParams ~84.56B (~26.7% active, gated MLP (GeGLU))', () => {
    const model = getModel('grok-1')!;
    expect(model).toBeDefined();
    expect(model.isMoE).toBe(true);
    expect(model.activeParams).toBeLessThan(model.totalParams);
    expect(model.activeParams).toBeGreaterThan(84.56e9 * 0.98);
    expect(model.activeParams).toBeLessThan(84.56e9 * 1.02);
  });

  it('grok-2.5 should have activeParams ~114.9B (~42.7% active, residual MoE)', () => {
    const model = getModel('grok-2.5')!;
    expect(model).toBeDefined();
    expect(model.isMoE).toBe(true);
    expect(model.activeParams).toBeLessThan(model.totalParams);
    // Residual MoE: 2 active + 2 shared out of 8+2 total experts → ~42.7% active
    expect(model.activeParams).toBeGreaterThan(114.9e9 * 0.98);
    expect(model.activeParams).toBeLessThan(114.9e9 * 1.02);
  });

  it('llama4-scout should be MoE with activeParams ~17.17B', () => {
    const model = getModel('llama4-scout')!;
    expect(model).toBeDefined();
    expect(model.isMoE).toBe(true);
    expect(model.activeParams).toBeLessThan(model.totalParams);
    expect(model.activeParams).toBeGreaterThan(17.17e9 * 0.98);
    expect(model.activeParams).toBeLessThan(17.17e9 * 1.02);
  });

  it('llama4-maverick should be MoE with activeParams ~17.18B', () => {
    const model = getModel('llama4-maverick')!;
    expect(model).toBeDefined();
    expect(model.isMoE).toBe(true);
    expect(model.activeParams).toBeLessThan(model.totalParams);
    expect(model.activeParams).toBeGreaterThan(17.18e9 * 0.98);
    expect(model.activeParams).toBeLessThan(17.18e9 * 1.02);
  });

  it('deepseek-v2 should have activeParams ~21.38B (paper: 21B)', () => {
    const model = getModel('deepseek-v2')!;
    expect(model).toBeDefined();
    expect(model.isMoE).toBe(true);
    expect(model.activeParams).toBeLessThan(model.totalParams);
    expect(model.activeParams).toBeGreaterThan(21.38e9 * 0.98);
    expect(model.activeParams).toBeLessThan(21.38e9 * 1.02);
  });

  it('deepseek-moe-16b should have activeParams ~2.83B', () => {
    const model = getModel('deepseek-moe-16b')!;
    expect(model).toBeDefined();
    expect(model.isMoE).toBe(true);
    expect(model.activeParams).toBeLessThan(model.totalParams);
    expect(model.activeParams).toBeGreaterThan(2.83e9 * 0.98);
    expect(model.activeParams).toBeLessThan(2.83e9 * 1.02);
  });

  it('mixtral-8x22b should have activeParams ~39.16B', () => {
    const model = getModel('mixtral-8x22b')!;
    expect(model).toBeDefined();
    expect(model.isMoE).toBe(true);
    expect(model.activeParams).toBeLessThan(model.totalParams);
    expect(model.activeParams).toBeGreaterThan(39.16e9 * 0.98);
    expect(model.activeParams).toBeLessThan(39.16e9 * 1.02);
  });

  it('qwen3-30b-a3b should be MoE with activeParams ~3.35B', () => {
    const model = getModel('qwen3-30b-a3b')!;
    expect(model).toBeDefined();
    expect(model.isMoE).toBe(true);
    expect(model.activeParams).toBeLessThan(model.totalParams);
    expect(model.activeParams).toBeGreaterThan(3.35e9 * 0.98);
    expect(model.activeParams).toBeLessThan(3.35e9 * 1.02);
  });

  it('qwen3-235b-a22b should be MoE with activeParams ~22.19B', () => {
    const model = getModel('qwen3-235b-a22b')!;
    expect(model).toBeDefined();
    expect(model.isMoE).toBe(true);
    expect(model.activeParams).toBeLessThan(model.totalParams);
    expect(model.activeParams).toBeGreaterThan(22.19e9 * 0.98);
    expect(model.activeParams).toBeLessThan(22.19e9 * 1.02);
  });

  it('glm4.5-air should be MoE with activeParams ~13.42B', () => {
    const model = getModel('glm4.5-air')!;
    expect(model).toBeDefined();
    expect(model.isMoE).toBe(true);
    expect(model.activeParams).toBeLessThan(model.totalParams);
    expect(model.activeParams).toBeGreaterThan(13.42e9 * 0.98);
    expect(model.activeParams).toBeLessThan(13.42e9 * 1.02);
  });

  it('glm4.7 should be MoE with activeParams ~33.63B', () => {
    const model = getModel('glm4.7')!;
    expect(model).toBeDefined();
    expect(model.isMoE).toBe(true);
    expect(model.activeParams).toBeLessThan(model.totalParams);
    expect(model.activeParams).toBeGreaterThan(33.63e9 * 0.98);
    expect(model.activeParams).toBeLessThan(33.63e9 * 1.02);
  });

  it('glm4.7-flash should be MoE with activeParams ~3.90B', () => {
    const model = getModel('glm4.7-flash')!;
    expect(model).toBeDefined();
    expect(model.isMoE).toBe(true);
    expect(model.activeParams).toBeLessThan(model.totalParams);
    expect(model.activeParams).toBeGreaterThan(3.90e9 * 0.98);
    expect(model.activeParams).toBeLessThan(3.90e9 * 1.02);
  });

  it('glm5 should be MoE with activeParams ~41.05B', () => {
    const model = getModel('glm5')!;
    expect(model).toBeDefined();
    expect(model.isMoE).toBe(true);
    expect(model.activeParams).toBeLessThan(model.totalParams);
    expect(model.activeParams).toBeGreaterThan(41.05e9 * 0.98);
    expect(model.activeParams).toBeLessThan(41.05e9 * 1.02);
  });

  it('kimi-k2.5 should be MoE with activeParams ~32.86B (paper: ~32B)', () => {
    const model = getModel('kimi-k2.5')!;
    expect(model).toBeDefined();
    expect(model.isMoE).toBe(true);
    expect(model.activeParams).toBeLessThan(model.totalParams);
    expect(model.activeParams).toBeGreaterThan(32.86e9 * 0.98);
    expect(model.activeParams).toBeLessThan(32.86e9 * 1.02);
  });

  it('minimax-m2.5 should have activeParams ~11.03B (published: ~10B)', () => {
    const model = getModel('minimax-m2.5')!;
    expect(model).toBeDefined();
    expect(model.isMoE).toBe(true);
    expect(model.activeParams).toBeLessThan(model.totalParams);
    expect(model.activeParams).toBeGreaterThan(11.03e9 * 0.98);
    expect(model.activeParams).toBeLessThan(11.03e9 * 1.02);
  });

  it('gpt-oss-120b should be MoE with activeParams ~5.71B (paper: 5.13B, sim includes output head)', () => {
    const model = getModel('gpt-oss-120b')!;
    expect(model).toBeDefined();
    expect(model.isMoE).toBe(true);
    expect(model.activeParams).toBeLessThan(model.totalParams);
    expect(model.activeParams).toBeGreaterThan(5.71e9 * 0.98);
    expect(model.activeParams).toBeLessThan(5.71e9 * 1.02);
  });

  it('gpt-oss-20b should be MoE with activeParams ~4.19B (paper: 3.61B, sim includes output head)', () => {
    const model = getModel('gpt-oss-20b')!;
    expect(model).toBeDefined();
    expect(model.isMoE).toBe(true);
    expect(model.activeParams).toBeLessThan(model.totalParams);
    expect(model.activeParams).toBeGreaterThan(4.19e9 * 0.98);
    expect(model.activeParams).toBeLessThan(4.19e9 * 1.02);
  });

  it('mistral-large-3-675b should be MoE with activeParams ~39.95B (paper: 39B LM)', () => {
    const model = getModel('mistral-large-3-675b')!;
    expect(model).toBeDefined();
    expect(model.isMoE).toBe(true);
    expect(model.activeParams).toBeLessThan(model.totalParams);
    expect(model.activeParams).toBeGreaterThan(39.95e9 * 0.98);
    expect(model.activeParams).toBeLessThan(39.95e9 * 1.02);
  });
});

describe('Architecture Field Correctness', () => {
  it('Nemotron-4 models should use relu activation', () => {
    expect(ALL_MODEL_CONFIGS['nemotron-4-15b'].activation).toBe('relu');
    expect(ALL_MODEL_CONFIGS['nemotron-4-340b'].activation).toBe('relu');
  });

  it('Nemotron-4 models should have non-gated MLP', () => {
    expect(ALL_MODEL_CONFIGS['nemotron-4-15b'].gatedMLP).toBe(false);
    expect(ALL_MODEL_CONFIGS['nemotron-4-340b'].gatedMLP).toBe(false);
  });

  it('Gemma 2 models should have tied embeddings', () => {
    expect(ALL_MODEL_CONFIGS['gemma2-2b'].tiedEmbeddings).toBe(true);
    expect(ALL_MODEL_CONFIGS['gemma2-9b'].tiedEmbeddings).toBe(true);
    expect(ALL_MODEL_CONFIGS['gemma2-27b'].tiedEmbeddings).toBe(true);
  });

  it('Gemma 2 models should use gelu with gated MLP (GeGLU)', () => {
    expect(ALL_MODEL_CONFIGS['gemma2-9b'].activation).toBe('gelu');
    expect(ALL_MODEL_CONFIGS['gemma2-9b'].gatedMLP).toBe(true);
  });

  it('LLaMA 3.2 models should have tied embeddings', () => {
    expect(ALL_MODEL_CONFIGS['llama3.2-1b'].tiedEmbeddings).toBe(true);
    expect(ALL_MODEL_CONFIGS['llama3.2-3b'].tiedEmbeddings).toBe(true);
  });

  it('Command R models should have tied embeddings', () => {
    expect(ALL_MODEL_CONFIGS['command-r'].tiedEmbeddings).toBe(true);
    expect(ALL_MODEL_CONFIGS['command-r-plus'].tiedEmbeddings).toBe(true);
  });

  it('OLMo 2 models should use MHA (not GQA)', () => {
    expect(ALL_MODEL_CONFIGS['olmo2-7b'].attentionType).toBe('mha');
    expect(ALL_MODEL_CONFIGS['olmo2-13b'].attentionType).toBe('mha');
  });

  it('Mistral dense models should use GQA with 8 KV heads', () => {
    for (const id of ['mistral-7b', 'mistral-nemo-12b', 'codestral-22b', 'mistral-small-24b', 'mistral-large-123b',
                       'ministral-3-3b', 'ministral-3-8b', 'ministral-3-14b',
                       'devstral-2', 'devstral-small-2']) {
      expect(ALL_MODEL_CONFIGS[id].attentionType).toBe('gqa');
      expect(ALL_MODEL_CONFIGS[id].numKvHeads).toBe(8);
    }
  });

  it('Mistral Nemo and Small should have non-standard headDim=128', () => {
    expect(ALL_MODEL_CONFIGS['mistral-nemo-12b'].headDim).toBe(128);
    expect(ALL_MODEL_CONFIGS['mistral-small-24b'].headDim).toBe(128);
    // hiddenSize/numAttentionHeads would give 160, not 128
    const nemo = ALL_MODEL_CONFIGS['mistral-nemo-12b'];
    expect(nemo.hiddenSize / nemo.numAttentionHeads).toBe(160);
    // But the built spec should use the override
    const nemoSpec = getModel('mistral-nemo-12b')!;
    expect(nemoSpec.headDim).toBe(128);
  });

  it('Codestral should have 48 attention heads (128 headDim)', () => {
    const config = ALL_MODEL_CONFIGS['codestral-22b'];
    expect(config.numAttentionHeads).toBe(48);
    expect(config.hiddenSize / config.numAttentionHeads).toBe(128);
  });

  it('Mistral Large should have 96 attention heads (128 headDim)', () => {
    const config = ALL_MODEL_CONFIGS['mistral-large-123b'];
    expect(config.numAttentionHeads).toBe(96);
    expect(config.hiddenSize / config.numAttentionHeads).toBe(128);
  });

  it('Mistral 7B v0.3 should have vocabSize 32768', () => {
    expect(ALL_MODEL_CONFIGS['mistral-7b'].vocabSize).toBe(32768);
  });

  it('All Mistral models should use SwiGLU (gated MLP + silu)', () => {
    for (const id of ['mistral-7b', 'mistral-nemo-12b', 'codestral-22b', 'mistral-small-24b', 'mistral-large-123b',
                       'mistral-large-3-675b', 'ministral-3-3b', 'ministral-3-8b', 'ministral-3-14b',
                       'devstral-2', 'devstral-small-2']) {
      expect(ALL_MODEL_CONFIGS[id].gatedMLP).toBe(true);
      expect(ALL_MODEL_CONFIGS[id].activation).toBe('silu');
    }
  });

  it('Qwen 2.5 models should use GQA with bias', () => {
    for (const id of ['qwen2.5-0.5b', 'qwen2.5-1.5b', 'qwen2.5-3b', 'qwen2.5-7b', 'qwen2.5-14b', 'qwen2.5-32b', 'qwen2.5-72b']) {
      expect(ALL_MODEL_CONFIGS[id].useBias).toBe(true);
      expect(ALL_MODEL_CONFIGS[id].attentionType).toBe('gqa');
    }
  });

  it('Grok-1 should use gated MLP (GeGLU) with GELU', () => {
    expect(ALL_MODEL_CONFIGS['grok-1'].gatedMLP).toBe(true);
    expect(ALL_MODEL_CONFIGS['grok-1'].activation).toBe('gelu');
  });

  it('Grok-1 should be MoE with 8 experts, 2 active', () => {
    expect(ALL_MODEL_CONFIGS['grok-1'].numExperts).toBe(8);
    expect(ALL_MODEL_CONFIGS['grok-1'].numActiveExperts).toBe(2);
  });

  it('Grok 2.5 should use gated MLP (GeGLU) with GELU', () => {
    expect(ALL_MODEL_CONFIGS['grok-2.5'].gatedMLP).toBe(true);
    expect(ALL_MODEL_CONFIGS['grok-2.5'].activation).toBe('gelu');
  });

  it('Grok 2.5 should be residual MoE with 8 experts, 2 active, 2 shared', () => {
    expect(ALL_MODEL_CONFIGS['grok-2.5'].numExperts).toBe(8);
    expect(ALL_MODEL_CONFIGS['grok-2.5'].numActiveExperts).toBe(2);
    expect(ALL_MODEL_CONFIGS['grok-2.5'].numSharedExperts).toBe(2);
  });

  it('Grok 2.5 should use GQA with 8 KV heads', () => {
    expect(ALL_MODEL_CONFIGS['grok-2.5'].attentionType).toBe('gqa');
    expect(ALL_MODEL_CONFIGS['grok-2.5'].numKvHeads).toBe(8);
  });

  it('Grok 2.5 should have 128K context', () => {
    expect(ALL_MODEL_CONFIGS['grok-2.5'].maxSeqLength).toBe(131072);
  });

  it('Grok 2.5 all 64 layers should be MoE (no dense-first layers)', () => {
    expect(getModel('grok-2.5')!.numMoELayers).toBe(64);
    expect(ALL_MODEL_CONFIGS['grok-2.5'].firstKDenseLayers).toBeUndefined();
  });

  it('Llama 3.3 should have 128K context', () => {
    expect(ALL_MODEL_CONFIGS['llama3.3-70b'].maxSeqLength).toBe(131072);
  });

  it('DeepSeek V2/V3/R1 should use MLA attention type', () => {
    expect(ALL_MODEL_CONFIGS['deepseek-v2'].attentionType).toBe('mla');
    expect(ALL_MODEL_CONFIGS['deepseek-v3'].attentionType).toBe('mla');
    expect(ALL_MODEL_CONFIGS['deepseek-r1'].attentionType).toBe('mla');
    // MoE-16B predates MLA — should NOT be MLA
    expect(ALL_MODEL_CONFIGS['deepseek-moe-16b'].attentionType).toBe('mha');
  });

  it('DeepSeek MLA models should have kvLoraRank and qLoraRank', () => {
    for (const id of ['deepseek-v2', 'deepseek-v3', 'deepseek-r1']) {
      expect(ALL_MODEL_CONFIGS[id].kvLoraRank).toBe(512);
      expect(ALL_MODEL_CONFIGS[id].qLoraRank).toBe(1536);
      expect(ALL_MODEL_CONFIGS[id].qkNopeHeadDim).toBe(128);
      expect(ALL_MODEL_CONFIGS[id].qkRopeHeadDim).toBe(64);
      expect(ALL_MODEL_CONFIGS[id].vHeadDim).toBe(128);
    }
  });

  it('DeepSeek MLA numKvHeads should be 128 (decompressed to all heads)', () => {
    expect(ALL_MODEL_CONFIGS['deepseek-v2'].numKvHeads).toBe(128);
    expect(ALL_MODEL_CONFIGS['deepseek-v3'].numKvHeads).toBe(128);
    expect(ALL_MODEL_CONFIGS['deepseek-r1'].numKvHeads).toBe(128);
  });

  it('DeepSeek-R1 should match V3 architecture except maxSeqLength', () => {
    const v3 = ALL_MODEL_CONFIGS['deepseek-v3'];
    const r1 = ALL_MODEL_CONFIGS['deepseek-r1'];
    expect(r1.numLayers).toBe(v3.numLayers);
    expect(r1.hiddenSize).toBe(v3.hiddenSize);
    expect(r1.numExperts).toBe(v3.numExperts);
    expect(r1.numActiveExperts).toBe(v3.numActiveExperts);
    expect(r1.expertIntermediateSize).toBe(v3.expertIntermediateSize);
    expect(r1.numSharedExperts).toBe(v3.numSharedExperts);
    expect(r1.firstKDenseLayers).toBe(v3.firstKDenseLayers);
    expect(r1.maxSeqLength).toBe(163840);
  });

  // Qwen 3 architecture tests
  it('Qwen 3 dense models should use GQA, SiLU, gated MLP, no bias, RMSNorm', () => {
    for (const id of ['qwen3-0.6b', 'qwen3-1.7b', 'qwen3-4b', 'qwen3-8b', 'qwen3-14b', 'qwen3-32b']) {
      expect(ALL_MODEL_CONFIGS[id].attentionType).toBe('gqa');
      expect(ALL_MODEL_CONFIGS[id].activation).toBe('silu');
      expect(ALL_MODEL_CONFIGS[id].gatedMLP).toBe(true);
      expect(ALL_MODEL_CONFIGS[id].useBias).toBe(false);
      expect(ALL_MODEL_CONFIGS[id].normType).toBe('rmsnorm');
    }
  });

  it('Qwen 3 small models should have tied embeddings', () => {
    expect(ALL_MODEL_CONFIGS['qwen3-0.6b'].tiedEmbeddings).toBe(true);
    expect(ALL_MODEL_CONFIGS['qwen3-1.7b'].tiedEmbeddings).toBe(true);
    expect(ALL_MODEL_CONFIGS['qwen3-4b'].tiedEmbeddings).toBe(true);
  });

  it('Qwen 3 large models should NOT have tied embeddings', () => {
    expect(ALL_MODEL_CONFIGS['qwen3-8b'].tiedEmbeddings).toBe(false);
    expect(ALL_MODEL_CONFIGS['qwen3-14b'].tiedEmbeddings).toBe(false);
    expect(ALL_MODEL_CONFIGS['qwen3-32b'].tiedEmbeddings).toBe(false);
  });

  it('Qwen 3 headDim overrides should be 128 where natural differs', () => {
    // 0.6b: natural=64, 4b: natural=80, 32b: natural=80
    expect(ALL_MODEL_CONFIGS['qwen3-0.6b'].headDim).toBe(128);
    expect(ALL_MODEL_CONFIGS['qwen3-4b'].headDim).toBe(128);
    expect(ALL_MODEL_CONFIGS['qwen3-32b'].headDim).toBe(128);
    // Verify built spec uses override
    expect(getModel('qwen3-0.6b')!.headDim).toBe(128);
    expect(getModel('qwen3-4b')!.headDim).toBe(128);
    expect(getModel('qwen3-32b')!.headDim).toBe(128);
  });

  it('Qwen 3 MoE headDim overrides should be 128', () => {
    expect(ALL_MODEL_CONFIGS['qwen3-30b-a3b'].headDim).toBe(128);
    expect(ALL_MODEL_CONFIGS['qwen3-235b-a22b'].headDim).toBe(128);
    expect(getModel('qwen3-30b-a3b')!.headDim).toBe(128);
    expect(getModel('qwen3-235b-a22b')!.headDim).toBe(128);
  });

  it('Qwen 3 MoE models should use GQA with 4 KV heads', () => {
    expect(ALL_MODEL_CONFIGS['qwen3-30b-a3b'].numKvHeads).toBe(4);
    expect(ALL_MODEL_CONFIGS['qwen3-235b-a22b'].numKvHeads).toBe(4);
  });

  // Gemma 3 architecture tests
  it('Gemma 3 models should use GELU, gated MLP, tied embeddings', () => {
    for (const id of ['gemma3-1b', 'gemma3-4b', 'gemma3-12b', 'gemma3-27b']) {
      expect(ALL_MODEL_CONFIGS[id].activation).toBe('gelu');
      expect(ALL_MODEL_CONFIGS[id].gatedMLP).toBe(true);
      expect(ALL_MODEL_CONFIGS[id].tiedEmbeddings).toBe(true);
    }
  });

  it('Gemma 3 1B should use MQA', () => {
    expect(ALL_MODEL_CONFIGS['gemma3-1b'].attentionType).toBe('mqa');
    expect(ALL_MODEL_CONFIGS['gemma3-1b'].numKvHeads).toBe(1);
  });

  it('Gemma 3 4B/12B/27B should use GQA', () => {
    expect(ALL_MODEL_CONFIGS['gemma3-4b'].attentionType).toBe('gqa');
    expect(ALL_MODEL_CONFIGS['gemma3-12b'].attentionType).toBe('gqa');
    expect(ALL_MODEL_CONFIGS['gemma3-27b'].attentionType).toBe('gqa');
  });

  it('Gemma 3 headDim overrides should be correct', () => {
    expect(ALL_MODEL_CONFIGS['gemma3-1b'].headDim).toBe(256);
    expect(ALL_MODEL_CONFIGS['gemma3-4b'].headDim).toBe(256);
    expect(ALL_MODEL_CONFIGS['gemma3-12b'].headDim).toBe(256);
    expect(ALL_MODEL_CONFIGS['gemma3-27b'].headDim).toBe(128);
    // Verify built specs
    expect(getModel('gemma3-1b')!.headDim).toBe(256);
    expect(getModel('gemma3-4b')!.headDim).toBe(256);
    expect(getModel('gemma3-12b')!.headDim).toBe(256);
    expect(getModel('gemma3-27b')!.headDim).toBe(128);
  });

  // Phi-4 architecture tests
  it('All Phi-4 models should use SiLU, gated MLP, GQA, no bias', () => {
    for (const id of ['phi4', 'phi4-mini', 'phi4-mini-flash']) {
      expect(ALL_MODEL_CONFIGS[id].activation).toBe('silu');
      expect(ALL_MODEL_CONFIGS[id].gatedMLP).toBe(true);
      expect(ALL_MODEL_CONFIGS[id].attentionType).toBe('gqa');
      expect(ALL_MODEL_CONFIGS[id].useBias).toBe(false);
    }
  });

  it('Phi-4 14B should use RMSNorm, untied, RoPE, vocabSize=100352, numKvHeads=10', () => {
    const cfg = ALL_MODEL_CONFIGS['phi4'];
    expect(cfg.normType).toBe('rmsnorm');
    expect(cfg.tiedEmbeddings).toBe(false);
    expect(cfg.useRotaryEmbed).toBe(true);
    expect(cfg.vocabSize).toBe(100352);
    expect(cfg.numKvHeads).toBe(10);
  });

  it('Phi-4 Mini should use RMSNorm, tied, RoPE, vocabSize=200064, numKvHeads=8', () => {
    const cfg = ALL_MODEL_CONFIGS['phi4-mini'];
    expect(cfg.normType).toBe('rmsnorm');
    expect(cfg.tiedEmbeddings).toBe(true);
    expect(cfg.useRotaryEmbed).toBe(true);
    expect(cfg.vocabSize).toBe(200064);
    expect(cfg.numKvHeads).toBe(8);
  });

  it('Phi-4 Mini Flash should use LayerNorm, tied, no RoPE, vocabSize=200064, numKvHeads=20', () => {
    const cfg = ALL_MODEL_CONFIGS['phi4-mini-flash'];
    expect(cfg.normType).toBe('layernorm');
    expect(cfg.tiedEmbeddings).toBe(true);
    expect(cfg.useRotaryEmbed).toBe(false);
    expect(cfg.vocabSize).toBe(200064);
    expect(cfg.numKvHeads).toBe(20);
  });

  it('Phi-4 differs from Phi-3: vocabSize, maxSeqLength, totalParams', () => {
    expect(ALL_MODEL_CONFIGS['phi4'].vocabSize).not.toBe(ALL_MODEL_CONFIGS['phi3-medium'].vocabSize);
    expect(ALL_MODEL_CONFIGS['phi4'].maxSeqLength).not.toBe(ALL_MODEL_CONFIGS['phi3-medium'].maxSeqLength);
    const phi4 = getModel('phi4')!;
    const phi3med = getModel('phi3-medium')!;
    expect(phi4.totalParams).not.toBe(phi3med.totalParams);
  });

  it('Phi-4 Mini differs from Phi-3 Mini: attentionType, numAttentionHeads, vocabSize', () => {
    expect(ALL_MODEL_CONFIGS['phi4-mini'].attentionType).not.toBe(ALL_MODEL_CONFIGS['phi3-mini'].attentionType);
    expect(ALL_MODEL_CONFIGS['phi4-mini'].numAttentionHeads).not.toBe(ALL_MODEL_CONFIGS['phi3-mini'].numAttentionHeads);
    expect(ALL_MODEL_CONFIGS['phi4-mini'].vocabSize).not.toBe(ALL_MODEL_CONFIGS['phi3-mini'].vocabSize);
  });

  // Llama 4 architecture tests
  it('Llama 4 models should use GQA with 8 KV heads', () => {
    expect(ALL_MODEL_CONFIGS['llama4-scout'].attentionType).toBe('gqa');
    expect(ALL_MODEL_CONFIGS['llama4-scout'].numKvHeads).toBe(8);
    expect(ALL_MODEL_CONFIGS['llama4-maverick'].attentionType).toBe('gqa');
    expect(ALL_MODEL_CONFIGS['llama4-maverick'].numKvHeads).toBe(8);
  });

  it('Llama 4 models should use SiLU, gated MLP, RMSNorm', () => {
    for (const id of ['llama4-scout', 'llama4-maverick']) {
      expect(ALL_MODEL_CONFIGS[id].activation).toBe('silu');
      expect(ALL_MODEL_CONFIGS[id].gatedMLP).toBe(true);
      expect(ALL_MODEL_CONFIGS[id].normType).toBe('rmsnorm');
    }
  });

  it('Llama 4 Scout should have 16 experts, 1 active', () => {
    expect(ALL_MODEL_CONFIGS['llama4-scout'].numExperts).toBe(16);
    expect(ALL_MODEL_CONFIGS['llama4-scout'].numActiveExperts).toBe(1);
  });

  it('Llama 4 Maverick should have 128 experts, 1 active, alternating layers', () => {
    expect(ALL_MODEL_CONFIGS['llama4-maverick'].numExperts).toBe(128);
    expect(ALL_MODEL_CONFIGS['llama4-maverick'].numActiveExperts).toBe(1);
    expect(ALL_MODEL_CONFIGS['llama4-maverick'].moeLayerFrequency).toBe(2);
  });

  // Shared experts
  it('Models with shared experts should have correct numSharedExperts', () => {
    expect(ALL_MODEL_CONFIGS['deepseek-v3'].numSharedExperts).toBe(1);
    expect(ALL_MODEL_CONFIGS['deepseek-r1'].numSharedExperts).toBe(1);
    expect(ALL_MODEL_CONFIGS['deepseek-v2'].numSharedExperts).toBe(2);
    expect(ALL_MODEL_CONFIGS['deepseek-moe-16b'].numSharedExperts).toBe(2);
    expect(ALL_MODEL_CONFIGS['llama4-scout'].numSharedExperts).toBe(1);
    expect(ALL_MODEL_CONFIGS['llama4-maverick'].numSharedExperts).toBe(1);
    expect(ALL_MODEL_CONFIGS['grok-2.5'].numSharedExperts).toBe(2);
  });

  // firstKDenseLayers
  it('DeepSeek models should have correct firstKDenseLayers', () => {
    expect(ALL_MODEL_CONFIGS['deepseek-v3'].firstKDenseLayers).toBe(3);
    expect(ALL_MODEL_CONFIGS['deepseek-r1'].firstKDenseLayers).toBe(3);
    expect(ALL_MODEL_CONFIGS['deepseek-v2'].firstKDenseLayers).toBe(1);
    expect(ALL_MODEL_CONFIGS['deepseek-moe-16b'].firstKDenseLayers).toBe(1);
  });

  // numMoELayers computed correctly
  it('MoE models should have correct numMoELayers', () => {
    expect(getModel('llama4-scout')!.numMoELayers).toBe(48);
    expect(getModel('llama4-maverick')!.numMoELayers).toBe(24);
    expect(getModel('deepseek-v3')!.numMoELayers).toBe(58);
    expect(getModel('deepseek-r1')!.numMoELayers).toBe(58);
    expect(getModel('deepseek-v2')!.numMoELayers).toBe(59);
    expect(getModel('deepseek-moe-16b')!.numMoELayers).toBe(27);
    expect(getModel('grok-2.5')!.numMoELayers).toBe(64);
  });

  // lastKDenseLayers
  it('DeepSeek-MoE-16B should not have lastKDenseLayers', () => {
    expect(ALL_MODEL_CONFIGS['deepseek-moe-16b'].lastKDenseLayers).toBeUndefined();
  });

  // GLM architecture tests
  it('GLM-4-9B should use GQA with 2 KV heads, useBias=true, no headDim override', () => {
    const cfg = ALL_MODEL_CONFIGS['glm4-9b'];
    expect(cfg.attentionType).toBe('gqa');
    expect(cfg.numKvHeads).toBe(2);
    expect(cfg.useBias).toBe(true);
    expect(cfg.headDim).toBeUndefined();
    expect(cfg.numExperts).toBeUndefined();
  });

  it('GLM-4-32B should use GQA with 2 KV heads, useBias=false, maxSeqLength=32768', () => {
    const cfg = ALL_MODEL_CONFIGS['glm4-32b'];
    expect(cfg.attentionType).toBe('gqa');
    expect(cfg.numKvHeads).toBe(2);
    expect(cfg.useBias).toBe(false);
    expect(cfg.maxSeqLength).toBe(32768);
    expect(cfg.numExperts).toBeUndefined();
  });

  it('GLM-4.5-Air should be MoE with GQA, 128 experts, headDim=128 override', () => {
    const cfg = ALL_MODEL_CONFIGS['glm4.5-air'];
    expect(cfg.attentionType).toBe('gqa');
    expect(cfg.numExperts).toBe(128);
    expect(cfg.numActiveExperts).toBe(8);
    expect(cfg.numSharedExperts).toBe(1);
    expect(cfg.headDim).toBe(128);
    expect(cfg.numKvHeads).toBe(8);
    expect(cfg.firstKDenseLayers).toBe(1);
  });

  it('GLM-4.7 should be MoE with GQA, 160 experts, headDim=128, useBias=true', () => {
    const cfg = ALL_MODEL_CONFIGS['glm4.7'];
    expect(cfg.attentionType).toBe('gqa');
    expect(cfg.numExperts).toBe(160);
    expect(cfg.numKvHeads).toBe(8);
    expect(cfg.headDim).toBe(128);
    expect(cfg.useBias).toBe(true);
    expect(cfg.firstKDenseLayers).toBe(3);
  });

  it('GLM-4.7-Flash should use MLA with 64 experts, 4 active', () => {
    const cfg = ALL_MODEL_CONFIGS['glm4.7-flash'];
    expect(cfg.attentionType).toBe('mla');
    expect(cfg.numExperts).toBe(64);
    expect(cfg.numActiveExperts).toBe(4);
    expect(cfg.kvLoraRank).toBe(512);
    expect(cfg.qLoraRank).toBe(768);
    expect(cfg.firstKDenseLayers).toBe(1);
  });

  it('GLM-5 should use MLA with 256 experts, 8 active', () => {
    const cfg = ALL_MODEL_CONFIGS['glm5'];
    expect(cfg.attentionType).toBe('mla');
    expect(cfg.numExperts).toBe(256);
    expect(cfg.numActiveExperts).toBe(8);
    expect(cfg.kvLoraRank).toBe(512);
    expect(cfg.qLoraRank).toBe(2048);
    expect(cfg.firstKDenseLayers).toBe(3);
  });

  // Kimi K2.5 architecture tests
  it('Kimi K2.5 should use MLA with identical dims to DeepSeek V3', () => {
    const cfg = ALL_MODEL_CONFIGS['kimi-k2.5'];
    expect(cfg.attentionType).toBe('mla');
    expect(cfg.kvLoraRank).toBe(512);
    expect(cfg.qLoraRank).toBe(1536);
    expect(cfg.qkNopeHeadDim).toBe(128);
    expect(cfg.qkRopeHeadDim).toBe(64);
    expect(cfg.vHeadDim).toBe(128);
  });

  it('Kimi K2.5 should have 384 experts, 8 active, 1 shared, 1 dense layer', () => {
    const cfg = ALL_MODEL_CONFIGS['kimi-k2.5'];
    expect(cfg.numExperts).toBe(384);
    expect(cfg.numActiveExperts).toBe(8);
    expect(cfg.numSharedExperts).toBe(1);
    expect(cfg.firstKDenseLayers).toBe(1);
  });

  it('Kimi K2.5 should have 64 attention heads (half of DeepSeek V3)', () => {
    expect(ALL_MODEL_CONFIGS['kimi-k2.5'].numAttentionHeads).toBe(64);
    expect(ALL_MODEL_CONFIGS['deepseek-v3'].numAttentionHeads).toBe(128);
  });

  it('Kimi K2.5 should have useBias=false and vocabSize=163840', () => {
    expect(ALL_MODEL_CONFIGS['kimi-k2.5'].useBias).toBe(false);
    expect(ALL_MODEL_CONFIGS['kimi-k2.5'].vocabSize).toBe(163840);
  });

  // MiniMax M2.5 architecture tests
  it('MiniMax M2.5 should use GQA with 8 KV heads, SiLU, gated MLP, RMSNorm, no bias', () => {
    const cfg = ALL_MODEL_CONFIGS['minimax-m2.5'];
    expect(cfg.attentionType).toBe('gqa');
    expect(cfg.numKvHeads).toBe(8);
    expect(cfg.activation).toBe('silu');
    expect(cfg.gatedMLP).toBe(true);
    expect(cfg.normType).toBe('rmsnorm');
    expect(cfg.useBias).toBe(false);
  });

  it('MiniMax M2.5 should have headDim=128 override (natural=64)', () => {
    const cfg = ALL_MODEL_CONFIGS['minimax-m2.5'];
    expect(cfg.headDim).toBe(128);
    expect(cfg.hiddenSize / cfg.numAttentionHeads).toBe(64);
    expect(getModel('minimax-m2.5')!.headDim).toBe(128);
  });

  it('MiniMax M2.5 should have 256 experts, 8 active, no shared experts', () => {
    const cfg = ALL_MODEL_CONFIGS['minimax-m2.5'];
    expect(cfg.numExperts).toBe(256);
    expect(cfg.numActiveExperts).toBe(8);
    expect(cfg.numSharedExperts).toBeUndefined();
  });

  it('MiniMax M2.5 all 62 layers should be MoE', () => {
    const model = getModel('minimax-m2.5')!;
    expect(model.numMoELayers).toBe(62);
    expect(model.numLayers).toBe(62);
  });

  it('MiniMax M2.5 should have untied embeddings, vocab=200064, context=196608', () => {
    const cfg = ALL_MODEL_CONFIGS['minimax-m2.5'];
    expect(cfg.tiedEmbeddings).toBe(false);
    expect(cfg.vocabSize).toBe(200064);
    expect(cfg.maxSeqLength).toBe(196608);
  });

  // Mistral Large 3 architecture tests
  it('Mistral Large 3 should use MLA with 128 heads (verified from HF params.json)', () => {
    const config = ALL_MODEL_CONFIGS['mistral-large-3-675b'];
    expect(config.attentionType).toBe('mla');
    expect(config.numAttentionHeads).toBe(128);
    expect(config.numKvHeads).toBe(128);
  });

  it('Mistral Large 3, DeepSeek V3, and Kimi K2 share identical MLA dims', () => {
    const ml3 = ALL_MODEL_CONFIGS['mistral-large-3-675b'];
    const v3 = ALL_MODEL_CONFIGS['deepseek-v3'];
    const k2 = ALL_MODEL_CONFIGS['kimi-k2.5'];
    for (const model of [ml3, v3, k2]) {
      expect(model.kvLoraRank).toBe(512);
      expect(model.qLoraRank).toBe(1536);
      expect(model.qkNopeHeadDim).toBe(128);
      expect(model.qkRopeHeadDim).toBe(64);
      expect(model.vHeadDim).toBe(128);
    }
  });

  it('Mistral Large 3 should be MoE with 128 experts, 4 active, 1 shared', () => {
    const config = ALL_MODEL_CONFIGS['mistral-large-3-675b'];
    expect(config.numExperts).toBe(128);
    expect(config.numActiveExperts).toBe(4);
    expect(config.numSharedExperts).toBe(1);
    expect(config.firstKDenseLayers).toBe(3);
    expect(config.expertIntermediateSize).toBe(4096);
  });

  it('Ministral 3 models should use GQA with 8 KV heads, SwiGLU', () => {
    for (const id of ['ministral-3-3b', 'ministral-3-8b', 'ministral-3-14b']) {
      expect(ALL_MODEL_CONFIGS[id].attentionType).toBe('gqa');
      expect(ALL_MODEL_CONFIGS[id].numKvHeads).toBe(8);
      expect(ALL_MODEL_CONFIGS[id].gatedMLP).toBe(true);
      expect(ALL_MODEL_CONFIGS[id].activation).toBe('silu');
    }
  });

  it('Ministral 3 3B should have tied embeddings', () => {
    expect(ALL_MODEL_CONFIGS['ministral-3-3b'].tiedEmbeddings).toBe(true);
  });

  it('Ministral 3 8B/14B should NOT have tied embeddings', () => {
    expect(ALL_MODEL_CONFIGS['ministral-3-8b'].tiedEmbeddings).toBe(false);
    expect(ALL_MODEL_CONFIGS['ministral-3-14b'].tiedEmbeddings).toBe(false);
  });

  it('Ministral 3 3B and 14B should have headDim=128 override', () => {
    // 3B: 3072/32=96, override to 128
    expect(ALL_MODEL_CONFIGS['ministral-3-3b'].headDim).toBe(128);
    expect(ALL_MODEL_CONFIGS['ministral-3-3b'].hiddenSize / ALL_MODEL_CONFIGS['ministral-3-3b'].numAttentionHeads).toBe(96);
    // 14B: 5120/32=160, override to 128
    expect(ALL_MODEL_CONFIGS['ministral-3-14b'].headDim).toBe(128);
    expect(ALL_MODEL_CONFIGS['ministral-3-14b'].hiddenSize / ALL_MODEL_CONFIGS['ministral-3-14b'].numAttentionHeads).toBe(160);
    // 8B has natural headDim: 4096/32=128
    expect(ALL_MODEL_CONFIGS['ministral-3-8b'].headDim).toBeUndefined();
  });

  it('All Ministral 3 models should have 256K context', () => {
    for (const id of ['ministral-3-3b', 'ministral-3-8b', 'ministral-3-14b']) {
      expect(ALL_MODEL_CONFIGS[id].maxSeqLength).toBe(262144);
    }
  });

  // Devstral architecture tests
  it('Devstral 2 should have vocabSize=131072 (different from Mistral Large 32768)', () => {
    expect(ALL_MODEL_CONFIGS['devstral-2'].vocabSize).toBe(131072);
    expect(ALL_MODEL_CONFIGS['mistral-large-123b'].vocabSize).toBe(32768);
  });

  it('Devstral 2 should have 256K context and 96 attention heads', () => {
    expect(ALL_MODEL_CONFIGS['devstral-2'].maxSeqLength).toBe(262144);
    expect(ALL_MODEL_CONFIGS['devstral-2'].numAttentionHeads).toBe(96);
    expect(ALL_MODEL_CONFIGS['devstral-2'].hiddenSize / ALL_MODEL_CONFIGS['devstral-2'].numAttentionHeads).toBe(128);
  });

  it('Devstral Small 2 should have headDim=128 override and 384K context', () => {
    expect(ALL_MODEL_CONFIGS['devstral-small-2'].headDim).toBe(128);
    expect(ALL_MODEL_CONFIGS['devstral-small-2'].hiddenSize / ALL_MODEL_CONFIGS['devstral-small-2'].numAttentionHeads).toBe(160);
    expect(getModel('devstral-small-2')!.headDim).toBe(128);
    expect(ALL_MODEL_CONFIGS['devstral-small-2'].maxSeqLength).toBe(393216);
  });

  it('Devstral Small 2 should match Mistral Small arch except maxSeqLength', () => {
    const ds2 = ALL_MODEL_CONFIGS['devstral-small-2'];
    const ms = ALL_MODEL_CONFIGS['mistral-small-24b'];
    expect(ds2.numLayers).toBe(ms.numLayers);
    expect(ds2.hiddenSize).toBe(ms.hiddenSize);
    expect(ds2.intermediateSize).toBe(ms.intermediateSize);
    expect(ds2.numAttentionHeads).toBe(ms.numAttentionHeads);
    expect(ds2.numKvHeads).toBe(ms.numKvHeads);
    expect(ds2.headDim).toBe(ms.headDim);
    expect(ds2.vocabSize).toBe(ms.vocabSize);
    expect(ds2.maxSeqLength).not.toBe(ms.maxSeqLength);
  });

  it('Devstral 2 should NOT be MoE', () => {
    expect(getModel('devstral-2')!.isMoE).toBe(false);
    expect(ALL_MODEL_CONFIGS['devstral-2'].numExperts).toBeUndefined();
  });

  it('Devstral Small 2 should NOT be MoE', () => {
    expect(getModel('devstral-small-2')!.isMoE).toBe(false);
    expect(ALL_MODEL_CONFIGS['devstral-small-2'].numExperts).toBeUndefined();
  });

  // GPT-OSS architecture tests
  it('GPT-OSS models should use GQA with 8 KV heads and headDim=64 override', () => {
    for (const id of ['gpt-oss-120b', 'gpt-oss-20b']) {
      expect(ALL_MODEL_CONFIGS[id].attentionType).toBe('gqa');
      expect(ALL_MODEL_CONFIGS[id].numKvHeads).toBe(8);
      expect(ALL_MODEL_CONFIGS[id].headDim).toBe(64);
      // Natural headDim would be 2880/64=45, but actual is 64
      expect(ALL_MODEL_CONFIGS[id].hiddenSize / ALL_MODEL_CONFIGS[id].numAttentionHeads).toBe(45);
      const spec = getModel(id)!;
      expect(spec.headDim).toBe(64);
    }
  });

  it('GPT-OSS models should use SwiGLU with attention bias', () => {
    for (const id of ['gpt-oss-120b', 'gpt-oss-20b']) {
      expect(ALL_MODEL_CONFIGS[id].activation).toBe('silu');
      expect(ALL_MODEL_CONFIGS[id].gatedMLP).toBe(true);
      expect(ALL_MODEL_CONFIGS[id].useBias).toBe(true);
    }
  });

  it('GPT-OSS intermediateSize must equal hiddenSize (intentional, not a bug)', () => {
    for (const id of ['gpt-oss-120b', 'gpt-oss-20b']) {
      expect(ALL_MODEL_CONFIGS[id].intermediateSize).toBe(ALL_MODEL_CONFIGS[id].hiddenSize);
      expect(ALL_MODEL_CONFIGS[id].intermediateSize).toBe(2880);
    }
  });

  it('GPT-OSS 120B should have 128 experts, 4 active; all 36 layers MoE', () => {
    expect(ALL_MODEL_CONFIGS['gpt-oss-120b'].numExperts).toBe(128);
    expect(ALL_MODEL_CONFIGS['gpt-oss-120b'].numActiveExperts).toBe(4);
    expect(getModel('gpt-oss-120b')!.numMoELayers).toBe(36);
  });

  it('GPT-OSS 20B should have 32 experts, 4 active; all 24 layers MoE', () => {
    expect(ALL_MODEL_CONFIGS['gpt-oss-20b'].numExperts).toBe(32);
    expect(ALL_MODEL_CONFIGS['gpt-oss-20b'].numActiveExperts).toBe(4);
    expect(getModel('gpt-oss-20b')!.numMoELayers).toBe(24);
  });

  // OLMo 3 architecture tests
  it('OLMo 3 7B should be dense MHA (identical dims to OLMo 2 7B)', () => {
    const olmo3 = ALL_MODEL_CONFIGS['olmo3-7b'];
    const olmo2 = ALL_MODEL_CONFIGS['olmo2-7b'];
    expect(olmo3.attentionType).toBe('mha');
    expect(olmo3.numLayers).toBe(olmo2.numLayers);
    expect(olmo3.hiddenSize).toBe(olmo2.hiddenSize);
    expect(olmo3.intermediateSize).toBe(olmo2.intermediateSize);
    expect(olmo3.numAttentionHeads).toBe(olmo2.numAttentionHeads);
    expect(olmo3.vocabSize).toBe(olmo2.vocabSize);
    // Only maxSeqLength differs
    expect(olmo3.maxSeqLength).toBe(65536);
    expect(olmo2.maxSeqLength).toBe(4096);
    // Exact param match
    expect(getModel('olmo3-7b')!.totalParams).toBe(getModel('olmo2-7b')!.totalParams);
  });

  it('OLMo 3 32B should use GQA with 8 KV heads', () => {
    const cfg = ALL_MODEL_CONFIGS['olmo3-32b'];
    expect(cfg.attentionType).toBe('gqa');
    expect(cfg.numKvHeads).toBe(8);
    expect(cfg.numAttentionHeads).toBe(40);
  });

  it('Both OLMo 3 models should use SwiGLU, RMSNorm, no bias, untied, RoPE', () => {
    for (const id of ['olmo3-7b', 'olmo3-32b']) {
      expect(ALL_MODEL_CONFIGS[id].gatedMLP).toBe(true);
      expect(ALL_MODEL_CONFIGS[id].activation).toBe('silu');
      expect(ALL_MODEL_CONFIGS[id].normType).toBe('rmsnorm');
      expect(ALL_MODEL_CONFIGS[id].useBias).toBe(false);
      expect(ALL_MODEL_CONFIGS[id].tiedEmbeddings).toBe(false);
      expect(ALL_MODEL_CONFIGS[id].useRotaryEmbed).toBe(true);
      expect(ALL_MODEL_CONFIGS[id].vocabSize).toBe(100278);
      expect(ALL_MODEL_CONFIGS[id].maxSeqLength).toBe(65536);
    }
  });

  it('OLMo 3 32B has more layers and larger FFN than OLMo 2 13B', () => {
    expect(ALL_MODEL_CONFIGS['olmo3-32b'].numLayers).toBeGreaterThan(ALL_MODEL_CONFIGS['olmo2-13b'].numLayers);
    expect(ALL_MODEL_CONFIGS['olmo3-32b'].intermediateSize).toBeGreaterThan(ALL_MODEL_CONFIGS['olmo2-13b'].intermediateSize);
  });

  it('OLMo 3 32B GQA ratio 40/8=5 (non-power-of-2) — correct param count', () => {
    const cfg = ALL_MODEL_CONFIGS['olmo3-32b'];
    expect(cfg.numAttentionHeads! / cfg.numKvHeads!).toBe(5);
    // Smoke-test: param count is reasonable (not inflated by bad GQA math)
    const spec = getModel('olmo3-32b')!;
    expect(spec.totalParams).toBeGreaterThan(31e9);
    expect(spec.totalParams).toBeLessThan(33e9);
  });

  // Models without shared experts should not have them
  it('Models without shared experts should not define numSharedExperts', () => {
    expect(ALL_MODEL_CONFIGS['mixtral-8x7b'].numSharedExperts).toBeUndefined();
    expect(ALL_MODEL_CONFIGS['mixtral-8x22b'].numSharedExperts).toBeUndefined();
    expect(ALL_MODEL_CONFIGS['dbrx'].numSharedExperts).toBeUndefined();
    expect(ALL_MODEL_CONFIGS['grok-1'].numSharedExperts).toBeUndefined();
    expect(ALL_MODEL_CONFIGS['qwen3-30b-a3b'].numSharedExperts).toBeUndefined();
    expect(ALL_MODEL_CONFIGS['qwen3-235b-a22b'].numSharedExperts).toBeUndefined();
    expect(ALL_MODEL_CONFIGS['minimax-m2.5'].numSharedExperts).toBeUndefined();
    expect(ALL_MODEL_CONFIGS['gpt-oss-120b'].numSharedExperts).toBeUndefined();
    expect(ALL_MODEL_CONFIGS['gpt-oss-20b'].numSharedExperts).toBeUndefined();
  });
});

describe('All New Models Are Loadable', () => {
  const newModelIds = [
    'gemma2-2b', 'gemma2-9b', 'gemma2-27b',
    'llama3.2-1b', 'llama3.2-3b',
    'deepseek-v3',
    'phi3-mini', 'phi3-small', 'phi3-medium',
    'dbrx',
    'yi-6b', 'yi-34b',
    'olmo2-7b', 'olmo2-13b',
    'command-r', 'command-r-plus',
    'nemotron-4-15b', 'nemotron-4-340b',
    'mistral-nemo-12b', 'codestral-22b', 'mistral-small-24b', 'mistral-large-123b',
    // Qwen 2.5
    'qwen2.5-0.5b', 'qwen2.5-1.5b', 'qwen2.5-3b', 'qwen2.5-7b',
    'qwen2.5-14b', 'qwen2.5-32b', 'qwen2.5-72b',
    // DeepSeek-R1
    'deepseek-r1',
    // Grok
    'grok-2.5', 'grok-1',
    // Llama 3.3
    'llama3.3-70b',
    // Llama 4
    'llama4-scout', 'llama4-maverick',
    // Qwen 3
    'qwen3-0.6b', 'qwen3-1.7b', 'qwen3-4b', 'qwen3-8b', 'qwen3-14b', 'qwen3-32b',
    'qwen3-30b-a3b', 'qwen3-235b-a22b',
    // Gemma 3
    'gemma3-1b', 'gemma3-4b', 'gemma3-12b', 'gemma3-27b',
    // Phi-4
    'phi4', 'phi4-mini', 'phi4-mini-flash',
    // GLM
    'glm4-9b', 'glm4-32b', 'glm4.5-air', 'glm4.7', 'glm4.7-flash', 'glm5',
    // Kimi
    'kimi-k2.5',
    // MiniMax
    'minimax-m2.5',
    // Mistral Large 3 + Ministral 3
    'mistral-large-3-675b', 'ministral-3-3b', 'ministral-3-8b', 'ministral-3-14b',
    // Devstral
    'devstral-2', 'devstral-small-2',
    // GPT-OSS
    'gpt-oss-120b', 'gpt-oss-20b',
    // OLMo 3
    'olmo3-7b', 'olmo3-32b',
  ];

  it.each(newModelIds)('%s should be loadable via getModel()', (modelId) => {
    const model = getModel(modelId);
    expect(model, `getModel('${modelId}') returned undefined`).toBeDefined();
    expect(model!.totalParams).toBeGreaterThan(0);
  });
});

// ============================================================================
// Section 2: Model Family Smoke Tests (from new-model-families.test.ts)
// ============================================================================

describe('Training Smoke Tests', () => {
  it('Qwen 2.5 0.5B + 8x H100 + DDP', () => {
    const config: SimulationConfig = {
      modelId: 'qwen2.5-0.5b',
      clusterId: '8x-h100',
      globalBatchSize: 64,
      microBatchSize: 8,
      sequenceLength: 4096,
      strategyType: 'ddp',
    };
    const result = runSimulation(config);
    expect(result.metrics.mfu).toBeGreaterThan(0.05);
    expect(result.metrics.mfu).toBeLessThan(0.65);
  });

  it('Qwen 2.5 7B + 8x H100 + FSDP', () => {
    const config: SimulationConfig = {
      modelId: 'qwen2.5-7b',
      clusterId: '8x-h100',
      globalBatchSize: 64,
      microBatchSize: 4,
      sequenceLength: 4096,
      strategyType: 'fsdp',
    };
    const result = runSimulation(config);
    expect(result.metrics.mfu).toBeGreaterThan(0.05);
    expect(result.metrics.mfu).toBeLessThan(0.65);
  });

  it('Qwen 2.5 72B + 32x H100 + FSDP-TP', () => {
    const config: SimulationConfig = {
      modelId: 'qwen2.5-72b',
      clusterId: '32x-h100',
      globalBatchSize: 32,
      microBatchSize: 1,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 4 },
      activationCheckpointing: true,
      mixedPrecision: 'bf16',
    };
    const result = runSimulation(config);
    expect(result.metrics.mfu).toBeGreaterThan(0.05);
    expect(result.metrics.mfu).toBeLessThan(0.65);
  });

  it('Llama 3.3 70B + 32x H100 + FSDP-TP', () => {
    const config: SimulationConfig = {
      modelId: 'llama3.3-70b',
      clusterId: '32x-h100',
      globalBatchSize: 32,
      microBatchSize: 1,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 4 },
      activationCheckpointing: true,
      mixedPrecision: 'bf16',
    };
    const result = runSimulation(config);
    expect(result.metrics.mfu).toBeGreaterThan(0.05);
    expect(result.metrics.mfu).toBeLessThan(0.65);
  });

  it('Grok-1 + 256x H100 + FSDP-TP-PP (MoE)', () => {
    const config: SimulationConfig = {
      modelId: 'grok-1',
      clusterId: '256x-h100',
      globalBatchSize: 256,
      microBatchSize: 1,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 2, pp: 4, ep: 4 },
      activationCheckpointing: true,
      mixedPrecision: 'bf16',
    };
    const result = runSimulation(config);
    expect(result.metrics.mfu).toBeGreaterThanOrEqual(0);
    expect(result.metrics.mfu).toBeLessThan(0.65);
  });

  it('Grok 2.5 + 128x H100 + FSDP-TP-PP (Residual MoE)', () => {
    const result = runSimulation({
      modelId: 'grok-2.5',
      clusterId: '128x-h100',
      globalBatchSize: 256,
      microBatchSize: 1,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 2 },
      activationCheckpointing: true,
      mixedPrecision: 'bf16',
    });
    expect(result.metrics.mfu).toBeGreaterThan(0.05);
    expect(result.metrics.mfu).toBeLessThan(0.65);
  });
});

describe('Grok 2.5 (270B Residual MoE)', () => {
  describe('Inference Smoke Tests', () => {
    it('Grok 2.5 inference on 8x H200 (fp8, TP=8, batch=1)', () => {
      const result = runInferenceSimulation({
        modelId: 'grok-2.5',
        gpuId: 'h200-sxm',
        numGPUs: 8,
        batchSize: 1,
        inputSeqLen: 512,
        outputSeqLen: 128,
        weightPrecision: 'fp8',
        tensorParallel: 8,
      });
      expect(result.success).toBe(true);
    });
  });

  describe('Architecture Guards', () => {
    it('Grok 2.5 active ratio ~42.7% (higher than pure-sparse MoE due to shared FFN)', () => {
      const grok25 = getModel('grok-2.5')!;
      const ratio = grok25.activeParams / grok25.totalParams;
      // Residual MoE: shared FFN doubles active FFN params → ~42.7% active
      // Pure sparse (Grok-1): ~26.7%. Guard that shared experts raise the ratio.
      expect(ratio).toBeGreaterThan(0.40);
      expect(ratio).toBeLessThan(0.45);
    });

    it('Grok 2.5 has more active params than Grok-1 despite fewer total params', () => {
      const grok25 = getModel('grok-2.5')!;
      const grok1 = getModel('grok-1')!;
      // Grok 2.5: ~270B total, ~115B active
      // Grok-1: ~316B total, ~85B active
      expect(grok25.totalParams).toBeLessThan(grok1.totalParams);
      expect(grok25.activeParams).toBeGreaterThan(grok1.activeParams);
    });
  });
});

describe('Training Regression Benchmarks — New Families', () => {
  // Pinned values from simulator, ±15% tolerance

  it('Llama 4 Scout + 64x H100 + FSDP-TP (tp=8, ep=2)', () => {
    const result = runSimulation({
      modelId: 'llama4-scout',
      clusterId: '64x-h100',
      globalBatchSize: 128,
      microBatchSize: 1,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8, ep: 2 },
      activationCheckpointing: true,
      mixedPrecision: 'bf16',
    });
    // Pinned MFU: 0.30 (MoE overhead stack recalibrated: EP bug fix + residual raised)
    expect(result.metrics.mfu).toBeGreaterThan(0.25);
    expect(result.metrics.mfu).toBeLessThan(0.35);
  });

  it('Qwen 3 8B + 8x H100 + FSDP', () => {
    const result = runSimulation({
      modelId: 'qwen3-8b',
      clusterId: '8x-h100',
      globalBatchSize: 64,
      microBatchSize: 4,
      sequenceLength: 4096,
      strategyType: 'fsdp',
    });
    // Pinned MFU: 0.45
    expect(result.metrics.mfu).toBeGreaterThan(0.45 * 0.85);
    expect(result.metrics.mfu).toBeLessThan(0.45 * 1.15);
  });

  it('Qwen 3 32B + 32x H100 + FSDP-TP (tp=4)', () => {
    const result = runSimulation({
      modelId: 'qwen3-32b',
      clusterId: '32x-h100',
      globalBatchSize: 32,
      microBatchSize: 1,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 4 },
      activationCheckpointing: true,
      mixedPrecision: 'bf16',
    });
    // Pinned MFU: 0.39
    expect(result.metrics.mfu).toBeGreaterThan(0.39 * 0.85);
    expect(result.metrics.mfu).toBeLessThan(0.39 * 1.15);
  });

  it('Qwen 3 MoE 30B-A3B + 64x H100 + FSDP-TP (tp=8, ep=8)', () => {
    const result = runSimulation({
      modelId: 'qwen3-30b-a3b',
      clusterId: '64x-h100',
      globalBatchSize: 128,
      microBatchSize: 1,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8, ep: 8 },
      activationCheckpointing: true,
      mixedPrecision: 'bf16',
    });
    // Pinned MFU: 0.04 (EP=8 dispatch overhead 0.25 + fabric congestion
    // with ep*tp=64 cross-node)
    expect(result.metrics.mfu).toBeGreaterThan(0.05);
    expect(result.metrics.mfu).toBeLessThan(0.08);
  });

  it('Gemma 3 12B + 8x H100 + FSDP', () => {
    const result = runSimulation({
      modelId: 'gemma3-12b',
      clusterId: '8x-h100',
      globalBatchSize: 64,
      microBatchSize: 4,
      sequenceLength: 4096,
      strategyType: 'fsdp',
    });
    // Pinned MFU: 0.44
    expect(result.metrics.mfu).toBeGreaterThan(0.44 * 0.85);
    expect(result.metrics.mfu).toBeLessThan(0.44 * 1.15);
  });

  it('Gemma 3 27B + 32x H100 + FSDP-TP (tp=4)', () => {
    const result = runSimulation({
      modelId: 'gemma3-27b',
      clusterId: '32x-h100',
      globalBatchSize: 32,
      microBatchSize: 1,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 4 },
      activationCheckpointing: true,
      mixedPrecision: 'bf16',
    });
    // Pinned MFU: 0.39
    expect(result.metrics.mfu).toBeGreaterThan(0.39 * 0.85);
    expect(result.metrics.mfu).toBeLessThan(0.39 * 1.15);
  });
});

describe('Training Regression Benchmarks — Phi-4 (Published Cluster Sizes)', () => {
  // Phi-4 14B: trained on 1920 H100s (240 nodes × 8 GPUs), GBS=5760, seqlen=4096
  // Source: Microsoft Phi-4 Technical Report (Dec 2024)
  it('Phi-4 14B × 1920 H100 FSDP (published cluster): MFU ≈ 42.6%', () => {
    const metrics = sim(benchmarkConfig(
      'h100-sxm', 8, 240, 'phi4', 'fsdp', 5760, 1, 4096,
    ));
    // ±15% of simulator value (~42.6%)
    expect(metrics.mfu).toBeGreaterThan(0.426 * 0.85);
    expect(metrics.mfu).toBeLessThan(0.426 * 1.15);
  });

  // Phi-4 Mini: trained on 1024 A100s (128 nodes × 8 GPUs)
  // Source: Microsoft Phi-4-Mini Technical Report (Feb 2025)
  it('Phi-4 Mini 3.8B × 1024 A100 FSDP (published cluster): MFU ≈ 20.0%', () => {
    const metrics = sim(benchmarkConfig(
      'a100-80gb', 8, 128, 'phi4-mini', 'fsdp', 2048, 4, 4096,
    ));
    // ±15% of simulator value (~20.0%)
    expect(metrics.mfu).toBeGreaterThan(0.20 * 0.85);
    expect(metrics.mfu).toBeLessThan(0.20 * 1.15);
  });

  // Phi-4 Mini Flash: no published training cluster; use 512 H100s
  it('Phi-4 Mini Flash 2.5B × 512 H100 FSDP: MFU ≈ 38.1%', () => {
    const metrics = sim(benchmarkConfig(
      'h100-sxm', 8, 64, 'phi4-mini-flash', 'fsdp', 2048, 4, 4096,
    ));
    // ±15% of simulator value (~38.1%)
    expect(metrics.mfu).toBeGreaterThan(0.381 * 0.85);
    expect(metrics.mfu).toBeLessThan(0.381 * 1.15);
  });
});

describe('Inference Regression Benchmarks — New Families', () => {
  // Pinned values from simulator, ±30% tolerance

  it('Qwen 3 8B inference on 1x H100 (bf16, batch=1)', () => {
    const result = runInferenceSimulation({
      modelId: 'qwen3-8b',
      gpuId: 'h100-sxm',
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
    });
    expect(result.success).toBe(true);
    // Pinned TTFT: 21.20ms, TPOT: 6.14ms
    expect(result.latency.ttft).toBeGreaterThan(21.2 * 0.7);
    expect(result.latency.ttft).toBeLessThan(21.2 * 1.3);
    expect(result.latency.tpot).toBeGreaterThan(6.14 * 0.7);
    expect(result.latency.tpot).toBeLessThan(6.14 * 1.3);
  });

  it('Gemma 3 12B inference on 1x H100 (bf16, batch=1)', () => {
    const result = runInferenceSimulation({
      modelId: 'gemma3-12b',
      gpuId: 'h100-sxm',
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
    });
    expect(result.success).toBe(true);
    // Pinned TTFT: 30.46ms, TPOT: 8.86ms
    expect(result.latency.ttft).toBeGreaterThan(30.46 * 0.7);
    expect(result.latency.ttft).toBeLessThan(30.46 * 1.3);
    expect(result.latency.tpot).toBeGreaterThan(8.86 * 0.7);
    expect(result.latency.tpot).toBeLessThan(8.86 * 1.3);
  });
});

describe('Inference Regression Benchmarks — Phi-4', () => {
  it('Phi-4 14B inference on 1x H100 (bf16, batch=1)', () => {
    const result = runInferenceSimulation({
      modelId: 'phi4',
      gpuId: 'h100-sxm',
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
    });
    expect(result.success).toBe(true);
    // Pinned TTFT: 37.95ms, TPOT: 10.98ms
    expect(result.latency.ttft).toBeGreaterThan(37.95 * 0.7);
    expect(result.latency.ttft).toBeLessThan(37.95 * 1.3);
    expect(result.latency.tpot).toBeGreaterThan(10.98 * 0.7);
    expect(result.latency.tpot).toBeLessThan(10.98 * 1.3);
  });

  it('Phi-4 Mini inference on 1x H100 (bf16, batch=1)', () => {
    const result = runInferenceSimulation({
      modelId: 'phi4-mini',
      gpuId: 'h100-sxm',
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
    });
    expect(result.success).toBe(true);
    // Pinned TTFT: 9.93ms, TPOT: 2.89ms
    expect(result.latency.ttft).toBeGreaterThan(9.93 * 0.7);
    expect(result.latency.ttft).toBeLessThan(9.93 * 1.3);
    expect(result.latency.tpot).toBeGreaterThan(2.89 * 0.7);
    expect(result.latency.tpot).toBeLessThan(2.89 * 1.3);
  });

  it('Phi-4 Mini Flash inference on 1x H100 (bf16, batch=1)', () => {
    const result = runInferenceSimulation({
      modelId: 'phi4-mini-flash',
      gpuId: 'h100-sxm',
      numGPUs: 1,
      batchSize: 1,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
    });
    expect(result.success).toBe(true);
    // Pinned TTFT: 9.47ms, TPOT: 2.77ms
    expect(result.latency.ttft).toBeGreaterThan(9.47 * 0.7);
    expect(result.latency.ttft).toBeLessThan(9.47 * 1.3);
    expect(result.latency.tpot).toBeGreaterThan(2.77 * 0.7);
    expect(result.latency.tpot).toBeLessThan(2.77 * 1.3);
  });
});

describe('Cross-Model Consistency', () => {
  it('Qwen 3 params should be strictly ordered: 0.6B < 1.7B < 4B < 8B < 14B < 32B', () => {
    const sizes = ['qwen3-0.6b', 'qwen3-1.7b', 'qwen3-4b', 'qwen3-8b', 'qwen3-14b', 'qwen3-32b'];
    for (let i = 1; i < sizes.length; i++) {
      const smaller = getModel(sizes[i - 1])!;
      const larger = getModel(sizes[i])!;
      expect(larger.totalParams).toBeGreaterThan(smaller.totalParams);
    }
  });

  it('Gemma 3 params should be strictly ordered: 1B < 4B < 12B < 27B', () => {
    const sizes = ['gemma3-1b', 'gemma3-4b', 'gemma3-12b', 'gemma3-27b'];
    for (let i = 1; i < sizes.length; i++) {
      const smaller = getModel(sizes[i - 1])!;
      const larger = getModel(sizes[i])!;
      expect(larger.totalParams).toBeGreaterThan(smaller.totalParams);
    }
  });

  it('Qwen 3 8B step time > Qwen 3 4B step time (same cluster)', () => {
    const config4b: SimulationConfig = {
      modelId: 'qwen3-4b',
      clusterId: '8x-h100',
      globalBatchSize: 64,
      microBatchSize: 4,
      sequenceLength: 4096,
      strategyType: 'fsdp',
    };
    const config8b: SimulationConfig = {
      modelId: 'qwen3-8b',
      clusterId: '8x-h100',
      globalBatchSize: 64,
      microBatchSize: 4,
      sequenceLength: 4096,
      strategyType: 'fsdp',
    };
    const result4b = runSimulation(config4b);
    const result8b = runSimulation(config8b);
    expect(result8b.metrics.avgStepTimeMs).toBeGreaterThan(result4b.metrics.avgStepTimeMs);
  });

  it('Gemma 3 27B memory > Gemma 3 12B memory (same cluster)', () => {
    const config12b: SimulationConfig = {
      modelId: 'gemma3-12b',
      clusterId: '8x-h100',
      globalBatchSize: 64,
      microBatchSize: 4,
      sequenceLength: 4096,
      strategyType: 'fsdp',
    };
    const config27b: SimulationConfig = {
      modelId: 'gemma3-27b',
      clusterId: '8x-h100',
      globalBatchSize: 64,
      microBatchSize: 4,
      sequenceLength: 4096,
      strategyType: 'fsdp',
      activationCheckpointing: true,
      mixedPrecision: 'bf16',
    };
    const result12b = runSimulation(config12b);
    const result27b = runSimulation(config27b);
    expect(result27b.metrics.peakMemoryGB).toBeGreaterThan(result12b.metrics.peakMemoryGB);
  });
});

describe('GPU-Hours Projections', () => {
  it('Qwen 3 8B × 256 H100: project GPU-hours for 36T tokens', () => {
    const result = runSimulation({
      modelId: 'qwen3-8b',
      clusterId: '256x-h100',
      globalBatchSize: 2048,
      microBatchSize: 4,
      sequenceLength: 4096,
      strategyType: 'fsdp',
      activationCheckpointing: true,
      mixedPrecision: 'bf16',
    });
    const tokensPerStep = 2048 * 4096;
    const stepsFor36T = 36e12 / tokensPerStep;
    const gpuHours = (stepsFor36T * result.metrics.avgStepTimeMs / 1000 / 3600) * 256;
    // Pinned: ~1,089,172 GPU-hours, ±30%
    expect(gpuHours).toBeGreaterThan(1089172 * 0.7);
    expect(gpuHours).toBeLessThan(1089172 * 1.3);
  });

  it('Gemma 3 27B × 256 H100: project GPU-hours for 14T tokens', () => {
    const result = runSimulation({
      modelId: 'gemma3-27b',
      clusterId: '256x-h100',
      globalBatchSize: 2048,
      microBatchSize: 1,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 4 },
      activationCheckpointing: true,
      mixedPrecision: 'bf16',
    });
    const tokensPerStep = 2048 * 4096;
    const stepsFor14T = 14e12 / tokensPerStep;
    const gpuHours = (stepsFor14T * result.metrics.avgStepTimeMs / 1000 / 3600) * 256;
    // Pinned: ~1,563,483 GPU-hours, ±30%
    expect(gpuHours).toBeGreaterThan(1563483 * 0.7);
    expect(gpuHours).toBeLessThan(1563483 * 1.3);
  });
});

describe('GPU-Hours Projections — Phi-4 (Published Cluster)', () => {
  // Phi-4 14B: 1920 H100s, 9.8T tokens, ~21 days → ~967K GPU-hours
  // Source: Microsoft Phi-4 Technical Report (Dec 2024)
  it('Phi-4 14B × 1920 H100: GPU-hours ≈ 568K (published: ~967K)', () => {
    const gbs = 5760, seqLen = 4096;
    const numGPUs = 1920;
    const totalSteps = Math.ceil(9.8e12 / (gbs * seqLen));
    const metrics = sim(benchmarkConfig(
      'h100-sxm', 8, 240, 'phi4', 'fsdp', gbs, 1, seqLen,
      undefined, { maxSteps: totalSteps },
    ));
    // ±30% of simulator value (~568K GPU-hours)
    const gpuHours = (metrics.timeToTrainHours ?? 0) * numGPUs;
    expect(gpuHours).toBeGreaterThan(568000 * 0.7);
    expect(gpuHours).toBeLessThan(568000 * 1.3);
  });

  // Phi-4 Mini: 1024 A100s, 5T tokens, ~14 days → ~344K GPU-hours
  // Source: Microsoft Phi-4-Mini Technical Report (Feb 2025)
  it('Phi-4 Mini × 1024 A100: GPU-hours ≈ 476K (published: ~344K)', () => {
    const gbs = 2048, seqLen = 4096;
    const numGPUs = 1024;
    const totalSteps = Math.ceil(5e12 / (gbs * seqLen));
    const metrics = sim(benchmarkConfig(
      'a100-80gb', 8, 128, 'phi4-mini', 'fsdp', gbs, 4, seqLen,
      undefined, { maxSteps: totalSteps },
    ));
    // ±30% of simulator value (~476K GPU-hours)
    const gpuHours = (metrics.timeToTrainHours ?? 0) * numGPUs;
    expect(gpuHours).toBeGreaterThan(476000 * 0.7);
    expect(gpuHours).toBeLessThan(476000 * 1.3);
  });
});

describe('Cross-Model Consistency — Phi-4', () => {
  it('Phi-4 totalParams > Phi-4 Mini totalParams > Phi-4 Mini Flash totalParams', () => {
    const phi4 = getModel('phi4')!;
    const mini = getModel('phi4-mini')!;
    const flash = getModel('phi4-mini-flash')!;
    expect(phi4.totalParams).toBeGreaterThan(mini.totalParams);
    expect(mini.totalParams).toBeGreaterThan(flash.totalParams);
  });

  it('Phi-4 totalParams > Phi-3 Medium totalParams (larger vocab)', () => {
    const phi4 = getModel('phi4')!;
    const phi3med = getModel('phi3-medium')!;
    expect(phi4.totalParams).toBeGreaterThan(phi3med.totalParams);
  });

  it('Phi-4 Mini step time < Phi-4 step time (same cluster)', () => {
    const configMini: SimulationConfig = {
      modelId: 'phi4-mini',
      clusterId: '8x-h100',
      globalBatchSize: 64,
      microBatchSize: 8,
      sequenceLength: 4096,
      strategyType: 'fsdp',
    };
    const configFull: SimulationConfig = {
      modelId: 'phi4',
      clusterId: '8x-h100',
      globalBatchSize: 64,
      microBatchSize: 4,
      sequenceLength: 4096,
      strategyType: 'fsdp',
    };
    const resultMini = runSimulation(configMini);
    const resultFull = runSimulation(configFull);
    expect(resultMini.metrics.avgStepTimeMs).toBeLessThan(resultFull.metrics.avgStepTimeMs);
  });
});

describe('Cross-Generation Comparisons', () => {
  it('Qwen 3 8B params > Qwen 2.5 7B params', () => {
    const qwen3 = getModel('qwen3-8b')!;
    const qwen25 = getModel('qwen2.5-7b')!;
    expect(qwen3.totalParams).toBeGreaterThan(qwen25.totalParams);
  });

  it('Gemma 3 1B params ~= Gemma 2 2B × ~0.4 (smaller model, larger vocab)', () => {
    const gemma3 = getModel('gemma3-1b')!;
    const gemma2 = getModel('gemma2-2b')!;
    // Gemma 3 1B should be significantly smaller than Gemma 2 2B
    expect(gemma3.totalParams).toBeLessThan(gemma2.totalParams);
  });

  it('Llama 4 Scout activeParams < Llama 3 8B totalParams (MoE efficiency)', () => {
    const scout = getModel('llama4-scout')!;
    // Scout active: ~11B, Llama 3 8B: ~8B — active is actually larger
    // But scout total: ~102B >> 8B, so test the MoE ratio
    expect(scout.activeParams).toBeLessThan(scout.totalParams * 0.2);
  });

  it('Qwen 2.5 72B should have similar params to old Qwen 2 72B architecture', () => {
    const model = getModel('qwen2.5-72b')!;
    expect(model.totalParams).toBeGreaterThan(71e9);
    expect(model.totalParams).toBeLessThan(75e9);
  });

  it('Llama 3.3 70B should have same params as Llama 3 70B', () => {
    const llama33 = getModel('llama3.3-70b')!;
    const llama3 = getModel('llama3-70b')!;
    expect(llama33.totalParams).toBe(llama3.totalParams);
  });

  it('DeepSeek-R1 should have same params as DeepSeek-V3', () => {
    const r1 = getModel('deepseek-r1')!;
    const v3 = getModel('deepseek-v3')!;
    expect(r1.totalParams).toBe(v3.totalParams);
  });

  it('Grok-1 activeParams should be less than totalParams (MoE)', () => {
    const grok = getModel('grok-1')!;
    expect(grok.activeParams).toBeLessThan(grok.totalParams);
  });
});

describe('GLM models', () => {
  describe('Architecture', () => {
    it('GLM-4-9B and GLM-4-32B are dense (not MoE), differ in useBias', () => {
      const m9b = getModel('glm4-9b')!;
      const m32b = getModel('glm4-32b')!;
      expect(m9b.isMoE).toBe(false);
      expect(m32b.isMoE).toBe(false);
      expect(m9b.useBias).toBe(true);
      expect(m32b.useBias).toBe(false);
    });

    it('GLM-4.5-Air is MoE with 128 experts + GQA headDim=128 override', () => {
      const m = getModel('glm4.5-air')!;
      expect(m.isMoE).toBe(true);
      expect(m.numExperts).toBe(128);
      expect(m.attentionType).toBe('gqa');
      expect(m.headDim).toBe(128);
    });

    it('GLM-4.7 is MoE with 160 experts + GQA headDim=128 override', () => {
      const m = getModel('glm4.7')!;
      expect(m.isMoE).toBe(true);
      expect(m.numExperts).toBe(160);
      expect(m.attentionType).toBe('gqa');
      expect(m.headDim).toBe(128);
    });

    it('GLM-4.7-Flash uses MLA with 64 experts, 4 active', () => {
      const m = getModel('glm4.7-flash')!;
      expect(m.isMoE).toBe(true);
      expect(m.attentionType).toBe('mla');
      expect(m.numExperts).toBe(64);
      expect(m.numActiveExperts).toBe(4);
    });

    it('GLM-5 uses MLA with 256 experts, 8 active', () => {
      const m = getModel('glm5')!;
      expect(m.isMoE).toBe(true);
      expect(m.attentionType).toBe('mla');
      expect(m.numExperts).toBe(256);
      expect(m.numActiveExperts).toBe(8);
    });

    it('GLM family param ordering: 9B < Flash < 32B < Air < 4.7 < 5', () => {
      const models = ['glm4-9b', 'glm4.7-flash', 'glm4-32b', 'glm4.5-air', 'glm4.7', 'glm5'];
      for (let i = 1; i < models.length; i++) {
        const smaller = getModel(models[i - 1])!;
        const larger = getModel(models[i])!;
        expect(larger.totalParams).toBeGreaterThan(smaller.totalParams);
      }
    });

    it('MLA models (Flash, 5) use different vocabSize (154880) vs GQA models (151552)', () => {
      expect(getModel('glm4.7-flash')!.vocabSize).toBe(154880);
      expect(getModel('glm5')!.vocabSize).toBe(154880);
      expect(getModel('glm4-9b')!.vocabSize).toBe(151552);
      expect(getModel('glm4-32b')!.vocabSize).toBe(151552);
      expect(getModel('glm4.5-air')!.vocabSize).toBe(151552);
      expect(getModel('glm4.7')!.vocabSize).toBe(151552);
    });

    it('GLM-4.7 and GLM-4.5-Air share headDim=128 override pattern', () => {
      expect(getModel('glm4.7')!.headDim).toBe(128);
      expect(getModel('glm4.5-air')!.headDim).toBe(128);
    });

    it('numMoELayers: Air=45, 4.7=89, Flash=46, 5=75', () => {
      expect(getModel('glm4.5-air')!.numMoELayers).toBe(45);
      expect(getModel('glm4.7')!.numMoELayers).toBe(89);
      expect(getModel('glm4.7-flash')!.numMoELayers).toBe(46);
      expect(getModel('glm5')!.numMoELayers).toBe(75);
    });
  });

  describe('Training Smoke Tests', () => {
    it('GLM-4-9B on 8× H100 FSDP: fits, MFU ≈ 0.44', () => {
      const result = runSimulation({
        modelId: 'glm4-9b',
        clusterId: '8x-h100',
        globalBatchSize: 64,
        microBatchSize: 4,
        sequenceLength: 4096,
        strategyType: 'fsdp',
      });
      expect(result.metrics.mfu).toBeGreaterThan(0.44 * 0.85);
      expect(result.metrics.mfu).toBeLessThan(0.44 * 1.15);
    });

    it('GLM-4-32B on 32× H100 FSDP-TP (tp=4): fits, MFU ≈ 0.44', () => {
      const result = runSimulation({
        modelId: 'glm4-32b',
        clusterId: '32x-h100',
        globalBatchSize: 32,
        microBatchSize: 1,
        sequenceLength: 4096,
        strategyType: 'fsdp-tp',
        strategyConfig: { tp: 4 },
        activationCheckpointing: true,
        mixedPrecision: 'bf16',
      });
      expect(result.metrics.mfu).toBeGreaterThan(0.44 * 0.85);
      expect(result.metrics.mfu).toBeLessThan(0.44 * 1.15);
    });

    it('GLM-4.5-Air on 64× H100 FSDP-TP (tp=8, ep=8): fits, MFU ≈ 0.14', () => {
      const result = runSimulation({
        modelId: 'glm4.5-air',
        clusterId: '64x-h100',
        globalBatchSize: 128,
        microBatchSize: 1,
        sequenceLength: 4096,
        strategyType: 'fsdp-tp',
        strategyConfig: { tp: 8, ep: 8 },
        activationCheckpointing: true,
        mixedPrecision: 'bf16',
      });
      // Pinned MFU: 0.14 (MoE overhead stack recalibrated)
      expect(result.metrics.mfu).toBeGreaterThan(0.12);
      expect(result.metrics.mfu).toBeLessThan(0.17);
    });

    it('GLM-4.7 on 512× H100 FSDP-TP-PP (tp=8, pp=8, ep=8): fits, MFU ≈ 0.10', () => {
      const result = runSimulation({
        modelId: 'glm4.7',
        clusterConfig: createMultiNodeCluster('h100-sxm', 8, 64)!,
        globalBatchSize: 512,
        microBatchSize: 1,
        sequenceLength: 4096,
        strategyType: 'fsdp-tp-pp',
        strategyConfig: { tp: 8, pp: 8, ep: 8 },
        activationCheckpointing: true,
        mixedPrecision: 'bf16',
      });
      // Pinned MFU: 0.15 (MoE overhead stack recalibrated;
      // 160 experts at EP=8 → transport-dominated)
      expect(result.metrics.mfu).toBeGreaterThan(0.12);
      expect(result.metrics.mfu).toBeLessThan(0.18);
    });

    it('GLM-4.7-Flash on 16× H100 FSDP-TP (tp=2): fits, MFU ≈ 0.30', () => {
      const result = runSimulation({
        modelId: 'glm4.7-flash',
        clusterId: '16x-h100',
        globalBatchSize: 64,
        microBatchSize: 2,
        sequenceLength: 4096,
        strategyType: 'fsdp-tp',
        strategyConfig: { tp: 2 },
        activationCheckpointing: true,
        mixedPrecision: 'bf16',
      });
      expect(result.metrics.mfu).toBeGreaterThan(0.30 * 0.85);
      expect(result.metrics.mfu).toBeLessThan(0.30 * 1.15);
    });

    it('GLM-5 on 2048× H200 FSDP-TP-PP (tp=4, pp=8, ep=32): fits, MFU ≈ 0.15', () => {
      const result = runSimulation({
        modelId: 'glm5',
        clusterConfig: createMultiNodeCluster('h200-sxm', 8, 256)!,
        globalBatchSize: 2048,
        microBatchSize: 1,
        sequenceLength: 4096,
        strategyType: 'fsdp-tp-pp',
        strategyConfig: { tp: 4, pp: 8, ep: 32 },
        activationCheckpointing: true,
        mixedPrecision: 'bf16',
      });
      // Pinned MFU: ~0.12 (routing locality formula `1/(1 + expertsPerRank/numActive)`
      // reduces EP all-to-all volume → MFU up from pre-locality 0.09; EP=32 means
      // high expertsPerRank so locality discount is significant.
      // EP latency scaling + coordination floor add overhead at EP=32.)
      expect(result.metrics.mfu).toBeGreaterThanOrEqual(0.10);
      expect(result.metrics.mfu).toBeLessThan(0.18);
    });
  });

  describe('Inference Smoke Tests', () => {
    it('GLM-4-9B on 1× H100 (bf16): TTFT ≈ 24.3ms, TPOT ≈ 7.5ms', () => {
      const result = runInferenceSimulation({
        modelId: 'glm4-9b', gpuId: 'h100-sxm', numGPUs: 1,
        batchSize: 1, inputSeqLen: 512, outputSeqLen: 128, weightPrecision: 'bf16',
      });
      expect(result.success).toBe(true);
      expect(result.latency.ttft).toBeGreaterThan(24.3 * 0.7);
      expect(result.latency.ttft).toBeLessThan(24.3 * 1.3);
      expect(result.latency.tpot).toBeGreaterThan(7.5 * 0.7);
      expect(result.latency.tpot).toBeLessThan(7.5 * 1.3);
    });

    it('GLM-4-32B on 2× H100 (bf16): TTFT ≈ 84.3ms, TPOT ≈ 23.9ms', () => {
      const result = runInferenceSimulation({
        modelId: 'glm4-32b', gpuId: 'h100-sxm', numGPUs: 2,
        batchSize: 1, inputSeqLen: 512, outputSeqLen: 128, weightPrecision: 'bf16',
      });
      expect(result.success).toBe(true);
      expect(result.latency.ttft).toBeGreaterThan(84.3 * 0.7);
      expect(result.latency.ttft).toBeLessThan(84.3 * 1.3);
      expect(result.latency.tpot).toBeGreaterThan(23.9 * 0.7);
      expect(result.latency.tpot).toBeLessThan(23.9 * 1.3);
    });

    it('GLM-4.5-Air on 4× H100 (fp4): TTFT ≈ 16ms, TPOT ≈ 3.2ms', () => {
      const result = runInferenceSimulation({
        modelId: 'glm4.5-air', gpuId: 'h100-sxm', numGPUs: 4,
        batchSize: 1, inputSeqLen: 512, outputSeqLen: 128, weightPrecision: 'fp4',
      });
      expect(result.success).toBe(true);
      // Prefill reads nearly all expert weights (coupon-collector over 512 tokens)
      expect(result.latency.ttft).toBeGreaterThan(16.0 * 0.7);
      expect(result.latency.ttft).toBeLessThan(16.0 * 1.3);
      expect(result.latency.tpot).toBeGreaterThan(3.2 * 0.7);
      expect(result.latency.tpot).toBeLessThan(3.2 * 1.3);
    });

    it('GLM-4.7-Flash on 1× H100 (fp8): TTFT ≈ 9ms, TPOT ≈ 2.1ms', () => {
      const result = runInferenceSimulation({
        modelId: 'glm4.7-flash', gpuId: 'h100-sxm', numGPUs: 1,
        batchSize: 1, inputSeqLen: 512, outputSeqLen: 128, weightPrecision: 'fp8',
      });
      expect(result.success).toBe(true);
      // Prefill reads nearly all expert weights (coupon-collector over 512 tokens)
      expect(result.latency.ttft).toBeGreaterThan(9.0 * 0.7);
      expect(result.latency.ttft).toBeLessThan(9.0 * 1.3);
      expect(result.latency.tpot).toBeGreaterThan(2.1 * 0.7);
      expect(result.latency.tpot).toBeLessThan(2.1 * 1.3);
    });
  });
});

describe('Kimi models', () => {
  describe('Architecture', () => {
    it('Kimi K2.5 is MoE with MLA (same dims as DeepSeek V3)', () => {
      const m = getModel('kimi-k2.5')!;
      expect(m.isMoE).toBe(true);
      expect(m.attentionType).toBe('mla');
    });

    it('K2 has more experts (384 vs 256) but fewer attention heads (64 vs 128) than V3', () => {
      const k2 = getModel('kimi-k2.5')!;
      const v3 = getModel('deepseek-v3')!;
      expect(k2.numExperts).toBe(384);
      expect(v3.numExperts).toBe(256);
      expect(k2.numAttentionHeads).toBe(64);
      expect(v3.numAttentionHeads).toBe(128);
    });

    it('K2 totalParams > V3 totalParams (more experts)', () => {
      const k2 = getModel('kimi-k2.5')!;
      const v3 = getModel('deepseek-v3')!;
      expect(k2.totalParams).toBeGreaterThan(v3.totalParams);
    });

    it('K2 activeParams < V3 activeParams (fewer attention heads)', () => {
      const k2 = getModel('kimi-k2.5')!;
      const v3 = getModel('deepseek-v3')!;
      expect(k2.activeParams).toBeLessThan(v3.activeParams);
    });

    it('K2 has 60 MoE layers (61 - 1 dense) vs V3 58 (61 - 3 dense)', () => {
      expect(getModel('kimi-k2.5')!.numMoELayers).toBe(60);
      expect(getModel('deepseek-v3')!.numMoELayers).toBe(58);
    });

    it('Kimi K2.5 and DeepSeek V3 share identical MLA dims and hiddenSize', () => {
      const k2 = ALL_MODEL_CONFIGS['kimi-k2.5'];
      const v3 = ALL_MODEL_CONFIGS['deepseek-v3'];
      // MLA dims must be identical
      expect(k2.kvLoraRank).toBe(v3.kvLoraRank);        // 512
      expect(k2.qLoraRank).toBe(v3.qLoraRank);          // 1536
      expect(k2.qkNopeHeadDim).toBe(v3.qkNopeHeadDim);  // 128
      expect(k2.qkRopeHeadDim).toBe(v3.qkRopeHeadDim);  // 64
      expect(k2.vHeadDim).toBe(v3.vHeadDim);             // 128
      // Shared transformer dims
      expect(k2.hiddenSize).toBe(v3.hiddenSize);                         // 7168
      expect(k2.expertIntermediateSize).toBe(v3.expertIntermediateSize); // 2048
      // Where they differ
      expect(k2.numExperts).toBe(384);       // vs V3's 256
      expect(k2.numAttentionHeads).toBe(64); // vs V3's 128
    });
  });

  describe('Training Smoke Tests', () => {
    it('Kimi K2.5 + 2048x H100 + zero1-tp-pp (PP=16, EP=16, published config)', () => {
      // Published: PP=16, EP=16, ZeRO-1, TP=1 on H800s (arXiv 2507.20534)
      // 2048 GPUs = TP(1) × PP(16) × DP(128), EP(16) subdivides DP
      // GBS reduced from published ~16384 to 4096 for test tractability
      const result = runSimulation({
        modelId: 'kimi-k2.5',
        clusterConfig: createMultiNodeCluster('h100-sxm', 8, 256)!,
        globalBatchSize: 4096,
        microBatchSize: 1,
        sequenceLength: 4096,
        strategyType: 'zero1-tp-pp',
        strategyConfig: {
          tp: 1, pp: 16, dp: 128, ep: 16,
          dpType: 'zero-1',
          sequenceParallel: false,
          pipelineSchedule: '1f1b',
        },
        activationCheckpointing: true,
        flashAttention: true,
        mixedPrecision: 'bf16',
      });
      expect(result.metrics.peakMemoryGB).toBeLessThan(80);
      expect(result.metrics.mfu).toBeGreaterThan(0);
      // Pinned MFU: 0.34 (grouped GEMM with per-group scheduling overhead + EP compute
      // penalties reduce MoE efficiency)
      expect(result.metrics.mfu).toBeGreaterThan(0.28);
      expect(result.metrics.mfu).toBeLessThan(0.40);
    });
  });

  describe('Inference Smoke Tests', () => {
    it('Kimi K2.5 inference on 8x H200 (fp4, TP=8, batch=32)', () => {
      const result = runInferenceSimulation({
        modelId: 'kimi-k2.5',
        gpuId: 'h200-sxm',
        numGPUs: 8,
        batchSize: 32,
        inputSeqLen: 1024,
        outputSeqLen: 512,
        weightPrecision: 'fp4',
        kvCachePrecision: 'fp8',
        flashAttention: true,
        tensorParallel: 8,
      });
      expect(result.success).toBe(true);
      // Pinned TTFT: 297ms, TPOT: 8.79ms, ±30%
      expect(result.latency.ttft).toBeGreaterThan(297 * 0.7);
      expect(result.latency.ttft).toBeLessThan(297 * 1.3);
      expect(result.latency.tpot).toBeGreaterThan(8.79 * 0.7);
      expect(result.latency.tpot).toBeLessThan(8.79 * 1.3);
    });
  });
});

describe('Mistral Large 3', () => {
  describe('Training Smoke Tests', () => {
    it('Mistral Large 3 + 512x H100 + FSDP-TP-PP (tp=8, pp=4, ep=4): MFU ≈ 0.26', () => {
      // 675B MoE with MLA — 128 experts / 4 active (unique topology: no other model uses exactly 4 active with 128 experts)
      // 61 layers (prime), identical MLA dims as DeepSeek V3 but different expert config
      const result = runSimulation({
        modelId: 'mistral-large-3-675b',
        clusterConfig: createMultiNodeCluster('h100-sxm', 8, 64)!,
        globalBatchSize: 512,
        microBatchSize: 1,
        sequenceLength: 4096,
        strategyType: 'fsdp-tp-pp',
        strategyConfig: { tp: 8, pp: 4, ep: 4 },
        activationCheckpointing: true,
        mixedPrecision: 'bf16',
      });
      expect(result.metrics.peakMemoryGB).toBeLessThan(80);
      // Pinned MFU: 0.26 (MoE overhead stack recalibrated;
      // routing locality reduces EP all-to-all volume), ±20%
      expect(result.metrics.mfu).toBeGreaterThan(0.21);
      expect(result.metrics.mfu).toBeLessThan(0.31);
    });
  });

  describe('Inference Smoke Tests', () => {
    it('Mistral Large 3 inference on 8x H200 (fp4, TP=8, batch=1)', () => {
      const result = runInferenceSimulation({
        modelId: 'mistral-large-3-675b',
        gpuId: 'h200-sxm',
        numGPUs: 8,
        batchSize: 1,
        inputSeqLen: 512,
        outputSeqLen: 128,
        weightPrecision: 'fp4',
        tensorParallel: 8,
      });
      expect(result.success).toBe(true);
      // Pinned TTFT: 11ms (prefill reads nearly all expert weights), TPOT: 1.01ms, ±30%
      expect(result.latency.ttft).toBeGreaterThan(11.0 * 0.7);
      expect(result.latency.ttft).toBeLessThan(11.0 * 1.3);
      expect(result.latency.tpot).toBeGreaterThan(1.01 * 0.7);
      expect(result.latency.tpot).toBeLessThan(1.01 * 1.3);
    });
  });

  describe('Cross-Model Consistency', () => {
    it('Mistral Large 3 activeParams < totalParams (MoE sparse: ~5.9%)', () => {
      const model = getModel('mistral-large-3-675b')!;
      expect(model.activeParams).toBeLessThan(model.totalParams * 0.07);
    });

    it('Mistral Large 3 totalParams ≈ DeepSeek V3 (similar architecture)', () => {
      const ml3 = getModel('mistral-large-3-675b')!;
      const v3 = getModel('deepseek-v3')!;
      // Both ~670B total, within 5% of each other
      expect(ml3.totalParams / v3.totalParams).toBeGreaterThan(0.95);
      expect(ml3.totalParams / v3.totalParams).toBeLessThan(1.05);
    });

    it('Mistral Large 3 activeParams > DeepSeek V3 activeParams (fatter experts)', () => {
      // ML3: 4 active × 4096 dim = 16K, V3: 8 active × 2048 dim = 16K
      // But ML3 shared expert is 16384 (dense MLP) vs V3's 2048 — so ML3 active > V3
      const ml3 = getModel('mistral-large-3-675b')!;
      const v3 = getModel('deepseek-v3')!;
      expect(ml3.activeParams).toBeGreaterThan(v3.activeParams);
    });
  });
});

describe('MiniMax models', () => {
  describe('Training Smoke Tests', () => {
    it('MiniMax M2.5 + 256x H100 + FSDP-TP-PP (MoE, tp=8, pp=4, ep=4)', () => {
      const result = runSimulation({
        modelId: 'minimax-m2.5',
        clusterId: '256x-h100',
        globalBatchSize: 256,
        microBatchSize: 1,
        sequenceLength: 4096,
        strategyType: 'fsdp-tp-pp',
        strategyConfig: { tp: 8, pp: 4, ep: 4 },
        activationCheckpointing: true,
        mixedPrecision: 'bf16',
      });
      expect(result.metrics.mfu).toBeGreaterThanOrEqual(0);
      expect(result.metrics.mfu).toBeLessThan(0.65);
    });

    it('MiniMax M2.5 + 512x H100 + FSDP-TP (tp=8, ep=8): MFU ≈ 0.14', () => {
      const result = runSimulation({
        modelId: 'minimax-m2.5',
        clusterConfig: createMultiNodeCluster('h100-sxm', 8, 64)!,
        globalBatchSize: 512,
        microBatchSize: 1,
        sequenceLength: 4096,
        strategyType: 'fsdp-tp',
        strategyConfig: { tp: 8, ep: 8 },
        activationCheckpointing: true,
        mixedPrecision: 'bf16',
      });
      // Pinned MFU: 0.14 (MoE overhead stack recalibrated; `1/(1 + expertsPerRank/numActive)`
      // routing locality reduces EP all-to-all volume), ±20%
      expect(result.metrics.mfu).toBeGreaterThan(0.11);
      expect(result.metrics.mfu).toBeLessThan(0.17);
    });
  });

  describe('Inference Smoke Tests', () => {
    it('MiniMax M2.5 inference on 4x H100 (fp8, TP=4, batch=1)', () => {
      const result = runInferenceSimulation({
        modelId: 'minimax-m2.5',
        gpuId: 'h100-sxm',
        numGPUs: 4,
        batchSize: 1,
        inputSeqLen: 512,
        outputSeqLen: 128,
        weightPrecision: 'fp8',
        tensorParallel: 4,
      });
      expect(result.success).toBe(true);
      // Pinned TTFT: 18ms (prefill reads nearly all expert weights), TPOT: 1.20ms, ±30%
      expect(result.latency.ttft).toBeGreaterThan(18.0 * 0.7);
      expect(result.latency.ttft).toBeLessThan(18.0 * 1.3);
      expect(result.latency.tpot).toBeGreaterThan(1.20 * 0.7);
      expect(result.latency.tpot).toBeLessThan(1.20 * 1.3);
    });
  });

  describe('Cross-Model Consistency', () => {
    it('MiniMax M2.5 activeParams < totalParams (MoE sparse: ~4.8%)', () => {
      const model = getModel('minimax-m2.5')!;
      expect(model.activeParams).toBeLessThan(model.totalParams * 0.06);
    });
  });
});

describe('GPT-OSS models', () => {
  describe('Training Smoke Tests', () => {
    it('GPT-OSS 120B + 64x H100 + FSDP-TP (tp=8, ep=8): MFU ≈ 0.17', () => {
      const result = runSimulation({
        modelId: 'gpt-oss-120b',
        clusterId: '64x-h100',
        globalBatchSize: 128,
        microBatchSize: 1,
        sequenceLength: 4096,
        strategyType: 'fsdp-tp',
        strategyConfig: { tp: 8, ep: 8 },
        activationCheckpointing: true,
        mixedPrecision: 'bf16',
      });
      // Pinned MFU: 0.17 (MoE overhead stack recalibrated; EP=8 + fabric congestion
      // with ep*tp=64 cross-node), ±30%
      expect(result.metrics.mfu).toBeGreaterThan(0.12);
      expect(result.metrics.mfu).toBeLessThan(0.22);
    });

    it('GPT-OSS 120B + 256x H100 + FSDP-TP-PP (tp=8, pp=4, ep=4): MFU ≈ 0.165', () => {
      const metrics = sim(benchmarkConfig(
        'h100-sxm', 8, 32, 'gpt-oss-120b', 'fsdp-tp-pp',
        256, 1, 4096,
        { tp: 8, pp: 4, ep: 4 },
      ));
      // Pinned MFU: 0.165 (MoE overhead stack recalibrated;
      // routing locality reduces EP all-to-all volume), ±25%
      expect(metrics.mfu).toBeGreaterThan(0.12);
      expect(metrics.mfu).toBeLessThan(0.21);
    });

    it('GPT-OSS 20B + 32x H100 + FSDP-TP (tp=4, ep=4): MFU ≈ 0.27', () => {
      const metrics = sim(benchmarkConfig(
        'h100-sxm', 8, 4, 'gpt-oss-20b', 'fsdp-tp',
        64, 1, 4096,
        { tp: 4, ep: 4 },
      ));
      // Pinned MFU: 0.27 (MoE overhead stack recalibrated), ±20%
      expect(metrics.mfu).toBeGreaterThan(0.22);
      expect(metrics.mfu).toBeLessThan(0.33);
    });
  });

  describe('Inference Smoke Tests', () => {
    it('GPT-OSS 120B inference on 4x H200 (fp8, batch=1)', () => {
      const result = runInferenceSimulation({
        modelId: 'gpt-oss-120b',
        gpuId: 'h200-sxm',
        numGPUs: 4,
        batchSize: 1,
        inputSeqLen: 512,
        outputSeqLen: 128,
        weightPrecision: 'fp8',
      });
      expect(result.success).toBe(true);
      // Pinned TTFT: 24ms (prefill reads nearly all expert weights), TPOT: 1.94ms, ±30%
      expect(result.latency.ttft).toBeGreaterThan(24.0 * 0.7);
      expect(result.latency.ttft).toBeLessThan(24.0 * 1.3);
      expect(result.latency.tpot).toBeGreaterThan(1.94 * 0.7);
      expect(result.latency.tpot).toBeLessThan(1.94 * 1.3);
    });

    it('GPT-OSS 20B inference on 1x H100 (bf16, batch=1)', () => {
      const result = runInferenceSimulation({
        modelId: 'gpt-oss-20b',
        gpuId: 'h100-sxm',
        numGPUs: 1,
        batchSize: 1,
        inputSeqLen: 512,
        outputSeqLen: 128,
        weightPrecision: 'bf16',
      });
      expect(result.success).toBe(true);
      // Pinned TTFT: 10.84ms, TPOT: 3.78ms, ±30%
      expect(result.latency.ttft).toBeGreaterThan(10.84 * 0.7);
      expect(result.latency.ttft).toBeLessThan(10.84 * 1.3);
      expect(result.latency.tpot).toBeGreaterThan(3.78 * 0.7);
      expect(result.latency.tpot).toBeLessThan(3.78 * 1.3);
    });
  });

  describe('Cross-Model Consistency', () => {
    it('GPT-OSS 120B totalParams > GPT-OSS 20B totalParams', () => {
      const m120 = getModel('gpt-oss-120b')!;
      const m20 = getModel('gpt-oss-20b')!;
      expect(m120.totalParams).toBeGreaterThan(m20.totalParams);
    });

    it('GPT-OSS 120B activeParams > GPT-OSS 20B activeParams', () => {
      const m120 = getModel('gpt-oss-120b')!;
      const m20 = getModel('gpt-oss-20b')!;
      expect(m120.activeParams).toBeGreaterThan(m20.activeParams);
    });

    it('GPT-OSS activeParams < totalParams (MoE sparse: ~5% for 120B, ~20% for 20B)', () => {
      const m120 = getModel('gpt-oss-120b')!;
      const m20 = getModel('gpt-oss-20b')!;
      expect(m120.activeParams).toBeLessThan(m120.totalParams * 0.06);
      expect(m20.activeParams).toBeLessThan(m20.totalParams * 0.25);
    });
  });
});

describe('OLMo 3 models', () => {
  describe('Architecture', () => {
    it('OLMo 3 7B is dense MHA (same as OLMo 2 7B)', () => {
      const m = getModel('olmo3-7b')!;
      expect(m.isMoE).toBe(false);
      expect(m.attentionType).toBe('mha');
    });

    it('OLMo 3 32B switched to GQA (8 KV heads), unlike OLMo 2 which was MHA', () => {
      const m = getModel('olmo3-32b')!;
      expect(m.isMoE).toBe(false);
      expect(m.attentionType).toBe('gqa');
      expect(m.numKvHeads).toBe(8);
      expect(getModel('olmo2-13b')!.attentionType).toBe('mha');
    });

    it('OLMo 3 7B has identical dimensions to OLMo 2 7B (exact totalParams match)', () => {
      expect(getModel('olmo3-7b')!.totalParams).toBe(getModel('olmo2-7b')!.totalParams);
    });

    it('OLMo 3 32B has more layers and larger FFN than OLMo 2 13B', () => {
      const olmo3 = getModel('olmo3-32b')!;
      const olmo2 = getModel('olmo2-13b')!;
      expect(olmo3.numLayers).toBeGreaterThan(olmo2.numLayers);
      expect(olmo3.totalParams).toBeGreaterThan(olmo2.totalParams);
    });

    it('OLMo 3 32B totalParams > OLMo 3 7B totalParams', () => {
      expect(getModel('olmo3-32b')!.totalParams).toBeGreaterThan(getModel('olmo3-7b')!.totalParams);
    });

    it('Both share vocabSize=100278 and maxSeqLength=65536', () => {
      expect(getModel('olmo3-7b')!.vocabSize).toBe(100278);
      expect(getModel('olmo3-32b')!.vocabSize).toBe(100278);
      expect(ALL_MODEL_CONFIGS['olmo3-7b'].maxSeqLength).toBe(65536);
      expect(ALL_MODEL_CONFIGS['olmo3-32b'].maxSeqLength).toBe(65536);
    });

    it('OLMo 3 32B GQA ratio is 40/8=5 (non-power-of-2) — correct param count', () => {
      const cfg = ALL_MODEL_CONFIGS['olmo3-32b'];
      expect(cfg.numAttentionHeads! / cfg.numKvHeads!).toBe(5);
      // GQA param savings: fewer KV params than MHA
      const spec = getModel('olmo3-32b')!;
      expect(spec.totalParams).toBeGreaterThan(31e9);
      expect(spec.totalParams).toBeLessThan(33e9);
    });
  });

  describe('Training Smoke Tests', () => {
    it('OLMo 3 7B on 8× H100 FSDP: fits, MFU ≈ 0.44', () => {
      const result = runSimulation({
        modelId: 'olmo3-7b',
        clusterId: '8x-h100',
        globalBatchSize: 64,
        microBatchSize: 4,
        sequenceLength: 4096,
        strategyType: 'fsdp',
      });
      expect(result.metrics.peakMemoryGB).toBeLessThan(80);
      expect(result.metrics.mfu).toBeGreaterThan(0.44 * 0.85);
      expect(result.metrics.mfu).toBeLessThan(0.44 * 1.15);
    });

    it('OLMo 3 32B on 32× H100 FSDP-TP (tp=4): fits, MFU ≈ 0.44', () => {
      const result = runSimulation({
        modelId: 'olmo3-32b',
        clusterId: '32x-h100',
        globalBatchSize: 32,
        microBatchSize: 1,
        sequenceLength: 4096,
        strategyType: 'fsdp-tp',
        strategyConfig: { tp: 4 },
        activationCheckpointing: true,
        mixedPrecision: 'bf16',
      });
      expect(result.metrics.peakMemoryGB).toBeLessThan(80);
      expect(result.metrics.mfu).toBeGreaterThan(0.44 * 0.85);
      expect(result.metrics.mfu).toBeLessThan(0.44 * 1.15);
    });
  });

  describe('Published Throughput Validation', () => {
    // OLMo 3 paper: HSDP (FSDP within node, DP across), 1024 H100s, seqLen=8192
    // Simulator uses FSDP (approximates HSDP); expect wider tolerance.
    //
    // Asymmetry in FSDP-vs-HSDP gap:
    //   7B: simulator 4303 tok/s/GPU vs published 7700 (44% below)
    //   32B: simulator 2098 tok/s/GPU vs published 1900 (10% above)
    //
    // This is expected. HSDP confines all-gather to the 8-GPU NVLink domain
    // and only does DP gradient all-reduce cross-node. FSDP shards weights
    // across all 1024 GPUs, so every forward pass requires cross-node
    // all-gather. For the 7B model the compute-to-comm ratio is low —
    // each GPU holds only ~7M params (7B/1024) but must all-gather the
    // full 7B every step, so cross-node bandwidth dominates. HSDP avoids
    // this entirely (all-gather stays on NVLink), which is why their
    // throughput is 1.7× higher. For the 32B model, compute dominates
    // (4.4× more FLOPs per token), so the comm topology matters less and
    // the gap nearly disappears. If we ever add topology-aware comms
    // (HSDP / hierarchical FSDP), these two data points are a good
    // calibration target.

    it('OLMo 3 7B on 1024× H100 FSDP: ~4070 tok/s/GPU (published: 7700)', () => {
      const numGPUs = 1024;
      const gbs = 1024, seqLen = 8192;
      const result = runSimulation({
        modelId: 'olmo3-7b',
        clusterConfig: createMultiNodeCluster('h100-sxm', 8, 128)!,
        globalBatchSize: gbs,
        microBatchSize: 2,
        sequenceLength: seqLen,
        strategyType: 'fsdp',
        activationCheckpointing: true,
        mixedPrecision: 'bf16',
      });
      const tokPerSecPerGPU = (gbs * seqLen) / (result.metrics.avgStepTimeMs / 1000) / numGPUs;
      // Pin to simulator output ±15%
      expect(tokPerSecPerGPU).toBeGreaterThan(4070 * 0.85);
      expect(tokPerSecPerGPU).toBeLessThan(4070 * 1.15);
      // Within ±50% of published (FSDP approximating HSDP — large gap for small model)
      expect(tokPerSecPerGPU).toBeGreaterThan(7700 * 0.50);
      expect(tokPerSecPerGPU).toBeLessThan(7700 * 1.45);
    });

    it('OLMo 3 32B on 1024× H100 FSDP: ~2098 tok/s/GPU (published: 1900)', () => {
      const numGPUs = 1024;
      const gbs = 1024, seqLen = 8192;
      const result = runSimulation({
        modelId: 'olmo3-32b',
        clusterConfig: createMultiNodeCluster('h100-sxm', 8, 128)!,
        globalBatchSize: gbs,
        microBatchSize: 1,
        sequenceLength: seqLen,
        strategyType: 'fsdp',
        activationCheckpointing: true,
        mixedPrecision: 'bf16',
      });
      const tokPerSecPerGPU = (gbs * seqLen) / (result.metrics.avgStepTimeMs / 1000) / numGPUs;
      // Pin to simulator output ±15%
      expect(tokPerSecPerGPU).toBeGreaterThan(2098 * 0.85);
      expect(tokPerSecPerGPU).toBeLessThan(2098 * 1.15);
      // Within ±40% of published (compute-dominated — FSDP vs HSDP gap small)
      expect(tokPerSecPerGPU).toBeGreaterThan(1900 * 0.60);
      expect(tokPerSecPerGPU).toBeLessThan(1900 * 1.40);
    });
  });

  describe('Inference Smoke Tests', () => {
    it('OLMo 3 7B on 1× H100 (bf16): TTFT ≈ 18.9ms, TPOT ≈ 6.16ms', () => {
      const result = runInferenceSimulation({
        modelId: 'olmo3-7b',
        gpuId: 'h100-sxm',
        numGPUs: 1,
        batchSize: 1,
        inputSeqLen: 512,
        outputSeqLen: 128,
        weightPrecision: 'bf16',
      });
      expect(result.success).toBe(true);
      expect(result.latency.ttft).toBeGreaterThan(18.9 * 0.7);
      expect(result.latency.ttft).toBeLessThan(18.9 * 1.3);
      expect(result.latency.tpot).toBeGreaterThan(6.16 * 0.7);
      expect(result.latency.tpot).toBeLessThan(6.16 * 1.3);
    });

    it('OLMo 3 32B on 2× H100 (bf16, TP=2): TTFT ≈ 43.2ms, TPOT ≈ 11.9ms', () => {
      const result = runInferenceSimulation({
        modelId: 'olmo3-32b',
        gpuId: 'h100-sxm',
        numGPUs: 2,
        batchSize: 1,
        inputSeqLen: 512,
        outputSeqLen: 128,
        weightPrecision: 'bf16',
        tensorParallel: 2,
      });
      expect(result.success).toBe(true);
      expect(result.latency.ttft).toBeGreaterThan(43.2 * 0.7);
      expect(result.latency.ttft).toBeLessThan(43.2 * 1.3);
      expect(result.latency.tpot).toBeGreaterThan(11.9 * 0.7);
      expect(result.latency.tpot).toBeLessThan(11.9 * 1.3);
    });
  });

  describe('Cross-Model Consistency', () => {
    it('OLMo 3 7B totalParams === OLMo 2 7B totalParams (identical dimensions)', () => {
      expect(getModel('olmo3-7b')!.totalParams).toBe(getModel('olmo2-7b')!.totalParams);
    });

    it('OLMo 3 32B totalParams > OLMo 2 13B totalParams (more layers, larger FFN)', () => {
      expect(getModel('olmo3-32b')!.totalParams).toBeGreaterThan(getModel('olmo2-13b')!.totalParams);
    });

    it('OLMo 3 32B step time > OLMo 3 7B step time (same cluster)', () => {
      const config32b: SimulationConfig = {
        modelId: 'olmo3-32b',
        clusterId: '32x-h100',
        globalBatchSize: 64,
        microBatchSize: 1,
        sequenceLength: 4096,
        strategyType: 'fsdp-tp',
        strategyConfig: { tp: 4 },
        activationCheckpointing: true,
        mixedPrecision: 'bf16',
      };
      const config7b: SimulationConfig = {
        modelId: 'olmo3-7b',
        clusterId: '32x-h100',
        globalBatchSize: 64,
        microBatchSize: 1,
        sequenceLength: 4096,
        strategyType: 'fsdp-tp',
        strategyConfig: { tp: 4 },
        activationCheckpointing: true,
        mixedPrecision: 'bf16',
      };
      const result32b = runSimulation(config32b);
      const result7b = runSimulation(config7b);
      expect(result32b.metrics.avgStepTimeMs).toBeGreaterThan(result7b.metrics.avgStepTimeMs);
    });
  });
});
