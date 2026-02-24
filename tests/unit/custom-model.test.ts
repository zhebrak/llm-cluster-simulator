/**
 * Tests for custom model validation and registry integration.
 */

import { describe, it, expect } from 'vitest';
import { validateModelJSON, KNOWN_FIELDS } from '../../src/utils/model-validator.ts';
import { modelRegistry, getModel } from '../../src/core/models/registry.ts';
import { buildModelSpec } from '../../src/core/models/primitives.ts';
import type { ModelConfig } from '../../src/types/index.ts';

const VALID_CONFIG = {
  numLayers: 32,
  hiddenSize: 4096,
  intermediateSize: 11008,
  numAttentionHeads: 32,
  vocabSize: 32000,
  maxSeqLength: 4096,
};

describe('validateModelJSON', () => {
  it('accepts valid minimal config', () => {
    const result = validateModelJSON(JSON.stringify(VALID_CONFIG), 'Test Model');
    expect(result.valid).toBe(true);
    expect(result.errors).toEqual([]);
    expect(result.config).not.toBeNull();
    expect(result.modelSpec).not.toBeNull();
    expect(result.config!.name).toBe('Test Model');
    // Param estimate should match buildModelSpec exactly
    const expectedSpec = buildModelSpec({ name: 'Test Model', ...VALID_CONFIG }, VALID_CONFIG.maxSeqLength);
    expect(result.modelSpec!.totalParams).toBe(expectedSpec.totalParams);
  });

  it('rejects invalid JSON', () => {
    const result = validateModelJSON('{ bad json', 'Test');
    expect(result.valid).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);
    expect(result.config).toBeNull();
  });

  it('rejects non-object JSON', () => {
    const result = validateModelJSON('[1,2,3]', 'Test');
    expect(result.valid).toBe(false);
    expect(result.errors[0]).toContain('object');
  });

  describe('required fields', () => {
    const requiredFields = ['numLayers', 'hiddenSize', 'intermediateSize', 'numAttentionHeads', 'vocabSize', 'maxSeqLength'];

    for (const field of requiredFields) {
      it(`errors when ${field} is missing`, () => {
        const cfg = { ...VALID_CONFIG } as Record<string, unknown>;
        delete cfg[field];
        const result = validateModelJSON(JSON.stringify(cfg), 'Test');
        expect(result.valid).toBe(false);
        expect(result.errors.some(e => e.includes(field))).toBe(true);
      });
    }

    for (const field of requiredFields) {
      it(`errors when ${field} is zero`, () => {
        const cfg = { ...VALID_CONFIG, [field]: 0 };
        const result = validateModelJSON(JSON.stringify(cfg), 'Test');
        expect(result.valid).toBe(false);
        expect(result.errors.some(e => e.includes(field))).toBe(true);
      });

      it(`errors when ${field} is negative`, () => {
        const cfg = { ...VALID_CONFIG, [field]: -1 };
        const result = validateModelJSON(JSON.stringify(cfg), 'Test');
        expect(result.valid).toBe(false);
        expect(result.errors.some(e => e.includes(field))).toBe(true);
      });
    }
  });

  it('errors when hiddenSize not divisible by numAttentionHeads', () => {
    const cfg = { ...VALID_CONFIG, hiddenSize: 4097, numAttentionHeads: 32 };
    const result = validateModelJSON(JSON.stringify(cfg), 'Test');
    expect(result.valid).toBe(false);
    expect(result.errors.some(e => e.includes('divisible'))).toBe(true);
  });

  it('errors when numKvHeads does not divide numAttentionHeads', () => {
    const cfg = { ...VALID_CONFIG, numKvHeads: 3 };
    const result = validateModelJSON(JSON.stringify(cfg), 'Test');
    expect(result.valid).toBe(false);
    expect(result.errors.some(e => e.includes('numKvHeads'))).toBe(true);
  });

  it('accepts valid numKvHeads', () => {
    const cfg = { ...VALID_CONFIG, numKvHeads: 8 };
    const result = validateModelJSON(JSON.stringify(cfg), 'Test');
    expect(result.valid).toBe(true);
  });

  describe('enum validations', () => {
    it('errors on invalid attentionType', () => {
      const cfg = { ...VALID_CONFIG, attentionType: 'xyz' };
      const result = validateModelJSON(JSON.stringify(cfg), 'Test');
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.includes('attentionType'))).toBe(true);
    });

    it('accepts valid attentionType', () => {
      for (const at of ['mha', 'mqa', 'gqa', 'mla']) {
        const cfg = { ...VALID_CONFIG, attentionType: at };
        const result = validateModelJSON(JSON.stringify(cfg), 'Test');
        expect(result.valid).toBe(true);
      }
    });

    it('errors on invalid activation', () => {
      const cfg = { ...VALID_CONFIG, activation: 'swish' };
      const result = validateModelJSON(JSON.stringify(cfg), 'Test');
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.includes('activation'))).toBe(true);
    });

    it('errors on invalid normType', () => {
      const cfg = { ...VALID_CONFIG, normType: 'batchnorm' };
      const result = validateModelJSON(JSON.stringify(cfg), 'Test');
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.includes('normType'))).toBe(true);
    });
  });

  describe('MoE validation', () => {
    it('errors when numExperts present but numActiveExperts missing', () => {
      const cfg = { ...VALID_CONFIG, numExperts: 8 };
      const result = validateModelJSON(JSON.stringify(cfg), 'Test');
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.includes('numActiveExperts'))).toBe(true);
    });

    it('errors when numActiveExperts > numExperts', () => {
      const cfg = { ...VALID_CONFIG, numExperts: 8, numActiveExperts: 16 };
      const result = validateModelJSON(JSON.stringify(cfg), 'Test');
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.includes('numActiveExperts'))).toBe(true);
    });

    it('accepts valid MoE config', () => {
      const cfg = { ...VALID_CONFIG, numExperts: 8, numActiveExperts: 2 };
      const result = validateModelJSON(JSON.stringify(cfg), 'Test');
      expect(result.valid).toBe(true);
      expect(result.modelSpec!.isMoE).toBe(true);
    });
  });

  it('warns on unknown fields but still valid', () => {
    const cfg = { ...VALID_CONFIG, foo: 123, bar: 'hello' };
    const result = validateModelJSON(JSON.stringify(cfg), 'Test');
    expect(result.valid).toBe(true);
    expect(result.warnings.some(w => w.includes('foo'))).toBe(true);
    expect(result.warnings.some(w => w.includes('bar'))).toBe(true);
  });

  it('warns on very large model', () => {
    const cfg = { ...VALID_CONFIG, hiddenSize: 131072, numAttentionHeads: 128, intermediateSize: 524288, numLayers: 200 };
    const result = validateModelJSON(JSON.stringify(cfg), 'Test');
    expect(result.valid).toBe(true);
    expect(result.warnings.some(w => w.toLowerCase().includes('large'))).toBe(true);
  });

  it('warns on suspiciously small model', () => {
    const cfg = { ...VALID_CONFIG, hiddenSize: 8, numAttentionHeads: 1, intermediateSize: 16, numLayers: 1, vocabSize: 100 };
    const result = validateModelJSON(JSON.stringify(cfg), 'Test');
    expect(result.valid).toBe(true);
    expect(result.warnings.some(w => w.toLowerCase().includes('small'))).toBe(true);
  });
});

describe('KNOWN_FIELDS sync guard', () => {
  it('covers every key in ModelConfig', () => {
    // Representative full config exercising all 31 ModelConfig fields
    const fullConfig: Required<ModelConfig> = {
      name: 'Full',
      family: 'test',
      numLayers: 32,
      hiddenSize: 4096,
      intermediateSize: 11008,
      numAttentionHeads: 32,
      numKvHeads: 8,
      headDim: 128,
      vocabSize: 32000,
      maxSeqLength: 4096,
      attentionType: 'gqa',
      normType: 'rmsnorm',
      activation: 'silu',
      gatedMLP: true,
      useBias: false,
      tiedEmbeddings: false,
      useRotaryEmbed: true,
      kvLoraRank: 512,
      qLoraRank: 1536,
      qkNopeHeadDim: 128,
      qkRopeHeadDim: 64,
      vHeadDim: 128,
      numExperts: 256,
      numActiveExperts: 8,
      expertIntermediateSize: 2048,
      moeLayerFrequency: 1,
      numSharedExperts: 1,
      sharedExpertIntermediateSize: 2048,
      firstKDenseLayers: 3,
      lastKDenseLayers: 0,
      routingDeviceLimit: 4,
    };

    const configKeys = Object.keys(fullConfig);
    const missing = configKeys.filter(k => !KNOWN_FIELDS.has(k));
    expect(missing, `KNOWN_FIELDS is missing: ${missing.join(', ')}`).toEqual([]);
  });
});

describe('Registry integration', () => {
  it('registerCustom makes model available via getModel', () => {
    const config: ModelConfig = { name: 'Custom Test', ...VALID_CONFIG };
    modelRegistry.registerCustom('custom-1', config);
    const spec = getModel('custom-1');
    expect(spec).not.toBeUndefined();
    expect(spec!.name).toBe('Custom Test');
    expect(spec!.totalParams).toBeGreaterThan(0);
  });

  it('re-register with different config updates spec', () => {
    const config1: ModelConfig = { name: 'Custom V1', ...VALID_CONFIG };
    modelRegistry.registerCustom('custom-1', config1);
    const spec1 = getModel('custom-1')!;

    const config2: ModelConfig = { name: 'Custom V2', ...VALID_CONFIG, numLayers: 64 };
    modelRegistry.registerCustom('custom-1', config2);
    const spec2 = getModel('custom-1')!;

    expect(spec2.name).toBe('Custom V2');
    expect(spec2.numLayers).toBe(64);
    expect(spec2.totalParams).toBeGreaterThan(spec1.totalParams);
  });

  it('getConfig returns the registered ModelConfig', () => {
    const config: ModelConfig = { name: 'Config Test', ...VALID_CONFIG };
    modelRegistry.registerCustom('custom-1', config);
    const retrieved = modelRegistry.getConfig('custom-1');
    expect(retrieved).toEqual(config);
  });

  it('supports multiple custom models with unique IDs', () => {
    const config1: ModelConfig = { name: 'Model A', ...VALID_CONFIG };
    const config2: ModelConfig = { name: 'Model B', ...VALID_CONFIG, numLayers: 48 };
    const config3: ModelConfig = { name: 'Model C', ...VALID_CONFIG, hiddenSize: 8192, numAttentionHeads: 64 };

    modelRegistry.registerCustom('custom-1', config1);
    modelRegistry.registerCustom('custom-2', config2);
    modelRegistry.registerCustom('custom-3', config3);

    expect(getModel('custom-1')!.name).toBe('Model A');
    expect(getModel('custom-2')!.name).toBe('Model B');
    expect(getModel('custom-3')!.name).toBe('Model C');
    expect(getModel('custom-2')!.numLayers).toBe(48);
    expect(getModel('custom-3')!.hiddenSize).toBe(8192);
  });

  it('remove deletes a custom model', () => {
    const config: ModelConfig = { name: 'To Delete', ...VALID_CONFIG };
    modelRegistry.registerCustom('custom-99', config);
    expect(getModel('custom-99')).not.toBeUndefined();

    const removed = modelRegistry.remove('custom-99');
    expect(removed).toBe(true);
    expect(getModel('custom-99')).toBeUndefined();
  });

  it('remove refuses to delete built-in models', () => {
    const removed = modelRegistry.remove('llama3-405b');
    expect(removed).toBe(false);
    expect(getModel('llama3-405b')).not.toBeUndefined();
  });
});
