/**
 * Model registry for managing built-in and custom models
 */

import type { ModelConfig, ModelSpec } from '../../types/index.ts';
import { buildModelSpec } from './primitives.ts';
import { ALL_MODEL_CONFIGS } from './architectures.ts';

export interface ModelMetadata {
  id: string;
  name: string;
  family: string;
  totalParams: number;
  activeParams: number;
  isBuiltIn: boolean;
  isMoE: boolean;
  maxSeqLength: number;
  source?: 'huggingface' | 'custom' | 'built-in';
  description?: string;
}

class ModelRegistry {
  private configs: Map<string, ModelConfig> = new Map();
  private specs: Map<string, ModelSpec> = new Map();
  private metadata: Map<string, ModelMetadata> = new Map();

  constructor() {
    // Load all built-in models
    for (const [id, config] of Object.entries(ALL_MODEL_CONFIGS)) {
      this.registerBuiltIn(id, config);
    }
  }

  private registerBuiltIn(id: string, config: ModelConfig): void {
    this.configs.set(id, config);

    // Build spec with default sequence length
    const spec = buildModelSpec(config, config.maxSeqLength);
    this.specs.set(id, spec);

    // Create metadata
    const metadata: ModelMetadata = {
      id,
      name: config.name,
      family: config.family ?? 'custom',
      totalParams: spec.totalParams,
      activeParams: spec.activeParams,
      isBuiltIn: true,
      isMoE: spec.isMoE,
      maxSeqLength: config.maxSeqLength,
      source: 'built-in',
    };
    this.metadata.set(id, metadata);
  }

  /**
   * Register a custom model configuration
   */
  registerCustom(id: string, config: ModelConfig, description?: string): ModelSpec {
    this.configs.set(id, config);

    const spec = buildModelSpec(config, config.maxSeqLength);
    this.specs.set(id, spec);

    const metadata: ModelMetadata = {
      id,
      name: config.name,
      family: config.family ?? 'custom',
      totalParams: spec.totalParams,
      activeParams: spec.activeParams,
      isBuiltIn: false,
      isMoE: spec.isMoE,
      maxSeqLength: config.maxSeqLength,
      source: 'custom',
      description,
    };
    this.metadata.set(id, metadata);

    return spec;
  }

  /**
   * Get model configuration by ID
   */
  getConfig(id: string): ModelConfig | undefined {
    return this.configs.get(id);
  }

  /**
   * Get model specification by ID
   * @param seqLength Optional sequence length override
   */
  getSpec(id: string, seqLength?: number): ModelSpec | undefined {
    const config = this.configs.get(id);
    if (!config) return undefined;

    // If custom seq length, rebuild spec
    if (seqLength && seqLength !== config.maxSeqLength) {
      return buildModelSpec(config, seqLength);
    }

    return this.specs.get(id);
  }

  /**
   * Get model metadata by ID
   */
  getMetadata(id: string): ModelMetadata | undefined {
    return this.metadata.get(id);
  }

  /**
   * Get all model metadata
   */
  getAllMetadata(): ModelMetadata[] {
    return Array.from(this.metadata.values());
  }

  /**
   * Remove a custom model
   */
  remove(id: string): boolean {
    const metadata = this.metadata.get(id);
    if (!metadata || metadata.isBuiltIn) {
      return false;
    }

    this.configs.delete(id);
    this.specs.delete(id);
    this.metadata.delete(id);
    return true;
  }

}

// Singleton instance
export const modelRegistry = new ModelRegistry();

/**
 * Quick access functions
 */
export function getModel(id: string, seqLength?: number): ModelSpec | undefined {
  return modelRegistry.getSpec(id, seqLength);
}

