/**
 * Validates user-provided ModelConfig JSON for the custom model editor.
 */

import type { ModelConfig, ModelSpec } from '../types/index.ts';
import { buildModelSpec } from '../core/models/primitives.ts';

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  config: ModelConfig | null;
  modelSpec: ModelSpec | null;
}

const REQUIRED_FIELDS = [
  'numLayers',
  'hiddenSize',
  'intermediateSize',
  'numAttentionHeads',
  'vocabSize',
  'maxSeqLength',
] as const;

export const KNOWN_FIELDS = new Set([
  'name', 'family',
  'numLayers', 'hiddenSize', 'intermediateSize', 'numAttentionHeads',
  'numKvHeads', 'headDim', 'vocabSize', 'maxSeqLength',
  'attentionType', 'normType', 'activation', 'gatedMLP', 'useBias',
  'tiedEmbeddings', 'useRotaryEmbed',
  'kvLoraRank', 'qLoraRank', 'qkNopeHeadDim', 'qkRopeHeadDim', 'vHeadDim',
  'numExperts', 'numActiveExperts', 'expertIntermediateSize',
  'moeLayerFrequency', 'numSharedExperts', 'sharedExpertIntermediateSize',
  'firstKDenseLayers', 'lastKDenseLayers', 'routingDeviceLimit',
]);

const VALID_ATTENTION_TYPES = new Set(['mha', 'mqa', 'gqa', 'mla']);
const VALID_ACTIVATIONS = new Set(['gelu', 'silu', 'relu']);
const VALID_NORM_TYPES = new Set(['layernorm', 'rmsnorm']);

export function validateModelJSON(json: string, name: string): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  // 1. Parse JSON
  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(json);
  } catch (e) {
    const msg = e instanceof Error ? e.message : 'Invalid JSON';
    return { valid: false, errors: [msg], warnings: [], config: null, modelSpec: null };
  }

  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    return { valid: false, errors: ['JSON must be an object'], warnings: [], config: null, modelSpec: null };
  }

  // 2. Check unknown fields
  for (const key of Object.keys(parsed)) {
    if (!KNOWN_FIELDS.has(key)) {
      warnings.push(`Unknown field: ${key}`);
    }
  }

  // 3. Required fields
  for (const field of REQUIRED_FIELDS) {
    const val = parsed[field];
    if (val === undefined || val === null) {
      errors.push(`${field}: required`);
    } else if (typeof val !== 'number' || !Number.isFinite(val) || val <= 0) {
      errors.push(`${field}: must be a positive number`);
    } else if (!Number.isInteger(val)) {
      errors.push(`${field}: must be an integer`);
    }
  }

  // If required fields have errors, stop early
  if (errors.length > 0) {
    return { valid: false, errors, warnings, config: null, modelSpec: null };
  }

  const hiddenSize = parsed.hiddenSize as number;
  const numAttentionHeads = parsed.numAttentionHeads as number;

  // 4. hiddenSize divisible by numAttentionHeads
  if (hiddenSize % numAttentionHeads !== 0) {
    errors.push(`hiddenSize (${hiddenSize}) must be divisible by numAttentionHeads (${numAttentionHeads})`);
  }

  // 5. numKvHeads
  if (parsed.numKvHeads !== undefined) {
    const kv = parsed.numKvHeads as number;
    if (typeof kv !== 'number' || !Number.isInteger(kv) || kv <= 0) {
      errors.push('numKvHeads: must be a positive integer');
    } else if (numAttentionHeads % kv !== 0) {
      errors.push(`numKvHeads (${kv}) must divide numAttentionHeads (${numAttentionHeads})`);
    }
  }

  // 6. Enum validations
  if (parsed.attentionType !== undefined && !VALID_ATTENTION_TYPES.has(parsed.attentionType as string)) {
    errors.push(`attentionType: must be one of ${[...VALID_ATTENTION_TYPES].join(', ')}`);
  }
  if (parsed.activation !== undefined && !VALID_ACTIVATIONS.has(parsed.activation as string)) {
    errors.push(`activation: must be one of ${[...VALID_ACTIVATIONS].join(', ')}`);
  }
  if (parsed.normType !== undefined && !VALID_NORM_TYPES.has(parsed.normType as string)) {
    errors.push(`normType: must be one of ${[...VALID_NORM_TYPES].join(', ')}`);
  }

  // 7. MoE validation
  if (parsed.numExperts !== undefined) {
    const ne = parsed.numExperts as number;
    if (typeof ne !== 'number' || !Number.isInteger(ne) || ne <= 0) {
      errors.push('numExperts: must be a positive integer');
    }
    if (parsed.numActiveExperts === undefined) {
      errors.push('numActiveExperts: required when numExperts is set');
    } else {
      const nae = parsed.numActiveExperts as number;
      if (typeof nae !== 'number' || !Number.isInteger(nae) || nae <= 0) {
        errors.push('numActiveExperts: must be a positive integer');
      } else if (typeof ne === 'number' && nae > ne) {
        errors.push(`numActiveExperts (${nae}) must be ≤ numExperts (${ne})`);
      }
    }
  }

  if (errors.length > 0) {
    return { valid: false, errors, warnings, config: null, modelSpec: null };
  }

  // Build the ModelConfig — name from input is authoritative
  const config: ModelConfig = { ...parsed, name } as unknown as ModelConfig;

  // Build ModelSpec
  let modelSpec: ModelSpec;
  try {
    modelSpec = buildModelSpec(config, config.maxSeqLength);
  } catch (e) {
    const msg = e instanceof Error ? e.message : 'Failed to build model spec';
    return { valid: false, errors: [`Build error: ${msg}`], warnings, config, modelSpec: null };
  }

  // Sanity checks
  if (modelSpec.totalParams > 50e12) {
    warnings.push('Very large model (>50T parameters)');
  }
  if (modelSpec.totalParams < 1e6) {
    warnings.push('Suspiciously small model (<1M parameters)');
  }

  return { valid: true, errors: [], warnings, config, modelSpec };
}
