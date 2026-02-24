import { describe, it, expect } from 'vitest';
import { findPresetBySlug } from '../../src/stores/config.ts';

describe('findPresetBySlug', () => {
  it('returns training preset for known training slug', () => {
    const match = findPresetBySlug('nemotron-4-340b');
    expect(match).not.toBeNull();
    expect(match!.mode).toBe('training');
    expect(match!.preset.slug).toBe('nemotron-4-340b');
  });

  it('returns inference preset for known inference slug', () => {
    const match = findPresetBySlug('deepseek-v3-r1-inference');
    expect(match).not.toBeNull();
    expect(match!.mode).toBe('inference');
    expect(match!.preset.slug).toBe('deepseek-v3-r1-inference');
  });

  it('is case-insensitive', () => {
    expect(findPresetBySlug('Nemotron-4-340B')).not.toBeNull();
  });

  it('returns null for unknown slug', () => {
    expect(findPresetBySlug('nonexistent')).toBeNull();
  });

  it('finds all training slugs', () => {
    const slugs = ['llama3-405b', 'nemotron-4-340b', 'gpt3-175b', 'deepseek-v3-r1', 'llama4-maverick', 'grok-2.5', 'qwen3-32b', 'olmo3-32b'];
    for (const slug of slugs) {
      const match = findPresetBySlug(slug);
      expect(match, `Missing training slug: ${slug}`).not.toBeNull();
      expect(match!.mode).toBe('training');
    }
  });

  it('finds all inference slugs', () => {
    const slugs = ['deepseek-v3-r1-inference', 'llama3.3-70b-inference', 'llama3.3-70b-int4-inference', 'qwen3-235b-inference', 'llama3.1-8b-inference', 'llama3.1-8b-a10g-inference'];
    for (const slug of slugs) {
      const match = findPresetBySlug(slug);
      expect(match, `Missing inference slug: ${slug}`).not.toBeNull();
      expect(match!.mode).toBe('inference');
    }
  });
});
