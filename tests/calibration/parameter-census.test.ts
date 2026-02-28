/**
 * Parameter Census Test (6b step 4)
 *
 * Asserts the parameter registry is complete and well-formed.
 * Catches drift: adding a parameter to code without updating the registry fails CI.
 */

import { describe, it, expect } from 'vitest';
import { PARAMETER_REGISTRY } from '../../src/core/parameter-registry.ts';

describe('Parameter Census', () => {
  it('registry has exactly 61 entries', () => {
    expect(PARAMETER_REGISTRY.length).toBe(61);
  });

  it('tier distribution: 11 physics, 33 grounded-empirical, 17 fitted', () => {
    const physics = PARAMETER_REGISTRY.filter(p => p.tier === 'physics').length;
    const grounded = PARAMETER_REGISTRY.filter(p => p.tier === 'grounded-empirical').length;
    const fitted = PARAMETER_REGISTRY.filter(p => p.tier === 'fitted').length;

    expect(physics).toBe(11);
    expect(grounded).toBe(33);
    expect(fitted).toBe(17);
  });

  it('every entry has non-empty mechanism and evidence', () => {
    for (const entry of PARAMETER_REGISTRY) {
      expect(entry.mechanism, `${entry.name} has empty mechanism`).toBeTruthy();
      expect(entry.evidence, `${entry.name} has empty evidence`).toBeTruthy();
    }
  });

  it('every fitted entry has non-empty affectedBenchmarks', () => {
    const fitted = PARAMETER_REGISTRY.filter(p => p.tier === 'fitted');
    for (const entry of fitted) {
      expect(
        entry.affectedBenchmarks.length,
        `Fitted param '${entry.name}' has no affectedBenchmarks`,
      ).toBeGreaterThan(0);
    }
  });

  it('no duplicate parameter names', () => {
    const names = PARAMETER_REGISTRY.map(p => p.name);
    const unique = new Set(names);
    expect(unique.size).toBe(names.length);
  });
});
