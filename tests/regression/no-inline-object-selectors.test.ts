/**
 * Static guard: no inline object selectors on Zustand stores
 *
 * An inline object selector like `useConfigStore(s => ({ a: s.a }))` creates a
 * new reference every render, causing an infinite re-render loop under Zustand v5.
 *
 * Safe alternatives:
 *   - Scalar selector:   useConfigStore(s => s.mode)
 *   - useShallow:        useConfigStore(useShallow(s => ({ a: s.a })))
 *   - Whole store:       useConfigStore()  (no selector)
 *
 * This test scans all .tsx files and fails if the dangerous pattern is found.
 */

import { describe, it, expect } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';

const SRC_DIR = path.resolve(__dirname, '../../src');

// Match `useConfigStore(s =>  ({` or `useSimulationStore(s => ({` but NOT
// if preceded by `useShallow(` on the same logical call.
// The negative lookahead (?!useShallow) prevents false positives.
const DANGEROUS_PATTERN =
  /use(?:Config|Simulation)Store\(\s*(?!useShallow)\(?s\s*=>\s*\(\{/;

function getTsxFiles(dir: string): string[] {
  const results: string[] = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      results.push(...getTsxFiles(full));
    } else if (entry.name.endsWith('.tsx')) {
      results.push(full);
    }
  }
  return results;
}

describe('no inline object selectors on Zustand stores', () => {
  const files = getTsxFiles(SRC_DIR);

  for (const file of files) {
    const rel = path.relative(path.resolve(__dirname, '../..'), file);
    it(rel, () => {
      const content = fs.readFileSync(file, 'utf-8');
      const lines = content.split('\n');
      const violations: string[] = [];

      for (let i = 0; i < lines.length; i++) {
        if (DANGEROUS_PATTERN.test(lines[i])) {
          violations.push(`  line ${i + 1}: ${lines[i].trim()}`);
        }
      }

      expect(
        violations,
        `Inline object selector found — use a scalar selector or useShallow:\n${violations.join('\n')}`,
      ).toHaveLength(0);
    });
  }
});
