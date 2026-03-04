/**
 * Lint test for game task formatting consistency.
 *
 * Ensures technical terms are backticked after their first {{glossary}}
 * occurrence in each difficulty tier. Prevents formatting drift as new tasks
 * are added.
 *
 * Only checks prose sections: briefing, hints, successExplanation.
 * Skips: learningObjectives, winningCriteria, expectedChanges, setup, etc.
 */

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'fs';
import { join } from 'path';
import { GLOSSARY } from '../../src/game/glossary.ts';

const TASK_DIR = join(__dirname, '../../src/game/tasks');

const TASK_FILES = [
  'training-beginner.ts',
  'training-intermediate.ts',
  'training-advanced.ts',
  'inference-beginner.ts',
  'inference-intermediate.ts',
  'inference-advanced.ts',
];

// Technical terms that should be backticked after their first {{glossary}}
// occurrence in a tier. Ordered by specificity (longer matches first).
const TECHNICAL_TERMS = [
  // Collective operations
  'AllReduce',
  'AllGather',
  'ReduceScatter',
  // Precision formats
  'BF16',
  'FP32',
  'FP16',
  'FP8',
  'INT8',
  'INT4',
  'NF4',
  // Strategy/parallelism codes
  'FSDP',
  'DDP',
  'ZeRO-3',
  'ZeRO-1',
  // Metrics
  'MFU',
  'HFU',
  'TPOT',
  'TTFT',
  'TFLOPS',
  // Memory/hardware
  'OOM',
  'HBM',
  'SRAM',
  'NVLink',
  'PCIe',
  'InfiniBand',
  // Training config
  'AdamW',
  '1F1B',
];

// Field names that start prose content (briefing, hints, successExplanation)
const PROSE_FIELDS = /^\s*(briefing|successExplanation|hints)\s*[:=]/;

// Field names that start non-prose content (skip these sections)
const SKIP_FIELDS =
  /^\s*(learningObjectives|winningCriteria|expectedChanges|setup|concept|title|id|mode|difficulty|order)\s*[:=]/;

/**
 * Check if a term occurrence is already properly formatted:
 * - Inside backticks: `TERM`
 * - Inside a glossary tag: {{termId|TERM}}
 * - Inside an escaped backtick pair in template literals: \`TERM\`
 */
function isProperlyFormatted(line: string, index: number): boolean {
  const before = line.substring(0, index);

  // Inside a {{...}} glossary tag
  const lastOpenBrace = before.lastIndexOf('{{');
  const lastCloseBrace = before.lastIndexOf('}}');
  if (lastOpenBrace > lastCloseBrace) {
    return true;
  }

  // Inside backticks (odd number of backticks before = inside `...`)
  const backticksBefore = (before.match(/`/g) || []).length;
  if (backticksBefore % 2 === 1) {
    return true;
  }

  // Escaped backtick in template literal: \`TERM\`
  if (before.endsWith('\\`')) {
    return true;
  }

  return false;
}

/**
 * Find un-backticked occurrences of a term in a line.
 * Returns array of column positions where the term appears unformatted.
 */
function findUnbacktickedOccurrences(line: string, term: string): number[] {
  const unbackticked: number[] = [];
  let searchFrom = 0;

  while (searchFrom < line.length) {
    const idx = line.indexOf(term, searchFrom);
    if (idx === -1) break;

    const charBefore = idx > 0 ? line[idx - 1] : ' ';
    const charAfter =
      idx + term.length < line.length ? line[idx + term.length] : ' ';

    // Word boundary: not alphanumeric or underscore (allow backtick/pipe as boundary)
    const isWordBoundaryBefore =
      !/[a-zA-Z0-9_-]/.test(charBefore) ||
      charBefore === '`' ||
      charBefore === '|';
    const isWordBoundaryAfter =
      !/[a-zA-Z0-9_]/.test(charAfter) ||
      charAfter === '`' ||
      charAfter === '\\';

    if (isWordBoundaryBefore && isWordBoundaryAfter) {
      if (!isProperlyFormatted(line, idx)) {
        unbackticked.push(idx);
      }
    }

    searchFrom = idx + term.length;
  }

  return unbackticked;
}

describe('Task formatting lint', () => {
  for (const filename of TASK_FILES) {
    describe(filename, () => {
      const filePath = join(TASK_DIR, filename);
      const content = readFileSync(filePath, 'utf-8');
      const lines = content.split('\n');

      // First pass: find all glossary-tagged terms in the file
      const glossarySeenTerms = new Set<string>();
      for (const line of lines) {
        const glossaryMatches = line.matchAll(/\{\{[^}|]+(?:\|([^}]+))?\}\}/g);
        for (const match of glossaryMatches) {
          const displayText =
            match[1] || match[0].replace(/\{\{|\}\}/g, '');
          for (const term of TECHNICAL_TERMS) {
            if (displayText.includes(term)) {
              glossarySeenTerms.add(term);
            }
          }
        }
      }

      // Helper: iterate prose lines and collect violations
      function forEachProseLine(
        callback: (line: string, lineNum: number) => void,
      ) {
        let inProse = false;
        for (let i = 0; i < lines.length; i++) {
          const line = lines[i];
          if (PROSE_FIELDS.test(line)) {
            inProse = true;
          } else if (SKIP_FIELDS.test(line)) {
            inProse = false;
          } else if (/^\s*\}[,;]?\s*$/.test(line)) {
            inProse = false;
          }
          if (!inProse) continue;
          const trimmed = line.trim();
          if (trimmed.startsWith('//') || trimmed.startsWith('*')) continue;
          if (
            !line.includes("'") &&
            !line.includes('`') &&
            !line.includes('"')
          ) {
            continue;
          }
          callback(line, i);
        }
      }

      it('should not have un-backticked technical terms in prose content', () => {
        const violations: string[] = [];

        forEachProseLine((line, i) => {
          for (const term of TECHNICAL_TERMS) {
            if (!glossarySeenTerms.has(term)) continue;
            const unbackticked = findUnbacktickedOccurrences(line, term);
            for (const col of unbackticked) {
              const start = Math.max(0, col - 20);
              const end = Math.min(line.length, col + term.length + 20);
              const context = line.substring(start, end).trim();
              violations.push(
                `Line ${i + 1}, col ${col + 1}: bare "${term}" in: ...${context}...`,
              );
            }
          }
        });

        if (violations.length > 0) {
          const shown = violations.slice(0, 20);
          const remaining = violations.length - shown.length;
          let message =
            `Found ${violations.length} un-backticked technical term(s):\n` +
            shown.join('\n');
          if (remaining > 0) {
            message += `\n... and ${remaining} more`;
          }
          expect.fail(message);
        }
      });

      it('should only reference valid glossary term IDs', () => {
        const glossaryRefs = [...content.matchAll(/\{\{([a-z0-9-]+)(?:\|[^}]*)?\}\}/g)];
        const invalid = glossaryRefs
          .map(m => ({ termId: m[1], line: content.substring(0, m.index).split('\n').length }))
          .filter(({ termId }) => !GLOSSARY[termId]);

        if (invalid.length > 0) {
          expect.fail(
            `Found ${invalid.length} unknown glossary term(s):\n` +
            invalid.map(({ termId, line }) => `Line ${line}: {{${termId}}}`).join('\n')
          );
        }
      });

      it('should not have partially-backticked compound strategy names', () => {
        // Catches patterns like `FSDP`+TP or ZeRO-1+`TP` where only part is backticked
        const PARTIAL_BACKTICK =
          /`[^`]+`\s*[+-]\s*[A-Z][A-Za-z0-9-]*|[A-Z][A-Za-z0-9-]*\s*[+-]\s*`[^`]+`/g;
        const violations: string[] = [];

        forEachProseLine((line, i) => {
          let match;
          PARTIAL_BACKTICK.lastIndex = 0;
          while ((match = PARTIAL_BACKTICK.exec(line)) !== null) {
            if (!isProperlyFormatted(line, match.index)) {
              violations.push(
                `Line ${i + 1}, col ${match.index + 1}: partially-backticked "${match[0]}"`,
              );
            }
          }
        });

        if (violations.length > 0) {
          expect.fail(
            `Found ${violations.length} partially-backticked compound term(s):\n` +
            violations.join('\n'),
          );
        }
      });

      it('should not have duplicate glossary annotations within the tier', () => {
        const glossaryPattern = /\{\{([a-z0-9-]+)(?:\|[^}]*)?\}\}/g;
        const seen = new Map<string, number>(); // termId → first line number
        const duplicates: string[] = [];

        for (let i = 0; i < lines.length; i++) {
          let match;
          glossaryPattern.lastIndex = 0;
          while ((match = glossaryPattern.exec(lines[i])) !== null) {
            const termId = match[1];
            if (seen.has(termId)) {
              duplicates.push(
                `Line ${i + 1}: duplicate {{${termId}}} (first at line ${seen.get(termId)})`,
              );
            } else {
              seen.set(termId, i + 1);
            }
          }
        }

        if (duplicates.length > 0) {
          expect.fail(
            `Found ${duplicates.length} duplicate glossary annotation(s):\n` +
            duplicates.join('\n'),
          );
        }
      });

      it('should backtick config assignments (VAR=N) in prose content', () => {
        // Config abbreviations that should be backticked when used as VAR=N
        const CONFIG_PATTERN =
          /\b(TP|PP|DP|EP|CP|SP|VP|MBS|GBS|GA|batch|seq_len)=\d/g;
        const violations: string[] = [];

        forEachProseLine((line, i) => {
          let match;
          CONFIG_PATTERN.lastIndex = 0;
          while ((match = CONFIG_PATTERN.exec(line)) !== null) {
            if (!isProperlyFormatted(line, match.index)) {
              const start = Math.max(0, match.index - 15);
              const end = Math.min(line.length, match.index + 20);
              const context = line.substring(start, end).trim();
              violations.push(
                `Line ${i + 1}, col ${match.index + 1}: bare "${match[0]}" in: ...${context}...`,
              );
            }
          }
        });

        if (violations.length > 0) {
          const shown = violations.slice(0, 20);
          const remaining = violations.length - shown.length;
          let message =
            `Found ${violations.length} bare config assignment(s):\n` +
            shown.join('\n');
          if (remaining > 0) {
            message += `\n... and ${remaining} more`;
          }
          expect.fail(message);
        }
      });
    });
  }
});
