/**
 * Formatting lint for RPG mission text and game task text.
 *
 * Validates:
 *   1. All {{termId}} references in mission text point to valid GLOSSARY entries
 *   2. All {{...}} tags use lowercase IDs (matching [a-z0-9-]+)
 *   3. Success narratives contain no {{...}} tags (skills shown as badges above)
 *   4. No fragile hardware-specific numbers (GB/s, TFLOPS) in narrative text
 */

import { describe, it, expect } from 'vitest';
import { ALL_MISSIONS } from '../../src/rpg/missions/index.ts';
import { ALL_TASKS } from '../../src/game/tasks/index.ts';
import { GLOSSARY } from '../../src/game/glossary.ts';

/** Extract all {{...}} tags from a string */
function extractTags(text: string): { tag: string; termId: string }[] {
  const results: { tag: string; termId: string }[] = [];
  const regex = /\{\{([^}]+)\}\}/g;
  let match;
  while ((match = regex.exec(text)) !== null) {
    results.push({ tag: match[0], termId: match[1] });
  }
  return results;
}

/** Collect all prose text from a mission (briefing, hints, successNarrative) */
function getMissionProseTexts(mission: typeof ALL_MISSIONS[0]) {
  const texts: { source: string; text: string }[] = [];
  if (mission.briefing) {
    texts.push({ source: 'briefing', text: mission.briefing });
  }
  if (mission.successNarrative) {
    texts.push({ source: 'successNarrative', text: mission.successNarrative });
  }
  for (let i = 0; i < mission.hints.length; i++) {
    texts.push({ source: `hint[${i}]`, text: mission.hints[i] });
  }
  return texts;
}

describe('RPG mission formatting lint', () => {
  it('should only reference valid glossary term IDs in {{...}} tags', () => {
    const invalid: string[] = [];

    for (const mission of ALL_MISSIONS) {
      for (const { source, text } of getMissionProseTexts(mission)) {
        for (const { tag, termId } of extractTags(text)) {
          // Only check tags that look like glossary refs (lowercase)
          if (/^[a-z0-9-]+$/.test(termId) && !GLOSSARY[termId]) {
            invalid.push(`${mission.id} ${source}: ${tag} — not in GLOSSARY`);
          }
        }
      }
    }

    if (invalid.length > 0) {
      expect.fail(
        `Found ${invalid.length} unknown glossary term(s):\n` +
        invalid.join('\n'),
      );
    }
  });

  it('should use lowercase IDs in all {{...}} tags (matching [a-z0-9-]+)', () => {
    const violations: string[] = [];

    for (const mission of ALL_MISSIONS) {
      for (const { source, text } of getMissionProseTexts(mission)) {
        for (const { tag, termId } of extractTags(text)) {
          if (!/^[a-z0-9-]+$/.test(termId)) {
            violations.push(`${mission.id} ${source}: ${tag} — not lowercase`);
          }
        }
      }
    }

    if (violations.length > 0) {
      expect.fail(
        `Found ${violations.length} non-lowercase {{...}} tag(s):\n` +
        violations.join('\n'),
      );
    }
  });

  it('should not have {{...}} tags in success narratives (skills shown as badges)', () => {
    const violations: string[] = [];

    for (const mission of ALL_MISSIONS) {
      if (!mission.successNarrative) continue;
      const tags = extractTags(mission.successNarrative);
      for (const { tag } of tags) {
        violations.push(`${mission.id} successNarrative: ${tag}`);
      }
    }

    if (violations.length > 0) {
      expect.fail(
        `Found ${violations.length} {{...}} tag(s) in success narratives (skills already shown as badges):\n` +
        violations.join('\n'),
      );
    }
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Hardware-specific numbers lint (advisory)
//
// Scans RPG missions AND game tasks for catalog-style hardware numbers
// (e.g. "900 GB/s", "312 TFLOPS") that violate the no-fragile-numbers rule.
// Exempt: tasks that specifically teach hardware comparison.
// ═══════════════════════════════════════════════════════════════════════

/** Collect all game task prose texts */
function getGameTaskProseTexts(task: typeof ALL_TASKS[0]) {
  const texts: { source: string; text: string }[] = [];
  texts.push({ source: 'briefing', text: task.briefing });
  texts.push({ source: 'successExplanation', text: task.successExplanation });
  for (let i = 0; i < task.hints.length; i++) {
    texts.push({ source: `hint[${i}]`, text: task.hints[i] });
  }
  for (const obj of task.learningObjectives) {
    texts.push({ source: 'learningObjective', text: obj });
  }
  return texts;
}

// Tasks/missions that specifically teach hardware differences are exempt
const HARDWARE_COMPARISON_EXEMPT = new Set([
  'training-advanced-07', // DeepSeek V3 hardware-specific
  'training-advanced-09', // GPU Architecture Comparison
]);

const BANDWIDTH_PATTERN = /\d+\s*GB\/s/;
const TFLOPS_PATTERN = /\d+\s*TFLOPS/;

describe('Hardware-specific numbers lint (advisory)', () => {
  it('RPG missions should not contain specific bandwidth numbers (GB/s)', () => {
    const warnings: string[] = [];

    for (const mission of ALL_MISSIONS) {
      for (const { source, text } of getMissionProseTexts(mission)) {
        if (BANDWIDTH_PATTERN.test(text)) {
          warnings.push(`${mission.id} ${source}: contains bandwidth number (${text.match(BANDWIDTH_PATTERN)![0]})`);
        }
      }
    }

    if (warnings.length > 0) {
      console.warn(
        `[ADVISORY] Found ${warnings.length} bandwidth number(s) in RPG missions:\n` +
        warnings.join('\n'),
      );
    }
    expect(warnings).toEqual([]);
  });

  it('RPG missions should not contain specific TFLOPS numbers', () => {
    const warnings: string[] = [];

    for (const mission of ALL_MISSIONS) {
      for (const { source, text } of getMissionProseTexts(mission)) {
        if (TFLOPS_PATTERN.test(text)) {
          warnings.push(`${mission.id} ${source}: contains TFLOPS number (${text.match(TFLOPS_PATTERN)![0]})`);
        }
      }
    }

    if (warnings.length > 0) {
      console.warn(
        `[ADVISORY] Found ${warnings.length} TFLOPS number(s) in RPG missions:\n` +
        warnings.join('\n'),
      );
    }
    expect(warnings).toEqual([]);
  });

  it('Game tasks should not contain specific bandwidth numbers (GB/s) — except hardware comparison tasks', () => {
    const warnings: string[] = [];

    for (const task of ALL_TASKS) {
      if (HARDWARE_COMPARISON_EXEMPT.has(task.id)) continue;
      for (const { source, text } of getGameTaskProseTexts(task)) {
        if (BANDWIDTH_PATTERN.test(text)) {
          warnings.push(`${task.id} ${source}: contains bandwidth number (${text.match(BANDWIDTH_PATTERN)![0]})`);
        }
      }
    }

    if (warnings.length > 0) {
      console.warn(
        `[ADVISORY] Found ${warnings.length} bandwidth number(s) in game tasks:\n` +
        warnings.join('\n'),
      );
    }
    expect(warnings).toEqual([]);
  });

  it('Game tasks should not contain specific TFLOPS numbers — except hardware comparison tasks', () => {
    const warnings: string[] = [];

    for (const task of ALL_TASKS) {
      if (HARDWARE_COMPARISON_EXEMPT.has(task.id)) continue;
      for (const { source, text } of getGameTaskProseTexts(task)) {
        if (TFLOPS_PATTERN.test(text)) {
          warnings.push(`${task.id} ${source}: contains TFLOPS number (${text.match(TFLOPS_PATTERN)![0]})`);
        }
      }
    }

    if (warnings.length > 0) {
      console.warn(
        `[ADVISORY] Found ${warnings.length} TFLOPS number(s) in game tasks:\n` +
        warnings.join('\n'),
      );
    }
    expect(warnings).toEqual([]);
  });
});
