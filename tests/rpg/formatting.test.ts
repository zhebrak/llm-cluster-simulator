/**
 * Formatting lint for RPG mission text.
 *
 * Validates:
 *   1. All {{termId}} references in mission text point to valid GLOSSARY entries
 *   2. All {{...}} tags use lowercase IDs (matching [a-z0-9-]+)
 *   3. Success narratives contain no {{...}} tags (skills shown as badges above)
 */

import { describe, it, expect } from 'vitest';
import { ALL_MISSIONS } from '../../src/rpg/missions/index.ts';
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
