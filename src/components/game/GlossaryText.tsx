/**
 * GlossaryText — parses task strings for glossary tooltips, paper links, code spans, and emphasis.
 *
 * Syntax:
 *   {{termId}}              → glossary tooltip with default display
 *   {{termId|display text}} → glossary tooltip with custom display
 *   [link text](url)        → clickable external link (opens new tab)
 *   `code`                  → inline code span (monospace, styled)
 *   **bold text**           → bold (strong) emphasis
 *   *italic text*           → italic emphasis
 */

import { GLOSSARY } from '../../game/glossary.ts';
import { Tooltip } from '../ui/Tooltip.tsx';

// Single-pass regex matching {{term}}, [text](url), `code`, **bold**, and *italic* tokens.
// Group 1: full glossary match  → groups 2=id, 3=custom display (optional)
// Group 4: full link match      → groups 5=text, 6=url
// Group 7: full backtick match  → group 8=inner content
// Group 9: full bold match      → group 10=inner content
// Group 11: full italic match   → group 12=inner content
const TOKEN_RE =
  /(\{\{([a-z0-9-]+)(?:\|([^}]*))?\}\})|(\[([^\]]+)\]\(([^)]+)\))|(`([^`]+)`)|(\*\*([^*]+)\*\*)|(\*([^*]+)\*)/g;

interface GlossaryTextProps {
  text: string;
}

export function GlossaryText({ text }: GlossaryTextProps) {
  if (!text) return null;

  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  let key = 0;

  // Reset regex state (it's global)
  TOKEN_RE.lastIndex = 0;

  while ((match = TOKEN_RE.exec(text)) !== null) {
    // Push plain text before this match
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }

    if (match[1]) {
      // Glossary term: {{id}} or {{id|display}}
      const termId = match[2];
      const customDisplay = match[3];
      const entry = GLOSSARY[termId];

      if (entry) {
        const displayText =
          customDisplay !== undefined && customDisplay !== ''
            ? customDisplay
            : entry.display;
        parts.push(
          <Tooltip key={key++} text={entry.definition} className="inline">
            <span className="border-b border-dotted border-gray-500 cursor-help hover:border-gray-300 transition-colors">
              {displayText}
            </span>
          </Tooltip>,
        );
      } else {
        // Unknown term — render as plain text
        if (import.meta.env.DEV) {
          console.warn(`[GlossaryText] Unknown glossary term: "${termId}"`);
        }
        parts.push(
          customDisplay !== undefined && customDisplay !== ''
            ? customDisplay
            : termId,
        );
      }
    } else if (match[4]) {
      // Link: [text](url)
      const linkText = match[5];
      const url = match[6];
      parts.push(
        <a
          key={key++}
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-accent hover:text-accent/80 underline underline-offset-2"
        >
          {linkText}
        </a>,
      );
    } else if (match[7]) {
      // Backtick code span: `code`
      parts.push(
        <code key={key++} className="px-1 py-0.5 bg-gray-800 rounded text-[0.85em] font-mono text-gray-200">
          {match[8]}
        </code>,
      );
    } else if (match[9]) {
      // Bold: **text**
      parts.push(
        <strong key={key++} className="font-bold">
          {match[10]}
        </strong>,
      );
    } else if (match[11]) {
      // Italic: *text*
      parts.push(
        <em key={key++} className="italic">
          {match[12]}
        </em>,
      );
    }

    lastIndex = match.index + match[0].length;
  }

  // Push remaining plain text
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return <>{parts}</>;
}
