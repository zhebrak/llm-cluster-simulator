/**
 * NarrativeBlocks — renders text with paragraph/list detection and glossary tooltips.
 * Splits on double newlines, detects bullet lists (lines starting with "- "),
 * renders each block as <p> or <ul> with <GlossaryText>.
 */

import { GlossaryText } from './GlossaryText.tsx';

interface NarrativeBlocksProps {
  text: string;
  textClass?: string;   // default "text-gray-400"
  mono?: boolean;        // default false
}

export function NarrativeBlocks({ text, textClass = 'text-gray-400', mono = false }: NarrativeBlocksProps) {
  const fontClass = mono ? ' font-mono' : '';

  return (
    <>
      {text.trim().split('\n\n').map((block, i) => {
        const lines = block.trim().split('\n');
        const isList = lines.every(l => l.trimStart().startsWith('- '));
        if (isList) {
          return (
            <ul key={i} className="space-y-1 ml-1 border-l-2 border-gray-700 pl-3">
              {lines.map((line, j) => (
                <li key={j} className={`text-sm ${textClass} leading-relaxed${fontClass}`}>
                  <GlossaryText text={line.trimStart().slice(2)} />
                </li>
              ))}
            </ul>
          );
        }
        return (
          <p key={i} className={`text-sm ${textClass} leading-relaxed${fontClass}`}>
            <GlossaryText text={block.trim()} />
          </p>
        );
      })}
    </>
  );
}
