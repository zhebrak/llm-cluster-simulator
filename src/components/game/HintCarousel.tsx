/**
 * HintCarousel — numbered hint dots, reveal button, and active hint card.
 * Controlled component: parent manages hint state (useState/useEffect).
 */

import { Lightbulb } from 'lucide-react';
import { GlossaryText } from './GlossaryText.tsx';

const themes = {
  yellow: {
    dotActive: 'bg-yellow-400/30 text-yellow-300',
    revealBtn: 'text-yellow-400/70 hover:text-yellow-400',
    card: 'text-yellow-200/80 bg-yellow-900/20 border border-yellow-800/30',
    icon: 'text-yellow-400',
  },
  amber: {
    dotActive: 'bg-amber-400/30 text-amber-300',
    revealBtn: 'text-amber-400/70 hover:text-amber-300',
    card: 'text-amber-200/80 bg-amber-900/20 border border-amber-700/30',
    icon: 'text-amber-400',
  },
} as const;

interface HintCarouselProps {
  hints: string[];
  hintsRevealed: number;
  activeHintIndex: number;
  onSelectHint: (index: number) => void;
  onRevealNext: () => void;
  hasMore: boolean;
  theme?: 'amber' | 'yellow';
  mono?: boolean;
}

export function HintCarousel({
  hints,
  hintsRevealed,
  activeHintIndex,
  onSelectHint,
  onRevealNext,
  hasMore,
  theme = 'yellow',
  mono = false,
}: HintCarouselProps) {
  const t = themes[theme];
  const fontClass = mono ? ' font-mono' : '';

  return (
    <>
      {/* Header row: dots + show-hint button */}
      <div className="flex items-center gap-1.5 mb-1.5">
        {hintsRevealed > 0 && <span className={`text-sm text-gray-400${fontClass}`}>Hints</span>}
        {hintsRevealed > 1 && hints.slice(0, hintsRevealed).map((_, i) => (
          <button
            key={i}
            onClick={() => onSelectHint(i)}
            className={`w-5 h-5 rounded-full text-xs cursor-pointer transition-colors${fontClass} ${
              i === activeHintIndex ? t.dotActive : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
            }`}
          >
            {i + 1}
          </button>
        ))}
        {hasMore && (
          <button
            onClick={onRevealNext}
            className={`text-sm ${t.revealBtn} cursor-pointer flex items-center gap-1 ml-auto${fontClass}`}
          >
            <Lightbulb className="w-3.5 h-3.5" />
            Show hint
          </button>
        )}
      </div>

      {/* Active hint card */}
      {hintsRevealed > 0 && activeHintIndex < hints.length && (
        <div className={`text-sm ${t.card} rounded px-2.5 py-1.5${fontClass} break-words`}>
          <Lightbulb className={`w-3.5 h-3.5 inline mr-1 ${t.icon}`} />
          <GlossaryText text={hints[activeHintIndex].trim()} />
        </div>
      )}
    </>
  );
}
