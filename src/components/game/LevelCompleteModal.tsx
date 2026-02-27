/**
 * Level complete modal — shown once when all tasks in a level are done
 */

import { Trophy, ArrowLeft, RotateCcw } from 'lucide-react';
import { MODE_LABELS, DIFFICULTY_LABELS } from '../../game/constants.ts';
import type { GameMode, GameDifficulty } from '../../game/types.ts';

interface LevelCompleteModalProps {
  mode: GameMode;
  difficulty: GameDifficulty;
  onBack: () => void;
  onReplay: () => void;
}

export function LevelCompleteModal({ mode, difficulty, onBack, onReplay }: LevelCompleteModalProps) {
  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div
        className="bg-gray-900 border border-yellow-500/30 rounded-xl p-6 max-w-md w-full mx-4 text-center"
        onClick={e => e.stopPropagation()}
      >
        <div className="w-16 h-16 rounded-full bg-yellow-500/20 flex items-center justify-center mx-auto mb-4">
          <Trophy className="w-8 h-8 text-yellow-400" />
        </div>

        <h3 className="text-xl font-semibold text-white mb-2">
          Level Complete!
        </h3>

        <p className="text-sm text-gray-400 mb-6">
          You've completed all {MODE_LABELS[mode]} {DIFFICULTY_LABELS[difficulty]} challenges.
          {difficulty !== 'advanced'
            ? ` Try the next difficulty level to learn more advanced concepts.`
            : ` You've mastered the ${MODE_LABELS[mode].toLowerCase()} curriculum!`
          }
        </p>

        <div className="flex items-center justify-center gap-3">
          <button
            onClick={onReplay}
            className="flex items-center justify-center gap-2 px-4 py-2.5 bg-accent hover:bg-accent/80 text-white rounded-lg font-medium transition-colors cursor-pointer"
          >
            <RotateCcw className="w-4 h-4" />
            Replay Level
          </button>
          <button
            onClick={onBack}
            className="flex items-center justify-center gap-2 px-4 py-2.5 bg-gray-800 hover:bg-gray-700 text-white rounded-lg font-medium transition-colors cursor-pointer"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Levels
          </button>
        </div>
      </div>
    </div>
  );
}
