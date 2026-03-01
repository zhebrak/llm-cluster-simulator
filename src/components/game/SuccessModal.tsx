/**
 * Success modal — celebration + explanation + next task button
 */

import { Check, ArrowRight, Trophy } from 'lucide-react';
import { useGameStore } from '../../stores/game.ts';
import { getTaskById, getTasksForLevel } from '../../game/tasks/index.ts';
import { CriteriaChecklist } from './CriteriaChecklist.tsx';
import { GlossaryText } from './GlossaryText.tsx';

export function SuccessModal() {
  const activeTaskId = useGameStore(s => s.activeTaskId);
  const activeMode = useGameStore(s => s.activeMode);
  const activeDifficulty = useGameStore(s => s.activeDifficulty);
  const lastValidation = useGameStore(s => s.lastValidation);
  const acknowledgeSuccess = useGameStore(s => s.acknowledgeSuccess);
  const dismissSuccess = useGameStore(s => s.dismissSuccess);

  if (!activeTaskId || !activeMode || !activeDifficulty) return null;
  const task = getTaskById(activeTaskId);
  if (!task) return null;

  const tasks = getTasksForLevel(activeMode, activeDifficulty);
  const currentIndex = tasks.findIndex(t => t.id === activeTaskId);
  const isLast = currentIndex >= tasks.length - 1;

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div
        className="bg-gray-900 border border-green-500/30 rounded-xl p-6 max-w-lg w-full mx-4 max-h-[80vh] overflow-y-auto"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-full bg-teal-500/20 flex items-center justify-center">
            <Check className="w-6 h-6 text-accent" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white">{task.title}</h3>
            <p className="text-sm text-accent">Challenge Complete</p>
          </div>
        </div>

        {/* Criteria recap */}
        <div className="mb-4 p-3 bg-gray-800/50 rounded-lg">
          <CriteriaChecklist criteria={task.winningCriteria} validation={lastValidation} />
        </div>

        {/* Explanation */}
        <div className="mb-6">
          <h4 className="text-sm font-medium text-gray-300 mb-2">What you learned</h4>
          <div className="space-y-2">
            {task.successExplanation.trim().split('\n\n').map((block, i) => {
              const lines = block.trim().split('\n');
              const isList = lines.every(l => l.trimStart().startsWith('- '));
              if (isList) {
                return (
                  <ul key={i} className="space-y-1 ml-1 border-l-2 border-gray-700 pl-3">
                    {lines.map((line, j) => (
                      <li key={j} className="text-sm text-gray-400 leading-relaxed">
                        <GlossaryText text={line.trimStart().slice(2)} />
                      </li>
                    ))}
                  </ul>
                );
              }
              return (
                <p key={i} className="text-sm text-gray-400 leading-relaxed">
                  <GlossaryText text={block.trim()} />
                </p>
              );
            })}
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex gap-2">
          <button
            onClick={dismissSuccess}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 text-gray-400 hover:text-white border border-gray-700 hover:border-gray-600 rounded-lg font-medium transition-colors cursor-pointer"
          >
            Explore Results
          </button>
          <button
            onClick={acknowledgeSuccess}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 bg-accent hover:bg-accent/80 text-white rounded-lg font-medium transition-colors cursor-pointer"
          >
            {isLast ? (
              <>
                <Trophy className="w-4 h-4" />
                Complete Level
              </>
            ) : (
              <>
                Next Task
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
