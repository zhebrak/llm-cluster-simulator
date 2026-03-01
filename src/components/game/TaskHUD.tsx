/**
 * Task HUD — collapsible floating card showing breadcrumb, objectives, hints during active task
 */

import { useState, useEffect } from 'react';
import { ArrowLeft, ChevronsUp, ChevronsDown, Lightbulb, X, Check } from 'lucide-react';
import { useGameStore } from '../../stores/game.ts';
import { getTaskById, getTasksForLevel } from '../../game/tasks/index.ts';
import { MODE_LABELS, DIFFICULTY_LABELS } from '../../game/constants.ts';
import { CriteriaChecklist } from './CriteriaChecklist.tsx';
import { GlossaryText } from './GlossaryText.tsx';
import { Tooltip } from '../ui/Tooltip.tsx';
import { ConfirmResetButton } from '../ui/ConfirmResetButton.tsx';

export function TaskHUD() {
  const activeTaskId = useGameStore(s => s.activeTaskId);
  const activeMode = useGameStore(s => s.activeMode);
  const activeDifficulty = useGameStore(s => s.activeDifficulty);
  const hintsRevealed = useGameStore(s => s.hintsRevealed);
  const attempts = useGameStore(s => s.attempts);
  const lastValidation = useGameStore(s => s.lastValidation);
  const approachValid = useGameStore(s => s.approachValid);
  const revealNextHint = useGameStore(s => s.revealNextHint);
  const openMenu = useGameStore(s => s.openMenu);
  const resetTask = useGameStore(s => s.resetTask);
  const exit = useGameStore(s => s.exit);
  const acknowledgeSuccess = useGameStore(s => s.acknowledgeSuccess);
  const reviewSuccess = useGameStore(s => s.reviewSuccess);

  const [hudExpanded, setHudExpanded] = useState(true);
  const [activeHintIndex, setActiveHintIndex] = useState(0);

  // Reset to expanded when task changes
  useEffect(() => {
    setHudExpanded(true);
    setActiveHintIndex(0);
  }, [activeTaskId]);

  // Auto-advance to newest hint when revealed
  useEffect(() => {
    if (hintsRevealed > 0) setActiveHintIndex(hintsRevealed - 1);
  }, [hintsRevealed]);

  if (!activeTaskId || !activeMode || !activeDifficulty) return null;
  const task = getTaskById(activeTaskId);
  if (!task) return null;

  const tasks = getTasksForLevel(activeMode, activeDifficulty);
  const taskIndex = tasks.findIndex(t => t.id === activeTaskId);
  const taskNumber = taskIndex + 1;
  const totalTasks = tasks.length;

  const isLast = taskIndex >= tasks.length - 1;
  const totalHints = task.hints.length;
  const hasMoreHints = hintsRevealed < totalHints;

  return (
    <div className={`bg-gray-900/50 border border-gray-800 rounded-xl mt-0 mb-4 backdrop-blur-sm${!hudExpanded ? ' border-l-2 border-l-accent' : ''}`}>
      {/* Top bar — always visible */}
      <div className="px-5 py-3 flex items-center gap-3">
        {/* Back to task list */}
        <Tooltip text="Back to task list">
          <button
            onClick={() => openMenu('tasks')}
            className="text-gray-400 hover:text-gray-300 cursor-pointer p-1 flex-shrink-0"
          >
            <ArrowLeft className="w-4 h-4" />
          </button>
        </Tooltip>

        {/* Breadcrumb — clickable segments */}
        <div className="text-sm text-gray-400 truncate flex items-center gap-0">
          <button
            onClick={() => openMenu('levels')}
            className="hover:text-white cursor-pointer transition-colors"
          >
            {MODE_LABELS[activeMode]}
          </button>
          <span className="text-gray-500 mx-1.5">&rsaquo;</span>
          <button
            onClick={() => openMenu('tasks')}
            className="hover:text-white cursor-pointer transition-colors"
          >
            {DIFFICULTY_LABELS[activeDifficulty]}
          </button>
          <span className="text-gray-500 mx-1.5">&rsaquo;</span>
          <span className="text-gray-300">Task {taskNumber} of {totalTasks}</span>
        </div>

        {/* Concept tag (collapsed only) */}
        {!hudExpanded && (
          <span className="text-xs px-1.5 py-0.5 rounded-full bg-accent/20 text-accent flex-shrink-0">
            {task.concept}
          </span>
        )}

        {/* Passed badge (collapsed only) */}
        {!hudExpanded && lastValidation?.passed && (
          <span className="flex items-center gap-1 text-xs text-green-400 flex-shrink-0">
            <Check className="w-3.5 h-3.5" />
          </span>
        )}

        {/* Mini progress bar (collapsed only) */}
        {!hudExpanded && (
          <div className="flex items-center gap-1.5 flex-shrink-0">
            <div className="w-16 h-1.5 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full bg-accent transition-all"
                style={{ width: `${totalTasks > 0 ? (taskNumber / totalTasks) * 100 : 0}%` }}
              />
            </div>
            <span className={`text-xs ${taskNumber > 0 ? 'text-accent' : 'text-gray-500'}`}>{taskNumber}/{totalTasks}</span>
          </div>
        )}

        {/* Spacer */}
        <div className="flex-1" />

        {/* Reset to task defaults */}
        <ConfirmResetButton onConfirm={resetTask} tooltip="Reset to task defaults" />

        {/* Expand/collapse toggle */}
        <Tooltip text={hudExpanded ? 'Collapse' : 'Expand'}>
          <button
            onClick={() => setHudExpanded(!hudExpanded)}
            className="text-gray-400 hover:text-gray-300 cursor-pointer p-1 flex-shrink-0"
          >
            {hudExpanded
              ? <ChevronsUp className="w-4 h-4" />
              : <ChevronsDown className="w-4 h-4" />}
          </button>
        </Tooltip>

        {/* Exit learning mode */}
        <Tooltip text="Exit learning mode">
          <button
            onClick={exit}
            className="text-gray-400 hover:text-gray-300 cursor-pointer p-1 flex-shrink-0"
          >
            <X className="w-4 h-4" />
          </button>
        </Tooltip>
      </div>

      {/* Expanded content */}
      <div
        className="transition-all duration-200 ease-in-out overflow-hidden"
        style={{ maxHeight: hudExpanded ? '600px' : '0px', opacity: hudExpanded ? 1 : 0 }}
      >
        <div className="px-5 pb-4">
          {/* Divider */}
          <div className="border-t border-gray-800 mb-3" />

          {/* Title + concept + attempts + restart */}
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-3">
              <h3 className="text-base font-semibold text-white">{task.title}</h3>
              <span className="text-sm px-2 py-0.5 rounded-full bg-accent/20 text-accent">
                {task.concept}
              </span>
            </div>
            <div className="flex items-center gap-3">
              {attempts > 0 && (
                <span className="text-sm text-gray-500">
                  {attempts} {attempts === 1 ? 'attempt' : 'attempts'}
                </span>
              )}

            </div>
          </div>

          {/* Briefing — always shown, split into paragraphs */}
          <div className="space-y-2 mb-3">
            {task.briefing.trim().split('\n\n').map((para, i) => (
              <p key={i} className="text-sm text-gray-300 leading-relaxed">
                <GlossaryText text={para.trim()} />
              </p>
            ))}
          </div>

          {/* Divider */}
          <div className="border-t border-gray-800 mb-3" />

          {/* Objectives + Hints side by side */}
          <div className="flex flex-col sm:flex-row gap-4">
            {/* Objectives */}
            <div className="flex-1">
              <div className="text-sm text-gray-400 mb-1.5">Objectives</div>
              <CriteriaChecklist criteria={task.winningCriteria} validation={lastValidation} />
              {lastValidation?.passed && !approachValid && (
                <div className="text-sm text-amber-200/90 bg-amber-900/20 border border-amber-700/30 rounded px-2.5 py-1.5 mt-2">
                  You hit the targets, but not by using the technique this task teaches. Re-read the briefing and try a different approach.
                </div>
              )}
              {lastValidation?.passed && approachValid && (
                <div className="flex items-center gap-2 mt-3 -ml-1">
                  <button
                    onClick={reviewSuccess}
                    className="text-sm text-gray-400 hover:text-white cursor-pointer transition-colors px-1 py-1 rounded"
                  >
                    Review
                  </button>
                  <button
                    onClick={acknowledgeSuccess}
                    className="text-sm px-3 py-1 bg-accent hover:bg-accent/80 text-white rounded cursor-pointer transition-colors"
                  >
                    {isLast ? 'Complete Level' : 'Next Task →'}
                  </button>
                </div>
              )}
            </div>

            {/* Hints */}
            <div className="flex-1">
              {/* Header row: dots + show-hint button */}
              <div className="flex items-center gap-1.5 mb-1.5">
                <span className="text-sm text-gray-400">Hints</span>
                {hintsRevealed > 1 && task.hints.slice(0, hintsRevealed).map((_, i) => (
                  <button
                    key={i}
                    onClick={() => setActiveHintIndex(i)}
                    className={`w-5 h-5 rounded-full text-xs cursor-pointer transition-colors ${
                      i === activeHintIndex
                        ? 'bg-yellow-400/30 text-yellow-300'
                        : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
                    }`}
                  >
                    {i + 1}
                  </button>
                ))}
                {hasMoreHints && (
                  <button
                    onClick={revealNextHint}
                    className="text-sm text-yellow-400/70 hover:text-yellow-400 cursor-pointer flex items-center gap-1 ml-auto"
                  >
                    <Lightbulb className="w-3.5 h-3.5" />
                    Show next hint
                  </button>
                )}
              </div>

              {/* Active hint card */}
              {hintsRevealed > 0 && (
                <div className="text-sm text-yellow-200/80 bg-yellow-900/20 border border-yellow-800/30 rounded px-2.5 py-1.5">
                  <Lightbulb className="w-3.5 h-3.5 inline mr-1 text-yellow-400" />
                  <GlossaryText text={task.hints[activeHintIndex].trim()} />
                </div>
              )}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
