/**
 * Game menu — level selection grid with progress tracking
 */

import { X, Trophy, ChevronRight, Lock } from 'lucide-react';
import { useGameStore } from '../../stores/game.ts';
import { getTasksForLevel, isTaskUnlocked } from '../../game/tasks/index.ts';
import { MODE_LABELS, DIFFICULTY_LABELS } from '../../game/constants.ts';
import type { GameMode, GameDifficulty } from '../../game/types.ts';

const MODES: GameMode[] = ['training', 'inference'];
const DIFFICULTIES: GameDifficulty[] = ['beginner', 'intermediate', 'advanced'];

const DIFFICULTY_DESCRIPTIONS: Record<GameMode, Record<GameDifficulty, string>> = {
  training: {
    beginner: 'Memory, precision, and scaling to multi-node clusters',
    intermediate: 'Tensor, pipeline, and sequence parallelism in depth',
    advanced: 'Real-world configs, fine-tuning, and full-scale optimization',
  },
  inference: {
    beginner: 'Weight memory, KV cache, and latency-throughput tradeoffs',
    intermediate: 'Multi-GPU serving, GPU selection, and speculative decoding',
    advanced: 'Production MoE serving, latency SLAs, and cost optimization',
  },
};

const DIFFICULTY_COLORS: Record<GameDifficulty, string> = {
  beginner: 'text-green-400 border-green-400/30',
  intermediate: 'text-yellow-400 border-yellow-400/30',
  advanced: 'text-red-400 border-red-400/30',
};

function LevelCard({ mode, difficulty }: { mode: GameMode; difficulty: GameDifficulty }) {
  const progress = useGameStore(s => s.progress);
  const selectLevel = useGameStore(s => s.selectLevel);
  const startTask = useGameStore(s => s.startTask);

  const key = `${mode}-${difficulty}`;
  const tasks = getTasksForLevel(mode, difficulty);
  const completed = progress[key] || [];
  const completedCount = completed.length;
  const total = tasks.length;
  const isComplete = completedCount >= total;

  // Find first incomplete task
  const firstIncomplete = tasks.find(t => !completed.includes(t.id));

  const handleClick = () => {
    selectLevel(mode, difficulty);
    if (firstIncomplete) {
      startTask(firstIncomplete.id);
    }
  };

  return (
    <button
      onClick={handleClick}
      className={`text-left p-2.5 sm:p-4 rounded-lg border bg-gray-800/50 hover:bg-gray-700/50 transition-colors cursor-pointer ${
        isComplete ? 'border-green-500/40' : DIFFICULTY_COLORS[difficulty].split(' ')[1]
      }`}
    >
      <div className="flex items-center justify-between mb-2">
        <span className={`text-xs sm:text-sm font-medium ${DIFFICULTY_COLORS[difficulty].split(' ')[0]}`}>
          {DIFFICULTY_LABELS[difficulty]}
        </span>
        {isComplete ? (
          <Trophy className="w-4 h-4 text-green-400" />
        ) : (
          <ChevronRight className="w-4 h-4 text-gray-500" />
        )}
      </div>
      <div className="text-xs text-gray-400 mb-3 hidden sm:block">
        {DIFFICULTY_DESCRIPTIONS[mode][difficulty]}
      </div>
      <div className="flex items-center gap-2 mt-2 sm:mt-0">
        <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${
              isComplete ? 'bg-green-400' : 'bg-accent'
            }`}
            style={{ width: `${total > 0 ? (completedCount / total) * 100 : 0}%` }}
          />
        </div>
        <span className="text-xs text-gray-500">{completedCount}/{total}</span>
      </div>
    </button>
  );
}

function LevelTaskList({ mode, difficulty }: { mode: GameMode; difficulty: GameDifficulty }) {
  const progress = useGameStore(s => s.progress);
  const startTask = useGameStore(s => s.startTask);
  const resetLevel = useGameStore(s => s.resetLevel);

  const key = `${mode}-${difficulty}`;
  const tasks = getTasksForLevel(mode, difficulty);
  const completed = progress[key] || [];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white">
          {MODE_LABELS[mode]} — {DIFFICULTY_LABELS[difficulty]}
        </h3>
        {completed.length > 0 && (
          <button
            onClick={() => resetLevel(mode, difficulty)}
            className="text-xs text-gray-500 hover:text-gray-300 cursor-pointer"
          >
            Reset Progress
          </button>
        )}
      </div>
      <div className="space-y-1">
        {tasks.map((task) => {
          const isCompleted = completed.includes(task.id);
          const isLocked = !isTaskUnlocked(task, completed);

          if (isLocked) {
            return (
              <div
                key={task.id}
                className="w-full text-left px-3 py-2 rounded-md flex items-center gap-3 opacity-40 cursor-not-allowed"
                title="Complete the previous task to unlock"
              >
                <span className="w-5 h-5 flex items-center justify-center rounded-full text-xs flex-shrink-0 bg-gray-700 text-gray-500">
                  <Lock className="w-3 h-3" />
                </span>
                <div className="flex-1 min-w-0">
                  <div className="text-sm text-gray-400 truncate">{task.title}</div>
                  <div className="text-xs text-gray-600 truncate">{task.concept}</div>
                </div>
              </div>
            );
          }

          return (
            <button
              key={task.id}
              onClick={() => startTask(task.id)}
              className="w-full text-left px-3 py-2 rounded-md hover:bg-gray-700/50 transition-colors flex items-center gap-3 cursor-pointer"
            >
              <span className={`w-5 h-5 flex items-center justify-center rounded-full text-xs flex-shrink-0 ${
                isCompleted
                  ? 'bg-green-400/20 text-green-400'
                  : 'bg-gray-700 text-gray-400'
              }`}>
                {isCompleted ? '✓' : task.order + 1}
              </span>
              <div className="flex-1 min-w-0">
                <div className="text-sm text-white truncate">{task.title}</div>
                <div className="text-xs text-gray-500 truncate">{task.concept}</div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function GameMenu() {
  const exit = useGameStore(s => s.exit);
  const clearLevel = useGameStore(s => s.clearLevel);
  const resumePreviousTask = useGameStore(s => s.resumePreviousTask);
  const previousTaskState = useGameStore(s => s.previousTaskState);
  const activeMode = useGameStore(s => s.activeMode);
  const activeDifficulty = useGameStore(s => s.activeDifficulty);
  const menuOpen = useGameStore(s => s.menuOpen);
  const closeMenu = useGameStore(s => s.closeMenu);
  const openMenu = useGameStore(s => s.openMenu);

  // menuOpen takes precedence (breadcrumb context); otherwise standalone context
  const showTaskList = menuOpen
    ? menuOpen === 'tasks'
    : activeMode !== null && activeDifficulty !== null;

  const handleBackdropClick = () => {
    if (menuOpen) closeMenu();
    else if (previousTaskState) resumePreviousTask();
    else if (showTaskList) clearLevel();
    else exit();
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={handleBackdropClick}>
      <div
        className="bg-gray-900 border border-gray-700 rounded-xl p-4 sm:p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4 sm:mb-6">
          <h2 className="text-lg sm:text-xl font-semibold text-white">
            {showTaskList ? 'Select a Level' : 'Learn Distributed Training & Inference'}
          </h2>
          <button onClick={menuOpen ? closeMenu : exit} className="text-gray-400 hover:text-gray-300 cursor-pointer">
            <X className="w-5 h-5" />
          </button>
        </div>

        {showTaskList && activeMode && activeDifficulty ? (
          <div>
            <button
              onClick={() => { if (menuOpen) openMenu('levels'); else clearLevel(); }}
              className="text-sm text-gray-400 hover:text-white mb-4 cursor-pointer"
            >
              ← To levels
            </button>
            <LevelTaskList mode={activeMode} difficulty={activeDifficulty} />
          </div>
        ) : (
          <div className="space-y-6">
            {MODES.map(mode => (
              <div key={mode}>
                <h3 className="text-sm font-medium text-gray-300 mb-3 uppercase tracking-wider">
                  {MODE_LABELS[mode]}
                </h3>
                <div className="grid grid-cols-3 gap-2 sm:gap-3">
                  {DIFFICULTIES.map(difficulty => (
                    <LevelCard key={`${mode}-${difficulty}`} mode={mode} difficulty={difficulty} />
                  ))}
                </div>
              </div>
            ))}
            <div className="pt-4 border-t border-gray-800 text-xs text-gray-500">
              Complete hands-on challenges to learn distributed ML concepts. Each task configures a scenario — adjust settings and run simulations to meet the objectives.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
