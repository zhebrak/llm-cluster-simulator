/**
 * Game mode store — manages Learn Distributed Training & Inference state
 *
 * UI derivation from state (no phase enum):
 *   !active                          → game invisible
 *   active && !activeTaskId          → GameMenu
 *   active && activeTaskId && passed → SuccessModal
 *   active && activeTaskId           → TaskHUD
 *
 * CONVENTION: All state mutations that need persistence MUST go through store actions.
 * Components must NOT call useGameStore.setState() directly — use named actions instead.
 */

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import type { GameMode, GameDifficulty, ValidationResult } from '../game/types.ts';
import { getTaskById, getTasksForLevel, isTaskUnlocked } from '../game/tasks/index.ts';
import { buildValidationContext, validateTask, captureTaskConfig, validateExpectedChanges } from '../game/validation.ts';
import type { TaskConfigSnapshot } from '../game/validation.ts';
import { GAME_STORAGE_KEY, PRE_GAME_CONFIG_KEY } from '../game/constants.ts';
import { applySetupToConfig } from '../game/setup.ts';
import { snapshotConfig, loadPersistedState, recoverOrphanedConfig } from './persistence.ts';
import { useConfigStore } from './config.ts';
import { useSimulationStore } from './simulation.ts';

// Lazy reference to RPG store to avoid circular imports (rpg.ts imports game.ts)
let _rpgStoreRef: { getState: () => { active: boolean; exit: () => void } } | null = null;

/** Called by rpg.ts after it creates its store, to register for mutual exclusion. */
export function _registerRPGStore(store: typeof _rpgStoreRef): void {
  _rpgStoreRef = store;
}

interface LastTaskState {
  mode: GameMode;
  difficulty: GameDifficulty;
  taskId: string;       // empty string = was on task list, not a specific task
  hintsRevealed: number;
  attempts: number;
}

interface PreviousTaskState {
  taskId: string;
  hintsRevealed: number;
  attempts: number;
  lastValidation: ValidationResult | null;
}

interface SavedTaskState {
  hintsRevealed: number;
  attempts: number;
  lastValidation: ValidationResult | null;
}

interface GameState {
  // Core
  active: boolean;
  activeMode: GameMode | null;
  activeDifficulty: GameDifficulty | null;
  activeTaskId: string | null;

  // Task state
  hintsRevealed: number;
  attempts: number;
  lastValidation: ValidationResult | null;

  // Progress
  progress: Record<string, string[]>;  // 'training-beginner' → completed task IDs
  taskStates: Record<string, SavedTaskState>;  // per-task saved state for revisits

  // Config snapshot
  savedConfigSnapshot: string | null;  // JSON string of config state

  // Task config snapshot for expected-change validation
  taskConfigSnapshot: TaskConfigSnapshot | null;
  approachValid: boolean;  // true when no expectedChanges or all checks pass

  // Debounce
  lastValidatedRunCounter: number;

  // Resume support
  lastTaskState: LastTaskState | null;        // persisted — resume position across exit/enter
  previousTaskState: PreviousTaskState | null; // transient — backdrop click returns to task

  // Level completion celebration (transient — not persisted)
  showLevelComplete: boolean;

  // Success dismissed — player chose "Explore Results" (transient — not persisted)
  successDismissed: boolean;

  // Menu overlay (transient — not persisted)
  menuOpen: false | 'tasks' | 'levels';

  // Actions
  dismissLevelComplete: () => void;
  dismissSuccess: () => void;
  reviewSuccess: () => void;
  openMenu: (view: 'tasks' | 'levels') => void;
  closeMenu: () => void;
  enter: () => void;
  exit: () => void;
  exitTask: () => void;
  selectLevel: (mode: GameMode, difficulty: GameDifficulty) => void;
  clearLevel: () => void;
  startTask: (taskId: string) => void;
  resetTask: () => void;
  revealNextHint: () => void;
  acknowledgeSuccess: () => void;
  dismissFailure: () => void;
  resetLevel: (mode: GameMode, difficulty: GameDifficulty) => void;
  resumePreviousTask: () => void;
  resetToLevelPicker: () => void;
}

function levelKey(mode: GameMode, difficulty: GameDifficulty): string {
  return `${mode}-${difficulty}`;
}

/**
 * Save current task state into taskStates dict (called inside immer set() callbacks).
 */
function saveCurrentTaskState(state: GameState): void {
  if (!state.activeTaskId) return;
  state.taskStates[state.activeTaskId] = {
    hintsRevealed: state.hintsRevealed,
    attempts: state.attempts,
    lastValidation: state.lastValidation,
  };
}

/** Load task fields — restore from saved state if completed, otherwise fresh defaults */
function loadTaskFields(state: GameState, taskId: string, levelProgress: string[]): void {
  const saved = levelProgress.includes(taskId) ? state.taskStates[taskId] : undefined;
  if (saved) {
    state.hintsRevealed = saved.hintsRevealed;
    state.attempts = saved.attempts;
    state.lastValidation = saved.lastValidation;
    state.successDismissed = true;
  } else {
    state.hintsRevealed = 0;
    state.attempts = 0;
    state.lastValidation = null;
    state.successDismissed = false;
  }
  state.lastValidatedRunCounter = 0;
}

/**
 * Persist game state to localStorage (excludes lastValidation).
 */
function persistGameState(state: GameState): void {
  try {
    const toSave = {
      active: state.active,
      activeMode: state.activeMode,
      activeDifficulty: state.activeDifficulty,
      activeTaskId: state.activeTaskId,
      hintsRevealed: state.hintsRevealed,
      attempts: state.attempts,
      progress: state.progress,
      taskStates: state.taskStates,
      savedConfigSnapshot: state.savedConfigSnapshot,
      taskConfigSnapshot: state.taskConfigSnapshot,
      lastValidatedRunCounter: state.lastValidatedRunCounter,
      lastTaskState: state.lastTaskState,
    };
    localStorage.setItem(GAME_STORAGE_KEY, JSON.stringify(toSave));
  } catch { /* localStorage full or unavailable */ }
}


/**
 * Apply task setup to config store using shared helpers, then capture
 * config snapshot for expected-change validation.
 */
function applyTaskSetup(taskId: string): void {
  const task = getTaskById(taskId);
  if (!task) return;

  applySetupToConfig(task.setup, task.mode);

  // Capture config snapshot for expected-change validation
  const snapshot = captureTaskConfig();
  useGameStore.setState(state => {
    state.taskConfigSnapshot = snapshot;
    state.approachValid = true;
  });
}

// Load initial state
const persisted = loadPersistedState<GameState>(GAME_STORAGE_KEY);

// Crash recovery: if pre-game config exists but game is not active, restore
recoverOrphanedConfig(PRE_GAME_CONFIG_KEY, persisted.active ?? false);

export const useGameStore = create<GameState>()(
  immer((set, get) => ({
    // Initialize from persisted state
    active: persisted.active ?? false,
    activeMode: (persisted.activeMode as GameMode) ?? null,
    activeDifficulty: (persisted.activeDifficulty as GameDifficulty) ?? null,
    activeTaskId: persisted.activeTaskId ?? null,
    hintsRevealed: persisted.hintsRevealed ?? 0,
    attempts: persisted.attempts ?? 0,
    lastValidation: null,  // always null on load — user re-runs to validate
    progress: persisted.progress ?? {},
    taskStates: (persisted as Record<string, unknown>).taskStates as Record<string, SavedTaskState> ?? {},
    savedConfigSnapshot: persisted.savedConfigSnapshot ?? null,
    taskConfigSnapshot: (persisted as Record<string, unknown>).taskConfigSnapshot as TaskConfigSnapshot | null ?? null,
    approachValid: true,
    lastValidatedRunCounter: persisted.lastValidatedRunCounter ?? 0,
    lastTaskState: (persisted as Record<string, unknown>).lastTaskState as LastTaskState | null ?? null,
    previousTaskState: null,  // transient — never persisted
    showLevelComplete: false,  // transient — never persisted
    successDismissed: false,  // transient — never persisted
    menuOpen: false,  // transient — never persisted

    dismissLevelComplete: () => {
      set(state => { state.showLevelComplete = false; });
    },

    dismissSuccess: () => {
      set(state => { state.successDismissed = true; });
    },

    reviewSuccess: () => {
      set(state => { state.successDismissed = false; });
    },

    openMenu: (view) => {
      set(state => { state.menuOpen = view; });
    },
    closeMenu: () => {
      set(state => { state.menuOpen = false; });
    },

    enter: () => {
      // Mutual exclusion: soft-exit RPG mode if active
      if (_rpgStoreRef?.getState().active) {
        _rpgStoreRef.getState().exit();
      }

      const snapshot = snapshotConfig();
      // Also write to crash-recovery key
      if (snapshot) {
        try {
          localStorage.setItem(PRE_GAME_CONFIG_KEY, snapshot);
        } catch { /* ignore */ }
      }

      const { lastTaskState } = get();

      set(state => {
        state.active = true;
        state.savedConfigSnapshot = snapshot;

        // Resume previous position if available
        if (lastTaskState) {
          state.activeMode = lastTaskState.mode;
          state.activeDifficulty = lastTaskState.difficulty;
          if (lastTaskState.taskId) {
            state.activeTaskId = lastTaskState.taskId;
            // Prefer richer savedTaskState (includes lastValidation) over lastTaskState
            const saved = state.taskStates[lastTaskState.taskId];
            if (saved) {
              state.hintsRevealed = saved.hintsRevealed;
              state.attempts = saved.attempts;
              state.lastValidation = saved.lastValidation;
              state.successDismissed = true;
            } else {
              state.hintsRevealed = lastTaskState.hintsRevealed;
              state.attempts = lastTaskState.attempts;
            }
          }
        }
      });

      // Re-apply task config if resuming into a specific task
      if (lastTaskState?.taskId) {
        applyTaskSetup(lastTaskState.taskId);
      }

      persistGameState(get());
    },

    exit: () => {
      const current = get();
      const { savedConfigSnapshot } = current;

      // Capture current position before clearing
      const lastTask: LastTaskState | null = current.activeTaskId
        ? { mode: current.activeMode!, difficulty: current.activeDifficulty!, taskId: current.activeTaskId, hintsRevealed: current.hintsRevealed, attempts: current.attempts }
        : current.activeMode && current.activeDifficulty
          ? { mode: current.activeMode, difficulty: current.activeDifficulty, taskId: '', hintsRevealed: 0, attempts: 0 }
          : null;

      // Race guard: deactivate FIRST so simulation subscriber ignores late-firing runs
      set(state => {
        saveCurrentTaskState(state);
        state.active = false;
        state.menuOpen = false;
        state.showLevelComplete = false;
        state.activeMode = null;
        state.activeDifficulty = null;
        state.activeTaskId = null;
        state.hintsRevealed = 0;
        state.attempts = 0;
        state.lastValidation = null;
        state.savedConfigSnapshot = null;
        state.taskConfigSnapshot = null;
        state.approachValid = true;
        state.lastValidatedRunCounter = 0;
        state.lastTaskState = lastTask;
        state.previousTaskState = null;
      });

      // Clean up crash-recovery key
      try {
        localStorage.removeItem(PRE_GAME_CONFIG_KEY);
      } catch { /* ignore */ }

      // Restore original config via config store (no page reload)
      if (savedConfigSnapshot) {
        useConfigStore.getState().restoreFromSnapshot(savedConfigSnapshot);
      }

      persistGameState(get());
    },

    exitTask: () => {
      const current = get();
      set(state => {
        saveCurrentTaskState(state);
        // Save task state for backdrop-click resume
        if (current.activeTaskId) {
          state.previousTaskState = {
            taskId: current.activeTaskId,
            hintsRevealed: current.hintsRevealed,
            attempts: current.attempts,
            lastValidation: current.lastValidation,
          };
        }
        state.activeTaskId = null;
        state.lastValidation = null;
        state.hintsRevealed = 0;
        state.attempts = 0;
      });
      persistGameState(get());
    },

    selectLevel: (mode: GameMode, difficulty: GameDifficulty) => {
      set(state => {
        state.activeMode = mode;
        state.activeDifficulty = difficulty;
        state.activeTaskId = null;
        state.lastValidation = null;
        state.showLevelComplete = false;
        if (state.menuOpen === 'levels') {
          state.menuOpen = 'tasks';
        }
      });
      persistGameState(get());
    },

    clearLevel: () => {
      set(state => {
        state.activeMode = null;
        state.activeDifficulty = null;
        state.activeTaskId = null;
        state.lastValidation = null;
        state.showLevelComplete = false;
      });
      persistGameState(get());
    },

    startTask: (taskId: string) => {
      const task = getTaskById(taskId);
      if (!task) return;

      // Sequential locking guard
      const { progress } = get();
      const key = levelKey(task.mode, task.difficulty);
      if (!isTaskUnlocked(task, progress[key] ?? [])) return;

      set(state => {
        saveCurrentTaskState(state);
        state.menuOpen = false;
        state.activeTaskId = taskId;
        state.previousTaskState = null;  // chose a different task
        loadTaskFields(state, taskId, progress[key] ?? []);
      });

      // Apply task setup to config
      applyTaskSetup(taskId);

      // Reset simulation so criteria show as gray (unevaluated)
      useSimulationStore.getState().reset();

      persistGameState(get());
    },

    resetTask: () => {
      const { activeTaskId } = get();
      if (!activeTaskId) return;

      // Re-apply task setup (resets config to task defaults)
      applyTaskSetup(activeTaskId);

      // Clear validation and saved state (explicit reset starts fresh)
      set(state => {
        state.lastValidation = null;
        state.lastValidatedRunCounter = 0;
        state.successDismissed = false;
        delete state.taskStates[activeTaskId];
      });

      // Reset simulation so criteria show as gray (unevaluated)
      useSimulationStore.getState().reset();
    },

    revealNextHint: () => {
      set(state => {
        const task = state.activeTaskId ? getTaskById(state.activeTaskId) : null;
        if (task && state.hintsRevealed < task.hints.length) {
          state.hintsRevealed += 1;
        }
        saveCurrentTaskState(state);
      });
      persistGameState(get());
    },

    acknowledgeSuccess: () => {
      const { activeMode, activeDifficulty, activeTaskId, lastValidation } = get();
      if (!lastValidation?.passed) return;
      if (!activeMode || !activeDifficulty || !activeTaskId) return;

      const key = levelKey(activeMode, activeDifficulty);
      const tasks = getTasksForLevel(activeMode, activeDifficulty);
      const currentIndex = tasks.findIndex(t => t.id === activeTaskId);
      const nextTask = currentIndex >= 0 && currentIndex < tasks.length - 1
        ? tasks[currentIndex + 1]
        : null;

      set(state => {
        // Save completed task state before advancing
        saveCurrentTaskState(state);

        // Record progress
        if (!state.progress[key]) {
          state.progress[key] = [];
        }
        if (!state.progress[key].includes(activeTaskId)) {
          state.progress[key].push(activeTaskId);
        }

        if (nextTask) {
          // Advance to next task
          state.activeTaskId = nextTask.id;
          loadTaskFields(state, nextTask.id, state.progress[key] ?? []);
        } else {
          // Level complete — show celebration
          state.activeTaskId = null;
          state.lastValidation = null;
          state.showLevelComplete = true;
        }
      });

      // Apply next task setup if advancing
      if (nextTask) {
        applyTaskSetup(nextTask.id);
        useSimulationStore.getState().reset();
      }

      persistGameState(get());
    },

    dismissFailure: () => {
      set(state => {
        state.lastValidation = null;
      });
      // No persist needed — lastValidation is transient
    },

    resetLevel: (mode: GameMode, difficulty: GameDifficulty) => {
      const key = levelKey(mode, difficulty);
      const tasks = getTasksForLevel(mode, difficulty);
      set(state => {
        state.progress[key] = [];
        state.showLevelComplete = false;
        // Clear saved task states for this level
        for (const task of tasks) {
          delete state.taskStates[task.id];
        }
      });
      persistGameState(get());

      // Jump to the first task of the level
      if (tasks.length > 0) {
        // progress is already cleared, so startTask's unlock guard will pass
        get().startTask(tasks[0].id);
      }
    },

    resumePreviousTask: () => {
      const { previousTaskState } = get();
      if (!previousTaskState) return;

      set(state => {
        state.activeTaskId = previousTaskState.taskId;
        state.hintsRevealed = previousTaskState.hintsRevealed;
        state.attempts = previousTaskState.attempts;
        state.lastValidation = previousTaskState.lastValidation;
        state.previousTaskState = null;
      });

      // Re-apply task config
      applyTaskSetup(previousTaskState.taskId);

      persistGameState(get());
    },

    resetToLevelPicker: () => {
      set(state => {
        saveCurrentTaskState(state);
        state.activeMode = null;
        state.activeDifficulty = null;
        state.activeTaskId = null;
        state.lastTaskState = null;
        state.menuOpen = false;
        state.showLevelComplete = false;
        state.previousTaskState = null;
        state.lastValidation = null;
        state.hintsRevealed = 0;
        state.attempts = 0;
        state.successDismissed = false;
      });
      persistGameState(get());
    },
  })),
);

/**
 * Simulation subscriber — validates criteria after each completed simulation run.
 */
useSimulationStore.subscribe((simState) => {
  const game = useGameStore.getState();
  if (!game.active || !game.activeTaskId) return;
  if (simState.status !== 'complete') return;
  if (simState.runCounter <= game.lastValidatedRunCounter) return;

  const task = getTaskById(game.activeTaskId);
  if (!task) return;

  const ctx = buildValidationContext(simState, game.activeMode);
  const result = validateTask(ctx, task.winningCriteria);

  // Validate expected changes (approach validation)
  let approachValid = true;
  if (task.expectedChanges && game.taskConfigSnapshot) {
    const currentConfig = captureTaskConfig();
    const { valid } = validateExpectedChanges(game.taskConfigSnapshot, currentConfig, task.expectedChanges);
    approachValid = valid;
  }

  useGameStore.setState(state => {
    state.lastValidation = result;
    state.approachValid = approachValid;
    state.lastValidatedRunCounter = simState.runCounter;
    state.attempts = state.attempts + 1;
    saveCurrentTaskState(state);
  });

  persistGameState(useGameStore.getState());
});
