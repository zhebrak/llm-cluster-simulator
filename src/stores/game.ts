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
import type { GameMode, GameDifficulty, ValidationResult, TaskSetup } from '../game/types.ts';
import { getTaskById, getTasksForLevel, isTaskUnlocked } from '../game/tasks/index.ts';
import { buildValidationContext, validateTask, captureTaskConfig, validateExpectedChanges } from '../game/validation.ts';
import type { TaskConfigSnapshot } from '../game/validation.ts';
import { TRAINING_DEFAULTS, INFERENCE_DEFAULTS } from '../game/defaults.ts';
import { useConfigStore } from './config.ts';
import type { TrainingConfig } from './config.ts';
import { useSimulationStore } from './simulation.ts';

const GAME_STORAGE_KEY = 'llm-sim-game';
const PRE_GAME_CONFIG_KEY = 'llm-sim-config-pre-game';
const CONFIG_STORAGE_KEY = 'llm-sim-config';

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
 * Load persisted game state from localStorage.
 */
function loadPersistedGameState(): Partial<GameState> {
  try {
    const raw = localStorage.getItem(GAME_STORAGE_KEY);
    if (!raw) return {};
    return JSON.parse(raw);
  } catch {
    return {};
  }
}

/**
 * Snapshot the current config state for crash recovery.
 */
function snapshotConfig(): string | null {
  try {
    const raw = localStorage.getItem(CONFIG_STORAGE_KEY);
    return raw;
  } catch {
    return null;
  }
}

/**
 * Restore config from a snapshot string.
 */
function restoreConfigSnapshot(snapshot: string | null): void {
  if (!snapshot) return;
  try {
    localStorage.setItem(CONFIG_STORAGE_KEY, snapshot);
    // Reload the page to pick up restored config cleanly
    // Instead, we'll just set it and let the next page load handle it
  } catch { /* ignore */ }
}

const VALID_STRATEGY_TYPES: Set<string> = new Set([
  'ddp', 'fsdp', 'zero-1', 'zero-3', 'auto',
  'fsdp-tp', 'zero1-tp',
  'ddp-tp-pp', 'zero1-tp-pp', 'fsdp-tp-pp',
]);

/**
 * Reset config store to naive training defaults.
 * FP32, no AC, no FA — beginner tasks need zero overrides.
 */
function resetToTrainingDefaults(configStore: ReturnType<typeof useConfigStore.getState>): void {
  configStore.setPrecision(TRAINING_DEFAULTS.mixedPrecision);
  configStore.setSequenceLength(TRAINING_DEFAULTS.sequenceLength);
  configStore.setTrainingParams({
    activationCheckpointing: TRAINING_DEFAULTS.activationCheckpointing,
    checkpointingGranularity: TRAINING_DEFAULTS.checkpointingGranularity,
    flashAttention: TRAINING_DEFAULTS.flashAttention,
    globalBatchSize: TRAINING_DEFAULTS.globalBatchSize,
    microBatchSize: TRAINING_DEFAULTS.microBatchSize,
    sequenceParallel: TRAINING_DEFAULTS.sequenceParallel,
    finetuningMethod: TRAINING_DEFAULTS.finetuningMethod,
    loraRank: TRAINING_DEFAULTS.loraRank,
    loraTargetModules: TRAINING_DEFAULTS.loraTargetModules,
  });
  configStore.setStrategyParams({
    tpDegree: TRAINING_DEFAULTS.tpDegree,
    ppDegree: TRAINING_DEFAULTS.ppDegree,
    epDegree: TRAINING_DEFAULTS.epDegree,
    cpDegree: TRAINING_DEFAULTS.cpDegree,
    pipelineSchedule: TRAINING_DEFAULTS.pipelineSchedule,
    interleavedStages: TRAINING_DEFAULTS.interleavedStages,
  });
}

/**
 * Reset config store to reasonable inference defaults.
 * BF16, FA/PA on, no batching or advanced features.
 */
function resetToInferenceDefaults(configStore: ReturnType<typeof useConfigStore.getState>): void {
  configStore.setInferenceParams({
    batchSize: INFERENCE_DEFAULTS.batchSize,
    inputSeqLen: INFERENCE_DEFAULTS.inputSeqLen,
    outputSeqLen: INFERENCE_DEFAULTS.outputSeqLen,
    weightPrecision: INFERENCE_DEFAULTS.weightPrecision,
    kvCachePrecision: INFERENCE_DEFAULTS.kvCachePrecision,
    flashAttention: INFERENCE_DEFAULTS.flashAttention,
    pagedAttention: INFERENCE_DEFAULTS.pagedAttention,
    continuousBatching: INFERENCE_DEFAULTS.continuousBatching,
    tensorParallel: INFERENCE_DEFAULTS.tensorParallel,
    expertParallel: INFERENCE_DEFAULTS.expertParallel,
    speculativeDecoding: INFERENCE_DEFAULTS.speculativeDecoding,
  });
}

/**
 * Apply task-specific overrides from TaskSetup onto config store.
 * Only fields explicitly set in the setup (not undefined) are applied.
 */
function applyTaskOverrides(setup: TaskSetup, configStore: ReturnType<typeof useConfigStore.getState>, mode: 'training' | 'inference'): void {
  if (mode === 'training') {
    if (setup.mixedPrecision !== undefined) configStore.setPrecision(setup.mixedPrecision);
    if (setup.sequenceLength !== undefined) configStore.setSequenceLength(setup.sequenceLength);

    const trainingOverrides: Partial<TrainingConfig> = {};
    if (setup.activationCheckpointing !== undefined) trainingOverrides.activationCheckpointing = setup.activationCheckpointing;
    if (setup.checkpointingGranularity !== undefined) trainingOverrides.checkpointingGranularity = setup.checkpointingGranularity;
    if (setup.flashAttention !== undefined) trainingOverrides.flashAttention = setup.flashAttention;
    if (setup.globalBatchSize !== undefined) trainingOverrides.globalBatchSize = setup.globalBatchSize;
    if (setup.microBatchSize !== undefined) trainingOverrides.microBatchSize = setup.microBatchSize;
    if (setup.sequenceParallel !== undefined) trainingOverrides.sequenceParallel = setup.sequenceParallel;
    if (setup.finetuningMethod !== undefined) trainingOverrides.finetuningMethod = setup.finetuningMethod;
    if (setup.loraRank !== undefined) trainingOverrides.loraRank = setup.loraRank;
    if (setup.loraTargetModules !== undefined) trainingOverrides.loraTargetModules = setup.loraTargetModules;
    if (Object.keys(trainingOverrides).length > 0) configStore.setTrainingParams(trainingOverrides);

    const strategyOverrides: Partial<TrainingConfig> = {};
    if (setup.tpDegree !== undefined) strategyOverrides.tpDegree = setup.tpDegree;
    if (setup.ppDegree !== undefined) strategyOverrides.ppDegree = setup.ppDegree;
    if (setup.epDegree !== undefined) strategyOverrides.epDegree = setup.epDegree;
    if (setup.cpDegree !== undefined) strategyOverrides.cpDegree = setup.cpDegree;
    if (setup.pipelineSchedule !== undefined) strategyOverrides.pipelineSchedule = setup.pipelineSchedule;
    if (setup.interleavedStages !== undefined) strategyOverrides.interleavedStages = setup.interleavedStages;
    if (Object.keys(strategyOverrides).length > 0) configStore.setStrategyParams(strategyOverrides);
  } else {
    const inferenceOverrides: Record<string, unknown> = {};
    if (setup.weightPrecision !== undefined) inferenceOverrides.weightPrecision = setup.weightPrecision;
    if (setup.kvCachePrecision !== undefined) inferenceOverrides.kvCachePrecision = setup.kvCachePrecision;
    if (setup.batchSize !== undefined) inferenceOverrides.batchSize = setup.batchSize;
    if (setup.inputSeqLen !== undefined) inferenceOverrides.inputSeqLen = setup.inputSeqLen;
    if (setup.outputSeqLen !== undefined) inferenceOverrides.outputSeqLen = setup.outputSeqLen;
    if (setup.tensorParallel !== undefined) inferenceOverrides.tensorParallel = setup.tensorParallel;
    if (setup.expertParallel !== undefined) inferenceOverrides.expertParallel = setup.expertParallel;
    if (setup.pagedAttention !== undefined) inferenceOverrides.pagedAttention = setup.pagedAttention;
    if (setup.continuousBatching !== undefined) inferenceOverrides.continuousBatching = setup.continuousBatching;
    // Note: TaskSetup uses 'speculativeDecoding', simulation config uses 'speculativeEnabled'.
    // The store handles the mapping at simulation time.
    if (setup.speculativeDecoding !== undefined) inferenceOverrides.speculativeDecoding = setup.speculativeDecoding;
    if (setup.numSpeculativeTokens !== undefined) inferenceOverrides.numSpeculativeTokens = setup.numSpeculativeTokens;
    if (setup.acceptanceRate !== undefined) inferenceOverrides.acceptanceRate = setup.acceptanceRate;
    if (setup.flashAttention !== undefined) inferenceOverrides.flashAttention = setup.flashAttention;
    if (Object.keys(inferenceOverrides).length > 0) configStore.setInferenceParams(inferenceOverrides);

    // Draft model requires separate store action (resolves ModelSpec from ID)
    if (setup.draftModelId !== undefined) {
      configStore.setDraftModel(setup.draftModelId);
    }
  }
}

/**
 * Apply task setup to config store: reset to deterministic base defaults,
 * then apply task-specific overrides. This ensures every task starts from
 * a known state regardless of user's prior config.
 */
function applyTaskSetup(taskId: string): void {
  const task = getTaskById(taskId);
  if (!task) return;

  const configStore = useConfigStore.getState();
  const { setup } = task;

  // Step 1: Switch mode if needed
  if (configStore.mode !== task.mode) {
    configStore.setMode(task.mode);
  }

  // Step 2: Set model and cluster
  configStore.setModel(setup.modelId);
  const numGPUs = setup.numGPUs ?? 1;
  const gpusPerNode = setup.gpusPerNode ?? Math.min(numGPUs, 8);
  configStore.setCustomCluster(setup.gpuId, numGPUs, gpusPerNode);
  configStore.setPricePerGPUHour(null);

  // Step 3: Set strategy (training only), then reset to base defaults
  if (task.mode === 'training') {
    if (setup.strategyType && VALID_STRATEGY_TYPES.has(setup.strategyType)) {
      configStore.setStrategy(setup.strategyType as TrainingConfig['strategyType']);
    }
    resetToTrainingDefaults(configStore);
  } else {
    resetToInferenceDefaults(configStore);
  }

  // Step 4: Apply task-specific overrides
  applyTaskOverrides(setup, configStore, task.mode);

  // Step 5: Capture config snapshot for expected-change validation
  const snapshot = captureTaskConfig();
  useGameStore.setState(state => {
    state.taskConfigSnapshot = snapshot;
    state.approachValid = true;
  });
}

// Load initial state
const persisted = loadPersistedGameState();

// Crash recovery: if pre-game config exists but game is not active, restore
try {
  const preGameConfig = localStorage.getItem(PRE_GAME_CONFIG_KEY);
  if (preGameConfig && !persisted.active) {
    restoreConfigSnapshot(preGameConfig);
    localStorage.removeItem(PRE_GAME_CONFIG_KEY);
  }
} catch { /* ignore */ }

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

      // Restore original config
      restoreConfigSnapshot(savedConfigSnapshot);
      // Clean up crash-recovery key
      try {
        localStorage.removeItem(PRE_GAME_CONFIG_KEY);
      } catch { /* ignore */ }
      set(state => {
        saveCurrentTaskState(state);
        state.menuOpen = false;
        state.showLevelComplete = false;
        state.active = false;
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
      persistGameState(get());
      // Reload to pick up restored config
      window.location.reload();
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

      // Check if incoming task is already completed and has saved state
      const completed = progress[key]?.includes(taskId) ?? false;

      set(state => {
        saveCurrentTaskState(state);
        state.menuOpen = false;
        state.activeTaskId = taskId;
        state.previousTaskState = null;  // chose a different task

        const saved = completed ? state.taskStates[taskId] : undefined;
        if (saved) {
          // Restore saved state for completed task
          state.hintsRevealed = saved.hintsRevealed;
          state.attempts = saved.attempts;
          state.lastValidation = saved.lastValidation;
          state.successDismissed = true;  // suppress auto-modal on revisit
        } else {
          state.hintsRevealed = 0;
          state.attempts = 0;
          state.lastValidation = null;
          state.successDismissed = false;
        }
        state.lastValidatedRunCounter = 0;
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

        state.successDismissed = false;

        if (nextTask) {
          // Advance to next task
          state.activeTaskId = nextTask.id;
          state.hintsRevealed = 0;
          state.attempts = 0;
          state.lastValidation = null;
          state.lastValidatedRunCounter = 0;
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
