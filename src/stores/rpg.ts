/**
 * RPG game mode store — manages mission progression, skills, and narrative state.
 *
 * State machine (RPGOverlay derives from state):
 *   !active                                         → null
 *   active && showArcComplete                       → ArcComplete
 *   active && !activeMissionId && !missionSelectDismissed → MissionSelect
 *   active && !activeMissionId && missionSelectDismissed  → null (RPG active, no UI)
 *   active && activeMissionId && passed && !dismissed → MissionHUD + MissionSuccess
 *   active && activeMissionId                       → MissionHUD
 *
 * Mutual exclusion: RPG and learning mode cannot be active simultaneously.
 * Both manipulate the config store — soft-exit one before entering the other.
 */

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import type { ValidationResult } from '../game/types.ts';
import type { TaskConfigSnapshot } from '../game/validation.ts';
import { buildValidationContext, validateTask, captureTaskConfig, validateExpectedChanges } from '../game/validation.ts';
import { applySetupToConfig } from '../game/setup.ts';
import { RPG_STORAGE_KEY, PRE_RPG_CONFIG_KEY } from '../rpg/constants.ts';
import { snapshotConfig, loadPersistedState, recoverOrphanedConfig } from './persistence.ts';
import { getMissionById, getMissionsForArc, isMissionUnlocked, ALL_ARCS, ALL_MISSIONS } from '../rpg/missions/index.ts';
import { HARDWARE_PROGRESSION } from '../rpg/hardware.ts';
import { useConfigStore } from './config.ts';
import { useSimulationStore } from './simulation.ts';
import { useGameStore, _registerRPGStore } from './game.ts';

interface RPGState {
  // Core
  active: boolean;
  activeMissionId: string | null;

  // Task state
  hintsRevealed: number;
  attempts: number;
  lastValidation: ValidationResult | null;
  approachValid: boolean;
  lastValidatedRunCounter: number;

  // Multi-objective state
  activeObjectiveId: string | null;
  clearedObjectiveIds: string[];

  // Progression
  completedMissions: string[];      // mission IDs
  missionStates: Record<string, {   // per-mission saved state
    hintsRevealed: number;
    attempts: number;
    lastValidation: ValidationResult | null;
    clearedObjectiveIds?: string[];
  }>;

  // Intro briefing
  introSeen: boolean;              // Permanent latch — once true, stays true
  seenHardwareTierIds: string[];   // Tier IDs the player has viewed in MissionSelect

  // Transient UI (NOT persisted)
  showingBriefing: boolean;        // True when re-reading briefing via button

  // Config snapshots
  savedConfigSnapshot: string | null;
  taskConfigSnapshot: TaskConfigSnapshot | null;

  // Resume
  lastMissionId: string | null;

  // Transient UI state
  showArcComplete: string | null;   // arcId or null
  successDismissed: boolean;
  menuOpen: boolean;
  missionSelectDismissed: boolean;
  // Actions
  enter: () => void;
  exit: () => void;
  startMission: (id: string) => void;
  resetMission: () => void;
  revealNextHint: () => void;
  acknowledgeMissionSuccess: () => void;
  dismissSuccess: () => void;
  reviewMissionSuccess: () => void;
  dismissArcComplete: () => void;
  openMenu: () => void;
  closeMenu: () => void;
  dismissMissionSelect: () => void;
  showMissionSelect: () => void;
  dismissIntro: () => void;
  showBriefing: () => void;
  markHardwareSeen: (ids: string[]) => void;
  resetProgress: (arcId: string) => void;
  selectObjective: (objectiveId: string) => void;
}

// ── Persistence helpers ──────────────────────────────────────────────────

function persistRPGState(state: RPGState): void {
  try {
    const toSave = {
      active: state.active,
      activeMissionId: state.activeMissionId,
      hintsRevealed: state.hintsRevealed,
      attempts: state.attempts,
      completedMissions: state.completedMissions,
      missionStates: state.missionStates,
      savedConfigSnapshot: state.savedConfigSnapshot,
      taskConfigSnapshot: state.taskConfigSnapshot,
      lastValidatedRunCounter: state.lastValidatedRunCounter,
      lastMissionId: state.lastMissionId,
      introSeen: state.introSeen,
      seenHardwareTierIds: state.seenHardwareTierIds,
      activeObjectiveId: state.activeObjectiveId,
      clearedObjectiveIds: state.clearedObjectiveIds,
    };
    localStorage.setItem(RPG_STORAGE_KEY, JSON.stringify(toSave));
  } catch { /* localStorage full or unavailable */ }
}


function restoreMissionState(state: RPGState, missionId: string): void {
  const saved = state.missionStates[missionId];
  const isCompleted = state.completedMissions.includes(missionId);

  if (isCompleted && saved) {
    state.hintsRevealed = saved.hintsRevealed;
    state.attempts = saved.attempts;
    state.lastValidation = saved.lastValidation;
    state.successDismissed = true;
    state.clearedObjectiveIds = saved.clearedObjectiveIds ?? [];
  } else if (saved) {
    state.hintsRevealed = saved.hintsRevealed;
    state.attempts = saved.attempts;
    state.lastValidation = saved.lastValidation;
    state.successDismissed = false;
    state.clearedObjectiveIds = saved.clearedObjectiveIds ?? [];
  } else {
    state.hintsRevealed = 0;
    state.attempts = 0;
    state.lastValidation = null;
    state.successDismissed = false;
    state.clearedObjectiveIds = [];
  }
}

function saveMissionState(state: RPGState): void {
  if (state.activeMissionId) {
    state.missionStates[state.activeMissionId] = {
      hintsRevealed: state.hintsRevealed,
      attempts: state.attempts,
      lastValidation: state.lastValidation,
      clearedObjectiveIds: state.clearedObjectiveIds.length > 0 ? state.clearedObjectiveIds : undefined,
    };
  }
}

// ── Load initial state ───────────────────────────────────────────────────

const persisted = loadPersistedState<RPGState>(RPG_STORAGE_KEY);

// Crash recovery: if pre-RPG config exists but RPG is not active, restore
recoverOrphanedConfig(PRE_RPG_CONFIG_KEY, persisted.active ?? false);

// ── Store ────────────────────────────────────────────────────────────────

export const useRPGStore = create<RPGState>()(
  immer((set, get) => ({
    active: persisted.active ?? false,
    activeMissionId: persisted.activeMissionId ?? null,
    hintsRevealed: persisted.hintsRevealed ?? 0,
    attempts: persisted.attempts ?? 0,
    lastValidation: null,
    approachValid: true,
    lastValidatedRunCounter: persisted.lastValidatedRunCounter ?? 0,
    activeObjectiveId: (persisted as Record<string, unknown>).activeObjectiveId as string | null ?? null,
    clearedObjectiveIds: (persisted as Record<string, unknown>).clearedObjectiveIds as string[] ?? [],
    completedMissions: persisted.completedMissions ?? [],
    missionStates: persisted.missionStates ?? {},
    introSeen: (persisted as Record<string, unknown>).introSeen as boolean ?? false,
    seenHardwareTierIds: (persisted as Record<string, unknown>).seenHardwareTierIds as string[] ?? [],
    showingBriefing: false,
    savedConfigSnapshot: persisted.savedConfigSnapshot ?? null,
    taskConfigSnapshot: persisted.taskConfigSnapshot ?? null,
    lastMissionId: persisted.lastMissionId ?? null,
    showArcComplete: null,
    successDismissed: false,
    menuOpen: false,
    missionSelectDismissed: false,
    enter: () => {
      // Mutual exclusion: soft-exit learning mode if active
      const game = useGameStore.getState();
      if (game.active) {
        game.exit();
      }

      const snapshot = snapshotConfig();
      if (snapshot) {
        try {
          localStorage.setItem(PRE_RPG_CONFIG_KEY, snapshot);
        } catch { /* ignore */ }
      }

      const { lastMissionId } = get();

      set(state => {
        state.active = true;
        state.savedConfigSnapshot = snapshot;
        state.missionSelectDismissed = false;

        // Resume previous mission if available (like game mode's enter)
        if (lastMissionId) {
          const mission = getMissionById(lastMissionId);
          if (mission && isMissionUnlocked(mission, state.completedMissions)) {
            state.activeMissionId = lastMissionId;
            restoreMissionState(state, lastMissionId);
          }
        }
      });

      // Config application is outside set() because applySetupToConfig mutates
      // the config store — can't be inside the immer producer.
      if (lastMissionId && get().activeMissionId) {
        const mission = getMissionById(lastMissionId);
        if (mission && mission.type !== 'pivot') {
          applySetupToConfig(mission.setup, mission.primaryMode);
          const taskSnapshot = captureTaskConfig();
          set(state => {
            state.taskConfigSnapshot = taskSnapshot;
            state.approachValid = true;
          });
          useSimulationStore.getState().reset();
        }
      }

      persistRPGState(get());
    },

    exit: () => {
      const { savedConfigSnapshot } = get();

      // Race guard: deactivate FIRST
      set(state => {
        saveMissionState(state);
        state.active = false;
        state.lastMissionId = state.activeMissionId || state.lastMissionId;
        state.activeMissionId = null;
        state.menuOpen = false;
        state.missionSelectDismissed = false;
        state.showArcComplete = null;
        state.lastValidation = null;
        state.hintsRevealed = 0;
        state.attempts = 0;
        state.savedConfigSnapshot = null;
        state.taskConfigSnapshot = null;
        state.approachValid = true;
        state.lastValidatedRunCounter = 0;
        state.successDismissed = false;
        state.activeObjectiveId = null;
        state.clearedObjectiveIds = [];
      });

      try {
        localStorage.removeItem(PRE_RPG_CONFIG_KEY);
      } catch { /* ignore */ }

      // Restore original config via config store (no page reload)
      if (savedConfigSnapshot) {
        useConfigStore.getState().restoreFromSnapshot(savedConfigSnapshot);
      }

      persistRPGState(get());
    },

    startMission: (id: string) => {
      const mission = getMissionById(id);
      if (!mission) return;

      const { completedMissions } = get();
      if (!isMissionUnlocked(mission, completedMissions)) return;

      // Pivot missions: set active mission but skip config/sim setup
      if (mission.type === 'pivot') {
        set(state => {
          saveMissionState(state);
          state.activeMissionId = id;
          state.menuOpen = false;
          state.showArcComplete = null;
          state.hintsRevealed = 0;
          state.attempts = 0;
          state.lastValidation = null;
          state.successDismissed = false;
          state.lastValidatedRunCounter = 0;
          state.activeObjectiveId = null;
          state.clearedObjectiveIds = [];
        });
        persistRPGState(get());
        return;
      }

      set(state => {
        // Save current mission state before switching
        saveMissionState(state);

        state.activeMissionId = id;
        state.menuOpen = false;
        state.showArcComplete = null;

        restoreMissionState(state, id);
        state.lastValidatedRunCounter = 0;

        state.activeObjectiveId = null;
      });

      // Multi-objective: auto-select first uncompleted objective
      if (mission.objectives && mission.objectives.length > 0) {
        const { clearedObjectiveIds } = get();
        const first = mission.objectives.find(
          o => !clearedObjectiveIds.includes(o.id)
        ) ?? mission.objectives[0];
        get().selectObjective(first.id);
        return;
      }

      // Single-objective path
      applySetupToConfig(mission.setup, mission.primaryMode);

      // Capture config snapshot for expected-change validation
      const snapshot = captureTaskConfig();
      set(state => {
        state.taskConfigSnapshot = snapshot;
        state.approachValid = true;
      });

      useSimulationStore.getState().reset();
      persistRPGState(get());
    },

    selectObjective: (objectiveId: string) => {
      const { activeMissionId } = get();
      if (!activeMissionId) return;

      const mission = getMissionById(activeMissionId);
      if (!mission || !mission.objectives) return;

      const objective = mission.objectives.find(o => o.id === objectiveId);
      if (!objective) return;

      // Apply objective's setup and mode
      applySetupToConfig(objective.setup, objective.primaryMode);

      const snapshot = captureTaskConfig();
      set(state => {
        state.activeObjectiveId = objectiveId;
        state.taskConfigSnapshot = snapshot;
        state.approachValid = true;
        state.lastValidation = null;
        state.lastValidatedRunCounter = 0;
      });

      useSimulationStore.getState().reset();
      persistRPGState(get());
    },

    resetMission: () => {
      const { activeMissionId, activeObjectiveId } = get();
      if (!activeMissionId) return;

      const mission = getMissionById(activeMissionId);
      if (!mission) return;

      // Multi-objective: reset active objective's setup
      if (mission.objectives && activeObjectiveId) {
        const objective = mission.objectives.find(o => o.id === activeObjectiveId);
        if (objective) {
          applySetupToConfig(objective.setup, objective.primaryMode);
        }
      } else {
        // Single-objective: re-apply mission setup
        applySetupToConfig(mission.setup, mission.primaryMode);
      }

      const snapshot = captureTaskConfig();
      set(state => {
        state.taskConfigSnapshot = snapshot;
        state.approachValid = true;
        state.lastValidation = null;
        state.lastValidatedRunCounter = 0;
      });

      useSimulationStore.getState().reset();
      persistRPGState(get());
    },

    revealNextHint: () => {
      const { activeMissionId, hintsRevealed } = get();
      if (!activeMissionId) return;
      const mission = getMissionById(activeMissionId);
      if (!mission) return;
      if (hintsRevealed >= mission.hints.length) return;

      set(state => {
        state.hintsRevealed = state.hintsRevealed + 1;
      });
      persistRPGState(get());
    },

    acknowledgeMissionSuccess: () => {
      const { activeMissionId, activeObjectiveId, clearedObjectiveIds } = get();
      if (!activeMissionId) return;

      const mission = getMissionById(activeMissionId);
      if (!mission) return;

      // Multi-objective: handle per-objective completion
      let pendingClear: string[] | null = null;
      if (mission.objectives && mission.objectives.length > 0 && activeObjectiveId) {
        const newCleared = [...clearedObjectiveIds];
        if (!newCleared.includes(activeObjectiveId)) {
          newCleared.push(activeObjectiveId);
        }

        const allObjectivesCleared = mission.objectives.every(o => newCleared.includes(o.id));

        if (!allObjectivesCleared) {
          // Mid-mission objective clear: advance to next uncompleted objective
          const nextObjective = mission.objectives.find(o => !newCleared.includes(o.id));
          set(state => {
            state.clearedObjectiveIds = newCleared;
            state.activeObjectiveId = null;
            state.lastValidation = null;
            state.lastValidatedRunCounter = 0;
            state.successDismissed = false;
            saveMissionState(state);
          });
          persistRPGState(get());
          if (nextObjective) {
            get().selectObjective(nextObjective.id);
          }
          return;
        }

        // All objectives cleared — fall through to full mission completion
        pendingClear = newCleared;
      }

      // Full mission success
      set(state => {
        // Apply pending objective clear if coming from last-objective completion
        if (pendingClear) {
          state.clearedObjectiveIds = pendingClear;
          state.activeObjectiveId = null;
          state.lastValidation = null;
          state.lastValidatedRunCounter = 0;
        }
        saveMissionState(state);

        // Record completion
        if (!state.completedMissions.includes(activeMissionId)) {
          state.completedMissions.push(activeMissionId);
        }

        // Check if arc is now fully complete
        const arcMissions = getMissionsForArc(mission.arcId);
        const allComplete = arcMissions.every(m =>
          state.completedMissions.includes(m.id)
        );

        if (allComplete) {
          const arcHasPivot = arcMissions.some(m => m.type === 'pivot');
          if (!arcHasPivot) {
            state.showArcComplete = mission.arcId;
          }
          state.activeMissionId = null;
        } else {
          // Pivot priority: if the arc's pivot is now unlocked, auto-start cutscene
          const arcPivot = arcMissions.find(m =>
            m.type === 'pivot' &&
            !state.completedMissions.includes(m.id) &&
            isMissionUnlocked(m, state.completedMissions)
          );

          if (arcPivot) {
            state.activeMissionId = arcPivot.id;
            state.lastMissionId = arcPivot.id;
          } else {
            // No pivot available — find next unlocked regular mission
            let nextMission = arcMissions.find(m =>
              !state.completedMissions.includes(m.id) &&
              isMissionUnlocked(m, state.completedMissions)
            );
            if (!nextMission) {
              nextMission = ALL_MISSIONS.find(m =>
                !state.completedMissions.includes(m.id) &&
                isMissionUnlocked(m, state.completedMissions)
              );
            }
            state.activeMissionId = null;
            state.lastMissionId = nextMission?.id ?? null;
          }
        }

        state.successDismissed = false;
        state.missionSelectDismissed = false;
        state.hintsRevealed = 0;
        state.attempts = 0;
        state.lastValidation = null;
        state.lastValidatedRunCounter = 0;
        state.activeObjectiveId = null;
        state.clearedObjectiveIds = [];
      });

      persistRPGState(get());
    },

    dismissSuccess: () => {
      set(state => { state.successDismissed = true; });
    },

    reviewMissionSuccess: () => {
      set(state => { state.successDismissed = false; });
    },

    dismissArcComplete: () => {
      set(state => {
        state.showArcComplete = null;
      });
      persistRPGState(get());
    },

    openMenu: () => {
      set(state => { state.menuOpen = true; });
    },

    closeMenu: () => {
      set(state => { state.menuOpen = false; });
    },

    dismissMissionSelect: () => {
      set(state => {
        if (!state.activeMissionId) {
          // No mission in progress — exit RPG mode entirely
          state.active = false;
        } else {
          state.missionSelectDismissed = true;
        }
      });
    },

    showMissionSelect: () => {
      set(state => { state.missionSelectDismissed = false; });
    },

    dismissIntro: () => {
      set(state => {
        state.introSeen = true;
        state.showingBriefing = false;
      });
      persistRPGState(get());
    },

    showBriefing: () => {
      set(state => { state.showingBriefing = true; });
    },

    markHardwareSeen: (ids: string[]) => {
      set(state => { state.seenHardwareTierIds = ids; });
      persistRPGState(get());
    },

    resetProgress: (arcId: string) => {
      const arc = ALL_ARCS.find(a => a.id === arcId);
      if (!arc) return;
      const resetOrder = arc.order;

      // Collect all mission IDs belonging to arcs with order >= resetOrder
      const arcsToReset = ALL_ARCS.filter(a => a.order >= resetOrder);
      const missionIdsToReset = new Set(
        ALL_MISSIONS.filter(m => arcsToReset.some(a => a.id === m.arcId)).map(m => m.id)
      );

      set(state => {
        state.completedMissions = state.completedMissions.filter(id => !missionIdsToReset.has(id));
        const newMissionStates: typeof state.missionStates = {};
        for (const [id, ms] of Object.entries(state.missionStates)) {
          if (!missionIdsToReset.has(id)) newMissionStates[id] = ms;
        }
        state.missionStates = newMissionStates;

        // Clear active task state if current mission belongs to a reset arc
        if (state.activeMissionId && missionIdsToReset.has(state.activeMissionId)) {
          state.activeMissionId = null;
          state.hintsRevealed = 0;
          state.attempts = 0;
          state.lastValidation = null;
          state.successDismissed = false;
          state.activeObjectiveId = null;
          state.clearedObjectiveIds = [];
        }
        if (state.lastMissionId && missionIdsToReset.has(state.lastMissionId)) {
          state.lastMissionId = null;
        }

        state.showArcComplete = null;

        // Recompute seenHardwareTierIds — keep tiers whose unlockedBy is still completed or null
        const survivingCompleted = state.completedMissions;
        state.seenHardwareTierIds = HARDWARE_PROGRESSION
          .filter(t => t.unlockedBy === null || survivingCompleted.includes(t.unlockedBy))
          .map(t => t.id);

        // Reset arc 1 clears everything including intro
        if (resetOrder === 1) {
          state.introSeen = false;
        }
      });
      persistRPGState(get());
    },

  })),
);

// Register with game store for mutual exclusion (avoids circular import)
_registerRPGStore(useRPGStore);

// ── Simulation subscriber ────────────────────────────────────────────────

useSimulationStore.subscribe((simState) => {
  const rpg = useRPGStore.getState();
  if (!rpg.active || !rpg.activeMissionId) return;
  if (simState.status !== 'complete') return;
  if (simState.runCounter <= rpg.lastValidatedRunCounter) return;

  const mission = getMissionById(rpg.activeMissionId);
  if (!mission) return;

  // Multi-objective: validate active objective's criteria
  let criteria = mission.winningCriteria;
  let expectedChanges = mission.expectedChanges;
  let mode = mission.primaryMode;

  if (mission.objectives && rpg.activeObjectiveId) {
    const objective = mission.objectives.find(o => o.id === rpg.activeObjectiveId);
    if (objective) {
      criteria = objective.winningCriteria;
      expectedChanges = objective.expectedChanges;
      mode = objective.primaryMode;
    }
  }

  // Multi-objective with no active objective selected → skip validation
  if (mission.objectives && !rpg.activeObjectiveId) return;

  const ctx = buildValidationContext(simState, mode);
  const result = validateTask(ctx, criteria);

  // Validate expected changes (approach validation)
  let approachValid = true;
  if (expectedChanges && rpg.taskConfigSnapshot) {
    const currentConfig = captureTaskConfig();
    const { valid } = validateExpectedChanges(rpg.taskConfigSnapshot, currentConfig, expectedChanges);
    approachValid = valid;
  }

  useRPGStore.setState(state => {
    state.lastValidation = result;
    state.approachValid = approachValid;
    state.lastValidatedRunCounter = simState.runCounter;
    state.attempts = state.attempts + 1;
    saveMissionState(state);
  });

  persistRPGState(useRPGStore.getState());
});
