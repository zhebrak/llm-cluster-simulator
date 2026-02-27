/**
 * Game overlay root — routes to Menu, HUD, Success, or LevelComplete based on store state
 *
 * State machine:
 *   !active                                          → nothing
 *   active && showLevelComplete                      → LevelCompleteModal
 *   active && !activeTaskId                           → GameMenu
 *   active && activeTaskId && passed                  → SuccessModal
 *   active && activeTaskId                            → TaskHUD
 */

import { useGameStore } from '../../stores/game.ts';
import { GameMenu } from './GameMenu.tsx';
import { TaskHUD } from './TaskHUD.tsx';
import { SuccessModal } from './SuccessModal.tsx';
import { LevelCompleteModal } from './LevelCompleteModal.tsx';

export function GameOverlay() {
  const active = useGameStore(s => s.active);
  const activeTaskId = useGameStore(s => s.activeTaskId);
  const activeMode = useGameStore(s => s.activeMode);
  const activeDifficulty = useGameStore(s => s.activeDifficulty);
  const lastValidation = useGameStore(s => s.lastValidation);
  const showLevelComplete = useGameStore(s => s.showLevelComplete);
  const clearLevel = useGameStore(s => s.clearLevel);
  const resetLevel = useGameStore(s => s.resetLevel);
  const menuOpen = useGameStore(s => s.menuOpen);

  if (!active) return null;

  // Level completion celebration (one-time after finishing last task)
  if (showLevelComplete && activeMode && activeDifficulty) {
    return (
      <LevelCompleteModal
        mode={activeMode}
        difficulty={activeDifficulty}
        onBack={clearLevel}
        onReplay={() => resetLevel(activeMode, activeDifficulty)}
      />
    );
  }

  // No active task → menu (level grid or task list)
  if (!activeTaskId) {
    return <GameMenu />;
  }

  // Active task with passing validation → success modal (HUD behind)
  if (lastValidation?.passed) {
    return (
      <>
        <TaskHUD />
        <SuccessModal />
      </>
    );
  }

  // Active task with menu open → HUD in background, modal on top
  if (menuOpen) {
    return (
      <>
        <TaskHUD />
        <GameMenu />
      </>
    );
  }

  // Active task, working → HUD only
  return <TaskHUD />;
}
