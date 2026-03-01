/**
 * Shared constants for game mode
 */

import type { GameMode, GameDifficulty } from './types.ts';

// ── localStorage keys ────────────────────────────────────────────────────

export const GAME_STORAGE_KEY = 'llm-sim-game';
export const PRE_GAME_CONFIG_KEY = 'llm-sim-config-pre-game';
export const CONFIG_STORAGE_KEY = 'llm-sim-config';

// ── Label maps ───────────────────────────────────────────────────────────

export const MODE_LABELS: Record<GameMode, string> = {
  training: 'Training',
  inference: 'Inference',
};

export const DIFFICULTY_LABELS: Record<GameDifficulty, string> = {
  beginner: 'Beginner',
  intermediate: 'Intermediate',
  advanced: 'Advanced',
};
