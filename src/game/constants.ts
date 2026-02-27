/**
 * Shared label maps for game mode UI
 */

import type { GameMode, GameDifficulty } from './types.ts';

export const MODE_LABELS: Record<GameMode, string> = {
  training: 'Training',
  inference: 'Inference',
};

export const DIFFICULTY_LABELS: Record<GameDifficulty, string> = {
  beginner: 'Beginner',
  intermediate: 'Intermediate',
  advanced: 'Advanced',
};
