/**
 * Task barrel — aggregates all tasks and provides lookup utilities
 */

import type { GameTask, GameMode, GameDifficulty } from '../types.ts';
import { TRAINING_BEGINNER_TASKS } from './training-beginner.ts';
import { TRAINING_INTERMEDIATE_TASKS } from './training-intermediate.ts';
import { TRAINING_ADVANCED_TASKS } from './training-advanced.ts';
import { INFERENCE_BEGINNER_TASKS } from './inference-beginner.ts';
import { INFERENCE_INTERMEDIATE_TASKS } from './inference-intermediate.ts';
import { INFERENCE_ADVANCED_TASKS } from './inference-advanced.ts';

export const ALL_TASKS: GameTask[] = [
  ...TRAINING_BEGINNER_TASKS,
  ...TRAINING_INTERMEDIATE_TASKS,
  ...TRAINING_ADVANCED_TASKS,
  ...INFERENCE_BEGINNER_TASKS,
  ...INFERENCE_INTERMEDIATE_TASKS,
  ...INFERENCE_ADVANCED_TASKS,
];

const taskMap = new Map<string, GameTask>();
for (const task of ALL_TASKS) {
  taskMap.set(task.id, task);
}

export function getTaskById(id: string): GameTask | undefined {
  return taskMap.get(id);
}

export function getTasksForLevel(mode: GameMode, difficulty: GameDifficulty): GameTask[] {
  return ALL_TASKS
    .filter(t => t.mode === mode && t.difficulty === difficulty)
    .sort((a, b) => a.order - b.order);
}

export function isTaskUnlocked(task: GameTask, completedIds: readonly string[]): boolean {
  if (task.order === 0) return true;
  const tasks = getTasksForLevel(task.mode, task.difficulty);
  const predecessor = tasks.find(t => t.order === task.order - 1);
  if (!predecessor) return true; // defensive
  return completedIds.includes(predecessor.id);
}
