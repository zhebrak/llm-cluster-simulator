/**
 * Sequential task locking tests
 *
 * Validates that tasks within each level are locked until their predecessor
 * is completed, while task 1 (order=0) is always unlocked.
 */

import { describe, it, expect } from 'vitest';
import { isTaskUnlocked, getTasksForLevel, getTaskById } from '../../src/game/tasks/index.ts';
import type { GameMode, GameDifficulty } from '../../src/game/types.ts';

const LEVELS: Array<{ mode: GameMode; difficulty: GameDifficulty }> = [
  { mode: 'training', difficulty: 'beginner' },
  { mode: 'training', difficulty: 'intermediate' },
  { mode: 'training', difficulty: 'advanced' },
  { mode: 'inference', difficulty: 'beginner' },
  { mode: 'inference', difficulty: 'intermediate' },
  { mode: 'inference', difficulty: 'advanced' },
];

describe('isTaskUnlocked', () => {
  describe('first task (order=0) is always unlocked', () => {
    for (const { mode, difficulty } of LEVELS) {
      it(`${mode}-${difficulty}: task 1 unlocked with no progress`, () => {
        const tasks = getTasksForLevel(mode, difficulty);
        expect(tasks.length).toBeGreaterThan(0);
        expect(isTaskUnlocked(tasks[0], [])).toBe(true);
      });
    }
  });

  describe('second task (order=1) is locked without completing first', () => {
    for (const { mode, difficulty } of LEVELS) {
      it(`${mode}-${difficulty}: task 2 locked when task 1 not completed`, () => {
        const tasks = getTasksForLevel(mode, difficulty);
        expect(tasks.length).toBeGreaterThanOrEqual(2);
        expect(isTaskUnlocked(tasks[1], [])).toBe(false);
      });
    }
  });

  describe('completing predecessor unlocks next task', () => {
    for (const { mode, difficulty } of LEVELS) {
      it(`${mode}-${difficulty}: task 2 unlocked after task 1 completed`, () => {
        const tasks = getTasksForLevel(mode, difficulty);
        expect(isTaskUnlocked(tasks[1], [tasks[0].id])).toBe(true);
      });
    }
  });

  it('task 3 is locked when only task 1 is completed (not task 2)', () => {
    const tasks = getTasksForLevel('training', 'beginner');
    expect(tasks.length).toBeGreaterThanOrEqual(3);
    // Task 3 (order=2) needs task 2 (order=1), not task 1
    expect(isTaskUnlocked(tasks[2], [tasks[0].id])).toBe(false);
  });

  it('task 3 is unlocked when tasks 1 and 2 are completed', () => {
    const tasks = getTasksForLevel('training', 'beginner');
    expect(isTaskUnlocked(tasks[2], [tasks[0].id, tasks[1].id])).toBe(true);
  });

  it('last task is locked without completing second-to-last', () => {
    const tasks = getTasksForLevel('training', 'beginner');
    const last = tasks[tasks.length - 1];
    // Complete all except the second-to-last
    const allButSecondToLast = tasks
      .filter(t => t.order !== last.order - 1)
      .map(t => t.id);
    expect(isTaskUnlocked(last, allButSecondToLast)).toBe(false);
  });

  it('last task is unlocked when second-to-last is completed', () => {
    const tasks = getTasksForLevel('training', 'beginner');
    const last = tasks[tasks.length - 1];
    const secondToLast = tasks[tasks.length - 2];
    expect(isTaskUnlocked(last, [secondToLast.id])).toBe(true);
  });

  it('cross-level progress does not unlock tasks in another level', () => {
    const trainingTasks = getTasksForLevel('training', 'beginner');
    const inferenceTasks = getTasksForLevel('inference', 'beginner');
    // Complete all training-beginner tasks
    const trainingIds = trainingTasks.map(t => t.id);
    // Inference task 2 should still be locked (needs inference task 1, not training)
    expect(isTaskUnlocked(inferenceTasks[1], trainingIds)).toBe(false);
  });

  it('sequential chain: each task N unlocked iff task N-1 in completedIds', () => {
    const tasks = getTasksForLevel('inference', 'intermediate');
    const completedIds: string[] = [];

    for (let i = 0; i < tasks.length; i++) {
      // Task i should be unlocked now
      expect(isTaskUnlocked(tasks[i], completedIds)).toBe(true);
      // Task i+1 (if exists) should be locked
      if (i + 1 < tasks.length) {
        expect(isTaskUnlocked(tasks[i + 1], completedIds)).toBe(false);
      }
      // "Complete" task i
      completedIds.push(tasks[i].id);
    }
  });
});

describe('startTask guard (store-level)', () => {
  it('startTask rejects locked tasks silently', async () => {
    // Import store dynamically to get a fresh instance context
    const { useGameStore } = await import('../../src/stores/game.ts');
    const store = useGameStore.getState();

    const tasks = getTasksForLevel('training', 'beginner');
    expect(tasks.length).toBeGreaterThanOrEqual(2);

    // Activate game mode and select level
    useGameStore.setState({
      active: true,
      activeMode: 'training',
      activeDifficulty: 'beginner',
      progress: {},
    });

    // Try to start task 2 (locked — task 1 not completed)
    useGameStore.getState().startTask(tasks[1].id);
    expect(useGameStore.getState().activeTaskId).toBeNull();
  });

  it('startTask accepts unlocked first task', async () => {
    const { useGameStore } = await import('../../src/stores/game.ts');

    const tasks = getTasksForLevel('training', 'beginner');

    useGameStore.setState({
      active: true,
      activeMode: 'training',
      activeDifficulty: 'beginner',
      progress: {},
      activeTaskId: null,
    });

    // Task 1 (order=0) is always unlocked
    useGameStore.getState().startTask(tasks[0].id);
    expect(useGameStore.getState().activeTaskId).toBe(tasks[0].id);
  });

  it('startTask accepts task 2 after task 1 completed', async () => {
    const { useGameStore } = await import('../../src/stores/game.ts');

    const tasks = getTasksForLevel('training', 'beginner');
    const key = 'training-beginner';

    useGameStore.setState({
      active: true,
      activeMode: 'training',
      activeDifficulty: 'beginner',
      progress: { [key]: [tasks[0].id] },
      activeTaskId: null,
    });

    useGameStore.getState().startTask(tasks[1].id);
    expect(useGameStore.getState().activeTaskId).toBe(tasks[1].id);
  });
});
