/**
 * Tests for game store — persistence, crash recovery, and state transitions
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

// Mock localStorage
const storage: Record<string, string> = {};
const localStorageMock = {
  getItem: vi.fn((key: string) => storage[key] ?? null),
  setItem: vi.fn((key: string, value: string) => { storage[key] = value; }),
  removeItem: vi.fn((key: string) => { delete storage[key]; }),
};
Object.defineProperty(globalThis, 'localStorage', { value: localStorageMock, writable: true });

// Mock window.location.reload to prevent actual reloads
Object.defineProperty(globalThis, 'window', {
  value: { location: { reload: vi.fn() } },
  writable: true,
});

// Import after mocks are set up
const { useGameStore } = await import('../../src/stores/game.ts');

const GAME_STORAGE_KEY = 'llm-sim-game';
const PRE_GAME_CONFIG_KEY = 'llm-sim-config-pre-game';

function resetStore() {
  useGameStore.setState({
    active: false,
    activeMode: null,
    activeDifficulty: null,
    activeTaskId: null,
    hintsRevealed: 0,
    attempts: 0,
    lastValidation: null,
    progress: {},
    taskStates: {},
    savedConfigSnapshot: null,
    lastValidatedRunCounter: 0,
    successDismissed: false,
  });
  // Clear storage
  for (const key of Object.keys(storage)) {
    delete storage[key];
  }
  vi.clearAllMocks();
}

describe('Game store — persistence', () => {
  beforeEach(resetStore);

  it('enter() persists active state and writes pre-game config key', () => {
    useGameStore.getState().enter();

    expect(localStorageMock.setItem).toHaveBeenCalled();
    const saved = JSON.parse(storage[GAME_STORAGE_KEY]);
    expect(saved.active).toBe(true);
  });

  it('exit() clears state and persists', () => {
    useGameStore.getState().enter();
    useGameStore.getState().exit();

    const saved = JSON.parse(storage[GAME_STORAGE_KEY]);
    expect(saved.active).toBe(false);
    expect(saved.activeTaskId).toBeNull();
  });

  it('exitTask() persists cleared task state', () => {
    useGameStore.getState().enter();
    useGameStore.setState({ activeTaskId: 'training-beginner-01', activeMode: 'training', activeDifficulty: 'beginner' });
    useGameStore.getState().exitTask();

    const state = useGameStore.getState();
    expect(state.activeTaskId).toBeNull();
    expect(state.hintsRevealed).toBe(0);
    expect(state.attempts).toBe(0);

    // Check it was persisted
    const saved = JSON.parse(storage[GAME_STORAGE_KEY]);
    expect(saved.activeTaskId).toBeNull();
  });

  it('clearLevel() clears mode/difficulty and persists', () => {
    useGameStore.getState().enter();
    useGameStore.getState().selectLevel('training', 'beginner');
    useGameStore.getState().clearLevel();

    const state = useGameStore.getState();
    expect(state.activeMode).toBeNull();
    expect(state.activeDifficulty).toBeNull();

    const saved = JSON.parse(storage[GAME_STORAGE_KEY]);
    expect(saved.activeMode).toBeNull();
    expect(saved.activeDifficulty).toBeNull();
  });
});

describe('Game store — acknowledgeSuccess', () => {
  beforeEach(resetStore);

  it('does nothing if lastValidation is not passed', () => {
    useGameStore.setState({
      active: true,
      activeMode: 'training',
      activeDifficulty: 'beginner',
      activeTaskId: 'training-beginner-01',
      lastValidation: null,
    });

    useGameStore.getState().acknowledgeSuccess();
    expect(useGameStore.getState().activeTaskId).toBe('training-beginner-01');
  });

  it('does nothing if lastValidation.passed is false', () => {
    useGameStore.setState({
      active: true,
      activeMode: 'training',
      activeDifficulty: 'beginner',
      activeTaskId: 'training-beginner-01',
      lastValidation: { passed: false, results: [] },
    });

    useGameStore.getState().acknowledgeSuccess();
    expect(useGameStore.getState().activeTaskId).toBe('training-beginner-01');
  });

  it('advances to next task when passed', () => {
    useGameStore.setState({
      active: true,
      activeMode: 'training',
      activeDifficulty: 'beginner',
      activeTaskId: 'training-beginner-01',
      lastValidation: { passed: true, results: [] },
    });

    useGameStore.getState().acknowledgeSuccess();

    const state = useGameStore.getState();
    expect(state.activeTaskId).toBe('training-beginner-03');
    expect(state.progress['training-beginner']).toContain('training-beginner-01');
    expect(state.hintsRevealed).toBe(0);
    expect(state.attempts).toBe(0);
  });

  it('returns to menu on last task completion', () => {
    useGameStore.setState({
      active: true,
      activeMode: 'training',
      activeDifficulty: 'beginner',
      activeTaskId: 'training-beginner-10',
      lastValidation: { passed: true, results: [] },
    });

    useGameStore.getState().acknowledgeSuccess();

    const state = useGameStore.getState();
    expect(state.activeTaskId).toBeNull();
    expect(state.progress['training-beginner']).toContain('training-beginner-10');
  });

  it('records progress and persists', () => {
    useGameStore.setState({
      active: true,
      activeMode: 'training',
      activeDifficulty: 'beginner',
      activeTaskId: 'training-beginner-01',
      lastValidation: { passed: true, results: [] },
    });

    useGameStore.getState().acknowledgeSuccess();

    const saved = JSON.parse(storage[GAME_STORAGE_KEY]);
    expect(saved.progress['training-beginner']).toContain('training-beginner-01');
  });
});

describe('Game store — crash recovery', () => {
  beforeEach(resetStore);

  it('enter() writes pre-game config key', () => {
    // Set some config in storage first
    storage['llm-sim-config'] = JSON.stringify({ test: true });

    useGameStore.getState().enter();

    expect(storage[PRE_GAME_CONFIG_KEY]).toBeDefined();
  });

  it('exit() removes pre-game config key', () => {
    storage[PRE_GAME_CONFIG_KEY] = JSON.stringify({ test: true });
    useGameStore.setState({ active: true, savedConfigSnapshot: JSON.stringify({ test: true }) });

    useGameStore.getState().exit();

    expect(storage[PRE_GAME_CONFIG_KEY]).toBeUndefined();
  });
});

describe('Game store — successDismissed', () => {
  beforeEach(resetStore);

  it('dismissSuccess() sets successDismissed to true', () => {
    useGameStore.getState().dismissSuccess();
    expect(useGameStore.getState().successDismissed).toBe(true);
  });

  it('reviewSuccess() sets successDismissed to false', () => {
    useGameStore.setState({ successDismissed: true });
    useGameStore.getState().reviewSuccess();
    expect(useGameStore.getState().successDismissed).toBe(false);
  });

  it('startTask() resets successDismissed', () => {
    useGameStore.setState({
      active: true,
      activeMode: 'training',
      activeDifficulty: 'beginner',
      successDismissed: true,
    });

    useGameStore.getState().startTask('training-beginner-01');
    expect(useGameStore.getState().successDismissed).toBe(false);
  });

  it('acknowledgeSuccess() resets successDismissed', () => {
    useGameStore.setState({
      active: true,
      activeMode: 'training',
      activeDifficulty: 'beginner',
      activeTaskId: 'training-beginner-01',
      lastValidation: { passed: true, results: [] },
      successDismissed: true,
    });

    useGameStore.getState().acknowledgeSuccess();
    expect(useGameStore.getState().successDismissed).toBe(false);
  });
});

describe('Game store — per-task state preservation', () => {
  beforeEach(resetStore);

  it('startTask() restores saved state for completed tasks', () => {
    const savedValidation = { passed: true, results: [{ criterion: 'test', passed: true, message: 'ok' }] };
    useGameStore.setState({
      active: true,
      activeMode: 'training',
      activeDifficulty: 'beginner',
      progress: { 'training-beginner': ['training-beginner-01'] },
      taskStates: {
        'training-beginner-01': {
          hintsRevealed: 2,
          attempts: 3,
          lastValidation: savedValidation,
        },
      },
    });

    useGameStore.getState().startTask('training-beginner-01');

    const state = useGameStore.getState();
    expect(state.activeTaskId).toBe('training-beginner-01');
    expect(state.hintsRevealed).toBe(2);
    expect(state.attempts).toBe(3);
    expect(state.lastValidation).toEqual(savedValidation);
    expect(state.successDismissed).toBe(true);  // suppress auto-modal
  });

  it('startTask() resets state for non-completed tasks', () => {
    useGameStore.setState({
      active: true,
      activeMode: 'training',
      activeDifficulty: 'beginner',
      progress: {},
    });

    useGameStore.getState().startTask('training-beginner-01');

    const state = useGameStore.getState();
    expect(state.activeTaskId).toBe('training-beginner-01');
    expect(state.hintsRevealed).toBe(0);
    expect(state.attempts).toBe(0);
    expect(state.lastValidation).toBeNull();
    expect(state.successDismissed).toBe(false);
  });

  it('resetTask() clears taskStates entry', () => {
    useGameStore.setState({
      active: true,
      activeMode: 'training',
      activeDifficulty: 'beginner',
      activeTaskId: 'training-beginner-01',
      taskStates: {
        'training-beginner-01': {
          hintsRevealed: 2,
          attempts: 3,
          lastValidation: { passed: true, results: [] },
        },
      },
    });

    useGameStore.getState().resetTask();

    const state = useGameStore.getState();
    expect(state.taskStates['training-beginner-01']).toBeUndefined();
  });

  it('resetLevel() clears taskStates for that level', () => {
    useGameStore.setState({
      active: true,
      activeMode: 'training',
      activeDifficulty: 'beginner',
      progress: { 'training-beginner': ['training-beginner-01', 'training-beginner-03'] },
      taskStates: {
        'training-beginner-01': {
          hintsRevealed: 2,
          attempts: 3,
          lastValidation: { passed: true, results: [] },
        },
        'training-beginner-03': {
          hintsRevealed: 1,
          attempts: 1,
          lastValidation: { passed: true, results: [] },
        },
        'training-intermediate-01': {
          hintsRevealed: 1,
          attempts: 2,
          lastValidation: { passed: true, results: [] },
        },
      },
    });

    useGameStore.getState().resetLevel('training', 'beginner');

    const state = useGameStore.getState();
    // Beginner task states should be cleared
    expect(state.taskStates['training-beginner-01']).toBeUndefined();
    expect(state.taskStates['training-beginner-03']).toBeUndefined();
    // Other level's task states should be preserved
    expect(state.taskStates['training-intermediate-01']).toBeDefined();
  });

  it('acknowledgeSuccess() saves completed task state before advancing', () => {
    const winValidation = { passed: true, results: [] };
    useGameStore.setState({
      active: true,
      activeMode: 'training',
      activeDifficulty: 'beginner',
      activeTaskId: 'training-beginner-01',
      hintsRevealed: 2,
      attempts: 5,
      lastValidation: winValidation,
      taskStates: {},
    });

    useGameStore.getState().acknowledgeSuccess();

    const state = useGameStore.getState();
    // Should have saved the completed task's state
    expect(state.taskStates['training-beginner-01']).toEqual({
      hintsRevealed: 2,
      attempts: 5,
      lastValidation: winValidation,
    });
    // Should have advanced to next task
    expect(state.activeTaskId).toBe('training-beginner-03');
  });

  it('taskStates is persisted to localStorage', () => {
    const savedValidation = { passed: true, results: [] };
    useGameStore.setState({
      active: true,
      activeMode: 'training',
      activeDifficulty: 'beginner',
      activeTaskId: 'training-beginner-01',
      hintsRevealed: 1,
      attempts: 2,
      lastValidation: savedValidation,
      taskStates: {},
    });

    // acknowledgeSuccess triggers persist
    useGameStore.getState().acknowledgeSuccess();

    const saved = JSON.parse(storage[GAME_STORAGE_KEY]);
    expect(saved.taskStates).toBeDefined();
    expect(saved.taskStates['training-beginner-01']).toEqual({
      hintsRevealed: 1,
      attempts: 2,
      lastValidation: savedValidation,
    });
  });
});
