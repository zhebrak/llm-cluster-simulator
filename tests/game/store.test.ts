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
    savedConfigSnapshot: null,
    lastValidatedRunCounter: 0,
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
    expect(state.activeTaskId).toBe('training-beginner-02');
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
