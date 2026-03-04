/**
 * Shared persistence helpers for game and RPG stores.
 * Both stores snapshot/restore the same config state (CONFIG_STORAGE_KEY),
 * so snapshotConfig() needs no parameters.
 */

import { CONFIG_STORAGE_KEY } from '../game/constants.ts';

/**
 * Snapshot the current config state from localStorage (for crash recovery).
 * Both game and RPG stores snapshot the same simulator config.
 */
export function snapshotConfig(): string | null {
  try {
    return localStorage.getItem(CONFIG_STORAGE_KEY);
  } catch {
    return null;
  }
}

/**
 * Load persisted state from localStorage by key.
 * Returns empty object if missing or corrupt.
 */
export function loadPersistedState<T>(key: string): Partial<T> {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return {};
    return JSON.parse(raw);
  } catch {
    return {};
  }
}

/**
 * Crash recovery: if a pre-mode config snapshot exists but the mode is not active,
 * restore it to CONFIG_STORAGE_KEY and clean up the orphaned key.
 */
export function recoverOrphanedConfig(preConfigKey: string, isActive: boolean): void {
  try {
    const preConfig = localStorage.getItem(preConfigKey);
    if (preConfig && !isActive) {
      localStorage.setItem(CONFIG_STORAGE_KEY, preConfig);
      localStorage.removeItem(preConfigKey);
    }
  } catch { /* ignore */ }
}
