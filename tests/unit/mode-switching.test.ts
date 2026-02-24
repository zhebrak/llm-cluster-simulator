/**
 * Mode Switching Tests
 *
 * Unit tests for the Zustand config store's mode switching behavior.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { useConfigStore } from '../../src/stores/config.ts';
import { useSimulationStore } from '../../src/stores/simulation.ts';

describe('Mode Switching', () => {
  beforeEach(() => {
    useConfigStore.getState().reset();
    useSimulationStore.getState().reset();
  });

  it('should initialize in training mode', () => {
    const config = useConfigStore.getState();
    expect(config.mode).toBe('training');
  });

  it('should switch to inference mode', () => {
    const store = useConfigStore.getState();
    store.setMode('inference');

    const config = useConfigStore.getState();
    expect(config.mode).toBe('inference');
  });

  it('should preserve model config when switching modes', () => {
    const store = useConfigStore.getState();
    store.setModel('llama2-70b');

    const modelBefore = useConfigStore.getState().modelId;

    store.setMode('inference');

    const modelAfter = useConfigStore.getState().modelId;
    expect(modelAfter).toBe(modelBefore);
    expect(modelAfter).toBe('llama2-70b');
  });

  it('should preserve cluster config when switching modes', () => {
    const store = useConfigStore.getState();
    store.setCluster('8x-h100');

    const clusterBefore = useConfigStore.getState().clusterId;

    store.setMode('inference');

    const clusterAfter = useConfigStore.getState().clusterId;
    expect(clusterAfter).toBe(clusterBefore);
  });

  it('should have separate training and inference configs', () => {
    const store = useConfigStore.getState();

    // Set training params
    store.setTrainingParams({ globalBatchSize: 2048 });

    // Set inference params
    store.setInferenceParams({ batchSize: 32, inputSeqLen: 1024 });

    const config = useConfigStore.getState();
    expect(config.training.globalBatchSize).toBe(2048);
    expect(config.inference.batchSize).toBe(32);
    expect(config.inference.inputSeqLen).toBe(1024);
  });
});
