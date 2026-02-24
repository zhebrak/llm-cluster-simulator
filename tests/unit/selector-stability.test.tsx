/**
 * Zustand selector stability tests
 *
 * Zustand v5 uses useSyncExternalStore where the selector is a useCallback
 * dependency. An inline object selector (s => ({ a: s.a, b: s.b })) creates a
 * new reference every render, which useSyncExternalStore sees as a change,
 * triggering an infinite re-render loop.
 *
 * These tests catch that bug class at the hook level under jsdom.
 */

import { describe, it, expect } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useConfigStore, DEMO_PRESETS } from '../../src/stores/config';
import { useSimulationStore } from '../../src/stores/simulation';

// ---------------------------------------------------------------------------
// a) Regression — prove the dangerous pattern actually blows up
// ---------------------------------------------------------------------------
describe('inline object selector (regression)', () => {
  it('triggers infinite loop with useConfigStore', () => {
    expect(() => {
      renderHook(() =>
        useConfigStore(s => ({ mode: s.mode, modelId: s.modelId })),
      );
    }).toThrow();
  });

  it('triggers infinite loop with useSimulationStore', () => {
    expect(() => {
      renderHook(() =>
        useSimulationStore(s => ({ status: s.status, error: s.error })),
      );
    }).toThrow();
  });
});

// ---------------------------------------------------------------------------
// b) All real selectors used in the codebase are stable
// ---------------------------------------------------------------------------
describe('production selectors are stable', () => {
  // Helper: render a hook twice and assert it does not cause excess renders
  function expectStable<T>(useSelector: () => T) {
    const { result, rerender } = renderHook(useSelector);
    const first = result.current;
    rerender();
    expect(result.current).toBe(first);
  }

  // --- useConfigStore selectors (from grep of src/**/*.tsx) ---

  it('s => s.modelId.startsWith("custom-")', () => {
    expectStable(() => useConfigStore(s => s.modelId.startsWith('custom-')));
  });

  it('s => s.training.ppDegree', () => {
    expectStable(() => useConfigStore(s => s.training.ppDegree));
  });

  it('s => s.training.numMicroBatches', () => {
    expectStable(() => useConfigStore(s => s.training.numMicroBatches));
  });

  it('s => s.training.gradientAccumulationSteps', () => {
    expectStable(() => useConfigStore(s => s.training.gradientAccumulationSteps));
  });

  it('s => s.training.pipelineSchedule', () => {
    expectStable(() => useConfigStore(s => s.training.pipelineSchedule));
  });

  it('s => s.training.interleavedStages', () => {
    expectStable(() => useConfigStore(s => s.training.interleavedStages));
  });

  it('s => s.inference.paretoVisiblePrecisions', () => {
    expectStable(() => useConfigStore(s => s.inference.paretoVisiblePrecisions));
  });

  it('s => s.mode', () => {
    expectStable(() => useConfigStore(s => s.mode));
  });

  // --- useSimulationStore selectors ---

  it('s => s.status', () => {
    expectStable(() => useSimulationStore(s => s.status));
  });

  it('s => s.inference.paretoResult', () => {
    expectStable(() => useSimulationStore(s => s.inference.paretoResult));
  });

  it('s => s.inference.paretoProgress', () => {
    expectStable(() => useSimulationStore(s => s.inference.paretoProgress));
  });

  it('s => s.inference.seqLenSweepResult', () => {
    expectStable(() => useSimulationStore(s => s.inference.seqLenSweepResult));
  });

  it('s => s.inference.inputSeqLen', () => {
    expectStable(() => useSimulationStore(s => s.inference.inputSeqLen));
  });
});

// ---------------------------------------------------------------------------
// c) Selectors stable across preset loads
// ---------------------------------------------------------------------------
describe('selectors stable across presets', () => {
  for (const preset of DEMO_PRESETS) {
    it(`stable after loading "${preset.slug}"`, () => {
      // Load the preset outside React
      act(() => {
        useConfigStore.getState().loadPresetBySlug(preset.slug);
      });

      // Key selectors that touch various state shapes
      const selectors = [
        () => useConfigStore(s => s.mode),
        () => useConfigStore(s => s.modelId.startsWith('custom-')),
        () => useConfigStore(s => s.training.ppDegree),
        () => useConfigStore(s => s.training.pipelineSchedule),
        () => useConfigStore(s => s.inference.paretoVisiblePrecisions),
        () => useSimulationStore(s => s.status),
      ];

      for (const useSelector of selectors) {
        const { result, rerender } = renderHook(useSelector);
        const first = result.current;
        rerender();
        expect(result.current).toBe(first);
      }
    });
  }
});
