import { useEffect, useCallback, useRef, useState } from 'react';
import { AppShell } from './components/layout/index.ts';
import { ResultsDashboard } from './components/visualization/index.ts';
import { useConfigStore, useSimulationStore, findPresetBySlug } from './stores/index.ts';
import { decodeShareURL } from './utils/share.ts';

function App() {
  const { runSimulation } = useSimulationStore();
  const loadedFromShare = useRef(false);
  const [presetError, setPresetError] = useState<string | null>(() => {
    const params = new URLSearchParams(window.location.search);
    const presetSlug = params.get('preset');
    if (presetSlug && !params.get('config') && !findPresetBySlug(presetSlug)) {
      return `Preset "${presetSlug}" not found`;
    }
    return null;
  });

  // Keyboard shortcuts
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.ctrlKey && e.key === 'Enter') {
      e.preventDefault();
      if (e.shiftKey) {
        // Ctrl+Shift+Enter - Auto-optimize
        const mode = useConfigStore.getState().mode;
        if (mode === 'training') {
          useSimulationStore.getState().autoOptimizeTraining();
        }
      } else {
        // Ctrl+Enter - Run simulation
        runSimulation();
      }
    }
  }, [runSimulation]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  // Apply URL-encoded config or preset if present
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);

    // ?config= takes precedence over ?preset=
    const encoded = params.get('config');
    if (encoded) {
      const config = decodeShareURL(encoded);
      if (!config) return;

      loadedFromShare.current = true;
      useConfigStore.getState().loadShareConfig(config);
      history.replaceState({}, '', window.location.pathname + '#shared');
      useSimulationStore.getState().runSimulation();
      return;
    }

    const presetSlug = params.get('preset');
    if (presetSlug) {
      const match = findPresetBySlug(presetSlug);
      if (match) {
        loadedFromShare.current = true;
        useConfigStore.getState().loadPresetBySlug(presetSlug);
        history.replaceState({}, '', window.location.pathname + '#preset');
        useSimulationStore.getState().runSimulation();
      } else {
        history.replaceState({}, '', window.location.pathname);
      }
    }
  }, []);

  // Auto-dismiss preset error after 5s
  useEffect(() => {
    if (!presetError) return;
    const timer = setTimeout(() => setPresetError(null), 5000);
    return () => clearTimeout(timer);
  }, [presetError]);

  // Clear #shared or #preset hash when user modifies config after loading
  useEffect(() => {
    const unsub = useConfigStore.subscribe(() => {
      if (loadedFromShare.current) {
        loadedFromShare.current = false;
        return;
      }
      const h = window.location.hash;
      if (h === '#shared' || h === '#preset') {
        history.replaceState({}, '', window.location.pathname);
      }
    });
    return unsub;
  }, []);

  return (
    <AppShell>
      {presetError && (
        <div className="fixed top-16 left-1/2 -translate-x-1/2 z-50 flex items-center gap-2 px-4 py-2 bg-gray-800 border border-gray-600 rounded-lg text-sm text-gray-300 shadow-lg">
          <span>{presetError}</span>
          <button onClick={() => setPresetError(null)} className="text-gray-500 hover:text-gray-300 ml-1 cursor-pointer">&times;</button>
        </div>
      )}
      <ResultsDashboard />
    </AppShell>
  );
}

export default App;
