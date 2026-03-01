import { useEffect, useCallback, useRef, useState } from 'react';
import { AppShell } from './components/layout/index.ts';
import { ResultsDashboard } from './components/visualization/index.ts';
import { GameOverlay } from './components/game/index.ts';
import { RPGOverlay } from './components/rpg/index.ts';
import { WelcomeModal } from './components/WelcomeModal.tsx';
import { useConfigStore, useSimulationStore, useGameStore, useRPGStore, findPresetBySlug } from './stores/index.ts';
import { decodeShareURL } from './utils/share.ts';

function App() {
  const { runSimulation } = useSimulationStore();
  const loadedFromShare = useRef(false);
  const [showWelcome, setShowWelcome] = useState(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.has('welcome')) {
      localStorage.removeItem('llm-sim-welcomed');
      history.replaceState({}, '', window.location.pathname);
      return true;
    }
    if (params.has('config') || params.has('preset') || params.has('learn') || params.has('rpg')) return false;
    return !localStorage.getItem('llm-sim-welcomed');
  });
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
    // Ctrl+Shift+W - Reset welcome modal (dev testing)
    if (e.ctrlKey && e.shiftKey && e.key === 'W') {
      e.preventDefault();
      localStorage.removeItem('llm-sim-welcomed');
      window.location.reload();
      return;
    }

    if (e.ctrlKey && e.key === 'Enter') {
      e.preventDefault();
      if (e.shiftKey) {
        // Ctrl+Shift+Enter - Auto-optimize (disabled in learning/RPG mode)
        const mode = useConfigStore.getState().mode;
        const gameActive = useGameStore.getState().active;
        const rpgActive = useRPGStore.getState().active;
        if (mode === 'training' && !gameActive && !rpgActive) {
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
    // If welcome modal is showing, deactivate any persisted game/RPG modes
    if (showWelcome) {
      const rpg = useRPGStore.getState();
      if (rpg.active) rpg.exit();
      const game = useGameStore.getState();
      if (game.active) game.exit();
      return;
    }

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

    if (params.has('rpg')) {
      history.replaceState({}, '', window.location.pathname + '#rpg');
      const rpg = useRPGStore.getState();
      if (!rpg.active) {
        rpg.enter();
      }
      return;
    }

    if (params.has('learn')) {
      history.replaceState({}, '', window.location.pathname + '#learn');
      const game = useGameStore.getState();
      if (!game.active) {
        game.enter();
      }
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
      {showWelcome && <WelcomeModal onDismiss={() => setShowWelcome(false)} />}
      {presetError && (
        <div className="fixed top-16 left-1/2 -translate-x-1/2 z-50 flex items-center gap-2 px-4 py-2 bg-gray-800 border border-gray-600 rounded-lg text-sm text-gray-300 shadow-lg">
          <span>{presetError}</span>
          <button onClick={() => setPresetError(null)} className="text-gray-500 hover:text-gray-300 ml-1 cursor-pointer">&times;</button>
        </div>
      )}
      <GameOverlay />
      <RPGOverlay />
      <ResultsDashboard />
    </AppShell>
  );
}

export default App;
