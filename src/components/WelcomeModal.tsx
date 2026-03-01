import { BarChart3, GraduationCap, Rocket } from 'lucide-react';
import { useGameStore, useRPGStore } from '../stores/index.ts';

const WELCOMED_KEY = 'llm-sim-welcomed';

interface WelcomeModalProps {
  onDismiss: () => void;
}

export function WelcomeModal({ onDismiss }: WelcomeModalProps) {
  const handleSimulator = () => {
    localStorage.setItem(WELCOMED_KEY, '1');
    const game = useGameStore.getState();
    if (game.active) {
      game.exit();
    }
    const rpg = useRPGStore.getState();
    if (rpg.active) {
      rpg.exit();
    }
    onDismiss();
  };

  const handleLearn = () => {
    localStorage.setItem(WELCOMED_KEY, '1');
    const game = useGameStore.getState();
    if (game.active) {
      game.resetToLevelPicker();
    } else {
      game.enter();
    }
    onDismiss();
  };

  const handleRPG = () => {
    localStorage.setItem(WELCOMED_KEY, '1');
    const rpg = useRPGStore.getState();
    if (!rpg.active) {
      rpg.enter();
    }
    onDismiss();
  };

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div
        className="bg-gray-900 border border-gray-700 rounded-xl p-8 max-w-2xl w-full mx-4"
        onClick={e => e.stopPropagation()}
      >
        <h2 className="text-xl font-semibold text-white text-center mb-2">
          Welcome to LLM Cluster Simulator
        </h2>
        <p className="text-sm text-gray-400 text-center mb-6">
          Simulate parallelism strategies, GPU memory, compute efficiency, and inference performance for large language models.
        </p>

        <div className="grid grid-cols-3 gap-4">
          <button
            onClick={handleSimulator}
            className="flex flex-col items-center gap-3 p-5 bg-gray-800 border border-gray-700 rounded-lg hover:border-gray-500 hover:bg-gray-700/50 transition-colors cursor-pointer"
          >
            <div className="w-12 h-12 rounded-full bg-blue-500/20 flex items-center justify-center">
              <BarChart3 className="w-6 h-6 text-blue-400" />
            </div>
            <span className="text-base font-medium text-white">Simulator</span>
            <span className="text-sm text-gray-400 text-center">
              Jump straight into configuring clusters
            </span>
          </button>

          <button
            onClick={handleLearn}
            className="flex flex-col items-center gap-3 p-5 bg-gray-800 border border-gray-700 rounded-lg hover:border-gray-500 hover:bg-gray-700/50 transition-colors cursor-pointer"
          >
            <div className="w-12 h-12 rounded-full bg-teal-500/20 flex items-center justify-center">
              <GraduationCap className="w-6 h-6 text-teal-400" />
            </div>
            <span className="text-base font-medium text-white">Learn</span>
            <span className="text-sm text-gray-400 text-center">
              Learn distributed training & inference interactively
            </span>
          </button>

          <button
            onClick={handleRPG}
            className="flex flex-col items-center gap-3 p-5 bg-gray-800 border border-gray-700 rounded-lg hover:border-gray-500 hover:bg-gray-700/50 transition-colors cursor-pointer"
          >
            <div className="w-12 h-12 rounded-full bg-amber-500/20 flex items-center justify-center">
              <Rocket className="w-6 h-6 text-amber-400" />
            </div>
            <span className="text-base font-medium text-white">Play</span>
            <span className="text-sm text-gray-400 text-center">
              Solve missions aboard a generation ship
            </span>
          </button>
        </div>
      </div>
    </div>
  );
}
