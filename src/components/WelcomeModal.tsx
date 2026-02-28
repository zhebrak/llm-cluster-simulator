import { BarChart3, GraduationCap } from 'lucide-react';
import { useGameStore } from '../stores/index.ts';

const WELCOMED_KEY = 'llm-sim-welcomed';

interface WelcomeModalProps {
  onDismiss: () => void;
}

export function WelcomeModal({ onDismiss }: WelcomeModalProps) {
  const handleSimulator = () => {
    localStorage.setItem(WELCOMED_KEY, '1');
    const game = useGameStore.getState();
    if (game.active) {
      game.exit();  // restores config, deactivates, reloads page
      return;       // reload happens — onDismiss unnecessary
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

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div
        className="bg-gray-900 border border-gray-700 rounded-xl p-8 max-w-lg w-full mx-4"
        onClick={e => e.stopPropagation()}
      >
        <h2 className="text-xl font-semibold text-white text-center mb-2">
          Welcome to LLM Cluster Simulator
        </h2>
        <p className="text-sm text-gray-400 text-center mb-6">
          Simulate parallelism strategies, GPU memory, compute efficiency, and inference performance for large language models.
        </p>

        <div className="grid grid-cols-2 gap-4">
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
            <div className="w-12 h-12 rounded-full bg-accent/20 flex items-center justify-center">
              <GraduationCap className="w-6 h-6 text-accent" />
            </div>
            <span className="text-base font-medium text-white">Learn</span>
            <span className="text-sm text-gray-400 text-center">
              Learn distributed training & inference interactively
            </span>
          </button>
        </div>
      </div>
    </div>
  );
}
