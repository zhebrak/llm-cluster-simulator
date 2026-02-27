/**
 * Header button to toggle game mode
 */

import { GraduationCap } from 'lucide-react';
import { Tooltip } from '../ui/Tooltip.tsx';
import { useGameStore } from '../../stores/game.ts';

export function GameModeButton() {
  const active = useGameStore(s => s.active);
  const enter = useGameStore(s => s.enter);
  const exit = useGameStore(s => s.exit);

  const handleClick = () => {
    if (active) {
      exit();
    } else {
      enter();
    }
  };

  return (
    <Tooltip text="Learn">
      <button
        onClick={handleClick}
        className={`flex items-center gap-2 px-3 py-2 text-sm rounded-lg transition-colors cursor-pointer ${
          active
            ? 'text-accent bg-accent/10 hover:bg-accent/20'
            : 'text-gray-500 hover:text-white hover:bg-gray-800/50'
        }`}
      >
        <GraduationCap className="w-4 h-4" />
        <span className="hidden sm:inline">Learn</span>
      </button>
    </Tooltip>
  );
}
