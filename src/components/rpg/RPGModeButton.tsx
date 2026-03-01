/**
 * Header button to toggle RPG mode
 */

import { Rocket } from 'lucide-react';
import { Tooltip } from '../ui/Tooltip.tsx';
import { useRPGStore } from '../../stores/rpg.ts';

export function RPGModeButton() {
  const active = useRPGStore(s => s.active);
  const enter = useRPGStore(s => s.enter);
  const exit = useRPGStore(s => s.exit);

  const handleClick = () => {
    if (active) {
      exit();
    } else {
      enter();
    }
  };

  return (
    <Tooltip text="Play">
      <button
        onClick={handleClick}
        className={`flex items-center gap-2 px-3 py-2 text-sm rounded-lg transition-colors cursor-pointer ${
          active
            ? 'text-amber-400 bg-amber-500/10 hover:bg-amber-500/20'
            : 'text-gray-500 hover:text-white hover:bg-gray-800/50'
        }`}
      >
        <Rocket className="w-4 h-4" />
        <span className="hidden sm:inline">Play</span>
      </button>
    </Tooltip>
  );
}
