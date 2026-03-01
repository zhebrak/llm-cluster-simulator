/**
 * Reset button with a full-circle spin animation on click.
 */

import { useState, useCallback } from 'react';
import { RotateCcw } from 'lucide-react';
import { Tooltip } from './Tooltip.tsx';

interface Props {
  onConfirm: () => void;
  tooltip?: string;
}

export function ConfirmResetButton({ onConfirm, tooltip = 'Reset to defaults' }: Props) {
  const [spinning, setSpinning] = useState(false);

  const handleClick = useCallback(() => {
    if (spinning) return;
    setSpinning(true);
    onConfirm();
    setTimeout(() => setSpinning(false), 500);
  }, [spinning, onConfirm]);

  return (
    <Tooltip text={tooltip}>
      <button
        onClick={handleClick}
        className="text-gray-400 hover:text-gray-300 cursor-pointer p-1 flex-shrink-0"
      >
        <RotateCcw
          className="w-4 h-4"
          style={spinning ? { animation: 'confirm-spin 0.5s ease-in-out' } : undefined}
        />
      </button>
    </Tooltip>
  );
}
