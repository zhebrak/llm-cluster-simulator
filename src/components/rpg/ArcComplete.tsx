/**
 * ArcComplete — celebration modal shown when all missions in an arc are done.
 */

import { Orbit, ArrowLeft } from 'lucide-react';
import { useRPGStore } from '../../stores/rpg.ts';
import { ALL_ARCS } from '../../rpg/missions/index.ts';

export function ArcComplete() {
  const showArcComplete = useRPGStore(s => s.showArcComplete);
  const dismissArcComplete = useRPGStore(s => s.dismissArcComplete);

  if (!showArcComplete) return null;

  const arc = ALL_ARCS.find(a => a.id === showArcComplete);
  if (!arc) return null;

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div
        className="bg-gray-950 border border-amber-500/30 rounded-xl p-6 max-w-md w-full mx-4 text-center"
        onClick={e => e.stopPropagation()}
      >
        <div className="w-16 h-16 rounded-full bg-amber-500/20 flex items-center justify-center mx-auto mb-4">
          <Orbit className="w-8 h-8 text-amber-400" />
        </div>

        <h3 className="text-xl font-semibold text-amber-300 mb-2 font-mono uppercase tracking-wider">
          Chapter Complete
        </h3>

        <p className="text-base text-amber-200 mb-1 font-mono">
          {arc.name}
        </p>
        <p className="text-sm text-gray-500 mb-6 font-mono">
          {arc.closureText ?? 'All missions complete.'}
        </p>

        <button
          onClick={dismissArcComplete}
          className="flex items-center justify-center gap-2 px-4 py-2.5 bg-teal-600 hover:bg-teal-500 text-white rounded-lg font-medium font-mono transition-colors cursor-pointer mx-auto"
        >
          <ArrowLeft className="w-4 h-4" />
          To Mission Log
        </button>
      </div>
    </div>
  );
}
