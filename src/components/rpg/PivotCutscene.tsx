/**
 * PivotCutscene — full-screen narrative modal for pivot missions (no sim task).
 * "Continue" button calls acknowledgeMissionSuccess() to mark the mission complete.
 */

import { Sparkles, ArrowRight } from 'lucide-react';
import { useRPGStore } from '../../stores/rpg.ts';
import { getMissionById } from '../../rpg/missions/index.ts';

export function PivotCutscene() {
  const activeMissionId = useRPGStore(s => s.activeMissionId);
  const acknowledgeMissionSuccess = useRPGStore(s => s.acknowledgeMissionSuccess);

  if (!activeMissionId) return null;
  const mission = getMissionById(activeMissionId);
  if (!mission || mission.type !== 'pivot') return null;

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50">
      <div
        className="bg-gray-950 border border-amber-500/30 rounded-xl p-8 max-w-lg w-full mx-4 max-h-[80vh] overflow-y-auto"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center gap-3 mb-6">
          <div className="w-12 h-12 rounded-full bg-amber-500/20 flex items-center justify-center">
            <Sparkles className="w-6 h-6 text-amber-400" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-amber-300 uppercase tracking-wider font-mono">
              {mission.title}
            </h2>
            <p className="text-xs text-gray-500 font-mono mt-0.5">
              {mission.subtitle}
            </p>
          </div>
        </div>

        {/* Briefing — narrative lead-in */}
        <div className="space-y-3 mb-6">
          {mission.briefing.trim().split('\n\n').map((block, i) => (
            <p key={i} className="text-sm text-sky-200/80 leading-relaxed font-mono">
              {block.trim()}
            </p>
          ))}
        </div>

        {/* Divider */}
        <div className="border-t border-gray-800 my-5" />

        {/* Success narrative — dramatic reveal */}
        <div className="space-y-3 mb-6">
          {mission.successNarrative.trim().split('\n\n').map((block, i) => (
            <p key={i} className="text-sm text-gray-300/90 leading-relaxed font-mono">
              {block.trim()}
            </p>
          ))}
        </div>

        {/* CTA */}
        <button
          onClick={acknowledgeMissionSuccess}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-teal-600 hover:bg-teal-500 text-white rounded-lg font-medium font-mono transition-colors cursor-pointer"
        >
          Continue
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
