/**
 * PivotCutscene — full-screen narrative modal for pivot missions (no sim task).
 * "Continue" button calls acknowledgeMissionSuccess() to mark the mission complete.
 */

import { Sparkles, ArrowRight } from 'lucide-react';
import { useRPGStore } from '../../stores/rpg.ts';
import { useTheme } from '../../hooks/useTheme.ts';
import { getMissionById, ALL_ARCS } from '../../rpg/missions/index.ts';

/** Map of mission IDs to hero image paths (dark/light variants). */
const PIVOT_IMAGES: Record<string, { dark: string; light: string }> = {
  'mission-1-8': { dark: '/signal_dark.png', light: '/signal_light.png' },
  'mission-2-11': { dark: '/life_dark.png', light: '/life_light.png' },
  'mission-3-7': { dark: '/ship_dark.png', light: '/ship_light.png' },
};

export function PivotCutscene() {
  const activeMissionId = useRPGStore(s => s.activeMissionId);
  const acknowledgeMissionSuccess = useRPGStore(s => s.acknowledgeMissionSuccess);
  const { theme } = useTheme();

  if (!activeMissionId) return null;
  const mission = getMissionById(activeMissionId);
  if (!mission || mission.type !== 'pivot') return null;
  const arc = ALL_ARCS.find(a => a.id === mission.arcId);

  const images = PIVOT_IMAGES[mission.id];
  const heroSrc = images ? (theme === 'dark' ? images.dark : images.light) : null;
  const isDark = theme === 'dark';

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50">
      <div
        className="bg-gray-950 border border-amber-500/30 rounded-xl max-w-lg w-full mx-4 max-h-[80vh] overflow-y-auto"
        onClick={e => e.stopPropagation()}
      >
        {/* Hero image with header overlaid (if available) */}
        {heroSrc ? (
          <div className="relative overflow-hidden rounded-t-xl" style={{ height: 160 }}>
            <img
              src={heroSrc}
              alt=""
              className="w-full h-full object-cover object-[center_35%]"
            />
            <div className="briefing-hero-fade absolute inset-x-0 bottom-0 h-24" />
            <div
              className="animate-scan-sweep absolute inset-x-0 h-px"
              style={{ background: 'linear-gradient(90deg, transparent, rgba(251,191,36,0.18), transparent)' }}
            />
            {/* Header pinned to bottom of image */}
            <div className="absolute inset-x-0 bottom-0 px-6 pb-3 flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-amber-500/20 flex items-center justify-center backdrop-blur-sm">
                <Sparkles className="w-5 h-5 text-amber-400" />
              </div>
              <div>
                <h2
                  className="text-lg font-semibold text-amber-300 uppercase tracking-wider font-mono"
                  style={{ textShadow: isDark
                    ? '0 1px 4px rgba(0,0,0,0.8)'
                    : '0 1px 4px rgba(255,255,255,0.9), 0 0 8px rgba(255,255,255,0.7)'
                  }}
                >
                  {mission.title}
                </h2>
                <p
                  className="text-xs text-amber-400/70 font-mono mt-0.5"
                  style={{ textShadow: isDark
                    ? '0 1px 3px rgba(0,0,0,0.8)'
                    : '0 1px 3px rgba(255,255,255,0.9), 0 0 6px rgba(255,255,255,0.7)'
                  }}
                >
                  {mission.subtitle}
                </p>
              </div>
            </div>
          </div>
        ) : (
          /* Fallback: plain header for pivots without images */
          <div className="flex items-center gap-3 px-6 pt-6 mb-1">
            <div className="w-10 h-10 rounded-full bg-amber-500/20 flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-amber-400" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-amber-300 uppercase tracking-wider font-mono">
                {mission.title}
              </h2>
              <p className="text-xs text-gray-500 font-mono mt-0.5">
                {mission.subtitle}
              </p>
            </div>
          </div>
        )}

        {/* Body */}
        <div className="px-6 pb-6 pt-4">
          {/* Briefing — narrative lead-in */}
          <div className="space-y-3 mb-6">
            {mission.briefing.trim().split('\n\n').map((block, i) => (
              <p key={i} className="text-sm text-gray-300 leading-relaxed font-mono">
                {block.trim()}
              </p>
            ))}
          </div>

          {/* Divider */}
          <div className="border-t border-gray-800 my-5" />

          {/* Success narrative — dramatic reveal */}
          <div className="space-y-3 mb-6">
            {mission.successNarrative.trim().split('\n\n').map((block, i) => (
              <p key={i} className="text-sm text-gray-400 leading-relaxed font-mono italic">
                {block.trim()}
              </p>
            ))}
          </div>

          {/* Arc closure epilogue (replaces ArcComplete modal for pivot-ending arcs) */}
          {arc?.closureText && (
            <p className="text-sm text-amber-300/70 leading-relaxed font-mono italic mt-4 border-t border-gray-800 pt-4">
              {arc.closureText}
            </p>
          )}

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
    </div>
  );
}
