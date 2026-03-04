/**
 * IntroBriefing — one-time intro shown on first RPG entry.
 * Re-accessible via "View Briefing" button in MissionSelect.
 * Shows arc-appropriate text and hero image based on player progression.
 */

import { Terminal } from 'lucide-react';
import { useRPGStore } from '../../stores/rpg.ts';
import { useTheme } from '../../hooks/useTheme.ts';
import { getActiveArc } from '../../rpg/missions/index.ts';
import { ModalBackdrop } from '../ui/ModalBackdrop.tsx';

export function IntroBriefing() {
  const dismissIntro = useRPGStore(s => s.dismissIntro);
  const introSeen = useRPGStore(s => s.introSeen);
  const completedMissions = useRPGStore(s => s.completedMissions);
  const { theme } = useTheme();

  const arc = getActiveArc(completedMissions);
  const isDark = theme === 'dark';

  const heroSrc = arc.heroImage
    ? (isDark ? arc.heroImage.dark : arc.heroImage.light)
    : (isDark ? '/ship_dark.png' : '/ship_light.png');

  const paragraphs = arc.briefing?.split('\n\n') ?? [];

  return (
    <ModalBackdrop backdropClass="bg-black/80">
      <div
        className="bg-gray-950 border border-amber-500/30 rounded-xl max-w-lg w-full mx-4"
        onClick={e => e.stopPropagation()}
      >
        {/* Hero image with header overlaid */}
        <div className="relative overflow-hidden rounded-t-xl" style={{ height: 140 }}>
          <img
            src={heroSrc}
            alt=""
            className="w-full h-full object-cover object-[center_35%]"
          />
          {/* Gradient fade to modal bg */}
          <div className="briefing-hero-fade absolute inset-x-0 bottom-0 h-20" />
          {/* Scan line */}
          <div
            className="animate-scan-sweep absolute inset-x-0 h-px"
            style={{ background: 'linear-gradient(90deg, transparent, rgba(251,191,36,0.18), transparent)' }}
          />
          {/* Header pinned to bottom of image */}
          <div className="absolute inset-x-0 bottom-0 px-5 pb-2.5 flex items-center gap-3">
            <div className="w-9 h-9 rounded-full bg-amber-500/20 flex items-center justify-center backdrop-blur-sm">
              <Terminal className="w-4 h-4 text-amber-400" />
            </div>
            <div>
              <h2
                className="text-base font-semibold text-amber-400 uppercase tracking-wider font-mono"
                style={{ textShadow: isDark
                  ? '0 1px 4px rgba(0,0,0,0.8)'
                  : '0 1px 4px rgba(255,255,255,0.9), 0 0 8px rgba(255,255,255,0.7)'
                }}
              >
                Mission Briefing
              </h2>
              <p
                className="text-xs text-amber-400/70 font-mono mt-0.5"
                style={{ textShadow: isDark
                  ? '0 1px 3px rgba(0,0,0,0.8)'
                  : '0 1px 3px rgba(255,255,255,0.9), 0 0 6px rgba(255,255,255,0.7)'
                }}
              >
                GSV Meridian — Compute Division
              </p>
            </div>
          </div>
        </div>

        {/* Body */}
        <div className="px-5 pb-5 pt-3 space-y-2.5">
          {paragraphs.map((text, i) => (
            <p key={i} className="text-sm text-gray-300 leading-relaxed font-mono">
              {text}
            </p>
          ))}

          {/* CTA */}
          <button
            onClick={dismissIntro}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-teal-600 hover:bg-teal-500 text-white rounded-lg font-medium font-mono transition-colors cursor-pointer !mt-5"
          >
            {introSeen ? 'Continue' : 'Begin'}
          </button>
        </div>
      </div>
    </ModalBackdrop>
  );
}
