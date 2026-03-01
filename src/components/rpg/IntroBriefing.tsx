/**
 * IntroBriefing — one-time intro shown on first RPG entry.
 * Re-accessible via "View Briefing" button in MissionSelect.
 */

import { Terminal } from 'lucide-react';
import { useRPGStore } from '../../stores/rpg.ts';

export function IntroBriefing() {
  const dismissIntro = useRPGStore(s => s.dismissIntro);
  const introSeen = useRPGStore(s => s.introSeen);

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50">
      <div
        className="bg-gray-950 border border-amber-500/30 rounded-xl p-6 max-w-lg w-full mx-4 max-h-[80vh] overflow-y-auto"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center gap-3 mb-5">
          <div className="w-10 h-10 rounded-full bg-amber-500/20 flex items-center justify-center">
            <Terminal className="w-5 h-5 text-amber-400" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-amber-400 uppercase tracking-wider font-mono">
              Mission Briefing
            </h2>
            <p className="text-xs text-gray-500 font-mono mt-0.5">
              GSV Meridian — Compute Division
            </p>
          </div>
        </div>

        {/* Body */}
        <div className="space-y-3 mb-6">
          <p className="text-sm text-gray-300 leading-relaxed font-mono">
            You are the Compute Officer aboard the GSV <span className="text-sky-300">Meridian</span>, a generation ship 80 years into its transit toward Kepler-442b. The ship carries 200 colonists in cryogenic stasis and a skeleton crew rotating through watch cycles.
          </p>
          <p className="text-sm text-gray-300 leading-relaxed font-mono">
            Every critical system on board — navigation, life support monitoring, long-range sensors, crew AI — runs on GPU-accelerated ML models. Your job is to keep them running on hardware that was state-of-the-art at launch — 80 years ago.
          </p>
          <p className="text-sm text-gray-300 leading-relaxed font-mono">
            Systems are failing. Power budgets are tight. The compute bay holds a handful of aging GPUs. Every configuration decision you make has consequences — an OOM crash can blind the sensors, and wasted GPU-hours drain reserves that keep the crew alive.
          </p>
          <p className="text-sm text-gray-400 leading-relaxed font-mono italic">
            Each mission presents a failing system. Diagnose the problem, configure the simulator to solve it, and bring the system back online. Hints are available if you get stuck.
          </p>
        </div>

        {/* CTA */}
        <button
          onClick={dismissIntro}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-amber-600 hover:bg-amber-500 text-white rounded-lg font-medium font-mono transition-colors cursor-pointer"
        >
          {introSeen ? 'Continue' : 'Begin'}
        </button>
      </div>
    </div>
  );
}
