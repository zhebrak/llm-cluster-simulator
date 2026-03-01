/**
 * MissionHUD — in-mission overlay with briefing, objectives, and hints.
 * Sci-fi terminal aesthetic: amber (titles/nav/hints), teal (buttons/objectives), gray (body).
 */

import { useState, useEffect } from 'react';
import { ChevronsUp, ChevronsDown, Lightbulb, X, Check } from 'lucide-react';
import { useRPGStore } from '../../stores/rpg.ts';
import { getMissionById } from '../../rpg/missions/index.ts';
import { ALL_ARCS } from '../../rpg/missions/index.ts';
import { CriteriaChecklist } from '../game/CriteriaChecklist.tsx';
import { GlossaryText } from '../game/GlossaryText.tsx';
import { Tooltip } from '../ui/Tooltip.tsx';
import { ConfirmResetButton } from '../ui/ConfirmResetButton.tsx';

export function MissionHUD() {
  const activeMissionId = useRPGStore(s => s.activeMissionId);
  const hintsRevealed = useRPGStore(s => s.hintsRevealed);
  const attempts = useRPGStore(s => s.attempts);
  const lastValidation = useRPGStore(s => s.lastValidation);
  const approachValid = useRPGStore(s => s.approachValid);
  const revealNextHint = useRPGStore(s => s.revealNextHint);
  const resetMission = useRPGStore(s => s.resetMission);
  const openMenu = useRPGStore(s => s.openMenu);
  const exit = useRPGStore(s => s.exit);
  const acknowledgeMissionSuccess = useRPGStore(s => s.acknowledgeMissionSuccess);
  const reviewMissionSuccess = useRPGStore(s => s.reviewMissionSuccess);

  const [hudExpanded, setHudExpanded] = useState(true);
  const [activeHintIndex, setActiveHintIndex] = useState(0);

  // Reset to expanded when mission changes
  useEffect(() => {
    setHudExpanded(true);
    setActiveHintIndex(0);
  }, [activeMissionId]);

  // Auto-advance to newest hint when revealed
  useEffect(() => {
    if (hintsRevealed > 0) setActiveHintIndex(hintsRevealed - 1);
  }, [hintsRevealed]);

  if (!activeMissionId) return null;
  const mission = getMissionById(activeMissionId);
  if (!mission) return null;

  const arc = ALL_ARCS.find(a => a.id === mission.arcId);
  const totalHints = mission.hints.length;
  const hasMoreHints = hintsRevealed < totalHints;
  const passed = lastValidation?.passed && approachValid;

  return (
    <div className={`bg-gray-900/50 border border-gray-800 rounded-xl mt-0 mb-4 backdrop-blur-sm${!hudExpanded ? ' border-l-2 border-l-amber-500/50' : ''}`}>
      {/* Top bar — always visible */}
      <div className="px-5 py-3 flex items-center gap-3">
        {/* Terminal breadcrumb — clickable to open mission select */}
        <div className="text-sm text-gray-500 font-mono truncate">
          <button onClick={openMenu} className="text-amber-400 hover:text-amber-300 cursor-pointer transition-colors font-semibold">[{arc?.name?.toUpperCase() ?? 'ARC'}]</button>
          <span className="mx-1.5 text-amber-400">&gt;</span>
          <button onClick={openMenu} className="text-amber-400 hover:text-amber-300 cursor-pointer transition-colors font-semibold">[{mission.title.toUpperCase()}]</button>
        </div>

        {/* Passed badge (collapsed only) */}
        {!hudExpanded && passed && (
          <span className="flex items-center gap-1 text-xs text-teal-400 flex-shrink-0">
            <Check className="w-3.5 h-3.5" />
          </span>
        )}

        <div className="flex-1" />

        {/* Reset to mission defaults */}
        <ConfirmResetButton onConfirm={resetMission} tooltip="Reset to mission defaults" />

        {/* Expand/collapse */}
        <Tooltip text={hudExpanded ? 'Collapse' : 'Expand'}>
          <button
            onClick={() => setHudExpanded(!hudExpanded)}
            className="text-gray-400 hover:text-gray-300 cursor-pointer p-1 flex-shrink-0"
          >
            {hudExpanded ? <ChevronsUp className="w-4 h-4" /> : <ChevronsDown className="w-4 h-4" />}
          </button>
        </Tooltip>

        {/* Exit RPG mode */}
        <Tooltip text="Exit RPG mode">
          <button
            onClick={exit}
            className="text-gray-400 hover:text-gray-300 cursor-pointer p-1 flex-shrink-0"
          >
            <X className="w-4 h-4" />
          </button>
        </Tooltip>
      </div>

      {/* Expanded content */}
      <div
        className="transition-all duration-200 ease-in-out overflow-hidden"
        style={{ maxHeight: hudExpanded ? '600px' : '0px', opacity: hudExpanded ? 1 : 0 }}
      >
        <div className="px-5 pb-4">
          <div className="border-t border-gray-800 mb-3" />

          {/* Title + attempts */}
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-base font-semibold text-amber-400 font-mono">{mission.title}</h3>
            {attempts > 0 && (
              <span className="text-sm text-gray-500 font-mono">
                {attempts} {attempts === 1 ? 'attempt' : 'attempts'}
              </span>
            )}
          </div>

          {/* Briefing */}
          <div className="space-y-2 mb-3">
            {mission.briefing.trim().split('\n\n').map((para, i) => (
              <p key={i} className="text-sm text-gray-300/80 leading-relaxed font-mono">
                <GlossaryText text={para.trim()} />
              </p>
            ))}
          </div>

          <div className="border-t border-gray-800 mb-3" />

          {/* Objectives + Hints */}
          <div className="flex flex-col sm:flex-row gap-4">
            {/* Objectives */}
            <div className="flex-1">
              <div className="text-sm text-gray-400 mb-1.5 font-mono">Objectives</div>
              <div className="font-mono">
                <CriteriaChecklist criteria={mission.winningCriteria} validation={lastValidation} />
              </div>
              {lastValidation?.passed && !approachValid && (
                <div className="text-sm text-amber-200/90 bg-amber-900/20 border border-amber-700/30 rounded px-2.5 py-1.5 mt-2 font-mono">
                  Target met, but not by the expected method. Re-read the briefing and try a different approach.
                </div>
              )}
              {passed && (
                <div className="flex items-center gap-2 mt-3 -ml-1">
                  <button
                    onClick={reviewMissionSuccess}
                    className="text-sm text-gray-400 hover:text-white cursor-pointer transition-colors px-1 py-1 rounded font-mono"
                  >
                    Review
                  </button>
                  <button
                    onClick={acknowledgeMissionSuccess}
                    className="text-sm px-3 py-1 bg-teal-600 hover:bg-teal-500 text-white rounded cursor-pointer transition-colors font-mono"
                  >
                    Complete Mission
                  </button>
                </div>
              )}
            </div>

            {/* Hints */}
            <div className="flex-1">
              <div className="flex items-center gap-1.5 mb-1.5">
                <span className="text-sm text-gray-400 font-mono">Hints</span>
                {hintsRevealed > 1 && mission.hints.slice(0, hintsRevealed).map((_, i) => (
                  <button
                    key={i}
                    onClick={() => setActiveHintIndex(i)}
                    className={`w-5 h-5 rounded-full text-xs cursor-pointer transition-colors font-mono ${
                      i === activeHintIndex
                        ? 'bg-amber-400/30 text-amber-300'
                        : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
                    }`}
                  >
                    {i + 1}
                  </button>
                ))}
                {hasMoreHints && (
                  <button
                    onClick={revealNextHint}
                    className="text-sm text-amber-400/70 hover:text-amber-300 cursor-pointer flex items-center gap-1 ml-auto font-mono"
                  >
                    <Lightbulb className="w-3.5 h-3.5" />
                    Show hint
                  </button>
                )}
              </div>

              {hintsRevealed > 0 && (
                <div className="text-sm text-amber-200/80 bg-amber-900/20 border border-amber-700/30 rounded px-2.5 py-1.5 font-mono">
                  <Lightbulb className="w-3.5 h-3.5 inline mr-1 text-amber-400" />
                  <GlossaryText text={mission.hints[activeHintIndex].trim()} />
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
