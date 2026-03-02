/**
 * MissionHUD — in-mission overlay with briefing, objectives, and hints.
 * Sci-fi terminal aesthetic: amber (titles/nav/hints), teal (buttons/objectives), gray (body).
 * Supports multi-objective missions with objective tabs.
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
import { useTheme } from '../../hooks/useTheme.ts';

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
  const reviewMissionSuccess = useRPGStore(s => s.reviewMissionSuccess);
  const acknowledgeMissionSuccess = useRPGStore(s => s.acknowledgeMissionSuccess);
  const activeObjectiveId = useRPGStore(s => s.activeObjectiveId);
  const clearedObjectiveIds = useRPGStore(s => s.clearedObjectiveIds);
  const selectObjective = useRPGStore(s => s.selectObjective);

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
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const heroSrc = arc?.heroImage
    ? (isDark ? arc.heroImage.dark : arc.heroImage.light)
    : (isDark ? '/ship_dark.png' : '/ship_light.png');
  const totalHints = mission.hints.length;
  const hasMoreHints = hintsRevealed < totalHints;

  const isMultiObjective = mission.objectives && mission.objectives.length > 0;
  const allObjectivesCleared = isMultiObjective && mission.objectives!.every(o => clearedObjectiveIds.includes(o.id));

  // For multi-objective: passed means active objective cleared OR all objectives cleared
  // For single-objective: passed means all criteria met with correct approach
  const activeObjCleared = activeObjectiveId ? clearedObjectiveIds.includes(activeObjectiveId) : false;
  const passed = isMultiObjective
    ? (allObjectivesCleared || (lastValidation?.passed && approachValid) || activeObjCleared)
    : (lastValidation?.passed && approachValid);

  // Get criteria for display
  const displayCriteria = isMultiObjective && activeObjectiveId
    ? mission.objectives!.find(o => o.id === activeObjectiveId)?.winningCriteria ?? []
    : mission.winningCriteria;

  return (
    <div className={`relative bg-gray-900/50 border border-gray-800 rounded-xl mt-0 mb-4 backdrop-blur-sm${!hudExpanded ? ' border-l-2 border-l-amber-500/50' : ''}`}>
      {/* Chapter hero — atmospheric background behind breadcrumbs + title */}
      <div
        className="absolute inset-x-0 top-0 pointer-events-none overflow-hidden rounded-t-xl"
        style={{ height: 140 }}
      >
        <img
          src={heroSrc}
          alt=""
          className="w-full h-full object-cover object-[center_30%]"
          style={{
            opacity: 0.18,
            maskImage: 'linear-gradient(to bottom, black 30%, transparent 100%), linear-gradient(to right, transparent 2%, black 15%, black 85%, transparent 98%)',
            WebkitMaskImage: 'linear-gradient(to bottom, black 30%, transparent 100%), linear-gradient(to right, transparent 2%, black 15%, black 85%, transparent 98%)',
            maskComposite: 'intersect',
            WebkitMaskComposite: 'source-in',
          } as React.CSSProperties}
        />
      </div>

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
        style={{ maxHeight: hudExpanded ? '700px' : '0px', opacity: hudExpanded ? 1 : 0 }}
      >
        <div className="px-5 pb-4 overflow-hidden">
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
            {mission.briefing.trim().split('\n\n').map((block, i) => {
              const lines = block.trim().split('\n');
              const isList = lines.every(l => l.trimStart().startsWith('- '));
              if (isList) {
                return (
                  <ul key={i} className="space-y-1 ml-1 border-l-2 border-gray-700 pl-3">
                    {lines.map((line, j) => (
                      <li key={j} className="text-sm text-gray-300/80 leading-relaxed font-mono">
                        <GlossaryText text={line.trimStart().slice(2)} />
                      </li>
                    ))}
                  </ul>
                );
              }
              return (
                <p key={i} className="text-sm text-gray-300/80 leading-relaxed font-mono">
                  <GlossaryText text={block.trim()} />
                </p>
              );
            })}
          </div>

          <div className="border-t border-gray-800 mb-3" />

          {/* Objectives + Hints */}
          <div className="grid gap-4 sm:grid-cols-2">
            {/* Left column: tabs + objectives */}
            <div className="min-w-0">
              {/* Multi-objective: Objective tabs */}
              {isMultiObjective && (
                <div className="grid grid-cols-2 gap-2 mb-3">
                  {mission.objectives!.map(obj => {
                    const isCleared = clearedObjectiveIds.includes(obj.id);
                    const isActive = activeObjectiveId === obj.id;
                    return (
                      <button
                        key={obj.id}
                        onClick={() => selectObjective(obj.id)}
                        className={`text-xs px-3 py-1.5 rounded-lg font-mono border transition-colors cursor-pointer flex items-start gap-1.5 min-w-0 text-left ${
                          isCleared
                            ? 'border-teal-500/40 text-teal-400 bg-teal-500/10'
                            : isActive
                              ? 'border-amber-500/50 text-amber-300 bg-amber-500/10'
                              : 'border-gray-700 text-gray-400 hover:border-gray-600 hover:text-gray-300'
                        }`}
                      >
                        {isCleared && <Check className="w-3 h-3 shrink-0" />}
                        <span>{obj.label}</span>
                      </button>
                    );
                  })}
                </div>
              )}
              {(!isMultiObjective || activeObjectiveId) && (
                <>
                  <div className="text-sm text-gray-400 mb-1.5 font-mono">Objectives</div>
                  <div className="font-mono">
                    <CriteriaChecklist criteria={displayCriteria} validation={lastValidation} cleared={activeObjCleared && !lastValidation} />
                  </div>
                  {lastValidation?.passed && !approachValid && (
                    <div className="text-sm text-amber-200/90 bg-amber-900/20 border border-amber-700/30 rounded px-2.5 py-1.5 mt-2 font-mono">
                      Target met, but not the way the briefing intended. Try a different approach.
                    </div>
                  )}
                  {passed && (
                    <div className="flex items-center gap-2 mt-3 -ml-1">
                      <button
                        onClick={isMultiObjective && !allObjectivesCleared ? acknowledgeMissionSuccess : reviewMissionSuccess}
                        className="text-sm px-3 py-1 bg-teal-600 hover:bg-teal-500 text-white rounded cursor-pointer transition-colors font-mono"
                      >
                        {isMultiObjective && !allObjectivesCleared ? 'Complete Objective' : 'Complete Mission'}
                      </button>
                    </div>
                  )}
                </>
              )}
            </div>

            {/* Hints */}
            <div className="min-w-0">
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
                <div className="text-sm text-amber-200/80 bg-amber-900/20 border border-amber-700/30 rounded px-2.5 py-1.5 font-mono break-words">
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
