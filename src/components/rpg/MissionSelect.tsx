/**
 * MissionSelect — arc-based mission picker with sci-fi terminal aesthetic.
 * Includes hardware inventory ("Compute Bay"), player title, mission subtitles,
 * pivot styling, hover skills tooltip, and briefing re-read button.
 */

import { useEffect, useMemo, useState, useRef, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { X, Award, Sparkles, Cpu, ChevronDown } from 'lucide-react';
import { useRPGStore } from '../../stores/rpg.ts';
import { useTheme } from '../../hooks/useTheme.ts';
import { ALL_ARCS, getActiveArc } from '../../rpg/missions/index.ts';
import { getMissionsForArc, isMissionUnlocked, getClosestUnmetPrereq } from '../../rpg/missions/index.ts';
import { ALL_SKILLS, getEarnedStars, getStarCounts, getPlayerTitle } from '../../rpg/skills.ts';
import { getAvailableTierIds, HARDWARE_PROGRESSION } from '../../rpg/hardware.ts';
import { ModalBackdrop } from '../ui/ModalBackdrop.tsx';
import { Tooltip } from '../ui/Tooltip.tsx';

/** Hover-triggered tooltip showing all skills with star progression. */
function SkillsTooltip({ completedMissions }: { completedMissions: string[] }) {
  const [show, setShow] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleEnter = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setShow(true), 120);
  }, []);

  const handleLeave = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setShow(false), 150);
  }, []);

  useEffect(() => {
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, []);

  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);

  useEffect(() => {
    if (!show || !containerRef.current) { setPos(null); return; }
    const r = containerRef.current.getBoundingClientRect();
    const pad = 8;
    const tipWidth = 272;
    let x = r.left + r.width / 2 - tipWidth / 2;
    x = Math.min(x, window.innerWidth - tipWidth - pad);
    x = Math.max(pad, x);
    setPos({ x, y: r.bottom + 4 });
  }, [show]);

  const allSkillIds = Object.keys(ALL_SKILLS);
  const { earned } = getStarCounts(completedMissions);

  return (
    <div
      ref={containerRef}
      className="relative inline-block"
      onMouseEnter={handleEnter}
      onMouseLeave={handleLeave}
    >
      <div
        className={`flex items-center gap-1.5 px-2 py-1 text-xs font-mono rounded border cursor-help transition-colors ${
          earned > 0
            ? 'text-sky-400/70 border-sky-500/30 hover:border-sky-400/40 hover:text-sky-300'
            : 'text-gray-600 border-gray-700 hover:border-gray-600 hover:text-gray-500'
        }`}
      >
        <Award className="w-3.5 h-3.5" />
        <span>Level {earned}</span>
      </div>

      {show && earned > 0 && createPortal(
        <div
          className="fixed z-[60] bg-gray-950 border border-gray-700 rounded-lg shadow-xl p-2.5 w-[272px]"
          style={{ left: pos?.x ?? -9999, top: pos?.y ?? -9999, opacity: pos ? 1 : 0 }}
          onMouseEnter={handleEnter}
          onMouseLeave={handleLeave}
        >
          <div className="space-y-1">
            {[...allSkillIds]
              .map(id => ({ id, stars: getEarnedStars(id, completedMissions) }))
              .filter(s => s.stars > 0)
              .sort((a, b) => b.stars - a.stars)
              .map(({ id, stars }) => (
                <div
                  key={id}
                  className="flex items-center gap-2 px-2 py-1 text-xs font-mono rounded text-sky-300"
                >
                  <span className="text-amber-400/80 text-base w-16 shrink-0 tracking-wider">{'\u2605'.repeat(stars)}</span>
                  <span>{ALL_SKILLS[id].name}</span>
                </div>
              ))}
          </div>
        </div>,
        document.body,
      )}
    </div>
  );
}

const CIPHER_CHARS = '0123456789abcdef#$@%&xX./:;!?+=~^';
const CIPHER_LENGTH = 18;

function randomCipherChar() {
  return CIPHER_CHARS[Math.floor(Math.random() * CIPHER_CHARS.length)];
}

/** Garbled hex display that slowly shuffles characters, simulating decryption. */
function EncryptedSlot() {
  const [chars, setChars] = useState(() =>
    Array.from({ length: CIPHER_LENGTH }, randomCipherChar),
  );

  useEffect(() => {
    const id = setInterval(() => {
      setChars(prev => {
        const next = [...prev];
        const swapCount = 3 + Math.floor(Math.random() * 3); // 3-5 chars
        for (let i = 0; i < swapCount; i++) {
          next[Math.floor(Math.random() * CIPHER_LENGTH)] = randomCipherChar();
        }
        return next;
      });
    }, 800);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="px-3 py-2 text-sm font-mono text-gray-500/40">
      <div className="flex items-start gap-3">
        <span className="w-16 shrink-0 text-right flex justify-end items-center">
          <span className="font-semibold">[UNKNOWN]</span>
        </span>
        <span className="tracking-widest text-gray-500 animate-cipher-pulse">{chars.join('')}</span>
      </div>
    </div>
  );
}

/** Hover-triggered tooltip showing available GPUs in the compute bay. */
function ComputeBayTooltip({ completedMissions, newTierIds }: { completedMissions: string[]; newTierIds: string[] }) {
  const [show, setShow] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleEnter = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setShow(true), 120);
  }, []);

  const handleLeave = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setShow(false), 150);
  }, []);

  useEffect(() => {
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, []);

  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);

  const tipRefCb = useCallback((node: HTMLDivElement | null) => {
    if (!node || !containerRef.current) { setPos(null); return; }
    const r = containerRef.current.getBoundingClientRect();
    const pad = 8;
    const tipWidth = node.offsetWidth;
    let x = r.left + r.width / 2 - tipWidth / 2;
    x = Math.min(x, window.innerWidth - tipWidth - pad);
    x = Math.max(pad, x);
    setPos({ x, y: r.bottom + 4 });
  }, []);

  const hasNew = newTierIds.length > 0;

  return (
    <div
      ref={containerRef}
      className="relative inline-block"
      onMouseEnter={handleEnter}
      onMouseLeave={handleLeave}
    >
      <div
        className={`flex items-center gap-1.5 px-2 py-1 text-xs font-mono rounded border cursor-help transition-colors ${
          hasNew
            ? 'text-sky-400/70 border-sky-500/30 hover:border-sky-400/40 hover:text-sky-300'
            : 'text-gray-600 border-gray-700 hover:border-gray-600 hover:text-gray-500'
        }`}
      >
        <Cpu className="w-3.5 h-3.5" />
        <span>Compute Bay</span>
        {hasNew && <span className="text-amber-400 font-semibold">NEW</span>}
      </div>

      {show && createPortal(
        <div
          ref={tipRefCb}
          className="fixed z-[60] bg-gray-950 border border-gray-700 rounded-lg shadow-xl p-2.5 w-fit max-w-[90vw]"
          style={{ left: pos?.x ?? -9999, top: pos?.y ?? -9999, opacity: pos ? 1 : 0 }}
          onMouseEnter={handleEnter}
          onMouseLeave={handleLeave}
        >
          <div className="grid grid-cols-2 gap-1.5">
            {(() => {
              const aggregated = new Map<string, { label: string; count: number }>();
              for (const tier of HARDWARE_PROGRESSION) {
                const isActive = tier.unlockedBy === null || completedMissions.includes(tier.unlockedBy);
                if (!isActive) continue;
                for (const slot of tier.gpus) {
                  const existing = aggregated.get(slot.gpuId);
                  if (existing) {
                    existing.count += slot.count;
                  } else {
                    aggregated.set(slot.gpuId, { label: slot.label, count: slot.count });
                  }
                }
              }
              return Array.from(aggregated.entries()).map(([gpuId, { label, count }]) => (
                <span
                  key={gpuId}
                  className="inline-flex items-center gap-1 px-2 py-1 text-xs font-mono rounded border bg-sky-500/15 text-sky-300 border-sky-500/30"
                >
                  {label} &times;{count}
                </span>
              ));
            })()}
          </div>
        </div>,
        document.body,
      )}
    </div>
  );
}

export function MissionSelect() {
  const completedMissions = useRPGStore(s => s.completedMissions);
  const startMission = useRPGStore(s => s.startMission);
  const menuOpen = useRPGStore(s => s.menuOpen);
  const closeMenu = useRPGStore(s => s.closeMenu);
  const dismissMissionSelect = useRPGStore(s => s.dismissMissionSelect);
  const seenHardwareTierIds = useRPGStore(s => s.seenHardwareTierIds);
  const markHardwareSeen = useRPGStore(s => s.markHardwareSeen);
  const showBriefing = useRPGStore(s => s.showBriefing);
  const resetProgress = useRPGStore(s => s.resetProgress);

  const baseClose = menuOpen ? closeMenu : dismissMissionSelect;
  const handleClose = useCallback(() => { setConfirmingResetArcId(null); baseClose(); }, [baseClose]);

  // Hardware inventory
  const availableTierIds = useMemo(
    () => getAvailableTierIds(completedMissions),
    [completedMissions],
  );
  const newTierIds = useMemo(
    () => availableTierIds.filter(id => !seenHardwareTierIds.includes(id)),
    [availableTierIds, seenHardwareTierIds],
  );

  // Mark hardware as seen on render
  useEffect(() => {
    if (newTierIds.length > 0) {
      markHardwareSeen(availableTierIds);
    }
  }, [availableTierIds, newTierIds.length, markHardwareSeen]);

  const playerTitle = getPlayerTitle(completedMissions);

  const [confirmingResetArcId, setConfirmingResetArcId] = useState<string | null>(null);

  // Collapse arcs where all gameplay missions are completed
  const [collapsedArcs, setCollapsedArcs] = useState<Set<string>>(() => {
    const set = new Set<string>();
    for (const arc of ALL_ARCS) {
      const gameplay = getMissionsForArc(arc.id).filter(m => m.type !== 'pivot');
      if (gameplay.length > 0 && gameplay.every(m => completedMissions.includes(m.id))) {
        set.add(arc.id);
      }
    }
    return set;
  });

  const toggleArc = useCallback((arcId: string) => {
    setCollapsedArcs(prev => {
      const next = new Set(prev);
      if (next.has(arcId)) next.delete(arcId);
      else next.add(arcId);
      return next;
    });
  }, []);

  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const activeArc = getActiveArc(completedMissions);
  const heroSrc = activeArc.heroImage
    ? (isDark ? activeArc.heroImage.dark : activeArc.heroImage.light)
    : (isDark ? '/ship_dark.png' : '/ship_light.png');

  return (
    <ModalBackdrop>
      <div
        className="bg-gray-950 border border-gray-800 rounded-xl max-w-lg w-full mx-4 max-h-[80vh] overflow-hidden flex flex-col"
        onClick={e => e.stopPropagation()}
      >
        {/* Hero image with title overlaid */}
        <div className="relative overflow-hidden rounded-t-xl shrink-0" style={{ height: 120 }}>
          <img
            src={heroSrc}
            alt=""
            className="w-full h-full object-cover object-[center_35%]"
          />
          <div className="briefing-hero-fade absolute inset-x-0 bottom-0 h-20" />
          <div
            className="animate-scan-sweep absolute inset-x-0 h-px"
            style={{ background: 'linear-gradient(90deg, transparent, rgba(251,191,36,0.18), transparent)' }}
          />
          {/* Title pinned to bottom of image */}
          <h2
            className="absolute left-8 bottom-2 text-lg font-semibold text-amber-400 uppercase tracking-wider font-mono"
            style={{ textShadow: isDark
              ? '0 1px 4px rgba(0,0,0,0.8)'
              : '0 1px 4px rgba(255,255,255,0.9), 0 0 8px rgba(255,255,255,0.7)'
            }}
          >
            Mission Log
          </h2>
          {/* Close button in top-right corner */}
          <Tooltip text={menuOpen ? 'Back to mission' : 'Exit game mode'} className="absolute top-2 right-2">
            <button
              onClick={handleClose}
              className="cursor-pointer p-1 rounded-full bg-black/30 hover:bg-black/50 backdrop-blur-sm transition-colors"
              style={{ color: 'rgba(255,255,255,0.7)' }}
              onMouseEnter={e => (e.currentTarget.style.color = 'rgba(255,255,255,0.95)')}
              onMouseLeave={e => (e.currentTarget.style.color = 'rgba(255,255,255,0.7)')}
            >
              <X className="w-4 h-4" />
            </button>
          </Tooltip>
        </div>

        {/* Sub-header */}
        <div className="pl-8 pr-5 pt-3 pb-1 shrink-0">
          <div className="flex items-center justify-between">
            <p className="text-xs text-amber-400/70 font-mono">
              Generation Ship Meridian
            </p>
            <button
              onClick={() => showBriefing()}
              className="text-xs text-gray-600 hover:text-amber-400 font-mono cursor-pointer transition-colors"
            >
              [BRIEFING]
            </button>
          </div>
          <div className="flex items-center gap-2 mt-2.5">
            {playerTitle && (
              <span className="text-xs text-sky-400/70 font-mono">
                {playerTitle}
              </span>
            )}
            <SkillsTooltip completedMissions={completedMissions} />
            <ComputeBayTooltip completedMissions={completedMissions} newTierIds={newTierIds} />
          </div>
        </div>

        {/* Arcs — scrollable */}
        <div className="overflow-y-auto pl-8 pr-5 pb-5 pt-2 space-y-6">
          {ALL_ARCS.map(arc => {
            const missions = getMissionsForArc(arc.id);

            // Hide arcs with no reachable missions (all locked)
            const hasReachableMission = missions.some(
              m => completedMissions.includes(m.id) || isMissionUnlocked(m, completedMissions)
            );
            if (!hasReachableMission) return null;

            const gameplayMissions = missions.filter(m => m.type !== 'pivot');
            const completedCount = gameplayMissions.filter(m => completedMissions.includes(m.id)).length;

            const isCollapsed = collapsedArcs.has(arc.id);

            return (
              <div key={arc.id}>
                {/* Arc header — clickable to toggle */}
                <button
                  onClick={() => toggleArc(arc.id)}
                  className="w-full text-left border-l-2 border-amber-500/40 pl-3 mb-3 cursor-pointer group"
                >
                  <div className="flex items-center gap-2">
                    <ChevronDown className={`w-3.5 h-3.5 text-amber-400/60 transition-transform ${isCollapsed ? '-rotate-90' : ''}`} />
                    <h3 className="text-sm font-semibold text-amber-400 uppercase tracking-wider font-mono">
                      {arc.name}
                    </h3>
                    <span className="text-xs text-gray-600 font-mono">
                      [{completedCount}/{gameplayMissions.length}]
                    </span>
                    {completedCount > 0 && (
                      confirmingResetArcId === arc.id ? (
                        <span onClick={e => e.stopPropagation()} className="text-xs font-mono text-gray-500">
                          Reset?{' '}
                          <span
                            onClick={() => { resetProgress(arc.id); setConfirmingResetArcId(null); }}
                            className="text-red-400 hover:text-red-300 cursor-pointer transition-colors"
                          >Yes</span>
                          {' / '}
                          <span
                            onClick={() => setConfirmingResetArcId(null)}
                            className="text-gray-400 hover:text-gray-300 cursor-pointer transition-colors"
                          >No</span>
                        </span>
                      ) : (
                        <span
                          onClick={e => { e.stopPropagation(); setConfirmingResetArcId(arc.id); }}
                          className="text-xs font-mono italic text-gray-700 hover:text-gray-500 transition-colors"
                        >
                          reset
                        </span>
                      )
                    )}
                  </div>
                  <p className="text-xs text-gray-500 font-mono mt-0.5">{arc.subtitle}</p>
                </button>

                {/* Missions — collapsible */}
                {!isCollapsed && <div className="space-y-1">
                  {missions.map(mission => {
                    const isCompleted = completedMissions.includes(mission.id);
                    const isUnlocked = isMissionUnlocked(mission, completedMissions);
                    const isPivot = mission.type === 'pivot';

                    {/* Locked pivot: encrypted slot */}
                    if (!isUnlocked && !isCompleted && isPivot) {
                      return <EncryptedSlot key={mission.id} />;
                    }

                    {/* Locked regular: title + prerequisite hint */}
                    if (!isUnlocked && !isCompleted) {
                      const unmetPrereq = getClosestUnmetPrereq(mission, completedMissions);
                      return (
                        <div key={mission.id} className="px-3 py-2 text-sm font-mono text-gray-500/40">
                          <div className="flex items-start gap-3">
                            <span className="w-16 shrink-0 text-right flex justify-end items-center">
                              <span className="font-semibold">[LOCKED]</span>
                            </span>
                            <div className="min-w-0">
                              <span className="font-semibold">{mission.title}</span>
                              {unmetPrereq && (
                                <p className="text-xs mt-0.5 text-gray-500">
                                  Requires {unmetPrereq.title}
                                </p>
                              )}
                            </div>
                          </div>
                        </div>
                      );
                    }

                    {/* Unlocked / completed */}
                    const statusEl = isCompleted ? (
                      <span className={`font-semibold ${isPivot ? 'text-amber-400/50' : 'text-teal-400/70'}`}>[DONE]</span>
                    ) : isPivot ? (
                      <Sparkles className="w-4 h-4 text-amber-400" />
                    ) : (
                      <span className="text-teal-400">&gt;</span>
                    );

                    const content = (
                      <div className="flex items-start gap-3">
                        <span className="w-16 shrink-0 text-right flex justify-end items-center">{statusEl}</span>
                        <div className="min-w-0">
                          <span className="font-semibold">
                            {isPivot && !isCompleted && <span className="text-amber-500/70 mr-1">[EVENT]</span>}
                            {mission.title}
                          </span>
                          {mission.subtitle && (
                            <p className={`text-xs mt-0.5 ${
                              isPivot ? 'text-amber-400/40' : 'text-gray-500'
                            }`}>
                              {mission.subtitle}
                            </p>
                          )}
                        </div>
                      </div>
                    );

                    return (
                      <button
                        key={mission.id}
                        onClick={() => startMission(mission.id)}
                        className={`w-full text-left px-3 py-2 text-sm font-mono rounded transition-all cursor-pointer ${
                          isCompleted
                            ? isPivot
                              ? 'text-amber-400/50 hover:text-amber-300/70 hover:bg-gray-700/30'
                              : 'text-teal-400/70 hover:text-teal-300 hover:bg-gray-700/30'
                            : isPivot
                              ? 'text-amber-300 hover:text-amber-200 hover:bg-gray-700/30'
                              : 'text-teal-300 hover:text-teal-200 hover:bg-gray-700/30'
                        }`}
                      >
                        {content}
                      </button>
                    );
                  })}
                </div>}
              </div>
            );
          })}
        </div>

      </div>
    </ModalBackdrop>
  );
}
