/**
 * MissionSelect — arc-based mission picker with sci-fi terminal aesthetic.
 * Includes hardware inventory ("Compute Bay"), player title, mission subtitles,
 * pivot styling, hover skills tooltip, and briefing re-read button.
 */

import { useEffect, useMemo, useState, useRef, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { X, Award, Sparkles, Cpu } from 'lucide-react';
import { useRPGStore } from '../../stores/rpg.ts';
import { ALL_ARCS } from '../../rpg/missions/index.ts';
import { getMissionsForArc, isMissionUnlocked, getMissionById } from '../../rpg/missions/index.ts';
import { ALL_SKILLS } from '../../rpg/skills.ts';
import { getAvailableTierIds, HARDWARE_PROGRESSION } from '../../rpg/hardware.ts';

function getPlayerTitle(completedCount: number, totalCount: number): string | null {
  if (completedCount >= totalCount && totalCount > 0) return 'Systems Specialist';
  if (completedCount >= 4) return 'Junior Compute Officer';
  if (completedCount >= 1) return 'Apprentice Compute Officer';
  return null;
}

/** Hover-triggered tooltip showing all skills (earned active, unearned dimmed). */
function SkillsTooltip({ earnedSkills }: { earnedSkills: string[] }) {
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
    const tipWidth = 272; // w-68 = 17rem = 272px
    let x = r.left;
    if (x + tipWidth > window.innerWidth - pad) {
      x = window.innerWidth - tipWidth - pad;
    }
    x = Math.max(pad, x);
    setPos({ x, y: r.bottom + 4 });
  }, [show]);

  const allSkillIds = Object.keys(ALL_SKILLS);
  const earnedSet = new Set(earnedSkills);
  const earnedCount = earnedSkills.length;

  return (
    <div
      ref={containerRef}
      className="relative inline-block"
      onMouseEnter={handleEnter}
      onMouseLeave={handleLeave}
    >
      <div
        className={`flex items-center gap-1.5 px-2 py-1 text-xs font-mono rounded border cursor-help transition-colors ${
          earnedCount > 0
            ? 'text-sky-400/70 border-sky-500/30 hover:border-sky-400/40 hover:text-sky-300'
            : 'text-gray-600 border-gray-700 hover:border-gray-600 hover:text-gray-500'
        }`}
      >
        <Award className="w-3.5 h-3.5" />
        <span>Skills {earnedCount}/{allSkillIds.length}</span>
      </div>

      {show && createPortal(
        <div
          className="fixed z-[60] bg-gray-950 border border-gray-700 rounded-lg shadow-xl p-2.5 w-[17rem]"
          style={{ left: pos?.x ?? -9999, top: pos?.y ?? -9999, opacity: pos ? 1 : 0 }}
          onMouseEnter={handleEnter}
          onMouseLeave={handleLeave}
        >
          <div className="flex flex-wrap gap-1.5">
            {allSkillIds.map(id => {
              const skill = ALL_SKILLS[id];
              const earned = earnedSet.has(id);
              return (
                <span
                  key={id}
                  className={`inline-flex items-center gap-1 px-2 py-1 text-xs font-mono rounded border ${
                    earned
                      ? 'bg-sky-500/15 text-sky-300 border-sky-500/30'
                      : 'text-gray-500 border-gray-700/50 opacity-40'
                  }`}
                >
                  {skill.name}
                </span>
              );
            })}
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
        const swapCount = 2 + Math.floor(Math.random() * 2); // 2-3 chars
        for (let i = 0; i < swapCount; i++) {
          next[Math.floor(Math.random() * CIPHER_LENGTH)] = randomCipherChar();
        }
        return next;
      });
    }, 3000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="px-3 py-2 text-sm font-mono text-gray-500/40">
      <div className="flex items-start gap-3">
        <span className="w-16 shrink-0 text-right flex justify-end items-center">
          <span className="font-semibold">[ENCRYPTED]</span>
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

  useEffect(() => {
    if (!show || !containerRef.current) { setPos(null); return; }
    const r = containerRef.current.getBoundingClientRect();
    const pad = 8;
    const tipWidth = 240;
    let x = r.left;
    if (x + tipWidth > window.innerWidth - pad) {
      x = window.innerWidth - tipWidth - pad;
    }
    x = Math.max(pad, x);
    setPos({ x, y: r.bottom + 4 });
  }, [show]);

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
          className="fixed z-[60] bg-gray-950 border border-gray-700 rounded-lg shadow-xl p-2.5 w-64"
          style={{ left: pos?.x ?? -9999, top: pos?.y ?? -9999, opacity: pos ? 1 : 0 }}
          onMouseEnter={handleEnter}
          onMouseLeave={handleLeave}
        >
          <div className="flex flex-wrap gap-1.5">
            {HARDWARE_PROGRESSION.flatMap(tier => {
              const isActive = tier.unlockedBy === null || completedMissions.includes(tier.unlockedBy);
              if (!isActive) return [];
              return tier.gpus.map(slot => (
                <span
                  key={slot.gpuId}
                  className="inline-flex items-center gap-1 px-2 py-1 text-xs font-mono rounded border bg-sky-500/15 text-sky-300 border-sky-500/30"
                >
                  {slot.label} &times;{slot.count}
                </span>
              ));
            })}
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
  const earnedSkills = useRPGStore(s => s.earnedSkills);
  const seenHardwareTierIds = useRPGStore(s => s.seenHardwareTierIds);
  const markHardwareSeen = useRPGStore(s => s.markHardwareSeen);
  const showBriefing = useRPGStore(s => s.showBriefing);
  const resetProgress = useRPGStore(s => s.resetProgress);

  const handleClose = menuOpen ? closeMenu : dismissMissionSelect;

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

  // Total mission count (for player title)
  const allMissions = useMemo(
    () => ALL_ARCS.flatMap(arc => getMissionsForArc(arc.id)),
    [],
  );
  const playerTitle = getPlayerTitle(completedMissions.length, allMissions.length);

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div
        className="bg-gray-950 border border-gray-800 rounded-xl p-6 max-w-lg w-full mx-4 max-h-[80vh] overflow-y-auto"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="mb-4">
          <div className="flex items-start justify-between">
            <div className="flex items-baseline gap-3">
              <h2 className="text-lg font-semibold text-amber-400 uppercase tracking-wider font-mono">
                Mission Log
              </h2>
              <button
                onClick={() => showBriefing()}
                className="text-xs text-gray-600 hover:text-amber-400 font-mono cursor-pointer transition-colors"
              >
                [BRIEFING]
              </button>
            </div>
            <button
              onClick={handleClose}
              className="text-gray-500 hover:text-gray-300 cursor-pointer p-1 -mt-1 -mr-1"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
          <p className="text-xs text-gray-500 font-mono mt-0.5">
            Generation Ship Meridian
          </p>
          <div className="flex items-center gap-2 mt-1">
            {playerTitle && (
              <span className="text-xs text-sky-400/70 font-mono">
                {playerTitle}
              </span>
            )}
            <SkillsTooltip earnedSkills={earnedSkills} />
            <ComputeBayTooltip completedMissions={completedMissions} newTierIds={newTierIds} />
          </div>
        </div>

        {/* Arcs */}
        <div className="space-y-6">
          {ALL_ARCS.map(arc => {
            const missions = getMissionsForArc(arc.id);
            const gameplayMissions = missions.filter(m => m.type !== 'pivot');
            const completedCount = gameplayMissions.filter(m => completedMissions.includes(m.id)).length;

            return (
              <div key={arc.id}>
                {/* Arc header */}
                <div className="border-l-2 border-amber-500/40 pl-3 mb-3">
                  <div className="flex items-center gap-2">
                    <h3 className="text-sm font-semibold text-amber-400 uppercase tracking-wider font-mono">
                      {arc.name}
                    </h3>
                    <span className="text-xs text-gray-600 font-mono">
                      [{completedCount}/{gameplayMissions.length}]
                    </span>
                    {completedCount > 0 && (
                      <button
                        onClick={() => resetProgress()}
                        className="text-xs font-mono text-gray-700 hover:text-gray-500 cursor-pointer transition-colors"
                      >
                        reset progress
                      </button>
                    )}
                  </div>
                  <p className="text-xs text-gray-500 font-mono mt-0.5">{arc.subtitle}</p>
                </div>

                {/* Missions */}
                <div className="space-y-1">
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
                      const unmetPrereq = mission.prerequisites
                        .filter(id => !completedMissions.includes(id))
                        .map(id => getMissionById(id))
                        .find(Boolean);
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
                </div>
              </div>
            );
          })}
        </div>

      </div>
    </div>
  );
}
