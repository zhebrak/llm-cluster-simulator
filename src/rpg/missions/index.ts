/**
 * Mission registry — barrel + DAG utilities.
 */

import type { RPGArc, RPGMission } from '../types.ts';
import { ARC1, ARC1_MISSIONS } from './arc1-survival.ts';
import { ARC2, ARC2_MISSIONS } from './arc2-discovery.ts';
import { ARC3, ARC3_MISSIONS, COMPLETION } from './arc3-wonder.ts';

export const ALL_ARCS: RPGArc[] = [ARC1, ARC2, ARC3];

export const ALL_MISSIONS: RPGMission[] = [...ARC1_MISSIONS, ...ARC2_MISSIONS, ...ARC3_MISSIONS];

export function getMissionById(id: string): RPGMission | undefined {
  return ALL_MISSIONS.find(m => m.id === id);
}

export function getMissionsForArc(arcId: string): RPGMission[] {
  return ALL_MISSIONS.filter(m => m.arcId === arcId).sort((a, b) => a.order - b.order);
}

/**
 * DAG unlock check: a mission is unlocked when all its prerequisites are completed.
 */
export function isMissionUnlocked(mission: RPGMission, completedIds: string[]): boolean {
  return mission.prerequisites.every(id => completedIds.includes(id));
}

/**
 * Return the arc matching the player's furthest progression.
 * Post-game completion state after mission-3-7 (The Reply).
 */
export function getActiveArc(completedMissions: string[]): RPGArc {
  if (completedMissions.includes('mission-3-7')) return COMPLETION;
  if (completedMissions.includes('mission-2-11')) return ARC3;
  if (completedMissions.includes('mission-1-8')) return ARC2;
  return ARC1;
}
