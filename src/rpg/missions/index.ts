/**
 * Mission registry — barrel + DAG utilities.
 */

import type { RPGArc, RPGMission } from '../types.ts';
import { ARC1, ARC1_MISSIONS } from './arc1-survival.ts';

export const ALL_ARCS: RPGArc[] = [ARC1];

export const ALL_MISSIONS: RPGMission[] = [...ARC1_MISSIONS];

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
