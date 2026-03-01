/**
 * RPG game mode types.
 * Reuses TaskSetup, WinningCriterion, ExpectedChange from the shared game types.
 */

import type { TaskSetup, WinningCriterion, ExpectedChange } from '../game/types.ts';

export interface RPGArc {
  id: string;           // 'arc1-survival'
  name: string;         // 'Survival'
  subtitle: string;     // 'The ship is old. Systems are failing.'
  description: string;
  order: number;
}

export interface RPGMission {
  id: string;            // 'mission-1-1'
  arcId: string;
  order: number;         // display order within arc
  title: string;
  subtitle: string;      // one-line teaser for mission select
  briefing: string;      // narrative + task framing (GlossaryText syntax)
  successNarrative: string; // post-success story + explanation

  type?: 'mission' | 'pivot';  // default: 'mission'. Pivots have no sim task.
  learningObjectives: string[];  // 2-4 action-oriented objectives

  primaryMode: 'training' | 'inference';
  setup: TaskSetup;
  winningCriteria: WinningCriterion[];
  expectedChanges?: ExpectedChange[];
  hints: string[];

  prerequisites: string[];   // mission IDs (DAG edges)
  skillsAwarded: string[];   // skill IDs earned on completion
}

export interface RPGSkill {
  id: string;
  name: string;
  description: string;
}
