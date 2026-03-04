/**
 * Skill catalog — single source of truth for all earnable RPG skills.
 * Missions reference skill IDs from this registry. Stars are derived from
 * completed missions: each mission awarding a skill adds one star.
 */

import type { RPGSkill, RPGMission } from './types.ts';
import { ALL_MISSIONS, getMissionById, isMissionUnlocked } from './missions/index.ts';

// ── Skill Catalog (7 skills, 26 total stars) ──────────────────────────

export const ALL_SKILLS: Record<string, RPGSkill> = {
  // ── Multi-star skills ───────────────────────────────────────────────
  'model-parallelism': {
    id: 'model-parallelism',
    name: 'Model Parallelism',
    description: 'Split models across GPUs using tensor, pipeline, context, and expert parallelism',
    starLabels: ['Weight Sharding', 'Prefill Acceleration', 'TP Tradeoffs', 'Pipeline Stages', 'Ring Attention', 'Expert Distribution'],
  },
  precision: {
    id: 'precision',
    name: 'Numerical Precision',
    description: 'Master precision tradeoffs for memory, speed, and quality',
    starLabels: ['Weight Quantization', 'Quantization for Cost', 'Mixed-Precision Training', 'FP8 Compute'],
  },
  hardware: {
    id: 'hardware',
    name: 'Hardware',
    description: 'Match workloads to GPU memory, bandwidth, and network topology',
    starLabels: ['Memory Capacity', 'Bandwidth Bottlenecks', 'Network Topology'],
  },
  'resource-efficiency': {
    id: 'resource-efficiency',
    name: 'Resource Efficiency',
    description: 'Minimize GPU cost through better hardware use',
    starLabels: ['Cost Optimization', 'Distributed Training', 'Training Efficiency', 'Pipeline Scheduling', 'Cluster Allocation'],
  },
  batching: {
    id: 'batching',
    name: 'Batching',
    description: 'Use batching and scheduling to maximize GPU utilization across requests',
    starLabels: ['Batch Sizing', 'Throughput Scaling', 'Dynamic Scheduling'],
  },
  'memory-optimization': {
    id: 'memory-optimization',
    name: 'Memory Optimization',
    description: 'Reduce GPU memory usage through weight sharding, activation checkpointing, and parameter-efficient fine-tuning',
    starLabels: ['Weight Sharding', 'Activation Checkpointing', 'Adapter Fine-tuning'],
  },

  'inference-optimization': {
    id: 'inference-optimization',
    name: 'Inference Optimization',
    description: 'Accelerate inference with KV cache management, Flash Attention, and speculative decoding',
    starLabels: ['Cache Optimization', 'Draft-Verify Pipeline'],
  },
};

// ── Helper functions ──────────────────────────────────────────────────

/** Missions that award a skill, sorted by arc order then mission order. */
export function getMissionsForSkill(skillId: string): RPGMission[] {
  return ALL_MISSIONS
    .filter(m => m.skillsAwarded.includes(skillId))
    .sort((a, b) => {
      // Arc order: arc1 < arc2
      if (a.arcId !== b.arcId) return a.arcId < b.arcId ? -1 : 1;
      return a.order - b.order;
    });
}

/** Earned stars for a skill = count of completed missions that award it. */
export function getEarnedStars(skillId: string, completedMissions: string[]): number {
  const completed = new Set(completedMissions);
  return getMissionsForSkill(skillId).filter(m => completed.has(m.id)).length;
}

/** Total stars earned / total possible across all skills. */
export function getStarCounts(completedMissions: string[]): {
  earned: number;
  total: number;
  touched: number;
  skillCount: number;
} {
  const skillIds = Object.keys(ALL_SKILLS);
  let earned = 0;
  let total = 0;
  let touched = 0;

  for (const skillId of skillIds) {
    const skill = ALL_SKILLS[skillId];
    const stars = getEarnedStars(skillId, completedMissions);
    total += skill.starLabels.length;
    earned += stars;
    if (stars > 0) touched++;
  }

  return { earned, total, touched, skillCount: skillIds.length };
}

/** Player rank title based on mission progress. Checked in priority (highest rank first). */
export function getPlayerTitle(completedMissions: string[]): string {
  const completed = new Set(completedMissions);
  // Helper: true when all of a pivot's prerequisites are complete
  const pivotReady = (id: string) => {
    const m = getMissionById(id);
    return m !== undefined && isMissionUnlocked(m, completedMissions);
  };
  if (pivotReady('mission-3-7'))        return 'First Contact Commander';
  if (completed.has('mission-3-1'))     return 'Cluster Commander';
  if (pivotReady('mission-2-11'))       return 'Chief Compute Officer';
  if (completed.has('mission-2-7'))     return 'Systems Architect';
  if (completed.has('mission-2-4'))     return 'Training Lead';
  if (pivotReady('mission-1-8'))        return 'Senior Compute Officer';
  if (completedMissions.length >= 4) return 'Inference Specialist';
  return 'Compute Officer';  // default rank — always shown
}

/** Detect a rank promotion when completing a mission. No badge for the first title (null→title). */
export function getPromotion(completedMissions: string[], missionId: string) {
  const oldTitle = getPlayerTitle(completedMissions);
  const withCurrent = completedMissions.includes(missionId)
    ? completedMissions : [...completedMissions, missionId];
  const newTitle = getPlayerTitle(withCurrent);
  return { newTitle, isPromotion: newTitle !== oldTitle };
}

/** Skills awarded by a mission with current star info (for MissionSuccess). */
export function getSkillAwardsForMission(
  missionId: string,
  completedMissions: string[],
): { skill: RPGSkill; earnedStars: number; starLabel: string }[] {
  const mission = ALL_MISSIONS.find(m => m.id === missionId);
  if (!mission) return [];

  // completedMissions should already include this mission for post-success display
  return mission.skillsAwarded.map(skillId => {
    const skill = ALL_SKILLS[skillId];
    if (!skill) return null;

    const earnedStars = getEarnedStars(skillId, completedMissions);
    // Find which star this mission corresponds to
    const missions = getMissionsForSkill(skillId);
    const missionIndex = missions.findIndex(m => m.id === missionId);
    const starLabel = skill.starLabels[missionIndex] ?? '';

    return { skill, earnedStars, starLabel };
  }).filter(Boolean) as { skill: RPGSkill; earnedStars: number; starLabel: string }[];
}
