/**
 * Skill catalog — single source of truth for all earnable RPG skills.
 * Missions reference skill IDs from this registry. Stars are derived from
 * completed missions: each mission awarding a skill adds one star.
 */

import type { RPGSkill, RPGMission } from './types.ts';
import { ALL_MISSIONS } from './missions/index.ts';

// ── Skill Catalog (8 skills, 24 total stars) ──────────────────────────

export const ALL_SKILLS: Record<string, RPGSkill> = {
  // ── Multi-star skills ───────────────────────────────────────────────
  'model-parallelism': {
    id: 'model-parallelism',
    name: 'Model Parallelism',
    description: 'Split models across GPUs using tensor, pipeline, context, and expert parallelism',
    starLabels: ['Weight Sharding', 'Prefill Acceleration', 'Pipeline Stages', 'Ring Attention', 'Expert Distribution', 'Pipeline Scheduling'],
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
    starLabels: ['Cost Optimization', 'Training Efficiency', 'Cluster Allocation'],
  },
  batching: {
    id: 'batching',
    name: 'Batching',
    description: 'Use batching and scheduling to maximize GPU utilization across requests',
    starLabels: ['Batch Sizing', 'Dynamic Scheduling'],
  },
  'data-parallelism': {
    id: 'data-parallelism',
    name: 'Data Parallelism',
    description: 'Replicate models and distribute data across GPUs for throughput',
    starLabels: ['Replica Scaling', 'Distributed Training'],
  },
  'memory-optimization': {
    id: 'memory-optimization',
    name: 'Memory Optimization',
    description: 'Reduce GPU memory usage through activation checkpointing and parameter-efficient fine-tuning',
    starLabels: ['Activation Checkpointing', 'Adapter Fine-tuning'],
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
