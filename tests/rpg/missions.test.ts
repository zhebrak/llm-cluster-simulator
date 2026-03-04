/**
 * RPG mission structure and calibration tests
 *
 * Validates:
 *   1. Mission structure (IDs, types, DAG consistency, new fields)
 *   2. Calibration for all Arc 1 missions: default → fail, solution → pass
 */

import { describe, it, expect } from 'vitest';
import { ALL_ARCS, ALL_MISSIONS, getMissionById, getMissionsForArc, isMissionUnlocked } from '../../src/rpg/missions/index.ts';
import { ALL_SKILLS, getMissionsForSkill } from '../../src/rpg/skills.ts';
import { evaluateCriterion } from '../../src/game/validation.ts';
import { buildInferenceContext } from '../helpers/simulation.ts';

// ── Structure tests ──────────────────────────────────────────────────

describe('RPG Mission Structure', () => {
  it('all missions have unique IDs', () => {
    const ids = ALL_MISSIONS.map(m => m.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('all missions reference valid arc IDs', () => {
    const arcIds = new Set(ALL_ARCS.map(a => a.id));
    for (const mission of ALL_MISSIONS) {
      expect(arcIds.has(mission.arcId), `Mission ${mission.id} references unknown arc ${mission.arcId}`).toBe(true);
    }
  });

  it('all mission prerequisites reference existing mission IDs', () => {
    const ids = new Set(ALL_MISSIONS.map(m => m.id));
    for (const mission of ALL_MISSIONS) {
      for (const prereq of mission.prerequisites) {
        expect(ids.has(prereq), `Mission ${mission.id} has unknown prerequisite ${prereq}`).toBe(true);
      }
    }
  });

  it('no self-referential prerequisites', () => {
    for (const mission of ALL_MISSIONS) {
      expect(mission.prerequisites.includes(mission.id), `Mission ${mission.id} lists itself as prerequisite`).toBe(false);
    }
  });

  it('non-pivot missions have at least one winning criterion (or objectives with criteria)', () => {
    for (const mission of ALL_MISSIONS) {
      if (mission.type === 'pivot') continue;
      // Multi-objective missions carry criteria on their objectives, not at top level
      const isMultiObjective = mission.objectives && mission.objectives.length > 0;
      if (isMultiObjective) {
        for (const obj of mission.objectives!) {
          expect(obj.winningCriteria.length, `Mission ${mission.id} objective ${obj.id} has no criteria`).toBeGreaterThan(0);
        }
      } else {
        expect(mission.winningCriteria.length, `Mission ${mission.id} has no criteria`).toBeGreaterThan(0);
      }
    }
  });

  it('non-pivot missions have at least one hint', () => {
    for (const mission of ALL_MISSIONS) {
      if (mission.type === 'pivot') continue;
      expect(mission.hints.length, `Mission ${mission.id} has no hints`).toBeGreaterThan(0);
    }
  });

  it('non-pivot missions have at least one learning objective', () => {
    for (const mission of ALL_MISSIONS) {
      if (mission.type === 'pivot') continue;
      expect(mission.learningObjectives.length, `Mission ${mission.id} has no learning objectives`).toBeGreaterThan(0);
    }
  });

  it('all missions have a subtitle', () => {
    for (const mission of ALL_MISSIONS) {
      expect(mission.subtitle.length, `Mission ${mission.id} has empty subtitle`).toBeGreaterThan(0);
    }
  });

  it('pivot missions have empty criteria, hints, and learning objectives', () => {
    for (const mission of ALL_MISSIONS) {
      if (mission.type !== 'pivot') continue;
      expect(mission.winningCriteria.length, `Pivot ${mission.id} should have no criteria`).toBe(0);
      expect(mission.hints.length, `Pivot ${mission.id} should have no hints`).toBe(0);
      expect(mission.learningObjectives.length, `Pivot ${mission.id} should have no learning objectives`).toBe(0);
    }
  });

  it('all awarded skills reference existing skill IDs', () => {
    for (const mission of ALL_MISSIONS) {
      for (const skillId of mission.skillsAwarded) {
        expect(ALL_SKILLS[skillId], `Mission ${mission.id} awards unknown skill ${skillId}`).toBeDefined();
      }
    }
  });

  it('starLabels count matches number of missions awarding each skill', () => {
    for (const [skillId, skill] of Object.entries(ALL_SKILLS)) {
      const missions = getMissionsForSkill(skillId);
      expect(
        skill.starLabels.length,
        `Skill ${skillId} has ${skill.starLabels.length} starLabels but ${missions.length} missions award it`,
      ).toBe(missions.length);
    }
  });

  it('getMissionById returns correct mission', () => {
    const m = getMissionById('mission-1-1');
    expect(m).toBeDefined();
    expect(m!.title).toBe('Wake-Up Call');
  });

  it('getMissionsForArc returns missions sorted by order', () => {
    const missions = getMissionsForArc('arc1-survival');
    expect(missions.length).toBe(8);
    for (let i = 1; i < missions.length; i++) {
      expect(missions[i].order).toBeGreaterThanOrEqual(missions[i - 1].order);
    }
  });
});

describe('RPG DAG Progression', () => {
  it('mission with no prerequisites is always unlocked', () => {
    const m = getMissionById('mission-1-1')!;
    expect(isMissionUnlocked(m, [])).toBe(true);
  });

  it('mission with prerequisites requires all completed', () => {
    const m16 = getMissionById('mission-1-6')!;
    expect(isMissionUnlocked(m16, [])).toBe(false);
    expect(isMissionUnlocked(m16, ['mission-1-2'])).toBe(false);
    expect(isMissionUnlocked(m16, ['mission-1-2', 'mission-1-3'])).toBe(true);
  });

  it('pivot mission prerequisites are correct', () => {
    const m18 = getMissionById('mission-1-8')!;
    expect(m18.type).toBe('pivot');
    expect(isMissionUnlocked(m18, [])).toBe(false);
    expect(isMissionUnlocked(m18, ['mission-1-3', 'mission-1-5'])).toBe(false);
    expect(isMissionUnlocked(m18, ['mission-1-3', 'mission-1-5', 'mission-1-6'])).toBe(true);
  });

  it('1-7 depends only on 1-6', () => {
    const m17 = getMissionById('mission-1-7')!;
    expect(m17.prerequisites).toEqual(['mission-1-6']);
    expect(isMissionUnlocked(m17, ['mission-1-6'])).toBe(true);
  });

  it('1-2, 1-3, 1-4, 1-5 all depend only on 1-1', () => {
    for (const id of ['mission-1-2', 'mission-1-3', 'mission-1-4', 'mission-1-5']) {
      const m = getMissionById(id)!;
      expect(m.prerequisites).toEqual(['mission-1-1']);
    }
  });
});

// ── Calibration tests ────────────────────────────────────────────────

describe('RPG Mission Calibration', () => {
  describe('Mission 1-1: Wake-Up Call', () => {
    const mission = getMissionById('mission-1-1')!;

    it('default setup (BF16 on T4) → OOM', () => {
      const ctx = buildInferenceContext(mission.setup);
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(false);
      expect(ctx!.memoryUtilization).toBeGreaterThan(1.0);
    });

    it('INT8 weight precision → success', () => {
      const ctx = buildInferenceContext({ ...mission.setup, weightPrecision: 'int8' });
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(true);
      expect(ctx!.memoryUtilization).toBeLessThan(1.0);
    });

    it('INT4 weight precision → success', () => {
      const ctx = buildInferenceContext({ ...mission.setup, weightPrecision: 'int4' });
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(true);
    });

    it('winning criteria pass with INT8', () => {
      const ctx = buildInferenceContext({ ...mission.setup, weightPrecision: 'int8' });
      for (const criterion of mission.winningCriteria) {
        expect(evaluateCriterion(ctx, criterion), `Criterion: ${criterion.label}`).toBe(true);
      }
    });
  });

  describe('Mission 1-2: The Upgrade', () => {
    const mission = getMissionById('mission-1-2')!;

    it('default setup (70B on RTX 4090) → OOM at any precision', () => {
      // BF16
      const ctxBF16 = buildInferenceContext(mission.setup);
      expect(ctxBF16).not.toBeNull();
      expect(ctxBF16!.success).toBe(false);

      // INT4
      const ctxINT4 = buildInferenceContext({ ...mission.setup, weightPrecision: 'int4' });
      expect(ctxINT4).not.toBeNull();
      expect(ctxINT4!.success).toBe(false);
    });

    it('switch to 8B model → success + >20 tok/s', () => {
      const ctx = buildInferenceContext({
        ...mission.setup,
        modelId: 'llama3.1-8b',
        weightPrecision: 'int8',
      });
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(true);
      expect(ctx!.throughput?.tokensPerSecond).toBeGreaterThan(20);

      // All criteria pass
      for (const criterion of mission.winningCriteria) {
        expect(evaluateCriterion(ctx, criterion), `Criterion: ${criterion.label}`).toBe(true);
      }
    });
  });

  describe('Mission 1-3: Slow Reflexes', () => {
    const mission = getMissionById('mission-1-3')!;

    it('default setup (8B INT8 on T4) → TPOT too high', () => {
      const ctx = buildInferenceContext(mission.setup);
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(true);
      expect(ctx!.latency?.tpot).toBeGreaterThan(18);
    });

    it('T4 INT4 batch=1 → still too slow', () => {
      const ctx = buildInferenceContext({ ...mission.setup, weightPrecision: 'int4' });
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(true);
      expect(ctx!.latency?.tpot).toBeGreaterThan(18);
    });

    it('T4 Q2_K batch=1 → still too slow', () => {
      const ctx = buildInferenceContext({ ...mission.setup, weightPrecision: 'q2_k' });
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(true);
      expect(ctx!.latency?.tpot).toBeGreaterThan(18);
    });

    it('T4 batch=4 → TPOT gets worse (physics blocks batch)', () => {
      const ctx = buildInferenceContext({ ...mission.setup, batchSize: 4 });
      const ctxDefault = buildInferenceContext(mission.setup);
      expect(ctx).not.toBeNull();
      expect(ctxDefault).not.toBeNull();
      expect(ctx!.latency?.tpot).toBeGreaterThanOrEqual(ctxDefault!.latency!.tpot);
    });

    it('switch to RTX 4090 → TPOT < 18ms', () => {
      const ctx = buildInferenceContext({ ...mission.setup, gpuId: 'rtx-4090' });
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(true);
      expect(ctx!.latency?.tpot).toBeLessThan(18);

      for (const criterion of mission.winningCriteria) {
        expect(evaluateCriterion(ctx, criterion), `Criterion: ${criterion.label}`).toBe(true);
      }
    });
  });

  describe('Mission 1-4: Cryo-Pod Monitoring', () => {
    const mission = getMissionById('mission-1-4')!;

    it('default setup (batch=1) → throughput < 400', () => {
      const ctx = buildInferenceContext(mission.setup);
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(true);
      expect(ctx!.throughput?.tokensPerSecond).toBeLessThan(400);
    });

    it('batch=16 → throughput > 400', () => {
      const ctx = buildInferenceContext({ ...mission.setup, batchSize: 16 });
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(true);
      expect(ctx!.throughput?.tokensPerSecond).toBeGreaterThan(400);

      for (const criterion of mission.winningCriteria) {
        expect(evaluateCriterion(ctx, criterion), `Criterion: ${criterion.label}`).toBe(true);
      }
    });
  });

  describe('Mission 1-5: Memory Leak', () => {
    const mission = getMissionById('mission-1-5')!;

    it('default setup (seqLen=131072, FA=off) → OOM', () => {
      const ctx = buildInferenceContext(mission.setup);
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(false);
    });

    it('FA alone (seqLen=131072) → still OOM', () => {
      const ctx = buildInferenceContext({ ...mission.setup, flashAttention: true });
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(false);
    });

    it('FA + KV int8 → success', () => {
      const ctx = buildInferenceContext({
        ...mission.setup, flashAttention: true, kvCachePrecision: 'int8',
      });
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(true);

      for (const criterion of mission.winningCriteria) {
        expect(evaluateCriterion(ctx, criterion), `Criterion: ${criterion.label}`).toBe(true);
      }
    });

    it('FA + KV fp8 → success', () => {
      const ctx = buildInferenceContext({
        ...mission.setup, flashAttention: true, kvCachePrecision: 'fp8',
      });
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(true);

      for (const criterion of mission.winningCriteria) {
        expect(evaluateCriterion(ctx, criterion), `Criterion: ${criterion.label}`).toBe(true);
      }
    });
  });

  describe('Mission 1-6: The Archive Vault', () => {
    const mission = getMissionById('mission-1-6')!;

    it('default setup (70B BF16, 4×A100, TP=1) → OOM', () => {
      const ctx = buildInferenceContext(mission.setup);
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(false);
    });

    it('TP=4 → success + TTFT < 500ms', () => {
      const ctx = buildInferenceContext({ ...mission.setup, tensorParallel: 4 });
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(true);
      expect(ctx!.latency?.ttft).toBeLessThan(500);

      for (const criterion of mission.winningCriteria) {
        expect(evaluateCriterion(ctx, criterion), `Criterion: ${criterion.label}`).toBe(true);
      }
    });

    it('TP=2 → success', () => {
      const ctx = buildInferenceContext({ ...mission.setup, tensorParallel: 2 });
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(true);
    });
  });

  describe('Mission 1-7: Fuel Budget', () => {
    const mission = getMissionById('mission-1-7')!;

    it('default setup (4×A100 TP=4 BF16) → numGPUs=4 fails criterion', () => {
      const ctx = buildInferenceContext(mission.setup);
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(true);
      // numGPUs criterion fails
      expect(ctx!.numGPUs).toBe(4);
      const gpuCriterion = mission.winningCriteria.find(c => c.field === 'numGPUs');
      expect(gpuCriterion).toBeDefined();
      expect(evaluateCriterion(ctx, gpuCriterion!)).toBe(false);
    });

    it('BF16 + TP=2 + numGPUs=2 → succeeds but memoryUtilization > 90%', () => {
      const ctx = buildInferenceContext({
        ...mission.setup,
        tensorParallel: 2,
        numGPUs: 2,
      });
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(true);
      expect(ctx!.memoryUtilization).toBeGreaterThan(0.90);
    });

    it('INT8 + TP=2 + numGPUs=2 → all criteria pass', () => {
      const ctx = buildInferenceContext({
        ...mission.setup,
        weightPrecision: 'int8',
        tensorParallel: 2,
        numGPUs: 2,
      });
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(true);
      expect(ctx!.numGPUs).toBe(2);
      expect(ctx!.memoryUtilization).toBeLessThan(0.90);

      for (const criterion of mission.winningCriteria) {
        expect(evaluateCriterion(ctx, criterion), `Criterion: ${criterion.label}`).toBe(true);
      }
    });
  });
});
