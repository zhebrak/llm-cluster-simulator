/**
 * Multi-objective mission tests
 *
 * Validates the multi-objective data model, objective structure,
 * and that objectives carry valid criteria and setups.
 */

import { describe, it, expect } from 'vitest';
import { ALL_MISSIONS, getMissionById } from '../../src/rpg/missions/index.ts';
import type { MissionObjective } from '../../src/rpg/types.ts';

// ── Structure tests ──────────────────────────────────────────────────

describe('Multi-objective mission structure', () => {
  const multiObjMissions = ALL_MISSIONS.filter(m => m.objectives && m.objectives.length > 0);

  it('at least one multi-objective mission exists', () => {
    expect(multiObjMissions.length).toBeGreaterThan(0);
  });

  it('mission 2-10 is multi-objective', () => {
    const m = getMissionById('mission-2-10')!;
    expect(m.objectives).toBeDefined();
    expect(m.objectives!.length).toBe(2);
  });

  it('multi-objective missions have empty top-level winningCriteria', () => {
    for (const m of multiObjMissions) {
      expect(m.winningCriteria, `${m.id} should have empty top-level criteria`).toEqual([]);
    }
  });

  it('multi-objective missions have empty top-level expectedChanges', () => {
    for (const m of multiObjMissions) {
      expect(m.expectedChanges, `${m.id} should have empty top-level expectedChanges`).toEqual([]);
    }
  });

  it('all objectives have unique IDs within their mission', () => {
    for (const m of multiObjMissions) {
      const ids = m.objectives!.map(o => o.id);
      expect(new Set(ids).size, `${m.id} has duplicate objective IDs`).toBe(ids.length);
    }
  });

  it('all objectives have a label', () => {
    for (const m of multiObjMissions) {
      for (const obj of m.objectives!) {
        expect(obj.label.length, `${m.id}/${obj.id} has empty label`).toBeGreaterThan(0);
      }
    }
  });

  it('all objectives have a valid primaryMode', () => {
    for (const m of multiObjMissions) {
      for (const obj of m.objectives!) {
        expect(['training', 'inference'], `${m.id}/${obj.id} has invalid mode`).toContain(obj.primaryMode);
      }
    }
  });

  it('all objectives have at least one winning criterion', () => {
    for (const m of multiObjMissions) {
      for (const obj of m.objectives!) {
        expect(obj.winningCriteria.length, `${m.id}/${obj.id} has no criteria`).toBeGreaterThan(0);
      }
    }
  });

  it('all objectives have a setup with modelId and gpuId', () => {
    for (const m of multiObjMissions) {
      for (const obj of m.objectives!) {
        expect(obj.setup.modelId, `${m.id}/${obj.id} missing modelId`).toBeDefined();
        expect(obj.setup.gpuId, `${m.id}/${obj.id} missing gpuId`).toBeDefined();
      }
    }
  });
});

// ── Mission 2-10 specific tests ──────────────────────────────────────

describe('Mission 2-10: The Protein Problem', () => {
  const mission = getMissionById('mission-2-10')!;
  const objTrain = mission.objectives!.find(o => o.id === 'obj-train')!;
  const objInfer = mission.objectives!.find(o => o.id === 'obj-infer')!;

  it('has training and inference objectives', () => {
    expect(objTrain.primaryMode).toBe('training');
    expect(objInfer.primaryMode).toBe('inference');
  });

  it('training objective uses H100s', () => {
    expect(objTrain.setup.gpuId).toBe('h100-sxm');
  });

  it('inference objective uses A100s', () => {
    expect(objInfer.setup.gpuId).toBe('a100-80gb');
  });

  it('objectives use different hardware pools (different GPU types)', () => {
    expect(objTrain.setup.gpuId).not.toBe(objInfer.setup.gpuId);
  });

  it('training objective has MFU criterion', () => {
    const mfuCriterion = objTrain.winningCriteria.find(c => c.field === 'mfu');
    expect(mfuCriterion).toBeDefined();
    expect(mfuCriterion!.operator).toBe('>');
  });

  it('inference objective has TTFT criterion', () => {
    const ttftCriterion = objInfer.winningCriteria.find(c => c.field === 'latency.ttft');
    expect(ttftCriterion).toBeDefined();
    expect(ttftCriterion!.operator).toBe('<');
  });

  it('both objectives guard modelId, gpuId, numGPUs', () => {
    for (const obj of [objTrain, objInfer]) {
      const guarded = (obj.expectedChanges ?? []).filter(e => e.check === 'unchanged').map(e => e.field);
      expect(guarded, `${obj.id} should guard modelId`).toContain('modelId');
      expect(guarded, `${obj.id} should guard gpuId`).toContain('gpuId');
      expect(guarded, `${obj.id} should guard numGPUs`).toContain('numGPUs');
    }
  });

  it('mission-level winningCriteria is empty (objectives carry their own)', () => {
    expect(mission.winningCriteria).toEqual([]);
  });

  it('mission-level expectedChanges is empty', () => {
    expect(mission.expectedChanges).toEqual([]);
  });

  it('mission has skills awarded (resource-efficiency)', () => {
    expect(mission.skillsAwarded).toContain('resource-efficiency');
  });
});

// ── Backward compatibility ───────────────────────────────────────────

describe('Single-objective missions are unaffected', () => {
  const singleObjMissions = ALL_MISSIONS.filter(
    m => m.type !== 'pivot' && (!m.objectives || m.objectives.length === 0),
  );

  it('single-objective missions have no objectives array (or empty)', () => {
    for (const m of singleObjMissions) {
      if (m.objectives) {
        expect(m.objectives.length, `${m.id} has unexpected objectives`).toBe(0);
      }
    }
  });

  it('single-objective gameplay missions have their own winningCriteria', () => {
    for (const m of singleObjMissions) {
      expect(m.winningCriteria.length, `${m.id} has no criteria`).toBeGreaterThan(0);
    }
  });
});
