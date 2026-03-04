/**
 * Arc 3 mission calibration tests
 *
 * Validates:
 *   1. DAG structure for Arc 3 missions
 *   2. Calibration for all gameplay missions: default → fail, solution → pass
 */

import { describe, it, expect } from 'vitest';
import { getMissionById, getMissionsForArc, isMissionUnlocked } from '../../src/rpg/missions/index.ts';
import {
  buildTrainingContext,
  buildInferenceContext,
  checkAllCriteria,
  checkAnyCriterionFails,
} from '../helpers/simulation.ts';

// ── DAG tests ────────────────────────────────────────────────────────

describe('Arc 3 DAG Structure', () => {
  it('Arc 3 has 7 missions', () => {
    const missions = getMissionsForArc('arc3-wonder');
    expect(missions.length).toBe(7);
  });

  it('missions sorted by order', () => {
    const missions = getMissionsForArc('arc3-wonder');
    for (let i = 1; i < missions.length; i++) {
      expect(missions[i].order).toBeGreaterThanOrEqual(missions[i - 1].order);
    }
  });

  it('3-1 requires 2-11 (Life pivot)', () => {
    const m = getMissionById('mission-3-1')!;
    expect(m.prerequisites).toEqual(['mission-2-11', 'mission-1-4']);
    expect(isMissionUnlocked(m, ['mission-2-11', 'mission-1-4'])).toBe(true);
    expect(isMissionUnlocked(m, [])).toBe(false);
  });

  it('3-2 and 3-3 are parallel after 3-1', () => {
    const m32 = getMissionById('mission-3-2')!;
    const m33 = getMissionById('mission-3-3')!;
    expect(m32.prerequisites).toEqual(['mission-3-1']);
    expect(m33.prerequisites).toEqual(['mission-3-1']);
  });

  it('3-4 requires 3-1 and 2-9', () => {
    const m = getMissionById('mission-3-4')!;
    expect(m.prerequisites).toEqual(['mission-3-1', 'mission-2-9']);
  });

  it('3-5 requires 2-6 + 3-2 + 3-3 + 3-4', () => {
    const m = getMissionById('mission-3-5')!;
    expect(m.prerequisites).toEqual(['mission-2-6', 'mission-3-2', 'mission-3-3', 'mission-3-4']);
    expect(isMissionUnlocked(m, ['mission-2-6', 'mission-3-2', 'mission-3-3', 'mission-3-4'])).toBe(true);
    expect(isMissionUnlocked(m, ['mission-3-2', 'mission-3-3', 'mission-3-4'])).toBe(false);
  });

  it('3-6 requires 3-5', () => {
    expect(getMissionById('mission-3-6')!.prerequisites).toEqual(['mission-3-5']);
  });

  it('3-7 (pivot) requires 3-6', () => {
    const m = getMissionById('mission-3-7')!;
    expect(m.type).toBe('pivot');
    expect(m.prerequisites).toEqual(['mission-3-6']);
  });

  it('3-5 is a multi-objective mission', () => {
    const m = getMissionById('mission-3-5')!;
    expect(m.objectives).toBeDefined();
    expect(m.objectives!.length).toBe(3);
    expect(m.objectives![0].id).toBe('obj-biosig-train');
    expect(m.objectives![1].id).toBe('obj-finetune');
    expect(m.objectives![2].id).toBe('obj-probe-infer');
  });

  it('3-6 is a multi-objective mission with 4 objectives', () => {
    const m = getMissionById('mission-3-6')!;
    expect(m.objectives).toBeDefined();
    expect(m.objectives!.length).toBe(4);
    expect(m.objectives![0].id).toBe('obj-longctx-train');
    expect(m.objectives![1].id).toBe('obj-moe-train');
    expect(m.objectives![2].id).toBe('obj-feed-infer');
    expect(m.objectives![3].id).toBe('obj-latency-infer');
  });
});

// ── Calibration tests ────────────────────────────────────────────────

describe('Arc 3 Mission Calibration', () => {
  describe('Mission 3-1: Landfall (continuous batching)', () => {
    const mission = getMissionById('mission-3-1')!;

    it('default setup (CB=off) → throughput < 12500 tok/s', () => {
      const ctx = buildInferenceContext(mission.setup);
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(true);
      expect(ctx!.throughput?.tokensPerSecond).toBeLessThan(12500);
      expect(checkAnyCriterionFails(ctx, mission.winningCriteria)).toBe(true);
    });

    it('CB=on → throughput > 12500 tok/s, all criteria pass', () => {
      const ctx = buildInferenceContext({ ...mission.setup, continuousBatching: true });
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(true);
      expect(ctx!.throughput?.tokensPerSecond).toBeGreaterThan(12500);
      expect(checkAllCriteria(ctx, mission.winningCriteria)).toBe(true);
    });

    it('regression: fp4 b=512 static (tightest non-CB) still below 12500', () => {
      const ctx = buildInferenceContext({
        ...mission.setup,
        weightPrecision: 'fp4',
        batchSize: 512,
        continuousBatching: false,
      });
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(true);
      expect(ctx!.throughput?.tokensPerSecond).toBeLessThan(12500);
    });
  });

  describe('Mission 3-2: Deep Signal (context parallelism)', () => {
    const mission = getMissionById('mission-3-2')!;

    it('default setup (CP=1) → OOM', () => {
      const ctx = buildTrainingContext(mission.setup);
      expect(ctx.success).toBe(false);
      expect(ctx.memoryUtilization).toBeGreaterThan(1.0);
    });

    it('CP=4 → fits in memory + MFU > 20%', () => {
      const ctx = buildTrainingContext({ ...mission.setup, cpDegree: 4 });
      expect(ctx.success).toBe(true);
      expect(ctx.mfu).toBeGreaterThan(0.20);
      expect(checkAllCriteria(ctx, mission.winningCriteria)).toBe(true);
    });

    // Physics-guard tests: document whether alternative approaches solve the OOM
    // These are blocked by expectedChanges regardless, but physics may provide a secondary barrier.

    it('physics guard: full AC at CP=1 → still OOM (physics blocks)', () => {
      const ctx = buildTrainingContext({
        ...mission.setup,
        checkpointingGranularity: 'full',
        cpDegree: 1,
      });
      // Full AC reduces activation memory via sqrt(N) checkpointing, but at 131K
      // the per-GPU activations are so large that even full AC can't save CP=1.
      expect(ctx.success).toBe(false);
      expect(ctx.memoryUtilization).toBeGreaterThan(1.0);
    });

    it('physics guard: LoRA at CP=1 → still OOM (physics blocks)', () => {
      const ctx = buildTrainingContext({
        ...mission.setup,
        finetuningMethod: 'lora',
        cpDegree: 1,
      });
      // LoRA reduces optimizer memory but activations still overflow at 131K.
      expect(ctx.success).toBe(false);
      expect(ctx.memoryUtilization).toBeGreaterThan(1.0);
    });

    it('physics guard: TP=16 at CP=1 → fits (NOT physics-blocked, needs expectedChanges lock)', () => {
      const ctx = buildTrainingContext({
        ...mission.setup,
        tpDegree: 16,
        cpDegree: 1,
      });
      // Cross-node TP=16 halves per-GPU activations via SP, which is enough at 131K.
      // This bypass is blocked by tpDegree: unchanged in expectedChanges, not physics.
      expect(ctx.success).toBe(true);
    });
  });

  describe('Mission 3-3: Alien Model (expert parallelism)', () => {
    const mission = getMissionById('mission-3-3')!;

    it('default setup (EP=1) → MFU < 55%', () => {
      const ctx = buildTrainingContext(mission.setup);
      expect(ctx.success).toBe(true);
      expect(ctx.mfu).toBeLessThan(0.55);
      expect(checkAnyCriterionFails(ctx, mission.winningCriteria)).toBe(true);
    });

    it('EP=2 → MFU < 55% (insufficient EP)', () => {
      const ctx = buildTrainingContext({ ...mission.setup, epDegree: 2 });
      expect(ctx.success).toBe(true);
      expect(ctx.mfu).toBeLessThan(0.55);
      expect(checkAnyCriterionFails(ctx, mission.winningCriteria)).toBe(true);
    });

    it('EP=1 bypass: TP/PP/GBS combos stay below 55% MFU (physics guard)', () => {
      // Without EP, all expert weights participate in FSDP AllGather even though
      // only ~17B of ~400B activate per token. This caps MFU around 22-23%.
      const bypassConfigs = [
        { tpDegree: 4, ppDegree: 1, epDegree: 1 },
        { tpDegree: 2, ppDegree: 1, epDegree: 1 },
        { tpDegree: 8, ppDegree: 2, epDegree: 1 },
        { tpDegree: 4, ppDegree: 2, epDegree: 1 },
        { tpDegree: 8, ppDegree: 1, epDegree: 1, globalBatchSize: 256 },
        { tpDegree: 8, ppDegree: 1, epDegree: 1, globalBatchSize: 512 },
        { tpDegree: 4, ppDegree: 1, epDegree: 1, globalBatchSize: 256 },
      ];
      for (const overrides of bypassConfigs) {
        const ctx = buildTrainingContext({ ...mission.setup, ...overrides });
        if (ctx.success) {
          expect(ctx.mfu).toBeLessThan(0.55);
        }
      }
    });

    it('EP=4 → MFU > 55%, all criteria pass', () => {
      const ctx = buildTrainingContext({ ...mission.setup, epDegree: 4 });
      expect(ctx.success).toBe(true);
      expect(ctx.mfu).toBeGreaterThan(0.55);
      expect(checkAllCriteria(ctx, mission.winningCriteria)).toBe(true);
    });

    it('EP=8 → MFU > 55%, all criteria pass', () => {
      const ctx = buildTrainingContext({ ...mission.setup, epDegree: 8 });
      expect(ctx.success).toBe(true);
      expect(ctx.mfu).toBeGreaterThan(0.55);
      expect(checkAllCriteria(ctx, mission.winningCriteria)).toBe(true);
    });

    it('EP=16 → MFU > 55%, all criteria pass', () => {
      const ctx = buildTrainingContext({ ...mission.setup, epDegree: 16 });
      expect(ctx.success).toBe(true);
      expect(ctx.mfu).toBeGreaterThan(0.55);
      expect(checkAllCriteria(ctx, mission.winningCriteria)).toBe(true);
    });
  });

  describe('Mission 3-4: Big Train (pipeline schedule)', () => {
    const mission = getMissionById('mission-3-4')!;

    it('default setup (1F1B v=1) → MFU < 39%', () => {
      const ctx = buildTrainingContext(mission.setup);
      expect(ctx.success).toBe(true);
      expect(ctx.mfu).toBeLessThan(0.39);
      expect(checkAnyCriterionFails(ctx, mission.winningCriteria)).toBe(true);
    });

    it('interleaved-1f1b v=2 → MFU < 39%, fails threshold', () => {
      const ctx = buildTrainingContext({
        ...mission.setup,
        pipelineSchedule: 'interleaved-1f1b',
        interleavedStages: 2,
      });
      expect(ctx.success).toBe(true);
      expect(ctx.mfu).toBeLessThan(0.39);
      expect(checkAnyCriterionFails(ctx, mission.winningCriteria)).toBe(true);
    });

    it('interleaved-1f1b v=2 → bubble > 10% (fails bubble criterion)', () => {
      const ctx = buildTrainingContext({
        ...mission.setup,
        pipelineSchedule: 'interleaved-1f1b',
        interleavedStages: 2,
      });
      expect(ctx.pipelineBubble).toBeGreaterThan(0.10);
    });

    it('interleaved-1f1b v=4 → bubble < 10% (passes bubble criterion)', () => {
      const ctx = buildTrainingContext({
        ...mission.setup,
        pipelineSchedule: 'interleaved-1f1b',
        interleavedStages: 4,
      });
      expect(ctx.pipelineBubble).toBeLessThan(0.10);
    });

    it('interleaved-1f1b v=2 + FP8 → bubble > 10% (FP8 does not reduce bubble)', () => {
      const ctx = buildTrainingContext({
        ...mission.setup,
        pipelineSchedule: 'interleaved-1f1b',
        interleavedStages: 2,
        mixedPrecision: 'fp8',
      });
      expect(ctx.pipelineBubble).toBeGreaterThan(0.10);
    });

    it('interleaved-1f1b v=4 → MFU > 39%, all criteria pass', () => {
      const ctx = buildTrainingContext({
        ...mission.setup,
        pipelineSchedule: 'interleaved-1f1b',
        interleavedStages: 4,
      });
      expect(ctx.success).toBe(true);
      expect(ctx.mfu).toBeGreaterThan(0.39);
      expect(ctx.pipelineBubble).toBeLessThan(0.10);
      expect(checkAllCriteria(ctx, mission.winningCriteria)).toBe(true);
    });
  });

  describe('Mission 3-5: Resource War (multi-objective)', () => {
    const mission = getMissionById('mission-3-5')!;
    const objBiosig = mission.objectives![0];
    const objFinetune = mission.objectives![1];
    const objProbe = mission.objectives![2];

    describe('Objective 1 — Training: Biosignature model', () => {
      it('default setup (BF16 no AC) → OOM', () => {
        const ctx = buildTrainingContext(objBiosig.setup);
        expect(ctx.success).toBe(false);
        expect(ctx.memoryUtilization).toBeGreaterThan(1.0);
        expect(checkAnyCriterionFails(ctx, objBiosig.winningCriteria)).toBe(true);
      });

      it('FP8 + AC → fits + MFU > 50% (AC path)', () => {
        const ctx = buildTrainingContext({
          ...objBiosig.setup,
          mixedPrecision: 'fp8',
          activationCheckpointing: true,
        });
        expect(ctx.success).toBe(true);
        expect(ctx.mfu).toBeGreaterThan(0.50);
        expect(checkAllCriteria(ctx, objBiosig.winningCriteria)).toBe(true);
      });

      it('FSDP-TP FP8 no AC → fits + MFU > 50% (TP path)', () => {
        const ctx = buildTrainingContext({
          ...objBiosig.setup,
          strategyType: 'fsdp-tp',
          mixedPrecision: 'fp8',
          activationCheckpointing: false,
          tpDegree: 2,
        });
        expect(ctx.success).toBe(true);
        expect(ctx.mfu).toBeGreaterThan(0.50);
        expect(checkAllCriteria(ctx, objBiosig.winningCriteria)).toBe(true);
      });
    });

    describe('Objective 2 — Training: Large model fine-tune', () => {
      it('default setup (full fine-tune 70B) → OOM', () => {
        const ctx = buildTrainingContext(objFinetune.setup);
        expect(ctx.success).toBe(false);
        expect(ctx.memoryUtilization).toBeGreaterThan(1.0);
        expect(checkAnyCriterionFails(ctx, objFinetune.winningCriteria)).toBe(true);
      });

      it('LoRA → fits in memory', () => {
        const ctx = buildTrainingContext({
          ...objFinetune.setup,
          finetuningMethod: 'lora',
        });
        expect(ctx.success).toBe(true);
        expect(ctx.memoryUtilization).toBeLessThan(1.0);
        expect(checkAllCriteria(ctx, objFinetune.winningCriteria)).toBe(true);
      });
    });

    describe('Objective 3 — Inference: Real-time probe analysis', () => {
      it('default setup (B=1 CB=off) → throughput << 800 tok/s', () => {
        const ctx = buildInferenceContext(objProbe.setup);
        expect(ctx).not.toBeNull();
        expect(ctx!.success).toBe(true);
        expect(ctx!.throughput?.tokensPerSecond).toBeLessThan(800);
        expect(checkAnyCriterionFails(ctx, objProbe.winningCriteria)).toBe(true);
      });

      it('B=32 + CB=on → throughput > 800 tok/s', () => {
        const ctx = buildInferenceContext({
          ...objProbe.setup,
          batchSize: 32,
          continuousBatching: true,
        });
        expect(ctx).not.toBeNull();
        expect(ctx!.success).toBe(true);
        expect(ctx!.throughput?.tokensPerSecond).toBeGreaterThan(800);
        expect(checkAllCriteria(ctx, objProbe.winningCriteria)).toBe(true);
      });
    });
  });

  describe('Mission 3-6: All Systems Nominal (capstone)', () => {
    const mission = getMissionById('mission-3-6')!;
    const objLongctx = mission.objectives![0];
    const objMoe = mission.objectives![1];
    const objFeed = mission.objectives![2];
    const objLatency = mission.objectives![3];

    describe('Objective 1 — Training: Long-context protocol model', () => {
      it('default setup (CP=1 1F1B v=2) → OOM', () => {
        const ctx = buildTrainingContext(objLongctx.setup);
        expect(ctx.success).toBe(false);
        expect(ctx.memoryUtilization).toBeGreaterThan(1.0);
      });

      it('CP=4 + interleaved v=4 → fits + MFU > 25.5%', () => {
        const ctx = buildTrainingContext({
          ...objLongctx.setup,
          cpDegree: 4,
          pipelineSchedule: 'interleaved-1f1b',
          interleavedStages: 4,
        });
        expect(ctx.success).toBe(true);
        expect(ctx.mfu).toBeGreaterThan(0.255);
        expect(checkAllCriteria(ctx, objLongctx.winningCriteria)).toBe(true);
      });
    });

    describe('Objective 2 — Training: MoE expert translation engine', () => {
      it('default setup (BF16 EP=1) → MFU < 55%', () => {
        const ctx = buildTrainingContext(objMoe.setup);
        expect(ctx.success).toBe(true);
        expect(ctx.mfu).toBeLessThan(0.55);
        expect(checkAnyCriterionFails(ctx, objMoe.winningCriteria)).toBe(true);
      });

      it('physics guard: EP=8 + BF16 → MFU < 55% (FP8 required)', () => {
        const ctx = buildTrainingContext({
          ...objMoe.setup,
          epDegree: 8,
        });
        expect(ctx.success).toBe(true);
        expect(ctx.mfu).toBeLessThan(0.55);
      });

      it('FP8 + EP=4 → MFU > 55%', () => {
        const ctx = buildTrainingContext({
          ...objMoe.setup,
          mixedPrecision: 'fp8',
          epDegree: 4,
        });
        expect(ctx.success).toBe(true);
        expect(ctx.mfu).toBeGreaterThan(0.55);
        expect(checkAllCriteria(ctx, objMoe.winningCriteria)).toBe(true);
      });

      it('FP8 + EP=8 → MFU > 55%', () => {
        const ctx = buildTrainingContext({
          ...objMoe.setup,
          mixedPrecision: 'fp8',
          epDegree: 8,
        });
        expect(ctx.success).toBe(true);
        expect(ctx.mfu).toBeGreaterThan(0.55);
        expect(checkAllCriteria(ctx, objMoe.winningCriteria)).toBe(true);
      });
    });

    describe('Objective 3 — Inference: High-throughput probe feed', () => {
      it('default setup (FP8 batch=1 CB=off) → throughput < 4000 tok/s', () => {
        const ctx = buildInferenceContext(objFeed.setup);
        expect(ctx).not.toBeNull();
        expect(ctx!.success).toBe(true);
        expect(ctx!.throughput?.tokensPerSecond).toBeLessThan(4000);
        expect(checkAnyCriterionFails(ctx, objFeed.winningCriteria)).toBe(true);
      });

      it('physics guard: batch=64 + CB=off → TTFT > 500ms (TTFT blocks non-CB bypass)', () => {
        const ctx = buildInferenceContext({
          ...objFeed.setup,
          batchSize: 64,
          continuousBatching: false,
        });
        expect(ctx).not.toBeNull();
        expect(ctx!.success).toBe(true);
        // Throughput exceeds 4000 tok/s without CB, but TTFT blocks it
        expect(ctx!.throughput?.tokensPerSecond).toBeGreaterThan(4000);
        expect(ctx!.latency?.ttft).toBeGreaterThan(500);
        expect(checkAnyCriterionFails(ctx, objFeed.winningCriteria)).toBe(true);
      });

      it('batch=32 + CB=on → throughput > 4000 tok/s + TTFT < 500ms', () => {
        const ctx = buildInferenceContext({
          ...objFeed.setup,
          batchSize: 32,
          continuousBatching: true,
        });
        expect(ctx).not.toBeNull();
        expect(ctx!.success).toBe(true);
        expect(ctx!.throughput?.tokensPerSecond).toBeGreaterThan(4000);
        expect(ctx!.latency?.ttft).toBeLessThan(500);
        expect(checkAllCriteria(ctx, objFeed.winningCriteria)).toBe(true);
      });
    });

    describe('Objective 4 — Inference: Low-latency translation', () => {
      it('default setup (TP=16) → throughput < 89 tok/s (fails throughput criterion)', () => {
        const ctx = buildInferenceContext(objLatency.setup);
        expect(ctx).not.toBeNull();
        expect(ctx!.success).toBe(true);
        expect(ctx!.throughput?.tokensPerSecond).toBeLessThan(89);
        expect(checkAnyCriterionFails(ctx, objLatency.winningCriteria)).toBe(true);
      });

      it('TP=12 → still cross-node, throughput < 89 tok/s (blocked)', () => {
        const ctx = buildInferenceContext({
          ...objLatency.setup,
          tensorParallel: 12,
        });
        expect(ctx).not.toBeNull();
        expect(ctx!.success).toBe(true);
        expect(ctx!.throughput?.tokensPerSecond).toBeLessThan(89);
        expect(checkAnyCriterionFails(ctx, objLatency.winningCriteria)).toBe(true);
      });

      it('TP=9 → still cross-node, throughput < 89 tok/s (blocked)', () => {
        const ctx = buildInferenceContext({
          ...objLatency.setup,
          tensorParallel: 9,
        });
        expect(ctx).not.toBeNull();
        expect(ctx!.success).toBe(true);
        expect(ctx!.throughput?.tokensPerSecond).toBeLessThan(89);
        expect(checkAnyCriterionFails(ctx, objLatency.winningCriteria)).toBe(true);
      });

      it('TP=8 → 2 replicas, throughput > 89 tok/s, all criteria pass', () => {
        const ctx = buildInferenceContext({
          ...objLatency.setup,
          tensorParallel: 8,
        });
        expect(ctx).not.toBeNull();
        expect(ctx!.success).toBe(true);
        expect(ctx!.latency?.tpot).toBeLessThan(23);
        expect(ctx!.throughput?.tokensPerSecond).toBeGreaterThan(89);
        expect(checkAllCriteria(ctx, objLatency.winningCriteria)).toBe(true);
      });
    });
  });
});
