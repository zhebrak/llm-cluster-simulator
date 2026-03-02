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
  SimulationEngine,
  type SimulationConfig,
} from '../../src/core/simulation/engine.ts';
import {
  runInferenceSimulation,
} from '../../src/core/inference/simulation.ts';
import {
  createMultiNodeCluster,
  createSingleNodeCluster,
} from '../../src/core/hardware/index.ts';
import { evaluateCriterion } from '../../src/game/validation.ts';
import { getGPUHourlyRate, calculateCostPerMillionTokens } from '../../src/core/cost/cloud.ts';
import { TRAINING_DEFAULTS, INFERENCE_DEFAULTS } from '../../src/game/defaults.ts';
import type { TaskSetup, WinningCriterion } from '../../src/game/types.ts';

// ── Helpers ──────────────────────────────────────────────────────────

function makeCluster(gpuId: string, numGPUs: number, gpusPerNode: number) {
  const numNodes = Math.ceil(numGPUs / gpusPerNode);
  if (numNodes === 1) return createSingleNodeCluster(gpuId, numGPUs);
  return createMultiNodeCluster(gpuId, gpusPerNode, numNodes);
}

function buildEffectiveTrainingConfig(setup: TaskSetup) {
  return {
    modelId: setup.modelId,
    gpuId: setup.gpuId,
    numGPUs: setup.numGPUs ?? 1,
    gpusPerNode: setup.gpusPerNode ?? Math.min(setup.numGPUs ?? 1, 8),
    strategyType: setup.strategyType ?? 'ddp',
    mixedPrecision: setup.mixedPrecision ?? TRAINING_DEFAULTS.mixedPrecision,
    sequenceLength: setup.sequenceLength ?? TRAINING_DEFAULTS.sequenceLength,
    activationCheckpointing: setup.activationCheckpointing ?? TRAINING_DEFAULTS.activationCheckpointing,
    checkpointingGranularity: setup.checkpointingGranularity ?? TRAINING_DEFAULTS.checkpointingGranularity,
    flashAttention: setup.flashAttention ?? TRAINING_DEFAULTS.flashAttention,
    globalBatchSize: setup.globalBatchSize ?? TRAINING_DEFAULTS.globalBatchSize,
    microBatchSize: setup.microBatchSize ?? TRAINING_DEFAULTS.microBatchSize,
    sequenceParallel: setup.sequenceParallel ?? TRAINING_DEFAULTS.sequenceParallel,
    finetuningMethod: setup.finetuningMethod ?? TRAINING_DEFAULTS.finetuningMethod,
    loraRank: setup.loraRank ?? TRAINING_DEFAULTS.loraRank,
    loraTargetModules: setup.loraTargetModules ?? TRAINING_DEFAULTS.loraTargetModules,
    tpDegree: setup.tpDegree ?? TRAINING_DEFAULTS.tpDegree,
    ppDegree: setup.ppDegree ?? TRAINING_DEFAULTS.ppDegree,
    epDegree: setup.epDegree ?? TRAINING_DEFAULTS.epDegree,
    cpDegree: setup.cpDegree ?? TRAINING_DEFAULTS.cpDegree,
    pipelineSchedule: setup.pipelineSchedule ?? TRAINING_DEFAULTS.pipelineSchedule,
    interleavedStages: setup.interleavedStages ?? TRAINING_DEFAULTS.interleavedStages,
  };
}

function buildEffectiveInferenceConfig(setup: TaskSetup) {
  return {
    modelId: setup.modelId,
    gpuId: setup.gpuId,
    numGPUs: setup.numGPUs ?? 1,
    weightPrecision: setup.weightPrecision ?? INFERENCE_DEFAULTS.weightPrecision,
    kvCachePrecision: setup.kvCachePrecision ?? INFERENCE_DEFAULTS.kvCachePrecision,
    batchSize: setup.batchSize ?? INFERENCE_DEFAULTS.batchSize,
    inputSeqLen: setup.inputSeqLen ?? INFERENCE_DEFAULTS.inputSeqLen,
    outputSeqLen: setup.outputSeqLen ?? INFERENCE_DEFAULTS.outputSeqLen,
    flashAttention: setup.flashAttention ?? INFERENCE_DEFAULTS.flashAttention,
    pagedAttention: setup.pagedAttention ?? INFERENCE_DEFAULTS.pagedAttention,
    continuousBatching: setup.continuousBatching ?? INFERENCE_DEFAULTS.continuousBatching,
    tensorParallel: setup.tensorParallel ?? INFERENCE_DEFAULTS.tensorParallel,
    expertParallel: setup.expertParallel ?? INFERENCE_DEFAULTS.expertParallel,
    speculativeDecoding: setup.speculativeDecoding ?? INFERENCE_DEFAULTS.speculativeDecoding,
    draftModelId: setup.draftModelId ?? INFERENCE_DEFAULTS.draftModelId,
    numSpeculativeTokens: setup.numSpeculativeTokens ?? INFERENCE_DEFAULTS.numSpeculativeTokens,
    acceptanceRate: setup.acceptanceRate ?? INFERENCE_DEFAULTS.acceptanceRate,
  };
}

function runTraining(cfg: ReturnType<typeof buildEffectiveTrainingConfig>) {
  const cluster = makeCluster(cfg.gpuId, cfg.numGPUs, cfg.gpusPerNode);
  const engine = new SimulationEngine();

  const strategyConfig: Record<string, unknown> = {};
  strategyConfig.tp = cfg.tpDegree;
  strategyConfig.pp = cfg.ppDegree;
  strategyConfig.ep = cfg.epDegree;
  strategyConfig.cp = cfg.cpDegree;
  strategyConfig.sequenceParallel = cfg.sequenceParallel;
  if (cfg.pipelineSchedule !== '1f1b') strategyConfig.pipelineSchedule = cfg.pipelineSchedule;
  if (cfg.interleavedStages !== 2) strategyConfig.interleavedStages = cfg.interleavedStages;

  const config: SimulationConfig = {
    modelId: cfg.modelId,
    clusterConfig: cluster,
    sequenceLength: cfg.sequenceLength,
    globalBatchSize: cfg.globalBatchSize,
    microBatchSize: cfg.microBatchSize,
    strategyType: cfg.strategyType as SimulationConfig['strategyType'],
    strategyConfig: Object.keys(strategyConfig).length > 0 ? strategyConfig : undefined,
    activationCheckpointing: cfg.activationCheckpointing,
    checkpointingGranularity: cfg.checkpointingGranularity,
    flashAttention: cfg.flashAttention,
    mixedPrecision: cfg.mixedPrecision as SimulationConfig['mixedPrecision'],
    finetuningMethod: cfg.finetuningMethod as SimulationConfig['finetuningMethod'],
    loraRank: cfg.loraRank,
    loraTargetModules: cfg.loraTargetModules as SimulationConfig['loraTargetModules'],
  };
  engine.configure(config);
  return engine.simulate();
}

function runInference(cfg: ReturnType<typeof buildEffectiveInferenceConfig>) {
  return runInferenceSimulation({
    modelId: cfg.modelId,
    gpuId: cfg.gpuId,
    numGPUs: cfg.numGPUs,
    batchSize: cfg.batchSize,
    inputSeqLen: cfg.inputSeqLen,
    outputSeqLen: cfg.outputSeqLen,
    weightPrecision: cfg.weightPrecision,
    kvCachePrecision: cfg.kvCachePrecision,
    flashAttention: cfg.flashAttention,
    pagedAttention: cfg.pagedAttention,
    continuousBatching: cfg.continuousBatching,
    tensorParallel: cfg.tensorParallel,
    expertParallel: cfg.expertParallel,
    speculativeEnabled: cfg.speculativeDecoding,
    draftModelId: cfg.draftModelId ?? undefined,
    numSpeculativeTokens: cfg.numSpeculativeTokens,
    acceptanceRate: cfg.acceptanceRate,
  });
}

function buildTrainingContext(setup: TaskSetup) {
  const cfg = buildEffectiveTrainingConfig(setup);
  const metrics = runTraining(cfg);
  return { success: metrics.memoryUtilization <= 1.0, ...metrics };
}

function buildInferenceContext(setup: TaskSetup) {
  const cfg = buildEffectiveInferenceConfig(setup);
  const result = runInference(cfg);
  if (!result) return null;
  const rate = getGPUHourlyRate(cfg.gpuId).rate;
  const memUtil = result.utilization?.memoryCapacityUtilization ?? 0;
  return {
    success: memUtil <= 1.0,
    ...result,
    numGPUs: cfg.numGPUs,
    memoryUtilization: memUtil,
    costPerMillionTokens: calculateCostPerMillionTokens(
      rate, cfg.numGPUs, result.throughput?.tokensPerSecond ?? 0,
    ),
  };
}

function checkAllCriteria(ctx: object | null, criteria: WinningCriterion[]): boolean {
  if (!ctx) return false;
  return criteria.every(c => evaluateCriterion(ctx, c));
}

function checkAnyCriterionFails(ctx: object | null, criteria: WinningCriterion[]): boolean {
  if (!ctx) return true;
  return criteria.some(c => !evaluateCriterion(ctx, c));
}

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
    expect(m.prerequisites).toEqual(['mission-2-11']);
    expect(isMissionUnlocked(m, ['mission-2-11'])).toBe(true);
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

  it('3-5 requires 3-2 + 3-3 + 3-4', () => {
    const m = getMissionById('mission-3-5')!;
    expect(m.prerequisites).toEqual(['mission-3-2', 'mission-3-3', 'mission-3-4']);
    expect(isMissionUnlocked(m, ['mission-3-2', 'mission-3-3', 'mission-3-4'])).toBe(true);
    expect(isMissionUnlocked(m, ['mission-3-2', 'mission-3-3'])).toBe(false);
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
    expect(m.winningCriteria).toEqual([]);
  });

  it('3-6 is a multi-objective mission with 4 objectives', () => {
    const m = getMissionById('mission-3-6')!;
    expect(m.objectives).toBeDefined();
    expect(m.objectives!.length).toBe(4);
    expect(m.objectives![0].id).toBe('obj-longctx-train');
    expect(m.objectives![1].id).toBe('obj-moe-train');
    expect(m.objectives![2].id).toBe('obj-feed-infer');
    expect(m.objectives![3].id).toBe('obj-latency-infer');
    expect(m.winningCriteria).toEqual([]);
  });
});

// ── Calibration tests ────────────────────────────────────────────────

describe('Arc 3 Mission Calibration', () => {
  describe('Mission 3-1: Landfall (continuous batching)', () => {
    const mission = getMissionById('mission-3-1')!;

    it('default setup (CB=off) → throughput < 1700 tok/s', () => {
      const ctx = buildInferenceContext(mission.setup);
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(true);
      expect(ctx!.throughput?.tokensPerSecond).toBeLessThan(1700);
      expect(checkAnyCriterionFails(ctx, mission.winningCriteria)).toBe(true);
    });

    it('CB=on → throughput > 1700 tok/s, all criteria pass', () => {
      const ctx = buildInferenceContext({ ...mission.setup, continuousBatching: true });
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(true);
      expect(ctx!.throughput?.tokensPerSecond).toBeGreaterThan(1700);
      expect(checkAllCriteria(ctx, mission.winningCriteria)).toBe(true);
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
  });

  describe('Mission 3-3: Alien Model (expert parallelism)', () => {
    const mission = getMissionById('mission-3-3')!;

    it('default setup (EP=1) → MFU < 30%', () => {
      const ctx = buildTrainingContext(mission.setup);
      expect(ctx.success).toBe(true);
      expect(ctx.mfu).toBeLessThan(0.30);
      expect(checkAnyCriterionFails(ctx, mission.winningCriteria)).toBe(true);
    });

    it('EP=1 bypass: TP/PP/GBS combos stay below 30% MFU (physics guard)', () => {
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
          expect(ctx.mfu).toBeLessThan(0.30);
        }
      }
    });

    it('EP=1 bypass: MBS=4 reaches >30% MFU (guarded by expectedChanges)', () => {
      // Increasing MBS from 2 to 4 at EP=1 pushes MFU above 30% — confirmed bypass.
      // This is blocked by the microBatchSize=unchanged guard in expectedChanges.
      const ctx = buildTrainingContext({
        ...mission.setup, tpDegree: 8, ppDegree: 1, epDegree: 1, microBatchSize: 4,
      });
      expect(ctx.success).toBe(true);
      expect(ctx.mfu).toBeGreaterThan(0.30);
    });

    it('EP=8 → MFU > 30%, all criteria pass', () => {
      const ctx = buildTrainingContext({ ...mission.setup, epDegree: 8 });
      expect(ctx.success).toBe(true);
      expect(ctx.mfu).toBeGreaterThan(0.30);
      expect(checkAllCriteria(ctx, mission.winningCriteria)).toBe(true);
    });
  });

  describe('Mission 3-4: Big Train (pipeline schedule)', () => {
    const mission = getMissionById('mission-3-4')!;

    it('default setup (1F1B v=1) → MFU < 37%', () => {
      const ctx = buildTrainingContext(mission.setup);
      expect(ctx.success).toBe(true);
      expect(ctx.mfu).toBeLessThan(0.37);
      expect(checkAnyCriterionFails(ctx, mission.winningCriteria)).toBe(true);
    });

    it('interleaved-1f1b v=4 → MFU > 37%, all criteria pass', () => {
      const ctx = buildTrainingContext({
        ...mission.setup,
        pipelineSchedule: 'interleaved-1f1b',
        interleavedStages: 4,
      });
      expect(ctx.success).toBe(true);
      expect(ctx.mfu).toBeGreaterThan(0.37);
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

      it('FP8 + AC → fits + MFU > 50%', () => {
        const ctx = buildTrainingContext({
          ...objBiosig.setup,
          mixedPrecision: 'fp8',
          activationCheckpointing: true,
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
      it('default setup (BF16 EP=1) → MFU < 40%', () => {
        const ctx = buildTrainingContext(objMoe.setup);
        expect(ctx.success).toBe(true);
        expect(ctx.mfu).toBeLessThan(0.40);
        expect(checkAnyCriterionFails(ctx, objMoe.winningCriteria)).toBe(true);
      });

      it('FP8 + EP=8 → MFU > 40%', () => {
        const ctx = buildTrainingContext({
          ...objMoe.setup,
          mixedPrecision: 'fp8',
          epDegree: 8,
        });
        expect(ctx.success).toBe(true);
        expect(ctx.mfu).toBeGreaterThan(0.40);
        expect(checkAllCriteria(ctx, objMoe.winningCriteria)).toBe(true);
      });
    });

    describe('Objective 3 — Inference: High-throughput probe feed', () => {
      it('default setup (BF16 CB=off) → throughput < 4000 tok/s', () => {
        const ctx = buildInferenceContext(objFeed.setup);
        expect(ctx).not.toBeNull();
        expect(ctx!.success).toBe(true);
        expect(ctx!.throughput?.tokensPerSecond).toBeLessThan(4000);
        expect(checkAnyCriterionFails(ctx, objFeed.winningCriteria)).toBe(true);
      });

      it('FP8 + CB=on → throughput > 4000 tok/s', () => {
        const ctx = buildInferenceContext({
          ...objFeed.setup,
          weightPrecision: 'fp8',
          continuousBatching: true,
        });
        expect(ctx).not.toBeNull();
        expect(ctx!.success).toBe(true);
        expect(ctx!.throughput?.tokensPerSecond).toBeGreaterThan(4000);
        expect(checkAllCriteria(ctx, objFeed.winningCriteria)).toBe(true);
      });
    });

    describe('Objective 4 — Inference: Low-latency translation', () => {
      // Note: TP=16 default already meets TPOT < 23ms (cross-node TP is fast enough
      // for small batch decode). The expectedChanges guard is what enforces the lesson —
      // the player must decrease TP to demonstrate understanding of intra-node topology.
      it('default setup (TP=16) → model loads successfully', () => {
        const ctx = buildInferenceContext(objLatency.setup);
        expect(ctx).not.toBeNull();
        expect(ctx!.success).toBe(true);
      });

      it('TP=8 → TPOT < 23ms, all criteria pass', () => {
        const ctx = buildInferenceContext({
          ...objLatency.setup,
          tensorParallel: 8,
        });
        expect(ctx).not.toBeNull();
        expect(ctx!.success).toBe(true);
        expect(ctx!.latency?.tpot).toBeLessThan(23);
        expect(checkAllCriteria(ctx, objLatency.winningCriteria)).toBe(true);
      });
    });
  });
});
