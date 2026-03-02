/**
 * Arc 2 mission calibration tests
 *
 * Validates:
 *   1. DAG structure for Arc 2 missions
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

describe('Arc 2 DAG Structure', () => {
  it('Arc 2 has 11 missions', () => {
    const missions = getMissionsForArc('arc2-discovery');
    expect(missions.length).toBe(11);
  });

  it('missions sorted by order', () => {
    const missions = getMissionsForArc('arc2-discovery');
    for (let i = 1; i < missions.length; i++) {
      expect(missions[i].order).toBeGreaterThanOrEqual(missions[i - 1].order);
    }
  });

  it('2-1 requires 1-8 (signal pivot)', () => {
    const m = getMissionById('mission-2-1')!;
    expect(m.prerequisites).toEqual(['mission-1-8']);
    expect(isMissionUnlocked(m, ['mission-1-8'])).toBe(true);
    expect(isMissionUnlocked(m, [])).toBe(false);
  });

  it('2-2 and 2-3 are parallel after 2-1', () => {
    const m22 = getMissionById('mission-2-2')!;
    const m23 = getMissionById('mission-2-3')!;
    expect(m22.prerequisites).toEqual(['mission-2-1']);
    expect(m23.prerequisites).toEqual(['mission-2-1']);
  });

  it('2-4 requires 1-6 (Archive Vault)', () => {
    const m = getMissionById('mission-2-4')!;
    expect(m.prerequisites).toEqual(['mission-1-6']);
  });

  it('2-5 requires 2-4, 2-6 requires 2-2 and 2-4', () => {
    expect(getMissionById('mission-2-5')!.prerequisites).toEqual(['mission-2-4']);
    expect(getMissionById('mission-2-6')!.prerequisites).toEqual(['mission-2-2', 'mission-2-4']);
  });

  it('2-7 requires 1-7, 2-2, and 2-4', () => {
    const m = getMissionById('mission-2-7')!;
    expect(m.prerequisites).toEqual(['mission-1-7', 'mission-2-2', 'mission-2-4']);
    expect(isMissionUnlocked(m, ['mission-1-7', 'mission-2-2', 'mission-2-4'])).toBe(true);
    expect(isMissionUnlocked(m, ['mission-1-7', 'mission-2-2'])).toBe(false);
  });

  it('2-8 requires 2-7, 2-9 requires 2-8', () => {
    expect(getMissionById('mission-2-8')!.prerequisites).toEqual(['mission-2-7']);
    expect(getMissionById('mission-2-9')!.prerequisites).toEqual(['mission-2-8']);
  });

  it('2-10 requires 2-5 and 2-7', () => {
    expect(getMissionById('mission-2-10')!.prerequisites).toEqual(['mission-2-5', 'mission-2-7']);
  });

  it('2-11 (pivot) requires 2-7, 2-8, 2-10 — NOT 2-9', () => {
    const m = getMissionById('mission-2-11')!;
    expect(m.type).toBe('pivot');
    expect(m.prerequisites).toEqual(['mission-2-7', 'mission-2-10']);
    expect(m.prerequisites).not.toContain('mission-2-9');
  });

  it('2-10 is a multi-objective mission', () => {
    const m = getMissionById('mission-2-10')!;
    expect(m.objectives).toBeDefined();
    expect(m.objectives!.length).toBe(2);
    expect(m.objectives![0].id).toBe('obj-train');
    expect(m.objectives![1].id).toBe('obj-infer');
    expect(m.winningCriteria).toEqual([]);
  });
});

// ── Calibration tests ────────────────────────────────────────────────

describe('Arc 2 Mission Calibration', () => {
  describe('Mission 2-1: First Light (TP for latency)', () => {
    const mission = getMissionById('mission-2-1')!;

    it('default setup (TP=2) → TTFT > 200ms', () => {
      const ctx = buildInferenceContext(mission.setup);
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(true);
      expect(ctx!.latency?.ttft).toBeGreaterThan(200);
      expect(checkAnyCriterionFails(ctx, mission.winningCriteria)).toBe(true);
    });

    it('TP=4 → TTFT < 200ms, all criteria pass', () => {
      const ctx = buildInferenceContext({ ...mission.setup, tensorParallel: 4 });
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(true);
      expect(ctx!.latency?.ttft).toBeLessThan(200);
      expect(checkAllCriteria(ctx, mission.winningCriteria)).toBe(true);
    });
  });

  describe('Mission 2-2: The Derelict (replica scaling)', () => {
    const mission = getMissionById('mission-2-2')!;

    it('default setup (TP=8) → 1 replica (fails numReplicas ≥ 3)', () => {
      const ctx = buildInferenceContext(mission.setup);
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(true);
      expect(ctx!.numReplicas).toBe(1);
      expect(checkAnyCriterionFails(ctx, mission.winningCriteria)).toBe(true);
    });

    it('TP=2 → 4 replicas + TTFT < 300ms', () => {
      const ctx = buildInferenceContext({ ...mission.setup, tensorParallel: 2 });
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(true);
      expect(ctx!.numReplicas).toBe(4);
      expect(ctx!.latency?.ttft).toBeLessThan(300);
      expect(checkAllCriteria(ctx, mission.winningCriteria)).toBe(true);
    });
  });

  describe('Mission 2-3: Ghost Writer (speculative decoding)', () => {
    const mission = getMissionById('mission-2-3')!;

    it('default setup (spec off) → no speedup field', () => {
      const ctx = buildInferenceContext(mission.setup);
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(true);
      // speculative is undefined when disabled, so criterion fails
      expect(checkAnyCriterionFails(ctx, mission.winningCriteria)).toBe(true);
    });

    it('spec on + draft model → speedup > 1.5', () => {
      const ctx = buildInferenceContext({
        ...mission.setup,
        speculativeDecoding: true,
        draftModelId: 'llama3.2-1b',
      });
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(true);
      expect(ctx!.speculative?.speedup).toBeGreaterThan(1.5);
      expect(checkAllCriteria(ctx, mission.winningCriteria)).toBe(true);
    });
  });

  describe('Mission 2-4: The Weight of Memory (FP32→BF16 + DDP→FSDP)', () => {
    const mission = getMissionById('mission-2-4')!;

    it('default setup (FP32 + DDP) → OOM', () => {
      const ctx = buildTrainingContext(mission.setup);
      expect(ctx.success).toBe(false);
      expect(ctx.memoryUtilization).toBeGreaterThan(1.0);
    });

    it('BF16 + DDP → still OOM', () => {
      const ctx = buildTrainingContext({ ...mission.setup, mixedPrecision: 'bf16' });
      expect(ctx.success).toBe(false);
      expect(ctx.memoryUtilization).toBeGreaterThan(1.0);
    });

    it('FP32 + FSDP → fits but MFU too low', () => {
      const ctx = buildTrainingContext({
        ...mission.setup,
        strategyType: 'fsdp',
      });
      expect(ctx.success).toBe(true);
      expect(ctx.mfu).toBeLessThan(0.30);
      expect(checkAnyCriterionFails(ctx, mission.winningCriteria)).toBe(true);
    });

    it('BF16 + FSDP → success + MFU > 30%', () => {
      const ctx = buildTrainingContext({
        ...mission.setup,
        mixedPrecision: 'bf16',
        strategyType: 'fsdp',
      });
      expect(ctx.success).toBe(true);
      expect(ctx.mfu).toBeGreaterThan(0.30);
      expect(ctx.memoryUtilization).toBeLessThan(1.0);
      expect(checkAllCriteria(ctx, mission.winningCriteria)).toBe(true);
    });
  });

  describe('Mission 2-5: Activation Avalanche (activation checkpointing)', () => {
    const mission = getMissionById('mission-2-5')!;

    it('default setup (no AC, seqLen=12288) → OOM', () => {
      const ctx = buildTrainingContext(mission.setup);
      expect(ctx.success).toBe(false);
      expect(ctx.memoryUtilization).toBeGreaterThan(1.0);
    });

    it('AC enabled → success', () => {
      const ctx = buildTrainingContext({
        ...mission.setup,
        activationCheckpointing: true,
      });
      expect(ctx.success).toBe(true);
      expect(ctx.memoryUtilization).toBeLessThan(1.0);
      expect(checkAllCriteria(ctx, mission.winningCriteria)).toBe(true);
    });
  });

  describe('Mission 2-6: The Adapter (LoRA)', () => {
    const mission = getMissionById('mission-2-6')!;

    it('default setup (full fine-tune 70B) → OOM', () => {
      const ctx = buildTrainingContext(mission.setup);
      expect(ctx.success).toBe(false);
      expect(ctx.memoryUtilization).toBeGreaterThan(1.0);
    });

    it('LoRA → success', () => {
      const ctx = buildTrainingContext({
        ...mission.setup,
        finetuningMethod: 'lora',
      });
      expect(ctx.success).toBe(true);
      expect(ctx.memoryUtilization).toBeLessThan(1.0);
      expect(checkAllCriteria(ctx, mission.winningCriteria)).toBe(true);
    });

    it('QLoRA → success', () => {
      const ctx = buildTrainingContext({
        ...mission.setup,
        finetuningMethod: 'qlora',
      });
      expect(ctx.success).toBe(true);
      expect(ctx.memoryUtilization).toBeLessThan(1.0);
      expect(checkAllCriteria(ctx, mission.winningCriteria)).toBe(true);
    });
  });

  describe('Mission 2-7: The Shipment (FP8 for large models)', () => {
    const mission = getMissionById('mission-2-7')!;

    it('default setup (405B BF16 TP=8) → OOM', () => {
      const ctx = buildInferenceContext(mission.setup);
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(false);
    });

    it('FP8 → success + throughput > 30 tok/s', () => {
      const ctx = buildInferenceContext({
        ...mission.setup,
        weightPrecision: 'fp8',
      });
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(true);
      expect(ctx!.throughput?.tokensPerSecond).toBeGreaterThan(30);
      expect(checkAllCriteria(ctx, mission.winningCriteria)).toBe(true);
    });
  });

  describe('Mission 2-8: Bandwidth Wall (network topology)', () => {
    const mission = getMissionById('mission-2-8')!;

    it('default setup (BF16 TP=16) → throughput < 80 tok/s', () => {
      const ctx = buildInferenceContext(mission.setup);
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(true);
      expect(ctx!.throughput?.tokensPerSecond).toBeLessThan(80);
      expect(checkAnyCriterionFails(ctx, mission.winningCriteria)).toBe(true);
    });

    it('FP8 TP=8 → 2 replicas, throughput > 80 tok/s', () => {
      const ctx = buildInferenceContext({
        ...mission.setup,
        weightPrecision: 'fp8',
        tensorParallel: 8,
      });
      expect(ctx).not.toBeNull();
      expect(ctx!.success).toBe(true);
      expect(ctx!.numReplicas).toBe(2);
      expect(ctx!.throughput?.tokensPerSecond).toBeGreaterThan(80);
      expect(checkAllCriteria(ctx, mission.winningCriteria)).toBe(true);
    });
  });

  describe('Mission 2-9: The Pipeline (pipeline parallelism)', () => {
    const mission = getMissionById('mission-2-9')!;

    it('default setup (PP=1) → success but MFU < 39%', () => {
      const ctx = buildTrainingContext(mission.setup);
      expect(ctx.success).toBe(true);
      expect(ctx.mfu).toBeLessThan(0.39);
      expect(checkAnyCriterionFails(ctx, mission.winningCriteria)).toBe(true);
    });

    it('PP=2 → success + MFU > 39%', () => {
      const ctx = buildTrainingContext({
        ...mission.setup,
        ppDegree: 2,
      });
      expect(ctx.success).toBe(true);
      expect(ctx.mfu).toBeGreaterThan(0.39);
      expect(checkAllCriteria(ctx, mission.winningCriteria)).toBe(true);
    });
  });

  describe('Mission 2-10: The Protein Problem (multi-objective)', () => {
    const mission = getMissionById('mission-2-10')!;
    const objTrain = mission.objectives![0];
    const objInfer = mission.objectives![1];

    describe('Objective 1 — Training: Protein model fine-tune', () => {
      it('default setup (BF16 MBS=8 no AC) → OOM', () => {
        const ctx = buildTrainingContext(objTrain.setup);
        expect(ctx.success).toBe(false);
        expect(ctx.memoryUtilization).toBeGreaterThan(1.0);
        expect(checkAnyCriterionFails(ctx, objTrain.winningCriteria)).toBe(true);
      });

      it('BF16 + AC → fits but MFU < 50%', () => {
        const ctx = buildTrainingContext({
          ...objTrain.setup,
          activationCheckpointing: true,
        });
        expect(ctx.success).toBe(true);
        expect(ctx.mfu).toBeLessThan(0.50);
      });

      it('FP8 + AC → fits + MFU > 50%', () => {
        const ctx = buildTrainingContext({
          ...objTrain.setup,
          mixedPrecision: 'fp8',
          activationCheckpointing: true,
        });
        expect(ctx.success).toBe(true);
        expect(ctx.mfu).toBeGreaterThan(0.50);
        expect(checkAllCriteria(ctx, objTrain.winningCriteria)).toBe(true);
      });
    });

    describe('Objective 2 — Inference: Signal analysis', () => {
      it('default setup (BF16 TP=2) → TTFT > 200ms', () => {
        const ctx = buildInferenceContext(objInfer.setup);
        expect(ctx).not.toBeNull();
        expect(ctx!.success).toBe(true);
        expect(ctx!.latency?.ttft).toBeGreaterThan(200);
        expect(checkAnyCriterionFails(ctx, objInfer.winningCriteria)).toBe(true);
      });

      it('INT8 TP=4 → TTFT < 200ms', () => {
        const ctx = buildInferenceContext({
          ...objInfer.setup,
          weightPrecision: 'int8',
          tensorParallel: 4,
        });
        expect(ctx).not.toBeNull();
        expect(ctx!.success).toBe(true);
        expect(ctx!.latency?.ttft).toBeLessThan(200);
        expect(checkAllCriteria(ctx, objInfer.winningCriteria)).toBe(true);
      });
    });
  });
});
