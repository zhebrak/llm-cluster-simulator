/**
 * Diagnostic test: prints detailed context values for all failing calibration tests.
 * Run: npx vitest run tests/game/calibration-diag.test.ts
 */

import { describe, it } from 'vitest';
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
import { ALL_TASKS, getTaskById } from '../../src/game/tasks/index.ts';
import { evaluateCriterion, resolvePath } from '../../src/game/validation.ts';
import { calculateCostPerMillionTokens, getGPUHourlyRate } from '../../src/core/cost/cloud.ts';
import { TRAINING_DEFAULTS, INFERENCE_DEFAULTS } from '../../src/game/defaults.ts';
import type { TaskSetup, WinningCriterion } from '../../src/game/types.ts';

// ── Helpers (same as calibration.test.ts) ──

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
  };
}

function runTraining(cfg: ReturnType<typeof buildEffectiveTrainingConfig>) {
  const gpusPerNode = cfg.gpusPerNode;
  const cluster = makeCluster(cfg.gpuId, cfg.numGPUs, gpusPerNode);
  const engine = new SimulationEngine();

  const strategyConfig: Record<string, unknown> = {};
  strategyConfig.tp = cfg.tpDegree;
  strategyConfig.pp = cfg.ppDegree;
  strategyConfig.ep = cfg.epDegree;
  strategyConfig.cp = cfg.cpDegree;
  if (cfg.sequenceParallel) strategyConfig.sequenceParallel = true;
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
    speculativeDecoding: cfg.speculativeDecoding,
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
  return {
    ...result,
    memoryUtilization: result.utilization?.memoryCapacityUtilization,
    costPerMillionTokens: calculateCostPerMillionTokens(
      rate, cfg.numGPUs, result.throughput?.tokensPerSecond ?? 0,
    ),
  };
}

function printCriteria(ctx: object | null, criteria: WinningCriterion[]) {
  if (!ctx) {
    console.log('    Context is NULL (sim returned null)');
    return;
  }
  for (const c of criteria) {
    const actual = resolvePath(ctx, c.field);
    const met = evaluateCriterion(ctx, c);
    console.log(`    ${met ? 'PASS' : 'FAIL'} ${c.field} ${c.operator} ${c.value}: actual = ${actual}`);
  }
}

// ── Winning override map (same as calibration.test.ts) ──

const WINNING_OVERRIDES: Record<string, Partial<TaskSetup>> = {
  'training-beginner-01': {},
  'training-beginner-02': { mixedPrecision: 'bf16' },
  'training-beginner-03': { mixedPrecision: 'bf16' },
  'training-beginner-04': { activationCheckpointing: true },
  'training-beginner-05': { flashAttention: true },
  'training-beginner-06': {},
  'training-beginner-07': { microBatchSize: 2, globalBatchSize: 128 },
  'training-beginner-08': { microBatchSize: 2, globalBatchSize: 256 },
  'training-beginner-09': { strategyType: 'fsdp' },
  'training-beginner-10': { microBatchSize: 4, globalBatchSize: 256 },
  'training-intermediate-01': { strategyType: 'fsdp-tp', tpDegree: 4 },
  'training-intermediate-02': { tpDegree: 4, globalBatchSize: 128 },
  'training-intermediate-03': { tpDegree: 4, globalBatchSize: 128 },
  'training-intermediate-04': { tpDegree: 8, ppDegree: 4 },
  'training-intermediate-05': { tpDegree: 8, ppDegree: 4, globalBatchSize: 256 },
  'training-intermediate-06': { tpDegree: 8, ppDegree: 8, pipelineSchedule: 'interleaved-1f1b', interleavedStages: 4, globalBatchSize: 128 },
  'training-intermediate-07': { sequenceParallel: true, tpDegree: 4 },
  'training-intermediate-08': { tpDegree: 4, globalBatchSize: 256 },
  'training-intermediate-09': { globalBatchSize: 128 },
  'training-intermediate-10': { mixedPrecision: 'fp8', tpDegree: 4 },
  'training-advanced-01': { tpDegree: 8, ppDegree: 4, globalBatchSize: 256, pipelineSchedule: 'interleaved-1f1b' },
  'training-advanced-02': { tpDegree: 2, epDegree: 8, mixedPrecision: 'fp8' },
  'training-advanced-03': { tpDegree: 8, ppDegree: 4, cpDegree: 2 },
  'training-advanced-04': { tpDegree: 8, sequenceParallel: true, microBatchSize: 2, globalBatchSize: 256 },
  'training-advanced-05': { finetuningMethod: 'lora' },
  'training-advanced-06': { finetuningMethod: 'qlora' },
  'training-advanced-07': { mixedPrecision: 'fp8', tpDegree: 4, ppDegree: 8, epDegree: 32, globalBatchSize: 8192, microBatchSize: 2 },
  'training-advanced-08': { tpDegree: 8, ppDegree: 12, pipelineSchedule: 'interleaved-1f1b', interleavedStages: 8, checkpointingGranularity: 'selective', globalBatchSize: 256 },
  'training-advanced-09': { tpDegree: 8, microBatchSize: 2, globalBatchSize: 128 },
  'training-advanced-10': { tpDegree: 8, ppDegree: 8, pipelineSchedule: 'interleaved-1f1b', interleavedStages: 2, globalBatchSize: 512 },
  'inference-beginner-01': {},
  'inference-beginner-02': { weightPrecision: 'int8' },
  'inference-beginner-03': {},
  'inference-beginner-04': { weightPrecision: 'int8' },
  'inference-beginner-05': { batchSize: 8 },
  'inference-beginner-06': { batchSize: 1 },
  'inference-beginner-07': { flashAttention: true },
  'inference-beginner-08': { kvCachePrecision: 'fp8' },
  'inference-beginner-09': { pagedAttention: true },
  'inference-beginner-10': { continuousBatching: true, batchSize: 16 },
  'inference-intermediate-01': { tensorParallel: 4 },
  'inference-intermediate-02': { tensorParallel: 8 },
  'inference-intermediate-03': { tensorParallel: 8, weightPrecision: 'fp8' },
  'inference-intermediate-04': { tensorParallel: 2 },
  'inference-intermediate-05': { tensorParallel: 4, batchSize: 16 },
  'inference-intermediate-06': { tensorParallel: 8, batchSize: 32 },
  'inference-intermediate-07': { tensorParallel: 4 },
  'inference-intermediate-08': {},
  'inference-intermediate-09': { tensorParallel: 4, speculativeDecoding: true },
  'inference-intermediate-10': { tensorParallel: 4, weightPrecision: 'fp8' },
  'inference-advanced-01': { tensorParallel: 8, weightPrecision: 'fp8' },
  'inference-advanced-02': { tensorParallel: 8, expertParallel: 2, weightPrecision: 'fp8', batchSize: 8 },
  'inference-advanced-03': { tensorParallel: 8, weightPrecision: 'fp8', speculativeDecoding: true },
  'inference-advanced-04': { tensorParallel: 4, speculativeDecoding: true },
  'inference-advanced-05': { tensorParallel: 2, weightPrecision: 'fp8', batchSize: 32 },
  'inference-advanced-06': { tensorParallel: 2, weightPrecision: 'fp8', batchSize: 32 },
  'inference-advanced-07': { tensorParallel: 16, weightPrecision: 'fp8', kvCachePrecision: 'fp8' },
  'inference-advanced-08': { tensorParallel: 8, weightPrecision: 'fp8', speculativeDecoding: true },
  'inference-advanced-09': { tensorParallel: 4, weightPrecision: 'int4' },
  'inference-advanced-10': { tensorParallel: 8, weightPrecision: 'fp8', batchSize: 16 },
};

const TUTORIAL_TASK_IDS = new Set([
  'training-beginner-01',
  'inference-beginner-01',
  'inference-beginner-03',
]);

// ── Failing "default should NOT pass" tasks ──

const FAILING_DEFAULT_NOT_PASS = [
  'training-beginner-05', 'training-beginner-06', 'training-beginner-07',
  'training-beginner-08', 'training-beginner-10',
  'training-intermediate-03', 'training-intermediate-04', 'training-intermediate-05',
  'training-intermediate-06', 'training-intermediate-08', 'training-intermediate-09',
  'training-intermediate-10',
  'training-advanced-01', 'training-advanced-04', 'training-advanced-08',
  'inference-beginner-04', 'inference-beginner-06', 'inference-beginner-07',
  'inference-intermediate-08',
];

// ── Failing "winning should pass" tasks ──

const FAILING_WINNING_PASS = [
  'training-beginner-02', 'training-beginner-03', 'training-beginner-04',
  'training-intermediate-01', 'training-intermediate-02', 'training-intermediate-07',
  'training-advanced-02', 'training-advanced-03', 'training-advanced-08',
  'training-advanced-09',
  'inference-beginner-09',
  'inference-intermediate-09',
  'inference-advanced-01', 'inference-advanced-08',
];

// ── 1. Default should NOT pass (but does) ──

describe('DIAGNOSTIC: Default configs that WRONGLY pass all criteria', () => {
  for (const taskId of FAILING_DEFAULT_NOT_PASS) {
    it(taskId, () => {
      const task = getTaskById(taskId)!;
      console.log(`\n=== ${taskId} (default should NOT pass) ===`);
      console.log(`  Mode: ${task.mode}`);

      const effectiveSetup = task.setup;
      if (task.mode === 'training') {
        const cfg = buildEffectiveTrainingConfig(effectiveSetup);
        console.log(`  Effective config: model=${cfg.modelId}, gpu=${cfg.gpuId}, numGPUs=${cfg.numGPUs}`);
        console.log(`    strategy=${cfg.strategyType}, precision=${cfg.mixedPrecision}`);
        console.log(`    FA=${cfg.flashAttention}, AC=${cfg.activationCheckpointing}`);
        console.log(`    GBS=${cfg.globalBatchSize}, MBS=${cfg.microBatchSize}`);
        console.log(`    TP=${cfg.tpDegree}, PP=${cfg.ppDegree}, EP=${cfg.epDegree}, CP=${cfg.cpDegree}`);
        const ctx = buildTrainingContext(effectiveSetup);
        console.log(`  Results: memUtil=${ctx.memoryUtilization?.toFixed(4)}, success=${ctx.success}`);
        console.log(`    mfu=${(ctx as any).mfu?.toFixed(4)}, tokensPerSecond=${(ctx as any).tokensPerSecond?.toFixed(0)}`);
        console.log(`    pipelineBubble=${(ctx as any).pipelineBubble?.toFixed(4)}`);
        console.log(`    communicationOverhead=${(ctx as any).communicationOverhead?.toFixed(4)}`);
        console.log('  Criteria evaluation:');
        printCriteria(ctx, task.winningCriteria);
      } else {
        const cfg = buildEffectiveInferenceConfig(effectiveSetup);
        console.log(`  Effective config: model=${cfg.modelId}, gpu=${cfg.gpuId}, numGPUs=${cfg.numGPUs}`);
        console.log(`    weightPrecision=${cfg.weightPrecision}, kvPrecision=${cfg.kvCachePrecision}`);
        console.log(`    FA=${cfg.flashAttention}, PA=${cfg.pagedAttention}, CB=${cfg.continuousBatching}`);
        console.log(`    batch=${cfg.batchSize}, inputSeq=${cfg.inputSeqLen}, outputSeq=${cfg.outputSeqLen}`);
        console.log(`    TP=${cfg.tensorParallel}, EP=${cfg.expertParallel}, specDec=${cfg.speculativeDecoding}`);
        const ctx = buildInferenceContext(effectiveSetup);
        if (ctx) {
          console.log(`  Results: memUtil=${ctx.memoryUtilization?.toFixed(4)}, success=${(ctx as any).success}`);
          console.log(`    throughput=${ctx.throughput?.tokensPerSecond?.toFixed(1)}`);
          console.log(`    tpot=${ctx.latency?.tpot?.toFixed(2)}`);
          console.log(`    ttft=${ctx.latency?.ttft?.toFixed(2)}`);
          console.log(`    speculative.speedup=${ctx.speculative?.speedup?.toFixed(2)}`);
        } else {
          console.log('  Results: NULL (sim returned null)');
        }
        console.log('  Criteria evaluation:');
        printCriteria(ctx, task.winningCriteria);
      }
    });
  }
});

// ── 2. Winning should pass (but doesn't) ──

describe('DIAGNOSTIC: Winning configs that WRONGLY fail criteria', () => {
  for (const taskId of FAILING_WINNING_PASS) {
    it(taskId, () => {
      const task = getTaskById(taskId)!;
      const overrides = WINNING_OVERRIDES[taskId]!;
      const setup = { ...task.setup, ...overrides };
      console.log(`\n=== ${taskId} (winning should pass) ===`);
      console.log(`  Mode: ${task.mode}`);
      console.log(`  Overrides: ${JSON.stringify(overrides)}`);

      if (task.mode === 'training') {
        const cfg = buildEffectiveTrainingConfig(setup);
        console.log(`  Effective config: model=${cfg.modelId}, gpu=${cfg.gpuId}, numGPUs=${cfg.numGPUs}`);
        console.log(`    strategy=${cfg.strategyType}, precision=${cfg.mixedPrecision}`);
        console.log(`    FA=${cfg.flashAttention}, AC=${cfg.activationCheckpointing}`);
        console.log(`    GBS=${cfg.globalBatchSize}, MBS=${cfg.microBatchSize}`);
        console.log(`    TP=${cfg.tpDegree}, PP=${cfg.ppDegree}, EP=${cfg.epDegree}, CP=${cfg.cpDegree}`);
        console.log(`    SP=${cfg.sequenceParallel}, pipelineSched=${cfg.pipelineSchedule}, interleavedStages=${cfg.interleavedStages}`);
        console.log(`    finetuning=${cfg.finetuningMethod}, loraRank=${cfg.loraRank}, loraTargets=${cfg.loraTargetModules}`);
        const ctx = buildTrainingContext(setup);
        console.log(`  Results: memUtil=${ctx.memoryUtilization?.toFixed(4)}, success=${ctx.success}`);
        console.log(`    mfu=${(ctx as any).mfu?.toFixed(4)}, hfu=${(ctx as any).hfu?.toFixed(4)}`);
        console.log(`    tokensPerSecond=${(ctx as any).tokensPerSecond?.toFixed(0)}`);
        console.log(`    pipelineBubble=${(ctx as any).pipelineBubble?.toFixed(4)}`);
        console.log(`    communicationOverhead=${(ctx as any).communicationOverhead?.toFixed(4)}`);
        console.log('  Criteria evaluation:');
        printCriteria(ctx, task.winningCriteria);
      } else {
        const cfg = buildEffectiveInferenceConfig(setup);
        console.log(`  Effective config: model=${cfg.modelId}, gpu=${cfg.gpuId}, numGPUs=${cfg.numGPUs}`);
        console.log(`    weightPrecision=${cfg.weightPrecision}, kvPrecision=${cfg.kvCachePrecision}`);
        console.log(`    FA=${cfg.flashAttention}, PA=${cfg.pagedAttention}, CB=${cfg.continuousBatching}`);
        console.log(`    batch=${cfg.batchSize}, inputSeq=${cfg.inputSeqLen}, outputSeq=${cfg.outputSeqLen}`);
        console.log(`    TP=${cfg.tensorParallel}, EP=${cfg.expertParallel}, specDec=${cfg.speculativeDecoding}`);
        const ctx = buildInferenceContext(setup);
        if (ctx) {
          console.log(`  Results: memUtil=${ctx.memoryUtilization?.toFixed(4)}, success=${(ctx as any).success}`);
          console.log(`    throughput=${ctx.throughput?.tokensPerSecond?.toFixed(1)}`);
          console.log(`    tpot=${ctx.latency?.tpot?.toFixed(2)}`);
          console.log(`    ttft=${ctx.latency?.ttft?.toFixed(2)}`);
          console.log(`    speculative.speedup=${ctx.speculative?.speedup?.toFixed(2)}`);
          console.log(`    memory.weights=${ctx.memory?.weights?.toFixed(0)}`);
        } else {
          console.log('  Results: NULL (sim returned null)');
        }
        console.log('  Criteria evaluation:');
        printCriteria(ctx, task.winningCriteria);
      }
    });
  }
});

// ── 3. Hint progression diagnostics ──

describe('DIAGNOSTIC: Hint progressions — memory values', () => {
  it('training-beginner-02: FP32 → BF16', () => {
    const task = getTaskById('training-beginner-02')!;
    console.log('\n=== training-beginner-02 hint progression ===');
    console.log(`  Setup: ${JSON.stringify(task.setup)}`);

    // Default (FP32)
    const defaultCfg = buildEffectiveTrainingConfig(task.setup);
    console.log(`  Default config: precision=${defaultCfg.mixedPrecision}, FA=${defaultCfg.flashAttention}, AC=${defaultCfg.activationCheckpointing}`);
    const defaultCtx = buildTrainingContext(task.setup);
    console.log(`  Default memUtil=${defaultCtx.memoryUtilization?.toFixed(4)}, success=${defaultCtx.success}`);

    // Fixed (BF16)
    const fixedSetup = { ...task.setup, mixedPrecision: 'bf16' as const };
    const fixedCfg = buildEffectiveTrainingConfig(fixedSetup);
    console.log(`  Fixed config: precision=${fixedCfg.mixedPrecision}, FA=${fixedCfg.flashAttention}, AC=${fixedCfg.activationCheckpointing}`);
    const fixedCtx = buildTrainingContext(fixedSetup);
    console.log(`  Fixed memUtil=${fixedCtx.memoryUtilization?.toFixed(4)}, success=${fixedCtx.success}`);
  });

  it('training-beginner-04: BF16 → BF16+AC', () => {
    const task = getTaskById('training-beginner-04')!;
    console.log('\n=== training-beginner-04 hint progression ===');
    console.log(`  Setup: ${JSON.stringify(task.setup)}`);

    // Default (BF16 from setup, no AC)
    const defaultCfg = buildEffectiveTrainingConfig(task.setup);
    console.log(`  Default config: precision=${defaultCfg.mixedPrecision}, FA=${defaultCfg.flashAttention}, AC=${defaultCfg.activationCheckpointing}`);
    console.log(`    GBS=${defaultCfg.globalBatchSize}, MBS=${defaultCfg.microBatchSize}`);
    const defaultCtx = buildTrainingContext(task.setup);
    console.log(`  Default memUtil=${defaultCtx.memoryUtilization?.toFixed(4)}, success=${defaultCtx.success}`);

    // Fixed (BF16+AC)
    const fixedSetup = { ...task.setup, activationCheckpointing: true };
    const fixedCfg = buildEffectiveTrainingConfig(fixedSetup);
    console.log(`  Fixed config: precision=${fixedCfg.mixedPrecision}, FA=${fixedCfg.flashAttention}, AC=${fixedCfg.activationCheckpointing}`);
    const fixedCtx = buildTrainingContext(fixedSetup);
    console.log(`  Fixed memUtil=${fixedCtx.memoryUtilization?.toFixed(4)}, success=${fixedCtx.success}`);

    // Also try BF16+AC+FA
    const fixedSetup2 = { ...task.setup, activationCheckpointing: true, flashAttention: true };
    const fixedCtx2 = buildTrainingContext(fixedSetup2);
    console.log(`  BF16+AC+FA memUtil=${fixedCtx2.memoryUtilization?.toFixed(4)}, success=${fixedCtx2.success}`);
  });
});
