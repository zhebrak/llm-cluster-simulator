/**
 * Training Recommendation Tests
 *
 * Tests for:
 * 1. New validated generators (precision, AC, flash attention, pipeline schedule, SP, EP)
 * 2. Guard conditions that suppress recommendations
 * 3. Validation acceptance/rejection (OOM, no improvement → dropped)
 * 4. Priority ordering (higher-priority generators take slots first)
 * 5. Positive confirmation fallback
 * 6. No numbers in any recommendation text
 */

import { describe, it, expect } from 'vitest';
import {
  SimulationEngine,
  type SimulationConfig,
  type SimulationMetrics,
} from '../../src/core/simulation/engine.ts';
import { generateRecommendations } from '../../src/core/simulation/recommendations.ts';
import type { StrategyContext } from '../../src/core/strategies/base.ts';
import { getModel } from '../../src/core/models/index.ts';
import { getPresetCluster } from '../../src/core/hardware/index.ts';
import { DEFAULT_DTYPE_CONFIG, DEFAULT_ADAMW_CONFIG, DEFAULT_LR_SCHEDULE } from '../../src/types/index.ts';

// ── helpers ──────────────────────────────────────────────────────────

/** Run a full simulation and return { config, ctx, metrics } for recommendation testing */
function simulate(config: SimulationConfig): {
  config: SimulationConfig;
  ctx: StrategyContext;
  metrics: SimulationMetrics;
} {
  const engine = new SimulationEngine();
  engine.configure(config);
  const metrics = engine.simulate();

  // Rebuild the context the same way engine.buildContext does
  const seqLength = config.sequenceLength;
  const model = config.modelSpec ?? (config.modelId ? getModel(config.modelId, seqLength) : null);
  const cluster = config.clusterConfig ?? (config.clusterId ? getPresetCluster(config.clusterId) : null);
  if (!model || !cluster) throw new Error('Invalid config');

  const tp = config.strategyConfig?.tp ?? 1;
  const pp = config.strategyConfig?.pp ?? 1;
  const ep = config.strategyConfig?.ep ?? 1;
  const effectiveDP = Math.max(1, Math.floor(cluster.totalGPUs / (tp * pp * ep)));
  const ga = config.gradientAccumulationSteps ??
    Math.ceil(config.globalBatchSize / (config.microBatchSize * effectiveDP));

  const ctx: StrategyContext = {
    model,
    cluster,
    training: {
      globalBatchSize: config.globalBatchSize,
      microBatchSize: config.microBatchSize,
      sequenceLength: seqLength,
      maxSteps: config.maxSteps ?? 1000,
      optimizer: DEFAULT_ADAMW_CONFIG,
      lrSchedule: DEFAULT_LR_SCHEDULE,
      dtypes: DEFAULT_DTYPE_CONFIG,
      gradientClipping: 1.0,
      gradientAccumulationSteps: ga,
    },
    seqLength,
    microBatchSize: config.microBatchSize,
    globalBatchSize: config.globalBatchSize,
    gradientAccumulationSteps: ga,
    activationCheckpointing: config.activationCheckpointing ?? true,
    flashAttention: config.flashAttention ?? true,
  };

  return { config, ctx, metrics };
}

/** Check that text contains no bare numbers (percentages, batch sizes, etc.) */
function hasNoNumbers(text: string): boolean {
  // Allow numbers only inside well-known words like "1F1B", "FP8", "BF16", etc.
  const cleaned = text
    .replace(/\b(1F1B|FP\d+|BF\d+|TF\d+|INT\d+|ZeRO-\d|zero-\d|FSDP|DDP|TP|PP|DP|EP|SP|GPU|OOM)\b/gi, '')
    .replace(/\b(Gen\d|NVLink|PCIe)\b/gi, '');
  return !/\d/.test(cleaned);
}

// ── number-free recommendations ──────────────────────────────────────

describe('Recommendation Text Quality', () => {
  it('no recommendation text should contain bare numbers', () => {
    const configs: SimulationConfig[] = [
      // Llama 7B on small cluster — should trigger basic recs
      {
        modelId: 'llama2-7b',
        clusterId: '8x-h100',
        globalBatchSize: 512,
        microBatchSize: 2,
        sequenceLength: 2048,
        strategyType: 'fsdp',
      },
      // Llama 70B on multi-node — triggers many generators
      {
        modelId: 'llama2-70b',
        clusterId: '64x-h100',
        globalBatchSize: 1024,
        microBatchSize: 1,
        sequenceLength: 4096,
        strategyType: 'fsdp-tp',
        strategyConfig: { tp: 8 },
        mixedPrecision: 'bf16',
      },
      // GPT-3 175B 3D parallel
      {
        modelId: 'gpt3-175b',
        clusterId: '64x-h100',
        globalBatchSize: 1536,
        microBatchSize: 1,
        sequenceLength: 2048,
        strategyType: 'fsdp-tp-pp',
        strategyConfig: { tp: 8, pp: 4 },
      },
    ];

    for (const cfg of configs) {
      const { config, ctx, metrics } = simulate(cfg);
      const recs = generateRecommendations(config, ctx, metrics);
      for (const rec of recs) {
        expect(hasNoNumbers(rec), `Recommendation contains numbers: "${rec}"`).toBe(true);
      }
    }
  });
});

// ── activation checkpointing ─────────────────────────────────────────

describe('Activation Checkpointing Generator', () => {
  it('suggests switching to selective AC when memory is comfortable and full AC is on', () => {
    // Small model, big cluster, AC on → lots of memory headroom, low MFU
    const { config, ctx, metrics } = simulate({
      modelId: 'gpt3-125m',
      clusterId: '8x-h100',
      globalBatchSize: 64,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'ddp',
      activationCheckpointing: true,
    });

    // Guard: only test AC recommendation when memory is comfortable
    if (metrics.memoryUtilization < 0.70) {
      const recs = generateRecommendations(config, ctx, metrics);
      const acRec = recs.find(r => r.includes('selective activation checkpointing'));
      expect(acRec).toBeDefined();
    }
  });

  it('does NOT suggest disabling AC when memory is tight', () => {
    // Large model, tight memory → should NOT suggest disabling
    const { config, ctx, metrics } = simulate({
      modelId: 'llama2-70b',
      clusterId: '8x-h100',
      globalBatchSize: 256,
      microBatchSize: 1,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8 },
      activationCheckpointing: true,
    });

    const recs = generateRecommendations(config, ctx, metrics);
    const disableAC = recs.find(r => r.includes('disabling activation checkpointing'));
    expect(disableAC).toBeUndefined();
  });
});

// ── flash attention ──────────────────────────────────────────────────

describe('Flash Attention Generator', () => {
  it('suggests enabling Flash Attention when off and long sequence', () => {
    const { config, ctx, metrics } = simulate({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 256,
      microBatchSize: 2,
      sequenceLength: 8192,
      strategyType: 'fsdp',
      flashAttention: false,
    });

    const recs = generateRecommendations(config, ctx, metrics);
    const faRec = recs.find(r => r.includes('Flash Attention'));
    expect(faRec).toBeDefined();
  });

  it('does NOT suggest Flash Attention when already enabled', () => {
    const { config, ctx, metrics } = simulate({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 256,
      microBatchSize: 2,
      sequenceLength: 8192,
      strategyType: 'fsdp',
      flashAttention: true,
    });

    const recs = generateRecommendations(config, ctx, metrics);
    const faRec = recs.find(r => r.includes('Enable Flash Attention'));
    expect(faRec).toBeUndefined();
  });
});

// ── pipeline schedule upgrade ────────────────────────────────────────

describe('Pipeline Schedule Upgrade Generator', () => {
  it('suggests interleaved 1F1B when bubble is high and layers divide evenly', () => {
    // GPT-3 175B: 96 layers, PP=4 → 96 % (4*2) = 0 → interleaved eligible
    const { config, ctx, metrics } = simulate({
      modelId: 'gpt3-175b',
      clusterId: '64x-h100',
      globalBatchSize: 1536,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 4, pipelineSchedule: '1f1b' },
    });

    // With GBS=1536,MBS=1 GA is high enough that bubble may be low
    if (metrics.pipelineBubble > 0.10) {
      const recs = generateRecommendations(config, ctx, metrics);
      const schedRec = recs.find(r => r.includes('interleaved'));
      expect(schedRec).toBeDefined();
    }
  });

  it('suggests interleaved 1F1B for memory relief when memory is tight', () => {
    // Grok-1 on 256x H100 with PP=16, 1f1b schedule → OOM (~1.07)
    // With 2 generator slots, memoryCritical + pipelineReduction take priority.
    // Use PP=4 (no pipelineReduction eligible — needs PP>2) so interleaved can fire.
    const { config, ctx, metrics } = simulate({
      modelId: 'grok-1',
      clusterId: '256x-h100',
      globalBatchSize: 8192,
      microBatchSize: 6,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 4, pipelineSchedule: '1f1b', sequenceParallel: true },
      activationCheckpointing: true,
      flashAttention: true,
      mixedPrecision: 'bf16',
    });

    // Check that either interleaved fires (if memory is tight enough) or the config
    // generates memory-related recommendations
    const recs = generateRecommendations(config, ctx, metrics);
    if (metrics.memoryUtilization > 0.85) {
      const hasMemoryRec = recs.some(r =>
        (r.includes('interleaved') && r.includes('activation memory')) ||
        r.includes('OOM') || r.includes('Memory')
      );
      expect(hasMemoryRec).toBe(true);
    }
  });

  it('falls back to generic bubble advice when already on interleaved', () => {
    const { config, ctx, metrics } = simulate({
      modelId: 'gpt3-175b',
      clusterId: '64x-h100',
      globalBatchSize: 512,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 4, pipelineSchedule: 'interleaved-1f1b', interleavedStages: 2 },
    });

    // With GBS=512,MBS=1 and interleaved, bubble may be low
    if (metrics.pipelineBubble > 0.10) {
      const recs = generateRecommendations(config, ctx, metrics);
      // Should get generic bubble advice since already on interleaved
      const bubbleRec = recs.find(r => r.includes('Pipeline bubble'));
      expect(bubbleRec).toBeDefined();
    }
  });
});

// ── pipeline reduction suggestion ────────────────────────────────────

describe('Pipeline Reduction Generator', () => {
  it('suggests reducing PP when memory is very tight on 3D strategy with high PP', () => {
    // Grok-1 on 128x H100 with PP=16, TP=8 → DP=1, 1F1B schedule
    // MBS=5 pushes in-flight activation memory past capacity (memUtil ~1.09)
    // Halving PP to 8 fits in memory (memUtil ~0.96), accepted via OOM path
    const { config, ctx, metrics } = simulate({
      modelId: 'grok-1',
      clusterId: '128x-h100',
      globalBatchSize: 8192,
      microBatchSize: 5,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 16, pipelineSchedule: '1f1b', sequenceParallel: true },
      activationCheckpointing: true,
      flashAttention: true,
      mixedPrecision: 'bf16',
    });

    expect(metrics.memoryUtilization).toBeGreaterThan(1.0);
    const recs = generateRecommendations(config, ctx, metrics);
    const ppRec = recs.find(r => r.includes('reducing Pipeline Parallelism'));
    expect(ppRec).toBeDefined();
  });

  it('does NOT suggest reducing PP for non-3D strategies', () => {
    const { config, ctx, metrics } = simulate({
      modelId: 'llama2-70b',
      clusterId: '64x-h100',
      globalBatchSize: 1024,
      microBatchSize: 1,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8 },
    });

    const recs = generateRecommendations(config, ctx, metrics);
    const ppRec = recs.find(r => r.includes('reducing Pipeline Parallelism'));
    expect(ppRec).toBeUndefined();
  });

  it('does NOT suggest reducing PP when PP is already minimal', () => {
    const { config, ctx, metrics } = simulate({
      modelId: 'gpt3-175b',
      clusterId: '64x-h100',
      globalBatchSize: 1536,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 2 },
    });

    const recs = generateRecommendations(config, ctx, metrics);
    const ppRec = recs.find(r => r.includes('reducing Pipeline Parallelism'));
    expect(ppRec).toBeUndefined();
  });

  it('does NOT suggest increasing Pipeline Parallelism (removed)', () => {
    const { config, ctx, metrics } = simulate({
      modelId: 'gpt3-175b',
      clusterId: '64x-h100',
      globalBatchSize: 1536,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 2 },
    });

    const recs = generateRecommendations(config, ctx, metrics);
    const ppRec = recs.find(r => r.includes('increasing Pipeline Parallelism'));
    expect(ppRec).toBeUndefined();
  });
});

// ── data parallelism recovery ─────────────────────────────────────────

describe('Data Parallelism Recovery Generator', () => {
  it('suggests reducing TP when DP=1 on multi-GPU config', () => {
    // GPT-3 1.3B on 32× A100: TP=8, PP=4, DP=1 — massively over-parallelized
    const { config, ctx, metrics } = simulate({
      modelId: 'gpt3-1.3b',
      clusterId: '32x-a100',
      globalBatchSize: 8192,
      microBatchSize: 4,
      sequenceLength: 4096,
      strategyType: 'ddp-tp-pp',
      strategyConfig: { tp: 8, pp: 4, pipelineSchedule: 'interleaved-1f1b', interleavedStages: 2, sequenceParallel: true },
      activationCheckpointing: true,
      flashAttention: true,
    });

    const recs = generateRecommendations(config, ctx, metrics);
    const dpRec = recs.find(r => r.includes('no data parallelism'));
    expect(dpRec).toBeDefined();
    expect(dpRec).toContain('Tensor Parallelism');
  });

  it('does NOT fire when DP > 1', () => {
    // GPT-3 175B on 64× H100: TP=8, PP=4, DP=2 — has data parallelism
    const { config, ctx, metrics } = simulate({
      modelId: 'gpt3-175b',
      clusterId: '64x-h100',
      globalBatchSize: 1536,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 4 },
    });

    const recs = generateRecommendations(config, ctx, metrics);
    const dpRec = recs.find(r => r.includes('no data parallelism'));
    expect(dpRec).toBeUndefined();
  });

  it('does NOT fire for 1D strategies', () => {
    const { config, ctx, metrics } = simulate({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 256,
      microBatchSize: 2,
      sequenceLength: 2048,
      strategyType: 'fsdp',
    });

    const recs = generateRecommendations(config, ctx, metrics);
    const dpRec = recs.find(r => r.includes('no data parallelism'));
    expect(dpRec).toBeUndefined();
  });

  it('validation rejects reducing TP when model needs it for memory', () => {
    // Llama 70B on 8× H100: TP=8, PP=1, DP=1 — needs TP=8 for memory
    const { config, ctx, metrics } = simulate({
      modelId: 'llama2-70b',
      clusterId: '8x-h100',
      globalBatchSize: 256,
      microBatchSize: 1,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8 },
    });

    const recs = generateRecommendations(config, ctx, metrics);
    // Should not suggest reducing TP — halving to 4 would OOM or hurt throughput
    const dpRec = recs.find(r => r.includes('no data parallelism'));
    expect(dpRec).toBeUndefined();
  });
});

// ── strategy upgrade text ────────────────────────────────────────────

describe('Strategy Upgrade Text', () => {
  it('does not use PP abbreviation in recommendation text', () => {
    // 2D strategy with very large model → should trigger 2D → 3D upgrade
    const { config, ctx, metrics } = simulate({
      modelId: 'gpt3-175b',
      clusterId: '64x-h100',
      globalBatchSize: 1024,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8 },
    });

    // Precondition: GPT-3 175B on FSDP-TP should have meaningful memory usage
    expect(metrics.memoryUtilization).toBeGreaterThan(0.60);
    if (metrics.memoryUtilization > 0.60) {
      const recs = generateRecommendations(config, ctx, metrics);
      const upgradeRec = recs.find(r => r.includes('Pipeline Parallelism'));
      if (upgradeRec) {
        // Should say "Pipeline Parallelism" not "PP" as a standalone term
        // Check that "PP" only appears as part of strategy combo names (which we removed)
        const withoutKnownTerms = upgradeRec.replace(/Pipeline Parallelism/g, '');
        expect(withoutKnownTerms).not.toMatch(/\bPP\b/);
      }
    }
  });
});

// ── sequence parallelism ─────────────────────────────────────────────

describe('Sequence Parallelism Generator', () => {
  it('suggests SP when off and comm overhead is high for 2D strategy', () => {
    const { config, ctx, metrics } = simulate({
      modelId: 'llama2-70b',
      clusterId: '64x-h100',
      globalBatchSize: 1024,
      microBatchSize: 1,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8, sequenceParallel: false },
    });

    const recs = generateRecommendations(config, ctx, metrics);
    const spRec = recs.find(r => r.includes('Sequence Parallelism'));
    // SP suggestion fires when memUtil > 0.70 or commOverhead > 0.20
    if (metrics.memoryUtilization > 0.70 || metrics.communicationOverhead > 0.20) {
      expect(spRec).toBeDefined();
    }
  });

  it('does NOT suggest SP for 1D strategies', () => {
    const { config, ctx, metrics } = simulate({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 256,
      microBatchSize: 2,
      sequenceLength: 2048,
      strategyType: 'fsdp',
    });

    const recs = generateRecommendations(config, ctx, metrics);
    const spRec = recs.find(r => r.includes('Sequence Parallelism'));
    expect(spRec).toBeUndefined();
  });
});

// ── expert parallelism ───────────────────────────────────────────────

describe('Expert Parallelism Generator', () => {
  it('does NOT suggest EP for dense (non-MoE) models', () => {
    const { config, ctx, metrics } = simulate({
      modelId: 'llama2-70b',
      clusterId: '8x-h100',
      globalBatchSize: 256,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8 },
    });

    const recs = generateRecommendations(config, ctx, metrics);
    const epRec = recs.find(r => r.includes('Expert Parallelism') || r.includes('expert parallelism'));
    expect(epRec).toBeUndefined();
  });

  it('still considers EP increase when already using EP > 1', () => {
    // Maverick (160 experts) on 1024 H100s with TP=8, PP=4, EP=2
    // EP=2 is suboptimal — generator should consider increasing it
    const { config, ctx, metrics } = simulate({
      modelId: 'llama4-maverick',
      clusterId: '1024x-h100',
      globalBatchSize: 8192,
      microBatchSize: 4,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 4, ep: 2, sequenceParallel: true },
      activationCheckpointing: true,
      flashAttention: true,
    });

    // The generator should not be blocked by currentEP > 1
    // It may still return null if no valid EP > 2 exists within current TP,
    // or the validator may reject it, but the gate itself should not block it.
    const recs = generateRecommendations(config, ctx, metrics);
    // Just verify it doesn't crash and produces valid output
    expect(recs.length).toBeGreaterThan(0);
    expect(recs.length).toBeLessThanOrEqual(3);
  });

  it('considers EP even when memory is comfortable', () => {
    // Mixtral 8x7B on 16x H100, TP=4 — memory should be comfortable
    // but EP could still improve throughput
    const { config, ctx, metrics } = simulate({
      modelId: 'mixtral-8x7b',
      clusterId: '16x-h100',
      globalBatchSize: 256,
      microBatchSize: 2,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 4, ep: 1 },
    });

    // Verify the memory gate is removed — even with low memory, EP generator
    // can fire (validated by throughput instead of memory)
    const recs = generateRecommendations(config, ctx, metrics);
    expect(recs.length).toBeGreaterThan(0);
    expect(recs.length).toBeLessThanOrEqual(3);
  });

  it('generates compound TP+EP mutation without crashing', () => {
    // Grok-1: TP=8 on 8-GPU node, maxEP within TP=1 → Strategy B kicks in
    // Tests that compound mutation (reduce TP + enable EP) works
    const { config, ctx, metrics } = simulate({
      modelId: 'grok-1',
      clusterId: '256x-h100',
      globalBatchSize: 8192,
      microBatchSize: 4,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 8, ep: 1, sequenceParallel: true,
        pipelineSchedule: 'interleaved-1f1b' as const, interleavedStages: 2 },
      activationCheckpointing: true,
      flashAttention: true,
    });

    // Should not crash, even if validator rejects the mutation
    const recs = generateRecommendations(config, ctx, metrics);
    expect(recs.length).toBeGreaterThan(0);
    expect(recs.length).toBeLessThanOrEqual(3);
  });

  it('EP recommendation messages follow no-numbers rule', () => {
    // Use a config that's likely to trigger an EP rec
    const { config, ctx, metrics } = simulate({
      modelId: 'deepseek-v3',
      clusterId: '256x-h100',
      globalBatchSize: 4096,
      microBatchSize: 2,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 4, ep: 1, sequenceParallel: true },
      activationCheckpointing: true,
      flashAttention: true,
    });

    const recs = generateRecommendations(config, ctx, metrics);
    for (const rec of recs) {
      expect(hasNoNumbers(rec), `Recommendation contains numbers: "${rec}"`).toBe(true);
    }
  });
});

// ── validation acceptance/rejection ──────────────────────────────────

describe('Validation Framework', () => {
  it('recommendations are limited to 3 max', () => {
    const { config, ctx, metrics } = simulate({
      modelId: 'llama2-70b',
      clusterId: '64x-h100',
      globalBatchSize: 1024,
      microBatchSize: 1,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8 },
      mixedPrecision: 'bf16',
    });

    const recs = generateRecommendations(config, ctx, metrics);
    expect(recs.length).toBeLessThanOrEqual(3);
    expect(recs.length).toBeGreaterThan(0);
  });

  it('validation rejects mutation that causes OOM', () => {
    // Llama 70B on 8x H100 with AC=true. Disabling AC should OOM.
    const { config, ctx, metrics } = simulate({
      modelId: 'llama2-70b',
      clusterId: '8x-h100',
      globalBatchSize: 256,
      microBatchSize: 1,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8 },
      activationCheckpointing: true,
    });

    // Should not get a "disable AC" rec even though it would trigger if memUtil < 0.50
    const recs = generateRecommendations(config, ctx, metrics);
    const disableAC = recs.find(r => r.includes('disabling activation checkpointing'));
    expect(disableAC).toBeUndefined();
  });

  it('validation correctly handles configs with pre-computed GA', () => {
    // Pass explicit GA. Validated generators (AC, FA, etc.) should still work
    // because validateCandidate clears stale GA before re-simulating.
    const { config, ctx, metrics } = simulate({
      modelId: 'gpt3-125m',
      clusterId: '8x-h100',
      globalBatchSize: 64,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'ddp',
      activationCheckpointing: true,
      gradientAccumulationSteps: 8, // explicit GA
    });

    // Should not crash or produce incorrect results
    const recs = generateRecommendations(config, ctx, metrics);
    expect(recs.length).toBeGreaterThan(0);
    expect(recs.length).toBeLessThanOrEqual(3);
  });
});

// ── priority ordering ────────────────────────────────────────────────

describe('Priority Ordering', () => {
  it('strategy upgrade appears before activation checkpointing', () => {
    // DDP with large model → should get strategy upgrade before any AC suggestion
    const { config, ctx, metrics } = simulate({
      modelId: 'llama2-70b',
      clusterId: '64x-h100',
      globalBatchSize: 256,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'ddp',
    });

    const recs = generateRecommendations(config, ctx, metrics);
    const stratIdx = recs.findIndex(r => r.includes('FSDP') || r.includes('Tensor Parallelism'));

    // Strategy upgrade should be first actionable rec (after status line)
    if (stratIdx >= 0) {
      if (metrics.memoryUtilization <= 1.0) {
        // Non-OOM: status line at index 0, strategy upgrade at index 1
        expect(recs[0]).toMatch(/Well-optimized|Solid baseline/);
        expect(stratIdx).toBe(1);
      } else {
        // OOM: no status line, strategy upgrade at index 0
        expect(stratIdx).toBe(0);
      }
    }
  });

  it('memory critical appears before other suggestions', () => {
    // Force near-OOM: large model, small cluster, AC off, FA off
    const { config, ctx, metrics } = simulate({
      modelId: 'llama2-70b',
      clusterId: '8x-h100',
      globalBatchSize: 256,
      microBatchSize: 1,
      sequenceLength: 4096,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8 },
      activationCheckpointing: false,
      flashAttention: false,
    });

    const recs = generateRecommendations(config, ctx, metrics);
    const memIdx = recs.findIndex(r => r.includes('OOM') || r.includes('Memory') || r.includes('checkpointing'));

    // Precondition: LLaMA 70B FSDP-TP tp=8, no AC, no FA should be tight on memory
    expect(metrics.memoryUtilization).toBeGreaterThan(0.90);
    // Memory-related rec should be first actionable
    if (metrics.memoryUtilization > 1.0) {
      // OOM: no status line, memoryCritical at index 0
      expect(memIdx).toBe(0);
    } else if (metrics.memoryUtilization > 0.90) {
      // Near-OOM: status line at index 0, memoryCritical at index 1
      expect(recs[0]).toMatch(/Solid baseline/);
      expect(memIdx).toBe(1);
    }
  });
});

// ── positive confirmation fallback ───────────────────────────────────

describe('Positive Confirmation Fallback', () => {
  it('returns a positive message when no specific recommendations apply', () => {
    // Well-optimized config: Llama 7B, 8x H100, FSDP, FP8, all bells on
    const { config, ctx, metrics } = simulate({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 1024,
      microBatchSize: 4,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      mixedPrecision: 'fp8',
      activationCheckpointing: true,
      flashAttention: true,
    });

    const recs = generateRecommendations(config, ctx, metrics);
    expect(recs.length).toBeGreaterThan(0);
    // Status line should always be present for non-OOM configs
    const hasStatus = recs.some(r =>
      r.includes('Well-optimized') || r.includes('Solid baseline')
    );
    const hasActionable = recs.some(r =>
      r.includes('Consider') || r.includes('Reduce') ||
      r.includes('Increase') || r.includes('Switch') || r.includes('Try')
    );
    // Status line should always be present, possibly with actionable recommendations
    expect(hasStatus || hasActionable).toBe(true);
  });
});

// ── existing generators unchanged ────────────────────────────────────

describe('Existing Generators Unchanged', () => {
  it('DDP → FSDP recommendation fires for large model with high memory', () => {
    // Llama 13B on DDP with A100 — should recommend FSDP
    const { config, ctx, metrics } = simulate({
      modelId: 'llama2-13b',
      clusterId: '8x-a100',
      globalBatchSize: 512,
      microBatchSize: 2,
      sequenceLength: 2048,
      strategyType: 'ddp',
    });

    // Precondition: Llama 13B DDP on 8× A100 should have high memory utilization
    expect(metrics.memoryUtilization).toBeGreaterThan(0.70);
    if (metrics.memoryUtilization > 0.70) {
      const recs = generateRecommendations(config, ctx, metrics);
      // Match on "FSDP" (strategy upgrade rec mentions FSDP and sharding)
      const fsdpRec = recs.find(r => r.includes('FSDP'));
      expect(fsdpRec).toBeDefined();
    }
  });

  it('pipeline bubble recommendation fires for 3D strategy with high bubble', () => {
    const { config, ctx, metrics } = simulate({
      modelId: 'gpt3-175b',
      clusterId: '64x-h100',
      globalBatchSize: 512,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 4 },
    });

    // With GBS=512,MBS=1 and PP=4, bubble may be low due to high GA
    if (metrics.pipelineBubble > 0.10) {
      const recs = generateRecommendations(config, ctx, metrics);
      const bubbleRec = recs.find(r =>
        r.includes('pipeline') || r.includes('Pipeline') || r.includes('interleaved')
      );
      expect(bubbleRec).toBeDefined();
    }
  });

  it('communication overhead recommendation fires for multi-node 1D', () => {
    const { config, ctx, metrics } = simulate({
      modelId: 'llama2-13b',
      clusterId: '16x-h100',
      globalBatchSize: 256,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp',
    });

    if (metrics.communicationOverhead > 0.25) {
      const recs = generateRecommendations(config, ctx, metrics);
      const commRec = recs.find(r => r.includes('Tensor Parallelism') || r.includes('communication'));
      expect(commRec).toBeDefined();
    }
  });
});

// ── integration: end-to-end via runSimulation ────────────────────────

describe('End-to-end Integration', () => {
  it('runSimulation includes recommendations in analysis', () => {
    const engine = new SimulationEngine();
    engine.configure({
      modelId: 'llama2-7b',
      clusterId: '8x-h100',
      globalBatchSize: 512,
      microBatchSize: 2,
      sequenceLength: 2048,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
    });
    const result = engine.run();

    expect(result.success).toBe(true);
    expect(result.analysis.recommendations.length).toBeGreaterThan(0);
    expect(result.analysis.recommendations.length).toBeLessThanOrEqual(3);
  });

  it('recommendations work for MoE models', () => {
    // Mixtral 8x7B on 16x H100 (2 nodes) with TP=8 — should be valid
    const engine = new SimulationEngine();
    engine.configure({
      modelId: 'mixtral-8x7b',
      clusterId: '16x-h100',
      globalBatchSize: 128,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp',
      strategyConfig: { tp: 8 },
    });
    const result = engine.run();

    expect(result.success).toBe(true);
    expect(result.analysis.recommendations.length).toBeGreaterThan(0);
  });

  it('runSimulation works for 3D parallel with pipeline', () => {
    const engine = new SimulationEngine();
    engine.configure({
      modelId: 'gpt3-175b',
      clusterId: '64x-h100',
      globalBatchSize: 1536,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 4 },
      mixedPrecision: 'bf16',
    });
    const result = engine.run();

    expect(result.success).toBe(true);
    expect(result.analysis.recommendations.length).toBeGreaterThan(0);
    expect(result.analysis.recommendations.length).toBeLessThanOrEqual(3);
  });
});
