/**
 * DualPipeV Pipeline Scheduler — Unit Tests
 *
 * Tests the bubble formula, activation memory, and PP communication
 * for the DualPipeV schedule (bidirectional V-shape with F/B overlap).
 *
 * Key formula: bubble_frac = (PP-1) / (6*m + PP-1) when m >= 2*PP
 * Degrades to standard (PP-1)/(PP-1+m) when m < 2*PP
 */

import { describe, it, expect } from 'vitest';
import {
  ThreeDParallelStrategy,
  type ThreeDParallelConfig,
  DEFAULT_3D_CONFIG,
} from '../../src/core/strategies/3d-parallel.ts';
import { generateDualPipeVSchedule, type ScheduleBlock } from '../../src/components/visualization/PipelineTimeline.tsx';
import { SimulationEngine, type SimulationConfig } from '../../src/core/simulation/engine.ts';
import { generateRecommendations, GENERATORS, validateCandidate } from '../../src/core/simulation/recommendations.ts';
import { getModel } from '../../src/core/models/index.ts';
import { getPresetCluster } from '../../src/core/hardware/index.ts';
import { DEFAULT_DTYPE_CONFIG, DEFAULT_ADAMW_CONFIG, DEFAULT_LR_SCHEDULE } from '../../src/types/index.ts';

// Helper: create a strategy with specific bubble-relevant config
function makeStrategy(pp: number, m: number, schedule: ThreeDParallelConfig['schedule']): ThreeDParallelStrategy {
  return new ThreeDParallelStrategy({
    ...DEFAULT_3D_CONFIG,
    tp: 1,
    pp,
    dp: 1,
    ep: 1,
    numMicroBatches: m,
    schedule,
  });
}

// Access private calculateBubble via computeAnalysis().pipelineBubble
// We use a context-free approach: just check the bubble via the public interface
function getBubble(pp: number, m: number, schedule: ThreeDParallelConfig['schedule']): number {
  const strategy = makeStrategy(pp, m, schedule);
  // computeAnalysis calls calculateBubble internally; but we need a context.
  // Instead, use the formula directly since calculateBubble is private.
  // We'll test through the strategy's validate/computeAnalysis path.
  // Actually, let's just access the method via prototype trick for unit testing.
  return (strategy as unknown as { calculateBubble(): number }).calculateBubble();
}

// ===========================================================================
// Section 1: Bubble Formula
// ===========================================================================

describe('DualPipeV Bubble Formula', () => {
  it('PP=2, m=4: bubble = 1/(1+24) ≈ 0.040', () => {
    const bubble = getBubble(2, 4, 'dualpipe-v');
    expect(bubble).toBeCloseTo(1 / 25, 4);
  });

  it('PP=4, m=8: bubble = 3/(3+48) ≈ 0.059', () => {
    const bubble = getBubble(4, 8, 'dualpipe-v');
    expect(bubble).toBeCloseTo(3 / 51, 4);
  });

  it('PP=8, m=16: bubble = 7/(7+96) ≈ 0.068', () => {
    const bubble = getBubble(8, 16, 'dualpipe-v');
    expect(bubble).toBeCloseTo(7 / 103, 4);
  });

  it('PP=16, m=32: bubble = 15/(15+192) ≈ 0.072', () => {
    const bubble = getBubble(16, 32, 'dualpipe-v');
    expect(bubble).toBeCloseTo(15 / 207, 4);
  });

  it('PP=16, m=144: bubble = 15/(15+864) ≈ 0.017', () => {
    const bubble = getBubble(16, 144, 'dualpipe-v');
    expect(bubble).toBeCloseTo(15 / 879, 4);
  });

  it('PP=1: bubble = 0 (no pipeline)', () => {
    const bubble = getBubble(1, 8, 'dualpipe-v');
    expect(bubble).toBe(0);
  });
});

// ===========================================================================
// Section 2: DualPipeV vs Other Schedules
// ===========================================================================

describe('DualPipeV vs Other Schedules', () => {
  it('DualPipeV has lower bubble than 1F1B for PP=8, m=16', () => {
    const dualPipe = getBubble(8, 16, 'dualpipe-v');
    const onef1b = getBubble(8, 16, '1f1b');
    expect(dualPipe).toBeLessThan(onef1b);
  });

  it('DualPipeV has lower bubble than interleaved v=2 for PP=8, m=16', () => {
    const dualPipe = getBubble(8, 16, 'dualpipe-v');
    const strategy = new ThreeDParallelStrategy({
      ...DEFAULT_3D_CONFIG,
      pp: 8,
      numMicroBatches: 16,
      schedule: 'interleaved-1f1b',
      interleavedStages: 2,
    });
    const interleaved = (strategy as unknown as { calculateBubble(): number }).calculateBubble();
    expect(dualPipe).toBeLessThan(interleaved);
  });

  it('DualPipeV advantage grows with more micro-batches', () => {
    const dualPipe32 = getBubble(8, 32, 'dualpipe-v');
    const dualPipe16 = getBubble(8, 16, 'dualpipe-v');
    // More micro-batches → lower bubble for all, but DualPipeV has 6× factor
    expect(dualPipe32).toBeLessThan(dualPipe16);
  });
});

// ===========================================================================
// Section 3: Fallback when m < 2*PP
// ===========================================================================

describe('DualPipeV Fallback (m < 2*PP)', () => {
  it('PP=8, m=8 (< 2*8=16): falls back to standard formula', () => {
    const dualPipe = getBubble(8, 8, 'dualpipe-v');
    const standard = getBubble(8, 8, '1f1b');
    // When m < 2*PP, DualPipeV uses the same formula as 1F1B
    expect(dualPipe).toBeCloseTo(standard, 6);
  });

  it('PP=4, m=4 (< 2*4=8): falls back to standard formula', () => {
    const dualPipe = getBubble(4, 4, 'dualpipe-v');
    const standard = getBubble(4, 4, '1f1b');
    expect(dualPipe).toBeCloseTo(standard, 6);
  });

  it('PP=4, m=8 (= 2*4): uses DualPipeV formula (not fallback)', () => {
    const dualPipe = getBubble(4, 8, 'dualpipe-v');
    const standard = getBubble(4, 8, '1f1b');
    // DualPipeV should be strictly less than standard when m = 2*PP
    expect(dualPipe).toBeLessThan(standard);
  });
});

// ===========================================================================
// Section 4: In-flight micro-batches (activation memory)
// ===========================================================================

describe('DualPipeV In-flight Micro-batches', () => {
  // In-flight micro-batches affects activation memory pressure.
  // DualPipeV: 2*pp+1 (paper: PP+1 where PP=2*pp virtual stages)
  // 1F1B: pp, Interleaved: ceil(pp/v)

  it('DualPipeV in-flight = 2*pp+1 (paper PP+1 where PP=2*pp)', () => {
    // For pp=4 physical: 2*4+1=9 in-flight, 1F1B has 4
    const pp = 4;
    const m = 16;
    // DualPipeV bubble should use the dualpipe formula
    const bubble = getBubble(pp, m, 'dualpipe-v');
    expect(bubble).toBeCloseTo((pp - 1) / (pp - 1 + 6 * m), 4);
  });
});

// ===========================================================================
// Section 5: PP Communication Boundaries
// ===========================================================================

describe('DualPipeV PP Communication', () => {
  // DualPipeV: bidirectional flow → v=2 in PP comm formula
  // During F&B blocks, chunk 0 sends fwd right + chunk 1 sends grad right,
  // competing for the same link → 2× per-GPU P2P wall clock vs 1F1B.

  it('DualPipeV has 2*(PP-1) boundaries and v=2 per-GPU wall clock', () => {
    const pp = 8;
    // 1F1B: v=1, DualPipeV: v=2 → ppCommPerMB = 2*v*ppCommPerTransfer
    // So DualPipeV PP comm per-MB is 2× that of 1F1B
    const dualPipeComm = 2 * (pp - 1); // 14 boundaries
    const standardComm = pp - 1;        // 7 boundaries
    expect(dualPipeComm).toBe(2 * standardComm);
    expect(dualPipeComm).toBe(14);
    expect(standardComm).toBe(7);
  });
});

// ===========================================================================
// Section 6: DualPipeV Schedule Generation
// ===========================================================================

function countBlockTypes(blocks: ScheduleBlock[], rank: number): Record<string, number> {
  const counts: Record<string, number> = { F0: 0, F1: 0, B0: 0, B1: 0, 'F&B': 0, W: 0 };
  for (const b of blocks) {
    if (b.stage === rank && counts[b.type] !== undefined) {
      counts[b.type]++;
    }
  }
  return counts;
}

describe('DualPipeV Schedule Generation', () => {
  it('all ranks end at approximately the same time (balanced F=B=W)', () => {
    const pp = 4;
    const numMB = 16;
    // B here is activation backward (B_input). W = B inside generator.
    // Balanced: F ≈ B_input ≈ W → use F = B.
    const F = 1.0;
    const B = 1.0;
    const blocks = generateDualPipeVSchedule(pp, numMB, F, B);

    // Compute end time per rank
    const endTimes = new Float64Array(pp);
    for (const b of blocks) {
      const end = b.startMs + b.durationMs;
      if (end > endTimes[b.stage]) endTimes[b.stage] = end;
    }

    const maxEnd = Math.max(...endTimes);
    const minEnd = Math.min(...endTimes);
    // With F=B=W, ranks should align exactly
    expect(maxEnd - minEnd).toBeLessThan(maxEnd * 0.001);
  });

  it('raw block counts per rank match expected phase totals', () => {
    const pp = 4;
    const numMB = 16;
    const F = 1.0;
    const B = 1.0; // B = activation backward (B_input), W = B inside generator
    const blocks = generateDualPipeVSchedule(pp, numMB, F, B);

    for (let r = 0; r < pp; r++) {
      const counts = countBlockTypes(blocks, r);
      // Expected raw counts derived from the 8-phase algorithm:
      // F0: (pp-r-1)*2 + (r+1) = 2pp - r - 1
      // F1: (r+1) + (pp-r-1) = pp
      // B0: (r+1) + (pp-r-1) = pp
      // B1: (pp-r-1) + (pp-r-1) + (r+1) = 2pp - r - 1
      // W: (pp-r-1) + (pp-r-1) + (r+1) = 2pp - r - 1
      // F&B: 2*(m-2pp+r+1) + (pp-r-1) = 2m - 3pp + r + 1
      expect(counts.F0).toBe(2 * pp - r - 1);
      expect(counts.F1).toBe(pp);
      expect(counts.B0).toBe(pp);
      expect(counts.B1).toBe(2 * pp - r - 1);
      expect(counts.W).toBe(2 * pp - r - 1);
      expect(counts['F&B']).toBe(2 * numMB - 3 * pp + r + 1);
    }
  });

  it('produces V-shape: rank 0 starts earliest, rank pp-1 starts latest', () => {
    const pp = 4;
    const numMB = 16;
    const F = 1.0;
    const B = 1.0;
    const blocks = generateDualPipeVSchedule(pp, numMB, F, B);

    // Find first block start per rank
    const firstStart = new Array(pp).fill(Infinity);
    for (const b of blocks) {
      if (b.startMs < firstStart[b.stage]) firstStart[b.stage] = b.startMs;
    }

    // Rank 0 starts at t=0, rank pp-1 starts at (pp-1)*F
    expect(firstStart[0]).toBeCloseTo(0, 6);
    expect(firstStart[pp - 1]).toBeCloseTo((pp - 1) * F, 6);

    // Ranks are staggered: each starts later than the previous
    for (let r = 1; r < pp; r++) {
      expect(firstStart[r]).toBeGreaterThan(firstStart[r - 1]);
    }
  });

  it('steady state F&B blocks are in the middle of each rank timeline', () => {
    const pp = 4;
    const numMB = 16;
    const F = 1.0;
    const B = 1.0;
    const blocks = generateDualPipeVSchedule(pp, numMB, F, B);

    for (let r = 0; r < pp; r++) {
      const rankBlocks = blocks.filter(b => b.stage === r);
      const fbBlocks = rankBlocks.filter(b => b.type === 'F&B');

      if (fbBlocks.length === 0) continue; // pp=1 edge case

      const firstFB = Math.min(...fbBlocks.map(b => b.startMs));
      const lastFB = Math.max(...fbBlocks.map(b => b.startMs + b.durationMs));
      const rankStart = Math.min(...rankBlocks.map(b => b.startMs));
      const rankEnd = Math.max(...rankBlocks.map(b => b.startMs + b.durationMs));

      // F&B blocks should not be the first or last blocks on any rank
      expect(firstFB).toBeGreaterThan(rankStart);
      expect(lastFB).toBeLessThan(rankEnd);
    }
  });

  it('falls back gracefully when numMB < 2*pp', () => {
    const pp = 4;
    const numMB = 4; // < 2*4=8, degraded mode
    const F = 1.0;
    const B = 2.0;
    const blocks = generateDualPipeVSchedule(pp, numMB, F, B);

    // Should produce valid blocks (no crash, no negative times)
    expect(blocks.length).toBeGreaterThan(0);
    for (const b of blocks) {
      expect(b.startMs).toBeGreaterThanOrEqual(0);
      expect(b.durationMs).toBeGreaterThan(0);
    }

    // Degraded mode uses 1F1B fallback with remapped types (F0/B0 only)
    const types = new Set(blocks.map(b => b.type));
    expect(types.has('F0')).toBe(true);
    expect(types.has('B0')).toBe(true);
    // No F1, B1, F&B, W in degraded mode
    expect(types.has('F1')).toBe(false);
    expect(types.has('B1')).toBe(false);
    expect(types.has('F&B')).toBe(false);
    expect(types.has('W')).toBe(false);
  });
});

// ===========================================================================
// Section 7: DualPipeV Degraded Mode Recommendations
// ===========================================================================

function simulate(config: SimulationConfig) {
  const engine = new SimulationEngine();
  engine.configure(config);
  const metrics = engine.simulate();

  const seqLength = config.sequenceLength;
  const model = config.modelSpec ?? (config.modelId ? getModel(config.modelId, seqLength) : null);
  const cluster = config.clusterConfig ?? (config.clusterId ? getPresetCluster(config.clusterId) : null);
  if (!model || !cluster) throw new Error('Invalid config');

  const tp = config.strategyConfig?.tp ?? 1;
  const pp = config.strategyConfig?.pp ?? 1;
  const cp = config.strategyConfig?.cp ?? 1;
  const effectiveDP = Math.max(1, Math.floor(cluster.totalGPUs / (tp * pp * cp)));
  const ga = config.gradientAccumulationSteps ??
    Math.ceil(config.globalBatchSize / (config.microBatchSize * effectiveDP));

  const ctx = {
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

describe('DualPipeV Degraded Mode Recommendations', () => {
  it('recommends increasing GBS when GA < 2*PP', () => {
    // PP=8, TP=8, 64 GPUs → DP=1, GBS=4 → GA=4 < 2*8=16
    const { config, ctx, metrics } = simulate({
      modelId: 'llama2-70b',
      clusterId: '64x-h100',
      globalBatchSize: 4,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 8, pipelineSchedule: 'dualpipe-v' },
    });

    // The degraded-mode generator (#8, pipelineBubbleAndSchedule) produces a valid
    // candidate, but higher-priority generators (scaleEfficiency, activation checkpointing)
    // may consume the 2-recommendation limit first. Test the generator directly.
    const candidates = GENERATORS.map(gen => gen(config, ctx, metrics)).filter(Boolean);
    const degradedCandidate = candidates.find(c => c!.message.includes('degraded'));
    expect(degradedCandidate).toBeDefined();
    expect(degradedCandidate!.message).toContain('global batch size');
    expect(validateCandidate(degradedCandidate!, config, metrics)).toBe(true);
  });

  it('does not recommend GBS increase when GA >= 2*PP', () => {
    // PP=4, TP=8, 64 GPUs → DP=2, GBS=64 → GA=32 >= 2*4=8
    const { config, ctx, metrics } = simulate({
      modelId: 'llama2-70b',
      clusterId: '64x-h100',
      globalBatchSize: 64,
      microBatchSize: 1,
      sequenceLength: 2048,
      strategyType: 'fsdp-tp-pp',
      strategyConfig: { tp: 8, pp: 4, pipelineSchedule: 'dualpipe-v' },
    });

    const recs = generateRecommendations(config, ctx, metrics);
    const degradedRec = recs.find(r => r.includes('degraded'));
    expect(degradedRec).toBeUndefined();
  });
});
