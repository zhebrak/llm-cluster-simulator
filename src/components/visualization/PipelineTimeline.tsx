/**
 * Pipeline schedule (1F1B) Gantt chart visualization.
 * Shows how micro-batches flow through pipeline stages, making bubble waste visible.
 */

import { useMemo, useRef, useState, useEffect, memo } from 'react';
import { GitBranch } from 'lucide-react';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface PipelineTimelineProps {
  ppDegree: number;
  numMicroBatches: number;
  gradientAccumulationSteps: number;
  totalForwardMs: number;      // metrics.timing.forward (includes bubble inflation)
  totalBackwardMs: number;     // metrics.timing.backward (includes bubble inflation)
  pipelineBubble: number;      // 0..1
  pipelineSchedule?: '1f1b' | 'interleaved-1f1b' | 'dualpipe-v';
  interleavedStages?: number;  // virtual stages per device (v)
}

export interface ScheduleBlock {
  stage: number;
  microBatch: number;
  type: 'forward' | 'backward' | 'F0' | 'F1' | 'B0' | 'B1' | 'F&B' | 'W';
  startMs: number;
  durationMs: number;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_VISUAL_MB = 32;
const MAX_VISUAL_STAGES = 12;

const COLORS = {
  forward: '#22c55e',     // green-500
  backward: '#3b82f6',    // blue-500
  bubble: '#ef4444',      // red-500
  label: '#9ca3af',       // gray-400
  tick: '#6b7280',        // gray-500
  get gridLine() {
    return getComputedStyle(document.documentElement).getPropertyValue('--color-chart-grid').trim() || 'rgba(55, 65, 81, 0.5)';
  },
};

// Interleaved colors: each virtual chunk (v=0,1,2,...) gets a distinct shade pair
const INTERLEAVED_FORWARD = ['#22c55e', '#06b6d4', '#a855f7', '#f59e0b', '#ec4899', '#14b8a6', '#f97316', '#8b5cf6'];
const INTERLEAVED_BACKWARD = ['#3b82f6', '#0891b2', '#7c3aed', '#d97706', '#db2777', '#0d9488', '#ea580c', '#6d28d9'];

// DualPipeV block colors
const DUALPIPE_COLORS: Record<string, string> = {
  'F0': '#22c55e',   // green — forward chunk 0
  'F1': '#06b6d4',   // cyan — forward chunk 1
  'B0': '#3b82f6',   // blue — backward chunk 0
  'B1': '#0891b2',   // teal — backward chunk 1
  'F&B': '#a855f7',  // purple — overlapped forward+backward
  'W': '#f59e0b',    // orange — weight gradient
};

const Y_LABEL_WIDTH = 56;
const ROW_HEIGHT = 28;
const BAR_PADDING = 2;
const X_AXIS_HEIGHT = 24;
const TOP_PAD = 4;

// ---------------------------------------------------------------------------
// Schedule generation — dynamic 1F1B with greedy earliest-ready-first
// ---------------------------------------------------------------------------

export function generate1F1BSchedule(
  pp: number,
  numMB: number,
  F: number,
  B: number,
  numDevices?: number,  // physical devices (< pp means device sharing for interleaved)
): ScheduleBlock[] {
  // Dynamic scheduling: each stage independently picks between B (preferred)
  // and F based on actual dependency readiness. No pre-computed op sequence.
  //
  // Dependencies:
  //   Forward:  F[s][m] depends on F[s-1][m]  (propagates stage 0 → PP-1)
  //   Backward: B[s][m] depends on B[s+1][m]  (propagates stage PP-1 → 0)
  //   Last stage: B[pp-1][m] depends on F[pp-1][m]
  //
  // B-preference: when both B and F are eligible at the same time on a stage,
  // B wins to keep the backward chain moving.
  //
  // Device sharing (interleaved): when numDevices < pp, multiple virtual stages
  // share a physical device's cursor. No two blocks on the same device can overlap.

  const devCount = numDevices ?? pp;
  const deviceOf = (stage: number) => stage % devCount;

  const forwardDone: boolean[][] = Array.from({ length: pp }, () => new Array(numMB).fill(false));
  const backwardDone: boolean[][] = Array.from({ length: pp }, () => new Array(numMB).fill(false));
  const forwardEnd: number[][] = Array.from({ length: pp }, () => new Array(numMB).fill(0));
  const backwardEnd: number[][] = Array.from({ length: pp }, () => new Array(numMB).fill(0));
  const deviceCursor = new Float64Array(devCount);
  const fCount = new Int32Array(pp);   // next forward micro-batch index per stage
  const bCount = new Int32Array(pp);   // next backward micro-batch index per stage
  const blocks: ScheduleBlock[] = [];
  const totalOps = pp * numMB * 2;

  while (blocks.length < totalOps) {
    let bestStage = -1;
    let bestStart = Infinity;
    let bestType: 'F' | 'B' = 'F';
    let bestMB = 0;

    for (let s = 0; s < pp; s++) {
      const dev = deviceOf(s);

      // Try backward first (B-preference keeps backward chain moving)
      if (bCount[s] < numMB) {
        const mb = bCount[s];
        let earliest = deviceCursor[dev];
        let eligible = true;

        if (s < pp - 1) {
          if (!backwardDone[s + 1][mb]) eligible = false;
          else earliest = Math.max(earliest, backwardEnd[s + 1][mb]);
        } else {
          // Last stage: backward depends on own forward
          if (!forwardDone[s][mb]) eligible = false;
          else earliest = Math.max(earliest, forwardEnd[s][mb]);
        }

        if (eligible && earliest < bestStart) {
          bestStart = earliest;
          bestStage = s;
          bestType = 'B';
          bestMB = mb;
        }
      }

      // Try forward
      if (fCount[s] < numMB) {
        const mb = fCount[s];
        let earliest = deviceCursor[dev];
        let eligible = true;

        if (s > 0) {
          if (!forwardDone[s - 1][mb]) eligible = false;
          else earliest = Math.max(earliest, forwardEnd[s - 1][mb]);
        }

        if (eligible && (earliest < bestStart || (earliest === bestStart && bestType === 'F' && dev === deviceOf(bestStage)))) {
          bestStart = earliest;
          bestStage = s;
          bestType = 'F';
          bestMB = mb;
        }
      }
    }

    if (bestStage < 0) break;

    const s = bestStage;
    const dur = bestType === 'F' ? F : B;
    const end = bestStart + dur;

    blocks.push({
      stage: s,
      microBatch: bestMB,
      type: bestType === 'F' ? 'forward' : 'backward',
      startMs: bestStart,
      durationMs: dur,
    });

    deviceCursor[deviceOf(s)] = end;
    if (bestType === 'F') {
      forwardEnd[s][bestMB] = end;
      forwardDone[s][bestMB] = true;
      fCount[s]++;
    } else {
      backwardEnd[s][bestMB] = end;
      backwardDone[s][bestMB] = true;
      bCount[s]++;
    }
  }

  return blocks;
}

// ---------------------------------------------------------------------------
// DualPipeV schedule generation — bidirectional V-shape
// ---------------------------------------------------------------------------

/**
 * Generate a simplified DualPipeV schedule visualization.
 *
 * Each rank r holds 2 chunks (chunk 0 = layer r, chunk 1 = layer 2pp-1-r).
 * The schedule shows the distinctive pattern:
 * - Warmup: F0 blocks flow left→right, F1 blocks flow right→left
 * - Steady state: F&B (overlapped forward+backward) blocks
 * - Cooldown: B1 blocks + W (weight grad) blocks
 */
export function generateDualPipeVSchedule(
  pp: number,
  numMB: number,
  F: number,  // per-chunk forward time (half of full forward)
  B: number,  // per-chunk backward time (half of full backward)
): ScheduleBlock[] {
  const m = numMB;
  const W = B; // weight grad ≈ activation backward (split from engine's full backward)
  const FB = F + B; // overlapped forward+backward time

  // Degraded mode: when m < 2*pp, the 8-phase formulas assume m >= 2*pp.
  // Fall back to standard 1F1B and remap block types for DualPipeV coloring.
  if (m < 2 * pp) {
    const fallback = generate1F1BSchedule(pp, m, F, B);
    return fallback.map(blk => ({
      ...blk,
      type: blk.type === 'forward' ? 'F0' as const : 'B0' as const,
    }));
  }

  // 8-phase DualPipeV algorithm (derived from deepseek-ai/DualPipe reference).
  // Each rank r (0..pp-1) starts at offset r*F and executes 8 phases sequentially.
  const blocks: ScheduleBlock[] = [];

  for (let r = 0; r < pp; r++) {
    let cursor = r * F; // staggered start
    let f0_idx = 0;
    let f1_idx = 0;
    let b0_idx = 0;
    let b1_idx = 0;
    let w_idx = 0;

    // Phase 1: F0 warmup — count = (pp - r - 1) * 2
    const phase1Count = (pp - r - 1) * 2;
    for (let i = 0; i < phase1Count; i++) {
      blocks.push({ stage: r, microBatch: f0_idx++, type: 'F0', startMs: cursor, durationMs: F });
      cursor += F;
    }

    // Phase 2: F0 + F1 interleave — count = r + 1
    const phase2Count = r + 1;
    for (let i = 0; i < phase2Count; i++) {
      blocks.push({ stage: r, microBatch: f0_idx++, type: 'F0', startMs: cursor, durationMs: F });
      cursor += F;
      blocks.push({ stage: r, microBatch: f1_idx++, type: 'F1', startMs: cursor, durationMs: F });
      cursor += F;
    }

    // Phase 3: B1 + W + F1 — count = pp - r - 1
    const phase3Count = pp - r - 1;
    for (let i = 0; i < phase3Count; i++) {
      blocks.push({ stage: r, microBatch: b1_idx++, type: 'B1', startMs: cursor, durationMs: B });
      cursor += B;
      blocks.push({ stage: r, microBatch: w_idx++, type: 'W', startMs: cursor, durationMs: W });
      cursor += W;
      blocks.push({ stage: r, microBatch: f1_idx++, type: 'F1', startMs: cursor, durationMs: F });
      cursor += F;
    }

    // Phase 4: Steady state — count = max(0, m - 2*pp + r + 1)
    // Each iteration: F&B chunk(0,1) = F0+B1 overlap, then F&B chunk(1,0) = F1+B0 overlap
    const phase4Count = Math.max(0, m - 2 * pp + r + 1);
    for (let i = 0; i < phase4Count; i++) {
      blocks.push({ stage: r, microBatch: f0_idx, type: 'F&B', startMs: cursor, durationMs: FB });
      f0_idx++; b1_idx++;
      cursor += FB;
      blocks.push({ stage: r, microBatch: f1_idx, type: 'F&B', startMs: cursor, durationMs: FB });
      f1_idx++; b0_idx++;
      cursor += FB;
    }

    // Phase 5: B1 + F&B chunk(1,0) drain — count = pp - r - 1
    // F&B chunk(1,0) = F1+B0 overlap
    const phase5Count = pp - r - 1;
    for (let i = 0; i < phase5Count; i++) {
      blocks.push({ stage: r, microBatch: b1_idx++, type: 'B1', startMs: cursor, durationMs: B });
      cursor += B;
      blocks.push({ stage: r, microBatch: f1_idx, type: 'F&B', startMs: cursor, durationMs: FB });
      f1_idx++; b0_idx++;
      cursor += FB;
    }

    // Phase 6: B1 + B0 cooldown — count = r + 1
    const phase6Count = r + 1;
    for (let i = 0; i < phase6Count; i++) {
      blocks.push({ stage: r, microBatch: b1_idx++, type: 'B1', startMs: cursor, durationMs: B });
      cursor += B;
      blocks.push({ stage: r, microBatch: b0_idx++, type: 'B0', startMs: cursor, durationMs: B });
      cursor += B;
    }

    // Phase 7: W + B0 — count = pp - r - 1
    const phase7Count = pp - r - 1;
    for (let i = 0; i < phase7Count; i++) {
      blocks.push({ stage: r, microBatch: w_idx++, type: 'W', startMs: cursor, durationMs: W });
      cursor += W;
      blocks.push({ stage: r, microBatch: b0_idx++, type: 'B0', startMs: cursor, durationMs: B });
      cursor += B;
    }

    // Phase 8: Final W — count = r + 1
    const phase8Count = r + 1;
    for (let i = 0; i < phase8Count; i++) {
      blocks.push({ stage: r, microBatch: w_idx++, type: 'W', startMs: cursor, durationMs: W });
      cursor += W;
    }
  }

  return blocks;
}

// ---------------------------------------------------------------------------
// Tick generation
// ---------------------------------------------------------------------------

function niceTimeTicks(maxMs: number, count: number): number[] {
  if (maxMs <= 0) return [0];
  const raw = maxMs / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const residual = raw / mag;
  const nice = residual <= 1.5 ? 1 : residual <= 3 ? 2 : residual <= 7 ? 5 : 10;
  const step = nice * mag;
  const ticks: number[] = [];
  for (let t = 0; t <= maxMs + step * 0.01; t += step) {
    ticks.push(+t.toFixed(6));
  }
  return ticks;
}

function formatTickLabel(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
  if (ms >= 10) return `${Math.round(ms)}ms`;
  if (ms >= 1) return `${ms.toFixed(1)}ms`;
  if (ms >= 0.01) return `${ms.toFixed(2)}ms`;
  return `${(ms * 1000).toFixed(0)}µs`;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export const PipelineTimeline = memo(function PipelineTimeline({
  ppDegree,
  numMicroBatches,
  gradientAccumulationSteps,
  totalForwardMs,
  totalBackwardMs,
  pipelineBubble,
  pipelineSchedule = '1f1b',
  interleavedStages = 1,
}: PipelineTimelineProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(600);

  // Responsive width
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width;
      if (w && w > 0) setWidth(w);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const ga = gradientAccumulationSteps;

  const isInterleaved = pipelineSchedule === 'interleaved-1f1b' && interleavedStages >= 2;
  const isDualPipeV = pipelineSchedule === 'dualpipe-v';
  const v = isInterleaved ? interleavedStages : 1;

  // Virtual stages = pp * v for interleaved (the scheduler sees all of them)
  // But we always render pp physical GPU rows
  const visualPP = Math.min(ppDegree, MAX_VISUAL_STAGES);
  const totalVirtualStages = isInterleaved ? visualPP * v : visualPP;

  // Try to show 2 complete batches (steps) to demonstrate the repeating pattern.
  // Fall back to 1 batch if 2 won't fit, possibly truncated mid-batch.
  const oneBatch = numMicroBatches;
  const twoBatches = numMicroBatches * 2;
  const minVisualMB = Math.min(oneBatch, totalVirtualStages * 3);
  const targetMB = twoBatches <= MAX_VISUAL_MB ? twoBatches : oneBatch;
  const visualMB = isDualPipeV
    ? Math.max(Math.min(oneBatch, MAX_VISUAL_MB), Math.min(visualPP * 2, MAX_VISUAL_MB))
    : Math.max(minVisualMB, Math.min(targetMB, MAX_VISUAL_MB));
  const batchesShown = visualMB >= twoBatches ? 2 : visualMB >= oneBatch ? 1 : 0;

  // Derive per-microbatch times (invert engine formula)
  const eff = Math.max(0.05, 1 - pipelineBubble);
  const perDeviceF = (totalForwardMs * eff) / ga;
  const perDeviceB = (totalBackwardMs * eff) / ga;

  // For interleaved, each virtual stage processes 1/v of the layers → 1/v of compute time
  // For DualPipeV, each rank holds 2 chunks → per-chunk time = perDevice / 2
  // F = per-chunk forward, B = per-chunk activation backward (= F), W = per-chunk weight grad (= B)
  // Engine's backwardMultiplier includes ALL backward (activation grads + weight grads + recompute).
  // With ckpt (mult=2.85): perDeviceB = 2.85×perDeviceF → per-chunk total B+W = perDeviceB/2 = 1.425F
  //   B_input = W = perDeviceB/4 = 0.7125F each (activation backward ≈ weight backward)
  // Without ckpt (mult=2): perDeviceB = 2×perDeviceF → per-chunk total B+W = perDeviceF
  //   B_input = W = perDeviceB/4 = 0.5F each
  const F = isInterleaved ? perDeviceF / v : isDualPipeV ? perDeviceF / 2 : perDeviceF;
  const B = isInterleaved ? perDeviceB / v : isDualPipeV ? perDeviceB / 4 : perDeviceB;
  const W = isDualPipeV ? B : 0; // weight grad (only shown for DualPipeV)

  const { blocks, maxTime, bubbleGaps } = useMemo(() => {
    const b = isDualPipeV
      ? generateDualPipeVSchedule(visualPP, visualMB, F, B)
      : generate1F1BSchedule(
          totalVirtualStages, visualMB, F, B,
          isInterleaved ? visualPP : undefined,
        );
    let mx = 0;
    for (const blk of b) {
      const end = blk.startMs + blk.durationMs;
      if (end > mx) mx = end;
    }

    // Compute idle gaps (bubble) per physical GPU row
    // For interleaved, group blocks by device (stage % pp)
    const numRows = isInterleaved ? visualPP : visualPP;
    const minGap = mx * 0.002;
    const gaps: { stage: number; startMs: number; durationMs: number }[] = [];
    for (let row = 0; row < numRows; row++) {
      const rowBlocks = b
        .filter((blk) => (isInterleaved ? blk.stage % visualPP : blk.stage) === row)
        .sort((a, b) => a.startMs - b.startMs);
      let cursor = 0;
      for (const blk of rowBlocks) {
        if (blk.startMs > cursor + minGap) {
          gaps.push({ stage: row, startMs: cursor, durationMs: blk.startMs - cursor });
        }
        cursor = blk.startMs + blk.durationMs;
      }
      if (cursor < mx - minGap) {
        gaps.push({ stage: row, startMs: cursor, durationMs: mx - cursor });
      }
    }

    return { blocks: b, maxTime: mx, bubbleGaps: gaps };
  }, [totalVirtualStages, visualMB, F, B, isInterleaved, isDualPipeV, visualPP]);

  const ticks = useMemo(() => niceTimeTicks(maxTime, 6), [maxTime]);

  // SVG dimensions — always pp physical rows
  const numRows = visualPP;
  const labelWidth = Y_LABEL_WIDTH;
  const chartWidth = width - labelWidth - 16; // 16px right padding
  const chartHeight = numRows * ROW_HEIGHT;
  const svgWidth = width;
  const svgHeight = chartHeight + X_AXIS_HEIGHT + TOP_PAD;

  const timeEnd = ticks[ticks.length - 1] || maxTime || 1;
  const xScale = (ms: number) => labelWidth + (ms / timeEnd) * chartWidth;

  // Minimum bar width for label visibility
  const MIN_LABEL_WIDTH = 24;

  // Row label: always physical GPU
  const rowLabel = (row: number) => `Stage ${row}`;

  // Color for a block: DualPipeV uses per-type colors, interleaved uses chunk colors
  const blockColor = (blk: ScheduleBlock) => {
    if (isDualPipeV) {
      return DUALPIPE_COLORS[blk.type] ?? COLORS.forward;
    }
    if (!isInterleaved) {
      return blk.type === 'forward' ? COLORS.forward : COLORS.backward;
    }
    const chunk = Math.floor(blk.stage / visualPP); // 0, 1, 2, ... v-1
    return blk.type === 'forward'
      ? INTERLEAVED_FORWARD[chunk % INTERLEAVED_FORWARD.length]
      : INTERLEAVED_BACKWARD[chunk % INTERLEAVED_BACKWARD.length];
  };

  // Physical GPU row for a block
  const rowOf = (blk: ScheduleBlock) => isInterleaved ? blk.stage % visualPP : blk.stage;

  return (
    <div
      className="bg-gray-900/50 border border-gray-800 rounded-xl p-5 animate-slide-up-sm"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-lg font-medium text-white flex items-center gap-2">
          <GitBranch className="w-5 h-5 text-cyan-400" />
          Pipeline Schedule ({isDualPipeV ? 'DualPipeV' : isInterleaved ? `Interleaved 1F1B, v=${v}` : '1F1B'})
        </h3>
        <div className="flex items-center gap-4 text-xs text-gray-400 flex-wrap">
          {isDualPipeV ? (
            <>
              {(['F0', 'F1', 'B0', 'B1', 'F&B', 'W'] as const).map(t => (
                <span key={t} className="flex items-center gap-1">
                  <span className="inline-block w-3 h-2.5 rounded-sm" style={{ background: DUALPIPE_COLORS[t] }} />
                  {t}
                </span>
              ))}
            </>
          ) : isInterleaved ? (
            <>
              {Array.from({ length: v }, (_, i) => (
                <span key={`v${i}`} className="flex items-center gap-1">
                  <span className="inline-block w-3 h-2.5 rounded-sm" style={{ background: INTERLEAVED_FORWARD[i % INTERLEAVED_FORWARD.length] }} />
                  <span className="inline-block w-3 h-2.5 rounded-sm" style={{ background: INTERLEAVED_BACKWARD[i % INTERLEAVED_BACKWARD.length] }} />
                  {'F/B' + String.fromCharCode(0x2080 + i)}
                </span>
              ))}
            </>
          ) : (
            <>
              <span className="flex items-center gap-1.5">
                <span className="inline-block w-3 h-2.5 rounded-sm" style={{ background: COLORS.forward }} />
                Forward
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block w-3 h-2.5 rounded-sm" style={{ background: COLORS.backward }} />
                Backward
              </span>
            </>
          )}
          <span className="flex items-center gap-1.5">
            <svg width="12" height="10" className="rounded-sm">
              <defs>
                <pattern id="legend-hatch" width="3" height="3" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                  <line x1="0" y1="0" x2="0" y2="3" stroke="rgba(239, 68, 68, 0.4)" strokeWidth="1" />
                </pattern>
              </defs>
              <rect width="12" height="10" rx="2" fill="url(#legend-hatch)" stroke={COLORS.bubble} strokeWidth={0.5} />
            </svg>
            Bubble
          </span>
        </div>
      </div>

      <p className="text-xs text-gray-500 mb-2">
        {batchesShown >= 2
          ? `Showing 2 batches (${visualMB} micro-batches)`
          : batchesShown === 1
            ? `Showing 1 batch (${oneBatch} micro-batches)`
            : `Showing ${visualMB} of ${oneBatch} micro-batches`}
        {isDualPipeV && visualMB < 2 * visualPP && ' (degraded: insufficient micro-batches for DualPipeV)'}
      </p>

      {/* Chart */}
      <div ref={containerRef} className="w-full overflow-hidden">
        <svg width={svgWidth} height={svgHeight} className="select-none">
          <defs>
            <pattern id="bubble-hatch" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <line x1="0" y1="0" x2="0" y2="4" stroke="rgba(239, 68, 68, 0.45)" strokeWidth="1.5" />
            </pattern>
          </defs>

          {/* Grid lines */}
          {ticks.map((t) => (
            <line
              key={`grid-${t}`}
              x1={xScale(t)}
              y1={TOP_PAD}
              x2={xScale(t)}
              y2={TOP_PAD + chartHeight}
              stroke={COLORS.gridLine}
              strokeDasharray="3 3"
            />
          ))}

          {/* Y-axis labels */}
          {Array.from({ length: numRows }, (_, row) => (
            <text
              key={`label-${row}`}
              x={labelWidth - 8}
              y={TOP_PAD + row * ROW_HEIGHT + ROW_HEIGHT / 2}
              textAnchor="end"
              dominantBaseline="central"
              fill={COLORS.label}
              fontSize={11}
            >
              {rowLabel(row)}
            </text>
          ))}

          {/* Bubble gaps (idle time) — diagonal hatch */}
          {bubbleGaps.map((gap, i) => {
            const x = xScale(gap.startMs);
            const w = Math.max(2, (gap.durationMs / timeEnd) * chartWidth);
            const y = TOP_PAD + gap.stage * ROW_HEIGHT + BAR_PADDING;
            const h = ROW_HEIGHT - BAR_PADDING * 2;
            return (
              <rect
                key={`bubble-${i}`}
                x={x}
                y={y}
                width={w}
                height={h}
                rx={2}
                fill="url(#bubble-hatch)"
                stroke={COLORS.bubble}
                strokeWidth={1}
                opacity={0.9}
              />
            );
          })}

          {/* Schedule blocks */}
          {blocks.map((blk, i) => {
            const row = rowOf(blk);
            const x = xScale(blk.startMs);
            const w = Math.max(2, (blk.durationMs / timeEnd) * chartWidth);
            const y = TOP_PAD + row * ROW_HEIGHT + BAR_PADDING;
            const h = ROW_HEIGHT - BAR_PADDING * 2;
            const fill = blockColor(blk);
            const label = isDualPipeV
              ? `${blk.type}${blk.type === 'W' || blk.type === 'F&B' ? '' : blk.microBatch}`
              : isInterleaved
                ? `${blk.type === 'forward' ? 'F' : 'B'}${blk.microBatch}·${Math.floor(blk.stage / visualPP)}`
                : `${blk.type === 'forward' ? 'F' : 'B'}${blk.microBatch}`;
            const showLabel = w >= MIN_LABEL_WIDTH;

            return (
              <g key={i}>
                <rect
                  x={x}
                  y={y}
                  width={w}
                  height={h}
                  rx={2}
                  fill={fill}
                  opacity={0.85}
                />
                {showLabel && (
                  <text
                    x={x + w / 2}
                    y={y + h / 2}
                    textAnchor="middle"
                    dominantBaseline="central"
                    fill="white"
                    fontSize={9}
                    fontWeight={500}
                  >
                    {label}
                  </text>
                )}
              </g>
            );
          })}

          {/* X-axis ticks */}
          {ticks.map((t) => (
            <text
              key={`tick-${t}`}
              x={xScale(t)}
              y={TOP_PAD + chartHeight + 16}
              textAnchor="middle"
              fill={COLORS.tick}
              fontSize={10}
            >
              {formatTickLabel(t)}
            </text>
          ))}
        </svg>
      </div>

      {/* Summary */}
      <div className="mt-3 flex items-center gap-4 text-xs text-gray-500">
        <span>Bubble <span className="text-gray-300">{(pipelineBubble * 100).toFixed(1)}%</span></span>
        <span>F = <span className="text-gray-300">{formatTickLabel(F)}</span> / micro-batch</span>
        <span>B = <span className="text-gray-300">{formatTickLabel(B)}</span> / micro-batch</span>
        {isDualPipeV && (
          <span>W = <span className="text-gray-300">{formatTickLabel(W)}</span> / micro-batch</span>
        )}
        {numMicroBatches !== gradientAccumulationSteps && (
          <span className="text-gray-400">
            {numMicroBatches} micro-batches / {gradientAccumulationSteps} accumulation steps
          </span>
        )}
      </div>
    </div>
  );
});
