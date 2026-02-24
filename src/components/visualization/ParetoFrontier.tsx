/**
 * Inference tradeoff charts:
 *   - Batch tab: batch-size vs latency lines (TPOT solid, TTFT dashed) per precision
 *   - Seq Len tab: input seq length vs latency lines per precision, with OOM cutoffs
 *   - TPOT tab:  cost vs TPOT scatter with Pareto frontier
 *
 * Custom SVG — no external charting library. Log-log axes, precision colors,
 * hover tooltip, click-to-apply, precision filter toggles.
 */

import { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import { Scale } from 'lucide-react';
import type { ParetoPoint, ParetoSweepResult } from '../../core/inference/index.ts';
import type { SeqLenSweepResult, SeqLenSweepPoint } from '../../core/inference/index.ts';
import { computeCostPerMToken } from '../../core/inference/index.ts';
import { formatLatency } from '../../types/base.ts';
import { useSimulationStore } from '../../stores/simulation.ts';
import { useConfigStore } from '../../stores/config.ts';
import { Tooltip } from '../ui/Tooltip.tsx';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CHART_HEIGHT = 260;
const PAD = { top: 12, right: 20, bottom: 40, left: 56 };

const PRECISION_COLORS: Record<string, string> = {
  bf16: '#3b82f6',  // blue-500
  fp16: '#3b82f6',
  fp32: '#3b82f6',
  fp8: '#8b5cf6',   // violet-500
  int8: '#8b5cf6',
  int4: '#f97316',  // orange-500
  fp4: '#f97316',
  q4_k_m: '#ec4899', // pink-500  — GGUF 4-bit
  q8_0: '#14b8a6',   // teal-500  — GGUF 8-bit
};

function precisionGroup(prec: string): string {
  if (prec === 'fp8' || prec === 'int8') return 'fp8';
  if (prec === 'int4' || prec === 'fp4') return 'int4';
  if (prec === 'q4_k_m') return 'q4_k_m';
  if (prec === 'q8_0') return 'q8_0';
  return 'bf16';
}

const FRONTIER_COLOR = '#6366f1'; // indigo-500
const CURRENT_COLOR = '#22c55e';  // green-500
const OOM_COLOR = '#ef4444';      // red-500

function precisionColor(prec: string): string {
  return PRECISION_COLORS[prec] ?? '#6b7280';
}

// ---------------------------------------------------------------------------
// Log-scale helpers
// ---------------------------------------------------------------------------

function logScale(value: number, min: number, max: number, pixels: number): number {
  const logMin = Math.log10(min);
  const logMax = Math.log10(max);
  const logVal = Math.log10(Math.max(value, min));
  return ((logVal - logMin) / (logMax - logMin)) * pixels;
}

function tightLogBounds(min: number, max: number): [number, number] {
  const logMin = Math.log10(min);
  const logMax = Math.log10(max);
  const span = logMax - logMin || 1;
  const margin = span * 0.08;
  return [Math.pow(10, logMin - margin), Math.pow(10, logMax + margin)];
}

function logTicks(min: number, max: number): number[] {
  const ticks: number[] = [];
  const startExp = Math.floor(Math.log10(min));
  const endExp = Math.ceil(Math.log10(max));
  for (let e = startExp; e <= endExp; e++) {
    const base = Math.pow(10, e);
    for (const m of [1, 2, 5]) {
      const v = base * m;
      if (v >= min * 0.99 && v <= max * 1.01) ticks.push(v);
    }
  }
  return ticks;
}

function formatTickLabel(v: number): string {
  if (v >= 1000) return `${(v / 1000).toFixed(v >= 10000 ? 0 : 1)}k`;
  if (v >= 1) return v < 10 ? v.toPrecision(2) : String(Math.round(v));
  if (v >= 0.01) return v.toFixed(2);
  return v.toPrecision(2);
}

function formatSeqLenTick(v: number): string {
  return Math.round(v).toLocaleString('en-US');
}

/** Diamond polygon for CB points on TPOT scatter */
function diamondPoints(cx: number, cy: number, r: number): string {
  return `${cx},${cy - r} ${cx + r},${cy} ${cx},${cy + r} ${cx - r},${cy}`;
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface BatchSweepPoint {
  batchSize: number;
  ttft: number;
  tpot: number;
  point: ParetoPoint;
}

type HoverTarget =
  | { kind: 'pareto'; point: ParetoPoint }
  | { kind: 'seqlen'; group: string; seqLen: number; ttft: number; tpot: number; memoryUtil: number; tokensPerSecond: number; isOomBoundary: boolean }
  | { kind: 'current'; metric: 'tpot' | 'ttft' | 'cost-tpot' };

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface ParetoFrontierProps {
  currentCostPerMToken: number;
  currentTtft: number;
  currentTpot: number;
  paretoResult: ParetoSweepResult | null;
  paretoProgress: number;
  currentTP: number;
  currentEP: number;
  currentCB: boolean;
  currentBatchSize: number;
  seqLenSweepResult: SeqLenSweepResult | null;
  currentInputSeqLen: number;
  gpuHourlyRate: number;
  numGPUs: number;
}

export function ParetoFrontier({
  currentCostPerMToken,
  currentTtft,
  currentTpot,
  paretoResult,
  paretoProgress,
  currentTP,
  currentEP,
  currentCB,
  currentBatchSize,
  seqLenSweepResult,
  currentInputSeqLen,
  gpuHourlyRate,
  numGPUs,
}: ParetoFrontierProps) {
  const [tab, setTab] = useState<'batch' | 'seqlen' | 'tpot'>('tpot');
  const [hovered, setHovered] = useState<HoverTarget | null>(null);
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(400);

  // Precision visibility from config store (persisted)
  const visiblePrecisions = useConfigStore(s => s.inference.paretoVisiblePrecisions);
  const togglePrecision = useCallback((group: string) => {
    const current = useConfigStore.getState().inference.paretoVisiblePrecisions;
    useConfigStore.getState().setInferenceParams({
      paretoVisiblePrecisions: { ...current, [group]: !current[group] },
    });
  }, []);

  // Responsive width
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(entries => {
      const w = entries[0]?.contentRect.width;
      if (w && w > 0) setWidth(w);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const allPoints = useMemo(() => paretoResult?.points ?? [], [paretoResult]);

  // Count points per precision group (unfiltered)
  const groupCounts = useMemo(() => {
    const counts: Record<string, number> = { bf16: 0, fp8: 0, int4: 0, q4_k_m: 0, q8_0: 0 };
    for (const p of allPoints) counts[precisionGroup(p.config.weightPrecision)]++;
    return counts;
  }, [allPoints]);

  // Filter by visible precisions
  const points = useMemo(() =>
    allPoints.filter(p => visiblePrecisions[precisionGroup(p.config.weightPrecision)] !== false),
    [allPoints, visiblePrecisions],
  );

  // ===== TPOT scatter data =====

  const frontier = useMemo(() => {
    if (points.length === 0) return [];
    const sorted = [...points].sort((a, b) => a.costPerMToken - b.costPerMToken);
    const result: ParetoPoint[] = [];
    let minLatency = Infinity;
    for (const p of sorted) {
      if (p.tpot < minLatency) {
        result.push(p);
        minLatency = p.tpot;
      }
    }
    return result;
  }, [points]);

  const frontierSet = useMemo(() => new Set(frontier), [frontier]);

  const scatterBounds = useMemo(() => {
    const costs = points.map(p => p.costPerMToken);
    const lats = points.map(p => p.tpot);
    if (currentCostPerMToken > 0 && isFinite(currentCostPerMToken)) costs.push(currentCostPerMToken);
    if (currentTpot > 0 && isFinite(currentTpot)) lats.push(currentTpot);
    if (costs.length === 0 || lats.length === 0) {
      return { xMin: 0.01, xMax: 100, yMin: 0.1, yMax: 10000 };
    }
    const [xLo, xHi] = tightLogBounds(Math.min(...costs), Math.max(...costs));
    const [yLo, yHi] = tightLogBounds(Math.min(...lats), Math.max(...lats));
    return { xMin: xLo, xMax: xHi, yMin: yLo, yMax: yHi };
  }, [points, currentCostPerMToken, currentTpot]);

  // ===== Batch sweep data =====

  const batchGroups = useMemo(() => {
    const groups = new Map<string, BatchSweepPoint[]>();
    for (const p of points) {
      if (p.config.tp !== currentTP) continue;
      if ((p.config.ep ?? 1) !== currentEP) continue;
      if (p.config.continuousBatching !== currentCB) continue;
      const group = precisionGroup(p.config.weightPrecision);
      if (!groups.has(group)) groups.set(group, []);
      groups.get(group)!.push({
        batchSize: p.config.batchSize,
        ttft: p.ttft,
        tpot: p.tpot,
        point: p,
      });
    }
    for (const pts of groups.values()) pts.sort((a, b) => a.batchSize - b.batchSize);
    return groups;
  }, [points, currentTP, currentEP, currentCB]);

  // Ghost batch groups: CB curves shown when static mode is active ("here's what CB gives you")
  const ghostBatchGroups = useMemo(() => {
    if (currentCB) return new Map<string, BatchSweepPoint[]>();
    const groups = new Map<string, BatchSweepPoint[]>();
    for (const p of points) {
      if (p.config.tp !== currentTP) continue;
      if ((p.config.ep ?? 1) !== currentEP) continue;
      if (!p.config.continuousBatching) continue;
      const group = precisionGroup(p.config.weightPrecision);
      if (!groups.has(group)) groups.set(group, []);
      groups.get(group)!.push({
        batchSize: p.config.batchSize,
        ttft: p.ttft,
        tpot: p.tpot,
        point: p,
      });
    }
    for (const pts of groups.values()) pts.sort((a, b) => a.batchSize - b.batchSize);
    return groups;
  }, [points, currentTP, currentEP, currentCB]);

  const batchBounds = useMemo(() => {
    const batches: number[] = [];
    const lats: number[] = [];
    // Include both active and ghost data so axes stay aligned
    for (const groups of [batchGroups, ghostBatchGroups]) {
      for (const pts of groups.values()) {
        for (const bp of pts) {
          batches.push(bp.batchSize);
          lats.push(bp.ttft, bp.tpot);
        }
      }
    }
    if (currentBatchSize > 0) batches.push(currentBatchSize);
    if (currentTtft > 0 && isFinite(currentTtft)) lats.push(currentTtft);
    if (currentTpot > 0 && isFinite(currentTpot)) lats.push(currentTpot);
    if (batches.length === 0 || lats.length === 0) {
      return { xMin: 1, xMax: 16384, yMin: 0.1, yMax: 10000 };
    }
    const [xLo, xHi] = tightLogBounds(Math.min(...batches), Math.max(...batches));
    const [yLo, yHi] = tightLogBounds(Math.min(...lats), Math.max(...lats));
    return { xMin: xLo, xMax: xHi, yMin: yLo, yMax: yHi };
  }, [batchGroups, ghostBatchGroups, currentBatchSize, currentTtft, currentTpot]);

  // ===== Seq len sweep data =====

  const seqLenGroups = useMemo(() => {
    if (!seqLenSweepResult) return new Map<string, SeqLenSweepPoint[]>();
    const groups = new Map<string, SeqLenSweepPoint[]>();
    for (const [prec, pts] of Object.entries(seqLenSweepResult.groups)) {
      if (visiblePrecisions[prec] === false) continue;
      if (pts.length > 0) groups.set(prec, pts);
    }
    return groups;
  }, [seqLenSweepResult, visiblePrecisions]);

  const seqLenBounds = useMemo(() => {
    const seqLens: number[] = [];
    const lats: number[] = [];
    for (const pts of seqLenGroups.values()) {
      for (const sp of pts) {
        seqLens.push(sp.seqLen);
        lats.push(sp.ttft, sp.tpot);
      }
    }
    // Include OOM cutoffs in x range — extend one power of 2 beyond the furthest
    if (seqLenSweepResult) {
      let maxOom = 0;
      for (const [prec, cutoff] of Object.entries(seqLenSweepResult.oomCutoffs)) {
        if (cutoff > 0 && visiblePrecisions[prec] !== false) {
          seqLens.push(cutoff);
          maxOom = Math.max(maxOom, cutoff);
        }
      }
      if (maxOom > 0) seqLens.push(maxOom * 1.2);
    }
    // Include current config position
    if (currentInputSeqLen > 0) seqLens.push(currentInputSeqLen);
    if (currentTtft > 0 && isFinite(currentTtft)) lats.push(currentTtft);
    if (currentTpot > 0 && isFinite(currentTpot)) lats.push(currentTpot);
    if (seqLens.length === 0 || lats.length === 0) {
      return { xMin: 128, xMax: 131072, yMin: 0.1, yMax: 10000 };
    }
    const [xLo, xHi] = tightLogBounds(Math.min(...seqLens), Math.max(...seqLens));
    const [yLo, yHi] = tightLogBounds(Math.min(...lats), Math.max(...lats));
    return { xMin: xLo, xMax: xHi, yMin: yLo, yMax: yHi };
  }, [seqLenGroups, seqLenSweepResult, visiblePrecisions, currentInputSeqLen, currentTtft, currentTpot]);

  // Active bounds
  const { xMin, xMax, yMin, yMax } = tab === 'batch' ? batchBounds : tab === 'seqlen' ? seqLenBounds : scatterBounds;

  const plotW = width - PAD.left - PAD.right;
  const plotH = CHART_HEIGHT - PAD.top - PAD.bottom;

  // Y axis: standard — low at bottom, high at top
  const toX = useCallback((v: number) => PAD.left + logScale(v, xMin, xMax, plotW), [xMin, xMax, plotW]);
  const toY = useCallback((v: number) => PAD.top + plotH - logScale(v, yMin, yMax, plotH), [yMin, yMax, plotH]);

  // Hover handler
  const handleMouseMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current) { setHovered(null); return; }
    const rect = svgRef.current.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    let bestTarget: HoverTarget | null = null;
    let bestDist = 144;

    if (tab === 'batch') {
      for (const pts of batchGroups.values()) {
        for (const bp of pts) {
          const px = toX(bp.batchSize);
          for (const lat of [bp.tpot, bp.ttft]) {
            const py = toY(lat);
            const d = (mx - px) ** 2 + (my - py) ** 2;
            if (d < bestDist) { bestDist = d; bestTarget = { kind: 'pareto', point: bp.point }; }
          }
        }
      }
      // Current config green dots
      if (currentBatchSize > 0 && currentTpot > 0 && isFinite(currentTpot)) {
        const d = (mx - toX(currentBatchSize)) ** 2 + (my - toY(currentTpot)) ** 2;
        if (d < bestDist) { bestDist = d; bestTarget = { kind: 'current', metric: 'tpot' }; }
      }
      if (currentBatchSize > 0 && currentTtft > 0 && isFinite(currentTtft)) {
        const d = (mx - toX(currentBatchSize)) ** 2 + (my - toY(currentTtft)) ** 2;
        if (d < bestDist) { bestDist = d; bestTarget = { kind: 'current', metric: 'ttft' }; }
      }
    } else if (tab === 'seqlen') {
      for (const [group, pts] of seqLenGroups) {
        for (let si = 0; si < pts.length; si++) {
          const sp = pts[si];
          const px = toX(sp.seqLen);
          const isOom = si === pts.length - 1 && !!seqLenSweepResult && seqLenSweepResult.oomCutoffs[group] > 0;
          for (const lat of [sp.tpot, sp.ttft]) {
            const py = toY(lat);
            const d = (mx - px) ** 2 + (my - py) ** 2;
            if (d < bestDist) {
              bestDist = d;
              bestTarget = { kind: 'seqlen', group, seqLen: sp.seqLen, ttft: sp.ttft, tpot: sp.tpot, memoryUtil: sp.memoryUtil, tokensPerSecond: sp.tokensPerSecond, isOomBoundary: isOom };
            }
          }
        }
      }
      // Current config green dots
      if (currentInputSeqLen > 0 && currentTpot > 0 && isFinite(currentTpot)) {
        const d = (mx - toX(currentInputSeqLen)) ** 2 + (my - toY(currentTpot)) ** 2;
        if (d < bestDist) { bestDist = d; bestTarget = { kind: 'current', metric: 'tpot' }; }
      }
      if (currentInputSeqLen > 0 && currentTtft > 0 && isFinite(currentTtft)) {
        const d = (mx - toX(currentInputSeqLen)) ** 2 + (my - toY(currentTtft)) ** 2;
        if (d < bestDist) { bestDist = d; bestTarget = { kind: 'current', metric: 'ttft' }; }
      }
    } else {
      for (const p of points) {
        const px = toX(p.costPerMToken);
        const py = toY(p.tpot);
        const d = (mx - px) ** 2 + (my - py) ** 2;
        if (d < bestDist) { bestDist = d; bestTarget = { kind: 'pareto', point: p }; }
      }
      // Current config green dot
      if (currentCostPerMToken > 0 && isFinite(currentCostPerMToken) && currentTpot > 0 && isFinite(currentTpot)) {
        const d = (mx - toX(currentCostPerMToken)) ** 2 + (my - toY(currentTpot)) ** 2;
        if (d < bestDist) { bestDist = d; bestTarget = { kind: 'current', metric: 'cost-tpot' }; }
      }
    }

    setHovered(bestTarget);
    if (bestTarget) setTooltipPos({ x: e.clientX, y: e.clientY });
  }, [tab, points, batchGroups, seqLenGroups, toX, toY, currentBatchSize, currentTpot, currentTtft, currentInputSeqLen, currentCostPerMToken, seqLenSweepResult]);

  const handleClick = useCallback(() => {
    if (!hovered || hovered.kind === 'current') return;
    if (hovered.kind === 'pareto') {
      const h = hovered.point;
      useConfigStore.getState().setInferenceParams({
        tensorParallel: h.config.tp,
        batchSize: h.config.batchSize,
        weightPrecision: h.config.weightPrecision,
        kvCachePrecision: h.config.kvCachePrecision,
        continuousBatching: h.config.continuousBatching,
        expertParallel: h.config.ep ?? 1,
      });
    } else {
      // seqlen hover — apply seqLen and precision
      const precMap: Record<string, { weight: string; kv: string }> = {
        bf16: { weight: 'bf16', kv: 'bf16' },
        fp8:  { weight: 'fp8',  kv: 'fp8' },
        int4: { weight: 'int4', kv: 'int4' },
      };
      const prec = precMap[hovered.group] ?? precMap.bf16;
      useConfigStore.getState().setInferenceParams({
        inputSeqLen: hovered.seqLen,
        weightPrecision: prec.weight as 'bf16' | 'fp8' | 'int4',
        kvCachePrecision: prec.kv as 'bf16' | 'fp8' | 'int4',
      });
    }
    useSimulationStore.getState().runInferenceSimulation();
  }, [hovered]);

  const xTicks = useMemo(() => logTicks(xMin, xMax), [xMin, xMax]);
  const yTicks = useMemo(() => logTicks(yMin, yMax), [yMin, yMax]);

  // TPOT scatter: frontier line path
  const frontierPath = useMemo(() => {
    if (frontier.length < 2) return '';
    return frontier.map((p, i) => {
      const x = toX(p.costPerMToken);
      const y = toY(p.tpot);
      return `${i === 0 ? 'M' : 'L'}${x},${y}`;
    }).join(' ');
  }, [frontier, toX, toY]);

  // Batch sweep: line paths per precision group
  const batchLinePaths = useMemo(() => {
    if (tab !== 'batch') return [];
    const paths: Array<{ group: string; tpotPath: string; ttftPath: string }> = [];
    for (const [group, pts] of batchGroups) {
      if (pts.length < 2) continue;
      const tpotPath = pts.map((bp, i) =>
        `${i === 0 ? 'M' : 'L'}${toX(bp.batchSize)},${toY(bp.tpot)}`
      ).join(' ');
      const ttftPath = pts.map((bp, i) =>
        `${i === 0 ? 'M' : 'L'}${toX(bp.batchSize)},${toY(bp.ttft)}`
      ).join(' ');
      paths.push({ group, tpotPath, ttftPath });
    }
    return paths;
  }, [tab, batchGroups, toX, toY]);

  // Ghost batch sweep: line paths for opposite batching mode
  const ghostLinePaths = useMemo(() => {
    if (tab !== 'batch') return [];
    const paths: Array<{ group: string; tpotPath: string; ttftPath: string }> = [];
    for (const [group, pts] of ghostBatchGroups) {
      if (pts.length < 2) continue;
      const tpotPath = pts.map((bp, i) =>
        `${i === 0 ? 'M' : 'L'}${toX(bp.batchSize)},${toY(bp.tpot)}`
      ).join(' ');
      const ttftPath = pts.map((bp, i) =>
        `${i === 0 ? 'M' : 'L'}${toX(bp.batchSize)},${toY(bp.ttft)}`
      ).join(' ');
      paths.push({ group, tpotPath, ttftPath });
    }
    return paths;
  }, [tab, ghostBatchGroups, toX, toY]);

  // Seq len sweep: line paths per precision group
  const seqLenLinePaths = useMemo(() => {
    if (tab !== 'seqlen') return [];
    const paths: Array<{ group: string; tpotPath: string; ttftPath: string }> = [];
    for (const [group, pts] of seqLenGroups) {
      if (pts.length < 2) continue;
      const tpotPath = pts.map((sp: SeqLenSweepPoint, i: number) =>
        `${i === 0 ? 'M' : 'L'}${toX(sp.seqLen)},${toY(sp.tpot)}`
      ).join(' ');
      const ttftPath = pts.map((sp: SeqLenSweepPoint, i: number) =>
        `${i === 0 ? 'M' : 'L'}${toX(sp.seqLen)},${toY(sp.ttft)}`
      ).join(' ');
      paths.push({ group, tpotPath, ttftPath });
    }
    return paths;
  }, [tab, seqLenGroups, toX, toY]);

  // Seq len sweep group counts (for legend)
  const seqLenGroupCounts = useMemo(() => {
    const counts: Record<string, number> = { bf16: 0, fp8: 0, int4: 0, q4_k_m: 0, q8_0: 0 };
    if (seqLenSweepResult) {
      for (const [prec, pts] of Object.entries(seqLenSweepResult.groups)) {
        counts[prec] = pts.length;
      }
    }
    return counts;
  }, [seqLenSweepResult]);

  const sweeping = paretoProgress > 0 && paretoProgress < 1;
  const hasData = tab === 'batch' ? batchGroups.size > 0 : tab === 'seqlen' ? seqLenGroups.size > 0 : points.length > 0;
  const hasAnyData = tab === 'seqlen' ? (seqLenSweepResult != null && Object.values(seqLenSweepResult.groups).some(g => g.length > 0)) : allPoints.length > 0;

  // OOM cutoffs visible on seqlen tab
  const visibleOomCutoffs = useMemo(() => {
    if (tab !== 'seqlen' || !seqLenSweepResult) return [];
    const ggufPrecs = new Set(['q2_k', 'q3_k_m', 'q4_k_m', 'q5_k_m', 'q6_k', 'q8_0']);
    return Object.entries(seqLenSweepResult.oomCutoffs)
      .filter(([prec, cutoff]) => cutoff > 0 && visiblePrecisions[prec] !== false && !ggufPrecs.has(prec));
  }, [tab, seqLenSweepResult, visiblePrecisions]);

  // Has any OOM cutoff (for legend)
  const hasOomCutoff = visibleOomCutoffs.length > 0;

  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-5 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-1">
        <h3 className="text-lg font-medium text-white flex items-center gap-2">
          <Scale className="w-5 h-5 text-purple-400" />
          {tab === 'batch' ? 'Batch Sweep' : tab === 'seqlen' ? 'Seq Len Sweep' : 'Cost / Latency'}
        </h3>
        <div className="flex items-center gap-1 bg-gray-800 rounded-lg p-0.5">
          <button
            className={`px-3 py-1 text-xs rounded-md transition-colors ${
              tab === 'seqlen' ? 'bg-gray-700 text-white' : 'text-gray-400 hover:text-gray-300'
            }`}
            onClick={() => setTab('seqlen')}
          >
            Seq Len
          </button>
          <button
            className={`px-3 py-1 text-xs rounded-md transition-colors ${
              tab === 'batch' ? 'bg-gray-700 text-white' : 'text-gray-400 hover:text-gray-300'
            }`}
            onClick={() => setTab('batch')}
          >
            Batch
          </button>
          <button
            className={`px-3 py-1 text-xs rounded-md transition-colors ${
              tab === 'tpot' ? 'bg-gray-700 text-white' : 'text-gray-400 hover:text-gray-300'
            }`}
            onClick={() => setTab('tpot')}
          >
            TPOT
          </button>
        </div>
      </div>
      <p className="text-xs text-gray-500 mb-3 leading-none">
        {tab === 'batch'
          ? 'Batch size trade-offs across precisions.'
          : tab === 'seqlen'
          ? 'Sequence length trade-offs across precisions.'
          : 'Pareto frontier across configurations.'}
      </p>

      {/* Progress bar */}
      {sweeping && tab !== 'seqlen' && (
        <div className="mb-3">
          <div className="h-1 w-full bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-indigo-500 transition-[width] duration-200 ease-linear rounded-full"
              style={{ width: `${(paretoProgress * 100).toFixed(0)}%` }}
            />
          </div>
          <p className="text-xs text-gray-500 mt-1">
            Exploring configurations... {(paretoProgress * 100).toFixed(0)}%
          </p>
        </div>
      )}

      {/* Chart */}
      <div ref={containerRef} className="w-full" style={{ height: CHART_HEIGHT }}>
        <svg
          ref={svgRef}
          width={width}
          height={CHART_HEIGHT}
          className="select-none"
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHovered(null)}
          onClick={handleClick}
          style={{ cursor: hovered && hovered.kind !== 'current' ? 'pointer' : 'default' }}
        >
          {/* Grid lines */}
          {xTicks.map(t => {
            const x = toX(t);
            return <line key={`xg-${t}`} x1={x} y1={PAD.top} x2={x} y2={PAD.top + plotH} stroke="var(--color-chart-grid)" strokeWidth={1} />;
          })}
          {yTicks.map(t => {
            const y = toY(t);
            return <line key={`yg-${t}`} x1={PAD.left} y1={y} x2={PAD.left + plotW} y2={y} stroke="var(--color-chart-grid)" strokeWidth={1} />;
          })}

          {/* X axis labels — BOTTOM */}
          {xTicks.map(t => (
            <text key={`xl-${t}`} x={toX(t)} y={PAD.top + plotH + 16} textAnchor="middle" fill="var(--color-text-muted)" fontSize={11}>
              {formatTickLabel(t)}
            </text>
          ))}
          <text x={PAD.left + plotW / 2} y={PAD.top + plotH + 32} textAnchor="middle" fill="var(--color-text-secondary)" fontSize={12}>
            {tab === 'batch' ? (currentCB ? 'Concurrent Requests (CB)' : 'Batch Size') : tab === 'seqlen' ? 'Input Sequence Length' : 'Cost ($/M tokens)'}
          </text>

          {/* Y axis labels — LEFT */}
          {yTicks.map(t => (
            <text key={`yl-${t}`} x={PAD.left - 8} y={toY(t) + 3} textAnchor="end" fill="var(--color-text-muted)" fontSize={11}>
              {formatTickLabel(t)}
            </text>
          ))}
          <text
            x={14}
            y={PAD.top + plotH / 2}
            textAnchor="middle"
            fill="var(--color-text-secondary)"
            fontSize={12}
            transform={`rotate(-90, 14, ${PAD.top + plotH / 2})`}
          >
            {tab === 'tpot' ? 'TPOT (ms)' : 'Latency (ms)'}
          </text>

          {/* ===== Batch sweep content ===== */}
          {tab === 'batch' && hasData && (
            <>
              {/* Ghost lines (opposite batching mode) */}
              {ghostLinePaths.map(({ group, tpotPath, ttftPath }) => (
                <g key={`gl-${group}`}>
                  <path d={tpotPath} fill="none" stroke={PRECISION_COLORS[group]} strokeWidth={1.5} opacity={0.15} />
                  <path d={ttftPath} fill="none" stroke={PRECISION_COLORS[group]} strokeWidth={1.5} opacity={0.15} strokeDasharray="6 3" />
                </g>
              ))}

              {/* Ghost dots (opposite batching mode) */}
              {Array.from(ghostBatchGroups.entries()).map(([group, pts]) =>
                pts.map((bp, i) => (
                  <g key={`gd-${group}-${i}`}>
                    <circle cx={toX(bp.batchSize)} cy={toY(bp.tpot)} r={2} fill={PRECISION_COLORS[group]} opacity={0.15} />
                    <circle cx={toX(bp.batchSize)} cy={toY(bp.ttft)} r={2} fill="none" stroke={PRECISION_COLORS[group]} strokeWidth={1} opacity={0.15} />
                  </g>
                ))
              )}

              {/* Lines per precision: solid=TPOT, dashed=TTFT */}
              {batchLinePaths.map(({ group, tpotPath, ttftPath }) => (
                <g key={`bl-${group}`}>
                  <path d={tpotPath} fill="none" stroke={PRECISION_COLORS[group]} strokeWidth={2} opacity={0.8} />
                  <path d={ttftPath} fill="none" stroke={PRECISION_COLORS[group]} strokeWidth={2} opacity={0.8} strokeDasharray="6 3" />
                </g>
              ))}

              {/* Dots per precision */}
              {Array.from(batchGroups.entries()).map(([group, pts]) =>
                pts.map((bp, i) => (
                  <g key={`bd-${group}-${i}`}>
                    <circle cx={toX(bp.batchSize)} cy={toY(bp.tpot)} r={3} fill={PRECISION_COLORS[group]} />
                    <circle cx={toX(bp.batchSize)} cy={toY(bp.ttft)} r={3} fill="none" stroke={PRECISION_COLORS[group]} strokeWidth={1.5} />
                  </g>
                ))
              )}

              {/* Current config: TPOT (filled) and TTFT (hollow) */}
              {currentBatchSize > 0 && currentTpot > 0 && isFinite(currentTpot) && (
                <>
                  <circle cx={toX(currentBatchSize)} cy={toY(currentTpot)} r={8} fill="none" stroke={CURRENT_COLOR} strokeWidth={1.5} opacity={0.4}>
                    <animate attributeName="r" values="6;10;6" dur="3s" repeatCount="indefinite" />
                    <animate attributeName="opacity" values="0.4;0.15;0.4" dur="3s" repeatCount="indefinite" />
                  </circle>
                  <circle cx={toX(currentBatchSize)} cy={toY(currentTpot)} r={5} fill={CURRENT_COLOR} />
                </>
              )}
              {currentBatchSize > 0 && currentTtft > 0 && isFinite(currentTtft) && (
                <>
                  <circle cx={toX(currentBatchSize)} cy={toY(currentTtft)} r={8} fill="none" stroke={CURRENT_COLOR} strokeWidth={1.5} opacity={0.4}>
                    <animate attributeName="r" values="6;10;6" dur="3s" repeatCount="indefinite" />
                    <animate attributeName="opacity" values="0.4;0.15;0.4" dur="3s" repeatCount="indefinite" />
                  </circle>
                  <circle cx={toX(currentBatchSize)} cy={toY(currentTtft)} r={5} fill={CURRENT_COLOR} />
                </>
              )}
            </>
          )}

          {/* ===== Seq len sweep content ===== */}
          {tab === 'seqlen' && hasData && (
            <>
              {/* Lines per precision: solid=TPOT, dashed=TTFT */}
              {seqLenLinePaths.map(({ group, tpotPath, ttftPath }) => (
                <g key={`sl-${group}`}>
                  <path d={tpotPath} fill="none" stroke={PRECISION_COLORS[group]} strokeWidth={2} opacity={0.8} />
                  <path d={ttftPath} fill="none" stroke={PRECISION_COLORS[group]} strokeWidth={2} opacity={0.8} strokeDasharray="6 3" />
                </g>
              ))}

              {/* Dots per precision (last dot red if precision OOMs) */}
              {Array.from(seqLenGroups.entries()).map(([group, pts]) =>
                pts.map((sp: SeqLenSweepPoint, i: number) => {
                  const isLast = i === pts.length - 1 && seqLenSweepResult && seqLenSweepResult.oomCutoffs[group] > 0;
                  const color = isLast ? OOM_COLOR : PRECISION_COLORS[group];
                  return (
                    <g key={`sd-${group}-${i}`}>
                      <circle cx={toX(sp.seqLen)} cy={toY(sp.tpot)} r={3} fill={color} />
                      <circle cx={toX(sp.seqLen)} cy={toY(sp.ttft)} r={3} fill="none" stroke={color} strokeWidth={1.5} />
                    </g>
                  );
                })
              )}

              {/* OOM cutoff vertical lines with labels */}
              {visibleOomCutoffs.map(([prec, cutoff]) => {
                const x = toX(cutoff);
                return (
                  <g key={`oom-${prec}`}>
                    <line
                      x1={x} y1={PAD.top} x2={x} y2={PAD.top + plotH}
                      stroke={OOM_COLOR}
                      strokeWidth={2}
                      strokeDasharray="4 4"
                      opacity={0.5}
                    />
                    <text
                      x={x} y={PAD.top - 3}
                      fill={OOM_COLOR} fontSize={10} fontWeight={500} opacity={0.8}
                      textAnchor="middle"
                    >
                      {prec.toUpperCase()}
                    </text>
                  </g>
                );
              })}

              {/* Current config: TPOT (filled) and TTFT (hollow) */}
              {currentInputSeqLen > 0 && currentTpot > 0 && isFinite(currentTpot) && (
                <>
                  <circle cx={toX(currentInputSeqLen)} cy={toY(currentTpot)} r={8} fill="none" stroke={CURRENT_COLOR} strokeWidth={1.5} opacity={0.4}>
                    <animate attributeName="r" values="6;10;6" dur="3s" repeatCount="indefinite" />
                    <animate attributeName="opacity" values="0.4;0.15;0.4" dur="3s" repeatCount="indefinite" />
                  </circle>
                  <circle cx={toX(currentInputSeqLen)} cy={toY(currentTpot)} r={5} fill={CURRENT_COLOR} />
                </>
              )}
              {currentInputSeqLen > 0 && currentTtft > 0 && isFinite(currentTtft) && (
                <>
                  <circle cx={toX(currentInputSeqLen)} cy={toY(currentTtft)} r={8} fill="none" stroke={CURRENT_COLOR} strokeWidth={1.5} opacity={0.4}>
                    <animate attributeName="r" values="6;10;6" dur="3s" repeatCount="indefinite" />
                    <animate attributeName="opacity" values="0.4;0.15;0.4" dur="3s" repeatCount="indefinite" />
                  </circle>
                  <circle cx={toX(currentInputSeqLen)} cy={toY(currentTtft)} r={5} fill={CURRENT_COLOR} />
                </>
              )}
            </>
          )}

          {/* ===== TPOT scatter content ===== */}
          {tab === 'tpot' && hasData && (
            <>
              {/* Non-frontier dots: diamond=CB, circle=static */}
              {points.map((p, i) => {
                if (frontierSet.has(p)) return null;
                const cx = toX(p.costPerMToken);
                const cy = toY(p.tpot);
                const color = precisionColor(p.config.weightPrecision);
                return p.config.continuousBatching ? (
                  <polygon
                    key={`p-${i}`}
                    points={diamondPoints(cx, cy, 3.5)}
                    fill={color}
                    opacity={0.3}
                  />
                ) : (
                  <circle
                    key={`p-${i}`}
                    cx={cx}
                    cy={cy}
                    r={3}
                    fill={color}
                    opacity={0.3}
                  />
                );
              })}

              {/* Frontier line */}
              {frontierPath && (
                <path d={frontierPath} fill="none" stroke={FRONTIER_COLOR} strokeWidth={2} opacity={0.6} />
              )}

              {/* Frontier dots: diamond=CB, circle=static */}
              {frontier.map((p, i) => {
                const cx = toX(p.costPerMToken);
                const cy = toY(p.tpot);
                const color = precisionColor(p.config.weightPrecision);
                return p.config.continuousBatching ? (
                  <polygon
                    key={`f-${i}`}
                    points={diamondPoints(cx, cy, 5)}
                    fill={color}
                    stroke={FRONTIER_COLOR}
                    strokeWidth={1}
                  />
                ) : (
                  <circle
                    key={`f-${i}`}
                    cx={cx}
                    cy={cy}
                    r={4}
                    fill={color}
                    stroke={FRONTIER_COLOR}
                    strokeWidth={1}
                  />
                );
              })}

              {/* Current config */}
              {currentCostPerMToken > 0 && isFinite(currentCostPerMToken) &&
               currentTpot > 0 && isFinite(currentTpot) && (
                <>
                  <circle cx={toX(currentCostPerMToken)} cy={toY(currentTpot)} r={8} fill="none" stroke={CURRENT_COLOR} strokeWidth={1.5} opacity={0.4}>
                    <animate attributeName="r" values="6;10;6" dur="3s" repeatCount="indefinite" />
                    <animate attributeName="opacity" values="0.4;0.15;0.4" dur="3s" repeatCount="indefinite" />
                  </circle>
                  <circle cx={toX(currentCostPerMToken)} cy={toY(currentTpot)} r={5} fill={CURRENT_COLOR} />
                </>
              )}
            </>
          )}

          {/* Hover highlight */}
          {hovered?.kind === 'pareto' && tab === 'batch' && (
            <>
              <circle cx={toX(hovered.point.config.batchSize)} cy={toY(hovered.point.tpot)} r={6} fill="none" stroke="white" strokeWidth={2} />
              <circle cx={toX(hovered.point.config.batchSize)} cy={toY(hovered.point.ttft)} r={6} fill="none" stroke="white" strokeWidth={2} />
            </>
          )}
          {hovered?.kind === 'seqlen' && tab === 'seqlen' && (
            <>
              <circle cx={toX(hovered.seqLen)} cy={toY(hovered.tpot)} r={6} fill="none" stroke="white" strokeWidth={2} />
              <circle cx={toX(hovered.seqLen)} cy={toY(hovered.ttft)} r={6} fill="none" stroke="white" strokeWidth={2} />
            </>
          )}
          {hovered?.kind === 'pareto' && tab === 'tpot' && (
            hovered.point.config.continuousBatching ? (
              <polygon points={diamondPoints(toX(hovered.point.costPerMToken), toY(hovered.point.tpot), 7)} fill="none" stroke="white" strokeWidth={2} />
            ) : (
              <circle cx={toX(hovered.point.costPerMToken)} cy={toY(hovered.point.tpot)} r={6} fill="none" stroke="white" strokeWidth={2} />
            )
          )}
          {hovered?.kind === 'current' && tab === 'batch' && (
            <circle cx={toX(currentBatchSize)} cy={toY(hovered.metric === 'tpot' ? currentTpot : currentTtft)} r={8} fill="none" stroke="white" strokeWidth={2} />
          )}
          {hovered?.kind === 'current' && tab === 'seqlen' && (
            <circle cx={toX(currentInputSeqLen)} cy={toY(hovered.metric === 'tpot' ? currentTpot : currentTtft)} r={8} fill="none" stroke="white" strokeWidth={2} />
          )}
          {hovered?.kind === 'current' && tab === 'tpot' && (
            <circle cx={toX(currentCostPerMToken)} cy={toY(currentTpot)} r={8} fill="none" stroke="white" strokeWidth={2} />
          )}

          {/* Empty state */}
          {!hasData && !sweeping && (
            <text x={PAD.left + plotW / 2} y={PAD.top + plotH / 2} textAnchor="middle" fill="var(--color-text-secondary)" fontSize={13}>
              {hasAnyData ? 'No matching configurations' : 'Run simulation to explore trade-offs'}
            </text>
          )}
        </svg>
      </div>

      {/* Tooltip */}
      {hovered && tooltipPos && (
        <div
          className="fixed z-50 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-xs text-gray-300 pointer-events-none"
          style={{ left: tooltipPos.x + 12, top: tooltipPos.y - 60 }}
        >
          {hovered.kind === 'current' ? (
            <div className="space-y-0.5">
              <div className="font-medium" style={{ color: CURRENT_COLOR }}>Current Config</div>
              <div className="border-t border-gray-700 my-1" />
              {tab === 'batch' && <div>Batch: {currentBatchSize}</div>}
              {tab === 'seqlen' && <div>Seq Len: {formatSeqLenTick(currentInputSeqLen)}</div>}
              {tab === 'tpot' && currentCostPerMToken > 0 && isFinite(currentCostPerMToken) && (
                <div>Cost: ${currentCostPerMToken < 1 ? currentCostPerMToken.toPrecision(3) : currentCostPerMToken.toFixed(2)}/M tok</div>
              )}
              <div>TTFT: {formatLatency(currentTtft)}</div>
              <div>TPOT: {formatLatency(currentTpot)}</div>
            </div>
          ) : hovered.kind === 'pareto' ? (
            <div className="space-y-0.5">
              <div className="font-medium text-white">
                {hovered.point.config.weightPrecision.toUpperCase()}
                {hovered.point.config.continuousBatching ? ' + CB' : ''}
              </div>
              <div>TP={hovered.point.config.tp}, Batch={hovered.point.config.batchSize}</div>
              {hovered.point.config.ep != null && hovered.point.config.ep > 1 && (
                <div>EP={hovered.point.config.ep}</div>
              )}
              <div className="border-t border-gray-700 my-1" />
              <div>Cost: ${hovered.point.costPerMToken < 1 ? hovered.point.costPerMToken.toPrecision(3) : hovered.point.costPerMToken.toFixed(2)}/M tok</div>
              <div>TTFT: {formatLatency(hovered.point.ttft)}</div>
              <div>TPOT: {formatLatency(hovered.point.tpot)}</div>
              <div className="text-gray-500">Mem: {(hovered.point.memoryUtil * 100).toFixed(0)}%</div>
            </div>
          ) : (
            (() => {
              const cost = gpuHourlyRate > 0 ? computeCostPerMToken(hovered.tokensPerSecond, numGPUs, gpuHourlyRate) : 0;
              return (
                <div className="space-y-0.5">
                  <div className="font-medium text-white flex items-center justify-between gap-3">
                    <span>{hovered.group.toUpperCase()}</span>
                    {hovered.isOomBoundary && <span style={{ color: OOM_COLOR }}>OOM</span>}
                  </div>
                  <div>Seq Len: {formatSeqLenTick(hovered.seqLen)}</div>
                  <div className="border-t border-gray-700 my-1" />
                  {cost > 0 && isFinite(cost) && (
                    <div>Cost: ${cost < 1 ? cost.toPrecision(3) : cost.toFixed(2)}/M tok</div>
                  )}
                  <div>TTFT: {formatLatency(hovered.ttft)}</div>
                  <div>TPOT: {formatLatency(hovered.tpot)}</div>
                  <div className="text-gray-500">Mem: {(hovered.memoryUtil * 100).toFixed(0)}%</div>
                </div>
              );
            })()
          )}
        </div>
      )}

      {/* Legend */}
      {hasAnyData && (
        <div className="mt-auto pt-2 text-sm text-gray-400">
          <div className="flex flex-wrap items-center gap-3">
            {([['bf16', 'BF16'], ['fp8', 'FP8'], ['int4', 'INT4'], ['q4_k_m', 'Q4_K_M'], ['q8_0', 'Q8_0']] as const).map(([key, label]) => {
              const count = tab === 'seqlen' ? (seqLenGroupCounts[key] ?? 0) : (groupCounts[key] ?? 0);
              const active = visiblePrecisions[key] !== false;
              const hasPoints = count > 0;
              return (
                <Tooltip key={key} text={!hasPoints ? `${label}: all configs OOM` : active ? `Hide ${label}` : `Show ${label}`}>
                  <button
                    className={`flex items-center gap-1 transition-opacity ${
                      !hasPoints ? 'opacity-25 cursor-default' : active ? 'opacity-100' : 'opacity-40'
                    }`}
                    onClick={hasPoints ? () => togglePrecision(key) : undefined}
                  >
                    <span className="inline-block w-2 h-2 rounded-full" style={{ backgroundColor: PRECISION_COLORS[key] }} />
                    {label}
                  </button>
                </Tooltip>
              );
            })}
          </div>
          <div className="flex flex-wrap items-center gap-3 mt-1">
            <Tooltip text="Your current config">
              <span className="flex items-center gap-1">
                <span className="inline-block w-2 h-2 rounded-full" style={{ backgroundColor: CURRENT_COLOR }} />
                Current
              </span>
            </Tooltip>
            {tab === 'tpot' && (
              <>
                <Tooltip text="Lowest TPOT at each cost">
                  <span className="flex items-center gap-1">
                    <span className="inline-block w-4 h-0.5 rounded" style={{ backgroundColor: FRONTIER_COLOR }} />
                    Frontier
                  </span>
                </Tooltip>
                <Tooltip text="Static batching">
                  <span className="flex items-center gap-1">
                    <svg width="10" height="10" viewBox="0 0 10 10"><circle cx="5" cy="5" r="3.5" fill="var(--color-text-muted)" /></svg>
                    Static
                  </span>
                </Tooltip>
                <Tooltip text="Continuous batching">
                  <span className="flex items-center gap-1">
                    <svg width="10" height="10" viewBox="0 0 10 10"><polygon points="5,1.5 8.5,5 5,8.5 1.5,5" fill="var(--color-text-muted)" /></svg>
                    CB
                  </span>
                </Tooltip>
              </>
            )}
            {(tab === 'batch' || tab === 'seqlen') && (
              <>
                <Tooltip text="Time per output token">
                  <span className="flex items-center gap-1">
                    <span className="inline-block w-4" style={{ borderTop: '2px solid var(--color-text-muted)' }} />
                    TPOT
                  </span>
                </Tooltip>
                <Tooltip text="Time to first token">
                  <span className="flex items-center gap-1">
                    <span className="inline-block w-4" style={{ borderTop: '2px dashed var(--color-text-muted)' }} />
                    TTFT
                  </span>
                </Tooltip>
              </>
            )}
            {tab === 'batch' && ghostBatchGroups.size > 0 && (
              <Tooltip text="CB curves for comparison">
                <span className="flex items-center gap-1 text-gray-500">
                  <span className="inline-block w-4" style={{ borderTop: '2px solid var(--color-text-muted)', opacity: 0.25 }} />
                  Ghost: CB=On
                </span>
              </Tooltip>
            )}
            {tab === 'seqlen' && hasOomCutoff && (
              <Tooltip text="Max sequence length before OOM">
                <span className="flex items-center gap-1">
                  <span className="inline-block w-4" style={{ borderTop: `2px dashed ${OOM_COLOR}` }} />
                  OOM
                </span>
              </Tooltip>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
