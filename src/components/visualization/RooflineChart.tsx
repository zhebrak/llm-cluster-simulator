/**
 * Roofline Model chart — log-log SVG visualization.
 *
 * Shows per-operation arithmetic intensity vs attained TFLOPS,
 * with GPU ceiling lines (compute + memory bandwidth).
 */

import { useMemo, useRef, useState, useEffect, useCallback, memo, useReducer } from 'react';
import { Mountain } from 'lucide-react';
import { createPortal } from 'react-dom';
import { useSimulationStore } from '../../stores/simulation.ts';
import type { ModelSpec } from '../../types/model.ts';
import type { GPUSpec } from '../../types/hardware.ts';
import type { DType } from '../../types/base.ts';
import type { SimulationMetrics } from '../../core/simulation/engine.ts';
import type { InferenceSimulationResult, InferencePrecision } from '../../types/inference.ts';
import {
  computeTrainingRoofline,
  computeInferenceRoofline,
  type RooflineData,
  type RooflinePoint,
  type TrainingRooflineConfig,
  type InferenceRooflineConfig,
} from '../../core/roofline/index.ts';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const X_MIN = 0.1;
const X_MAX = 10000;
const PAD = { top: 20, right: 30, bottom: 50, left: 60 };
const SVG_HEIGHT = 260;

const CATEGORY_COLORS: Record<RooflinePoint['category'], string> = {
  'forward-matmul': '#22c55e',
  'forward-other': '#3b82f6',
  optimizer: '#f97316',
  aggregate: '#ef4444',
  prefill: '#06b6d4',
  decode: '#f97316',
};

const CATEGORY_LABELS: Record<RooflinePoint['category'], string> = {
  'forward-matmul': 'Matmul (fwd/bwd)',
  'forward-other': 'Non-matmul',
  optimizer: 'Optimizer',
  aggregate: 'Measured',
  prefill: 'Prefill',
  decode: 'Decode',
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const log10 = Math.log10;
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

function formatAI(v: number): string {
  if (v >= 1000) return `${(v / 1000).toFixed(1)}K`;
  if (v >= 1) return v.toFixed(1);
  return v.toFixed(2);
}

function formatTFLOPS(v: number): string {
  if (v >= 1000) return `${(v / 1000).toFixed(1)}K`;
  if (v >= 1) return v.toFixed(1);
  if (v >= 0.01) return v.toFixed(2);
  return v.toFixed(3);
}

/** Nice log-spaced tick values across [min, max] (powers of 10). */
function logTicks(min: number, max: number): number[] {
  const ticks: number[] = [];
  const startPow = Math.floor(log10(min));
  const endPow = Math.ceil(log10(max));
  for (let p = startPow; p <= endPow; p++) {
    const base = Math.pow(10, p);
    for (const m of [1, 2, 5]) {
      const v = base * m;
      if (v >= min && v <= max) ticks.push(v);
    }
  }
  return ticks;
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

type TrainingProps = {
  mode: 'training';
  model: ModelSpec;
  gpu: GPUSpec;
  dtype: DType;
  metrics: SimulationMetrics;
  trainingConfig: TrainingRooflineConfig;
  inferenceResult?: never;
  inferenceConfig?: never;
};

type InferenceProps = {
  mode: 'inference';
  model: ModelSpec;
  gpu: GPUSpec;
  dtype: InferencePrecision;
  metrics?: never;
  trainingConfig?: never;
  inferenceResult: InferenceSimulationResult;
  inferenceConfig: InferenceRooflineConfig;
};

type RooflineChartProps = TrainingProps | InferenceProps;

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export const RooflineChart = memo(function RooflineChart(props: RooflineChartProps) {
  const { mode, model, gpu, dtype } = props;
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

  // Only recompute when simulation completes — ignore intermediate sidebar edits
  const simStatus = useSimulationStore(s => s.status);
  const [generation, bumpGeneration] = useReducer((n: number) => n + 1, 0);
  useEffect(() => {
    if (simStatus === 'complete') bumpGeneration();
  }, [simStatus]);

  const data: RooflineData = useMemo(() => {
    void generation; // dependency gate
    if (mode === 'training') {
      return computeTrainingRoofline(model, gpu, dtype as DType, props.metrics, props.trainingConfig);
    }
    return computeInferenceRoofline(model, gpu, dtype as InferencePrecision, props.inferenceResult, props.inferenceConfig);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [generation]);

  const { ceiling, points } = data;

  // Chart dimensions
  const chartWidth = width - PAD.left - PAD.right;
  const chartHeight = SVG_HEIGHT - PAD.top - PAD.bottom;
  const svgWidth = width;
  const svgHeight = SVG_HEIGHT;

  // Ridge point (needed for piecewise X scale)
  const ridge = ceiling.ridgePoint;

  // Y range: bottom aligns with the lower end of the BW line, top at 2× peak
  const bwLineStartTFLOPS = X_MIN * ceiling.peakBandwidthTBps;
  const yMin = bwLineStartTFLOPS * 0.5;
  const yMax = ceiling.peakComputeTFLOPS * 2;

  // Piecewise log X scale: compress memory-bound region (left of ridge),
  // expand compute-bound region (right of ridge) so both ceiling lines
  // appear roughly equal length.
  const RIDGE_FRAC = 0.35; // ridge at 35% from left
  const xScale = useCallback((v: number) => {
    const clamped = clamp(v, X_MIN, X_MAX);
    if (clamped <= ridge) {
      // Memory-bound region: [X_MIN, ridge] → [0, RIDGE_FRAC] of chart
      const t = (log10(clamped) - log10(X_MIN)) / (log10(ridge) - log10(X_MIN));
      return PAD.left + t * RIDGE_FRAC * chartWidth;
    }
    // Compute-bound region: [ridge, X_MAX] → [RIDGE_FRAC, 1] of chart
    const t = (log10(clamped) - log10(ridge)) / (log10(X_MAX) - log10(ridge));
    return PAD.left + (RIDGE_FRAC + t * (1 - RIDGE_FRAC)) * chartWidth;
  }, [chartWidth, ridge]);

  const yScale = useCallback((v: number) => {
    const clamped = clamp(v, yMin, yMax);
    return PAD.top + chartHeight - ((log10(clamped) - log10(yMin)) / (log10(yMax) - log10(yMin))) * chartHeight;
  }, [chartHeight, yMin, yMax]);

  // Ticks
  const xTicks = useMemo(() => logTicks(X_MIN, X_MAX), []);
  const yTicks = useMemo(() => logTicks(yMin, yMax), [yMin, yMax]);

  // Ceiling path: bandwidth slope from X_MIN to ridge, compute flat from ridge to X_MAX
  const ceilingPath = useMemo(() => {
    const bwPoints: string[] = [];
    const steps = 50;
    for (let i = 0; i <= steps; i++) {
      const x = X_MIN * Math.pow(ridge / X_MIN, i / steps);
      const y = Math.min(x * ceiling.peakBandwidthTBps, ceiling.peakComputeTFLOPS);
      bwPoints.push(`${xScale(x)},${yScale(y)}`);
    }
    const bandwidthLine = `M ${bwPoints.join(' L ')}`;
    const computeLine = `M ${xScale(ridge)},${yScale(ceiling.peakComputeTFLOPS)} L ${xScale(X_MAX)},${yScale(ceiling.peakComputeTFLOPS)}`;
    return { bandwidthLine, computeLine };
  }, [xScale, yScale, ceiling, ridge]);

  // Tooltip state
  const [tooltip, setTooltip] = useState<{
    point: RooflinePoint;
    x: number;
    y: number;
  } | null>(null);
  const isTouchRef = useRef(false);

  const handleMouseEnter = useCallback((point: RooflinePoint, event: React.MouseEvent) => {
    if (isTouchRef.current) return;
    const rect = (event.target as SVGElement).closest('svg')?.getBoundingClientRect();
    if (!rect) return;
    setTooltip({
      point,
      x: event.clientX,
      y: event.clientY,
    });
  }, []);

  const handleMouseLeave = useCallback(() => setTooltip(null), []);

  // Collect unique categories for legend
  const legendCategories = useMemo(() => {
    const cats = new Set(points.map(p => p.category));
    return Array.from(cats);
  }, [points]);

  // Dot radius from FLOPs fraction
  const dotRadius = (p: RooflinePoint) => {
    if (p.isAggregate) return 8;
    return Math.max(4, Math.min(12, 4 + 8 * Math.sqrt(p.flopsFraction)));
  };

  // Dot labels omitted — legend below chart is sufficient
  const shouldLabel = (_p: RooflinePoint) => false;

  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-5 animate-slide-up-sm flex flex-col">
      <h3 className="text-lg font-medium text-white mb-1 flex items-center gap-2">
        <Mountain className="w-5 h-5 text-cyan-400" />
        Roofline Model
      </h3>
      <p className="text-xs text-gray-500 mb-3 leading-none">
        {mode === 'training'
          ? 'Memory Bandwidth vs Compute'
          : 'Prefill (compute-bound) vs decode (memory-bound) regimes.'}
      </p>

      <div ref={containerRef} className="w-full overflow-hidden">
        <svg width={svgWidth} height={svgHeight} className="select-none">
          {/* Grid lines */}
          {xTicks.map((t) => (
            <line
              key={`xg-${t}`}
              x1={xScale(t)} y1={PAD.top}
              x2={xScale(t)} y2={PAD.top + chartHeight}
              stroke="var(--color-chart-grid)" strokeDasharray="2 3"
            />
          ))}
          {yTicks.map((t) => (
            <line
              key={`yg-${t}`}
              x1={PAD.left} y1={yScale(t)}
              x2={PAD.left + chartWidth} y2={yScale(t)}
              stroke="var(--color-chart-grid)" strokeDasharray="2 3"
            />
          ))}

          {/* Bandwidth ceiling (amber) */}
          <path
            d={ceilingPath.bandwidthLine}
            fill="none"
            stroke="#f59e0b"
            strokeWidth={2.5}
            opacity={0.9}
          />

          {/* Compute ceiling (green) */}
          <path
            d={ceilingPath.computeLine}
            fill="none"
            stroke="#22c55e"
            strokeWidth={2.5}
            opacity={0.9}
          />

          {/* Ridge point dashed vertical line */}
          {ridge > X_MIN && ridge < X_MAX && (
            <>
              <line
                x1={xScale(ridge)} y1={PAD.top}
                x2={xScale(ridge)} y2={PAD.top + chartHeight}
                stroke="var(--color-chart-axis)" strokeDasharray="4 4"
              />
              <text
                x={xScale(ridge)}
                y={PAD.top - 4}
                textAnchor="middle"
                fill="var(--color-text-muted)"
                fontSize={11}
                fontWeight={500}
              >
                Ridge: {formatAI(ridge)}
              </text>
            </>
          )}

          {/* Ceiling labels — positioned above the lines, larger for readability */}
          {(() => {
            // BW label: placed at ~1/6 along the slope, pushed top-left away from line
            const bwLabelAI = Math.pow(ridge, 0.15) * Math.pow(X_MIN, 0.85);
            const bwLabelX = xScale(bwLabelAI) - 18;
            const bwLabelY = yScale(bwLabelAI * ceiling.peakBandwidthTBps) - 56;
            return (
              <text
                x={Math.max(PAD.left + 8, bwLabelX)}
                y={Math.min(bwLabelY, PAD.top + chartHeight - 16)}
                fill="#f59e0b"
                fontSize={13}
                fontWeight={600}
                opacity={0.9}
              >
                {ceiling.peakBandwidthTBps.toFixed(1)} TB/s
              </text>
            );
          })()}
          <text
            x={PAD.left + chartWidth - 6}
            y={yScale(ceiling.peakComputeTFLOPS) - 12}
            textAnchor="end"
            fill="#22c55e"
            fontSize={13}
            fontWeight={600}
            opacity={0.9}
          >
            {ceiling.peakComputeTFLOPS.toFixed(0)} TFLOPS
          </text>

          {/* Operation dots */}
          {points.map((p, i) => {
            if (p.arithmeticIntensity <= 0 || p.attainedTFLOPS <= 0) return null;
            const cx = xScale(p.arithmeticIntensity);
            const cy = yScale(p.attainedTFLOPS);
            const r = dotRadius(p);
            const color = CATEGORY_COLORS[p.category];

            return (
              <g key={i}>
                {/* Aggregate dot: pulsating ring */}
                {p.isAggregate && (
                  <circle
                    cx={cx} cy={cy} r={r + 3}
                    fill="none"
                    stroke={color}
                    strokeWidth={1.5}
                    opacity={0.4}
                  >
                    <animate attributeName="r" values={`${r + 1};${r + 5};${r + 1}`} dur="3s" repeatCount="indefinite" />
                    <animate attributeName="opacity" values="0.4;0.15;0.4" dur="3s" repeatCount="indefinite" />
                  </circle>
                )}
                <circle
                  cx={cx} cy={cy} r={r}
                  fill={color}
                  opacity={p.isAggregate ? 1.0 : 0.75}
                  stroke={p.isAggregate ? '#fff' : color}
                  strokeWidth={p.isAggregate ? 1.5 : 0.5}
                  className="cursor-pointer"
                  onMouseEnter={(e) => handleMouseEnter(p, e)}
                  onMouseLeave={handleMouseLeave}
                  onTouchStart={() => { isTouchRef.current = true; }}
                />
                {/* Label — boundary-aware positioning */}
                {shouldLabel(p) && (() => {
                  const chartRight = PAD.left + chartWidth;
                  const chartTop = PAD.top;
                  // Decode always labels to the left to avoid overlapping with prefill
                  const placeLeft = p.category === 'decode' || cx > chartRight - chartWidth * 0.3;
                  // If dot is near the top, shift label below
                  const nearTop = cy < chartTop + 20;
                  const labelX = placeLeft ? cx - r - 4 : cx + r + 4;
                  const labelY = nearTop ? cy + r + 12 : cy + r + 14;
                  const anchor = placeLeft ? 'end' : 'start';
                  return (
                    <text
                      x={labelX}
                      y={labelY}
                      textAnchor={anchor}
                      fill="var(--color-text-secondary)"
                      fontSize={12}
                      fontWeight={500}
                      className="pointer-events-none"
                    >
                      {p.label}
                    </text>
                  );
                })()}
              </g>
            );
          })}

          {/* X-axis labels — pixel-distance filtered to prevent overlap */}
          {(() => {
            const MIN_GAP = 36; // minimum px between tick labels
            let lastPx = -Infinity;
            return xTicks.filter(t => {
              const px = xScale(t);
              if (px - lastPx < MIN_GAP) return false;
              lastPx = px;
              return true;
            }).map((t) => (
              <text
                key={`xl-${t}`}
                x={xScale(t)}
                y={PAD.top + chartHeight + 18}
                textAnchor="middle"
                fill="var(--color-text-muted)"
                fontSize={11}
              >
                {formatAI(t)}
              </text>
            ));
          })()}
          <text
            x={PAD.left + chartWidth / 2}
            y={PAD.top + chartHeight + 40}
            textAnchor="middle"
            fill="var(--color-text-secondary)"
            fontSize={12}
            fontWeight={500}
          >
            Arithmetic Intensity (FLOPs/byte)
          </text>

          {/* Y-axis labels */}
          {yTicks.filter((_, i) => i % 2 === 0 || yTicks.length < 8).map((t) => (
            <text
              key={`yl-${t}`}
              x={PAD.left - 6}
              y={yScale(t) + 3}
              textAnchor="end"
              fill="var(--color-text-muted)"
              fontSize={11}
            >
              {formatTFLOPS(t)}
            </text>
          ))}
          <text
            x={14}
            y={PAD.top + chartHeight / 2}
            textAnchor="middle"
            fill="var(--color-text-secondary)"
            fontSize={12}
            fontWeight={500}
            transform={`rotate(-90, 14, ${PAD.top + chartHeight / 2})`}
          >
            TFLOPS
          </text>
        </svg>
      </div>

      {/* Legend — pinned to bottom of card */}
      <div className="flex items-center gap-4 mt-auto pt-2 text-sm text-gray-400 flex-wrap">
        {legendCategories.map((cat) => (
          <span key={cat} className="flex items-center gap-1.5">
            <span
              className="inline-block w-2.5 h-2.5 rounded-full"
              style={{ background: CATEGORY_COLORS[cat] }}
            />
            {CATEGORY_LABELS[cat]}
          </span>
        ))}
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-6 h-0.5" style={{ background: '#f59e0b' }} />
          BW ceiling
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-6 h-0.5" style={{ background: '#22c55e' }} />
          Compute ceiling
        </span>
      </div>

      {/* Tooltip */}
      {tooltip && createPortal(
        <div
          className="fixed px-3 py-2.5 bg-gray-800 border border-gray-700 rounded-lg text-xs text-gray-300 z-50 pointer-events-none"
          style={{
            top: tooltip.y - 80,
            left: tooltip.x - 272,
            maxWidth: 260,
          }}
        >
          <div className="font-medium text-white mb-1">{tooltip.point.label}</div>
          <div className="space-y-0.5 text-gray-400">
            <div className="flex justify-between gap-4">
              <span>Arithmetic Intensity</span>
              <span className="text-gray-300">{formatAI(tooltip.point.arithmeticIntensity)} FLOPs/byte</span>
            </div>
            <div className="flex justify-between gap-4">
              <span>{(tooltip.point.isAggregate || tooltip.point.category === 'prefill' || tooltip.point.category === 'decode') ? 'Attained' : 'Roofline Peak'}</span>
              <span className="text-gray-300">{formatTFLOPS(tooltip.point.attainedTFLOPS)} TFLOPS</span>
            </div>
            {!tooltip.point.isAggregate && tooltip.point.flopsFraction > 0.001 && (
              <div className="flex justify-between gap-4">
                <span>FLOPs Share</span>
                <span className="text-gray-300">{(tooltip.point.flopsFraction * 100).toFixed(1)}%</span>
              </div>
            )}
            <div className="flex justify-between gap-4">
              <span>Region</span>
              <span className="text-gray-300">
                {tooltip.point.arithmeticIntensity >= ceiling.ridgePoint
                  ? 'Compute-bound'
                  : 'Memory-bound'}
              </span>
            </div>
            {tooltip.point.isAggregate && props.mode === 'training' && props.metrics && (
              <>
                {props.metrics.fp8HwUtil != null && (
                  <div className="flex justify-between gap-4">
                    <span>FP8 HW Util</span>
                    <span className="text-gray-300">{parseFloat((props.metrics.fp8HwUtil * 100).toFixed(1))}%</span>
                  </div>
                )}
                <div className="flex justify-between gap-4">
                  <span>MFU</span>
                  <span className="text-gray-300">{parseFloat((props.metrics.mfu * 100).toFixed(1))}%</span>
                </div>
                <div className="flex justify-between gap-4">
                  <span>Gap to Peak</span>
                  <span className="text-gray-300">
                    {formatTFLOPS(ceiling.peakComputeTFLOPS - tooltip.point.attainedTFLOPS)} TFLOPS
                  </span>
                </div>
              </>
            )}
          </div>
        </div>,
        document.body,
      )}
    </div>
  );
});
