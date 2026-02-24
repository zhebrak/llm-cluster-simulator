/**
 * ShareCard — off-screen component rendered for PNG export.
 * All colors use inline styles (not Tailwind classes) because html-to-image
 * clones into an SVG foreignObject where CSS cascade/variables don't resolve.
 */

import { forwardRef } from 'react';
import { Zap, TrendingUp, Clock, Cpu, Timer, HardDrive, DollarSign } from 'lucide-react';
import { formatNumber, formatTime, formatBytes, formatLatency, formatTokensShort } from '../../types/base.ts';
import type { MemoryBreakdown, TimingBreakdown } from '../../types/base.ts';
import { AVAILABLE_STRATEGIES } from '../../core/strategies/index.ts';

// ── Theme palette ────────────────────────────────────────────────────────────

interface Palette {
  bg: string;
  card: string;
  cardBorder: string;
  panel: string;
  panelBorder: string;
  barTrack: string;
  divider: string;
  text: string;
  textSecondary: string;
  textMuted: string;
  textDim: string;
  textFaint: string;
  yellow: string;
  green: string;
  blue: string;
  purple: string;
  cyan: string;
  amber: string;
  rose: string;
  orange: string;
  yellowBg: string;
  greenBg: string;
  blueBg: string;
  purpleBg: string;
  cyanBg: string;
  amberBg: string;
  roseBg: string;
  orangeBg: string;
  barBlue: string;
  barRed: string;
  barOrange: string;
  barGreen: string;
  barPurple: string;
  barGray: string;
}

const DARK: Palette = {
  bg: '#111827',
  card: 'rgba(31, 41, 55, 0.8)',
  cardBorder: '#374151',
  panel: 'rgba(31, 41, 55, 0.5)',
  panelBorder: '#374151',
  barTrack: '#1f2937',
  divider: '#374151',
  text: '#ffffff',
  textSecondary: '#e5e7eb',
  textMuted: '#9ca3af',
  textDim: '#6b7280',
  textFaint: '#4b5563',
  yellow: '#facc15',
  green: '#4ade80',
  blue: '#60a5fa',
  purple: '#a78bfa',
  cyan: '#22d3ee',
  amber: '#fbbf24',
  rose: '#fb7185',
  orange: '#fb923c',
  yellowBg: 'rgba(234, 179, 8, 0.1)',
  greenBg: 'rgba(34, 197, 94, 0.1)',
  blueBg: 'rgba(59, 130, 246, 0.1)',
  purpleBg: 'rgba(139, 92, 246, 0.1)',
  cyanBg: 'rgba(34, 211, 238, 0.1)',
  amberBg: 'rgba(251, 191, 36, 0.1)',
  roseBg: 'rgba(251, 113, 133, 0.1)',
  orangeBg: 'rgba(251, 146, 60, 0.1)',
  barBlue: '#3b82f6',
  barRed: '#ef4444',
  barOrange: '#f97316',
  barGreen: '#22c55e',
  barPurple: '#8b5cf6',
  barGray: '#6b7280',
};

const LIGHT: Palette = {
  bg: '#f8f9fc',
  card: 'rgba(240, 244, 248, 0.8)',
  cardBorder: '#e2e8f0',
  panel: 'rgba(240, 244, 248, 0.8)',
  panelBorder: '#e2e8f0',
  barTrack: '#e2e8f0',
  divider: '#e2e8f0',
  text: '#1a2b3c',
  textSecondary: '#1a2b3c',
  textMuted: '#718096',
  textDim: '#a0aec0',
  textFaint: '#718096',
  yellow: '#c9a020',
  green: '#40a070',
  blue: '#4090c0',
  purple: '#8060b0',
  cyan: '#30a0b0',
  amber: '#d97706',
  rose: '#e11d48',
  orange: '#ea580c',
  yellowBg: 'rgba(201, 160, 32, 0.1)',
  greenBg: 'rgba(64, 160, 112, 0.1)',
  blueBg: 'rgba(64, 144, 192, 0.1)',
  purpleBg: 'rgba(128, 96, 176, 0.1)',
  cyanBg: 'rgba(48, 160, 176, 0.1)',
  amberBg: 'rgba(217, 119, 6, 0.1)',
  roseBg: 'rgba(225, 29, 72, 0.1)',
  orangeBg: 'rgba(234, 88, 12, 0.1)',
  barBlue: '#80b8e0',
  barRed: '#e09090',
  barOrange: '#e0b070',
  barGreen: '#80d0a0',
  barPurple: '#b090d0',
  barGray: '#c0c8d0',
};

function getPalette(): Palette {
  const theme = document.documentElement.getAttribute('data-theme');
  return theme === 'light' ? LIGHT : DARK;
}

// ── Strategy display name lookup ────────────────────────────────────────────

function getStrategyDisplayName(strategyType: string): string {
  return AVAILABLE_STRATEGIES.find(s => s.id === strategyType)?.name ?? strategyType.toUpperCase();
}

// ── Inline SVG logo (favicon GPU die) — avoids CORS issues with <img> ──────

function LogoIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="share-bg" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" style={{ stopColor: '#2d3a4a', stopOpacity: 1 }} />
          <stop offset="100%" style={{ stopColor: '#1e2a38', stopOpacity: 1 }} />
        </linearGradient>
        <linearGradient id="share-die" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" style={{ stopColor: '#3d6b6e', stopOpacity: 1 }} />
          <stop offset="100%" style={{ stopColor: '#4a7c7e', stopOpacity: 1 }} />
        </linearGradient>
      </defs>
      <rect width="512" height="512" rx="96" ry="96" fill="url(#share-bg)" />
      <rect x="72" y="108" width="368" height="296" rx="24" ry="24" fill="none" stroke="#6aaeae" strokeWidth="14" />
      <rect x="104" y="152" width="60" height="80" rx="8" fill="#6aaeae" opacity="0.4" />
      <rect x="104" y="280" width="60" height="80" rx="8" fill="#6aaeae" opacity="0.4" />
      <rect x="348" y="152" width="60" height="80" rx="8" fill="#6aaeae" opacity="0.4" />
      <rect x="348" y="280" width="60" height="80" rx="8" fill="#6aaeae" opacity="0.4" />
      <rect x="186" y="144" width="140" height="224" rx="12" ry="12" fill="url(#share-die)" />
      <g fill="#8ecece" opacity="0.75">
        <rect x="200" y="168" width="32" height="32" rx="4" />
        <rect x="240" y="168" width="32" height="32" rx="4" />
        <rect x="280" y="168" width="32" height="32" rx="4" />
        <rect x="200" y="216" width="32" height="32" rx="4" />
        <rect x="240" y="216" width="32" height="32" rx="4" />
        <rect x="280" y="216" width="32" height="32" rx="4" />
        <rect x="200" y="264" width="32" height="32" rx="4" />
        <rect x="240" y="264" width="32" height="32" rx="4" />
        <rect x="280" y="264" width="32" height="32" rx="4" />
        <rect x="200" y="312" width="32" height="32" rx="4" />
        <rect x="240" y="312" width="32" height="32" rx="4" />
        <rect x="280" y="312" width="32" height="32" rx="4" />
      </g>
    </svg>
  );
}

// ── Format helpers ──────────────────────────────────────────────────────────

function formatTrainingTime(hours: number): string {
  if (hours < 24) return `${parseFloat(hours.toFixed(1))} hours`;
  if (hours < 24 * 30) return `${parseFloat((hours / 24).toFixed(1))} days`;
  if (hours < 24 * 365) return `${parseFloat((hours / 24 / 30).toFixed(1))} months`;
  return `${parseFloat((hours / 24 / 365).toFixed(1))} years`;
}

function formatCost(cost: number): string {
  if (cost >= 1e6) return `$${(cost / 1e6).toFixed(1)}M`;
  if (cost >= 1e5) return `$${(cost / 1e3).toFixed(0)}K`;
  if (cost >= 1e3) return `$${(cost / 1e3).toFixed(1)}K`;
  return `$${cost.toFixed(0)}`;
}

// ── Reusable sub-components ─────────────────────────────────────────────────

function ShareMetricCard({ label, value, subValue, color, iconBg, icon, p }: {
  label: string;
  value: string;
  subValue?: string;
  color: string;
  iconBg: string;
  icon: React.ReactNode;
  p: Palette;
}) {
  return (
    <div
      className="rounded-lg p-3 flex-1 min-w-0"
      style={{ backgroundColor: p.card, border: `1px solid ${p.cardBorder}` }}
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-[10px] uppercase tracking-wide" style={{ color: p.textMuted }}>{label}</p>
          <p className="text-xl font-semibold mt-0.5" style={{ color }}>{value}</p>
          {subValue && subValue.split('\n').map((line, i) => (
            <p key={i} className="text-[10px] mt-0.5 whitespace-nowrap" style={{ color: p.textDim }}>{line}</p>
          ))}
        </div>
        <div className="p-1.5 rounded-lg" style={{ backgroundColor: iconBg }}>
          {icon}
        </div>
      </div>
    </div>
  );
}

/** Horizontal bar for memory/timing breakdowns */
function ShareBar({ label, value, max, barColor, p, formatFn }: {
  label: string;
  value: number;
  max: number;
  barColor: string;
  p: Palette;
  formatFn: (v: number) => string;
}) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div>
      <div className="flex justify-between text-xs mb-1" style={{ color: p.textMuted }}>
        <span>{label}</span>
        <span>{formatFn(value)} ({pct.toFixed(1)}%)</span>
      </div>
      <div className="h-2 rounded-full overflow-hidden" style={{ backgroundColor: p.barTrack }}>
        <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: barColor }} />
      </div>
    </div>
  );
}

// ── Props ───────────────────────────────────────────────────────────────────

interface TrainingShareCardProps {
  mode: 'training';
  modelName: string;
  gpuName: string;
  numGPUs: number;
  gpusPerNode: number;
  strategyType: string;
  precision: string;
  // Config parameters
  sequenceLength: number;
  globalBatchSize: number;
  microBatchSize: number;
  activationCheckpointing: boolean;
  checkpointingGranularity: 'full' | 'selective';
  resolvedStoredLayers?: number;
  modelNumLayers?: number;
  flashAttention: boolean;
  sequenceParallel: boolean;
  tpDegree: number;
  ppDegree: number;
  cpDegree: number;
  epDegree: number;
  pipelineSchedule: '1f1b' | 'interleaved-1f1b' | 'dualpipe-v';
  interleavedStages: number;
  finetuningMethod?: string;
  loraRank?: number;
  loraTargetModules?: string;
  // Key metrics
  mfu: number;
  hfu: number;
  modelFlopsMfu?: number;
  fp8HwUtil?: number;
  tokensPerSecondPerGPU: number;
  tokensPerSecondCluster: number;
  stepTimeMs: number;
  tflopsPerGPU: number;
  samplesPerSecond: number;
  // Memory
  memoryPerGPU: MemoryBreakdown;
  gpuMemoryGB: number;
  memoryUtilization: number;
  // Timing
  timing: TimingBreakdown;
  pipelineBubble: number;
  // Training projection (optional — only when timeToTrainHours exists)
  timeToTrainHours?: number;
  targetTokens?: number;
  trainingSteps?: number;
  chinchillaRatio?: number;
  estimatedCost?: number;
  gpuCostPerHour?: number;
  gpuHours?: number;
}

interface InferenceShareCardProps {
  mode: 'inference';
  modelName: string;
  gpuName: string;
  numGPUs: number;
  gpusPerNode: number;
  precision: string;
  ttft: number;
  tpot: number;
  tokensPerSecond: number;
  totalLatency: number;
  // Model info
  totalParams: number;
  activeParams?: number;
  isMoE?: boolean;
  // Memory breakdown
  memoryWeights: number;
  memoryKvCache: number;
  memoryActivations: number;
  memoryTotal: number;
  gpuMemoryGB: number;
  memoryUtilization: number;
  // Runtime
  kvCacheUtilization: number;
  kvCacheMemoryUsed: number;
  prefillTime: number;
  decodeTime: number;
  // Config
  batchSize: number;
  inputSeqLen: number;
  outputSeqLen: number;
  kvCachePrecision: string;
  flashAttention: boolean;
  pagedAttention: boolean;
  continuousBatching: boolean;
  tensorParallel: number;
  expertParallel: number;
  numReplicas: number;
  perReplicaThroughput: number;
  // Speculative decoding
  speculative?: import('../../types/inference.ts').SpeculativeMetrics;
  draftModelName?: string;
  numSpeculativeTokens?: number;
  acceptanceRate?: number;
  // Cost
  costPerMToken?: number;
  gpuCostPerHour?: number;
}

export type ShareCardProps = TrainingShareCardProps | InferenceShareCardProps;

// ── Main component ──────────────────────────────────────────────────────────

export const ShareCard = forwardRef<HTMLDivElement, ShareCardProps>(
  function ShareCard(props, ref) {
    const { mode, modelName, gpuName, numGPUs, gpusPerNode } = props;
    const p = getPalette();

    const numNodes = Math.ceil(numGPUs / gpusPerNode);
    const gpuLabel = `${numGPUs}× ${gpuName}${numNodes > 1 ? ` (${numNodes} nodes)` : ''}`;
    const subtitle = mode === 'training'
      ? `${modelName}  ·  ${gpuLabel}  ·  ${getStrategyDisplayName((props as TrainingShareCardProps).strategyType)}`
      : (() => {
          const inf = props as InferenceShareCardProps;
          const activeStr = inf.isMoE && inf.activeParams
            ? `${formatNumber(inf.activeParams)} active params`
            : `${formatNumber(inf.totalParams)} params`;
          return `${modelName}  ·  ${activeStr}  ·  ${gpuLabel}`;
        })();

    return (
      <div
        ref={ref}
        className="p-6"
        style={{
          width: 880,
          backgroundColor: p.bg,
          color: p.text,
          fontFamily: 'system-ui, -apple-system, sans-serif',
        }}
      >
        {/* Header */}
        <div className="flex items-center gap-3 mb-2">
          <LogoIcon className="w-8 h-8" />
          <span className="text-xl font-semibold" style={{ color: p.textSecondary }}>LLM Cluster Simulator</span>
        </div>

        {/* Config subtitle */}
        <p className="text-sm mb-1" style={{ color: p.textMuted }}>{subtitle}</p>

        {/* Parameter line (training only) */}
        {mode === 'training' && (() => {
          const t = props as TrainingShareCardProps;
          const batchParts = [
            `seq=${t.sequenceLength >= 1024 ? `${t.sequenceLength / 1024}k` : t.sequenceLength}`,
            `GBS=${t.globalBatchSize}`,
            `MBS=${t.microBatchSize}`,
            'AdamW',
          ];
          const flags: string[] = [];
          if (t.finetuningMethod && t.finetuningMethod !== 'full') {
            const label = t.finetuningMethod === 'qlora' ? 'QLoRA' : 'LoRA';
            flags.push(`${label} r=${t.loraRank}`);
          }
          if (t.activationCheckpointing) {
            if (t.checkpointingGranularity === 'selective') {
              flags.push(t.resolvedStoredLayers != null && t.modelNumLayers != null && t.resolvedStoredLayers < t.modelNumLayers
                ? `SAC ${t.resolvedStoredLayers}/${t.modelNumLayers}`
                : 'SAC');
            } else {
              flags.push('AC');
            }
          }
          if (t.flashAttention) flags.push('FA');
          if (t.sequenceParallel) flags.push('SP');
          flags.push(t.precision.toUpperCase());
          const groups = [batchParts.join(' · ')];
          groups.push(flags.join(' · '));
          const parParts: string[] = [];
          if (t.tpDegree > 1) parParts.push(`TP=${t.tpDegree}`);
          if (t.ppDegree > 1) parParts.push(`PP=${t.ppDegree}`);
          if (t.epDegree > 1) parParts.push(`EP=${t.epDegree}`);
          if (t.cpDegree > 1) parParts.push(`CP=${t.cpDegree}`);
          if (parParts.length > 0) groups.push(parParts.join(' · '));
          if (t.ppDegree > 1) {
            const sched = t.pipelineSchedule === 'dualpipe-v'
              ? 'DualPipeV'
              : t.pipelineSchedule === 'interleaved-1f1b'
                ? `Interleaved 1F1B (v=${t.interleavedStages})`
                : '1F1B';
            groups.push(sched);
          }
          return <p className="text-xs mb-4 mt-0.5" style={{ color: p.textDim }}>{groups.join('  |  ')}</p>;
        })()}
        {mode === 'inference' && (() => {
          const inf = props as InferenceShareCardProps;
          const parts = [
            `input=${inf.inputSeqLen >= 1024 ? `${inf.inputSeqLen / 1024}k` : inf.inputSeqLen}`,
            `output=${inf.outputSeqLen >= 1024 ? `${inf.outputSeqLen / 1024}k` : inf.outputSeqLen}`,
            `batch=${inf.batchSize}`,
          ];
          const flags: string[] = [];
          flags.push(`W:${inf.precision.toUpperCase()} · KV:${inf.kvCachePrecision.toUpperCase()}`);
          if (inf.flashAttention) flags.push('FA');
          if (inf.continuousBatching) flags.push('CB');
          if (inf.speculative) {
            const specParts = ['Spec'];
            if (inf.draftModelName) specParts[0] = `Spec: ${inf.draftModelName}`;
            if (inf.numSpeculativeTokens) specParts.push(`K=${inf.numSpeculativeTokens}`);
            if (inf.acceptanceRate != null) specParts.push(`α=${(inf.acceptanceRate * 100).toFixed(0)}%`);
            flags.push(specParts.join(' '));
          }
          const groups = [parts.join(' · ')];
          groups.push(flags.join(' · '));
          const parParts: string[] = [];
          if (inf.tensorParallel > 1) parParts.push(`TP=${inf.tensorParallel}`);
          if (inf.expertParallel > 1) parParts.push(`EP=${inf.expertParallel}`);
          if (parParts.length > 0) groups.push(parParts.join(' · '));
          if (inf.numReplicas > 1) groups.push(`${inf.numReplicas} replicas`);
          return <p className="text-xs mb-4 mt-0.5" style={{ color: p.textDim }}>{groups.join('  |  ')}</p>;
        })()}

        {/* Content */}
        {mode === 'training' ? (
          <TrainingContent p={p} {...props as TrainingShareCardProps} />
        ) : (
          <InferenceContent p={p} {...props as InferenceShareCardProps} />
        )}

        {/* Footer */}
        <div className="mt-4 text-xs text-right" style={{ color: p.textFaint }}>
          simulator.zhebrak.io
        </div>
      </div>
    );
  }
);

// ── Training layout ─────────────────────────────────────────────────────────

function TrainingContent(props: TrainingShareCardProps & { p: Palette }) {
  const { p } = props;
  const showHfu = props.hfu !== props.mfu;
  const showModelFlopsMfu = props.modelFlopsMfu != null;
  const showFp8HwUtil = props.fp8HwUtil != null;
  const { memoryPerGPU, gpuMemoryGB, timing, pipelineBubble } = props;
  const gpuMemBytes = gpuMemoryGB * (1024 ** 3);

  return (
    <div className="space-y-4">
      {/* Row 1: 2×2 metric cards */}
      <div className="grid grid-cols-4 gap-2">
        <ShareMetricCard p={p}
          label={showFp8HwUtil ? 'FP8 HW Util' : 'MFU'}
          value={showFp8HwUtil
            ? `${parseFloat((props.fp8HwUtil! * 100).toFixed(1))}%`
            : `${parseFloat((props.mfu * 100).toFixed(1))}%`}
          subValue={[
            showFp8HwUtil ? `${parseFloat((props.mfu * 100).toFixed(1))}% MFU (vs BF16 peak)` : undefined,
            showHfu ? `${parseFloat((props.hfu * 100).toFixed(1))}% HFU (incl. recompute)` : undefined,
            showModelFlopsMfu ? `${parseFloat((props.modelFlopsMfu! * 100).toFixed(1))}% Model FLOPs MFU` : undefined,
          ].filter(Boolean).join('\n') || 'Model FLOPS Utilization'}
          color={p.yellow} iconBg={p.yellowBg}
          icon={<Zap className="w-4 h-4" style={{ color: p.yellow }} />}
        />
        <ShareMetricCard p={p}
          label="Throughput/GPU"
          value={`${formatNumber(Math.round(props.tokensPerSecondPerGPU))} tok/s`}
          subValue={`Cluster: ${formatNumber(Math.round(props.tokensPerSecondCluster))} tok/s`}
          color={p.purple} iconBg={p.purpleBg}
          icon={<TrendingUp className="w-4 h-4" style={{ color: p.purple }} />}
        />
        <ShareMetricCard p={p}
          label="Step Time"
          value={formatTime(props.stepTimeMs)}
          subValue={`${props.samplesPerSecond >= 100 ? Math.round(props.samplesPerSecond) : parseFloat(props.samplesPerSecond.toFixed(1))} samples/sec`}
          color={p.blue} iconBg={p.blueBg}
          icon={<Clock className="w-4 h-4" style={{ color: p.blue }} />}
        />
        <ShareMetricCard p={p}
          label="TFLOPS/GPU"
          value={props.tflopsPerGPU.toFixed(1)}
          subValue="achieved (useful work)"
          color={p.green} iconBg={p.greenBg}
          icon={<Cpu className="w-4 h-4" style={{ color: p.green }} />}
        />
      </div>

      {/* Row 2: Training Projection — single row */}
      {props.timeToTrainHours != null && (
        <div className="rounded-lg p-3" style={{ backgroundColor: p.panel, border: `1px solid ${p.panelBorder}` }}>
          <h4 className="text-xs font-medium mb-2 flex items-center gap-1.5" style={{ color: p.textSecondary }}>
            <DollarSign className="w-3.5 h-3.5" style={{ color: p.green }} />
            Training Projection
          </h4>
          <div className="grid grid-cols-6 gap-4">
            {props.targetTokens != null && (
              <div>
                <p className="text-[10px]" style={{ color: p.textMuted }}>Target Tokens</p>
                <p className="text-base font-semibold" style={{ color: p.text }}>{formatTokensShort(props.targetTokens)}</p>
              </div>
            )}
            {props.trainingSteps != null && (
              <div>
                <p className="text-[10px]" style={{ color: p.textMuted }}>Training Steps</p>
                <p className="text-base font-semibold" style={{ color: p.text }}>{formatNumber(props.trainingSteps)}</p>
              </div>
            )}
            <div>
              <p className="text-[10px]" style={{ color: p.textMuted }}>Training Time</p>
              <p className="text-base font-semibold" style={{ color: p.text }}>{formatTrainingTime(props.timeToTrainHours)}</p>
            </div>
            <div>
              <p className="text-[10px]" style={{ color: p.textMuted }}>GPU Hours</p>
              <p className="text-base font-semibold" style={{ color: p.text }}>{props.gpuHours != null ? formatNumber(props.gpuHours) : '—'}</p>
            </div>
            {props.chinchillaRatio != null && (
              <div>
                <p className="text-[10px]" style={{ color: p.textMuted }}>vs Chinchilla</p>
                <p className="text-base font-semibold" style={{ color: p.text }}>{props.chinchillaRatio.toFixed(1)}x</p>
              </div>
            )}
            {props.estimatedCost != null && (
              <div>
                <p className="text-[10px]" style={{ color: p.textMuted }}>Est. Cost</p>
                <p className="text-base font-semibold" style={{ color: p.green }}>{formatCost(props.estimatedCost)}</p>
                {props.gpuCostPerHour != null && (
                  <p className="text-[9px]" style={{ color: p.textDim }}>${props.gpuCostPerHour.toFixed(2)}/GPU·hr</p>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Row 3: Memory + Timing side by side */}
      <div className="grid grid-cols-2 gap-4">
        {/* Memory per GPU */}
        <div className="rounded-lg p-3" style={{ backgroundColor: p.panel, border: `1px solid ${p.panelBorder}` }}>
          <h4 className="text-xs font-medium mb-2 flex items-center gap-1.5" style={{ color: p.textSecondary }}>
            <HardDrive className="w-3.5 h-3.5" style={{ color: p.blue }} />
            Memory per GPU
          </h4>
          <div className="space-y-2">
            <ShareBar p={p} label="Parameters" value={memoryPerGPU.parameters} max={gpuMemBytes} barColor={p.barBlue} formatFn={v => formatBytes(v, v >= 1e9 ? 1 : 0)} />
            <ShareBar p={p} label="Gradients" value={memoryPerGPU.gradients} max={gpuMemBytes} barColor={p.barRed} formatFn={v => formatBytes(v, v >= 1e9 ? 1 : 0)} />
            <ShareBar p={p} label="Optimizer" value={memoryPerGPU.optimizerStates} max={gpuMemBytes} barColor={p.barOrange} formatFn={v => formatBytes(v, v >= 1e9 ? 1 : 0)} />
            <ShareBar p={p} label="Activations" value={memoryPerGPU.peakActivations} max={gpuMemBytes} barColor={p.barGreen} formatFn={v => formatBytes(v, v >= 1e9 ? 1 : 0)} />
            <div className="pt-2 mt-1 flex justify-between text-xs" style={{ borderTop: `1px solid ${p.divider}` }}>
              <span style={{ color: p.textMuted }}>Total (incl. overhead)</span>
              <span className="font-medium" style={{ color: p.text }}>{formatBytes(memoryPerGPU.total)} / {gpuMemoryGB} GB ({(props.memoryUtilization * 100).toFixed(1)}%)</span>
            </div>
          </div>
        </div>

        {/* Timing Breakdown */}
        <div className="rounded-lg p-3" style={{ backgroundColor: p.panel, border: `1px solid ${p.panelBorder}` }}>
          <h4 className="text-xs font-medium mb-2 flex items-center gap-1.5" style={{ color: p.textSecondary }}>
            <Clock className="w-3.5 h-3.5" style={{ color: p.purple }} />
            Timing Breakdown
          </h4>
          <div className="space-y-2">
            <ShareBar p={p} label="Forward" value={timing.forward} max={props.stepTimeMs} barColor={p.barGreen} formatFn={v => formatTime(v, 0)} />
            <ShareBar p={p} label="Backward" value={timing.backward} max={props.stepTimeMs} barColor={p.barBlue} formatFn={v => formatTime(v, 0)} />
            <ShareBar p={p} label="Communication" value={timing.communication + timing.scaleOverhead} max={props.stepTimeMs} barColor={p.barPurple} formatFn={v => formatTime(v, 0)} />
            <ShareBar p={p} label="Optimizer" value={timing.optimizer} max={props.stepTimeMs} barColor={p.barOrange} formatFn={v => formatTime(v, 0)} />
            <div className="pt-2 mt-1 grid grid-cols-2 gap-4 text-sm" style={{ borderTop: `1px solid ${p.divider}` }}>
              <div>
                <span style={{ color: p.textMuted }}>Exposed Comm.</span>
                <p className="font-medium" style={{ color: p.text }}>{(((timing.communication + timing.scaleOverhead - timing.overlap) / props.stepTimeMs) * 100).toFixed(1)}%</p>
              </div>
              {pipelineBubble > 0 && (
                <div>
                  <span style={{ color: p.textMuted }}>Pipeline Bubble</span>
                  <p className="font-medium" style={{ color: p.text }}>{(pipelineBubble * 100).toFixed(1)}%</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Inference layout ─────────────────────────────────────────────────────────

function InferenceContent(props: InferenceShareCardProps & { p: Palette }) {
  const { p } = props;
  const gpuMemBytes = props.gpuMemoryGB * (1024 ** 3);

  return (
    <div className="space-y-4">
      {/* Row 1: key metrics — 5 columns when cost available, 2×2 otherwise */}
      <div className={`grid ${props.costPerMToken != null && props.costPerMToken > 0 ? 'grid-cols-5' : 'grid-cols-2'} gap-2`}>
        <ShareMetricCard p={p}
          label="TTFT"
          value={formatLatency(props.ttft)}
          subValue="Time to First Token"
          color={p.amber} iconBg={p.amberBg}
          icon={<Timer className="w-4 h-4" style={{ color: p.amber }} />}
        />
        <ShareMetricCard p={p}
          label="TPOT"
          value={formatLatency(props.tpot)}
          subValue="Time Per Output Token"
          color={p.rose} iconBg={p.roseBg}
          icon={<Clock className="w-4 h-4" style={{ color: p.rose }} />}
        />
        <ShareMetricCard p={p}
          label="Throughput"
          value={`${formatNumber(Math.round(props.tokensPerSecond))} tok/s`}
          subValue={props.numReplicas > 1 ? `${formatNumber(Math.round(props.perReplicaThroughput))} tok/s × ${props.numReplicas} replicas` : 'Single replica'}
          color={p.purple} iconBg={p.purpleBg}
          icon={<TrendingUp className="w-4 h-4" style={{ color: p.purple }} />}
        />
        <ShareMetricCard p={p}
          label="Total Latency"
          value={formatLatency(props.totalLatency)}
          subValue="Total generation time"
          color={p.blue} iconBg={p.blueBg}
          icon={<Zap className="w-4 h-4" style={{ color: p.blue }} />}
        />
        {props.costPerMToken != null && props.costPerMToken > 0 && (
          <ShareMetricCard p={p}
            label="Est. Cost"
            value={props.costPerMToken >= 100 ? `$${props.costPerMToken.toFixed(0)}/M` :
                   props.costPerMToken >= 1 ? `$${props.costPerMToken.toFixed(2)}/M` :
                   `$${props.costPerMToken.toPrecision(3)}/M`}
            subValue={props.gpuCostPerHour != null ? `$${props.gpuCostPerHour.toFixed(2)}/GPU·hr` : 'per million tokens'}
            color={p.green} iconBg={p.greenBg}
            icon={<DollarSign className="w-4 h-4" style={{ color: p.green }} />}
          />
        )}
      </div>

      {/* Row 2: Memory per GPU + Runtime side by side */}
      <div className="grid grid-cols-2 gap-4">
        {/* Memory per GPU */}
        <div className="rounded-lg p-3" style={{ backgroundColor: p.panel, border: `1px solid ${p.panelBorder}` }}>
          <h4 className="text-xs font-medium mb-2 flex items-center gap-1.5" style={{ color: p.textSecondary }}>
            <HardDrive className="w-3.5 h-3.5" style={{ color: p.blue }} />
            Memory per GPU
          </h4>
          <div className="space-y-2">
            <ShareBar p={p} label="Model Weights" value={props.memoryWeights} max={gpuMemBytes} barColor={p.barBlue} formatFn={v => formatBytes(v, v >= 1e9 ? 1 : 0)} />
            <ShareBar p={p} label="KV Cache" value={props.memoryKvCache} max={gpuMemBytes} barColor={p.barPurple} formatFn={v => formatBytes(v, v >= 1e9 ? 1 : 0)} />
            <ShareBar p={p} label="Activations" value={props.memoryActivations} max={gpuMemBytes} barColor={p.barGreen} formatFn={v => formatBytes(v, v >= 1e9 ? 1 : 0)} />
            <div className="pt-2 mt-1 flex justify-between text-xs" style={{ borderTop: `1px solid ${p.divider}` }}>
              <span style={{ color: p.textMuted }}>Total (incl. overhead)</span>
              <span className="font-medium" style={{ color: p.text }}>{formatBytes(props.memoryTotal)} / {props.gpuMemoryGB} GB ({(props.memoryUtilization * 100).toFixed(1)}%)</span>
            </div>
          </div>
        </div>

        {/* Runtime */}
        <div className="rounded-lg p-3" style={{ backgroundColor: p.panel, border: `1px solid ${p.panelBorder}` }}>
          <h4 className="text-xs font-medium mb-2 flex items-center gap-1.5" style={{ color: p.textSecondary }}>
            <Timer className="w-3.5 h-3.5" style={{ color: p.purple }} />
            Runtime
          </h4>
          <div className="space-y-2">
            <ShareBar p={p} label="Cache Utilization" value={props.kvCacheUtilization} max={100} barColor={p.barPurple} formatFn={v => `${v.toFixed(1)}%`} />
            <div className="flex justify-between text-[10px] -mt-1 mb-1" style={{ color: p.textDim }}>
              <span>KV cache</span>
              <span>{formatBytes(props.kvCacheMemoryUsed)}</span>
            </div>
            <ShareBar p={p} label="Prefill Time" value={props.prefillTime} max={props.totalLatency} barColor={p.barBlue} formatFn={v => formatLatency(v)} />
            <ShareBar p={p} label="Decode Time" value={props.decodeTime} max={props.totalLatency} barColor={p.barGreen} formatFn={v => formatLatency(v)} />
          </div>
        </div>
      </div>

      {/* Row 3: Speculative Decoding (if enabled) */}
      {props.speculative && (
        <div className="rounded-lg p-3" style={{ backgroundColor: p.panel, border: `1px solid ${p.panelBorder}` }}>
          <h4 className="text-xs font-medium mb-2 flex items-center gap-1.5" style={{ color: p.textSecondary }}>
            <Zap className="w-3.5 h-3.5" style={{ color: p.cyan }} />
            Speculative Decoding
          </h4>
          <div className="grid grid-cols-4 gap-4">
            {props.draftModelName && (
              <div>
                <p className="text-[10px]" style={{ color: p.textMuted }}>Draft Model</p>
                <p className="text-sm font-semibold" style={{ color: p.text }}>{props.draftModelName}</p>
              </div>
            )}
            <div>
              <p className="text-[10px]" style={{ color: p.textMuted }}>Speedup</p>
              <p className="text-sm font-semibold" style={{ color: p.green }}>{props.speculative.speedup.toFixed(2)}x</p>
            </div>
            <div>
              <p className="text-[10px]" style={{ color: p.textMuted }}>Accepted Tokens</p>
              <p className="text-sm font-semibold" style={{ color: p.text }}>{props.speculative.expectedAcceptedTokens.toFixed(1)}</p>
            </div>
            <div>
              <p className="text-[10px]" style={{ color: p.textMuted }}>Effective TPOT</p>
              <p className="text-sm font-semibold" style={{ color: p.text }}>{formatLatency(props.speculative.effectiveTpot)}</p>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
