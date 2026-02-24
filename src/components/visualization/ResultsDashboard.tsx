/**
 * Results dashboard showing simulation metrics
 * Supports training, inference, and comparison modes
 */

import { useState, useRef, useCallback, useMemo, useEffect } from 'react';
import { createPortal } from 'react-dom';

/** Animates width from 0 to target via CSS transition (replaces framer-motion). */
function AnimatedBar({ width, delay = 0, className }: { width: string; delay?: number; className: string }) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => setMounted(true));
    return () => cancelAnimationFrame(id);
  }, []);
  return (
    <div
      className={`${className} transition-[width] duration-500 ease-out`}
      style={{ width: mounted ? width : '0%', transitionDelay: `${delay}s` }}
    />
  );
}
import { Activity, Cpu, HardDrive, Zap, Clock, DollarSign, TrendingUp, Timer, Info, Lightbulb, Copy, Check, Link2, ImageIcon, Download, ChevronDown, ChevronRight } from 'lucide-react';
import { useSimulationStore } from '../../stores/simulation.ts';
import { useConfigStore, type TrainingGoal } from '../../stores/config.ts';
import { formatBytes, formatNumber, formatInteger, formatTime, formatLatency, formatTokensShort } from '../../types/base.ts';
import type { ModelSpec } from '../../types/index.ts';
import type { InferenceSimulationResult } from '../../types/inference.ts';
import type { SimulationMetrics } from '../../core/simulation/index.ts';
import type { LoraTargetModules } from '../../core/strategies/lora.ts';

import type { SimulationResult } from '../../types/index.ts';
import { InfoPanel } from './InfoPanel.tsx';
import { AssumptionsPanel } from './AssumptionsPanel.tsx';
import { PipelineTimeline } from './PipelineTimeline.tsx';
import { GPUGridPanel, InferenceGPUGridPanel } from './GPUGrid.tsx';

import { exportConfigToJSON, type ExportableConfig, type ExportableInferenceConfig } from '../../utils/export.ts';
import { toExponent, buildShareURL } from '../../utils/share.ts';
import { getGPUHourlyRate } from '../../core/cost/index.ts';
import { ShareCard, type ShareCardProps } from './ShareCard.tsx';
import { copyImageToClipboard, downloadImage } from '../../utils/png-export.ts';
import { ParetoFrontier } from './ParetoFrontier.tsx';
import { RooflineChart } from './RooflineChart.tsx';
import { Tooltip } from '../ui/Tooltip.tsx';

function CollapsibleDeepDive({ modelSpec, gpu, precision, metrics, configSnapshot, numGPUs, sequenceLength, globalBatchSize, pipelineSnapshot }: {
  modelSpec: ModelSpec;
  gpu: import('../../types/hardware.ts').GPUSpec;
  precision: import('../../types/base.ts').DType;
  metrics: SimulationMetrics;
  configSnapshot: NonNullable<import('../../stores/simulation.ts').TrainingSimulationState['configSnapshot']> | null;
  numGPUs: number;
  sequenceLength: number;
  globalBatchSize: number;
  pipelineSnapshot: { ppDegree: number; numMicroBatches: number; gradientAccumulationSteps: number; pipelineSchedule: '1f1b' | 'interleaved-1f1b' | 'dualpipe-v'; interleavedStages: number };
}) {
  const [open, setOpen] = useState(false);
  const hasPipeline = pipelineSnapshot.ppDegree > 1;
  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-xl mt-4">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-5 py-3 text-left cursor-pointer hover:bg-gray-800/50 transition-colors rounded-xl"
      >
        <Activity className="w-5 h-5 text-blue-400 flex-shrink-0" />
        <h3 className="text-lg font-medium text-white flex-1">Deep Dive</h3>
        <ChevronRight className={`w-4 h-4 text-gray-500 transition-transform ${open ? 'rotate-90' : ''}`} />
      </button>
      {open && (
        <div className="px-5 pb-5 space-y-6">
          {hasPipeline && (
            <div>
              <PipelineTimeline
                ppDegree={pipelineSnapshot.ppDegree}
                numMicroBatches={pipelineSnapshot.numMicroBatches}
                gradientAccumulationSteps={pipelineSnapshot.gradientAccumulationSteps}
                totalForwardMs={metrics.timing.forward}
                totalBackwardMs={metrics.timing.backward}
                pipelineBubble={metrics.pipelineBubble}
                pipelineSchedule={pipelineSnapshot.pipelineSchedule}
                interleavedStages={pipelineSnapshot.interleavedStages}
              />
            </div>
          )}
          <div>
            <RooflineChart
              mode="training"
              model={modelSpec}
              gpu={gpu}
              dtype={precision}
              metrics={metrics}
              trainingConfig={{
                tp: configSnapshot?.tpDegree ?? 1,
                pp: configSnapshot?.ppDegree ?? 1,
                dp: configSnapshot?.dpDegree ?? Math.max(1, numGPUs / ((configSnapshot?.tpDegree ?? 1) * (configSnapshot?.ppDegree ?? 1))),
                ep: configSnapshot?.epDegree ?? 1,
                seqLength: sequenceLength,
                microBatchSize: configSnapshot?.microBatchSize ?? 1,
                globalBatchSize,
                activationCheckpointing: configSnapshot?.activationCheckpointing ?? true,
                checkpointingGranularity: configSnapshot?.checkpointingGranularity ?? 'full',
                flashAttention: configSnapshot?.flashAttention ?? true,
                dpType: (configSnapshot?.strategyType?.includes('fsdp') ? 'fsdp' : configSnapshot?.strategyType?.includes('zero-1') ? 'zero-1' : 'ddp') as 'ddp' | 'fsdp' | 'zero-1',
                sequenceParallel: configSnapshot?.sequenceParallel ?? false,
                finetuningMethod: (configSnapshot?.finetuningMethod as 'full' | 'lora' | 'qlora') ?? 'full',
                loraRank: configSnapshot?.loraRank as number | undefined,
                loraTargetModules: configSnapshot?.loraTargetModules as LoraTargetModules | undefined,
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function SimulationDisclaimer() {
  return (
    <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-3 text-xs text-gray-400">
      <div className="flex items-start gap-2">
        <Info className="w-4 h-4 mt-0.5 flex-shrink-0 text-gray-500" />
        <p>
          Theoretical estimates validated against published benchmarks. Actual performance will vary.
        </p>
      </div>
    </div>
  );
}

// Textarea fallback for clipboard copy in non-HTTPS contexts
function copyFallback(text: string): void {
  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.style.position = 'fixed';
  textarea.style.opacity = '0';
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand('copy');
  document.body.removeChild(textarea);
}

/** Export button: click copies PNG to clipboard, chevron dropdown offers "Download PNG" and "Copy JSON". */
function ExportButton({ onCopy, onDownload, onCopyConfig, copied, configCopied, disabled }: {
  onCopy: () => void;
  onDownload: () => void;
  onCopyConfig: () => void;
  copied: boolean;
  configCopied: boolean;
  disabled: boolean;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  const anyCopied = copied || configCopied;

  return (
    <div ref={ref} className="relative flex">
      <button
        onClick={onCopy}
        disabled={disabled}
        className={`flex items-center gap-1.5 pl-3 pr-1.5 py-2 text-sm rounded-l-lg transition-colors cursor-pointer ${
          anyCopied ? 'bg-green-500/20 text-green-400' : 'bg-gray-800 hover:bg-gray-700 text-gray-300'
        } disabled:opacity-50 disabled:cursor-not-allowed`}
      >
        {anyCopied ? <Check className="w-4 h-4" /> : <ImageIcon className="w-4 h-4" />}
        {anyCopied ? 'Copied!' : 'Export'}
      </button>
      <button
        onClick={() => setOpen(v => !v)}
        disabled={disabled}
        className={`flex items-center px-1 py-2 text-sm rounded-r-lg border-l transition-colors cursor-pointer ${
          anyCopied ? 'bg-green-500/20 text-green-400 border-green-500/30' : 'bg-gray-800 hover:bg-gray-700 text-gray-300 border-gray-700'
        } disabled:opacity-50 disabled:cursor-not-allowed`}
      >
        <ChevronDown className="w-3.5 h-3.5" />
      </button>
      {open && (
        <div className="absolute top-full right-0 mt-1 bg-gray-800 border border-gray-700 rounded-lg shadow-xl z-50 min-w-[160px] py-1">
          <button
            onClick={() => { onDownload(); setOpen(false); }}
            className="flex items-center gap-2 w-full px-3 py-2 text-sm text-gray-300 hover:bg-gray-700 cursor-pointer"
          >
            <Download className="w-4 h-4" />
            Download PNG
          </button>
          <button
            onClick={() => { onCopyConfig(); setOpen(false); }}
            className="flex items-center gap-2 w-full px-3 py-2 text-sm text-gray-300 hover:bg-gray-700 cursor-pointer"
          >
            {configCopied ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
            {configCopied ? 'Copied!' : 'Copy JSON'}
          </button>
        </div>
      )}
    </div>
  );
}

// Training goal display names
const GOAL_LABELS: Record<TrainingGoal, string> = {
  chinchilla: 'Chinchilla Optimal',
  'heavy-overtrain': 'Heavy Overtrain',
  finetune: 'Fine-tuning',
  custom: 'Custom',
};

// Per-GPU-hour cost lookup — sourced from shared cost module

// Shared Analysis panel for both training and inference
function AnalysisPanel({ bottleneck, bottleneckColor, recommendations }: {
  bottleneck: string;
  bottleneckColor: string;
  recommendations: string[];
}) {
  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-5">
      <h3 className="text-lg font-medium text-white mb-4 flex items-center gap-2">
        <Lightbulb className="w-5 h-5 text-yellow-400" />
        Analysis
      </h3>
      <div className="flex gap-6">
        <div className="flex-shrink-0">
          <div className="flex items-center gap-3">
            <span className="text-gray-400">Bottleneck:</span>
            <span className={`px-3 py-1.5 rounded-lg text-sm font-medium ${bottleneckColor}`}>
              {bottleneck}
            </span>
          </div>
        </div>
        {recommendations.length > 0 ? (
          <>
            <div className="w-px bg-gray-700 self-stretch" />
            <div className="flex-1 min-w-0">
              <ul className="space-y-1">
                {recommendations.map((rec, i) => (
                  <li key={i} className="text-base text-gray-300 flex items-start gap-2">
                    <span className="text-indigo-400 flex-shrink-0">•</span>
                    <span className="leading-snug">{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
          </>
        ) : (
          <>
            <div className="w-px bg-gray-700 self-stretch" />
            <div className="flex-1 min-w-0">
              <ul className="space-y-1">
                <li className="text-base text-gray-400 flex items-start gap-2">
                  <span className="text-indigo-400 flex-shrink-0">•</span>
                  <span className="leading-snug">Well-configured for this model and cluster.</span>
                </li>
              </ul>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

interface TrainingProjectionProps {
  metrics: SimulationMetrics;
  maxSteps: number;
  globalBatchSize: number;
  sequenceLength: number;
  numGPUs: number;
  modelParams: number;
  activeParams: number;  // For Chinchilla calculation (active params for MoE)
  trainingGoal: TrainingGoal;
  targetTokens: number;
  gpuId?: string;
}

function TrainingProjection({
  metrics,
  maxSteps,
  globalBatchSize,
  sequenceLength,
  numGPUs,
  modelParams,
  activeParams,
  trainingGoal,
  targetTokens,
  gpuId,
}: TrainingProjectionProps) {
  const totalTokens = maxSteps * globalBatchSize * sequenceLength;
  const gpuHours = metrics.timeToTrainHours! * numGPUs;

  // Chinchilla optimal tokens for comparison (uses active params for MoE)
  const chinchillaOptimal = activeParams * 20;
  const tokensVsChinchilla = totalTokens / chinchillaOptimal;

  // Format training time
  const formatTrainingTime = (hours: number) => {
    if (hours < 24) return `${parseFloat(hours.toFixed(1))} hours`;
    if (hours < 24 * 30) return `${parseFloat((hours / 24).toFixed(1))} days`;
    if (hours < 24 * 365) return `${parseFloat((hours / 24 / 30).toFixed(1))} months`;
    return `${parseFloat((hours / 24 / 365).toFixed(1))} years`;
  };

  // Get GPU-specific pricing (custom override or default)
  const customPrice = useConfigStore.getState().pricePerGPUHour;
  const gpuPricing = getGPUHourlyRate(gpuId || '');
  const estimatedCostPerGPUHour = customPrice ?? gpuPricing.rate;
  const estimatedCost = gpuHours * estimatedCostPerGPUHour;

  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-5">
      <h3 className="text-lg font-medium text-white mb-4 flex items-center gap-2">
        <DollarSign className="w-5 h-5 text-green-400" />
        Training Projection
        <span className="ml-2 px-2 py-0.5 text-xs bg-gray-800 rounded text-gray-400">
          {GOAL_LABELS[trainingGoal]}
        </span>
      </h3>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <div>
          <p className="text-xs text-gray-400">Target Tokens</p>
          <p className="text-lg font-semibold text-white">{formatTokensShort(targetTokens)}</p>
        </div>
        <div>
          <p className="text-xs text-gray-400">Training Steps</p>
          <p className="text-lg font-semibold text-white">{formatNumber(maxSteps)}</p>
        </div>
        <div>
          <p className="text-xs text-gray-400">Training Time</p>
          <p className="text-lg font-semibold text-white">
            {formatTrainingTime(metrics.timeToTrainHours!)}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-400">GPU Hours</p>
          <p className="text-lg font-semibold text-white">{formatNumber(gpuHours)}</p>
        </div>
        <div>
          <p className="text-xs text-gray-400">vs Chinchilla</p>
          <p className="text-lg font-semibold text-white">
            {tokensVsChinchilla.toFixed(1)}x
          </p>
        </div>
        <div className="relative group">
          <p className="text-xs text-gray-400 flex items-center gap-1">
            Est. Cost
            <Info className="w-3 h-3 text-gray-500 cursor-help" />
          </p>
          <p className="text-lg font-semibold text-green-400">
            ${estimatedCost >= 1e6 ? `${(estimatedCost / 1e6).toFixed(1)}M` :
              estimatedCost >= 1e5 ? `${(estimatedCost / 1e3).toFixed(0)}K` :
              estimatedCost >= 1e3 ? `${(estimatedCost / 1e3).toFixed(1)}K` :
              estimatedCost.toFixed(0)}
          </p>
          {/* Pricing assumptions tooltip */}
          <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-xs text-gray-300 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-10">
            <div className="font-medium text-white mb-1">{customPrice != null ? 'Custom Rate' : 'Pricing Assumptions'}</div>
            <div>{customPrice != null ? `$${customPrice.toFixed(2)}/GPU-hour` : `${gpuPricing.name}: $${gpuPricing.rate.toFixed(2)}/GPU-hour`}</div>
            <div className="text-gray-400 mt-1">{customPrice != null ? 'User-specified rate' : 'Based on est. cloud rates (2026)'}</div>
            <div className="text-gray-400">Excludes storage, network, support</div>
            <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-gray-800"></div>
          </div>
        </div>
      </div>

      {/* Context note */}
      <div className="mt-4 pt-3 border-t border-gray-800 text-xs text-gray-500">
        <p>
          Chinchilla optimal: {formatTokensShort(chinchillaOptimal)} tokens for {formatNumber(activeParams)}{activeParams !== modelParams ? ' active' : ''} params.
          {tokensVsChinchilla < 0.8 && ' Consider training longer for better downstream performance.'}
          {tokensVsChinchilla > 2 && ' Heavy overtraining may have diminishing returns.'}
        </p>
      </div>
    </div>
  );
}

interface MetricCardProps {
  label: string;
  value: string;
  subValue?: string;
  icon: React.ReactNode;
  color: string;
  children?: React.ReactNode;
}

function MetricCard({ label, value, subValue, icon, color, children }: MetricCardProps) {
  return (
    <div
      className="bg-gray-900/50 border border-gray-800 rounded-xl p-4 animate-slide-up"
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs text-gray-400 uppercase tracking-wide">{label}</p>
          <p className={`text-2xl font-semibold mt-1 ${color}`}>{value}</p>
          {subValue && subValue.split('\n').map((line, i) => (
            <p key={i} className="text-xs text-gray-500 mt-1">{line}</p>
          ))}
          {children}
        </div>
        <div className={`p-2 rounded-lg ${color.replace('text-', 'bg-').replace('400', '500/10')}`}>
          {icon}
        </div>
      </div>
    </div>
  );
}

interface ProgressBarProps {
  label: React.ReactNode;
  value: number;
  max: number;
  color: string;
  showPercentage?: boolean;
  unit?: 'percent' | 'bytes' | 'time';
}

function ProgressBar({ label, value, max, color, unit = 'percent' }: ProgressBarProps) {
  const percentage = Math.min((value / max) * 100, 100);

  const formatLabel = () => {
    const pct = `${percentage.toFixed(1)}%`;
    if (unit === 'bytes') return `${formatBytes(value, value >= 1e9 ? 1 : 0)} (${pct})`;
    if (unit === 'time') return `${formatTime(value, 0)} (${pct})`;
    return pct;
  };

  return (
    <div>
      <div className="flex justify-between text-xs text-gray-400 mb-1">
        <span>{label}</span>
        <span>{formatLabel()}</span>
      </div>
      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
        <AnimatedBar width={`${percentage}%`} className={`h-full ${color} rounded-full`} />
      </div>
    </div>
  );
}

interface InferenceResultsProps {
  result: InferenceSimulationResult;
  gpuMemoryGB: number;
  modelName: string;
  modelSpec: ModelSpec | null;
  gpuName: string;
  numGPUs: number;
  tensorParallel: number;
  expertParallel: number;
  continuousBatching: boolean;
  batchSize: number;
}

function InferenceResults({ result, gpuMemoryGB, modelName, modelSpec, gpuName, numGPUs, tensorParallel, expertParallel: simExpertParallel, continuousBatching: simContinuousBatching, batchSize: simBatchSize }: InferenceResultsProps) {
  const { latency, throughput, memory, kvCacheState, utilization, speculative } = result;
  const isCustomModel = useConfigStore(s => s.modelId.startsWith('custom-'));
  const paretoResult = useSimulationStore(s => s.inference.paretoResult);
  const paretoProgress = useSimulationStore(s => s.inference.paretoProgress);
  const seqLenSweepResult = useSimulationStore(s => s.inference.seqLenSweepResult);
  const simInputSeqLen = useSimulationStore(s => s.inference.inputSeqLen);
  const clusterConfig = useConfigStore.getState().clusterConfig;
  const gpu = clusterConfig?.node.gpu ?? null;
  const node = clusterConfig?.node ?? null;
  const gpuId = useConfigStore.getState().gpuId;
  const gpusPerReplica = Math.max(1, tensorParallel) * Math.max(1, simExpertParallel);
  const numReplicas = Math.max(1, Math.floor(numGPUs / gpusPerReplica));
  const perReplicaThroughput = numReplicas > 0 ? throughput.tokensPerSecond / numReplicas : 0;

  // Cost per million tokens
  // throughput.tokensPerSecond already includes all replicas
  const customPrice = useConfigStore.getState().pricePerGPUHour;
  const gpuPricing = getGPUHourlyRate(gpuId || '');
  const effectiveRate = customPrice ?? gpuPricing.rate;
  const costPerMToken = throughput.tokensPerSecond > 0
    ? (effectiveRate * numGPUs / 3600) / throughput.tokensPerSecond * 1e6
    : 0;

  const [specTooltip, setSpecTooltip] = useState<{ top: number; left: number } | null>(null);
  const specIconRef = useRef<HTMLSpanElement>(null);
  const isTouchRef = useRef(false);
  const showSpecTooltip = useCallback(() => {
    if (isTouchRef.current) return;
    const rect = specIconRef.current?.getBoundingClientRect();
    if (rect) setSpecTooltip({ top: rect.top, left: rect.right + 8 });
  }, []);
  const hideSpecTooltip = useCallback(() => setSpecTooltip(null), []);

  const [copied, setCopied] = useState(false);
  const [shareCopied, setShareCopied] = useState(false);
  const [exportCopied, setExportCopied] = useState(false);
  const [exporting, setExporting] = useState(false);
  const shareCardRef = useRef<HTMLDivElement>(null);

  const inferenceShareCardProps: ShareCardProps = useMemo(() => {
    const inf = useConfigStore.getState().inference;
    return {
      mode: 'inference' as const,
      modelName,
      gpuName,
      numGPUs,
      gpusPerNode: useConfigStore.getState().gpusPerNode,
      precision: inf.weightPrecision,
      ttft: latency.ttft,
      tpot: latency.tpot,
      tokensPerSecond: throughput.tokensPerSecond,
      totalLatency: latency.totalLatency,
      // Model info
      totalParams: modelSpec?.totalParams ?? 0,
      activeParams: modelSpec?.isMoE ? (modelSpec.activeParams ?? modelSpec.totalParams) : undefined,
      isMoE: modelSpec?.isMoE,
      // Memory breakdown
      memoryWeights: memory.weights,
      memoryKvCache: memory.kvCache,
      memoryActivations: memory.activations,
      memoryTotal: memory.total,
      gpuMemoryGB,
      memoryUtilization: utilization.memoryCapacityUtilization,
      // Runtime
      kvCacheUtilization: kvCacheState.utilizationPercent,
      kvCacheMemoryUsed: kvCacheState.memoryUsed,
      prefillTime: latency.prefillTime,
      decodeTime: latency.decodeTime,
      // Config
      batchSize: inf.batchSize,
      inputSeqLen: inf.inputSeqLen,
      outputSeqLen: inf.outputSeqLen,
      kvCachePrecision: inf.kvCachePrecision,
      flashAttention: inf.flashAttention,
      pagedAttention: inf.pagedAttention,
      continuousBatching: inf.continuousBatching,
      tensorParallel: tensorParallel,
      expertParallel: simExpertParallel,
      numReplicas,
      perReplicaThroughput,
      // Speculative decoding
      speculative: speculative ?? undefined,
      draftModelName: speculative ? (useConfigStore.getState().inference.draftModelSpec?.name ?? useConfigStore.getState().inference.draftModelId ?? undefined) : undefined,
      numSpeculativeTokens: speculative ? useConfigStore.getState().inference.numSpeculativeTokens : undefined,
      acceptanceRate: speculative ? useConfigStore.getState().inference.acceptanceRate : undefined,
      // Cost
      costPerMToken: costPerMToken > 0 ? costPerMToken : undefined,
      gpuCostPerHour: effectiveRate > 0 ? effectiveRate : undefined,
    };
  }, [modelName, gpuName, numGPUs, gpuMemoryGB, modelSpec, latency, throughput, memory, kvCacheState, utilization, tensorParallel, simExpertParallel, numReplicas, perReplicaThroughput, costPerMToken, effectiveRate, speculative]);

  const handleExportImage = useCallback(async (download: boolean) => {
    if (!shareCardRef.current || exporting) return;
    setExporting(true);
    try {
      if (download) {
        const slug = `${modelName}-${gpuName}`.toLowerCase().replace(/[^a-z0-9]+/g, '-');
        await downloadImage(shareCardRef.current, `llm-sim-${slug}.png`);
      } else {
        await copyImageToClipboard(shareCardRef.current);
        setExportCopied(true);
        setTimeout(() => setExportCopied(false), 2000);
      }
    } finally {
      setExporting(false);
    }
  }, [exporting, modelName, gpuName]);

  const handleShare = () => {
    const url = buildShareURL();
    if (navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(url).then(() => {
        setShareCopied(true);
        setTimeout(() => setShareCopied(false), 2000);
      }).catch(() => { copyFallback(url); setShareCopied(true); setTimeout(() => setShareCopied(false), 2000); });
    } else {
      copyFallback(url);
      setShareCopied(true);
      setTimeout(() => setShareCopied(false), 2000);
    }
  };
  const handleCopyConfig = () => {
    const config = useConfigStore.getState();
    const inf = config.inference;
    const infTP = inf.tensorParallel || 1;
    const infEP = inf.expertParallel || 1;
    const infIsMoE = config.modelSpec?.isMoE ?? false;

    const exportConfig: ExportableInferenceConfig = {
      version: '1.0',
      timestamp: new Date().toISOString(),
      mode: 'inference',
      config: {
        modelId: config.modelId,
        clusterId: config.clusterId,
        gpuId: config.gpuId,
        numGPUs: config.numGPUs,
        gpusPerNode: config.gpusPerNode,
        batchSize: inf.batchSize,
        inputSeqLen: inf.inputSeqLen,
        outputSeqLen: inf.outputSeqLen,
        weightPrecision: inf.weightPrecision,
        kvCachePrecision: inf.kvCachePrecision,
        flashAttention: inf.flashAttention,
        pagedAttention: inf.pagedAttention,
        continuousBatching: inf.continuousBatching,
        ...(infTP > 1 ? { tensorParallel: infTP } : {}),
        ...(infIsMoE && infEP > 1 ? { expertParallel: infEP } : {}),
        ...(inf.speculativeDecoding ? {
          speculativeDecoding: true,
          draftModelId: inf.draftModelId ?? undefined,
          numSpeculativeTokens: inf.numSpeculativeTokens,
          acceptanceRate: inf.acceptanceRate,
        } : {}),
        // Pricing
        ...(config.pricePerGPUHour != null ? { pricePerGPUHour: config.pricePerGPUHour } : {}),
      },
      metrics: {
        ttftMs: +latency.ttft.toFixed(2),
        tpotMs: +latency.tpot.toFixed(3),
        totalLatencyMs: +latency.totalLatency.toFixed(1),
        tokensPerSecond: Math.round(throughput.tokensPerSecond),
        prefillTokensPerSecond: Math.round(throughput.prefillTokensPerSecond),
        decodeTokensPerSecond: Math.round(throughput.decodeTokensPerSecond),
        numReplicas,
        memory: {
          weightsGB: +(memory.weights / (1024 ** 3)).toFixed(2),
          kvCacheGB: +(memory.kvCache / (1024 ** 3)).toFixed(2),
          activationsGB: +(memory.activations / (1024 ** 3)).toFixed(2),
          overheadGB: +(memory.overhead / (1024 ** 3)).toFixed(2),
          totalPerGPU_GB: +(memory.total / (1024 ** 3)).toFixed(2),
        },
        ...(costPerMToken > 0 ? { costPerMTokens: +costPerMToken.toFixed(4) } : {}),
      },
    };
    const json = exportConfigToJSON(exportConfig);
    try {
      const textarea = document.createElement('textarea');
      textarea.value = json;
      textarea.style.position = 'fixed';
      textarea.style.opacity = '0';
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      navigator.clipboard?.writeText(json).then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      });
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
        <h2 className="text-2xl font-semibold text-white">Inference Results</h2>
        <p className="text-gray-400 mt-1 flex items-center gap-1.5">
          {modelName} on {formatInteger(numGPUs)}x {gpuName}
          {(modelSpec || gpu) && (
            <span
              ref={specIconRef}
              className="inline-flex"
              onMouseEnter={showSpecTooltip}
              onMouseLeave={hideSpecTooltip}
              onTouchStart={() => { isTouchRef.current = true; }}
            >
              <Info className="w-3.5 h-3.5 text-gray-500 cursor-help" />
            </span>
          )}
          {specTooltip && createPortal(
            <div
              className="fixed bg-gray-800 border border-gray-700 rounded-lg text-xs text-gray-300 z-50 pointer-events-none flex"
              style={{ top: specTooltip.top, left: specTooltip.left }}
            >
              {/* Left: model / chip / network specs */}
              <div className="px-4 py-3 w-72 shrink-0">
              {modelSpec && (
                <>
                  <div className="font-medium text-white mb-1.5">{modelSpec.name}</div>
                  <div className="space-y-0.5 text-gray-400">
                    <div className="flex justify-between"><span>Parameters</span><span className="text-gray-300">{formatNumber(modelSpec.totalParams)}</span></div>
                    <div className="flex justify-between"><span>Layers</span><span className="text-gray-300">{modelSpec.numLayers}</span></div>
                    <div className="flex justify-between"><span>Hidden size</span><span className="text-gray-300">{formatInteger(modelSpec.hiddenSize)}</span></div>
                    <div className="flex justify-between"><span>Attn heads</span><span className="text-gray-300">{modelSpec.numAttentionHeads}{modelSpec.attentionType === 'mla' ? ' (MLA)' : modelSpec.numKvHeads !== modelSpec.numAttentionHeads ? ` (${modelSpec.numKvHeads} KV)` : ''}</span></div>
                    <div className="flex justify-between"><span>Vocab size</span><span className="text-gray-300">{formatInteger(modelSpec.vocabSize)}</span></div>
                    <div className="flex justify-between"><span>Max seq length</span><span className="text-gray-300">{formatInteger(modelSpec.maxSeqLength)}</span></div>
                    {modelSpec.isMoE && modelSpec.numExperts && (
                      <div className="flex justify-between"><span>Experts</span><span className="text-gray-300">{modelSpec.numActiveExperts}/{modelSpec.numExperts} active</span></div>
                    )}
                  </div>
                </>
              )}
              {modelSpec && gpu && (
                <div className="border-t border-gray-700 my-2"></div>
              )}
              {gpu && (
                <>
                  <div className="font-medium text-white mb-1.5">{gpu.name}</div>
                  <div className="space-y-0.5 text-gray-400">
                    <div className="flex justify-between"><span>Memory</span><span className="text-gray-300">{gpu.memoryGB} GB</span></div>
                    <div className="flex justify-between"><span>Mem bandwidth</span><span className="text-gray-300">{gpu.memoryBandwidthTBps} TB/s</span></div>
                    <div className="flex justify-between"><span>BF16 / FP16</span><span className="text-gray-300">{gpu.bf16TFLOPS || '—'} / {gpu.fp16TFLOPS} TFLOPS</span></div>
                    {gpu.fp8TFLOPS > 0 && (
                      <div className="flex justify-between"><span>FP8</span><span className="text-gray-300">{gpu.fp8TFLOPS} TFLOPS</span></div>
                    )}
                    <div className="flex justify-between"><span>TDP</span><span className="text-gray-300">{gpu.tdpWatts}W</span></div>
                    {gpu.estimated && (
                      <div className="text-gray-500 mt-1 text-[10px]">Limited real-world benchmark data</div>
                    )}
                  </div>
                </>
              )}
              {node && (
                <>
                  <div className="border-t border-gray-700 my-2"></div>
                  <div className="font-medium text-white mb-1.5">Network</div>
                  <div className="space-y-0.5 text-gray-400">
                    <div className="flex justify-between"><span>Intra-node</span><span className="text-gray-300">{node.intraNodeInterconnect.name}</span></div>
                    <div className="flex justify-between"><span>Intra-node BW</span><span className="text-gray-300">{node.intraNodeInterconnect.bandwidthGBps} GB/s</span></div>
                    {clusterConfig && clusterConfig.numNodes > 1 && (
                      <>
                        <div className="flex justify-between"><span>Inter-node</span><span className="text-gray-300">{node.interNodeInterconnect.name}</span></div>
                        <div className="flex justify-between"><span>Inter-node BW</span><span className="text-gray-300">{node.interNodeInterconnect.bandwidthGBps} GB/s</span></div>
                      </>
                    )}
                  </div>
                </>
              )}
              </div>
              {/* Right: GPU partitioning grid */}
              {numGPUs > 1 && (
                <div className="border-l border-gray-700 px-3 py-3 max-w-80 max-h-[70vh] overflow-auto">
                  <InferenceGPUGridPanel embedded />
                </div>
              )}
            </div>,
            document.body,
          )}
        </p>
        </div>
        <div className="flex items-center gap-2">
          {isCustomModel ? (
          <Tooltip text="Sharing is not available for custom models">
          <button
            disabled
            className="flex items-center gap-2 px-3 py-2 text-sm rounded-lg bg-gray-700 text-gray-500 cursor-not-allowed"
          >
            <Link2 className="w-4 h-4" />
            Share
          </button>
          </Tooltip>
          ) : (
          <button
            onClick={handleShare}
            className={`flex items-center gap-2 px-3 py-2 text-sm rounded-lg transition-colors ${
              shareCopied
                ? 'bg-green-500/20 text-green-400 cursor-pointer'
                : 'bg-accent hover:bg-accent text-white cursor-pointer'
            }`}
            aria-label="Copy shareable link to clipboard"
          >
            {shareCopied ? <Check className="w-4 h-4" /> : <Link2 className="w-4 h-4" />}
            {shareCopied ? 'Copied!' : 'Share'}
          </button>
          )}
          <ExportButton
            onCopy={() => handleExportImage(false)}
            onDownload={() => handleExportImage(true)}
            onCopyConfig={handleCopyConfig}
            copied={exportCopied}
            configCopied={copied}
            disabled={exporting}
          />
        </div>
      </div>

      {/* Off-screen share card for PNG export */}
      {createPortal(
        <div style={{ position: 'fixed', left: -9999, top: 0, pointerEvents: 'none' }}>
          <ShareCard ref={shareCardRef} {...inferenceShareCardProps} />
        </div>,
        document.body,
      )}

      {/* Key Latency Metrics */}
      <div className={`grid grid-cols-1 md:grid-cols-2 ${costPerMToken > 0 ? 'lg:grid-cols-5' : 'lg:grid-cols-4'} gap-4`}>
        <MetricCard
          label="TTFT"
          value={formatLatency(latency.ttft)}
          subValue="Time to First Token"
          icon={<Timer className="w-5 h-5 text-amber-400" />}
          color="text-amber-400"
        />
        <MetricCard
          label="TPOT"
          value={formatLatency(latency.tpot)}
          subValue="Time Per Output Token"
          icon={<Clock className="w-5 h-5 text-rose-400" />}
          color="text-rose-400"
        />
        <MetricCard
          label="Throughput"
          value={`${formatNumber(Math.round(throughput.tokensPerSecond))} tok/s`}
          subValue={numReplicas > 1 ? `${formatNumber(Math.round(perReplicaThroughput))} tok/s per replica × ${numReplicas}` : 'Single replica'}
          icon={<TrendingUp className="w-5 h-5 text-purple-400" />}
          color="text-purple-400"
        />
        <MetricCard
          label="Total Latency"
          value={formatLatency(latency.totalLatency)}
          subValue="Total generation time"
          icon={<Zap className="w-5 h-5 text-blue-400" />}
          color="text-blue-400"
        />
        {costPerMToken > 0 && (
          <MetricCard
            label="Cost"
            value={costPerMToken >= 100 ? `$${costPerMToken.toFixed(0)}` :
                   costPerMToken >= 1 ? `$${costPerMToken.toFixed(2)}` :
                   `$${costPerMToken.toPrecision(3)}`}
            subValue="per M tokens"
            icon={<DollarSign className="w-5 h-5 text-green-400" />}
            color="text-green-400"
          />
        )}
      </div>

      {/* Memory and Utilization */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Memory Breakdown */}
        <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-5">
          <h3 className="text-lg font-medium text-white mb-4 flex items-center gap-2">
            <HardDrive className="w-5 h-5 text-blue-400" />
            Memory per GPU
          </h3>
          <div className="space-y-3">
            <ProgressBar
              label="Model Weights"
              value={memory.weights}
              max={gpuMemoryGB * (1024 ** 3)}
              color="bg-blue-500"
            />
            <ProgressBar
              label="KV Cache"
              value={memory.kvCache}
              max={gpuMemoryGB * (1024 ** 3)}
              color="bg-purple-500"
            />
            <ProgressBar
              label="Activations"
              value={memory.activations}
              max={gpuMemoryGB * (1024 ** 3)}
              color="bg-green-500"
            />
            <div className="pt-2 mt-2 border-t border-gray-700">
              <div className="flex justify-between text-sm">
                <span className="text-gray-400 flex items-center gap-1">
                  Total
                  <div className="relative group/infmem inline-flex">
                    <Info className="w-3 h-3 text-gray-500 cursor-help" />
                    <div className="absolute bottom-1/2 left-full translate-y-1/2 ml-2 px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-xs text-gray-300 opacity-0 group-hover/infmem:opacity-100 transition-opacity pointer-events-none z-10 w-72">
                      <div className="font-medium text-white mb-1.5">Includes hidden overhead (~{formatBytes(memory.overhead, 0)})</div>
                      <div className="space-y-1 text-gray-400 leading-relaxed">
                        <p><span className="text-gray-300">CUDA + fragmentation</span> — driver context, cuDNN/cuBLAS handles, allocator block rounding</p>
                        <p><span className="text-gray-300">Runtime buffers</span> — attention scratch space, sampling workspace</p>
                      </div>
                      <div className="absolute top-1/2 -translate-y-1/2 right-full border-4 border-transparent border-r-gray-800"></div>
                    </div>
                  </div>
                </span>
                <span className="text-white font-medium">
                  {formatBytes(memory.total)} / {gpuMemoryGB} GB
                </span>
              </div>
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>Utilization</span>
                <span className={utilization.memoryCapacityUtilization >= 0.9 ? 'text-red-400 font-medium' : utilization.memoryCapacityUtilization >= 0.8 ? 'text-amber-400 font-medium' : ''}>{(utilization.memoryCapacityUtilization * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        </div>

        {/* KV Cache Details */}
        <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-5">
          <h3 className="text-lg font-medium text-white mb-4 flex items-center gap-2">
            <Timer className="w-5 h-5 text-purple-400" />
            Runtime
          </h3>
          <div className="space-y-4">
            <div>
              <ProgressBar
                label={
                  <span className="flex items-center gap-1">
                    Cache Utilization
                    <span className="relative group/pa inline-flex">
                      <Info className="w-3 h-3 text-gray-500 cursor-help" />
                      <span className="absolute bottom-1/2 left-full translate-y-1/2 ml-2 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-xs text-gray-300 opacity-0 group-hover/pa:opacity-100 transition-opacity pointer-events-none z-10 w-48">
                        Assumes Paged Attention (no KV cache fragmentation)
                        <span className="absolute top-1/2 -translate-y-1/2 right-full border-4 border-transparent border-r-gray-800"></span>
                      </span>
                    </span>
                  </span>
                }
                value={kvCacheState.utilizationPercent}
                max={100}
                color="bg-purple-500"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>KV cache</span>
                <span>{formatBytes(kvCacheState.memoryUsed)}</span>
              </div>
            </div>
            <ProgressBar
              label="Prefill Time"
              value={latency.prefillTime}
              max={latency.totalLatency}
              color="bg-cyan-500"
              unit="time"
            />
            <ProgressBar
              label="Decode Time"
              value={latency.decodeTime}
              max={latency.totalLatency}
              color="bg-blue-500"
              unit="time"
            />
          </div>
        </div>
      </div>

      {/* Analysis */}
      <AnalysisPanel
        bottleneck={utilization.bottleneck.replace('_', ' ')}
        bottleneckColor={utilization.bottleneck === 'memory_capacity' ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'}
        recommendations={result.recommendations}
      />

      {/* Speculative Decoding (if enabled) */}
      {speculative && (
        <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-5">
          <h3 className="text-lg font-medium text-white mb-4 flex items-center gap-2">
            <Zap className="w-5 h-5 text-yellow-400" />
            Speculative Decoding
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-xs text-gray-400">Expected Accepted</p>
              <p className="text-lg font-semibold text-white">
                {speculative.expectedAcceptedTokens.toFixed(1)} tokens
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-400">Speedup</p>
              <p className="text-lg font-semibold text-green-400">
                {speculative.speedup.toFixed(2)}x
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-400">Draft Overhead</p>
              <p className="text-lg font-semibold text-white">
                {formatLatency(speculative.draftModelOverhead)}
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-400">Effective TPOT</p>
              <p className="text-lg font-semibold text-cyan-400">
                {formatLatency(speculative.effectiveTpot)}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Pareto + Roofline side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">
        <ParetoFrontier
          currentCostPerMToken={costPerMToken}
          currentTtft={latency.ttft}
          currentTpot={latency.tpot}
          paretoResult={paretoResult}
          paretoProgress={paretoProgress}
          currentTP={tensorParallel}
          currentEP={simExpertParallel}
          currentCB={simContinuousBatching}
          currentBatchSize={simBatchSize}
          seqLenSweepResult={seqLenSweepResult}
          currentInputSeqLen={simInputSeqLen}
          gpuHourlyRate={effectiveRate}
          numGPUs={numGPUs}
        />
        {modelSpec && gpu && (
          <RooflineChart
            mode="inference"
            model={modelSpec}
            gpu={gpu}
            dtype={useConfigStore.getState().inference.weightPrecision}
            inferenceResult={result}
            inferenceConfig={{
              inputSeqLen: useConfigStore.getState().inference.inputSeqLen,
              outputSeqLen: useConfigStore.getState().inference.outputSeqLen,
              batchSize: Math.ceil(useConfigStore.getState().inference.batchSize / numReplicas),
              tp: tensorParallel,
              continuousBatching: simContinuousBatching,
            }}
          />
        )}
      </div>

      {/* Disclaimer */}
      <SimulationDisclaimer />
    </div>
  );
}

interface TrainingResultsProps {
  metrics: SimulationMetrics;
  result: SimulationResult;
  gpuMemoryGB: number;
  modelName: string;
  modelSpec: ModelSpec | null;
  gpuName: string;
  numGPUs: number;
  maxSteps: number;
  globalBatchSize: number;
  sequenceLength: number;
  configSnapshot: NonNullable<import('../../stores/simulation.ts').TrainingSimulationState['configSnapshot']> | null;
}

function TrainingResults({
  metrics,
  result,
  gpuMemoryGB,
  modelName,
  modelSpec,
  gpuName,
  numGPUs,
  maxSteps,
  globalBatchSize,
  sequenceLength,
  configSnapshot,
}: TrainingResultsProps) {
  const isCustomModel = useConfigStore(s => s.modelId.startsWith('custom-'));
  const clusterConfig = useConfigStore.getState().clusterConfig;
  const gpu = clusterConfig?.node.gpu ?? null;
  const node = clusterConfig?.node ?? null;

  const [specTooltip, setSpecTooltip] = useState<{ top: number; left: number } | null>(null);
  const specIconRef = useRef<HTMLSpanElement>(null);
  const isTouchRef = useRef(false);
  const showSpecTooltip = useCallback(() => {
    if (isTouchRef.current) return;
    const rect = specIconRef.current?.getBoundingClientRect();
    if (rect) setSpecTooltip({ top: rect.bottom + 6, left: rect.left + rect.width / 2 });
  }, []);
  const hideSpecTooltip = useCallback(() => setSpecTooltip(null), []);

  // Pipeline params from simulation snapshot, falling back to live config
  // Select individual values to avoid creating a new object reference per render
  // (inline object selectors cause infinite useSyncExternalStore loops in Zustand v5)
  const livePpDegree = useConfigStore(s => s.training.ppDegree);
  const liveNumMicroBatches = useConfigStore(s => s.training.numMicroBatches);
  const liveGA = useConfigStore(s => s.training.gradientAccumulationSteps);
  const livePipelineSchedule = useConfigStore(s => s.training.pipelineSchedule);
  const liveInterleavedStages = useConfigStore(s => s.training.interleavedStages);
  const pipelineSnapshot = useMemo(() => ({
    ppDegree: configSnapshot?.ppDegree ?? livePpDegree,
    numMicroBatches: configSnapshot?.numMicroBatches ?? liveNumMicroBatches,
    gradientAccumulationSteps: configSnapshot?.gradientAccumulationSteps ?? liveGA,
    pipelineSchedule: configSnapshot?.pipelineSchedule ?? livePipelineSchedule,
    interleavedStages: configSnapshot?.interleavedStages ?? liveInterleavedStages,
  }), [configSnapshot, livePpDegree, liveNumMicroBatches, liveGA, livePipelineSchedule, liveInterleavedStages]);

  const [copied, setCopied] = useState(false);
  const [shareCopied, setShareCopied] = useState(false);
  const [exportCopied, setExportCopied] = useState(false);
  const [exporting, setExporting] = useState(false);
  const shareCardRef = useRef<HTMLDivElement>(null);

  const trainingShareCardProps: ShareCardProps = useMemo(() => {
    const live = useConfigStore.getState();
    const gpuId = configSnapshot?.gpuId ?? live.gpuId;
    const customPrice = live.pricePerGPUHour;
    const rate = customPrice ?? getGPUHourlyRate(gpuId).rate;
    const gpuHours = metrics.timeToTrainHours != null ? metrics.timeToTrainHours * numGPUs : undefined;
    const targetTokens = configSnapshot?.targetTokens ?? live.training.targetTokens;
    const activeParams = configSnapshot?.activeParams ?? live.modelSpec?.activeParams ?? live.modelSpec?.totalParams ?? 0;
    const chinchillaOptimal = activeParams * 20;
    return {
      mode: 'training' as const,
      modelName,
      gpuName,
      numGPUs,
      gpusPerNode: configSnapshot?.gpusPerNode ?? live.gpusPerNode,
      strategyType: configSnapshot?.strategyType ?? live.training.strategyType,
      precision: configSnapshot?.precision ?? live.precision,
      sequenceLength: configSnapshot?.sequenceLength ?? live.sequenceLength,
      globalBatchSize: configSnapshot?.globalBatchSize ?? live.training.globalBatchSize,
      microBatchSize: configSnapshot?.microBatchSize ?? live.training.microBatchSize,
      activationCheckpointing: configSnapshot?.activationCheckpointing ?? live.training.activationCheckpointing,
      checkpointingGranularity: configSnapshot?.checkpointingGranularity ?? live.training.checkpointingGranularity,
      resolvedStoredLayers: metrics.resolvedStoredLayers,
      modelNumLayers: live.modelSpec?.numLayers,
      flashAttention: configSnapshot?.flashAttention ?? live.training.flashAttention,
      sequenceParallel: configSnapshot?.sequenceParallel ?? live.training.sequenceParallel,
      tpDegree: configSnapshot?.tpDegree ?? live.training.tpDegree,
      ppDegree: configSnapshot?.ppDegree ?? live.training.ppDegree,
      cpDegree: configSnapshot?.cpDegree ?? live.training.cpDegree,
      epDegree: configSnapshot?.epDegree ?? live.training.epDegree,
      pipelineSchedule: configSnapshot?.pipelineSchedule ?? live.training.pipelineSchedule,
      interleavedStages: configSnapshot?.interleavedStages ?? live.training.interleavedStages,
      finetuningMethod: configSnapshot?.finetuningMethod ?? live.training.finetuningMethod,
      loraRank: configSnapshot?.loraRank ?? live.training.loraRank,
      loraTargetModules: configSnapshot?.loraTargetModules ?? live.training.loraTargetModules,
      mfu: metrics.mfu,
      hfu: metrics.hfu,
      modelFlopsMfu: metrics.modelFlopsMfu,
      fp8HwUtil: metrics.fp8HwUtil,
      tokensPerSecondPerGPU: metrics.tokensPerSecond / numGPUs,
      tokensPerSecondCluster: metrics.tokensPerSecond,
      stepTimeMs: metrics.stepTimeMs,
      tflopsPerGPU: metrics.tflopsPerGPU,
      samplesPerSecond: metrics.samplesPerSecond,
      memoryPerGPU: metrics.memoryPerGPU,
      gpuMemoryGB,
      memoryUtilization: metrics.memoryUtilization,
      timing: metrics.timing,
      pipelineBubble: metrics.pipelineBubble,
      timeToTrainHours: metrics.timeToTrainHours,
      targetTokens,
      trainingSteps: maxSteps,
      chinchillaRatio: chinchillaOptimal > 0 ? targetTokens / chinchillaOptimal : undefined,
      gpuHours,
      gpuCostPerHour: rate,
      estimatedCost: gpuHours != null ? gpuHours * rate : undefined,
    };
  }, [modelName, gpuName, numGPUs, gpuMemoryGB, metrics, configSnapshot, maxSteps]);

  const handleExportImage = useCallback(async (download: boolean) => {
    if (!shareCardRef.current || exporting) return;
    setExporting(true);
    try {
      if (download) {
        const slug = `${modelName}-${gpuName}`.toLowerCase().replace(/[^a-z0-9]+/g, '-');
        await downloadImage(shareCardRef.current, `llm-sim-${slug}.png`);
      } else {
        await copyImageToClipboard(shareCardRef.current);
        setExportCopied(true);
        setTimeout(() => setExportCopied(false), 2000);
      }
    } finally {
      setExporting(false);
    }
  }, [exporting, modelName, gpuName]);

  const handleShare = () => {
    const url = buildShareURL();
    if (navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(url).then(() => {
        setShareCopied(true);
        setTimeout(() => setShareCopied(false), 2000);
      }).catch(() => { copyFallback(url); setShareCopied(true); setTimeout(() => setShareCopied(false), 2000); });
    } else {
      copyFallback(url);
      setShareCopied(true);
      setTimeout(() => setShareCopied(false), 2000);
    }
  };
  const handleCopyConfig = () => {
    // Use snapshot (frozen at simulation time) for all config fields.
    // Fall back to live config only if snapshot is null (pre-snapshot state).
    const live = useConfigStore.getState();
    const snap = configSnapshot;
    const sTP = snap?.tpDegree ?? live.training.tpDegree;
    const sPP = snap?.ppDegree ?? live.training.ppDegree;
    const sEP = snap?.epDegree ?? live.training.epDegree;
    const sCP = snap?.cpDegree ?? live.training.cpDegree;
    const sMoE = snap?.isMoE ?? live.modelSpec?.isMoE ?? false;
    const sFT = snap?.finetuningMethod ?? live.training.finetuningMethod;
    const sSP = snap?.sequenceParallel ?? live.training.sequenceParallel;
    const sSchedule = snap?.pipelineSchedule ?? live.training.pipelineSchedule;
    const sStages = snap?.interleavedStages ?? live.training.interleavedStages;
    const sGpuId = snap?.gpuId ?? live.gpuId;
    const sNumGPUs = snap?.numGPUs ?? live.numGPUs;

    const hasTP = sTP > 1;
    const hasPP = sPP > 1;
    const hasEP = sMoE && sEP > 1;
    const hasSP = hasTP && sSP;

    const exportConfig: ExportableConfig = {
      version: '1.0',
      timestamp: new Date().toISOString(),
      config: {
        // Model
        modelId: snap?.modelId ?? live.modelId,
        // Cluster
        clusterId: snap?.clusterId ?? live.clusterId,
        gpuId: sGpuId,
        numGPUs: sNumGPUs,
        gpusPerNode: snap?.gpusPerNode ?? live.gpusPerNode,
        // Batch
        globalBatchSize: snap?.globalBatchSize ?? live.training.globalBatchSize,
        microBatchSize: snap?.microBatchSize ?? live.training.microBatchSize,
        numMicroBatches: snap?.numMicroBatches ?? live.training.numMicroBatches,
        sequenceLength: snap?.sequenceLength ?? live.sequenceLength,
        // Parallelism — only include dimensions that are active
        strategyType: snap?.strategyType ?? live.training.strategyType,
        dpDegree: snap?.dpDegree ?? live.training.dpDegree,
        ...(hasTP ? { tpDegree: sTP } : {}),
        ...(hasPP ? { ppDegree: sPP } : {}),
        ...(hasEP ? { epDegree: sEP } : {}),
        ...(sCP > 1 ? { cpDegree: sCP } : {}),
        ...(hasPP && sSchedule !== '1f1b' ? {
          pipelineSchedule: sSchedule,
          interleavedStages: sStages,
        } : {}),
        // Optimizations
        precision: snap?.precision ?? live.precision,
        activationCheckpointing: snap?.activationCheckpointing ?? live.training.activationCheckpointing,
        ...((snap?.checkpointingGranularity ?? live.training.checkpointingGranularity) === 'selective'
          ? { checkpointingGranularity: 'selective' as const } : {}),
        ...(live.training.activationCheckpointing && live.training.checkpointingGranularity === 'selective' && live.training.selectiveStoredLayers !== 'auto'
          ? { selectiveStoredLayers: live.training.selectiveStoredLayers } : {}),
        ...(hasSP ? { sequenceParallel: true } : {}),
        flashAttention: snap?.flashAttention ?? live.training.flashAttention,
        // Fine-tuning (omit when full fine-tuning)
        ...(sFT !== 'full' ? {
          finetuningMethod: sFT,
          loraRank: snap?.loraRank ?? live.training.loraRank,
          loraTargetModules: snap?.loraTargetModules ?? live.training.loraTargetModules,
        } : {}),
        // Training scale
        trainingGoal: snap?.trainingGoal ?? live.training.trainingGoal,
        targetTokens: toExponent(snap?.targetTokens ?? live.training.targetTokens),
        // Pricing
        ...(live.pricePerGPUHour != null ? { pricePerGPUHour: live.pricePerGPUHour } : {}),
      },
      metrics: {
        // Efficiency (percentages matching dashboard display)
        mfuPct: +((metrics.mfu * 100).toFixed(1)),
        hfuPct: +((metrics.hfu * 100).toFixed(1)),
        ...(metrics.modelFlopsMfu != null
          ? { modelFlopsMfuPct: +((metrics.modelFlopsMfu * 100).toFixed(1)) }
          : {}),
        ...(metrics.fp8HwUtil != null
          ? { fp8HwUtilPct: +((metrics.fp8HwUtil * 100).toFixed(1)) }
          : {}),
        tflopsPerGPU: Math.round(metrics.tflopsPerGPU),
        // Throughput
        tokensPerSecond: Math.round(metrics.tokensPerSecond),
        // Timing
        stepTimeMs: Math.round(metrics.stepTimeMs),
        // Memory
        memoryPerGPU: +(metrics.memoryPerGPU.total / (1024 ** 3)).toFixed(2),
        memoryUtilizationPct: +((metrics.memoryUtilization * 100).toFixed(1)),
        // Overheads (percentages)
        communicationOverheadPct: +((metrics.communicationOverhead * 100).toFixed(1)),
        ...(hasPP ? { pipelineBubblePct: +((metrics.pipelineBubble * 100).toFixed(1)) } : {}),
        // Training projection
        ...(metrics.timeToTrainHours != null ? {
          timeToTrainHours: +metrics.timeToTrainHours.toFixed(1),
          gpuHours: Math.round(metrics.timeToTrainHours * sNumGPUs),
          estimatedCost: Math.round(metrics.timeToTrainHours * sNumGPUs * (live.pricePerGPUHour ?? getGPUHourlyRate(sGpuId).rate)),
        } : {}),
      },
    };
    const json = exportConfigToJSON(exportConfig);
    // Fallback for non-HTTPS contexts where navigator.clipboard is unavailable
    try {
      const textarea = document.createElement('textarea');
      textarea.value = json;
      textarea.style.position = 'fixed';
      textarea.style.opacity = '0';
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Last resort: navigator.clipboard (works on HTTPS)
      navigator.clipboard?.writeText(json).then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      });
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-2xl font-semibold text-white">Training Results</h2>
          <p className="text-gray-400 mt-1 flex items-center gap-1.5">
            {modelName} on {formatInteger(numGPUs)}x {gpuName}
            {(modelSpec || gpu) && (
              <span
                ref={specIconRef}
                className="inline-flex"
                onMouseEnter={showSpecTooltip}
                onMouseLeave={hideSpecTooltip}
                onTouchStart={() => { isTouchRef.current = true; }}
              >
                <Info className="w-3.5 h-3.5 text-gray-500 cursor-help" />
              </span>
            )}
            {specTooltip && createPortal(
              <div
                className="fixed bg-gray-800 border border-gray-700 rounded-lg text-xs text-gray-300 z-50 pointer-events-none flex"
                style={{ top: specTooltip.top, left: specTooltip.left, transform: 'translateX(-50%)' }}
              >
                {/* Left: model / chip / network specs */}
                <div className="px-4 py-3 w-72 shrink-0">
                  {modelSpec && (
                    <>
                      <div className="font-medium text-white mb-1.5">{modelSpec.name}</div>
                      <div className="space-y-0.5 text-gray-400">
                        <div className="flex justify-between"><span>Parameters</span><span className="text-gray-300">{formatNumber(modelSpec.totalParams)}</span></div>
                        <div className="flex justify-between"><span>Layers</span><span className="text-gray-300">{modelSpec.numLayers}</span></div>
                        <div className="flex justify-between"><span>Hidden size</span><span className="text-gray-300">{formatInteger(modelSpec.hiddenSize)}</span></div>
                        <div className="flex justify-between"><span>Attn heads</span><span className="text-gray-300">{modelSpec.numAttentionHeads}{modelSpec.attentionType === 'mla' ? ' (MLA)' : modelSpec.numKvHeads !== modelSpec.numAttentionHeads ? ` (${modelSpec.numKvHeads} KV)` : ''}</span></div>
                        <div className="flex justify-between"><span>Vocab size</span><span className="text-gray-300">{formatInteger(modelSpec.vocabSize)}</span></div>
                        <div className="flex justify-between"><span>Max seq length</span><span className="text-gray-300">{formatInteger(modelSpec.maxSeqLength)}</span></div>
                        {modelSpec.isMoE && modelSpec.numExperts && (
                          <div className="flex justify-between"><span>Experts</span><span className="text-gray-300">{modelSpec.numActiveExperts}/{modelSpec.numExperts} active</span></div>
                        )}
                      </div>
                    </>
                  )}
                  {modelSpec && gpu && (
                    <div className="border-t border-gray-700 my-2"></div>
                  )}
                  {gpu && (
                    <>
                      <div className="font-medium text-white mb-1.5">{gpu.name}</div>
                      <div className="space-y-0.5 text-gray-400">
                        <div className="flex justify-between"><span>Memory</span><span className="text-gray-300">{gpu.memoryGB} GB</span></div>
                        <div className="flex justify-between"><span>Mem bandwidth</span><span className="text-gray-300">{gpu.memoryBandwidthTBps} TB/s</span></div>
                        <div className="flex justify-between"><span>BF16 / FP16</span><span className="text-gray-300">{gpu.bf16TFLOPS || '—'} / {gpu.fp16TFLOPS} TFLOPS</span></div>
                        {gpu.fp8TFLOPS > 0 && (
                          <div className="flex justify-between"><span>FP8</span><span className="text-gray-300">{gpu.fp8TFLOPS} TFLOPS</span></div>
                        )}
                        <div className="flex justify-between"><span>TDP</span><span className="text-gray-300">{gpu.tdpWatts}W</span></div>
                        {gpu.estimated && (
                          <div className="text-gray-500 mt-1 text-[10px]">Limited real-world benchmark data</div>
                        )}
                      </div>
                    </>
                  )}
                  {node && (
                    <>
                      <div className="border-t border-gray-700 my-2"></div>
                      <div className="font-medium text-white mb-1.5">Network</div>
                      <div className="space-y-0.5 text-gray-400">
                        <div className="flex justify-between"><span>Intra-node</span><span className="text-gray-300">{node.intraNodeInterconnect.name}</span></div>
                        <div className="flex justify-between"><span>Intra-node BW</span><span className="text-gray-300">{node.intraNodeInterconnect.bandwidthGBps} GB/s</span></div>
                        {clusterConfig && clusterConfig.numNodes > 1 && (
                          <>
                            <div className="flex justify-between"><span>Inter-node</span><span className="text-gray-300">{node.interNodeInterconnect.name}</span></div>
                            <div className="flex justify-between"><span>Inter-node BW</span><span className="text-gray-300">{node.interNodeInterconnect.bandwidthGBps} GB/s</span></div>
                          </>
                        )}
                      </div>
                    </>
                  )}
                </div>
                {/* Right: GPU partitioning grid */}
                {numGPUs > 1 && (
                  <div className="border-l border-gray-700 px-3 py-3 max-w-80 max-h-[70vh] overflow-auto">
                    <GPUGridPanel embedded />
                  </div>
                )}
              </div>,
              document.body,
            )}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {isCustomModel ? (
          <Tooltip text="Sharing is not available for custom models">
          <button
            disabled
            className="flex items-center gap-2 px-3 py-2 text-sm rounded-lg bg-gray-700 text-gray-500 cursor-not-allowed"
          >
            <Link2 className="w-4 h-4" />
            Share
          </button>
          </Tooltip>
          ) : (
          <button
            onClick={handleShare}
            className={`flex items-center gap-2 px-3 py-2 text-sm rounded-lg transition-colors ${
              shareCopied
                ? 'bg-green-500/20 text-green-400 cursor-pointer'
                : 'bg-accent hover:bg-accent text-white cursor-pointer'
            }`}
            aria-label="Copy shareable link to clipboard"
          >
            {shareCopied ? <Check className="w-4 h-4" /> : <Link2 className="w-4 h-4" />}
            {shareCopied ? 'Copied!' : 'Share'}
          </button>
          )}
          <ExportButton
            onCopy={() => handleExportImage(false)}
            onDownload={() => handleExportImage(true)}
            onCopyConfig={handleCopyConfig}
            copied={exportCopied}
            configCopied={copied}
            disabled={exporting}
          />
        </div>
      </div>

      {/* Off-screen share card for PNG export */}
      {createPortal(
        <div style={{ position: 'fixed', left: -9999, top: 0, pointerEvents: 'none' }}>
          <ShareCard ref={shareCardRef} {...trainingShareCardProps} />
        </div>,
        document.body,
      )}

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label={metrics.fp8HwUtil != null ? 'FP8 HW Util' : 'MFU'}
          value={metrics.fp8HwUtil != null
            ? `${parseFloat((metrics.fp8HwUtil * 100).toFixed(1))}%`
            : `${parseFloat((metrics.mfu * 100).toFixed(1))}%`}
          subValue={metrics.fp8HwUtil != null
            ? `${parseFloat((metrics.mfu * 100).toFixed(1))}% MFU (vs BF16 peak)`
            : metrics.hfu !== metrics.mfu
              ? `${parseFloat((metrics.hfu * 100).toFixed(1))}% HFU (incl. recompute)`
              : 'Model FLOPS Utilization'}
          icon={<Zap className="w-5 h-5 text-yellow-400" />}
          color="text-yellow-400"
        >
          {metrics.fp8HwUtil != null && metrics.hfu !== metrics.mfu && (
            <p className="text-xs text-gray-500 mt-1">
              {parseFloat((metrics.hfu * 100).toFixed(1))}% HFU (incl. recompute)
            </p>
          )}
          {metrics.modelFlopsMfu != null && (
            <p className="text-xs text-gray-500 mt-1 flex items-center gap-1">
              {parseFloat((metrics.modelFlopsMfu * 100).toFixed(1))}% Model FLOPs MFU
              <Tooltip text="Uses actual model FLOPs (incl. quadratic attention) rather than the 6PD approximation. Differs at long sequence lengths.">
                <Info className="w-3 h-3 text-gray-500 cursor-help" />
              </Tooltip>
            </p>
          )}
        </MetricCard>
        <MetricCard
          label="Throughput/GPU"
          value={`${formatNumber(Math.round(metrics.tokensPerSecond / numGPUs))} tok/s`}
          subValue={`Cluster: ${formatNumber(Math.round(metrics.tokensPerSecond))} tok/s`}
          icon={<TrendingUp className="w-5 h-5 text-purple-400" />}
          color="text-purple-400"
        />
        <MetricCard
          label="Step Time"
          value={formatTime(metrics.stepTimeMs)}
          subValue={`${metrics.samplesPerSecond >= 100 ? Math.round(metrics.samplesPerSecond) : parseFloat(metrics.samplesPerSecond.toFixed(1))} samples/sec`}
          icon={<Clock className="w-5 h-5 text-blue-400" />}
          color="text-blue-400"
        />
        <MetricCard
          label="TFLOPS/GPU"
          value={metrics.tflopsPerGPU.toFixed(1)}
          subValue="achieved (useful work)"
          icon={<Cpu className="w-5 h-5 text-green-400" />}
          color="text-green-400"
        />
      </div>

      {/* Training Projection */}
      {metrics.timeToTrainHours && (
        <TrainingProjection
          metrics={metrics}
          maxSteps={maxSteps}
          globalBatchSize={globalBatchSize}
          sequenceLength={sequenceLength}
          numGPUs={numGPUs}
          modelParams={configSnapshot?.modelParams ?? modelSpec?.totalParams ?? 0}
          activeParams={configSnapshot?.activeParams ?? modelSpec?.activeParams ?? modelSpec?.totalParams ?? 0}
          trainingGoal={(configSnapshot?.trainingGoal ?? useConfigStore.getState().training.trainingGoal) as TrainingGoal}
          targetTokens={configSnapshot?.targetTokens ?? useConfigStore.getState().training.targetTokens}
          gpuId={configSnapshot?.gpuId ?? useConfigStore.getState().gpuId}
        />
      )}

      {/* Memory and Timing */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Memory Breakdown */}
        <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-5">
          <h3 className="text-lg font-medium text-white mb-4 flex items-center gap-2">
            <HardDrive className="w-5 h-5 text-blue-400" />
            Memory per GPU
          </h3>
          <div className="space-y-3">
            <ProgressBar
              label="Parameters"
              value={metrics.memoryPerGPU.parameters}
              max={gpuMemoryGB * (1024 ** 3)}
              color="bg-blue-500"
              unit="bytes"
            />
            <ProgressBar
              label="Gradients"
              value={metrics.memoryPerGPU.gradients}
              max={gpuMemoryGB * (1024 ** 3)}
              color="bg-red-500"
              unit="bytes"
            />
            <ProgressBar
              label="Optimizer States"
              value={metrics.memoryPerGPU.optimizerStates}
              max={gpuMemoryGB * (1024 ** 3)}
              color="bg-orange-500"
              unit="bytes"
            />
            {/* Activations: two-tone bar (base activations solid + gather buffers faded) */}
            {(() => {
              const base = metrics.memoryPerGPU.activations;
              const peak = metrics.memoryPerGPU.peakActivations;
              const gatherBuffers = peak - base;
              const hasGatherBuffers = gatherBuffers > base * 0.01; // >1% of activations
              const basePct = Math.min((base / (gpuMemoryGB * (1024 ** 3))) * 100, 100);
              const gatherPct = Math.min((gatherBuffers / (gpuMemoryGB * (1024 ** 3))) * 100, 100 - basePct);
              const totalPct = Math.min((peak / (gpuMemoryGB * (1024 ** 3))) * 100, 100);
              return (
                <div>
                  <div className="flex justify-between text-xs text-gray-400 mb-1">
                    <span>Activations</span>
                    <span>{formatBytes(peak, peak >= 1e9 ? 1 : 0)} ({totalPct.toFixed(1)}%)</span>
                  </div>
                  <div className="h-2 bg-gray-800 rounded-full overflow-hidden flex">
                    <AnimatedBar width={`${basePct}%`} className="h-full bg-green-500" />
                    {hasGatherBuffers && (
                      <AnimatedBar width={`${gatherPct}%`} delay={0.1} className="h-full bg-green-500/25" />
                    )}
                  </div>
                </div>
              );
            })()}
            <div className="pt-2 mt-2 border-t border-gray-700">
              <div className="flex justify-between text-sm">
                <span className="text-gray-400 flex items-center gap-1">
                  Total
                  <div className="relative group/meminfo inline-flex">
                    <Info className="w-3 h-3 text-gray-500 cursor-help" />
                    <div className="absolute bottom-1/2 left-full translate-y-1/2 ml-2 px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-xs text-gray-300 opacity-0 group-hover/meminfo:opacity-100 transition-opacity pointer-events-none z-10 w-72">
                      <div className="font-medium text-white mb-1.5">Includes hidden overhead (~{formatBytes(metrics.memoryPerGPU.temporary + metrics.memoryPerGPU.reserved, 0)})</div>
                      <div className="space-y-1 text-gray-400 leading-relaxed">
                        <p><span className="text-gray-300">NCCL buffers</span> (~{formatBytes(metrics.memoryPerGPU.temporary, 0)}) — gradient buckets, AllGather/ReduceScatter scratch for comm overlap</p>
                        <p><span className="text-gray-300">CUDA + fragmentation</span> (~{formatBytes(metrics.memoryPerGPU.reserved, 0)}) — driver context, cuDNN/cuBLAS handles, allocator block rounding</p>
                        {metrics.memoryPerGPU.peakActivations - metrics.memoryPerGPU.activations > metrics.memoryPerGPU.activations * 0.01 && (
                          <p><span className="text-gray-300">FSDP gather buffers</span> (~{formatBytes(metrics.memoryPerGPU.peakActivations - metrics.memoryPerGPU.activations, 0)}) — all-gathered layer params/grads from DP peers, shown as <span className="inline-block w-2 h-2 rounded-sm bg-green-500/25 align-middle"></span> in activations bar</p>
                        )}
                      </div>
                      <div className="absolute top-1/2 -translate-y-1/2 right-full border-4 border-transparent border-r-gray-800"></div>
                    </div>
                  </div>
                </span>
                <span className="text-white font-medium">
                  {formatBytes(metrics.memoryPerGPU.total)} / {gpuMemoryGB} GB
                </span>
              </div>
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>Utilization</span>
                <span className={metrics.memoryUtilization >= 0.9 ? 'text-red-400 font-medium' : metrics.memoryUtilization >= 0.8 ? 'text-amber-400 font-medium' : ''}>{(metrics.memoryUtilization * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        </div>

        {/* Timing Breakdown */}
        <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-5 relative">
          <h3 className="text-lg font-medium text-white mb-4 flex items-center gap-2">
            <Clock className="w-5 h-5 text-purple-400" />
            Timing Breakdown
          </h3>
          <div className="space-y-3">
            {/* Forward Pass with tooltip */}
            {(() => {
              const t = metrics.timing;
              const pct = Math.min((t.forward / metrics.stepTimeMs) * 100, 100);
              const hasMoeFwd = t.moeExpertFwdMs != null;
              // Mini stacked bar segments (dark → light)
              const fwdSegments: { label: string; ms: number; color: string }[] = [];
              if (hasMoeFwd) {
                if (t.moeExpertFwdMs! > 0) fwdSegments.push({ label: 'Expert Compute', ms: t.moeExpertFwdMs!, color: 'bg-green-700' });
                if (t.moeDenseFwdMs! > 0) fwdSegments.push({ label: 'Dense/Attention', ms: t.moeDenseFwdMs!, color: 'bg-green-500' });
                if (t.epCommFwdMs! > 0) fwdSegments.push({ label: 'EP All-to-All (fwd)', ms: t.epCommFwdMs!, color: 'bg-green-400' });
                if (t.moeOverheadFwdMs! > 0) fwdSegments.push({ label: 'Overhead', ms: t.moeOverheadFwdMs!, color: 'bg-green-300' });
              } else {
                if (t.forwardComputeMs != null) fwdSegments.push({ label: 'Matmul Compute', ms: t.forwardComputeMs, color: 'bg-green-600' });
                if (t.forwardNonMatmulMs != null && t.forwardNonMatmulMs > 0) fwdSegments.push({ label: 'Non-Matmul Ops', ms: t.forwardNonMatmulMs, color: 'bg-green-400' });
              }
              const segTotal = fwdSegments.reduce((s, x) => s + x.ms, 0);
              return (
                <div>
                  <div className="flex justify-between text-xs text-gray-400 mb-1">
                    <span className="flex items-center gap-1">
                      Forward Pass
                      {fwdSegments.length > 0 && (
                        <div className="relative group/fwdinfo inline-flex">
                          <Info className="w-3 h-3 text-gray-500 cursor-help" />
                          <div className="absolute bottom-1/2 left-full translate-y-1/2 ml-2 px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-xs text-gray-300 opacity-0 group-hover/fwdinfo:opacity-100 transition-opacity pointer-events-none z-10 w-72">
                            <div className="font-medium text-white mb-2">Forward Pass Breakdown</div>
                            {segTotal > 0 && (
                              <div className="h-2 rounded-full overflow-hidden flex mb-2">
                                {fwdSegments.map((seg, i) => (
                                  <div key={i} className={`h-full ${seg.color}`} style={{ width: `${(seg.ms / segTotal) * 100}%` }} />
                                ))}
                              </div>
                            )}
                            <div className="space-y-1 text-gray-400">
                              {fwdSegments.map((seg, i) => (
                                <div key={i} className="flex justify-between">
                                  <span className="flex items-center gap-1.5">
                                    <span className={`inline-block w-2 h-2 rounded-sm ${seg.color}`}></span>
                                    {seg.label}
                                  </span>
                                  <span className="text-gray-300">{formatTime(seg.ms, 0)} ({segTotal > 0 ? ((seg.ms / segTotal) * 100).toFixed(0) : 0}%)</span>
                                </div>
                              ))}
                            </div>
                            {hasMoeFwd && t.groupedGemmFactor != null && (
                              <div className="mt-1.5 pt-1.5 border-t border-gray-700 text-gray-500">
                                Grouped GEMM efficiency: {(t.groupedGemmFactor * 100).toFixed(0)}%
                              </div>
                            )}
                            <div className="absolute top-1/2 -translate-y-1/2 right-full border-4 border-transparent border-r-gray-800"></div>
                          </div>
                        </div>
                      )}
                    </span>
                    <span>{formatTime(t.forward, 0)} ({pct.toFixed(1)}%)</span>
                  </div>
                  <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                    <AnimatedBar width={`${pct}%`} className="h-full bg-green-500 rounded-full" />
                  </div>
                </div>
              );
            })()}
            {/* Backward Pass with tooltip */}
            {(() => {
              const t = metrics.timing;
              const pct = Math.min((t.backward / metrics.stepTimeMs) * 100, 100);
              const hasMoeBwd = t.moeExpertBwdMs != null;
              // Dark → light
              const bwdSegments: { label: string; ms: number; color: string }[] = [];
              if (hasMoeBwd) {
                if (t.moeExpertBwdMs! > 0) bwdSegments.push({ label: 'Expert Compute', ms: t.moeExpertBwdMs!, color: 'bg-blue-700' });
                if (t.moeDenseBwdMs! > 0) bwdSegments.push({ label: 'Dense/Attention', ms: t.moeDenseBwdMs!, color: 'bg-blue-500' });
                if (t.epCommBwdMs! > 0) bwdSegments.push({ label: 'EP All-to-All (bwd)', ms: t.epCommBwdMs!, color: 'bg-blue-400' });
                if (t.moeOverheadBwdMs! > 0) bwdSegments.push({ label: 'Overhead', ms: t.moeOverheadBwdMs!, color: 'bg-blue-300' });
              } else {
                if (t.backwardComputeMs != null) bwdSegments.push({ label: 'Backward Compute', ms: t.backwardComputeMs, color: 'bg-blue-600' });
                if (t.backwardNonMatmulMs != null && t.backwardNonMatmulMs > 0) bwdSegments.push({ label: 'Non-Matmul Ops', ms: t.backwardNonMatmulMs, color: 'bg-blue-400' });
              }
              const segTotal = bwdSegments.reduce((s, x) => s + x.ms, 0);
              return (
                <div>
                  <div className="flex justify-between text-xs text-gray-400 mb-1">
                    <span className="flex items-center gap-1">
                      Backward Pass
                      {bwdSegments.length > 0 && (
                        <div className="relative group/bwdinfo inline-flex">
                          <Info className="w-3 h-3 text-gray-500 cursor-help" />
                          <div className="absolute bottom-1/2 left-full translate-y-1/2 ml-2 px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-xs text-gray-300 opacity-0 group-hover/bwdinfo:opacity-100 transition-opacity pointer-events-none z-10 w-72">
                            <div className="font-medium text-white mb-2">Backward Pass Breakdown</div>
                            {segTotal > 0 && (
                              <div className="h-2 rounded-full overflow-hidden flex mb-2">
                                {bwdSegments.map((seg, i) => (
                                  <div key={i} className={`h-full ${seg.color}`} style={{ width: `${(seg.ms / segTotal) * 100}%` }} />
                                ))}
                              </div>
                            )}
                            <div className="space-y-1 text-gray-400">
                              {bwdSegments.map((seg, i) => (
                                <div key={i} className="flex justify-between">
                                  <span className="flex items-center gap-1.5">
                                    <span className={`inline-block w-2 h-2 rounded-sm ${seg.color}`}></span>
                                    {seg.label}
                                  </span>
                                  <span className="text-gray-300">{formatTime(seg.ms, 0)} ({segTotal > 0 ? ((seg.ms / segTotal) * 100).toFixed(0) : 0}%)</span>
                                </div>
                              ))}
                            </div>
                            <div className="absolute top-1/2 -translate-y-1/2 right-full border-4 border-transparent border-r-gray-800"></div>
                          </div>
                        </div>
                      )}
                    </span>
                    <span>{formatTime(t.backward, 0)} ({pct.toFixed(1)}%)</span>
                  </div>
                  <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                    <AnimatedBar width={`${pct}%`} className="h-full bg-blue-500 rounded-full" />
                  </div>
                </div>
              );
            })()}
            {/* Communication: two-tone bar (exposed solid + overlapped faded), includes scale overhead */}
            {(() => {
              const t = metrics.timing;
              const comm = t.communication + t.scaleOverhead;
              const overlap = t.overlap;
              const exposed = comm - overlap;
              const exposedPct = Math.min((exposed / metrics.stepTimeMs) * 100, 100);
              const overlapPct = Math.min((overlap / metrics.stepTimeMs) * 100, 100 - exposedPct);
              const totalPct = Math.min((comm / metrics.stepTimeMs) * 100, 100);
              // Per-dimension comm breakdown (dark → light)
              const commDims: { label: string; ms: number; color: string }[] = [];
              if ((t.dpGross ?? 0) > 0) commDims.push({ label: 'DP', ms: t.dpGross!, color: 'bg-purple-700' });
              if ((t.tpExposed ?? 0) > 0) commDims.push({ label: 'TP', ms: t.tpExposed!, color: 'bg-purple-500' });
              if ((t.ppExposed ?? 0) > 0) commDims.push({ label: 'PP', ms: t.ppExposed!, color: 'bg-purple-400' });
              if ((t.epCommunication ?? 0) > 0) commDims.push({ label: 'EP All-to-All (step)', ms: t.epCommunication!, color: 'bg-purple-300' });
              if ((t.cpExposed ?? 0) > 0) commDims.push({ label: 'CP', ms: t.cpExposed!, color: 'bg-purple-200' });
              const dimTotal = commDims.reduce((s, x) => s + x.ms, 0);
              return (
                <div>
                  <div className="flex justify-between text-xs text-gray-400 mb-1">
                    <span className="flex items-center gap-1">
                      Communication
                      <div className="relative group/commbar inline-flex">
                        <Info className="w-3 h-3 text-gray-500 cursor-help" />
                        <div className="absolute bottom-1/2 left-full translate-y-1/2 ml-2 px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-xs text-gray-300 opacity-0 group-hover/commbar:opacity-100 transition-opacity pointer-events-none z-10 w-72">
                          <div className="font-medium text-white mb-2">Communication Breakdown</div>
                          {commDims.length > 1 && dimTotal > 0 && (
                            <>
                              <div className="h-2 rounded-full overflow-hidden flex mb-2">
                                {commDims.map((dim, i) => (
                                  <div key={i} className={`h-full ${dim.color}`} style={{ width: `${(dim.ms / dimTotal) * 100}%` }} />
                                ))}
                              </div>
                              <div className="space-y-1 text-gray-400 mb-2">
                                {commDims.map((dim, i) => (
                                  <div key={i} className="flex justify-between">
                                    <span className="flex items-center gap-1.5">
                                      <span className={`inline-block w-2 h-2 rounded-sm ${dim.color}`}></span>
                                      {dim.label}
                                    </span>
                                    <span className="text-gray-300">{formatTime(dim.ms, 0)}</span>
                                  </div>
                                ))}
                              </div>
                              <div className="border-t border-gray-700 pt-1.5 mt-1.5"></div>
                            </>
                          )}
                          <div className="space-y-1 text-gray-400">
                            <div className="flex justify-between"><span>Total comm time</span><span className="text-gray-300">{formatTime(comm, 0)}</span></div>
                            <div className="flex justify-between"><span className="flex items-center gap-1.5"><span className="inline-block w-2 h-2 rounded-sm bg-purple-500/25"></span>Overlapped</span><span className="text-gray-300">{formatTime(overlap, 0)}</span></div>
                            <div className="flex justify-between"><span className="flex items-center gap-1.5"><span className="inline-block w-2 h-2 rounded-sm bg-purple-500"></span>Exposed</span><span className="text-gray-300">{formatTime(exposed, 0)}</span></div>
                          </div>
                          <div className="mt-1.5 pt-1.5 border-t border-gray-700 text-gray-500">
                            {overlap > t.communication * 0.5
                              ? 'Good overlap — most comm hidden behind compute'
                              : 'Low overlap — comm is stalling compute'}
                          </div>
                          {t.scaleOverhead > 0 && (
                            <div className="flex justify-between mt-1 text-gray-500">
                              <span>Cluster scale overhead</span>
                              <span className="text-gray-400">{formatTime(t.scaleOverhead, 0)}</span>
                            </div>
                          )}
                          <div className="absolute top-1/2 -translate-y-1/2 right-full border-4 border-transparent border-r-gray-800"></div>
                        </div>
                      </div>
                    </span>
                    <span>{formatTime(comm, 0)} ({totalPct.toFixed(1)}%)</span>
                  </div>
                  <div className="h-2 bg-gray-800 rounded-full overflow-hidden flex">
                    <AnimatedBar width={`${exposedPct}%`} className="h-full bg-purple-500" />
                    <AnimatedBar width={`${overlapPct}%`} delay={0.1} className="h-full bg-purple-500/25" />
                  </div>
                </div>
              );
            })()}
            <ProgressBar
              label="Optimizer"
              value={metrics.timing.optimizer}
              max={metrics.stepTimeMs}
              color="bg-orange-500"
              unit="time"
            />
            <div className="pt-2 mt-2 border-t border-gray-700">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-400">Exposed Comm.</span>
                  <p className="text-white font-medium">
                    {(((metrics.timing.communication + metrics.timing.scaleOverhead - metrics.timing.overlap) / metrics.stepTimeMs) * 100).toFixed(1)}%
                  </p>
                </div>
                {metrics.pipelineBubble > 0 && (
                <div>
                  <span className="text-gray-400">Pipeline Bubble</span>
                  <p className="text-white font-medium">
                    {(metrics.pipelineBubble * 100).toFixed(1)}%
                  </p>
                </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Analysis & Recommendations */}
      {result.analysis && (
        <AnalysisPanel
          bottleneck={result.analysis.bottleneck}
          bottleneckColor={
            result.analysis.bottleneck === 'compute' ? 'bg-green-500/20 text-green-400' :
            result.analysis.bottleneck === 'memory' ? 'bg-orange-500/20 text-orange-400' :
            'bg-purple-500/20 text-purple-400'
          }
          recommendations={result.analysis.recommendations}
        />
      )}

      {/* Deep Dive (collapsible): Pipeline Schedule + Roofline */}
      {modelSpec && gpu && (
        <CollapsibleDeepDive
          modelSpec={modelSpec}
          gpu={gpu}
          precision={(configSnapshot?.precision ?? 'bf16') as import('../../types/base.ts').DType}
          metrics={metrics}
          configSnapshot={configSnapshot}
          numGPUs={numGPUs}
          sequenceLength={sequenceLength}
          globalBatchSize={globalBatchSize}
          pipelineSnapshot={pipelineSnapshot}
        />
      )}

      {/* Modeling Assumptions */}
      <AssumptionsPanel resolvedStoredLayers={metrics.resolvedStoredLayers} numLayers={modelSpec?.numLayers} />

      {/* Disclaimer */}
      <SimulationDisclaimer />
    </div>
  );
}

export function ResultsDashboard() {
  const simulation = useSimulationStore();
  const config = useConfigStore();

  const { status, inference } = simulation;
  // Legacy fallback for training mode
  const metrics = simulation.metrics;
  const result = simulation.result;

  if (status === 'idle') {
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-3 text-gray-400 text-sm">
          <Activity className="w-5 h-5" />
          <span>Configure settings in the sidebar and click Run or press <kbd className="px-1.5 py-0.5 bg-gray-800 border border-gray-700 rounded text-xs font-mono text-gray-200">Ctrl+Enter</kbd> to see results</span>
        </div>

        {/* Show configuration info panel when idle */}
        <InfoPanel />
      </div>
    );
  }

  const gpuMemoryGB = config.clusterConfig?.node.gpu.memoryGB ?? 80;
  const modelName = config.modelSpec?.name ?? 'Unknown Model';
  const gpuName = config.clusterConfig?.node.gpu.name ?? 'Unknown GPU';
  const numGPUs = config.numGPUs;

  // Handle inference mode
  if (config.mode === 'inference') {
    if (status === 'error' || !inference.result) {
      const errorMsg = inference.result?.errors.join('; ') ?? simulation.error ?? 'An error occurred during simulation.';
      const recommendations = inference.result?.recommendations ?? [];

      return (
        <div className="flex flex-col items-center justify-center h-full text-center">
          <div className="p-4 rounded-full bg-red-500/10 mb-4">
            <Activity className="w-8 h-8 text-red-400" />
          </div>
          <h2 className="text-xl font-semibold text-red-400 mb-2">
            Inference Simulation Error
          </h2>
          <p className="text-gray-500 max-w-md">
            {errorMsg}
          </p>
          {recommendations.length > 0 && (
            <div className="mt-5 text-left max-w-lg">
              <p className="text-base font-medium text-gray-300 flex items-center gap-1.5 mb-2">
                <Lightbulb className="w-4 h-4 text-yellow-400 shrink-0" />
                Suggestions
              </p>
              <ul className="text-base text-gray-400 space-y-1.5 ml-6 list-disc">
                {recommendations.map((rec, i) => (
                  <li key={i}>{rec}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      );
    }

    return (
      <InferenceResults
        result={inference.result}
        gpuMemoryGB={gpuMemoryGB}
        modelName={modelName}
        modelSpec={config.modelSpec ?? null}
        gpuName={gpuName}
        numGPUs={numGPUs}
        tensorParallel={inference.tensorParallel}
        expertParallel={inference.expertParallel}
        continuousBatching={inference.continuousBatching}
        batchSize={inference.batchSize}
      />
    );
  }


  // Training mode (default)
  if (status === 'error' || !metrics || !result) {
    const suggestions = result?.analysis?.recommendations ?? [];
    return (
      <div className="flex flex-col items-center justify-center h-full text-center">
        <div className="p-4 rounded-full bg-red-500/10 mb-4">
          <Activity className="w-8 h-8 text-red-400" />
        </div>
        <h2 className="text-xl font-semibold text-red-400 mb-2">
          Simulation Error
        </h2>
        <p className="text-gray-500 max-w-md">
          {result?.state.error ?? simulation.error ?? 'An error occurred during simulation.'}
        </p>
        {suggestions.length > 0 && (
          <div className="mt-5 text-left w-full max-w-md">
            <p className="text-base font-medium text-gray-300 flex items-center gap-1.5 mb-2">
              <Lightbulb className="w-4 h-4 text-yellow-400 shrink-0" />
              Suggestions
            </p>
            <ul className="text-base text-gray-400 space-y-1.5 ml-6 list-disc">
              {suggestions.map((rec, i) => (
                <li key={i}>{rec}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  }

  // Use config snapshot from simulation run (not live config)
  const snapshot = simulation.training.configSnapshot;

  return (
    <TrainingResults
      metrics={metrics}
      result={result}
      gpuMemoryGB={gpuMemoryGB}
      modelName={modelName}
      modelSpec={config.modelSpec ?? null}
      gpuName={snapshot?.gpuName ?? gpuName}
      numGPUs={snapshot?.numGPUs ?? numGPUs}
      maxSteps={snapshot?.maxSteps ?? Math.ceil(config.training.targetTokens / (config.training.globalBatchSize * config.sequenceLength))}
      globalBatchSize={snapshot?.globalBatchSize ?? config.training.globalBatchSize}
      sequenceLength={snapshot?.sequenceLength ?? config.sequenceLength}
      configSnapshot={snapshot}
    />
  );
}
