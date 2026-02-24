/**
 * Info Panel - Displays model specifications and networking configuration
 */

import { useState, useMemo } from 'react';
import { Info, Cpu, Network, Layers, Database, Zap, AlertTriangle, Copy, Check } from 'lucide-react';
import { Tooltip } from '../ui/Tooltip.tsx';
import { useConfigStore } from '../../stores/config.ts';
import { formatBytes, formatInteger } from '../../types/base.ts';
import { DTYPE_PRESETS, DTYPE_BYTES } from '../../types/index.ts';
import { getSimulationMetrics, type SimulationMetrics } from '../../core/simulation/engine.ts';
import { modelRegistry } from '../../core/models/registry.ts';
import { computeLoraTrainableParams, NF4_BYTES_PER_PARAM } from '../../core/strategies/lora.ts';
import { GPUGridPanel, InferenceGPUGridPanel } from './GPUGrid.tsx';

interface InfoCardProps {
  title: string;
  icon: React.ReactNode;
  action?: React.ReactNode;
  children: React.ReactNode;
}

function InfoCard({ title, icon, action, children }: InfoCardProps) {
  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3 flex items-center gap-2">
        {icon}
        {title}
        {action && <span className="ml-auto">{action}</span>}
      </h3>
      <div className="space-y-2 text-xs">
        {children}
      </div>
    </div>
  );
}

interface InfoRowProps {
  label: string;
  value: string | number;
  unit?: string;
}

function InfoRow({ label, value, unit }: InfoRowProps) {
  return (
    <div className="flex justify-between">
      <span className="text-gray-500">{label}</span>
      <span className="text-gray-300 font-mono">
        {value}{unit && <span className="text-gray-500 ml-1">{unit}</span>}
      </span>
    </div>
  );
}

export function ModelSpecPanel() {
  const { modelSpec, modelId, customModels } = useConfigStore();
  const [copied, setCopied] = useState(false);

  if (!modelSpec) {
    return null;
  }

  const attentionLabel = {
    mha: 'Multi-Head (MHA)',
    mqa: 'Multi-Query (MQA)',
    gqa: 'Grouped-Query (GQA)',
    mla: 'Multi-head Latent (MLA)',
  }[modelSpec.attentionType];

  const kvRatio = modelSpec.numAttentionHeads / modelSpec.numKvHeads;

  const handleCopyJSON = () => {
    const cfg = modelId.startsWith('custom-')
      ? customModels[modelId]
      : modelRegistry.getConfig(modelId);
    if (!cfg) return;
    const text = JSON.stringify(cfg, null, 2);
    try {
      const textarea = document.createElement('textarea');
      textarea.value = text;
      textarea.style.position = 'fixed';
      textarea.style.opacity = '0';
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      navigator.clipboard?.writeText(text).then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      });
    }
  };

  const copyAction = (
    <Tooltip text="Copy model config as JSON">
      <button
        onClick={handleCopyJSON}
        className={`flex items-center gap-1 text-xs font-normal ${copied ? 'text-green-400' : 'text-gray-500 hover:text-gray-300'} cursor-pointer transition-colors`}
      >
        {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
        {copied ? 'Copied' : 'Copy'}
      </button>
    </Tooltip>
  );

  return (
    <InfoCard title="Model Architecture" icon={<Layers className="w-4 h-4 text-blue-400" />} action={copyAction}>
      <InfoRow label="Name" value={modelSpec.name} />
      <InfoRow label="Parameters" value={`${(modelSpec.totalParams / 1e9).toFixed(2)}B`} />
      <InfoRow label="Active Params" value={`${(modelSpec.activeParams / 1e9).toFixed(2)}B`} />
      <div className="border-t border-gray-800 my-2" />
      <InfoRow label="Layers" value={modelSpec.numLayers} />
      <InfoRow label="Hidden Size" value={formatInteger(modelSpec.hiddenSize)} />
      <InfoRow label="Intermediate" value={formatInteger(modelSpec.intermediateSize)} />
      <InfoRow label="Vocab Size" value={formatInteger(modelSpec.vocabSize)} />
      <div className="border-t border-gray-800 my-2" />
      <InfoRow label="Attention" value={attentionLabel} />
      <InfoRow label="Heads" value={modelSpec.numAttentionHeads} />
      <InfoRow label="KV Heads" value={modelSpec.numKvHeads} />
      {kvRatio > 1 && <InfoRow label="KV Ratio" value={`${kvRatio}:1`} />}
      <InfoRow label="Head Dim" value={modelSpec.headDim} />
      <div className="border-t border-gray-800 my-2" />
      <InfoRow label="Max Seq Length" value={formatInteger(modelSpec.maxSeqLength)} />
      <InfoRow label="FLOPs/Token (fwd)" value={`${(modelSpec.flopsPerToken / 1e9).toFixed(2)}G`} />
      {modelSpec.isMoE && (
        <>
          <div className="border-t border-gray-800 my-2" />
          <InfoRow label="MoE Experts" value={modelSpec.numExperts || 0} />
          <InfoRow label="Active Experts" value={modelSpec.numActiveExperts || 0} />
        </>
      )}
    </InfoCard>
  );
}

export function NetworkingPanel() {
  const { clusterConfig, numGPUs, gpusPerNode } = useConfigStore();

  if (!clusterConfig) {
    return null;
  }

  const { node, topology, interNodeBandwidthGBps, interNodeLatencyUs } = clusterConfig;
  const numNodes = Math.ceil(numGPUs / gpusPerNode);
  const isMultiNode = numNodes > 1;

  return (
    <InfoCard title="Network Configuration" icon={<Network className="w-4 h-4 text-green-400" />}>
      <InfoRow label="Topology" value={topology} />
      <InfoRow label="Nodes" value={numNodes} />
      <InfoRow label="GPUs/Node" value={gpusPerNode} />

      <div className="border-t border-gray-800 my-2" />
      <div className="text-gray-400 font-medium mb-1">Intra-Node</div>
      <InfoRow
        label={node.intraNodeInterconnect.name}
        value={node.intraNodeInterconnect.bidirectionalGBps}
        unit="GB/s"
      />
      <InfoRow
        label="Latency"
        value={node.intraNodeInterconnect.latencyUs}
        unit="µs"
      />
      {node.hasNvSwitch && (
        <div className="flex items-center gap-1 text-green-400 mt-1">
          <Zap className="w-3 h-3" />
          <span>NVSwitch enabled</span>
        </div>
      )}

      {isMultiNode && (
        <>
          <div className="border-t border-gray-800 my-2" />
          <div className="text-gray-400 font-medium mb-1">Inter-Node</div>
          <InfoRow
            label={node.interNodeInterconnect.name}
            value={interNodeBandwidthGBps}
            unit="GB/s"
          />
          <InfoRow
            label="Latency"
            value={interNodeLatencyUs}
            unit="µs"
          />
        </>
      )}
    </InfoCard>
  );
}

export function HardwarePanel() {
  const { clusterConfig, numGPUs } = useConfigStore();

  if (!clusterConfig) {
    return null;
  }

  const { node, totalMemoryGB, totalTFLOPS } = clusterConfig;
  const gpu = node.gpu;

  return (
    <InfoCard title="Hardware" icon={<Cpu className="w-4 h-4 text-purple-400" />}>
      <InfoRow label="GPU" value={gpu.name} />
      <InfoRow label="Architecture" value={gpu.architecture} />
      <InfoRow label="Count" value={numGPUs} />

      <div className="border-t border-gray-800 my-2" />
      <div className="text-gray-400 font-medium mb-1">Per GPU</div>
      <InfoRow label="Memory" value={gpu.memoryGB} unit="GB" />
      <InfoRow label="Bandwidth" value={`${(gpu.memoryBandwidthTBps * 1000).toFixed(0)}`} unit="GB/s" />
      <InfoRow label="BF16 TFLOPS" value={gpu.bf16TFLOPS} />
      <InfoRow label="FP8 TFLOPS" value={gpu.fp8TFLOPS} />
      <InfoRow label="TDP" value={gpu.tdpWatts} unit="W" />

      <div className="border-t border-gray-800 my-2" />
      <div className="text-gray-400 font-medium mb-1">Cluster Total</div>
      <InfoRow label="Memory" value={formatInteger(totalMemoryGB)} unit="GB" />
      <InfoRow label="BF16 TFLOPS" value={formatInteger(totalTFLOPS)} />
      {gpu.estimated && (
        <>
          <div className="border-t border-gray-800 my-2" />
          <div className="text-gray-600">Limited real-world benchmark data</div>
        </>
      )}
    </InfoCard>
  );
}

export function LatencyAssumptionsPanel() {
  const { mode } = useConfigStore();

  if (mode === 'inference') {
    return (
      <InfoCard title="Inference Assumptions" icon={<Info className="w-4 h-4 text-orange-400" />}>
        <div className="text-gray-400 space-y-2">
          <ul className="list-disc list-inside space-y-1 text-gray-500">
            <li>Prefill: compute-bound (2P FLOPs)</li>
            <li>Decode: memory-BW-bound (weights + KV cache)</li>
            <li>Compute MFU: 40%</li>
            <li>BW efficiency: 35-85% (scales with model size)</li>
            <li>Dequant overhead: 5-20% (INT4: 20%, FP8: 5%)</li>
            <li>W4A16/GGUF at BF16 TFLOPS; FP8 at native FP8 TFLOPS</li>
            <li>Continuous batching: 1-2% overhead, 0-10% prefill interference</li>
          </ul>
          <div className="border-t border-gray-800 my-2" />
          <p className="text-orange-400 text-xs opacity-80">
            Real-world performance varies with serving framework
            and quantization kernel quality.
          </p>
        </div>
      </InfoCard>
    );
  }

  return (
    <InfoCard title="Communication Assumptions" icon={<Info className="w-4 h-4 text-orange-400" />}>
      <div className="text-gray-400 space-y-2">
        <p>Simulation uses the following assumptions:</p>
        <ul className="list-disc list-inside space-y-1 text-gray-500">
          <li>NCCL-optimized collectives</li>
          <li>Compute-communication overlap where possible</li>
          <li>Ring AllReduce for DP gradients</li>
          <li>AllReduce for TP, point-to-point for PP</li>
          <li>Fat-tree topology for multi-node</li>
          <li>80-92% achieved bandwidth efficiency</li>
        </ul>
        <div className="border-t border-gray-800 my-2" />
        <p className="text-orange-400 text-xs opacity-80">
          Note: Real-world performance varies based on network congestion,
          driver versions, and system configuration.
        </p>
      </div>
    </InfoCard>
  );
}

export function MemoryBreakdownPanel() {
  const config = useConfigStore();
  const { modelSpec, precision, mode, clusterConfig, training, sequenceLength } = config;

  // Try to compute strategy-aware memory estimate
  const memResult = useMemo<{ metrics: SimulationMetrics; error?: undefined } | { metrics?: undefined; error: string } | null>(() => {
    if (!modelSpec || !clusterConfig || mode !== 'training') return null;
    try {
      const metrics = getSimulationMetrics({
        modelSpec,
        clusterConfig,
        globalBatchSize: training.globalBatchSize,
        microBatchSize: training.microBatchSize,
        sequenceLength,
        gradientAccumulationSteps: training.gradientAccumulationSteps,
        strategyType: training.strategyType,
        strategyConfig: {
          tp: training.tpDegree,
          pp: training.ppDegree,
          dp: training.dpDegree,
          ep: training.epDegree,
          cp: training.cpDegree,
          numMicroBatches: training.numMicroBatches,
          sequenceParallel: training.sequenceParallel,
          pipelineSchedule: training.pipelineSchedule,
          interleavedStages: training.interleavedStages,
        },
        activationCheckpointing: training.activationCheckpointing,
        checkpointingGranularity: training.checkpointingGranularity,
        flashAttention: training.flashAttention,
        mixedPrecision: precision,
        finetuningMethod: training.finetuningMethod,
        loraRank: training.loraRank,
        loraTargetModules: training.loraTargetModules,
      });
      return { metrics };
    } catch (e) {
      console.warn('[MemoryEstimate] getSimulationMetrics failed:', e);
      return { error: e instanceof Error ? e.message : String(e) };
    }
  }, [modelSpec, clusterConfig, mode, training, sequenceLength, precision]);

  if (!modelSpec) {
    return null;
  }

  // Strategy-aware display
  if (memResult?.metrics) {
    const mem = memResult.metrics.memoryPerGPU;
    const util = memResult.metrics.memoryUtilization;
    const gpuMemGB = clusterConfig!.node.gpu.memoryGB;
    const isOOM = util > 1.0;
    const utilPct = Math.min(util * 100, 100);

    const barColor = isOOM ? 'bg-red-500' : util >= 0.95 ? 'bg-red-500' : util >= 0.80 ? 'bg-yellow-500' : 'bg-emerald-500';
    const pctColor = isOOM ? 'text-red-400' : util >= 0.95 ? 'text-red-400' : util >= 0.80 ? 'text-yellow-400' : 'text-emerald-400';

    // Overhead = gather buffers + NCCL temporaries + CUDA reserved
    const gatherOverhead = mem.peakActivations - mem.activations;
    const overhead = gatherOverhead + mem.temporary + mem.reserved;

    return (
      <InfoCard title="Memory Estimate" icon={<Database className="w-4 h-4 text-cyan-400" />}>
        <div className="flex items-baseline justify-between mb-2">
          <span className="text-gray-400 text-xs">Per GPU</span>
          <span className={`font-mono text-xs ${isOOM ? 'text-red-400' : 'text-gray-300'}`}>
            {formatBytes(mem.total, 1)} / {gpuMemGB.toFixed(0)} GB
            <span className={`ml-3 ${pctColor}`}>
              {isOOM ? 'OOM' : `${utilPct.toFixed(0)}%`}
            </span>
          </span>
        </div>

        {/* Utilization bar */}
        <div className="w-full h-1.5 bg-gray-800 rounded-full mb-3 overflow-hidden">
          <div
            className={`h-full rounded-full ${barColor} transition-all duration-300`}
            style={{ width: `${utilPct}%` }}
          />
        </div>

        <InfoRow label="Parameters" value={formatBytes(mem.parameters)} />
        <InfoRow label="Gradients" value={formatBytes(mem.gradients)} />
        <InfoRow label="Optimizer" value={formatBytes(mem.optimizerStates)} />
        <InfoRow label="Activations" value={formatBytes(mem.activations)} />
        {/* Overhead row with info tooltip */}
        <div className="flex justify-between">
          <span className="text-gray-500 flex items-center gap-1">
            Overhead
            <span className="relative group/overhead inline-flex">
              <Info className="w-3 h-3 text-gray-600 cursor-help" />
              <span className="absolute top-1/2 left-full -translate-y-1/2 ml-2 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-xs text-gray-300 opacity-0 group-hover/overhead:opacity-100 transition-opacity pointer-events-none z-10 w-56">
                {gatherOverhead > 0 && <span className="block">FSDP gather buffers: {formatBytes(gatherOverhead, 0)}</span>}
                {mem.temporary > 0 && <span className="block">NCCL buffers: {formatBytes(mem.temporary, 0)}</span>}
                <span className="block">CUDA + fragmentation: {formatBytes(mem.reserved, 0)}</span>
              </span>
            </span>
          </span>
          <span className="text-gray-300 font-mono">{formatBytes(overhead)}</span>
        </div>
        <div className="border-t border-gray-800 my-2" />
        <InfoRow label="Total" value={formatBytes(mem.total, 1)} />

        {isOOM && (
          <div className="flex items-center gap-1.5 mt-2 text-red-400 text-xs">
            <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0" />
            <span>Out of memory</span>
          </div>
        )}
      </InfoCard>
    );
  }

  // Show error if simulation failed (debug aid)
  if (memResult?.error) {
    console.warn('[MemoryEstimate]', memResult.error);
  }

  // Fallback: baseline display (inference mode or invalid config)
  const dtypes = DTYPE_PRESETS[precision] ?? DTYPE_PRESETS.bf16;
  const isLoRA = training.finetuningMethod === 'lora' || training.finetuningMethod === 'qlora';
  const isQLoRA = training.finetuningMethod === 'qlora';

  let paramMemory: number;
  let gradientMemory: number;
  let optimizerMemory: number;

  if (isLoRA) {
    const trainableParams = computeLoraTrainableParams(modelSpec, training.loraRank, training.loraTargetModules);
    const baseStorageBytes = isQLoRA ? NF4_BYTES_PER_PARAM : DTYPE_BYTES[dtypes.params];
    paramMemory = modelSpec.totalParams * baseStorageBytes + trainableParams * 2; // base + adapters in BF16
    gradientMemory = trainableParams * 2; // adapter grads only, BF16
    optimizerMemory = trainableParams * 12; // master + m + v for adapters only
  } else {
    paramMemory = modelSpec.totalParams * DTYPE_BYTES[dtypes.params];
    gradientMemory = modelSpec.totalParams * DTYPE_BYTES[dtypes.gradients];
    const needsMasterCopy = dtypes.params !== 'fp32' && dtypes.params !== 'tf32';
    optimizerMemory = modelSpec.totalParams * (needsMasterCopy ? 12 : 8);
  }

  return (
    <InfoCard title="Baseline Memory" icon={<Database className="w-4 h-4 text-cyan-400" />}>
      <div className="text-gray-400 text-xs mb-2">Per GPU before sharding</div>
      <InfoRow label="Parameters" value={formatBytes(paramMemory)} />
      <InfoRow label="Gradients" value={formatBytes(gradientMemory)} />
      <InfoRow label="Optimizer" value={formatBytes(optimizerMemory)} />
      <div className="border-t border-gray-800 my-2" />
      <InfoRow
        label="Training Total"
        value={formatBytes(paramMemory + gradientMemory + optimizerMemory)}
      />
      <InfoRow
        label="Inference Only"
        value={formatBytes(modelSpec.totalParams * DTYPE_BYTES[dtypes.params])}
      />
    </InfoCard>
  );
}

/**
 * Combined info panel showing all configuration details
 */
export function InfoPanel() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      <ModelSpecPanel />
      <HardwarePanel />
      <NetworkingPanel />
      <MemoryBreakdownPanel />
      <GPUGridPanel />
      <InferenceGPUGridPanel />
      <LatencyAssumptionsPanel />
    </div>
  );
}
