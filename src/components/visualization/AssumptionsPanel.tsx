/**
 * Assumptions Panel - Displays comprehensive modeling assumptions
 *
 * This component provides transparency about the simulation's modeling
 * choices and their impact on accuracy.
 */

import { useState } from 'react';
import { Info, Cpu, Wifi, Database, AlertTriangle, ChevronRight } from 'lucide-react';

interface SectionProps {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}

function Section({ title, icon, children }: SectionProps) {
  return (
    <div className="space-y-2">
      <h4 className="text-sm font-medium text-gray-300 flex items-center gap-2">
        {icon}
        {title}
      </h4>
      <ul className="space-y-1 text-xs text-gray-400 ml-6">
        {children}
      </ul>
    </div>
  );
}

function AssumptionItem({ children }: { children: React.ReactNode }) {
  return (
    <li className="flex items-start gap-2">
      <span className="text-gray-600 mt-0.5">-</span>
      <span>{children}</span>
    </li>
  );
}

export function AssumptionsPanel({ resolvedStoredLayers, numLayers }: {
  resolvedStoredLayers?: number;
  numLayers?: number;
} = {}) {
  const [open, setOpen] = useState(false);

  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-xl mt-4">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-5 py-3 text-left cursor-pointer hover:bg-gray-800/50 transition-colors rounded-xl"
      >
        <Info className="w-5 h-5 text-blue-400 flex-shrink-0" />
        <h3 className="text-lg font-medium text-white flex-1">Modeling Assumptions</h3>
        <ChevronRight className={`w-4 h-4 text-gray-500 transition-transform ${open ? 'rotate-90' : ''}`} />
      </button>

      {open && <div className="grid grid-cols-1 md:grid-cols-2 gap-6 px-5 pt-2 pb-5">
        {/* FLOP Calculation */}
        <Section title="FLOP Calculation" icon={<Cpu className="w-4 h-4 text-purple-400" />}>
          <AssumptionItem>
            Training: <span className="text-gray-300">6P x tokens</span> without checkpointing
          </AssumptionItem>
          <AssumptionItem>
            Training: <span className="text-gray-300">8P x tokens</span> with activation checkpointing
          </AssumptionItem>
          <AssumptionItem>
            Forward only: <span className="text-gray-300">2P x tokens</span>
          </AssumptionItem>
          <AssumptionItem>
            MFU reference: dtype-specific tensor core peak
          </AssumptionItem>
        </Section>

        {/* Memory Model */}
        <Section title="Memory Model" icon={<Database className="w-4 h-4 text-cyan-400" />}>
          <AssumptionItem>
            Attention: <span className="text-gray-300">O(seq²)</span> standard, <span className="text-gray-300">O(seq)</span> with Flash Attention
          </AssumptionItem>
          <AssumptionItem>
            Optimizer: AdamW = <span className="text-gray-300">12 bytes/param</span> (m + v + master)
          </AssumptionItem>
          <AssumptionItem>
            Fragmentation: <span className="text-gray-300">7%</span> + ~1GB CUDA context reserve
          </AssumptionItem>
          <AssumptionItem>
            Full AC: <span className="text-gray-300">&radic;N</span> layers stored (Chen et al. 2016 optimal segmenting)
          </AssumptionItem>
          <AssumptionItem>
            {resolvedStoredLayers != null && numLayers != null && resolvedStoredLayers < numLayers
              ? <>Selective AC: <span className="text-gray-300">{resolvedStoredLayers} of {numLayers}</span> layers store MLP+shared activations, {numLayers - resolvedStoredLayers} use full recompute (budget mode &mdash; auto-sized to fit GPU memory)</>
              : <>Selective AC: all layers store MLP+shared activations, attention recomputed per-layer</>}
          </AssumptionItem>
        </Section>

        {/* Communication */}
        <Section title="Communication Overlap" icon={<Wifi className="w-4 h-4 text-green-400" />}>
          <AssumptionItem>
            DDP overlap: <span className="text-gray-300">50-96%</span> (bucket model, compute/comm dependent)
          </AssumptionItem>
          <AssumptionItem>
            FSDP overlap: per-layer pipeline, <span className="text-gray-300">(L-1)/L</span> in compute-bound regime
          </AssumptionItem>
          <AssumptionItem>
            TP overlap: dynamic <span className="text-gray-300">C/(C+T)</span>, ~50% (small models) to ~95% (large models)
          </AssumptionItem>
          <AssumptionItem>
            PP overlap: dynamic, up to <span className="text-gray-300">95%</span> (nearly free for large models)
          </AssumptionItem>
          <AssumptionItem>
            DP bandwidth: <span className="text-gray-300">~80%</span> of wire, degrades with DP group size
          </AssumptionItem>
        </Section>

        {/* MoE / Expert Parallelism */}
        <Section title="MoE / Expert Parallelism" icon={<Cpu className="w-4 h-4 text-amber-400" />}>
          <AssumptionItem>
            MFU uses <span className="text-gray-300">activeParams</span> (only active experts per token)
          </AssumptionItem>
          <AssumptionItem>
            Expert memory sharded by EP, shared params by TP&times;PP&times;DP
          </AssumptionItem>
          <AssumptionItem>
            All-to-All: <span className="text-gray-300">4&times;</span> per MoE layer (fwd+bwd dispatch+combine)
          </AssumptionItem>
          <AssumptionItem>
            EP overlap: slack-based, up to <span className="text-gray-300">70%</span> of idle compute cycles
          </AssumptionItem>
          <AssumptionItem>
            Capacity factor: <span className="text-gray-300">1.15&times;</span> on expert buffers (routing imbalance headroom)
          </AssumptionItem>
        </Section>

        {/* Large-Cluster Scaling */}
        <Section title="Large-Cluster Scaling" icon={<Cpu className="w-4 h-4 text-red-400" />}>
          <AssumptionItem>
            Straggler penalty: <span className="text-gray-300">&sigma;&radic;(2&middot;ln N)</span> order statistics, &sigma;=1.4% per-node CV
          </AssumptionItem>
        </Section>

        {/* Not Modeled */}
        <Section title="Not Modeled" icon={<AlertTriangle className="w-4 h-4 text-amber-400" />}>
          <AssumptionItem>CPU offloading (ZeRO-Infinity)</AssumptionItem>
          <AssumptionItem>Dynamic batching / padding overhead</AssumptionItem>
          <AssumptionItem>GPU/network hardware failures (checkpoint recovery downtime)</AssumptionItem>
          <AssumptionItem>Thermal throttling / power limits</AssumptionItem>
          <AssumptionItem>Data loading / preprocessing time</AssumptionItem>
          <AssumptionItem>Checkpoint I/O during training</AssumptionItem>
        </Section>
      </div>}

    </div>
  );
}
