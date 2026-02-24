/**
 * GPU Partitioning Grid — interactive 2D visualization showing how GPUs
 * are partitioned across TP / PP / DP dimensions.
 *
 * Rows = physical nodes, columns = GPUs per node.
 * Color = PP stage, border grouping = TP group.
 * Diagonal stripes on cells = model is sharded across TP group (not a full copy).
 */

import { LayoutGrid } from 'lucide-react';
import { useConfigStore } from '../../stores/config.ts';

// ── Helpers ──────────────────────────────────────────────────────────────────

interface GPUAssignment {
  rank: number;
  tpRank: number;
  ppStage: number;
  cpRank: number;
  dpRank: number;
  epRank: number;
}

/** Megatron-LM rank-mapping convention, extended with EP and CP dimensions. */
function computeGPUAssignment(rank: number, tp: number, pp: number, ep: number = 1, cp: number = 1): GPUAssignment {
  // Layout: [DP][EP][CP][PP][TP] — innermost varies fastest
  // CP between PP and EP: each CP rank holds seq/cp tokens
  return {
    rank,
    tpRank: rank % tp,
    ppStage: Math.floor(rank / tp) % pp,
    cpRank: Math.floor(rank / (tp * pp)) % cp,
    epRank: Math.floor(rank / (tp * pp * cp)) % ep,
    dpRank: Math.floor(rank / (tp * pp * cp * ep)),
  };
}

const PP_COLORS = [
  'bg-blue-500',   // Stage 0
  'bg-purple-500', // Stage 1
  'bg-cyan-500',   // Stage 2
  'bg-orange-500', // Stage 3
  'bg-green-500',  // Stage 4
  'bg-red-500',    // Stage 5
  'bg-yellow-500', // Stage 6
  'bg-pink-500',   // Stage 7
];

/** Diagonal stripe overlay style — signals "sharded slice, not a full copy". */
const TP_STRIPE_STYLE: React.CSSProperties = {
  backgroundImage:
    'repeating-linear-gradient(45deg, transparent, transparent 2px, rgba(255,255,255,0.10) 2px, rgba(255,255,255,0.10) 4px)',
};

// ── Sub-components ───────────────────────────────────────────────────────────

function GPUCell({
  assignment,
  tp,
  pp,
  ep,
  cp,
  ghost,
  compact,
}: {
  assignment: GPUAssignment;
  tp: number;
  pp: number;
  ep: number;
  cp: number;
  ghost?: boolean;
  compact?: boolean;
}) {
  const color = PP_COLORS[assignment.ppStage % PP_COLORS.length];
  const epSuffix = ep > 1 ? ` EP${assignment.epRank}` : '';
  const cpSuffix = cp > 1 ? ` CP${assignment.cpRank}` : '';
  const tooltip =
    tp === 1 && pp === 1
      ? `GPU ${assignment.rank} | DP${assignment.dpRank}${cpSuffix}${epSuffix}`
      : `GPU ${assignment.rank} | TP${assignment.tpRank} PP${assignment.ppStage} DP${assignment.dpRank}${cpSuffix}${epSuffix}`;

  return (
    <div
      className={`${compact ? 'w-[18px] h-[18px]' : 'w-5 h-5'} rounded-sm ${color} ${ghost ? 'opacity-25' : 'opacity-90 hover:opacity-100'} cursor-default transition-opacity`}
      style={tp > 1 ? TP_STRIPE_STYLE : undefined}
      title={ghost ? undefined : tooltip}
    />
  );
}

function NodeRow({
  nodeIndex,
  gpuCount,
  startRank,
  tp,
  pp,
  ep,
  cp,
  ghost,
  compact,
  wideLabel,
}: {
  nodeIndex: number;
  gpuCount: number;
  startRank: number;
  tp: number;
  pp: number;
  ep: number;
  cp: number;
  ghost?: boolean;
  compact?: boolean;
  wideLabel?: boolean;
}) {
  // Build TP groups (chunks of `tp` GPUs)
  const tpGroups: GPUAssignment[][] = [];
  let currentGroup: GPUAssignment[] = [];

  for (let i = 0; i < gpuCount; i++) {
    const rank = startRank + i;
    const a = computeGPUAssignment(rank, tp, pp, ep, cp);
    currentGroup.push(a);
    if (currentGroup.length === tp || i === gpuCount - 1) {
      tpGroups.push(currentGroup);
      currentGroup = [];
    }
  }

  // Group TP groups into CP groups: each CP group spans `pp` consecutive TP groups
  // Layout is [DP][EP][CP][PP][TP], so CP groups are contiguous
  const tpGroupsPerCP = Math.max(1, pp);
  const cpGroups: { cpRank: number; tpGroups: GPUAssignment[][] }[] = [];
  for (let i = 0; i < tpGroups.length; i += tpGroupsPerCP) {
    const slice = tpGroups.slice(i, i + tpGroupsPerCP);
    const cpRank = slice[0]?.[0]?.cpRank ?? Math.floor(i / tpGroupsPerCP);
    cpGroups.push({ cpRank, tpGroups: slice });
  }

  // Group CP groups into EP groups: each EP group spans `cp` consecutive CP groups
  const cpGroupsPerEP = Math.max(1, cp);
  const epGroups: { epRank: number; cpGroups: typeof cpGroups }[] = [];
  for (let i = 0; i < cpGroups.length; i += cpGroupsPerEP) {
    const slice = cpGroups.slice(i, i + cpGroupsPerEP);
    const epRank = slice[0]?.tpGroups[0]?.[0]?.epRank ?? Math.floor(i / cpGroupsPerEP);
    epGroups.push({ epRank, cpGroups: slice });
  }

  const renderTPGroup = (group: GPUAssignment[], gi: number) => (
    <div
      key={gi}
      className={
        tp > 1
          ? `flex gap-0.5 border rounded-sm ${compact ? 'p-px' : 'p-0.5'} ${ghost ? 'border-gray-800/40' : 'border-teal-500/70'}`
          : 'flex gap-0.5'
      }
    >
      {group.map((a) => (
        <GPUCell key={a.rank} assignment={a} tp={tp} pp={pp} ep={ep} cp={cp} ghost={ghost} compact={compact} />
      ))}
    </div>
  );

  const renderCPGroup = (cpGroup: typeof cpGroups[number], ci: number) => {
    if (cp <= 1) {
      // No CP border — render TP groups directly
      return cpGroup.tpGroups.map((group, gi) => renderTPGroup(group, gi));
    }
    return (
      <div
        key={ci}
        className={`flex gap-0.5 border border-dotted rounded-md ${compact ? 'px-0.5 pt-1 pb-0.5' : 'px-0.5 pt-1 pb-1'} relative ${
          ghost ? 'border-gray-800/30' : 'border-green-500/80'
        }`}
      >
        {!ghost && (
          <span className={`absolute ${compact ? '-top-1.5 left-1 text-[7px] bg-gray-800' : '-top-2 left-1.5 text-[9px] bg-gray-900'} px-0.5 rounded font-medium text-green-500`}>
            CP{cpGroup.cpRank}
          </span>
        )}
        {cpGroup.tpGroups.map((group, gi) => renderTPGroup(group, gi))}
      </div>
    );
  };

  const outerGap = (ep > 1 || cp > 1) ? (compact ? 'gap-3' : 'gap-6') : (compact ? 'gap-1' : 'gap-2');

  return (
    <div className={`flex items-center gap-0.5 ${ghost ? 'pointer-events-none' : ''}`}>
      <span className={`${compact ? `text-[9px] ${wideLabel ? 'w-5' : 'w-3.5'}` : `text-xs ${wideLabel ? 'w-6' : 'w-4'}`} font-mono flex-shrink-0 ${ghost ? 'text-gray-700' : 'text-gray-600'} -ml-0.5`}>
        N{nodeIndex}
      </span>
      <div className={`flex ${outerGap} flex-wrap`}>
        {ep > 1 ? (
          // Render EP group borders containing CP groups
          epGroups.map((epGroup) => (
            <div
              key={epGroup.epRank}
              className={`flex gap-0.5 border border-dashed rounded-md ${compact ? `px-0.5 ${cp > 1 ? 'pt-2' : 'pt-1'} pb-0.5` : `px-0.5 ${cp > 1 ? 'pt-3' : 'pt-1'} pb-1`} relative ${
                ghost ? 'border-gray-800/30' : 'border-orange-500/80'
              }`}
            >
              {!ghost && (
                <span className={`absolute ${compact ? '-top-1.5 left-1 text-[7px] bg-gray-800' : '-top-2 left-1.5 text-[9px] bg-gray-900'} px-0.5 rounded font-medium text-orange-500`}>
                  EP{epGroup.epRank}
                </span>
              )}
              {epGroup.cpGroups.map((cpGroup, ci) => renderCPGroup(cpGroup, ci))}
            </div>
          ))
        ) : (
          // No EP grouping — render CP groups (or flat TP groups if cp=1)
          cpGroups.map((cpGroup, ci) => renderCPGroup(cpGroup, ci))
        )}
      </div>
    </div>
  );
}

/** Vertical arrow column — downward or upward. */
function VerticalArrow({ label, direction }: { label: string; direction: 'down' | 'up' }) {
  const isDown = direction === 'down';
  return (
    <div className="flex flex-col items-center flex-shrink-0 w-6">
      {isDown && <span className="text-[10px] text-gray-500 font-medium mb-0.5">{label}</span>}
      {!isDown && <span className="text-gray-500 text-xs leading-none mb-0.5">{'\u25B2'}</span>}
      <div className="flex-1 w-px bg-gray-500" />
      {isDown && <span className="text-gray-500 text-xs leading-none mt-0.5">{'\u25BC'}</span>}
      {!isDown && <span className="text-[10px] text-gray-500 font-medium mt-0.5">{label}</span>}
    </div>
  );
}

/** FWD ↓ left, grid center, BWD ↑ right, curved "loss" at bottom. */
function PipelineFlow({ children }: { children: React.ReactNode; compact?: boolean }) {
  return (
    <div className="flex flex-col items-center">
      <div className="w-fit">
        {/* Main row: FWD | grid | BWD */}
        <div className="flex items-stretch gap-px">
          <VerticalArrow label="FWD" direction="down" />
          <div>{children}</div>
          <VerticalArrow label="BWD" direction="up" />
        </div>
        {/* Loss arrow: constrained to same width as FWD/grid/BWD row above */}
        <div className="flex items-center gap-1 mt-1" style={{ marginLeft: 12, marginRight: 12 }}>
          <span className="text-[10px] text-gray-500 font-medium flex-shrink-0">LOSS</span>
          <div className="flex-1 h-px bg-gray-500" />
          <span className="text-gray-500 text-xs leading-none flex-shrink-0">{'\u25B6'}</span>
        </div>
      </div>
    </div>
  );
}

function GPULegend({
  tp,
  pp,
  dp,
  ep,
  cp,
  numGPUs,
  visibleStages,
  compact,
}: {
  tp: number;
  pp: number;
  dp: number;
  ep: number;
  cp: number;
  numGPUs: number;
  visibleStages?: Set<number>;
  compact?: boolean;
}) {
  const is1D = tp === 1 && pp === 1 && ep === 1 && cp === 1;

  return (
    <div className={`${compact ? 'space-y-0.5 mt-1.5' : 'space-y-1.5 mt-3'}`}>
      {/* Formula line */}
      <div className={`text-xs font-mono text-gray-400`}>
        {is1D ? (
          <>DP={dp} = {numGPUs} GPUs</>
        ) : (
          <span style={{ wordSpacing: '-2px' }}>TP={tp} × PP={pp}{cp > 1 ? ` × CP=${cp}` : ''} × DP={dp} = {numGPUs} GPUs{ep > 1 ? ` (EP=${ep})` : ''}</span>
        )}
      </div>

      {/* Swatches */}
      <div className={`flex flex-wrap items-center ${compact ? 'gap-2' : 'gap-3'} text-xs text-gray-500`}>
        {is1D ? (
          <div className="flex items-center gap-1">
            <div className={`${compact ? 'w-2 h-2' : 'w-3 h-3'} rounded-sm bg-blue-500`} />
            <span>DP ({dp} replicas)</span>
          </div>
        ) : (
          <>
            {pp > 1 &&
              Array.from({ length: pp }, (_, i) => (
                visibleStages && !visibleStages.has(i) ? null : (
                  <div key={i} className="flex items-center gap-0.5">
                    <div className={`${compact ? 'w-2 h-2' : 'w-3 h-3'} rounded-sm ${PP_COLORS[i % PP_COLORS.length]}`} />
                    <span>Stage {i}</span>
                  </div>
                )
              ))}
            {tp > 1 && (
              <div className="flex items-center gap-1">
                <div className={`flex gap-px border border-teal-500/70 rounded-sm p-px`}>
                  <div className={`${compact ? 'w-1.5 h-1.5' : 'w-2.5 h-2.5'} rounded-sm bg-indigo-400/60`} style={TP_STRIPE_STYLE} />
                  <div className={`${compact ? 'w-1.5 h-1.5' : 'w-2.5 h-2.5'} rounded-sm bg-indigo-400/60`} style={TP_STRIPE_STYLE} />
                </div>
                <span>TP group</span>
              </div>
            )}
            {cp > 1 && (
              <div className="flex items-center gap-1">
                <div className="flex gap-px border border-dotted border-green-500/80 rounded px-0.5 py-px">
                  <div className={`${compact ? 'w-1.5 h-1.5' : 'w-2.5 h-2.5'} rounded-sm bg-blue-500/60`} />
                  <div className={`${compact ? 'w-1.5 h-1.5' : 'w-2.5 h-2.5'} rounded-sm bg-blue-500/60`} />
                </div>
                <span>CP group</span>
              </div>
            )}
            {ep > 1 && (
              <div className="flex items-center gap-1">
                <div className="flex gap-px border border-dashed border-orange-500/80 rounded px-0.5 py-px">
                  <div className={`${compact ? 'w-1.5 h-1.5' : 'w-2.5 h-2.5'} rounded-sm bg-blue-500/60`} />
                  <div className={`${compact ? 'w-1.5 h-1.5' : 'w-2.5 h-2.5'} rounded-sm bg-blue-500/60`} />
                </div>
                <span>EP group</span>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ── Main exported component ──────────────────────────────────────────────────

export function GPUGridPanel({ embedded, hideTitle }: { embedded?: boolean; hideTitle?: boolean } = {}) {
  const { numGPUs, gpusPerNode, training, mode } = useConfigStore();

  if (mode !== 'training' || numGPUs <= 1) return null;

  const tp = training.tpDegree;
  const pp = training.ppDegree;
  const dp = training.dpDegree;
  const ep = training.epDegree;
  const cp = training.cpDegree;
  const numNodes = Math.ceil(numGPUs / gpusPerNode);
  const is1D = tp === 1 && pp === 1 && ep === 1 && cp === 1;
  const is3D = pp > 1;

  // ── Scale / truncation ────────────────────────────────────────────────────
  let nodesToRender: number;
  let showGhostNodes = false; // faded trailing rows for DDP truncation
  let showReplicaBox = false; // dashed bounding box for 3D truncation
  let annotation: string | null = null;

  if (numNodes <= 12 && ep <= 1 && cp <= 1) {
    nodesToRender = numNodes;
  } else if (numNodes <= 12 && (ep > 1 || cp > 1)) {
    // Show one DP replica with EP/CP group boxes, even for small clusters
    nodesToRender = Math.max(1, Math.ceil((tp * pp * cp * ep) / gpusPerNode));
    const independentReplicas = Math.floor(dp / ep);
    showReplicaBox = independentReplicas > 1;
    if (independentReplicas > 1) {
      annotation = `\u00d7${independentReplicas} replicas (${numNodes} nodes)`;
    }
  } else if (is1D) {
    nodesToRender = 3;
    showGhostNodes = true;
    annotation = `+ ${numNodes - nodesToRender - 2} more nodes (${numGPUs} GPUs, ${numNodes} nodes)`;
  } else {
    // Show one full DP replica (including EP and CP groups)
    nodesToRender = Math.ceil((tp * pp * cp * ep) / gpusPerNode);
    const independentReplicas = Math.floor(dp / Math.max(ep, 1));
    showReplicaBox = independentReplicas > 1;
    if (independentReplicas > 1) {
      annotation = ep > 1
        ? `\u00d7${independentReplicas} replicas, ${ep} EP groups (${numNodes} nodes)`
        : `\u00d7${independentReplicas} DP replicas (${numNodes} nodes)`;
    } else {
      annotation = ep > 1
        ? `Full cluster shown — ${ep} EP groups (${numNodes} nodes)`
        : `Full cluster shown (${numNodes} nodes)`;
    }
  }

  // Ghost node indices (2 faded rows after the real ones for DDP)
  const ghostCount = showGhostNodes ? Math.min(2, numNodes - nodesToRender) : 0;

  // Uniform wide labels when any rendered node index is >= 10
  const maxRenderedNodeIndex = nodesToRender + ghostCount - 1;
  const wideLabel = maxRenderedNodeIndex >= 10;

  // ── Node row truncation for high PP ────────────────────────────────────────
  const MAX_VISIBLE_NODES = 8;
  const TRUNCATE_HEAD = 4;
  const TRUNCATE_TAIL = 4;
  const truncateNodes = nodesToRender > MAX_VISIBLE_NODES;
  const hiddenNodes = truncateNodes ? nodesToRender - TRUNCATE_HEAD - TRUNCATE_TAIL : 0;

  // Compute which PP stages are visible (for legend filtering when truncated)
  let visibleStages: Set<number> | undefined;
  if (truncateNodes && pp > 1) {
    visibleStages = new Set<number>();
    const visibleNodeIndices = [
      ...Array.from({ length: TRUNCATE_HEAD }, (_, i) => i),
      ...Array.from({ length: TRUNCATE_TAIL }, (_, i) => nodesToRender - TRUNCATE_TAIL + i),
    ];
    for (const ni of visibleNodeIndices) {
      for (let g = 0; g < gpusPerNode; g++) {
        visibleStages.add(computeGPUAssignment(ni * gpusPerNode + g, tp, pp, ep, cp).ppStage);
      }
    }
  }

  const renderNodeRow = (ni: number, ghost?: boolean) => {
    const gpuCount = ghost ? gpusPerNode : Math.min(gpusPerNode, numGPUs - ni * gpusPerNode);
    return (
      <NodeRow
        key={ghost ? `ghost-${ni}` : ni}
        nodeIndex={ni}
        gpuCount={gpuCount}
        startRank={ni * gpusPerNode}
        tp={tp}
        pp={pp}
        ep={ep}
        cp={cp}
        ghost={ghost}
        compact={embedded}
        wideLabel={wideLabel}
      />
    );
  };

  const gridRows = (
    <div className={`${embedded ? 'space-y-0.5 px-2' : ep > 1 ? 'space-y-1.5 px-2' : 'space-y-1 px-2'}`}>
      {truncateNodes ? (
        <>
          {Array.from({ length: TRUNCATE_HEAD }, (_, ni) => renderNodeRow(ni))}
          <div className="text-xs text-gray-600 font-mono pl-8 py-0.5">
            {'\u22EE'} {hiddenNodes} more nodes
          </div>
          {Array.from({ length: TRUNCATE_TAIL }, (_, i) => {
            const ni = nodesToRender - TRUNCATE_TAIL + i;
            return renderNodeRow(ni);
          })}
        </>
      ) : (
        Array.from({ length: nodesToRender }, (_, ni) => renderNodeRow(ni))
      )}

      {/* Faded ghost nodes for DDP scale hint */}
      {ghostCount > 0 && Array.from({ length: ghostCount }, (_, gi) => {
        const ni = nodesToRender + gi;
        return renderNodeRow(ni, true);
      })}

    </div>
  );

  const content = (
    <>
      {showReplicaBox ? (
        <div className={`border border-dashed border-gray-600 rounded-lg w-fit ${embedded ? 'p-1 pt-2.5 mt-3' : 'p-2 pt-3.5 mx-auto mt-4'} relative`}>
          <span className={`absolute -top-2.5 left-3 ${embedded ? 'bg-gray-800 text-[8px]' : 'bg-gray-900 text-[10px]'} px-1.5 rounded text-gray-300 font-medium`}>
            1 DP Replica{ep > 1 && ` (${ep} EP groups)`}
          </span>
          {is3D ? <PipelineFlow compact={embedded}>{gridRows}</PipelineFlow> : gridRows}
        </div>
      ) : is3D ? (
        <PipelineFlow compact={embedded}>{gridRows}</PipelineFlow>
      ) : (
        gridRows
      )}

      {annotation && (
        <div className={`${embedded ? 'text-[8px] pl-2 py-0.5' : 'text-[10px] pl-2 py-1'} text-gray-600 font-mono`}>
          {showGhostNodes ? '' : '\u22EE '}{annotation}
        </div>
      )}

      <GPULegend tp={tp} pp={pp} dp={dp} ep={ep} cp={cp} numGPUs={numGPUs} visibleStages={visibleStages} compact={embedded} />
    </>
  );

  if (embedded) {
    return (
      <div className="flex flex-col items-center">
        {!hideTitle && (
          <div className="font-medium text-white mb-3 flex items-center gap-1.5 self-start">
            <LayoutGrid className="w-3.5 h-3.5 text-blue-400" />
            GPU Partitioning
          </div>
        )}
        {content}
      </div>
    );
  }

  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-4 overflow-x-auto">
      <h3 className="text-sm font-medium text-gray-300 mb-4 flex items-center gap-2">
        <LayoutGrid className="w-4 h-4 text-blue-400" />
        GPU Partitioning
      </h3>
      {content}
    </div>
  );
}

// ── Inference GPU Grid ──────────────────────────────────────────────────────

function InferenceGPULegend({
  tp,
  ep,
  numReplicas,
  numGPUs,
  compact,
}: {
  tp: number;
  ep: number;
  numReplicas: number;
  numGPUs: number;
  compact?: boolean;
}) {
  const isSimple = tp === 1 && ep === 1;

  return (
    <div className={`${compact ? 'space-y-0.5 mt-1.5' : 'space-y-1.5 mt-3'}`}>
      {/* Formula line */}
      <div className="text-xs font-mono text-gray-400">
        {isSimple ? (
          <>{numGPUs} replicas</>
        ) : ep > 1 ? (
          <span style={{ wordSpacing: '-2px' }}>TP={tp} × EP={ep} × {numReplicas} replicas = {numGPUs} GPUs</span>
        ) : (
          <span style={{ wordSpacing: '-2px' }}>TP={tp} × {numReplicas} replicas = {numGPUs} GPUs</span>
        )}
      </div>

      {/* Swatches */}
      <div className={`flex flex-wrap items-center ${compact ? 'gap-2' : 'gap-3'} text-xs text-gray-500`}>
        {isSimple ? (
          <div className="flex items-center gap-1">
            <div className={`${compact ? 'w-2 h-2' : 'w-3 h-3'} rounded-sm bg-blue-500`} />
            <span>Replica ({numReplicas} instances)</span>
          </div>
        ) : (
          <>
            {tp > 1 && (
              <div className="flex items-center gap-1">
                <div className="flex gap-px border border-teal-500/70 rounded-sm p-px">
                  <div className={`${compact ? 'w-1.5 h-1.5' : 'w-2.5 h-2.5'} rounded-sm bg-blue-400/60`} style={TP_STRIPE_STYLE} />
                  <div className={`${compact ? 'w-1.5 h-1.5' : 'w-2.5 h-2.5'} rounded-sm bg-blue-400/60`} style={TP_STRIPE_STYLE} />
                </div>
                <span>TP group</span>
              </div>
            )}
            {ep > 1 && (
              <div className="flex items-center gap-1">
                <div className="flex gap-px border border-dashed border-orange-500/80 rounded px-0.5 py-px">
                  <div className={`${compact ? 'w-1.5 h-1.5' : 'w-2.5 h-2.5'} rounded-sm bg-blue-500/60`} />
                  <div className={`${compact ? 'w-1.5 h-1.5' : 'w-2.5 h-2.5'} rounded-sm bg-blue-500/60`} />
                </div>
                <span>EP group</span>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export function InferenceGPUGridPanel({ embedded, hideTitle }: { embedded?: boolean; hideTitle?: boolean } = {}) {
  const { numGPUs, gpusPerNode, inference, mode } = useConfigStore();

  if (mode !== 'inference' || numGPUs <= 1) return null;

  const tp = inference.tensorParallel || 1;
  const ep = inference.expertParallel || 1;
  const gpusPerReplica = tp * ep;
  const numReplicas = Math.max(1, Math.floor(numGPUs / gpusPerReplica));
  const numNodes = Math.ceil(numGPUs / gpusPerNode);

  // ── Scale / truncation ──────────────────────────────────────────────────
  const nodesPerReplica = Math.max(1, Math.ceil(gpusPerReplica / gpusPerNode));
  const isSimple = tp === 1 && ep === 1;

  let nodesToRender: number;
  let showGhostNodes = false;
  let showReplicaBox = false;
  let annotation: string | null = null;

  if (isSimple) {
    // Pure replicas — show a few GPU cells + ghost hint
    if (numNodes <= 12) {
      nodesToRender = numNodes;
    } else {
      nodesToRender = 3;
      showGhostNodes = true;
      annotation = `+ ${numNodes - nodesToRender - 2} more nodes (${numGPUs} GPUs, ${numNodes} nodes)`;
    }
  } else {
    // TP and/or EP — show one replica
    nodesToRender = Math.min(nodesPerReplica, 8);
    showReplicaBox = numReplicas > 1;
    if (numReplicas > 1) {
      annotation = `\u00d7${numReplicas} replicas (${numNodes} nodes)`;
    } else {
      annotation = `Full cluster shown (${numNodes} nodes)`;
    }
  }

  const ghostCount = showGhostNodes ? Math.min(2, numNodes - nodesToRender) : 0;
  const maxRenderedNodeIndex = nodesToRender + ghostCount - 1;
  const wideLabel = maxRenderedNodeIndex >= 10;

  // Node row truncation for large replicas
  const truncateNodes = nodesPerReplica > 8 && !isSimple;
  const hiddenNodes = truncateNodes ? nodesPerReplica - 8 : 0;

  const renderNodeRow = (ni: number, ghost?: boolean) => {
    const gpuCount = ghost
      ? gpusPerNode
      : isSimple
        ? Math.min(gpusPerNode, numGPUs - ni * gpusPerNode)
        : Math.min(gpusPerNode, gpusPerReplica - ni * gpusPerNode);
    if (gpuCount <= 0) return null;
    return (
      <NodeRow
        key={ghost ? `ghost-${ni}` : ni}
        nodeIndex={ni}
        gpuCount={gpuCount}
        startRank={ni * gpusPerNode}
        tp={tp}
        pp={1}
        ep={ep}
        cp={1}
        ghost={ghost}
        compact={embedded}
        wideLabel={wideLabel}
      />
    );
  };

  const gridRows = (
    <div className={`${embedded ? 'space-y-0.5 px-2' : ep > 1 ? 'space-y-1.5 px-2' : 'space-y-1 px-2'}`}>
      {truncateNodes ? (
        <>
          {Array.from({ length: 4 }, (_, ni) => renderNodeRow(ni))}
          <div className="text-xs text-gray-600 font-mono pl-8 py-0.5">
            {'\u22EE'} {hiddenNodes} more nodes
          </div>
          {Array.from({ length: 4 }, (_, i) => {
            const ni = nodesPerReplica - 4 + i;
            return renderNodeRow(ni);
          })}
        </>
      ) : (
        Array.from({ length: nodesToRender }, (_, ni) => renderNodeRow(ni))
      )}

      {/* Faded ghost nodes for simple replica scale hint */}
      {ghostCount > 0 && Array.from({ length: ghostCount }, (_, gi) => {
        const ni = nodesToRender + gi;
        return renderNodeRow(ni, true);
      })}
    </div>
  );

  const content = (
    <>
      {showReplicaBox ? (
        <div className={`border border-dashed border-gray-600 rounded-lg w-fit ${embedded ? 'p-1 pt-2.5 mt-3' : 'p-2 pt-3.5 mx-auto mt-4'} relative`}>
          <span className={`absolute -top-2.5 left-3 ${embedded ? 'bg-gray-800 text-[8px]' : 'bg-gray-900 text-[10px]'} px-1.5 rounded text-gray-300 font-medium`}>
            1 Serving Replica{ep > 1 && ` (${ep} EP groups)`}
          </span>
          {gridRows}
        </div>
      ) : (
        gridRows
      )}

      {annotation && (
        <div className={`${embedded ? 'text-[8px] pl-2 py-0.5' : 'text-[10px] pl-2 py-1'} text-gray-600 font-mono`}>
          {showGhostNodes ? '' : '\u22EE '}{annotation}
        </div>
      )}

      <InferenceGPULegend tp={tp} ep={ep} numReplicas={numReplicas} numGPUs={numGPUs} compact={embedded} />
    </>
  );

  if (embedded) {
    return (
      <div className="flex flex-col items-center">
        {!hideTitle && (
          <div className="font-medium text-white mb-3 flex items-center gap-1.5 self-start">
            <LayoutGrid className="w-3.5 h-3.5 text-blue-400" />
            GPU Partitioning
          </div>
        )}
        {content}
      </div>
    );
  }

  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-4 overflow-x-auto">
      <h3 className="text-sm font-medium text-gray-300 mb-4 flex items-center gap-2">
        <LayoutGrid className="w-4 h-4 text-blue-400" />
        GPU Partitioning
      </h3>
      {content}
    </div>
  );
}
