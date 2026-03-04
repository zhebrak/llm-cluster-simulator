/**
 * Unified Overlap Runtime Model — Physics-Based
 *
 * Computes communication/compute overlap for each parallelism dimension using
 * theoretical timeline models × scheduling efficiency (η=0.96). Two-term
 * protocol overhead: proportional (per-byte NCCL framing) + fixed
 * (per-collective setup/sync).
 *
 * Strategy-specific overlap:
 *  - FSDP: min(1, ratio) × (N-1)/N × η × dpOverlapCeiling
 *  - DDP/ZeRO: bucket-model with tail drain × η
 *  - TP/PP: C/(C+T) × η
 *
 * η (~0.96) captures CUDA stream scheduling imperfection, NCCL channel
 * contention, Python GIL overhead, and memory allocator pressure.
 */

// =========================================================================
// Physics-Based Overlap Constants
// =========================================================================

/** Scheduling efficiency: captures CUDA stream scheduling imperfection,
 *  NCCL channel contention, Python GIL overhead, memory allocator pressure.
 *  Single value for all strategies — per-strategy differences emerge from
 *  the timeline geometry (prefetching, bucketing, pipelining). */
export const SCHEDULING_EFFICIENCY = 0.96;

/** PyTorch DDP AllReduce bucket size in bytes. Framework constant. */
export const BUCKET_SIZE_BYTES = 25e6;

/** Per-collective fixed overhead for cross-node (IB) collectives in ms.
 *  Captures NCCL QP setup, sync barrier, buffer allocation. Dominates for
 *  small messages; amortized for large ones. NVLink collectives use 10%
 *  of this value (~5µs vs ~50µs). Single tunable. */
export const PER_COLLECTIVE_OVERHEAD_MS = 0.050;

// =========================================================================
// Protocol Overhead Constants
// =========================================================================

export const PROTOCOL_OVERHEAD = {
  /** TP within node (NVLink): ~5% NCCL protocol overhead. */
  tp_nvlink: 0.05,
  /** TP cross-node (IB): ~15% protocol overhead. */
  tp_crossnode: 0.15,
  /** FSDP/ZeRO-3 AllGather/ReduceScatter: moderate overhead. */
  dp_fsdp: 0.10,
  /** DDP/ZeRO-1/ZeRO-2 AllReduce: higher overhead for protocol, chunking, bucketing. */
  dp_ddp: 0.20,
  /** EP All-to-All: moderate overhead. */
  dp_ep: 0.10,
  /** PP P2P: low overhead. */
  pp: 0.05,
} as const;

export type ProtocolOverheadType = keyof typeof PROTOCOL_OVERHEAD;

// =========================================================================
// Scoped Override API (for testing without leaked global state)
// =========================================================================

export type ProtocolOverheadMap = { [K in ProtocolOverheadType]: number };

export interface OverlapConstants {
  schedulingEfficiency: number;
  bucketSizeBytes: number;
  perCollectiveOverheadMs: number;
  protocolOverhead: ProtocolOverheadMap;
}

const _defaults: OverlapConstants = {
  schedulingEfficiency: SCHEDULING_EFFICIENCY,
  bucketSizeBytes: BUCKET_SIZE_BYTES,
  perCollectiveOverheadMs: PER_COLLECTIVE_OVERHEAD_MS,
  protocolOverhead: { ...PROTOCOL_OVERHEAD },
};

let _current: OverlapConstants = { ..._defaults, protocolOverhead: { ...PROTOCOL_OVERHEAD } };

/**
 * Execute `fn` with temporary overlap constant overrides.
 * Restores original values even if `fn` throws.
 */
export function withOverlapOverrides<T>(
  overrides: Partial<OverlapConstants>,
  fn: () => T,
): T {
  const saved = { ..._current, protocolOverhead: { ..._current.protocolOverhead } };
  if (overrides.protocolOverhead) {
    Object.assign(_current.protocolOverhead, overrides.protocolOverhead);
  }
  const { protocolOverhead: _po, ...rest } = overrides;
  Object.assign(_current, rest);
  try {
    return fn();
  } finally {
    _current = saved;
  }
}

/** Read current overlap constants (for testing/introspection). */
export function getOverlapConstants(): Readonly<OverlapConstants> {
  return _current;
}

/** Set a single overlap constant (non-protocol) for sensitivity analysis. */
export function setOverlapConstant<K extends keyof Omit<OverlapConstants, 'protocolOverhead'>>(
  key: K, value: OverlapConstants[K]
): void {
  (_current as unknown as Record<string, unknown>)[key] = value;
}

/** Set a single protocol overhead entry for sensitivity analysis. */
export function setProtocolOverheadEntry(key: ProtocolOverheadType, value: number): void {
  _current.protocolOverhead[key] = value;
}

/** Reset all overlap constants to defaults. */
export function resetOverlapConstants(): void {
  _current = { ..._defaults, protocolOverhead: { ..._defaults.protocolOverhead } };
}

// =========================================================================
// Overlap Computation Functions (Physics-Based)
// =========================================================================

// =========================================================================
// Per-Layer FSDP Pipeline Overlap Model
// =========================================================================

/**
 * FSDP scheduling efficiency — independent from the global SCHEDULING_EFFICIENCY
 * (0.96) used by DDP/TP/PP/EP overlap models. These are physically distinct
 * mechanisms: FSDP uses CUDA stream prefetch pipelining, while DDP uses bucketed
 * AllReduce overlap.
 *
 * With backward prefetch (FSDP2 default): CUDA stream sync + memory allocator
 * latency for prefetched AllGather buffer → ~2-5% overhead.
 *
 * Without backward prefetch (FSDP1): manual scheduling, less overlap opportunity.
 */
const FSDP_ETA_PREFETCH = 0.95;
const FSDP_ETA_NO_PREFETCH = 0.80;

export interface FSDPPipelineInput {
  fwdComputePerLayer: number[];  // per-layer fwd compute (ms). length=1 → uniform
  bwdComputePerLayer: number[];  // per-layer bwd compute (ms). length=1 → uniform
  allGatherPerLayer: number;      // AG time per layer (ms), uniform across layers
  reduceScatterPerLayer: number;  // RS time per layer (ms), uniform across layers
  numLayers: number;              // total layers (allows single-element arrays for uniform)
  backwardPrefetch: boolean;      // FSDP2 default: true
}

/**
 * Per-layer FSDP pipeline overlap model.
 *
 * Returns **exposed comm per micro-batch in ms** (not a fraction). The caller
 * multiplies by GA linearly.
 *
 * Models the real FSDP2 per-layer pipeline where AllGather(layer N) overlaps
 * compute(layer N-1):
 *
 * Forward (layer 0 → L-1):
 *   Layer 0: AG is cold start — fully exposed
 *   Layer i>0: AG(i) overlaps with compute(i-1), degraded by η
 *   Exposed = AG + Σ_{i=1}^{L-1} max(0, AG - fwdCompute[i-1] × η)
 *
 * Backward (layer L-1 → 0):
 *   Layer L-1: AG is cold start — fully exposed
 *   Layer i<L-1: AG(i) overlaps with backward(i+1), degraded by η
 *   RS(i) overlaps with backward(i-1) for i>0; layer 0's RS is fully exposed
 *   Exposed = AG + RS
 *     + Σ_{i=0}^{L-2} max(0, AG - bwdCompute[i+1] × η)
 *     + Σ_{i=1}^{L-1} max(0, RS - bwdCompute[i-1] × η)
 *
 * Compute-bound regime (common case — large models): all max(0,...) terms
 * vanish → exposedPerMB = 2×AG + RS, independent of model depth.
 */
export function computeFSDPExposedComm(input: FSDPPipelineInput): number {
  const { allGatherPerLayer: AG, reduceScatterPerLayer: RS, numLayers: L, backwardPrefetch } = input;
  const η = backwardPrefetch ? FSDP_ETA_PREFETCH : FSDP_ETA_NO_PREFETCH;

  if (L <= 0) return 0;
  if (L === 1) return 2 * AG + RS;  // single layer: all cold start, no overlap

  const fwdUniform = input.fwdComputePerLayer.length === 1;
  const bwdUniform = input.bwdComputePerLayer.length === 1;

  // --- Forward pass: layer 0 → L-1 ---
  // Cold start: AG(0) fully exposed
  let fwdExposed = AG;

  if (fwdUniform) {
    // Closed-form: all layers have the same compute
    const excess = Math.max(0, AG - input.fwdComputePerLayer[0] * η);
    fwdExposed += (L - 1) * excess;
  } else {
    // Per-layer loop: AG(i) overlaps with compute(i-1)
    for (let i = 1; i < L; i++) {
      const compute_prev = input.fwdComputePerLayer[i - 1];
      fwdExposed += Math.max(0, AG - compute_prev * η);
    }
  }

  // --- Backward pass: layer L-1 → 0 ---
  // Cold start: AG(L-1) fully exposed
  let bwdExposed = AG;

  // AG overlap: AG(i) overlaps with backward(i+1) for i = 0..L-2
  if (bwdUniform) {
    const agExcess = Math.max(0, AG - input.bwdComputePerLayer[0] * η);
    bwdExposed += (L - 1) * agExcess;
  } else {
    for (let i = 0; i < L - 1; i++) {
      const compute_next = input.bwdComputePerLayer[i + 1];
      bwdExposed += Math.max(0, AG - compute_next * η);
    }
  }

  // RS overlap: RS(0) is fully exposed (cold start), RS(i>0) overlaps backward(i-1)
  bwdExposed += RS;  // RS(0) cold start

  if (bwdUniform) {
    const rsExcess = Math.max(0, RS - input.bwdComputePerLayer[0] * η);
    bwdExposed += (L - 1) * rsExcess;
  } else {
    for (let i = 1; i < L; i++) {
      const compute_prev = input.bwdComputePerLayer[i - 1];
      bwdExposed += Math.max(0, RS - compute_prev * η);
    }
  }

  return fwdExposed + bwdExposed;
}

export interface DDPOverlapInput {
  commTime: number;
  backwardTime: number;
  gradientBytes: number;
}

/**
 * DDP bucketed AllReduce overlap.
 *
 * PyTorch DDP fires AllReduce per bucket (25MB default). First bucket fires
 * after first few backward layers finish. The last bucket's AllReduce fires
 * when backward completes, so it has zero backward compute to overlap with
 * (tail drain). Only (N-1)/N of comm time is overlappable.
 *
 * Formula:
 *   overlappableFraction = (numBuckets - 1) / numBuckets
 *   theoreticalOverlap = overlappableFraction × min(1, backwardTime / commTime)
 *
 * Returns absolute overlap time (ms), not a fraction.
 */
export function computeDDPOverlap(input: DDPOverlapInput): number {
  const c = _current;
  const numBuckets = Math.max(1, Math.ceil(input.gradientBytes / c.bucketSizeBytes));
  const overlappableFraction = (numBuckets - 1) / numBuckets;
  const theoreticalOverlap = input.commTime > 0
    ? overlappableFraction * Math.min(1.0, input.backwardTime / input.commTime)
    : 1.0;
  return input.commTime * theoreticalOverlap * c.schedulingEfficiency;
}

export interface ZeROGradOverlapInput {
  stage: 1 | 2;
  overlapComm: boolean;
  gradSyncTime: number;
  backwardTime: number;
  gradientBytes: number;
}

/**
 * ZeRO-1/2 gradient sync overlap.
 *
 * Uses same bucket model as DDP (including tail drain) but only for the
 * grad sync portion. Param AllGather after optimizer is sequential (no overlap).
 */
export function computeZeROGradOverlap(input: ZeROGradOverlapInput): number {
  if (!input.overlapComm) return 0;
  const c = _current;
  const numBuckets = Math.max(1, Math.ceil(input.gradientBytes / c.bucketSizeBytes));
  const overlappableFraction = (numBuckets - 1) / numBuckets;
  const theoreticalOverlap = input.gradSyncTime > 0
    ? overlappableFraction * Math.min(1.0, input.backwardTime / input.gradSyncTime)
    : 1.0;
  return input.gradSyncTime * theoreticalOverlap * c.schedulingEfficiency;
}

export interface TPOverlapInput {
  computePerMB: number;
  tpCommWithOverhead: number;
}

/**
 * TP overlap efficiency.
 *
 * Formula: C/(C+T) × η
 *
 * No separate SP bonus — with SP enabled, comm volume changes (AllGather +
 * ReduceScatter vs AllReduce), and T reflects this. The scheduling
 * efficiency η captures the remaining CUDA/NCCL overhead.
 *
 * No per-strategy caps — η inherently limits overlap below 1.0.
 *
 * TODO: This models TP all-reduce as a monolithic cost. Modern frameworks decompose
 * it into ReduceScatter + AllGather, where AG overlaps with the next GEMM. For
 * compute-dominant configs (hidden/tp >> 1, i.e., all practical training), both
 * models converge to near-complete hiding. The monolithic model predicts ~4% exposed
 * (from the η term) vs <1% for per-layer RS/AG pipelining — a difference of <0.1pp
 * MFU for tier 1 anchors. Compare with computeFSDPExposedComm() which already
 * implements per-layer RS/AG decomposition for FSDP.
 */
export function computeTPOverlap(input: TPOverlapInput): number {
  const c = _current;
  return (input.computePerMB / (input.computePerMB + input.tpCommWithOverhead)) * c.schedulingEfficiency;
}

export interface PPOverlapInput {
  computePerMB: number;
  ppCommPerMB: number;
}

/**
 * PP overlap efficiency.
 *
 * Formula: C/(C+T) × η
 *
 * TODO: Cross-node PP with TP>1 can use scatter-gather (Megatron-LM) to achieve
 * tp × perNicBW effective bandwidth. The comm model in 3d-parallel.ts and
 * pipeline-parallel.ts uses single-P2P bandwidth. Impact <0.01pp MFU — PP comm
 * is dwarfed by compute for 175B+ models (C/(C+T) hides 95%+).
 */
export function computePPOverlap(input: PPOverlapInput): number {
  const c = _current;
  return (input.computePerMB / (input.computePerMB + input.ppCommPerMB)) * c.schedulingEfficiency;
}

/**
 * EP slack overlap: EP all-to-all can overlap with compute slack (idle cycles
 * when physics floor > actual compute). All-to-all is a barrier (dispatch must
 * complete before expert compute starts), so overlap is limited to the slack
 * window — not pipelined like DP ring-reduce. CUDA stream scheduling efficiency
 * is the same η that governs all comm/compute overlap.
 */
export function computeEPSlackOverlap(
  epCommTime: number,
  computeSlack: number,
): number {
  return Math.min(epCommTime, computeSlack * _current.schedulingEfficiency);
}

/**
 * Apply protocol overhead to raw communication time.
 * Two-term model: proportional (per-byte NCCL framing) + fixed (per-collective setup).
 *
 * Formula: raw × (1 + overhead[type]) + numCollectives × perCollectiveOverhead
 *
 * NVLink collectives have ~10× lower per-collective overhead than IB collectives
 * (SM-to-SM setup vs QP/routing setup). Caller passes intraNode to select tier.
 *
 * @param raw - Raw bandwidth-based communication time in ms
 * @param type - Protocol overhead type (determines proportional fraction)
 * @param numCollectives - Number of discrete collective operations (default 0)
 * @param intraNode - True if all participants are on the same node (NVLink), default false
 */
export function applyProtocolOverhead(
  raw: number,
  type: ProtocolOverheadType,
  numCollectives: number = 0,
  intraNode: boolean = false,
): number {
  if (raw === 0) return 0;  // No communication (dp=1) → no overhead
  const perColl = intraNode
    ? _current.perCollectiveOverheadMs * 0.1   // NVLink: ~5µs per collective
    : _current.perCollectiveOverheadMs;         // IB: ~50µs per collective
  return raw * (1 + _current.protocolOverhead[type]) + numCollectives * perColl;
}
