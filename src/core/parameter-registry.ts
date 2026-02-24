/**
 * Parameter Registry
 *
 * Machine-readable catalog of every empirical constant in the simulator.
 * Each entry documents the parameter's value, location, calibration tier,
 * physical mechanism, and provenance.
 *
 * Tier definitions:
 *   physics            -- Derived from first principles or hardware specs
 *   grounded-empirical -- Informed by published measurements, narrow range
 *   fitted             -- Tuned to match benchmark data, may drift
 *   redundant          -- Derived from other parameters (not independently tuned)
 *
 * This file is the single source of truth for auditing calibration constants.
 * Strategy files still contain the numeric literals (for locality and readability);
 * this registry exists for discoverability, versioning, and automated drift detection.
 */

// ---------------------------------------------------------------------------
// Interface
// ---------------------------------------------------------------------------

export interface ParameterEntry {
  name: string;
  value: number | string;
  file: string;
  category:
    | 'compute-efficiency'
    | 'dp-scaling'
    | 'scale-overhead'
    | 'overlap'
    | 'moe'
    | 'hardware'
    | 'memory'
    | 'inference'
    | 'backward-multiplier'
    | 'pp-comm';
  tier: 'physics' | 'grounded-empirical' | 'fitted' | 'redundant';
  mechanism: string;
  affectedBenchmarks: string[];
  introducedIn: string;
  lastValidatedAt: string;
  owner: string;
  evidence: string;
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

export const PARAMETER_REGISTRY: ParameterEntry[] = [

  // =========================================================================
  // Compute Efficiency (base.ts)
  // =========================================================================

  {
    name: 'runtimeResidual',
    value: 0.655,
    file: 'src/core/strategies/base.ts',
    category: 'compute-efficiency',
    tier: 'fitted',
    mechanism:
      'Dense runtime efficiency residual: kernel launch gaps, CUDA stream ' +
      'scheduling, autograd overhead, memory allocator pressure. Framework-level ' +
      'optimizations (kernel fusion, memory planning, communication scheduling) ' +
      'that the analytical model does not credit. ' +
      'Bounded [0,1] by construction. Validated range: 125M-671B.',
    affectedBenchmarks: ['all dense'],
    introducedIn: 'v4.0',
    lastValidatedAt: '2026-02-22',
    owner: 'calibration',
    evidence:
      'Dense median is -0.3pp across 7 Tier 1 anchors. ' +
      'Separate from MoE residual to center dense and MoE medians independently.',
  },
  {
    name: 'moeRuntimeResidual',
    value: 0.97,
    file: 'src/core/strategies/base.ts',
    category: 'compute-efficiency',
    tier: 'fitted',
    mechanism:
      'MoE runtime efficiency residual: captures per-layer MoE overhead from router ' +
      'backward computation, permutation backward (reverse scatter/gather), auxiliary ' +
      'load-balance loss gradient, and expert scheduling (grouped GEMM dispatch with ' +
      'variable batch sizes). Near 1.0 because most MoE overhead is explicitly modeled ' +
      'explicitly (expert GEMM efficiency, EP transport, load imbalance).',
    affectedBenchmarks: ['DeepSeek V3', 'Mixtral 8x22B', 'Qwen2 57B-A14B', 'MoE models'],
    introducedIn: 'v4.0',
    lastValidatedAt: '2026-02-22',
    owner: 'calibration',
    evidence:
      'MoE median is +0.9pp across 3 Tier 1 MoE anchors. ' +
      'Separate from dense residual; dense-MoE gap is ~1.5pp.',
  },
  {
    name: 'computeSaturationThreshold (Hopper)',
    value: 5_000_000,
    file: 'src/core/hardware/gpu.ts',
    category: 'compute-efficiency',
    tier: 'fitted',
    mechanism:
      'Compute-efficiency saturation threshold = matmul threshold (1M) × 4.77. ' +
      'Unified with kernel-level getMatmulSaturationFactor() — same physics (SM occupancy), ' +
      'one threshold table, one multiplier. Raised threshold gives 5-12% gradient across ' +
      'realistic training configs in the multiplicative efficiency model.',
    affectedBenchmarks: ['all'],
    introducedIn: 'v4.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence:
      'Diagnostic: power-0.3 curve at 5M yields implied residual CV=2.2%. ' +
      'Other architectures scaled proportionally (Ampere 0.75×, Blackwell 1.25×).',
  },
  // =========================================================================
  // DP Scaling (base.ts)
  // =========================================================================

  {
    name: 'DP_BW_REF',
    value: 64,
    file: 'src/core/strategies/base.ts',
    category: 'dp-scaling',
    tier: 'grounded-empirical',
    mechanism:
      'Reference DP group size for bandwidth degradation. Below 64 inter-node ' +
      'participants, topology is well-mapped (within-pod) and no penalty applies.',
    affectedBenchmarks: ['large-scale-dp'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'MegaScale NSDI 2024; NVIDIA DGX SuperPOD topology documentation',
  },
  {
    name: 'DP_BW_ALPHA',
    value: 0.15,
    file: 'src/core/strategies/base.ts',
    category: 'dp-scaling',
    tier: 'fitted',
    mechanism:
      'Log-scale bandwidth degradation rate for large DP groups. ' +
      'Penalty = 1 / (1 + 0.15 * log2(dp / ref)). Models multi-hop routing, ' +
      'NCCL ring/tree inefficiency, and fabric congestion at scale. ' +
      'Also used for EP BW degradation via getDPGroupSizePenalty(ep, { ref: 8 }). ' +
      'When EP all-to-all crosses nodes (ep*tp > gpusPerNode), DP penalty ' +
      'uses fabricParticipants = dp + ep to model shared IB fabric congestion.',
    affectedBenchmarks: ['large-scale-dp'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'MegaScale NSDI 2024 bandwidth measurements through DP=192; fabric congestion modeling',
  },
  {
    name: 'DP_BW_FLOOR',
    value: 0.40,
    file: 'src/core/strategies/base.ts',
    category: 'dp-scaling',
    tier: 'fitted',
    mechanism:
      'Minimum DP bandwidth fraction. MegaScale (NSDI \'24) shows continued degradation ' +
      'through DP=192 with no plateau. Floor binds at dp≈51000 — effectively never for ' +
      'real configs. Optimizer caps DP at 512, so extreme DP is excluded by construction.',
    affectedBenchmarks: ['large-scale-dp'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-23',
    owner: 'calibration',
    evidence: 'MegaScale NSDI 2024 bandwidth measurements through DP=192',
  },
  {
    name: 'DP_LATENCY_REF',
    value: 64,
    file: 'src/core/strategies/base.ts',
    category: 'dp-scaling',
    tier: 'grounded-empirical',
    mechanism:
      'Reference DP group size for per-collective latency overhead. Zero below ' +
      'dp=64 (within-pod, topology well-mapped).',
    affectedBenchmarks: ['large-scale-dp'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'NCCL tree latency profiling: ~log2(N) with ~3-5us per inter-node hop',
  },
  {
    name: 'DP_LATENCY_PER_LOG2_MS',
    value: 0.030,
    file: 'src/core/strategies/base.ts',
    category: 'dp-scaling',
    tier: 'grounded-empirical',
    mechanism:
      'Per-collective latency per log2 step above reference. 30us per log2 step ' +
      'is conservative-realistic (2x log2(N) hops + sync jitter).',
    affectedBenchmarks: ['large-scale-dp'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'NCCL tree latency profiling on IB fat-tree fabrics',
  },
  // =========================================================================
  // Scale Overhead (base.ts)
  // =========================================================================

  {
    name: 'STRAGGLER_SIGMA',
    value: 0.014,
    file: 'src/core/strategies/base.ts',
    category: 'scale-overhead',
    tier: 'grounded-empirical',
    mechanism:
      'Per-node step-time coefficient of variation. Order-statistics model: ' +
      'E[max(N)] ≈ μ + σ√(2·ln(N)). Measurable from production step-time variance.',
    affectedBenchmarks: ['large-scale-dp'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'Derived from order statistics; sigma fitted to MegaScale/Meta training curves',
  },

  // =========================================================================
  // Overlap Models (various files)
  // =========================================================================

  // Physics-based overlap model. Two-term protocol overhead: proportional
  // (per-byte NCCL framing) + fixed (per-collective setup/sync via
  // PER_COLLECTIVE_OVERHEAD_MS).
  {
    name: 'SCHEDULING_EFFICIENCY',
    value: 0.96,
    file: 'src/core/strategies/overlap.ts',
    category: 'overlap',
    tier: 'fitted',
    mechanism:
      'Scheduling efficiency (η): captures CUDA stream scheduling imperfection, ' +
      'NCCL channel contention, Python GIL overhead, memory allocator pressure. ' +
      'Single value for all strategies — per-strategy differences emerge from ' +
      'timeline geometry (prefetching, bucketing, pipelining).',
    affectedBenchmarks: ['all'],
    introducedIn: 'v2.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'Grid search across calibration benchmarks; H100/A100 profiling',
  },
  {
    name: 'BUCKET_SIZE_BYTES',
    value: 25e6,
    file: 'src/core/strategies/overlap.ts',
    category: 'overlap',
    tier: 'physics',
    mechanism:
      'PyTorch DDP AllReduce bucket size (25MB default). Used in DDP/ZeRO-1/2 ' +
      'bucketed overlap timeline model to compute numBuckets and firstBucketDelay.',
    affectedBenchmarks: ['ddp', 'zero-1', 'zero-2'],
    introducedIn: 'v2.0',
    lastValidatedAt: '2026-02-21',
    owner: 'framework',
    evidence: 'PyTorch DDP source code: torch/nn/parallel/distributed.py',
  },
  {
    name: 'PER_COLLECTIVE_OVERHEAD_MS',
    value: 0.050,
    file: 'src/core/strategies/overlap.ts',
    category: 'overlap',
    tier: 'fitted',
    mechanism:
      'Per-collective fixed overhead for cross-node (IB) collectives in ms. ' +
      'Captures NCCL QP setup, sync barrier, buffer allocation. Dominates for ' +
      'small messages; amortized for large ones. NVLink collectives use 10% ' +
      'of this value (~5µs vs ~50µs). Two-term model: raw × (1 + protocol) + ' +
      'numCollectives × perCollectiveOverhead.',
    affectedBenchmarks: ['all'],
    introducedIn: 'v2.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'NCCL QP setup profiling; grid search across calibration benchmarks',
  },
  {
    name: 'PROTOCOL_OVERHEAD (TP NVLink)',
    value: 0.05,
    file: 'src/core/strategies/overlap.ts',
    category: 'overlap',
    tier: 'grounded-empirical',
    mechanism:
      'NCCL protocol overhead for TP AllReduce within node (NVLink). ' +
      'Low overhead due to NVLink kernel-level optimizations.',
    affectedBenchmarks: ['all-3d'],
    introducedIn: 'v2.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'NCCL profiling on NVLink interconnects',
  },
  {
    name: 'PROTOCOL_OVERHEAD (DP FSDP)',
    value: 0.10,
    file: 'src/core/strategies/overlap.ts',
    category: 'overlap',
    tier: 'fitted',
    mechanism:
      'NCCL protocol overhead for FSDP/ZeRO-3 AllGather/ReduceScatter. ' +
      'Moderate overhead from per-layer collective scheduling.',
    affectedBenchmarks: ['llama-405b', 'deepseek-v3'],
    introducedIn: 'v2.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'FSDP profiling on multi-node clusters',
  },
  // =========================================================================
  // MoE (3d-parallel.ts)
  // =========================================================================

  {
    name: 'groupedGemmEfficiency.baseline',
    value: '1/1.05 ≈ 0.952',
    file: 'src/core/hardware/gpu.ts',
    category: 'moe',
    tier: 'grounded-empirical',
    mechanism:
      'Grouped GEMM efficiency baseline — applied to all MoE models at/above the ' +
      'expertGemmDimThreshold (1536). Subsumes the old GROUPED_GEMM_OVERHEAD = 0.05 ' +
      'constant: baseline = 1/1.05 ≈ 0.952 reproduces the same 5% penalty. Below ' +
      'threshold, a power-law penalty with expert count scaling applies instead.',
    affectedBenchmarks: ['deepseek-v3', 'mixtral-8x7b', 'mixtral-8x22b', 'qwen3-30b-a3b'],
    introducedIn: 'v10.0',
    lastValidatedAt: '2026-02-22',
    owner: 'calibration',
    evidence:
      'Continuity with old GROUPED_GEMM_OVERHEAD at threshold. SonicMoE (arXiv:2512.14080) ' +
      'grouped expert GEMM benchmarks: 49-57% of H100 BF16 peak for small expert dims.',
  },
  {
    name: 'expertGemmDimThreshold',
    value: 1536,
    file: 'src/core/hardware/gpu.ts',
    category: 'moe',
    tier: 'fitted',
    mechanism:
      'Architecture-dependent threshold for grouped GEMM efficiency penalty. Models with ' +
      'expertIntermediateSize >= threshold get baseline treatment only. Uses raw (not /tp) ' +
      'expert intermediate size — architectural property. Set to 1536 to exclude Qwen3 ' +
      '235B (expertInt=1536, published gap +3%) while penalizing Qwen3 30B (expertInt=768, ' +
      'published gap is +80%).',
    affectedBenchmarks: ['qwen3-30b-a3b', 'qwen3-235b-a22b'],
    introducedIn: 'v10.0',
    lastValidatedAt: '2026-02-22',
    owner: 'calibration',
    evidence:
      'Qwen3 30B (expertInt=768) needs penalty; Qwen3 235B (expertInt=1536) should not. ' +
      'DeepSeek V3 (expertInt=2048) and GPT-OSS (expertInt=2880) well above threshold.',
  },
  {
    name: 'EXPERT_GEMM_EXPONENT',
    value: 0.80,
    file: 'src/core/hardware/gpu.ts',
    category: 'moe',
    tier: 'fitted',
    mechanism:
      'Power-law exponent for grouped GEMM efficiency degradation below threshold. ' +
      'factor = baseline × (dimRatio)^exponent. Higher exponent = steeper penalty curve. ' +
      'Perturbable parameter with get/set pair.',
    affectedBenchmarks: ['qwen3-30b-a3b'],
    introducedIn: 'v10.0',
    lastValidatedAt: '2026-02-22',
    owner: 'calibration',
    evidence:
      'Sensitivity sweep: (0.6, cs=0.78), (0.7, cs=0.65), (0.8, cs=0.55) all hit ' +
      'Qwen3 30B target of 241 HFU TFLOPS. Selected 0.80 for steeper physical curve. ' +
      'Single-point fitted (Qwen3 30B at 241 HFU TFLOPS).',
  },
  {
    name: 'EXPERT_COUNT_SCALE',
    value: 0.55,
    file: 'src/core/hardware/gpu.ts',
    category: 'moe',
    tier: 'fitted',
    mechanism:
      'Expert count scaling factor for grouped GEMM efficiency. When expertsPerGPU > 8, ' +
      'countPenalty = 1 + countScale × log2(expertsPerGPU/8). More experts per GPU → ' +
      'more L2 cache pressure from weight matrix thrashing. Perturbable parameter.',
    affectedBenchmarks: ['qwen3-30b-a3b'],
    introducedIn: 'v10.0',
    lastValidatedAt: '2026-02-22',
    owner: 'calibration',
    evidence:
      'Co-fitted with EXPERT_GEMM_EXPONENT. Qwen3 30B has 16 experts/GPU (EP=8, 128 total), ' +
      'total weight working set ~150MB vs H100 L2 50MB → significant cache thrashing. ' +
      'Single-point fitted (Qwen3 30B at 241 HFU TFLOPS).',
  },
  {
    name: 'baseAlltoallEfficiency (NVLink)',
    value: 0.55,
    file: 'src/core/strategies/3d-parallel.ts',
    category: 'moe',
    tier: 'grounded-empirical',
    mechanism:
      'NCCL All-to-All bandwidth efficiency on NVLink (intra-node EP). ' +
      'All-to-All achieves lower BW than point-to-point due to multi-destination ' +
      'traffic and NCCL scheduling overhead.',
    affectedBenchmarks: ['intra-node-ep'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'NCCL All-to-All profiling on NVLink: 50-60% of peak',
  },
  {
    name: 'baseAlltoallEfficiency (IB)',
    value: 0.45,
    file: 'src/core/strategies/3d-parallel.ts',
    category: 'moe',
    tier: 'grounded-empirical',
    mechanism:
      'NCCL All-to-All bandwidth efficiency on InfiniBand (cross-node EP). ' +
      'Multi-hop routing and fabric congestion further reduce efficiency.',
    affectedBenchmarks: ['cross-node-ep'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'NCCL All-to-All profiling on IB: 40-50% of peak',
  },
  {
    name: 'alltoallLatencyMs (NVLink)',
    value: 0.05,
    file: 'src/core/strategies/3d-parallel.ts',
    category: 'moe',
    tier: 'physics',
    mechanism:
      'Per-collective All-to-All latency on NVLink (~50us). NCCL synchronization ' +
      'plus kernel launch overhead for intra-node EP.',
    affectedBenchmarks: ['intra-node-ep'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-21',
    owner: 'physics',
    evidence: 'NCCL latency measurements on NVLink 4.0 (DGX H100)',
  },
  {
    name: 'alltoallLatencyMs (IB)',
    value: 0.1,
    file: 'src/core/strategies/3d-parallel.ts',
    category: 'moe',
    tier: 'physics',
    mechanism:
      'Per-collective All-to-All latency on InfiniBand (~100us). Higher due to ' +
      'PCIe traversal and IB fabric hop latency.',
    affectedBenchmarks: ['cross-node-ep'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-21',
    owner: 'physics',
    evidence: 'NCCL latency measurements on IB HDR/NDR fabric',
  },
  {
    name: 'MOE_CAPACITY_FACTOR',
    value: 1.15,
    file: 'src/core/strategies/3d-parallel.ts',
    category: 'moe',
    tier: 'grounded-empirical',
    mechanism:
      'Buffer capacity factor for MoE dispatch/combine. Provides 15% headroom ' +
      'for imperfect router load balancing (range 1.0-1.25 in practice).',
    affectedBenchmarks: ['deepseek-v3', 'mixtral-8x7b'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'Switch Transformer; Mixtral; standard MoE framework defaults',
  },

  // =========================================================================
  // Backward Multipliers (various files)
  // =========================================================================

  {
    name: 'backward (full ckpt)',
    value: 2.85,
    file: 'src/core/strategies/3d-parallel.ts',
    category: 'backward-multiplier',
    tier: 'physics',
    mechanism:
      'Backward pass time = 2.85x forward with full activation checkpointing. ' +
      'Recompute cost is physically identical regardless of DP type (FSDP/DDP/ZeRO). ' +
      'FSDP AllGather/backward overlap is credited solely via computeFSDPExposedComm() — ' +
      'using a reduced multiplier would double-count the same overlap window.',
    affectedBenchmarks: ['all-ckpt'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-21',
    owner: 'physics',
    evidence: 'Double-count fix: overlap credited in computeFSDPExposedComm(), not backward multiplier',
  },
  // All strategies use 2.85 uniformly for full AC; FSDP overlap credited via computeFSDPExposedComm().
  {
    name: 'backward (no ckpt)',
    value: 2.0,
    file: 'src/core/strategies/base.ts',
    category: 'backward-multiplier',
    tier: 'physics',
    mechanism:
      'Backward pass time = 2x forward without activation checkpointing. ' +
      'Standard backward: activation gradient (1x) + weight gradient (1x) = 2x.',
    affectedBenchmarks: ['all-no-ckpt'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-21',
    owner: 'physics',
    evidence: 'First-principles: backward = dL/dx + dL/dW, each ~1x forward FLOPs',
  },
  {
    name: 'backward (selective)',
    value: '2.0 + f',
    file: 'src/core/strategies/3d-parallel.ts',
    category: 'backward-multiplier',
    tier: 'physics',
    mechanism:
      'Backward with selective activation checkpointing: 2.0 + f, where f = ' +
      'selectiveRecomputeFraction (model-dependent, typically 13-33%). Only ' +
      'attention linear projections (Q/K/V/O) are recomputed from the saved ' +
      'LN1 input. MLP activations are always kept.',
    affectedBenchmarks: ['selective-ckpt'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-21',
    owner: 'physics',
    evidence: 'Selective recompute analysis per Korthikanti et al. 2022',
  },
  {
    name: 'backward (selective, stored-layers blend)',
    value: '(K/N)(2.0+f) + ((N-K)/N)(2.85)',
    file: 'src/core/strategies/base.ts',
    category: 'backward-multiplier',
    tier: 'physics',
    mechanism:
      'Blended backward multiplier when selective AC uses stored-layers K < N. ' +
      'Layers 1..K use selective recompute (2.0 + f), layers K+1..N use full ' +
      'recompute (2.85). The overall multiplier is the weighted blend. At K=N, ' +
      'equals standard selective (2.0 + f). At K=0, equals full AC (2.85). ' +
      'Matches OLMo-core budget mode and Megatron-LM --recompute-num-layers.',
    affectedBenchmarks: ['olmo3-32b', 'selective-ckpt-budget'],
    introducedIn: 'v11.0',
    lastValidatedAt: '2026-02-23',
    owner: 'physics',
    evidence:
      'First-principles: proportional blend of selective and full recompute costs. ' +
      'Validated against OLMo 3 32B (43.4% MFU at K=35/64, published ~41%).',
  },

  // =========================================================================
  // Hardware / Roofline (base.ts, gpu.ts)
  // =========================================================================

  {
    name: 'ELEMENTWISE_BW_EFFICIENCY',
    value: 0.65,
    file: 'src/core/strategies/base.ts',
    category: 'hardware',
    tier: 'grounded-empirical',
    mechanism:
      'Effective memory bandwidth utilization for elementwise/non-matmul kernels ' +
      '(LayerNorm, SiLU, residual adds). ~65% of peak HBM bandwidth due to ' +
      'kernel launch overhead and non-coalesced access patterns.',
    affectedBenchmarks: ['all'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'NVIDIA A100/H100 elementwise kernel profiling',
  },
  {
    name: 'KERNEL_LAUNCH_OVERHEAD_MS',
    value: 0.04,
    file: 'src/core/strategies/base.ts',
    category: 'hardware',
    tier: 'physics',
    mechanism:
      'Per-layer kernel dispatch + autograd bookkeeping overhead (~40us per layer). ' +
      'CUDA kernel launch latency + PyTorch autograd graph construction.',
    affectedBenchmarks: ['all'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-21',
    owner: 'physics',
    evidence: 'CUDA kernel launch profiling; PyTorch autograd overhead measurements',
  },
  {
    name: 'BW_EFFICIENCY (optimizer)',
    value: 0.75,
    file: 'src/core/strategies/base.ts',
    category: 'hardware',
    tier: 'grounded-empirical',
    mechanism:
      'Memory bandwidth efficiency for optimizer step (AdamW streaming through ' +
      'HBM: grads + m + v + master + params). 75% due to sequential read/write ' +
      'pattern with good coalescing.',
    affectedBenchmarks: ['all'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'AdamW kernel profiling on H100 SXM; near-sequential streaming pattern',
  },

  // =========================================================================
  // Memory (base.ts)
  // =========================================================================

  {
    name: 'baseReserved',
    value: 1.0e9,
    file: 'src/core/strategies/base.ts',
    category: 'memory',
    tier: 'grounded-empirical',
    mechanism:
      'Base reserved GPU memory for CUDA context, cuDNN workspace, and framework ' +
      'overhead (1.0 GB). This is approximately constant across GPU types.',
    affectedBenchmarks: ['all'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'PyTorch + CUDA context memory profiling on A100/H100',
  },
  {
    name: 'fragmentationReserve',
    value: 0.07,
    file: 'src/core/strategies/base.ts',
    category: 'memory',
    tier: 'grounded-empirical',
    mechanism:
      '7% of GPU memory reserved for PyTorch caching allocator fragmentation, ' +
      'variable tensor sizes, and gradient accumulation patterns.',
    affectedBenchmarks: ['all'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'PyTorch caching allocator fragmentation measurements on A100/H100',
  },
  {
    name: 'STORED_LAYERS_CAPACITY_FRACTION',
    value: 0.95,
    file: 'src/core/strategies/base.ts',
    category: 'memory',
    tier: 'grounded-empirical',
    mechanism:
      'Auto-resolve stored-layers targets 95% of GPU capacity — leaves 5% headroom ' +
      'for runtime allocations (NCCL workspace, CUDA graphs, peak transient buffers) ' +
      'that the analytical model does not track. Matches OLMo-core/Megatron-LM practice ' +
      'of leaving a memory margin for selective AC budget mode.',
    affectedBenchmarks: ['olmo3-32b', 'selective-ckpt-budget'],
    introducedIn: 'v12.0',
    lastValidatedAt: '2026-02-23',
    owner: 'calibration',
    evidence:
      'OLMo-core defaults to ~95% memory utilization for selective AC budget mode. ' +
      'Megatron-LM recommends similar headroom for --recompute-num-layers.',
  },
  {
    name: 'bucketSizeMB',
    value: 25,
    file: 'src/core/strategies/base.ts',
    category: 'memory',
    tier: 'grounded-empirical',
    mechanism:
      'Default gradient bucketing buffer size (25 MB). Matches PyTorch DDP default ' +
      'bucket_cap_mb=25. Used for temporary communication buffer sizing.',
    affectedBenchmarks: ['ddp', 'zero-1', 'zero-2'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'PyTorch DDP default configuration; NCCL bucket size best practices',
  },

  // =========================================================================
  // PP Communication
  // =========================================================================

  {
    name: 'PP comm overhead',
    value: 1.05,
    file: 'src/core/strategies/pipeline-parallel.ts',
    category: 'pp-comm',
    tier: 'grounded-empirical',
    mechanism:
      'PP point-to-point communication overhead factor (1.05x). 5% overhead for ' +
      'NCCL send/recv protocol, serialization, and CUDA stream synchronization.',
    affectedBenchmarks: ['pp-benchmarks'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'NCCL P2P profiling on NVLink and IB',
  },
  {
    name: 'PP overlap cap',
    value: 0.95,
    file: 'src/core/strategies/3d-parallel.ts',
    category: 'pp-comm',
    tier: 'grounded-empirical',
    mechanism:
      'Maximum PP comm/compute overlap in 3D parallel. P2P activation transfers ' +
      'overlap well with compute in steady state, capped at 95%.',
    affectedBenchmarks: ['all-3d'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'Megatron-LM PP profiling; 1F1B steady-state comm overlap',
  },
  {
    name: 'PP_STAGE_TRANSITION_MS',
    value: 0.020,
    file: 'src/core/strategies/3d-parallel.ts',
    category: 'pp-comm',
    tier: 'grounded-empirical',
    mechanism:
      'Per-virtual-stage transition latency: NCCL P2P setup + stream sync + kernel launch. ' +
      'Each PP boundary × each virtual stage direction incurs ~20µs of non-overlappable latency. ' +
      'Interleaved-1F1B: 2×v×(pp-1) serial transitions/MB. DualPipe-V: 2×(pp-1) (bidirectional overlap).',
    affectedBenchmarks: ['all-3d'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-22',
    owner: 'calibration',
    evidence: 'Estimated from NCCL P2P profiling (NVLink H100). Material at v≥4, negligible at v≤2.',
  },

  // =========================================================================
  // Comm Overhead Multipliers
  // =========================================================================

  {
    name: 'TP overhead (NVLink)',
    value: 1.05,
    file: 'src/core/strategies/3d-parallel.ts',
    category: 'pp-comm',
    tier: 'grounded-empirical',
    mechanism:
      'TP AllReduce overhead on NVLink: 5% for NCCL protocol framing, ' +
      'ring/tree coordination, and CUDA stream sync.',
    affectedBenchmarks: ['all-3d'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'NCCL AllReduce profiling on NVLink 4.0',
  },
  {
    name: 'TP overhead (cross-node)',
    value: 1.15,
    file: 'src/core/strategies/3d-parallel.ts',
    category: 'pp-comm',
    tier: 'grounded-empirical',
    mechanism:
      'TP AllReduce overhead on IB (cross-node TP): 15% for hierarchical ' +
      'AllReduce protocol, PCIe traversal, and IB fabric routing.',
    affectedBenchmarks: ['cross-node-tp'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'NCCL hierarchical AllReduce profiling on IB HDR',
  },
  {
    name: 'DP overhead (FSDP/ZeRO-3)',
    value: 1.10,
    file: 'src/core/strategies/3d-parallel.ts',
    category: 'pp-comm',
    tier: 'grounded-empirical',
    mechanism:
      'FSDP/ZeRO-3 AllGather+ReduceScatter overhead: 10% for complex collective ' +
      'patterns (per-layer AllGather + ReduceScatter).',
    affectedBenchmarks: ['all-fsdp', 'all-zero-3'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'FSDP profiling; NCCL AllGather/ReduceScatter overhead measurements',
  },
  {
    name: 'DP overhead (ZeRO-1/2)',
    value: 1.20,
    file: 'src/core/strategies/ddp.ts',
    category: 'pp-comm',
    tier: 'grounded-empirical',
    mechanism:
      'DDP/ZeRO-1/2 AllReduce overhead: 20% for protocol, chunking, bucketing, ' +
      'and NCCL tree/ring coordination overhead.',
    affectedBenchmarks: ['ddp', 'zero-1', 'zero-2'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'PyTorch DDP AllReduce profiling; DeepSpeed ZeRO overhead analysis',
  },

  // =========================================================================
  // GBS Convergence Ceiling (exploration.ts / optimizer.ts)
  // =========================================================================

  {
    name: 'B_CRIT_COEFFICIENT',
    value: 20,
    file: 'src/core/simulation/exploration.ts',
    category: 'dp-scaling',
    tier: 'grounded-empirical',
    mechanism:
      'Critical batch size scaling coefficient: B_crit = coeff × sqrt(totalParams). ' +
      'From McCandlish et al. (2018) "An Empirical Model of Large-Batch Training". ' +
      'Beyond B_crit, gradient noise dominates and larger batches yield diminishing ' +
      'convergence benefit. Used to cap GBS inflation in the auto-optimizer.',
    affectedBenchmarks: ['all'],
    introducedIn: 'v7.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'McCandlish et al. 2018; validated against published GBS for LLaMA, GPT-3, DeepSeek V3',
  },
  {
    name: 'B_CRIT_FLOOR_TOKENS',
    value: 4_000_000,
    file: 'src/core/simulation/exploration.ts',
    category: 'dp-scaling',
    tier: 'grounded-empirical',
    mechanism:
      'Minimum critical batch size floor (4M tokens). Prevents unreasonably low ' +
      'GBS ceilings for small models where sqrt scaling would underestimate ' +
      'efficient batch sizes.',
    affectedBenchmarks: ['all'],
    introducedIn: 'v7.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'Small-model training practice: 7B–32B models commonly train at GBS 512–1024 with seq=4096',
  },

  // =========================================================================
  // Inference
  // =========================================================================

  {
    name: 'PREFILL_RESIDUAL',
    value: 0.40,
    file: 'src/core/inference/latency.ts',
    category: 'inference',
    tier: 'fitted',
    mechanism:
      'Prefill MFU residual after GPU architecture effects. Effective prefill MFU = ' +
      'getMemoryBandwidthScaling(gpu, "bf16") × PREFILL_RESIDUAL. H100 SXM bf16 is ' +
      'the reference (memBWScaling = 1.0 → effective MFU = 0.40). Used in estimateTTFT, ' +
      'estimateTPOT compute floor, and speculative decoding verification.',
    affectedBenchmarks: ['inference-latency'],
    introducedIn: 'v6.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'Inference serving benchmarks on H100; vLLM and TensorRT-LLM prefill profiling',
  },
  {
    name: 'DECODE_SAMPLING_OVERHEAD_MS',
    value: 0.03,
    file: 'src/core/inference/latency.ts',
    category: 'inference',
    tier: 'grounded-empirical',
    mechanism:
      'Fixed per-decode-step overhead for argmax/top-k sampling (0.03ms). ' +
      'Added to estimateTPOT after max(memoryTime, computeTime).',
    affectedBenchmarks: ['inference-latency'],
    introducedIn: 'v6.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'Sampling kernel profiling on H100; negligible but nonzero overhead',
  },

  // =========================================================================
  // Context Parallel (3d-parallel.ts)
  // =========================================================================

  {
    name: 'CP ring jitter floor',
    value: '0.03 + 0.005*(cp-1)',
    file: 'src/core/strategies/3d-parallel.ts',
    category: 'overlap',
    tier: 'grounded-empirical',
    mechanism:
      'Minimum exposed transfer fraction for ring attention. Scheduling jitter ' +
      'scales with CP degree: more peers = more coordination overhead. ' +
      'CP=2 (robust overlap): 3.5%. CP=16 (marginal, fragile): 10.5%.',
    affectedBenchmarks: ['ring-attention'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'Ring attention profiling with variable CP degrees; zigzag scheduling analysis',
  },
  {
    name: 'CP ring BW efficiency (NVLink)',
    value: 0.90,
    file: 'src/core/strategies/3d-parallel.ts',
    category: 'overlap',
    tier: 'grounded-empirical',
    mechanism:
      'P2P bandwidth efficiency for CP ring KV exchange on NVLink. P2P achieves ' +
      'higher BW than collectives (simpler protocol, no coordination).',
    affectedBenchmarks: ['ring-attention'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'NCCL P2P profiling on NVLink 4.0',
  },
  {
    name: 'CP ring BW efficiency (IB)',
    value: 0.85,
    file: 'src/core/strategies/3d-parallel.ts',
    category: 'overlap',
    tier: 'grounded-empirical',
    mechanism:
      'P2P bandwidth efficiency for CP ring KV exchange on InfiniBand. Lower ' +
      'than NVLink due to PCIe traversal and IB fabric latency.',
    affectedBenchmarks: ['ring-attention'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'NCCL P2P profiling on IB HDR/NDR',
  },
  {
    name: 'CP all-gather overlap (interleaved)',
    value: 0.15,
    file: 'src/core/strategies/3d-parallel.ts',
    category: 'overlap',
    tier: 'fitted',
    mechanism:
      'Fraction of CP AllGather KV hidden by non-attention compute from previous ' +
      'microbatch when using interleaved PP or DualPipeV. Interleaving creates ' +
      'compute slack from layer-norm and projection overlap.',
    affectedBenchmarks: ['cp-all-gather'],
    introducedIn: 'v1.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence: 'Megatron-LM CP profiling with interleaved 1F1B',
  },

  // =========================================================================
  // Derived Alternatives
  // =========================================================================

  {
    name: 'loadImbalanceFactor (derived)',
    value: '1 + σ√(2 ln E)',
    file: 'src/core/physics/derived.ts',
    category: 'moe',
    tier: 'physics',
    mechanism:
      'Multinomial order statistics for MoE expert load imbalance. Active in ' +
      '3d-parallel.ts with LOAD_BALANCE_DAMPING applied. Raw prediction is ' +
      'conservative (overestimates); damping corrects for aux-loss routing.',
    affectedBenchmarks: ['deepseek-v3', 'mixtral-8x7b', 'mixtral-8x22b'],
    introducedIn: 'v8.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence:
      'Multinomial extreme value theory: E[max of E iid Binomial draws]. ' +
      'Caller must EP-adjust token count.',
  },
  {
    name: 'cpCausalWorkDistribution (derived)',
    value: '{ diagonal: 0.5, diagonalSteps: 1, normalSteps: CP-2 }',
    file: 'src/core/strategies/3d-parallel.ts',
    category: 'overlap',
    tier: 'physics',
    mechanism:
      'Per-step compute model for ring attention causal masking. Returns structural ' +
      'description (not scalar): 1 diagonal step at 0.5× compute + (CP-2) normal ' +
      'steps at 1× compute. Used in ring attention exposed comm calculation.',
    affectedBenchmarks: ['ring-attention'],
    introducedIn: 'v8.0',
    lastValidatedAt: '2026-02-21',
    owner: 'calibration',
    evidence:
      'Ring attention geometry: CP-1 steps, 1 has triangular self-attention (half compute).',
  },
  {
    name: 'reservedMemoryDecomposition (derived)',
    value: 'cudaContext + fragmentation (7% of GiB capacity)',
    file: 'src/core/physics/derived.ts',
    category: 'memory',
    tier: 'physics',
    mechanism:
      'Two-component reserved memory model: CUDA context (arch-dependent 0.8–1.4 GB) ' +
      '+ PyTorch caching allocator fragmentation (7% of GiB capacity). GiB/GB gap ' +
      'is handled by gpuCapacityBytes (gpuMemoryGB × 1024³) in the capacity model.',
    affectedBenchmarks: ['all'],
    introducedIn: 'v8.0',
    lastValidatedAt: '2026-02-22',
    owner: 'calibration',
    evidence:
      'CUDA context estimated from torch.cuda.mem_get_info() deltas. 7% fragmentation ' +
      'calibrated against published configs (GPT-3 91.3%, LLaMA 405B 94.9% utilization).',
  },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Filter registry entries by category.
 */
export function getParametersByCategory(
  category: ParameterEntry['category'],
): ParameterEntry[] {
  return PARAMETER_REGISTRY.filter((p) => p.category === category);
}

/**
 * Look up a single entry by name (first match).
 */
export function getParameterByName(name: string): ParameterEntry | undefined {
  return PARAMETER_REGISTRY.find((p) => p.name === name);
}
