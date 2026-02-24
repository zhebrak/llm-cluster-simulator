# Parallelism Strategies Reference

> **See also:** [PHYSICS.md](PHYSICS.md) (formulas) | [MODELS.md](MODELS.md) (architectures) | [HARDWARE.md](HARDWARE.md) (interconnects)

## Table of Contents
1. [Strategy Architecture](#1-strategy-architecture)
2. [DDP (Distributed Data Parallel)](#2-ddp-distributed-data-parallel)
3. [ZeRO-1 (Optimizer Sharding)](#3-zero-1-optimizer-sharding)
4. [FSDP / ZeRO-3 (Full Sharding)](#4-fsdp--zero-3-full-sharding)
5. [Tensor Parallel](#5-tensor-parallel)
6. [Sequence Parallel](#6-sequence-parallel)
7. [Pipeline Parallel](#7-pipeline-parallel)
8. [3D Parallel (Unified Engine)](#8-3d-parallel-unified-engine)
9. [LoRA/QLoRA Fine-tuning](#9-loraqolra-fine-tuning)
10. [Selective Activation Checkpointing](#10-selective-activation-checkpointing)

---

## 1. Strategy Architecture

### Base Class

All strategies inherit from `ParallelismStrategy` (defined in `src/core/strategies/base.ts`),
which declares five abstract methods:

- `computeMemoryPerGPU(ctx)` -- per-GPU memory breakdown (params, grads, optimizer, activations,
  temporary buffers, reserved)
- `computeCommunication(ctx)` -- per-step byte volumes by dimension (DP, TP, PP, EP, CP)
- `computeTiming(ctx)` -- wall-clock breakdown (forward, backward, optimizer, comm, overlap)
- `validate(ctx)` -- hard errors and soft warnings
- `generateEvents(ctx)` -- timeline events for visualization

The base class provides `computeAnalysis()`, which calls the first four methods and derives:

- **MFU** (PaLM definition): `6 * activeParams * totalTokens / (stepTime * totalGPU_TFLOPS)`.
  Always measured against BF16 peak, even when training in FP8/FP4. Never includes recompute.
- **HFU**: `(6+2f) * activeParams * totalTokens / (stepTime * totalGPU_TFLOPS)`,
  where f = recomputeFraction (0 = no AC, 0.13–0.22 = selective, 1.0 = full).
  LoRA: `(4+2f)`.
- **Scale overhead**: Order-statistics straggler model. BSP training waits for the slowest node:
  overhead = σ√(2·ln(N)) where σ=0.014 per-node CV.

### Strategy Routing

All combined strategies (fsdp-tp, zero1-tp, ddp-tp-pp, zero1-tp-pp, fsdp-tp-pp) instantiate
`ThreeDParallelStrategy` internally with different `dpType` values. The eight strategies exposed
in the UI are:

| UI Name       | Internal class             | dpType   | Dimensions       |
|---------------|----------------------------|----------|------------------|
| DDP           | `DDPStrategy`              | -        | DP               |
| ZeRO-1        | `ZeROStrategy(stage=1)`    | -        | DP               |
| FSDP          | `FSDPStrategy`             | -        | DP               |
| FSDP + TP     | `ThreeDParallelStrategy`   | fsdp     | TP x DP          |
| ZeRO-1 + TP   | `ThreeDParallelStrategy`   | zero-1   | TP x DP          |
| DDP + TP + PP  | `ThreeDParallelStrategy`   | ddp      | TP x PP x DP     |
| ZeRO-1+TP+PP  | `ThreeDParallelStrategy`   | zero-1   | TP x PP x DP     |
| FSDP + TP + PP | `ThreeDParallelStrategy`   | fsdp     | TP x PP x DP     |

ZeRO-3 is equivalent to FSDP and shares the same engine implementation. Sequence
parallelism is controlled by a checkbox (defaults ON) on all 2D/3D strategies.

**Source files:**
- `src/core/strategies/base.ts` -- abstract class, MFU/HFU, shared helpers
- `src/core/strategies/index.ts` -- strategy registry and factory
- `src/core/strategies/presets.ts` -- preset configurations

### Compute Efficiency Model

All strategies share a common GPU kernel utilization formula (`computeComputeEfficiency()`):

```
efficiency = saturation × memBWScaling × runtimeResidual
```

- **Saturation**: `getComputeSaturationFactor()` — power-0.3 curve based on per-GPU GEMM dimensions
- **memBWScaling**: `getMemoryBandwidthScaling()` — Amdahl model for non-matmul ops (dtype-dependent)
- **runtimeResidual**: 0.655 — single fitted parameter (kernel launch gaps, CUDA stream scheduling, autograd overhead)
- **MoE dispatch overhead**: grouped GEMM overhead (5%) + EP coordination (AllToAll coordination overhead for cross-node EP)

Bounded [0,1] by construction. Communication costs are modeled explicitly and never double-counted with efficiency.

See [PHYSICS.md §2](PHYSICS.md#2-compute-efficiency-model) for full derivation and parameter values.

### DP Scaling Functions

Two independent degradation mechanisms for large DP groups (all with threshold at 64):

1. **`getDPGroupSizePenalty(dp)`** -- bandwidth degradation from fabric congestion.
   Returns 1.0 for dp <= 64, log degradation above. Second-tier penalty for dp > 256
   (multi-rail saturation, ECMP collisions). Floor at 0.40.
2. **`getDPCollectiveLatencyMs(dp)`** -- per-collective latency from NCCL tree depth.
   Returns 0 for dp <= 64, ~30us per log2 step above.

**DP group size input:**
- Standalone strategies (DDP, FSDP, ZeRO): `numNodes` -- NCCL hierarchical
  collectives run intra-node rings first, then inter-node tree. Fabric
  congestion scales with inter-node hops, not total GPUs.
- 3D strategies: `dp` directly (DP degree ~ numNodes when TP fills the node).

See [PHYSICS.md §5](PHYSICS.md#5-dp-scaling-laws) for penalty curves and thresholds.

---

## 2. DDP (Distributed Data Parallel)

Full model replica on every GPU. Gradients synchronized via AllReduce after backward.

### Memory

Each GPU holds a complete copy of all model states:

| Component       | Formula                    | Notes                          |
|-----------------|----------------------------|--------------------------------|
| Parameters      | P x 2 bytes                | BF16 weights                   |
| Gradients       | P x 4 bytes                | FP32 gradient accumulation     |
| Optimizer       | P x 12 bytes               | AdamW: master(4) + m1(4) + m2(4) |
| **Total/param** | **18 bytes**               |                                |

### Communication

- `2 * modelSize * gradDtype * (dp-1)/dp` -- ring AllReduce of gradients
- Once per optimizer step (not per micro-batch)

### Timing

- Overlap: Physics-based bucket model. PyTorch DDP fires AllReduce per 25MB bucket;
  first bucket fires after `1/numBuckets` of backward completes. Overlap efficiency =
  `min(1, backwardTime × (1 - 1/numBuckets) / commTime) × η` where η=0.96.
- Backward multiplier: 2.85x (full ckpt), 2.0+f (selective, f=model-dependent), 2.0x (none)
- Comm overhead: two-term model — 1.20× proportional + 50µs/bucket (IB) or 5µs/bucket (NVLink)
- DP scaling: uses `numNodes` as group size (NCCL hierarchical collectives
  scale with inter-node hops, not total GPUs)

**Source:** `src/core/strategies/ddp.ts`

---

## 3. ZeRO-1 (Optimizer Sharding)

Partitions only optimizer states across DP ranks. Parameters and gradients remain replicated.

### Memory

| Component       | Formula                    |
|-----------------|----------------------------|
| Parameters      | P x 2 bytes                |
| Gradients       | P x 4 bytes                |
| Optimizer       | P x 12 / dp bytes          |
| **Total/param** | **6 + 12/dp bytes**        |

### Communication

Two phases per step:
1. **Gradient AllReduce**: `2 * gradBytes * (N-1)/N` -- overlaps with backward (bucket model × η)
2. **Param AllGather after optimizer**: `paramBytes * (N-1)/N` -- sequential (no overlap)

### Timing

- Backward multiplier: 2.85x (full ckpt), 2.0+f (selective, f=model-dependent), 2.0x (none)
- Comm overhead: two-term model — 1.20× proportional + 50µs/bucket (IB) or 5µs/bucket (NVLink)
- Only gradient sync overlaps with backward; param AllGather is fully sequential

**Source:** `src/core/strategies/zero.ts`

---

## 4. FSDP / ZeRO-3 (Full Sharding)

Fully shards parameters, gradients, and optimizer states across all DP ranks. Parameters are
gathered on-demand per layer and released after use.

### Memory

| Component       | Formula                    |
|-----------------|----------------------------|
| Parameters      | P x 2 / dp bytes           |
| Gradients       | P x 4 / dp bytes           |
| Optimizer       | P x 12 / dp bytes          |
| **Total/param** | **18/dp bytes**            |

Additional per-GPU overhead:
- Gathered parameter buffer: 1 layer's params (gathered from DP peers before compute)
- Gathered gradient buffer: 1 layer's grads (for ReduceScatter)
- Prefetch buffers: 2 layers ahead (default `prefetchCount=2`)

### Communication

Per micro-batch, per layer:
- **Forward**: AllGather params (reconstruct layer from shards)
- **Backward**: AllGather params (for gradient computation) + ReduceScatter grads

Total per step: `(2 * paramBytes + gradBytes) * (N-1)/N * GA`

### Timing

- **Overlap**: Per-layer physics simulation via `computeFSDPExposedComm()`. AllGather(N+1)
  prefetched during backward(N). Layer 0's AllGather is not overlapped, giving `(N-1)/N`
  theoretical coverage. Scheduling efficiency η=0.95 (with backward prefetch) or 0.80
  (without prefetch) — η already incorporates prefetch behavior.
- **Backward multiplier**: 2.85x (full ckpt), 2.0+f (selective, f=model-dependent), 2.0x (none)
- **Comm overhead**: two-term model — 1.10× proportional + 50µs/collective (IB) or 5µs/collective (NVLink)
- Communication scales with GA (per micro-batch collectives)

**Source:** `src/core/strategies/fsdp.ts`

---

## 5. Tensor Parallel

Splits individual layers across TP ranks using Megatron-LM style parallelism.

### Sharding

- **Attention**: Q/K/V projections split by heads (column-parallel), output projection
  row-parallel
- **MLP**: First linear column-parallel, second linear row-parallel
- **Embedding/Output**: Optionally parallelized (column-parallel / row-parallel)
- **Norms**: Replicated (small)

### Communication

Per layer, fwd+bwd:
- **Without SP**: 2 AllReduce per layer (post-attention, post-MLP), each `2 * H * tokens * bytes * (tp-1)/tp`
- **With SP**: 4 AllGather + 4 ReduceScatter per layer (fwd+bwd), same total volume but
  chunked/streamed

FP8 quantized collectives on Hopper+ with Transformer Engine (1 byte per element instead of 2).

### Cross-Node TP

When TP exceeds GPUs per node, a hierarchical all-reduce model is used:

```
effectiveBW = (tp-1)/tp / ((G-1)/G/nvBW + (N-1)/N/ibBW)
```

where G = gpusPerNode, N = ceil(tp/G). Overhead factor 1.15x (vs 1.05x for NVLink).
Overlap uses same `C/(C+T) × η` formula; higher T from IB overhead naturally
reduces overlap efficiency.

### TP Overlap Model

Layer-level pipelining where next layer's compute overlaps current layer's communication:

```
tpOverlapEff = C/(C+T) × η
```

- C = computePerMB (forward + backward)
- T = tpCommWithOverhead (includes 1.05x NVLink or 1.15x IB protocol overhead)
- η = 0.96 (scheduling efficiency — CUDA stream, NCCL channel contention, GIL)
- With SP enabled, comm volume changes (AllGather + ReduceScatter vs AllReduce),
  and T reflects this. The scheduling efficiency η captures the remaining overhead.
- Exposed per step = `T * (1 - overlap) * GA`

### Activation Memory

- **With SP**: exact 1/tp per rank -- all tensors effectively split
  (TP-sharded along hidden dim, SP-sharded along sequence dim)
- **Without SP**: weighted average --
  `(sharded/tp + replicated) / (sharded + replicated)` where replicated = 4h
  (LN inputs, residuals)

### Timing

- Backward multiplier: 2.85x (full ckpt), 2.0+f (selective, f=model-dependent), 2.0x (none)

**Source:** `src/core/strategies/tensor-parallel.ts`

---

## 6. Sequence Parallel

Extends TP by distributing LayerNorm and Dropout computations across the sequence dimension.
Uses the same TP process group -- SP is always paired with TP.

### Key Differences from Plain TP

| Aspect              | TP Only                          | TP + SP                        |
|---------------------|----------------------------------|--------------------------------|
| LN/Dropout          | Replicated                       | Split along sequence dim       |
| Activation memory   | Weighted (some replicated)       | Exact 1/tp per rank            |
| Comm ops            | AllReduce                        | AllGather + ReduceScatter      |
| Overlap             | C/(C+T) × η                     | C/(C+T) × η (T reflects AG+RS volume) |

### Communication

Per layer (fwd+bwd): 4 AllGather + 4 ReduceScatter, each `H * tokens * bytes * (tp-1)/tp`.

Same cross-node TP model and overlap formula as tensor-parallel. SP changes
communication ops (AllGather + ReduceScatter vs AllReduce) but the overlap
formula `C/(C+T) × η` is the same -- T reflects the actual comm volume.

**Source:** `src/core/strategies/sequence-parallel.ts`

---

## 7. Pipeline Parallel

Splits the model vertically into PP stages, each holding a contiguous set of layers. Micro-batches
flow through the pipeline with various scheduling strategies.

### Schedules

| Schedule           | Bubble Formula                  | In-Flight MBs      | Notes                |
|--------------------|---------------------------------|---------------------|----------------------|
| GPipe              | `(pp-1)/(pp-1+m)`              | m                   | All-fwd then all-bwd |
| 1F1B               | `(pp-1)/(pp-1+m)`              | pp                  | Interleaved fwd/bwd  |
| Interleaved 1F1B   | `(pp-1)/(pp-1+m*v)`            | ceil(pp/v)          | Virtual stages reduce bubble and activation memory |
| DualPipeV          | `(pp-1)/(pp-1+6*m)` (m>=2*pp)  | 2*pp+1              | V-shape (DeepSeek V3) |
| Zero-bubble        | 0.05 (flat)                     | pp                  | Theoretical minimum  |

### PP Bandwidth

PP uses point-to-point communication between adjacent stages:

- **Intra-node**: NVLink bandwidth
- **Cross-node**: `interNodeBandwidthGBps / gpusPerNode` (per-GPU -- one NIC per P2P transfer)
- **Mixed placement**: `min(intra, perGPU_inter)` -- cross-node link bottlenecks the pipeline

Stage placement is determined by: `stagesPerNode = floor(gpusPerNode / tp)`

### PP Communication Volume

Per micro-batch: 2 P2P transfers (forward activation + backward gradient), each of size
`seqLength * microBatchSize * hiddenSize * commBytes`.

For interleaved schedules: `2 * v` transfers per micro-batch (one pair per virtual stage
boundary). DualPipeV uses v=2 with bidirectional contention.

### PP Overlap Model

```
ppOverlapEff = C/(C+T) × η
```

where η=0.96 (same scheduling efficiency as all strategies).

Exposed per step = `ppCommPerMB * (1 - overlap) * GA` (1.05× P2P overhead is baked into ppCommPerMB via `applyProtocolOverhead()`)

### Timing

- Backward multiplier: 2.85x (full ckpt), 2.0+f (selective, f=model-dependent), 2.0x (none)
- Pipeline efficiency: `max(0.05, 1 - bubble)` -- floor at 5%

**Source:** `src/core/strategies/pipeline-parallel.ts`

---

## 8. 3D Parallel (Unified Engine)

All combined strategies route through `ThreeDParallelStrategy`. This is the most complex
strategy, combining TP, PP, CP, DP, and EP dimensions.

### GPU Layout

```
totalGPUs = TP x PP x CP x DP
```

EP subdivides DP (not a separate dimension in the product). EP must divide DP.

### Configuration

```typescript
interface ThreeDParallelConfig {
  tp: number;                  // Tensor parallel degree
  pp: number;                  // Pipeline parallel degree
  dp: number;                  // Data parallel degree
  ep: number;                  // Expert parallel degree (MoE, default 1)
  cp: number;                  // Context parallel degree (default 1)
  cpImplementation: 'ring' | 'all-gather';  // Ring attention (P2P overlap) vs AllGather (Megatron-LM)
  numMicroBatches: number;     // For pipeline scheduling
  schedule: PipelineSchedule;
  interleavedStages: number;   // Virtual stages (v). 1 = standard, >=2 = interleaved
  sequenceParallel: boolean;   // SP with TP (defaults ON)
  dpType: 'ddp' | 'fsdp' | 'zero-1' | 'zero-2' | 'zero-3';
  activationCheckpointing: boolean;
  checkpointingGranularity: 'full' | 'selective';
}
```

### Memory Model

**Parameters**: Shared params sharded by TP x PP (+ DP for FSDP/ZeRO-3). Expert params
additionally sharded by EP. Gradients follow the same sharding with ZeRO-2 also sharding grads.

**Activations**: Per-stage, per-micro-batch, scaled by:
- SP multiplier (1/tp with SP, weighted average without)
- In-flight micro-batches (schedule-dependent)
- CP (each rank holds seq/cp tokens)
- MoE router/dispatch overhead with EP capacity factor 1.15

**Peak activations** (FSDP/ZeRO-3): base activations + gathered param buffer + gathered grad
buffer + 2-layer prefetch buffers.

### DP Communication

**DP bandwidth detection**: `baseDPBW` is structurally derived from cluster topology:
```
dpRanksPerNode = stagesPerNode >= pp ? floor(stagesPerNode / pp) : 0
baseDPBW = dpRanksPerNode >= dp ? NVLink BW : IB BW
```
This ensures DP communication uses NVLink when all DP ranks fit within a node, and IB
otherwise.

Depends on dpType:
- **DDP**: AllReduce gradients once per step. Bucket model overlap (25MB buckets × η).
- **FSDP/ZeRO-3**: AllGather+ReduceScatter per layer per micro-batch. Scales with GA.
  Per-layer physics simulation via `computeFSDPExposedComm()` with η=0.95 (prefetch) / 0.80 (no prefetch).
- **ZeRO-1/ZeRO-2**: Gradient sync (bucket model × η) + sequential param AllGather after
  optimizer.

For MoE with EP, expert params use smaller DP/EP groups (less congestion) while shared params
use full DP. Penalties blended by harmonic mean of volume fractions.

### TP Communication

Same layer-level overlap model as standalone TP:

```
tpOverlapEff = C/(C+T) × η
exposed = T * (1-overlap) * GA
```

FP8 quantized collectives on Hopper+ for TP (1 byte). PP/CP comm stays BF16.

### PP Communication

Per-GPU wall-clock from first principles:
```
ppCommPerMB = 2 * v * (activationBytes / ppBW)
ppOverlapEff = C/(C+T) × η
ppExposed = ppCommPerMB * (1 - ppOverlapEff) * GA + ppTransitionOverheadMs * GA
```

Uses `min(intra, perGPU_inter)` for mixed intra/cross-node PP placement.

### Context Parallel

Each CP rank holds `seqLength/cp` tokens. Two implementations are supported:

**Ring Attention** (`cpImplementation: 'ring'`): A CP-step pipeline within the attention
layer. Per-chunk attention compute (QK^T + softmax + scores×V) overlaps with KV P2P transfer.

Per-step compute varies due to causal masking (`cpCausalWorkDistribution`):
- 1 diagonal step: self-attention chunk with triangular mask → 0.5× compute
- CP-2 normal steps: full cross-attention chunks → 1× compute

```
jitterFloor = 0.03 + 0.005 * (CP - 1)  // 3.5% at CP=2, 10.5% at CP=16

exposedDiagonalFwd = max(T * jitterFloor, T - computeFwd * 0.5)
exposedNormalFwd   = max(T * jitterFloor, T - computeFwd)
exposedFwd = 1 * exposedDiagonalFwd + (CP - 2) * exposedNormalFwd

exposedDiagonalBwd = max(T * jitterFloor, T - computeBwd * 0.5)
exposedNormalBwd   = max(T * jitterFloor, T - computeBwd)
exposedBwd = 1 * exposedDiagonalBwd + (CP - 2) * exposedNormalBwd

exposed = fwdPasses * exposedFwd + exposedBwd
```

**All-Gather** (`cpImplementation: 'all-gather'`): AllGather KV before attention, fully
exposed. Used by Meta and Megatron-LM. Simpler but higher communication overhead.

```
allGatherTime = kvBlockBytes * (CP-1) / (bandwidth * collectiveEff)
overlap = isInterleaved ? 0.15 : 0.05
exposed = fwdPasses * allGatherTime * (1 - overlap)
```

- Per-chunk attention FLOPs: `2 × numHeads × chunkSeq² × (qkDim + vDim) × MBS / TP`
- Backward attention has 2× forward FLOPs (dQ, dK, dV gradients)
- Forward passes: 2 with AC (forward + recompute), 1 without
- Ring P2P BW efficiency: 0.90 (NVLink) / 0.85 (IB)
- All-gather collective efficiency: 0.90 (NVLink) / 0.82 (IB)
- CP comm uses BF16 floor (same convention as PP)
- CP placement: NVLink if tp*cp fits in node, else IB (per-GPU P2P bandwidth)
- All-gather backward has no ReduceScatter — each rank computes local dK/dV

### Expert Parallel

EP subdivides DP -- shared params use full DP sharding, expert params use DP/EP subgroups.

**Per-layer MoE timing model:**
```
router + permute + dispatch_alltoall -> expert_compute/(tp*ep) -> combine_alltoall + unpermute
+ getGroupedGemmEfficiency() penalty on expert compute (baseline ≈ 0.952, see [PHYSICS.md §3](PHYSICS.md#3-moe-specific-efficiency))
+ ALLTOALL_COORD_OVERHEAD (0.30 × transport, EP>1 only)
```

**Physics floor**: Dual-floor model when EP > 1: non-expert FLOPs are divided by `tp × pp`,
expert FLOPs are divided by `ep × pp` (with grouped GEMM penalty). The two floors sum.
When EP = 1: single floor weighted by routed FLOP fraction. See [PHYSICS.md §8](PHYSICS.md#8-overlap-models) for formulas.

**EP slack overlap**: When physics floor > compute-only time, GPU has slack cycles. EP dispatch
overlaps with this slack at η (schedulingEfficiency = 0.96) — the same scheduling efficiency
used for all comm/compute overlap.

**EP all-to-all bandwidth**:
- Base: 0.55 (NVLink) / 0.45 (IB)
- Group size penalty: `1/(1 + 0.18*log2(ep/8))` for ep > 8
- Latency: 50us (NVLink) / 100us (IB) per collective

**Routing locality**: Reduces all-to-all volume based on expert placement:
1. Density-based: `1/(1 + expertsPerRank/numActive)` -- universal, no free params
2. Device-limited routing (optional): `min(densityLocality, routingDeviceLimit/ep)` --
   only V3/R1 (limit=4) and V2 (limit=6) set this

**Load imbalance**: Derived from multinomial order statistics. The expected maximum
load across E experts uses `σ = √((1-p)/tokensPerExpert)` where `p = topK/numExperts`:
```
EP > 1: loadImbalanceFactor = 1.0 (load imbalance modeling pending validation)
EP = 1: loadImbalanceFactor = 1.02
```

**Expert GEMM roofline**: Small expert GEMMs can be memory-bandwidth-bound. Roofline factor
`min(1.0, arithmeticIntensity * peakBW / effectiveTFLOPS)` applied to expert compute time.

### Backward Multiplier

| Condition                    | Multiplier |
|------------------------------|------------|
| Full checkpointing           | 2.85x      |
| Selective checkpointing      | 2.0 + f    |
| No checkpointing             | 2.0x       |

Where `f = getSelectiveRecomputeFraction(model)` (typically 0.13-0.33, model-dependent).
All strategies use the same backward multiplier for compute. **EP communication uses a
separate backward multiplier**: 2× with activation checkpointing (recompute re-runs
dispatch+combine, gradient backward reverses them), 1× without. This prevents EP comm
from being inflated by the compute backward multiplier (2.85× with full AC).
FSDP/ZeRO-3 AllGather/recompute overlap is credited via `computeFSDPExposedComm()`, not as
a reduced multiplier.

### Efficiency

- Efficiency: `saturation × memBWScaling × runtimeResidual` (0.655) — same multiplicative formula as all strategies
- MoE dispatch overhead: grouped GEMM penalty via `getGroupedGemmEfficiency()` + AllToAll coordination for EP>1 (see [PHYSICS.md §3 — MoE-Specific Efficiency](PHYSICS.md#3-moe-specific-efficiency))

**Source:** `src/core/strategies/3d-parallel.ts`

---

## 9. LoRA/QLoRA Fine-tuning

LoRA (Low-Rank Adaptation) freezes base model weights and adds small trainable low-rank
adapter matrices. QLoRA compresses base weights to NF4 (4-bit).

### Configuration

```typescript
type FinetuningMethod = 'full' | 'lora' | 'qlora';
type LoraTargetModules = 'q_v' | 'q_k_v_o' | 'all_linear';

interface LoraConfig {
  method: FinetuningMethod;
  rank: number;               // 4, 8, 16, 32, 64
  targetModules: LoraTargetModules;
}
```

### Trainable Parameters

Per-layer adapter params depend on target modules:
- **q_v**: Q + V adapters only (2 projection pairs)
- **q_k_v_o**: Q + K + V + O adapters (4 projection pairs)
- **all_linear**: All attention + MLP adapters. For MoE, skips routed experts -- only
  attention + dense MLP + shared expert MLP

Each adapter pair: A[d, r] + B[r, out_dim] per projection.
GQA-aware: K/V use `numKvHeads * headDim` (smaller than Q/O for GQA models).
MLA-aware: uses `kvLoraRank` / `qLoraRank` for DeepSeek V2/V3/R1.

### MFU and HFU

| Mode                 | MFU        | HFU                  |
|----------------------|------------|----------------------|
| LoRA, no ckpt        | 4PD        | 4PD                  |
| LoRA, full ckpt      | 4PD        | 6PD (4 + 2 recompute)|
| Full, no ckpt        | 6PD        | 6PD                  |
| Full, full ckpt      | 6PD        | 8PD                  |

LoRA uses 4PD because only forward (2PD) + activation grads (2PD) are computed -- frozen
weight gradients are skipped.

### Backward Multiplier

```
overhead = 1.0 + max(0.05, 2 * trainableParams / totalParams)
```

The 0.05 floor always dominates for models >= 7B (even rank=64 all_linear on 7B is ~2.3%
of params). Effectively flat 1.05 in practice.

| Checkpointing | Multiplier                          |
|---------------|-------------------------------------|
| None          | 1.05x                               |
| Selective     | f + 1.05x (f = attn recompute fraction) |
| Full          | 2.05x (1.0 recompute + 1.05x)      |

### QLoRA Specifics

- **NF4 storage**: 0.515 bytes/param (0.5 + 3% scale overhead for 1 FP16 scale per 64 values)
- **Dequantization**: Bandwidth-bound, 2.5 bytes/param (reads 0.5 NF4 + writes 2.0 BF16).
  Included in forward time. 7B: ~5ms (negligible). 70B: ~50ms at pp=1 (significant).
- **FSDP comm asymmetry**: Forward AllGather gathers full base weights at NF4 (0.515 bytes).
  Backward ReduceScatter only adapter grads at BF16 (tiny volume).

### TP Adapter Sharding

Follows Megatron-LM conventions:
- **Column-parallel** (Q, K, V, gate, up): A replicated, B sharded by TP.
  Per rank: `r * (d + out/tp)`
- **Row-parallel** (O, down): A sharded by TP, B replicated.
  Per rank: `r * (in/tp + d)`

Effective sharding ratio: ~(1 + 1/tp) / 2 for symmetric dimensions.

### Memory Ratio

- LoRA / full: ~0.35 on DDP (single GPU), ~0.63 on FSDP 8x
  (FSDP already shards base weights, diluting LoRA savings)
- QLoRA 70B fits 80GB GPU (~48.7 GB)
- Adapter gradients: always BF16, even with FP8 compute
- Activation memory: unchanged by LoRA (same intermediate tensors needed for activation grads)

**Source:** `src/core/strategies/lora.ts`

---

## 10. Selective Activation Checkpointing

Three-way selector: **Disabled** / **Selective** / **Full**.

Controlled by two fields: `activationCheckpointing: boolean` and
`checkpointingGranularity: 'full' | 'selective'`.

### Per-Layer Activation Decomposition

Each transformer layer's activations are decomposed into three groups:

**Shared (always kept):**
- LN1 input (h) + post-attention residual (h) + LN2 output (h) = 3h per token
- The post-attention residual serves double duty as both the residual and LN2 input

**Attention (discarded in selective, kept in none):**
- LN1 output (h) + attention output (h) + Q, K, V projections + attention scores + dropout mask
- Without Flash Attention: attention scores use 2.5x multiplier per [Korthikanti et al. 2022](https://arxiv.org/abs/2205.05198)
  (pre-softmax scores 2B + post-softmax probs 2B + dropout mask 1B = 5 bytes per element)
- With Flash Attention: O(seq) tile memory, no multiplier

**MLP (always kept):**
- MLP intermediates: 3I for gated MLP (SwiGLU: gate + up + product), 2I for standard
- MLP dropout mask (1 byte per token)

### Memory Comparison

| Mode      | What is stored                     | Memory order           |
|-----------|------------------------------------|------------------------|
| None      | All N layers, all activations      | Highest                |
| Selective | All N layers, shared + MLP only    | ~70-80% of none        |
| Full      | sqrt(N) layers, all activations    | Lowest                 |

Selective stores activations for every layer but discards the attention portion. During backward,
the Q/K/V/O linear projections are recomputed from the saved LN1 input. The recompute cost
is model-dependent (13-30% of per-layer FLOPs), determined by the attention-to-MLP ratio.

Full uses sqrt(N) checkpointing: only stores activations at evenly-spaced checkpoint boundaries,
recomputes everything between checkpoints during backward.

### Backward Multipliers

| Granularity | Multiplier | Explanation                      |
|-------------|------------|----------------------------------|
| Full        | 2.85x      | Full forward recompute           |
| Selective   | 2.0+f      | Attention linear recompute       |
| None        | 2.0x       | No recompute                     |

Where `f = getSelectiveRecomputeFraction(model)` — fraction of per-layer FLOPs from Q/K/V/O
projections. GQA models: 13-22%, MHA models: 25-33%.
All strategies use the same backward multiplier. FSDP/ZeRO-3 AllGather/recompute overlap
is credited via `computeFSDPExposedComm()`, not as a reduced multiplier.

### HFU Treatment

HFU uses the continuous formula `(6+2f)PD`, where f is the recompute fraction:
- f = 0 (no AC): HFU = 6PD (same as MFU)
- f = selectiveRecomputeFraction (selective AC, typically 0.13–0.22): HFU ≈ 6.3–6.4 PD
- f = 1.0 (full AC): HFU = 8PD

`usesActivationCheckpointing()` only controls the HFU **display label** (whether the
dashboard shows a separate HFU line), not the math. The continuous `(6+2f)` formula
applies in all cases. LoRA uses `(4+2f)`.

### Context Parallel Interaction

With CP and activation checkpointing (full or selective), the forward ring runs twice
(forward + recompute). Selective AC specifically recomputes attention, requiring the
same ring traversal count as full AC. The CP ring buffer uses 3x multiplier for memory
(forward recompute + backward + receive) instead of 2x (compute + receive).

### Share URL

Optional `acg` field: 's' = selective, omitted = full.

**Source:** `src/core/strategies/base.ts` (estimateActivationMemory, usesActivationCheckpointing),
all strategy files for backward multiplier constants.
