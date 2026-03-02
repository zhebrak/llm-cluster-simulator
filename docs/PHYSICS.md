# Physics Model Reference

Complete reference for every physical factor, formula, constant, and modeling assumption
in the ML Cluster Simulator. All values are sourced directly from the codebase. Function
names are cited for traceability; line numbers are omitted (they change with edits).

> **See also:** [STRATEGIES.md](STRATEGIES.md) (implementations) | [HARDWARE.md](HARDWARE.md) (GPU specs) | [BENCHMARKS.md](BENCHMARKS.md) (validation)

---

## Table of Contents

1. [MFU and HFU Definitions](#1-mfu-and-hfu-definitions)
2. [Compute Efficiency Model](#2-compute-efficiency-model)
3. [MoE-Specific Efficiency](#3-moe-specific-efficiency)
4. [GPU-Level Scaling](#4-gpu-level-scaling)
5. [DP Scaling Laws](#5-dp-scaling-laws)
6. [Scale Overhead (Stragglers)](#6-scale-overhead-stragglers)
7. [Communication Overhead Multipliers](#7-communication-overhead-multipliers)
8. [Overlap Models](#8-overlap-models)
9. [Backward Multipliers](#9-backward-multipliers)
10. [Memory Model](#10-memory-model)
11. [Pipeline Bubble Formulas](#11-pipeline-bubble-formulas)
12. [Optimizer Step Timing](#12-optimizer-step-timing)
13. [Full Penalty Audit](#13-full-penalty-audit)

---

## 1. MFU and HFU Definitions

**Source:** `base.ts:ParallelismStrategy.computeAnalysis()`

### MFU (Model FLOPS Utilization) -- PaLM definition

```
MFU = (6 * P * D) / (stepTime_s * N * peak_bf16_TFLOPS * 1e12)
```

| Symbol | Meaning |
|--------|---------|
| P | `activeParams` -- for MoE models only active experts contribute per token |
| D | `totalTokens = seqLength * globalBatchSize` |
| N | `totalGPUs` |
| peak_bf16 | `getEffectiveTFLOPS(gpu, 'bf16')` -- **always BF16**, even when training in FP8/FP4 |

MFU always uses BF16 peak as the denominator. This is the industry convention (matches
DeepSeek, Meta, NVIDIA published numbers). FP8/FP4 accelerate matmul kernels but MFU
is reported against the standard training peak so that numbers are comparable across
papers.

The numerator counts **useful work only**:

- Forward: 2PD
- Backward: 4PD (2PD activation gradients + 2PD weight gradients)
- Total: **6PD**

Activation checkpointing recompute is NOT included in MFU -- it is overhead, not useful
work toward convergence.

### HFU (Hardware FLOPS Utilization)

HFU includes recompute overhead to show how hard the hardware is actually working:

| Condition | HFU numerator |
|-----------|---------------|
| Full training, no checkpointing | 6PD (same as MFU) |
| Full training, full checkpointing | 8PD (extra forward pass during backward) |
| Full training, selective checkpointing | 6PD (recompute 13-30%, not counted) |
| LoRA, no checkpointing | 4PD (no frozen weight grads) |
| LoRA, full checkpointing | 6PD (4PD + 2PD recompute) |

**Selective checkpointing and HFU:** `usesActivationCheckpointing()` returns `false`
for selective granularity. Recompute is 13-30% of per-layer FLOPs (attention linear
projections), but HFU ~= MFU for selective because the difference is small enough
that showing a separate HFU value adds noise without insight.

### Implementation detail

```typescript
const modelFlops = (ctx.lora ? 4 : 6) * activeParams * totalTokens;
const hardwareFlops = ctx.lora
  ? (checkpointingEnabled ? 6 : 4) * activeParams * totalTokens
  : (checkpointingEnabled ? 8 : 6) * activeParams * totalTokens;
```

### Model FLOPs MFU

When sequence length is long enough for quadratic attention FLOPs to exceed the `6P`
approximation by >10%, the simulator also reports **Model FLOPs MFU**:

```
Model FLOPs MFU = 3 × flopsPerToken × D / (stepTime_s × N × peak_bf16_TFLOPS × 1e12)
```

`flopsPerToken` is computed by `getModel(id, seqLength)` and includes the full
quadratic attention cost (QK^T, softmax, scores×V scale as seq²). The `3×` multiplier
accounts for forward + backward passes (same as the `6P = 3 × 2P` convention).
For LoRA fine-tuning, the multiplier is `2×` instead (no frozen weight gradients).

At short sequences (≤8K for most models), attention is <10% of total compute and
Model FLOPs MFU ≈ PaLM MFU. At 131K, actual training FLOPs ≈ 2.34 × `6PD` for
405B due to quadratic attention, so Model FLOPs MFU ≈ 2.34× PaLM MFU.

This metric helps reconcile published MFU numbers that use actual model FLOPs (e.g.,
[Meta ISCA 2025](https://arxiv.org/abs/2407.21783) reports 38% for 405B at 131K — our Model FLOPs MFU shows ~38.3% for
this config using the all-gather CP implementation that Meta uses).

Only displayed when `modelFlopsMfu / palmMfu > 1.10`.

---

## 2. Compute Efficiency Model

**Source:** `base.ts:computeComputeEfficiency()`

### Principle

Efficiency captures **pure GPU kernel utilization only** -- how well the matmul and
non-matmul kernels saturate the hardware. All communication costs (TP all-reduce, DP
AllGather, PP P2P, EP all-to-all) are modeled explicitly and separately. There are no
TP/DP penalties inside the efficiency term to avoid double-counting.

### Formula

```
efficiency = saturation × memBWScaling × runtimeResidual
```

Bounded [0,1] by construction: each factor is individually bounded. No cap needed.

### Components

| Component | Value | Notes |
|-----------|-------|-------|
| saturation | `getComputeSaturationFactor()` | Power-0.3 curve with raised thresholds (5M for Hopper). Based on per-GPU GEMM dimensions (tokens × hidden/TP) |
| memBWScaling | `getMemoryBandwidthScaling()` | Amdahl model for non-matmul ops (see [GPU-Level Scaling](#4-gpu-level-scaling)) |
| runtimeResidual (dense) | `0.655` | Kernel launch gaps, CUDA stream scheduling, autograd overhead, memory allocator pressure. Framework-level optimizations (kernel fusion, memory planning, communication scheduling) partially offset these losses |
| runtimeResidual (MoE) | `0.97` | Near 1.0 because most MoE overhead is modeled explicitly (expert GEMM efficiency, EP transport, load imbalance). Captures only residual per-layer overhead: router backward, permutation backward, auxiliary load-balance loss gradients, expert scheduling. Applied only via `3d-parallel.ts` when `model.isMoE = true` |

### Saturation thresholds (getComputeSaturationFactor)

Separate from `getMatmulSaturationFactor()` which uses low thresholds (~1M) for
expert GEMM saturation in `3d-parallel.ts`.

| Architecture | Threshold | Ratio to Hopper |
|-------------|-----------|-----------------|
| Hopper (H100) | 5M | 1.00× |
| Blackwell (B200) | 6.25M | 1.25× |
| Ampere (A100) | 3.75M | 0.75× |
| Ada (L40S) | 3.125M | 0.625× |

### Effective TFLOPS

```
effectiveTFLOPS = getTrainingTFLOPS(gpu, computeDtype) * efficiency
```

### Stability

Two fitted parameters (dense and MoE runtime residuals); saturation and memBW are derived
from hardware specs. The split captures a systematic ~2.5pp gap between dense and MoE
benchmarks: dense models undershoot published MFU on average, MoE models overshoot.

Implied residual across 6 dense train-set benchmarks: mean=0.655, CV=0%.

| Benchmark | Efficiency | Saturation | MemBW | Implied Residual |
|-----------|-----------|------------|-------|-----------------|
| LLaMA 3.1 405B | 65.5% | 1.000 | 1.000 | 0.655 |
| GPT-3 175B | 67.0% | 0.949 | 1.078 | 0.655 |
| IBM FSDP 7B | 70.6% | 1.000 | 1.078 | 0.655 |
| Nemotron-4 340B | 65.5% | 1.000 | 1.000 | 0.655 |
| OLMo 3 32B | 65.5% | 1.000 | 1.000 | 0.655 |

MoE models use 0.97 when routed through `3d-parallel.ts` (the only strategy with
MoE-specific modeling). Other strategies (FSDP, DDP, ZeRO) use the dense residual.

### Example: LLaMA 2 7B on FSDP (A100)

- Tokens per micro-batch: 8192, hidden/TP: 4096
- Elements: 33.6M > threshold 3.75M → saturation = 1.0
- memBWScaling: 1.078 (A100)
- runtimeResidual: 0.655 (dense)
- Efficiency: 1.0 × 1.078 × 0.655 = 0.706

### Example: LLaMA 3.1 405B (H100, TP=8)

- Tokens per micro-batch: 8192, hidden/TP: 2048
- Elements: 16.8M > threshold 5M → saturation = 1.0
- memBWScaling: 1.0 (H100 is reference)
- runtimeResidual: 0.655 (dense)
- Efficiency: 1.0 × 1.0 × 0.655 = 0.655

---

## 3. MoE-Specific Efficiency

**Source:** `base.ts:computeComputeEfficiency()`, `3d-parallel.ts:computeTiming()`

### MoE compute model

When EP > 1, experts are distributed full-rank across EP ranks (not TP-sharded).
Each EP rank holds complete expert weights for its local experts. TP applies only
to attention and shared experts. Per-GPU routed expert compute divides by EP only
(not TP × EP). When EP = 1, experts are TP-sharded like dense layers (divide by TP).

Expert GEMM efficiency is modeled via three mechanisms:

1. **Matmul saturation**: `getMatmulSaturationFactor()` applied at expert GEMM
   dimensions. When EP > 1: full-rank (`tokPerExpert × expertIntermediate`).
   When EP = 1: TP-sharded (`tokPerExpert × expertIntermediate/tp`). Saturation
   floor 0.50. Small expert GEMMs that under-fill tensor cores receive a
   multiplicative penalty.

2. **Grouped GEMM efficiency**: `getGroupedGemmEfficiency()` models throughput
   degradation of grouped expert GEMMs relative to dense GEMMs. Expert GEMMs in
   grouped context achieve 70-85% of equivalent dense GEMM throughput (directionally
   validated against SonicMoE/CUTLASS benchmarks). The factor uses raw
   `expertIntermediateSize` (not /tp) — an architectural property, not a deployment
   property.

   ```
   dimThreshold = 1536  (Hopper; architecture-dependent)
   baseLine     = 1/1.05 ≈ 0.952   (subsumes old GROUPED_GEMM_OVERHEAD = 0.05)
   floor        = 0.30

   if expertIntermediateSize >= dimThreshold:
     return baseLine

   dimRatio   = expertIntermediateSize / dimThreshold
   baseFactor = max(floor, baseLine × dimRatio^0.80)    // EXPERT_GEMM_EXPONENT

   if expertsPerGPU > 8:
     countPenalty = 1 + 0.55 × log₂(expertsPerGPU / 8)  // EXPERT_COUNT_SCALE
     return max(floor, baseFactor / countPenalty)

   return baseFactor
   ```

   Modulated by the roofline factor:
   `effectiveFactor = 1.0 - min(roofline, 1.0) × (1.0 - rawFactor)`, so the
   penalty diminishes when individual GEMMs are already memory-bandwidth-bound
   (cache thrashing doesn't matter if the GEMM doesn't benefit from cache).
   Directionally validated against SonicMoE (arXiv:2512.14080), cuBLAS benchmarks,
   H100 L2 = 50MB.

3. **MoE dispatch overhead**: `coordOverheadPerLayer = 0` (zeroed pending high-EP
   anchor, EP>=16). The parameters `ALLTOALL_COORD_OVERHEAD` and `EP_MIN_COORD_MS`
   exist in the parameter registry but are not applied. Reintroduction depends on
   validation with high-EP calibration data.

### EP all-to-all latency scaling

**Source:** `3d-parallel.ts:computeTiming()`

Per-layer all-to-all latency uses a constant base value (no scaling applied):

```
baseLatency = epWithinNode ? 0.05 : 0.10  ms
alltoallLatencyMs = baseLatency
```

EP latency scaling (`EP_LATENCY_ALPHA`) is zeroed pending high-EP anchor (EP>=16).
The parameter exists in the registry but the scaling formula is not applied —
all EP values use the flat base latency.

| EP  | NVLink latency | IB latency |
|-----|----------------|------------|
| ≤4  | 0.050 ms       | 0.100 ms   |
| 8   | 0.050 ms       | 0.100 ms   |
| 16  | 0.050 ms       | 0.100 ms   |
| 32  | 0.050 ms       | 0.100 ms   |
| 64  | 0.050 ms       | 0.100 ms   |

**Device-limited routing:** Models with `routingDeviceLimit = M` (e.g. DeepSeek V3
with M=4) route each token to at most M EP ranks. BW group-size penalty
(`getDPGroupSizePenalty(ep, {ref:8})`) uses raw `ep` because NCCL scheduling
operates over the full communicator regardless of routing sparsity. Volume
calculations are unaffected — they already account for device-limited routing via
`routingLocality`.

### Expert GEMM roofline factor

**Source:** `base.ts:getExpertGemmRooflineFactor()`

Small expert GEMMs (low M after token routing, small K or N) can be
memory-bandwidth-bound rather than compute-bound.

```
rooflineFactor = min(1.0, AI * peakBW / effectiveTFLOPS)
```

Where:
- EP > 1: `AI = gemmIntensity(tokPerExpert, hiddenSize, expertIntermediate, bytesPerElem)` (full-rank)
- EP = 1: `AI = gemmIntensity(tokPerExpert, hiddenSize/tp, expertIntermediate/tp, bytesPerElem)` (TP-sharded)
- `gemmIntensity(M, K, N, b) = 2*M*K*N / ((M*K + K*N + M*N) * b)`

Returns 1.0 when compute-bound, <1.0 when bandwidth-bound. Applied as a multiplicative
factor on expert compute time.

### Load imbalance

**Source:** `3d-parallel.ts:computeTiming()`

Token routing is imperfect. With EP>1, the slowest GPU among EP ranks determines
expert compute time (cross-GPU sync barrier).

```
EP > 1:  loadImbalanceFactor = 1.0   (zeroed pending 3rd MoE anchor — see code comment)
EP = 1:  loadImbalanceFactor = 1.02
```

EP>1 load imbalance modeling is zeroed pending validation with a 3rd MoE calibration
anchor. The multinomial order-statistics formula and `LOAD_BALANCE_DAMPING` exist in
`physics/derived.ts` but are not applied. EP=1 uses 1.02 — no cross-GPU sync barrier,
just minor grouped GEMM inefficiency from variable token counts per expert.

### Router overhead

```
routerTime = tokens * hiddenSize * numExperts * 2 / effectiveTFLOPS
```

Gating network forward pass (linear projection) + top-k selection.

### Permutation overhead

Memory-bandwidth-bound token reordering for dispatch and combine:

```
permutationBytes = 4 * tokens * hiddenSize * DTYPE_BYTES[activationDtype]
permutationTime  = permutationBytes / memBW
```

### Routing locality (EP all-to-all volume reduction)

**Source:** `3d-parallel.ts:computeCommunication()`, `computeTiming()`

Two complementary effects reduce all-to-all traffic below worst-case uniform random:

**1. Density-based locality:**
```
densityLocality = 1 / (1 + expertsPerRank / numActive)
```

Learned routers preferentially activate local experts proportional to supply/demand:
- density=1 (supply=demand): 50% cross-rank
- density=4 (large local pool): 20% cross-rank
- density -> infinity (all local): 0% cross-rank

**2. Device-limited routing (optional):**

Some architectures hard-cap the number of EP groups each token can contact. When
`routingDeviceLimit` (M) is set on the model:

```
routingLocality = min(densityLocality, M / ep)
```

Only DeepSeek V3/R1 (M=4) and V2 (M=6) use this. For V3 with EP=32:
density-based = 0.50, device-limited = 4/32 = 0.125.

---

## 4. GPU-Level Scaling

**Source:** `gpu.ts`

### Memory bandwidth scaling (Amdahl)

**Function:** `getMemoryBandwidthScaling(gpu, computeDtype)`

Approximately 15% of training compute time is memory-bandwidth-bound (non-matmul ops:
LayerNorm, Softmax, GELU/SiLU, activation I/O). GPUs with higher bandwidth relative
to compute spend less time on these ops.

```
MEMBW_BOUND_FRACTION = 0.15
referenceOI = 989 / 3.35 = 295.2  (H100 SXM BF16)

oi = peakTFLOPS / memoryBandwidthTBps
normalizedTime = (1 - 0.15) + 0.15 * (oi / referenceOI)
scaling = 1 / normalizedTime
```

Returns >1 for GPUs with relatively more bandwidth per FLOP (faster non-matmul ops).
Applied multiplicatively in the efficiency formula.

### Matmul saturation factor

**Function:** `getMatmulSaturationFactor(tokensPerMicroBatch, hiddenSizePerGPU, gpu)`

Large matrices fully saturate tensor core SMs; small matrices leave SMs idle.

```
matmulElements = tokensPerMicroBatch * hiddenSizePerGPU
ratio = matmulElements / threshold
factor = ratio >= 1.0 ? 1.0 : ratio^0.3
```

Smooth power-law degradation: at 50% of threshold, throughput is ~81%. Only truly
tiny matmuls (<5% of threshold) see catastrophic undersaturation.

Applied multiplicatively in the efficiency formula.

**Saturation thresholds by architecture:**

| Architecture | Threshold | Examples |
|-------------|-----------|---------|
| Hopper | 1,048,576 (~1M) | H100, H200, H800 |
| Blackwell | 1,310,720 (~1.3M) | B200, GB200 |
| Ampere | 786,432 (~768K) | A100, A800 |
| Ada | 655,360 (~640K) | L40S, L4, RTX 4090 |
| CDNA3 | 1,048,576 (~1M) | MI300X, MI325X |
| CDNA4 | 1,310,720 (~1.3M) | MI350X |
| CDNA2 | 786,432 (~768K) | MI250X |
| Volta | 524,288 (~512K) | V100 |
| Turing | 524,288 (~512K) | T4 |

### Training TFLOPS (FP8/FP4 Amdahl model)

**Function:** `getTrainingTFLOPS(gpu, dtype)`

For FP8/FP4, only matmul ops (80% of step time) get the 2x/4x speedup. Non-matmul
ops (LayerNorm, softmax, GELU, loss) stay at BF16 rate.

```
MATMUL_TIME_FRACTION = 0.80

matmulSpeedup = rawTFLOPS / bf16TFLOPS
effectiveTimeFraction = 0.80 / matmulSpeedup + 0.20
trainingTFLOPS = bf16TFLOPS / effectiveTimeFraction
```

**Example:** H100 SXM FP8:
- Raw FP8: 1979 TFLOPS, BF16: 989 TFLOPS
- matmulSpeedup = 2.0
- effectiveTimeFraction = 0.80/2 + 0.20 = 0.60
- trainingTFLOPS = 989 / 0.60 = 1648 (not raw 1979)

For BF16/FP16/TF32/FP32: returns `getEffectiveTFLOPS(gpu, dtype)` unchanged (no
matmul/non-matmul split needed).

### Transformer Engine overhead

**Source:** `gpu.ts:getTrainingTFLOPS()` (lines 1048-1052)

FP8/FP4 GEMMs require per-tensor quantize/dequantize (FP16→FP8 casting), amax
history tracking, and delayed scaling factor updates. FP4 has additional scaling
complexity. The Amdahl-adjusted TFLOPS is further reduced by a TE overhead factor:

```
trainingTFLOPS = amdahlTFLOPS × TE_OVERHEAD
```

| Dtype | TE_OVERHEAD | Overhead |
|-------|-------------|----------|
| FP8   | 0.93        | 7%       |
| FP4   | 0.90        | 10%      |

### FP8 quantized collectives

**Source:** `3d-parallel.ts:computeMemoryPerGPU()`, `computeCommunication()`

When `activationDtype === 'fp8'` and `gpu.hasTransformerEngine` (Hopper+):
- **TP all-reduce:** 1 byte/element (FP8 quantized collective)
- **PP/CP comm:** BF16 floor (2 bytes) -- activations/KV blocks need precision
- **EP all-to-all:** uses `activationDtype` directly

Non-Hopper GPUs fall back to BF16 floor for all communication.

---

## 5. DP Scaling Laws

**Source:** `base.ts:getDPGroupSizePenalty()`, `getDPCollectiveLatencyMs()`

Two independent degradation mechanisms, both with a threshold at dp=64 (within-pod
boundary where network topology is well-mapped). Below dp=64, both functions return
their best-case values. Above 64, log-scale degradation.

### BW penalty (fabric congestion)

```
getDPGroupSizePenalty(dp):
  dp <= 64:  1.0
  dp > 64:   max(0.40, 1 / (1 + 0.15 * log2(dp/64)))
  dp > 256:  above × 1/(1 + 0.08 * log2(dp/256))   [second-tier]
```

Models multi-hop routing through IB fat-tree fabric, network congestion from
concurrent collectives, and NCCL scheduling overhead. Second-tier penalty
(dp > 256) captures additional congestion from multi-rail saturation, ECMP
collisions, and adaptive routing overhead in large IB fabrics.

Floor of 0.40 binds at dp ~ 51,000 (effectively never).

### Per-collective latency

```
getDPCollectiveLatencyMs(dp):
  dp <= 64: 0
  dp > 64:  0.030 * log2(dp/64)   (30 microseconds per log2 step)
```

NCCL tree latency scales as ~log2(N) with ~3-5 microseconds per inter-node hop.
The 30 microseconds per log2 step accounts for 2x log2(N) hops plus sync jitter.

### DP group size

The DP group size input to penalty functions depends on the strategy type:
- **Standalone strategies** (DDP, FSDP, ZeRO): `numNodes` (inter-node
  participants). NCCL hierarchical collectives run intra-node rings/trees
  first, then inter-node reduction. Fabric congestion scales with inter-node
  hops, not total GPUs. For example, 1280 GPUs on 160 nodes uses
  dpGroupSize=160, not 1280.
- **3D strategies**: the DP degree directly. DP degree ~ numNodes when TP fills
  the node, so the distinction rarely matters for 3D.

[MegaScale (NSDI 2024)](https://www.usenix.org/conference/nsdi24/presentation/jiang-ziheng) validates continued degradation through DP=192 with no plateau.

| dp | BW penalty | Latency | Notes |
|----|-----------|---------|-------|
| 32 | 1.00 | 0.00 ms | |
| 64 | 1.00 | 0.00 ms | |
| 128 | 0.87 | 0.03 ms | |
| 256 | 0.77 | 0.06 ms | |
| 512 | 0.65 | 0.09 ms | Second-tier kicks in |
| 1024 | 0.56 | 0.12 ms | Both tiers active |

---

## 6. Scale Overhead (Stragglers) — Order Statistics Model

**Source:** `base.ts:ParallelismStrategy.computeAnalysis()`

BSP (Bulk Synchronous Parallel) training synchronizes at each step: the slowest of
N nodes gates the step. Node-level jitter (OS scheduling, thermal throttling, NIC
congestion) means each node's step time is a random variable with coefficient of
variation σ. The expected maximum of N i.i.d. random variables grows as:

```
E[max(X₁...Xₙ)] ≈ μ + σ√(2·ln(N))
```

This gives the straggler overhead formula (1 parameter):

```
STRAGGLER_SIGMA = 0.014   (per-node step-time CV)

if (numNodes > 1):
  overhead = σ × √(2·ln(N))
  scaleOverhead = timing.total × overhead
  timing.total += scaleOverhead
```

**Properties:**
- Zero at N=1 (single node, no synchronization barrier)
- Continuous from N=2 (no threshold cliff)
- Monotonically increasing, concave (diminishing marginal overhead)
- `Math.log` = natural log (ln), not log2

The order-statistics model uses a single measurable parameter (σ) rather than multiple
fitted constants.

| Nodes | Overhead (σ=0.014) |
|-------|-------------------|
| 1     | 0%                |
| 32    | 3.3%              |
| 128   | 4.4%              |
| 256   | 4.7%              |
| 512   | 4.9%              |
| 2048  | 5.5%              |

---

## 7. Communication Overhead Model

**Source:** `overlap.ts:applyProtocolOverhead()`

Two-term model applied to raw bandwidth-derived communication times:

```
commTimeWithOverhead = raw × (1 + protocolOverhead[type]) + numCollectives × perCollectiveOverhead
```

**Term 1 — Proportional overhead**: Per-byte NCCL protocol framing. Scales with
message size. Captures ring/tree coordination, serialization, RDMA setup per chunk.

**Term 2 — Fixed per-collective overhead**: Per-collective setup/sync cost. Dominates
for small messages; amortized for large ones. Two tiers:
- **IB (cross-node)**: 50µs per collective — QP setup, barrier, buffer allocation
- **NVLink (intra-node)**: 5µs per collective — SM-to-SM setup, 10× lower than IB

When `raw === 0` (single-GPU, dp=1), returns 0 — no overhead for non-existent communication.

### Protocol overhead multipliers

| Type | Multiplier | Context | Source |
|------|-----------|---------|--------|
| TP (NVLink) | 1.05 | All | `overlap.ts:PROTOCOL_OVERHEAD.tp_nvlink` |
| TP (cross-node) | 1.15 | All | `overlap.ts:PROTOCOL_OVERHEAD.tp_crossnode` |
| PP | 1.05 | All PP paths (unified) | `overlap.ts:PROTOCOL_OVERHEAD.pp` |
| DP (FSDP/ZeRO-3) | 1.10 | All | `overlap.ts:PROTOCOL_OVERHEAD.dp_fsdp` |
| DP (DDP/ZeRO-1/2) | 1.20 | All | `overlap.ts:PROTOCOL_OVERHEAD.dp_ddp` |
| EP | 1.10 | All | `overlap.ts:PROTOCOL_OVERHEAD.dp_ep` |

### Per-collective overhead (PER_COLLECTIVE_OVERHEAD_MS = 0.050)

| Strategy | numCollectives | Notes |
|----------|---------------|-------|
| TP | 2 × denseLayers + 1 × moeLayers (per stage) | Dense: attention + MLP; MoE with EP>1: attention only (expert FFN uses EP, not TP) |
| FSDP/ZeRO-3 | numLayers × 3 | AllGather(fwd) + AllGather(bwd) + ReduceScatter per layer |
| DDP/ZeRO-1/2 | numBuckets = ceil(gradBytes / 25MB) | One AllReduce per 25MB bucket |
| PP | 1 | Single P2P transfer per transfer event |
| SP | 2 × numLayers | AllGather + ReduceScatter per layer |

**Gradient bucketing**: DDP and ZeRO use 25 MB default bucket size for AllReduce
chunking (`BUCKET_SIZE_BYTES` in `overlap.ts`).

**Inference TP AllReduce**: Uses tree-round alpha with layer-pipelining overlap —
independent constants from training (5us/round NVLink, 15us/round PCIe). See
`docs/INFERENCE.md` for the full model.

**Hierarchical AllReduce** (`collectives/cost-model.ts`): When DP spans multiple
nodes, intra-node and inter-node phases overlap by 30% with a combined efficiency
factor of 0.9.

### Cross-node TP bandwidth model

When TP spans multiple nodes, the simulator uses a hierarchical all-reduce model:

```
G = gpusPerNode
N = ceil(tp / G)

effectiveBW = ((tp-1)/tp) / ((G-1)/G / nvlinkBW + (N-1)/N / ibBW)
```

This models: ReduceScatter(NVLink) + AllReduce(IB) + AllGather(NVLink).

**Inference** uses the same bandwidth formula (`latency.ts:calculateLatencyWithTP()`).
The alpha model differs: inference uses per-phase tree-round latency
(`nvlinkAlpha × 2×2×ceil(log2(G)) + ibAlpha × 2×ceil(log2(N))`, where ibAlpha = 25us)
while training uses a flat per-collective overhead (`PER_COLLECTIVE_OVERHEAD_MS = 50us`
in `overlap.ts`). Both converge for N=2 (2 IB rounds × 25us = 50us).

### FSDP standalone overhead

**Source:** `fsdp.ts:computeTiming()`

Standalone FSDP uses a comm overhead of `1.10` (AllGather/ReduceScatter: moderate
NCCL protocol overhead).

---

## 8. Overlap Models

The overlap model uses a physics-based approach: theoretical timeline overlap ×
single scheduling efficiency η=0.96. Per-strategy differences emerge from
timeline geometry (prefetching, bucketing, pipelining), not from tuned constants.

**Source:** `src/core/strategies/overlap.ts`

Net overlap constants: 3 (η=0.96 + BUCKET_SIZE_BYTES=25MB + PER_COLLECTIVE_OVERHEAD_MS=0.050)
\+ 6 protocol overhead multipliers.

### TP layer-level overlap

**Source:** `overlap.ts:computeTPOverlap()`, `3d-parallel.ts`, `tensor-parallel.ts`,
`sequence-parallel.ts`

TP communication is pipelined with compute at the layer level -- while one layer
computes, the all-reduce for the previous layer completes.

```
C = computePerMB                       (forward + backward time per microbatch)
T = (tpCommTime / pp) * tpOverhead     (per-PP-stage TP comm with overhead)

tpOverlapEff = C/(C+T) × η
exposedTPPerStep = T * (1 - tpOverlapEff) * gradientAccumulationSteps
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| η | 0.96 | Scheduling efficiency (all strategies) |

With SP enabled, communication ops change from AllReduce to AllGather + ReduceScatter,
and T reflects the actual volume. The scheduling efficiency η inherently limits
overlap below 1.0.

For large models (C >> T): overlap approaches η=0.96, exposed is ~4% of T per microbatch.
For small models (C ~ T): overlap is ~0.50, exposed is ~50% of T per microbatch.

### DP prefetch overlap (FSDP/ZeRO-3)

**Source:** `overlap.ts:computeFSDPExposedComm()`, `3d-parallel.ts`, `fsdp.ts`

FSDP/ZeRO-3 pipelines AllGather of the next layer's weights with backward compute
of the current layer. Layer 0's AllGather cannot be overlapped, giving (N-1)/N
theoretical coverage where N = numLayers.

Per-layer pipeline simulation (`computeFSDPExposedComm()`): each layer's AllGather
and ReduceScatter overlap with adjacent layers' compute, degraded by scheduling
efficiency η.

**Forward (layer 0 → L-1):**
- Layer 0: AG is cold start — fully exposed (no prior compute to overlap with)
- Layer i>0: AG(i) overlaps with compute(i-1) × η

**Backward (layer L-1 → 0):**
- Layer L-1: AG is cold start — fully exposed
- Layer i<L-1: AG(i) overlaps with backward(i+1) × η
- RS(0) is fully exposed (cold start); RS(i>0) overlaps with backward(i-1) × η

```
exposedPerMB = fwdExposed + bwdExposed

fwdExposed = AG + Σ_{i=1}^{L-1} max(0, AG - fwdCompute[i-1] × η)
bwdExposed = AG + RS
           + Σ_{i=0}^{L-2} max(0, AG - bwdCompute[i+1] × η)
           + Σ_{i=1}^{L-1} max(0, RS - bwdCompute[i-1] × η)
```

| Condition | η | Notes |
|-----------|---|-------|
| With backward prefetch (FSDP2 default) | 0.95 | CUDA stream sync + prefetch buffer latency |
| Without backward prefetch (FSDP1) | 0.80 | Manual scheduling, less overlap opportunity |

In the compute-bound regime (common case — large models): all max(0,...) terms
vanish, so exposedPerMB = 2×AG + RS, independent of model depth.

### DP bucket overlap (DDP / ZeRO-1 / ZeRO-2)

**Source:** `overlap.ts:computeDDPOverlap()`, `overlap.ts:computeZeROGradOverlap()`

PyTorch DDP fires AllReduce per bucket (25MB default). The first bucket fires
after the first few backward layers finish. ZeRO-1/2 use the same bucket model
for the gradient sync portion; param AllGather after optimizer is sequential.

```
numBuckets = ceil(gradientBytes / BUCKET_SIZE_BYTES)
firstBucketDelay = 1 / numBuckets
theoreticalOverlap = min(1, backwardTime × (1 - firstBucketDelay) / commTime)
overlapTime = commTime × theoreticalOverlap × η
```

Returns absolute overlap time (ms), not a fraction. The bucket model naturally
produces lower overlap for small models (few buckets, high first-bucket delay)
and higher overlap for large models (many buckets, low first-bucket delay).

### 3D parallel DP bandwidth detection

**Source:** `3d-parallel.ts:computeTiming()`

For 3D strategies, the DP communication bandwidth is structurally detected from
the cluster topology rather than hardcoded to inter-node bandwidth:

```
stagesPerNode = floor(gpusPerNode / tp)        (already computed for PP)
dpRanksPerNode = stagesPerNode >= pp ? floor(stagesPerNode / pp) : 0
baseDPBW = dpRanksPerNode >= dp
  ? cluster.node.intraNodeInterconnect.bandwidthGBps   (all DP intra-node → NVLink)
  : cluster.interNodeBandwidthGBps                      (DP crosses nodes → IB)
```

This mirrors the PP placement pattern (`stagesPerNode >= pp → NVLink, else IB`).
When all DP ranks fit within one node (e.g., small DP degree), NVLink bandwidth is
used instead of IB, correctly reflecting that no inter-node traffic is needed.

### PP overlap

**Source:** `overlap.ts:computePPOverlap()`, `3d-parallel.ts:computeTiming()`

PP point-to-point transfers overlap with compute using the same C/(C+T) × η model:

```
computePerMB = forwardTimePerMB + backwardTimePerMB
ppCommPerMB = 2 * v * ppCommPerTransfer

ppOverlapEff = C/(C+T) × η

ppTransitionsPerMB = 2 × v × (pp - 1)           // interleaved-1F1B
                   = 2 × (pp - 1)               // DualPipe-V (bidirectional overlap)
ppTransitionOverhead = ppTransitionsPerMB × PP_STAGE_TRANSITION_MS

ppExposedPerStep = ppCommPerMB × (1 - ppOverlapEff) × GA
                 + ppTransitionOverhead × GA
```

Where `v` is virtual stages: 1 for standard 1F1B, `interleavedStages` for interleaved
1F1B, 2 for DualPipeV. Each virtual-stage boundary incurs a non-overlappable transition
(NCCL P2P setup + stream sync + kernel launch ≈ 20µs). DualPipe-V's bidirectional
schedule overlaps fwd/bwd transitions, so serial transitions = 2×(pp-1) regardless of v.

PP communication is nearly free for large models (C >> T) because each microbatch's
forward/backward time dwarfs the activation transfer time.

### PP bandwidth model

```
stagesPerNode = floor(gpusPerNode / tp)
perNicBW = perPortBandwidthGBps   (InfiniBand) or aggregate / numNICs (others)
concurrentP2PStreams = min(pp - 1, numNICs)
ppPerGPUBW = min(perNicBW, interNodeBandwidthGBps / concurrentP2PStreams)

ppBandwidth =
  all intra-node (stagesPerNode >= pp):  intraNodeBW
  all cross-node (stagesPerNode <= 1):   ppPerGPUBW
  mixed:                                 min(intraNodeBW, ppPerGPUBW)
```

Cross-node PP bandwidth is limited by NIC contention: each P2P stream uses one NIC,
so concurrent streams share the aggregate fabric bandwidth. For DGX (8 NICs, 8 GPUs)
with NDR: `min(50, 400/min(pp-1, 8))` = 50 GB/s — numerically identical to the
previous `400/8 = 50` formula. The `numNICs` field enables non-DGX topologies where
GPU count and NIC count differ.

### CP timing model

**Source:** `3d-parallel.ts:computeTiming()`

Two CP implementations are supported:

#### Ring attention (`cpImplementation: 'ring'`)

A CP-step pipeline within the attention layer. At each step, KV P2P transfer overlaps
with blockwise attention compute (QK^T + softmax + scores×V) on one KV chunk.
Step 0 uses local KV (no transfer). Steps 1..CP-1 each overlap one KV transfer with
one chunk's attention compute.

Per-step compute varies due to causal masking. `cpCausalWorkDistribution(cp)` decomposes
the CP-1 transfer steps into two types:

- **1 diagonal step**: The self-attention chunk has a triangular causal mask, so only
  50% of the FLOPs execute (`diagonalComputeFraction = 0.5`).
- **CP-2 normal steps**: Full cross-attention chunks with no masking.

```
chunkSeq = seqLength / cp

attnFLOPsPerChunk = 2 * numHeads * chunkSeq² * (qkDim + vDim) * MBS / TP
computeTimeFwd = attnFLOPsPerChunk / effectiveTFLOPS
computeTimeBwd = computeTimeFwd * 2

jitterFloor = 0.03 + 0.005 * (CP - 1)  // 3.5% at CP=2, 10.5% at CP=16

// Diagonal step (causal mask → half compute, harder to hide transfer)
exposedDiagonalFwd = max(transferTime * jitterFloor,
                         transferTime - computeTimeFwd * 0.5)
exposedNormalFwd   = max(transferTime * jitterFloor,
                         transferTime - computeTimeFwd)
exposedFwd = 1 * exposedDiagonalFwd + (CP - 2) * exposedNormalFwd

exposedDiagonalBwd = max(transferTime * jitterFloor,
                         transferTime - computeTimeBwd * 0.5)
exposedNormalBwd   = max(transferTime * jitterFloor,
                         transferTime - computeTimeBwd)
exposedBwd = 1 * exposedDiagonalBwd + (CP - 2) * exposedNormalBwd

fwdPasses = activationCheckpointing ? 2 : 1
cpExposedPerStep = (fwdPasses * exposedFwd + exposedBwd) * layersPerGPU * GA
```

P2P BW efficiency: 0.90 (NVLink) / 0.85 (IB). CP comm uses BF16 floor (same as PP).
The jitter floor prevents the model from predicting zero overhead at high CP degrees
where multi-peer coordination introduces scheduling fragility.

#### All-gather CP (`cpImplementation: 'all-gather'`)

Used by Meta and Megatron-LM. AllGather KV before attention — fully exposed, no
per-step overlap. Simpler than ring attention but higher communication overhead.

```
allGatherVolume = kvBlockBytes * (CP - 1)
collectiveEff = cpWithinNode ? 0.90 : 0.82  // NCCL ring all-gather
allGatherTime = allGatherVolume / (cpBandwidth * collectiveEff)

// Partial overlap with non-attention compute from previous microbatch
overlap = isInterleaved ? 0.15 : 0.05
exposedAllGather = allGatherTime * (1 - overlap)

// Forward only — backward dK/dV is local (no ReduceScatter needed)
cpExposedPerStep = fwdPasses * exposedAllGather * layersPerGPU * GA
```

**405B CP=16 131K:** Ring → ~19.4% PaLM MFU (45.5% Model FLOPs MFU).
All-gather → ~16.4% PaLM MFU (38.3% Model FLOPs MFU, published: 38%).

### MoE physics floor

**Source:** `3d-parallel.ts:computeTiming()`

The physics floor is the minimum forward time dictated by peak throughput for total
per-GPU FLOPs. For MoE models, expert and non-expert FLOPs have different parallelism
divisors and throughput factors:

```
EP > 1 (dual-floor model):
  nonExpertFloor = tokens × (flopsPerToken - routedExpertFlops) / (tp × pp) / peakTFLOPS
  expertFloor    = tokens × routedExpertFlops / (ep × pp) / (peakTFLOPS × groupedGemmFactor)
  physicsFloor   = nonExpertFloor + expertFloor

EP = 1:
  baseFloor    = tokens × flopsPerToken / (tp × pp) / peakTFLOPS
  physicsFloor = baseFloor × (1 + routedFlopFraction × (1/groupedGemmFactor - 1))
```

When EP > 1, experts are EP-distributed (÷ep) while non-expert layers are TP-sharded
(÷tp). Each component's floor uses its actual per-GPU FLOP count and throughput. The
expert floor includes the grouped GEMM efficiency penalty. When EP = 1, all FLOPs
(including routed experts) are TP-sharded, so a single floor is weighted by the routed
FLOP fraction.

### EP slack overlap

**Source:** `3d-parallel.ts:computeTiming()`

When the physics floor exceeds the per-layer EP compute time, the GPU has "slack"
compute cycles where EP all-to-all transfers can be overlapped:

```
floorSlack = max(0, physicsFloor - computeOnlyForward)
epOverlapWithSlack = min(epCommForward, floorSlack * η)
epCommExposedFwd = epCommForward - epOverlapWithSlack
```

The η factor (schedulingEfficiency = 0.96) is the same CUDA stream scheduling
efficiency used for all comm/compute overlap (TP, DP, PP). All-to-all is a
barrier (dispatch must complete before expert compute starts), so overlap is
limited to the slack window — not pipelined like DP ring-reduce.

---

## 9. Backward Multipliers

**Source:** all strategy files (`3d-parallel.ts`, `pipeline-parallel.ts`, `fsdp.ts`,
`ddp.ts`, `zero.ts`, `tensor-parallel.ts`, `sequence-parallel.ts`)

The backward multiplier is the ratio of backward time to forward time. It accounts
for: (1) activation gradient computation, (2) weight gradient computation, and
(3) optional forward recompute for activation checkpointing.

### Full training

| Granularity | Multiplier |
|-------------|------------|
| No checkpointing | 2.00 |
| Selective | 2.0 + f |
| Full | 2.85 |

Where `f = getSelectiveRecomputeFraction(model)` is the fraction of per-layer FLOPs
from attention linear projections (Q, K, V, O matmuls). Typical values: GQA models
13-22%, MHA models 25-33%. Computed from model architecture in `base.ts`.

**All strategies use 2.85 for full checkpointing.** FSDP/ZeRO-3 AllGather/recompute
overlap is credited via `computeFSDPExposedComm()` as a communication overlap factor,
not as a reduced backward multiplier.

**EP communication in backward:** For MoE models with EP>1, EP all-to-all
(dispatch+combine) volume is symmetric in forward and backward — gradients traverse
the same tensor shapes as activations. The backward multiplier applies only to compute;
EP comm uses a separate multiplier: 2.0 with activation checkpointing (1× recompute +
1× gradient), 1.0 without (gradient only). This prevents over-counting EP comm in the
backward pass.

**FP8 quantized EP dispatch:** When using FP8 mixed precision with Hopper Transformer
Engine, EP dispatch+combine transfers FP8-quantized activations (volume uses FP8 bytes).
The BF16→FP8 quantization before dispatch and FP8→BF16 dequantization after combine add
memory-bandwidth-bound overhead: 2 critical-path conversions per MoE layer per direction,
partially overlapped with NCCL transfer via CUDA stream pipelining (~50% hidden).

**Why selective multiplier is model-dependent:** Selective checkpointing discards
attention activations (Q/K/V/O projection outputs, attention scores, dropout masks).
During backward, the four linear projections are recomputed from the saved LN1 input.
The recompute cost depends on the ratio of attention to MLP FLOPs, which varies by
architecture: GQA models have fewer KV heads (smaller recompute), gated MLP models
have larger MLP (smaller relative attention fraction).

**Stored-layers blended multiplier:** When storedLayers K < N, the backward
multiplier blends selective and full AC proportionally. Computed by
`getEffectiveBackwardMultiplier()` in `base.ts`:

```
bwd_mult = (K/N) * (2.0 + f) + ((N-K)/N) * 2.85
```

At K=N (all layers selective): `2.0 + f` (standard selective).
At K=0 (all layers full AC): `2.85` (standard full AC).
Intermediate K values produce a weighted blend. This multiplier is used by all
strategies when selective AC with stored-layers < N is active.

### LoRA/QLoRA

**Source:** `lora.ts:getLoraBackwardMultiplier()`

```
overhead = 1.0 + max(0.05, 2 * trainableParams / totalParams)
```

| Condition | Multiplier |
|-----------|-----------|
| No checkpointing | 1 * overhead = ~1.05 |
| Selective | f + 1 * overhead = ~1.18-1.38 |
| Full | 1 + 1 * overhead = ~2.05 |

Where `f = getSelectiveRecomputeFraction(model)` (typically 0.13-0.33).

The 0.05 floor represents fixed PEFT framework overhead (hooks, adapter forward/backward,
gradient routing). For models >= 7B, even r=64 all_linear produces ~2.3% trainable
params, so `2 * trainable/total < 0.05` and the floor dominates. In practice, the
multiplier is effectively flat.

### CP ring pass counting

**Source:** `3d-parallel.ts:computeTiming()`

```
fwdPasses = activationCheckpointing ? 2 : 1
// Total exposed = fwdPasses × exposedFwd + exposedBwd
```

Without AC: 1 forward ring + 1 backward ring. With AC (full or selective): 2 forward
rings (forward + recompute) + 1 backward ring. Selective AC specifically recomputes
attention, requiring the same ring traversal count as full AC.

---

## 10. Memory Model

### Per-parameter bytes

**Source:** `base.ts:calculateOptimizerMemory()`

| Component | BF16/FP16 (AdamW) | FP8/FP4 (AdamW) | FP32 (AdamW) |
|-----------|------------------|----------------|-------------|
| Parameters | 2 | 1 or 0.5 | 4 |
| Gradients (fp32) | 4 | 4 | 4 |
| Optimizer: master copy | 4 (fp32) | 2 (bf16, TE convention) | 0 (already fp32) |
| Optimizer: momentum (m1) | 4 | 4 | 4 |
| Optimizer: variance (m2) | 4 | 4 | 4 |
| **Total** | **18** | **15-16** | **16** |

DDP stores gradients in **fp32** (4 bytes/param), not bf16.

For other optimizers:

| Optimizer | With master copy | Without master copy |
|-----------|-----------------|-------------------|
| AdamW/Adam | 8 + masterBytes | 8 |
| SGD/Lion | 4 + masterBytes | 4 |
| Adafactor | 4 (no master) | 4 |

### Activation memory per layer

**Source:** `base.ts:estimateActivationMemory()`

```
perLayerActivation = (sharedActivations + attentionActivations + mlpActivations)
```

**Shared activations (always kept):**
```
sharedActivations = 3 * h * tokens * bytesPerElement
```
Three tensors: LN1 input (h), post-attention residual (h), LN2 output (h).

**Attention activations (discarded in selective checkpointing):**
```
attentionActivations = (2*h + qDim + 2*kvDim) * tokens * bytesPerElement
                     + tokens                          (dropout mask, 1 byte each)
                     + attentionScoreMemory
```
Five tensors: LN1 output (h), attention output (h), Q projection (qDim), K projection
(kvDim), V projection (kvDim), plus dropout mask and attention scores.

**Attention score memory:**
- Flash Attention: `numHeads * seqLen * bytesPerElement * microBatchSize` -- O(seq) tile memory
- Standard attention: `numHeads * seqLen^2 * bytesPerElement * microBatchSize * 2.5`
  - The 2.5x accounts for: pre-softmax scores (2B) + post-softmax probs (2B) + dropout
    mask (1B) = 5 bytes per element vs 2 bytes for a single tensor. Per [Korthikanti
    et al. 2022](https://arxiv.org/abs/2205.05198), Table 1.

**MLP activations (always kept):**
```
mlpIntermCoeff = gatedMLP ? 3 : 2   (SwiGLU: gate + up + product; standard: up + activation)
mlpActivations = mlpIntermCoeff * intermediateSize * tokens * bytesPerElement
               + tokens                                    (dropout mask)
```

**GQA/MQA:** K and V projections use `numKvHeads * headDim` (smaller than numAttentionHeads * headDim
for grouped/multi-query attention).

### Checkpointing modes

```
Full:      sqrt(N) layers * max(densePerLayer, moePerLayer)
Selective: all N layers * (shared + MLP)  +  1 layer attention (transient recompute)
None:      all N layers * full per-layer
```

**Selective checkpointing stores:**
- Every layer: shared activations (3h) + MLP intermediates (kept for backward)
- NO sqrt(N) approximation -- every layer stores its MLP + shared portion
- During backward, one layer's attention activations are recomputed transiently

### Stored-Layers Selective AC (Budget Mode)

When selective AC activations exceed GPU memory, the simulator auto-resolves a
stored-layers count K <= N. Layers 1..K use selective checkpointing (store
MLP+shared, discard attention). Layers K+1..N use full checkpointing (sqrt(N-K)
segments). This matches OLMo-core's "budget mode" and Megatron-LM's
`--recompute-num-layers`.

**Source:** `base.ts:solveMaxStoredLayers()`, `base.ts:getEffectiveBackwardMultiplier()`

The backward multiplier blends proportionally:

```
bwd_mult = (K/N) * (2.0 + f_selective) + ((N-K)/N) * 2.85
```

Where f_selective is the selective recompute fraction (attention FLOPs / total
per-layer FLOPs), computed by `getSelectiveRecomputeFraction(model)`.

Auto-resolve: for each strategy, the simulator finds max K such that total
memory (params + grads + optimizer + activations(K) + buffers + reserved) <= GPU
capacity. For 3D parallel, resolution is per-pipeline-stage (not global K / PP).

Activation memory with stored-layers:

```
activationMem(K) = K * selectivePerLayer
                 + sqrt(N - K) * fullPerLayer
                 + transient (one layer's attention activations for recompute)
```

Where:
- `selectivePerLayer = sharedActivations + mlpActivations` (no attention)
- `fullPerLayer = max(densePerLayer, moePerLayer)` (full layer activations)
- `transient = attentionActivations` (one layer during backward recompute)

The solver (`solveMaxStoredLayers`) iterates K from N down to 0, returning the
largest K that fits within the available activation budget. O(N) for N<=200
layers; the loop body is pure arithmetic.

### SP activation scaling

**Source:** `3d-parallel.ts:computeMemoryPerGPU()`

With Sequence Parallelism (SP): all activation tensors are effectively `1/tp` per rank.
TP-sharded tensors split along hidden dim, SP-sharded tensors split along sequence dim.

```
SP enabled:   activationMultiplier = 1 / tp
SP disabled:  activationMultiplier = (sharded/tp + replicated) / (sharded + replicated)
              where replicated = 4h (LN inputs, residuals)
```

### Reserved memory

**Source:** `base.ts:calculateReservedMemory()`, `derived.ts:reservedMemoryDecomposition()`

```
reserved = CUDA_context + 7% × GiB_capacity
```

- CUDA context: ~1 GB — CUDA runtime, cuDNN workspace,
  framework overhead. Estimated from `torch.cuda.mem_get_info()` deltas on empty context.
- 7% fragmentation: PyTorch caching allocator fragmentation, variable tensor sizes,
  gradient accumulation patterns. Calibrated against published configs (GPT-3 at 91.3%,
  LLaMA 405B at 94.9% memory utilization).
- GiB/GB gap: handled by `gpuCapacityBytes` (`gpuMemoryGB × 1024³`) in the capacity
  model, not in reserved memory.

### FSDP gather buffers

During forward/backward, FSDP must all-gather full layer parameters from DP peers:

```
gatheredParamBuffer = paramsPerLayer / tp * DTYPE_BYTES[paramDtype]
prefetchBuffers = gatheredParamBuffer * 2           (prefetch 2 layers ahead)
```

### MoE activation overhead

```
MOE_CAPACITY_FACTOR = 1.15    (headroom for imperfect load balancing)

routerLogits = tokens * numExperts * DTYPE_BYTES[activationDtype]
dispatchBuffers = tokens * numActiveExperts * 4     (int32 indices)
moeOverhead = (routerLogits + dispatchBuffers) * moeLayersPerStage * 1.15
```

---

## 11. Pipeline Bubble Formulas

**Source:** `3d-parallel.ts:calculateBubble()`, `pipeline-parallel.ts:calculateBubble()`

All formulas express the fraction of pipeline wall-clock time that is idle.

### Schedules

| Schedule | Bubble formula | Notes |
|----------|---------------|-------|
| GPipe | `(pp-1) / (pp-1 + m)` | All forward, then all backward |
| 1F1B | `(pp-1) / (pp-1 + m)` | One Forward One Backward interleaved |
| Interleaved 1F1B | `(pp-1) / (pp-1 + m*v)` | v = virtual stages per device |
| DualPipeV | `(pp-1) / (pp-1 + 6*m)` when m >= 2*pp | V-shape bidirectional pipeline |
| DualPipeV (fallback) | `(pp-1) / (pp-1 + m)` when m < 2*pp | Falls back to standard |
| Zero-bubble | `0.05` | Near-zero (small scheduling overhead) |

Where `m = numMicroBatches` and `pp = pipeline stages`.

All formulas are bounded in [0, 1). The formula `(pp-1)/(pp-1+m)` is always <1.0 for
any positive m, so no clamping is needed.

### In-flight microbatches (peak activation memory)

| Schedule | In-flight | Impact |
|----------|----------|--------|
| GPipe | m | All microbatches stored during forward |
| 1F1B | pp | At most pp active per stage |
| Interleaved 1F1B | ceil(pp/v) | Fewer per virtual stage |
| DualPipeV | 2*pp + 1 | V-shape requires 2 streams |

### Bubble examples

| pp | m | Schedule | Bubble |
|----|---|----------|--------|
| 4 | 8 | 1F1B | 27.3% |
| 4 | 16 | 1F1B | 15.8% |
| 8 | 16 | 1F1B | 30.4% |
| 8 | 16 | Interleaved (v=4) | 9.9% |
| 12 | 24 | Interleaved (v=8) | 5.4% |
| 16 | 144 | DualPipeV | 1.7% |

---

## 12. Optimizer Step Timing

**Source:** `base.ts:computeOptimizerStepTime()`

The optimizer step is **memory-bandwidth-bound**: it streams all states through HBM
(read gradients, momentum, variance, master weights; compute updates; write back).

```
BW_EFFICIENCY = 0.75

totalBytes = paramsPerGPU * bytesPerParam
effectiveBW = gpu.memoryBandwidthTBps * 1e12 * 0.75
time_ms = (totalBytes / effectiveBW) * 1000
```

The 0.75 BW efficiency is empirical -- PyTorch optimizer kernels do not saturate
HBM due to mixed read/write patterns and small tensor operations.

### Bytes per parameter by optimizer

| Optimizer | With master copy | Without master copy |
|-----------|-----------------|-------------------|
| AdamW/Adam | 30 | 28 |
| SGD/Lion | 22 | 20 |
| Adafactor | 22 | 20 |

Master copy is needed when `paramDtype != fp32/tf32`. FP8/FP4 use BF16 master (2 bytes)
instead of FP32 master (4 bytes), but the optimizer still reads/writes the momentum and
variance in fp32.

For AdamW with master copy (most common):
- Read: grads(4) + m1(4) + m2(4) + master(4) = 16
- Write: m1(4) + m2(4) + master(4) + params(2) = 14
- Total: 30 bytes/param

### paramsPerGPU calculation

**Source:** `base.ts:getOptimizerParamsPerGPU()`

For MoE models with EP:
```
shared = sharedParams + routerParams
expert = expertParams

sharedShard = tp * pp * (dpType != 'ddp' ? dp : 1)
expertShard = tp * pp * ep * (dpType != 'ddp' ? expertDPReplicas : 1)

paramsPerGPU = shared / sharedShard + expert / expertShard
```

Where `expertDPReplicas = max(1, floor(dp / ep))` -- the number of replicas within each
EP subgroup.

---

## 13. Full Penalty Audit

Every multiplicative and additive factor in the simulator, organized by what it affects.
Factors are orthogonal -- no double-counting. Compute efficiency captures GPU kernel
utilization; communication volumes, bandwidths, and overlaps are modeled separately.

### Compute efficiency factors (affect `effectiveTFLOPS`)

| Factor | Value | Tier | Trigger | Position | Source |
|--------|-------|------|---------|----------|--------|
| Runtime residual (dense) | 0.655 | fitted | Dense models (or non-3D strategies) | Multiplicative | `base.ts:computeComputeEfficiency()` |
| Runtime residual (MoE) | 0.97 | fitted | MoE models via 3D parallel | Multiplicative | `base.ts:computeComputeEfficiency()` |
| Compute saturation | 0 to 1.0 | fitted | Small per-GPU GEMMs | Multiplicative | `gpu.ts:getComputeSaturationFactor()` |
| memBW scaling | ~0.87 to ~1.08 | grounded-empirical | Always | Multiplicative | `gpu.ts:getMemoryBandwidthScaling()` |
| TE overhead | 0.93 (FP8), 0.90 (FP4) | grounded-empirical | FP8/FP4 dtype | Multiplicative on TFLOPS | `gpu.ts:getTrainingTFLOPS()` |
| MoE dispatch overhead | 0 (zeroed pending high-EP anchor) | fitted | MoE models, EP>1 | Additive per MoE layer | `3d-parallel.ts:computeTiming()` |

### Expert compute factors (affect MoE expert time)

| Factor | Value | Tier | Trigger | Cap | Source |
|--------|-------|------|---------|-----|--------|
| Expert GEMM roofline | 0 to 1.0 | physics | MoE, small GEMMs | None | `base.ts:getExpertGemmRooflineFactor()` |
| Expert GEMM saturation | 1.0 to 1.5 | grounded-empirical | MoE, small per-expert matrices | 1.5× | `3d-parallel.ts:computeTiming()` |
| Grouped GEMM efficiency | 0.30 to 0.952 | fitted | MoE, expertInt < 1536 | baseline 0.952 | `gpu.ts:getGroupedGemmEfficiency()` |
| Load imbalance | 1.0 for EP>1 (zeroed pending validation), 1.02 for EP=1 | fitted / physics | MoE | None | `3d-parallel.ts:computeTiming()` |
| Router overhead | tokens * H * E * 2 / TFLOPS | physics | MoE | None | `3d-parallel.ts:computeTiming()` |
| Permutation overhead | 4 * tokens * H * bytes / memBW | physics | MoE | None | `3d-parallel.ts:computeTiming()` |

### Communication volume factors

| Factor | Value | Tier | Trigger | Affects | Source |
|--------|-------|------|---------|---------|--------|
| FP8 TP comm | 1 byte/elem | physics | FP8 + TE (Hopper+) | TP comm volume | `3d-parallel.ts` |
| BF16 floor | 2 bytes/elem | physics | PP/CP always | PP/CP comm volume | `3d-parallel.ts` |
| Routing locality | density or device-limited | physics | ep > 1 | EP all-to-all volume | `3d-parallel.ts` |
| DP ring factor | (n-1)/n | physics | dp > 1 | DP comm volume | `3d-parallel.ts`, `fsdp.ts` |

### Communication overhead (two-term model, affects comm time)

Applied via `applyProtocolOverhead(raw, type, numCollectives, intraNode)`:
`raw × (1 + protocolOverhead[type]) + numCollectives × perCollectiveOverhead`

| Factor | Value | Tier | Trigger | Source |
|--------|-------|------|---------|--------|
| TP overhead (NVLink) | 1.05× + 5µs/coll | grounded-empirical | tp > 1, intra-node | `overlap.ts:applyProtocolOverhead()` |
| TP overhead (cross-node) | 1.15× + 50µs/coll | grounded-empirical | tp > gpusPerNode | `overlap.ts:applyProtocolOverhead()` |
| PP overhead | 1.05× + 5µs or 50µs/coll | grounded-empirical | pp > 1 | `overlap.ts:applyProtocolOverhead()` |
| DP overhead (FSDP/ZeRO-3) | 1.10× + 50µs/coll | fitted | dp > 1 | `overlap.ts:applyProtocolOverhead()` |
| DP overhead (DDP/ZeRO-1/2) | 1.20× + 50µs/coll | fitted | dp > 1 | `overlap.ts:applyProtocolOverhead()` |
| PER_COLLECTIVE_OVERHEAD_MS | 0.050 (IB), 0.005 (NVLink) | fitted | Any comm | `overlap.ts` |
| PP_STAGE_TRANSITION_MS | 0.020 ms/transition | grounded-empirical | pp > 1 | `3d-parallel.ts:computeTiming()` |


### Communication bandwidth degradation

| Factor | Value | Tier | Trigger | Floor | Source |
|--------|-------|------|---------|-------|--------|
| DP BW penalty (tier 1) | 1/(1+0.15*log2(dp/64)) | fitted | dp > 64 | 0.40 | `base.ts:getDPGroupSizePenalty()` |
| DP BW penalty (tier 2) | above × 1/(1+0.08*log2(dp/256)) | grounded-empirical | dp > 256 | 0.40 | `base.ts:getDPGroupSizePenalty()` |
| DP collective latency | 0.030 ms * log2(dp/64) | grounded-empirical | dp > 64 | None | `base.ts:getDPCollectiveLatencyMs()` |
| EP all-to-all BW eff (NVLink) | 0.55 * groupPenalty | grounded-empirical | ep > 1, intra-node | None | `3d-parallel.ts:computeTiming()` |
| EP all-to-all BW eff (IB) | 0.45 * groupPenalty | grounded-empirical | ep > 1, cross-node | None | `3d-parallel.ts:computeTiming()` |
| EP group penalty | 1/(1+0.15*log2(ep/8)) | fitted | ep > 8 | None | `3d-parallel.ts` via `getDPGroupSizePenalty(ep, {ref:8})` |
| EP all-to-all latency (NVLink) | 0.05 ms per collective | physics | ep > 1, intra-node | None | `3d-parallel.ts:computeTiming()` |
| EP all-to-all latency (IB) | 0.10 ms per collective | physics | ep > 1, cross-node | None | `3d-parallel.ts:computeTiming()` |

### Overlap factors (reduce exposed comm time)

Physics-based overlap model. All strategies use η=0.96 scheduling efficiency.
Per-strategy differences emerge from timeline geometry (prefetching, bucketing, pipelining).
Two-term protocol overhead: proportional (per-byte) + fixed (per-collective at 50µs IB / 5µs NVLink).

| Factor | Value | Tier | Trigger | Cap | Source |
|--------|-------|------|---------|-----|--------|
| TP overlap | C/(C+T) × η | physics | tp > 1 | η=0.96 | `overlap.ts:computeTPOverlap()` |
| DP overlap (FSDP/ZeRO-3) | Per-layer pipeline: AG/RS overlap adjacent compute × η | physics | dp > 1 | η=0.95 (prefetch) / 0.80 (no prefetch) | `overlap.ts:computeFSDPExposedComm()` |
| DP overlap (DDP) | bucket model × η | physics | dp > 1 | η=0.96 | `overlap.ts:computeDDPOverlap()` |
| DP overlap (ZeRO-1/2) | bucket model × η (grad sync only) | physics | dp > 1 | η=0.96 | `overlap.ts:computeZeROGradOverlap()` |
| PP overlap | C/(C+T) × η | physics | pp > 1 | η=0.96 | `overlap.ts:computePPOverlap()` |
| CP overlap | Per-step pipeline: max(0, T-C_attn) | physics | cp > 1 | None (physics-based) | `3d-parallel.ts:computeTiming()` |
| CP causal work distribution | Per-step: 1 diagonal (0.5× compute) + (CP-2) normal (1× compute) | physics | cp > 1 | None | `derived.ts:cpCausalWorkDistribution()`, `3d-parallel.ts` |
| EP slack overlap | min(epComm, slack * η) | physics | ep > 1, MoE | η of slack | `overlap.ts:computeEPSlackOverlap()` |

### Pipeline factors

| Factor | Value | Tier | Trigger | Source |
|--------|-------|------|---------|--------|
| 1F1B bubble | (pp-1)/(pp-1+m) | physics | pp > 1 | `3d-parallel.ts:calculateBubble()` |
| Interleaved bubble | (pp-1)/(pp-1+m*v) | physics | pp > 1, interleaved | `3d-parallel.ts:calculateBubble()` |
| DualPipeV bubble | (pp-1)/(pp-1+6*m) | physics | pp > 1, m >= 2*pp | `3d-parallel.ts:calculateBubble()` |
| Zero-bubble | 0.05 | grounded-empirical | pp > 1, zero-bubble | `3d-parallel.ts:calculateBubble()` |

### Backward / timing multipliers

| Factor | Value | Tier | Trigger | Source |
|--------|-------|------|---------|--------|
| Backward (no ckpt) | 2.00 | physics | All strategies | Strategy files |
| Backward (selective) | 2.0 + f | physics | selective + any | `base.ts`, all strategy files |
| Backward (selective, stored-layers) | (K/N)(2.0+f) + ((N-K)/N)(2.85) | physics | selective + storedLayers < N | `base.ts:getEffectiveBackwardMultiplier()` |
| Backward (full) | 2.85 | physics | full + any | All strategy files |
| LoRA backward (no ckpt) | ~1.05 | physics | LoRA/QLoRA | `lora.ts:getLoraBackwardMultiplier()` |
| LoRA backward (selective) | f + ~1.05 | physics | LoRA + selective | `lora.ts:getLoraBackwardMultiplier()` |
| LoRA backward (full) | ~2.05 | physics | LoRA + full | `lora.ts:getLoraBackwardMultiplier()` |
| CP fwd passes | 2 (ckpt) / 1 (no ckpt) | physics | cp > 1 | `3d-parallel.ts:computeTiming()` |
| CP bwd attn multiplier | 2× fwd attn FLOPs | physics | cp > 1 | `3d-parallel.ts:computeTiming()` |

### Memory factors

| Factor | Value | Tier | Trigger | Source |
|--------|-------|------|---------|--------|
| Reserved memory | 1.0 GB + 7% × GiB_capacity | grounded-empirical | Always | `base.ts:calculateReservedMemory()` |
| SP activation scaling | 1/tp | physics | SP enabled | `3d-parallel.ts:computeMemoryPerGPU()` |
| MoE capacity factor | 1.15 | grounded-empirical | MoE models | `3d-parallel.ts:computeMemoryPerGPU()` |
| Optimizer BW efficiency | 0.75 | grounded-empirical | Always | `base.ts:computeOptimizerStepTime()` |
| FSDP prefetch buffers | 2 layers | physics | FSDP/ZeRO-3 | `3d-parallel.ts:computeMemoryPerGPU()` |
| Non-FA attention score multiplier | 2.5x | physics | Flash Attention off | `base.ts:estimateActivationMemory()` |
| MLP intermediate coeff (gated) | 3 | physics | Gated MLP (SwiGLU) | `base.ts:estimateActivationMemory()` |
| MLP intermediate coeff (standard) | 2 | physics | Standard MLP | `base.ts:estimateActivationMemory()` |
| Full ckpt layer count | sqrt(N) | physics | Full checkpointing | `base.ts:estimateActivationMemory()` |
| Stored-layers selective AC | K selective + sqrt(N-K) full | physics | selective + storedLayers < N | `base.ts:estimateActivationMemory()`, `base.ts:solveMaxStoredLayers()` |
| NF4 bytes per param (QLoRA) | 0.515 | physics | QLoRA | `lora.ts:NF4_BYTES_PER_PARAM` |
| QLoRA dequant BW | 2.5 bytes/param | physics | QLoRA | `lora.ts:getQloraDequantTimeMs()` |

### Scale factors

| Factor | Value | Tier | Trigger | Source |
|--------|-------|------|---------|--------|
| STRAGGLER_SIGMA (σ) | 0.014 | grounded-empirical | numNodes > 1 | `base.ts:computeAnalysis()` |

---

### Orthogonality guarantee

These factors are intentionally orthogonal -- there is no double-counting between them:

- **Compute efficiency** captures how well GPU kernels utilize the hardware (SM
  occupancy, warp scheduling, memory access patterns). It does NOT include any
  communication penalties.

- **Communication volumes** are computed from first principles (tensor sizes, ring
  factors, dtype bytes). Communication overhead multipliers account for NCCL protocol
  overhead on top of the pure bandwidth cost.

- **Overlap models** determine what fraction of communication is hidden behind compute.
  Only the exposed (non-overlapped) fraction adds to step time.

- **DP scaling laws** degrade the effective bandwidth and overlap quality at scale.
  They are independent of the compute efficiency model.

- **Scale overhead** adds a separate straggler penalty on top of the fully computed
  step time. It does not interact with any other factor.

- **Pipeline bubble** is a scheduling inefficiency (idle time) that is orthogonal to
  both compute efficiency and communication.

The total step time is:

```
stepTime = forwardTime + backwardTime + optimizerTime
         + sum(exposedComm)           -- non-overlapped communication
         + scaleOverhead              -- straggler penalty at large scale
```

Where `forwardTime` and `backwardTime` already incorporate the pipeline bubble via
`/ pipelineEfficiency`.

### Parameter Census

Source of truth: `src/core/parameter-registry.ts` (52 entries).
CI enforcement: `tests/calibration/parameter-census.test.ts`.

| Tier | Count | Description |
|------|-------|-------------|
| physics | 11 | Derived from first principles or hardware specs |
| grounded-empirical | 28 | Informed by published measurements, narrow range |
| fitted | 13 | Tuned to match benchmark data, may drift |
| **Total** | **52** | |

Of 13 fitted parameters, 8 are perturbable (continuous scalars with get/set
pairs). The sensitivity analysis also perturbs 3 grounded-empirical parameters
(PER_COLLECTIVE_OVERHEAD_MS, PROTOCOL_OVERHEAD DP FSDP, PP_STAGE_TRANSITION_MS),
for 11 total.

### Sensitivity Analysis Results

±5% perturbation across 18 Tier 1+2 benchmarks. Run: `npx tsx scripts/run-sensitivity-analysis.ts`.

| Parameter | Value | Max \|ΔMFU\| (±5%) | Classification |
|-----------|-------|---------------------|----------------|
| runtimeResidual | 0.655 | 5.43pp | high-sensitivity |
| moeRuntimeResidual | 0.970 | 3.56pp | high-sensitivity |
| SCHEDULING_EFFICIENCY | 0.960 | 1.05pp | high-sensitivity |
| DP_BW_ALPHA | 0.150 | 0.16pp | low-sensitivity |
| EXPERT_GEMM_EXPONENT | 0.800 | 0.64pp | low-sensitivity |
| EXPERT_COUNT_SCALE | 0.550 | 0.41pp | low-sensitivity |
| PROTOCOL_OVERHEAD (DP FSDP) | 0.100 | 0.13pp | low-sensitivity |
| PER_COLLECTIVE_OVERHEAD_MS | 0.0500 | 0.09pp | low-sensitivity |
| PP_STAGE_TRANSITION_MS | 0.0200 | 0.07pp | low-sensitivity |
| CP all-gather overlap (interleaved) | 0.150 | 0.05pp | low-sensitivity |
| DP_BW_FLOOR | 0.400 | 0.00pp | structural guard |
| PROTOCOL_OVERHEAD (DP DDP) | 0.200 | 0.00pp | dead-weight |
| ALLTOALL_COORD_OVERHEAD | 0 | 0.00pp | dead-weight |
| EP_LATENCY_ALPHA | 0 | 0.00pp | dead-weight |
| EP_MIN_COORD_MS | 0 | 0.00pp | dead-weight |
| LOAD_BALANCE_DAMPING | 0 | 0.00pp | dead-weight |
| loadImbalance EP=1 | 0.0200 | 0.00pp | dead-weight |
| CP all-gather overlap (non-interleaved) | 0.0500 | 0.00pp | dead-weight |

**Key findings:**
- 3 parameters are high-sensitivity: `runtimeResidual`, `moeRuntimeResidual`,
  and `SCHEDULING_EFFICIENCY`. These are well-constrained — their values
  meaningfully shape MFU predictions.
- 7 parameters are low-sensitivity: measurable effect but <1pp MFU per 5% change.
  Current values are appropriate; could be rounded to clean values without loss.
  Tier 2 MoE benchmarks (Qwen3 30B-A3B, Qwen2 57B-A14B, Mixtral 8x22B) exercise
  `EXPERT_GEMM_EXPONENT` and `EXPERT_COUNT_SCALE` — these have measurable
  sensitivity only when MoE expert-parallel configs are included.
- 7 parameters are dead-weight: truly zero sensitivity across all 18 benchmarks.
  `PROTOCOL_OVERHEAD (DP DDP)` has negligible DP comm; `EP_MIN_COORD_MS` and
  `loadImbalance EP=1` remain at boundary conditions; `CP all-gather overlap
  (non-interleaved)` has no non-interleaved CP benchmark. Four zeroed parameters
  (`ALLTOALL_COORD_OVERHEAD`, `EP_LATENCY_ALPHA`, `EP_MIN_COORD_MS`,
  `LOAD_BALANCE_DAMPING`) are pending anchor data.
- 1 parameter is a structural guard: `DP_BW_FLOOR` exists as an extrapolation
  safety net that binds at DP ≈ 92700 — unreachable in any practical config.
  It is intentionally unconstrainable.

### Inference Sensitivity Analysis

11 perturbable inference parameters: 0 physics, 5 grounded-empirical, 5 fitted,
plus 1 memory-only (grounded-empirical). ±5% perturbation across 5 inference
benchmarks measuring throughput, TPOT, and TTFT deltas.

Run: `npx tsx scripts/run-inference-sensitivity-analysis.ts`.

**Benchmarks:** LLaMA 8B 1×H100 BF16 B=32, LLaMA 70B 4×H100 TP=4 BF16 B=16,
DeepSeek V3 8×H200 TP=4 EP=2 FP8 B=32, LLaMA 70B 1×H200 INT4 B=8,
LLaMA 8B 1×H100 CB B=64.

| Parameter | Value | Tier | Max \|ΔThroughput\| | Max \|ΔTPOT\| | Max \|ΔTTFT\| | Classification |
|-----------|-------|------|---------------------|---------------|---------------|----------------|
| PREFILL_RESIDUAL | 0.400 | fitted | 4.85% | 0.00% | 10.03% | high-sensitivity |
| BW_EFF_FLOOR | 0.350 | fitted | 4.57% | 4.73% | 0.00% | high-sensitivity |
| BW_EFF_SCALE | 5.00 | fitted | 1.06% | 1.18% | 0.00% | high-sensitivity |
| TP_COMM_EFFICIENCY | 0.800 | grounded-empirical | 0.53% | 0.38% | 1.45% | conditional |
| EP_PREFILL_OVERLAP | 0.150 | fitted | 0.20% | 0.00% | 1.42% | conditional |
| CB_SCHEDULING_BASE | 0.010 | grounded-empirical | 0.14% | 0.15% | 0.00% | conditional |
| DECODE_SAMPLING_OVERHEAD_MS | 0.030 | grounded-empirical | 0.04% | 0.04% | 0.00% | low-sensitivity |
| CB_PREFILL_INTERFERENCE_MAX | 0.100 | fitted | 0.02% | 0.00% | 0.91% | conditional |
| NVLINK_PER_ROUND_MS | 0.005 | grounded-empirical | 0.01% | 0.02% | 0.00% | conditional |
| PCIE_PER_ROUND_MS | 0.015 | grounded-empirical | 0.00% | 0.00% | 0.00% | conditional (no PCIe benchmark) |
| MEMORY_OVERHEAD_FACTOR | 0.100 | grounded-empirical | 0.00% | 0.00% | 0.00% | structural zero |

**Key findings:**
- 3 parameters are high-sensitivity: `PREFILL_RESIDUAL` (fitted, dominates TTFT),
  `BW_EFF_FLOOR` (fitted, dominates TPOT for small models), and `BW_EFF_SCALE`
  (fitted, shapes the bandwidth efficiency sigmoid transition). All three are
  fitted parameters — the most impactful inference constants are the least
  directly measurable.
- 5 parameters are conditional: zero sensitivity on configs that don't use the
  feature (TP params zero at TP=1, CB params zero without continuous batching,
  EP params zero without expert parallelism, PCIe alpha zero on NVLink GPUs).
- 1 parameter is low-sensitivity: `DECODE_SAMPLING_OVERHEAD_MS` (grounded-empirical,
  0.03ms fixed overhead per decode step — negligible relative to memory/compute time).
- 1 parameter is a structural zero: `MEMORY_OVERHEAD_FACTOR` affects memory
  sizing only, not throughput/TPOT/TTFT.

**Inference physics notes:**
- `getBandwidthEfficiency()` takes total HBM bytes read per step (weights + KV
  cache) — bus saturation depends on total streaming volume, not weights alone.
- `decodeFLOPs()` and `prefillFLOPs()` include attention score FLOPs (QK^T +
  scores×V) that scale with sequence length. Impact: negligible at typical
  decode (<4K context), significant at 32K+ prefill (+25% FLOPs).
