# Inference Simulation Reference

Reference for all inference latency, memory, KV cache, speculative decoding,
continuous batching, and optimizer models used by the ML Cluster Simulator.

> **See also:** [HARDWARE.md](HARDWARE.md) (GPU specs) | [MODELS.md](MODELS.md) (architectures) | [BENCHMARKS.md](BENCHMARKS.md) (inference benchmarks)

## Table of Contents

1. [Two-Phase Model](#1-two-phase-model)
2. [TTFT Calculation](#2-ttft-calculation)
3. [TPOT Calculation](#3-tpot-calculation)
4. [KV Cache](#4-kv-cache)
5. [Memory Breakdown](#5-memory-breakdown)
6. [Speculative Decoding](#6-speculative-decoding)
7. [Continuous Batching](#7-continuous-batching)
8. [Inference Optimizer](#8-inference-optimizer)
9. [Roofline Model and Utilization](#9-roofline-model-and-utilization)

---

## 1. Two-Phase Model

LLM inference has two distinct phases with fundamentally different hardware
bottlenecks:

- **Prefill** (compute-bound): Processes all input tokens in parallel. Produces
  the first output token. Measured by **TTFT** (Time to First Token). Each token
  requires ~2 FLOPs per parameter, and the entire prompt is batched into a
  single large matmul.

- **Decode** (memory-bandwidth-bound): Generates output tokens one at a time.
  Each step reads all model weights plus the KV cache. Measured by **TPOT**
  (Time Per Output Token). The arithmetic intensity is low, so decode is
  dominated by how fast the GPU can stream bytes from HBM.

The simulation takes `max(compute_time, memory_time)` at each phase.

Source: `src/core/inference/latency.ts`

---

## 2. TTFT Calculation

`estimateTTFT()` computes the time to process the entire input prompt.

```
prefillFLOPs = 2 * activeParams * inputSeqLen * batchSize
ttft_ms = max(computeTime, memoryTime)
```

- `computeTime = prefillFLOPs / (gpuTFLOPS * 1e12 * saturationFactor * MFU)`
- `memoryTime = weightBytes / memoryBandwidth_bytes_per_sec`
- MFU: **0.40** × bandwidth scaling (= 0.40 at H100 SXM reference); `saturationFactor` from `getMatmulSaturationFactor()`

**MoE models**: Uses `activeParams` for compute FLOPs. Weight bytes read per
step use a coupon-collector formula over the number of tokens processed —
different for prefill vs decode:

- **Prefill**: `batchSize × inputSeqLen` tokens routed through experts (nearly all
  experts touched even at batch=1 with moderate sequence lengths)
- **Decode**: `batchSize` tokens per step (at batch=1 only active experts read)

```
fractionTouched = 1 - (1 - numActive / numExperts) ^ numTokens
weightBytes = (sharedParams + fractionTouched * routedExpertParams) * precisionBytes
```

**Tensor parallelism**: Prefill compute divides by TP, with per-layer AllReduce
overhead. For MoE+EP, shared FLOPs scale with 1/TP and expert FLOPs with
1/(TP*EP). EP adds all-to-all communication (85% overlapped with compute).

Source: `src/core/inference/latency.ts:estimateTTFT()`, `calculateLatencyWithTP()`

---

## 3. TPOT Calculation

`estimateTPOT()` computes the time for a single decode step.

```
totalBytesPerStep = weightBytes + kvCacheBytes
memoryTime = totalBytesPerStep / (memoryBandwidth * bandwidthEfficiency)
computeTime = (2 * activeParams * batchSize) / (gpuTFLOPS * 1e12 * saturation * MFU)
tpot_ms = max(memoryTime, computeTime) * 1000
```

The compute floor ensures decode transitions from memory-bound to compute-bound
at very large batch sizes.

### Quantization Dequant Overhead

Quantized formats require on-the-fly dequantization before matmul (unpacking,
group metadata reads, scale/zero application). This inflates effective bytes
read per decode step:

```
effectiveBytes = weightBytes * dequantOverhead(weightPrecision, gpu)
               + kvCacheBytes * dequantOverhead(kvCachePrecision, gpu)
```

| Precision | Overhead | Rationale |
|-----------|----------|-----------|
| q2_k, q3_k_m | 1.15 | GGUF sub-4-bit: k-quants with fused GGML kernels |
| INT4 (GPTQ/AWQ) | 1.20 | Generic 4-bit: separate dequant kernels |
| q4_k_m | 1.10 | GGUF 4-bit: highly optimized fused GGML dequant |
| q5_k_m, q6_k | 1.08 | GGUF 5-6 bit: simple dequant, well-optimized kernels |
| INT8, q8_0 | 1.10 | Simple byte-level dequant |
| FP8 (native) | 1.05 | Transformer Engine GPUs (Hopper/Blackwell/Ada/CDNA3+) |
| FP8 (software) | 1.10 | Pre-Hopper GPUs: software dequant |
| FP4 | 1.05 | Blackwell-only: native FP4 tensor cores |
| BF16/FP16/FP32 | 1.0 | Native precision, no dequant |

The overhead only inflates the bandwidth term (`memoryTimeMs`). At large batch
sizes where decode transitions to compute-bound, the roofline `max(memoryTime,
computeTime)` correctly lets matmul throughput dominate.

FP8 native vs software: determined by `gpu.hasTransformerEngine` (true for
Hopper, Blackwell, Ada, CDNA3, CDNA4).

Source: `src/core/inference/latency.ts:getQuantizationOverhead()`

### Bandwidth Efficiency

Model-size-dependent sigmoid reflecting HBM burst utilization:

```
efficiency = max(0.35, min(0.85, 0.35 + 0.50 * (1 - 1 / (1 + weightGB / 5))))
```

| Model Size | Efficiency |
|------------|------------|
| Tiny / draft | ~0.35 |
| 1B | ~0.43 |
| 7B | ~0.72 |
| 70B | ~0.83 |
| 405B | ~0.85 |

### Tensor Parallelism

TP AllReduce uses a three-term model: tree-round alpha + bandwidth + layer-pipelining
overlap (`latency.ts:calculateLatencyWithTP()`).

**Per-collective latency** = alpha + bandwidth term:

```
numRounds = 2 × ceil(log2(tp))
alphaPerCollective = perRoundAlpha × numRounds     // 5us NVLink, 15us PCIe per round
bwPerCollective = (hiddenBytes / (commBW × 0.80)) × 2 × ((tp-1)/tp)
arPerCollective = alphaPerCollective + bwPerCollective
```

**Collective count** = `2 × numDenseLayers + arPerMoELayer × numMoELayers`, where
`arPerMoELayer = 1` when EP > 1 (expert FFN uses EP, not TP), else 2. Dense layers
have 2 AllReduces (attention output + MLP output).

**Layer-pipelining overlap**: each AllReduce hides behind the next sub-layer's
compute+memory work (different HW resources: NVLink/PCIe vs HBM). The last
collective is always fully exposed:

```
sublayerTime = baseExecTime / numTPCollectives
exposedPerCollective = max(0, arPerCollective - sublayerTime)
perStepAllReduce = (numTPCollectives - 1) × exposedPerCollective + arPerCollective
```

For large models (70B+), sublayer weight reads (~40-56us) exceed AllReduce time
(~30us at TP=8 NVLink), so AllReduce is fully hidden and TP scaling is near-perfect.
For small models (8-14B), AllReduce dominates the tiny per-sublayer reads, exposing
significant overhead — making TP=1 replicas more efficient for throughput.

For MoE models with EP, the expert FLOPs fraction is capped at **0.95** to
account for shared-layer compute that is not parallelized by EP.

Without MLA: `tpot = baseTpot / tp + perStepAllReduce + epAllToAll`.

With MLA (DeepSeek V2/V3/R1): weights scale with 1/TP (and 1/EP for experts),
but KV cache is **replicated** across TP ranks since all heads need the full
compressed latent.

### W4A16 vs FP8 Compute Model

The compute throughput used for the `max(memoryTime, computeTime)` roofline
depends on weight precision:

- **INT4 / GGUF (W4A16)**: Weights are dequantized to BF16 before matmul.
  Compute throughput = `bf16TFLOPS` (or `fp16TFLOPS` fallback).
- **FP8**: Uses native FP8 tensor cores. Compute throughput = `fp8TFLOPS`.
- **BF16/FP16**: Native precision. Compute throughput = `bf16TFLOPS`.

This distinction explains mixed-precision Pareto frontiers: INT4 dominates at
low batch sizes (fewer weight bytes to read), while FP8 overtakes at high batch
sizes (2x faster compute outweighs 2x more weight bytes when compute-bound).

Source: `src/core/inference/latency.ts:getInferenceComputeTFLOPS()`

### Average TPOT

`calculateLatencyMetrics()` averages initial and final TPOT:
`avgTpot = (initialTpot + finalTpot) / 2`, since KV cache grows linearly.

Source: `src/core/inference/latency.ts:estimateTPOT()`

---

## 4. KV Cache

Stores Key and Value projection outputs from previous tokens to avoid
recomputing attention at each decode step.

### Standard Formula

```
perTokenPerLayer = 2 * numKvHeads * headDim * precisionBytes
totalKVCache = perTokenPerLayer * numLayers * seqLen * batchSize
```

### MLA (Multi-head Latent Attention)

DeepSeek V2/V3/R1 store a compressed latent instead of separate K/V:

```
perTokenPerLayer = (kvLoraRank + qkRopeHeadDim) * precisionBytes
```

V3/R1: 576 values per layer (512 + 64). No factor of 2. Replicated across TP
ranks (not split).

### Attention Type Variants

| Type | numKvHeads | KV Cache Reduction |
|------|------------|-------------------|
| MHA | = numAttentionHeads | 1.0x |
| GQA | < numAttentionHeads | numKvHeads / numAttentionHeads |
| MQA | = 1 | 1 / numAttentionHeads |
| MLA | N/A | (kvLoraRank + qkRopeHeadDim) / (2 * nHeads * headDim) |

### Paged Attention

Without paging: pre-allocated contiguous buffers waste memory.
Fragmentation factor: `min(maxSeqLen / actualSeqLen, 2.0)`. Cap of 2.0x is
conservative ([Kwon et al., 2023](https://arxiv.org/abs/2309.06180) reports 60-80% waste). With paged attention,
no fragmentation penalty.

### Precision Support

| Format | Bytes/element |
|--------|--------------|
| fp32 | 4.0 |
| fp16 / bf16 | 2.0 |
| fp8 / int8 | 1.0 |
| int4 / fp4 | 0.5 |
| q2_k | 0.375 |
| q3_k_m | 0.486 |
| q4_k_m | 0.604 |
| q5_k_m | 0.709 |
| q6_k | 0.821 |
| q8_0 | 1.0625 |

GGUF formats use block-wise quantization; compute is always FP16.

Source: `src/core/inference/kv-cache.ts`

---

## 5. Memory Breakdown

Inference memory differs from training: no gradients or optimizer states, but
KV cache that grows with sequence length.

| Component | Formula | Notes |
|-----------|---------|-------|
| Weights | `totalParams * precisionBytes` | All experts loaded |
| KV Cache | `kvCachePerToken * seqLen * batchSize` | Grows linearly |
| Activations | `max(hidden, attnScores, mlp)` | One layer at a time |
| Overhead | `(weights + kvCache) * 0.10` | CUDA context, buffers (`kv-cache.ts`) |

**KV cache overhead factor**: 10% applied on top of weights + KV cache to account
for framework buffers, CUDA context, and temporary allocations (`kv-cache.ts`).

**Max KV cache memory**: 85% of available memory (after weights) is reserved for
KV cache. The remaining 15% covers activations and runtime overhead (`memory.ts`).

**With TP**: shared params / TP, routed experts / (TP * EP). Standard KV heads
split across ranks; MLA cache replicated. Activations divided by TP.

**EP weight formula**: `(sharedParams/tp + routedExpertParams/(tp*ep)) * bytes`

**FlashAttention**: Replaces O(seq^2) attention scores with tiled O(blockSize^2)
buffer (blockSize=128).

Source: `src/core/inference/memory.ts`

---

## 6. Speculative Decoding

A small "draft" model proposes K candidate tokens; the "target" model verifies
all K in a single forward pass (parallel, like prefill).

### Expected Accepted Tokens

Geometric series plus guaranteed bonus token ([Leviathan et al.](https://arxiv.org/abs/2211.17192)):

```
E[accepted] = alpha * (1 - alpha^K) / (1 - alpha) + 1
```

### Effective TPOT

```
draftTime = K * estimateTPOT(draftModel, ...)
verifyTime = prefillFLOPs(targetModel, K+1) / (compute * MFU)
effectiveTpot = (draftTime + verifyTime) / E[accepted]
speedup = baselineTPOT / effectiveTpot
```

Verification processes K+1 positions (K draft + 1 bonus). With TP, both draft
and verification use TP-aware calculations with AllReduce overhead.

### Acceptance Rate Estimation

- Base: `0.5 + 0.4 * sqrt(draftParams / targetParams)`
- Same family bonus: +0.05
- Same architecture bonus: +0.05
- Clamped to [0.30, 0.95]; uses `activeParams` for MoE

### Optimal K

`findOptimalSpecConfig()` searches K=1..16 and returns the K with maximum
speedup. Typical range: 3-8. When speedup <= 1.0, a warning is shown. The
`isSpeculativeDecodingBeneficial()` threshold is 1.2x.

Source: `src/core/inference/speculative.ts`

---

## 7. Continuous Batching

Iteration-level scheduling: slots are freed as requests complete and
immediately filled with new requests.

### Model

Steady-state slot-cycling: all B slots occupied, each cycling through
prefill then decode.

- **Scheduling overhead**: `0.01 + 0.01 * min(1, B/128)` (1-2% of TPOT)
- **Prefill interference**: `0.10 * min(1, (B-1)/32)` (up to 10% slowdown)
- **TTFT**: batch=1 prefill time with interference:
  `cbTTFT = batch1TTFT * (1 + interference)`

### Throughput

```
effectiveTpot = baseTpot * (1 + overhead)
cycleTime = cbTTFT + effectiveTpot * outputSeqLen
cbThroughput = B * outputSeqLen * 1000 / cycleTime
```

When combined with speculative decoding, CB overhead applies to each
draft-verify cycle: `specTpot = effectiveTpot_spec * (1 + overhead)`.

Source: `src/core/inference/latency.ts:calculateContinuousBatchingMetrics()`

---

## 8. Inference Optimizer

Pure function: takes `InferenceSimulationConfig` + target (`'throughput'` |
`'latency'`), returns best configuration found.

### Phase 1: Fix (max 10 iterations)

Resolves OOM by applying generators in priority order:
1. Enable paged attention
2. Increase TP (tries 2, 4, 8)
3. Increase EP (powers of 2 up to numExperts)
4. Combined TP + EP
5. Suggest quantization (heuristic)
6. Reduce batch size (halving)
7. Reduce context length (heuristic)

Each generator pre-validates by re-simulating to confirm OOM is resolved.

### Phase 2: Greedy (max 100 iterations)

Iteratively applies the best single-mutation improvement from 18 success
generators (flash attention, paged attention, KV precision match, TP/EP
adjustments, batch size, continuous batching, speculative decoding, etc.).
Threshold: >= 0.5% improvement. Stops when no generator improves.

### Phase 3: Explore (grid search)

Cartesian product over TP, EP, weight precision (bf16/fp8/int4/q4_k_m/q8_0),
batch size (1-16384), and continuous batching. Budget: 20,000 simulations
(random sampling if exceeded). Post-explore greedy refinement (max 20 iters).

### Replicas

`numReplicas = floor(numGPUs / (TP * EP))`. Independent serving instances.
Throughput scales linearly. Batch distributed:
`batchPerReplica = ceil(batchSize / numReplicas)`.

### Metric Comparison

```
throughput: newMetric > oldMetric / 0.995
latency:    newMetric < oldMetric * 0.995
```

Source: `src/core/inference/optimizer.ts`, `src/core/inference/exploration.ts`,
`src/core/inference/recommendations.ts`

---

## 9. Roofline Model and Utilization

```
prefillIntensity = prefillFLOPs / weightsBytes          (scales with seq len)
decodeIntensity  = decodeFLOPs / (weightsBytes + kvCache)  (scales with batch)
ridgePoint       = peakFLOPS / peakBandwidth             (ops/byte)
```

If intensity > ridgePoint: compute-bound. If below: memory-bandwidth-bound.

**Bottleneck classification**:
1. `memory_capacity` -- utilization > 90%
2. `memory_bandwidth` -- decode intensity below ridge (common case)
3. `compute` -- both phases compute-bound (rare, very large batch)

Source: `src/core/inference/latency.ts:calculateUtilizationMetrics()`
