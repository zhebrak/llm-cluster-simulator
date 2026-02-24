# Hardware Reference

Reference for all GPU specifications, interconnects, cluster topology, and cost
modeling used by the ML Cluster Simulator.

> **See also:** [PHYSICS.md](PHYSICS.md) (GPU-level scaling formulas)

## Table of Contents

1. [GPU Specifications](#1-gpu-specifications)
2. [Precision Handling](#2-precision-handling)
3. [Interconnects](#3-interconnects)
4. [Cluster Topology](#4-cluster-topology)
5. [Flash Attention Compatibility](#5-flash-attention-compatibility)
6. [Cost Model](#6-cost-model)

---

## 1. GPU Specifications

All TFLOPS values are **dense** (non-sparse). NVIDIA markets 2:4 structured
sparsity numbers that are 2x higher, but standard training does not use
structured sparsity. Since MFU is defined as `6PD / (time * peak)`, the peak
must be the dense rate to produce meaningful utilization percentages.

Memory values use decimal SI units (1 GB = 10^9 bytes), matching vendor
datasheets. NVIDIA's "80 GB" is physically 80 GiB (~85.9 x 10^9 bytes); the
simulator uses the marketing value, making memory estimates ~7% conservative.
The reserved memory model (1.0 GB base + 7% fragmentation) absorbs the gap.

### NVIDIA Datacenter GPUs

| GPU | Arch | Mem (GB) | BF16 TFLOPS | FP8 TFLOPS | FP4 TFLOPS | Mem BW (TB/s) | NVLink | TE | TDP (W) |
|-----|------|----------|-------------|------------|------------|---------------|--------|----|---------|
| A100 40GB | Ampere | 40 | 312 | -- | -- | 1.555 | v3 (300 GB/s) | No | 400 |
| A100 80GB | Ampere | 80 | 312 | -- | -- | 2.039 | v3 (300 GB/s) | No | 400 |
| A800 80GB | Ampere | 80 | 312 | -- | -- | 2.039 | v3 (200 GB/s) | No | 400 |
| H100 SXM | Hopper | 80 | 989 | 1979 | -- | 3.35 | v4 (450 GB/s) | Yes | 700 |
| H100 PCIe | Hopper | 80 | 756 | 1513 | -- | 2.0 | -- | Yes | 350 |
| H100 NVL | Hopper | 94 | 835 | 1671 | -- | 3.9 | v4 (300 GB/s) | Yes | 400 |
| H800 SXM | Hopper | 80 | 989 | 1979 | -- | 3.35 | v4 (200 GB/s) | Yes | 700 |
| H200 SXM | Hopper | 141 | 989 | 1979 | -- | 4.8 | v4 (450 GB/s) | Yes | 700 |
| B200 | Blackwell | 180 | 2250 | 4500 | 9000 | 7.7 | v5 (900 GB/s) | Yes | 1000 |
| GB200 NVL | Blackwell | 186 | 2500 | 5000 | 10000 | 8.0 | v5 (900 GB/s) | Yes | 1200 |

**TE** = Transformer Engine (hardware FP8 support). Required for FP8 quantized
collectives in TP communication.

**China-export variants** (A800, H800): Identical compute to their unrestricted
counterparts but with reduced NVLink bandwidth. A800: 200 GB/s (vs A100's 300).
H800: 200 GB/s (vs H100's 450). Fewer physical NVLink links.

**B200 and GB200** are marked `estimated: true` -- specs are from datasheets but
not yet validated against real training benchmarks.

### AMD Datacenter GPUs

| GPU | Arch | Mem (GB) | BF16 TFLOPS | FP8 TFLOPS | FP4 TFLOPS | Mem BW (TB/s) | Intra-node | TE | TDP (W) |
|-----|------|----------|-------------|------------|------------|---------------|------------|----|---------|
| MI210 | CDNA2 | 64 | 181 | -- | -- | 1.6 | IF 200 GB/s | No | 300 |
| MI250X (per GCD) | CDNA2 | 64 | 191.5 | -- | -- | 1.638 | IF 200 GB/s | No | 250 |
| MI300X | CDNA3 | 192 | 1307 | 2614 | -- | 5.3 | IF 448 GB/s | Yes | 750 |
| MI325X | CDNA3 | 256 | 1307 | 2614 | -- | 6.0 | IF 448 GB/s | Yes | 750 |
| MI350X | CDNA4 | 288 | 2307 | 4614 | 9228 | 8.0 | IF 538 GB/s | Yes | 1000 |

**IF** = AMD Infinity Fabric (intra-node interconnect, functionally equivalent
to NVLink for the simulator's communication model).

**MI250X** is a dual-GCD MCM package. The simulator models each GCD as a
separate device (matching ROCm's view). A "node" with 4 physical MI250X cards
appears as 8 GPUs (GCDs). Memory and bandwidth values are per-GCD (half of
package totals).

**MI325X** uses the same CDNA3 compute dies as MI300X with upgraded HBM3e
memory (256 GB, 6.0 TB/s vs 192 GB, 5.3 TB/s). Compute TFLOPS are identical.

### Professional / Workstation GPUs

| GPU | Arch | Mem (GB) | BF16 TFLOPS | FP8 TFLOPS | Mem BW (TB/s) | NVLink | TE | TDP (W) |
|-----|------|----------|-------------|------------|---------------|--------|----|---------|
| L40 | Ada | 48 | 181 | 362 | 0.864 | -- | Yes | 300 |
| L40S | Ada | 48 | 362 | 733 | 0.864 | -- | Yes | 350 |
| A10 | Ampere | 24 | 125 | -- | 0.6 | -- | No | 150 |
| A10G | Ampere | 24 | 70 | -- | 0.6 | -- | No | 150 |
| L4 | Ada | 24 | 121 | 242 | 0.3 | -- | Yes | 72 |
| RTX 6000 Ada | Ada | 48 | 182.2 | 364.4 | 0.96 | -- | Yes | 300 |

### Consumer GPUs

| GPU | Arch | Mem (GB) | BF16 TFLOPS | FP8 TFLOPS | Mem BW (TB/s) | TE | TDP (W) |
|-----|------|----------|-------------|------------|---------------|----|---------|
| RTX 4090 | Ada | 24 | 165.2 | 330.3 | 1.008 | Yes | 450 |
| RTX 3090 | Ampere | 24 | 142 | -- | 0.936 | No | 350 |

RTX 4090 BF16 TFLOPS listed as 165.2 (with FP32 accumulate, training-relevant).
NVIDIA's marketing "330.3" figure is pure BF16 without FP32 accumulate.

### Legacy GPUs

| GPU | Arch | Mem (GB) | FP16 TFLOPS | Mem BW (TB/s) | NVLink | TDP (W) |
|-----|------|----------|-------------|---------------|--------|---------|
| V100 32GB | Volta | 32 | 125 | 0.9 | v2 (150 GB/s) | 300 |
| T4 | Turing | 16 | 65 | 0.32 | -- | 70 |

V100 and T4 lack BF16 support. The simulator falls back to FP16 TFLOPS when
BF16 is requested.

**Source:** `src/core/hardware/gpu.ts`

---

## 2. Precision Handling

### TFLOPS Fallback Chain

`getEffectiveTFLOPS()` returns the best available rate for a given dtype,
falling back through a chain when a GPU lacks native support:

```
fp8  -> bf16 -> fp16
fp4  -> fp8  -> bf16 -> fp16
bf16 -> fp16
tf32 -> fp32
```

For example, requesting FP8 on an A100 (no FP8 hardware) returns the BF16 rate
(312 TFLOPS). `getPrecisionFallbackWarning()` returns a human-readable warning
when a fallback occurs.

### Training TFLOPS

`getTrainingTFLOPS()` models the Amdahl's law split between matmul and
non-matmul operations:

- **`MATMUL_TIME_FRACTION = 0.80`** -- 80% of training step time is in matmul
  ops (attention QKV, projections, MLP layers)
- **Non-matmul ops** (20%): LayerNorm, Softmax, GeLU/SiLU, loss computation --
  always run at BF16 rate regardless of compute dtype

For FP8, the matmul portion runs at 2x BF16 rate, but the non-matmul portion
does not benefit:

```
effectiveSpeedup = 1 / (0.80/2 + 0.20) = 1.667x  (not 2x)
```

H100 SXM FP8 example: `989 / 0.60 = 1648 TFLOPS` effective (not the raw 1979).

For BF16/FP16/TF32/FP32, `getTrainingTFLOPS()` returns `getEffectiveTFLOPS()`
unchanged (no split needed since matmul and non-matmul run at the same rate).

MFU always uses `getEffectiveTFLOPS(gpu, 'bf16')` as the denominator -- BF16
peak regardless of compute dtype. This matches the industry convention (PaLM
paper definition).

### Memory Bandwidth Scaling

`getMemoryBandwidthScaling()` accounts for the ~15% of training time that is
memory-bandwidth-bound (LayerNorm, Softmax, activation I/O):

- Uses Amdahl's law with H100 SXM as reference point (OI ~ 295 FLOP/byte)
- `memBoundFraction = 0.15`
- Returns >1 for GPUs with relatively more bandwidth per FLOP (these GPUs spend
  less time on memory-bound ops)
- Returns <1 for GPUs that are relatively bandwidth-starved

### Matmul Saturation

`getMatmulSaturationFactor()` models tensor core underutilization for small
matmul dimensions:

```
factor = min(1, (elements / threshold)^0.3)
```

Smooth power-law degradation -- at 50% of threshold, throughput is ~81%. Only
truly tiny matmuls (aggressive TP on small models) see measurable degradation.

Per-architecture saturation thresholds (M x K product):

| Architecture | Threshold | Example GPU |
|-------------|-----------|-------------|
| Hopper | ~1M | H100 |
| Blackwell | ~1.3M | B200 |
| Ampere | ~768K | A100 |
| Ada | ~640K | L40S |
| CDNA3 | ~1M | MI300X |
| CDNA4 | ~1.3M | MI350X |
| CDNA2 | ~768K | MI250X |
| Volta | ~512K | V100 |
| Turing | ~512K | T4 |

### DTYPE_BYTES Map

Byte widths used throughout the simulator for memory calculations:

| Dtype | Bytes |
|-------|-------|
| fp32 | 4 |
| tf32 | 4 |
| fp16 | 2 |
| bf16 | 2 |
| fp8 | 1 |
| fp4 | 0.5 |

**Source:** `src/core/hardware/gpu.ts`, `src/types/base.ts`

---

## 3. Interconnects

### Intra-Node Interconnects

`getIntraNodeInterconnect()` selects the appropriate intra-node fabric based on
GPU vendor and architecture:

**NVIDIA NVLink:**

| Version | Per-GPU BW (uni) | Bidirectional | Latency | GPUs |
|---------|-----------------|---------------|---------|------|
| NVLink 2.0 | 150 GB/s | 300 GB/s | 1.2 us | V100 |
| NVLink 3.0 | 300 GB/s | 600 GB/s | 1.0 us | A100 |
| NVLink 4.0 | 450 GB/s | 900 GB/s | 0.8 us | H100, H200 |
| NVLink 5.0 | 900 GB/s | 1800 GB/s | 0.6 us | B200, GB200 |

China-export variants override NVLink bandwidth per-GPU: H800 at 200 GB/s
(8 links vs H100's 18), A800 at 200 GB/s (8 links vs A100's 12). The simulator
detects this when `gpu.nvlinkBandwidthGBps < spec.bandwidthGBps` and creates a
modified spec.

**AMD Infinity Fabric:**

| Variant | Per-GPU BW (uni) | Bidirectional | Latency | GPUs |
|---------|-----------------|---------------|---------|------|
| IF (MI250) | 200 GB/s | 400 GB/s | 0.6 us | MI250X, MI210 |
| IF (MI300) | 448 GB/s | 896 GB/s | 0.5 us | MI300X, MI325X |
| IF (MI350) | 538 GB/s | 1075 GB/s | 0.4 us | MI350X |

AMD GPUs are routed to Infinity Fabric (not PCIe) in `getIntraNodeInterconnect()`
based on `gpu.vendor === 'amd'`.

**PCIe Fallback:**

GPUs without NVLink or Infinity Fabric fall back to PCIe based on architecture:

| Generation | BW (uni) | Bidirectional | Latency | Architectures |
|-----------|----------|---------------|---------|---------------|
| PCIe Gen3 | 15.75 GB/s | 31.5 GB/s | 1.5 us | Volta, Turing |
| PCIe Gen4 | 31.5 GB/s | 63 GB/s | 1.2 us | Ampere, Ada |
| PCIe Gen5 | 63 GB/s | 126 GB/s | 1.0 us | Hopper |
| PCIe Gen6 | 126 GB/s | 252 GB/s | 0.8 us | Blackwell |

### NVSwitch

`getNvSwitchSpec()` returns the NVSwitch specification for GPUs that support it.
NVSwitch provides **full bisection bandwidth** within a node -- any GPU can
communicate with any other GPU at full line rate simultaneously.

| NVSwitch Gen | Per-GPU BW | Bidirectional | Latency | GPUs |
|-------------|-----------|---------------|---------|------|
| A100 | 300 GB/s | 600 GB/s | 0.5 us | A100, A800, V100 |
| H100 | 450 GB/s | 900 GB/s | 0.4 us | H100, H800, H200 |
| B200 | 900 GB/s | 1800 GB/s | 0.3 us | B200, GB200 |

NVSwitch is activated when `gpu.hasNvSwitch && numGPUs > 2`. For 2-GPU configs
(e.g., H100 NVL bridge), direct NVLink point-to-point is used instead.

China-export NVSwitch bandwidth is reduced to match the per-GPU NVLink bandwidth
(e.g., H800 NVSwitch at 200 GB/s per GPU instead of 450).

### Inter-Node Interconnects

**InfiniBand** (bandwidth is aggregate for a DGX node with 8 NICs):

| Version | Per-Port BW | Aggregate (8x) | Bidirectional | Latency | Ports/Switch |
|---------|------------|-----------------|---------------|---------|-------------|
| HDR | 25 GB/s | 200 GB/s | 400 GB/s | 0.6 us | 40 |
| NDR | 50 GB/s | 400 GB/s | 800 GB/s | 0.5 us | 64 |
| XDR | 100 GB/s | 800 GB/s | 1600 GB/s | 0.4 us | 72 |

DGX systems use 8 network adapters per node to match internal GPU bandwidth.
The IB version is auto-detected from GPU architecture (see IB Auto-Detection
below). NDR is the fallback for unknown architectures.

**RoCE v2 (RDMA over Converged Ethernet):**

| Speed | BW (uni) | Bidirectional | Latency |
|-------|----------|---------------|---------|
| 100 GbE | 12.5 GB/s | 25 GB/s | 2.0 us |
| 200 GbE | 25 GB/s | 50 GB/s | 1.5 us |
| 400 GbE | 50 GB/s | 100 GB/s | 1.2 us |

**Standard Ethernet:**

| Speed | BW (uni) | Bidirectional | Latency |
|-------|----------|---------------|---------|
| 10 GbE | 1.25 GB/s | 2.5 GB/s | 50 us |
| 25 GbE | 3.125 GB/s | 6.25 GB/s | 30 us |
| 100 GbE | 12.5 GB/s | 25 GB/s | 10 us |

### Cross-Node P2P Bandwidth

For point-to-point operations (pipeline parallel P2P sends), the effective
per-GPU cross-node bandwidth is:

```
crossNodeP2PBW = interNodeBandwidthGBps / gpusPerNode
```

This models one NIC per P2P transfer. A DGX H100 with NDR (400 GB/s aggregate)
provides 50 GB/s per GPU for PP P2P communication.

**Source:** `src/core/hardware/interconnect.ts`, `src/types/hardware.ts`

---

## 4. Cluster Topology

### ClusterConfig

`createCluster()` builds a `ClusterConfig` with these computed properties:

| Property | Derivation |
|----------|-----------|
| `totalGPUs` | `numNodes * gpusPerNode` |
| `totalMemoryGB` | `totalGPUs * gpu.memoryGB` |
| `totalTFLOPS` | `totalGPUs * getEffectiveTFLOPS(gpu, 'bf16')` |
| `interNodeBandwidthGBps` | From selected InfiniBand/RoCE/Ethernet spec |
| `interNodeLatencyUs` | From selected inter-node interconnect spec |

### Cluster Factory Functions

- **`createSingleNodeCluster(gpuId, numGPUs)`** -- single-node topology, no
  inter-node interconnect needed
- **`createMultiNodeCluster(gpuId, gpusPerNode, numNodes, ibVersion?)`** -- main
  factory for multi-node clusters; `ibVersion` is optional and auto-detected
  from GPU architecture when not provided (see IB Auto-Detection below)
- **`createNode(gpu, numGPUs, cpuCores, ramGB, interNodeInterconnect)`** --
  low-level node builder; auto-selects intra-node interconnect via
  `getIntraNodeInterconnect()` and NVSwitch via `getNvSwitchSpec()`

### IB Auto-Detection

`getDefaultIBVersion(architecture)` maps GPU architecture to the appropriate
InfiniBand version:

| Architecture | IB Version | Aggregate Node BW | Era |
|-------------|------------|-------------------|-----|
| `ampere`, `volta`, `turing`, `cdna2` | HDR | 200 GB/s | A100, V100, T4, MI250X |
| `hopper`, `cdna3`, `ada` | NDR | 400 GB/s | H100, H200, MI300X, MI325X, L40S |
| `blackwell`, `cdna4` | XDR | 800 GB/s | B200, GB200, MI350X |
| Unknown | NDR | 400 GB/s | Fallback |

`createMultiNodeCluster('a100-80gb', 8, 16)` automatically gets HDR (200 GB/s)
inter-node bandwidth. An explicit `ibVersion` parameter overrides the
auto-detection.

### Topology Types

- `'single-node'` -- all GPUs on one machine
- `'fat-tree'` -- multi-node with non-blocking fabric (default for multi-node)
- `'dragonfly'`, `'torus'`, `'ring'` -- defined in types but fat-tree is the
  standard for DGX SuperPOD deployments

### Cluster Size Categories

| Category | GPU counts |
|----------|-----------|
| Small | 1, 2, 4, 8 |
| Medium | 16, 32, 64 |
| Large | 128, 256, 512 |
| XLarge | 1024, 2048, 4096 |

### DGX Node Presets

| Preset | GPU | GPUs/Node | Intra-node | Inter-node |
|--------|-----|-----------|-----------|------------|
| DGX A100 | A100 80GB | 8 | NVSwitch (A100) | IB HDR |
| DGX H100 | H100 SXM | 8 | NVSwitch (H100) | IB NDR |
| DGX H200 | H200 SXM | 8 | NVSwitch (H100) | IB NDR |
| DGX B200 | B200 | 8 | NVSwitch (B200) | IB XDR |
| GB200 NVL72 Tray | GB200 | 4 | NVSwitch (B200) | IB XDR |

AMD node presets:

| Preset | GPU | GPUs/Node | Intra-node | Inter-node |
|--------|-----|-----------|-----------|------------|
| MI250X Node | MI250X (per GCD) | 8 | IF (MI250) | IB HDR |
| MI300X Node | MI300X | 8 | IF (MI300) | IB NDR |
| MI325X Node | MI325X | 8 | IF (MI300) | IB NDR |
| MI350X Node | MI350X | 8 | IF (MI350) | IB XDR |

PCIe inference server presets (no NVSwitch):

| Preset | GPU | GPUs/Node | Intra-node | Inter-node |
|--------|-----|-----------|-----------|------------|
| T4 PCIe | T4 | 4 | PCIe Gen3 | Ethernet 25GbE |
| L4 PCIe | L4 | 8 | PCIe Gen4 | Ethernet 100GbE |
| A10G PCIe | A10G | 4 | PCIe Gen4 | Ethernet 100GbE |
| L40S PCIe | L40S | 8 | PCIe Gen4 | Ethernet 100GbE |
| A100 PCIe | A100 80GB | 8 | PCIe Gen4 | RoCE 200GbE |
| H100 PCIe | H100 PCIe | 8 | PCIe Gen5 | RoCE 400GbE |

**Source:** `src/core/hardware/topology.ts`, `src/core/hardware/presets.ts`

---

## 5. Flash Attention Compatibility

`supportsFlashAttention()` checks the GPU architecture:

| Architecture | Flash Attention | Notes |
|-------------|----------------|-------|
| Ampere | Yes | A100, A10, A10G, RTX 3090 |
| Ada | Yes | L40, L40S, L4, RTX 4090, RTX 6000 Ada |
| Hopper | Yes | H100, H200 |
| Blackwell | Yes | B200, GB200 |
| CDNA2 | Yes | MI250X, MI210 (ROCm FA) |
| CDNA3 | Yes | MI300X, MI325X |
| CDNA4 | Yes | MI350X |
| Volta | **No** | V100 -- lacks hardware support |
| Turing | **No** | T4 -- lacks hardware support |

### Impact on Simulation

**With Flash Attention:** Attention score memory uses O(seq) tile memory:

```
numHeads * seqLength * dtype_bytes * microBatchSize
```

**Without Flash Attention:** Attention score memory uses O(seq^2) with a 2.5x
multiplier per [Korthikanti et al. 2022](https://arxiv.org/abs/2205.05198) (Table 1):

```
numHeads * seqLength^2 * dtype_bytes * microBatchSize * 2.5
```

The 2.5x accounts for: pre-softmax scores (2 bytes), post-softmax probabilities
(2 bytes), and dropout mask (1 byte) = 5 bytes per element vs 2 bytes baseline.

For large sequence lengths, the O(seq^2) vs O(seq) difference dominates
activation memory. A 405B model at 128K sequence length without Flash Attention
would require orders of magnitude more activation memory.

**Source:** `src/core/hardware/gpu.ts:supportsFlashAttention()`

---

## 6. Cost Model

### GPU Hourly Rates (GPU_HOURLY_RATES)

Median reserved-capacity rates ($/GPU-hour, 1yr commitment) used for dashboard
cost estimates:

| GPU | $/GPU-hr | GPU | $/GPU-hr |
|-----|---------|-----|---------|
| GB200 | $6.00 | B200 | $5.00 |
| MI350X | $4.50 | H200 SXM | $3.50 |
| H100 NVL | $3.00 | MI325X | $2.50 |
| H100 SXM | $2.50 | H800 SXM | $2.25 |
| H100 PCIe | $2.25 | MI300X | $2.00 |
| A100 80GB | $1.75 | A800 80GB | $1.50 |
| A100 40GB | $1.50 | MI250X | $1.50 |
| MI210 | $1.00 | L40 | $1.00 |
| L40S | $1.00 | A10G | $1.00 |
| RTX 6000 Ada | $0.75 | A10 | $0.50 |
| L4 | $0.50 | V100 | $0.50 |
| RTX 4090 | $0.50 | T4 | $0.25 |
| RTX 3090 | $0.25 | | |

`getGPUHourlyRate(gpuId)` returns the rate for a GPU ID, falling back to $2.50
for unknown GPUs.

**Source:** `src/core/cost/cloud.ts`
