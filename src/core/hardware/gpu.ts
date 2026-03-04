/**
 * GPU specifications — all values use decimal SI units.
 *
 * Memory:    GB  = 10⁹ bytes  (matches NVIDIA/AMD marketing datasheets)
 * Bandwidth: TB/s = 10¹² bytes/s, GB/s = 10⁹ bytes/s (JEDEC convention)
 * Compute:   TFLOPS = 10¹² FLOPS (always decimal)
 *
 * NVIDIA's "80 GB" is physically 80 GiB (~85.9 × 10⁹ bytes).
 * We use the marketing value (80 × 10⁹), making memory estimates ~7%
 * conservative. This is intentional — the reserved memory model
 * (1.0 GB base + 7% fragmentation) absorbs the gap.
 *
 * Sources for each GPU: NVIDIA/AMD datasheets, arxiv.org, techpowerup.com.
 * All TFLOPS values are DENSE (non-sparse). NVIDIA marketing often quotes
 * "with sparsity" (2:4 structured sparsity = 2x dense). Since standard
 * training FLOPS formulas (6*P*D) compute dense operations, we must use
 * dense peak TFLOPS to get correct MFU calculations.
 */
// See docs/HARDWARE.md and docs/PHYSICS.md for GPU specs and scaling formulas.

import type { GPUSpec } from '../../types/index.ts';

/**
 * NVIDIA A100 GPUs (Ampere architecture)
 */
export const A100_40GB: GPUSpec = {
  id: 'a100-40gb',
  name: 'NVIDIA A100 40GB',
  vendor: 'nvidia',
  architecture: 'ampere',

  memoryGB: 40,
  memoryBandwidthTBps: 1.555,

  fp32TFLOPS: 19.5,
  tf32TFLOPS: 156,        // TF32 tensor core ops
  fp16TFLOPS: 312,
  bf16TFLOPS: 312,
  fp8TFLOPS: 0,           // No FP8 on Ampere
  fp4TFLOPS: 0,           // No FP4 on Ampere
  int8TOPS: 624,
  int4TOPS: 1248,         // 2x INT8

  hasTensorCores: true,
  tensorCoreTFLOPS: 312,

  tdpWatts: 400,

  nvlinkBandwidthGBps: 300, // 600 GB/s bidirectional / 2
  nvlinkVersion: 3,
  pcieBandwidthGBps: 31.5, // PCIe Gen4 x16

  hasTransformerEngine: false,
  hasNvSwitch: true,
};

export const A100_80GB: GPUSpec = {
  id: 'a100-80gb',
  name: 'NVIDIA A100 80GB',
  vendor: 'nvidia',
  architecture: 'ampere',

  memoryGB: 80,
  memoryBandwidthTBps: 2.039,

  fp32TFLOPS: 19.5,
  tf32TFLOPS: 156,
  fp16TFLOPS: 312,
  bf16TFLOPS: 312,
  fp8TFLOPS: 0,
  fp4TFLOPS: 0,
  int8TOPS: 624,
  int4TOPS: 1248,

  hasTensorCores: true,
  tensorCoreTFLOPS: 312,

  tdpWatts: 400,

  nvlinkBandwidthGBps: 300,
  nvlinkVersion: 3,
  pcieBandwidthGBps: 31.5,

  hasTransformerEngine: false,
  hasNvSwitch: true,
};

/**
 * NVIDIA A800 80GB (Ampere, China-export variant)
 *
 * Identical compute to A100 80GB but with reduced NVLink bandwidth:
 * 8 NVLink 3.0 links (vs 12 on A100) → 200 GB/s uni (400 bidi).
 * Source: Lenovo A800 product guide; confirmed by multiple hardware databases.
 */
export const A800_80GB: GPUSpec = {
  id: 'a800-80gb',
  name: 'NVIDIA A800 80GB',
  vendor: 'nvidia',
  architecture: 'ampere',

  memoryGB: 80,
  memoryBandwidthTBps: 2.039,      // Same HBM2e as A100 80GB

  fp32TFLOPS: 19.5,
  tf32TFLOPS: 156,
  fp16TFLOPS: 312,
  bf16TFLOPS: 312,
  fp8TFLOPS: 0,                    // No FP8 on Ampere
  fp4TFLOPS: 0,
  int8TOPS: 624,
  int4TOPS: 1248,

  hasTensorCores: true,
  tensorCoreTFLOPS: 312,

  tdpWatts: 400,

  nvlinkBandwidthGBps: 200,        // 400 bidi — reduced from A100's 300 uni
  nvlinkVersion: 3,
  pcieBandwidthGBps: 31.5,

  hasTransformerEngine: false,
  hasNvSwitch: true,
};

/**
 * NVIDIA H100 GPUs (Hopper architecture)
 */
export const H100_SXM: GPUSpec = {
  id: 'h100-sxm',
  name: 'NVIDIA H100 SXM',
  vendor: 'nvidia',
  architecture: 'hopper',

  memoryGB: 80,
  memoryBandwidthTBps: 3.35,

  fp32TFLOPS: 67,
  tf32TFLOPS: 989,         // Dense TF32 tensor core ops
  fp16TFLOPS: 989,         // Dense FP16 (sparse: 1979)
  bf16TFLOPS: 989,         // Dense BF16 (sparse: 1979)
  fp8TFLOPS: 1979,         // Dense FP8 with Transformer Engine (sparse: 3958)
  fp4TFLOPS: 0,            // No FP4 on Hopper
  int8TOPS: 1979,          // Dense INT8 (sparse: 3958)
  int4TOPS: 3958,          // Dense INT4 (sparse: 7916)

  hasTensorCores: true,
  tensorCoreTFLOPS: 989,   // Dense BF16 peak

  tdpWatts: 700,

  nvlinkBandwidthGBps: 450, // 900 GB/s bidirectional
  nvlinkVersion: 4,
  pcieBandwidthGBps: 63, // PCIe Gen5 x16

  hasTransformerEngine: true,
  hasNvSwitch: true,
};

export const H100_PCIE: GPUSpec = {
  id: 'h100-pcie',
  name: 'NVIDIA H100 PCIe',
  vendor: 'nvidia',
  architecture: 'hopper',

  memoryGB: 80,
  memoryBandwidthTBps: 2.0,

  fp32TFLOPS: 51,
  tf32TFLOPS: 756,         // Dense TF32
  fp16TFLOPS: 756,         // Dense FP16 (sparse: 1513)
  bf16TFLOPS: 756,         // Dense BF16 (sparse: 1513)
  fp8TFLOPS: 1513,         // Dense FP8 (sparse: 3026)
  fp4TFLOPS: 0,
  int8TOPS: 1513,          // Dense INT8 (sparse: 3026)
  int4TOPS: 3026,          // Dense INT4 (sparse: 6052)

  hasTensorCores: true,
  tensorCoreTFLOPS: 756,   // Dense BF16 peak

  tdpWatts: 350,

  nvlinkBandwidthGBps: 0, // No NVLink on PCIe version
  nvlinkVersion: null,
  pcieBandwidthGBps: 63,

  hasTransformerEngine: true,
  hasNvSwitch: false,
};

export const H100_NVL: GPUSpec = {
  id: 'h100-nvl',
  name: 'NVIDIA H100 NVL',
  vendor: 'nvidia',
  architecture: 'hopper',

  memoryGB: 94,
  memoryBandwidthTBps: 3.9,

  fp32TFLOPS: 60,
  tf32TFLOPS: 835,         // Dense TF32 (lower clocks than SXM at 400W TDP)
  fp16TFLOPS: 835,         // Dense FP16 (sparse: 1671)
  bf16TFLOPS: 835,         // Dense BF16 (sparse: 1671)
  fp8TFLOPS: 1671,         // Dense FP8 (sparse: 3341)
  fp4TFLOPS: 0,
  int8TOPS: 1671,          // Dense INT8 (sparse: 3341)
  int4TOPS: 3341,          // Dense INT4 (sparse: 6683)

  hasTensorCores: true,
  tensorCoreTFLOPS: 835,   // Dense BF16 peak

  tdpWatts: 400,

  nvlinkBandwidthGBps: 300, // NVL bridge: 600 GB/s bidi = 300 uni
  nvlinkVersion: 4,
  pcieBandwidthGBps: 63,

  hasTransformerEngine: true,
  hasNvSwitch: false, // Typically 2-GPU NVLink bridge
};

/**
 * NVIDIA H800 SXM (Hopper, China-export variant)
 *
 * Identical compute to H100 SXM but with reduced NVLink bandwidth:
 * 8 NVLink 4.0 links (vs 18 on H100) → 200 GB/s uni (400 bidi).
 * Source: ISCA 2025 paper (arXiv 2505.09343), DeepSeek GH #28.
 */
export const H800_SXM: GPUSpec = {
  id: 'h800-sxm',
  name: 'NVIDIA H800 SXM',
  vendor: 'nvidia',
  architecture: 'hopper',

  memoryGB: 80,
  memoryBandwidthTBps: 3.35,       // Same HBM3 as H100 SXM

  fp32TFLOPS: 67,
  tf32TFLOPS: 989,
  fp16TFLOPS: 989,
  bf16TFLOPS: 989,
  fp8TFLOPS: 1979,
  fp4TFLOPS: 0,
  int8TOPS: 1979,
  int4TOPS: 3958,

  hasTensorCores: true,
  tensorCoreTFLOPS: 989,

  tdpWatts: 700,

  nvlinkBandwidthGBps: 200,        // 400 bidi — reduced from H100's 450 uni
  nvlinkVersion: 4,
  pcieBandwidthGBps: 63,

  hasTransformerEngine: true,
  hasNvSwitch: true,
};

/**
 * NVIDIA H200 GPUs (Hopper with HBM3e)
 */
export const H200_SXM: GPUSpec = {
  id: 'h200-sxm',
  name: 'NVIDIA H200 SXM',
  vendor: 'nvidia',
  architecture: 'hopper',

  memoryGB: 141,
  memoryBandwidthTBps: 4.8,

  fp32TFLOPS: 67,
  tf32TFLOPS: 989,         // Dense TF32
  fp16TFLOPS: 989,         // Dense FP16 (sparse: 1979)
  bf16TFLOPS: 989,         // Dense BF16 (sparse: 1979)
  fp8TFLOPS: 1979,         // Dense FP8 (sparse: 3958)
  fp4TFLOPS: 0,
  int8TOPS: 1979,          // Dense INT8 (sparse: 3958)
  int4TOPS: 3958,          // Dense INT4 (sparse: 7916)

  hasTensorCores: true,
  tensorCoreTFLOPS: 989,   // Dense BF16 peak

  tdpWatts: 700,

  nvlinkBandwidthGBps: 450,
  nvlinkVersion: 4,
  pcieBandwidthGBps: 63,

  hasTransformerEngine: true,
  hasNvSwitch: true,
};

/**
 * NVIDIA B200 GPUs (Blackwell architecture)
 */
export const B200: GPUSpec = {
  id: 'b200',
  name: 'NVIDIA B200',
  vendor: 'nvidia',
  architecture: 'blackwell',
  estimated: true,

  memoryGB: 180,             // HGX air-cooled variant (Lenovo ThinkSystem datasheet)
  memoryBandwidthTBps: 7.7,  // HBM3e at 1000W TDP

  fp32TFLOPS: 75,            // Dense FP32 at 1000W
  tf32TFLOPS: 1100,          // Dense TF32 (sparse: 2200)
  fp16TFLOPS: 2250,          // Dense FP16 (sparse: 4500)
  bf16TFLOPS: 2250,          // Dense BF16 (sparse: 4500)
  fp8TFLOPS: 4500,           // Dense FP8 (sparse: 9000)
  fp4TFLOPS: 9000,           // Dense FP4 (sparse: 18000)
  int8TOPS: 4500,            // Dense INT8 (sparse: 9000)
  int4TOPS: 9000,            // Dense INT4 (sparse: 18000)

  hasTensorCores: true,
  tensorCoreTFLOPS: 2250,    // Dense BF16 peak

  tdpWatts: 1000,

  nvlinkBandwidthGBps: 900,  // 1800 GB/s bidirectional, NVLink 5
  nvlinkVersion: 5,
  pcieBandwidthGBps: 63,     // PCIe Gen5 x16 (HGX baseboard)

  hasTransformerEngine: true,
  hasNvSwitch: true,
};

export const GB200: GPUSpec = {
  id: 'gb200',
  name: 'NVIDIA GB200 NVL',
  vendor: 'nvidia',
  architecture: 'blackwell',
  estimated: true,

  memoryGB: 186,             // 372 GB per superchip ÷ 2, liquid-cooled NVL variant
  memoryBandwidthTBps: 8.0,  // HBM3e at 1200W TDP

  fp32TFLOPS: 80,            // 160 TFLOPS per superchip ÷ 2
  tf32TFLOPS: 1250,          // Dense TF32 (sparse: 2500)
  fp16TFLOPS: 2500,          // Dense FP16 (sparse: 5000)
  bf16TFLOPS: 2500,          // Dense BF16 (sparse: 5000)
  fp8TFLOPS: 5000,           // Dense FP8 (sparse: 10000)
  fp4TFLOPS: 10000,          // Dense FP4 (sparse: 20000)
  int8TOPS: 5000,            // Dense INT8 (sparse: 10000)
  int4TOPS: 10000,           // Dense INT4 (sparse: 20000)

  hasTensorCores: true,
  tensorCoreTFLOPS: 2500,    // Dense BF16 peak

  tdpWatts: 1200,            // Liquid-cooled, higher power envelope than HGX B200

  nvlinkBandwidthGBps: 900,  // 1800 GB/s bidirectional, NVLink 5
  nvlinkVersion: 5,
  pcieBandwidthGBps: 126,    // PCIe Gen6 x16 (NVL rack)

  hasTransformerEngine: true,
  hasNvSwitch: true,
};

/**
 * AMD MI250X (CDNA2 architecture, per-GCD)
 *
 * Dual-GCD MCM package (500W total). Each GCD = 1 device in ROCm.
 * 110 CUs per GCD, 128 GB HBM2e total (64 GB per GCD).
 * Frontier supercomputer GPU — published MFU: 32-38% at scale.
 */
export const MI250X: GPUSpec = {
  id: 'mi250x',
  name: 'AMD MI250X (per GCD)',
  vendor: 'amd',
  architecture: 'cdna2',

  memoryGB: 64,                   // 128 GB total / 2 GCDs
  memoryBandwidthTBps: 1.638,     // 3.2768 TB/s total / 2 GCDs

  fp32TFLOPS: 47.9,               // MFMA FP32 per GCD
  tf32TFLOPS: 0,                  // No TF32 on CDNA2
  fp16TFLOPS: 191.5,              // 110 CUs × 4 × 256 × 1.7 GHz
  bf16TFLOPS: 191.5,              // Same as FP16 on CDNA2
  fp8TFLOPS: 0,                   // No FP8 on CDNA2
  fp4TFLOPS: 0,                   // No FP4 on CDNA2
  int8TOPS: 191.5,                // CDNA2: INT8 = FP16 rate
  int4TOPS: 191.5,                // CDNA2: INT4 = FP16 rate

  hasTensorCores: true,           // AMD Matrix Cores
  tensorCoreTFLOPS: 191.5,        // BF16 peak

  tdpWatts: 250,                  // ~500W package / 2 GCDs

  nvlinkBandwidthGBps: 0,         // Uses AMD Infinity Fabric
  nvlinkVersion: null,
  pcieBandwidthGBps: 31.5,        // PCIe Gen4 x16

  hasTransformerEngine: false,     // No FP8 support
  hasNvSwitch: false,
};

/**
 * AMD MI210 (CDNA2 architecture, single-GCD PCIe)
 *
 * Same CDNA2 die as MI250X but single-GCD PCIe form factor.
 * 104 CUs (vs 110 on MI250X), 64 GB HBM2e.
 */
export const MI210: GPUSpec = {
  id: 'mi210',
  name: 'AMD MI210',
  vendor: 'amd',
  architecture: 'cdna2',

  memoryGB: 64,                   // HBM2e
  memoryBandwidthTBps: 1.6,       // AMD MI210 spec: "Up to 1.6 TB/s"

  fp32TFLOPS: 45.3,               // MFMA FP32 (104 CUs vs 110)
  tf32TFLOPS: 0,                  // No TF32 on CDNA2
  fp16TFLOPS: 181,                // 104 CUs × 4 × 256 × 1.7 GHz
  bf16TFLOPS: 181,                // Same as FP16 on CDNA2
  fp8TFLOPS: 0,                   // No FP8 on CDNA2
  fp4TFLOPS: 0,                   // No FP4 on CDNA2
  int8TOPS: 181,                  // CDNA2: INT8 = FP16 rate
  int4TOPS: 181,                  // CDNA2: INT4 = FP16 rate

  hasTensorCores: true,           // AMD Matrix Cores
  tensorCoreTFLOPS: 181,          // BF16 peak

  tdpWatts: 300,                  // PCIe form factor

  nvlinkBandwidthGBps: 0,
  nvlinkVersion: null,
  pcieBandwidthGBps: 31.5,        // PCIe Gen4 x16

  hasTransformerEngine: false,     // No FP8
  hasNvSwitch: false,
};

/**
 * AMD MI350X (CDNA4 architecture)
 *
 * 8 XCD chiplets, first AMD GPU with FP4 support.
 * 288 GB HBM3e, 8 TB/s memory bandwidth.
 */
export const MI350X: GPUSpec = {
  id: 'mi350x',
  name: 'AMD MI350X',
  vendor: 'amd',
  architecture: 'cdna4',

  memoryGB: 288,                  // HBM3e, 8 stacks
  memoryBandwidthTBps: 8.0,       // Confirmed

  fp32TFLOPS: 144,                // FP32 vector
  tf32TFLOPS: 1153,               // Estimated: BF16/2 (CDNA3 ratio)
  fp16TFLOPS: 2307,               // Dense matrix, confirmed
  bf16TFLOPS: 2307,               // Dense matrix, confirmed
  fp8TFLOPS: 4614,                // Dense, 2× BF16
  fp4TFLOPS: 9228,                // Dense, 2× FP8 — new in CDNA4
  int8TOPS: 4614,
  int4TOPS: 9228,

  hasTensorCores: true,           // AMD Matrix Cores
  tensorCoreTFLOPS: 2307,         // BF16 peak

  tdpWatts: 1000,                 // Air-cooled OAM

  nvlinkBandwidthGBps: 0,         // Uses AMD Infinity Fabric
  nvlinkVersion: null,
  pcieBandwidthGBps: 63,          // PCIe Gen5

  hasTransformerEngine: true,      // FP8 support
  hasNvSwitch: false,
};

/**
 * AMD MI300X (CDNA3 architecture)
 */
export const MI300X: GPUSpec = {
  id: 'mi300x',
  name: 'AMD MI300X',
  vendor: 'amd',
  architecture: 'cdna3',

  memoryGB: 192,
  memoryBandwidthTBps: 5.3,

  fp32TFLOPS: 163,
  tf32TFLOPS: 653,         // AMD equivalent matrix ops
  fp16TFLOPS: 1307,
  bf16TFLOPS: 1307,
  fp8TFLOPS: 2614,
  fp4TFLOPS: 0,            // No FP4 on CDNA3
  int8TOPS: 2614,
  int4TOPS: 5228,

  hasTensorCores: true, // AMD Matrix Cores
  tensorCoreTFLOPS: 1307,

  tdpWatts: 750,

  nvlinkBandwidthGBps: 0, // Uses AMD Infinity Fabric
  nvlinkVersion: null,
  pcieBandwidthGBps: 63,

  hasTransformerEngine: true, // FP8 support
  hasNvSwitch: false,
};

/**
 * AMD MI325X (CDNA3 architecture — same compute dies as MI300X, HBM3e memory)
 */
export const MI325X: GPUSpec = {
  id: 'mi325x',
  name: 'AMD MI325X',
  vendor: 'amd',
  architecture: 'cdna3',     // Same CDNA3 compute dies (XCDs) as MI300X, with HBM3e memory

  memoryGB: 256,
  memoryBandwidthTBps: 6.0,

  fp32TFLOPS: 163,
  tf32TFLOPS: 653,
  fp16TFLOPS: 1307,
  bf16TFLOPS: 1307,
  fp8TFLOPS: 2614,
  fp4TFLOPS: 0,            // CDNA3 compute die — no FP4
  int8TOPS: 2614,
  int4TOPS: 5228,

  hasTensorCores: true,
  tensorCoreTFLOPS: 1307,

  tdpWatts: 750,

  nvlinkBandwidthGBps: 0,
  nvlinkVersion: null,
  pcieBandwidthGBps: 63,

  hasTransformerEngine: true,
  hasNvSwitch: false,
};

/**
 * NVIDIA T4 (Turing architecture, TU104 — inference-focused)
 *
 * Key characteristics:
 * - 16 GB GDDR6 (NOT HBM), 320 GB/s bandwidth
 * - 70W TDP, single-slot low-profile passive cooling
 * - PCIe Gen3 only, no NVLink
 * - Turing Tensor Cores: FP16=65 TFLOPS, INT8=130 TOPS, INT4=260 TOPS
 * - No BF16, TF32, or FP8 support (pre-Ampere)
 * - No sparsity support (pre-Ampere), all values are inherently dense
 */
export const T4: GPUSpec = {
  id: 't4',
  name: 'NVIDIA T4',
  vendor: 'nvidia',
  architecture: 'turing',

  memoryGB: 16,
  memoryBandwidthTBps: 0.32,    // 320 GB/s (GDDR6, 256-bit @ 10 Gbps)

  fp32TFLOPS: 8.1,
  tf32TFLOPS: 0,                // No TF32 on Turing
  fp16TFLOPS: 65,               // Tensor Core FP16 (mixed-precision FP16/FP32 accumulate)
  bf16TFLOPS: 0,                // No BF16 on Turing (introduced in Ampere)
  fp8TFLOPS: 0,                 // No FP8 on Turing
  fp4TFLOPS: 0,                 // No FP4 on Turing
  int8TOPS: 130,                // Tensor Core INT8
  int4TOPS: 260,                // Tensor Core INT4

  hasTensorCores: true,         // 2nd gen Turing Tensor Cores (320 cores)
  tensorCoreTFLOPS: 65,         // FP16 Tensor Core peak (no BF16 available)

  tdpWatts: 70,

  nvlinkBandwidthGBps: 0,      // No NVLink — inference scale-out via PCIe
  nvlinkVersion: null,
  pcieBandwidthGBps: 15.75,    // PCIe Gen3 x16

  hasTransformerEngine: false,  // Pre-Hopper, no Transformer Engine
  hasNvSwitch: false,
};

/**
 * Older/Consumer GPUs for reference
 */
export const V100_32GB: GPUSpec = {
  id: 'v100-32gb',
  name: 'NVIDIA V100 32GB',
  vendor: 'nvidia',
  architecture: 'volta',
  memoryGB: 32,
  memoryBandwidthTBps: 0.9,
  fp32TFLOPS: 15.7,
  tf32TFLOPS: 0,           // No TF32 on Volta
  fp16TFLOPS: 125,
  bf16TFLOPS: 0,           // No BF16 on Volta
  fp8TFLOPS: 0,
  fp4TFLOPS: 0,
  int8TOPS: 0,
  int4TOPS: 0,
  hasTensorCores: true,
  tensorCoreTFLOPS: 125,
  tdpWatts: 300,
  nvlinkBandwidthGBps: 150,
  nvlinkVersion: 2 as number | null,
  pcieBandwidthGBps: 15.75,
  hasTransformerEngine: false,
  hasNvSwitch: true,
};

/**
 * NVIDIA L40S (Ada Lovelace, AD102 — full tensor mode at higher TDP)
 */
export const L40S: GPUSpec = {
  id: 'l40s',
  name: 'NVIDIA L40S',
  vendor: 'nvidia',
  architecture: 'ada',

  memoryGB: 48,
  memoryBandwidthTBps: 0.864,

  fp32TFLOPS: 91.6,
  tf32TFLOPS: 183.2,
  fp16TFLOPS: 362,
  bf16TFLOPS: 362,
  fp8TFLOPS: 733,
  fp4TFLOPS: 0,
  int8TOPS: 733,
  int4TOPS: 1466,

  hasTensorCores: true,
  tensorCoreTFLOPS: 362,

  tdpWatts: 350,

  nvlinkBandwidthGBps: 0,
  nvlinkVersion: null,
  pcieBandwidthGBps: 31.5,

  hasTransformerEngine: true,
  hasNvSwitch: false,
};

/**
 * NVIDIA L40 (Ada Lovelace, AD102 — standard tensor mode)
 */
export const L40: GPUSpec = {
  id: 'l40',
  name: 'NVIDIA L40',
  vendor: 'nvidia',
  architecture: 'ada',

  memoryGB: 48,
  memoryBandwidthTBps: 0.864,

  fp32TFLOPS: 90.52,
  tf32TFLOPS: 90.52,
  fp16TFLOPS: 181,
  bf16TFLOPS: 181,
  fp8TFLOPS: 362,
  fp4TFLOPS: 0,
  int8TOPS: 362,
  int4TOPS: 724,

  hasTensorCores: true,
  tensorCoreTFLOPS: 181,

  tdpWatts: 300,

  nvlinkBandwidthGBps: 0,
  nvlinkVersion: null,
  pcieBandwidthGBps: 31.5,

  hasTransformerEngine: true,
  hasNvSwitch: false,
};

/**
 * NVIDIA A10 (Ampere, GA102)
 */
export const A10: GPUSpec = {
  id: 'a10',
  name: 'NVIDIA A10',
  vendor: 'nvidia',
  architecture: 'ampere',

  memoryGB: 24,
  memoryBandwidthTBps: 0.6,

  fp32TFLOPS: 31.2,
  tf32TFLOPS: 62.5,
  fp16TFLOPS: 125,
  bf16TFLOPS: 125,
  fp8TFLOPS: 0,
  fp4TFLOPS: 0,
  int8TOPS: 250,
  int4TOPS: 500,

  hasTensorCores: true,
  tensorCoreTFLOPS: 125,

  tdpWatts: 150,

  nvlinkBandwidthGBps: 0,
  nvlinkVersion: null,
  pcieBandwidthGBps: 31.5,

  hasTransformerEngine: false,
  hasNvSwitch: false,
};

/**
 * NVIDIA A10G (Ampere, GA10B — AWS custom variant)
 */
export const A10G: GPUSpec = {
  id: 'a10g',
  name: 'NVIDIA A10G',
  vendor: 'nvidia',
  architecture: 'ampere',

  memoryGB: 24,
  memoryBandwidthTBps: 0.6,

  fp32TFLOPS: 31.52,
  tf32TFLOPS: 35,           // Half of BF16 (Ampere convention)
  fp16TFLOPS: 70,           // GA10B die, lower than A10's 125
  bf16TFLOPS: 70,           // Same as FP16 on Ampere
  fp8TFLOPS: 0,
  fp4TFLOPS: 0,
  int8TOPS: 140,            // 2× BF16
  int4TOPS: 280,            // 2× INT8

  hasTensorCores: true,
  tensorCoreTFLOPS: 70,     // BF16 peak

  tdpWatts: 150,

  nvlinkBandwidthGBps: 0,
  nvlinkVersion: null,
  pcieBandwidthGBps: 31.5,

  hasTransformerEngine: false,
  hasNvSwitch: false,
};

/**
 * NVIDIA L4 (Ada Lovelace, AD104)
 */
export const L4: GPUSpec = {
  id: 'l4',
  name: 'NVIDIA L4',
  vendor: 'nvidia',
  architecture: 'ada',

  memoryGB: 24,
  memoryBandwidthTBps: 0.3,

  fp32TFLOPS: 30.3,
  tf32TFLOPS: 60.5,
  fp16TFLOPS: 121,
  bf16TFLOPS: 121,
  fp8TFLOPS: 242,
  fp4TFLOPS: 0,
  int8TOPS: 242,
  int4TOPS: 485,

  hasTensorCores: true,
  tensorCoreTFLOPS: 121,

  tdpWatts: 72,

  nvlinkBandwidthGBps: 0,
  nvlinkVersion: null,
  pcieBandwidthGBps: 31.5,

  hasTransformerEngine: true,
  hasNvSwitch: false,
};

/**
 * NVIDIA RTX 6000 Ada (Ada Lovelace, AD102)
 */
export const RTX_6000_ADA: GPUSpec = {
  id: 'rtx-6000-ada',
  name: 'NVIDIA RTX 6000 Ada',
  vendor: 'nvidia',
  architecture: 'ada',

  memoryGB: 48,
  memoryBandwidthTBps: 0.96,

  fp32TFLOPS: 91.1,
  tf32TFLOPS: 91.1,
  fp16TFLOPS: 182.2,
  bf16TFLOPS: 182.2,
  fp8TFLOPS: 364.4,
  fp4TFLOPS: 0,
  int8TOPS: 364.4,
  int4TOPS: 728.8,

  hasTensorCores: true,
  tensorCoreTFLOPS: 182.2,

  tdpWatts: 300,

  nvlinkBandwidthGBps: 0,
  nvlinkVersion: null,
  pcieBandwidthGBps: 31.5,

  hasTransformerEngine: true,
  hasNvSwitch: false,
};

/**
 * NVIDIA RTX 4090 (Ada Lovelace, AD102 — consumer)
 */
export const RTX_4090: GPUSpec = {
  id: 'rtx-4090',
  name: 'NVIDIA RTX 4090',
  vendor: 'nvidia',
  architecture: 'ada',

  memoryGB: 24,
  memoryBandwidthTBps: 1.008,

  fp32TFLOPS: 82.6,
  tf32TFLOPS: 82.6,
  fp16TFLOPS: 165.2,       // FP16 with FP32 accumulate (training-relevant); marketing "330.3" is pure FP16
  bf16TFLOPS: 165.2,       // BF16 with FP32 accumulate (training-relevant); marketing "330.3" is pure BF16
  fp8TFLOPS: 330.3,
  fp4TFLOPS: 0,
  int8TOPS: 330.3,
  int4TOPS: 660.6,

  hasTensorCores: true,
  tensorCoreTFLOPS: 165.2,

  tdpWatts: 450,

  nvlinkBandwidthGBps: 0,
  nvlinkVersion: null,
  pcieBandwidthGBps: 31.5,

  hasTransformerEngine: true,
  hasNvSwitch: false,
};

/**
 * NVIDIA RTX 3090 (Ampere, GA102 — consumer)
 */
export const RTX_3090: GPUSpec = {
  id: 'rtx-3090',
  name: 'NVIDIA RTX 3090',
  vendor: 'nvidia',
  architecture: 'ampere',

  memoryGB: 24,
  memoryBandwidthTBps: 0.936,

  fp32TFLOPS: 35.6,
  tf32TFLOPS: 71,
  fp16TFLOPS: 142,
  bf16TFLOPS: 142,
  fp8TFLOPS: 0,
  fp4TFLOPS: 0,
  int8TOPS: 284,
  int4TOPS: 568,

  hasTensorCores: true,
  tensorCoreTFLOPS: 142,

  tdpWatts: 350,

  nvlinkBandwidthGBps: 0,
  nvlinkVersion: null,
  pcieBandwidthGBps: 31.5,

  hasTransformerEngine: false,
  hasNvSwitch: false,
};

/**
 * All GPU specifications
 */
export const ALL_GPUS: Record<string, GPUSpec> = {
  'a100-40gb': A100_40GB,
  'a100-80gb': A100_80GB,
  'a800-80gb': A800_80GB,
  'h100-sxm': H100_SXM,
  'h100-pcie': H100_PCIE,
  'h100-nvl': H100_NVL,
  'h800-sxm': H800_SXM,
  'h200-sxm': H200_SXM,
  'b200': B200,
  'gb200': GB200,
  'mi250x': MI250X,
  'mi210': MI210,
  'mi350x': MI350X,
  'mi300x': MI300X,
  'mi325x': MI325X,
  't4': T4,
  'v100-32gb': V100_32GB,
  'l40': L40,
  'l40s': L40S,
  'a10': A10,
  'a10g': A10G,
  'l4': L4,
  'rtx-6000-ada': RTX_6000_ADA,
  'rtx-4090': RTX_4090,
  'rtx-3090': RTX_3090,
};

/**
 * GPU categories for UI
 */
export const GPU_CATEGORIES = {
  datacenter: [
    'a100-40gb',
    'a100-80gb',
    'a800-80gb',
    'h100-pcie',
    'h100-nvl',
    'h100-sxm',
    'h800-sxm',
    'h200-sxm',
  ],
  nextGen: ['b200', 'gb200'],
  amd: ['mi250x', 'mi300x', 'mi325x', 'mi350x'],
  professional: ['a10g', 'a10', 'l4', 'l40', 'l40s', 'rtx-6000-ada'],
  consumer: ['rtx-3090', 'rtx-4090'],
  legacy: ['t4', 'v100-32gb'],
};

/**
 * Get GPU by ID
 */
export function getGPU(id: string): GPUSpec | undefined {
  return ALL_GPUS[id];
}

/**
 * Check if GPU supports Flash Attention
 * Requires Ampere+ (NVIDIA sm_80+) or CDNA2+ (AMD ROCm)
 * Volta and Turing lack the necessary hardware support
 */
export function supportsFlashAttention(gpu: GPUSpec): boolean {
  const unsupported: Set<string> = new Set(['volta', 'turing']);
  return !unsupported.has(gpu.architecture);
}

/**
 * Get effective TFLOPS for a given dtype
 * Returns 0 if the GPU doesn't support that precision
 */
export function getEffectiveTFLOPS(gpu: GPUSpec, dtype: string): number {
  switch (dtype) {
    case 'fp32':
      return gpu.fp32TFLOPS;
    case 'tf32':
      return gpu.tf32TFLOPS || gpu.fp32TFLOPS;
    case 'fp16':
      return gpu.fp16TFLOPS;
    case 'bf16':
      return gpu.bf16TFLOPS || gpu.fp16TFLOPS;
    case 'fp8':
      return gpu.fp8TFLOPS || gpu.bf16TFLOPS || gpu.fp16TFLOPS;
    case 'fp4':
      return gpu.fp4TFLOPS || gpu.fp8TFLOPS || gpu.bf16TFLOPS || gpu.fp16TFLOPS;
    case 'int8':   // W8A16: LLM.int8, bitsandbytes — dequant to BF16, compute at BF16
      return gpu.bf16TFLOPS || gpu.fp16TFLOPS;
    case 'int4':
      return gpu.int4TOPS || gpu.int8TOPS || gpu.bf16TFLOPS || gpu.fp16TFLOPS;
    default:
      return gpu.bf16TFLOPS || gpu.fp16TFLOPS;
  }
}

/**
 * Memory bandwidth scaling factor for training step time.
 *
 * ~15% of training compute time is memory-bandwidth-bound (non-matmul ops:
 * LayerNorm, Softmax, GELU/SiLU, activation I/O). GPUs with higher memory
 * bandwidth relative to their compute spend less time on these ops.
 *
 * Uses Amdahl's law with H100 SXM as reference (factor = 1.0).
 * Returns > 1 for GPUs with relatively more bandwidth per FLOP (faster).
 */
export function getMemoryBandwidthScaling(gpu: GPUSpec, computeDtype: string): number {
  const peakTFLOPS = getEffectiveTFLOPS(gpu, computeDtype);
  if (peakTFLOPS <= 0 || gpu.memoryBandwidthTBps <= 0) return 1.0;

  // Operational intensity: TFLOPS / (TB/s) = FLOP/byte
  const oi = peakTFLOPS / gpu.memoryBandwidthTBps;

  // Reference: H100 SXM (989 bf16 TFLOPS, 3.35 TB/s → OI ≈ 295)
  const referenceOI = 989 / 3.35;

  // Fraction of compute time that's memory-bandwidth-bound at reference OI
  const memBoundFraction = 0.15;

  // Amdahl's law: time = (1-f) + f × (OI/refOI)
  // Higher OI = more compute-heavy relative to BW = more time on mem-bound ops
  const normalizedTime = (1 - memBoundFraction) + memBoundFraction * (oi / referenceOI);
  return 1 / normalizedTime;
}

/**
 * Fraction of training step time spent in matmul ops (vs non-matmul: LayerNorm, softmax, GELU, loss).
 * FP8 only accelerates matmul; non-matmul stays at BF16 rate.
 */
const MATMUL_TIME_FRACTION = 0.80;

/**
 * Training-effective TFLOPS accounting for the matmul/non-matmul split.
 *
 * For FP8/FP4: only matmul ops (80% of step time) get the 2×/4× speedup.
 * Non-matmul ops (LayerNorm, softmax, GELU, loss) stay at BF16 rate.
 *
 * H100 SXM FP8: effectiveSpeedup = 1/(0.80/2 + 0.20) = 1.667×
 *   → trainingTFLOPS = 989 / 0.60 ≈ 1648 (not raw 1979)
 *
 * For BF16/FP16/TF32/FP32: returns getEffectiveTFLOPS unchanged.
 * For GPUs without native FP8 (e.g. A100): matmulSpeedup=1.0, returns BF16 rate.
 *
 * Used for timing calculations. MFU uses getEffectiveTFLOPS(gpu, 'bf16') — BF16 peak regardless of compute dtype (industry convention).
 */
export function getTrainingTFLOPS(gpu: GPUSpec, dtype: string): number {
  const bf16TFLOPS = getEffectiveTFLOPS(gpu, 'bf16');
  const rawTFLOPS = getEffectiveTFLOPS(gpu, dtype);

  // For non-reduced-precision dtypes, no matmul/non-matmul split needed
  if (dtype !== 'fp8' && dtype !== 'fp4') {
    return rawTFLOPS;
  }

  // matmulSpeedup = raw / bf16 (e.g., 1979/989 ≈ 2.0 for H100 FP8)
  // If GPU lacks native support, rawTFLOPS == bf16TFLOPS → speedup = 1.0
  if (bf16TFLOPS <= 0) return rawTFLOPS;
  const matmulSpeedup = rawTFLOPS / bf16TFLOPS;

  if (matmulSpeedup <= 1.0) return rawTFLOPS; // no speedup → return as-is

  // Amdahl's law: time = matmul_frac/speedup + non_matmul_frac
  const effectiveTimeFraction = MATMUL_TIME_FRACTION / matmulSpeedup + (1 - MATMUL_TIME_FRACTION);
  const amdahlTFLOPS = bf16TFLOPS / effectiveTimeFraction;

  // Transformer Engine overhead: FP8 GEMMs require per-tensor quantize/dequantize
  // (FP16→FP8 casting), amax history tracking, and delayed scaling factor updates.
  // Published measurements: ~5-8% overhead. FP4 has additional scaling complexity.
  const TE_OVERHEAD = dtype === 'fp4' ? 0.90 : 0.93;
  return amdahlTFLOPS * TE_OVERHEAD;
}

/**
 * GPU SM saturation factor: models how well a matmul of given dimensions
 * utilizes the GPU's streaming multiprocessors.
 *
 * Uses a smooth curve: factor = min(1, (elements / threshold)^0.3)
 * GPUs degrade gracefully — at 50% of threshold, throughput is ~81%.
 * Only truly tiny matmuls (< 5% of threshold) see catastrophic undersaturation.
 *
 * One threshold table per architecture, with a multiplier for context:
 * - Kernel-level (multiplier=1): individual GEMM saturation (expert GEMMs, inference).
 *   Threshold ~1M for Hopper — "can this single kernel fill all 132 SMs?"
 * - Global efficiency (multiplier=5): used in the multiplicative efficiency model
 *   (saturation × memBW × residual) in computeComputeEfficiency(). Higher threshold
 *   provides a 5-12% gradient across realistic training configs.
 *
 * Both model the same physics (SM occupancy vs problem size), unified with one
 * threshold table to avoid over-parameterization.
 */
function saturationFactor(
  matmulElements: number,
  gpu: GPUSpec,
  thresholdMultiplier: number = 1.0,
): number {
  const threshold = getGPUSaturationThreshold(gpu) * thresholdMultiplier;
  const ratio = matmulElements / threshold;
  if (ratio >= 1.0) return 1.0;
  return Math.pow(ratio, 0.3);
}

/**
 * Kernel-level matmul saturation (threshold multiplier = 1×).
 * Used for individual GEMM sizing: expert GEMMs in 3d-parallel.ts, inference prefill/decode.
 */
export function getMatmulSaturationFactor(
  tokensPerMicroBatch: number,
  hiddenSizePerGPU: number,
  gpu: GPUSpec
): number {
  return saturationFactor(tokensPerMicroBatch * hiddenSizePerGPU, gpu, 1.0);
}

// Exact ratio between compute-efficiency and kernel-level saturation thresholds.
// Both model SM occupancy; compute-efficiency uses raised thresholds (≈4.77×)
// for a 5-12% gradient in the multiplicative efficiency model.
const COMPUTE_THRESHOLD_MULT = 5_000_000 / 1_048_576; // ≈ 4.77

/**
 * Global compute-efficiency saturation (threshold multiplier ≈ 4.77×).
 * Used in computeComputeEfficiency() where the raised threshold provides a
 * 5-12% gradient across realistic training configs (MBS≥1, seq≥2048, TP≤8).
 */
export function getComputeSaturationFactor(
  tokensPerMicroBatch: number,
  hiddenSizePerGPU: number,
  gpu: GPUSpec
): number {
  return saturationFactor(tokensPerMicroBatch * hiddenSizePerGPU, gpu, COMPUTE_THRESHOLD_MULT);
}

/**
 * Problem size (M × K product) where tensor cores reach ~100% throughput.
 * Calibrated against published training benchmarks: thresholds set so that
 * standard configs (MBS≥1, seq≥2048, TP≤8 on large models) are unaffected.
 */
function getGPUSaturationThreshold(gpu: GPUSpec): number {
  const arch = gpu.architecture?.toLowerCase();
  switch (arch) {
    case 'hopper':    return 1_048_576;   // H100: ~1M elements
    case 'blackwell': return 1_310_720;   // B200: ~1.3M elements
    case 'ampere':    return 786_432;     // A100: ~768K elements
    case 'ada':       return 655_360;     // L40S: ~640K elements
    case 'cdna3':     return 1_048_576;   // MI300X: ~1M elements
    case 'cdna4':     return 1_310_720;   // MI350X: ~1.3M elements
    case 'cdna2':     return 786_432;     // MI250X: ~768K elements
    case 'volta':     return 524_288;     // V100: ~512K elements
    case 'turing':    return 524_288;     // T4: ~512K elements
    default:          return 786_432;     // Conservative default
  }
}

// =========================================================================
// Grouped GEMM efficiency model for fine-grained MoE experts
// =========================================================================

/**
 * Perturbable: power-law exponent for dimension ratio in grouped GEMM efficiency.
 * Controls how steeply efficiency drops as expert intermediate size decreases.
 * Fitted with one anchor: Qwen3 30B-A3B at 241 HFU TFLOPS (NeMo, 16×H100).
 */
let _expertGemmExponent = 0.80;
export function getExpertGemmExponent(): number { return _expertGemmExponent; }
export function setExpertGemmExponent(v: number): void { _expertGemmExponent = v; }

/**
 * Perturbable: scaling factor for expert-count-driven L2 cache pressure.
 * More experts per GPU → more weight matrices thrashing L2 → lower throughput.
 * Fitted with one anchor: Qwen3 30B-A3B at 241 HFU TFLOPS (NeMo, 16×H100).
 */
let _expertCountScale = 0.55;
export function getExpertCountScale(): number { return _expertCountScale; }
export function setExpertCountScale(v: number): void { _expertCountScale = v; }

/**
 * Architecture-dependent threshold: expert intermediate size at or above which
 * grouped GEMM efficiency is at baseline (no extra penalty beyond the 5% overhead
 * that replaces the old GROUPED_GEMM_OVERHEAD constant).
 *
 * Physical basis: expert weight working set vs L2 cache capacity. H100 L2 = 50MB.
 * At expertIntermediateSize=1536 with 8 experts/GPU: 8×3×H×1536×2B fits comfortably.
 * Below 1536, per-expert weight matrices are too small for good tile utilization in
 * grouped GEMM kernels, and cache thrashing between experts dominates.
 *
 * Set at 1536 rather than 2048: models with expertInt≥1536 (DeepSeek V3, Qwen3 235B,
 * GPT-OSS) achieve reasonable grouped GEMM efficiency. Only truly fine-grained
 * models (Qwen3 30B at expertInt=768) fall below this threshold.
 */
export function getExpertGemmDimThreshold(gpu: GPUSpec): number {
  const arch = gpu.architecture?.toLowerCase();
  switch (arch) {
    case 'hopper':    return 1536;
    case 'blackwell': return 1536;  // B200 has larger L2 but also larger tiles
    case 'ampere':    return 1536;
    case 'ada':       return 1536;
    case 'cdna3':     return 1536;
    case 'cdna4':     return 1536;
    case 'cdna2':     return 1536;
    default:          return 1536;
  }
}

/**
 * Grouped GEMM efficiency factor for MoE expert compute.
 *
 * Models throughput degradation of grouped expert GEMMs relative to dense GEMMs.
 * Returns a factor in [floor, baseline] where baseline ≈ 0.952 (= 1/1.05, subsuming
 * the old 5% GROUPED_GEMM_OVERHEAD constant) and floor = 0.30.
 *
 * Uses raw expertIntermediateSize (not divided by TP) — this is an architectural
 * property of the model, not a deployment property. TP slicing distributes computation
 * but doesn't change the fundamental grouped GEMM characteristics.
 *
 * Physical basis (directional):
 * - Total expert weight working set vs L2 cache capacity (H100: 50MB)
 * - Number of expert groups: more groups → more cache thrashing, worse wave scheduling
 * - Per-expert weight matrix size: smaller → less data reuse per group
 *
 * Fitted parameter with one anchor point: Qwen3 30B-A3B at 241 HFU TFLOPS on 16×H100.
 * Directionally validated against published grouped GEMM benchmarks:
 * - SonicMoE (arXiv:2512.14080): grouped expert GEMM achieves 49-57% of H100 BF16 peak
 * - cuBLAS: single large BF16 GEMM (4096²) achieves 72% of peak
 * - DeepGEMM++: grouped forward at similar dims achieves ~49% of peak
 *
 * @param expertIntermediateSize - Raw (not /tp) expert intermediate dimension
 * @param expertsPerGPU - Number of expert weight matrices resident on each GPU
 * @param gpu - GPU spec (for architecture-dependent threshold)
 */
export function getGroupedGemmEfficiency(
  expertIntermediateSize: number,
  expertsPerGPU: number,
  gpu: GPUSpec,
): number {
  const dimThreshold = getExpertGemmDimThreshold(gpu);

  // Baseline: subsumes the old GROUPED_GEMM_OVERHEAD = 0.05 constant.
  // At/above threshold, standard MoE overhead only — no extra penalty.
  const baseLine = 1.0 / 1.05;  // ≈ 0.952
  const floor = 0.30;

  if (expertIntermediateSize >= dimThreshold) {
    return baseLine;
  }

  const dimRatio = expertIntermediateSize / dimThreshold;
  const baseFactor = Math.max(floor, baseLine * Math.pow(dimRatio, _expertGemmExponent));

  // Expert count scaling: more experts per GPU → more L2 pressure from weight thrashing
  if (expertsPerGPU > 8) {
    const countPenalty = 1 + _expertCountScale * Math.log2(expertsPerGPU / 8);
    return Math.max(floor, baseFactor / countPenalty);
  }

  return baseFactor;
}

/**
 * Check if a GPU supports a given precision
 */
export function gpuSupportsPrecision(gpu: GPUSpec, dtype: string): boolean {
  switch (dtype) {
    case 'fp32':
      return gpu.fp32TFLOPS > 0;
    case 'tf32':
      return gpu.tf32TFLOPS > 0;
    case 'fp16':
      return gpu.fp16TFLOPS > 0;
    case 'bf16':
      return gpu.bf16TFLOPS > 0;
    case 'fp8':
      return gpu.fp8TFLOPS > 0;
    case 'fp4':
      return gpu.fp4TFLOPS > 0;
    case 'int8':
      return gpu.int8TOPS > 0;
    case 'int4':
      return gpu.int4TOPS > 0;
    default:
      return false;
  }
}

/**
 * Returns a warning string if the GPU doesn't natively support the given precision
 * and will fall back to a lower precision. Returns null if natively supported.
 */
export function getPrecisionFallbackWarning(gpu: GPUSpec, dtype: string): string | null {
  if (gpuSupportsPrecision(gpu, dtype)) return null;

  // Determine what it falls back to
  const fallbackTFLOPS = getEffectiveTFLOPS(gpu, dtype);
  const bf16TFLOPS = getEffectiveTFLOPS(gpu, 'bf16');
  const fallbackName = fallbackTFLOPS === bf16TFLOPS ? 'BF16' :
    fallbackTFLOPS === gpu.fp16TFLOPS ? 'FP16' :
    fallbackTFLOPS === gpu.fp32TFLOPS ? 'FP32' : 'a lower precision';

  return `${gpu.name} does not have native ${dtype.toUpperCase()} support. Falling back to ${fallbackName} — performance will match ${fallbackName}, not ${dtype.toUpperCase()}.`;
}

