/**
 * Hardware specification types
 */

// GPU vendor
export type GPUVendor = 'nvidia' | 'amd' | 'intel';

// GPU generation/architecture
export type GPUArchitecture =
  | 'turing'      // T4
  | 'volta'       // V100
  | 'ampere'      // A100, A10, A10G, RTX 3090
  | 'ada'         // L40S, L40, L4, RTX 4090, RTX 6000 Ada
  | 'hopper'      // H100, H200
  | 'blackwell'   // B100, B200
  | 'cdna2'       // MI250X, MI210
  | 'cdna3'       // MI300X, MI325X
  | 'cdna4';      // MI350X

/**
 * GPU hardware specification.
 *
 * Unit conventions:
 *   memoryGB            — binary gibibytes (1 "GB" = 2³⁰ bytes = 1 GiB).
 *                         NVIDIA labels HBM capacity as "GB" but ships GiB.
 *                         80 "GB" = 80 GiB = 85,899,345,920 bytes.
 *                         Use gpuCapacityBytes() for byte conversion.
 *   memoryBandwidthTBps — decimal terabytes/sec (1 TB/s = 10¹² bytes/s). JEDEC standard.
 *   *TFLOPS / *TOPS      — 10¹² ops/sec. Always decimal.
 *   nvlinkBandwidthGBps — decimal GB/s per direction (10⁹ bytes/s).
 *   pcieBandwidthGBps   — decimal GB/s (10⁹ bytes/s).
 *
 * Memory uses binary units (GiB labeled "GB"). All other quantities use decimal.
 * The formatBytes() display function uses base-1024 to match.
 */
export interface GPUSpec {
  id: string;
  name: string;
  vendor: GPUVendor;
  architecture: GPUArchitecture;

  // Memory
  memoryGB: number;
  memoryBandwidthTBps: number;  // TB/s

  // Compute (TFLOPS/TOPS)
  fp32TFLOPS: number;
  tf32TFLOPS: number;      // TensorFloat-32 (Ampere+)
  fp16TFLOPS: number;
  bf16TFLOPS: number;
  fp8TFLOPS: number;
  fp4TFLOPS: number;       // FP4 (Blackwell)
  int8TOPS: number;
  int4TOPS: number;        // INT4 for quantized inference

  // Tensor core specs
  hasTensorCores: boolean;
  tensorCoreTFLOPS: number;     // Peak tensor core TFLOPS (usually BF16)

  // Power
  tdpWatts: number;

  // Interconnect capabilities
  nvlinkBandwidthGBps: number;  // Per-direction GB/s (0 if no NVLink)
  nvlinkVersion: number | null; // 3, 4, 5 or null
  pcieBandwidthGBps: number;

  // Special features
  hasTransformerEngine: boolean; // H100+ FP8 support
  hasNvSwitch: boolean;         // Whether typically used with NVSwitch

  // Metadata
  estimated?: boolean;          // true = specs unvalidated against real training benchmarks
}

// Interconnect type
export type InterconnectType =
  | 'nvlink'
  | 'nvswitch'
  | 'pcie'
  | 'infiniband'
  | 'roce'
  | 'ethernet';

// Interconnect specification
export interface InterconnectSpec {
  type: InterconnectType;
  name: string;
  bandwidthGBps: number;       // Unidirectional bandwidth GB/s
  bidirectionalGBps: number;   // Bidirectional bandwidth GB/s
  latencyUs: number;           // Base latency in microseconds
  isFullDuplex: boolean;
}

// NVLink versions
export const NVLINK_SPECS: Record<number, InterconnectSpec> = {
  2: {
    type: 'nvlink',
    name: 'NVLink 2.0',
    bandwidthGBps: 150,
    bidirectionalGBps: 300,
    latencyUs: 1.2,
    isFullDuplex: true,
  },
  3: {
    type: 'nvlink',
    name: 'NVLink 3.0',
    bandwidthGBps: 300,
    bidirectionalGBps: 600,
    latencyUs: 1,
    isFullDuplex: true,
  },
  4: {
    type: 'nvlink',
    name: 'NVLink 4.0',
    bandwidthGBps: 450,
    bidirectionalGBps: 900,
    latencyUs: 0.8,
    isFullDuplex: true,
  },
  5: {
    type: 'nvlink',
    name: 'NVLink 5.0',
    bandwidthGBps: 900,
    bidirectionalGBps: 1800,
    latencyUs: 0.6,
    isFullDuplex: true,
  },
};

// InfiniBand specs
export interface InfiniBandSpec extends InterconnectSpec {
  type: 'infiniband';
  version: 'hdr' | 'ndr' | 'xdr';
  portsPerSwitch: number;
  perPortBandwidthGBps: number;  // Single port bandwidth
}

// InfiniBand specs - bandwidthGBps is AGGREGATE for a DGX node (8 NICs)
// DGX systems use 8 network adapters to match internal GPU bandwidth
export const INFINIBAND_SPECS: Record<string, InfiniBandSpec> = {
  hdr: {
    type: 'infiniband',
    name: 'InfiniBand HDR (8x)',
    version: 'hdr',
    perPortBandwidthGBps: 25,      // 200 Gb/s = 25 GB/s per port
    bandwidthGBps: 200,            // 8 ports × 25 GB/s = 200 GB/s aggregate
    bidirectionalGBps: 400,        // Full duplex aggregate
    latencyUs: 0.6,
    isFullDuplex: true,
    portsPerSwitch: 40,
  },
  ndr: {
    type: 'infiniband',
    name: 'InfiniBand NDR (8x)',
    version: 'ndr',
    perPortBandwidthGBps: 50,      // 400 Gb/s = 50 GB/s per port
    bandwidthGBps: 400,            // 8 ports × 50 GB/s = 400 GB/s aggregate
    bidirectionalGBps: 800,        // Full duplex aggregate
    latencyUs: 0.5,
    isFullDuplex: true,
    portsPerSwitch: 64,
  },
  xdr: {
    type: 'infiniband',
    name: 'InfiniBand XDR (8x)',
    version: 'xdr',
    perPortBandwidthGBps: 100,     // 800 Gb/s = 100 GB/s per port
    bandwidthGBps: 800,            // 8 ports × 100 GB/s = 800 GB/s aggregate
    bidirectionalGBps: 1600,       // Full duplex aggregate
    latencyUs: 0.4,
    isFullDuplex: true,
    portsPerSwitch: 72,        // NVIDIA Quantum-X800 Q3200-RA
  },
};

// Node configuration (single server/machine)
export interface NodeSpec {
  id: string;
  name: string;
  numGPUs: number;
  gpu: GPUSpec;
  intraNodeInterconnect: InterconnectSpec;  // NVLink/NVSwitch
  interNodeInterconnect: InterconnectSpec;  // IB/RoCE
  cpuCores: number;
  ramGB: number;
  hasNvSwitch: boolean;
  numNICs: number;  // Number of network adapters (NICs) per node
}

// Cluster topology types
export type TopologyType =
  | 'single-node'
  | 'fat-tree'
  | 'dragonfly'
  | 'torus'
  | 'ring';

// Cluster configuration
export interface ClusterConfig {
  id: string;
  name: string;
  numNodes: number;
  gpusPerNode: number;
  totalGPUs: number;
  node: NodeSpec;
  topology: TopologyType;
  interNodeBandwidthGBps: number;
  interNodeLatencyUs: number;

  // Derived properties
  totalMemoryGB: number;
  totalTFLOPS: number;
}

// GPU utilization state
export type GPUState =
  | 'idle'
  | 'compute-forward'
  | 'compute-backward'
  | 'compute-optimizer'
  | 'memory-transfer'
  | 'communication';

// GPU runtime metrics
export interface GPUMetrics {
  id: number;
  state: GPUState;
  memoryUsedGB: number;
  memoryTotalGB: number;
  computeUtilization: number;  // 0-1
  memoryUtilization: number;   // 0-1
  powerWatts: number;
  temperatureC: number;
}

// Cluster runtime metrics
export interface ClusterMetrics {
  gpus: GPUMetrics[];
  totalMemoryUsedGB: number;
  totalMemoryGB: number;
  avgComputeUtilization: number;
  avgMemoryUtilization: number;
  totalPowerWatts: number;
  networkUtilization: number;  // 0-1
}
