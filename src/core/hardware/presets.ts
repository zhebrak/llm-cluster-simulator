/**
 * Hardware presets for common configurations
 * DGX systems and cloud instances
 */

import type { NodeSpec, ClusterConfig } from '../../types/index.ts';
import {
  A100_80GB,
  H100_SXM,
  H100_PCIE,
  H200_SXM,
  B200,
  GB200,
  MI250X,
  MI300X,
  MI325X,
  MI350X,
  T4,
  L4,
  L40S,
  A10G,
} from './gpu.ts';
import { NVSWITCH_SPECS, INFINITY_FABRIC_SPECS, PCIE_SPECS, ETHERNET_SPECS, ROCE_SPECS } from './interconnect.ts';
import { INFINIBAND_SPECS } from '../../types/index.ts';
import { createCluster } from './topology.ts';

/**
 * NVIDIA DGX Systems
 */
export const DGX_A100: NodeSpec = {
  id: 'dgx-a100',
  name: 'DGX A100',
  numGPUs: 8,
  gpu: A100_80GB,
  intraNodeInterconnect: NVSWITCH_SPECS['nvswitch-a100'],
  interNodeInterconnect: INFINIBAND_SPECS.hdr,
  cpuCores: 128, // 2x AMD EPYC 7742
  ramGB: 2048,
  hasNvSwitch: true,
  numNICs: 8,
};

export const DGX_H100: NodeSpec = {
  id: 'dgx-h100',
  name: 'DGX H100',
  numGPUs: 8,
  gpu: H100_SXM,
  intraNodeInterconnect: NVSWITCH_SPECS['nvswitch-h100'],
  interNodeInterconnect: INFINIBAND_SPECS.ndr,
  cpuCores: 112, // 2x Intel Xeon Platinum 8480C
  ramGB: 2048,
  hasNvSwitch: true,
  numNICs: 8,
};

export const DGX_H200: NodeSpec = {
  id: 'dgx-h200',
  name: 'DGX H200',
  numGPUs: 8,
  gpu: H200_SXM,
  intraNodeInterconnect: NVSWITCH_SPECS['nvswitch-h100'],
  interNodeInterconnect: INFINIBAND_SPECS.ndr,
  cpuCores: 112,
  ramGB: 2048,
  hasNvSwitch: true,
  numNICs: 8,
};

export const DGX_B200: NodeSpec = {
  id: 'dgx-b200',
  name: 'DGX B200',
  numGPUs: 8,
  gpu: B200,
  intraNodeInterconnect: NVSWITCH_SPECS['nvswitch-b200'],
  interNodeInterconnect: INFINIBAND_SPECS.xdr,
  cpuCores: 144,
  ramGB: 4096,
  hasNvSwitch: true,
  numNICs: 8,
};

/**
 * GB200 NVL72 compute tray — 4 GPUs per tray, 18 trays per rack (72 total).
 * All 72 GPUs share a single NVLink 5 domain via NVSwitch.
 */
export const GB200_NVL72_TRAY: NodeSpec = {
  id: 'gb200-nvl72-tray',
  name: 'GB200 NVL72 Tray',
  numGPUs: 4,
  gpu: GB200,
  intraNodeInterconnect: NVSWITCH_SPECS['nvswitch-b200'],
  interNodeInterconnect: INFINIBAND_SPECS.xdr,
  cpuCores: 72, // 2 Grace CPUs per tray (72 Arm cores each)
  ramGB: 960,   // 480 GB LPDDR5X per Grace CPU × 2
  hasNvSwitch: true,
  numNICs: 4,
};

/**
 * AMD Instinct Platform Nodes
 */

// MI250X: Frontier-style node — 4 MI250X cards = 8 GCDs per node
// Slingshot-11 inter-node (~25 GB/s), approximated as IB HDR
export const AMD_MI250X_NODE: NodeSpec = {
  id: 'amd-mi250x-node',
  name: 'AMD MI250X Node',
  numGPUs: 8, // 8 GCDs (4 physical cards)
  gpu: MI250X,
  intraNodeInterconnect: INFINITY_FABRIC_SPECS['if-mi250'],
  interNodeInterconnect: INFINIBAND_SPECS.hdr,
  cpuCores: 128, // 2x AMD EPYC 7763 (Frontier)
  ramGB: 512,
  hasNvSwitch: false,
  numNICs: 4, // 4 Slingshot NICs (Frontier)
};

// MI300X: 8x OAM per node, Infinity Fabric intra-node
export const AMD_MI300X_NODE: NodeSpec = {
  id: 'amd-mi300x-node',
  name: 'AMD MI300X Node',
  numGPUs: 8,
  gpu: MI300X,
  intraNodeInterconnect: INFINITY_FABRIC_SPECS['if-mi300'],
  interNodeInterconnect: INFINIBAND_SPECS.ndr,
  cpuCores: 192, // 2x AMD EPYC 9654
  ramGB: 1536,
  hasNvSwitch: false,
  numNICs: 8,
};

// MI325X: Same platform as MI300X, upgraded HBM3e
export const AMD_MI325X_NODE: NodeSpec = {
  id: 'amd-mi325x-node',
  name: 'AMD MI325X Node',
  numGPUs: 8,
  gpu: MI325X,
  intraNodeInterconnect: INFINITY_FABRIC_SPECS['if-mi300'],
  interNodeInterconnect: INFINIBAND_SPECS.ndr,
  cpuCores: 192,
  ramGB: 1536,
  hasNvSwitch: false,
  numNICs: 8,
};

// MI350X: Next-gen CDNA4, 8x OAM per node
export const AMD_MI350X_NODE: NodeSpec = {
  id: 'amd-mi350x-node',
  name: 'AMD MI350X Node',
  numGPUs: 8,
  gpu: MI350X,
  intraNodeInterconnect: INFINITY_FABRIC_SPECS['if-mi350'],
  interNodeInterconnect: INFINIBAND_SPECS.xdr,
  cpuCores: 192,
  ramGB: 2048,
  hasNvSwitch: false,
  numNICs: 8,
};

/**
 * PCIe Inference Servers
 * Standard rack servers with PCIe GPUs — typical for inference workloads.
 * No NVSwitch; intra-node communication over PCIe bus.
 */

// T4: Budget inference, small/quantized models. 70W, PCIe Gen3, GDDR6.
export const PCIE_T4_SERVER: NodeSpec = {
  id: 'pcie-t4',
  name: 'T4 PCIe Server',
  numGPUs: 4,
  gpu: T4,
  intraNodeInterconnect: PCIE_SPECS['pcie-gen3'],
  interNodeInterconnect: ETHERNET_SPECS['eth-25g'],
  cpuCores: 64,
  ramGB: 256,
  hasNvSwitch: false,
  numNICs: 1,
};

// L4: Cost-efficient inference, successor to T4. 72W, PCIe Gen4. (AWS g6, GCP g2)
export const PCIE_L4_SERVER: NodeSpec = {
  id: 'pcie-l4',
  name: 'L4 PCIe Server',
  numGPUs: 8,
  gpu: L4,
  intraNodeInterconnect: PCIE_SPECS['pcie-gen4'],
  interNodeInterconnect: ETHERNET_SPECS['eth-100g'],
  cpuCores: 96,
  ramGB: 384,
  hasNvSwitch: false,
  numNICs: 1,
};

// A10G: AWS g5 instances. 24GB GDDR6, 150W, PCIe Gen4.
export const PCIE_A10G_SERVER: NodeSpec = {
  id: 'pcie-a10g',
  name: 'A10G PCIe Server',
  numGPUs: 4,
  gpu: A10G,
  intraNodeInterconnect: PCIE_SPECS['pcie-gen4'],
  interNodeInterconnect: ETHERNET_SPECS['eth-100g'],
  cpuCores: 96,
  ramGB: 768,
  hasNvSwitch: false,
  numNICs: 1,
};

// L40S: 48GB GDDR6, Transformer Engine, FP8. (AWS g6e)
export const PCIE_L40S_SERVER: NodeSpec = {
  id: 'pcie-l40s',
  name: 'L40S PCIe Server',
  numGPUs: 8,
  gpu: L40S,
  intraNodeInterconnect: PCIE_SPECS['pcie-gen4'],
  interNodeInterconnect: ETHERNET_SPECS['eth-100g'],
  cpuCores: 128,
  ramGB: 1024,
  hasNvSwitch: false,
  numNICs: 1,
};

// A100 80GB PCIe server (no NVSwitch, unlike DGX A100)
export const PCIE_A100_SERVER: NodeSpec = {
  id: 'pcie-a100',
  name: 'A100 PCIe Server',
  numGPUs: 8,
  gpu: A100_80GB,
  intraNodeInterconnect: PCIE_SPECS['pcie-gen4'],
  interNodeInterconnect: ROCE_SPECS['roce-200g'],
  cpuCores: 96,
  ramGB: 1024,
  hasNvSwitch: false,
  numNICs: 1,
};

// H100 PCIe server (80GB, no NVSwitch)
export const PCIE_H100_SERVER: NodeSpec = {
  id: 'pcie-h100',
  name: 'H100 PCIe Server',
  numGPUs: 8,
  gpu: H100_PCIE,
  intraNodeInterconnect: PCIE_SPECS['pcie-gen5'],
  interNodeInterconnect: ROCE_SPECS['roce-400g'],
  cpuCores: 128,
  ramGB: 1024,
  hasNvSwitch: false,
  numNICs: 1,
};

/**
 * NVIDIA DGX SuperPOD configurations
 */
export const DGX_SUPERPOD_A100 = (numNodes: number): ClusterConfig =>
  createCluster(DGX_A100, numNodes, 'fat-tree', `superpod-a100-${numNodes}`);

export const DGX_SUPERPOD_H100 = (numNodes: number): ClusterConfig =>
  createCluster(DGX_H100, numNodes, 'fat-tree', `superpod-h100-${numNodes}`);

/**
 * All DGX nodes
 */
export const ALL_DGX_NODES: Record<string, NodeSpec> = {
  'dgx-a100': DGX_A100,
  'dgx-h100': DGX_H100,
  'dgx-h200': DGX_H200,
  'dgx-b200': DGX_B200,
};

/**
 * Preset cluster configurations for quick selection
 */
export const CLUSTER_PRESETS: Record<string, () => ClusterConfig> = {
  // Single GPU
  '1x-a100': () => createCluster(
    { ...DGX_A100, numGPUs: 1 } as NodeSpec,
    1,
    'single-node',
    '1x-a100'
  ),
  '1x-h100': () => createCluster(
    { ...DGX_H100, numGPUs: 1 } as NodeSpec,
    1,
    'single-node',
    '1x-h100'
  ),
  '1x-h200': () => createCluster(
    { ...DGX_H200, numGPUs: 1 } as NodeSpec,
    1,
    'single-node',
    '1x-h200'
  ),
  '1x-b200': () => createCluster(
    { ...DGX_B200, numGPUs: 1 } as NodeSpec,
    1,
    'single-node',
    '1x-b200'
  ),

  // Small clusters (single node)
  '8x-a100': () => createCluster(DGX_A100, 1, 'single-node', '8x-a100'),
  '8x-h100': () => createCluster(DGX_H100, 1, 'single-node', '8x-h100'),
  '8x-h200': () => createCluster(DGX_H200, 1, 'single-node', '8x-h200'),
  '8x-b200': () => createCluster(DGX_B200, 1, 'single-node', '8x-b200'),

  // Medium clusters (multi-node)
  '16x-a100': () => createCluster(DGX_A100, 2, 'fat-tree', '16x-a100'),
  '16x-h100': () => createCluster(DGX_H100, 2, 'fat-tree', '16x-h100'),
  '16x-h200': () => createCluster(DGX_H200, 2, 'fat-tree', '16x-h200'),
  '16x-b200': () => createCluster(DGX_B200, 2, 'fat-tree', '16x-b200'),

  '32x-a100': () => createCluster(DGX_A100, 4, 'fat-tree', '32x-a100'),
  '32x-h100': () => createCluster(DGX_H100, 4, 'fat-tree', '32x-h100'),
  '32x-h200': () => createCluster(DGX_H200, 4, 'fat-tree', '32x-h200'),
  '32x-b200': () => createCluster(DGX_B200, 4, 'fat-tree', '32x-b200'),

  '64x-a100': () => createCluster(DGX_A100, 8, 'fat-tree', '64x-a100'),
  '64x-h100': () => createCluster(DGX_H100, 8, 'fat-tree', '64x-h100'),
  '64x-h200': () => createCluster(DGX_H200, 8, 'fat-tree', '64x-h200'),
  '64x-b200': () => createCluster(DGX_B200, 8, 'fat-tree', '64x-b200'),

  // Large clusters
  '128x-a100': () => createCluster(DGX_A100, 16, 'fat-tree', '128x-a100'),
  '128x-h100': () => createCluster(DGX_H100, 16, 'fat-tree', '128x-h100'),
  '128x-h200': () => createCluster(DGX_H200, 16, 'fat-tree', '128x-h200'),
  '128x-b200': () => createCluster(DGX_B200, 16, 'fat-tree', '128x-b200'),

  '256x-a100': () => createCluster(DGX_A100, 32, 'fat-tree', '256x-a100'),
  '256x-h100': () => createCluster(DGX_H100, 32, 'fat-tree', '256x-h100'),
  '256x-h200': () => createCluster(DGX_H200, 32, 'fat-tree', '256x-h200'),
  '256x-b200': () => createCluster(DGX_B200, 32, 'fat-tree', '256x-b200'),

  '512x-a100': () => createCluster(DGX_A100, 64, 'fat-tree', '512x-a100'),
  '512x-h100': () => createCluster(DGX_H100, 64, 'fat-tree', '512x-h100'),
  '512x-h200': () => createCluster(DGX_H200, 64, 'fat-tree', '512x-h200'),
  '512x-b200': () => createCluster(DGX_B200, 64, 'fat-tree', '512x-b200'),

  // XL clusters
  '1024x-a100': () => createCluster(DGX_A100, 128, 'fat-tree', '1024x-a100'),
  '1024x-h100': () => createCluster(DGX_H100, 128, 'fat-tree', '1024x-h100'),
  '1024x-h200': () => createCluster(DGX_H200, 128, 'fat-tree', '1024x-h200'),
  '1024x-b200': () => createCluster(DGX_B200, 128, 'fat-tree', '1024x-b200'),

  '2048x-a100': () => createCluster(DGX_A100, 256, 'fat-tree', '2048x-a100'),
  '2048x-h100': () => createCluster(DGX_H100, 256, 'fat-tree', '2048x-h100'),
  '2048x-h200': () => createCluster(DGX_H200, 256, 'fat-tree', '2048x-h200'),
  '2048x-b200': () => createCluster(DGX_B200, 256, 'fat-tree', '2048x-b200'),

  '4096x-a100': () => createCluster(DGX_A100, 512, 'fat-tree', '4096x-a100'),
  '4096x-h100': () => createCluster(DGX_H100, 512, 'fat-tree', '4096x-h100'),
  '4096x-h200': () => createCluster(DGX_H200, 512, 'fat-tree', '4096x-h200'),
  '4096x-b200': () => createCluster(DGX_B200, 512, 'fat-tree', '4096x-b200'),

  // AMD MI300X (8 per node)
  '8x-mi300x': () => createCluster(AMD_MI300X_NODE, 1, 'single-node', '8x-mi300x'),
  '16x-mi300x': () => createCluster(AMD_MI300X_NODE, 2, 'fat-tree', '16x-mi300x'),
  '32x-mi300x': () => createCluster(AMD_MI300X_NODE, 4, 'fat-tree', '32x-mi300x'),
  '64x-mi300x': () => createCluster(AMD_MI300X_NODE, 8, 'fat-tree', '64x-mi300x'),
  '128x-mi300x': () => createCluster(AMD_MI300X_NODE, 16, 'fat-tree', '128x-mi300x'),
  '256x-mi300x': () => createCluster(AMD_MI300X_NODE, 32, 'fat-tree', '256x-mi300x'),
  '512x-mi300x': () => createCluster(AMD_MI300X_NODE, 64, 'fat-tree', '512x-mi300x'),
  '1024x-mi300x': () => createCluster(AMD_MI300X_NODE, 128, 'fat-tree', '1024x-mi300x'),

};

/**
 * Preset labels for UI display
 */
export const PRESET_LABELS: Record<string, string> = {
  '1x-a100': '1x A100 80GB',
  '1x-h100': '1x H100 SXM',
  '1x-h200': '1x H200 SXM',
  '1x-b200': '1x B200',
  '8x-a100': '8x A100 80GB (DGX A100)',
  '8x-h100': '8x H100 SXM (DGX H100)',
  '8x-h200': '8x H200 SXM (DGX H200)',
  '8x-b200': '8x B200 (DGX B200)',
  '16x-a100': '16x A100 80GB',
  '16x-h100': '16x H100 SXM',
  '16x-h200': '16x H200 SXM',
  '16x-b200': '16x B200',
  '32x-a100': '32x A100 80GB',
  '32x-h100': '32x H100 SXM',
  '32x-h200': '32x H200 SXM',
  '32x-b200': '32x B200',
  '64x-a100': '64x A100 80GB',
  '64x-h100': '64x H100 SXM',
  '64x-h200': '64x H200 SXM',
  '64x-b200': '64x B200',
  '128x-a100': '128x A100 80GB',
  '128x-h100': '128x H100 SXM',
  '128x-h200': '128x H200 SXM',
  '128x-b200': '128x B200',
  '256x-a100': '256x A100 80GB',
  '256x-h100': '256x H100 SXM',
  '256x-h200': '256x H200 SXM',
  '256x-b200': '256x B200',
  '512x-a100': '512x A100 80GB',
  '512x-h100': '512x H100 SXM',
  '512x-h200': '512x H200 SXM',
  '512x-b200': '512x B200',
  '1024x-a100': '1024x A100 80GB',
  '1024x-h100': '1024x H100 SXM',
  '1024x-h200': '1024x H200 SXM',
  '1024x-b200': '1024x B200',
  '2048x-a100': '2048x A100 80GB',
  '2048x-h100': '2048x H100 SXM',
  '2048x-h200': '2048x H200 SXM',
  '2048x-b200': '2048x B200',
  '4096x-a100': '4096x A100 80GB',
  '4096x-h100': '4096x H100 SXM',
  '4096x-h200': '4096x H200 SXM',
  '4096x-b200': '4096x B200',

  // AMD MI300X / MI325X / MI350X
  '8x-mi300x': '8x MI300X (1 node)',
  '16x-mi300x': '16x MI300X',
  '32x-mi300x': '32x MI300X',
  '64x-mi300x': '64x MI300X',
  '128x-mi300x': '128x MI300X',
  '256x-mi300x': '256x MI300X',
  '512x-mi300x': '512x MI300X',
  '1024x-mi300x': '1024x MI300X',
};

/**
 * Preset categories for grouped UI display
 */
export const PRESET_CATEGORIES: { name: string; presets: string[] }[] = [
  {
    name: '1–8 GPUs',
    presets: [
      '1x-a100', '1x-h100', '1x-h200', '1x-b200',
      '8x-a100', '8x-h100', '8x-h200', '8x-mi300x', '8x-b200',
    ],
  },
  {
    name: '16–64 GPUs',
    presets: [
      '16x-a100', '16x-h100', '16x-h200', '16x-mi300x', '16x-b200',
      '32x-a100', '32x-h100', '32x-h200', '32x-mi300x', '32x-b200',
      '64x-a100', '64x-h100', '64x-h200', '64x-mi300x', '64x-b200',
    ],
  },
  {
    name: '128–512 GPUs',
    presets: [
      '128x-a100', '128x-h100', '128x-h200', '128x-mi300x', '128x-b200',
      '256x-a100', '256x-h100', '256x-h200', '256x-mi300x', '256x-b200',
      '512x-a100', '512x-h100', '512x-h200', '512x-mi300x', '512x-b200',
    ],
  },
  {
    name: '1024–4096 GPUs',
    presets: [
      '1024x-a100', '1024x-h100', '1024x-h200', '1024x-mi300x', '1024x-b200',
      '2048x-a100', '2048x-h100', '2048x-h200', '2048x-b200',
      '4096x-a100', '4096x-h100', '4096x-h200', '4096x-b200',
    ],
  },
];

/**
 * Inference cluster presets — real-world inference GPU configurations.
 * Uses PCIe servers for 1–4 GPU configs (T4, L4, A10G, L40S, H100 PCIe),
 * DGX/HGX nodes for 8 GPU configs, and multi-node for 16–32 GPUs.
 */
export const INFERENCE_CLUSTER_PRESETS: Record<string, () => ClusterConfig> = {
  // 1 GPU — budget / small models
  'inf-1x-t4': () => createCluster({ ...PCIE_T4_SERVER, numGPUs: 1 }, 1, 'single-node', 'inf-1x-t4'),
  'inf-1x-l4': () => createCluster({ ...PCIE_L4_SERVER, numGPUs: 1 }, 1, 'single-node', 'inf-1x-l4'),
  'inf-1x-a10g': () => createCluster({ ...PCIE_A10G_SERVER, numGPUs: 1 }, 1, 'single-node', 'inf-1x-a10g'),
  'inf-1x-l40s': () => createCluster({ ...PCIE_L40S_SERVER, numGPUs: 1 }, 1, 'single-node', 'inf-1x-l40s'),
  'inf-1x-a100': () => createCluster({ ...PCIE_A100_SERVER, numGPUs: 1 }, 1, 'single-node', 'inf-1x-a100'),
  'inf-1x-h100': () => createCluster({ ...PCIE_H100_SERVER, numGPUs: 1 }, 1, 'single-node', 'inf-1x-h100'),

  // 2 GPUs — TP=2 for 13B-class or multi-replica
  'inf-2x-l4': () => createCluster({ ...PCIE_L4_SERVER, numGPUs: 2 }, 1, 'single-node', 'inf-2x-l4'),
  'inf-2x-a10g': () => createCluster({ ...PCIE_A10G_SERVER, numGPUs: 2 }, 1, 'single-node', 'inf-2x-a10g'),
  'inf-2x-l40s': () => createCluster({ ...PCIE_L40S_SERVER, numGPUs: 2 }, 1, 'single-node', 'inf-2x-l40s'),
  'inf-2x-a100': () => createCluster({ ...PCIE_A100_SERVER, numGPUs: 2 }, 1, 'single-node', 'inf-2x-a100'),
  'inf-2x-h100': () => createCluster({ ...PCIE_H100_SERVER, numGPUs: 2 }, 1, 'single-node', 'inf-2x-h100'),

  // 4 GPUs — TP=4 for 70B-class models
  'inf-4x-l4': () => createCluster({ ...PCIE_L4_SERVER, numGPUs: 4 }, 1, 'single-node', 'inf-4x-l4'),
  'inf-4x-a10g': () => createCluster({ ...PCIE_A10G_SERVER, numGPUs: 4 }, 1, 'single-node', 'inf-4x-a10g'),
  'inf-4x-l40s': () => createCluster({ ...PCIE_L40S_SERVER, numGPUs: 4 }, 1, 'single-node', 'inf-4x-l40s'),
  'inf-4x-a100': () => createCluster({ ...PCIE_A100_SERVER, numGPUs: 4 }, 1, 'single-node', 'inf-4x-a100'),
  'inf-4x-h100': () => createCluster({ ...PCIE_H100_SERVER, numGPUs: 4 }, 1, 'single-node', 'inf-4x-h100'),

  // 8 GPUs — full node (DGX/HGX with NVSwitch, or PCIe servers)
  'inf-8x-l40s': () => createCluster(PCIE_L40S_SERVER, 1, 'single-node', 'inf-8x-l40s'),
  'inf-8x-a100': () => createCluster(DGX_A100, 1, 'single-node', 'inf-8x-a100'),
  'inf-8x-h100': () => createCluster(DGX_H100, 1, 'single-node', 'inf-8x-h100'),
  'inf-8x-mi300x': () => createCluster(AMD_MI300X_NODE, 1, 'single-node', 'inf-8x-mi300x'),

  // 16 GPUs — 2 nodes, multi-replica or PP serving
  'inf-16x-l40s': () => createCluster(PCIE_L40S_SERVER, 2, 'fat-tree', 'inf-16x-l40s'),
  'inf-16x-a100': () => createCluster(DGX_A100, 2, 'fat-tree', 'inf-16x-a100'),
  'inf-16x-h100': () => createCluster(DGX_H100, 2, 'fat-tree', 'inf-16x-h100'),
  'inf-16x-mi300x': () => createCluster(AMD_MI300X_NODE, 2, 'fat-tree', 'inf-16x-mi300x'),

  // 32 GPUs — 4 nodes, large-scale serving
  'inf-32x-l40s': () => createCluster(PCIE_L40S_SERVER, 4, 'fat-tree', 'inf-32x-l40s'),
  'inf-32x-a100': () => createCluster(DGX_A100, 4, 'fat-tree', 'inf-32x-a100'),
  'inf-32x-h100': () => createCluster(DGX_H100, 4, 'fat-tree', 'inf-32x-h100'),
  'inf-32x-mi300x': () => createCluster(AMD_MI300X_NODE, 4, 'fat-tree', 'inf-32x-mi300x'),

  // 64 GPUs — 8 nodes
  'inf-64x-a100': () => createCluster(DGX_A100, 8, 'fat-tree', 'inf-64x-a100'),
  'inf-64x-h100': () => createCluster(DGX_H100, 8, 'fat-tree', 'inf-64x-h100'),
  'inf-64x-mi300x': () => createCluster(AMD_MI300X_NODE, 8, 'fat-tree', 'inf-64x-mi300x'),

  // 128 GPUs — 16 nodes
  'inf-128x-a100': () => createCluster(DGX_A100, 16, 'fat-tree', 'inf-128x-a100'),
  'inf-128x-h100': () => createCluster(DGX_H100, 16, 'fat-tree', 'inf-128x-h100'),
  'inf-128x-mi300x': () => createCluster(AMD_MI300X_NODE, 16, 'fat-tree', 'inf-128x-mi300x'),
};

// Merge inference presets into CLUSTER_PRESETS so getPresetCluster() works for both modes
Object.assign(CLUSTER_PRESETS, INFERENCE_CLUSTER_PRESETS);

/**
 * Inference preset labels
 */
export const INFERENCE_PRESET_LABELS: Record<string, string> = {
  // 1 GPU
  'inf-1x-t4': '1x T4',
  'inf-1x-l4': '1x L4',
  'inf-1x-a10g': '1x A10G',
  'inf-1x-l40s': '1x L40S',
  'inf-1x-a100': '1x A100 80GB',
  'inf-1x-h100': '1x H100 PCIe',
  // 2 GPUs
  'inf-2x-l4': '2x L4',
  'inf-2x-a10g': '2x A10G',
  'inf-2x-l40s': '2x L40S',
  'inf-2x-a100': '2x A100 80GB',
  'inf-2x-h100': '2x H100 PCIe',
  // 4 GPUs
  'inf-4x-l4': '4x L4',
  'inf-4x-a10g': '4x A10G',
  'inf-4x-l40s': '4x L40S',
  'inf-4x-a100': '4x A100 80GB',
  'inf-4x-h100': '4x H100 PCIe',
  // 8 GPUs
  'inf-8x-l40s': '8x L40S',
  'inf-8x-a100': '8x A100 80GB SXM',
  'inf-8x-h100': '8x H100 SXM',
  'inf-8x-mi300x': '8x MI300X',
  // 16 GPUs
  'inf-16x-l40s': '16x L40S',
  'inf-16x-a100': '16x A100 80GB SXM',
  'inf-16x-h100': '16x H100 SXM',
  'inf-16x-mi300x': '16x MI300X',
  // 32 GPUs
  'inf-32x-l40s': '32x L40S',
  'inf-32x-a100': '32x A100 80GB SXM',
  'inf-32x-h100': '32x H100 SXM',
  'inf-32x-mi300x': '32x MI300X',
  // 64 GPUs
  'inf-64x-a100': '64x A100 80GB SXM',
  'inf-64x-h100': '64x H100 SXM',
  'inf-64x-mi300x': '64x MI300X',
  // 128 GPUs
  'inf-128x-a100': '128x A100 80GB SXM',
  'inf-128x-h100': '128x H100 SXM',
  'inf-128x-mi300x': '128x MI300X',
};

// Merge inference labels into PRESET_LABELS
Object.assign(PRESET_LABELS, INFERENCE_PRESET_LABELS);

/**
 * Inference preset categories for grouped UI display
 */
export const INFERENCE_PRESET_CATEGORIES: { name: string; presets: string[] }[] = [
  {
    name: '1 GPU',
    presets: ['inf-1x-t4', 'inf-1x-l4', 'inf-1x-a10g', 'inf-1x-l40s', 'inf-1x-a100', 'inf-1x-h100'],
  },
  {
    name: '2–4 GPUs',
    presets: [
      'inf-2x-l4', 'inf-2x-a10g', 'inf-2x-l40s', 'inf-2x-a100', 'inf-2x-h100',
      'inf-4x-l4', 'inf-4x-a10g', 'inf-4x-l40s', 'inf-4x-a100', 'inf-4x-h100',
    ],
  },
  {
    name: '8–32 GPUs',
    presets: [
      'inf-8x-l40s', 'inf-8x-a100', 'inf-8x-h100', 'inf-8x-mi300x',
      'inf-16x-l40s', 'inf-16x-a100', 'inf-16x-h100', 'inf-16x-mi300x',
      'inf-32x-l40s', 'inf-32x-a100', 'inf-32x-h100', 'inf-32x-mi300x',
    ],
  },
  {
    name: '64–128 GPUs',
    presets: [
      'inf-64x-a100', 'inf-64x-h100', 'inf-64x-mi300x',
      'inf-128x-a100', 'inf-128x-h100', 'inf-128x-mi300x',
    ],
  },
];

/**
 * Get preset cluster
 */
export function getPresetCluster(presetId: string): ClusterConfig | undefined {
  const preset = CLUSTER_PRESETS[presetId];
  return preset ? preset() : undefined;
}
