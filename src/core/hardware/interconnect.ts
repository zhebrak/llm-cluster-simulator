/**
 * Interconnect specifications for GPU clusters
 */

import type { InterconnectSpec, GPUSpec, InfiniBandSpec } from '../../types/index.ts';
import { NVLINK_SPECS, INFINIBAND_SPECS } from '../../types/index.ts';

/**
 * PCIe specifications
 */
export const PCIE_SPECS: Record<string, InterconnectSpec> = {
  'pcie-gen3': {
    type: 'pcie',
    name: 'PCIe Gen3 x16',
    bandwidthGBps: 15.75,
    bidirectionalGBps: 31.5,
    latencyUs: 1.5,
    isFullDuplex: true,
  },
  'pcie-gen4': {
    type: 'pcie',
    name: 'PCIe Gen4 x16',
    bandwidthGBps: 31.5,
    bidirectionalGBps: 63,
    latencyUs: 1.2,
    isFullDuplex: true,
  },
  'pcie-gen5': {
    type: 'pcie',
    name: 'PCIe Gen5 x16',
    bandwidthGBps: 63,
    bidirectionalGBps: 126,
    latencyUs: 1.0,
    isFullDuplex: true,
  },
  'pcie-gen6': {
    type: 'pcie',
    name: 'PCIe Gen6 x16',
    bandwidthGBps: 126,
    bidirectionalGBps: 252,
    latencyUs: 0.8,
    isFullDuplex: true,
  },
};

/**
 * NVSwitch specifications
 */
export const NVSWITCH_SPECS: Record<string, InterconnectSpec> = {
  'nvswitch-a100': {
    type: 'nvswitch',
    name: 'NVSwitch (A100)',
    bandwidthGBps: 300, // Per-GPU to switch
    bidirectionalGBps: 600,
    latencyUs: 0.5,
    isFullDuplex: true,
  },
  'nvswitch-h100': {
    type: 'nvswitch',
    name: 'NVSwitch (H100)',
    bandwidthGBps: 450,
    bidirectionalGBps: 900,
    latencyUs: 0.4,
    isFullDuplex: true,
  },
  'nvswitch-b200': {
    type: 'nvswitch',
    name: 'NVSwitch (B200)',
    bandwidthGBps: 900,
    bidirectionalGBps: 1800,
    latencyUs: 0.3,
    isFullDuplex: true,
  },
};

/**
 * RoCE (RDMA over Converged Ethernet) specifications
 */
export const ROCE_SPECS: Record<string, InterconnectSpec> = {
  'roce-100g': {
    type: 'roce',
    name: 'RoCE v2 100GbE',
    bandwidthGBps: 12.5,
    bidirectionalGBps: 25,
    latencyUs: 2.0,
    isFullDuplex: true,
  },
  'roce-200g': {
    type: 'roce',
    name: 'RoCE v2 200GbE',
    bandwidthGBps: 25,
    bidirectionalGBps: 50,
    latencyUs: 1.5,
    isFullDuplex: true,
  },
  'roce-400g': {
    type: 'roce',
    name: 'RoCE v2 400GbE',
    bandwidthGBps: 50,
    bidirectionalGBps: 100,
    latencyUs: 1.2,
    isFullDuplex: true,
  },
};

/**
 * Standard Ethernet specifications
 */
export const ETHERNET_SPECS: Record<string, InterconnectSpec> = {
  'eth-10g': {
    type: 'ethernet',
    name: 'Ethernet 10GbE',
    bandwidthGBps: 1.25,
    bidirectionalGBps: 2.5,
    latencyUs: 50,
    isFullDuplex: true,
  },
  'eth-25g': {
    type: 'ethernet',
    name: 'Ethernet 25GbE',
    bandwidthGBps: 3.125,
    bidirectionalGBps: 6.25,
    latencyUs: 30,
    isFullDuplex: true,
  },
  'eth-100g': {
    type: 'ethernet',
    name: 'Ethernet 100GbE',
    bandwidthGBps: 12.5,
    bidirectionalGBps: 25,
    latencyUs: 10,
    isFullDuplex: true,
  },
};

/**
 * AMD Infinity Fabric specifications
 */
export const INFINITY_FABRIC_SPECS: Record<string, InterconnectSpec> = {
  'if-mi250': {
    type: 'nvlink', // Using nvlink type as closest equivalent for AMD IF
    name: 'AMD Infinity Fabric (MI250)',
    bandwidthGBps: 200,  // Intra-package GCD-to-GCD
    bidirectionalGBps: 400,
    latencyUs: 0.6,
    isFullDuplex: true,
  },
  'if-mi300': {
    type: 'nvlink', // Using nvlink type as closest equivalent
    name: 'AMD Infinity Fabric (MI300)',
    bandwidthGBps: 448, // 896 GB/s bidirectional / 2
    bidirectionalGBps: 896,
    latencyUs: 0.5,
    isFullDuplex: true,
  },
  'if-mi350': {
    type: 'nvlink', // Using nvlink type as closest equivalent for AMD IF
    name: 'AMD Infinity Fabric (MI350)',
    bandwidthGBps: 538,  // 7× IF4 links aggregate
    bidirectionalGBps: 1075,
    latencyUs: 0.4,
    isFullDuplex: true,
  },
};

/**
 * All interconnect specifications
 */
export const ALL_INTERCONNECTS: Record<string, InterconnectSpec> = {
  ...NVLINK_SPECS,
  ...Object.fromEntries(
    Object.entries(INFINIBAND_SPECS).map(([k, v]) => [`ib-${k}`, v])
  ),
  ...PCIE_SPECS,
  ...NVSWITCH_SPECS,
  ...ROCE_SPECS,
  ...ETHERNET_SPECS,
  ...INFINITY_FABRIC_SPECS,
};

/**
 * Get interconnect spec by ID
 */
export function getInterconnect(id: string): InterconnectSpec | undefined {
  return ALL_INTERCONNECTS[id];
}

/**
 * Get appropriate intra-node interconnect for a GPU
 */
export function getIntraNodeInterconnect(gpu: GPUSpec): InterconnectSpec {
  const arch = gpu.architecture?.toLowerCase();
  // AMD GPUs use Infinity Fabric instead of NVLink
  if (gpu.vendor === 'amd') {
    switch (arch) {
      case 'cdna2':
        return INFINITY_FABRIC_SPECS['if-mi250'];
      case 'cdna3':
        return INFINITY_FABRIC_SPECS['if-mi300'];
      case 'cdna4':
        // MI350X uses newer IF4; MI325X has CDNA3 compute die → MI300's IF
        if (gpu.id === 'mi350x') return INFINITY_FABRIC_SPECS['if-mi350'];
        return INFINITY_FABRIC_SPECS['if-mi300'];
      default:
        return PCIE_SPECS['pcie-gen4'];
    }
  }

  if (gpu.nvlinkVersion) {
    const spec = NVLINK_SPECS[gpu.nvlinkVersion];
    // China-export variants (H800, A800) share the NVLink version but with fewer
    // physical links, reducing effective per-GPU bandwidth through the switch fabric.
    if (gpu.nvlinkBandwidthGBps > 0 && gpu.nvlinkBandwidthGBps < spec.bandwidthGBps) {
      return {
        ...spec,
        name: `${spec.name} (${gpu.nvlinkBandwidthGBps} GB/s)`,
        bandwidthGBps: gpu.nvlinkBandwidthGBps,
        bidirectionalGBps: gpu.nvlinkBandwidthGBps * 2,
      };
    }
    return spec;
  }

  // Fall back to PCIe based on architecture
  switch (arch) {
    case 'blackwell':
      return PCIE_SPECS['pcie-gen6'];
    case 'hopper':
      return PCIE_SPECS['pcie-gen5'];
    case 'ampere':
    case 'ada':
      return PCIE_SPECS['pcie-gen4'];
    case 'volta':
    case 'turing':
      return PCIE_SPECS['pcie-gen3'];
    default:
      return PCIE_SPECS['pcie-gen4'];
  }
}

/**
 * Get NVSwitch spec for a GPU architecture
 */
export function getNvSwitchSpec(gpu: GPUSpec): InterconnectSpec | undefined {
  if (!gpu.hasNvSwitch) return undefined;
  const arch = gpu.architecture?.toLowerCase();

  let spec: InterconnectSpec | undefined;
  switch (arch) {
    case 'blackwell':
      spec = NVSWITCH_SPECS['nvswitch-b200']; break;
    case 'hopper':
      spec = NVSWITCH_SPECS['nvswitch-h100']; break;
    case 'ampere':
    case 'volta':
      spec = NVSWITCH_SPECS['nvswitch-a100']; break;
    case 'ada':
      return undefined;
    default:
      return undefined;
  }
  // China-export variants have reduced per-GPU bandwidth through NVSwitch
  if (spec && gpu.nvlinkBandwidthGBps > 0 && gpu.nvlinkBandwidthGBps < spec.bandwidthGBps) {
    return {
      ...spec,
      name: `${spec.name} (${gpu.nvlinkBandwidthGBps} GB/s)`,
      bandwidthGBps: gpu.nvlinkBandwidthGBps,
      bidirectionalGBps: gpu.nvlinkBandwidthGBps * 2,
    };
  }
  return spec;
}

/**
 * Calculate effective bandwidth considering topology
 * For NVSwitch: full bisection bandwidth
 * For ring NVLink: limited by ring structure
 */
export function getEffectiveBandwidth(
  interconnect: InterconnectSpec,
  _numGPUs: number,
  hasNvSwitch: boolean
): number {
  if (hasNvSwitch || interconnect.type === 'nvswitch') {
    // Full bisection bandwidth with NVSwitch
    return interconnect.bandwidthGBps;
  }

  if (interconnect.type === 'nvlink') {
    // Ring topology: effective bandwidth is half for AllReduce
    return interconnect.bandwidthGBps;
  }

  // For other interconnects, no scaling
  return interconnect.bandwidthGBps;
}

/**
 * Per-NIC bandwidth for P2P transfers.
 * InfiniBand: perPortBandwidthGBps from spec.
 * Others: aggregate / numNICs.
 */
export function getPerNicBandwidthGBps(
  interconnect: InterconnectSpec,
  numNICs: number,
): number {
  if (interconnect.type === 'infiniband') {
    return (interconnect as InfiniBandSpec).perPortBandwidthGBps;
  }
  return interconnect.bandwidthGBps / numNICs;
}

