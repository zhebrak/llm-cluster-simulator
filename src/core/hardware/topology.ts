/**
 * Cluster topology configurations
 */

import type {
  GPUSpec,
  NodeSpec,
  ClusterConfig,
  TopologyType,
  InterconnectSpec,
} from '../../types/index.ts';
import { getGPU, getEffectiveTFLOPS } from './gpu.ts';
import { getIntraNodeInterconnect, getNvSwitchSpec } from './interconnect.ts';
import { INFINIBAND_SPECS } from '../../types/index.ts';

/**
 * Create a node specification
 */
export function createNode(
  gpu: GPUSpec,
  numGPUs: number,
  cpuCores: number = 128,
  ramGB: number = 1024,
  interNodeInterconnect?: InterconnectSpec,
  id?: string,
  numNICs?: number,
): NodeSpec {
  const intraNode = getIntraNodeInterconnect(gpu);
  const hasNvSwitch = gpu.hasNvSwitch && numGPUs > 2;

  return {
    id: id ?? `node-${gpu.id}-${numGPUs}gpu`,
    name: `${numGPUs}x ${gpu.name}`,
    numGPUs,
    gpu,
    intraNodeInterconnect: hasNvSwitch ? getNvSwitchSpec(gpu) ?? intraNode : intraNode,
    interNodeInterconnect: interNodeInterconnect ?? INFINIBAND_SPECS.ndr,
    cpuCores,
    ramGB,
    hasNvSwitch,
    numNICs: numNICs ?? numGPUs,
  };
}

/**
 * Create a cluster configuration
 */
export function createCluster(
  node: NodeSpec,
  numNodes: number,
  topology: TopologyType = 'fat-tree',
  id?: string
): ClusterConfig {
  const totalGPUs = numNodes * node.numGPUs;
  const totalMemoryGB = totalGPUs * node.gpu.memoryGB;
  const totalTFLOPS = totalGPUs * getEffectiveTFLOPS(node.gpu, 'bf16');

  return {
    id: id ?? `cluster-${numNodes}x${node.numGPUs}-${node.gpu.id}`,
    name: `${totalGPUs}x ${node.gpu.name}`,
    numNodes,
    gpusPerNode: node.numGPUs,
    totalGPUs,
    node,
    topology,
    interNodeBandwidthGBps: node.interNodeInterconnect.bandwidthGBps,
    interNodeLatencyUs: node.interNodeInterconnect.latencyUs,
    totalMemoryGB,
    totalTFLOPS,
  };
}

/**
 * Create a single-node cluster
 */
export function createSingleNodeCluster(
  gpuId: string,
  numGPUs: number
): ClusterConfig | undefined {
  const gpu = getGPU(gpuId);
  if (!gpu) return undefined;

  const node = createNode(gpu, numGPUs);
  return createCluster(node, 1, 'single-node');
}

/**
 * Auto-detect IB version from GPU architecture.
 * Derived from DGX node specs in presets.ts:
 * - HDR (200 Gb/s): A100/V100/T4 era, MI250X
 * - NDR (400 Gb/s): H100/H200 era, MI300X/MI325X, L40S
 * - XDR (800 Gb/s): B200/GB200 era, MI350X
 */
function getDefaultIBVersion(architecture: string): 'hdr' | 'ndr' | 'xdr' {
  switch (architecture) {
    case 'ampere': case 'volta': case 'turing': case 'cdna2':
      return 'hdr';
    case 'hopper': case 'cdna3': case 'ada':
      return 'ndr';
    case 'blackwell': case 'cdna4':
      return 'xdr';
    default:
      return 'ndr';
  }
}

/**
 * Create a multi-node cluster with default settings.
 * IB version auto-detected from GPU architecture when not specified.
 */
export function createMultiNodeCluster(
  gpuId: string,
  gpusPerNode: number,
  numNodes: number,
  ibVersion?: 'hdr' | 'ndr' | 'xdr'
): ClusterConfig | undefined {
  const gpu = getGPU(gpuId);
  if (!gpu) return undefined;

  const resolvedIB = ibVersion ?? getDefaultIBVersion(gpu.architecture);
  const node = createNode(
    gpu,
    gpusPerNode,
    128,
    1024,
    INFINIBAND_SPECS[resolvedIB]
  );

  return createCluster(node, numNodes, 'fat-tree');
}

/**
 * Common cluster sizes for presets
 */
export const CLUSTER_SIZES = {
  small: [1, 2, 4, 8],
  medium: [16, 32, 64],
  large: [128, 256, 512],
  xlarge: [1024, 2048, 4096],
};

