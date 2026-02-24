/**
 * Hardware module exports
 */

// GPU specifications
export {
  A100_40GB,
  A100_80GB,
  A800_80GB,
  H100_SXM,
  H100_PCIE,
  H100_NVL,
  H800_SXM,
  H200_SXM,
  B200,
  GB200,
  MI300X,
  MI325X,
  V100_32GB,
  L40S,
  L40,
  A10,
  A10G,
  L4,
  RTX_6000_ADA,
  RTX_4090,
  RTX_3090,
  ALL_GPUS,
  GPU_CATEGORIES,
  getGPU,
  getEffectiveTFLOPS,
  getTrainingTFLOPS,
  getMemoryBandwidthScaling,
  supportsFlashAttention,
  gpuSupportsPrecision,
  getPrecisionFallbackWarning,
} from './gpu.ts';

// Interconnect specifications
export {
  PCIE_SPECS,
  NVSWITCH_SPECS,
  ROCE_SPECS,
  ETHERNET_SPECS,
  INFINITY_FABRIC_SPECS,
  ALL_INTERCONNECTS,
  getInterconnect,
  getIntraNodeInterconnect,
  getNvSwitchSpec,
  getEffectiveBandwidth,
} from './interconnect.ts';

// Topology utilities
export {
  createNode,
  createCluster,
  createSingleNodeCluster,
  createMultiNodeCluster,
  CLUSTER_SIZES,
} from './topology.ts';

// Presets
export {
  DGX_A100,
  DGX_H100,
  DGX_H200,
  DGX_B200,
  DGX_SUPERPOD_A100,
  DGX_SUPERPOD_H100,
  ALL_DGX_NODES,
  CLUSTER_PRESETS,
  PRESET_LABELS,
  getPresetCluster,
} from './presets.ts';
