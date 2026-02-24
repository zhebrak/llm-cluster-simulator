/**
 * GPU Specification Validation Tests
 *
 * Validates all 11 GPU specs against official datasheets, checks internal
 * consistency, verifies interconnect mapping, and runs simulation smoke tests.
 *
 * All TFLOPS values are DENSE (non-sparse).
 */

import { describe, it, expect } from 'vitest';
import {
  ALL_GPUS,
  T4,
  V100_32GB,
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
  MI250X,
  MI210,
  MI350X,
  MI300X,
  MI325X,
  L40S,
  L40,
  A10,
  A10G,
  L4,
  RTX_6000_ADA,
  RTX_4090,
  RTX_3090,
} from '../../src/core/hardware/gpu.ts';
import {
  getIntraNodeInterconnect,
  getNvSwitchSpec,
  PCIE_SPECS,
  INFINITY_FABRIC_SPECS,
} from '../../src/core/hardware/interconnect.ts';
import { NVLINK_SPECS } from '../../src/types/index.ts';
import { createMultiNodeCluster } from '../../src/core/hardware/topology.ts';
import type { SimulationConfig } from '../../src/core/simulation/engine.ts';
import { getValidatedSimulationMetrics } from '../helpers/validated-metrics.ts';

// ---------------------------------------------------------------------------
// Section 1: Spec accuracy — pin each GPU's key values to official numbers
// ---------------------------------------------------------------------------

describe('GPU spec accuracy: V100 32GB', () => {
  const gpu = V100_32GB;
  it('architecture is volta', () => expect(gpu.architecture).toBe('volta'));
  it('memoryGB = 32', () => expect(gpu.memoryGB).toBe(32));
  it('memoryBandwidthTBps = 0.9', () => expect(gpu.memoryBandwidthTBps).toBe(0.9));
  it('fp32TFLOPS = 15.7', () => expect(gpu.fp32TFLOPS).toBe(15.7));
  it('tf32TFLOPS = 0 (no TF32 on Volta)', () => expect(gpu.tf32TFLOPS).toBe(0));
  it('fp16TFLOPS = 125', () => expect(gpu.fp16TFLOPS).toBe(125));
  it('bf16TFLOPS = 0 (no BF16 on Volta)', () => expect(gpu.bf16TFLOPS).toBe(0));
  it('fp8TFLOPS = 0', () => expect(gpu.fp8TFLOPS).toBe(0));
  it('tdpWatts = 300', () => expect(gpu.tdpWatts).toBe(300));
  it('nvlinkBandwidthGBps = 150', () => expect(gpu.nvlinkBandwidthGBps).toBe(150));
  it('pcieBandwidthGBps = 15.75 (Gen3 x16)', () => expect(gpu.pcieBandwidthGBps).toBe(15.75));
});

describe('GPU spec accuracy: A100 40GB', () => {
  const gpu = A100_40GB;
  it('architecture is ampere', () => expect(gpu.architecture).toBe('ampere'));
  it('memoryGB = 40', () => expect(gpu.memoryGB).toBe(40));
  it('memoryBandwidthTBps = 1.555', () => expect(gpu.memoryBandwidthTBps).toBe(1.555));
  it('fp32TFLOPS = 19.5', () => expect(gpu.fp32TFLOPS).toBe(19.5));
  it('tf32TFLOPS = 156', () => expect(gpu.tf32TFLOPS).toBe(156));
  it('bf16TFLOPS = 312', () => expect(gpu.bf16TFLOPS).toBe(312));
  it('fp8TFLOPS = 0 (no FP8 on Ampere)', () => expect(gpu.fp8TFLOPS).toBe(0));
  it('int8TOPS = 624', () => expect(gpu.int8TOPS).toBe(624));
  it('tdpWatts = 400', () => expect(gpu.tdpWatts).toBe(400));
  it('nvlinkBandwidthGBps = 300', () => expect(gpu.nvlinkBandwidthGBps).toBe(300));
  it('pcieBandwidthGBps = 31.5 (Gen4 x16)', () => expect(gpu.pcieBandwidthGBps).toBe(31.5));
});

describe('GPU spec accuracy: A100 80GB', () => {
  const gpu = A100_80GB;
  it('memoryGB = 80', () => expect(gpu.memoryGB).toBe(80));
  it('memoryBandwidthTBps = 2.039', () => expect(gpu.memoryBandwidthTBps).toBe(2.039));
  it('same compute as A100 40GB (bf16 = 312)', () => expect(gpu.bf16TFLOPS).toBe(312));
  it('tdpWatts = 400', () => expect(gpu.tdpWatts).toBe(400));
});

describe('GPU spec accuracy: A800 80GB (China-export A100)', () => {
  const gpu = A800_80GB;
  it('architecture is ampere', () => expect(gpu.architecture).toBe('ampere'));
  it('memoryGB = 80', () => expect(gpu.memoryGB).toBe(80));
  it('memoryBandwidthTBps = 2.039 (same as A100 80GB)', () => expect(gpu.memoryBandwidthTBps).toBe(2.039));
  it('fp32TFLOPS = 19.5 (identical to A100)', () => expect(gpu.fp32TFLOPS).toBe(19.5));
  it('tf32TFLOPS = 156', () => expect(gpu.tf32TFLOPS).toBe(156));
  it('bf16TFLOPS = 312', () => expect(gpu.bf16TFLOPS).toBe(312));
  it('fp8TFLOPS = 0 (no FP8 on Ampere)', () => expect(gpu.fp8TFLOPS).toBe(0));
  it('int8TOPS = 624', () => expect(gpu.int8TOPS).toBe(624));
  it('tdpWatts = 400', () => expect(gpu.tdpWatts).toBe(400));
  it('nvlinkBandwidthGBps = 200 (reduced from A100 300)', () => expect(gpu.nvlinkBandwidthGBps).toBe(200));
  it('nvlinkVersion = 3 (same as A100)', () => expect(gpu.nvlinkVersion).toBe(3));
  it('pcieBandwidthGBps = 31.5 (Gen4 x16)', () => expect(gpu.pcieBandwidthGBps).toBe(31.5));
  it('hasNvSwitch = true', () => expect(gpu.hasNvSwitch).toBe(true));
  it('hasTransformerEngine = false', () => expect(gpu.hasTransformerEngine).toBe(false));
});

describe('GPU spec accuracy: H100 SXM', () => {
  const gpu = H100_SXM;
  it('architecture is hopper', () => expect(gpu.architecture).toBe('hopper'));
  it('memoryGB = 80', () => expect(gpu.memoryGB).toBe(80));
  it('memoryBandwidthTBps = 3.35', () => expect(gpu.memoryBandwidthTBps).toBe(3.35));
  it('fp32TFLOPS = 67', () => expect(gpu.fp32TFLOPS).toBe(67));
  it('tf32TFLOPS = 989', () => expect(gpu.tf32TFLOPS).toBe(989));
  it('bf16TFLOPS = 989', () => expect(gpu.bf16TFLOPS).toBe(989));
  it('fp8TFLOPS = 1979', () => expect(gpu.fp8TFLOPS).toBe(1979));
  it('int8TOPS = 1979', () => expect(gpu.int8TOPS).toBe(1979));
  it('tdpWatts = 700', () => expect(gpu.tdpWatts).toBe(700));
  it('nvlinkBandwidthGBps = 450', () => expect(gpu.nvlinkBandwidthGBps).toBe(450));
  it('pcieBandwidthGBps = 63 (Gen5 x16)', () => expect(gpu.pcieBandwidthGBps).toBe(63));
});

describe('GPU spec accuracy: H100 PCIe', () => {
  const gpu = H100_PCIE;
  it('memoryGB = 80', () => expect(gpu.memoryGB).toBe(80));
  it('memoryBandwidthTBps = 2.0', () => expect(gpu.memoryBandwidthTBps).toBe(2.0));
  it('fp32TFLOPS = 51', () => expect(gpu.fp32TFLOPS).toBe(51));
  it('bf16TFLOPS = 756', () => expect(gpu.bf16TFLOPS).toBe(756));
  it('fp8TFLOPS = 1513', () => expect(gpu.fp8TFLOPS).toBe(1513));
  it('tdpWatts = 350', () => expect(gpu.tdpWatts).toBe(350));
  it('nvlinkBandwidthGBps = 0 (standalone PCIe)', () => expect(gpu.nvlinkBandwidthGBps).toBe(0));
  it('hasNvSwitch = false', () => expect(gpu.hasNvSwitch).toBe(false));
});

describe('GPU spec accuracy: H100 NVL', () => {
  const gpu = H100_NVL;
  it('memoryGB = 94 (47 GB × 2)', () => expect(gpu.memoryGB).toBe(94));
  it('memoryBandwidthTBps = 3.9', () => expect(gpu.memoryBandwidthTBps).toBe(3.9));
  it('bf16TFLOPS = 835 (lower clocks at 400W TDP)', () => expect(gpu.bf16TFLOPS).toBe(835));
  it('tdpWatts = 400', () => expect(gpu.tdpWatts).toBe(400));
  it('nvlinkBandwidthGBps = 300 (NVL bridge)', () => expect(gpu.nvlinkBandwidthGBps).toBe(300));
  it('hasNvSwitch = false (2-GPU bridge)', () => expect(gpu.hasNvSwitch).toBe(false));
});

describe('GPU spec accuracy: H800 SXM (China-export H100)', () => {
  const gpu = H800_SXM;
  it('architecture is hopper', () => expect(gpu.architecture).toBe('hopper'));
  it('memoryGB = 80', () => expect(gpu.memoryGB).toBe(80));
  it('memoryBandwidthTBps = 3.35 (same as H100 SXM)', () => expect(gpu.memoryBandwidthTBps).toBe(3.35));
  it('fp32TFLOPS = 67 (identical to H100)', () => expect(gpu.fp32TFLOPS).toBe(67));
  it('tf32TFLOPS = 989', () => expect(gpu.tf32TFLOPS).toBe(989));
  it('bf16TFLOPS = 989', () => expect(gpu.bf16TFLOPS).toBe(989));
  it('fp8TFLOPS = 1979', () => expect(gpu.fp8TFLOPS).toBe(1979));
  it('int8TOPS = 1979', () => expect(gpu.int8TOPS).toBe(1979));
  it('tdpWatts = 700', () => expect(gpu.tdpWatts).toBe(700));
  it('nvlinkBandwidthGBps = 200 (reduced from H100 450)', () => expect(gpu.nvlinkBandwidthGBps).toBe(200));
  it('nvlinkVersion = 4 (same as H100)', () => expect(gpu.nvlinkVersion).toBe(4));
  it('pcieBandwidthGBps = 63 (Gen5 x16)', () => expect(gpu.pcieBandwidthGBps).toBe(63));
  it('hasNvSwitch = true', () => expect(gpu.hasNvSwitch).toBe(true));
  it('hasTransformerEngine = true', () => expect(gpu.hasTransformerEngine).toBe(true));
});

describe('GPU spec accuracy: H200 SXM', () => {
  const gpu = H200_SXM;
  it('memoryGB = 141', () => expect(gpu.memoryGB).toBe(141));
  it('memoryBandwidthTBps = 4.8', () => expect(gpu.memoryBandwidthTBps).toBe(4.8));
  it('bf16TFLOPS = 989 (same Hopper die)', () => expect(gpu.bf16TFLOPS).toBe(989));
  it('fp8TFLOPS = 1979', () => expect(gpu.fp8TFLOPS).toBe(1979));
  it('tdpWatts = 700', () => expect(gpu.tdpWatts).toBe(700));
  it('nvlinkBandwidthGBps = 450', () => expect(gpu.nvlinkBandwidthGBps).toBe(450));
});

describe('GPU spec accuracy: B200 (HGX, 1000W air-cooled)', () => {
  const gpu = B200;
  it('architecture is blackwell', () => expect(gpu.architecture).toBe('blackwell'));
  it('memoryGB = 180', () => expect(gpu.memoryGB).toBe(180));
  it('memoryBandwidthTBps = 7.7', () => expect(gpu.memoryBandwidthTBps).toBe(7.7));
  it('fp32TFLOPS = 75', () => expect(gpu.fp32TFLOPS).toBe(75));
  it('tf32TFLOPS = 1100 (dense)', () => expect(gpu.tf32TFLOPS).toBe(1100));
  it('bf16TFLOPS = 2250 (dense)', () => expect(gpu.bf16TFLOPS).toBe(2250));
  it('fp8TFLOPS = 4500', () => expect(gpu.fp8TFLOPS).toBe(4500));
  it('fp4TFLOPS = 9000', () => expect(gpu.fp4TFLOPS).toBe(9000));
  it('tdpWatts = 1000', () => expect(gpu.tdpWatts).toBe(1000));
  it('nvlinkBandwidthGBps = 900', () => expect(gpu.nvlinkBandwidthGBps).toBe(900));
  it('pcieBandwidthGBps = 63 (Gen5 x16)', () => expect(gpu.pcieBandwidthGBps).toBe(63));
});

describe('GPU spec accuracy: GB200 (NVL, 1200W liquid-cooled)', () => {
  const gpu = GB200;
  it('architecture is blackwell', () => expect(gpu.architecture).toBe('blackwell'));
  it('memoryGB = 186 (372/superchip ÷ 2)', () => expect(gpu.memoryGB).toBe(186));
  it('memoryBandwidthTBps = 8.0', () => expect(gpu.memoryBandwidthTBps).toBe(8.0));
  it('fp32TFLOPS = 80 (160/superchip ÷ 2)', () => expect(gpu.fp32TFLOPS).toBe(80));
  it('tf32TFLOPS = 1250 (dense)', () => expect(gpu.tf32TFLOPS).toBe(1250));
  it('bf16TFLOPS = 2500 (dense)', () => expect(gpu.bf16TFLOPS).toBe(2500));
  it('fp8TFLOPS = 5000', () => expect(gpu.fp8TFLOPS).toBe(5000));
  it('fp4TFLOPS = 10000', () => expect(gpu.fp4TFLOPS).toBe(10000));
  it('tdpWatts = 1200', () => expect(gpu.tdpWatts).toBe(1200));
  it('nvlinkBandwidthGBps = 900', () => expect(gpu.nvlinkBandwidthGBps).toBe(900));
  it('pcieBandwidthGBps = 126 (Gen6 x16)', () => expect(gpu.pcieBandwidthGBps).toBe(126));
});

describe('B200 vs GB200 differentiation', () => {
  it('GB200 has higher TDP (1200W vs 1000W)', () => {
    expect(GB200.tdpWatts).toBeGreaterThan(B200.tdpWatts);
  });
  it('GB200 has higher bf16 TFLOPS', () => {
    expect(GB200.bf16TFLOPS).toBeGreaterThan(B200.bf16TFLOPS);
  });
  it('GB200 has more memory (186 vs 180)', () => {
    expect(GB200.memoryGB).toBeGreaterThan(B200.memoryGB);
  });
  it('GB200 has higher memory bandwidth', () => {
    expect(GB200.memoryBandwidthTBps).toBeGreaterThan(B200.memoryBandwidthTBps);
  });
});

describe('GPU spec accuracy: MI300X', () => {
  const gpu = MI300X;
  it('architecture is cdna3', () => expect(gpu.architecture).toBe('cdna3'));
  it('memoryGB = 192', () => expect(gpu.memoryGB).toBe(192));
  it('memoryBandwidthTBps = 5.3', () => expect(gpu.memoryBandwidthTBps).toBe(5.3));
  it('fp32TFLOPS = 163', () => expect(gpu.fp32TFLOPS).toBe(163));
  it('bf16TFLOPS = 1307', () => expect(gpu.bf16TFLOPS).toBe(1307));
  it('fp8TFLOPS = 2614', () => expect(gpu.fp8TFLOPS).toBe(2614));
  it('fp4TFLOPS = 0 (no FP4 on CDNA3)', () => expect(gpu.fp4TFLOPS).toBe(0));
  it('tdpWatts = 750', () => expect(gpu.tdpWatts).toBe(750));
  it('nvlinkBandwidthGBps = 0 (Infinity Fabric)', () => expect(gpu.nvlinkBandwidthGBps).toBe(0));
  it('pcieBandwidthGBps = 63 (Gen5 x16)', () => expect(gpu.pcieBandwidthGBps).toBe(63));
});

describe('GPU spec accuracy: MI325X', () => {
  const gpu = MI325X;
  it('memoryGB = 256', () => expect(gpu.memoryGB).toBe(256));
  it('memoryBandwidthTBps = 6.0', () => expect(gpu.memoryBandwidthTBps).toBe(6.0));
  it('bf16TFLOPS = 1307 (same CDNA3 die)', () => expect(gpu.bf16TFLOPS).toBe(1307));
  it('fp8TFLOPS = 2614', () => expect(gpu.fp8TFLOPS).toBe(2614));
  it('fp4TFLOPS = 0 (CDNA3 die has no FP4)', () => expect(gpu.fp4TFLOPS).toBe(0));
  it('tdpWatts = 750', () => expect(gpu.tdpWatts).toBe(750));
  it('nvlinkBandwidthGBps = 0', () => expect(gpu.nvlinkBandwidthGBps).toBe(0));
});

describe('GPU spec accuracy: MI250X (per GCD)', () => {
  const gpu = MI250X;
  it('architecture is cdna2', () => expect(gpu.architecture).toBe('cdna2'));
  it('memoryGB = 64 (128/2 GCDs)', () => expect(gpu.memoryGB).toBe(64));
  it('memoryBandwidthTBps = 1.638', () => expect(gpu.memoryBandwidthTBps).toBe(1.638));
  it('fp32TFLOPS = 47.9', () => expect(gpu.fp32TFLOPS).toBe(47.9));
  it('tf32TFLOPS = 0 (no TF32 on CDNA2)', () => expect(gpu.tf32TFLOPS).toBe(0));
  it('bf16TFLOPS = 191.5', () => expect(gpu.bf16TFLOPS).toBe(191.5));
  it('fp16TFLOPS = 191.5', () => expect(gpu.fp16TFLOPS).toBe(191.5));
  it('fp8TFLOPS = 0 (no FP8 on CDNA2)', () => expect(gpu.fp8TFLOPS).toBe(0));
  it('fp4TFLOPS = 0 (no FP4 on CDNA2)', () => expect(gpu.fp4TFLOPS).toBe(0));
  it('int8TOPS = 191.5', () => expect(gpu.int8TOPS).toBe(191.5));
  it('int4TOPS = 191.5', () => expect(gpu.int4TOPS).toBe(191.5));
  it('tdpWatts = 250 (~500W/2)', () => expect(gpu.tdpWatts).toBe(250));
  it('nvlinkBandwidthGBps = 0 (Infinity Fabric)', () => expect(gpu.nvlinkBandwidthGBps).toBe(0));
  it('pcieBandwidthGBps = 31.5 (Gen4 x16)', () => expect(gpu.pcieBandwidthGBps).toBe(31.5));
  it('hasTransformerEngine = false', () => expect(gpu.hasTransformerEngine).toBe(false));
});

describe('GPU spec accuracy: MI210', () => {
  const gpu = MI210;
  it('architecture is cdna2', () => expect(gpu.architecture).toBe('cdna2'));
  it('memoryGB = 64', () => expect(gpu.memoryGB).toBe(64));
  it('memoryBandwidthTBps = 1.6', () => expect(gpu.memoryBandwidthTBps).toBe(1.6));
  it('fp32TFLOPS = 45.3 (104 CUs)', () => expect(gpu.fp32TFLOPS).toBe(45.3));
  it('tf32TFLOPS = 0 (no TF32 on CDNA2)', () => expect(gpu.tf32TFLOPS).toBe(0));
  it('bf16TFLOPS = 181', () => expect(gpu.bf16TFLOPS).toBe(181));
  it('fp16TFLOPS = 181', () => expect(gpu.fp16TFLOPS).toBe(181));
  it('fp8TFLOPS = 0', () => expect(gpu.fp8TFLOPS).toBe(0));
  it('fp4TFLOPS = 0', () => expect(gpu.fp4TFLOPS).toBe(0));
  it('int8TOPS = 181', () => expect(gpu.int8TOPS).toBe(181));
  it('tdpWatts = 300', () => expect(gpu.tdpWatts).toBe(300));
  it('nvlinkBandwidthGBps = 0', () => expect(gpu.nvlinkBandwidthGBps).toBe(0));
  it('pcieBandwidthGBps = 31.5 (Gen4 x16)', () => expect(gpu.pcieBandwidthGBps).toBe(31.5));
  it('hasTransformerEngine = false', () => expect(gpu.hasTransformerEngine).toBe(false));
});

describe('GPU spec accuracy: MI350X', () => {
  const gpu = MI350X;
  it('architecture is cdna4', () => expect(gpu.architecture).toBe('cdna4'));
  it('memoryGB = 288', () => expect(gpu.memoryGB).toBe(288));
  it('memoryBandwidthTBps = 8.0', () => expect(gpu.memoryBandwidthTBps).toBe(8.0));
  it('fp32TFLOPS = 144', () => expect(gpu.fp32TFLOPS).toBe(144));
  it('tf32TFLOPS = 1153', () => expect(gpu.tf32TFLOPS).toBe(1153));
  it('bf16TFLOPS = 2307', () => expect(gpu.bf16TFLOPS).toBe(2307));
  it('fp16TFLOPS = 2307', () => expect(gpu.fp16TFLOPS).toBe(2307));
  it('fp8TFLOPS = 4614', () => expect(gpu.fp8TFLOPS).toBe(4614));
  it('fp4TFLOPS = 9228 (new in CDNA4)', () => expect(gpu.fp4TFLOPS).toBe(9228));
  it('int8TOPS = 4614', () => expect(gpu.int8TOPS).toBe(4614));
  it('int4TOPS = 9228', () => expect(gpu.int4TOPS).toBe(9228));
  it('tdpWatts = 1000', () => expect(gpu.tdpWatts).toBe(1000));
  it('nvlinkBandwidthGBps = 0 (Infinity Fabric)', () => expect(gpu.nvlinkBandwidthGBps).toBe(0));
  it('pcieBandwidthGBps = 63 (Gen5)', () => expect(gpu.pcieBandwidthGBps).toBe(63));
  it('hasTransformerEngine = true (FP8 support)', () => expect(gpu.hasTransformerEngine).toBe(true));
});

describe('GPU spec accuracy: L40S', () => {
  const gpu = L40S;
  it('architecture is ada', () => expect(gpu.architecture).toBe('ada'));
  it('memoryGB = 48', () => expect(gpu.memoryGB).toBe(48));
  it('memoryBandwidthTBps = 0.864', () => expect(gpu.memoryBandwidthTBps).toBe(0.864));
  it('fp32TFLOPS = 91.6', () => expect(gpu.fp32TFLOPS).toBe(91.6));
  it('tf32TFLOPS = 183.2', () => expect(gpu.tf32TFLOPS).toBe(183.2));
  it('bf16TFLOPS = 362', () => expect(gpu.bf16TFLOPS).toBe(362));
  it('fp8TFLOPS = 733', () => expect(gpu.fp8TFLOPS).toBe(733));
  it('fp4TFLOPS = 0 (no FP4 on Ada)', () => expect(gpu.fp4TFLOPS).toBe(0));
  it('int8TOPS = 733', () => expect(gpu.int8TOPS).toBe(733));
  it('tdpWatts = 350', () => expect(gpu.tdpWatts).toBe(350));
  it('no NVLink', () => {
    expect(gpu.nvlinkBandwidthGBps).toBe(0);
    expect(gpu.nvlinkVersion).toBeNull();
  });
  it('pcieBandwidthGBps = 31.5 (Gen4 x16)', () => expect(gpu.pcieBandwidthGBps).toBe(31.5));
  it('hasTransformerEngine = true', () => expect(gpu.hasTransformerEngine).toBe(true));
});

describe('GPU spec accuracy: L40', () => {
  const gpu = L40;
  it('architecture is ada', () => expect(gpu.architecture).toBe('ada'));
  it('memoryGB = 48', () => expect(gpu.memoryGB).toBe(48));
  it('memoryBandwidthTBps = 0.864', () => expect(gpu.memoryBandwidthTBps).toBe(0.864));
  it('fp32TFLOPS = 90.52', () => expect(gpu.fp32TFLOPS).toBe(90.52));
  it('bf16TFLOPS = 181 (half of L40S)', () => expect(gpu.bf16TFLOPS).toBe(181));
  it('fp8TFLOPS = 362', () => expect(gpu.fp8TFLOPS).toBe(362));
  it('tdpWatts = 300', () => expect(gpu.tdpWatts).toBe(300));
  it('no NVLink', () => {
    expect(gpu.nvlinkBandwidthGBps).toBe(0);
    expect(gpu.nvlinkVersion).toBeNull();
  });
});

describe('GPU spec accuracy: A10', () => {
  const gpu = A10;
  it('architecture is ampere', () => expect(gpu.architecture).toBe('ampere'));
  it('memoryGB = 24', () => expect(gpu.memoryGB).toBe(24));
  it('memoryBandwidthTBps = 0.6', () => expect(gpu.memoryBandwidthTBps).toBe(0.6));
  it('fp32TFLOPS = 31.2', () => expect(gpu.fp32TFLOPS).toBe(31.2));
  it('bf16TFLOPS = 125', () => expect(gpu.bf16TFLOPS).toBe(125));
  it('fp8TFLOPS = 0 (no FP8 on Ampere)', () => expect(gpu.fp8TFLOPS).toBe(0));
  it('int8TOPS = 250', () => expect(gpu.int8TOPS).toBe(250));
  it('tdpWatts = 150', () => expect(gpu.tdpWatts).toBe(150));
  it('no NVLink', () => {
    expect(gpu.nvlinkBandwidthGBps).toBe(0);
    expect(gpu.nvlinkVersion).toBeNull();
  });
  it('hasTransformerEngine = false', () => expect(gpu.hasTransformerEngine).toBe(false));
});

describe('GPU spec accuracy: A10G', () => {
  const gpu = A10G;
  it('architecture is ampere', () => expect(gpu.architecture).toBe('ampere'));
  it('memoryGB = 24', () => expect(gpu.memoryGB).toBe(24));
  it('bf16TFLOPS = 70 (GA10B die)', () => expect(gpu.bf16TFLOPS).toBe(70));
  it('fp8TFLOPS = 0', () => expect(gpu.fp8TFLOPS).toBe(0));
  it('int8TOPS = 140', () => expect(gpu.int8TOPS).toBe(140));
  it('tdpWatts = 150', () => expect(gpu.tdpWatts).toBe(150));
  it('no NVLink', () => {
    expect(gpu.nvlinkBandwidthGBps).toBe(0);
    expect(gpu.nvlinkVersion).toBeNull();
  });
});

describe('GPU spec accuracy: L4', () => {
  const gpu = L4;
  it('architecture is ada', () => expect(gpu.architecture).toBe('ada'));
  it('memoryGB = 24', () => expect(gpu.memoryGB).toBe(24));
  it('memoryBandwidthTBps = 0.3', () => expect(gpu.memoryBandwidthTBps).toBe(0.3));
  it('fp32TFLOPS = 30.3', () => expect(gpu.fp32TFLOPS).toBe(30.3));
  it('bf16TFLOPS = 121', () => expect(gpu.bf16TFLOPS).toBe(121));
  it('fp8TFLOPS = 242', () => expect(gpu.fp8TFLOPS).toBe(242));
  it('int8TOPS = 242', () => expect(gpu.int8TOPS).toBe(242));
  it('tdpWatts = 72', () => expect(gpu.tdpWatts).toBe(72));
  it('no NVLink', () => {
    expect(gpu.nvlinkBandwidthGBps).toBe(0);
    expect(gpu.nvlinkVersion).toBeNull();
  });
});

describe('GPU spec accuracy: RTX 6000 Ada', () => {
  const gpu = RTX_6000_ADA;
  it('architecture is ada', () => expect(gpu.architecture).toBe('ada'));
  it('memoryGB = 48', () => expect(gpu.memoryGB).toBe(48));
  it('memoryBandwidthTBps = 0.96', () => expect(gpu.memoryBandwidthTBps).toBe(0.96));
  it('fp32TFLOPS = 91.1', () => expect(gpu.fp32TFLOPS).toBe(91.1));
  it('bf16TFLOPS = 182.2', () => expect(gpu.bf16TFLOPS).toBe(182.2));
  it('fp8TFLOPS = 364.4', () => expect(gpu.fp8TFLOPS).toBe(364.4));
  it('tdpWatts = 300', () => expect(gpu.tdpWatts).toBe(300));
  it('no NVLink', () => {
    expect(gpu.nvlinkBandwidthGBps).toBe(0);
    expect(gpu.nvlinkVersion).toBeNull();
  });
});

describe('GPU spec accuracy: RTX 4090', () => {
  const gpu = RTX_4090;
  it('architecture is ada', () => expect(gpu.architecture).toBe('ada'));
  it('memoryGB = 24', () => expect(gpu.memoryGB).toBe(24));
  it('memoryBandwidthTBps = 1.008', () => expect(gpu.memoryBandwidthTBps).toBe(1.008));
  it('fp32TFLOPS = 82.6', () => expect(gpu.fp32TFLOPS).toBe(82.6));
  it('bf16TFLOPS = 165.2', () => expect(gpu.bf16TFLOPS).toBe(165.2));
  it('fp8TFLOPS = 330.3', () => expect(gpu.fp8TFLOPS).toBe(330.3));
  it('fp4TFLOPS = 0', () => expect(gpu.fp4TFLOPS).toBe(0));
  it('tdpWatts = 450', () => expect(gpu.tdpWatts).toBe(450));
  it('no NVLink', () => {
    expect(gpu.nvlinkBandwidthGBps).toBe(0);
    expect(gpu.nvlinkVersion).toBeNull();
  });
});

describe('GPU spec accuracy: RTX 3090', () => {
  const gpu = RTX_3090;
  it('architecture is ampere', () => expect(gpu.architecture).toBe('ampere'));
  it('memoryGB = 24', () => expect(gpu.memoryGB).toBe(24));
  it('memoryBandwidthTBps = 0.936', () => expect(gpu.memoryBandwidthTBps).toBe(0.936));
  it('fp32TFLOPS = 35.6', () => expect(gpu.fp32TFLOPS).toBe(35.6));
  it('bf16TFLOPS = 142', () => expect(gpu.bf16TFLOPS).toBe(142));
  it('fp8TFLOPS = 0 (no FP8 on Ampere)', () => expect(gpu.fp8TFLOPS).toBe(0));
  it('int8TOPS = 284', () => expect(gpu.int8TOPS).toBe(284));
  it('tdpWatts = 350', () => expect(gpu.tdpWatts).toBe(350));
  it('no NVLink', () => {
    expect(gpu.nvlinkBandwidthGBps).toBe(0);
    expect(gpu.nvlinkVersion).toBeNull();
  });
  it('hasTransformerEngine = false', () => expect(gpu.hasTransformerEngine).toBe(false));
});

describe('L40 vs L40S: tensor throughput difference', () => {
  it('L40S has ~2x bf16 of L40', () => {
    expect(L40S.bf16TFLOPS).toBe(362);
    expect(L40.bf16TFLOPS).toBe(181);
  });
});

describe('A10 vs A10G: tensor throughput difference', () => {
  it('A10 has ~1.8x bf16 of A10G', () => {
    expect(A10.bf16TFLOPS).toBe(125);
    expect(A10G.bf16TFLOPS).toBe(70);
    expect(A10.bf16TFLOPS / A10G.bf16TFLOPS).toBeCloseTo(1.786, 2);
  });
});

// ---------------------------------------------------------------------------
// Section 2: Spec consistency checks across GPUs
// ---------------------------------------------------------------------------

describe('Spec consistency: compute hierarchy', () => {
  const allGpus = Object.values(ALL_GPUS);

  it('BF16 >= FP16 (or both 0) for every GPU', () => {
    for (const gpu of allGpus) {
      if (gpu.bf16TFLOPS > 0) {
        expect(gpu.bf16TFLOPS, gpu.name).toBeGreaterThanOrEqual(gpu.fp16TFLOPS);
      }
    }
  });

  it('FP8 >= BF16 for Hopper+ GPUs (Transformer Engine doubles throughput)', () => {
    for (const gpu of allGpus) {
      if (gpu.hasTransformerEngine && gpu.fp8TFLOPS > 0) {
        expect(gpu.fp8TFLOPS, gpu.name).toBeGreaterThanOrEqual(gpu.bf16TFLOPS);
      }
    }
  });

  it('FP4 >= FP8 for Blackwell and CDNA4 GPUs', () => {
    for (const gpu of allGpus) {
      if ((gpu.architecture === 'blackwell' || gpu.architecture === 'cdna4') && gpu.fp4TFLOPS > 0) {
        expect(gpu.fp4TFLOPS, gpu.name).toBeGreaterThanOrEqual(gpu.fp8TFLOPS);
      }
    }
  });

  it('INT8 >= BF16 for GPUs with INT8 support', () => {
    for (const gpu of allGpus) {
      if (gpu.int8TOPS > 0 && gpu.bf16TFLOPS > 0) {
        expect(gpu.int8TOPS, gpu.name).toBeGreaterThanOrEqual(gpu.bf16TFLOPS);
      }
    }
  });

  it('all TFLOPS/TOPS values are non-negative', () => {
    for (const gpu of allGpus) {
      expect(gpu.fp32TFLOPS, `${gpu.name} fp32`).toBeGreaterThanOrEqual(0);
      expect(gpu.tf32TFLOPS, `${gpu.name} tf32`).toBeGreaterThanOrEqual(0);
      expect(gpu.fp16TFLOPS, `${gpu.name} fp16`).toBeGreaterThanOrEqual(0);
      expect(gpu.bf16TFLOPS, `${gpu.name} bf16`).toBeGreaterThanOrEqual(0);
      expect(gpu.fp8TFLOPS, `${gpu.name} fp8`).toBeGreaterThanOrEqual(0);
      expect(gpu.fp4TFLOPS, `${gpu.name} fp4`).toBeGreaterThanOrEqual(0);
      expect(gpu.int8TOPS, `${gpu.name} int8`).toBeGreaterThanOrEqual(0);
      expect(gpu.int4TOPS, `${gpu.name} int4`).toBeGreaterThanOrEqual(0);
    }
  });
});

describe('Spec consistency: memory bandwidth scaling', () => {
  it('V100 < A100 40GB < A100 80GB < H100 SXM < H200 < B200', () => {
    expect(V100_32GB.memoryBandwidthTBps).toBeLessThan(A100_40GB.memoryBandwidthTBps);
    expect(A100_40GB.memoryBandwidthTBps).toBeLessThan(A100_80GB.memoryBandwidthTBps);
    expect(A100_80GB.memoryBandwidthTBps).toBeLessThan(H100_SXM.memoryBandwidthTBps);
    expect(H100_SXM.memoryBandwidthTBps).toBeLessThan(H200_SXM.memoryBandwidthTBps);
    expect(H200_SXM.memoryBandwidthTBps).toBeLessThan(B200.memoryBandwidthTBps);
  });
});

describe('Spec consistency: NVLink bandwidth scaling', () => {
  it('V100 < A100 < H100 SXM < B200', () => {
    expect(V100_32GB.nvlinkBandwidthGBps).toBeLessThan(A100_40GB.nvlinkBandwidthGBps);
    expect(A100_40GB.nvlinkBandwidthGBps).toBeLessThan(H100_SXM.nvlinkBandwidthGBps);
    expect(H100_SXM.nvlinkBandwidthGBps).toBeLessThan(B200.nvlinkBandwidthGBps);
  });
});

describe('Spec consistency: features absent where expected', () => {
  it('T4 has no BF16, TF32, FP8, FP4 (Turing)', () => {
    expect(T4.bf16TFLOPS).toBe(0);
    expect(T4.tf32TFLOPS).toBe(0);
    expect(T4.fp8TFLOPS).toBe(0);
    expect(T4.fp4TFLOPS).toBe(0);
  });

  it('V100 has no BF16, TF32, FP8, FP4', () => {
    expect(V100_32GB.bf16TFLOPS).toBe(0);
    expect(V100_32GB.tf32TFLOPS).toBe(0);
    expect(V100_32GB.fp8TFLOPS).toBe(0);
    expect(V100_32GB.fp4TFLOPS).toBe(0);
  });

  it('A100 has no FP8 or FP4', () => {
    expect(A100_40GB.fp8TFLOPS).toBe(0);
    expect(A100_40GB.fp4TFLOPS).toBe(0);
    expect(A100_80GB.fp8TFLOPS).toBe(0);
    expect(A100_80GB.fp4TFLOPS).toBe(0);
  });

  it('Hopper has no FP4', () => {
    expect(H100_SXM.fp4TFLOPS).toBe(0);
    expect(H100_PCIE.fp4TFLOPS).toBe(0);
    expect(H100_NVL.fp4TFLOPS).toBe(0);
    expect(H200_SXM.fp4TFLOPS).toBe(0);
  });

  it('MI300X and MI325X have no FP4 (CDNA3 die)', () => {
    expect(MI300X.fp4TFLOPS).toBe(0);
    expect(MI325X.fp4TFLOPS).toBe(0);
  });

  it('CDNA2 has no TF32, FP8, FP4', () => {
    for (const gpu of [MI250X, MI210]) {
      expect(gpu.tf32TFLOPS, gpu.name).toBe(0);
      expect(gpu.fp8TFLOPS, gpu.name).toBe(0);
      expect(gpu.fp4TFLOPS, gpu.name).toBe(0);
    }
  });

  it('AMD GPUs have no NVLink', () => {
    const amdGpus = [MI250X, MI210, MI300X, MI325X, MI350X];
    for (const gpu of amdGpus) {
      expect(gpu.nvlinkBandwidthGBps, gpu.name).toBe(0);
      expect(gpu.nvlinkVersion, gpu.name).toBeNull();
    }
  });

  it('Ada GPUs have no FP4 (only Blackwell has FP4)', () => {
    expect(L40S.fp4TFLOPS).toBe(0);
    expect(L40.fp4TFLOPS).toBe(0);
    expect(L4.fp4TFLOPS).toBe(0);
    expect(RTX_6000_ADA.fp4TFLOPS).toBe(0);
    expect(RTX_4090.fp4TFLOPS).toBe(0);
  });

  it('Ampere consumer/professional GPUs have no FP8', () => {
    expect(A10.fp8TFLOPS).toBe(0);
    expect(A10G.fp8TFLOPS).toBe(0);
    expect(RTX_3090.fp8TFLOPS).toBe(0);
  });

  it('All new GPUs have no NVLink', () => {
    const newGpus = [T4, L40S, L40, A10, A10G, L4, RTX_6000_ADA, RTX_4090, RTX_3090];
    for (const gpu of newGpus) {
      expect(gpu.nvlinkBandwidthGBps, gpu.name).toBe(0);
      expect(gpu.nvlinkVersion, gpu.name).toBeNull();
      expect(gpu.hasNvSwitch, gpu.name).toBe(false);
    }
  });
});

// ---------------------------------------------------------------------------
// Section 3: Interconnect mapping tests
// ---------------------------------------------------------------------------

describe('Interconnect mapping: intra-node', () => {
  it('V100 (NVLink 2) → NVLink spec', () => {
    const ic = getIntraNodeInterconnect(V100_32GB);
    expect(ic.type).toBe('nvlink');
    // NVLink version 2 spec
    expect(ic.bandwidthGBps).toBe(NVLINK_SPECS[2].bandwidthGBps);
  });

  it('A100 (NVLink 3) → NVLink 3 spec', () => {
    const ic = getIntraNodeInterconnect(A100_80GB);
    expect(ic.type).toBe('nvlink');
    expect(ic.bandwidthGBps).toBe(300);
  });

  it('H100 SXM (NVLink 4) → NVLink 4 spec', () => {
    const ic = getIntraNodeInterconnect(H100_SXM);
    expect(ic.type).toBe('nvlink');
    expect(ic.bandwidthGBps).toBe(450);
  });

  it('H800 SXM (NVLink 4, reduced BW) → NVLink 4 at 200 GB/s', () => {
    const ic = getIntraNodeInterconnect(H800_SXM);
    expect(ic.type).toBe('nvlink');
    expect(ic.bandwidthGBps).toBe(200);  // NOT 450 like H100
    expect(ic.bidirectionalGBps).toBe(400);
  });

  it('A800 80GB (NVLink 3, reduced BW) → NVLink 3 at 200 GB/s', () => {
    const ic = getIntraNodeInterconnect(A800_80GB);
    expect(ic.type).toBe('nvlink');
    expect(ic.bandwidthGBps).toBe(200);  // NOT 300 like A100
    expect(ic.bidirectionalGBps).toBe(400);
  });

  it('H100 PCIe (no NVLink) → PCIe Gen5', () => {
    const ic = getIntraNodeInterconnect(H100_PCIE);
    expect(ic.type).toBe('pcie');
    expect(ic.bandwidthGBps).toBe(63);
  });

  it('H100 NVL (NVLink 4, 300 GB/s bridge) → NVLink 4 at 300 GB/s', () => {
    const ic = getIntraNodeInterconnect(H100_NVL);
    expect(ic.type).toBe('nvlink');
    expect(ic.bandwidthGBps).toBe(300);  // NVL bridge: 300 uni, not full 450
  });

  it('B200 (NVLink 5) → NVLink 5 spec', () => {
    const ic = getIntraNodeInterconnect(B200);
    expect(ic.type).toBe('nvlink');
    expect(ic.bandwidthGBps).toBe(900);
  });

  it('MI300X → Infinity Fabric (MI300)', () => {
    const ic = getIntraNodeInterconnect(MI300X);
    expect(ic.name).toBe('AMD Infinity Fabric (MI300)');
    expect(ic.bandwidthGBps).toBe(INFINITY_FABRIC_SPECS['if-mi300'].bandwidthGBps);
  });

  it('MI325X → Infinity Fabric (MI300, same CDNA3 die)', () => {
    const ic = getIntraNodeInterconnect(MI325X);
    expect(ic.name).toBe('AMD Infinity Fabric (MI300)');
    expect(ic.bandwidthGBps).toBe(448);
  });

  it('MI250X → Infinity Fabric (MI250)', () => {
    const ic = getIntraNodeInterconnect(MI250X);
    expect(ic.name).toBe('AMD Infinity Fabric (MI250)');
    expect(ic.bandwidthGBps).toBe(200);
  });

  it('MI210 → Infinity Fabric (MI250, same CDNA2)', () => {
    const ic = getIntraNodeInterconnect(MI210);
    expect(ic.name).toBe('AMD Infinity Fabric (MI250)');
    expect(ic.bandwidthGBps).toBe(200);
  });

  it('MI350X → Infinity Fabric (MI350)', () => {
    const ic = getIntraNodeInterconnect(MI350X);
    expect(ic.name).toBe('AMD Infinity Fabric (MI350)');
    expect(ic.bandwidthGBps).toBe(538);
  });

  it('T4 (Turing, no NVLink) → PCIe Gen3', () => {
    const ic = getIntraNodeInterconnect(T4);
    expect(ic.type).toBe('pcie');
    expect(ic.bandwidthGBps).toBe(PCIE_SPECS['pcie-gen3'].bandwidthGBps);
  });

  it('Ada GPUs (no NVLink) → PCIe Gen4', () => {
    const adaGpus = [L40S, L40, L4, RTX_6000_ADA, RTX_4090];
    for (const gpu of adaGpus) {
      const ic = getIntraNodeInterconnect(gpu);
      expect(ic.type, gpu.name).toBe('pcie');
      expect(ic.bandwidthGBps, gpu.name).toBe(PCIE_SPECS['pcie-gen4'].bandwidthGBps);
    }
  });

  it('Ampere consumer GPUs (no NVLink) → PCIe Gen4', () => {
    const ampereGpus = [A10, A10G, RTX_3090];
    for (const gpu of ampereGpus) {
      const ic = getIntraNodeInterconnect(gpu);
      expect(ic.type, gpu.name).toBe('pcie');
      expect(ic.bandwidthGBps, gpu.name).toBe(PCIE_SPECS['pcie-gen4'].bandwidthGBps);
    }
  });
});

describe('Interconnect mapping: NVSwitch', () => {
  it('V100 (hasNvSwitch) → nvswitch capped at NVLink 150 GB/s', () => {
    const sw = getNvSwitchSpec(V100_32GB);
    expect(sw).toBeDefined();
    expect(sw!.bandwidthGBps).toBe(150);  // NVLink 2 is 150 GB/s — NVSwitch can't exceed it
  });

  it('A100 (hasNvSwitch) → nvswitch-a100', () => {
    const sw = getNvSwitchSpec(A100_80GB);
    expect(sw).toBeDefined();
    expect(sw!.bandwidthGBps).toBe(300);
  });

  it('H100 SXM (hasNvSwitch) → nvswitch-h100', () => {
    const sw = getNvSwitchSpec(H100_SXM);
    expect(sw).toBeDefined();
    expect(sw!.bandwidthGBps).toBe(450);
  });

  it('H800 SXM (hasNvSwitch, reduced BW) → nvswitch-h100 at 200 GB/s', () => {
    const sw = getNvSwitchSpec(H800_SXM);
    expect(sw).toBeDefined();
    expect(sw!.bandwidthGBps).toBe(200);  // NOT 450 like H100
    expect(sw!.bidirectionalGBps).toBe(400);
  });

  it('A800 80GB (hasNvSwitch, reduced BW) → nvswitch-a100 at 200 GB/s', () => {
    const sw = getNvSwitchSpec(A800_80GB);
    expect(sw).toBeDefined();
    expect(sw!.bandwidthGBps).toBe(200);  // NOT 300 like A100
    expect(sw!.bidirectionalGBps).toBe(400);
  });

  it('H100 PCIe (no NvSwitch) → undefined', () => {
    expect(getNvSwitchSpec(H100_PCIE)).toBeUndefined();
  });

  it('H100 NVL (no NvSwitch) → undefined', () => {
    expect(getNvSwitchSpec(H100_NVL)).toBeUndefined();
  });

  it('B200 (hasNvSwitch) → nvswitch-b200', () => {
    const sw = getNvSwitchSpec(B200);
    expect(sw).toBeDefined();
    expect(sw!.bandwidthGBps).toBe(900);
  });

  it('MI300X (no NvSwitch) → undefined', () => {
    expect(getNvSwitchSpec(MI300X)).toBeUndefined();
  });

  it('T4 (no NvSwitch) → undefined', () => {
    expect(getNvSwitchSpec(T4)).toBeUndefined();
  });

  it('Ada GPUs (no NvSwitch) → undefined', () => {
    const adaGpus = [L40S, L40, L4, RTX_6000_ADA, RTX_4090];
    for (const gpu of adaGpus) {
      expect(getNvSwitchSpec(gpu), gpu.name).toBeUndefined();
    }
  });

  it('Ampere consumer GPUs (no NvSwitch) → undefined', () => {
    const ampereGpus = [A10, A10G, RTX_3090];
    for (const gpu of ampereGpus) {
      expect(getNvSwitchSpec(gpu), gpu.name).toBeUndefined();
    }
  });
});

// ---------------------------------------------------------------------------
// Section 4: GPU × strategy simulation smoke tests
// ---------------------------------------------------------------------------

function makeConfig(
  gpuId: string,
  modelId: string,
  strategyType: SimulationConfig['strategyType'],
  totalGPUs: number,
  numNodes: number,
  strategyConfig?: SimulationConfig['strategyConfig'],
): SimulationConfig {
  return {
    clusterId: undefined,
    clusterConfig: createMultiNodeCluster(gpuId, totalGPUs / numNodes, numNodes)!,
    modelId,
    globalBatchSize: totalGPUs * 2,
    microBatchSize: 2,
    sequenceLength: 2048,
    strategyType,
    strategyConfig,
  };
}

function assertFiniteAndValid(metrics: { stepTimeMs: number; mfu: number }, label: string) {
  expect(metrics.stepTimeMs, `${label}: stepTimeMs > 0`).toBeGreaterThan(0);
  expect(metrics.stepTimeMs, `${label}: stepTimeMs finite`).toBeLessThan(Infinity);
  expect(metrics.mfu, `${label}: mfu > 0`).toBeGreaterThan(0);
  expect(metrics.mfu, `${label}: mfu ≤ 1`).toBeLessThanOrEqual(1);
}

describe('GPU × strategy smoke tests', () => {
  // T4: tiny models only (16 GB GDDR6 memory)
  describe('T4', () => {
    it('gpt3-125m + ddp', () => {
      const m = getValidatedSimulationMetrics(makeConfig('t4', 'gpt3-125m', 'ddp', 8, 1));
      assertFiniteAndValid(m, 'T4/gpt3-125m/ddp');
    });
    it('gpt3-125m + fsdp', () => {
      const m = getValidatedSimulationMetrics(makeConfig('t4', 'gpt3-125m', 'fsdp', 8, 1));
      assertFiniteAndValid(m, 'T4/gpt3-125m/fsdp');
    });
  });

  // V100: small model only (32 GB memory)
  describe('V100', () => {
    it('gpt3-125m + ddp', () => {
      const m = getValidatedSimulationMetrics(makeConfig('v100-32gb', 'gpt3-125m', 'ddp', 8, 1));
      assertFiniteAndValid(m, 'V100/gpt3-125m/ddp');
    });
    it('gpt3-125m + fsdp', () => {
      const m = getValidatedSimulationMetrics(makeConfig('v100-32gb', 'gpt3-125m', 'fsdp', 8, 1));
      assertFiniteAndValid(m, 'V100/gpt3-125m/fsdp');
    });
  });

  // A100 40GB
  describe('A100 40GB', () => {
    it('gpt3-1.3b + ddp', () => {
      const m = getValidatedSimulationMetrics(makeConfig('a100-40gb', 'gpt3-1.3b', 'ddp', 8, 1));
      assertFiniteAndValid(m, 'A100-40/gpt3-1.3b/ddp');
    });
    it('llama3-8b + fsdp-tp (tp=2)', () => {
      const m = getValidatedSimulationMetrics(makeConfig('a100-40gb', 'llama3-8b', 'fsdp-tp', 8, 1, { tp: 2 }));
      assertFiniteAndValid(m, 'A100-40/llama3-8b/fsdp-tp');
    });
  });

  // A100 80GB
  describe('A100 80GB', () => {
    it('gpt3-1.3b + ddp', () => {
      const m = getValidatedSimulationMetrics(makeConfig('a100-80gb', 'gpt3-1.3b', 'ddp', 8, 1));
      assertFiniteAndValid(m, 'A100-80/gpt3-1.3b/ddp');
    });
    it('llama3-8b + fsdp-tp (tp=2)', () => {
      const m = getValidatedSimulationMetrics(makeConfig('a100-80gb', 'llama3-8b', 'fsdp-tp', 8, 1, { tp: 2 }));
      assertFiniteAndValid(m, 'A100-80/llama3-8b/fsdp-tp');
    });
  });

  // H800 SXM
  describe('H800 SXM', () => {
    it('gpt3-1.3b + ddp', () => {
      const m = getValidatedSimulationMetrics(makeConfig('h800-sxm', 'gpt3-1.3b', 'ddp', 8, 1));
      assertFiniteAndValid(m, 'H800-SXM/gpt3-1.3b/ddp');
    });
    it('llama3-8b + fsdp-tp (tp=2)', () => {
      const m = getValidatedSimulationMetrics(makeConfig('h800-sxm', 'llama3-8b', 'fsdp-tp', 8, 1, { tp: 2 }));
      assertFiniteAndValid(m, 'H800-SXM/llama3-8b/fsdp-tp');
    });
  });

  // A800 80GB
  describe('A800 80GB', () => {
    it('gpt3-1.3b + ddp', () => {
      const m = getValidatedSimulationMetrics(makeConfig('a800-80gb', 'gpt3-1.3b', 'ddp', 8, 1));
      assertFiniteAndValid(m, 'A800-80/gpt3-1.3b/ddp');
    });
    it('llama3-8b + fsdp-tp (tp=2)', () => {
      const m = getValidatedSimulationMetrics(makeConfig('a800-80gb', 'llama3-8b', 'fsdp-tp', 8, 1, { tp: 2 }));
      assertFiniteAndValid(m, 'A800-80/llama3-8b/fsdp-tp');
    });
  });

  // H100 SXM
  describe('H100 SXM', () => {
    it('gpt3-1.3b + ddp', () => {
      const m = getValidatedSimulationMetrics(makeConfig('h100-sxm', 'gpt3-1.3b', 'ddp', 8, 1));
      assertFiniteAndValid(m, 'H100-SXM/gpt3-1.3b/ddp');
    });
    it('llama3-8b + fsdp-tp (tp=2)', () => {
      const m = getValidatedSimulationMetrics(makeConfig('h100-sxm', 'llama3-8b', 'fsdp-tp', 8, 1, { tp: 2 }));
      assertFiniteAndValid(m, 'H100-SXM/llama3-8b/fsdp-tp');
    });
  });

  // H100 PCIe
  describe('H100 PCIe', () => {
    it('gpt3-1.3b + ddp', () => {
      const m = getValidatedSimulationMetrics(makeConfig('h100-pcie', 'gpt3-1.3b', 'ddp', 8, 1));
      assertFiniteAndValid(m, 'H100-PCIe/gpt3-1.3b/ddp');
    });
    it('llama3-8b + fsdp-tp (tp=2)', () => {
      const m = getValidatedSimulationMetrics(makeConfig('h100-pcie', 'llama3-8b', 'fsdp-tp', 8, 1, { tp: 2 }));
      assertFiniteAndValid(m, 'H100-PCIe/llama3-8b/fsdp-tp');
    });
  });

  // H100 NVL
  describe('H100 NVL', () => {
    it('gpt3-1.3b + ddp', () => {
      const m = getValidatedSimulationMetrics(makeConfig('h100-nvl', 'gpt3-1.3b', 'ddp', 8, 1));
      assertFiniteAndValid(m, 'H100-NVL/gpt3-1.3b/ddp');
    });
    it('llama3-8b + fsdp-tp (tp=2)', () => {
      const m = getValidatedSimulationMetrics(makeConfig('h100-nvl', 'llama3-8b', 'fsdp-tp', 8, 1, { tp: 2 }));
      assertFiniteAndValid(m, 'H100-NVL/llama3-8b/fsdp-tp');
    });
  });

  // H200
  describe('H200 SXM', () => {
    it('llama3-8b + fsdp-tp (tp=4)', () => {
      const m = getValidatedSimulationMetrics(makeConfig('h200-sxm', 'llama3-8b', 'fsdp-tp', 8, 1, { tp: 4 }));
      assertFiniteAndValid(m, 'H200/llama3-8b/fsdp-tp');
    });
  });

  // B200
  describe('B200', () => {
    it('llama3-8b + fsdp-tp (tp=4)', () => {
      const m = getValidatedSimulationMetrics(makeConfig('b200', 'llama3-8b', 'fsdp-tp', 8, 1, { tp: 4 }));
      assertFiniteAndValid(m, 'B200/llama3-8b/fsdp-tp');
    });
  });

  // GB200
  describe('GB200', () => {
    it('llama3-8b + fsdp-tp (tp=4)', () => {
      const m = getValidatedSimulationMetrics(makeConfig('gb200', 'llama3-8b', 'fsdp-tp', 8, 1, { tp: 4 }));
      assertFiniteAndValid(m, 'GB200/llama3-8b/fsdp-tp');
    });
  });

  // MI300X
  describe('MI300X', () => {
    it('llama3-8b + ddp', () => {
      const m = getValidatedSimulationMetrics(makeConfig('mi300x', 'llama3-8b', 'ddp', 8, 1));
      assertFiniteAndValid(m, 'MI300X/llama3-8b/ddp');
    });
    it('llama3-8b + fsdp-tp (tp=2)', () => {
      const m = getValidatedSimulationMetrics(makeConfig('mi300x', 'llama3-8b', 'fsdp-tp', 8, 1, { tp: 2 }));
      assertFiniteAndValid(m, 'MI300X/llama3-8b/fsdp-tp');
    });
  });

  // MI325X
  describe('MI325X', () => {
    it('llama3-8b + ddp', () => {
      const m = getValidatedSimulationMetrics(makeConfig('mi325x', 'llama3-8b', 'ddp', 8, 1));
      assertFiniteAndValid(m, 'MI325X/llama3-8b/ddp');
    });
    it('llama3-8b + fsdp-tp (tp=2)', () => {
      const m = getValidatedSimulationMetrics(makeConfig('mi325x', 'llama3-8b', 'fsdp-tp', 8, 1, { tp: 2 }));
      assertFiniteAndValid(m, 'MI325X/llama3-8b/fsdp-tp');
    });
  });

  // MI250X (64GB per GCD)
  describe('MI250X', () => {
    it('gpt3-1.3b + ddp', () => {
      const m = getValidatedSimulationMetrics(makeConfig('mi250x', 'gpt3-1.3b', 'ddp', 8, 1));
      assertFiniteAndValid(m, 'MI250X/gpt3-1.3b/ddp');
    });
    it('gpt3-1.3b + fsdp-tp (tp=2)', () => {
      const m = getValidatedSimulationMetrics(makeConfig('mi250x', 'gpt3-1.3b', 'fsdp-tp', 8, 1, { tp: 2 }));
      assertFiniteAndValid(m, 'MI250X/gpt3-1.3b/fsdp-tp');
    });
  });

  // MI210 (64GB)
  describe('MI210', () => {
    it('gpt3-1.3b + ddp', () => {
      const m = getValidatedSimulationMetrics(makeConfig('mi210', 'gpt3-1.3b', 'ddp', 8, 1));
      assertFiniteAndValid(m, 'MI210/gpt3-1.3b/ddp');
    });
    it('gpt3-1.3b + fsdp-tp (tp=2)', () => {
      const m = getValidatedSimulationMetrics(makeConfig('mi210', 'gpt3-1.3b', 'fsdp-tp', 8, 1, { tp: 2 }));
      assertFiniteAndValid(m, 'MI210/gpt3-1.3b/fsdp-tp');
    });
  });

  // MI350X (288GB)
  describe('MI350X', () => {
    it('llama3-8b + ddp', () => {
      const m = getValidatedSimulationMetrics(makeConfig('mi350x', 'llama3-8b', 'ddp', 8, 1));
      assertFiniteAndValid(m, 'MI350X/llama3-8b/ddp');
    });
    it('llama3-8b + fsdp-tp (tp=4)', () => {
      const m = getValidatedSimulationMetrics(makeConfig('mi350x', 'llama3-8b', 'fsdp-tp', 8, 1, { tp: 4 }));
      assertFiniteAndValid(m, 'MI350X/llama3-8b/fsdp-tp');
    });
  });

  // L40S (48GB)
  describe('L40S', () => {
    it('gpt3-1.3b + ddp', () => {
      const m = getValidatedSimulationMetrics(makeConfig('l40s', 'gpt3-1.3b', 'ddp', 8, 1));
      assertFiniteAndValid(m, 'L40S/gpt3-1.3b/ddp');
    });
    it('llama3-8b + fsdp-tp (tp=2)', () => {
      const m = getValidatedSimulationMetrics(makeConfig('l40s', 'llama3-8b', 'fsdp-tp', 8, 1, { tp: 2 }));
      assertFiniteAndValid(m, 'L40S/llama3-8b/fsdp-tp');
    });
  });

  // L40 (48GB)
  describe('L40', () => {
    it('gpt3-1.3b + ddp', () => {
      const m = getValidatedSimulationMetrics(makeConfig('l40', 'gpt3-1.3b', 'ddp', 8, 1));
      assertFiniteAndValid(m, 'L40/gpt3-1.3b/ddp');
    });
    it('llama3-8b + fsdp-tp (tp=2)', () => {
      const m = getValidatedSimulationMetrics(makeConfig('l40', 'llama3-8b', 'fsdp-tp', 8, 1, { tp: 2 }));
      assertFiniteAndValid(m, 'L40/llama3-8b/fsdp-tp');
    });
  });

  // RTX 6000 Ada (48GB)
  describe('RTX 6000 Ada', () => {
    it('gpt3-1.3b + ddp', () => {
      const m = getValidatedSimulationMetrics(makeConfig('rtx-6000-ada', 'gpt3-1.3b', 'ddp', 8, 1));
      assertFiniteAndValid(m, 'RTX6000Ada/gpt3-1.3b/ddp');
    });
    it('llama3-8b + fsdp-tp (tp=2)', () => {
      const m = getValidatedSimulationMetrics(makeConfig('rtx-6000-ada', 'llama3-8b', 'fsdp-tp', 8, 1, { tp: 2 }));
      assertFiniteAndValid(m, 'RTX6000Ada/llama3-8b/fsdp-tp');
    });
  });

  // A10 (24GB)
  describe('A10', () => {
    it('gpt3-125m + ddp', () => {
      const m = getValidatedSimulationMetrics(makeConfig('a10', 'gpt3-125m', 'ddp', 8, 1));
      assertFiniteAndValid(m, 'A10/gpt3-125m/ddp');
    });
    it('gpt3-125m + fsdp-tp (tp=2)', () => {
      const m = getValidatedSimulationMetrics(makeConfig('a10', 'gpt3-125m', 'fsdp-tp', 8, 1, { tp: 2 }));
      assertFiniteAndValid(m, 'A10/gpt3-125m/fsdp-tp');
    });
  });

  // A10G (24GB)
  describe('A10G', () => {
    it('gpt3-125m + ddp', () => {
      const m = getValidatedSimulationMetrics(makeConfig('a10g', 'gpt3-125m', 'ddp', 8, 1));
      assertFiniteAndValid(m, 'A10G/gpt3-125m/ddp');
    });
    it('gpt3-125m + fsdp-tp (tp=2)', () => {
      const m = getValidatedSimulationMetrics(makeConfig('a10g', 'gpt3-125m', 'fsdp-tp', 8, 1, { tp: 2 }));
      assertFiniteAndValid(m, 'A10G/gpt3-125m/fsdp-tp');
    });
  });

  // L4 (24GB)
  describe('L4', () => {
    it('gpt3-125m + ddp', () => {
      const m = getValidatedSimulationMetrics(makeConfig('l4', 'gpt3-125m', 'ddp', 8, 1));
      assertFiniteAndValid(m, 'L4/gpt3-125m/ddp');
    });
    it('gpt3-125m + fsdp-tp (tp=2)', () => {
      const m = getValidatedSimulationMetrics(makeConfig('l4', 'gpt3-125m', 'fsdp-tp', 8, 1, { tp: 2 }));
      assertFiniteAndValid(m, 'L4/gpt3-125m/fsdp-tp');
    });
  });

  // RTX 4090 (24GB)
  describe('RTX 4090', () => {
    it('gpt3-125m + ddp', () => {
      const m = getValidatedSimulationMetrics(makeConfig('rtx-4090', 'gpt3-125m', 'ddp', 8, 1));
      assertFiniteAndValid(m, 'RTX4090/gpt3-125m/ddp');
    });
    it('gpt3-125m + fsdp-tp (tp=2)', () => {
      const m = getValidatedSimulationMetrics(makeConfig('rtx-4090', 'gpt3-125m', 'fsdp-tp', 8, 1, { tp: 2 }));
      assertFiniteAndValid(m, 'RTX4090/gpt3-125m/fsdp-tp');
    });
  });

  // RTX 3090 (24GB)
  describe('RTX 3090', () => {
    it('gpt3-125m + ddp', () => {
      const m = getValidatedSimulationMetrics(makeConfig('rtx-3090', 'gpt3-125m', 'ddp', 8, 1));
      assertFiniteAndValid(m, 'RTX3090/gpt3-125m/ddp');
    });
    it('gpt3-125m + fsdp-tp (tp=2)', () => {
      const m = getValidatedSimulationMetrics(makeConfig('rtx-3090', 'gpt3-125m', 'fsdp-tp', 8, 1, { tp: 2 }));
      assertFiniteAndValid(m, 'RTX3090/gpt3-125m/fsdp-tp');
    });
  });
});
