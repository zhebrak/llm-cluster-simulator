/**
 * Strategies module exports
 */

// Import for internal use in getStrategy factory
import { ParallelismStrategy as BaseParallelismStrategy } from './base.ts';
import { ddpStrategy as ddp } from './ddp.ts';
import { FSDPStrategy, type FSDPConfig } from './fsdp.ts';
import {
  createZeROStrategy as createZeRO,
  type ZeROConfig,
} from './zero.ts';
import {
  createTensorParallelStrategy as createTP,
  type TensorParallelConfig,
} from './tensor-parallel.ts';
import {
  createPipelineParallelStrategy as createPP,
  type PipelineParallelConfig,
} from './pipeline-parallel.ts';
import {
  createSequenceParallelStrategy as createSP,
} from './sequence-parallel.ts';
import {
  create3DParallelStrategy as create3D,
  type ThreeDParallelConfig,
} from './3d-parallel.ts';
import type { PipelineSchedule } from '../../types/index.ts';

// Re-export base
export type { StrategyContext } from './base.ts';
export {
  ParallelismStrategy,
  calculateParamMemory,
  calculateGradientMemory,
  calculateOptimizerMemory,
  estimateActivationMemory,
  calculateReservedMemory,
  calculateTemporaryMemory,
  gpuCapacityBytes,
} from './base.ts';

// Re-export DDP
export { DDPStrategy, ddpStrategy } from './ddp.ts';

// Re-export FSDP
export type { FSDPConfig } from './fsdp.ts';
export { FSDPStrategy, fsdpStrategy, DEFAULT_FSDP_CONFIG } from './fsdp.ts';

// Re-export ZeRO
export type { ZeROStage, ZeROConfig } from './zero.ts';
export {
  ZeROStrategy,
  zeroStage1Strategy,
  createZeROStrategy,
  DEFAULT_ZERO_CONFIG,
} from './zero.ts';

// Re-export Tensor Parallel
export type { TensorParallelConfig } from './tensor-parallel.ts';
export {
  TensorParallelStrategy,
  createTensorParallelStrategy,
  DEFAULT_TP_CONFIG,
} from './tensor-parallel.ts';

// Re-export Pipeline Parallel
export type { PipelineParallelConfig } from './pipeline-parallel.ts';
export {
  PipelineParallelStrategy,
  createPipelineParallelStrategy,
  DEFAULT_PP_CONFIG,
} from './pipeline-parallel.ts';

// Re-export Sequence Parallel
export type { SequenceParallelConfig } from './sequence-parallel.ts';
export {
  SequenceParallelStrategy,
  createSequenceParallelStrategy,
  DEFAULT_SP_CONFIG,
} from './sequence-parallel.ts';

// Re-export 3D Parallel
export type { ThreeDParallelConfig } from './3d-parallel.ts';
export {
  ThreeDParallelStrategy,
  create3DParallelStrategy,
  autoConfig3DParallel,
  DEFAULT_3D_CONFIG,
} from './3d-parallel.ts';

// Re-export LoRA
export type { FinetuningMethod, LoraTargetModules, LoraConfig } from './lora.ts';
export {
  computeLoraTrainableParams,
  computeLoraParamsPerRank,
  getQloraDequantTimeMs,
  getLoraBackwardMultiplier,
  NF4_BYTES_PER_PARAM,
} from './lora.ts';

// Strategy factory using imported references
export function getStrategy(type: string, config?: Record<string, unknown>): BaseParallelismStrategy | undefined {
  switch (type) {
    case 'ddp':
      return ddp;
    case 'fsdp':
      return new FSDPStrategy(config as Partial<FSDPConfig>);
    case 'zero-1':
      return createZeRO(1, config as Partial<ZeROConfig>);
    case 'zero-2':
      return createZeRO(2, config as Partial<ZeROConfig>);
    case 'zero-3':
      return createZeRO(3, config as Partial<ZeROConfig>);
    case 'tp':
    case 'tensor-parallel':
      return createTP(
        (config?.degree as number) ?? 2,
        config as Partial<TensorParallelConfig>
      );
    case 'pp':
    case 'pipeline-parallel':
      return createPP(
        (config?.degree as number) ?? 2,
        (config?.numMicroBatches as number) ?? 8,
        (config?.schedule as PipelineSchedule) ?? '1f1b',
        config as Partial<PipelineParallelConfig>
      );
    case 'sp':
    case 'sequence-parallel':
      return createSP((config?.tpDegree as number) ?? 2);
    case '3d':
    case '3d-parallel':
      return create3D(
        (config?.tp as number) ?? 1,
        (config?.pp as number) ?? 1,
        (config?.dp as number) ?? 1,
        config as Partial<ThreeDParallelConfig>
      );
    default:
      return undefined;
  }
}

// All available strategies
export const STRATEGY_GROUPS = [
  { id: 'data-parallel', name: 'Data Parallelism' },
  { id: 'hybrid-parallel', name: 'Hybrid Parallelism' },
  { id: '3d-parallel', name: '3D Parallelism' },
] as const;

export type StrategyGroupId = typeof STRATEGY_GROUPS[number]['id'];

export const AVAILABLE_STRATEGIES = [
  { id: 'ddp', name: 'DDP', group: 'data-parallel' as StrategyGroupId, description: 'Simple data parallelism, good for models that fit in GPU memory' },
  { id: 'zero-1', name: 'ZeRO-1', group: 'data-parallel' as StrategyGroupId, description: 'Sharded optimizer states (DeepSpeed)' },
  { id: 'fsdp', name: 'FSDP (ZeRO-3)', group: 'data-parallel' as StrategyGroupId, description: 'Fully sharded params, grads, and optimizer states' },
  { id: 'zero1-tp', name: 'ZeRO-1 + TP', group: 'hybrid-parallel' as StrategyGroupId, description: 'ZeRO-1 across nodes, TP within nodes (DeepSpeed)' },
  { id: 'fsdp-tp', name: 'FSDP + TP', group: 'hybrid-parallel' as StrategyGroupId, description: 'FSDP across nodes, TP within nodes (LLaMA 3 style)' },
  { id: 'ddp-tp-pp', name: 'DDP + TP + PP', group: '3d-parallel' as StrategyGroupId, description: 'Megatron-LM: TP within node, PP across nodes, DDP for data' },
  { id: 'zero1-tp-pp', name: 'ZeRO-1 + TP + PP', group: '3d-parallel' as StrategyGroupId, description: 'Megatron-DeepSpeed: ZeRO-1 + TP + PP (BLOOM style)' },
  { id: 'fsdp-tp-pp', name: 'FSDP + TP + PP', group: '3d-parallel' as StrategyGroupId, description: 'TorchTitan: FSDP + TP + PP for largest models' },
];
