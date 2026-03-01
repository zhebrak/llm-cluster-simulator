/**
 * Skill catalog — single source of truth for all earnable RPG skills.
 * Missions reference skill IDs from this registry.
 */

import type { RPGSkill } from './types.ts';

export const ALL_SKILLS: Record<string, RPGSkill> = {
  quantization: {
    id: 'quantization',
    name: 'Quantization',
    description: 'Reduce weight precision to fit models in limited GPU memory',
  },
  'model-selection': {
    id: 'model-selection',
    name: 'Model Selection',
    description: 'Choose the right model size for your hardware constraints',
  },
  'gpu-bandwidth': {
    id: 'gpu-bandwidth',
    name: 'GPU Bandwidth',
    description: 'Understand how memory bandwidth determines decode throughput',
  },
  'batch-size': {
    id: 'batch-size',
    name: 'Batch Sizing',
    description: 'Use batching to amortize weight reads across concurrent requests',
  },
  'kv-cache': {
    id: 'kv-cache',
    name: 'KV Cache Management',
    description: 'Manage KV cache memory with Flash Attention, quantization, and paging',
  },
  'tensor-parallelism': {
    id: 'tensor-parallelism',
    name: 'Tensor Parallelism',
    description: 'Split model layers across GPUs for capacity and latency',
  },
  'cost-optimization': {
    id: 'cost-optimization',
    name: 'Cost Optimization',
    description: 'Minimize GPU spend through quantization and efficient allocation',
  },
};
