/**
 * Hybrid Strategy Combination Validator
 *
 * Validates combinations of data parallelism backends (DDP, FSDP, ZeRO-1/2/3)
 * with model parallelism strategies (TP, PP) based on published research and
 * framework documentation.
 */

export type DPType = 'ddp' | 'fsdp' | 'zero-1' | 'zero-2' | 'zero-3';

export interface HybridCombinationRule {
  dpType: DPType;
  tp: boolean;  // true if TP > 1
  pp: boolean;  // true if PP > 1
  valid: boolean | 'caution';
  warning?: string;
  reference?: string;
}

/**
 * Rules for hybrid strategy combinations based on published research:
 *
 * Valid combinations:
 * - DDP + TP: Standard 2D parallelism
 * - DDP + PP: Megatron-LM default
 * - DDP + TP + PP: GPT-3 175B (Megatron-LM)
 * - FSDP + TP: LLaMA 3 405B (Meta)
 * - FSDP + PP: TorchTitan
 * - FSDP + TP + PP: TorchTitan 3D
 * - ZeRO-1 + TP: DeepSpeed
 * - ZeRO-1 + TP + PP: BLOOM 176B, MT-NLG 530B
 *
 * Caution:
 * - ZeRO-2 + TP: Higher communication than ZeRO-1
 * - ZeRO-3 + TP: 1.5x DDP communication volume
 *
 * Not recommended:
 * - ZeRO-2 + PP: Reduce-scatter per micro-batch
 * - ZeRO-3 + PP: Excessive parameter gathering per micro-batch
 */
export const HYBRID_COMBINATION_RULES: HybridCombinationRule[] = [
  // ZeRO-2 + PP: Not recommended (with or without TP)
  {
    dpType: 'zero-2',
    tp: false,
    pp: true,
    valid: false,
    warning: 'ZeRO-2 + PP is not recommended: reduce-scatter occurs per micro-batch, causing excessive communication overhead. Consider ZeRO-1 + PP instead.',
    reference: 'DeepSpeed documentation',
  },
  {
    dpType: 'zero-2',
    tp: true,
    pp: true,
    valid: false,
    warning: 'ZeRO-2 + TP + PP is not recommended: reduce-scatter occurs per micro-batch. Consider ZeRO-1 + TP + PP (BLOOM/MT-NLG style) instead.',
    reference: 'DeepSpeed documentation',
  },

  // ZeRO-3 + PP: Not recommended (with or without TP)
  {
    dpType: 'zero-3',
    tp: false,
    pp: true,
    valid: false,
    warning: 'ZeRO-3 + PP is not recommended: parameter gathering occurs per micro-batch, causing excessive communication. Consider FSDP + PP or ZeRO-1 + PP instead.',
    reference: 'DeepSpeed documentation',
  },
  {
    dpType: 'zero-3',
    tp: true,
    pp: true,
    valid: false,
    warning: 'ZeRO-3 + TP + PP is not recommended: excessive parameter gathering per micro-batch. Consider FSDP + TP + PP (TorchTitan style) or ZeRO-1 + TP + PP instead.',
    reference: 'DeepSpeed documentation',
  },

  // ZeRO-2 + TP (no PP): Caution
  {
    dpType: 'zero-2',
    tp: true,
    pp: false,
    valid: 'caution',
    warning: 'ZeRO-2 + TP has higher communication than ZeRO-1 + TP due to gradient sharding. Consider ZeRO-1 + TP for better performance.',
    reference: 'DeepSpeed ZeRO paper',
  },

  // ZeRO-3 + TP (no PP): Caution
  {
    dpType: 'zero-3',
    tp: true,
    pp: false,
    valid: 'caution',
    warning: 'ZeRO-3 + TP has ~1.5x DDP communication volume due to parameter gathering. Consider FSDP + TP (LLaMA 3 style) for better efficiency.',
    reference: 'ZeRO paper, LLaMA 3 paper',
  },
];

export interface HybridValidationResult {
  severity: 'error' | 'warning';
  message: string;
  reference?: string;
}

/**
 * Validates a hybrid parallelism combination and returns any warnings or errors.
 *
 * @param config - The parallelism configuration to validate
 * @returns Validation result with severity and message, or null if valid
 */
export function validateHybridCombination(config: {
  dpType: DPType;
  tp: number;
  pp: number;
  gpusPerNode?: number;
}): HybridValidationResult | null {
  const { dpType, tp, pp } = config;

  const hasTP = tp > 1;
  const hasPP = pp > 1;

  // Find matching rule
  const rule = HYBRID_COMBINATION_RULES.find(r =>
    r.dpType === dpType &&
    r.tp === hasTP &&
    r.pp === hasPP
  );

  if (!rule) {
    // No rule means combination is valid
    return null;
  }

  if (rule.valid === false) {
    return {
      severity: 'warning', // Use warning not error - let users proceed if they want
      message: rule.warning!,
      reference: rule.reference,
    };
  }

  if (rule.valid === 'caution') {
    return {
      severity: 'warning',
      message: rule.warning!,
      reference: rule.reference,
    };
  }

  return null;
}

/**
 * Validates that TP degree doesn't exceed GPUs per node (cross-node TP).
 * Cross-node TP requires NVLink bandwidth which is 5-10x faster than InfiniBand.
 *
 * @param tp - Tensor parallelism degree
 * @param gpusPerNode - Number of GPUs per node
 * @returns Validation result if TP crosses nodes, null otherwise
 */
export function validateTPTopology(
  tp: number,
  gpusPerNode: number
): HybridValidationResult | null {
  if (tp > gpusPerNode) {
    return {
      severity: 'warning',
      message: `TP=${tp} exceeds GPUs per node (${gpusPerNode}). ` +
        'Cross-node TP requires NVLink-level bandwidth; InfiniBand is 5-10x slower. ' +
        'Consider reducing TP to fit within a node.',
      reference: 'Megatron-LM best practices',
    };
  }
  return null;
}

/**
 * Get recommended hybrid configurations for a given scenario.
 */
export function getRecommendedHybridConfigs(scenario: {
  modelSizeB: number;
  numGPUs: number;
  gpusPerNode: number;
  memoryPerGPU: number;
}): Array<{
  name: string;
  description: string;
  config: { dpType: DPType; tp: number; pp: number };
}> {
  const { modelSizeB, gpusPerNode } = scenario;
  const recommendations = [];

  // Small models (< 10B): DDP or FSDP sufficient
  if (modelSizeB < 10) {
    recommendations.push({
      name: 'FSDP',
      description: 'Sufficient for models < 10B parameters',
      config: { dpType: 'fsdp' as DPType, tp: 1, pp: 1 },
    });
  }

  // Medium models (10-70B): FSDP + TP or ZeRO-1 + TP
  if (modelSizeB >= 10 && modelSizeB < 100) {
    const tp = gpusPerNode;
    recommendations.push({
      name: 'FSDP + TP (LLaMA style)',
      description: `TP=${tp} within node, FSDP across nodes`,
      config: { dpType: 'fsdp' as DPType, tp, pp: 1 },
    });
    recommendations.push({
      name: 'ZeRO-1 + TP (DeepSpeed style)',
      description: `TP=${tp} within node, ZeRO-1 across nodes`,
      config: { dpType: 'zero-1' as DPType, tp, pp: 1 },
    });
  }

  // Large models (70B+): Need 3D parallelism
  if (modelSizeB >= 70) {
    const tp = gpusPerNode;
    const pp = Math.min(16, Math.ceil(modelSizeB / 40)); // ~40B per PP stage
    recommendations.push({
      name: 'FSDP + TP + PP (TorchTitan style)',
      description: `TP=${tp}, PP=${pp}, FSDP for remaining GPUs`,
      config: { dpType: 'fsdp' as DPType, tp, pp },
    });
    recommendations.push({
      name: 'ZeRO-1 + TP + PP (BLOOM/MT-NLG style)',
      description: `TP=${tp}, PP=${pp}, ZeRO-1 for remaining GPUs`,
      config: { dpType: 'zero-1' as DPType, tp, pp },
    });
  }

  return recommendations;
}
