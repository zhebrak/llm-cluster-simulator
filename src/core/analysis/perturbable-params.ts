/**
 * Perturbable Fitted Parameters
 *
 * Maps all 11 perturbable parameters (8 fitted + 3 grounded-empirical) from the parameter registry to
 * their get/set accessor pairs. Used by sensitivity analysis to perturb each
 * parameter independently and measure MFU delta.
 */

import type { PerturbableParam } from './sensitivity.ts';

import {
  getRuntimeResidual, setRuntimeResidual,
  getDpBwAlpha, setDpBwAlpha,
  getDpBwFloor, setDpBwFloor,
} from '../strategies/base.ts';

import {
  getCpAllGatherOverlapInterleaved, setCpAllGatherOverlapInterleaved,
  getPpStageTransitionMs, setPpStageTransitionMs,
} from '../strategies/3d-parallel.ts';

import {
  getExpertGemmExponent, setExpertGemmExponent,
  getExpertCountScale, setExpertCountScale,
} from '../hardware/gpu.ts';

import {
  getOverlapConstants,
  setOverlapConstant,
  setProtocolOverheadEntry,
  resetOverlapConstants,
} from '../strategies/overlap.ts';

/**
 * Returns all perturbable parameters with get/set wrappers.
 *
 * Each entry provides the parameter name (matching parameter-registry.ts),
 * a getValue() that reads the current value, and a setValue() that mutates it.
 * Callers must restore original values after perturbation.
 */
export function getAllPerturbableFittedParams(): PerturbableParam[] {
  return [
    // base.ts
    { name: 'runtimeResidual', getValue: () => getRuntimeResidual(), setValue: (v: number) => setRuntimeResidual(v) },
    { name: 'moeRuntimeResidual', getValue: () => getRuntimeResidual(true), setValue: (v: number) => setRuntimeResidual(v, true) },
    { name: 'DP_BW_ALPHA', getValue: getDpBwAlpha, setValue: setDpBwAlpha },
    { name: 'DP_BW_FLOOR', getValue: getDpBwFloor, setValue: setDpBwFloor },
    // overlap.ts — wrapped via setOverlapConstant/setProtocolOverheadEntry
    {
      name: 'SCHEDULING_EFFICIENCY',
      getValue: () => getOverlapConstants().schedulingEfficiency,
      setValue: (v: number) => setOverlapConstant('schedulingEfficiency', v),
    },
    {
      name: 'PER_COLLECTIVE_OVERHEAD_MS',
      getValue: () => getOverlapConstants().perCollectiveOverheadMs,
      setValue: (v: number) => setOverlapConstant('perCollectiveOverheadMs', v),
    },
    {
      name: 'PROTOCOL_OVERHEAD (DP FSDP)',
      getValue: () => getOverlapConstants().protocolOverhead.dp_fsdp,
      setValue: (v: number) => setProtocolOverheadEntry('dp_fsdp', v),
    },

    // 3d-parallel.ts
    { name: 'CP all-gather overlap (interleaved)', getValue: getCpAllGatherOverlapInterleaved, setValue: setCpAllGatherOverlapInterleaved },
    { name: 'PP_STAGE_TRANSITION_MS', getValue: getPpStageTransitionMs, setValue: setPpStageTransitionMs },

    // gpu.ts — grouped GEMM efficiency for fine-grained MoE
    { name: 'EXPERT_GEMM_EXPONENT', getValue: getExpertGemmExponent, setValue: setExpertGemmExponent },
    { name: 'EXPERT_COUNT_SCALE', getValue: getExpertCountScale, setValue: setExpertCountScale },
  ];
}

/**
 * Reset all perturbable parameters to their default values.
 * Useful as afterEach() cleanup in tests.
 */
export function resetAllPerturbableParams(): void {
  setRuntimeResidual(0.655);
  setRuntimeResidual(0.97, true);
  setDpBwAlpha(0.15);
  setDpBwFloor(0.40);
  resetOverlapConstants();
  setCpAllGatherOverlapInterleaved(0.15);
  setPpStageTransitionMs(0.020);
  setExpertGemmExponent(0.80);
  setExpertCountScale(0.55);
}
