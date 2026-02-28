/**
 * Perturbable Inference Parameters
 *
 * Maps all 11 perturbable inference parameters to their get/set accessor pairs.
 * Used by inference sensitivity analysis to perturb each parameter independently
 * and measure throughput/TPOT/TTFT deltas.
 */

import type { PerturbableParam } from './sensitivity.ts';

import {
  getPrefillResidual, setPrefillResidual,
  getDecodeSamplingOverheadMs, setDecodeSamplingOverheadMs,
  getNvlinkPerRoundMs, setNvlinkPerRoundMs,
  getPciePerRoundMs, setPciePerRoundMs,
  getTpCommEfficiency, setTpCommEfficiency,
  getEpPrefillOverlap, setEpPrefillOverlap,
  getBwEffFloor, setBwEffFloor,
  getBwEffScale, setBwEffScale,
  getCbSchedulingBase, setCbSchedulingBase,
  getCbPrefillInterferenceMax, setCbPrefillInterferenceMax,
  resetInferenceLatencyParams,
} from '../inference/latency.ts';

import {
  getMemoryOverheadFactor, setMemoryOverheadFactor,
  resetInferenceMemoryParams,
} from '../inference/memory.ts';

/**
 * Returns all 11 perturbable inference parameters with get/set wrappers.
 */
export function getAllPerturbableInferenceParams(): PerturbableParam[] {
  return [
    // latency.ts — fitted
    { name: 'PREFILL_RESIDUAL', getValue: getPrefillResidual, setValue: setPrefillResidual },
    { name: 'BW_EFF_FLOOR', getValue: getBwEffFloor, setValue: setBwEffFloor },
    { name: 'BW_EFF_SCALE', getValue: getBwEffScale, setValue: setBwEffScale },
    { name: 'EP_PREFILL_OVERLAP', getValue: getEpPrefillOverlap, setValue: setEpPrefillOverlap },
    { name: 'CB_PREFILL_INTERFERENCE_MAX', getValue: getCbPrefillInterferenceMax, setValue: setCbPrefillInterferenceMax },

    // latency.ts — grounded-empirical
    { name: 'DECODE_SAMPLING_OVERHEAD_MS', getValue: getDecodeSamplingOverheadMs, setValue: setDecodeSamplingOverheadMs },
    { name: 'NVLINK_PER_ROUND_MS', getValue: getNvlinkPerRoundMs, setValue: setNvlinkPerRoundMs },
    { name: 'PCIE_PER_ROUND_MS', getValue: getPciePerRoundMs, setValue: setPciePerRoundMs },
    { name: 'TP_COMM_EFFICIENCY', getValue: getTpCommEfficiency, setValue: setTpCommEfficiency },
    { name: 'CB_SCHEDULING_BASE', getValue: getCbSchedulingBase, setValue: setCbSchedulingBase },

    // memory.ts — grounded-empirical
    { name: 'MEMORY_OVERHEAD_FACTOR', getValue: getMemoryOverheadFactor, setValue: setMemoryOverheadFactor },
  ];
}

/**
 * Reset all perturbable inference parameters to their default values.
 */
export function resetAllPerturbableInferenceParams(): void {
  resetInferenceLatencyParams();
  resetInferenceMemoryParams();
}
