/**
 * GPU hourly pricing for dashboard cost estimates
 *
 * Median reserved-capacity rates ($/GPU-hour, 1yr commitment)
 * Sources: AWS/GCP/Azure 1yr RI, Lambda/CoreWeave reserved, RunPod committed (Feb 2026)
 */

/**
 * Simple per-GPU-hour pricing estimates
 */
export const GPU_HOURLY_RATES: Record<string, { rate: number; name: string }> = {
  'h200-sxm': { rate: 3.50, name: 'H200 SXM' },
  'h100-sxm': { rate: 2.50, name: 'H100 SXM' },
  'h800-sxm': { rate: 2.25, name: 'H800 SXM' },
  'h100-pcie': { rate: 2.25, name: 'H100 PCIe' },
  'h100-nvl': { rate: 3.00, name: 'H100 NVL' },
  'b200': { rate: 5.00, name: 'B200' },
  'gb200': { rate: 6.00, name: 'GB200' },
  'a100-80gb': { rate: 1.75, name: 'A100 80GB' },
  'a800-80gb': { rate: 1.50, name: 'A800 80GB' },
  'a100-40gb': { rate: 1.50, name: 'A100 40GB' },
  'v100-32gb': { rate: 0.50, name: 'V100' },
  't4': { rate: 0.25, name: 'T4' },
  'l4': { rate: 0.50, name: 'L4' },
  'l40': { rate: 1.00, name: 'L40' },
  'l40s': { rate: 1.00, name: 'L40S' },
  'a10': { rate: 0.50, name: 'A10' },
  'a10g': { rate: 1.00, name: 'A10G' },
  'rtx-6000-ada': { rate: 0.75, name: 'RTX 6000 Ada' },
  'rtx-4090': { rate: 0.50, name: 'RTX 4090' },
  'rtx-3090': { rate: 0.25, name: 'RTX 3090' },
  'mi210': { rate: 1.00, name: 'MI210' },
  'mi250x': { rate: 1.50, name: 'MI250X' },
  'mi300x': { rate: 2.00, name: 'MI300X' },
  'mi325x': { rate: 2.50, name: 'MI325X' },
  'mi350x': { rate: 4.50, name: 'MI350X' },
};

/** Default rate when GPU ID is not found */
export const DEFAULT_GPU_HOURLY_RATE = { rate: 2.50, name: 'GPU' };

/** Get hourly rate for a GPU, with fallback to default */
export function getGPUHourlyRate(gpuId: string): { rate: number; name: string } {
  return GPU_HOURLY_RATES[gpuId] ?? DEFAULT_GPU_HOURLY_RATE;
}
