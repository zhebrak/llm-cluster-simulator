/**
 * Base types for the distributed training simulator
 */

// Data types for tensors with their byte sizes
export type DType = 'fp32' | 'tf32' | 'fp16' | 'bf16' | 'fp8' | 'fp4';

export const DTYPE_BYTES: Record<DType, number> = {
  fp32: 4,
  tf32: 4,    // TF32 uses 19 bits but stored in fp32 containers
  fp16: 2,
  bf16: 2,
  fp8: 1,
  fp4: 0.5,
};

// Tensor specification
export interface TensorSpec {
  shape: number[];
  dtype: DType;
}

// Calculate tensor size in bytes
export function tensorBytes(spec: TensorSpec): number {
  const elements = spec.shape.reduce((a, b) => a * b, 1);
  return elements * DTYPE_BYTES[spec.dtype];
}

// Memory breakdown for training
export interface MemoryBreakdown {
  parameters: number;      // Model parameters in bytes
  gradients: number;       // Gradient storage in bytes
  optimizerStates: number; // Optimizer states (momentum, variance, etc.)
  activations: number;     // Activation memory for forward pass
  peakActivations: number; // Peak activation memory (backward pass)
  temporary: number;       // Temporary buffers
  reserved: number;        // Framework/CUDA reserved memory
  total: number;           // Total memory requirement
}

// Computation breakdown
export interface ComputeBreakdown {
  forward: number;         // Forward pass FLOPs
  backward: number;        // Backward pass FLOPs (typically 2x forward)
  total: number;           // Total FLOPs per iteration
}

// Communication breakdown
export interface CommunicationBreakdown {
  dataParallel: number;    // Bytes for DP communication (AllReduce gradients)
  tensorParallel: number;  // Bytes for TP communication (AllReduce activations)
  pipelineParallel: number; // Bytes for PP communication (send/recv activations)
  expertParallel: number;  // Bytes for EP communication (All-to-All)
  contextParallel?: number; // Bytes for CP communication (ring attention KV exchange)
  total: number;
}

// Timing breakdown in milliseconds
export interface TimingBreakdown {
  forward: number;
  backward: number;
  optimizer: number;
  communication: number;
  epCommunication?: number;  // EP all-to-all comm embedded in forward/backward (MoE only)
  overlap: number;         // Time saved by compute-communication overlap
  scaleOverhead: number;   // Large-cluster straggler + congestion overhead
  total: number;           // Total step time
  // Forward/backward sub-breakdown (all strategies)
  forwardComputeMs?: number;     // Matmul compute in forward (GA-scaled, pipeline-adjusted)
  forwardNonMatmulMs?: number;   // Non-matmul in forward: norms, activations, residual adds
  backwardComputeMs?: number;    // Matmul compute in backward (incl. recompute overhead if AC)
  backwardNonMatmulMs?: number;  // Non-matmul in backward
  // Per-dimension comm breakdown (all strategies)
  tpExposed?: number;              // TP comm after layer-overlap applied
  ppExposed?: number;              // PP P2P after C/(C+T) overlap applied
  dpGross?: number;                // DP comm after protocol overhead, before overlap
  cpExposed?: number;              // CP comm after jitter/overlap
  pipelineBubbleFraction?: number; // Bubble fraction (0-1)
  // MoE compute decomposition — forward pass only (3D parallel, MoE only)
  moeExpertFwdMs?: number;   // Expert compute in forward (routed + shared, incl. grouped GEMM + load imbalance)
  moeDenseFwdMs?: number;    // Dense/attention compute in forward (dense layers + MoE-layer attention + non-matmul)
  moeOverheadFwdMs?: number; // Dispatch overhead in forward (router + permutation + coordination)
  epCommFwdMs?: number;      // EP all-to-all in forward (dispatch + combine, after slack overlap)
  // MoE compute decomposition — backward pass only
  moeExpertBwdMs?: number;
  moeDenseBwdMs?: number;
  moeOverheadBwdMs?: number;
  epCommBwdMs?: number;      // EP all-to-all in backward (1× without AC, 2× with AC: recompute + gradient)
  groupedGemmFactor?: number;    // Effective grouped GEMM efficiency factor
  routedFlopFraction?: number;   // routedFlops / flopsPerToken
}

// Units helper
export type ByteUnit = 'B' | 'KB' | 'MB' | 'GB' | 'TB';
export type FlopUnit = 'FLOP' | 'KFLOP' | 'MFLOP' | 'GFLOP' | 'TFLOP' | 'PFLOP';
export type TimeUnit = 'us' | 'ms' | 's' | 'min' | 'hr' | 'day';

/** Convert bytes to display "GB" (GiB, matching GPU memory convention). */
export const bytesToGB = (bytes: number): number => bytes / (1024 ** 3);

/**
 * Format byte count for display. Uses binary base (1024) so that
 * displayed "GB" matches GPU vendor labels (which are GiB in practice).
 */
export function formatBytes(bytes: number, decimals = 2): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes: ByteUnit[] = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(decimals))} ${sizes[i]}`;
}

export function formatFlops(flops: number, decimals = 2): string {
  if (flops === 0) return '0 FLOP';
  const k = 1000;
  const sizes: FlopUnit[] = ['FLOP', 'KFLOP', 'MFLOP', 'GFLOP', 'TFLOP', 'PFLOP'];
  const i = Math.min(Math.floor(Math.log(flops) / Math.log(k)), sizes.length - 1);
  return `${parseFloat((flops / Math.pow(k, i)).toFixed(decimals))} ${sizes[i]}`;
}

export function formatTime(ms: number, decimals = 2): string {
  if (ms < 1) return `${(ms * 1000).toFixed(decimals)} µs`;
  if (ms < 1000) return `${ms.toFixed(decimals)} ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(decimals)} s`;
  if (ms < 3600000) return `${(ms / 60000).toFixed(decimals)} min`;
  if (ms < 86400000) return `${(ms / 3600000).toFixed(decimals)} hr`;
  return `${(ms / 86400000).toFixed(decimals)} day`;
}

export function formatNumber(n: number, decimals = 1): string {
  if (n === 0) return '0';
  if (Math.abs(n) < 1000) return n.toFixed(decimals).replace(/\.0+$/, '');
  const k = 1000;
  const sizes = ['', 'k', 'M', 'B', 'T'];
  const i = Math.min(Math.floor(Math.log(Math.abs(n)) / Math.log(k)), sizes.length - 1);
  const formatted = (n / Math.pow(k, i)).toFixed(decimals).replace(/\.0+$/, '');
  return `${formatted}${sizes[i]}`;
}

/**
 * Format integers with comma separators (for architecture values like 4,096)
 */
export function formatInteger(n: number): string {
  return Math.round(n).toLocaleString('en-US');
}

/** Format latency: >= 1s -> seconds, otherwise ms. No false precision. */
export function formatLatency(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
  if (ms >= 10) return `${Math.round(ms)} ms`;
  if (ms >= 1) return `${ms.toFixed(1)} ms`;
  return `${ms.toFixed(2)} ms`;
}

/** Format tokens with appropriate suffix */
export function formatTokensShort(tokens: number): string {
  if (tokens >= 1e12) return `${(tokens / 1e12).toFixed(1)}T`;
  if (tokens >= 1e9) return `${(tokens / 1e9).toFixed(0)}B`;
  if (tokens >= 1e6) return `${(tokens / 1e6).toFixed(0)}M`;
  return tokens.toString();
}
