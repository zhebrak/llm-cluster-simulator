/**
 * Derived physics quantities for replacing empirical constants.
 *
 * Standalone functions that compute physically-grounded values as alternatives
 * to hardcoded constants (e.g. loadImbalance log curve, CP causal work
 * distribution, reserved memory 1.5 GB + 12%).
 */

import type { GPUArchitecture } from '../../types/hardware.ts';

// ---------------------------------------------------------------------------
// MoE Load Imbalance (multinomial order statistics)
// ---------------------------------------------------------------------------

/**
 * Derive expert load imbalance from routing distribution using multinomial
 * order statistics. Replaces the fitted `1 + 0.05 * log2(max(E,8)/8)`.
 *
 * Model: each token independently routes to topK of numExperts experts.
 * Per-expert token count follows Binomial(N, p) where p = topK/numExperts
 * and N = tokensPerMicrobatch. The max-loaded expert determines wall-clock
 * time. By extreme value theory for the max of numExperts iid draws:
 *
 *   E[max] ≈ μ + σ × √(2 ln(numExperts))
 *
 * where σ = √((1-p) / tokensPerExpert).
 *
 * The returned factor is multiplicative: computeTime × loadImbalanceFactor.
 *
 * NOTE: Raw multinomial prediction with no damping — conservative
 * (overestimates imbalance). Damping factor should be calibrated against
 * benchmark data before use.
 *
 * IMPORTANT: The caller must EP-adjust tokensPerMicrobatch. With EP, each
 * expert group sees tokensPerMicrobatch / EP tokens for load variance.
 * Pass the EP-adjusted count, not the raw microbatch token count.
 *
 * @param numExperts - total number of routed experts
 * @param topK - number of experts activated per token
 * @param tokensPerMicrobatch - tokens arriving at this EP partition (EP-adjusted)
 * @returns multiplicative load imbalance factor (>= 1.0)
 */
export function loadImbalanceFactor(
  numExperts: number,
  topK: number,
  tokensPerMicrobatch: number,
): number {
  if (numExperts <= 1 || tokensPerMicrobatch <= 0) return 1.0;

  const p = topK / numExperts;
  const tokensPerExpert = tokensPerMicrobatch * topK / numExperts;

  if (tokensPerExpert <= 0 || p <= 0 || p >= 1) return 1.0;

  const sigma = Math.sqrt((1 - p) / tokensPerExpert);
  return 1 + sigma * Math.sqrt(2 * Math.log(numExperts));
}

// ---------------------------------------------------------------------------
// 2. CP Causal Work Distribution
// ---------------------------------------------------------------------------

/**
 * Structural description of ring attention compute under causal masking.
 * NOT collapsed to a single penalty scalar — the physics is per-step.
 */
export interface CPCausalWorkDistribution {
  /** Compute fraction for the diagonal (self-attention) chunk: 0.5 due to triangular mask */
  diagonalComputeFraction: number;
  /** Number of diagonal steps per ring iteration (always 1) */
  diagonalSteps: number;
  /** Number of full cross-attention steps (CP - 2) */
  normalSteps: number;
}

/**
 * Derive per-step compute model for ring attention with causal masking.
 *
 * Ring attention performs CP-1 receive-compute-send steps per rank:
 *   - (CP-2) steps have full cross-attention chunks: computeTime = fullChunkCompute
 *   - 1 step has the diagonal (self-attention) chunk: computeTime = 0.5 × fullChunkCompute
 *
 * Per-step exposed transfer model:
 *   exposedDiagonal = max(0, transferTime - 0.5 × fullCompute)
 *   exposedNormal = max(0, transferTime - fullCompute)
 *   totalExposed = exposedDiagonal + (CP-2) × exposedNormal
 *
 * @param cp - context parallelism degree (must be >= 2)
 * @returns structural work distribution
 */
export function cpCausalWorkDistribution(cp: number): CPCausalWorkDistribution {
  if (cp < 2) {
    return { diagonalComputeFraction: 1.0, diagonalSteps: 0, normalSteps: 0 };
  }
  return {
    diagonalComputeFraction: 0.5,
    diagonalSteps: 1,
    normalSteps: cp - 2,
  };
}

// ---------------------------------------------------------------------------
// 3. Reserved Memory Decomposition
// ---------------------------------------------------------------------------

/**
 * Physical decomposition of reserved GPU memory into two components:
 *   1. CUDA context — per-GPU-family context + cuDNN workspace
 *   2. Fragmentation — PyTorch caching allocator fragmentation (7% of GiB capacity)
 */
export interface ReservedMemoryDecomposition {
  /** CUDA context + cuDNN workspace (architecture-dependent, 0.8–1.4 GB) */
  cudaContextBytes: number;
  /** PyTorch caching allocator fragmentation (7% of physical GiB capacity) */
  fragmentationBytes: number;
  /** Total reserved = cudaContextBytes + fragmentationBytes */
  totalBytes: number;
}

/**
 * CUDA context size lookup by GPU architecture family.
 *
 * Estimated from typical `torch.cuda.mem_get_info()` deltas on empty context.
 * Assumes CUDA 12.4 toolkit. CUDA context size varies with toolkit version
 * and driver — these are estimates, not precisely measured per-GPU.
 */
const CUDA_CONTEXT_GB: Record<string, number> = {
  volta: 0.8,
  turing: 0.8,
  ampere: 1.0,
  ada: 1.0,
  hopper: 1.2,
  cdna3: 1.2,
  blackwell: 1.4,
  cdna4: 1.4,
  // cdna2 not in this list; falls back to default
};

const DEFAULT_CUDA_CONTEXT_GB = 1.0;

/**
 * Decompose reserved GPU memory into physically meaningful components.
 *
 * Components:
 *   1. CUDA context — per-GPU-family context + cuDNN workspace (0.8–1.4 GB)
 *   2. Fragmentation — PyTorch caching allocator fragmentation (7% of GiB capacity)
 *
 * @param gpuMemoryGB - GPU memory in GiB (vendor-marketed, e.g. 80 for A100 "80 GB")
 * @param architecture - GPU architecture family
 * @returns decomposition with individual components and total
 */
export function reservedMemoryDecomposition(
  gpuMemoryGB: number,
  architecture: GPUArchitecture,
): ReservedMemoryDecomposition {
  // 1. CUDA context (architecture-dependent)
  const cudaContextGB = CUDA_CONTEXT_GB[architecture] ?? DEFAULT_CUDA_CONTEXT_GB;
  const cudaContextBytes = cudaContextGB * 1e9;

  // 2. Fragmentation: 7% of physical (GiB) capacity
  const physicalBytes = gpuMemoryGB * (1024 ** 3);
  const fragmentationBytes = physicalBytes * 0.07;

  const totalBytes = cudaContextBytes + fragmentationBytes;

  return {
    cudaContextBytes,
    fragmentationBytes,
    totalBytes,
  };
}
