/**
 * Speculative Decoding for LLM Inference
 *
 * Speculative decoding uses a small "draft" model to propose multiple tokens,
 * which the main "target" model verifies in a single forward pass.
 *
 * The key insight: verification can happen in parallel for K tokens,
 * while sequential generation would need K separate forward passes.
 *
 * Speedup depends on:
 * 1. How well the draft model predicts the target model (acceptance rate)
 * 2. How fast the draft model is compared to the target model
 * 3. Number of speculative tokens (K)
 */

import type { ModelSpec } from '../../types/model.ts';
import type { GPUSpec } from '../../types/hardware.ts';
import type {
  InferencePrecision,
  SpeculativeDecodingConfig,
  SpeculativeMetrics,
  SpeculativeTokenState,
} from '../../types/inference.ts';
import { estimateTPOT, prefillFLOPs, getGPUTFLOPS, moeWeightBytesPerStep, calculateLatencyWithTP, getBandwidthEfficiency, getPrefillEfficiency } from './latency.ts';
import { totalKVCacheMemory, getPrecisionBytes } from './kv-cache.ts';
import { getIntraNodeInterconnect, getCollectiveBandwidth } from '../hardware/interconnect.ts';

/**
 * Calculate expected number of accepted tokens
 *
 * With acceptance rate α and K speculative tokens:
 * Expected accepted = Σ(i=1 to K) α^i = α(1 - α^K) / (1 - α)
 *
 * Plus the guaranteed token from target model.
 */
export function expectedAcceptedTokens(
  numSpeculativeTokens: number,
  acceptanceRate: number
): number {
  const K = numSpeculativeTokens;
  const alpha = acceptanceRate;

  if (alpha >= 1.0) {
    return K; // All tokens accepted
  }

  if (alpha <= 0) {
    return 1; // No draft tokens accepted, but target still provides one token
  }

  // Geometric series: α + α² + α³ + ... + αᴷ = α(1 - αᴷ) / (1 - α)
  const expectedFromDraft = alpha * (1 - Math.pow(alpha, K)) / (1 - alpha);

  // Plus one guaranteed token from target (either accepted last draft or target's own token)
  return expectedFromDraft + 1;
}

/**
 * Calculate theoretical speedup from speculative decoding
 *
 * Speedup = expected_tokens_per_iteration / cost_of_iteration
 *
 * Cost includes:
 * 1. Draft model generating K tokens
 * 2. Target model verifying K tokens (single forward pass)
 */
export function theoreticalSpeedup(
  numSpeculativeTokens: number,
  acceptanceRate: number
): number {
  const expectedTokens = expectedAcceptedTokens(numSpeculativeTokens, acceptanceRate);

  // In the ideal case (draft is free), speedup equals expected tokens
  // In practice, draft model overhead reduces this
  return expectedTokens;
}

/**
 * Calculate draft model overhead
 * How much slower is generation due to draft model computation?
 * Uses activeParams for MoE models (compute scales with active, not total).
 */
export function draftModelOverhead(
  draftModel: ModelSpec,
  targetModel: ModelSpec
): number {
  const targetParams = targetModel.activeParams ?? targetModel.totalParams;
  const draftParams = draftModel.activeParams ?? draftModel.totalParams;
  return draftParams / targetParams;
}

/**
 * Estimate time for draft model to generate K tokens
 */
export function estimateDraftTime(
  draftModel: ModelSpec,
  numTokens: number,
  currentSeqLen: number,
  batchSize: number,
  gpu: GPUSpec,
  precision: InferencePrecision = 'bf16',
  kvCachePrecision?: InferencePrecision,
  tpDegree: number = 1,
  epDegree: number = 1,
  gpusPerNode?: number,
  interNodeBandwidthGBps?: number,
): number {
  const kvPrec = kvCachePrecision ?? precision;
  if (tpDegree > 1 || epDegree > 1) {
    // Use TP-aware TPOT for draft model running on same TP group
    const draftLatency = calculateLatencyWithTP(
      draftModel, currentSeqLen, numTokens, batchSize, gpu, tpDegree, precision, epDegree, kvPrec,
      true, gpusPerNode, interNodeBandwidthGBps,
    );
    // Draft generates tokens sequentially: K × per-token latency
    return draftLatency.tpot * numTokens;
  }

  // Single-GPU path: sequential generation
  let totalTime = 0;
  for (let i = 0; i < numTokens; i++) {
    const seqLen = currentSeqLen + i;
    totalTime += estimateTPOT(draftModel, seqLen, batchSize, gpu, precision, kvPrec);
  }
  return totalTime;
}

/**
 * Estimate time for target model to verify K tokens
 * Verification is a single forward pass with K tokens
 */
export function estimateVerificationTime(
  targetModel: ModelSpec,
  numTokensToVerify: number,
  currentSeqLen: number,
  batchSize: number,
  gpu: GPUSpec,
  precision: InferencePrecision = 'bf16',
  kvCachePrecision?: InferencePrecision,
  tpDegree: number = 1,
  _epDegree: number = 1,
  gpusPerNode?: number,
  interNodeBandwidthGBps?: number,
): number {
  const kvPrec = kvCachePrecision ?? precision;
  const tp = Math.max(1, tpDegree);

  // Verification is similar to prefill: process multiple tokens in parallel
  // EP has minimal impact on verification (small K-token forward pass)
  const flops = prefillFLOPs(targetModel, numTokensToVerify * batchSize);
  const gpuTFLOPS = getGPUTFLOPS(gpu, precision);
  const mfu = getPrefillEfficiency(gpu, precision);

  // TP shards compute across GPUs
  const computeTime = flops / (gpuTFLOPS * 1e12 * mfu * tp) * 1000;

  // Also need to read weights and KV cache
  // Verification processes multiple tokens — weight reads are batch-aware for MoE
  const weightsBytes = moeWeightBytesPerStep(targetModel, numTokensToVerify * batchSize, precision);
  // Per-GPU weights and KV cache with TP sharding
  const perGPUWeightsBytes = weightsBytes / tp;
  const kvBytes = totalKVCacheMemory(targetModel, currentSeqLen, batchSize, kvPrec) / tp;
  const bandwidth = gpu.memoryBandwidthTBps * 1e12;

  // Apply bandwidth efficiency (consistent with estimateTPOT)
  const bandwidthEfficiency = getBandwidthEfficiency(perGPUWeightsBytes + kvBytes);
  const memoryTime = (perGPUWeightsBytes + kvBytes) / (bandwidth * bandwidthEfficiency) * 1000;

  let commOverhead = 0;
  if (tp > 1) {
    // TP AllReduce overhead for verification (same as prefill AllReduce)
    const bytes = getPrecisionBytes(precision);
    const hiddenBytes = batchSize * numTokensToVerify * targetModel.hiddenSize * bytes;
    // Use hierarchical bandwidth for cross-node TP
    const crossesNodes = gpusPerNode !== undefined
      && interNodeBandwidthGBps !== undefined
      && tp > gpusPerNode
      && tp % gpusPerNode === 0;
    const ic = getIntraNodeInterconnect(gpu);
    const intraNodeBW = getCollectiveBandwidth(ic);
    let commBW: number;
    if (crossesNodes) {
      const G = gpusPerNode!;
      const N = Math.ceil(tp / G);
      commBW = ((tp - 1) / tp) / ((G - 1) / G / intraNodeBW + (N - 1) / N / interNodeBandwidthGBps!);
    } else {
      commBW = intraNodeBW;
    }
    const commBandwidthBps = commBW * 1e9;
    commOverhead = (hiddenBytes / commBandwidthBps) * 1000 * 2 * targetModel.numLayers;
  }

  return Math.max(computeTime, memoryTime) + commOverhead;
}

/**
 * Calculate effective TPOT with speculative decoding
 */
export function effectiveTPOTWithSpeculation(
  draftModel: ModelSpec,
  targetModel: ModelSpec,
  numSpeculativeTokens: number,
  acceptanceRate: number,
  currentSeqLen: number,
  batchSize: number,
  gpu: GPUSpec,
  precision: InferencePrecision = 'bf16',
  kvCachePrecision?: InferencePrecision,
  tpDegree: number = 1,
  epDegree: number = 1,
  gpusPerNode?: number,
  interNodeBandwidthGBps?: number,
): number {
  // Time for one speculation iteration
  const draftTime = estimateDraftTime(
    draftModel, numSpeculativeTokens, currentSeqLen, batchSize, gpu, precision, kvCachePrecision, tpDegree, epDegree,
    gpusPerNode, interNodeBandwidthGBps,
  );

  // Target verifies K+1 positions: K draft tokens + 1 bonus token
  const verifyTime = estimateVerificationTime(
    targetModel, numSpeculativeTokens + 1, currentSeqLen, batchSize, gpu, precision, kvCachePrecision, tpDegree, epDegree,
    gpusPerNode, interNodeBandwidthGBps,
  );

  const iterationTime = draftTime + verifyTime;

  // Expected tokens per iteration
  const expectedTokens = expectedAcceptedTokens(numSpeculativeTokens, acceptanceRate);

  // Effective TPOT
  return iterationTime / expectedTokens;
}

/**
 * Calculate full speculative decoding metrics
 */
export function calculateSpeculativeMetrics(
  config: SpeculativeDecodingConfig,
  targetModel: ModelSpec,
  currentSeqLen: number,
  batchSize: number,
  gpu: GPUSpec,
  precision: InferencePrecision = 'bf16',
  kvCachePrecision?: InferencePrecision,
  tpDegree: number = 1,
  epDegree: number = 1,
  baselineTPOTOverride?: number,
  gpusPerNode?: number,
  interNodeBandwidthGBps?: number,
): SpeculativeMetrics | null {
  if (!config.enabled || !config.draftModel) {
    return null;
  }

  const kvPrec = kvCachePrecision ?? precision;
  const { draftModel, numSpeculativeTokens, acceptanceRate } = config;

  // Draft time for K tokens (TP-aware when TP > 1)
  const draftModelOverhead = estimateDraftTime(
    draftModel, numSpeculativeTokens, currentSeqLen, batchSize, gpu, precision, kvCachePrecision, tpDegree, epDegree,
    gpusPerNode, interNodeBandwidthGBps,
  );

  // Verification time (TP-aware when TP > 1)
  // Target verifies K+1 positions: K draft tokens + 1 bonus token
  const verificationTime = estimateVerificationTime(
    targetModel, numSpeculativeTokens + 1, currentSeqLen, batchSize, gpu, precision, kvCachePrecision, tpDegree, epDegree,
    gpusPerNode, interNodeBandwidthGBps,
  );

  // Expected tokens
  const expectedTokens = expectedAcceptedTokens(numSpeculativeTokens, acceptanceRate);

  // Baseline TPOT for speedup calculation.
  // When overridden (e.g., CB-adjusted TPOT), use that so speedup reflects the actual alternative.
  let baselineTPOT: number;
  if (baselineTPOTOverride !== undefined) {
    baselineTPOT = baselineTPOTOverride;
  } else if (tpDegree > 1 || epDegree > 1) {
    const tpLatency = calculateLatencyWithTP(
      targetModel, currentSeqLen, 1, batchSize, gpu, tpDegree, precision, epDegree, kvPrec,
      true, gpusPerNode, interNodeBandwidthGBps,
    );
    baselineTPOT = tpLatency.tpot;
  } else {
    baselineTPOT = estimateTPOT(targetModel, currentSeqLen, batchSize, gpu, precision, kvPrec);
  }

  // Effective TPOT with speculation
  const effectiveTpot = (draftModelOverhead + verificationTime) / expectedTokens;

  // Speedup
  const speedup = baselineTPOT / effectiveTpot;

  return {
    expectedAcceptedTokens: expectedTokens,
    speedup,
    draftModelOverhead,
    verificationTime,
    effectiveTpot,
  };
}

/**
 * Simulate speculative decoding for visualization
 * Returns a sequence of speculation iterations
 */
export function simulateSpeculativeDecoding(
  config: SpeculativeDecodingConfig,
  targetModel: ModelSpec,
  inputSeqLen: number,
  outputSeqLen: number,
  batchSize: number,
  gpu: GPUSpec,
  precision: InferencePrecision = 'bf16',
  kvCachePrecision?: InferencePrecision,
): {
  iterations: SpeculativeIteration[];
  totalTokens: number;
  totalTime: number;
  averageSpeedup: number;
} {
  if (!config.enabled || !config.draftModel) {
    return {
      iterations: [],
      totalTokens: 0,
      totalTime: 0,
      averageSpeedup: 1.0,
    };
  }

  const iterations: SpeculativeIteration[] = [];
  let currentSeqLen = inputSeqLen;
  let totalTokens = 0;
  let totalTime = 0;

  // Random but deterministic acceptance pattern based on acceptance rate
  const rng = createSeededRandom(42);

  while (totalTokens < outputSeqLen) {
    const iteration = simulateIteration(
      config,
      targetModel,
      currentSeqLen,
      batchSize,
      gpu,
      precision,
      kvCachePrecision,
      rng,
      outputSeqLen - totalTokens
    );

    iterations.push(iteration);
    totalTokens += iteration.acceptedCount;
    totalTime += iteration.draftTime + iteration.verifyTime;
    currentSeqLen += iteration.acceptedCount;
  }

  // Calculate baseline time for comparison
  const kvPrec = kvCachePrecision ?? precision;
  const baselineTime = estimateTPOT(
    targetModel, inputSeqLen + outputSeqLen / 2, batchSize, gpu, precision, kvPrec
  ) * outputSeqLen;

  const averageSpeedup = baselineTime / totalTime;

  return {
    iterations,
    totalTokens,
    totalTime,
    averageSpeedup,
  };
}

/**
 * Single speculation iteration
 */
export interface SpeculativeIteration {
  draftTokens: SpeculativeTokenState[];
  acceptedCount: number;
  draftTime: number;
  verifyTime: number;
}

/**
 * Simulate a single speculation iteration
 */
function simulateIteration(
  config: SpeculativeDecodingConfig,
  targetModel: ModelSpec,
  currentSeqLen: number,
  batchSize: number,
  gpu: GPUSpec,
  precision: InferencePrecision,
  kvCachePrecision: InferencePrecision | undefined,
  rng: () => number,
  remainingTokens: number
): SpeculativeIteration {
  const { draftModel, numSpeculativeTokens, acceptanceRate } = config;

  // Don't speculate more than needed
  const K = Math.min(numSpeculativeTokens, remainingTokens);

  // Simulate draft tokens
  const draftTokens: SpeculativeTokenState[] = [];
  let acceptedCount = 0;

  for (let i = 0; i < K; i++) {
    const isAccepted = rng() < acceptanceRate;

    if (isAccepted && acceptedCount === i) {
      // Token accepted (all previous tokens were also accepted)
      acceptedCount++;
      draftTokens.push({
        tokenId: i,
        token: `token_${i}`,
        position: currentSeqLen + i,
        status: 'accepted',
        draftProbability: 0.8 + rng() * 0.15,
        targetProbability: 0.7 + rng() * 0.25,
      });
    } else {
      // Token rejected or previous token was rejected
      draftTokens.push({
        tokenId: i,
        token: `token_${i}`,
        position: currentSeqLen + i,
        status: i < acceptedCount ? 'accepted' : 'rejected',
        draftProbability: 0.3 + rng() * 0.4,
        targetProbability: 0.1 + rng() * 0.3,
      });
    }
  }

  // Target always provides one bonus token (Leviathan et al. protocol)
  acceptedCount++;

  // Calculate times
  const draftTime = estimateDraftTime(
    draftModel!, K, currentSeqLen, batchSize, gpu, precision, kvCachePrecision
  );

  // Target verifies K+1 positions: K draft tokens + 1 bonus token
  const verifyTime = estimateVerificationTime(
    targetModel, K + 1, currentSeqLen, batchSize, gpu, precision, kvCachePrecision
  );

  return {
    draftTokens,
    acceptedCount,
    draftTime,
    verifyTime,
  };
}

/**
 * Create a seeded random number generator for reproducible results
 */
function createSeededRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  };
}

/**
 * Find optimal speculative decoding configuration
 */
export function findOptimalSpecConfig(
  draftModel: ModelSpec,
  targetModel: ModelSpec,
  estimatedAcceptanceRate: number,
  currentSeqLen: number,
  batchSize: number,
  gpu: GPUSpec,
  precision: InferencePrecision = 'bf16',
  kvCachePrecision?: InferencePrecision,
  tpDegree: number = 1,
  epDegree: number = 1,
  gpusPerNode?: number,
  interNodeBandwidthGBps?: number,
): {
  optimalK: number;
  expectedSpeedup: number;
  breakdown: { k: number; speedup: number; effectiveTpot: number }[];
} {
  const breakdown: { k: number; speedup: number; effectiveTpot: number }[] = [];
  let maxSpeedup = 1.0;
  let optimalK = 1;

  const kvPrec = kvCachePrecision ?? precision;
  let baselineTPOT: number;
  if (tpDegree > 1 || epDegree > 1) {
    baselineTPOT = calculateLatencyWithTP(
      targetModel, currentSeqLen, 1, batchSize, gpu, tpDegree, precision, epDegree, kvPrec,
      true, gpusPerNode, interNodeBandwidthGBps,
    ).tpot;
  } else {
    baselineTPOT = estimateTPOT(targetModel, currentSeqLen, batchSize, gpu, precision, kvPrec);
  }

  for (let k = 1; k <= 16; k++) {
    const effectiveTpot = effectiveTPOTWithSpeculation(
      draftModel, targetModel, k, estimatedAcceptanceRate,
      currentSeqLen, batchSize, gpu, precision, kvCachePrecision, tpDegree, epDegree,
      gpusPerNode, interNodeBandwidthGBps,
    );

    const speedup = baselineTPOT / effectiveTpot;

    breakdown.push({ k, speedup, effectiveTpot });

    if (speedup > maxSpeedup) {
      maxSpeedup = speedup;
      optimalK = k;
    }
  }

  return {
    optimalK,
    expectedSpeedup: maxSpeedup,
    breakdown,
  };
}

/**
 * Estimate acceptance rate based on model similarity
 * This is a heuristic based on parameter ratio and family
 */
export function estimateAcceptanceRate(
  draftModel: ModelSpec,
  targetModel: ModelSpec
): number {
  // Parameter ratio (larger draft = higher acceptance)
  // Use activeParams for MoE — acceptance scales with computation, not storage
  const targetParams = targetModel.activeParams ?? targetModel.totalParams;
  const draftParams = draftModel.activeParams ?? draftModel.totalParams;
  const paramRatio = draftParams / targetParams;

  // Base acceptance rate from parameter ratio
  // Empirically, ~70% acceptance for 1/10th size draft, ~90% for 1/3rd size
  let baseAcceptance = 0.5 + 0.4 * Math.sqrt(paramRatio);

  // Same family bonus
  if (draftModel.family === targetModel.family) {
    baseAcceptance += 0.05;
  }

  // Same architecture bonus
  if (draftModel.numAttentionHeads === targetModel.numAttentionHeads &&
      draftModel.hiddenSize === targetModel.hiddenSize) {
    baseAcceptance += 0.05;
  }

  // Clamp to valid range
  return Math.max(0.3, Math.min(0.95, baseAcceptance));
}

/**
 * Check if speculative decoding would be beneficial
 */
export function isSpeculativeDecodingBeneficial(
  draftModel: ModelSpec,
  targetModel: ModelSpec,
  numSpeculativeTokens: number,
  acceptanceRate: number,
  currentSeqLen: number,
  batchSize: number,
  gpu: GPUSpec,
  precision: InferencePrecision = 'bf16',
  kvCachePrecision?: InferencePrecision,
  tpDegree: number = 1,
  epDegree: number = 1,
  gpusPerNode?: number,
  interNodeBandwidthGBps?: number,
): {
  beneficial: boolean;
  speedup: number;
  reason: string;
} {
  const metrics = calculateSpeculativeMetrics(
    {
      enabled: true,
      draftModel,
      numSpeculativeTokens,
      acceptanceRate,
    },
    targetModel,
    currentSeqLen,
    batchSize,
    gpu,
    precision,
    kvCachePrecision,
    tpDegree,
    epDegree,
    undefined,
    gpusPerNode,
    interNodeBandwidthGBps,
  );

  if (!metrics) {
    return {
      beneficial: false,
      speedup: 1.0,
      reason: 'Speculative decoding not configured',
    };
  }

  if (metrics.speedup < 1.0) {
    return {
      beneficial: false,
      speedup: metrics.speedup,
      reason: `Draft model overhead (${metrics.draftModelOverhead.toFixed(1)}ms) exceeds benefit`,
    };
  }

  if (metrics.speedup < 1.2) {
    return {
      beneficial: false,
      speedup: metrics.speedup,
      reason: 'Speedup too small to justify complexity',
    };
  }

  return {
    beneficial: true,
    speedup: metrics.speedup,
    reason: `Expected ${metrics.speedup.toFixed(2)}x speedup with ${(acceptanceRate * 100).toFixed(0)}% acceptance rate`,
  };
}
