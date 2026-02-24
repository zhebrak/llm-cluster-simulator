/**
 * Contract tests for the physics-based overlap model.
 *
 * Each test feeds representative inputs and asserts the output matches a
 * hand-computed expected value. These lock the formula — if the formula
 * changes, the contract test must be updated in the same commit.
 */
import { describe, it, expect } from 'vitest';
import {
  computeFSDPExposedComm,
  computeDDPOverlap,
  computeZeROGradOverlap,
  computeTPOverlap,
  computePPOverlap,
  applyProtocolOverhead,
  withOverlapOverrides,
  getOverlapConstants,
  SCHEDULING_EFFICIENCY,
  BUCKET_SIZE_BYTES,
  PER_COLLECTIVE_OVERHEAD_MS,
  PROTOCOL_OVERHEAD,
} from '../../src/core/strategies/overlap.ts';

const η = 0.96; // must match SCHEDULING_EFFICIENCY

// ── Constant values ──

describe('Overlap constants', () => {
  it('exported constants match documented values', () => {
    expect(SCHEDULING_EFFICIENCY).toBe(η);
    expect(BUCKET_SIZE_BYTES).toBe(25e6);
    expect(PER_COLLECTIVE_OVERHEAD_MS).toBe(0.050);
  });

  it('protocol overhead constants match documented values', () => {
    expect(PROTOCOL_OVERHEAD.tp_nvlink).toBe(0.05);
    expect(PROTOCOL_OVERHEAD.tp_crossnode).toBe(0.15);
    expect(PROTOCOL_OVERHEAD.dp_fsdp).toBe(0.10);
    expect(PROTOCOL_OVERHEAD.dp_ddp).toBe(0.20);
    expect(PROTOCOL_OVERHEAD.dp_ep).toBe(0.10);
    expect(PROTOCOL_OVERHEAD.pp).toBe(0.05);
  });
});

// ── FSDP per-layer pipeline model ──

describe('computeFSDPExposedComm', () => {
  // η values hardcoded in overlap.ts
  const ηPrefetch = 0.95;
  const ηNoPrefetch = 0.80;

  it('compute-bound uniform: exposed = 2×AG + RS', () => {
    // When per-layer compute >> AG and RS, all max(0,...) terms vanish.
    // Exposed = AG (fwd cold) + AG (bwd cold) + RS (bwd cold) = 2*AG + RS
    const result = computeFSDPExposedComm({
      fwdComputePerLayer: [50],
      bwdComputePerLayer: [100],
      allGatherPerLayer: 1,
      reduceScatterPerLayer: 0.5,
      numLayers: 32,
      backwardPrefetch: true,
    });
    expect(result).toBeCloseTo(2 * 1 + 0.5, 10);  // 2.5 ms
  });

  it('comm-bound uniform: per-layer excess accumulates', () => {
    // fwdCompute = 0.5, bwdCompute = 1.0, AG = 5, RS = 3, L = 4
    // η = 0.95
    // Fwd: AG + 3 * max(0, 5 - 0.5*0.95) = 5 + 3*(5-0.475) = 5 + 3*4.525 = 18.575
    // Bwd AG: AG + 3 * max(0, 5 - 1.0*0.95) = 5 + 3*(5-0.95) = 5 + 3*4.05 = 17.15
    // Bwd RS: RS + 3 * max(0, 3 - 1.0*0.95) = 3 + 3*(3-0.95) = 3 + 3*2.05 = 9.15
    // Total = 18.575 + 17.15 + 9.15 = 44.875
    const result = computeFSDPExposedComm({
      fwdComputePerLayer: [0.5],
      bwdComputePerLayer: [1.0],
      allGatherPerLayer: 5,
      reduceScatterPerLayer: 3,
      numLayers: 4,
      backwardPrefetch: true,
    });
    expect(result).toBeCloseTo(44.875, 10);
  });

  it('single layer: all cold start, no overlap possible', () => {
    const result = computeFSDPExposedComm({
      fwdComputePerLayer: [50],
      bwdComputePerLayer: [100],
      allGatherPerLayer: 2,
      reduceScatterPerLayer: 1,
      numLayers: 1,
      backwardPrefetch: true,
    });
    // 2*AG + RS = 4 + 1 = 5
    expect(result).toBeCloseTo(5, 10);
  });

  it('without prefetch: lower η → more exposed comm', () => {
    // borderline case: compute just barely hides AG with η=0.95, but not with η=0.80
    // fwdCompute = 5.5, AG = 5, L = 2
    // With prefetch (η=0.95): max(0, 5 - 5.5*0.95) = max(0, 5-5.225) = 0
    // Without prefetch (η=0.80): max(0, 5 - 5.5*0.80) = max(0, 5-4.4) = 0.6
    const withPrefetch = computeFSDPExposedComm({
      fwdComputePerLayer: [5.5],
      bwdComputePerLayer: [11],
      allGatherPerLayer: 5,
      reduceScatterPerLayer: 2.5,
      numLayers: 2,
      backwardPrefetch: true,
    });
    const withoutPrefetch = computeFSDPExposedComm({
      fwdComputePerLayer: [5.5],
      bwdComputePerLayer: [11],
      allGatherPerLayer: 5,
      reduceScatterPerLayer: 2.5,
      numLayers: 2,
      backwardPrefetch: false,
    });
    expect(withoutPrefetch).toBeGreaterThan(withPrefetch);
  });

  it('heterogeneous layers: MoE dense vs expert layers', () => {
    // Simulates a PP stage with 2 dense + 4 MoE layers
    // Dense fwd = 2ms (borderline), MoE fwd = 8ms (deeply compute-bound)
    // AG = 1.5, RS = 0.8
    const fwd = [2, 2, 8, 8, 8, 8];
    const bwd = [4, 4, 16, 16, 16, 16];
    const result = computeFSDPExposedComm({
      fwdComputePerLayer: fwd,
      bwdComputePerLayer: bwd,
      allGatherPerLayer: 1.5,
      reduceScatterPerLayer: 0.8,
      numLayers: 6,
      backwardPrefetch: true,
    });
    // Fwd: AG=1.5 (cold) + max(0,1.5-2*0.95)=0 + max(0,1.5-2*0.95)=0
    //   + max(0,1.5-8*0.95)=0 + max(0,1.5-8*0.95)=0 + max(0,1.5-8*0.95)=0 = 1.5
    // Bwd AG: AG=1.5 (cold) + sum of max(0, 1.5 - bwd[i+1]*0.95) for i=0..4
    //   i=0: max(0,1.5-4*0.95)=max(0,1.5-3.8)=0
    //   i=1: max(0,1.5-16*0.95)=0, i=2..4: all 0 → bwdAG = 1.5
    // Bwd RS: RS=0.8 (cold) + sum of max(0, 0.8 - bwd[i-1]*0.95) for i=1..5
    //   all 0 since bwd >= 4 and 4*0.95 > 0.8 → bwdRS = 0.8
    // Total = 1.5 + 1.5 + 0.8 = 3.8
    expect(result).toBeCloseTo(2 * 1.5 + 0.8, 10);
  });

  it('closed-form matches loop for uniform input', () => {
    const AG = 2, RS = 1, fwd = 3, bwd = 6, L = 64;
    const uniform = computeFSDPExposedComm({
      fwdComputePerLayer: [fwd],
      bwdComputePerLayer: [bwd],
      allGatherPerLayer: AG,
      reduceScatterPerLayer: RS,
      numLayers: L,
      backwardPrefetch: true,
    });
    // Explicit per-layer arrays
    const fwdArr = Array(L).fill(fwd);
    const bwdArr = Array(L).fill(bwd);
    const perLayer = computeFSDPExposedComm({
      fwdComputePerLayer: fwdArr,
      bwdComputePerLayer: bwdArr,
      allGatherPerLayer: AG,
      reduceScatterPerLayer: RS,
      numLayers: L,
      backwardPrefetch: true,
    });
    expect(uniform).toBeCloseTo(perLayer, 10);
  });

  it('zero layers returns 0', () => {
    const result = computeFSDPExposedComm({
      fwdComputePerLayer: [50],
      bwdComputePerLayer: [100],
      allGatherPerLayer: 1,
      reduceScatterPerLayer: 0.5,
      numLayers: 0,
      backwardPrefetch: true,
    });
    expect(result).toBe(0);
  });
});

// ── DDP overlap ──

describe('computeDDPOverlap', () => {
  it('bucketed overlap with large gradient (tail drain)', () => {
    // gradientBytes = 1e9, BUCKET_SIZE = 25e6
    // numBuckets = ceil(1e9 / 25e6) = 40
    // overlappableFraction = 39/40 = 0.975 (last bucket exposed)
    // theoretical = 0.975 * min(1.0, 200/100) = 0.975
    // overlapTime = 100 * 0.975 * 0.96 = 93.6
    const result = computeDDPOverlap({
      commTime: 100,
      backwardTime: 200,
      gradientBytes: 1e9,
    });
    expect(result).toBeCloseTo(100 * (39 / 40) * η, 10);
  });

  it('comm-dominated: backward shorter than comm', () => {
    // numBuckets = 40, overlappableFraction = 0.975
    // theoretical = 0.975 * min(1.0, 50/100) = 0.975 * 0.5 = 0.4875
    // overlapTime = 100 * 0.4875 * 0.96 = 46.8
    const result = computeDDPOverlap({
      commTime: 100,
      backwardTime: 50,
      gradientBytes: 1e9,
    });
    expect(result).toBeCloseTo(100 * (39 / 40) * 0.5 * η, 10);
  });

  it('small model: few buckets → significant tail drain', () => {
    // gradientBytes = 50e6, numBuckets = ceil(50e6/25e6) = 2
    // overlappableFraction = 1/2 = 0.5 (half the comm is tail-exposed)
    // theoretical = 0.5 * min(1.0, 200/100) = 0.5
    // overlapTime = 100 * 0.5 * 0.96 = 48
    const result = computeDDPOverlap({
      commTime: 100,
      backwardTime: 200,
      gradientBytes: 50e6,
    });
    expect(result).toBeCloseTo(100 * 0.5 * η, 10);
  });

  it('zero comm time returns zero overlap', () => {
    const result = computeDDPOverlap({
      commTime: 0,
      backwardTime: 200,
      gradientBytes: 1e9,
    });
    expect(result).toBe(0);
  });
});

// ── ZeRO-1/2 overlap ──

describe('computeZeROGradOverlap', () => {
  it('ZeRO-1 with overlap: bucket model with tail drain', () => {
    // gradientBytes = 1e9, numBuckets = 40
    // overlappableFraction = 39/40 = 0.975
    // theoretical = 0.975 * min(1.0, 200/100) = 0.975
    // overlap = 100 * 0.975 * 0.96 = 93.6
    const result = computeZeROGradOverlap({
      stage: 1, overlapComm: true,
      gradSyncTime: 100, backwardTime: 200, gradientBytes: 1e9,
    });
    expect(result).toBeCloseTo(100 * (39 / 40) * η, 10);
  });

  it('ZeRO-2 with overlap: same bucket model with tail drain', () => {
    const result = computeZeROGradOverlap({
      stage: 2, overlapComm: true,
      gradSyncTime: 100, backwardTime: 200, gradientBytes: 1e9,
    });
    expect(result).toBeCloseTo(100 * (39 / 40) * η, 10);
  });

  it('overlap disabled returns 0', () => {
    const result = computeZeROGradOverlap({
      stage: 1, overlapComm: false,
      gradSyncTime: 100, backwardTime: 200, gradientBytes: 1e9,
    });
    expect(result).toBe(0);
  });
});

// ── TP overlap ──

describe('computeTPOverlap', () => {
  it('C/(C+T) × η', () => {
    // C=100, T=5, C/(C+T) = 100/105 ≈ 0.9524
    // overlap = 0.9524 * 0.96 ≈ 0.9143
    const result = computeTPOverlap({
      computePerMB: 100,
      tpCommWithOverhead: 5,
    });
    expect(result).toBeCloseTo((100 / 105) * η, 10);
  });

  it('large comm: lower overlap', () => {
    // C=80, T=50, C/(C+T) = 80/130 ≈ 0.6154
    // overlap = 0.6154 * 0.96 ≈ 0.5908
    const result = computeTPOverlap({
      computePerMB: 80,
      tpCommWithOverhead: 50,
    });
    expect(result).toBeCloseTo((80 / 130) * η, 10);
  });
});

// ── PP overlap ──

describe('computePPOverlap', () => {
  it('C/(C+T) × η', () => {
    // C=100, T=1, C/(C+T) = 100/101 ≈ 0.9901
    // overlap = 0.9901 * 0.96 ≈ 0.9505
    const result = computePPOverlap({ computePerMB: 100, ppCommPerMB: 1 });
    expect(result).toBeCloseTo((100 / 101) * η, 10);
  });

  it('equal compute and comm', () => {
    // C/(C+T) = 0.50, overlap = 0.50 * 0.96 = 0.48
    const result = computePPOverlap({ computePerMB: 50, ppCommPerMB: 50 });
    expect(result).toBeCloseTo(0.50 * η, 10);
  });
});

// ── Protocol overhead (two-term model) ──

describe('applyProtocolOverhead', () => {
  it('proportional-only when numCollectives=0 (backward compat)', () => {
    expect(applyProtocolOverhead(100, 'tp_nvlink')).toBeCloseTo(105, 10);
    expect(applyProtocolOverhead(100, 'tp_crossnode')).toBeCloseTo(115, 10);
    expect(applyProtocolOverhead(100, 'dp_fsdp')).toBeCloseTo(110, 10);
    expect(applyProtocolOverhead(100, 'dp_ddp')).toBeCloseTo(120, 10);
    expect(applyProtocolOverhead(100, 'dp_ep')).toBeCloseTo(110, 10);
    expect(applyProtocolOverhead(100, 'pp')).toBeCloseTo(105, 10);
  });

  it('adds per-collective overhead for cross-node (IB)', () => {
    // raw=100, type=dp_ddp (20% proportional), 10 collectives, cross-node
    // = 100 * 1.20 + 10 * 0.050 = 120 + 0.50 = 120.50
    expect(applyProtocolOverhead(100, 'dp_ddp', 10, false)).toBeCloseTo(120.50, 10);
  });

  it('adds 10× lower per-collective overhead for intra-node (NVLink)', () => {
    // raw=100, type=dp_ddp (20% proportional), 10 collectives, intra-node
    // = 100 * 1.20 + 10 * 0.005 = 120 + 0.05 = 120.05
    expect(applyProtocolOverhead(100, 'dp_ddp', 10, true)).toBeCloseTo(120.05, 10);
  });

  it('returns 0 when raw is 0 (single GPU / no comm)', () => {
    // Even with numCollectives > 0, if raw=0 there's no communication
    expect(applyProtocolOverhead(0, 'dp_ddp', 10, false)).toBe(0);
  });

  it('per-collective dominates for small messages', () => {
    // raw=0.1ms, type=dp_fsdp (10% proportional), 96 collectives, cross-node
    // proportional = 0.1 * 0.10 = 0.01
    // fixed = 96 * 0.050 = 4.80
    // total = 0.1 + 0.01 + 4.80 = 4.91
    const result = applyProtocolOverhead(0.1, 'dp_fsdp', 96, false);
    expect(result).toBeCloseTo(4.91, 10);
  });
});

// ── Scoped override API ──

describe('withOverlapOverrides', () => {
  it('temporarily overrides constants and restores them', () => {
    const before = getOverlapConstants().schedulingEfficiency;
    expect(before).toBe(η);

    const inside = withOverlapOverrides({ schedulingEfficiency: 0.99 }, () => {
      return getOverlapConstants().schedulingEfficiency;
    });
    expect(inside).toBe(0.99);

    const after = getOverlapConstants().schedulingEfficiency;
    expect(after).toBe(η);
  });

  it('restores on throw', () => {
    try {
      withOverlapOverrides({ schedulingEfficiency: 0.99 }, () => {
        throw new Error('test');
      });
    } catch { /* expected */ }

    expect(getOverlapConstants().schedulingEfficiency).toBe(η);
  });

  it('affects computation functions', () => {
    // With η=0.96: C/(C+T) = 0.5 → overlap = 0.5 * 0.96 = 0.48
    const normal = computeTPOverlap({ computePerMB: 50, tpCommWithOverhead: 50 });
    expect(normal).toBeCloseTo(0.48, 10);

    // With η=1.0: overlap = 0.5 * 1.0 = 0.50
    const overridden = withOverlapOverrides({ schedulingEfficiency: 1.0 }, () => {
      return computeTPOverlap({ computePerMB: 50, tpCommWithOverhead: 50 });
    });
    expect(overridden).toBeCloseTo(0.50, 10);
  });

  it('overrides protocol overhead', () => {
    const normal = applyProtocolOverhead(100, 'dp_fsdp');
    expect(normal).toBeCloseTo(110, 10);

    const overridden = withOverlapOverrides(
      { protocolOverhead: { ...PROTOCOL_OVERHEAD, dp_fsdp: 0.20 } },
      () => applyProtocolOverhead(100, 'dp_fsdp'),
    );
    expect(overridden).toBeCloseTo(120, 10);
  });

  it('overrides perCollectiveOverheadMs', () => {
    // Default: 10 collectives × 0.050 = 0.50
    const normal = applyProtocolOverhead(100, 'dp_ddp', 10, false);
    expect(normal).toBeCloseTo(120.50, 10);

    // Override to 0.10: 10 × 0.10 = 1.0
    const overridden = withOverlapOverrides(
      { perCollectiveOverheadMs: 0.10 },
      () => applyProtocolOverhead(100, 'dp_ddp', 10, false),
    );
    expect(overridden).toBeCloseTo(121.0, 10);
  });

  it('supports nesting', () => {
    const result = withOverlapOverrides({ schedulingEfficiency: 0.90 }, () => {
      const inner = withOverlapOverrides({ schedulingEfficiency: 0.50 }, () => {
        return computeTPOverlap({ computePerMB: 50, tpCommWithOverhead: 50 });
      });
      // After inner scope, should be back to 0.90
      return [inner, computeTPOverlap({ computePerMB: 50, tpCommWithOverhead: 50 })];
    });
    // inner: 0.5 * 0.50 = 0.25
    expect(result[0]).toBeCloseTo(0.25, 10);
    // outer: 0.5 * 0.90 = 0.45
    expect(result[1]).toBeCloseTo(0.45, 10);

    // After outer scope, back to default
    expect(computeTPOverlap({ computePerMB: 50, tpCommWithOverhead: 50 })).toBeCloseTo(0.48, 10);
  });

  it('overrides bucket size affects DDP overlap', () => {
    // Default BUCKET_SIZE = 25e6, gradientBytes = 100e6
    // numBuckets = ceil(100e6/25e6) = 4, overlappableFraction = 3/4 = 0.75
    // theoretical = 0.75 * min(1.0, 200/100) = 0.75
    // overlap = 100 * 0.75 * 0.96 = 72
    const normal = computeDDPOverlap({ commTime: 100, backwardTime: 200, gradientBytes: 100e6 });
    expect(normal).toBeCloseTo(100 * 0.75 * η, 10);

    // With smaller buckets: numBuckets = 100, overlappableFraction = 99/100
    const overridden = withOverlapOverrides({ bucketSizeBytes: 1e6 }, () => {
      return computeDDPOverlap({ commTime: 100, backwardTime: 200, gradientBytes: 100e6 });
    });
    // numBuckets = 100, overlappableFraction = 0.99
    // theoretical = 0.99 * 1.0 = 0.99
    // overlap = 100 * 0.99 * 0.96 = 95.04 (more buckets = less tail drain)
    expect(overridden).toBeCloseTo(100 * 0.99 * η, 10);
  });
});
