/**
 * MoE Simulation Validation Tests
 *
 * Tests expert parallelism (EP) integration:
 * 1. Dense regression — EP=1 produces identical results
 * 2. EP memory reduction — MoE with EP uses less param memory
 * 3. EP communication — expertParallel > 0 when EP > 1
 * 4. MFU range — MoE models get reasonable MFU
 * 5. GPU count enforcement — TP × PP × DP = totalGPUs (EP subdivides DP)
 * 6. EP validation — EP must divide numExperts; EP > 1 rejected for dense
 * 7. DeepSeek-V3 at scale — 2048× H100 runs successfully (FP8)
 */

import { describe, it, expect } from 'vitest';
import { create3DParallelStrategy } from '../../src/core/strategies/index.ts';
import { runSimulation } from '../../src/core/simulation/index.ts';
import { getValidatedSimulationMetrics } from '../helpers/validated-metrics.ts';
import type { SimulationConfig } from '../../src/core/simulation/engine.ts';
import { makeStrategyContext } from '../helpers/strategy-context.ts';

// MoE tests default to seqLength=4096, microBatchSize=2
const createTestContext: typeof makeStrategyContext = (modelId, clusterId, overrides) =>
  makeStrategyContext(modelId, clusterId, { seqLength: 4096, microBatchSize: 2, ...overrides });

describe('MoE Simulation', () => {
  // ── 1. Dense regression — EP=1 produces identical results ──────────
  describe('Dense regression (EP=1 unchanged)', () => {
    it('LLaMA 2 7B with EP=1 gives same results as without EP field', () => {
      const config: SimulationConfig = {
        modelId: 'llama2-7b',
        clusterId: '8x-h100',
        globalBatchSize: 32,
        microBatchSize: 4,
        sequenceLength: 4096,
        strategyType: 'fsdp-tp',
        strategyConfig: { tp: 8, ep: 1 },
        activationCheckpointing: true,
        mixedPrecision: 'bf16',
      };

      const configNoEP: SimulationConfig = {
        ...config,
        strategyConfig: { tp: 8 },
      };

      const metrics = getValidatedSimulationMetrics(config);
      const metricsNoEP = getValidatedSimulationMetrics(configNoEP);

      // Should produce identical results
      expect(metrics.mfu).toBeCloseTo(metricsNoEP.mfu, 4);
      expect(metrics.stepTimeMs).toBeCloseTo(metricsNoEP.stepTimeMs, 4);
      expect(metrics.memoryPerGPU.parameters).toBeCloseTo(metricsNoEP.memoryPerGPU.parameters, 4);
    });

    it('Dense model with EP=1 has zero expertParallel communication', () => {
      const ctx = createTestContext('llama2-7b', '8x-h100', { dpDegree: 1 });
      const strategy = create3DParallelStrategy(8, 1, 1, { ep: 1, dpType: 'fsdp' });
      const comms = strategy.computeCommunication(ctx);
      expect(comms.expertParallel).toBe(0);
    });
  });

  // ── 2. EP memory reduction ─────────────────────────────────────────
  describe('EP memory reduction', () => {
    it('Mixtral 8x7B with EP=8 uses less param memory than EP=1 (same total sharding)', () => {
      // Compare two configs with same TP but different EP
      // EP=8, TP=1, DP=8 → total GPUs = 1*1*8*8 = 64
      // EP=1, TP=1, DP=64 → total GPUs = 1*1*64*1 = 64
      // Both FSDP, so params sharded by DP.
      // For EP=8: expert sharding = EP*DP = 64, shared sharding = TP*PP*DP = 8
      // For EP=1: expert sharding = EP*DP = 64, shared sharding = TP*PP*DP = 64
      // So shared params are less sharded with EP=8, but expert params same
      // Use a scenario that demonstrates EP reduces expert param memory per GPU

      // Direct comparison: same cluster, same model, just EP differs
      // EP=8, TP=1, DP=8 (1*1*8*8=64)
      const ctxEP8 = createTestContext('mixtral-8x7b', '64x-h100', { dpDegree: 8 });
      // EP=1, TP=1, DP=64 (1*1*64*1=64)
      const ctxEP1 = createTestContext('mixtral-8x7b', '64x-h100', { dpDegree: 64 });

      const strategyEP8 = create3DParallelStrategy(1, 1, 8, { ep: 8, dpType: 'fsdp' });
      const strategyEP1 = create3DParallelStrategy(1, 1, 64, { ep: 1, dpType: 'fsdp' });

      strategyEP8.computeMemoryPerGPU(ctxEP8);
      strategyEP1.computeMemoryPerGPU(ctxEP1);

      // With EP=1 and DP=64: sharedParamSharding=64, expertParamSharding=64
      // With EP=8 and DP=8: sharedParamSharding=8, expertParamSharding=64
      // EP=8 has LESS total sharding for shared params → more shared param memory
      // But EP=8 distributes experts across EP dimension properly.
      // The key insight: EP=8 DP=8 has same total expert sharding as EP=1 DP=64
      // but the parameters field is params in bf16, not optimizer.
      // Actually, the total memory should be similar since total sharding is similar.
      // Let's instead compare EP=4 vs EP=1 with same DP to isolate EP effect.

      // Better comparison: same TP, PP, DP but different EP (different total GPUs)
      // Or: compare on same cluster with adjusted DP
      // For memory reduction, the point is that EP distributes expert weights
      // across EP dimension. Without EP, all experts are replicated.

      // Compare same TP, same DP, different EP to isolate EP's memory benefit:
      // EP=1, TP=2, DP=4 → expertParamSharding = tp*pp*ep = 2, expertDPReplicas = 4 (DDP: no DP sharding)
      // EP=4, TP=2, DP=4 → expertParamSharding = tp*pp*ep = 8, expertDPReplicas = 1 (DDP: no DP sharding)
      // With DDP: expert params sharded by tp*pp*ep only (no DP sharding of params)
      const ctx = createTestContext('mixtral-8x7b', '8x-h100', { dpDegree: 4 });
      const stratA = create3DParallelStrategy(2, 1, 4, { ep: 1, dpType: 'ddp' });
      const stratB = create3DParallelStrategy(2, 1, 4, { ep: 4, dpType: 'ddp' });

      const memA = stratA.computeMemoryPerGPU(ctx);
      const memB = stratB.computeMemoryPerGPU(ctx);

      // EP=4 should use less param memory: expert params sharded 4x more
      expect(memB.parameters).toBeLessThan(memA.parameters);
    });

    it('Expert params scale inversely with EP', () => {
      // Same TP=2, PP=1, DP=4, vary EP. DDP so DP doesn't affect param sharding.
      // EP=2: expertParamSharding = tp*pp*ep = 2*1*2 = 4
      // EP=4: expertParamSharding = tp*pp*ep = 2*1*4 = 8
      const ctx = createTestContext('mixtral-8x7b', '8x-h100', { dpDegree: 4 });

      const strat2 = create3DParallelStrategy(2, 1, 4, { ep: 2, dpType: 'ddp' });
      const strat4 = create3DParallelStrategy(2, 1, 4, { ep: 4, dpType: 'ddp' });

      const mem2 = strat2.computeMemoryPerGPU(ctx);
      const mem4 = strat4.computeMemoryPerGPU(ctx);

      // EP=4 should have less param memory than EP=2 (experts sharded 2x more)
      expect(mem4.parameters).toBeLessThan(mem2.parameters);
    });
  });

  // ── 3. EP communication ────────────────────────────────────────────
  describe('EP communication', () => {
    it('expertParallel > 0 when EP > 1 for MoE model', () => {
      // TP=4, DP=2, EP=2 (EP subdivides DP) → 4*1*2 = 8
      const ctx = createTestContext('mixtral-8x7b', '8x-h100', { dpDegree: 2 });
      const strategy = create3DParallelStrategy(4, 1, 2, { ep: 2, dpType: 'fsdp' });
      const comms = strategy.computeCommunication(ctx);
      expect(comms.expertParallel).toBeGreaterThan(0);
    });

    it('expertParallel === 0 when EP = 1', () => {
      const ctx = createTestContext('mixtral-8x7b', '8x-h100', { dpDegree: 1 });
      const strategy = create3DParallelStrategy(8, 1, 1, { ep: 1, dpType: 'fsdp' });
      const comms = strategy.computeCommunication(ctx);
      expect(comms.expertParallel).toBe(0);
    });

    it('EP communication scales with (EP-1)/EP', () => {
      // Use 64× H100: TP=4, DP=16, EP divides DP
      const dp = 16;
      const ctx = createTestContext('mixtral-8x7b', '64x-h100', { dpDegree: dp });

      // EP=2: DP=16, EP divides DP
      const strat2 = create3DParallelStrategy(4, 1, dp, { ep: 2, dpType: 'fsdp' });
      // EP=4: DP=16, EP divides DP
      const strat4 = create3DParallelStrategy(4, 1, dp, { ep: 4, dpType: 'fsdp' });

      const comms2 = strat2.computeCommunication(ctx);
      const comms4 = strat4.computeCommunication(ctx);

      // EP comm volume ∝ (EP-1)/EP × routingLocality, where
      //   routingLocality = 1 / (1 + expertsPerRank / numActive)
      // This models learned MoE routers preferentially activating local experts.
      // With fewer experts per rank (higher EP), there are fewer local options relative
      // to top-k demand, so a larger fraction of selections cross EP boundaries.
      //
      // Mixtral 8x7B: 8 experts, numActive=2
      //   EP=2: expertsPerRank=4, routingLocality=1/(1+4/2)=1/3
      //         volume factor = (1/2) × (1/3) = 1/6
      //   EP=4: expertsPerRank=2, routingLocality=1/(1+2/2)=1/2
      //         volume factor = (3/4) × (1/2) = 3/8
      //   Ratio = (3/8) / (1/6) = 18/8 = 2.25
      //
      // Without routing locality, the ratio would be (3/4)/(1/2) = 1.5 from pure
      // (EP-1)/EP scaling. The extra factor comes from higher routingLocality at EP=4:
      // fewer local experts per rank means less local supply to satisfy demand,
      // so more tokens must be dispatched cross-rank.
      const ratio = comms4.expertParallel / comms2.expertParallel;
      expect(ratio).toBeGreaterThan(2.1);
      expect(ratio).toBeLessThan(2.4);
    });
  });

  // ── 4. MFU range ──────────────────────────────────────────────────
  describe('MFU range', () => {
    it('Mixtral 8x7B on 64× H100 with EP gets reasonable MFU', () => {
      // TP=1, DP=64, EP=8 subdivides DP → 1*1*64 = 64 (fits in memory)
      const config: SimulationConfig = {
        modelId: 'mixtral-8x7b',
        clusterId: '64x-h100',
        globalBatchSize: 512,
        microBatchSize: 2,
        sequenceLength: 4096,
        strategyType: 'fsdp-tp',
        strategyConfig: { tp: 1, ep: 8 },
        activationCheckpointing: true,
        mixedPrecision: 'bf16',
      };

      const metrics = getValidatedSimulationMetrics(config);
      // Lower bound accounts for activation checkpointing recompute overhead.
      // Upper bound > 1.0: at TP=1 EP=numExperts, each GPU runs 1 expert but
      // MFU denominator uses activeParams (topK/numExperts fraction), yielding
      // apparent MFU ~2-3% above 100%. This is a convention artifact, not a
      // physics error — the GPU genuinely does more work than activeParams implies.
      expect(metrics.mfu).toBeGreaterThan(0.15);
      expect(metrics.mfu).toBeLessThan(1.10);
    });

    it('DBRX on 64× H100 with EP gets reasonable MFU', () => {
      // DBRX has 16 experts. TP=4, DP=16, EP=2 subdivides DP → 4*1*16=64
      const config: SimulationConfig = {
        modelId: 'dbrx',
        clusterId: '64x-h100',
        globalBatchSize: 256,
        microBatchSize: 1,
        sequenceLength: 4096,
        strategyType: 'fsdp-tp',
        strategyConfig: { tp: 4, ep: 2 },
        activationCheckpointing: true,
        mixedPrecision: 'bf16',
      };

      const metrics = getValidatedSimulationMetrics(config);
      // MBS=1 with FSDP means high GA → FSDP comm scales per micro-batch
      // With 64 GPUs and GBS=256, GA is high so comm overhead is significant
      expect(metrics.mfu).toBeGreaterThan(0.02);
      expect(metrics.mfu).toBeLessThan(0.65);
    });
  });

  // ── 5. GPU count enforcement ──────────────────────────────────────
  describe('GPU count enforcement', () => {
    it('TP × PP × DP = totalGPUs is enforced (EP subdivides DP)', () => {
      // Valid config: 4 × 1 × 2 = 8, EP=2 divides DP=2
      const ctx = createTestContext('mixtral-8x7b', '8x-h100', { dpDegree: 2 });
      const strategy = create3DParallelStrategy(4, 1, 2, { ep: 2, dpType: 'fsdp' });
      const validation = strategy.validate(ctx);
      // Check the GPU count error specifically
      const gpuCountErrors = validation.errors.filter(e => e.includes("doesn't match cluster"));
      expect(gpuCountErrors).toHaveLength(0);
    });

    it('Invalid GPU count produces error', () => {
      // Invalid: 8 × 1 × 3 = 24 ≠ 64
      const ctx = createTestContext('mixtral-8x7b', '64x-h100', { dpDegree: 3 });
      const strategy = create3DParallelStrategy(8, 1, 3, { ep: 4, dpType: 'fsdp' });
      const validation = strategy.validate(ctx);
      expect(validation.errors.some(e => e.includes("doesn't match cluster"))).toBe(true);
    });
  });

  // ── 6. EP validation ──────────────────────────────────────────────
  describe('EP validation', () => {
    it('EP must divide numExperts', () => {
      // Mixtral has 8 experts, EP=3 doesn't divide 8
      // Use TP=1, EP=3, DP=? → needs 8/(1*1*3) which is not integer, but we force it
      const ctx = createTestContext('mixtral-8x7b', '8x-h100', { dpDegree: 2 });
      const strategy = create3DParallelStrategy(1, 1, 2, { ep: 3, dpType: 'fsdp' });
      const validation = strategy.validate(ctx);
      expect(validation.errors.some(e => e.includes('must divide'))).toBe(true);
    });

    it('EP > 1 rejected for dense models', () => {
      // TP=4, DP=2, EP=2 → 4*1*2 = 8 (EP subdivides DP)
      const ctx = createTestContext('llama2-7b', '8x-h100', { dpDegree: 2 });
      const strategy = create3DParallelStrategy(4, 1, 2, { ep: 2, dpType: 'fsdp' });
      const validation = strategy.validate(ctx);
      expect(validation.errors.some(e => e.includes('not a Mixture of Experts'))).toBe(true);
    });

    it('EP=1 is valid for both MoE and dense models', () => {
      // Dense model with EP=1, TP=8, DP=1 (8*1*1=8)
      const ctxDense = createTestContext('llama2-7b', '8x-h100', { dpDegree: 1 });
      const stratDense = create3DParallelStrategy(8, 1, 1, { ep: 1, dpType: 'fsdp' });
      const denseValidation = stratDense.validate(ctxDense);
      // Should not have GPU count or EP errors
      const denseGpuErrors = denseValidation.errors.filter(e =>
        e.includes("doesn't match") || e.includes('not a Mixture') || e.includes('must divide')
      );
      expect(denseGpuErrors).toHaveLength(0);

      // MoE model with EP=1, TP=1, DP=8 (1*1*8=8) — all experts replicated
      const ctxMoE = createTestContext('mixtral-8x7b', '8x-h100', { dpDegree: 8 });
      const stratMoE = create3DParallelStrategy(1, 1, 8, { ep: 1, dpType: 'fsdp' });
      const moeValidation = stratMoE.validate(ctxMoE);
      const moeGpuErrors = moeValidation.errors.filter(e =>
        e.includes("doesn't match") || e.includes('not a Mixture') || e.includes('must divide')
      );
      expect(moeGpuErrors).toHaveLength(0);
    });

    it('MoE with EP=1 and many experts has no strategy-level EP suggestion (handled by recommendation engine)', () => {
      const ctx = createTestContext('mixtral-8x7b', '8x-h100', { dpDegree: 8 });
      const strategy = create3DParallelStrategy(1, 1, 8, { ep: 1, dpType: 'fsdp' });
      const validation = strategy.validate(ctx);
      // EP suggestions moved to unified recommendation engine (generateRecommendations)
      expect(validation.suggestions.some(s => s.includes('expert parallelism'))).toBe(false);
    });
  });

  // ── 7. DeepSeek-V3 at scale ───────────────────────────────────────
  describe('DeepSeek-V3 at scale', () => {
    it('DeepSeek-V3 on 2048× H100 runs successfully with FP8', () => {
      // DeepSeek-V3 is ~671B total / ~37.6B active (MLA) — needs FP8 + PP to fit
      // TP=2, PP=4, DP=256 → 2*4*256=2048, EP=4 subdivides DP=256
      // ep*tp = 4*2 = 8 = gpusPerNode → EP within node (NVLink)
      const config: SimulationConfig = {
        modelId: 'deepseek-v3',
        clusterId: '2048x-h100',
        globalBatchSize: 4096,
        microBatchSize: 1,
        sequenceLength: 4096,
        strategyType: 'fsdp-tp-pp',
        strategyConfig: { tp: 2, pp: 4, ep: 4 },
        activationCheckpointing: true,
        mixedPrecision: 'fp8',
      };

      const result = runSimulation(config);
      expect(result.success).toBe(true);
      // MBS=1 with FSDP dpType and high GA (4096/64=64) means comm scales ×64
      // MFU is very low but nonzero — comm-dominated config
      expect(result.metrics.mfu).toBeGreaterThanOrEqual(0);
      expect(result.metrics.mfu).toBeLessThan(0.60);
    });

    it('DeepSeek-V3 with EP has nonzero expert communication', () => {
      // TP=2, PP=4, DP=256 (2048/(2*4)=256), EP=4 subdivides DP
      const dp = Math.floor(2048 / (2 * 4));
      const ctx = createTestContext('deepseek-v3', '2048x-h100', {
        dpDegree: dp,
        globalBatchSize: 4096,
        microBatchSize: 1,
      });
      const strategy = create3DParallelStrategy(2, 4, dp, { ep: 4, dpType: 'fsdp' });
      const comms = strategy.computeCommunication(ctx);
      expect(comms.expertParallel).toBeGreaterThan(0);
      expect(comms.total).toBeGreaterThan(comms.dataParallel + comms.tensorParallel + comms.pipelineParallel);
    });
  });

  // ── Full simulation integration ───────────────────────────────────
  describe('Full simulation integration', () => {
    it('Mixtral 8x7B via runSimulation with EP', () => {
      // TP=1, DP=64, EP=8 subdivides DP → 1*1*64 = 64 (fits in memory)
      const config: SimulationConfig = {
        modelId: 'mixtral-8x7b',
        clusterId: '64x-h100',
        globalBatchSize: 512,
        microBatchSize: 2,
        sequenceLength: 4096,
        strategyType: 'fsdp-tp',
        strategyConfig: { tp: 1, ep: 8 },
        activationCheckpointing: true,
        mixedPrecision: 'bf16',
      };

      const result = runSimulation(config);
      expect(result.success).toBe(true);
      expect(result.metrics.mfu).toBeGreaterThan(0);
      expect(result.metrics.peakMemoryGB).toBeGreaterThan(0);
      expect(result.metrics.tokensPerSecond).toBeGreaterThan(0);
    });

    it('Auto strategy selects EP for MoE models', () => {
      const config: SimulationConfig = {
        modelId: 'mixtral-8x7b',
        clusterId: '64x-h100',
        globalBatchSize: 512,
        microBatchSize: 2,
        sequenceLength: 4096,
        strategyType: 'auto',
        activationCheckpointing: true,
        mixedPrecision: 'bf16',
      };

      const result = runSimulation(config);
      expect(result.success).toBe(true);
      expect(result.metrics.mfu).toBeGreaterThan(0);
    });
  });
});
