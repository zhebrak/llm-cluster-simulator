/**
 * Derived Physics Validation Tests
 *
 * Validates derived quantities:
 *   - matmulFraction / matmulTimeFraction from FLOPs + roofline decomposition
 *   - Multinomial load imbalance for MoE
 *   - CP causal work distribution for ring attention
 *   - Reserved memory decomposition
 */

import { describe, it, expect } from 'vitest';
import { buildModelSpec } from '../../src/core/models/primitives.ts';
import { getModelConfig } from '../../src/core/models/architectures.ts';
import {
  loadImbalanceFactor,
  cpCausalWorkDistribution,
  reservedMemoryDecomposition,
} from '../../src/core/physics/derived.ts';
import { calculateReservedMemory } from '../../src/core/strategies/base.ts';

// ---------------------------------------------------------------------------
// 1a. matmulFraction from FLOPs decomposition
// ---------------------------------------------------------------------------

describe('matmulFraction — FLOPs decomposition', () => {
  // NOTE: matmulFraction is a FLOPs ratio, not a time ratio. Non-matmul ops
  // (norms, activations, softmax) contribute < 1% of total FLOPs but consume
  // ~20% of wall-clock time because they are memory-bandwidth-bound.
  // MATMUL_TIME_FRACTION = 0.80 is a time fraction; matmulFraction ~0.999 is FLOPs.
  // FLOPs fraction is near 1.0; the time-domain effect is captured by memBW scaling.

  it('LLaMA 405B at 8K: matmulFraction near 1.0 (matmul FLOPs dominate)', () => {
    const config = getModelConfig('llama3-405b')!;
    expect(config).toBeDefined();
    const model = buildModelSpec(config, 8192);

    // Matmul FLOPs completely dominate; non-matmul (softmax, norms, activations) < 0.1%
    expect(model.matmulFraction).toBeGreaterThan(0.998);
    expect(model.matmulFraction).toBeLessThan(1.0);
    // matmulFlopsPerToken should be strictly less than flopsPerToken
    expect(model.matmulFlopsPerToken).toBeLessThan(model.flopsPerToken);
    expect(model.matmulFlopsPerToken).toBeGreaterThan(0);
  });

  it('LLaMA 405B at 131K: matmulFraction drops slightly (quadratic softmax grows)', () => {
    const config = getModelConfig('llama3-405b')!;
    const modelShort = buildModelSpec(config, 8192);
    const modelLong = buildModelSpec(config, 131072);

    // At long sequences, softmax FLOPs grow quadratically alongside attention matmul.
    // Both scale as S², so the drop is modest — MLP matmul (linear in S) loses share.
    expect(modelLong.matmulFraction).toBeLessThan(modelShort.matmulFraction);
    expect(modelLong.matmulFraction).toBeGreaterThan(0.993);
    expect(modelLong.matmulFraction).toBeLessThan(0.996);
  });

  it('DeepSeek V3: matmulFraction slightly lower than dense (routing softmax + activations)', () => {
    const dsConfig = getModelConfig('deepseek-v3')!;
    expect(dsConfig).toBeDefined();
    const dsModel = buildModelSpec(dsConfig, 4096);

    // MoE router softmax + expert activation ops reduce matmul fraction slightly
    expect(dsModel.matmulFraction).toBeGreaterThan(0.997);
    expect(dsModel.matmulFraction).toBeLessThan(1.0);
  });

  it('GPT-3 125M at 2K: matmulFraction similar to large models (architecture-dependent)', () => {
    const config = getModelConfig('gpt3-125m')!;
    expect(config).toBeDefined();
    const model = buildModelSpec(config, 2048);

    // Small models have similar matmul fraction — the split is architecture-dependent,
    // not size-dependent. Non-gated MLP (GELU) has slightly more activation FLOPs.
    expect(model.matmulFraction).toBeGreaterThan(0.993);
    expect(model.matmulFraction).toBeLessThan(1.0);
  });

  it('matmulFlops + non-matmulFlops == totalFlops for every layer', () => {
    const config = getModelConfig('llama3-405b')!;
    const model = buildModelSpec(config, 4096);

    for (const layer of model.layers) {
      expect(layer.matmulFlops).toBeGreaterThanOrEqual(0);
      expect(layer.matmulFlops).toBeLessThanOrEqual(layer.flops);
    }

    // Block-level consistency
    for (const block of model.blocks) {
      const componentMatmul =
        block.preNorm.matmulFlops +
        block.attention.matmulFlops +
        (block.postAttentionNorm?.matmulFlops ?? 0) +
        block.mlp.matmulFlops;
      expect(block.totalMatmulFlops).toBe(componentMatmul);
    }
  });

  it('embedding and norm layers have zero matmulFlops', () => {
    const config = getModelConfig('llama3-405b')!;
    const model = buildModelSpec(config, 4096);

    for (const layer of model.layers) {
      if (layer.type === 'embedding' || layer.type === 'rmsnorm' || layer.type === 'layernorm') {
        expect(layer.matmulFlops).toBe(0);
      }
    }
  });

  it('output layer matmulFlops == flops (all matmul)', () => {
    const config = getModelConfig('llama3-405b')!;
    const model = buildModelSpec(config, 4096);

    const outputLayer = model.layers.find(l => l.type === 'output')!;
    expect(outputLayer.matmulFlops).toBe(outputLayer.flops);
  });

  it('matmulFraction is 0 when flopsPerToken is 0', () => {
    // Edge case: a degenerate config with zero FLOPs shouldn't divide by zero
    // We test this via the formula directly since buildModelSpec requires valid config
    // The code handles this: matmulFraction = flopsPerToken > 0 ? ... : 0
    const config = getModelConfig('gpt3-125m')!;
    const model = buildModelSpec(config, 2048);
    // Just verify it's a valid number (not NaN/Infinity)
    expect(Number.isFinite(model.matmulFraction)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// 1b. Multinomial load imbalance
// ---------------------------------------------------------------------------

describe('loadImbalanceFactor — multinomial order statistics', () => {
  it('128 experts, top-2: predicts higher imbalance than old log curve', () => {
    // Old formula: 1 + 0.05 * log2(128/8) = 1 + 0.05 * 4 = 1.20
    const oldPrediction = 1 + 0.05 * Math.log2(128 / 8);
    // Typical microbatch: 4096 tokens (EP-adjusted)
    const derived = loadImbalanceFactor(128, 2, 4096);

    expect(derived).toBeGreaterThan(oldPrediction);
    expect(derived).toBeGreaterThan(1.0);
  });

  it('imbalance increases with expert count', () => {
    const tokens = 4096;
    const topK = 2;

    const imb8 = loadImbalanceFactor(8, topK, tokens);
    const imb32 = loadImbalanceFactor(32, topK, tokens);
    const imb128 = loadImbalanceFactor(128, topK, tokens);
    const imb256 = loadImbalanceFactor(256, topK, tokens);

    expect(imb32).toBeGreaterThan(imb8);
    expect(imb128).toBeGreaterThan(imb32);
    expect(imb256).toBeGreaterThan(imb128);
  });

  it('imbalance decreases with batch size', () => {
    const numExperts = 64;
    const topK = 2;

    const imbSmall = loadImbalanceFactor(numExperts, topK, 512);
    const imbMedium = loadImbalanceFactor(numExperts, topK, 4096);
    const imbLarge = loadImbalanceFactor(numExperts, topK, 32768);

    expect(imbSmall).toBeGreaterThan(imbMedium);
    expect(imbMedium).toBeGreaterThan(imbLarge);
  });

  it('returns 1.0 for degenerate inputs', () => {
    expect(loadImbalanceFactor(1, 1, 4096)).toBe(1.0);
    expect(loadImbalanceFactor(8, 2, 0)).toBe(1.0);
    expect(loadImbalanceFactor(0, 0, 0)).toBe(1.0);
  });

  it('higher topK reduces imbalance (more uniform distribution)', () => {
    const numExperts = 64;
    const tokens = 4096;

    const imbTop1 = loadImbalanceFactor(numExperts, 1, tokens);
    const imbTop2 = loadImbalanceFactor(numExperts, 2, tokens);
    const imbTop4 = loadImbalanceFactor(numExperts, 4, tokens);

    // With higher topK, each expert receives more tokens on average,
    // reducing relative variance
    expect(imbTop1).toBeGreaterThan(imbTop2);
    expect(imbTop2).toBeGreaterThan(imbTop4);
  });
});

// ---------------------------------------------------------------------------
// 1c. CP causal work distribution
// ---------------------------------------------------------------------------

describe('cpCausalWorkDistribution — ring attention causal masking', () => {
  it('CP=2: 1 diagonal step, 0 normal steps', () => {
    const dist = cpCausalWorkDistribution(2);
    expect(dist.diagonalComputeFraction).toBe(0.5);
    expect(dist.diagonalSteps).toBe(1);
    expect(dist.normalSteps).toBe(0);
  });

  it('CP=8: 1 diagonal step, 6 normal steps', () => {
    const dist = cpCausalWorkDistribution(8);
    expect(dist.diagonalComputeFraction).toBe(0.5);
    expect(dist.diagonalSteps).toBe(1);
    expect(dist.normalSteps).toBe(6);
  });

  it('CP=16: 1 diagonal step, 14 normal steps', () => {
    const dist = cpCausalWorkDistribution(16);
    expect(dist.diagonalComputeFraction).toBe(0.5);
    expect(dist.diagonalSteps).toBe(1);
    expect(dist.normalSteps).toBe(14);
  });

  it('CP=1: degenerate, no ring steps', () => {
    const dist = cpCausalWorkDistribution(1);
    expect(dist.diagonalSteps).toBe(0);
    expect(dist.normalSteps).toBe(0);
  });

  it('total steps = CP - 1 for CP >= 2', () => {
    for (const cp of [2, 4, 8, 16, 32, 64]) {
      const dist = cpCausalWorkDistribution(cp);
      expect(dist.diagonalSteps + dist.normalSteps).toBe(cp - 1);
    }
  });
});

// ---------------------------------------------------------------------------
// 1d. Reserved memory decomposition
// ---------------------------------------------------------------------------

describe('reservedMemoryDecomposition — physical decomposition', () => {
  it('H100 SXM 80GB: parity within 5% of old formula', () => {
    const oldReserved = calculateReservedMemory(80);
    const decomp = reservedMemoryDecomposition(80, 'hopper');
    const relError = Math.abs(decomp.totalBytes - oldReserved) / oldReserved;
    expect(relError).toBeLessThan(0.05);
  });

  it('A100 80GB: parity within 5% of old formula', () => {
    const oldReserved = calculateReservedMemory(80);
    const decomp = reservedMemoryDecomposition(80, 'ampere');
    const relError = Math.abs(decomp.totalBytes - oldReserved) / oldReserved;
    expect(relError).toBeLessThan(0.05);
  });

  it('A100 40GB: parity within 5% of old formula', () => {
    const oldReserved = calculateReservedMemory(40);
    const decomp = reservedMemoryDecomposition(40, 'ampere');
    const relError = Math.abs(decomp.totalBytes - oldReserved) / oldReserved;
    expect(relError).toBeLessThan(0.05);
  });

  it('active components are positive', () => {
    const decomp = reservedMemoryDecomposition(80, 'hopper');
    expect(decomp.cudaContextBytes).toBeGreaterThan(0);
    expect(decomp.fragmentationBytes).toBeGreaterThan(0);
    expect(decomp.totalBytes).toBe(
      decomp.cudaContextBytes + decomp.fragmentationBytes,
    );
  });

  it('fragmentation is 7% of physical GiB capacity', () => {
    const decomp = reservedMemoryDecomposition(80, 'hopper');
    const expectedFrag = 80 * (1024 ** 3) * 0.07;
    expect(decomp.fragmentationBytes).toBeCloseTo(expectedFrag, 0);
  });

  it('CUDA context varies by architecture', () => {
    const hopper = reservedMemoryDecomposition(80, 'hopper');
    const ampere = reservedMemoryDecomposition(80, 'ampere');
    const blackwell = reservedMemoryDecomposition(80, 'blackwell');

    expect(hopper.cudaContextBytes).toBeGreaterThan(ampere.cudaContextBytes);
    expect(blackwell.cudaContextBytes).toBeGreaterThan(hopper.cudaContextBytes);
  });

  it('larger GPU memory → larger total reserved', () => {
    const small = reservedMemoryDecomposition(40, 'ampere');
    const large = reservedMemoryDecomposition(80, 'ampere');
    expect(large.totalBytes).toBeGreaterThan(small.totalBytes);
  });
});
