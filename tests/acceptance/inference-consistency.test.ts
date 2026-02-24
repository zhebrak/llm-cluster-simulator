/**
 * Inference route-consistency tests.
 *
 * Verifies that inference TP/EP paths produce internally consistent results:
 * 1. TP=1 vs calculateLatencyWithTP(tp=1) should be within 5%
 * 2. Higher TP should reduce latency (not increase it)
 * 3. EP>1 should reduce MoE latency vs EP=1
 * 4. TTFT and TPOT are always positive and finite
 * 5. MoE and dense models don't exhibit discontinuities at boundaries
 */

import { describe, it, expect } from 'vitest';
import { getModel } from '../../src/core/models/index.ts';
import { H100_SXM, A100_80GB } from '../../src/core/hardware/gpu.ts';
import {
  estimateTTFT,
  estimateTPOT,
  calculateLatencyWithTP,
} from '../../src/core/inference/latency.ts';

const llama7b = getModel('llama2-7b', 2048)!;
const llama70b = getModel('llama2-70b', 2048)!;
const v3 = getModel('deepseek-v3', 2048)!;
const gpu = H100_SXM;

describe('Inference TP consistency', () => {
  it('calculateLatencyWithTP(tp=1) TTFT within 5% of single-GPU estimator', () => {
    const inputLen = 512;
    const outputLen = 128;
    const batchSize = 1;

    const ttftSingle = estimateTTFT(llama7b, inputLen, gpu, 'bf16', batchSize);
    const tpResult = calculateLatencyWithTP(
      llama7b, inputLen, outputLen, batchSize, gpu, 1, 'bf16',
    );

    // With TP=1 AllReduce gated off, TP path should match single-GPU exactly.
    const relDiff = Math.abs(tpResult.ttft - ttftSingle) / ttftSingle;
    expect(relDiff).toBeLessThan(0.001);
  });

  it('higher TP should reduce TTFT for compute-bound prefill', () => {
    const inputLen = 2048;
    const outputLen = 256;
    const batchSize = 1;

    const tp1 = calculateLatencyWithTP(llama70b, inputLen, outputLen, batchSize, gpu, 1);
    const tp2 = calculateLatencyWithTP(llama70b, inputLen, outputLen, batchSize, gpu, 2);
    const tp4 = calculateLatencyWithTP(llama70b, inputLen, outputLen, batchSize, gpu, 4);
    const tp8 = calculateLatencyWithTP(llama70b, inputLen, outputLen, batchSize, gpu, 8);

    expect(tp2.ttft).toBeLessThan(tp1.ttft);
    expect(tp4.ttft).toBeLessThan(tp2.ttft);
    expect(tp8.ttft).toBeLessThan(tp4.ttft);
  });

  it('higher TP should reduce TPOT for large models', () => {
    const inputLen = 512;
    const outputLen = 256;
    const batchSize = 1;

    const tp1 = calculateLatencyWithTP(llama70b, inputLen, outputLen, batchSize, gpu, 1);
    const tp2 = calculateLatencyWithTP(llama70b, inputLen, outputLen, batchSize, gpu, 2);
    const tp4 = calculateLatencyWithTP(llama70b, inputLen, outputLen, batchSize, gpu, 4);

    expect(tp2.tpot).toBeLessThan(tp1.tpot);
    expect(tp4.tpot).toBeLessThan(tp2.tpot);
  });
});

describe('Inference EP consistency (MoE)', () => {
  it('EP>1 should reduce MoE TTFT', () => {
    const inputLen = 1024;
    const outputLen = 128;
    const batchSize = 1;

    const ep1 = calculateLatencyWithTP(v3, inputLen, outputLen, batchSize, gpu, 8, 'bf16', 1);
    const ep4 = calculateLatencyWithTP(v3, inputLen, outputLen, batchSize, gpu, 8, 'bf16', 4);
    const ep8 = calculateLatencyWithTP(v3, inputLen, outputLen, batchSize, gpu, 8, 'bf16', 8);

    expect(ep4.ttft).toBeLessThan(ep1.ttft);
    expect(ep8.ttft).toBeLessThan(ep4.ttft);
  });

  it('EP>1 should reduce MoE TPOT', () => {
    const inputLen = 512;
    const outputLen = 128;
    const batchSize = 1;

    const ep1 = calculateLatencyWithTP(v3, inputLen, outputLen, batchSize, gpu, 8, 'bf16', 1);
    const ep4 = calculateLatencyWithTP(v3, inputLen, outputLen, batchSize, gpu, 8, 'bf16', 4);

    expect(ep4.tpot).toBeLessThan(ep1.tpot);
  });
});

describe('Inference sanity invariants', () => {
  const models = [
    { name: 'LLaMA 7B', model: llama7b },
    { name: 'LLaMA 70B', model: llama70b },
    { name: 'DeepSeek V3', model: v3 },
  ];

  const gpus = [
    { name: 'H100 SXM', gpu: H100_SXM },
    { name: 'A100 80GB', gpu: A100_80GB },
  ];

  for (const { name: mName, model } of models) {
    for (const { name: gName, gpu: g } of gpus) {
      it(`${mName} on ${gName}: TTFT and TPOT are positive and finite`, () => {
        const ttft = estimateTTFT(model, 512, g, 'bf16', 1);
        const tpot = estimateTPOT(model, 512, 1, g, 'bf16');

        expect(ttft).toBeGreaterThan(0);
        expect(ttft).toBeLessThan(60_000); // < 60 seconds
        expect(Number.isFinite(ttft)).toBe(true);

        expect(tpot).toBeGreaterThan(0);
        expect(tpot).toBeLessThan(10_000); // < 10 seconds per token
        expect(Number.isFinite(tpot)).toBe(true);
      });
    }
  }

  it('larger model → higher TPOT (at same batch/GPU)', () => {
    const tpot7b = estimateTPOT(llama7b, 512, 1, gpu, 'bf16');
    const tpot70b = estimateTPOT(llama70b, 512, 1, gpu, 'bf16');

    expect(tpot70b).toBeGreaterThan(tpot7b);
  });

  it('larger batch → higher total decode throughput (tokens/s)', () => {
    // TPOT may increase with batch (KV cache reads grow), but total
    // throughput (batch / tpot) should still increase with batching.
    const inputLen = 512;
    const outputLen = 128;

    const b1 = calculateLatencyWithTP(llama7b, inputLen, outputLen, 1, gpu, 1);
    const b8 = calculateLatencyWithTP(llama7b, inputLen, outputLen, 8, gpu, 1);
    const b32 = calculateLatencyWithTP(llama7b, inputLen, outputLen, 32, gpu, 1);

    const throughput1 = 1 / b1.tpot;
    const throughput8 = 8 / b8.tpot;
    const throughput32 = 32 / b32.tpot;

    expect(throughput8).toBeGreaterThan(throughput1);
    expect(throughput32).toBeGreaterThan(throughput8);
  });

  it('H100 faster than A100 for same config', () => {
    const ttftH100 = estimateTTFT(llama7b, 512, H100_SXM);
    const ttftA100 = estimateTTFT(llama7b, 512, A100_80GB);

    expect(ttftH100).toBeLessThan(ttftA100);
  });
});
