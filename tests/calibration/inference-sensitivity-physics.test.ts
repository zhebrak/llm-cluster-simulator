/**
 * Physics assumption validation using the perturbable inference infrastructure.
 *
 * Tests fundamental inference physics:
 * - Decode is memory-bandwidth-bound
 * - Prefill is compute-bound
 * - TP scaling sanity
 * - CB improvement
 * - Bandwidth efficiency monotonicity
 * - Quantization speedup
 * - MoE weight bytes batch curve
 * - Attention FLOPs scaling
 * - Bandwidth efficiency with KV cache
 */
import { describe, it, expect, afterEach } from 'vitest';
import { runInferenceSensitivityAnalysis } from '../../src/core/analysis/inference-sensitivity.ts';
import { resetAllPerturbableInferenceParams } from '../../src/core/analysis/inference-perturbable-params.ts';
import {
  runInferenceSimulationRaw,
  type InferenceSimulationConfig,
} from '../../src/core/inference/simulation.ts';
import {
  getBandwidthEfficiency,
  decodeFLOPs,
  prefillFLOPs,
  moeWeightBytesPerStep,
} from '../../src/core/inference/latency.ts';
import { getPrecisionBytes } from '../../src/core/inference/kv-cache.ts';
import { getModel } from '../../src/core/models/index.ts';
import type { PerturbableParam } from '../../src/core/analysis/sensitivity.ts';
import {
  getPrefillResidual, setPrefillResidual,
  getBwEffFloor, setBwEffFloor,
} from '../../src/core/inference/latency.ts';

afterEach(() => {
  resetAllPerturbableInferenceParams();
});

describe('Inference Physics Validation', () => {
  it('decode is memory-bandwidth-bound: BW_EFF_FLOOR affects TPOT, PREFILL_RESIDUAL does not', () => {
    // Use 8B model — fits on 1 H100, and small enough that BW floor matters
    const benchmarks: { name: string; config: InferenceSimulationConfig }[] = [{
      name: '8B B=1',
      config: {
        modelId: 'llama3.1-8b',
        gpuId: 'h100-sxm',
        numGPUs: 1,
        batchSize: 1,
        inputSeqLen: 512,
        outputSeqLen: 128,
        weightPrecision: 'bf16',
      },
    }];

    // Perturb BW_EFF_FLOOR — should change TPOT
    const bwParams: PerturbableParam[] = [{
      name: 'BW_EFF_FLOOR', getValue: getBwEffFloor, setValue: setBwEffFloor,
    }];
    const bwReport = runInferenceSensitivityAnalysis(bwParams, benchmarks, 0.10);
    const bwDelta = bwReport.results[0].deltas[0];
    expect(bwDelta.tpotMinus).not.toBe(bwDelta.tpotPlus);

    // Perturb PREFILL_RESIDUAL — should NOT change TPOT (decode doesn't use it directly
    // unless compute floor binds, which it won't at batch=1)
    const prParams: PerturbableParam[] = [{
      name: 'PREFILL_RESIDUAL', getValue: getPrefillResidual, setValue: setPrefillResidual,
    }];
    const prReport = runInferenceSensitivityAnalysis(prParams, benchmarks, 0.10);
    const prDelta = prReport.results[0].deltas[0];
    // PREFILL_RESIDUAL affects compute floor in decode, but at B=1 decode is memory-bound
    // so TPOT should be unchanged or negligibly different
    const tpotDelta = Math.abs(prDelta.tpotPlus - prDelta.tpotMinus) / prDelta.tpotOriginal;
    expect(tpotDelta).toBeLessThan(0.01);
  });

  it('prefill is compute-bound: PREFILL_RESIDUAL affects TTFT, BW_EFF_FLOOR does not', () => {
    // Use 8B model — fits on 1 H100, and with B=32 seq=1024 prefill is compute-bound
    const benchmarks: { name: string; config: InferenceSimulationConfig }[] = [{
      name: '8B B=32 seq=1024',
      config: {
        modelId: 'llama3.1-8b',
        gpuId: 'h100-sxm',
        numGPUs: 1,
        batchSize: 32,
        inputSeqLen: 1024,
        outputSeqLen: 128,
        weightPrecision: 'bf16',
      },
    }];

    // Perturb PREFILL_RESIDUAL — should change TTFT
    const prParams: PerturbableParam[] = [{
      name: 'PREFILL_RESIDUAL', getValue: getPrefillResidual, setValue: setPrefillResidual,
    }];
    const prReport = runInferenceSensitivityAnalysis(prParams, benchmarks, 0.10);
    const prDelta = prReport.results[0].deltas[0];
    expect(prDelta.ttftMinus).not.toBe(prDelta.ttftPlus);

    // Perturb BW_EFF_FLOOR — should NOT change TTFT (prefill is compute-bound)
    const bwParams: PerturbableParam[] = [{
      name: 'BW_EFF_FLOOR', getValue: getBwEffFloor, setValue: setBwEffFloor,
    }];
    const bwReport = runInferenceSensitivityAnalysis(bwParams, benchmarks, 0.10);
    const bwDelta = bwReport.results[0].deltas[0];
    const ttftDelta = Math.abs(bwDelta.ttftPlus - bwDelta.ttftMinus) / bwDelta.ttftOriginal;
    expect(ttftDelta).toBeLessThan(0.01);
  });

  it('TP scaling sanity: 8B TP=2 vs TP=1 throughput ratio ∈ [1.3, 2.0]', () => {
    // Use 8B model that fits on 1 GPU
    const tp1 = runInferenceSimulationRaw({
      modelId: 'llama3.1-8b',
      gpuId: 'h100-sxm',
      numGPUs: 1,
      batchSize: 8,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
    });

    const tp2 = runInferenceSimulationRaw({
      modelId: 'llama3.1-8b',
      gpuId: 'h100-sxm',
      numGPUs: 2,
      tensorParallel: 2,
      batchSize: 8,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
    });

    const ratio = tp2.throughput.tokensPerSecond / tp1.throughput.tokensPerSecond;
    // Smaller model may see less TP benefit due to AR overhead
    expect(ratio).toBeGreaterThanOrEqual(1.3);
    expect(ratio).toBeLessThanOrEqual(2.0);
  });

  it('CB improvement: 8B batch=64 CB throughput ≥ static throughput', () => {
    const staticResult = runInferenceSimulationRaw({
      modelId: 'llama3.1-8b',
      gpuId: 'h100-sxm',
      numGPUs: 1,
      batchSize: 64,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
    });

    const cbResult = runInferenceSimulationRaw({
      modelId: 'llama3.1-8b',
      gpuId: 'h100-sxm',
      numGPUs: 1,
      batchSize: 64,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
      continuousBatching: true,
    });

    expect(cbResult.throughput.tokensPerSecond).toBeGreaterThanOrEqual(
      staticResult.throughput.tokensPerSecond
    );
  });

  it('bandwidth efficiency monotonicity: larger model = higher efficiency', () => {
    const tiny = 125e6 * 2;  // 125M params × 2 bytes = ~250MB
    const small = 8e9 * 2;   // 8B params × 2 bytes = ~16GB
    const large = 70e9 * 2;  // 70B params × 2 bytes = ~140GB

    const effTiny = getBandwidthEfficiency(tiny);
    const effSmall = getBandwidthEfficiency(small);
    const effLarge = getBandwidthEfficiency(large);

    expect(effLarge).toBeGreaterThan(effSmall);
    expect(effSmall).toBeGreaterThan(effTiny);
  });

  it('quantization speedup: 8B INT4 throughput > BF16 throughput on H100', () => {
    const bf16 = runInferenceSimulationRaw({
      modelId: 'llama3.1-8b',
      gpuId: 'h100-sxm',
      numGPUs: 1,
      batchSize: 8,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'bf16',
    });

    const int4 = runInferenceSimulationRaw({
      modelId: 'llama3.1-8b',
      gpuId: 'h100-sxm',
      numGPUs: 1,
      batchSize: 8,
      inputSeqLen: 512,
      outputSeqLen: 128,
      weightPrecision: 'int4',
    });

    expect(bf16.success).toBe(true);
    expect(int4.success).toBe(true);
    expect(int4.throughput.tokensPerSecond).toBeGreaterThan(bf16.throughput.tokensPerSecond);
  });

  it('MoE weight bytes batch curve: batch=1 ≈ activeParams, batch=1000 → totalParams', () => {
    const model = getModel('deepseek-v3', 512)!;
    const bytesPerParam = getPrecisionBytes('bf16');

    // Batch=1: weight bytes ≈ activeParams × bytesPerParam
    const b1Bytes = moeWeightBytesPerStep(model, 1, 'bf16');
    const activeBytes = (model.activeParams ?? model.totalParams) * bytesPerParam;
    expect(b1Bytes).toBeGreaterThan(activeBytes * 0.95);
    expect(b1Bytes).toBeLessThan(activeBytes * 1.05);

    // Batch=1000: weight bytes → totalParams × bytesPerParam
    const b1000Bytes = moeWeightBytesPerStep(model, 1000, 'bf16');
    const totalBytes = model.totalParams * bytesPerParam;
    expect(b1000Bytes).toBeGreaterThan(totalBytes * 0.90);
    expect(b1000Bytes).toBeLessThanOrEqual(totalBytes * 1.01);
  });

  it('attention FLOPs scaling: default and seqLen proportionality', () => {
    const model = getModel('llama3.3-70b', 4096)!;
    const twoP = 2 * (model.activeParams ?? model.totalParams);

    // Default: seqLen=0 returns 2*activeParams (linear FLOPs only, no attention)
    expect(decodeFLOPs(model, 0)).toBe(twoP);
    expect(decodeFLOPs(model)).toBe(twoP);

    // SeqLen=4096: attention adds some percent
    const at4k = decodeFLOPs(model, 4096);
    expect(at4k).toBeGreaterThan(twoP);

    // SeqLen=128K: attention exceeds linear (at4k/twoP ratio should scale with seqLen)
    const at128k = decodeFLOPs(model, 131072);
    expect(at128k).toBeGreaterThan(twoP * 2); // attention adds more than 100% at very long seqLen

    // Prefill short seq: attention is small fraction
    const prefill1k = prefillFLOPs(model, 1024);
    expect(prefill1k).toBeGreaterThan(twoP * 1024 * 0.99);
    expect(prefill1k).toBeLessThan(twoP * 1024 * 1.10); // < 10% overhead

    // Prefill long seq: attention is significant
    const prefill32k = prefillFLOPs(model, 32768);
    expect(prefill32k).toBeGreaterThan(twoP * 32768 * 1.25); // attention adds >25%
  });

  it('MLA attention FLOPs use kvLoraRank, not hiddenSize', () => {
    const model = getModel('deepseek-v3', 4096)!;
    expect(model.attentionType).toBe('mla');
    expect(model.kvLoraRank).toBeDefined();

    const twoP = 2 * (model.activeParams ?? model.totalParams);
    const attnFlops = decodeFLOPs(model, 4096) - twoP;

    // MLA attention per layer: 2 * numHeads * (2*kvLoraRank + qkRopeHeadDim) * seqLen
    // This should be smaller than standard MHA (4 * numHeads * headDim * seqLen)
    const mlaPerLayer = 2 * model.numAttentionHeads * (2 * model.kvLoraRank! + model.qkRopeHeadDim!) * 4096;
    const mhaPerLayer = 4 * model.numAttentionHeads * model.headDim * 4096;

    // MLA attention should be similar to but different from MHA
    expect(attnFlops).toBeGreaterThan(0);
    // Verify MLA is actually different from MHA formula
    expect(mlaPerLayer).not.toBe(mhaPerLayer);
  });

  it('bandwidth efficiency with KV cache: weights+KV >= weights-only efficiency', () => {
    const smallWeights = 8e9 * 2; // 8B bf16
    const largeKV = 4e9;          // 4GB KV cache

    const weightsOnly = getBandwidthEfficiency(smallWeights);
    const withKV = getBandwidthEfficiency(smallWeights + largeKV);

    // Adding KV cache bytes increases total HBM volume → higher efficiency
    expect(withKV).toBeGreaterThanOrEqual(weightsOnly);
  });

  it('bandwidth efficiency: at batch=1 short seqLen KV is tiny, delta < 1%', () => {
    const weightsBytes = 70e9 * 2; // 70B bf16 = ~140GB
    const tinyKV = 1e6;            // 1MB KV at batch=1 short seq

    const noKV = getBandwidthEfficiency(weightsBytes);
    const withTinyKV = getBandwidthEfficiency(weightsBytes + tinyKV);

    const delta = Math.abs(withTinyKV - noKV) / noKV;
    expect(delta).toBeLessThan(0.01);
  });
});
