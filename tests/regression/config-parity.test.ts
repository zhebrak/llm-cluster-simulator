/**
 * Config Parity Test
 *
 * Verifies that the canonical published training configs match the
 * corresponding UI demo presets on all training-relevant fields.
 *
 * This catches drift between the canonical source of truth
 * (tests/fixtures/published-training-configs.ts) and the demo presets
 * that users interact with in the UI (src/stores/config.ts).
 */

import { describe, it, expect } from 'vitest';
import { PUBLISHED_TRAINING_CONFIGS } from '../../src/data/published-training-configs.ts';
import { DEMO_PRESETS } from '../../src/stores/config.ts';

// Mapping: canonical config ID → demo preset slug
const PARITY_MAP: [string, string][] = [
  ['gpt3-175b', 'gpt3-175b'],
  ['llama3-405b-8k', 'llama3-405b'],
  ['nemotron-4-340b', 'nemotron-4-340b'],
  ['deepseek-v3-fp8-h800', 'deepseek-v3-r1'],
];

describe('Config Parity: Published Configs ↔ Demo Presets', () => {
  for (const [canonicalId, presetSlug] of PARITY_MAP) {
    it(`${canonicalId} matches demo preset "${presetSlug}"`, () => {
      const canonical = PUBLISHED_TRAINING_CONFIGS.get(canonicalId);
      expect(canonical, `Canonical config "${canonicalId}" not found`).toBeDefined();

      const preset = DEMO_PRESETS.find(p => p.slug === presetSlug);
      expect(preset, `Demo preset "${presetSlug}" not found`).toBeDefined();

      const pc = canonical!;
      const dp = preset!;

      // Model and sequence length
      expect(pc.modelId, 'modelId').toBe(dp.modelId);
      expect(pc.sequenceLength, 'sequenceLength').toBe(dp.sequenceLength);

      // Strategy type
      expect(pc.strategyType, 'strategyType').toBe(dp.strategyType);

      // Parallelism degrees
      const totalGPUs = pc.gpusPerNode * pc.numNodes;
      const tp = pc.strategyConfig?.tp ?? 1;
      const pp = pc.strategyConfig?.pp ?? 1;
      const cp = pc.strategyConfig?.cp ?? 1;
      const ep = pc.strategyConfig?.ep ?? 1;
      const dp_degree = Math.floor(totalGPUs / (tp * pp * cp));

      expect(tp, 'tp').toBe(dp.tpDegree);
      expect(pp, 'pp').toBe(dp.ppDegree);
      expect(dp_degree, 'dp').toBe(dp.dpDegree);
      expect(ep, 'ep').toBe(dp.epDegree);

      // Batch sizes
      expect(pc.globalBatchSize, 'globalBatchSize').toBe(dp.globalBatchSize);
      expect(pc.microBatchSize, 'microBatchSize').toBe(dp.microBatchSize);

      // Activation checkpointing
      expect(pc.activationCheckpointing, 'activationCheckpointing').toBe(dp.activationCheckpointing);
      if (pc.checkpointingGranularity) {
        expect(pc.checkpointingGranularity, 'checkpointingGranularity').toBe(
          dp.checkpointingGranularity ?? 'full',
        );
      }

      // Flash attention
      expect(pc.flashAttention, 'flashAttention').toBe(dp.flashAttention);

      // Mixed precision
      expect(pc.mixedPrecision ?? 'bf16', 'mixedPrecision').toBe(dp.precision);

      // Sequence parallel
      const canonicalSP = pc.strategyConfig?.sequenceParallel ?? false;
      expect(canonicalSP, 'sequenceParallel').toBe(dp.sequenceParallel);

      // Pipeline schedule
      const canonicalPS = pc.strategyConfig?.pipelineSchedule ?? '1f1b';
      expect(canonicalPS, 'pipelineSchedule').toBe(dp.pipelineSchedule);

      // Interleaved stages (only relevant for interleaved schedules)
      if (canonicalPS === 'interleaved-1f1b' || dp.pipelineSchedule === 'interleaved-1f1b') {
        const canonicalIS = pc.strategyConfig?.interleavedStages ?? 2;
        expect(canonicalIS, 'interleavedStages').toBe(dp.interleavedStages);
      }
    });
  }
});
