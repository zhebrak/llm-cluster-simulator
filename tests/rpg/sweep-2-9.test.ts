/**
 * Sweep: Mission 2-9 config space
 *
 * Initial: TP=8 PP=1 DP=2, 16 H100s (2 nodes, 8/node), LLaMA 70B
 * Constraint: totalGPUs = TP * PP * DP = 16, gpusPerNode=8
 *
 * Enumerate all valid (TP, PP) combos where TP*PP divides 16 and DP≥1.
 * Report MFU, memUtil, success, and whether PP crosses nodes.
 */

import { describe, it } from 'vitest';
import {
  SimulationEngine,
  type SimulationConfig,
} from '../../src/core/simulation/engine.ts';
import {
  createMultiNodeCluster,
  createSingleNodeCluster,
} from '../../src/core/hardware/index.ts';

const NUM_GPUS = 16;
const GPUS_PER_NODE = 8;

function makeCluster(gpuId: string, numGPUs: number, gpusPerNode: number) {
  const numNodes = Math.ceil(numGPUs / gpusPerNode);
  if (numNodes === 1) return createSingleNodeCluster(gpuId, numGPUs);
  return createMultiNodeCluster(gpuId, gpusPerNode, numNodes);
}

function run(tp: number, pp: number) {
  const dp = NUM_GPUS / (tp * pp);
  const cluster = makeCluster('h100-sxm', NUM_GPUS, GPUS_PER_NODE);
  const engine = new SimulationEngine();

  const config: SimulationConfig = {
    modelId: 'llama3.3-70b',
    clusterConfig: cluster,
    sequenceLength: 4096,
    globalBatchSize: 64,
    microBatchSize: 1,
    strategyType: 'fsdp-tp-pp',
    strategyConfig: {
      tp,
      pp,
      sequenceParallel: true,
    },
    activationCheckpointing: true,
    checkpointingGranularity: 'full',
    flashAttention: true,
    mixedPrecision: 'bf16',
  };
  engine.configure(config);
  return engine.simulate();
}

describe('Mission 2-9 config sweep', () => {
  it('sweep all valid TP×PP combos', () => {
    const tpValues = [1, 2, 4, 8];
    const ppValues = [1, 2, 4, 8, 16];

    console.log('\n' + '='.repeat(100));
    console.log('Mission 2-9 sweep: LLaMA 70B, 16× H100-SXM (2 nodes × 8)');
    console.log('Setup: fsdp-tp-pp, bf16, GBS=64, MBS=1, AC=full, FA=on, SP=on');
    console.log('Winning criteria: success=true AND MFU > 39%');
    console.log('='.repeat(100));
    console.log(
      'TP'.padStart(4),
      'PP'.padStart(4),
      'DP'.padStart(4),
      'GA'.padStart(4),
      'MFU%'.padStart(7),
      'HFU%'.padStart(7),
      'memUtil'.padStart(8),
      'OOM?'.padStart(6),
      'stepMs'.padStart(8),
      'tok/s'.padStart(8),
      'bubble%'.padStart(9),
      'commOH%'.padStart(9),
      'PP-xNode?'.padStart(11),
      'DP-xNode?'.padStart(11),
      'WINS?'.padStart(7),
    );
    console.log('-'.repeat(100));

    for (const tp of tpValues) {
      for (const pp of ppValues) {
        if (tp * pp > NUM_GPUS) continue;
        const dp = NUM_GPUS / (tp * pp);
        if (!Number.isInteger(dp) || dp < 1) continue;

        // PP crosses nodes if PP stages span more than one node
        const gpusPerPPGroup = tp * pp; // GPUs in one PP pipeline
        const ppCrossesNode = gpusPerPPGroup > GPUS_PER_NODE;
        // DP crosses nodes if there's more than one node in a DP group
        const dpCrossesNode = dp > 1 && (tp * pp) <= GPUS_PER_NODE;
        // Actually: DP ranks are on different nodes whenever numNodes > 1 and dp > 1
        const dpAcrossNodes = dp > 1 && NUM_GPUS / GPUS_PER_NODE > 1;

        try {
          const m = run(tp, pp);
          const ga = Math.ceil(64 / (1 * dp));
          const success = m.memoryUtilization <= 1.0;
          const wins = success && m.mfu > 0.39;

          console.log(
            String(tp).padStart(4),
            String(pp).padStart(4),
            String(dp).padStart(4),
            String(ga).padStart(4),
            (m.mfu * 100).toFixed(1).padStart(7),
            (m.hfu * 100).toFixed(1).padStart(7),
            m.memoryUtilization.toFixed(3).padStart(8),
            (success ? '  OK' : ' OOM').padStart(6),
            m.stepTimeMs.toFixed(0).padStart(8),
            m.tokensPerSecond.toFixed(0).padStart(8),
            (m.pipelineBubble * 100).toFixed(1).padStart(9),
            (m.communicationOverhead * 100).toFixed(1).padStart(9),
            (ppCrossesNode ? 'YES' : 'no').padStart(11),
            (dpAcrossNodes ? 'YES' : 'no').padStart(11),
            (wins ? '<<WIN>>' : '').padStart(7),
          );
        } catch (e) {
          console.log(
            String(tp).padStart(4),
            String(pp).padStart(4),
            String(dp).padStart(4),
            '  -'.padStart(4),
            'ERROR'.padStart(7),
            String(e instanceof Error ? e.message : e).substring(0, 40),
          );
        }
      }
    }
    console.log('='.repeat(100));
  });
});
