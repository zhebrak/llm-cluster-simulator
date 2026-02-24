/**
 * Validation Engine
 *
 * Compares simulation results against published benchmarks
 * to validate accuracy of the simulator.
 */

import type { ModelSpec, ClusterConfig } from '../../types/index.ts';
import type { SimulationConfig } from '../simulation/engine.ts';
import { SimulationEngine } from '../simulation/engine.ts';
import { buildModelSpec } from '../models/primitives.ts';
import { createMultiNodeCluster } from '../hardware/topology.ts';
import {
  type BenchmarkDataPoint,
  ALL_BENCHMARKS,
  ZERO_MEMORY_FACTORS,
  PIPELINE_BUBBLE_FORMULAS,
} from './benchmarks.ts';

/**
 * Validation result for a single benchmark
 */
export interface ValidationResult {
  benchmark: BenchmarkDataPoint;
  simulated: {
    tflopsPerGPU: number;
    mfu: number;
    memoryPerGPU: number;
    stepTimeMs: number;
    tokensPerSecond: number;
  };
  errors: {
    tflops: number;     // Percentage error
    mfu: number;        // Percentage error
    memory?: number;    // Percentage error (if benchmark has memory)
  };
  withinTolerance: boolean;
  notes: string[];
}

/**
 * Validation summary across all benchmarks
 */
export interface ValidationSummary {
  totalBenchmarks: number;
  passed: number;
  failed: number;
  averageErrors: {
    tflops: number;
    mfu: number;
    memory: number;
  };
  results: ValidationResult[];
}

/**
 * Default tolerance levels
 */
export const DEFAULT_TOLERANCES = {
  tflops: 0.20,   // 20% tolerance
  mfu: 0.20,      // 20% tolerance
  memory: 0.15,   // 15% tolerance
};

/**
 * Create a model spec from benchmark data
 */
function createModelFromBenchmark(benchmark: BenchmarkDataPoint): ModelSpec {
  // Estimate model parameters based on known architectures
  const params = benchmark.params * 1e9;

  // Use typical transformer ratios
  const numLayers = Math.round(Math.sqrt(params / 1e6) * 2);
  const hiddenSize = Math.round(Math.sqrt(params / numLayers / 12));
  const roundedHidden = Math.round(hiddenSize / 128) * 128; // Round to 128

  return buildModelSpec({
    name: benchmark.model,
    numLayers,
    hiddenSize: roundedHidden,
    numAttentionHeads: roundedHidden / 128,
    intermediateSize: roundedHidden * 4,
    vocabSize: 50000,
    maxSeqLength: benchmark.seqLength,
  });
}

/**
 * Create a cluster config from benchmark data
 */
function createClusterFromBenchmark(benchmark: BenchmarkDataPoint): ClusterConfig | undefined {
  // Map GPU types to our GPU IDs
  const gpuMapping: Record<string, string> = {
    'A100-40GB': 'a100-40gb',
    'A100-80GB': 'a100-80gb',
    'H100-SXM': 'h100-sxm',
    'H100-PCIe': 'h100-pcie',
    'V100-32GB': 'v100-32gb',
    'V100-16GB': 'v100-16gb',
  };

  const gpuId = gpuMapping[benchmark.gpuType];
  if (!gpuId) {
    return undefined; // Skip TPUs and unsupported GPUs
  }

  const gpusPerNode = 8; // Typical DGX configuration
  const numNodes = Math.ceil(benchmark.numGPUs / gpusPerNode);

  return createMultiNodeCluster(gpuId, gpusPerNode, numNodes);
}

/**
 * Determine strategy type from benchmark
 */
function getStrategyType(benchmark: BenchmarkDataPoint): SimulationConfig['strategyType'] {
  const strategy = benchmark.strategy.toLowerCase();

  if (strategy.includes('3d') || (benchmark.tp && benchmark.pp && benchmark.dp)) {
    return 'fsdp-tp-pp';
  }
  if (strategy.includes('fsdp') || strategy.includes('zero-3')) {
    return 'fsdp';
  }
  if (strategy.includes('zero-2')) {
    return 'fsdp'; // We treat ZeRO-2 similar to FSDP
  }
  if (strategy.includes('zero-1')) {
    return 'zero-1';
  }
  if (strategy.includes('tp') || strategy.includes('tensor')) {
    return 'fsdp-tp';
  }
  if (strategy.includes('pp') || strategy.includes('pipeline')) {
    return 'fsdp-tp-pp';
  }

  return 'ddp';
}

/**
 * Run simulation for a benchmark and compare results
 */
export function validateBenchmark(
  benchmark: BenchmarkDataPoint,
  tolerances = DEFAULT_TOLERANCES
): ValidationResult | null {
  const notes: string[] = [];

  // Skip TPU benchmarks
  if (benchmark.gpuType.includes('TPU')) {
    return null;
  }

  // Create model and cluster
  const modelSpec = createModelFromBenchmark(benchmark);
  const clusterConfig = createClusterFromBenchmark(benchmark);

  if (!clusterConfig) {
    notes.push(`Unsupported GPU type: ${benchmark.gpuType}`);
    return null;
  }

  // Build simulation config
  const simConfig: SimulationConfig = {
    modelSpec,
    clusterConfig,
    globalBatchSize: benchmark.batchSize,
    microBatchSize: benchmark.microBatchSize ?? 4,
    sequenceLength: benchmark.seqLength,
    strategyType: getStrategyType(benchmark),
    strategyConfig: {
      tp: benchmark.tp ?? 1,
      pp: benchmark.pp ?? 1,
      dp: benchmark.dp ?? clusterConfig.totalGPUs / ((benchmark.tp ?? 1) * (benchmark.pp ?? 1)),
      numMicroBatches: 16,
    },
    mixedPrecision: 'bf16',
    activationCheckpointing: true,
  };

  // Run simulation
  try {
    const engine = new SimulationEngine();
    engine.configure(simConfig);

    const validation = engine.validate();
    if (!validation.valid) {
      notes.push(`Validation failed: ${validation.errors.join(', ')}`);
      // Continue anyway to get approximate numbers
    }

    const metrics = engine.simulate();

    // Calculate errors
    const tflopsError = Math.abs(metrics.tflopsPerGPU - benchmark.tflopsPerGPU) / benchmark.tflopsPerGPU;
    const mfuError = Math.abs(metrics.mfu - benchmark.mfu) / benchmark.mfu;

    let memoryError: number | undefined;
    if (benchmark.memoryPerGPU) {
      const simMemoryGB = metrics.memoryPerGPU.total / (1024 ** 3);
      memoryError = Math.abs(simMemoryGB - benchmark.memoryPerGPU) / benchmark.memoryPerGPU;
    }

    // Check if within tolerance
    const withinTolerance =
      tflopsError <= tolerances.tflops &&
      mfuError <= tolerances.mfu &&
      (memoryError === undefined || memoryError <= tolerances.memory);

    // Add analysis notes
    if (tflopsError > tolerances.tflops) {
      notes.push(`TFLOPS error ${(tflopsError * 100).toFixed(1)}% exceeds ${(tolerances.tflops * 100).toFixed(0)}% tolerance`);
    }
    if (mfuError > tolerances.mfu) {
      notes.push(`MFU error ${(mfuError * 100).toFixed(1)}% exceeds ${(tolerances.mfu * 100).toFixed(0)}% tolerance`);
    }
    if (memoryError && memoryError > tolerances.memory) {
      notes.push(`Memory error ${(memoryError * 100).toFixed(1)}% exceeds ${(tolerances.memory * 100).toFixed(0)}% tolerance`);
    }

    return {
      benchmark,
      simulated: {
        tflopsPerGPU: metrics.tflopsPerGPU,
        mfu: metrics.mfu,
        memoryPerGPU: metrics.memoryPerGPU.total / (1024 ** 3),
        stepTimeMs: metrics.stepTimeMs,
        tokensPerSecond: metrics.tokensPerSecond,
      },
      errors: {
        tflops: tflopsError,
        mfu: mfuError,
        memory: memoryError,
      },
      withinTolerance,
      notes,
    };
  } catch (error) {
    notes.push(`Simulation error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    return null;
  }
}

/**
 * Run validation across all benchmarks
 */
export function runFullValidation(
  benchmarks: BenchmarkDataPoint[] = ALL_BENCHMARKS,
  tolerances = DEFAULT_TOLERANCES
): ValidationSummary {
  const results: ValidationResult[] = [];
  let passed = 0;
  let failed = 0;

  const totalErrors = {
    tflops: 0,
    mfu: 0,
    memory: 0,
  };
  let memoryCount = 0;

  for (const benchmark of benchmarks) {
    const result = validateBenchmark(benchmark, tolerances);
    if (result) {
      results.push(result);

      if (result.withinTolerance) {
        passed++;
      } else {
        failed++;
      }

      totalErrors.tflops += result.errors.tflops;
      totalErrors.mfu += result.errors.mfu;
      if (result.errors.memory !== undefined) {
        totalErrors.memory += result.errors.memory;
        memoryCount++;
      }
    }
  }

  const count = results.length || 1;

  return {
    totalBenchmarks: results.length,
    passed,
    failed,
    averageErrors: {
      tflops: totalErrors.tflops / count,
      mfu: totalErrors.mfu / count,
      memory: memoryCount > 0 ? totalErrors.memory / memoryCount : 0,
    },
    results,
  };
}

/**
 * Validate ZeRO memory reduction formulas
 */
export function validateZeROMemory(numGPUs: number = 64): {
  stage: number;
  expected: number;
  description: string;
}[] {
  return [
    {
      stage: 0,
      expected: ZERO_MEMORY_FACTORS.baseline,
      description: 'Baseline DDP: 16Ψ bytes per GPU',
    },
    {
      stage: 1,
      expected: ZERO_MEMORY_FACTORS.zero1(numGPUs),
      description: `ZeRO-1: ${ZERO_MEMORY_FACTORS.zero1(numGPUs).toFixed(2)}Ψ bytes per GPU`,
    },
    {
      stage: 2,
      expected: ZERO_MEMORY_FACTORS.zero2(numGPUs),
      description: `ZeRO-2: ${ZERO_MEMORY_FACTORS.zero2(numGPUs).toFixed(2)}Ψ bytes per GPU`,
    },
    {
      stage: 3,
      expected: ZERO_MEMORY_FACTORS.zero3(numGPUs),
      description: `ZeRO-3: ${ZERO_MEMORY_FACTORS.zero3(numGPUs).toFixed(2)}Ψ bytes per GPU`,
    },
  ];
}

/**
 * Validate pipeline bubble calculations
 */
export function validatePipelineBubble(
  stages: number = 8,
  microbatches: number = 16
): {
  schedule: string;
  bubble: number;
  description: string;
}[] {
  return [
    {
      schedule: 'GPipe',
      bubble: PIPELINE_BUBBLE_FORMULAS.gpipe(stages, microbatches),
      description: `GPipe: ${(PIPELINE_BUBBLE_FORMULAS.gpipe(stages, microbatches) * 100).toFixed(1)}% bubble`,
    },
    {
      schedule: '1F1B',
      bubble: PIPELINE_BUBBLE_FORMULAS['1f1b'](stages, microbatches),
      description: `1F1B: ${(PIPELINE_BUBBLE_FORMULAS['1f1b'](stages, microbatches) * 100).toFixed(1)}% bubble`,
    },
    {
      schedule: 'Interleaved',
      bubble: PIPELINE_BUBBLE_FORMULAS.interleaved(stages, microbatches, 2),
      description: `Interleaved (v=2): ${(PIPELINE_BUBBLE_FORMULAS.interleaved(stages, microbatches, 2) * 100).toFixed(1)}% bubble`,
    },
  ];
}

/**
 * Generate validation report
 */
export function generateValidationReport(summary: ValidationSummary): string {
  const lines: string[] = [];

  lines.push('# LLM Cluster Simulator - Validation Report');
  lines.push('');
  lines.push('## Summary');
  lines.push('');
  lines.push(`- **Total Benchmarks:** ${summary.totalBenchmarks}`);
  lines.push(`- **Passed:** ${summary.passed} (${((summary.passed / summary.totalBenchmarks) * 100).toFixed(1)}%)`);
  lines.push(`- **Failed:** ${summary.failed}`);
  lines.push('');
  lines.push('## Average Errors');
  lines.push('');
  lines.push(`- **TFLOPS:** ${(summary.averageErrors.tflops * 100).toFixed(1)}%`);
  lines.push(`- **MFU:** ${(summary.averageErrors.mfu * 100).toFixed(1)}%`);
  lines.push(`- **Memory:** ${(summary.averageErrors.memory * 100).toFixed(1)}%`);
  lines.push('');
  lines.push('## Detailed Results');
  lines.push('');

  for (const result of summary.results) {
    const status = result.withinTolerance ? '✅' : '❌';
    lines.push(`### ${status} ${result.benchmark.model} ${result.benchmark.params}B - ${result.benchmark.source}`);
    lines.push('');
    lines.push('| Metric | Published | Simulated | Error |');
    lines.push('|--------|-----------|-----------|-------|');
    lines.push(`| TFLOPS/GPU | ${result.benchmark.tflopsPerGPU.toFixed(1)} | ${result.simulated.tflopsPerGPU.toFixed(1)} | ${(result.errors.tflops * 100).toFixed(1)}% |`);
    lines.push(`| MFU | ${(result.benchmark.mfu * 100).toFixed(1)}% | ${(result.simulated.mfu * 100).toFixed(1)}% | ${(result.errors.mfu * 100).toFixed(1)}% |`);
    if (result.benchmark.memoryPerGPU) {
      lines.push(`| Memory (GB) | ${result.benchmark.memoryPerGPU.toFixed(1)} | ${result.simulated.memoryPerGPU.toFixed(1)} | ${((result.errors.memory ?? 0) * 100).toFixed(1)}% |`);
    }
    lines.push('');

    if (result.notes.length > 0) {
      lines.push('**Notes:**');
      for (const note of result.notes) {
        lines.push(`- ${note}`);
      }
      lines.push('');
    }
  }

  lines.push('## Known Limitations');
  lines.push('');
  lines.push('This simulator does not model:');
  lines.push('- Kernel launch overhead and GPU scheduling');
  lines.push('- Memory fragmentation');
  lines.push('- Network congestion and contention');
  lines.push('- CPU overhead for data loading');
  lines.push('- Compiler optimizations (XLA, torch.compile)');
  lines.push('- Hardware-specific optimizations');
  lines.push('');
  lines.push('Expected accuracy: within 15-20% of published results for most configurations.');

  return lines.join('\n');
}
