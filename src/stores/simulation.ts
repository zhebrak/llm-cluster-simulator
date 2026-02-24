/**
 * Simulation state store
 * Supports both training and inference simulation modes
 */

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import type {
  SimulationResult,
  SimulationEvent,
  SimulationStatus,
  InferenceSimulationResult,
  InferenceEvent,
} from '../types/index.ts';
import type { SimulationMetrics, SimulationConfig, ChangelogEntry } from '../core/simulation/index.ts';
import { SimulationEngine, optimizeTraining } from '../core/simulation/index.ts';
import { InferenceSimulationEngine, optimizeInference } from '../core/inference/index.ts';
import type { InferenceChangelogEntry } from '../core/inference/index.ts';
import type { ParetoSweepResult, SeqLenSweepResult } from '../core/inference/index.ts';
import { runSeqLenSweep } from '../core/inference/index.ts';
import { runParetoSweep, buildSweepKey } from '../core/inference/index.ts';
import { getGPUHourlyRate } from '../core/cost/index.ts';
import { useConfigStore } from './config.ts';

/**
 * Training simulation metrics (from SimulationMetrics)
 */
export interface TrainingSimulationState {
  result: SimulationResult | null;
  metrics: SimulationMetrics | null;
  events: SimulationEvent[];
  // Config snapshot at run time (results should not change when config changes)
  configSnapshot: {
    modelId: string;
    clusterId: string;
    gpuId: string;
    gpuName: string;
    numGPUs: number;
    gpusPerNode: number;
    // Batch
    globalBatchSize: number;
    microBatchSize: number;
    numMicroBatches: number;
    sequenceLength: number;
    gradientAccumulationSteps: number;
    // Parallelism
    strategyType: string;
    tpDegree: number;
    ppDegree: number;
    dpDegree: number;
    epDegree: number;
    cpDegree: number;
    pipelineSchedule: '1f1b' | 'interleaved-1f1b' | 'dualpipe-v';
    interleavedStages: number;
    // Optimizations
    precision: string;
    activationCheckpointing: boolean;
    checkpointingGranularity: 'full' | 'selective';
    sequenceParallel: boolean;
    flashAttention: boolean;
    // Training scale
    trainingGoal: string;
    targetTokens: number;
    maxSteps: number;
    // Fine-tuning
    finetuningMethod: string;
    loraRank?: number;
    loraTargetModules?: string;
    // Model
    modelParams: number;
    activeParams: number;  // For Chinchilla scaling (active params for MoE)
    isMoE: boolean;
  } | null;
}

/**
 * Inference simulation metrics
 */
export interface InferenceSimulationState {
  result: InferenceSimulationResult | null;
  events: InferenceEvent[];
  // Config snapshot at run time (charts should not change when sidebar changes)
  tensorParallel: number;
  expertParallel: number;
  continuousBatching: boolean;
  batchSize: number;
  inputSeqLen: number;
  // Pareto sweep
  paretoResult: ParetoSweepResult | null;
  paretoProgress: number;  // 0-1
  paretoSweepKey: string;
  // Sequence length sweep
  seqLenSweepResult: SeqLenSweepResult | null;
}

export interface OptimizationSummary {
  changelog: ChangelogEntry[] | InferenceChangelogEntry[];
  beforeMetric: number;
  afterMetric: number;
  totalSimulations: number;
  target: 'training' | 'throughput' | 'latency';
}

export interface SimulationState {
  // Status
  status: SimulationStatus;
  error: string | null;
  runCounter: number;

  // Training results
  training: TrainingSimulationState;

  // Inference results
  inference: InferenceSimulationState;

  // Optimization
  isOptimizing: boolean;
  optimizationResult: OptimizationSummary | null;

  // Playback
  playbackSpeed: number;
  currentTime: number;
  isPlaying: boolean;

  // Legacy compatibility
  result: SimulationResult | null;
  metrics: SimulationMetrics | null;
  events: SimulationEvent[];

  // Actions
  runSimulation: () => Promise<void>;
  runTrainingSimulation: () => Promise<void>;
  runInferenceSimulation: () => Promise<void>;
  autoOptimizeTraining: () => Promise<void>;
  autoOptimizeInference: (target: 'throughput' | 'latency') => Promise<void>;
  setPlaybackSpeed: (speed: number) => void;
  setCurrentTime: (time: number) => void;
  togglePlayback: () => void;
  reset: () => void;
}

// Module-level abort controller for Pareto sweep (outside immer state)
let paretoAbortController: AbortController | null = null;

export const useSimulationStore = create<SimulationState>()(
  immer((set, get) => ({
    status: 'idle',
    error: null,
    runCounter: 0,

    training: {
      result: null,
      metrics: null,
      events: [],
      configSnapshot: null,
    },

    inference: {
      result: null,
      events: [],
      tensorParallel: 1,
      expertParallel: 1,
      continuousBatching: false,
      batchSize: 1,
      inputSeqLen: 1024,
      paretoResult: null,
      paretoProgress: 0,
      paretoSweepKey: '',
      seqLenSweepResult: null,
    },

    isOptimizing: false,
    optimizationResult: null,

    playbackSpeed: 1,
    currentTime: 0,
    isPlaying: false,

    // Legacy
    result: null,
    metrics: null,
    events: [],

    runSimulation: async () => {
      const config = useConfigStore.getState();

      if (config.mode === 'training') {
        await get().runTrainingSimulation();
      } else {
        await get().runInferenceSimulation();
      }
    },

    runTrainingSimulation: async () => {
      const config = useConfigStore.getState();

      // Validate configuration
      if (!config.modelSpec || !config.clusterConfig) {
        set(state => {
          state.status = 'error';
          state.error = 'Model or cluster not configured';
        });
        return;
      }

      set(state => {
        state.status = 'running';
        state.runCounter++;
        state.error = null;
      });

      try {
        // Build simulation config from store state
        // maxSteps is computed from targetTokens / (globalBatchSize * sequenceLength)
        const simConfig: SimulationConfig = {
          modelSpec: config.modelSpec,
          clusterConfig: config.clusterConfig,
          globalBatchSize: config.training.globalBatchSize,
          microBatchSize: config.training.microBatchSize,
          sequenceLength: config.sequenceLength,
          gradientAccumulationSteps: config.training.gradientAccumulationSteps,
          maxSteps: Math.ceil(config.training.targetTokens / (config.training.globalBatchSize * config.sequenceLength)),
          strategyType: config.training.strategyType,
          strategyConfig: {
            tp: config.training.tpDegree,
            pp: config.training.ppDegree,
            dp: config.training.dpDegree,
            ep: config.training.epDegree,
            cp: config.training.cpDegree,
            cpImplementation: config.training.cpImplementation,
            numMicroBatches: config.training.numMicroBatches,
            sequenceParallel: config.training.sequenceParallel,
            pipelineSchedule: config.training.pipelineSchedule,
            interleavedStages: config.training.interleavedStages,
          },
          activationCheckpointing: config.training.activationCheckpointing,
          checkpointingGranularity: config.training.checkpointingGranularity,
          selectiveStoredLayers: config.training.selectiveStoredLayers !== 'auto' ? config.training.selectiveStoredLayers : undefined,
          flashAttention: config.training.flashAttention,
          mixedPrecision: config.precision,
          finetuningMethod: config.training.finetuningMethod,
          loraRank: config.training.finetuningMethod !== 'full' ? config.training.loraRank : undefined,
          loraTargetModules: config.training.finetuningMethod !== 'full' ? config.training.loraTargetModules : undefined,
        };

        // Run simulation
        const engine = new SimulationEngine();
        engine.configure(simConfig);

        const validation = engine.validate();
        if (!validation.valid) {
          const result = engine.run();
          set(state => {
            state.status = 'error';
            state.error = validation.errors.join('; ');
            state.training.result = result;
            state.result = result;
          });
          return;
        }

        const result = engine.run();
        const metrics = engine.simulate();

        set(state => {
          state.status = result.success ? 'complete' : 'error';
          state.training.result = result;
          state.training.metrics = metrics;
          state.training.events = result.events;
          state.training.configSnapshot = {
            modelId: config.modelId,
            clusterId: config.clusterId,
            gpuId: config.gpuId,
            gpuName: config.clusterConfig?.node.gpu.name ?? '',
            numGPUs: config.numGPUs,
            gpusPerNode: config.gpusPerNode,
            // Batch
            globalBatchSize: config.training.globalBatchSize,
            microBatchSize: config.training.microBatchSize,
            numMicroBatches: config.training.numMicroBatches,
            sequenceLength: config.sequenceLength,
            gradientAccumulationSteps: config.training.gradientAccumulationSteps,
            // Parallelism
            strategyType: config.training.strategyType,
            tpDegree: config.training.tpDegree,
            ppDegree: config.training.ppDegree,
            dpDegree: config.training.dpDegree,
            epDegree: config.training.epDegree,
            cpDegree: config.training.cpDegree,
            pipelineSchedule: config.training.pipelineSchedule,
            interleavedStages: config.training.interleavedStages,
            // Optimizations
            precision: config.precision,
            activationCheckpointing: config.training.activationCheckpointing,
            checkpointingGranularity: config.training.checkpointingGranularity,
            sequenceParallel: config.training.sequenceParallel,
            flashAttention: config.training.flashAttention,
            // Fine-tuning
            finetuningMethod: config.training.finetuningMethod,
            ...(config.training.finetuningMethod !== 'full' ? {
              loraRank: config.training.loraRank,
              loraTargetModules: config.training.loraTargetModules,
            } : {}),
            // Training scale
            trainingGoal: config.training.trainingGoal,
            targetTokens: config.training.targetTokens,
            maxSteps: Math.ceil(config.training.targetTokens / (config.training.globalBatchSize * config.sequenceLength)),
            // Model
            modelParams: config.modelSpec!.totalParams,
            activeParams: config.modelSpec!.activeParams ?? config.modelSpec!.totalParams,
            isMoE: config.modelSpec!.isMoE ?? false,
          };
          // Legacy
          state.result = result;
          state.metrics = metrics;
          state.events = result.events;
          state.error = result.success ? null : result.state.error ?? 'Simulation failed';
          state.currentTime = 0;
          state.isPlaying = false;
        });
      } catch (e) {
        set(state => {
          state.status = 'error';
          state.error = e instanceof Error ? e.message : 'Unknown error';
        });
      }
    },

    runInferenceSimulation: async () => {
      const config = useConfigStore.getState();

      // Validate configuration
      if (!config.modelSpec || !config.clusterConfig) {
        set(state => {
          state.status = 'error';
          state.error = 'Model or cluster not configured';
        });
        return;
      }

      // Abort any in-flight Pareto sweep
      paretoAbortController?.abort();
      paretoAbortController = null;

      set(state => {
        state.status = 'running';
        state.runCounter++;
        state.error = null;
      });

      try {
        const engine = new InferenceSimulationEngine();

        engine.configure({
          modelSpec: config.modelSpec,
          gpu: config.clusterConfig.node.gpu,
          numGPUs: config.numGPUs,
          batchSize: config.inference.batchSize,
          inputSeqLen: config.inference.inputSeqLen,
          outputSeqLen: config.inference.outputSeqLen,
          weightPrecision: config.inference.weightPrecision,
          kvCachePrecision: config.inference.kvCachePrecision,
          flashAttention: config.inference.flashAttention,
          pagedAttention: config.inference.pagedAttention,
          continuousBatching: config.inference.continuousBatching,
          tensorParallel: config.inference.tensorParallel,
          expertParallel: config.inference.expertParallel,
          speculativeEnabled: config.inference.speculativeDecoding,
          draftModelSpec: config.inference.draftModelSpec ?? undefined,
          numSpeculativeTokens: config.inference.numSpeculativeTokens,
          acceptanceRate: config.inference.acceptanceRate,
        });

        const result = engine.run();

        // Compute seq-len sweep synchronously (lightweight: ~45 sims)
        const seqLenSweep = runSeqLenSweep(
          {
            modelSpec: config.modelSpec,
            gpu: config.clusterConfig.node.gpu,
            numGPUs: config.numGPUs,
            batchSize: config.inference.batchSize,
            outputSeqLen: config.inference.outputSeqLen,
            flashAttention: config.inference.flashAttention,
            pagedAttention: config.inference.pagedAttention,
            tensorParallel: config.inference.tensorParallel,
            expertParallel: config.inference.expertParallel,
            continuousBatching: config.inference.continuousBatching,
          },
          config.modelSpec,
          config.clusterConfig.node.gpu,
        );

        set(state => {
          state.status = result.success ? 'complete' : 'error';
          state.inference.result = result;
          state.inference.events = result.events;
          state.inference.tensorParallel = config.inference.tensorParallel ?? 1;
          state.inference.expertParallel = config.inference.expertParallel ?? 1;
          state.inference.continuousBatching = config.inference.continuousBatching ?? false;
          state.inference.batchSize = config.inference.batchSize ?? 1;
          state.inference.inputSeqLen = config.inference.inputSeqLen ?? 1024;
          state.inference.seqLenSweepResult = seqLenSweep;
          state.error = result.success ? null : result.errors.join('; ');
          state.currentTime = 0;
          state.isPlaying = false;
        });

        // Kick off background Pareto sweep (non-blocking)
        const newSweepKey = buildSweepKey(config.modelId, config.gpuId, config.numGPUs, config.inference.inputSeqLen, config.inference.outputSeqLen, config.inference.kvCachePrecision);
        const currentSweepKey = get().inference.paretoSweepKey;

        // Skip sweep if the envelope hasn't changed (same model/GPU/numGPUs)
        if (newSweepKey === currentSweepKey && get().inference.paretoResult) {
          return;
        }

        const gpu = config.clusterConfig.node.gpu;
        const customPrice = config.pricePerGPUHour;
        const gpuPricing = getGPUHourlyRate(config.gpuId || '');
        const effectiveRate = customPrice ?? gpuPricing.rate;

        // Build base config for sweep (uses modelId/gpuId so candidates resolve internally)
        const baseConfig = {
          modelId: config.modelId,
          modelSpec: config.modelSpec,
          gpuId: config.gpuId,
          gpu,
          numGPUs: config.numGPUs,
          inputSeqLen: config.inference.inputSeqLen,
          outputSeqLen: config.inference.outputSeqLen,
          kvCachePrecision: config.inference.kvCachePrecision,
          flashAttention: config.inference.flashAttention,
          pagedAttention: config.inference.pagedAttention,
        };

        const controller = new AbortController();
        paretoAbortController = controller;

        set(state => {
          state.inference.paretoProgress = 0;
        });

        // Fire and forget — the sweep updates state as it progresses
        runParetoSweep(
          baseConfig,
          config.modelSpec,
          gpu,
          effectiveRate,
          config.numGPUs,
          (fraction) => {
            set(state => {
              state.inference.paretoProgress = fraction;
            });
          },
          controller.signal,
        ).then((sweepResult) => {
          if (sweepResult) {
            set(state => {
              state.inference.paretoResult = sweepResult;
              state.inference.paretoProgress = 1;
              state.inference.paretoSweepKey = newSweepKey;
            });
          }
        }).catch(() => {
          // Sweep failed — silently ignore, the main sim already completed
        });
      } catch (e) {
        set(state => {
          state.status = 'error';
          state.error = e instanceof Error ? e.message : 'Unknown error';
        });
      }
    },

    autoOptimizeTraining: async () => {
      const config = useConfigStore.getState();

      if (!config.modelSpec || !config.clusterConfig) {
        set(state => {
          state.status = 'error';
          state.error = 'Model or cluster not configured';
        });
        return;
      }

      set(state => {
        state.isOptimizing = true;
        state.status = 'running';
        state.runCounter++;
        state.error = null;
        state.optimizationResult = null;
      });

      // Yield a frame so React can render the running state (animation trigger)
      await new Promise(r => setTimeout(r, 0));

      try {
        // Build simulation config (same as runTrainingSimulation)
        const simConfig: SimulationConfig = {
          modelSpec: config.modelSpec,
          clusterConfig: config.clusterConfig,
          globalBatchSize: config.training.globalBatchSize,
          microBatchSize: config.training.microBatchSize,
          sequenceLength: config.sequenceLength,
          gradientAccumulationSteps: config.training.gradientAccumulationSteps,
          maxSteps: Math.ceil(config.training.targetTokens / (config.training.globalBatchSize * config.sequenceLength)),
          strategyType: config.training.strategyType,
          strategyConfig: {
            tp: config.training.tpDegree,
            pp: config.training.ppDegree,
            dp: config.training.dpDegree,
            ep: config.training.epDegree,
            cp: config.training.cpDegree,
            cpImplementation: config.training.cpImplementation,
            numMicroBatches: config.training.numMicroBatches,
            sequenceParallel: config.training.sequenceParallel,
            pipelineSchedule: config.training.pipelineSchedule,
            interleavedStages: config.training.interleavedStages,
          },
          activationCheckpointing: config.training.activationCheckpointing,
          checkpointingGranularity: config.training.checkpointingGranularity,
          selectiveStoredLayers: config.training.selectiveStoredLayers !== 'auto' ? config.training.selectiveStoredLayers : undefined,
          flashAttention: config.training.flashAttention,
          mixedPrecision: config.precision,
          finetuningMethod: config.training.finetuningMethod,
          loraRank: config.training.finetuningMethod !== 'full' ? config.training.loraRank : undefined,
          loraTargetModules: config.training.finetuningMethod !== 'full' ? config.training.loraTargetModules : undefined,
        };

        // Yield to UI before heavy compute
        await new Promise(r => setTimeout(r, 10));

        const result = optimizeTraining(
          simConfig,
          config.training.targetTokens,
          config.sequenceLength,
        );

        // Apply optimized config back to config store
        const opt = result.optimizedConfig;
        const osc = opt.strategyConfig ?? {};
        const configStore = useConfigStore.getState();

        // Apply strategy first (this resets parallelism degrees)
        if (opt.strategyType !== config.training.strategyType) {
          configStore.setStrategy(opt.strategyType);
        }

        // Apply parallelism params
        configStore.setStrategyParams({
          tpDegree: osc.tp ?? 1,
          ppDegree: osc.pp ?? 1,
          epDegree: osc.ep ?? 1,
          cpDegree: osc.cp ?? 1,
          sequenceParallel: osc.sequenceParallel ?? false,
          pipelineSchedule: osc.pipelineSchedule ?? '1f1b',
          interleavedStages: osc.interleavedStages ?? 1,
        });

        // Apply training params
        configStore.setTrainingParams({
          globalBatchSize: opt.globalBatchSize,
          microBatchSize: opt.microBatchSize,
          activationCheckpointing: opt.activationCheckpointing ?? true,
          checkpointingGranularity: opt.checkpointingGranularity ?? 'full',
          flashAttention: opt.flashAttention ?? true,
        });

        // Apply precision
        if (opt.mixedPrecision && opt.mixedPrecision !== config.precision) {
          configStore.setPrecision(opt.mixedPrecision);
        }

        // Store optimization summary
        set(state => {
          state.optimizationResult = {
            changelog: result.changelog,
            beforeMetric: result.beforeMetric,
            afterMetric: result.afterMetric,
            totalSimulations: result.totalSimulations,
            target: 'training',
          };
          state.isOptimizing = false;
        });

        // Run simulation with optimized config
        await get().runTrainingSimulation();
      } catch (e) {
        set(state => {
          state.isOptimizing = false;
          state.status = 'error';
          state.error = e instanceof Error ? e.message : 'Optimization failed';
        });
      }
    },

    autoOptimizeInference: async (target: 'throughput' | 'latency') => {
      const config = useConfigStore.getState();

      if (!config.modelSpec || !config.clusterConfig) {
        set(state => {
          state.status = 'error';
          state.error = 'Model or cluster not configured';
        });
        return;
      }

      set(state => {
        state.isOptimizing = true;
        state.status = 'running';
        state.runCounter++;
        state.error = null;
        state.optimizationResult = null;
      });

      // Yield a frame so React can render the running state (animation trigger)
      await new Promise(r => setTimeout(r, 0));

      try {
        const infConfig = {
          modelSpec: config.modelSpec,
          gpu: config.clusterConfig.node.gpu,
          numGPUs: config.numGPUs,
          batchSize: config.inference.batchSize,
          inputSeqLen: config.inference.inputSeqLen,
          outputSeqLen: config.inference.outputSeqLen,
          weightPrecision: config.inference.weightPrecision,
          kvCachePrecision: config.inference.kvCachePrecision,
          flashAttention: config.inference.flashAttention,
          pagedAttention: config.inference.pagedAttention,
          continuousBatching: config.inference.continuousBatching,
          tensorParallel: config.inference.tensorParallel,
          expertParallel: config.inference.expertParallel,
          speculativeEnabled: config.inference.speculativeDecoding,
          draftModelSpec: config.inference.draftModelSpec ?? undefined,
          numSpeculativeTokens: config.inference.numSpeculativeTokens,
          acceptanceRate: config.inference.acceptanceRate,
        };

        // Yield to UI before heavy compute
        await new Promise(r => setTimeout(r, 10));

        const result = optimizeInference(infConfig, target);

        // Apply optimized config back to config store
        const opt = result.optimizedConfig;
        const configStore = useConfigStore.getState();

        configStore.setInferenceParams({
          tensorParallel: opt.tensorParallel ?? 1,
          expertParallel: opt.expertParallel ?? 1,
          batchSize: opt.batchSize ?? 1,
          weightPrecision: opt.weightPrecision ?? 'bf16',
          kvCachePrecision: opt.kvCachePrecision ?? 'bf16',
          continuousBatching: opt.continuousBatching ?? false,
          flashAttention: opt.flashAttention ?? true,
        });

        set(state => {
          state.optimizationResult = {
            changelog: result.changelog,
            beforeMetric: result.beforeMetric,
            afterMetric: result.afterMetric,
            totalSimulations: result.totalSimulations,
            target,
          };
          state.isOptimizing = false;
        });

        // Run simulation with optimized config
        await get().runInferenceSimulation();
      } catch (e) {
        set(state => {
          state.isOptimizing = false;
          state.status = 'error';
          state.error = e instanceof Error ? e.message : 'Optimization failed';
        });
      }
    },

    setPlaybackSpeed: (speed: number) => {
      set(state => {
        state.playbackSpeed = speed;
      });
    },

    setCurrentTime: (time: number) => {
      set(state => {
        state.currentTime = time;
      });
    },

    togglePlayback: () => {
      set(state => {
        state.isPlaying = !state.isPlaying;
      });
    },

    reset: () => {
      paretoAbortController?.abort();
      paretoAbortController = null;
      set(state => {
        state.status = 'idle';
        state.error = null;
        state.training = { result: null, metrics: null, events: [], configSnapshot: null };
        state.inference = { result: null, events: [], tensorParallel: 1, expertParallel: 1, continuousBatching: false, batchSize: 1, inputSeqLen: 1024, paretoResult: null, paretoProgress: 0, paretoSweepKey: '', seqLenSweepResult: null };
        state.isOptimizing = false;
        state.optimizationResult = null;
        state.result = null;
        state.metrics = null;
        state.events = [];
        state.currentTime = 0;
        state.isPlaying = false;
      });
    },
  }))
);
