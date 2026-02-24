/**
 * Type exports for distributed training simulator
 */

// Base types
export type {
  DType,
  TensorSpec,
  MemoryBreakdown,
  ComputeBreakdown,
  CommunicationBreakdown,
  TimingBreakdown,
  ByteUnit,
  FlopUnit,
  TimeUnit,
} from './base.ts';

export {
  DTYPE_BYTES,
  tensorBytes,
  formatBytes,
  formatFlops,
  formatTime,
  formatNumber,
} from './base.ts';

// Model types
export type {
  AttentionType,
  LayerType,
  LayerSpec,
  EmbeddingLayerSpec,
  AttentionLayerSpec,
  MLPLayerSpec,
  NormLayerSpec,
  MoELayerSpec,
  OutputLayerSpec,
  AnyLayerSpec,
  TransformerBlockSpec,
  ModelSpec,
  ModelConfig,
  DTypeConfig,
} from './model.ts';

export {
  DEFAULT_DTYPE_CONFIG,
  DTYPE_PRESETS,
} from './model.ts';

// Hardware types
export type {
  GPUVendor,
  GPUArchitecture,
  GPUSpec,
  InterconnectType,
  InterconnectSpec,
  InfiniBandSpec,
  NodeSpec,
  TopologyType,
  ClusterConfig,
  GPUState,
  GPUMetrics,
  ClusterMetrics,
} from './hardware.ts';

export {
  NVLINK_SPECS,
  INFINIBAND_SPECS,
} from './hardware.ts';

// Strategy types
export type {
  ParallelismDimension,
  StrategyType,
  PipelineSchedule,
  DataParallelConfig,
  TensorParallelConfig,
  PipelineParallelConfig,
  ContextParallelConfig,
  ExpertParallelConfig,
  ActivationCheckpointConfig,
  StrategyConfig,
  StrategyValidation,
  StrategyAnalysis,
  StrategyComparison,
} from './strategy.ts';

export {
  DEFAULT_DP_CONFIG,
  DEFAULT_TP_CONFIG,
  DEFAULT_PP_CONFIG,
  DEFAULT_CP_CONFIG,
  DEFAULT_EP_CONFIG,
  DEFAULT_ACTIVATION_CHECKPOINT_CONFIG,
} from './strategy.ts';

// Training types
export type {
  OptimizerType,
  LRSchedulerType,
  OptimizerConfig,
  LRScheduleConfig,
  TrainingHyperparams,
  TrainingConfig,
  TrainingRunStats,
  ChinchillaOptimal,
  TrainingCostProjection,
} from './training.ts';

export {
  DEFAULT_ADAMW_CONFIG,
  DEFAULT_LR_SCHEDULE,
  getOptimizerStateMultiplier,
  getChinchillaOptimalTokens,
  getComputeOptimalFlops,
} from './training.ts';

// Simulation types
export type {
  SimulationEventBase,
  EventCategory,
  TrainingPhase,
  CollectiveOp,
  SimulationStartEvent,
  SimulationEndEvent,
  PhaseStartEvent,
  PhaseEndEvent,
  LayerStartEvent,
  LayerEndEvent,
  MemoryAllocateEvent,
  MemoryFreeEvent,
  MemorySnapshotEvent,
  ComputeStartEvent,
  ComputeEndEvent,
  CollectiveStartEvent,
  CollectiveProgressEvent,
  CollectiveEndEvent,
  MetricsUpdateEvent,
  SimulationEvent,
  EventStream,
  GPUTimeline,
  SimulationStatus,
  SimulationState,
  SimulationResult,
  EventFilter,
} from './simulation.ts';

// Inference types
export type {
  InferencePrecision,
  PrecisionSpec,
  InferencePhase,
  InferenceMemoryBreakdown,
  KVCacheConfig,
  KVCacheState,
  LatencyMetrics,
  ThroughputMetrics,
  UtilizationMetrics,
  SpeculativeDecodingConfig,
  SpeculativeMetrics,
  SpeculativeTokenState,
  InferenceConfig,
  InferenceSimulationResult,
  InferenceEventType,
  InferenceEventBase,
  TokenGeneratedEvent,
  KVCacheUpdateEvent,
  InferenceMemoryEvent,
  SpeculativeDraftEvent,
  SpeculativeVerifyEvent,
  SpeculativeResultEvent,
  PhaseEvent as InferencePhaseEvent,
  InferenceEvent,
} from './inference.ts';

export {
  PRECISION_SPECS,
  DEFAULT_INFERENCE_CONFIG,
} from './inference.ts';
