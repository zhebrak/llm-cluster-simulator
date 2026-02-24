/**
 * Simulation event types
 */

import type { GPUState } from './hardware.ts';

// Base event
export interface SimulationEventBase {
  id: string;
  timestamp: number;         // Milliseconds from simulation start
  duration: number;          // Event duration in milliseconds
  gpuId: number;             // Which GPU this event is for (-1 for global)
  rank?: number;             // Global rank if applicable
}

// Event categories
export type EventCategory =
  | 'simulation'
  | 'phase'
  | 'layer'
  | 'memory'
  | 'compute'
  | 'collective'
  | 'metrics';

// Training phase
export type TrainingPhase =
  | 'forward'
  | 'backward'
  | 'optimizer'
  | 'communication';

// Collective operation types
export type CollectiveOp =
  | 'all-reduce'
  | 'all-gather'
  | 'reduce-scatter'
  | 'all-to-all'
  | 'broadcast'
  | 'reduce'
  | 'send'
  | 'recv'
  | 'barrier';

// Simulation start/end events
export interface SimulationStartEvent extends SimulationEventBase {
  type: 'simulation-start';
  category: 'simulation';
  config: {
    totalGPUs: number;
    totalSteps: number;
    modelName: string;
    strategyName: string;
  };
}

export interface SimulationEndEvent extends SimulationEventBase {
  type: 'simulation-end';
  category: 'simulation';
  metrics: {
    totalTimeMs: number;
    avgStepTimeMs: number;
    tokensPerSecond: number;
    mfu: number;
  };
}

// Phase events
export interface PhaseStartEvent extends SimulationEventBase {
  type: 'phase-start';
  category: 'phase';
  phase: TrainingPhase;
  stepNumber: number;
  microBatchId?: number;
}

export interface PhaseEndEvent extends SimulationEventBase {
  type: 'phase-end';
  category: 'phase';
  phase: TrainingPhase;
  stepNumber: number;
  microBatchId?: number;
}

// Layer events
export interface LayerStartEvent extends SimulationEventBase {
  type: 'layer-start';
  category: 'layer';
  layerName: string;
  layerIndex: number;
  phase: TrainingPhase;
  inputShape: number[];
}

export interface LayerEndEvent extends SimulationEventBase {
  type: 'layer-end';
  category: 'layer';
  layerName: string;
  layerIndex: number;
  phase: TrainingPhase;
  flops: number;
}

// Memory events
export interface MemoryAllocateEvent extends SimulationEventBase {
  type: 'memory-allocate';
  category: 'memory';
  tensorName: string;
  sizeBytes: number;
  memoryType: 'parameter' | 'gradient' | 'optimizer' | 'activation' | 'temporary';
}

export interface MemoryFreeEvent extends SimulationEventBase {
  type: 'memory-free';
  category: 'memory';
  tensorName: string;
  sizeBytes: number;
}

export interface MemorySnapshotEvent extends SimulationEventBase {
  type: 'memory-snapshot';
  category: 'memory';
  totalAllocated: number;
  breakdown: {
    parameters: number;
    gradients: number;
    optimizer: number;
    activations: number;
    temporary: number;
  };
  peakMemory: number;
}

// Compute events
export interface ComputeStartEvent extends SimulationEventBase {
  type: 'compute-start';
  category: 'compute';
  operation: string;
  flops: number;
  inputShapes: number[][];
  outputShape: number[];
}

export interface ComputeEndEvent extends SimulationEventBase {
  type: 'compute-end';
  category: 'compute';
  operation: string;
  actualTFLOPS: number;
  efficiency: number;        // Achieved vs peak TFLOPS
}

// Collective communication events
export interface CollectiveStartEvent extends SimulationEventBase {
  type: 'collective-start';
  category: 'collective';
  operation: CollectiveOp;
  sizeBytes: number;
  numRanks: number;
  algorithm: 'ring' | 'tree' | 'recursive-halving' | 'direct';
  isIntraNode: boolean;
}

export interface CollectiveProgressEvent extends SimulationEventBase {
  type: 'collective-progress';
  category: 'collective';
  operation: CollectiveOp;
  progress: number;          // 0-1
  bytesSent: number;
  bytesReceived: number;
}

export interface CollectiveEndEvent extends SimulationEventBase {
  type: 'collective-end';
  category: 'collective';
  operation: CollectiveOp;
  actualBandwidthGBps: number;
  efficiency: number;        // Achieved vs theoretical bandwidth
}

// Metrics update event
export interface MetricsUpdateEvent extends SimulationEventBase {
  type: 'metrics-update';
  category: 'metrics';
  stepNumber: number;
  stepTimeMs: number;
  tokensPerSecond: number;
  samplesPerSecond: number;
  mfu: number;
  hfu: number;
  memoryUsedGB: number;
  memoryPeakGB: number;
}

// Union of all event types
export type SimulationEvent =
  | SimulationStartEvent
  | SimulationEndEvent
  | PhaseStartEvent
  | PhaseEndEvent
  | LayerStartEvent
  | LayerEndEvent
  | MemoryAllocateEvent
  | MemoryFreeEvent
  | MemorySnapshotEvent
  | ComputeStartEvent
  | ComputeEndEvent
  | CollectiveStartEvent
  | CollectiveProgressEvent
  | CollectiveEndEvent
  | MetricsUpdateEvent;

// Event stream for visualization
export interface EventStream {
  events: SimulationEvent[];
  currentIndex: number;
  isComplete: boolean;
  totalDurationMs: number;
}

// GPU timeline for visualization
export interface GPUTimeline {
  gpuId: number;
  rank: number;
  events: SimulationEvent[];
  stateHistory: Array<{
    timestamp: number;
    state: GPUState;
    duration: number;
  }>;
}

// Simulation state
export type SimulationStatus =
  | 'idle'
  | 'configuring'
  | 'validating'
  | 'running'
  | 'paused'
  | 'complete'
  | 'error';

export interface SimulationState {
  status: SimulationStatus;
  progress: number;          // 0-1
  currentStep: number;
  totalSteps: number;
  currentTimeMs: number;
  totalTimeMs: number;
  playbackSpeed: number;     // 1 = realtime, 10 = 10x speed
  events: EventStream;
  timelines: GPUTimeline[];
  error?: string;
}

// Simulation result
export interface SimulationResult {
  success: boolean;
  state: SimulationState;
  events: SimulationEvent[];
  metrics: {
    avgStepTimeMs: number;
    minStepTimeMs: number;
    maxStepTimeMs: number;
    totalTimeMs: number;
    tokensPerSecond: number;
    samplesPerSecond: number;
    mfu: number;
    hfu: number;
    communicationOverhead: number;
    pipelineBubble: number;
    peakMemoryGB: number;
  };
  analysis: {
    bottleneck: 'compute' | 'memory' | 'communication';
    recommendations: string[];
  };
}

// Event filter for visualization
export interface EventFilter {
  categories: EventCategory[];
  phases: TrainingPhase[];
  gpuIds: number[];
  minDurationMs: number;
  showOverlapped: boolean;
}

