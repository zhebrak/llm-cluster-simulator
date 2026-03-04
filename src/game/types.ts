/**
 * Game mode types for the Learn Distributed Training & Inference feature
 */

export type GameMode = 'training' | 'inference';
export type GameDifficulty = 'beginner' | 'intermediate' | 'advanced';

export interface TaskSetup {
  // Hardware
  modelId: string;
  gpuId: string;
  numGPUs?: number;       // default: 1
  gpusPerNode?: number;   // default: derived from cluster
  strategyType?: string;  // training only, default: 'ddp'

  // Training config overrides (applied on top of base defaults)
  mixedPrecision?: 'fp32' | 'tf32' | 'fp16' | 'bf16' | 'fp8' | 'fp4';
  activationCheckpointing?: boolean;
  checkpointingGranularity?: 'full' | 'selective';
  flashAttention?: boolean;
  globalBatchSize?: number;
  microBatchSize?: number;
  sequenceLength?: number;
  sequenceParallel?: boolean;
  tpDegree?: number;
  ppDegree?: number;
  epDegree?: number;
  cpDegree?: number;
  pipelineSchedule?: '1f1b' | 'interleaved-1f1b' | 'dualpipe-v';
  interleavedStages?: number;
  finetuningMethod?: 'full' | 'lora' | 'qlora';
  loraRank?: number;
  loraTargetModules?: 'q_v' | 'q_k_v_o' | 'all_linear';

  // Inference config overrides (applied on top of base defaults)
  weightPrecision?: string;
  kvCachePrecision?: string;
  batchSize?: number;
  inputSeqLen?: number;
  outputSeqLen?: number;
  tensorParallel?: number;
  expertParallel?: number;
  pagedAttention?: boolean;
  continuousBatching?: boolean;
  speculativeDecoding?: boolean;
  draftModelId?: string;
  numSpeculativeTokens?: number;
  acceptanceRate?: number;
}

export interface WinningCriterion {
  field: string;          // Dot-path: 'mfu', 'memoryUtilization', 'latency.ttft'
  operator: '>' | '>=' | '<' | '<=' | '==' | '!=';
  value: number | boolean | string;
  label: string;          // Human-readable: "MFU > 30%"
}

export interface ValidationResult {
  passed: boolean;
  results: Array<{ criterion: WinningCriterion; met: boolean }>;
}

export interface ExpectedChange {
  field: string;    // Key in TaskConfigSnapshot (flat namespace)
  check: 'changed' | 'unchanged' | 'increased' | 'decreased' | 'enabled' | 'disabled';
  label: string;    // For debugging/testing (not shown in UI)
}

export interface GameTask {
  id: string;             // 'training-beginner-01'
  mode: GameMode;
  difficulty: GameDifficulty;
  order: number;          // 0-9
  title: string;
  briefing: string;       // Scenario + what to explore
  concept: string;        // One-liner: "Mixed Precision Training"
  learningObjectives: string[];  // 2-4 concise, action-oriented objectives per task
  setup: TaskSetup;       // Minimal scene-setting
  winningCriteria: WinningCriterion[];
  expectedChanges?: ExpectedChange[];  // Config validation: required/protected parameter changes
  hints: string[];        // 2-3 progressive educational hints
  successExplanation: string; // Post-success lesson
}
