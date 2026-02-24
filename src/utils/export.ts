/**
 * Export utilities
 */

/**
 * Exportable configuration snapshot
 */
export interface ExportableConfig {
  version: string;
  timestamp: string;
  config: {
    // Model
    modelId: string;
    // Cluster
    clusterId: string;
    gpuId: string;
    numGPUs: number;
    gpusPerNode: number;
    // Batch
    globalBatchSize: number;
    microBatchSize: number;
    numMicroBatches: number;
    sequenceLength: number;
    // Parallelism — only active dimensions are included
    strategyType: string;
    dpDegree: number;
    tpDegree?: number;  // Omitted for 1D strategies (ddp, zero-1, fsdp)
    ppDegree?: number;  // Omitted when no pipeline parallelism
    epDegree?: number;  // Omitted for dense models or EP=1
    cpDegree?: number;  // Omitted when CP=1
    pipelineSchedule?: string;   // Omitted when no PP or default 1F1B
    interleavedStages?: number;  // Omitted when no PP or default 1F1B
    // Optimizations
    precision: string;
    activationCheckpointing: boolean;
    checkpointingGranularity?: 'full' | 'selective'; // Omitted when full (default) or ckpt=false
    selectiveStoredLayers?: number;  // Omitted when auto or not selective AC
    sequenceParallel?: boolean;  // Omitted when no TP
    flashAttention?: boolean;
    // Fine-tuning (omitted when full fine-tuning)
    finetuningMethod?: string;
    loraRank?: number;
    loraTargetModules?: string;
    // Training scale
    trainingGoal: string;
    targetTokens: string; // Exponent notation, e.g. "15e12"
    // Pricing (omitted when using default rate)
    pricePerGPUHour?: number;
  };
  metrics?: {
    // Efficiency (percentages, e.g. 44.4 = 44.4%)
    mfuPct: number;
    hfuPct: number;
    modelFlopsMfuPct?: number;  // Only when > 10% above PaLM MFU
    tflopsPerGPU: number;
    // Throughput
    tokensPerSecond: number;
    // Timing
    stepTimeMs: number;
    // Memory
    memoryPerGPU: number;
    memoryUtilizationPct: number;
    // Overheads (percentages, e.g. 12.3 = 12.3%)
    communicationOverheadPct: number;
    pipelineBubblePct?: number;  // Omitted when no pipeline parallelism
    // Training projection
    timeToTrainHours?: number;
    gpuHours?: number;
    estimatedCost?: number;
  };
}

/**
 * Exportable inference configuration snapshot
 */
export interface ExportableInferenceConfig {
  version: string;
  timestamp: string;
  mode: 'inference';
  config: {
    modelId: string;
    clusterId: string;
    gpuId: string;
    numGPUs: number;
    gpusPerNode: number;
    batchSize: number;
    inputSeqLen: number;
    outputSeqLen: number;
    weightPrecision: string;
    kvCachePrecision: string;
    flashAttention: boolean;
    pagedAttention: boolean;
    continuousBatching: boolean;
    tensorParallel?: number;   // Omitted when TP=1
    expertParallel?: number;   // Omitted for dense models or EP=1
    speculativeDecoding?: boolean;
    draftModelId?: string;
    numSpeculativeTokens?: number;
    acceptanceRate?: number;
    // Pricing (omitted when using default rate)
    pricePerGPUHour?: number;
  };
  metrics?: {
    ttftMs: number;
    tpotMs: number;
    totalLatencyMs: number;
    tokensPerSecond: number;
    prefillTokensPerSecond: number;
    decodeTokensPerSecond: number;
    numReplicas: number;
    memory: {
      weightsGB: number;
      kvCacheGB: number;
      activationsGB: number;
      overheadGB: number;
      totalPerGPU_GB: number;
    };
    costPerMTokens?: number;
  };
}

/**
 * Export configuration to JSON
 */
export function exportConfigToJSON(config: ExportableConfig | ExportableInferenceConfig): string {
  return JSON.stringify(config, null, 2);
}

/**
 * Download text content as file
 */
export function downloadAsFile(content: string, filename: string, mimeType: string = 'text/plain'): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
