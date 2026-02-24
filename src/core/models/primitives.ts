/**
 * Model primitives - tensor math and layer calculations
 */

import {
  type DType,
  type ModelConfig,
  type ModelSpec,
  type AttentionType,
  type EmbeddingLayerSpec,
  type AttentionLayerSpec,
  type MLPLayerSpec,
  type NormLayerSpec,
  type MoELayerSpec,
  type OutputLayerSpec,
  type TransformerBlockSpec,
  DTYPE_BYTES,
} from '../../types/index.ts';

/**
 * Calculate memory for storing parameters
 * @param params Number of parameters
 * @param dtype Data type
 * @returns Size in bytes
 */
export function paramMemory(params: number, dtype: DType): number {
  return params * DTYPE_BYTES[dtype];
}

/**
 * Calculate FLOPs for matrix multiplication: (M, K) x (K, N) -> (M, N)
 * FLOPs = 2 * M * K * N (multiply-add)
 */
export function matmulFlops(m: number, k: number, n: number): number {
  return 2 * m * k * n;
}

/**
 * Calculate FLOPs for batched matrix multiplication
 * Shape: (B, M, K) x (B, K, N) -> (B, M, N)
 */
export function batchedMatmulFlops(b: number, m: number, k: number, n: number): number {
  return b * matmulFlops(m, k, n);
}

/**
 * Calculate embedding layer parameters and FLOPs
 */
export function createEmbeddingLayer(
  vocabSize: number,
  embeddingDim: number,
  name: string = 'embedding'
): EmbeddingLayerSpec {
  const params = vocabSize * embeddingDim;
  // Embedding lookup is essentially a gather, minimal FLOPs
  // But for throughput calculations, count as 0 compute FLOPs
  const flops = 0;
  // Activation size per token = embeddingDim * dtype_bytes (computed later)
  const activationSize = embeddingDim;

  return {
    type: 'embedding',
    name,
    params,
    flops,
    matmulFlops: 0,
    activationSize,
    inputShape: [1], // token id
    outputShape: [embeddingDim],
    vocabSize,
    embeddingDim,
  };
}

/**
 * Calculate attention layer parameters and FLOPs
 * For a single attention layer with QKV projection, attention, and output projection
 */
export function createAttentionLayer(
  hiddenSize: number,
  numHeads: number,
  numKvHeads: number,
  headDim: number,
  attentionType: AttentionType,
  seqLength: number,
  useBias: boolean = false,
  layerIndex: number = 0,
  mlaConfig?: {
    kvLoraRank: number;
    qLoraRank: number;
    qkNopeHeadDim: number;
    qkRopeHeadDim: number;
    vHeadDim: number;
  }
): AttentionLayerSpec {
  if (attentionType === 'mla' && mlaConfig) {
    const { kvLoraRank, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, vHeadDim } = mlaConfig;

    // MLA param counts per layer:
    // q_a_proj: h → q_lora_rank
    const qAParams = hiddenSize * qLoraRank;
    // q_b_proj: q_lora_rank → nH × (qk_nope + qk_rope)
    const qBParams = qLoraRank * numHeads * (qkNopeHeadDim + qkRopeHeadDim);
    // kv_a_proj: h → (kv_lora_rank + qk_rope)
    const kvAParams = hiddenSize * (kvLoraRank + qkRopeHeadDim);
    // kv_b_proj: kv_lora_rank → nH × (qk_nope + v_head)
    const kvBParams = kvLoraRank * numHeads * (qkNopeHeadDim + vHeadDim);
    // o_proj: nH × v_head → h
    const outParams = numHeads * vHeadDim * hiddenSize;

    const params = qAParams + qBParams + kvAParams + kvBParams + outParams;

    // FLOPs: each weight contributes 2×params FLOPs per token (matmul)
    // Q: compress + decompress
    const qFlops = matmulFlops(seqLength, hiddenSize, qLoraRank)
                 + matmulFlops(seqLength, qLoraRank, numHeads * (qkNopeHeadDim + qkRopeHeadDim));
    // KV: compress + decompress
    const kvFlops = matmulFlops(seqLength, hiddenSize, kvLoraRank + qkRopeHeadDim)
                  + matmulFlops(seqLength, kvLoraRank, numHeads * (qkNopeHeadDim + vHeadDim));
    // Attention scores: Q@K^T with full qk dim
    const qkDim = qkNopeHeadDim + qkRopeHeadDim;
    const attnScoreFlops = batchedMatmulFlops(numHeads, seqLength, qkDim, seqLength);
    const softmaxFlops = 5 * numHeads * seqLength * seqLength;
    const attnOutFlops = batchedMatmulFlops(numHeads, seqLength, seqLength, vHeadDim);
    // Output projection
    const outProjFlops = matmulFlops(seqLength, numHeads * vHeadDim, hiddenSize);

    // FA2 fused kernel (Q@K^T, scores@V) gets ~1.5× from FP8, not 2×.
    // Tested: splitting the Amdahl model worsens FP8 calibration by ~5pp
    // (DeepSeek V3 H800: 43.1% → 41.0%). The uniform MATMUL_TIME_FRACTION=0.80
    // in gpu.ts already absorbs this empirically — leave as-is.
    const layerMatmulFlops = qFlops + kvFlops + attnScoreFlops + attnOutFlops + outProjFlops;
    const flops = layerMatmulFlops + softmaxFlops;

    // Activation memory: compressed latents + decompressed Q/KV + output
    const activationSize = (
      numHeads * (qkNopeHeadDim + qkRopeHeadDim) + // Q
      numHeads * (qkNopeHeadDim + vHeadDim) + // decompressed K,V
      hiddenSize // output
    );

    return {
      type: 'attention',
      name: `layer_${layerIndex}_attention`,
      params,
      flops,
      matmulFlops: layerMatmulFlops,
      activationSize,
      inputShape: [seqLength, hiddenSize],
      outputShape: [seqLength, hiddenSize],
      attentionType,
      numHeads,
      numKvHeads,
      headDim,
      hiddenSize,
      useFlashAttention: true,
      useRotaryEmbed: true,
    };
  }

  // Standard MHA/GQA/MQA path
  // Query projection: hiddenSize -> numHeads * headDim
  const qParams = hiddenSize * numHeads * headDim + (useBias ? numHeads * headDim : 0);

  // Key/Value projections: hiddenSize -> numKvHeads * headDim
  const kvParams = 2 * (hiddenSize * numKvHeads * headDim + (useBias ? numKvHeads * headDim : 0));

  // Output projection: numHeads * headDim -> hiddenSize
  const outParams = numHeads * headDim * hiddenSize + (useBias ? hiddenSize : 0);

  const params = qParams + kvParams + outParams;

  // FLOPs calculation per token (for a sequence of seqLength)
  // QKV projections: 3 matmuls of (1, hiddenSize) x (hiddenSize, proj_dim)
  const qkvFlops = matmulFlops(seqLength, hiddenSize, numHeads * headDim) +
    2 * matmulFlops(seqLength, hiddenSize, numKvHeads * headDim);

  // Attention scores: Q @ K^T for each head
  // (numHeads, seqLength, headDim) x (numHeads, headDim, seqLength) -> (numHeads, seqLength, seqLength)
  const attnScoreFlops = batchedMatmulFlops(numHeads, seqLength, headDim, seqLength);

  // Softmax FLOPs: ~5 ops per element (exp, sum, div, etc.)
  const softmaxFlops = 5 * numHeads * seqLength * seqLength;

  // Attention output: scores @ V
  // (numHeads, seqLength, seqLength) x (numHeads, seqLength, headDim) -> (numHeads, seqLength, headDim)
  const attnOutFlops = batchedMatmulFlops(numHeads, seqLength, seqLength, headDim);

  // Output projection
  const outProjFlops = matmulFlops(seqLength, numHeads * headDim, hiddenSize);

  // FA2 fused kernel (Q@K^T, scores@V) gets ~1.5× from FP8, not 2×.
  // Tested: splitting the Amdahl model worsens FP8 calibration by ~5pp
  // (DeepSeek V3 H800: 43.1% → 41.0%). The uniform MATMUL_TIME_FRACTION=0.80
  // in gpu.ts already absorbs this empirically — leave as-is.
  const layerMatmulFlops = qkvFlops + attnScoreFlops + attnOutFlops + outProjFlops;
  const flops = layerMatmulFlops + softmaxFlops;

  // Activation memory per token (for forward pass)
  // QKV activations + attention scores + attention output
  const activationSize = (
    numHeads * headDim + // Q
    numKvHeads * headDim * 2 + // K, V
    hiddenSize // output
  );

  return {
    type: 'attention',
    name: `layer_${layerIndex}_attention`,
    params,
    flops,
    matmulFlops: layerMatmulFlops,
    activationSize,
    inputShape: [seqLength, hiddenSize],
    outputShape: [seqLength, hiddenSize],
    attentionType,
    numHeads,
    numKvHeads,
    headDim,
    hiddenSize,
    useFlashAttention: true,
    useRotaryEmbed: true,
  };
}

/**
 * Calculate MLP/FFN layer parameters and FLOPs
 * Standard FFN: Linear(hidden -> intermediate) -> Activation -> Linear(intermediate -> hidden)
 * Gated FFN: Linear(hidden -> intermediate) * SiLU(Linear(hidden -> intermediate)) -> Linear(intermediate -> hidden)
 */
export function createMLPLayer(
  hiddenSize: number,
  intermediateSize: number,
  gatedMLP: boolean,
  useBias: boolean = false,
  activation: 'gelu' | 'silu' | 'relu' = 'silu',
  seqLength: number = 1,
  layerIndex: number = 0
): MLPLayerSpec {
  let params: number;
  let flops: number;

  let layerMatmulFlops: number;

  if (gatedMLP) {
    // Gate projection + up projection + down projection
    // W_gate: hiddenSize -> intermediateSize
    // W_up: hiddenSize -> intermediateSize
    // W_down: intermediateSize -> hiddenSize
    params = 3 * hiddenSize * intermediateSize;
    if (useBias) {
      params += 2 * intermediateSize + hiddenSize;
    }

    layerMatmulFlops = (
      matmulFlops(seqLength, hiddenSize, intermediateSize) + // gate
      matmulFlops(seqLength, hiddenSize, intermediateSize) + // up
      matmulFlops(seqLength, intermediateSize, hiddenSize)   // down
    );
    // FLOPs: projections + SiLU activation + element-wise multiply
    flops = layerMatmulFlops +
      seqLength * intermediateSize * 5 + // SiLU activation (~5 ops)
      seqLength * intermediateSize;      // element-wise multiply
  } else {
    // Standard FFN: up + down
    params = 2 * hiddenSize * intermediateSize;
    if (useBias) {
      params += intermediateSize + hiddenSize;
    }

    layerMatmulFlops = (
      matmulFlops(seqLength, hiddenSize, intermediateSize) +
      matmulFlops(seqLength, intermediateSize, hiddenSize)
    );
    flops = layerMatmulFlops +
      seqLength * intermediateSize * 5; // activation
  }

  // Activation memory: intermediate activations
  const activationSize = intermediateSize + hiddenSize;

  return {
    type: 'mlp',
    name: `layer_${layerIndex}_mlp`,
    params,
    flops,
    matmulFlops: layerMatmulFlops,
    activationSize,
    inputShape: [seqLength, hiddenSize],
    outputShape: [seqLength, hiddenSize],
    hiddenSize,
    intermediateSize,
    activation,
    useBias,
    gatedMLP,
  };
}

/**
 * Create normalization layer (LayerNorm or RMSNorm)
 */
export function createNormLayer(
  hiddenSize: number,
  normType: 'layernorm' | 'rmsnorm',
  seqLength: number = 1,
  layerIndex: number = 0,
  position: 'pre_attn' | 'post_attn' | 'pre_mlp' | 'post_mlp' | 'final' = 'pre_attn'
): NormLayerSpec {
  // RMSNorm: only gamma (scale)
  // LayerNorm: gamma + beta
  const params = normType === 'rmsnorm' ? hiddenSize : 2 * hiddenSize;

  // FLOPs: mean, variance, normalize, scale (and shift for LayerNorm)
  // RMSNorm: ~4 ops per element
  // LayerNorm: ~6 ops per element
  const flops = seqLength * hiddenSize * (normType === 'rmsnorm' ? 4 : 6);

  return {
    type: normType,
    name: `layer_${layerIndex}_${position}_norm`,
    params,
    flops,
    matmulFlops: 0, // All element-wise ops (mean, variance, normalize, scale)
    activationSize: hiddenSize,
    inputShape: [seqLength, hiddenSize],
    outputShape: [seqLength, hiddenSize],
    hiddenSize,
    eps: 1e-5,
  };
}

/**
 * Create MoE (Mixture of Experts) layer
 */
interface MoELayerOptions {
  capacityFactor?: number;
  seqLength?: number;
  layerIndex?: number;
  numSharedExperts?: number;
  sharedExpertIntermediateSize?: number; // default = intermediateSize (same as routed experts)
  gatedMLP?: boolean;
}

export function createMoELayer(
  hiddenSize: number,
  intermediateSize: number,
  numExperts: number,
  numActiveExperts: number,
  options: MoELayerOptions = {}
): MoELayerSpec {
  const {
    capacityFactor = 1.25,
    seqLength = 1,
    layerIndex = 0,
    numSharedExperts = 0,
    gatedMLP = true,
  } = options;
  const sharedIntermediate = options.sharedExpertIntermediateSize ?? intermediateSize;

  // gated: gate + up + down projections (3 matrices); non-gated: up + down (2 matrices)
  const mlpMatrices = gatedMLP ? 3 : 2;
  const paramsPerExpert = mlpMatrices * hiddenSize * intermediateSize;
  const paramsPerSharedExpert = mlpMatrices * hiddenSize * sharedIntermediate;
  const totalExpertParams = numExperts * paramsPerExpert + numSharedExperts * paramsPerSharedExpert;

  // Router: hiddenSize -> numExperts (routes to routed experts only)
  const routerParams = hiddenSize * numExperts;

  const params = totalExpertParams + routerParams;

  // FLOPs: router + (active routed + shared) experts' MLPs
  const routerMatmulFlops = matmulFlops(seqLength, hiddenSize, numExperts);
  const routerSoftmaxFlops = seqLength * numExperts * 5; // softmax
  const routerFlops = routerMatmulFlops + routerSoftmaxFlops;

  // Each token goes to numActiveExperts routed + numSharedExperts shared
  const upProjections = gatedMLP ? 2 : 1; // gated: gate + up matmuls; non-gated: up only
  const routedProjectionFlops = numActiveExperts * (
    matmulFlops(seqLength, hiddenSize, intermediateSize) * upProjections +
    matmulFlops(seqLength, intermediateSize, hiddenSize) // down
  );
  const sharedProjectionFlops = numSharedExperts * (
    matmulFlops(seqLength, hiddenSize, sharedIntermediate) * upProjections +
    matmulFlops(seqLength, sharedIntermediate, hiddenSize) // down
  );
  const expertProjectionFlops = routedProjectionFlops + sharedProjectionFlops;
  const routedActivationOps = gatedMLP
    ? seqLength * intermediateSize * 6   // SiLU + element-wise multiply
    : seqLength * intermediateSize * 5;  // activation only
  const sharedActivationOps = gatedMLP
    ? seqLength * sharedIntermediate * 6
    : seqLength * sharedIntermediate * 5;
  const expertActivationFlops = numActiveExperts * routedActivationOps
    + numSharedExperts * sharedActivationOps;

  const layerMatmulFlops = routerMatmulFlops + expertProjectionFlops;
  const flops = routerFlops + expertProjectionFlops + expertActivationFlops;

  // Activation memory: router logits + active expert activations
  const activationSize = numExperts
    + numActiveExperts * (intermediateSize + hiddenSize)
    + numSharedExperts * (sharedIntermediate + hiddenSize);

  const sharedExpertSize = numSharedExperts * paramsPerSharedExpert;

  return {
    type: 'moe',
    name: `layer_${layerIndex}_moe`,
    params,
    flops,
    matmulFlops: layerMatmulFlops,
    activationSize,
    inputShape: [seqLength, hiddenSize],
    outputShape: [seqLength, hiddenSize],
    numExperts,
    numActiveExperts,
    numSharedExperts,
    expertSize: paramsPerExpert,
    sharedExpertSize: sharedExpertSize > 0 ? sharedExpertSize : undefined,
    hiddenSize,
    intermediateSize,
    capacityFactor,
  };
}

/**
 * Create output/LM head layer
 */
export function createOutputLayer(
  hiddenSize: number,
  vocabSize: number,
  tiedEmbeddings: boolean,
  seqLength: number = 1
): OutputLayerSpec {
  // If tied embeddings, params are shared with embedding layer
  const params = tiedEmbeddings ? 0 : hiddenSize * vocabSize;

  // FLOPs: final linear projection
  const flops = matmulFlops(seqLength, hiddenSize, vocabSize);

  return {
    type: 'output',
    name: 'output',
    params,
    flops,
    matmulFlops: flops, // All FLOPs are matmul (final linear projection)
    activationSize: vocabSize,
    inputShape: [seqLength, hiddenSize],
    outputShape: [seqLength, vocabSize],
    hiddenSize,
    vocabSize,
    tiedEmbeddings,
  };
}

/**
 * Create a complete transformer block (pre-norm architecture)
 */
export function createTransformerBlock(
  config: ModelConfig,
  seqLength: number,
  layerIndex: number
): TransformerBlockSpec {
  const headDim = config.headDim ?? config.hiddenSize / config.numAttentionHeads;
  const numKvHeads = config.numKvHeads ?? config.numAttentionHeads;

  const preNorm = createNormLayer(
    config.hiddenSize,
    config.normType ?? 'rmsnorm',
    seqLength,
    layerIndex,
    'pre_attn'
  );

  const mlaConfig = config.attentionType === 'mla' && config.kvLoraRank
    ? {
        kvLoraRank: config.kvLoraRank,
        qLoraRank: config.qLoraRank ?? 1536,
        qkNopeHeadDim: config.qkNopeHeadDim ?? 128,
        qkRopeHeadDim: config.qkRopeHeadDim ?? 64,
        vHeadDim: config.vHeadDim ?? 128,
      }
    : undefined;

  const attention = createAttentionLayer(
    config.hiddenSize,
    config.numAttentionHeads,
    numKvHeads,
    headDim,
    config.attentionType ?? 'mha',
    seqLength,
    config.useBias ?? false,
    layerIndex,
    mlaConfig
  );

  const postAttentionNorm = createNormLayer(
    config.hiddenSize,
    config.normType ?? 'rmsnorm',
    seqLength,
    layerIndex,
    'pre_mlp'
  );

  // Determine if this layer is MoE or dense
  const hasMoE = config.numExperts && config.numExperts > 1;
  const freq = config.moeLayerFrequency ?? 1;
  const firstDense = config.firstKDenseLayers ?? 0;
  const lastDense = config.lastKDenseLayers ?? 0;
  const isMoELayer = hasMoE
    && layerIndex >= firstDense
    && layerIndex < config.numLayers - lastDense
    && (freq === 1 || (layerIndex + 1) % freq === 0);

  let mlp: MLPLayerSpec | MoELayerSpec;
  if (isMoELayer) {
    mlp = createMoELayer(
      config.hiddenSize,
      config.expertIntermediateSize ?? config.intermediateSize,
      config.numExperts!,
      config.numActiveExperts ?? 2,
      {
        seqLength,
        layerIndex,
        numSharedExperts: config.numSharedExperts ?? 0,
        sharedExpertIntermediateSize: config.sharedExpertIntermediateSize,
        gatedMLP: config.gatedMLP ?? true,
      }
    );
  } else {
    mlp = createMLPLayer(
      config.hiddenSize,
      config.intermediateSize,
      config.gatedMLP ?? true,
      config.useBias ?? false,
      config.activation ?? 'silu',
      seqLength,
      layerIndex
    );
  }

  const totalParams = preNorm.params + attention.params + postAttentionNorm.params + mlp.params;
  const totalFlops = preNorm.flops + attention.flops + postAttentionNorm.flops + mlp.flops;
  const totalMatmulFlops = preNorm.matmulFlops + attention.matmulFlops + postAttentionNorm.matmulFlops + mlp.matmulFlops;

  return {
    index: layerIndex,
    preNorm,
    attention,
    postAttentionNorm,
    mlp,
    totalParams,
    totalFlops,
    totalMatmulFlops,
  };
}

/**
 * Build complete model specification from config
 */
export function buildModelSpec(config: ModelConfig, seqLength: number = 2048): ModelSpec {
  // Validate core model config
  if (config.numLayers <= 0) throw new Error('numLayers must be positive');
  if (config.hiddenSize <= 0) throw new Error('hiddenSize must be positive');
  if (config.numAttentionHeads <= 0) throw new Error('numAttentionHeads must be positive');
  if (config.vocabSize <= 0) throw new Error('vocabSize must be positive');
  if (config.intermediateSize <= 0) throw new Error('intermediateSize must be positive');
  if (config.maxSeqLength <= 0) throw new Error('maxSeqLength must be positive');
  if (seqLength <= 0) throw new Error('seqLength must be positive');

  const headDim = config.headDim ?? config.hiddenSize / config.numAttentionHeads;
  const numKvHeads = config.numKvHeads ?? config.numAttentionHeads;

  // Create embedding layer
  const embedding = createEmbeddingLayer(config.vocabSize, config.hiddenSize);

  // Create transformer blocks
  const blocks: TransformerBlockSpec[] = [];
  for (let i = 0; i < config.numLayers; i++) {
    blocks.push(createTransformerBlock(config, seqLength, i));
  }

  // Final norm
  const finalNorm = createNormLayer(
    config.hiddenSize,
    config.normType ?? 'rmsnorm',
    seqLength,
    config.numLayers,
    'final'
  );

  // Output layer
  const output = createOutputLayer(
    config.hiddenSize,
    config.vocabSize,
    config.tiedEmbeddings ?? false,
    seqLength
  );

  // Calculate totals
  const embeddingParams = embedding.params;
  const attentionParams = blocks.reduce((sum, b) => sum + b.attention.params, 0);
  const mlpParams = blocks.reduce((sum, b) => sum + b.mlp.params, 0);
  const normParams = blocks.reduce((sum, b) =>
    sum + b.preNorm.params + (b.postAttentionNorm?.params ?? 0), 0
  ) + finalNorm.params;

  const totalParams = embeddingParams + attentionParams + mlpParams + normParams + output.params;

  // For MoE, active params = embedding + attention + active_experts_per_layer + dense_mlps + norms + output
  const isMoE = !!(config.numExperts && config.numExperts > 1);
  if (isMoE) {
    if (!config.numActiveExperts || config.numActiveExperts <= 0) {
      throw new Error('MoE model must have positive numActiveExperts');
    }
    if (config.numActiveExperts > config.numExperts!) {
      throw new Error(`numActiveExperts (${config.numActiveExperts}) cannot exceed numExperts (${config.numExperts})`);
    }
  }
  const numMoELayers = blocks.filter(b => b.mlp.type === 'moe').length;
  let activeParams = totalParams;
  if (isMoE) {
    let activeMlpParams = 0;
    for (const block of blocks) {
      if (block.mlp.type === 'moe') {
        const moe = block.mlp as MoELayerSpec;
        const routerP = moe.hiddenSize * moe.numExperts;
        activeMlpParams += moe.numActiveExperts * moe.expertSize + (moe.sharedExpertSize ?? 0) + routerP;
      } else {
        activeMlpParams += block.mlp.params; // Dense layers: 100% active
      }
    }
    activeParams = embeddingParams + attentionParams + activeMlpParams + normParams + output.params;
  }

  // FLOPs per token (forward pass)
  const flopsPerToken = (
    blocks.reduce((sum, b) => sum + b.totalFlops, 0) +
    finalNorm.flops +
    output.flops
  ) / seqLength;

  // Matmul FLOPs per token — subset of flopsPerToken from GEMM/batched GEMM operations.
  // Non-matmul ops (norms, activations, residual adds, softmax) have arithmetic intensity
  // < 10 ops/byte, well below any GPU's ops:byte ratio. They are unconditionally
  // memory-bandwidth-bound, so memBWBoundFraction = 1 - matmulFraction is a valid
  // direct substitution for the hardcoded MEMBW_BOUND_FRACTION.
  const matmulFlopsPerToken = (
    blocks.reduce((sum, b) => sum + b.totalMatmulFlops, 0) +
    finalNorm.matmulFlops +
    output.matmulFlops
  ) / seqLength;
  const matmulFraction = flopsPerToken > 0 ? matmulFlopsPerToken / flopsPerToken : 0;

  // Collect all layers
  const layers = [
    embedding,
    ...blocks.flatMap(b => [b.preNorm, b.attention, b.postAttentionNorm!, b.mlp]),
    finalNorm,
    output,
  ].filter(Boolean);

  return {
    name: config.name,
    family: config.family ?? 'custom',

    numLayers: config.numLayers,
    hiddenSize: config.hiddenSize,
    intermediateSize: config.intermediateSize,
    numAttentionHeads: config.numAttentionHeads,
    numKvHeads,
    headDim,
    vocabSize: config.vocabSize,
    maxSeqLength: config.maxSeqLength,

    attentionType: config.attentionType ?? 'mha',
    useRotaryEmbed: config.useRotaryEmbed ?? true,

    kvLoraRank: config.kvLoraRank,
    qLoraRank: config.qLoraRank,
    qkNopeHeadDim: config.qkNopeHeadDim,
    qkRopeHeadDim: config.qkRopeHeadDim,
    vHeadDim: config.vHeadDim,

    normType: config.normType ?? 'rmsnorm',
    normEps: 1e-5,

    activation: config.activation ?? 'silu',
    gatedMLP: config.gatedMLP ?? true,
    useBias: config.useBias ?? false,

    isMoE,
    numExperts: config.numExperts,
    numActiveExperts: config.numActiveExperts,
    numSharedExperts: config.numSharedExperts,
    sharedExpertIntermediateSize: config.sharedExpertIntermediateSize,
    numMoELayers: isMoE ? numMoELayers : undefined,
    expertIntermediateSize: config.expertIntermediateSize,
    routingDeviceLimit: config.routingDeviceLimit,

    tiedEmbeddings: config.tiedEmbeddings ?? false,

    totalParams,
    activeParams,
    embeddingParams,
    attentionParams,
    mlpParams,

    flopsPerToken,
    matmulFlopsPerToken,
    matmulFraction,

    layers,
    blocks,
  };
}

/**
 * Calculate training FLOPs for one forward-backward pass
 * Forward FLOPs ≈ 2 * params * tokens (for matmul-heavy models)
 * Backward FLOPs ≈ 4 * params * tokens (2x forward due to gradient computation)
 * Total ≈ 6 * params * tokens
 */
export function trainingFlops(params: number, tokens: number): number {
  return 6 * params * tokens;
}

