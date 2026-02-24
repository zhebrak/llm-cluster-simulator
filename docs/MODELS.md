# Model Registry Reference

> **See also:** [PHYSICS.md](PHYSICS.md) (FLOPs/MFU formulas) | [STRATEGIES.md](STRATEGIES.md) (parallelism)

## Table of Contents
1. [Model Registry Architecture](#1-model-registry-architecture)
2. [Architecture Types](#2-architecture-types)
3. [MLA (Multi-head Latent Attention)](#3-mla-multi-head-latent-attention)
4. [MoE Mixed Architectures](#4-moe-mixed-architectures)
5. [Parameter Counting](#5-parameter-counting)
6. [FLOPs Per Token](#6-flops-per-token)
7. [Chinchilla Scaling](#7-chinchilla-scaling)
8. [Model Catalog](#8-model-catalog)

---

## 1. Model Registry Architecture

The model system follows a two-step pipeline: **ModelConfig** (user-facing specification)
is transformed by `buildModelSpec()` into **ModelSpec** (computed, immutable).

### Pipeline

```
ModelConfig  -->  buildModelSpec(config, seqLength)  -->  ModelSpec
(user input)         (tensor math)                     (computed fields)
```

### Key Entry Points

- `getModel(id, seqLength?)` -- looks up a `ModelConfig` by ID, calls `buildModelSpec(config, seqLength)`,
  returns a `ModelSpec`. When `seqLength !== maxSeqLength`, the model is rebuilt so that attention FLOPs
  scale with actual training sequence length rather than the model's maximum.
- `getModelConfig(id)` -- returns the raw `ModelConfig` without building a spec.
- `listModels()` -- returns `ModelMetadata[]` for all registered models.
- `searchModels(query)` -- fuzzy search by name, family, or ID.
- `registerModel(id, config)` -- registers a custom model at runtime.

### Source Files

| File | Purpose |
|------|---------|
| `src/types/model.ts` | Type definitions for `ModelConfig`, `ModelSpec`, `AttentionType`, layer specs |
| `src/core/models/registry.ts` | `ModelRegistry` singleton, `getModel()`, search/filter API |
| `src/core/models/architectures.ts` | All built-in `ModelConfig` definitions, `MODEL_FAMILIES` |
| `src/core/models/primitives.ts` | `buildModelSpec()`, layer constructors, FLOPs math |
| `src/core/models/moe.ts` | MoE-specific utilities: memory breakdown, communication, expert placement |

### ModelConfig Fields

```typescript
interface ModelConfig {
  name: string;
  family?: string;

  // Core architecture
  numLayers: number;
  hiddenSize: number;
  intermediateSize: number;       // MLP intermediate dimension
  numAttentionHeads: number;
  numKvHeads?: number;            // For GQA/MQA (default: numAttentionHeads)
  headDim?: number;               // Override (default: hiddenSize / numAttentionHeads)
  vocabSize: number;
  maxSeqLength: number;

  // Architecture choices
  attentionType?: AttentionType;  // 'mha' | 'mqa' | 'gqa' | 'mla'
  normType?: 'layernorm' | 'rmsnorm';
  activation?: 'gelu' | 'silu' | 'relu';
  gatedMLP?: boolean;             // SwiGLU / GeGLU (3 weight matrices instead of 2)
  useBias?: boolean;
  tiedEmbeddings?: boolean;
  useRotaryEmbed?: boolean;

  // MLA fields (DeepSeek V2/V3/R1, Mistral Large 3, GLM-4.7 Flash, GLM-5, Kimi K2.5)
  kvLoraRank?: number;
  qLoraRank?: number;
  qkNopeHeadDim?: number;
  qkRopeHeadDim?: number;
  vHeadDim?: number;

  // MoE fields
  numExperts?: number;
  numActiveExperts?: number;
  expertIntermediateSize?: number;
  moeLayerFrequency?: number;     // Every Nth layer is MoE (default 1)
  numSharedExperts?: number;      // Always-active experts per MoE layer (default 0)
  firstKDenseLayers?: number;     // Initial dense layers before MoE starts (default 0)
  lastKDenseLayers?: number;      // Trailing dense layers (default 0, currently unused)
  sharedExpertIntermediateSize?: number; // Shared expert MLP intermediate dim (default = expertIntermediateSize)
  routingDeviceLimit?: number;    // Max EP groups per token (DeepSeek V3: M=4, V2: M=6)
}
```

### ModelSpec Computed Fields

`buildModelSpec()` adds these computed values on top of all `ModelConfig` fields:

| Field | Description |
|-------|-------------|
| `totalParams` | Total parameter count (all experts included) |
| `activeParams` | Parameters active per forward pass (MoE: shared + active experts; dense: same as total) |
| `embeddingParams` | Parameters in embedding layer |
| `attentionParams` | Parameters in all attention layers |
| `mlpParams` | Parameters in all MLP/MoE layers |
| `flopsPerToken` | Forward-pass FLOPs per token at the given sequence length |
| `matmulFlopsPerToken` | Matmul-only FLOPs per token |
| `matmulFraction` | matmulFlopsPerToken / flopsPerToken |
| `isMoE` | Whether the model uses Mixture of Experts |
| `numMoELayers` | Count of MoE layers (for mixed dense/MoE architectures) |
| `layers` | Flat list of all layer specs |
| `blocks` | Array of `TransformerBlockSpec` (one per transformer layer) |

---

## 2. Architecture Types

### Attention Types

| Type | Key | Description | numKvHeads |
|------|-----|-------------|------------|
| Multi-Head Attention | `mha` | Standard transformer attention | = numAttentionHeads |
| Multi-Query Attention | `mqa` | Single KV head shared across all Q heads | = 1 |
| Grouped-Query Attention | `gqa` | KV heads shared within groups | < numAttentionHeads, > 1 |
| Multi-head Latent Attention | `mla` | Low-rank compressed KV cache (DeepSeek) | = numAttentionHeads (decompressed) |

### MLP Types

**Standard MLP** (`gatedMLP: false`): Two weight matrices.
- Up projection: `hiddenSize -> intermediateSize`
- Down projection: `intermediateSize -> hiddenSize`
- Parameters per layer: `2 * hiddenSize * intermediateSize`
- Used by: GPT-3, Nemotron-4

**Gated MLP** (`gatedMLP: true`): Three weight matrices (SwiGLU, GeGLU, or gated ReLU).
- Gate projection: `hiddenSize -> intermediateSize`
- Up projection: `hiddenSize -> intermediateSize`
- Down projection: `intermediateSize -> hiddenSize`
- Parameters per layer: `3 * hiddenSize * intermediateSize`
- Used by: LLaMA, Mistral, DeepSeek, Qwen, Gemma, Phi, OLMo, and most modern models

### Normalization Types

| Type | Key | Parameters | Notes |
|------|-----|-----------|-------|
| RMSNorm | `rmsnorm` | gamma only (hiddenSize) | LLaMA and most modern models |
| LayerNorm | `layernorm` | gamma + beta (2 * hiddenSize) | GPT-2/3, Nemotron-4, DBRX, Command-R |

### Activation Functions

| Function | Key | Models |
|----------|-----|--------|
| SiLU (SwiGLU) | `silu` | LLaMA, Mistral, DeepSeek, Qwen, OLMo, Phi-3 Mini/Medium, Phi-4 |
| GeLU (GeGLU) | `gelu` | GPT-3, Gemma, Phi-3 Small, Grok |
| ReLU | `relu` | Nemotron-4 (ReLU-squared approximated as relu) |

### Head Dimension

Usually computed as `hiddenSize / numAttentionHeads`, but several model families decouple
head dimension from hidden size via an explicit `headDim` override:

| Model | Hidden | Heads | Natural headDim | Actual headDim |
|-------|--------|-------|----------------|----------------|
| Qwen3 0.6B | 1024 | 16 | 64 | 128 |
| Qwen3 4B | 2560 | 32 | 80 | 128 |
| Qwen3 32B | 5120 | 64 | 80 | 128 |
| Gemma 3 1B | 1152 | 4 | 288 | 256 |
| Gemma 3 4B | 2560 | 8 | 320 | 256 |
| Gemma 3 12B | 3840 | 16 | 240 | 256 |
| Gemma 3 27B | 5376 | 32 | 168 | 128 |
| GLM-4.5 Air | 4096 | 96 | ~43 | 128 |
| GLM-4.7 353B | 5120 | 96 | ~53 | 128 |
| MiniMax M2.5 | 3072 | 48 | 64 | 128 |
| GPT-OSS 120B/20B | 2880 | 64 | 45 | 64 |
| Devstral Small 2 | 5120 | 32 | 160 | 128 |

When `headDim` differs from the natural ratio, the Q projection maps from `hiddenSize`
to `numHeads * headDim`, which may be larger or smaller than `hiddenSize`.

---

## 3. MLA (Multi-head Latent Attention)

Multi-head Latent Attention compresses the KV cache into a low-rank latent representation,
dramatically reducing inference memory. Used by DeepSeek V2/V3/R1, Mistral Large 3,
GLM-4.7 Flash, GLM-5, and Kimi K2.5.

### How It Works

Instead of storing full K and V tensors per head, MLA compresses KV into a joint latent:

1. **KV compression**: `h -> (kvLoraRank + qkRopeHeadDim)` via `kv_a_proj`
2. **KV decompression**: `kvLoraRank -> nHeads * (qkNopeHeadDim + vHeadDim)` via `kv_b_proj`
3. **Q compression**: `h -> qLoraRank` via `q_a_proj`
4. **Q decompression**: `qLoraRank -> nHeads * (qkNopeHeadDim + qkRopeHeadDim)` via `q_b_proj`
5. **Output projection**: `nHeads * vHeadDim -> h` via `o_proj`

### KV Cache Size

Per-layer KV cache stores only the compressed latent: `kvLoraRank + qkRopeHeadDim` values.

For DeepSeek V3/R1: `512 + 64 = 576` values per layer (vs 2 * 128 * 128 = 32768 for standard GQA
with 128 KV heads).

The compressed latent is **replicated** across TP ranks (not split), because the low-rank
representation cannot be meaningfully partitioned. This means inference TPOT scales differently:
weight memory scales with `1/TP`, but KV cache stays constant.

### MLA Fields in ModelConfig

| Field | DeepSeek V3/R1 | DeepSeek V2 | Mistral Large 3 | GLM-4.7 Flash | GLM-5 | Kimi K2.5 |
|-------|---------------|-------------|-----------------|---------------|-------|-----------|
| `kvLoraRank` | 512 | 512 | 512 | 512 | 512 | 512 |
| `qLoraRank` | 1536 | 1536 | 1536 | 768 | 2048 | 1536 |
| `qkNopeHeadDim` | 128 | 128 | 128 | 192 | 192 | 128 |
| `qkRopeHeadDim` | 64 | 64 | 64 | 64 | 64 | 64 |
| `vHeadDim` | 128 | 128 | 128 | 256 | 256 | 128 |

Note: `deepseek-moe-16b` uses standard MHA attention (not MLA).

### Parameter Counting for MLA

Per-layer attention parameters:

```
q_a_proj:  h * qLoraRank
q_b_proj:  qLoraRank * nHeads * (qkNopeHeadDim + qkRopeHeadDim)
kv_a_proj: h * (kvLoraRank + qkRopeHeadDim)
kv_b_proj: kvLoraRank * nHeads * (qkNopeHeadDim + vHeadDim)
o_proj:    nHeads * vHeadDim * h
```

Source: `src/core/models/primitives.ts:createAttentionLayer()`, `src/core/inference/kv-cache.ts`

---

## 4. MoE Mixed Architectures

Modern MoE models use mixed dense/MoE architectures with several configurable dimensions.

### MoE Fields

| Field | Description | Default |
|-------|-------------|---------|
| `numExperts` | Total number of routed experts | -- |
| `numActiveExperts` | Experts activated per token (top-k) | -- |
| `expertIntermediateSize` | Per-expert MLP width | = `intermediateSize` |
| `moeLayerFrequency` | Every Nth layer is MoE (1 = all) | 1 |
| `numSharedExperts` | Always-active experts alongside routed ones | 0 |
| `firstKDenseLayers` | Initial layers that are always dense | 0 |
| `routingDeviceLimit` | Max EP groups each token contacts (M) | -- |

### numMoELayers Computation

The number of MoE layers is computed dynamically based on the configuration:

```
For each layer index i in [0, numLayers):
  isMoE = (numExperts > 1)
        AND (i >= firstKDenseLayers)
        AND (i < numLayers - lastKDenseLayers)
        AND (freq == 1 OR (i + 1) % freq == 0)
```

### Routing Device Limit

The optional `routingDeviceLimit` (M) caps how many EP groups each token can contact during
all-to-all communication. This implements device-limited routing as described in the DeepSeek
papers. Routing locality is computed as `min(densityLocality, M/ep)` where density-based
locality is `1/(1 + expertsPerRank/numActive)`.

Only DeepSeek V3/R1 (M=4) and V2 (M=6) set this field. The device limit only becomes
meaningful when `EP > numExperts / numActive`.

### MoE Model Examples

| Model | Experts | Active | Shared | Freq | FirstK | Expert Width | Total Params | Active Params |
|-------|---------|--------|--------|------|--------|-------------|-------------|--------------|
| DeepSeek V3/R1 | 256 | 8 | 1 | 1 | 3 | 2048 | ~671B | ~37.6B |
| DeepSeek V2 | 160 | 6 | 2 | 1 | 1 | 1536 | ~236B | ~21.4B |
| DeepSeek MoE-16B | 64 | 6 | 2 | 1 | 1 | 1408 | ~16B | ~2.8B |
| Mistral Large 3 | 128 | 4 | 1 | 1 | 3 | 4096 | ~675B | ~39.9B |
| Kimi K2.5 | 384 | 8 | 1 | 1 | 1 | 2048 | ~1T | ~32.9B |
| LLaMA 4 Maverick | 128 | 1 | 1 | 2 | 0 | 8192 | ~400B | ~17B |
| LLaMA 4 Scout | 16 | 1 | 1 | 1 | 0 | 8192 | ~109B | ~17B |
| Mixtral 8x22B | 8 | 2 | 0 | 1 | 0 | 16384 | ~141B | ~39.2B |
| Mixtral 8x7B | 8 | 2 | 0 | 1 | 0 | 14336 | ~47B | ~12.9B |
| Grok-1 | 8 | 2 | 0 | 1 | 0 | 32768 | ~314B | ~86B |
| Grok 2.5 | 8 | 2 | 2 | 1 | 0 | 16384 | ~270B | ~115B |
| Qwen3 30B-A3B | 128 | 8 | 0 | 1 | 0 | 768 | ~30B | ~3B |
| Qwen3 235B-A22B | 128 | 8 | 0 | 1 | 0 | 1536 | ~235B | ~22B |
| GLM-5 | 256 | 8 | 1 | 1 | 3 | 2048 | ~743B | ~41.1B |
| GLM-4.7 353B | 160 | 8 | 1 | 1 | 3 | 1536 | ~353B | ~33.6B |
| GLM-4.7 Flash | 64 | 4 | 1 | 1 | 1 | 1536 | ~30B | ~3.9B |
| GLM-4.5 Air | 128 | 8 | 1 | 1 | 1 | 1408 | ~107B | ~13.4B |
| MiniMax M2.5 | 256 | 8 | 0 | 1 | 0 | 1536 | ~230B | ~11B |
| GPT-OSS 120B | 128 | 4 | 0 | 1 | 0 | 2880 | ~120B | ~5B |
| GPT-OSS 20B | 32 | 4 | 0 | 1 | 0 | 2880 | ~21B | ~4B |
| DBRX | 16 | 4 | 0 | 1 | 0 | 10752 | ~132B | ~36.5B |

Note: DeepSeek V3/R1 and Mistral Large 3 have 61 layers (prime number), making interleaved
PP scheduling impossible with PP>1. Only standard 1F1B works.

Source: `src/core/models/architectures.ts`, `src/core/models/primitives.ts:createTransformerBlock()`

---

## 5. Parameter Counting

### Dense Models

```
totalParams = embeddingParams + attentionParams + mlpParams + normParams + outputParams
```

Where:
- **Embedding**: `vocabSize * hiddenSize`
- **Attention per layer (MHA/GQA)**: QKV projections + output projection
  - Q: `hiddenSize * numHeads * headDim` (+ bias if enabled)
  - KV: `2 * hiddenSize * numKvHeads * headDim` (+ bias if enabled)
  - Output: `numHeads * headDim * hiddenSize` (+ bias if enabled)
- **MLP per layer (gated)**: `3 * hiddenSize * intermediateSize`
- **MLP per layer (standard)**: `2 * hiddenSize * intermediateSize`
- **Norm per layer**: RMSNorm: `hiddenSize`, LayerNorm: `2 * hiddenSize`
  (2 norms per transformer block + 1 final norm)
- **Output**: `hiddenSize * vocabSize` (0 if tied embeddings)

### MoE Models

```
totalParams = sharedParams + routerParams + allExpertParams

Where:
  sharedParams   = embedding + attention + norms + denseLayerMLPs + sharedExpertMLPs + output
  routerParams   = numMoELayers * hiddenSize * numExperts
  allExpertParams = numMoELayers * numExperts * (mlpMatrices * hiddenSize * expertIntermediateSize)
  mlpMatrices    = 3 (gated) or 2 (standard)
```

### Active Parameters (MoE)

Active parameters represent what participates in each forward pass:

```
activeParams = embedding + attention + norms + output
             + denseLayerMLPs (100% active)
             + routerParams
             + numMoELayers * (numActive + numShared) * paramsPerExpert
```

For dense models: `activeParams = totalParams`.

Source: `src/core/models/primitives.ts:buildModelSpec()`

---

## 6. FLOPs Per Token

FLOPs are computed per token for the forward pass. The total `flopsPerToken` is the sum
of all block FLOPs (attention + MLP + norms) plus the final norm and output projection,
divided by `seqLength`.

### Attention FLOPs (MHA/GQA)

Per layer, for a sequence of length S:

| Operation | FLOPs |
|-----------|-------|
| Q projection | `2 * S * h * (nHeads * headDim)` |
| K projection | `2 * S * h * (nKvHeads * headDim)` |
| V projection | `2 * S * h * (nKvHeads * headDim)` |
| Attention scores (Q @ K^T) | `2 * nHeads * S * headDim * S` |
| Softmax | `5 * nHeads * S * S` |
| Attention output (scores @ V) | `2 * nHeads * S * S * headDim` |
| Output projection | `2 * S * (nHeads * headDim) * h` |

Attention scores and output are quadratic in sequence length -- this is why `getModel(id, seqLength)`
rebuilds the model when the training sequence length differs from `maxSeqLength`.

### MLP FLOPs

Per layer, for a sequence of length S:

**Gated MLP (SwiGLU/GeGLU)**:
```
gate:       2 * S * h * I
up:         2 * S * h * I
SiLU + mul: S * I * 6
down:       2 * S * I * h
```

**Standard MLP**:
```
up:         2 * S * h * I
activation: S * I * 5
down:       2 * S * I * h
```

Where `h = hiddenSize`, `I = intermediateSize`.

### MoE FLOPs

For MoE layers, the MLP FLOPs use `(numActive + numShared)` experts worth of compute with
`expertIntermediateSize` instead of `intermediateSize`. Router FLOPs are negligible.

### Training FLOPs

For training, the standard approximation is:
- **MFU (PaLM definition)**: `6 * P * D / (time * peak)` -- useful work only, never includes recompute
- **HFU**: `(6+2f) * P * D / (time * peak)` where f = recomputeFraction (0 = no AC, 0.13–0.22 = selective, 1.0 = full). LoRA: `(4+2f)`

Where P = `activeParams` for MoE models, D = tokens processed.

Source: `src/core/models/primitives.ts`

---

## 7. Chinchilla Scaling

Chinchilla-optimal training uses: `D_optimal = 20 * activeParams`

For MoE models, this uses `activeParams` (not `totalParams`) because training FLOPs
scale with active parameters. Using `totalParams` would produce nonsensical token counts:

| Model | activeParams | Chinchilla D (active) | If using totalParams |
|-------|-------------|----------------------|---------------------|
| LLaMA 3 8B (dense) | 8B | 160B tokens | 160B (same) |
| DeepSeek V3 | 37.6B | 752B tokens | 13.4T (nonsensical) |
| Grok-1 | ~86B | ~1.7T tokens | ~6.3T (inflated) |

### Token Presets

| Preset | Description |
|--------|-------------|
| Chinchilla | `20 * activeParams` |
| Heavy overtrain | `200 * activeParams` (10x Chinchilla) |
| Finetune | 1B tokens (fixed) |
| Custom | User-specified |

Source: `src/stores/config.ts`

---

## 8. Model Catalog

The registry contains 89 models across 16 families. Models are grouped into families for
the UI selector; some older models are hidden from the dropdown but remain resolvable by ID
(for shared URLs and backward compatibility).

### GPT Family

GPT-3 architecture: MHA, standard (non-gated) MLP, LayerNorm, GeLU, bias enabled, no RoPE.

| ID | Name | Layers | Hidden | Intermediate | Heads | KV Heads | Vocab | MaxSeq |
|----|------|--------|--------|-------------|-------|----------|-------|--------|
| `gpt3-125m` | GPT-3 (125M) | 12 | 768 | 3072 | 12 | 12 | 50257 | 2048 |
| `gpt3-1.3b` | GPT-3 (1.3B) | 24 | 2048 | 8192 | 16 | 16 | 50257 | 2048 |
| `gpt3-6.7b` | GPT-3 (6.7B) | 32 | 4096 | 16384 | 32 | 32 | 50257 | 2048 |
| `gpt3-13b` | GPT-3 (13B) | 40 | 5140 | 20560 | 40 | 40 | 50257 | 2048 |
| `gpt3-175b` | GPT-3 (175B) | 96 | 12288 | 49152 | 96 | 96 | 50257 | 2048 |

### GPT-OSS Family

OpenAI open-source MoE models (2025): GQA, SwiGLU gated MLP, RMSNorm, RoPE+YaRN, bias enabled.
Non-standard headDim=64 (wider attention). All layers are MoE with intentionally narrow expert MLPs.

| ID | Name | Layers | Hidden | Heads | Experts | Active | Expert Width | Vocab | MaxSeq |
|----|------|--------|--------|-------|---------|--------|-------------|-------|--------|
| `gpt-oss-120b` | GPT-OSS (120B) | 36 | 2880 | 64 | 128 | 4 | 2880 | 201088 | 131072 |
| `gpt-oss-20b` | GPT-OSS (20B) | 24 | 2880 | 64 | 32 | 4 | 2880 | 201088 | 131072 |

### LLaMA 2 Family

GQA (70B only, 7B/13B are MHA), gated SwiGLU, RMSNorm, RoPE.

| ID | Name | Layers | Hidden | Intermediate | Heads | KV Heads | Vocab | MaxSeq |
|----|------|--------|--------|-------------|-------|----------|-------|--------|
| `llama2-7b` | LLaMA-2 (7B) | 32 | 4096 | 11008 | 32 | 32 (MHA) | 32000 | 4096 |
| `llama2-13b` | LLaMA-2 (13B) | 40 | 5120 | 13824 | 40 | 40 (MHA) | 32000 | 4096 |
| `llama2-70b` | LLaMA-2 (70B) | 80 | 8192 | 28672 | 64 | 8 (GQA) | 32000 | 4096 |

### LLaMA 3 Family

GQA for all sizes, larger vocab (128K), longer context. Includes LLaMA 3, 3.1, 3.2, and 3.3.

| ID | Name | Layers | Hidden | Intermediate | Heads | KV Heads | Vocab | MaxSeq | Tied |
|----|------|--------|--------|-------------|-------|----------|-------|--------|------|
| `llama3.2-1b` | LLaMA-3.2 (1B) | 16 | 2048 | 8192 | 32 | 8 | 128256 | 131072 | Yes |
| `llama3.2-3b` | LLaMA-3.2 (3B) | 28 | 3072 | 8192 | 24 | 8 | 128256 | 131072 | Yes |
| `llama3.1-8b` | LLaMA-3.1 (8B) | 32 | 4096 | 14336 | 32 | 8 | 128256 | 131072 | No |
| `llama3.3-70b` | LLaMA-3.3 (70B) | 80 | 8192 | 28672 | 64 | 8 | 128256 | 131072 | No |
| `llama3-405b` | LLaMA-3.1 (405B) | 126 | 16384 | 53248 | 128 | 8 | 128256 | 131072 | No |

Hidden (superseded by 3.1): `llama3-8b` (8K context), `llama3-70b` (8K context).

### LLaMA 4 Family

MoE with GQA (q=40, kv=8), headDim=128, shared experts, RMSNorm, SwiGLU, RoPE.

| ID | Name | Layers | Hidden | Experts | Active | Shared | Freq | Expert Width | Vocab | MaxSeq |
|----|------|--------|--------|---------|--------|--------|------|-------------|-------|--------|
| `llama4-scout` | Scout (109B) | 48 | 5120 | 16 | 1 | 1 | 1 | 8192 | 202048 | 10M |
| `llama4-maverick` | Maverick (400B) | 48 | 5120 | 128 | 1 | 1 | 2 | 8192 | 202048 | 1M |

Maverick uses `moeLayerFrequency: 2` -- alternating dense and MoE layers (24 dense + 24 MoE).

### Mistral / Mixtral Family

Dense models: GQA, SwiGLU, RMSNorm, RoPE. MoE models: sparse mixture of experts.
Includes Ministral 3, Devstral 2, and Mistral Large 3 (MLA + MoE).

| ID | Name | Layers | Hidden | Heads | KV Heads | Experts | Active | Vocab | MaxSeq |
|----|------|--------|--------|-------|----------|---------|--------|-------|--------|
| `mistral-large-3-675b` | Mistral Large 3 (675B) | 61 | 7168 | 128 | 128 | 128 | 4 | 131072 | 262144 |
| `devstral-2` | Devstral 2 (123B) | 88 | 12288 | 96 | 8 | -- | -- | 131072 | 262144 |
| `devstral-small-2` | Devstral Small 2 (24B) | 40 | 5120 | 32 | 8 | -- | -- | 131072 | 393216 |
| `ministral-3-14b` | Ministral 3 (14B) | 40 | 5120 | 32 | 8 | -- | -- | 131072 | 262144 |
| `ministral-3-8b` | Ministral 3 (8B) | 34 | 4096 | 32 | 8 | -- | -- | 131072 | 262144 |
| `ministral-3-3b` | Ministral 3 (3B) | 26 | 3072 | 32 | 8 | -- | -- | 131072 | 262144 |
| `mixtral-8x22b` | Mixtral 8x22B (141B) | 56 | 6144 | 48 | 8 | 8 | 2 | 32768 | 65536 |

Mistral Large 3 uses MLA attention (DeepSeek V3 architecture clone) with 128 experts, shared=1, firstK=3.

Hidden: `mistral-7b`, `mistral-nemo-12b`, `codestral-22b`, `mistral-small-24b`,
`mistral-large-123b`, `mixtral-8x7b`.

### DeepSeek Family

MoE with fine-grained experts, shared experts, MLA (V2/V3/R1), and device-limited routing.

| ID | Name | Layers | Hidden | Attn | Experts | Active | Shared | FirstK | Vocab | MaxSeq |
|----|------|--------|--------|------|---------|--------|--------|--------|-------|--------|
| `deepseek-v3` | V3/R1 (671B) | 61 | 7168 | MLA | 256 | 8 | 1 | 3 | 129280 | 163840 |
| `deepseek-v2` | V2 (236B) | 60 | 5120 | MLA | 160 | 6 | 2 | 1 | 102400 | 128000 |
| `deepseek-moe-16b` | MoE (16B) | 28 | 2048 | MHA | 64 | 6 | 2 | 1 | 102400 | 4096 |

Hidden: `deepseek-r1` (identical architecture to V3, only `maxSeqLength` differs).

V3/R1 have 61 layers (prime) -- interleaved PP scheduling is impossible with PP>1.

### Kimi Family

DeepSeek V3 variant with 384 experts (50% more than V3), MLA attention with identical dimensions.

| ID | Name | Layers | Hidden | Attn | Experts | Active | Shared | FirstK | Vocab | MaxSeq |
|----|------|--------|--------|------|---------|--------|--------|--------|-------|--------|
| `kimi-k2.5` | Kimi K2.5 (1T) | 61 | 7168 | MLA | 384 | 8 | 1 | 1 | 163840 | 131072 |

Key differences from V3: 384 experts (vs 256), 64 attention heads (vs 128), 1 dense layer (vs 3).

### Qwen 2.5 Family

GQA, RMSNorm, SiLU, gated MLP, RoPE, bias enabled.

| ID | Name | Layers | Hidden | Intermediate | Heads | KV Heads | Vocab | MaxSeq |
|----|------|--------|--------|-------------|-------|----------|-------|--------|
| `qwen2.5-0.5b` | Qwen2.5 (0.5B) | 24 | 896 | 4864 | 14 | 2 | 151936 | 32768 |
| `qwen2.5-1.5b` | Qwen2.5 (1.5B) | 28 | 1536 | 8960 | 12 | 2 | 151936 | 32768 |
| `qwen2.5-3b` | Qwen2.5 (3B) | 36 | 2048 | 11008 | 16 | 2 | 151936 | 32768 |
| `qwen2.5-7b` | Qwen2.5 (7B) | 28 | 3584 | 18944 | 28 | 4 | 152064 | 32768 |
| `qwen2.5-14b` | Qwen2.5 (14B) | 48 | 5120 | 13824 | 40 | 8 | 152064 | 131072 |
| `qwen2.5-32b` | Qwen2.5 (32B) | 64 | 5120 | 27648 | 40 | 8 | 152064 | 32768 |
| `qwen2.5-72b` | Qwen2.5 (72B) | 80 | 8192 | 29568 | 64 | 8 | 152064 | 32768 |

### Qwen 3 Family

Dense models: GQA(kv=8), no bias, tied embeddings (small sizes), headDim=128 override.
MoE models: 128 experts, 8 active, GQA(kv=4), no shared experts.

| ID | Name | Layers | Hidden | Intermediate | Heads | KV Heads | Vocab | MaxSeq |
|----|------|--------|--------|-------------|-------|----------|-------|--------|
| `qwen3-0.6b` | Qwen3 (0.6B) | 28 | 1024 | 3072 | 16 | 8 | 151936 | 40960 |
| `qwen3-1.7b` | Qwen3 (1.7B) | 28 | 2048 | 6144 | 16 | 8 | 151936 | 40960 |
| `qwen3-4b` | Qwen3 (4B) | 36 | 2560 | 9728 | 32 | 8 | 151936 | 40960 |
| `qwen3-8b` | Qwen3 (8B) | 36 | 4096 | 12288 | 32 | 8 | 151936 | 40960 |
| `qwen3-14b` | Qwen3 (14B) | 40 | 5120 | 17408 | 40 | 8 | 151936 | 40960 |
| `qwen3-32b` | Qwen3 (32B) | 64 | 5120 | 25600 | 64 | 8 | 151936 | 40960 |
| `qwen3-30b-a3b` | Qwen3 MoE (30B-A3B) | 48 | 2048 | -- | 32 | 4 | 151936 | 40960 |
| `qwen3-235b-a22b` | Qwen3 MoE (235B-A22B) | 94 | 4096 | -- | 64 | 4 | 151936 | 40960 |

MoE models have very small per-expert FFN width (768 for 30B-A3B, 1536 for 235B-A22B) with
128 fine-grained experts. Small expert widths are modeled by `getGroupedGemmEfficiency()`
in `gpu.ts`, which applies a power-law throughput penalty below a 1536-width threshold.

### Gemma Family

Google Gemma models. Gemma 2: GeGLU, GQA, tied embeddings, 256K vocab, 8K context.
Gemma 3: GeLU, decoupled headDim, larger vocab (262K), longer context, sliding window attention
(not modeled).

**Gemma 2:**

| ID | Name | Layers | Hidden | Intermediate | Heads | KV Heads | Vocab | MaxSeq |
|----|------|--------|--------|-------------|-------|----------|-------|--------|
| `gemma2-2b` | Gemma 2 (2B) | 26 | 2304 | 9216 | 8 | 4 | 256000 | 8192 |
| `gemma2-9b` | Gemma 2 (9B) | 42 | 3584 | 14336 | 16 | 8 | 256000 | 8192 |
| `gemma2-27b` | Gemma 2 (27B) | 46 | 4608 | 36864 | 32 | 16 | 256000 | 8192 |

**Gemma 3:**

| ID | Name | Layers | Hidden | Intermediate | Heads | KV Heads | headDim | Vocab | MaxSeq |
|----|------|--------|--------|-------------|-------|----------|---------|-------|--------|
| `gemma3-1b` | Gemma 3 (1B) | 26 | 1152 | 6912 | 4 | 1 (MQA) | 256 | 262144 | 32768 |
| `gemma3-4b` | Gemma 3 (4B) | 34 | 2560 | 10240 | 8 | 4 | 256 | 262208 | 131072 |
| `gemma3-12b` | Gemma 3 (12B) | 48 | 3840 | 15360 | 16 | 8 | 256 | 262208 | 131072 |
| `gemma3-27b` | Gemma 3 (27B) | 62 | 5376 | 21504 | 32 | 16 | 128 | 262208 | 131072 |

### GLM Family

Zhipu AI / THUDM models. Dense (GLM-4) and MoE (GLM-4.5 Air, GLM-4.7, GLM-5).
GLM-4.7 Flash and GLM-5 use MLA attention.

| ID | Name | Layers | Hidden | Attn | Experts | Active | Shared | Vocab | MaxSeq |
|----|------|--------|--------|------|---------|--------|--------|-------|--------|
| `glm5` | GLM-5 (743B) | 78 | 6144 | MLA | 256 | 8 | 1 | 154880 | 202752 |
| `glm4.7` | GLM-4.7 (353B) | 92 | 5120 | GQA | 160 | 8 | 1 | 151552 | 202752 |
| `glm4.7-flash` | GLM-4.7 Flash (30B) | 47 | 2048 | MLA | 64 | 4 | 1 | 154880 | 202752 |
| `glm4.5-air` | GLM-4.5 Air (107B) | 46 | 4096 | GQA | 128 | 8 | 1 | 151552 | 131072 |
| `glm4-32b` | GLM-4 (32B) | 61 | 6144 | GQA | -- | -- | -- | 151552 | 32768 |
| `glm4-9b` | GLM-4 (9B) | 40 | 4096 | GQA | -- | -- | -- | 151552 | 131072 |

### Grok Family

xAI models. MoE with GeGLU (gated GeLU), GQA, RMSNorm, RoPE.

| ID | Name | Layers | Hidden | Heads | KV Heads | Experts | Active | Shared | Vocab | MaxSeq |
|----|------|--------|--------|-------|----------|---------|--------|--------|-------|--------|
| `grok-2.5` | Grok 2.5 (270B) | 64 | 8192 | 64 | 8 | 8 | 2 | 2 | 131072 | 131072 |
| `grok-1` | Grok-1 (314B) | 64 | 6144 | 48 | 8 | 8 | 2 | 0 | 131072 | 8192 |

Grok 2.5 uses residual MoE: every layer has both a shared dense FFN and 8 sparse experts.
The shared FFN is modeled as 2 shared experts (equivalent parameter count).

### MiniMax Family

MiniMax M2.5: MoE with 256 experts (8 active, 0 shared), all 62 layers are MoE.
Non-standard headDim=128 (natural 3072/48=64).

| ID | Name | Layers | Hidden | Heads | Experts | Active | Expert Width | Vocab | MaxSeq |
|----|------|--------|--------|-------|---------|--------|-------------|-------|--------|
| `minimax-m2.5` | MiniMax-M2.5 (230B) | 62 | 3072 | 48 | 256 | 8 | 1536 | 200064 | 196608 |

### Phi Family

Microsoft Phi-3 and Phi-4 models. RMSNorm (except Phi-4 Mini Flash: LayerNorm), gated MLP, RoPE.

**Phi-3:**

| ID | Name | Layers | Hidden | Intermediate | Heads | KV Heads | Vocab | MaxSeq |
|----|------|--------|--------|-------------|-------|----------|-------|--------|
| `phi3-mini` | Phi-3 Mini (3.8B) | 32 | 3072 | 8192 | 32 | 32 (MHA) | 32064 | 131072 |
| `phi3-small` | Phi-3 Small (7B) | 32 | 4096 | 14336 | 32 | 8 | 100352 | 131072 |
| `phi3-medium` | Phi-3 Medium (14B) | 40 | 5120 | 17920 | 40 | 10 | 32064 | 131072 |

**Phi-4:**

| ID | Name | Layers | Hidden | Intermediate | Heads | KV Heads | Vocab | MaxSeq | Tied |
|----|------|--------|--------|-------------|-------|----------|-------|--------|------|
| `phi4` | Phi-4 (14B) | 40 | 5120 | 17920 | 40 | 10 | 100352 | 16384 | No |
| `phi4-mini` | Phi-4 Mini (3.8B) | 32 | 3072 | 8192 | 24 | 8 | 200064 | 131072 | Yes |
| `phi4-mini-flash` | Phi-4 Mini Flash (2.5B) | 32 | 2560 | 10240 | 40 | 20 | 200064 | 262144 | Yes |

Phi-4 Mini Flash uses LayerNorm (not RMSNorm) and no RoPE.

### OLMo Family

AI2 fully open models. MHA (7B sizes) or GQA (32B), RMSNorm, SwiGLU, RoPE.

**OLMo 2:**

| ID | Name | Layers | Hidden | Intermediate | Heads | KV Heads | Vocab | MaxSeq |
|----|------|--------|--------|-------------|-------|----------|-------|--------|
| `olmo2-7b` | OLMo 2 (7B) | 32 | 4096 | 11008 | 32 | 32 (MHA) | 100278 | 4096 |
| `olmo2-13b` | OLMo 2 (13B) | 40 | 5120 | 13824 | 40 | 40 (MHA) | 100278 | 4096 |
| `olmo2-32b` | OLMo 2 (32B) | 64 | 5120 | 27648 | 40 | 8 (GQA) | 100352 | 4096 |

**OLMo 3:**

| ID | Name | Layers | Hidden | Intermediate | Heads | KV Heads | Vocab | MaxSeq |
|----|------|--------|--------|-------------|-------|----------|-------|--------|
| `olmo3-7b` | OLMo 3 (7B) | 32 | 4096 | 11008 | 32 | 32 (MHA) | 100278 | 65536 |
| `olmo3-32b` | OLMo 3 (32B) | 64 | 5120 | 27648 | 40 | 8 (GQA) | 100278 | 65536 |

OLMo 3 7B is identical to OLMo 2 7B except for maxSeqLength (4096 to 65536).

### Nemotron-4 Family

NVIDIA models. Non-standard architecture: ReLU-squared (approximated as relu), non-gated MLP,
LayerNorm, GQA, RoPE.

| ID | Name | Layers | Hidden | Intermediate | Heads | KV Heads | Vocab | MaxSeq |
|----|------|--------|--------|-------------|-------|----------|-------|--------|
| `nemotron-4-15b` | Nemotron-4 (15B) | 32 | 6144 | 24576 | 48 | 8 | 256000 | 4096 |
| `nemotron-4-340b` | Nemotron-4 (340B) | 96 | 18432 | 73728 | 96 | 8 | 256000 | 4096 |

### Hidden Models

These models are resolvable by ID but hidden from the UI selector (superseded by newer versions
or pruned from the dropdown):

| ID | Name | Family | Why Hidden |
|----|------|--------|-----------|
| `mistral-7b` | Mistral (7B) | mistral | Superseded by Ministral 3 |
| `mistral-nemo-12b` | Mistral Nemo (12B) | mistral | Superseded by Ministral 3 |
| `codestral-22b` | Codestral (22B) | mistral | Superseded by Devstral 2 |
| `mistral-small-24b` | Mistral Small (24B) | mistral | Superseded by Devstral Small 2 |
| `mistral-large-123b` | Mistral Large (123B) | mistral | Superseded by Mistral Large 3 |
| `mixtral-8x7b` | Mixtral 8x7B (47B) | mistral | Older MoE, kept for benchmarks |
| `llama3-8b` | LLaMA-3 (8B) | llama3 | Superseded by LLaMA 3.1 |
| `llama3-70b` | LLaMA-3 (70B) | llama3 | Superseded by LLaMA 3.3 |
| `deepseek-r1` | DeepSeek-R1 (671B) | deepseek | Same architecture as V3 |
| `dbrx` | DBRX (132B) | dbrx | Databricks, less commonly used |
| `yi-6b` | Yi (6B) | yi | 01.AI, less commonly used |
| `yi-34b` | Yi (34B) | yi | 01.AI, less commonly used |
| `command-r` | Command R (35B) | command-r | Cohere, less commonly used |
| `command-r-plus` | Command R+ (104B) | command-r | Cohere, less commonly used |
| `megatron-turing-530b` | Megatron-Turing NLG (530B) | gpt3 | GPT-3 architecture, less commonly used |
| `bloom-176b` | BLOOM (176B) | bloom | BigScience, less commonly used |
| `qwen2-57b-a14b` | Qwen2 MoE (57B-A14B) | qwen | Superseded by Qwen3 MoE variants |
| `olmo2-32b` | OLMo 2 (32B) | olmo2 | Backend-only, benchmark calibration |

### UI Family Grouping

Models in the UI selector are organized into these families (alphabetical, largest first within each):

1. **DeepSeek** -- V3, V2, MoE-16B
2. **Gemma** -- Gemma 3 (27B, 12B, 4B, 1B), Gemma 2 (27B, 9B, 2B)
3. **GLM** -- GLM-5, GLM-4.7, GLM-4.7 Flash, GLM-4.5 Air, GLM-4 32B, GLM-4 9B
4. **GPT** -- GPT-OSS 120B, GPT-OSS 20B, GPT-3 175B/13B/6.7B/1.3B/125M
5. **Grok** -- Grok 2.5, Grok-1
6. **Kimi** -- K2.5
7. **LLaMA-4** -- Maverick, Scout
8. **LLaMA-3** -- 3.3-70B, 3.2-3B, 3.2-1B, 3.1-405B, 3.1-8B
9. **LLaMA-2** -- 70B, 13B, 7B
10. **MiniMax** -- M2.5
11. **Mistral** -- Large 3 675B, Ministral 3 (14B, 8B, 3B), Devstral 2, Devstral Small 2, Mixtral 8x22B
12. **Nemotron-4** -- 340B, 15B
13. **OLMo** -- OLMo 3 (32B, 7B), OLMo 2 (13B, 7B)
14. **Phi** -- Phi-4 (14B, Mini, Mini Flash), Phi-3 (Medium, Small, Mini)
15. **Qwen 3** -- 235B-A22B, 32B, 30B-A3B, 14B, 8B, 4B, 1.7B, 0.6B
16. **Qwen 2.5** -- 72B, 32B, 14B, 7B, 3B, 1.5B, 0.5B
