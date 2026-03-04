import type { GameTask } from '../types.ts';
// Glossary annotations: {{termId}} or {{termId|display text}} for tooltip terms.
// [text](url) for paper links in successExplanation.
// Annotate FIRST OCCURRENCE per difficulty tier only. Check earlier tasks before adding.

export const TRAINING_ADVANCED_TASKS: GameTask[] = [
  // -----------------------------------------------------------------------
  // 1. 3D Parallelism
  // -----------------------------------------------------------------------
  {
    id: 'training-advanced-01',
    mode: 'training',
    difficulty: 'advanced',
    order: 0,
    title: '3D Parallelism',
    concept: 'Tensor + Pipeline + Data Parallelism Combined',
    learningObjectives: [
      'Combine `TP` + `PP` + `DP` for 100B+ parameter models where no single dimension suffices',
      'Know published GPT-3 175B config (`TP`=8, `PP`=8 on 1024 A100s) as a reference point',
      'Understand `ZeRO-1` + `PP` reduces replicated weights per GPU, making 175B feasible',
      'Recognize interleaved scheduling as critical for reducing pipeline bubbles at large `PP`',
    ],
    briefing:
      'You have a 128-GPU H100 cluster and a mission: ' +
      'train GPT-3 175B. With {{zero1|ZeRO-1}}, optimizer state is sharded but parameters ' +
      'and gradients are replicated per {{dp|DP}} rank. At `TP=8` and `PP=1`, each GPU holds ' +
      '`175B/8 ≈ 22B` params — 44 GB of weights plus 88 GB of `FP32` gradients, far ' +
      'exceeding 80 GB. You need all three dimensions: {{tp|TP}} to split layers within ' +
      'a node, {{pp|PP}} to reduce per-GPU replicated memory, and DP to scale throughput. ' +
      'Find a configuration that balances all three and achieves solid {{mfu|MFU}}.',
    setup: {
      modelId: 'gpt3-175b',
      gpuId: 'h100-sxm',
      numGPUs: 128,
      strategyType: 'zero1-tp-pp',
      mixedPrecision: 'bf16',
      flashAttention: true,
      activationCheckpointing: true,
      sequenceParallel: true,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'No OOM' },
      { field: 'mfu', operator: '>', value: 0.40, label: 'MFU > 40%' },
    ],
    expectedChanges: [
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      '`TP` should stay within a node to use fast {{nvlink|NVLink}}. Without `PP`, replicated weights and gradients exceed per-GPU memory — you need all three dimensions.',
      'With `TP` filling each node, the remaining `TP` groups must be split between `PP` and `DP`. `PP` reduces per-GPU replicated memory, but too much `PP` cuts `DP` throughput. Try different `PP` values to find the balance.',
      'Enable interleaved `1F1B` pipeline schedule to reduce the bubble fraction. Activation checkpointing helps with memory if PP is small.',
    ],
    successExplanation:
      '{{3d-parallel|3D parallelism}} is the standard recipe for training 100B+ parameter models. ' +
      '`TP` handles the intra-layer split (fast `NVLink`), `PP` handles the inter-layer split ' +
      '(cross-node, pipelined), and DP scales throughput. With `ZeRO-1`, `PP` is essential ' +
      'because it reduces the replicated parameter and gradient memory per GPU.\n\nThe ' +
      'published [GPT-3 (Brown et al., 2020)](https://arxiv.org/abs/2005.14165) config used `TP=8`, `PP=8` on 1024 A100s — the same principles ' +
      'apply here with fewer GPUs. Interleaved scheduling is critical for reducing ' +
      'pipeline bubbles as `PP` grows.',
  },

  // -----------------------------------------------------------------------
  // 2. Expert Parallelism
  // -----------------------------------------------------------------------
  {
    id: 'training-advanced-02',
    mode: 'training',
    difficulty: 'advanced',
    order: 2,
    title: 'Expert Parallelism',
    concept: 'Sparse model sharding and token routing',
    learningObjectives: [
      'Understand `EP` subdivides the `DP` dimension: each GPU holds a fraction of experts',
      'Know `EP` uses all-to-all communication to route tokens to the correct GPU',
      'Understand device-limited routing (M=4 for DeepSeek V3) bounds all-to-all communication volume',
      'Know `EP` reduces per-GPU memory and expert compute, enabling larger `MoE` models',
    ],
    briefing:
      'DeepSeek V3 is a 671B-parameter {{moe|Mixture-of-Experts}} model with 256 routed ' +
      'experts, but only ~37B {{active-params|active parameters}} per token. The experts are ' +
      'the bulk of the model weight — and {{ep|Expert Parallelism (EP)}} lets you {{sharding|shard}} ' +
      'them across GPUs so each device only holds a fraction of the expert weights. ' +
      '`EP` subdivides the Data Parallel dimension: with `DP=8` and `EP=4`, each expert ' +
      'group spans 4 GPUs and gradient averaging happens across `DP/EP=2` replicas. ' +
      'The tradeoff is {{all-to-all}} communication to route tokens to the right expert. ' +
      'Your task: get DeepSeek V3 running efficiently on 256 H100s using EP.',
    setup: {
      modelId: 'deepseek-v3',
      gpuId: 'h100-sxm',
      numGPUs: 256,
      strategyType: 'fsdp-tp',
      mixedPrecision: 'bf16',
      flashAttention: true,
      activationCheckpointing: true,
      sequenceParallel: true,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'No OOM' },
      { field: 'mfu', operator: '>', value: 0.20, label: 'MFU > 20%' },
    ],
    expectedChanges: [
      { field: 'epDegree', check: 'increased', label: 'Increased EP degree' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'Set `EP` in the strategy configuration. `EP` must divide the `DP` degree. Higher `EP` means each GPU holds fewer experts, reducing memory and per-GPU compute.',
      'DeepSeek V3 has 61 layers (a prime number), which makes `PP` difficult. Focus on `TP` + `EP` within `FSDP-TP`.',
      'Try `FP8` mixed precision — DeepSeek V3 was designed for `FP8` training with Transformer Engine. Also try a 3D strategy (`fsdp-tp-pp`) if memory is tight.',
    ],
    successExplanation:
      'Expert Parallelism is essential for large `MoE` models. Without `EP`, every GPU ' +
      'must hold all 256 experts in memory even though only 8 are active per token. ' +
      '`EP=8` means each GPU holds 32 experts, dramatically reducing memory.\n\nThe cost ' +
      'is all-to-all communication to dispatch tokens to the correct GPU. DeepSeek V3 ' +
      'uses {{device-limited-routing|device-limited routing}} (`M=4`) to keep this communication manageable — tokens ' +
      'are only sent to at most 4 devices regardless of `EP` degree.',
  },

  // -----------------------------------------------------------------------
  // 3. Context Parallelism
  // -----------------------------------------------------------------------
  {
    id: 'training-advanced-03',
    mode: 'training',
    difficulty: 'advanced',
    order: 3,
    title: 'Context Parallelism',
    concept: 'Long-sequence memory bottlenecks',
    learningObjectives: [
      'Understand `CP` splits the sequence dimension across GPUs for long-context training',
      'Know ring attention pipelines `KV` block transfers with attention computation',
      'Recognize `CP` is only useful when activation memory from long sequences is the bottleneck',
      'Know `CP` x `TP` x `PP` x `DP` must not exceed totalGPUs',
    ],
    briefing:
      'LLaMA 3.1 405B was trained with a context window of up to 128K tokens. ' +
      'At that sequence length, {{activation-memory|attention activations}} alone consume enormous memory — ' +
      'far more than model weights. {{cp|Context Parallelism (CP)}} splits the sequence ' +
      'dimension across GPUs: each GPU processes a chunk of the sequence and ' +
      'communicates `KV` blocks via {{ring-attention|ring attention}} or all-gather. Your challenge: ' +
      'LLaMA 3.1 405B on 512 H100s with `TP=8` and `PP=8` at a 32K sequence length `OOM`s — ' +
      'the activation memory at 32K tokens is too large even with full activation ' +
      'checkpointing. Enable CP to split the sequence, make it fit, and achieve ' +
      'reasonable training throughput.',
    setup: {
      modelId: 'llama3-405b',
      gpuId: 'h100-sxm',
      numGPUs: 512,
      strategyType: 'fsdp-tp-pp',
      mixedPrecision: 'bf16',
      flashAttention: true,
      activationCheckpointing: true,
      sequenceLength: 32768,
      tpDegree: 8,
      ppDegree: 8,
      sequenceParallel: true,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'No OOM' },
      { field: 'mfu', operator: '>', value: 0.25, label: 'MFU > 25%' },
    ],
    expectedChanges: [
      { field: 'cpDegree', check: 'increased', label: 'Increased CP degree' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'The default config OOMs because activation memory at 32K tokens is enormous even with full activation checkpointing. Look for a parallelism dimension that splits the sequence across GPUs to reduce per-GPU activation memory.',
      'Context Parallelism splits the sequence dimension. `TP × PP × CP × DP` must not exceed total GPUs — calculate the resulting `DP` after adding `CP`. More `CP` means less `DP`, which means more gradient accumulation steps per batch — and more micro-batches reduce the pipeline bubble fraction. `CP` is only effective when seqLength/`CP` >= 8192.',
      'Try adding `CP`. Ring attention (the default implementation) overlaps `KV` transfer with attention computation, making `CP` nearly free when compute dominates.',
    ],
    successExplanation:
      '`CP` is the fourth dimension of parallelism, specifically ' +
      'designed for long-context training. It shards the sequence so each GPU only ' +
      'stores activations for its chunk. Meta used `CP=16` for the 128K stage of ' +
      '[LLaMA 3 (Meta AI, 2024)](https://arxiv.org/abs/2407.21783) training.\n\n[Ring Attention (Liu et al., 2023)](https://arxiv.org/abs/2310.01889) cleverly pipelines `KV` block transfers with ' +
      'attention computation, making `CP` nearly free when compute dominates. The key ' +
      'insight: `CP` is only useful when sequence length is long enough that activation ' +
      'memory is the bottleneck — for short sequences, it just adds overhead.\n\n' +
      'A subtler effect: more `CP` means less `DP` (since `TP × PP × CP × DP = totalGPUs`), ' +
      'which increases gradient accumulation steps per batch. More micro-batches shrink the ' +
      'pipeline bubble fraction `(PP-1)/(PP-1+m)`, improving throughput. With `PP=8`, this ' +
      'effect is significant.',
  },

  // -----------------------------------------------------------------------
  // 4. The Communication Budget
  // -----------------------------------------------------------------------
  {
    id: 'training-advanced-04',
    mode: 'training',
    difficulty: 'advanced',
    order: 1,
    title: 'The Communication Budget',
    concept: 'Compute-to-Communication Ratio and TP Degree',
    learningObjectives: [
      'Understand compute-to-communication ratio (C/T) as the key distributed training metric',
      'Know that higher `TP` shrinks per-GPU matmuls while communication stays constant, worsening C/T',
      'Understand reducing `TP` increases per-GPU compute and improves C/T, at the cost of more memory',
      'Know overlap engineering: when compute >> communication, transfers hide behind matmuls',
    ],
    briefing:
      'Every distributed training job spends time on communication — {{allreduce|AllReduce}} for `TP`, ' +
      '{{allgather|AllGather}} for {{fsdp|FSDP}}, point-to-point for `PP`. The goal is to hide as much ' +
      'communication as possible behind computation. The compute-to-communication ratio (C/T) ' +
      'determines how well this works.\n\n' +
      'On a 64-GPU H100 cluster with LLaMA 3.3 70B, the current config uses `TP=4`. ' +
      'Run it and observe the `MFU`. The communication overhead is eating into training efficiency. ' +
      'Adjust the `TP` degree to improve the C/T ratio and push `MFU` above 42%.',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'h100-sxm',
      numGPUs: 64,
      strategyType: 'fsdp-tp',
      mixedPrecision: 'bf16',
      flashAttention: true,
      activationCheckpointing: true,
      tpDegree: 4,
      globalBatchSize: 128,
      sequenceParallel: true,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'No OOM' },
      { field: 'mfu', operator: '>', value: 0.42, label: 'MFU > 42%' },
    ],
    expectedChanges: [
      { field: 'tpDegree', check: 'changed', label: 'Changed TP degree' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'Each `TP` rank computes only 1/`TP` of each layer. Higher `TP` means smaller matmuls and more exposed communication. Lower `TP` means larger matmuls that can better hide communication — but uses more memory per GPU.',
      'Think about the extreme cases: at `TP=1`, there is zero `TP` communication but each GPU must hold more of the model. At `TP=8`, communication dominates. Where is the best balance for this model and cluster?',
    ],
    successExplanation:
      'Communication overhead is the central cost of distributed training. The key ' +
      'lever is the compute-to-communication ratio (C/T): when compute per layer is ' +
      'much larger than communication per layer, the GPU can overlap transfers behind ' +
      'matrix multiplications.\n\nEach `TP` degree splits layers into smaller per-GPU matmuls ' +
      'while `TP` `AllReduce` volume stays roughly constant. Reducing `TP` gives each GPU ' +
      'larger matmuls that run long enough to hide the communication behind useful work. ' +
      'The tradeoff is more memory per GPU, which `FSDP` can help manage.',
  },

  // -----------------------------------------------------------------------
  // 5. LoRA at Scale
  // -----------------------------------------------------------------------
  {
    id: 'training-advanced-05',
    mode: 'training',
    difficulty: 'advanced',
    order: 4,
    title: 'LoRA at Scale',
    concept: 'Frozen weights and trainable adapters',
    learningObjectives: [
      'Understand `LoRA`: frozen base model + small trainable adapter matrices (A, B)',
      'Know `LoRA` uses `4PD` FLOPs (no backward through frozen weights) vs `6PD` for full fine-tuning',
      'Know optimizer states only apply to adapter params (~0.1-3% of total depending on target modules)',
      'Understand target module choices: q_v (minimal), q_k_v_o (balanced), all_linear (maximum)',
    ],
    briefing:
      'Up to now, every task has focused on pre-training from scratch. But most practitioners fine-tune ' +
      'existing models for specific tasks.\n\n' +
      'Full {{fine-tuning}} of LLaMA 3.3 70B requires storing full model weights, gradients, ' +
      'and {{optimizer-states}} — roughly 18 bytes per parameter, or over 1.2 TB total. ' +
      '{{lora|LoRA}} (Low-Rank Adaptation) freezes the base model and trains small {{adapter-matrices|adapter ' +
      'matrices}}, slashing the trainable parameter count to ~1-3% of the original. ' +
      'This dramatically reduces optimizer memory (only adapter params need Adam states) ' +
      'and gradient memory. Your task: get LLaMA 3.3 70B fine-tuning running on 8 H100s ' +
      'with `FSDP`, keeping memory utilization under 90%.',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'h100-sxm',
      numGPUs: 8,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
      flashAttention: true,
      activationCheckpointing: true,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'No OOM' },
      { field: 'memoryUtilization', operator: '<', value: 0.90, label: 'Memory < 90%' },
    ],
    expectedChanges: [
      { field: 'finetuningMethod', check: 'changed', label: 'Changed fine-tuning method' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'Full fine-tuning updates every parameter, which means optimizer states (Adam\'s momentum and variance) must be stored for all of them. When the model\'s knowledge is already learned from pre-training, do you really need to update every single weight?',
      'Parameter-efficient methods freeze the pre-trained weights and only train small additional matrices. Optimizer states and gradients are only needed for the tiny trainable portion — dramatically reducing memory. Different target module choices control how many adapter pairs are inserted.',
      'Explore the Fine-tuning section in the sidebar. Switching from full fine-tuning to a parameter-efficient method should dramatically reduce memory.',
    ],
    successExplanation:
      '[LoRA (Hu et al., 2021)](https://arxiv.org/abs/2106.09685) works by inserting low-rank decomposition matrices (A and B) alongside ' +
      'frozen pre-trained weights. During fine-tuning, only A and B are updated. ' +
      'For a 70B model with rank 16 targeting Q/K/V/O projections, trainable params ' +
      'drop to ~0.1% of total.\n\nThis means optimizer states (12 bytes/param for Adam) ' +
      'only apply to the tiny adapter, saving hundreds of GB across the cluster. ' +
      'The `MFU` formula also changes: `LoRA` uses `4PD` (no backward pass through frozen ' +
      'weights) instead of `6PD` for full fine-tuning.',
  },

  // -----------------------------------------------------------------------
  // 6. QLoRA on Budget GPUs
  // -----------------------------------------------------------------------
  {
    id: 'training-advanced-06',
    mode: 'training',
    difficulty: 'advanced',
    order: 5,
    title: 'QLoRA on Budget GPUs',
    concept: '4-bit base weights with full-precision adapters',
    learningObjectives: [
      'Understand `NF4` quantization: ~0.5 bytes/param for frozen base model (4x reduction from `BF16`)',
      'Know adapters stay in `BF16` for training stability',
      'Understand dequantization (`NF4` -> `BF16`) during forward pass adds ~5-50ms latency',
      'Recognize `QLoRA` enables fine-tuning 70B on 4x 48GB GPUs — democratizing large model adaptation',
    ],
    briefing:
      'Not everyone has H100s. The {{l40s|L40S}} is a 48 GB Ada Lovelace GPU — powerful for ' +
      'inference but tight for training large models. LLaMA 3.3 70B in `BF16` needs ~140 GB ' +
      'just for weights, far beyond 4 GPUs with 192 GB total memory. Even with `FSDP` sharding ' +
      'and `LoRA` from the previous task, the frozen base weights at 2 bytes per parameter still ' +
      'dominate memory. {{qlora|QLoRA}} takes this further: {{nf4|NormalFloat (NF4)}} quantization ' +
      'stores frozen weights at roughly 0.5 bytes per parameter instead of 2 — a 4× reduction. ' +
      'Your mission: fit LLaMA 3.3 70B on these 4 L40S GPUs.',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'l40s',
      numGPUs: 4,
      gpusPerNode: 4,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
      flashAttention: true,
      activationCheckpointing: true,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'No OOM' },
      { field: 'memoryUtilization', operator: '<', value: 1.0, label: 'Fits in memory' },
    ],
    expectedChanges: [
      { field: 'finetuningMethod', check: 'changed', label: 'Changed fine-tuning method' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'The previous task used `LoRA`, which freezes the base model but still stores it in `BF16`. If the weights are frozen and never updated, do they really need full precision? Quantizing them could free enormous amounts of memory.',
      'The Fine-tuning section offers methods beyond standard `LoRA`. One variant combines adapter training with aggressive base model quantization — storing frozen weights in 4-bit format.',
      'With `NF4` quantization, 70B frozen parameters drop from ~140 GB to ~36 GB for the base weights. `FSDP` shards that across your GPUs. Activation checkpointing can help if memory is still tight after switching methods.',
    ],
    successExplanation:
      '[QLoRA (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314) combines two ideas: 4-bit NormalFloat {{quantization}} for the frozen base ' +
      'model and `LoRA` adapters trained in full precision. The base model consumes ' +
      '~0.515 bytes per parameter (`NF4` + quantization metadata), down from 2 bytes ' +
      'in `BF16`. For 70B parameters, that is ~36 GB total or ~9 GB per GPU with `FSDP` ' +
      'across 4 GPUs. The adapters and their Adam states are tiny by comparison.\n\n' +
      'A dequantization step converts `NF4` back to `BF16` during the forward pass — ' +
      'bandwidth-bound but fast (~5ms for 7B, ~50ms for 70B). This makes fine-tuning ' +
      '70B models feasible on hardware that could never do full training.',
  },

  // -----------------------------------------------------------------------
  // 7. Reproducing DeepSeek V3
  // -----------------------------------------------------------------------
  {
    id: 'training-advanced-07',
    mode: 'training',
    difficulty: 'advanced',
    order: 7,
    title: 'Reproducing DeepSeek V3',
    concept: 'Composing all parallelism dimensions',
    learningObjectives: [
      'Apply all parallelism dimensions simultaneously: `TP`=4, `PP`=8, `EP`=32, `FP8`',
      'Understand H800 limitations (`NVLink` 400 GB/s vs H100 900 GB/s) and `FP8` as partial mitigation',
      'Know Chinchilla scaling: compute-optimal training uses D ≈ 20 × active_params tokens',
      'Understand overtraining economics: D >> 20N is deliberate when inference cost dominates',
    ],
    briefing:
      'DeepSeek published that V3 (671B `MoE`, 37B active) achieved 43.7% `MFU` on ' +
      '2048 {{h800|H800}} GPUs using {{fp8|FP8}} training with a custom DualPipe schedule. The H800 ' +
      'has the same compute as H100 but reduced `NVLink` bandwidth. Your challenge: reproduce this result on 256 H800 ' +
      'nodes (2048 GPUs). The published config used `TP=4, PP=8, EP=32`, and `FP8` ' +
      'mixed precision. Can you match their `MFU`? This is a masterclass in how all ' +
      'the dimensions of parallelism interact — `TP` for intra-layer, `PP` for inter-layer, ' +
      '`EP` for expert sharding, and `FSDP` for the rest.',
    setup: {
      modelId: 'deepseek-v3',
      gpuId: 'h800-sxm',
      numGPUs: 2048,
      strategyType: 'fsdp-tp-pp',
      mixedPrecision: 'bf16',
      flashAttention: true,
      activationCheckpointing: true,
      sequenceParallel: true,
    },
    winningCriteria: [
      { field: 'mfu', operator: '>', value: 0.35, label: 'MFU > 35%' },
    ],
    expectedChanges: [
      { field: 'precision', check: 'changed', label: 'Changed precision' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'The H800 has reduced high-bandwidth intra-node interconnect bandwidth compared to H100. What does that imply for the maximum useful `TP` degree? With 256 experts across 2048 GPUs, how should `EP` be sized?',
      'DeepSeek V3 has 61 layers — a prime number. What `PP` values divide evenly (or approximately)? `FP8` training halves `TP` communication volume — does that change the optimal `TP`?',
      'The published config used `TP=4`, `PP=8`, `EP=32`, `FP8` with `GBS=8192`, `MBS=2`. The DualPipe-V schedule is available in the pipeline schedule selector.',
    ],
    successExplanation:
      '[DeepSeek V3 (DeepSeek AI, 2024)](https://arxiv.org/abs/2412.19437) is one of the most efficient large-scale training runs ever ' +
      'published. Key design choices: `FP8` training halves the `TP` communication volume ' +
      '(1 byte per activation instead of 2), `EP=32` distributes the 256 experts so each ' +
      'GPU holds only 8 experts, and the DualPipe schedule overlaps forward and backward ' +
      'micro-batches to reduce pipeline bubbles.\n\nDevice-limited routing (`M=4`) caps ' +
      'all-to-all communication by ensuring tokens are only dispatched to at most 4 ' +
      'devices. The H800 interconnect limitation is partially offset by `FP8` halving ' +
      'the bytes on the wire.\n\n' +
      'The Chinchilla scaling law (Hoffmann et al., 2022) found that compute-optimal training uses ' +
      '`D ≈ 20 × N` tokens, where N is active parameters. For DeepSeek V3 with 37.6B active parameters, ' +
      'Chinchilla-optimal would be ~752B tokens. But V3 was trained on 14.8T tokens — roughly 20× ' +
      'overtraining. This is deliberate: a smaller overtrained model can match a larger compute-optimal ' +
      'one at inference time, but costs far less to serve per token. The 2.66M H800 GPU-hours spent on ' +
      'extra training pays for itself after serving billions of inference requests. You can explore this ' +
      'tradeoff using the Training Projection panel — try switching between Chinchilla Optimal and Heavy ' +
      'Overtrain in the sidebar to see how training time and cost change.',
  },

  // -----------------------------------------------------------------------
  // 8. Nemotron 340B
  // -----------------------------------------------------------------------
  {
    id: 'training-advanced-08',
    mode: 'training',
    difficulty: 'advanced',
    order: 8,
    title: 'Nemotron 340B',
    concept: 'Pipeline efficiency at 340 billion parameters',
    learningObjectives: [
      'Apply `TP`, `PP`, interleaved scheduling, and selective AC simultaneously to match a published config',
      'Understand `v=8` with `PP=12` achieves finest granularity (1 layer per virtual stage)',
      'Understand selective AC: keep MLP activations (expensive to recompute), discard attention activations (cheap with `FA`)',
      'Recognize the memory tradeoff: higher v means more in-flight microbatches, selective AC partially offsets this',
    ],
    briefing:
      'NVIDIA trained Nemotron-4 340B on 6144 H100 GPUs and published 41-42% `MFU`. ' +
      'The published config used `TP=8, PP=12` with interleaved `1F1B` scheduling and `v=8` — giving ' +
      'the finest possible pipeline granularity (1 layer per virtual stage). They also used ' +
      '{{selective-checkpointing|selective activation checkpointing}}: instead of recomputing everything in the backward pass, ' +
      'only attention activations are discarded (cheap to recompute with {{flash-attention|Flash Attention}}) while MLP ' +
      'activations are kept (expensive to recompute). Your challenge: configure 128 H100s to reproduce ' +
      'this result. You will need to set up the parallelism dimensions, pipeline schedule, and ' +
      'checkpointing strategy to push `MFU` above the target.',
    setup: {
      modelId: 'nemotron-4-340b',
      gpuId: 'h100-sxm',
      numGPUs: 128,
      strategyType: 'fsdp-tp-pp',
      mixedPrecision: 'bf16',
      flashAttention: true,
      activationCheckpointing: true,
      sequenceParallel: true,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'No OOM' },
      { field: 'mfu', operator: '>', value: 0.45, label: 'MFU > 45%' },
    ],
    expectedChanges: [
      { field: 'checkpointingGranularity', check: 'changed', label: 'Changed checkpointing granularity' },
      { field: 'activationCheckpointing', check: 'unchanged', label: 'Did not disable checkpointing' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'The published config: `TP=8, PP=12` with interleaved `1F1B` and `v=8`. Nemotron has 96 layers — `96/12/8 = 1` layer per virtual stage.',
      'Full activation checkpointing recomputes everything in the backward pass, including expensive MLP intermediates. Selective checkpointing keeps those and only discards attention activations — which Flash Attention makes cheap to recompute. Look for the checkpointing granularity selector.',
      'Higher `v` shrinks the pipeline bubble but increases activation memory from more in-flight microbatches. Selective checkpointing helps offset this memory cost.',
    ],
    successExplanation:
      'Nemotron-4 340B ([NVIDIA, 2024](https://arxiv.org/abs/2402.16819)) combined several techniques to achieve 41-42% `MFU` at scale. ' +
      '`PP=12` with `v=8` gives the finest interleaving granularity — each virtual stage is just 1 layer, ' +
      'reducing the pipeline bubble from ~58% (no interleaving) to ~10%.\n\n' +
      'Selective activation checkpointing was equally important. Full checkpointing recomputes all ' +
      'activations in the backward pass, but MLP intermediates dominate compute in large models. ' +
      'Selective checkpointing keeps MLP activations (expensive to recompute) and only discards ' +
      'attention activations (cheap with Flash Attention). This recovers significant throughput versus ' +
      'full checkpointing while still providing enough memory savings to support the high `v`.\n\n' +
      'The combination is powerful: interleaving needs more activation memory (more in-flight microbatches), ' +
      'and selective checkpointing uses more memory than full — but together they fit because the memory ' +
      'savings from selective are still substantial, and the throughput gains from both compound.',
  },

  // -----------------------------------------------------------------------
  // 9. GPU Architecture Comparison
  // -----------------------------------------------------------------------
  {
    id: 'training-advanced-09',
    mode: 'training',
    difficulty: 'advanced',
    order: 6,
    title: 'GPU Architecture Comparison',
    concept: 'Hardware Selection and MFU Impact',
    learningObjectives: [
      'Understand `MFU` denominator varies by GPU — higher peak makes the same workload report lower `MFU`',
      'Know the same model can report lower `MFU` on faster hardware but higher absolute throughput',
      'Know training cost = GPU-hours × $/GPU-hour, and higher `MFU` reduces GPU-hours for a fixed token budget',
    ],
    briefing:
      '`MFU` is defined as actual throughput divided by theoretical peak — but that ' +
      'peak varies wildly between GPU generations. You are starting on 16 {{a100|A100}} GPUs ' +
      '(2 nodes) with `FSDP+TP` and all standard optimizations enabled.\n\n' +
      'Run the simulation and observe the `MFU`. Now your challenge: switch to {{h100|H100 SXM}} GPUs ' +
      'and achieve over 40% `MFU`. The H100 SXM has roughly 3× the peak `BF16` compute of the A100 — ' +
      'this means the `MFU` denominator triples. You will need to tune the parallelism ' +
      'configuration to achieve the target on the faster hardware.',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'a100-80gb',
      numGPUs: 16,
      strategyType: 'fsdp-tp',
      mixedPrecision: 'bf16',
      flashAttention: true,
      activationCheckpointing: true,
      sequenceParallel: true,
      tpDegree: 8,
    },
    winningCriteria: [
      { field: 'gpuId', operator: '==', value: 'h100-sxm', label: 'GPU: H100 SXM' },
      { field: 'mfu', operator: '>', value: 0.40, label: 'MFU > 40%' },
    ],
    expectedChanges: [
      { field: 'gpuId', check: 'changed', label: 'Changed GPU type' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
    ],
    hints: [
      'Run the simulation first on A100 and note the `MFU`. Then switch the GPU to H100 SXM in the sidebar. The peak `TFLOPS` denominator is roughly 3× larger — `MFU` will change accordingly.',
      'On H100 SXM, fill each node with `TP` for fast `NVLink` communication. The 70B model partitions cleanly across 8 GPUs. Tune batch size and `TP` degree for the H100\'s characteristics.',
      'The key insight is that `MFU` compares against each GPU\'s own peak. A100 is easier to saturate (lower peak), so it reports higher `MFU` for the same workload. H100 delivers more absolute throughput but against a much higher denominator.',
    ],
    successExplanation:
      'GPU architecture determines both the numerator (actual `TFLOPS`) and denominator ' +
      '(peak `TFLOPS`) of `MFU`. H100 has roughly 3× the peak of A100 but rarely achieves 3× ' +
      'the actual throughput — memory bandwidth, interconnect, and kernel efficiency ' +
      'all create diminishing returns. This is why the same model may report 50% `MFU` ' +
      'on A100 and 40% `MFU` on H100, yet H100 still produces more tokens per second. ' +
      'When choosing hardware, look at absolute throughput and cost-per-token, not just ' +
      '`MFU` percentage.\n\n' +
      'Try switching back to A100 in the sidebar to see the contrast: higher `MFU` because the lower peak ' +
      'is easier to saturate, but significantly fewer tokens per second in absolute terms. ' +
      'In production, cost-per-token matters more than `MFU`. Training cost = GPU-hours × $/GPU-hour, and ' +
      'higher `MFU` means shorter training time, which means fewer GPU-hours for the same token budget.',
  },

  // -----------------------------------------------------------------------
  // 10. The Optimal Configuration
  // -----------------------------------------------------------------------
  {
    id: 'training-advanced-10',
    mode: 'training',
    difficulty: 'advanced',
    order: 9,
    title: 'The Optimal Configuration',
    concept: 'Full-Scale Training Optimization',
    learningObjectives: [
      'Synthesize all training techniques into a single coherent configuration for 405B on 512 GPUs',
      'Balance memory, communication overhead, and pipeline efficiency simultaneously',
      'Understand selective AC as the right balance for very large models (less overhead than full)',
      'Know Meta published config (`TP=8`, `PP=16`, `DP=128` on 16K GPUs) at ~40% `MFU` as reference',
    ],
    briefing:
      'This is the final challenge. You have 512 H100 GPUs — 64 nodes of 8 GPUs ' +
      'each — and the task is to train LLaMA 3.1 405B, one of the largest dense ' +
      'models ever published. Meta used 16,384 GPUs; you have 512. Every decision ' +
      'matters: `TP` degree, `PP` degree, pipeline schedule, virtual stages, sequence ' +
      'parallelism, activation checkpointing granularity, micro-batch size, {{gbs|global ' +
      'batch size}}, and mixed precision. You need to balance memory (405B parameters ' +
      'plus activations for 126 layers), communication (`TP` within nodes, `PP` across ' +
      'nodes, `FSDP` gradients), and pipeline efficiency (bubble fraction). This is ' +
      'where everything you have learned comes together. Achieve over 35% `MFU`.',
    setup: {
      modelId: 'llama3-405b',
      gpuId: 'h100-sxm',
      numGPUs: 512,
      strategyType: 'fsdp-tp-pp',
      mixedPrecision: 'bf16',
      flashAttention: true,
      activationCheckpointing: true,
      sequenceParallel: true,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'No OOM' },
      { field: 'mfu', operator: '>', value: 0.35, label: 'MFU > 35%' },
    ],
    expectedChanges: [
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'Start with `TP=8` (one node), then choose PP. 405B has 126 layers — pick a PP that divides the layers evenly. Then calculate the resulting DP from `TP × PP × DP = totalGPUs`.',
      'Interleaved `1F1B` with virtual stages is critical for pipeline efficiency. Experiment with `v` values — higher `v` shrinks the bubble but uses more activation memory.',
      'Enable sequence parallelism, selective activation checkpointing, and Flash Attention. For 405B, every GB of memory savings counts.',
    ],
    successExplanation:
      'Training 405B on 512 GPUs is a constrained optimization problem. With `TP=8` ' +
      '(filling one node for `NVLink`), the remaining 64 "TP groups" must be split ' +
      'between `PP` and `DP`. More `PP` means less memory pressure per device but larger ' +
      'pipeline bubbles and less `DP` throughput scaling. More `DP` means more gradient ' +
      'communication but better utilization.\n\nThe sweet spot typically involves `PP=8-16` ' +
      'with interleaved scheduling, leaving `DP=4-8` for throughput. Meta\'s published ' +
      'config (`TP=8`, `PP=16`, `DP=128` on 16K GPUs) achieved ~40% `MFU` — at smaller ' +
      'scale, the proportions shift but the principles are identical.\n\n' +
      'Selective activation checkpointing (discarding only attention activations) preserves more ' +
      'activations than full checkpointing, reducing recompute overhead while still saving significant ' +
      'memory. For 405B, this is often the right balance.\n\n' +
      'The inference track covers how to deploy this 405B model for serving — the parallelism choices change when decode bandwidth replaces training compute as the bottleneck.',
  },
];
