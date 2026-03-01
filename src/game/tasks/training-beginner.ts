import type { GameTask } from '../types.ts';
// Glossary annotations: {{termId}} or {{termId|display text}} for tooltip terms.
// [text](url) for paper links in successExplanation.
// Annotate FIRST OCCURRENCE per difficulty tier only. Check earlier tasks before adding.

export const TRAINING_BEGINNER_TASKS: GameTask[] = [
  // ── Task 1: Your First Training Run ──────────────────────────────────
  {
    id: 'training-beginner-01',
    mode: 'training',
    difficulty: 'beginner',
    order: 0,
    title: 'Your First Training Run',
    briefing:
      `Welcome to the world of distributed training! Before we tackle massive models on hundreds of GPUs, let's start with the basics.\n\n` +
      `You have a single NVIDIA {{a100|A100}} GPU with 80 GB of memory, and a small model: GPT-3 125M (125 million parameters). ` +
      `This is a toy-sized language model by modern standards, but it's perfect for getting started.\n\n` +
      'Your goal: get the model to train at over 50k tokens per second. The default configuration uses {{fp32|FP32}} precision, ' +
      'which runs on standard {{cuda-cores|CUDA cores}}. But the A100 has specialized {{tensor-cores|Tensor Cores}} designed for ' +
      'reduced-precision arithmetic — dramatically faster than CUDA cores. Find the right setting.',
    concept: 'Unlocking Tensor Core throughput with mixed precision',
    learningObjectives: [
      'Understand that FP32 runs on slow CUDA cores while BF16 activates Tensor Cores for 16x throughput',
      'Know that MFU measures useful compute as a fraction of GPU peak TFLOPS',
      'Recognize that small models underutilize large GPUs (low MFU even at high throughput)',
    ],
    setup: {
      modelId: 'gpt3-125m',
      gpuId: 'a100-80gb',
      numGPUs: 1,
      strategyType: 'ddp',
    },
    winningCriteria: [
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Training runs successfully',
      },
      {
        field: 'tokensPerSecond',
        operator: '>',
        value: 50000,
        label: 'Throughput > 50k tok/s',
      },
    ],
    expectedChanges: [
      { field: 'precision', check: 'changed', label: 'Changed precision from FP32' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
    ],
    hints: [
      'Run the simulation first at `FP32` and observe the throughput — it will be far below 50k tok/s. The GPU is running on slow CUDA cores.',
      'The A100 has Tensor Cores that accelerate reduced-precision matrix multiplications. Look for the Precision setting in the sidebar.',
      'Switch precision from `FP32` to {{bf16|BF16}}. Tensor Cores process `BF16` at 312 {{tflops|TFLOPS}} vs 19.5 `TFLOPS` for `FP32` — a 16× increase in peak throughput.',
    ],
    successExplanation:
      `Switching to reduced precision (for example, \`BF16\` or \`FP16\`) unlocked the A100's Tensor Cores, jumping from ~12k to ~154k tok/s. ` +
      `A 125M parameter model in \`FP32\` uses about 2.25 GB total (18 bytes/param for weights, gradients, and optimizer states), ` +
      `leaving plenty of headroom on an 80 GB GPU — the issue was never memory, but compute throughput.\n\n` +
      `The {{mfu|MFU}} you see tells you what fraction ` +
      `of the GPU's theoretical peak compute is being used for useful work. For a small model on a single GPU, \`MFU\` ` +
      `is typically modest — small models don't generate enough arithmetic to keep the GPU's thousands of cores busy. ` +
      `As models grow larger, we'll see \`MFU\` improve, but new challenges will emerge.`,
  },

  // ── Task 2: Activation Memory ───────────────────────────────────────
  {
    id: 'training-beginner-03',
    mode: 'training',
    difficulty: 'beginner',
    order: 1,
    title: 'Activation Memory',
    briefing:
      'GPT-3 1.3B fits comfortably on a single A100-80GB in `BF16` — the model state is only about 23 GB. ' +
      'But look at the current setup: {{gbs|Global Batch Size}} (GBS) is 64, and the {{mbs|Micro-Batch Size}} (MBS) is also 64. ' +
      'That means the GPU tries to run all 64 sequences through the forward pass at once, storing intermediate ' +
      'tensors ({{activation-memory|activations}}) for every sequence simultaneously.\n\n' +
      'Run the simulation — you\'ll hit {{oom|OOM}}. The model weights are fine, but 64 sequences worth of activations ' +
      'is far more than the remaining memory can hold.\n\n' +
      'The fix: {{gradient-accumulation|Gradient Accumulation}} (GA). Instead of processing all 64 sequences in ' +
      'one giant forward pass, GA breaks the batch into smaller chunks. The GPU runs multiple ' +
      'forward/backward passes, accumulating gradients, then updates the optimizer once. The formula is ' +
      '`GA = GBS / MBS`. The training result is mathematically identical — same gradients, same ' +
      'convergence — but peak memory drops dramatically.\n\n' +
      'Your goal: keep GBS at 64 and make the training fit in memory.',
    concept: 'Forward pass memory growth',
    learningObjectives: [
      'Understand that activation memory scales linearly with micro-batch size (each sequence stores layer intermediates for backprop)',
      'Know the gradient accumulation formula: GA = GBS / MBS',
      'Recognize that GA produces identical gradients to processing the full batch at once',
    ],
    setup: {
      modelId: 'gpt3-1.3b',
      gpuId: 'a100-80gb',
      numGPUs: 1,
      strategyType: 'ddp',
      mixedPrecision: 'bf16',
      flashAttention: true,
      globalBatchSize: 64,
      microBatchSize: 64,
    },
    winningCriteria: [
      {
        field: 'memoryUtilization',
        operator: '<',
        value: 1.0,
        label: 'No OOM',
      },
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Training runs successfully',
      },
    ],
    expectedChanges: [
      { field: 'microBatchSize', check: 'decreased', label: 'Reduced micro-batch size' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'globalBatchSize', check: 'unchanged', label: 'Kept GBS constant' },
    ],
    hints: [
      'Look at the memory breakdown — model weights are a small fraction of total memory. Where is the rest going? Think about what happens during a forward pass when MBS is large.',
      'Activation memory scales with how many sequences the GPU processes simultaneously. Which setting controls that? Changing it won\'t affect the total batch — another mechanism compensates automatically.',
      'Reduce the Micro-Batch Size. Gradient accumulation steps increase as you lower `MBS` (`GA = GBS / MBS`), preserving the same effective batch size with less memory per forward pass.',
    ],
    successExplanation:
      `Gradient accumulation is how real training runs achieve massive batch sizes without running out of memory. ` +
      `The GPU processes \`MBS\` sequences at a time, computes gradients, and accumulates them over \`GA\` steps before ` +
      `a single optimizer update. The result is mathematically identical to processing the full GBS at once.\n\n` +
      `The key formula: \`GA = GBS / MBS\`. By reducing MBS while keeping GBS fixed, GA increases automatically. ` +
      `For example, with \`GBS=64\` and \`MBS=4\`, \`GA=16\` — sixteen forward/backward passes, ` +
      `each holding activations for just 4 sequences instead of 64.\n\n` +
      `In practice, MBS is tuned to the largest value that fits in GPU memory, while GBS is set based on training ` +
      `convergence requirements. GA bridges them automatically. When you scale to multiple GPUs later, the work ` +
      `is split across devices and the formula expands to \`GA = GBS / (MBS × DP)\`.`,
  },

  // ── Task 3: Understanding GPU Memory ─────────────────────────────────
  {
    id: 'training-beginner-02',
    mode: 'training',
    difficulty: 'beginner',
    order: 2,
    title: 'Understanding GPU Memory',
    briefing:
      `Gemma 3 4B has about 4 billion parameters. ` +
      `You have a single A100-80GB GPU.\n\n` +
      `Try running the simulation. You'll hit an Out of Memory (\`OOM\`) error. Training a model requires far more ` +
      `memory than just the weights: you need space for {{gradients}}, {{optimizer-states|optimizer states}} (like Adam's momentum and variance), ` +
      `and {{activation-memory|activation memory}} from the {{forward-pass|forward pass}}.\n\n` +
      `In \`FP32\` (32-bit floating point), the model state alone takes about 16 bytes per parameter — ` +
      `64 GB for 4B parameters. Add activation memory and you exceed the 80 GB GPU.\n\n` +
      `Your task: change one setting to make this model fit in memory.`,
    concept: 'GPU memory breakdown and precision formats',
    learningObjectives: [
      'Identify training memory components: parameters, gradients, optimizer states (16 bytes/param in FP32), and activations',
      'Recognize OOM as the signal that total memory exceeds GPU capacity',
      'Use BF16 mixed precision to reduce memory while unlocking Tensor Core compute',
    ],
    setup: {
      modelId: 'gemma3-4b',
      gpuId: 'a100-80gb',
      numGPUs: 1,
      strategyType: 'ddp',
      flashAttention: true,
    },
    winningCriteria: [
      {
        field: 'memoryUtilization',
        operator: '<',
        value: 1.0,
        label: 'No OOM',
      },
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Training runs successfully',
      },
    ],
    expectedChanges: [
      { field: 'precision', check: 'changed', label: 'Changed precision from FP32' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
    ],
    hints: [
      'Look at how much memory each component uses. The optimizer states alone in `FP32` take 8 bytes per parameter. Can you use a smaller number format?',
      'Modern GPUs like the A100 have hardware support for `BF16` (bfloat16), which uses 2 bytes instead of 4 per value. Check the Precision setting.',
      'Switch the training precision from `FP32` to `BF16`. This halves the memory for parameters and gradients, and the optimizer uses mixed precision as well.',
    ],
    successExplanation:
      `\`BF16\` {{mixed-precision|mixed precision}} stores parameters in 2 bytes instead of 4, and activations are ` +
      `also stored in \`BF16\` during the forward pass, halving activation memory. While the ` +
      `optimizer states actually use 12 bytes per parameter in \`BF16\` mode (\`FP32\` master weights + ` +
      `momentum + variance), the activation savings are enough to bring the total under 80 GB.\n\n` +
      `Combined with the 16x compute speedup from Tensor Cores, \`BF16\` is strictly better for ` +
      `training on modern GPUs. \`BF16\` has the same exponent range as \`FP32\` (unlike {{fp16|FP16}}), ` +
      `making it robust for training without loss scaling. Nearly all modern LLM training uses \`BF16\` or \`FP8\`.`,
  },

  // ── Task 4: Activation Checkpointing ─────────────────────────────────
  {
    id: 'training-beginner-04',
    mode: 'training',
    difficulty: 'beginner',
    order: 3,
    title: 'Activation Checkpointing',
    briefing:
      `Gemma 3 4B in \`BF16\` fits on the A100, but just barely — memory utilization is above 95%. ` +
      `That leaves almost no headroom for larger batch sizes or longer sequences.\n\n` +
      `When a model runs its forward pass, it saves intermediate results (activations) at each layer — these are ` +
      `needed later during the {{backward-pass|backward pass}} to compute gradients. For a model with many layers, activation memory ` +
      `can be substantial.\n\n` +
      `Activation checkpointing is a classic memory-compute tradeoff: instead of storing all activations, you discard ` +
      `most of them and recompute them during the backward pass. This adds roughly 25-40% more total compute (the backward pass takes about 2.5× the forward time instead of 2×) ` +
      `but dramatically reduces peak memory.\n\n` +
      `Your goal: bring memory utilization below 90%.`,
    concept: 'The compute-memory tradeoff',
    learningObjectives: [
      'Understand that activation memory scales linearly with layer count and is needed for backprop',
      'Know AC discards activations in forward pass and recomputes them in backward pass',
      'Understand the compute cost: backward multiplier increases from 2x to ~2.5-2.85x',
      'Distinguish MFU (useful work only) from HFU (includes recompute overhead)',
    ],
    setup: {
      modelId: 'gemma3-4b',
      gpuId: 'a100-80gb',
      numGPUs: 1,
      strategyType: 'ddp',
      mixedPrecision: 'bf16',
      flashAttention: true,
    },
    winningCriteria: [
      {
        field: 'memoryUtilization',
        operator: '<',
        value: 0.90,
        label: 'Memory utilization < 90%',
      },
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Training runs successfully',
      },
    ],
    expectedChanges: [
      { field: 'activationCheckpointing', check: 'enabled', label: 'Enabled activation checkpointing' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
    ],
    hints: [
      'Activation memory from the forward pass is the main contributor beyond model state. There is a setting that trades compute for memory by discarding and recomputing activations.',
      'Enable full activation checkpointing in the sidebar. It discards activations during the forward pass and recomputes them in the backward pass, cutting activation memory significantly.',
    ],
    successExplanation:
      `Activation checkpointing is one of the most important memory optimization techniques in deep learning. ` +
      `Without it, activation memory scales linearly with the number of layers — for a 40-layer 13B model, ` +
      `that's a lot of saved tensors. With full checkpointing, you recompute activations layer by layer during ` +
      `backprop, reducing peak activation memory dramatically.\n\n` +
      `The cost is extra compute: the backward pass takes roughly 2.5-2.85x the forward pass time instead of 2x. ` +
      `This is reflected in the difference between \`MFU\` (useful work only) and {{hfu|HFU}} (includes recompute). ` +
      `Almost every large-scale training run uses some form of checkpointing — the memory savings are essential ` +
      `for fitting large models and large batch sizes.\n\n` +
      `There is also a lighter variant called {{selective-checkpointing|selective checkpointing}}. Instead of discarding all activations, ` +
      `selective checkpointing keeps {{mlp|MLP}} activations (expensive to recompute) and only discards attention activations ` +
      `(cheap to recompute with Flash Attention). It saves less memory but adds less compute overhead — you will ` +
      `encounter this in the advanced track.`,
  },

  // ── Task 5: Flash Attention ──────────────────────────────────────────
  {
    id: 'training-beginner-05',
    mode: 'training',
    difficulty: 'beginner',
    order: 4,
    title: 'Flash Attention',
    briefing:
      `{{self-attention|Self-attention}} is the core operation in Transformers, but standard attention has a significant cost: ` +
      `materializes an \`N × N\` {{attention-matrix|attention matrix}} (where N is the sequence length), consuming memory that's quadratic ` +
      `in sequence length.\n\n` +
      `You have Gemma 3 4B on a single A100-80GB with \`BF16\` and activation checkpointing enabled. ` +
      `Look at the memory utilization — even with checkpointing, the attention score matrices are ` +
      `taking significant memory.\n\n` +
      `{{flash-attention|Flash Attention}} is an IO-aware attention algorithm that never materializes the full \`N × N\` matrix. Instead, ` +
      `it computes attention in tiles, dramatically reducing memory usage (from quadratic to linear in sequence length) ` +
      `and improving speed by reducing memory bandwidth bottlenecks.\n\n` +
      `Your goal: get memory utilization below 85%.`,
    concept: 'Quadratic attention and tiled computation',
    learningObjectives: [
      'Understand standard attention materializes `O(N²)` attention score matrix in HBM',
      'Know Flash Attention tiles computation in SRAM, achieving `O(N)` memory',
      'Recognize FA also improves speed 2-4x via better memory access patterns',
    ],
    setup: {
      modelId: 'gemma3-4b',
      gpuId: 'a100-80gb',
      numGPUs: 1,
      strategyType: 'ddp',
      mixedPrecision: 'bf16',
      activationCheckpointing: true,
    },
    winningCriteria: [
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Training runs successfully',
      },
      {
        field: 'memoryUtilization',
        operator: '<',
        value: 0.85,
        label: 'Memory utilization < 85%',
      },
    ],
    expectedChanges: [
      { field: 'flashAttention', check: 'enabled', label: 'Enabled Flash Attention' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
    ],
    hints: [
      'Flash Attention eliminates the quadratic attention score memory. The savings grow with sequence length — at short sequences they are modest, but at long sequences they are essential.',
      'Enable Flash Attention in the sidebar. It reduces attention memory from `O(N²)` to `O(N)`, freeing up significant headroom.',
    ],
    successExplanation:
      `[Flash Attention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) is now the default in virtually every training framework. Instead of ` +
      `computing the full \`N × N\` attention score matrix in {{hbm|HBM}} (GPU main memory), it tiles the computation to fit ` +
      `in {{sram|SRAM}} (the fast on-chip cache), never materializing the full matrix.\n\n` +
      `Flash Attention eliminates \`O(N²)\` \`HBM\` traffic for attention scores by tiling the computation in \`SRAM\`. ` +
      `The throughput benefit depends on sequence length — at shorter sequences the attention \`HBM\` traffic is a small ` +
      `fraction of total time, but at longer sequences (8K+) it grows quadratically and becomes the dominant bottleneck ` +
      `without Flash Attention. The memory savings (\`O(N²)\` → \`O(N)\` activation memory) can be the difference between fitting and \`OOM\` ` +
      `at long contexts. There is no reason to leave FA off on supported hardware.`,
  },

  // ── Task 6: Scaling to Multiple GPUs ─────────────────────────────────
  {
    id: 'training-beginner-06',
    mode: 'training',
    difficulty: 'beginner',
    order: 5,
    title: 'Scaling to Multiple GPUs',
    briefing:
      `So far, we've trained on a single GPU. But real training runs use many GPUs working together. ` +
      `Let's scale out to a full 8-GPU node.\n\n` +
      `Data Distributed Parallel ({{ddp|DDP}}) is the simplest form of distributed training. Each GPU holds a complete ` +
      `copy of the model and processes a different mini-batch of data. After each backward pass, gradients are ` +
      `synchronized across GPUs using an {{allreduce|AllReduce}} collective operation.\n\n` +
      `With 8 GPUs and \`DDP\`, you process 8x as much data per step. But synchronization is not free — ` +
      `it adds communication overhead. The question is: how much throughput do you actually gain?\n\n` +
      `Your goal: achieve at least 100k tokens per second. You currently have a single GPU.`,
    concept: 'Multi-GPU gradient synchronization',
    learningObjectives: [
      'Understand DDP: each GPU holds full model copy, processes different data, syncs gradients via AllReduce',
      'Know fast intra-node interconnects (NVLink, Infinity Fabric) enable near-linear DDP scaling within a node',
      'Observe near-linear throughput scaling with DDP on fast intra-node interconnects',
    ],
    setup: {
      modelId: 'gpt3-1.3b',
      gpuId: 'a100-80gb',
      numGPUs: 1,
      gpusPerNode: 8,
      strategyType: 'ddp',
      mixedPrecision: 'bf16',
      flashAttention: true,
    },
    winningCriteria: [
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Training runs successfully',
      },
      {
        field: 'tokensPerSecond',
        operator: '>',
        value: 100000,
        label: 'Throughput > 100k tok/s',
      },
    ],
    expectedChanges: [
      { field: 'numGPUs', check: 'increased', label: 'Increased GPU count' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'A single GPU cannot reach 100k tok/s. Scale up the number of GPUs to add more data-parallel workers.',
      '{{nvlink|NVLink}} provides high bandwidth between GPUs in the same node (600 GB/s bidirectional on A100), making `AllReduce` gradient sync very fast. `DDP` throughput scales nearly linearly within a node.',
    ],
    successExplanation:
      `\`DDP\` is the foundation of distributed training. Within a single 8-GPU node connected by \`NVLink\`, ` +
      `the \`AllReduce\` gradient synchronization overlaps with the backward pass computation, hiding most of ` +
      `the communication cost.\n\n` +
      `The key insight: \`DDP\` scales data throughput (tokens/sec) nearly linearly when communication is fast. ` +
      `Each GPU computes on its own data, then a ring-\`AllReduce\` synchronizes gradients — the whole gradient ` +
      `tensor is reduced in \`2*(N-1)/N * size\` bytes, pipelined across the \`NVLink\` ring. ` +
      `For small models on fast interconnects, \`DDP\` approaches perfect linear scaling. ` +
      `For larger models or slower interconnects, the overhead grows — and that's where more advanced ` +
      `strategies like \`FSDP\` come in.`,
  },

  // ── Task 7: When Models Don't Fit ──────────────────────────────────
  {
    id: 'training-beginner-07',
    mode: 'training',
    difficulty: 'beginner',
    order: 6,
    title: 'When Models Don\'t Fit',
    briefing:
      'In the previous task, `DDP` scaled beautifully within a single node — each GPU held a full copy ' +
      'of the model and processed different data. That works when the model fits on one GPU.\n\n' +
      'Qwen 3 14B has about 14 billion parameters. With `BF16` training and `AdamW`, each GPU needs to ' +
      'store parameters (2 bytes), gradients (4 bytes), and optimizer states (12 bytes) — that\'s ' +
      '18 bytes per parameter. For 14B parameters: `14B × 18 = 252 GB`. An A100 has 80 GB. `DDP` ' +
      '`DDP` replicates all of this on every GPU, so adding more GPUs doesn\'t help with memory.\n\n' +
      'Fully Sharded Data Parallel ({{fsdp|FSDP}}) fixes this. Instead of replicating everything, `FSDP` ' +
      '{{sharding|shards}} the parameters, gradients, and optimizer states across all GPUs. Each GPU stores ' +
      'only `1/N` of the model state. Before each layer\'s computation, an {{allgather|AllGather}} collective ' +
      'assembles the full parameters temporarily; after the backward pass, a {{reducescatter|ReduceScatter}} ' +
      'distributes gradients back to their shards.\n\n' +
      'Your task: make this model fit in memory and train successfully.',
    concept: 'Sharding model state across GPUs',
    learningObjectives: [
      'Understand that DDP replicates full model state (18 bytes/param for BF16 + AdamW) on every GPU',
      'Know that FSDP shards parameters, gradients, and optimizer states across GPUs (equivalent to ZeRO-3)',
      'Understand the AllGather (before forward) and ReduceScatter (after backward) communication pattern',
      'Compare DDP memory (18 bytes/param replicated) vs FSDP memory (18/N bytes/param sharded)',
    ],
    setup: {
      modelId: 'qwen3-14b',
      gpuId: 'a100-80gb',
      numGPUs: 8,
      gpusPerNode: 8,
      strategyType: 'ddp',
      mixedPrecision: 'bf16',
      flashAttention: true,
      activationCheckpointing: true,
    },
    winningCriteria: [
      {
        field: 'memoryUtilization',
        operator: '<',
        value: 1.0,
        label: 'No OOM',
      },
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Training runs successfully',
      },
    ],
    expectedChanges: [
      { field: 'strategyType', check: 'changed', label: 'Changed strategy from DDP' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'Run `DDP` first to confirm it OOMs. Then look at the strategy selector — there is a strategy that shards all model state across GPUs instead of replicating it.',
      'Switch the strategy from `DDP` to `FSDP`. `FSDP` (equivalent to DeepSpeed ZeRO Stage 3) shards parameters, gradients, and optimizer states across all GPUs, reducing per-GPU memory by the sharding factor.',
    ],
    successExplanation:
      `\`FSDP\` is one of the most important innovations in distributed training. It implements the same idea as ` +
      `DeepSpeed [ZeRO (Rajbhandari et al., 2019)](https://arxiv.org/abs/1910.02054) Stage 3: shard everything across the data-parallel group.\n\n` +
      `With \`DDP\`, a 14B model needs ~18 bytes/param per GPU (params + grads + optimizer). That's ~252 GB per GPU — ` +
      `impossible on 80 GB. With \`FSDP\` across 8 GPUs, each GPU stores only \`1/8\` of the model state, bringing ` +
      `per-GPU usage to ~31 GB for model state.\n\n` +
      `The communication cost is higher than \`DDP\`: \`FSDP\` needs an \`AllGather\` before each forward layer and a ` +
      `\`ReduceScatter\` after each backward layer. But with \`NVLink\` within a node, this communication is fast and ` +
      `can overlap with compute. \`FSDP\` is the default strategy for training models from 7B to 70B on single nodes.`,
  },

  // ── Task 8: Micro-Batch Size & Memory ─────────────────────────────
  {
    id: 'training-beginner-08',
    mode: 'training',
    difficulty: 'beginner',
    order: 7,
    title: 'Micro-Batch Size & Memory',
    briefing:
      'In the previous task, you discovered `FSDP` — a strategy that shards model state across GPUs, ' +
      'fitting models far too large for `DDP`. But `FSDP` only fixes model state memory. ' +
      'Activation memory — the intermediate values stored during the forward pass for backpropagation — ' +
      'still lives entirely on each GPU.\n\n' +
      'LLaMA 3.1 8B is on 8 GPUs with `FSDP` and no {{activation-checkpointing|activation checkpointing}} — the GPU stores all ' +
      'intermediate activations for the {{backward-pass|backward pass}}. `MBS` is set to 4: each GPU processes 4 sequences ' +
      'in a single forward pass, storing activations for all 4 simultaneously.\n\n' +
      'The problem: 4 sequences of activations at seqLen=4096 far exceeds GPU memory alongside the sharded model state. ' +
      '\n\nYour goal: fit in memory and achieve `MFU` above 50%.',
    concept: 'Micro-batch size as a memory control knob',
    learningObjectives: [
      'Understand that activation memory scales linearly with MBS — each sequence stores layer intermediates for backprop',
      'Know GA = GBS / (MBS × DP): reducing MBS increases GA, keeping GBS and gradient quality constant',
      'Recognize that FSDP shards model state but not activations — MBS is the lever for activation memory',
      'Know that GPT-3 used GBS=3.2M tokens with small MBS and high GA + DP',
    ],
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 'a100-80gb',
      numGPUs: 8,
      gpusPerNode: 8,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
      flashAttention: true,
      activationCheckpointing: false,
      globalBatchSize: 64,
      microBatchSize: 4,
    },
    winningCriteria: [
      {
        field: 'memoryUtilization',
        operator: '<',
        value: 1.0,
        label: 'No OOM',
      },
      {
        field: 'mfu',
        operator: '>',
        value: 0.50,
        label: 'MFU > 50%',
      },
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Training runs successfully',
      },
    ],
    expectedChanges: [
      { field: 'microBatchSize', check: 'decreased', label: 'Reduced micro-batch size' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'Activation memory scales linearly with MBS — the GPU stores intermediate activations for all sequences in the micro-batch simultaneously. The current MBS is too large to fit alongside `FSDP` model state.',
      'Reduce `MBS` and let `GA` adjust. Gradient accumulation scales inversely with `MBS` — reducing `MBS` lets `GA` rise automatically to preserve the same effective batch size.',
      'MBS barely affects `MFU` at this scale — it is purely a memory knob. Use the smallest MBS that fits, and GA fills the gap automatically.',
    ],
    successExplanation:
      `\`MBS\` controls per-GPU memory: activation memory scales linearly with micro-batch size because each ` +
      `sequence in the micro-batch stores its own intermediate tensors for the backward pass. Without ` +
      `activation checkpointing, all layer activations are retained — at \`MBS=4\`, that's 4× the activation ` +
      `memory of \`MBS=1\`.\n\n` +
      `The formula \`GBS = MBS × GA × DP\` means reducing MBS automatically increases GA, preserving the ` +
      `effective batch size. For example, \`MBS=2\` with \`GBS=64\` and \`DP=8\` gives \`GA=4\`, fitting comfortably ` +
      `in memory while maintaining high \`MFU\`.\n\n` +
      `In practice, tune \`MBS\` to the largest value that fits in GPU memory — more samples per forward pass ` +
      `means better hardware utilization. Then GA fills the gap to reach the desired GBS. Activation ` +
      `checkpointing (from task 4) is the other lever for memory, but it adds recomputation overhead ` +
      `that would drop \`MFU\` below the target here.`,
  },

  // ── Task 9: Going Multi-Node ────────────────────────────────────────
  {
    id: 'training-beginner-09',
    mode: 'training',
    difficulty: 'beginner',
    order: 8,
    title: 'Going Multi-Node',
    briefing:
      'So far you have trained within a single node where GPUs communicate via a fast ' +
      'intra-node interconnect like {{nvlink|NVLink}} — providing hundreds of GB/s of bandwidth. ' +
      'But a single 8-GPU node has limited total compute. To train faster, you need more GPUs.\n\n' +
      'When you add a second node, GPUs across nodes communicate via a network fabric like ' +
      '{{infiniband|InfiniBand}}. Inter-node links are typically several times slower ' +
      'than intra-node interconnects. ' +
      '`FSDP`\'s `AllGather` and `ReduceScatter` collectives must cross this slower link ' +
      'for the inter-node portion of the sharded parameters.\n\n' +
      'Your task: scale beyond one node to exceed 40,000 tokens per second. ' +
      'Remember to adjust your Global Batch Size — ' +
      'each GPU processes at least one micro-batch per step, so `GBS` should be at least `MBS` × DP.\n\n' +
      'Your goal: throughput above 40,000 tokens per second.',
    concept: 'Multi-node training and inter-node communication',
    learningObjectives: [
      'Understand that intra-node interconnects (e.g. NVLink) are much faster than inter-node fabrics (e.g. InfiniBand)',
      'Know that FSDP AllGather/ReduceScatter must cross the inter-node link when spanning nodes',
      'Recognize that GBS must be at least MBS × DP for all GPUs to contribute useful work',
      'Observe that throughput scales with more GPUs, but efficiency per GPU decreases slightly at multi-node scale',
    ],
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 'a100-80gb',
      numGPUs: 8,
      gpusPerNode: 8,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
      flashAttention: true,
      activationCheckpointing: true,
    },
    winningCriteria: [
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Training runs successfully',
      },
      {
        field: 'tokensPerSecond',
        operator: '>',
        value: 40000,
        label: 'Throughput > 40k tok/s',
      },
    ],
    expectedChanges: [
      { field: 'numGPUs', check: 'increased', label: 'Increased GPU count' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'Increase the number of GPUs beyond 8 to create a multi-node cluster. Each node has 8 GPUs connected by `NVLink` internally, and nodes connect via `InfiniBand`.',
      'Make sure Global Batch Size keeps pace with the number of GPUs. With `MBS=1` and 16 GPUs, `GBS` should be at least 16. Otherwise, some GPU work does not contribute to the effective batch.',
    ],
    successExplanation:
      'Scaling beyond a single node increased throughput substantially, though ' +
      '`MFU` dropped slightly. Here\'s why:\n\n' +
      'Within each node, GPUs communicate via a fast intra-node interconnect (e.g. `NVLink` at ~600 GB/s). ' +
      'Between nodes, they use a network fabric like `InfiniBand` (~200 GB/s for HDR). The inter-node ' +
      'link is typically several times slower. `FSDP`\'s `AllGather` and `ReduceScatter` collectives ' +
      'must traverse this slower link for the inter-node portion.\n\n' +
      'However, `FSDP`\'s per-layer pipelining is remarkably effective at hiding communication: ' +
      '`AllGather`(layer N) overlaps with compute(layer N-1). For compute-bound models like LLaMA 8B, ' +
      'per-layer compute far exceeds per-layer communication, so the multi-node overhead is modest — ' +
      'typically 2-5% `MFU` loss.\n\n' +
      'The key rule for scaling: `GBS ≥ MBS × DP`. Each GPU must process at least one micro-batch ' +
      'per step. If GBS is smaller than the number of GPUs, the effective batch is rounded up, ' +
      'and some compute does not contribute to gradient quality — wasting efficiency.\n\n' +
      'In the intermediate track, you\'ll learn tensor parallelism and pipeline parallelism — ' +
      'techniques that split the model itself across GPUs, enabling training of 70B+ parameter models ' +
      'that require more memory than even `FSDP` can provide at small scale.',
  },

  // ── Task 10: Your First Efficiency Win ───────────────────────────────
  {
    id: 'training-beginner-10',
    mode: 'training',
    difficulty: 'beginner',
    order: 9,
    title: 'Your First Efficiency Win',
    briefing:
      `You've learned the fundamentals: precision, activation checkpointing, Flash Attention, batch sizing, ` +
      `\`DDP\`, and \`FSDP\`. Now put it all together.\n\n` +
      `You have 8 NVIDIA {{h100|H100}} SXM GPUs — the current standard for AI training. Each \`H100\` delivers 989 \`BF16\` \`TFLOPS\` ` +
      `(over 3x the A100) and the \`NVLink\` 4.0 interconnect provides 900 GB/s of bandwidth.\n\n` +
      `Your mission: train LLaMA 3.1 8B with \`FSDP\` on these 8 \`H100\`s and achieve \`MFU\` above 50%. Published benchmarks ` +
      `for 8B models on \`H100\` clusters report \`MFU\` in the 50-57% range.\n\n` +
      `Here's a twist: \`FSDP\` shards model state so efficiently across 8 GPUs that you may have more memory ` +
      `headroom than you think. Every optimization has a cost — even activation checkpointing adds recomputation ` +
      `overhead that shows up as lower \`MFU\` (the recompute is not "useful work"). For maximum efficiency, use only ` +
      `every optimization is helping or hurting in this configuration.`,
    concept: 'Combining techniques for production-grade efficiency',
    learningObjectives: [
      'Recognize that not every optimization is always beneficial — AC hurts when memory is unconstrained',
      'Combine BF16, FA, FSDP, and batch sizing for production-grade 50%+ MFU',
      'Understand that MFU directly determines training time and cost for a fixed token budget',
    ],
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 'h100-sxm',
      numGPUs: 8,
      gpusPerNode: 8,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
      flashAttention: true,
      activationCheckpointing: true,
    },
    winningCriteria: [
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Training runs successfully',
      },
      {
        field: 'mfu',
        operator: '>',
        value: 0.50,
        label: 'MFU > 50%',
      },
    ],
    expectedChanges: [
      { field: 'activationCheckpointing', check: 'disabled', label: 'Disabled activation checkpointing' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      '`MFU` is held back by the backward pass recomputing activations (extra compute that does not count as useful work in `MFU`).',
      '`FSDP` shards the model state across all GPUs — per-GPU model state is tiny compared to GPU capacity. You have plenty of memory headroom. Do you really need activation checkpointing?',
      'Disable activation checkpointing — the 8B model fits comfortably with `FSDP` across 8 GPUs. Without the recomputation overhead, `MFU` improves significantly. Try increasing MBS for better utilization.',
    ],
    successExplanation:
      `Congratulations — you've completed the beginner track! You achieved over 50% \`MFU\` by recognizing ` +
      `that not every optimization is always needed.\n\n` +
      `Activation checkpointing trades compute for memory. With \`FSDP\` sharding 8B across 8 GPUs, per-GPU ` +
      `model state is only ~2.5 GB — leaving over 70 GB for activations. There is no memory pressure, so ` +
      `the recomputation overhead of \`AC\` simply wastes cycles.\n\n` +
      `Let's recap the beginner track:\n\n` +
      `- \`BF16\` precision: unlocked Tensor Core throughput (16x over \`FP32\`)\n` +
      `- Flash Attention: eliminated quadratic attention memory, improved speed\n` +
      `- Activation checkpointing: a powerful tool when memory is tight — but not always needed\n` +
      `- Multi-node scaling: inter-node fabrics connect nodes, throughput scales with more GPUs\n` +
      `- \`FSDP\`: sharded model state to use memory efficiently across GPUs\n\n` +
      `In the intermediate track, you'll learn tensor parallelism, pipeline parallelism, and how to scale ` +
      `beyond a single node to train models with tens or hundreds of billions of parameters.\n\n` +
      `Check the Training Projection panel to see how your \`MFU\` improvement translates to training time and cost.`,
  },
];
