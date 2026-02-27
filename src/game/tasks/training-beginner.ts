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
      `You have a single NVIDIA A100 GPU with 80 GB of memory, and a small model: GPT-3 125M (125 million parameters). ` +
      `This is a toy-sized language model by modern standards, but it's perfect for learning the ropes.\n\n` +
      'Your goal: get the model to train at over 50k tokens per second. The default configuration uses FP32 precision, ' +
      'which runs on standard CUDA cores at 19.5 {{tflops|TFLOPS}}. But the A100 has specialized {{tensor-cores|Tensor Cores}} that accelerate BF16 operations ' +
      'at 312 TFLOPS — a 16× improvement. Find the setting that unlocks this performance.',
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
    hints: [
      'Run the simulation first at FP32 and observe the throughput — it will be far below 50k tok/s. The GPU is running on slow CUDA cores.',
      'The A100 has Tensor Cores that accelerate reduced-precision matrix multiplications. Look for the Precision setting in the sidebar.',
      'Switch precision from FP32 to BF16. Tensor Cores process BF16 at 312 TFLOPS vs 19.5 TFLOPS for FP32 — a 16× increase in peak throughput.',
    ],
    successExplanation:
      `Switching from FP32 to BF16 unlocked the A100's Tensor Cores, jumping from ~12k to ~154k tok/s. ` +
      `A 125M parameter model in FP32 uses about 2.25 GB total (18 bytes/param for weights, gradients, and optimizer states), ` +
      `leaving plenty of headroom on an 80 GB GPU — the issue was never memory, but compute throughput.\n\n` +
      `The MFU you see tells you what fraction ` +
      `of the GPU's theoretical peak compute is being used for useful work. For a small model on a single GPU, MFU ` +
      `is typically modest — small models don't generate enough arithmetic to keep the GPU's thousands of cores busy. ` +
      `As models grow larger, we'll see MFU improve, but new challenges will emerge.`,
  },

  // ── Task 2: Understanding GPU Memory ─────────────────────────────────
  {
    id: 'training-beginner-02',
    mode: 'training',
    difficulty: 'beginner',
    order: 1,
    title: 'Understanding GPU Memory',
    briefing:
      `Time to train a real model. Gemma 3 4B has about 4 billion parameters — 32 times larger than our previous model. ` +
      `You still have a single A100-80GB GPU.\n\n` +
      `Try running the simulation. You'll hit an Out of Memory ({{oom|OOM}}) error. Training a model requires far more ` +
      `memory than just the weights: you need space for gradients, {{optimizer-states|optimizer states}} (like Adam's momentum and variance), ` +
      `and {{activation-memory|activation memory}} from the forward pass.\n\n` +
      `In FP32 (32-bit floating point), the model state alone takes about 16 bytes per parameter — ` +
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
        label: 'Memory fits on GPU (utilization < 100%)',
      },
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Training runs successfully',
      },
    ],
    hints: [
      'Look at how much memory each component uses. The optimizer states alone in FP32 take 8 bytes per parameter. Can you use a smaller number format?',
      'Modern GPUs like the A100 have hardware support for BF16 (bfloat16), which uses 2 bytes instead of 4 per value. Check the Precision setting.',
      'Switch the training precision from FP32 to BF16. This halves the memory for parameters and gradients, and the optimizer uses mixed precision as well.',
    ],
    successExplanation:
      `BF16 mixed precision stores parameters in 2 bytes instead of 4, and activations are ` +
      `also stored in BF16 during the forward pass, halving activation memory. While the ` +
      `optimizer states actually use 12 bytes per parameter in BF16 mode (FP32 master weights + ` +
      `momentum + variance), the activation savings are enough to bring the total under 80 GB.\n\n` +
      `Combined with the 16x compute speedup from Tensor Cores, BF16 is strictly better for ` +
      `training on modern GPUs. BF16 has the same exponent range as FP32 (unlike FP16), ` +
      `making it robust for training without loss scaling. Nearly all modern LLM training uses BF16 or FP8.`,
  },

  // ── Task 3: Activation Memory ───────────────────────────────────────
  {
    id: 'training-beginner-03',
    mode: 'training',
    difficulty: 'beginner',
    order: 2,
    title: 'Activation Memory',
    briefing:
      'GPT-3 1.3B fits comfortably on a single A100-80GB in BF16 — the model state is only about 23 GB. ' +
      'But look at the current setup: {{gbs|Global Batch Size}} (GBS) is 64, and the {{mbs|Micro-Batch Size}} (MBS) is also 64. ' +
      'That means the GPU tries to run all 64 sequences through the forward pass at once, storing intermediate ' +
      'tensors ({{activation-memory|activations}}) for every sequence simultaneously.\n\n' +
      'Run the simulation — you\'ll hit OOM. The model weights are fine, but 64 sequences worth of activations ' +
      'is far more than the remaining memory can hold.\n\n' +
      'The fix: {{gradient-accumulation|Gradient Accumulation}} (GA). Instead of processing all 64 sequences in ' +
      'one giant forward pass, reduce MBS so the GPU processes a few sequences at a time. It runs multiple ' +
      'forward/backward passes, accumulating gradients, then updates the optimizer once. The formula is ' +
      '`GA = GBS / (MBS × DP)`. The training result is mathematically identical — same gradients, same ' +
      'convergence — but peak memory drops dramatically.\n\n' +
      'Your goal: keep GBS at 64 and make the training fit in memory.',
    concept: 'Activation memory, micro-batch size, and gradient accumulation',
    learningObjectives: [
      'Understand that activation memory scales linearly with micro-batch size (each sequence stores layer intermediates for backprop)',
      'Know the gradient accumulation formula: GA = GBS / (MBS × DP)',
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
        label: 'Memory fits on GPU (utilization < 100%)',
      },
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Training runs successfully',
      },
    ],
    hints: [
      'Look at the memory breakdown — model weights are a small fraction of total memory. Where is the rest going? Think about what happens during a forward pass when MBS is large.',
      'Activation memory scales with how many sequences the GPU processes simultaneously. Which setting controls that? Changing it won\'t affect the total batch — another mechanism compensates automatically.',
      'Reduce the Micro-Batch Size. Gradient accumulation (GA = GBS / (MBS × DP)) automatically increases to preserve the same effective batch size with less memory per forward pass.',
    ],
    successExplanation:
      `Gradient accumulation is how real training runs achieve massive batch sizes without running out of memory. ` +
      `The GPU processes MBS sequences at a time, computes gradients, and accumulates them over GA steps before ` +
      `a single optimizer update. The result is mathematically identical to processing the full GBS at once.\n\n` +
      `The key formula: \`GA = GBS / (MBS × DP)\`. With \`GBS=64\`, \`MBS=4\`, \`DP=1\` → \`GA=16\`. Sixteen forward/backward passes, ` +
      `each holding activations for just 4 sequences instead of 64 — a 16× reduction in peak activation memory.\n\n` +
      `In practice, MBS is tuned to the largest value that fits in GPU memory, while GBS is set based on training ` +
      `convergence requirements. GA bridges them automatically. Later, when you scale to multiple GPUs, gradient ` +
      `accumulation takes on a second role: each GA step provides compute that overlaps with inter-GPU communication.`,
  },

  // ── Task 4: Activation Checkpointing ─────────────────────────────────
  {
    id: 'training-beginner-04',
    mode: 'training',
    difficulty: 'beginner',
    order: 3,
    title: 'Activation Checkpointing',
    briefing:
      `Gemma 3 4B in BF16 fits on the A100, but just barely — memory utilization is above 95%. ` +
      `That leaves almost no headroom for larger batch sizes or longer sequences.\n\n` +
      `When a model runs its forward pass, it saves intermediate results (activations) at each layer — these are ` +
      `needed later during the {{backward-pass|backward pass}} to compute gradients. For a model with many layers, activation memory ` +
      `can be substantial.\n\n` +
      `Activation checkpointing is a classic memory-compute tradeoff: instead of storing all activations, you discard ` +
      `most of them and recompute them during the backward pass. This adds roughly 25-40% more total compute (the backward pass takes about 2.5× the forward time instead of 2×) ` +
      `but dramatically reduces peak memory.\n\n` +
      `Your goal: bring memory utilization below 90% using activation checkpointing.`,
    concept: 'Trading compute for memory with activation checkpointing',
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
      `This is reflected in the difference between MFU (useful work only) and HFU (includes recompute). ` +
      `Almost every large-scale training run uses some form of checkpointing — the memory savings are essential ` +
      `for fitting large models and large batch sizes.\n\n` +
      `There is also a lighter variant called selective checkpointing. Instead of discarding all activations, ` +
      `selective checkpointing keeps MLP activations (expensive to recompute) and only discards attention activations ` +
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
      `{{self-attention|Self-attention}} is the core operation in Transformers, but it has a dirty secret: standard attention ` +
      `materializes an N x N {{attention-matrix|attention matrix}} (where N is the sequence length), consuming memory that's quadratic ` +
      `in sequence length.\n\n` +
      `You have Gemma 3 4B on a single A100-80GB with BF16 and activation checkpointing enabled. ` +
      `Look at the memory utilization — even with checkpointing, the attention score matrices are ` +
      `taking significant memory.\n\n` +
      `Flash Attention is an IO-aware attention algorithm that never materializes the full N x N matrix. Instead, ` +
      `it computes attention in tiles, dramatically reducing memory usage (from quadratic to linear in sequence length) ` +
      `and improving speed by reducing memory bandwidth bottlenecks.\n\n` +
      `Your goal: enable Flash Attention and get memory utilization below 85%.`,
    concept: 'Memory-efficient attention with Flash Attention',
    learningObjectives: [
      'Understand standard attention materializes O(N^2) attention score matrix in HBM',
      'Know Flash Attention tiles computation in SRAM, achieving O(N) memory',
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
    hints: [
      'Flash Attention eliminates the quadratic attention score memory. The savings grow with sequence length — at short sequences they are modest, but at long sequences they are essential.',
      'Enable Flash Attention in the sidebar. It reduces attention memory from O(N²) to O(N), freeing up significant headroom.',
    ],
    successExplanation:
      `[Flash Attention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) is now the default in virtually every training framework. Instead of ` +
      `computing the full N x N attention score matrix in HBM (GPU main memory), it tiles the computation to fit ` +
      `in SRAM (the fast on-chip cache), never materializing the full matrix.\n\n` +
      `The memory savings grow dramatically with sequence length: at seq_len=2048 the savings are moderate, but at ` +
      `32K or 128K tokens, standard attention would require tens of GB just for the score matrix. Flash Attention ` +
      `also improves speed by 2-4x through better memory access patterns. If you see Flash Attention available, ` +
      `there is essentially never a reason to leave it off.`,
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
      `With 8 GPUs and DDP, you process 8x as much data per step. But synchronization is not free — ` +
      `it adds communication overhead. The question is: how much throughput do you actually gain?\n\n` +
      `Your goal: train GPT-3 1.3B on 8 A100 GPUs with DDP and achieve at least 100k tokens per second. ` +
      `You have a single GPU right now — scale out to use all 8 GPUs in the node.`,
    concept: 'Data parallelism with DDP',
    learningObjectives: [
      'Understand DDP: each GPU holds full model copy, processes different data, syncs gradients via AllReduce',
      'Know NVLink provides 900 GB/s within a node, enabling near-linear DDP scaling',
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
    hints: [
      'A single GPU cannot reach 100k tok/s. Scale up the number of GPUs to add more data-parallel workers.',
      '{{nvlink|NVLink}} provides 900 GB/s of bandwidth between A100s in the same node, making AllReduce gradient sync very fast. DDP throughput scales nearly linearly within a node.',
    ],
    successExplanation:
      `DDP is the workhorse of distributed training. Within a single 8-GPU node connected by NVLink, ` +
      `the AllReduce gradient synchronization overlaps with the backward pass computation, hiding most of ` +
      `the communication cost.\n\n` +
      `The key insight: DDP scales data throughput (tokens/sec) nearly linearly when communication is fast. ` +
      `Each GPU computes on its own data, then a ring-AllReduce synchronizes gradients — the whole gradient ` +
      `tensor is reduced in \`2*(N-1)/N * size\` bytes, pipelined across the NVLink ring. ` +
      `For small models on fast interconnects, DDP approaches perfect linear scaling. ` +
      `For larger models or slower interconnects, the overhead grows — and that's where more advanced ` +
      `strategies like FSDP come in.`,
  },

  // ── Task 7: Batch Size & Gradient Accumulation ─────────────────────
  {
    id: 'training-beginner-07',
    mode: 'training',
    difficulty: 'beginner',
    order: 6,
    title: 'Batch Size & Gradient Accumulation',
    briefing:
      'You have LLaMA 3.1 8B on 16 A100 GPUs across 2 nodes with {{fsdp|FSDP}}. Run the simulation — ' +
      '{{mfu|MFU}} is terrible. The GPUs spend most of their time waiting for network transfers ' +
      'instead of computing.\n\n' +
      'Here is the problem: with `GBS=8`, `MBS=1`, and `DP=16`, gradient accumulation is ' +
      '`GA = GBS/(MBS×DP) = 8/(1×16) = 1`. Every single forward/backward pass triggers a full AllGather and ' +
      'ReduceScatter over {{infiniband|InfiniBand}}. There is zero compute to overlap with, so every FSDP collective ' +
      'is fully exposed.\n\n' +
      'In Task 3 you learned that GA bridges GBS and MBS for memory. Here GA plays a different role: ' +
      'each accumulation step is a forward/backward pass that provides compute to overlap with inter-GPU ' +
      'communication. With GA=1, there is nothing to hide behind.\n\n' +
      'Your goal: achieve MFU above 35%.',
    concept: 'Effective batch size and communication overlap',
    learningObjectives: [
      'Understand that GBS determines gradient accumulation steps: GA = GBS / (MBS x DP)',
      'Know that GA=1 exposes all FSDP communication with no compute overlap',
      'Recognize that larger GBS adds GA steps, allowing FSDP to overlap communication with compute',
      'Understand batch size impact is dramatic at multi-node scale (inter-node InfiniBand bottleneck)',
    ],
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 'a100-80gb',
      numGPUs: 16,
      gpusPerNode: 8,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
      flashAttention: true,
      activationCheckpointing: true,
      globalBatchSize: 8,
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
        value: 0.35,
        label: 'MFU > 35%',
      },
    ],
    hints: [
      'The current GA is very low — FSDP communication is fully exposed with almost no compute to hide behind. Which setting controls the number of gradient accumulation steps?',
      'Increase GBS to raise GA. More accumulation steps mean more forward/backward passes between each gradient sync, giving FSDP compute to overlap with inter-node communication.',
    ],
    successExplanation:
      `The key formula is \`GA = GBS / (MBS × DP)\`. At \`GBS=8\` with 16 data-parallel GPUs, \`GA=1\` — every ` +
      `micro-batch triggers fully exposed FSDP communication across InfiniBand. Increasing to \`GBS=64\` ` +
      `gives \`GA=4\`: four forward/backward passes per optimizer step, each providing compute that ` +
      `overlaps with the previous step's communication.\n\n` +
      `This matters far more across InfiniBand (~50 GB/s effective per GPU) than within a node on ` +
      `NVLink (~450 GB/s effective). On a single NVLink node, FSDP communication is so fast that GA ` +
      `barely affects MFU. But at multi-node scale, \`GA=1\` means GPUs spend most of their time waiting ` +
      `for network transfers.\n\n` +
      `In practice, you tune GBS for training convergence (learning rate, batch size scaling laws) and ` +
      `MBS for GPU memory — GA is the derived quantity that bridges them. GPT-3 used a GBS of 3.2 million ` +
      `tokens, achievable only through high GA combined with data parallelism.`,
  },

  // ── Task 8: Micro-Batch Size & Memory ─────────────────────────────
  {
    id: 'training-beginner-08',
    mode: 'training',
    difficulty: 'beginner',
    order: 7,
    title: 'Micro-Batch Size & Memory',
    briefing:
      'Your previous task showed that Global Batch Size controls throughput by enabling communication overlap. ' +
      'Now explore the other side: Micro-Batch Size controls memory.\n\n' +
      'LLaMA 3.1 8B is on 8 GPUs with FSDP and no {{activation-checkpointing|activation checkpointing}} — the GPU stores all ' +
      'intermediate activations for the {{backward-pass|backward pass}}. MBS is set to 4: each GPU processes 4 sequences ' +
      'in a single forward pass, storing activations for all 4 simultaneously. With `GBS=64` and `DP=8`, ' +
      '`GA = GBS/(MBS×DP) = 64/(4×8) = 2`.\n\n' +
      'The problem: 4 sequences of activations at seqLen=4096 far exceeds GPU memory. ' +
      'Reduce MBS to fit, and let gradient accumulation bridge the gap.\n\n' +
      'Your goal: fit in memory and achieve MFU above 50%.',
    concept: 'Micro-batch size as a memory control knob',
    learningObjectives: [
      'Know GA = GBS / (MBS x DP) and understand its role in amortizing communication',
      'Understand that each GA step is a forward/backward pass that overlaps with communication',
      'Recognize that GA=1 means every micro-batch triggers fully exposed FSDP collectives',
      'Know that GPT-3 used GBS=3.2M tokens, achievable only through high GA + DP',
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
        label: 'Memory fits on GPU (utilization < 100%)',
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
    hints: [
      'Activation memory scales linearly with MBS — the GPU stores intermediate activations for all sequences in the micro-batch simultaneously. The current MBS is too large to fit alongside FSDP model state.',
      'Reduce MBS and let GA adjust. `GA = GBS/(MBS×DP)` increases automatically when you lower MBS, keeping the same effective batch size with less memory per step.',
      'MBS barely affects MFU at this scale — it is purely a memory knob. The real efficiency lever was GBS (from the previous task).',
    ],
    successExplanation:
      `MBS controls per-GPU memory: activation memory scales linearly with micro-batch size because each ` +
      `sequence in the micro-batch stores its own intermediate tensors for the backward pass. Without ` +
      `activation checkpointing, all layer activations are retained — at \`MBS=4\`, that's 4× the activation ` +
      `memory of \`MBS=1\`.\n\n` +
      `The formula \`GBS = MBS × GA × DP\` means reducing MBS automatically increases GA, preserving the ` +
      `effective batch size. \`MBS=2\` with \`GBS=64\` and \`DP=8\` gives \`GA=4\`, fitting comfortably in memory ` +
      `while maintaining high MFU.\n\n` +
      `In practice, tune MBS to the largest value that fits in GPU memory — more samples per forward pass ` +
      `means better hardware utilization. Then GA fills the gap to reach the desired GBS. Activation ` +
      `checkpointing (from task 4) is the other lever for memory, but it adds recomputation overhead ` +
      `that would drop MFU below the target here.`,
  },

  // ── Task 9: FSDP — Sharding Everything ──────────────────────────────
  {
    id: 'training-beginner-09',
    mode: 'training',
    difficulty: 'beginner',
    order: 8,
    title: 'FSDP: Sharding Everything',
    briefing:
      `Qwen 3 14B has about 14 billion parameters — too large for a single A100 even with BF16. You have 8 GPUs. DDP replicates the full model on every GPU — ` +
      `parameters, gradients, and optimizer states are all duplicated 8 times. That's enormously wasteful.\n\n` +
      `Fully Sharded Data Parallel (FSDP) fixes this. Instead of replicating everything, FSDP {{sharding|shards}} the ` +
      `parameters, gradients, and optimizer states across all GPUs. Each GPU only stores 1/N of the model state. ` +
      `Before each layer's computation, an {{allgather|AllGather}} collective assembles the full parameters temporarily; ` +
      `after the backward pass, a {{reducescatter|ReduceScatter}} distributes gradients back to their shards.\n\n` +
      `The result: memory usage per GPU drops dramatically, at the cost of more communication.\n\n` +
      `Your task: Qwen 3 14B OOMs with DDP on 8 A100-80GB GPUs (even with BF16 and activation checkpointing, ` +
      `the replicated optimizer states are huge). Switch to FSDP to make it fit.`,
    concept: 'Fully Sharded Data Parallel (ZeRO-3 / FSDP)',
    learningObjectives: [
      'Understand FSDP shards parameters, gradients, and optimizer states across GPUs (= ZeRO-3)',
      'Know per-GPU model state drops to ~1/N of the full model',
      'Understand AllGather (before forward) and ReduceScatter (after backward) communication pattern',
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
        label: 'Memory fits on GPU (utilization < 100%)',
      },
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Training runs successfully',
      },
    ],
    hints: [
      'Run DDP first to confirm it OOMs. Then look at the strategy selector — there is a strategy that shards all model state across GPUs instead of replicating it.',
      'Switch the strategy from DDP to FSDP. FSDP (equivalent to DeepSpeed ZeRO Stage 3) shards parameters, gradients, and optimizer states across all GPUs, reducing per-GPU memory by the sharding factor.',
    ],
    successExplanation:
      `FSDP is one of the most important innovations in distributed training. It implements the same idea as ` +
      `DeepSpeed [ZeRO (Rajbhandari et al., 2019)](https://arxiv.org/abs/1910.02054) Stage 3: shard everything across the data-parallel group.\n\n` +
      `With DDP, a 14B model needs ~18 bytes/param per GPU (params + grads + optimizer). That's ~252 GB per GPU — ` +
      `impossible on 80 GB. With FSDP across 8 GPUs, each GPU stores only \`1/8\` of the model state, bringing ` +
      `per-GPU usage to ~31 GB for model state.\n\n` +
      `The communication cost is higher than DDP: FSDP needs an AllGather before each forward layer and a ` +
      `ReduceScatter after each backward layer. But with NVLink within a node, this communication is fast and ` +
      `can overlap with compute. FSDP is the default strategy for training models from 7B to 70B on single nodes.`,
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
      `DDP, and FSDP. Now put it all together.\n\n` +
      `You have 8 NVIDIA {{h100|H100}} SXM GPUs — the current workhorse of AI training. Each H100 delivers 989 BF16 TFLOPS ` +
      `(over 3x the A100) and the NVLink 4.0 interconnect provides 900 GB/s of bandwidth.\n\n` +
      `Your mission: train LLaMA 3.1 8B with FSDP on these 8 H100s and achieve MFU above 50%. Published benchmarks ` +
      `for 8B models on H100 clusters report MFU in the 50-57% range.\n\n` +
      `Here's a twist: FSDP shards model state so efficiently across 8 GPUs that you may have more memory ` +
      `headroom than you think. Every optimization has a cost — even activation checkpointing adds recomputation ` +
      `overhead that shows up as lower MFU (the recompute is not "useful work"). For maximum efficiency, use only ` +
      `the optimizations you actually need.`,
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
    hints: [
      'MFU is held back by the backward pass recomputing activations (extra compute that does not count as useful work in MFU).',
      'FSDP shards the model state across all GPUs — per-GPU model state is tiny compared to GPU capacity. You have plenty of memory headroom. Do you really need activation checkpointing?',
      'Disable activation checkpointing — the 8B model fits comfortably with FSDP across 8 GPUs. Without the recomputation overhead, MFU improves significantly. Try increasing MBS for better utilization.',
    ],
    successExplanation:
      `Congratulations — you've completed the beginner track! You achieved over 50% MFU by recognizing ` +
      `that not every optimization is always needed.\n\n` +
      `Activation checkpointing trades compute for memory. With FSDP sharding 8B across 8 GPUs, per-GPU ` +
      `model state is only ~2.5 GB — leaving over 70 GB for activations. There is no memory pressure, so ` +
      `the recomputation overhead of AC simply wastes cycles.\n\n` +
      `Let's recap the beginner track:\n\n` +
      `- BF16 precision: unlocked Tensor Core throughput (16x over FP32)\n` +
      `- Flash Attention: eliminated quadratic attention memory, improved speed\n` +
      `- Activation checkpointing: a powerful tool when memory is tight — but not always needed\n` +
      `- Batch size tuning: saturated GPU cores with large matrix multiplications\n` +
      `- FSDP: sharded model state to use memory efficiently across GPUs\n\n` +
      `In the intermediate track, you'll learn tensor parallelism, pipeline parallelism, and how to scale ` +
      `beyond a single node to train models with tens or hundreds of billions of parameters.\n\n` +
      `Check the Training Projection panel to see how your MFU improvement translates to training time and cost.`,
  },
];
