import type { GameTask } from '../types.ts';
// Glossary annotations: {{termId}} or {{termId|display text}} for tooltip terms.
// [text](url) for paper links in successExplanation.
// Annotate FIRST OCCURRENCE per difficulty tier only. Check earlier tasks before adding.

export const TRAINING_INTERMEDIATE_TASKS: GameTask[] = [
  // ── 1. Why Tensor Parallelism? ──────────────────────────────────────
  {
    id: 'training-intermediate-01',
    mode: 'training',
    difficulty: 'intermediate',
    order: 0,
    title: 'Why Tensor Parallelism?',
    concept: 'Tensor Parallelism for large model memory',
    learningObjectives: [
      'Recognize when FSDP alone is insufficient: activation memory too large for single-GPU compute',
      'Understand TP partitions weight matrices so each GPU computes a slice of each layer',
      'Know TP reduces both parameter and activation memory by ~1/TP per GPU',
    ],
    briefing:
      'LLaMA 3.3 70B has 70 billion parameters. Even with {{fsdp|FSDP}} sharding model state across 16 GPUs, the per-GPU {{activation-memory|activation memory}} for a ' +
      '70B model is enormous — each GPU must compute the full forward pass through its share of ' +
      'layers, storing all intermediate tensors. TP reduces activation memory by splitting each layer across GPUs. When FSDP alone runs ' +
      'out of memory, you need to split the model weights themselves across GPUs ' +
      'within a node. That is what Tensor Parallelism (TP) does: it partitions ' +
      'each layer\'s weight matrices so that every GPU computes only a slice of ' +
      'each matrix multiply. Your mission: switch from pure FSDP to FSDP+TP and ' +
      'get the model to fit in memory.',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'h100-sxm',
      numGPUs: 16,
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
        label: 'Simulation succeeds',
      },
      {
        field: 'mfu',
        operator: '>',
        value: 0.40,
        label: 'MFU > 40%',
      },
    ],
    hints: [
      'FSDP across 16 GPUs works, but MFU is limited by inter-node FSDP ' +
        'communication. With 16 GPUs across 2 nodes, the AllGather and ' +
        'ReduceScatter cross InfiniBand for every layer.',
      'Switch the strategy from "FSDP" to "FSDP + TP" in the sidebar. TP ' +
        'handles intra-node communication over {{nvlink|NVLink}} (900 GB/s) while FSDP ' +
        'handles inter-node communication. This reduces the volume of slow ' +
        'inter-node traffic.',
      'Try different TP degrees with FSDP+TP. TP AllReduces stay on NVLink, ' +
        'and FSDP overlaps its inter-node comms with compute. Balance TP ' +
        '(memory reduction) against DP (throughput scaling).',
    ],
    successExplanation:
      'Tensor Parallelism splits each weight matrix column-wise (or row-wise) ' +
      'across TP GPUs. Because each GPU only computes a fraction of each layer, ' +
      'both the parameter memory and the activation memory shrink by roughly 1/TP.\n\n' +
      'This is why TP is the first tool to reach for when a model is too large for ' +
      'FSDP alone. The tradeoff: TP requires an AllReduce after every transformer ' +
      'layer, so it works best over the fast NVLink interconnect within a single node.',
  },

  // ── 2. TP Degree Selection ──────────────────────────────────────────
  {
    id: 'training-intermediate-02',
    mode: 'training',
    difficulty: 'intermediate',
    order: 1,
    title: 'TP Degree Selection',
    concept: 'Choosing the right TP degree for throughput',
    learningObjectives: [
      'Understand the TP tradeoff: lower TP gives larger GEMMs but more memory; higher TP gives more communication per layer',
      'Know TP should be a power of 2 and stay within an NVLink-connected node',
      'Experiment to find optimal TP for a given model/GPU combination',
    ],
    briefing:
      'You got LLaMA 3.3 70B to fit in memory by adding TP. But not all TP degrees ' +
      'are equal: higher TP means less computation per GPU per layer but more ' +
      '{{allreduce|AllReduce}} communication after every layer. On 16 H100s (2 nodes), TP could be 1, 2, ' +
      '4, or 8. Your goal is to find the TP degree that achieves the best ' +
      'throughput, measured by {{mfu|MFU}} above 40%. Experiment with different TP values ' +
      'and observe how MFU changes.',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'h100-sxm',
      numGPUs: 16,
      strategyType: 'fsdp-tp',
      mixedPrecision: 'bf16',
      flashAttention: true,
      activationCheckpointing: true,
      sequenceParallel: true,
    },
    winningCriteria: [
      {
        field: 'mfu',
        operator: '>',
        value: 0.40,
        label: 'MFU > 40%',
      },
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Simulation succeeds',
      },
    ],
    hints: [
      'Higher TP means each GPU computes less per layer but communicates more (AllReduce every layer). Lower TP means larger matrix multiplies (better GPU utilization) but more per-GPU memory. Remember: `DP = totalGPUs / TP`.',
      'Experiment with different TP degrees. Lower TP gives more DP for throughput scaling, higher TP reduces memory. Also make sure your batch size is large enough to keep GPUs fed.',
    ],
    successExplanation:
      'The optimal TP degree balances two forces: (1) each GPU needs enough work ' +
      'per layer to saturate its compute units (favors lower TP), and (2) each GPU ' +
      'must have enough memory to hold activations (favors higher TP).\n\nFor 70B on ' +
      '8 H100s, `TP=4` with `DP=2` is often the sweet spot -- each GPU runs `70B/4` worth ' +
      'of parameters per layer while still getting data-parallel throughput scaling. ' +
      'TP should always be a power of 2 and stay within a single NVLink-connected node.',
  },

  // ── 3. The NVLink Boundary ─────────────────────────────────────────
  {
    id: 'training-intermediate-03',
    mode: 'training',
    difficulty: 'intermediate',
    order: 2,
    title: 'The NVLink Boundary',
    concept: 'Why TP must stay within a node',
    learningObjectives: [
      'Discover that cross-node TP over InfiniBand is catastrophically slow vs intra-node NVLink',
      'Understand 2D parallelism: TP for intra-node (NVLink), FSDP for inter-node (InfiniBand)',
      'Know why frequent TP AllReduces need fast NVLink while FSDP tolerates slower IB',
    ],
    briefing:
      'You have LLaMA 3.3 70B on 32 H100 GPUs across 4 nodes. The current configuration uses ' +
      'TP=16 — spanning two full nodes. MFU is terrible. Investigate why and fix it.\n\n' +
      'Your goal: achieve MFU above 40% while keeping memory utilization below 60%.',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'h100-sxm',
      numGPUs: 32,
      strategyType: 'fsdp-tp',
      mixedPrecision: 'bf16',
      flashAttention: true,
      activationCheckpointing: true,
      sequenceParallel: true,
      tpDegree: 16,
    },
    winningCriteria: [
      {
        field: 'mfu',
        operator: '>',
        value: 0.40,
        label: 'MFU > 40%',
      },
      {
        field: 'memoryUtilization',
        operator: '<',
        value: 0.60,
        label: 'Memory < 60%',
      },
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Simulation succeeds',
      },
    ],
    hints: [
      'Look at the TP degree relative to GPUs per node. TP AllReduce happens at every transformer layer — what interconnect is it using?',
      'Reduce TP so it fits within a single node. Let FSDP handle the inter-node communication instead.',
    ],
    successExplanation:
      '2D parallelism (FSDP+TP) is the workhorse of modern LLM training. The ' +
      'principle is simple: use the fastest interconnect for the most frequent ' +
      'communication. TP AllReduces happen every layer (high frequency), so they ' +
      'must go over NVLink (~900 GB/s). When TP=16 spans two nodes, those AllReduces ' +
      'cross InfiniBand (~400 GB/s) — a massive bandwidth drop that destroys MFU.\n\n' +
      'FSDP AllGathers happen once per layer but can ' +
      'be pipelined and overlapped with compute, so they tolerate the slower ' +
      'InfiniBand. Keeping TP within a node and FSDP across nodes ' +
      'is the standard recipe for clusters up to a few hundred GPUs.',
  },

  // ── 4. Pipeline Parallelism Intro ───────────────────────────────────
  {
    id: 'training-intermediate-04',
    mode: 'training',
    difficulty: 'intermediate',
    order: 3,
    title: 'Pipeline Parallelism Intro',
    concept: 'Splitting models across pipeline stages',
    learningObjectives: [
      'Understand PP assigns contiguous blocks of layers to different pipeline stages',
      'Know 3D parallelism: TP (intra-layer) + PP (inter-layer) + DP (data); TP x PP x DP = totalGPUs',
      'Understand why ZeRO-1 (optimizer sharding) pairs with PP: PP reduces replicated weights, making FSDP overhead unnecessary',
      'Know layers must divide evenly by PP degree',
    ],
    briefing:
      'GPT-3 175B is one of the largest dense models. With {{zero1|ZeRO-1}}+TP, the ' +
      '{{optimizer-states|optimizer state}} is sharded across DP ranks but parameters and gradients ' +
      'are replicated. At TP=8, each GPU holds `175B/8 ≈ 22B` params — that is ' +
      '44 GB of BF16 weights plus 88 GB of FP32 gradients, far exceeding an ' +
      'A100\'s 80 GB. Pipeline Parallelism (PP) splits the model\'s layers ' +
      'across pipeline stages, reducing the per-GPU weight and gradient memory. ' +
      'You have 64 A100 GPUs. Add PP to get the model running.',
    setup: {
      modelId: 'gpt3-175b',
      gpuId: 'a100-80gb',
      numGPUs: 64,
      strategyType: 'zero1-tp-pp',
      mixedPrecision: 'bf16',
      flashAttention: true,
      activationCheckpointing: true,
      sequenceParallel: true,
    },
    winningCriteria: [
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Simulation succeeds',
      },
      {
        field: 'mfu',
        operator: '>',
        value: 0.35,
        label: 'MFU > 35%',
      },
    ],
    hints: [
      'GPT-3 175B has 96 layers. PP divides these across pipeline stages, ' +
        'reducing the replicated weights and gradients per GPU. The number of ' +
        'layers must divide evenly by PP.',
      'The constraint is `TP × PP × DP = totalGPUs`. Keep TP within a node. ' +
        'Then try different PP values — more PP reduces memory per GPU but also ' +
        'reduces DP (which limits throughput). Check the pipeline timeline to ' +
        'see how microbatches flow through stages.',
    ],
    successExplanation:
      'Pipeline Parallelism is the third dimension of 3D parallelism. While TP ' +
      'splits individual layers horizontally (across matrix columns) and ZeRO-1 ' +
      'shards optimizer state, PP splits the model vertically — each stage ' +
      'handles a contiguous block of layers. With ZeRO-1, parameters and ' +
      'gradients are replicated per DP rank, so PP is essential to reduce the ' +
      'per-GPU memory footprint for very large models.\n\nThe cost is the ' +
      '"pipeline bubble": when the first stage starts its forward pass, the last ' +
      'stage is idle, and vice versa. We will tackle the bubble in the next task.\n\n' +
      'This task uses ZeRO-1 (optimizer-only sharding) instead of FSDP. With PP already splitting ' +
      'layers across stages, FSDP\'s parameter sharding adds AllGather/ReduceScatter overhead with ' +
      'diminishing memory benefit — the replicated weights per DP rank are already reduced by PP. ' +
      'ZeRO-1 keeps gradient AllReduce simple and efficient.',
  },

  // ── 5. The Pipeline Bubble ──────────────────────────────────────────
  {
    id: 'training-intermediate-05',
    mode: 'training',
    difficulty: 'intermediate',
    order: 4,
    title: 'The Pipeline Bubble',
    concept: 'Reducing pipeline idle time with microbatches',
    learningObjectives: [
      'Know the bubble formula: (pp-1)/(pp-1+m) where m = number of microbatches',
      'Understand more microbatches reduce the bubble fraction',
      'Know m = GA = GBS/(MBS x DP) — batch size directly controls pipeline efficiency',
      'Recognize the bubble as the primary throughput cost of PP',
    ],
    briefing:
      'You got GPT-3 175B running with pipeline parallelism. But look at the ' +
      'pipeline bubble metric — it might be alarmingly high. The bubble represents ' +
      'the fraction of time that pipeline stages sit idle, waiting for activations ' +
      'from the previous stage or gradients from the next. The formula is ' +
      '`(pp-1)/(pp-1+m)` where `m` is the number of {{microbatches}}. More microbatches ' +
      'mean more work to fill the pipeline and less idle time. With `PP=4`, `TP=8`, ' +
      'and a small global batch size, the bubble is over 20%. Reduce it below 20% ' +
      'by increasing the global batch size (which increases the number of microbatches).',
    setup: {
      modelId: 'gpt3-175b',
      gpuId: 'a100-80gb',
      numGPUs: 64,
      strategyType: 'zero1-tp-pp',
      mixedPrecision: 'bf16',
      flashAttention: true,
      activationCheckpointing: true,
      tpDegree: 8,
      ppDegree: 4,
      globalBatchSize: 16,
      sequenceParallel: true,
    },
    winningCriteria: [
      {
        field: 'pipelineBubble',
        operator: '<',
        value: 0.20,
        label: 'Pipeline bubble < 20%',
      },
    ],
    hints: [
      'The number of microbatches (`m`) equals the {{gradient-accumulation|gradient accumulation}} steps: ' +
        '`GBS / (MBS × DP)`. To increase `m`, raise the global batch size. The ' +
        'pipeline Gantt chart visualizes the bubble — gaps between forward/backward ' +
        'blocks shrink as you add microbatches.',
      'With `PP=4`, `bubble = (pp-1)/(pp-1+m)`. The starting GBS is small, giving ' +
        'very few microbatches per DP rank. Try increasing GBS until the bubble ' +
        'drops below the threshold. Fewer PP stages also means less bubble at ' +
        'the same batch size.',
    ],
    successExplanation:
      'The pipeline bubble is inherent to synchronous pipeline parallelism: ' +
      'during the "ramp up" the later stages wait for activations, and during ' +
      '"ramp down" the earlier stages wait for gradients. The 1F1B (one-forward- ' +
      'one-backward) schedule minimizes the number of in-flight microbatches while ' +
      'keeping stages busy.\n\nThe key lever is the number of microbatches: each ' +
      'additional microbatch amortizes the fixed startup/shutdown cost across more ' +
      'useful work. In practice, teams aim for `m >= 4×pp` to keep bubbles under 20%.',
  },

  // ── 6. Interleaved Scheduling ───────────────────────────────────────
  {
    id: 'training-intermediate-06',
    mode: 'training',
    difficulty: 'intermediate',
    order: 5,
    title: 'Interleaved Scheduling',
    concept: 'Virtual pipeline stages to cut the bubble further',
    learningObjectives: [
      'Understand virtual pipeline stages: each rank gets v non-contiguous chunks of layers',
      'Know interleaved bubble = (pp-1)/((pp-1+m) x v) — v divides the bubble',
      'Understand the tradeoff: more VP = less bubble but more P2P communication and memory',
      'Know layers per physical stage must be divisible by v',
    ],
    briefing:
      'You reduced the pipeline bubble with more microbatches. But there is ' +
      'another powerful trick: interleaved {{1f1b|1F1B}} scheduling. Instead of assigning ' +
      'each pipeline stage one contiguous block of layers, you assign it multiple ' +
      'smaller "virtual stages." With `v` virtual stages per rank, the effective ' +
      'bubble shrinks by a factor of `v` because each stage\'s chunk of work is ' +
      'smaller and the pipeline fills faster. For GPT-3 175B with 96 layers and ' +
      '`PP=8`, setting `v=4` gives each stage 3 layers per virtual stage (`96/8/4=3`). ' +
      'The current config uses `PP=8` with 1F1B and the bubble is above 12%. ' +
      'Switch to interleaved scheduling to get it below 12%.',
    setup: {
      modelId: 'gpt3-175b',
      gpuId: 'a100-80gb',
      numGPUs: 64,
      strategyType: 'zero1-tp-pp',
      mixedPrecision: 'bf16',
      flashAttention: true,
      activationCheckpointing: true,
      tpDegree: 8,
      ppDegree: 8,
      globalBatchSize: 16,
      sequenceParallel: true,
    },
    winningCriteria: [
      {
        field: 'pipelineBubble',
        operator: '<',
        value: 0.12,
        label: 'Pipeline bubble < 12%',
      },
    ],
    hints: [
      'Look for the pipeline schedule selector in the sidebar. Switch from ' +
        '"1F1B" to "Interleaved 1F1B." You will also need to set the number ' +
        'of virtual pipeline stages (v).',
      'With interleaved scheduling, the effective bubble becomes ' +
        '`(pp-1)/((pp-1+m)×v)`. The number of layers per physical stage must be ' +
        'divisible by `v`. GPT-3 has 96 layers; with `PP=8` that is 12 layers per ' +
        'stage. Check which values of `v` evenly divide the layers per stage. ' +
        'The pipeline timeline shows the interleaved schedule — compare it to ' +
        'standard 1F1B.',
      'Try interleaved 1F1B with a `v` that gives fine-grained stages, and ' +
        'increase GBS so each DP rank has enough microbatches. Higher `v` shrinks ' +
        'the bubble dramatically — even modest values make a big difference.',
    ],
    successExplanation:
      'Interleaved 1F1B assigns each pipeline rank v non-contiguous chunks of ' +
      'layers instead of one contiguous block. This shrinks the "granularity" of ' +
      'each pipeline step, so the pipeline fills and drains v times faster.\n\nThe ' +
      'tradeoff is more point-to-point communication (activations must hop between ' +
      'stages more often) and slightly more complex scheduling. [Megatron-LM (Narayanan et al., 2021)](https://arxiv.org/abs/2104.04473) and ' +
      'similar frameworks use interleaved scheduling by default for large PP ' +
      'degrees. Nemotron 340B, for example, uses `PP=12` with `v=8` to achieve a ' +
      'bubble around 10%.',
  },

  // ── 7. Sequence Parallelism ─────────────────────────────────────────
  {
    id: 'training-intermediate-07',
    mode: 'training',
    difficulty: 'intermediate',
    order: 6,
    title: 'Sequence Parallelism',
    concept: 'Splitting activations along the sequence dimension',
    learningObjectives: [
      'Understand SP partitions activations along sequence dimension for non-TP ops (LayerNorm, dropout)',
      'Know SP converts AllReduce into ReduceScatter + AllGather with identical total volume',
      'Know with SP enabled, ALL activations become 1/TP per rank (both matmul and non-matmul)',
      'Recognize SP as nearly free with TP — same communication volume, better memory and overlap',
    ],
    briefing:
      'Even with TP splitting weight matrices, some operations like {{layernorm|LayerNorm}} ' +
      'and {{dropout}} are replicated on every TP rank -- each GPU holds the full ' +
      'sequence of activations for these ops. Sequence Parallelism (SP) fixes ' +
      'this by partitioning activations along the sequence dimension for these ' +
      'replicated regions. The AllReduce in TP becomes a {{reducescatter|ReduceScatter}} + ' +
      '{{allgather|AllGather}} pair, but the activation memory per GPU drops to 1/TP even for ' +
      'non-matmul ops. You have 70B on 16 H100s with TP=4. Enable SP and push ' +
      'MFU above 41%. Without SP, replicated activations waste memory and the ' +
      'TP AllReduce pattern is less overlap-friendly.',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'h100-sxm',
      numGPUs: 16,
      strategyType: 'fsdp-tp',
      mixedPrecision: 'bf16',
      flashAttention: true,
      activationCheckpointing: true,
      sequenceParallel: false,
      tpDegree: 4,
      globalBatchSize: 128,
    },
    winningCriteria: [
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Simulation succeeds',
      },
      {
        field: 'mfu',
        operator: '>',
        value: 0.41,
        label: 'MFU > 41%',
      },
    ],
    hints: [
      'Without SP, TP splits weight-related activations (Q, K, V projections) ' +
        'but leaves LayerNorm inputs, residual connections, and dropout masks ' +
        'fully replicated on every TP rank. These "replicated" activations can ' +
        'account for a significant fraction of memory.',
      'Look for the "Sequence Parallelism" checkbox in the sidebar. It appears ' +
        'when you are using a 2D or 3D strategy with TP. Enable it and watch ' +
        'the memory utilization drop.',
      'With SP enabled, ALL activations (both matmul and non-matmul) become ' +
        'effectively 1/TP per rank — not just the weight-related ones. The ' +
        'memory savings compound with the number of layers.',
    ],
    successExplanation:
      'Sequence Parallelism is a nearly free optimization when used with TP. ' +
      'Instead of an AllReduce (which is ReduceScatter + AllGather combined), SP ' +
      'keeps the ReduceScatter and AllGather as separate operations around the ' +
      'non-tensor-parallel regions. Between them, each GPU holds only `1/TP` of the ' +
      'sequence.\n\nThe total communication volume is identical to TP\'s AllReduce, ' +
      'but the memory savings are substantial -- especially for long sequences ' +
      'where activation memory dominates. SP is enabled by default in most modern ' +
      'training frameworks like Megatron-LM.',
  },

  // ── 8. Scaling to a Cluster ─────────────────────────────────────────
  {
    id: 'training-intermediate-08',
    mode: 'training',
    difficulty: 'intermediate',
    order: 7,
    title: 'Scaling to a Cluster',
    concept: 'Multi-node training with inter-node communication',
    learningObjectives: [
      'Understand inter-node InfiniBand (~400 GB/s) as a bottleneck at 4+ nodes vs intra-node NVLink (~900 GB/s)',
      'Know the standard pattern: TP intra-node, FSDP/DP inter-node',
      'Understand FSDP backward prefetch overlaps AllGather with compute, hiding inter-node cost',
      'Know larger batch sizes improve compute-to-communication ratio at multi-node scale',
    ],
    briefing:
      'Training at scale means running across many nodes connected by InfiniBand ' +
      'rather than NVLink. With 64 H100s (8 nodes of 8 GPUs), inter-node FSDP ' +
      'communication becomes a real bottleneck. The AllGather to reconstruct ' +
      'weights and the ReduceScatter for gradients now traverse the network ' +
      'fabric. Batch size, TP degree, and overlap efficiency all matter more at ' +
      'this scale. Configure FSDP+TP to achieve MFU above 40% with memory ' +
      'utilization below 40% on 64 GPUs.',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'h100-sxm',
      numGPUs: 64,
      strategyType: 'fsdp-tp',
      mixedPrecision: 'bf16',
      flashAttention: true,
      activationCheckpointing: true,
      sequenceParallel: true,
    },
    winningCriteria: [
      {
        field: 'mfu',
        operator: '>',
        value: 0.40,
        label: 'MFU > 40%',
      },
      {
        field: 'memoryUtilization',
        operator: '<',
        value: 0.40,
        label: 'Memory < 40%',
      },
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Simulation succeeds',
      },
    ],
    hints: [
      'TP should stay within a node (TP <= 8) so its frequent AllReduces use ' +
        'fast NVLink. FSDP spans all GPUs across nodes, communicating over ' +
        'InfiniBand. The DP degree is determined by totalGPUs / TP.',
      'Larger batch sizes help: more gradient accumulation steps mean more ' +
        'compute per communication round. FSDP can overlap its AllGather with ' +
        'the forward pass ({{backward-prefetch|backward prefetch}}), hiding much of the cost.',
      'Try different TP degrees within a node and increase GBS significantly. ' +
        'Enable SP for activation memory savings. FSDP overlaps well with large ' +
        'enough batches.',
    ],
    successExplanation:
      'Scaling from 1-2 nodes to 4+ nodes introduces inter-node communication ' +
      'as a significant factor.\n\nThe key insights: (1) keep TP within a node so ' +
      'its frequent AllReduces use NVLink, (2) let FSDP handle inter-node ' +
      'communication because its AllGather can be overlapped with compute via ' +
      'backward prefetch, and (3) use large enough batch sizes so that compute ' +
      'time dominates communication time. This is the standard "TP intra-node, ' +
      'DP inter-node" pattern used by virtually all large-scale training runs.',
  },

  // ── 9. MoE: Mixture of Experts ─────────────────────────────────────
  {
    id: 'training-intermediate-09',
    mode: 'training',
    difficulty: 'intermediate',
    order: 8,
    title: 'MoE: Mixture of Experts',
    concept: 'Training sparse Mixture-of-Experts models',
    learningObjectives: [
      'Understand MoE: total params != active params; router selects top-K experts per token',
      'Know MoE total params make DDP infeasible — 47B × 18 bytes/param far exceeds single-GPU memory',
      'Know FSDP shards all params including idle experts, reducing per-GPU memory',
      'Know MFU uses active FLOPs (6 x activeParams x tokens), not total params',
    ],
    briefing:
      'Mixtral 8x7B is a Mixture-of-Experts model: it has 8 expert FFN blocks ' +
      'per layer, but the {{router}} activates only 2 for each token. This means the ' +
      'total parameter count is large (~47B) but the {{active-params|active parameters}} per token ' +
      'are much smaller (~13B).\n\n' +
      'The setup uses DDP, which replicates ALL 47B parameters on every GPU — that is ' +
      '~846 GB of model state per GPU. Run the simulation and observe the OOM. Then switch ' +
      'to the strategy that shards everything across GPUs.\n\n' +
      'Your goal: get Mixtral running and achieve MFU above 35%.',
    setup: {
      modelId: 'mixtral-8x7b',
      gpuId: 'h100-sxm',
      numGPUs: 16,
      strategyType: 'ddp',
      mixedPrecision: 'bf16',
      flashAttention: true,
      activationCheckpointing: true,
      globalBatchSize: 128,
    },
    winningCriteria: [
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Simulation succeeds',
      },
      {
        field: 'mfu',
        operator: '>',
        value: 0.35,
        label: 'MFU > 35%',
      },
    ],
    hints: [
      'DDP replicates all parameters on every GPU. With 47B total params, the model state per GPU is enormous. Switch to a strategy that shards model state across GPUs.',
      'FSDP shards parameters, gradients, and optimizer states — reducing per-GPU memory by the number of GPUs. MoE models benefit hugely because all expert weights are sharded.',
    ],
    successExplanation:
      'Mixture-of-Experts models are "sparse" -- they have many parameters but ' +
      'activate only a subset per token. Mixtral routes each token to 2 of 8 ' +
      'experts, so compute scales with ~13B active params while knowledge is ' +
      'stored in ~47B total params.\n\nDDP is infeasible for MoE models because it ' +
      'replicates ALL parameters (including idle experts) on every GPU. FSDP shards ' +
      'everything across GPUs, bringing per-GPU memory to a manageable level. ' +
      'At larger scale, Expert Parallelism (EP) ' +
      'assigns different experts to different GPUs, requiring All-to-All ' +
      'communication to route tokens -- but that is an advanced topic. EP is covered in the ' +
      'advanced track, along with the All-to-All communication pattern it uses.',
  },

  // ── 10. Precision Frontier: FP8 ────────────────────────────────────
  {
    id: 'training-intermediate-10',
    mode: 'training',
    difficulty: 'intermediate',
    order: 9,
    title: 'Precision Frontier: FP8',
    concept: 'FP8 mixed precision for higher throughput',
    learningObjectives: [
      'Know H100 FP8 Tensor Cores deliver ~2x BF16 TFLOPS (1978 vs 989)',
      'Understand Transformer Engine handles per-tensor dynamic scaling for FP8 stability',
      'Know FP8 also halves TP communication volume (1 byte vs 2 bytes per activation)',
      'Recognize FP8 as the production frontier for Hopper+ training',
    ],
    briefing:
      'H100 GPUs have dedicated FP8 Tensor Cores that deliver 2x the FLOPS of ' +
      'BF16. FP8 training uses 8-bit floating point for matrix multiplications ' +
      'while keeping master weights and critical accumulations in higher ' +
      'precision. With {{transformer-engine|Transformer Engine}} handling the dynamic scaling, FP8 can ' +
      'nearly double throughput with minimal accuracy impact for large models. ' +
      'You have LLaMA 3.3 70B on 16 H100s (2 nodes) in BF16. Switch to FP8 ' +
      'precision and push MFU above 40%.',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'h100-sxm',
      numGPUs: 16,
      strategyType: 'fsdp-tp',
      mixedPrecision: 'bf16',
      flashAttention: true,
      activationCheckpointing: true,
      sequenceParallel: true,
    },
    winningCriteria: [
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Simulation succeeds',
      },
      {
        field: 'mfu',
        operator: '>',
        value: 0.40,
        label: 'MFU > 40%',
      },
    ],
    hints: [
      'FP8 is available on Hopper (H100) and newer GPUs. Look for the ' +
        'precision selector in the sidebar and switch from BF16 to FP8. The ' +
        'peak TFLOPS doubles, so the same compute finishes in half the time.',
      'FP8 also reduces communication volume: TP AllReduce sends FP8 tensors ' +
        '(1 byte per element) instead of BF16 (2 bytes), halving TP ' +
        'communication time. This makes higher TP degrees more practical.',
      'Combine FP8 precision with a moderate TP degree and a large enough ' +
        'batch size. With the 2× FLOPS boost and reduced communication, the ' +
        'MFU target is achievable. Enable Sequence Parallelism for extra ' +
        'memory savings.',
    ],
    successExplanation:
      'FP8 training is a game-changer for Hopper and newer GPUs. The H100\'s ' +
      'FP8 Tensor Cores deliver ~2000 TFLOPS vs ~990 TFLOPS for BF16. ' +
      'NVIDIA\'s Transformer Engine handles per-tensor dynamic scaling to ' +
      'maintain training stability: it tracks the range of activations and ' +
      'weights, choosing scale factors that maximize the use of FP8\'s limited ' +
      'dynamic range.\n\nThe communication benefit is equally important -- TP ' +
      'AllReduce with FP8 collectives halves the bandwidth requirement, which ' +
      'is critical as models scale. DeepSeek V3 used FP8 training on H800s to ' +
      'achieve 43.7% MFU at massive scale.',
  },
];
