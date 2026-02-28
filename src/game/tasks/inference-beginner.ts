import type { GameTask } from '../types.ts';
// Glossary annotations: {{termId}} or {{termId|display text}} for tooltip terms.
// [text](url) for paper links in successExplanation.
// Annotate FIRST OCCURRENCE per difficulty tier only. Check earlier tasks before adding.

export const INFERENCE_BEGINNER_TASKS: GameTask[] = [
  // ── 01 ── Your First Inference ──────────────────────────────────────
  {
    id: 'inference-beginner-01',
    mode: 'inference',
    difficulty: 'beginner',
    order: 0,
    title: 'Your First Inference',
    concept: 'Running a Language Model',
    learningObjectives: [
      'Understand inference decode is memory-bandwidth-bound: GPU reads all weights from HBM per output token',
      'Know that reducing weight precision reduces bytes read per token, proportionally increasing throughput',
      'Identify the fixed (weights) vs dynamic (KV cache) memory components of inference',
    ],
    briefing:
      'You have a LLaMA 3.1 8B model on an {{rtx-4090|RTX 4090}} — the most popular GPU for local LLM inference, ' +
      'with 24 GB of GDDR6X memory. The model is loaded with {{bf16|BF16}} weights (2 bytes per parameter), ' +
      'consuming about 16 GB. It fits, and the model runs.\n\n' +
      'LLM inference {{decode}} is memory-bandwidth bound: the GPU must read every model weight from memory for every ' +
      'single output token. The fewer bytes per parameter, the faster tokens are generated — throughput is ' +
      'inversely proportional to the bytes read per step.\n\n' +
      'Your goal: push throughput above 55 tokens per second.',
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 'rtx-4090',
      numGPUs: 1,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Inference succeeds' },
      { field: 'throughput.tokensPerSecond', operator: '>', value: 55, label: 'Throughput > 55 tok/s' },
    ],
    expectedChanges: [
      { field: 'weightPrecision', check: 'changed', label: 'Changed weight precision' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'Inference decode is memory-bandwidth bound: the GPU reads all weights from memory for every output token. Throughput is inversely proportional to the bytes read per step.',
      'Reducing the bytes per parameter means fewer bytes to read per token. Look at the weight precision selector.',
      'Lower-precision formats (FP8, INT8, INT4) use fewer bytes per parameter than BF16. Try one.',
    ],
    successExplanation:
      'Inference decode is fundamentally memory-bandwidth bound. The GPU reads all model weights from memory for every ' +
      'output token, but performs relatively little computation per byte. Reducing weight precision means fewer bytes ' +
      'read per token, directly increasing throughput.\n\n' +
      'Notice the memory breakdown: model weights take a fixed amount of memory, while the KV cache grows with each ' +
      'generated token. In the coming tasks, you will learn to manage these resources as models get larger and GPUs get smaller.',
  },

  // ── 02 ── Weight Memory ─────────────────────────────────────────────
  {
    id: 'inference-beginner-02',
    mode: 'inference',
    difficulty: 'beginner',
    order: 1,
    title: 'Weight Memory',
    concept: 'Model size and precision formats',
    learningObjectives: [
      'Calculate weight memory: params x bytes_per_param (8B x 2 = 16 GB in BF16)',
      'Use INT8/INT4 quantization to reduce weight memory 2-4x',
      'Know T4 is a Turing GPU without native BF16 support — quantization is essential for older hardware',
    ],
    briefing:
      'LLaMA 3.1 8B is a serious model — in BF16, its weights alone consume about 16 GB. ' +
      'You need to run it on a T4, which has only 16 GB of memory. ' +
      'The model barely fits at BF16, leaving almost no room for {{kv-cache|KV cache}} — ' +
      'use {{quantization}} to free memory for real workloads.',
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 't4',
      numGPUs: 1,
    },
    winningCriteria: [
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Inference succeeds',
      },
      {
        field: 'memoryUtilization',
        operator: '<',
        value: 1.0,
        label: 'Fits in GPU memory',
      },
      {
        field: 'memory.weights',
        operator: '<',
        value: 9e9,
        label: 'Weights < 9 GB (quantized)',
      },
    ],
    expectedChanges: [
      { field: 'weightPrecision', check: 'changed', label: 'Changed weight precision' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'BF16 uses 2 bytes per parameter — for an 8B model, that is nearly all of the T4\'s 16 GB. Even if weights fit, there is almost no room for KV cache.',
      'Try reducing the weight precision. {{int8|INT8}} uses 1 byte per parameter (half of BF16). {{int4|INT4}} or GPTQ-INT4 uses 0.5 bytes (a quarter). Both free up significant memory for KV cache.',
      'Note: the T4 does not support BF16 natively (it is a Turing GPU). Lower-precision formats like INT8 and INT4 are a natural fit for this hardware.',
    ],
    successExplanation:
      'Weight quantization is the single most important technique for deploying large models on smaller GPUs. ' +
      'By reducing precision from 16 bits to 8 or 4 bits per parameter, you cut weight memory by 2-4x ' +
      'with surprisingly little quality loss.\n\nIn production, INT4 quantization (GPTQ, AWQ) is the standard ' +
      'approach for running 7B+ models on consumer and edge hardware.',
  },

  // ── 03 ── The KV Cache ──────────────────────────────────────────────
  {
    id: 'inference-beginner-03',
    mode: 'inference',
    difficulty: 'beginner',
    order: 2,
    title: 'The KV Cache',
    concept: 'KV Cache Memory',
    learningObjectives: [
      'Understand KV cache stores attention keys and values for all tokens in all sequences',
      'Know KV cache scales linearly with both batch size and sequence length',
      'Distinguish fixed memory (weights, loaded once) from dynamic memory (KV cache, grows per request)',
    ],
    briefing:
      'The KV (Key-Value) cache stores the attention keys and values for every token generated so far. ' +
      'Without it, the model would need to reprocess the entire sequence for every new token.\n\n' +
      'You have LLaMA 3.1 8B on an A100-80GB, serving a batch of 128 requests with 4096-token prompts. ' +
      'At this scale, the KV cache for 128 sequences × 4096 tokens is enormous — the model runs out of memory.\n\n' +
      'Your task: make this configuration fit within the GPU\'s 80 GB. ' +
      'Observe how KV cache memory scales with batch size and {{sequence-length|sequence length}}.',
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 'a100-80gb',
      numGPUs: 1,
      batchSize: 128,
      inputSeqLen: 4096,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Inference succeeds' },
      { field: 'memoryUtilization', operator: '<', value: 1.0, label: 'Fits in GPU memory' },
    ],
    expectedChanges: [
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'The KV cache consumes massive memory at this batch size and sequence length — the configuration OOMs. Which of those two dimensions can you reduce?',
      'KV cache memory = 2 (K and V) × num_layers × kv_heads × head_dim × seq_len × batch_size × bytes_per_value. Each dimension multiplies linearly. LLaMA 3.1 8B uses GQA with 8 KV heads (not the full 32 query heads), so the cache per request is smaller than MHA — but many concurrent requests still add up.',
      'Try reducing the batch size significantly. Alternatively, use FP8 KV cache precision to halve KV cache memory at the same batch size.',
    ],
    successExplanation:
      'The KV cache is the dynamic memory component of inference. Model weights are loaded once and stay constant, ' +
      'but the KV cache grows with every token in every sequence in the batch.\n\nFor long-context applications ' +
      '(documents, conversations), the KV cache can easily exceed the weight memory. Understanding this tradeoff ' +
      'between fixed weight memory and dynamic KV cache memory is fundamental to inference deployment.\n\n' +
      'Not all attention architectures create equal KV caches. LLaMA 3.1 8B uses Grouped-Query Attention ' +
      '({{gqa|GQA}}) — only 8 KV heads shared across 32 query heads, a 4× reduction compared to standard MHA. ' +
      'LLaMA 3.3 70B also uses GQA with 8 KV heads across 64 query heads, an 8× reduction. GQA dramatically ' +
      'shrinks the KV cache, which is one reason modern models have proportionally smaller KV caches than ' +
      'you might expect. MQA (Multi-Query Attention) takes this further with just 1 KV head.',
  },

  // ── 04 ── Quantization Benefits ─────────────────────────────────────
  {
    id: 'inference-beginner-04',
    mode: 'inference',
    difficulty: 'beginner',
    order: 3,
    title: 'Quantization Benefits',
    concept: 'Bandwidth savings from reduced precision',
    learningObjectives: [
      'Understand quantization improves throughput, not just memory (decode is bandwidth-bound)',
      'Know INT8/INT4 reduce bytes transferred per parameter, increasing effective bandwidth',
      'Recognize quantized models can run faster than full-precision ones due to bandwidth savings',
    ],
    briefing:
      'Qwen 3 14B has about 14 billion parameters. In BF16, the weights alone take ~28 GB — more than ' +
      'the RTX 3090\'s 24 GB of memory. The model does not fit at full precision.\n\n' +
      'Decode is memory-bandwidth bound: the GPU reads all weights from memory for every output token. ' +
      'Quantization reduces the bytes read per parameter — fewer bytes means both fitting the model AND ' +
      'faster token generation.\n\n' +
      'You have a single {{rtx-3090|RTX 3090}} (24 GB, Ampere architecture — no FP8 support). ' +
      'Your goal: fit the model and push throughput above 30 tokens per second.',
    setup: {
      modelId: 'qwen3-14b',
      gpuId: 'rtx-3090',
      numGPUs: 1,
    },
    winningCriteria: [
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Inference succeeds',
      },
      {
        field: 'throughput.tokensPerSecond',
        operator: '>',
        value: 30,
        label: 'Throughput > 30 tok/s',
      },
    ],
    expectedChanges: [
      { field: 'weightPrecision', check: 'changed', label: 'Changed weight precision' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'At BF16 (2 bytes/param), Qwen 3 14B requires 28 GB — exceeding the RTX 3090\'s 24 GB. You must quantize just to fit the model.',
      'The RTX 3090 is an Ampere GPU — it does not support FP8. Use {{int8|INT8}} (1 byte/param) or {{int4|INT4}} (0.5 bytes/param) to fit the model and improve throughput. Calculate the weight memory at each precision to see what fits.',
    ],
    successExplanation:
      'On the RTX 3090, quantization serves a dual purpose: fitting the model (28 GB at BF16 exceeds 24 GB) ' +
      'AND improving throughput. Inference decode is memory-bandwidth-bound (the GPU spends more time loading weights ' +
      'than computing).\n\nBy using INT8 or INT4, you transfer fewer bytes per parameter, which means ' +
      'higher effective bandwidth utilization and faster token generation. On consumer and older GPUs without ' +
      'FP8 support, INT8 and INT4 are the standard quantization options — and they work well.',
  },

  // ── 05 ── Batch Size and Throughput ─────────────────────────────────
  {
    id: 'inference-beginner-05',
    mode: 'inference',
    difficulty: 'beginner',
    order: 4,
    title: 'Batch Size and Throughput',
    concept: 'Batched Inference',
    learningObjectives: [
      'Understand decode is bandwidth-bound: GPU loads all weights per step regardless of batch size',
      'Know batching amortizes weight reads across multiple sequences for near-linear throughput scaling',
      'Observe that throughput scales with batch until memory fills (KV cache grows linearly with batch)',
    ],
    briefing:
      'The {{l4|L4}} is Google Cloud\'s most popular inference GPU — 24 GB of memory and modest bandwidth. ' +
      'At batch size 1, LLaMA 3.1 8B in BF16 generates very few tokens per second — the GPU wastes most of its capacity on a single request.\n\n' +
      'By processing multiple requests simultaneously (batching), you amortize the weight-loading cost ' +
      'across more useful work. Your goal: push throughput above 60 tokens per second.',
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 'l4',
      numGPUs: 1,
    },
    winningCriteria: [
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Inference succeeds',
      },
      {
        field: 'throughput.tokensPerSecond',
        operator: '>',
        value: 60,
        label: 'Throughput > 60 tok/s',
      },
    ],
    expectedChanges: [
      { field: 'batchSize', check: 'increased', label: 'Increased batch size' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'At batch size 1, the L4 loads all 16 GB of BF16 weights to produce a single token. With modest bandwidth, single-request throughput is very low — most compute sits idle. Run it and observe.',
      'Try increasing the batch size to 4, 8, or 16. Each additional request in the batch gets "free" compute because the weights are already being loaded.',
      'Watch the memory utilization as you increase batch size — the KV cache grows linearly with batch size. The L4 has 24 GB total. Weights consume a significant share, leaving limited room for KV cache. The Batch chart shows how throughput scales with batch size across precisions.',
    ],
    successExplanation:
      'Batching is the fundamental technique for efficient inference serving. A single decode step loads the ' +
      'full model weights regardless of batch size, so adding more sequences to the batch increases throughput ' +
      'almost linearly until you hit the compute ceiling or run out of memory for KV caches.\n\n' +
      'On the L4, with modest memory bandwidth, the effect of batching is especially dramatic — single-request ' +
      'throughput is low, but batching multiplies it several times over. Cloud inference GPUs like the L4 are designed for ' +
      'cost-efficient serving at moderate batch sizes.',
  },

  // ── 06 ── TTFT vs TPOT ─────────────────────────────────────────────
  {
    id: 'inference-beginner-06',
    mode: 'inference',
    difficulty: 'beginner',
    order: 5,
    title: 'TTFT vs TPOT',
    concept: 'Prefill vs Decode Latency',
    learningObjectives: [
      'Distinguish prefill (compute-bound, determines TTFT) from decode (bandwidth-bound, determines TPOT)',
      'Know TTFT depends on input length; TPOT depends on weight reads per step',
      'Understand interactive apps care about TPOT; batch processing cares about total throughput',
    ],
    briefing:
      'Inference has two distinct phases. {{prefill|Prefill}} processes the entire input prompt in parallel — ' +
      'it is compute-bound and determines the Time to First Token ({{ttft|TTFT}}). Decode generates output tokens ' +
      'one at a time — it is memory-bandwidth-bound and each step takes Time Per Output Token ({{tpot|TPOT}}). ' +
      'You\'re serving LLaMA 3.1 8B on an {{h100|H100}} at batch size 32. At this batch size, each decode step generates ' +
      'tokens for all 32 sequences simultaneously, increasing per-token latency. Reduce TPOT below 7 ms by ' +
      'understanding what drives per-token latency.',
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 'h100-sxm',
      numGPUs: 1,
      batchSize: 32,
    },
    winningCriteria: [
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Inference succeeds',
      },
      {
        field: 'latency.tpot',
        operator: '<',
        value: 7,
        label: 'TPOT < 7 ms',
      },
    ],
    expectedChanges: [
      { field: 'batchSize', check: 'decreased', label: 'Decreased batch size' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'TTFT (Time to First Token) is the latency before the first output token appears. It depends on input length because the entire prompt must be processed. TPOT (Time Per Output Token) is the latency for each subsequent token. The Batch chart shows TTFT (dashed) and TPOT (solid) lines — they diverge at larger batch sizes.',
      'TPOT at batch size 1 is dominated by memory bandwidth: the GPU must load all model weights to produce each token. Larger batches increase TPOT because each decode step generates tokens for all sequences simultaneously.',
      'If TPOT is too high, reduce the batch size. At batch=1, the GPU only generates one token per decode step, minimizing per-token latency.',
    ],
    successExplanation:
      'Understanding TTFT vs TPOT is critical for serving design. Interactive applications (chatbots) ' +
      'care most about TPOT — users perceive the streaming speed of generated text. Batch processing ' +
      'applications care more about total throughput.\n\nPrefill is compute-bound (benefits from faster FLOPS), ' +
      'while decode is bandwidth-bound (benefits from higher memory bandwidth and smaller models). ' +
      'This fundamental asymmetry drives many inference optimization choices.\n\n' +
      'For long prompts (32K+ tokens), TTFT can be hundreds of milliseconds. Production techniques like ' +
      'chunked prefill split long prompts into smaller chunks processed across multiple steps, allowing ' +
      'decode requests to interleave — reducing TTFT variance at the cost of slightly higher average TTFT.',
  },

  // ── 07 ── Flash Attention for Inference ─────────────────────────────
  {
    id: 'inference-beginner-07',
    mode: 'inference',
    difficulty: 'beginner',
    order: 6,
    title: 'Flash Attention for Inference',
    concept: 'Attention memory at long context lengths',
    learningObjectives: [
      'Understand FA eliminates O(N^2) attention memory, critical at 32K+ token context',
      'Know FA impact grows quadratically with sequence length — moderate at 2K, essential at 32K+',
      'Recognize FA as mandatory for long-context inference (OOM without it)',
    ],
    briefing:
      'Standard attention computes the full N × N {{attention-matrix|attention matrix}}, consuming O(N²) memory. ' +
      'At a 32K sequence length, this matrix alone consumes several GB — enough to overflow GPU memory.\n\n' +
      'You have LLaMA 3.1 8B on an A100-80GB with {{flash-attention|Flash Attention}} disabled and a 32K input sequence. ' +
      'The simulation will OOM because the attention score matrix is too large.\n\n' +
      'Flash Attention tiles the computation to avoid materializing the full matrix, ' +
      'reducing memory from O(N²) to O(N). Your goal: make this long-context configuration fit in memory.',
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 'a100-80gb',
      numGPUs: 1,
      flashAttention: false,
      inputSeqLen: 32768,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Inference succeeds' },
      { field: 'memoryUtilization', operator: '<', value: 1.0, label: 'Fits in GPU memory' },
    ],
    expectedChanges: [
      { field: 'flashAttention', check: 'enabled', label: 'Enabled Flash Attention' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
      { field: 'inputSeqLen', check: 'unchanged', label: 'Did not change input sequence length' },
    ],
    hints: [
      'Without Flash Attention at 32K sequence length, the attention score matrix (batch × heads × 32768² × bytes) consumes several GB. The configuration OOMs.',
      'Flash Attention computes attention in {{sram|SRAM}} tiles, never materializing the full N × N matrix in HBM. It is mathematically identical — only the memory pattern changes.',
      'Enable Flash Attention. Memory utilization drops dramatically — from OOM to well within capacity. The savings grow quadratically with sequence length.',
    ],
    successExplanation:
      'Flash Attention is now the default in virtually all production inference systems. ' +
      'Beyond memory savings, it also accelerates prefill by eliminating O(N²) HBM reads/writes for attention scores — ' +
      'the attention computation stays in fast on-chip SRAM.\n\nAt 32K tokens, standard attention requires several GB ' +
      'just for the score matrix, making it impossible on most GPUs. Flash Attention reduces this to O(N) memory, ' +
      'enabling long-context inference that would otherwise be impossible. ' +
      'At long sequences this can significantly reduce prefill time.',
  },

  // ── 08 ── KV Cache Precision ────────────────────────────────────────
  {
    id: 'inference-beginner-08',
    mode: 'inference',
    difficulty: 'beginner',
    order: 7,
    title: 'KV Cache Precision',
    concept: 'Compressing per-request memory',
    learningObjectives: [
      'Understand KV cache quantization (FP8/INT8) halves the dynamic per-request memory',
      'Know this can double the number of concurrent requests a server handles',
      'Recognize KV cache precision is independent of weight precision — both can be optimized separately',
    ],
    briefing:
      'As you learned earlier, the KV cache grows with sequence length and batch size. ' +
      'For large batches or long contexts, the KV cache can dominate total memory usage. ' +
      'One technique is to store the cached keys and values at lower precision — {{fp8|FP8}} or INT8 — ' +
      'instead of BF16. Run Qwen 3 14B on a single H100 and keep memory utilization below 75%.',
    setup: {
      modelId: 'qwen3-14b',
      gpuId: 'h100-sxm',
      numGPUs: 1,
      batchSize: 48,
      inputSeqLen: 4096,
    },
    winningCriteria: [
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Inference succeeds',
      },
      {
        field: 'memoryUtilization',
        operator: '<',
        value: 0.75,
        label: 'Memory utilization < 75%',
      },
    ],
    expectedChanges: [
      { field: 'kvCachePrecision', check: 'changed', label: 'Changed KV cache precision' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'At BF16 KV precision with batch=48 and seq_len=4096, the KV cache consumes significant memory on top of the model weights — pushing utilization above 75%.',
      'Try setting KV cache precision to FP8 or INT8 (1 byte per value) — this halves the KV cache memory, allowing you to serve longer sequences or larger batches.',
      'KV cache quantization is especially impactful for GQA models (like Qwen 3) where the cache is already compact. The savings become even more important for MHA models or very long contexts.',
    ],
    successExplanation:
      'KV cache quantization is a complementary technique to weight quantization. ' +
      'While weight quantization reduces the fixed memory cost, KV cache quantization reduces the dynamic, ' +
      'per-request cost.\n\nIn production serving with hundreds of concurrent requests, the KV cache often ' +
      'exceeds weight memory by 5-10x. Reducing KV cache precision from BF16 to FP8 doubles the number ' +
      'of concurrent requests a server can handle. Modern inference engines like vLLM support FP8 KV cache ' +
      'with minimal quality impact. Models with GQA (like LLaMA 3.3 70B with only 8 KV heads vs 64 query ' +
      'heads) already have compact KV caches — quantization provides relative savings regardless of ' +
      'attention type.',
  },

  // ── 09 ── Latency vs Throughput ─────────────────────────────────────
  {
    id: 'inference-beginner-09',
    mode: 'inference',
    difficulty: 'beginner',
    order: 8,
    title: 'Latency vs Throughput',
    concept: 'Balancing per-token latency and aggregate throughput',
    learningObjectives: [
      'Understand the fundamental latency-throughput tradeoff in LLM serving',
      'Know that low batch = fast TPOT but low throughput; high batch = high throughput but slower TPOT',
      'Find the batch size sweet spot that satisfies both latency and throughput constraints',
      'Know paged attention enables 2-4x more concurrent requests via efficient KV memory allocation',
    ],
    briefing:
      'In production serving, you face a fundamental tradeoff. Low batch sizes give fast per-token latency (TPOT) ' +
      'but waste GPU bandwidth — the GPU reads all weights to produce few tokens. High batch sizes amortize ' +
      'weight reads across more tokens, improving throughput, but each token takes slightly longer.\n\n' +
      'You have Qwen 3 14B on a single H100. At batch=1, TPOT is fast but throughput is very low — the GPU wastes most of its bandwidth serving a single request. ' +
      'Your goal: achieve both TPOT under 15 ms AND throughput above 250 tokens per second. ' +
      'Find the batch size sweet spot that satisfies both constraints simultaneously.',
    setup: {
      modelId: 'qwen3-14b',
      gpuId: 'h100-sxm',
      numGPUs: 1,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Inference succeeds' },
      { field: 'latency.tpot', operator: '<', value: 15, label: 'TPOT < 15 ms' },
      { field: 'throughput.tokensPerSecond', operator: '>', value: 250, label: 'Throughput > 250 tok/s' },
    ],
    expectedChanges: [
      { field: 'batchSize', check: 'increased', label: 'Increased batch size' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'At batch=1, TPOT is fast but throughput is far below the 250 tok/s target. You need both constraints met simultaneously — look for a batch size that satisfies both.',
      'Increase batch size gradually. Each additional sequence amortizes the weight read cost, boosting throughput while TPOT increases slowly. There is a sweet spot where both constraints are satisfied. The Batch chart shows how TPOT and throughput change with batch size — use it to find the sweet spot.',
      'In production, {{paged-attention|paged attention}} eliminates KV cache fragmentation, allowing 2-4× more concurrent requests than naive allocation.',
    ],
    successExplanation:
      'The latency-throughput tradeoff is the central design decision in LLM serving. ' +
      'Interactive applications (chatbots) prioritize low TPOT for responsive streaming. ' +
      'Batch applications prioritize high throughput for cost efficiency.\n\n' +
      'At batch=1, the GPU reads all weights to produce one token — most compute capacity is wasted. ' +
      'At batch=4, the same weight read produces 4 tokens, nearly 4× throughput with only a small TPOT increase. ' +
      'In production, [PagedAttention (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180) eliminates KV cache fragmentation, allowing serving systems like vLLM ' +
      'to handle 2-4× more concurrent requests than naive allocation.',
  },

  // ── 10 ── Continuous Batching ───────────────────────────────────────
  {
    id: 'inference-beginner-10',
    mode: 'inference',
    difficulty: 'beginner',
    order: 9,
    title: 'Continuous Batching',
    concept: 'Dynamic request scheduling',
    learningObjectives: [
      'Understand static batching blocks short requests until the longest finishes',
      'Know continuous batching inserts new requests as slots open, maximizing GPU utilization',
      'Recognize continuous batching + paged attention as the foundation of production serving (vLLM, TGI)',
    ],
    briefing:
      'Static batching waits until all sequences in a batch finish generating before accepting new requests. ' +
      'If one sequence generates 10 tokens and another generates 500, the short request is blocked until ' +
      'the long one finishes. {{continuous-batching|Continuous batching}} (also called iteration-level batching) inserts new requests ' +
      'into the batch as soon as a slot opens, maximizing GPU utilization. ' +
      'Your goal: achieve throughput above 500 tokens per second with LLaMA 3.1 8B on an H100.',
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 'h100-sxm',
      numGPUs: 1,
    },
    winningCriteria: [
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Inference succeeds',
      },
      {
        field: 'throughput.tokensPerSecond',
        operator: '>',
        value: 500,
        label: 'Throughput > 500 tok/s',
      },
    ],
    expectedChanges: [
      { field: 'continuousBatching', check: 'enabled', label: 'Enabled continuous batching' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'Enable continuous batching in the configuration. It allows the serving engine to insert new requests into the batch without waiting for all current requests to complete.',
      'Continuous batching works best with a reasonable batch size — try 8, 16, or 32. The system can keep the batch full by replacing completed sequences immediately.',
      'Combine continuous batching with other optimizations you have learned: Flash Attention, weight quantization, or KV cache quantization. Each one contributes to higher overall throughput.',
    ],
    successExplanation:
      'Continuous batching processes new requests one at a time as slots free up, rather than waiting for an entire ' +
      'batch to complete. The key advantage: in static batching, prefilling all B requests together takes B× longer ' +
      'than a single request — this is dead time where no decode tokens are produced. CB replaces this batch-N prefill ' +
      'with single-request prefills, dramatically reducing the prefill bottleneck.\n\nThe throughput advantage grows with ' +
      'batch size and input sequence length. Production engines like vLLM and TGI combine CB with paged attention and ' +
      'chunked prefill for further gains the simulator does not model — those are framework-level optimizations that ' +
      'build on the same core scheduling principle.',
  },
];
