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
      'Know that halving weight precision (FP32 -> BF16) halves bytes read, nearly doubling throughput',
      'Identify the fixed (weights) vs dynamic (KV cache) memory components of inference',
    ],
    briefing:
      'You have a LLaMA 3.1 8B model on a powerful A100 GPU with 80 GB of memory. ' +
      'The model is loaded with FP32 weights — 4 bytes per parameter, consuming about 32 GB.\n\n' +
      'LLM inference decode is memory-bandwidth bound: the GPU must read all model weights from {{hbm|HBM}} for every ' +
      'single output token. At FP32, that means reading 32 GB per token. At BF16 (2 bytes/param), only 16 GB.\n\n' +
      'Your goal: achieve more than 60 tokens per second. The FP32 configuration cannot reach this — ' +
      'the A100\'s 2.0 TB/s memory bandwidth limits FP32 decode to about 36 tokens/sec. ' +
      'Switch to BF16 to halve the bytes read per token.',
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 'a100-80gb',
      numGPUs: 1,
      weightPrecision: 'fp32',
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Inference succeeds' },
      { field: 'throughput.tokensPerSecond', operator: '>', value: 60, label: 'Throughput > 60 tok/s' },
    ],
    hints: [
      'Run at FP32 first and observe the throughput — it is well below the 60 tok/s target. The GPU must read all 32 GB of FP32 weights for every single output token.',
      'Inference decode is memory-bandwidth bound: throughput is inversely proportional to weight bytes. Halving precision doubles throughput. Look at the Precision selector.',
      'Switch weight precision from FP32 to BF16. This halves the bytes read per decode step, roughly doubling throughput.',
    ],
    successExplanation:
      'Inference decode is fundamentally memory-bandwidth bound. The GPU reads all model weights from HBM for every ' +
      'output token, but performs relatively little computation per byte. Halving weight precision from FP32 to BF16 ' +
      'halves the bytes transferred, nearly doubling throughput.\n\nNotice the memory breakdown: model weights take a fixed ' +
      'amount of memory, while the KV cache grows with each generated token. In the coming tasks, you will learn to manage ' +
      'these resources as models get larger and GPUs get smaller.',
  },

  // ── 02 ── Weight Memory ─────────────────────────────────────────────
  {
    id: 'inference-beginner-02',
    mode: 'inference',
    difficulty: 'beginner',
    order: 1,
    title: 'Weight Memory',
    concept: 'Weight Quantization',
    learningObjectives: [
      'Calculate weight memory: params x bytes_per_param (8B x 2 = 16 GB in BF16)',
      'Use INT8/INT4 quantization to reduce weight memory 2-4x',
      'Know T4 is a Turing GPU without native BF16 support — quantization is essential for older hardware',
    ],
    briefing:
      'LLaMA 3.1 8B is a serious model — in {{bf16|BF16}}, its weights alone consume about 16 GB. ' +
      'You need to run it on a T4, which has only 16 GB of memory. ' +
      'The model barely fits at BF16, leaving almost no room for {{kv-cache|KV cache}} — ' +
      'use quantization to free memory for real workloads.',
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
      'Your task: reduce the batch size or use KV cache quantization to fit within the GPU\'s 80 GB. ' +
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
      '(GQA) — only 8 KV heads shared across 32 query heads, a 4× reduction compared to standard MHA. ' +
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
    concept: 'INT8/INT4 Quantization for Throughput',
    learningObjectives: [
      'Understand quantization improves throughput, not just memory (decode is bandwidth-bound)',
      'Know INT8/INT4 reduce bytes transferred per parameter, increasing effective bandwidth',
      'Recognize quantized models can run faster than full-precision ones due to bandwidth savings',
    ],
    briefing:
      'Qwen 3 14B has about 14 billion parameters. In BF16, the weights alone take ~28 GB. ' +
      'Decode is memory-bandwidth bound: the GPU reads all weights from HBM for every output token. ' +
      'Quantization reduces the bytes read per parameter — fewer bytes means faster token generation.\n\n' +
      'You have a single L40S GPU. Push throughput above 30 tokens per second using quantization.',
    setup: {
      modelId: 'qwen3-14b',
      gpuId: 'l40s',
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
    hints: [
      'At BF16 (2 bytes/param), the L40S must read 28 GB of weights per token. Quantization reduces the bytes read, directly improving throughput.',
      'Look at the Weight Precision selector. Lower precision means fewer bytes transferred per decode step.',
    ],
    successExplanation:
      'Quantization is not just about fitting models in memory — it directly improves throughput. ' +
      'Inference decode is memory-bandwidth-bound (the GPU spends more time loading weights ' +
      'than computing).\n\nBy using INT8 or INT4, you transfer fewer bytes per parameter, which means ' +
      'higher effective bandwidth utilization and faster token generation. This is why quantized models ' +
      'often run faster than full-precision ones, not just smaller.',
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
      'Serving a single request at a time wastes most of the GPU\'s compute capacity. ' +
      '{{decode|Decoding}} is memory-bandwidth-bound: the GPU loads all the weights to produce just one token. ' +
      'By processing multiple requests simultaneously (batching), you amortize the weight-loading cost ' +
      'across more useful work. Increase the batch size on LLaMA 3.1 8B to push throughput above 200 tokens per second.',
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 'a100-80gb',
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
        value: 200,
        label: 'Throughput > 200 tok/s',
      },
    ],
    hints: [
      'At batch size 1, the GPU loads all the model weights to produce a single token. Most of the compute units sit idle during this memory transfer.',
      'Try increasing the batch size to 4, 8, or 16. Each additional request in the batch gets "free" compute because the weights are already being loaded.',
      'Watch the memory utilization as you increase batch size — the KV cache grows linearly with batch size. There is a sweet spot where you maximize throughput before running out of memory. The Batch chart shows how throughput scales with batch size across precisions.',
    ],
    successExplanation:
      'Batching is the fundamental technique for efficient inference serving. A single decode step loads the ' +
      'full model weights regardless of batch size, so adding more sequences to the batch increases throughput ' +
      'almost linearly until you hit the compute ceiling or run out of memory for KV caches.\n\n' +
      'Production serving systems like vLLM and TGI are designed around this principle — they batch ' +
      'many concurrent requests to maximize GPU utilization.',
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
      'Inference has two distinct phases. Prefill processes the entire input prompt in parallel — ' +
      'it is compute-bound and determines the Time to First Token (TTFT). Decode generates output tokens ' +
      'one at a time — it is memory-bandwidth-bound and each step takes Time Per Output Token (TPOT). ' +
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
    concept: 'Flash Attention',
    learningObjectives: [
      'Understand FA eliminates O(N^2) attention memory, critical at 32K+ token context',
      'Know FA impact grows quadratically with sequence length — moderate at 2K, essential at 32K+',
      'Recognize FA as mandatory for long-context inference (OOM without it)',
    ],
    briefing:
      'Standard attention computes the full N × N {{attention-matrix|attention matrix}}, consuming O(N²) memory. ' +
      'At a 32K sequence length, this matrix alone consumes several GB — enough to overflow GPU memory.\n\n' +
      'You have LLaMA 3.1 8B on an A100-80GB with Flash Attention disabled and a 32K input sequence. ' +
      'The simulation will OOM because the attention score matrix is too large.\n\n' +
      'Flash Attention tiles the computation to avoid materializing the full matrix, ' +
      'reducing memory from O(N²) to O(N). Enable it to make this long-context configuration fit.',
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
    hints: [
      'Without Flash Attention at 32K sequence length, the attention score matrix (batch × heads × 32768² × bytes) consumes several GB. The configuration OOMs.',
      'Flash Attention computes attention in {{sram|SRAM}} tiles, never materializing the full N × N matrix in HBM. It is mathematically identical — only the memory pattern changes.',
      'Enable Flash Attention. Memory utilization drops dramatically — from OOM to well within capacity. The savings grow quadratically with sequence length.',
    ],
    successExplanation:
      'Flash Attention is now the default in virtually all production inference systems. ' +
      'Beyond memory savings, it is also faster because it reduces HBM reads/writes — the attention ' +
      'computation stays in fast on-chip SRAM.\n\nAt 32K tokens, standard attention requires several GB ' +
      'just for the score matrix, making it impossible on most GPUs. Flash Attention reduces this to O(N) memory, ' +
      'enabling long-context inference that would otherwise be impossible. ' +
      'The A100 and newer GPUs support it natively through their tensor core architecture.',
  },

  // ── 08 ── KV Cache Precision ────────────────────────────────────────
  {
    id: 'inference-beginner-08',
    mode: 'inference',
    difficulty: 'beginner',
    order: 7,
    title: 'KV Cache Precision',
    concept: 'KV Cache Quantization',
    learningObjectives: [
      'Understand KV cache quantization (FP8/INT8) halves the dynamic per-request memory',
      'Know this can double the number of concurrent requests a server handles',
      'Recognize KV cache precision is independent of weight precision — both can be optimized separately',
    ],
    briefing:
      'As you learned earlier, the KV cache grows with sequence length and batch size. ' +
      'For large batches or long contexts, the KV cache can dominate total memory usage. ' +
      'One technique is to store the cached keys and values at lower precision — {{fp8|FP8}} or INT8 — ' +
      'instead of BF16. Run Qwen 3 14B on a single H100 and use KV cache quantization to keep ' +
      'memory utilization below 75%.',
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
    hints: [
      'At BF16 KV precision with batch=48 and seq_len=4096, the KV cache consumes significant memory on top of the model weights — pushing utilization above 75%.',
      'Try setting KV cache precision to FP8 or INT8 (1 byte per value) — this halves the KV cache memory, allowing you to serve longer sequences or larger batches.',
      'KV cache quantization is especially impactful for {{gqa|GQA}} models (like Qwen 3) where the cache is already compact. The savings become even more important for MHA models or very long contexts.',
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
      'You have Qwen 3 14B on a single H100. At batch=1, TPOT is about 11 ms but throughput is only ~92 tok/s. ' +
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
    hints: [
      'At batch=1, TPOT is fast but throughput is far below the 250 tok/s target. You need both constraints met simultaneously — look for a batch size that satisfies both.',
      'Increase batch size gradually. Each additional sequence amortizes the weight read cost, boosting throughput while TPOT increases slowly. There is a sweet spot where both constraints are satisfied.',
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
    concept: 'Continuous (Iteration-Level) Batching',
    learningObjectives: [
      'Understand static batching blocks short requests until the longest finishes',
      'Know continuous batching inserts new requests as slots open, maximizing GPU utilization',
      'Recognize continuous batching + paged attention as the foundation of production serving (vLLM, TGI)',
    ],
    briefing:
      'Static batching waits until all sequences in a batch finish generating before accepting new requests. ' +
      'If one sequence generates 10 tokens and another generates 500, the short request is blocked until ' +
      'the long one finishes. Continuous batching (also called iteration-level batching) inserts new requests ' +
      'into the batch as soon as a slot opens, maximizing GPU utilization. ' +
      'Enable continuous batching for LLaMA 3.1 8B on an H100 and achieve throughput above 500 tokens per second.',
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
    hints: [
      'Enable continuous batching in the configuration. It allows the serving engine to insert new requests into the batch without waiting for all current requests to complete.',
      'Continuous batching works best with a reasonable batch size — try 8, 16, or 32. The system can keep the batch full by replacing completed sequences immediately.',
      'Combine continuous batching with other optimizations you have learned: Flash Attention, weight quantization, or KV cache quantization. Each one contributes to higher overall throughput.',
    ],
    successExplanation:
      'Continuous batching is the scheduling innovation that made modern LLM serving practical. ' +
      'Without it, GPU utilization drops as sequences finish at different times and batch slots sit empty. ' +
      'With continuous batching, the GPU stays fully utilized by immediately filling empty slots with ' +
      'waiting requests.\n\nCombined with paged attention, this enables serving systems like vLLM, TGI, ' +
      'and TensorRT-LLM to achieve 10-20x higher throughput than naive static batching. ' +
      'These two techniques — continuous batching and paged attention — are the foundation of ' +
      'every production LLM serving stack.',
  },
];
