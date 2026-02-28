import type { GameTask } from '../types.ts';
// Glossary annotations: {{termId}} or {{termId|display text}} for tooltip terms.
// [text](url) for paper links in successExplanation.
// Annotate FIRST OCCURRENCE per difficulty tier only. Check earlier tasks before adding.

export const INFERENCE_INTERMEDIATE_TASKS: GameTask[] = [
  // ── 1. TP for Large Models ──────────────────────────────────────────
  {
    id: 'inference-intermediate-01',
    mode: 'inference',
    difficulty: 'intermediate',
    order: 0,
    title: 'Tensor Parallelism for Large Models',
    concept: 'Multi-GPU weight distribution',
    learningObjectives: [
      'Use TP to serve models exceeding single-GPU memory (70B needs ~140 GB in BF16)',
      'Know TP splits weight matrices so each GPU stores and computes 1/TP of weights',
      'Understand TP requires a fast intra-node interconnect (e.g. NVLink) for efficient AllReduce at every transformer layer',
    ],
    briefing:
      'You need to serve LLaMA 3.3 70B for real-time inference. At 140 GB in BF16, the model does not fit on a single 80 GB H100. ' +
      'Tensor parallelism (TP) splits each layer\'s weight matrices across multiple GPUs, so each GPU only stores and computes a fraction of the weights. ' +
      'You have 4 H100 GPUs connected via {{nvlink|NVLink}}. Configure TP so the model fits in memory and runs successfully.',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'h100-sxm',
      numGPUs: 4,
      gpusPerNode: 4,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Inference runs successfully' },
      { field: 'memoryUtilization', operator: '<', value: 1.0, label: 'Model fits in GPU memory' },
    ],
    expectedChanges: [
      { field: 'tensorParallel', check: 'increased', label: 'Increased tensor parallelism' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'A 70B parameter model in BF16 needs roughly 140 GB just for weights. A single 80 GB GPU cannot hold it.',
      'Set Tensor Parallel to split weights across GPUs. TP=2 halves per-GPU weight memory, TP=4 quarters it.',
      'With 4 GPUs, try TP=4. Each GPU stores a quarter of the weights, well within 80 GB.',
    ],
    successExplanation:
      'Tensor parallelism distributes weight matrices across GPUs so each device only stores `1/TP` of the parameters. ' +
      'For a 70B model in BF16 (~140 GB), `TP=4` reduces per-GPU weights to ~35 GB, comfortably fitting within 80 GB.\n\n' +
      'TP requires high-bandwidth interconnects like NVLink because GPUs must exchange activations at every layer via AllReduce. ' +
      'This is why TP is typically limited to GPUs within a single node.',
  },

  // ── 2. TP Degree vs Latency ─────────────────────────────────────────
  {
    id: 'inference-intermediate-02',
    mode: 'inference',
    difficulty: 'intermediate',
    order: 1,
    title: 'TP Degree vs Latency Tradeoffs',
    concept: 'How tensor parallel degree affects per-token decode latency',
    learningObjectives: [
      'Know higher TP reduces TPOT by reducing per-GPU weight reads',
      'Understand diminishing returns: each TP doubling adds more AllReduce overhead',
      'Recognize the resource waste tradeoff: TP=8 is fast but dedicates 8 GPUs to one instance',
    ],
    briefing:
      'You have 8 H100 GPUs and want to minimize the time per output token ({{tpot|TPOT}}) for LLaMA 3.3 70B. ' +
      'More TP means each GPU reads fewer weight bytes per decode step, reducing memory-bandwidth latency. ' +
      'But each additional TP rank adds AllReduce communication overhead. ' +
      'Find the TP setting that achieves TPOT under 8 ms. Experiment with TP=2, TP=4, and TP=8 to see the tradeoff.',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'h100-sxm',
      numGPUs: 8,
      gpusPerNode: 8,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Inference runs successfully' },
      { field: 'latency.tpot', operator: '<', value: 8.0, label: 'TPOT < 8 ms' },
    ],
    expectedChanges: [
      { field: 'tensorParallel', check: 'increased', label: 'Increased tensor parallelism' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'Decode is memory-bandwidth bound: TPOT is proportional to weight bytes read per GPU. Doubling TP halves the weight reads.',
      'Higher TP reduces TPOT by splitting weight reads across GPUs, but each doubling adds more AllReduce overhead. Try different TP values and observe the diminishing returns.',
      'Higher TP degrees will meet the target. The diminishing returns at high TP are because communication overhead grows. The Pareto frontier (TPOT tab) shows cost-latency tradeoffs across configurations.',
    ],
    successExplanation:
      'Decode latency is dominated by reading model weights from HBM. Higher TP reduces per-GPU weight reads proportionally — ' +
      'each doubling of TP halves the memory-bandwidth time per decode step.\n\nThe AllReduce overhead (exchanging activations at ' +
      'every layer) grows with TP degree, but a high-bandwidth intra-node interconnect keeps the net effect as a large latency reduction. ' +
      'At some point, communication starts to dominate and returns diminish rapidly.',
  },

  // ── 3. Serving LLaMA 3.1 405B ──────────────────────────────────────
  {
    id: 'inference-intermediate-03',
    mode: 'inference',
    difficulty: 'intermediate',
    order: 2,
    title: 'Serving a 405B Parameter Model',
    concept: 'Memory constraints at extreme scale',
    learningObjectives: [
      'Combine TP and quantization to fit models with 800+ GB weights',
      'Know FP8 has native H100 support (Transformer Engine) with ~0.5% quality loss',
      'Calculate per-GPU memory: total_weight_bytes / TP to verify fit',
    ],
    briefing:
      'Your team needs to deploy LLaMA 3.1 405B for inference. In BF16, the model weighs over 800 GB — far too large ' +
      'even for 8 H100 GPUs with TP=8 (640 GB total HBM). ' +
      'You need to combine tensor parallelism with weight quantization to fit this model. ' +
      'Combine tensor parallelism with weight quantization to make it fit.',
    setup: {
      modelId: 'llama3-405b',
      gpuId: 'h100-sxm',
      numGPUs: 8,
      gpusPerNode: 8,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Inference runs successfully' },
    ],
    expectedChanges: [
      { field: 'weightPrecision', check: 'changed', label: 'Changed weight precision' },
      { field: 'tensorParallel', check: 'increased', label: 'Increased tensor parallelism' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'At BF16 (2 bytes/param), 405B parameters require far more memory than 8 GPUs can provide even with TP=8. Calculate the per-GPU weight memory to see why.',
      'FP8 or INT8 cuts weight memory in half (1 byte/param). Check whether that fits per GPU with TP=8.',
      'Set Tensor Parallel to 8 and Weight Precision to FP8 or INT8. H100 GPUs have native FP8 support via {{transformer-engine|Transformer Engine}}, so FP8 has minimal overhead.',
    ],
    successExplanation:
      'Very large models require combining parallelism with quantization. At BF16, 405B parameters need ~810 GB, which exceeds 8 x 80 GB = 640 GB. ' +
      'Switching to FP8 (1 byte/param) reduces total weight memory to ~405 GB, or ~51 GB per GPU with `TP=8`.\n\n' +
      'H100 GPUs support FP8 natively in their Tensor Cores, so the quality loss is minimal (~0.5%) and there is no significant ' +
      'dequantization overhead. INT4 would reduce memory even further but at greater quality cost.',
  },

  // ── 4. When TP Wastes GPUs ─────────────────────────────────────────
  {
    id: 'inference-intermediate-04',
    mode: 'inference',
    difficulty: 'intermediate',
    order: 3,
    title: 'When TP Wastes GPUs',
    concept: 'Over-sharding and replicas vs tensor parallelism',
    learningObjectives: [
      'Know that over-sharding with TP makes AllReduce overhead dominate when per-GPU weights are small',
      'Understand replicas provide linear throughput scaling with zero communication overhead',
      'Recognize the tradeoff: TP reduces latency per request, replicas maximize aggregate throughput',
    ],
    briefing:
      'You have LLaMA 3.1 8B on 8 H100 GPUs with TP=8 and batch=1. Each GPU holds only 1/8 of the model — ' +
      'about 2 GB of weights. The per-GPU weight read is nearly instant, but the AllReduce ' +
      'across all 8 ranks at every layer adds significant overhead.\n\n' +
      'When a model easily fits on fewer GPUs, maximum TP wastes most of the GPU time on communication. ' +
      'Find a configuration that achieves over 1000 tokens per second aggregate throughput.',
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 'h100-sxm',
      numGPUs: 8,
      gpusPerNode: 8,
      tensorParallel: 8,
      batchSize: 1,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Inference runs successfully' },
      { field: 'throughput.tokensPerSecond', operator: '>', value: 1000, label: 'Throughput > 1000 tok/s' },
    ],
    expectedChanges: [
      { field: 'tensorParallel', check: 'decreased', label: 'Decreased tensor parallelism' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'LLaMA 8B in BF16 is ~16 GB — it fits easily on a single 80 GB GPU. With TP=8, each GPU holds only ~2 GB of weights. The weight read is nearly instant, but 8-way AllReduce at every layer is not.',
      'Reducing TP creates independent replicas: TP=4 gives 2 replicas, TP=2 gives 4, TP=1 gives 8. Each replica runs without any communication overhead. Total throughput = per-replica throughput × number of replicas.',
      'Try different TP values and watch aggregate throughput. The sweet spot balances per-replica speed (some TP helps) against replica count (more replicas = more parallelism).',
    ],
    successExplanation:
      'With TP=8, each GPU reads only ~2 GB of weights per decode step — nearly instant on high-bandwidth HBM. But the AllReduce ' +
      'communication after every layer adds overhead that dominates the tiny compute time. The result: 8 GPUs produce ' +
      'fewer tokens than they could as independent replicas.\n\n' +
      'Reducing TP frees GPUs to run independent model replicas. Each replica processes its own requests with zero ' +
      'communication overhead, and total throughput scales linearly with replica count. ' +
      'This is the fundamental distinction between latency-oriented serving (max TP, lowest TPOT per request) ' +
      'and throughput-oriented serving (min viable TP, max replicas for aggregate tokens/sec).',
  },

  // ── 5. KV Cache at Scale ───────────────────────────────────────────
  {
    id: 'inference-intermediate-05',
    mode: 'inference',
    difficulty: 'intermediate',
    order: 4,
    title: 'KV Cache Memory at Long Contexts',
    concept: 'Dynamic memory pressure at scale',
    learningObjectives: [
      'Understand KV cache at long contexts with high batch can exceed available GPU memory',
      'Know KV cache quantization (FP8/INT8) halves dynamic per-request memory',
      'Recognize KV cache precision as an independent lever from weight precision',
      'Understand batch amortization: more concurrent sequences per decode step = higher throughput',
      'Know that freeing KV memory through quantization unlocks higher batch sizes',
    ],
    briefing:
      'You are building a long-context document analysis service with LLaMA 3.3 70B on 4 H100 GPUs. ' +
      'The current config uses TP=4, batch=32, and 16K input sequences — but it OOMs. The {{kv-cache|KV cache}} ' +
      'for 32 concurrent sequences at 16,640 tokens each is enormous, ' +
      'pushing total memory beyond 80 GB per GPU.\n\n' +
      'Your task: achieve over 135 tokens/sec throughput while fitting in memory. ' +
      'Simply reducing batch size loses too much throughput — you need to manage the KV cache ' +
      'AND maintain a high batch size.',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'h100-sxm',
      numGPUs: 4,
      gpusPerNode: 4,
      inputSeqLen: 16384,
      outputSeqLen: 256,
      batchSize: 32,
      tensorParallel: 4,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Inference runs successfully' },
      { field: 'throughput.tokensPerSecond', operator: '>', value: 135, label: 'Throughput > 135 tok/s' },
    ],
    expectedChanges: [
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'The model weights fit at TP=4, but the KV cache for 32 sequences at 16K context pushes memory over the limit. Reducing batch size fits but throughput drops below the target.',
      'KV cache precision is independent of weight precision — quantizing KV to FP8 halves the dynamic memory. The freed memory lets you keep the batch size high.',
      'This is a two-step optimization: (1) free memory via KV cache quantization, then (2) exploit that headroom to maintain or increase batch size for throughput.',
    ],
    successExplanation:
      'KV cache memory = `numLayers × numKVHeads × headDim × seqLen × batchSize × 2 (K and V) × bytesPerElement`. ' +
      'For LLaMA 3.3 70B at 16K context, each request\'s KV cache is several GB. Batching multiplies this linearly.\n\n' +
      'Simply reducing batch size fits in memory but sacrifices throughput — each decode step reads all model weights ' +
      'regardless of batch size, so fewer sequences per step means less work per weight read.\n\n' +
      'KV cache quantization (FP8 or INT8) halves the bytes per element, directly halving the dynamic memory. ' +
      'This frees enough headroom to keep batch=32 (or higher), maintaining the throughput benefit of batch amortization. ' +
      'The pattern — free memory, then exploit it for batch — is standard in production serving.',
  },

  // ── 6. Batch Throughput ────────────────────────────────────────────
  {
    id: 'inference-intermediate-06',
    mode: 'inference',
    difficulty: 'intermediate',
    order: 5,
    title: 'Maximizing Batch Throughput',
    concept: 'Batching requests to amortize weight reads across sequences',
    learningObjectives: [
      'Understand weight reads are constant per step regardless of batch size',
      'Know throughput becomes sub-linear at very large batches (KV cache reads grow with batch)',
      'Target throughput-optimized configs for high-QPS serving workloads',
      'Recognize the throughput-latency tension: excessive batching degrades per-request TPOT',
    ],
    briefing:
      'Your inference cluster needs to handle high request volume. With LLaMA 3.3 70B on 8 {{a100|A100}}-80GB GPUs ' +
      'and TP=8 already configured, a single request generates relatively few tokens per second. But the {{decode}} phase is ' +
      'memory-bandwidth bound — the GPU reads all model weights for every single token, regardless of batch size. ' +
      'By batching multiple requests, you amortize the weight read cost across sequences.\n\n' +
      'Your SLA requires TPOT under 15 ms per token AND throughput above 1200 tok/s. ' +
      'Blindly maximizing batch size will violate the latency constraint.',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'a100-80gb',
      numGPUs: 8,
      gpusPerNode: 8,
      tensorParallel: 8,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Inference runs successfully' },
      { field: 'throughput.tokensPerSecond', operator: '>', value: 1200, label: 'Throughput > 1200 tok/s' },
      { field: 'latency.tpot', operator: '<', value: 15, label: 'TPOT < 15 ms' },
    ],
    expectedChanges: [
      { field: 'batchSize', check: 'increased', label: 'Increased batch size' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'At batch=1, each decode step reads all weights to produce one token. At batch=N, the same weight read produces N tokens. But very high batch sizes push TPOT above the SLA.',
      'There is a productive range of batch sizes that satisfy both constraints. Try moderate values and watch both throughput and TPOT — the sweet spot is where throughput exceeds the target while TPOT remains under the ceiling.',
    ],
    successExplanation:
      'Decode is memory-bandwidth bound: the GPU must read all model weights from HBM every decode step. ' +
      'At batch=1, one weight read produces one token. At batch=N, the same weight read produces N tokens — ' +
      'throughput scales nearly linearly in the memory-bound regime.\n\n' +
      'But at very high batch sizes, KV cache reads per step grow large enough to compete with weight reads for bandwidth. ' +
      'TPOT rises because each decode step takes longer, even though throughput still increases. ' +
      'The productive range is where throughput meets the target AND TPOT stays under the SLA — ' +
      'this is the fundamental tension in production serving between maximizing throughput and honoring latency guarantees.',
  },

  // ── 7. Latency Optimization ────────────────────────────────────────
  {
    id: 'inference-intermediate-07',
    mode: 'inference',
    difficulty: 'intermediate',
    order: 6,
    title: 'Minimizing Per-Token Latency',
    concept: 'Optimizing TPOT and cost on consumer GPUs',
    learningObjectives: [
      'Know TPOT ~ weight_bytes / (memory_bandwidth x efficiency) in the bandwidth-bound regime',
      'Identify three TPOT levers: TP (split weights), quantization (fewer bytes), small batch (less KV reads)',
      'Understand cost per token = (GPU_cost × num_GPUs) / throughput — batching is the cost lever',
      'Recognize the three-way tension: latency (low batch, high TP), throughput (high batch), and cost (amortize GPU time)',
    ],
    briefing:
      'You are building a real-time coding assistant using 4 {{rtx-4090|RTX 4090}} GPUs connected via {{pcie|PCIe}}. ' +
      'LLaMA 3.3 70B at BF16 requires ~140 GB — far more than 4 × 24 GB = 96 GB. ' +
      'You need aggressive quantization (INT4: ~35 GB total) to fit the model across 4 GPUs with TP=4.\n\n' +
      'You have two constraints: TPOT under 25 ms for responsiveness, AND cost under $4 per million tokens ' +
      'for economic viability. Meeting the latency target alone is not enough — at batch=1, the GPUs ' +
      'spend most of their time idle between requests, making per-token cost very high.',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'rtx-4090',
      numGPUs: 4,
      gpusPerNode: 4,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Inference runs successfully' },
      { field: 'latency.tpot', operator: '<', value: 25.0, label: 'TPOT < 25 ms' },
      { field: 'costPerMillionTokens', operator: '<', value: 4.0, label: 'Cost < $4/M tokens' },
    ],
    expectedChanges: [
      { field: 'weightPrecision', check: 'changed', label: 'Changed weight precision' },
      { field: 'tensorParallel', check: 'increased', label: 'Increased tensor parallelism' },
      { field: 'batchSize', check: 'increased', label: 'Increased batch size' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'At BF16, 70B requires ~140 GB — it does not fit on 4 × 24 GB GPUs even with TP=4. INT4 (0.5 bytes/param) dramatically reduces weight memory. The RTX 4090 has no FP8 support — INT4 or INT8 are the quantization options.',
      'INT4 + TP=4 meets the latency target, but at batch=1 the cost is very high. Each decode step reads all weights to produce only one token — the GPU time per token is expensive.',
      'Increasing batch size amortizes the GPU cost across multiple sequences per step. A modest increase keeps TPOT well under the ceiling while dramatically reducing cost per token.',
    ],
    successExplanation:
      'Running 70B models on 4 RTX 4090 GPUs is one of the most popular hobbyist and startup setups. ' +
      'The key constraints are: (1) 24 GB per GPU requires INT4 quantization to fit, ' +
      '(2) PCIe interconnect adds significant TP AllReduce overhead, ' +
      'and (3) the RTX 4090 lacks FP8 support, so INT4/INT8 are the only quantization options.\n\n' +
      'Meeting the latency target alone (INT4 + TP=4 + batch=1) is straightforward, but cost per token is high ' +
      'because each decode step produces only one token while paying for 4 GPUs. Increasing batch amortizes ' +
      'the GPU cost across multiple sequences — each additional request in the batch adds minimal TPOT ' +
      'overhead but proportionally reduces cost per token. This three-way optimization (latency + cost + memory) ' +
      'is the core engineering problem for consumer GPU serving.',
  },

  // ── 8. GPU Selection ───────────────────────────────────────────────
  {
    id: 'inference-intermediate-08',
    mode: 'inference',
    difficulty: 'intermediate',
    order: 7,
    title: 'Choosing the Right GPU for Inference',
    concept: 'Memory bandwidth as the key GPU metric for decode performance',
    learningObjectives: [
      'Know memory bandwidth is the critical GPU spec for decode, not compute TFLOPS',
      'Compare GPUs by memory bandwidth — higher bandwidth means faster decode',
      'Understand arithmetic intensity for decode is far below the GPU roofline — bandwidth is the bottleneck',
    ],
    briefing:
      'You want to serve Qwen 3 14B on a single GPU and need at least 120 tokens/sec throughput. ' +
      'You start on an {{a10g|A10G}} — the most popular AWS inference GPU with 24 GB of memory and ' +
      'relatively low memory bandwidth. Even with quantization, decode throughput is limited.\n\n' +
      'For inference, the GPU spec that matters most during decode is not compute (TFLOPS) but memory bandwidth — ' +
      'because decode reads all weights every step and does minimal computation per byte. ' +
      'Find a GPU and configuration that exceeds 120 tokens per second.',
    setup: {
      modelId: 'qwen3-14b',
      gpuId: 'a10g',
      numGPUs: 1,
      gpusPerNode: 1,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Inference runs successfully' },
      { field: 'throughput.tokensPerSecond', operator: '>', value: 120, label: 'Throughput > 120 tok/s' },
    ],
    expectedChanges: [
      { field: 'gpuId', check: 'changed', label: 'Changed GPU type' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
    ],
    hints: [
      'The A10G has relatively low memory bandwidth. Even with aggressive quantization, single-request decode throughput is modest. Try other GPUs with higher bandwidth.',
      'Compare memory bandwidth across GPUs — decode throughput is roughly proportional to bandwidth. Check the GPU specs in the sidebar to find higher-bandwidth options.',
      'A high-bandwidth GPU with quantization (INT8 or FP8) can easily exceed 120 tok/s. The key insight: bandwidth, not TFLOPS, determines decode speed.',
    ],
    successExplanation:
      'The A10G is affordable and popular on AWS, but its modest bandwidth caps decode throughput. ' +
      'For single-request decode: throughput ≈ `1 / (weight_bytes / bandwidth)`. ' +
      'A higher-bandwidth GPU reads the same weight bytes much faster, producing proportionally more tokens per second. ' +
      'The throughput difference between GPUs is roughly proportional to their bandwidth ratio.\n\nBoth GPUs have significant compute capacity, ' +
      'but compute is almost irrelevant for batch=1 decode — the arithmetic intensity (FLOPs per byte read) is far below ' +
      'the GPU\'s ridge point. Memory bandwidth is the bottleneck, making it the single most important GPU spec for inference.',
  },

  // ── 9. Speculative Decoding ────────────────────────────────────────
  {
    id: 'inference-intermediate-09',
    mode: 'inference',
    difficulty: 'intermediate',
    order: 8,
    title: 'Speculative Decoding',
    concept: 'Accelerating autoregressive generation',
    learningObjectives: [
      'Understand draft-verify paradigm: small model proposes K tokens, large model verifies in one forward pass',
      'Know verification of K tokens costs ~1 target forward pass (same compute, K+1 positions)',
      'Understand acceptance rate determines effective speedup; same-family drafts yield higher acceptance',
    ],
    briefing:
      'Speculative decoding uses a small, fast "draft" model to generate K candidate tokens, then verifies them ' +
      'in parallel with the large "target" model. If the draft model\'s predictions match the target\'s distribution, ' +
      'you generate multiple tokens per target forward pass instead of one. ' +
      'Enable speculative decoding for LLaMA 3.3 70B on 4 H100 GPUs, using a smaller LLaMA model as the draft. ' +
      'Observe the speedup in effective TPOT.',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'h100-sxm',
      numGPUs: 4,
      gpusPerNode: 4,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Inference runs successfully' },
      { field: 'latency.tpot', operator: '<', value: 9, label: 'TPOT < 9 ms' },
    ],
    expectedChanges: [
      { field: 'speculativeDecoding', check: 'enabled', label: 'Enabled speculative decoding' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'First set TP so the 70B model fits in memory. Then enable Speculative Decoding and select a draft model.',
      'LLaMA 3.1 8B is a good draft model — same family, architecturally compatible with the 70B target.',
      'The speedup depends on the acceptance rate: how often the draft model\'s predictions match the target. Typical rates are 60-80% for same-family models.',
    ],
    successExplanation:
      'Speculative decoding exploits the fact that verification (checking K draft tokens) costs about the same as generating ' +
      'one token — both require a single forward pass through the target model. If the draft model proposes K=4 tokens ' +
      'and 3 are accepted, you get 3 tokens for the cost of 1 target forward pass plus the cheap draft overhead.\n\n' +
      'The effective TPOT = (draft_time + verify_time) / accepted_tokens. With ~70% acceptance rate and K=4, ' +
      'this yields roughly 1.8x speedup. The technique works best when the draft model is much smaller than the target ' +
      'and the two models share similar training distributions. ' +
      'See [Speculative Decoding (Leviathan et al., 2022)](https://arxiv.org/abs/2211.17192) for the original analysis.',
  },

  // ── 10. FP8 Inference ──────────────────────────────────────────────
  {
    id: 'inference-intermediate-10',
    mode: 'inference',
    difficulty: 'intermediate',
    order: 9,
    title: 'FP8 Quantized Inference',
    concept: 'Hardware-native reduced precision',
    learningObjectives: [
      'Know FP8 halves bandwidth requirement, yielding ~2x decode throughput on H100',
      'Understand FP8 dequantization is handled natively by Transformer Engine on Hopper GPUs',
      'Know quality impact is ~0.5% for most tasks — acceptable for production serving',
    ],
    briefing:
      'H100 GPUs support FP8 natively through their Transformer Engine. By storing model weights in FP8 (1 byte) ' +
      'instead of BF16 (2 bytes), you halve the memory bandwidth required during decode — and bandwidth is the bottleneck. ' +
      'Serve LLaMA 3.3 70B on 4 H100 GPUs and switch to FP8 weight precision. ' +
      'Target at least 120 tokens/sec, which BF16 cannot achieve in this configuration.',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'h100-sxm',
      numGPUs: 4,
      gpusPerNode: 4,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Inference runs successfully' },
      { field: 'throughput.tokensPerSecond', operator: '>', value: 120, label: 'Throughput > 120 tok/s' },
    ],
    expectedChanges: [
      { field: 'weightPrecision', check: 'changed', label: 'Changed weight precision' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'At BF16, throughput is below the target even with TP. The bottleneck is memory bandwidth: each GPU reads a large share of the weights every decode step.',
      'Switch Weight Precision to FP8. This halves per-GPU weight memory, nearly doubling decode throughput.',
      'FP8 on H100 has near-zero dequantization overhead because the Transformer Engine handles FP8 natively in hardware. Quality loss is under 0.5%.',
    ],
    successExplanation:
      'In the memory-bound decode regime, throughput is inversely proportional to bytes read per step. ' +
      'FP8 stores each weight in 1 byte instead of 2, halving the data that must be streamed from HBM. ' +
      'On H100, FP8 dequantization is handled by the Transformer Engine with minimal overhead (~5%), ' +
      'so the speedup is close to the theoretical 2x.\n\nThe quality impact is small: FP8 (E4M3) preserves 99.5% of model quality ' +
      'for most tasks. This makes FP8 the standard production precision for inference on Hopper GPUs. ' +
      'For older GPUs without native FP8 (like A100), the dequantization overhead is larger, reducing the benefit.',
  },
];
