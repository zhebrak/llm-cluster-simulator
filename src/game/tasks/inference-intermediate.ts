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
    concept: 'Splitting model weights across GPUs with tensor parallelism',
    learningObjectives: [
      'Use TP to serve models exceeding single-GPU memory (70B needs ~140 GB in BF16)',
      'Know TP splits weight matrices so each GPU stores and computes 1/TP of weights',
      'Understand TP requires NVLink for efficient AllReduce at every transformer layer',
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
    hints: [
      'Decode is memory-bandwidth bound: TPOT is proportional to weight bytes read per GPU. Doubling TP halves the weight reads.',
      'Higher TP reduces TPOT by splitting weight reads across GPUs, but each doubling adds more AllReduce overhead. Try different TP values and observe the diminishing returns.',
      'Higher TP degrees will meet the target. The diminishing returns at high TP are because communication overhead grows. The Pareto frontier (TPOT tab) shows cost-latency tradeoffs across configurations.',
    ],
    successExplanation:
      'Decode latency is dominated by reading model weights from HBM. With `TP=8`, each GPU reads only 1/8 of the weights per step, ' +
      'cutting the memory-bandwidth time roughly 8x compared to a single GPU.\n\nThe AllReduce overhead (exchanging activations at ' +
      'every layer) grows with TP degree, but NVLink is fast enough that the net effect is still a large latency reduction. ' +
      'Beyond `TP=8`, communication starts to dominate and returns diminish rapidly.',
  },

  // ── 3. Serving LLaMA 3.1 405B ──────────────────────────────────────
  {
    id: 'inference-intermediate-03',
    mode: 'inference',
    difficulty: 'intermediate',
    order: 2,
    title: 'Serving a 405B Parameter Model',
    concept: 'Fitting very large models using quantization and tensor parallelism',
    learningObjectives: [
      'Combine TP and quantization to fit models with 800+ GB weights',
      'Know FP8 has native H100 support (Transformer Engine) with ~0.5% quality loss',
      'Calculate per-GPU memory: total_weight_bytes / TP to verify fit',
    ],
    briefing:
      'Your team needs to deploy LLaMA 3.1 405B for inference. In BF16, the model weighs over 800 GB — far too large ' +
      'even for 8 H100 GPUs with TP=8 (640 GB total HBM). ' +
      'You need to combine tensor parallelism with weight quantization to fit this model. ' +
      'Explore {{fp8|FP8}}, INT8, or INT4 weight precision along with TP=8 to make it work.',
    setup: {
      modelId: 'llama3-405b',
      gpuId: 'h100-sxm',
      numGPUs: 8,
      gpusPerNode: 8,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Inference runs successfully' },
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

  // ── 4. MoE Inference ───────────────────────────────────────────────
  {
    id: 'inference-intermediate-04',
    mode: 'inference',
    difficulty: 'intermediate',
    order: 3,
    title: 'Mixture-of-Experts Inference',
    concept: 'Active vs total parameters in MoE models',
    learningObjectives: [
      'Understand MoE memory = total params (all experts resident), compute = active params per token',
      'Know MoE is more memory-bandwidth-bound than a dense model with the same active size',
      'Recognize the memory-compute decoupling as the key advantage of MoE architecture',
    ],
    briefing:
      'Mixtral 8x7B has 47B total parameters but only activates ~13B per token through its top-2 {{router|routing}} across 8 experts. ' +
      'Despite the total parameter count, {{moe|MoE}} inference has a unique property: compute cost scales with {{active-params|active parameters}}, ' +
      'but memory must hold all expert weights. ' +
      'You have 2 H100 GPUs. Configure tensor parallelism so the full model fits in memory and runs successfully.',
    setup: {
      modelId: 'mixtral-8x7b',
      gpuId: 'h100-sxm',
      numGPUs: 2,
      gpusPerNode: 2,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Inference runs successfully' },
    ],
    hints: [
      'Mixtral 8x7B has 47B total parameters. At BF16, that is ~94 GB — too large for one 80 GB GPU.',
      'With TP=2, each GPU stores half the total weights. Check whether that fits within 80 GB with room for KV cache and activations.',
      'Although only ~13B parameters are active per token (lower compute), all 47B must reside in GPU memory because any expert might be selected.',
    ],
    successExplanation:
      'MoE models decouple compute from memory: the router selects a small subset of experts per token (active params), ' +
      'but all expert weights must be stored in GPU memory because routing is input-dependent. Mixtral has 47B total / ~13B active params.\n\n' +
      'Memory sizing uses total params (`47B × 2 bytes` = ~94 GB), while FLOPs use active params (~13B). ' +
      'This means MoE models offer better compute-per-parameter than dense models, but memory requirements still scale with total size.\n\n' +
      'MoE models also interact with attention architecture choices. Mixtral uses GQA (8 KV heads vs 32 ' +
      'query heads), keeping KV cache compact relative to its 47B total params. DeepSeek V2/V3 use ' +
      'Multi-Latent Attention (MLA), compressing KV cache even further by projecting keys and values into ' +
      'a low-rank latent space.',
  },

  // ── 5. KV Cache at Scale ───────────────────────────────────────────
  {
    id: 'inference-intermediate-05',
    mode: 'inference',
    difficulty: 'intermediate',
    order: 4,
    title: 'KV Cache Memory at Long Contexts',
    concept: 'Managing KV cache memory with KV cache quantization',
    learningObjectives: [
      'Understand KV cache at long contexts with high batch can exceed available GPU memory',
      'Know KV cache quantization (FP8/INT8) halves dynamic per-request memory',
      'Recognize KV cache precision as an independent lever from weight precision',
    ],
    briefing:
      'You are building a long-context document analysis service with LLaMA 3.3 70B on 4 H100 GPUs. ' +
      'The current config uses TP=4, batch=32, and 16K input sequences — but it OOMs. The {{kv-cache|KV cache}} ' +
      'for 32 concurrent sequences at 16,640 tokens each is enormous, ' +
      'pushing total memory beyond 80 GB per GPU.\n\n' +
      'Your task: make this configuration fit in memory. ' +
      'The KV cache is the bottleneck — manage it.',
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
      { field: 'throughput.tokensPerSecond', operator: '>', value: 100, label: 'Throughput > 100 tok/s' },
    ],
    hints: [
      'The model weights fit at TP=4, but the KV cache for 32 sequences at 16K context pushes memory over the limit. Look for a way to reduce KV cache memory.',
      'KV cache precision is independent of weight precision — reduce KV cache bytes per element to halve the dynamic memory.',
    ],
    successExplanation:
      'KV cache memory = `numLayers × numKVHeads × headDim × seqLen × batchSize × 2 (K and V) × bytesPerElement`. ' +
      'For LLaMA 3.3 70B at 16K context, each request\'s KV cache is several GB. Batching multiplies this linearly.\n\n' +
      'KV cache quantization (FP8 or INT8) halves the bytes per element, directly halving the dynamic memory. ' +
      'This is independent of weight precision — you can run FP8 KV cache alongside BF16 weights. ' +
      'In production, KV cache quantization is standard because the quality impact on attention scores is minimal.',
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
    ],
    briefing:
      'Your inference cluster needs to handle high request volume. With LLaMA 3.3 70B on 8 H100 GPUs ' +
      'and TP=8 already configured, a single request generates ~158 tokens/sec. But the {{decode}} phase is ' +
      'memory-bandwidth bound — the GPU reads all model weights for every single token, regardless of batch size. ' +
      'By batching multiple requests, you amortize the weight read cost across sequences. ' +
      'Find a batch size that achieves over 2k tok/s total throughput.',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'h100-sxm',
      numGPUs: 8,
      gpusPerNode: 8,
      tensorParallel: 8,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Inference runs successfully' },
      { field: 'throughput.tokensPerSecond', operator: '>', value: 2000, label: 'Throughput > 2k tok/s' },
    ],
    hints: [
      'At batch=1, each decode step reads all weights to produce one token. At batch=N, the same weight read produces N tokens. Increase the batch size.',
      'Throughput scales nearly linearly with batch size in the memory-bound regime. Watch for memory limits as KV cache grows.',
    ],
    successExplanation:
      'Decode is memory-bandwidth bound: the GPU must read all model weights from HBM every decode step. ' +
      'At batch=1, one weight read produces one token. At batch=N, the same weight read produces N tokens — ' +
      'throughput scales nearly linearly.\n\nThe limit comes from KV cache: each sequence adds its own KV cache to the read, ' +
      'and GPU memory eventually fills up. In practice, throughput grows sub-linearly at very large batches because ' +
      'KV cache reads become significant relative to weight reads.',
  },

  // ── 7. Latency Optimization ────────────────────────────────────────
  {
    id: 'inference-intermediate-07',
    mode: 'inference',
    difficulty: 'intermediate',
    order: 6,
    title: 'Minimizing Per-Token Latency',
    concept: 'Optimizing TPOT for real-time applications',
    learningObjectives: [
      'Know TPOT ~ weight_bytes / (memory_bandwidth x efficiency) in the bandwidth-bound regime',
      'Identify three TPOT levers: TP (split weights), quantization (fewer bytes), small batch (less KV reads)',
      'Understand the fundamental tension: latency optimization (batch=1, max TP) vs throughput optimization (large batch)',
    ],
    briefing:
      'You are building a real-time coding assistant that needs fast token generation for responsive streaming. ' +
      'The target is under 15 ms per output token (TPOT) with LLaMA 3.3 70B on 4 H100 GPUs. ' +
      'TPOT is determined by how fast GPUs can read model weights from HBM during the memory-bound decode phase. ' +
      'Explore tensor parallelism, batch size, and weight precision to minimize TPOT.',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'h100-sxm',
      numGPUs: 4,
      gpusPerNode: 4,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Inference runs successfully' },
      { field: 'latency.tpot', operator: '<', value: 15.0, label: 'TPOT < 15 ms' },
    ],
    hints: [
      'TPOT is proportional to weight bytes read per GPU. Higher TP reduces per-GPU weights. Lower precision reduces bytes per parameter.',
      'With TP=4 at BF16, TPOT is close to the target. Keep batch size at 1 for lowest latency (no extra KV cache reads).',
      'Using FP8 with TP=4 roughly halves TPOT by halving the bytes read per weight. But TP=4 at BF16 with batch=1 may also meet the target — try it first.',
    ],
    successExplanation:
      'TPOT in the memory-bound regime equals `weight_bytes / (memory_bandwidth × efficiency)`. ' +
      'Three levers reduce it: (1) TP splits weights across GPUs, reducing per-GPU reads proportionally. ' +
      '(2) Quantization (FP8/INT8) halves bytes per parameter. (3) Small batch sizes minimize additional KV cache reads.\n\n' +
      'For real-time applications, the priority is low latency per token (small batch, high TP) rather than high throughput ' +
      '(large batch). There is a fundamental tension: latency optimization (`batch=1`, max TP) wastes GPU compute, ' +
      'while throughput optimization (large batch) increases per-token latency.',
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
      'Compare GPUs by bandwidth: H100 SXM (3.35 TB/s) vs A100 (2.0 TB/s)',
      'Understand arithmetic intensity for decode is far below the GPU roofline — bandwidth is the bottleneck',
    ],
    briefing:
      'You want to serve Qwen 3 14B on a single GPU and need at least 120 tokens/sec throughput. ' +
      'For inference, the GPU spec that matters most during decode is not compute (TFLOPS) but memory bandwidth — ' +
      'because decode reads all weights every step and does minimal computation per byte. ' +
      'You start with an H100 SXM — observe baseline, then switch to other GPUs to understand how ' +
      'memory bandwidth affects decode throughput. Find a configuration that exceeds 120 tokens/sec.',
    setup: {
      modelId: 'qwen3-14b',
      gpuId: 'h100-sxm',
      numGPUs: 1,
      gpusPerNode: 1,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Inference runs successfully' },
      { field: 'throughput.tokensPerSecond', operator: '>', value: 120, label: 'Throughput > 120 tok/s' },
    ],
    hints: [
      'Qwen 3 14B in BF16 fits on any 80 GB GPU. The question is how fast each GPU can read those weights — memory bandwidth is the deciding spec for decode throughput.',
      'Compare the memory bandwidth of different GPUs. Higher bandwidth means faster weight reads and higher decode throughput. The throughput difference between GPUs is roughly proportional to their bandwidth ratio.',
      'If a single GPU at BF16 batch=1 falls short of the throughput target, try weight quantization (INT8 or FP8) to halve weight reads, or increase batch size to amortize them.',
    ],
    successExplanation:
      'For single-request decode, throughput is approximately: `params × bytes_per_param / (bandwidth × efficiency)`. ' +
      'The H100 SXM (3.35 TB/s bandwidth) reads 28 GB of weights in ~11 ms, giving ~86 tok/s. ' +
      'The A100 80GB (2.0 TB/s) takes ~17 ms, giving ~56 tok/s.\n\nBoth GPUs have enormous compute capacity (hundreds of TFLOPS), ' +
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
    concept: 'Using a small draft model to speed up large model decoding',
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
    concept: 'Halving weight memory reads with FP8 precision',
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
