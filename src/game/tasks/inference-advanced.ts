import type { GameTask } from '../types.ts';
// Glossary annotations: {{termId}} or {{termId|display text}} for tooltip terms.
// [text](url) for paper links in successExplanation.
// Annotate FIRST OCCURRENCE per difficulty tier only. Check earlier tasks before adding.

export const INFERENCE_ADVANCED_TASKS: GameTask[] = [
  // Task 1: DeepSeek V3 Inference
  {
    id: 'inference-advanced-01',
    mode: 'inference',
    difficulty: 'advanced',
    order: 0,
    title: 'DeepSeek V3 Inference',
    briefing:
      'DeepSeek V3 is a 671B-parameter {{moe|Mixture-of-Experts}} model with 256 experts, ' +
      'of which only 8 are active per token. Despite its massive total parameter count, ' +
      'the {{active-params|active parameter}} footprint (~37B) makes it surprisingly efficient at inference time. ' +
      'The challenge is fitting those 671B parameters into GPU memory while maintaining good throughput. ' +
      'You have 8 H100 GPUs — configure {{tp|tensor parallelism}} and quantization to serve this model. ' +
      'Consider that MoE models have a unique memory profile: all expert weights must be loaded ' +
      'even though only a fraction are active for each token.',
    concept: 'MoE Inference at Scale',
    learningObjectives: [
      'Understand MoE memory dominance: 671B params all must be resident even though only ~37B are active',
      'Know INT4 is often necessary for single-node MoE inference (FP8 at TP=8 barely exceeds 80 GB/GPU)',
      'Calculate per-GPU memory at various precisions with TP to find feasible configurations',
    ],
    setup: {
      modelId: 'deepseek-v3',
      gpuId: 'h100-sxm',
      numGPUs: 8,
      gpusPerNode: 8,
    },
    winningCriteria: [
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Model fits in memory',
      },
      {
        field: 'throughput.tokensPerSecond',
        operator: '>',
        value: 50,
        label: 'Throughput > 50 tok/s',
      },
    ],
    hints: [
      'DeepSeek V3 has 671B total parameters but only ~37B active per token. All 671B must fit in memory across your GPUs.',
      'With 8 H100s (80GB each), calculate whether FP8 (1 byte/param) fits per GPU at TP=8. If not, try INT4 (0.5 bytes/param) which further halves memory.',
      'Use TP=8 to shard the model across all 8 GPUs. {{nvlink|NVLink}} provides high-bandwidth communication within a single node, making TP=8 efficient here.',
    ],
    successExplanation:
      'MoE models like DeepSeek V3 have a unique inference characteristic: memory is dominated by ' +
      'the total parameter count (all experts must be resident), but compute scales with active parameters only. ' +
      'This means MoE models are more memory-bound during decode than their dense counterparts of the same ' +
      'active size.\n\nAggressive quantization (INT4) is often necessary for MoE inference on a single node — ' +
      'with 671B parameters, even FP8 with TP=8 barely fits. INT4 provides comfortable headroom while ' +
      'preserving the compute advantage of sparse activation.',
  },

  // Task 2: Expert Parallel Inference
  {
    id: 'inference-advanced-02',
    mode: 'inference',
    difficulty: 'advanced',
    order: 1,
    title: 'Expert Parallel Inference',
    briefing:
      'Expert Parallelism (EP) is a technique specific to MoE models that distributes experts ' +
      'across GPUs rather than replicating them. Instead of every GPU holding all 256 experts, ' +
      'each GPU holds a subset, and tokens are routed to the appropriate GPU via {{all-to-all}} communication. ' +
      'You have 16 H100 GPUs (2 nodes) serving DeepSeek V3. Configure EP to efficiently distribute ' +
      'the expert parameters while keeping tensor parallelism for the dense attention layers. ' +
      'The goal is to achieve higher throughput than pure TP alone by reducing per-GPU memory pressure ' +
      'and enabling larger batch sizes.',
    concept: 'Expert Parallelism for MoE Serving',
    learningObjectives: [
      'Understand EP distributes experts across GPUs instead of replicating all on each',
      'Know EP uses all-to-all communication (different from TP all-reduce)',
      'Understand device routing limits (M=4 for DeepSeek V3) bound all-to-all volume',
      'Know EP frees per-GPU memory, enabling larger batches and higher throughput',
    ],
    setup: {
      modelId: 'deepseek-v3',
      gpuId: 'h100-sxm',
      numGPUs: 16,
      gpusPerNode: 8,
    },
    winningCriteria: [
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Model fits in memory',
      },
      {
        field: 'throughput.tokensPerSecond',
        operator: '>',
        value: 100,
        label: 'Throughput > 100 tok/s',
      },
    ],
    hints: [
      'Expert Parallelism distributes experts across GPUs. With EP, each GPU holds a fraction of the experts, dramatically reducing per-GPU memory.',
      'Combine TP and EP: use TP within a node (TP=8 over NVLink) and EP across nodes. This keeps the fast NVLink for dense all-reduce and uses IB for expert routing.',
      'With less memory consumed by expert weights, you can increase batch size to improve throughput. EP trades all-to-all communication for better memory efficiency and higher batch utilization.',
    ],
    successExplanation:
      'Expert Parallelism is the key to scaling MoE inference beyond a single node. By distributing ' +
      'experts across GPUs, each GPU needs far less memory for expert weights, freeing capacity for larger ' +
      'KV caches and batch sizes.\n\nThe all-to-all communication pattern is different from TP\'s all-reduce: ' +
      'it sends each token to only the GPUs hosting its selected experts. DeepSeek V3\'s routing device ' +
      'limit (M=4) means each token contacts at most 4 EP groups, which bounds all-to-all volume and makes ' +
      'EP scaling efficient even at high degrees.',
  },

  // Task 3: Speculative Decoding Mastery (merged from old IA-03 + IA-04)
  {
    id: 'inference-advanced-03',
    mode: 'inference',
    difficulty: 'advanced',
    order: 2,
    title: 'Speculative Decoding Mastery',
    briefing:
      'Speculative decoding uses a small, fast "draft" model to propose K candidate tokens, then verifies ' +
      'them in parallel with the large "target" model. Verification of K tokens costs about the same as ' +
      'generating 1 token (a single forward pass with K+1 positions).\n\n' +
      'Two decisions matter: (1) which draft model to use — same family for high acceptance rates, but small ' +
      'enough to run quickly; and (2) how many tokens K to draft per step — higher K means more potential ' +
      'speedup but more wasted draft compute when tokens are rejected.\n\n' +
      'You have LLaMA 3.1 405B on 16 H100 GPUs. Enable speculative decoding, choose a draft model, and ' +
      'tune K to achieve both low latency and good throughput.',
    concept: 'Speculative Decoding: Draft Selection and K-Tuning',
    learningObjectives: [
      'Choose appropriate draft model: 10-100x smaller than target, same model family for high acceptance',
      'Understand quality-speed tradeoff: larger drafts accept more but have more overhead',
      'Know E[tokens/step] = (1 - alpha^(K+1)) / (1 - alpha) where alpha = acceptance rate',
      'Understand optimal K depends on the ratio of draft generation time to target verification time',
    ],
    setup: {
      modelId: 'llama3-405b',
      gpuId: 'h100-sxm',
      numGPUs: 16,
      gpusPerNode: 8,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Model fits in memory' },
      { field: 'latency.tpot', operator: '<', value: 12, label: 'TPOT < 12 ms' },
      { field: 'throughput.tokensPerSecond', operator: '>', value: 80, label: 'Throughput > 80 tok/s' },
    ],
    hints: [
      'Select a draft model from the same LLaMA family. The draft-to-target size ratio and K together determine the speedup.',
      'The optimal K depends on the draft-to-verify time ratio and acceptance rate. Tune K alongside the draft model choice.',
    ],
    successExplanation:
      'The art of speculative decoding lies in two choices: draft model and K. An ideal draft is ' +
      '10-100x smaller than the target while matching its distribution for high acceptance rates. ' +
      'Same-family models work best because they share tokenization and training data.\n\n' +
      'The expected tokens per verification step follows a geometric series: `E[tokens] = (1 - alpha^(K+1)) / (1 - alpha)`. ' +
      'The wall-clock time per step is `K * draft_time + verify_time`. There is an optimal K that balances ' +
      'draft overhead against verification efficiency — too high and draft time dominates, too low and you ' +
      'lose the speculative advantage. See [Speculative Decoding (Leviathan et al., 2022)](https://arxiv.org/abs/2211.17192) for the original analysis.',
  },

  // Task 4: When More TP Hurts
  {
    id: 'inference-advanced-04',
    mode: 'inference',
    difficulty: 'advanced',
    order: 3,
    title: 'When More TP Hurts',
    briefing:
      'You have Qwen 3 14B on 8 H100 GPUs with TP=8. Each GPU holds only 1/8 of the model weights — ' +
      'about 3.5 GB. The AllReduce communication after every layer dominates the decode step time because ' +
      'the per-GPU weight reads are tiny. Throughput is far below what 8 GPUs should deliver.\n\n' +
      'When a model fits on fewer GPUs, using maximum TP wastes GPUs on communication overhead. ' +
      'Find a configuration that achieves over 675 tokens per second.',
    concept: 'TP vs Replicas for Throughput',
    learningObjectives: [
      'Understand that over-sharding with TP makes AllReduce overhead dominate decode time',
      'Know that replicas provide linear throughput scaling with zero communication overhead',
      'Distinguish latency-oriented (max TP) from throughput-oriented (min TP, max replicas) serving',
    ],
    setup: {
      modelId: 'qwen3-14b',
      gpuId: 'h100-sxm',
      numGPUs: 8,
      gpusPerNode: 8,
      tensorParallel: 8,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Inference succeeds' },
      { field: 'throughput.tokensPerSecond', operator: '>', value: 675, label: 'Throughput > 675 tok/s' },
    ],
    hints: [
      'Qwen 3 14B in BF16 is about 28 GB — it fits on a single 80 GB GPU. Why use all 8 GPUs for one instance?',
      'Reduce TP to create multiple independent replicas. Total throughput = per_replica_throughput × num_replicas.',
    ],
    successExplanation:
      'When a model fits on fewer GPUs than available, using maximum TP is counterproductive. ' +
      'TP=8 means each GPU holds only 3.5 GB of weights but must still perform AllReduce communication ' +
      'at every layer — the overhead dominates the tiny per-GPU computation.\n\n' +
      'With TP=1 and 8 replicas, each GPU independently serves requests with zero communication overhead. ' +
      'Total throughput scales linearly with replica count. This is the fundamental distinction between ' +
      'latency-oriented serving (max TP to minimize per-request TPOT) and throughput-oriented serving ' +
      '(min TP, max replicas to maximize aggregate tokens/sec).',
  },

  // Task 5: Multi-Replica Serving
  {
    id: 'inference-advanced-05',
    mode: 'inference',
    difficulty: 'advanced',
    order: 4,
    title: 'Multi-Replica Serving',
    briefing:
      'LLaMA 3.3 70B on 8 H100 GPUs with TP=4. At BF16, each GPU holds ~35 GB of weights — it fits, ' +
      'but all 8 GPUs are dedicated to a single instance. With TP=4, 4 GPUs are used per replica and ' +
      'the other 4 sit idle. But reducing TP below 4 at BF16 would not fit the 140 GB model.\n\n' +
      'The trick: FP8 weight quantization halves weight memory to ~70 GB total, making TP=2 feasible ' +
      '(35 GB per GPU). With TP=2, you can run 4 replicas instead of 2 — doubling aggregate throughput.\n\n' +
      'Configure TP and weight precision to maximize total throughput above 250 tok/s.',
    concept: 'Multi-Replica Inference with Quantization',
    learningObjectives: [
      'Know replicas provide linear throughput scaling with no communication overhead',
      'Understand FP8 quantization enables lower TP, which enables more replicas',
      'Distinguish latency-oriented (max TP, batch=1) from throughput-oriented (min TP, max replicas) configs',
    ],
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'h100-sxm',
      numGPUs: 8,
      gpusPerNode: 8,
      tensorParallel: 4,
      weightPrecision: 'bf16',
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
        value: 250,
        label: 'Throughput > 250 tok/s',
      },
    ],
    hints: [
      'At BF16 with TP=4, the model fits but you only have 2 replicas. Quantization can reduce weight memory enough to lower TP.',
      'Lower TP means more replicas. Explore weight precision options that halve memory, enabling a smaller TP degree.',
    ],
    successExplanation:
      'Multi-replica serving is the throughput-optimal strategy when you have more GPUs than a single ' +
      'model instance needs. FP8 quantization halves weight memory, enabling TP=2 instead of TP=4 — ' +
      'which doubles the replica count from 2 to 4.\n\n' +
      'For throughput-oriented serving (high QPS, less concerned about individual latency), minimizing TP ' +
      'and maximizing replicas is almost always better. Production serving systems like vLLM and TensorRT-LLM ' +
      'automatically manage multi-replica deployments.',
  },

  // Task 6: Cost-per-Token Optimization
  {
    id: 'inference-advanced-06',
    mode: 'inference',
    difficulty: 'advanced',
    order: 5,
    title: 'Cost-per-Token Optimization',
    briefing:
      'In production serving, cost efficiency often matters more than raw throughput or latency. ' +
      'The cost per million tokens is determined by: (GPU_hourly_rate * num_GPUs) / tokens_per_second * 1e6 / 3600. ' +
      'Every optimization that increases throughput directly reduces cost. But there is a subtlety: ' +
      'using fewer GPUs reduces the numerator, and sometimes a smaller but well-optimized configuration ' +
      'beats a larger one on cost even if absolute throughput is lower. ' +
      'Minimize the cost of serving LLaMA 3.3 70B on 8 A100-80GB GPUs. The A100 does not support FP8, so INT8 is the best quantization option. How does GPU choice affect cost efficiency?',
    concept: 'Cost-Efficient Inference Optimization',
    learningObjectives: [
      'Know cost formula: (GPU_hourly_rate x numGPUs / 3600) / tokens_per_sec x 1e6',
      'Understand throughput is the primary cost lever — 2x throughput = 0.5x cost',
      'Combine quantization + multi-replica + batching for cost-efficient production serving',
    ],
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'a100-80gb',
      numGPUs: 8,
      gpusPerNode: 8,
    },
    winningCriteria: [
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Inference succeeds',
      },
      {
        field: 'costPerMillionTokens',
        operator: '<',
        value: 5.0,
        label: 'Cost < $5.00/M tokens',
      },
    ],
    hints: [
      'Cost = (rate × numGPUs / 3600) / tokensPerSecond × 1e6. Check the GPU hourly rate in the sidebar — cheaper GPUs have less bandwidth but lower cost.',
      'Multi-replica serving is key: TP=2 with 4 replicas gives you 4x the throughput of a single instance. Combined with quantization, this dramatically improves cost efficiency.',
      'Enable INT8 quantization (A100 does not support FP8), use continuous batching if available, and consider paged attention to maximize the batch size each replica can handle. Every token/second you gain directly reduces $/M tokens. The Pareto frontier shows the cost-latency frontier — points on the frontier are non-dominated.',
    ],
    successExplanation:
      'Cost-per-token optimization is a holistic challenge that combines all inference techniques.' +
      ' Different GPUs have different price-performance ratios: the A100 is cheaper per hour but has less bandwidth, ' +
      'while the H100 costs more but delivers higher throughput.' +
      ' The key techniques are: ' +
      'quantization (more throughput per GPU), multi-replica serving (linear throughput scaling), ' +
      'batching (amortize memory bandwidth across requests), and paged attention (efficient memory utilization ' +
      'for larger batches).\n\nIn practice, the biggest lever is usually throughput: doubling throughput halves ' +
      'cost. This is why aggressive quantization (FP8, INT4) is so popular in production — the small quality ' +
      'loss is worth the 2-4x cost reduction.',
  },

  // Task 7: Long Context Inference
  {
    id: 'inference-advanced-07',
    mode: 'inference',
    difficulty: 'advanced',
    order: 6,
    title: 'Long Context Inference',
    briefing:
      'Long context inference (128K+ tokens) presents unique challenges. The {{kv-cache|KV cache}} grows linearly ' +
      'with sequence length, and attention computation grows quadratically. At 128K tokens, the KV cache ' +
      'alone can consume tens of gigabytes per request, severely limiting batch size and throughput. ' +
      'LLaMA 3.1 405B supports up to 128K context. Configure a 16-GPU H100 setup to handle long-context ' +
      'inference. You will need to carefully manage memory: the model weights and KV cache must both fit, ' +
      'and techniques like KV cache quantization and {{paged-attention|paged attention}} become essential.',
    concept: 'Long-Context Inference Memory Management',
    learningObjectives: [
      'Understand KV cache dominates memory at 128K+ context (can exceed weight memory per request)',
      'Know compound optimization: TP + KV cache quantization + paged attention + weight quantization',
      'Recognize prefix caching as a technique for shared system prompts, reusing KV cache across requests',
    ],
    setup: {
      modelId: 'llama3-405b',
      gpuId: 'h100-sxm',
      numGPUs: 16,
      gpusPerNode: 8,
      inputSeqLen: 128000,
    },
    winningCriteria: [
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Model fits in memory',
      },
    ],
    hints: [
      'At 128K sequence length, the KV cache for a 405B model is enormous — even with {{gqa|GQA}} reducing KV heads. KV cache quantization (FP8 or INT8) halves KV cache memory. Enable it alongside weight quantization.',
      'Paged attention avoids internal fragmentation in KV cache allocation — instead of reserving max_seq_len per request, it allocates pages on demand. This is critical at long context lengths where reserved-but-unused memory wastes capacity.',
    ],
    successExplanation:
      'Long-context inference is memory-dominated. For LLaMA 3.1 405B at 128K context, the KV cache per ' +
      'request is approximately: `2 (K+V) * 128 (layers) * 8 (KV heads) * 128 (head_dim) * 128000 * bytes`. ' +
      'Even with GQA reducing KV heads from 128 to 8, this is massive.\n\nIn production, long-context serving ' +
      'typically uses: (1) TP to shard both weights and KV cache, (2) KV cache quantization to FP8, ' +
      '(3) paged attention for efficient memory utilization, and sometimes (4) prefix caching — when many requests share the same system prompt, the KV cache ' +
      'for that prefix can be computed once and reused across all requests, saving significant prefill ' +
      'compute and memory.',
  },

  // Task 8: Speculative Decoding for MoE
  {
    id: 'inference-advanced-08',
    mode: 'inference',
    difficulty: 'advanced',
    order: 7,
    title: 'Speculative Decoding for MoE',
    briefing:
      'Speculative decoding is especially interesting for MoE models. During standard autoregressive decoding, ' +
      'an MoE model activates only a fraction of its experts per token, making it memory-bandwidth-bound ' +
      '(all expert weights must be loaded but few are used). Speculative decoding can help by verifying ' +
      'multiple tokens in a single forward pass, amortizing the weight-loading cost across more useful tokens. ' +
      'Configure DeepSeek V3 (671B total, ~37B active) on 8 H100 GPUs. ' +
      'The 671B parameters require aggressive quantization to fit — even FP8 (84 GB/GPU ' +
      'with TP=8) barely exceeds 80 GB. INT4 quantization brings it to ~42 GB/GPU, ' +
      'leaving room for KV cache and a small draft model.',
    concept: 'Speculative Decoding with MoE Target Models',
    learningObjectives: [
      'Understand MoE amplifies spec decoding benefit: all expert weights loaded but few used per token',
      'Know verification amortizes weight-loading cost across all accepted tokens in one pass',
      'Combine INT4 + TP + spec decoding for practical MoE serving on single node',
    ],
    setup: {
      modelId: 'deepseek-v3',
      gpuId: 'h100-sxm',
      numGPUs: 8,
      gpusPerNode: 8,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Model fits in memory' },
      { field: 'throughput.tokensPerSecond', operator: '>', value: 50, label: 'Throughput > 50 tok/s' },
    ],
    hints: [
      'MoE models are strongly memory-bandwidth-bound during decode because all 671B parameters must be loaded but only ~37B are used. Speculative decoding helps by producing multiple tokens per weight-loading cycle.',
      'With 671B parameters and TP=8, check whether FP8 fits per GPU. If not, you need more aggressive quantization to leave room for KV cache and the draft model.',
      'Use TP=8 to shard across all GPUs, and INT4 weight precision. The draft model is small enough to stay in BF16 alongside the quantized target model.',
    ],
    successExplanation:
      'Speculative decoding is particularly effective for MoE models because of the memory-bandwidth ' +
      'bottleneck. In standard decoding, an MoE model must load all expert weights from HBM each step ' +
      'even though only a small fraction contribute to the output. Each verified token in speculative ' +
      'decoding essentially gets a "free ride" on that weight-loading pass.\n\nThe verification step processes ' +
      '`K+1` positions in a single forward pass — the attention cost grows with K, but the expert weight ' +
      'loading cost stays constant. This makes the effective memory-bandwidth utilization much higher.',
  },

  // Task 9: Consumer GPU Inference
  {
    id: 'inference-advanced-09',
    mode: 'inference',
    difficulty: 'advanced',
    order: 8,
    title: 'Consumer GPU Inference',
    briefing:
      'Not every deployment has access to expensive H100s. The NVIDIA {{l40s|L40S}} is a data center GPU with 48GB ' +
      'of memory and {{pcie|PCIe}} connectivity (no NVLink). Serving LLaMA 3.3 70B on 4 L40S GPUs requires aggressive ' +
      'quantization and careful tensor parallelism configuration. The lack of NVLink means TP communication ' +
      'goes over PCIe at ~31.5 GB/s per GPU — roughly 10x slower than NVLink. This makes the TP overhead ' +
      'more significant and favors lower TP degrees or heavier quantization to reduce communication volume. ' +
      'Fit LLaMA 3.3 70B on 4 L40S GPUs and achieve a working deployment.',
    concept: 'Budget GPU Inference with PCIe Interconnect',
    learningObjectives: [
      'Understand PCIe (~31.5 GB/s) vs NVLink (~900 GB/s) impact: ~28x slower TP communication',
      'Know aggressive quantization (INT4/INT8) compensates for less memory and bandwidth',
      'Deploy 70B on 4x 48GB L40S GPUs — a practical budget serving configuration',
    ],
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'l40s',
      numGPUs: 4,
      gpusPerNode: 4,
    },
    winningCriteria: [
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Model fits in memory',
      },
      {
        field: 'memoryUtilization',
        operator: '<',
        value: 1.0,
        label: 'Memory utilization < 100%',
      },
    ],
    hints: [
      'LLaMA 70B in BF16 needs roughly 2 bytes per parameter — calculate whether 4 L40S GPUs (48GB each) can hold it with TP=4. You will likely need quantization.',
      'INT4 quantization cuts weight memory dramatically, leaving room for KV cache and activations. The quality trade-off is moderate — INT4 is widely used in production for 70B-class models.',
      'L40S has no NVLink, so TP communication uses PCIe (~31.5 GB/s). TP=4 over PCIe will have noticeable overhead but is still functional. Minimize communication by using aggressive quantization.',
    ],
    successExplanation:
      'Serving large models on budget GPUs is a practical reality for many organizations. The key ' +
      'differences from premium GPUs are: (1) less memory per GPU (48GB vs 80GB), requiring more aggressive ' +
      'quantization; (2) PCIe instead of NVLink for TP communication, adding ~5-10x more communication ' +
      'latency per all-reduce; (3) lower memory bandwidth, making decode even more bandwidth-bound.\n\n' +
      'Aggressive quantization (INT4/INT8) is essential — it simultaneously reduces memory footprint, ' +
      'communication volume, and memory bandwidth demand. The quality-efficiency trade-off is usually ' +
      'acceptable for production workloads.',
  },

  // Task 10: The Full Serving Stack
  {
    id: 'inference-advanced-10',
    mode: 'inference',
    difficulty: 'advanced',
    order: 9,
    title: 'The Full Serving Stack',
    briefing:
      'This final challenge brings everything together. You are deploying LLaMA 3.1 405B — one of the largest ' +
      'dense models — on 16 H100 GPUs for production serving. Your goal is to achieve high throughput by ' +
      'combining every technique you have learned: tensor parallelism for model sharding, quantization for ' +
      'memory efficiency, paged attention for KV cache management, {{speculative-decoding|speculative decoding}} for {{decode}} speedup, ' +
      'and multi-replica serving if possible. Each technique contributes to the final throughput number, and ' +
      'the art is in how they compose together. A well-optimized 405B serving setup should push significant ' +
      'throughput on 16 H100s.',
    concept: 'Full-Stack Inference Optimization',
    learningObjectives: [
      'Compose all techniques: TP, quantization, paged attention, spec decoding, multi-replica, continuous batching',
      'Understand TP=8 x 2 replicas can beat TP=16 for throughput (avoid cross-node TP overhead)',
      'Know production frameworks (vLLM, TRT-LLM, SGLang) orchestrate these techniques automatically',
    ],
    setup: {
      modelId: 'llama3-405b',
      gpuId: 'h100-sxm',
      numGPUs: 16,
      gpusPerNode: 8,
    },
    winningCriteria: [
      {
        field: 'success',
        operator: '==',
        value: true,
        label: 'Model fits in memory',
      },
      {
        field: 'throughput.tokensPerSecond',
        operator: '>',
        value: 100,
        label: 'Throughput > 100 tok/s',
      },
    ],
    hints: [
      'LLaMA 405B needs aggressive quantization to fit. Calculate per-GPU weight memory at FP8 with TP=8 on one node — check if it fits in 80GB with room for KV cache. Cross-node TP (TP=16) is another option but adds IB communication overhead.',
      'Consider TP=8 with 2 replicas (one per node) instead of TP=16. Two replicas double throughput while avoiding cross-node TP overhead. Batch each replica independently for maximum utilization.',
      'Layer all optimizations: FP8 weights, FP8 KV cache, paged attention, flash attention, and speculative decoding with a small LLaMA draft model. Each optimization compounds — FP8 + speculation + multi-replica together can deliver dramatic throughput improvements.',
    ],
    successExplanation:
      'Production-grade LLM serving is the composition of many techniques, each addressing a different ' +
      'bottleneck. Quantization reduces memory and bandwidth demand. TP enables serving models larger than ' +
      'one GPU. Multi-replica scales throughput linearly. Paged attention improves memory utilization for ' +
      'larger batches. Speculative decoding accelerates autoregressive generation. Continuous batching ' +
      'maximizes GPU utilization across requests.\n\nIn real deployments, frameworks like vLLM, TensorRT-LLM, ' +
      'and SGLang orchestrate all of these techniques together, often automatically tuning parameters like ' +
      'TP degree and batch size based on the hardware and model.',
  },
];
