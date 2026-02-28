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
    expectedChanges: [
      { field: 'weightPrecision', check: 'changed', label: 'Changed weight precision' },
      { field: 'tensorParallel', check: 'increased', label: 'Increased tensor parallelism' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'DeepSeek V3 has 671B total parameters but only ~37B active per token. All 671B must fit in memory across your GPUs.',
      'With 8 H100s (80GB each), calculate whether FP8 (1 byte/param) fits per GPU at `TP=8`. If not, try INT4 (0.5 bytes/param) which further halves memory.',
      'Use `TP=8` to shard the model across all 8 GPUs. {{nvlink|NVLink}} provides high-bandwidth communication within a single node, making `TP=8` efficient here.',
    ],
    successExplanation:
      'MoE models like DeepSeek V3 have a unique inference characteristic: memory is dominated by ' +
      'the total parameter count (all experts must be resident), but compute scales with active parameters only. ' +
      'This means MoE models are more memory-bound during decode than their dense counterparts of the same ' +
      'active size.\n\nAggressive quantization is often necessary for MoE inference on a single node — ' +
      'with 671B parameters, even `FP8` with `TP`=8 barely fits. `INT4` provides comfortable headroom while ' +
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
      '{{ep|Expert Parallelism (EP)}} is a technique specific to MoE models that distributes experts ' +
      'across GPUs rather than replicating them. Instead of every GPU holding all 256 experts, ' +
      'each GPU holds a subset, and tokens are routed to the appropriate GPU via {{all-to-all}} communication. ' +
      'You have 16 H100 GPUs (2 nodes) serving DeepSeek V3. Configure EP to efficiently distribute ' +
      'the expert parameters while keeping tensor parallelism for the dense attention layers. ' +
      'The goal is to achieve higher throughput than pure TP alone by reducing per-GPU memory pressure ' +
      'and enabling larger batch sizes.',
    concept: 'Scaling MoE beyond single-node memory',
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
    expectedChanges: [
      { field: 'expertParallel', check: 'increased', label: 'Increased expert parallelism' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'Expert Parallelism distributes experts across GPUs. With EP, each GPU holds a fraction of the experts, dramatically reducing per-GPU memory.',
      'Combine TP and EP: use TP within a node (`TP=8` over `NVLink`) and EP across nodes. This keeps fast `NVLink` for dense all-reduce and uses the inter-node fabric for expert routing.',
      'With less memory consumed by expert weights, you can increase batch size to improve throughput. EP trades all-to-all communication for better memory efficiency and higher batch utilization.',
    ],
    successExplanation:
      '`EP` is the key to scaling MoE inference beyond a single node. By distributing ' +
      'experts across GPUs, each GPU needs far less memory for expert weights, freeing capacity for larger ' +
      'KV caches and batch sizes.\n\nThe all-to-all communication pattern is different from `TP`\'s all-reduce: ' +
      'it sends each token to only the GPUs hosting its selected experts. DeepSeek V3\'s ' +
      '{{device-limited-routing|device-limited routing}} (`M=4`) means each token contacts at most 4 `EP` groups, which bounds all-to-all volume and makes ' +
      '`EP` scaling efficient even at high degrees.',
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
    concept: 'Draft-verify optimization and tuning',
    learningObjectives: [
      'Choose appropriate draft model: 10-100x smaller than target, same model family for high acceptance',
      'Understand quality-speed tradeoff: larger drafts accept more but have more overhead',
      'Know E[tokens/step] = (1 - alpha^(K+1)) / (1 - alpha) where alpha = acceptance rate',
      'Understand optimal K depends on the ratio of draft generation time to target verification time',
      'Know speculative decoding is especially effective for MoE models where all expert weights are loaded but few are used — verification amortizes this bandwidth waste',
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
    expectedChanges: [
      { field: 'speculativeDecoding', check: 'enabled', label: 'Enabled speculative decoding' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'Select a draft model from the same LLaMA family. The draft-to-target size ratio and K together determine the speedup.',
      'The optimal K depends on the draft-to-verify time ratio and acceptance rate. Tune K alongside the draft model choice.',
    ],
    successExplanation:
      'The key to speculative decoding lies in two choices: draft model and K. An ideal draft is ' +
      '10-100x smaller than the target while matching its distribution for high acceptance rates. ' +
      'Same-family models work best because they share tokenization and training data.\n\n' +
      'The expected tokens per verification step follows a geometric series: `E[tokens] = (1 - alpha^(K+1)) / (1 - alpha)`. ' +
      'The wall-clock time per step is `K * draft_time + verify_time`. There is an optimal K that balances ' +
      'draft overhead against verification efficiency — too high and draft time dominates, too low and you ' +
      'lose the speculative advantage. See [Speculative Decoding (Leviathan et al., 2022)](https://arxiv.org/abs/2211.17192) for the original analysis.\n\n' +
      'Speculative decoding is especially powerful for MoE models like DeepSeek V3. During standard decode, ' +
      'the GPU loads all expert weights from memory but only a fraction contribute to the output token. ' +
      'Speculative decoding amortizes this bandwidth waste across all verified tokens in a single forward pass — ' +
      'each accepted token gets a "free ride" on the expert weight loading.',
  },

  // Task 4: Optimizing Time to First Token
  {
    id: 'inference-advanced-04',
    mode: 'inference',
    difficulty: 'advanced',
    order: 3,
    title: 'Optimizing Time to First Token',
    briefing:
      'You\'ve configured LLaMA 3.3 70B on 8 H100 GPUs with `TP=2` and 4 replicas — a setup optimized ' +
      'for aggregate decode throughput. But this deployment serves a document analysis workload: users ' +
      'submit 32K-token documents and need a fast initial response.\n\n' +
      'With `TP=2`, each replica processes the entire 32K-token prefill using only 2 GPUs. Run the ' +
      'config and check the {{ttft|TTFT}} — for long-input workloads, the prefill phase dominates ' +
      'total latency. Think about what hardware resource limits prefill speed, and how to apply more ' +
      'of it.',
    concept: 'Prefill-Decode Hardware Asymmetry',
    learningObjectives: [
      'Understand that prefill is compute-bound: TTFT scales with model_FLOPs × input_tokens / (TFLOPS × TP)',
      'Know that increasing TP divides prefill compute across GPUs, reducing TTFT nearly linearly',
      'Recognize that TTFT optimization (max TP) conflicts with throughput optimization (min TP, max replicas)',
    ],
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'h100-sxm',
      numGPUs: 8,
      gpusPerNode: 8,
      tensorParallel: 2,
      inputSeqLen: 32768,
      outputSeqLen: 128,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Inference succeeds' },
      { field: 'latency.ttft', operator: '<', value: 2500, label: 'TTFT < 2500 ms' },
    ],
    expectedChanges: [
      { field: 'tensorParallel', check: 'increased', label: 'Increased tensor parallelism' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'Run the initial config and check `TTFT` in the dashboard. Prefill processes all 32K input tokens through every layer — a massive compute task. With only 2 GPUs computing in parallel, most of the fleet sits idle during prefill.',
      'Prefill is compute-bound: dominated by matrix multiplications through every layer. Unlike decode (which reads weights once per token), prefill multiplies weights by thousands of input tokens simultaneously. More TP means more GPUs share this compute.',
    ],
    successExplanation:
      'Prefill (processing the prompt) and decode (generating tokens) have fundamentally different ' +
      'hardware bottlenecks. Decode reads all model weights from `HBM` for each output token — it\'s ' +
      'memory-bandwidth-bound, so bandwidth determines speed. But prefill processes thousands of tokens ' +
      'simultaneously through massive matrix multiplications, making it compute-bound — `TFLOPS` determines ' +
      'speed.\n\n' +
      'For this 32K-token document, prefill requires approximately `2 × activeParams × inputTokens` FLOPs ' +
      'for linear projections alone, plus quadratic attention FLOPs. Increasing `TP` divides this compute ' +
      'across more GPUs, reducing `TTFT` nearly linearly.\n\n' +
      'This creates a fundamental tension in serving configuration: throughput-optimized setups (as in earlier ' +
      'tasks) use minimal `TP` with many replicas for aggregate tokens/sec. `TTFT`-optimized setups use maximum ' +
      '`TP` to minimize per-request prefill latency. Production systems choose based on workload — chatbots ' +
      'with short inputs optimize for decode throughput, while RAG and document analysis with long inputs ' +
      'optimize for `TTFT`.\n\n' +
      'You may have noticed total throughput stayed roughly constant regardless of `TP` — at `batch=1`, ' +
      'higher `TP` makes each replica faster but leaves fewer replicas, and these effects cancel. ' +
      '`TP` is a latency optimization. The latency-throughput tradeoff becomes meaningful when batch size ' +
      'increases, allowing each replica to amortize `TP`\'s communication overhead across more tokens.',
  },

  // Task 5: Multi-Replica Serving
  {
    id: 'inference-advanced-05',
    mode: 'inference',
    difficulty: 'advanced',
    order: 4,
    title: 'Multi-Replica Serving',
    briefing:
      'LLaMA 3.3 70B on 8 H100 GPUs with `TP=4`. At BF16, each GPU holds ~35 GB of weights — it fits, ' +
      'but all 8 GPUs are dedicated to a single instance. With `TP=4`, 4 GPUs are used per replica and ' +
      'the other 4 sit idle. But reducing TP below 4 at BF16 would not fit the 140 GB model.\n\n' +
      'The key insight: quantization reduces weight memory, enabling lower `TP`. Lower `TP` frees GPUs ' +
      'for more replicas, and each replica provides linear throughput scaling with zero communication overhead.\n\n' +
      'Configure TP and weight precision to maximize total throughput above 250 tok/s.',
    concept: 'Throughput scaling via instance replication',
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
    expectedChanges: [
      { field: 'weightPrecision', check: 'changed', label: 'Changed weight precision' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'At BF16 with `TP=4`, the model fits but you only have 2 replicas. Quantization can reduce weight memory enough to lower TP.',
      'Lower TP means more replicas. Explore weight precision options that halve memory, enabling a smaller TP degree.',
    ],
    successExplanation:
      'Multi-replica serving is the throughput-optimal strategy when you have more GPUs than a single ' +
      'model instance needs. Quantization reduces weight memory, enabling lower `TP` and thus more replicas ' +
      'from the same GPU budget.\n\n' +
      'For throughput-oriented serving (high QPS, less concerned about individual latency), minimizing `TP` ' +
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
      'beats a larger one on cost even if absolute throughput is lower.\n\n' +
      'You have 8 {{l40s|L40S}} GPUs — significantly cheaper per hour than A100s or H100s. ' +
      'Each L40S has 48 GB of memory with `PCIe` connectivity. The L40S supports FP8 (Ada architecture). ' +
      'Minimize the cost of serving LLaMA 3.3 70B.',
    concept: 'Cost-Efficient Inference Optimization',
    learningObjectives: [
      'Know cost formula: (GPU_hourly_rate x numGPUs / 3600) / tokens_per_sec x 1e6',
      'Understand throughput is the primary cost lever — 2x throughput = 0.5x cost',
      'Combine quantization + multi-replica + batching for cost-efficient production serving',
    ],
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'l40s',
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
    expectedChanges: [
      { field: 'weightPrecision', check: 'changed', label: 'Changed weight precision' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'Cost = (rate × numGPUs / 3600) / tokensPerSecond × 1e6. Check the GPU hourly rates in the sidebar — the L40S is significantly cheaper than the H100. Lower hourly cost means even moderate throughput can achieve good cost efficiency.',
      'INT4 quantization dramatically reduces weight memory. Calculate whether it enables a lower TP degree — and thus more replicas — on 48 GB GPUs. More replicas means linear throughput scaling.',
      'FP8 is available on the L40S (Ada architecture). INT4 enables `TP=2` → 4 replicas. Combined with batching and paged attention, this maximizes throughput per dollar. The `PCIe` interconnect constrains TP — prefer fewer TP ranks with more replicas.',
    ],
    successExplanation:
      'Cost-per-token optimization is a holistic challenge that combines all inference techniques. ' +
      'Cheaper GPUs can offer compelling price-performance ratios: with a lower hourly rate, ' +
      'even moderate throughput can achieve excellent cost efficiency.\n\n' +
      'The key techniques are: quantization (enabling lower `TP` per replica), ' +
      'multi-replica serving (linear throughput scaling), ' +
      'batching (amortize bandwidth across requests), and paged attention (efficient memory for larger batches). ' +
      'On GPUs with `PCIe` interconnect rather than high-bandwidth links, lower `TP` degrees with more replicas ' +
      'tend to be more cost-effective — each `TP` `AllReduce` over `PCIe` is relatively slow, so minimizing `TP` ' +
      'while maximizing replica count is the cost-optimal strategy.',
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
    expectedChanges: [
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
      { field: 'inputSeqLen', check: 'unchanged', label: 'Did not change input sequence length' },
    ],
    hints: [
      'At 128K sequence length, the KV cache for a 405B model is enormous — even with {{gqa|GQA}} reducing KV heads. KV cache quantization (FP8 or INT8) halves KV cache memory. Enable it alongside weight quantization.',
      'Paged attention avoids internal fragmentation in KV cache allocation — instead of reserving max_seq_len per request, it allocates pages on demand. This is critical at long context lengths where reserved-but-unused memory wastes capacity.',
    ],
    successExplanation:
      'Long-context inference is memory-dominated. For LLaMA 3.1 405B at 128K context, the KV cache per ' +
      'request is approximately: `2 (K+V) * 128 (layers) * 8 (KV heads) * 128 (head_dim) * 128000 * bytes`. ' +
      'Even with `GQA` reducing KV heads from 128 to 8, this is massive.\n\nIn production, long-context serving ' +
      'typically uses: (1) `TP` to shard both weights and KV cache, (2) KV cache quantization to `FP8`, ' +
      '(3) paged attention for efficient memory utilization, and sometimes (4) prefix caching — when many requests share the same system prompt, the KV cache ' +
      'for that prefix can be computed once and reused across all requests, saving significant prefill ' +
      'compute and memory.',
  },

  // Task 8: Latency SLA Optimization
  {
    id: 'inference-advanced-08',
    mode: 'inference',
    difficulty: 'advanced',
    order: 7,
    title: 'Latency SLA Optimization',
    concept: 'Multi-constraint serving for production SLAs',
    learningObjectives: [
      'Understand TTFT depends on prefill compute (input length × model FLOPs) while TPOT depends on weight bandwidth',
      'Know batch size creates tension: larger batch improves throughput but increases TPOT',
      'Identify weight quantization as a lever that improves all three metrics simultaneously',
      'Recognize production serving requires satisfying multiple SLA dimensions at once',
    ],
    briefing:
      'Production chatbot serving requires meeting multiple SLA constraints simultaneously. ' +
      'Users expect a fast first response ({{ttft|TTFT}} under 500 ms), smooth token streaming ' +
      '({{tpot|TPOT}} under 15 ms), and the system must handle high request volume ' +
      '(throughput over 200 tokens per second).\n\n' +
      'You have LLaMA 3.3 70B on 8 H100 GPUs with `TP=8`. The current config uses a large batch ' +
      'and moderately long input sequences. Run it and observe: the batch size pushes `TPOT` above the SLA, ' +
      'while the prefill compute makes `TTFT` sluggish. Find the configuration that satisfies all ' +
      'three constraints at once.',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'h100-sxm',
      numGPUs: 8,
      gpusPerNode: 8,
      tensorParallel: 8,
      batchSize: 64,
      inputSeqLen: 4096,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Inference succeeds' },
      { field: 'latency.ttft', operator: '<', value: 500, label: 'TTFT < 500 ms' },
      { field: 'latency.tpot', operator: '<', value: 15, label: 'TPOT < 15 ms' },
      { field: 'throughput.tokensPerSecond', operator: '>', value: 200, label: 'Throughput > 200 tok/s' },
    ],
    expectedChanges: [
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'Run the initial config and check `TTFT`, `TPOT`, and throughput in the dashboard. With `batch=64` at 4K input, both latency metrics exceed their targets. The three constraints pull in different directions.',
      'Batch size is the primary lever. Lower batch reduces `TPOT` but also reduces throughput. Weight quantization (FP8) improves all three metrics: faster weight reads reduce `TPOT`, faster prefill reduces `TTFT`, and higher per-step efficiency boosts throughput.',
      'Use FP8 weight precision and find the batch size sweet spot. The Batch chart shows how `TPOT` and throughput change with batch size — look for the region where both constraints are met simultaneously.',
    ],
    successExplanation:
      'Production LLM serving is a multi-constraint optimization problem. `TTFT` is compute-bound ' +
      '(proportional to input length and model FLOPs), `TPOT` is bandwidth-bound (proportional to ' +
      'weight bytes read per step), and throughput depends on batch utilization.\n\n' +
      'The key insight: these constraints interact through batch size. Higher batch amortizes weight ' +
      'reads across more tokens (higher throughput) but each decode step takes longer (higher `TPOT`). ' +
      'Weight quantization is the "free lunch" that improves all three: fewer bytes to read means ' +
      'faster decode (lower `TPOT`), faster prefill (lower `TTFT`), and higher effective throughput.\n\n' +
      'Real serving systems like vLLM, TensorRT-LLM, and SGLang continuously balance these tradeoffs ' +
      'with dynamic batching, request scheduling, and SLA-aware admission control.',
  },

  // Task 9: Running MoE on Consumer Hardware
  {
    id: 'inference-advanced-09',
    mode: 'inference',
    difficulty: 'advanced',
    order: 8,
    title: 'Running MoE on Consumer Hardware',
    concept: 'Sparse model memory on consumer GPUs',
    learningObjectives: [
      'Understand MoE models require memory for ALL experts but compute only uses active subset',
      'Know 47B total params at BF16 (94 GB) requires TP=4 across 4x 24GB consumer GPUs',
      'Understand PCIe TP is slower than NVLink but viable for small TP degrees',
      'Recognize MoE throughput advantages: fewer active FLOPs per token means faster generation',
    ],
    briefing:
      'Mixtral 8x7B is the most popular {{moe|MoE}} model for local inference. It has 47B total ' +
      'parameters but only activates ~13B per token (top-2 of 8 experts). All 47B parameters ' +
      'must reside in GPU memory because routing is input-dependent — any expert might be selected.\n\n' +
      'You have 4 {{rtx-4090|RTX 4090}} GPUs with 24 GB each (96 GB total). At BF16, Mixtral needs ' +
      '~94 GB for weights alone — it barely fits even with `TP=4`, and leaves almost no room for ' +
      'KV cache. Quantization is essential to get comfortable memory headroom.\n\n' +
      'Your goal: get Mixtral running with good throughput. Think about how MoE memory ' +
      'differs from a dense model of the same active size.',
    setup: {
      modelId: 'mixtral-8x7b',
      gpuId: 'rtx-4090',
      numGPUs: 4,
      gpusPerNode: 4,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Model fits in memory' },
      { field: 'throughput.tokensPerSecond', operator: '>', value: 40, label: 'Throughput > 40 tok/s' },
    ],
    expectedChanges: [
      { field: 'weightPrecision', check: 'changed', label: 'Changed weight precision' },
      { field: 'tensorParallel', check: 'increased', label: 'Increased tensor parallelism' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'At BF16 (2 bytes/param), Mixtral\'s 47B total parameters far exceed what 4 × 24 GB GPUs can hold even with `TP=4`. You need to quantize the weights to free memory for KV cache and operating headroom.',
      'INT8 (1 byte/param) roughly halves weight memory, making `TP=4` across 24 GB GPUs comfortable. INT4 (0.5 bytes/param) gives even more headroom. Calculate the per-GPU weight size yourself: total params × bytes per param ÷ TP. The {{pcie|PCIe}} interconnect makes `TP=4` slower than `NVLink` — choose the quantization that leaves room for batch size.',
      'Despite having 47B total parameters, Mixtral computes with only ~13B active per token — so each decode step does fewer FLOPs than a 47B dense model. The bandwidth bottleneck is the total weight loading (all experts), but the compute is efficient.',
    ],
    successExplanation:
      'MoE models on consumer hardware highlight the memory-compute decoupling. Mixtral\'s 47B ' +
      'parameters all reside in GPU memory, but the router only activates 2 of 8 experts per ' +
      'token (~13B active). Memory is sized by total params (47B), requiring quantization on ' +
      '24 GB GPUs. Compute scales with active params (~13B), giving competitive generation speed. ' +
      'Bandwidth-wise, all expert weights are loaded each decode step even though most go unused.\n\n' +
      'Despite the bandwidth overhead, Mixtral on 4 RTX 4090 GPUs with `INT4`/`INT8` quantization is one ' +
      'of the most popular local inference setups. The `PCIe` interconnect adds `TP` overhead compared ' +
      'to `NVLink`, but for `TP`=4 the impact is manageable. Tools like llama.cpp, Ollama, and vLLM ' +
      'make this configuration accessible to anyone with a multi-GPU desktop.',
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
    expectedChanges: [
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'LLaMA 405B needs aggressive quantization to fit. Calculate per-GPU weight memory at FP8 with `TP=8` on one node — check if it fits in 80GB with room for KV cache. Cross-node TP (`TP=16`) is another option but adds IB communication overhead.',
      'Consider `TP=8` with 2 replicas (one per node) instead of `TP=16`. Two replicas double throughput while avoiding cross-node TP overhead. Batch each replica independently for maximum utilization.',
      'Layer all optimizations: FP8 weights, FP8 KV cache, paged attention, flash attention, and speculative decoding with a small LLaMA draft model. Each optimization compounds — FP8 + speculation + multi-replica together can deliver dramatic throughput improvements.',
    ],
    successExplanation:
      'Production-grade LLM serving is the composition of many techniques, each addressing a different ' +
      'bottleneck. Quantization reduces memory and bandwidth demand. `TP` enables serving models larger than ' +
      'one GPU. Multi-replica scales throughput linearly. Paged attention improves memory utilization for ' +
      'larger batches. Speculative decoding accelerates autoregressive generation. Continuous batching ' +
      'maximizes GPU utilization across requests.\n\nIn real deployments, frameworks like vLLM, TensorRT-LLM, ' +
      'and SGLang orchestrate all of these techniques together, often automatically tuning parameters like ' +
      '`TP` degree and batch size based on the hardware and model.',
  },
];
