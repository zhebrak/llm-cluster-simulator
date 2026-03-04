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
      'You have 8 H100 GPUs — configure {{tp|tensor parallelism}} and quantization to serve this model ' +
      'with at least 250 tokens per second.',
    concept: 'MoE Inference at Scale',
    learningObjectives: [
      'Understand `MoE` memory dominance: 671B params all must be resident even though only ~37B are active',
      'Know `INT4` is often necessary for single-node `MoE` inference (`FP8` at `TP`=8 barely exceeds 80 GB/GPU)',
      'Calculate per-GPU memory at various precisions with `TP` to find feasible configurations',
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
        value: 250,
        label: 'Throughput > 250 tok/s',
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
      'With 8 H100s (80GB each), calculate whether `FP8` (1 byte/param) fits per GPU at `TP=8`. If not, try `INT4` (0.5 bytes/param) which further halves memory.',
      'Use `TP=8` to shard the model across all 8 GPUs. {{nvlink|NVLink}} provides high-bandwidth communication within a single node, making `TP=8` efficient here.',
    ],
    successExplanation:
      '`MoE` models like DeepSeek V3 have a unique inference characteristic: memory is dominated by ' +
      'the total parameter count (all experts must be resident), but compute scales with active parameters only. ' +
      'This means `MoE` models are more memory-bound during decode than their dense counterparts of the same ' +
      'active size.\n\nAggressive quantization is often necessary for `MoE` inference on a single node — ' +
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
      '{{ep|Expert Parallelism (EP)}} is a technique specific to `MoE` models that distributes experts ' +
      'across GPUs rather than replicating them. Instead of every GPU holding all 256 experts, ' +
      'each GPU holds a subset, and tokens are routed to the appropriate GPU via {{all-to-all}} communication. ' +
      'You have 16 H100 GPUs (2 nodes) serving DeepSeek V3 in `FP8`. Configure `EP` to efficiently distribute ' +
      'the expert parameters while keeping tensor parallelism for the dense attention layers. ' +
      'The goal is to achieve higher throughput than pure `TP` alone by reducing per-GPU memory pressure ' +
      'and enabling larger batch sizes.',
    concept: 'Scaling MoE beyond single-node memory',
    learningObjectives: [
      'Understand `EP` distributes experts across GPUs instead of replicating all on each',
      'Know `EP` uses all-to-all communication (different from `TP` `AllReduce`)',
      'Understand device routing limits (M=4 for DeepSeek V3) bound all-to-all volume',
      'Know `EP` frees per-GPU memory, enabling larger batches and higher throughput',
    ],
    setup: {
      modelId: 'deepseek-v3',
      gpuId: 'h100-sxm',
      numGPUs: 16,
      gpusPerNode: 8,
      weightPrecision: 'fp8',
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
      'Expert Parallelism distributes experts across GPUs. With `EP`, each GPU holds a fraction of the experts, dramatically reducing per-GPU memory.',
      'Combine `TP` and `EP`: use `TP` within a node (`TP=8` over `NVLink`) and `EP` across nodes. This keeps fast `NVLink` for dense `AllReduce` and uses the inter-node fabric for expert routing.',
      'With less memory consumed by expert weights, you can increase batch size to improve throughput. `EP` trades all-to-all communication for better memory efficiency and higher batch utilization.',
    ],
    successExplanation:
      '`EP` is the key to scaling `MoE` inference beyond a single node. By distributing ' +
      'experts across GPUs, each GPU needs far less memory for expert weights, freeing capacity for larger ' +
      '`KV` caches and batch sizes.\n\nThe all-to-all communication pattern is different from `TP`\'s `AllReduce`: ' +
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
      'You have LLaMA 3.1 405B in `FP8` on 16 H100 GPUs. Enable speculative decoding, choose a draft model, and ' +
      'tune K to achieve both low latency and good throughput.',
    concept: 'Draft-verify optimization and tuning',
    learningObjectives: [
      'Choose appropriate draft model: 10-100x smaller than target, same model family for high acceptance',
      'Understand quality-speed tradeoff: larger drafts accept more but have more overhead',
      'Know E[tokens/step] = (1 - alpha^(K+1)) / (1 - alpha) where alpha = acceptance rate',
      'Understand optimal K depends on the ratio of draft generation time to target verification time',
      'Know speculative decoding is especially effective for `MoE` models where all expert weights are loaded but few are used — verification amortizes this bandwidth waste',
    ],
    setup: {
      modelId: 'llama3-405b',
      gpuId: 'h100-sxm',
      numGPUs: 16,
      gpusPerNode: 8,
      weightPrecision: 'fp8',
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Model fits in memory' },
      { field: 'latency.tpot', operator: '<', value: 12, label: 'TPOT < 12 ms' },
      { field: 'throughput.tokensPerSecond', operator: '>', value: 120, label: 'Throughput > 120 tok/s' },
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
      'Speculative decoding is especially powerful for `MoE` models like DeepSeek V3. During standard decode, ' +
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
      'Understand that prefill is compute-bound: `TTFT` scales with model_FLOPs × input_tokens / (`TFLOPS` × `TP`)',
      'Know that increasing `TP` divides prefill compute across GPUs, reducing `TTFT` nearly linearly',
      'Recognize that `TTFT` optimization (max `TP`) conflicts with throughput optimization (min `TP`, max replicas)',
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
      'Prefill is compute-bound: dominated by matrix multiplications through every layer. Unlike decode (which reads weights once per token), prefill multiplies weights by thousands of input tokens simultaneously. More `TP` means more GPUs share this compute.',
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
      'LLaMA 3.3 70B on 8 H100 GPUs with `TP=4`. At `BF16`, each GPU holds ~35 GB of weights — it fits, ' +
      'but all 8 GPUs are dedicated to a single instance. With `TP=4`, 4 GPUs are used per replica and ' +
      'the other 4 sit idle. But reducing `TP` below 4 at `BF16` would not fit the 140 GB model.\n\n' +
      'The key insight: quantization reduces weight memory, enabling lower `TP`. Lower `TP` frees GPUs ' +
      'for more replicas, and each replica provides linear throughput scaling with zero communication overhead.\n\n' +
      'Find a configuration that pushes total throughput above 260 tok/s.',
    concept: 'Throughput scaling via instance replication',
    learningObjectives: [
      'Know replicas provide linear throughput scaling with no communication overhead',
      'Understand `FP8` quantization enables lower `TP`, which enables more replicas',
      'Distinguish latency-oriented (max `TP`, batch=1) from throughput-oriented (min `TP`, max replicas) configs',
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
        value: 260,
        label: 'Throughput > 260 tok/s',
      },
    ],
    expectedChanges: [
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'At `BF16` with `TP=4`, the model fits but only 2 replicas run. Think about what levers affect total throughput — replica count, batch size, and weight precision all play a role.',
      'Quantization frees weight memory: lower `TP` creates more replicas, each adding linear throughput. Alternatively, even at `TP=4`, adjusting batch size can push total throughput above the target with 2 replicas.',
    ],
    successExplanation:
      'With 8 GPUs and `TP=4`, two replicas serve in parallel. There are multiple ways to push throughput above the target.\n\n' +
      '`FP8` halves weight memory, enabling `TP`=2 — that means 4 replicas ' +
      'instead of 2, nearly doubling throughput with zero communication overhead. This is the throughput-optimal ' +
      'strategy when you have more GPUs than a single instance needs.\n\n' +
      'Adjusting batch size at `TP=4` amortizes weight reads across more sequences per decode step. ' +
      'Fewer replicas, but each is more efficient per step.\n\n' +
      'In production, minimizing `TP` and maximizing replicas generally wins for high-QPS workloads. Serving frameworks ' +
      'like vLLM and TensorRT-LLM automatically manage multi-replica deployments.',
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
      'The cost per million tokens is determined by: `(GPU_hourly_rate × num_GPUs) / tokens_per_second × 1e6 / 3600`. ' +
      'Every optimization that increases throughput directly reduces cost. But there is a subtlety: ' +
      'using fewer GPUs reduces the numerator, and sometimes a smaller but well-optimized configuration ' +
      'beats a larger one on cost even if absolute throughput is lower.\n\n' +
      'You have 8 {{l40s|L40S}} GPUs — significantly cheaper per hour than A100s or H100s. ' +
      'Each L40S has 48 GB of memory with `PCIe` connectivity. You\'re serving LLaMA 3.3 70B in `FP8`. ' +
      'Minimize the cost per token.',
    concept: 'Cost-Efficient Inference Optimization',
    learningObjectives: [
      'Know cost formula: `(GPU_hourly_rate × numGPUs / 3600) / tokens_per_sec × 1e6`',
      'Understand throughput is the primary cost lever — 2x throughput = 0.5x cost',
      'Combine multi-replica serving + batching + optimal `TP` degree for cost-efficient production serving',
    ],
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'l40s',
      numGPUs: 8,
      gpusPerNode: 8,
      weightPrecision: 'fp8',
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
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'Cost = `(rate × numGPUs / 3600) / tokensPerSecond × 1e6`. The `L40S` is generally cheaper per hour than an `A100` or `H100`, so even moderate throughput can achieve good cost efficiency.',
      'On `PCIe` GPUs, `TP` communication is slower than on high-bandwidth intra-node links. What is the tradeoff between fewer, faster replicas (higher `TP`) vs. more, slower ones (lower `TP`)?',
      'Minimizing `TP` gives more replicas — but each replica is slower. Maximizing `TP` makes each replica fast — but you have fewer. On `PCIe`, the communication penalty for high `TP` is steep. Find the `TP` degree that maximizes total throughput per dollar. The Pareto frontier chart shows cost-latency tradeoffs across configurations.',
    ],
    successExplanation:
      'Cost-per-token optimization is a holistic challenge. ' +
      'Cheaper GPUs can offer compelling price-performance ratios: with a lower hourly rate, ' +
      'even moderate throughput can achieve excellent cost efficiency.\n\n' +
      'The key techniques are: multi-replica serving (linear throughput scaling) ' +
      'and batching (amortize bandwidth across requests). ' +
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
      'with sequence length, and attention computation grows quadratically. At 128K tokens, the `KV` cache ' +
      'alone can consume tens of gigabytes per request, severely limiting batch size and throughput. ' +
      'LLaMA 3.1 405B supports up to 128K context. Configure a 16-GPU H100 setup to handle long-context ' +
      'inference. With weights in `FP8`, the main memory challenge is the `KV` cache — at 128K tokens it can ' +
      'consume tens of gigabytes per GPU. `KV` cache quantization becomes essential. ' +
      '{{paged-attention|Paged attention}} is already active — it eliminates `KV` cache fragmentation so memory scales with actual usage, not worst-case allocation.',
    concept: 'Long-Context Inference Memory Management',
    learningObjectives: [
      'Understand `KV` cache dominates memory at 128K+ context (can exceed weight memory per request)',
      'Know `KV` cache quantization (`FP8`) halves `KV` memory — visible as a significant drop in memory utilization',
      'Recognize prefix caching as a technique for shared system prompts, reusing `KV` cache across requests',
    ],
    setup: {
      modelId: 'llama3-405b',
      gpuId: 'h100-sxm',
      numGPUs: 16,
      gpusPerNode: 8,
      inputSeqLen: 128000,
      weightPrecision: 'fp8',
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
        value: 0.75,
        label: 'Memory utilization < 75%',
      },
    ],
    expectedChanges: [
      { field: 'kvCachePrecision', check: 'changed', label: 'Changed KV cache precision' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
      { field: 'inputSeqLen', check: 'unchanged', label: 'Did not change input sequence length' },
    ],
    hints: [
      'At 128K sequence length, the `KV` cache for a 405B model is enormous — even with {{gqa|GQA}} reducing `KV` heads. With weights already in `FP8`, the `KV` cache becomes the dominant memory consumer per GPU. Run the config with `TP=8` or `TP=16` and watch the memory utilization bar — it needs to drop below 75%.',
      'Look for the `KV` cache precision selector in the sidebar. Quantizing the `KV` cache from `BF16` to `FP8` halves its memory footprint — at 128K context, that visibly drops the memory bar. This is the key lever for long-context serving.',
    ],
    successExplanation:
      'Long-context inference is memory-dominated. For LLaMA 3.1 405B at 128K context, the `KV` cache per ' +
      'request is approximately: `2 (K+V) * 128 (layers) * 8 (KV heads) * 128 (head_dim) * 128000 * bytes`. ' +
      'Even with `GQA` reducing `KV` heads from 128 to 8, this is massive.\n\nWith weight quantization (`FP8`) and ' +
      '`TP` already configured, the remaining challenge is `KV` cache management: (1) `KV` cache quantization to `FP8`, ' +
      '(2) paged attention for efficient memory utilization, and sometimes (3) prefix caching — when many requests share the same system prompt, the `KV` cache ' +
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
      'Find the batch sweet spot where all three SLA constraints are simultaneously satisfied',
      'Recognize production serving requires satisfying multiple SLA dimensions at once',
    ],
    briefing:
      'Production chatbot serving requires meeting multiple SLA constraints simultaneously. ' +
      'Users expect a fast first response (`TTFT` under 500 ms), smooth token streaming ' +
      '(`TPOT` under 10 ms), and the system must handle high request volume ' +
      '(throughput over 500 tokens per second).\n\n' +
      'You have LLaMA 3.3 70B in `FP8` on 8 H100 GPUs with `TP=8`, `batch=24`, and 4K input sequences. ' +
      'Run it and observe the metrics. The three constraints pull in different directions — ' +
      'find the configuration that satisfies all three constraints at once.',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'h100-sxm',
      numGPUs: 8,
      gpusPerNode: 8,
      tensorParallel: 8,
      batchSize: 24,
      inputSeqLen: 4096,
      weightPrecision: 'fp8',
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Inference succeeds' },
      { field: 'latency.ttft', operator: '<', value: 500, label: 'TTFT < 500 ms' },
      { field: 'latency.tpot', operator: '<', value: 10, label: 'TPOT < 10 ms' },
      { field: 'throughput.tokensPerSecond', operator: '>', value: 500, label: 'Throughput > 500 tok/s' },
    ],
    expectedChanges: [
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'Run the initial config and check all three metrics. `TPOT` may look fine, but `TTFT` exceeds the SLA. Think about what drives `TTFT` — it depends on prefill compute, which scales with batch size and input length.',
      'Reducing batch size helps `TTFT` but hurts throughput. Increasing batch helps throughput but worsens `TPOT` and `TTFT`. The feasible region where all three constraints are met may be narrow.',
      '`TTFT` scales with `batch_size × input_tokens` (total prefill compute). `TPOT` scales with step time (which grows with batch). `Throughput = batch / step_time`. Look for the batch size that threads the needle. The Pareto frontier chart below shows cost-latency tradeoffs across configurations.',
    ],
    successExplanation:
      'Production LLM serving is a multi-constraint optimization problem. `TTFT` is compute-bound ' +
      '(proportional to input length and model FLOPs), `TPOT` is bandwidth-bound (proportional to ' +
      'weight bytes read per step), and throughput depends on batch utilization.\n\n' +
      'The key insight: these constraints interact through batch size. Higher batch amortizes weight ' +
      'reads across more tokens (higher throughput) but each decode step takes longer (higher `TPOT`), ' +
      'and prefill must process more concurrent sequences (higher `TTFT`). The art of SLA optimization ' +
      'is finding the batch sweet spot where all constraints intersect.\n\n' +
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
      'Know 47B total params at BF16 (94 GB) needs multi-GPU sharding: `TP` splits all weights, `EP` splits experts across GPUs',
      'Recognize quantization as the key enabler — reducing bytes/param frees memory for KV cache and batching',
      'Understand MoE throughput: fewer active FLOPs per token means faster generation than an equivalently-sized dense model',
    ],
    briefing:
      'Mixtral 8x7B is the most popular `MoE` model for local inference. It has 47B total ' +
      'parameters but only activates ~13B per token (top-2 of 8 experts). All 47B parameters ' +
      'must reside in GPU memory because routing is input-dependent — any expert might be selected.\n\n' +
      'You have 4 {{rtx-4090|RTX 4090}} GPUs with 24 GB each (96 GB total). At BF16, Mixtral needs ' +
      '~94 GB for weights alone — far too much for a single GPU. You can shard with `TP` (split all ' +
      'weights) or `EP` (split experts across GPUs). Either way, quantization is essential for ' +
      'comfortable memory headroom.\n\n' +
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
      { field: 'memoryUtilization', operator: '<', value: 0.90, label: 'Memory utilization < 90%' },
    ],
    expectedChanges: [
      { field: 'weightPrecision', check: 'changed', label: 'Changed weight precision' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU type' },
    ],
    hints: [
      'At `BF16` (2 bytes/param), Mixtral\'s 47B total parameters need ~94 GB — far more than a single 24 GB GPU. You need quantization AND multi-GPU sharding. Two sharding options: `TP` splits all weight matrices, `EP` places different experts on different GPUs.',
      '`INT8` (1 byte/param) roughly halves weight memory. `INT4` (0.5 bytes/param) gives even more headroom. Calculate per-GPU weight size: `total_params × bytes_per_param ÷ sharding_degree`. With `TP=4`, every weight is split 4 ways. With `EP=4`, each GPU holds 2 of 8 experts plus the shared layers.',
      'Despite having 47B total parameters, Mixtral computes with only ~13B active per token — so each decode step does fewer FLOPs than a 47B dense model. The bandwidth cost is still the full weight loading (all experts must be resident), but the compute is efficient.',
    ],
    successExplanation:
      'MoE models on consumer hardware highlight the memory-compute decoupling. Mixtral\'s 47B ' +
      'parameters all reside in GPU memory, but the router only activates 2 of 8 experts per ' +
      'token (~13B active). Memory is sized by total params (47B), requiring quantization on ' +
      '24 GB GPUs.\n\n' +
      'Two sharding strategies work here: `TP` splits every weight matrix across all GPUs (each ' +
      'GPU holds 1/TP of every layer), while `EP` assigns different experts to different GPUs ' +
      '(each GPU holds all shared layers plus a subset of experts). On `PCIe`, both approaches ' +
      'incur communication overhead — `TP` needs `AllReduce` per layer, `EP` needs `AllToAll` for ' +
      'expert routing. The right choice depends on model architecture and interconnect bandwidth.\n\n' +
      'Despite the bandwidth overhead, Mixtral on 4 RTX 4090 GPUs with `INT4`/`INT8` quantization is one ' +
      'of the most popular local inference setups. Tools like llama.cpp, Ollama, and vLLM ' +
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
      'memory efficiency, {{speculative-decoding|speculative decoding}} for {{decode}} speedup, ' +
      'and multi-replica serving if possible. Each technique contributes to the final throughput number, and ' +
      'the art is in how they compose together. A well-optimized 405B serving setup should push significant ' +
      'throughput on 16 H100s.',
    concept: 'Full-Stack Inference Optimization',
    learningObjectives: [
      'Compose all techniques: `TP`, quantization, spec decoding, multi-replica, continuous batching',
      'Understand `TP=8` x 2 replicas can beat `TP=16` for throughput (avoid cross-node `TP` overhead)',
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
      'LLaMA 405B needs aggressive quantization to fit. Calculate per-GPU weight memory at `FP8` with `TP=8` on one node — check if it fits in 80GB with room for `KV` cache. Cross-node `TP` (`TP=16`) is another option but adds IB communication overhead.',
      'Consider `TP=8` with 2 replicas (one per node) instead of `TP=16`. Two replicas double throughput while avoiding cross-node `TP` overhead. Batch each replica independently for maximum utilization.',
      'Layer all optimizations: `FP8` weights, `FP8` `KV` cache, flash attention, continuous batching, and speculative decoding with a small LLaMA draft model. Each optimization compounds — `FP8` + speculation + multi-replica together can deliver dramatic throughput improvements.',
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
