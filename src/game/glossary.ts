/**
 * Glossary term definitions for tooltip annotations in task text.
 *
 * Usage in task strings:
 *   {{termId}}              → renders glossary display with tooltip
 *   {{termId|custom text}}  → renders custom text with tooltip
 */

export interface GlossaryEntry {
  display: string;
  definition: string;
}

export const GLOSSARY: Record<string, GlossaryEntry> = {
  // ── Core Metrics ──────────────────────────────────────────────────────
  mfu: {
    display: 'MFU',
    definition:
      'Model FLOPs Utilization — fraction of GPU peak compute used for useful model computation. Higher is better.',
  },
  hfu: {
    display: 'HFU',
    definition:
      'Hardware FLOPs Utilization — includes activation recompute work. HFU > MFU when checkpointing is enabled.',
  },
  tflops: {
    display: 'TFLOPS',
    definition:
      'Tera Floating-Point Operations Per Second — 10¹² FLOPs/s. Measures GPU compute throughput.',
  },
  oom: {
    display: 'OOM',
    definition:
      'Out of Memory — GPU memory demand exceeds capacity, crashing the job.',
  },

  // ── Precision Formats ─────────────────────────────────────────────────
  fp32: {
    display: 'FP32',
    definition:
      '32-bit floating point. 4 bytes per value. Used for optimizer states and master weights.',
  },
  fp16: {
    display: 'FP16',
    definition:
      '16-bit floating point. 2 bytes per value. Narrower exponent range than BF16 — requires loss scaling for stable training.',
  },
  bf16: {
    display: 'BF16',
    definition:
      'Brain Float 16 — same exponent range as FP32, fewer mantissa bits. 2 bytes. Standard for LLM training.',
  },
  fp8: {
    display: 'FP8',
    definition:
      '8-bit floating point. 1 byte per value. Requires Hopper (H100+) GPUs. Doubles peak TFLOPS vs BF16.',
  },
  int8: {
    display: 'INT8',
    definition:
      '8-bit integer quantization. 1 byte per value. Common for inference weight compression.',
  },
  int4: {
    display: 'INT4',
    definition:
      '4-bit integer quantization. 0.5 bytes per value. Aggressive compression for inference.',
  },
  nf4: {
    display: 'NF4',
    definition:
      '4-bit NormalFloat — quantization format optimized for normally-distributed weights. Used by QLoRA. ~0.5 bytes/param.',
  },
  'mixed-precision': {
    display: 'mixed precision',
    definition:
      'Training with lower precision (BF16/FP8) for speed while keeping master weights and critical accumulations in FP32.',
  },
  quantization: {
    display: 'quantization',
    definition:
      'Reducing numerical precision of model weights or activations to use less memory and bandwidth. Common formats: INT8, INT4, NF4.',
  },

  // ── Hardware ──────────────────────────────────────────────────────────
  nvlink: {
    display: 'NVLink',
    definition:
      'High-bandwidth GPU-to-GPU interconnect within a node. Bandwidth varies by generation: 300 GB/s (V100) to 1800 GB/s (B200) bidirectional.',
  },
  infiniband: {
    display: 'InfiniBand',
    definition:
      'High-speed network fabric for connecting GPU nodes. One of several options alongside RoCEv2 and proprietary fabrics. Bandwidth varies: HDR 200 GB/s, NDR 400 GB/s, XDR 800 GB/s per node.',
  },
  pcie: {
    display: 'PCIe',
    definition:
      'PCI Express — standard bus connecting GPUs to CPUs. Much slower than NVLink (~64 GB/s for PCIe 4.0).',
  },
  hbm: {
    display: 'HBM',
    definition:
      'High Bandwidth Memory — stacked DRAM on the GPU package. Provides the GPU\'s main memory (e.g., 80 GB on H100).',
  },
  sram: {
    display: 'SRAM',
    definition:
      'Static RAM — small, fast on-chip cache in each streaming multiprocessor. Orders of magnitude faster than HBM but much smaller.',
  },
  'cuda-cores': {
    display: 'CUDA cores',
    definition:
      'Standard GPU processing units for general-purpose computation. Much slower than Tensor Cores for matrix operations used in deep learning.',
  },
  'tensor-cores': {
    display: 'Tensor Cores',
    definition:
      'Specialized GPU hardware units for matrix multiply-accumulate. Much faster than general CUDA cores for deep learning.',
  },
  'transformer-engine': {
    display: 'Transformer Engine',
    definition:
      'NVIDIA library for FP8 mixed precision on Hopper+ GPUs. Handles per-tensor dynamic scaling for numerical stability.',
  },
  h100: {
    display: 'H100',
    definition:
      'NVIDIA H100 SXM — Hopper generation. 80 GB HBM3, 989 BF16 TFLOPS, 1979 FP8 TFLOPS.',
  },
  h800: {
    display: 'H800',
    definition:
      'NVIDIA H800 SXM — H100 variant with reduced NVLink (400 GB/s vs 900 GB/s). Same compute.',
  },
  a100: {
    display: 'A100',
    definition:
      'NVIDIA A100 — Ampere generation. 80 GB HBM2e, 312 BF16 TFLOPS.',
  },
  l40s: {
    display: 'L40S',
    definition:
      'NVIDIA L40S — Ada Lovelace. 48 GB GDDR6, 362 BF16 TFLOPS. PCIe only (no NVLink).',
  },
  'rtx-4090': {
    display: 'RTX 4090',
    definition:
      'NVIDIA GeForce RTX 4090 — Ada Lovelace consumer GPU. 24 GB GDDR6X, 330 BF16 TFLOPS. PCIe only.',
  },
  'rtx-3090': {
    display: 'RTX 3090',
    definition:
      'NVIDIA GeForce RTX 3090 — Ampere consumer GPU. 24 GB GDDR6X, 71 BF16 TFLOPS. PCIe only.',
  },
  l4: {
    display: 'L4',
    definition:
      'NVIDIA L4 — Ada Lovelace inference GPU. 24 GB GDDR6, 121 BF16 TFLOPS. PCIe only, low power (72W).',
  },
  t4: {
    display: 'T4',
    definition:
      'NVIDIA T4 — Turing generation inference GPU. 16 GB GDDR6, 320 GB/s bandwidth. No BF16/FP8 support.',
  },
  a10g: {
    display: 'A10G',
    definition:
      'NVIDIA A10G — Ampere inference GPU. 24 GB GDDR6, 70 BF16 TFLOPS. Popular on AWS for inference workloads.',
  },

  // ── Parallelism Strategies ────────────────────────────────────────────
  dp: {
    display: 'DP',
    definition:
      'Data Parallelism — replicating the model across GPUs, each processing different data. Gradients are synchronized after each step.',
  },
  ddp: {
    display: 'DDP',
    definition:
      'Distributed Data Parallel — DP with AllReduce gradient sync. Every GPU holds a full model copy.',
  },
  fsdp: {
    display: 'FSDP',
    definition:
      'Fully Sharded Data Parallel — shards parameters, gradients, and optimizer states across GPUs. Each GPU holds 1/N of the state.',
  },
  tp: {
    display: 'TP',
    definition:
      'Tensor Parallelism — splits each layer\'s weight matrices across GPUs. Requires an AllReduce after every layer. Best over fast intra-node interconnects (NVLink, Infinity Fabric).',
  },
  pp: {
    display: 'PP',
    definition:
      'Pipeline Parallelism — splits model layers into stages across GPU groups. Stages pass activations forward and gradients backward.',
  },
  sp: {
    display: 'SP',
    definition:
      'Sequence Parallelism — partitions activations along the sequence dimension for LayerNorm/dropout. Reduces activation memory to 1/TP per rank.',
  },
  cp: {
    display: 'CP',
    definition:
      'Context Parallelism — splits long sequences across GPUs via ring attention. Each GPU processes a chunk, exchanging KV blocks. For 32K+ sequences.',
  },
  ep: {
    display: 'EP',
    definition:
      'Expert Parallelism — distributes MoE experts across GPUs. Each GPU holds a subset. Requires All-to-All communication.',
  },
  zero1: {
    display: 'ZeRO-1',
    definition:
      'ZeRO Stage 1 — shards optimizer states across GPUs but replicates parameters and gradients. Lower communication than ZeRO-3 but higher memory per GPU.',
  },
  zero3: {
    display: 'ZeRO-3',
    definition:
      'ZeRO Stage 3 (DeepSpeed) — equivalent to FSDP. Shards parameters, gradients, and optimizer states across all GPUs.',
  },
  '3d-parallel': {
    display: '3D parallelism',
    definition:
      'Combining TP (intra-node, fast interconnect) + PP (cross-node, pipelined) + DP (throughput scaling). Standard recipe for 100B+ models.',
  },
  '2d-parallel': {
    display: '2D parallelism',
    definition:
      'Combining two parallelism dimensions — typically FSDP (inter-node) + TP (intra-node). Standard approach for clusters up to a few hundred GPUs.',
  },

  // ── Communication Operations ──────────────────────────────────────────
  allreduce: {
    display: 'AllReduce',
    definition:
      'Collective: sums tensors across all GPUs and distributes the result back to each. Used for gradient sync in DDP and weight sync in TP.',
  },
  allgather: {
    display: 'AllGather',
    definition:
      'Collective: each GPU contributes its shard and receives the full tensor. FSDP uses AllGather to reconstruct weights before computation.',
  },
  reducescatter: {
    display: 'ReduceScatter',
    definition:
      'Collective: reduces (sums) across GPUs and scatters so each gets one shard. FSDP uses it for gradient sharding.',
  },
  'all-to-all': {
    display: 'All-to-All',
    definition:
      'Each GPU sends different data to every other GPU. Used by Expert Parallelism to route tokens to the correct expert GPU.',
  },
  'ring-allreduce': {
    display: 'ring AllReduce',
    definition:
      'Efficient AllReduce where GPUs form a ring, each sending to its neighbor. Bandwidth-optimal for large messages.',
  },

  // ── Training Concepts ─────────────────────────────────────────────────
  'activation-memory': {
    display: 'activation memory',
    definition:
      'Intermediate tensors from the forward pass needed for backpropagation. Scales with batch size, sequence length, and model width.',
  },
  'activation-checkpointing': {
    display: 'activation checkpointing',
    definition:
      'Discards forward-pass activations and recomputes them during backpropagation. Trades ~33% more compute for dramatically less memory.',
  },
  'selective-checkpointing': {
    display: 'selective checkpointing',
    definition:
      'Discards only attention activations (cheap to recompute with Flash Attention), keeps MLP activations. Less overhead than full checkpointing.',
  },
  'gradient-accumulation': {
    display: 'gradient accumulation',
    definition:
      'Processing multiple micro-batches before synchronizing gradients. Increases effective batch size without increasing per-GPU memory.',
  },
  mbs: {
    display: 'micro-batch size',
    definition:
      'Samples per GPU per forward pass. Larger MBS = better GPU utilization but more activation memory.',
  },
  gbs: {
    display: 'global batch size',
    definition:
      'Total samples across all GPUs per optimization step. GBS = MBS × DP × GA.',
  },
  'forward-pass': {
    display: 'forward pass',
    definition:
      'Computing model outputs by passing input through all layers sequentially. Stores intermediate activations needed for the backward pass.',
  },
  'backward-pass': {
    display: 'backward pass',
    definition:
      'Computing gradients by propagating loss backward through the network. Requires stored activations from the forward pass.',
  },
  gradients: {
    display: 'gradients',
    definition:
      'Derivatives of the loss with respect to each parameter. Computed in the backward pass and used by the optimizer to update weights.',
  },
  'optimizer-states': {
    display: 'optimizer states',
    definition:
      'Per-parameter state maintained by the optimizer. Adam stores momentum + variance: 8 extra bytes per parameter.',
  },
  'backward-prefetch': {
    display: 'backward prefetch',
    definition:
      'FSDP optimization: prefetching the next layer\'s parameters during backward computation, overlapping AllGather with compute.',
  },
  mlp: {
    display: 'MLP',
    definition:
      'Multi-Layer Perceptron — the feed-forward network in each transformer layer. Typically two or three linear layers with an activation function.',
  },
  'flash-attention': {
    display: 'Flash Attention',
    definition:
      'IO-aware attention algorithm using tiling to minimize HBM reads/writes. Reduces attention memory from `O(N²)` to `O(N)`.',
  },
  sharding: {
    display: 'sharding',
    definition:
      'Splitting data across multiple GPUs so each holds a fraction. FSDP shards model state; EP shards experts.',
  },

  // ── Pipeline Parallelism Concepts ─────────────────────────────────────
  'pipeline-bubble': {
    display: 'pipeline bubble',
    definition:
      'Idle time when pipeline stages wait for data. Fraction: (PP-1)/(PP-1+m), where m = number of microbatches.',
  },
  'pipeline-stage': {
    display: 'pipeline stage',
    definition:
      'A group of consecutive model layers assigned to GPUs in pipeline parallelism. Stages pass activations and gradients between each other.',
  },
  '1f1b': {
    display: '1F1B',
    definition:
      'One-Forward-One-Backward — pipeline schedule alternating forward/backward micro-batches to minimize in-flight activations.',
  },
  'interleaved-1f1b': {
    display: 'interleaved 1F1B',
    definition:
      'Pipeline schedule with virtual stages: each GPU runs v non-contiguous layer chunks, shrinking the bubble by factor v.',
  },
  'virtual-stages': {
    display: 'virtual stages',
    definition:
      'In interleaved scheduling, each GPU handles multiple non-contiguous layer chunks. More virtual stages = smaller bubble but more communication.',
  },
  microbatches: {
    display: 'microbatches',
    definition:
      'Small sub-batches processed sequentially through the pipeline. More microbatches = smaller pipeline bubble. Equal to gradient accumulation steps.',
  },

  // ── Attention & Sequence ──────────────────────────────────────────────
  'self-attention': {
    display: 'self-attention',
    definition:
      'Core transformer operation: each token attends to all others. `Scores = softmax(QK^T/√d)·V`. Compute and memory scale quadratically with sequence length.',
  },
  'attention-matrix': {
    display: 'attention matrix',
    definition:
      'The `N × N` matrix of attention scores between all token pairs. Standard computation requires `O(N²)` memory.',
  },
  'sequence-length': {
    display: 'sequence length',
    definition:
      'Number of tokens in the input. Affects attention memory (quadratic) and activation memory (linear). Training: 2K-8K typical, up to 128K for long-context.',
  },
  layernorm: {
    display: 'LayerNorm',
    definition:
      'Layer Normalization — normalizes activations across the feature dimension. Replicated across TP ranks unless SP is enabled.',
  },
  dropout: {
    display: 'dropout',
    definition:
      'Regularization that randomly zeros activations during training. Replicated across TP ranks unless SP is enabled.',
  },
  gqa: {
    display: 'GQA',
    definition:
      'Grouped Query Attention — fewer KV heads than query heads, reducing KV cache memory. Used in LLaMA 2 70B+, LLaMA 3, etc.',
  },

  // ── MoE ───────────────────────────────────────────────────────────────
  moe: {
    display: 'MoE',
    definition:
      'Mixture of Experts — multiple parallel FFN blocks per layer with a router selecting k active experts per token. High capacity, sparse compute.',
  },
  router: {
    display: 'router',
    definition:
      'Gating network in MoE that decides which experts process each token. Outputs top-k routing weights.',
  },
  'active-params': {
    display: 'active parameters',
    definition:
      'Parameters used per forward pass. In MoE, only k/N experts are active per token — MFU is computed against active params.',
  },
  'device-limited-routing': {
    display: 'device-limited routing',
    definition:
      'MoE routing constraint limiting how many devices each token can be sent to, reducing All-to-All communication. DeepSeek V3: M=4.',
  },

  // ── Fine-tuning ───────────────────────────────────────────────────────
  lora: {
    display: 'LoRA',
    definition:
      'Low-Rank Adaptation — freezes base weights, trains small adapter matrices (A, B) at each layer. Trainable params: ~0.1-3% of total.',
  },
  qlora: {
    display: 'QLoRA',
    definition:
      'Quantized LoRA — stores frozen base model in NF4 (4-bit), trains adapters in BF16. Enables fine-tuning 70B+ models on consumer GPUs.',
  },
  'fine-tuning': {
    display: 'fine-tuning',
    definition:
      'Adapting a pre-trained model to a specific task by continuing training on new data. Full (all params) or parameter-efficient (LoRA/QLoRA).',
  },
  'adapter-matrices': {
    display: 'adapter matrices',
    definition:
      'Small trainable matrices in LoRA: A (d×r) and B (r×d), where rank r << d. Injected alongside frozen weights.',
  },

  // ── Inference ─────────────────────────────────────────────────────────
  'arithmetic-intensity': {
    display: 'arithmetic intensity',
    definition:
      'Ratio of compute operations (FLOPs) to bytes transferred from memory. Low intensity (like decode) means the GPU is bottlenecked by memory bandwidth, not compute.',
  },
  'kv-cache': {
    display: 'KV cache',
    definition:
      'Stores previously computed Key/Value tensors for all past tokens, avoiding recomputation during autoregressive generation. Memory scales with sequence × batch.',
  },
  ttft: {
    display: 'TTFT',
    definition:
      'Time to First Token — latency from prompt to first output token. Dominated by prefill (prompt processing). Compute-bound.',
  },
  tpot: {
    display: 'TPOT',
    definition:
      'Time Per Output Token — latency between consecutive generated tokens. Dominated by weight loading. Memory-bandwidth-bound.',
  },
  prefill: {
    display: 'prefill',
    definition:
      'First inference phase: processing the entire input prompt to populate KV cache. Compute-bound — benefits from parallelism.',
  },
  decode: {
    display: 'decode',
    definition:
      'Second inference phase: generating tokens one at a time, reading all weights each step. Memory-bandwidth-bound — benefits from quantization.',
  },
  'continuous-batching': {
    display: 'continuous batching',
    definition:
      'Inserting new requests as slots free up rather than waiting for the full batch. Maximizes GPU utilization for serving.',
  },
  'paged-attention': {
    display: 'PagedAttention',
    definition:
      'KV cache memory management inspired by OS virtual memory paging. Eliminates fragmentation from pre-allocated contiguous buffers.',
  },
  'speculative-decoding': {
    display: 'speculative decoding',
    definition:
      'A small draft model generates candidate tokens, then the large model verifies them in parallel. Correct tokens are accepted free.',
  },
  'ring-attention': {
    display: 'ring attention',
    definition:
      'CP implementation: GPUs form a ring, passing KV blocks to neighbors while computing attention on local chunks. Overlaps communication with compute.',
  },
};
