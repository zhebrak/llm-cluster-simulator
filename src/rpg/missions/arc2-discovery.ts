/**
 * Arc 2: Discovery — the signal is real. Science teams mobilize.
 * Missions teach: TP for latency, replica scaling, speculative decoding,
 * mixed precision, FSDP, activation checkpointing, LoRA, FP8, network
 * topology, pipeline parallelism, and multi-objective optimization.
 *
 * DAG:
 *   INFERENCE BRANCH              TRAINING BRANCH
 *
 *   1-8 (Signal pivot)            1-6 (Archive Vault)
 *    │                              │
 *    ▼                              ▼
 *   2-1 (TP latency)             2-4 (BF16 + FSDP)
 *    ├──────────┐                   │
 *    ▼          ▼                   ▼
 *   2-2 (Derelict)  2-3 (Spec)  2-5 (Activation ckpt)
 *    │                              │
 *    │                              ▼
 *    ├──────────────────────────► 2-6 (LoRA)
 *    │
 *    │   HARDWARE/ADVANCED
 *    │
 *    │   1-7 (Fuel Budget)
 *    │    │
 *    ▼    ▼       ▼
 *    2-7 (Shipment) ← 2-4
 *         │
 *         ▼
 *    2-8 (Bandwidth Wall)
 *         │
 *         ▼
 *    2-9 (Pipeline)        ← side mission
 *
 *    2-10 (Protein/MFU) ← 2-5, 2-7
 *
 *    ⭐ 2-11 (Life) ← 2-7, 2-10
 */

import type { RPGArc, RPGMission } from '../types.ts';

export const ARC2: RPGArc = {
  id: 'arc2-discovery',
  name: 'Discovery',
  subtitle: 'A signal from Kepler-442b. Science doesn\'t wait',
  description: 'Analyze the alien signal. Train new models. Push the ship\'s compute to its limits.',
  order: 2,
  briefing: `The long-range scanner has detected a structured, repeating signal from Kepler-442b. Dr. Chen's team confirms it — encoded information, not cosmic noise. For the first time in 80 years, the ship has a reason to do more than survive.

Science teams are coming out of cryo rotation. Chen needs real-time spectral analysis. Okafor wants to fine-tune navigation models on new sensor data. Every department is suddenly demanding GPU time — and the compute bay you nursed through survival is now the bottleneck for discovery.

For the first time, the ship's models will need to train — not just on data from Earth, but on something no human has ever seen. The problems ahead are harder, the hardware demands are steeper, and there is no resupply coming. Somewhere ahead, something is broadcasting — and the Meridian needs to be ready.`,
  heroImage: { dark: '/signal_dark.png', light: '/signal_light.png' },
};

export const ARC2_MISSIONS: RPGMission[] = [
  // ── Mission 2-1: First Light ────────────────────────────────────────
  {
    id: 'mission-2-1',
    arcId: 'arc2-discovery',
    order: 1,
    title: 'First Light',
    subtitle: 'The signal demands speed',
    learningObjectives: [
      'Identify time-to-first-token as compute-bound during prefill',
      'Use tensor parallelism to reduce prefill latency by splitting compute across GPUs',
      'Understand that TP trades communication overhead for parallel compute speed',
    ],
    briefing: `Dr. Chen's team has decoded enough of the signal to know it's structured — possibly a language. But the analysis model needs to process spectral windows in near-real-time. Each window is brief. If the model can't respond before the next window arrives, they lose data forever.

The 70B model is loaded and running, but only two of the four available GPUs are doing any work. Chen watches the spectral windows slip by, then pulls up the latency trace. "Two GPUs idle. Two overloaded. We're losing data every cycle this stays like this."

Bring the time-to-first-token below the spectral window threshold. The model fits in memory. The hardware is there. The current configuration isn't using all the hardware available.`,
    successNarrative: `Spectral data streams through the model in real-time now. Chen's team watches as patterns emerge — repeating structures that look almost like syntax. "There," Chen says, pointing at the display. "Repeating subsequences. That's not noise." With all four GPUs splitting each layer's computation, prefill completes before the next window arrives.

Tensor parallelism trades communication for speed. Each layer requires an AllReduce to synchronize partial results across GPUs, but within a single node, the high-bandwidth interconnect makes this nearly free. For prefill-heavy workloads, TP is the most direct lever: double the GPUs sharing the work, roughly halve the time-to-first-token.`,
    primaryMode: 'inference',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'a100-80gb',
      numGPUs: 4,
      tensorParallel: 2,
      weightPrecision: 'int8',
      batchSize: 1,
      inputSeqLen: 1024,
      outputSeqLen: 512,
    },
    winningCriteria: [
      { field: 'latency.ttft', operator: '<', value: 200, label: 'TTFT < 200ms' },
    ],
    expectedChanges: [
      { field: 'tensorParallel', check: 'increased', label: 'Increased tensor parallelism' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU' },
      { field: 'numGPUs', check: 'unchanged', label: 'Did not change GPU count' },
      { field: 'weightPrecision', check: 'unchanged', label: 'Did not change precision' },
      { field: 'inputSeqLen', check: 'unchanged', label: 'Did not change input sequence length' },
      { field: 'outputSeqLen', check: 'unchanged', label: 'Did not change output sequence length' },
    ],
    hints: [
      'Prefill processes the entire input through every transformer layer before producing any output. It\'s compute-bound — the bottleneck is raw TFLOPS, not memory bandwidth. More GPUs sharing the matrix multiplications means faster prefill.',
      'Tensor parallelism splits each layer\'s weight matrices across GPUs. Every GPU computes a slice of the output in parallel. Doubling TP roughly halves the time each layer takes — at the cost of an AllReduce after every layer.',
      'You have 4 GPUs but only 2 are doing work. The Tensor Parallel setting controls how many GPUs split each layer. Consider what happens if you use all available GPUs.',
    ],
    prerequisites: ['mission-1-8'],
    skillsAwarded: ['model-parallelism'],
  },

  // ── Mission 2-2: The Derelict ───────────────────────────────────────
  {
    id: 'mission-2-2',
    arcId: 'arc2-discovery',
    order: 2,
    title: 'The Derelict',
    subtitle: 'An abandoned station holds forgotten hardware',
    learningObjectives: [
      'Understand the tradeoff between per-replica latency and number of concurrent replicas',
      'Calculate replica count as floor(totalGPUs / GPUsPerReplica)',
      'Choose a TP degree that balances latency and concurrency for multi-stream workloads',
    ],
    briefing: `An EVA team discovers a derelict relay station drifting near the ship. Inside: four more A100 GPUs, still functional after decades in vacuum. Chen needs three concurrent analysis streams — different spectral bands being decoded simultaneously.

Right now, all eight GPUs are yoked together serving a single stream. Fast — but only one at a time. When the second spectral band arrives, it queues. The third waits behind it. Chen is watching two-thirds of the incoming data queue up and go unanalyzed while the model processes one band at a time.

Three spectral bands, three simultaneous requests. The hardware is there. Find a configuration that serves all three concurrently without unacceptable latency on any one of them.`,
    successNarrative: `Three spectral bands stream in parallel. Each analysis pipeline runs on its own replica, fast enough to keep up with the incoming data. Chen's team can finally cross-reference patterns across frequency ranges simultaneously.

TP is a tradeoff between speed and concurrency. Higher TP means fewer, faster replicas. Lower TP means more replicas, each slightly slower. With 8 GPUs and TP=2, you get 4 replicas — more than enough for 3 concurrent streams, with latency still well within the window.

For throughput-sensitive multi-stream workloads, replicas beat raw single-stream speed.`,
    primaryMode: 'inference',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'a100-80gb',
      numGPUs: 8,
      tensorParallel: 8,
      weightPrecision: 'int8',
      batchSize: 1,
      inputSeqLen: 1024,
      outputSeqLen: 512,
    },
    winningCriteria: [
      { field: 'numReplicas', operator: '>=', value: 3, label: 'At least 3 analysis streams' },
      { field: 'latency.ttft', operator: '<', value: 300, label: 'TTFT < 300ms' },
    ],
    expectedChanges: [
      { field: 'tensorParallel', check: 'decreased', label: 'Reduced tensor parallelism' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU' },
      { field: 'numGPUs', check: 'unchanged', label: 'Did not change GPU count' },
      { field: 'inputSeqLen', check: 'unchanged', label: 'Did not change input sequence length' },
      { field: 'outputSeqLen', check: 'unchanged', label: 'Did not change output sequence length' },
    ],
    hints: [
      'Each replica is an independent copy of the model that can serve a separate request. The number of replicas is floor(total GPUs / GPUs per replica). With TP=8 on 8 GPUs, all eight GPUs are dedicated to a single model copy — one replica.',
      'There\'s a fundamental tradeoff: higher TP means faster single-stream latency (more GPUs per request), but fewer concurrent streams. Lower TP means more replicas but slower individual responses.',
      'If each replica needs fewer GPUs, you can fit more replicas. Consider what TP degree would give you at least 3 replicas while keeping latency acceptable.',
    ],
    prerequisites: ['mission-2-1'],
    skillsAwarded: ['data-parallelism'],
  },

  // ── Mission 2-3: Ghost Writer ───────────────────────────────────────
  {
    id: 'mission-2-3',
    arcId: 'arc2-discovery',
    order: 3,
    title: 'Ghost Writer',
    subtitle: 'A small mind drafts; a great mind verifies',
    learningObjectives: [
      'Understand why autoregressive decode is memory-bandwidth-bound for large models',
      'Explain how speculative decoding amortizes the cost of large-model forward passes',
      'Enable speculative decoding and interpret the speedup metric',
    ],
    briefing: `Chen's token-by-token signal decoding is working but agonizingly slow. The 70B model produces brilliant analysis — but watching it generate output is excruciating. Token. Pause. Token. Pause. Each one dragged through billions of parameters one at a time.

Chen pulls up the GPU utilization monitor. The compute units are barely flickering during decode — the hardware is waiting, not working. "Single-digit utilization," she says flatly. "The hardware can do the math. It's spending all its time reading."

There has to be a way to get more tokens per unit time out of this model without changing the model itself.`,
    successNarrative: `The signal decoder accelerates dramatically. Draft tokens stream from the small model, and the 70B verifier accepts most of them in bulk — each verification pass producing multiple output tokens for the cost of one.

Speculative decoding exploits the asymmetry between draft and verify. The small draft model proposes several candidate tokens cheaply. The large model verifies them all in parallel (a single forward pass, roughly the same cost as generating one token normally). Accepted tokens are free throughput. The speedup depends on the acceptance rate — similar model families tend to agree on most predictions.

For bandwidth-bound decode, speculative decoding is one of the few ways to break the single-token bottleneck without adding hardware.`,
    primaryMode: 'inference',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'a100-80gb',
      numGPUs: 4,
      tensorParallel: 4,
      weightPrecision: 'int8',
      batchSize: 1,
      speculativeDecoding: false,
      inputSeqLen: 1024,
      outputSeqLen: 512,
    },
    winningCriteria: [
      { field: 'speculative.speedup', operator: '>', value: 1.5, label: 'Decode throughput > 1.5x baseline' },
    ],
    expectedChanges: [
      { field: 'speculativeDecoding', check: 'enabled', label: 'Enabled speculative decoding' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change target model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU' },
      { field: 'numGPUs', check: 'unchanged', label: 'Did not change GPU count' },
      { field: 'tensorParallel', check: 'unchanged', label: 'Did not change TP' },
      { field: 'inputSeqLen', check: 'unchanged', label: 'Did not change input sequence length' },
      { field: 'outputSeqLen', check: 'unchanged', label: 'Did not change output sequence length' },
    ],
    hints: [
      'Autoregressive decoding generates one token at a time — each requiring a full forward pass through all parameters. For large models, decode is memory-bandwidth-bound: the GPU spends most of its time reading weights from memory, not computing. The compute units are starving for work.',
      'A single forward pass through the large model costs the same whether it produces one token or verifies several candidate tokens at once. If something could cheaply propose candidates for the large model to check in bulk, you could get multiple tokens for the cost of one forward pass.',
      'Look for the Speculative Decoding toggle in the inference settings. The system will automatically pair a compatible draft model. The speedup depends on the acceptance rate — similar model families tend to agree more often.',
    ],
    prerequisites: ['mission-2-1'],
    skillsAwarded: ['inference-optimization'],
  },

  // ── Mission 2-4: The Weight of Memory ───────────────────────────────
  {
    id: 'mission-2-4',
    arcId: 'arc2-discovery',
    order: 4,
    title: 'The Weight of Memory',
    subtitle: 'The ship has never trained before',
    learningObjectives: [
      'Calculate per-parameter memory cost at FP32 (20 bytes) vs BF16 (18 bytes)',
      'Understand why DDP replicates full model state on every GPU',
      'Use FSDP to shard weights, gradients, and optimizer states across GPUs',
    ],
    briefing: `Chief Engineer Okafor wants to fine-tune the navigation model on real debris field data — the pre-trained model keeps misclassifying asteroid fragments. "The manual says FP32 for numerical safety," Okafor says. "So we run FP32." She hits launch and watches the system crash instantly. Out of memory.

Okafor adjusts the precision and tries again. Another crash — different stage, same result. The memory diagnostic shows each GPU independently holding far more data than it should need to. "Four GPUs," she mutters, "and each one's trying to carry everything alone."

Make the 8B model train on four 80 GB GPUs.`,
    successNarrative: `The navigation model begins training on real debris field data. Within minutes, the loss curve drops — the model is learning to distinguish asteroid fragments from harmless dust clouds. Okafor watches the loss curve for a long moment. "So the manual was wrong about FP32. Good to know."

Two bottlenecks had to fall. FP32 stores each parameter in 4 bytes, with 4-byte gradients and 12 bytes of optimizer state — 20 bytes per parameter, 160 GB for an 8B model. BF16 cuts parameter and gradient storage while keeping FP32 optimizer masters internally, dropping total to ~18 bytes per param. But even at BF16, DDP replicates everything on every GPU — 144 GB per device, still exceeding 80 GB.

FSDP (Fully Sharded Data Parallelism) shards the model state across all GPUs. Each GPU stores only 1/N of the weights, gradients, and optimizer states, gathering what it needs before each layer. Combined with BF16, the 8B model fits comfortably across four GPUs.`,
    primaryMode: 'training',
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 'a100-80gb',
      numGPUs: 4,
      strategyType: 'ddp',
      mixedPrecision: 'fp32',
      globalBatchSize: 32,
      microBatchSize: 2,
      sequenceLength: 4096,
      activationCheckpointing: true,
      flashAttention: true,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Training fits in memory' },
      { field: 'mfu', operator: '>', value: 0.30, label: 'MFU > 30%' },
    ],
    expectedChanges: [
      { field: 'precision', check: 'changed', label: 'Changed training precision' },
      { field: 'strategyType', check: 'changed', label: 'Changed distributed strategy' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU' },
      { field: 'numGPUs', check: 'unchanged', label: 'Did not change GPU count' },
      { field: 'sequenceLength', check: 'unchanged', label: 'Did not change sequence length' },
    ],
    hints: [
      'Training stores far more than just the model weights. Look at the memory breakdown on the dashboard — gradients and optimizer states dwarf the weights themselves. What happens to total memory when you change precision?',
      'Reduced precision (like BF16) uses fewer bytes per parameter for weights and gradients. The optimizer still keeps FP32 master copies internally, but total memory per parameter drops significantly. Check what precision options are available.',
      'DDP replicates the full model state on every GPU — each device holds all weights, gradients, and optimizer states independently. Look for a distribution strategy that doesn\'t replicate the full state on every device — one that shards it across GPUs instead.',
    ],
    prerequisites: ['mission-1-6'],
    skillsAwarded: ['precision', 'data-parallelism'],
  },

  // ── Mission 2-5: Activation Avalanche ───────────────────────────────
  {
    id: 'mission-2-5',
    arcId: 'arc2-discovery',
    order: 5,
    title: 'Activation Avalanche',
    subtitle: 'The backward pass remembers everything',
    learningObjectives: [
      'Identify activation memory as distinct from weight/optimizer memory',
      'Understand that activation memory scales with layers × micro-batch size × sequence length',
      'Enable activation checkpointing to trade compute for activation memory',
    ],
    briefing: `Okafor's fine-tune launches successfully with FSDP — the weights and optimizer states fit. The forward pass completes. Then, midway through the backward pass, the system crashes. Out of memory again, but this time the memory diagnostic tells a different story: the model state is well within budget. Something else is filling the GPUs.

The forward pass ran fine. The backward pass exploded. Whatever is accumulating between forward and backward is proportional to the model's depth and the amount of data in each micro-batch — and at these settings, it exceeds what the GPUs can hold.

Make the backward pass complete without changing the batch size, sequence length, or strategy.`,
    successNarrative: `The backward pass completes without interruption. Gradients flow through every layer, and the optimizer takes its first step. Okafor watches the loss begin to decrease. "So the forward pass was fine," she says slowly. "It's the backward pass that remembers everything. Noted."

Activation checkpointing breaks the memory-compute tradeoff in favor of memory. During forward, instead of storing every layer's intermediate tensors (attention outputs, MLP intermediates, normalization results), the system discards them. During backward, it recomputes each layer's activations on-the-fly just before they're needed for gradient computation.

The price is time — each layer's activations must be recomputed from scratch during the backward pass, effectively running the forward pass twice. But activation memory drops to a single layer's worth at any given moment, instead of all layers stacked in memory simultaneously. When the alternative is a crash, that trade is easy to accept.`,
    primaryMode: 'training',
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 'a100-80gb',
      numGPUs: 4,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
      globalBatchSize: 64,
      microBatchSize: 4,
      sequenceLength: 12288,
      activationCheckpointing: false,
      flashAttention: true,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Training fits in memory' },
    ],
    expectedChanges: [
      { field: 'activationCheckpointing', check: 'enabled', label: 'Enabled activation checkpointing' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU' },
      { field: 'numGPUs', check: 'unchanged', label: 'Did not change GPU count' },
      { field: 'strategyType', check: 'unchanged', label: 'Did not change strategy' },
      { field: 'precision', check: 'unchanged', label: 'Did not change precision' },
      { field: 'sequenceLength', check: 'unchanged', label: 'Did not change sequence length' },
    ],
    hints: [
      'During the forward pass, each transformer layer produces intermediate tensors that the backward pass needs for gradient computation. These "activations" accumulate across all layers — the deeper the model and the larger the micro-batch, the more memory they consume. This is separate from weight and optimizer memory.',
      'There is a classic compute-memory tradeoff: instead of storing all layer activations, discard them during forward and recompute them during backward. Full checkpointing reduces activation memory to O(1) per layer — only one layer\'s worth is held at a time — at the cost of extra compute.',
      'Look for the Activation Checkpointing setting in the training configuration. "Full" checkpointing discards and recomputes all layer activations. "Selective" discards only the activations that are cheapest to recompute while keeping the more expensive ones.',
    ],
    prerequisites: ['mission-2-4'],
    skillsAwarded: ['memory-optimization'],
  },

  // ── Mission 2-6: The Adapter ────────────────────────────────────────
  {
    id: 'mission-2-6',
    arcId: 'arc2-discovery',
    order: 6,
    title: 'The Adapter',
    subtitle: "Can't we just teach it the new part?",
    learningObjectives: [
      'Understand why full fine-tuning of 70B models requires massive memory for optimizer states',
      'Explain how LoRA freezes base weights and trains small adapter matrices',
      'Apply LoRA or QLoRA to fit a 70B fine-tune on limited hardware',
    ],
    briefing: `The science team wants to fine-tune the 70B model for signal pattern recognition. Full fine-tuning at BF16 with FSDP across 8 A100s — surely that's enough hardware? Not even close. Okafor runs the numbers. "Optimizer state alone is over a terabyte," she reports. "We're not even close."

But do all 70 billion parameters really need to change? The model already understands language, reasoning, structure. The signal patterns are a narrow specialization layered on top of vast general knowledge. Updating every parameter to learn one new task is a poor trade.

Find a way to specialize the 70B model for signal patterns without the memory cost of updating every parameter.`,
    successNarrative: `The 70B model begins adapting to signal patterns. The adapters are tiny — less than 1% of total parameters — but they're enough to specialize the model's attention mechanism for the new domain.

LoRA (Low-Rank Adaptation) freezes the entire base model and injects small trainable matrices into each attention layer. Only these adapters need gradients and optimizer states. The frozen base weights become read-only, eliminating their optimizer overhead entirely.

QLoRA goes further: it compresses the frozen base weights to 4-bit NF4 format, cutting their storage by 4×. Combined with small BF16 adapters, a 70B model fine-tune fits on hardware that couldn't dream of full fine-tuning.`,
    primaryMode: 'training',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'a100-80gb',
      numGPUs: 8,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
      globalBatchSize: 16,
      microBatchSize: 1,
      sequenceLength: 4096,
      activationCheckpointing: true,
      finetuningMethod: 'full',
      flashAttention: true,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Training fits in memory' },
    ],
    expectedChanges: [
      { field: 'finetuningMethod', check: 'changed', label: 'Changed fine-tuning method' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU' },
      { field: 'numGPUs', check: 'unchanged', label: 'Did not change GPU count' },
      { field: 'sequenceLength', check: 'unchanged', label: 'Did not change sequence length' },
    ],
    hints: [
      'Full fine-tuning updates every parameter. The optimizer must store momentum and variance for each one — this state is the dominant memory consumer during training. The more parameters you train, the more optimizer state you carry. Consider whether every parameter really needs updating.',
      'If most of the model\'s knowledge is already correct, you can freeze the base weights and train only a small number of new parameters. Frozen parameters need no gradients and no optimizer state — their memory cost drops dramatically.',
      'Look for the Fine-tuning Method selector in training settings. There are multiple parameter-efficient approaches — some inject small trainable matrices while keeping the base model at full precision, others also compress the frozen base weights to a lower-bit format. Either can make a 70B fine-tune fit on hardware that couldn\'t handle full training.',
    ],
    prerequisites: ['mission-2-2', 'mission-2-4'],
    skillsAwarded: ['memory-optimization'],
  },

  // ── Mission 2-7: The Shipment ───────────────────────────────────────
  {
    id: 'mission-2-7',
    arcId: 'arc2-discovery',
    order: 7,
    title: 'The Shipment',
    subtitle: 'Sixteen new GPUs, straight from Earth',
    learningObjectives: [
      'Calculate weight memory as parameters × bytes-per-parameter and compare against total GPU capacity',
      'Understand that reduced-precision formats (FP8, INT8) halve per-parameter storage compared to BF16',
      'Deploy a model that exceeds total GPU memory at BF16 by switching to a more compact precision',
    ],
    briefing: `A resupply drone from Earth catches the Meridian — launched years after departure on a faster trajectory, it has finally closed the gap. Inside: two sealed compute modules, each containing 8 H100 GPUs with 80 GB each and high-bandwidth interconnects. And on the drone's solid-state storage banks: model weights too large to transmit over the interstellar relay, including a 405-billion-parameter model — far beyond anything the ship has run before.

"Deploy the 405B immediately," Lindqvist orders. "Chen's team has been waiting for a model large enough to decode the deeper structure — give it to them." The configuration is loaded, weights distributed across all eight GPUs, and the system immediately reports out of memory. The model's weight footprint at current precision simply exceeds the total capacity of the cluster.

No parallelism trick can help when the raw data doesn't fit. But these new H100 GPUs have capabilities the old hardware didn't. Make the 405B model load and serve requests.`,
    successNarrative: `The 405B model loads for the first time. Chen's analysis immediately jumps in resolution — structures in the signal that were noise at 70B now resolve into distinct, repeating patterns. The deeper model sees things the smaller one couldn't.

Dropping from BF16 (2 bytes per parameter) to an 8-bit format (1 byte) halves weight memory from 810 GB to ~405 GB. Eight H100s with 80 GB each provide 640 GB — more than enough.

The H100's Transformer Engine natively supports 8-bit matrix multiplications — both FP8 and INT8 — at roughly double the throughput of BF16. Reduced precision doesn't just save memory; it makes the model faster on the same hardware.`,
    primaryMode: 'inference',
    setup: {
      modelId: 'llama3-405b',
      gpuId: 'h100-sxm',
      numGPUs: 8,
      gpusPerNode: 8,
      tensorParallel: 8,
      weightPrecision: 'bf16',
      batchSize: 1,
      inputSeqLen: 1024,
      outputSeqLen: 512,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Model fits in memory' },
      { field: 'throughput.tokensPerSecond', operator: '>', value: 30, label: 'Throughput > 30 tok/s' },
    ],
    expectedChanges: [
      { field: 'weightPrecision', check: 'changed', label: 'Changed weight precision' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU' },
      { field: 'numGPUs', check: 'unchanged', label: 'Did not change GPU count' },
      { field: 'tensorParallel', check: 'unchanged', label: 'Did not change TP' },
      { field: 'inputSeqLen', check: 'unchanged', label: 'Did not change input sequence length' },
      { field: 'outputSeqLen', check: 'unchanged', label: 'Did not change output sequence length' },
    ],
    hints: [
      'Calculate the total weight memory: parameters × bytes per parameter at the current precision. Then calculate total GPU memory: number of GPUs × memory per GPU. If the first number exceeds the second, no parallelism strategy can help — the data itself doesn\'t fit.',
      'Different precision formats store each parameter in different numbers of bytes. BF16 uses 2 bytes. Some newer GPU architectures support formats that use fewer bytes per parameter while maintaining inference quality — and also increase effective compute throughput.',
      'Look for the Weight Precision selector in inference settings. Consider what precision options the H100 hardware supports that the older GPUs didn\'t. Halving the bytes per parameter halves the total weight memory.',
    ],
    prerequisites: ['mission-1-7', 'mission-2-2', 'mission-2-4'],
    skillsAwarded: ['precision'],
  },

  // ── Mission 2-8: Bandwidth Wall ─────────────────────────────────────
  {
    id: 'mission-2-8',
    arcId: 'arc2-discovery',
    order: 8,
    title: 'Bandwidth Wall',
    subtitle: 'Not all connections are equal',
    learningObjectives: [
      'Distinguish intra-node vs inter-node interconnect bandwidth',
      'Understand why cross-node TP creates a communication bottleneck at every layer',
      'Keep TP within a single node and use replicas across nodes for throughput',
    ],
    briefing: `"We have sixteen H100s," Lindqvist says. "I want the 405B running across all of them." The obvious approach: TP=16, splitting each layer across every GPU. Double the hardware, double the speed.

But the results are abysmal. Sixteen GPUs are barely outperforming eight. Lindqvist reviews the throughput reports with visible frustration. "I doubled the hardware and got nothing. Explain." Spreading the work across both nodes has introduced a bottleneck that didn't exist when everything ran on a single node.

Get the 405B model serving at high throughput across all 16 GPUs.`,
    successNarrative: `The 405B model serves requests from both nodes simultaneously. Two replicas, each running at full intra-node speed, deliver combined throughput that dwarfs the cross-node TP approach.

Not all connections are equal. Within a single node, GPU interconnects deliver hundreds of GB/s per link — TP AllReduce costs almost nothing. Between nodes, the inter-node fabric has significantly less bandwidth. TP=16 across 2 nodes forced every layer's synchronization through that slower link.

By keeping TP within a node (TP=8) and using the second node as a separate replica, each model copy runs at full intra-node speed. The inter-node fabric handles only the load balancer traffic — negligible compared to per-layer AllReduce. Two fast replicas beat one slow one.`,
    primaryMode: 'inference',
    setup: {
      modelId: 'llama3-405b',
      gpuId: 'h100-sxm',
      numGPUs: 16,
      gpusPerNode: 8,
      tensorParallel: 16,
      weightPrecision: 'bf16',
      batchSize: 1,
      inputSeqLen: 1024,
      outputSeqLen: 512,
    },
    winningCriteria: [
      { field: 'throughput.tokensPerSecond', operator: '>', value: 80, label: 'Throughput > 80 tok/s' },
    ],
    expectedChanges: [
      { field: 'tensorParallel', check: 'decreased', label: 'Reduced tensor parallelism' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU' },
      { field: 'inputSeqLen', check: 'unchanged', label: 'Did not change input sequence length' },
      { field: 'outputSeqLen', check: 'unchanged', label: 'Did not change output sequence length' },
    ],
    hints: [
      'Not all GPU connections are equal. Within a single node, GPUs communicate through a high-bandwidth interconnect — fundamentally faster than the inter-node network fabric. TP requires synchronization after every transformer layer. Consider what happens when every layer\'s synchronization must cross the slower link.',
      'Keeping TP within a single node confines the per-layer synchronization to the fast local interconnect. Crossing the node boundary for every layer is what makes cross-node TP devastatingly slow — the per-layer overhead compounds across dozens of layers.',
      'If each model copy uses only one node\'s worth of GPUs, the second node can serve as a separate replica handling requests in parallel. You may need to adjust weight precision to fit the model at the new TP degree — the new GPUs support precision formats the old hardware didn\'t.',
    ],
    prerequisites: ['mission-2-7'],
    skillsAwarded: ['hardware'],
  },

  // ── Mission 2-9: The Pipeline ───────────────────────────────────────
  {
    id: 'mission-2-9',
    arcId: 'arc2-discovery',
    order: 9,
    title: 'The Pipeline',
    subtitle: 'Layers flow through a pipe',
    learningObjectives: [
      'Understand pipeline parallelism as vertical model splitting across nodes',
      'Compare P2P pipeline communication vs collective AllGather communication costs',
      'Configure PP to move cross-node communication from expensive collectives to cheap P2P transfers',
    ],
    briefing: `Okafor needs both compute nodes training together on the 70B model — the dataset is enormous and time is critical. "Every step, the AllGather has to cross the inter-node link," she says, pointing at the MFU trace. "We're spending more time waiting on the network than doing math."

There has to be a better way to divide work between the two nodes — one that doesn't force heavy collective traffic across the slower inter-node fabric every step.`,
    successNarrative: `Both compute nodes are training the 70B model with pipeline-parallel efficiency. Each node handles half the layers, with activations flowing between them as a simple point-to-point stream.

Not all cross-node communication is equal. FSDP AllGather is a collective — every GPU participates, and the operation is bounded by the slowest link. Pipeline parallelism uses point-to-point transfers: one GPU sends activations to the next stage. P2P transfers are much cheaper over inter-node fabric.

The tradeoff is the pipeline bubble — idle time during ramp-up and ramp-down of the micro-batch pipeline. But for large models across multiple nodes, the bubble cost is far less than the collective communication overhead it replaces.`,
    primaryMode: 'training',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'h100-sxm',
      numGPUs: 16,
      gpusPerNode: 8,
      strategyType: 'fsdp-tp-pp',
      mixedPrecision: 'bf16',
      globalBatchSize: 64,
      microBatchSize: 1,
      sequenceLength: 4096,
      activationCheckpointing: true,
      flashAttention: true,
      tpDegree: 8,
      ppDegree: 1,
      sequenceParallel: true,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Training fits in memory' },
      { field: 'mfu', operator: '>', value: 0.39, label: 'MFU > 39%' },
    ],
    expectedChanges: [
      { field: 'ppDegree', check: 'increased', label: 'Increased pipeline parallelism' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU' },
      { field: 'numGPUs', check: 'unchanged', label: 'Did not change GPU count' },
      { field: 'sequenceLength', check: 'unchanged', label: 'Did not change sequence length' },
    ],
    hints: [
      'There are two fundamentally different ways to split a model across devices. One splits each layer horizontally — every GPU computes a slice of every layer (requiring collective synchronization). The other splits vertically — different devices handle different layers, with data flowing through them like an assembly line.',
      'Collective operations (like AllGather) require every participating GPU to communicate. Point-to-point transfers only require neighbors to exchange data. Over slower inter-node fabric, the type of communication matters as much as the volume. Consider which parallelism dimensions are currently crossing the node boundary.',
      'The Pipeline Parallel setting controls vertical model splitting. Increasing it introduces a "pipeline bubble" (idle time during ramp-up/ramp-down), but changes the nature of cross-node communication. Consider the tradeoff: some idle time versus expensive collective operations every step.',
    ],
    prerequisites: ['mission-2-8'],
    skillsAwarded: ['model-parallelism'],
  },

  // ── Mission 2-10: The Protein Problem ───────────────────────────────
  {
    id: 'mission-2-10',
    arcId: 'arc2-discovery',
    order: 10,
    title: 'The Protein Problem',
    subtitle: 'Two critical jobs. One ship',
    learningObjectives: [
      'Optimize training MFU through precision, checkpointing, and batch sizing',
      'Apply inference optimization skills to a latency-constrained workload',
      'Manage two independent workloads across different hardware pools',
    ],
    briefing: `Signal analysis has revealed molecular structures embedded in the data — possibly protein folding patterns. Dr. Patel needs to fine-tune a model for enzymatic analysis, but Chen's signal decoding can't stop.

"The folding geometries in this signal — they're not random. Something is manufacturing proteins down there," Patel says, spreading the analysis across the briefing table. "I need the H100 cluster."

Chen doesn't look up from the signal feed. "I need continuous coverage. Every gap in the decoder is data we lose permanently."

Lindqvist makes the call. "Chen keeps the A100 array for inference. Patel gets the H100s for training. No sharing, no queuing." Patel opens his mouth — Lindqvist cuts him off. "Make it work."

Two workloads, two hardware pools, no room for compromise. Make them both work.`,
    successNarrative: `Both workloads run simultaneously. Patel's protein model trains at high efficiency on the H100s while Chen's signal decoder serves responses at acceptable latency on the A100s. The ship's compute resources are fully utilized, every GPU earning its power draw.

Production ML runs multiple workloads on limited hardware. Training demands high throughput and GPU utilization. Inference demands low latency and reliable memory headroom. Optimizing one doesn't optimize the other — each requires its own set of choices about precision, parallelism, and batching.

The skills from every previous mission converge here. FP8 for training throughput. TP for inference latency. Activation checkpointing for memory headroom. Quantization for model fitting. No single trick — a toolbox.`,
    primaryMode: 'training',
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 'h100-sxm',
      numGPUs: 8,
      gpusPerNode: 8,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
      globalBatchSize: 64,
      microBatchSize: 8,
      sequenceLength: 4096,
      activationCheckpointing: false,
      flashAttention: true,
    },
    winningCriteria: [],
    expectedChanges: [],
    objectives: [
      {
        id: 'obj-train',
        label: 'Training: Protein model fine-tune',
        primaryMode: 'training',
        setup: {
          modelId: 'llama3.1-8b',
          gpuId: 'h100-sxm',
          numGPUs: 8,
          gpusPerNode: 8,
          strategyType: 'fsdp',
          mixedPrecision: 'bf16',
          globalBatchSize: 64,
          microBatchSize: 8,
          sequenceLength: 4096,
          activationCheckpointing: false,
          flashAttention: true,
        },
        winningCriteria: [
          { field: 'success', operator: '==', value: true, label: 'Training fits in memory' },
          { field: 'mfu', operator: '>', value: 0.50, label: 'MFU > 50%' },
        ],
        expectedChanges: [
          { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
          { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU' },
          { field: 'numGPUs', check: 'unchanged', label: 'Did not change GPU count' },
          { field: 'sequenceLength', check: 'unchanged', label: 'Did not change sequence length' },
        ],
      },
      {
        id: 'obj-infer',
        label: 'Inference: Signal analysis',
        primaryMode: 'inference',
        setup: {
          modelId: 'llama3.3-70b',
          gpuId: 'a100-80gb',
          numGPUs: 4,
          tensorParallel: 2,
          weightPrecision: 'bf16',
          batchSize: 1,
          inputSeqLen: 1024,
          outputSeqLen: 512,
        },
        winningCriteria: [
          { field: 'success', operator: '==', value: true, label: 'Model loads successfully' },
          { field: 'latency.ttft', operator: '<', value: 200, label: 'TTFT < 200ms' },
        ],
        expectedChanges: [
          { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
          { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU' },
          { field: 'numGPUs', check: 'unchanged', label: 'Did not change GPU count' },
          { field: 'inputSeqLen', check: 'unchanged', label: 'Did not change input sequence length' },
          { field: 'outputSeqLen', check: 'unchanged', label: 'Did not change output sequence length' },
        ],
      },
    ],
    hints: [
      'The default config OOMs. Check the memory breakdown — what\'s consuming the most? Activation memory scales with micro-batch size and sequence length, while model weights and optimizer states are fixed costs. Understanding which category dominates tells you which knobs matter.',
      'There are several ways to trade between memory and compute efficiency: precision affects both arithmetic throughput and memory footprint, checkpointing reduces activation memory at the cost of recomputation, and batch sizing changes how fully the GPU\'s compute units are saturated. Each combination has different tradeoffs — experiment to find one that clears 50% MFU.',
      'This mission has two objectives on different hardware pools. Training rewards throughput — maximizing useful compute per GPU-second. Inference rewards latency — minimizing time-to-first-token. The optimization strategies are different: for inference, consider how tensor parallelism and weight precision affect TTFT.',
    ],
    prerequisites: ['mission-2-5', 'mission-2-7'],
    skillsAwarded: ['resource-efficiency'],
  },

  // ── Mission 2-11: PIVOT — Life ──────────────────────────────────────
  {
    id: 'mission-2-11',
    arcId: 'arc2-discovery',
    order: 11,
    title: 'Life',
    subtitle: 'Not a message. Not a machine',
    type: 'pivot',
    learningObjectives: [],
    briefing: `Dr. Patel's protein model produces its first results. The enzymatic catalysis patterns in the signal aren't random — they're self-replicating. Consuming energy, adapting, evolving.

Not a message. Not a machine. Life.

Captain Lindqvist is quiet for a long time after Patel finishes the briefing. Then: "All-hands assembly. Twenty minutes."`,
    successNarrative: `The bridge is silent as Patel presents the findings. "On Earth, we'd call these chaperone proteins," he says, his voice unsteady. "But the folding geometry is wrong — it's solving for a different gravity. A different ocean. This isn't contamination. This is an independent origin."

Self-replicating molecular structures. Metabolic cycles. Evolutionary pressure operating on proteins no Earth biochemist has ever seen.

Lindqvist stares at the viewscreen for a long moment — the weight of two hundred sleeping lives behind every decision he's ever made, and now this. "Two hundred colonists left Earth looking for a home," he says finally. "Amend the mission brief."

The universe is not empty. And the Meridian is heading straight for it.`,
    primaryMode: 'inference',
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 't4',
    },
    winningCriteria: [],
    hints: [],
    prerequisites: ['mission-2-7', 'mission-2-10'],
    skillsAwarded: [],
  },
];
