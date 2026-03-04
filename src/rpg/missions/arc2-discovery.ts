/**
 * Arc 2: Discovery — the signal is real. Science teams mobilize.
 * Missions teach: TP for latency, replica scaling, speculative decoding,
 * mixed precision, FSDP, activation checkpointing, LoRA, FP8, network
 * topology, pipeline parallelism, and multi-objective optimization.
 * See RPG Text Style Guide in ../types.ts for briefing/hint/success narrative rules.
 *
 * DAG:
 *   INFERENCE BRANCH              TRAINING BRANCH          HARDWARE (optional side chain)
 *
 *   1-8 (Signal pivot)            1-6 (Archive Vault)      1-7 (Fuel Budget)
 *    │                              │                        │
 *    ▼                              ▼                        │
 *   2-1 (TP latency)             2-4 (BF16 + FSDP)         │
 *    ├──────────┐                   │                        │
 *    ▼          ▼                   ▼                        │
 *   2-2 (Derelict)  2-3 (Spec)  2-5 (Activation ckpt)      │
 *    │                              │                        │
 *    │                              ▼                        │
 *    ├──────────────────────────► 2-6 (LoRA)                │
 *    │                                                       │
 *    ├──────┐                       │         ┌──── 2-2 ────┤
 *    │      │                       │         │      2-4 ───┘
 *    │      │                       │         ▼
 *    │      │                     2-7 (Shipment) ← 1-7, 2-2, 2-4
 *    │      │                       │
 *    │      │                       ▼
 *    │      │                     2-8 (Bandwidth Wall)
 *    │      │                       │
 *    │      │                       ▼
 *    │      │                     2-9 (Pipeline) ──────────► (3-4)
 *    │      │
 *    ▼      ▼
 *    2-10 (Protein/MFU) ← 2-2, 2-5, 2-7
 *
 *    ⭐ 2-11 (Life) ← 2-10
 */

import type { RPGArc, RPGMission } from '../types.ts';

export const ARC2: RPGArc = {
  id: 'arc2-discovery',
  name: 'Discovery',
  subtitle: 'A signal from Ross 128b',
  description: 'Analyze the alien signal. Train new models. Push the ship\'s compute to its limits.',
  order: 2,
  briefing: `The long-range scanner has detected a structured, repeating signal from Ross 128b. Dr. Chen's team confirms it — encoded information, not cosmic noise. For the first time in 80 years, the ship has a reason to do more than survive.

Science teams are coming out of cryo rotation. Chen needs real-time spectral analysis. Okafor wants to fine-tune navigation models on new sensor data. Every department is suddenly demanding GPU time — and the compute bay you nursed through survival is now the bottleneck for discovery.

For the first time, the ship's models will need to train — not just on data from Earth, but on something no human has ever seen. The problems ahead are harder, the hardware demands are steeper, and every watt matters. Somewhere ahead, something is broadcasting — and the Meridian needs to be ready.`,
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
      { field: 'latency.ttft', operator: '<', value: 300, label: 'TTFT < 300ms' },
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
    briefing: `An EVA team discovers a derelict relay station drifting near the ship. Inside: four more A100 GPUs, still functional after decades in vacuum. Engineering hauls them back and wires them into the compute bay — eight A100s now online. Chen needs three concurrent analysis streams — different spectral bands being decoded simultaneously.

Right now, all eight GPUs are yoked together serving a single stream. Fast — but only one at a time. When the second spectral band arrives, it queues. The third waits behind it. Chen is watching two-thirds of the incoming data queue up and go unanalyzed while the model processes one band at a time.

Three spectral bands, three simultaneous requests. Eight GPUs, more than enough silicon. Find a configuration that serves all three concurrently without unacceptable latency on any one of them.`,
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
      { field: 'latency.ttft', operator: '<', value: 600, label: 'TTFT < 600ms' },
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
    skillsAwarded: ['model-parallelism'],
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

The model stays. The throughput has to change.`,
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
      'Calculate total per-parameter training memory: FP32 model state (16 bytes) vs BF16 mixed precision (18 bytes)',
      'Understand why DDP replicates full model state on every GPU',
      'Use FSDP to shard weights, gradients, and optimizer states across GPUs',
    ],
    briefing: `Chief Engineer Okafor wants to fine-tune the navigation model on real debris field data — the pre-trained model keeps misclassifying asteroid fragments. "The manual says full 32-bit precision for numerical safety," Okafor says. "So we run full precision." She hits launch and watches the system crash instantly. Out of memory.

Okafor adjusts the precision and tries again. Another crash — different stage, same result. The memory diagnostic shows each GPU independently holding far more data than it should need to. "Four GPUs," she mutters, "and each one's trying to carry everything alone."

Make the 8B model train on four 80 GB GPUs.`,
    successNarrative: `The navigation model begins training on real debris field data. Within minutes, the loss curve drops — the model is learning to distinguish asteroid fragments from harmless dust clouds. Okafor watches the loss curve for a long moment. "So the manual was wrong about FP32. Good to know."

Precision was the first thing to fix. FP32 costs 16 bytes per parameter — 4 for the weight, 4 for the gradient, 8 for the optimizer. That's 128 GB for an 8B model, already beyond a single 80 GB GPU. Switching to BF16 seems obvious, but here's the twist: mixed precision actually makes model state *bigger*. The optimizer still needs FP32 master weights, so the total climbs to 18 bytes per parameter — 144 GB. BF16 earns its keep elsewhere: the hardware runs half-precision math faster, and activations — the real memory hog — shrink by half.

Even so, 144 GB per GPU is still too much. That's because DDP copies everything onto every device — four GPUs, four identical copies, each one over capacity. FSDP breaks the deadlock by sharding weights, gradients, and optimizer states across all GPUs, each device holding only its 1/N slice. Combined with BF16, the 8B model fits comfortably across four.`,
    primaryMode: 'training',
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 'a100-80gb',
      numGPUs: 4,
      strategyType: 'ddp',
      mixedPrecision: 'fp32',
      globalBatchSize: 32,
      microBatchSize: 2,
      sequenceLength: 2048,
      activationCheckpointing: false,
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
    ],
    hints: [
      'Training stores far more than just the model weights. Look at the memory breakdown on the dashboard — gradients and optimizer states dwarf the weights themselves. What happens to total memory when you change precision?',
      'Reduced precision (like BF16) uses fewer bytes per parameter for weights and gradients. The optimizer still keeps FP32 master copies internally, but total memory per parameter drops significantly. Check what precision options are available.',
      'DDP replicates the full model state on every GPU — each device holds all weights, gradients, and optimizer states independently. Look for a distribution strategy that doesn\'t replicate the full state on every device — one that shards it across GPUs instead.',
    ],
    prerequisites: ['mission-1-6'],
    skillsAwarded: ['precision', 'resource-efficiency'],
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
    briefing: `The navigation fine-tune is running — FSDP sharding solved the memory problem. Now Chen's team needs longer context windows to capture the deeper patterns in the Ross 128b signal. Okafor scales up the sequence length and batch size, same distributed setup that worked before.

The weights and optimizer states fit. Midway through training — loss of signal. Out of memory, but the diagnostic says model state is well within budget. Something else is filling the GPUs.

Make it fit without changing the batch size, sequence length, or strategy.`,
    successNarrative: `The backward pass completes without interruption. Gradients flow through every layer, and the optimizer takes its first step. Okafor watches the loss begin to decrease. "So the forward pass was fine," she says slowly. "The backward pass remembers everything."

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
      'During the forward pass, each transformer layer produces intermediate tensors that the backward pass needs for gradient computation. These "activations" accumulate across all layers — the deeper the model, the longer the sequence, and the larger the micro-batch, the more memory they consume. This is separate from weight and optimizer memory.',
      'There is a classic compute-memory tradeoff: instead of storing all layer activations, discard them during forward and recompute them during backward. Full checkpointing means only one layer\'s activations are held at a time during backward, instead of all layers stacked simultaneously — at the cost of extra compute.',
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
    briefing: `The science team wants to fine-tune the 70B model for signal pattern recognition. Full fine-tuning distributed across all eight A100s — surely that's enough hardware? Not even close. Okafor runs the numbers. "Optimizer state alone is 840 gigabytes," she reports. "We're not even close."

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
    ],
    hints: [
      'Full fine-tuning updates every parameter. The optimizer must store momentum and variance for each one — this state is the dominant memory consumer during training. The more parameters you train, the more optimizer state you carry. Consider whether every parameter really needs updating.',
      'If most of the model\'s knowledge is already correct, you can freeze the base weights and train only a small number of new parameters. Frozen parameters need no gradients and no optimizer state — their memory cost drops dramatically.',
      'Look for the Fine-tuning Method selector in training settings. There are multiple parameter-efficient approaches — some inject small trainable matrices while keeping the base model at full precision, others also compress the frozen base weights to a lower-bit format. Either can make a 70B fine-tune fit on hardware that couldn\'t handle full training.',
    ],
    prerequisites: ['mission-2-2', 'mission-2-4'],
    skillsAwarded: ['memory-optimization'],
  },

  // ── Mission 2-7: The Shipment ──────────────────────────────────────
  // FP8 as compute accelerator — seqLen=16384 physics-enforces the lesson:
  //   BF16 ceiling is 33.8% MFU, FP8 floor is 50.3% MFU — 45% threshold
  //   sits cleanly between. No artificial locks needed.
  {
    id: 'mission-2-7',
    arcId: 'arc2-discovery',
    order: 7,
    title: 'The Shipment',
    subtitle: 'Sixteen new GPUs, straight from Earth',
    learningObjectives: [
      'Understand that FP8 activates dedicated hardware acceleration on Hopper GPUs, increasing compute throughput beyond what BF16 achieves',
      'Recognize the difference between precision as a storage format and precision as a compute accelerator',
    ],
    briefing: `A resupply drone from Earth catches the Meridian — launched years after departure on a faster trajectory, it has finally closed the gap. Inside: two sealed compute modules, each containing eight of the most powerful GPUs ever built, and on the drone's storage banks, a 405-billion-parameter model too large to transmit over the interstellar relay.

Okafor finally has training-capable hardware. She loads an 8B model for biosignature sequence training on the new cluster and launches the first run. The results come back and she stares at the utilization readout, frowning. "These are supposed to be the fastest GPUs ever built. Why is the compute utilization no better than the old hardware?"`,
    successNarrative: `Okafor's training run reveals something unexpected about the new hardware. FP8 isn't just a smaller number format — the H100's Transformer Engine has dedicated FP8 matrix multiply units that process roughly twice the operations per cycle compared to BF16 Tensor Cores. Same model, same config, same GPUs — switching the precision format activates fundamentally faster hardware.

This is the difference between precision as storage (fewer bytes per parameter, which saves memory) and precision as a compute accelerator (a different arithmetic format that the hardware executes faster). The A100s never had this capability. The H100s had it all along — it just needed to be turned on.

Okafor watches the utilization readout climb past 50%. "Now that's what I expected from this hardware."`,
    primaryMode: 'training',
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 'h100-sxm',
      numGPUs: 8,
      gpusPerNode: 8,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
      globalBatchSize: 64,
      microBatchSize: 2,
      sequenceLength: 16384,
      activationCheckpointing: true,
      checkpointingGranularity: 'selective',
      flashAttention: true,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Training fits in memory' },
      { field: 'mfu', operator: '>', value: 0.45, label: 'MFU > 45%' },
    ],
    expectedChanges: [
      { field: 'precision', check: 'changed', label: 'Changed training precision' },
    ],
    hints: [
      'The training run works at BF16, but the MFU is lower than you\'d expect from hardware this powerful. The utilization bottleneck isn\'t communication or memory — it\'s arithmetic throughput. These GPUs have dedicated hardware for a precision format the ship has never used.',
      'FP8 precision activates dedicated hardware acceleration. BF16 Tensor Cores process one set of operations per cycle; FP8 units process roughly twice as many. The effective throughput increase depends on what fraction of the workload is matrix multiplication.',
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
      'Understand why cross-node model splits create a communication bottleneck at every layer',
      'Keep model splits within a single node and use replicas across nodes',
      'Achieve both fast per-request response time and high aggregate throughput',
    ],
    briefing: `Chen has been running signal analysis on the 70B model, but the finer structures in the data keep slipping through — the model isn't large enough to resolve them. Okafor loads the 405B from the resupply drone onto one compute module, then two. The throughput drops.

"We doubled the hardware," Lindqvist says. "Why is it slower?" Okafor pulls the communication trace. Within each module, transfers between GPUs are near-instant. Between modules, every synchronization crosses a much slower link — and it happens at every layer.

Chen needs the 405B running at full speed — the scanner submits individual observations and waits for each analysis before targeting the next, so both total throughput and per-request speed matter. Get it serving at the throughput the science team needs across all sixteen GPUs.`,
    successNarrative: `The 405B model serves requests from both modules simultaneously. Each replica runs at full intra-node speed, and the batched requests amortize the per-token weight reads across hundreds of concurrent analyses. Chen's spectral backlog clears within minutes.

Not all connections are equal. Within a single module, the fast GPU interconnect makes per-layer synchronization nearly free. Between modules, the inter-module fabric has a fraction of that bandwidth. Spreading the model across both modules forced every layer's synchronization through the slower link — and with hundreds of layers, the overhead dominated.

By keeping each model copy within one module and using the second module as a separate replica, the synchronization stays on fast local links. Then batching transforms each replica from a single-request server to a high-throughput engine — the same weight read serves hundreds of concurrent tokens, pushing utilization from idle to saturated.`,
    primaryMode: 'inference',
    setup: {
      modelId: 'llama3-405b',
      gpuId: 'h100-sxm',
      numGPUs: 16,
      gpusPerNode: 8,
      tensorParallel: 16,
      weightPrecision: 'fp8',
      batchSize: 1,
      inputSeqLen: 1024,
      outputSeqLen: 512,
    },
    winningCriteria: [
      { field: 'throughput.tokensPerSecond', operator: '>', value: 2000, label: 'Throughput > 2,000 tok/s' },
      { field: 'latency.tpot', operator: '<', value: 25, label: 'Per-token latency < 25ms' },
    ],
    expectedChanges: [
      { field: 'tensorParallel', check: 'decreased', label: 'Reduced tensor parallelism' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU' },
    ],
    hints: [
      'Not all GPU connections are equal. Within a single compute module, GPUs communicate through a high-bandwidth local interconnect. Between modules, the inter-module fabric is much slower. When every layer\'s synchronization must cross that slower link, the overhead compounds across all layers.',
      'Keeping the model split within a single module confines the per-layer synchronization to the fast local interconnect. The second module can then serve as an independent copy handling separate requests in parallel — two fast replicas instead of one slow split. But at batch size 1, each replica processes just one request at a time. The hardware can handle far more concurrency.',
      'Once you fix the topology (model split within each module, not across both), increase the batch size. More concurrent requests per replica means the weight reads are amortized across all of them — each GPU reads the weights once and applies them to the entire batch simultaneously. The scanner needs each individual request answered quickly too — not just high total throughput.',
    ],
    prerequisites: ['mission-2-7', 'mission-1-4'],
    skillsAwarded: ['hardware', 'batching'],
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
    briefing: `Okafor needs both compute nodes training together on the 70B model — the dataset is enormous and time is critical. "Eight-way split per node is solid — fast interconnect, no bottleneck," she says. "But every step, the weight synchronization has to cross between nodes. More time waiting on the network than doing math."

The way the nodes share work can't be right — not if every step forces this much traffic across the slower inter-node fabric.

The training recipe has been validated — batch size, precision, and sequence length are calibrated for convergence on this dataset. The only question is how the two nodes should coordinate.`,
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
      { field: 'tpDegree', check: 'unchanged', label: 'Tensor split is fixed (checkpoint format)' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU' },
      { field: 'numGPUs', check: 'unchanged', label: 'Did not change GPU count' },
      { field: 'globalBatchSize', check: 'unchanged', label: 'Batch size is calibrated for convergence' },
      { field: 'microBatchSize', check: 'unchanged', label: 'Micro-batch size is calibrated for convergence' },
      { field: 'precision', check: 'unchanged', label: 'Training precision is fixed for numerical stability' },
      { field: 'sequenceLength', check: 'unchanged', label: 'Sequence length is fixed by sensor protocol' },
    ],
    hints: [
      'There are two fundamentally different ways to split a model across devices. One splits each layer horizontally — every GPU computes a slice of every layer (requiring collective synchronization). The other splits vertically — different devices handle different layers, with data flowing through them like an assembly line.',
      'Collective operations (like AllGather) require every participating GPU to communicate. Point-to-point transfers only require neighbors to exchange data. Over slower inter-node fabric, the type of communication matters as much as the volume. The tensor-parallel split can\'t change mid-run — consider what other dimensions could change how nodes coordinate.',
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
    subtitle: 'Split the fleet',
    learningObjectives: [
      'Optimize training MFU through precision and activation checkpointing',
      'Apply inference optimization skills to a latency-constrained workload',
      'Manage two independent workloads across different hardware pools',
    ],
    briefing: `Signal analysis has revealed molecular structures embedded in the data — possibly protein folding patterns. Dr. Patel needs to fine-tune a model for enzymatic analysis, but Chen's signal decoding can't stop.

"The folding geometries in this signal — they're not random," Patel says, spreading the analysis across the briefing table. "I need the H100 cluster. Full fine-tuning — the biochemistry is too alien for adapters — with batches of eight structural families."

Chen doesn't look up from the signal feed. "I need continuous coverage. Every gap in the decoder is data we lose permanently."

Lindqvist makes the call. "Chen keeps the A100 array for inference. Patel gets the H100s for training. No sharing, no queuing." Patel opens his mouth — Lindqvist cuts him off. "Make it work."

Two workloads, two hardware pools, no room for compromise. Make them both work.`,
    successNarrative: `Both workloads run simultaneously. Patel's protein model trains at high efficiency on the H100s while Chen's signal decoder serves responses at acceptable latency on the A100s. The ship's compute resources are fully utilized, every GPU earning its power draw.

On a ship with finite power and finite silicon, two workloads means two different sets of tradeoffs. Training needs throughput — batch size, precision, parallelism all tuned to keep the compute units fed. Inference needs responsiveness — low latency, headroom for burst traffic, enough memory to serve without crashing. The same configuration won't do both.

Everything you've done on this ship has been practice for this.`,
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
          { field: 'mfu', operator: '>', value: 0.52, label: 'MFU > 52%' },
        ],
        expectedChanges: [
          { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
          { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU' },
          { field: 'numGPUs', check: 'unchanged', label: 'Did not change GPU count' },
          { field: 'microBatchSize', check: 'unchanged', label: 'Did not change micro-batch size' },
          { field: 'finetuningMethod', check: 'unchanged', label: 'Did not change fine-tuning method' },
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
          { field: 'latency.ttft', operator: '<', value: 300, label: 'TTFT < 300ms' },
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
      'The default config OOMs — activation memory at this batch size dwarfs everything else. You need to both fit in memory and hit high MFU. Two knobs matter: precision controls arithmetic throughput, and checkpointing controls how much activation memory the backward pass retains.',
      'These H100s have dedicated FP8 hardware that roughly doubles arithmetic throughput compared to BF16. But FP8 alone doesn\'t solve the activation memory problem — you still need a way to keep activation memory under control during the backward pass.',
      'This mission has two objectives on different hardware pools. Training rewards throughput — maximizing useful compute per GPU-second. Inference rewards latency — minimizing time-to-first-token. The optimization strategies are different: for inference, consider how tensor parallelism and weight precision affect TTFT.',
    ],
    prerequisites: ['mission-2-2', 'mission-2-5', 'mission-2-7'],
    skillsAwarded: ['resource-efficiency'],
  },

  // ── Mission 2-11: PIVOT — Life ──────────────────────────────────────
  {
    id: 'mission-2-11',
    arcId: 'arc2-discovery',
    order: 11,
    title: 'Life',
    subtitle: 'We are not alone',
    type: 'pivot',
    learningObjectives: [],
    briefing: `Dr. Patel's protein model produces its first results. The enzymatic catalysis patterns in the signal aren't random — they're self-replicating. Consuming energy, adapting, evolving.

Life.

Captain Lindqvist is quiet for a long time after Patel finishes the briefing. Then: "All-hands assembly. Twenty minutes."`,
    successNarrative: `No one speaks as Patel presents the findings. "On Earth, we'd call these chaperone proteins," he says, his voice unsteady. "But the folding geometry is wrong — it's solving for a different gravity. A different ocean. This isn't contamination. This is an independent origin."

Self-replicating molecular structures. Metabolic cycles. Evolutionary pressure operating on proteins no Earth biochemist has ever seen.

Lindqvist grips the command rail. Everything he's done on this ship — every hard call, every compromise — led here. "We left Earth looking for a home," he says finally. "Amend the mission brief."

Something on Ross 128b is alive. And the Meridian has nowhere else to go.`,
    primaryMode: 'inference',
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 't4',
    },
    winningCriteria: [],
    hints: [],
    prerequisites: ['mission-2-10'],
    skillsAwarded: [],
  },
];
