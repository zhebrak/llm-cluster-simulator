/**
 * Arc 3: Wonder — the ship reaches Kepler-442b and prepares for first contact.
 * Missions teach: continuous batching, context parallelism, expert parallelism,
 * pipeline scheduling, multi-workload allocation, and full-system capstone.
 *
 * DAG:
 *   2-11 (Life pivot)
 *     |
 *     v
 *   3-1 (Landfall: continuous batching)
 *     |----------+-----------.
 *     v          v           |
 *   3-2 (Deep  3-3 (Alien   |
 *   Signal:CP)  Model:EP)   |
 *     |          |     3-4 (Big Train: PP sched) <- also requires 2-9
 *     |          |       |
 *     v          v       v
 *   3-5 (Resource War: multi-obj allocation) <- requires 3-2 + 3-3 + 3-4
 *     |
 *     v
 *   3-6 (First Contact Protocol: capstone)
 *     |
 *     v
 *   3-7 (The Reply: pivot)
 */

import type { RPGArc, RPGMission } from '../types.ts';

export const ARC3: RPGArc = {
  id: 'arc3-wonder',
  name: 'Wonder',
  subtitle: 'Life. Beyond any doubt',
  description: 'Reach the planet. Decode the biology. Train the translation model. Prepare for what comes next.',
  order: 3,
  briefing: `Eighty years of void. The Meridian's hull is scored by micrometeorites, its systems patched and repatched by generations of engineers who never saw the sky. But now — Kepler-442b fills the forward viewports.

The planet is alive. Dr. Patel's enzymatic analysis confirmed it: self-replicating molecular systems, metabolic cycles, evolutionary pressure operating on chemistry no Earth biochemist has ever seen. Not a message. Not a machine. Life.

Nobody planned for this. The Meridian was built to settle an empty world, not to comprehend a living one. Every probe return reveals deeper complexity. Whatever is broadcasting down there — can it hear us?`,
  heroImage: { dark: '/life_dark.png', light: '/life_light.png' },
};

/** Post-game state: shown in mission log after all arcs are complete. */
export const COMPLETION: RPGArc = {
  id: 'completion',
  name: 'First Contact',
  subtitle: 'The reply is just the beginning',
  description: '',
  order: 4,
  briefing: `Three hours and eleven minutes after transmission, the receiver array spikes.

The signal is not an echo. Not a reflection of what was sent. It carries a different structure — new patterns built on the same molecular encoding the crew has spent weeks learning to decode. The biosignature model processes the incoming data in real-time, its continuous batching pipeline handling the stream without missing a beat.

The translation model produces its first output: the alien response contains references to the crew's original message. They answered.

Patel watches the biosignature model output scroll, hands trembling. "It's referencing our molecular encoding," he whispers. "It learned our format." Chen reads the translation aloud to the bridge, her voice precise even now. Okafor stands behind the compute bay console, arms crossed, watching every GPU utilization metric hold steady. Lindqvist stares at the viewscreen for a long moment, then speaks into the ship-wide channel:

"All hands. This is the Captain. We came looking for a home. We found neighbors."

The reply is just the beginning.`,
  heroImage: { dark: '/ship_dark.png', light: '/ship_light.png' },
};

export const ARC3_MISSIONS: RPGMission[] = [
  // ── Mission 3-1: Landfall (Continuous Batching) ─────────────────────
  {
    id: 'mission-3-1',
    arcId: 'arc3-wonder',
    order: 1,
    title: 'Landfall',
    subtitle: 'A thousand streams from the surface',
    learningObjectives: [
      'Understand why static batching wastes compute on padding tokens',
      'Enable continuous batching to process requests without idle gaps',
      'Recognize that continuous batching improves throughput without changing the model or hardware',
    ],
    briefing: `Kepler-442b fills the viewports. After eighty years of void, the Meridian enters orbit and launches its first atmospheric probes. Within minutes, data starts streaming back — spectral readings, atmospheric composition, surface imagery — dozens of concurrent feeds, each demanding model inference.

The 70B model is loaded and serving, but the throughput is abysmal. Short probe queries — a quick spectral classification, a mineral identification — come back in milliseconds, then sit waiting. And waiting. Trapped in a batch alongside a massive atmospheric composition analysis that takes ten times longer. Nothing moves until the slowest request finishes, and every idle slot burns GPU cycles on empty tokens.

Dr. Chen watches probe feeds queue up and timeout. The model is powerful enough. The hardware is sufficient. The scheduling strategy is the bottleneck — and Kepler-442b won't wait.

Increase inference throughput to keep up with the probe data streams. The model, hardware, and precision are already optimal. The problem is how requests are scheduled.`,
    successNarrative: `Probe data flows through the model without interruption. Short requests complete and release their slots immediately; new requests fill the gaps without waiting for the longest sequence. The GPU never wastes cycles on padding tokens.

Continuous batching replaces the static batch paradigm entirely. Instead of processing a fixed batch and waiting for every sequence to finish, the scheduler inserts new requests into freed slots mid-batch. Each token position does useful work at every step.

The throughput improvement is dramatic — the same model on the same hardware processes significantly more tokens per second, simply by eliminating the scheduling waste that static batching forces.

Meanwhile, a secondary analysis thread flags something unexpected in the geological data: the planet's crust is rich in semiconductor-grade silicates. Okafor reads the mineral assay twice, then files a fabrication proposal before the last probe feed even closes. "If those silicates are what I think they are," she tells the bridge, "we can build GPUs down there."`,
    primaryMode: 'inference',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'h100-sxm',
      numGPUs: 8,
      gpusPerNode: 8,
      tensorParallel: 8,
      weightPrecision: 'bf16',
      batchSize: 16,
      inputSeqLen: 2048,
      outputSeqLen: 512,
      continuousBatching: false,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Model loads successfully' },
      { field: 'throughput.tokensPerSecond', operator: '>', value: 1700, label: 'Throughput > 1700 tok/s' },
    ],
    expectedChanges: [
      { field: 'continuousBatching', check: 'enabled', label: 'Enabled continuous batching' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU' },
      { field: 'numGPUs', check: 'unchanged', label: 'Did not change GPU count' },
      { field: 'tensorParallel', check: 'unchanged', label: 'Did not change TP' },
      { field: 'weightPrecision', check: 'unchanged', label: 'Did not change precision' },
      { field: 'batchSize', check: 'unchanged', label: 'Did not change batch size' },
      { field: 'inputSeqLen', check: 'unchanged', label: 'Did not change input sequence length' },
      { field: 'outputSeqLen', check: 'unchanged', label: 'Did not change output sequence length' },
    ],
    hints: [
      'Static batching pads every request to the length of the longest sequence in the batch. If one request needs 2048 tokens and another needs 128, both consume the same GPU resources per step. Every padded position is wasted compute — and with 16 concurrent requests, the waste compounds.',
      'There is a scheduling approach that doesn\'t wait for the entire batch to finish. Instead of padding short requests to match the longest, it lets completed requests exit and new requests enter mid-batch. Each GPU cycle processes only real tokens.',
      'Look for the Continuous Batching toggle in inference settings. It changes the scheduling strategy without altering the model, hardware, or precision. The same resources do more useful work per second.',
    ],
    prerequisites: ['mission-2-11'],
    skillsAwarded: ['batching'],
  },

  // ── Mission 3-2: Deep Signal (Context Parallelism) ──────────────────
  {
    id: 'mission-3-2',
    arcId: 'arc3-wonder',
    order: 2,
    title: 'Deep Signal',
    subtitle: 'Cut it short and you lose everything',
    learningObjectives: [
      'Understand that ultra-long sequences cause activation memory to exceed GPU capacity',
      'Use context parallelism (ring attention) to split sequences across GPUs',
      'Recognize the tradeoff between CP degree and data parallelism',
    ],
    briefing: `The biosignature data contains ultra-long repeating molecular patterns — sequences that span over a hundred thousand tokens. Dr. Chen's team has been truncating them to fit, but the truncated analyses miss the long-range dependencies that give the patterns meaning. "If we truncate below 131K tokens, we lose the long-range dependencies," Chen says. "That's not an option."

The 405B model is configured for training on the full 131K-token sequences across the newly fabricated compute cluster — sixty-four nodes of H100 GPUs, painstakingly assembled from planetary silicates by Okafor's engineering team — but the system crashes immediately. The memory diagnostic shows activation memory — the intermediate tensors stored for backward computation — exploding beyond GPU capacity. The model weights fit with FP8 quantization, but the sheer length of the sequence produces activations that overflow every GPU.

Shortening the sequence is not an option — the molecular patterns only make sense at full length. The hardware and precision are fixed. Find a way to train on the complete 131K sequences without running out of memory.`,
    successNarrative: `The full 131K-token sequences flow through the model. For the first time, the analysis captures the complete molecular pattern — long-range dependencies that were invisible at shorter contexts now resolve into clear, repeating structures.

Context parallelism splits the sequence dimension across GPUs using ring attention. Each GPU processes a chunk of the sequence and passes key-value fragments to its neighbor in a ring pattern, so every chunk attends to every other chunk without any single GPU holding the full sequence's activations.

The tradeoff is subtle: increasing CP reduces data parallelism, which increases per-GPU model state under FSDP. Too little CP and activations overflow; too much CP and model state grows back. The right balance fits everything in memory.`,
    primaryMode: 'training',
    setup: {
      modelId: 'llama3-405b',
      gpuId: 'h100-sxm',
      numGPUs: 512,
      gpusPerNode: 8,
      strategyType: 'fsdp-tp-pp',
      mixedPrecision: 'fp8',
      globalBatchSize: 128,
      microBatchSize: 1,
      sequenceLength: 131072,
      activationCheckpointing: true,
      checkpointingGranularity: 'selective',
      flashAttention: true,
      sequenceParallel: true,
      tpDegree: 8,
      ppDegree: 4,
      cpDegree: 1,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Training fits in memory' },
      { field: 'mfu', operator: '>', value: 0.20, label: 'MFU > 20%' },
    ],
    expectedChanges: [
      { field: 'cpDegree', check: 'increased', label: 'Increased context parallelism' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU' },
      { field: 'numGPUs', check: 'unchanged', label: 'Did not change GPU count' },
      { field: 'sequenceLength', check: 'unchanged', label: 'Did not change sequence length' },
      { field: 'precision', check: 'unchanged', label: 'Did not change precision' },
      { field: 'tpDegree', check: 'unchanged', label: 'Did not change TP' },
      { field: 'ppDegree', check: 'unchanged', label: 'Did not change PP' },
    ],
    hints: [
      'Activation memory scales with sequence length — each transformer layer stores intermediate tensors proportional to the number of tokens. At 131K tokens, even with selective checkpointing, the per-GPU activation footprint exceeds memory. The model weights and optimizer states fit fine under FSDP — it\'s the activations that overflow.',
      'If the sequence is too long for one GPU, it can be split across multiple GPUs. Ring attention distributes chunks of the sequence across a group of GPUs, with each GPU processing its chunk and exchanging key-value fragments with its neighbors. No single GPU needs to hold the full sequence\'s activations.',
      'CP and DP share the same GPU budget — every increase in CP comes at the cost of DP. Less DP means each GPU holds more model state under FSDP. Too little CP and activations overflow; too much and model state does. There may be only one value that fits.',
    ],
    prerequisites: ['mission-3-1'],
    skillsAwarded: ['model-parallelism'],
  },

  // ── Mission 3-3: Alien Model (Expert Parallelism) ───────────────────
  {
    id: 'mission-3-3',
    arcId: 'arc3-wonder',
    order: 3,
    title: 'Alien Model',
    subtitle: 'No single mind can hold all of it',
    learningObjectives: [
      'Understand that MoE models replicate all expert weights on every GPU without EP',
      'Use expert parallelism to distribute experts across GPUs, reducing redundancy',
      'Recognize the tradeoff between EP communication (all-to-all) and compute efficiency',
    ],
    briefing: `Nothing on Kepler-442b works the way Earth biology does. Over a hundred distinct biochemical subsystems — photosynthesis analogs, pressure-regulation cascades, silicon-based structural chemistry — each triggered by different environmental conditions. The complexity isn't in any single pathway. It's in the sheer variety.

"The alien biology routes stimuli to specialist subsystems," Patel explains, pacing the lab. "Photosynthesis analog triggers one cascade. Pressure regulation triggers a completely different one. The architecture should mirror the organism — specialists, not generalists." His team found a model built for exactly this kind of problem — massive total capacity, but designed so that only a fraction activates for any given input. It should be ideal. Instead, the utilization dashboard tells a different story: training efficiency is terrible. The memory breakdown shows the full weight replicated on every GPU, even though most of it sits idle at any given moment.

The architecture fits the biology. The distribution across the cluster does not.`,
    successNarrative: `The MoE model trains at dramatically higher efficiency. Each GPU holds a subset of the experts, and an all-to-all communication step routes tokens to the correct expert regardless of which GPU holds it.

Expert parallelism distributes expert weights across GPUs. With EP=N, each GPU stores only a fraction of the experts. Tokens are dispatched to the GPU holding the relevant expert via all-to-all communication, processed, then returned.

The tradeoff is communication: all-to-all transfers move tokens between GPUs at every MoE layer. But the efficiency gain from eliminating redundant expert compute far outweighs the communication cost. The experts are specialized — distributing them is natural.`,
    primaryMode: 'training',
    setup: {
      modelId: 'llama4-maverick',
      gpuId: 'h100-sxm',
      numGPUs: 128,
      gpusPerNode: 8,
      strategyType: 'fsdp-tp-pp',
      mixedPrecision: 'fp8',
      globalBatchSize: 128,
      microBatchSize: 2,
      sequenceLength: 4096,
      activationCheckpointing: true,
      checkpointingGranularity: 'selective',
      flashAttention: true,
      sequenceParallel: true,
      tpDegree: 8,
      ppDegree: 1,
      epDegree: 1,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Training fits in memory' },
      { field: 'mfu', operator: '>', value: 0.30, label: 'MFU > 30%' },
    ],
    expectedChanges: [
      { field: 'epDegree', check: 'increased', label: 'Increased expert parallelism' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU' },
      { field: 'numGPUs', check: 'unchanged', label: 'Did not change GPU count' },
      { field: 'sequenceLength', check: 'unchanged', label: 'Did not change sequence length' },
      { field: 'precision', check: 'unchanged', label: 'Did not change precision' },
    ],
    hints: [
      'Without expert parallelism, every GPU holds a complete copy of all expert weights. The router selects only a few active experts per token, but the rest still consume memory and their parameters still participate in FSDP communication. The redundancy wastes both memory bandwidth and compute — MFU suffers because GPU cycles go to managing inactive experts.',
      'Expert parallelism is conceptually simple: instead of replicating all experts everywhere, distribute them. Each GPU holds 1/EP of the experts. When a token routes to an expert on another GPU, an all-to-all communication step moves it there and back. The per-GPU expert memory and compute drop proportionally.',
      'Look for the Expert Parallelism (EP) setting. EP must divide the data parallel degree. Even EP=2 distributes experts across pairs of GPUs, halving the redundancy. Higher EP reduces redundancy further but increases all-to-all communication volume.',
    ],
    prerequisites: ['mission-3-1'],
    skillsAwarded: ['model-parallelism'],
  },

  // ── Mission 3-4: Big Train (Pipeline Schedule Optimization) ─────────
  {
    id: 'mission-3-4',
    arcId: 'arc3-wonder',
    order: 4,
    title: 'Big Train',
    subtitle: 'Humanity\'s most important training run',
    learningObjectives: [
      'Understand the pipeline bubble as wasted GPU time during ramp-up/ramp-down',
      'Compare 1F1B schedule (large bubble) vs interleaved 1F1B (smaller bubble)',
      'Configure interleaved scheduling with appropriate virtual pipeline stages',
    ],
    briefing: `The crew is building the translation model — the most important model humanity will ever train. The 405B model runs across the full compute cluster, layers divided across a deep chain of stages. But Chief Engineer Okafor watches the utilization trace and sees a damning pattern: at every training step, GPUs flicker to life one by one, hold steady for a brief window of real work, then go dark one by one. The idle time at the edges dwarfs the productive time in the middle.

"I stopped checking the manual," Okafor says, forwarding you the utilization trace. "I just send it to you now."

The chain can't be shortened — the model needs every stage to fit in memory. But the dead time isn't inevitable. It depends on how work flows through.

Same depth. Same hardware. There has to be a better way to fill the gaps.`,
    successNarrative: `The pipeline bubble shrinks dramatically. GPUs that were idle during ramp-up and ramp-down now process virtual stages — smaller chunks of layers that interleave through the pipeline more efficiently.

Interleaved 1F1B divides each physical pipeline stage into multiple virtual stages. Instead of each GPU processing all its layers as one block, it processes them in smaller interleaved chunks. This reduces the bubble from proportional to the number of physical stages to proportional to the number of virtual stages — significantly smaller.

The translation model trains faster. Each step completes with less wasted time, and the total time-to-completion drops. The translation model will be ready before the window opens.`,
    primaryMode: 'training',
    setup: {
      modelId: 'llama3-405b',
      gpuId: 'h100-sxm',
      numGPUs: 128,
      gpusPerNode: 8,
      strategyType: 'fsdp-tp-pp',
      mixedPrecision: 'bf16',
      globalBatchSize: 64,
      microBatchSize: 1,
      sequenceLength: 8192,
      activationCheckpointing: true,
      checkpointingGranularity: 'selective',
      flashAttention: true,
      sequenceParallel: true,
      tpDegree: 8,
      ppDegree: 8,
      pipelineSchedule: '1f1b',
      interleavedStages: 1,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Training fits in memory' },
      { field: 'mfu', operator: '>', value: 0.37, label: 'MFU > 37%' },
    ],
    expectedChanges: [
      { field: 'pipelineSchedule', check: 'changed', label: 'Changed pipeline schedule' },
      { field: 'interleavedStages', check: 'increased', label: 'Increased interleaved stages' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU' },
      { field: 'numGPUs', check: 'unchanged', label: 'Did not change GPU count' },
      { field: 'ppDegree', check: 'unchanged', label: 'Did not change pipeline parallelism' },
      { field: 'sequenceLength', check: 'unchanged', label: 'Did not change sequence length' },
    ],
    hints: [
      'Picture the pipeline: the first micro-batch enters stage 1 while stages 2 through N sit idle. By the time it reaches the last stage, every GPU upstream has been waiting. Then the same ramp-down happens in reverse. The more stages in the chain, the more time GPUs spend dark at the edges of each step.',
      'If each physical stage is divided into smaller virtual stages, the pipeline can interleave them more tightly. Instead of one big forward-backward block per stage, multiple smaller blocks overlap, reducing the idle time at ramp-up and ramp-down. The key is the pipeline schedule type and the number of virtual stages.',
      'The current schedule type ignores the interleaved stages setting entirely. Look for a different pipeline schedule that actually uses virtual stages. More virtual stages means tighter interleaving — try increasing them until the bubble shrinks enough to clear the MFU target.',
    ],
    prerequisites: ['mission-3-1', 'mission-2-9'],
    skillsAwarded: ['model-parallelism'],
  },

  // ── Mission 3-5: Resource War (Multi-Objective Allocation) ──────────
  {
    id: 'mission-3-5',
    arcId: 'arc3-wonder',
    order: 5,
    title: 'Resource War',
    subtitle: 'Three science teams. One cluster.',
    learningObjectives: [
      'Reduce activation memory and increase compute utilization for high training MFU',
      'Apply LoRA/QLoRA to fit large model fine-tuning on limited hardware',
      'Balance inference batch sizing and continuous batching for throughput',
    ],
    briefing: `Three days until the transmission window opens. Three science teams in the same corridor, each convinced their workload matters most.

Chen's team needs to train the biosignature sequence model — the foundation for everything the crew will say to whatever is down there. Patel's group is fine-tuning the 70B model on the latest protein analysis data, but their full fine-tune keeps crashing. "The full fine-tune has crashed four times," Patel says. "I won't reduce the model — if we use something smaller, we miss the enzymatic subtleties." And Okafor's probe analysis pipeline needs real-time inference on the A100 array — batch-one latency won't cut it when the probes are streaming data from the surface.

Lindqvist makes the call: each team gets its own hardware pool. No sharing, no queuing. Chen pulls up her config and hands it to you without a word. She knows you'll see the problem before she finishes explaining it.

Three objectives. Complete them all.`,
    successNarrative: `All three science teams run simultaneously. Chen's biosignature model trains at peak efficiency with the right precision and memory management. Patel's 70B fine-tune fits in memory using parameter-efficient methods. Okafor's probes stream data through the inference pipeline at high throughput.

Resource allocation in production ML is never about one workload. Training demands throughput and GPU utilization. Fine-tuning demands memory efficiency. Inference demands latency and concurrent serving. Each requires different optimization strategies — precision, parallelism, batching, adapter methods — applied to different hardware constraints.

The models are ready. What happens next is not a compute problem.`,
    primaryMode: 'training',
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 'h100-sxm',
      numGPUs: 8,
      gpusPerNode: 8,
      strategyType: 'fsdp',
      mixedPrecision: 'bf16',
      globalBatchSize: 64,
      microBatchSize: 4,
      sequenceLength: 8192,
      activationCheckpointing: false,
      flashAttention: true,
    },
    winningCriteria: [],
    expectedChanges: [],
    objectives: [
      {
        id: 'obj-biosig-train',
        label: 'Training: Biosignature sequence model',
        primaryMode: 'training',
        setup: {
          modelId: 'llama3.1-8b',
          gpuId: 'h100-sxm',
          numGPUs: 8,
          gpusPerNode: 8,
          strategyType: 'fsdp',
          mixedPrecision: 'bf16',
          globalBatchSize: 64,
          microBatchSize: 4,
          sequenceLength: 8192,
          activationCheckpointing: false,
          flashAttention: true,
        },
        winningCriteria: [
          { field: 'success', operator: '==', value: true, label: 'Training fits in memory' },
          { field: 'mfu', operator: '>', value: 0.50, label: 'MFU > 50%' },
        ],
        expectedChanges: [
          { field: 'precision', check: 'changed', label: 'Changed training precision' },
          { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
          { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU' },
          { field: 'numGPUs', check: 'unchanged', label: 'Did not change GPU count' },
          { field: 'sequenceLength', check: 'unchanged', label: 'Did not change sequence length' },
        ],
      },
      {
        id: 'obj-finetune',
        label: 'Training: Large model fine-tune',
        primaryMode: 'training',
        setup: {
          modelId: 'llama3.3-70b',
          gpuId: 'h100-sxm',
          numGPUs: 8,
          gpusPerNode: 8,
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
      },
      {
        id: 'obj-probe-infer',
        label: 'Inference: Real-time probe analysis',
        primaryMode: 'inference',
        setup: {
          modelId: 'llama3.3-70b',
          gpuId: 'a100-80gb',
          numGPUs: 4,
          tensorParallel: 4,
          weightPrecision: 'bf16',
          batchSize: 1,
          inputSeqLen: 2048,
          outputSeqLen: 512,
          continuousBatching: false,
        },
        winningCriteria: [
          { field: 'success', operator: '==', value: true, label: 'Model loads successfully' },
          { field: 'throughput.tokensPerSecond', operator: '>', value: 800, label: 'Throughput > 800 tok/s' },
        ],
        expectedChanges: [
          { field: 'batchSize', check: 'increased', label: 'Increased batch size' },
          { field: 'continuousBatching', check: 'enabled', label: 'Enabled continuous batching' },
          { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
          { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU' },
          { field: 'numGPUs', check: 'unchanged', label: 'Did not change GPU count' },
        ],
      },
    ],
    hints: [
      'The biosignature model OOMs from activations at BF16. Reduce what each GPU stores for the backward pass and switch to a precision the H100 supports natively for higher TFLOPS.',
      'The inference objective starts with batch=1 and no continuous batching. Both are throughput killers. Larger batches amortize weight reads across more tokens, and continuous batching eliminates padding waste. Together they can dramatically increase tokens per second.',
      'For the fine-tune, full fine-tuning of 70B parameters requires optimizer state for every weight — far exceeding available memory. LoRA and QLoRA freeze the base model and train only small adapter matrices, eliminating most optimizer memory. The model still learns effectively with a tiny fraction of trainable parameters.',
    ],
    prerequisites: ['mission-3-2', 'mission-3-3', 'mission-3-4'],
    skillsAwarded: ['resource-efficiency'],
  },

  // ── Mission 3-6: First Contact Protocol (Capstone) ──────────────────
  {
    id: 'mission-3-6',
    arcId: 'arc3-wonder',
    order: 6,
    title: 'All Systems Nominal',
    subtitle: 'One chance to say hello',
    learningObjectives: [
      'Combine context parallelism with pipeline scheduling for long-context training',
      'Combine expert parallelism with FP8 precision for MoE training efficiency',
      'Apply continuous batching with weight quantization for inference throughput',
      'Diagnose cross-node TP latency and reduce to intra-node TP',
    ],
    briefing: `The message will be transmitted in six hours. Four systems must be operational simultaneously — the long-context protocol model, the expert translation engine, the high-throughput data feed, and the low-latency response translator. Each system uses different hardware, different models, and different optimization strategies. Each one must work.

This is not a drill. This is not an exercise. The crew has spent eighty years crossing interstellar space. Everything learned about compute — from the first batch size optimization on a pair of T4s to the multi-node pipeline schedules that trained the translation model — converges here.

Captain Lindqvist stares at the transmission parameters one last time. Eight minutes to the surface. If this goes wrong, there's no one to call. He turns to you. "I used to assign hardware pools. Now I just ask how many you need." A pause. "All stations report ready."

Four objectives. All must pass. No new concepts — just mastery under pressure.`,
    successNarrative: `All four systems report green. The long-context protocol model holds the full 131K-token sequence in memory — context parallelism splitting what no single GPU could hold, interleaved pipeline scheduling filling the gaps that would otherwise waste half the cluster. The MoE translation engine distributes its experts across the node, FP8 arithmetic doubling the throughput that BF16 left on the table. The probe feed streams through continuous batching at thousands of tokens per second, weight quantization turning memory savings into raw speed. And the response translator — TP confined to a single node, fast interconnect doing what cross-node fabric never could — answers in milliseconds.

The transmission window opens in six hours. The systems are ready. The crew is ready. What comes back is not a compute problem.`,
    primaryMode: 'training',
    setup: {
      modelId: 'llama3-405b',
      gpuId: 'h100-sxm',
      numGPUs: 512,
      gpusPerNode: 8,
      strategyType: 'fsdp-tp-pp',
      mixedPrecision: 'fp8',
      globalBatchSize: 128,
      microBatchSize: 1,
      sequenceLength: 131072,
      activationCheckpointing: true,
      checkpointingGranularity: 'selective',
      flashAttention: true,
      sequenceParallel: true,
      tpDegree: 8,
      ppDegree: 4,
      cpDegree: 1,
      pipelineSchedule: '1f1b',
      interleavedStages: 2,
    },
    winningCriteria: [],
    expectedChanges: [],
    objectives: [
      {
        id: 'obj-longctx-train',
        label: 'Training: Long-context protocol model',
        primaryMode: 'training',
        setup: {
          modelId: 'llama3-405b',
          gpuId: 'h100-sxm',
          numGPUs: 512,
          gpusPerNode: 8,
          strategyType: 'fsdp-tp-pp',
          mixedPrecision: 'fp8',
          globalBatchSize: 128,
          microBatchSize: 1,
          sequenceLength: 131072,
          activationCheckpointing: true,
          checkpointingGranularity: 'selective',
          flashAttention: true,
          sequenceParallel: true,
          tpDegree: 8,
          ppDegree: 4,
          cpDegree: 1,
          pipelineSchedule: '1f1b',
          interleavedStages: 2,
        },
        winningCriteria: [
          { field: 'success', operator: '==', value: true, label: 'Training fits in memory' },
          { field: 'mfu', operator: '>', value: 0.255, label: 'MFU > 25.5%' },
        ],
        expectedChanges: [
          { field: 'cpDegree', check: 'increased', label: 'Increased context parallelism' },
          { field: 'pipelineSchedule', check: 'changed', label: 'Changed pipeline schedule' },
          { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
          { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU' },
          { field: 'numGPUs', check: 'unchanged', label: 'Did not change GPU count' },
          { field: 'sequenceLength', check: 'unchanged', label: 'Did not change sequence length' },
          { field: 'tpDegree', check: 'unchanged', label: 'Did not change TP' },
          { field: 'ppDegree', check: 'unchanged', label: 'Did not change PP' },
        ],
      },
      {
        id: 'obj-moe-train',
        label: 'Training: MoE expert translation engine',
        primaryMode: 'training',
        setup: {
          modelId: 'llama4-maverick',
          gpuId: 'h100-sxm',
          numGPUs: 128,
          gpusPerNode: 8,
          strategyType: 'fsdp-tp-pp',
          mixedPrecision: 'bf16',
          globalBatchSize: 128,
          microBatchSize: 2,
          sequenceLength: 4096,
          activationCheckpointing: true,
          checkpointingGranularity: 'selective',
          flashAttention: true,
          sequenceParallel: true,
          tpDegree: 8,
          ppDegree: 1,
          epDegree: 1,
        },
        winningCriteria: [
          { field: 'success', operator: '==', value: true, label: 'Training fits in memory' },
          { field: 'mfu', operator: '>', value: 0.40, label: 'MFU > 40%' },
        ],
        expectedChanges: [
          { field: 'epDegree', check: 'increased', label: 'Increased expert parallelism' },
          { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
          { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU' },
          { field: 'numGPUs', check: 'unchanged', label: 'Did not change GPU count' },
          { field: 'sequenceLength', check: 'unchanged', label: 'Did not change sequence length' },
        ],
      },
      {
        id: 'obj-feed-infer',
        label: 'Inference: High-throughput probe feed',
        primaryMode: 'inference',
        setup: {
          modelId: 'llama3.3-70b',
          gpuId: 'h100-sxm',
          numGPUs: 8,
          gpusPerNode: 8,
          tensorParallel: 8,
          weightPrecision: 'bf16',
          batchSize: 32,
          inputSeqLen: 4096,
          outputSeqLen: 1024,
          continuousBatching: false,
        },
        winningCriteria: [
          { field: 'success', operator: '==', value: true, label: 'Model loads successfully' },
          { field: 'throughput.tokensPerSecond', operator: '>', value: 4000, label: 'Throughput > 4000 tok/s' },
        ],
        expectedChanges: [
          { field: 'weightPrecision', check: 'changed', label: 'Changed weight precision' },
          { field: 'continuousBatching', check: 'enabled', label: 'Enabled continuous batching' },
          { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
          { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU' },
          { field: 'numGPUs', check: 'unchanged', label: 'Did not change GPU count' },
        ],
      },
      {
        id: 'obj-latency-infer',
        label: 'Inference: Low-latency translation',
        primaryMode: 'inference',
        setup: {
          modelId: 'llama3-405b',
          gpuId: 'h100-sxm',
          numGPUs: 16,
          gpusPerNode: 8,
          tensorParallel: 16,
          weightPrecision: 'int8',
          batchSize: 1,
          inputSeqLen: 1024,
          outputSeqLen: 256,
        },
        winningCriteria: [
          { field: 'success', operator: '==', value: true, label: 'Model loads successfully' },
          { field: 'latency.tpot', operator: '<', value: 23, label: 'TPOT < 23ms' },
        ],
        expectedChanges: [
          { field: 'tensorParallel', check: 'decreased', label: 'Reduced tensor parallelism' },
          { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
          { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU' },
          { field: 'numGPUs', check: 'unchanged', label: 'Did not change GPU count' },
        ],
      },
    ],
    hints: [
      'The long-context objective combines two Arc 3 skills: the sequence is too long at CP=1 (activation memory overflow), and the basic 1F1B schedule wastes pipeline time. You need both context parallelism AND interleaved scheduling. Think back to missions 3-2 and 3-4.',
      'The MoE training objective starts with BF16 and EP=1 — expert weights replicated everywhere, low efficiency. You need EP to distribute experts, and the precision format that the H100 supports natively for higher compute throughput. Both changes together push MFU above the target.',
      'The inference objectives are about combining techniques. The probe feed needs both weight quantization (to increase effective compute throughput) and continuous batching. The translation needs to recognize that TP=16 crosses node boundaries — the high-bandwidth intra-node interconnect only connects GPUs within a single node.',
    ],
    prerequisites: ['mission-3-5'],
    skillsAwarded: [],
  },

  // ── Mission 3-7: PIVOT — The Reply ──────────────────────────────────
  {
    id: 'mission-3-7',
    arcId: 'arc3-wonder',
    order: 7,
    title: 'The Reply',
    subtitle: 'Three hours of silence. Then — a response',
    type: 'pivot',
    learningObjectives: [],
    briefing: `At 14:37 ship time, the transmission fires. Eighty years of travel, months of analysis, days of frantic optimization — compressed into a single electromagnetic pulse aimed at the planet below.

The bridge crew holds its breath. Somewhere below decks, two hundred colonists sleep on, oblivious. The signal propagates at the speed of light — eight minutes to the surface, eight minutes for a response to return. Sixteen minutes of existential uncertainty.

The minutes pass. Chen monitors the receiver array, eyes fixed on the spectral display, counting dead cycles. Patel double-checks the biosignature decoder, hands not quite steady. Okafor keeps the inference pipeline warm, ready to process whatever comes back — she hasn't looked away from the utilization dashboard in twenty minutes.

Sixteen minutes. Nothing. Thirty minutes. Nothing. An hour. The silence stretches. "Maintain stations," Lindqvist says quietly. No one moves.`,
    successNarrative: `Three hours and eleven minutes after transmission, the receiver array spikes.

The signal is not an echo. Not a reflection of what was sent. It carries a different structure — new patterns built on the same molecular encoding the crew has spent weeks learning to decode. The biosignature model processes the incoming data in real-time, its continuous batching pipeline handling the stream without missing a beat.

The translation model produces its first output: the alien response contains references to the crew's original message. They answered.

Patel watches the biosignature model output scroll, hands trembling. "It's referencing our molecular encoding," he whispers. "It learned our format." Chen reads the translation aloud to the bridge, her voice precise even now. Okafor stands behind the compute bay console, arms crossed, watching every GPU utilization metric hold steady. Lindqvist stares at the viewscreen for a long moment, then speaks into the ship-wide channel:

"All hands. This is the Captain. We came looking for a home. We found neighbors."

The reply is just the beginning.`,
    primaryMode: 'inference',
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 't4',
    },
    winningCriteria: [],
    hints: [],
    prerequisites: ['mission-3-6'],
    skillsAwarded: [],
  },
];
