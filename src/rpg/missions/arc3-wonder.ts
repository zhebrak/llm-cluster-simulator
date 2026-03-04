/**
 * Arc 3: Wonder — the ship reaches Ross 128b and prepares for first contact.
 * Missions teach: continuous batching, context parallelism, expert parallelism,
 * pipeline scheduling, multi-workload allocation, and full-system capstone.
 * See RPG Text Style Guide in ../types.ts for briefing/hint/success narrative rules.
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
 *   3-5 (Resource War: multi-obj allocation) <- requires 2-6 + 3-2 + 3-3 + 3-4
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
  briefing: `Eighty years of void. The Meridian's hull is scored by micrometeorites, its systems patched and repatched by generations of engineers who never saw the sky. But now — Ross 128b glows in the forward viewports, brighter each day.

The planet is alive. Dr. Patel's enzymatic analysis confirmed it: self-replicating molecular systems, metabolic cycles, evolutionary pressure operating on chemistry no Earth biochemist has ever seen. Life.

Nobody planned for this. The Meridian was built to settle an empty world, not to comprehend a living one. Every probe return reveals deeper complexity. Whatever is broadcasting from the surface — can it hear us?`,
  heroImage: { dark: '/life_dark.png', light: '/life_light.png' },
};

/** Post-game state: shown in mission log after all arcs are complete. */
export const COMPLETION: RPGArc = {
  id: 'completion',
  name: 'First Contact',
  subtitle: 'The reply is just the beginning',
  description: '',
  order: 4,
  briefing: `Ship's log, GSV Meridian. Contact day one.

The reply arrived three hours and eleven minutes after transmission. Whatever is on the surface studied what we sent and answered in kind.

Chen is running the translation model around the clock. Patel has not left the biosignature console since the first signal resolved. Okafor started a second fabrication run overnight — more compute modules from the belt silicates. She says we'll need the capacity. She hasn't stopped smiling.

Captain Lindqvist recorded a message for the colonists. They'll hear it when they wake: "We came looking for a home. We found neighbors."

The reply is just the beginning.`,
  heroImage: { dark: '/contact_dark.png', light: '/contact_light.png' },
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
    briefing: `The Meridian decelerates into the inner system, eighty years of void finally giving way to starlight on hull plating. Atmospheric probes dispatched during approach have reached the surface, and data streams back across the light-delay gap — spectral readings, atmospheric composition, surface imagery — a constant stream of concurrent feeds, each demanding model inference.

The 70B model is loaded and serving, but the throughput is abysmal. Short probe queries — a quick spectral classification, a mineral identification — come back in milliseconds, then sit waiting. And waiting. Trapped in a batch alongside a massive atmospheric composition analysis that takes ten times longer. Nothing moves until the slowest request finishes, and every idle slot burns GPU cycles on empty tokens.

Dr. Chen watches probe feeds queue up and timeout. The model is powerful enough. The hardware is sufficient. The scheduling strategy is the bottleneck — and Ross 128b won't wait.

Increase inference throughput to keep up with the probe data streams. The model, hardware, and precision are already optimal. The problem is how requests are scheduled.`,
    successNarrative: `Probe data flows through the model without interruption. Short requests complete and release their slots immediately; new requests fill the gaps without waiting for the longest sequence. The GPU never wastes cycles on padding tokens.

Continuous batching replaces the static batch paradigm entirely. Instead of processing a fixed batch and waiting for every sequence to finish, the scheduler inserts new requests into freed slots mid-batch. Each token position does useful work at every step.

The throughput improvement is dramatic — the same model on the same hardware processes significantly more tokens per second, simply by eliminating the scheduling waste that static batching forces.

Meanwhile, a secondary analysis thread flags something unexpected in the probe data: the inner asteroid belt is rich in semiconductor-grade silicates. Okafor reads the mineral assay twice, then dispatches mining drones before the last probe feed even closes.

Her engineering team sets up a fabrication line in the Meridian's cargo bay. The silicates are better than she hoped — pure enough for semiconductor-grade wafers. Node after node comes online: dozens of compute modules, each packed with GPUs forged from asteroid material. "We arrived with two nodes," Okafor reports to the bridge. "We now have a proper cluster."`,
    primaryMode: 'inference',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'h100-sxm',
      numGPUs: 8,
      gpusPerNode: 8,
      tensorParallel: 8,
      weightPrecision: 'fp8',
      batchSize: 128,
      inputSeqLen: 2048,
      outputSeqLen: 512,
      continuousBatching: false,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Model loads successfully' },
      { field: 'throughput.tokensPerSecond', operator: '>', value: 12500, label: 'Throughput > 12,500 tok/s' },
    ],
    expectedChanges: [
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU' },
      { field: 'numGPUs', check: 'unchanged', label: 'Did not change GPU count' },
    ],
    hints: [
      'Static batching pads every request to the length of the longest sequence in the batch. If one request needs 2048 tokens and another needs 128, both consume the same GPU resources per step. Every padded position is wasted compute — and with hundreds of concurrent requests, the waste compounds.',
      'There is a scheduling approach that doesn\'t wait for the entire batch to finish. Instead of padding short requests to match the longest, it lets completed requests exit and new requests enter mid-batch. Each GPU cycle processes only real tokens.',
      'Look for the Continuous Batching toggle in inference settings. It changes the scheduling strategy without altering the model, hardware, or precision. The same resources do more useful work per second.',
    ],
    prerequisites: ['mission-2-11', 'mission-1-4'],
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

The 405B model is configured for training on the full 131K-token sequences across the largest slice of Okafor's asteroid-fabricated cluster they can get — but the system crashes immediately. The memory diagnostic tells the story: the model weights fit with compression, but the working memory needed during training explodes beyond GPU capacity. Every GPU overflows.

The training pipeline is already battle-tested — the parallelism layout, checkpointing strategy, and training procedure were locked down after weeks of convergence tuning. Chen's team won't risk destabilizing what works. The only thing new is the sequence length, and shortening it is not an option — the molecular patterns only make sense at full length. Find a way to process the complete 131K sequences without restructuring the existing pipeline.`,
    successNarrative: `The full 131K-token sequences flow through the model. For the first time, the analysis captures the complete molecular pattern — long-range dependencies that were invisible at shorter contexts now resolve into coherent patterns.

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
      { field: 'checkpointingGranularity', check: 'unchanged', label: 'Did not change checkpointing strategy' },
      { field: 'finetuningMethod', check: 'unchanged', label: 'Did not change training method' },
      { field: 'strategyType', check: 'unchanged', label: 'Did not change strategy' },
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
    briefing: `Nothing on Ross 128b works the way Earth biology does. Over a hundred distinct biochemical subsystems — photosynthesis analogs, pressure-regulation cascades, silicon-based structural chemistry — each triggered by different environmental conditions. The complexity isn't in any single pathway. It's in the sheer variety.

Okafor allocates a section of the asteroid-fabricated cluster for the biology team. "The alien biology routes stimuli to specialist subsystems," Patel explains, pacing the lab. "Photosynthesis analog triggers one cascade. Pressure regulation triggers a completely different one. The architecture should mirror the organism — specialists, not generalists." His team found a model built for exactly this kind of problem — massive total capacity, but designed so that only a fraction activates for any given input. It should be ideal. Instead, the utilization dashboard tells a different story: training efficiency is terrible. The memory breakdown shows the full weight replicated on every GPU, even though most of it sits idle at any given moment.

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
      { field: 'mfu', operator: '>', value: 0.55, label: 'MFU > 55%' },
    ],
    expectedChanges: [
      { field: 'epDegree', check: 'increased', label: 'Increased expert parallelism' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU' },
      { field: 'numGPUs', check: 'unchanged', label: 'Did not change GPU count' },
      { field: 'precision', check: 'unchanged', label: 'Did not change precision' },
    ],
    hints: [
      'Without expert parallelism, every GPU holds a complete copy of all expert weights — but the router activates only a few per token. The rest consume memory and FSDP communication bandwidth for weights the GPU never computes on.',
      'Expert parallelism is conceptually simple: instead of replicating all experts everywhere, distribute them. Each GPU holds 1/EP of the experts. When a token routes to an expert on another GPU, an all-to-all communication step moves it there and back. The per-GPU expert memory and compute drop proportionally.',
      'Look for the Expert Parallelism (EP) setting. EP must divide the data parallel degree. With 128 experts, you need enough EP to meaningfully reduce the per-GPU expert count — too little and the communication overhead outweighs the savings. Higher EP reduces redundancy further but increases all-to-all volume.',
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
    briefing: `The crew is building the translation model — the most important model humanity will ever train. The 405B model runs across a dedicated partition of Okafor's asteroid-fabricated cluster, layers divided across a deep chain of stages. But Okafor watches the utilization trace and sees a damning pattern: at every training step, GPUs flicker to life one by one, hold steady for a brief window of real work, then go dark one by one. The idle time at the edges dwarfs the productive time in the middle.

"I stopped checking the manual," Okafor says, forwarding you the utilization trace. "I just send it to you now."

The chain can't be shortened — the model needs every stage to fit in memory. The precision and batch size were set during Phase 1 — changing them would require reconverging from checkpoint. But the dead time isn't inevitable. It depends on how work flows through.

Same depth. Same hardware. The dead time can't be inevitable.`,
    successNarrative: `The pipeline bubble shrinks dramatically. GPUs that were idle during ramp-up and ramp-down now process virtual stages — smaller chunks of layers that interleave through the pipeline more efficiently.

Interleaved 1F1B divides each physical pipeline stage into multiple virtual stages. Instead of each GPU processing all its layers as one block, it processes them in smaller interleaved chunks. This reduces the bubble by roughly a factor of v — each physical stage is subdivided into smaller chunks that interleave through the pipeline more tightly, significantly shrinking the idle time at each step's edges.

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
      { field: 'mfu', operator: '>', value: 0.39, label: 'MFU > 39%' },
      { field: 'pipelineBubble', operator: '<', value: 0.10, label: 'Pipeline bubble < 10%' },
    ],
    expectedChanges: [
      { field: 'pipelineSchedule', check: 'changed', label: 'Changed pipeline schedule' },
      { field: 'interleavedStages', check: 'increased', label: 'Increased interleaved stages' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU' },
      { field: 'numGPUs', check: 'unchanged', label: 'Did not change GPU count' },
      { field: 'ppDegree', check: 'unchanged', label: 'Did not change pipeline parallelism' },
      { field: 'precision', check: 'unchanged', label: 'Did not change precision' },
      { field: 'globalBatchSize', check: 'unchanged', label: 'Did not change batch size' },
    ],
    hints: [
      'Picture the pipeline: the first micro-batch enters stage 1 while stages 2 through N sit idle. By the time it reaches the last stage, every GPU upstream has been waiting. Then the same ramp-down happens in reverse. The more stages in the chain, the more time GPUs spend dark at the edges of each step.',
      'If each physical stage is divided into smaller virtual stages, the pipeline can interleave them more tightly. Instead of one big forward-backward block per stage, multiple smaller blocks overlap, reducing the idle time at ramp-up and ramp-down. The key is the pipeline schedule type and the number of virtual stages.',
      'The pipeline\'s depth can\'t change mid-training, but the schedule can. The current schedule type ignores the interleaved stages setting entirely — look for one that uses virtual stages. More virtual stages means tighter interleaving. Try increasing them until the bubble shrinks enough to clear the MFU target.',
    ],
    prerequisites: ['mission-3-1', 'mission-2-9'],
    skillsAwarded: ['resource-efficiency'],
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

Chen's team needs to train the biosignature sequence model — the foundation for everything the crew will say to whatever is on the surface. Patel's group is fine-tuning the 70B model on the latest protein analysis data, but their full fine-tune keeps crashing. "The full fine-tune has crashed four times," Patel says. "I won't reduce the model — if we use something smaller, we miss the enzymatic subtleties." And Okafor's probe analysis pipeline needs real-time inference on the A100 array — batch-one latency won't cut it when the probes are streaming data from the surface.

Lindqvist makes the call: each team gets its own hardware pool. Chen pulls up her config and hands it to you without a word. She knows you'll see the problem before she finishes explaining it.

Three objectives. Complete them all.`,
    successNarrative: `All three science teams run simultaneously. Chen's biosignature model trains at peak efficiency — memory optimization solved the overflow, and FP8 unlocked the compute throughput the H100s were built for. Patel's 70B fine-tune fits in memory using parameter-efficient methods. Okafor's probes stream data through the inference pipeline at high throughput.

Three teams, three hardware pools, three different bottlenecks. The training run needed raw compute — memory sharding and FP8 to keep the GPUs saturated. The fine-tune needed memory headroom — adapter methods to sidestep the full optimizer footprint. The inference pipeline needed speed — low latency and enough parallelism to keep pace with the probe stream. No single configuration could have done all three.

The models are ready. Three days until the window opens.`,
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
          { field: 'latency.ttft', operator: '<', value: 750, label: 'TTFT < 750ms' },
        ],
        expectedChanges: [
          { field: 'batchSize', check: 'increased', label: 'Increased batch size' },
          { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
          { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU' },
          { field: 'numGPUs', check: 'unchanged', label: 'Did not change GPU count' },
        ],
      },
    ],
    hints: [
      'Look at what\'s consuming memory in the biosignature config — model state fits under FSDP, but the forward pass stores far more at this sequence length. There\'s more than one way to reduce what each GPU has to hold. And once memory is solved, consider what the H100 hardware can deliver beyond standard precision.',
      'The 70B fine-tune crashes because optimizer state scales with trainable parameter count. What if most parameters didn\'t need gradients at all?',
      'The probe pipeline processes one request at a time. Batching helps throughput, but watch what happens to TTFT as batch size grows under static batching — prefill isn\'t free when every request in the batch has to wait.',
    ],
    prerequisites: ['mission-2-6', 'mission-3-2', 'mission-3-3', 'mission-3-4'],
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
      'Use expert parallelism to improve MoE training efficiency',
      'Apply batch scaling and continuous batching for inference throughput',
      'Diagnose cross-node TP latency and reduce to intra-node TP',
    ],
    briefing: `The message will be transmitted in six hours. Four systems must be operational — and none of them are ready. A protocol model holding the full transmission in context, the compute pipeline running at capacity. A translation engine drawing on its specialists without waste. A data feed serving concurrent transmissions. A response path that translates instantly. Each system needs a different fix. Each one must work.

The crew has spent eighty years crossing interstellar space. Everything learned about compute — from the first throughput fix on a pair of T4s to the multi-node training runs that built the translation model — converges here.

Captain Lindqvist stares at the transmission parameters one last time. Eight minutes to the surface. If this goes wrong, there's no one to call. He turns to you. "I used to assign hardware pools. Now I just ask how many you need." A pause. "All stations report ready."`,
    successNarrative: `All four systems report green. The long-context protocol model holds the full 131K-token sequence in memory — context parallelism splitting what no single GPU could hold, interleaved pipeline scheduling filling the gaps that would otherwise waste half the cluster. The MoE translation engine distributes its experts across the node, FP8 arithmetic doubling the throughput that BF16 left on the table. The probe feed streams through continuous batching at thousands of tokens per second, serving requests as they arrive without the padding waste of static batches. And the response translator — TP confined to a single node, fast interconnect doing what cross-node fabric never could — answers in milliseconds.

The transmission window opens in six hours. The systems are ready. The crew is ready. What happens next is not a compute problem.`,
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
          { field: 'globalBatchSize', check: 'unchanged', label: 'Did not change batch size' },
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
          { field: 'mfu', operator: '>', value: 0.55, label: 'MFU > 55%' },
        ],
        expectedChanges: [
          { field: 'epDegree', check: 'increased', label: 'Increased expert parallelism' },
          { field: 'globalBatchSize', check: 'unchanged', label: 'Did not change batch size' },
          { field: 'sequenceLength', check: 'unchanged', label: 'Did not change sequence length' },
          { field: 'tpDegree', check: 'unchanged', label: 'Did not change TP' },
          { field: 'modelId', check: 'unchanged', label: 'Did not change model' },
          { field: 'gpuId', check: 'unchanged', label: 'Did not change GPU' },
          { field: 'numGPUs', check: 'unchanged', label: 'Did not change GPU count' },
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
          weightPrecision: 'fp8',
          batchSize: 1,
          inputSeqLen: 4096,
          outputSeqLen: 1024,
          continuousBatching: false,
        },
        winningCriteria: [
          { field: 'success', operator: '==', value: true, label: 'Model loads successfully' },
          { field: 'throughput.tokensPerSecond', operator: '>', value: 4000, label: 'Throughput > 4000 tok/s' },
          { field: 'latency.ttft', operator: '<', value: 500, label: 'TTFT < 500ms' },
        ],
        expectedChanges: [
          { field: 'batchSize', check: 'increased', label: 'Increased batch size' },
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
          { field: 'throughput.tokensPerSecond', operator: '>', value: 89, label: 'Throughput > 89 tok/s' },
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
      'Four objectives, four different bottlenecks. One overflows memory on long sequences. One wastes compute on replicated specialists. One serves requests one at a time. One sends data across slow fabric when fast fabric is available nearby.',
      'The training problems both need two things: a parallelism change that fixes the structural issue, plus a compute or scheduling optimization to clear the target. The inference problems are about how work is distributed — one needs more concurrent requests with smarter scheduling, the other needs its communication to stay on fast interconnect.',
      'Split the sequence dimension and fix the pipeline schedule. Distribute the experts and upgrade arithmetic precision. Scale the batch and change how requests are scheduled. Check how many GPUs fit on one node.',
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
    subtitle: 'Not an echo',
    type: 'pivot',
    learningObjectives: [],
    briefing: `At 14:37 ship time, the transmission fires. Eighty years of travel, months of analysis, days of frantic optimization — compressed into a single electromagnetic pulse aimed at the planet below.

The bridge crew holds its breath. Somewhere below decks, the colonists sleep on, oblivious. The signal propagates at the speed of light — eight minutes to the surface, eight minutes for a response to return. Sixteen minutes of existential uncertainty.

The minutes pass. Chen monitors the receiver array, eyes fixed on the spectral display, counting dead cycles. Patel double-checks the biosignature decoder, hands not quite steady. Okafor keeps the inference pipeline warm, ready to process whatever comes back — she hasn't looked away from the utilization dashboard in twenty minutes.

Sixteen minutes. Nothing. Thirty minutes. Nothing. An hour. The silence stretches. "Maintain stations," Lindqvist says quietly. No one moves.`,
    successNarrative: `Three hours and eleven minutes after transmission, the receiver array spikes.

The signal is not an echo. Not a reflection of what was sent. It carries a different structure — new patterns built on the same molecular encoding the crew has spent weeks learning to decode. The biosignature model processes the incoming data in real-time, its continuous batching pipeline handling the stream without missing a beat.

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
