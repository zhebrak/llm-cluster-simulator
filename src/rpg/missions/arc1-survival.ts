/**
 * Arc 1: Survival — the ship is old, systems are failing.
 * Missions teach inference basics: memory, quantization, bandwidth, batching, TP, cost.
 * See RPG Text Style Guide in ../types.ts for briefing/hint/success narrative rules.
 *
 * DAG (1-1 is sole entry point):
 *   1-1 Wake-Up Call
 *    ├── 1-2 The Upgrade
 *    ├── 1-3 Slow Reflexes ──────────────┐
 *    ├── 1-4 Cryo-Pod Monitoring         │
 *    └── 1-5 Memory Leak ───────────┐    │
 *                                    │    │
 *         1-6 Archive Vault ← 1-2 + 1-3  │
 *         │                          │    │
 *         1-7 Fuel Budget ← 1-6     │    │
 *                                    │    │
 *         ⭐ 1-8 The Signal ← 1-6 + 1-5 + 1-3
 */

import type { RPGArc, RPGMission } from '../types.ts';

export const ARC1: RPGArc = {
  id: 'arc1-survival',
  name: 'Survival',
  subtitle: 'The ship is old. Systems are failing',
  description: 'Critical ship systems need restoring. Each mission brings another subsystem back online.',
  order: 1,
  briefing: `You are the Compute Officer aboard the GSV Meridian, a generation ship 80 years into its transit toward Ross 128b. The ship carries 200 colonists in cryogenic stasis and a skeleton crew rotating through watch cycles.

Every critical system on board — navigation, life support monitoring, long-range sensors, crew AI — runs on GPU-accelerated ML models. Your job is to keep them running on hardware that was state-of-the-art at launch — 80 years ago.

Fragments of your predecessor's maintenance logs survive in local storage — Compute Officer Martinez, the last to hold this post. The entries stop at Year 61. The ship has been running without a dedicated Compute Officer for nearly two decades.

Systems are failing. Power budgets are tight. The compute bay holds a few aging GPUs. Every configuration decision you make has consequences — an OOM crash can blind the sensors, and wasted GPU-hours drain reserves that keep the crew alive.`,
};

export const ARC1_MISSIONS: RPGMission[] = [
  // ── Mission 1-1: Wake-Up Call ──────────────────────────────────────
  {
    id: 'mission-1-1',
    arcId: 'arc1-survival',
    order: 1,
    title: 'Wake-Up Call',
    subtitle: 'The sensor array has gone dark',
    learningObjectives: [
      'Diagnose an OOM failure caused by weight precision',
      'Apply weight quantization to fit a model in limited VRAM',
      'Understand the tradeoff between precision and memory footprint',
    ],
    briefing: `You're jolted awake by alarms. The generation ship's long-range sensor array has gone offline — the inference model that processes sensor data crashed during a power fluctuation.

The diagnostic readout is blunt: the model's weight memory nearly fills the GPU on its own, leaving no room for the working memory that inference requires. Total memory demand exceeds what the card can hold.

Get the sensor model running. The hardware can't be swapped — you'll need to find a way to fit the model into the memory you have.`,
    successNarrative: `The sensor array flickers back to life. Stars resolve on the navigation display for the first time in hours.

Reducing weight precision compressed the model to fit within the GPU's memory budget. Each parameter now takes fewer bytes to store — a small trade in numerical accuracy for a large reduction in memory footprint.

Quantization like this is one of the most practical tools on a resource-constrained ship. The model runs slightly less precisely, but it runs.

*Log entry, Compute Officer Martinez, Year 42: "T4s handle INT8 weights for the full sensor suite. Anything larger is a pipe dream with this VRAM. But she holds."*`,
    primaryMode: 'inference',
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 't4',
      numGPUs: 1,
      weightPrecision: 'fp16',
      inputSeqLen: 1024,
      outputSeqLen: 512,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Model fits in GPU memory' },
    ],
    expectedChanges: [
      { field: 'numGPUs', check: 'unchanged', label: 'Did not add more GPUs' },
      { field: 'weightPrecision', check: 'changed', label: 'Changed weight precision' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change the model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change the GPU' },
    ],
    hints: [
      'The model needs more memory than the GPU has. Is there a way to reduce how much memory the model takes?',
      'The Weight Precision selector in the sidebar controls how many bytes each parameter takes. Lower precision means a smaller footprint.',
    ],
    prerequisites: [],
    skillsAwarded: ['precision'],
  },

  // ── Mission 1-2: The Upgrade ──────────────────────────────────────
  {
    id: 'mission-1-2',
    arcId: 'arc1-survival',
    order: 2,
    title: 'The Upgrade',
    subtitle: 'A signal from mission control',
    learningObjectives: [
      'Calculate minimum weight memory: params × bytes_per_param at maximum quantization',
      'Recognize when no quantization can bridge the gap between model size and GPU memory',
      'Select an appropriately-sized model for the hardware constraint',
    ],
    briefing: `The navigation system needs an upgrade. Mission control's last transmission included a 70B-parameter model — far more capable than what we've been running.

But the RTX 4090 has only 24GB of memory. The diagnostics panel shows the model overflowing memory by a wide margin, even after you try every quantization option available.

Some models are simply too large for a given GPU — no amount of compression can fit 70 billion parameters into 24GB. Find a model that actually fits the hardware.`,
    successNarrative: `Navigation charts update with fresh trajectory data. The smaller model processes waypoints efficiently within the 4090's memory budget.

Even at maximum compression — INT4, half a byte per parameter — the 70B model needs ~35GB. The 4090 has 24GB. No amount of quantization can close that gap. When a model's minimum footprint exceeds the hardware, the only move is a different model.

Good model selection starts here: check whether the model can physically fit before optimizing anything else.`,
    primaryMode: 'inference',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'rtx-4090',
      numGPUs: 1,
      inputSeqLen: 1024,
      outputSeqLen: 512,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Model fits in GPU memory' },
      { field: 'throughput.tokensPerSecond', operator: '>', value: 20, label: 'Throughput > 20 tok/s' },
    ],
    expectedChanges: [
      { field: 'numGPUs', check: 'unchanged', label: 'Did not add more GPUs' },
      { field: 'modelId', check: 'changed', label: 'Changed to a different model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change the GPU' },
    ],
    hints: [
      'Try every precision option and watch the memory breakdown. Even at the most aggressive quantization, calculate the floor — minimum bytes per parameter times 70 billion — and compare that to the GPU\'s total VRAM.',
      'The model itself is too large for this GPU at any precision. You need to switch to a smaller model entirely.',
      "What's the smallest model class that fits at maximum compression? Check the registry for models whose minimum footprint falls under 24GB.",
    ],
    prerequisites: ['mission-1-1'],
    skillsAwarded: ['hardware'],
  },

  // ── Mission 1-3: Slow Reflexes ────────────────────────────────────
  {
    id: 'mission-1-3',
    arcId: 'arc1-survival',
    order: 3,
    title: 'Slow Reflexes',
    subtitle: 'Asteroids faster than the scanner',
    learningObjectives: [
      'Identify memory bandwidth as the bottleneck for autoregressive decode',
      'Compare GPU bandwidth specs to predict relative per-token latency',
      'Choose a GPU based on bandwidth requirements, not just VRAM',
    ],
    briefing: `The sensor model is back online, but the response time is unacceptable. Asteroids appear on the display seconds after they enter scanner range — at current velocities, seconds are the entire margin between course correction and hull breach.

The model runs on a T4 GPU at reduced precision. It fits in memory fine, but each response takes an eternity — the crew can't track targets in real time.`,
    successNarrative: `Asteroid tracking snaps to real-time. Objects now appear on the display the instant they enter scanner range.

The T4 had enough memory — the bottleneck was memory bandwidth. Autoregressive decode reads the full weight matrix for every output token, so per-token speed scales directly with how fast the GPU can stream data from memory. The new card's higher bandwidth translates directly into faster token generation.

For decode-heavy workloads, memory bandwidth is the spec that matters most.

*Log entry, CO Martinez, Year 58: "Replaced the bandwidth monitoring script again. The T4 reads fast enough — for now. Margins are shrinking. When this card dies, the ship loses reflexes."*`,
    primaryMode: 'inference',
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 't4',
      numGPUs: 1,
      weightPrecision: 'int8',
      batchSize: 1,
      speculativeDecoding: false,
      inputSeqLen: 1024,
      outputSeqLen: 512,
    },
    winningCriteria: [
      { field: 'latency.tpot', operator: '<', value: 18, label: 'Per-token decode < 18 ms' },
    ],
    expectedChanges: [
      { field: 'numGPUs', check: 'unchanged', label: 'Did not add more GPUs' },
      { field: 'gpuId', check: 'changed', label: 'Upgraded to a faster GPU' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change the model' },
      { field: 'speculativeDecoding', check: 'unchanged', label: 'Did not use speculative decoding' },
    ],
    hints: [
      'Autoregressive decode reads the entire weight matrix for every token generated. The time per token is limited by how fast the GPU can read from memory — its memory bandwidth.',
      "Compare the T4's memory bandwidth to the RTX 4090's in the GPU specs — one has significantly higher bandwidth than the other.",
      'The RTX 4090 has significantly higher memory bandwidth than the T4. Since per-token decode speed scales directly with bandwidth, switching to the higher-bandwidth card should push you well below the latency target.',
    ],
    prerequisites: ['mission-1-1'],
    skillsAwarded: ['hardware'],
  },

  // ── Mission 1-4: Cryo-Pod Monitoring ──────────────────────────────
  {
    id: 'mission-1-4',
    arcId: 'arc1-survival',
    order: 4,
    title: 'Cryo-Pod Monitoring',
    subtitle: 'Vital signs slipping through the cracks',
    learningObjectives: [
      'Understand how batching amortizes weight reads across concurrent requests',
      'Increase batch size to transition from bandwidth-bound to compute-bound regime',
      'Predict throughput scaling with batch size (near-linear until compute ceiling)',
    ],
    briefing: `The cryo-pod monitoring system has been running diagnostics one pod at a time. With 200 colonists in stasis, the full sweep takes too long — vital signs could deteriorate before the system cycles back around.

The model handles individual queries fine on the RTX 4090. But processing pods sequentially means each query reads the entire weight matrix from memory, wastes the compute cores, then starts over.

Processing multiple queries at once should be more efficient. The GPU reads the weights once and applies them to all queries simultaneously.`,
    successNarrative: `The monitoring sweep completes in seconds instead of minutes. Vital sign alerts trigger immediately across all 200 pods.

At a batch size of 1, the GPU reads the full model weights for every query — compute cores sitting idle, waiting on memory. By batching multiple queries together, the same weight read serves all of them simultaneously. The memory bandwidth cost is amortized, and throughput scales nearly linearly until the compute units saturate.

Fewer sweeps, faster cycles, no pod left waiting.`,
    primaryMode: 'inference',
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 'rtx-4090',
      numGPUs: 1,
      weightPrecision: 'int8',
      batchSize: 1,
      inputSeqLen: 1024,
      outputSeqLen: 512,
    },
    winningCriteria: [
      { field: 'throughput.tokensPerSecond', operator: '>', value: 400, label: 'Throughput > 400 tok/s' },
    ],
    expectedChanges: [
      { field: 'numGPUs', check: 'unchanged', label: 'Did not add more GPUs' },
      { field: 'batchSize', check: 'increased', label: 'Increased batch size' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change the model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change the GPU' },
    ],
    hints: [
      'At batch=1, every token generation reads the full model weights from GPU memory. The compute cores are underutilized — they finish the math long before the next weight read arrives.',
      'Batching amortizes the weight read across N queries. Batch=N means the weights are read once and applied to N tokens simultaneously, approaching N× throughput.',
      'Increase the batch size until throughput plateaus. Watch it scale nearly linearly at first, then flatten as the compute units saturate.',
    ],
    prerequisites: ['mission-1-1'],
    skillsAwarded: ['batching'],
  },

  // ── Mission 1-5: Memory Leak ──────────────────────────────────────
  {
    id: 'mission-1-5',
    arcId: 'arc1-survival',
    order: 5,
    title: 'Memory Leak',
    subtitle: 'The crew AI forgets mid-sentence',
    learningObjectives: [
      'Diagnose long-context memory failures from KV cache growth and attention workspace',
      'Enable Flash Attention to eliminate O(N^2) attention memory',
      'Understand KV cache growth and apply cache quantization to reduce per-token memory',
    ],
    briefing: `The crew AI handles short exchanges fine, but extended conversations crash it. Crew members report the system freezing mid-sentence during long diagnostic discussions — the ones where context matters most.

The memory trace tells the story: as conversations grow, GPU memory climbs steadily — then, past a certain length, it spikes catastrophically. Short exchanges barely register. Long ones hit a wall. The model's weights haven't changed. Something else is growing with every token the conversation produces, and at 131K tokens it overwhelms the GPU entirely.

The model isn't too large. The conversation is.`,
    successNarrative: `The crew AI terminal stabilizes. Extended diagnostic sessions run to completion, even at maximum context length.

The fix targeted runtime memory, not weight memory. Flash Attention computes attention in tiles rather than materializing the full matrix, collapsing workspace from O(N²) to O(N). Combined with KV cache quantization, the dynamic memory that accumulates with every generated token is finally under control.

Long conversations no longer mean runaway memory growth.

*Log entry, CO Martinez, Year 55: "Crew AI crashes again on long sessions. Short conversations fine — anything past ten minutes, gone. Been meaning to look into it. Always something more urgent."*`,
    primaryMode: 'inference',
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 'rtx-4090',
      numGPUs: 1,
      weightPrecision: 'int8',
      flashAttention: false,
      kvCachePrecision: 'bf16',
      inputSeqLen: 131072,
      outputSeqLen: 512,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Model fits in GPU memory' },
    ],
    expectedChanges: [
      { field: 'numGPUs', check: 'unchanged', label: 'Did not add more GPUs' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change the model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change the GPU' },
      { field: 'inputSeqLen', check: 'unchanged', label: 'Did not change input sequence length' },
      { field: 'outputSeqLen', check: 'unchanged', label: 'Did not change output sequence length' },
    ],
    hints: [
      'Two memory consumers grow with sequence length: the attention workspace (O(N^2) without Flash Attention) and the KV cache (linear, stores key/value pairs for every past token).',
      'Flash Attention computes attention in tiles, reducing workspace memory from O(N^2) to O(N). Enable it in the sidebar.',
      'Flash Attention alone isn\'t enough at this context length — the KV cache still overflows. KV cache quantization (INT8 or FP8) halves per-token cache memory.',
    ],
    prerequisites: ['mission-1-1'],
    skillsAwarded: ['inference-optimization'],
  },

  // ── Mission 1-6: The Archive Vault ────────────────────────────────
  {
    id: 'mission-1-6',
    arcId: 'arc1-survival',
    order: 6,
    title: 'The Archive Vault',
    subtitle: 'Four dormant GPUs near the reactor',
    learningObjectives: [
      'Deploy a model too large for a single GPU using tensor parallelism',
      'Understand TP as weight matrix sharding across GPUs with AllReduce synchronization',
      'Configure TP degree based on model size and per-GPU memory',
    ],
    briefing: `Engineering found a sealed compute module near the reactor core — four A100 GPUs with 80GB each, connected by high-bandwidth interconnect. They power on, but no one knows if they still work after 80 years in vacuum.

The long-range scanner needs a major upgrade. A 70B-parameter model would give far better resolution than the 8B model currently running, but at BF16 precision it requires ~140GB — far more than any single GPU can hold.

Prove these GPUs are operational. If the model is too large for one card, maybe four cards together can carry it.`,
    successNarrative: `The long-range scanner's resolution jumps by an order of magnitude. The 70B model resolves objects that the 8B model couldn't distinguish from noise. The A100s are confirmed operational.

With tensor parallelism, the weight matrices are split across GPUs — each one stores and computes its fraction of every layer, then an AllReduce synchronizes the partial activations before the next layer begins. The 140GB model now fits across four 80GB cards.

The tradeoff is communication overhead between GPUs, but for a model this size, it's the most direct way to get it running.

*Log entry, CO Martinez, Year 61: "Found sealed compute module near reactor core. Four A100s, never powered. Okafor keeps asking about training mode — says we should be adapting the models to real sensor data instead of running decade-old weights. Told her the power draw isn't worth it. Hope I'm right."*`,
    primaryMode: 'inference',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'a100-80gb',
      numGPUs: 4,
      weightPrecision: 'bf16',
      inputSeqLen: 1024,
      outputSeqLen: 512,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Model fits in GPU memory' },
      { field: 'latency.ttft', operator: '<', value: 500, label: 'TTFT < 500ms' },
    ],
    expectedChanges: [
      { field: 'numGPUs', check: 'unchanged', label: 'Did not add more GPUs' },
      { field: 'tensorParallel', check: 'increased', label: 'Increased tensor parallelism' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change the model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change the GPU' },
      { field: 'inputSeqLen', check: 'unchanged', label: 'Did not change input sequence length' },
      { field: 'outputSeqLen', check: 'unchanged', label: 'Did not change output sequence length' },
    ],
    hints: [
      'At BF16, the 70B model needs ~140GB of VRAM. Each A100 has 80GB — the model must be split across multiple GPUs.',
      'If the model needs ~140GB and each GPU has 80GB, you need to split the model across enough GPUs that each one holds less than 80GB. Think about what division achieves that.',
      'Divide the total weight memory by different TP degrees to see how much each GPU would hold. Higher TP reduces per-GPU memory but adds communication overhead — find the degree that fits.',
    ],
    prerequisites: ['mission-1-2', 'mission-1-3'],
    skillsAwarded: ['memory-optimization', 'model-parallelism'],
  },

  // ── Mission 1-7: Fuel Budget ──────────────────────────────────────
  {
    id: 'mission-1-7',
    arcId: 'arc1-survival',
    order: 7,
    title: 'Fuel Budget',
    subtitle: 'The reactor can\'t sustain this',
    learningObjectives: [
      'Optimize compute cost by reducing GPU count through quantization',
      'Combine quantization with TP adjustment to maintain latency under cost constraints',
      'Evaluate the cost-performance tradeoff: every freed GPU reduces resource consumption',
    ],
    briefing: `The reactor feeds four A100 GPUs running the 70B scanner model — but the power draw is unsustainable. Engineering calculates the cost equivalent: every GPU-hour drains reserves that could keep life support running longer.

Four GPUs at full load — that's the single biggest power draw on the ship outside the drive itself. There has to be a way to get the same capability from less hardware.

Halve your costs. The scanner's input and output windows are fixed by the sensor protocol — you can't shorten them. Keep latency under control and leave memory headroom — a GPU running near capacity will crash if the scanner hits a burst.`,
    successNarrative: `Two A100s power down. The scanner holds steady — same model, same responsiveness, half the power draw. Engineering redirects the freed capacity to environmental systems.

Quantization and parallelism work together. At BF16, the 70B model needs 140GB and four GPUs. At INT8, it shrinks to 70GB — two GPUs with TP=2. The same capability, half the hardware.

Every freed GPU is power the ship can spend elsewhere. Cost optimization on a generation ship isn't abstract — it's life support.`,
    primaryMode: 'inference',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'a100-80gb',
      numGPUs: 4,
      tensorParallel: 4,
      weightPrecision: 'bf16',
      inputSeqLen: 1024,
      outputSeqLen: 512,
    },
    winningCriteria: [
      { field: 'memoryUtilization', operator: '<', value: 0.85, label: 'Memory utilization < 85%' },
      { field: 'latency.ttft', operator: '<', value: 600, label: 'TTFT < 600ms' },
      { field: 'numGPUs', operator: '<=', value: 2, label: 'Using 2 or fewer GPUs' },
    ],
    expectedChanges: [
      { field: 'numGPUs', check: 'decreased', label: 'Reduced GPU count' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change the model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change the GPU' },
      { field: 'pricePerGPUHour', check: 'unchanged', label: 'Did not change GPU pricing' },
      { field: 'inputSeqLen', check: 'unchanged', label: 'Did not change input sequence length' },
      { field: 'outputSeqLen', check: 'unchanged', label: 'Did not change output sequence length' },
    ],
    hints: [
      'Look at the current configuration — four GPUs running the same model. Every GPU you can free up saves that fraction of the power budget. The question is: can the model fit on fewer cards if you compress the weights?',
      'Quantization shrinks the model so fewer GPUs are needed. If you halve the bytes per parameter, you halve the total weight memory — how many 80GB cards does the compressed model need?',
      'Combine precision reduction with fewer GPUs. The math should tell you the minimum — calculate the compressed weight size and see how many 80GB cards it needs.',
    ],
    prerequisites: ['mission-1-6'],
    skillsAwarded: ['resource-efficiency', 'precision'],
  },

  // ── Mission 1-8: The Signal (Pivot) ───────────────────────────────
  {
    id: 'mission-1-8',
    arcId: 'arc1-survival',
    order: 8,
    title: 'The Signal',
    subtitle: 'Every display on the bridge lights up',
    type: 'pivot',
    learningObjectives: [],
    briefing: `03:00 ship time. The bridge is dark except for instrument glow and the night watch dozing at their station. Then every display lights up at once.

A structured, repeating signal from the direction of Ross 128b. Not cosmic noise. Not instrument artifacts. A pattern with structure. With purpose.

The sensor array confirms it independently: the signal is real, originating from the star system the Meridian has been falling toward for 80 years. The crew AI flags Dr. Chen for an emergency wake cycle.`,
    successNarrative: `Dr. Chen arrives on the bridge, still shaking off cryo-fog. She stares at the signal analysis for a long time.

"This isn't natural," she says quietly. "The repetition structure, the frequency modulation — this is encoded information. Something is broadcasting from Ross 128b."

The bridge is silent. The void isn't empty after all.`,
    primaryMode: 'inference',
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 't4',
    },
    winningCriteria: [],
    hints: [],
    prerequisites: ['mission-1-3', 'mission-1-5', 'mission-1-6'],
    skillsAwarded: [],
  },
];
