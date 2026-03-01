/**
 * Arc 1: Survival — the ship is old, systems are failing.
 * Missions teach inference basics: memory, quantization, bandwidth, batching, TP, cost.
 *
 * DAG (1-1 is sole entry point):
 *   1-1 Wake-Up Call
 *    ├── 1-2 The Wrong Model
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
  subtitle: 'The ship is old. Systems are failing.',
  description: 'Critical ship systems need restoring. Each mission brings another subsystem back online.',
  order: 1,
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

The diagnostic readout is blunt: the model was loaded at full precision, and the GPU's memory is completely exhausted. The card simply doesn't have enough VRAM to hold all the weights.

Get the sensor model running. The hardware can't be swapped — you'll need to find a way to fit the model into the memory you have.`,
    successNarrative: `The sensor array flickers back to life. Stars resolve on the navigation display for the first time in hours.

By reducing the weight precision, you compressed the model to fit within the GPU's memory budget. Each parameter now takes fewer bytes to store — a small trade in numerical accuracy for a dramatic reduction in memory footprint.

Quantization like this is one of the most practical tools on a resource-constrained ship. The model runs slightly less precisely, but it runs.`,
    primaryMode: 'inference',
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 't4',
      numGPUs: 1,
      weightPrecision: 'bf16',
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Model fits in GPU memory' },
    ],
    expectedChanges: [
      { field: 'weightPrecision', check: 'changed', label: 'Changed weight precision' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change the model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change the GPU' },
    ],
    hints: [
      'The model weights consume `parameters × bytes_per_parameter` of GPU memory. At full precision, each parameter takes 2 bytes. How much memory is that for an 8B parameter model?',
      'Look at the Weight Precision selector in the sidebar. Lower-precision formats store each parameter in fewer bytes — some use just 1 byte per parameter.',
      'Try INT8 — it halves the memory footprint compared to BF16 while maintaining reasonable accuracy for inference.',
    ],
    prerequisites: [],
    skillsAwarded: ['quantization'],
  },

  // ── Mission 1-2: The Wrong Model ──────────────────────────────────
  {
    id: 'mission-1-2',
    arcId: 'arc1-survival',
    order: 2,
    title: 'The Wrong Model',
    subtitle: 'The upgrade that doesn\'t fit',
    learningObjectives: [
      'Calculate minimum weight memory: params × bytes_per_param at maximum quantization',
      'Recognize when no quantization can bridge the gap between model size and GPU memory',
      'Select an appropriately-sized model for the hardware constraint',
    ],
    briefing: `The navigation system needs an upgrade. Mission control's last transmission included a 70B-parameter model — far more capable than what we've been running.

But the RTX 4090 has only 24GB of VRAM. The diagnostics panel shows the model overflowing memory by a wide margin, even after you try every quantization option available.

Some models are simply too large for a given GPU — no amount of compression can fit 70 billion parameters into 24GB. Find a model that actually fits the hardware.`,
    successNarrative: `Navigation charts update with fresh trajectory data. The smaller model processes waypoints efficiently within the 4090's memory budget.

Even at maximum compression — INT4, half a byte per parameter — the 70B model needs ~35GB. The 4090 has 24GB. No amount of quantization can close that gap. When a model's minimum footprint exceeds the hardware, the only move is a different model.

Good model selection starts here: check whether the model can physically fit before optimizing anything else.`,
    primaryMode: 'inference',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'rtx-4090',
      numGPUs: 1,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Model fits in GPU memory' },
      { field: 'throughput.tokensPerSecond', operator: '>', value: 20, label: 'Throughput > 20 tok/s' },
    ],
    expectedChanges: [
      { field: 'modelId', check: 'changed', label: 'Changed to a different model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change the GPU' },
    ],
    hints: [
      'Calculate the minimum weight memory: 70B × 0.5 bytes/param (INT4) = 35GB. The RTX 4090 has 24GB. No quantization level can close that gap.',
      'The model itself is too large for this GPU at any precision. You need to switch to a smaller model entirely.',
      "Try selecting a model from the registry that's small enough to fit — an 8B model at INT4 uses only ~4GB of VRAM.",
    ],
    prerequisites: ['mission-1-1'],
    skillsAwarded: ['model-selection'],
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
      'Compare GPU bandwidth specs to predict relative throughput',
      'Choose a GPU based on bandwidth requirements, not just VRAM',
    ],
    briefing: `The sensor model is back online, but the response time is unacceptable. Asteroids appear on the display seconds after they enter scanner range — at current velocities, that's too slow for collision avoidance.

The model runs on a T4 GPU with INT8 precision. It fits in memory fine, but the decode step — generating each output token — is painfully slow. Each token requires reading the entire model's weights from GPU memory.

The bottleneck isn't compute or memory capacity. It's how fast the GPU can read data from its memory. Find a GPU with higher memory bandwidth.`,
    successNarrative: `Asteroid tracking snaps to real-time. Objects now appear on the display the instant they enter scanner range.

The T4 had enough memory — the bottleneck was memory bandwidth. Autoregressive decode reads the full weight matrix for every output token, so throughput scales directly with how fast the GPU can stream data from memory. The new card's higher bandwidth translates straight into more tokens per second.

For decode-heavy workloads, memory bandwidth is the spec that matters most.`,
    primaryMode: 'inference',
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 't4',
      numGPUs: 1,
      weightPrecision: 'int8',
    },
    winningCriteria: [
      { field: 'throughput.tokensPerSecond', operator: '>', value: 60, label: 'Throughput > 60 tok/s' },
    ],
    expectedChanges: [
      { field: 'gpuId', check: 'changed', label: 'Changed to a different GPU' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change the model' },
    ],
    hints: [
      'Autoregressive decode reads the entire weight matrix for every token generated. Throughput is limited by how fast the GPU can read from memory — its memory bandwidth.',
      "Compare the T4's memory bandwidth to the RTX 4090's in the GPU specs — one has significantly higher bandwidth than the other.",
      'Switch to the RTX 4090. Its significantly higher memory bandwidth should push throughput well above the target.',
    ],
    prerequisites: ['mission-1-1'],
    skillsAwarded: ['gpu-bandwidth'],
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

At a batch size of 1, the GPU reads the full model weights for every single query — compute cores sitting idle, waiting on memory. By batching multiple queries together, the same weight read serves all of them simultaneously. The memory bandwidth cost is amortized, and throughput scales nearly linearly until the compute units saturate.

Fewer sweeps, faster cycles, no pod left waiting.`,
    primaryMode: 'inference',
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 'rtx-4090',
      numGPUs: 1,
      weightPrecision: 'int8',
      batchSize: 1,
    },
    winningCriteria: [
      { field: 'throughput.tokensPerSecond', operator: '>', value: 400, label: 'Throughput > 400 tok/s' },
    ],
    expectedChanges: [
      { field: 'batchSize', check: 'increased', label: 'Increased batch size' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change the model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change the GPU' },
    ],
    hints: [
      'At batch=1, every token generation reads the full model weights from GPU memory. The compute cores are underutilized — they finish the math long before the next weight read arrives.',
      'Batching amortizes the weight read across N queries. Batch=N means the weights are read once and applied to N tokens simultaneously, approaching N× throughput.',
      'Try increasing the batch size to 8, 16, or higher. Watch throughput scale with batch size until the compute units saturate.',
    ],
    prerequisites: ['mission-1-1'],
    skillsAwarded: ['batch-size'],
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
      'Apply KV cache quantization or paged attention to reduce dynamic memory',
    ],
    briefing: `The crew AI handles short exchanges fine, but extended conversations crash it. Crew members report the system freezing mid-sentence during long diagnostic discussions — the ones where context matters most.

The diagnostic log reveals a memory spike during long-context inference. Two culprits: the attention mechanism's workspace grows with the square of the sequence length, and the KV cache — which stores previous token representations — accumulates linearly with every token generated.

At 32K tokens, the combined memory demand overflows the GPU. The model weights are fine; it's the runtime memory that's the problem.`,
    successNarrative: `Dr. Chen's crew AI terminal stabilizes. Extended diagnostic sessions run to completion, even at maximum context length.

The fix targeted runtime memory, not weight memory. Flash Attention computes attention in tiles rather than materializing the full matrix, collapsing workspace from O(N²) to O(N). Combined with KV cache quantization, the dynamic memory that accumulates with every generated token is finally under control.

Long conversations no longer mean a death spiral of growing memory.`,
    primaryMode: 'inference',
    setup: {
      modelId: 'llama3.1-8b',
      gpuId: 'rtx-4090',
      numGPUs: 1,
      weightPrecision: 'int8',
      flashAttention: false,
      pagedAttention: false,
      kvCachePrecision: 'bf16',
      inputSeqLen: 32768,
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Model fits in GPU memory' },
    ],
    expectedChanges: [
      { field: 'modelId', check: 'unchanged', label: 'Did not change the model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change the GPU' },
      { field: 'weightPrecision', check: 'unchanged', label: 'Did not change weight precision' },
      { field: 'inputSeqLen', check: 'unchanged', label: 'Did not change input sequence length' },
    ],
    hints: [
      'Two memory consumers grow with sequence length: the attention workspace (O(N^2) without Flash Attention) and the KV cache (linear, stores key/value pairs for every past token).',
      'Flash Attention computes attention in tiles, reducing workspace memory from O(N^2) to O(N). Enable it in the sidebar.',
      'If Flash Attention alone isn\'t enough, KV cache quantization (INT8) halves per-token cache memory, and paged attention eliminates internal fragmentation.',
    ],
    prerequisites: ['mission-1-1'],
    skillsAwarded: ['kv-cache'],
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

Prove these GPUs are operational. The model's weight matrices can be split across them — each GPU holds a fraction and synchronizes with the others after each layer.`,
    successNarrative: `The long-range scanner's resolution jumps by an order of magnitude. The 70B model resolves objects that the 8B model couldn't distinguish from noise. The A100s are confirmed operational.

With tensor parallelism, the weight matrices are split across GPUs — each one stores and computes its fraction of every layer, then an AllReduce synchronization merges the partial results. The 140GB model now lives comfortably across four 80GB cards.

The tradeoff is communication overhead between GPUs, but for a model this size, there's no other way to get it running.`,
    primaryMode: 'inference',
    setup: {
      modelId: 'llama3.3-70b',
      gpuId: 'a100-80gb',
      numGPUs: 4,
      weightPrecision: 'bf16',
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Model fits in GPU memory' },
      { field: 'latency.ttft', operator: '<', value: 500, label: 'TTFT < 500ms' },
    ],
    expectedChanges: [
      { field: 'tensorParallel', check: 'increased', label: 'Increased tensor parallelism' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change the model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change the GPU' },
    ],
    hints: [
      'At BF16, the 70B model needs ~140GB of VRAM. Each A100 has 80GB — the model must be split across multiple GPUs.',
      'Tensor parallelism shards weight matrices across GPUs. Each GPU holds 1/TP of the model and they synchronize after each layer via AllReduce.',
      'Try TP=2 (70GB/GPU) or TP=4 (35GB/GPU). Higher TP reduces per-GPU memory but adds communication overhead.',
    ],
    prerequisites: ['mission-1-2', 'mission-1-3'],
    skillsAwarded: ['tensor-parallelism'],
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
      'Evaluate the cost-performance tradeoff: every freed GPU saves $/hr',
    ],
    briefing: `The reactor feeds four A100 GPUs running the 70B scanner model — but the power draw is unsustainable. Engineering calculates the cost equivalent: every GPU-hour drains reserves that could keep life support running longer.

The model works at TP=4, but you're using twice the hardware you need. Quantization can shrink the model's memory footprint so it fits on fewer GPUs. Fewer GPUs means less power, less heat, and more reserves for the journey ahead.

Halve your costs. Keep the scanner model running, keep latency under control, but free up GPUs for other critical systems.`,
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
    },
    winningCriteria: [
      { field: 'success', operator: '==', value: true, label: 'Model fits in GPU memory' },
      { field: 'latency.ttft', operator: '<', value: 500, label: 'TTFT < 500ms' },
      { field: 'numGPUs', operator: '<=', value: 2, label: 'Using 2 or fewer GPUs' },
    ],
    expectedChanges: [
      { field: 'numGPUs', check: 'decreased', label: 'Reduced GPU count' },
      { field: 'modelId', check: 'unchanged', label: 'Did not change the model' },
      { field: 'gpuId', check: 'unchanged', label: 'Did not change the GPU' },
      { field: 'pricePerGPUHour', check: 'unchanged', label: 'Did not change GPU pricing' },
    ],
    hints: [
      'Each A100 draws significant power. Four GPUs cost far more per hour than two — halving the count directly halves the energy drain.',
      'Quantization shrinks the model so fewer GPUs are needed. INT8 halves weight memory: 70B × 1 byte = 70GB, which fits on two A100s with TP=2.',
      'Try setting weight precision to INT8, reducing numGPUs to 2, and setting TP=2. At INT4 (35GB), the model could even fit on a single GPU.',
    ],
    prerequisites: ['mission-1-6'],
    skillsAwarded: ['cost-optimization'],
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
    briefing: `0300 ship time. The bridge is empty except for the night watch when the long-range scanner — the 70B model you upgraded in the Archive Vault — flags an anomaly.

A structured, repeating signal from the direction of Kepler-442b. Not cosmic noise. Not instrument artifacts. A pattern with information content.

The sensor array you restored in Mission 1 confirms: the signal is real, originating from the star system you've been traveling toward for 80 years. The crew AI flags Dr. Chen for an emergency wake cycle.`,
    successNarrative: `Dr. Chen arrives on the bridge, still shaking off cryo-fog. She stares at the signal analysis for a long time.

"This isn't natural," she says. "The repetition structure, the frequency modulation — this is encoded information. Someone — or something — is broadcasting from Kepler-442b."

The ship's mission has fundamentally changed. You're no longer just surviving the journey. Whatever awaits at Kepler-442b is actively communicating.

Every system you've restored — sensors, navigation, crew AI, the scanner — serves a new purpose. The real adventure is just beginning.`,
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
