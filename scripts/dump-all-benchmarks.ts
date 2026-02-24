#!/usr/bin/env npx tsx
/**
 * Dump all benchmark configs through the simulator and output markdown tables
 * matching the exact section structure in README.md and docs/BENCHMARKS.md.
 *
 * Sections:
 *   - README.md table
 *   - §2.1 Primary — Hopper (H100, H800)
 *   - §2.2 Primary — Ampere (A100)
 *   - §2.3 Secondary
 *   - §3  Throughput Cross-Check
 *
 * Usage:
 *   npx tsx scripts/dump-all-benchmarks.ts
 */

import {
  type SimulationConfig,
  type SimulationMetrics,
  getSimulationMetrics,
} from '../src/core/simulation/engine.ts';
import { getValidatedSimulationMetrics } from '../tests/helpers/validated-metrics.ts';
import { PUBLISHED, toSimConfig } from '../src/data/published-training-configs.ts';
import { benchmarkConfig } from '../tests/helpers/benchmark-config.ts';

// ---------------------------------------------------------------------------
// Benchmark entry definition
// ---------------------------------------------------------------------------
interface BenchmarkEntry {
  label: string;          // e.g. "LLaMA 3.1 405B 8K"
  gpuCount: number;
  gpuLabel: string;       // e.g. "16384 H100", "2048 H800"
  strategy: string;       // e.g. "3D (TP=8 PP=16 DP=128)"
  configDetails: string;  // e.g. "Interleaved v=4, GBS=2048 MBS=1 seq=8192"
  ac: string;             // "Off" | "Full" | "Selective"
  published: string;      // "~40%" or "--"
  delta?: string;         // computed from sim vs published
  source: string;         // "[Meta paper](...) §3.3.2, Table 4"
  notes?: string;         // for secondary/sim-only table Notes column
  config: SimulationConfig;
  useRaw: boolean;
  // README inclusion
  inReadme?: boolean;
  readmeLabel?: string;   // different label for README table
  readmeStrategy?: string;  // shorter strategy label for README
  readmeSource?: string;    // shorter source for README
  // Throughput cross-check
  throughputCheck?: {
    publishedTFLOPS?: string;  // e.g. "138"
    publishedTokPerSGPU?: string; // e.g. "3700 tok/s/GPU"
    notes: string;
  };
}

// ---------------------------------------------------------------------------
// §2.1 Primary — Hopper (H100, H800)
// ---------------------------------------------------------------------------
const PRIMARY_HOPPER: BenchmarkEntry[] = [
  {
    label: 'LLaMA 3.1 405B 8K',
    gpuCount: 16384, gpuLabel: '16384 H100',
    strategy: '3D (TP=8 PP=16 DP=128)',
    configDetails: 'Interleaved v=4, GBS=2048 MBS=1 seq=8192',
    ac: 'Off',
    published: '~40%',
    source: '[Meta paper](https://arxiv.org/abs/2407.21783) §3.3.2, Table 4',
    config: toSimConfig(PUBLISHED.llama3_405b_8k),
    useRaw: true,
    inReadme: true,
    readmeLabel: 'LLaMA 3.1 405B',
    readmeStrategy: '3D (TP8 PP16)',
    readmeSource: '[Meta](https://arxiv.org/abs/2407.21783) Table 4',
    throughputCheck: {
      publishedTFLOPS: '~400',
      notes: '+2%, AC off',
    },
  },
  {
    label: 'LLaMA 3.1 405B 131K',
    gpuCount: 16384, gpuLabel: '16384 H100',
    strategy: '3D (TP=8 PP=16 CP=16 DP=8)',
    configDetails: 'Interleaved v=4, all-gather CP, GBS=2048 MBS=1 seq=131072',
    ac: 'Full',
    published: '38%\\*',
    source: '[Meta paper](https://arxiv.org/abs/2407.21783) §3.3.2, Table 4',
    config: toSimConfig(PUBLISHED.llama3_405b_131k),
    useRaw: true,
    inReadme: true,
    readmeStrategy: '3D + CP16',
    readmeSource: '[Meta](https://arxiv.org/abs/2407.21783) Table 4',
  },
  {
    label: 'DeepSeek V3 671B FP8†',
    gpuCount: 2048, gpuLabel: '2048 H800',
    strategy: '3D (TP=4 PP=8 DP=64 EP=32)',
    configDetails: 'DualPipeV, GBS=8192 MBS=2 seq=4096',
    ac: 'Full',
    published: '43.7%',
    source: '[DeepSeek V3 paper](https://arxiv.org/abs/2412.19437) §3.1; MFU from [ISCA 2025](https://arxiv.org/abs/2505.09343) Table 4',
    config: toSimConfig(PUBLISHED.deepseek_v3_fp8_h800),
    useRaw: false,
    inReadme: true,
    readmeLabel: 'DeepSeek V3 671B FP8',
    readmeStrategy: '3D + EP32',
    readmeSource: '[DeepSeek](https://arxiv.org/abs/2412.19437) §3.1',
  },
  {
    label: 'Nemotron-4 340B',
    gpuCount: 6144, gpuLabel: '6144 H100',
    strategy: '3D (TP=8 PP=12 DP=64)',
    configDetails: 'VP=8 interleaved, GBS=768 MBS=1 seq=4096',
    ac: 'Selective',
    published: '41-42%',
    source: '[Megatron-Core blog](https://developer.nvidia.com/blog/train-generative-ai-models-more-efficiently-with-new-nvidia-megatron-core-functionalities/) Table 2',
    config: toSimConfig(PUBLISHED.nemotron_4_340b),
    useRaw: false,
    inReadme: true,
    readmeStrategy: '3D (TP8 PP12)',
    readmeSource: '[NVIDIA](https://developer.nvidia.com/blog/train-generative-ai-models-more-efficiently-with-new-nvidia-megatron-core-functionalities/) Table 2',
    throughputCheck: {
      notes: 'Different GBS scaling',
    },
  },
  {
    label: 'OLMo 2 32B',
    gpuCount: 1280, gpuLabel: '1280 H100',
    strategy: 'FSDP (DP=1280)',
    configDetails: 'GBS=2048 MBS=1 seq=4096',
    ac: 'Selective',
    published: '~38%',
    source: '[AI2 OLMo 2 blog](https://allenai.org/blog/olmo2-32B)',
    config: toSimConfig(PUBLISHED.olmo2_32b),
    useRaw: false,
    inReadme: true,
    readmeStrategy: 'FSDP (DP=1280)',
    readmeSource: '[AI2](https://allenai.org/blog/olmo2-32B)',
  },
  {
    label: 'OLMo 3 32B',
    gpuCount: 1024, gpuLabel: '1024 H100',
    strategy: 'FSDP (DP=1024)',
    configDetails: 'GBS=1024 MBS=1 seq=8192',
    ac: 'Selective',
    published: '~41%',
    source: '[OLMo 3 paper](https://arxiv.org/abs/2512.13961)',
    config: toSimConfig(PUBLISHED.olmo3_32b),
    useRaw: false,
    inReadme: true,
    readmeStrategy: 'FSDP (DP=1024)',
    readmeSource: '[OLMo 3](https://arxiv.org/abs/2512.13961)',
  },
  {
    label: 'Nemotron-4 15B DGXC',
    gpuCount: 64, gpuLabel: '64 H100',
    strategy: 'FSDP-TP (TP=2 DP=32)',
    configDetails: 'GBS=256 MBS=4 seq=4096',
    ac: 'Selective',
    published: '~56%',
    source: '[NVIDIA DGXC](https://github.com/NVIDIA/dgxc-benchmarking)',
    config: toSimConfig(PUBLISHED.nemotron_4_15b_dgxc),
    useRaw: false,
  },
  {
    label: 'Mixtral 8x22B MoE',
    gpuCount: 128, gpuLabel: '128 H100',
    strategy: '3D (TP=2 PP=8 DP=8 EP=4)',
    configDetails: 'GBS=256 MBS=1 seq=4096',
    ac: 'Full',
    published: '46.3%',
    source: '[MoE Parallel Folding](https://arxiv.org/abs/2504.14960) (2025)',
    config: toSimConfig(PUBLISHED.mixtral_8x22b_ep4),
    useRaw: false,
  },
];

// ---------------------------------------------------------------------------
// §2.2 Primary — Ampere (A100)
// ---------------------------------------------------------------------------
const PRIMARY_AMPERE: BenchmarkEntry[] = [
  {
    label: 'GPT-3 175B',
    gpuCount: 1024, gpuLabel: '1024 A100',
    strategy: '3D (TP=8 PP=8 DP=16)',
    configDetails: 'Interleaved v=2, GBS=1536 MBS=1 seq=2048',
    ac: 'Full',
    published: '44.2%',
    source: '[Narayanan et al. 2021](https://arxiv.org/abs/2104.04473) §5.1',
    config: toSimConfig(PUBLISHED.gpt3_175b),
    useRaw: false,
    inReadme: true,
    readmeStrategy: '3D (TP8 PP8)',
    readmeSource: '[Megatron-LM](https://arxiv.org/abs/2104.04473) §5.1',
    throughputCheck: {
      publishedTFLOPS: '138',
      notes: '-2%, Megatron-LM',
    },
  },
  {
    label: 'IBM LLaMA 2 7B',
    gpuCount: 128, gpuLabel: '128 A100',
    strategy: 'FSDP (DP=128)',
    configDetails: 'GBS=256 MBS=2 seq=4096',
    ac: 'Off',
    published: '57%',
    source: '[IBM blog 2023](https://research.ibm.com/blog/pytorch-fsdp)',
    config: toSimConfig(PUBLISHED.ibm_llama2_7b),
    useRaw: false,
    throughputCheck: {
      publishedTokPerSGPU: '3700 tok/s/GPU',
      notes: '+19% overshoot',
    },
  },
  {
    label: 'BLOOM 176B',
    gpuCount: 384, gpuLabel: '384 A100',
    strategy: 'ZeRO-1-TP-PP (TP=4 PP=12 DP=8)',
    configDetails: 'GBS=2048 MBS=2 seq=2048',
    ac: 'Full',
    published: '~48%',
    source: '[BigScience 2022](https://arxiv.org/abs/2211.05100)',
    config: toSimConfig(PUBLISHED.bloom_176b),
    useRaw: true,
    throughputCheck: {
      publishedTFLOPS: '150',
      notes: '8PD convention (includes AC recompute)',
    },
  },
];

// ---------------------------------------------------------------------------
// §2.3 Secondary
// ---------------------------------------------------------------------------
const SECONDARY: BenchmarkEntry[] = [
  {
    label: 'MT-NLG 530B',
    gpuCount: 2240, gpuLabel: '2240 A100',
    strategy: 'ZeRO-1-TP-PP (TP=8 PP=35 DP=8)',
    configDetails: 'GBS=1920 MBS=1 seq=2048',
    ac: 'Full',
    published: '~40%',
    source: '[Smith et al. 2022](https://arxiv.org/abs/2201.11990)',
    notes: '8PD convention; largest model anchor',
    config: toSimConfig(PUBLISHED.mt_nlg_530b),
    useRaw: false,
    throughputCheck: {
      publishedTFLOPS: '126',
      notes: '8PD convention (includes AC recompute)',
    },
  },
  {
    label: 'DBRX 132B MoE',
    gpuCount: 16, gpuLabel: '16 H100',
    strategy: 'ZeRO-1-TP (TP=4)',
    configDetails: 'GBS=32 MBS=1 seq=2048',
    ac: 'Full',
    published: '~37.5%',
    source: '',
    notes: '‡ Derived from >50% HFU × 6/8',
    config: benchmarkConfig(
      'h100-sxm', 8, 2, 'dbrx',
      'zero1-tp', 32, 1, 2048,
      { tp: 4 },
    ),
    useRaw: true,
  },
  {
    label: 'Mosaic LLaMA 70B',
    gpuCount: 512, gpuLabel: '512 H100',
    strategy: 'FSDP-TP (TP=8 DP=64)',
    configDetails: 'GBS=1024 MBS=2 seq=2048',
    ac: 'Full',
    published: '41.25%',
    source: '',
    notes: '[llm-foundry](https://github.com/mosaicml/llm-foundry)',
    config: toSimConfig(PUBLISHED.mosaic_70b),
    useRaw: false,
  },
  {
    label: 'NeMo 405B CP=2',
    gpuCount: 1024, gpuLabel: '1024 H100',
    strategy: '3D (TP=8 PP=8 CP=2 DP=8)',
    configDetails: 'Interleaved v=2, GBS=512 MBS=1 seq=8192',
    ac: 'Selective',
    published: '53.6%',
    source: '',
    notes: '§ Implied from 763 TFLOPS',
    config: toSimConfig(PUBLISHED.nemo_405b_cp2),
    useRaw: true,
  },
  {
    label: 'LLaMA 2 70B',
    gpuCount: 2048, gpuLabel: '2048 A100',
    strategy: 'FSDP-TP (TP=8 DP=256)',
    configDetails: 'GBS=512 MBS=1 seq=4096',
    ac: 'Full',
    published: '~35-40%',
    source: '',
    notes: 'Estimated range',
    config: benchmarkConfig(
      'a100-80gb', 8, 256, 'llama2-70b',
      'fsdp-tp', 512, 1, 4096,
      { tp: 8 },
    ),
    useRaw: false,
  },
  {
    label: 'Qwen2 57B-A14B MoE',
    gpuCount: 64, gpuLabel: '64 H100',
    strategy: '3D (TP=2 PP=4 DP=8 EP=4)',
    configDetails: 'GBS=256 MBS=1 seq=4096',
    ac: 'Full',
    published: '35.3%',
    source: '',
    notes: 'Published MFU reflects early Megatron-Core EP baseline; [MoE Parallel Folding](https://arxiv.org/abs/2504.14960)',
    config: toSimConfig(PUBLISHED.qwen2_57b_a14b_ep4),
    useRaw: false,
  },
  {
    label: 'MT-NLG 530B (350n)',
    gpuCount: 2800, gpuLabel: '2800 A100',
    strategy: 'ZeRO-1-TP-PP (TP=8 PP=35 DP=10)',
    configDetails: 'GBS=1920 MBS=1 seq=2048',
    ac: 'Full',
    published: '~39%',
    source: '',
    notes: 'DP scaling point',
    config: toSimConfig(PUBLISHED.mt_nlg_530b_350),
    useRaw: false,
  },
  {
    label: 'MT-NLG 530B (420n)',
    gpuCount: 3360, gpuLabel: '3360 A100',
    strategy: 'ZeRO-1-TP-PP (TP=8 PP=35 DP=12)',
    configDetails: 'GBS=1920 MBS=1 seq=2048',
    ac: 'Full',
    published: '~36%',
    source: '',
    notes: 'DP scaling point',
    config: toSimConfig(PUBLISHED.mt_nlg_530b_420),
    useRaw: false,
  },
];

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------
interface BenchmarkResult {
  entry: BenchmarkEntry;
  metrics: SimulationMetrics;
  error?: string;
}

function runBenchmark(entry: BenchmarkEntry): BenchmarkResult {
  try {
    const metrics = entry.useRaw
      ? getSimulationMetrics(entry.config)
      : getValidatedSimulationMetrics(entry.config);
    return { entry, metrics };
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error(`ERROR running ${entry.label}: ${msg}`);
    return {
      entry,
      metrics: {} as SimulationMetrics,
      error: msg,
    };
  }
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/** Format MFU as percentage string, e.g. "39.8%" */
function pct(v: number): string {
  return (v * 100).toFixed(1) + '%';
}

/** Compute delta string between sim MFU and published string (percentage) */
function computeDelta(simMfu: number, published: string): string {
  const m = published.match(/([\d.]+)%/);
  if (!m) return '--';
  const pubVal = parseFloat(m[1]);
  const delta = (simMfu * 100) - pubVal;
  return (delta >= 0 ? '+' : '') + delta.toFixed(1) + '%';
}

/** Compute delta in pp (percentage points) for secondary table */
function computeDeltaPp(simMfu: number, published: string): string {
  const m = published.match(/([\d.]+)%/);
  if (!m) return '--';
  const pubVal = parseFloat(m[1]);
  const delta = (simMfu * 100) - pubVal;
  return (delta >= 0 ? '+' : '') + delta.toFixed(1) + 'pp';
}

/** Get the MFU string for display — uses modelFlopsMfu for * entries */
function getSimMfuStr(entry: BenchmarkEntry, metrics: SimulationMetrics): string {
  if (entry.published.includes('*')) {
    return pct(metrics.modelFlopsMfu ?? metrics.mfu) + '\\*';
  }
  return pct(metrics.mfu);
}

/** Get the raw MFU value for delta computation */
function getSimMfuVal(entry: BenchmarkEntry, metrics: SimulationMetrics): number {
  if (entry.published.includes('*')) {
    return metrics.modelFlopsMfu ?? metrics.mfu;
  }
  return metrics.mfu;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

// Run all benchmarks
const allEntries = [...PRIMARY_HOPPER, ...PRIMARY_AMPERE, ...SECONDARY];
const results: BenchmarkResult[] = allEntries.map(runBenchmark);

const errors = results.filter(r => r.error);
if (errors.length > 0) {
  console.error(`WARNING: ${errors.length} benchmark(s) failed!`);
  for (const e of errors) console.error(`  ${e.entry.label}: ${e.error}`);
  process.exit(1);
}

// Build lookup by label for throughput cross-check
const resultsByLabel = new Map<string, BenchmarkResult>();
for (const r of results) resultsByLabel.set(r.entry.label, r);

// Slice indices for each section
const hopperEnd = PRIMARY_HOPPER.length;
const ampereEnd = hopperEnd + PRIMARY_AMPERE.length;
// ===================================================================
// Section 1: README.md table
// ===================================================================
console.log('## README.md — Benchmarks Table\n');
console.log('| Model | GPUs | Strategy | Sim MFU | Published MFU | Source |');
console.log('|-------|------|----------|---------|---------------|--------|');

for (const r of results) {
  if (!r.entry.inReadme) continue;
  const e = r.entry;
  const simMfu = getSimMfuStr(e, r.metrics);
  const label = e.readmeLabel ?? e.label;
  const gpus = e.gpuLabel.replace(' ', '× ');
  const strategy = e.readmeStrategy ?? e.strategy;
  const pub = e.published;
  const source = e.readmeSource ?? e.source;
  console.log(`| ${label} | ${gpus} | ${strategy} | ${simMfu} | ${pub} | ${source} |`);
}

console.log();

// ===================================================================
// Section 2.1: Primary — Hopper (H100, H800)
// ===================================================================
console.log('## BENCHMARKS.md §2.1 — Primary — Hopper (H100, H800)\n');
console.log('| Model | GPUs | Strategy | Config Details | AC | Sim MFU | Published | Delta | Source |');
console.log('|-------|------|----------|----------------|----|---------|-----------|-------|--------|');

for (const r of results.slice(0, hopperEnd)) {
  const e = r.entry;
  const simMfu = getSimMfuStr(e, r.metrics);
  const delta = e.published === '--'
    ? '--'
    : computeDelta(getSimMfuVal(e, r.metrics), e.published);
  console.log(`| ${e.label} | ${e.gpuLabel} | ${e.strategy} | ${e.configDetails} | ${e.ac} | ${simMfu} | ${e.published} | ${delta} | ${e.source} |`);
}

console.log();

// ===================================================================
// Section 2.2: Primary — Ampere (A100)
// ===================================================================
console.log('## BENCHMARKS.md §2.2 — Primary — Ampere (A100)\n');
console.log('| Model | GPUs | Strategy | Config Details | AC | Sim MFU | Published | Delta | Source |');
console.log('|-------|------|----------|----------------|----|---------|-----------|-------|--------|');

for (const r of results.slice(hopperEnd, ampereEnd)) {
  const e = r.entry;
  const simMfu = getSimMfuStr(e, r.metrics);
  const delta = e.published === '--'
    ? '--'
    : computeDelta(getSimMfuVal(e, r.metrics), e.published);
  console.log(`| ${e.label} | ${e.gpuLabel} | ${e.strategy} | ${e.configDetails} | ${e.ac} | ${simMfu} | ${e.published} | ${delta} | ${e.source} |`);
}

console.log();

// ===================================================================
// Section 2.3: Secondary
// ===================================================================
console.log('## BENCHMARKS.md §2.3 — Secondary\n');
console.log('| Model | GPUs | Strategy | Config Details | AC | Sim | Published | Delta | Notes |');
console.log('|-------|------|----------|----------------|----|-----|-----------|-------|-------|');

for (const r of results.slice(ampereEnd)) {
  const e = r.entry;
  const simMfu = pct(r.metrics.mfu);
  const delta = computeDeltaPp(r.metrics.mfu, e.published);
  const notes = e.notes ?? '';
  console.log(`| ${e.label} | ${e.gpuLabel} | ${e.strategy} | ${e.configDetails} | ${e.ac} | ${simMfu} | ${e.published} | ${delta} | ${notes} |`);
}

console.log();

// ===================================================================
// Section 3: Throughput Cross-Check
// ===================================================================
console.log('## BENCHMARKS.md §3 — Throughput Cross-Check\n');
console.log('| Model | GPUs | Sim TFLOPS/GPU | Published | Notes |');
console.log('|-------|------|----------------|-----------|-------|');

// Specific throughput entries, in doc order
const throughputOrder = [
  'GPT-3 175B',
  'LLaMA 3.1 405B 8K',
  'LLaMA 3 8B',
  'IBM LLaMA 2 7B',
  'Nemotron-4 340B',
  'BLOOM 176B',
  'MT-NLG 530B',
];

for (const label of throughputOrder) {
  const r = resultsByLabel.get(label);
  if (!r || !r.entry.throughputCheck) continue;
  const e = r.entry;
  const tc = e.throughputCheck;
  const m = r.metrics;

  let simCol: string;
  if (tc.publishedTokPerSGPU) {
    // IBM entry: report tok/s/GPU
    const tokPerGPU = Math.round(m.tokensPerSecond / e.gpuCount);
    simCol = `${tokPerGPU} tok/s/GPU`;
  } else if (label === 'Nemotron-4 340B') {
    // Nemotron: report step time
    simCol = `${m.stepTimeMs.toFixed(1)}ms/step`;
  } else {
    // Standard TFLOPS/GPU
    const tflops = m.tflopsPerGPU;
    // Use ~ prefix for LLaMA 405B to match doc convention
    if (label.includes('405B')) {
      simCol = `~${Math.round(tflops)}`;
    } else {
      simCol = tflops.toFixed(1);
    }
  }

  const pubCol = tc.publishedTFLOPS ?? tc.publishedTokPerSGPU ?? '--';
  // Nemotron published column
  const publishedStr = label === 'Nemotron-4 340B' ? '8-10.3s (pub)' : pubCol;

  console.log(`| ${label === 'LLaMA 3.1 405B 8K' ? 'LLaMA 3.1 405B' : label} | ${e.gpuLabel} | ${simCol} | ${publishedStr} | ${tc.notes} |`);
}

console.log();
console.log(`All ${results.length} benchmarks completed successfully.`);
