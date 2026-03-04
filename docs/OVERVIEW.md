# Documentation Overview

Entry point for the ML Cluster Simulator documentation system.

## What This Simulator Does

A client-side distributed training and inference simulator for ML workloads.
Given a model, GPU cluster, and parallelism strategy, it estimates:

- **Training**: MFU, throughput, step time, memory breakdown, cost, GPU-hours
- **Inference**: TTFT, TPOT, throughput, KV cache memory, bottleneck analysis

What it does NOT model: custom fused kernel speedups (FlashAttention reduces
memory but not simulated compute time; fused optimizer), NCCL async P2P
pipelining, packet-level network congestion (fabric bandwidth scaling IS
modeled), OS-level jitter beyond stochastic straggler penalties, or weight
sparsity.

## Architecture

```
ConfigStore (Zustand + Immer)
    ↓
SimulationEngine.configure()
    ↓
selectStrategy()  →  ddp / fsdp / zero-1 / create3DParallelStrategy()
    ↓
buildContext()    →  model spec, cluster config, hyperparams, GA
    ↓
strategy.computeAnalysis(context)
    ├── computeMemoryPerGPU()   →  MemoryBreakdown
    ├── computeCommunication()  →  comm volumes + bandwidth
    ├── computeTiming()         →  fwd/bwd/comm/optimizer time
    └── computeAnalysis()       →  MFU, HFU, throughput, cost
    ↓
generateRecommendations()  →  up to 3 actionable suggestions
    ↓
SimulationResult  →  Dashboard
```

### Directory Map

```
src/core/
  strategies/     All parallelism strategies (base, ddp, fsdp, zero, tp, sp, pp, 3d, lora, overlap)
  simulation/     Engine, recommendations, exploration, optimizer
  hardware/       GPU specs, interconnects, topology, presets
  models/         Model registry, architectures, primitives, MoE
  inference/      Latency, KV cache, memory, speculative, continuous batching, optimizer, exploration
  roofline/       Compute roofline model
  cost/           GPU hourly rates
  analysis/       Sensitivity analysis, perturbable parameters
  collectives/    Collective operation primitives
  physics/        Derived physics constants
  validation/     Config validators, benchmark validation
src/game/       Learn Mode tasks, validation, glossary, setup
src/rpg/        Space RPG missions, skills, hardware progression, scoring
src/stores/       Zustand config store
src/components/   React UI components
src/utils/        Share URL encoding, formatting
```

## Key Design Decisions

1. **All combined strategies use `create3DParallelStrategy()`** with different
   `dpType` (fsdp/ddp/zero-1). The 8 UI strategies map to this single engine.

2. **EP subdivides DP** — not a separate GPU dimension. `totalGPUs = TP×PP×CP×DP`.
   EP must divide DP. Expert params use smaller DP/EP AllReduce groups.

3. **MFU always uses BF16 peak** — even for FP8 training. Industry convention
   (PaLM definition). FP8 benefits appear as higher MFU, not higher peak.

4. **Efficiency = pure GPU compute** — all communication costs modeled
   explicitly. No double-counting between efficiency penalties and comm models.

5. **GA auto-derived**: `ceil(GBS / (MBS × DP))`. Not user-editable.

6. **Benchmark configs specify correct AC type** (full/selective/off) matching
   published training setups. AC type dramatically affects MFU.

7. **DP scaling uses `numNodes`** for standalone strategies (DDP, FSDP, ZeRO)
   and `dp` for 3D strategies. NCCL hierarchical collectives run intra-node
   rings first, then inter-node tree -- fabric congestion scales with
   inter-node hops, not total GPUs.

## State Management

- **Zustand + Immer** middleware for immutable state updates
- `TrainingConfig.strategyType` type union must match `SimulationConfig.strategyType`
- `setStrategy()` updates `state.training.*` parallelism fields and calls `recalcGA()`

### Share URL Wire Format

Version 1 (`src/utils/share.ts`). Base64url-encoded JSON with short keys:

- Training: `v`, `mode`, `model`, `gpu`, `n`, `gpn`, `gbs`, `mbs`, `seq`,
  `st`, `tp`, `pp`, `dp`, `ep`, `cp`, `pr`, `ckpt`, `fa`, `sp`, `ps`, `is`,
  `goal`, `tok`, `price`, optional `acg` (selective checkpointing), `sl` (selective stored layers),
  `cpi` (CP implementation: `'a'` = all-gather, omit = ring), `ft`/`lr`/`ltm` (LoRA)
- Inference: `v`, `mode`, `model`, `gpu`, `n`, `gpn`, `bs`, `iseq`, `oseq`,
  `wpr`, `kvpr`, `fa`, `pa`, `cb`, `tp`, `ep`, `sd`, `dm`, `nst`, `ar`, `price`

### localStorage

`STORAGE_VERSION = 1`. Adding new fields is safe (merged onto defaults).
Renaming or removing fields requires a version bump.

## Versioning

Three independent version numbers:

| Version | Location | Current | Breaking Change |
|---------|----------|---------|-----------------|
| App | `package.json` | 1.0.0 | Cosmetic — bump for milestones |
| Wire format | `share.ts` `v: 1` | 1 | Old shared links stop working |
| Storage | `config.ts` `STORAGE_VERSION` | 1 | Users' saved configs reset |

Adding new optional fields to wire format or storage does NOT require a bump
(both have fallback-to-defaults logic). Changing simulation math, adding
models/GPUs also does not require a bump.

## Testing

- Runner: `npx vitest run tests/`
- Type check: `npx tsc -b` (always run — tests pass at runtime even with type errors)
- Directories: `tests/unit/`, `tests/acceptance/`, `tests/calibration/`,
  `tests/regression/`, `tests/analysis/`, `tests/golden/`, `tests/helpers/`
- Bounds: ±15-25% of actual simulator output, verified against published values
- Philosophy: enforce correctness, never document known bugs as passing tests

## Definitions

| Term | Definition |
|------|------------|
| MFU | Model FLOPS Utilization: `6PD / (time × N × peak_bf16)`. Useful work only. |
| HFU | Hardware FLOPS Utilization: `(6+2f) × P × D / (time × N × peak_bf16)`, f=0 (no AC), 0.13–0.22 (selective), 1.0 (full). LoRA: `(4+2f)`. |
| Efficiency | Pure GPU compute efficiency (kernel utilization). All comm explicit. |
| GA | Gradient Accumulation steps: `ceil(GBS / (MBS × DP))` |
| DP | Data Parallelism — replicate model, shard data |
| TP | Tensor Parallelism — shard layers across GPUs within a node |
| PP | Pipeline Parallelism — shard layers across pipeline stages |
| SP | Sequence Parallelism — extends TP to shard along sequence dimension |
| EP | Expert Parallelism — shard MoE experts across DP subgroups |
| CP | Context Parallelism — split sequence across GPUs for long contexts (ring or all-gather) |
| Pipeline bubble | Idle time from PP stage imbalance. 1F1B: `(pp-1)/(pp-1+m)`. Interleaved 1F1B: `(pp-1)/(pp-1+m*v)`, v=virtual stages. DualPipeV: `(pp-1)/(pp-1+6*m)` when m≥2pp. |

## How to Read These Docs

### For Humans

1. Start here (OVERVIEW.md) for orientation
2. Read [HARDWARE.md](HARDWARE.md) for GPU specs — foundational inputs
3. Read [MODELS.md](MODELS.md) for architectures and parameter counting
4. Read [PHYSICS.md](PHYSICS.md) for simulation math — builds on hardware + models
5. Read [STRATEGIES.md](STRATEGIES.md) for per-strategy implementations
6. Read [INFERENCE.md](INFERENCE.md) for the inference latency model
7. Read [OPTIMIZER.md](OPTIMIZER.md) for auto-optimization and recommendations
8. Read [BENCHMARKS.md](BENCHMARKS.md) for calibration and validation
9. Read [LEARNING.md](LEARNING.md) for the interactive learning modes

### For Agents

1. Read OVERVIEW.md for orientation
2. Jump to the specific doc for your task:
   - Simulation math → [PHYSICS.md](PHYSICS.md) (use anchor links in TOC)
   - Strategy details → [STRATEGIES.md](STRATEGIES.md)
   - GPU/interconnect specs → [HARDWARE.md](HARDWARE.md)
   - Model architecture → [MODELS.md](MODELS.md)
   - Inference latency → [INFERENCE.md](INFERENCE.md)
   - Optimizer/recommendations → [OPTIMIZER.md](OPTIMIZER.md)
   - Calibration data → [BENCHMARKS.md](BENCHMARKS.md)
   - Learn Mode / RPG → [LEARNING.md](LEARNING.md)

### File Index

| File | Lines | Content |
|------|-------|---------|
| [OVERVIEW.md](OVERVIEW.md) | ~180 | This file — architecture, definitions, reading guide |
| [PHYSICS.md](PHYSICS.md) | ~1550 | All formulas, constants, rationale, penalty audit |
| [STRATEGIES.md](STRATEGIES.md) | ~700 | All parallelism strategy implementations |
| [HARDWARE.md](HARDWARE.md) | ~450 | GPU specs, interconnects, topology, cost model |
| [MODELS.md](MODELS.md) | ~750 | Model registry, architecture types, parameter counting |
| [INFERENCE.md](INFERENCE.md) | ~400 | Inference latency, KV cache, speculative decoding |
| [OPTIMIZER.md](OPTIMIZER.md) | ~350 | Recommendation engine, auto-optimizer, grid search |
| [BENCHMARKS.md](BENCHMARKS.md) | ~690 | Calibration data, known gaps, source references |
| [LEARNING.md](LEARNING.md) | ~120 | Learn Mode (60 tasks), Space RPG (26 missions) |
