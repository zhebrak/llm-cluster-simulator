# Recommendation Engine & Auto-Optimizer Reference

> **See also:** [STRATEGIES.md](STRATEGIES.md) (strategy definitions) | [PHYSICS.md](PHYSICS.md) (simulation model)

## Table of Contents

1. [Recommendation Engine](#1-recommendation-engine)
2. [Generator Details](#2-generator-details)
3. [Auto-Optimizer](#3-auto-optimizer)
4. [Exploration Grid](#4-exploration-grid)
---

## 1. Recommendation Engine

The recommendation engine is a unified system that produces actionable
configuration suggestions after every simulation run. It lives in
`src/core/simulation/recommendations.ts`.

### Integration with the simulation engine

`engine.run()` always calls `generateRecommendations()` after simulation
completes. Both OOM and valid configs use the same engine -- there is no
separate OOM-specific path. The result is attached to the `SimulationResult`
returned to the UI.

### How generators work

- There are **15 generators** in a priority-ordered array (`GENERATORS`).
- The engine iterates through them and keeps the **first 2 non-null results**
  that pass validation.
- A positive confirmation message ("Well-optimized..." or "Solid baseline...")
  is prepended at position 0 when the config has no validation errors, so the
  UI shows up to 3 items total: 1 status + up to 2 actionable suggestions.

### Validation framework

Generators return a `RecommendationCandidate` with an optional
`configMutation` function and an optional `validationMode` (`'throughput'` or
`'memory'`).

When `configMutation` is present, `validateCandidate()` re-simulates the
mutated config and checks:

1. Memory utilization does not exceed the candidate's `maxMemoryUtilization`
   ceiling (defaults to 1.0).
2. **If current config is OOM** (memUtil > 1.0): any mutation that fits in
   memory is accepted. Throughput comparison is meaningless when the current
   config cannot run.
3. **Throughput mode**: mutated `tokensPerSecond` must exceed current by at
   least 1%. Uses `tokensPerSecond` (not MFU) because MFU's denominator
   changes with precision -- FP8 doubles peak TFLOPS.
4. **Memory mode**: mutated memory must improve by at least 1%. Throughput
   regression is allowed up to 1% when current memory utilization >= 85%,
   otherwise no regression is tolerated.

Before re-simulating, `gradientAccumulationSteps` is cleared so
`buildContext()` recomputes GA for the new parallelism layout. Mutations that
change TP/PP/EP alter DP, which invalidates GA.

Generators without `configMutation` (heuristic-only) always pass validation.

### Config-fix generators

When the config has validation errors (but is not merely OOM), a separate
`CONFIG_FIX_GENERATORS` array is used instead of `GENERATORS`. This contains
two generators:

- **tpHeadsFix**: TP must evenly divide attention heads. Suggests the largest
  valid TP below current that divides heads.
- **pipelineLayersFix**: PP x v must evenly divide model layers. Suggests
  switching to standard 1F1B or reducing PP to a valid divisor.

---

## 2. Generator Details

All 15 generators in priority order. Generators marked with **\*** provide a
`configMutation` and are validated via re-simulation. Generators marked with
**(heuristic)** return a message only.

### 1. strategyUpgrade *

Suggests moving to a more capable parallelism strategy.

| Trigger | Mutation |
|---------|----------|
| DDP + model >7B + memUtil >70% | DDP -> FSDP (memory mode) |
| 1D + model >30B + multi-node | 1D -> FSDP-TP, TP=gpusPerNode |
| 1D + model >13B + commOverhead >30% | 1D -> FSDP-TP, TP=gpusPerNode |
| 2D + model >70B + memUtil >85% | 2D -> FSDP-TP-PP, PP=2 (memory mode) |
| DDP-TP-PP + memUtil >75% | DDP-TP-PP -> FSDP-TP-PP (memory mode) |

### 2. memoryCritical (heuristic)

Fires when memUtil > 95%. Composes a message based on what is already enabled
(activation checkpointing, Flash Attention, micro-batch size). Does not
provide a mutation -- the advice involves training-semantic changes (batch
size, sequence length, strategy).

### 3. pipelineReduction *

Suggests reducing PP when memory > 90% on 3D strategies with PP > 2. Halves
PP (must evenly divide model layers). Rationale: total FSDP sharding stays
the same (TP x PP x DP = totalGPUs), but more DP means less gradient
accumulation and better throughput.

### 4. scaleEfficiency (heuristic)

Diagnostic generator for low-MFU or over-provisioned clusters.

- MBS exceeds samples-per-GPU and MFU < 35%: suggest increasing GBS.
- MFU < 15% with large batch: suggest fewer GPUs or larger model.
- MFU < 15% with high memory: suggest increasing GBS.
- Model <50M params/GPU on >32 GPUs: cluster is over-provisioned.

### 5. dataParallelism *

Fires when DP = 1 on multi-GPU 2D/3D configs. All GPUs are consumed by
TP/PP/CP with no data-parallel replicas. Tries halving TP first (higher comm
overhead per degree), then halving PP if TP is already 1. Validated: confirms
throughput improves.

### 6. tpTopology *

Two triggers:

- **Cross-node TP** (tp > gpusPerNode): suggests reducing TP to stay within a
  single node for NVLink-speed communication.
- **Under-utilized intra-node BW** (tp < gpusPerNode, model >13B, heads
  divide evenly, commOverhead >20%): suggests increasing TP to fill the node.

### 7. activationCheckpointing *

Three-way suggestions based on current AC state and memory utilization:

| Current state | Trigger | Suggestion |
|---------------|---------|------------|
| AC off | memUtil > 80% | Enable selective AC |
| Selective | memUtil > 100% (OOM) | Switch to full AC |
| Full | memUtil < 70% | Switch to selective AC |

### 8. flashAttention *

Suggests enabling Flash Attention when disabled and GPU supports it. Only
fires when memUtil > 60% or sequence length >= 4096. Validated in memory mode.

### 9. pipelineBubbleAndSchedule *

Dual trigger for 3D strategies with PP > 1. Evaluated in this order:

1. **DualPipeV degraded mode**: if current schedule is `dualpipe-v` and GA <
   2 x PP, suggests increasing GBS to provide sufficient micro-batches.
2. **DualPipeV upgrade**: if bubble > 10% and GA >= 2 x PP, suggests
   switching to `dualpipe-v` (near-zero bubble from bidirectional overlap).
3. **Interleaved upgrade**: if schedule is `1f1b` and (bubble > 10% or
   memUtil > 85%), suggests `interleaved-1f1b` with the largest valid v
   (virtual pipeline stages). Message varies based on trigger (bubble vs
   memory). Interleaved reduces both bubble AND activation memory (in-flight
   microbatches: PP -> ceil(PP/v)).
4. **Increase v**: if already on interleaved and bubble > 10%, suggests the
   next valid v.
5. **Fallback**: generic bubble advice (heuristic only, no mutation).

### 10. sequenceParallelism *

Suggests enabling SP when TP > 1 and SP is off (on 2D or 3D strategies).
Only fires when memUtil > 70% or commOverhead > 20%. Validated in memory
mode. SP distributes activation memory across the TP group -- all activation
tensors become effectively 1/tp per rank.

### 11. contextParallelism *

Suggests enabling CP = 2 when memUtil > 70%, seqLength >= 8192, and the
split would keep chunks >= 8192 tokens. Validates that the new parallelism
layout is feasible (TP x PP x CP x DP = totalGPUs). Memory mode validation.

### 12. expertParallelism *

For MoE models with >= 8 experts. Two strategies:

- **Strategy A**: increase EP within current TP layout. Finds the largest
  valid EP that divides both numExperts and DP.
- **Strategy B**: reduce TP to free GPUs for EP (compound mutation). Tries
  halving TP iteratively, checking if valid EP candidates appear.

Validation mode is memory when memUtil > 85%, otherwise throughput.

### 13. communicationOverhead *

Fires when commOverhead > 25% and MBS < 4 with memory headroom (memUtil <
65%). Suggests doubling micro-batch size to amortize communication.

### 14. microBatchSizing *

Fires when MFU < 40%, MBS < samples-per-GPU, GA >= 8, commOverhead > 15%,
and memUtil < 75%. Suggests doubling MBS to reduce gradient accumulation
steps and communication overhead.

### 15. gbsScaling *

Tries both GBS x 2 and GBS / 2. Each candidate is fully simulated and
scored by `timeToTrainHours`. Picks whichever direction reduces total
training time by at least 0.5%.

---

## 3. Auto-Optimizer

The auto-optimizer is a pure function in `src/core/simulation/optimizer.ts`
that takes a `SimulationConfig`, target token count, and sequence length, and
returns an `OptimizationResult` with the best config found.

### Scoring metric

All phases optimize **time-to-train** (hours), not MFU or throughput alone.
Time-to-train accounts for both step time and number of steps:

```
timeToTrain = stepTimeMs * ceil(targetTokens / tokensPerStep) / 3.6e6
```

### Phase 1: Fix (max 10 iterations)

Resolves OOM conditions. Iterates through all `GENERATORS`, applies each
mutation, and keeps whichever one reduces memory utilization the most. Accepts
any mutation that reduces memory, even if the result is still OOM. Multiple
iterations compound (e.g., DDP -> FSDP -> FSDP-TP).

If the initial config is completely invalid (simulation throws), falls back to
a safe FSDP config with TP=1 PP=1 before giving up.

### Phase 2: Greedy improvement (max 200 iterations)

Single-mutation hill climbing. Each iteration:

1. Builds strategy context for the current config.
2. Runs all `GENERATORS` to produce candidate mutations.
3. Simulates each mutation, rejecting any that exceed `maxSafeMemoryUtil`.
4. Keeps the mutation with the best time-to-train improvement (>= 0.5%).
5. Stops when no mutation improves.

### GBS alignment snap (between Phase 2 and Phase 3)

If greedy mutations leave `GBS % (MBS × DP) != 0`, GBS is snapped down to
the nearest multiple of `MBS × DP` so gradient accumulation is always an
exact integer. The snap floor is `MBS × DP` (i.e. GA >= 1).

### Phase 3: Explore (max 5000 candidates)

Brute-force grid search via `generateTrainingCandidates()`. Every candidate
is simulated and scored by time-to-train. The best candidate replaces the
current config if it improves by at least 0.5%.

### Post-explore refinement (max 50 iterations)

A second greedy pass (same logic as Phase 2) runs on the best config found
during exploration to locally optimize it.

### Final GBS alignment snap

The same `MBS × DP` snap runs again after post-explore refinement and before
building the result, catching any misalignment introduced by the second
greedy pass.

### Memory safety threshold

```typescript
maxSafeMemoryUtil = totalGPUs >= 256 ? 0.85 : 0.87
```

Larger clusters need more headroom for NCCL memory fragmentation, straggler
GPU memory spikes, and collective buffer overhead.

### Optimizer caps

Two hard caps prevent the optimizer from exploring configurations outside the
published benchmark range:

```
MAX_OPTIMIZER_DP = 512          // DP groups >512 are unvalidated
MAX_PIPELINE_MICROBATCHES = 64  // No published pipeline config uses >64
```

The DP cap applies to all phases (fix, greedy, explore). It prevents the
optimizer from dropping PP to inflate DP on large clusters (e.g. LLaMA 405B
on 16384 GPUs pushing to DP=2048).

The micro-batch cap applies only when PP > 1. It prevents degenerate pipeline
configs where hundreds of micro-batches mask bubble overhead that the formula
`(pp-1)/(pp-1 + 6×m)` underestimates (in-flight activation memory pressure,
per-micro-batch launch overhead, scheduling latency). PP=1 configs retain the
looser GA ≤ 256 cap since there is no bubble to exploit.

Both caps are enforced in `exploration.ts` (Phase 3 grid) and via
`exceedsOptimizerCaps()` in `optimizer.ts` (Phases 1, 2, and post-explore).

### Result structure

```typescript
interface OptimizationResult {
  success: boolean;              // true if final config fits in memory
  originalConfig: SimulationConfig;
  optimizedConfig: SimulationConfig;
  changelog: ChangelogEntry[];   // field-by-field diff
  beforeMetric: number;          // time-to-train (hours), Infinity if OOM
  afterMetric: number;           // time-to-train (hours) after optimization
  totalSimulations: number;      // total simulate() calls across all phases
  phases: { fix: number; greedy: number; explore: number };
}
```

---

## 4. Exploration Grid

The exploration grid lives in `src/core/simulation/exploration.ts` and is
used by the optimizer's Phase 3.

### Search dimensions

| Dimension | Values | Constraints |
|-----------|--------|-------------|
| Strategy | ddp, fsdp, zero-1, fsdp-tp, zero1-tp, ddp-tp-pp, zero1-tp-pp, fsdp-tp-pp | 1D requires TP=1 PP=1; 2D requires TP>1 PP=1; 3D requires TP>1 or PP>1 |
| TP | 1, 2, 4, 8 | <= gpusPerNode, must divide numAttentionHeads |
| PP | 1, 2, 4, 8, 16 | <= numLayers |
| EP | 1, 2, 4, 8, 16, 32 | MoE only, must divide numExperts, must divide DP |
| CP | 1, 2, 4, 8 | seqLength/CP >= 8192; CP > 1 requires seqLength >= 8192 |
| GBS | 4x span around current | 5 raw values: current/4, current/2, current, current×2, current×4 (floor 64, ceiling from critical-batch-size scaling law). Each raw value is snapped to the nearest multiple of `MBS × DP` so GA is always an exact integer; the snap floor is `MBS × DP` (guarantees GBS >= MBS × DP). |
| MBS | 1, 2, 4, 8 | GA <= 256 (GA <= 64 when PP > 1) |
| DP | derived from TP×PP×CP | DP <= 512 |
| AC | full, selective | Always enabled during optimization |
| Precision | current (or bf16/fp16 if current is fp32) | Never auto-switches to FP8 |
| Pipeline schedule | 1f1b; interleaved-1f1b with v in {2,3,4,6,8}; dualpipe-v | Interleaved: numLayers % (PP×v) = 0. DualPipeV: GA >= 2*PP |

### Parallelism constraint

```
TP x PP x CP x DP = totalGPUs
EP divides DP
```

SP is enabled for all 2D and 3D candidates.

### Budget management

The grid can produce a very large Cartesian product. The budget is
`MAX_EXPLORE_SIMS` (5000) for dense models and `MAX_EXPLORE_SIMS_MOE`
(10000) for MoE models, whose EP dimension inflates the grid ~6×.

When candidates exceed the budget, stratified sampling preserves coverage
across (strategy, AC granularity, EP) strata. 70% of the budget is allocated
evenly across strata (guaranteeing representation of rare-but-important
corners like a single optimal `{zero1-tp-pp, selective, EP=8}` config),
and 30% is allocated proportionally to stratum size. Within each stratum,
a seeded Fisher-Yates shuffle selects the sampled candidates. Each candidate
is simulated and scored by time-to-train.


