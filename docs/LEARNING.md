# Learning Modes

Two interactive modes teach distributed training and inference through the simulator itself. Both use the same simulation engine and validation framework. Only one can be active at a time.

## Learn Mode

60 structured tasks across 6 tracks. Entry: **Learn** button in header, or `?learn` URL parameter.

### Tracks

| Track | Tasks | Topics |
|-------|-------|--------|
| Training Beginner | 10 | Mixed precision, activation memory, checkpointing, Flash Attention, DDP, FSDP, multi-node |
| Training Intermediate | 10 | Tensor parallelism, NVLink boundary, pipeline parallelism, bubble reduction, SP, MoE, FP8 |
| Training Advanced | 10 | 3D parallelism, EP, CP, communication budgets, LoRA/QLoRA, reproducing published configs |
| Inference Beginner | 10 | Quantization, KV cache, bandwidth bottlenecks, batching, TTFT/TPOT, continuous batching |
| Inference Intermediate | 10 | TP for large models, TP degree tradeoffs, 405B serving, long-context KV, speculative decoding |
| Inference Advanced | 10 | MoE inference, expert parallelism, long context, latency SLAs, cost optimization, full stack |

Tasks are linear within each track. Progress is tracked per-track and persisted to localStorage.

### Task Structure

Each task provides:
- **Briefing** — scenario description with glossary-annotated terms (`{{termId}}` → inline tooltip)
- **Learning objectives** — 2–4 action-oriented goals
- **Setup** — initial config applied to the simulator (model, GPU, strategy, parallelism, precision, etc.)
- **Winning criteria** — conditions on simulation output (`mfu > 0.30`, `success == true`, `latency.ttft < 500`)
- **Expected changes** — validates the player modified the intended parameters (e.g., `precision: changed`, `numGPUs: unchanged`)
- **Hints** — 2–3 progressive reveals (concept → principle → answer)
- **Success explanation** — post-completion educational text with glossary terms and paper links

### Glossary

38 terms defined in `src/game/glossary.ts`. Used across all task text via `{{termId}}` or `{{termId|custom display text}}` syntax. Rendered as inline tooltips. First occurrence per track only.

### Key Files

```
src/game/
  types.ts              GameTask, WinningCriterion, ExpectedChange, ValidationResult
  tasks/                6 files, 10 tasks each (training-*.ts, inference-*.ts)
  tasks/index.ts        ALL_TASKS array, getTaskById(), getTasksForLevel()
  validation.ts         Config validation, expected-change checks
  setup.ts              Applies TaskSetup config overrides
  glossary.ts           38 term definitions
  constants.ts          Mode/difficulty labels, storage key
src/stores/game.ts      Zustand store (active state, progress, task states, hints, attempts)
src/components/game/    GameOverlay, GameMenu, TaskHUD, SuccessModal, HintCarousel, etc.
```

## Space RPG

26 missions across 3 narrative arcs set aboard a generation ship. Entry: **Play** button in header, or `?play` URL parameter.

### Arc Structure

| Arc | Missions | Focus |
|-----|----------|-------|
| 1 — Survival | 8 | Inference: quantization, bandwidth, batching, TP, KV cache, cost |
| 2 — Discovery | 11 | Mixed: TP scaling, speculative decoding, FSDP, checkpointing, LoRA, FP8, PP, multi-objective |
| 3 — Wonder | 7 | Advanced: continuous batching, CP, EP, pipeline scheduling, multi-workload capstone |

### Mission Types

- **Regular** — simulation task with winning criteria, expected changes, hints, success narrative
- **Multi-objective** — 2–4 independent sub-tasks (different modes/hardware), all must pass. Missions 2-10, 3-5, 3-6
- **Pivot** — narrative-only (no simulation). Story breakpoints between arcs. Missions 1-8, 2-11, 3-7

### Progression

**Prerequisites (DAG):** Missions unlock when all prerequisites are complete — not strictly linear. Some arcs have parallel branches that converge at later missions.

**Hardware tiers:** GPU availability is gated by mission completion:

| Tier | Unlocked by | Hardware |
|------|-------------|----------|
| Starting | — | T4 (2×16 GB), RTX 4090 (4×24 GB) |
| Archive Vault | Mission 1-6 | +4× A100 80 GB |
| Derelict Station | Mission 2-2 | +4× A100 80 GB |
| Resupply Drone | Mission 2-7 | +16× H100 SXM 80 GB |
| Asteroid Forge | Mission 3-1 | +496× H100 SXM 80 GB |

**Skills:** 7 skills with 26 total stars. Each mission awards 1–3 skill stars on completion.

| Skill | Stars |
|-------|-------|
| Model Parallelism | 6 |
| Resource Efficiency | 5 |
| Numerical Precision | 4 |
| Hardware | 3 |
| Batching | 3 |
| Memory Optimization | 3 |
| Inference Optimization | 2 |

**Ranks:** Title progression based on mission milestones — Compute Officer (default) through First Contact Commander (8 ranks total).

### Mission Structure

Same validation framework as Learn Mode: winning criteria + expected changes + hints + success text. Briefings are narrative (no ML jargon); hints and success explanations are technical.

### Key Files

```
src/rpg/
  types.ts                RPGArc, RPGMission, MissionObjective, HardwareTier
  missions/index.ts       ALL_ARCS, ALL_MISSIONS, DAG utilities
  missions/arc1-survival.ts   8 missions
  missions/arc2-discovery.ts  11 missions
  missions/arc3-wonder.ts     7 missions + completion arc
  skills.ts               7 skills, star computation, rank titles, promotions
  hardware.ts             5 hardware tiers, GPU unlock progression
  scoring.ts              Win condition evaluation
src/stores/rpg.ts         Zustand store (mission states, progress, config snapshots)
src/components/rpg/       RPGOverlay, MissionSelect, MissionHUD, MissionSuccess, ArcComplete, etc.
```

## Shared Mechanics

- **Mutual exclusion:** Entering one mode soft-exits the other.
- **Config snapshots:** Both modes save the current config on entry and restore it on exit.
- **Persistence:** Progress stored in localStorage (`llm-sim-game` / `llm-sim-rpg`). Survives page reloads.
- **Validation:** Both use the simulation engine to evaluate winning criteria against live results. Expected changes compare config snapshots (before vs. after).
- **Hints:** Progressive reveal, no penalty for using them. Concept → principle → answer.
