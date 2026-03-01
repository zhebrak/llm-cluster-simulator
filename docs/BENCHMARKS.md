# Calibration Benchmarks Reference

> **See also:** [PHYSICS.md](PHYSICS.md) (formulas) | [HARDWARE.md](HARDWARE.md) (GPU/IB specs) | [MODELS.md](MODELS.md) (model definitions)

## Table of Contents
1. [Validation Methodology](#1-validation-methodology)
2. [Training MFU Benchmarks](#2-training-mfu-benchmarks)
3. [Throughput Cross-Check](#3-throughput-cross-check)
4. [DP Scaling Validation](#4-dp-scaling-validation)
5. [GPU-Hours Projections](#5-gpu-hours-projections)
6. [Memory Formula Validation](#6-memory-formula-validation)
7. [Inference Benchmarks](#7-inference-benchmarks)
8. [Known Calibration Gaps](#8-known-calibration-gaps)
9. [Source References](#9-source-references)

---

## 1. Validation Methodology

### Target Accuracy

All calibration tests target +-15-25% of published results. Test bounds are set
tightly around actual simulator output (+-15%), then verified that the published
value falls within the range. A 5x range (e.g., 400-2000) is not a real test.

### MFU vs HFU Confusion

Many published papers report "MFU" when they actually mean HFU (Hardware FLOPS
Utilization, which includes activation recompute overhead). Values published as
"MFU" above 50% are almost certainly HFU. The simulator computes both:

- **MFU** (PaLM definition): `6 * P * D / (time * N * peak_bf16)` -- useful
  work only, never includes recompute. This is the primary metric shown on the
  dashboard.
- **HFU**: `(6+2f) * P * D / (time * N * peak_bf16)` where f = recomputeFraction
  (0 = no AC, 0.13–0.22 = selective, 1.0 = full). LoRA: `(4+2f)`. Shown as a
  sub-value when checkpointing is on.

Both are computed in `src/core/strategies/base.ts:computeAnalysis()`.

### Activation Checkpointing Type Matters

The correct AC type (full / selective / off) must be specified per benchmark to
match published training setups. AC type dramatically affects MFU:

| Benchmark | AC Type | Sim MFU | Published |
|-----------|---------|---------|-----------|
| Nemotron-4 340B | Selective | 41.2% | 41-42% |
| LLaMA 3.1 405B 8K | Off | 41.1% | ~40% |
| GPT-3 175B | Full | 42.0% | 44.2% |
| IBM LLaMA 2 7B | Off | 57.0% | 57% |

### Testing Philosophy

- Tests must enforce correctness, not document known bugs.
- Never set thresholds to accept values known to be wrong. Fix the code first,
  then tighten the thresholds.
- Comments like "Active ~11B (shared dense MLP not modeled, actual is ~17B)"
  with ranges accepting 9-14B are anti-patterns. The test should enforce ~17B
  and the model code should produce ~17B.
- Test runner: `npx vitest run tests/`
- Test directories: `tests/unit/`, `tests/acceptance/`, `tests/calibration/`,
  `tests/regression/`

---

## 2. Training MFU Benchmarks

### 2.1 Primary — Hopper (H100, H800)

| Model | GPUs | Strategy | Config Details | AC | Sim MFU | Published | Delta | Source |
|-------|------|----------|----------------|----|---------|-----------|-------|--------|
| LLaMA 3.1 405B 8K | 16384 H100 | 3D (TP=8 PP=16 DP=128) | Interleaved v=4, GBS=2048 MBS=1 seq=8192 | Off | 41.1% | ~40% | +1.1% | [Meta paper](https://arxiv.org/abs/2407.21783) §3.3.2, Table 4 |
| LLaMA 3.1 405B 131K | 16384 H100 | 3D (TP=8 PP=16 CP=16 DP=8) | Interleaved v=4, all-gather CP, GBS=2048 MBS=1 seq=131072 | Full | 37.2%\* | 38%\* | -0.8% | [Meta paper](https://arxiv.org/abs/2407.21783) §3.3.2, Table 4 |
| DeepSeek V3 671B FP8† | 2048 H800 | 3D (TP=4 PP=8 DP=64 EP=32) | DualPipeV, GBS=8192 MBS=2 seq=4096 | Full | 44.7% | 43.7% | +1.0% | [DeepSeek V3 paper](https://arxiv.org/abs/2412.19437) §3.1; MFU from [ISCA 2025](https://arxiv.org/abs/2505.09343) Table 4 |
| Nemotron-4 340B | 6144 H100 | 3D (TP=8 PP=12 DP=64) | VP=8 interleaved, GBS=768 MBS=1 seq=4096 | Selective | 41.2% | 41-42% | -0.8% | [Megatron-Core blog](https://developer.nvidia.com/blog/train-generative-ai-models-more-efficiently-with-new-nvidia-megatron-core-functionalities/) Table 2 |
| OLMo 3 32B | 1024 H100 | FSDP (DP=1024) | GBS=1024 MBS=1 seq=8192 | Selective | 43.4% | ~41% | +2.4% | [OLMo 3 paper](https://arxiv.org/abs/2512.13961); stored-layers auto-resolves at 95% capacity target |
| Nemotron-4 15B DGXC | 64 H100 | FSDP-TP (TP=2 DP=32) | GBS=256 MBS=4 seq=4096 | Selective | 54.3% | ~56% | -1.7% | [NVIDIA DGXC](https://github.com/NVIDIA/dgxc-benchmarking) |
| Mixtral 8x22B MoE | 128 H100 | 3D (TP=2 PP=8 DP=8 EP=4) | GBS=256 MBS=1 seq=4096 | Full | 43.9% | 46.3% | -2.4% | [MoE Parallel Folding](https://arxiv.org/abs/2504.14960) (2025) |

\* Model FLOPs MFU — PaLM MFU uses `6PD` (constant per token), but at 131K seq the actual
  model FLOPs are 2.34× higher due to quadratic attention (PaLM MFU: 16.6%). With
  all-gather CP (matching Meta's Megatron-LM implementation), Model FLOPs MFU matches
  published 38%. The dashboard reports both metrics for this config.

† Sim uses TP=4 PP=8 EP=32 (different decomposition from paper's PP=16 EP=64 no TP);
  matches published throughput.

### 2.2 Primary — Ampere (A100)

| Model | GPUs | Strategy | Config Details | AC | Sim MFU | Published | Delta | Source |
|-------|------|----------|----------------|----|---------|-----------|-------|--------|
| GPT-3 175B | 1024 A100 | 3D (TP=8 PP=8 DP=16) | Interleaved v=2, GBS=1536 MBS=1 seq=2048 | Full | 42.0% | 44.2% | -2.2% | [Narayanan et al. 2021](https://arxiv.org/abs/2104.04473) §5.1 |
| IBM LLaMA 2 7B | 128 A100 | FSDP (DP=128) | GBS=256 MBS=2 seq=4096 | Off | 57.0% | 57% | -0.0% | [IBM blog 2023](https://research.ibm.com/blog/pytorch-fsdp) |
| BLOOM 176B | 384 A100 | ZeRO-1-TP-PP (TP=4 PP=12 DP=8) | GBS=2048 MBS=2 seq=2048 | Full | 45.1% | ~48% | -2.9% | [BigScience 2022](https://arxiv.org/abs/2211.05100) |

**MT-NLG 530B**: +3.4pp overshoot (2022). DP scaling points validated in §4. Demoted to T2.

### 2.3 Secondary

| Model | GPUs | Strategy | Config Details | AC | Sim | Published | Delta | Notes |
|-------|------|----------|----------------|----|-----|-----------|-------|-------|
| OLMo 2 32B | 1280 H100 | FSDP (DP=1280) | GBS=2048 MBS=1 seq=4096 | Selective | 40.4% | ~38% | +2.4pp | Same arch as OLMo 3 (shorter seq); [AI2 OLMo 2 blog](https://allenai.org/blog/olmo2-32B) |
| MT-NLG 530B | 2240 A100 | ZeRO-1-TP-PP (TP=8 PP=35 DP=8) | GBS=1920 MBS=1 seq=2048 | Full | 43.4% | ~40% | +3.4pp | Largest model anchor |
| DBRX 132B MoE | 16 H100 | ZeRO-1-TP (TP=4) | GBS=32 MBS=1 seq=2048 | Full | 40.2% | ~37.5%‡ | +2.7pp | ‡ Derived from >50% HFU × 6/8 |
| Mosaic LLaMA 70B | 512 H100 | FSDP-TP (TP=8 DP=64) | GBS=1024 MBS=2 seq=2048 | Full | 38.3% | 41.25% | -3.0pp | [llm-foundry](https://github.com/mosaicml/llm-foundry) |
| NeMo 8B | 8 H100 | FSDP (DP=8) | GBS=128 MBS=1 seq=8192 | Selective | 45.6% | 45.7%§ | -0.1pp | § Implied from 725 TFLOPS |
| NeMo 70B | 64 H100 | 3D (TP=4 PP=8 DP=2) | Interleaved v=2, GBS=128 MBS=1 seq=8192 | Selective | 45.1% | 48.3%§ | -3.2pp | § Implied from 727 TFLOPS |
| NeMo 405B CP=2 | 1024 H100 | 3D (TP=8 PP=8 CP=2 DP=8) | Interleaved v=2, GBS=512 MBS=1 seq=8192 | Selective | 44.3% | 53.6%§ | -9.3pp | § Implied from 763 TFLOPS |
| LLaMA 2 70B | 2048 A100 | FSDP-TP (TP=8 DP=256) | GBS=512 MBS=1 seq=4096 | Full | 44.9% | ~35-40% | -- | Estimated range |
| Qwen2 57B-A14B MoE | 64 H100 | 3D (TP=2 PP=4 DP=8 EP=4) | GBS=256 MBS=1 seq=4096 | Full | 46.7% | 35.3% | +11.4pp | Published MFU reflects early Megatron-Core EP baseline, not optimized; [MoE Parallel Folding](https://arxiv.org/abs/2504.14960) |
| MT-NLG 530B (350n) | 2800 A100 | ZeRO-1-TP-PP (TP=8 PP=35 DP=10) | GBS=1920 MBS=1 seq=2048 | Full | 42.1% | ~39% | +3.1pp | DP scaling point |
| MT-NLG 530B (420n) | 3360 A100 | ZeRO-1-TP-PP (TP=8 PP=35 DP=12) | GBS=1920 MBS=1 seq=2048 | Full | 40.8% | ~36% | +4.8pp | DP scaling point |

‡ DBRX published ">50% MFU" is HFU. With full AC: MFU = HFU × 6/8. Lower
bound: 50% × 0.75 = 37.5%.

§ NeMo TFLOPS convention: NeMo reports TFLOPS using `4 × flopsPerToken` (not
`6P`). Implied MFU derived by converting published TFLOPS → step time → 6PD
MFU. Gap grows with parallelism complexity: 8B +0.8%, 70B +8.1%, 405B +21.0%.

#### NeMo Automodel MoE Control Benchmarks

Control benchmarks for the grouped GEMM efficiency model. All entries have
`expertIntermediateSize ≥ 1536` (at/above threshold) → baseline factor (0.952).
All configs use TP=1, so the expert TP treatment only affects the physics
floor (expert FLOPs ÷ EP instead of ÷ TP). Overshoot vs published is expected:
NeMo Automodel uses framework-level optimizations (automodel dispatch scheduling,
kernel fusion, memory management) that account for ~50-70% of measured step time.
These are framework-level overheads that an analytical physics simulator does not
model. The +108-224% gaps bound framework overhead, not calibration targets.

Source: [NeMo Automodel](https://docs.nvidia.com/nemo/automodel). All published
numbers are BF16 HFU TFLOPS on H100 SXM. `rawSim` used (skip memory validation).

| Model | GPUs | Strategy | Config Details | AC | Sim HFU | Published HFU | Delta | Notes |
|-------|------|----------|----------------|----|---------|---------------|-------|-------|
| DeepSeek V3 671B | 256 H100 | 3D (TP=1 PP=8 EP=16) | DualPipeV, GBS=512 MBS=1 seq=4096 | Full | ~520 | 250 | +108% | 256e/8a, expertInt=2048 |
| DeepSeek V3 671B | 1024 H100 | 3D (TP=1 PP=8 EP=64) | DualPipeV, GBS=2048 MBS=1 seq=4096 | Full | ~620 | 216 | +187% | Larger cluster, more EP overhead |
| Kimi K2 1T | 256 H100 | 3D (TP=1 PP=16 EP=16) | DualPipeV, GBS=512 MBS=1 seq=4096 | Full | ~612 | 189 | +224% | 384e/8a, expertInt=2048 |
| GPT-OSS 120B | 64 H100 | 3D (TP=1 PP=1 EP=32) | GBS=512 MBS=4 seq=4096 | Full | ~615 | 231 | +166% | 128e/4a, expertInt=2880 |
| GPT-OSS 20B | 8 H100 | 3D (TP=1 PP=1 EP=8) | GBS=512 MBS=8 seq=4096 | Full | ~801 | 279 | +187% | 32e/4a, expertInt=2880 |

#### NeMo Automodel Fine-Grained MoE

Qwen3 30B-A3B EP8 has `expertIntermediateSize = 768` (below 1536 threshold) →
`getGroupedGemmEfficiency()` in `gpu.ts` applies a power-law throughput penalty.
Fine-grained MoE with small expert width suffers from L2 cache thrashing between
expert weight matrices and wave quantization in grouped GEMM kernels. The factor
uses raw `expertIntermediateSize` (architectural property, not divided by TP) with
threshold 1536: at/above → baseline (0.952), below → power-law penalty scaled by
expert count per GPU. Directionally validated against SonicMoE (arXiv:2512.14080)
grouped GEMM benchmarks (49-57% of H100 BF16 peak). Overshoot is lower than
control benchmarks because the grouped GEMM penalty reduces simulated throughput.

| Model | GPUs | Strategy | Config Details | AC | Sim HFU | Published HFU | Delta | Notes |
|-------|------|----------|----------------|----|---------|---------------|-------|-------|
| Qwen3 30B-A3B | 16 H100 | 3D (TP=1 PP=2 EP=8) | GBS=512 MBS=2 seq=4096 | Full | ~383 | 241 | +59% | 128e/8a, expertInt=768 |

### 2.4 Calibration Notes

#### DeepSeek V3 FP8 (custom recomputation matching full AC)

Custom selective recomputation (RMSNorm + MLA up-proj + SwiGLU output) has
similar total recompute overhead to our full AC model, so full AC is used. Sim
gives 44.7% vs published 43.7% (+1.0pp delta). Uses MoE runtime residual (0.97)
via 3D parallel strategy, near 1.0 because most MoE overhead is modeled
explicitly (expert GEMM efficiency, EP transport, load imbalance).
EP backward comm uses a separate multiplier (2× with AC, 1× without) instead of
the compute backward multiplier (2.85×), reflecting symmetric all-to-all volume.
FP8 dispatch adds quant/dequant overhead (BF16↔FP8 conversions on critical path,
~50% hidden by CUDA stream overlap). EP coordination penalty:
`0.030 × crossNodeFraction × volumeScale × log2(ep)` scales with EP degree for
cross-node overhead (L2 cache pollution, sync barriers, NCCL buffer management).

#### Nemotron-4 340B (PP=12, VP=8 pipeline config)

[Megatron-Core blog](https://developer.nvidia.com/blog/train-generative-ai-models-more-efficiently-with-new-nvidia-megatron-core-functionalities/) Table 2 specifies PP=12,
VP=8 (virtual_pipeline_model_parallel_size=8). 96 layers / 12 PP / 8 VP = 1
layer per virtual stage, giving 10.3% pipeline bubble.

#### LLaMA 3.1 405B (AC off for 8K pre-training)

[Meta paper](https://arxiv.org/abs/2407.21783) §3.3.2 (Table 4) explicitly states "pre-train on
sequences of 8K tokens without activation checkpointing." AC=off is used
per the published config, producing 41.1% MFU.

#### IBM LLaMA 2 7B (AC off for peak throughput)

[IBM blog](https://research.ibm.com/blog/pytorch-fsdp) explicitly states "activation checkpointing was
turned off for 7B for the highest throughput." Published 57% is true MFU.
Sim gives 57.0% (-0.0pp).

#### BLOOM 176B (PP divisibility)

[BigScience 2022](https://arxiv.org/abs/2211.05100) reports 150 TFLOPS using
Megatron-LM `72Bslh²` (6PD). PP=12 on 70 layers is not cleanly
divisible (5.83 layers/stage); sim uses Math.ceil(). ALiBi positional
encoding (no RoPE), non-gated MLP (4h intermediate). Marginally OOMs in
sim (81.25 GB vs 80 GB limit) — uses raw sim (bypasses memory validation).

#### MT-NLG 530B (+3.4pp overshoot at DP=8, T2)

[Smith et al. 2022](https://arxiv.org/abs/2201.11990) reports 126 TFLOPS
at DP=8. Sim overshoots at +3.4pp (43.4% vs ~40%). Three DP scale points
validated: DP=8 (43.4%), DP=10 (42.1%), DP=12 (40.8%) — strictly
decreasing, matching published trend. Demoted to T2: 2022 paper with
limited config detail. DP scaling points remain valuable.

#### Nemotron-4 15B DGXC (highest MFU anchor)

[NVIDIA DGXC](https://github.com/NVIDIA/dgxc-benchmarking) reports 56% genuine 6PD MFU
on 64 H100s with FSDP-TP (TP=2, DP=32). Highest MFU anchor. Distinct from
paper's production training (34.3% MFU at 3072 GPUs with TP=8).

#### OLMo 3 32B (stored-layers selective AC)

[OLMo 3 paper](https://arxiv.org/abs/2512.13961) reports ~41% MFU on 1024 H100s
with FSDP, seq=8192, selective AC. Full selective AC at this sequence length
exceeds 80 GB. The simulator auto-resolves stored-layers at 95% of GPU capacity
(5% headroom for NCCL workspace, CUDA graphs, transient buffers): fewer layers
use selective checkpointing (store MLP+shared, discard attention) and more
use full recompute. Sim gives 43.4% MFU (+2.4pp) with ~94% memory utilization.
The 95% capacity target matches OLMo-core/Megatron-LM practice of leaving a
memory margin for budget mode.

#### NeMo TFLOPS conversion (4×flopsPerToken convention)

NeMo published TFLOPS use a different formula (4 × flopsPerToken, not 6P).
The ratio `(4 × flopsPerToken) / (6P)` varies: ~1.61 (8B), ~1.52 (70B),
~1.44 (405B). Converting to implied step time: 8B matches within +0.8%
(FSDP), 70B +8.1% (TP+PP, interleaved 1F1B), 405B +21.0% (TP+PP+CP,
interleaved 1F1B). The gap grows with parallelism complexity, not the TFLOPS
convention. Nemotron-4 340B (same Megatron-Core codebase, reported as MFU
directly) matches within -0.8pp (§2.1).

---

## 3. Throughput Cross-Check

| Model | GPUs | Sim TFLOPS/GPU | Published | Notes |
|-------|------|----------------|-----------|-------|
| GPT-3 175B | 1024 A100 | 131.0 | 138 | -5%, Megatron-LM |
| LLaMA 3.1 405B | 16384 H100 | ~407 | ~400 | +2%, AC off |
| LLaMA 3 8B | 8 H100 | 421.3 | -- | Single node reference |
| IBM LLaMA 2 7B | 128 A100 | 4398 tok/s/GPU | 3700 tok/s/GPU | +19% overshoot |
| Nemotron-4 340B | 6144 H100 | 2569.4ms/step | 8-10.3s (pub) | Different GBS scaling |
| BLOOM 176B | 384 A100 | 140.6 | 150 | Megatron-LM, AC recompute in published |
| MT-NLG 530B | 2240 A100 | 135.4 | 126 | Megatron-LM, AC recompute in published |

---

## 4. DP Scaling Validation

Validated against [MegaScale (NSDI '24)](https://www.usenix.org/conference/nsdi24/presentation/jiang-ziheng) DP scaling curves.

| Model | GPUs | TP | PP | DP Range | MFU Pattern |
|-------|------|----|----|----------|-------------|
| GPT-3 175B | 512-6144 | 8 | 8 | 4 -> 192 | Strictly monotone decrease |
| LLaMA 70B | 512-1024 | 8 | 1 | 64 -> 128 | 3-15% relative drop |
| LLaMA 3 405B | 8192-16384 | 8 | 16 | 64 -> 128 | >2% relative drop |
| MT-NLG 530B | 2240-3360 | 8 | 35 | 8 -> 10 -> 12 | Strictly monotone decrease |

Key findings:
- MFU strictly decreases with DP degree (validated GPT-3 175B: DP=4 through
  DP=192 on A800).
- BW penalty and overlap ceiling continue degrading at extreme DP -- no
  plateau in penalty functions past DP=64.
- DP scaling is model- and configuration-dependent: large models (405B) show
  smaller relative drops than small models because they are more
  compute-dominated.

---

## 5. GPU-Hours Projections

### Published Training Runs

| Model | Tokens | GPUs | Sim GPU-Hours | Published GPU-Hours | Delta | Notes |
|-------|--------|------|---------------|---------------------|-------|-------|
| LLaMA 2 7B | 2T | 1024 A100 | ~295K | ~184K | +60% | FSDP DP=1024, AC on |
| LLaMA 2 70B | 2T | 2048 A100 | ~1.52M | ~1.72M | -12% | FSDP-TP (TP=8) |
| LLaMA 3 8B | 15T | 512 H100 | ~549K | 1.3M | -58% | Published includes annealing, context extension, post-training |
| LLaMA 3 70B | 15T | 2048 H100 | ~4.06M | 6.4M | -37% | Same operational overhead |
| LLaMA 3.1 405B | 15T | 16384 H100 | ~25.76M | 30.84M | -16% | AC=off (Meta §3.3.2), interleaved 1F1B |
| DeepSeek V3 | 14.8T | 2048 H800 | ~2.11M | 2.79M | -24% | FP8, DualPipe-V, TP=4 PP=8 EP=32 |
| GPT-3 175B | 300B | 1024 V100 | ~1.54M | ~3.1M | -50% | Published est. from training duration |
| Phi-4 14B | 9.8T | 1920 H100 | ~588K | ~967K | -39% | [Microsoft 2024](https://arxiv.org/abs/2412.08905), FSDP |
| Phi-4 Mini 3.8B | 5T | 1024 A100 | ~450K | ~344K | +31% | Sim overshoots published |

### Cross-Model Consistency

- 70B requires more GPU-hours than 7B for the same token count (validated).
- Step time strictly increases with model size on the same hardware
  (125M < 7B < 13B < 70B, validated on H100 FSDP).
- DDP memory strictly increases with model size (validated).

### Notes on GPU-Hours Accuracy

**Published GPU-hours include non-training overhead** (data pipeline,
checkpointing to storage, evaluation runs, cluster restarts, hardware
failures). Raw GPU-hours comparisons are less meaningful than MFU comparisons
because they conflate training efficiency with operational overhead.

Key patterns:

- **LLaMA 3.1 405B**: closest match (-16%). Large-scale 3D parallel is
  dominated by compute, minimizing the framework overhead gap.
- **DeepSeek V3**: -24% gap consistent with DualPipe scheduling overhead,
  checkpoint writes, and cluster coordination not modeled.
- **LLaMA 3 8B/70B**: large gaps (-63%, -37%) because published figures include
  annealing phases, context extension training, and post-training experiments.
- **LLaMA 2 7B**: +60% overshoot — published 184K figure likely covers only the
  final training phase, not total compute. Sim uses AC=on; published likely used AC=off.
- **Phi-4 Mini**: +31% overshoot — A100 FSDP at 128 nodes, sim may be
  pessimistic on A100 comm model at this scale.
- **GPT-3 175B**: -50% gap from V100 framework overhead (pre-Megatron
  optimizations, FP16 mixed precision vs modern BF16 pipelines).

---

## 6. Memory Formula Validation

### ZeRO Reduction Factors

Baseline: BF16 training with AdamW, per-parameter byte counts:

| Strategy | Per-Param Bytes | Breakdown |
|----------|----------------|-----------|
| DDP | 18 | params(2 bf16) + grads(4 fp32) + optimizer(12: master(4) + m1(4) + m2(4)) |
| ZeRO-1 (N GPUs) | 6 + 12/N | params(2) + grads(4) + optimizer(12/N) |
| ZeRO-2 (N GPUs) | 2 + 16/N | params(2) + grads(4/N) + optimizer(12/N) |
| FSDP / ZeRO-3 (N GPUs) | 18/N | All state sharded across N GPUs |

FP8 + AdamW: params(1) + grads(4 fp32) + optimizer(master(2 bf16) + m1(4) + m2(4)) = 15 bytes/param.

### Validated Memory Points

| Config | Sim Memory/GPU | Expected | Notes |
|--------|----------------|----------|-------|
| LLaMA 2 7B DDP 8x H100 | ~140 GB | ~121 GB static + activations | 6.7B x 18 = 121 GB static |
| LLaMA 2 7B FSDP 8x H100 | ~34.4 GB | ~15.1 GB static + activations | 121 GB / 8 = 15.1 GB static |
| GPT-3 175B DDP 8x H100 | >2.5 TB | ~2.8 TB static | 175B x 16 = 2.8 TB, impossible without sharding |
| GPT-3 175B FSDP 256x A100 | <80 GB | ~12.3 GB static + activations | Fits in 80 GB A100 |

### Memory Ordering Invariants (Validated)

For the same model (LLaMA 3 8B, 8x H100):

```
DDP total > ZeRO-1 total > ZeRO-3 total ~ FSDP total
```

- ZeRO-1: Same param memory as DDP, optimizer sharded ~8x.
- ZeRO-3 and FSDP: Within 2x of each other (both shard all state).
- FSDP-TP (16 GPUs, TP=4, DP=4): Lower param memory than FSDP (8 GPUs)
  because total sharding = TP x DP = 16 > 8.

### Gradient Memory

DDP stores gradients in **fp32** (4 bytes/param), not bf16. Total per-param for
BF16 training: 2 (params) + 4 (grads) + 12 (optimizer) = **18 bytes** when
grads are fp32, or 2 + 2 + 12 = **16 bytes** when grads are bf16. The
simulator uses the fp32 gradient convention for DDP (matching PyTorch default)
and bf16 for FSDP (ReduceScatter in bf16).

### Activation Memory

`estimateActivationMemory()` in `base.ts` counts all backward-required tensors
per layer:

- LayerNorm inputs: 2h
- LayerNorm outputs: 2h
- Q, K, V projections: qDim + 2 x kvDim
- Attention output: h
- Gated MLP (SwiGLU): 3 x intermediateSize (gated) or 2 x intermediateSize (standard)

Full coefficient: `5h + qDim + 2*kvDim + mlpIntermCoeff*I`

Selective checkpointing: Stores all N layers x (shared + MLP activations),
discards attention activations (cheap to recompute with Flash Attention).

SP activation multiplier: `1/tp` (exact). With Sequence Parallelism, all
activation tensors are effectively 1/tp per rank -- TP-sharded tensors split
along hidden dim, SP-sharded tensors split along sequence dim.

---

## 7. Inference Benchmarks

> **Inference benchmarks validate the physics** (memory fits, compute-boundedness
> transitions, scaling directions, TP near-linearity) but **not absolute
> throughput against optimized serving runtimes**. vLLM, TensorRT-LLM, and
> SGLang use continuous batching, paged attention, CUDA graph replay, and fused
> kernels that the analytical model does not capture. Sim throughput is typically
> 2-10x lower than optimized runtimes.

### Weight Memory

| Model | Precision | Sim Weights | Expected | Notes |
|-------|-----------|-------------|----------|-------|
| LLaMA 2 7B | BF16 | ~12.6 GB | 6.74B × 2 / 1024³ = 12.6 GiB | Exact match |
| LLaMA 2 70B | BF16 | ~128 GB | 69.0B × 2 / 1024³ = 128.5 GiB | Exact match |
| LLaMA 2 7B | INT8 | ~6.3 GB | Half of BF16 | Exact match |
| LLaMA 2 7B | INT4 | ~3.1 GB | Quarter of BF16 | Exact match |

### KV Cache Memory

Formula: `2 * num_layers * num_kv_heads * head_dim * precision_bytes * seq_len * batch`

| Model | Attention | Per-Token KV | Notes |
|-------|-----------|-------------|-------|
| LLaMA 2 7B | MHA (32 KV heads) | ~0.5 MB | 2 x 32 x 32 x 128 x 2 bytes |
| LLaMA 2 70B | GQA (8 KV groups) | ~2.5 MB | Reduced by GQA factor |
| DeepSeek V3 | MLA | Compressed latent | 576 values/layer, replicated across TP |

### Inference Latency (TTFT)

| Model | GPU | TP | Batch | Sim TTFT | Published | Notes |
|-------|-----|----|----|----------|-----------|-------|
| LLaMA 3 8B | H100 x1 | 1 | 1 | ~21ms | -- | Analytical model baseline |
| LLaMA 3 8B | A100 x1 | 1 | 1 | ~61.6ms | -- | 2.9x H100 (TFLOPS ratio) |
| LLaMA 3 70B | H100 x8 | 8 | 1 | ~23ms | ~123ms | Sim faster (no API overhead) |
| Mixtral 8x7B | H100 x2 | 2 | 1 | ~16.8ms | sub-200ms | MoE uses activeParams |
| Qwen 2.5 7B | H100 x1 | 1 | 1 | ~19.8ms | -- | Similar to LLaMA 8B class |
| Grok-1 314B | H200 x16 | 16 | 1 | ~13.2ms | -- | MoE, large TP |

### Inference Throughput

| Model | GPU | TP | Batch | Sim tok/s | Published | Notes |
|-------|-----|----|----|-----------|-----------|-------|
| LLaMA 3 8B | H100 x1 | 1 | 128 | ~6,111 | ~12,500 (vLLM) | Physics-only, no CUDA graphs |
| LLaMA 3 70B | H100 x8 | 8 | 64 | ~4,710 | -- | Compute-bound at high batch |
| LLaMA 2 7B | H100 x1 | 1 | 128 | ~4,307 | -- | Bandwidth-bound |

### Consumer GPU Inference

| Model | GPU | Precision | Sim tok/s | Published | Source |
|-------|-----|-----------|-----------|-----------|--------|
| LLaMA 2 7B | RTX 4090 | BF16 | 51.7 | 50-55 | [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA) |
| LLaMA 2 7B | RTX 4090 | INT4 (GPTQ) | 127.1 | 150-194 | [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA) |
| LLaMA 3 8B | RTX 4090 | BF16 | 45.2 | 45-50 | [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA) |
| Mistral 7B | RTX 4090 | BF16 | 49.4 | 50-58 | [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA) |
| LLaMA 3 8B | RTX 4090 | INT4 (GPTQ) | 116.1 | ~150 | [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA) |
| LLaMA 2 7B | RTX 4090 | Q4_K_M | 120.1 | 125-150 | [llama.cpp benchmarks](https://github.com/ggerganov/llama.cpp) |
| LLaMA 3 8B | RTX 4090 | Q4_K_M | 109.3 | 104-150 | [llama.cpp benchmarks](https://github.com/ggerganov/llama.cpp) |
| LLaMA 2 7B | RTX 3090 | BF16 | 48.0 | 45-50 | [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA) |
| LLaMA 2 7B | RTX 3090 | Q4_K_M | 111.6 | 100-112 | [llama.cpp benchmarks](https://github.com/ggerganov/llama.cpp) |

INT4 (GPTQ/AWQ) undershoots published — Marlin/TRT-LLM use W4A16 kernels that
dequantize inside the matmul at register level, avoiding the separate bandwidth
cost our model applies (1.20× overhead). Q4_K_M (GGUF) uses fused GGML kernels
with lower overhead (1.10×) and matches published ranges.

RTX 4090 / RTX 3090 throughput ratio: ~1.077 (matches hardware memory
bandwidth ratio: 1.008 / 0.936 TB/s = 1.077). Validated across BF16, INT4,
and Q4_K_M precisions.

### Server GPU Inference

Datacenter GPU decode throughput at batch=1. Sim throughput is 2-10× lower
than optimized runtimes (vLLM, TRT-LLM) — the analytical model does not
capture CUDA graphs, fused kernels, or continuous batching.

| Model | GPU | Precision | TP | Sim tok/s | Published | Notes |
|-------|-----|-----------|----|-----------|-----------|----|
| LLaMA 2 7B | H100 SXM | BF16 | 1 | 171 | 350-400 (vLLM) | Baseline server GPU |
| LLaMA 3 8B | H100 SXM | BF16 | 1 | 150 | 300-350 (vLLM) | |
| LLaMA 3 70B | H100 SXM | BF16 | 8 | 138 | 900-1200 (vLLM) | TP comm overhead modeled |
| LLaMA 2 7B | A100 80GB | BF16 | 1 | 104 | 105-125 (vLLM) | A100 coverage |
| LLaMA 3 8B | A100 80GB | BF16 | 1 | 91 | 90-110 (vLLM) | |
| LLaMA 2 7B | H200 SXM | BF16 | 1 | 245 | -- | H200 coverage |
| LLaMA 3 8B | H200 SXM | BF16 | 1 | 214 | -- | |
| LLaMA 3 70B | H200 SXM | FP8 | 1 | 53 | -- | 70B FP8 fits 141 GB |
| LLaMA 2 7B | B200 SXM | BF16 | 1 | 391 | -- | Fastest server GPU |

**Hardware ratios** (strongest tests — bandwidth ratios are physical constants):

| GPU Pair | BW Ratio (theoretical) | Sim Ratio (BF16) | Sim Ratio (INT4) | Sim Ratio (FP8) |
|----------|------------------------|-------------------|-------------------|-----------------|
| H200 / H100 | 1.433 (4.8/3.35) | 1.430 | 1.425 | 1.428 |
| A100 / H100 | 0.609 (2.039/3.35) | 0.610 | 0.612 | -- |
| B200 / H100 | 2.299 (7.7/3.35) | 2.283 | 2.262 | -- |

Ratios are stable across precisions (BF16/INT4/FP8 within ±2%), confirming
the simulator correctly models bandwidth-bound decode.

**Quantization speedup on H100** (LLaMA 2 7B, batch=1):

| Precision | tok/s | Speedup vs BF16 |
|-----------|-------|-----------------|
| BF16 | 171 | 1.00× |
| INT8 | 272 | 1.59× |
| FP8 | 284 | 1.66× |
| INT4 | 419 | 2.44× |

**Roofline transition**: H200/H100 ratio decreases from 1.42 (batch=1,
BW-bound) to 1.21 (batch=128, partially compute-bound), validating the
roofline model's compute-bound transition.

### Speculative Decoding

Published references: [NVIDIA TRT-LLM blog](https://developer.nvidia.com/blog/boost-llama-3-3-70b-inference-throughput-3x-with-nvidia-tensorrt-llm-speculative-decoding/) (H200: 3.55x for 70B+1B, K=10
FP8), [AMD ROCm](https://rocm.blogs.amd.com/artificial-intelligence/spec_decode_mi300x/README.html) (MI300X: 1.5-2.0x BF16 K=8).

| Target | Draft | GPU | Precision | K | Sim Speedup | Published | Notes |
|--------|-------|-----|-----------|---|-------------|-----------|-------|
| LLaMA 70B | 1B | H200 | FP8 | 10 | 2.53x | 3.55x (TRT-LLM) | Engine optimizations not modeled |
| LLaMA 70B | 8B | H200 | FP8 | 10 | 1.36x | 2.63x (TRT-LLM) | Larger draft reduces speedup |
| LLaMA 3 70B | 8B | MI300X | BF16 | 8 | 1.56x | 1.5-2.0x (AMD) | Close match |
| LLaMA 2 70B | 7B | H100 | FP8 | 5 | 1.74x | 1.5-3.0x (lit.) | K=5 sweet spot |
| LLaMA 3 8B | 125M | H100 | BF16 | 5 | 2.41x | -- | Consumer-scale model |
| LLaMA 2 7B | 125M | RTX 4090 | BF16 | 5 | 2.37x | -- | Consumer GPU |

Key validated properties:
- Acceptance rate monotonically increases with draft/target size ratio.
- Optimal K is in [3, 12] range.
- Batch degradation: speedup drops from ~2.47x (batch=1) to ~0.43x
  (batch=128) for 70B+1B on H100 FP8 TP=2 -- high batch amortizes decode,
  making speculation overhead dominant.
- MoE targets: marginal benefit (expert routing adds latency to verification).

Sim speedups are lower than TRT-LLM published because the physics-only model
does not capture engine-level speculation optimizations (tree-based
verification, fused kernels, CUDA graph replay).

### H800 / A800 Benchmarks

China-export GPU variants with reduced interconnect bandwidth.

| Model | Config | GPU | Sim MFU | Published | Notes |
|-------|--------|-----|---------|-----------|-------|
| DeepSeek V3 671B FP8 | 2048 GPUs, TP=4 PP=8 EP=32 | H800 | 45.0% | 43.7% | +1.3pp |
| DeepSeek V3 671B BF16 | Same config | H800 | ~0.87 memUtil | -- | Fits with EP activation reduction |

- H800 < H100 MFU for the same config (reduced NVLink: 400 vs 900 GB/s).
  H800 retains >50% of H100 MFU because NVLink is not the sole bottleneck
  (IB, compute efficiency also factor).
- A800 <= A100 MFU, delta < 5 percentage points absolute. Single-node FSDP
  comm overhead is small, so the NVLink reduction (400 vs 600 GB/s) has
  limited impact.

### Validated Properties

- **Bandwidth efficiency**: Model-size-dependent sigmoid:
  `0.35 + 0.50 * (1 - 1/(1 + totalGB/5))` where `totalGB` = weights + KV cache.
  Tiny models: ~0.37, 1B: ~0.49, 7B: ~0.72, 70B: ~0.83, 405B: ~0.85.
- **TP scaling**: Near-linear throughput improvement up to within-node TP
  (validated). Cross-node TP degrades due to inter-node bandwidth limits.
- **Batch scaling**: Throughput scales approximately linearly with batch size.
- **MoE inference FLOPs**: Uses `activeParams` (not `totalParams`) for MoE models.

---

## 8. Known Calibration Gaps

### Published "MFU" > 50% = HFU

Multiple papers report "MFU" values above 50% (e.g., DBRX ">50% MFU"). These
are almost certainly HFU (includes recompute overhead) mislabeled as MFU. True
MFU at 50%+ requires impossibly high GPU utilization for current hardware. When
a benchmark claims >50% "MFU" with activation checkpointing enabled, assume it
is HFU.

### NeMo Automodel MoE overshoot (+108–224%)

The simulator overshoots NeMo Automodel published HFU TFLOPS by +108–224% on
MoE models (DeepSeek V3, Kimi K2, GPT-OSS). This reflects framework-level
overhead (automodel dispatch scheduling, kernel fusion, memory management) that
accounts for ~50–70% of measured step time — overhead an analytical physics
simulator does not model. These are framework overhead bounds, not calibration
targets. Fine-grained MoE (Qwen3 30B-A3B, expertIntermediateSize < 1536) has a
smaller gap (+59%) due to the grouped GEMM efficiency penalty reducing simulated
throughput. See [§2.3 NeMo Automodel MoE Control Benchmarks](#nemo-automodel-moe-control-benchmarks)
and [§2.3 Fine-Grained MoE](#nemo-automodel-fine-grained-moe) for details.

---

## 9. Source References

### Training Papers

- **[Narayanan et al. 2021](https://arxiv.org/abs/2104.04473)**: "Efficient Large-Scale Language Model Training on
  GPU Clusters Using Megatron-LM" -- GPT-3 175B benchmark, 138 TFLOPS/GPU on
  A100, pipeline parallelism formulas.

- **[Chowdhery et al. 2022](https://arxiv.org/abs/2204.02311)**: "PaLM: Scaling Language Modeling with Pathways" --
  Defines MFU (Model FLOPS Utilization) as `6PD / (time * peak)`, useful work
  only, no recompute. The standard MFU definition used throughout.

- **[Touvron et al. 2023](https://arxiv.org/abs/2307.09288)**: "LLaMA 2: Open Foundation and Fine-Tuned Chat
  Models" -- 184k-1.72M GPU-hours on A100, training configuration details.

- **[Dubey et al. 2024](https://arxiv.org/abs/2407.21783)**: "The LLaMA 3 Herd of Models" -- 16384 H100 training,
  ~400 TFLOPS/GPU (Table 4), 38-43% MFU, CP=16 at 131K sequences, §3.3.2: no AC for 8K.

- **[DeepSeek-AI 2024](https://arxiv.org/abs/2412.19437)**: "DeepSeek-V3 Technical Report" -- 2048 H800 training,
  42.9% BF16-equivalent MFU with FP8, DualPipe scheduling, EP=32 with
  device-limited routing (M=4).

- **[DeepSeek-AI 2025](https://arxiv.org/abs/2505.09343)**: "Insights into DeepSeek-V3/R1 Training
  Infrastructure" -- 43.73% non-causal MFU (Table 4), training system details.

- **[NVIDIA 2024](https://developer.nvidia.com/blog/train-generative-ai-models-more-efficiently-with-new-nvidia-megatron-core-functionalities/)**: "Megatron-Core Benchmarks" -- Nemotron-4 340B: PP=12 VP=8
  (Table 2), 41-42% MFU, selective activation recomputation.

- **[Cai et al. 2025](https://arxiv.org/abs/2504.14960)**: "MoE Parallel Folding for Scalable MoE Training" --
  Megatron-Core baseline MFU for Mixtral 8x22B (46.3%) and Qwen2 57B-A14B
  (35.3%) on 128/64 H100s with EP=4. Intra-node EP benchmarks.

- **[NVIDIA NeMo Automodel](https://docs.nvidia.com/nemo/automodel)**: NeMo Automodel performance summary --
  BF16 HFU TFLOPS on H100 SXM for DeepSeek V3 (250/216 at 256/1024 GPUs),
  Kimi K2 (189 at 256 GPUs), GPT-OSS 120B (231 at 64 GPUs), GPT-OSS 20B
  (279 at 8 GPUs). Control benchmarks for grouped GEMM efficiency model.

- **[SonicMoE (arXiv:2512.14080)](https://arxiv.org/abs/2512.14080)**: Grouped expert GEMM benchmarks --
  57% of H100 BF16 peak (up-proj, E=128) and 49% (down-proj). Directional
  validation for the grouped GEMM efficiency penalty model.

- **[BigScience 2022](https://arxiv.org/abs/2211.05100)**: "BLOOM: A 176B-Parameter Open-Access Multilingual
  Language Model" -- 384 A100s, ZeRO-1-TP-PP (TP=4 PP=12), 150 TFLOPS/GPU,
  ALiBi positional encoding, non-gated MLP.

- **[Smith et al. 2022](https://arxiv.org/abs/2201.11990)**: "Using DeepSpeed and Megatron to Train Megatron-Turing
  NLG 530B" -- 2240-3360 A100s, ZeRO-1-TP-PP (TP=8 PP=35), 113-126
  TFLOPS/GPU, DP scaling from DP=8 to DP=12.

- **[NVIDIA DGXC 2024](https://github.com/NVIDIA/dgxc-benchmarking)**: "DGXC Benchmarking" -- Nemotron-4 15B on
  64 H100s, FSDP-TP (TP=2 DP=32), 56% MFU (genuine 6PD), highest MFU anchor.

### Methodology Papers

- **[Korthikanti et al. 2022](https://arxiv.org/abs/2205.05198)**: "Reducing Activation Recomputation in Large
  Transformer Models" -- Selective checkpointing formulas, activation memory
  analysis, non-Flash-Attention score memory multiplier (2.5x: pre-softmax +
  post-softmax + dropout mask).

- **[Leviathan et al. 2023](https://arxiv.org/abs/2211.17192)**: "Fast Inference from Transformers via Speculative
  Decoding" -- Theoretical speedup curves for given acceptance rates.

### Systems Papers

- **[MegaScale (NSDI '24)](https://www.usenix.org/conference/nsdi24/presentation/jiang-ziheng)**: Large-scale training systems -- DP scaling
  validation, MFU degradation curves from DP=4 to DP=192.

- **[IBM 2023](https://research.ibm.com/blog/pytorch-fsdp)**: "FSDP Training of LLaMA" (blog post) -- 57% MFU for LLaMA 2
  7B on 128 A100s, AC explicitly off.

- **[OLMo 2 Team 2025](https://arxiv.org/abs/2501.00656)**: "OLMo 2: The best fully open language model
  to date" -- OLMo 2 7B/13B/32B training details, FSDP at scale.

- **[AI2 2025](https://allenai.org/blog/olmo2-32B)**: OLMo 2 32B blog -- ~38% MFU on 1280 H100s,
  selective activation checkpointing.

- **[OLMo 3 Team 2025](https://arxiv.org/abs/2512.13961)**: "OLMo 3" -- OLMo 3 32B training on
  1024 H100s, ~41% MFU, FSDP, seq=8192, selective AC.

- **[Brown et al. 2020](https://arxiv.org/abs/2005.14165)**: "Language Models are Few-Shot Learners" -- GPT-3 175B
  training on 1024 V100s, ~3.1M GPU-hours estimated from training duration.

- **[Microsoft 2024](https://arxiv.org/abs/2412.08905)**: "Phi-4 Technical Report" -- 1920 H100s, 9.8T tokens,
  ~21 days. Phi-3 Mini/Phi-4 Mini training details.

- **[NVIDIA TRT-LLM 2024](https://developer.nvidia.com/blog/boost-llama-3-3-70b-inference-throughput-3x-with-nvidia-tensorrt-llm-speculative-decoding/)**: Speculative decoding benchmarks -- H200: 3.55x
  speedup for LLaMA 70B+1B with K=10 FP8.

- **[AMD ROCm 2024](https://rocm.blogs.amd.com/artificial-intelligence/spec_decode_mi300x/README.html)**: Speculative decoding on MI300X -- 1.5-2.0x speedup for
  LLaMA 70B with BF16 K=8.

### GPU and Interconnect Specs

See [HARDWARE.md §1](HARDWARE.md#1-gpu-specifications) for GPU specs and
[HARDWARE.md §3](HARDWARE.md#3-interconnects) for IB auto-detection.

An explicit `ibVersion` parameter overrides the auto-detection.

---

## Appendix: Compute Efficiency Model

See [PHYSICS.md §2 — Compute Efficiency](PHYSICS.md#2-compute-efficiency-model) for the full
efficiency formula, [§5 — DP Scaling](PHYSICS.md#5-dp-scaling-laws) for group-size penalties,
[§7 — Communication Overhead](PHYSICS.md#7-communication-overhead-model) for the two-term
protocol model, and [§8 — Overlap Models](PHYSICS.md#8-overlap-models) for TP/PP/FSDP overlap.
