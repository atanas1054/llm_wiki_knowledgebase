---
title: PhysicalAI-Autonomous-Vehicles Benchmark
type: concept
sources: [raw/papers/Qwen-Drive-1.0_ An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving.md, raw/papers/DriveWAM_ Video Generative Priors Enable Scalable World-Action Modeling for Autonomous Driving.md, raw/papers/Alpamayo-R1_ Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail.md, "raw/papers/ReDrive_ Shaping Representations with World Modeling for End-to-End Driving.md"]
related: [sources/redrive.md, concepts/foundation-backbones-for-ad.md, concepts/alpasim-benchmark.md, sources/qwen-drive-1.0.md, sources/drivewam.md, sources/alpamayo-r1.md, concepts/nuscenes-waymo-evals.md, concepts/navsim-benchmark.md, concepts/world-model-for-ad.md, research-directions.md]
created: 2026-08-17
updated: 2026-09-30
confidence: medium
---

## What It Is

A large-scale real-world driving benchmark released by NVIDIA alongside Alpamayo-R1 ([[sources/alpamayo-r1.md]]). Per [[sources/drivewam.md]] (the first wiki source to evaluate on it):

- **~1,700 hours** of driving logs
- **306,152 clips** of 20 seconds each
- Official splits: **153,625 train / 90,928 val / 61,599 test**
- Front-view camera stream + ego-motion labels (multi-sensor logs exist; papers so far use front-view only)

Note: the wiki's [[sources/alpamayo-r1.md]] page (ingested 2026-04) recorded "all evaluations on internal NVIDIA datasets" as a limitation. DriveWAM's usage shows the dataset side has since been publicly released as a benchmark — that limitation is now partially superseded, though Alpamayo's AlpaSim closed-loop evaluation remains internal.

## Metrics

Open-loop trajectory imitation: **ADE** (Average Displacement Error) and **FDE** (Final Displacement Error) over 3-second and 4-second future horizons. All caveats from [[concepts/nuscenes-waymo-evals.md]] apply — displacement error against a logged human trajectory is not closed-loop driving quality, and there is no reactive simulation.

## Reported Results (DriveWAM's curated 1,000-clip test subset)

| Method | Params | ADE@3s ↓ | FDE@3s ↓ | ADE@4s ↓ | FDE@4s ↓ | Training data |
|---|---|---:|---:|---:|---:|---|
| VaVAM (released ckpt, ≤3s) | 1.3B | 2.31 | 4.32 | – | – | ~1,700 h OpenDV |
| Alpamayo-1.5 | 10B | 0.80 | 2.31 | 1.44 | 4.18 | ~80,000 h (incl. PhysicalAI-AV train) |
| **DriveWAM** | 5B + 8B | **0.47** | **1.35** | **0.83** | **2.47** | 100k curated clips (~556 h) |

DriveWAM roughly halves Alpamayo-1.5's ADE/FDE at both horizons while training on ~2 orders of magnitude less data — though on in-distribution curated clips, with a frozen 8B VLM in the loop.

## Test-Subset Caveat

There is currently **no standard public test protocol** in the wiki's sources: DriveWAM curates its own 1,000-clip test subset (VLM tagging with Qwen3-VL-8B, rule-weighted interest scores; rare-event + high-interest + 200 common-scene clips). Comparisons on this subset are the curating paper's own construction; VaVAM is further handicapped (checkpoint supports only 3s), and Alpamayo-1.5 is evaluated under a single-trajectory front-camera protocol chosen by DriveWAM. Treat cross-method numbers as indicative, not leaderboard-grade, until an official test protocol is adopted by multiple papers.

## The Standard Split, a Leakage-Free Subset, and a Reversal {#qwen-drive-reproduction}

[[sources/qwen-drive-1.0.md]] is the second wiki source to evaluate here and the first to use **the official 644-example split** rather than a self-curated subset. It also names a problem with that split and supplies a fix:

> "The standard 644-example split overlaps with publicly available training data by official construction. To ensure a fair comparison, we also report results on a leakage-free subset curated from held-out test clips."

That 700-frame leakage-free subset is the first contamination-controlled protocol on this benchmark. Each method predicts **six** trajectories per scene, scored by average ADE and minADE at 3 s and 5 s.

| Method | 644 Avg 3 s | 644 Avg 5 s | 644 min 3 s | 644 min 5 s | 700 Avg 3 s | 700 Avg 5 s | 700 min 3 s | 700 min 5 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Alpamayo-R1-10B | 0.37 | 1.13 | **0.16** | **0.48** | 0.41 | 1.22 | **0.18** | **0.51** |
| Alpamayo-1.5-10B | **0.35** | **1.05** | **0.16** | 0.50 | **0.36** | **1.06** | 0.17 | 0.49 |
| DriveWAM | 0.67 | – | 0.37 | – | 0.69 | – | 0.38 | – |
| SimWAM | 0.41 | – | 0.38 | – | 0.43 | – | 0.40 | – |
| Qwen-Drive-1.0-SFT w/o reasoning | 0.38 | 1.07 | 0.34 | 0.96 | 0.43 | 1.24 | 0.39 | 1.11 |
| Qwen-Drive-1.0-SFT w/ reasoning | 0.37 | 1.07 | 0.34 | 0.97 | 0.42 | 1.23 | 0.39 | 1.11 |
| Qwen-Drive-1.0-RL | 0.42 | 1.11 | 0.38 | 1.00 | 0.47 | 1.27 | 0.43 | 1.15 |

The leakage-free subset costs every method 0.04–0.06 m at 3 s — **contamination on the standard split is real but small**, which is a more useful finding than either number alone.

### The ordering reverses against DriveWAM's table {#reversal}

The table above this section, from [[sources/drivewam.md]]'s own curated 1,000-clip subset, has **DriveWAM at 0.47 ADE@3s and Alpamayo-1.5 at 0.80**. Reproduced on the official split: **DriveWAM 0.67, Alpamayo-1.5 0.35.**

| | DriveWAM's curated subset (its own paper) | Official 644 split (Qwen-Drive's reproduction) |
|---|---:|---:|
| DriveWAM | **0.47** | 0.67 |
| Alpamayo-1.5 | 0.80 | **0.35** |
| Ratio | DriveWAM 1.7× better | Alpamayo-1.5 1.9× better |

The splits differ, so this is not a contradiction of one number — it is **the failure this page's Test-Subset Caveat predicted, now realized.** The caveat stands and hardens: *no cross-paper ADE comparison on PhysicalAI-AV is currently safe unless both rows come from the same paper.* The wiki records both orderings and ranks neither.

Two candidate explanations, neither tested: DriveWAM curated for rare events and high-interest clips, which may suit its own inductive bias; and DriveWAM's protocol evaluated Alpamayo-1.5 under a single-trajectory front-camera setting of DriveWAM's choosing, while this table gives every method six candidates.

### Candidate diversity is now a reported axis

Because six trajectories are scored, the avg-ADE/minADE gap measures proposal spread:

- **Alpamayo-1.5**: 0.36 → 0.17 — wide, and the best of both.
- **DriveWAM**: 0.69 → 0.38 — broad coverage, weak typical accuracy.
- **SimWAM**: 0.43 → 0.40 — narrow.
- **Qwen-Drive-1.0**: 0.42 → 0.39 — narrow; the paper says so itself, "our candidates remain concentrated around similar motions."

Qwen-Drive attributes part of the gap to training scale — **~80,000 h and 3 M CoC traces for Alpamayo-1.5 against PhysicalAI-AV's 156 K clips ≈ 900 raw hours before sparse frame sampling.** That is the clearest statement available of what this dataset is and is not: large by public standards, two orders below a frontier AV program's internal corpus.

## Data Scaling on PAI-AV {#scaling}

[[sources/qwen-drive-1.0.md]] runs a second scaling study here, training its Planning Expert on PAI-AV alone at 0.17 M / 0.35 M / 0.69 M / 1.04 M / 1.38 M samples. On the standard split, 5 s Avg ADE **1.34 → 1.05** and Avg FDE **4.18 → 3.24**, monotonic, with the leakage-free subset following the same trend and **no sign of saturation.**

Together with [[sources/drivewam.md]]'s 4 k → 20 k → 100 k clip study (ADE@4s 1.01 → 0.94 → 0.83), this benchmark has now produced **two independent unsaturated real-log scaling curves from different architectures**, which is the role it is best suited for.

## AlpaSim: the closed-loop pairing this page asked for {#alpasim-answer}

This page's second open question was: *"Does performance on curated rare-event clips predict closed-loop behavior? No paper has yet paired PhysicalAI-AV with a reactive evaluation."*

**It has now been paired.** [[sources/qwen-drive-1.0.md]] evaluates on **AlpaSim** with **PAI-AV-NuRec v26.02** — the same logs reconstructed for closed-loop simulation with novel-view synthesis — across 916 scenarios and six methods. Full results and caveats are on [[concepts/alpasim-benchmark.md]]. The answer to the question as posed is **no**:

| Method | PAI-AV Avg ADE@3s (leakage-free) | AlpaSim at-fault score |
|---|---:|---:|
| Alpamayo-R1 | 0.41 | **0.58** |
| Alpamayo-1.5 | **0.36** | 0.45 |
| DriveWAM | 0.69 | 0.53 |
| Qwen-Drive-1.0-RL | 0.47 | 0.37 |
| SimWAM | 0.43 | 0.30 |

Open-loop displacement and closed-loop score disagree on nearly every pair: DriveWAM has the *worst* ADE and the second-best at-fault score (by barely moving — 35 % progress); Alpamayo-1.5 has the best ADE and is third. Every caveat from [[concepts/nuscenes-waymo-evals.md]] about displacement metrics applies, and this is the first time this page can point at a measurement rather than an argument.

**The note at the top of this page about AlpaSim being internal is now superseded**: an external group has run it, on public reconstructions, with a named version.

## As Pretraining Video for a NAVSIM Model {#pretraining-use}

[[sources/redrive.md]] is the first ingested paper to use this dataset as **unlabeled pretraining video** for a model evaluated elsewhere. An 80-hour subset of the front wide-angle camera joins nuScenes and navtrain in a V-JEPA-style masked-latent pretraining stage, sampled uniformly across the three sources.

Two details are reusable:
- **Camera alignment.** The front camera here is a 120° f-theta lens, far from nuPlan's pinhole front camera. ReDrive maps each target nuPlan pixel ray back through the per-clip f-theta calibration and resamples bilinearly, and drops clips whose field of view cannot cover the target. Anyone mixing this dataset with nuPlan-derived data faces the same mismatch.
- **What it bought.** The whole driving-domain pretraining stage is worth **+0.3 PDMS** at 4-frame clips and roughly +1.2 at 16-frame clips over the stock V-JEPA 2 checkpoint. The contribution of the PAI-AV subset by itself is not ablated, so there is no evidence here that cross-dataset pretraining video helps a NAVSIM planner.

This is a different use from the scaling studies above: there the dataset supplies labelled trajectories, here it supplies frames only.

## Role in the Wiki

This is the wiki's first large-scale *real-world data-scaling* benchmark: DriveWAM's 4k → 20k → 100k clip study (ADE@4s 1.01 → 0.94 → 0.83 with guidance) is run here, complementing NAVSIM (closed-loop non-reactive, small) and Bench2Drive (CARLA closed-loop, synthetic).

## Open Questions

*Wiki-wide open questions are collected in [[research-directions.md]].*

- Will other groups adopt the official 644-example split and the leakage-free subset? Two papers, two protocols, opposite orderings is where this benchmark currently stands.
- ~~Does performance on curated rare-event clips predict closed-loop behavior?~~ **Answered, negatively** - see [AlpaSim](#alpasim-answer) and [[concepts/alpasim-benchmark.md]]. The open version of the question is now whether the *ordering* holds when the reproduced methods are run by their own authors.
- Why does DriveWAM's ordering against Alpamayo-1.5 invert between subsets by a factor of ~3? Curation bias, candidate-count protocol, and ADE definition are all untested candidates.
- Does the leakage-free subset become the default? It costs 0.04-0.06 m uniformly, so adopting it is nearly free and removes a known contamination.
- Both scaling studies here stop while still improving. Nobody has found the knee on real logs.
