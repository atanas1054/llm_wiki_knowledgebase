---
title: AlpaSim Closed-Loop Benchmark
type: concept
sources: [raw/papers/Qwen-Drive-1.0_ An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving.md, raw/papers/Alpamayo-R1_ Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail.md]
related: [concepts/physicalai-av-benchmark.md, concepts/navsim-benchmark.md, concepts/hugsim-benchmark.md, concepts/bench2drive.md, concepts/navhard-ood-evaluation.md, concepts/world-model-for-ad.md, concepts/nuscenes-waymo-evals.md, sources/qwen-drive-1.0.md, sources/alpamayo-r1.md, sources/drivewam.md, sources/simwam.md, research-directions.md]
created: 2026-09-18
updated: 2026-09-30
confidence: medium
---

## What It Is

A **closed-loop** driving simulator built on reconstructed real-world logs, introduced alongside NVIDIA's Alpamayo line and used publicly for the first time in this wiki by [[sources/qwen-drive-1.0.md]]. The planner is re-queried as its own actions change subsequent observations, and **PAI-AV-NuRec** supplies novel-view synthesis for the frames the recorded log never contained:

> "AlpaSim evaluates closed-loop behavior under compounding errors. We use PAI-AV-NuRec version 26.02 to simulate 916 scenarios with novel views as the ego vehicle deviates from recorded logs."

This closes a gap the wiki has carried since the [[sources/alpamayo-r1.md]] ingest, which recorded "all evaluations on internal NVIDIA datasets" as a limitation, and which [[concepts/physicalai-av-benchmark.md]] partially retired when the dataset was released while noting that "Alpamayo's AlpaSim closed-loop evaluation remains internal." **It is no longer internal**: an external group has now run it on six methods, including two it did not author.

## Why It Matters Here

AlpaSim is the only evaluation in this wiki that is simultaneously **closed-loop, reactive to ego deviation, and built on real logs**. The other closed-loop benchmarks are synthetic ([[concepts/bench2drive.md]], CARLA) or reconstruction-based on smaller scenario sets ([[concepts/hugsim-benchmark.md]]). [[concepts/navsim-benchmark.md]] is single-shot and non-reactive by construction, and [[concepts/navhard-ood-evaluation.md]] adds distribution shift but not compounding error.

That makes it the natural place to ask the wiki's fourth open thread — *do PDMS gains transfer to closed-loop?* — and the answer it returns is unflattering.

## Metrics

| Metric | Direction | What it measures |
|---|---|---|
| Close encounter rate, **all** | ↓ | fraction of scenarios with a close-proximity event of any cause |
| Close encounter rate, **at-fault** | ↓ | same, excluding events not caused by the ego vehicle |
| Off-road rate | ↓ | roadway departures |
| Progress | ↑ | scenario completion |
| AlpaSim score, **all** / **at-fault** | ↑ | average distance travelled between events |

**The score is a ratio, and that is the whole difficulty of reading this benchmark.** Distance travelled divided by event count rewards a policy that travels little as strongly as one that avoids events — and the at-fault variant additionally forgives every collision another agent causes. A vehicle that stops and stays stopped scores well on at-fault metrics while being struck from behind repeatedly.

## Results (916 scenarios, PAI-AV-NuRec v26.02)

From [[sources/qwen-drive-1.0.md]] Table 7. **All comparison methods are reproduced by that paper under one setting**; parameter counts exclude LLM token embeddings.

| Method | Params | CE all ↓ | CE at-fault ↓ | Off-road ↓ | Progress ↑ | Score all ↑ | Score at-fault ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Alpamayo-R1** | 9.8 B | **19.0** | 6.0 | 17.0 | **67.0** | **0.36** | **0.58** |
| Alpamayo-1.5 | 9.8 B | 37.0 | 11.0 | 16.0 | 59.0 | 0.23 | 0.45 |
| DriveWAM | 15.0 B | 56.0 | **5.0** | **8.0** | 35.0 | 0.10 | 0.53 |
| SimWAM | 6.0 B | 35.0 | 22.0 | 19.0 | 62.0 | 0.22 | 0.30 |
| Qwen-Drive-1.0-SFT w/ reasoning | 5.0 B | 38.0 | 12.0 | 24.0 | 54.0 | 0.16 | 0.27 |
| Qwen-Drive-1.0-RL | 5.0 B | 41.0 | 11.0 | 12.0 | 48.0 | 0.16 | 0.37 |

### The ordering inverts against NAVSIM {#inversion}

| Method | NAVSIM-v1 PDMS | AlpaSim at-fault score | AlpaSim progress |
|---|---:|---:|---:|
| [[sources/simwam.md]] | **91.5** — highest world-action model in the wiki | **0.30** — last | 62 % |
| [[sources/qwen-drive-1.0.md]] (RL) | 90.7 | 0.37 | 48 % |
| [[sources/drivewam.md]] | 90.1 | 0.53 | **35 %** |
| Alpamayo-1.5 | not reported | 0.45 | 59 % |
| [[sources/alpamayo-r1.md]] | not reported | **0.58** | **67 %** |

**The two best closed-loop methods report no NAVSIM result at all, and the best NAVSIM method is last.** This is the strongest single piece of evidence in the wiki that the pseudo-closed-loop ladder and interactive driving quality are different quantities — stronger than [[sources/geowam.md]]'s navtest/navhard asymmetry, because it crosses from non-reactive to genuinely reactive rather than between two variants of one scorer.

Three caveats before over-reading it:

1. **Reproductions, not self-reports.** SimWAM and DriveWAM were run in AlpaSim by a third party under an input protocol that party chose. Neither paper designed for this benchmark, and neither has responded.
2. **The score metric flatters degeneracy.** See the two behavioural diagnoses below — DriveWAM's 0.53 and SimWAM's 0.30 are not simply "better" and "worse."
3. **Single run per method, no seeds, no confidence intervals**, on 916 scenarios with percentage-point granularity.

### Two named behavioural failure modes {#behaviour}

Qwen-Drive's analysis is the useful part of the table and generalizes beyond it:

**Degenerate caution (DriveWAM).** At-fault score 0.53 with 35 % progress: "we find that it often remains stationary or advances only briefly. This behavior reduces ego-at-fault and off-road events, which can raise the at-fault score because the metric divides the traveled distance by the corresponding event count. However, a nearly stationary ego vehicle remains susceptible to interactions caused by following traffic." Its all-event close-encounter rate is **56 %**, the worst in the table, and its all-event score **0.10**, also the worst.

**Degenerate aggression (SimWAM).** 62 % progress with 22 % at-fault encounters and 19 % off-road: "its driving policy is considerably more aggressive, frequently accelerating forward while failing to decelerate sufficiently for preceding vehicles or obstacles."

**The rule this page adopts**: an AlpaSim score is uninterpretable without progress and both close-encounter rates beside it. Report all six columns or none.

### RL moves closed-loop behaviour opposite to how it moves NAVSIM {#rl-direction}

Qwen-Drive's Stage-4 RL — trained on NAVSIM PDMS, WOD-E2E RFS and PAI-AV displacement, none of them closed-loop — changes AlpaSim behaviour substantially:

| | SFT | RL | Δ |
|---|---:|---:|---|
| Off-road | 24.0 | **12.0** | **halved** |
| At-fault score | 0.27 | **0.37** | +0.10 |
| At-fault CE | 12.0 | 11.0 | −1 |
| Progress | 54.0 | **48.0** | **−6** |
| All-event CE | 38.0 | 41.0 | +3 |

On NAVSIM the same update *raised* ego progress 82.4 → 84.8, and the paper describes it as "particularly effective at reducing conservative, low-progress behavior." **In closed loop it produced the opposite: safer and more conservative.** Nothing in the paper reconciles these; the wiki records it as a concrete case of one policy change scoring progress-positive under a non-reactive protocol and progress-negative under a reactive one.

The transfer itself is worth noting on the positive side: **rewards computed entirely on open-loop and pseudo-closed-loop metrics halved the closed-loop off-road rate.** Reward design does reach closed-loop behaviour; it just does not reach it in the direction the source metric suggests.

## Open Questions

*Wiki-wide open questions are collected in [[research-directions.md]].*

1. **Is the ordering inversion real or a protocol artifact?** The decisive test is a self-reported AlpaSim number from SimWAM's or DriveWAM's own authors. Until then this is one group's reproduction of two methods that were not designed for it.
2. **What input cadence does closed-loop driving want?** Qwen-Drive's own hypothesis for its deficit is observation sampling — Alpamayo-1.5's dense 0.4 s history against its own 4 frames spanning 1.5 s — and AlpaSim replans over short intervals. Untested, and a one-variable experiment in its released codebase.
3. **Does anything above 92 PDMS survive here?** Nothing in the wiki's NAVSIM top eight has been run in any reactive simulator. [[sources/drivesuprim.md]], [[sources/clear.md]], [[sources/da-wam.md]] and [[sources/wcog-vla.md]] are all unmeasured under compounding error. *(Lint 2026-09-30: one entry above 92 now has a closed-loop number, though not in AlpaSim. [[sources/mm-future.md]] (93.4 PDMS) scores 32.3 HD-Score on HUGSIM's 436 scenarios, below WA-JEPA's 44.6 (91.8 PDMS) and PhysWAM's 35.5: a second case of NAVSIM order not carrying over.)*
4. **How does AlpaSim relate to HUGSIM?** Both reconstruct real logs and re-query the planner; neither has been run alongside the other by any paper. If the two agree on a ranking, the wiki gains a usable closed-loop axis; if they disagree, it gains another protocol-drift problem.
5. **Is version 26.02 stable?** [[concepts/hugsim-benchmark.md]] records how quickly a versioned reconstruction benchmark makes earlier numbers incomparable. Qwen-Drive names its version, which is the right practice; the wiki should watch for the second paper to report a different one.
