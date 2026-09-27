---
title: Evaluation Variance and Single-Run Reporting
type: concept
sources: [raw/papers/WA-JEPA_ Rethinking the Video JEPA Paradigm forWorld-Action Modeling in Autonomous Driving.md, raw/papers/CoWorld-VLA_ Thinking in a Multi-Expert World Model for Autonomous Driving.md, "raw/papers/SUV_ Future Scene Understanding as Video Generation for End-to-End Driving.md", raw/papers/Adaptive-WAM_ Quality-Guided Early-Exit Planningfrom Intermediate Video-Diffusion Features.md, raw/papers/How Can Driving World Models Do Counterfactual Prediction_.md, "raw/papers/[-0.5mm] GRAVA GRAVA_ Grounded Reasoning-to-Action Representation and Learning for Autonomous Driving.md", "raw/papers/Metis_ A Generalizable and Efficient World-Action Model for Autonomous Driving and Urban Navigation.md", raw/papers/SimWAM_ A Simple World Action Model for End-to-End Autonomous Driving.md, raw/papers/Drive-HWM_ Hierarchical World Models for Dynamic-Latent Guided Autonomous Driving.md, raw/papers/DA-WAM_ Decision-Aligned Future Latents for Driving World Models.md, raw/papers/BrainWAM_ Action-Space Coordination of Semantic Priors and Predictive Dynamics for Autonomous Driving.md, raw/papers/Auto-JEPA_ A Latent World Model of Continuous Intent for End-to-End Autonomous Driving.md, raw/papers/See Tomorrow, Act Today_ Foresight-Driven Autonomous Driving.md, raw/papers/DriveFuture_ Future-Aware Latent World Models for Autonomous Driving.md, raw/papers/Driving Intents Amplify Planning-Oriented Reinforcement Learning.md]
related: [sources/wa-jepa.md, sources/coworld-vla.md, sources/suv.md, sources/adaptive-wam.md, sources/driving-wm-counterfactuals.md, sources/grava.md, sources/metis.md, sources/simwam.md, sources/drive-hwm.md, sources/da-wam.md, sources/brainwam.md, sources/auto-jepa.md, sources/foresight.md, sources/drivefuture.md, sources/dial.md, concepts/navsim-benchmark.md, concepts/navhard-ood-evaluation.md, concepts/wam-attention-masks.md, concepts/inference-latency.md]
created: 2026-09-27
updated: 2026-09-27
confidence: medium
---

# Evaluation Variance and Single-Run Reporting

## What It Tracks

How much of a reported difference could be noise, and which kind of noise. Twenty-two source pages flag "single run, no seed variance". This page separates the **four sources of variance** that are routinely conflated and records every measurement the wiki has.

---

## Four Different Things Called "Variance"

| Source | What varies | Measured in the wiki? | Size |
|---|---|---|---|
| **1. Evaluator determinism** | Nothing. The same checkpoint is re-scored | [[sources/suv.md]]: 6 repeats, identical to 2 decimals | 0 |
| **2. Sampler seed** | Diffusion/flow noise with fixed weights | [[sources/wa-jepa.md]], [[sources/coworld-vla.md]] | **0.013–0.053** EPDMS std |
| **3. Scene sampling** | Which scenes are in the test split | Computable from [[sources/suv.md]] | SE of the mean ≈ **0.16** on navtest |
| **4. Training seed** | Initialization, data order, the whole run | **Never, on any driving benchmark** | Unknown |

**The number that licenses ablation claims is #4, and nobody has measured it.** [[sources/coworld-vla.md]] is the first paper to say so: variance over independently trained models "was not conducted because of the high training cost".

---

## The Measurements

### Sampler seed (fixed weights)

| Paper | Sampler | Seeds | Metric | Mean | Std |
|---|---|---:|---|---:|---:|
| [[sources/wa-jepa.md]] | 12-step flow | 10 | EPDMS | 91.70 | **0.053** (range 0.18) |
| [[sources/coworld-vla.md]] | 10-step rectified flow | 6 | PDMS / EPDMS | 89.98 / 89.99 | **0.013 / 0.014** |

### Scene-level dispersion

[[sources/suv.md]] reports per-scene standard deviations of **17.84 PDMS / 18.21 EPDMS over 12,146 navtest scenes**. The standard error of a mean is therefore about 0.16.

- **For a paired comparison** (two models on the same scenes), the relevant SE is smaller and depends on how correlated the two models' per-scene scores are. No paper reports it.
- **navhard is much noisier.** There are only 244 or 450 Stage-1 scenes, depending on the paper; see [[concepts/navhard-ood-evaluation.md]]. If navhard's per-scene SD is similar, the SE is roughly **0.9–1.2 points**. That is the scale of several recent "navhard-only" mechanism effects (+2.2 Metis mask, +2.4 Metis video prior), though smaller than SUV's +4.1.

### Selection effects that inflate single numbers

- [[sources/adaptive-wam.md]] reports the **validation-best checkpoint per seed, aggregated over ten seeds**. That is a selection procedure, not a variance estimate, and it is optimistic relative to a single run.
- [[sources/suv.md]] selects checkpoints on the NAVSIM validation PDM score.
- [[sources/dial.md]] compares methods at their best held-out checkpoint, even though several later collapse.
- Best-of-N oracle rows ([[concepts/best-of-n.md]]) are the extreme case.

### Outside NAVSIM

[[sources/driving-wm-counterfactuals.md]] reports means over five seeds with a **±maximum deviation** (not a standard deviation). This is the only multi-seed generative evaluation in the wiki.

---

## What This Means for Reading the Wiki

1. **Sampler noise is not the binding constraint.** At 0.01–0.05 it is 2–3 orders of magnitude below the **+2.0 to +3.8** EPDMS shift from the [evaluator correction](navsim-benchmark.md#evaluator-drift-this-table-mixes-two-protocols). Protocol drift matters far more than seeds for cross-paper comparison.
2. **Ablation deltas under ~0.5 should be read as "no detectable effect".** Examples: SimWAM's mask ablation (0.2 spread), SUV's per-stream access on navtest (≤0.2), Qwen-Drive's reasoning effect (+0.4), GRAVA's +0.18 margin for the best purely autoregressive model. This is not because sampler noise is that large, but because training-seed variance is unmeasured and plausibly several times larger.
3. **navhard effects need to be several points to be interpretable.** The recent pattern of "invisible on navtest, large on navhard" ([[concepts/wam-attention-masks.md]]) is the most important open finding that currently rests on single runs over a few hundred scenes.
4. **Headline margins of 0.1–0.7 between top methods are within unmeasured training noise.** Examples: SUV vs Metis top-6, GRAVA vs Curious-VLA, SUV vs Poutine on WOD-E2E (+0.03 RFS).

---

## Open Questions

- **What is training-seed variance for a NAVSIM planner?** Three seeds of one cheap model (a DiffusionDrive-class planner trains in hours) would bound it and settle rule 2 above.
- **What is paired per-scene variance between two published checkpoints?** It needs only released checkpoints and the evaluator, no training.
- **What is navhard's per-scene SD?** Any paper that reports per-scene navhard scores would let every navhard effect in the wiki be given an error bar.
