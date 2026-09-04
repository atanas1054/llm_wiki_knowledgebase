---
title: "Adaptive-WAM: Quality-Guided Early-Exit Planning from Intermediate Video-Diffusion Features"
type: source-summary
sources: [raw/papers/Adaptive-WAM_ Quality-Guided Early-Exit Planningfrom Intermediate Video-Diffusion Features.md]
related: [sources/geoworldad.md, concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/foundation-backbones-for-ad.md, concepts/selection-based-planning.md, concepts/adaptive-routing.md, concepts/nuscenes-waymo-evals.md, concepts/rl-for-ad.md, sources/drivelaw.md, sources/brainwam.md, sources/foresight.md, sources/simwam.md, sources/driveva.md, sources/drivewam.md, sources/recogdrive.md, sources/auto-jepa.md, sources/da-wam.md, sources/wa-jepa.md, sources/clear.md, sources/drivesuprim.md, sources/latent-wam.md, sources/drivevla-w0.md, sources/epona.md, sources/policy-world-model.md]
created: 2026-09-04
updated: 2026-09-04
confidence: high
---

**Paper**: Adaptive-WAM: Quality-Guided Early-Exit Planning from Intermediate Video-Diffusion Features
**Authors**: Sining Ang, Yuguang Yang, Yan Wang
**Orgs**: Institute for AI Industry Research (AIR), Tsinghua University + University of Science and Technology of China + Beihang University
**arXiv**: 2608.06008v1
**Code**: announced, not yet released

---

## Summary

This paper asks the question the wiki has been circling for three ingests — *how much of a video diffusion model must you actually run to plan?* — and it is the first to answer it by **separating two axes everyone else has been varying together**:

- **The video-noise level** (which diffusion timestep the backbone is conditioned on): five indices spanning a 40-step schedule change the score by **at most 0.15 PDMS**.
- **The DiT depth** (how many transformer blocks you evaluate): the spread across six readout depths is **5.85 PDMS after imitation and 4.80 after RL**, and **block 15 of 30 beats the full-depth exit by 4.80**.

That second result is new to this wiki. Nobody has previously reported *where in the network* to read a video prior, and it turns out to matter roughly forty times more than the noise level everyone has been ablating.

Adaptive-WAM builds the obvious system on top: attach six independent [[sources/recogdrive.md]]-style 5-step trajectory diffusion heads to Wan2.2-TI2V-5B at blocks {5, 9, 15, 18, 22, 30}, decode one trajectory per exit, and let a DINOv2-Small quality scorer terminate execution once a decoded plan looks good enough. **90.8 PDMS on NAVSIM v1 / 89.9 EPDMS on v2 at 170 ms average end-to-end latency on an A100** — the fastest world-action model in the wiki by a factor of three, against a full 40-step video rollout that costs **13.22 s** on the same hardware.

A separate fixed-exit variant with 64 proposals reaches 92.6 PDMS, but it is a different model with privileged pseudo-expert supervision and belongs in the [selection-based](../concepts/selection-based-planning.md) family, not next to single-trajectory scores.

---

## The Motivating Analysis

![[paradigm_comparison.png|Predetermined vs adaptive WAM interfaces: video-backbone WAMs, multimodal WAMs, and Adaptive-WAM's quality-routed exits]]

**Figure 1**: (a) Video-backbone WAMs follow a predetermined frame/action generation path; (b) multimodal WAMs generate visual and action streams along a predetermined path; (c) Adaptive-WAM decodes one trajectory per attempted exit and routes by predicted quality. Future-video prediction supervises training but is not required by the deployed planner.

### Video noise level is nearly irrelevant (Table 13)

All rows are single conditional forwards at the stated sampling index of a 40-step schedule.

| Configuration | idx 1 | idx 9 | idx 17 | idx 25 | idx 32 | **Range** |
|---|---:|---:|---:|---:|---:|---:|
| Block 15, single trajectory | 86.44 | 86.56 | **86.57** | 86.55 | 86.50 | **0.13** |
| Block 18, single trajectory | 84.02 | **84.14** | 83.99 | 84.12 | 84.01 | **0.15** |
| Fixed B15, 64 proposals | 92.01 | **92.12** | 92.11 | 92.05 | 92.07 | 0.11 |
| Fixed B18, 64 proposals | 92.45 | 92.55 | **92.59** | 92.45 | 92.43 | 0.14 |

The paper is careful about scope: *"These results support robustness to the tested noise levels; they do not claim invariance to every possible video timestep or scheduler."* Index 17 is fixed for everything downstream.

### DiT depth matters, and the last block is the worst one (Tables 1, 14)

| Block | 5 | 9 | **15** | 18 | 22 | 30 (full) |
|---|---:|---:|---:|---:|---:|---:|
| Imitation learning | 81.94 | 83.60 | **86.56** | 84.14 | 83.62 | 80.71 |
| + planner-only RL | 86.02 | 87.56 | **90.62** | 88.92 | 87.42 | 85.82 |
| RL gain | 4.08 | 3.96 | 4.06 | 4.78 | 3.80 | 5.11 |

All exits share architecture, optimizer, batch size, epoch count, and head capacity, so **differences are attributable to backbone depth rather than head capacity** — an unusually clean control.

Two things stand out. The **mid-network exit is best at both stages**, and the **full-depth exit is worst**, by 5.85 PDMS after imitation and 4.80 after RL. And **planner RL lifts every exit by 3.8–5.1 points**, without changing the depth ordering.

### But depth ordering is not scene-wise dominance (Tables 15–17)

Post-RL Jaccard overlap of the per-exit sets of scenes scoring ≥ 90, averaged over ten aligned runs:

| Jaccard | B5 | B9 | B15 | B18 | B22 | B30 |
|---|---:|---:|---:|---:|---:|---:|
| B5 | 1.00 | 0.80 | 0.81 | 0.77 | 0.69 | 0.70 |
| B9 | 0.80 | 1.00 | **0.82** | 0.77 | 0.70 | 0.69 |
| B15 | 0.81 | **0.82** | 1.00 | 0.79 | 0.74 | 0.73 |
| B18 | 0.77 | 0.77 | 0.79 | 1.00 | 0.78 | 0.74 |
| B22 | **0.69** | 0.70 | 0.74 | 0.78 | 1.00 | 0.78 |
| B30 | 0.70 | **0.69** | 0.73 | 0.74 | 0.78 | 1.00 |

Off-diagonal overlap runs 0.69–0.82 — substantial but incomplete. The directional large-advantage counts (scenes where one exit beats another by ≥ 50 points, mean over ten paired runs) make the same point sharply: post-RL, **block 15 beats block 30 on 598.6 scenes, but block 30 beats block 15 on 422.4**. A globally dominant exit is not a scene-wise dominant one, which is the whole justification for routing rather than just picking block 15.

The paper also reports that the maximum cell-wise standard deviation across these matrices **falls from 182.82 pre-RL to 84.94 post-RL** — planner RL stabilizes the depth-wise behavior as well as improving it.

---

## Method

![[main_architecture.png|Adaptive-WAM: Wan2.2 backbone with six intermediate trajectory exits, DINOv2-Small quality scorer, and threshold-based early exit]]

**Figure 4**: Wan2.2 retains video supervision while six intermediate blocks feed independent ReCogDrive-style trajectory heads. At inference, one trajectory is decoded per attempted exit; the DINOv2-Small scorer either returns the best accumulated trajectory or continues from the cached hidden state. The future-scene branch denotes training supervision rather than the deployed path.

### One conditional forward, no rollout

Given observation $o=(I, S_{\mathrm{ego}}, L_{\mathrm{nav}})$ and a programmatic text description $d(o)$, the hidden state at block $\ell$ is

$$h_{\ell}=F_{1:\ell}\bigl(I,\,d(o);\,s^{\star}\bigr),\qquad s^{\star}=17$$

This is a **single video-feature forward** at a fixed noise index — no iterative denoising loop, no classifier-free-guidance unconditional branch, no VAE video decode. It is separate from the five DDIM steps each trajectory head runs.

Each exit has a projection $P_\ell$ and an independent diffusion head $G_\ell$:

$$\tau_{\ell}=G_{\ell}\bigl(P_{\ell}(h_{\ell}),\,S_{\mathrm{ego}},\,L_{\mathrm{nav}}\bigr)$$

Heads share architecture and budget and **do not exchange features or predictions**.

### The quality controller

A fine-tuned DINOv2-Small encoder takes only the current front image and a candidate trajectory (no ego state, no navigation command), embeds the flattened 8×3 poses with an MLP, and emits six independent two-layer MLP heads for $\mathcal{R}=\{\mathrm{NC, DAC, DDC, TTC, EP, Comf}\}$. Predictions are composed through the normalized PDMS formula, $Q(\hat{\mathbf r})=100\,\Gamma(\hat{\mathbf r})$.

The controller maintains the best trajectory *accumulated across attempted exits* and terminates at the first $j$ with $\hat q_j \ge \eta$:

$$\hat{\tau}_{j}=\operatorname*{arg\,max}_{\tau_{\ell_m}\in\mathcal{A}_j} Q(\hat{\mathbf r}_{\ell_m}),\qquad \hat q_j=\max_{m\le j} Q(\hat{\mathbf r}_{\ell_m})$$

After a rejected exit only the *unevaluated* blocks execute; hidden states and cached scores are reused.

**The design choice worth recording** is why the scorer predicts components rather than a ranking. More than **95% of diagnostic scenes contain candidate groups that are jointly perfect, jointly zero, or tied at the top**, so a rank loss would be fitting noise. The scorer uses equal-weight soft-label BCE on un-binarized evaluator components and acts as an *exit-quality verifier*, not a total-order ranker. This is the clearest statement in the wiki of a problem several selection-based papers have worked around without naming — see [[concepts/selection-based-planning.md]].

### Training

1. **Video-domain adaptation + imitation.** Nine-frame clips (anchor + 8 future frames at 2 Hz), resized 1600×900 → 1280×704. Text conditions are generated programmatically from deployment-available attributes (map metadata, discretized ego speed, a maneuver derived from *past* ego poses only, traffic density). Wan is adapted with **LoRA**; projections and heads train fully:

   $$\mathcal{L}_{\mathrm{actor}}=\lambda_{\mathrm{vid}}\mathcal{L}_{\mathrm{vid}}+\sum_{\ell\in\mathcal{E}}\lambda_{\ell}\mathcal{L}_{\mathrm{traj}}^{\ell}$$

   Video and trajectory objectives share the same $s^\star=17$ forward — no second backbone pass.

2. **Scorer training**, alternating with the actor, on **stop-gradient** trajectories so $\nabla_{\theta_{\mathrm{Wan}},\theta_G}\mathcal{L}_{\mathrm{score}}=0$. The scorer cannot shift the actor's trajectory distribution.

3. **Planner-only DiffGRPO.** Wan backbone and scorer frozen; each head refined with the full five-step denoising chain treated as one action, rewarded by the NAVSIM evaluator score. No routing decision propagates to the actor.

Reported layer-wise statistics take the **validation-best checkpoint per seed, aggregated over ten seeds**.

---

## Results

### Table 2 — NAVSIM v1 navtest (PDMS)

| Method | Input | NC | DAC | TTC | Comf. | EP | PDMS ↑ |
|---|:-:|---:|---:|---:|---:|---:|---:|
| *Traditional end-to-end* | | | | | | | |
| VADv2-𝒱₈₁₉₂ | C | 97.2 | 89.1 | 91.6 | 100 | 76.0 | 80.9 |
| UniAD | C | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| TransFuser | CL | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| PARA-Drive | C | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| ReCogDrive-IL | C | 98.1 | 94.7 | 94.2 | 100 | 80.9 | 86.5 |
| DiffusionDrive | CL | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| *World-model planners* | | | | | | | |
| LAW | C | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 |
| Epona | C | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 |
| DriveVLA-W0 | C | 98.4 | 95.3 | 95.2 | 100 | 80.9 | 87.2 |
| PWM | C | 98.6 | 95.9 | 95.4 | 100 | 81.8 | 88.1 |
| WoTE | CL | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| DriveVA | C | 99.2 | 97.5 | **98.7** | 100 | 83.5 | 90.5 |
| **Adaptive-WAM (single trajectory)** | C | 98.6 | 97.9 | 95.6 | 100 | 85.1 | **90.8** |
| **Adaptive-WAM (fixed B22, 64 prop.)** | C | **99.8** | **98.3** | 98.3 | 100 | **86.6** | **92.6** |

Front camera only, no LiDAR. **EP 85.1 is the standout sub-score** — above DriveVA's 83.5 and every other world-model entry in the table.

**A useful side-effect**: this table supplies DriveVA's per-metric sub-scores, which [[sources/driveva.md]] records as unavailable because its own source was truncated. It also disambiguates DriveVA's headline number — the paper states its 92.6 exceeds *"DriveVA's mixed-data headline result by 1.7 points and its NAVSIM-only result by 2.1"*, i.e. **DriveVA is 90.9 with mixed data and 90.5 on NAVSIM alone**, and Adaptive-WAM compares against the NAVSIM-only figure. That is the right choice and worth crediting.

Table 2's caption states *"baselines follow DriveVA"*, which explains both the composition of the table and the reappearance of DriveVLA-W0 at 87.2 — the same non-headline value now propagating through [[sources/drivelaw.md]], [[sources/brainwam.md]], and this paper.

### Table 3 — NAVSIM v2 navtest (EPDMS)

| Method | Input | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS ↑ |
|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Human Agent** | – | 100 | 100 | 99.8 | 100 | 87.4 | 100 | 100 | 98.1 | **90.1** | **90.3** |
| DiffusionDrive | CL | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| ReCogDrive | C | 98.3 | 95.2 | 99.5 | 99.8 | 87.1 | 97.5 | 96.6 | 98.3 | 86.5 | 83.6 |
| Epona | C | 97.1 | 95.7 | 99.3 | 99.7 | **88.6** | 96.3 | **97.0** | 98.0 | 67.8 | 85.1 |
| DriveVLA-W0 | C | **98.5** | **99.1** | 98.0 | 99.7 | 86.4 | **98.1** | 93.2 | 97.9 | 58.9 | 86.1 |
| **Adaptive-WAM** | C | **98.5** | 98.0 | **99.5** | **99.8** | 87.6 | 97.4 | 95.4 | 98.2 | 75.5 | **89.9** |

**89.9 EPDMS sits 0.4 below the human agent's 90.3** in the same table — the closest approach to the human reference this wiki has recorded on v2. EC 75.5 is the weak sub-score, better than DriveVLA-W0's 58.9 and Epona's 67.8 but well below the human 90.1 and below [[sources/brainwam.md]]'s 85.8.

Only five baselines, and the table mixes evaluator conventions — see Limitations.

### Table 5 / 23 — The adaptive trade-off (A100, batch 1)

| Policy | PDMS ↑ | Exit by B15 | Latency ↓ |
|---|---:|---:|---:|
| Fixed B15 | 90.62 | 100.0% | 190 ms |
| Adaptive η=70 | 88.49 | 98.8% | **112 ms** |
| Adaptive η=80 | 90.64 | 95.2% | 143 ms |
| **Adaptive η=90** | **90.79** | 94.1% | 170 ms |
| Adaptive η=95 | 90.75 | 65.9% | 284 ms |
| Full path (B30) | 85.82 | 0.0% | 320 ms |

At the selected η=90, routing beats the strongest fixed exit by **+0.17 PDMS at 10% lower latency**, with 94% of scenes terminating within the first three exits. The η=70 row is the useful control: aggressive early exit is fastest but **loses 2.13 points**, so the gain is genuinely from *conditional* allocation rather than from being shallow.

### Efficiency in context (Appendix K)

| Path | Cost |
|---|---:|
| Adaptive planner (η=90), end-to-end | **170 ms** |
| Fixed block-15 planner | 190 ms |
| Fixed full-depth planner (+1 VAE decode) | 320 ms |
| VAE image encoding alone | ~50 ms |
| Mean conditional DiT per denoising step | 149.40 ms |
| Mean unconditional DiT per denoising step | 147.80 ms |
| **Full 40-step CFG video generation** (80 DiT forwards) | **13.22 s** |
| — of which: denoising loop | 12.05 s |
| — VAE video decoding | 0.90 s |
| Peak allocated memory (video path) | 31.19 GiB |

**This is the wiki's first full decomposition of what a driving video generator actually costs**, and it reframes the efficiency debate. The gap between planning from an intermediate feature (170 ms) and synthesizing the future it encodes (13.22 s) is a factor of 78.

### Zero-shot NAVSIM → nuScenes (Tables 4, 25, 26)

| Method | FT | L2 1s | L2 2s | L2 3s | **L2 Avg ↓** | Col 1s | Col 2s | Col 3s | **Col Avg ↓** |
|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|
| UniAD | ✓ | 0.48 | 0.96 | 1.65 | 1.03 | 0.05 | 0.17 | 0.71 | 0.31 |
| GenAD | ✓ | 0.36 | 0.83 | 1.55 | 0.91 | 0.06 | 0.23 | 1.00 | 0.43 |
| Epona | ✓ | 0.61 | 1.17 | 1.98 | 1.25 | 0.01 | 0.22 | 0.85 | 0.36 |
| DriveVLA-W0 | – | 0.43 | 1.26 | 2.60 | 1.43 | 0.22 | 0.66 | 1.42 | 0.77 |
| PWM | – | 2.06 | 3.91 | 6.00 | 3.99 | 0.12 | 0.15 | 0.86 | 0.36 |
| DriveVA | – | **0.33** | 0.76 | **1.43** | **0.84** | **0.00** | **0.07** | **0.12** | **0.06** |
| **Adaptive-WAM** | – | 0.35 | **0.71** | 1.58 | 0.88 | **0.00** | 0.09 | 0.15 | 0.08 |

Consistent with the wiki's existing finding that NAVSIM-trained WAMs transfer to nuScenes as *safety* priors — every zero-shot entry except DriveVLA-W0 and PWM beats the fine-tuned baselines on collision rate by roughly an order of magnitude. Adaptive-WAM narrowly trails DriveVA on both averages, but DriveVA runs the full backbone and generates future images.

---

## Ablations

### Table 21 — Wan adaptation strategy

| Wan training | Single trajectory | Fixed B22, 64 prop. |
|---|---:|---:|
| **Frozen** | **84.20** | 89.91 |
| Separate LoRA + cached features | 84.95 | 90.80 |
| **Joint LoRA** | **90.62** | **92.59** |
| Full fine-tuning | 90.64 | 92.54 |

**This is the most consequential ablation in the paper for the rest of the wiki.** A frozen video backbone scores 84.20; jointly LoRA-adapting it scores 90.62 — a **6.42-point gap**. Fine-tuning the backbone *separately* and then caching its features recovers only 0.75 of that.

Two wiki designs sit squarely on the losing side of this comparison: [[sources/foresight.md]] freezes Epona entirely and uses it as the primary encoder, and [[sources/drivelaw.md]] caches Video-DiT block features for a separately-optimized planner (though it also updates both modules in stage 3, a point its own page flags as unreconciled). Adaptive-WAM does not test their architectures, so this is not a refutation — but it is a strong prior that *the video prior must be adapted jointly with the action objective*, and it is measured with everything else held fixed.

**Full fine-tuning adds 0.02 over LoRA**, making this the wiki's third data point on the LoRA question, agreeing with [[sources/da-wam.md]] (LoRA > full FT by 0.36) against [[sources/latent-wam.md]] (LoRA collapsed geometric distillation, 89.3 → 68.5).

### Table 22 — Video-DiT features vs. static visual backbones

| Visual backbone | Single trajectory | Fixed exit, 64 prop. |
|---|---:|---:|
| ViT-Small | 83.91 | 92.17 |
| ViT-Base | 85.62 | 92.21 |
| ViT-Large | 88.88 | 92.31 |
| **Wan intermediate features** | **90.62** | **92.59** |

Wan beats ViT-L by **1.74** in the single-trajectory setting and 6.71 over ViT-S. The gap narrows to 0.28 with 64 proposals, because a scorer choosing among many candidates compensates for a weaker representation. **That collapse is itself informative**: multi-proposal scoring masks representation quality, which is a caution for reading any selection-based leaderboard as evidence about encoders.

### Table 20 — Scorer backbone

| Scorer backbone | Selected-trajectory score |
|---|---:|
| Wan-B22 | **92.62** |
| DINO-Small | **92.59** |
| Wan-B18 | 92.59 |
| Wan-B30 | 92.57 |
| ResNet-50 | 92.55 |
| DINO-Base | 92.54 |
| Wan-B9 | 92.44 |
| Wan-B5 | 92.24 |
| ResNet-34 | 92.19 |
| Wan-B15 | 92.11 |
| ViT-Base | 91.20 |
| ViT-Small | 91.17 |

The best Wan exit beats DINOv2-Small by **0.03** while requiring a full world-model forward at every attempted exit. DINO-Small is used. Note the ordering here is unrelated to the *planning* depth ordering — Wan-B15 is the best planning exit and the second-*worst* Wan scorer, which suggests the two roles want different features.

### Table 18 — Scorer reliability (12,146 scenes)

| Diagnostic | Rate |
|---|---:|
| Exact top-score selection | 91.2% |
| Selection within 5 points | 94.4% |
| Failure with ≥ 20-point gap | 0.57% |
| Failure with ≥ 50-point gap | **0.42%** |

The paper's own framing is appropriately cautious: *"passing the quality threshold is not a formal safety certificate."* 0.42% of 12,146 scenes is **51 events where the controller accepts a trajectory at least 50 points worse than an available near-perfect one.**

---

## Qualitative

![[supp_early_vs_deep.png|Layer-wise trajectory overlays showing cross-depth complementarity and score ties]]

**Figure 5**: Overlays of the six fixed-exit trajectories, green for high-scoring and red for erroneous. The cases illustrate both cross-depth complementarity and the metric ties that motivate a verifier rather than a strict ranker.

**Figures 2 and 3 have captions in the source but no image files** — the Jaccard heatmap and pairwise-advantage matrices exist only as Appendix G tables, which is where the numbers above come from.

---

## Implementation Summary

| Item | Setting |
|---|---|
| Backbone | Wan2.2-TI2V-5B, LoRA-adapted, single conditional forward at noise index 17/40 |
| Exits | Blocks {5, 9, 15, 18, 22, 30}; independent ReCogDrive-style 5-step trajectory DiTs |
| Scorer | Fine-tuned DINOv2-Small + trajectory MLP → six component heads; soft-label BCE, no binarization, no rank loss |
| Routing | Accumulate best-so-far; terminate at first $\hat q_j \ge \eta$; **η = 90** |
| Output | 8 poses $(x,y,\theta)$ over 4 s at 0.5 s |
| Input | Front camera only, no LiDAR; ego state + navigation command; programmatic text condition |
| RL | Planner-only DiffGRPO, backbone and scorer frozen, NAVSIM evaluator reward |
| Seeds | Validation-best checkpoint per seed, aggregated over **ten seeds** |
| Aux model | Fixed B22, 64 proposals, CLOVER-derived pseudo-expert targets, 4 GPUs × batch 5 × 4 accum × 80 epochs |
| Latency | **170 ms** adaptive / 190 ms fixed B15 / 320 ms full depth, A100 80GB batch 1 |

---

## Limitations

1. **90.8 single-trajectory is mid-frontier, and the table inherits DriveVA's comparison set.** Above it in this wiki: CLEAR/DA-WAM 93.7, DriveSuprim 93.5, Drive-JEPA 93.3, HybridDriveVLA 92.1, WA-JEPA 91.8, DynVLA 91.7, SimWAM 91.5, FLARE 91.4, DiffusionDriveV2 91.2, SGDrive 91.1. None appears in Table 2. The "state-of-the-art among compared world-model planners" phrasing is accurate and appropriately hedged, but the hedge is doing real work.

2. **92.6 is a different model and should not be read as a single-trajectory result.** It is a fixed block-22 exit with **64 proposals**, no adaptive routing, a different (non-diffusion, four-block refinement) decoder, and **CLOVER-derived pseudo-expert targets scored with the true NAVSIM evaluator using training-time map and future occupancy**. That is privileged supervision of the same kind the wiki flags for Hydra-MDP distillation, [[sources/auto-jepa.md]], and [[sources/da-wam.md]]. It belongs in the selection-based family alongside DriveSuprim 93.5 and CLEAR 93.7, where it does not lead.

3. **Table 13 and Table 19 report identical numbers under incompatible descriptions.** Table 13's bottom two rows are captioned as planning scores from "a fixed-exit 64-proposal model"; Table 19's rows carry the same values (92.01/92.12/92.11/92.05/92.07 and 92.45/92.55/92.59/92.45/92.43) but are captioned a "Wan-based scorer pretest", and Table 20 explicitly says such values "measure the true score of the selected candidate and are not end-to-end planner PDMS." One of those descriptions is wrong, and the 92.6 headline lives in the same numeric neighborhood (92.59 in Table 21, 92.62 in Table 20). The provenance of the headline is not cleanly traceable from the paper.

4. **"Validation-best checkpoint over ten seeds" is a selection procedure, not variance reporting.** Taking the best-on-validation checkpoint per seed and aggregating is optimistic relative to single-run reporting, and the paper never reports a PDMS standard deviation — only cell-wise std for the pairwise-advantage counts. The wiki still has exactly one measured PDMS/EPDMS seed std ([[sources/wa-jepa.md]], 0.053). The 10-seed protocol is nonetheless better discipline than almost every paper here.

5. **The routing gain over a fixed exit is +0.17 PDMS**, unaccompanied by variance, and smaller than WA-JEPA's measured seed std would suggest is resolvable. The defensible claim is the *latency* one: 170 ms vs 190 ms at no loss. The headline "47% below the 320 ms fixed full-depth planner" compares against a configuration that is also **4.80 PDMS worse** — nobody would deploy it, so that comparison is not an operating-point trade-off.

6. **The scorer trains on evaluator-provided component targets**, so the deployed controller is distilled from the benchmark's own metric. Its failure modes are the paper's to own and it does own them (51 scenes at ≥50-point gaps), but a metric-distilled verifier is not evidence of scene understanding.

7. **The noise-robustness result does not directly refute [[sources/drivelaw.md]]'s collapse.** Adaptive-WAM varies the noise index conditioning a **single forward pass**; DriveLaW extracts from a latent that has been through *t* actual denoising iterations, so its t=10 latents carry different activation statistics than a one-shot forward at any index. The two experiments are compatible: noise level per se is cheap to get wrong, but iterating the denoiser and reading late may still be harmful for reasons unrelated to noise level. The paper scopes its claim correctly and does not overreach; the wiki should not either.

8. **Five noise indices, one scheduler, one backbone family.** The paper says so explicitly.

9. **The v2 table has five baselines and mixes evaluator conventions.** DiffusionDrive at 84.5 matches the wiki's *corrected* cohort while ReCogDrive 83.6 and DriveVLA-W0 86.1 match the *pre-fix* cohort. This is the fourth ingested table shown to mix, after GeoWAM, DA-WAM, and BrainWAM. 89.9 EPDMS therefore cannot be placed against the wiki's v2 leaderboard.

10. **NAVSIM and zero-shot nuScenes only.** No navhard, no Bench2Drive, no HUGSIM, no reactive closed loop. The routing thesis — spend more compute on harder scenes — is exactly the claim a hard/OOD split would test, and navhard exists.

11. **Front camera only, no LiDAR**, and six independent trajectory heads whose combined parameter count is never reported. Worst-case routing (η=95) evaluates most exits and costs **284 ms, worse than the 190 ms fixed baseline**, so the latency win is an average, not a bound — a relevant distinction for a safety-critical scheduler.

12. **Code is announced but not released.**

---

## Key Cross-References

- **This reframes the wiki's central denoising dispute** — see [[concepts/world-model-for-ad.md]]. The field has been treating "how much denoising" as one axis; Adaptive-WAM shows it is at least three, and that the one nobody varied (readout depth) dominates the one everybody varied (noise level) by ~40×. It also converges with [[sources/drivelaw.md]] (t=1) and [[sources/brainwam.md]] (1 step) on the same practical conclusion: **one forward pass through the video DiT is what you need**, leaving [[sources/foresight.md]]'s 100-step schedule as the outlier.

- **A new design axis**: readout depth. No other wiki paper reports which layer of a video prior it reads from. Intermediate beats final by 4.80 PDMS post-RL, with per-scene complementarity (Jaccard 0.69–0.82) that no single fixed depth captures.

- **Depth, confirmed on a second backbone**: [[sources/geoworldad.md]] runs the analogous study on a 24-block StreamVGGT decoder and reaches a compatible but sharper conclusion — four selected layers {4, 11, 17, 23} consumed by successive refinement stages score 89.3, against 88.2 for the final layer alone iterated four times and **87.6 for all 24 layers fed into one interaction stage**. So the readout point matters, as measured here, but the best answer is several depths consumed progressively rather than one well-chosen depth; concatenating everything into a single stage is worse than either. It also independently confirms that using a geometry foundation model off the shelf is costly: an anchor-frame StreamVGGT beats a from-scratch planner by only 0.6 PDMS, the same shape as this paper's frozen-Wan 84.20.
- **The frozen-backbone verdict**: Table 21's frozen 84.20 vs joint-LoRA 90.62 is a 6.42-point argument against using an unadapted generative prior as a planning encoder, which is precisely [[sources/foresight.md]]'s design. See [[concepts/foundation-backbones-for-ad.md]].

- **Wan2.2-TI2V-5B, fifth paper**: [[sources/simwam.md]] 91.5 (training-time only), [[sources/driveva.md]] 90.5 NAVSIM-only / 90.9 mixed (joint denoising), [[sources/drivewam.md]] 90.1 (chunked inverse dynamics), [[sources/brainwam.md]] 89.5 (8-token action bridge), Adaptive-WAM 90.8 (early-exit features). Adaptive-WAM is second on score and **first by a wide margin on latency**.

- **Adaptive routing**: [[concepts/adaptive-routing.md]] — this is the wiki's second scene-conditioned compute allocator after [[sources/clear.md]], and the first to route over *backbone depth* rather than candidate budget. The two are orthogonal and composable.

- **The tie problem, named**: [[concepts/selection-based-planning.md]] — >95% of scenes have candidate groups that are jointly perfect, jointly zero, or top-tied. Adaptive-WAM's response (component regression as verification, not ranking) is the cleanest treatment of this in the wiki, and it explains why so many selection-based papers report near-identical oracle ceilings.

- **CLOVER lineage**: the first author is CLOVER's first author, and CLOVER supplies both [[sources/auto-jepa.md]]'s scorer initialization and this paper's pseudo-expert target protocol. CLOVER remains un-ingested and is now referenced by two wiki papers.

- **Fills a recorded gap**: [[sources/driveva.md]]'s page notes its NAVSIM sub-scores were truncated in source. Table 2 supplies them (NC 99.2, DAC 97.5, TTC 98.7, Comf 100, EP 83.5) and disambiguates 90.9 (mixed data) from 90.5 (NAVSIM only).
