---
title: "LWDrive: Layer-Wise World-Model-Guided Vision-Language Model Planning for Autonomous Driving"
type: source-summary
sources: [raw/papers/LWDrive_ Layer-Wise World-Model-Guided Vision-Language ModelPlanning for Autonomous Driving.md]
related: [concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/foundation-backbones-for-ad.md, concepts/selection-based-planning.md, concepts/dual-system-vla.md, concepts/nuscenes-waymo-evals.md, concepts/rl-for-ad.md, sources/geoworldad.md, sources/adaptive-wam.md, sources/brainwam.md, sources/foresight.md, sources/simwam.md, sources/drivelaw.md, sources/da-wam.md, sources/wcog-vla.md, sources/onevl.md, sources/flare.md, sources/drivevla-w0.md, sources/drivesuprim.md, sources/hybriddriveVLA.md, sources/clear.md, sources/drive-jepa.md, sources/sgdrive.md, sources/recogdrive.md, sources/epona.md, sources/diffusiondrive.md, sources/autovla.md, sources/wa-jepa.md, sources/geowam.md, sources/wam-diff.md, sources/dynvla.md, sources/diffusiondrive-v2.md, sources/senna2.md, sources/drivewam.md]
created: 2026-09-04
updated: 2026-09-27
confidence: high
---

**Paper**: LWDrive: Layer-Wise World-Model-Guided Vision-Language Model Planning for Autonomous Driving
**Authors**: Chen Yang, Yuhao Wei, Ze Xu, Ziheng Zou, Shuang Liang, Delin Ouyang, Lingfeng Qi, Jie Li, Guofa Li
**Org**: Chongqing University
**arXiv**: 2606.29879v2

---

## Summary

LWDrive's thesis is that a VLM's trajectory should be treated as a **coarse plan to be refined**, not as an output — and that the refinement should read the VLM's *internal* representations at several depths rather than only its final layer. Qwen2.5-VL-3B emits an intent-aware coarse trajectory; a **Foresight Cascade Planner (FCP)** then expands a candidate pool and refines it over six stages, each one consuming hidden states from a different Qwen layer through **Bridge Attention** and grounding the proposals in multi-view BEV features. A score head trained against NAVSIM's own PDMS composition picks the winner. **92.0 PDMS on NAVSIM-v1, 89.6 EPDMS on v2, 0.37 m L2 on nuScenes ST-P3** — and no reinforcement learning anywhere.

The world model is a **future-frame VAE-latent denoising head attached to the VLM's hidden states during Stage-1 training only**. It never runs at inference and no future image is required at deployment, which places LWDrive in the training-time-only camp alongside [[sources/simwam.md]], [[sources/drivevla-w0.md]], [[sources/flare.md]], and [[sources/onevl.md]].

Two results carry the page, and they pull in opposite directions:

1. **The readout-depth axis, measured on a VLM for the first time — and it is nearly flat.** Reading only the final Qwen layer scores **91.8**; reading six layers across the cascade scores **92.0**. That is **+0.2**, against [[sources/adaptive-wam.md]]'s **+4.80** on a video DiT and [[sources/geoworldad.md]]'s **+1.1** on a geometry decoder at matched iteration count. Worse for the paper's framing, the sub-metrics show it is a *trade* rather than a gain: layer-wise buys DAC (+0.4) and TTC (+0.5) and gives back ego progress (−0.8). The design in the title is the smallest measured effect in the paper.

2. **The world-model column is only ever varied in the configuration the paper spends the rest of its length arguing is inadequate.** Rows 1–2 of Table 4 turn WM supervision on and off with **no FCP at all** (84.5 → 86.3, +1.8 on a directly decoded VLM trajectory). There is no row for the full FCP without WM supervision — so the claim that FCP's foresight features are *world-model-shaped* is asserted, never measured. The +7.5 from 84.5 to 92.0 is overwhelmingly the refinement stack, not the world model.

What LWDrive does establish cleanly is that a **3B VLM plus a well-built proposal refiner reaches 92.0 without RL**, on four cameras, with a 280×504 front-view input to the VLM. That places it eighth among non-Best-of-N NAVSIM-v1 entries in this wiki.

---

## Positioning

![[figure1_pver2.png|Three VLM planning paradigms: direct trajectory decoding, single-stage VLM/backbone fusion, and LWDrive's world-model-guided coarse-to-fine refinement]]

**Figure 1**: Comparison of VLM planning paradigms for autonomous driving. (a) Direct VLM-to-trajectory decoding lacks fine-grained correction. (b) Single-stage VLM/backbone fusion injects VLM semantics only once. (c) LWDrive performs world-model-guided coarse-to-fine refinement.

The paper's taxonomy is a two-way split with itself as the third option:

- **(a) Direct decoding** — trajectories or actions decoded straight from VLM representations. "Capturing high-level driving intentions but often lacking precise geometric and temporal constraints." Its own Table 4 row 1 prices this at **84.5 PDMS**.
- **(b) Single-stage fusion** — a driving backbone whose features are fused with VLM representations once, before an action expert. "VLM semantics are usually injected only once and remain weakly coupled with subsequent trajectory correction."
- **(c) LWDrive** — the VLM output is an intent anchor; refinement consumes VLM representations repeatedly, at increasing depth.

Category (b) is where most of this wiki's dual-system work sits ([[concepts/dual-system-vla.md]]), and the complaint is a fair characterisation of it. What LWDrive adds is not a new coupling *site* but a coupling *schedule*: the planner's refinement stages are interleaved with the backbone's depth. [[sources/geoworldad.md]] independently arrived at the same structure on a geometry decoder, and neither paper cites the other.

---

## Method

![[113.png|LWDrive architecture: Qwen2.5-VL branch with world head and trajectory decoder, feeding a Foresight Cascade Planner with BEV grounding and a score head]]

**Figure 2**: Overall architecture. Future-frame world-model supervision guides the VLM toward predictive scene representations and an intent-aware coarse trajectory. The FCP progressively refines a candidate pool using layer-wise foresight features, temporal states, action-query memories, and multi-view BEV representations; a score head ranks the refined candidates.

### Stage 1 — World-model-supervised coarse planning

Qwen2.5-VL-3B is first adapted to ego-centric driving on the **Impromptu** dataset, then trained with two objectives.

**Future-frame world modelling.** A frozen VAE encodes the future image $I_{t+\Delta}$ to a clean latent $z_{t+\Delta}$. The denoising condition concatenates the final-layer vision hidden state and the action-query hidden state, $c_{t}=[F_{t}^{\mathrm{V}};h_{t}^{\mathrm{A}}]$, and a world head predicts the clean latent from a noisy one:

$$\hat{z}_{t+\Delta}=D_{\theta}(z^{\tau}_{t+\Delta},c_{t},\tau),\qquad \mathcal{L}_{\mathrm{wm}}=d(\hat{z}_{t+\Delta},z_{t+\Delta}).$$

**Coarse trajectory.** A decoder emits $\hat{Y}^{0}$ under SmoothL1 against the expert:

$$\mathcal{L}_{\mathrm{traj}}=\frac{1}{T}\sum_{t=1}^{T}\mathrm{SmoothL1}(\hat{y}^{0}_{t}-y^{\ast}_{t}),\qquad \mathcal{L}_{\mathrm{stage1}}=\lambda_{\mathrm{wm}}\mathcal{L}_{\mathrm{wm}}+\lambda_{\mathrm{traj}}\mathcal{L}_{\mathrm{traj}}.$$

**The loss weighting is worth recording: $\lambda_{\mathrm{wm}}=15.0$ against $\lambda_{\mathrm{traj}}=1.0$.** Stage 1 is fifteen-to-one a world-modelling run, and the ratio is never ablated. Only the VLM branch, the world head, and the trajectory decoder train here — FCP and the score head are excluded so that "the refinement modules [do not disturb] the world-model-supervised VLM representation."

No future image is needed at inference. This is future prediction as a **representation-shaping objective**, not as an inference-time computation.

### Stage 2 — Foresight Cascade Planner

![[figure17fix.png|Foresight Cascade Planner: Bridge Attention over proposal, action-query, ego-state and VLM foresight memories, followed by BEV-grounded residual trajectory updates]]

**Figure 3**: Bridge Attention injects proposal interaction, action-query memory, ego-state context, and layer-wise VLM foresight features; BEV refinement grounds the proposals with multi-view geometric cues and predicts residual updates.

**Pool initialisation.** The *pooled action-query latent* — not the decoded coarse trajectory — is the intent anchor:

$$Q^{0},P^{0}=\mathrm{TPM}(z_{\mathrm{AQ}},e_{t},E_{\mathrm{init}})$$

combining the action-query latent, ego-state encoding, and learnable proposal embeddings into $N_{\mathrm{p}}$ candidates. **$N_{\mathrm{p}}$ is never given a value anywhere in the paper.**

**Sparse layer-wise cascade.** With $L$ Qwen action layers and refinement interval $M$, refinement happens at $l_{r}=rM$ for $r=1,\ldots,R$. The implementation gives $L=36$ and $R=6$, so $M=6$ and the taps are layers $\{6,12,18,24,30,36\}$. Each stage collects

$$C_{r}=(H_{\mathrm{V}}^{l_{r}},H_{\mathrm{A}}^{l_{r}},B_{t},s_{t}),\qquad Q^{r},P^{r}=\mathcal{F}_r(Q^{r-1},P^{r-1},C_{r})$$

— selected-layer vision/task hidden states, selected-layer action-query hidden states, current-frame multi-view BEV features, and ego state. The stated rationale: "Earlier action layers retain local visual and motion-related cues, while later layers encode higher-level intention and scene-level reasoning."

**Bridge Attention** is a single attention over concatenated multi-source memory — proposal-pool self memory, action-query and ego-state memory, and VLM foresight memory:

$$\mathrm{BA}(Q)=\mathrm{Attn}(Q,K_{\mathrm{mem}},V_{\mathrm{mem}}).$$

Candidates therefore interact with each other *and* absorb VLM features in the same operation. This is the proposal-interaction mechanism that fixed-vocabulary selectors get from a scoring head ([[concepts/selection-based-planning.md]]), fused with the layer-wise injection.

**BEV residual refinement.** Proposals then cross-attend to current-frame BEV features (ResNet-34 over four camera views) and each stage predicts a residual:

$$\Delta Y_{i}^{r}=\mathrm{MLP}_{\mathrm{ref}}(Q_{i}^{r}),\qquad Y_{i}^{r}=Y_{i}^{r-1}+\Delta Y_{i}^{r}.$$

The residual form is deliberate: it "prevents the refinement module from overwriting the coarse VLM intention in a single step."

### Scoring and training

$$S_{i}=\mathrm{MLP}_{\mathrm{score}}(\mathrm{Pool}(Q_{i}^{R})),\qquad \hat{Y}=Y^{R}_{\arg\max_i S_i}$$

Targets come from **non-reactive log simulation of every candidate** — the NAVSIM PDMS composition itself:

$$\hat{S}_{i}=\mathrm{NC}_{i}\cdot\mathrm{DAC}_{i}\cdot\frac{5\mathrm{EP}_{i}+5\mathrm{TTC}_{i}+2\mathrm{C}_{i}}{12},\qquad \mathcal{L}_{\mathrm{score}}=\mathrm{BCE}(S,\hat{S}).$$

This is privileged simulator distillation of the Hydra-MDP class, identical in kind to [[sources/drivesuprim.md]], [[sources/drive-jepa.md]], [[sources/da-wam.md]], [[sources/geoworldad.md]], and [[sources/adaptive-wam.md]]'s auxiliary model. **92.0 is not a single-trajectory result.**

Proposals are supervised min-over-$N$ at every stage with a discount on earlier ones:

$$\ell_{i}^{r}=\frac{1}{T}\sum_{t=1}^{T}\mathrm{SmoothL1}(y_{i,t}^{r}-y_{t}^{\ast}),\qquad \mathcal{L}_{\mathrm{ref}}=\sum_{r=1}^{R}\beta^{R-r}\min_{i}\ell_{i}^{r}.$$

$\beta$, $\lambda_{\mathrm{ref}}$, and $\lambda_{\mathrm{score}}$ are unreported.

**In Stage 2 the Qwen backbone and action-query embeddings are frozen.** The FCP therefore consumes *fixed* foresight features shaped by a separately-run Stage 1 — structurally the "separate LoRA, then cache features" rung of [[sources/adaptive-wam.md]]'s adaptation ladder, which on a video DiT recovered only 0.75 of a 6.42-PDMS gap versus joint adaptation. Different backbone family, so this is a prior rather than a refutation, but LWDrive runs no experiment on it.

### Implementation

| | |
|---|---|
| VLM | Qwen2.5-VL-3B, **front view only at 280×504**, 36 action layers |
| BEV | ResNet-34 over **four** camera views |
| Refinement | 6 modules ($M=6$) |
| Stage 1 | 8,000 steps, lr 2e-4, global batch 48, $\lambda_{\mathrm{wm}}/\lambda_{\mathrm{traj}}=15/1$, **~100 hours** |
| Stage 2 | 12 epochs, lr 3e-5, global batch 64, Qwen frozen, **~80 hours** |

**No GPU count, no latency, no FPS, and no parameter count appears anywhere in the paper.**

---

## Results

### Table 1 — NAVSIM-v1 navtest (PDMS)

| Method | Venue | Sensors | NC ↑ | DAC ↑ | TTC ↑ | C ↑ | EP ↑ | PDMS ↑ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Human driver | NeurIPS 2024 | – | 100.0 | 100.0 | 100.0 | 99.9 | 87.5 | **94.8** |
| PDM-Closed ⚠ | PMLR 2023 | – | 94.6 | 99.8 | 89.9 | 86.9 | 99.9 | 89.1 |
| *E2E-based Methods* | | | | | | | | |
| UniAD | CVPR 2023 | 6×C | 97.8 | 91.9 | 92.9 | 100.0 | 78.8 | 83.4 |
| LTF / TransFuser | TPAMI 2022/2023 | 3×C+L | 97.4 | 92.8 | 92.4 | 100.0 | 79.0 | 83.8 |
| DriveX-S | ICCV 2025 | – | 97.5 | 94.0 | 93.0 | 100.0 | 79.7 | 84.5 |
| PRIX | arXiv 2025 | C | 98.1 | 96.3 | 94.1 | 100.0 | 82.3 | 87.8 |
| DiffusionDrive | CVPR 2025 | 3×C+L | 98.2 | 96.2 | 94.7 | 100.0 | 82.2 | 88.1 |
| Hydra-MDP++ | CVPR 2025 | 3×C+L | 98.6 | 98.6 | 95.1 | 100.0 | 85.7 | 91.0 |
| iPad | CVPR 2025 | – | 98.6 | 98.3 | 94.9 | 100.0 | **88.0** | 91.7 |
| *World-Model-based Methods* | | | | | | | | |
| DrivingGPT | ICCV 2025 | 1×C | 98.9 | 90.7 | 94.9 | 95.6 | 79.7 | 82.4 |
| World4Drive | ICCV 2025 | – | 97.4 | 94.3 | 92.8 | 100.0 | 79.9 | 85.1 |
| Epona | ICCV 2025 | 3×C | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 |
| WoTE | ICCV 2025 | 3×C+L | 98.5 | 96.8 | 94.4 | 99.9 | 81.9 | 88.3 |
| DriveWorld-VLA | CVPR 2026 | 3×C | **99.1** | 98.2 | 96.1 | 100.0 | 85.9 | 91.3 |
| *VLA-based Methods* | | | | | | | | |
| FSDrive | NeurIPS 2025 | – | 98.2 | 93.8 | 93.3 | 99.9 | 80.1 | 85.1 |
| AutoVLA | NeurIPS 2025 | 3×C | 98.4 | 95.6 | **98.0** | 99.9 | 81.9 | 89.1 |
| DriveVLA-W0 | ICLR 2026 | 1×C | 98.7 | **99.1** | 95.3 | 99.3 | 83.3 | 90.2 |
| ReCogDrive | ICLR 2026 | 3×C | 97.9 | 97.3 | 94.9 | 100.0 | **87.3** | 90.8 |
| SGDrive | CVPR 2026 | – | 98.6 | 97.8 | 96.2 | 100.0 | 85.8 | 91.1 |
| **LWDrive (ours)** | – | 4×C | 98.8 | 98.4 | 96.2 | 99.8 | **87.3** | **92.0** |

**The PDM-Closed row is scrambled.** Canonical NAVSIM values for PDM-Closed are TTC 86.9, Comfort 99.9, EP 89.9; this table prints TTC 89.9, Comfort 86.9, EP 99.9 — a three-column cyclic permutation. An ego progress of 99.9 would exceed the human driver's 87.5 by 12 points, and a comfort of 86.9 is implausible for a rule-based planner. The PDMS (89.1) is correct, so this is a transcription error in one row, not a different evaluation.

**Otherwise the v1 hygiene is good on what is present.** UniAD 83.4, DiffusionDrive 88.1, WoTE 88.3, Epona 86.2, FSDrive 85.1, iPad 91.7, AutoVLA 89.1, SGDrive 91.1, and DriveWorld-VLA 91.3 all match this wiki's canonical values. ReCogDrive appears at **90.8**, the NeurIPS camera-ready figure rather than the earlier 89.6 — the same choice [[sources/wam-diff.md]] and [[sources/wcog-vla.md]] make. DriveVLA-W0 is cited at its **90.2 anchor headline** rather than the 87.2 reimplementation row that [[sources/drivelaw.md]], [[sources/brainwam.md]], and [[sources/adaptive-wam.md]] propagate.

**What is absent is the frontier.** [[sources/drivesuprim.md]] does not appear in the v1 table at all — despite appearing in the v2 table — and neither do [[sources/clear.md]] 93.7, [[sources/da-wam.md]] 93.7, [[sources/drive-jepa.md]] 93.3, [[sources/wcog-vla.md]] 92.9, [[sources/hybriddriveVLA.md]] 92.1, [[sources/wa-jepa.md]] 91.8, [[sources/dynvla.md]] 91.7, or [[sources/simwam.md]] 91.5. The abstract does not claim SOTA; the results section says "the best PDMS **among the compared methods**," which is accurate but weak.

**Three methods new to the wiki**: DriveX-S 84.5, PRIX 87.8 / 84.2, World4Drive 85.1.

### Table 2 — NAVSIM-v2 navtest (EPDMS)

| Method | NC ↑ | DAC ↑ | DDC ↑ | TLC ↑ | EP ↑ | TTC ↑ | LK ↑ | HC ↑ | EC ↑ | EPDMS ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Human Agent | 100.0 | 100.0 | 99.8 | 100.0 | 87.4 | 100.0 | 100.0 | 98.1 | 90.1 | **90.3** |
| *E2E-based Methods* | | | | | | | | | | |
| Ego Status MLP | 93.1 | 77.9 | 92.7 | 99.6 | 86.0 | 91.5 | 89.4 | **98.3** | 85.4 | 64.0 |
| TransFuser | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | **98.3** | 87.2 | 76.7 |
| Hydra-MDP++ | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| DriveSuprim | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | **98.3** | 77.0 | 83.1 |
| ARTEMIS | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | **98.3** | 98.3 ⚠ | 83.1 |
| PRIX | 98.0 | 95.6 | 99.5 | 99.8 | 87.4 | 97.2 | **97.1** | **98.3** | 87.6 | 84.2 |
| DiffusionDrive | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | **98.3** | 87.7 | 84.5 |
| *VLA / World-Model-based Methods* | | | | | | | | | | |
| DriveVLA-W0 | 98.5 | **99.1** | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 |
| DriveWorld-VLA | 98.6 | **99.1** | **99.6** | 99.8 | 87.4 | 97.9 | 97.0 | 97.8 | 78.6 | 86.8 |
| **LWDrive (ours)** | **98.8** | 98.4 | 99.0 | 99.7 | **90.3** | **98.6** | 96.3 | 97.9 | 73.3 | **89.6** |

**EP 90.3 is the highest NAVSIM-v2 ego progress recorded anywhere in this wiki**, ahead of [[sources/geoworldad.md]]'s 89.1 and [[sources/diffusiondrive-v2.md]]'s 88.9 — and it exceeds the Human Agent's 87.4 in the same table by 2.9 points. TTC 98.6 ties GeoWorldAD for the best v2 value here. **EC 73.3 is the cost**: below every other method in its own table except Hydra-MDP++ (70.9) and DriveVLA-W0 (58.9). The profile is an aggressive planner, and the paper does not discuss the comfort regression.

The 89.6 EPDMS sits **0.7 below the Human Agent's 90.3** in its own table — the second-closest approach to the human reference this wiki records on v2, after [[sources/adaptive-wam.md]]'s 0.4.

**This table is the sixth ingested table shown to mix evaluator conventions**, and it mixes them the same way [[sources/geoworldad.md]]'s does: TransFuser 76.7, ARTEMIS 83.1, and DriveVLA-W0 86.1 are pre-fix values, while DiffusionDrive 84.5 and DriveWorld-VLA 86.8 are corrected ones, and DriveSuprim 83.1 / Hydra-MDP++ 81.4 match neither cohort (this wiki carries 87.1 and 84.1). See [Evaluator Drift](../concepts/navsim-benchmark.md#evaluator-drift-this-table-mixes-two-protocols).

**Every overlapping row is digit-identical to GeoWorldAD's table** — TransFuser, DriveSuprim, DiffusionDrive, and DriveVLA-W0 all match across nine submetrics. The two papers evidently drew from the same source. That does not validate either, but it does mean **LWDrive's 89.6 and GeoWorldAD's 90.4 sit in the same reference frame** and can be compared with each other: GeoWorldAD is 0.8 ahead.

**ARTEMIS EC = 98.3 again.** This is the third ingested paper to print ARTEMIS's EC as a duplicate of its HC, after [[sources/da-wam.md]] and [[sources/wcog-vla.md]]; [[sources/brainwam.md]] remains the only source for 89.1, and four papers report "–". The tally is now **three to one for 98.3**, which makes the duplication look like a widely-copied transcription error rather than a genuine value.

### Table 3 — nuScenes ST-P3 open-loop

| Method | $L_2$ (m) ↓ | CR (%) ↓ |
|---|---:|---:|
| *Non-Autoregressive* | | |
| ST-P3 | 2.11 | 0.71 |
| VAD | 1.25 | 1.09 |
| Ego-MLP | 0.78 | 0.38 |
| UniAD | 0.69 | 0.12 |
| InsightDrive | 0.44 | 0.15 |
| BEV-Planner | 0.55 | 0.59 |
| *Autoregressive* | | |
| DriveVLM | 0.40 | 0.27 |
| GPT-Driver | 0.44 | 0.17 |
| OccWorld | 0.77 | 0.32 |
| Doe-1 | 0.70 | 0.21 |
| RDA-Driver | 0.40 | **0.10** |
| OpenEMMA | 2.81 | – |
| DME-Driver | 0.98 | 0.29 |
| OmniDrive | 0.84 | 0.94 |
| AutoVLA (action only) | 0.43 | 0.19 |
| AutoVLA (w/ CoT) | 0.48 | 0.13 |
| **LWDrive (ours)** | **0.37** | 0.16 |

0.37 m is the best $L_2$ in the table, by 0.03 over DriveVLM and RDA-Driver — and RDA-Driver's collision rate (0.10%) is better. **LWDrive consumes ego status**, so the [[concepts/nuscenes-waymo-evals.md]] caveat applies in full: the table's own Ego-MLP row (0.78 / 0.38) and BEV-Planner row (0.55) exist precisely because ego-status shortcuts dominate this metric. A 0.03 m margin on ST-P3 $L_2$ is not evidence of planning quality.

**The paper never states whether the nuScenes result is zero-shot or from a separate nuScenes training run.** Given the wiki's zero-shot WAM cluster, that distinction matters and is unrecoverable from the text.

**One statement in the results section deserves quoting**, because no other ingested paper says it out loud:

> "PDMS/EPDMS follow official evaluator: scores are scenario-averaged, while sub-metrics are independently averaged and cannot reconstruct aggregate scores."

That is the correct description of why the wiki's [evaluator-drift analysis](../concepts/navsim-benchmark.md#evaluator-drift-this-table-mixes-two-protocols) has to reason from *submetric identity across papers* rather than by recomputing aggregates.

---

## Ablation — Table 4 (NAVSIM-v1)

| WM | LW | BA | BR | NC ↑ | DAC ↑ | EP ↑ | TTC ↑ | C ↑ | PDMS ↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| ✗ | ✗ | ✗ | ✗ | 98.2 | 92.7 | 79.1 | 93.9 | 100.0 | **84.5** |
| ✓ | ✗ | ✗ | ✗ | 98.4 | 94.7 | 81.5 | 94.7 | 99.9 | **86.3** |
| ✓ | ✓ | ✓ | ✗ | 98.9 | 97.2 | 83.9 | 95.2 | 99.8 | **89.3** |
| ✓ | ✓ | ✗ | ✓ | 98.7 | 97.7 | 84.6 | 95.1 | 99.7 | **90.0** |
| ✓ | Final | ✓ | ✓ | 98.7 | 98.0 | **88.1** | 95.7 | 99.7 | **91.8** |
| ✓ | ✓ | ✓ | ✓ | **98.8** | **98.4** | 87.3 | **96.2** | 99.8 | **92.0** |

WM = world-model supervision, LW = layer-wise foresight, BA = Bridge Attention, BR = BEV refinement. "Final" uses only the final-layer VLM feature.

### The layer-wise result is a wash, and the sub-metrics say so

Rows 5 and 6 are the paper's title claim, at matched refinement-stage count:

| | NC | DAC | EP | TTC | PDMS |
|---|---:|---:|---:|---:|---:|
| Final layer only | 98.7 | 98.0 | **88.1** | 95.7 | 91.8 |
| Layer-wise {6,12,18,24,30,36} | 98.8 | **98.4** | 87.3 | **96.2** | **92.0** |
| Δ | +0.1 | **+0.4** | **−0.8** | **+0.5** | **+0.2** |

**Multi-depth readout buys drivable-area compliance and time-to-collision and gives back ego progress.** That is the same directional finding [[sources/geoworldad.md]] reports — "multi-scale buys safety, iterating buys progress" — reproduced on a completely different backbone family, which is the more interesting half of the result. What does not reproduce is the magnitude.

**Readout depth across three backbones:**

| Paper | Backbone | Comparison | Δ PDMS |
|---|---|---|---:|
| [[sources/adaptive-wam.md]] | Wan2.2-TI2V-5B video DiT, 30 blocks | best single exit (block 15) vs. final block | **+4.80** |
| [[sources/geoworldad.md]] | StreamVGGT geometry decoder, 24 blocks | 4 layers vs. final layer, both ×4 iterations | **+1.1** |
| **LWDrive** | **Qwen2.5-VL-3B, 36 layers** | **6 layers vs. final layer, both ×6 stages** | **+0.2** |

The ordering is suggestive and worth stating as a hypothesis rather than a conclusion: **the value of intermediate readout appears to scale with how much of the backbone's final layer is committed to a non-planning output.** A video DiT's last block is committed to predicting noise; a geometry decoder's is committed to point maps; a VLM's last layer is committed to *language*, which for a driving VLM already fine-tuned on trajectory-style responses may be closer to the planning target than either. Nothing in these three papers tests that, and the three architectures differ on many other axes.

The practical consequence for this wiki: **the +4.80 figure should not be generalised to VLM backbones.** The many designs on [[concepts/foundation-backbones-for-ad.md]] that read a VLM's final hidden state are, on this one measurement, leaving about 0.2 PDMS on the table, not 5.

### The world-model column tests the wrong configuration

Rows 1 → 2 add world-model supervision to a **bare VLM with no FCP**: 84.5 → 86.3, **+1.8 PDMS**. That is a legitimate and useful training-time-only world-model result, and it is measured on a directly decoded VLM trajectory.

**But the paper's thesis is that FCP's foresight features are world-model-shaped**, and the row that would test it — full FCP, WM supervision off — does not exist. Three things follow:

1. The +1.8 is measured in the configuration the paper elsewhere calls insufficient (Figure 1(a)). Whether it survives once six refinement stages and BEV grounding are added is unknown, and refinement stacks are exactly the sort of thing that absorbs representational deficits.
2. The word "Foresight" in *Foresight Cascade Planner* is unearned by any experiment. The cascade demonstrably works (+5.7 over the WM-supervised coarse trajectory); that its inputs are *foresight* features rather than merely deep features is asserted.
3. **The stack, not the world model, produces 92.0.** Of the +7.5 from row 1 to row 6: WM supervision +1.8, the FCP stack +5.5, layer-wise readout +0.2.

Removing BEV refinement costs **2.7** (89.3 vs 92.0) and removing Bridge Attention costs **2.0** (90.0 vs 92.0) — both larger than either headline mechanism. LWDrive is, by its own numbers, a proposal-refinement paper.

### Where it sits in the training-time-only camp

| Paper | Training-time future target | Measured value |
|---|---|---|
| [[sources/drivevla-w0.md]] | AR + diffusion future frames | amplifies the data-scaling law |
| [[sources/flare.md]] | DINOv2 semantic features | 86.9 SFT → 91.4 RFT |
| [[sources/simwam.md]] | video tokens, isolated attention mask | inference-time access worth ~0.0 |
| [[sources/onevl.md]] | future frame via training-time visual decoder | folded into 88.84 |
| [[sources/adaptive-wam.md]] | video prediction, reads intermediate activations | joint adaptation worth +6.42 |
| **LWDrive** | **VAE latent of one future frame, $\lambda_{\mathrm{wm}}=15$** | **+1.8, coarse stage only** |

---

## Visualization

![[visualization_two_scenarios_wide.png|Two scenes showing the front-view image, the world head's predicted future frame, the trajectory projected onto the front view, and the BEV trajectory]]

**Figure 4**: For each scene: current front-view image, the future frame predicted by the world-model head, the planned trajectory projected onto the front view, and the corresponding BEV result.

This is the only evidence that the world head learns anything future-like. Note that the decoded future frames are shown **for illustration only** — the deployed system never decodes them and the planner never sees them.

---

## Limitations

1. **The central claim is untested.** No ablation runs the full FCP without world-model supervision. The +1.8 for WM is measured only on a bare VLM decoder with no refinement, which is the configuration the paper argues against everywhere else. One row would fix this.

2. **The layer-wise mechanism in the title is worth +0.2 PDMS**, and the sub-metrics show it as a safety-for-progress trade (DAC +0.4, TTC +0.5, EP −0.8) rather than a gain. No seed variance is reported, and 0.2 is inside the range where [[sources/wa-jepa.md]]'s 0.053 EPDMS seed std provides only weak reassurance — that measurement was for a stochastic sampler on a fixed model, not for training-seed variation.

3. **No latency, FPS, parameter count, or GPU count anywhere.** The system runs Qwen2.5-VL-3B over 36 layers with six interleaved refinement modules, plus a ResNet-34 BEV encoder over four views, plus per-candidate scoring. This wiki tracks [[sources/adaptive-wam.md]] at 170 ms, [[sources/brainwam.md]] at 475–644 ms, [[sources/simwam.md]] at 518 ms, and [[sources/foresight.md]] at 900 ms; LWDrive cannot be placed on that axis at all. "About 100 hours" and "about 80 hours" of training are quoted without saying on how many devices.

4. **The score head is trained on NAVSIM's own PDMS composition** via non-reactive log simulation of every candidate — privileged benchmark distillation, and it makes 92.0 a selection result rather than a single-trajectory one. $N_{\mathrm{p}}$, the candidate-pool size, is never stated, so the selection budget is unknown.

5. **The PDM-Closed row in Table 1 has three columns cyclically permuted** (TTC/C/EP printed as 89.9/86.9/99.9 against canonical 86.9/99.9/89.9). EP 99.9 exceeds the human driver's 87.5.

6. **DriveSuprim is in the v2 table but absent from the v1 table**, along with the entire current v1 frontier (CLEAR/DA-WAM 93.7, DriveSuprim 93.5, Drive-JEPA 93.3, WCog-VLA 92.9, HybridDriveVLA 92.1, WA-JEPA 91.8, DynVLA 91.7, SimWAM 91.5). The results section's "best among the compared methods" is accurate; the introduction's framing is not.

7. **The v2 table mixes evaluator conventions** — pre-fix TransFuser/ARTEMIS/DriveVLA-W0 alongside corrected DiffusionDrive/DriveWorld-VLA, plus DriveSuprim 83.1 and Hydra-MDP++ 81.4 matching neither cohort. Sixth ingested table to do so. It also carries ARTEMIS EC = 98.3, a duplicate of that method's HC.

8. **EC 73.3 on v2 is the third-worst in its own table** and is not mentioned. Given EP 90.3 exceeds the human reference, the comfort regression is the expected other side of an aggressive planner and deserved a sentence.

9. **The nuScenes result is unspecified as to protocol.** Zero-shot or separately trained is not stated; ego status is consumed; 0.37 m leads by 0.03 m over methods with better collision rates.

10. **Stage-2 freezing is never ablated.** The VLM is frozen while the FCP trains, so foresight features are effectively cached — the rung of [[sources/adaptive-wam.md]]'s adaptation ladder that recovered only 0.75 of a 6.42-PDMS gap on a video backbone. [[sources/brainwam.md]] measured freezing as *helping* by 0.7 on a two-branch design. Both are different backbones; LWDrive supplies no measurement of its own.

11. **Unreported hyperparameters**: $N_{\mathrm{p}}$, $\beta$, $\lambda_{\mathrm{ref}}$, $\lambda_{\mathrm{score}}$, $\Delta$ (the future-frame offset), and the number of denoising steps in the world head. $\lambda_{\mathrm{wm}}=15.0$ against $\lambda_{\mathrm{traj}}=1.0$ is reported but never ablated, despite being a 15:1 weighting on the paper's central objective.

12. **No navhard, no Bench2Drive, no HUGSIM, no reactive evaluation.** For a method claiming future-aware representations, navhard is the conspicuous omission — [[sources/geowam.md]] remains the only wiki paper whose future-prediction gain is larger under the reactive protocol (+4.9 navhard vs +0.6 navtest).

13. **Four cameras for BEV, one 280×504 front view for the VLM.** The mismatch is unexplained: NAVSIM supplies eight cameras and wiki entries use one, three, four, or six. Nothing tests whether the fourth view matters, or whether the VLM's low input resolution limits the coarse plan.

14. **Text/table cross-references are wrong throughout** — the NAVSIM-v1 results paragraph says "reported in Table 2," and every footnote's citation list says "Cited by: Table 2" regardless of which table. Cosmetic, but it makes the paper harder to check.

15. **Code "will be made publicly available"** — not released at the time of ingest.

---

## Key Cross-References

- **Readout depth, third backbone family**: [[concepts/foundation-backbones-for-ad.md]] — LWDrive is the first measurement on a **VLM**, and the +0.2 result bounds how far [[sources/adaptive-wam.md]]'s +4.80 generalises. The sub-metric direction (multi-depth buys safety, costs progress) reproduces [[sources/geoworldad.md]]'s finding on a geometry decoder.
- **The same architecture, arrived at independently**: [[sources/geoworldad.md]] interleaves planner refinement stages with backbone depth on StreamVGGT ({4,11,17,23} × 5 stages, 64 proposals, simulator-distilled scorer); LWDrive does it on Qwen2.5-VL ({6,…,36} × 6 stages, $N_p$ proposals, simulator-distilled scorer). Neither cites the other. Their NAVSIM-v2 tables share every overlapping baseline row digit-for-digit, so **91.0/90.4 and 92.0/89.6 are unusually comparable** — GeoWorldAD is ahead on v2 by 0.8, behind on v1 by 1.0.
- **Training-time-only world modelling**: [[concepts/world-model-for-ad.md]] — LWDrive joins that camp with a VAE-latent future-frame objective weighted 15:1 against the trajectory loss, and contributes the wiki's first ablation of such an objective *on a VLM's hidden states* (+1.8), with the caveat that it is measured pre-refinement.
- **Selection at the top of the leaderboard**: [[concepts/selection-based-planning.md]] — with LWDrive added, **all eight of the highest non-BoN NAVSIM-v1 entries in this wiki train a scorer against the benchmark's own metric** *(lint note, 2026-09-27: [[sources/drive-hwm.md]], ingested later, reports 93.8 in its tables and 93.3 in its prose with no scorer, no RL and no BoN, which breaks the generalization if either number stands)*: CLEAR 93.7, DA-WAM 93.7, DriveSuprim 93.5, Drive-JEPA 93.3, WCog-VLA 92.9, Adaptive-WAM (aux) 92.6, HybridDriveVLA 92.1, LWDrive 92.0. There is no entry above 92 in this wiki that does not.
- **RL is not required**: LWDrive reaches 92.0 with no GRPO, no RFT, and no preference optimisation — only SFT plus a simulator-distilled scorer. Compare [[sources/wcog-vla.md]], where RFT alone is +3.6 of a +8.5 total, and [[sources/sgdrive.md]], where RFT is +3.7. See [[concepts/rl-for-ad.md]].
- **Ego progress above the human reference**: EP 90.3 on v2 against a Human Agent at 87.4, with EC 73.3. The clearest progress-versus-comfort trade in the wiki's v2 records, and the mirror of [[sources/drivelaw.md]]'s NC 99.0 / EP 81.3 profile.
- **New methods for the gap list**: DriveX-S (ICCV 2025, 84.5), PRIX (2507.17596, 87.8 v1 / 84.2 v2), World4Drive (2507.00603, 85.1), InsightDrive (2503.13047), DiffRefiner (AAAI 2026), Doe-1 (2412.09627), RDA-Driver (2408.13890), DME-Driver (2401.03641). iPad 91.7, DriveWorld-VLA 91.3 / 86.8, and ReCogDrive 90.8 are corroborated against existing wiki values.
