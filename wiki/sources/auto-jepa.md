---
title: "Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving"
type: source-summary
sources: [raw/papers/Auto-JEPA_ A Latent World Model of Continuous Intent for End-to-End Autonomous Driving.md]
related: [sources/wa-jepa.md, concepts/world-model-for-ad.md, concepts/selection-based-planning.md, concepts/intent-conditioned-planning.md, concepts/navsim-benchmark.md, concepts/perception-for-planning.md, concepts/counterfactual-prediction.md, concepts/foundation-backbones-for-ad.md, sources/drive-jepa.md, sources/latent-wam.md, sources/deepsight.md, sources/drivesuprim.md, sources/simwam.md, sources/drivelaw.md, sources/sgdrive.md]
created: 2026-09-02
updated: 2026-09-02
confidence: high
---

# Auto-JEPA

Auto-JEPA is a latent world model whose prediction target is **the ego trajectory itself**, not the scene. A frozen V-JEPA 2 encoder plus a 24-layer Transformer predictor maps (four front-camera frames, four historical ego positions, route command) to eight latent tokens that are aligned with the frozen encoding of the ground-truth future trajectory. That predicted "intent" is then used at inference as a **retrieval key** into a fixed memory of 110,335 recorded trajectories; a scene-conditioned scorer and a drivable-area gate pick the final candidate.

The paper's argument is that planning does not require reconstructing the future world — only the parts of it that change the ego action — and that supervising on future *ego motion* is enough to make the model attend to those parts. Its evidence for the second claim is a controlled semantic-occlusion study rather than an ablation.

**Source**: `raw/papers/Auto-JEPA_ A Latent World Model of Continuous Intent for End-to-End Autonomous Driving.md`
**arXiv**: https://arxiv.org/html/2607.29031v1
**Code**: https://github.com/NoctYang/Auto-JEPA
**Authors**: Jiwei Yang, Zhengxian Chen, Chaosheng Huang, Jun Li (School of Vehicle and Mobility, Tsinghua University; State Key Laboratory of Intelligent Green Vehicle and Mobility)

> **Naming warning**: Auto-JEPA is a *different paper* from [[sources/drive-jepa.md]], despite both using V-JEPA 2 on NAVSIM. Drive-JEPA re-pretrains the encoder on 208 h of driving video and then trains a proposal-centric planner; Auto-JEPA keeps the off-the-shelf V-JEPA 2 encoder frozen and applies the JEPA *objective* to a trajectory latent space. See [Relationships](#relationships).

## Key Takeaways

- **The world-model target is the ego trajectory latent.** Every other world model in this wiki predicts something about the scene — pixels, video latents, DINO features, occupancy, BEV state. Auto-JEPA predicts an encoding of what the ego will *do*. Scene dynamics are retained only through their implications for that motion.
- **The predicted latent is load-bearing at inference**, unlike the training-time-only latent world models ([[sources/latent-wam.md]], FLARE, Drive-JEPA). It *is* the query. This makes Auto-JEPA a third position in the [[concepts/world-model-for-ad.md#test-time-imagination]] debate: prediction runs at decision time and matters, but what is predicted is not a future world state.
- **No trajectory generator anywhere in the system.** Candidates come from recorded ground-truth geometry; nothing is regressed, denoised, or decoded into waypoints at inference. The trajectory decoder used to learn the latent space is discarded after stage 1.
- **91.3 PDMS on NAVSIM v1 navtest with one front camera at 256×256**, a frozen visual encoder, and no perception labels of any kind. Only the history encoder, command encoder, JEPA predictor, scorer, and gate are trained.
- **The intent predictor alone is worth 87.6 PDMS**; the CLOVER-initialized scene scorer adds +3.7 and the drivable-area gate +0.3. Read carefully, the headline number is a retrieval system plus a strong inherited scorer, not the JEPA component alone.
- **Semantic occlusion is the paper's most original contribution.** Masking dynamic-agent regions across all four input frames changes the predicted intent 2.97× as much as equal-area random masks (mean $1-\cos$ of 0.080 vs. 0.027 over 15,364 validation scenes), larger in 71.1% of them. The model was never given boxes, agent identities, or interaction labels.
- **NAVSIM v2 is protocol-split**: 85.6 EPDMS under the original evaluator, 89.1 under the updated official implementation with human-behavior filtering. The two are not interchangeable; the first is comparable to the source-reported baselines in the paper's own table, and the second belongs to the corrected-protocol cohort later mapped out by [[sources/wa-jepa.md]], where it ranks ninth.

## Method

The pipeline runs in four separately optimized stages, but the deployed interface is still sensor-to-trajectory with no intermediate perception output.

### Stage 1: Trajectory Latent-Space Pretraining

A trajectory autoencoder learns the target space before any visual training happens. The future ego trajectory is eight planar waypoints over a 4 s horizon at 0.5 s intervals:

$$
\mathbf{Y}=[(x_{1},y_{1}),\ldots,(x_{8},y_{8})]\in\mathbf{R}^{8\times 2}
$$

Coordinates are normalized by a scale factor of 64. The encoder $E_\mathrm{traj}$ is four Transformer blocks (1024-d, 16 heads, MLP ratio 4, eight Fourier frequency bands for coordinate encoding) producing eight temporally aligned latent tokens; the decoder is four self-attention blocks predicting waypoint *increments* that are cumulatively summed.

$$
\mathbf{Z}^{+}=E_{\mathrm{traj}}(\mathbf{Y})\in\mathbf{R}^{8\times 1024},\qquad\hat{\mathbf{Y}}=D_{\mathrm{traj}}(\mathbf{Z}^{+})
$$

$$
\mathcal{L}_{\mathrm{traj}}=\mathcal{L}_{xy}+2.0\,\mathcal{L}_{\mathrm{end}}+0.5\,\mathcal{L}_{\mathrm{vel}}+0.2\,\mathcal{L}_{\mathrm{acc}}
$$

The four terms supervise waypoint coordinates, the final endpoint, velocity, and acceleration. **The decoder is then discarded and the encoder frozen.** This is the design decision the rest of the system depends on: the same frozen encoder defines the prediction target *and* encodes every memory entry, so predicted intents and executable trajectories live in one space and cosine similarity is meaningful across them.

The paper is explicit that the eight tokens describe **one continuous future realization**, not eight maneuver classes or eight queries — a point worth holding against the multi-anchor and vocabulary methods in [[concepts/selection-based-planning.md]].

### Stage 2: Visual Intent Prediction

Input is $\mathbf{X}=(\mathbf{I},\mathbf{H},\mathbf{C})$: four front-camera frames at 256×256, four historical ego positions $\mathbf{H}\in\mathbf{R}^{4\times2}$, and a 4-d route command. A **frozen V-JEPA 2** encoder extracts visual tokens; small history and command encoders project the rest to 1024-d. A 24-layer, 16-head, 1024-d Transformer predictor fuses them with eight learnable future-time query tokens:

$$
\hat{\mathbf{Z}}=P_{\theta}(\mathbf{F}_{v},\mathbf{F}_{h},\mathbf{F}_{c})\in\mathbf{R}^{8\times 1024}
$$

Augmentation is temporal frame masking (p=0.3, one to three frames) and random erasing (p=0.2); when a frame is masked its paired ego position is masked too, which is what prevents the predictor from reading motion history as a shortcut.

### Joint-Embedding Objectives

No ADE/FDE supervision anywhere. Three terms operate in the frozen latent space:

**Feature alignment** — Smooth L1 on normalized latents:
$$
\mathcal{L}_{\mathrm{feat}}=\mathrm{SmoothL1}\!\left(\mathrm{Norm}(\hat{\mathbf{Z}}),\mathrm{Norm}(\mathbf{Z}^{+})\right)
$$

**Token-wise cosine alignment** — per future time step:
$$
\mathcal{L}_{\mathrm{cos}}=\frac{1}{8}\sum_{t=1}^{8}\left(1-\frac{\hat{\mathbf{z}}_{t}^{\top}\mathbf{z}^{+}_{t}}{\|\hat{\mathbf{z}}_{t}\|_{2}\|\mathbf{z}^{+}_{t}\|_{2}}\right)
$$

**Batch-level InfoNCE** — the anti-collapse term. Latent sequences are flattened and normalized, each scene's own target is the positive, other scenes' targets are negatives, $\tau=0.07$, targets gathered across GPUs to enlarge the negative set:
$$
\mathcal{L}_{\mathrm{NCE}}=-\frac{1}{B}\sum_{i=1}^{B}\log\frac{\exp(\hat{\mathbf{q}}_{i}^{\top}\mathbf{k}_{i}/\tau)}{\sum_{j=1}^{B}\exp(\hat{\mathbf{q}}_{i}^{\top}\mathbf{k}_{j}/\tau)}
$$

$$
\mathcal{L}_{\mathrm{intent}}=0.1\,\mathcal{L}_{\mathrm{feat}}+2.0\,\mathcal{L}_{\mathrm{cos}}+\mathcal{L}_{\mathrm{NCE}}
$$

The stated reason for InfoNCE is that "positive alignment alone may map distinct driving scenes to similar representations." This is the JEPA collapse problem in its retrieval-specific form: an alignment-only objective could satisfy itself by predicting the dataset-mean intent, which would be a catastrophic retrieval key even though the alignment loss looks healthy. Note the practical consequence — retrieval quality depends on a **batch-level** discriminative signal, so the effective negative set is bounded by batch size × GPU count (8/GPU here, 1–2 GPUs).

### Stage 3: Non-Parametric Trajectory Retrieval

The memory holds $N=110{,}335$ ground-truth trajectories from NAVSIM training data, each encoded once by the same frozen $E_\mathrm{traj}$ and stored with its waypoints:

$$
\mathcal{M}=\left\{(\mathbf{Z}_{n},\mathbf{Y}_{n})\right\}_{n=1}^{N}
$$

Query and memory latents are flattened and $\ell_2$-normalized; retrieval is flat cosine similarity $r_{n}=\mathbf{q}^{\top}\mathbf{m}_{n}$ over the whole memory, keeping $K=300$. navtest scenes are excluded from memory construction.

"Flat" matters: the eight tokens are concatenated before normalization rather than compared token-by-token, so a single similarity score covers the whole 4 s realization and cannot trade early-horizon match against late-horizon mismatch.

### Stage 4: Scene Scoring and Feasibility Gating

Latent similarity measures intent compatibility, not safety. Two independently trained modules follow.

**Scene-conditioned utility branch.** $s_{k}=S_{\phi}(\mathbf{F}_{\mathrm{scene}},\mathbf{e},\mathbf{Y}_{k})$, **initialized from the publicly released CLOVER trajectory scorer** and re-optimized on Auto-JEPA's ground-truth-only retrieval distribution. Supervision is collision, drivable-area, TTC, comfort, and ego-progress labels plus a within-scene ranking objective:

$$
\mathcal{L}_{\mathrm{score}}=\mathcal{L}_{\mathrm{comp}}+0.5\,\mathcal{L}_{\mathrm{rank}}
$$

Ranking temperature 0.05; candidates within 0.02 of the best target score count as near-optimal; comfort-failure candidates get weight 5.0. Labels are generated offline from the NAVSIM training metric cache via the NAVSIM/CLOVER `get_sub_score` evaluator on the batched `navsim_v1_style` relabeling path with per-proposal two-way rollout **disabled**. No gradients reach the visual encoder, predictor, or memory.

**Drivable-area feasibility gate.** $p^{\mathrm{DAC}}_{k}=G_{\psi}(\mathbf{F}_{\mathrm{scene}},\mathbf{e},\mathbf{Y}_{k})$ predicts DAC-failure probability from frozen candidate features, seven kinematic features (measured ego speed, first two candidate speeds, signed and absolute initial speed mismatch, two finite-difference acceleration terms), and candidate-set context via self-attention across the 300 proposals — so the gate can compare a proposal against its alternatives rather than judging it in isolation. Trunk is 256-d with dropout 0.1.

$$
\mathcal{L}_{\mathrm{gate}}=\mathcal{L}_{\mathrm{BCE}}+0.3\,\mathcal{L}_{\mathrm{rank}}
$$

DAC-failing candidates get positive-class weight 8; ranking margin 1.0. At inference $m_{k}=\mathbf{1}[p^{\mathrm{DAC}}_{k}\leq 0.2]$ masks candidates before a masked argmax:

$$
k^{*}=\arg\max_{k:m_{k}=1}s_{k},\qquad\mathbf{Y}^{*}=\mathbf{Y}_{k^{*}}
$$

If every candidate is rejected the system falls back to ungated utility ranking. Evaluator labels are never available at inference.

## Figures

![[intro_selective_future_intent.png]]

**Figure 1.** Selective response to action-relevant scene information. In the same scene, occluding a non-interacting adjacent vehicle leaves the predicted intent and selected trajectory essentially unchanged, while occluding the interacting lead vehicle shifts both. Annotations are used only for analysis, never as model input.

![[trajectory_latent_and_intent_prediction.png]]

**Figure 2.** The two learning stages. Stage 1 trains the trajectory autoencoder to define the future-trajectory target space, then discards the decoder and freezes the encoder. Stage 2 predicts the continuous driving-intent latent from visual observations, ego-motion history, and navigation command, aligning it with the frozen target representation.

![[intent_jepa_overview.png]]

**Figure 3.** Full pipeline. Training aligns the predicted latent with the frozen encoding of the ground-truth future trajectory. Inference retrieves 300 candidates from the ground-truth-only latent memory, ranks them with the scene-conditioned scorer, and filters drivable-area violations with the independent gate. Snowflakes mark frozen modules — the visual encoder, the trajectory encoder, and the memory.

![[selective_intent_three_scene_compact.png]]

**Figure 4.** Per-vehicle occlusion across three scenes. Cyan marks the lower-impact vehicle, rose the higher-impact one. Curves show deviation from the unoccluded trajectory on a shared 0–4 m scale, with terminal deviations $\Delta p_T$ listed in the same order. Differences grow with the horizon in the open-road and lead-interaction scenes; the stop-and-go scene shows a smaller shift because the unoccluded plan advances only 0.17 m, leaving little room for deviation.

![[semantic_occlusion_three_scene_panels.png]]

**Figure 5.** Representative controls from the full-validation semantic occlusion protocol. Dynamic-agent regions and independently sampled equal-area random regions are masked consistently across all four input frames; bars report cosine similarity to the unoccluded intent.

## Tables

### Table 1: NAVSIM v1 navtest

C and L denote camera and LiDAR input. NC = no-at-fault collision, DAC = drivable-area compliance, TTC = time to collision, C = comfort, EP = ego progress.

| Method | Venue | Sensors | NC ↑ | DAC ↑ | TTC ↑ | C ↑ | EP ↑ | PDMS ↑ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Human | – | – | 100.0 | 100.0 | 100.0 | 99.9 | 87.5 | 94.8 |
| *End-to-End Planning Methods* | | | | | | | | |
| TransFuser | TPAMI 2023 | 3× C+L | 97.7 | 92.8 | 92.8 | 100.0 | 79.2 | 84.0 |
| PARA-Drive | CVPR 2024 | 6× C | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| Hydra-MDP | CVPR 2024 | 3× C+L | 98.3 | 96.0 | 94.6 | 100.0 | 78.7 | 86.5 |
| DiffusionDrive | CVPR 2025 | 3× C+L | 98.2 | 96.2 | 94.7 | 100.0 | 82.2 | 88.1 |
| *World-Model-Based Methods* | | | | | | | | |
| LAW | ICLR 2025 | 1× C | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 |
| DrivingGPT | ICCV 2025 | 1× C | 98.9 | 90.7 | 94.9 | 95.6 | 79.7 | 82.4 |
| WoTE | ICCV 2025 | 3× C+L | 98.5 | 96.8 | 94.4 | 99.9 | 81.9 | 88.3 |
| Epona | ICCV 2025 | 3× C | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 |
| *VLA-Based Methods* | | | | | | | | |
| AutoVLA | NeurIPS 2025 | 3× C | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 |
| RecogDrive | ICLR 2026 | 3× C | 98.2 | 97.8 | 95.2 | 99.8 | 83.5 | 89.6 |
| AdaThinkDrive | ICRA 2026 | 1× C | 98.4 | 97.8 | 95.2 | 100.0 | 84.4 | 90.3 |
| DriveVLA-W0 | ICLR 2026 | 1× C | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.2 |
| Curious-VLA | CVPR 2026 Findings | 1× C | 98.4 | 96.9 | 97.9 | 98.1 | 88.5 | 90.3 |
| **Auto-JEPA (Ours)** | – | **1× C** | 98.4 | 98.3 | 95.0 | **100.0** | **87.1** | **91.3** |

Auto-JEPA's EP of 87.1 is second only to Curious-VLA's 88.5 in this table and close to the human 87.5 — unusual for a method with no learned generator, and attributable to the memory being built from human trajectories.

### Table 2: NAVSIM v2

Where multiple backbones are reported, the strongest source-reported configuration is used. The unmarked Auto-JEPA row uses the original evaluation implementation; the † row uses the updated official implementation. All other methods are source-reported.

| Method | NC ↑ | DAC ↑ | DDC ↑ | TL ↑ | EP ↑ | TTC ↑ | LK ↑ | HC ↑ | EC ↑ | EPDMS ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *End-to-End Planning Methods* | | | | | | | | | | |
| TransFuser | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| VADv2 | 97.3 | 91.7 | 98.2 | 99.9 | 77.6 | 92.7 | 66.0 | 100.0 | 97.4 | 76.6 |
| DiffusionDrive | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| HydraMDP++ (ViT-L) | 98.5 | 98.5 | 99.5 | 99.7 | 87.4 | 97.9 | 95.8 | 98.2 | 75.7 | 85.6 |
| DriveSuprim (ViT-L) | 98.4 | 98.6 | 99.6 | 99.8 | 90.5 | 97.8 | 97.0 | 98.3 | 78.6 | 87.1 |
| *VLA-Based Methods* | | | | | | | | | | |
| DriveVLA-W0 | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 |
| ReCogDrive | 98.3 | 95.2 | 99.5 | 99.8 | 87.1 | 97.5 | 96.6 | 98.3 | 86.5 | 83.6 |
| Curious-VLA | 98.4 | 96.9 | 99.2 | 99.8 | 88.5 | 97.9 | 96.9 | 98.1 | 81.5 | 85.3 |
| DriveWorld-VLA | 98.6 | 99.1 | 99.6 | 99.8 | 87.4 | 97.9 | 97.0 | 97.8 | 78.6 | 86.8 |
| Auto-JEPA (Ours) | 98.5 | 98.7 | 98.2 | 97.2 | 90.5 | 97.9 | 84.0 | 97.8 | 75.4 | 85.6 |
| **Auto-JEPA (Ours)** † | 98.5 | 98.7 | 98.3 | **99.7** | **90.5** | 97.9 | **94.7** | 97.8 | 75.2 | **89.1** |

The evaluator change moves TL 97.2 → 99.7 and LK 84.0 → 94.7 while every other submetric stays within 0.1. A 3.5-point EPDMS swing from two submetrics is a property of the evaluator, not of the planner.

### Table 3: Component Ablation (NAVSIM v1 navtest, same Top-300 memory)

| Intent | Scorer | Gate | NC | DAC | TTC | C | EP | PDMS |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ✗ | ✓ | ✓ | 83.1 | 85.3 | 76.2 | 86.3 | 37.8 | 52.6 |
| ✓ | ✗ | ✓ | 98.1 | 96.4 | 94.0 | 100.0 | 81.7 | 87.6 |
| ✓ | ✓ | ✗ | 98.5 | 97.9 | 95.0 | 99.9 | 86.9 | 91.0 |
| ✓ | ✓ | ✓ | 98.4 | 98.3 | 95.0 | 100.0 | 87.1 | **91.3** |

Appendix D notes these rows must be read **conditionally, not as a sequential build-up**: the scorer-free row still has the gate, and the no-gate row still has the scorer. So the scorer is worth +3.7 PDMS *given* the gate, and the gate +0.3 PDMS (+0.4 DAC) *given* the scorer. The ✗-intent row replaces the predicted intent with a fixed codebook medoid — a scene-independent constant, which is why it collapses to 52.6.

### Table 4: Candidate-Pool Size Sensitivity

| Candidate pool size $K$ | Selected PDMS ↑ |
|---:|---:|
| 1 | 87.6 |
| 200 | 91.1 |
| 300 | **91.3** |

$K=1$ is pure intent retrieval with no selection at all — the nearest memory entry, returned directly. That it scores 87.6 is the cleanest measurement of what the JEPA component contributes on its own. Going to 200 buys +3.5; 200 → 300 buys +0.2, which the paper reads as approaching saturation under the current memory and predictor.

### Table 5: Final Hyperparameters

| Setting | Trajectory AE | Intent predictor |
|---|---|---|
| Epochs | up to 40 | 10 total |
| Batch size | 256/GPU | 8/GPU |
| Optimizer | AdamW | AdamW |
| Learning rate | $2\times10^{-4}$ | $10^{-5}$ |
| Weight decay | 0.05 | 0.05 |
| Dropout | 0.1 | 0.1 |
| Gradient clipping | 1.0 | 1.0 |
| Precision | BF16 | BF16 |
| Selection | lowest val ADE | epoch-10 checkpoint |

| Setting | Scene scorer | DAC gate |
|---|---|---|
| Epochs | 5 + 3 continuation | selected checkpoint |
| Batch size | 32 | 32 |
| Optimizer | AdamW | AdamW |
| Learning rate | $10^{-5}\rightarrow2\times10^{-6}$ | $3\times10^{-4}$ |
| Auxiliary LR (ego adapter) | $10^{-4}\rightarrow2\times10^{-5}$ | – |
| Weight decay | 0.01 | $10^{-3}$ |
| Dropout | pretrained | 0.1 |
| Gradient clipping | 1.0 | 1.0 |
| Precision | BF16 | BF16 |
| Selection | best val score | val recall/utility |

Distributed training is plain data parallelism, so effective global batch is 256/512 for trajectory pretraining and 8/16 for the intent predictor on one or two GPUs. Scorer and gate training are single-GPU because candidate features are precomputed. Hardware is A100-SXM4 80 GB; Python 3.12.

## Semantic Occlusion Protocol

This is the part of the paper worth reading even if the benchmark numbers date quickly, and the methodology is written up in [[concepts/perception-for-planning.md]].

The intervention: for every valid scene in the complete validation split, form a dynamic-agent mask from the projected regions of visible traffic participants and apply it **consistently to all four input frames**. The control masks an equal total image area with independently sampled random regions. Both preserve ego-motion history and navigation command, so the only thing varying is visual evidence. Response is measured as

$$
\Delta_{\mathrm{intent}}=1-\cos\left(\hat{\mathbf{Z}},\hat{\mathbf{Z}}_{m}\right)
$$

with the eight temporal tokens flattened first.

| Intervention | Mean $\Delta_\mathrm{intent}$ |
|---|---:|
| Dynamic-agent masking | 0.080 |
| Equal-area random masking | 0.027 |
| Ratio | **2.97×** |
| Larger response on dynamic-agent mask | **71.1% of scenes** |

$n = 15{,}364$ valid scenes; seed 42 for the random generator.

Two things make this more than a saliency plot. First, it is **paired and area-matched** — the standard failure of attention-map arguments is that the salient region is also the big or central one, and this design at least removes the area confound. Second, the model receives **no object boxes, agent identities, interaction labels, or surrounding-agent motion annotations** at any point, so the selectivity is a consequence of the training target, not of a supervised detector.

Figure 4 pushes further to individual vehicles: occluding a vehicle that affects future driving moves both the latent and the selected trajectory substantially more than occluding a non-influential one, and the gap widens with the prediction horizon.

## Relationships

- **[[sources/drive-jepa.md]]** — the confusable neighbor. Both use V-JEPA 2 on NAVSIM, and the distinction is worth stating precisely:

  | | Drive-JEPA | Auto-JEPA |
  |---|---|---|
  | What JEPA is applied to | Driving video (208 h curated) | The trajectory latent space |
  | Prediction target | Masked video representations | Frozen encoding of the GT future trajectory |
  | Encoder | Re-pretrained ViT-L | Off-the-shelf V-JEPA 2, frozen |
  | Role of prediction | Training-time representation shaping | Inference-time retrieval key |
  | Trajectory source | 32 refined online proposals | Retrieved GT geometry, no generator |
  | NAVSIM v1 | 93.3 PDMS | 91.3 PDMS |

  Drive-JEPA invests in the encoder and pays with a heavy pretraining stage; Auto-JEPA invests in the target space and pays with the memory's coverage limits.

- **[[sources/latent-wam.md]]** — the closest match on philosophy (compact latent prediction, no pixel decoder, no VLM) and the clearest contrast on target. Latent-WAM predicts future *world status* tokens and uses geometric distillation from WorldMirror to keep lane/drivable-area structure; Auto-JEPA predicts future *ego motion* and keeps scene structure only insofar as it changes that motion. Latent-WAM also deploys its predictor differently: the latent dynamics shape training, then a trajectory decoder runs at inference.

- **[[sources/drivesuprim.md]]** — the strongest fixed-vocabulary selector in the wiki (93.5 PDMS). Both narrow a large candidate set to a small one and score it, but the narrowing mechanism differs fundamentally: DriveSuprim's Stage 1 is a *learned coarse scorer* over 8192 fixed clusters, Auto-JEPA's is *cosine retrieval in a learned latent space* over 110,335 recorded trajectories. Auto-JEPA's memory is ~13× larger and un-clustered, so the geometry is real rather than a centroid.

- **[[sources/sgdrive.md]]** — both argue for planning-relevant rather than exhaustive scene modeling, from opposite directions. SGDrive *supervises* relevance explicitly (safety-critical boxes chosen by proximity to the ego trajectory, occupancy, goal pose) and needs 3D annotation to do it; Auto-JEPA never names an agent and lets the ego-motion target induce relevance, then measures the result by occlusion. SGDrive's ablation says structured perception of the *present* is worth +2.5 PDMS and future forecasting only +0.8; Auto-JEPA has no present-perception term at all and gets its selectivity for free.

- **[[sources/simwam.md]] / [[sources/drivelaw.md]]** — the two papers that established that conditioning on a *generated future* does not help planning. Auto-JEPA is consistent with both and sharpens the statement: it runs a predictive model at inference and it does help (+34.7 PDMS over the constant-intent baseline; the whole system depends on it), but what it predicts is an action latent, not a world state. See [[concepts/world-model-for-ad.md#test-time-imagination]].

- **[[sources/deepsight.md]]** — the other paper predicting several future latents in one pass. DeepSight regresses DINOv3 features of five future BEV frames; Auto-JEPA regresses eight tokens of one trajectory. Both avoid autoregressive rollout, and both find the multi-token temporal structure carries the useful signal.

- **CLOVER** (arXiv 2605.15120, not ingested) — Auto-JEPA's scorer is *initialized from CLOVER's public checkpoint*, and the paper reports CLOVER at 90.4 EPDMS on NAVSIM v2 with a learned generator–scorer pipeline. This is now a high-priority gap: a component this load-bearing (+3.7 PDMS) should not be an un-ingested dependency.

- **[[sources/wa-jepa.md]]** — the third V-JEPA 2 paper in the wiki, and the one whose critique comes closest to landing on Auto-JEPA. WA-JEPA argues V-JEPA's deterministic regression "is insufficient for generating entirely unseen future tokens," and measures a 1.0-EPDMS penalty for using it on multi-view scene latents. Auto-JEPA's objective is also deterministic (alignment + cosine, with InfoNCE only as an anti-collapse term) — but its target is a **single ego trajectory**, far lower-entropy than a scene, so a conditional mean remains a usable retrieval key rather than a blur. Read together, the two suggest the operative variable is the entropy of the prediction target, not the objective in isolation. WA-JEPA also supplies the corrected/pre-fix EPDMS partition that reframes Auto-JEPA's two v2 numbers.

- **Intent terminology** — Auto-JEPA's "intent" is *not* the discrete maneuver variable of [[concepts/intent-conditioned-planning.md]] (DIAL's eight classes, PaIR-Drive's intention tokens). It is a continuous 8×1024 latent describing one specific future realization. It is closer to SGDrive's continuous goal pose, but richer: a full temporal trajectory encoding rather than a single terminal point. See that page's [Continuous Goal as Intent](../concepts/intent-conditioned-planning.md) section.

## Limitations

**Protocol and comparison**

- **The 89.1 EPDMS headline is not comparable to the table it appears in.** It requires the updated official NAVSIM v2 implementation *with human-behavior filtering enabled*, while every baseline row is source-reported under the older protocol. Under the matched evaluator Auto-JEPA scores **85.6** — tying HydraMDP++ and below DriveSuprim (87.1) and DriveWorld-VLA (86.8) in its own table. The paper does state this, but the abstract and conclusion lead with 89.1.

  **Refined after ingesting [[sources/wa-jepa.md]]**, which reports both columns across a whole leaderboard and identifies a corrected-protocol cohort. The right comparison for 89.1 is *within that cohort*, where it places ninth: WA-JEPA 91.7, Discrete-WAM 90.4, SparseDriveV2 90.1, CoWorld-VLA 90.0, DriveFuture 89.9, WAM-Diff 89.7, DriveFine 89.7, Latent-WAM 89.3, then Auto-JEPA 89.1. So 89.1 is not uninterpretable — it is a mid-pack corrected-protocol result, 2.6 below the current leader. See [[concepts/navsim-benchmark.md]].
- The entire 3.5-point gap between the two protocols comes from TL (97.2 → 99.7) and LK (84.0 → 94.7). LK 84.0 is a genuine outlier — 12 points below the next-worst camera method — and the paper does not explain what the original evaluator was measuring there or why a retrieval planner would be uniquely exposed to it.
- Table 1's comparison set omits the actual NAVSIM v1 frontier: DriveSuprim 93.5, Drive-JEPA 93.3, CLEAR 93.7, HybridDriveVLA 92.1, DynVLA 91.7, DriveFine 91.8, SimWAM 91.5, iPad, GoalFlow. Against the wiki's table 91.3 is a solid mid-frontier result, not a leader. This is the standard scope caveat tracked in [[concepts/navsim-benchmark.md]].
- EC 75.2/75.4 is near the bottom of the wiki's NAVSIM v2 entries — better only than DriveVLA-W0's 58.9 and HydraMDP++'s 75.7. Retrieval has no frame-to-frame continuity mechanism: consecutive frames can retrieve different memory entries, and nothing in the scorer or gate penalizes the jump. Drive-JEPA hit exactly this problem and needed a momentum-aware selector to fix it (EC 47.9 → 84.8); Auto-JEPA does not have one and does not discuss the issue.
- One deterministic run of one checkpoint, no seed variance, explicitly because full NAVSIM evaluation is expensive. Differences under ~0.5 PDMS in the ablations should not be read as real.

**Attribution of the gain**

- **The scorer is inherited, not built.** It is initialized from CLOVER's released checkpoint and contributes +3.7 of the 91.3. Pure intent retrieval is 87.6. The paper's claim is about the intent representation, but the headline number is substantially a statement about CLOVER's scorer applied to a new candidate distribution — and there is no row showing what CLOVER's scorer does on CLOVER's own candidates versus Auto-JEPA's.
- **"No perception annotations" is true but "no privileged supervision" is not.** The scorer and gate train on labels generated by the NAVSIM/CLOVER `get_sub_score` evaluator over the training metric cache — collision, DAC, TTC, comfort, EP. This is simulator-label distillation, the same privileged signal Hydra-MDP-style methods use, and it is where most of the safety behavior comes from. The annotation-free claim covers the intent predictor only.
- Label generation disables per-proposal two-way rollout, so training labels and evaluation labels come from different rollout protocols. The paper flags this but does not quantify the mismatch.
- The ✗-intent ablation row (52.6) is a weak control: it substitutes a *scene-independent constant*, so it demonstrates that scene conditioning matters at all rather than that *this* intent representation is good. A meaningful control would be retrieval keyed by ego history alone, by the command alone, or by a regressed trajectory encoded through the same frozen encoder. None is reported, so the value of the JEPA objective specifically — against, say, a Smooth-L1 waypoint regressor whose output is encoded and used as the query — is untested.
- Checkpoint selection for the intent predictor is "the completed epoch-10 checkpoint," not a validation criterion. Every other stage has one.

**Method-level ceilings**

- **The planner cannot produce a maneuver the memory does not contain.** The paper is candid about this: if no feasible trajectory is in the retrieved pool, neither the scorer nor the gate can synthesize one. The K=200 → 300 saturation (+0.2) suggests enlarging the pool is not the fix.
- Memory is built from NAVSIM training trajectories, so retrieval inherits that distribution — including the forward-heavy bias DriveSuprim documents (only ~8% of NAVSIM GT trajectories turn more than 30°). Auto-JEPA has no analogue of DriveSuprim's rotation augmentation and reports no turn-vs-straight breakdown, so turn performance is unmeasured.
- Memory cost and retrieval latency are not reported anywhere: 110,335 × 8 × 1024 float latents is ~3.6 GB at FP32 (~1.8 GB at BF16), and a flat cosine scan over the full memory runs per frame. There is no ANN index, no latency number, and no FPS — a conspicuous omission for a paper whose selling point is a lightweight planner. Every competing method in the wiki reports latency.
- The all-rejected fallback restores ungated ranking, but its firing rate is never reported. If the gate rejects all 300 candidates often, the gate's measured +0.3 PDMS is an average over two different systems.
- The gate threshold $\tau_\mathrm{DAC}=0.2$ is fixed, with no sensitivity sweep and no calibration analysis, despite the paper naming scorer calibration as a known failure source.

**Scope**

- NAVSIM only. No Bench2Drive, HUGSIM, navhard, nuScenes, or Waymo — so the open-loop, non-reactive, 4 s result carries all the weight. This matters more than usual for a retrieval planner: a 4 s non-reactive horizon is exactly the regime where a memory of recorded human trajectories should look best.
- 256×256 front-camera input, well below the 1024×256 used by TransFuser, HydraMDP++, DriveSuprim, and GoalFlow. This is a genuine efficiency claim, but it also means the visual evidence available for fine-grained decisions is limited, and there is no resolution ablation.
- By the paper's own admission the learned intent "does not provide the scene-level forecasts required by applications such as interactive simulation or counterfactual environment generation." The representation is planning-only by construction — the flip side of the efficiency argument.

**The occlusion study's own limits**

- The random control matches *area* but not *shape, contiguity, or position*. Dynamic-agent masks are object-shaped, road-level, and clustered near the vanishing point; the random control is described only as "independently sampled equal-area random regions," which may include sky and periphery. Some of the 2.97× is plausibly a road-region effect rather than an agent-semantics effect. A stronger control — equal-area masks placed on the drivable surface, or on static road furniture — is not run.
- 71.1% of scenes show the larger response, meaning **28.9% respond more to random masking**. That tail is not characterized, and it is where a selectivity claim would be falsified.
- Absolute magnitudes are small: $1-\cos$ of 0.080 corresponds to cosine similarity 0.92, so removing *every visible dynamic agent from all four frames* still leaves the intent latent 92% aligned with the unoccluded one. The relative ratio is the real finding; the model is not dramatically dependent on agents in absolute terms.
- The link from latent change to *behavioral* change is only shown qualitatively, on three hand-picked scenes in Figure 4. There is no dataset-level statistic connecting $\Delta_\mathrm{intent}$ to a change in the selected trajectory or in PDMS, which is what would establish that the selectivity is functionally load-bearing rather than a property of the embedding.
