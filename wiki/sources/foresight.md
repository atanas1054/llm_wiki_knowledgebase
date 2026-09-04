---
title: "See Tomorrow, Act Today: Foresight-Driven Autonomous Driving (ForeSight)"
type: source-summary
sources: [raw/papers/See Tomorrow, Act Today_ Foresight-Driven Autonomous Driving.md]
related: [sources/adaptive-wam.md, sources/brainwam.md, concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/nuscenes-waymo-evals.md, concepts/foundation-backbones-for-ad.md, concepts/perception-for-planning.md, sources/epona.md, sources/drivelaw.md, sources/simwam.md, sources/drivewam.md, sources/driveva.md, sources/policy-world-model.md, sources/drivevla-w0.md, sources/da-wam.md, sources/wa-jepa.md, sources/latent-wam.md, sources/dreameraD.md, sources/diffusiondrive.md, sources/recogdrive.md, sources/futuresightdrive.md, sources/geowam.md, sources/onedrive.md]
created: 2026-09-04
updated: 2026-09-04
confidence: high
---

**Paper**: See Tomorrow, Act Today: Foresight-Driven Autonomous Driving
**Authors**: Bozhou Zhang, Nan Song, Yuang Wang, Jiankang Deng, Xiatian Zhu, Li Zhang
**Orgs**: Fudan University (School of Data Science) + Shanghai Innovation Institute + Imperial College London + University of Surrey
**arXiv**: 2605.07195v1
**Code**: https://github.com/LogosRoboticsGroup/ForeSight

---

## Summary

ForeSight takes the most literal possible position in the wiki's central world-model dispute: it makes a **frozen 2.5B foundation video generator the planner's primary visual encoder**, runs it forward to an actual imagined future at inference, and conditions trajectory decoding on that future. Everything else — a 52M TransFuser current-frame encoder, a 21M action decoder — is explicitly framed as supplementary.

The result is **89.3 PDMS on NAVSIM navtest** at **900 ms inference**, of which **870 ms is the world model**.

That pairing is the paper's real contribution to this wiki, and it cuts against the paper's own thesis. ForeSight is the cleanest available measurement of what the imagine-then-act paradigm costs and buys, because it isolates both sides: Table 3 shows that adding the foundation world model with naive attention is worth **+0.3 PDMS** (86.8 → 87.1), with the remaining +2.2 coming from the machinery built to compress and route its output; and Table 5 shows that the denoising budget responsible for most of that 870 ms is worth **+1.3 PDMS from 25 to 100 steps**, with the last 25 steps worth **+0.1**.

Two secondary results are independently useful. **Table 5's direction directly opposes [[sources/drivelaw.md]]'s Table 6** — ForeSight finds more denoising monotonically better, DriveLaW finds the near-clean future catastrophic. And **Table 7** removes the current-frame encoder entirely, leaving a planner driven *only* by generated futures, which still scores 88.2.

---

## Core Idea: The World Model Is the Encoder, Not an Auxiliary

![[cmp.png|Three paradigms: reactive planning, world model as auxiliary component, and ForeSight's world-model-centric design]]

**Figure 1**: Paradigm comparison. (a) Reactive end-to-end planning based on current observations (VAD, UniAD, DiffusionDrive). (b) A lightweight world model used as an auxiliary component for alignment or simplified prediction (LAW, navigation-guided sparse representation, SeerDrive). (c) ForeSight: a foundation world model centric framework where future scene imagination drives action prediction.

The paper's taxonomy of prior work is the same three-way split [[sources/drivelaw.md]] drew, arriving at a different fourth option:

1. **Reactive perception-to-planning** — UniAD, VAD, DiffusionDrive. Conditions on history and present only.
2. **World model as auxiliary supervision** — LAW, World4Drive, and the navigation-guided sparse representation line. Future prediction shapes representations via reconstruction loss; the planner never reads a future.
3. **World model as simplified predictor** — WoTE, SeerDrive. A lightweight BEV world model acts as trajectory selector or provides coarse future features.

DriveLaW's fourth option was *read the generator's mid-denoising internals*. ForeSight's is **run the generator to completion and read the finished future**. These are the two extremes of the same axis, from the same year, and their ablations disagree — see [Denoising Steps](#denoising-steps) below.

---

## Method

![[pipeline.png|ForeSight pipeline: frozen foundation world model plus lightweight current encoder feeding a state-based action decoder through WM-QFormer and factorized attention]]

**Figure 2**: Overview of ForeSight. A foundation world model is introduced into an end-to-end planning framework with current-frame features as an additional supplement. A WM-QFormer compresses future features with a set of frame queries and adapts them to the action head. State queries explicitly represent time steps and factorized attention handles feature interaction.

### WM encoder (the primary source)

The world model is inherited unchanged from an existing foundation model — the paper restricts itself to diffusion-based ones (Vista, Epona, Drive-WM). It takes current-frame images conditioned on motion attributes (yaws, poses) or commands, and at a **selected denoising step $t_{\rm d}$** the latent features are sampled as the future visual representation:

$$F_{\rm wm}={\rm WM}^{(t_{\rm d})}(\mathcal{I},F_{\rm cond}),\qquad F_{\rm wm}\in\mathbb{R}^{T_{\rm wm}\times C_{\rm wm}\times H\times W}$$

$t_{\rm d}$ is described as adjustable to trade efficiency for performance. **Its value is never reported**, which is a problem for interpreting Table 5 (see Limitations).

### Current encoder (the supplement)

A lightweight TransFuser-style Transformer over multi-view images, LiDAR, and ego status:

$$F_{\rm cur}={\rm Encoder}(\mathcal{I},\mathcal{P},\mathcal{E})$$

The stated justification is concrete and worth recording: **most foundation world models generate front-view only**, so a planner reading only generated futures is blind to the sides. NAVSIM uses 3 camera views and nuScenes 6, while Epona and Vista both produce a single forward view.

### State-based interactive decoding

**Time state queries** $Q_{\rm s}\in\mathbb{R}^{M\times T_{\rm f}\times C}$ — $M$ planning modes × $T_{\rm f}$ future steps — following the authors' own BridgeAD. Each query is bound to one future timestep, which is what makes temporal alignment with the world model's per-frame outputs possible at all.

**WM-QFormer** is a spatiotemporal Transformer with $N_{\rm wm}$ learnable queries per frame, compressing $F_{\rm wm}$ to $F'_{\rm wm}\in\mathbb{R}^{T_{\rm wm}\times N_{\rm wm}\times C}$. The paper's rationale is explicitly a *denoising* one: generated frames "contain abundant fine-grained textures and noise, which, if directly exposed to trajectory queries, may introduce interference into the planning process." This is the same diagnosis DriveLaW gave for its t=10 collapse — ForeSight's answer is to filter the finished future rather than to read an earlier one.

**Factorized attention** splits present and future into two cross-attentions:

$$\begin{split}Q_{\rm s}&={\rm CrossAttn}(Q_{\rm s},F_{\rm cur}),\\ Q_{\rm s}&={\rm CrossAttn}(Q_{\rm s}+E_{\rm s},F^{\prime}_{\rm wm}+E_{\rm wm}),\end{split}$$

then $\mathcal{T}={\rm TrajDecoder}(Q_{\rm s})$. Every state query sees the whole present; time embeddings $E_{\rm s},E_{\rm wm}$ bias each query toward its temporally adjacent future frames. Sinusoidal positional embeddings are used so the two sequences can have different lengths ($T_{\rm wm}\neq T_{\rm f}$ in general, though both are set equal in the experiments).

### Two-phase training

1. **Action pretraining** — current encoder + action decoder only, no WM, no WM-QFormer. 80 epochs on NAVSIM, 12 on nuScenes.
2. **Post-training** — WM features and WM-QFormer introduced; all components jointly optimized **except the world model, which stays fully frozen**. 20 epochs NAVSIM, 6 nuScenes.

The stated motivation is stability: joint training from scratch is unstable because of "the imbalance in representational capacity between the two feature sources" — a 2.5B pretrained generator against a 52M encoder trained from scratch.

Loss is unchanged across both phases:

$$\mathcal{L}=\lambda_{1}\mathcal{L}_{\rm bev}+\lambda_{2}\mathcal{L}_{\rm traj}$$

with $\lambda_1,\lambda_2$ unreported.

---

## Results

### Table 1 — NAVSIM navtest (PDMS, closed-loop metrics)

| Type | Method | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| *Planning model* | UniAD | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| | PARA-Drive | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| | TransFuser | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| | DRAMA | 98.0 | 93.1 | 94.8 | 100 | 80.1 | 85.5 |
| | Hydra-MDP++ | 97.6 | 96.0 | 93.1 | 100 | 80.4 | 86.6 |
| | DiffusionDrive | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| | Hydra-NeXt | 98.1 | 97.7 | 94.6 | 100 | 81.8 | 88.6 |
| | GoalFlow | 98.4 | **98.3** | 94.6 | 100 | 85.0 | 90.3 |
| | ReCogDrive | 97.9 | 97.3 | 94.9 | 100 | **87.3** | **90.8** |
| *World model* | DrivingGPT | **98.9** | 90.7 | 94.9 | 95.6 | 79.7 | 82.4 |
| | Epona | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 |
| *Planning with WM* | LAW | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 |
| | World4Drive | 97.4 | 94.3 | 92.8 | 100 | 79.9 | 85.1 |
| | WoTE | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| | SeerDrive | 98.4 | 97.0 | 94.9 | 99.9 | 83.2 | 88.9 |
| | **ForeSight (Ours)** | 98.8 | 97.2 | 94.8 | 100 | 83.5 | **89.3** |

**Protocol note**: every baseline row matches this wiki's canonical NAVSIM-v1 values (UniAD 83.4, TransFuser 84.0, DiffusionDrive 88.1, Epona 86.2, LAW 84.6, WoTE 88.3). ForeSight is protocol-clean on v1 — worth stating explicitly given the [evaluator-drift](../concepts/navsim-benchmark.md#three-protocols) problems documented elsewhere in this wiki.

**Margin within its own category**: +0.4 over SeerDrive, +1.0 over WoTE. Both are the "Simple WM" methods Table 4 argues against.

### Table 2 — nuScenes validation (open-loop, ResNet-50 backbone except UniAD/R101)

| Type | Method | L2 1s | L2 2s | L2 3s | **Avg ↓** | Col 1s | Col 2s | Col 3s | **Avg ↓** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| *Planning model* | BEV-Planner | 0.28 | **0.42** | **0.68** | **0.46** | **0.04** | 0.37 | 1.07 | 0.49 |
| | PARA-Drive | 0.25 | 0.46 | 0.74 | 0.48 | 0.14 | 0.23 | 0.39 | 0.25 |
| | VAD-Base | 0.41 | 0.70 | 1.05 | 0.72 | 0.07 | 0.17 | 0.41 | 0.22 |
| | GenAD | 0.28 | 0.49 | 0.78 | 0.52 | 0.08 | 0.14 | 0.34 | 0.19 |
| | UniAD | 0.44 | 0.67 | 0.96 | 0.69 | 0.04 | 0.08 | 0.23 | 0.12 |
| | BridgeAD | 0.29 | 0.57 | 0.92 | 0.59 | **0.01** | **0.05** | 0.22 | 0.09 |
| | MomAD | 0.31 | 0.57 | 0.91 | 0.60 | **0.01** | **0.05** | 0.22 | 0.09 |
| | SparseDrive | 0.29 | 0.58 | 0.96 | 0.61 | **0.01** | **0.05** | **0.18** | **0.08** |
| *Planning with WM* | LAW | 0.26 | 0.57 | 1.01 | 0.61 | 0.14 | 0.21 | 0.54 | 0.30 |
| | World4Drive | **0.23** | 0.47 | 0.81 | 0.50 | 0.02 | 0.12 | 0.33 | 0.16 |
| | **ForeSight (Ours)** | 0.36 | 0.55 | 0.93 | 0.62 | 0.04 | 0.12 | 0.37 | 0.18 |

**ForeSight does not win a single column here.** It is beaten on average L2 by seven of ten baselines including both other world-model methods, and on average collision by five. The paper's characterization — "our method demonstrates competitive performance" — is accurate but is the weakest claim in the paper, and it is preceded by a hedge that nuScenes scenarios are "relatively simple" and its metrics "not entirely comprehensive" (citing the ego-status critique). See [[concepts/nuscenes-waymo-evals.md]].

### Table 6 — Generation quality after 2 Hz finetuning (nuPlan)

| Method | FVD₁₀ ↓ |
|---|---:|
| Epona | **50.77** |
| ForeSight (finetuned Epona) | 54.63 |

The world model is finetuned from Epona's native 5 Hz to NAVSIM's 2 Hz. The paper reads this as "retains nearly the same generation capability"; the number is a **3.86 FVD regression**. Not comparable to Epona's 82.8 nuScenes FVD recorded in [[sources/epona.md]] — different dataset, different clip length.

### Efficiency (Discussion 4)

| Component | Params | Latency (H100) |
|---|---:|---:|
| Foundation world model (Epona) | 2.5 B | ~870 ms |
| Current encoder (TransFuser) | 52 M | — |
| Action decoder (+ WM-QFormer) | 21 M | — |
| **Total** | **~2.57 B** | **900 ms** |

**96.7% of inference time is the world model.** For context in this wiki: [[sources/simwam.md]] reaches 91.5 PDMS at 518 ms by making future generation training-time-only; [[sources/diffusiondrive.md]] reaches 88.1 at 45 FPS (~22 ms); [[sources/onedrive.md]] runs 156 ms. ForeSight's 900 ms is the slowest NAVSIM planner recorded here.

---

## Ablations

### Table 3 — Components (NAVSIM navtest)

| ID | w. WM | WM-QFormer | State queries | Factorized attn | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|---:|:-:|:-:|:-:|:-:|---:|---:|---:|---:|---:|---:|
| 1 | | | | | 97.8 | 95.6 | 93.4 | 100 | 81.6 | 86.8 |
| 2 | ✓ | | | | 97.8 | 95.9 | 93.3 | 100 | 82.1 | **87.1** |
| 3 | ✓ | ✓ | | | 98.6 | 96.2 | **95.0** | 100 | 81.3 | 87.9 |
| 4 | ✓ | ✓ | ✓ | | 98.6 | 96.8 | **95.0** | 100 | 82.0 | 88.5 |
| 5 | ✓ | ✓ | | ✓ | 98.4 | 96.5 | **95.3** | 100 | 81.6 | 88.2 |
| 6 | ✓ | ✓ | ✓ | ✓ | **98.8** | **97.2** | 94.8 | 100 | **83.5** | **89.3** |

**Read row 1 → row 2 carefully.** Bolting a frozen 2.5B foundation world model onto the baseline and cross-attending to its output with vanilla attention buys **+0.3 PDMS**. The paper says this plainly ("a slight improvement... indicating that future features can benefit the planning process, but not in a straightforward manner") and treats it as motivation for the WM-QFormer.

The alternative reading, which the paper does not offer: **+0.3 is what the imagined future is worth on its own**, and the +2.2 above it is what a purpose-built compression-and-routing stack is worth. Rows 3–6 cannot separate these, because WM-QFormer, state queries, and factorized attention are all defined relative to WM features and none is ablated against a no-WM baseline. State queries in particular are a BridgeAD mechanism with no intrinsic world-model dependency, and it is never tested without one.

### Table 4 — Foundation vs. simplified world models

| | DAC ↑ | TTC ↑ | EP ↑ | PDMS ↑ |
|---|---:|---:|---:|---:|
| w/o WM | 95.6 | 93.4 | 81.6 | 86.8 |
| Simple WM (WoTE/SeerDrive-style) | 96.3 | 93.4 | 82.2 | 87.5 |
| **Found. WM (ours)** | **97.2** | **94.8** | **83.5** | **89.3** |

Argued as +1.8 for foundation over simplified. **But the "Simple WM" row is a reimplementation inside ForeSight's own pipeline scoring 87.5, while the actual published WoTE (88.3) and SeerDrive (88.9) both beat it — SeerDrive by 1.4.** Against published numbers the foundation world model is worth +0.4 over the best simplified one, at roughly 50× the parameters. The paper's own Table 1 contains both figures.

### Table 5 — Number of denoising steps {#denoising-steps}

| Steps | DAC ↑ | TTC ↑ | EP ↑ | PDMS ↑ |
|---:|---:|---:|---:|---:|
| 25 | 96.4 | **94.8** | 81.3 | 88.0 |
| 50 | 96.6 | **95.2** | 81.5 | 88.3 |
| 75 | **97.3** | 94.7 | **83.5** | 89.2 |
| 100 | 97.2 | 94.8 | **83.5** | **89.3** |

Monotone and saturating: +1.2 from 25→75, **+0.1 from 75→100**. The paper's own recommendation is 75 steps as the efficiency/accuracy sweet spot, while the headline configuration uses 100.

**This is the wiki's only result pointing *toward* imagine-then-act, and it is in direct tension with [[sources/drivelaw.md]]'s Table 6**, where conditioning on a near-clean generated future collapses the policy to 23.2 PDMS. The two experiments are not the same variable — DriveLaW fixes the schedule and moves the *extraction point* along it; ForeSight changes the *total schedule length* and (presumably) extracts at or near the end. But they bear on the same question, and they answer it oppositely: ForeSight says a better-formed future plans better, DriveLaW says a better-formed future plans worse. Full treatment in [[concepts/world-model-for-ad.md]].

Interpretation is hampered because **$t_{\rm d}$ is never given**, so it is not possible to tell whether the 25-step row extracts from an equivalent point on a shorter schedule or from a genuinely coarser latent.

### Table 7 — Removing the current encoder (NAVSIM navtest)

| | DAC ↑ | TTC ↑ | EP ↑ | PDMS ↑ |
|---|---:|---:|---:|---:|
| w/o Current | 96.3 | **95.4** | 81.7 | 88.2 |
| ForeSight | **97.2** | 94.8 | **83.5** | **89.3** |

This is the most interesting supplementary result. A planner driven **only by generated front-view futures**, with no multi-view images, no LiDAR, and no current-frame encoder at all, reaches 88.2 — matching WoTE (88.3) and beating the full pipeline's own no-WM baseline (86.8) by 1.4. The current encoder is worth +1.1, concentrated in DAC (+0.9) and EP (+1.8), which is what one would expect from restoring side views and LiDAR geometry.

The paper's forward-looking claim is that as world models gain multi-view and high-resolution generation, the current encoder can be dropped entirely. Note the trade is not free even now: removing it *improves* TTC (95.4 vs 94.8).

### Table 8 — Alternative world-model architecture (nuScenes)

| Method | L2 1s | L2 2s | L2 3s | Avg ↓ | Col 1s | Col 2s | Col 3s | Avg ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ForeSight-Vista | 0.42 | 0.63 | **0.88** | 0.64 | 0.08 | 0.22 | 0.51 | 0.27 |
| **ForeSight-Epona** | **0.36** | **0.55** | 0.93 | **0.62** | **0.04** | **0.12** | **0.37** | **0.18** |

Offered as evidence that the framework is architecture-agnostic. It is — but the substitution is **worse on 6 of 8 columns**, with average collision rate rising 50% (0.18 → 0.27). Architecture-agnosticism is demonstrated in the weak direction: the framework tolerates a different generator, it does not benefit from one.

---

## Qualitative Results and Failure Cases

![[visual.png|ForeSight planned trajectories in BEV alongside the 8 generated future frames]]

**Figure 3**: NAVSIM navtest. Left panel is the planned trajectory in BEV, right panel is the generated future video over the next 8 timesteps. Ground truth green, planned trajectory orange. (a) and (c) are interaction scenarios; (b) and (c) include turning behaviors.

![[visual_supp.png|Additional ForeSight qualitative results including congestion and fast driving]]

**Figure 4**: Additional results — turning in (a) and (c), traffic congestion in (b), fast driving in (d).

![[visual_fail.png|Two ForeSight failure cases: over-conservative decoding and long-range generation breakdown]]

**Figure 5**: Failure cases. Both are diagnostically valuable and the paper analyzes them honestly:

- **(a) The world model is right and the planner ignores it.** On a right turn the generator correctly predicts both the turning motion and the post-turn scene, but the action decoder produces "an overly conservative and slow trajectory." The paper's own conclusion: "the world model and the action model should be more tightly coupled so that the planner can better leverage future predictions." This is a direct admission that the frozen-WM + separately-pretrained-decoder design does not fully exploit the future it pays 870 ms to generate.
- **(b) The world model is wrong and the planner is fine anyway.** On a fast, highly winding road the generator degrades in the later frames as curvature increases, yet the trajectory stays accurate — which the paper credits to the current-frame encoder. Read against Table 7, this is the clearest statement of the actual division of labor: **the current encoder is the robustness floor and the world model is the upside**.

---

## Implementation Details

- **World model**: Epona (2.5B) for both NAVSIM and nuScenes; Vista for the nuScenes ablation. Epona is **finetuned on nuPlan from 5 Hz to 2 Hz**, then frozen for all pipeline training.
- **Current encoder**: TransFuser-based, 52M.
- **Inputs**: NAVSIM 3 cameras + LiDAR at 1024×256; nuScenes 6 cameras, images only, 640×360.
- **Horizons**: NAVSIM 4 s / 8 steps / 20 modes; nuScenes 3 s / 6 steps / 6 modes. Future frames generated = planning steps (8 and 6), conditioned on the current frame only.
- **Training**: 8× NVIDIA H100. NAVSIM batch 8, 80 pretrain + 20 post-train epochs. nuScenes batch 1, 12 + 6 epochs. lr 1e-4, AdamW.
- **Inference**: 100 denoising steps in the headline configuration (75 recommended).
- **Not reported**: $t_{\rm d}$, $\lambda_1$, $\lambda_2$, $N_{\rm wm}$, $C$, $C_{\rm wm}$, WM-QFormer depth and parameter count, FPS, seed variance.

---

## Limitations

1. **The headline mechanism is worth +0.3 PDMS.** Table 3 rows 1→2 add the entire 2.5B foundation world model under vanilla attention and gain 0.3. Everything above that comes from WM-QFormer, state queries, and factorized attention — none of which is ablated without the world model present, and one of which (state queries) is a general planning mechanism from the authors' own BridgeAD. The causal story ("future imagination drives action prediction") is not separable from "a better-designed action decoder" on this evidence.

2. **900 ms, of which 870 ms is the world model, for 89.3 PDMS.** [[sources/simwam.md]] reaches 91.5 at 518 ms with future generation removed at inference entirely. Under this wiki's numbers, spending 96.7% of the compute budget on test-time imagination is dominated by not doing it. The paper acknowledges the cost and defers to future improvements in world-model efficiency.

3. **Table 4's "Simple WM" comparison is against a reimplementation that underperforms the published methods.** Simple WM scores 87.5; WoTE and SeerDrive are 88.3 and 88.9 in ForeSight's own Table 1. The claimed +1.8 for foundation world models is +0.4 against real published simplified world models — at ~50× the parameters and far higher latency.

4. **nuScenes is a loss presented as a draw.** 0.62 avg L2 and 0.18 avg collision place ForeSight below World4Drive (0.50 / 0.16) — the other world-model entry in its own table — and below SparseDrive, BridgeAD, MomAD, PARA-Drive, and BEV-Planner on L2. Described as "competitive performance."

5. **Potential train/test overlap that is never addressed.** Epona is finetuned on **nuPlan**, and NAVSIM navtest is a **nuPlan subset**. The paper does not state whether navtest scenes were excluded from that finetuning. This is the same unexamined gap the wiki flagged for [[sources/wa-jepa.md]] and [[sources/geowam.md]], but sharper here, because the finetuned generator is the primary encoder rather than a pretraining initialization.

6. **The denoising-step ablation is uninterpretable without $t_{\rm d}$.** Table 5 varies the total schedule length while the extraction step is an unreported free parameter. Whether the 25-step row is a coarser latent or an equivalently-positioned latent on a shorter schedule changes what the result means — and changes whether it genuinely conflicts with DriveLaW's Table 6 or merely appears to.

7. **The generator gets worse at generating.** FVD 50.77 → 54.63 after the 2 Hz finetune. Reported as "nearly the same."

8. **The alternative-backbone experiment goes the wrong way.** ForeSight-Vista loses on 6 of 8 nuScenes columns and raises average collision by 50%. Architecture-agnosticism is shown as tolerance, not as transferable benefit.

9. **The paper's own failure case (a) is evidence against its thesis.** A correct imagined future paired with an over-conservative trajectory is exactly the outcome predicted by "the planner is not really using the future," and the proposed remedy — tighter coupling — concedes that the current design does not deliver on the framing.

10. **NAVSIM-v1 only.** No NAVSIM-v2 / EPDMS, no navhard, no Bench2Drive, no HUGSIM, no reactive closed loop. For a method whose entire claim is anticipation in "dynamic, interactive scenarios," the absence of any reactive benchmark is the most consequential evaluation gap. NAVSIM's non-reactive 4 s horizon is precisely where anticipation has least room to pay off.

11. **Single runs, no seed variance,** against ablation deltas as small as +0.1 (Table 5, 75→100 steps) and +0.3 (Table 3, the world model itself). WA-JEPA's measured 0.053 EPDMS seed std is the wiki's only reference point and training-seed variance is typically larger.

12. **No RL.** Acknowledged in the paper's own limitations as the natural next step, and it is a real gap: ReCogDrive (90.8) and GoalFlow (90.3), the two methods above ForeSight in its own Table 1, both exceed it partly through mechanisms ForeSight does not use.

---

## Key Cross-References

- **The central dispute**: [[concepts/world-model-for-ad.md]] — ForeSight is now the wiki's most committed imagine-then-act system and its clearest cost accounting. Its +0.3-for-the-world-model and 870-ms-of-900 results belong next to SimWAM's isolated mask and DA-WAM's shared-vs-per-candidate ablation.
- **The direct opponent**: [[sources/drivelaw.md]] — same benchmark, same year, opposite conclusion about how finished the imagined future should be. DriveLaW: 89.1 at t=1, 23.2 at t=10. ForeSight: 88.0 at 25 steps, 89.3 at 100.
- **The tiebreaker, against it**: [[sources/brainwam.md]] measures the same axis as DriveLaW in a third architecture and lands on DriveLaW's side — with decoupled video/action timesteps, **one video denoising step of three recovers 89.3 of an achievable 89.5 PDMS**, and steps 2-3 add 0.2 then nothing for 169 ms. Two of three papers that have measured denoising depth now say the planner wants an early, barely-formed latent. ForeSight is the outlier and also the only one that never reports its extraction step, so the live hypothesis is that its monotone sweep reflects a shifting extraction point rather than a real preference for finished futures. BrainWAM also reaches 89.3 at **475 ms on an H20** against ForeSight's 900 ms on an H100.
- **The two design choices measured against it**: [[sources/adaptive-wam.md]] varies both of ForeSight's defining decisions with everything else held fixed. **Freezing the generator costs 6.42 PDMS** (frozen Wan 84.20 vs joint LoRA 90.62; separately-tuned-then-cached features recover only 0.75 of that), which is a direct argument against using an unadapted Epona as the primary encoder. And it separates the **noise index** from the **readout depth**, finding the former worth <=0.15 PDMS across five indices of a 40-step schedule while the latter is worth 4.80 — so ForeSight's 25-to-100-step sweep is varying the axis that barely matters while never reporting {m d}$ or which layer it reads. Adaptive-WAM also profiles the alternative on one A100: **170 ms to plan from an intermediate feature, 13.22 s to run the full 40-step rollout**. Its architecture is not ForeSight's, so this is a strong prior rather than a refutation.
- **The efficient alternative**: [[sources/simwam.md]] — 91.5 PDMS at 518 ms with the future generated only during training. On PDMS per millisecond, SimWAM dominates ForeSight by a wide margin.
- **Shared world model**: [[sources/epona.md]] — Epona is ForeSight's generator, as it is [[sources/dreameraD.md]]'s. Three wiki papers now build on Epona and reach 86.2 (Epona itself), 88.7 (DreamerAD, latent RL), and 89.3 (ForeSight, generated-future conditioning).
- **Shared-future configuration**: [[sources/da-wam.md]] — ForeSight generates **one** future per scene and conditions all 20 trajectory modes on it. That is exactly DA-WAM's configuration (c), which its matched ablation measures at 0.50 PDMS *below* predicting no future at all. ForeSight's +0.3 for the same configuration is the closest thing to an independent check the wiki has, and the two are within each other's plausible noise.
- **Multi-view gap**: [[concepts/perception-for-planning.md]] — the stated reason for the current encoder is that foundation world models are front-view only. Table 7 quantifies the cost of losing side views and LiDAR at +1.1 PDMS.
- **Backbone choice**: [[concepts/foundation-backbones-for-ad.md]] — Epona vs. Vista under a fixed planner, joining DriveLaW's representation sweep and SimWAM's video-prior swap.
- **nuScenes framing**: [[concepts/nuscenes-waymo-evals.md]] — a paper hedging the benchmark's validity in the same paragraph that reports its own result on it.
