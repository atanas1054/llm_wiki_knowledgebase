---
title: "WA-JEPA: Rethinking the Video JEPA Paradigm for World-Action Modeling in Autonomous Driving"
type: source-summary
sources: [raw/papers/WA-JEPA_ Rethinking the Video JEPA Paradigm forWorld-Action Modeling in Autonomous Driving.md]
related: [concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/hugsim-benchmark.md, concepts/foundation-backbones-for-ad.md, concepts/diffusion-planner.md, concepts/best-of-n.md, sources/drive-jepa.md, sources/auto-jepa.md, sources/latent-wam.md, sources/simwam.md, sources/drivelaw.md, sources/deepsight.md, sources/flare.md, sources/had.md, sources/drivevla-w0.md, sources/wam-diff.md, sources/drivefine.md]
created: 2026-09-02
updated: 2026-09-02
confidence: high
---

# WA-JEPA

WA-JEPA argues that V-JEPA is the right *representation* for driving and the wrong *architecture* for planning, then rebuilds it. Three changes: random spatiotemporal masking becomes **hybrid future masking** (predict future frames from past context), deterministic latent regression becomes **conditional flow matching** over future latents, and scene-only modeling becomes **joint scene-action denoising** in one MMDiT predictor.

The empirical payoff is 91.7 EPDMS on NAVSIM-v2 and — more interesting — a 0.4462 HD-Score on 436 zero-shot HUGSIM closed-loop scenarios, roughly 1.4× the next-best method under a rescored common protocol. The methodological payoff is larger than either: this is the first paper in the wiki to report **NAVSIM seed variance** (10 seeds, std 0.053), to **rescore every closed-loop baseline under a pinned commit**, and to **partition the NAVSIM-v2 leaderboard by evaluator version**. That last one has consequences for the wiki's own tables.

**Source**: `raw/papers/WA-JEPA_ Rethinking the Video JEPA Paradigm forWorld-Action Modeling in Autonomous Driving.md`
**arXiv**: https://arxiv.org/html/2608.20974v1
**Code**: https://github.com/AFARI-Research/WA-JEPA
**Authors**: Xinlin Wang, Yujiao Xiang, Yuheng Zhou, Jingqi Wang, Minqing Huang (corresponding), Jiajie Huang, Dongxu Wei (project lead), Tingguang Zhou, Xiyang Wang, Gong Chen, Zhi Xu, Feiyang Tan, Hangning Zhou, Mu Yang — Afari Intelligent Drive, UESTC, Southeast University, BUPT, Tianjin University

> **Third JEPA paper, third completely different thing.** WA-JEPA, [[sources/drive-jepa.md]], and [[sources/auto-jepa.md]] all build on V-JEPA 2 and all evaluate on NAVSIM. They share almost nothing else. See [Three JEPA Papers](#three-jepa-papers-compared) below before citing any of them.

## Key Takeaways

- **The paper's core claim is that V-JEPA's own objective is wrong for planning**, on three counts: random masking is a *completion* task with no future-directed component; L1/L2 regression cannot generate genuinely unseen future tokens; and V-JEPA 2's action-conditioned variant needs a goal image plus MPC, which is not online planning. This is a direct critique of the mechanism [[sources/drive-jepa.md]] relies on.
- **Flow matching beats regression, and regression is worse than nothing.** In Stage 2, joint modeling alone gives 91.1 EPDMS; adding regression-based future prediction *drops* it to 90.7; adding flow-matching future prediction raises it to 91.7. This is the wiki's first evidence that the *form* of the future-prediction objective matters, not just its presence.
- **The diagnosis is measured, not asserted.** Regression collapses temporal variation: directional-similarity collapse gap 0.30 → 0.10 and change-magnitude ratio 0.45 → 0.80 when switching to flow matching. A deterministic L2 objective on a multimodal future converges to a temporal mean, so consecutive predicted frames become near-identical.
- **V-JEPA 2 initialization is worth +5.7 EPDMS over the next-best encoder** (89.5 vs. MAE 83.8, DINOv3 83.8, SigLIP2 83.1), with everything else held fixed and no Stage 1 pretraining in any arm. This independently corroborates Drive-JEPA's Table 7 — though both papers share a confound (see [Limitations](#limitations)).
- **Zero-shot closed-loop transfer is the strongest result.** HD-Score 0.4462 across 436 HUGSIM scenarios from four source datasets, none used in either training stage, versus 0.3252 (DrivoR), 0.3124 (UniAD), 0.2310 (LTF), 0.1393 (VAD). Best on all four source datasets individually.
- **Its own numbers come with unusually honest bookkeeping.** The paper reports EPDMS* (pre-fix evaluator) and EPDMS (corrected) as separate columns, names the NAVSIM devkit commit, pins the HUGSIM commit, rescores every closed-loop baseline itself, and reports three HUGSIM aggregation rules plus a full seed-variance table.

## Method

### The Critique of V-JEPA

V-JEPA 2 trains a predictor to infer target latents from context latents:

$$
\min_{\theta,\psi}\left\|P_{\psi}(E_{\theta}(\alpha))-\mathrm{sg}(E_{\bar{\theta}}(\beta))\right\|_{1},\qquad \bar{\theta}\leftarrow\mu\bar{\theta}+(1-\mu)\theta
$$

with $E_\theta$ the online encoder, $E_{\bar\theta}$ an EMA target encoder, and $\mathrm{sg}$ stop-gradient. WA-JEPA keeps this skeleton — online/EMA encoders, latent targets, no pixel reconstruction — and changes what is masked and how the predictor is trained.

### Stage 1: Hybrid Future-Masked Pre-training

Pretraining runs on multi-view nuPlan driving video with **no action supervision**. Each sample has $H$ historical and $K$ future frames from $C$ synchronized cameras; each camera stream is processed independently by a shared V-JEPA 2 ViT-L online encoder.

History tokens are always visible and produce the context:

$$
\mathcal{Z}_{\mathrm{ctx}}=E_{\theta}\left(\mathcal{X}_{1:H}\right)
$$

Masking applies **only to future tokens**, under two patterns:

- **Full-mask** — every future token is a prediction target. Nothing about the future is observed, so the model must predict strictly past-to-future. This is the causal branch.
- **Patch-mask** — a subset of future tokens stays visible as conditioning; the rest are predicted. This retains V-JEPA 2's partial-masking style and lowers the learning difficulty.

Visible future tokens are scattered back into a full-length sequence of learnable mask tokens:

$$
Z_{\mathrm{cond}}^{(m)}=\Phi^{(m)}\left(Z_{\mathrm{mask}},E_{\theta}\left(X_{H+1:H+K},M^{(m)}\right)\right),\qquad m\in\{\mathrm{full},\mathrm{patch}\}
$$

Under Full-mask, $Z_{\mathrm{cond}}^{(\mathrm{full})}$ is entirely learnable mask tokens. The EMA encoder supplies clean unmasked targets $\mathcal{Z}^{*}_{\mathrm{future}}=E_{\bar\theta}(\mathcal{X}_{H+1:H+K})$, used only as supervision.

### Flow Matching over Latent Futures

The replacement for regression. Sample $\epsilon_\mathrm{future}\sim\mathcal{N}(0,I)$ and flow time $t$, and interpolate linearly:

$$
\mathcal{Z}_{t}=(1-t)\epsilon_{\mathrm{future}}+t\,\mathcal{Z}_{\mathrm{future}}^{*}
$$

The predictor uses **$x$-prediction** (clean-endpoint) parameterization rather than velocity prediction:

$$
\hat{\mathcal{Z}}_{\mathrm{future}}=P_{\psi}^{\mathrm{future}}\left(\mathcal{Z}_{\mathrm{ctx}},\mathcal{Z}_{\mathrm{cond}},\mathcal{Z}_{t},t\right)
$$

$$
\mathcal{L}_{\mathrm{Stage~1}}=\frac{1}{N}\left\|\hat{\mathcal{Z}}_{\mathrm{future}}-\mathrm{sg}\left(\mathcal{Z}_{\mathrm{future}}^{*}\right)\right\|_{2}^{2}
$$

The predictor is MMDiT-style ([^7], SD3's architecture), doing joint self-attention between context and future scene tokens.

Note what this does and does not change. The *loss* is still an MSE against a clean latent — but it is an MSE conditioned on a noise level $t$, so the model learns a family of denoisers rather than a single conditional mean. That distinction is what the temporal-collapse metrics below actually measure.

### Stage 2: Joint World-Action Modeling

Initialized from Stage 1. **Only Full-mask is used**, "for consistency with the causal nature of driving" — future images now serve solely to build supervision targets and are never fed to the predictor.

Actions are noised in a normalized space in parallel with the scene:

$$
\tilde{\mathcal{Y}}_{t}=(1-t)\epsilon_{y}+t\,\bar{\mathcal{Y}}_{H+1:H+K},\qquad \bar{\mathcal{Y}}=\mathrm{Norm}(\mathcal{Y})
$$

Noisy future actions, historical actions, and ego state are separately encoded and concatenated into action tokens:

$$
\mathcal{T}_{\mathrm{act}}=\operatorname{Concat}\left[F_{n}(\tilde{\mathcal{Y}}_{t}),\,F_{h}(\mathcal{Y}_{1:H}),\,F_{s}(s)\right]
$$

($F_n$ linear; $F_h,F_s$ MLPs.) One MMDiT predictor then produces two output streams:

$$
\hat{\mathcal{Z}}_{\mathrm{future}}=P_{\psi}^{\mathrm{future}}\left(\mathcal{Z}_{\mathrm{ctx}},\mathcal{Z}_{\mathrm{cond}},\mathcal{Z}_{t},\;\mathrm{sg}(\mathcal{T}_{\mathrm{act}}),\,t\right)
$$
$$
\hat{\bar{\mathcal{Y}}}_{H+1:H+K}=P_{\psi}^{\mathrm{act}}\left(\mathcal{Z}_{\mathrm{ctx}},\mathcal{Z}_{\mathrm{cond}},\mathcal{Z}_{t},\,\mathcal{T}_{\mathrm{act}},\,t\right)
$$

**The asymmetric stop-gradient is the load-bearing design detail.** The scene stream *reads* action tokens but gradients from $\mathcal{L}_\mathrm{future}$ are blocked at that interface, so future-scene prediction never updates the action stream. The action stream, by contrast, attends to *differentiable* context and future scene tokens — so action supervision does shape the scene representation. The intent is stated directly in Appendix C: preserve future-scene modeling while pushing the scene representation toward future dynamics that matter for ego planning. It is a one-way coupling, and the direction chosen is the opposite of what most world-model planners do.

$$
\mathcal{L}_{\mathrm{act}}=\frac{1}{K}\sum_{k=1}^{K}\left\|\hat{\bar{\mathbf{y}}}_{k}-\bar{\mathbf{y}}_{k}\right\|_{2}^{2},\qquad
\mathcal{L}_{\mathrm{Stage~2}}=\lambda_{\mathrm{future}}\mathcal{L}_{\mathrm{future}}+\lambda_{\mathrm{act}}\mathcal{L}_{\mathrm{act}}
$$

$\lambda_\mathrm{future}$ and $\lambda_\mathrm{act}$ are never given a value anywhere in the paper.

### Planning Inference

Inputs are historical multi-view observations, historical actions, and ego state. Future scene latents *and* actions both start from Gaussian noise. Each of 12 sampling steps estimates clean endpoints, converts them to velocities along the linear flow paths, and integrates. No future images and no ground-truth actions are needed.

So future latents **are** generated at inference, jointly with the trajectory — WA-JEPA is squarely in the imagine-then-act camp of [[concepts/world-model-for-ad.md#test-time-imagination]], alongside DriveVA's joint video-action denoising. Whether the *inference-time generation* or the *training objective* produces the gain is not separated by any ablation here; see [Limitations](#limitations).

### Implementation

| Setting | Value |
|---|---|
| Cameras | 4 (left, front, right, rear) |
| Input | 4 historical frames at 256×512 |
| Output | 8 actions at 2 Hz — $(x_k, y_k, \phi_k)$ including heading |
| Backbone | V-JEPA 2 ViT-L |
| Predictor | MMDiT-style joint future-action |
| Sampling steps | 12 |
| Optimizer | AdamW, bfloat16, DeepSpeed ZeRO-2 |
| Learning rates | encoder $1\times10^{-5}$, scene projector $1\times10^{-4}$, joint predictor $1.5\times10^{-4}$ |
| Weight decay | 0.04 |
| Stage 1 hardware | **64 × A800**, batch 4/GPU |
| Stage 2 hardware | **32 × A800**, batch 4/GPU |
| Stage 1 data | nuPlan multi-view driving video |
| Stage 2 data | NAVSIM navtrain only |

Predicting heading $\phi_k$ alongside position is worth noting — most NAVSIM planners in the wiki output $(x,y)$ waypoints only.

## Figures

![[method_comparison_vertical.png]]

**Figure 1.** Paradigm comparison. (a) existing WAMs and their choices of world representation and scene-action coupling; (b) coupled video-based WAMs sharing a generative process over future visual content and actions, inheriting VAE latent spaces optimized for reconstruction; (c) JEPA-style latent prediction. WA-JEPA bridges semantic representation learning and predictive world modeling via future-frame masking.

![[framework_overview.png]]

**Figure 2.** The two stages. Stage 1 adapts V-JEPA 2 to multi-view driving video by predicting future representations under full-future and patch-level masks, with an EMA target encoder supplying clean latents. Stage 2 initializes from that checkpoint and jointly predicts future scene representations and ego actions through the Joint Future-Action Flow Predictor.

![[future_prediction_comparison_transposed.png]]

**Figure 3.** Target-referenced PCA of future latent predictions, flow matching vs. direct regression. A separate PCA basis is fitted to each method's EMA target representations and applied to both target and predicted latents; each map covers two consecutive future frames. Flow matching retains visibly sharper spatial structure across the horizon while regression grows progressively smoother — the qualitative counterpart of the collapse metrics.

> **Figure 4 is missing from the source conversion.** Only its caption survives ("Temporal representation preservation with flow matching (FM) and direct regression (Reg.)"). The two numbers it plots — directional-similarity collapse gap 0.30 → 0.10 and change-magnitude collapse 0.45 → 0.80 — are stated in the body text and reproduced below, so nothing quantitative is lost.

![[hugsim_vis.png]]

**Figure 5.** Zero-shot HUGSIM closed-loop rollouts. Rows are the four source datasets, columns are turning, oncoming-vehicle, and overtaking scenarios. Each pair shows the front camera with the projected 4 s plan and the BEV view with planned trajectory and detected objects; green marks the drivable corridor and the ego box, yellow-red the plan, orange other agents.

![[traj.png]]

**Figure 6.** NAVSIM trajectory predictions across (a) left and right turns, (b) fork and gateway navigation, and (c) stopping and straight driving.

## Tables

### Table 1: NAVSIM-v2 navtest

EPDMS* is the pre-fix evaluator; EPDMS is the corrected protocol (NAVSIM devkit commit `359c7f7`, which recomputes the multiplicative and weighted terms after applying the human-reference penalty filter). **The two columns are not comparable to each other.** † marks results using auxiliary simulator-derived supervision.

| Method | Backbone | NC ↑ | DAC ↑ | DDC ↑ | TLC ↑ | EP ↑ | TTC ↑ | LK ↑ | HC ↑ | EC ↑ | EPDMS* ↑ | EPDMS ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *End-to-End Methods* | | | | | | | | | | | | |
| TransFuser | RegNetY-3.2GF | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 | – |
| ARTEMIS | ResNet-34 | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | – | 83.1 | – |
| Hydra-MDP++ | V2-99 | 98.8 | 97.8 | 99.1 | 100 | 84.0 | 95.3 | 70.1 | – | 96.8 | 84.1 | – |
| DiffusionDrive | ResNet-34 | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | – | 84.5 |
| Drive-JEPA | ResNet-34 | 98.8 | 97.4 | 99.0 | 99.8 | 83.5 | 98.0 | 96.2 | 98.1 | 85.6 | 85.4 | – |
| Drive-JEPA † | ViT-L | 98.4 | 98.6 | 99.1 | 99.8 | 88.4 | 97.8 | 97.6 | 97.9 | 84.8 | 87.8 | – |
| DiffusionDriveV2 | ResNet-34 | 97.7 | 96.6 | 99.2 | 99.8 | 88.9 | 97.2 | 96.0 | 97.8 | 91.0 | 85.5 | 87.5 |
| DriveSuprim | V2-99 | 97.8 | 97.9 | 99.5 | 99.9 | 90.6 | 97.1 | 96.6 | 98.3 | 77.9 | 86.0 | – |
| SparseDriveV2 | ResNet-34 | 98.1 | 98.1 | 99.6 | 99.8 | 91.1 | 97.3 | 96.9 | 98.2 | 78.4 | 86.7 | 90.1 |
| *VLA Methods* | | | | | | | | | | | | |
| ReCogDrive | InternVL3 | 98.3 | 95.2 | 99.5 | 99.8 | 87.1 | 97.5 | 96.6 | 98.3 | 86.5 | 83.6 | – |
| WAM-Flow | Janus-1.5B | 98.5 | 94.5 | 99.5 | 99.8 | 86.9 | 96.8 | 97.4 | 97.6 | 73.9 | 84.7 | – |
| WAM-Diff | LLaDA-V | 99.0 | 98.4 | 99.3 | 99.9 | 87.0 | 98.6 | 96.2 | 98.1 | 78.5 | – | 89.7 |
| *World-Action and World-Model Methods* | | | | | | | | | | | | |
| DriveVLA-W0 | Emu3-8B | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 | – |
| DriveWorld-VLA | InternVL | 98.6 | 99.1 | 99.6 | 99.8 | 87.4 | 97.9 | 97.0 | 97.8 | 78.6 | – | 86.8 |
| CoWorld-VLA | Qwen3 | 99.1 | 97.0 | 99.6 | 99.9 | 87.9 | 98.5 | 97.7 | 98.2 | 86.2 | 86.2 | 90.0 |
| DreamerAD | Transformer-1.3B | 98.0 | 97.2 | 99.5 | 99.8 | 87.8 | 97.4 | 97.5 | 98.3 | 72.4 | – | 87.7 |
| Latent-WAM | DINOv2-B | 98.1 | 97.3 | 99.6 | 99.8 | 87.7 | 97.3 | 97.6 | 98.1 | 87.3 | – | 89.3 |
| DriveFuture | V2-99 | 98.8 | 99.1 | 99.6 | 99.9 | 86.6 | 98.4 | 96.4 | 98.3 | 74.8 | 86.4 | 89.9 |
| Discrete-WAM | Transformer-1B | 98.5 | 98.2 | 99.7 | 99.8 | 90.5 | 97.9 | 97.2 | 98.3 | 78.1 | 87.0 | 90.4 |
| **WA-JEPA (ours)** | **ViT-L** | **99.4** | 98.2 | **99.7** | 99.9 | 87.8 | **98.9** | **98.3** | 98.3 | 88.1 | **88.0** | **91.7** |

NC 99.4, TTC 98.9, and LK 98.3 are the best in this table; EC 88.1 is the best among all non-Hydra-MDP++ entries, which matters because comfort is the metric most world-model planners sacrifice. On the pre-fix column the margin is much narrower: 88.0 vs. Drive-JEPA† 87.8.

**Six methods appear in both columns**, which makes the correction's size measurable:

| Method | EPDMS* (pre-fix) | EPDMS (corrected) | Δ |
|---|---:|---:|---:|
| DiffusionDriveV2 | 85.5 | 87.5 | +2.0 |
| DriveFuture | 86.4 | 89.9 | +3.5 |
| CoWorld-VLA | 86.2 | 90.0 | +3.8 |
| SparseDriveV2 | 86.7 | 90.1 | +3.4 |
| Discrete-WAM | 87.0 | 90.4 | +3.4 |
| **WA-JEPA** | **88.0** | **91.7** | **+3.7** |

### Table 2: Zero-Shot Closed-Loop HUGSIM (436 scenarios)

All values on $[0,1]$. Every baseline was **rescored by the authors** under HUGSIM commit `ead17f2` (which includes the PR #57 trajectory-to-heading coordinate-order correction), sharing scenarios, controller, commands, aggregation, and metric implementation while keeping each method's native sensor configuration.

| | WA-JEPA | LTF | DrivoR | UniAD | VAD |
|---|---:|---:|---:|---:|---:|
| *All 436 scenarios* | | | | | |
| NC | **0.6856** | 0.4428 | 0.5217 | 0.6555 | 0.4117 |
| DAC | **0.9635** | 0.9275 | 0.9559 | 0.9320 | 0.9028 |
| TTC | **0.6120** | 0.3751 | 0.4620 | 0.5156 | 0.2798 |
| Comf. | 0.6620 | 0.9478 | 0.9390 | 0.6633 | **0.9534** |
| PDMS | **0.5717** | 0.3653 | 0.4475 | 0.4940 | 0.2831 |
| RC | **0.5689** | 0.3804 | 0.4721 | 0.4383 | 0.3006 |
| **HD-Score** | **0.4462** | 0.2310 | 0.3252 | 0.3124 | 0.1393 |
| *HD-Score by difficulty* | | | | | |
| Easy ($n{=}80$) | **0.7977** | 0.6608 | 0.7799 | 0.6395 | 0.4197 |
| Medium ($n{=}157$) | **0.5563** | 0.1547 | 0.2911 | 0.3718 | 0.0849 |
| Hard ($n{=}96$) | **0.3060** | 0.1204 | 0.2000 | 0.2099 | 0.0770 |
| Extreme ($n{=}103$) | 0.1362 | 0.1167 | **0.1407** | 0.0632 | 0.0626 |

Two rows deserve attention that the paper's narrative does not give them. **Comfort is WA-JEPA's worst metric by a wide margin** — 0.6620 against 0.95 for LTF, DrivoR, and VAD. And **on the Extreme tier WA-JEPA loses to DrivoR** (0.1362 vs. 0.1407); that tier is 103 of 436 scenarios, roughly a quarter of the benchmark. The gains are concentrated in Medium (+0.265 over DrivoR) and Hard (+0.106).

### Table 5: HUGSIM Aggregation Robustness

| Aggregation | WA-JEPA | LTF | DrivoR | UniAD | VAD |
|---|---:|---:|---:|---:|---:|
| Primary (difficulty-weighted by count 80/157/96/103) | **0.4462** | 0.2310 | 0.3252 | 0.3124 | 0.1393 |
| Dataset-uniform | **0.4483** | 0.2300 | 0.3246 | 0.3085 | 0.1304 |
| Scenario-uniform | **0.4464** | 0.2243 | 0.3194 | 0.3082 | 0.1266 |

Ranking is unchanged under all three rules. This is the kind of check almost no ingested paper runs.

### Table 6: HUGSIM Per-Dataset HD-Score

None of these four datasets is used in either training stage.

| Dataset | $n$ | WA-JEPA | LTF | DrivoR | UniAD | VAD |
|---|---:|---:|---:|---:|---:|---:|
| nuScenes | 88 | **0.4725** | 0.3334 | 0.3830 | 0.3405 | 0.2069 |
| KITTI-360 | 113 | **0.2963** | 0.0969 | 0.2175 | 0.0550 | 0.0272 |
| Waymo | 108 | **0.5542** | 0.2478 | 0.4025 | 0.4372 | 0.1376 |
| PandaSet | 127 | **0.4702** | 0.2419 | 0.2955 | 0.4012 | 0.1500 |

KITTI-360 is the hardest domain for everyone (0.2963 best), Waymo the easiest (0.5542). Winning all four separately is stronger evidence of domain-general transfer than the aggregate.

### Table 3: NAVSIM-v1 navtest

| Method | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|---|---:|---:|---:|---:|---:|---:|
| *End-to-End Methods* | | | | | | |
| TransFuser | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| DiffusionDrive | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| Drive-JEPA | 98.7 | 96.2 | 95.5 | 100 | 82.9 | 89.0 |
| *VLA Methods* | | | | | | |
| ReCogDrive | 97.9 | 97.3 | 94.9 | 100 | 87.3 | 90.8 |
| WAM-Flow | 99.2 | 98.3 | 97.0 | 99.7 | 82.3 | 90.3 |
| WAM-Diff | 99.1 | 98.3 | 96.5 | 99.9 | 84.4 | 91.0 |
| *World-Action and World-Model Methods* | | | | | | |
| CoWorld-VLA | 99.1 | 96.9 | 96.4 | 100 | 83.9 | 89.9 |
| DriveVLA-W0 | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.2 |
| DriveWorld-VLA | 99.1 | 98.2 | 96.1 | 100 | 85.9 | 91.3 |
| DriveLaW | 99.0 | 97.1 | 96.7 | 100 | 81.3 | 89.1 |
| DriveFuture | 98.8 | 99.1 | 95.4 | 100 | 84.2 | 90.7 |
| **WA-JEPA (ours)** | **99.5** | 98.3 | **97.7** | **100** | 85.0 | **91.8** |

**The Drive-JEPA row here is its 89.0 perception-free baseline, not its 93.3 full planner** — see [Limitations](#limitations). NC 99.5 is the highest NAVSIM-v1 no-at-fault-collision score in the wiki, edging DriveLaW's 99.0.

### Table 4: Ablations (NAVSIM-v2 navtest)

**(a) Vision encoder initialization.** Every variant is trained directly in Stage 2 with no Stage 1 pretraining, sharing architecture, data, and optimization.

| Encoder | EPDMS ↑ |
|---|---:|
| SigLIP2 | 83.1 |
| MAE | 83.8 |
| DINOv3 | 83.8 |
| **V-JEPA 2** | **89.5** |

The three image-level alternatives land within 0.7 of each other; V-JEPA 2 is +5.7 above the best of them. The paper reads this as the *objective* mattering rather than the choice among image-level objectives.

**(b) Stage 1 masking strategy.** Row 1 skips Stage 1 entirely and uses the stock V-JEPA 2 checkpoint for Stage 2.

| Patch-mask | Full-mask | EPDMS ↑ |
|---|---|---:|
| | | 89.5 |
| ✓ | | 91.0 |
| | ✓ | 91.3 |
| ✓ | ✓ | **91.7** |

Either mask alone recovers most of the gain (+1.5 / +1.8); together they add another +0.7 / +0.4. Full-mask — the strictly causal branch — is the stronger single choice, which is the result the paper's thesis predicts.

**(c) Stage 2 components.** Row 1 is an action-only baseline initialized from Stage 1 and fine-tuned with action supervision, using only historical latents through cross-attention.

| Joint | FM | Reg. | EPDMS ↑ |
|---|---|---|---:|
| | | | 89.9 |
| | ✓ | | 90.8 |
| ✓ | | | 91.1 |
| ✓ | | ✓ | 90.7 |
| ✓ | ✓ | | **91.7** |

**Read rows 3-5 together — this is the paper's sharpest result.** Joint modeling with no future-latent supervision reaches 91.1. Adding *regression* future prediction takes it **down** to 90.7. Adding *flow-matching* future prediction takes it up to 91.7. A badly-chosen future-prediction objective is worse than having none at all, and the swing between the two objectives is 1.0 EPDMS — larger than the margin separating the top four methods in Table 1.

### Temporal Representation Metrics

The diagnosis for why regression underperforms. Both metrics are computed against EMA targets, on the $K=64$ most dynamic camera-spatial locations per instance (ranked by the target's mean adjacent-step change, so static regions cannot dominate), over $F=4$ future token steps, averaged across a common 0-36k training interval.

**Directional similarity collapse gap** — how much more mutually similar the predicted future steps are than the targets:

$$
\Delta_{\mathrm{dir}}=C(\widehat{\mathbf{z}})-C(\mathbf{z}),\qquad C(\mathbf{q})=\frac{1}{N_{\mathrm{cos}}}\sum_{i}\sum_{r\in\mathcal{A}_{i}}\sum_{t\neq u}\cos\left(\mathbf{q}_{i,r,t},\mathbf{q}_{i,r,u}\right)
$$

**Change-magnitude ratio** — predicted temporal variation relative to target:

$$
R_{\Delta}=\frac{D(\widehat{\mathbf{z}})}{\max(D(\mathbf{z}),\epsilon)},\qquad D(\mathbf{q})=\frac{1}{N_{\Delta}}\sum_{i}\sum_{r\in\mathcal{A}_{i}}\sum_{t=0}^{F-2}\left\|\mathbf{q}_{i,r,t+1}-\mathbf{q}_{i,r,t}\right\|_{2}
$$

| Objective | $\Delta_\mathrm{dir}$ (lower better) | $R_\Delta$ (closer to 1 better) |
|---|---:|---:|
| Direct regression | 0.30 | 0.45 |
| **Flow matching** | **0.10** | **0.80** |

Regression under-produces temporal change by more than half and makes consecutive predicted frames excessively parallel — the signature of a conditional mean over a multimodal future. Flow matching cuts the excess similarity by two-thirds and recovers most of the variation.

One methodological caveat the paper states plainly: for flow matching the metric uses the **one-step $x$-prediction at the sampled training flow time**, not the multi-step inference sampler. So this measures the learned denoiser's behavior, not the deployed rollout's.

### Table 7: Seed-Level Variability

10 seeds, model parameters and scenarios fixed, sampling noise reinitialized per seed, all sub-metrics and EPDMS computed independently per seed.

| Statistic | EPDMS |
|---|---|
| Number of seeds | 10 |
| Mean | 91.7014 |
| Standard deviation | 0.0531 |
| Standard error | 0.0168 |
| 95% $t$-confidence interval | [91.6634, 91.7393] |
| Median | 91.6960 |
| Range | [91.6294, 91.8070] |

**This is the first NAVSIM seed-variance table in the wiki**, and its message is reassuring in a way the wiki should absorb: for a stochastic sampler, seed-to-seed EPDMS spread is ~0.05 with a full range of 0.18. Ablation deltas above ~0.2 are therefore meaningful for this method, and the wiki's habitual complaint about single-run NAVSIM numbers is, at least here, a smaller worry than the evaluator-version problem.

## Three JEPA Papers Compared

The wiki now holds three papers that build on V-JEPA 2 and evaluate on NAVSIM. They differ on essentially every axis that matters:

| | [[sources/drive-jepa.md]] | [[sources/auto-jepa.md]] | **WA-JEPA** |
|---|---|---|---|
| What the JEPA objective is applied to | Driving video representations | The trajectory latent space | Driving video representations |
| Masking | V-JEPA random spatiotemporal | n/a (no masking) | **Hybrid future masking** |
| Prediction objective | L1 latent regression | Alignment + cosine + InfoNCE | **Conditional flow matching** |
| Prediction target | Masked video latents | Frozen encoding of the GT future trajectory | Future multi-view scene latents |
| Encoder | Re-pretrained ViT-L (208 h) | Stock V-JEPA 2, **frozen** | Re-pretrained ViT-L (nuPlan) |
| Scene-action coupling | None — separate planner | None — retrieval | **Joint MMDiT, asymmetric stop-grad** |
| World model at inference | No | Predicts an *action* latent | **Yes — future scene latents generated** |
| Trajectory source | 32 refined online proposals | Retrieved GT geometry | Flow-matched continuous output |
| Cameras | 1 front | 1 front | 4 (L/F/R/rear) |
| NAVSIM-v1 | 93.3 PDMS | 91.3 PDMS | 91.8 PDMS |
| NAVSIM-v2 | 87.8 EPDMS* | 85.6 EPDMS* / 89.1 EPDMS | 88.0 EPDMS* / **91.7 EPDMS** |
| Closed-loop | Bench2Drive 64.52 DS | None | **HUGSIM 0.4462 HD-Score** |

**WA-JEPA's critique lands on Drive-JEPA and mostly misses Auto-JEPA.** The paper names Drive-JEPA directly — it "attempts to bridge this gap... but still relies on a separate downstream trajectory planner" — and Drive-JEPA is exactly the target of both structural complaints: it uses V-JEPA's random-mask completion objective, and its latent prediction never reaches the action. Auto-JEPA sidesteps the critique by changing the target rather than the objective. Its prediction target is a *single ego trajectory*, which is far lower-entropy than a four-camera scene, so the mean-seeking failure WA-JEPA measures barely applies; a conditional mean over plausible trajectories is itself a usable retrieval key, whereas a conditional mean over plausible futures is a blur.

That contrast is the most useful thing to take from putting the three side by side: **the right prediction objective depends on the entropy of the target.** Deterministic alignment suffices for a low-dimensional, weakly multimodal target (Auto-JEPA's trajectory); generative modeling is necessary for a high-dimensional, strongly multimodal one (WA-JEPA's scene). Drive-JEPA sits in the awkward middle — a high-entropy target with a deterministic objective — which is precisely what WA-JEPA's Table 4(c) says costs 1.0 EPDMS.

## Relationships

- **[[sources/simwam.md]] / [[sources/drivelaw.md]]** — the two papers that established no benefit from conditioning on generated futures at inference. **WA-JEPA does not contradict them, and does not test them.** Its Table 4(c) removes the future-prediction *training objective and* the inference-time generation together; SimWAM's isolated mask removed only the inference dependency while keeping the objective. So WA-JEPA is evidence for the objective — which SimWAM also supports — and silent on the inference path. The control it does not run is the one SimWAM ran.
- **[[sources/latent-wam.md]]** — the closest architectural neighbor: multi-camera, no pixel decoder, compact latent future prediction, no VLM. Latent-WAM compresses to 16 scene queries per view and predicts future latent *world status* with a causal Transformer under a deterministic objective, plus WorldMirror geometric distillation. WA-JEPA keeps full ViT-L token grids, predicts with flow matching, and couples action into the same predictor. Latent-WAM reaches 89.3 corrected EPDMS at 104M params / 107 ms; WA-JEPA reaches 91.7 with no reported latency and 96 A800s of training.
- **[[sources/deepsight.md]]** — the other multi-frame parallel latent predictor. DeepSight regresses DINOv3 features for five future BEV frames in one pass; that is exactly the deterministic-regression setup WA-JEPA's Table 4(c) finds harmful. The targets differ (frozen DINOv3 features vs. EMA JEPA latents) and the architectures differ, so this is a hypothesis rather than a refutation — but it is a directly testable one, and DeepSight's horizon-dependence result becomes more interesting in that light.
- **[[sources/flare.md]]** — annotation-free DINOv2 future-feature prediction, also deterministic. Same open question as DeepSight.
- **[[sources/drivevla-w0.md]]** — the supervision-deficit framing WA-JEPA cites approvingly as the motivation for WAMs. WA-JEPA's EC of 88.1 against DriveVLA-W0's 58.9 is the sharpest comfort contrast in Table 1.
- **[[sources/had.md]] / [[sources/latent-wam.md]] on HUGSIM** — the wiki's two existing HUGSIM entries (30.8 and 28.9 HDS) are **not comparable** to WA-JEPA's 44.62. Different scenario count (345-era vs. 436), different commit, and WA-JEPA applies the PR #57 heading-order fix. WA-JEPA does not mention either method. See [[concepts/hugsim-benchmark.md]].
- **Six un-ingested methods appear in Table 1 at or above 89.9 corrected EPDMS**: Discrete-WAM (90.4), SparseDriveV2 (90.1), CoWorld-VLA (90.0), DriveFuture (89.9), plus DrivoR on HUGSIM. CoWorld-VLA shares four authors with WA-JEPA (Huang, Xiang, Zhou, Yang). These are now the wiki's most consequential gaps on NAVSIM-v2.

## Limitations

**Comparison scope**

- **The NAVSIM-v1 table cites Drive-JEPA's 89.0 perception-free baseline rather than its 93.3 full planner** — a 4.3-PDMS understatement of its nearest methodological competitor, in a table where WA-JEPA claims 91.8. The submetrics confirm the identification (98.7 / 96.2 / 95.5 / 100 / 82.9 is Drive-JEPA's perception-free ViT-L row). The v2 table does cite Drive-JEPA's strong configuration, flagged †, which makes the v1 choice harder to read as an oversight.
- The v1 table also omits DriveSuprim 93.5, CLEAR 93.7, HybridDriveVLA 92.1, SimWAM 91.5, DynVLA 91.7, and DriveFine. Against the wiki's actual v1 frontier, 91.8 PDMS is mid-pack, not a record. The paper's v1 claim is stated modestly ("attained a PDMS of 91.8"), so this is a table-scope issue rather than an overclaim.
- On the pre-fix v2 column the margin is 88.0 vs. Drive-JEPA† 87.8 — **+0.2**, not the +1.3/+1.6 the abstract quotes. Those larger margins are computed in the corrected column against SparseDriveV2 90.1 and Discrete-WAM 90.4, which is a legitimate apples-to-apples comparison; but a reader who only sees the pre-fix column would draw a very different conclusion about how far ahead this method is.

**Closed-loop result**

- **WA-JEPA loses the Extreme tier to DrivoR** (0.1362 vs. 0.1407), which is 103 of 436 scenarios. The paper says gains are "largest on the medium and hard levels" and bolds per-row bests, so it is not concealed — but "best HD-Score" in the abstract sits alongside a loss on the hardest quarter of the benchmark.
- **Comfort is 0.6620 against ~0.95 for three of four baselines.** A flow-matching planner sampling from noise at 12 steps has no mechanism enforcing kinematic smoothness across closed-loop timesteps, and nothing in the method addresses it. This is the same structural weakness that forced Drive-JEPA's momentum-aware selector and left [[sources/auto-jepa.md]] with EC 75.2 — except here it shows up in closed loop, where it compounds.
- HUGSIM baselines are LTF, DrivoR, UniAD, and VAD — three of them pre-2024 architectures. No world-model or VLA method appears. The 1.4× margin over DrivoR is real but the field is not the current one.
- DrivoR's published scores were obtained on the 345-scenario release; the authors rescore it on 436, which is the right call, but it means the DrivoR number here is the authors' reproduction rather than a published result.

**Attribution of the gain**

- **Nothing separates the training objective from the inference-time generation.** Every row of Table 4(c) that removes future prediction removes both. Given that SimWAM and DriveLaW both found the inference path contributes nothing, the natural hypothesis is that WA-JEPA's +0.6 (91.1 → 91.7) is entirely a training effect and the 12-step scene denoising at inference is wasted compute. An isolated-mask control would settle it in one run.
- **The encoder ablation confounds "JEPA objective" with "video pretraining."** MAE, DINOv3, and SigLIP2 are all *image-level*; V-JEPA 2 is the only video-pretrained entry. The paper concludes "the gap tracks the V-JEPA 2 pre-training objective," but a video-pretrained non-JEPA control (VideoMAE, a Wan/Cosmos encoder, an inflated DINOv3) is absent. Drive-JEPA's Table 7 has the identical hole. Two papers now report the same +5-to-+13-point result with the same missing arm.
- **The stop-gradient design is described but never ablated.** Appendix C explains the asymmetric gradient flow and its rationale at length; no row shows what happens with symmetric gradients or with no action→scene conditioning. For a design the paper singles out as enabling "action supervision to directly shape planning-relevant world representations," that is the ablation a reader most wants.
- $\lambda_\mathrm{future}$ and $\lambda_\mathrm{act}$ are never reported. The balance between the two losses is the central hyperparameter of a joint model and it is absent from the paper and its appendices.

**Cost and reproducibility**

- **64 A800 GPUs for Stage 1 and 32 for Stage 2**, with no wall-clock or GPU-hour total anywhere. This is among the most expensive recipes in the wiki, and the paper's framing as a "scalable paradigm" is not accompanied by any efficiency number.
- **No latency, FPS, or parameter count.** 12 MMDiT sampling steps over four camera streams of ViT-L tokens is not obviously deployable, and every comparable method in the wiki reports inference cost — Latent-WAM 107 ms, SimWAM 518 ms, HAD 30.4 FPS.
- No scaling study on either axis. Stage 1 data volume is never quantified (just "multi-view driving videos from nuPlan"), and no encoder-size or predictor-size sweep is run. This is the axis [[sources/drivelaw.md]] and [[sources/simwam.md]] found most informative for video priors, and WA-JEPA leaves it untouched.

**Data and protocol**

- **Stage 1 pretrains on nuPlan; NAVSIM navtest is derived from OpenScene, which is derived from nuPlan.** The paper never states whether navtest scenes are excluded from the pretraining corpus. Stage 1 uses no action supervision, so this is not label leakage — but visual familiarity with the evaluation scenes would still inflate results, and it is a one-sentence disclosure the paper does not make. Drive-JEPA's corpus (CoVLA, DrivingDojo, OpenScene) has a milder version of the same issue.
- The 10-seed variance study covers only the main experiment. Every ablation in Table 4 is presumably single-seed, so the 0.3-0.4 differences within Table 4(b) sit near the measured seed noise floor of ±0.05 std — probably real, but not demonstrated to be.
- Only NAVSIM and HUGSIM. No Bench2Drive, no navhard, no nuScenes, no Waymo open-loop.

**Source conversion**

- Figure 4 (the temporal-collapse bar chart) is missing its image in the local markdown; only the caption survives. Both plotted values appear in the body text, so this costs nothing quantitatively.
