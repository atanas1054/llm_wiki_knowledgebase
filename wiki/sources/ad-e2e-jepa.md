---
title: "AD-E2E-JEPA: A Joint-Embedding Predictive Architecture For End-to-End Autonomous Driving"
type: source-summary
sources: ["raw/papers/AD-E2E-JEPA_ A Joint-Embedding Predictive Architecture For End-to-End Autonomous Driving.md"]
related: [concepts/world-model-for-ad.md, concepts/selection-based-planning.md, concepts/navsim-benchmark.md, concepts/foundation-backbones-for-ad.md, concepts/inference-latency.md, concepts/evaluation-variance.md, concepts/counterfactual-prediction.md, sources/wa-jepa.md, sources/drive-jepa.md, sources/auto-jepa.md, sources/da-wam.md, sources/dreameraD.md, sources/latent-wam.md, sources/epona.md, sources/hydra-mdp-pp.md, sources/drivesuprim.md, sources/flare.md, sources/deepsight.md, sources/redrive.md]
created: 2026-09-30
updated: 2026-09-30
confidence: medium
---

# AD-E2E-JEPA

**Paper**: AD-E2E-JEPA: A Joint-Embedding Predictive Architecture For End-to-End Autonomous Driving
**arXiv**: 2609.34085v1
**Code**: `github.com/HaoranZhuExplorer/AD-E2E-JEPA`
**Authors / orgs**: not captured in the clipping. The repository owner, the AD-L-JEPA / AD-LiST-JEPA naming lineage and the self-citations point to Haoran Zhu and Anna Choromanska's group. This is an inference, not something the source states.
**Source**: `raw/papers/AD-E2E-JEPA_ A Joint-Embedding Predictive Architecture For End-to-End Autonomous Driving.md`

---

## What It Is

The wiki's first paper that **trains no driving policy for its headline result**. An action-conditioned JEPA world model is trained on a frozen DINOv3 ViT-L, and "planning" is a search: roll the world model out under each trajectory of a fixed vocabulary and pick the one whose predicted final latent is closest to the latent of the **ground-truth future frame**, 4 s ahead.

Two things follow from that setup, and both matter more than the scores:

1. **The goal is an oracle.** Figure 2 labels it "oracle goal $I_{t+F}$". The method is shown the recorded future image, which contains the expert's endpoint. Its EPDMS is therefore a **world-model diagnostic**, not a planning result, and it must not be ranked against any planner in [[concepts/navsim-benchmark.md]]. The paper frames it this way itself ("to isolate world-model quality from policy learning").
2. **It is the setup [[sources/wa-jepa.md]] ruled out.** One of WA-JEPA's three objections to V-JEPA 2 is that its action-conditioned variant needs a goal image plus model-predictive control, which is not online planning. AD-E2E-JEPA adopts exactly that protocol on purpose and uses it as a measuring instrument.

The engineering contribution is a **two-layer convolutional projector** on top of the frozen encoder, regularized with SIGReg. It cuts the token grid from 16×32×1024 to 4×8×256 and makes per-candidate rollouts about 100× cheaper than DINO-WM / JEPA-WM. A secondary result transfers the pretrained projector into a small imitation-learning planner: **80.2 → 85.4 EPDMS**.

---

## Key Takeaways

- **Zero-shot, oracle-goal, full navtest (12,146 scenes)**: 67.3 EPDMS / 84.1 EPDMS† at 256 candidates (0.8 s on an A100); **72.9 / 86.5 at 8,192 candidates (18.2 s)**. FDE 4.0 m / 2.8 m against a goal about 20 m away.
- **The "100×" holds only at 256 candidates.** 0.8 s against 91.8 s (DINO-WM) and 101.0 s (JEPA-WM) is 115–126×. The 72.9 headline costs 18.2 s, which is 5× faster than DINO-WM at 256. The abstract quotes the speedup and the best score in the same sentence; they belong to different configurations.
- **A blind policy beats the world model that is shown the future.** The paper's own imitation-learning model, which never sees a future frame, scores 85.4 EPDMS. The oracle-goal search scores 67.3–72.9. The human trajectory it is trying to recover scores about 94.5 under the corrected evaluator ([[concepts/navsim-benchmark.md#residual-sign]]). The gap is mostly **DAC (86.0 vs 93.7) and EC (43.8 vs 88.7)**.
- **The comparison against DINO-WM / JEPA-WM exists only on 100 scenes**, and on those 100 scenes compression is *not* free at matched training: top-1 hit rate falls from 45% (JEPA-WM) to 27%. The version that matches JEPA-WM (47%) has 7× the data and a rollout loss the baselines were not given.
- **The 100-scene subset misorders the paper's own variants.** Its best configuration on the subset (76.6) is third of four on the full set (63.5). Figure 1 plots the subset numbers.
- **No ablation isolates the named contribution.** There is no projector-without-SIGReg row, no stride-1 projector, no CLS-level vs patch-level SIGReg comparison, no λ sweep and no compression-ratio sweep.
- **The projector transfers to imitation learning: +5.2 EPDMS** (80.2 → 85.4, corrected evaluator, one front camera, no perception labels). The baseline is a *randomly initialized* projector, not the absence of one.
- **Rollout training is the largest measured effect** (top-1 hit rate 31.0 → 53.8 on trainval), and **more data helps only with it**: without the rollout loss, 7× more data changes EPDMS by −0.3.

---

## Method

### Task setting

Front camera only. At time $t$ the agent sees image–pose pairs $(\mathbf I_{t-K:t},\mathbf P_{t-K:t})$ and must produce future poses $\hat{\mathbf P}_{t+1:t+F}$, with $\mathbf P_j=[x_j,y_j,\theta_j]^\top$ relative to the current ego pose. $K{+}1=4$ frames (2 s of history), $F=8$ frames (4 s), at 2 Hz.

### Baseline: JEPA-WM adapted to driving

The starting point is the best configuration reported by JEPA-WM (Terver et al.): a frozen **DINOv3 ViT-L** encoder, an **AdaLN** predictor with **RoPE**, and optional rollout training. The action is the relative pose change between consecutive frames, following [[sources/epona.md]]:

$$\mathbf a_t=\operatorname{Relative}(\mathbf P_t,\mathbf P_{t+1})=[\Delta x_{t\to t+1},\Delta y_{t\to t+1},\Delta\theta_{t\to t+1}]^\top$$

$$s_{t-K:t+1}=\operatorname{Enc}(\mathbf I_{t-K:t+1}),\qquad \hat s_{t-K+1:t+1}=\operatorname{Pred}\big(s_{t-K:t},E_a(\mathbf a_{t-K:t})\big),\qquad \mathcal L_{\mathrm{pred}}=\operatorname{MSE}(\hat s,s)$$

No anti-collapse term is needed here, because the targets come from a frozen encoder.

### The projector

Two convolutional layers with stride 2×2, shared between the history and future branches:

$$z_{t-K:t+1}=\operatorname{Proj}(s_{t-K:t+1}),\qquad \hat z_{t-K+1:t+1}=\operatorname{Pred}'\big(z_{t-K:t},E_a(\mathbf a_{t-K:t})\big)$$

| | Before | After | Ratio |
|---|---|---|---:|
| Token grid | 16 × 32 | 4 × 8 | 16× fewer tokens |
| Embedding width | 1024 | 256 | 4× narrower |
| Values per frame | 524,288 | 8,192 | 64× |

The projector is trainable, so the targets are no longer fixed and can collapse. Two mechanisms prevent it: a **stop-gradient** on the projected target, and **SIGReg**.

$$\mathcal L^{\mathrm{proj}}_{\mathrm{pred}}=\operatorname{MSE}\big(\hat z_{t-K+1:t+1},\operatorname{sg}(z_{t-K+1:t+1})\big)$$

### SIGReg at patch level

SIGReg pushes embeddings toward an isotropic Gaussian by testing random one-dimensional projections across the batch (Epps–Pulley test $T$; Cramér–Wold justifies the use of 1-D projections). LeWorldModel applies it to the global CLS embedding per time step. AD-E2E-JEPA applies it **independently at every patch location and time step**, then averages:

$$\mathcal L^{t-K:t+1}_{\mathrm{SIGReg}}=\frac{1}{NH'W'M}\sum_{l=1}^{NH'W'}\sum_{m=1}^{M}T\Big(\big\{\langle z_{l,b},\bm u^{(m)}\rangle\big\}_{b=1}^{B}\Big)$$

with $N=K+2$ frames, $M=1024$ random directions and 17 integration knots over $[0,3]$ (the LeWorld defaults). The total loss is

$$\mathcal L=\mathcal L^{\mathrm{proj}}_{\mathrm{pred}}+\lambda\,\mathcal L^{t-K:t+1}_{\mathrm{SIGReg}},\qquad \lambda=0.09\text{ by default}$$

### Optional rollout training

Without it, the model is trained only on one-step teacher-forced prediction and then asked to roll out eight steps at test time. The rollout variant trains over the whole horizon:

$$\mathcal L^{\mathrm{multi}}=\frac{\overline{\mathcal L}_{\mathrm{TF}}+\overline{\mathcal L}_{2}+\cdots+\overline{\mathcal L}_{F}}{F}+\lambda\,\mathcal L^{t-K:t+F}_{\mathrm{SIGReg}}$$

- **Teacher-forcing term** (Appendix A.2.1). For each $k=0,\dots,F-1$ the predictor takes ground-truth context. The first window is supervised in full; later windows only at their final step:

$$\overline{\mathcal L}_{\mathrm{TF}}=\frac{(K+1)\,\mathcal L^{\mathrm{proj}}_{\mathrm{pred}}[0]+\sum_{k=1}^{F-1}\mathcal L^{\mathrm{proj}}_{\mathrm{pred}}[k]}{K+F}$$

- **Rollout consistency terms** (A.2.2). One autoregressive rollout of $F$ steps from $z_{t-K:t}$, keeping a sliding window of $K+1$ embeddings: drop the oldest, append the newest prediction. Gradients are stopped through the autoregressive context before every predictor call after the first (truncated backpropagation through time):

$$\hat z^{\mathrm{AR},(j)}_{t-K+j:t+j}=\operatorname{Pred}'\Big(\operatorname{sg}\big(\tilde z^{\mathrm{AR},(j-1)}_{t-K+j-1:t+j-1}\big),E_a(\mathbf a_{t-K+j-1:t+j-1})\Big),\qquad \overline{\mathcal L}_k=\operatorname{MSE}\big(\hat z^{\mathrm{AR},(k)}_{t+k},\operatorname{sg}(z_{t+k})\big),\ k=2,\dots,F$$

- **SIGReg** over all frames from $t-K$ to $t+F$.

This is the same exposure-bias repair as [[sources/epona.md]]'s chain-of-forward training, applied in latent space.

### Zero-shot goal-conditioned planning

The cross-entropy method used by DINO-WM and JEPA-WM is judged too slow. The search runs instead over the 8,192-trajectory clustered vocabulary (the VADv2 / [[sources/hydra-mdp-pp.md]] vocabulary), **sorted by angular coordinate and subsampled at even intervals** to 256 by default.

$$\mathcal C^i=\big\|z_{t+F}-\hat z^{\,i}_{t+F}\big\|_2^2,\qquad i^*=\operatorname*{argmin}_{i}\ \mathcal C^i$$

- $z_{t+F}$ is the projected encoding of the **real future frame**.
- $\hat z^{\,i}_{t+F}$ is the end of an 8-step autoregressive rollout under candidate $i$.
- Only the final latent enters the cost. The seven intermediate predictions are computed and discarded.

### Metrics the paper introduces

| Metric | Definition |
|---|---|
| **EPDMS** | NAVSIM-v2: $\mathrm{NC}\cdot\mathrm{DAC}\cdot\mathrm{DDC}\cdot\mathrm{TLC}\cdot\dfrac{5\,\mathrm{EP}+5\,\mathrm{TTC}+2\,\mathrm{LK}+2\,\mathrm{HC}+2\,b_i\,\mathrm{EC}}{14+2\,b_i}$, where $b_i=1$ only if a valid neighboring scene exists for extended comfort |
| **EPDMS†** | The weighted-average term alone, without the four multiplicative safety terms. Motivated by "zero-shot planning does not explicitly optimize for safety" |
| **Planning time** | Mean per-scene time on one A100 80 GB, all candidates evaluated in parallel |
| **FDE, Δx, Δy, Δθ** | Error between the selected trajectory's final pose and the ground-truth pose of the goal image: Euclidean, longitudinal, lateral, heading |
| **Hit rate @k** | Append the ground-truth trajectory to the candidate set, roll out everything, and report how often the ground truth ranks in the top $k$ by cost. $\mathrm{HitRate}@k=\frac1Q\sum_n\mathbb I[\operatorname{rank}(C^{\mathrm{gt}}_n)\le k]$, $k\in\{1,5\}$. Adapted from a diagnostic in the DrivoR codebase |

### Transfer to imitation learning

Discard the predictor. Keep DINOv3 and the projector, add temporal and patch positional embeddings, concatenate an encoded driving command, and decode with one cross-attention layer (future-trajectory queries as Q, projected patches as K/V) plus an MLP, trained with MSE on the ground-truth trajectory. The architecture is adapted from [[sources/drive-jepa.md]]'s perception-free baseline. Everything is fully fine-tuned. Input is four front-camera frames at 256×512.

---

## Figures

![[overview.png|AD-E2E-JEPA overview: history frames in, rollouts over a trajectory vocabulary, selection by minimal latent distance to the future frame; bar charts for EPDMS, planning time, FDE, Δx, Δy, Δθ, top-1 hit rate and the imitation-learning boost across LeWM, DINO-WM, JEPA-WM and two AD-E2E-JEPA variants]]

*Figure 1: The task and the headline comparison. **Every bar here is from the 100-scene subset** (the caption says so). EPDMS 76.6 and 72.3 for the two AD-E2E-JEPA variants become 63.5 and 67.3 on the full test set, and their order reverses.*

![[architecture 1.png|AD-E2E-JEPA architecture: (a) frozen DINOv3 encoder with a shared learnable patch projector, SIGReg on both branches, AdaLN + RoPE patch predictor conditioned on relative-pose actions, MSE with stop-gradient, optional F-step rollout loss; (a.1) the projector as two stride-2 Conv2D layers taking 16×32×1024 to 4×8×256; (b) zero-shot goal-conditioned planning over an angularly subsampled 8192-trajectory vocabulary with an oracle goal image; (c) transfer of encoder and projector to imitation learning]]

*Figure 2: (a) the world model, (a.1) the projector, (b) zero-shot planning with the **oracle goal** $I_{t+F}$, (c) transfer to imitation learning.*

![[visualization 1.png|Bird's-eye view of one scene with the trajectory selected by LeWM, DINO-WM, JEPA-WM and AD-E2E-JEPA next to the ground-truth trajectory]]

*Figure 3: One qualitative scene. LeWM overshoots; the other three are close to the ground truth. This is the paper's only qualitative result.*

![[imitation_learning_architecture.png|Downstream imitation-learning architecture: DINOv3 encoder and patch projector, time and patch embeddings added, concatenated with an encoded driving command as keys and values; future trajectory queries through cross-attention and an MLP decoder, MSE against ground-truth trajectories]]

*Figure 4: The imitation-learning planner used for the transfer experiment.*

---

## Tables

### Table 1: Training configurations

| Split | Variant | GPUs | Batch | Learning rate | λ | Training time | GPU-hours (computed) |
|---|---|---|---:|---:|---:|---|---:|
| navtrain | LeWM | 4× A100 | **8** | 1e-4 | 0.09 | 1 d | 96 |
| navtrain | DINO-WM | 4× A100 | 64 | 1e-4 | – | 11 h | 44 |
| navtrain | JEPA-WM | 4× A100 | 64 | 1e-4 | – | 13 h | 52 |
| navtrain | AD-E2E-JEPA | 1× A100 | 128 | 1e-4 | 0.09 | 20 h | **20** |
| navtrain | + rollout | 1× A100 | 128 | 1e-4 | 0.09 | 1 d 22 h | 46 |
| trainval | AD-E2E-JEPA | 4× A100 | 512 | 2e-4 | **0.025** | 2 d 5 h | 212 |
| trainval | + rollout | 4× A100 | 256 | 1.4e-4 | 0.09 | 4 d 2 h | 392 |

AdamW, 30 epochs, one warm-up epoch, cosine schedule, learning rate scaled with the square root of the batch size. navtrain is described as 10 h of 2 Hz video and the training part of trainval as 70 h. The last column is this wiki's arithmetic.

### Table 2: Zero-shot goal-conditioned planning (256 candidates unless stated)

Time in seconds per scene on an A100. FDE / Δx / Δy in metres, Δθ in degrees. Hit rate is top-1 / top-5 in %.

**100 subsampled test scenes** (extended comfort excluded, since the subset has no temporally adjacent scenes)

| Method | Train split | EPDMS ↑ | EPDMS† ↑ | Time ↓ | FDE ↓ | Δx ↓ | Δy ↓ | Δθ ↓ | Hit rate ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LeWM | navtrain | 48.3 | 73.9 | 0.7 | 12.4 | 11.3 | 2.7 | 13.6 | 6 / 18 |
| DINO-WM | navtrain | 68.3 | 91.4 | 91.8 | 3.9 | 3.4 | 1.1 | 5.9 | 40 / 73 |
| JEPA-WM | navtrain | 74.2 | 90.9 | 101.0 | 4.0 | 3.4 | 1.2 | 5.2 | 45 / 75 |
| AD-E2E-JEPA | navtrain | 76.6 | 92.4 | 0.8 | 4.2 | 3.9 | 1.0 | 4.7 | 27 / 59 |
| + rollout | navtrain | 70.4 | 92.2 | 0.8 | 3.5 | 3.2 | 1.0 | 4.1 | 34 / 67 |
| AD-E2E-JEPA | trainval | 72.1 | 89.8 | 0.8 | 4.7 | 4.4 | 0.9 | 3.4 | 33 / 55 |
| + rollout | trainval | 72.3 | 91.9 | 0.8 | 3.2 | 3.0 | 0.7 | 3.6 | 47 / 78 |

**Full 12,146 test scenes**

| Method | Train split | EPDMS ↑ | EPDMS† ↑ | Time ↓ | FDE ↓ | Δx ↓ | Δy ↓ | Δθ ↓ | Hit rate ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LeWM | navtrain | 39.8 | 66.7 | 0.7 | 14.6 | 13 | 3.5 | 17.2 | 5.7 / 11.3 |
| AD-E2E-JEPA | navtrain | 63.5 | 80.1 | 0.8 | 6.3 | 5.9 | 1.2 | 6.3 | 32.9 / 65.3 |
| + rollout | navtrain | 64.9 | 83.1 | 0.8 | 4.5 | 4.0 | 1.2 | 4.1 | 42.6 / 71.0 |
| AD-E2E-JEPA | trainval | 63.2 | 80.1 | 0.8 | 6.2 | 5.8 | 1.1 | 4.5 | 31.0 / 64.8 |
| + rollout | trainval | **67.3** | **84.1** | 0.8 | 4.0 | 3.6 | 1.1 | 3.5 | **53.8 / 82.7** |
| + rollout, 512 traj. | trainval | 69.2 | 85.0 | 1.4 | 3.6 | 3.3 | 0.9 | 2.9 | 45.2 / 73.9 |
| + rollout, 1024 traj. | trainval | 70.5 | 85.5 | 2.5 | 3.2 | 2.9 | 0.8 | 2.5 | 36.3 / 64.3 |
| + rollout, 2048 traj. | trainval | 71.5 | 86.0 | 4.7 | 3.0 | 2.7 | 0.8 | 2.2 | 27.4 / 53.2 |
| + rollout, 4096 traj. | trainval | 72.1 | 86.3 | 9.3 | 2.9 | 2.6 | 0.8 | 2.1 | 20.7 / 43.3 |
| + rollout, 8192 traj. | trainval | **72.9** | **86.5** | 18.2 | **2.8** | 2.5 | 0.8 | 2.0 | 15.3 / 33.8 |

DINO-WM and JEPA-WM are absent from the full set because evaluating them would take more than 10 days (12,146 × 101 s ≈ 14 days).

### Table 3: NAVSIMv2 navtest stage 1 — imitation-learning transfer

V: single-view (SV) or multi-view (MV). Type: perception-free (PF) or perception-based (PB; uses perception annotations or trajectory-score labels). Fr.: input frames. EPDMS\* is the evaluator before the human-filter fix at NAVSIM commit `359c7f7`; EPDMS is after. The four named methods are "reference methods rather than direct baselines".

| Method | V | Type | Fr. | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS\* | EPDMS |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Transfuser | MV | PF | 1 | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 | – |
| Latent-WAM | MV | PF | 4 | 98.1 | 97.3 | 99.6 | 99.8 | 87.7 | 97.3 | 97.6 | 98.1 | 72.4 ⚠ | – | 89.3 |
| WA-JEPA | MV | PF | 4 | 99.4 | 98.2 | 99.7 | 99.9 | 87.8 | 98.9 | 98.3 | 98.3 | 88.1 | 88.0 | 91.7 |
| Drive-JEPA | SV | PB | 2 | 98.4 | 98.6 | 99.1 | 99.8 | 88.4 | 97.8 | 97.6 | 97.9 | 84.8 | 87.8 | – |
| *DINOv3* | | | | | | | | | | | | | | |
| + rand. proj. | SV | PF | 4 | 96.8 | 89.8 | 98.3 | 99.7 | 87.1 | 95.7 | 94.8 | 98.3 | 84.1 | – | 80.2 |
| **+ AD-E2E-JEPA proj.** | SV | PF | 4 | 97.7 | 93.7 | 99.2 | 99.8 | 87.3 | 96.8 | 97.0 | 98.4 | 88.7 | – | **85.4** |

⚠ **Latent-WAM's EC is 87.3 in its own paper** ([[sources/latent-wam.md]]). 72.4 is DreamerAD's EC, the row directly above Latent-WAM in [[sources/wa-jepa.md]]'s Table 1. Every other digit of the four reference rows matches that table, so this looks like a row-slip while copying from it.

### Table 4: EPDMS details, 100 subsampled test scenes

| Method | Train split | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS | EPDMS† |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LeWM | navtrain | 78.5 | 77.0 | 84.0 | 97.0 | 68.6 | 76.0 | 86.0 | 70.0 | – | 48.3 | 73.9 |
| DINO-WM | navtrain | 96.0 | 79.0 | 96.5 | 99.0 | 87.8 | 93.0 | 92.0 | 96.0 | – | 68.3 | 91.4 |
| JEPA-WM | navtrain | 94.5 | 87.0 | 96.5 | 99.0 | 87.6 | 92.0 | 94.0 | 93.0 | – | 74.2 | 90.9 |
| AD-E2E-JEPA | navtrain | 96.0 | 87.0 | 95.5 | 99.0 | 88.0 | 95.0 | 94.0 | 95.0 | – | 76.6 | 92.4 |
| + rollout | navtrain | 96.0 | 77.0 | 95.5 | 100.0 | 87.9 | 95.0 | 93.0 | 95.0 | – | 70.4 | 92.2 |
| AD-E2E-JEPA | trainval | 94.0 | 82.0 | 96.5 | 100.0 | 87.2 | 92.0 | 91.0 | 90.0 | – | 72.1 | 89.8 |
| + rollout | trainval | 95.5 | 82.0 | 98.0 | 99.0 | 87.3 | 93.0 | 97.0 | 96.0 | – | 72.3 | 91.9 |

### Table 5: EPDMS details, full 12,146 test scenes

| Method | Train split | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS | EPDMS† |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LeWM | navtrain | 82.0 | 67.5 | 81.4 | 98.5 | 66.1 | 79.5 | 79.0 | 66.7 | 13.7 | 39.8 | 66.7 |
| AD-E2E-JEPA | navtrain | 93.1 | 82.4 | 95.6 | 99.5 | 80.6 | 90.9 | 89.4 | 89.6 | 21.7 | 63.5 | 80.1 |
| + rollout | navtrain | 94.7 | 80.9 | 93.8 | 99.7 | 84.0 | 92.5 | 87.9 | 91.8 | 34.6 | 64.9 | 83.1 |
| AD-E2E-JEPA | trainval | 92.8 | 82.8 | 95.3 | 99.4 | 82.0 | 90.6 | 89.8 | 88.5 | 19.5 | 63.2 | 80.1 |
| + rollout | trainval | 95.2 | 82.0 | 94.4 | 99.6 | 85.2 | 93.3 | 89.3 | 92.3 | 35.5 | 67.3 | 84.1 |
| + rollout, 512 traj. | trainval | 95.9 | 83.6 | 95.5 | 99.6 | 85.6 | 94.4 | 90.4 | 93.5 | 36.9 | 69.2 | 85.0 |
| + rollout, 1024 traj. | trainval | 96.4 | 84.2 | 96.1 | 99.6 | 85.6 | 95.0 | 91.0 | 94.1 | 38.5 | 70.5 | 85.5 |
| + rollout, 2048 traj. | trainval | 96.7 | 84.8 | 96.4 | 99.7 | 85.6 | 95.4 | 91.5 | 94.5 | 40.9 | 71.5 | 86.0 |
| + rollout, 4096 traj. | trainval | 96.7 | 85.4 | 96.4 | 99.7 | 85.5 | 95.6 | 91.5 | 95.0 | 42.5 | 72.1 | 86.3 |
| + rollout, 8192 traj. | trainval | 96.9 | 86.0 | 96.7 | 99.7 | 85.4 | 95.7 | 91.8 | 95.1 | 43.8 | 72.9 | 86.5 |

**The tables are internally consistent.** Two checks, both this wiki's arithmetic:
- On the 100-scene subset, EPDMS† has no per-scene indicator, so it should equal $(5\,\mathrm{EP}+5\,\mathrm{TTC}+2\,\mathrm{LK}+2\,\mathrm{HC})/14$ of the printed means. It does on all seven rows, to within 0.06.
- On the full set, the printed EPDMS† sits between the closed form with and without extended comfort. Solving for the share of scenes with a valid neighbor gives **0.82–0.83 on every row** (LeWM, both splits, both vocabulary extremes).

---

## Reading the Results

### 1. What the oracle goal makes this a test of

The search answers one question: *which vocabulary trajectory, fed to the world model, best reproduces the real image 4 s later?* That is **inverse dynamics by analysis-by-synthesis**. In a mostly static scene it is close to relative camera-pose estimation between two frames about 20 m apart.

Three consequences:

- **EPDMS here is pose-recovery accuracy passed through a driving metric.** A perfect world model with an unlimited vocabulary would return the human trajectory, which scores about 94.5 under the corrected evaluator. The best variant recovers 72.9.
- **It does not test whether the model predicts other agents.** A model that warped the static background correctly and ignored agent motion would score nearly the same. [[sources/auto-jepa.md]]'s occlusion protocol would be the instrument for that, and it is not used.
- **Two obvious baselines are missing.** (a) A learned inverse-dynamics or pose regressor $f(I_t,I_{t+F})\to\mathbf P_{t+F}$, which would show whether a *world model* is needed for this task at all. (b) The vocabulary oracle: pick the candidate nearest the ground-truth endpoint. Without (b), nobody can say how much of FDE 4.0 m at 256 candidates is the world model and how much is the coarseness of an angular subsample.

### 2. The projector is not free at matched training

The only rows where AD-E2E-JEPA and JEPA-WM share data (navtrain) are on the 100-scene subset:

| navtrain, 100 scenes | EPDMS | FDE | Δθ | Top-1 / top-5 hit | Time |
|---|---:|---:|---:|---:|---:|
| JEPA-WM (dense 512 × 1024) | 74.2 | 4.0 | 5.2 | **45 / 75** | 101.0 s |
| AD-E2E-JEPA (32 × 256) | 76.6 | 4.2 | 4.7 | **27 / 59** | 0.8 s |
| AD-E2E-JEPA + rollout | 70.4 | 3.5 | 4.1 | 34 / 67 | 0.8 s |

- EPDMS and FDE are within noise of each other (see point 4).
- **Top-1 hit rate drops 18 points**, a 40% relative loss in how often the model ranks the true trajectory first. With a binomial standard error near 5 points per row, this is the one subset difference large enough to take seriously.
- The paper says so: its "geodesic accuracy and hit rate are lower than those of DINO-WM and JEPA-WM".
- The variant that matches JEPA-WM's hit rate (47 / 78) uses **7× the data and the rollout loss**. Whether JEPA-WM was trained with rollout is not stated, and neither baseline was trained on trainval.

So the supported claim is: *100× cheaper at a real loss of ranking reliability, recoverable with more data and rollout training.* "While retaining planning performance" holds for EPDMS and FDE only.

### 3. What the 100× buys, and where it stops

| Candidates | Time (s) | ms per candidate | Speedup vs JEPA-WM @256 | EPDMS | FDE (m) |
|---:|---:|---:|---:|---:|---:|
| 256 | 0.8 | 3.1 | 126× | 67.3 | 4.0 |
| 512 | 1.4 | 2.7 | 72× | 69.2 | 3.6 |
| 1,024 | 2.5 | 2.4 | 40× | 70.5 | 3.2 |
| 2,048 | 4.7 | 2.3 | 21× | 71.5 | 3.0 |
| 4,096 | 9.3 | 2.3 | 11× | 72.1 | 2.9 |
| 8,192 | 18.2 | 2.2 | 5.5× | 72.9 | 2.8 |

- Cost is linear in candidates at about 2.2 ms each, plus a small fixed cost.
- **32× more candidates buys +5.6 EPDMS and −1.2 m FDE at 23× the time.** Each doubling is worth less than the last (+1.9, +1.3, +1.0, +0.6, +0.8).
- The gain arrives through the safety terms and comfort (NC +1.7, DAC +4.0, DDC +2.3, EC +8.3). Ego progress is flat (85.2 → 85.4).
- 0.8 s for a 4 s plan is not real-time at the benchmark's own 2 Hz. It is fast enough to evaluate a full test set in 2.7 hours, which is what the speedup actually enables: **a JEPA world model can now be scored on all of navtest.**

### 4. The 100-scene subset cannot rank these variants

| Variant | EPDMS, 100 scenes | EPDMS, full set | Rank on subset → full |
|---|---:|---:|---|
| navtrain | **76.6** | 63.5 | 1 → 3 |
| navtrain + rollout | 70.4 | 64.9 | 4 → 2 |
| trainval | 72.1 | 63.2 | 3 → 4 |
| trainval + rollout | 72.3 | **67.3** | 2 → 1 |

- The subset's submetrics are counts out of 100. "+ rollout" on navtrain loses 6.2 EPDMS because DAC goes from 87 to 77, which is **ten scenes**.
- Subset scores are 5–13 points above full-set scores. About half of that is the excluded extended-comfort term (closed form for the navtrain variant: 86.8 without EC against the reported 80.1 with it) and the rest is the sample.
- The subset is not uniformly easier: its FDE is lower but its hit rates are also lower than on the full set in three of four rows.
- The paper concedes the point: the subset result "may partly reflect variance from evaluating only 100 scenes".
- **Consequence**: the DINO-WM / JEPA-WM comparison, which exists only on these 100 scenes, supports "roughly comparable on EPDMS and FDE" and nothing finer. See [[concepts/evaluation-variance.md]].

### 5. Rollout training, and data that helps only with it

Full test set, 256 candidates:

| | navtrain (10 h) | trainval (70 h) | Effect of 7× data |
|---|---:|---:|---:|
| No rollout: EPDMS / FDE / top-1 | 63.5 / 6.3 / 32.9 | 63.2 / 6.2 / 31.0 | **−0.3 / −0.1 / −1.9** |
| Rollout: EPDMS / FDE / top-1 | 64.9 / 4.5 / 42.6 | 67.3 / 4.0 / 53.8 | +2.4 / −0.5 / +11.2 |
| Effect of rollout | +1.4 / −1.8 / +9.7 | **+4.1 / −2.2 / +22.8** | |

- A model trained only on one-step prediction does not improve with seven times the data when it is asked for eight-step rollouts.
- **Caveat**: the trainval no-rollout row is the only one with λ = 0.025 and batch 512, "tuned heuristically based on early training loss curves". The data effect and the hyperparameter change are confounded in that cell.
- Rollout training **lowers DAC** on both splits (82.4 → 80.9, 82.8 → 82.0) while raising almost everything else. The paper does not comment.

### 6. Hit rate is not comparable across vocabulary sizes

Top-1 hit rate falls from 53.8% at 256 candidates to 15.3% at 8,192 while FDE *improves*. The ground truth is appended to the candidate set, so a denser vocabulary contains more near-duplicates of it that can outrank it. Chance is 0.39% at 256 and 0.012% at 8,192, so the model is far above chance everywhere. The paper notes the trend in one sentence. **Read hit rate only within one vocabulary size.**

### 7. Longitudinal error dominates

At 256 candidates the final-pose error is Δx 3.6 m against Δy 1.1 m, and a full vocabulary moves Δx to 2.5 and Δy only to 0.8. Two candidate causes, not separated by the paper:
- The 256 subsample is spaced evenly in **angle**, which guarantees lateral coverage and leaves trajectory length to chance.
- Distance travelled along a straight road changes a front-camera image less than a heading change does, more so at 32 tokens per frame.

### 8. A blind policy against an oracle-goal search

| | Sees the future frame? | NC | DAC | DDC | EP | TTC | LK | HC | EC | EPDMS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot, 256 cand. | **Yes** | 95.2 | 82.0 | 94.4 | 85.2 | 93.3 | 89.3 | 92.3 | 35.5 | 67.3 |
| Zero-shot, 8,192 cand. | **Yes** | 96.9 | 86.0 | 96.7 | 85.4 | 95.7 | 91.8 | 95.1 | 43.8 | 72.9 |
| IL, random projector | No | 96.8 | 89.8 | 98.3 | 87.1 | 95.7 | 94.8 | 98.3 | 84.1 | 80.2 |
| IL, pretrained projector | No | 97.7 | 93.7 | 99.2 | 87.3 | 96.8 | 97.0 | 98.4 | 88.7 | 85.4 |

- The search loses to a small regression head on every submetric, with privileged information.
- **DAC**: matching only the final latent does not constrain the path, and a mean lateral error near 1 m at 20 m is enough to leave the drivable area in 14–18% of scenes.
- **EC 19.5–43.8 is the lowest extended comfort recorded in the wiki** (previous lows: World4Drive 53.9 in [[sources/latent-wam.md]]'s table, DriveVLA-W0 58.9). Each frame is searched independently over a coarse vocabulary, so consecutive plans jump. EC rises with rollout training and with vocabulary size, which fits that reading.
- This is the same lesson as [[sources/drive-jepa.md]]'s momentum-aware selector and [[sources/auto-jepa.md]]'s EC 75.2: **per-frame selection needs an explicit continuity term.**

### 9. The imitation-learning transfer

+5.2 EPDMS from initializing a two-layer conv projector, arriving through DAC (+3.9), EC (+4.6), LK (+2.2), TTC (+1.1), NC (+0.9) and DDC (+0.9). Ego progress barely moves (+0.2).

What the comparison does and does not show:
- **The control is a random projector, not no projector.** A randomly initialized 16× spatial bottleneck trained only through an MSE trajectory loss is a handicapped baseline. The missing row is DINOv3 with its full 512 tokens and no projector.
- **Which checkpoint supplied the projector is not stated** (navtrain or the 70 h trainval model, with or without rollout). If it is the trainval model, the pretrained arm saw about 7× more video than the imitation stage.
- Single run, no seeds.
- **The random-projector row is almost TransFuser's row.** Seven of nine submetrics are within 0.5 (LK +2.1 and EC −3.1 are the exceptions), yet it scores 80.2 where TransFuser scores 76.7. That 3.5-point difference is the evaluator correction, visible on near-identical submetrics. See [[concepts/navsim-benchmark.md]].

---

## Five JEPA Papers Compared

| | [[sources/drive-jepa.md]] | [[sources/auto-jepa.md]] | [[sources/wa-jepa.md]] | [[sources/da-wam.md]] | **AD-E2E-JEPA** |
|---|---|---|---|---|---|
| Encoder | V-JEPA 2, re-pretrained | V-JEPA 2, frozen | V-JEPA 2, re-pretrained | V-JEPA 2.1 + LoRA | **DINOv3 ViT-L, frozen** + learned projector |
| Prediction target | Masked video latents | Future ego-trajectory latent | Future multi-view scene latents | Future scene latent per candidate | Future projected patch latents per candidate |
| Objective | L1 regression, EMA target | Alignment + cosine + InfoNCE | Flow matching, EMA target | Feature regression, EMA target | **MSE + stop-gradient + SIGReg, no EMA** |
| Action-conditioned? | No | No | Reads actions, stop-gradient | Yes (action as query) | **Yes (relative-pose tokens, AdaLN)** |
| Futures per scene at inference | 0 | 0 (one action latent) | 1 | 32 | **256–8,192** |
| Prediction horizon | – | 4 s trajectory | 4 s | 0.5 s | **4 s, 8 autoregressive steps** |
| Who picks the trajectory | Scorer + momentum | Retrieval + scorer + gate | The flow sampler | A learned scorer | **Latent distance to a real future frame** |
| Policy trained? | Yes | Yes | Yes | Yes | **No** (zero-shot); yes (transfer) |
| NAVSIM-v2 | 87.8\* | 85.6\* / 89.1 | 88.0\* / 91.7 | 87.7 (unclear) | 85.4 (IL transfer); 67.3–72.9 oracle-goal, not comparable |

- **It is the only one of the five not built on V-JEPA.** The choice is inherited from JEPA-WM, whose ablation prefers DINOv3 over DINOv2 and V-JEPA 2 for world-model planning. [[sources/wa-jepa.md]] and [[sources/drive-jepa.md]] find the opposite for imitation planning (V-JEPA 2 89.5 vs DINOv3 83.8). The two results are about different uses of the encoder; see [[concepts/foundation-backbones-for-ad.md]].
- **It is the only one with no EMA target encoder.** Collapse is handled by the stop-gradient and a distributional regularizer. Which of the two carries the load is not ablated.
- **It deterministically regresses patch latents 4 s out**, the objective [[sources/wa-jepa.md]] measured as worse than no future prediction on multi-view scene latents. It is not contradicted here, because nothing downstream is compared against a generative objective. The wiki's entropy-of-the-target reading predicts that a 32-token, 256-wide, Gaussian-regularized target is much less exposed than full ViT-L grids.

---

## Relationships

- **[[sources/da-wam.md]]**: the closest design. Both predict one future per candidate with a shared action-conditioned predictor. DA-WAM predicts 0.5 s for 32 candidates and hands the latent to a learned scorer (+0.15 PDMS). AD-E2E-JEPA predicts 4 s for 256–8,192 candidates and needs no scorer, because it compares against a real future. DA-WAM's open problem was that 31 of its 32 futures are never checked for being futures. **Hit rate is a direct check of that property**: does the rollout under the true action match reality better than the rollouts under the other actions? Here it does in 54% of scenes against 256 alternatives.
- **[[sources/dreameraD.md]]**: the deployable version of the same loop. DreamerAD also rolls a latent world model out over 256 vocabulary trajectories, then scores the latents with a learned reward model instead of an oracle goal. AD-E2E-JEPA plus a reward head is DreamerAD in JEPA space at 3 ms per candidate.
- **[[sources/wa-jepa.md]]**: cited twice. Once for the training bill (64 + 32 A800s), against which this paper positions itself as the cheap alternative (20 GPU-hours on navtrain). And its Table 1 is the source of the reference rows here, including the both-columns convention.
- **[[sources/drive-jepa.md]]**: supplies the imitation-learning architecture. Its self-supervised-pretraining claim is the one this paper extends from the encoder to a projector.
- **[[sources/auto-jepa.md]]**: the opposite minimal position. Auto-JEPA predicts only what the ego will do and cannot be rolled out. AD-E2E-JEPA predicts only what the scene will look like under a given ego action and has no opinion about what the ego should do. Neither is a planner without an added component (retrieval memory and scorer; a goal).
- **[[sources/flare.md]] / [[sources/deepsight.md]]**: the other frozen-DINO future-feature predictors. Both use the prediction as a training-time auxiliary loss. This paper is the first here to *use* a DINO-space prediction at decision time.
- **[[sources/hydra-mdp-pp.md]] / [[sources/drivesuprim.md]]**: owners of the 8,192 vocabulary. They score it with heads distilled from the simulator; this paper scores it with no simulator labels at all, and needs the future instead.
- **[[sources/redrive.md]]** (ingested the same day): the same signal used as a gradient. ReDrive freezes an action-conditioned JEPA predictor and trains its planner so that the predictor's output under the planner's own rollout matches the real future representation. That is this paper's selection cost, differentiated. It adds 0.2–0.3 PDMS to an already-supervised planner, which agrees with the reading here that the signal is real and weak. ReDrive also supplies **counter-evidence to the projector hypothesis** this page fed into [[concepts/foundation-backbones-for-ad.md#encoder-by-job]]: its DINOv2 encoder, trained end to end under a future-prediction loss, still trails V-JEPA 2 by 5.5 PDMS. And it reports only action *sensitivity* for its predictor, where the hit rate introduced here would test accuracy.
- **Un-ingested and load-bearing**: JEPA-WM (Terver et al., the baseline and the source of every architectural default), DINO-WM, LeWorldModel (LeWM), LeJEPA (SIGReg) and "Temporal Straightening for Latent Planning" (the stride-1 projector this work extends).

---

## Limitations

**The headline setting**

1. **Oracle goal.** Zero-shot planning requires the ground-truth future frame. No goal generator, learned cost or reward head is proposed, so there is no path from the 67.3–72.9 numbers to a deployable planner. The deployable result is the 85.4 imitation model.
2. **The oracle-goal search is beaten by the paper's own blind policy** (72.9 at best vs 85.4), with DAC and EC the main losses.
3. **No upper or lower reference for the protocol**: no vocabulary-nearest-endpoint oracle, no inverse-dynamics or pose-regression baseline, no human-trajectory score.
4. **The cost uses only the final frame.** Path shape between $t$ and $t+F$ is unconstrained by the objective.

**Attribution**

5. **No ablation of the contribution.** SIGReg, the stop-gradient, the patch-level formulation, stride 2 vs stride 1, λ and the compression ratio are all unablated. The claim that SIGReg "preserves planning performance" rests on a comparison against models that differ in more than SIGReg.
6. **The LeWM baseline differs in three ways at once**: a ViT-L trained from scratch on 10 h of video, a single CLS token, and **batch size 8** for a regularizer that is a batch statistic. The "+23.7 EPDMS over LeWM" headline is not a measurement of any one of them.
7. **DINO-WM and JEPA-WM were not given the winning recipe.** They are trained on navtrain only, and their rollout status is unstated.
8. **Data scaling is confounded** with λ, batch size and learning rate in the trainval no-rollout row.
9. **The transfer experiment lacks a no-projector control**, does not say which checkpoint was transferred, and is a single run.

**Evaluation**

10. **The only comparison with dense-token world models is on 100 scenes**, where the paper's own four variants are misordered relative to the full set. Figure 1 plots those numbers.
11. **"Retaining planning performance" omits hit rate**, which drops from 45% to 27% at matched training.
12. **Speedup and best score are different configurations**: 100× at 256 candidates, 5× at 8,192.
13. **One transcription error in Table 3** (Latent-WAM EC 72.4 for 87.3).
14. **Front camera, NAVSIM only.** No navhard, no closed loop, no second dataset. "Generalization to unseen scenes" is asserted for the zero-shot setting and tested only in-distribution.
15. **Does not test agent dynamics.** Nothing separates predicting the static scene under ego motion from predicting what other agents do.
16. No parameter counts for the predictor or projector, no latency for the imitation model, one qualitative example.

**Source conversion**

17. The author list and affiliations are empty in the clipping. Equations 1–2 and 4–6 are split across display blocks but complete. All four figures and five tables are present.

---

## Key Cross-References

- [[concepts/world-model-for-ad.md]] — Pattern 38: the world model alone as the planner; per-candidate rollouts at scale; hit rate as a check on per-candidate futures.
- [[concepts/selection-based-planning.md]] — vocabulary selection with no scorer; vocabulary-size scaling under a linear per-candidate cost.
- [[concepts/navsim-benchmark.md]] — the 85.4 row; why the zero-shot rows stay out of the table; the near-matched TransFuser pair; residual signs on this table.
- [[concepts/foundation-backbones-for-ad.md]] — DINOv3 for world-model planning against V-JEPA 2 for imitation; a pretrained projector as a cheap adaptation.
- [[concepts/inference-latency.md]] — per-candidate rollout cost; token count as the lever.
- [[concepts/evaluation-variance.md]] — the 100-scene subset as a measured example of scene-sampling variance.
- [[concepts/counterfactual-prediction.md]] — goal-conditioned search conditions on the factual future to recover the *action*, not the world.
