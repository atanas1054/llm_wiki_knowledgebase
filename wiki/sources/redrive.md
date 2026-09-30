---
title: "ReDrive: Shaping Representations with World Modeling for End-to-End Driving"
type: source-summary
sources: ["raw/papers/ReDrive_ Shaping Representations with World Modeling for End-to-End Driving.md"]
related: [concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/navhard-ood-evaluation.md, concepts/foundation-backbones-for-ad.md, concepts/wam-attention-masks.md, concepts/inference-latency.md, concepts/evaluation-variance.md, concepts/physicalai-av-benchmark.md, concepts/counterfactual-prediction.md, concepts/diffusion-planner.md, sources/wa-jepa.md, sources/drive-jepa.md, sources/da-wam.md, sources/auto-jepa.md, sources/ad-e2e-jepa.md, sources/metis.md, sources/simwam.md, sources/suv.md, sources/reworld.md, sources/drivelaw.md, sources/drivefuture.md, sources/latent-wam.md, sources/drivesuprim.md, sources/physwam.md, sources/diffusiondrive.md]
created: 2026-09-30
updated: 2026-09-30
confidence: medium
---

# ReDrive

**Paper**: ReDrive: Shaping Representations with World Modeling for End-to-End Driving
**Authors**: Yueting Zhu, Shaoyu Chen, Yuehao Song, Hui Sun, Qian Zhang, Wenyu Liu, Xinggang Wang
**Orgs**: Huazhong University of Science & Technology, Horizon Robotics (the DiffusionDrive group; Liu and Wang are also on DriveLaW and ReWorld)
**arXiv**: 2609.33854v1
**Code**: none stated
**Source**: `raw/papers/ReDrive_ Shaping Representations with World Modeling for End-to-End Driving.md`

---

## What It Is

An encoder–planner with **nothing else at inference**: a V-JEPA 2 ViT-L video encoder (304M) and a flow-matching Action DiT (103M) over four front-camera frames. A 153M future predictor exists only during training.

The training recipe has three stages:

1. **Driving-domain pretraining.** V-JEPA-style masked latent prediction on nuScenes, navtrain and 80 h of PhysicalAI-AV front video.
2. **Joint training.** The encoder feeds two branches. The planner learns the trajectory by flow matching. The future predictor regresses the **EMA-target representation of the next four frames, conditioned on the ground-truth trajectory**. Both losses update the encoder; only the planning loss updates the planner.
3. **Planner adaptation.** Encoder and predictor are frozen. The planner is rolled out for five steps from noise, its trajectory is fed to the frozen predictor, and the feature loss against the *real* future is backpropagated through the trajectory into the planner.

Headline numbers: **91.0 PDMS (NAVSIM v1), 90.8 EPDMS (NAVSIM v2, corrected evaluator), 34.4 EPDMS (navhard)**, with no perception labels, no scorer, no RL and no future prediction at inference.

---

## Key Takeaways

- **90.8 corrected EPDMS is third in the wiki among methods with no scorer and no RL**, behind [[sources/wa-jepa.md]] (91.7) and [[sources/suv.md]] (91.0), and the **highest for a world-model-trained planner that predicts no future at inference** (SUV's no-access variant is 90.7). Its v2 table omits both.
- **The named mechanism is worth +0.7 PDMS.** Future prediction in Stage 2 moves 89.2 → 89.9. The other terms: V-JEPA 2 over DINOv2 +5.5, unfreezing the encoder +3.5, 16-frame over 4-frame pretraining +0.9, Stage 1 +0.3, Stage 3 +0.3.
- **"Improves representations by 3.5" does not isolate world modeling.** The frozen-probe comparison (86.7 → 90.2) is between a pretrained encoder and one fine-tuned with *both* the planning loss and the future loss. The paper's own Table 5 attributes +3.5 to planning-loss fine-tuning alone and +0.7 to future prediction.
- **Deterministic regression of future latents helps here (+0.7)**, where [[sources/wa-jepa.md]] measured it as worse than nothing (−0.4) on the same backbone family. The difference is where the prediction goes and what it is conditioned on.
- **A third encoder sweep agrees**: V-JEPA 2 89.9 against DINOv2 84.4 and MAE 83.9, with the image encoders given the same driving-domain pretraining *and* the same future-prediction training.
- **Stage 3 turns the world model into a differentiable critic**: the planner is trained so that its own rollout, passed through the frozen predictor, reproduces the real future. It is the gradient version of [[sources/ad-e2e-jepa.md]]'s oracle-goal search, and it is worth +0.2 to +0.3.
- **navhard 34.4 is the best result for a model with no inference-time future**, and the most Stage-1-heavy profile recorded (82.3 / 42.3).
- **The comparison tables need care.** The v2 table is digit-identical to WA-JEPA's corrected column with its top entries absent; the v1 DriveSuprim row pairs one model's sub-scores with another's PDMS; two v1 rows have EP and TTC swapped.

---

## Method

### Stage 1: driving-domain pretraining

The encoder starts from the released V-JEPA 2 ViT-L checkpoint. A context encoder sees visible tokens, an EMA target encoder sees the unmasked clip, and a light predictor regresses the target latents at masked positions $\mathcal M$:

$$\mathcal L_{\mathrm{ssl}}=\frac{1}{|\mathcal M|}\sum_{i\in\mathcal M}\big\|\hat z_i-z_i^{\mathrm{tgt}}\big\|_1$$

Clips are 16 frames at 10 Hz and 256×512, with multi-block masking (8 small blocks at scale 0.15, 2 large at 0.70), for 100 epochs. Only the encoder is kept.

**Data.** nuScenes `CAM_FRONT` from the 700 training scenes; nuPlan front-camera sequences from the navtrain sensor data; an 80 h subset of NVIDIA PhysicalAI-AV, front wide camera. The three sources are sampled uniformly. "Only videos from the training splits of nuScenes and nuPlan are used for pretraining."

**Camera alignment.** PhysicalAI-AV's 120° f-theta camera is remapped to nuPlan's front-camera geometry: each target pixel ray is mapped back through the per-clip calibration and bilinearly resampled. Clips whose field of view cannot cover the target are dropped.

### Stage 2: joint training

A video is split into history $X_{1:T}$ and future $X_{T+1:2T}$ with $T=4$.

$$Z_h=E_\theta(X_{1:T}),\qquad Z_f^{\mathrm{tgt}}=E^{\mathrm{tgt}}(X_{T+1:2T}),\qquad \hat Z_f=P_\phi(Z_h,A)$$

$A$ are action tokens encoded from the **ground-truth** future trajectory, injected by cross-attention. The target encoder is an EMA of $E_\theta$ (decay 0.9999) and carries no gradient.

$$\mathcal L_{\mathrm{feat}}=\frac1N\big\|\operatorname{Norm}(\hat Z_f)-\operatorname{Norm}(Z_f^{\mathrm{tgt}})\big\|_1$$

The planner is an Action DiT (28 layers, width 512) with AdaLN for the noise level and ego status, and cross-attention to $Z_h$:

$$\tau_t=(1-t)\tau+t\epsilon,\qquad \hat v_t=G_\psi(\tau_t,t,c,Z_h),\qquad v_t=\epsilon-\tau,\qquad \mathcal L_{\mathrm{plan}}=\|\hat v_t-v_t\|_2^2$$

$$\mathcal L=\mathcal L_{\mathrm{plan}}+\lambda\,\mathcal L_{\mathrm{feat}},\qquad \lambda=0.1$$

**Who is updated by what.**

| Loss | Encoder $E_\theta$ | Predictor $P_\phi$ | Planner $G_\psi$ |
|---|:-:|:-:|:-:|
| $\mathcal L_{\mathrm{plan}}$ | ✓ | – | ✓ |
| $\mathcal L_{\mathrm{feat}}$ | ✓ | ✓ | **✗** |

The predictor is conditioned on the recorded trajectory, so the feature loss has no path into the planner. The stated reason: "This decoupling prevents the prediction objective from steering the trajectory policy toward actions that are easier to predict rather than better for planning."

### Stage 3: planner adaptation

Encoder and predictor are frozen. The planner generates $\hat\tau$ by a five-step rollout from Gaussian noise. $\hat\tau$ replaces the ground-truth trajectory as the predictor's condition, and the resulting $\mathcal L_{\mathrm{feat}}$, still measured against the real future representation, is propagated **through the generated trajectory** into the planner.

$$\mathcal L_{\mathrm{plan}}=\mathcal L_{\mathrm{fm}}+\lambda_{\mathrm{traj}}\mathcal L_{\mathrm{traj}},\qquad \mathcal L_{\mathrm{adapt}}=\mathcal L_{\mathrm{plan}}+\lambda_{\mathrm{feat}}\mathcal L_{\mathrm{feat}}$$

$\mathcal L_{\mathrm{fm}}$ is applied at each denoising step and $\mathcal L_{\mathrm{traj}}$ is an L1 loss on the final generated trajectory. $\lambda_{\mathrm{traj}}$ and $\lambda_{\mathrm{feat}}$ are not reported. The predictor stays frozen so that imperfect generated trajectories cannot "alter the learned correspondence between ego motion and future scene evolution".

### Inference

Encoder plus Action DiT, five flow steps. The predictor and target encoder are gone.

---

## Figures

![[intro1 1.png|Three kinds of visual representation for driving: a BEV representation trained with detection and segmentation, an image representation trained by masked reconstruction, and a video representation trained by regressing a target encoder's latents]]

*Figure 1(a): Existing visual representations. The clipping holds only panel (a); any further panel of Figure 1 is missing.*

![[overview 1.png|ReDrive's three stages: self-supervised pretraining with a masked predictor and EMA target encoder; joint training of video encoder, future predictor conditioned on the ground-truth action, and DiT planner; planner adaptation with frozen encoder and future predictor conditioned on the planner's own multi-step rollout]]

*Figure 2: The three-stage procedure. In Stage 3 the planner's rollout becomes the predictor's condition and only the planner is trainable.*

![[vis 2.png|Four driving scenes with front-camera and BEV views, comparing the human trajectory (yellow), Drive-JEPA (green) and ReDrive (red)]]

*Figure 3: Qualitative comparison of the human trajectory, Drive-JEPA and ReDrive.*

![[analysis.png|Left: cosine similarity between future representations predicted under different lateral trajectory offsets, plotted against offset. Right: the 15 by 15 pairwise cosine similarity matrix, near 1 on the diagonal and falling to about 0.72 for the most different offsets]]

*Figure 4: Sensitivity of the future predictor to the trajectory condition with the history representation fixed. The ground-truth trajectory is shifted laterally by 15 offsets from −0.8 to 0.8 in normalized action space.*

![[camera_alignment.png|Three PhysicalAI-AV wide-angle frames on the left and their geometrically remapped versions with nuPlan-style camera geometry on the right]]

*Figure 5: Camera alignment for the PhysicalAI-AV pretraining clips.*

![[sup2_vis.png|Four further scenes with front-camera and BEV views comparing human, Drive-JEPA and ReDrive trajectories]]

*Figure 6: Additional qualitative comparisons.*

![[action_sweep_grid.png|A grid of twelve scenes, each with a scatter of cosine similarities against lateral offset and a pairwise similarity matrix, all showing similarity decaying as the offsets diverge]]

*Figure 7: The sensitivity analysis of Figure 4 repeated across twelve scenes.*

---

## Tables

### Table 1: NAVSIM v1

Columns as printed (NC, DAC, EP, C, TTC). ⚠ marks rows discussed under [Table problems](#table-problems).

| Type | Method | Inputs | NC | DAC | EP | C | TTC | PDMS |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Perception-based | Transfuser | C + L | 97.7 | 92.8 | 79.2 | 100 | 92.8 | 84.0 |
| | VADv2 ⚠ | Camera | 97.2 | 89.1 | 91.6 | 100 | 76.0 | 80.9 |
| | UniAD ⚠ | Camera | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| | Hydra-MDP | C + L | 98.4 | 97.7 | 85.0 | 100 | 94.5 | 89.9 |
| | Hydra-MDP++ ⚠ | C + L | 97.6 | 96.0 | 80.4 | 100 | 93.1 | 86.6 |
| | DiffusionDrive | C + L | 98.2 | 96.2 | 82.2 | 100 | 94.7 | 88.1 |
| | GoalFlow | C + L | 98.4 | 98.3 | 85.0 | 100 | 94.6 | 90.3 |
| | DriveDPO | C + L | 98.5 | 98.1 | 84.3 | 100 | 94.8 | 90.0 |
| | DriveSuprim ⚠ | Camera | 98.6 | 98.6 | 91.3 | 100 | 95.5 | 89.9 |
| Perception-free | LAW | C + L | 97.4 | 93.3 | 78.8 | 100 | 91.9 | 83.8 |
| | World4Drive | C + L | 97.4 | 94.3 | 79.9 | 100 | 92.8 | 85.1 |
| | Epona | Camera | 97.9 | 95.1 | 80.4 | 99.9 | 93.8 | 86.2 |
| | Drive-JEPA | Camera | 98.7 | 96.2 | 82.9 | 100 | 95.5 | 89.0 |
| | DriveLaW | Camera | 99.0 | 97.1 | 81.3 | 100 | 96.7 | 89.1 |
| | ReWorld | Camera | 99.1 | 98.2 | 82.0 | 99.8 | 97.7 | 90.4 |
| | DAWN | Camera | 98.7 | 95.9 | 84.3 | 100 | 96.0 | 89.1 |
| | **ReDrive** | Camera | **99.1** | 97.9 | 84.3 | 100 | 97.2 | **91.0** |

### Table 2: NAVSIM v2 (all rows stated to be corrected-evaluator)

| Method | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DiffusionDrive | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| DiffusionDriveV2 | 97.7 | 96.6 | 99.2 | 99.8 | 88.9 | 97.2 | 96.0 | 97.8 | 91.0 | 87.5 |
| WAM-Diff | 99.0 | 98.4 | 99.3 | 99.9 | 87.0 | 98.6 | 96.2 | 98.1 | 78.5 | 89.7 |
| DreamerAD | 98.0 | 97.2 | 99.5 | 99.8 | 87.8 | 97.4 | 97.5 | 98.3 | 72.4 | 87.7 |
| Latent-WAM | 98.1 | 97.3 | 99.6 | 99.8 | 87.7 | 97.3 | 97.6 | 98.1 | 87.3 | 89.3 |
| DriveFuture | 98.8 | 99.1 | 99.6 | 99.9 | 86.6 | 98.4 | 96.4 | 98.3 | 74.8 | 89.9 |
| CoWorld-VLA | 99.1 | 97.0 | 99.6 | 99.9 | 87.9 | 98.5 | 97.7 | 98.2 | 86.2 | 90.0 |
| SparseDriveV2 | 98.1 | 98.1 | 99.6 | 99.8 | 91.1 | 97.3 | 96.9 | 98.2 | 78.4 | 90.1 |
| **ReDrive** | 99.1 | 97.8 | 99.6 | 99.9 | 87.6 | **98.7** | **98.2** | 98.3 | 86.4 | **90.8** |

### Table 3: NAVSIM v2 navhard

S. is the per-stage score; EPDMS is the combined score.

| Method | Stage | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | S. | EPDMS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LTF | S1 | 97.3 | 80.2 | 97.8 | 99.3 | 83.4 | 96.2 | 92.9 | 97.8 | 71.1 | 61.3 | |
| | S2 | 79.4 | 69.0 | 85.6 | 98.5 | 83.8 | 76.7 | 47.9 | 97.0 | 70.6 | 39.2 | 24.4 |
| DiffusionDrive | S1 | 96.8 | 86.0 | 98.8 | 99.3 | 84.0 | 95.8 | 96.7 | 97.6 | 79.6 | 66.7 | |
| | S2 | 80.1 | 72.8 | 84.4 | 98.4 | 85.9 | 76.6 | 46.4 | 96.3 | 72.8 | 40.5 | 27.5 |
| GTRS-DP | S1 | 94.7 | 78.8 | 96.1 | 99.5 | 83.0 | 94.4 | 92.0 | 97.5 | 72.8 | – | |
| | S2 | 80.3 | 74.4 | 84.9 | 98.0 | 81.9 | 78.8 | 45.4 | 96.7 | 70.1 | – | 23.8 |
| GuideFlow | S1 | 96.6 | 80.5 | 96.3 | 99.3 | 82.3 | 94.9 | 91.5 | 97.7 | 67.8 | – | |
| | S2 | 87.3 | 76.7 | 88.8 | 99.2 | 84.3 | 85.1 | 49.7 | 93.1 | 44.5 | – | 27.1 |
| ReCogDrive | S1 | 96.4 | 78.9 | 98.7 | 99.8 | 82.6 | 95.6 | 94.4 | 97.6 | 74.2 | 67.7 | |
| | S2 | 80.2 | 65.0 | 82.4 | 98.7 | 85.2 | 76.9 | 43.8 | 96.6 | 71.8 | 37.6 | 25.7 |
| SGDrive | S1 | 95.8 | 87.6 | 97.8 | 99.8 | 84.4 | 94.7 | 92.9 | 97.8 | 28.9 | 71.1 | |
| | S2 | 79.4 | 65.4 | 79.1 | 98.9 | 88.9 | 75.3 | 42.7 | 96.4 | 29.6 | 35.2 | 25.5 |
| Metis | S1 | 96.6 | 87.8 | 99.0 | 99.3 | 84.5 | 95.6 | 97.8 | 97.8 | 77.8 | 75.8 | |
| | S2 | 79.6 | 73.3 | 84.9 | 97.8 | 85.8 | 76.6 | 47.7 | 95.4 | 75.3 | 41.7 | 32.2 |
| **ReDrive** | S1 | 97.9 | 93.6 | 99.4 | 99.6 | 84.2 | 96.9 | 97.8 | 97.8 | 73.3 | **82.3** | |
| | S2 | 82.2 | 75.8 | 84.3 | 98.2 | 87.3 | 77.9 | 47.8 | 95.9 | 54.2 | **42.3** | **34.4** |

### Table 4: Visual pretraining (PDMS after Stage 2)

| Pretraining | DINOv2 | MAE | V-JEPA 2, 4 frames | 8 frames | 12 frames | 16 frames |
|---|---:|---:|---:|---:|---:|---:|
| PDMS | 84.4 | 83.9 | 89.9 | 90.4 | 90.7 | 90.8 |

DINOv2, MAE and the Stage-1 encoder have comparable parameter counts, start from their own released checkpoints, are "further pretrained on the same driving data for the same number of epochs" and go through the same Stage 2.

### Table 5: Encoder fine-tuning and future prediction in Stage 2 (4-frame pretrained encoder)

| Encoder | Future predictor | PDMS |
|---|:-:|---:|
| Frozen | ✗ | 85.7 |
| Trainable | ✗ | 89.2 |
| Trainable | ✓ | 89.9 |

### Table 6: The three stages (4-frame pretraining)

| Stage 1 | Stage 2 | Stage 3 | PDMS |
|:-:|:-:|:-:|---:|
| ✗ | ✗ | ✗ | 88.9 |
| ✓ | ✗ | ✗ | 89.2 |
| ✓ | ✓ | ✗ | 89.9 |
| ✓ | ✓ | ✓ | 90.2 |

Row 1 is the stock V-JEPA 2 checkpoint with encoder and planner trained jointly.

### Table 7: Frozen encoders with newly trained planners (8-frame pretraining)

| Encoder | Planner | PDMS |
|---|---|---:|
| 8-frame pretrained, frozen | Reinitialized | 86.7 |
| Stage-2 trained, frozen | Reinitialized | 90.2 |
| Stage-2 joint training (reference) | – | 90.4 |

### Table 8: Pretraining configuration

| Module | Setting |
|---|---|
| Encoder | ViT-L, depth 24, width 1024, 16 heads, 304M |
| Predictor | depth 12, width 384, 12 heads, 22M |
| Input | 16 frames, 10 Hz, 256×512, patch 16×16, tubelet 2 |
| Masking | 8 small blocks (scale 0.15), 2 large blocks (scale 0.70) |
| Optimization | AdamW, LR 5.25e-4, weight decay 0.04, bfloat16, 100 epochs |
| Target encoder | EMA, decay 0.99925 |

### Table 9: Model configuration for joint training

| Module | Setting |
|---|---|
| Encoder | ViT-L, depth 24, width 1024, 16 heads, 304M |
| Future predictor | depth 12, width 768, 12 heads, MLP ratio 4, 153M |
| Action DiT | depth 28, width 512, 16 heads, 103M |
| Target encoder | EMA, decay 0.9999 |

### Table 10: Optimization for joint training

| Configuration | Setting |
|---|---|
| Observed / predicted frames | 4 / 4 |
| Resolution | 256×512 |
| Optimizer | AdamW, β = (0.9, 0.95), weight decay 1e-5 |
| Learning rate | 3e-5, constant with 1,000 warm-up steps |
| Gradient clipping | 1.0 |
| Planning loss / prediction loss | flow-matching MSE / L1 |
| Prediction weight λ | 0.1 |
| Diffusion timesteps / inference steps | 1,000 / 5 |
| Precision | bfloat16 |

Joint training runs 35K steps on navtrain; Stage 3 runs 5K steps. The trajectory is 8 steps over 4 s at 2 Hz. All experiments use 16 NVIDIA H20 GPUs.

---

## Reading the Results

### 1. Where the 91.0 comes from

The ablations are run with 4-frame pretraining and end at 90.2. The headline uses 16-frame pretraining.

| Step | PDMS | Δ | Source |
|---|---:|---:|---|
| DINOv2 or MAE encoder, same pipeline through Stage 2 | 84.4 / 83.9 | | Table 4 |
| Frozen 4-frame driving-pretrained V-JEPA 2 + planner | 85.7 | | Table 5 |
| Stock V-JEPA 2, encoder trainable | 88.9 | | Table 6 |
| + Stage 1 driving-domain pretraining (4 frames) | 89.2 | **+0.3** | Table 6 |
| + Stage 2 future prediction | 89.9 | **+0.7** | Tables 5, 6 |
| + Stage 3 planner adaptation | 90.2 | **+0.3** | Table 6 |
| 16-frame instead of 4-frame pretraining (after Stage 2) | 90.8 | +0.9 | Table 4 |
| Headline (16 frames, all stages) | 91.0 | +0.2 over 90.8 | Table 1 |

- **The largest effects are the backbone and whether it is trained**: V-JEPA 2 over DINOv2 is +5.5, and unfreezing the encoder is +3.5.
- **The three stages add +1.3 in total at 4 frames.** Each of the three increments is 0.3–0.7, single runs with a stochastic five-step sampler. By the wiki's working rule ([[concepts/evaluation-variance.md]]), Stage 1 and Stage 3 are at the "no detectable effect" level taken individually.
- **Pretraining clip length matters more than any single stage** (+0.9 from 0.4 s to 1.6 s clips), and is monotone across four settings.

### 2. What "shaping representations with world modeling" is worth {#named-mechanism}

The introduction claims "a 3.5-point PDMS improvement over representations learned without our complete training pipeline", from Table 7: a frozen pretrained encoder with a fresh planner scores 86.7, a frozen Stage-2 encoder with a fresh planner scores 90.2.

That comparison changes two things at once. The Stage-2 encoder was fine-tuned by the planning loss and by the future-prediction loss. Table 5 separates them:

| What changes | Δ PDMS |
|---|---:|
| Encoder frozen → trainable under the planning loss alone | **+3.5** (85.7 → 89.2) |
| + future prediction | **+0.7** (89.2 → 89.9) |

So Table 7 shows that Stage 2's gain lives in the encoder and survives a planner reset, which is a useful result. It does not show that world modeling put it there. **The missing row is a frozen encoder from Stage 2 *without* the future predictor, with a fresh planner.** On the paper's own numbers, about five-sixths of the representational gain is ordinary end-to-end fine-tuning.

This is one more paper in which the mechanism in the title is the smaller term of its own ablation.

### 3. Regression on future latents: a third sign {#regression-sign}

| Paper | Target | Objective | Conditioned on | Prediction reaches the planner? | Effect vs. no future prediction |
|---|---|---|---|---|---:|
| [[sources/wa-jepa.md]] | EMA ViT-L scene latents, 4 cameras | Regression | Noisy actions (stop-gradient) | **Yes**, the action stream reads it | **−0.4** |
| [[sources/drivefuture.md]] | 16-token BEV latent | MSE | – | Yes, as a condition | +1.2 |
| **ReDrive** | EMA ViT-L latents of 4 future frames, 1 camera | L1 after normalization | **Ground-truth trajectory** | **No** | **+0.7** |

The wiki's standing explanation is that regression fails in proportion to the entropy of the target ([[concepts/world-model-for-ad.md#objective-form]]). ReDrive's target is a full ViT-L token grid, as high-dimensional as WA-JEPA's per camera, and regression helps. Two differences could explain it, and the paper tests neither:

- **Conditioning on the true trajectory removes the largest source of multimodality.** Most of the uncertainty in a front-camera future is what the ego does. Given the recorded trajectory, the conditional mean is much closer to a real future.
- **A blurred prediction cannot hurt a planner that never reads it.** In WA-JEPA the action stream attends to predicted future tokens, so a collapsed prediction is an input. In ReDrive the prediction is a loss on the encoder and is then thrown away.

Either way, "deterministic regression on scene latents is harmful" does not hold once the prediction is action-conditioned and discarded.

### 4. Stage 3 is a differentiable goal-matching loss

In Stage 3 the planner minimizes, through a frozen world model,

$$\big\|\operatorname{Norm}\big(P_\phi(Z_h,\hat A(\hat\tau))\big)-\operatorname{Norm}(Z_f^{\mathrm{tgt}})\big\|_1$$

where $Z_f^{\mathrm{tgt}}$ is the representation of the future that was actually recorded. The planner is rewarded for producing the trajectory that, according to the world model, leads to the expert's future. This is an imitation loss expressed in representation space.

- **It is the gradient form of [[sources/ad-e2e-jepa.md]]'s search.** That paper selects, from a vocabulary, the trajectory whose rollout is nearest the real future frame and recovers 67–73 EPDMS from that signal alone. ReDrive applies the same signal as an auxiliary gradient on a trained planner and gains 0.2–0.3 PDMS. Both say the signal is real and weak next to direct trajectory supervision.
- **It reopens the path Stage 2 closed.** Stage 2 keeps the feature loss away from the planner so the policy is not steered "toward actions that are easier to predict". Stage 3 sends exactly that loss into the planner. The difference is that the encoder and predictor are now frozen, and the paper does not discuss the tension.
- **Its effect is not isolated.** Stage 3 adds three things: a flow-matching loss at every step of an on-policy rollout, an L1 loss on the final generated trajectory, and the feature loss. Table 6 switches all three together. The +0.3 could come entirely from training through the sampler.

### 5. A third encoder sweep, with one new control

| Encoder | Drive-JEPA (PDMS) | WA-JEPA (EPDMS) | **ReDrive (PDMS)** |
|---|---:|---:|---:|
| DINOv2 | 76.1 | – | 84.4 |
| DINOv3 | – | 83.8 | – |
| MAE | did not converge | 83.8 | 83.9 |
| V-JEPA 2 | 86.1 | 89.5 | 88.9 stock / 89.9 (4-frame driving-pretrained) |

Three papers, same ordering: V-JEPA 2 leads the DINO-family row by +10, +5.7 and +5.5. ReDrive adds a control the other two lack: **the image encoders receive the same driving-domain pretraining and the same Stage 2, including the future-prediction loss.** They still trail by 5.5–6.0. So neither driving data nor a future-prediction objective applied during fine-tuning closes the gap.

Two things remain open:
- Every alternative is still image-level. There is no video-pretrained non-JEPA encoder, the same missing arm as in the other two papers.
- How DINOv2 and MAE were "further pretrained" (with which objective) and how an image encoder is applied to four frames are not described.

The frame sweep is the most useful new evidence on this question. Within one objective and one backbone, longer pretraining clips give 89.9 → 90.4 → 90.7 → 90.8. Temporal context during pretraining has a measurable, monotone, saturating effect.

**The price of driving-domain adaptation is small here.** Stock V-JEPA 2 with a trainable encoder already reaches 88.9. Stage 1 adds +0.3 at 4 frames and at most about +1.2 at 16 frames. Drive-JEPA measured +2.9 and WA-JEPA +1.5 to +2.2 for the same step. 100 epochs on 16 H20 GPUs is a large bill for that.

### 6. The v2 table {#v2-table}

- **90.8 is self-declared corrected**, and its residual agrees (closed form 90.25, −0.55).
- **Every comparison row is digit-identical, across all nine sub-scores, to the corrected column of [[sources/wa-jepa.md]]'s Table 1**: DiffusionDrive 84.5, DiffusionDriveV2 87.5, WAM-Diff 89.7, DreamerAD 87.7, Latent-WAM 89.3, DriveFuture 89.9, CoWorld-VLA 90.0, SparseDriveV2 90.1.
- **Three entries of that column are absent: WA-JEPA 91.7, Discrete-WAM 90.4 and DriveWorld-VLA 86.8.** The first is above ReDrive and the second would be the runner-up. WA-JEPA is not in ReDrive's reference list.
- The text says ReDrive "achieves the best overall performance and outperforms the second-best SparseDriveV2 by 0.7 points". Against the wiki's corrected cohort it is **third**, behind WA-JEPA 91.7 and SUV 91.0.

Against WA-JEPA, which uses the same V-JEPA 2 ViT-L family, ReDrive is lower on every sub-score (NC 99.1 vs 99.4, DAC 97.8 vs 98.2, TTC 98.7 vs 98.9, EC 86.4 vs 88.1). WA-JEPA uses four cameras and denoises future latents at inference; ReDrive uses one camera and predicts nothing. **0.9 EPDMS is the current price of that simplification**, across two papers and with camera count confounded.

### 7. Table problems {#table-problems}

| Row | What is printed | What it should be |
|---|---|---|
| DriveSuprim (v1) | NC 98.6, DAC 98.6, EP 91.3, TTC 95.5 → **89.9** | Those are the ViT-L model's sub-scores, whose PDMS is **93.5**. 89.9 is the ResNet-34 model (97.8 / 97.3 / 86.7 / 93.6). The row would place DriveSuprim below ReDrive; the model with those sub-scores is 2.5 above it |
| VADv2, UniAD (v1) | EP 91.6 / TTC 76.0; EP 92.9 / TTC 78.8 | EP and TTC are swapped (EP 76.0 / TTC 91.6; EP 78.8 / TTC 92.9) |
| Hydra-MDP++ (v1) | Inputs "C + L" | Camera only |
| Drive-JEPA (v1) | 89.0 | Correct for the block it is in: this is Drive-JEPA's perception-free baseline. Its full planner scores 93.3 with simulator-distilled supervision |

The DriveSuprim row is caught by the closed-form check introduced with [[sources/physwam.md]]: a genuine v1 PDMS sits above the closed form of its own sub-scores. Every other row here is +1.3 to +6.0 above; DriveSuprim's is **2.0 below**.

"Among perception-free approaches, ReDrive achieves the best PDMS of 91.0" omits [[sources/wa-jepa.md]] (91.8, perception-free, no scorer) and Drive-HWM (93.3 or 93.8).

The source also has caption slips: the text refers to the pretraining table as Tab. 5 and the stages table as Tab. 7, and cites "Tab. 10 and Tab. 10" for Tables 9 and 10.

### 8. navhard

- **34.4 combined** (Stage 1 82.3, Stage 2 42.3). In the wiki's unscored cohort it ranks below SpanVLA 40.1, PhysWAM 38.1, SUV 36.9, GeoWAM 36.6, EponaV2 36.1, 4D-WAM 35.9, World4Drive 34.9 and DriveFuture 34.6, and above NavFormer 34.1 and Metis 32.2.
- **It is the best navhard result for a planner with no future at inference.** The previous no-access entries were SUV's no-access variant (32.8) and Metis (32.2).
- **The profile is Stage-1-heavy.** Its Stage-1 score ties SUV's 82.3 for the best unscored value, and its Stage-1 DAC (93.6) is second only to SUV's 94.2. Its Stage-2 score (42.3) is below SUV (43.9) and PhysWAM (48.8). Extended comfort falls from 73.3 to 54.2 between stages.
- **Every baseline row is digit-identical to [[sources/metis.md]]'s Table 1**, including the LTF row that Metis copied from GTRS (24.4). The table therefore contains no method above 32.2. Which navhard split was run is not stated.

### 9. The future-predictor analysis shows sensitivity, not accuracy

Figure 4 holds the history fixed, shifts the ground-truth trajectory laterally by 15 offsets and compares the predicted future representations with each other. Similarity is about 0.97 between neighbours and falls to about 0.72 at the extremes.

This establishes that the predictor's output depends on the action. It does not establish that the dependence is correct, because predictions are compared with each other and never with a real future. The matching accuracy test is [[sources/ad-e2e-jepa.md]]'s hit rate: does the prediction under the true trajectory match the recorded future better than the predictions under shifted ones? ReDrive has everything needed to report it and does not. Stage 3's gain depends on exactly that property.

### 10. Cost

407M parameters at inference (304M + 103M), five flow steps, one front camera, four frames at 256×512. **No latency is reported**, which is a notable omission for a paper whose argument is inference simplicity. Training holds 560M trainable parameters plus an EMA encoder.

---

## Relationships

- **[[sources/wa-jepa.md]]**: the nearest method and the absent comparison. Same backbone family, same driving-domain JEPA pretraining step, same flow-matching planner, opposite choices on everything else: flow matching vs regression for the future, action reads the future vs never, generation at inference vs none, four cameras vs one. WA-JEPA leads by 0.9 EPDMS and 0.8 PDMS.
- **[[sources/drive-jepa.md]]**: the baseline in both qualitative figures and the closest predecessor. Drive-JEPA's perception-free model is a driving-pretrained V-JEPA encoder with a simple decoder (89.0). ReDrive is that design with a trainable encoder, a flow-matching DiT and action-conditioned future prediction (91.0).
- **[[sources/metis.md]]**: the same attention pattern in a different space. Metis lets the future video read the action and never lets the action read the future, on a video generator. ReDrive does it with JEPA latents and with the *ground-truth* action, so no gradient from the future loss reaches the planner at all. Metis is the one WAM in ReDrive's navhard table.
- **[[sources/simwam.md]]**: the other training-time-only world model from HUST, not cited. SimWAM reaches 91.5 PDMS with RL on a 5B video prior; ReDrive reaches 91.0 without RL on a 304M encoder.
- **[[sources/da-wam.md]]**: also keeps JEPA supervision live during planner training with an EMA target, and also predicts action-conditioned futures. DA-WAM keeps the prediction at inference and scores candidates with it; ReDrive discards it.
- **[[sources/ad-e2e-jepa.md]]**: two uses of one signal (see [point 4](#4-stage-3-is-a-differentiable-goal-matching-loss)). Its transfer experiment is also the sequential version of ReDrive's Stage 2: pretrain with action-conditioned prediction, then fine-tune for imitation.
- **[[sources/reworld.md]] / [[sources/drivelaw.md]]**: the previous "best perception-free" entries in ReDrive's v1 table, from a group sharing two senior authors. ReWorld improved a video generator's latents for a planner that reads them; ReDrive drops the generator.
- **[[sources/suv.md]]**: ties ReDrive on navhard Stage 1 (82.3) and leads overall (36.9) with inference-time access to a generated future. ReDrive sits between SUV's access and no-access variants.
- **Un-ingested**: DAWN (89.1 PDMS, "world-action interactive models"), GoalFlow (90.3), DriveDPO (90.0), GuideFlow, GTRS-DP, SparseDriveV2.

---

## Limitations

**Attribution**

1. **The title mechanism is +0.7 PDMS** in a pipeline where the backbone choice is +5.5 and encoder fine-tuning is +3.5.
2. **The "+3.5 from the full pipeline" claim is not isolated** from planning-loss fine-tuning. The needed control (Stage 2 without the predictor, frozen, fresh planner) is absent.
3. **Stage 3 bundles three losses** and is ablated only as a whole. $\lambda_{\mathrm{traj}}$ and $\lambda_{\mathrm{feat}}$ are not reported.
4. **All ablations are PDMS on navtest, single runs**, with increments of 0.3–0.9. No ablation is reported on v2 or navhard, where the wiki's other training-time mechanisms show their largest effects.
5. **Ablations stop at 4 or 8 frames; the headline uses 16.** The full pipeline's stage-by-stage behaviour at 16 frames is not shown.

**Comparison**

6. **The v2 table omits WA-JEPA (91.7) and Discrete-WAM (90.4)** while carrying every other row of the column they appear in, and claims the best result.
7. **The v1 DriveSuprim row is wrong** by 3.6 PDMS in ReDrive's favour, and two other rows have swapped columns.
8. **The navhard table is Metis's**, with no method above 32.2.

**Method and reporting**

9. **No latency, FPS or throughput.**
10. **Pretraining and fine-tuning frame rates differ** (10 Hz clips in Stage 1, 2 Hz frames in Stage 2), which is not discussed.
11. **The future predictor is never evaluated for accuracy**, only for sensitivity to the action.
12. **The pretraining corpus size is not given** beyond the 80 h PhysicalAI-AV subset.
13. **One front camera, NAVSIM only.** No closed-loop benchmark, no second dataset.
14. No code.

**Source conversion**

15. Figure 1 is present only as panel (a). Table captions in the text are off by one or two in places. The author line is in the body; the front matter's author field is empty.

---

## Key Cross-References

- [[concepts/world-model-for-ad.md]] — Pattern 40: an action-conditioned JEPA predictor as a representation shaper, then as a frozen differentiable critic; a third sign for regression on future latents.
- [[concepts/navsim-benchmark.md]] — the 91.0 and 90.8 rows; a v2 table that is another paper's column with its top removed; the DriveSuprim row.
- [[concepts/navhard-ood-evaluation.md]] — 34.4, the best no-access result.
- [[concepts/foundation-backbones-for-ad.md]] — a third V-JEPA 2 encoder sweep; clip length in pretraining; the adaptation ladder for a video encoder.
- [[concepts/wam-attention-masks.md]] — future reads the ground-truth action; planner isolated in Stage 2 and coupled through a frozen predictor in Stage 3.
- [[concepts/physicalai-av-benchmark.md]] — PhysicalAI-AV used as pretraining video for a NAVSIM model.
- [[concepts/inference-latency.md]] — another simplicity claim with no latency.
