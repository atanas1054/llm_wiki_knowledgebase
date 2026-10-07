---
title: "ForeDrive: Foresight-Guided End-to-End Autonomous Driving with a Planning-Relevant Latent World Model"
type: source-summary
sources: ["raw/papers/ForeDrive_ Foresight-Guided End-to-End Autonomous Driving with a Planning-Relevant Latent World Model.md"]
related: [concepts/world-model-for-ad.md, concepts/evaluation-variance.md, concepts/navsim-benchmark.md, concepts/diffusion-planner.md, concepts/foundation-backbones-for-ad.md, concepts/wam-attention-masks.md, concepts/inference-latency.md, concepts/nuscenes-waymo-evals.md, sources/diffusiondrive.md, sources/drivelaw.md, sources/redrive.md, sources/wa-jepa.md, sources/drive-jepa.md, sources/latent-wam.md, sources/drivefuture.md, sources/da-wam.md, sources/geoworldad.md, sources/coworld-vla.md, sources/suv.md, sources/policy-world-model.md, sources/ad-e2e-jepa.md, research-directions.md]
created: 2026-10-06
updated: 2026-10-06
confidence: medium
---

# ForeDrive: Foresight-Guided End-to-End Autonomous Driving with a Planning-Relevant Latent World Model

**Paper**: ForeDrive: Foresight-Guided End-to-End Autonomous Driving with a Planning-Relevant Latent World Model
**Authors**: Sinuo Wang, Zichong Gu, Yuhan Huang, Wenxin Wen, Xun Yang, Yiqing Zhang, Xingyu Zhang, Ningyu Che, Jie Ling, Qiankun Yu, Wei Liu, Jing Xu, Xinggang Wang
**Orgs**: Huazhong University of Science and Technology; Shanghai Zaofu Intelligent Technology (the affiliation markers are only partly preserved in the clipping). Xinggang Wang is also the senior author of [[sources/diffusiondrive.md]], whose planner ForeDrive reuses.
**arXiv**: 2609.26299v1 (September 2026)
**Code**: none stated
**Source**: `raw/papers/ForeDrive_ Foresight-Guided End-to-End Autonomous Driving with a Planning-Relevant Latent World Model.md`

---

## What It Is

A small, single-camera planner that **predicts the future in latent space and then plans with that prediction**, trained in one stage by imitation only.

1. **Encoder**: DINOv3 ViT-B/16 on one front-camera image (256×512), producing 512 patch tokens.
2. **Latent world model**: a 4-layer Transformer predicts DINOv3-space latents and ego status (velocity, acceleration, navigation command) at 1, 2, 3 and 4 s, in one forward pass under a frame-level block-causal mask. Targets come from an EMA copy of the encoder applied to the real future frames (JEPA-style, L1 loss).
3. **Planner**: DiffusionDrive's truncated-diffusion DiT (20 anchors, two 5-layer stages, 2 DDIM steps), conditioned on
   - current tokens fused with the predicted futures by gated cross-attention (current tokens are the queries and the residual path);
   - an ego memory of the current status plus the predicted future ego motion;
   - **Trajectory-Adaptive Bias (TAB)**: each candidate trajectory is projected into the image and adds a Gaussian proximity bias to its cross-attention logits, so it looks at the pixels along its own path.
4. **Asymmetric optimization**: planning gradients update the encoder; the predictor is trained only by the forecasting losses; injected futures are stop-gradient.

**Results** (no RL, no scorer, one camera):

| | ViT-B (default, 118.4M) | ViT-L |
|---|---:|---:|
| NAVSIM v1 PDMS | **89.9** (3 training seeds: 89.7–89.9, sd 0.1) | 90.4 |
| NAVSIM v2 EPDMS (corrected-like by residual) | **90.0** | 90.6 |
| nuScenes zero-shot L2 / collision (its own re-evaluation) | 0.48 m / 0.11% | – |
| Latency (H20, FP32, batch 1) | 58.2 ms (foresight +26 ms) | – |

---

## Key Takeaways

- **The first training-seed variance measurement on NAVSIM in the wiki.** The default recipe retrained from three seeds gives 89.7–89.9 PDMS, sd 0.1. Every earlier variance figure here was sampler noise with fixed weights. Three seeds of one recipe is a small sample, and the ablations themselves are single runs.
- **Predicting the future helps only if the planner reads the prediction.** As an auxiliary loss alone, latent forecasting is worth +0.1 PDMS (ViT-B); feeding the predicted futures to the planner brings the world model to +0.7. The pattern holds at ViT-S (+0.3 → +0.6) and ViT-L (+0.1 → +0.5).
- **Better forecasts made worse plans.** Across four training paradigms, the two with the most accurate forecasts (latent cosine 0.843 and 0.838) plan worst (87.9, 88.6) and the least accurate (0.759) plans best (89.9). A probe finds ForeDrive's predicted latents more useful for reading out the expert trajectory (ADE 1.55) than the real future latents (2.10).
- **The planner treats the future as context, not as a sequence.** Shuffling the four horizons at test time leaves PDMS unchanged (89.4); giving the planner the real future latents instead of its predictions adds only +0.2. Removing the futures costs 1.4, mostly progress and drivable area. **Feeding a wrong future is worse than none** (−2.4, with TTC down 2 points).
- **Blocking planning gradients into the predictor is worth +0.4** (89.6 vs 89.2) at matched final forecast quality. Second stop-gradient result in the wiki after [[sources/wa-jepa.md]].
- **TAB, a geometric attention prior, is worth as much as the world model**: +0.6 to +0.7 at every encoder size, and the two add (+1.0 to +1.6 together).
- **The "state of the art among imitation methods" is table-scoped.** Within the wiki, [[sources/redrive.md]] (one front camera, no RL, no scorer) scores 91.0 PDMS / 90.8 EPDMS against ForeDrive's 89.9 / 90.0 (ViT-L: 90.4 / 90.6), and is not in ForeDrive's tables. Its v2 table also mixes evaluator conventions (§3).

---

## Method

### Planning-relevant latent world model

- **Online / EMA encoding.** $z_0 = E_\theta(I_0) \in \mathbb R^{N\times d}$ ($N = 512$, $d = 768$). The EMA target encoder $E_{\bar\theta}$ copies only the vision encoder (decay 0.999) and encodes future frames into stop-gradient targets; future frames are never used at inference.
- **Causal latent prediction.**
$$\{(\hat z_t, \hat s_t)\}_{t\in\mathcal H} = P_\psi(z_0, s_0, Q_{\mathcal H}),\qquad \mathcal H = \{1,2,3,4\}\text{ s}$$
  Learned query tokens per horizon; a frame-level block-causal mask (earlier horizons visible to later ones, not the reverse); one forward pass, not an autoregressive rollout. Predictor: 4 pre-LN layers, width 512, 8 heads, FFN 2048.
- **Foresight supervision.**
$$\mathcal L_\text{lat} = \frac1H\sum_{t\in\mathcal H} w_t\,\ell_t,\qquad \mathcal L_\text{status} = \frac1H\sum_{t\in\mathcal H}\big[\text{CE}(\hat s^c_t, s^c_t) + \text{MSE}(\hat s^m_t, s^m_t)\big]$$
  $\ell_t$ is token-averaged L1 against the EMA latents (no patch mask); $w_t \propto 1/(i+1)$ on the 2 Hz index $i$, rescaled to unit mean; $s^c$ is the 4-way navigation command and $s^m$ velocity and acceleration.

### Foresight-guided generative planning

- **Current-primary latent fusion.** Current tokens $\bar c$ and futures $\bar f_t$ are projected to width 256 with shared spatial and per-horizon time embeddings; each horizon gets a **sample-shared** sigmoid gate $g^\text{vis}_t$ (a learned dataset-level weight, not a per-sample reliability). One cross-attention layer: queries $\bar c$, keys and values the concatenated gated futures, residual on $\bar c$.
- **Future-status injection.** $\text{sg}(\hat s^m_t)$ embedded, gated by a second sample-shared gate, concatenated with the current ego embedding: $E_\text{ego} = [e_0; \tilde e_1; \dots; \tilde e_H]$, read by every decoder layer. Predicted navigation commands are not injected.
- **Planner.** DiffusionDrive's truncated diffusion over 20 k-means anchors; two cascaded 5-layer DiT stages with separate weights; trajectory $\tau \in \mathbb R^{8\times 3}$ over 4 s; hard closest-anchor assignment, focal classification loss and L1 regression on the winning mode:
$$\mathcal L_\text{plan} = \sum_{k=1}^{2}\big(\lambda_\text{cls}\mathcal L^{(k)}_\text{focal} + \lambda_\text{reg}\mathcal L^{(k)}_1\big)$$
  Inference: two truncated DDIM steps from $t\approx 8$; the mode is the classification argmax, with no external scorer.
- **Trajectory-Adaptive Bias.** Each candidate's poses are projected into the front camera (valid if depth > $10^{-3}$ and inside the image). For each mode-token pair the maximum Gaussian affinity (bandwidth 0.25) to the projected waypoints becomes a log-bias (floor $10^{-6}$) added to the trajectory-to-image attention logits in every decoder layer, recomputed at stage boundaries. A mode with no valid projection gets a constant shift, which leaves its softmax unchanged.

### Asymmetric latent optimization

$$\mathcal L_\text{plan} \rightarrow E_\theta,\ \mathcal F,\ \text{planner},\qquad \mathcal L_\text{plan} \not\rightarrow P_\psi$$

$$\mathcal L = \lambda_\text{traj}\mathcal L_\text{plan} + \lambda_z\mathcal L_\text{lat} + \lambda_s\mathcal L_\text{status}$$

One stage, 100 epochs, AdamW $6\times10^{-4}$, batch 1,024 on 32 H20 GPUs; loss weights classification / regression / trajectory / latent / status = 10 / 8 / 12 / 5 / 0.1. ViT-B uses layer-wise learning-rate decay 0.7 and stochastic depth 0.15; ViT-S is fully fine-tuned; ViT-L uses decay 0.8. Checkpoints are selected on 18,179 held-out navtrain samples; navtest is used only for final reporting.

---

## Figures

*Figure 2 (accuracy against model size on NAVSIM; its data are Table 10 below) and supplementary Figure 3 (early-training forecast probes) are not in the clipping.*

![[Fig1.png|Three rows comparing representation paradigms. (a) Reactive planner: encoder, current latent, planner, trajectory, with no future. (b) Latent world model: an encoder and world model predict a future latent; separately, a frozen world model feeds a planner. (c) ForeDrive: world-model loss and trajectory loss both shape the encoder; the current latent goes to both the world model and the planner, and the predicted future latent also enters the planner]]

*Figure 1: (a) Reactive planner, (b) latent world model used for pretraining or auxiliary loss, (c) ForeDrive.*

![[Fig3.png|Architecture diagram. Top, latent world model: a trainable vision encoder encodes the current image; with an ego-state token it enters a causal Transformer with a block-causal mask; outputs are compared by a status loss with future ego states and by a latent loss with a frozen EMA encoder applied to future ground-truth images at t+1 to t+T; a detach marker stops gradients. Bottom, diffusion planner: noisy plan anchors are iteratively denoised; latent fusion uses current tokens as queries and horizon-gated future latents as keys and values; an ego query feeds in; Trajectory-Adaptive Bias projects candidate trajectories into the camera image to form an attention prior heatmap; the selected trajectory is output]]

*Figure 3: Overview.*

![[Fig4.png|Three driving scenes in rows, each shown for ForeDrive, DiffusionDrive and DriveLaW with the expert trajectory in green and the prediction in red. (a) A right turn out of a driveway; ForeDrive's prediction follows the expert closely while the others diverge. (b) A left turn at an intersection; ForeDrive curves left near the expert, while DiffusionDrive and DriveLaW go straight. (c) A right turn across a crosswalk; all three are offset from the expert, ForeDrive least]]

*Figure 4: Trajectories against DiffusionDrive and DriveLaW (green: expert; red: prediction).*

![[SupFig3.png|Four pairs of front-camera images with attention heatmaps from the planner's trajectory queries. Without TAB the attention is diffuse over the lower image and sky; with TAB it concentrates on a band of road along the selected trajectory, drawn in yellow with the expert in cyan]]

*Supplementary Figure 1: Trajectory-to-image attention without and with TAB (ViT-B Base, no world model).*

![[SupFig1.png|Three scenes, each with the front camera image and two bird's-eye plans: planning with future and planning without future. (a) A multi-agent intersection where the plan without future collides with a turning car. (b) A right turn where the plan without future collides with a vehicle in the crossing lane. (c) A left turn where the plan without future leaves the drivable area; the plans with future follow the expert]]

*Supplementary Figure 2: Full model against Base + TAB (no world model). Per the text, per-scene PDMS 0 without futures in all three, and 1.00 / 0.58 / 1.00 with them. Selected examples, not a controlled ablation.*

![[SupFig2.png|Four scenes. For each, the current front image and a bird's-eye plan, then four pairs of ground-truth future frames and RGB decodings of the predicted future latents at t equals 1 to 4. The decodings are blurry but keep road layout, nearby vehicles and buildings, and change viewpoint consistently with the ego motion]]

*Supplementary Figure 4: An RGB probe trained on the frozen encoder's EMA latents, applied to the predicted futures. Diagnostic only.*

---

## Tables

### Table 1: NAVSIM v1 navtest

C = camera, C+L = camera + LiDAR. The last column is computed here: reported PDMS minus the closed form of its sub-scores ([[concepts/navsim-benchmark.md#physwam]]).

| Method | Venue | Mod. | NC | DAC | TTC | Comf. | EP | PDMS | Reported − closed form |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| *Traditional end-to-end* | | | | | | | | | |
| UniAD | CVPR'23 | C | 97.8 | 91.9 | 92.9 | 100.0 | 78.8 | 83.4 | +4.1 |
| TransFuser | TPAMI'23 | C+L | 97.7 | 92.8 | 92.8 | 100.0 | 79.2 | 84.0 | +3.9 |
| PARA-Drive | CVPR'24 | C | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 | +4.0 |
| DRAMA | ISRR'24 | C+L | 98.0 | 93.1 | 94.8 | 100.0 | 80.1 | 85.5 | +3.8 |
| ReCogDrive-IL | ICLR'26 | C | 98.1 | 94.7 | 94.2 | 100.0 | 80.9 | 86.5 | +3.2 |
| PRIX | RA-L'26 | C | 98.1 | 96.3 | 94.1 | 100.0 | 82.3 | 87.8 | +2.6 |
| DiffusionDrive | CVPR'25 | C+L | 98.2 | 96.2 | 94.7 | 100.0 | 82.2 | 88.1 | +2.7 |
| MeanFuser | CVPR'26 | C | 98.6 | 97.0 | 95.0 | 100.0 | 82.8 | 89.0 | +2.2 |
| DiffRefiner-R34 | AAAI'26 | C | 98.4 | 97.4 | 95.3 | 100.0 | 83.4 | 89.4 | +2.1 |
| *World-model and video–action* | | | | | | | | | |
| LAW | ICLR'25 | C | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 | +4.0 |
| Epona | ICCV'25 | C | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 | +3.1 |
| DriveVLA-W0 | ICLR'26 | C | 98.4 | 95.3 | 95.2 | 100.0 | 80.9 | 87.2 | +2.8 |
| PWM | NeurIPS'25 | C | 98.6 | 95.9 | 95.4 | 100.0 | 81.8 | 88.1 | +2.5 |
| WoTE | ICCV'25 | C+L | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 | +2.2 |
| DriveLaW | CVPR'26 | C | 99.0 | 97.1 | 96.7 | 100.0 | 81.3 | 89.1 | +1.8 |
| **ForeDrive** | – | C | 98.6 | 97.6 | 95.6 | 100.0 | **83.8** | **89.9** | +1.9 |
| ForeDrive (ViT-L) | – | C | 98.7 | 97.9 | 96.1 | 100.0 | 84.3 | 90.4 | +1.7 |

Every row passes the closed-form check (+1.7 to +4.1). DriveVLA-W0's 87.2 is its flow-matching-head variant, not its 90.2 / 93.0 headlines.

### Table 2: NAVSIM v2 navtest (one-stage EPDMS)

The last column is computed here: closed form of the printed sub-scores minus reported ([[concepts/navsim-benchmark.md#residual-sign]]).

| Method | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS | Residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *Traditional end-to-end* | | | | | | | | | | | |
| TransFuser | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 | +1.3 |
| ReCogDrive-IL | 98.2 | 94.5 | 99.3 | 99.9 | 87.4 | 97.3 | 97.1 | 98.3 | 87.2 | 86.6 | −1.0 |
| PRIX | 98.0 | 95.6 | 99.5 | 99.8 | 87.4 | 97.2 | 97.1 | 98.3 | 87.6 | 84.2 | +2.4 |
| DiffusionDrive | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 | +2.5 |
| DiffRefiner-R34 | 98.5 | 97.4 | 99.6 | 99.8 | 87.6 | 97.7 | 97.7 | 98.3 | 86.2 | 86.2 | +2.7 |
| MeanFuser | 98.3 | 97.2 | 99.6 | 99.8 | 87.6 | 97.4 | 97.3 | 98.3 | 88.2 | 89.5 | −0.9 |
| *World-model and video–action* | | | | | | | | | | | |
| World4Drive | 97.8 | 96.3 | 99.4 | 99.8 | 88.3 | 97.1 | 97.7 | 98.0 | 53.9 | 84.8 | −1.5 |
| Epona | 97.1 | 95.7 | 99.3 | 99.7 | 88.6 | 96.3 | 97.0 | 98.0 | 67.8 | 85.1 | −1.7 |
| DriveVLA-W0 | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 | −1.3 |
| WorldRFT | 97.8 | 96.5 | 99.5 | 99.8 | 88.5 | 97.0 | 97.4 | 98.1 | 69.1 | 86.7 | −1.4 |
| Drive-JEPA | 98.4 | 98.6 | 99.1 | 99.8 | 88.4 | 97.8 | 97.6 | 97.9 | 84.8 | 87.8 | +1.7 |
| Latent-WAM | 98.1 | 97.3 | 99.6 | 99.8 | 87.7 | 97.3 | 97.6 | 98.1 | 87.3 | 89.3 | −0.9 |
| **ForeDrive** | 98.6 | 97.6 | 99.6 | 99.8 | 87.6 | 97.9 | **97.9** | 98.3 | 87.5 | **90.0** | **−0.6** |
| ForeDrive (ViT-L) | 98.7 | 97.9 | 99.6 | 99.9 | 87.6 | 98.2 | 97.7 | 98.4 | 88.1 | 90.6 | −0.6 |

### Table 3: Zero-shot nuScenes (VAD / ST-P3 protocol, all rows re-evaluated by the authors)

| Method | L2 1 s | L2 2 s | L2 3 s | L2 avg | Col. 1 s | Col. 2 s | Col. 3 s | Col. avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PWM (FT, nuScenes-trained) | 0.41 | 0.75 | 1.17 | 0.78 | 0.02 | 0.10 | 0.35 | 0.16 |
| Epona | 0.96 | 1.59 | 2.32 | 1.62 | 0.09 | 0.27 | 0.67 | 0.34 |
| DriveLaW | 0.30 | 0.48 | 0.83 | 0.54 | 0.23 | 0.16 | 0.19 | 0.19 |
| **ForeDrive** | **0.22** | **0.44** | **0.79** | **0.48** | **0.01** | **0.09** | 0.23 | **0.11** |

### Table 4: World model × TAB (ViT-B, NAVSIM v1)

| Setting | WM | TAB | PDMS | Δ vs Base |
|---|:-:|:-:|---:|---:|
| DiffusionDrive (external reference, C+L) | | | 88.1 | – |
| Base | | | 88.9 | – |
| Base + TAB | | ✓ | 89.5 | +0.6 |
| Base + WM | ✓ | | 89.6 | +0.7 |
| Base + WM + TAB | ✓ | ✓ | **89.9** | **+1.0** |

### Table 5: World-model components, TAB off (ViT-B)

| Setting | Aux. loss | Visual injection | Status injection | PDMS | Δ vs Base |
|---|:-:|:-:|:-:|---:|---:|
| Base | | | | 88.9 | – |
| Auxiliary | ✓ | | | 89.0 | +0.1 |
| Visual only | ✓ | ✓ | | 89.4 | +0.5 |
| Status only | ✓ | | ✓ | 89.3 | +0.4 |
| Full WM | ✓ | ✓ | ✓ | **89.6** | **+0.7** |

### Table 6: Training paradigms

| Paradigm | $\mathcal L_\text{lat}$ ↓ | Latent cosine ↑ | $\mathcal L_\text{status}$ ↓ | PDMS |
|---|---:|---:|---:|---:|
| Two-stage, frozen world model | **5.93** | **0.843** | 0.604 | 87.9 |
| Joint, encoder detached from planning | 6.07 | 0.838 | 0.599 | 88.6 |
| Joint, auxiliary only (no injection) | 7.44 | 0.769 | 0.536 | 89.6 |
| Full joint | 7.64 | 0.759 | **0.531** | **89.9** |

### Table 7: Future-injection interfaces (TAB and status injection off)

| Interface | PDMS | Δ vs gated |
|---|---:|---:|
| **Gated fusion** | **89.4** | 0.0 |
| Current only | 88.9 | −0.5 |
| Future only | 84.2 | −5.2 |
| Concatenation | 89.2 | −0.2 |
| Dual memory | 89.0 | −0.4 |
| Ungated fusion | 89.0 | −0.4 |

### Table 8: Encoder capacity (DINOv3; vision-encoder parameters)

| Encoder | Params | Base | Full | Δ |
|---|---:|---:|---:|---:|
| ViT-S/16 | 21M | 87.4 | 89.0 | +1.6 |
| ViT-B/16 | 86M | 88.9 | 89.9 | +1.0 |
| ViT-L/16 | 300M | 89.3 | 90.4 | +1.1 |

### Supplementary Table 1: From DiffusionDrive to the ViT-S Base (NAVSIM v1)

| Step | Configuration | PDMS |
|---|---|---:|
| Ref. | DiffusionDrive (camera + LiDAR BEV) | 88.1 |
| (a) | DINOv3 ViT-S, front camera only | 86.4 |
| (b) | (a) + current-status FiLM | 86.6 |
| (c) | (b) + DiT deepened from 1 to 5 layers per stage (= ViT-S Base) | 87.4 |

### Supplementary Table 2: Interventions on the future latents at test time (gated fusion only; TAB and status off)

| Mode | NC | DAC | EP | TTC | C | DDC | PDMS | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Oracle (EMA latents of the real future) | 98.9 | 97.0 | 83.5 | 96.0 | 100.0 | 98.5 | 89.6 | +0.2 |
| Horizon shuffle | 98.7 | 97.1 | 83.6 | 95.4 | 100.0 | 98.3 | 89.4 | +0.0 |
| **Normal (predicted)** | 98.7 | 97.1 | 83.6 | 95.4 | 100.0 | 98.3 | **89.4** | 0 |
| Token mask (75%) | 98.9 | 96.7 | 82.7 | 95.7 | 100.0 | 98.4 | 89.0 | −0.4 |
| Zero | 98.9 | 95.8 | 81.2 | 95.9 | 99.9 | 98.4 | 88.0 | −1.4 |
| Persistence (current latent tiled) | 98.3 | 95.6 | 81.7 | 93.7 | 100.0 | 98.1 | 87.0 | −2.4 |
| Cross-sample (another scene's prediction) | 98.0 | 95.9 | 81.7 | 93.2 | 99.9 | 98.3 | 87.0 | −2.4 |

### Supplementary Table 3: Predicted horizon sets

| Horizons (s) | Without TAB | With TAB |
|---|---:|---:|
| {1} | 89.1 | 89.6 |
| {2} | 89.3 | 89.8 |
| {1, 2} | 89.3 | 89.7 |
| {1, 2, 3, 4} | **89.6** | **89.9** |

### Supplementary Tables 4–7: Encoder-scale replications

| | ViT-S | ViT-B (main) | ViT-L |
|---|---:|---:|---:|
| Base | 87.4 | 88.9 | 89.3 |
| + TAB | 88.1 (+0.7) | 89.5 (+0.6) | 90.0 (+0.7) |
| + WM | 88.0 (+0.6) | 89.6 (+0.7) | 89.8 (+0.5) |
| + WM + TAB | 89.0 (+1.6) | 89.9 (+1.0) | 90.4 (+1.1) |
| WM, auxiliary only | 87.7 (+0.3) | 89.0 (+0.1) | 89.4 (+0.1) |
| WM, visual injection | 87.8 (+0.4) | 89.4 (+0.5) | 89.5 (+0.2) |
| WM, status injection | 87.9 (+0.5) | 89.3 (+0.4) | 89.6 (+0.3) |
| WM, both injections | 88.0 (+0.6) | 89.6 (+0.7) | 89.8 (+0.5) |

The ViT-S and ViT-L columns combine Supplementary Tables 4–7; the ViT-B column is from main Tables 4 and 5.

### Supplementary Table 8: Stop-gradient on the injection path (ViT-B, TAB off)

| Setting | $L_1$ ↓ | Cosine ↑ | $\mathcal L_s$ ↓ | PDMS |
|---|---:|---:|---:|---:|
| **Inject + detach** | 8.06 | 0.743 | 0.532 | **89.6** |
| Inject + joint | 8.08 | 0.747 | 0.514 | 89.2 |

### Supplementary Table 9: Frozen-backbone trajectory probe (two-layer MLP on one latent source)

| Source | ADE ↓ | FDE ↓ | ADE, horizons shuffled |
|---|---:|---:|---:|
| **ForeDrive predicted futures** | **1.55** | **2.77** | 2.52 |
| Prediction-optimized model's predicted futures | 1.83 | 3.24 | 3.38 |
| EMA latents of the real future | 2.10 | 3.61 | 2.93 |
| Current frame only | 2.31 | 4.08 | – |

### Supplementary Table 10: Parameters and scores for Figure 2

| Method | Params (M) | PDMS | EPDMS |
|---|---:|---:|---:|
| TransFuser | 55.9 | 84.0 | 76.7 |
| PRIX | 37 | 87.8 | 84.2 |
| DiffusionDrive | 60.7 | 88.1 | 84.5 |
| MeanFuser | 54.6 | 89.0 | 89.5 |
| DiffRefiner-R34 | 74.8 | 89.4 | 86.2 |
| DriveSuprim-R34 | 61 | 89.9 | 83.1 |
| Epona | 2,500 | 86.2 | 85.1 |
| DriveVLA-W0 | 7,500 | 87.2 | 86.1 |
| **ForeDrive** | **118.4** | **89.9** | **90.0** |

### Results given only in the text

| Quantity | Value |
|---|---|
| Training-seed spread (ViT-B default) | 89.7–89.9 PDMS over 3 seeds, sd 0.1; the tables report the primary run |
| Latency | 58.2 ms per frame (17.2 FPS), 1.03 GB peak, one H20, FP32, batch 1; +26 ms for latent foresight over Base |
| Parameters | 118.4M at evaluation (EMA encoder excluded) |
| Horizon permutation at test time (TAB and status off) | 89.4, unchanged |
| Zeroing futures at test time (same setting) | 88.0 |
| Train / validation split | 85,109 / 18,179 navtrain samples |

---

## Reading the Results

### 1. Training-seed variance {#seed-variance}

- **This is the quantity the wiki has been missing.** Every variance figure on [[concepts/evaluation-variance.md]] fixes the weights and varies the sampler or the scene set. ForeDrive retrains its default recipe three times: 89.7–89.9 PDMS, sd 0.1.
- **What it supports.** For this recipe on navtest, differences of a few tenths between two separately trained runs are within the spread of three seeds; the paper's +1.0 (Base → Full) is about ten standard deviations, its +0.7 world-model and +0.6 TAB effects six or seven.
- **What it does not.** Three seeds give a rough estimate of the sd. The ablations themselves are single runs, Base was not re-seeded, and the paper's 89.9 headline is "the primary run", the top of its own seed range. A different architecture (for example a large video model fine-tuned briefly) can have a very different seed spread.

### 2. A future the planner reads, not one it forecasts {#context-not-forecast}

| Evidence | Reading |
|---|---|
| Auxiliary forecasting alone: +0.1 (ViT-B), +0.3 (ViT-S), +0.1 (ViT-L) | A forecasting loss shapes the encoder very little |
| Injection: +0.4 to +0.6 more | The gain is in consuming the prediction |
| Horizon shuffle at test: 0.0 | The planner does not use the horizons' order |
| Real future latents at test: +0.2 | A better forecast barely helps a planner trained on predictions |
| Most accurate forecasts (Table 6) plan worst | Forecast accuracy and planning value move apart |
| Predicted latents beat real future latents in a trajectory probe (ADE 1.55 vs 2.10) | Planning gradients made the predicted latents carry the plan |

- **The predicted "future" behaves like extra planning-shaped context computed from the present frame.** That is a training-time-shaped feature (planning gradients reach the encoder, the predictor regresses toward EMA targets), and its value does not depend on matching the real future.
- **It matches what the wiki's [shared-future](../concepts/world-model-for-ad.md) results suggest and qualifies them.** [[sources/da-wam.md]] found a shared future *harmful* (92.81 vs 93.31 without one); [[sources/geoworldad.md]] found it worth +1.7 / +2.8; ForeDrive finds +0.4 to +0.7 at matched training. None of the three tests whether the gain needs the future at all, except ForeDrive's probes, which say it largely does not.
- **The objective-form rule.** ForeDrive regresses (L1) onto single-frame front-camera DINOv3 latents, consumes the regressed predictions, and gains from them. [[sources/wa-jepa.md]] found regression on multi-view latents worse than no future (90.7 vs 91.1). The wiki's entropy reading predicts this: one camera at 512 tokens is a lower-entropy target than four. The predictions are over-smoothed (the paper says so), which is the expected regression failure, yet harmless here because the planner uses them as context.

### 3. The comparison tables {#tables}

- **v1 rows all pass the closed-form check**, and DriveVLA-W0 is carried at its 87.2 variant.
- **v2 mixes conventions.** By residual sign, five rows are pre-fix-like (TransFuser +1.3, PRIX +2.4, DiffusionDrive +2.5, DiffRefiner +2.7, Drive-JEPA +1.7) and nine corrected-like, including ForeDrive (−0.6). So "+2.2 over Drive-JEPA" compares across evaluators; the same-protocol margins are +0.5 over MeanFuser (89.5) and +0.7 over Latent-WAM (89.3). ReCogDrive-IL appears at 86.6 with a corrected-like residual, a row the wiki has not seen before (its other ReCogDrive rows are 83.6, pre-fix).
- **Where it sits in the wiki.** Among v1 single-pass entries with no RL and no inference-time scorer, 89.9 is below Drive-HWM (93.3–93.8), ReDrive 91.0, DriveVA 90.9, SUV 90.8 and ReWorld 90.4, and level with DriveSuprim's ResNet-34 row; WA-JEPA (91.8) uses four cameras. In the corrected v2 cohort, 90.0 ties CoWorld-VLA and is below WA-JEPA 91.7, MM-Future 91.5, SUV 91.0, ReDrive 90.8, Discrete-WAM 90.4 (not ingested), PhysWAM 90.3, and SparseDriveV2 and MomWorld at 90.1. Its ViT-L 90.6 would be fifth. See [[concepts/navsim-benchmark.md#foredrive]].
- **What it leads**: per parameter. Among entries at or above 90.0 corrected EPDMS that report a size, 118.4M is the smallest (ReDrive 407M, CoWorld-VLA 2B, SUV about 6B, PhysWAM 15.2B), and 58 ms is among the fastest camera planners recorded ([[concepts/inference-latency.md]]).

### 4. The nuScenes table

- All four rows were re-evaluated by the authors "under a shared metric implementation". PWM's L2 matches its own paper (0.78) but its collision does not (0.16 here, 0.07 in [[sources/policy-world-model.md]] without ego status), so these collision rates are not comparable with other tables.
- ForeDrive's 0.48 m L2 average is under a different averaging convention from the zero-shot WAM cluster on [[concepts/nuscenes-waymo-evals.md]] (DriveVA 0.84, SimWAM 0.96), so neither ordering can be read across.

### 5. TAB

- **A geometric prior, not a learned module**: candidate trajectories projected into the image bias which tokens each candidate attends to. +0.6 / +0.7 at every encoder size, additive with the world model.
- It is the single-camera counterpart of DiffusionDrive's deformable spatial cross-attention, whose removal cost DiffusionDrive 32 PDMS ([[concepts/diffusion-planner.md]]). The attention maps (Supplementary Figure 1) show it doing what it is built to do: attention moves from the sky and lower image onto the road along the plan.

### 6. Scale and substitution

- The combined gain shrinks from ViT-S (+1.6) to ViT-B/L (+1.0 / +1.1); foresight alone from +0.6 to +0.5. Better current features leave less for foresight to add, the same direction as [[sources/coworld-vla.md]]'s representation × planner interaction. Single runs.
- An image-pretrained encoder (DINOv3 ViT-L) with a JEPA-style predictive loss reaches 90.4 PDMS. [[sources/redrive.md]] reports V-JEPA 2 ViT-L at 91.0 and DINOv2 under a predictive loss at 84.4 in its own planner. The planners differ, so this is not a controlled comparison, but the image-versus-video gap looks much smaller here.

---

## Relationships

- **[[sources/diffusiondrive.md]]**: the planner, the anchors and the truncated-diffusion recipe are DiffusionDrive's; the senior author is shared. Dropping LiDAR and BEV for one front camera costs 1.7 PDMS before anything else is added.
- **[[sources/redrive.md]]**: the closest comparison in the wiki and absent from ForeDrive's tables: one front camera, JEPA-style future prediction, flow/diffusion planner, no RL, no scorer, 91.0 / 90.8. ReDrive predicts nothing at inference; ForeDrive consumes its prediction.
- **[[sources/wa-jepa.md]]**: the other stop-gradient between future and action, and the regression-is-harmful result that ForeDrive's setting does not reproduce.
- **[[sources/drive-jepa.md]]**, **[[sources/latent-wam.md]]**: JEPA-style or compact latent world models used as pretraining or auxiliary signals, the paradigm ForeDrive contrasts itself with (Figure 1b).
- **[[sources/drivelaw.md]]**, **[[sources/drivefuture.md]]**: planners conditioned on a future; both found that a clean or oracle future hurts a planner trained otherwise. ForeDrive's oracle intervention (+0.2) is the mild version of the same mismatch.
- **[[sources/da-wam.md]]**, **[[sources/geoworldad.md]]**: the other shared-future measurements (§2).
- **[[sources/coworld-vla.md]]**: the other single-front-camera, no-RL, no-scorer entry at 90.0 corrected EPDMS.

---

## Limitations

**Evidence**

1. **Single runs for every ablation**; the 3-seed spread covers only the headline recipe. Many ablation differences are 0.1–0.4 PDMS.
2. **NAVSIM navtest and nuScenes only**: no navhard, HUGSIM, Bench2Drive or any reactive test, which the paper names as its limitation. Several wiki results suggest future-related mechanisms show their size on navhard, not navtest.
3. **Comparison scope**: ReDrive, SUV, DriveVA, ReWorld and WA-JEPA are absent; the v2 table mixes evaluator conventions.
4. **nuScenes numbers are re-evaluated** with a collision implementation that disagrees with at least one source paper.
5. **The headline is the primary run**, which is the top of its seed range.

**Method**

6. **One front camera**: anything outside its field of view is invisible to both the planner and the world model.
7. **The predicted futures are over-smoothed** (the paper's words), and the planner ignores their order; the "world model" contributes context more than forecasts.
8. **Wrong futures are worse than none** (−2.4 PDMS, TTC −2): a world-model error in deployment would hurt more than a missing prediction.
9. **Latency is FP32 forward time** excluding preprocessing, on one H20.

**Source conversion**

10. Six of eight figures are embedded; Figure 2 and supplementary Figure 3 are missing (Figure 2's data are Supplementary Table 10). All main and supplementary tables are reproduced; Supplementary Tables 4–7 are merged into one. The affiliation markers in the author line are only partly preserved.

---

## Key Cross-References

- [[concepts/evaluation-variance.md]] — the first training-seed measurement on NAVSIM.
- [[concepts/world-model-for-ad.md]] — a consumed latent future that works as context (Pattern 48); forecast quality against planning value.
- [[concepts/wam-attention-masks.md]] — action reads future, with a stop-gradient into the predictor.
- [[concepts/navsim-benchmark.md]] — placement on both ladders and the mixed v2 table.
- [[concepts/diffusion-planner.md]] — Trajectory-Adaptive Bias on a DiffusionDrive planner.
- [[concepts/foundation-backbones-for-ad.md]] — DINOv3 S/B/L under a fixed pipeline.
- [[concepts/inference-latency.md]] — 58 ms for a consumed latent world model.
- [[concepts/nuscenes-waymo-evals.md]] — a re-evaluated zero-shot table.
