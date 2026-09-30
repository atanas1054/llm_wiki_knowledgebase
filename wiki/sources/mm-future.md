---
title: "MM-Future: Multi-Mode Joint World–Action Modeling for Autonomous Driving"
type: source-summary
sources: ["raw/papers/MM-Future_ Multi-Mode Joint World–Action Modeling for Autonomous Driving.md"]
related: [concepts/world-model-for-ad.md, concepts/selection-based-planning.md, concepts/navsim-benchmark.md, concepts/hugsim-benchmark.md, concepts/wam-attention-masks.md, concepts/inference-latency.md, concepts/diffusion-planner.md, concepts/foundation-backbones-for-ad.md, concepts/counterfactual-prediction.md, concepts/evaluation-variance.md, sources/da-wam.md, sources/drivefuture.md, sources/latent-wam.md, sources/wa-jepa.md, sources/physwam.md, sources/drivesuprim.md, sources/metis.md, sources/suv.md, sources/simwam.md, sources/ad-e2e-jepa.md, sources/redrive.md, sources/momworld.md, sources/lwdrive.md, sources/hydra-mdp-pp.md, sources/drivereferee.md]
created: 2026-09-30
updated: 2026-09-30
confidence: medium
---

# MM-Future

**Paper**: MM-Future: Multi-Mode Joint World–Action Modeling for Autonomous Driving
**Authors**: Shuai Liu, Hechangle Gong, Hao Jiang, Runlin He, Junxiang Zhan, Kai Huang, Sheng Yang, Shaoqing Ren
**Orgs**: NIO; Artificial General Intelligence Institute, University of Science and Technology of China; Sun Yat-sen University; Beihang University
**arXiv**: 2609.20377v1
**Code**: not stated
**Source**: `raw/papers/MM-Future_ Multi-Mode Joint World–Action Modeling for Autonomous Driving.md`

---

## What It Is

A latent world–action model that generates **many trajectory–future pairs at once** and picks one with a scorer.

1. **MM-Tokens.** A LoRA-adapted DINOv2-S reads four cameras. Learned queries compress each two-frame chunk of register tokens into 64 tokens of width 256. There is no reconstruction loss of any kind.
2. **Joint flow.** $M$ hypotheses are started from paired noise: a trajectory prior drawn from a Gaussian mixture fitted to K-means trajectory clusters, and independent Gaussian noise for the future MM-Tokens. A 16-layer transformer denoises each pair together, with action and scene tokens attending to each other. Hypotheses never see each other.
3. **Best-of-Many training.** Only the hypothesis whose trajectory is closest to the expert receives the loss, on both its action and its future tokens.
4. **Future-conditioned scorer.** Each trajectory is scored from the history and from *its own* predicted future, with heads trained on the NAVSIM simulator's sub-scores.

**Headline numbers** (64 hypotheses, 233 ms on one H800):

| | Trained on navtrain | Trained on trainval |
|---|---:|---:|
| NAVSIM v1 PDMS | 93.4 | **94.0** |
| NAVSIM v2 EPDMS (corrected) | – | **91.5** |
| HUGSIM HD-Score, zero-shot, 436 scenarios | – | **32.3** |

---

## Key Takeaways

- **Most of the score is hypothesis count plus a simulator-trained scorer, with no world model involved.** Action-only generation goes from 84.1 PDMS with one mode to 92.3 with 32 modes and a history-only scorer (+8.2). Pairing each trajectory with a generated future adds +0.6. Letting the scorer read that future adds +0.4.
- **The headline 94.0 uses more training data.** It is trained on trainval. On navtrain alone it is 93.4, which is below DriveSuprim's 93.5 in the paper's own table and +0.3 over DrivoR, the register-token planner it builds on.
- **Pairing helps in one of four interaction designs.** Action-only scores 92.3. Three joint variants score 92.3–92.5. Only the bidirectional variant with modality-specific branches reaches 92.9.
- **It is a second per-candidate-future scorer, and the second small positive.** +0.4 PDMS here, +0.15 in [[sources/da-wam.md]]. Here the futures already exist, so reading them costs no latency.
- **Only one hypothesis per scene ever has its future supervised.** The futures paired with the other trajectories are never compared with anything real. The paper shows they *differ* with the action (left-most against right-most: 17% larger RMS distance), not that they are right.
- **On v2 it buys progress with rule compliance.** EP 92.2 is 4.8 above the human agent's. DDC, TL, LK and HC are each the lowest or second lowest in its table. The scorer is trained on the v1 components only.
- **HUGSIM 32.3 leads a table of four baselines** and sits below three entries the wiki already has on the same 436 scenarios: WA-JEPA 44.6, PhysWAM 35.5 and BeyondDrive 34.8.
- **It is cheap for what it rolls out.** About 3.2 ms per extra paired hypothesis; 64 futures of 4 s each in 233 ms.

---

## Method

![[mmfuture_framework.png|Overview of MM-Future. Visual transformers encode history images into historical MM-Tokens; an EMA copy encodes future images into target MM-Tokens for training only. The ground-truth trajectory and future tokens are noised, with the action noise drawn from a Gaussian-mixture prior. A transformer with a shared attention and separate AdaLN and feed-forward branches for an action stream and a scene stream outputs M modes, each a trajectory plus future tokens. A future-conditioned scorer predicts metric scores for proposal selection; Best-of-Many selection and supervision is used in training]]

*Figure 2: Overview. Online encoder for history, EMA encoder for future targets, $M$ paired hypotheses from a modality-aware transformer, Best-of-Many supervision, and a future-conditioned scorer.*

### Setup

Observation $\mathcal O=(\mathbf I^h,\mathbf s^h,c)$: history images from $V$ cameras, ego states, navigation command. Output: $M$ paired modes

$$\mathcal H_m=\big(\hat{\boldsymbol\tau}_m,\hat{\mathbf X}^+_m\big),\qquad \{\mathcal H_m\}_{m=1}^{M}\sim p_\Theta(\boldsymbol\tau,\mathbf X^+\mid\mathcal O)$$

with $\hat{\boldsymbol\tau}_m\in\mathbb R^{T_a\times3}$ and $\hat{\mathbf X}^+_m\in\mathbb R^{C_f\times N_x\times d}$ ($C_f$ future chunks of $N_x$ tokens).

### MM-Tokens

$$\mathbf P^v_t=\mathcal E(\mathbf I^v_t)\in\mathbb R^{L_v\times d},\qquad \mathbf P^{(j)}=\big[\mathbf P^1_t;\dots;\mathbf P^V_t\big]_{t\in\mathcal T_j},\qquad \mathbf X_j=\mathcal A\big(\mathbf Q^{(j)},\mathbf P^{(j)}\big)\in\mathbb R^{N_x\times d}$$

- $\mathcal E$ is a ViT whose **register tokens** are kept (the DrivoR design); $\mathcal A$ is a 4-layer attention module over learned chunk queries $\mathbf Q^{(j)}$.
- History tokens $\mathbf X^h=E_{\theta_e}(\mathbf I^h)$ come from the online encoder. Future targets come from an EMA copy: $\bar{\mathbf X}^+=\operatorname{sg}\big(E_{\bar\theta_e}(\mathbf I^+)\big)$, $\bar\theta_e\leftarrow\mu\bar\theta_e+(1-\mu)\theta_e$, $\mu=0.999$.
- Dense patches are discarded after the queries read them. No RGB or BEV reconstruction.
- Size: a two-frame, four-camera chunk at 336×560 is 7,680 patches; it becomes 64 tokens.

### Joint conditional flow

- **Action tokens**: $\mathbf a^{gt}=E_a(\boldsymbol\tau^{gt})\in\mathbb R^{T_a\times4}$, normalized $(\Delta x,\Delta y,\sin\psi,\cos\psi)$, with a deterministic decoder $D_a$.
- **Sources**: $\boldsymbol\epsilon^a_m$ from a Gaussian-mixture noise (GMN) distribution whose components are the mean and variance of K-means trajectory clusters (from MeanFuser); $\boldsymbol\epsilon^x_m\sim\mathcal N(0,I)$ independently.
- **Path**, with independent flow times for the two streams $\kappa\in\{a,x\}$:

$$\mathbf z^\kappa=(1-\rho^\kappa)\boldsymbol\epsilon^\kappa+\rho^\kappa\mathbf y^\kappa,\qquad (\hat{\mathbf y}^a,\hat{\mathbf y}^x)=G_{\theta_g}\big(\mathbf z^a,\mathbf z^x;\mathbf X^h,\mathbf s^h,c,\rho^a,\rho^x\big)$$

- **Generator**: shared multimodal attention, modality-specific AdaLN and feed-forward branches. History tokens are a clean prefix (they attend only to history); action and future tokens attend to history and to each other. The mode dimension is folded into the batch, so modes share weights and never exchange tokens.
- **$x$-prediction** with a velocity loss:

$$\hat{\mathbf u}^\kappa=\frac{\hat{\mathbf y}^\kappa-\mathbf z^\kappa}{1-\rho^\kappa},\qquad \mathbf u^\kappa=\frac{\mathbf y^\kappa-\mathbf z^\kappa}{1-\rho^\kappa}$$

### Best-of-Many supervision

$$d_m=\big\|D_a(\hat{\mathbf y}^a_m)-\boldsymbol\tau^{gt}\big\|_1,\qquad m_{\mathrm{win}}=\arg\min_m d_m,\qquad \mathcal L_{\mathrm{BoM}}=\sum_{\kappa\in\{a,x\}}\lambda_\kappa\big\|\hat{\mathbf u}^\kappa_{m_{\mathrm{win}}}-\mathbf u^\kappa_{m_{\mathrm{win}}}\big\|_2^2$$

The trajectory picks the winner and the same index supervises both streams. Non-winning hypotheses receive no generation loss for that example.

### Future-conditioned scorer

$$\mathbf q^h_{1:M}=\mathcal D_h\big(g_\tau(\operatorname{sg}[\hat{\boldsymbol\tau}_{1:M}]),\mathbf X^h\big),\qquad \mathbf q^+_m=\mathcal D_f\big(\mathbf q^h_m,\operatorname{sg}[\hat{\mathbf X}^+_m]\big)+g_s(\mathbf s_0),\qquad \ell_m=\mathbf H(\mathbf q^+_m)$$

- $\mathcal D_h$: proposals compare with each other by self-attention and read the history by cross-attention.
- $\mathcal D_f$: block-diagonal over $m$. The score of mode $m$ can read $\hat{\mathbf X}^+_m$ and no other future.
- $\mathbf H$ predicts logits for the PDMS components. Targets $y_m$ come from "the official training-time pseudo-simulator" run on the sampled trajectories.
- Gradients are stopped on the trajectories and future tokens, so the generator is not trained to make scoring easier.

$$\mathcal L_{\mathrm{score}}=\frac1M\sum_m\operatorname{BCEWithLogits}(\ell_m,y_m),\qquad \mathcal L=\mathcal L_{\mathrm{BoM}}+\lambda_s\mathcal L_{\mathrm{score}}$$

### Implementation

| Item | Value |
|---|---|
| Input | 2 s history, 4 s prediction, 2 Hz; front, front-left, front-right, back cameras at 336×560 |
| Encoder | DINOv2-S, LoRA rank 32; 4-layer attention module trained from scratch |
| MM-Tokens | 64 tokens × 256-d per two-frame chunk (2 history chunks, 4 future chunks) |
| Generator | 16 layers, width 1024 |
| Sampling | 64 pairs (main results); 2 Euler steps |
| Training | AdamW, 25 epochs, batch 64, LR $2\times10^{-4}$, weight decay 0.01 |
| Loss weights | action / scene / scoring = 1.0 / 0.1 / 1.0 |
| Latency | End-to-end forward, one H800, batch 1, bf16 |

Diversity statistics, a representation-capacity study and an efficiency analysis are in a supplement that is not part of the clipping.

---

## Figures

![[teaser 3.png|Three WAM paradigms. (a) Cascaded: action proposals feed an action-to-generation model that outputs multiple scene rollouts. (b) Joint: a single paired noise gives a single paired rollout. (c) MM-Future: multiple paired noises give multiple paired rollouts]]

*Figure 1: Cascaded WAMs have many modes and one-way influence; joint WAMs have two-way influence and one mode; MM-Future generates multiple paired rollouts jointly.*

Figure 2 is in the Method section.

![[futurex_xtoken_attention_panorama.png|MM-Token attention overlaid on the three front cameras at frames −2, 0, +2 and +4 for three scenes: a right turn, a left turn, and a low-light queue]]

*Figure 3: Aggregate MM-Token-to-patch attention at −1.0, 0.0, +1.0 and +2.0 s. Attention stays on the intersection foreground and road users (A), moves toward the turning corridor (B), and stays on the lead vehicle at night (C). Future frames are encoded for the figure only.*

![[futurex_multimode_convergence.png|Validation PDM score against optimization steps for single-mode and multi-mode training. M = 16 and M = 32 reach 0.80 at 3.8k steps; M = 1 reaches it at 17.5k steps]]

*Figure 4: Training convergence on NAVSIM v1 navval. Final values read from the plot: about 0.825 ($M=1$), 0.905 ($M=16$), 0.92 ($M=32$).*

---

## Tables

### Table 1: NAVSIM v2 navtest ("official corrected EPDMS")

| Method | NC | DAC | DDC | TL | EP | TTC | LK | HC | EC | EPDMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *E2E-based* | | | | | | | | | | |
| DiffusionDriveV2 | 97.7 | 96.6 | 99.2 | 99.8 | 88.9 | 97.2 | 96.0 | 97.8 | 91.0 | 85.5 |
| MeanFuser | 98.3 | 97.2 | 99.6 | 99.8 | 87.6 | 97.4 | 97.3 | 98.3 | 88.2 | 89.5 |
| UniTeD | 99.1 | 97.2 | 99.6 | 99.9 | 86.9 | 98.5 | 98.4 | 99.9 | 87.3 | 90.1 |
| *VLA-based* | | | | | | | | | | |
| DriveWorld-VLA | 98.6 | 99.1 | 99.6 | 99.8 | 87.4 | 97.9 | 97.0 | 97.8 | 78.6 | 86.8 |
| IRR-Drive-4B | 97.0 | 98.3 | 98.9 | 99.5 | 92.3 | 96.8 | 95.8 | 97.6 | 82.2 | 89.0 |
| DriveFine | 98.7 | 97.3 | 99.5 | 99.8 | 88.7 | 97.8 | 97.7 | 98.4 | 83.8 | 89.7 |
| *WAM-based* | | | | | | | | | | |
| Latent-WAM | 98.1 | 97.3 | 99.6 | 99.8 | 87.7 | 97.3 | 97.6 | 98.1 | 87.3 | 89.3 |
| GraphWorld | 98.4 | 98.8 | 99.1 | 99.1 | 85.9 | 97.9 | 96.0 | 97.8 | 74.6 | 89.5 |
| DriveFuture | 98.8 | 99.1 | 99.6 | 99.9 | 86.6 | 98.4 | 96.4 | 98.3 | 74.8 | 89.9 |
| **MM-Future** | 99.0 | 98.8 | 98.8 | 99.4 | 92.2 | **98.6** | 95.4 | 96.3 | 89.2 | **91.5** |

### Table 2: NAVSIM v1 navtest

| Method | NC | DAC | TTC | Comf. | EP | PDMS |
|---|---:|---:|---:|---:|---:|---:|
| *E2E-based* | | | | | | |
| DiffusionDrive | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| MeanFuser | 98.6 | 97.0 | 95.0 | 100 | 82.8 | 89.0 |
| GaussianFusion | 98.7 | 98.1 | 95.7 | - | 88.2 | 92.0 |
| DriveSuprim | 98.6 | 98.6 | 95.5 | 100 | 91.3 | 93.5 |
| DrivoR (train) | 98.9 | 98.3 | 96.2 | 100 | 89.1 | 93.1 |
| DrivoR (trainval) | 99.0 | 98.9 | 96.7 | 100 | 90.0 | 93.7 |
| *VLA-based* | | | | | | |
| AutoVLA | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 |
| DriveVLA-W0 | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.2 |
| ReCogDrive | 97.9 | 97.3 | 94.9 | 100.0 | 87.3 | 90.8 |
| DriveWorld-VLA | 99.1 | 98.2 | 96.1 | 100 | 85.9 | 91.3 |
| IRR-Drive-4B | 98.0 | 98.3 | 93.7 | 100 | 88.5 | 91.3 |
| DriveFine | 98.8 | 99.2 | 96.2 | 100 | 86.9 | 91.8 |
| *WAM-based* | | | | | | |
| Epona | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 |
| WoTE | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| GraphWorld | 99.0 | 97.1 | 95.5 | 100 | 83.2 | 90.1 |
| DriveFuture | 98.8 | 99.1 | 95.4 | 100 | 84.2 | 90.7 |
| **MM-Future (train)** | 99.0 | 98.6 | 96.2 | 100 | 90.2 | 93.4 |
| **MM-Future (trainval)** | 98.7 | 99.0 | 95.8 | 100 | **91.6** | **94.0** |

Subscripts on DrivoR and MM-Future say whether the *train* or the *trainval* set was used for training.

### Table 3: Ablation on NAVSIM v1 navtest

| Variant | Future tokens | # Hyp. | Coupling | Scorer | NC | DAC | TTC | EP | PDMS | Latency (ms) |
|---|:-:|---:|---|---|---:|---:|---:|---:|---:|---:|
| *A. Multi-mode generation* | | | | | | | | | | |
| Action, single mode | No | 1 | Action only | – | 97.4 | 93.2 | 92.5 | 79.3 | 84.1 | 52 |
| Action, multiple modes | No | 16 | Action only | Hist. | 98.3 | 97.7 | 93.9 | 88.6 | 91.1 | 51 |
| Action, multiple modes | No | 32 | Action only | Hist. | 98.3 | 98.2 | 94.3 | 90.4 | 92.3 | 65 |
| Paired, single mode | Yes | 1 | Action ↔ Scene | – | 97.8 | 93.8 | 93.0 | 80.4 | 85.1 | 81 |
| Paired, multiple modes | Yes | 16 | Action ↔ Scene | Hist. | 98.7 | 97.9 | 94.9 | 88.7 | 91.7 | 79 |
| Paired, multiple modes | Yes | 32 | Action ↔ Scene | Hist. | 98.8 | 98.3 | 95.5 | 90.2 | 92.9 | 132 |
| *B. Modality-aware scene–action interaction* | | | | | | | | | | |
| Single DiT, one-way | Yes | 32 | Action ← Scene | Hist. | 98.6 | 97.9 | 94.9 | 90.3 | 92.4 | 124 |
| Single DiT, bidirectional | Yes | 32 | Action ↔ Scene | Hist. | 98.5 | 98.1 | 95.0 | 89.8 | 92.3 | 123 |
| Mixture-of-DiT, one-way | Yes | 32 | Action ← Scene | Hist. | 98.5 | 98.3 | 94.7 | 90.1 | 92.5 | 132 |
| Mixture-of-DiT, bidirectional | Yes | 32 | Action ↔ Scene | Hist. | 98.8 | 98.3 | 95.5 | 90.2 | 92.9 | 132 |
| *C. Future-conditioned proposal scoring* | | | | | | | | | | |
| History only | Yes | 32 | Action ↔ Scene | Hist. | 98.8 | 98.3 | 95.5 | 90.2 | 92.9 | 132 |
| History + paired future | Yes | 32 | Action ↔ Scene | Hist.+paired | 98.9 | 98.5 | 96.0 | 90.3 | 93.3 | 131 |

### Table 4: HUGSIM closed loop, 436 scenarios, zero-shot from the NAVSIM v1 model

E / M / H / X = Easy / Medium / Hard / Extreme.

| Method | RC E | RC M | RC H | RC X | RC Avg. | HDS E | HDS M | HDS H | HDS X | HDS Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VAD | 38.7 | 27.0 | 25.5 | 23.0 | 27.9 | 24.3 | 9.9 | 10.4 | 8.2 | 12.3 |
| Latent-TransFuser | 68.4 | 40.7 | 36.9 | 25.5 | 41.4 | 52.8 | 24.6 | 19.8 | 8.1 | 24.8 |
| UniAD | 58.6 | 41.2 | 40.4 | 26.0 | 40.6 | 48.7 | 29.5 | 27.3 | 14.3 | 28.6 |
| Latent-WAM | 84.2 | 42.5 | 30.6 | 35.5 | 45.9 | 72.5 | 24.0 | 12.2 | 18.1 | 28.9 |
| **MM-Future** | 64.8 | **51.5** | 39.5 | 22.8 | 44.5 | 53.8 | **40.0** | 27.3 | 8.6 | **32.3** |

### Results given only in the text

| Quantity | Value |
|---|---|
| Main model latency (64 proposals) | 233 ms |
| Steps to reach 0.80 validation PDM score | 3.8k ($M=16$, $32$) against 17.5k ($M=1$) |
| Futures paired with left-most against right-most actions, relative to those paired with similar actions | RMS distance +17.3%, cosine distance +38.9% |

---

## Reading the Results

### 1. Where 93.4 comes from {#decomposition}

Chained from Table 3 and Table 2 (navtrain, v1 PDMS):

| Step | PDMS | Δ | Latency |
|---|---:|---:|---:|
| One trajectory, action-only flow | 84.1 | – | 52 ms |
| 32 trajectories + a history-only scorer trained on simulator sub-scores | 92.3 | **+8.2** | 65 ms |
| + generate a paired future for each (bidirectional, modality-specific branches) | 92.9 | +0.6 | 132 ms |
| + let the scorer read each trajectory's own future | 93.3 | +0.4 | 131 ms |
| 64 hypotheses (the Table 2 model) | 93.4 | +0.1 | 233 ms |
| Train on trainval instead of navtrain | 94.0 | +0.6 | 233 ms |

- **World modeling is worth +1.0 of the 9.3 points between one trajectory and the navtrain result**, and it doubles the latency at 32 hypotheses.
- **"Multi-mode" is two things the ablation does not separate.** Going from 1 to 16 or 32 modes changes the training objective (Best-of-Many over sampled sources) and adds a scorer that selects. There is no row with many modes and no scorer, and no oracle-over-$M$ row, so the split between generating better candidates and choosing among them is unknown.
- **The single-mode baselines are weak** (84.1 and 85.1), below DiffusionDrive's 88.1. Two Euler steps from one draw of a mixture prior is not a tuned single-trajectory planner, so +8.2 overstates what selection adds to a good one.
- **The increments of interest are 0.4–0.6 from single runs.** [[sources/drivereferee.md]] measured a 95% half-width of about 0.2 for the difference between two checkpoints on navtest from scene sampling alone.
- **The external check agrees with the internal one.** DrivoR, the register-token planner this encoder design comes from, is 93.1 on navtrain and 93.7 on trainval. MM-Future is +0.3 on both.

### 2. The interaction ablation {#interaction}

| Generator | Action reads scene only | Bidirectional |
|---|---:|---:|
| No future tokens (action only) | 92.3 | |
| Single DiT | 92.4 | 92.3 |
| Mixture-of-DiT (modality-specific AdaLN and FFN) | 92.5 | **92.9** |

- Three of the four joint variants are within 0.2 of generating no future at all.
- The bidirectional edge is worth +0.4 with modality-specific branches and −0.1 without.
- In the wiki's mask vocabulary ([[concepts/wam-attention-masks.md]]) "one-way" is *action reads future* and "bidirectional" adds *future reads action*. On the Wan2.2 family, bidirectional was flat on navtest ([[sources/simwam.md]]) or worst ([[sources/metis.md]]). Here it is the best cell, by a margin those papers would call noise. The models differ in almost everything else (latent tokens against video, a scorer on top, trained from scratch).
- No variant is evaluated on v2, navhard or HUGSIM, which is where the mask effects recorded in the wiki appear.

### 3. Per-candidate futures {#per-candidate}

| | [[sources/da-wam.md]] | MM-Future |
|---|---|---|
| Candidates | 32 generated proposals | 32 (ablation) / 64 (main) |
| How each future is made | A predictor queried by each finished trajectory | Co-denoised with its trajectory from paired noise |
| Horizon | 0.5 s | 4 s |
| Future target | EMA V-JEPA latents | EMA MM-Tokens (learned end to end) |
| Scorer reads | History, action, its own future | History, action, its own future (block-diagonal) |
| Effect of reading the per-candidate future | +0.15 PDMS | **+0.4 PDMS** |
| Shared-future control | −0.50 | Not run |
| Which futures are supervised | The expert-matched one | The Best-of-Many winner |

- **Same sign, same order of magnitude, in two independent designs.** That is the supportable statement. Neither effect is outside single-run noise by itself.
- **The largest sub-score change is TTC (+0.5)**, which is what a candidate-specific future should help with.
- **The unsupervised-futures question carries over.** In each training scene only the winning pair's future is compared with reality. The scorer then reads 63 other futures that were never trained to be consequences of their trajectories. The paper's evidence is that futures vary with the paired action (left-most against right-most: RMS +17.3%, cosine +38.9%). That is a sensitivity check. It does not show the left-turn future is what a left turn would cause ([[concepts/counterfactual-prediction.md]]).
- **There is no prediction-quality number.** No error between predicted and EMA-target tokens, even for the winning mode.

### 4. The v2 profile: progress above the human, rule terms at the bottom of the table {#v2-profile}

| Sub-score | MM-Future | Human agent (corrected) | Rank among the 10 rows of Table 1 |
|---|---:|---:|---|
| EP | 92.2 | 87.4 | 2nd (IRR-Drive 92.3) |
| TTC | 98.6 | 100 | 1st |
| NC | 99.0 | 100 | 2nd |
| DDC | 98.8 | 99.8 | Lowest |
| TL | 99.4 | 100 | 2nd lowest |
| LK | 95.4 | 100 | Lowest |
| HC | 96.3 | 98.1 | Lowest |
| EC | 89.2 | 90.1 | 2nd |

- The scorer's heads are the **v1** components (NC, DAC, TTC, comfort, EP). Driving direction, traffic lights and lane keeping are not among its targets, and those are the terms where it is weakest.
- EP 92.2 is the highest v2 ego progress of any ingested method ([[sources/lwdrive.md]] held that at 90.3).
- The residual check agrees with the caption: closed form 91.0 against 91.5 reported (−0.5, corrected-like).
- The v2 number comes from the trainval model. No navtrain-only EPDMS is given.

### 5. HUGSIM {#hugsim}

| HD-Score, 436 scenarios | Easy | Medium | Hard | Extreme | Overall | Source |
|---|---:|---:|---:|---:|---:|---|
| WA-JEPA (pinned commit) | 79.8 | 55.6 | 30.6 | 13.6 | **44.6** | [[sources/wa-jepa.md]] |
| PhysWAM | 86.9 | 30.1 | 25.2 | 13.4 | 35.5 | [[sources/physwam.md]] |
| BeyondDrive (as reported) | 65.6 | 31.4 | 26.3 | 16.2 | 34.8 | via PhysWAM |
| DrivoR (rescored by WA-JEPA) | 78.0 | 29.1 | 20.0 | 14.1 | 32.5 | via WA-JEPA |
| **MM-Future** | 53.8 | 40.0 | 27.3 | 8.6 | 32.3 | This paper |
| Latent-WAM | 72.5 | 24.0 | 12.2 | 18.1 | 28.9 | This paper's table |

- **"The highest average HD-Score" holds among four baselines.** Three published results on the same scenario count are higher and none is cited.
- **The profile is unusual.** Its Easy score is the lowest of the recent methods and its Extreme score is the lowest on this list, while Medium is second only to WA-JEPA. Average route completion (44.5) is below Latent-WAM's (45.9).
- **The averages are episode-weighted with the 436-scenario counts** (80 / 157 / 96 / 103): that reproduces 32.3 and 44.5. It also reproduces Latent-WAM's 28.9 and 45.9 exactly, which bears on where the wiki had filed Latent-WAM; see [[concepts/hugsim-benchmark.md#mm-future]].
- UniAD's average is printed as 28.6. The weighted mean of its printed tiers is 28.9, which is what PhysWAM's table has.
- No HUGSIM commit, no sub-metrics (NC, DAC, TTC, comfort), no per-dataset breakdown.

### 6. Table notes

**v1.**
- Every row passes the closed-form check (+0.7 to +3.1), including both MM-Future rows (+1.3, +1.4). DriveSuprim is at its correct ViT-L 93.5 with matching sub-scores.
- The paper compares its navtrain model with DrivoR (train) and its trainval model with DrivoR (trainval). It does not remark that DriveSuprim (93.5, navtrain) is above its navtrain model.
- Not in the table: [[sources/da-wam.md]] and CLEAR (93.7), Drive-JEPA (93.3), WCog-VLA (92.9). "Best aggregate planning score" rests on the trainval row.
- DriveVLA-W0 is at its anchor-based 90.2.

**v2.**
- The caption says corrected EPDMS. **DiffusionDriveV2 is printed at 85.5, its original-evaluator value** (residual +2.1; its corrected value is 87.5). DriveWorld-VLA 86.8 has the pre-fix signature too (+2.6), a standing exception on [[concepts/navsim-benchmark.md]].
- **GraphWorld's residual is −2.6**, about twice the largest negative residual recorded so far (−1.3). Its TL and DDC are 99.1. The check cannot say more than that the row is unusual.
- Not in the table: [[sources/wa-jepa.md]] (91.7), [[sources/suv.md]] (91.0), [[sources/redrive.md]] (90.8), PhysWAM (90.3), MomWorld (90.1). MM-Future's 91.5 is second to WA-JEPA in the wiki's corrected cohort.
- New to the wiki: UniTeD (90.1), MeanFuser (89.5 / 89.0), IRR-Drive-4B (89.0 / 91.3), GraphWorld's NAVSIM rows (89.5 / 90.1), GaussianFusion (92.0 PDMS; its author list includes S. Liu and K. Huang, matching two names on this paper).

### 7. Cost {#cost}

| Hypotheses | Action only | Paired |
|---:|---:|---:|
| 1 | 52 ms | 81 ms |
| 16 | 51 ms | 79 ms |
| 32 | 65 ms | 132 ms |
| 64 | – | 233 ms |

- From 16 to 64 hypotheses the paired model adds about **3.2 ms per hypothesis**, each being 256 future tokens and 8 action tokens over two Euler steps.
- The single-mode rows are not faster than the 16-mode rows. The paper does not say whether they use the same number of sampling steps.
- The scorer adds nothing measurable (131 against 132 ms).
- Parameter count is not given.

---

## Relationships

- **[[sources/da-wam.md]]**: the other scorer that reads one future per candidate. See [the comparison](#per-candidate). DA-WAM's taxonomy has a cell "(d) one latent per candidate" that it occupied alone; MM-Future is a second occupant that generates trajectory and future jointly. DA-WAM is not cited.
- **DrivoR** (not ingested): the source of the register-token encoder and the closest baseline, +0.3 PDMS behind on both training sets. It is a scorer-based planner and the second-highest navhard entry in the wiki (54.6).
- **[[sources/wa-jepa.md]]**: the nearest generative recipe. Both flow-match future latents whose targets come from an EMA encoder, jointly with actions. WA-JEPA generates one pair from a V-JEPA 2 encoder with no scorer; MM-Future generates 64 from 64-token chunks and scores them. WA-JEPA is ahead on v2 (91.7) and on HUGSIM (44.6) and is not cited.
- **[[sources/drivefuture.md]]**: named the strongest WAM baseline (90.7 / 89.9). It also gives each candidate its own future, inside a diffusion planner, and submits through a GTRS-Dense scorer.
- **[[sources/latent-wam.md]]**: the compact-token predecessor (16 scene queries per view) and the HUGSIM comparison row.
- **[[sources/ad-e2e-jepa.md]]**: per-candidate rollouts as search, with the cost measured. About 2.2 ms per candidate for a 32-token state there; about 3.2 ms per paired hypothesis here.
- **[[sources/metis.md]] / [[sources/suv.md]] / [[sources/simwam.md]]**: the mask family. MM-Future is a bidirectional design with a per-candidate scorer on top.
- **[[sources/momworld.md]]**: the opposite choice, one action-free future shared by 16,384 candidates.
- **[[sources/hydra-mdp-pp.md]]**: the origin of scorer heads trained on simulator sub-scores.
- **Not ingested**: MeanFuser (the Gaussian-mixture noise prior), UniTeD, IRR-Drive, GraphWorld, MAP-World, IDOL, SeerDrive, DAWN, Discrete-WAM, ForgeDrive, SparseWorld.

---

## Limitations

**Attribution**

1. **The named mechanism is about a tenth of the gain.** +1.0 PDMS for paired generation and paired-future scoring, against +8.2 for many modes with a history-only scorer.
2. **Generation and selection are not separated.** No multi-mode row without a scorer, no oracle over the hypotheses.
3. **Single runs**, with the deltas of interest at 0.4–0.6.
4. **Ablations are v1 navtest only.** Nothing on v2, navhard or HUGSIM, and the ablated model has 32 hypotheses while the reported one has 64.

**The world model**

5. **Non-winning futures are never supervised**, and the scorer reads them.
6. **No prediction-quality metric** and no collapse analysis for tokens whose targets come from an EMA copy of the same encoder. The action and scoring losses are what keep the encoder informative; the scene loss weight is 0.1.
7. **The tokens are not interpretable**, which is the paper's own stated limitation. Attention maps are the only evidence of what they hold.

**Comparison**

8. **The headline uses trainval.** On navtrain, 93.4.
9. **The scorer is trained on the benchmark's simulator**, like every entry above 92 PDMS in this wiki ([[concepts/selection-based-planning.md#top-of-leaderboard]]).
10. **HUGSIM and v2 tables omit the stronger published results** (WA-JEPA on both; PhysWAM and BeyondDrive on HUGSIM).
11. **The v2 table is captioned corrected and carries a pre-fix row.**

**Scope**

12. **No navhard**, where scorer-based methods are separated most clearly and where DrivoR is strong.
13. **No parameter count, no code.**
14. **Rule-compliance sub-scores are the lowest in its own v2 table.**

**Source conversion**

15. All four figures and four tables are present. The supplement is not. The front matter's author field is empty; the body gives names and a footnote-style affiliation list without the author-to-affiliation mapping.

---

## Key Cross-References

- [[concepts/world-model-for-ad.md]] — Pattern 43: many paired scene–action hypotheses generated jointly, each scored against its own future.
- [[concepts/selection-based-planning.md]] — a +8.2 / +0.6 / +0.4 decomposition; a new entry above 92 PDMS that trains a scorer on the simulator.
- [[concepts/navsim-benchmark.md]] — 93.4 (navtrain) / 94.0 (trainval) PDMS; 91.5 corrected EPDMS.
- [[concepts/hugsim-benchmark.md]] — 32.3 on 436 scenarios; evidence that Latent-WAM's row is a 436-scenario result.
- [[concepts/wam-attention-masks.md]] — a 2×2 of one-way against bidirectional, single against modality-specific branches.
- [[concepts/inference-latency.md]] — 233 ms for 64 four-second futures; 3.2 ms per hypothesis.
- [[concepts/diffusion-planner.md]] — a Gaussian-mixture flow source with Best-of-Many supervision and independent flow times per modality.
- [[concepts/foundation-backbones-for-ad.md]] — DINOv2-S with LoRA and register tokens as a learned scene tokenizer.
- [[concepts/counterfactual-prediction.md]] — action sensitivity of unsupervised futures is not accuracy.
