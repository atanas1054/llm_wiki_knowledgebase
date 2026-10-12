---
title: "HyWorldVLA: A Vision-Language-Action Model with Hybrid World Modeling for Autonomous Driving"
type: source-summary
sources: ["raw/papers/HyWorldVLA_ A Vision-Language-Action Model with Hybrid World Modeling for Autonomous Driving.md"]
related: [concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/selection-based-planning.md, concepts/evaluation-variance.md, concepts/foundation-backbones-for-ad.md, concepts/visual-tokenization.md, concepts/action-tokenization.md, concepts/inference-latency.md, concepts/teacher-pseudo-labels.md, concepts/navhard-ood-evaluation.md, sources/drivevla-w0.md, sources/drive-hwm.md, sources/coworld-vla.md, sources/onevl.md, sources/latent-wam.md, sources/dreameraD.md, sources/drivelaw.md, sources/explorevla.md, sources/epona.md, sources/resworld.md, sources/foredrive.md, sources/wa-jepa.md, sources/redrive.md, sources/world4scorer.md, sources/momworld.md, sources/hydra-mdp-pp.md, sources/recogdrive.md, sources/futuresightdrive.md, sources/policy-world-model.md, sources/autovla.md, sources/suv.md, sources/geowam.md, research-directions.md, sources/idol.md]
created: 2026-10-12
updated: 2026-10-12
confidence: medium
---

# HyWorldVLA: A Vision-Language-Action Model with Hybrid World Modeling for Autonomous Driving

**Paper**: HyWorldVLA: A Vision-Language-Action Model with Hybrid World Modeling for Autonomous Driving
**Authors**: Quanfu Yu, Xian Wu\*, Hao Xu\*, Liulong Ma† (\* equal contribution, † corresponding)
**Orgs**: Automotive New Technology Research Institute, BYD Company Limited
**arXiv**: 2607.20988v1 (23 July 2026), CC BY-NC-SA 4.0
**Code**: none announced. No latency and no parameter count are reported; the backbone is Emu3 (8B).
**Source**: `raw/papers/HyWorldVLA_ A Vision-Language-Action Model with Hybrid World Modeling for Autonomous Driving.md`. The clipping has all text and all eight tables but keeps only the first tile of each multi-panel figure, and it drops the (b) sub-captions of Figures 3, 4 and 11. The full figures were rebuilt from the 112 tiles of the arXiv HTML (v1) into eleven composites (`raw/assets/hyworldvla_fig3.jpg` to `hyworldvla_fig13.jpg`); Figures 1 and 2 are the clipping's own files.

**Confidence: medium.** The method and the navtest ablations are internally consistent, but every named effect except two is under 0.7 PDMS on single runs. The robustness result, the paper's distinctive claim, rests on a 655-scene subset that is not released and whose selection is not described. The baseline tables carry shifted rows (§[Table problems](#table-problems)).

---

## What It Is

**DriveVLA-W0's autoregressive Emu3 VLA with a second, latent world-model target added.** The base is [[sources/drivevla-w0.md]]'s: Emu3 backbone, VQ image tokens, FAST action tokens, interleaved language–vision–action chunks, a π0-style action expert joined by shared attention, and a Hydra-MDP trajectory-vocabulary head. The predicted frames in Figure 4 are 256×144, DriveVLA-W0's input resolution. HyWorldVLA adds a video-VAE latent that the VLM must also predict, and keeps only that latent in the second stage.

| Stage | Data | What the VLM predicts | Loss |
|---|---|---|---|
| 0. Video VAE | nuPlan; 8 frames at 216×216; captions from Qwen3.6-plus | – (encoder gives the target $z_2$) | Reconstruction + GAN (0.5, after 50K steps) + KL (1e-6), 100K steps |
| 1. Pre-training | OpenScene, "over 120 hours", 20 s clips | Language, VQ image tokens and FAST action tokens for 6 one-second chunks (1 s history, 4 s future), **plus** $\hat z_2$ from a learnable query $Q$ through an MLP | CE + CE + 0.5·CE (vision) + 0.1·MSE (latent); 4,000 steps |
| 2. Co-fine-tuning | NAVSIM trainval (100K+ frames) | **Only $\hat z_2$** (no image tokens) | Action loss + 0.1·MSE (latent); 4,000 steps, batch 96 |

The action expert reads the VLM through joint attention and takes $[\hat z_2$, historical actions, driving commands$]$ as its own sequence. Two heads are reported: a **selection head** (KL between predicted scores over an offline trajectory vocabulary and "ground-truth" scores built from the expert trajectory and the evaluation metrics) and a **flow-matching head**.

**Results**

| | HyWorldVLA | DriveVLA-W0 (same stack, pixel target only) |
|---|---:|---:|
| NAVSIM-v1 PDMS, selection head (headline) | **90.59** | 90.2 (published); 89.91 (HyWorldVLA's own "Pure WAM" ablation) |
| NAVSIM-v1 PDMS, flow-matching head | 88.76 | 87.2 (published row, copied) |
| NAVSIM-v2 EPDMS | 89.71 (corrected-like by residual, −0.91; evaluator unnamed) | 86.1 (shared-block row) |
| 655 rain/fog scenes from OpenScene test, PDMS | **86.87** | 61.18 (rerun) |
| Latency / parameters / code | not reported | 74 ms query head (its own paper) |

---

## Key Takeaways

- **One of three systems in the wiki on DriveVLA-W0's Emu3 stack**, with DriveVLA-W0 itself and [[sources/drive-hwm.md]] (posted later, in September 2026; neither cites the other). Against its own pixel-only ablation the hybrid is worth **+0.68 PDMS** (89.91 → 90.59); against DriveVLA-W0's published anchor-head row, +0.39; with flow matching, +1.56 over DriveVLA-W0's published FM row (87.2 → 88.76). All single runs.
- **The pixel target is used only in pre-training; the latent target is kept through fine-tuning and read at inference.** That is what separates it from the wiki's other mixed-target designs ([[sources/coworld-vla.md]], [[sources/onevl.md]], [[sources/drive-hwm.md]]), all of which the paper either does not cite or files under one of its two "pure" categories.
- **The named components are each worth 0.24–0.42 PDMS on navtest**: VAE text guidance +0.24, latent conditioning of the action expert +0.30, latent supervision during fine-tuning +0.42. The one large navtest effect is pixel tokens over latent-only, **+3.09** (87.50 → 90.59). There is **no no-world-model row**, so whether the latent-only model (87.50) is better or worse than no future prediction at all cannot be read.
- **The "latent" is a video-VAE code, not a semantic feature.** $z_2$ is a 4-channel VideoVAEPlus latent with 8× spatial and 4× temporal compression (about 4×2×27×27 for an 8-frame 216×216 clip, if the stated factors apply), regressed with MSE from one query token's hidden state. It is trained to reconstruct pixels, rain and fog included. The paper's account of robustness ("latent prediction in the VAE space inherently improves noise resilience") is never tested against a semantic latent target.
- **The robustness effect is large and comes from the fine-tuning loss.** On 655 rain/fog scenes: HyWorldVLA 86.87, without latent supervision in fine-tuning **73.18**, pure pixel model 69.95, DriveVLA-W0 61.18, DriveLaW 67.49, WoTE 60.65. Raising the latent weight $\lambda_3$ from 0.05 to 1.0 moves this subset **84.52 → 92.94** while navtest moves 90.47 → 89.75. The same mechanism is worth +0.42 on navtest and +13.7 on the subset: the navtest-small / hard-split-large shape that [[concepts/navhard-ood-evaluation.md]] records for navhard.
- **But the subset is weak evidence as built.** It is not released, how the 655 cases were chosen is not stated, and the baselines' run protocol (WoTE's LiDAR, DriveLaW's checkpoint) is not given. At $\lambda_3 = 1.0$ the model scores **higher on the "noisy" subset (92.94) than any model in the paper scores on navtest**, so the subset is not uniformly hard. Two pixel-only Emu3 models 0.29 apart on navtest (DriveVLA-W0 90.2, Pure WAM 89.91) are **8.8 apart on the subset** (61.18 vs 69.95), so checkpoint-to-checkpoint spread there is large.
- **The illumination test uses AI-edited images, and one comparison panel is duplicated.** The "dark" inputs of Figures 3(a) and 11 carry a visible *豆包AI生成* ("generated by Doubao AI") watermark and are 2732×1534 against the 1920×1080 originals: they are generative edits, not controlled photometric changes. In Figure 3(a), the main-text robustness figure, **HyWorldVLA's "Light" and "Dark" panels are the same image file** in the arXiv HTML, so the consistency it shows for HyWorldVLA is not demonstrated by two outputs. Figure 11's panels are distinct files.
- **The v2 table shifts two rows.** The row labelled Latent-WAM (87.7, EC 72.4) is [[sources/dreameraD.md]]'s published row digit for digit, and the row labelled DreamerAD is DriveWorld-VLA's (86.8). Latent-WAM's real score is **89.3** (EC 87.3), so the margin over it is +0.4, not +2.0. The table also mixes evaluators (own row corrected-like, shared block pre-fix-like); it is the **tenth** user of the shared baseline block recorded in [[concepts/navsim-benchmark.md]].
- **Mid-ladder on both benchmarks.** 90.59 PDMS is below about 30 single-pass v1 entries in the wiki. 89.71 EPDMS, if corrected, sits level with WAM-Diff and DriveFine (89.7) and below about a dozen corrected-cohort entries (WA-JEPA 91.7 … DriveFuture 89.9), none of which is in its table.

---

## Method

![[paradigm14.png|Figure 1: Comparison of world model paradigms. (a) Pixel-based world model: iteratively predicts actions and images, prone to scene noise; (b) latent-based world model: predicts latent states, weak in spatiotemporal dynamics modeling; (c) hybrid world model: simultaneously models iterative action-image prediction and future latent state prediction.]]

**Figure 1**: Comparison of world model paradigms. (a) Pixel-based: predicts actions and images, prone to scene noise. (b) Latent-based: predicts latent states, weak in spatiotemporal dynamics. (c) Hybrid: both.

![[framework3.png|Figure 2: Overview of HyWorldVLA. A text-guided video VAE provides latent targets; pre-training predicts text, image and action tokens plus a future latent from a learnable query Q; co-fine-tuning predicts only the latent, which an action model reads through shared attention.]]

**Figure 2**: Overview. A text-guided video VAE compresses and reconstructs clips and provides the ground-truth latent $z_2$. In pre-training, interleaved text, image and action tokens are predicted autoregressively and a learnable query $Q$ aggregates context to predict $\hat z_2$. In co-fine-tuning, the history and $Q$ go through the VLM to give global context and $\hat z_2$, which pass through joint attention to the action expert.

### Problem setup

At time $t$ the model sees front-camera frames $\mathbf V_{t-H:t}$, past waypoints $\mathbf W_{t-H:t}$ with $w_t=(x_t,y_t,\theta_t)$, and navigation commands $\mathbf L_{t-H:t}$, and outputs $\mathbf W_{t+1:t+T}$. $H = 1$ s, $T = 4$ s.

### Text-guided video VAE (§3.2)

VideoVAEPlus (cross-modal video VAE): a spatial encoder built on the Stable Diffusion image VAE with 3D convolutions (8× spatial, full temporal length) gives $\mathbf z_1\in\mathbb R^{c\times T\times\frac H8\times\frac W8}$; a temporal autoencoder of 3D ResNet blocks (4× temporal) gives $\mathbf z_2\in\mathbb R^{c'\times\frac T4\times\frac H8\times\frac W8}$. Each block has text cross-attention: visual patches are query and value, **Flan-T5 text embeddings are the keys**, added back residually. Captions come from **Qwen3.6-plus**. The paper credits text guidance with suppressing motion ghosting, edge blur and flicker; no reconstruction metric is reported.

$$\mathcal L_\mathrm{vae}=\mathcal L_\mathrm{rec}+\lambda_\mathrm{GAN}\mathcal L_\mathrm{GAN}+\lambda_\mathrm{KL}\mathcal L_\mathrm{KL}$$

Fine-tuned from the 4-channel checkpoint on nuPlan, 8 frames per segment at 216×216, 100K steps on 4 Alibaba PPUs, batch 1 per device.

### Pre-training (§3.3)

The clip $\mathbf V_{t-H:t+T}$ is split into $N=(H+T)/t_\mathrm{chunk}+1=6$ one-second chunks. Per chunk: the **first frame** is VQ-tokenized (VQGAN-style, Emu3's tokenizer), the ego-relative action chunk is FAST-tokenized, and the command is tokenized with Emu3's text tokenizer. One learnable query $Q$ sits inside the sequence; its hidden state goes through an MLP to $\hat{\mathbf z}_2$.

$$\mathcal S=[L_q^1,V_q^1,A_q^1,L_q^2,V_q^2,Q,A_q^2,L_q^3,\ldots,V_q^N,A_q^N]$$

$$\mathcal L_\mathrm{world}=\sum_j\mathrm{CE}(\hat L_q^j,L_q^j)+\sum_j\mathrm{CE}(\hat A_q^j,A_q^j)+\lambda_1\sum_j\mathrm{CE}(\hat v_q^j,v_q^j)+\lambda_2\|\hat{\mathbf z}_2-\mathbf z_2\|_2^2,\qquad\lambda_1=0.5,\ \lambda_2=0.1$$

Causal masking throughout. The equation places $Q$ after the second chunk's image tokens; which future window $\mathbf z_2$ encodes is described only as "the future video sequence". Learning rate 2.2e-4, 4,000 steps, batch 4 "across two nodes, each equipped with 16 PPUs" (whether per device or total is not stated).

### Co-fine-tuning (§3.4)

The VLM and action expert are trained jointly with π0-style joint attention. The action expert's sequence is $[\hat{\mathbf z}_2$, historical actions, commands$]$; hidden states at the action positions go through an MLP head.

$$\text{flow matching: }\ \mathcal L=\|\hat v-v\|_2^2+\lambda_3\|\hat z-z\|_2^2\qquad\text{selection: }\ \mathcal L=\mathrm{KL}(p_\mathrm{pred}\,\|\,p_\mathrm{gt})+\lambda_3\|\hat z-z\|_2^2,\qquad \lambda_3=0.1$$

For the selection head, $p_\mathrm{gt}$ is "computed by comprehensively considering the expert trajectory as well as the metrics of the evaluation": a Hydra-MDP-style target that puts the benchmark's metric into training ([[concepts/selection-based-planning.md]]). The vocabulary size is not stated. NAVSIM trainval, 4,000 steps, batch 96, learning rate 5e-5.

---

## Results

### Table 1: NAVSIM v1 (navtest)

Residual = closed form $\mathrm{NC}\cdot\mathrm{DAC}\cdot(5\mathrm{EP}+5\mathrm{TTC}+2\mathrm C)/12$ − reported (computed here). The paper's group labels are kept, including its spelling "Word Model".

| Method | Ref | Sensors | NC ↑ | DAC ↑ | TTC ↑ | C. ↑ | EP ↑ | PDMS ↑ | Residual |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Human | – | – | 100 | 100 | 100 | 99.9 | 87.5 | 94.8 | −0.0 |
| *End-to-end-based* | | | | | | | | | |
| UniAD | CVPR'23 | 6x Cam | 97.8 | 91.9 | 92.9 | 100.0 | 78.8 | 83.4 | −4.1 |
| TransFuser | TPAMI'23 | 3x Cam + L | 97.7 | 92.8 | 92.8 | 100.0 | 79.2 | 84.0 | −3.9 |
| PARA-Drive | CVPR'24 | 6x Cam | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 | −4.0 |
| Hydra-MDP | arXiv'24 | 3x Cam + L | 98.3 | 96.0 | 94.6 | 100.0 | 78.7 | 86.5 | −2.6 |
| DiffusionDrive | CVPR'25 | 3x Cam + L | 98.2 | 96.2 | 94.7 | 100.0 | 82.2 | 88.1 | −2.7 |
| *VLA-based* | | | | | | | | | |
| AutoVLA | NeurIPS'25 | 3x Cam | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 | −2.9 |
| ReCogDrive-8B | ICLR'26 | 3x Cam | 97.8 | 97.7 | 94.9 | 100.0 | 86.3 | 90.5 | −2.4 |
| *Pixel-based "Word" Model* | | | | | | | | | |
| WoTE | ICCV'25 | 3x Cam + L | 98.5 | 96.8 | 94.4 | 99.9 | 81.9 | 88.3 | −2.4 |
| FSDrive | NeurIPS'25 | 1x Cam | 98.2 | 93.8 | 93.3 | 99.9 | 80.1 | 85.1 | −3.2 |
| ReSim | NeurIPS'25 | 1x Cam | – | – | – | – | – | 86.6 | – |
| PWM | NeurIPS'25 | 1x Cam | 98.6 | 95.9 | 95.4 | 100.0 | 81.8 | 88.1 | −2.5 |
| CoWorld-VLA | arXiv'26 | 1x Cam | 99.2 | 96.8 | 96.6 | 100.0 | 83.6 | 89.8 | −1.7 |
| DriveLaW | CVPR'26 | 1x Cam | 99.0 | 97.1 | 96.7 | 100.0 | 81.3 | 89.1 | −1.8 |
| DriveVLA-W0 | ICLR'26 | 1x Cam | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.2 | −1.2 |
| *Latent-based "Word" Model* | | | | | | | | | |
| LAW | ICLR'25 | 1x Cam | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 | −4.0 |
| World4Drive | ICCV'25 | 1x Cam | 97.4 | 94.3 | 92.8 | 100.0 | 79.9 | 85.1 | −3.7 |
| Epona | ICCV'25 | 1x Cam | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 | −3.1 |
| DreamerAD | arXiv'26 | 1x Cam | 98.0 | 97.2 | 94.3 | 100.0 | 83.1 | 88.7 | −2.4 |
| OneVL | arXiv'26 | 1x Cam | – | – | – | – | – | 88.8 | – |
| WorldRFT | AAAI'26 | 1x Cam | 97.8 | 96.8 | 94.0 | 100.0 | 81.7 | 87.8 | −2.7 |
| ResWorld | ICLR'26 | 4x Cam + L | 98.9 | 96.5 | 95.6 | 100.0 | 83.1 | 89.0 | −2.0 |
| **HyWorldVLA** | – | 1x Cam | 98.6 | 98.9 | 95.1 | 99.3 | 84.6 | **90.59** | −1.4 |

All residuals fall in the usual v1 band (−1.2 to −4.1), the headline among them. Three rows differ from the wiki's records: **WoTE's TTC is 94.4 here and 94.9 in every other copy**, including [[sources/idol.md]]'s; **CoWorld-VLA's published row is 99.1 / 97.0 / 96.5 / 100 / 84.0 / 90.0** ([[sources/coworld-vla.md]]), not this one; ReCogDrive-8B's 90.5 is the later-version figure also printed by FLARE and SGDrive.

### Table 2: NAVSIM v2 (navtest, extended metrics)

Residual = closed form $\mathrm{NC}\cdot\mathrm{DAC}\cdot\mathrm{DDC}\cdot\mathrm{TLC}\cdot(5\mathrm{EP}+5\mathrm{TTC}+2\mathrm{LK}+2\mathrm{HC}+2\mathrm{EC})/16$ − reported. Positive residuals of +1 to +4 fingerprint the pre-fix evaluator, slightly negative ones the corrected one ([[concepts/navsim-benchmark.md#residual-sign]]).

| Method | NC ↑ | DAC ↑ | DDC ↑ | TLC ↑ | EP ↑ | TTC ↑ | LK ↑ | HC ↑ | EC ↑ | EPDMS ↑ | Residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *End-to-end-based* | | | | | | | | | | | |
| TransFuser | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.4 | 98.3 | 87.2 | 76.7 | +1.25 |
| HydraMDP++ | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 | +2.16 |
| DriveSuprim | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 | +2.28 |
| ARTEMIS | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | – | 83.1 | – |
| DiffusionDrive | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 | +2.48 |
| *VLA-based* | | | | | | | | | | | |
| ReCogDrive | 98.3 | 95.2 | 98.3 | 99.8 | 87.1 | 97.5 | 96.6 | 99.5 | 86.5 | 83.6 | +1.79 |
| *Pixel-based "Word" Model* | | | | | | | | | | | |
| DriveVLA-W0 | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 | −1.31 |
| ExploreVLA | 98.8 | 96.2 | 99.6 | 99.8 | 87.1 | 98.2 | 97.8 | 98.3 | 86.8 | 88.8 | −0.68 |
| *Latent-based "Word" Model* | | | | | | | | | | | |
| DriveWorld-VLA | 98.6 | 99.1 | 99.6 | 99.8 | 87.4 | 97.9 | 97.0 | 97.9 | 78.6 | 86.8 | +2.65 |
| DreamerAD ⚠ *(this is DriveWorld-VLA's row)* | 98.6 | 99.1 | 99.6 | 99.8 | 87.4 | 97.9 | 97.0 | 97.9 | 78.6 | 86.8 | +2.65 |
| Latent-WAM ⚠ *(this is DreamerAD's row)* | 98.0 | 97.2 | 99.5 | 99.8 | 87.8 | 97.4 | 97.5 | 98.3 | 72.4 | 87.7 | −1.24 |
| **HyWorldVLA** | 98.8 | 98.2 | 99.5 | 99.9 | 87.8 | 98.1 | 96.2 | 98.3 | 77.4 | **89.71** | **−0.91** |
| *Latent-WAM's published row, for reference* ([[sources/latent-wam.md]]) | 98.1 | 97.3 | 99.6 | 99.8 | 87.7 | 97.3 | 97.6 | 98.1 | 87.3 | 89.3 | −0.88 |

The paper's group header rows have nine cells where the table has eleven columns; nothing else is affected. The evaluator is not named. The own row is corrected-like; the five rows of the shared baseline block that can be checked are pre-fix-like; DriveVLA-W0 is the block's known odd row.

---

## Ablations (navtest, selection head)

### Table 3: Components

| Config | NC ↑ | DAC ↑ | TTC ↑ | C. ↑ | EP ↑ | PDMS ↑ | Δ vs full |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pure WAM (pixel tokens only) | 98.3 | 98.6 | 94.5 | 99.6 | 84.0 | 89.91 | −0.68 |
| Pure LWM (latent only) | 97.2 | 97.8 | 92.0 | 98.9 | 82.4 | 87.50 | −3.09 |
| w/o language guidance in latent | 98.5 | 98.8 | 94.7 | 99.6 | 84.5 | 90.35 | −0.24 |
| w/o latent condition in action expert | 98.6 | 98.7 | 94.9 | 99.3 | 84.3 | 90.29 | −0.30 |
| w/o latent supervision in co-fine-tuning | 98.6 | 98.9 | 95.3 | 99.8 | 83.3 | 90.17 | −0.42 |
| **Ours** | 98.6 | 98.9 | 95.1 | 99.3 | 84.6 | **90.59** | – |

- **Pure WAM is the matched DriveVLA-W0 control**, and it lands 0.29 below DriveVLA-W0's published 90.2 on the same architecture. So "+0.39 over DriveVLA-W0" (Table 1) and "+0.68 over pure WAM" (Table 3) are the same comparison measured against two baselines.
- **Pure LWM lowers every sub-score**, TTC most (−3.1). With no row lacking both targets, it is not known whether the latent-only model is worse than no world model.
- **w/o latent supervision in co-fine-tuning keeps the latent pathway** (the action expert still reads $\hat z_2$, now trained only by the action loss). It has the best TTC and comfort of any row and the lowest EP; the loss trades progress for nothing on safety.

### Table 4: Pre-training loss weights

(The text cites this as "Table 5".)

| $\lambda_1$ (vision tokens) | $\lambda_2$ (latent) | NC ↑ | DAC ↑ | TTC ↑ | C. ↑ | EP ↑ | PDMS ↑ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.1 | 98.6 | 98.7 | 94.9 | 99.6 | 84.6 | 90.48 |
| **0.5** | **0.1** | 98.6 | 98.9 | 95.1 | 99.3 | 84.6 | **90.59** |
| 1.0 | 0.1 | 98.5 | 98.5 | 94.9 | 99.6 | 83.5 | 89.83 |
| 0.5 | 0.2 | 98.6 | 98.9 | 95.0 | 99.5 | 84.4 | 90.47 |
| 0.5 | 0.5 | 98.6 | 98.6 | 94.5 | 99.4 | 84.9 | 90.34 |
| 0.5 | 1.0 | 98.6 | 98.6 | 94.7 | 99.5 | 84.4 | 90.16 |

### Table 5: Fine-tuning latent weight

| $\lambda_3$ | NC ↑ | DAC ↑ | TTC ↑ | C. ↑ | EP ↑ | PDMS ↑ |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 98.5 | 98.8 | 95.0 | 99.3 | 84.6 | 90.47 |
| **0.1** | 98.6 | 98.9 | 95.1 | 99.3 | 84.6 | **90.59** |
| 0.2 | 98.6 | 98.7 | 94.6 | 99.4 | 83.9 | 90.01 |
| 1.0 | 98.6 | 98.3 | 95.0 | 99.4 | 83.4 | 89.75 |

Every navtest configuration except Pure LWM lies in 89.75–90.59, a 0.84-point band, against a training-seed sd of 0.1 measured once for another planner ([[concepts/evaluation-variance.md]]). The residuals of the Table 3–5 rows sit between −1.38 and −1.73 except Pure LWM (−2.75), so sub-scores and aggregates are mutually consistent.

### Table 7 (Appendix B.1): Flow-matching action expert

| Method | NC ↑ | DAC ↑ | TTC ↑ | C. ↑ | EP ↑ | PDMS ↑ |
|---|---:|---:|---:|---:|---:|---:|
| DriveVLA-W0 (fm) | 98.4 | 95.3 | 95.2 | 100.0 | 80.9 | 87.2 |
| **Ours (fm)** | 98.8 | 96.3 | 95.5 | 100.0 | 82.9 | **88.76** |

The DriveVLA-W0 row is that paper's published flow-matching result (87.2 in its Table 4), not a rerun. The selection head beats flow matching by **1.83** here; in DriveVLA-W0 the anchor head beat flow matching by 3.0 (90.2 vs 87.2). Both gaps are larger than the world-model effects in this paper, and the selection head is the one whose targets come from the evaluation metric.

---

## Scene-Noise Robustness

### The quantitative test (§4.5, Table 6)

655 cases from the OpenScene test split, "all of which contain non-uniform noise due to rain and fog". How they were found, whether they lie in navtest, and whether they will be released are not stated.

| Method / config | NC ↑ | DAC ↑ | TTC ↑ | C. ↑ | EP ↑ | PDMS ↑ | Same model on navtest | Drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WoTE | 98.9 | 76.0 | 64.4 | 100.0 | 64.4 | 60.65 | 88.3 | −27.7 |
| DriveLaW | 99.4 | 75.7 | 82.9 | 100.0 | 69.5 | 67.49 | 89.1 | −21.6 |
| DriveVLA-W0 | 93.4 | 76.2 | 83.1 | 98.6 | 59.8 | 61.18 | 90.2 | −29.0 |
| Pure WAM | 99.1 | 75.7 | 94.3 | 99.8 | 67.8 | 69.95 | 89.91 | −20.0 |
| w/o latent supervision in co-fine-tuning | 99.2 | 78.5 | 94.8 | 99.7 | 70.9 | 73.18 | 90.17 | −17.0 |
| **Ours** | 99.7 | 91.1 | 96.9 | 99.2 | 84.4 | **86.87** | 90.59 | −3.7 |

### Table 8 (Appendix B.2): Fine-tuning latent weight on the noisy subset

| $\lambda_3$ | NC ↑ | DAC ↑ | TTC ↑ | C. ↑ | EP ↑ | PDMS ↑ | Same model on navtest |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 *(Table 6)* | 99.2 | 78.5 | 94.8 | 99.7 | 70.9 | 73.18 | 90.17 |
| 0.05 | 99.5 | 88.7 | 96.6 | 99.1 | 82.4 | 84.52 | 90.47 |
| 0.1 | 99.7 | 91.1 | 96.9 | 99.2 | 84.4 | 86.87 (+2.35) | 90.59 |
| 0.2 | 99.1 | 94.8 | 94.5 | 99.4 | 88.1 | 89.45 (+2.58) | 90.01 |
| 1.0 | 98.8 | 98.3 | 95.1 | 100.0 | 90.8 | 92.94 (+3.49) | 89.75 |

(The paper prints the bracketed increments; the λ3 = 0 row and the navtest column are added here from Tables 3, 5 and 6.)

**What the two tables show.**

1. **The effect is in drivable-area compliance and progress.** Every model without fine-tuning latent supervision fails DAC in about a quarter of these scenes (75.7–78.5); HyWorldVLA fails it in 8.9%, and at $\lambda_3 = 1.0$ in 1.7%. NC barely moves.
2. **A tiny loss weight does most of the work**: λ3 0 → 0.05 is +11.3 on the subset and +0.3 on navtest. That is the same mechanism measured as small on navtest and large elsewhere, the shape [[concepts/navhard-ood-evaluation.md]] records for SUV, Metis and GeoWAM on navhard.
3. **The subset is not uniformly hard.** At λ3 = 1.0 it scores 92.94 with EP 90.8 and DAC 98.3, above that model's navtest 89.75 (EP 83.4, DAC 98.3). A model that has stopped failing drivable area finds these scenes easier than navtest. The subset's composition (speed, scenario mix) is not reported, so "robustness to noise" and "different scenario mix" cannot be separated from the table.
4. **Spread between similar checkpoints is large.** DriveVLA-W0 and Pure WAM, two pixel-only Emu3 models within 0.29 PDMS on navtest, are 8.77 apart on the subset. Any single comparison on it needs that calibration. At 655 scenes the scene-sampling standard error alone is at least about 0.7 PDMS (navtest's per-scene sd of 17.84 over √655), against 0.16 on full navtest ([[concepts/evaluation-variance.md]]); the subset's own sd is probably larger.
5. **The paper's reading is a trade-off**: larger λ3 helps the noisy subset and hurts navtest beyond 0.1, which it names a future direction. The other reading, that the latent MSE acts as a regularizer whose strength sets how conservative the planner is in degraded scenes, is not tested; nor is a semantic latent target, or augmentation with synthetic weather as a control.
6. **WoTE's row repeats 64.4 for TTC and EP.** TTC at 64.4 with NC at 98.9 would mean time-to-collision violations in a third of scenes with almost no collisions; possibly a transcription error. WoTE uses camera and LiDAR; whether it was given LiDAR is not stated.

### The qualitative tests

![[hyworldvla_fig3.jpg|Figure 3: Hybrid world modeling improves robustness to scene noise. (a) Turn right under fluctuating illumination (light vs dark input). (b) Turn left and go straight under non-uniform rain and fog noise (hazy, blurry). Columns: front-view image, HyWorldVLA BEV, DriveVLA-W0 BEV.]]

**Figure 3**: Hybrid world modeling improves the robustness to scene noise. (a) DriveVLA-W0 suffers from severe behavioural variance under illumination changes, while HyWorldVLA retains consistency. (b) DriveVLA-W0 is overly conservative, sacrificing traffic efficiency, while HyWorldVLA keeps accurate, human-aligned trajectories. Legend: NPC and ego boxes, dashed lane centre lines, green dots ground truth, salmon dots prediction. *(Wiki note: the "Dark" front image carries a Doubao "AI生成" watermark, bottom right; HyWorldVLA's Light and Dark BEV panels in (a) are the same file in the arXiv HTML. The (a) sub-caption says "turn right" while Appendix C.4 calls it "the left-turn scenario in the main text".)*

![[hyworldvla_fig11.jpg|Figure 11: Turn right and go straight under fluctuating illumination. (a) DriveVLA-W0 goes straight under a right-turn instruction in the dark version while HyWorldVLA turns. (b) DriveVLA-W0 changes from a lane change to going straight; HyWorldVLA keeps going straight.]]

**Figure 11**: Turn right and go straight under the fluctuating illumination. (a) DriveVLA-W0 goes straight under the right-turn instruction; HyWorldVLA turns. (b) DriveVLA-W0's behaviour changes from changing lanes to going straight; HyWorldVLA keeps going straight. *(Both "Dark" images carry the Doubao watermark; here the four HyWorldVLA panels are distinct files.)*

![[hyworldvla_fig12.jpg|Figure 12: Remain stationary at a red-light intersection under blur noise. DriveVLA-W0 crawls during red-light queuing, while HyWorldVLA remains stationary. Two cases.]]

**Figure 12**: Remain stationary at a red-light intersection under blur noise (two cases). DriveVLA-W0 crawls while queuing; HyWorldVLA stays put. The input frames show raindrops on the lens, not synthetic blur.

![[hyworldvla_fig13.jpg|Figure 13: Go straight under blur noise. DriveVLA-W0 slows down because of the blurry image; HyWorldVLA keeps a reasonable speed, closer to the human trajectory. Six cases.]]

**Figure 13**: Go straight under the blur noise (six cases). DriveVLA-W0 slows because of the blurry image; HyWorldVLA keeps a reasonable speed, closer to human driving.

---

## Visualizations

### Future frame generation (Appendix C.1)

![[hyworldvla_fig4.jpg|Figure 4: Future image generation. (a) Right turn with a limited front view: the model forecasts lane markings absent from the input. (b) Meeting an oncoming bus on a narrow road. Rows: input frames V t-1 and V t, ground-truth future V t+1 to V t+4, predicted future.]]

**Figure 4**: Future image generation at 1 s spacing, 256×144. (a) During a right turn the model forecasts the central double yellow line, absent from the input frames. (b) The paper says the model forecasts the oncoming bus's motion and the ego's rightward yielding. *(Wiki note: in (b) the predicted frames for t+2 to t+4 are nearly identical while the ground truth keeps moving, so the yielding manoeuvre is not visible in the prediction. Which checkpoint generated the frames is not stated. The failure cases in Figures 8–10 show predicted frames beside the planner's own trajectories, so the planning model apparently still decodes image tokens after a fine-tuning stage that no longer trains them.)*

### Planning comparisons (Appendix C.2)

![[hyworldvla_fig5.jpg|Figure 5: Turn right and stop behind a truck with its trunk open. DriveVLA-W0 tries to nudge past the stopped truck; HyWorldVLA stops at a fitting distance.]]

**Figure 5**: Turn right and stop behind a truck with its trunk open. DriveVLA-W0 tries to nudge past; HyWorldVLA stops at a suitable distance.

![[hyworldvla_fig6.jpg|Figure 6: Left turn at a small intersection with a limited view. DriveVLA-W0 goes straight; HyWorldVLA turns left.]]

**Figure 6**: Left turn at a small intersection with a limited view. DriveVLA-W0 goes straight where it should turn left; HyWorldVLA turns.

![[hyworldvla_fig7.jpg|Figure 7: Turn left at a construction intersection. DriveVLA-W0 enters the opposite lane; HyWorldVLA turns left and keeps to the right lane.]]

**Figure 7**: Turn left at the construction intersection. DriveVLA-W0 goes into the opposite lane; HyWorldVLA keeps to the right lane.

### Failure cases (Appendix C.3)

![[hyworldvla_fig8.jpg|Figure 8: Failure case, small critical elements. The model nudges past a queuing vehicle; its world-model prediction treats the vehicle as slowly moving, missing the red light and brake lights. Rows: planning inputs and trajectory, predicted future frames (WM), ground truth (GT).]]

**Figure 8**: Deficits in capturing critical small elements. The model tries to nudge past a queuing vehicle. Its world-model frames show the vehicle as slowly moving; the red light ahead and the brake lights, a few pixels each, are missed.

![[hyworldvla_fig9.jpg|Figure 9: Failure case, front-view limits on a left turn. The model crosses the central double yellow line; the predicted frames lack the double road marking.]]

**Figure 9**: Constraints of a front-view camera on a left turn. The model crosses the double yellow line; the predicted frames lack the marking, since the history never showed the target lane.

![[hyworldvla_fig10.jpg|Figure 10: Failure case, front-view limits on a right turn. The model rear-ends a queuing lead vehicle; the predicted frames miss the preceding vehicle.]]

**Figure 10**: Constraints of a front-view camera on a right turn. The model rear-ends a queuing lead vehicle that the predicted frames miss. The paper concludes multi-view input is needed and names the camera-count–latency trade-off as unresolved.

In all three failures the generated future and the plan fail together, which is what a planner that reads its own prediction would do. The paper uses them as diagnostics; it does not test whether correcting the prediction would correct the plan.

---

## Table Problems {#table-problems}

1. **Two shifted rows in Table 2.** "Latent-WAM" = DreamerAD's published row on all ten numbers; "DreamerAD" = DriveWorld-VLA's row (HC 97.9 against 97.8 in the two other wiki copies). Latent-WAM's own 89.3 / EC 87.3 is absent. The text's claim of "significantly outperforming ... latent-based WM baselines including Latent-WAM" rests on the wrong row.
2. **A mixed-evaluator v2 table**, the tenth user of the shared baseline block. Comparisons with TransFuser, HydraMDP++, DriveSuprim, DiffusionDrive and ReCogDrive cross evaluators; against ExploreVLA (88.8, corrected-like here) the margin is +0.9.
3. **CoWorld-VLA at 89.8** on sub-scores matching no row the wiki holds; its published row is 90.0.
4. **DriveVLA-W0 (fm) in Table 7** is a copied published row, not a rerun.
5. **Category assignments**: WoTE, a BEV world model with LiDAR, and CoWorld-VLA, whose four targets include V-JEPA and VGGT latents, are filed as pixel-based; Epona, a latent-diffusion video model, as latent-based beside DriveLaW, also a latent video model, as pixel-based; OneVL, which decodes future frames from its latent tokens, as latent-based. The pixel/latent split the paper argues from is not applied consistently to its own baselines.
6. **Text–table and caption mismatches**: "Table 5" for Table 4; Figure 3(a) "turn right" vs "left-turn scenario" in C.4.
7. **Figures**: Figure 3(a) reuses one file for HyWorldVLA's two conditions; the illumination inputs are Doubao-generated edits (watermarked).

---

## Limitations

1. **Small, single-run navtest effects.** Every component except pixel tokens is worth 0.24–0.42 PDMS; the headline gain over the matched pixel model is 0.68. No seeds, no confidence intervals.
2. **No no-world-model control**, so the paper cannot say whether a latent-only target helps at all.
3. **The robustness subset is private and its selection undescribed**, its baselines' run protocol is unstated, and its scores are not comparable to navtest scores (§above).
4. **Generative image edits as a controlled perturbation**; one duplicated figure panel.
5. **The latent target is a pixel-reconstruction code.** The claim that latent prediction is inherently noise-robust is not tested against a semantic target (DINO, V-JEPA) or against augmentation.
6. **Pre-training data overlap is unstated.** OpenScene is the pre-training corpus and navtest derives from OpenScene ([[concepts/navsim-benchmark.md]]); the pre-training objective includes the ego's future action tokens and future frames. Excluding test logs is not mentioned, as in several other wiki papers.
7. **One front camera at 256×144**; the paper's own failure cases (Figures 9 and 10) are field-of-view failures.
8. **No latency, no parameter count, no code**, for an 8B autoregressive backbone with an action expert.
9. **No navhard, HUGSIM, Bench2Drive or closed-loop result**; the noisy subset is the only out-of-distribution test.
10. **The headline uses a metric-supervised trajectory vocabulary**; the unsupervised flow-matching head scores 88.76.
11. **Comparison scope stops at 90.5 on v1 and 88.8 on v2**, below much of the wiki's ladder on both.

---

## Relation to Other Wiki Pages

- [[sources/drivevla-w0.md]]: the base stack and the main baseline (anchor head 90.2★, FM 87.2, v2 86.1 with EC 58.9). HyWorldVLA's EC 77.4 is +18.5 over it.
- [[sources/drive-hwm.md]]: the other Emu3-8B derivative with two future targets (a V-JEPA flow latent and next-frame RGB), posted in September 2026. Drive-HWM runs the two at two rates and fuses by FiLM; HyWorldVLA drops the pixel target after pre-training. Neither cites the other (Drive-HWM is later).
- [[sources/coworld-vla.md]], [[sources/onevl.md]]: earlier mixed-target designs (four parallel targets; latent tokens with a visual decoder), both cited here but filed as single-paradigm methods.
- [[sources/drivelaw.md]]: holds the planner fixed and finds video latents > VLM hidden states > BEV (89.1 / 86.5 / 84.1). HyWorldVLA's pixel-token vs VAE-latent vs both comparison (89.91 / 87.50 / 90.59) is a second fixed-planner representation comparison, with the opposite order between pixels and latents, but with a regressed VAE code rather than a generator's internal latent.
- [[sources/wa-jepa.md]], [[sources/redrive.md]], [[sources/foredrive.md]], [[sources/world4scorer.md]]: the objective-form debate (regression on future latents harmful vs helpful). HyWorldVLA's MSE on a single-camera VAE code helps at λ3 = 0.1 and hurts above 0.2 on navtest, and helps monotonically on its noisy subset ([[concepts/world-model-for-ad.md]]).
- [[sources/foredrive.md]]: the closest consumed-shared-latent result. Reading the predicted future is worth +0.4 to +0.6 there and +0.30 here.
- [[sources/momworld.md]]: the other ingested paper with weather-corruption tests (nuScenes-C; margins of 0.001–0.04 points). It is later (September 2026), so HyWorldVLA's NAVSIM rain/fog subset is the wiki's first weather-robustness test on NAVSIM.
- [[sources/latent-wam.md]], [[sources/dreameraD.md]]: the misattributed v2 rows.
- [[concepts/selection-based-planning.md]]: the Hydra-MDP-style head with metric-derived targets; selection over FM worth +1.83 here and +3.0 in DriveVLA-W0.

---

## Open Questions

- **Is the robustness the latent target or the regularization?** A matched run with a semantic latent target (DINOv2 / V-JEPA features) in place of the VAE code, and one with synthetic weather augmentation and no latent loss, would separate "latent prediction" from "any extra loss that makes the planner cautious in degraded scenes".
- **Does the effect survive a released, documented subset?** The 655 scenes, their selection rule and their navtest overlap would let the 13.7-point fine-tuning effect be checked, and a navhard run would show whether the same mechanism moves the standard hard split.
- **Does the latent-only model beat no world model?** One row (no pixel tokens, no latent) would place Pure LWM's 87.50.
- **Why does the hybrid help pixel tokens more than the latent helps alone?** Pixel tokens over latent-only is +3.09; the latent over pixel-only is +0.68. Whether the pixel stage mainly trains the backbone for the action task, as DriveVLA-W0's transfer table suggests, is untested.
