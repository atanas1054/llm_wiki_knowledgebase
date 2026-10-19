---
title: "RAF-VLA: Representation Alignment with the Future for End-to-End Autonomous Driving"
type: source-summary
sources: ["raw/papers/RAF-VLA_ Representation Alignment with the Future for End-to-End Autonomous Driving.md"]
related: [sources/drivewla.md, sources/drivevla-w0.md, sources/flare.md, sources/hyworldvla.md, sources/lvdrive.md, sources/deepsight.md, sources/world4scorer.md, sources/foredrive.md, sources/reworld.md, sources/walt.md, sources/simvta.md, sources/explorevla.md, sources/sgdrive.md, sources/futuresightdrive.md, sources/policy-world-model.md, sources/recogdrive.md, sources/adathinkdrive.md, sources/autovla.md, sources/curious-vla.md, sources/nord.md, sources/hydra-mdp-pp.md, sources/da-wam.md, concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/rl-for-ad.md, concepts/best-of-n.md, concepts/inference-latency.md, concepts/action-tokenization.md, research-directions.md]
created: 2026-10-19
updated: 2026-10-19
confidence: medium
---

# RAF-VLA: Representation Alignment with the Future for End-to-End Autonomous Driving

**Paper**: RAF-VLA: Representation Alignment with the Future for End-to-End Autonomous Driving
**Authors**: Dogun Kim, Yongjae Lee, Joonhee Lim, Yeina Lee, Junhyeok Park, Moogeun Park, Dongsuk Kum (the clipping's corresponding-author marks are garbled)
**Orgs**: KAIST, Graduate School of Mobility and Robotics Program
**arXiv**: 2609.17728v1 (September 2026)
**Code**: none stated
**Source**: `raw/papers/RAF-VLA_ Representation Alignment with the Future for End-to-End Autonomous Driving.md`. The clipping has all text and Tables I–VI and `RAF-VLA_Figure_2.png`. Figure 1 is only an external arXiv link, and Figure 3 is missing from both the clipping and the arXiv HTML. Added to `raw/assets/` (new files only): `RAF-VLA_Main_figure1.png`, downloaded from the arXiv HTML, and `raf_vla_fig3.png`, cropped from page 6 of the arXiv PDF at 300 dpi.

**Confidence: medium.** The central ablation is controlled and repeated through RL. Alignment adds +1.21 PDMS after SFT and +1.18 after GRPO, and every alignment configuration tried beats no alignment. All v1 rows pass the closed-form check. But it is one run per cell on navtest. Its horizons and query count were picked on the test split. The no-alignment baseline also removes the world-query tokens. Its v2 table mixes three evaluator conventions, and the "training samples seen" comparison is the authors' own estimate.

---

## What It Is

**A DriveVLA-W0-style Mixture-of-Transformers VLA with eight extra "world query" tokens whose hidden states are regressed onto a frozen Cosmos encoder's representation of the front frame 1 s and 4 s ahead, then read by the action expert at inference.**

1. **Vision-Language Expert**: Qwen2.5-VL-7B-Instruct over a front-camera image sequence (with history) and a navigation instruction, followed by $K=4$ learnable world queries for each aligned horizon (1 s and 4 s). The queries attend to image and language tokens. Attention between horizons is blockwise causal and bidirectional within one.
2. **Action Expert**: a smaller transformer (width 1024) over a status token (recent ego poses and the one-hot command, through an MLP) and learnable action queries. It shares joint attention with the VL expert layer by layer, reading the VL expert but not read by it. A "query-based action head with multiple trajectory anchors", following DriveVLA-W0 and Hydra-MDP, decodes the trajectory with DriveVLA-W0's loss $\mathcal L_\text{act}$.
3. **Future-Aligned SFT**: $\mathcal L_\text{SFT}=\mathcal L_\text{act}+\lambda\,\frac1{|\mathcal H_F|}\sum_h\lVert f(I_{t+h})-g_\phi(\mathbf Z_{t,h})\rVert_2^2$, with $\lambda=1$. Here $f$ is "a pretrained encoder from the Cosmos world model" (frozen; which one, and the target's shape, are not stated), $g_\phi$ a two-layer MLP, and $\mathbf Z_{t,h}$ the four concatenated world-query states. After SFT, $f$ and $g_\phi$ are dropped. **The world queries stay, and the action expert reads them at inference.**
4. **RFT**: GRPO with the PDMS reward ($G=8$, KL 0.04, 1.5 epochs). The vision encoder, world queries and action head are frozen; how a group of trajectories is sampled from an anchor head is not described.

**Results**

| Benchmark | RAF-VLA | Notes |
|---|---:|---|
| NAVSIM-v1 PDMS (navtest), single sample | **90.0** (90.03) | Front camera, navtrain; DriveVLA-W0 90.2, SGDrive 91.1 in its table |
| NAVSIM-v1, best-of-6 | 94.2 | Oracle selection |
| NAVSIM-v2 EPDMS (navtest), SFT only | 86.2 | EC 65.7 |
| Inference (H200, batch 1) | 153 ms (152 ms without alignment) | |
| Training | SFT 10.96 h, RFT 7 h on 4 × H200 | SFT +3.8% for alignment |

---

## Key Takeaways

- **Future alignment adds about +1.2 PDMS, and the gain survives GRPO.**
  - SFT: 87.92 → 89.13.
  - SFT then RFT: 88.85 → 90.03.
  - GRPO itself adds +0.93 without alignment and +0.90 with it, so the auxiliary is worth as much as the RL stage.
  - Every alignment variant tried beats none by +0.68 to +1.21 (two horizon sets, four query counts), so the sign is robust even though the chosen settings are navtest maxima.
- **This is a consumed latent future, not a training-only loss.** The paper's framing is "without future generation". Its world queries are trained to regress future representations, kept at inference, and attended by the action expert, like [[sources/hyworldvla.md]]'s latent query (+0.30) and [[sources/foredrive.md]]'s consumed latents (+0.4 to +0.6). They cost 1 ms. The no-alignment baseline removes the queries entirely, so the extra tokens and the alignment loss are priced together, and no arm hides the queries from the action expert.
- **The closest prior is uncited and stronger.** [[sources/flare.md]] (OpenDriveLab and Li Auto) also adds future-feature regression to a front-camera VLA and then runs GRPO. It reaches 91.4 PDMS, and its future-feature objective is worth +3.5 in SFT, against RAF-VLA's +1.2. Neither paper compares with the other.
- **90.0 PDMS ties DriveVLA-W0 (90.2) without generating images**, on a different backbone (Qwen2.5-VL-7B against Emu3-8B) and with the same action-head design. On the wiki's v1 ladder 90.0 is mid-table. Its table omits DriveFine 90.7, WAM-Diff 91.0, FLARE 91.4, WCog-VLA 92.9 and CLEAR 93.7, all earlier.
- **Best-of-6 94.2 is not "the highest overall"** outside its table: Curious-VLA's best-of-6 is 94.8 and TransDiffuser's best-of-N 94.9 ([[concepts/best-of-n.md]]).
- **The v2 claim crosses evaluators.** "86.2, matching the best reported SFT result" sits in a table with three conventions. TransFuser, Hydra-MDP++, ReCogDrive and DiffusionDrive are pre-fix-like (+1.3 to +2.7). VADv2 76.6 and Hydra-MDP 79.8 come from Hydra-MDP++'s own formula. DriveVLA-W0, SGDrive and RAF-VLA are corrected-like. On the corrected scale DiffusionDrive is 88.2 ([[concepts/navsim-benchmark.md]]), above RAF-VLA. EC is 65.7, the per-frame-selection signature of anchor heads (DriveVLA-W0 58.9).
- **"Substantially fewer training samples seen"** (Figure 1b: about 0.5M against 1.2M–27M) is estimated by the authors "from reported configurations, official implementations, and official dataset splits". It counts neither the backbone's nor the Cosmos encoder's pretraining.

---

## Method

![[RAF-VLA_Main_figure1.png|Fig. 1: Overview of RAF-VLA. (a) Overall architecture of RAF-VLA, where future-frame representations directly guide the policy's hidden states to shape planning-relevant representations. (b) RAF-VLA achieves competitive planning performance with substantially fewer training samples seen than recent VLA planners that learn to generate future scenes. (c) Two-stage training strategy consisting of Future-Aligned SFT followed by GRPO-based RFT. Downloaded from the arXiv HTML; the clipping links to it externally.]]

**Fig. 1** *(from the arXiv HTML)*: (a) Images at $t+h_1$ and $t+h_2$ go through frozen world encoders. Trainable MLPs map the world-query states toward those targets (dashed red: training only). Image, language and world-query tokens enter the VLM; status and action queries enter the action expert; an action head outputs the trajectory. (b) PDMS against training samples seen, by the authors' estimate: Ours (SFT) about 89.1 and Ours (RFT) about 90.0, both near 0.5M; DriveVLA-W0 90.2 at about 1.2M; ExploreVLA 90.4 at about 3.1M; FSDrive 85.1 at about 8M; SGDrive 91.1 at about 27M (broken x-axis). (c) Stage 1: Future-Aligned SFT against future images and the ground-truth trajectory. Stage 2: GRPO over $G$ rollouts.

![[RAF-VLA_Figure_2.png|Fig. 2: Mixture-of-Transformers architecture. (a) Joint attention between the two experts. (b) Attention mask across token groups.]]

**Fig. 2**: (a) One MoT layer: each expert has its own LayerNorm, QKV and FFN, and they share a joint attention. (b) The mask: image tokens attend to themselves, language to image and language, the $t+h_1$ world queries to image, language and themselves, the $t+h_2$ queries also to the $t+h_1$ queries. Status and action queries attend to everything.

### Training details (§IV-B)

| | Future-Aligned SFT | RFT (GRPO) |
|---|---|---|
| Epochs, batch | 4, 144 | 1.5, 288 |
| Learning rate | 1e-5, cosine | 2e-6 |
| Frozen | VLM vision encoder, world encoder | vision encoder, world queries, action head (world encoder and projector removed) |
| Other | horizons 1 s and 4 s, $K=4$, $\lambda=1$ | $G=8$, $\beta=0.04$, PDMS reward |
| Time (4 × H200) | about 11 h (10.96 h; 10.56 h without alignment) | about 7 h |

---

## Results

### Table I: NAVSIM v1 navtest

Residual = closed form − reported, computed here; † best-of-6.

| Method | Venue | Sensors | NC | DAC | EP | TTC | C. | PDMS | Residual |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| UniAD | CVPR'23 | 6 Cam | 97.8 | 91.9 | 78.8 | 92.9 | 100.0 | 83.4 | −4.12 |
| TransFuser | TPAMI'23 | 3 Cam + L | 97.7 | 92.8 | 79.2 | 92.8 | 100.0 | 84.0 | −3.91 |
| LAW | ICLR'25 | 1 Cam | 96.4 | 95.4 | 81.7 | 88.7 | 99.9 | 84.6 | −3.99 |
| Epona | ICCV'25 | 1 Cam | 97.9 | 95.1 | 80.4 | 93.8 | 99.9 | 86.2 | −3.12 |
| Hydra-MDP | arXiv'24 | 3 Cam + L | 98.3 | 96.0 | 78.7 | 94.6 | 100.0 | 86.5 | −2.63 |
| DiffusionDrive | CVPR'25 | 3 Cam + L | 98.2 | 96.2 | 82.2 | 94.7 | 100.0 | 88.1 | −2.72 |
| WoTE | ICCV'25 | 3 Cam + L | 98.5 | 96.8 | 81.9 | 94.9 | 99.9 | 88.3 | −2.18 |
| DriveLaW | CVPR'26 | 1 Cam | 99.0 | 97.1 | 81.3 | 96.7 | 100.0 | 89.1 | −1.78 |
| *Textual-reasoning VLA* | | | | | | | | | |
| AutoVLA | NeurIPS'25 | 3 Cam | 98.4 | 95.6 | 81.9 | 98.0 | 99.9 | 89.1 | −2.92 |
| ReCogDrive | ICLR'26 | 3 Cam | 98.2 | 97.8 | 83.5 | 95.2 | 99.8 | 89.6 | −2.12 |
| AdaThinkDrive | ICRA'26 | 1 Cam | 98.4 | 97.8 | 84.4 | 95.2 | 100.0 | 90.3 | −2.24 |
| AutoVLA † | | 3 Cam | 99.1 | 97.1 | 87.6 | 97.1 | 100.0 | 92.1 | −2.01 |
| AdaThinkDrive † | | 1 Cam | 99.1 | 98.8 | 87.9 | 97.2 | 100.0 | 93.0 | −1.17 |
| *World-modeling VLA* | | | | | | | | | |
| DrivingGPT | ICCV'25 | 1 Cam | 98.9 | 90.7 | 79.7 | 94.9 | 95.6 | 82.4 | −2.85 |
| FSDrive | NeurIPS'25 | 3 Cam | 98.2 | 93.8 | 80.1 | 93.3 | 99.9 | 85.1 | −3.21 |
| PWM | NeurIPS'25 | 1 Cam | 98.6 | 95.9 | 81.8 | 95.4 | 100.0 | 88.1 | −2.53 |
| DriveVLA-W0 | ICLR'26 | 1 Cam | 98.7 | 99.1 | 83.3 | 95.3 | 99.3 | 90.2 | −1.22 |
| ExploreVLA | ECCV'26 | 1 Cam | 98.8 | 98.4 | 83.5 | 96.5 | 99.9 | 90.4 | −1.30 |
| SGDrive | CVPR'26 | 1 Cam | 98.6 | 97.8 | 85.8 | 96.2 | 100.0 | 91.1 | −1.90 |
| DriveVLA-W0 † | | 1 Cam | 99.3 | 97.4 | 88.3 | 97.0 | 99.9 | 93.0 | −2.22 |
| ExploreVLA † | | 1 Cam | 99.4 | 98.9 | 88.3 | 98.3 | 99.7 | 93.7 | −0.93 |
| **RAF-VLA** | – | 1 Cam | 98.0 | 98.5 | 85.2 | 93.9 | 99.7 | **90.0** | −1.92 |
| **RAF-VLA †** | – | 1 Cam | 99.0 | 99.6 | 89.8 | 97.3 | 100.0 | **94.2** | −0.90 |

Every row passes. ReCogDrive is the 89.6 arXiv row, labelled with the ICLR'26 venue whose camera-ready the wiki records at 90.8. RAF-VLA's TTC (93.9) is below every VLA row except FSDrive (93.3); its strengths are EP and DAC.

### Table II: NAVSIM v2 navtest (RAF-VLA after Future-Aligned SFT only)

| Method | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS | Residual | Reads as |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TransFuser | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 | +1.28 | pre-fix |
| VADv2 | 97.3 | 91.7 | 98.2 | 99.9 | 77.6 | 92.7 | 66.0 | 100.0 | 97.4 | 76.6 | −1.20 | Hydra-MDP++ own formula |
| Hydra-MDP | 97.5 | 96.3 | 98.3 | 99.9 | 80.1 | 93.0 | 65.5 | 100.0 | 97.4 | 79.8 | +0.38 | Hydra-MDP++ own formula |
| Hydra-MDP++ | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 | +2.16 | pre-fix-like |
| ReCogDrive | 98.3 | 95.2 | 99.5 | 99.8 | 87.1 | 97.5 | 96.6 | 98.3 | 86.5 | 83.6 | +2.69 | pre-fix-like |
| DiffusionDrive | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 | +2.48 | pre-fix (corrected 88.2) |
| DriveVLA-W0 | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 | −1.31 | corrected-like |
| SGDrive | 98.6 | 94.3 | 99.5 | 99.9 | 86.0 | 97.9 | 96.1 | 98.3 | 85.9 | 86.2 | −0.70 | corrected-like |
| **RAF-VLA** | 97.5 | 98.2 | 99.4 | 99.8 | 86.9 | 97.0 | 92.7 | 98.0 | 65.7 | **86.2** | −1.17 | corrected-like |

The rows from ReCogDrive's table (VADv2 76.6, Hydra-MDP 79.8, with HC exactly 100) are the "third convention lineage" recorded on [[concepts/navsim-benchmark.md]]. TransFuser to DriveVLA-W0 is the shared baseline block.

### Table III: alignment with and without RFT

| ID | Alignment | SFT | RFT | NC | DAC | EP | TTC | C. | PDMS |
|---|:-:|:-:|:-:|---:|---:|---:|---:|---:|---:|
| 1 | | ✓ | | 97.86 | 98.58 | 81.01 | 93.03 | 99.52 | 87.92 |
| 2 | | ✓ | ✓ | 97.42 | 98.39 | 84.31 | 92.38 | 99.51 | 88.85 |
| 3 | ✓ | ✓ | | 98.04 | 98.71 | 82.99 | 93.60 | 99.79 | 89.13 |
| 4 | ✓ | ✓ | ✓ | 97.98 | 98.46 | 85.21 | 93.86 | 99.73 | **90.03** |

Alignment raises EP most (+1.98 SFT, +0.90 RFT), and in RFT it also stops NC and TTC from falling (row 2 vs 1: NC −0.44, TTC −0.65; row 4 vs 3: −0.06, +0.26).

### Table IV: latency (navtest, 4 s trajectories, H200, batch 1)

| Alignment | Latency | Δ |
|---|---:|---:|
| ✗ | 0.152 s | – |
| ✓ | 0.153 s | +0.001 s |

### Table V: which horizons (SFT, navtest)

| ID | 1 s | 2 s | 3 s | 4 s | PDMS |
|---|:-:|:-:|:-:|:-:|---:|
| 1 | | | | | 87.92 |
| 2 | ✓ | | | | 88.93 |
| 3 | | | | ✓ | 88.97 |
| 4 | ✓ | ✓ | | | 88.60 |
| 5 | ✓ | | | ✓ | **89.13** |
| 6 | ✓ | ✓ | ✓ | ✓ | 88.62 |

### Table VI: world queries per horizon (SFT, navtest)

| $K$ | 2 | **4** | 8 | 16 |
|---|---:|---:|---:|---:|
| PDMS | 88.76 | **89.13** | 88.64 | 88.65 |

Sub-scores for Tables V and VI are in the clipping; every row passes the closed-form check (−1.6 to −2.1). The text calls both tables "Table VI".

---

## Qualitative

![[raf_vla_fig3.png|Fig. 3: Qualitative comparison of trajectory predictions on the NAVSIM navtest split. From left to right: front-camera input, vanilla SFT without alignment, Future-Aligned SFT, and Future-Aligned SFT followed by GRPO-based RFT (black: GT, red/orange/green: prediction from each model). Cropped from page 6 of the arXiv PDF; neither the clipping nor the arXiv HTML has the image.]]

**Fig. 3** *(cropped from the arXiv PDF)*: Two navtest scenes. Top, a left turn at a junction: the unaligned SFT model turns right against the command; the aligned model follows the left turn, and RFT tightens it onto the human path. Bottom, a van stopped ahead in the lane: the unaligned model's plan is short of the human's; the aligned plans extend toward it. Two chosen scenes.

---

## Table Audit {#table-audit}

1. **All 34 v1 rows pass the closed-form check** (Tables I, III, V, VI; −0.90 to −4.12).
2. **The v2 table mixes three conventions** (see Table II). The claim "matching the best reported SFT result" compares RAF-VLA's corrected-like 86.2 with pre-fix rows and Hydra-MDP++-formula rows. Whether DriveVLA-W0's 86.1 and ReCogDrive's 83.6 are SFT results is not stated.
3. **Comparison scope (v1)**: single-sample entries above 90.0 that predate it and are missing include DriveFine 90.7, WAM-Diff 91.0, DiffusionDriveV2 91.2, FLARE 91.4, LWDrive 92.0, WCog-VLA 92.9 and CLEAR 93.7. FLARE, the nearest method, is not cited.
4. **Best-of-6**: "the highest overall performance" holds within its table only.
5. **Samples seen** (Figure 1b) are the authors' estimates; the paper gives no per-method numbers.

---

## Limitations

The paper's own: alternative target encoders and projector designs are untested, and so are multi-view inputs and richer modalities.

The wiki adds:

1. **Extra tokens and alignment are confounded.** The baseline removes the world queries, so +1.2 prices eight extra tokens that the action expert reads plus the regression loss. A queries-without-loss arm is missing.
2. **No content control.** There is no current-frame target, which [[sources/world4scorer.md]] found to be the decisive control (+0.95 for a +2 s target, −0.31 for the current frame), and no comparison of Cosmos against other encoders.
3. **The target is unspecified.** "A pretrained encoder from the Cosmos world model" is probably a reconstruction-trained video tokenizer. Its output shape, and whether it is pooled to match four query states, are not given. That matters because [[sources/flare.md]] found spatial targets (+3.5) better than global ones (+2.5).
4. **Hyperparameters chosen on navtest, single runs.** Horizon and query count are the test-split maxima, 0.3–0.5 above neighbouring settings.
5. **Front camera only**; NAVSIM's eight-camera rig is unused.
6. **The RFT sampler is undescribed** for an anchor head, and so is whether $\mathcal L_\text{act}$ includes Hydra-MDP-style metric targets. DriveVLA-W0's 90.2★ does ([[concepts/selection-based-planning.md]]).
7. **v2 is SFT only** and EC is low (65.7). No navhard, no closed loop.

---

## Relation to Other Wiki Pages

- [[sources/drivewla.md]] *(2026-10-19)*: the opposite design a month later. DriveWLA keeps a full video DiT and trains it with no future loss; RAF-VLA drops the video model and keeps a future-representation loss.
- [[sources/drivevla-w0.md]]: the architecture (MoT with an action expert, query-based anchor head, $\mathcal L_\text{act}$) and the main comparison. RAF-VLA matches its 90.2 with a future-representation loss in place of image generation, on a different VLM.
- [[sources/flare.md]]: the nearest method, uncited. It also adds future-feature regression to a front-camera VLA (DINOv2 patches, action-conditioned, L1 + cosine) followed by GRPO. Its SFT effect is +3.5, against +1.2 here, and 91.4 after RFT, against 90.0.
- [[sources/hyworldvla.md]]: also has a VLM predict a future latent through a query token for the action expert to read (a video-VAE latent, MSE; reading it is worth +0.30).
- [[sources/lvdrive.md]], [[sources/deepsight.md]]: Bench2Drive VLAs that predict future latents in one forward pass. DeepSight even uses the same name, "World Queries". LVDrive's best target is a reconstruction codebook (VQGAN) and DeepSight's DINOv3. RAF-VLA's Cosmos encoder is probably a third reconstruction-trained target, and it helps.
- [[sources/world4scorer.md]], [[sources/foredrive.md]]: future-feature auxiliaries on DrivoR and DiffusionDrive-style planners (+0.95 for a +2 s target, +0.1 to +0.6 for ForeDrive's). World4Scorer has the current-frame control RAF-VLA lacks.
- [[sources/reworld.md]], [[sources/walt.md]]: REPA-style alignment of a planner's internal stream to frozen world-model features, +0.4 and +0.35 ([[concepts/action-tokenization.md#walt]]). RAF-VLA's aligned states are future slots rather than the action stream itself.
- [[sources/simvta.md]]: the RL-time counterpart. Keeping a video loss during RL adds +2.2 over RL alone, whereas RAF-VLA drops its alignment loss for RL and keeps the gain anyway.
- [[sources/explorevla.md]], [[sources/sgdrive.md]], [[sources/futuresightdrive.md]], [[sources/policy-world-model.md]]: the world-modeling VLAs it contrasts with. [[sources/recogdrive.md]], [[sources/adathinkdrive.md]], [[sources/autovla.md]]: the textual-reasoning group. [[sources/nord.md]]: cited for annotation cost.
- Not ingested: **REPA / REPA-E / iREPA / ARRA** (representation alignment in generative models), **Cosmos** (the target encoder), **DrivingGPT**, **WoTE**.

---

## Open Questions

- **Is it the loss or the tokens?** Keep the world queries and drop the alignment loss, and mask the action expert from the queries at inference.
- **Is the target a future?** Align to the current frame's Cosmos representation, or to a shuffled future.
- **Spatial or pooled targets?** FLARE's spatial-over-global result suggests a patch-level target could be worth more than four pooled query states.
- **Does the gain hold on navhard and v2 after RFT?** Only v1 is reported after RL.
