---
title: "SUV: Future Scene Understanding as Video Generation for End-to-End Driving"
type: source-summary
sources: ["raw/papers/SUV_ Future Scene Understanding as Video Generation for End-to-End Driving.md"]
related: [sources/metis.md, concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/navhard-ood-evaluation.md, concepts/nuscenes-waymo-evals.md, concepts/perception-for-planning.md, concepts/foundation-backbones-for-ad.md, concepts/mixture-of-experts.md, concepts/diffusion-planner.md, sources/simwam.md, sources/driveva.md, sources/drivewam.md, sources/coworld-vla.md, sources/drivelaw.md, sources/wa-jepa.md, sources/drivefuture.md, sources/geowam.md, sources/spanvla.md, sources/drivevla-w0.md, sources/explorevla.md, sources/adaptive-wam.md, sources/qwen-drive-1.0.md, sources/dial.md, sources/grava.md, sources/physwam.md]
created: 2026-09-27
updated: 2026-09-30
confidence: medium
---

**Paper**: SUV: Future Scene Understanding as Video Generation for End-to-End Driving
**Authors**: Yibo Yuan, Jiacheng Fu, Jiangtong Zhu, Yi Li, Jianhua Han, Meng Tian, Zhuohan Liu, Zhiwei Xiong, Hang Xu, Jianwu Fang, Jianru Xue
**Orgs**: Xi'an Jiaotong University, University of Science and Technology of China, Yinwang Intelligent Technology Co., Ltd., Fudan University
**arXiv**: 2608.03084v1
**Code**: announced at `github.com/ASH-2046/SUV`

---

## Source Integrity Note

The clipping is complete: main text, references, and a four-section supplement. All eight main tables and all seven supplementary tables are present.

**Figures**:
- Present: main Figs. 1–3 and supplementary Figs. 2–3.
- **Supplementary Fig. 1** (zero-shot in-house transfer) links to an external arXiv URL and was not downloaded.
- **Supplementary Figs. 4–5** (horizon-wise future-scene curves) are captioned with no image. The prose states their orderings, but no numbers.

**Misnumbered cross-references**:
- The prose cites the navtest table (Table 1) as "Table 2".
- It cites the 2×2 supervision/access ablation (Table 6) as "Table 7".

**One scrambled baseline row.** In supplementary Table 3, Drive-JEPA's row reads NC 98.7 / DAC 96.2 / TTC 100 / Comf 95.5, which looks like a TTC/Comf column swap.

The NAVSIM-v1 table's column headers were lost in the HTML conversion. They are recovered as NC / DAC / TTC / Comf / EP / PDMS, which matches the Human row (100 / 100 / 100 / 99.9 / 87.5 / 94.8) and TransFuser's canonical values.

---

## Summary

SUV turns a pretrained video generator (**Wan2.2-5B**) into a driving policy by making it generate **four future video streams at once**, all over the same 4 s horizon, camera and pixel grid:
- **RGB**
- **Semantic segmentation**, rendered with a fixed color palette
- **Relative depth**, rendered through the Turbo colormap
- **Instance tracks**, rendered as class color × brightness-encoded ID

The three structured streams are produced offline by frozen **SAM 3** and **Depth Anything 3** from the recorded future frames. They are then encoded by the same frozen video VAE, so a single shared video expert produces all four with **no stream-specific heads**. Stream-specific text prompts tell the expert which target it is generating.

A separate **~1B action expert** (Mixture-of-Transformers: its own weights, shared attention space) denoises 8 waypoints $(x,y,\psi)$ by flow matching, jointly with the video streams. A **directed attention mask** controls what each token group can see:
- Observation tokens see only themselves.
- Each future stream sees the observation and itself, but **not the other streams**.
- **Action tokens see everything.**

The action therefore reads the evolving future-stream latents at every block and every solver step. Nothing is decoded to pixels at inference.

**Headline results** (single front camera, one trajectory, no scorer, no RL):
- **91.0 EPDMS** on NAVSIM-v2 navtest (corrected evaluator).
- **36.9 EPDMS** on navhard.
- **7.94 RFS** on WOD-E2E test.
- **90.8 PDMS** on NAVSIM-v1.

The model is about 6B parameters, trained for 60 epochs on 8×H200. At 2 solver steps it runs at 288 ms on an RTX 4090 with the same navtest score. At 10 steps it takes 1,356 ms.

**The paper's most useful result is not the headline.** It is the **2×2 ablation that separates the training signal (structured supervision) from the inference path (future access)**. Future access is worth **+0.3 on navtest but +4.1 on navhard**. [[sources/simwam.md]] found that future access gives no navtest gain on the same backbone. SUV's result suggests that finding was a property of navtest, not of the mechanism. See [below](#access-navhard).

---

## Positioning

- **Its closest relatives use the same backbone, and none of them is cited.**
  - [[sources/simwam.md]] (91.5 PDMS v1): Wan2.2-5B video expert plus a separate action DiT, with future access *removed* by an isolated mask.
  - [[sources/driveva.md]] (90.9): Wan2.2-TI2V-5B, one DiT jointly denoising video and action.
  - [[sources/drivewam.md]] (90.1): Wan2.2-TI2V-5B, chunked autoregressive rollout with an inverse-dynamics action readout.
  - **On NAVSIM-v1, SUV's 90.8 is below both SimWAM and DriveVA.** Its v1 table stops at ReCogDrive 90.8 and cites DriveVLA-W0 at 88.4 (its reimplementation-style row), not the 90.2 headline.
  - SUV does cite Fast-WAM, the intellectual antecedent of SimWAM's no-access argument.
- **Its conceptual sibling is [[sources/coworld-vla.md]].** CoWorld-VLA also uses four future targets in one latent space, including a Wan2.2-5B video branch, but it discards the video model before planning and supervises typed VLM tokens with teacher features. SUV instead keeps the generator at inference and renders the teachers' outputs *as videos*. The two are opposite answers to "multiple future targets, one model".
- **"Generation as a universal interface for perception"** (Vision Banana, SenseNova-Vision, GenCeption), applied to *future* perception and then read by a planner. This is new in the wiki: every earlier multi-target world model used dedicated heads or feature-regression readouts.

---

## Method

### Framing

![[teaser 2.png|Three paradigms for future scene understanding: (a) multi-task frameworks with a shared representation and task-specific outputs; (b) language-based VLAs serializing scene and trajectory as tokens; (c) SUV, one shared video expert generating four future streams with the action expert attending to their latent tokens]]

*Figure 1: (a) Multi-task methods share a scene representation but retain task-specific outputs. (b) Language-based methods serialize scene information and trajectories as tokens. (c) SUV uses one shared video expert to generate four future streams and lets the action expert attend to their latent tokens for planning.*

**Input**: $K=4$ front-camera frames at 2 Hz (2 s of history) at 640×384, plus a navigation command and ego state.

**Output**: $\mathbf a_t=\{(x,y,\psi)_{t+h}\}_{h=1}^{8}$ and four $T=8$-frame future videos $\mathbf V_t$, modeled jointly as

$$p_\theta(\mathbf V_t,\mathbf a_t\mid\mathbf o_t,\mathbf c_t).$$

### Architecture and the token-interaction mask

![[pipeline 1.png|SUV architecture: input RGB video and four future streams (RGB, segmentation, depth, instance tracks) pass through a frozen video encoder into a Wan2.2-initialized video expert; noisy trajectory tokens pass through an action encoder into a separate action expert; both share joint video-action attention; command and ego status go through a prompt generator and frozen text encoder into cross-attention. Right: token-interaction mask in which Obs attends only to Obs, each future stream attends to Obs and itself, and Action attends to all groups]]

*Figure 2: Left: the shared video expert generates four future streams while the action expert denoises the trajectory using their latent tokens. Right: the token-interaction mask preserves observation-to-stream pathways, blocks cross-stream interaction and action-to-future feedback, and lets action queries read every token group.*

The visible key groups for each query group $g$ are:

$$\mathcal A(g)=\begin{cases}\{\mathrm{obs}\}&g=\mathrm{obs}\\ \{\mathrm{obs},m\}&g=m\in\{\text{rgb, seg, depth, track}\}\\ \{\mathrm{obs},\mathrm{act}\}\cup\mathcal M&g=\mathrm{act}\end{cases}$$

**Why cross-stream attention is blocked.** Cross-stream attention "would introduce interactions absent from video pretraining". Each stream therefore behaves like an ordinary Wan image-to-video generation that shares the same prefix. Consistency between streams comes only from sharing the prefix, the grid, the timestamps and the weights. Fig. 3 argues it holds qualitatively. **Cross-stream consistency is never measured.**

**Expert sizes.** Both experts have 30 blocks and 24 heads × 128 dimensions, giving a 3,072-dimensional shared attention space.
- Video expert: hidden size 3,072, FFN 14,336.
- Action expert: hidden size 1,024, FFN 4,096, projected up to 3,072 for attention.

This is the same MoT layout as SimWAM and [[sources/adaptive-wam.md]]'s family (see [[concepts/mixture-of-experts.md]]).

**Conditioning.** Both experts cross-attend to a common prompt (command plus ego state). Fig. 2 shows this as a text string, e.g. "Turn left. 1.22 m/s · a>0 · yaw=0°". Each stream also gets a modality prompt. The text encoder and VAE are frozen; both experts and the ego-state projection are trained.

### Structured targets as video

The targets are built as follows (Supplement §4.1):

| Stream | Teacher | Encoding |
|---|---|---|
| Segmentation | SAM 3, 11 text prompts, confidence 0.5, ≤32 detections per prompt, ≤128 per frame; road rendered as a fill layer | Fixed 12-color palette; decoded by nearest color |
| Relative depth | DA3-LARGE on all $T$ frames jointly | Per-clip 1st/99th-percentile normalization → 256-entry Turbo LUT; decoded by nearest LUT entry |
| Instance tracks | SAM 3 video session, 6 class prompts propagated from the first future frame | Class base color × 7 brightness offsets (ID mod 7); decoded by color matching, connected components, and Hungarian linking across adjacent frames |

**Two design consequences:**
1. **Depth is clip-relative.** Normalization is per clip, so the depth stream encodes ordering and relative shape, not metric distance.
2. **Track identity is modulo 7 within a class.** An eighth car of the same class reuses the first car's color, and the decoder can link them only by IoU and centroid distance.

Neither consequence is discussed as a limitation.

### Training and inference

**Training loss**: $\mathcal L=\tfrac14\sum_{m}\mathcal L^m_{\mathrm{FM}}+\mathcal L^{\mathrm{act}}_{\mathrm{FM}}$. This is rectified-flow MSE with a shifted time schedule ($\kappa=5$) and a bell-shaped weight.
- The video streams share one flow time; the action gets an **independent** time. As with SimWAM and DriveVA, this lets the action be trained against video latents at any noise level.

**Optimization**:
- AdamW, learning rate 1e-4, weight decay 0.01, cosine schedule.
- 60 epochs, batch 8 per GPU on 8×H200.
- **Checkpoints selected on the NAVSIM validation PDM score.**

**Inference**: video and action flow times are synchronized on a shared grid from 1 to 0 (10 Euler steps by default). The action reads the current partially denoised future latents at every step, and the VAE decoder is used only for visualization and evaluation.

---

## Results

### Table 1 — NAVSIM-v2 navtest (the paper states the corrected official EPDMS)

| Method | Sensors | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Human Agent | – | 100 | 100 | 99.8 | 100 | 87.4 | 100 | 100 | 98.1 | 90.1 | 94.5 |
| *Traditional E2E* | | | | | | | | | | | |
| DiffusionDriveV2 | 3×C+L | 97.7 | 96.6 | 99.2 | 99.8 | 88.9 | 97.2 | 96.0 | 97.8 | 91.0 | 87.5 |
| DriveSuprim | 3×C | 98.4 | 98.6 | 99.6 | 99.8 | 90.5 | 97.8 | 97.0 | 98.3 | 78.6 | 87.1 |
| SparseDriveV2 | 3×C | 98.1 | 98.1 | 99.6 | 99.8 | 91.1 | 97.3 | 96.9 | 98.2 | 78.4 | 90.1 |
| *VLA-based* | | | | | | | | | | | |
| DriveVLA-W0 | 1×C | 98.4 | 95.2 | 99.4 | 99.9 | 86.6 | 97.9 | 97.8 | 98.3 | 82.7 | 86.9 |
| DriveWorld-VLA | 3×C | 98.6 | 99.1 | 99.6 | 99.8 | 87.4 | 97.9 | 97.0 | 97.8 | 78.6 | 86.8 |
| SGDrive | 1×C | 98.6 | 94.3 | 99.5 | 99.8 | 86.0 | 97.9 | 96.1 | 98.3 | 85.9 | 86.2 |
| DriveFine | 1×C | 98.7 | 97.3 | 99.5 | 99.8 | 88.7 | 97.8 | 97.7 | 98.4 | 83.8 | 89.7 |
| *WAM-based* | | | | | | | | | | | |
| EponaV2 | 1×C | 98.5 | 97.4 | 99.5 | 99.9 | 87.9 | 98.1 | 97.7 | 98.2 | 77.4 | 88.9 |
| Metis | 1×C | 98.4 | 97.2 | 99.6 | 99.8 | 87.8 | 97.7 | 97.8 | 98.4 | 88.0 | 89.5 |
| Metis (Top 6) | 1×C | 98.5 | 97.5 | 99.6 | 99.8 | 87.9 | 97.8 | 98.0 | 98.4 | 90.0 | 90.3 |
| **SUV** | 1×C | **99.1** | 97.8 | **99.7** | 99.8 | 87.8 | **98.7** | **98.1** | 98.4 | 88.3 | **91.0** |

**Protocol audit, using the wiki's [evaluator-drift partition](../concepts/navsim-benchmark.md#evaluator-drift-this-table-mixes-two-protocols).** The caption says "corrected", but the rows come from both conventions:

| Row | Value here | Wiki classification |
|---|---:|---|
| DiffusionDriveV2 | 87.5 | Corrected ✓ |
| SparseDriveV2 | 90.1 | Corrected ✓ |
| DriveFine | 89.7 | Corrected ✓ |
| DriveWorld-VLA | 86.8 | Corrected ✓ |
| EponaV2 | 88.9 | Matches GeoWAM/GeoWorldAD ✓ |
| **DriveSuprim** | **87.1** | **Pre-fix** |
| SGDrive | 86.2 | Unclassified (SGDrive's own number) |
| **DriveVLA-W0** | **86.9** | **Neither known value** (pre-fix 86.1); probably a reproduction, since the paper times DriveVLA-W0 itself in Table 8 |

**This is another v2 table that mixes conventions, at least the seventh ingested.** SUV's own 91.0 is stated to be corrected. Under that protocol it is **second in the wiki, behind [[sources/wa-jepa.md]] at 91.7**, which is absent from the table. *(Lint 2026-09-30: now third. [[sources/mm-future.md]] reports 91.5 on the corrected protocol, from a trainval model with 64 scored proposals.)* It is above SparseDriveV2 90.1, CoWorld-VLA 90.0, DriveFuture 89.9, WAM-Diff/DriveFine 89.7 and LWDrive 89.6. It is the best corrected-protocol result without a trajectory scorer or best-of-N; WA-JEPA also has neither, so WA-JEPA still leads.

**Where the score comes from.** NC 99.1, TTC 98.7 and LK 98.1 are the table's best. EP 87.8 is 3.3 below SparseDriveV2 and equal to the Human Agent's 87.4 (+0.4). SUV's profile is safety-heavy rather than progress-heavy, the pattern the wiki keeps recording for world-model planners.

**Submetric residual.** Computed from the mean submetrics, the aggregate is 90.5. The reported value is 91.0, so the residual is **−0.5** (computed − reported, the [NAVSIM page's convention](../concepts/navsim-benchmark.md#submetric-residual)). A small negative residual is the pattern of *corrected-protocol* rows, where known pre-fix rows sit at +1 to +4 ([residual-sign heuristic](../concepts/navsim-benchmark.md#residual-sign)). That is consistent with SUV's claim to use the corrected evaluator. [[sources/drive-hwm.md]]'s +5.25 remains the anomaly.

**The Metis comparison is probably like-for-like.** SUV copies [[sources/metis.md]]'s 89.5 / 90.3 rows verbatim. Metis's own table uses pre-fix baselines, but Metis's own row has a corrected-looking residual (−0.6), so the +1.5 margin over Metis is plausibly fair.

### Table 2 — NAVSIM-v2 navhard (two-stage)

| Method | Stage | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | S. | EPDMS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LTF | S1 | 96.2 | 79.6 | 99.1 | 99.6 | 84.1 | 95.1 | 94.2 | 97.6 | 79.1 | – | |
| | S2 | 77.8 | 70.2 | 84.3 | 98.1 | 85.1 | 85.1 | 45.4 | 95.7 | 76.0 | – | 25.1 |
| GuideFlow | S1 | 96.6 | 80.5 | 96.3 | 99.3 | 82.3 | 94.9 | 91.5 | 97.7 | 67.8 | – | |
| | S2 | 87.3 | 76.7 | 88.8 | 99.2 | 84.3 | 85.1 | 49.7 | 93.1 | 44.5 | – | 27.1 |
| DriveVLA-W0 | S1 | 96.8 | 83.3 | 99.0 | 99.6 | 84.6 | 95.3 | 96.4 | 97.6 | 78.2 | – | |
| | S2 | 76.8 | 64.3 | 79.9 | 98.3 | 89.2 | 75.0 | 46.8 | 95.8 | 53.1 | – | 24.4 |
| ReCogDrive | S1 | 96.4 | 78.9 | 98.7 | 99.8 | 82.6 | 95.6 | 94.4 | 97.6 | 74.2 | 67.7 | |
| | S2 | 80.2 | 65.0 | 82.4 | 98.7 | 85.2 | 76.9 | 43.8 | 96.6 | 71.8 | 37.6 | 25.7 |
| SGDrive | S1 | 95.8 | 87.6 | 97.8 | 99.8 | 84.4 | 94.7 | 92.9 | 97.8 | 28.9 | 71.1 | |
| | S2 | 79.4 | 65.4 | 79.1 | 98.9 | 88.9 | 75.3 | 42.7 | 96.4 | 29.6 | 35.2 | 25.5 |
| EponaV2 | S1 | 97.3 | 90.7 | 99.4 | 100 | 83.3 | 97.3 | 97.3 | 97.6 | 60.9 | – | |
| | S2 | 83.6 | 78.0 | 88.0 | 98.9 | 86.0 | 80.3 | 50.1 | 96.1 | 52.0 | – | 36.1 |
| Metis | S1 | 96.6 | 87.8 | 99.0 | 99.3 | 84.5 | 95.6 | 97.8 | 97.8 | 77.8 | 75.8 | |
| | S2 | 79.6 | 73.3 | 84.9 | 97.8 | 85.8 | 76.6 | 47.7 | 95.4 | 75.3 | 41.7 | 32.2 |
| **SUV** | S1 | 96.9 | **94.2** | 99.3 | 99.6 | 84.1 | 95.6 | 96.7 | 97.8 | 79.6 | **82.3** | |
| | S2 | 82.7 | 74.2 | 85.9 | 98.4 | 86.0 | 78.9 | 47.2 | 95.9 | 69.1 | **43.9** | **36.9** |

**Where 36.9 ranks.** Within the wiki's [unscored cohort](../concepts/navhard-ood-evaluation.md#scorer-cohort), 36.9 is **second, behind [[sources/spanvla.md]] at 40.1**. It is just above [[sources/geowam.md]] at 36.6 and EponaV2 at 36.1. EponaV2's 36.1 and DriveVLA-W0's 24.4 match GeoWAM's table, which confirms those rows. The five scorer-equipped methods at 42–55.5 are all higher and all absent here.

**The Stage-1 lead is DAC.** DAC is 94.2 against a next-best of 90.7. Stage-2 lane keeping is 47.2, inside the 45–50 band that every unscored method falls into ([Stage 2 collapse](../concepts/navhard-ood-evaluation.md#lk-correction)).

### Table 3 — WOD-E2E test

| Method | ADE 3/5 s ↓ | RFS ↑ |
|---|---|---:|
| AutoVLA | 1.35 / 2.96 | 7.56 |
| HMVLM | 1.33 / 3.07 | 7.74 |
| Fast-dDrive | 1.25 / 2.91 | 7.82 |
| IRL-VLA | 1.22 / 2.82 | 7.89 |
| Poutine-Base | 1.27 / 2.94 | 7.91 |
| **SUV** | 1.24 / 2.90 | **7.94** |

The margin over Poutine-Base is +0.03 RFS. [[sources/qwen-drive-1.0.md]] reports **7.91** and cites MindVLA-U1 at **7.87** with RL. **Both are absent here.** On the wiki's [WOD-E2E table](../concepts/nuscenes-waymo-evals.md), 7.94 is a competitive test score, not a separable lead.

The WOD-E2E model is trained separately (5 frames of history, 10 future frames, 20 waypoints at 4 Hz over 5 s). Evaluation uses the step-100,000 checkpoint on all 1,505 test frames.

### Supplement Table 3 — NAVSIM-v1 navtest (PDMS)

| Method | NC | DAC | TTC | Comf | EP | PDMS |
|---|---:|---:|---:|---:|---:|---:|
| Human | 100 | 100 | 100 | 99.9 | 87.5 | 94.8 |
| TransFuser | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| DiffusionDrive | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| World4Drive | 97.4 | 94.3 | 92.8 | 100 | 79.9 | 85.1 |
| Hydra-MDP++ | 97.6 | 96.0 | 93.1 | 100 | 80.4 | 86.6 |
| SeerDrive | 98.4 | 97.0 | 94.9 | 99.9 | 83.2 | 88.9 |
| Drive-JEPA ⚠ | 98.7 | 96.2 | 100 | 95.5 | 82.9 | 89.0 |
| AutoVLA | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 |
| DriveVLA-W0 | 98.7 | 96.2 | 95.5 | 100 | 82.2 | 88.4 |
| SGDrive-IL | 98.6 | 95.1 | 95.4 | 100 | 81.2 | 87.4 |
| AutoDrive-P³ | 99.1 | 97.4 | 96.5 | 100 | 84.8 | 90.6 |
| ReCogDrive | 97.9 | 97.3 | 94.9 | 100 | 87.3 | 90.8 |
| ImagiDrive | 98.6 | 96.2 | 94.5 | 100 | 80.5 | 87.4 |
| PWM | 98.6 | 95.9 | 95.4 | 100 | 81.8 | 88.1 |
| DriveLaW | 99.0 | 97.1 | 96.7 | 100 | 81.3 | 89.1 |
| EponaV2 | 98.6 | 97.9 | 95.7 | 100 | 84.8 | 90.4 |
| Metis | 98.3 | 97.1 | 94.7 | 100 | 83.4 | 89.1 |
| Metis (Top 6) | 98.5 | 97.5 | 95.1 | 100 | 84.0 | 89.7 |
| **SUV** | **99.1** | 97.8 | 96.7 | 100 | 84.6 | **90.8** |

⚠ The Drive-JEPA row's TTC and Comf values appear swapped. The swap is inherited: [[sources/metis.md]]'s Table 9 carries the identical row. Its 89.0 is Drive-JEPA's perception-free baseline, not its 93.3 planner. DriveVLA-W0 appears at 88.4, not its 90.2 headline. SGDrive appears as its IL-only 87.4. SUV's claim to tie ReCogDrive for the lead holds only in this table; the wiki's v1 ladder has about twenty entries above 90.8.

### Reproducibility (Supplement §2.2)

Six repeated evaluations of the fixed checkpoint returned **90.84 PDMS and 91.01 EPDMS** every time. Inference is deterministic, so this is not a variance estimate.

The scene-level SD is **17.84 (v1)** and **18.21 (v2)** over 12,146 scenes, which gives a standard error of the mean of about **0.16**. That bounds *evaluation* noise only. As [[sources/coworld-vla.md]] notes, nobody has measured training-seed variance for a NAVSIM planner.

---

## Future-Scene Prediction

**Multi-stream training does not hurt RGB.** The mean PSNR / SSIM is 19.55 / 0.5844, against 19.44 / 0.5820 for RGB-only training. The multi-stream model is higher at every horizon, but the authors say correctly that this does not establish an improvement.

**Table 4 — initialization** (matched settings; means over the horizons):

| Init. | PSNR ↑ | SSIM ↑ | mIoU ↑ | AbsRel ↓ | AssA@50 ↑ |
|---|---:|---:|---:|---:|---:|
| Random | 18.51 | 0.5432 | 49.7 | 26.8 | 81.4 |
| Wan2.2-5B | 19.55 | 0.5844 | 64.2 | 19.5 | 84.4 |

The video prior helps most on segmentation (+14.5 mIoU) and depth (−7.3 AbsRel). **No planning score is reported for random initialization.** The wiki's recurring "does the video prior matter for *planning*?" question therefore remains open for this paper. DriveWAM's result on the same backbone is the relevant data point.

**Table 5 — native generation vs. Generate-then-Perceive** (the frozen teachers applied to generated RGB):

| Route | mIoU ↑ | δ₁ ↑ | AbsRel ↓ | AssA@50 ↑ |
|---|---:|---:|---:|---:|
| Generate-then-Perceive | **66.81** | 70.99 | 22.13 | **86.88** |
| Native generation (SUV) | 64.22 | **77.04** | **19.49** | 84.39 |

**Native generation wins on depth and loses on segmentation and tracking.** The paper summarizes this as "competitive", which is fair. The ordering holds at every horizon (the figure is missing; this is per the prose). The reason native generation is still worth having is that the planner reads the latents directly, in one pass, with no teacher at inference. Planning with Generate-then-Perceive outputs is never tried.

**Every structured metric measures agreement with SAM 3 and DA3, not ground truth.** The paper says so clearly. Depth is also scored after a per-clip affine fit.

![[vis 1.png|Lane-change scenario: past video and BEV planning (GT green, SUV red, EPDMS 100) alongside jointly generated RGB, segmentation, depth and instance-track futures at 0.5, 1.0, 2.0, 3.0 and 4.0 s]]

*Figure 3: Qualitative multi-stream prediction and planning in a lane-change scenario. Left: past video and planning. Right: the generated future streams keep clear visual content, road layout, depth maps and consistent vehicle IDs across 4 seconds.*

---

## Planning Ablations

### Table 6 — structured supervision × future access {#access-navhard}

All rows are supervised on the RGB future. "S/G/I" adds segmentation, depth and track supervision. "Access" lets the action expert read the future streams.

| S/G/I | Access | navtest EPDMS | navhard S1 | navhard S2 | navhard EPDMS |
|---|---|---:|---:|---:|---:|
| ✗ | ✗ | 89.7 | 77.0 | 39.9 | 30.5 |
| ✗ | ✓ | 90.6 | 80.0 | 43.8 | 35.0 |
| ✓ | ✗ | 90.7 | 79.8 | 41.5 | 32.8 |
| ✓ | ✓ | **91.0** | **82.3** | **43.9** | **36.9** |

Read off the table:

| Effect | navtest | navhard |
|---|---:|---:|
| Access, RGB-only supervision | +0.9 | **+4.5** |
| Access, with S/G/I | +0.3 | **+4.1** |
| S/G/I, no access | +1.0 | +2.3 |
| S/G/I, with access | +0.4 | +1.9 |

**What this says about [[sources/simwam.md]]'s null.** SimWAM's mask ablation (same Wan2.2-5B MoT family) found future access worth 0.0 ± 0.2 PDMS on navtest. It is the wiki's main evidence that test-time imagination is unnecessary. **SUV reproduces that null on navtest (+0.3) and finds a large effect on navhard (+4.1, with both stages rising by about 2.5).** navhard is where the observation departs from the logged one, and that is exactly where reading an imagined future should help. So the null result is better read as **"navtest cannot detect it"** than as "it does not exist".

Caveats:
- Single runs.
- navhard has only 450 Stage-1 scenes, so its noise floor is much higher than navtest's. The +4.1 is large but has no error bar.
- SUV's access condition reads *four* streams, SimWAM's reads only RGB. However, SUV's RGB-only row already shows +4.5, so the extra streams are not what produces the navhard effect.

**The baseline is already strong.** The no-S/G/I, no-access row (a Wan2.2-5B WAM trained with an RGB future objective, with the future discarded at inference) scores **89.7 EPDMS**. That alone is above every VLA row in Table 1. Most of SUV's number is the video-prior WAM recipe. The paper's two contributions add +1.3 on navtest and +6.4 on navhard.

### Table 7 — removing access to one structured stream

All training objectives and RGB access are kept.

| Variant | navtest | navhard S1 | S2 | navhard EPDMS |
|---|---:|---:|---:|---:|
| Full | 91.0 | 82.3 | 43.9 | 36.9 |
| w/o Seg. access | 90.9 | 82.6 | 43.2 | 36.4 |
| w/o Depth access | 90.8 | 81.2 | 43.9 | 35.8 |
| w/o Track access | 90.9 | 82.0 | 42.6 | 35.4 |

- On navtest, every removal is within 0.2.
- On navhard:
  - Removing track access costs the most overall (−1.5), mostly in Stage 2.
  - Removing depth access costs −1.1, all of it in Stage 1.
  - Removing segmentation access costs −0.5.
- Instance dynamics matter most under observation shift. This is plausible but unreplicated.

### Table 8 — accuracy vs. latency (RTX 4090, mean of 500 runs)

| Model | Steps | navtest | navhard | ms | Hz |
|---|---:|---:|---:|---:|---:|
| DriveVLA-W0 | – | 86.9 | 24.4 | 690 | 1.45 |
| SUV | 1 | 89.8 | 33.0 | 177 | 5.65 |
| SUV | 2 | **91.0** | 36.1 | 288 | 3.48 |
| SUV | 10 | 91.0 | **36.9** | 1,356 | 0.74 |

- **Two steps reach the navtest headline at 288 ms.** Eight more steps add only 0.8 on navhard, at 4.7× the latency.
- One step still scores 89.8, above every VLA in Table 1.
- For comparison: [[sources/adaptive-wam.md]] reads the same backbone family in one forward pass at 170 ms, and [[sources/drivelaw.md]] found early-step latents better than late ones.
- **The paper does not report the no-access variant's latency.** That variant needs no video denoising at inference. It is the obvious deployment baseline, and its 90.7 navtest is 0.3 below SUV at what is presumably a fraction of the cost.

---

## Qualitative Results

![[vis_supp_success.png|Three high-curvature turns on NAVSIM-v2 navtest (wide signalized intersection, curved urban road with nearby traffic, wet-road turn), each with current observation, BEV trajectories (human green, SUV red) and jointly generated RGB, segmentation, relative depth and instance tracks at five horizons]]

*Supplementary Figure 2: High-curvature turns on NAVSIM-v2 navtest, each with the current observation, BEV trajectories, and jointly generated RGB, segmentation, relative depth and instance tracks at five horizons.*

![[vis_supp_conservative.png|Dense urban traffic case: SUV follows the recorded human route but advances less over 4 s, EPDMS 84.1, with jointly generated future streams]]

*Supplementary Figure 3: Lower progress in dense urban traffic. The SUV trajectory follows the recorded human route but advances less over the 4-s horizon (EPDMS 84.1).*

The one failure case the paper shows is **under-progress**, consistent with EP being SUV's weakest navtest submetric. Supplementary Figure 1 (zero-shot on two in-house clips, ADE 0.08 m and 0.15 m) is an external image and not in the clipping. Two hand-picked clips are an anecdote, not evidence of transfer.

---

## Limitations

1. **The same-backbone predecessors are missing.** SimWAM (91.5 PDMS), DriveVA (90.9) and DriveWAM (90.1) all turn Wan2.2(-TI2V)-5B into a driving policy, and none is cited or compared. On NAVSIM-v1, SUV's 90.8 is below the first two.
2. **The v2 table mixes evaluator conventions** despite its "corrected" caption: DriveSuprim is at the pre-fix 87.1, and DriveVLA-W0's 86.9 matches no known value. **WA-JEPA (91.7, corrected) is omitted and beats SUV.**
3. **The navhard claim ("outperforms a broad set") omits SpanVLA (40.1, unscored) and the whole scorer cohort (42–55.5).**
4. **The WOD-E2E margin is +0.03 RFS**, and the table omits Qwen-Drive-1.0 (7.91) and MindVLA-U1 (7.87).
5. **The v1 table carries weaker variants of three baselines** (DriveVLA-W0 88.4, Drive-JEPA 89.0, SGDrive-IL 87.4) and has one row with apparently swapped columns.
6. **All structured-future metrics measure agreement with the teachers**, and depth is clip-relative and affine-aligned. Nothing is checked against LiDAR depth or annotated tracks, although NAVSIM has both.
7. **Cross-stream consistency is claimed from one qualitative figure and never measured**, even though the mask deliberately blocks cross-stream attention.
8. **Encoding limits go unmentioned.** Track IDs are encoded modulo 7 per class, and depth is normalized per clip. Both limit what the planner can extract from the streams.
9. **Single runs.** The deterministic re-evaluation is not training variance. The navhard access effect (+4.1) rests on 450 Stage-1 scenes with no error bar.
10. **No planning result for random video initialization**, so the value of the video prior *for planning* is not measured here.
11. **No latency for the no-access variant**, which is the natural cheap baseline (90.7 navtest, 32.8 navhard).
12. **Checkpoints are selected on the NAVSIM validation PDM score.** This is standard practice, but it is a mild selection effect on a 0.3–0.7-point headline margin.
13. **Scene coverage is one front camera at 640×384**, so there is no rear or side awareness for lane changes. This is the same constraint as most 1×C entries.

---

## Key Cross-References

- [[concepts/world-model-for-ad.md]] — Pattern 36: **perception tasks as extra generated video streams**. Also [the test-time imagination dispute](../concepts/world-model-for-ad.md#test-time-imagination): the first controlled access ablation that reports navhard, where access is worth +4.1 against +0.3 on navtest.
- [[concepts/navsim-benchmark.md]] — 91.0 on the corrected v2 cohort (second when ingested, behind WA-JEPA 91.7; third since MM-Future's 91.5); another mixed-convention v2 table; 90.8 on v1; the 2-step/288 ms operating point.
- [[concepts/navhard-ood-evaluation.md]] — 36.9 combined, second in the unscored cohort behind SpanVLA 40.1; an inference-path effect that is visible only on navhard.
- [[concepts/nuscenes-waymo-evals.md]] — 7.94 RFS on WOD-E2E test, beside Qwen-Drive's 7.91.
- [[concepts/perception-for-planning.md]] — future segmentation, depth and tracks rendered as video and read by the planner as latents: a head-free route.
- [[sources/simwam.md]] — the same backbone family and MoT layout, and the no-access null that SUV reproduces on navtest and breaks on navhard.
- [[sources/metis.md]] — the third member of the same-backbone [mask family](metis.md#mask-family). Its bidirectional mask is the *worst* on navhard, which conflicts with SUV's access gain unless the harm comes from the future reading the action.
- [[sources/coworld-vla.md]] — the four-target sibling that discards its generator at planning time.
- [[sources/driveva.md]], [[sources/drivewam.md]] — uncited Wan2.2-5B predecessors.
- [[sources/physwam.md]] — does not cite SUV and shares its central device: depth written as a video and generated by the same video expert through the frozen VAE. The differences are instructive. PhysWAM's depth is **metric and scored against LiDAR**, where SUV's is relative and scored against its DA3 teacher. PhysWAM's attention is bidirectional, where SUV's is one-way. And PhysWAM finds the depth stream alone worth nothing for planning (−0.3 PDMS), where SUV's three structured streams are worth +1.0 with access disabled. PhysWAM reports 38.1 navhard and 90.3 navtest against 36.9 and 91.0 here, at 9.4 GPU-seconds per plan against 288 ms. It confirms the 450 / 5,462 navhard split size given here.
