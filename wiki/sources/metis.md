---
title: "Metis: A Generalizable and Efficient World-Action Model for Autonomous Driving and Urban Navigation"
type: source-summary
sources: ["raw/papers/Metis_ A Generalizable and Efficient World-Action Model for Autonomous Driving and Urban Navigation.md"]
related: [sources/hydra-mdp-pp.md, concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/navhard-ood-evaluation.md, concepts/mixture-of-experts.md, concepts/foundation-backbones-for-ad.md, concepts/best-of-n.md, sources/simwam.md, sources/suv.md, sources/driveva.md, sources/drivewam.md, sources/drivelaw.md, sources/epona.md, sources/drivevla-w0.md, sources/drivefine.md, sources/sgdrive.md, sources/vega.md, sources/recogdrive.md, sources/brainwam.md, sources/adaptive-wam.md, sources/wa-jepa.md]
created: 2026-09-27
updated: 2026-09-27
confidence: medium
---

**Paper**: Metis: A Generalizable and Efficient World-Action Model for Autonomous Driving and Urban Navigation
**Authors**: Jingyu Li, Zhe Liu, Dongnan Hu, Junjie Wu, Zipei Ma, Wenxiao Wu, Chao Han, Zhihui Hao, Zhikang Liu, Kun Zhan, Jiankang Deng, Xiatian Zhu, Li Zhang
**Orgs**: Fudan University, Shanghai Innovation Institute, HKU, Tongji, **Li Auto**, HUST, Imperial College London, University of Surrey
**arXiv**: 2606.15869v1
**Code**: `github.com/LogosRoboticsGroup/Metis`

---

## Source Integrity Note

**Coverage.** The clipping is complete: main text, appendix A–H, and all twelve tables. Nine of ten figures are present. **Fig. 3, the asymmetric attention mask, has a caption but no image.** The mask is fully specified in the text, so nothing is lost.

**Caption and citation errors:**
- The Fig. 7 caption says "outdoor", but the image is the *indoor* deployment (`real_world_indoor_case_1.png`).
- The prose cites "Table 5" for the mask ablation (actually Table 4) and for image size (Table 4), and "Table 12" for denoising steps (Table 5 in the main text).

**Three internal inconsistencies matter:**

1. **Table 6 labels the main configuration "Wan2.2-14B".** Everywhere else, including implementation details, Appendix F and the 6B total, the video expert is **Wan2.2-5B**. The 1.04B-AE "14B" row reproduces the headline numbers (89.1 / 89.5 / 32.2) exactly, so the label is a typo, or the ablation is not what the text describes.
2. **Table 10's navhard columns are copied from Table 5.** The AE-capacity rows 0.21B / 0.45B / 1.04B carry navhard S1/S2/EPDMS of **76.0/40.7/31.2, 74.5/41.4/31.4, 75.8/41.7/32.2**. These are digit for digit the **2-, 5- and 10-denoising-step rows** of Table 5. Two different ablations producing identical three-number navhard triples is not plausible. The navhard half of the capacity ablation should be treated as unreported. Its navtest column (88.2 / 88.6 / 89.5) differs from Table 5's and may be genuine.
3. **The navhard scale disagrees with SUV's.** Appendix F describes navhard as **244** Stage-1 and **4,164** Stage-2 scenarios. [[sources/suv.md]] describes it as 450 and 5,462, yet copies Metis's navhard rows verbatim. Which split each paper ran is unresolved.

---

## Summary

Metis is a **Wan2.2-5B video expert + ~1B action expert** Mixture-of-Transformers world-action model (WAM), 6B parameters in total. It has **one attention rule**:
- Action tokens attend only to the current observation.
- Future-video tokens attend to the current observation **and to the action tokens**.

**Training** uses joint flow matching ($\mathcal L=\mathcal L_{\text{act}}+\mathcal L_{\text{video}}$, λ=1). The video expert learns to generate a future *consistent with the planned trajectory*, and the video loss back-propagates through the action tokens' keys and values into the action expert.

**Inference** drops the video branch entirely. With the video branch gone, the model is a ~1B flow-matching planner reading one pass of Wan features.

**Results.** One front camera at 640×768, no RL, no scorer:
- **89.5 EPDMS** on NAVSIM-v2 navtest (90.3 oracle best-of-6).
- **32.2 EPDMS** on navhard.
- **89.1 PDMS** on NAVSIM-v1 (89.7 best-of-6).
- **Second on CityWalker** urban-navigation orientation error, behind the pretrained ABot-N0 overall.
- Zero-shot qualitative deployment on a Unitree Go2 quadruped.
- **147 ms at 2 steps on an RTX 4090** (89.2 EPDMS); 480 ms at 10 steps.

**Why it matters for the wiki.** It completes a **three-paper mask family on one backbone**. [[sources/simwam.md]], Metis and [[sources/suv.md]] share Wan2.2-5B, a ~1B hidden-1024 action expert, joint flow matching, and nearly identical optimization recipes (60 epochs, 8×H200, AdamW 1e-4 / wd 0.01 / cosine). Between them they test four of the possible video↔action visibility patterns. Metis's specific contribution is the one direction the others did not test: **future video reads the action, the action never reads the future**. On navhard it beats the fully isolated mask by +2.2. See [the mask family](#mask-family).

---

## Method

![[intro1.png|Three WAM paradigms: (a) VLA-based WAMs with autoregressive token-based future prediction before action; (b) video-generation WAMs with tightly coupled joint video-action prediction; (c) Metis, which decouples video generation from action inference via masked asymmetric attention so planning needs no video generation]]

*Figure 1: (a) VLA-based WAMs rely on autoregressive token-based future prediction for action planning. (b) Video generation-based WAMs use tightly coupled architectures to jointly predict future video and actions. (c) Metis decouples video generation from action inference via masked asymmetric attention, enabling efficient planning without video generation.*

![[pipeline1 1.png|Metis overview: Wan2.2-5B video generation expert and ~1B action expert in a Mixture-of-Transformers, jointly trained with flow matching on future video and trajectory; at inference only the action expert runs, conditioned on the current observation]]

*Figure 2: Video and action are jointly learned during training, while inference directly predicts actions from the current observation.*

### Formulation

The paper sets its design against two existing WAM inference modes:
- **Joint denoising**: $(a,v)\sim p_\theta(\cdot\mid o_t,l)$.
- **Inverse dynamics**: $v\sim p_\theta(v\mid o_t,l)$, then $a\sim p_\theta(a\mid o_t,l,v)$.

Both require denoising high-dimensional future video at test time. Metis instead adopts **decoupled inference**, citing Fast-WAM: $a_{t:t+H}\sim p_\theta(a\mid z(o_t,l))$, where $z$ comes from one forward pass of the video backbone.

### Architecture

**Video generation expert (VGE).** Wan2.2-5B, with its video VAE and T5 text encoder reused.

**Action expert (AE).**
- A DiT with the same depth as the VGE and hidden size $d_a=1024$, about 1B parameters.
- An agent-state encoder encodes the ego state, which is combined with the language instruction as conditioning.
- All tokens cross-attend to the language embedding first. They then enter a shared attention space through expert-specific projections, and each expert keeps its own FFN and output head. This is the same MoT layout as SimWAM and SUV.

**Outputs.**
- NAVSIM: $(x,y,\theta)$ at 8 waypoints over 4 s.
- CityWalker: $(x,y)$ at 5 waypoints.
- Video and action chunks are aligned 1:1 in time.

### The asymmetric mask

| Query group ↓ / Key group → | Current obs | Future video | Action |
|---|:-:|:-:|:-:|
| Current obs | ✓ | ✗ | ✗ |
| Future video | ✓ | ✓ | **✓** |
| Action | ✓ | ✗ | ✓ |

**Losses**:

$$\mathcal L_{\text{act}}=\mathbb E\|u_\theta(a^{(s)},s\mid o_t,l)-\dot a\|^2,\qquad \mathcal L_{\text{video}}=\mathbb E\|u_\phi(z^{(s)},s\mid o_t,\hat a,l)-\dot z\|^2$$

Here $\hat a$ is "the predicted future action sequence", i.e. the action tokens at their current noise level.

The paper's argument (App. A.1) has two halves:
- **Training.** Video conditioned on the action is a "loose coupling": video-loss gradients flow into the AE and "implicitly optimize" it.
- **Inference.** Joint masks inject "generation noise" into the action. This matches [[sources/simwam.md]]'s efficiency argument, but gives a different causal story for accuracy.

**Mechanism not isolated.** Whether the gain comes from gradient flow into the action tokens or from the video expert learning action-conditioned futures is untested. It could be checked with a stop-gradient on the action K/V.

### Training

- **NAVSIM**: 640×768 input, 60 epochs, batch 64.
- **CityWalker**: 384×384, 30 epochs.
- AdamW, learning rate 1e-4, weight decay 0.01, cosine schedule, bf16, gradient clipping 1.0, 8×H200.
- Inference: 10 steps with CFG 1.0 (i.e. no guidance).
- The paper says it trains on "the navtrain subset (1,192 scenarios)", and its tables mark competitors with † for "full navtrain". What data Metis uses is not specified further. See [Limitations](#limitations).

---

## Results

### Table 1 — NAVSIM-v2 navhard (two-stage)

† = copied from GTRS; \* = copied from GuideFlow; all others reproduced.

| Method | Ref. | Stage | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | S. | EPDMS |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PDM-Closed (privileged) | – | S1 | 94.4 | 78.8 | 100 | 99.5 | 100 | 93.5 | 99.3 | 87.7 | 36.0 | – | |
| | | S2 | 88.1 | 90.6 | 96.3 | 98.5 | 100 | 83.1 | 73.7 | 91.5 | 25.4 | – | 51.3 |
| *Traditional E2E* | | | | | | | | | | | | | |
| LTF† | T-PAMI22 | S1 | 97.3 | 80.2 | 97.8 | 99.3 | 83.4 | 96.2 | 92.9 | 97.8 | 71.1 | 61.3 | |
| | | S2 | 79.4 | 69.0 | 85.6 | 98.5 | 83.8 | 76.7 | 47.9 | 97.0 | 70.6 | 39.2 | 24.4 |
| DiffusionDrive† | CVPR25 | S1 | 96.8 | 86.0 | 98.8 | 99.3 | 84.0 | 95.8 | 96.7 | 97.6 | 79.6 | 66.7 | |
| | | S2 | 80.1 | 72.8 | 84.4 | 98.4 | 85.9 | 76.6 | 46.4 | 96.3 | 72.8 | 40.5 | 27.5 |
| GTRS-DP\* | CVPRW25 | S1 | 94.7 | 78.8 | 96.1 | 99.5 | 83.0 | 94.4 | 92.0 | 97.5 | 72.8 | – | |
| | | S2 | 80.3 | 74.4 | 84.9 | 98.0 | 81.9 | 78.8 | 45.4 | 96.7 | 70.1 | – | 23.8 |
| GuideFlow\* | CVPR26 | S1 | 96.6 | 80.5 | 96.3 | 99.3 | 82.3 | 94.9 | 91.5 | 97.7 | 67.8 | – | |
| | | S2 | 87.3 | 76.7 | 88.8 | 99.2 | 84.3 | 85.1 | 49.7 | 93.1 | 44.5 | – | 27.1 |
| *VLA-based* | | | | | | | | | | | | | |
| ReCogDrive | ICLR26 | S1 | 96.4 | 78.9 | 98.7 | 99.8 | 82.6 | 95.6 | 94.4 | 97.6 | 74.2 | 67.7 | |
| | | S2 | 80.2 | 65.0 | 82.4 | 98.7 | 85.2 | 76.9 | 43.8 | 96.6 | 71.8 | 37.6 | 25.7 |
| SGDrive | CVPR26 | S1 | 95.8 | 87.6 | 97.8 | 99.8 | 84.4 | 94.7 | 92.9 | 97.8 | 28.9 | 71.1 | |
| | | S2 | 79.4 | 65.4 | 79.1 | 98.9 | 88.9 | 75.3 | 42.7 | 96.4 | 29.6 | 35.2 | 25.5 |
| *WAM-based* | | | | | | | | | | | | | |
| **Metis** | – | S1 | 96.6 | 87.8 | 99.0 | 99.3 | 84.5 | 95.6 | 97.8 | 97.8 | 77.8 | **75.8** | |
| | | S2 | 79.6 | 73.3 | 84.9 | 97.8 | 85.8 | 76.6 | 47.7 | 95.4 | 75.3 | **41.7** | **32.2** |

**The "best in both stages" claim holds only against this table.** In the wiki's [unscored navhard cohort](../concepts/navhard-ood-evaluation.md#scorer-cohort), 32.2 ranks **below** SpanVLA 40.1, SUV 36.9, GeoWAM 36.6, EponaV2 36.1, World4Drive 34.9, DriveFuture-unscored 34.6 and NavFormer 34.1. The paper's own "+7.9 DAC in Stage 2" is measured only against the two VLA rows.

**Baseline provenance differs from other papers.** Its DiffusionDrive (27.5) and LTF (24.4) differ from the values in GeoWAM/SUV (LTF 25.1) and DriveFuture (DiffusionDrive 24.2). Rows labelled "copied from GTRS" do not match what other papers copy from other sources.

### Table 2 — NAVSIM-v2 navtest

\* = RL; ‡ = best-of-6.

| Method | Sensors | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS | Residual† |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Human Agent | – | 100 | 100 | 99.8 | 100 | 87.4 | 100 | 100 | 98.1 | 90.1 | 90.3 | +4.1 |
| TransFuser | 3×C+L | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 | +1.3 |
| Hydra-MDP++ | 3×C+L | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 | |
| GTRS-Dense | 3×C | 97.6 | 97.5 | 99.0 | 99.9 | 87.9 | 97.0 | 95.9 | 97.5 | 55.9 | 82.3 | |
| DriveSuprim | 3×C | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 | +2.3 |
| ARTEMIS | 3×C+L | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | – | 83.1 | |
| DiffusionDrive | 3×C+L | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 | +2.5 |
| World4Drive | 3×C | 97.8 | 96.3 | 99.4 | 99.8 | 88.3 | 97.1 | 97.7 | 98.0 | 53.9 | 84.8 | |
| Drive-JEPA | 1×C | 98.8 | 97.4 | 99.0 | 99.8 | 83.5 | 98.0 | 96.2 | 98.1 | 85.6 | 85.4 | +1.8 |
| WorldRFT\* | 3×C | 97.8 | 96.5 | 99.5 | 99.8 | 88.5 | 97.0 | 97.4 | 98.1 | 69.1 | 86.7 | −1.4 |
| ReCogDrive\* | 1×C | 98.3 | 95.2 | 99.5 | 99.8 | 87.1 | 97.5 | 96.6 | 98.3 | 86.5 | 83.6 | +2.7 |
| SGDrive | 1×C | 98.6 | 94.3 | 99.5 | 99.9 | 86.0 | 97.9 | 96.1 | 98.3 | 85.9 | 86.2 | −0.7 |
| Vega | 1×C | 98.9 | 95.3 | 99.4 | 99.9 | 87.0 | 98.4 | 96.1 | 98.3 | 76.3 | 86.9 | −1.0 |
| DriveFine\* | 1×C | 98.7 | 97.3 | 98.8 | 99.8 | 88.2 | 97.8 | 97.7 | 98.4 | 84.7 | 87.1 | +1.2 |
| Epona | 1×C | 97.1 | 95.7 | 99.3 | 99.7 | 88.6 | 96.3 | 97.0 | 98.0 | 67.8 | 85.1 | −1.7 |
| DriveVLA-W0 † | 1×C | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 | −1.3 |
| **Metis** | 1×C | 98.4 | 97.2 | **99.6** | 99.8 | 87.8 | 97.7 | 97.8 | 98.4 | 88.0 | **89.5** | **−0.6** |
| Metis ‡ (BoN-6) | 1×C | 98.5 | 97.5 | 99.6 | 99.8 | 87.9 | 97.8 | 98.0 | 98.4 | 90.0 | 90.3 | −0.7 |

†Residual = closed-form EPDMS from the mean submetrics **minus** the reported value. This is the [convention of the NAVSIM page](../concepts/navsim-benchmark.md#submetric-residual). Rows are computed here where needed for the audit.

**Protocol audit: Metis compares a corrected-looking own row against pre-fix baselines.**
- The table's anchors are **pre-fix**: Human Agent **90.3** (94.5 under the corrected evaluator, with identical submetrics), TransFuser 76.7, ReCogDrive 83.6, DriveVLA-W0 86.1, DriveFine **87.1**.
- The residual separates two known pairs cleanly:
  - **Human Agent**: +4.1 pre-fix vs. −0.1 corrected.
  - **DriveFine**: +1.2 pre-fix (this table) vs. −0.8 corrected (89.7, [[sources/suv.md]]).
- Other rows follow the same split:
  - Pre-fix-looking (positive residual): DiffusionDrive, DriveSuprim, ReCogDrive, Drive-JEPA.
  - Corrected-looking (negative residual): SUV, SparseDriveV2 and EponaV2 all sit at −0.5 to −1.1.
- **Metis's own row sits at −0.6, with the corrected group.**
- If that reading is right, the headline **"+2.4 over prior VLA methods"** (vs. DriveFine 87.1) becomes **−0.2** against DriveFine's corrected 89.7.
- **"Metis BoN-6 = 90.3" also equals the pre-fix Human Agent row**, a coincidence the paper does not remark on.
- The residual is a heuristic, not a proof. DriveVLA-W0 86.1 (−1.3), classified pre-fix by [[sources/wa-jepa.md]], sits on the "wrong" side. See [[concepts/navsim-benchmark.md#residual-sign]].

**Where 89.5 ranks if corrected.** Behind WA-JEPA 91.7, SUV 91.0, Discrete-WAM 90.4, SparseDriveV2 90.1, CoWorld-VLA 90.0, DriveFuture 89.9, WAM-Diff/DriveFine 89.7 and LWDrive 89.6. That is mid-pack, and consistent with SUV's direct comparison (Metis 89.5 vs. SUV 91.0 on identical submetrics).

### Table 9 — NAVSIM-v1 navtest (PDMS)

| Method | Sensors | NC | DAC | EP | TTC | C | PDMS |
|---|---|---:|---:|---:|---:|---:|---:|
| Human Agent | – | 100 | 100 | 87.5 | 100 | 99.9 | 94.8 |
| UniAD | 6×C | 97.8 | 91.9 | 78.8 | 92.9 | 100 | 83.4 |
| TransFuser | 3×C+L | 97.7 | 92.8 | 79.2 | 92.8 | 100 | 84.0 |
| PARA-Drive | 6×C | 97.9 | 92.4 | 79.3 | 93.0 | 99.8 | 84.0 |
| LAW | 1×C | 96.4 | 95.4 | 81.7 | 88.7 | 99.9 | 84.6 |
| World4Drive | 3×C | 97.4 | 94.3 | 79.9 | 92.8 | 100 | 85.1 |
| DRAMA | 3×C+L | 98.0 | 93.1 | 80.1 | 94.8 | 100 | 85.5 |
| Hydra-MDP++ | 3×C+L | 97.6 | 96.0 | 80.4 | 93.1 | 100 | 86.6 |
| ARTEMIS | 3×C+L | 98.3 | 95.1 | 81.4 | 94.3 | 100 | 87.0 |
| WorldRFT\* | 3×C | 97.5 | 96.0 | 80.9 | 94.0 | 100 | 87.0 |
| DiffusionDrive | 3×C+L | 98.2 | 96.2 | 82.2 | 94.7 | 100 | 88.1 |
| WorldDrive | 1×C | 98.4 | 96.2 | 81.9 | 95.1 | 100 | 88.1 |
| WoTE | 3×C+L | 98.5 | 96.8 | 81.9 | 94.9 | 99.9 | 88.3 |
| SeerDrive | 3/6×C+L | 98.4 | 97.0 | 83.2 | 94.9 | 99.9 | 88.9 |
| Drive-JEPA ⚠ | 1×C | 98.7 | 96.2 | 82.9 | 100.0 | 95.5 | 89.0 |
| AutoVLA-IL | 3×C | 96.9 | 92.4 | 75.8 | 88.1 | 99.1 | 80.5 |
| ReCogDrive-IL | 1×C | 98.1 | 94.7 | 80.9 | 94.2 | 100 | 86.5 |
| SGDrive-IL | 1×C | 98.6 | 95.1 | 81.2 | 95.4 | 100 | 87.4 |
| Vega | 1×C | 98.9 | 95.3 | 81.6 | 96.1 | 100 | 87.9 |
| Epona | 1×C | 97.9 | 95.1 | 80.4 | 93.8 | 99.9 | 86.2 |
| ImagiDrive | 1×C | 98.6 | 96.2 | 80.5 | 94.5 | 100 | 87.4 |
| PWM† | 1×C | 98.6 | 95.9 | 81.8 | 95.4 | 100 | 88.1 |
| DriveVLA-W0† | 1×C | 98.7 | 96.2 | 82.2 | 95.5 | 100 | 88.4 |
| UniWorldVLA† | 1×C | 98.7 | 96.7 | 83.2 | 96.1 | 100 | 89.4 |
| DriveLaW† | 1×C | 99.0 | 97.1 | 81.3 | 96.7 | 100 | 89.1 |
| **Metis** | 1×C | 98.3 | 97.1 | 83.4 | 94.7 | 100 | **89.1** |
| Metis ‡ | 1×C | 98.5 | 97.5 | 84.0 | 95.1 | 100 | 89.7 |

- **"State-of-the-art, reaching 89.7" relies on the best-of-6 row.** Single-sample Metis (89.1) **ties DriveLaW and is below UniWorldVLA (89.4)** in its own table.
- This is also the source of the **Drive-JEPA TTC/Comf column swap** that [[sources/suv.md]] copied (TTC 100.0, C 95.5). It is now traced to Metis.
- In the wiki's v1 ladder, 89.1 is well down. It is below both same-backbone siblings: SimWAM 91.5 (90.3 before RL) and DriveVA 90.9.

### Table 3 — CityWalker (L2 m ↓ / MAOE ° ↓)

| Method | Metric | Mean | Turn 8% | Crossing 12% | Detour 12% | Proximity 6% | Crowd 7% | Other 55% | All |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ABot-N0\* (pretrained) | MAOE | **11.2** | 21.3 | **9.8** | 12.8 | **8.1** | 8.8 | **6.3** | **7.6** |
| GNM | L2 | 1.22 | 2.36 | 1.36 | 1.42 | 0.88 | 0.76 | 0.55 | 0.74 |
| | MAOE | 16.2 | 31.1 | 14.8 | 12.5 | 14.7 | 12.8 | 11.0 | 12.1 |
| ViNT | L2 | 1.30 | 1.91 | 1.13 | 1.14 | 0.77 | 0.66 | 0.57 | 0.70 |
| | MAOE | 16.5 | 31.1 | 15.4 | 12.9 | 14.8 | 13.3 | 11.6 | 12.6 |
| NoMaD | L2 | 1.39 | 2.49 | 1.56 | 1.55 | 1.06 | 0.95 | 0.76 | 0.74 |
| | MAOE | 19.1 | 35.1 | 18.5 | 15.6 | 18.1 | 14.3 | 12.8 | 12.1 |
| CityWalker | L2 | 1.11 | 1.27 | 1.00 | 1.15 | 1.06 | 1.12 | 1.06 | 1.07 |
| | MAOE | 15.2 | 26.6 | 14.1 | 13.9 | 14.3 | 12.0 | 10.4 | 11.5 |
| **Metis** | L2 | **0.71** | **0.69** | **0.77** | **0.66** | **0.76** | **0.77** | 0.63 | **0.64** |
| | MAOE | 11.8 | **19.5** | 11.9 | **9.1** | 11.4 | **8.0** | 7.8 | 9.8 |

- **Among fine-tuned methods, Metis leads clearly** on both L2 and MAOE (CityWalker's own model: 11.5° → 9.8° All).
- **Against ABot-N0 it loses overall**: MAOE Mean 11.2 vs 11.8, All 7.6 vs 9.8. It wins only on Turn, Detour and Crowd. The paper reports exactly those three wins and does not state the overall loss.
- The fine-tuned GNM and NoMaD rows share the same All-L2 (0.74) and All-MAOE (12.1), which looks like a copying error in the baseline table.

![[cw_vis1.png|CityWalker qualitative results: zero-shot Epona vs. zero-shot Metis, and fine-tuned Metis trajectories]]

*Figure 4: Qualitative results on CityWalker: zero-shot Epona, zero-shot Metis, and fine-tuned Metis.*

---

## Ablations

### Table 4 / Table 8 — attention mask (320×384) {#mask-ablation}

| Variant | Action reads future | Future reads action | v1 DAC | v1 EP | PDMS | v2 DAC | v2 EP | EPDMS | navhard |
|---|:-:|:-:|---:|---:|---:|---:|---:|---:|---:|
| Joint | ✓ | ✓ | 95.7 | 81.6 | 87.1 | 96.5 | 87.6 | 87.4 | 28.0 |
| Isolated | ✗ | ✗ | 96.5 | 82.6 | 88.0 | 96.9 | 87.7 | 88.3 | 29.4 |
| **Asymmetric (Metis)** | ✗ | **✓** | 96.9 | 82.9 | 88.3 | 97.0 | 87.8 | **88.8** | **31.6** |
| Asymmetric @ 640×768 | ✗ | ✓ | – | – | 89.1 | 97.5 | – | 89.5 | 32.2 |

Differences:

| Comparison | navtest (v1 / v2) | navhard |
|---|---:|---:|
| Asymmetric vs. isolated | +0.3 / +0.5 | **+2.2** |
| Isolated vs. joint | +0.9 PDMS | +1.4 |

**Letting the future read the action is worth little on navtest and +2.2 on navhard. Full bidirectional coupling is the worst variant everywhere.**

Caveats: this is at reduced resolution, with single runs. Stage-level navhard numbers are not given for the mask variants.

### Table 11 — co-training at all

| Training | PDMS | EPDMS |
|---|---:|---:|
| Action expert alone | 87.4 | 87.9 |
| + video co-training (asymmetric) | 89.1 | 89.5 |

Video co-training is worth +1.7 PDMS / +1.6 EPDMS. What "AE alone" reads (Wan features frozen? trained?) is not stated. There is no navhard number, which is the benchmark where the paper's mechanism shows.

### Table 5 / Table 12 — denoising steps and latency

| Steps | PDMS | navtest EPDMS | navhard S1 | S2 | navhard EPDMS | RTX 4090 ms | H200 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 87.0 | 87.2 | 74.5 | 39.8 | 30.4 | 110 | 100 |
| 2 | 88.9 | 89.2 | 76.0 | 40.7 | 31.2 | 147 | 140 |
| 5 | 89.0 | 89.4 | 74.5 | 41.4 | 31.4 | 280 | 240 |
| 10 | 89.1 | 89.5 | 75.8 | 41.7 | 32.2 | 480 | 430 |

These are all action-only, with no video generated.

### Table 7 — latency vs. other WAMs (RTX 4090)

| Method | PDMS | EPDMS | Latency (s) |
|---|---:|---:|---:|
| Epona | 86.2 | 85.1 | 0.32 |
| PWM | 87.3 | – | 0.57 |
| PWM (w/ video) | 88.1 | – | 0.83 |
| Metis (w/ video, 10 steps) | 89.0 | 89.5 | 1.38 |
| Metis (action-only, 2 steps) | 88.9 | 89.2 | 0.17 |

**The "up to 8× speedup" compares a 10-step run that also generates video against a 2-step action-only run**, so it mixes two factors. Table 12 separates them:
- **Skipping video at a matched 10 steps** saves 1.38 → 0.48 s, i.e. **2.9×**.
- **Cutting steps from 10 to 2** saves a further 3.3×.

**Generating video cannot change Metis's action by construction**, because the action never attends to the future. The "w/ video" row is therefore the same policy with extra computation. Its 89.5 EPDMS equals action-only at 10 steps. Its 89.0 PDMS differs from action-only's 89.1 by 0.1, which is presumably sampling noise.

### Table 6 — video expert and action expert size

| VGE | AE | PDMS | EPDMS | navhard |
|---|---|---:|---:|---:|
| Wan2.1-1.3B | ~0.24B | 88.5 | 88.8 | 28.8 |
| "Wan2.2-14B" ⚠ | ~0.21B | 88.2 | 88.2 | 31.2 |
| "Wan2.2-14B" ⚠ | ~1.04B | 89.1 | 89.5 | 32.2 |

⚠ These rows should read Wan2.2-5B; see [integrity note](#source-integrity-note).
- **The smaller video prior matches on navtest (88.8 vs. 88.2 with a similar AE) but loses 2.4 on navhard.** Like the mask ablation, the difference shows up only on navhard.
- Scaling the AE from 0.21B to 1.04B gives +1.3 navtest.

**Table 10 (AE capacity).** The navtest column is 88.2 / 88.6 / 89.5 for 0.21B / 0.45B / 1.04B. **Its navhard columns duplicate Table 5's step rows** and are not used here.

---

## The Mask Family: Three Papers, One Backbone {#mask-family}

[[sources/simwam.md]], Metis and [[sources/suv.md]] use the same Wan2.2-5B video expert with a ~1B hidden-1024 action expert in a MoT, and the same training recipe (joint flow matching with λ=1, 60 epochs, 8×H200, AdamW 1e-4 / wd 0.01 / cosine). Together they cover four visibility patterns:

| Pattern | Action → future | Future → action | Needs video at inference | SimWAM (v1 PDMS) | Metis @320×384 (v2 / navhard) | SUV (v2 / navhard) |
|---|:-:|:-:|:-:|---:|---:|---:|
| Bidirectional ("joint") | ✓ | ✓ | yes | 90.2 | 87.4 / **28.0** | – |
| Action reads future | ✓ | ✗ | yes | 90.1 | – | **91.0 / 36.9** |
| Isolated | ✗ | ✗ | no | **90.3** | 88.3 / 29.4 | 90.7 / 32.8 |
| Future reads action | ✗ | ✓ | no | – | **88.8 / 31.6** | – |

(SUV's rows include segmentation/depth/track supervision; its RGB-only pair is 89.7/30.5 isolated → 90.6/35.0 with access.)

**What the three papers agree on:**
- **On navtest the mask barely matters.** Excluding the joint variant, every contrast is within 0.5.
- **On navhard it matters a lot.** Metis: +2.2 for future-reads-action. SUV: +4.1 to +4.5 for action-reads-future. SimWAM never reported navhard.

**Where they conflict: letting the action read the future.** SUV finds one-way action→future reading gives the largest navhard gain in either paper. Metis finds the bidirectional mask, which includes the same pathway, is the **worst** variant on navhard (−1.4 vs. isolated). The two results are compatible only if **the harm comes from the future also reading the action**. That would be a feedback loop in which a noisy action conditions a noisy future, which the action then reads. It is consistent with Metis's "generation noise" explanation. It is also consistent with [[sources/brainwam.md]]'s finding that symmetric unmasked coupling hurts. It is a hypothesis, not a measurement: **no paper has run the "action reads future" and "future reads action" variants together**, which would complete the 2×2.

**What this does to the test-time-imagination question.** The training-time-only camp's best variant (Metis's asymmetric mask, 31.6 at low resolution / 32.2 at full) and the imagine-then-act camp's best variant (SUV's access mask, 36.9) differ by ~4.7 navhard points. But they come from two papers with different resolutions and target sets. The same-paper deltas are Metis +2.2 over isolated and SUV +4.1 over isolated, which puts SUV's inference-time route ahead. Single runs in both.

---

## Qualitative Results

![[navsim_vis1.png|NAVSIM turning scenarios: Metis trajectories compared with ReCogDrive, aligning better with road geometry]]

*Figure 5: Qualitative evaluation of turning scenarios on NAVSIM (vs. ReCogDrive).*

![[video_gen_vis_1.png|Video generation expert outputs vs. ground truth in static and dynamic scenes; distant background vehicles lose detail in dynamic intersections]]

*Figure 8: Generated frames (top) vs. ground truth (bottom). Static scenes are high quality; distant vehicles lose detail at dynamic intersections.*

![[supp_navsim_vis1.png|Additional NAVSIM qualitative results: lane following and turning]]

*Figure 9: Additional NAVSIM results: lane following and turning.*

![[supp_navsim_fail_vis.png|NAVSIM failure case: slight long-horizon deviation attributed to monocular input]]

*Figure 10: Failure case. Slight long-horizon deviation, which the authors attribute to the single front camera.*

![[real_world_outdoor_case_1.png|Zero-shot outdoor nighttime deployment on a Unitree Go2 quadruped with obstacle avoidance]]

*Figure 6: Zero-shot outdoor deployment on a Unitree Go2 (faces anonymized).*

![[real_world_indoor_case_1.png|Zero-shot indoor daytime deployment on a Unitree Go2 quadruped with obstacle avoidance]]

*Figure 7: Zero-shot indoor deployment. The paper's caption wrongly says "outdoor".*

The real-robot evidence is **four qualitative examples, with no success rate, no trial count, and no baseline numbers**, even though §4 says "we deploy our model and the baseline methods on the same Unitree Go2". The paper does not state which model (NAVSIM or CityWalker) runs on the robot, or how the ego-state input is supplied.

**No video-quality metric (FVD, FID, PSNR) is reported.** The only evidence on the VGE's output is Fig. 8.

---

## Limitations {#limitations}

1. **The v2 table very likely mixes conventions.** Its baselines are pre-fix (Human 90.3, DriveFine 87.1), while Metis's own row has a corrected-looking residual. Against DriveFine's corrected 89.7, the claimed "+2.4 over VLA" becomes −0.2. [[sources/wa-jepa.md]] 91.7 and [[sources/suv.md]] 91.0 are both higher under the corrected protocol.
2. **"SOTA on v1" rests on best-of-6 (89.7).** Single-sample 89.1 ties DriveLaW and trails UniWorldVLA (89.4) in its own table, and is below the same-backbone SimWAM (90.3 before RL) and DriveVA (90.9).
3. **"Best on navhard" holds only in its own table.** Seven unscored wiki entries exceed 32.2.
4. **CityWalker: second overall.** ABot-N0 leads on Mean and All MAOE; the paper reports only the three sub-scenarios it wins.
5. **Copied ablation rows.** Table 10's navhard columns duplicate Table 5's step rows. Table 6 labels the 5B model "Wan2.2-14B".
6. **The speedup conflates two factors.** The 8× compares 10-step-with-video against 2-step-without; the matched-step saving is 2.9×. The "w/ video" row cannot differ in action by construction.
7. **The mechanism is not isolated.** Gradient flow through the action K/V versus action-conditioned video learning is untested, and the co-training ablation has no navhard number.
8. ~~**Training data is unclear.**~~ *Resolved by [[sources/hydra-mdp-pp.md]], which describes navtrain/navtest as "1192 and 136 scenarios", i.e. **log counts**. "1,192 scenarios" is the full navtrain, not a subset; Metis's † markers remain unexplained.*
9. **The navhard split is ambiguous.** 244/4,164 scenarios here vs. 450/5,462 in SUV, which copies Metis's rows. Metis's DiffusionDrive/LTF baselines differ from other papers' copies.
10. **Real-world results are qualitative only.** There is no success rate, and the baselines said to be deployed are never reported.
11. **No video-generation metrics, no seeds, no RL, one front camera**, and navhard (244 Stage-1 scenes) carries every mechanism claim.

---

## Key Cross-References

- [[concepts/world-model-for-ad.md]] — Pattern 37 (future reads the action, the action never reads the future). It also completes the [mask family](../concepts/world-model-for-ad.md#mask-family) with SimWAM and SUV, and shows a conflict with SUV on whether the action should read the future, plausibly resolved by feedback direction.
- [[concepts/navsim-benchmark.md]] — 89.5 v2 (probably corrected, compared against pre-fix baselines) and 89.1 v1. The **residual-sign heuristic** for classifying a row's protocol ([#residual-sign](../concepts/navsim-benchmark.md#residual-sign)), validated on two known pairs. The source of the Drive-JEPA column swap.
- [[concepts/navhard-ood-evaluation.md]] — 32.2 unscored; the 244 vs. 450 split discrepancy; the video-prior-scale and mask effects that appear only on navhard.
- [[sources/simwam.md]] — same backbone and recipe. Isolated mask; its navtest null matches Metis's ±0.5 on navtest.
- [[sources/suv.md]] — same backbone and recipe. The action reads the future; it copies Metis's v2/navhard rows and the Drive-JEPA swap.
- [[sources/brainwam.md]] — symmetric, unmasked coupling hurts. The same direction as Metis's joint variant.
