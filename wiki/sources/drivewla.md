---
title: "DriveWLA: Unifying Future Dynamics and Scene Semantics through World-Language-Action Modeling for Autonomous Driving"
type: source-summary
sources: ["raw/papers/DriveWLA_ Unifying Future Dynamics and Scene Semantics throughWorld-Language-Action Modeling for Autonomous Driving.md"]
related: [sources/drivelaw.md, sources/recogdrive.md, sources/brainwam.md, sources/drivewam.md, sources/simwam.md, sources/drivevla-w0.md, sources/reworld.md, sources/sgdrive.md, sources/qwen-drive-1.0.md, sources/wa-jepa.md, sources/drivor.md, sources/physwam.md, sources/mm-future.md, sources/coworld-vla.md, sources/drivefuture.md, sources/latent-wam.md, sources/diffusiondrive-v2.md, sources/driveva.md, sources/simvta.md, sources/had.md, sources/rad.md, sources/adaptive-wam.md, sources/raf-vla.md, sources/policy-world-model.md, sources/unidrivevla.md, concepts/world-model-for-ad.md, concepts/wam-attention-masks.md, concepts/dual-system-vla.md, concepts/navsim-benchmark.md, concepts/hugsim-benchmark.md, concepts/nuscenes-waymo-evals.md, concepts/rl-for-ad.md, concepts/inference-latency.md, concepts/mixture-of-experts.md, research-directions.md]
created: 2026-10-19
updated: 2026-10-19
confidence: medium
---

# DriveWLA: Unifying Future Dynamics and Scene Semantics through World-Language-Action Modeling

**Paper**: DriveWLA: Unifying Future Dynamics and Scene Semantics through World-Language-Action Modeling for Autonomous Driving
**Authors**: Shuzheng Wang, Haoxuan Li, Shengzhen Li, Chenxi Wu, Yajing Sun, Lu Hou, Wenyu Liu, Xinggang Wang (corresponding)
**Orgs**: Huazhong University of Science and Technology (Xinggang Wang's group: DiffusionDrive, RAD, VAD); Yinwang Intelligent Technology Co. Ltd.
**arXiv**: 2610.17903v1 (October 2026)
**Code**: none stated
**Source**: `raw/papers/DriveWLA_ Unifying Future Dynamics and Scene Semantics throughWorld-Language-Action Modeling for Autonomous Driving.md`. The clipping has all text and Tables 1–8 and four figures (`drivewla_overview.png`, `attention_mask.png`, `expert_ablation_vis.png`, `representation_pca_vis.png`). Figure 3(b) was missing and was downloaded from the arXiv HTML into `raw/assets/il_rl_vis.png` (a new file).

**Confidence: medium.** The tables are clean (all 20 v1 rows pass the closed-form check), the ablations are on one checkpoint, and the HUGSIM protocol is stated. Against that:
- Every number is a single run.
- The "World Expert" receives no video loss.
- Its v2 table, captioned "all corrected", carries two pre-fix-like rows.
- Both "state of the art" claims hold only within its own tables.
- The 970 ms latency has no hardware named.
- Bench2Drive is open-loop only.

---

## What It Is

**Three experts in one Mixture-of-Transformers with a fixed attention mask: a video-pretrained DiT (World), a frozen driving VLM behind a trainable adapter (Language), and a flow-matching trajectory DiT (Action). All three are trained by the trajectory loss alone, then the action expert is refined with Flow-GRPO.**

1. **World Expert**: DriveLaW-Video, built on LTX-Video-2B. A frozen VAE encodes four front-view frames (768×1344) into $Z_\text{obs}$, and the future latent $Z_\text{fut}$ starts as Gaussian noise. A trainable 28-block DiT (width 2048) processes $[Z_\text{obs};Z_\text{fut}]$ **in one forward pass**. Nothing is denoised iteratively or decoded, and **no video-prediction loss is used**, so the "future" positions are shaped only by the planning loss through attention.
2. **Language Expert**: ReCogDrive's driving-pretrained InternVL3-2B, frozen, over the current front image and a prompt (ego motion, command, instruction). Its final-layer hidden states feed a trainable 28-block adapter (width 512). There is no text decoding.
3. **Action Expert**: a 28-block flow-matching DiT (width 512) conditioned on four past poses, the ego state and the command; eight $(x,y,\theta)$ waypoints over 4 s; 10 Euler steps.
4. **Cross-Expert Attention**: the three experts' QKV are concatenated in every block under a fixed mask. World and Language see each other; Action sees both; neither sees Action.
5. **Flow-GRPO** on the action expert's LoRA only (rank 16), on training scenes where the imitation policy scores PDMS below 90.
   - 8 samples per scene from an SDE version of the flow.
   - **Factorized advantage**: each sub-score is standardized within the group and weighted (NC, DAC, C 2; EP, TTC 5), then scaled by a shrinkage factor $s^2/(s^2+\tau^2)$ on the group's PDMS variance.
   - BC regularization 0.1, step discount 0.8.

**Results**

| Benchmark | DriveWLA | Notes |
|---|---:|---|
| NAVSIM-v1 PDMS (navtest) | **92.3** (90.8 imitation only) | SimWAM 91.5, DriveWorld-VLA 91.3, SGDrive 91.1 in its table |
| NAVSIM-v2 EPDMS (navtest) | **90.1** | CoWorld-VLA 90.0, DriveFuture 89.9 in its table; EC 82.5 |
| HUGSIM, 436 scenarios, zero-shot | **41.47 HD-Score**; Extreme tier 15.87 | WA-JEPA's protocol and baselines; WA-JEPA's own 44.6 not printed |
| nuScenes open-loop, zero-shot | 0.94 m / 0.09% (UniAD metric, with ego state) | Front camera only |
| Bench2Drive, open-loop on recorded frames | 1.61 m / 1.06% | Not the closed-loop Driving Score |
| Latency, 10 steps | 970 ms (hardware not stated) | 540 ms at 1 step (49.2 PDMS) |
| Training | 30 epochs imitation on 16 A100s; 8 epochs Flow-GRPO | |

---

## Key Takeaways

- **Without the language expert it is DriveLaW; with it, +1.7; with RL, +1.5.** The world-plus-action model scores 89.1 PDMS, exactly DriveLaW's published score with the same video backbone. Adding the frozen VLM through attention gives 90.8, and Flow-GRPO 92.3. Removing the world expert instead gives 86.2.
- **The "World Expert" is a video-pretrained backbone, not a trained predictor.** Its future tokens start from noise, pass through the DiT once, and receive no video loss. Fine-tuning it on the trajectory loss is worth +2.8 over freezing it (88.0 → 90.8). In the appendix PCA, future latents 1 and 2 (the first and second 2 s) look almost identical to each other, as history latents 1 and 4 do; the paper reads them as "structured predictive context". It is the single-pass readout of a video prior used by [[sources/drivelaw.md]] and [[sources/adaptive-wam.md]], with every layer read through attention.
- **A frozen VLM in shared attention with the video stream helps here.** World↔Language in both directions gives 90.8; one direction 89.8–90.1; neither 88.4. Letting the action stream write back costs 0.4. [[sources/brainwam.md]] found the opposite: a VLM in shared attention with a video stream that is denoised iteratively (87.8 against 88.1 without it). DriveWLA's world stream is a single noise-level pass, which is the obvious difference.
- **The RL recipe change is small.** Original Flow-GRPO gives +1.3 (90.8 → 92.1, mostly DAC and EP). Factorized advantages with variance shrinkage add +0.2 more (92.3), single runs. RL is run only on the training scenes the imitation policy scores below 90.
- **"State of the art" holds only within its tables.**
  - On v1, 92.3 is above its table but below at least seven earlier VLM or world-model entries in the wiki: WCog-VLA 92.9 (same InternVL3-2B), CLEAR 93.7, DA-WAM 93.7, Drive-HWM 93.3/93.8, MM-Future 94.0, World4Scorer 94.0, ChainFlow-VLA 94.8.
  - On v2, 90.1 is below WA-JEPA 91.7, MM-Future 91.5, SUV 91.0, ReDrive 90.8 and CLOVER 90.4 on the corrected scale.
- **HUGSIM: second on the 436-scenario set, and best on its Extreme tier.** 41.47 HD-Score, against WA-JEPA's 44.62, which this paper uses for its protocol and every baseline row but leaves out. Its Extreme tier (15.87) is the first above the 0.06–0.14 band recorded for every earlier method on that set. Easy (73.87) is below DrivoR and WA-JEPA, and comfort (75.78) is low as for other world-model planners. It uses one front camera.
- **The zero-shot open-loop numbers are protocol-sensitive.** nuScenes collision of 0.09%, below every fine-tuned row, comes from a front-camera NAVSIM model given ego state, under the UniAD collision definition ([[concepts/nuscenes-waymo-evals.md]]). Bench2Drive is L2 and collision on recorded frames, which the appendix states and the abstract's "closed-loop" framing blurs.

---

## Method

![[drivewla_overview.png|Figure 1: Model architecture and training strategy. DriveWLA unifies World, Language, and Action Experts through Cross-Expert Attention. The training strategy consists of trajectory imitation learning, which jointly optimizes all experts, followed by reinforcement learning refinement of the Action Expert.]]

**Figure 1**: Left, three columns of blocks:
- World Expert: frames → frozen VAE → future-dynamic tokens plus noise → World DiT block.
- Action Expert: noisy action, command and ego status → encoder → action tokens → Action DiT block.
- Language Expert: image and "Describe the scene." → frozen VLM → scene-semantic tokens → VLM adapter block.

Each block's QKV joins one Cross-Expert Attention, ×N. Right: Stage I imitation (World, Action and Language blocks trainable, $\mathcal L_\text{BC}$). Stage II RL: K samples, scored by a driving simulator into factorized advantages; only the Action LoRA is trainable.

![[attention_mask.png|Figure 2: Query-Key/Value Visibility in Cross-Expert Attention.]]

**Figure 2**: Rows are queries, columns keys and values. World and Language queries see World and Language keys; Action queries see all three.

### Cross-Expert Attention (§4.2, A.2)

$$O^\ell=\mathrm{softmax}\!\left(\frac{Q^\ell (K^\ell)^\top}{\sqrt d}+M\right)V^\ell,\qquad M_{(W,A,L)}=\begin{pmatrix}0&-\infty&0\\0&0&0\\0&-\infty&0\end{pmatrix}$$

### Flow-GRPO with factorized advantages (§4.3, App. C)

$$z_{i,m}=\frac{r_{i,m}-\mu_m}{\sigma_m+\epsilon},\qquad A_i=\frac{\sum_m w_m z_{i,m}}{\sqrt{\mathrm{Var}(\sum_m w_m z_{i,m})+\epsilon}}\cdot\frac{s^2}{s^2+\tau^2}$$

$$\mathcal L=-\frac1K\sum_i\frac1T\sum_t\gamma^{t-1}\log\pi_\theta(a^i_{t-1}\mid a^i_t,\mathcal C_d)\,A_i+\lambda_\text{BC}\mathcal L_\text{BC}$$

$w_\text{NC}=w_\text{DAC}=w_\text{C}=2$, $w_\text{EP}=w_\text{TTC}=5$; $\tau=0.1$ ($s^2$ is the group's PDMS variance); $K=8$; 10 stochastic steps; $\gamma=0.8$; $\lambda_\text{BC}=0.1$; 8 epochs. Unlike PDMS, the weighting treats NC and DAC as additive terms rather than gates.

---

## Results

### Table 1: NAVSIM v1 navtest

Residual = closed form − reported, computed here.

| Method | VLM | Video | NC | DAC | TTC | Comf. | EP | PDMS | Residual |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| *VLM-based* | | | | | | | | | |
| ReCogDrive | InternVL3-2B | – | 97.9 | 97.3 | 94.9 | 100.0 | 87.3 | 90.8 | −2.61 |
| SGDrive | InternVL3-2B | – | 98.6 | 97.8 | 96.2 | 100.0 | 85.8 | 91.1 | −1.90 |
| Qwen-Drive-1.0 | Qwen3.5-4B | – | 98.6 | 98.2 | 95.9 | 100.0 | 84.8 | 90.7 | −1.66 |
| *World-model-based* | | | | | | | | | |
| DriveLaW | – | LTX-2B | 99.0 | 97.1 | 96.7 | 100.0 | 81.3 | 89.1 | −1.78 |
| ReWorld | – | LTX-2B | 99.1 | 98.2 | 97.7 | 99.8 | 82.0 | 90.4 | −1.35 |
| SimWAM | – | Wan2.2-5B | 98.4 | 98.7 | 95.5 | 100.0 | 86.4 | 91.5 | −1.70 |
| *VLM–WM* | | | | | | | | | |
| DriveVLA-W0 | Emu3-8B (unified) | | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.2 | −1.22 |
| DriveWAM | Qwen3-VL-8B | Wan2.2-5B | 98.3 | 98.1 | 95.2 | 100.0 | 84.3 | 90.1 | −1.90 |
| BrainWAM | Qwen3-VL-4B | Wan2.2-5B | 98.1 | 97.5 | 94.9 | 100.0 | 83.8 | 89.5 | −2.34 |
| DriveWorld-VLA | InternVL3-2B | ResNet-34 (BEV) | 99.1 | 98.2 | 96.1 | 100.0 | 85.9 | 91.3 | −1.28 |
| **DriveWLA** | InternVL3-2B | LTX-2B | 99.1 | 98.8 | 97.0 | 100.0 | 86.4 | **92.3** | −1.16 |

ReCogDrive's 90.8 is labelled InternVL3-2B here; the wiki's ReCogDrive page describes the 8B model (89.6 on arXiv).

### Table 2: NAVSIM v2 navtest ("all results use the corrected official evaluator")

| Method | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS | Residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DiffusionDrive | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 | **+2.48** |
| DiffusionDriveV2 | 97.7 | 96.6 | 99.2 | 99.8 | 88.9 | 97.2 | 96.0 | 97.8 | 91.0 | 87.5 | +0.10 |
| DriveWorld-VLA | 98.6 | 99.1 | 99.6 | 99.8 | 87.4 | 97.9 | 97.0 | 97.8 | 78.6 | 86.8 | **+2.64** |
| Latent-WAM | 98.1 | 97.3 | 99.6 | 99.8 | 87.7 | 97.3 | 97.6 | 98.1 | 87.3 | 89.3 | −0.88 |
| CoWorld-VLA | 99.1 | 97.0 | 99.6 | 99.9 | 87.9 | 98.5 | 97.7 | 98.2 | 86.2 | 90.0 | −0.56 |
| DriveFuture | 98.8 | 99.1 | 99.6 | 99.9 | 86.6 | 98.4 | 96.4 | 98.3 | 74.8 | 89.9 | −0.76 |
| **DriveWLA** | 98.8 | 97.8 | 99.7 | 99.9 | 87.4 | 98.8 | 97.8 | 98.4 | 82.5 | **90.1** | −0.57 |

DiffusionDrive's 84.5 is the pre-fix value (corrected 88.2, [[concepts/navsim-benchmark.md]]), and DriveWorld-VLA's 86.8 has the same pre-fix signature, so the caption is wrong for two of seven rows. This table does not have the shifted Latent-WAM / DriveWorld-VLA rows found in [[sources/hyworldvla.md]]'s (here Latent-WAM 89.3 and DriveWorld-VLA 86.8, as published).

### Table 3: HUGSIM zero-shot, 436 scenarios (WA-JEPA's protocol)

| Method | NC | DAC | TTC | Comf. | PDMS | RC | Easy | Medium | Hard | Extreme | **HD-Score** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VAD | 41.20 | 90.30 | 28.00 | 95.30 | 28.30 | 30.10 | 42.00 | 8.50 | 7.70 | 6.30 | 13.90 |
| UniAD | 65.60 | 93.20 | 51.60 | 66.30 | 49.40 | 43.80 | 64.00 | 37.20 | 21.00 | 6.30 | 31.20 |
| LTF | 44.30 | 92.80 | 37.50 | 94.80 | 36.50 | 38.00 | 66.10 | 15.50 | 12.00 | 11.70 | 23.10 |
| DrivoR | 52.20 | 95.60 | 46.20 | 93.90 | 44.80 | 47.20 | **78.00** | 29.10 | 20.00 | 14.10 | 32.50 |
| **DriveWLA** | **65.80** | 95.06 | **59.27** | 75.78 | **54.92** | **54.99** | 73.87 | **47.91** | **31.41** | **15.87** | **41.47** |
| *WA-JEPA (from [[sources/wa-jepa.md]]; not in this table)* | *68.56* | *96.35* | *61.20* | *66.20* | *57.17* | *56.89* | *79.77* | *55.63* | *30.60* | *13.62* | *44.62* |

The four baselines are WA-JEPA's rescorings, rescaled ×100. The tier weights 80 / 157 / 96 / 103 reproduce DriveWLA's 41.47 exactly. The paper also gives source-dataset counts: nuScenes 88, KITTI-360 113, Waymo 108, PandaSet 127. Within each tier, scores are averaged per dataset and then across datasets. Cameras: DriveWLA 1, LTF 3, the others 4.

### Table 4: zero-shot open loop (nuScenes, Bench2Drive)

| Dataset | Method | Fine-tuned | L2 1 / 2 / 3 s | L2 avg | Coll. 1 / 2 / 3 s | Coll. avg |
|---|---|:-:|---|---:|---|---:|
| nuScenes | UniAD | ✓ | 0.48 / 0.96 / 1.65 | 1.03 | 0.05 / 0.17 / 0.71 | 0.31 |
| | OpenDriveVLA | ✓ | 0.19 / 0.58 / 1.24 | 0.67 | 0.02 / 0.18 / 0.70 | 0.30 |
| | UniDriveVLA-L | ✓ | 0.30 / 0.66 / 1.25 | 0.74 | 0.02 / 0.18 / 0.40 | 0.20 |
| | DriveVLA-W0 | ✗ | 0.43 / 1.26 / 2.60 | 1.43 | 0.22 / 0.66 / 1.42 | 0.77 |
| | PWM | ✗ | 2.06 / 3.91 / 6.00 | 3.99 | 0.12 / 0.15 / 0.86 | 0.36 |
| | **DriveWLA** | ✗ | 0.38 / 0.83 / 1.60 | **0.94** | 0.03 / 0.03 / 0.20 | **0.09** |
| Bench2Drive | DriveVLA-W0 | ✗ | 1.01 / 2.77 / 5.22 | 3.00 | 1.49 / 2.53 / 3.53 | 2.52 |
| | PWM | ✗ | 1.70 / 2.74 / 3.97 | 2.80 | 4.01 / 3.73 / 3.53 | 3.76 |
| | DriveVA | ✗ | 0.69 / 1.29 / 2.03 | 1.33 | 1.38 / 1.97 / 2.65 | 1.79 |
| | UNIVERSE | ✗ | 0.67 / 1.29 / 2.03 | 1.33 | 0.75 / 1.12 / 1.70 | 1.19 |
| | **DriveWLA** | ✗ | 0.77 / 1.53 / 2.53 | 1.61 | 0.34 / 0.97 / 1.88 | **1.06** |

UniAD protocol (per-horizon, averaged over the three horizons). The zero-shot baselines come from DriveVA's and UNIVERSE's tables. The UniAD row has no ego-state input; OpenDriveVLA, UniDriveVLA-L and DriveWLA do. On Bench2Drive DriveWLA's L2 is worse than DriveVA's and UNIVERSE's, and its collision is lower.

### Tables 5–8: ablations (imitation checkpoint, navtest, single runs)

| Table 5: experts | PDMS |
|---|---:|
| Language + Action (no World) | 86.2 |
| World + Action (no Language) | 89.1 |
| All three | 90.8 |

| Table 6: training | NC | DAC | TTC | Comf. | EP | PDMS |
|---|---:|---:|---:|---:|---:|---:|
| Frozen World Expert | 98.5 | 96.1 | 96.4 | 100.0 | 80.4 | 88.0 |
| Trainable World Expert | 99.2 | 97.7 | 97.2 | 100.0 | 84.0 | 90.8 |
| + Flow-GRPO | 99.1 | 98.7 | 97.1 | 100.0 | 86.1 | 92.1 |
| + Flow-GRPO† (factorized, shrinkage) | 99.1 | 98.8 | 97.0 | 100.0 | 86.4 | **92.3** |

| Table 7: W→L | L→W | W/L→A | A→W/L | PDMS |
|:-:|:-:|:-:|:-:|---:|
| ✓ | ✓ | ✓ | ✗ | **90.8** |
| ✓ | ✗ | ✓ | ✗ | 90.1 |
| ✗ | ✓ | ✓ | ✗ | 89.8 |
| ✓ | ✓ | ✓ | ✓ | 90.4 |
| ✗ | ✗ | ✓ | ✗ | 88.4 |

| Table 8: Euler steps | 1 | 5 | **10** | 15 | 20 |
|---|---:|---:|---:|---:|---:|
| PDMS | 49.2 | 90.6 | **90.8** | 90.8 | 90.7 |
| Latency | 540 ms | 810 ms | 970 ms | 1160 ms | 1390 ms |

At one step, comfort collapses to 1.7. The text refers to these tables as "Tab. 6" and "Tab. 8" in places where Tables 5 and 7 are meant.

---

## Visualizations

![[expert_ablation_vis.png|Figure 3(a): Qualitative visualization of expert complementarity. Removing either the Language or World Expert leads to planning failures in different scenes, highlighting their complementary contributions.]]

![[il_rl_vis.png|Figure 3(b): RL refinement corrects the imitation policy and discovers a valid alternative trajectory distinct from the expert demonstration. Recovered from the arXiv HTML.]]

**Figure 3**: (a) Two left turns with the full model and each one-expert ablation. In the first, removing the World Expert puts the plan off the drivable area (DAC 0). In the second, removing the Language Expert does (DAC 0). (b) *(recovered from the arXiv HTML)* A right turn at a yield sign: the imitation plan cuts the corner (DAC 0); after RL it stays on the road, along a path different from the human's (DAC 1).

![[representation_pca_vis.png|Figure 4: World Expert latent features. Each row shows one driving scene. The columns show the front-view image, history latents 1 and 4, and future latents 1 and 2. All PCA-projected latent maps use the same color scale.]]

**Figure 4**: PCA of the World Expert's latents for three scenes. History latents 1 and 4 are nearly identical, and so are future latents 1 and 2 (each covering 2 s). The future maps keep the scene's layout in a different colour range. Nothing in the figure shows motion between the two future groups.

---

## Table Audit {#table-audit}

1. **All 20 v1 rows pass the closed-form check** (−1.16 for DriveWLA; −8.19 for the collapsed 1-step row).
2. **The v2 caption is wrong for two rows**: DiffusionDrive 84.5 and DriveWorld-VLA 86.8 are pre-fix-like (+2.5, +2.6). The other five are corrected-like or ambiguous. DriveWLA's lead over the table's corrected rows is +0.1 to +0.2.
3. **HUGSIM**: the baselines are WA-JEPA's rows, and WA-JEPA itself (44.62) is not printed. The tier-weighted average reproduces exactly.
4. **SOTA claims**: on v1, at least seven earlier VLM or world-model entries in the wiki are higher (see Key Takeaways); on v2, at least five corrected-scale entries. Its v1 table is limited to VLM, world-model and hybrid methods, which excludes the scorer cohort.
5. **Text slips**: the text calls Table 5 "Tab. 6" and Table 7 "Tab. 8"; the abstract presents nuScenes and Bench2Drive (open loop) alongside HUGSIM as zero-shot "closed-loop" evidence.

---

## Limitations

The paper states none.

The wiki adds:

1. **No video objective.** The "future dynamics" expert is trained only by the trajectory loss, from Gaussian-noise future tokens in one pass. Whether its future positions carry predictions is untested (no probe, no shuffled-history control), and the PCA shows little change across the future horizon.
2. **Removing an expert removes a whole network**: the 2B video DiT, or the 2B VLM with its adapter. "Predictive dynamics contribute more" (−4.6 against −1.7) also measures capacity and video pretraining.
3. **Single runs**: the factorized advantage's +0.2 and the mask rows (within 1 point) are within run-to-run noise ([[concepts/evaluation-variance.md]]).
4. **RL scene selection**: Flow-GRPO uses only training scenes below 90 PDMS, a curriculum that is not ablated.
5. **Latency is about 1 s with no hardware named**, and one Euler step collapses (49.2).
6. **Front camera only**, on NAVSIM and HUGSIM alike.
7. **No navhard and no Bench2Drive closed loop**; the HUGSIM claim is zero-shot, single run.

---

## Relation to Other Wiki Pages

- [[sources/drivelaw.md]]: supplies the World Expert (DriveLaW-Video on LTX-2B). Without the language expert DriveWLA scores DriveLaW's 89.1. DriveLaW reads a mid-denoising latent as the planning state; DriveWLA reads every layer of one pass through attention.
- [[sources/recogdrive.md]]: supplies the frozen InternVL3-2B. [[sources/sgdrive.md]] and [[sources/onedrive.md]] use the same initialization.
- [[sources/brainwam.md]], [[sources/drivewam.md]], [[sources/simwam.md]], [[sources/drivevla-w0.md]]: the VLM-plus-world-model designs it compares with. BrainWAM's finding that a VLM in shared attention hurts an iteratively denoised video stream is the counterpoint to DriveWLA's bidirectional gain ([[concepts/wam-attention-masks.md]]).
- [[sources/reworld.md]]: DriveLaW's other descendant (90.4), from the same Xiaomi-HUST line.
- [[sources/wa-jepa.md]]: the source of its HUGSIM protocol and baseline rows, not its own 44.62. [[sources/physwam.md]] (35.5) and [[sources/mm-future.md]] (32.3) are the other 436-scenario entries.
- [[sources/coworld-vla.md]], [[sources/drivefuture.md]], [[sources/latent-wam.md]], [[sources/diffusiondrive-v2.md]]: its v2 comparison rows.
- [[sources/simvta.md]], [[sources/had.md]], [[sources/rad.md]]: other RL recipes on driving rewards. SimVTA also uses Flow-GRPO with a BC anchor and a 0.6 denoising discount; HAD decouples metrics in its RL; RAD factorizes advantages over lateral and longitudinal actions.
- [[sources/adaptive-wam.md]]: reads a single conditional forward pass of a video DiT for planning; DriveWLA is another single-pass reader.
- [[sources/driveva.md]], [[sources/policy-world-model.md]], [[sources/unidrivevla.md]]: rows in its zero-shot table.
- [[sources/raf-vla.md]]: the same month, the opposite design choice. RAF-VLA keeps a future-representation loss and drops the video model; DriveWLA keeps the video model and drops the loss.
- Not ingested: **DriveWorld-VLA** (91.3 / 86.8), **UNIVERSE**, **OpenDriveVLA**, **EponaV2**'s flow-GRPO, **Flow-GRPO** itself.

---

## Open Questions

- **Are the future tokens predictions?** Probe them against the logged future latents, or shuffle the history frames at test time.
- **Would a video loss on the World Expert help or hurt?** The paper trains it by the trajectory loss alone; DriveLaW's two-stage recipe and SimWAM's joint video loss are the alternatives.
- **Is the bidirectional World↔Language gain specific to a single-pass world stream?** Run the same mask with iterative video denoising (BrainWAM's setting).
- **Does the 41.47 HUGSIM score need RL?** Report the imitation checkpoint (90.8) on HUGSIM.
