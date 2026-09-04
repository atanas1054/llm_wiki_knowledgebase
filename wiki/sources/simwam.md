---
title: "SimWAM: A Simple World Action Model for End-to-End Autonomous Driving"
type: source-summary
sources: [raw/papers/SimWAM_ A Simple World Action Model for End-to-End Autonomous Driving.md]
related: [sources/adaptive-wam.md, sources/brainwam.md, sources/foresight.md, concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/foundation-backbones-for-ad.md, concepts/rl-for-ad.md, concepts/nuscenes-waymo-evals.md, concepts/diffusion-planner.md, sources/drivewam.md, sources/driveva.md, sources/epona.md, sources/policy-world-model.md, sources/drivevla-w0.md, sources/flare.md, sources/explorevla.md, sources/futuresightdrive.md, sources/drivedreamer-policy.md, sources/dreameraD.md, sources/recogdrive.md, sources/driving-wm-counterfactuals.md, concepts/counterfactual-prediction.md]
created: 2026-08-17
updated: 2026-09-04
confidence: high
---

**Paper**: SimWAM: A Simple World Action Model for End-to-End Autonomous Driving
**Authors**: Zongchuang Zhao, Xin Zhou, Tianyang Xu, Zhengyang Sun, Kaixuan Zhou, Honglin Li, Dingkang Liang, Xiang Bai
**Org**: Huazhong University of Science & Technology + Dongfeng Research & Development Institute
**arXiv**: 2608.07468v2
**Code**: https://github.com/H-EmbodVis/SimWAM/ (code and weights released)

---

## Summary

SimWAM attacks the central cost of the world-action paradigm: **imagine-then-act planners put video synthesis inside the real-time loop**. Its claim is that explicit future generation at test time is unnecessary — future-video prediction is valuable as a *training-time supervision signal* only.

The architecture is deliberately plain. A pretrained video expert (Wan2.2-5B DiT + its VAE + T5) and a lightweight action expert (1.02B DiT, hidden 1024) are co-trained under joint flow matching, sharing no parameters and interacting only through a shared attention stream. An **isolated attention mask** lets both the future-frame tokens and the action tokens attend to the current-observation latents $z(o_t)$ while keeping them mutually invisible. Because the action tokens never see future-frame tokens, inference simply drops the future-frame branch and predicts the trajectory directly. A second stage converts the deterministic flow ODE into a marginal-preserving SDE (Flow-GRPO style) and reinforces the action expert with GRPO against the NAVSIM PDM reward, updating only LoRA adapters.

Results: **91.5 PDMS on NAVSIM navtest** with a single front camera, at 518 ms latency; zero-shot to nuScenes with the **lowest average collision rate (0.04%)** in its table without any nuScenes supervision. The headline ablation is that video co-training alone lifts an action-only baseline from 86.6 to 90.3 PDMS, and RL adds a further 1.2.

---

## Core Idea: Training-Time World Modeling, Direct Inference

![[x1 40.png|SimWAM achieves the best PDMS with substantially lower latency than world-model-based planners on NAVSIM]]

**Figure 1**: PDMS versus latency. SimWAM sits at the top-left — best PDMS among the plotted world-model planners with substantially lower latency.

Existing driving WAMs factorize planning as an integral over generated future latents:

$$p_\theta(a_{t+1:t+H}\mid o_t,s_t,l)=\int p_\theta(z_{t+1:t+N}\mid o_t,s_t,l)\,p_\theta(a_{t+1:t+H}\mid o_t,s_t,l,z_{t+1:t+N})\,\mathrm{d}z_{t+1:t+N}$$

SimWAM collapses this to a direct policy interface:

$$p_\theta(a_{t+1:t+H}\mid o_t,s_t,l)=p_\theta\big(a_{t+1:t+H}\mid z(o_t),s_t,l\big)$$

The traffic-dynamics prior lives in $z(o_t)$, shaped during training by the future-video objective. The intellectual antecedent is **Fast-WAM** ("do world action models need test-time future imagination?"), which argued that video co-training helps through training-time representation learning rather than test-time imagination; SimWAM is the driving-domain instantiation of that claim.

---

## Method

![[x2 38.png|Overview of SimWAM: joint video-action training with an isolated attention mask]]

**Figure 2**: During training the video and action DiTs are jointly optimized for future-frame generation and trajectory prediction via shared attention, while the isolated mask prevents action tokens from accessing future-frame tokens. During inference and RL, the model predicts trajectories directly without generating future frames.

### Problem setup

Input: front-camera observation $o_t$, ego state $s_t$ (velocity, acceleration, yaw rate), navigation command $l$. Output: $a_{t+1:t+H}$ with $a_i=(x_i,y_i,\theta_i)$ — 8 waypoints over 4 s at 2 Hz.

### Video expert

Wan2.2-5B video DiT with its video VAE and T5 text encoder. The VAE maps each frame to latent tokens; the navigation command enters through T5 cross-attention. The **current frame is a clean condition**; the $N=8$ future frames are noised and reconstructed with flow matching. This is the stock video-generation objective — no driving-specific prediction module is added.

### Action expert

A lightweight DiT (hidden $d_a=1024$, 1.02B) conditioned on $c=\{z(o_t), s_t, l\}$, with a small MLP embedding the ego state. It predicts the trajectory velocity field $v_{\theta_a}(a^\tau_{t+1:t+H},\tau,c)$ under rectified flow; integrating the ODE maps noise to a planned trajectory.

### Isolated attention mask

The shared attention stream contains $z(o_t)$, the future-frame latents $z_{t+1:t+N}$, and the action tokens. Both future-frame and action tokens attend to $z(o_t)$; **they cannot see each other**. This single structural change is what makes the future-frame branch droppable at inference — and it is also what makes RL possible without future-frame rollout.

### Co-training objective

$$\mathcal{L}=\mathcal{L}^{\text{act}}_{\text{FM}}+\lambda\,\mathcal{L}^{\text{vid}}_{\text{FM}},\qquad \lambda=1$$

with the standard rectified-flow loss $\mathcal{L}_{\text{FM}}=\mathbb{E}\big[\|v_\theta(x_\tau,\tau,c)-(\epsilon-x)\|_2^2\big]$ on the trajectory and on the future-frame latents respectively.

### Reinforcement learning (ODE → SDE + GRPO)

The deterministic flow ODE cannot explore alternative maneuvers and has no tractable transition density. Following Flow-GRPO, SimWAM substitutes a marginal-preserving SDE:

$$\mathrm{d}x_{\tau}=\Big[v_{\theta}(x_{\tau},\tau)+\tfrac{\sigma_{\tau}^{2}}{2\tau}\big(x_{\tau}+(1{-}\tau)\,v_{\theta}(x_{\tau},\tau)\big)\Big]\mathrm{d}\tau+\sigma_{\tau}\,\mathrm{d}w,\qquad\sigma_{\tau}=a\sqrt{\tfrac{\tau}{1-\tau}}$$

Each Euler-Maruyama step gives an isotropic Gaussian transition with tractable log-likelihood for importance sampling. $G=8$ candidates per scenario are scored by the compositional NAVSIM PDM reward, group-relative advantages drive a clipped policy update, and **only rank-32 LoRA adapters ($\alpha=16$) on the action expert's attention projections are updated** — preserving the distilled motion prior. RL is run on **hard navtrain scenes only** (imitation PDMS below 90).

---

## Results

### NAVSIM navtest (Table 1)

C = camera, L = LiDAR.

| Method | Reference | Sensors | NC ↑ | DAC ↑ | EP ↑ | TTC ↑ | C ↑ | PDMS ↑ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Human Agent | – | – | 100.0 | 100.0 | 87.5 | 100.0 | 99.9 | 94.8 |
| *Traditional E2E planners* | | | | | | | | |
| UniAD | CVPR'23 | 6×C | 97.8 | 91.9 | 78.8 | 92.9 | 100.0 | 83.4 |
| TransFuser | TPAMI'22 | 3×C+L | 97.7 | 92.8 | 79.2 | 92.8 | 100.0 | 84.0 |
| ARTEMIS | RA-L'25 | 3×C+L | 98.3 | 95.1 | 81.4 | 94.3 | 100.0 | 87.0 |
| WorldRFT | AAAI'26 | 3×C | 97.8 | 96.8 | 81.7 | 94.0 | 100.0 | 87.8 |
| DiffusionDrive | CVPR'25 | 3×C+L | 98.2 | 96.2 | 82.2 | 94.7 | 100.0 | 88.1 |
| WoTE | ICCV'25 | 3×C+L | 98.5 | 96.8 | 81.9 | 94.9 | 99.9 | 88.3 |
| SeerDrive | NeurIPS'25 | 3×C+L | 98.4 | 97.0 | 83.2 | 94.9 | 99.9 | 88.9 |
| *VLM-based planners* | | | | | | | | |
| ImagiDrive | ICRA'26 | 1×C | 98.6 | 96.2 | 80.5 | 94.5 | 100.0 | 87.4 |
| Vega | arXiv'26 | 1×C | 98.9 | 95.3 | 81.6 | 96.1 | 100.0 | 87.9 |
| AutoVLA | NeurIPS'25 | 3×C | 98.4 | 95.6 | 81.9 | 98.0 | 99.9 | 89.1 |
| DriveDreamer-Policy | arXiv'26 | 3×C | 98.4 | 97.1 | 83.5 | 95.1 | 100.0 | 89.2 |
| UniWorldVLA | arXiv'26 | 1×C | 98.7 | 96.7 | 83.2 | 96.1 | 100.0 | 89.4 |
| DriveVLA-W0 | ICLR'26 | 1×C | 98.7 | 99.1 | 83.3 | 95.3 | 99.3 | 90.2 |
| ExploreVLA | ECCV'26 | 1×C | 98.8 | 98.4 | 83.5 | 96.5 | 99.9 | 90.4 |
| ReCogDrive | ICLR'26 | 1×C | 97.9 | 97.3 | **87.3** | 94.9 | 100.0 | 90.8 |
| SGDrive | CVPR'26 | 1×C | 98.6 | 97.8 | 85.8 | 96.2 | 100.0 | 91.1 |
| *World-model-based planners* | | | | | | | | |
| Epona | ICCV'25 | 1×C | 97.9 | 95.1 | 80.4 | 93.8 | 99.9 | 86.2 |
| PWM | NeurIPS'25 | 1×C | 98.6 | 95.9 | 81.8 | 95.4 | 100.0 | 88.1 |
| DriveLaW | CVPR'26 | 1×C | **99.0** | 97.1 | 81.3 | 96.7 | 100.0 | 89.1 |
| DriveWAM | arXiv'26 | 1×C | 98.3 | 98.1 | 84.3 | 95.2 | 100.0 | 90.1 |
| **SimWAM (ours)** | – | **1×C** | 98.4 | **98.7** | 86.4 | 95.5 | 100.0 | **91.5** |

Best DAC (98.7) and best EP among world-model planners; +1.4 over [[sources/drivewam.md]] and +2.4 over DriveLaW under the same single-camera setting. Note the DriveWAM row reproduces exactly the numbers recorded during that paper's ingest — a useful cross-paper consistency check.

### Zero-shot nuScenes open-loop (Table 6)

\* = front camera only. "Finetune ✗" = evaluated directly from NAVSIM training.

| Method | Finetune | Input | Aux. supervision | L2 1s | L2 2s | L2 3s | **L2 Avg ↓** | Col 1s | Col 2s | Col 3s | **Col Avg ↓** |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ST-P3 | ✓ | Camera | Map&Box&Depth | 1.33 | 2.11 | 2.90 | 2.11 | 0.23 | 0.62 | 1.27 | 0.71 |
| UniAD | ✓ | Camera | Map&Box&Motion | 0.48 | 0.96 | 1.65 | 1.03 | 0.05 | 0.17 | 0.71 | 0.31 |
| OccNet | ✓ | Camera | 3D-Occ&Map&Box | 1.29 | 2.13 | 2.99 | 2.14 | 0.21 | 0.59 | 1.37 | 0.72 |
| OccWorld | ✓ | Camera | 3D-Occ | 0.52 | 1.27 | 2.41 | 1.40 | 0.12 | 0.40 | 2.08 | 0.87 |
| VAD-Tiny | ✓ | Camera | Map&Box&Motion | 0.60 | 1.23 | 2.06 | 1.30 | 0.31 | 0.53 | 1.33 | 0.72 |
| VAD-Base | ✓ | Camera | Map&Box&Motion | 0.54 | 1.15 | 1.98 | 1.22 | 0.04 | 0.39 | 1.17 | 0.53 |
| GenAD | ✓ | Camera | Map&Box&Motion | 0.36 | 0.83 | 1.55 | 0.91 | 0.06 | 0.23 | 1.00 | 0.43 |
| Doe-1 | ✓ | Camera\* | QA | 0.50 | 1.18 | 2.11 | 1.26 | 0.04 | 0.37 | 1.19 | 0.53 |
| Epona | ✓ | Camera\* | None | 0.61 | 1.17 | 1.98 | 1.25 | 0.01 | 0.22 | 0.85 | 0.36 |
| DriveVA | ✗ | Camera\* | None | 0.33 | 0.76 | **1.43** | **0.84** | 0.00 | 0.07 | 0.12 | 0.06 |
| DriveWAM | ✗ | Camera\* | None | **0.28** | 0.81 | 1.80 | 0.96 | 0.00 | 0.05 | 0.14 | 0.06 |
| **SimWAM (ours)** | ✗ | Camera\* | None | 0.29 | 0.82 | 1.77 | 0.96 | 0.00 | **0.03** | **0.11** | **0.04** |

SimWAM's collision rate is the lowest in the table despite no nuScenes supervision. Its L2 (0.96) trails DriveVA (0.84); the paper argues L2 rewards agreement with dataset-specific expert trajectories while collision rate more directly measures safe interaction.

This table is independently valuable to the wiki: it supplies **absolute zero-shot numbers for [[sources/driveva.md]]** (0.84 L2 / 0.06 collision), which DriveVA's own paper reported only as percentage improvements over PWM.

---

## Ablations

### Component analysis (Table 2)

| Configuration | NC | DAC | EP | TTC | PDMS |
|---|---:|---:|---:|---:|---:|
| Action-only | 97.6 | 95.7 | 81.7 | 92.6 | 86.6 |
| + Video | 98.7 | 98.0 | 83.9 | 95.9 | 90.3 |
| + RL | 98.4 | 98.7 | **86.4** | 95.5 | **91.5** |

Video co-training is the dominant contributor: **+3.7 PDMS**, improving every sub-metric. RL adds +1.2 more, concentrated in EP (83.9 → 86.4) and DAC, with small regressions in NC and TTC — the usual progress/safety trade.

### Attention mask (Table 3) — the paper's most consequential result

| Mask | NC | DAC | EP | TTC | PDMS |
|---|---:|---:|---:|---:|---:|
| Bidirectional | 98.4 | 98.0 | 84.7 | 95.1 | 90.2 |
| Action → video | 98.5 | 97.8 | 84.3 | 95.5 | 90.1 |
| **Isolated** | **98.7** | 98.0 | 83.9 | **95.9** | **90.3** |

Letting the action expert attend to future-frame tokens gives **no measurable benefit** (90.2 and 90.1 versus 90.3), while forcing future-frame instantiation at inference. The isolated mask also yields the best NC and TTC. This is a direct empirical challenge to the imagine-then-act premise shared by DriveVA, DriveWAM, FSDrive, and PWM. Note the spread is only 0.2 PDMS with no seed variance reported, so the honest reading is "future conditioning is not necessary," not "future conditioning is harmful."

### Video backbone flexibility (Table 4)

| Video model | NC | DAC | EP | TTC | PDMS |
|---|---:|---:|---:|---:|---:|
| LTX-Video | 98.1 | 97.2 | 83.1 | 94.3 | 88.7 |
| Wan2.1-1.3B | 98.6 | 98.1 | 84.0 | 95.9 | 90.2 |
| Cosmos-Predict2.5 | 98.7 | 98.0 | **84.2** | **96.0** | **90.4** |
| Wan2.2-5B | 98.7 | 98.0 | 83.9 | 95.9 | 90.3 |

The backbone is swappable without touching the action expert or inference pipeline. Two findings worth separating: **prior quality matters** (lightweight LTX-Video loses 1.6 PDMS), but **scale does not** — Wan2.1-1.3B (90.2) essentially matches Wan2.2-5B (90.3) at a quarter the size. Cosmos-Predict2.5, pretrained on driving video, is best, suggesting domain relevance beats raw capacity.

### Action expert scaling (Table 5)

| Action DiT | NC | DAC | EP | TTC | PDMS |
|---|---:|---:|---:|---:|---:|
| 0.21 B | 98.6 | 97.8 | 84.0 | 95.4 | 89.9 |
| 0.45 B | 98.6 | 97.9 | 83.8 | 95.9 | 90.1 |
| 1.02 B | 98.7 | 98.0 | 83.9 | 95.9 | 90.3 |

Monotone but shallow: 5× the parameters buys 0.4 PDMS. Combined with Table 4, both scaling axes are flat in this regime — the 0.21B action expert with Wan2.1-1.3B would be a much cheaper system at roughly 89.9–90.2.

### Exploration sampler (Table 7)

| Sampler | NC | DAC | EP | TTC | PDMS |
|---|---:|---:|---:|---:|---:|
| Random noise | 97.7 | 98.4 | **88.0** | 94.1 | 91.3 |
| SDE | **98.4** | **98.7** | 86.4 | **95.5** | **91.5** |

Naive random perturbation explores aggressively — best EP in the paper (88.0, above even ReCogDrive's 87.3) — but degrades NC and TTC through less structured maneuvers. The marginal-preserving SDE trades 1.6 EP for better safety and a higher aggregate.

### RL training dynamics (Figure 3)

![[x3 34.png|RL training dynamics: hard-subset training outperforms full-navtrain training]]

**Figure 3**: The star marks the imitation checkpoint. Training on the hard subset (imitation PDMS below 90) consistently beats training on all navtrain scenes, peaking at 91.5 at 15k steps. Easy scenes are already handled by imitation and dilute the reward signal. Both curves decline slightly past 15k steps — diminishing returns, and an implicit early-stopping dependency.

### Future-video target (Table 8)

| Target | NC | DAC | EP | TTC | PDMS |
|---|---:|---:|---:|---:|---:|
| 4 f, 2 s, 2 Hz | 98.6 | 97.7 | 83.9 | 95.5 | 89.9 |
| 4 f, 4 s, 1 Hz | 98.7 | 97.9 | 84.2 | 95.6 | 90.2 |
| 8 f, 4 s, 2 Hz | 98.7 | 98.0 | 83.9 | 95.9 | **90.3** |

**Temporal coverage matters more than frame density**: halving the horizon (4 s → 2 s) costs 0.4 PDMS, while halving the frame rate at a fixed 4 s horizon costs only 0.1.

### Input resolution (Table 9)

| Resolution | NC | DAC | EP | TTC | PDMS | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| 192×352 | 98.2 | 97.1 | 83.0 | 94.9 | 88.9 | 509 |
| 384×672 | 98.7 | 98.0 | 83.9 | 95.9 | 90.3 | 518 |
| 768×1344 | 98.7 | 98.1 | **84.3** | **96.1** | **90.6** | 573 |

Resolution is nearly free (+1.4 PDMS for +9 ms going to 384×672), because latency is dominated by the sampler rather than the encoder. The chosen 384×672 is not the accuracy optimum — 768×1344 is 0.3 higher for 55 ms more.

### Sampling steps (Table 10)

| Steps | NC | DAC | EP | TTC | PDMS | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 97.4 | 91.3 | 79.1 | 83.3 | 68.9 | 115 |
| 5 | 98.6 | 97.9 | 84.0 | 95.6 | 90.1 | 297 |
| 10 | 98.7 | 98.0 | 83.9 | 95.9 | **90.3** | 518 |
| 20 | 98.6 | 98.0 | 83.9 | 95.8 | 90.2 | 968 |

The sampler has converged by 10 steps. One step collapses (68.9) — unlike [[sources/driveva.md]] (2 steps) and truncated-diffusion planners, SimWAM's action flow needs a real step budget. The 5-step setting is the practical sweet spot: 90.1 PDMS at 297 ms, 43% cheaper than the default for 0.2 PDMS.

The latency scaling (≈45 ms per step plus ≈70 ms fixed) implies the action expert re-attends to the observation representation at every denoising step; the paper does not break down where the fixed cost sits, so how much of the Wan2.2-5B stack actually runs at inference is not fully specified.

### Qualitative

![[x4 32.png|Qualitative comparison of imitation-trained and reinforced models on two navtest scenarios]]

**Figure 4**: Ours-IL versus Ours-RL on two navtest scenes. The imitation model is conservative, advancing only a short distance at an intersection and along a narrow street; after RL the model completes more of each maneuver while staying in the drivable area with safe clearance. Red ellipses mark the progress differences — a visual counterpart to the EP gain in Table 2.

---

## Implementation Details

- **Video expert**: Wan2.2-5B + its VAE + T5 encoder. **Action expert**: DiT, hidden 1024, 1.02B.
- **Input**: single front camera at 384×672. **Output**: 8 waypoints over 4 s at 2 Hz; video expert predicts the matching 8 future frames.
- **Joint training**: AdamW, cosine schedule, initial lr $10^{-4}$, 100 epochs, $\lambda=1$.
- **RL**: rank-32 LoRA ($\alpha=16$) on the action expert's attention projections only; $G=8$ trajectories per scenario; lr $5\times10^{-5}$; hard navtrain subset (imitation PDMS < 90); peak at 15k steps.
- **Data**: navtrain 103,288 scenes; navtest 12,146 scenes. Latency measured on a single A100.

---

## Limitations

1. **"New state of the art in end-to-end planning" is comparison-scope limited.** Table 1 omits every wiki entry above 91.5: CLEAR (93.7), DriveSuprim (93.5), Drive-JEPA (93.3), HybridDriveVLA (92.1), DynVLA/Reasoning-VLA (91.7), and DiffusionDriveV2 (91.2) ties it closely at 91.2. Within its own table the claim holds (SGDrive 91.1 is next), and the WAM-class comparison is genuinely current. See [[concepts/navsim-benchmark.md]].
2. **The isolated-mask conclusion rests on a 0.2 PDMS spread.** Bidirectional 90.2, action→video 90.1, isolated 90.3 — no seed variance, no confidence intervals. The efficiency argument for the isolated mask is solid; the accuracy argument is within noise. The defensible claim is that test-time future conditioning is *unnecessary*, not that it is worse.
3. **Latency framing is not fully apples-to-apples.** Figure 1's advantage over imagine-then-act planners is real, but 518 ms (or 297 ms at 5 steps) is still far above non-VLM planners in the wiki — DiffusionDrive runs at 45 FPS, HAD at 30.4 FPS, OneDrive at 156 ms. SimWAM is efficient *for a 5B-video-backbone WAM*, not efficient in absolute terms.
4. **Cross-paper discrepancy on DriveWAM's nuScenes results.** Table 6 attributes zero-shot nuScenes numbers (0.96 L2 / 0.06 collision) to DriveWAM, but the DriveWAM v1 clipping ingested in this wiki reports **no nuScenes evaluation at all** — only NAVSIM and PhysicalAI-AV. Either SimWAM reproduced these itself without saying so, or it cites a later DriveWAM revision. Treat the DriveWAM row in Table 6 as unverified against [[sources/drivewam.md]].
5. **Single benchmark for closed-loop-style evaluation.** NAVSIM-v1 only — no NAVSIM-v2/EPDMS, no navhard, no Bench2Drive or HUGSIM. nuScenes is open-loop ([[concepts/nuscenes-waymo-evals.md]] caveats apply). The extended-comfort and lane-keeping behavior that NAVSIM-v2 exposes is untested.
6. **RL depends on early stopping and a hand-set difficulty threshold.** Both curves in Figure 3 decline after 15k steps, and the hard subset is defined by a fixed PDMS < 90 cut. No sensitivity analysis for the threshold, and the peak is selected on the evaluation benchmark.
7. **No comfort reported in the ablations.** Tables 2–5 and 7–10 omit the C column even though it enters PDMS; only Table 1 reports it (100.0). Sub-metric trade-offs involving comfort cannot be checked.
8. **The video expert's inference cost is unquantified.** The paper says the future-frame decoder "could be discarded after training," but never reports the parameter count or FLOPs actually executed at inference, and latency scales with sampling steps in a way that suggests the shared stack is re-entered per step.
9. **Several Table 1 baselines are absent from this wiki** (UniWorldVLA 89.4, SeerDrive 88.9, ImagiDrive 87.4, WorldRFT 87.8), so their numbers are transcribed but not independently corroborated here. Two have since been ingested and **both confirm SimWAM's transcription exactly**: [[sources/sgdrive.md]] reports 91.1, and [[sources/drivelaw.md]] reports 89.1 with matching sub-scores (NC 99.0 / DAC 97.1 / EP 81.3 / TTC 96.7). SimWAM's table is accurate where it can be checked.

---

## Key Cross-References

- **World-model pattern**: [[concepts/world-model-for-ad.md]] — SimWAM is the wiki's cleanest "training-time-only world model that still uses a video generative backbone as the representation source," and its mask ablation is the sharpest test yet of whether test-time imagination pays.
- **Direct WAM rivals**: [[sources/drivewam.md]] (90.1, imagine-then-act, inverse dynamics) and [[sources/driveva.md]] (90.9, joint denoising) — all three fine-tune Wan-family video DiTs; SimWAM is the only one that drops future generation at inference.
- **Same conclusion, different mechanism**: [[sources/drivevla-w0.md]] and [[sources/flare.md]] also use world modeling purely as training-time signal, but from VLM/DINOv2 backbones rather than a video generative model.
- **RL**: [[concepts/rl-for-ad.md]] — Flow-GRPO SDE exploration plus LoRA-only updates on the action expert, and hard-subset scene selection that echoes PlannerRFT's below-90 subset finding.
- **A caveat on that backbone swap**: [[sources/adaptive-wam.md]] shows that *readout depth* within a single video DiT is worth up to 4.80 PDMS, with the mid-network block beating the final one. SimWAM's four-way prior comparison reads one depth per backbone and the spread across all four priors is 1.7 — smaller than the within-backbone depth effect. That does not invalidate the conclusion that prior scale barely matters, but it means part of the 1.7 could be readout-point mismatch across architectures of different depths rather than prior quality. Matching relative depth would settle it.
- **Backbones**: [[concepts/foundation-backbones-for-ad.md]] — the backbone-swap table (LTX-Video / Wan2.1-1.3B / Cosmos-Predict2.5 / Wan2.2-5B) is the wiki's only controlled comparison of video priors under a fixed planner.
- **Unexpected ally**: [[sources/drivelaw.md]] is the imagine-then-act WAM SimWAM beats by the widest margin (89.1 vs 91.5), yet its own denoising-step ablation independently supports SimWAM's thesis — conditioning on latents from the *first* denoising step scores 89.1 while nearly-clean generated futures at t=10 collapse the policy to 23.2 PDMS. Two papers, opposite designs, same conclusion about test-time imagination.
- **Nearest competitor**: [[sources/sgdrive.md]] (91.1 PDMS) is the runner-up in SimWAM's own table and takes the opposite route to world knowledge — supervised structured symbolic state (occupancy, agent boxes, goal pose) inside a 2B VLM rather than a video generative prior. Both reach ~91 with a single front camera and no future generation at inference, from entirely different supervision. SGDrive needs 3D and occupancy annotations; SimWAM needs only raw video.
- **The clearest cost contrast**: [[sources/foresight.md]] is the imagine-then-act design taken to its limit — a frozen 2.5B generator as the planner's primary encoder, run to a finished future at inference — and it reaches **89.3 PDMS at 900 ms, 870 ms of which is the world model**. SimWAM reaches 91.5 at 518 ms with the same future generated only during training. ForeSight's own Table 3 also prices the mechanism SimWAM's mask removed: adding the foundation world model under vanilla attention is worth **+0.3 PDMS**. Two caveats in ForeSight's favour: its Table 7 shows a planner on generated futures *alone* still scores 88.2, and its Table 5 disputes DriveLaW's denoising sweep.
- **Where the mask ablation stops applying**: [[sources/brainwam.md]] runs the analogous test with a *third* stream in the pool and gets the opposite sign. SimWAM's bidirectional mask (video + action, no VLM) scores 90.2 against 90.3 isolated — no effect. BrainWAM's Tri-MoT (VLM + video + action, symmetric and unmasked) scores **87.8, below its own WAM-only branch at 88.1**, diagnosed as modality competition: action tokens take the clean pretrained VLM stream as a shortcut and underuse the still-denoising video stream. The two results are consistent and jointly bound the claim — joint attention over video and action is harmless, but adding a clean semantic stream to a denoising one is not. See [Where MoT Breaks](../concepts/mixture-of-experts.md#modality-competition).
- **The escape hatch, tested**: [[sources/driving-wm-counterfactuals.md]] attacks the one defence SimWAM's mask ablation leaves standing — that imagined futures must matter for counterfactual evaluation even if they do not help NAVSIM planning. On a CARLA benchmark with matched counterfactual ground truth, action-conditioned generation from Vista and DrivingWorld recovers only 0.38 / 0.31 of the event signal, so the machinery is not delivering counterfactuals either. It does not test SimWAM's own setting (comparing candidate maneuvers before acting is rung 2, not rung 3), but it removes the retrospective version of the argument.
