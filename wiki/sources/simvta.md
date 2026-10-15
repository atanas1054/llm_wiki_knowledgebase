---
title: "SimVTA: Simulation-Grounded Video-Trajectory Alignment for Driving World Action Models"
type: source-summary
sources: ["raw/papers/SimVTA_ Simulation-Grounded Video-Trajectory Alignment for Driving World Action Models.md"]
related: [sources/drivor.md, concepts/world-model-for-ad.md, concepts/rl-for-ad.md, concepts/data-driven-simulators.md, concepts/navsim-benchmark.md, concepts/hugsim-benchmark.md, concepts/evaluation-variance.md, concepts/inference-latency.md, sources/epona.md, sources/dreameraD.md, sources/foresight.md, sources/walt.md, sources/simwam.md, sources/simscale.md, sources/rad.md, sources/plannerrft.md, sources/recogdrive.md, sources/driveva.md, sources/wa-jepa.md, sources/mm-future.md, sources/foredrive.md, sources/world4scorer.md, sources/physwam.md, sources/learning-to-drive-from-a-world-model.md, research-directions.md]
created: 2026-10-14
updated: 2026-10-14
confidence: medium
---

# SimVTA: Simulation-Grounded Video-Trajectory Alignment for Driving World Action Models

**Paper**: SimVTA: Simulation-Grounded Video-Trajectory Alignment for Driving World Action Models
**Authors**: Changhui Jing, Haochen Tian (project lead), Hongchen Li, Tianyu Li, Caojun Wang, Zihao Zhang, Yanhao Liu, Chengen Xie, Bailin Li, Hao Zhao, Zhenjie Yang (correspondence)
**Orgs**: Shanghai Innovation Institute; OpenDriveLab at The University of Hong Kong; Li Auto; Tongji University; Tsinghua University
**arXiv**: 2610.13781v1 (October 2026)
**Code**: project page opendrivelab.com/SimVTA and github.com/OpenDriveLab/SimVTA ("results and demo"; a code release is not stated)
**Source**: `raw/papers/SimVTA_ Simulation-Grounded Video-Trajectory Alignment for Driving World Action Models.md`. The clipping has all text, all 15 tables and all 7 figures (`raw/assets/1 1.png`, `2.png` … `7.png`). Three in-text table references are off by one (the shared-encoder ablation is cited as Table 6, the video-GRPO comparison as Table 8 and the shuffle control as Table 10; they are Tables 5, 7 and 9).

**Confidence: medium.** The ablation set is unusually complete for this wiki (a gradient-path control, a correspondence control, an offline-cache control, a budget control) and four configurations are repeated over three seeds. But the control the central claim needs, a video loss on the *logged* future kept during RL, is not run; the initialization's video quality is not reported, so the FID/FVD gain may be recovery from what RL broke; the HUGSIM result has no matched pre-RL run; and the v2 table compares across evaluators. One backbone (Epona).

---

## What It Is

**RL post-training for a world-action model in which every trajectory the policy explores gets its own video target, rendered by a 3DGS simulator.** The model is Epona (spatiotemporal transformer encoder, TrajDiT trajectory head, VideoDiT next-frame head; [[sources/epona.md]]), adapted to NAVSIM at 2 Hz (86.2 PDMS). Post-training repeats four steps:

1. **Sample** $G=32$ trajectories per scene from the flow-matching trajectory head with Flow-GRPO's SDE.
2. **Execute** each with NAVSIM's PDM executor and score it: $R=\mathrm{NC}\cdot\mathrm{DAC}\cdot(10\,\mathrm{EP}+5\,\mathrm{TTC}+2\,\mathrm{C})/17$ (ReCogDrive's weights).
3. **Render** the front-view future along the executed path for the $K=4$ highest-reward candidates that pass safety and pose-support checks, using World Engine (OpenDriveLab's 3DGS simulator over reconstructed navtrain scenes; not ingested).
4. **Update** with trajectory GRPO on all 32 candidates, a behaviour-cloning anchor to the frozen initial TrajDiT, and a flow-matching video loss on the rendered candidates whose teacher-forced prediction error is at or above the scene mean ("Q2"). Video gradients reach the shared encoder, scaled by 0.2.

At inference the model predicts a trajectory directly: no simulator, no video, no reranking.

**Results**

| Benchmark | SimVTA | Matched comparison |
|---|---:|---|
| NAVSIM-v1 PDMS (navtest) | **91.3** (3 seeds: 91.3 / 91.4 / 91.3) | Epona initialization 86.2; trajectory GRPO + BC without video (Exp0) 89.1 |
| NAVSIM-v2 EPDMS (navtest, v2.2 devkit) | 88.2 (corrected-like, residual −0.76) | Epona initialization 85.0 (re-evaluated, −1.70) |
| HUGSIM, 345 "pre-challenge" scenarios | 51.5 RC / **38.8 HD-Score** | DrivoR 49.8 / 35.7 (copied from its paper); no pre-RL run |
| FID / FVD along logged trajectories | 13.1 / 46.3 | Exp0 20.4 / 105.4; World Engine renders 7.6 / 21.8 |
| Training | 16 H20 GPUs, ZeRO-2 with CPU offload | Steps, wall-clock and rendering cost not reported |

---

## Key Takeaways

- **Of the +5.1 PDMS, +2.9 is standard trajectory RL and +2.2 is the video loss.** Exp0 (Flow-GRPO with the PDM reward and the BC anchor, no video) is 89.1, already above [[sources/dreameraD.md]]'s 88.7 on the same Epona initialization. Adding the Q2 video loss with gradients into the encoder gives 91.3. The abstract's "+5.1" is the whole recipe; the paper's Appendix F says so.
- **The video loss helps only through the shared encoder.** Restricting its gradients to VideoDiT (Exp1) leaves PDMS at 89.1, identical to no video. With encoder gradients, 91.3. This is the training-time-only pattern ([[sources/simwam.md]], DriveVLA-W0): the video head is discarded at inference and its value is what it does to the representation. Epona's own pretraining ablation found the same thing at larger scale (−8.1 PDMS without joint video training).
- **Wrong video targets are much worse than none.** Shuffling the rendered videos among a scene's Q2 candidates (Exp3) gives 83.1: 8.2 below matched targets, 6.0 below no video and 3.1 below the initialization, with NC −3.0, DAC −5.4 and TTC −4.2 against Exp0. ForeDrive found the consumption-side version (a wrong predicted future costs 2.4, none costs 1.4); this is the supervision-side version and it is larger. One caution: the shuffle also pairs each rendered history frame with another path's poses inside the encoder's own input, so part of the collapse may be broken frame–pose consistency rather than a wrong future as such.
- **The missing control is the logged future.** Every arm with video uses rendered off-log targets. The obvious alternative, keeping Epona's pretraining loss (logged trajectory, logged video) during GRPO, is never run. So the paper shows that matched targets beat mismatched ones and beat none, but not that *off-log* targets beat the logged one. The offline-cache arm sits between: renders of initial-policy trajectories, cached once, give 90.4, and refreshing them each round gives +0.9 more.
- **Selection effects are small but seed-robust.** Q2 alone 91.3, Q1+Q2 91.0, Q2+Q3 90.9, all four quadrants 90.9, video GRPO on all rendered candidates 91.0, twice the rendering budget 91.2. Across three seeds Exp2 beats Exp5 and Exp7 on every seed, by 0.1–0.5. The mixed-pool arms also change the RL advantage for unsafe candidates, and the number of video-trained candidates differs between arms, as the paper notes.
- **The gain is mostly ego progress.** By the closed form, of Exp0 → Exp2's +2.9, EP accounts for +1.2, NC +0.8, DAC +0.4, TTC +0.4. Initialization → SimVTA: EP +2.5, DAC +1.9, TTC +1.2, NC +0.2. The training reward weights EP 10/17, twice its PDMS weight. On v2, extended comfort falls 67.6 → 60.0 and traffic-light compliance 99.7 → 99.0.
- **On the wiki's ladders 91.3 is mid-table.** About twenty single-pass v1 entries are higher, including SimWAM 91.5 and WA-JEPA 91.8, which the paper lists only in an appendix table, not in Table 1. It is the highest Epona-family entry (Epona 86.2, DreamerAD 88.7, ForeSight 89.3, WALT 89.8). On v2, 88.2 sits in the lower part of the corrected cohort, below EponaV2's 88.9, which is absent from the paper.
- **The HUGSIM result is a different 345-scenario protocol and has no control.** SimVTA beats DrivoR (93.1 PDMS on NAVSIM, 1.8 above it) by 3.1 HD-Score on the same set; the arithmetic below confirms all five rows share one weighting. But the initialization and Exp0 were not run on HUGSIM, so nothing there is attributable to the video loss, as the paper itself says.

---

## Method

![[1 1.png|Figure 1: From logged futures to candidate-matched supervision. (1) WAM pretraining uses the ground-truth (GT) trajectory and its logged future video. (2) Trajectory GRPO explores candidate trajectories using driving rewards, without matched video supervision. (3) World Engine renders reconstructed scenes along selected candidate paths, providing matched supervision independent of the model's predictions.]]

**Figure 1**: Three regimes. (1) Pretraining: the logged trajectory with its logged video ("human reliance", "limited to logged actions"). (2) Trajectory GRPO: reward-guided exploration, but the explored trajectories have no video ("unmatched visual futures"). (3) SimVTA: a 3DGS simulator renders the future of each selected candidate (A, B, C), giving trajectory–video alignment and selective video supervision.

![[2.png|Figure 2: Simulation-grounded video–trajectory alignment. (a) Training overview. (b) Trajectory learning combines GRPO with a behavior-cloning (BC) regularizer. (c) Video supervision selects high-reward, higher-error futures. (d) World Engine renders matched futures along executed paths.]]

**Figure 2**: (a) Images pass through a frozen image encoder and ego motion through a motion encoder into the WAM (Epona's spatiotemporal transformer, the shared encoder). Trajectory samples are scored, the 3D Gaussian renderer produces targets, and a frozen image encoder turns them into latents for the video loss. (b) Trajectory loss = GRPO loss (rewards) + BC loss (frozen teacher planner). (c) The reward × error quadrants; selected videos feed the video loss through a stop-gradient on the target. (d) Executed poses → camera poses → render in the 3D Gaussian scene → matched targets.

### Sampling and execution (§3.1)

$$F=h_\theta(o),\qquad \tau_i\sim\pi_\psi(\cdot\mid F),\quad i=1,\ldots,G;\qquad \tilde\tau_i=\mathcal E(o,\tau_i)$$

$h_\theta$ is the shared encoder over four observed frames and ego-motion history. The PDM executor $\mathcal E$ tracks each planned trajectory; the executed path $\tilde\tau_i$, not the plan, is what gets rendered and what conditions video prediction. Eight waypoints over 4 s; PDM evaluates at 10 Hz, video poses are sampled at 2 Hz.

### Rendering (§3.2, Appendix A)

$$V_i^{\rm sim}=\mathcal S(o,\tilde\tau_i),\qquad V_i^{\rm pred}=W_\phi(h_\theta(o),\tilde\tau_i)$$

A candidate is eligible for rendering only if it executes, stays inside VideoDiT's pose support at every 2 Hz step ($0\le\Delta x\le16$ m, $|\Delta y|\le1.5$ m, $|\Delta\mathrm{yaw}|\le32^\circ$, no reversing), and scores at least 0.999 on NC, DAC, TTC and DDC. Eligible candidates are ranked by $R_i$ and the top $K$ are rendered. These filters do not remove anything from the trajectory loss. A scene with no eligible candidate, or a maximum reward of at most $10^{-6}$, gets no video loss. **Every rendered future in the default recipe is therefore a safe, high-reward one.**

Rendering is online: each sampling round renders the current policy's candidates, so the targets follow the policy as it changes.

### Video loss (§3.3, Appendix B)

Flow matching on image latents of the rendered frames:

$$x_{i,r,t}=t\,z^{\rm sim}_{i,r}+(1-t)\epsilon,\qquad u^\star_{i,r}=z^{\rm sim}_{i,r}-\epsilon$$

$$\mathcal L_{\rm video}(i)=\mathbb E_{t,\epsilon,r}\Big[\tfrac{1}{d_z}\big\|D_\phi(x_{i,r,t},t;c_{\theta,i,r},p_{\theta,i,r})-\mathrm{sg}(u^\star_{i,r})\big\|_2^2\Big]$$

with history condition $c_{\theta,i,r}=h_\theta(o,z^{\rm sim}_{i,<r},\mathrm{sg}(\Delta\tilde\tau_{i,<r}))$ (teacher forcing: preceding *rendered* frames and executed pose changes, a rolling window) and motion embedding $p_{\theta,i,r}=e_\theta(\mathrm{sg}(\Delta\tilde\tau_{i,r}))$. One future frame per candidate per update, cycling $r=1+(s\bmod 8)$.

**Gradient scaling into shared parameters.** With $g_\eta(x)=\mathrm{sg}(x)+\eta(x-\mathrm{sg}(x))$, the loss for candidates allowed to update the encoder uses $g_\eta(c)$ and $g_\eta(p)$, the others use detached conditions:

$$\mathcal L^{\rm sel}_{\rm video}=\frac1Z\sum_i\Big[m^{\rm shared}_i\,\mathcal L^{\rm joint}_{\rm video}(i)+(m^{\rm head}_i-m^{\rm shared}_i)\,\mathcal L^{\rm head}_{\rm video}(i)\Big],\qquad \eta=0.2$$

### Target selection (§3.3, Appendix B–C)

For each valid rendered candidate a detached teacher-forced error at fixed flow time $t_\star=0.5$, shared noise per observation:

$$E^{\rm TF}_i=0.7\sum_{r=1}^{H}\alpha_r\frac{\|e_{i,r}\|_2^2}{d_z}+0.3\sum_{r=2}^{H}\frac{\alpha_r}{\sum_{j\ge2}\alpha_j}\frac{\|\sqrt{\omega_{i,r}}\odot(e_{i,r}-e_{i,r-1})\|_2^2}{d_z},\qquad \alpha_r\propto0.85^{\,r-1},\ H=8$$

where $e_{i,r}$ is the one-step reconstruction error of frame $r$ and $\omega$ up-weights tokens that change between rendered frames (clipped to [0.25, 4]). Candidates at or above the scene's mean error are "high error". With $K=4$, Q2 holds one to three candidates per scene in the usual case.

### Trajectory objective (§3.4, Appendix C)

$$\mathcal L=\mathcal L_{\rm traj}+\lambda_v\,\mathcal L^{\rm sel}_{\rm video}+\lambda_{\rm BC}\,\mathcal L_{\rm anchor},\qquad \lambda_v=\lambda_{\rm BC}=0.1$$

- **Trajectory GRPO.** Flow-GRPO's SDE over 10 denoising steps (noise scale 0.3, floor 0.01). The per-step score is a per-coordinate Gaussian log density with the normalizer dropped, clipped to $[-5,2]$ and averaged. Group-normalized advantages $A_i=(R_i-\bar R)/(s_R+10^{-8})$ weight the scores, discounted by $0.6^{N-1-k}$ over denoising steps. **There is no importance ratio** between current and sampling policies, and no KL term. Gradients reach TrajDiT and the shared encoder.
- **BC anchor.** The frozen initial TrajDiT generates a deterministic ODE chain from the *current* encoder's (detached) features; the student maximizes the same clipped score on that chain's transitions, without the discount.
- Video prediction error never enters the trajectory advantages.

### Implementation (Table 11)

| Parameter | Value |
|---|---|
| Sampled trajectories / rendering budget | $G=32$ / $K=4$ |
| Batch per GPU / default seed | 4 / 1234 |
| Hardware | 16 NVIDIA H20 GPUs |
| Learning rate / AdamW betas | $10^{-5}$ / (0.9, 0.95), halved at steps 300 and 1500 |
| Weight decay | 0.01 |
| Trajectory denoising steps / noise scale | 10 / 0.3 |
| Diffusion-coefficient floor / denoising discount | 0.01 / 0.6 |
| Video loss / BC weights | 0.1 / 0.1 |
| Video gradient scale into shared parameters | $\eta=0.2$ |
| Video-error horizon / temporal discount | $H=8$ / 0.85 |
| Video candidates per scene per update | all valid Q2 |
| Trained future frames per candidate per update | 1 |
| Precision / optimizer | bfloat16 / ZeRO-2, CPU offload |

The total number of steps is not given.

---

## Results

![[3.png|Figure 3: Planning results and video-supervision comparisons. (a) NAVSIM v1. (b) NAVSIM v2 and HUGSIM. (c) Video-supervision configurations and rendering budget on NAVSIM v1.]]

**Figure 3**: (a) Epona 86.2, Exp0 89.1, Exp1 89.1, SimVTA 91.3 (y-axis starts at 80). (b) v2: Epona 85.0 → 88.2; HUGSIM HD-Score: DrivoR 35.7 → SimVTA 38.8. (c) Target selection: Q2 91.3, Q1+Q2 91.0, Q2+Q3 90.9, Q1–Q4 90.9, Q2 at twice the budget 91.2, on an x-axis spanning 90.8–91.4.

### Table 1: NAVSIM v1 navtest

Residual = closed form − reported, computed here. A per-scene-averaged PDMS sits above the closed form of its mean sub-scores, so negative residuals are normal ([[concepts/navsim-benchmark.md#physwam]]).

| Method | Group | NC | DAC | TTC | Comf. | EP | PDMS | Residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| UniAD | E2E | 97.8 | 91.9 | 92.9 | 100.0 | 78.8 | 83.4 | −4.12 |
| DiffusionDrive | E2E | 98.2 | 96.2 | 94.7 | 100.0 | 82.2 | 88.1 | −2.72 |
| WoTE | E2E | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 | −2.18 |
| MeanFuser + BeyondDrive | E2E | 98.5 | 98.4 | 95.1 | 100.0 | 84.2 | 90.3 | −1.74 |
| ResAD | E2E | 98.9 | 97.8 | 94.9 | 100.0 | 87.0 | 90.6 | −1.17 |
| AutoVLA | VLM | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 | −2.92 |
| Uni-World VLA | VLM | 98.7 | 96.7 | 96.1 | 100.0 | 83.2 | 89.4 | −2.19 |
| ExploreVLA | VLM | 98.8 | 98.4 | 96.5 | 99.9 | 83.5 | 90.4 | −1.30 |
| ReCogDrive | VLM | 97.9 | 97.3 | 94.9 | 100.0 | 87.3 | 90.8 | −2.61 |
| SGDrive | VLM | 98.6 | 97.8 | 96.2 | 100.0 | 85.8 | 91.1 | −1.90 |
| LAW | World model | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 | −3.99 |
| Epona (re-evaluated) | World model | 97.8 | 95.0 | 93.9 | 99.9 | 80.3 | 86.2 | −3.29 |
| PWM | World model | 98.6 | 95.9 | 95.4 | 100.0 | 81.8 | 88.1 | −2.53 |
| DreamerAD | World model | 98.0 | 97.2 | 94.3 | 100.0 | 83.1 | 88.7 | −2.41 |
| DriveWAM | World model | 98.3 | 98.1 | 95.2 | 100.0 | 84.3 | 90.1 | −1.90 |
| DriveVA | World model | 99.2 | 97.5 | 98.7 | 100.0 | 83.5 | 90.9 | −1.35 |
| **SimVTA** | World model | 98.0 | 97.2 | 97.0 | 100.0 | 86.7 | **91.3** | −2.51 |

Every row passes the check. The re-evaluated Epona row differs from Epona's published one by 0.1 on four sub-scores (97.9 / 95.1 / 93.8 / 80.4) with the same 86.2. DriveVA's sub-scores are digit-identical to the copy in [[sources/adaptive-wam.md]] that filled DriveVA's truncated table ([[sources/driveva.md]]), **but Adaptive-WAM prints them with 90.5 (DriveVA trained on NAVSIM only) and SimVTA with 90.9** (its caption: "DriveVA also uses CARLA training data", the mixed-data headline). Both aggregates pass the closed-form check (−0.95, −1.35), so one table pairs these sub-scores with the other configuration's score and the arithmetic cannot say which. The VLM baselines' scores are taken from SimWAM's table; **SimWAM's own 91.5 is not in Table 1** and appears only in Table 15.

### Table 2: NAVSIM v2 navtest

Residual = closed form − reported (NAVSIM-v2 formula, [[concepts/navsim-benchmark.md#submetric-residual]]).

| Method | Group | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS | Residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ResAD | E2E | 97.8 | 97.2 | 99.5 | 99.8 | 88.2 | 96.9 | 97.0 | 98.4 | 88.2 | 85.5 | +2.57 |
| DriveSuprim | E2E | 98.4 | 98.6 | 99.6 | 99.8 | 90.5 | 97.8 | 97.0 | 98.3 | 78.6 | 87.1 | +2.67 |
| DiffRefiner | E2E | 98.6 | 98.5 | 99.7 | 99.9 | 87.6 | 98.1 | 98.1 | 98.3 | 87.9 | 87.4 | +3.11 |
| SGDrive | VLM | 98.6 | 94.3 | 99.5 | 99.9 | 86.0 | 97.9 | 96.1 | 98.3 | 85.9 | 86.2 | −0.70 |
| DriveWorld-VLA | VLM | 98.6 | 99.1 | 99.6 | 99.8 | 87.4 | 97.9 | 97.0 | 97.8 | 78.6 | 86.8 | +2.64 |
| ELF-VLA-8B | VLM | 98.9 | 98.1 | 99.4 | 99.8 | 88.5 | 98.4 | 96.9 | 98.3 | 87.2 | 87.1 | +3.09 |
| Epona (re-evaluated) | World model | 97.2 | 95.6 | 99.3 | 99.7 | 88.4 | 96.4 | 96.9 | 97.9 | 67.6 | 85.0 | −1.70 |
| SafeDrive | World model | 99.5 | 99.0 | 99.7 | 99.9 | 88.6 | 98.9 | 97.5 | 98.2 | 81.9 | 87.5 | +4.03 |
| DreamerAD | World model | 98.0 | 97.2 | 99.5 | 99.8 | 87.8 | 97.4 | 97.5 | 98.3 | 72.4 | 87.7 | −1.24 |
| **SimVTA** | World model | 98.5 | 98.5 | 99.8 | 99.0 | 90.8 | 98.5 | 97.5 | 99.0 | 60.0 | **88.2** | **−0.76** |

**The column mixes evaluators.** SimVTA and its Epona re-evaluation were scored with the NAVSIM v2.2 devkit (single-stage EPDMS, LQR tracking, non-reactive agents) and both sit on the corrected side, as do DreamerAD and SGDrive. Six rows are pre-fix-like (+2.6 to +4.0). By closed form, five of those six (SafeDrive 91.5, DiffRefiner 90.5, ELF-VLA 90.2, DriveSuprim 89.8, DriveWorld-VLA 89.4) sit above SimVTA's 87.4, so "88.2" is not the top of a common-scale version of this table. The same-scale margins are +0.5 over DreamerAD and +3.2 over the initialization. A stated v2.2 row agreeing with the residual-sign heuristic is one more confirmation of it. This is not the shared baseline block (no TransFuser, Hydra-MDP++ or human row).

### Table 3: HUGSIM closed loop (345 pre-challenge scenarios)

Baselines are copied from DrivoR's paper (Driving on Registers, Table 2); SimVTA is run "using the same corrected evaluator and controller". Values ×100.

| Method | RC E | RC M | RC H | RC X | RC Avg | HDS E | HDS M | HDS H | HDS X | HDS Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VAD | 51.3 | 31.1 | 25.3 | 26.5 | 31.4 | 36.3 | 9.5 | 8.0 | 11.5 | 13.4 |
| LTF | 67.8 | 35.1 | 26.2 | 40.5 | 38.9 | 58.9 | 18.0 | 9.8 | 25.9 | 23.7 |
| UniAD | 78.4 | 60.5 | 33.6 | 17.8 | 45.9 | 64.9 | 45.8 | 20.6 | 6.6 | 32.7 |
| DrivoR | **80.9** | 50.5 | 33.8 | **47.1** | 49.8 | **73.3** | 34.6 | 18.8 | **32.5** | 35.7 |
| **SimVTA** | 78.8 | **57.6** | **38.3** | 40.1 | **51.5** | 69.9 | **45.0** | **23.3** | 27.2 | **38.8** |

**Weighting check (computed here).** Tier counts of about 49 / 126 / 86 / 84 (summing to 345) reproduce all five rows' RC and HD-Score averages to within 0.06, so the averages are count-weighted and the five rows share one scenario set. **HAD-L's 345-era averages do not fit these weights** (50.6 / 33.1 predicted against 47.5 / 30.8 reported), nor the 436-scenario weights; HAD-L was aggregated over a different set. See [[concepts/hugsim-benchmark.md#simvta]].

- SimVTA leads on Medium and Hard; DrivoR leads on Easy and Extreme (RC 47.1 vs 40.1, HDS 32.5 vs 27.2), as the paper notes.
- Extreme-tier HD-Scores reach 25.9–32.5 here for LTF, SimVTA and DrivoR. In the 436-scenario rescoring no method exceeds 14 on that tier, and DrivoR scores 14.1 there. The tier labels do not mean the same thing across the two sets.
- No sub-metrics (NC, DAC, TTC, comfort), no per-dataset split, no commit hash.

### Table 4: Video quality (front view, NAVSIM v1 navtest, 8 frames over 4 s at 2 Hz)

| Source | FID ↓ | FVD ↓ |
|---|---:|---:|
| World Engine renderings along logged trajectories | 7.6 | 21.8 |
| WAM predictions: trajectory GRPO only (Exp0) | 20.4 | 105.4 |
| WAM predictions: SimVTA (Exp2) | 13.1 | 46.3 |

Predictions are autoregressive along the logged trajectory, compared with the corresponding real frames (same scenes). Clip count and resolution are not stated, so these values cannot be placed against the recorded-vs-recorded FVD floor ([[concepts/evaluation-variance.md#fvd-floor]]). **The initialization's own FID/FVD is not reported, nor Exp1's.** Exp0 has no video loss at all while GRPO changes the encoder its video head reads, so the comparison may measure how much video quality RL removes and the video loss preserves, rather than an improvement over Epona.

### Table 15 (Appendix F): recent WAMs as reported by their papers

| Method | NAVSIM v1 PDMS | NAVSIM v2 EPDMS |
|---|---:|---:|
| SimWAM | 91.5 | 90.2 |
| Latent-WAM | – | 89.3 |
| WA-JEPA | 91.8 | 91.7 |
| SimVTA | 91.3 | 88.2 |

The paper notes that training data, initialization and evaluators differ, and that SimWAM's v1 and v2 numbers come from its post-RL and pre-RL checkpoints. All three are above SimVTA on v2 and two of three on v1.

![[7.png|Figure 7: Reported planning results across benchmarks. (a) NAVSIM v1 PDMS and v2 EPDMS, with NAVSIM v2 component-score differences (SimVTA minus the NAVSIM-adapted Epona initialization). (b) HUGSIM RC and HD-Score using benchmark aggregation. (c) RC and (d) HD-Score for DrivoR and by scenario difficulty. Values correspond to Tables 1, 2, and 3.]]

**Figure 7**: (a) v1 86.2 → 91.3, v2 85.0 → 88.2, and the v2 component deltas against the initialization: NC +1.3, DAC +2.9, DDC +0.5, **TLC −0.7**, EP +2.4, TTC +2.1, LK +0.6, HC +1.1, **EC −7.6** (all consistent with Table 2). (b) HUGSIM RC and HD-Score for the five methods. (c, d) Per-tier RC and HD-Score for DrivoR and SimVTA.

---

## Ablations (NAVSIM v1 navtest)

### Table 12: configurations

| Setting | $K$ | Rendered pool | Video-training set | Video objective | PDMS |
|---|---:|---|---|---|---:|
| Exp0 | 0 | None | None | None | 89.1 |
| Exp1 | 4 | High reward | Q2, VideoDiT only | FM | 89.1 |
| **Exp2** | 4 | High reward | Q2 | FM | **91.3** |
| Exp3 | 4 | High reward | Q2, shuffled within scene | FM | 83.1 |
| Exp4 | 4 | 2 high + 2 low | Q2 + Q3 | FM | 90.9 |
| Exp5 | 4 | High reward | Q1 + Q2 | FM | 91.0 |
| Exp6 | 4 | 2 high + 2 low | Q1–Q4 (including unsafe) | FM | 90.9 |
| Exp7 | 8 | High reward | Q2 | FM | 91.2 |
| Exp8 | 4 | High reward | All valid rendered | Video GRPO | 91.0 |
| Exp9 | 4 | Initial-policy cache | Q2 from cache | FM | 90.4 |

Exp4 and Exp6 also truncate unsafe candidates' positive trajectory advantages to zero ($A_i\leftarrow\min(A_i,0)$), so they differ from Exp2 in the RL objective too. In Exp3, scenes without a valid within-scene permutation (a single Q2 candidate) get no video loss.

### Table 13: sub-scores

| Setting | NC | DAC | TTC | Comf. | EP | PDMS | Residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exp0 | 97.1 | 96.7 | 96.0 | 99.8 | 83.6 | 89.1 | −3.22 |
| Exp1 | 97.6 | 96.7 | 96.4 | 99.7 | 83.3 | 89.1 | −2.75 |
| **Exp2** | 98.0 | 97.2 | 97.0 | 100.0 | 86.7 | **91.3** | −2.51 |
| Exp3 | 94.1 | 91.3 | 91.8 | 99.6 | 81.2 | 83.1 | −6.91 |
| Exp4 | 97.4 | 97.3 | 96.4 | 99.9 | 86.4 | 90.9 | −2.94 |
| Exp5 | 98.0 | 97.1 | 97.0 | 99.9 | 86.1 | 91.0 | −2.56 |
| Exp6 | 97.7 | 97.2 | 96.6 | 99.9 | 86.5 | 90.9 | −2.64 |
| Exp7 | 97.8 | 97.3 | 96.9 | 99.9 | 86.6 | 91.2 | −2.60 |
| Exp8 | 97.9 | 97.2 | 96.9 | 99.9 | 86.2 | 91.0 | −2.56 |
| Exp9 | 97.1 | 97.2 | 95.9 | 99.9 | 85.9 | 90.4 | −3.19 |

All rows pass the closed-form check; Exp3's larger gap fits a score pulled down by more failing scenes. Main-text Tables 5–10 are slices of this table:

| Main-text table | Question | Rows | PDMS |
|---|---|---|---|
| 5 | Does the video loss need to reach the encoder? | Exp0 none / Exp1 VideoDiT / Exp2 VideoDiT + encoder | 89.1 / 89.1 / 91.3 (NC 97.1 / 97.6 / 98.0) |
| 6 | Which quadrants to supervise? | Exp4 Q2+Q3 / Exp6 Q1–Q4 / Exp5 Q1+Q2 / Exp2 Q2 | 90.9 / 90.9 / 91.0 / 91.3 |
| 7 | Select-then-flow-match or video GRPO? | Exp8 video GRPO on all rendered / Exp2 Q2 + FM | 91.0 / 91.3 |
| 8 | Online or offline rendering? | Exp0 none / Exp9 fixed cache / Exp2 online | 89.1 / 90.4 / 91.3 |
| 9 | Does pairing matter? | Exp3 shuffled / Exp2 matched | 83.1 (NC 94.1) / 91.3 (NC 98.0) |
| 10 | More rendering? | Exp7 2× / Exp2 1× | 91.2 / 91.3 |

**Video GRPO (Exp8)**: five stochastic Euler steps (noise 0.3) generate future latents per candidate; reward = negative framewise latent MSE against the render, temporally weighted; a clipped-ratio policy objective over the video head and shared parameters, with groups formed by a scene's trajectory conditions. It replaces flow matching and scores 91.0.

### Table 14: three training seeds

| Setting | Seed 1234 | Seed 20260901 | Seed 20260910 | Mean ± SD |
|---|---:|---:|---:|---:|
| Exp0 | 89.1 | 89.2 | 89.0 | 89.1 ± 0.1 |
| **Exp2** | 91.3 | 91.4 | 91.3 | 91.3 ± 0.1 |
| Exp5 | 91.0 | 90.9 | 91.0 | 91.0 ± 0.1 |
| Exp7 | 91.1 | 91.2 | 91.2 | 91.2 ± 0.1 |

Computed sds are 0.10 for Exp0 and 0.06 for the other three. The seeds vary the post-training run from one fixed Epona checkpoint. Exp2 is above Exp5 and Exp7 on every seed. The paper says results are "averaged over three independent runs", but Table 14 covers only these four settings, so whether Exp1, Exp3, Exp4, Exp6, Exp8 and Exp9 are means or single runs is not stated.

---

## Visualizations

![[4.png|Figure 4: Qualitative planning comparison. Left: planned trajectories of SimVTA (A, blue) and Epona (B, orange). Right: World Engine renderings along the corresponding paths at 1–4 s. Red boxes highlight Epona's collision and road-departure regions.]]

**Figure 4**: Two scenes, World Engine renderings along each planner's path at +1 to +4 s. Scene 1 (a right turn): Epona's path runs into a parked bus. Scene 2: Epona's path leaves the road onto the right shoulder. *(Wiki note: these are renders, not model predictions, so they illustrate the planners' paths through the simulator. Along Epona's off-log paths the bus at +4 s and the shoulder at +3–4 s are visibly smeared, which is the render degradation away from the log that Appendix G concedes.)*

![[5.png|Figure 5: Matched futures for alternative actions. Left: shared observation history and executed paths for going straight (A), changing lanes to the left (B), and changing lanes to the right (C). Right: corresponding future frames rendered by the simulator. Colors match the paths and image rows.]]

**Figure 5**: One history, three executed paths (straight, left lane change, right lane change) and the rendered futures along each. This is what a matched target looks like. Other vehicles in the renders keep moving; whether they follow the log or react is not stated anywhere in the paper.

![[6.png|Figure 6: Video–trajectory alignment before and after post-training. Left: observation history and trajectory in BEV. Right: World Engine renderings and predictions from SimVTA and Epona.]]

**Figure 6**: One scene, one straight trajectory. World Engine render, SimVTA prediction and Epona prediction at +1 to +4 s. Epona generates a grey car on the right at +3 and +4 s that is absent from the BEV and the render; SimVTA's prediction does not. One scene, along a near-logged path; no quantitative check of whether predictions follow off-log paths.

---

## Table Audit {#table-audit}

1. **v1 rows are consistent.** All 17 Table 1 rows and all ten ablation rows pass the closed-form check (−1.17 to −4.12; Exp3 −6.91). Copied rows match the wiki's copies (DreamerAD, ExploreVLA, SGDrive RFT, DriveWAM, PWM, ReCogDrive's 90.8), except that DriveVA's sub-scores, identical to Adaptive-WAM's copy, carry 90.9 here and 90.5 there.
2. **v2 mixes evaluators** (above). SimVTA, Epona, DreamerAD and SGDrive corrected-like; ResAD, DriveSuprim, DiffRefiner, DriveWorld-VLA, ELF-VLA and SafeDrive pre-fix-like. SafeDrive's +4.03 matches its residual in [[sources/world4scorer.md]]'s and [[sources/idol.md]]'s tables. DriveSuprim (87.1, EC 78.6) is the row SUV's table also carries; DiffRefiner (87.4) is a different row from IDOL's 86.2.
3. **Comparison scope.** Table 1's world-model block stops at SimVTA, while Table 15 shows SimWAM (91.5) and WA-JEPA (91.8) above it. Also absent: every world-model entry above 91.3 in the wiki (DriveReferee 92.0, DA-WAM 93.7, Drive-HWM, MM-Future, World4Scorer), and on v2 EponaV2 (88.9, same model family) and the corrected cohort above 88.2. The paper claims a gain over its initialization, not state of the art, so this is a framing issue rather than a false claim.
4. **HUGSIM** baselines are copied, and the five rows are internally consistent on one weighting. The "corrected evaluator and controller" is attributed to DrivoR's paper without naming the correction. *(2026-10-14: [[sources/drivor.md]] names it: a heading fix for the trajectory passed to the controller and a fix to the comfort metric's acceleration bounds, applied to all rows it reports.)*
5. **Figure scales.** Figure 3(a)'s axis starts at 80 and Figure 3(c)'s spans 90.8–91.4, so 0.1–0.4 PDMS differences fill most of the panel.
6. **Table numbering.** Three in-text references point one table too far (above).

---

## Limitations

The paper's own (Appendix G): renders degrade away from logged poses and renderer fidelity is measured only at logged viewpoints; a high prediction error may reflect a bad render rather than a model weakness; observation histories stay log-based and rendered targets differ from real images; teacher-forced selection cannot see autoregressive error accumulation; one backbone; added training cost. Robustness under shifted histories and reactive traffic is left to future work.

The wiki adds:

1. **No logged-future control.** GRPO with Epona's original video loss (logged trajectory, logged video) is the natural baseline for "video supervision during RL" and is not run, so the value of *off-log* targets specifically is not isolated.
2. **The error-based selection may select artifacts.** Q2 is defined as the higher-error half of rendered futures, and the paper concedes high error can come from poor renders. No check (renderer quality against $E^{\rm TF}$, or Q2 rates by scene reconstruction quality).
3. **Video quality evaluated only along logged trajectories**, with no initialization row and no Exp1 row (Table 4). Whether predictions along off-log paths follow those paths, the paper's motivating property, is shown in one figure only.
4. **The encoder is trained on rendered frames** as teacher-forced history. NAVSIM tests on real frames; HUGSIM tests on another group's 3DGS reconstructions. Whether 3DGS familiarity contributes to the HUGSIM result is not tested ([[concepts/data-driven-simulators.md#shortcut]]).
5. **HUGSIM without a control**: no initialization or Exp0 run, as the paper states. A different scenario set from the wiki's 436-scenario entries.
6. **Reward = metric.** The training reward is computed from the benchmark's own PDM sub-scores on navtrain, and video targets are restricted to candidates scoring ≥ 0.999 on four of them.
7. **Cost unreported**: total steps, wall-clock, rendering throughput and memory. Inference latency is not reported either; the deployed path is Epona's planning-only path (about 50 ms on an RTX 4090 at 10 steps, per [[sources/epona.md]]).
8. **Mixed v2 evaluators** and a v1 world-model block that stops at the paper's own row.
9. **One camera, one backbone, one benchmark family for training** (navtrain scenes with World Engine reconstructions; how many scenes have reconstructions is not stated).

---

## Relation to Other Wiki Pages

- [[sources/epona.md]]: the initialization. Epona's own ablation (−8.1 PDMS without joint video training) is the pretraining-time counterpart of Exp0 vs Exp2 (−2.2 during RL).
- [[sources/dreameraD.md]]: same initialization, opposite design. DreamerAD replaces the simulator with Epona's latent rollouts and a learned reward (88.7 / 87.7). SimVTA keeps two simulators, PDM for reward and 3DGS for video, and its RL alone (89.1) already passes DreamerAD. On v2 both rows are corrected-like, so +0.5 is a same-scale margin.
- [[sources/foresight.md]] (frozen Epona as encoder, 89.3) and [[sources/walt.md]] (frozen EponaV2, 89.8): the other Epona-family entries. SimVTA is the highest on v1 and below EponaV2's 88.9 on v2.
- [[sources/simwam.md]]: the same deployment (video trained, never generated at inference) and the same Flow-GRPO SDE, with a different RL scope (LoRA on the action expert, hard scenes only, +1.2 on a 90.3 base). SimVTA's RL updates the shared encoder and TrajDiT fully and is worth +2.9 on an 86.2 base.
- [[sources/simscale.md]], [[sources/rad.md]]: the other 3DGS training uses. SimScale renders fixed perturbations as policy *inputs* with pseudo-expert labels; RAD renders on-policy inputs for PPO; SimVTA renders on-policy *targets* for a video loss while the policy sees only logged images. SimScale's first author is SimVTA's project lead (Haochen Tian), and Tianyu Li is on both; the NAVSIM paper, as cited, lists a T. Li and a Z. Yang (SimVTA's corresponding author is Zhenjie Yang).
- [[sources/learning-to-drive-from-a-world-model.md]]: the inverse arrangement. There a world model is the simulator for a policy; here a simulator is the teacher for a world model.
- [[sources/foredrive.md]]: a wrong consumed future costs more than none (−2.4 vs −1.4). SimVTA's shuffle is the supervision-side analogue (−6.0 vs none).
- [[sources/world4scorer.md]]: its current-frame-target control (−0.31) is the other content control on an auxiliary future target in the wiki; SimVTA's is far larger.
- [[sources/physwam.md]], [[sources/learning-to-drive-from-a-world-model.md]]: the wiki's measured checks of whether generated video follows the action. SimVTA trains for that property and does not measure it.
- [[sources/wa-jepa.md]], [[sources/mm-future.md]], [[sources/physwam.md]]: HUGSIM entries on the 436-scenario set, not comparable with SimVTA's 345-scenario numbers.
- [[sources/plannerrft.md]]: the same group's closed-loop reinforcement fine-tuning of diffusion planners (nuPlan).
- [[sources/recogdrive.md]]: source of the 10 / 5 / 2 reward weights.
- Not ingested: **World Engine** (arXiv 2606.19836, the renderer and post-training framework SimVTA runs on, same group); **DrivoR** (now ingested as [[sources/drivor.md]]; the HUGSIM baseline and the source of the protocol); **ReSim** (a video simulator trained on simulated non-expert trajectories for action controllability, the nearest prior on the generation side); **CausalDrive** (a learned renderer); **Fast-WAM**; **DriveZero**; **EponaV2**.

---

## Open Questions

- **Do off-log targets beat the logged future?** Run GRPO with Epona's original logged-trajectory video loss at the same weight and gradient scale. If it matches 91.3, the benefit is "keep a video loss during RL", not "render the policy's futures".
- **Does Q2 select model weakness or render artifacts?** Correlate $E^{\rm TF}$ with reconstruction quality (novel-view PSNR or an artifact score), or repeat Exp2 with targets filtered by render quality.
- **Does the video loss improve prediction or prevent its decay?** Report FID/FVD for the initialization and Exp1, and along off-log executed paths, not only logged ones.
- **Why does shuffling hurt more than nothing?** Shuffle only the target frame while keeping history frames and poses consistent; if the collapse shrinks, it was frame–pose inconsistency in the encoder's input.
- **Does any of it survive closed loop?** HUGSIM or navhard for the initialization, Exp0 and Exp2 under one protocol.
