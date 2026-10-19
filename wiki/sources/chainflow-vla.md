---
title: "ChainFlow-VLA: Causal Flow Planning with Vision-Language Models"
type: source-summary
sources: ["raw/papers/ChainFlow-VLA_ Causal Flow Planning with Vision-Language Models.md"]
related: [sources/toad.md, sources/drivor.md, sources/recogdrive.md, sources/clover.md, sources/world4scorer.md, sources/mm-future.md, sources/coworld-vla.md, sources/wa-jepa.md, sources/onedrive.md, sources/diffusiondrive.md, sources/drivesuprim.md, sources/maneuverflow.md, sources/momworld.md, sources/sgdrive.md, sources/spanvla.md, sources/drivevla-w0.md, sources/autovla.md, sources/futuresightdrive.md, sources/hydra-mdp-pp.md, sources/orion.md, sources/da-wam.md, concepts/navsim-benchmark.md, concepts/selection-based-planning.md, concepts/diffusion-planner.md, concepts/dual-system-vla.md, concepts/vlm-domain-adaptation.md, concepts/best-of-n.md, concepts/evaluation-variance.md, concepts/inference-latency.md, research-directions.md]
created: 2026-10-19
updated: 2026-10-19
confidence: medium
---

# ChainFlow-VLA: Causal Flow Planning with Vision-Language Models

**Paper**: ChainFlow-VLA: Causal Flow Planning with Vision-Language Models
**Authors**: Xiyang Wang, Xinlin Wang, Tingguang Zhou, Gong Chen, Xingtai Gui, Zhi Xu, Xiaolei Wu, Feiyang Tan, Hangning Zhou, Mu Yang
**Orgs**: Afari Intelligent Drive; Tianjin University; University of Macau (the clipping does not map authors to affiliations). **The lab behind [[sources/coworld-vla.md]] and [[sources/wa-jepa.md]]**: eight of the ten authors are on WA-JEPA and five on CoWorld-VLA, all under the `AFARI-Research` GitHub organization.
**arXiv**: 2605.23270v1 (May 2026)
**Code**: github.com/AFARI-Research/ChainFlow-VLA
**Source**: `raw/papers/ChainFlow-VLA_ Causal Flow Planning with Vision-Language Models.md`. The clipping has all text, all five tables and all four figures (`p1.png`, `overview 2.png`, `3_methods_5x4.png`, `vlm_bev_5_3.png` in `raw/assets/`).

**Confidence: medium.** The tables are clean (every row passes the closed-form check) and code is released. But every number is a single run on NAVSIM-v1 navtest. The headline appears to be the best of a five-point denoising-step sweep on the test split. Nothing is reported on NAVSIM-v2, navhard, closed loop or latency. The VLM's +0.7 is not separated from adding a second, larger encoder, although one arm partly controls for it.

---

## What It Is

**DrivoR's camera-only planner with its trajectory decoder replaced by an autoregressive rollout through a bicycle model, followed by ReCogDrive's frozen VLM and DiT planner recast as a residual refiner, and a scorer that picks one refined candidate.**

1. **Backbone (from [[sources/drivor.md]]).** DINOv2 on multi-view images, LoRA-fine-tuned, giving scene tokens (the paper calls them "BEV-style driving features"); ego status through an MLP; learned trajectory queries.
2. **Chain: autoregressive proposals.** $K$ parallel hypotheses. At each step a decoder cell (time-pose embedding, cross-attention to scene tokens, FFN, self-attention, MLP) predicts acceleration and yaw rate $(a_t, \omega_t)$, and a **bicycle kinematic model** produces the next state. The fed-back state is detached (Figure 2), so there is no backpropagation through time. Each step is deterministic.
3. **Flow: VLM-guided residual diffusion.** A DiT denoises a residual $\Delta Y_k$ added to each proposal: AdaLN conditioning on timestep and ego status, the mean of the VLM hidden states added to the input, and cross-attention to the full hidden-state sequence. The VLM is **ReCogDrive's 2B InternVL model** (supervised fine-tuning on environment-understanding and trajectory QA), frozen. It sees **the front camera only** plus a text prompt. DDIM, 4 steps by default.
4. **Scorer.** A scoring head "following iPad" gives each refined candidate utility scores; the highest is chosen.

The DiT block is ReCogDrive's planner design ([[sources/recogdrive.md]]: AdaLN on ego status and timestep, pooled VLM states in the input, cross-attention to the full sequence). What is new is that it predicts a correction to a strong E2E planner's proposal instead of a whole trajectory.

**Results**

| Benchmark | ChainFlow-VLA | Notes |
|---|---:|---|
| NAVSIM-v1 PDMS (navtest), navtrain | **93.6** | DriveSuprim 93.5 and RAP-DINO 93.8 (10× data) in the same table |
| NAVSIM-v1 PDMS (navtest), trainval | **94.8** (94.85 at 12 steps; **94.72 at the default 4**) | Human driver 94.8; DrivoR trainval 93.7 |
| On DiffusionDrive / iPad (no VLM, navtrain) | 88.1 → 88.9 / 91.7 → 92.7 | Table 5 |
| NAVSIM-v2, navhard, closed loop, latency | **not reported** | |
| Training | 8 × A800; Stage 1 25 epochs (Chain + LoRA encoder), Stage 2 40 epochs (Flow) | AdamW, lr 2e-4 scaled by $\sqrt{B/64}$ |

---

## Key Takeaways

- **The highest single-pass NAVSIM-v1 score in the wiki, with the val split.** 94.8 trainval ties the human driver and is above DrivoR + 134k SimScale scenes (94.6), CLOVER (94.5) and MM-Future trainval (94.0). On navtrain alone it is 93.6, +0.1 over DriveSuprim in its own table and below [[sources/world4scorer.md]]'s 94.0 (published later).
- **"Matches human-level" is a PDMS tie with a different profile.** The human scores NC, DAC and TTC 100 and EP 87.5. ChainFlow-VLA scores NC 99.2, DAC 99.0, TTC 97.2 and EP 91.9: more progress, less safety. By closed form on mean sub-scores the human is 94.77 and ChainFlow-VLA 93.73.
- **The headline is probably the best of a step sweep on navtest.** The default 4-step model is 94.72 in Tables 3 and 4, which would print as 94.7. The 12-step run is 94.85, the abstract's number. Steps 2 / 4 / 8 / 12 / 16 give 94.68 / 94.72 / 94.74 / 94.85 / 94.67, non-monotonic and within 0.18. Table 1's 94.8 and Table 2's full-model row are most plausibly the 12-step run.
- **Almost all of the Flow stage's gain is the VLM.** On DrivoR (trainval, Table 2): the AR Chain +0.3 (93.7 → 94.0); the residual DiT conditioned on DrivoR's own scene tokens +0.1 (94.1); swapping that condition for VLM hidden states +0.7 (94.8). EP carries the gain (90.0 → 91.9). Each step is one run.
- **What the VLM was fine-tuned on matters, which is a partial control for "just a second encoder".** With a VLM fine-tuned on action QA only, the refiner scores 94.11, the same as the scene-token refiner (94.1). With environment- and trajectory-QA fine-tuning it scores 94.72. "Action QA" is not defined.
- **Refining beats regenerating by 1.8.** A DiT that predicts the whole trajectory scores 92.89, below DrivoR (93.7) and the AR proposals alone (94.0); predicting the residual scores 94.72. ReCogDrive's planner design, used to generate from scratch, loses points against the planner it was meant to refine.
- **The two-stage action expert transfers to weaker bases.** Without a VLM, on navtrain: DiffusionDrive 88.1 → 88.9 with 6 AR modes in place of 20 clustered anchors; iPad 91.7 → 92.7 at 64 modes. One run each; iPad + ChainFlow (92.7) is still below DrivoR on navtrain (93.1).

---

## Method

![[p1.png|Figure 1: Comparison of different paradigms for integrating VLM into end-to-end autonomous driving. (a) VLM-guided pipeline that predicts high-level guidance to steer an end-to-end model, which introduces an information bottleneck and limits fine-grained trajectory refinement. (b) Feature-level fusion that combines VLM and perception backbones via a fusion module followed by an action expert, but lacks a principled mechanism to enforce consistency between local dynamics and global trajectory structure. (c) Ours (ChainFlow-VLA) formulates trajectory prediction as a unified causal–flow process, where an AR generator produces temporally consistent proposals that are refined by a diffusion model in the residual space. Fine-tuned VLM representations are injected as semantic flow conditioning to guide global trajectory refinement.]]

**Figure 1**: Three ways to put a VLM into an E2E planner. (a) VLM → action expert (ORION, OpenDriveVLA). (b) VLM and backbone fused, then an action expert (LatentVLA, ReCogDrive, SGDrive). (c) ChainFlow-VLA: the backbone feeds an AR generator, and the VLM conditions only the diffusion refiner that corrects its proposals.

![[overview 2.png|Figure 2: ChainFlow-VLA framework. The model first performs Autoregressive Trajectory Generation (Chain) to produce K causal proposals, which are then refined via VLM-Guided Residual Diffusion (Flow). By learning the residuals between AR proposals and ground-truth trajectories, the model unifies causal rollout with VLM-based semantic guidance, formulating planning as a mixture of VLM-conditioned residual distributions over AR-induced modes.]]

**Figure 2**: Multi-view images go through DINOv2; ego status through an MLP; with a trajectory query they enter the Chain. Top left, the AR decoder cell: time-pose embedding, cross-attention to "BEV features", FFN, self-attention, MLP, bicycle motion model, output $y_t^n$, fed back through a detached MLP for $T$ steps. Top right, the DiT block: residual proposals plus noise, a time-pose embedding and the **mean** of the VLM hidden states go through Linear, AdaLN, self-attention, AdaLN, cross-attention to the VLM hidden states, AdaLN, FFN, $\times N$, with ego status in the AdaLN conditioning. Bottom: the **front view** and a prompt go to the VLM; its hidden states condition Flow. The box marked "Training Only" lists the VLM's fine-tuning tasks (scene perception and description, traffic status assessment, key traffic element identification, motion planning and prediction), decoded by the LLM head and not used at inference. The scorer picks among the refined candidates.

### Formulation (§3)

The paper writes the plan distribution as a mixture over AR modes:

$$P(Y\mid\mathcal O)\approx\sum_{k=1}^K P(Y\mid Y_\mathrm{AR}^{(k)},h_\mathrm{VLM})\,P(Y_\mathrm{AR}^{(k)}\mid\mathcal O)$$

and each component in residual space, $P(Y\mid Y_\mathrm{AR}^{(k)},h_\mathrm{VLM})=P(\Delta Y_k\mid Y_\mathrm{AR}^{(k)},h_\mathrm{VLM})$ with $Y=Y_\mathrm{AR}^{(k)}+\Delta Y_k$. In practice the mixture weights $P(Y_\mathrm{AR}^{(k)}\mid\mathcal O)$ are never estimated: the AR stage is deterministic, and the final choice is the argmax of a learned utility. "Causal" means temporal autoregressive order, not causal inference.

### Chain (§4.2)

$$(a_t^{(k)},\omega_t^{(k)})=H_\theta(y_{<t}^{(k)},\mathcal O),\qquad y_t^{(k)}=\mathrm{Bicycle}(y_{t-1}^{(k)},a_t^{(k)},\omega_t^{(k)})$$

### Flow (§4.3)

Residual target $\Delta Y_k^*=Y^*-Y_\mathrm{AR}^{(k)}$; noised $\mathbf z_t^{(k)}=\sqrt{\bar\alpha_t}\,\Delta Y_k^*+\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon$; the network predicts $\hat{\boldsymbol\epsilon}^{(k)}=\boldsymbol\epsilon_\theta(\mathbf z_t^{(k)},t,c_\mathrm{ego},h_\mathrm{VLM},Y_\mathrm{AR}^{(k)})$. At inference a DDIM residual is sampled per proposal and $\hat Y_k=Y_\mathrm{AR}^{(k)}+\hat{\Delta Y}_k$. The residual idea is credited to ResAD.

### Training (§4.5, §5.1)

- **Stage I**: $\mathcal L_\mathrm{traj}+\lambda_1\mathcal L_\mathrm{scorer}$, winner-take-all on the closest mode (following DrivoR). 25 epochs, LoRA on the image encoder.
- **Stage II**: $\lambda_2\mathcal L_\mathrm{diff}+\lambda_3\mathcal L_\mathrm{traj}+\lambda_4\mathcal L_\mathrm{scorer}$ with $(\lambda_1..\lambda_4)=(1, 10, 20, 4)$. 40 epochs.
- **Asymmetric assignment**: the diffusion loss is computed only for the proposal closest to the expert, $k^*=\arg\min_k\lVert Y_\mathrm{AR}^{(k)}-Y^*\rVert_2$; $\mathcal L_\mathrm{traj}$ is applied to the refined outputs, at twice the diffusion weight. **At inference every proposal is refined, but in training only one residual per scene had a target.**
- Per-GPU batch 8 on 8 A800s; 10% linear warm-up, cosine decay.

---

## Results

### Table 1: NAVSIM-v1 navtest

Residual = closed form − reported, computed here; negative is normal for per-scene-averaged PDMS ([[concepts/navsim-benchmark.md#physwam]]).

| Method | Venue | PDMS | NC | DAC | EP | TTC | Comf. | Residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| *End-to-end* | | | | | | | | |
| UniAD | CVPR'23 | 83.4 | 97.8 | 91.9 | 78.8 | 92.9 | 100.0 | −4.12 |
| VADv2 | arXiv'24 | 80.9 | 97.2 | 89.1 | 76.0 | 91.6 | 100.0 | −5.99 |
| iPad | arXiv'25 | 91.7 | 98.6 | 98.3 | 88.0 | 94.9 | 100.0 | −1.68 |
| DiffusionDrive | CVPR'25 | 88.1 | 98.2 | 96.2 | 82.2 | 94.7 | 100.0 | −2.72 |
| Hydra-MDP++ | arXiv'25 | 91.0 | 98.6 | 98.6 | 85.7 | 95.1 | 100.0 | −1.56 |
| Centaur | arXiv'25 | 92.6 | 99.5 | 98.9 | 85.9 | 98.0 | 100.0 | −0.80 |
| TrajDiff (train) | arXiv'25 | 86.4 | 98.0 | 95.0 | 80.8 | 93.7 | 100.0 | −3.19 |
| TrajDiff (trainval) | arXiv'25 | 88.5 | 98.1 | 97.0 | 82.7 | 94.3 | 100.0 | −2.46 |
| DriveSuprim | AAAI'26 | 93.5 | 98.6 | 98.6 | 91.3 | 95.5 | 100.0 | −1.63 |
| RAP-DINO † | ICLR'26 | 93.8 | 99.1 | 98.9 | 90.3 | 96.7 | 100.0 | −1.10 |
| DrivoR (train) | CVPR'26 | 93.1 | 98.9 | 98.3 | 89.1 | 96.2 | 100.0 | −1.84 |
| DrivoR (trainval) | CVPR'26 | 93.7 | 99.0 | 98.9 | 90.0 | 96.7 | 100.0 | −1.21 |
| *VLA-based* | | | | | | | | |
| UniVLA | RSS'25 | 81.7 | 96.9 | 91.1 | 76.8 | 91.7 | 96.7 | −5.50 |
| AutoVLA | NeurIPS'25 | 89.1 | 98.4 | 95.6 | 81.9 | 98.0 | 99.9 | −2.92 |
| FSDrive | NeurIPS'25 | 85.1 | 98.2 | 93.8 | 80.1 | 93.3 | 99.9 | −3.21 |
| DriveVLA-W0 | ICLR'26 | 90.2 | 98.7 | 99.1 | 83.3 | 95.3 | 99.3 | −1.22 |
| ReCogDrive | ICLR'26 | 90.8 | 97.9 | 97.3 | 87.3 | 94.9 | 100.0 | −2.61 |
| SGDrive | CVPR'26 | 91.1 | 98.6 | 97.8 | 85.8 | 96.2 | 100.0 | −1.90 |
| SpanVLA | arXiv'26 | 90.3 | 99.1 | 97.1 | 86.3 | 95.2 | 100.0 | −1.49 |
| LatentVLA | arXiv'26 | 92.4 | 98.9 | 98.2 | 88.2 | 96.0 | 100.0 | −1.67 |
| *Baselines & oracle* | | | | | | | | |
| Human driver | NeurIPS'24 | 94.8 | 100.0 | 100.0 | 87.5 | 100.0 | 99.9 | −0.03 |
| **ChainFlow-VLA (train)** | – | **93.6** | 98.8 | 98.6 | 90.8 | 96.1 | 100.0 | −1.50 |
| **ChainFlow-VLA (trainval)** | – | **94.8** | 99.2 | 99.0 | 91.9 | 97.2 | 99.9 | −1.07 |

† RAP-DINO is pre-trained on a private dataset 10× larger than navtrain. Every row passes the check. The columns are ordered EP before TTC, unlike most NAVSIM tables.

### Table 2: component ablation (trainval; ID 0 is the DrivoR baseline)

Without VLM guidance the DiT is conditioned on DrivoR's scene tokens.

| ID | AR Gen. (Chain) | DiT Refiner (Flow) | VLM Guidance | PDMS | NC | DAC | EP | TTC | Comf. |
|---|:-:|:-:|:-:|---:|---:|---:|---:|---:|---:|
| 0 | ✗ | ✗ | ✗ | 93.7 | 99.0 | 98.9 | 90.0 | 96.7 | 100.0 |
| 1 | ✓ | ✗ | ✗ | 94.0 (+0.3) | 99.1 | 98.9 | 90.8 | 96.7 | 99.9 |
| 2 | ✓ | ✓ | ✗ | 94.1 (+0.4) | 99.1 | 98.9 | 91.0 | 96.7 | 99.9 |
| 3 | ✓ | ✓ | ✓ | **94.8 (+1.1)** | 99.2 | 99.0 | 91.9 | 97.2 | 99.9 |

Residuals −1.21, −1.11, −1.13, −1.07. Row 3's sub-scores are Table 1's trainval row exactly.

### Table 3: DiT design choices (trainval, 4 denoising steps)

| (a) Modeling target | PDMS |
|---|---:|
| Trajectory space | 92.89 |
| Residual space | **94.72** |

| (b) DiT blocks | PDMS |
|---|---:|
| 8 | 94.64 |
| 12 | **94.72** |

| (c) VLM guidance source | PDMS |
|---|---:|
| Action QA | 94.11 |
| Env. & Traj. QA | **94.72** |

### Table 4: denoising steps at inference (trainval)

| $N_\mathrm{step}$ | 2 | 4 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|---:|
| PDMS | 94.68 | 94.72 | 94.74 | **94.85** | 94.67 |

The paper keeps 4 steps as the default "to balance performance and inference efficiency" and reports no timing.

### Table 5: ChainFlow on other planners (no VLM features, navtrain)

| Method | Modes | PDMS | NC | DAC | EP | TTC | Comf. | Residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DiffusionDrive | 20 | 88.1 | 98.2 | 96.2 | 82.2 | 94.7 | 100.0 | −2.72 |
| + ChainFlow | 6 | **88.9** (+0.8) | 98.3 | 96.8 | 83.0 | 95.0 | 99.9 | −2.48 |
| iPad | 64 | 91.7 | 98.6 | 98.3 | 88.0 | 94.9 | 100.0 | −1.68 |
| + ChainFlow | 64 | **92.7** (+1.0) | 99.0 | 98.7 | 88.4 | 96.1 | 99.9 | −1.31 |

For DiffusionDrive, ChainFlow replaces the clustered anchors with AR proposals. The baseline rows are the published ones, not reruns.

---

## Visualizations

![[3_methods_5x4.png|Figure 3: Qualitative comparison of trajectory predictions on representative NAVSIM scenarios. GT trajectories are shown in green. Predicted trajectories from ReCogDrive, DrivoR, and ChainFlow-VLA are visualized in red, purple, and orange, respectively.]]

**Figure 3**: Five navtest scenes, front concatenated view plus BEV for ReCogDrive, DrivoR and ChainFlow-VLA. Row 1, a roundabout: ReCogDrive cuts toward the island and off the drivable area; ChainFlow-VLA follows the human. Row 2, a left-turn ramp: both baselines drift into another lane. Row 3, a sharp turn: ReCogDrive runs off the road and DrivoR overshoots. Row 4, a right turn at an intersection: the baselines end in a parked vehicle, while ChainFlow-VLA goes past the human's endpoint. Row 5, a static barrier on the right, which the baselines hit. Chosen by the authors.

![[vlm_bev_5_3.png|Figure 4: Qualitative comparison between BEV-conditioned and VLM-conditioned refinement. GT trajectories are shown in green, while trajectories refined using backbone BEV features (ChainFlow-BEV) and semantic VLM features (ChainFlow-VLA) are shown in red and orange, respectively.]]

**Figure 4**: The same model refined with DrivoR's scene tokens (ChainFlow-BEV, Table 2 row 2) and with VLM hidden states (row 3). Row 1, a right turn: the BEV-conditioned plan heads straight. Rows 2, 4 and 5 (a narrow road beside a bus, an intersection turn, a roundabout): the BEV-conditioned plan reaches a road boundary. Row 3, slow car-following on a curved road: the BEV-conditioned plan runs into the lead vehicle. In several scenes the VLM-conditioned plan ends beyond the human's endpoint, consistent with the EP gain (90.0 → 91.9). Five chosen scenes, against a 0.7-point aggregate difference.

---

## Table Audit {#table-audit}

1. **All 31 rows across Tables 1, 2 and 5 pass the closed-form check** (−0.03 for the human to −5.99 for VADv2). The paper's own rows sit at −1.07 to −1.50, inside the band of its DrivoR base (−1.21, −1.84).
2. **Which run is the headline.** Tables 3 and 4 give the default configuration (residual, 12 blocks, Env. & Traj. QA, 4 steps) as 94.72, while Tables 1 and 2 print 94.8 and the abstract 94.85. The simplest reading is that the printed trainval row is the 12-step run, the maximum of a five-point sweep on navtest. Read that way, the VLM increment in Table 2 (94.1 → 94.8) compares a 12-step full model with a row of unstated step count; at 4 steps it would be 94.72 − 94.1 = 0.62.
3. **"Significantly outperforming prior end-to-end (93.8) and VLA-based (92.4) models."** The 93.8 is RAP-DINO, trained with 10× data; against DrivoR trainval the margin is +1.1. On navtrain the margin over prior work is +0.1 (DriveSuprim 93.5) and RAP-DINO stays ahead. The VLA best of 92.4 (LatentVLA) holds in its own table. CoWorld-VLA (same lab, 90.0, cited in related work) is not in it. DrivoR's appendix rows with SimScale data (94.0, 94.6) are not printed, although RAP-DINO is, with a note. CLOVER (94.5) appeared on arXiv about a week earlier. CLEAR (93.7) and WCog-VLA (92.9) postdate it.
4. **Citations.** VADv2 (80.9, the NAVSIM reproduction) is cited to VAD's ICCV paper. ReCogDrive's 90.8 is labelled ICLR'26; [[concepts/navsim-benchmark.md]] records DreamerAD's table calling 90.8 the NeurIPS camera-ready, against 89.6 on arXiv.
5. **The main model's $K$ is never stated.** Table 5 gives 6 modes on DiffusionDrive and 64 on iPad. The scorer is "following iPad"; whether it replaces DrivoR's gradient-stopped sub-score scorer is not said.

---

## Limitations

The paper's own: the VLM is a general driving-oriented model trained on environment and trajectory QA, and a score- or judge-oriented VLM may suit refinement better.

The wiki adds:

1. **NAVSIM-v1 only.** No v2 EPDMS, no navhard, no closed loop, no nuScenes or Bench2Drive. Every top-of-v1 claim in the wiki now carries the question of whether gains above about 93 measure driving or selection ([[concepts/best-of-n.md]]).
2. **Single runs, small increments.** The named components are worth +0.3, +0.1 and +0.7 PDMS; the step sweep spans 0.18 with no trend. The wiki's training-seed sd for NAVSIM recipes is 0.04–0.1 ([[concepts/evaluation-variance.md]]).
3. **The headline is selected on the test split** (12 of five step settings), and it is the trainval model.
4. **The VLM's effect is not isolated from adding a second encoder.** The VLM is a separate 2B InternVL, with its own vision encoder over the front camera, fine-tuned following ReCogDrive (whose recipe uses 3.1M QA pairs, 775K of them NAVSIM-derived; [[sources/recogdrive.md]]). The paper does not say which data the 2B model saw. The Action-QA arm (94.11, equal to the scene-token arm) is a partial control: same architecture, different fine-tuning, no gain. There is no early-fusion arm with the same VLM, so the paper's claim that VLMs work better as late conditioners than as fused features rests on cross-paper comparisons (LatentVLA, ReCogDrive).
5. **The refiner is trained on one residual per scene and applied to all $K$.** Only the proposal closest to the expert gets a diffusion target. Proposals far from any expert are corrected at test time by a model that never saw a target for them, and the scorer then chooses. The same pattern is open for per-candidate futures ([[concepts/world-model-for-ad.md]]: 31 of 32 or 63 of 64 unsupervised).
6. **"Diffusion" with a strong regression loss.** $\mathcal L_\mathrm{traj}$ on refined outputs has twice the diffusion weight and sampling uses 4 DDIM steps. Sample diversity is never measured, so whether the refiner behaves stochastically is unknown.
7. **Error accumulation is asserted, not measured.** The AR decoder beats DrivoR's one-token-per-trajectory decoder by 0.3 in one run, with no horizon-wise error analysis. Causality and the bicycle model are not separated. In DrivoR's own ablation, one token per pose without a kinematic model scored 83.9 against 90.0 ([[sources/drivor.md]]), so the bicycle model plausibly does much of the work.
8. **No cost figures.** DrivoR's 110 ms on an A100 plus a 2B VLM forward pass plus $K$ residuals × 4–12 DiT steps; nothing is timed.
9. **The scorer's labels are not stated.** It "follows iPad" and outputs "utility scores"; its DrivoR base trains its scorer on the NAVSIM oracle's sub-scores, as does every other entry above 92 ([[concepts/selection-based-planning.md#top-of-leaderboard]]).

---

## Relation to Other Wiki Pages

- [[sources/toad.md]] *(2026-10-19)*: a month later, DrivoR's group reaches 94.7 PDMS (DrivoR + 134k SimScale + test-time search) and calls it "the top entry"; its table omits ChainFlow-VLA's earlier 94.8. Both use a bicycle model in acceleration and yaw rate, ChainFlow-VLA to generate proposals and TOAD to search around one.
- [[sources/drivor.md]]: the base. Table 2's row 0 is DrivoR's trainval model, Stage I follows its WTA training, and the backbone is its DINOv2 stack. ChainFlow-VLA adds +1.1 to it; [[sources/clover.md]] adds +0.74 to the same base by retraining generator and scorer against the evaluator, and [[sources/world4scorer.md]] +0.50 by swapping the scorer at fixed candidates. All three start from DrivoR, but only CLOVER and World4Scorer report how much of their gain is ranking. ChainFlow-VLA reports no oracle.
- [[sources/recogdrive.md]]: supplies the frozen 2B VLM and the DiT design. ReCogDrive's planner generates whole trajectories from VLM states (90.8 with RL); here it predicts residuals around DrivoR-family proposals, and the trajectory-space version of it scores 92.89.
- [[sources/coworld-vla.md]], [[sources/wa-jepa.md]]: the same lab. CoWorld-VLA (90.0, May 2026) is cited in related work but left out of Table 1. WA-JEPA (August 2026, eight shared authors) cites ChainFlow-VLA in related work as "VLM-guided trajectory refinement" and leaves its 94.8 out of its own v1 table, where it reports 91.8.
- [[sources/onedrive.md]]: another InternVL3-2B model initialized from ReCogDrive. The two papers sit at opposite ends of [[concepts/dual-system-vla.md]]: OneDrive puts everything in one causal decoder, while ChainFlow-VLA keeps the VLM frozen and outside the action path except as a condition.
- [[sources/momworld.md]], [[sources/maneuverflow.md]]: two other refine-after-generate designs. MomWorld's residual flow moves the selected plan and is not re-scored; ManeuverFlow refines inside the pool and re-scores. ChainFlow-VLA refines every proposal and scores after.
- [[sources/diffusiondrive.md]]: Table 5 replaces its 20 clustered anchors with 6 AR proposals (+0.8).
- [[sources/sgdrive.md]], [[sources/spanvla.md]], [[sources/drivevla-w0.md]], [[sources/autovla.md]], [[sources/futuresightdrive.md]], [[sources/drivesuprim.md]], [[sources/hydra-mdp-pp.md]]: baselines in Table 1, at the values the wiki carries.
- [[sources/orion.md]]: the paper's example of paradigm (a), VLM-predicted guidance steering an action expert.
- Not ingested: **iPad** (the scorer design and a Table 5 base), **ResAD** (residual trajectory modeling), **LatentVLA** (92.4, the feature-fusion VLA it contrasts with), **Centaur**, **TrajDiff** (shares authors), **RAP**, **UniVLA**, **OpenDriveVLA**, **DiffVLA**, **DriveMoE**.

---

## Open Questions

- **Does the VLM help through what it knows or through what it sees?** Condition the refiner on the same VLM's vision encoder alone, and on a VLM fine-tuned on unrelated QA, at matched compute.
- **Does the +0.7 survive on navhard and NAVSIM-v2?** EP carries the gain, and the scorer cohort's v1 gains often vanish on navhard ([[sources/clover.md]]: +0.8 navtest, +0.0 navhard).
- **What is the oracle over the refined candidates, and how much of the gain is ranking?** The identity CLOVER used would answer it from released checkpoints.
- **Is the refiner stochastic in practice?** Report the spread of $\hat{\Delta Y}_k$ across sampler seeds; with a regression loss at twice the diffusion weight it may be close to deterministic.
- **Does refining with residual targets only on the expert-closest proposal hurt the other proposals?** Supervise every proposal toward an evaluator-filtered target (as CLOVER's pseudo-experts would provide) and compare.
