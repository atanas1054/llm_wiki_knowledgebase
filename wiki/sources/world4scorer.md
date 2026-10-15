---
title: "World4Scorer: Outcome-Grounded World Modeling for Autonomous Driving"
type: source-summary
sources: ["raw/papers/Outcome-Grounded World Modeling for Autonomous Driving.md"]
related: [sources/clover.md, sources/drivor.md, concepts/selection-based-planning.md, concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/bench2drive.md, concepts/evaluation-variance.md, concepts/inference-latency.md, concepts/counterfactual-prediction.md, concepts/teacher-pseudo-labels.md, concepts/best-of-n.md, concepts/foundation-backbones-for-ad.md, sources/da-wam.md, sources/mm-future.md, sources/drive-jepa.md, sources/adaptive-wam.md, sources/auto-jepa.md, sources/hydra-mdp-pp.md, sources/drivesuprim.md, sources/geowam.md, sources/foredrive.md, sources/ad-e2e-jepa.md, sources/raw2drive.md, sources/recogdrive.md, sources/wa-jepa.md, sources/drivereferee.md, research-directions.md]
created: 2026-10-06
updated: 2026-10-14
confidence: medium
---

# World4Scorer: Outcome-Grounded World Modeling for Autonomous Driving

**Paper**: Outcome-Grounded World Modeling for Autonomous Driving (method name: World4Scorer)
**Authors**: Jieyuan Pei, Meiyi Lu, Sining Ang, Yubo Zhao, Zhangyi Hu, Mingwei Xu, Haokai Ding, Wei Li, Zihan You, Jianwei Zheng, Li Yu, Yifeng Pan, Ji Tao, Rongjunchen Zhang, Yan Wang (corresponding)
**Orgs**: Institute for AI Industry Research (AIR), Tsinghua University; HiThink Research; HKUST (Guangzhou); Zhejiang University; USTC; SMBU; University of Washington; MBZUAI; Zhejiang University of Technology; Southeast University; Changan Automobile. **The CLOVER group**: Sining Ang and Yan Wang are authors of CLOVER ([[sources/clover.md]], ingested 2026-10-14) and of [[sources/adaptive-wam.md]].
**arXiv**: 2609.36438v1 (September 2026)
**Code**: project page, GitHub and Hugging Face links are given; the reproducibility statement says code will be released on publication.
**Source**: `raw/papers/Outcome-Grounded World Modeling for Autonomous Driving.md`. The clipping stops after the Table 1 caption. Sections 4.1–5, Tables 1–13, Figures 3–14 and Appendices A–E were recovered from the arXiv HTML (v1); the missing figure files were downloaded into `raw/assets/`.

---

## What It Is

A **generate-and-select planner whose scorer is a trajectory-conditioned latent predictor**. It is built on [[sources/drivor.md]] (ingested 2026-10-14): the same LoRA-adapted DINOv2 encoder, trajectory generator, 64 candidates and training split.

1. **Encoder and generator** (DrivoR's): DINOv2 ViT-S/14 with rank-32 LoRA encodes four cameras into scene tokens $s_t$; a transformer generates $N=64$ trajectories, trained by winner-take-all L1 imitation.
2. **Scorer = "latent world model"**: an action encoder $\phi$ turns each (stop-gradient) candidate into a query; a 4-block predictor $P$ (self-attention among candidates, cross-attention to scene tokens) maps it to a state $z_i$; score heads $H(z_i+e_t)$ predict NC, DAC, DDC, TTC, EP and comfort.
3. **Two kinds of supervision on the same predictor**:
   - **Outcome labels for every candidate**: NAVSIM-simulator sub-scores of all 64 generated candidates (computed online) plus 16 perturbed trajectories per step from a **scene-matched CLOVER bank**, up to 80 labelled states per scene.
   - **One observed future**: a separate query for the logged trajectory predicts, through a linear readout $g$, the frozen DINOv2 embedding of the front view **2 s later, mean-pooled to 384 dimensions** (cosine loss).
4. **Inertial re-ranking** (training-free): each candidate is checked against the previous frame's selected plan with **NAVSIM-v2's own extended-comfort criterion** (LQR + bicycle rollouts, RMS of acceleration, jerk, yaw rate, yaw acceleration against the benchmark thresholds); incompatible candidates get a $-\log 100$ penalty.

**Results**

| | World4Scorer | With inertial re-ranking |
|---|---:|---:|
| NAVSIM v1 PDMS (navtest) | **94.0** | – |
| NAVSIM v2 EPDMS (navtest, latest devkit; corrected-like by residual) | 91.4 | **93.0** |
| Bench2Drive DS / SR (adapted system) | 73.27 / 46.36% | – |
| OGBench-Cube success (outcome scorer added to frozen LeWM) | 73.64% (LeWM 68.91%) | – |
| Latency (A100, fp32, batch 1) / parameters | 106.5 ms / 35.52M | + 125.4 ms on CPU |

---

## Key Takeaways

- **The highest navtrain-trained single-pass v1 score in the wiki (94.0) and, with re-ranking, the highest v2 score (93.0).** Both are scorer-cohort results: the scorer is trained on the benchmark's own sub-scores. Without re-ranking, 91.4 EPDMS is third in the corrected cohort, behind WA-JEPA 91.7 and MM-Future 91.5, neither of which is in its table.
- **The v2 lead is the re-ranking, and the re-ranking is portable.** At fixed weights and candidates it lifts extended comfort 77.0 → 92.8 and EPDMS +1.6. Applied to DrivoR's released model it gives +1.61 (90.66 → 92.27), which is also above every other v2 entry in the wiki. It is the benchmark's EC criterion run at inference on the previous plan.
- **The "world model" part rests on one single-seed row.** Adding the observed-future target: 93.01 → 93.96 PDMS (+0.95). The same objective with a **current-frame** target scores 92.70 (−0.31). That is the wiki's first control that keeps a future-prediction objective and pathway and removes only the temporal content of the target. The current and +2 s targets have mean cosine 0.94, and the paper says the single-seed gap "does not by itself isolate" temporal content.
- **The fixed-pool comparison isolates the scorer.** On the 64 candidates of DrivoR's released model, whose oracle is 99.33 PDMS, DrivoR's own scorer leaves a regret of 5.64 (93.69) and World4Scorer's leaves 5.14 (94.19). That **+0.50 is a scorer-only gain**, and World4Scorer's scorer on DrivoR's pool beats World4Scorer's full system (94.0), so the generator contributes nothing extra.
- **"Candidate states guide selection" is shown, but not that they are predictive.** Shuffling states among candidates costs 14.47 PDMS (all 100 shuffles lower it). The trajectory reaches the score heads only through its state, and the same shuffle on DrivoR's non-predictive scorer costs +14.11 regret. The test shows the scorer reads each candidate. It does not show the state predicts that candidate's future.
- **NAVSIM v1 and v2 disagree on 43% of scenes, and extended comfort is 98% of the disagreement.** Using simulator sub-scores on frozen pools, the v1-optimal candidate fails the EC check in 99.7% of disagreement scenes, and picking the v2-optimal one costs 1.3 PDMS on v1. In 38% of scenes no candidate is both top-quartile in progress and EC-passing.
- **The v2 table is the most uniformly corrected-like in the wiki**, and it carries **DiffusionDrive at 88.2** with the block's sub-scores (residual −0.9). [[sources/geowam.md]] printed the same row, which this wiki had filed as an anomaly. Two independent tables now suggest that 88.2 is DiffusionDrive's corrected score and the circulating 84.5 its pre-fix one (§4).
- **Bench2Drive is an adapted system with extra inputs** (CARLA ground-truth ego state, a route point 20 m ahead, route re-ranking at weight 4σ), compared with published rows its authors call unmatched. 73.27 DS is below twelve entries on [[concepts/bench2drive.md]].

---

## Method

### Shared prediction and scoring

$$q_{t,i}=\phi(\operatorname{sg}[\tau_{t,i}]),\quad Q_t=[q_{t,1},\dots,q_{t,N}],\quad Z_t=P(Q_t,s_t),\qquad \boldsymbol\ell_{t,i}=H(z_{t,i}+e_t)$$

Four transformer blocks (self-attention among candidates, cross-attention to scene tokens, FFN; width 256, FFN 1024), then output normalization. Candidate coordinates are detached, so scoring gradients do not reach the generator through its outputs; they do train $\phi$, $P$, $H$ and the scene features.

**Ranking score** with $p_{t,i,m}=\sigma(\ell_{t,i,m})$, $\mathcal G=\{\mathrm{NC,DAC,DDC}\}$, $\mathcal A=\{\mathrm{TTC,EP,C}\}$:

$$\hat J_{t,i}=\sum_{m\in\mathcal G}\alpha_m\log p_{t,i,m}+\log\Big(\sum_{m\in\mathcal A}\beta_m p_{t,i,m}\Big)$$

Weights $(\alpha_\mathrm{NC},\alpha_\mathrm{DAC},\alpha_\mathrm{DDC},\beta_\mathrm{TTC},\beta_\mathrm{EP},\beta_\mathrm C)$ = (1, 1, 0, 5, 5, 2) for v1 (the metric's own) and **(10, 13, 6, 14, 15, 2.1) for v2**; how the v2 weights were chosen is not stated. No head predicts TLC, LK, HC or EC.

### Outcome-grounded predictive learning

**Future anchor (JEPA-style).** The logged trajectory is queried alone, with scene tokens detached:

$$z_t^\mathrm{exec}=P([\phi(\tau_t^\mathrm{exec})],\operatorname{sg}[s_t]),\qquad u_t^\mathrm{fut}=f_\mathrm{DINO}(o_{t+2\mathrm s}),\qquad \mathcal L_\mathrm{fut}=\frac1M\sum_n\big[1-\cos\big(g(z_n^\mathrm{exec}),u_n^\mathrm{fut}\big)\big]$$

$f_\mathrm{DINO}$ is a frozen DINOv2 ViT-S/14 with registers on the front view, mean-pooled to 384 dimensions. Because the executed query is a singleton, self-attention reduces to $W_Vx$ and **the self-attention query and key projections get no gradient from this loss: 524,288 of 4,504,320 shared parameters (11.6%)**, trained by the outcome losses alone.

**Outcome losses.** With $b(\ell,y)=\log(1+e^\ell)-y\ell$ and $\mathcal M=\{\mathrm{NC,DAC,DDC,TTC,EP,C}\}$:

$$\mathcal L_\mathrm{score}=\sum_{m\in\mathcal M}\langle b(\ell_m,y_m)\rangle_\mathrm{gen},\qquad \mathcal L_\mathrm{bank}=\frac15\sum_{m\ne\mathrm{EP}}\langle b(\ell_m,y_m)\rangle_\mathrm{bank}+0.1\,\langle|\sigma(\ell_\mathrm{EP})-y_\mathrm{EP}|\rangle_\mathrm{bank}$$

Generated progress uses soft-label cross-entropy; NC/DDC targets of 0.5 are mapped to 0; invalid labels are masked.

**Total loss**: $\mathcal L=\mathcal L_\mathrm{WTA}+\mathcal L_\mathrm{score}+\lambda_\mathrm{fut}\mathcal L_\mathrm{fut}+\lambda_\mathrm{bank}\mathcal L_\mathrm{bank}$ with $\lambda_\mathrm{fut}=1$, $\lambda_\mathrm{bank}=0.5$. WTA L1 is summed for the logged 4 s trajectory and the same trajectory time-warped to 5 s.

**Candidate bank** (following CLOVER): scene-matched perturbed trajectories with precomputed simulator labels; 54.5 valid entries per scene on average *(2026-10-14: more than the 50 + 3 per scene that [[sources/clover.md]] retains, so probably drawn from its scored pool of up to 180)*, 16 sampled per step (30.8% of a scene's pool); 310 scenes have no bank entries and 10.7% cannot fill the quota in every safety category.

**Simulator labels** (Appendix B.1): a model-independent cache (ego state, centerline, lanes, drivable area, PDM planner progress, 51 occupancy steps as STRtree polygons); each candidate resampled to 40 poses at 0.1 s and tracked by LQR on a kinematic bicycle model for 4 s; polygon queries for NC/DAC/DDC. Training progress is normalized by max(candidate, PDM planner progress). **Online labelling of 64 candidates per sample on CPU (NumPy, Shapely) dominates training time.**

### Proposition 1 (future supervision and decision stability)

Around a fitted $\theta_0$, with generalized Gauss–Newton curvatures $G_\mathrm{out}$ (plus $\mu I$) and $G_\mathrm{fut}$, and $\rho_{ij}(G)=\max\{|c_{ij}^\top\delta|:\delta^\top G\delta\le\epsilon^2\}$ for $c_{ij}=\nabla_\theta(\hat J_i-\hat J_j)$:

$$\rho_{ij}(G_\mathrm{out}+\lambda_\mathrm{fut}G_\mathrm{fut})\le\rho_{ij}(G_\mathrm{out}),$$

strictly iff $J_\mathrm{fut}G_\mathrm{out}^{-1}c_{ij}\ne0$. If $\hat J_i-\hat J_j>\rho^\mathrm{out+fut}_{ij}$ the linearized ordering holds throughout. Appendix C.4 adds the same identity for extra outcome labels: $\rho(G)^2-\rho(G_+)^2=\epsilon^2a^\top(I+BG^{-1}B^\top)^{-1}a$, $a=BG^{-1}c_{ij}$. The paper limits the claim to local first-order sensitivity, not training dynamics or planning performance.

### Inertial re-ranking

$$c_{t,i}=c_\mathrm{EC}(\pi_{t-1},\tau_{t,i})\in\{0,1\},\qquad \Psi_{t,i}=\log(0.01+0.99\,c_{t,i}),\qquad i_t^\star=\arg\max_i\big[\hat J_{t,i}+\lambda_\mathrm{state}\Psi_{t,i}\big]$$

$\lambda_\mathrm{state}=1$. Frames of each log are processed in order and the selected plan carried forward; LQR rollouts at 0.1 s are compared over the shared 3.5 s window; the first frame uses the learned score; ties go to the lowest index. Recomputed EC matches the benchmark on all 10,040 adjacent pairs (2,106 frames have no valid pair and are not scored for EC). Logged human plans pass the check on 90.1% of pairs.

### Training

DrivoR's public software stack; 103,288 examples (the size of navtrain); 4 devices × 16 (batch 64); AdamW (0.9, 0.999, wd 0.01), peak LR 2×10⁻⁴, 10% warmup then cosine, 25 epochs; 672×1148 images, no augmentation; seed 3. Reported checkpoint: epoch 23, chosen on validation. **The Table 2a controls use the last epoch.** About 40 hours on four 80 GB A800/A100 GPUs.

---

## Figures

![[teaser_paradigms_compact.png|Three panels. (a) Generate and select: four camera views and ego state feed a trajectory generator; its three candidates go to a scorer whose trajectory scores carry simulator check or cross marks; outcome gradients flow down. (b) World-model planning: a world model produces a future state (an observed future image) and candidates; a scorer scores them; future-prediction gradients go to the world model. (c) Ours: a trajectory generator feeds World4Scorer, which outputs trajectory scores with simulator labels and, during training only, future states: one observed future image between two question-marked unobserved futures]]

*Figure 1: Gradient routing in generator–scorer planning. (a) Outcome labels train the scorer. (b) A world model adds future prediction, but only the executed trajectory has an observed future. (c) World4Scorer: outcome labels supervise every candidate's state; the observed future anchors the predictor.*

![[Figure2_lewm.png|Top, overall architecture: multi-view images enter a LoRA-tuned DINO encoder giving scene tokens s_t; a trajectory generator outputs tau_1 to tau_64; an action encoder feeds a four-times-repeated Latent World Model block with self-attention among candidate queries, cross-attention to scene tokens, and an FFN; outputs z_1 to z_64 plus ego state e_t go to score heads giving scores such as 0.3, 0.6, 0.9, 0.2; inertial re-ranking with the previous plan picks the final trajectory. Bottom, joint training: left, the logged plan tau-exec goes through the latent world model with stop-gradient on s_t, a visual readout is compared by a future loss with a frozen DINO embedding of the image at t plus 2 s; right, 64 generated and 16 bank trajectories go through the same world model in separate calls, and score heads are trained by an outcome loss against simulator labels]]

*Figure 2: Overview. (a) 64 generated candidates scored by World4Scorer, then inertial re-ranking. (b) Future observations supervise visual prediction for the executed trajectory; outcome labels supervise all generated candidates and 16 bank trajectories.*

![[fig_identifiability.svg|Two 3D score surfaces over candidate space labelled Model A (blue) and Model B (grey), both passing through the labelled outcome points. Left, outcome constraints only: Model A ranks candidate i above j, Model B ranks j above i. Right, with future constraints added: both surfaces agree that i is above j while still differing elsewhere]]

*Figure 3: Complementary supervision. Two score surfaces fit the outcome labels but favour different candidates; future constraints remove that disagreement here while allowing different predictions elsewhere. Illustrative.*

![[fig_bank.svg|Bar chart of simulator outcome score fractions. Generated proposals: 12% in 0 to 0.1, 24% in 0.1 to 0.9, 64% in 0.9 to 1. Candidate bank: 35%, 47%, 17%]]

*Figure 4: Candidate bank. Generated proposals are 64% high-scoring (0.9–1); the bank is 35% low-scoring (0–0.1) and 47% mid-range.*

![[fig_bev.svg|Two bird's-eye panels of a right turn into a street with parked cars. (a) Previous frame: human, original and re-ranked plans overlap along the curve. (b) Next frame: the original framewise choice jumps to a longer straight continuation, while the re-ranked plan stays near the previous plan and the human trajectory]]

*Figure 5: Consecutive plans through a right turn, 0.5 s apart. In (b) re-ranking picks a plan closer to the previous choice and to the human trajectory.*

![[b2d_main.png|Bench2Drive comparison of Ours (purple) and VAD (yellow). (a) Unsignalized right turn, eight camera frames each; VAD drifts out of its lane. (b) Rainy intersection; VAD enters the opposing lane and collides with an oncoming vehicle. (c) Enlarged collision sequence at 19.45 to 20.95 s, and route maps for (a) and (b) with the collision marked]]

*Figure 6: Closed-loop trajectories on Bench2Drive. VAD leaves its lane in (a), scoring 77.77, and collides in (b), scoring 43.93; Ours completes both with DS 100.*

![[fig_selection_story.svg|Twelve bird's-eye scenes in three columns: left turns, straight, right turns. In each, Ours (purple with a star) follows or ends near the logged human path (brown diamonds); the plans chosen with shuffled states (grey solid) and with the scene-mean state (grey dashed) diverge, for example going straight in a left-turn scene or turning in a straight scene]]

*Figure 7: Candidate-specific states guide selection. Twelve selected scenes where Ours reaches pool-best v1 PDMS; grey marks the shuffled-state and scene-mean interventions on the same 64-candidate pool.*

![[fig4_query_scan.png|Six junction scenes. Each row: the candidate plans (left, straight and right highlighted), the current photograph, three decoded readouts for the left, straight and right queries, and the observed future image. The readouts shift viewpoint with the query while keeping scene layout]]

*Figure 8: Visual readouts for alternative trajectory queries, +2 s, from a separately trained reconstruction network. Only the logged trajectory has an observed future; the paper states these are not tests of counterfactual accuracy.*

![[cube_pair_bar.png|Three OGBench-Cube episodes with a robot arm, eight frames each for LeWM (yellow) and Ours (purple). (a) Early release: LeWM drops the cube at step 10 and fails 32.95 cm from the goal; Ours succeeds at 3.98 cm. (b) Transport away from the goal: LeWM fails at 39.91 cm, Ours succeeds at 2.89 cm. (c) Release during transport: LeWM fails at 21.32 cm, Ours succeeds at 3.37 cm]]

*Figure 9: Outcome-based scoring corrects unsuccessful LeWM plans (success: within 4 cm of the goal cube). Selected cases.*

![[decode_pipeline.png|Diagram: left, straight and right trajectory queries and scene tokens pass through the frozen latent world model and visual readout to give three predicted features u-hat; the observed future image passes through frozen DINO to give u-fut; each feature, with the current patch grid X_t, goes through a trainable CondGrid network and a frozen RAE decoder to produce four images]]

*Figure 10: From query to image. CondGrid is trained post hoc on 6,000 navtrain current/future pairs with the driving model frozen.*

![[appendix_query_examples.png|Six further road scenes (broad junctions, urban streets, curved roads), each with plans, current photo, three query readouts and the observed future at +2 s]]

*Figure 11: Additional visual readouts under alternative plans. Qualitative.*

![[appendix_condgrid_full.png|Gallery of thirty navtest scenes, three per row, each with four images: current frame, executed-query readout, full-grid decoder ceiling, and observed future at +2 s, sorted by decreasing current-to-future feature change]]

*Figure 12: Complete post-hoc decoding gallery (30 pre-specified scenes). Low-change scenes are legible because the current grid supplies spatial structure; large viewpoint changes leave the pooled feature underdetermined.*

![[fig_scene_states.svg|Four scenes. Top: first two principal components of 64 candidate states, coloured purple (no collision and no drivable-area violation), red (collision) and yellow (drivable-area violation); the groups form separate clusters. Bottom: the same bank trajectories in the ego frame, forward 0 to 60 m and lateral minus 4 to 4 m, in the same colours]]

*Figure 13: Candidate states within four navtrain scenes (two principal components explain 64–75% of state variance).*

![[b2d_showcase_5s.png|Six Bench2Drive scenarios, one per row, each with six front-camera frames at 5 s intervals and a small route map showing the reference route dashed and the driven path in purple]]

*Figure 14: Six completed Bench2Drive scenarios (DS and route completion 100 each). Selected cases.*

---

## Tables

### Table 1: NAVSIM v1 and v2 navtest (%)

Superscripts 1 and 2 mark native v1 and v2 sub-scores; only the v2 NC and DAC are printed. The last two columns are computed here ([[concepts/navsim-benchmark.md#residual-sign]]). **v1 check** = reported PDMS − NC²·DAC²·(5 TTC¹ + 5 EP¹ + 2 C)/12, approximate because v1 NC/DAC are not printed; a per-scene-averaged PDMS normally sits above it. **v2 residual** = closed form − reported EPDMS; negative is corrected-like, positive pre-fix-like.

| Method | Venue | In | NC² | DAC² | TLC | DDC | EP¹ | EP² | TTC¹ | TTC² | C | LK | HC | EC | PDMS | EPDMS | v1 check | v2 residual |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *End-to-end* | | | | | | | | | | | | | | | | | | |
| TransFuser | TPAMI'23 | C+L | 97.8 | 92.1 | 99.8 | 99.2 | 79.2 | 87.5 | 92.8 | 96.6 | 100.0 | 95.7 | 98.3 | 88.7 | 84.0 | 83.9 | +4.4 | −1.1 |
| DiffusionDrive | CVPR'25 | C+L | 98.2 | 96.2 | 99.8 | 99.5 | 82.2 | 87.4 | 94.7 | 97.3 | 100.0 | 97.0 | 98.3 | 87.4 | 88.1 | 88.2 | +2.7 | −0.9 |
| DiffusionDrive-V2 | arXiv'25 | C | 97.7 | 96.6 | 99.8 | 99.2 | 87.5 | 88.9 | 94.8 | 97.2 | 99.9 | 96.0 | 97.8 | 91.0 | 91.2 | 87.5 | +3.8 | +0.1 |
| DAWN | arXiv'26 | C | 97.3 | 92.0 | 99.7 | 99.1 | 84.3 | 87.4 | 96.0 | 96.6 | 100.0 | 96.0 | 98.3 | 85.5 | 89.1 | 83.2 | +6.9 | −1.4 |
| LWDrive | arXiv'26 | C | 98.8 | 98.4 | 99.7 | 99.0 | 87.3 | 90.3 | 96.2 | 98.6 | 99.8 | 96.3 | 97.9 | 73.3 | 92.0 | 89.6 | +1.5 | −0.9 |
| UniTeD | ECCV'26 | C | 99.1 | 97.2 | 99.9 | 99.6 | 84.1 | 86.9 | 96.6 | 98.5 | 100.0 | 98.4 | 99.9 | 87.3 | 90.2 | 90.1 | +1.6 | −0.4 |
| DrivoR | CVPR'26 | C | 99.3 | 99.2 | 99.8 | 99.7 | 89.9 | 87.4 | 96.7 | 98.8 | 100.0 | 95.8 | 98.0 | 75.5 | 93.7 | 90.7 | +0.7 | −0.7 |
| *Vision-language-action* | | | | | | | | | | | | | | | | | | |
| ReCogDrive | ICLR'26 | C | 98.3 | 97.7 | 99.8 | 98.4 | 87.3 | 89.9 | 94.9 | 97.4 | 100.0 | 90.9 | 97.8 | 27.6 | 90.8 | 82.7 | +1.9 | −2.0 |
| DriveVLA-W0 | ICLR'26 | C | 98.5 | 99.1 | 99.7 | 98.0 | 83.3 | 86.4 | 95.3 | 98.1 | 99.3 | 93.2 | 97.9 | 58.9 | 90.2 | 86.1 | +1.4 | −1.3 |
| DriveWorld-VLA | ICML'26 | C | 98.6 | 99.1 | 99.8 | 99.6 | 85.9 | 87.4 | 96.1 | 97.9 | 100.0 | 97.0 | 97.8 | 78.6 | 91.3 | 86.8 | +0.9 | +2.6 |
| ExploreVLA | ECCV'26 | C | 98.8 | 96.2 | 99.8 | 99.6 | 83.5 | 87.1 | 96.5 | 98.2 | 99.9 | 97.8 | 98.3 | 86.8 | 90.4 | 88.8 | +3.3 | −0.7 |
| WAM-Flow | CVPR'26 | C | 99.5 | 98.4 | 99.9 | 99.4 | 82.3 | 84.6 | 97.0 | 99.3 | 99.7 | 96.9 | 96.9 | 33.8 | 90.3 | 84.9 | +0.9 | −1.4 |
| *World-model augmented* | | | | | | | | | | | | | | | | | | |
| WoTE | ICCV'25 | C+L | 98.5 | 96.8 | 99.8 | 98.8 | 81.9 | 86.1 | 94.9 | 97.9 | 99.9 | 95.5 | 98.3 | 82.9 | 88.3 | 87.6 | +2.2 | −1.0 |
| WorldRFT | AAAI'26 | C | 97.8 | 96.5 | 99.8 | 99.5 | 81.7 | 88.5 | 94.0 | 97.0 | 100.0 | 97.4 | 98.1 | 69.1 | 87.8 | 86.7 | +3.0 | −1.4 |
| GraphWorld | arXiv'26 | C | 98.4 | 98.8 | 99.1 | 99.1 | 83.2 | 85.9 | 95.5 | 97.9 | 100.0 | 96.0 | 97.8 | 74.6 | 90.1 | 89.5 | +1.5 | −2.6 |
| Discrete-WAM | arXiv'26 | C | 98.5 | 98.2 | 99.8 | 99.7 | 88.7 | 90.5 | 95.3 | 97.9 | 100.0 | 97.2 | 98.3 | 78.1 | 92.2 | 90.4 | +1.9 | −0.8 |
| DriveFuture | arXiv'26 | C | 98.8 | 99.1 | 99.9 | 99.6 | 84.2 | 86.6 | 95.4 | 98.4 | 100.0 | 96.4 | 98.3 | 74.8 | 90.7 | 89.9 | +1.1 | −0.8 |
| SafeDrive | CVPR'26 | C+L | 99.5 | 99.0 | 99.9 | 99.7 | 84.3 | 88.6 | 97.2 | 98.9 | 100.0 | 97.5 | 98.2 | 81.9 | 91.6 | 87.5 | +0.7 | **+4.0** |
| Drive-JEPA | arXiv'26 | C | 98.4 | 98.6 | 99.8 | 99.7 | 91.4 | 91.4 | 96.2 | 97.8 | 99.9 | 97.6 | 97.9 | 76.9 | 93.7 | 90.8 | +1.7 | −0.9 |
| **World4Scorer** | – | C | 99.3 | 99.2 | 99.6 | 99.7 | 91.3 | 90.0 | 96.0 | 98.8 | 100.0 | 96.1 | 97.8 | 77.0 | **94.0** | 91.4 | +0.7 | −0.6 |
| **+ inertial re-ranking** | – | C | 99.3 | 99.2 | 99.6 | 99.7 | – | 89.8 | – | 98.7 | – | 96.2 | 97.8 | **92.8** | – | **93.0** | – | −0.3 |

### Table 2: Ablations

**(a) Auxiliary target** (native v1 PDMS; same architecture, seed and bank supervision; last epoch)

| Aux. target | Realized future | PDMS | Δ |
|:-:|:-:|---:|---:|
| ✗ | ✗ | 93.01 | – |
| ✓ (current frame) | ✗ | 92.70 | −0.31 |
| ✓ | ✓ | **93.96** | **+0.95** |

**(b) Outcome-label permutation during training** (512 training scenes, 60 epochs; 511 held-out scenes from disjoint logs. Sel. = selected PDM on 0–1, random pick 0.524; Order = pairwise ordering accuracy)

| Permuted | Sel. | Order |
|---|---:|---:|
| none | 0.728 | 64.2% |
| candidate bank | 0.647 | 55.6% |
| all | 0.461 | 51.5% |

**(c) Inertial re-ranking** (native v2, fixed weights and candidates)

| Score | off | on | Δ |
|---|---:|---:|---:|
| EC | 77.0 | 92.8 | +15.8 |
| EP | 90.0 | 89.8 | −0.2 |
| EPDMS | 91.4 | 93.0 | +1.6 |

### Tables 3 and 13: Bench2Drive (Driving Score)

Table 3 in the main text is the subset marked †. Published reference rows; training data and inputs are not matched.

| Method | DS |
|---|---:|
| AD-MLP (2024) | 18.05 |
| TCP (2022) | 40.70 |
| UniAD-Tiny (2023) | 40.73 |
| VAD (2023) † | 42.35 |
| UniAD-Base (2023) | 45.81 |
| TCP-traj w/o distillation (2022) | 49.30 |
| TCP-traj (2022) | 59.90 |
| WoTE (2025b) † | 61.71 |
| ThinkTwice (2023b) † | 62.44 |
| DriveAdapter (2023a) † | 64.22 |
| Drive-JEPA (2026) † | 64.52 |
| SafeDrive (2026) † | 66.77 |
| ReCogDrive (2026b) | 71.36 |
| **World4Scorer** † | **73.27** (SR 46.36%) |

### Table 4: Candidate states affect selection

All 12,146 navtest scenes, DrivoR's released 64 candidates, World4Scorer's epoch-23 scorer. Shuffled = mean of 100 fixed-point-free permutations.

| Candidate state | PDMS ↑ | Drop ↓ |
|---|---:|---:|
| Own state | 94.19 | – |
| Shuffled states | 79.73 | 14.47 |
| Scene-mean state | 71.10 | 23.09 |

### Table 5: OGBench-Cube

| Method | Success (%) |
|---|---:|
| GCIQL (reported by LeWM) | 64 |
| GCIVL (reported by LeWM) | 56 |
| PLDM (reported by LeWM) | 65 |
| LeWM (11 seeds × 50 paired episodes, rerun here) | 68.91 |
| **LeWM + outcome-based scoring** | **73.64** |

### Table 6: Notation (abridged)

| Symbol | Meaning |
|---|---|
| $t$, $i,j$, $N$ | Decision time, candidate indices, candidate count ($N=64$) |
| $o_t$, $o_{t+2\mathrm s}$; $s_t$, $e_t$ | Current and +2 s observation; scene tokens and encoded ego state |
| $\tau_{t,i}$, $\mathcal C_t$, $\tau_t^\mathrm{exec}$ | Candidate, candidate set, logged trajectory |
| $\phi$, $P$, $g$, $H$ | Action encoder, shared predictor, visual readout, score heads |
| $z_{t,i}$, $z_t^\mathrm{exec}$; $f_\mathrm{DINO}$, $u_t^\mathrm{fut}$ | Candidate and executed-query states; frozen encoder and its future embedding |
| $\hat J_{t,i}$, $\alpha_m$, $\beta_m$ | Log-domain ranking score and component weights |
| $\mathcal L_\mathrm{WTA}$, $\mathcal L_\mathrm{score}$, $\mathcal L_\mathrm{fut}$, $\mathcal L_\mathrm{bank}$; $\lambda_\mathrm{fut}=1$, $\lambda_\mathrm{bank}=0.5$ | Losses and weights |
| $\pi_{t-1}$, $c_{t,i}$, $\Psi_{t,i}$, $\lambda_\mathrm{state}=1$ | Previous plan, EC compatibility, penalty, its weight |
| $A_F$, $A_O$, $G_O$, $G$, $c_{ij}$, $\rho_{ij}$ | Analysis quantities of Proposition 1 |
| $\widehat u_{t,i}$, $X_t$, $\widehat X_{t+2\mathrm s,i}$ | Predicted feature, current patch grid, reconstructed future grid |
| $\lambda_\mathrm{cube}$ | Weight of the learned utility in the LeWM planner |

### Table 7: Training configuration

| Setting | Value |
|---|---|
| Data and batches | 103,288 examples; 4 devices × 16 samples (64 total) |
| Optimization | AdamW, (β₁, β₂) = (0.9, 0.999), ε = 10⁻⁸, weight decay 0.01 |
| Schedule | 25 epochs; peak LR 2×10⁻⁴; linear warmup over the first 10% of steps, cosine to zero |
| Image input | 672×1148, ImageNet normalization, no crop, flip or colour jitter |
| Backbone | DINOv2 ViT-S/14 with rank-32 LoRA |
| Predictor | Four decoder layers; width 256, feed-forward 1024 |
| Random seed | 3 |

### Table 8: Compute and parameter scale

GFLOPs count one multiply–accumulate as one operation and exclude fused attention. Latency: fp32, batch 1, A100.

| Method | Image size | Cams | Encoder | Params | GFLOPs | Latency (ms) |
|---|---|---:|---|---:|---:|---:|
| RAP-Dino | 448×768 | 4 | ViT-H | 888M | 4760 | 690 |
| GTRS-Dense | 512×2048 | 1 | ViT-L | 321M | 1730 | 400 |
| ZTRS | 512×2048 | 2 | V2-99 | 81M | 840 | 193 |
| GTRS-Aug | 512×2048 | 1 | V2-99 | 171M | 439 | 243 |
| GTRS-Dense | 512×2048 | 1 | V2-99 | 81M | 404 | 96 |
| DrivoR | 672×1148 | 4 | ViT-S | 41M | 351 | 110 |
| **World4Scorer** | 672×1148 | 4 | ViT-S | 35.52M | 349.9 (927.1 with attention) | 106.5 |

World4Scorer's latency excludes inertial re-ranking (125.4 ms on CPU for 64 candidates). Generation and scoring take 1.4 of 106.5 ms. Same graph: 75.6 ms with TF32, 23.0 ms in fp16 (encoder 20.2 ms); all reported scores use fp32.

### Table 9: Cross-query readout consistency

Patch-wise cosine between query-conditioned reconstructed grids, pooled over three query pairs and six scenes (higher = less variation between queries).

| Region | Grid cosine |
|---|---:|
| Road | 0.750 |
| Dynamic objects | 0.639 |
| Other static | 0.606 |

### Table 10: Candidate-state interventions with a fixed pool

DrivoR's released 64 candidates on all 12,146 v1 test scenes; common pool oracle **99.33 PDMS**; selected PDMS = oracle − regret. Intervals bootstrap over the 136 evaluation logs.

| Intervention | Oracle regret (95% CI) | Δ vs. matched (95% CI) |
|---|---:|---:|
| Matched candidate state | 5.14 [4.66, 5.62] | – |
| Scene-mean replacement | 28.23 [25.97, 30.50] | +23.09 [+20.98, +25.17] |
| Within-scene derangement | 19.86 [18.65, 21.08] | +14.72 [+13.72, +15.75] |

### Table 11: Extended-comfort failure taxonomy

2,717 framewise EC failures among 10,040 consecutive pairs.

| Failure class | Share of failures | Repaired by Ψ |
|---|---:|---:|
| Score–continuity conflict (a passing alternative exists and loses by more than the near-tie threshold) | 71.3% | 97.6% |
| Near-tie (a passing alternative within 4.1×10⁻⁵) | 0.7% | 95% |
| Pool-limited (no candidate passes) | 27.9% | 3.8% |

### Table 12: History-source controls for inertial re-ranking

On "Coverage", a variant whose generator is also trained to cover high-scoring bank trajectories (fixed pool and scores).

| Previous plan supplied | v2 EPDMS |
|---|---:|
| Memoryless (framewise) | 89.9 |
| Wrong-log previous plan | 89.5 |
| Same-log, two-step-old plan | 91.5 |
| **Correct previous plan** | **92.5** |
| Copy previous plan (no selection) | 82.0 |

Correct vs. wrong-log: +2.96 [2.63, 3.34]; vs. two-step-old: +1.01 [0.86, 1.16].

### Results given only in the text

| Quantity | Value |
|---|---|
| State reassignment, 100 repeats | 77.44–81.99 PDMS, mean 79.73 (log-cluster 95% CI [78.64, 80.85]); every repeat below matched; loss 14.47 [13.65, 15.29]; selection changes on 100% of scenes; ordering accuracy 49.9% |
| DrivoR's own scorer, same test | Matched regret 5.64; reassignment +14.11 [+13.01, +15.19]; ordering accuracy 72.5% → 50.2% |
| v1 vs v2 objectives on frozen pools (simulator sub-scores) | Different argmax in 43% of scenes; within-scene rank correlation 0.69 (0.96 without EC; argmax disagreement 3%); v1-optimal fails EC in 99.7% of disagreements; v2-optimal costs 1.3 v1 PDMS; utility gap 98% EC, 3% DDC, −1% EP/TTC/C; v2-optimal +50.4 EC, −2.7 EP |
| Progress–comfort tension | EP = 1.0 for 43% of candidates; no top-quartile-EP and EC-passing candidate in 38% of scenes (57% in the most dynamic quarter); within-scene EP–EC correlation +0.07 |
| Pool limits on EC | DrivoR: 7.3% of pairs have no compatible candidate (EC ceiling 92.66). World4Scorer after re-ranking: 91.7% of residual failures are pool-limited; human plans pass on 16.5% of pool-limited pairs |
| Consecutive selections | Re-ranking changes 1,939 pairs from fail to pass and 18 from pass to fail; heading jitter −32%, lateral discrepancy −21%, consecutive-plan L2 −11%; changed selections move from 1.87 to 1.51 m of the logged plan |
| Re-ranking on DrivoR | 90.66 → 92.27 EPDMS (+1.61) |
| λ_state sensitivity (Coverage pool) | 0.25 / 0.5 / 1 / 2 → 92.42 / 92.47 / 92.49 / 92.45 |
| Future target geometry | Current vs +2 s DINOv2 features: mean cosine 0.94 (5th percentile 0.80); effective rank 49 of 384; cross-scene pairing 0.649; true vs shuffled AUC 0.981 (Cohen's d 2.84); same-log pairs 0.76 |
| Candidate-state geometry (128 navtrain scenes × 64 bank candidates) | Effective rank 17.9 of 256; within-scene differences 75.8% of variance; 5-NN outcome-group agreement 80.2% in state space vs 76.8% in trajectory space vs 52.0% shuffled; state beats trajectory in 98 of 128 scenes |
| Readout fidelity | Observed-future vector improves regional patch cosine +0.005 to +0.041 over a vector-free branch; for the selected query whole-grid cosine 0.681 (guidance 2) / 0.707 (guidance 1) vs 0.701 without the vector |
| Bench2Drive setup | Route point 20 m ahead (on the turn side in 97.4% of left- and 100% of right-turn frames); CARLA ego pose, velocity, acceleration (reference agents estimate them from GNSS/IMU); command retention; route re-ranking at weight 4σ; vocabulary negatives instead of the CLOVER bank; future loss weight 0.1, predictor gradient scaled 0.05; 15 epochs; map-loading radius changed on 8 routes to avoid crashes |
| OGBench-Cube | 379/550 → 405/550 (+4.73, 95% CI [+2.18, +7.27]); 39 rescued, 13 lost; 9 seeds up, 1 tie, 1 down; seed 42: 72% → 80%; λ_cube = 0.3 tuned on seeds 200–203; scorer is an 819 → 256 → 128 → 1 MLP on LeWM features trained on simulator outcomes of LeWM's own CEM candidates (seeds 100–107) |
| Compute | 4 × 80 GB A800/A100, about 40 h; A30 batch-32 inference 2.6 FPS, 14 GiB, 1.3 h for navtest |

---

## Reading the Results

### 1. What the world model is worth {#future-anchor}

| Evidence | What it shows | What it does not |
|---|---|---|
| Table 2a: no target 93.01, current-frame target 92.70, +2 s target 93.96 | A future target helps (+0.95) where a present one does not (−0.31), with objective and pathway fixed | Single seed, last-epoch checkpoints; ForeDrive's training-seed sd (0.1) belongs to another recipe |
| Proposition 1 | Adding any supervised readout can only narrow the local range of score differences | It holds for the current-frame target too, which hurt. The proposition cannot explain why one target helps and the other does not |
| Table 2b: label permutation | Outcome labels must be paired with their trajectories | Any supervised scorer would show this; it is a sanity check, run on 512 scenes |
| Table 4 / Table 10: state shuffling | The scorer reads each candidate's own state | The trajectory reaches the heads only through its state, and DrivoR's non-predictive scorer loses the same amount (+14.11 regret) |
| Appendix C.3: outcome clustering | States group by simulator outcome (80.2%) | Raw trajectories nearly do too (76.8%), and the scenes are training scenes |
| Figures 8, 10–12: readouts | Query changes shift the decoded viewpoint | The predicted vector does not raise whole-grid similarity to the observed future (0.681 vs 0.701 without it); alternatives have no ground truth |

**Reading.** The architecture already makes a "latent world model" out of a standard scorer: a scorer that cross-attends scene tokens per candidate (DrivoR's design) has a candidate state too. The distinctive addition is one cosine loss on a 384-d pooled front-view feature 2 s ahead, worth +0.95 in one run. The target is mostly the present scene (cosine 0.94), so whatever it contributes it does through a small residual of change. The paper says this itself.

**What the outcome supervision does that DA-WAM and MM-Future do not.** [[sources/da-wam.md]] and [[sources/mm-future.md]] predict one future per candidate and supervise only the expert-matched one (31 of 32, 63 of 64 unsupervised). World4Scorer supervises every candidate's state with simulator outcomes and leaves only the visual target restricted to one. That answers the wiki's standing objection to per-candidate futures, at the price of privileged simulator labels and no visual future for anything but the logged plan.

### 2. Inertial re-ranking is the evaluator's comfort check {#inertial-reranking}

- **The mechanism is the metric.** The compatibility test is NAVSIM-v2's extended-comfort criterion with its thresholds, re-implemented (it matches the benchmark on all 10,040 pairs), applied to the previous selected plan. It needs no labels or future information at deployment, but it needs a stateful agent that sees navtest frames in log order, which is not how most NAVSIM agents are evaluated.
- **Its effect is mostly the metric term**: EC +15.8, EP −0.2, EPDMS +1.6. EC 92.8 is above the human agent's 90.1 in [[sources/mm-future.md]]'s corrected table, and human plans pass the same check on 90.1% of pairs.
- **It is portable**: DrivoR 90.66 → 92.27. A comparison of World4Scorer with re-ranking (93.0) against DrivoR with re-ranking (92.27) gives +0.7, the same as framewise (91.4 vs 90.7).
- **The controls show it uses continuity, not a generic prior**: the correct previous plan beats a wrong-log one by 2.96 and a two-step-old one by 1.01; a wrong-log plan is worse than none (89.5 vs 89.9); copying the previous plan scores 82.0. Changed selections also move closer to the human (1.87 → 1.51 m) and jitter falls.
- **It has a ceiling set by the pool**: 27.9% of framewise failures have no compatible candidate, mostly in the most dynamic quarter of scenes, and re-ranking repairs only 3.8% of them.
- **It is the second continuity fix in the wiki** after [[sources/drive-jepa.md]]'s momentum-aware selector (EC 47.9 → 84.8), and the larger one. See [[concepts/selection-based-planning.md#world4scorer]].

### 3. The scorer, isolated {#scorer-isolated}

| On DrivoR's released 64-candidate pools (navtest) | PDMS |
|---|---:|
| Pool oracle | 99.33 |
| World4Scorer's scorer | 94.19 |
| DrivoR's scorer | 93.69 |
| Shuffled World4Scorer states (100 draws) | 79.73 |
| No ranking signal (scene-mean state; a fixed index wins) | 71.10 |
| World4Scorer full system, its own generator (Table 1) | 94.0 |

- **+0.50 for the scorer at fixed candidates.** This is the cleanest number in the paper and the paper does not present it as a result; it comes from the regret table's two matched rows (5.64 vs 5.14).
- **The generator adds nothing on top**: World4Scorer's scorer does better on DrivoR's candidates than on its own.
- **The ranking signal is worth about 23 PDMS on this pool** (71.10 with a fixed index → 94.19), and about 5 PDMS of headroom remain. That is a navtest analogue of [[sources/drivefuture.md]]'s navhard scorer price ([[concepts/selection-based-planning.md#scorer-price]]).
- **Data caveat (inference)**: Table 1 carries DrivoR at 93.7, which [[sources/mm-future.md]]'s table gives for DrivoR trained on trainval (93.1 on navtrain), and the released checkpoint measures 93.69 here. World4Scorer's 103,288 training examples are navtrain's count. If the released checkpoint is the trainval model, the generator pool comes from a model trained on more data, and World4Scorer's +0.3 over DrivoR is against the stronger row.

### 4. The comparison tables {#tables}

- **Own rows pass both checks**: v1 +0.7 above the closed form (small, as expected at 94), v2 −0.6 framewise and −0.3 re-ranked, corrected-like, consistent with "the latest official NAVSIM devkits".
- **The v2 column is the most uniformly corrected-like in the wiki.** Eighteen of twenty-one rows have negative residuals. The positives are DiffusionDriveV2 (+0.1) and DriveWorld-VLA (+2.6), two of the residual-sign heuristic's standing exceptions, and **SafeDrive (+4.0)**, new to the wiki and as large as the pre-fix human agent's residual (+4.1).
- **DiffusionDrive 88.2 appears with the block's sub-scores** (within 0.3 on every term; closed form 87.3; residual −0.9). [[sources/geowam.md]] printed the same 88.2 row and the wiki filed it as an anomaly because the circulating 84.5 has nearly the same sub-scores. World4Scorer does not cite GeoWAM and its row differs in the last digit on three terms. **The simpler reading now is that 88.2 is DiffusionDrive's corrected score and 84.5, which has a pre-fix-like +2.5 residual, is pre-fix**, which also explains the 84.5 row's standing exception. See [[concepts/navsim-benchmark.md#world4scorer]].
- **TransFuser at 83.9 with different sub-scores** (DAC 92.1, LK 95.7, EC 88.7 against the block's 89.9, 92.7, 87.2), corrected-like (−1.1). It is within 0.1 of GeoWAM's 84.0, which GeoWAM printed beside the block's sub-scores.
- **New or changed rows**: DrivoR 93.7 / 90.7; Drive-JEPA **93.7 / 90.8** (the Drive-JEPA page has 93.3 / 87.8; here EP¹ = EP² = 91.4 and EC 76.9 against 84.8); ReCogDrive 82.7 with **EC 27.6** (the shared block has 83.6, EC 87.7) and WAM-Flow 84.9 with **EC 33.8** (its own paper: 84.7, EC 73.9); SafeDrive 91.6 / 87.5; DAWN 89.1 / 83.2 (v1 check +6.9, the largest here, probably because its v1 DAC is not printed); UniTeD 90.2 PDMS. The paper does not say which baselines it re-evaluated.
- **Internal references**: Section 4.2 calls the main NAVSIM table "Table 4" (it is Table 1), and Table 4's caption contrasts its pool with "the Table 4 pool" (Table 10's caption says it is not "the generator-pool result").
- **Omitted**: on v1, CLEAR and DA-WAM (93.7) and DriveSuprim (93.5), all below 94.0. On v2, WA-JEPA (91.7) and MM-Future (91.5), both above the framewise 91.4; SUV, ReDrive and PhysWAM; and the group's own CLOVER (90.4 EPDMS as reported by [[sources/auto-jepa.md]]). *(2026-10-14: [[sources/clover.md]] also reports **94.5 PDMS** on v1, above this paper's 94.0, with its training split unstated; its baseline is DrivoR's released model with the same pool statistics this paper measures (oracle 99.33, selected 93.69). On v2 CLOVER's 90.4 is below the 90.66 this paper measures for DrivoR.)* WoTE's row matches GeoWAM's on every sub-score (87.6 here, 87.7 there).

### 5. NAVSIM v1 and v2 reward different plans {#v1-v2}

Appendix B.2 runs both utilities on frozen candidate pools with simulator sub-scores, so prediction error is out of the picture:

- They pick different candidates in **43%** of scenes; without EC, in 3%.
- **98%** of the utility gap is extended comfort; driving direction 3%; progress, TTC and comfort together −1%.
- The v1-optimal candidate fails the EC check in **99.7%** of disagreement scenes; the v2-optimal costs **1.3 PDMS** on v1.
- Progress is saturated (EP = 1.0 for 43% of candidates) and barely correlated with EC within a scene (+0.07); in 38% of scenes no candidate is both top-quartile in progress and EC-passing.

So a single checkpoint cannot be optimal on both ladders, and the per-benchmark weights (1, 1, 0, 5, 5, 2) vs (10, 13, 6, 14, 15, 2.1) are a necessary part of reporting both. The v2 weights' provenance is not stated; [[sources/hydra-mdp-pp.md]] found grid-searched weights worth +1.5 PDMS.

### 6. Bench2Drive {#bench2drive}

- **73.27 DS / 46.36% SR** on the 220 routes, from an adapted system: a route-point input, CARLA ground-truth ego state (the reference agents estimate it from GNSS and IMU), command retention and route re-ranking at weight 4σ, none ablated.
- **On the wiki's ladder** it is above Raw2Drive (71.36) and below twelve entries, from Hydra-NeXt (73.86) and MomWorld (74.07) up to LinkVLA (91.01). Its SR is below Raw2Drive's 50.24.
- **"ReCogDrive 71.36"** in Table 13 equals Raw2Drive's DS to the hundredth; the wiki's ReCogDrive page records no Bench2Drive result. Unverified. *(lint 2026-10-06: 71.36 is also [[sources/deepsight.md]]'s Bench2Drive success rate, so the value appears twice on the ladder already.)*
- No future-loss or bank ablation in CARLA, and the future loss is down-weighted there (0.1, gradient into the predictor ×0.05).

### 7. OGBench-Cube

- A separate MLP scorer on LeWM's frozen predicted and goal features, trained on simulator outcomes of LeWM's own CEM candidates, added to LeWM's distance cost. **+4.73 points (95% CI +2.18 to +7.27)** over 11 seeds of 50 paired episodes; 39 episodes rescued, 13 lost.
- **It tests outcome-based scoring, not World4Scorer's mechanism**: no shared predictor, no future anchor. The upper three rows are LeWM's reported numbers, not reruns.

### 8. Cost

- 35.52M parameters and 106.5 ms (fp32, A100) are about DrivoR's, and the scorer is 1.4 ms of it. Re-ranking adds 125.4 ms on CPU, so the v2 configuration runs at about 232 ms unless the LQR loop is vectorised.
- The training cost is online simulation: 64 candidates per sample labelled on CPU every step.

---

## Relationships

- **[[sources/drivor.md]]**: the base system. Same encoder, generator, candidate count and software stack; World4Scorer replaces its scorer. Also the encoder source of [[sources/mm-future.md]] and second on navhard (54.6).
- **CLOVER** ([[sources/clover.md]], ingested 2026-10-14): the bank and the simulator-label protocol. Same group (Sining Ang, Yan Wang). CLOVER is now load-bearing in three wiki papers: [[sources/auto-jepa.md]] (scorer initialization), [[sources/adaptive-wam.md]] (pseudo-expert targets) and this one (bank).
- **[[sources/da-wam.md]]**, **[[sources/mm-future.md]]**: per-candidate futures read by a scorer, supervised only for the expert. World4Scorer is the third per-candidate design and the first to supervise every candidate's state, with outcomes rather than futures.
- **[[sources/drive-jepa.md]]**: the other continuity selector (momentum-aware, EC 47.9 → 84.8), and a Drive-JEPA row (93.7 / 90.8) this paper reports that the wiki had not seen.
- **[[sources/hydra-mdp-pp.md]]**, **[[sources/drivesuprim.md]]**: the simulator-distilled multi-head scorer recipe and its log-domain weighted combination, which World4Scorer reuses.
- **[[sources/geowam.md]]**: shares the DiffusionDrive 88.2 and WoTE rows; see §4.
- **[[sources/foredrive.md]]**: the other recent "future vs no future" control on navtest (auxiliary forecasting +0.1, consumption +0.4–0.6); its training-seed sd is the nearest yardstick for Table 2a.
- **[[sources/ad-e2e-jepa.md]]**: LeWM appears there as a baseline; per-frame selection without continuity gave it the lowest EC in the wiki.
- **[[sources/drivereferee.md]]**: the other paper that executes the evaluator's own logic at inference (collision and drivable-area gates on a predicted map).

---

## Limitations

**Evidence**

1. **The world-model claim rests on one single-seed, last-epoch comparison** (+0.95; current-frame target −0.31). No repeat, no navhard, no v2 version of it.
2. **The bank's contribution at full scale is not measured.** Table 2b permutes labels on 512 scenes; no full-data run drops the bank.
3. **RQ3 does not test prediction.** State shuffling breaks the only path by which a trajectory reaches the score heads; the same test on DrivoR's scorer gives the same result.
4. **Proposition 1 applies to any extra supervised readout**, including the current-frame target that hurt, and bounds variation rather than correctness.
5. **The v2 headline is the re-ranking**, a training-free use of the evaluator's own EC criterion that transfers to DrivoR (+1.61). Framewise, two wiki entries are higher.
6. **Bench2Drive is an adapted, unablated system** with privileged ego state and route-following inputs, compared with unmatched published rows; it is below twelve wiki entries.
7. **No navhard, HUGSIM or reactive result**, from a method built on the second-highest navhard system.
8. **Comparison scope**: WA-JEPA, MM-Future, CLEAR, DA-WAM, DriveSuprim and CLOVER are absent; several baseline rows are new or differ from their sources, with no statement of which were re-evaluated.

**Method**

9. **Privileged supervision**: every outcome label comes from NAVSIM's simulator with ground-truth agents and maps, the [[concepts/selection-based-planning.md#top-of-leaderboard]] pattern. The scorer predicts only v1 components; v2 weights are unexplained.
10. **The future target is one pooled front-view vector**, effective rank 49 and cosine 0.94 to the present; 11.6% of the shared predictor never sees it.
11. **Re-ranking needs ordered, stateful evaluation** and cannot fix pool-limited frames (27.9% of failures).
12. **Training cost is CPU simulation** of every generated candidate at every step.

**Source conversion**

13. The local clipping ends at the Table 1 caption; Sections 4.1–5, all tables, Figures 3–5 and 7–14 and the appendices were recovered from the arXiv HTML, and 11 figure files (6 PNG, 5 SVG) were downloaded to `raw/assets/`. `raw/` text was not modified. The Appendix C.4 proof is summarized, not reproduced.

---

## Key Cross-References

- [[concepts/selection-based-planning.md]] — the scorer as a predictor; a fixed-pool scorer price; the second continuity fix.
- [[concepts/world-model-for-ad.md]] — Pattern 49: outcome-supervised candidate states anchored by one observed future.
- [[concepts/navsim-benchmark.md]] — 94.0 / 93.0 placement, the residual audit, DiffusionDrive 88.2, v1 vs v2 objectives.
- [[concepts/bench2drive.md]] — an adapted generator–scorer entry at 73.27.
- [[concepts/evaluation-variance.md]] — log-clustered bootstrap intervals; single-seed ablations.
- [[concepts/inference-latency.md]] — 106.5 ms plus 125 ms of CPU re-ranking; a scorer-cohort latency table.
- [[concepts/counterfactual-prediction.md]] — outcome labels as the supervision available for unexecuted plans.
