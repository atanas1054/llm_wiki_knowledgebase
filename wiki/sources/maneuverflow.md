---
title: "ManeuverFlow: Maneuver Hypothesis Generation with Future Evidence for Trajectory Evaluation and Refinement in End-to-End Autonomous Driving"
type: source-summary
sources: ["raw/papers/ManeuverFlow_ Maneuver Hypothesis Generation with Future Evidence for Trajectory Evaluation and Refinement in End-to-End Autonomous Driving.md"]
related: [concepts/selection-based-planning.md, concepts/intent-conditioned-planning.md, concepts/world-model-for-ad.md, concepts/best-of-n.md, concepts/diffusion-planner.md, concepts/mixture-of-experts.md, concepts/navsim-benchmark.md, concepts/worldengine-benchmark.md, concepts/data-driven-simulators.md, concepts/evaluation-variance.md, concepts/inference-latency.md, sources/idol.md, sources/dial.md, sources/mm-future.md, sources/da-wam.md, sources/momworld.md, sources/world4scorer.md, sources/resworld.md, sources/foredrive.md, sources/geoworldad.md, sources/diffusiondrive.md, sources/diffusiondrive-v2.md, sources/pair-drive.md, sources/hydra-mdp-pp.md, sources/simvta.md, research-directions.md]
created: 2026-10-14
updated: 2026-10-14
confidence: medium
---

# ManeuverFlow: Maneuver Hypothesis Generation with Future Evidence for Trajectory Evaluation and Refinement in End-to-End Autonomous Driving

**Paper**: ManeuverFlow: Maneuver Hypothesis Generation with Future Evidence for Trajectory Evaluation and Refinement in End-to-End Autonomous Driving
**Authors**: Yuhao Wei, Yong Chen, Chen Yang, Yingchen Wang, Delin Ouyang, Lingfeng Qi, Qi Lan, Zhanhua Zhang, Jie Li, Guofa Li
**Orgs**: Chongqing University; Artificial Intelligence Center, Geely Automotive Research Institute
**arXiv**: 2610.13409v1 (October 2026)
**Code**: link given (github.com/wyh-autonomy/maneuverflow)
**Source**: `raw/papers/ManeuverFlow_ Maneuver Hypothesis Generation with Future Evidence for Trajectory Evaluation and Refinement in End-to-End Autonomous Driving.md`. The clipping has all text, all main and appendix tables, and eight of nine figures. Figure S2 (the rule-aligned M-GMP-8 prior) is missing from it; it was downloaded from the arXiv HTML as `raw/assets/maneuverflow_figS2.png`. The appendix's own text has small reference slips: "Figures S2 and S2" for S1–S2, Section 5.2 citing Table S5 for retrieval numbers that are in Table S4, and a Section 6.2 with no 6.1 (also absent from the HTML).

**Confidence: medium.** The headline rows are means of three independently trained checkpoints with small spreads, and two ablations are properly matched (a parameter-matched generator control and a prior-only control). But the component ablations have no seeds; the "future evidence" ablation removes the world-model branch entirely, so its training-time effect on the encoder is not separated from its inference-time use; the scorer and refiner are trained on online NAVSIM-evaluator labels; the v2 table mixes evaluators; and the latency figure covers only part of the pipeline.

---

## What It Is

**A generate–score–refine–rescore planner on a ResNet-34 TransFuser, with a maneuver-structured one-step generator and a scorer that reads a shared predicted future along each candidate's path.** Per planning step:

1. **Hypothesize.** A classifier predicts weights over eight trajectory-derived "maneuver" families (stop, cruise, accelerate, decelerate, left/right lane change, left/right turn). The top three are kept and 64 candidates are split among them. Each candidate starts from a sample of its family's fixed Gaussian (the Maneuver Gaussian Mixture Prior, M-GMP) and is transported to a trajectory in **one step** by an Improved-MeanFlow field whose feed-forward layers are **experts routed by the maneuver index** (M³F).
2. **Verify.** A world model predicts **one** future BEV feature field (4 states at 1–4 s, absolute and residual) from the current BEV map, the same for every candidate. A learned probe samples the current map and the time-aligned future states along each candidate's footprint, giving each candidate its own "evidence". A set scorer ("Metric Prism") predicts the five NAVSIM-v1 PDM factors per candidate and ranks by a fixed log-PDM composition.
3. **Improve.** A Frenet-frame refiner (ECFR) predicts longitudinal, lateral and heading residuals for the 16 best pre-scored candidates, conditioned on their evidence. The 16 refined candidates replace their bases, all 64 re-query the field and are re-scored, and the top one is returned.

Two configurations: **TF** (three cameras + LiDAR raster) and **LTF** (cameras only, a learned latent in place of LiDAR).

**Results**

| Benchmark | ManeuverFlow | Closest matched comparison |
|---|---:|---|
| NAVSIM-v1 PDMS (navtest) | **90.58 ± 0.05** (TF); 89.98 ± 0.06 (LTF) | TransFuser 84.01 in its own ablation; MeanFuser 89.0, DriveSuprim (R34) 89.9 in its table |
| NAVSIM-v2 EPDMS (navtest, v1 checkpoints, no v2 training) | **90.14 ± 0.04** (TF, corrected-like, residual −0.80); 89.70 (LTF) | MeanFuser 89.5 (corrected-like) |
| WorldEngine closed loop, 288 navtest-derived 3DGS scenarios (LTF) | PDMS\* **75.19 ± 0.85** (logged traffic) / **74.58 ± 0.64** (IDM traffic) | MeanFuser 73.81 / 73.28 |
| Speed | "Proposal FPS" 53.86 at $N=64$, perception through generator only | – |
| Training | 4 × RTX 4090, 60 + 10 epochs on navtrain, three runs | – |

---

## Key Takeaways

- **The ladder is mostly the scorer and the refiner.** TransFuser regression 84.01 → candidate MeanFlow with a current-feature scorer 86.14 (+2.13) → maneuver prior plus a bigger generator 87.91 (+1.77, two changes at once) → maneuver-routed experts 88.35 (+0.44 at matched parameter count) → future evidence 89.93 (+1.58) → refiner 90.58 (+0.65). The generator-side changes account for about 2.2 of the 6.6 points over TransFuser.
- **Future evidence and refinement substitute for each other.** Future evidence is worth +1.58 before refinement and +0.55 after it; refinement is worth +1.68 without future evidence and +0.65 with it. Together +2.23, against 3.26 for the two measured alone. The paper calls this sub-additive; the refiner without any future is the larger single effect.
- **The future is shared but read per candidate, and that reading is worth more than the wiki's per-candidate scorers recorded.** One action-free BEV forecast per scene, sampled along each candidate's footprint: +1.58 before refinement, against +0.15 ([[sources/da-wam.md]]) and +0.4 ([[sources/mm-future.md]]) for scorers given a separate future per candidate, and −0.50 for DA-WAM's *shared* future. It suggests DA-WAM's shared-future penalty came from the readout (one averaged future, no candidate indexing) rather than from sharing. Two cautions: the ablation removes the world model's training loss along with its inference-time evidence, and the forecast cannot show how other agents respond to a particular candidate, which the paper concedes.
- **Candidate count barely matters; the selector is the limit.** With top-3 routing, 8 candidates give 90.06 and 64 give 90.58; 128 gives 90.56. Over the same range the frozen generator's best-of-$N$ rises from 92.20 to 94.25, so the gap between oracle and selected plan widens from 2.1 to 3.7 points. Metric Prism ranks the pool's evaluator-best candidate first in 40% of scenes and picks one within one PDMS point of it in 75%. The paper's motivation is that small candidate sets limit discovery of good trajectories; its own sweep says small sets cost about half a point.
- **The maneuver prior beats K-means by +0.54, in the full pipeline.** Rule-aligned families (bins on mean heading at ±4.5° and ±13°, and on a speed-trend proxy) against eight K-means clusters with everything else matched. K-means clusters are the Gaussian-mixture prior used by MeanFuser and [[sources/mm-future.md]]. One comparison, no seeds.
- **"Maneuver" means a heading-and-speed bin.** A "lane change" is a mean heading between 4.5° and 13°, which includes gentle curves; the families are not tied to lane topology, as the appendix says. Recall@3 of 98.3% measures whether the expert's nearest family is among the top three predicted, not semantic accuracy.
- **Seed spreads are small on NAVSIM and ten times larger in closed loop.** Three training runs: sd 0.04–0.06 on PDMS and EPDMS, 0.64–0.92 on WorldEngine (288 scenarios). The WorldEngine margin over MeanFuser (+1.4 / +1.3) is about 1.5–2 of ManeuverFlow's own sds, and the baselines report none.
- **On the wiki's ladders.** v1 90.58 is below the ResNet-34 entries DiffusionDriveV2 91.2, PaIR-Drive 91.2 and iPad 91.1, none in its table, and above IDOL's 90.0. v2 90.14 (corrected-like) is the highest ResNet-34 row the wiki records, level with MomWorld (90.1) and below eight ingested corrected-cohort entries (WA-JEPA 91.7 down to PhysWAM 90.3).

---

## Method

![[fg.png|Figure 1: Candidate-based planning paradigms. (a) Query candidates without explicit mode structure, scored from current features. (b) Fixed-anchor candidates evaluated with per-candidate future rollouts. (c) ManeuverFlow with maneuver-structured generation, shared-future evidence, and Frenet refinement before joint re-scoring.]]

**Figure 1**: (a) Learned trajectory queries decoded and scored from current features. (b) WoTE-style fixed anchors with an offset head and a world-model rollout per candidate. (c) ManeuverFlow: maneuver queries and a maneuver GMM feed M³F; the world model produces one future, a trajectory-evidence refiner improves candidates, and a scorer ranks them. The appendix states that (a) is a schematic of the mode-concentration risk, not a measured collapse.

![[fg2.png|Figure 3: Overview of ManeuverFlow. Scene-conditioned M-GMP and the shared planning context drive one-step M3F generation of base hypotheses, which Metric Prism pre-scores using trajectory-specific evidence from a shared future BEV field. ECFR refines the selected R hypotheses, which join the retained bases for evidence re-query and joint re-scoring.]]

**Figure 3**: Perception (multi-view images, LiDAR, ego status; future perception at training time only) → perception encoder → maneuver routing, BEV features, ego features → M³F (Maneuver-GMP source, DiT blocks with maneuver experts, inference at $t-r=1$). World model: $\mathrm{BEV_{abs}}$ at $T{+}1 \ldots T{+}K$ and their consecutive residuals $\mathrm{BEV_{res}}$; trajectory sampling builds evidence tokens per candidate; the Evidence-Coupled Frenet Refiner updates candidates, which are sampled again; the Metric Prism decoder predicts NC, DAC, EP, TTC, C and the PDMS composition per candidate, then ranks. $\mathcal L_{\rm FBEV}$ ties predicted BEV states to encoded future observations.

### Planning context and the maneuver prior (§3.2, Appendix 1)

$$(B_n,Q^A_n,q^{\rm man}_n,c^{\rm SN}_n)=\mathcal E_\chi(\mathcal O^\chi_n),\qquad \pi^{\rm man}_n=\mathrm{Softmax}\big(\mathcal D_{\rm man}(q^{\rm man}_n,c^{\rm SN}_n)\big)$$

Each expert trajectory in navtrain is binned by three quantities: first-to-last displacement $D_n$, mean heading $\bar\psi_n$ (degrees) and a speed-trend proxy $\Delta\bar s_n$ (late minus early mean step length; threshold $\eta_s=0.25$ m per 0.5 s step).

**Table S1: maneuver families and navtrain occupancy**

| ID | Family | Condition | Count (%) |
|---:|---|---|---:|
| 0 | Stop/Crawl | $D_n<5$ m | 12,044 (11.66) |
| 1 | Cruise | $\lvert\bar\psi_n\rvert\le4.5^\circ$, $\lvert\Delta\bar s_n\rvert\le\eta_s$ | 8,695 (8.42) |
| 2 | Accelerating | $\lvert\bar\psi_n\rvert\le4.5^\circ$, $\Delta\bar s_n>\eta_s$ | 17,339 (16.79) |
| 3 | Decelerating | $\lvert\bar\psi_n\rvert\le4.5^\circ$, $\Delta\bar s_n<-\eta_s$ | 24,501 (23.72) |
| 4 | Left lane change | $4.5^\circ<\bar\psi_n\le13^\circ$ | 9,330 (9.03) |
| 5 | Left turn | $\bar\psi_n>13^\circ$ | 18,174 (17.60) |
| 6 | Right lane change | $-13^\circ\le\bar\psi_n<-4.5^\circ$ | 5,422 (5.25) |
| 7 | Right turn | $\bar\psi_n<-13^\circ$ | 7,783 (7.54) |

The counts sum to 103,288, navtrain's sample count. Each family gets a fixed factorized Gaussian over normalized per-step displacements and heading. The classifier is trained toward a soft target $y^{\rm man}_{n,m}\propto\exp(-d_{n,m})$ from the expert's normalized distance to each fitted component; the nearest component $m^\star_n$ is the training route for M³F and the target for recall@$k$.

![[kmeans_gmp_atlas.png|Figure S1: KMeans-8 control prior before M3F transport. (a) KMeans components in mean normalized displacement space; crosses mark component centers. (b) Decoded source trajectories for components C0–C7.]]

**Figure S1**: The K-means-8 control prior. Clusters are separated mainly along forward displacement; one cluster each sits at positive and negative lateral offset.

![[maneuverflow_figS2.png|Figure S2: Rule-aligned M-GMP-8 prior used by ManeuverFlow before M3F transport. (a) Expert trajectories organized by the trajectory-derived maneuver families in Table S1; crosses mark component means. (b) Decoded source trajectories for the eight components. (Recovered from the arXiv HTML.)]]

**Figure S2** *(recovered from the arXiv HTML)*: The rule-aligned prior. Cruise, accelerate and decelerate overlap along the zero-lateral axis and differ in forward displacement; the lane-change and turn families sit off-axis. Decoded means show the lane-change families as gentle lateral drifts and the turns as clear arcs.

### M³F: maneuver-routed one-step transport (§3.3)

![[fg3.png|Figure 4: Architecture of M3F. Interpolated trajectory-state tokens first query BEV and ego features, then agent memories. Time-conditioned AdaLN modulates each block, while maneuver-routed experts parameterize the interval-averaged field for one-step trajectory transport.]]

**Figure 4**: An M-GMP tokenizer turns the interpolated state into queries; a time MLP on $(t-r,t,r)$ drives AdaLN scale, gate and shift; each block cross-attends to BEV and ego features, then to agent tokens; the FFN is one of eight experts (E1–E8) chosen by the maneuver code; an iMF decoder outputs the average velocity $u_\theta$. Inference uses one full interval ($t-r=1$).

Improved MeanFlow's boundary-field construction, with $z^t=(1-t)x^{\rm gt}+t\,e$ and $v^c=e-x^{\rm gt}$:

$$v^b_\theta=u_\theta(z^t,t,t\mid C,w^{\rm man}),\quad J_\theta=\mathrm{JVP}\big[u_\theta;(z^t,r,t);(v^b_\theta,0,1)\big],\quad \tilde v_\theta=u_\theta+(t-r)\,\mathrm{sg}[J_\theta],\quad \mathcal L_{\rm M^3F}=\mathbb E\,\|\tilde v_\theta-v^c\|_1$$

Training routes all six source samples per scene through the expert's nearest component; $(r,t)$ is drawn as $(0,1)$ w.p. 0.25, $(q,q)$ w.p. 0.375, or an ordered pair w.p. 0.375, with $q=\mathrm{sigmoid}(\xi)$, $\xi\sim\mathcal N(-0.4,1)$. Inference: top-3 components, renormalized weights, 64 candidates by largest-remainder allocation, one evaluation $\tau=\Gamma^{-1}\big(e-u_\theta(e,0,1\mid C,w^{\rm man})\big)$.

### Shared future field and Metric Prism (§3.4, Appendix 2)

$$\widehat{\mathcal F}_n=\mathcal W(B_n,c^{\rm SN}_n)=\{(\widehat B^{\rm abs}_{n,j},\widehat B^{\rm res}_{n,j})\}_{j=1}^{4},\qquad E_{n,i}=\mathcal P_\zeta(\tau_{n,i};B_n,\widehat{\mathcal F}_n),\qquad S_{n,i}=\mathcal C_{\rm PDM}(\mathbf s_{n,i})$$

- **World model**: a recurrent predictor on the low-resolution BEV grid. Targets are the *same encoder's* features of the real future frames at +1 to +4 s (stop-gradient), SE(2)-warped into the current ego frame with an overlap mask. Loss: Smooth-L1 plus cosine on absolute features, Smooth-L1 on residuals. Under LTF the target pass uses three future camera views.
- **Evidence probe**: vehicle-footprint anchors at each candidate waypoint are bilinearly sampled from the pooled current map, and at waypoints 2, 4, 6, 8 from the matching future state; fused with trajectory, ego-state and route embeddings.
- **Metric Prism**: one 512-d token per candidate, four layers of set self-attention plus cross-attention to that candidate's evidence, five sigmoid heads (NC, DAC, EP, TTC, Comf). Ranking score:

$$S_{n,i}=\log(s^{\rm NC}+\epsilon)+\log(s^{\rm DAC}+\epsilon)+\log(5s^{\rm EP}+5s^{\rm TTC}+2s^{\rm Comf}+\epsilon)$$

- **Labels**: at every Stage-2 step the current model generates 64 candidates per scene and the **NAVSIM v1 evaluator** scores them. Weighted BCE with factor weights (5, 5, 5, 5, 2), label smoothing and extra weight on low-scoring labels (NC weights 1.7 / 3.1 / 1.0 for labels 0 / 0.5 / 1).

### ECFR: Frenet refinement toward the best same-maneuver candidate (§3.5, Appendix 3)

$$\Delta^{\rm F}_{n,i}=R_\eta(\tau_{n,i},E_{n,i}),\qquad \tau^\dagger_{n,i}=\arg\max_{\tau_{n,j}:\,m_{n,j}=m_{n,i}}P_{\rm eval}(\tau_{n,j}),\qquad P_{\rm eval}=y^{\rm NC}y^{\rm DAC}\tfrac{5y^{\rm EP}+5y^{\rm TTC}+2y^{\rm Comf}}{12}$$

The target is the longitudinal, lateral and heading offset, in the base trajectory's Frenet frame, from each base to the evaluator-best candidate of its own maneuver family in the same 64-candidate pool (Smooth-L1, weight 5). A best base has a zero target. At inference 16 of 64 are refined and all 64 re-scored, unlike [[sources/momworld.md]]'s refiner, whose output is not re-scored.

### Training (§3.6, Table S2)

| | Stage 1 | Stage 2 |
|---|---|---|
| Trainable | Perception, auxiliary heads (map, agents, maneuver), M³F | All inference-time modules |
| Supervision | $\mathcal L_{\rm aux}+\mathcal L_{\rm M^3F}$ | + future BEV, factor labels and Frenet targets from the current model's 64 candidates |
| Loss weights (aux / gen / verify / refine) | 1 / 4 / – / – | 1 / 4 / 2 / 2 |
| LR | $4\times10^{-4}$ | $2\times10^{-4}$, ×0.2 for perception and M³F |
| Epochs / updates | 60 / 159,600 | 10 / 26,600 |
| Batch / precision / hardware | 32 / FP32 / 4 × RTX 4090 | same |

Modules: M³F 6 DiT-MoE blocks with 8 FFN experts (width 256); Metric Prism 4 layers at 512; ECFR a 2-layer Frenet transformer at 512. Perception: ResNet-34 TransFuser, front-left/front/right cameras cropped and stitched to 256×1024, LiDAR raster 64×64 m at 0.25 m. Eight waypoints over 4 s.

---

## Results

### Table 1: NAVSIM v1 navtest (ResNet-34)

Residual = closed form − reported, computed here; negative values are normal for per-scene-averaged PDMS ([[concepts/navsim-benchmark.md#physwam]]).

| Method | NC | DAC | TTC | Comf. | EP | PDMS | Residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| UniAD | 97.8 | 91.9 | 92.9 | 100.0 | 78.8 | 83.4 | −4.12 |
| TransFuser (TF) | 97.7 | 92.8 | 92.8 | 100.0 | 79.2 | 84.0 | −3.91 |
| World4Drive | 97.4 | 94.3 | 92.8 | 100.0 | 79.9 | 85.1 | −3.70 |
| GoalFlow (TF) | 98.3 | 93.8 | 94.3 | 100.0 | 79.8 | 85.7 | −3.45 |
| DiffusionDrive (TF) | 98.2 | 96.2 | 94.7 | 100.0 | 82.2 | 88.1 | −2.72 |
| WoTE (TF) | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 | −2.18 |
| MeanFuser (LTF) | 98.6 | 97.0 | 95.0 | 100.0 | 82.8 | 89.0 | −2.20 |
| VADv2 ⚠ (ICLR 2026) | 98.3 | 97.4 | 95.7 | 100.0 | 82.3 | 89.3 | −2.33 |
| DriveSuprim (R34) | 97.8 | 97.3 | 93.6 | 100.0 | 86.7 | 89.9 | −2.55 |
| **ManeuverFlow (LTF)** | 98.49 | 97.40 | 95.13 | 99.99 | 84.87 | 89.98 | −2.05 |
| **ManeuverFlow (TF)** | 98.42 | 98.01 | 95.19 | 99.99 | 85.36 | **90.58** | −1.94 |

Every row passes the closed-form check. ⚠ **VADv2 at 89.3 matches no VADv2 row in the wiki**, where VADv2 appears at 80.9 (DAC 89.1) in [[sources/idol.md]]'s and other tables. It is cited as the ICLR 2026 version, so the published model was probably re-evaluated or retrained. IDOL's v2 table carries an unidentified "VADv2 85.8" row with DAC 98.3 that the wiki could not place; that row and this one have compatible sub-scores (NC 98.0 vs 98.3, DAC 98.3 vs 97.4) and may be the v1 and v2 halves of the same new VADv2. DriveSuprim's 89.9 here carries its ResNet-34 sub-scores, unlike the mixed row in [[sources/redrive.md]]'s table.

### Table 2: NAVSIM v2 navtest

| Method | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS | Residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ego Status MLP | 93.1 | 77.9 | 92.7 | 99.6 | 86.0 | 91.5 | 89.4 | 98.3 | 85.4 | 64.0 | −4.00 |
| TransFuser (TF) | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 | +1.28 |
| DriveSuprim | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 | +2.28 |
| DiffusionDrive (TF) | 98.2 | 96.3 | 99.4 | 99.8 | 87.4 | 97.4 | 97.0 | 98.3 | 87.7 | 88.3 | −0.94 |
| MeanFuser (LTF) | 98.3 | 97.2 | 99.6 | 99.8 | 87.6 | 97.4 | 97.3 | 98.3 | 88.2 | 89.5 | −0.90 |
| **ManeuverFlow (LTF)** | 98.61 | 98.13 | 99.25 | 99.80 | 87.27 | 97.96 | 96.13 | 98.26 | 85.17 | 89.70 | −0.72 |
| **ManeuverFlow (TF)** | 98.38 | 97.94 | 99.41 | 99.76 | 88.75 | 97.75 | 96.86 | 98.32 | 86.54 | **90.14** | −0.80 |

The caption says "a single official benchmark version", but by residual sign the column mixes evaluators: TransFuser 76.7 and DriveSuprim 83.1 are the shared block's pre-fix rows, while DiffusionDrive 88.3, MeanFuser 89.5 and both ManeuverFlow rows are corrected-like. The same-scale margin is +0.64 over MeanFuser. **DiffusionDrive at 88.3 is a third corrected-like DiffusionDrive row** (after World4Scorer's and GeoWAM's 88.2), on sub-scores within 0.4 of the 84.5 row's, which supports reading 84.5 as the pre-fix value ([[concepts/navsim-benchmark.md#world4scorer]]). These are v1 checkpoints evaluated without v2 training; EC 86.5 is high for a scored pipeline (World4Scorer framewise 77.0).

### Table 3: WorldEngine closed loop (all 288 released scenarios)

| Method | Traffic | SR ↑ | EP ↑ | PDMS\* ↑ |
|---|---|---:|---:|---:|
| VADv2 | Log | 77.08 | 48.00 | 63.81 |
| VADv2 | IDM | 73.66 | 46.71 | 60.98 |
| DiffusionDrive | Log | 86.24 | 45.09 | 69.83 |
| DiffusionDrive | IDM | 85.85 | 43.88 | 68.22 |
| MeanFuser | Log | 87.25 | 55.90 | 73.81 |
| MeanFuser | IDM | 87.15 | 55.60 | 73.28 |
| **ManeuverFlow (LTF)** | Log | **87.48 ± 0.92** | **60.68 ± 0.91** | **75.19 ± 0.85** |
| **ManeuverFlow (LTF)** | IDM | **87.92 ± 0.87** | **60.89 ± 0.81** | **74.58 ± 0.64** |

WorldEngine (OpenDriveLab's 3DGS simulator over NAVSIM logs; not ingested) renders cameras only, so only LTF runs. The 288 scenarios are "high-risk" scenes reconstructed from **navtest** logs; background traffic is either replayed (Log) or IDM-controlled, and the ego replans from rendered observations. SR is the fraction of episodes without collision or drivable-area violation; PDMS\* is a closed-loop composite that is not NAVSIM's PDMS. Where the baseline rows come from (the benchmark paper or the authors' runs) is not stated. See [[concepts/worldengine-benchmark.md]].

- **Reactive traffic changes little.** IDM against Log: ManeuverFlow −0.61, MeanFuser −0.53, DiffusionDrive −1.61, VADv2 −2.83 PDMS\*, and ManeuverFlow's SR rises.
- **ManeuverFlow's gain over MeanFuser is progress** (EP +4.8 / +5.3), with SR level (+0.2 / +0.8).
- **VADv2 is last here and third of four on NAVSIM v1** among the same methods (89.3, above MeanFuser's 89.0) if the two rows are the same model, which the paper does not say.

### Table S3: three training runs

| | PDMS (v1) | EPDMS (v2) | PDMS\* Log | PDMS\* IDM |
|---|---:|---:|---:|---:|
| TF | 90.58 ± 0.05 | 90.14 ± 0.04 | – | – |
| LTF | 89.98 ± 0.06 | 89.70 ± 0.06 | 75.19 ± 0.85 | 74.58 ± 0.64 |

Mean ± sample sd over three independently trained checkpoints, each averaged over five evaluations. Sub-score sds are 0.00–0.14 (EC the largest). The paper notes this is dispersion, not a confidence interval, and that the component ablations have no matched-seed uncertainty.

---

## Ablations (NAVSIM v1 navtest, TF)

### Table 4: cumulative components

| Model | Components | Prior | Flow | Evidence | Refiner | NC | EP | PDMS |
|---|---|---|---|---|---|---:|---:|---:|
| Base | Direct planner (TransFuser) | – | – | – | – | 97.73 | 79.24 | 84.01 |
| $\mathcal M_0$ | Candidate-based MeanFlow | Single | Dense MF | Cur. | – | 97.21 | 81.05 | 86.14 |
| $\mathcal M_1$ | + M-GMP and Dense-XL | M-GMP-8 | Dense-XL | Cur. | – | 97.02 | 81.63 | 87.91 |
| $\mathcal M_2$ | + maneuver-routed flow | M-GMP-8 | M³F | Cur. | – | 97.64 | 82.12 | 88.35 |
| $\mathcal M_3$ | + future evidence | M-GMP-8 | M³F | Cur.+Fut. | – | 98.52 | 84.41 | 89.93 |
| $\mathcal M_4^{\rm Cur}$ | w/o future branch | M-GMP-8 | M³F | Cur. | ECFR | 98.27 | 84.50 | 90.03 |
| $\mathcal M_4^{\rm KM}$ | K-means prior control | KMeans-8 | M³F | Cur.+Fut. | ECFR | 98.42 | 84.53 | 90.04 |
| **$\mathcal M_4$** | Complete model | M-GMP-8 | M³F | Cur.+Fut. | ECFR | 98.42 | 85.36 | **90.58** |

Dense-XL is a dense flow field matched to M³F's total parameter count. DAC, TTC and comfort are not reported for the ablation rows, so the closed-form check cannot be run on them.

| Effect | Without the other | With the other |
|---|---:|---:|
| Future evidence | +1.58 ($\mathcal M_2\to\mathcal M_3$) | +0.55 ($\mathcal M_4^{\rm Cur}\to\mathcal M_4$) |
| ECFR refinement | +1.68 ($\mathcal M_2\to\mathcal M_4^{\rm Cur}$) | +0.65 ($\mathcal M_3\to\mathcal M_4$) |
| Both | +2.23 ($\mathcal M_2\to\mathcal M_4$) | |

Before refinement, future evidence raises EP by 2.29 and NC by 0.88; refinement on top of it raises EP by 0.95 and lowers NC by 0.10. The "w/o future branch" row disables the predicted-future branch; whether its future-BEV loss (which trains the shared encoder) is also off is implied but not stated, so the +0.55 / +1.58 combine any training-time shaping with the inference-time evidence.

### Table 5: routing depth and proposal budget

| $k$ | Recall@$k$ (%) | $N=8$ | 16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 87.20 | 89.85 | 90.04 | 90.26 | 90.33 | 90.42 |
| 2 | 96.46 | 90.04 | 90.15 | 90.43 | 90.54 | 90.59 |
| 3 | 98.30 | 90.06 | 90.28 | 90.41 | **90.58** † | 90.56 |
| 4 | 99.21 | 90.10 | 90.27 | 90.42 | 90.57 | 90.67 |
| | M³F activation bound (MB) | 1.11 | 2.07 | 3.99 | 7.82 | 15.50 |
| | Proposal FPS (perception → M³F) | 66.46 | 63.69 | 58.62 | 53.86 | 47.12 |

† default, fixed before evaluating on navtest; the rest are post-hoc sweeps with frozen checkpoints. The whole grid spans 89.85–90.67; two cells exceed the default ($k=4$, $N=128$: 90.67; $k=2$, $N=128$: 90.59).

### Table S4: Metric Prism's ranking within the unrefined 64-candidate pool

| Top-$\kappa$ | Exact pool oracle (%) | Within 1 PDMS point (%) |
|---:|---:|---:|
| 1 | 40.14 | 75.28 |
| 3 | 45.61 | 79.57 |
| 5 | 48.14 | 81.16 |
| 10 | 52.69 | 83.18 |

### Table S5 and Figure 2: frozen-generator oracle

![[appendix_sampling_oracle_draft.png|Figure 2: Post-hoc candidate-quality analysis of the frozen M3F across 12,146 NAVSIM v1 navtest scenes, excluded from training and model selection. Curves report best and Top-κ mean PDMS over nested pools; HumanAgent 94.8 is a dashed reference.]]

| $N$ | Best | Top-3 mean | Top-6 mean | Top-12 mean | Selected (Table 5, $k=3$) | Oracle − selected |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 92.20 | 90.90 | 89.18 | – | 90.06 | 2.14 |
| 16 | 92.92 | 91.98 | 91.00 | 89.19 | 90.28 | 2.64 |
| 32 | 93.57 | 92.81 | 92.06 | 91.05 | 90.41 | 3.16 |
| 64 | 94.25 | 93.54 | 92.92 | 92.14 | 90.58 | 3.67 |
| 128 | 94.61 | 94.03 | 93.55 | 92.93 | 90.56 | 4.05 |
| 256 | 94.85 | 94.47 | 94.08 | 93.58 | – | – |
| 512 | 95.02 | 94.79 | 94.47 | 94.08 | – | – |

Nested prefixes of one 512-candidate draw per scene from the unrefined generator; the last two columns are added here and compare an unrefined oracle with the refined, selected output, so they are approximate. Best-of-256 (94.85) passes the human agent's 94.8; the appendix cautions that this is an aggregate order statistic, not per-scene superiority. The oracle keeps rising with $N$ while the selected plan does not.

---

## Visualizations

![[qualitative_2x3.png|Figure 5: Six urban scenes with the top-three maneuver components shown in paired BEV and front-view projections.]]

**Figure 5**: Six scenes, three routed components each (pink, blue, cyan). In every scene one component (pink) appears as a single straight line, so its samples barely differ; the other two are bundles curving to either side, some across occupied adjacent lanes.

![[m3f_expert_specialization_two_scene.png|Figure S3: Controlled expert sweep and final-plan provenance. (a,b) Paired BEV/front views of two NAVSIM scenes. Thin E0–E7 trajectories are controlled responses sharing a fixed M-GMP source state, scene context, and transport interval; the source is omitted, dashed green is the logged reference, and thick red is the final trajectory. The final plan comes from the full N=64 base/refined pool after ECFR, evidence re-query, and Metric Prism re-scoring rather than from the displayed sweep, with provenance E6 → ECFR in (a) and E7 → ECFR in (b).]]

**Figure S3**: One source state pushed through each of the eight experts. (a) A lane-change scene: the experts range from sharp left paths (E4, E5) to straight and slightly right ones, and the final plan comes from E6 (the right-lane-change expert) after refinement, close to the logged reference. (b) A right turn at an intersection: the final plan comes from E7 (right turn) after refinement and nearly overlaps the logged reference. The experts produce visibly different paths from the same source, so the routing is not inert.

![[four_model_qualitative_comparison_appendix_450dpi_q95.jpg|Figure S4: Descriptive, non-budget-matched gallery on six shared scenes with paired BEV/front views. (a) ManeuverFlow: a qualitative N=32 M3F draw, routed 16/8/8 and colored by component. (b) WoTE: the top 64 of 256 predictions by final reward. (c) DiffusionDrive: all 20 candidates ordered by logits. (d) MeanFuser: eight native proposal heads plus the fused output.]]

**Figure S4**: The same six scenes for four planners, each with its own display convention (ManeuverFlow 32 draws, WoTE top 64 of 256, DiffusionDrive 20, MeanFuser 8 plus the fused output). ManeuverFlow's candidates form three separated bundles; WoTE's and DiffusionDrive's fan out thinly around straight; MeanFuser's eight proposals cluster close to its fused output. The paper says the gallery is descriptive and not budget-matched.

---

## Table Audit {#table-audit}

1. **v1 is consistent.** All eleven rows pass the closed-form check (−1.94 to −4.12). MeanFuser, WoTE, DiffusionDrive, DriveSuprim, World4Drive (85.1) and the ResNet-34 GoalFlow (85.7) match the wiki's other copies.
2. **VADv2 at 89.3** is unlike every other VADv2 row in the wiki (80.9) and probably belongs with IDOL's unexplained v2 "VADv2 85.8" (above).
3. **v2 mixes evaluators** despite the "single official benchmark version" caption: two pre-fix-like rows (TransFuser +1.28, DriveSuprim +2.28), four corrected-like. The corrected-cohort entries above 90.14 are absent (WA-JEPA 91.7, MM-Future 91.5, World4Scorer 91.4, SUV 91.0, ReDrive 90.8 and others).
4. **v1 scope.** "Highest listed" is accurate, but the ResNet-34 rows above 90.58 in the wiki (DiffusionDriveV2 91.2, PaIR-Drive 91.2, iPad 91.1) are not listed. The first two use RL.
5. **WorldEngine baselines** have no stated source and no spread.
6. **Reference slips** in the appendix (above). Figure 1(a) is a schematic, as the appendix says.

---

## Limitations

The paper's own (Appendix 6.3): the maneuver families are trajectory-space bins, not map-semantic maneuvers; top-3 routing can miss rare or compositional behaviour; the learned PDM factors and figures do not establish safety; the shared future cannot model interactive responses to alternative ego plans; NAVSIM v2 and WorldEngine test transfer of a fixed selector and closed-loop replanning on navtest-derived scenes, not independent domains or real vehicles.

The wiki adds:

1. **Evaluator labels at every training step.** The scorer's targets and the refiner's targets both come from running the NAVSIM v1 evaluator on 64 freshly generated candidates per scene, with ground-truth simulation. It is a scorer-cohort method in the sense of [[concepts/selection-based-planning.md#top-of-leaderboard]].
2. **No seeds on the ablations**, so effects of 0.4–0.6 (routed experts, K-means control, future after refinement) rest on single comparisons, although the main model's own seed sd (0.05) is small.
3. **The future-evidence ablation removes training and inference together.** A model trained with the future-BEV loss but scored without future evidence would separate them.
4. **Latency is partial.** Proposal FPS (53.86 at $N=64$) covers perception through M³F; the world model, evidence probe, two scoring passes and the refiner are excluded, and the hardware for the timing is not stated.
5. **No navhard, HUGSIM or Bench2Drive.** WorldEngine is built from navtest logs, the evaluation split of the training benchmark.
6. **The selector leaves more on the table as the pool grows** (oracle − selected 2.1 → 4.1 from $N=8$ to 128), and Metric Prism is not compared with another scorer on the same pool.

---

## Relation to Other Wiki Pages

- [[sources/idol.md]]: the closest design on the same backbone class (ResNet-34 TransFuser). IDOL rolls out a latent BEV future *per* anchor (256) and refines every anchor from it; ManeuverFlow predicts *one* future and reads it along each of 64 paths, then refines 16. Feedback is worth +1.3 to +2.7 in IDOL and future evidence +1.58 / +0.55 here. ManeuverFlow 90.58 / 90.14 against IDOL 90.0 / 89.6; ManeuverFlow cites WoTE, IDOL's base, but not IDOL.
- [[sources/da-wam.md]], [[sources/mm-future.md]]: per-candidate futures read by a scorer (+0.15, +0.4) and DA-WAM's shared future (−0.50). ManeuverFlow's shared future with a per-candidate readout is a fourth configuration; see [[concepts/world-model-for-ad.md#shared-future-path-readout]].
- [[sources/momworld.md]]: a shared future memory inside a scorer, and a residual flow on the chosen plan that is not re-scored. ManeuverFlow's refined candidates are re-scored against the retained ones.
- [[sources/dial.md]]: eight rule-derived intents from trajectory geometry, used for classifier-free guidance and intent-balanced GRPO on WOD-E2E. ManeuverFlow uses an eight-family rule ontology as separate Gaussian sources and routed experts, and reports the recall of its family predictor, which DIAL did not.
- [[sources/mm-future.md]]: uses MeanFuser's K-means Gaussian-mixture noise prior; ManeuverFlow's K-means-8 control is that kind of prior and scores 0.54 below the rule-aligned one.
- [[sources/world4scorer.md]], [[concepts/best-of-n.md]]: fixed-pool oracles. On DrivoR's 64 candidates the oracle is 99.33 against 94.19 for the best scorer; on ManeuverFlow's own 64 unrefined candidates it is 94.25 against 90.58 selected.
- [[sources/resworld.md]]: cited for refining a plan with predicted future information; ManeuverFlow's future field likewise carries consecutive-state residuals.
- [[sources/geoworldad.md]], [[sources/lwdrive.md]], [[sources/hydra-mdp-pp.md]]: other refine-and-score pipelines with BCE heads on simulator scores.
- [[sources/simvta.md]]: World Engine as a training renderer; here it is the closed-loop benchmark. See [[concepts/worldengine-benchmark.md]].
- Not ingested: **MeanFuser** (one-step MeanFlow with a K-means GMM prior that fuses its proposals; ManeuverFlow's main foil), **GoalFlow**, **WoTE**, **World Engine**, **Improved MeanFlow (iMF)**, **MM-FM** (flow matching for multimodal distributions), the **ICLR 2026 VADv2**, **Gen-Drive**, **DTPP**.

---

## Open Questions

- **Is the shared-future effect a readout effect?** Give DA-WAM-style shared and per-candidate futures the same path-sampling probe, or give ManeuverFlow's field an averaged readout; if the averaged readout reproduces DA-WAM's −0.5, the harm of shared futures is in the readout.
- **Training-time or inference-time future?** Train with the future-BEV loss and score without the future evidence.
- **Why does more candidates not help?** The oracle rises 2 points from 8 to 64 candidates while the selected plan rises 0.5. A scorer swap on the same pool (as World4Scorer did on DrivoR's) would say whether Metric Prism or the evaluator-label objective is the limit.
- **Does the maneuver prior help where maneuvers matter?** Per-family scores, or a navhard run, would show whether the +0.54 over K-means concentrates in turning and lane-change scenes.
