---
title: "CLOVER: Closed-Loop Value Estimation & Ranking for End-to-End Autonomous Driving Planning"
type: source-summary
sources: ["raw/papers/CLOVER_ Closed-Loop Value Estimation & Ranking for End-to-End Autonomous Driving Planning.md"]
related: [sources/sparsedrivev2.md, sources/drivor.md, sources/world4scorer.md, sources/adaptive-wam.md, sources/auto-jepa.md, sources/curious-vla.md, sources/maneuverflow.md, sources/drive-jepa.md, sources/wa-jepa.md, sources/da-wam.md, sources/coworld-vla.md, sources/diffusiondrive-v2.md, sources/drivesuprim.md, sources/hydra-mdp-pp.md, sources/simscale.md, sources/drivefuture.md, sources/dvgt-2.md, sources/ad-e2e-jepa.md, concepts/selection-based-planning.md, concepts/navsim-benchmark.md, concepts/navhard-ood-evaluation.md, concepts/best-of-n.md, concepts/evaluation-variance.md, concepts/inference-latency.md, concepts/teacher-pseudo-labels.md, concepts/nuscenes-waymo-evals.md, research-directions.md]
created: 2026-10-14
updated: 2026-10-14
confidence: medium
---

# CLOVER: Closed-Loop Value Estimation & Ranking for End-to-End Autonomous Driving Planning

**Authors**: Sining Ang, Yuguang Yang, Canyu Chen, Yan Wang
**Orgs**: Institute for AI Industry Research (AIR), Tsinghua University; University of Science and Technology of China; Beihang University
**arXiv**: 2605.15120v2 (v1 May 2026)
**Code**: github.com/WilliamXuanYu/CLOVER ("code and generated data will be released"). [[sources/auto-jepa.md]] initializes its scorer from "the publicly released CLOVER trajectory scorer", so at least the scorer checkpoint was released.
**Source**: `raw/papers/CLOVER_ Closed-Loop Value Estimation & Ranking for End-to-End Autonomous Driving Planning.md`. The clipping has all five figures and all 26 tables; Table 3 (navhard) is flattened to one line of numbers and is rebuilt below.

**The CLOVER group.** Three of the four authors (Ang, Yang, Wang) are the whole author list of [[sources/adaptive-wam.md]]; Ang and Wang are on [[sources/world4scorer.md]]; Chen, Yang and Wang are on [[sources/curious-vla.md]], whose proposal-diversity protocol this paper uses. All four are AIR Tsinghua papers. CLOVER (May 2026) comes before Adaptive-WAM (August) and World4Scorer (September), which both reuse its artifacts; before this ingest it was cited by three wiki papers as a dependency: Auto-JEPA's scorer initialization, Adaptive-WAM's pseudo-expert targets and World4Scorer's candidate bank.

**Confidence: medium.** The appendix is unusually complete: both NAVSIM-v2 evaluator columns, three training seeds, a full anchor-weight sweep, a fixed-proposal scorer diagnostic and stage-wise proposal statistics. Every printed row passes the wiki's arithmetic checks. Against that: the training split is never stated, the anchor weights were swept on the test split, all ablations are reported on navtest, the v2 "state of the art" leaves out the paper's own base model (which a later paper by the same group measures above it), the navhard result does not move from that base, and the nuScenes training is not described at all.

---

## What It Is

**[[sources/drivor.md]]'s generate-and-score planner, unchanged at inference, trained in two new stages against the NAVSIM evaluator.**

- **Architecture** (DrivoR's): DINOv2 ViT-S with rank-32 LoRA over four cameras (front, left, right, back), 16 scene tokens per camera; a transformer generator with learnable trajectory queries produces $K=64$ trajectories of 8 poses over 4 s; a separate scorer embeds each trajectory's waypoints, attends to the same scene tokens, and predicts PDMS sub-scores. Selection is the argmax of the sub-scores composed by the PDMS formula (v1 weights NC/DAC/DDC/TTC/EP/Comfort = 1/1/0/5/5/2). No LiDAR.
- **Stage 1, evaluator-filtered coverage.** For every training scene a rule-based generator builds 261 candidates from six families (lateral shifts, off-road boundaries, acceleration profiles, stop-go, approach-brake, overshoot) using the **training metric cache: route centerline, drivable-area map and future obstacle occupancy**. Up to 180 are scored by the NAVSIM PDM evaluator; 50 are kept by a coverage rule over sub-score patterns and score bins, plus boundary interpolations where a small lateral change drops PDMS by more than 0.25. At training time up to $M=8$ with PDMS ≥ 0.8 are sampled by farthest-point sampling. The generator gets a min-over-proposals L1 to the logged trajectory plus the same loss averaged over the pseudo-experts; the scorer is pretrained on evaluator sub-scores.
- **Stage 2, conservative self-distillation.** 30 cycles alternate one epoch of **scorer fitting** (the evaluator scores the current 64 proposals; the scorer regresses the true sub-scores) and one epoch of **generator refinement**: a frozen teacher generates 64 proposals, the learned scorer picks a top-$k$ set and a vector-Pareto set (non-dominated in predicted sub-scores), and the student covers both by set-level L1, with a stability L1 to the teacher's proposals and the logged-trajectory loss. Backbone frozen.
- **Optional anchor reranking for NAVSIM-v2.** A second, single-mode planner (same architecture, diversity settings off, plain imitation) supplies a reference trajectory; candidates are reranked by $2.0\,s_k - 0.2\,\mathrm{XYRMS}_k/5 - 0.5\,\mathrm{HeadRMS}_k/0.35$.
- **Not RL**, the paper insists: no environment interaction, all feedback computed offline from logged scenes by the rule-based evaluator.

| Benchmark | CLOVER | Notes |
|---|---:|---|
| NAVSIM-v1 PDMS (navtest) | **94.5** | Seeds 94.488 / 94.502 / 94.492; DrivoR (trainval) 93.7 in the same table; human 94.8 |
| NAVSIM-v2 EPDMS (navtest), corrected evaluator | **90.4** | 89.3 without the anchor; EC 75.5 (35.0 without the anchor) |
| NAVSIM-v2 EPDMS*, original evaluator | 87.2 | Same model; the fix is worth +3.2 |
| navhard two-stage EPDMS | 48.3 | Equal to DrivoR's 48.3 in the same table |
| nuScenes open-loop | 0.31 m / 0.10% (ST-P3 protocol); 0.65 m / 0.30% (UniAD protocol) | Training on nuScenes not described |
| Latency | about 110 ms per scene, A100, $K=64$ | "In the NAVSIM-v2 setting"; DrivoR's own figure is also 110 ms |
| Training | 4 × A100 + a 64-core CPU for evaluator calls; 30–40 h Stage 1, 60–80 h Stage 2 | 4–5 days in total |

---

## Key Takeaways

- **The gain is in ranking, not in coverage.** The paper's own identity splits a deployed gain into oracle gain plus regret reduction. From its Table 23, Stage 2 against DrivoR's pools: **oracle +0.05 PDMS (99.33 → 99.39), regret −0.69 (5.64 → 4.95)**, so about 93% of the +0.74 is better ranking. Stage 1 lifts the oracle to 99.76 and Stage 2 gives most of that back. The "DrivoR baseline" statistics (oracle 99.33, selected 93.69, gap 5.64) are digit-identical to [[sources/world4scorer.md]]'s measurement of DrivoR's released model, so this is a regret ladder on one benchmark: DrivoR's scorer 5.64, World4Scorer's scorer on DrivoR's pools 5.14, CLOVER on its own pools about 4.95.
- **94.5 is the highest single-pass v1 number on the wiki's ladder without synthetic data**, 0.1 below DrivoR with 134k SimScale scenes (94.6) and 0.3 below the human (94.8). **The training split is never stated.** Its single-expert baseline is DrivoR's released model (93.69), which the wiki reads as the trainval one, and its Table 1 prints DrivoR's trainval row; the comparison it makes is therefore most plausibly trainval against trainval. World4Scorer's later claim to the top navtrain-trained 94.0 stands or falls with this split.
- **On NAVSIM-v2 the training gain disappears.** CLOVER prints both evaluator columns (87.2* / 90.4), and every labelled row's residual sign agrees with its column. But its table omits its own base: World4Scorer, from the same group four months later, measures DrivoR's released model at **90.66 corrected EPDMS**, above CLOVER's 90.4, with nearly the same sub-scores (CLOVER is at most 0.2 higher on NC, DAC, TTC and HC and 0.2–0.6 lower on DDC, EP and LK; EC is 75.5 in both). Without the anchor CLOVER scores 89.3.
- **Diversity breaks extended comfort, and a second model buys it back.** Without the anchor EC is **35.0**; with it, 75.5, at a cost of 5.5 points of EP. That is the level DrivoR's narrower pool reaches with no anchor at all (75.5 in World4Scorer's measurement). The fix is stateless (no previous plan is needed, unlike [[sources/world4scorer.md]]'s inertial re-ranking or [[sources/drive-jepa.md]]'s momentum selector), but it runs a second network, and its three weights were swept on the split the result is reported on.
- **navhard does not move.** 48.3 against DrivoR's 48.3, on the same corrected scale (product check −1.8, inside DrivoR's corrected band). Stage-1 safety rises (DAC 95.1 → 98.0, NC 98.8 → 99.4) and EC falls (73.3 → 56.0; Stage 2 76.2 → 57.3). +0.8 on navtest, 0.0 on navhard: the opposite of the navtest-small, navhard-large pattern the wiki has recorded for several inference-path mechanisms.
- **Closing the loop is what makes Stage 2 work, at least in the paper's ablations.** A non-iterative joint update collapses below 10 PDMS; self-distillation without scorer refitting saturates at 93.8–94.0; the full alternating loop reaches 94.5. Scalar top-$k$ targets by true PDMS give 93.9; vector-Pareto targets from the learned scorer give 94.5 and more qualified clusters (8.71 against 6.45). Single runs on navtest.
- **The scorer's backbone hardly matters on a fixed pool; its training does.** On CLOVER-generated 64-candidate pools, scorers trained in isolation select 92.4 (DINOv2-S) to 92.8 (Wan2.2-5B video features) on navtest, all below the deployed system's 94.5. Wan2.2-5B's train score is 97.7 against 95.0 for DINOv2-S, so its +0.4 comes with the largest train–test gap.
- **Three training seeds within 0.014 PDMS** (94.488–94.502) and 0.014 EPDMS: the tightest training-seed spread in the wiki. Which stages were re-run is not stated; the Stage-2 backbone is frozen.
- **The theorem is a one-line total-variation bound**: if the scorer-selected targets hold more truly-high-scoring trajectories than the current pool and the update stays close to a mixture of the two, the high-score mass grows. The enrichment half is checked, in aggregate, on the final model's navtest proposals (69.74% full-score among predicted ≥ 0.95, against 35.42% pooled); the conservativeness half is never measured.

---

## Method

![[image1.png|Figure 1: Left, conventional proposal-selection planners: a generator trained by single-expert imitation and a scorer trained to rank fixed proposals, decoupled, so ranking feedback never reshapes the proposals. Right, CLOVER: a trajectory-level scorer supervised by true PDMS sub-scores (safety, progress, comfort), and a generator refined toward the scorer's choices under stability guidance from a frozen teacher; generator and scorer improve together.]]

**Figure 1**: The left panel marks "no feedback to proposal generation" and a "candidate quality bottleneck"; the right panel's loop feeds true PDMS sub-scores to the scorer and scorer-selected targets plus frozen-teacher stability to the generator.

![[clover_overview2.png|Figure 2: Overview. Top, inference: four surround images through a DINOv2-Small backbone into scene features; ego status and trajectory queries into the trajectory generator; K diverse candidates scored per trajectory and the argmax selected; an optional anchor branch (a non-diverse single-mode generator) supplies a reference to reduce diversity-induced jitter. Bottom left, Stage 1: diversity supervision from evaluator-scored pseudo-expert trajectories and score supervision for the student scorer. Bottom right, Stage 2: the student generator's K candidates are scored by the true evaluator to fit the scorer; the scorer supplies Pareto and top-k self-distillation targets, and a frozen teacher generator stabilizes the student. "Refinement, not RL: no environment exploration, no unknown-state search, no step-wise credit assignment."]]

**Figure 2**: The three boxes are inference, Stage 1 ("diversity pretraining") and Stage 2 ("iterative self-distillation refinement"). The evaluator appears only in training.

### Selection (§3.1)

Each trajectory is $\tau=\{(x_t,y_t,\theta_t)\}_{t=1}^{8}$ at 0.5 s. The generator outputs $\mathcal T_\theta(o)=\{\tau_i\}_{i=1}^{K}$, the scorer $\hat{\mathbf r}_i=S_\phi(o,\tau_i)$ (sub-scores for collision, drivable area, TTC, progress, comfort), and

$$\tau^\ast=\arg\max_{\tau_i\in\mathcal T_\theta(o)}\Gamma_\mathrm{PDMS}\big(S_\phi(o,\tau_i)\big).$$

During training the rule-based evaluator $E$ gives $\mathbf r_i^\mathrm{true}=E(o,\tau_i)$; it is non-differentiable and is not used at deployment.

### Stage 1 (§3.2)

$$\mathcal L_\mathrm{stage1}=\lambda_\mathrm{gt}\min_i\lVert\tau_i-\tau^\mathrm{gt}\rVert_1+\lambda_\mathrm{pe}\frac{1}{|\mathcal E(o)|}\sum_{e\in\mathcal E(o)}\min_i\lVert\tau_i-e\rVert_1+\lambda_\mathrm{score}\frac{1}{|\mathcal Q(o)|}\sum_{\tau\in\mathcal Q(o)}\ell\big(S_\phi(o,\tau),E(o,\tau)\big)$$

$\mathcal E(o)$ is the pseudo-expert set (at most 8), $\mathcal Q(o)$ the trajectories the scorer is trained on, $\ell$ a per-component classification or regression loss. Weights: imitation 1.0, pseudo-expert 0.5, final score 1.0 (Table 16).

### Stage 2 (§3.3)

Scorer phase: $\mathcal L_\mathrm{critic}=\frac1K\sum_i\ell(S_\phi(o,\tau_i),\mathbf r_i^\mathrm{true})$ on the current generator's proposals. Generator phase, with a frozen teacher's proposals $\mathcal T^\mathrm{tea}$, the scorer's top-$k$ set $\mathcal K^\mathrm{tea}$ and its vector-Pareto set $\mathcal P^\mathrm{tea}$:

$$\mathcal L_\mathrm{set}(\mathcal T^\mathrm{stu},\mathcal A)=\frac{1}{|\mathcal A|}\sum_{\tau\in\mathcal A}\min_i\lVert\tau_i^\mathrm{stu}-\tau\rVert_1$$

$$\mathcal L_\mathrm{gen}=\lambda_\mathrm{traj}\mathcal L_\mathrm{gt}+\lambda_\mathrm{topk}\mathcal L_\mathrm{set}(\mathcal T^\mathrm{stu},\mathcal K^\mathrm{tea})+\lambda_\mathrm{pareto}\mathcal L_\mathrm{set}(\mathcal T^\mathrm{stu},\mathcal P^\mathrm{tea})+\lambda_\mathrm{stab}\frac1K\sum_i\lVert\tau_i^\mathrm{stu}-\mathrm{sg}(\tau_i^\mathrm{tea})\rVert_1$$

Table 18 gives trajectory 0.1, Pareto 1.0, stability 0.05, a Pareto set of 2–8 and a teacher reward threshold of 0.4. It gives **no top-$k$ weight**, and Table 8 calls the deployed variant "vector-Pareto targets", so whether the top-$k$ term is used is unclear.

### Anchor reranking (§3.4, Appendix C)

$$\mathrm{Penalty}_k=\lambda_{xy}\frac{\mathrm{XYRMS}_k}{5.0}+\lambda_\psi\frac{\mathrm{HeadRMS}_k}{0.35},\qquad Q_k=\lambda_s s_k-\mathrm{Penalty}_k,\qquad k^\ast=\arg\max_k Q_k$$

with RMS position and wrapped-heading deviations from the anchor over the whole trajectory; $\lambda_s=2.0$, $\lambda_{xy}=0.2$, $\lambda_\psi=0.5$. No candidate is discarded.

---

## Theory (§4, Appendix D)

**Theorem 1.** With $p_t(o)$ the share of current proposals in the high-score region $\mathcal H_o=\{\tau: R^\ast(o,\tau)\ge r_\mathrm{high}\}$ and $q_t(o)$ the share in the scorer-selected target set: if $q_t\ge p_t+\xi_t$ and $\mathrm{TV}\big(\mu_{t+1},(1-\alpha_t)\mu_t+\alpha_t\nu_t\big)\le\eta_t$, then

$$p_{t+1}(o)\ge p_t(o)+\alpha_t\xi_t(o)-\eta_t(o),$$

and the same holds for the expected true score with $\beta_t$ in place of $\xi_t$. The proof is two lines: the mixture has $p_t+\alpha_t\xi_t$ high-score mass, and total variation moves any event's probability by at most $\eta_t$.

**Auxiliary results** (Appendix D): approximate monotonicity of the true objective within a trust region ($J^\ast(\pi_{t+1})\ge J^\ast(\pi_t)-2\epsilon_t-\alpha_t$); refitting the scorer every cycle makes the accumulated extrapolation error linear rather than quadratic in the number of cycles ($T\epsilon+T\rho$ against $T\epsilon_0+\frac{T(T+1)}2\rho$); the scorer's Pareto set lies in the true $(2\delta_t+\kappa)$-approximate Pareto set under a uniform sub-score error $\delta_t$; a uniform margin $\gamma>2\epsilon$ is sufficient for enrichment. The deployed gain decomposes exactly as $V_T-V_0=[O_T-O_0]+[G_0-G_T]$ (oracle gain plus regret reduction).

### Table 14: empirical checks of selected-set enrichment (navtest, 12,146 scenes × 64 proposals)

| Validation target | Metric / protocol | Result | Paper's interpretation |
|---|---|---|---|
| Stage-1 support | Pooled full-score proposals | 35.42% | Stage 1 provides non-zero high-score support |
| Worst-case margin | $P(\gamma_o>2\epsilon_{\max,o})$, elite/reject 0.05/0.05 | 10.64% | Worst-case calibration is too strict |
| High-prob. margin | $P(\gamma_o>2\epsilon_{p75,o})$, elite/reject 0.01/0.01 | 92.50% | Quantile-level separation widely satisfied |
| High-prob. margin | $P(\gamma_o>2\epsilon_{p90,o})$, elite/reject 0.01/0.01 | 82.37% | Most scenes satisfy high-probability separation |
| High-prob. margin | $P(\gamma_o>2\epsilon_{p95,o})$, elite/reject 0.01/0.01 | 71.56% | Strict quantile separation still often holds |
| Pairwise separability | high ≥ 0.95, low ≤ 0.50 | 94.72% | Scorer usually ranks true high above low |
| High-score precision | $s\ge0.90$ | mean $R^\ast$ 0.9529, $P(R^\ast\ge0.90)$ 90.16% | Scorer-high proposals are reliable |
| High-score precision | $s\ge0.95$ | mean $R^\ast$ 0.9753, $P(R^\ast\ge0.90)$ 96.69%, $P(R^\ast=1)$ 69.74% | High-confidence targets strongly enriched |
| Top-$k$ quality | scorer Top-1 | mean best $R^\ast$ 0.9448, oracle gap 0.0495 | Single top selection already high quality |
| Top-$k$ quality | scorer Top-8 | mean best $R^\ast$ 0.9656, gap 0.0283 | Top-$k$ improves high-score coverage |
| Top-$k$ quality | scorer Top-16 | mean best $R^\ast$ 0.9730, gap 0.0209 | Larger sets further reduce the gap |

Enrichment gap at the pooled level: $q-p\approx69.74\%-35.42\%=34.32\%$.

**What is and is not checked.** These statistics are computed on the final model's proposals on the **test split**, pooled over scenes and thresholded on predicted score. The theorem is about per-scene target sets inside training, cycle by cycle, against the current pool. The conservativeness condition ($\eta_t$ small relative to $\alpha_t\xi_t$) has no measurement. The empirical support that does bear on it is in the ablations: a non-iterative update collapses (Table 7).

---

## Results

### Table 1: NAVSIM-v1 navtest

Residual = closed form − reported, computed here; negative is normal for a per-scene-averaged PDMS ([[concepts/navsim-benchmark.md#physwam]]).

| Method | Ref. | Venue | NC | DAC | TTC | Comf. | EP | PDMS | Residual |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| PDMS-Closed | Dauner et al. | PMLR'23 | 94.6 | 99.8 | 89.9 | 86.9 | 99.9 | 89.1 | −0.76 |
| Human driver | NAVSIM | NeurIPS'24 | 100 | 100 | 100 | 99.9 | 87.5 | 94.8 | −0.03 |
| Ego-stat. MLP | NAVSIM | NeurIPS'24 | 93.0 | 77.3 | 83.6 | 100 | 62.8 | 65.6 | −9.77 |
| UniVLA | | arXiv'25 | 96.9 | 91.1 | 91.7 | 96.7 | 76.8 | 81.7 | −5.50 |
| DrivingGPT | | ICCV'24 | 98.9 | 90.7 | 94.9 | 95.6 | 79.7 | 82.4 | −2.85 |
| UniAD | | CVPR'23 | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 | −4.12 |
| LTF | | TPAMI'22 | 97.4 | 92.8 | 92.4 | 100 | 79.0 | 83.8 | −4.18 |
| PARA-Drive | | CVPR'24 | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 | −4.01 |
| DriveX-S | | ICCV'25 | 97.5 | 94.0 | 93.0 | 100 | 79.7 | 84.5 | −3.28 |
| World4Drive | | ICCV'25 | 97.4 | 94.3 | 92.8 | 100 | 79.9 | 85.1 | −3.70 |
| DRAMA | | ISRR'24 | 98.0 | 93.1 | 94.8 | 100 | 80.1 | 85.5 | −3.80 |
| VAD-v2 ⚠ | arXiv 2402.13243 | arXiv'24 | 98.1 | 94.8 | 94.3 | 100 | 80.6 | 86.2 | −2.93 |
| PRIX | | arXiv'25 | 98.1 | 96.3 | 94.1 | 100 | 82.3 | 87.8 | −2.62 |
| DiffusionDrive | | CVPR'25 | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 | −2.72 |
| DIVER | | arXiv'25 | 98.5 | 96.5 | 94.9 | 100 | 82.6 | 88.3 | −2.16 |
| AutoVLA | | NeurIPS'25 | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 | −2.92 |
| DriveVLA-W0 | | ICLR'26 | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.2 | −1.22 |
| ReCogDrive | | ICLR'26 | 97.9 | 97.3 | 94.9 | 100 | 87.3 | 90.8 | −2.61 |
| Hydra-MDP++ | | arXiv'25 | 98.6 | 98.6 | 95.1 | 100 | 85.7 | 91.0 | −1.56 |
| iPad | | arXiv'25 | 98.6 | 98.3 | 94.9 | 100 | 88.0 | 91.7 | −1.68 |
| Centaur | | arXiv'25 | 99.5 | 98.9 | 98.0 | 100 | 85.9 | 92.6 | −0.80 |
| DriveSuprim | | arXiv'25 | 98.6 | 98.6 | 95.5 | 100 | 91.3 | 93.5 | −1.63 |
| DrivoR | | CVPR'26 | 99.0 | 98.9 | 96.7 | 100 | 90.0 | 93.7 | −1.21 |
| **CLOVER** | | | **99.1** | **99.0** | **96.9** | 100 | **91.7** | **94.5** | **−1.05** |

**Every row passes, and every baseline row is [[sources/drivor.md]]'s Table 1 camera-only block** (the same rows and the same residuals as on DrivoR's page). DrivoR is its trainval row (93.7), not the navtrain 93.1. DrivoR's RAP-DINO (93.8, 10× data), best-of-N and SimScale rows are left out. ⚠ VAD-v2 86.2 is the value DrivoR also cites to the 2024 arXiv paper; three VADv2 values circulate ([[concepts/navsim-benchmark.md#drivor]]). CLOVER's EP 91.7 is the highest EP in the table apart from PDMS-Closed's 99.9 (DriveSuprim 91.3). "First or second across all sub-scores" fails on TTC, where AutoVLA and Centaur (98.0) are above CLOVER's 96.9.

### Table 2: NAVSIM-v2 navtest

EPDMS* = original evaluator; EPDMS = updated official implementation (the human-behavior filtering fix). Residual = closed form of the printed sub-scores − reported ([[concepts/navsim-benchmark.md#residual-sign]]).

| Method | NC | DAC | DDC | TL | EP | TTC | LK | HC | EC | EPDMS* | EPDMS | Residual vs EPDMS* | Residual vs EPDMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ego-MLP | 93.1 | 77.9 | 92.7 | 99.6 | 86.0 | 91.5 | 89.4 | 98.3 | 85.4 | 64.0 | – | −4.00 | |
| Transfuser | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 | – | +1.28 | |
| ARTEMIS ⚠ | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | 98.3 | 83.1 | – | +2.03 | |
| DiffusionDriveV2 | 97.7 | 96.6 | 99.2 | 99.8 | 88.9 | 97.2 | 96.0 | 97.8 | 91.0 | 85.5 | 87.5 | +2.10 | +0.10 |
| Hydra-MDP++ | 98.4 | 98.0 | 99.4 | 99.8 | 87.5 | 97.7 | 95.3 | 98.3 | 77.4 | 85.1 | – | +2.67 | |
| DriveSuprim | 97.8 | 97.9 | 99.5 | 99.9 | 90.6 | 97.1 | 96.6 | 98.3 | 77.9 | 86.0 | – | +2.28 | |
| SparseDriveV2 | 98.1 | 98.1 | 99.6 | 99.8 | 91.1 | 97.3 | 96.9 | 98.2 | 78.4 | 86.7 | 90.1 | +2.32 | −1.08 |
| **CLOVER** | **99.4** | **99.3** | 99.5 | 99.8 | 86.9 | **98.9** | 95.2 | 98.2 | 75.5 | **87.2** | **90.4** | +2.65 | **−0.55** |

- **The labels and the residual signs agree on every labelled row.** Pre-fix-only rows sit at +1.28 to +2.67; the corrected values of SparseDriveV2 and CLOVER at −1.08 and −0.55; DiffusionDriveV2's corrected 87.5 at +0.10, its standing exception. CLOVER's sub-scores are the corrected run's (they equal Table 13's best configuration: EP 0.8689, EC 0.7550, HC 0.9823).
- **The fix is +3.2 on CLOVER** (87.2 → 90.4), inside the +2.0 to +3.8 band ([[concepts/navsim-benchmark.md#evaluator-drift-this-table-mixes-two-protocols]]). The DiffusionDriveV2 (85.5 / 87.5) and SparseDriveV2 (86.7 / 90.1) pairs are the ones [[sources/wa-jepa.md]] prints; CLOVER (AIR Tsinghua, May 2026) predates WA-JEPA (Afari, August 2026), so the SparseDriveV2 delta (+3.4) is now reported by two independent groups. *(2026-10-14, corrected after ingesting [[sources/sparsedrivev2.md]]: SparseDriveV2's own Table 3 (March 2026) prints both pairs first, and CLOVER's seven baseline rows are that table's rows digit for digit, ARTEMIS EC 98.3 included. The SparseDriveV2 and DiffusionDriveV2 pairs in CLOVER and WA-JEPA are copies of one source, not independent measurements.)*
- **Where the rows come from.** Ego-MLP, Transfuser and ARTEMIS are the shared baseline block's rows; ⚠ ARTEMIS's EC is printed as 98.3, the duplicate of its own HC that several other tables carry. Hydra-MDP++ 85.1 is the ViT/L row [[sources/da-wam.md]] also prints (the block's is 81.4). DriveSuprim 86.0 (EP 90.6, EC 77.9) is the third DriveSuprim v2 value, as in [[sources/wa-jepa.md]] and [[sources/coworld-vla.md]]. *(2026-10-14: one earlier source explains all seven rows: [[sources/sparsedrivev2.md]]'s Table 3, which labels Hydra-MDP++ 85.1 and DriveSuprim 86.0 V2-99. DA-WAM and WA-JEPA postdate CLOVER.)*
- **"New state of the art" on v2 holds against this table, not against its base.** DrivoR has no navtest v2 row in its own paper and none here. [[sources/world4scorer.md]] later measures DrivoR's released model at **90.66** (EC 75.5) with "the latest official devkits". Every corrected-cohort entry the wiki has above 90.4 (WA-JEPA, MM-Future, World4Scorer, SUV, ReDrive, ForeDrive ViT-L) is dated August–September 2026, after CLOVER; GeoWorldAD (July) ties it at 90.4.

### Table 3: navhard two-stage (rebuilt from the flattened clipping)

Stage 1:

| Method | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RAP-DINO | 97.1 | 94.4 | 98.8 | 99.8 | 83.9 | 96.9 | 94.7 | 96.4 | 66.2 |
| GTRS-D | 98.9 | 96.2 | 99.4 | 99.3 | 72.9 | 98.9 | 95.1 | 96.9 | 39.1 |
| GTRS-A | 98.9 | 95.1 | 99.1 | 99.6 | 76.2 | 99.1 | 94.9 | 97.6 | 54.2 |
| DrivoR | 98.8 | 95.1 | 98.9 | 100 | 72.6 | 98.7 | 94.0 | 97.6 | 73.3 |
| **CLOVER** | **99.4** | **98.0** | **99.9** | 100 | 75.7 | 98.9 | 94.4 | 97.1 | 56.0 |

Stage 2 and combined:

| Method | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS | Stage CFs | Product − reported |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| RAP-DINO | 83.2 | 83.9 | 87.4 | 98.0 | 86.9 | 80.4 | 52.3 | 95.2 | 52.4 | 39.6 | 80.1 × 46.2 | −2.58 |
| GTRS-D | 91.2 | 89.4 | 94.4 | 98.8 | 69.5 | 90.0 | 54.3 | 94.0 | 48.7 | 45.0 | 77.5 × 56.6 | −1.09 |
| GTRS-A ⚠ | 88.1 | 88.8 | 89.3 | 98.9 | 98.9 | 85.9 | 53.7 | 96.8 | 56.9 | 45.4 | 79.5 × 57.8 | +0.55 |
| DrivoR | 90.2 | 88.4 | 91.9 | 98.6 | 70.0 | 88.0 | 50.1 | 98.5 | 76.2 | **48.3** | 80.5 × 56.0 | −3.23 |
| **CLOVER** | 89.8 | 88.2 | 93.2 | 99.0 | 74.2 | 87.7 | 52.6 | 97.0 | 57.3 | **48.3** | 83.2 × 55.9 | **−1.81** |

- **The four baseline rows are [[sources/drivor.md]]'s corrected navhard block**, with the same stage closed forms and residuals: RAP-DINO (ViT-H, 10× data), GTRS-Dense and GTRS-Aug at V2-99, DrivoR on navtrain. ⚠ The GTRS-Aug row carries DrivoR's suspect Stage-2 EP of 98.9 (every other row has 64–87). ZTRS (48.1) from the same block is left out.
- **CLOVER is corrected-like** (−1.81, inside DrivoR's calibrated corrected band of −1.1 to −4.5), so the tie with DrivoR is on one scale. By the product of stage closed forms CLOVER is 1.4 ahead (46.5 against 45.1), within the check's ±2-point error.
- **"Matching the strongest reported result"** leaves out DrivoR's SimScale rows (52.3, 54.6) and SimScale's 53.2 from the same table it copies, which use extra data, and [[sources/drivefuture.md]]'s 55.5 (arXiv 2605.09701, a week earlier).
- Whether the navhard model is the v1 or the v2 checkpoint, which inference weights it uses, and whether the anchor is applied are not stated.

### Table 4: nuScenes open-loop

| Method | ST-P3 L2 (m) | ST-P3 Collision (%) | UniAD L2 (m) | UniAD Collision (%) |
|---|---:|---:|---:|---:|
| ST-P3 | 2.11 | 0.71 | – | – |
| VAD | 0.37 | 0.14 | – | – |
| UniAD | 0.69 | 0.12 | 1.03 | 0.31 |
| EMMA | 0.32 | – | – | – |
| OpenEMMA | 2.81 | – | – | – |
| AutoVLA | 0.48 | 0.13 | 0.86 | 0.35 |
| Impromptu VLA | 0.33 | 0.13 | 0.67 | 0.38 |
| OpenDriveVLA | 0.33 | 0.10 | 0.67 | 0.30 |
| **CLOVER** | **0.31** | **0.10** | **0.65** | **0.30** |

- **The nuScenes training is not described.** Stage 1 needs NAVSIM's metric cache and Stage 2 the PDM evaluator; neither exists for nuScenes. The paper does not say which stages were run there, how many cameras were used, or whether the model was retrained.
- **Ego status is an input and is not marked**, as for every DrivoR-style planner. See [[concepts/nuscenes-waymo-evals.md#clover]].
- **"ST-P3" names a different convention here than in [[sources/dvgt-2.md]]'s table.** CLOVER prints UniAD at 0.69 / 0.12 under "ST-P3" and at 1.03 / 0.31 under "UniAD"; DVGT-2 prints UniAD's 1.03 / 0.31 in its unmarked "ST-P3 convention" block. CLOVER's 0.31 m is a temporal-average number.
- The margins are 0.01 m on L2 (EMMA 0.32) and ties on collision (OpenDriveVLA).

---

## Proposal Quality and Diversity (§5.3, Appendix H)

![[diversity_visualization.png|Figure 3: Qualitative proposal diversity on one scene, front view and BEV for each method, candidates coloured by true PDM score. Left two panels, the DrivoR baseline: all 64 candidates concentrated along one path. Right two panels, CLOVER: a wider fan of branches, many of them high-scoring.]]

**Figure 3**: The baseline's 64 candidates overlap into one bundle; CLOVER's spread laterally and in progress, with the high-scoring (yellow-green) ones covering several branches.

### Table 5: stage-wise proposal statistics (64 proposals, 12,146 navtest scenes)

| Metric | Baseline (DrivoR) | Stage 1 | Stage 2 |
|---|---:|---:|---:|
| Selected PDMS ↑ | 0.9369 | 0.9413 | 0.9448 ⚠ |
| Oracle@64 PDMS ↑ | 0.9933 | 0.9976 | 0.9939 |
| Gap (Oracle − Selected) ↓ | 0.0564 | 0.0563 | 0.0495 |
| Mean PDMS over 64 ↑ | 0.7972 | 0.7374 | 0.8277 |
| Std PDMS over 64 | 0.2152 | 0.3013 | 0.2235 |
| Count(PDMS > 0.95) ↑ | 30.94 | 24.78 | 31.97 |
| Count(PDMS < 0.50) ↓ | 9.05 | 12.21 | 6.83 |
| Pairwise ADE ↑ | 1.80 | 5.97 | 5.20 |
| Pairwise FDE ↑ | 4.47 | 10.70 | 8.25 |
| Endpoint Area ↑ | 5.78 | 46.18 | 25.56 |
| Trajectory Effective Rank ↑ | 1.14 | 1.24 | 1.27 |
| Qualified Cluster Count@2m ↑ | 6.02 | 10.91 | 8.71 |
| Top-6 Real Pairwise ADE ↑ | 0.91 | 2.11 | 1.75 |
| Top-6 Real Pairwise FDE ↑ | 2.20 | 3.55 | 2.59 |

⚠ Stage 2's selected PDMS is 0.9448 here and 0.9443 in Tables 22–23. Only 0.9443 matches the printed gap (0.9939 − 0.9443 = 0.0496; 0.9939 − 0.9448 = 0.0491). The analysis checkpoint therefore selects 94.43, slightly below the headline 94.5 and the seed runs (94.488–94.502). The baseline column (oracle 0.9933, selected 0.9369) matches [[sources/world4scorer.md]]'s measurement of DrivoR's released model.

### Table 22: scorer-ranked top-$K$ quality

TopK-Oracle@K is the best true PDMS among the scorer's top $K$; TopK-Mean@K the mean true PDMS within them.

| Metric | $K$ | Baseline | Stage 1 | Stage 2 | Stage 1 − Base | Stage 2 − Stage 1 |
|---|---:|---:|---:|---:|---:|---:|
| TopK-Oracle@K | 1 | 0.9369 | 0.9413 | 0.9443 | +0.0043 | +0.0031 |
| TopK-Oracle@K | 2 | 0.9461 | 0.9514 | 0.9517 | +0.0054 | +0.0002 |
| TopK-Oracle@K | 3 | 0.9512 | 0.9570 | 0.9554 | +0.0058 | −0.0017 |
| TopK-Oracle@K | 6 | 0.9603 | 0.9660 | 0.9625 | +0.0057 | −0.0035 |
| TopK-Mean@K | 1 | 0.9369 | 0.9413 | 0.9443 | +0.0043 | +0.0031 |
| TopK-Mean@K | 2 | 0.9353 | 0.9402 | 0.9435 | +0.0049 | +0.0033 |
| TopK-Mean@K | 3 | 0.9345 | 0.9383 | 0.9430 | +0.0039 | +0.0046 |
| TopK-Mean@K | 6 | 0.9320 | 0.9346 | 0.9416 | +0.0026 | +0.0070 |

**Stage 2 trades headroom for precision.** From $K=3$ up, Stage 1's scorer-ranked top-$K$ contains better trajectories than Stage 2's, while Stage 2's top choices are more uniformly good.

### Table 23: full proposal-set quality

| Metric | Baseline | Stage 1 | Stage 2 | Stage 1 − Base | Stage 2 − Stage 1 | Stage 2 − Base |
|---|---:|---:|---:|---:|---:|---:|
| Selected PDMS ↑ | 0.9369 | 0.9413 | 0.9443 | +0.0043 | +0.0031 | +0.0074 |
| Oracle@64 PDMS ↑ | 0.9933 | 0.9976 | 0.9939 | +0.0042 | −0.0037 | +0.0005 |
| Gap ↓ | 0.0564 | 0.0563 | 0.0495 | −0.0001 | −0.0068 | −0.0069 |
| Mean PDMS over 64 ↑ | 0.7972 | 0.7374 | 0.8277 | −0.0597 | +0.0903 | +0.0306 |
| Std PDMS over 64 | 0.2152 | 0.3013 | 0.2235 | +0.0861 | −0.0778 | +0.0084 |
| Count(PDMS > 0.95) ↑ | 30.94 | 24.78 | 31.97 | −6.16 | +7.19 | +1.03 |
| Count(PDMS > 0.90) ↑ | 40.24 | 32.13 | 39.90 | −8.10 | +7.77 | −0.33 |
| Count(PDMS < 0.50) ↓ | 9.05 | 12.21 | 6.83 | +3.16 | −5.39 | −2.22 |

**The decomposition the paper's identity gives**: Stage 2 − Base = +0.0074 selected = **+0.0005 oracle + 0.0069 regret reduction**. Stage 1's own +0.0043 is the reverse: +0.0042 oracle, 0.0001 regret.

### Table 24: full proposal diversity

| Metric | Baseline | Stage 1 | Stage 2 | Stage 1 − Base | Stage 2 − Stage 1 | Stage 2 − Base |
|---|---:|---:|---:|---:|---:|---:|
| Pairwise ADE | 1.8004 | 5.9724 | 5.1956 | +4.1720 | −0.7768 | +3.3952 |
| Pairwise FDE | 4.4660 | 10.6973 | 8.2536 | +6.2313 | −2.4438 | +3.7876 |
| Endpoint Std Radius | 3.9033 | 10.2638 | 7.9477 | +6.3604 | −2.3160 | +4.0444 |
| Endpoint Area | 5.7837 | 46.1758 | 25.5615 | +40.3921 | −20.6143 | +19.7777 |
| Trajectory Effective Rank | 1.1401 | 1.2362 | 1.2748 | +0.0962 | +0.0385 | +0.1347 |

### Table 25: quality-aware diversity (proposals with PDMS ≥ 0.8, or the six best by true PDMS)

| Metric | Baseline | Stage 1 | Stage 2 | Stage 1 − Base | Stage 2 − Stage 1 | Stage 2 − Base |
|---|---:|---:|---:|---:|---:|---:|
| Qualified Count (PDMS ≥ 0.8) | 50.16 | 43.97 | 50.44 | −6.19 | +6.47 | +0.27 |
| Qualified Cluster Count@2m | 6.02 | 10.91 | 8.71 | +4.90 | −2.20 | +2.70 |
| Qualified Pairwise ADE | 1.5576 | 3.7988 | 3.6727 | +2.2412 | −0.1261 | +2.1151 |
| Qualified Pairwise FDE | 3.8024 | 6.8121 | 5.6674 | +3.0097 | −1.1447 | +1.8650 |
| Top-6 Real Pairwise ADE | 0.9110 | 2.1111 | 1.7507 | +1.2001 | −0.3603 | +0.8398 |
| Top-6 Real Pairwise FDE | 2.1955 | 3.5542 | 2.5910 | +1.3587 | −0.9633 | +0.3954 |

### Table 26: endpoint clusters by radius

| Radius | Baseline | Stage 1 | Stage 2 |
|---|---:|---:|---:|
| 1 m | 16.77 | 32.77 | 23.20 |
| 2 m | 8.07 | 19.81 | 13.81 |
| 3 m | 5.29 | 14.65 | 9.88 |
| 4 m | 3.97 | 11.24 | 8.07 |

![[diversity_visualization_appendix.png|Figure 5: Six more scenes. In each row the left two panels are the DrivoR baseline (front view and BEV, all 64 candidates) and the right two are CLOVER after Stage 2. Green triangles are the human trajectory, red circles the scorer-selected one, orange squares the oracle with the highest true PDMS, others coloured by true PDMS (viridis). The baseline's candidates form one narrow bundle in every scene; CLOVER's spread across the lane and in progress, many of them high-scoring.]]

**Figure 5**: In all six rows the baseline's selected, oracle and human trajectories overlap inside one bundle; CLOVER's selected and oracle trajectories stay close to each other inside a wider fan.

---

## Ablations (navtest; single runs)

### Table 6: components (NAVSIM-v1)

| Setting | Diversity sup. | Stage-2 refinement | PDMS |
|---|:-:|:-:|---:|
| Single-expert supervision | – | – | 93.7 |
| Diversity only | ✓ | – | 94.1 |
| Closed-loop only | – | ✓ | 93.8 |
| CLOVER full | ✓ | ✓ | 94.5 |

Diversity +0.4, refinement alone +0.1, both +0.8. The single-expert row is DrivoR's 93.69.

### Table 7: Stage-2 design

| Stage-2 variant | PDMS |
|---|---:|
| Non-iterative generator/scorer update | < 10 |
| Self-distillation only, 64 trajectories | 94.0 |
| Self-distillation only, 32 trajectories | 93.8 |
| Self-distillation only, 16 trajectories | 93.8 |
| Self-distillation only, 8 trajectories | 93.8 |
| Full alternating closed-loop refinement | 94.5 |

"Self-distillation only" is described as "without scorer fitting"; which scorer then selects its targets (the frozen Stage-1 one, presumably) is not said.

### Table 8: teacher-set construction

Q-Cluster = Qualified Cluster Count@2m among proposals with PDMS ≥ 0.8; Top-6 ADE/FDE among the six best by true PDMS.

| Teacher targets | PDMS | Q-Cluster@2m | Top-6 ADE | Top-6 FDE |
|---|---:|---:|---:|---:|
| Real-PDMS top-$k$ | 93.9 | 6.45 | 1.04 | 2.31 |
| Real-PDMS top-$k$ + distance suppression | 94.3 | 7.49 | 1.66 | 2.47 |
| Vector-Pareto targets (ours) | 94.5 | 8.71 | 1.75 | 2.59 |

The rows change two things at once: real against predicted scores, and scalar against vector selection. **Picking targets with the true evaluator does worse than picking them with the learned scorer's Pareto set**, but no row isolates which change matters.

### Table 9: anchor reranking (NAVSIM-v2)

| Inference setting | EC | EPDMS |
|---|---:|---:|
| Single-mode anchor only | 77.6 | 88.7 |
| CLOVER without anchor | 35.0 | 89.3 |
| CLOVER + anchor soft reranking | 75.5 | 90.4 |

**Number of proposals** (text): $K$ = 32 / 64 / 96 gives 93.9 / 94.5 / 94.4 PDMS. [[sources/drivor.md]] found the same saturation (64 and 128 both 90.0 on navval).

---

## Appendix A: Pseudo-Expert Generation

Data flow: metric cache → candidate families → drivable-area / occupancy pre-check → PDM scoring → coverage-aware selection → post-processing → training-time sampling. **Privileged inputs**: current ego state, route centerline, drivable-area map, future obstacle occupancy and the NAVSIM PDM evaluator.

Candidates live in a centerline frame: progress $d_t=d_0+\sum_{k\le t}v_k\Delta t$, lateral offset $\mathrm{lat}_t$, mapped back by $x_t=x_\mathrm{cl}(d_t)-\mathrm{lat}_t\sin h_\mathrm{cl}(d_t)$, $y_t=y_\mathrm{cl}(d_t)+\mathrm{lat}_t\cos h_\mathrm{cl}(d_t)$, $\theta_t=h_\mathrm{cl}(d_t)$. Lateral transitions use the quintic smooth step $s(r)=6r^5-15r^4+10r^3$.

### Table 10: candidate families

| Family | Count | Purpose |
|---|---:|---|
| Constant-speed lateral transitions | 200 | Smooth and aggressive lateral shifts at different speeds |
| Off-road boundary cases | 12 | Populate low-score drivable-area violations |
| Acceleration profiles | 18 | Vary longitudinal progress and speed |
| Stop-go behaviors | 9 | Safe but low-progress |
| Approach-brake behaviors | 10 | TTC / comfort trade-offs |
| Overshoot lateral motions | 12 | Aggressive lateral dynamics, comfort variation |
| **Total** | **261** | Per scene |

**Selection.** Lightweight checks label each candidate feasible, near-feasible or infeasible; at most 180 go to the full evaluator, mostly feasible and near-feasible with a few infeasible ones kept. Each scored candidate gets a coverage key (DAC, Collision, TTC, Comfort, ProgressBin), and candidates are picked greedily by $\mathrm{cost}_i=10\cdot\mathrm{cov\_count}(\mathrm{key}_i)+\mathrm{bin\_count}(\mathrm{score\_bin}_i)$, up to 50 per scene. Post-processing drops wrong-direction candidates (DDC < 1) and interpolates one extra candidate at up to three score boundaries per scene. At training time candidates with PDMS ≥ 0.8 are sorted by score and thinned by farthest-point sampling to 8; the logged trajectory is added if not covered, and is the fallback when nothing qualifies.

### Table 11: pseudo-expert hyperparameters

| Hyperparameter | Value |
|---|---|
| Future poses | 8 |
| Sampling interval | 0.5 s |
| Speed candidates | (0, 2, 4, 6, 8, 10, 12, 15) m/s |
| Regular lateral offsets | (−3.5, −2.0, −1.0, −0.5, 0, 0.5, 1.0, 2.0, 3.5) m |
| Off-road lateral offsets | (−7.0, −5.5, 5.5, 7.0) m |
| Transition portions | (0.35, 0.6, 1.0) |
| Acceleration rates | (−2.0, −1.0, −0.5, 0.5, 1.0, 2.0) m/s² |
| Raw candidates per scene | 261 |
| Maximum PDM-scored candidates per scene | 180 |
| Retained pool per scene | 50 |
| Score-drop threshold for boundary interpolation | 0.25 |
| Maximum score boundaries per scene | 3 |
| Interpolation samples per boundary | 1 |
| Training-time score threshold | 0.8 |
| Training-time pseudo-expert top-$K$ | 8 |

![[pseudo_expert_visualization.png|Figure 4: Pseudo-expert pools on six scenes, front view and BEV, coloured by true PDM score (0 to 1). The pools span lateral offsets across the lane and beyond its edge, different progress, and boundary cases; high-scoring candidates (yellow) cluster around feasible paths, low-scoring ones (purple) leave the drivable area or close on vehicles.]]

**Figure 4**: Six scenes (curves, an intersection, multi-lane roads with traffic). The pools deliberately keep low-scoring and boundary candidates next to the feasible ones.

**Against the wiki's candidate banks.** [[sources/world4scorer.md]]'s bank "following CLOVER" averages 54.5 valid entries per scene, while the pool described here keeps at most 50 plus at most 3 interpolations and then drops wrong-direction entries; the bank is probably drawn from the scored pool (up to 180) rather than the retained one. Its score mix (35% low, 47% mid, 17% high) fits the coverage rule's deliberate spread across score bins.

---

## Appendix B: The NAVSIM Scores

$$\mathrm{PDMS}=\mathrm{NC}\times\mathrm{DAC}\times\frac{5\,\mathrm{EP}+5\,\mathrm{TTC}+2\,\mathrm C}{12}$$

### Table 12: EPDMS composition (NAVSIM v2)

| Metric | Weight | Range |
|---|---|---|
| No at-fault Collisions (NC) | multiplier | {0, ½, 1} |
| Drivable Area Compliance (DAC) | multiplier | {0, 1} |
| Driving Direction Compliance (DDC) | multiplier | {0, ½, 1} |
| Traffic Light Compliance (TLC) | multiplier | {0, 1} |
| Ego Progress (EP) | 5 | [0, 1] |
| Time to Collision (TTC) | 5 | {0, 1} |
| Lane Keeping (LK) | 2 | {0, 1} |
| History Comfort (HC) | 2 | {0, 1} |
| Extended Comfort (EC) | 2 | {0, 1} |

The false-positive filter sets $\mathrm{filter}_m(\mathrm{agent},\mathrm{human})=1$ when $m(\mathrm{human})=0$ and $m(\mathrm{agent})$ otherwise; EPDMS is the product of the filtered multipliers times the weighted mean of the filtered weighted terms. The appendix also restates the two-stage aggregation (a first-stage score times a Gaussian-kernel-weighted second-stage score).

---

## Appendix C: Anchor Sweep

### Table 13: coarse sweep (scores are normalized EPDMS; reported on the evaluation split)

| $\lambda_s$ | $\lambda_{xy}$ | $\lambda_\psi$ | Score | EP | EC | HC |
|---:|---:|---:|---:|---:|---:|---:|
| – | – | – | 0.89322 | 0.92408 | 0.3500 | 0.96459 |
| **2.00** | **0.20** | **0.50** | **0.90356** | 0.8689 | 0.7550 | 0.9823 |
| 1.75 | 0.20 | 0.75 | 0.90213 | 0.8613 | 0.7641 | 0.9825 |
| 2.00 | 0.20 | 0.75 | 0.90292 | 0.8652 | 0.7594 | 0.9825 |
| 2.25 | 0.20 | 0.75 | 0.90222 | 0.8689 | 0.7509 | 0.9823 |
| 1.75 | 0.20 | 1.00 | 0.90212 | 0.8596 | 0.7667 | 0.9825 |
| 2.00 | 0.20 | 1.00 | 0.90212 | 0.8630 | 0.7605 | 0.9825 |
| 2.25 | 0.20 | 1.00 | 0.90168 | 0.8663 | 0.7539 | 0.9824 |
| 2.00 | 0.25 | 0.50 | 0.90285 | 0.8633 | 0.7610 | 0.9825 |
| 2.00 | 0.25 | 0.75 | 0.90194 | 0.8601 | 0.7640 | 0.9825 |
| 1.75 | 0.35 | 0.75 | 0.90086 | 0.8502 | 0.7698 | 0.9828 |
| 2.00 | 0.35 | 0.75 | 0.90177 | 0.8534 | 0.7679 | 0.9826 |
| 2.25 | 0.35 | 0.75 | 0.90216 | 0.8564 | 0.7660 | 0.9825 |
| 1.75 | 0.35 | 1.00 | 0.90021 | 0.8492 | 0.7718 | 0.9828 |
| 2.00 | 0.35 | 1.00 | 0.90118 | 0.8521 | 0.7699 | 0.9826 |
| 2.25 | 0.35 | 1.00 | 0.90180 | 0.8549 | 0.7679 | 0.9825 |

- **Swept on the test split.** The text says "validation/evaluation split", and the best configuration's 0.90356 is the reported navtest result. The fourteen anchored configurations span 0.90021–0.90356, so picking the best of them is worth at most 0.34; the worst anchored configuration is still +0.70 over no anchor.
- **Where the +1.0 comes from**, in weighted terms: EC +40.5 points (+5.1 of the 16-point weighted sum), EP −5.5 (−1.7), HC +1.8 (+0.2).
- Stronger heading or position anchoring raises EC further and lowers EP and the total, so the comfort–progress trade is continuous.

---

## Appendix E: Implementation

### Table 15: base architecture

| Hyperparameter | Value |
|---|---|
| Input cameras | front, left, right, back |
| LiDAR | disabled |
| Backbone | DINOv2 ViT-S (`timm/vit_small_patch14_reg4_dinov2.lvd142m`) |
| LoRA | rank 32 |
| Scene tokens per camera | 16 |
| Horizon / interval / poses | 4.0 s / 0.5 s / 8 |
| Proposals | 64 |
| Transformer hidden / FFN | 256 / 1024 |
| Trajectory representation | $(x,y,\theta)$, one token per trajectory |
| NAVSIM v1 deployment weights | NC/DAC/DDC/TTC/EP/Comfort = 1/1/0/5/5/2 |

### Table 16: Stage 1

| Hyperparameter | Value |
|---|---|
| Epochs / batch / GPUs / workers | 25 / 32 / 4 / 4 |
| Optimizer, LR, precision | AdamW, 2×10⁻⁴, 16-mixed |
| Seed | 2 |
| Ray scoring | enabled, 96 threads per node |
| Pseudo-expert top-$K$ / threshold / loss weight | 8 / 0.8 / 0.5 |
| Pseudo-expert on all refinement layers | false |
| Trajectory imitation / final score loss weights | 1.0 / 1.0 |
| Prediction cross-entropy / L1 / area loss weights | 1.0 / 0.1 / 2.0 |

### Table 17: pseudo-expert generation

| Hyperparameter | Value |
|---|---|
| Poses / interval | 8 / 0.5 s |
| Speed candidates | (0, 2, 4, 6, 8, 10, 12, 15) m/s |
| Regular / off-road lateral offsets | (−3.5 … 3.5) m / (−7.0, −5.5, 5.5, 7.0) m |
| Transition portions | (0.35, 0.6, 1.0) |
| Acceleration rates | (−2.0, −1.0, −0.5, 0.5, 1.0, 2.0) |
| Score bins | (0.0, 0.2, 0.4, 0.6, 0.8, 1.01) |
| Progress bins | (0.0, 0.2, 0.5, 0.8, 1.01) |
| Retained per scene / max scored per scene | 50 / 180 |

### Table 18: Stage 2

| Hyperparameter | Value |
|---|---|
| Initialization | Stage-1 checkpoint |
| Cycles (scorer epoch + generator epoch) | 30 (60 epochs in total), starting with scorer fitting |
| Batch / GPUs / workers / LR | 32 / 4 / 8 / 3×10⁻⁵ |
| Freeze visual backbone | true |
| Detach proposals in scorer; sanitize proposals; XY limit | true; true; 100.0 |
| Ray scoring | enabled, 96 threads per node |
| Loss weights: trajectory / intermediate / final score / Pareto / teacher stability | 0.1 / 0.02 / 1.0 / 1.0 / 0.05 |
| Pareto set size | 2–8 |
| Teacher reward threshold | 0.4 |

### Table 19: compute

| Item | Value |
|---|---|
| GPUs | 4 × NVIDIA A100 |
| CPU | AMD EPYC 7742, 64 cores |
| Stage 1 / Stage 2 | 30–40 h / 60–80 h |
| Total | 4–5 days |

Training time includes real-time scoring with the NAVSIM evaluator, so it depends on CPU throughput. Latency (§5.1): about 110 ms per scene on one A100 with $K=64$, "in the NAVSIM-v2 setting". The same figure as [[sources/drivor.md]]'s own (107 ms of it the ViT); whether the anchor model's forward pass is included is not stated.

---

## Appendix F: Fixed-Proposal Scorer Diagnostic

Each navtrain and navtest scene gets 64 candidates from the trained CLOVER generator, labelled with true PDMS sub-scores; scorers with different backbones are trained on that fixed set and scored by the true PDMS of their top-1.

### Table 20: scorer backbones

| Scorer backbone | Train score | Test score |
|---|---:|---:|
| DINOv2-Small (deployed) | 95.0 | 92.4 |
| DINOv2-Base | 95.3 | 92.6 |
| DINOv3-Small | 95.1 | 92.4 |
| DINOv3-Base | 95.1 | 92.7 |
| Wan2.2-5B features (LoRA, block 15) | 97.7 | 92.8 |

- **The backbone is worth at most +0.4 on navtest**, and the video-generation model's +0.4 comes with a 4.9-point train–test gap against 2.6 for DINOv2-S. This agrees with [[sources/adaptive-wam.md]]'s finding (same first author) that with 64 proposals Wan features beat ViT-L by only 0.28.
- **Every isolated scorer selects about 2 points below the deployed system's 94.5.** The paper says the fixed pool "differs from a fully deployed CLOVER system" and that the scores should not be compared, without saying how it differs. If the pools are the deployed generator's navtest proposals, the deployed scorer's advantage is its training (Stage-1 pretraining and 30 cycles of refitting on the co-adapting generator), not its architecture.
- The authors release this as a scorer benchmark.

---

## Appendix G: Seeds

### Table 21: three training seeds

| Setting | Seed 1 | Seed 2 | Seed 3 | Max − Min |
|---|---:|---:|---:|---|
| NAVSIM v1 PDMS | 94.488 | 94.502 | 94.492 | < 0.02 |
| NAVSIM v2 EPDMS | 90.354 | 90.358 | 90.368 | < 0.01 ⚠ |
| NAVSIM v2 EPDMS* | 87.201 | 87.207 | 87.210 | < 0.01 |

⚠ The EPDMS range is 0.014, not under 0.01. The reported best anchor configuration (0.90356) is none of the three. Which stages each seed re-runs is not said; Table 16 lists Stage 1 with seed 2.

---

## Table and Text Audit

| Check | Result |
|---|---|
| v1 closed form, 24 rows | All pass (−0.03 to −9.77); CLOVER −1.05. Rows are DrivoR's Table 1 camera-only block |
| v2 residual signs against stated columns | Agree on every labelled row (pre-fix +1.28 to +2.67; corrected −0.55, −1.08; DiffusionDriveV2 +0.10, its known exception) |
| navhard product check | CLOVER −1.81 (corrected-like); the four baseline rows reproduce DrivoR's corrected block, including the suspect GTRS-Aug Stage-2 EP 98.9 |
| Table 5 vs Tables 22–23 | Stage-2 selected 0.9448 vs 0.9443; only 0.9443 fits the printed gap (Table 14 repeats 0.9448 with gap 0.0495) |
| Table 21 | v2 range 0.014 printed as "< 0.01" |
| Table 18 vs §3.3 | $\mathcal L_\mathrm{gen}$ has a top-$k$ term; Table 18 lists no top-$k$ weight |
| Table 9 vs Table 13 | Consistent (89.3 ↔ 0.89322; 90.4 ↔ 0.90356) |
| Text: "ranking first or second across all PDMS sub-scores" (v1) | Not quite: first on EP and PDMS, second on NC (Centaur 99.5) and DAC (DriveVLA-W0 99.1), **third on TTC** (AutoVLA and Centaur 98.0 against 96.9), Comfort tied at 100 |
| Text: "lowest ST-P3 L2 error" (nuScenes) | True by 0.01 m over EMMA |

---

## Limitations

1. **Training split not stated.** The headline comparison is against DrivoR's trainval row, and the baseline in every analysis is DrivoR's released (probably trainval) model, so CLOVER is most plausibly trained on navtrain + navval. That matters for its place above World4Scorer's navtrain-only 94.0.
2. **Model selection on the test split.** The anchor weights are the best of a 15-point sweep scored on navtest (worth at most 0.34), $K$ was chosen from 32 / 64 / 96 on navtest, and every ablation is a navtest number. [[sources/drivor.md]] ran its ablations on navval.
3. **The v2 headline omits the base model**, which the same group later measures at 90.66 against CLOVER's 90.4. Without the anchor CLOVER is 89.3, below both.
4. **The v2 configuration runs a second network** (the single-mode anchor). The 110 ms latency is DrivoR's single-network figure, and the paper says CLOVER "introduces no additional inference-time modules over the base DrivoR-style architecture" in the same paragraph that describes the anchor branch.
5. **Privileged, evaluator-in-the-loop training.** Pseudo-experts are built from the training metric cache (route centerline, drivable area, future obstacle occupancy) and filtered by the evaluator; the scorer is refit to the evaluator's sub-scores on the model's own proposals every cycle. This is the [top-of-leaderboard](../concepts/selection-based-planning.md#top-of-leaderboard) pattern in its strongest form, and it is evaluated only on NAVSIM's own metric (plus an undescribed nuScenes run).
6. **No transfer to navhard**: +0.8 on navtest, +0.0 on navhard, with EC 17–19 points below DrivoR in both stages.
7. **The theory checks half its conditions**, in aggregate, on the test split's final proposals; the conservative-update bound is never measured.
8. **nuScenes is unreproducible from the text**: no training description, ego status unmarked, a convention label that other papers use for the other convention.
9. **Single runs for every ablation**; the seed study covers only the full model. With a 0.014 seed range, the ablation differences of 0.1–0.4 would be meaningful if the ablations were equally stable, which is not shown.
10. **No closed-loop evaluation**, no HUGSIM (which DrivoR reports), four of NAVSIM's eight cameras.

---

## Relation to Other Wiki Pages

- [[sources/drivor.md]]: the architecture, the baseline and most of the comparison rows. CLOVER changes only training; its v1 and navhard tables copy DrivoR's (v1 camera-only block, navhard corrected block).
- [[sources/world4scorer.md]] (same group, September 2026): keeps DrivoR's generator and replaces the scorer, uses a candidate bank "following CLOVER", measures DrivoR's released model at 93.69 / 90.66 with the same pool statistics CLOVER prints, and leaves CLOVER out of its tables. On v1 its scorer-only gain on DrivoR's pools (+0.50) and CLOVER's regret reduction on its own (+0.69) are the two measurements of where a DrivoR-stack gain comes from. On v2 its inertial re-ranking (+1.6) and CLOVER's anchor (+1.1) are the two continuity fixes on the same stack.
- [[sources/adaptive-wam.md]] (same authors, August 2026): its auxiliary 92.6 PDMS model uses CLOVER's pseudo-expert targets; its "tie problem" analysis is on 12,146 scenes, the same navtest set.
- [[sources/auto-jepa.md]] (a different Tsinghua group): initializes its scorer from CLOVER's released checkpoint (+3.7 of its 91.3) and reports CLOVER at 90.4 EPDMS, which is the corrected value.
- [[sources/curious-vla.md]] (shared authors): the proposal-diversity protocol of Table 5; both papers argue that imitation of one trajectory collapses diversity.
- [[sources/maneuverflow.md]]: also trains its scorer on online evaluator labels of its own candidates and finds the selector, not the generator, limits the score.
- [[sources/drive-jepa.md]], [[sources/world4scorer.md]]: the other two EC fixes ([[concepts/selection-based-planning.md#clover]]).
- [[sources/wa-jepa.md]], [[sources/coworld-vla.md]], [[sources/drivefuture.md]], [[sources/ad-e2e-jepa.md]]: the other tables printing both evaluator columns.
- [[sources/dvgt-2.md]]: the opposite use of the "ST-P3" label on nuScenes.
- [[sources/sparsedrivev2.md]] (ingested 2026-10-14): the source of this paper's v2 baseline rows (86.7 / 90.1 for itself).
- Not ingested: **GTRS / GTRS-Dense / GTRS-Aug**, **RAP**, **iPad**, **Evadrive**, **GoalFlow**.

---

## Open Questions

- **What does the generator add at a fixed scorer?** Scoring CLOVER's pools with DrivoR's scorer, or DrivoR's pools with CLOVER's scorer, would split the +0.74 between candidates and ranking directly instead of through the oracle identity.
- **Was the 94.5 model trained on navtrain or on trainval?**
- **Can the anchor be replaced by a stateful rule** (inertial re-ranking) or folded into the scorer, and does the diversity-EC trade survive a scorer that predicts EC?
- **Why do isolated scorers select about 2 points below the deployed one** on what the paper describes as CLOVER-generated pools?
- **Would navhard move** with the anchor, with off-log training data like DrivoR's SimScale rows, or with pseudo-experts generated in the rendered Stage-2 states?
- **Does the enrichment condition hold cycle by cycle in training**, and how large is $\eta_t$?
