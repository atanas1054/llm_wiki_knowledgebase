---
title: "RiSE-Drive: Risk-Sensitive Long-Horizon Candidate Evaluation for End-to-End Autonomous Driving"
type: source-summary
sources: ["raw/papers/RiSE-Drive_ Risk-Sensitive Long-Horizon Candidate Evaluation for End-to-End Autonomous Driving.md"]
related: [sources/drivor.md, sources/world4scorer.md, sources/clover.md, sources/toad.md, sources/chainflow-vla.md, sources/mm-future.md, sources/da-wam.md, sources/simscale.md, sources/drivefuture.md, sources/idol.md, sources/simvta.md, sources/hydra-mdp-pp.md, sources/drivesuprim.md, sources/drive-jepa.md, sources/resworld.md, concepts/selection-based-planning.md, concepts/navsim-benchmark.md, concepts/navhard-ood-evaluation.md, concepts/hugsim-benchmark.md, concepts/counterfactual-prediction.md, concepts/world-model-for-ad.md, concepts/evaluation-variance.md, concepts/inference-latency.md, research-directions.md]
created: 2026-11-12
updated: 2026-11-12
confidence: low
---

# RiSE-Drive: Risk-Sensitive Long-Horizon Candidate Evaluation for End-to-End Autonomous Driving

**Paper**: RiSE-Drive: Risk-Sensitive Long-Horizon Candidate Evaluation for End-to-End Autonomous Driving
**Authors**: Yu Zhao, Hongyi Duan, Ming Yin, Hongyuan Zhang, Fan Yang, Hongzhi Wang, Xiubo Liang
**Orgs**: School of Software Technology, Zhejiang University (Ningbo; Fan Yang at Hangzhou)
**Venue**: ACM Multimedia 2026 (MM '26, Rio de Janeiro, November 10–14, 2026); DOI 10.1145/3767308.3836337
**arXiv**: 2611.13790v1 (November 2026)
**Code**: none stated, no project page
**Source**: `raw/papers/RiSE-Drive_ Risk-Sensitive Long-Horizon Candidate Evaluation for End-to-End Autonomous Driving.md`. The clipping has the method, all five tables, Table A.1, all three figures (`rise-teaser.png`, `risepipeline.png`, `syb.png`, already in `raw/assets/`) and the references. Two pieces of text are missing from it and were read from the arXiv HTML (v1): the three-item contributions list in §1 and the §5.2 paragraph on the NAVSIM-v2 results. Both are summarized below with short quotes. The HTML confirms that nothing else is missing: there is no sensitivity table, no NavSafe or navtest-EPDMS table, no code link and no latency figure.

**Confidence: low.** The tables are internally consistent: every NAVSIM-v1 row passes the closed-form check, the navhard row sits on the corrected scale, and the HUGSIM averages fit the 345-scenario weighting. What is low is the ability to tell what produced the numbers:
- The supervision that defines the method appears only in the appendix. The "future conditions" are four scripted perturbations of other vehicles, and the metric behind the branch targets is never named. How the perturbations are run inside NAVSIM is described in one sentence.
- The training split, epochs and starting checkpoint are not stated. They decide whether 94.2 should be read against DrivoR's 93.1 (navtrain) or 93.7 (trainval).
- The ablations are single runs on the test splits, and the named mechanism, risk-sensitive ranking, is their smallest step (+0.3 PDMS).
- §5.3 promises a "sensitivity to the future-branch and quantile design" study. It does not exist. §5.1 promises navtest EPDMS and NavSafe results; neither is reported.
- No code.

---

## What It Is

**DrivoR with its scorer replaced by an evaluator trained on the benchmark's scores of every candidate under four scripted behaviours of other vehicles, and ranking by mean, worst case and spread across those four.**

DrivoR's encoder (DINOv2 ViT-S, LoRA, 64 register tokens) and its 64-query generator are kept. Its scoring decoder is removed. For each candidate the new evaluator:

1. Takes four branch codes, one per "future condition", computed by four heads from the **mean of the 64 scene tokens**.
2. Runs a gated recurrence over the candidate's own 8 poses, once per branch, starting from a mix of the scene mean, a candidate embedding and the branch code.
3. Decodes a discounted sum of per-step outcomes per branch ($\hat y_{i,l}$), and 10 ordered quantiles from the state averaged over steps and branches.
4. Ranks by $\hat J_i=\hat\mu_i-0.5\,(\hat\mu_i-\widehat{\mathrm{CVaR}}_{0.2})-0.1\,\hat\sigma_i$.

The appendix says what the branches are. Each is a fixed pair of motions applied in the benchmark rollout: a lead vehicle's longitudinal acceleration (low or high "pressure") and an adjacent same-direction vehicle's lateral velocity toward the ego corridor (low or high). Every candidate is scored under all four, and the four scores supervise the four branches.

**Results**

| Benchmark | RiSE-Drive | DrivoR (same table) | Δ |
|---|---:|---:|---:|
| NAVSIM-v1 PDMS (navtest) | **94.2** | 93.1 (navtrain model) | +1.1 |
| navhard two-stage EPDMS | **51.3** | 48.3 (corrected) | +3.0 |
| HUGSIM, 345 scenarios: RC / HD-Score | 50.1 / **39.5** | 49.8 / 35.7 | +0.3 / +3.8 |
| Parameters, trainable / total | 18.9M / 42.3M | 18.8M / 41.8M | +0.1M / +0.5M |

Training: AdamW, learning rate 2 × 10⁻⁴, cosine decay, 4 × RTX 4090. Split, epochs and initialization not stated.

---

## Key Takeaways

- **The new ingredient is extra labels: four scripted counterfactuals of other vehicles per candidate.** Every scorer on DrivoR's stack so far learns the oracle's score of each candidate in the logged scene ([[sources/drivor.md]], [[sources/world4scorer.md]], [[sources/clover.md]]). RiSE-Drive learns its score in four modified scenes: the lead vehicle accelerates or brakes, an adjacent vehicle drifts away from or toward the ego lane. The observation is unchanged; only the future is altered. That is rung-2 interventional supervision of the scorer on other agents' behaviour, with no rendering ([[concepts/counterfactual-prediction.md#rise-drive]]). The paper's §3 instead presents the conditions as samples from a scene-conditional distribution $p(\omega\mid o_t)$. In fact they are four fixed, equally weighted settings, and they collapse to one in scenes without a lead or adjacent vehicle.
- **"Tail risk" is the worst of four moderate conditions.** Each condition's motion is the logged pair closest to the median of its quadrant (median splits of the training logs), so the "high" settings sit near the 75th percentile, not in the tail. With four equally weighted targets, the empirical lower-20% CVaR is the single worst branch. The training target is therefore $J^\star=0.5\,\mu^\star+0.5\min_l y^\star_{i,l}-0.1\,\sigma^\star$, a fixed blend of mean and worst case. Figure 1's "possible risky conditions" are quantile bars, not conditions.
- **"Long-horizon" is the candidate's own 4 s.** The recurrence runs over the candidate's $H=8$ poses, the same horizon PDMS already scores, with $\gamma=0.99$ (weights 1.00 to 0.92). The latent state has no target other than the per-branch outcome sum. No agent, map or scene state is predicted. The "latent future rollout" is a trajectory encoder, not a world model ([[concepts/world-model-for-ad.md#consequence-propagation]]).
- **The evaluator sees the scene only through the mean of the 64 register tokens.** DrivoR's scorer cross-attends to the tokens, and its Figure 6 shows that scoring draws on different cameras for different candidates. As written, RiSE-Drive's evaluator has no per-candidate access to the scene, and it still scores higher. Whether some other path exists is not said; Table 4's B-II replaces DrivoR's scoring decoder with "a lightweight direct score head" and every later row builds on B-II.
- **On navhard this is the first evaluator-only change on DrivoR's architecture, and it is worth +3.0 on the same corrected scale.** 48.3 → 51.3 (product of stage closed forms 45.1 → 50.0). The gain is in Stage 2: NC 90.2 → 91.5, DAC 88.4 → 89.8, DDC 91.9 → 94.6, TTC 88.0 → 91.3, LK 50.1 → 54.9, with EP 70.0 → 68.7. For comparison, [[sources/clover.md]] retrains generator and scorer against the evaluator and gains +0.0, and [[sources/toad.md]]'s test-time search on the SimScale model gains +1.7. Everything is retrained here (Table 5 counts the LoRA and generator as optimized), so this is not a fixed-pool swap like World4Scorer's +0.50 on navtest. It is one run.
- **The risk-sensitive ranking, the paper's title, is the smallest step in its ladder.** From a weakened baseline (DrivoR with a direct score head, 91.9 / 45.6), the future-condition generator adds +0.7 PDMS / +1.2 EPDMS, the candidate encoder +0.3 / +0.7, the step encoder and propagation +0.6 / +1.7, the quantile head +0.4 / +1.4, and risk-sensitive ranking +0.3 / +0.7. Single runs on navtest and navhard. The ladder adds the branch labels in its first row, together with the branch generator, so labels and architecture are never separated.
- **The safety-progress trade the paper describes is visible on all three benchmarks.** v1: NC +0.6, TTC +2.2, EP −0.4. navhard Stage 2: TTC +3.3, EP −1.3. HUGSIM: HD-Score +3.8 but Easy RC −4.4 and Extreme RC −0.8, with HD gains growing with difficulty (+1.3 / +2.6 / +3.9 / +6.7).
- **Comparison scope.** "Best PDMS of 94.2" holds within Table 1, which leaves out DrivoR's own trainval (93.7) and SimScale (94.0, 94.6) rows, [[sources/world4scorer.md]] 94.0 (navtrain), [[sources/mm-future.md]] 94.0, [[sources/clover.md]] 94.5, [[sources/chainflow-vla.md]] 94.8 and DriveSuprim's published 93.5 (carried at its ResNet-34 89.9). "Best EPDMS of 51.3" on navhard leaves out DriveFuture 55.5, DrivoR + SimScale 52.3 / 54.6, SimScale 53.2 and TOAD 56.3. "Best average RC" on HUGSIM leaves out [[sources/simvta.md]] (51.5 on the same 345 scenarios; RiSE-Drive still leads its HD-Score, 39.5 vs 38.8).

---

## Method

![[risepipeline.png|Figure 2: Overview of the RiSE-Drive pipeline. Multi-view observations are encoded, and the proposal generator produces K ego-trajectory candidates. A future-condition generator predicts L branch-specific condition codes. For each candidate–condition pair, RiSE-Drive initializes a latent consequence state, propagates it along the trajectory, and decodes one branch-level long-horizon outcome sample. The branchwise consequence traces are pooled over time and across branches and mapped to ordered outcome quantiles by a distribution head. The final ranking score combines the mean estimate, lower-tail CVaR, and cross-condition instability.]]

**Figure 2**: Five blocks. *Candidate trajectory generation* (DrivoR's, unchanged): multi-view images → perception encoder → scene tokens → proposal generator with ego states → $K$ candidates. *Future-condition modeling*: mean pooling of the scene tokens → a "summarized scene token" → future-condition generator → $L$ codes. *Consequence propagation*: a trajectory-level encoder ($K$ embeddings) and a step-level encoder ($K\times H$ embeddings) feed state initialization and propagation; instantaneous and terminal outcomes give $\hat y_{i,l}$. *Distributional outcome estimation*: time pooling → branch pooling → distribution head → $\mathbf q_i$. *Risk-sensitive ranking*: mean estimate, lower-tail CVaR and cross-condition instability combine into the final score. The figure has no box for the branch labels, which are the method's supervision.

### Baseline: DrivoR as a generate-and-score planner (§3.1)

Observation $o_t=(\{\mathbf I_t^{(n)}\}_{n=1}^N,\mathbf e_t)$, scene tokens $\mathbf R_t=F_\mathrm{enc}(o_t)\in\mathbb R^{N_r\times D}$, candidates $\mathcal T_t=\{\tau_1,\dots,\tau_K\}=G_\mathrm{prop}(\mathbf R_t,\mathbf e_t)$ with $\tau_i=\{(x_{i,h},y_{i,h},\theta_{i,h})\}_{h=1}^H$. DrivoR's scorer predicts sub-scores $\mathbf s_i=G_\mathrm{score}(\mathbf R_t,\tau_i)\in\mathbb R^C$, aggregates $u_i=\boldsymbol\lambda^\top\mathbf s_i$ and takes the argmax.

§3.2 recasts the score as a point estimate of an outcome variable $Y_i=Y(\tau_i,\omega)$, $\omega\sim p(\omega\mid o_t)$, and proposes ranking by statistics of its distribution.

### Future conditions (§4.2)

$$\bar{\mathbf r}_t=\frac1{N_r}\sum_{j=1}^{N_r}\mathbf R_t^{(j)},\qquad \omega_l=P^{(l)}_\mathrm{ctx}(\bar{\mathbf r}_t)\in\mathbb R^{d_\omega},\quad l=1,\dots,L$$

$L$ separate heads on the mean-pooled scene. The code index $l$ is tied to a condition only through the supervision ($\hat y_{i,l}$ is regressed onto the score under condition $l$).

### Candidate-conditioned consequence propagation (§4.3)

$$\mathbf c_i=E_\tau(\tau_i),\qquad \mathbf z_{i,h}=E_p(\mathbf p_{i,h}),\qquad \mathbf h_{i,l,0}=W_r\bar{\mathbf r}_t+W_c\mathbf c_i+W_\omega\omega_l+\mathbf b_0$$

$$\tilde{\mathbf h}_{i,l,h}=\phi(W_h\mathbf h_{i,l,h-1}+W_p\mathbf z_{i,h}+W_s\bar{\mathbf r}_t+\mathbf b_h),\qquad \mathbf g_{i,l,h}=\sigma(U_h\mathbf h_{i,l,h-1}+U_p\mathbf z_{i,h}+U_s\bar{\mathbf r}_t+\mathbf b_g)$$

$$\mathbf h_{i,l,h}=\mathbf g_{i,l,h}\odot\tilde{\mathbf h}_{i,l,h}+(1-\mathbf g_{i,l,h})\odot\mathbf h_{i,l,h-1}$$

$$\hat y_{i,l}=\sum_{h=1}^H\gamma^{h-1}g_r(\mathbf h_{i,l,h})+\gamma^H g_T(\mathbf h_{i,l,H})$$

A GRU-style recurrence over the candidate's poses. The branch code enters only the initial state. The scene enters only as the constant vector $\bar{\mathbf r}_t$. Only the sum $\hat y_{i,l}$ is supervised, so how it splits across steps and the terminal term is not identified.

### Distributional outcome estimation (§4.4)

$$\bar{\mathbf h}_{i,l}=\frac1H\sum_h\mathbf h_{i,l,h},\qquad \tilde{\mathbf h}_i=\frac1L\sum_l\bar{\mathbf h}_{i,l}$$

$$q_i^{(1)}=h_0(\tilde{\mathbf h}_i),\qquad q_i^{(m)}=q_i^{(m-1)}+\mathrm{softplus}\big(h_m(\tilde{\mathbf h}_i)\big),\qquad \hat\mu_i=\frac1M\sum_m q_i^{(m)}$$

The quantile head reads the state averaged over steps and branches, not the four branch predictions. The quantile levels $\alpha_m$ are not given.

### Risk-sensitive ranking (§4.5)

$$\widehat{\mathrm{CVaR}}^{\downarrow}_\alpha(Y_i)=\frac1{M_\alpha}\sum_{m=1}^{M_\alpha}q_i^{(m)},\quad M_\alpha=\lceil\alpha M\rceil,\qquad \hat\Delta_i=\hat\mu_i-\widehat{\mathrm{CVaR}}^{\downarrow}_\alpha$$

$$\hat\sigma_i=\sqrt{\frac1L\sum_l(\hat y_{i,l}-\bar y_i)^2},\qquad \hat J_i=\hat\mu_i-\lambda_\mathrm{tail}\hat\Delta_i-\lambda_\mathrm{var}\hat\sigma_i$$

With $M=10$ and $\alpha=0.2$, $M_\alpha=2$: CVaR is the mean of the two lowest predicted quantiles. Mean and CVaR come from the quantile head and $\hat\sigma$ from the branch predictions.

### Training (§4.6)

$$\mathcal L_\mathrm{branch}=\frac1{KL}\sum_{i,l}\ell_\mathrm{Huber}(\hat y_{i,l},y^\star_{i,l}),\qquad \mathcal L_\mathrm{quant}=\frac1{KML}\sum_{i,m,l}\rho_{\alpha_m}\big(y^\star_{i,l}-q_i^{(m)}\big),\qquad \rho_\alpha(u)=u(\alpha-\mathbf 1[u<0])$$

$$J^\star_i=\mu^\star_i-\lambda_\mathrm{tail}\big(\mu^\star_i-\mathrm{CVaR}^{\star\downarrow}_\alpha(Y_i)\big)-\lambda_\mathrm{var}\sigma^\star_i,\qquad \mathcal L_\mathrm{rank}=\sum_{i\neq j}\max\big(0,\ \delta-\mathrm{sign}(J^\star_i-J^\star_j)(\hat J_i-\hat J_j)\big)$$

$$\mathcal L=\lambda_b\mathcal L_\mathrm{branch}+\lambda_q\mathcal L_\mathrm{quant}+\lambda_r\mathcal L_\mathrm{rank}$$

$\mu^\star$, $\sigma^\star$ and CVaR$^\star$ are taken over the $L=4$ branch targets of each candidate. Read with the discrete formula above, the lowest 20% of four equally weighted atoms is one atom, the minimum. The pinball loss has the same effect on the quantile head: if the 10 levels are evenly spaced (unstated), the two lowest both target the worst branch, so $\widehat{\mathrm{CVaR}}$ estimates the worst-branch score. The pairwise hinge runs over all $64\times63$ ordered pairs per scene.

**Settings (§5)**: $K=64$, $L=4$, $M=10$, $d_\omega=d_c=d_p=d_h=256$, $\alpha=0.2$, $\gamma=0.99$, $\lambda_\mathrm{tail}=0.5$, $\lambda_\mathrm{var}=0.1$, $\delta=0.1$, $\lambda_b=\lambda_q=\lambda_r=1$. Backbone as DrivoR: four cameras, DINOv2 ViT-S with 16 registers per camera, LoRA rank 32, 4-layer decoders of width 256.

### Branch construction (Appendix A.2, Table A.1) {#branches}

Two quantities are measured in the same training-log window:
- *Longitudinal pressure*: the negative of the lead vehicle's longitudinal acceleration (high = the lead brakes).
- *Lateral approach*: the lateral velocity of a same-direction adjacent vehicle, positive toward the ego corridor.

The windows are split at the median of longitudinal pressure, then each half at its own median of lateral approach. Each quadrant's representative is the observed pair closest to the quadrant's componentwise median, after scaling both quantities by their training-set interquartile range.

**Table A.1: four joint traffic conditions**

| Branch $l$ | Condition | Longitudinal pressure | Lateral approach |
|---:|---|---|---|
| 1 | C0 Open passage | Low | Low |
| 2 | C1 Longitudinal constraint | High | Low |
| 3 | C2 Lateral approach | Low | High |
| 4 | C3 Combined constraint | High | High |

"Starting from a shared initial state, the benchmark applies these targets smoothly under its road and dynamics constraints. An axis is active only when the relevant vehicle and a feasible motion are available." Benchmark rollout and scoring give one $y^\star_{i,l}$ per candidate and branch, which form "an equally weighted empirical outcome distribution". What the score is (PDMS, EPDMS, sub-scores) is not stated, nor which simulator or agent model applies the motions. NAVSIM replays agents from the log, so this needs code the paper does not describe. The representative values themselves are not printed.

What follows from the construction:
- The condition values are the same in every scene. Only whether an axis is active depends on the scene.
- "Low" pressure is the median of the lower half of a median split, so in C0 and C2 the lead vehicle typically accelerates away. "High" is the median of the upper half, roughly the 75th percentile. No branch represents a rare event.
- Where neither vehicle exists, all four targets equal the nominal score, $\sigma^\star=0$ and CVaR$^\star=\mu^\star$, and ranking reduces to the mean. The share of navtrain scenes in which each axis is active is not reported.

---

## Results

### Table 1: NAVSIM-v1, navtest

Columns as printed (EP before TTC). Closed form $=\mathrm{NC}\times\mathrm{DAC}\times(5\,\mathrm{TTC}+2\,\mathrm{Comf}+5\,\mathrm{EP})/12$ and residual = closed form − reported, computed here ([[concepts/navsim-benchmark.md#submetric-residual]]).

| Method | Input | Backbone | Venue (as printed) | NC | DAC | EP | TTC | Comf. | PDMS | Residual |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Human | – | – | – | 100 | 100 | 87.5 | 100 | 99.9 | 94.8 | −0.02 |
| DrivingGPT | C | VQ-VAE/GPT | ICCV 2025 | 98.9 | 90.7 | 79.7 | 94.9 | 95.6 | 82.4 | −2.85 |
| VADv2 ⚠ | C | ResNet-34 | arXiv 2024 | 97.9 | 91.7 | 77.6 | 92.9 | 100 | 83.0 | −4.26 |
| UniAD | C | ResNet-34 | CVPR 2023 | 97.8 | 91.9 | 78.8 | 92.9 | 100 | 83.4 | −4.12 |
| LTF | C | ResNet-34 | TPAMI 2022 | 97.4 | 92.8 | 79.0 | 92.4 | 100 | 83.8 | −4.18 |
| TransFuser | C & L | ResNet-34 | TPAMI 2022 | 97.7 | 92.8 | 79.2 | 92.8 | 100 | 84.0 | −3.91 |
| PARA-Drive | C | ResNet-34 | CVPR 2024 | 97.9 | 92.4 | 79.3 | 93.0 | 99.8 | 84.0 | −4.01 |
| DriveX-S | C | InternImage-Tiny | ICCV 2025 | 97.5 | 94.0 | 79.7 | 93.0 | 100 | 84.5 | −3.28 |
| LAW | C | ResNet-34 | ICLR 2025 | 96.4 | 95.4 | 81.7 | 88.7 | 99.9 | 84.6 | −3.99 |
| World4Drive | C | ResNet-34 | ICCV 2025 | 97.4 | 94.3 | 79.9 | 92.8 | 100 | 85.1 | −3.70 |
| DRAMA | C & L | ResNet-34 | arXiv 2024 | 98.0 | 93.1 | 80.1 | 94.8 | 100 | 85.5 | −3.80 |
| GoalFlow | C & L | ResNet-34 | CVPR 2025 | 98.3 | 93.8 | 79.8 | 94.3 | 100 | 85.7 | −3.45 |
| Hydra-MDP | C & L | ResNet-34 | arXiv 2024 | 98.3 | 96.0 | 78.7 | 94.6 | 100 | 86.5 | −2.63 |
| Hydra-MDP++ ⚠ | C & L | ResNet-34 | arXiv 2024 | 97.6 | 96.0 | 80.4 | 93.1 | 100 | 86.6 | −3.25 |
| ARTEMIS | C & L | ResNet-34 | arXiv 2025 | 98.3 | 95.1 | 81.4 | 94.3 | 100 | 87.0 | −2.98 |
| DiffusionDrive | C & L | ResNet-34 | CVPR 2025 | 98.2 | 96.2 | 82.2 | 94.7 | 100 | 88.1 | −2.72 |
| WoTE | C & L | ResNet-34 | ICCV 2025 | 98.5 | 96.8 | 81.9 | 94.9 | 99.9 | 88.3 | −2.18 |
| DIVER | C & L | ResNet-34 | arXiv 2025 | 98.5 | 96.5 | 82.6 | 94.9 | 100 | 88.3 | −2.16 |
| MindDrive | C & L | ResNet-34 | arXiv 2025 | 98.4 | 97.0 | 82.8 | 95.1 | 99.9 | 88.9 | −2.26 |
| MeanFuser | C | ResNet-34 | arXiv 2026 | 98.6 | 97.0 | 82.8 | 95.0 | 100 | 89.0 | −2.20 |
| AutoVLA | C | Qwen2.5-VL-3B | NeurIPS 2025 | 98.4 | 95.6 | 81.9 | 98.0 | 99.9 | 89.1 | −2.92 |
| DriveSuprim ⚠ | C | ResNet-34 | arXiv 2025 | 97.8 | 97.3 | 86.7 | 93.6 | 100 | 89.9 | −2.55 |
| DriveVLA-W0 | C | Emu3-8B | arXiv 2025 | 98.7 | 99.1 | 83.3 | 95.3 | 99.3 | 90.2 | −1.22 |
| ReCogDrive | C | InternVL3-2B | ICLR 2026 | 97.9 | 97.3 | 87.3 | 94.9 | 100 | 90.8 | −2.61 |
| iPad ⚠ | C | ResNet-34 | arXiv 2025 | 98.6 | 98.3 | 88.0 | 94.9 | 100 | 91.7 | −1.68 |
| Centaur | C | V2-99 | arXiv 2025 | 99.5 | 98.9 | 85.9 | 98.0 | 100 | 92.6 | −0.80 |
| DrivoR | C | ViT-S | CVPR 2026 | 98.9 | 98.3 | 89.1 | 96.2 | 100 | 93.1 | −1.84 |
| **RiSE-Drive** | C | ViT-S | – | **99.5** | 98.5 | 88.7 | **98.4** | 100 | **94.2** | −1.46 |

- **Every row passes** (−0.02 to −4.26). RiSE-Drive's −1.46 sits beside DrivoR's −1.84.
- **Δ against DrivoR**: NC +0.6, DAC +0.2, EP −0.4, TTC +2.2, Comf 0. RiSE-Drive's TTC (98.4) is the table's highest, above Centaur and AutoVLA (98.0).
- ⚠ labels: see the [table audit](#table-audit).

### Table 2: NAVSIM-v2 navhard two-stage

Stage closed forms ($\mathrm{NC}\cdot\mathrm{DAC}\cdot\mathrm{DDC}\cdot\mathrm{TLC}\cdot(5\,\mathrm{TTC}+5\,\mathrm{EP}+2\,\mathrm{LK}+2\,\mathrm{HC}+2\,\mathrm{EC})/16$), their product and "product − reported" computed here. On navhard, pre-fix rows sit at about −1.0 to +2.0 and corrected rows at about −1.1 to −4.5 ([[concepts/navhard-ood-evaluation.md#evaluator-fix]]).

| Method | Backbone | Stage | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | Stage CF | EPDMS | Product − reported |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LTF | ResNet-34 | 1 | 96.2 | 79.5 | 99.1 | 99.5 | 84.1 | 95.1 | 94.2 | 97.5 | 79.1 | 67.8 | 23.1 | +0.49 (pre-fix-like) |
| | | 2 | 77.7 | 70.2 | 84.2 | 98.0 | 85.1 | 75.6 | 45.4 | 95.7 | 75.9 | 34.8 | | |
| DiffusionDrive∗ | ResNet-34 | 1 | 96.0 | 79.7 | 97.4 | 99.5 | 81.3 | 93.1 | 90.8 | 96.8 | 73.8 | 64.6 | 24.2 | +1.72 (pre-fix-like) |
| | | 2 | 82.1 | 72.2 | 88.5 | 98.7 | 85.1 | 78.8 | 49.2 | 89.3 | 71.2 | 40.1 | | |
| GuideFlow | ResNet-34 | 1 | 96.6 | 80.5 | 96.3 | 99.3 | 82.3 | 94.9 | 91.5 | 97.7 | 67.8 | 65.1 | 27.1 | +2.20 (pre-fix-like) |
| | | 2 | 87.3 | 76.7 | 88.8 | 99.2 | 84.3 | 85.1 | 49.7 | 93.1 | 44.5 | 45.0 | | |
| RAP-DINO | ViT-H | 1 | 97.1 | 94.4 | 98.8 | 99.8 | 83.9 | 96.9 | 94.7 | 96.4 | 66.2 | 80.1 | 39.6 | −2.58 (corrected) |
| | | 2 | 83.2 | 83.9 | 87.4 | 98.0 | 86.9 | 80.4 | 52.3 | 95.2 | 52.4 | 46.2 | | |
| DriveSuprim | V2-99 | 1 | 98.9 | 95.1 | 99.2 | 99.6 | 76.1 | 99.1 | 94.7 | 97.6 | 54.2 | 79.5 | 42.1 | +0.65 (pre-fix-like) |
| | | 2 | 87.9 | 88.8 | 89.6 | 98.8 | 80.3 | 86.0 | 53.5 | 97.1 | 56.1 | 53.8 | | |
| GTRS-Dense | V2-99 | 1 | 98.9 | 96.2 | 99.4 | 99.3 | 72.9 | 98.9 | 95.1 | 96.9 | 39.1 | 77.5 | 45.0 | −1.09 (corrected) |
| | | 2 | 91.2 | 89.4 | 94.4 | 98.8 | 69.5 | 90.0 | 54.3 | 94.0 | 48.7 | 56.6 | | |
| DiffVLA | Vicuna-v1.5 | 1 | 95.7 | 99.2 | 100 | 100 | 85.9 | 96.4 | 97.1 | 95.0 | 84.2 | 86.9 | 45.0 | +2.28 (pre-fix-like) |
| | | 2 | 81.2 | 88.8 | 94.6 | 99.0 | 86.0 | 76.4 | 59.8 | 98.6 | 80.4 | 54.4 | | |
| GTRS-Aug ⚠ | V2-99 | 1 | 98.9 | 95.1 | 99.1 | 99.6 | 76.2 | 99.1 | 94.9 | 97.6 | 54.2 | 79.5 | 45.4 | +0.55 |
| | | 2 | 88.1 | 88.8 | 89.3 | 98.9 | **98.9** | 85.9 | 53.7 | 96.8 | 56.9 | 57.8 | | |
| ZTRS | V2-99 | 1 | 98.9 | 97.6 | 100 | 100 | 66.7 | 98.9 | 96.2 | 96.7 | 44.0 | 78.5 | 48.1 | −1.51 (corrected) |
| | | 2 | 91.1 | 90.4 | 95.8 | 99.0 | 63.6 | 89.8 | 60.4 | 97.6 | 66.1 | 59.3 | | |
| DrivoR | ViT-S | 1 | 98.8 | 95.1 | 98.9 | 100 | 72.6 | 98.7 | 94.0 | 97.6 | 73.3 | 80.5 | 48.3 | −3.23 (corrected) |
| | | 2 | 90.2 | 88.4 | 91.9 | 98.6 | 70.0 | 88.0 | 50.1 | 98.5 | 76.2 | 56.0 | | |
| **RiSE-Drive** | ViT-S | 1 | 99.0 | 96.8 | 99.3 | 100 | 72.8 | 99.0 | 94.8 | 97.7 | 73.5 | **82.7** | **51.3** | −1.27 (corrected-like) |
| | | 2 | 91.5 | 89.8 | 94.6 | 98.8 | 68.7 | 91.3 | 54.9 | 98.4 | 76.7 | **60.5** | | |

∗ "results reproduced with the official code repository or official checkpoint".

- **Δ against DrivoR.** Stage 1: DAC +1.7, LK +0.8, the rest within ±0.4; stage closed form +2.2. Stage 2: NC +1.3, DAC +1.4, DDC +2.7, TLC +0.2, EP −1.3, TTC +3.3, LK +4.8, HC −0.1, EC +0.5; stage closed form +4.5. The product rises 45.1 → 50.0 (+4.9) against +3.0 reported.
- **The paper's paragraph on this table** (from the arXiv HTML): "RiSE-Drive achieves the best EPDMS of 51.3, outperforming DrivoR by 3.0 points and the strongest previous result in the table by 3.2 points", with the gain "concentrated on Stage-II safety terms" and the EP reduction read as "more cautious decisions under perturbation". +3.2 is the margin over ZTRS (48.1), not over DrivoR (48.3), which is in the table.
- **Stage-2 lane keeping 54.9** is in the scored band (DrivoR 50.1, ZTRS 60.4), above the 45–50 band of unscored planners ([[concepts/navhard-ood-evaluation.md#lk-correction]]).

### Table 3: HUGSIM closed loop (Appendix A.1 corrects RiSE-Drive's Easy RC)

| Method | RC E | RC M | RC H | RC X | RC Avg | HDS E | HDS M | HDS H | HDS X | HDS Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VAD | 51.3 | 31.1 | 25.3 | 26.5 | 31.4 | 36.3 | 9.5 | 8.0 | 11.5 | 13.4 |
| LTF | 67.8 | 35.1 | 26.2 | 40.5 | 38.9 | 58.9 | 18.0 | 9.8 | 25.9 | 23.7 |
| UniAD | 78.4 | 60.5 | 33.6 | 17.8 | 45.9 | 64.9 | 45.8 | 20.6 | 6.6 | 32.7 |
| DrivoR | 80.9 | 50.5 | 33.8 | 47.1 | 49.8 | 73.3 | 34.6 | 18.8 | 32.5 | 35.7 |
| **RiSE-Drive** | 76.5 | 53.3 | 34.1 | 46.3 | **50.1** | 74.6 | 37.2 | 22.7 | 39.2 | **39.5** |
| *Δ vs DrivoR (computed)* | −4.4 | +2.8 | +0.3 | −0.8 | +0.3 | +1.3 | +2.6 | +3.9 | +6.7 | +3.8 |

- **Protocol, by arithmetic.** The four baseline rows are [[sources/drivor.md]]'s Table 2 digit for digit. The tier weights that fit DrivoR's and SimVTA's 345-scenario table (49 / 126 / 86 / 84, [[concepts/hugsim-benchmark.md#simvta]]) give RiSE-Drive 50.10 RC and 39.38 HD-Score against 50.1 / 39.5 printed. The paper names neither the scenario set nor the controller fix; it says only that Table 3 "evaluates fully closed-loop transfer on HUGSIM". Whether the evaluated model is the NAVSIM-v1 checkpoint is not stated.
- **The Easy-RC correction is consistent.** The appendix says the conference version misprinted Easy RC as 79.5. With 79.5 the weighted average would be 50.5; the printed 50.1 fits 76.5, so the averages were computed from the corrected value all along.
- **The shape.** HD-Score rises more with difficulty (+1.3 to +6.7) while route completion falls on Easy and Extreme. That is the same cautious profile [[sources/toad.md]] shows on HUGSIM ([[concepts/hugsim-benchmark.md#toad]]).

### Table 4: component ablation (navtest PDMS / NC; navhard EPDMS / Stage-2 TTC)

| Configuration | PDMS | NC | EPDMS | TTC | Δ PDMS | Δ EPDMS |
|---|---:|---:|---:|---:|---:|---:|
| B-I: DrivoR | 93.1 | 98.9 | 48.3 | 88.0 | | |
| B-II: DrivoR w/o SD (direct score head) | 91.9 | 98.4 | 45.6 | 84.6 | −1.2 | −2.7 |
| + FCG (future-condition generator) | 92.6 | 98.7 | 46.8 | 86.5 | +0.7 | +1.2 |
| + CTE (candidate trajectory encoder) | 92.9 | 98.8 | 47.5 | 87.3 | +0.3 | +0.7 |
| + SLE + LCP (step encoder, consequence propagation) | 93.5 | 99.1 | 49.2 | 89.6 | +0.6 | +1.7 |
| + DOE (distributional outcome estimation) | 93.9 | 99.3 | 50.6 | 90.5 | +0.4 | +1.4 |
| + RSR (risk-sensitive ranking) | **94.2** | **99.5** | **51.3** | **91.3** | +0.3 | +0.7 |

Δ columns computed here. "SD" is not expanded; the text says B-II "replaces its scoring decoder with a lightweight direct score head". Each row adds to the row above. The text does not say what an intermediate row computes, for example how FCG's branch codes are used before there is any propagation, or how candidates are ranked before RSR (by $\hat\mu$, presumably). Nor does it say whether $\mathcal L_\mathrm{rank}$ belongs to RSR. Every row is a single run on the test splits.

- From B-II the evaluator is worth +2.3 PDMS and +5.7 EPDMS; from DrivoR, +1.1 and +3.0.
- **DrivoR's scorer design is itself worth +1.2 / +2.7** over a direct score head (B-I against B-II), the navhard counterpart of DrivoR's own navval ablation (disentangled scorer +5.3 over a shared decoder).

### Table 5: parameters

| Configuration | Optimized params (M) | Total params (M) | Frozen (computed) | PDMS | EPDMS | HD-Score Avg |
|---|---:|---:|---:|---:|---:|---:|
| DrivoR | 18.8 | 41.8 | 23.0 | 93.1 | 48.3 | 35.7 |
| B-II | 17.2 | 39.9 | 22.7 | 91.9 | 45.6 | 33.8 |
| RiSE-Drive | 18.9 | 42.3 | 23.4 | 94.2 | 51.3 | 39.5 |

DrivoR's 18.8 / 41.8 are its Table 4(b)(h) values. The frozen part, the same ViT-S weights in all three, comes out at 23.0, 22.7 and 23.4M. The new evaluator costs +1.7M over B-II. No latency is reported. The evaluator runs $64\times4$ recurrences of 8 steps at width 256 per scene, which should be small next to DrivoR's 107 ms ViT ([[concepts/inference-latency.md]]).

---

## Visualizations

![[rise-teaser.png|Figure 1: Why deterministic mean scoring can fail. Two ego candidates may have similar mean scores, yet one can still hide a small set of catastrophic future-condition branches. RiSE-Drive evaluates each candidate under multiple future-condition branches, forms a distribution over branch-level long-horizon outcomes, and ranks candidates with risk-sensitive statistics such as the mean, lower quantiles, and CVaR.]]

**Figure 1**: A schematic, not data. Candidate 1 scores 3.04 on average and candidate 2 scores 2.98, so a mean scorer picks candidate 1. Below, each candidate's ten "M-quantiles" are drawn as bars. Candidate 1's lowest three are red ("Risky"), with 20% quantile 2.09 and 80% quantile 4.23: a "high lower-tail shortfall". Candidate 2's are 2.45 and 3.30 with no risky bars: "low shortfall". The x-axis labels the bars "possible risky conditions" and "moderate or safe conditions", which conflates quantiles with conditions. The method has four conditions and ten quantiles.

![[syb.png|Figure 3: Qualitative results of RiSE-Drive on representative NavSafe safety-critical scenarios. Columns (A)–(D): unprotected left turn, exit ramp, yielding, roundabout, two temporal snapshots each. Magenta: RiSE-Drive plan; green: human future trajectory; orange dashed: ego history; blue: future trajectories of surrounding vehicles.]]

**Figure 3**: Four NavSafe scenes in BEV, two snapshots each: (A) an unprotected left turn through a dense junction, (B) an exit ramp, (C) yielding at a junction with pedestrians, (D) a roundabout. In all eight panels the magenta plan lies on or next to the green human trajectory. No baseline is drawn, no failure is shown and no NavSafe number exists anywhere in the paper, so the figure shows that the plans look human-like in these scenes, not that RiSE-Drive does better than DrivoR there. NavSafe is Centaur's diagnostic split (difficult navtest cases aligned with NHTSA pre-crash categories, manually verified), not ingested.

---

## Table Audit {#table-audit}

1. **v1 passes.** 28 rows, residuals −0.02 to −4.26; RiSE-Drive −1.46.
2. **Labels in Table 1.**
   - "VADv2" 83.0 (97.9 / 91.7 / 77.6 / 92.9 / 100) is not a fourth VADv2 value. It is the row [[sources/hydra-mdp-pp.md]] marks VADv2† (camera + LiDAR) and [[sources/sgdrive.md]] and [[sources/recogdrive.md]] label Hydra-MDP-𝒱8192; [[sources/had.md]] also prints it as camera-only VADv2.
   - Hydra-MDP++ 86.6 is labelled camera + LiDAR. Its own paper's ResNet-34 row is image-only.
   - iPad 91.7 is labelled ResNet-34. [[sources/drive-jepa.md]]'s table has iPad ResNet-34 at 91.1 and ViT-L at 91.7, both on sub-scores different from this row ([[sources/drivor.md]] carries the same row unlabelled).
   - DriveSuprim is the correctly labelled ResNet-34 89.9. Its published 93.5 (ViT-L), in DrivoR's table, is not shown.
   - MindDrive (arXiv 2512.04441, "bridging world models and vision-language model", 88.9) and MeanFuser (89.0) are not ingested. A MindDrive with ORION's group appears in the wiki's Bench2Drive tables; whether it is the same paper is not established.
3. **navhard mixes evaluators.** The corrected rows are [[sources/drivor.md]]'s after-fix block (RAP-DINO, GTRS-Dense, GTRS-Aug, ZTRS, DrivoR). LTF, DiffusionDrive, GuideFlow, DriveSuprim and DiffVLA are [[sources/drivefuture.md]]'s pre-fix-like rows. On one scale those five would sit about 3 higher ([[concepts/navhard-ood-evaluation.md#evaluator-fix]]); none reaches RiSE-Drive.
4. **The "reproduced" asterisk is on a circulating row.** DiffusionDrive∗ (24.2) matches all 18 sub-scores of the row in [[sources/drivefuture.md]]'s and [[sources/idol.md]]'s tables. Either all three print one earlier reproduction or the asterisk was carried over.
5. **GTRS-Aug** is DrivoR's suspect row (Stage-2 EP 98.9 where every other row has 64–87); it also appears in [[sources/clover.md]]'s and [[sources/toad.md]]'s tables.
6. **The text's "+3.2 over the strongest previous result in the table"** measures from ZTRS (48.1); DrivoR (48.3) is in the table.
7. **HUGSIM**: baselines copied from DrivoR; averages fit the 345-scenario weighting; the Easy-RC correction agrees with the printed average.
8. **Table 5's frozen parameters** differ by up to 0.7M across three configurations that share the frozen backbone.
9. **Promised but absent**: the §5.3 sensitivity study (branch count, quantile count, $\alpha$, $\lambda$), the navtest EPDMS$_1$ results announced in §5.1, and any NavSafe numbers.
10. **"Improving candidate evaluation yields larger gains than backbone scaling alone"** (§5.2) has no backbone-scaling experiment behind it.

---

## Limitations

The paper states none; its conclusion asks for "tighter integration between proposal generation, consequence modeling, and risk-sensitive decision making".

The wiki adds:

1. **The supervision is underspecified.** The target metric, the simulator that executes the perturbed agents, the representative values and the share of scenes in which each axis is active are not given. Without them the method cannot be reproduced, and there is no code.
2. **The framing does not match the construction.** §3 and §4 describe a scene-conditional distribution of future conditions, "catastrophic" branches and tail risk. The appendix gives four fixed, equally weighted quadrant medians of two vehicle motions. The resulting criterion is a blend of mean and worst of four.
3. **Labels and architecture are confounded.** No row trains DrivoR's own scorer on the four branch labels. Its first ablation step adds the labels and the branch generator together, so the ladder cannot say whether the gain needs anything but the labels.
4. **The named mechanism is small.** Risk-sensitive ranking adds +0.3 PDMS / +0.7 EPDMS. $\lambda_\mathrm{tail}$, $\lambda_\mathrm{var}$, $\alpha$, $L$ and $M$ are never varied, so whether risk sensitivity helps at all, compared with ranking by the mean of the four branches, is untested.
5. **Single runs on the test splits** for every ablation. Steps of 0.3–0.7 PDMS are a few times the training-seed sd measured elsewhere (0.04–0.1, [[concepts/evaluation-variance.md#training-seed]]).
6. **Training split unstated.** Against DrivoR's trainval model (93.7) the v1 gain would be +0.5, the size of World4Scorer's fixed-pool scorer swap.
7. **Only two kinds of agent behaviour.** A braking lead and a merging neighbour. Pedestrians, oncoming traffic, crossing traffic at junctions and agents that react to the ego are not modelled, yet NavSafe's showcase scenes are mostly junctions.
8. **The scene enters as a mean vector**, so the evaluator cannot attend to the part of the scene a candidate passes through. The paper does not discuss this.
9. **"Long-horizon" is 4 s**, the horizon every NAVSIM scorer already uses.
10. **No latency, no oracle, no per-branch accuracy.** Whether the evaluator predicts which condition is worst for which candidate, or the selection regret against its pool, is not reported.
11. **Comparison scope** on all three benchmarks (see Key Takeaways).

---

## Relation to Other Wiki Pages

- [[sources/drivor.md]]: the base. Same encoder and generator architecture; DrivoR's rows are copied into all three tables. RiSE-Drive replaces the scorer, adding +1.1 PDMS, +3.0 navhard EPDMS and +3.8 HUGSIM HD-Score, and removes the scorer's cross-attention to the scene tokens.
- [[sources/world4scorer.md]]: the other scorer replacement on DrivoR. On DrivoR's fixed pools it gains +0.50 PDMS with outcome-supervised candidate states that cross-attend to the scene. It reports no navhard. RiSE-Drive retrains everything, so the two are not on one footing ([[concepts/selection-based-planning.md#rise-drive]]).
- [[sources/clover.md]]: retrains DrivoR's generator and scorer against the evaluator: +0.8 PDMS, +0.0 navhard. RiSE-Drive's +3.0 on navhard with labels from perturbed agents is the opposite navtest/navhard shape.
- [[sources/toad.md]]: test-time search with DrivoR's scorer adds +1.7 navhard EPDMS to DrivoR + 134k SimScale and lifts HUGSIM HD-Score by driving more cautiously, the same profile as here.
- [[sources/simscale.md]]: also adds scorer labels in states the log never visited, but by moving the ego and rendering it in 3DGS. RiSE-Drive moves other agents and renders nothing. The two have never been combined or compared.
- [[sources/da-wam.md]], [[sources/mm-future.md]], [[sources/idol.md]]: per-candidate "futures" for scoring. RiSE-Drive's per-candidate, per-branch states have no future target at all ([[concepts/world-model-for-ad.md#consequence-propagation]]).
- [[sources/resworld.md]]: another "world model" shaped only by a planning-side loss.
- [[sources/drivefuture.md]], [[sources/idol.md]]: the source of the pre-fix navhard rows and the DiffusionDrive∗ row.
- [[sources/simvta.md]]: on the same 345-scenario HUGSIM set (51.5 RC / 38.8 HD-Score), not in RiSE-Drive's table.
- [[sources/hydra-mdp-pp.md]]: the origin of the 83.0 row printed as VADv2.
- Not ingested: **Centaur** (test-time training, 92.6; NavSafe's source), **MindDrive** (2512.04441), **MeanFuser**, **GTRS / ZTRS**, **DiffVLA**, **GuideFlow**, **RAP**, **NSF-HRPT** (the authors' own safety-critical scenario assessment, cited in related work).

---

## Open Questions

- **Labels or ranking?** Train DrivoR's unchanged scorer on the four branch scores, combined by the mean, by the worst case and by RiSE-Drive's blend. That separates the perturbed-agent labels from the evaluator architecture and from risk sensitivity. Also mirrored in [[research-directions.md]] (section 5).
- **Does the evaluator need the scene?** Give it cross-attention to the register tokens, as DrivoR's scorer has. If that does not help, the evaluator is mostly learning which trajectory shapes survive a braking lead and a merging neighbour.
- **Does it predict the right branch?** Report per-branch prediction error and how often the predicted worst branch is the true worst.
- **Do risk weights work as an inference-time behaviour knob?** $\lambda_\mathrm{tail}$ and $\lambda_\mathrm{var}$ could be changed after training; the sweep is missing ([[research-directions.md]], re-weighting question).
- **Would other perturbations transfer?** Pedestrians, oncoming and crossing traffic, and reactive agents. The HUGSIM gains on Hard and Extreme are the only evidence that robustness to two scripted motions carries beyond them.
- **What is the training split, and which checkpoint is run on HUGSIM?**
