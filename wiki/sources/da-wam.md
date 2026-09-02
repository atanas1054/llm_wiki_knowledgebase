---
title: "DA-WAM: Decision-Aligned Future Latents for Driving World Models"
type: source-summary
sources: [raw/papers/DA-WAM_ Decision-Aligned Future Latents for Driving World Models.md]
related: [concepts/world-model-for-ad.md, concepts/selection-based-planning.md, concepts/navsim-benchmark.md, concepts/counterfactual-prediction.md, concepts/foundation-backbones-for-ad.md, concepts/best-of-n.md, sources/wa-jepa.md, sources/auto-jepa.md, sources/drive-jepa.md, sources/latent-wam.md, sources/geowam.md, sources/simwam.md, sources/drivelaw.md, sources/drivesuprim.md, sources/dreameraD.md]
created: 2026-09-02
updated: 2026-09-02
confidence: high
---

# DA-WAM

DA-WAM's claim is about *where* a world model's prediction enters the pipeline. Predicting the future is not enough; the prediction has to be **decision-informative** — each candidate trajectory must be scored against the future predicted *for that specific trajectory*. Existing world-model planners either decouple predictive pretraining from planner optimization, or predict one future and share it across all candidates, which "dilutes the action-specific consequences that ought to guide selection."

So DA-WAM builds a one-to-one correspondence: 32 candidates in, 32 distinct future latents out, each scored with its own. A LoRA-adapted V-JEPA 2.1 online encoder paired with an EMA target keeps JEPA supervision running *throughout* planner training rather than freezing after pretraining. Dense predictive supervision goes only to the expert-matched candidate — offline logs record one future — while safety-critical hard negatives supply contrast near planning boundaries.

It reports **93.7 PDMS on NAVSIM-v1** (tying the wiki's previous best) and 87.7 EPDMS on NAVSIM-v2.

**Source**: `raw/papers/DA-WAM_ Decision-Aligned Future Latents for Driving World Models.md`
**arXiv**: https://arxiv.org/html/2608.19085v2
**Code**: https://github.com/LeapWM/da-wam
**Authors**: Ruiguo Zhong, Benshan Ma, Xiaolong Chen, Lang Zhang, Mingyue Feng, Yaonong Wang, Pei Liu, Jun Ma — HKUST (Guangzhou), **Leapmotor**, HKUST

> **Fourth JEPA paper in the wiki**, after [[sources/drive-jepa.md]], [[sources/auto-jepa.md]], and [[sources/wa-jepa.md]] — and the first to cite two of the others. See [Four JEPA Papers](#four-jepa-papers-compared).

## Key Takeaways

- **The wiki's first controlled evidence that inference-time future conditioning can help — and a mechanism for why it usually doesn't.** Table 3 holds everything fixed and varies only how the future is used: no future 93.31, **shared global future 92.81**, current-latent 93.25, action-conditioned future 93.46, + hard negatives 93.68. A future *shared* across candidates is **worse than predicting no future at all**.
- **The magnitudes are small and worth stating plainly.** Action-conditioned future prediction buys **+0.15 PDMS** over the no-future baseline. Hard negatives — which have nothing to do with world modeling — buy another +0.22. Single run, no seed variance.
- **The no-future baseline is already 93.31 PDMS**, which would rank third in this wiki on its own. Most of what makes DA-WAM strong is its planner and scorer, not its world model.
- **LoRA beats full fine-tuning for JEPA adaptation** (92.98 vs. 92.62), and an EMA target beats frozen/separate/shared (93.68 / 92.98 / 93.10 / 93.34). This **contradicts [[sources/latent-wam.md]]**, where LoRA collapsed geometric distillation from 89.3 to 68.5 EPDMS — the two papers reach opposite conclusions about LoRA for two different distillation targets.
- **The predicted future horizon is 0.5 seconds** while the trajectory is 8 poses. The paper never discusses how a half-second latent is supposed to encode the consequences of a multi-second maneuver.
- **31 of 32 predicted futures receive no feature-level supervision.** Only the expert-matched candidate gets $\mathcal{L}_\mathrm{pred}$; the rest are shaped solely by scorer gradients, and no diagnostic shows whether they are meaningful futures or just conditioning features.
- Introduces **V-JEPA 2.1** to the wiki — a dense-feature variant whose objective is worth +0.69 PDMS frozen and +0.24 under LoRA.

## Method

### The Taxonomy It Argues Against

Figure 1 sets up four designs, and the argument is structural rather than empirical:

| Design | Future representation reaches the scorer? | Per-candidate? |
|---|---|---|
| (a) Trajectory-only prediction | No | – |
| (b) Loosely coupled latent fusion | Yes | No — single proposal, so no candidate comparison |
| (c) One future shared across candidates | Yes | **No — prediction–action mismatch** |
| (d) **DA-WAM** | Yes | **Yes — one latent per candidate** |

The paper's diagnosis of (c) is the interesting part: with a shared future, "the scorer may therefore rely primarily on geometric cues rather than the scene-conditioned future content that distinguishes safe from unsafe outcomes." Table 3 later measures exactly this failure.

### Observation Encoding with Live JEPA Supervision

The online encoder is **V-JEPA 2.1** with **LoRA** injected into selected transformer layers; the base network stays frozen while LoRA parameters receive gradients from *both* future prediction and trajectory planning:

$$
Z_{t}=E_{\theta}(X_{t}),\qquad Z_{t}\in\mathbb{R}^{M\times D}
$$

During training only, the observed future frame goes through an EMA target with stop-gradient:

$$
Z_{t+\Delta}=\operatorname{sg}\left(E_{\bar{\theta}}(X_{t+\Delta})\right),\qquad \bar{\theta}\leftarrow\mu\bar{\theta}+(1-\mu)\theta
$$

**This is the paper's first structural claim**: predictive supervision continues *during* planner optimization rather than stopping after pretraining, so the latent space can adapt to the scoring objective. Compare Drive-JEPA (pretrain, then freeze and train a planner), Auto-JEPA (frozen encoder throughout), Latent-WAM (EMA latent target but the dynamics branch is discarded at test time), and WA-JEPA (two stages, but Stage 2 does keep the future objective live).

### Action-Conditioned Future Prediction

Each candidate is encoded to an action representation and used as a **query** into the scene tokens:

$$
a_{i}=E_{\tau}(\tau_{i}),\qquad
\widehat{Z}_{i}=P_{\phi}\left(Q=a_{i},\;K=Z_{t},\;V=Z_{t}\right),\quad i=1,\ldots,N
$$

The predictor is **shared across all candidates** — deliberately, so that differences among the $\widehat Z_i$ come from the action queries rather than from candidate-specific parameters. A single observation therefore yields $N$ distinct latent futures.

**Expert matching.** Offline logs record only the executed future, so dense supervision is restricted to the candidate closest to the expert:

$$
i^{\mathrm{exp}}=\arg\min_{i}\operatorname{ADE}\left(\tau_{i},\tau^{\mathrm{exp}}\right),\qquad
\mathcal{L}_{\mathrm{pred}}=\frac{1}{M}\sum_{m=1}^{M}\ell\left(\widehat{Z}_{i^{\mathrm{exp}},m},Z_{t+\Delta,m}\right)
$$

The remaining $N-1$ latents "cannot receive direct feature-level supervision because their corresponding outcomes are unobserved" and are optimized only through downstream scoring losses. This is the honest handling of the counterfactual data problem — applying the observed future to unexecuted actions would be plainly wrong — but it leaves 31 of 32 predicted futures without a target. See [Limitations](#limitations).

### Future-Latent-Conditioned Scoring

A scoring transformer cross-attends the scene tokens, the action representation, and **that candidate's own** predicted future:

$$
h_{i}=S_{\psi}^{\mathrm{enc}}\left(Z_{t},\widehat{Z}_{i},a_{i}\right)
$$

The encoder "preserves fine-grained token-level interactions rather than pooling futures into a coarse proposal-invariant vector" — the pooling step being precisely what design (c) does wrong. Parameters are shared across candidates, so score differences come from geometry and predicted outcome, not from per-candidate weights.

**Factorized heads** decode interpretable planning factors before the utility score:

$$
\widehat{\mathbf{q}}_{i}=\left[\widehat{q}_{i}^{\mathrm{NC}},\widehat{q}_{i}^{\mathrm{DAC}},\widehat{q}_{i}^{\mathrm{EP}},\widehat{q}_{i}^{\mathrm{TTC}},\widehat{q}_{i}^{\mathrm{Comfort}}\right],\qquad
\widehat{s}_{i}=S_{\psi}^{\mathrm{score}}\left(h_{i},\widehat{\mathbf{q}}_{i}\right)
$$

Each factor is supervised by simulation-derived or rule-based metrics — Hydra-MDP-style distillation, and the same privileged-supervision caveat that applies to [[sources/auto-jepa.md]]'s CLOVER scorer.

### Safety-Critical Hard Negatives

The scorer's failure mode without them: "randomly sampled candidate sets often exhibit large geometric differences, allowing the scorer to rely on coarse cues such as curvature and speed rather than scene-dependent safety consequences." Negatives are retrieved from an offline trajectory bank under dual constraints — geometrically close to the expert, but substantially worse in safety:

$$
d_{\mathrm{traj}}(\tau_{j}^{-},\tau^{\mathrm{exp}})<\epsilon_{\mathrm{geo}},\qquad
\Delta_{\mathrm{safety}}(\tau_{j}^{-},\tau^{\mathrm{exp}})>\epsilon_{\mathrm{safety}}
$$

Each $\tau^-_j$ is appended to the candidate set, gets its own future latent, and enters the same shared scorer — but is excluded from expert matching and from $\mathcal{L}_\mathrm{pred}$, since its visual future is unobserved. It still receives factor, utility, and ranking targets.

This is the same insight as DriveSuprim's hard-negative analysis ([[concepts/selection-based-planning.md]]), reached by a different route: DriveSuprim concentrates hard negatives by coarse-to-fine filtering of a fixed vocabulary; DA-WAM *retrieves* them under an explicit geometric-proximity + safety-divergence constraint.

### Objectives

$$
\mathcal{L}_{\mathrm{factor}}=\sum_{i}\sum_{k\in\mathcal{K}}\lambda_{k}\ell_{k}\left(\widehat{q}_{i}^{k},q_{i}^{k}\right),\qquad
\mathcal{L}_{\mathrm{score}}=\sum_{i}\ell_{\mathrm{score}}\left(\widehat{s}_{i},s_{i}\right)
$$

$$
\mathcal{L}_{\mathrm{rank}}=-\sum_{(i,j)}\left[y_{ij}\log\sigma\left(\widehat{s}_{i}-\widehat{s}_{j}\right)+(1-y_{ij})\log\sigma\left(\widehat{s}_{j}-\widehat{s}_{i}\right)\right],\qquad y_{ij}=\mathbb{I}[s_i>s_j]
$$

$$
\mathcal{L}=\lambda_{\mathrm{pred}}\mathcal{L}_{\mathrm{pred}}+\lambda_{\mathrm{factor}}\mathcal{L}_{\mathrm{factor}}+\lambda_{\mathrm{score}}\mathcal{L}_{\mathrm{score}}+\lambda_{\mathrm{rank}}\mathcal{L}_{\mathrm{rank}}
$$

Pairs involving hard negatives are oversampled or upweighted. $\mathcal{L}_\mathrm{pred}$ applies only to the expert-matched candidate; all candidates including hard negatives contribute to the other three.

### Inference

Only the online encoder, predictor, and scorer run. No future observations, no expert priors, no EMA target. Each candidate is evaluated with its own predicted future, and $\tau^\star=\arg\max_i \widehat s_i$.

**So the future latents are generated at inference and are load-bearing** — this is an imagine-then-act design, and the one that finally produces positive controlled evidence for that camp. See [[concepts/world-model-for-ad.md#test-time-imagination]].

### Implementation

| Setting | Value |
|---|---|
| Input | **2 historical frames, front camera only** |
| Candidates | 32, each 8 future ego poses |
| **Prediction horizon** | **0.5 s into the future** |
| Encoder | V-JEPA 2.1, LoRA-adapted online + EMA target |
| Training | 20 epochs, 8 GPUs, batch 8/GPU |
| Checkpoint | selected by validation performance |
| Unreported | $\mu$, all $\lambda$, $\epsilon_\mathrm{geo}$, $\epsilon_\mathrm{safety}$, $M$, $D$, latency, params |

## Figures

![[pipeline_compare.png]]

**Figure 1.** Prediction–action alignment in trajectory scoring. (a) trajectory-only prediction gives the scorer no explicit future representation; (b) loosely coupled latent fusion adds a future but generates only one proposal, precluding candidate-specific comparison; (c) sharing one future latent across candidates creates a prediction–action mismatch; (d) DA-WAM predicts a distinct future per candidate and scores each with its own, establishing one-to-one correspondence.

![[overview2.png]]

**Figure 2.** Architecture. The online encoder maps $X_t$ to scene tokens $Z_t$; each candidate's action representation $a_i$ is combined with them by predictor $P_\phi$ to forecast a candidate-specific $\widehat Z_i$; a shared scorer evaluates $(Z_t,a_i,\widehat Z_i)$ to predict planning factors and a utility score. **Training** adds an EMA target encoder extracting $Z_{t+\Delta}$ from the observed future frame, supervising only the expert-matched prediction, plus hard negatives for boundary discrimination. **Inference** activates only the online encoder, predictor, and scorer.

![[counterfactual_trajectory_supervision2.png]]

**Figure 3.** Safety-critical hard-negative supervision. Conventional training scores a sparse candidate set with rule-based NC/DAC/TTC factors. DA-WAM additionally retrieves expert-proximate hard negatives — geometrically similar to the expert but different in safety outcome. Generated candidates and hard negatives query the same scene representation and share one future-latent-conditioned scorer. Hard-negative labels are training-only planning targets, never observed future representations.

![[camera_bev_score_comparison_32.png]]

**Figure 4.** Trajectory selection across (a) a large left turn, (b) tight traffic, and (c) a yielding conflict, showing camera views, BEV trajectories, and per-scene metric scores. DiffusionDrive is blue, DrivoR orange, DA-WAM green, expert dashed. In (a) all methods are collision-free but DA-WAM tracks the expert more closely with the highest EP and PDMS; in (b) and (c) both baselines incur NC and TTC failures that DA-WAM avoids. Three hand-picked scenes; no failure cases shown.

## Tables

### Table 1: NAVSIM-v1 navtest (camera-only)

| Method | Venue | NC | DAC | TTC | Comfort | EP | PDMS |
|---|---|---:|---:|---:|---:|---:|---:|
| PDM-Closed | CoRL'23 | 94.6 | 99.8 | 89.9 | 86.9 | 99.9 | 89.1 |
| **Human driver** | NeurIPS'24 | 100.0 | 100.0 | 100.0 | 99.9 | 87.5 | **94.8** |
| Ego-stat. MLP | NeurIPS'24 | 93.0 | 77.3 | 83.6 | 100.0 | 62.8 | 65.6 |
| UniVLA | ICLR'26 | 96.9 | 91.1 | 91.7 | 96.7 | 76.8 | 81.7 |
| DrivingGPT | ICCV'25 | 98.9 | 90.7 | 94.9 | 95.6 | 79.7 | 82.4 |
| UniAD | CVPR'23 | 97.8 | 91.9 | 92.9 | 100.0 | 78.8 | 83.4 |
| DriveX-S | ICCV'25 | 97.5 | 94.0 | 93.0 | 100.0 | 79.7 | 84.5 |
| World4Drive | ICCV'25 | 97.4 | 94.3 | 92.8 | 100.0 | 79.9 | 85.1 |
| VAD-v2 | ICLR'26 | 98.1 | 94.8 | 94.3 | 100.0 | 80.6 | 86.2 |
| PRIX | RA-L'26 | 98.1 | 96.3 | 94.1 | 100.0 | 82.3 | 87.8 |
| DiffusionDrive | CVPR'25 | 98.2 | 96.2 | 94.7 | 100.0 | 82.2 | 88.1 |
| DIVER | TPAMI'26 | 98.5 | 96.5 | 94.9 | 100.0 | 82.6 | 88.3 |
| AutoVLA | NeurIPS'25 | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 |
| DriveVLA-W0 | ICLR'26 | 98.7 | **99.1** | 95.3 | 99.3 | 83.3 | 90.2 |
| ReCogDrive | ICLR'26 | 97.9 | 97.3 | 94.9 | 100.0 | 87.3 | 90.8 |
| Hydra-MDP++ | arXiv'25 | 98.6 | 98.6 | 95.1 | 100.0 | 85.7 | 91.0 |
| DiffusionDriveV2 | arXiv'25 | 98.3 | 97.9 | 94.8 | 99.9 | 87.5 | 91.2 |
| iPad | arXiv'25 | 98.6 | 98.3 | 94.9 | 100.0 | 88.0 | 91.7 |
| SparseDriveV2 | arXiv'26 | 98.5 | 98.4 | 95.0 | 99.9 | 88.6 | 92.0 |
| Centaur | arXiv'25 | **99.5** | 98.9 | **98.0** | 100.0 | 85.9 | 92.6 |
| DrivoR | CVPR'26 | 98.9 | 98.3 | 96.2 | 100.0 | 89.1 | 93.1 |
| DriveSuprim | AAAI'26 | 98.6 | 98.6 | 95.5 | 100.0 | **91.3** | 93.5 |
| **DA-WAM** | – | 99.1 | 98.9 | 96.8 | 99.8 | 90.0 | **93.7** |

This is one of the better-populated NAVSIM-v1 tables in the wiki — it includes DrivoR, Centaur, SparseDriveV2, iPad, and DIVER, none of which are ingested. It still omits CLEAR (93.7), Drive-JEPA (93.3), HybridDriveVLA (92.1), WA-JEPA (91.8), DynVLA (91.7), and SimWAM (91.5).

**93.7 ties [[sources/clear.md]] for the highest non-BoN NAVSIM-v1 result in the wiki**, with a different balance: DA-WAM has higher NC (99.1 vs. CLEAR's 99.1 — equal) and TTC 96.8, while DriveSuprim retains the best EP at 91.3.

### Table 2: NAVSIM-v2 navtest

| Method | Backbone | NC | DAC | DDC | TL | EP | TTC | LK | HC | EC | EPDMS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ego Status MLP | ResNet-34 | 93.1 | 77.9 | 92.7 | 99.6 | 86.0 | 91.5 | 89.4 | 98.3 | 85.4 | 64.0 |
| TransFuser | ResNet-34 | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| Hydra-MDP++ | ResNet-34 | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| DriveSuprim | ResNet-34 | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| ARTEMIS | ResNet-34 | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | 98.3 * | 83.1 |
| DiffusionDriveV2 | ResNet-34 | 97.7 | 96.6 | 99.2 | 99.8 | 88.9 | 97.2 | 96.0 | 97.8 | **91.0** | 87.5 |
| SparseDriveV2 | ResNet-34 | 98.1 | 98.1 | 99.6 | 99.8 | **91.1** | 97.3 | 96.9 | 98.2 | 78.4 | 86.7 |
| Hydra-MDP++ | ViT/L | 98.4 | 98.0 | 99.4 | 99.8 | 87.5 | 97.7 | 95.3 | 98.3 | 77.4 | 85.1 |
| DriveSuprim | ViT/L | 97.8 | 97.9 | 99.5 | 99.9 | 90.6 | 97.1 | 96.6 | 98.3 | 77.9 | 86.0 |
| **DA-WAM** | ViT/L | 98.4 | 98.4 | 99.1 | 99.9 | 88.6 | **97.9** | **97.6** | 97.8 | 79.6 | **87.7** |

\* ARTEMIS's EC of 98.3 duplicates its HC and is reported as "–" in both [[sources/wa-jepa.md]] and [[sources/geowam.md]]; likely a transcription error.

**Protocol note.** This table is another internally mixed one, in the pattern [[sources/geowam.md]] exposed. TransFuser 76.7, SparseDriveV2 86.7, DriveSuprim ViT/L 86.0, and ARTEMIS 83.1 all match WA-JEPA's **pre-fix (EPDMS\*)** column; DiffusionDriveV2 87.5 matches WA-JEPA's **corrected** column (its pre-fix value is 85.5). The paper's claim of "exceeding the strongest comparison by 0.2 points" is measured against that one corrected number. Which convention DA-WAM's own 87.7 belongs to is undeterminable. See [[concepts/navsim-benchmark.md]].

On the useful side, DA-WAM's TransFuser row **agrees with WA-JEPA and Drive-JEPA at 76.7**, making the tally three papers at 76.7 against GeoWAM's 84.0 from identical submetrics.

### Table 3: Future-Prediction Configuration (matched ablation, NAVSIM-v1)

Training data, initialization, proposal generator, optimization schedule, checkpoint rule, and evaluation protocol are held fixed across rows.

| Configuration | Hard neg. | PDMS | NC | DAC | EP | TTC | Comfort |
|---|---|---:|---:|---:|---:|---:|---:|
| No Future Prediction | – | 93.31 | 98.45 | 98.27 | **91.36** | 95.48 | 99.99 |
| **Shared Global Future** | – | **92.81** | 99.02 | 98.46 | 88.68 | 96.54 | 99.99 |
| Current-Latent Conditioning | – | 93.25 | 98.44 | 98.19 | 91.38 | 95.49 | 99.94 |
| Action-Conditioned Future | ✗ | 93.46 | 98.88 | 98.58 | 90.47 | 96.33 | 99.69 |
| Action-Conditioned Future | ✓ | **93.68** | **99.11** | **98.88** | 89.97 | **96.81** | 99.77 |

**This is the wiki's most directly relevant ablation on the test-time-imagination question, and it deserves careful reading.**

- **A shared future is worse than no future** (92.81 vs. 93.31). The mechanism is visible in the submetrics: NC and TTC *improve* (99.02, 96.54) while EP collapses from 91.36 to 88.68. A future averaged over candidates makes the scorer uniformly cautious rather than discriminative — it cannot tell which candidate causes the hazard, so it penalizes progress everywhere.
- **A parallel pathway alone does nothing.** Current-latent conditioning (93.25) is statistically indistinguishable from no future (93.31), ruling out "extra capacity" as the explanation for any gain.
- **Action conditioning recovers and slightly exceeds the baseline**: +0.15 over no-future, +0.65 over shared, +0.21 over current-latent.
- **Hard negatives contribute more than the world model does**: +0.22 vs. +0.15. And they trade EP (90.47 → 89.97) for NC/DAC/TTC, the same safety-for-progress exchange the shared-future row makes more crudely.

The honest summary is that **the paper's headline mechanism is worth 0.15 PDMS on a single run**, and its more robust finding is the negative one: the configuration most world-model planners actually use costs 0.5 PDMS relative to not modeling the future at all.

### Table 4: Predictive-Representation Ablation (NAVSIM-v1)

"Dense loss ✓" = the V-JEPA **2.1** dense latent objective; ✗ = the V-JEPA 2.0 objective.

| Adaptation | Dense loss | Target | PDMS |
|---|---|---|---:|
| *Online-encoder adaptation and predictive objective* | | | |
| Frozen | ✗ | Frozen | 91.26 |
| Frozen | ✓ | Frozen | 91.95 |
| LoRA | ✗ | Frozen | 92.74 |
| LoRA | ✓ | Frozen | 92.98 |
| Full ft. | ✓ | Frozen | 92.62 |
| *Target-encoder policy (LoRA + dense loss)* | | | |
| LoRA | ✓ | Separate | 93.10 |
| LoRA | ✓ | Shared | 93.34 |
| LoRA | ✓ | **EMA** | **93.68** |

Three clean readings. The **2.1 dense objective** is worth +0.69 frozen and +0.24 under LoRA. **LoRA beats full fine-tuning by 0.36** — consistent with the usual story that full fine-tuning destroys a pretrained predictive representation. And the **EMA target beats every alternative**, +0.70 over a frozen target and +0.34 over a shared one, which is the cleanest isolation of the EMA mechanism in the wiki.

Cumulatively the representation choices are worth **+2.42 PDMS** (91.26 → 93.68) — an order of magnitude more than the action-conditioning mechanism the paper is named for.

### Table 5: Candidate Count

| Candidates | 1 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|
| PDMS | 87.11 | 90.76 | 91.89 | **93.68** | 93.68 |

Saturates exactly at 32. Worth comparing to [[sources/auto-jepa.md]], which needs $K=300$ retrieved candidates to reach 91.3 and saturates between 200 and 300 — a generated 32-candidate set outperforms a retrieved 300-candidate one here, though the scorers differ entirely.

## Four JEPA Papers Compared

| | [[sources/drive-jepa.md]] | [[sources/auto-jepa.md]] | [[sources/wa-jepa.md]] | **DA-WAM** |
|---|---|---|---|---|
| Backbone | V-JEPA 2 → re-pretrained | V-JEPA 2, **frozen** | V-JEPA 2 → re-pretrained | **V-JEPA 2.1 + LoRA** |
| JEPA target | Masked video latents | Frozen trajectory-latent encoding | Future multi-view scene latents (EMA) | Future scene latents (EMA) |
| Objective | L1 regression | Alignment + cosine + InfoNCE | **Flow matching** | Feature regression |
| JEPA live during planner training? | No | No | Yes (Stage 2) | **Yes** |
| Future is per-candidate? | – | – | **No — one future** | **Yes — one per candidate** |
| Future used at inference? | No | Predicts an *action* latent | Yes | **Yes, as scorer conditioning** |
| Trajectory source | 32 refined proposals | Retrieval from 110k memory | Flow-matched continuous | 32 generated candidates + scorer |
| NAVSIM-v1 | 93.3 | 91.3 | 91.8 | **93.7** |
| NAVSIM-v2 | 87.8* | 85.6* / 89.1 | 88.0* / 91.7 | 87.7 (convention unclear) |

**DA-WAM is the only one whose future prediction is candidate-specific**, and that is precisely the axis it argues everyone else gets wrong. Its Table 3 result — shared future worse than no future — is a direct, if small, criticism of WA-JEPA's design, which generates one future stream alongside one action. Neither paper cites the other.

## Relationships

- **[[sources/simwam.md]] / [[sources/drivelaw.md]]** — the two papers that established no benefit from inference-time future conditioning. **DA-WAM does not overturn them; it explains them.** SimWAM's isolated-mask ablation removed access to a *single shared* future stream and lost nothing; DA-WAM measures that exact configuration as **actively harmful** (92.81 vs. 93.31). Both papers are consistent with a sharper claim: *shared* futures are useless-to-harmful, and only per-candidate futures help. That said, DA-WAM's positive effect is +0.15 PDMS on a single run, so the reframing is better supported in its negative half than its positive half.
- **[[sources/latent-wam.md]]** — the direct contradiction on LoRA. Latent-WAM found Base-LoRA collapsed geometric distillation from 89.3 to 68.5 EPDMS and concluded low-rank adaptation is "too restrictive" for aligning high-dimensional spatial features. DA-WAM finds LoRA *beats* full fine-tuning by 0.36 PDMS for JEPA latent adaptation. The targets differ — WorldMirror geometric features vs. EMA video latents — and the plausible reconciliation is that geometric distillation demands large representational movement while JEPA adaptation mainly needs to *avoid destroying* a pretrained predictive prior. Neither paper tests the other's setting. See [[concepts/foundation-backbones-for-ad.md]].
- **[[sources/drivesuprim.md]]** — the same hard-negative insight from the opposite direction. DriveSuprim concentrates hard negatives by coarse-to-fine filtering of an 8192 vocabulary and reports that only filtering helps (+0.8 EPDMS) while extra decoder depth does nothing. DA-WAM *retrieves* hard negatives under explicit geometric-proximity and safety-divergence constraints and measures +0.22 PDMS. Both diagnose the identical failure: a scorer trained on geometrically diverse candidates learns curvature and speed rather than scene-conditioned safety.
- **[[sources/auto-jepa.md]]** — DA-WAM cites it accurately ("predicts continuous intent embeddings with a frozen visual backbone to rank candidate paths") and inverts two of its choices: frozen encoder → LoRA-adapted with live JEPA supervision, and one intent latent as retrieval key → one future latent per candidate as scorer conditioning. Both end up as scoring-based planners with simulator-derived factor supervision.
- **[[sources/dreameraD.md]]** — the closest existing "learned scorer over candidates using world-model features." DreamerAD scores 256 Gaussian-sampled vocabulary candidates with a latent reward model trained on video features; DA-WAM scores 32 generated candidates with a factorized head conditioned on per-candidate future latents. DreamerAD's reward model sees one latent world state; DA-WAM's sees a different one per candidate — the same distinction Figure 1 draws.
- **[[sources/geowam.md]]** — GeoWAM's action head is deterministic and unimodal; DA-WAM's is a scorer over 32 candidates. Both report that their world-modeling addition is worth well under one point on navtest (+0.6 and +0.15 respectively), which is a pattern worth watching.
- **Nine un-ingested methods** appear at 91.0+ PDMS in Table 1: **DrivoR** (93.1, also GeoWAM's HUGSIM baseline), **Centaur** (92.6), **SparseDriveV2** (92.0), **iPad** (91.7), **Hydra-MDP++** (91.0), plus DIVER, PRIX, DriveX-S, UniVLA. **DriveFuture**, **IDOL**, **LCDrive**, **BeyondDrive**, **GTRS**, and **ZTRS** are cited as the closest related work and none is ingested.

## Limitations

**The central mechanism is small**

- **Action-conditioned future prediction is worth +0.15 PDMS** over a no-future baseline that already scores 93.31. Hard negatives, which are not a world-model contribution at all, add +0.22. The representation choices in Table 4 are worth +2.42. The paper's title mechanism is the smallest measured effect in it.
- **Single run, no seed variance.** [[sources/wa-jepa.md]] measured seed std of 0.053 EPDMS for a stochastic sampler; training-seed variance for a scorer is typically larger. A 0.15-point gap is not demonstrated to exceed noise, and the paper does not claim otherwise — but the abstract's "state-of-the-art performance" and the framing throughout rest on it.
- The most robust result is the **negative** one: shared-future conditioning costs 0.50 PDMS versus no future at all. That is a 3× larger effect than the positive finding and it indicts a design many published WAMs use.

**The future being predicted is very short**

- **The predictor forecasts 0.5 seconds ahead** while each candidate spans 8 future ego poses. Action-specific consequences — collisions, lane departures, traffic-rule violations, exactly what the introduction promises the scorer will exploit — mostly occur well beyond 0.5 s. The paper never discusses the horizon choice, never ablates it, and never explains how a half-second latent distinguishes candidates whose outcomes diverge at 2–4 s.
- Only **2 historical frames from a single front camera**. Compare WA-JEPA's 4 frames × 4 cameras and GeoWAM's 3 frames × 8 cameras.

**31 of 32 futures are unsupervised**

- Dense predictive supervision reaches only the expert-matched candidate. The paper is right that applying the observed future to unexecuted actions would be wrong, but the consequence is that **the large majority of predicted "futures" have no feature-level target** and are shaped entirely by scorer gradients. Nothing shows they are futures rather than convenient conditioning features — no decoding, no divergence statistics across candidates, no check that a hard-braking candidate's latent differs from a full-throttle candidate's in the way physics requires. WA-JEPA's temporal-collapse diagnostics are exactly the right tool and are not applied.
- The paper calls these "counterfactual latent futures" and "candidate-specific counterfactual evidence" throughout. Per [[concepts/counterfactual-prediction.md]], action-conditioned prediction without abduction is **rung 2 (interventional), not counterfactual** — and here the situation is the strict one the wiki flagged: the query action is specified rather than observed, and no factual continuation conditions the prediction. The computation is appropriate for planning; the terminology is not.

**Protocol and comparison**

- The NAVSIM-v2 table mixes conventions — TransFuser, SparseDriveV2, DriveSuprim, and ARTEMIS at WA-JEPA's pre-fix values, DiffusionDriveV2 at its corrected value — and the "+0.2 over the strongest comparison" claim is measured against the one corrected number. DA-WAM's own 87.7 cannot be assigned to either convention.
- ARTEMIS's EC of 98.3 duplicates its HC and is "–" in two other papers' tables; a transcription error carried into a published comparison.
- The NAVSIM-v1 table omits CLEAR 93.7, Drive-JEPA 93.3, HybridDriveVLA 92.1, WA-JEPA 91.8, DynVLA 91.7, and SimWAM 91.5. It is nonetheless better populated than most, including five methods the wiki has not ingested.

**Missing specification**

- **The EMA momentum $\mu$ is never given**, despite the EMA target being worth +0.70 PDMS and being the single largest isolated design gain in the paper. Nor are $\lambda_\mathrm{pred}$, $\lambda_\mathrm{factor}$, $\lambda_\mathrm{score}$, $\lambda_\mathrm{rank}$, the per-factor $\lambda_k$, $\epsilon_\mathrm{geo}$, $\epsilon_\mathrm{safety}$, the LoRA rank, which layers receive LoRA, $M$, or $D$.
- **No latency, FPS, or parameter count**, despite running the predictor 32 times per frame plus a token-level scorer over each candidate's full future latent. This is the paper's most obvious deployment question and it is unaddressed.
- Hard negatives require an offline trajectory bank and safety-metric evaluation; the bank's size and construction are not described.

**Scope**

- **NAVSIM only** — no navhard, HUGSIM, Bench2Drive, nuScenes, or Waymo. For a paper whose thesis is that future prediction sharpens safety-critical discrimination, the absence of any reactive or OOD evaluation is a notable gap; [[concepts/navhard-ood-evaluation.md]] is where that claim would be tested.
- Factor heads are supervised by simulation-derived metrics, so "no perception annotations" would be true but "no privileged supervision" would not — the same distinction that applies to Auto-JEPA's CLOVER-initialized scorer and every Hydra-MDP-style distillation method.
