---
title: "DriveFuture: Future-Aware Latent World Models for Autonomous Driving"
type: source-summary
sources: [raw/papers/DriveFuture_ Future-Aware Latent World Models for Autonomous Driving.md]
related: [concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/navhard-ood-evaluation.md, concepts/diffusion-planner.md, concepts/intent-conditioned-planning.md, concepts/selection-based-planning.md, concepts/best-of-n.md, concepts/foundation-backbones-for-ad.md, sources/wa-jepa.md, sources/da-wam.md, sources/drivelaw.md, sources/geowam.md, sources/geoworldad.md, sources/latent-wam.md, sources/adaptive-wam.md, sources/reworld.md, sources/drivesuprim.md, sources/spanvla.md, sources/drivefine.md, sources/auto-jepa.md, sources/drivevla-w0.md, sources/diffusiondrive.md, sources/diffusiondrive-v2.md, sources/dial.md, sources/elf-vla.md, sources/lwdrive.md, sources/simwam.md]
created: 2026-09-11
updated: 2026-09-11
confidence: high
---

**Paper**: DriveFuture: Future-Aware Latent World Models for Autonomous Driving
**Authors**: Yufeng Hong, Xiaotian Zhou, Yingyan Li (equal contribution); Xiangpo Zhou, Lin Liu, Yadan Luo, Shaoqing Xu, Lei Yang, Ziying Song (corresponding)
**Orgs**: Institute of Automation, Chinese Academy of Sciences · Beihang · Beijing Jiaotong · University of Queensland · University of Macau · NTU · Yanshan University / Beijing Institute of Technology
**arXiv**: 2605.09701v1
**Code**: not released

---

## Summary

DriveFuture's thesis is a role reversal, not a new predictor. Every latent world model in this wiki treats the predicted future latent as something to be **produced** — a regression target ([[sources/latent-wam.md]], DeepSight, FLARE), a flow-matching target ([[sources/wa-jepa.md]]), a generated video state ([[sources/drivelaw.md]]), a geometric forecast ([[sources/geowam.md]]). DriveFuture treats it as something the planner **reads**: the future latent is an explicit conditioning context injected into every denoising step of a trajectory DiT, and its only training signal is whether it improves trajectory denoising.

The architecture is small and the pipeline is a TransFuser stack:

| Module | What it does | Size |
|---|---|---|
| Perception encoder $\phi_{\mathrm{enc}}$ | V2-99 → $16{\times}64$ BEV anchors → 64 BEV tokens + 1 ego token | $\mathbf{Z}_t\in\mathbb{R}^{65\times256}$ |
| Latent Dynamics Predictor $f_\psi$ | 4 decoder layers, 16 learnable future queries, cross-attending $[\mathbf{Z}_t\,\|\,\mathbf{E}_\tau]$ | $\hat{\mathbf{Z}}_{t+T}\in\mathbb{R}^{16\times256}$ |
| Future Alignment Adapter | **Training only.** One cross-attention: $\hat{\mathbf{Z}}_{t+T}$ queries the stop-gradient **GT future BEV bank** $\mathbf{Z}_{t+T}$ | 1 MHA layer, no residual |
| LatentAlign | Sigmoid anneal from the grounded latent to the self-predicted one | schedule only |
| Planning Decoder | 5-layer DiT, DDPM, cross-attends scene then future, zero-init on the future projection | 100 proposals |

Headlines: **55.5 EPDMS on NAVSIM-v2 navhard (1st on the public leaderboard as of April 2026), 89.9 corrected EPDMS on NAVSIM-v2 navtest, 90.7 PDMS on NAVSIM-v1 navtest.**

**Three readings this page arrives at, in descending order of how much they change the wiki:**

1. **The 55.5 is 34.6 of model and 20.9 of scorer.** The ablations are run "without GTRS-Dense scorer" and the best ablation row is **34.6**; Table 7 reports the unscored model's stage-wise submetrics and *they match that row digit-for-digit on all six shared columns*. So the full unscored system scores 34.6 and the GTRS-Dense scorer is worth **+20.9 EPDMS** — 5.6× the entire measured contribution of future-frame grounding (+3.7). See [The Decomposition](#decomposition).
2. **Three of the nine swept hyper-parameter settings score *below* the no-future-frame baseline.** The annealing inflection alone swings 5.7 EPDMS across a $\pm0.12$ range. The mechanism helps inside a narrow band and is net harmful outside it — and the paper reports this without remarking on it.
3. **It is the wiki's second paper to tabulate both EPDMS columns** (after [[sources/wa-jepa.md]]; [[sources/drivefine.md]] reported both only for itself), and it independently confirms WA-JEPA's pre-fix/corrected pair for itself (86.4 → 89.9) and for DiffusionDriveV2 (85.5 → 87.5) — while **contradicting** WA-JEPA's cohort assignment for ReCogDrive and DriveVLA-W0. It also publishes the column mapping for the shared baseline block. See [Two Columns](#two-columns).

The paper was already a named gap on [[concepts/navsim-benchmark.md]] and in [[sources/wa-jepa.md]], [[sources/auto-jepa.md]] and [[sources/da-wam.md]] — the "89.9 corrected EPDMS, not ingested" row. It is now ingested and the row checks out.

---

## Positioning

![[motivationv13.png|DriveFuture motivation: (a) prior latent world models simulate futures as prediction targets; (b) DriveFuture conditions planning on future latents, GT at training and predicted at inference; (c) leaderboard results]]

**Figure 1**: (a) Existing latent world models (LAW, World4Drive, WorldRFT, DriveWorld, DriveWorld-VLA, DriveVLA-W0, DriveLaW) simulate future latent states and use them as prediction targets or supervision signals, without explicitly shaping the current representation for planning. (b) DriveFuture uses future latent states as direct conditions for the planning process — GT future states during training, predicted future states at inference. (c) SOTA on the NAVSIM-v2 navhard and NAVSIM-v1 navtest leaderboards.

The framing the paper picks is deliberately cinematic — *"bringing future knowledge back to the present"*, which it attributes to *The Terminator* — but the technical claim underneath it is precise and worth separating from the packaging:

> "In most existing systems, future information is introduced only at the output level, e.g., by generating multiple candidate trajectories or scoring future outcomes after the current representation has already been formed. As a result, they improve trajectory generation or selection, but do not fundamentally reshape how the current latent state itself is learned for planning."

That is a claim about **where in the pipeline the future enters**, and it cuts across the taxonomy on [[concepts/world-model-for-ad.md]] in a way none of the existing patterns do. Appendix B.2 makes the objective explicit: where a conventional latent world model minimises $D(f_\psi(\mathbf{Z}_t,a_t),\mathbf{Z}^\star_{t+T})$, DriveFuture minimises the trajectory denoising loss *through* $f_\psi$, so

> "the predicted future latent is not required to reconstruct every visual detail of the future scene. It only needs to preserve the future information that the planner can exploit."

**There is no future-prediction loss anywhere in DriveFuture.** $\mathcal{L}=\lambda_{\mathrm{plan}}\mathcal{L}_{\mathrm{plan}}+\lambda_{\mathrm{bev}}\mathcal{L}_{\mathrm{BEV}}$, both weighted 10. The world model's only gradients come from trajectory denoising and from the cross-attention that grounds it against real future observations. This is the third option the [objective-form](../concepts/world-model-for-ad.md#objective-form) discussion did not have: not regression, not flow matching, but **no prediction objective at all**.

**Lineage.** Yingyan Li (CASIA) is the name and institution of LAW's first author, and LAW is the paper DriveFuture positions itself as the successor to — the same relationship [[sources/reworld.md]] has to [[sources/drivelaw.md]]. The author list also shares Shaoqing Xu and Ziying Song with [[sources/elf-vla.md]], and Lin Liu + Ziying Song with GuideFlow and **DriveWorld-VLA** — which is the one NAVSIM-v1 method in its own table that DriveFuture does not beat (91.3 vs. 90.7), and which is not ingested here.

---

## Method

![[methodv8.png|DriveFuture architecture: shared perception encoder, Latent Dynamics Predictor conditioned on a tokenised trajectory intent, Future Alignment Adapter grounding against GT future BEV, LatentAlign annealing, and a future-conditioned diffusion Planning Decoder]]

**Figure 2**: Multi-view observations at $t$ are encoded by a shared Perception Encoder into $\mathbf{Z}_t$. The Latent Dynamics Predictor conditions on $\mathbf{Z}_t$ and a tokenised trajectory intent to produce $\hat{\mathbf{Z}}_{t+T}$. During training the future observation at $t{+}T$ is encoded by the **same** encoder into $\mathbf{Z}_{t+T}$; the Future Alignment Adapter grounds $\hat{\mathbf{Z}}_{t+T}$ against it by cross-attention, yielding $\mathbf{Z}^c_{t+T}$. LatentAlign anneals the planning condition from $\mathbf{Z}^c_{t+T}$ towards $\hat{\mathbf{Z}}_{t+T}$ over training, closing the train–inference gap. At inference the adapter is bypassed and the Planning Decoder consumes $\mathbf{Z}_t$ and $\hat{\mathbf{Z}}_{t+T}$ as dual conditioning contexts.

### Latent Dynamics Predictor

$$\hat{\mathbf{Z}}_{t+T}=f_\psi\!\left(\mathbf{Q}_f;\,[\mathbf{Z}_t\,\|\,\mathbf{E}_\tau]\right),\qquad \mathbf{Q}_f\in\mathbb{R}^{K\times d},\ K=16,\ d=256$$

Four pre-norm decoder layers (self-attn → cross-attn over the joint context → FFN, FFN width 2048). The trajectory tokenizer $\phi_\tau$ maps an absolute trajectory to $T{=}8$ tokens through normalised finite differences plus sine/cosine heading:

$$\phi_\tau(\boldsymbol{\tau})_k=\operatorname{LN}\!\left(W_\tau[\bar{\Delta x}_k,\bar{\Delta y}_k,\sin\theta_k,\cos\theta_k]^\top+\mathbf{p}_k\right)$$

**The 16-token output is described as a deliberate bottleneck** — "it prevents the world model from copying dense future appearance" — against a 64-token future BEV bank. That 4:1 compression is the structural reason the adapter is cheap and the reason the conditioning footprint is independent of BEV resolution.

### Conditioning source randomisation

The predictor needs a trajectory it does not have at inference. Training draws the intent from three sources with $(p_{\mathrm{gt}},p_{\mathrm{kin}},p_\varnothing)=(0.4,0.4,0.2)$:

$$\mathbf{E}_\tau\sim\begin{cases}\phi_\tau(\boldsymbol{\tau}^{\mathrm{gt}}) & 0.4\\ \phi_\tau(\boldsymbol{\tau}^{\mathrm{kin}}) & 0.4\\ \mathbf{E}^\varnothing & 0.2\end{cases}$$

with $\boldsymbol{\tau}^{\mathrm{kin}}$ a constant-acceleration rollout from $(v_x,v_y,a_x,a_y)$. The stated purpose of the null branch is calibration, not regularisation: *"without an unconditional baseline, guidance would be an uncalibrated difference between two conditional predictions rather than a proper correction direction."* This is CFG dropout applied to a **world model's input**, which is not the same thing as [[sources/dial.md]]'s intent-CFG on a flow field — there the dropped condition is the generator's own; here it is one level upstream.

### Future Alignment Adapter — the training-time oracle

$$\mathbf{Z}_{t+T}=\operatorname{sg}\!\left(\phi_{\mathrm{enc}}(\mathbf{I}_{t+T})\right)\in\mathbb{R}^{64\times d},\qquad \tilde{\mathbf{Z}}_{t+T}=\operatorname{MHA}\!\left(\operatorname{LN}(\hat{\mathbf{Z}}_{t+T}),\,\mathbf{Z}_{t+T},\,\mathbf{Z}_{t+T}\right)$$

Three design points are load-bearing and stated:

- **Stop-gradient on the future encoder branch**, "to prevent the future branch from acting as a shortcut."
- **No residual from $\hat{\mathbf{Z}}_{t+T}$**, so "the oracle condition is a pure selection of ground-truth future evidence." The prediction supplies the *query* — what to look for — and the real future supplies the *content*. That makes the oracle trajectory-aware rather than a generic future summary.
- **Fallback at sequence boundaries**: samples without a future frame use $\hat{\mathbf{Z}}_{t+T}$ unrefined.

### LatentAlign

$$\alpha(e)=1-\sigma\!\left(\beta(e-e_0)\right),\qquad \tilde{\mathbf{Z}}^c_{t+T}=\alpha(e)\,\mathbf{Z}^c_{t+T}+(1-\alpha(e))\,\hat{\mathbf{Z}}_{t+T},\qquad e_0=\rho_E E$$

Early training gets the oracle; late training gets only the self-prediction, matching inference. This is scheduled sampling applied to a world-model condition, and **it is the first instance in the wiki of a privileged training-time signal that is explicitly annealed to zero rather than simply dropped at test time**. Compare the privileged-supervision cases already on file — Hydra-MDP distillation, [[sources/auto-jepa.md]]'s CLOVER scorer, [[sources/adaptive-wam.md]]'s pseudo-expert targets, [[sources/da-wam.md]]'s factor heads. Those distil a simulator's *labels*; this one hands the planner a real future *observation* and then takes it away.

### Planning Decoder

DDPM over the differential action $\mathbf{a}=(\Delta x,\Delta y,\sin\theta,\cos\theta)\in\mathbb{R}^{8\times4}$:

$$\hat{\boldsymbol{\epsilon}}=\epsilon_\theta\!\left(\mathbf{a}_s,\,s,\;\mathbf{C}_{\mathrm{scene}}=[\mathbf{e}_s\,\|\,\mathbf{Z}_t],\;\mathbf{Z}^c_{t+T}\right)$$

Each block cross-attends the scene **then** the future, "so that environmental geometry is consumed before the future-semantic correction is applied," and the future cross-attention's output projection is **zero-initialised** so the module starts as identity — the standard adapter warm-start, here used to preserve a planner pretrained without future conditioning.

### Progressive Foresight Guidance (PFG) — breaking the circularity

At inference $\hat{\mathbf{Z}}_{t+T}=f_\psi(\mathbf{Z}_t;\mathbf{E}_\tau)$ needs an intent, and the intent is what is being denoised. PFG supplies two surrogates and mixes three CFG branches:

$$\hat{\boldsymbol{\epsilon}}=\hat{\boldsymbol{\epsilon}}_\varnothing+w_{\mathrm{kin}}(r)\left(\hat{\boldsymbol{\epsilon}}_{\mathrm{kin}}-\hat{\boldsymbol{\epsilon}}_\varnothing\right)+w_{\mathrm{tw}}(r)\left(\hat{\boldsymbol{\epsilon}}_{\mathrm{tw}}-\hat{\boldsymbol{\epsilon}}_\varnothing\right)$$

The second surrogate is the Tweedie estimate of the clean trajectory from the current noisy sample, $\hat{\mathbf{a}}_0^{(s)}=(\mathbf{a}_s-\sqrt{1-\bar\alpha_s}\,\hat{\boldsymbol{\epsilon}}_\varnothing)/\sqrt{\bar\alpha_s}$, integrated by `cumsum`. **Appendix B.6 derives why the schedule has to be phase-dependent**, and this is the cleanest piece of reasoning in the paper: if $\hat{\boldsymbol{\epsilon}}_\varnothing=\epsilon+\delta$ then

$$\hat{\mathbf{a}}_0^{(s)}-\mathbf{a}_0=-\sqrt{\tfrac{1-\bar\alpha_s}{\bar\alpha_s}}\,\delta$$

so the surrogate's error is amplified by a factor that diverges at high noise. Hence $w_{\mathrm{kin}}$ decays as a cosine and cuts off at $\rho=0.7$; $w_{\mathrm{tw}}$ rises from $\nu=0.3$; they overlap on $(0.3,0.7)$ and hand over. Defaults $w^{\max}_{\mathrm{kin}}=1.5$, $w^{\max}_{\mathrm{tw}}=2.5$.

**A consequence the paper does not draw.** $\boldsymbol{\tau}^{\mathrm{tw}}$ is computed from *each proposal's own* noisy sample, so the Tweedie-branch future latent is **per-proposal**, recomputed at every step where $w_{\mathrm{tw}}>0$. The $\varnothing$ and kinematic branches are computed once per scene and broadcast. [[sources/da-wam.md]]'s taxonomy files DriveFuture under "(b) loosely coupled latent fusion — one proposal, nothing to compare"; that is right for two of the three branches and wrong for the third. DriveFuture is, in the low-noise phase, a **per-candidate future** method of exactly the kind DA-WAM claims as novel — applied inside the generator instead of the scorer. See [[concepts/world-model-for-ad.md]].

### Training configuration

NAVSIM navtrain, Adam $1{\times}10^{-4}$, batch 16 with 5-step accumulation, 100–130 epochs, bf16, grad clip 1.0, **8× NVIDIA 5090**, cache-only feature loading. Inputs are two temporal frames and two stitched panoramas (front = left-front + front + right-front; rear likewise), both $2048\times512$. BEV semantic auxiliary is a 7-class cross-entropy on an upsampled BEV map. GTRS-Dense scorer selects among 100 proposals for the leaderboard submissions.

---

## Results

### Table 1 — NAVSIM-v2 navhard (combined two-stage EPDMS)

| Method | Backbone | Stage | NC↑ | DAC↑ | DDC↑ | TL↑ | EP↑ | TTC↑ | LK↑ | HC↑ | EC↑ | EPDMS↑ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *E2E-based* | | | | | | | | | | | | |
| TransFuser | ResNet-34 | 1 | 96.2 | 79.5 | 99.1 | 99.5 | 84.1 | 95.1 | 94.2 | 97.5 | 79.1 | 23.1 |
| | | 2 | 77.7 | 70.2 | 84.2 | 98.0 | 85.1 | 75.6 | 45.4 | 95.7 | 75.9 | |
| [[sources/diffusiondrive.md]] | ResNet-34 | 1 | 96.0 | 79.7 | 97.4 | 99.5 | 81.3 | 93.1 | 90.8 | 96.8 | 73.8 | 24.2 |
| | | 2 | 82.1 | 72.2 | 88.5 | 98.7 | 85.1 | 78.8 | 49.2 | 89.3 | 71.2 | |
| GuideFlow | ResNet-34 | 1 | 96.6 | 80.5 | 96.3 | 99.3 | 82.3 | 94.9 | 91.5 | 97.7 | 67.8 | 27.1 |
| | | 2 | 87.3 | 76.7 | 88.8 | 99.2 | 84.3 | 85.1 | 49.7 | 93.1 | 44.5 | |
| Senna-E2E | ResNet-50 | 1 | 95.6 | 86.0 | 98.9 | 99.6 | 83.9 | 95.1 | 95.3 | 97.6 | 75.6 | 27.2 |
| | | 2 | 78.6 | 74.8 | 84.8 | 98.2 | 88.2 | 75.7 | 46.9 | 96.0 | 65.8 | |
| [[sources/drivesuprim.md]] | V2-99 | 1 | 98.9 | 95.1 | 99.2 | 99.6 | 76.1 | 99.1 | 94.7 | 97.6 | 54.2 | 42.1 |
| | | 2 | 87.9 | 88.8 | 89.6 | 98.8 | 80.3 | 86.0 | 53.5 | 97.1 | 56.1 | |
| ZTRS | V2-99 | 1 | 98.9 | 97.6 | 100.0 | 100.0 | 66.7 | 98.9 | 96.2 | 96.7 | 44.0 | 48.1 |
| | | 2 | 91.1 | 90.4 | 95.8 | 99.0 | 63.6 | 89.8 | 60.4 | 97.6 | 66.1 | |
| GTRS-E | V2-99+EVA-ViT-L+ViT-L | 1 | 98.9 | 99.3 | 99.8 | 99.8 | 75.2 | 98.4 | 96.0 | 97.6 | 51.6 | 49.4 |
| | | 2 | 92.3 | 93.3 | 94.6 | 99.2 | 73.1 | 91.2 | 53.9 | 96.7 | 56.8 | |
| SimScale | V2-99 | 1 | 99.6 | 99.1 | 99.9 | 100.0 | 69.6 | 99.6 | 95.8 | 95.6 | 28.4 | 53.2 |
| | | 2 | 94.5 | 94.2 | 95.8 | 99.2 | 75.8 | 92.8 | 60.1 | 96.1 | 43.2 | |
| DrivoR | ViT-S | 1 | 99.1 | 98.2 | 99.3 | 99.8 | 75.4 | 98.7 | 94.9 | 97.6 | 70.2 | 54.6 |
| | | 2 | 92.3 | 91.6 | 97.3 | 99.1 | 75.7 | 90.6 | 56.1 | 98.4 | 44.7 | |
| *VLA-based* | | | | | | | | | | | | |
| [[sources/spanvla.md]] | Qwen2.5-VL-3B | 1 | 98.4 | 94.3 | 97.8 | 99.9 | 85.7 | 97.2 | 94.2 | 97.6 | 72.1 | 40.1 |
| | | 2 | 86.9 | 84.3 | 87.1 | 98.2 | 85.5 | 82.7 | 62.3 | 96.8 | 67.4 | |
| DiffVLA | V2-99 + ViT-L/14 | 1 | 95.7 | 99.2 | 100.0 | 100.0 | 85.9 | 96.4 | 97.1 | 95.0 | 84.2 | 45.0 |
| | | 2 | 81.2 | 88.8 | 94.6 | 99.0 | 86.0 | 76.4 | 59.8 | 98.6 | 80.4 | |
| *World-model-based* | | | | | | | | | | | | |
| MindDrive | ResNet-34 | 1 | 96.1 | 86.0 | 98.8 | 99.3 | 83.3 | 95.6 | 94.4 | 97.6 | 74.7 | 30.9 |
| | | 2 | 82.6 | 79.1 | 86.4 | 98.0 | 85.3 | 79.4 | 49.2 | 96.5 | 71.0 | |
| World4Drive | ResNet-34 | 1 | 97.3 | 89.1 | 97.6 | 99.7 | 60.5 | 96.8 | 87.7 | 93.1 | 60.0 | 34.9 |
| | | 2 | 91.4 | 82.0 | 91.0 | 98.5 | 53.1 | 90.6 | 52.3 | 93.3 | 62.8 | |
| **DriveFuture** | V2-99 | 1 | **99.8** | **99.8** | **100** | 99.6 | 85.7 | **99.8** | **98.7** | 97.6 | 66.2 | **55.5** |
| | | 2 | 90.6 | 87.5 | 94.1 | 99.1 | 84.6 | 88.8 | 58.3 | 93.5 | 45.6 | |

**This is by far the largest navhard table in the wiki**, and it changes the picture on [[concepts/navhard-ood-evaluation.md]] substantially. The previous leaderboard there came from [[sources/geowam.md]] and topped out at **36.6**; it contained no trajectory-scoring method at all. Here, every entry above 42 is a selection-based planner (DriveSuprim, ZTRS, GTRS-E, SimScale, DrivoR) or DriveFuture with a scorer bolted on.

**The prose does not match the table.** §4.3 says DriveFuture "demonstrates consistent improvements across compliance and safety-related metrics, including 99.1 DAC and 95.4 TTC" — but the navhard row is 99.8/87.5 DAC and 99.8/88.8 TTC by stage. **99.1 DAC and 95.4 TTC are the NAVSIM-v1 numbers** from Table 3. The navhard claim is supported by different numbers than the ones quoted for it.

### Table 2 — NAVSIM-v2 navtest {#two-columns}

EPDMS* = pre-fix evaluator; EPDMS = corrected official implementation with human-behaviour filtering.

| Method | NC↑ | DAC↑ | DDC↑ | TL↑ | EP↑ | TTC↑ | LK↑ | HC↑ | EC↑ | EPDMS*↑ | EPDMS↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TransFuser | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 | – |
| [[sources/diffusiondrive.md]] | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | – | 84.5 |
| Hydra-MDP++ | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 | – |
| [[sources/drivesuprim.md]] | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 | – |
| ARTEMIS | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | 98.3 | 83.1 | – |
| [[sources/diffusiondrive-v2.md]] | 97.7 | 96.6 | 99.2 | 99.8 | 88.9 | 97.2 | 96.0 | 97.8 | 91.0 | 85.5 | 87.5 |
| DriveWorld-VLA | 98.6 | 99.1 | 99.6 | 99.8 | 87.4 | 97.9 | 97.0 | 97.8 | 78.6 | – | 86.8 |
| [[sources/drivevla-w0.md]] | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | – | 86.1 |
| ReCogDrive | 98.3 | 95.2 | 98.3 | 99.8 | 87.1 | 97.5 | 96.6 | 99.5 | 86.5 | – | 83.6 |
| [[sources/latent-wam.md]] | 98.1 | 97.3 | 99.6 | 99.8 | 87.7 | 97.3 | 97.6 | 98.1 | 87.3 | – | 89.3 |
| **DriveFuture** | **98.8** | **99.1** | 99.6 | **99.9** | 86.6 | **98.4** | 96.4 | 98.3 | 74.8 | 86.4 | **89.9** |

**Four things this table settles or unsettles for [[concepts/navsim-benchmark.md]]** — it is the second here to print both columns as a table, after [[sources/wa-jepa.md]]:

1. **The wiki's DriveFuture row is verified at the primary source.** 86.4 pre-fix → 89.9 corrected, a +3.5 delta, exactly as [[sources/wa-jepa.md]] reported it second-hand. So is DiffusionDriveV2's 85.5 → 87.5 (+2.0). Two of the eight measured correction deltas on that page now have two independent sources.
2. **It contradicts WA-JEPA's cohort assignment twice.** WA-JEPA places ReCogDrive 83.6 and DriveVLA-W0 86.1 in the **pre-fix** cohort; DriveFuture puts both in the **corrected** column. One of the two attributions is wrong and nothing in either paper resolves it.
3. **It is the fourth paper drawing on the shared baseline block, and the first to label it.** Its Transfuser (76.7), DriveSuprim (83.1), DiffusionDrive (84.5) and DriveVLA-W0 (86.1) rows are **digit-identical across all nine submetrics** to [[sources/geoworldad.md]]'s, the same block [[sources/lwdrive.md]] and [[sources/brainwam.md]] carry. The wiki had *inferred* that this block mixes conventions; DriveFuture sorts it into two labelled columns and **says so** — TransFuser, Hydra-MDP++, DriveSuprim and ARTEMIS pre-fix; DiffusionDrive, ReCogDrive, DriveVLA-W0, DriveWorld-VLA and Latent-WAM corrected. It also agrees with GeoWAM and BrainWAM on Hydra-MDP++ at 81.4 against WA-JEPA's 84.1.
4. **It propagates the ARTEMIS EC = 98.3 duplicate** that DA-WAM, WCog-VLA and LWDrive also carry — a value identical to that method's own HC, where other papers print "–". Fourth occurrence; the tally is now four-for-98.3 against one-for-89.1.

On TransFuser, DriveFuture sides with the majority: 76.7 in the **pre-fix** column, corrected left blank. It is the sixth table in the wiki reporting 76.7, against GeoWAM's 84.0 from identical submetrics — and it adds the information that **no paper here has computed a corrected TransFuser number at all**, which leaves GeoWAM's 84.0 unrefuted rather than outvoted: the other five all report the *pre-fix* value.

### Table 3 — NAVSIM-v1 navtest (PDMS)

| Method | NC↑ | DAC↑ | TTC↑ | Comf.↑ | EP↑ | PDMS↑ |
|---|---:|---:|---:|---:|---:|---:|
| Human | 100 | 100 | 100 | 99.9 | 87.5 | 94.8 |
| Constant Velocity | 69.9 | 58.8 | 49.3 | 100 | 49.3 | 21.6 |
| *E2E-based* | | | | | | |
| VADv2 | 97.2 | 89.1 | 91.6 | 100 | 76.0 | 80.9 |
| UniAD | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| TransFuser | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| PARA-Drive | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| DRAMA | 98.0 | 93.1 | 94.8 | 100 | 80.1 | 85.5 |
| GoalFlow | 98.3 | 93.8 | 94.3 | 100 | 79.8 | 85.7 |
| Hydra-MDP | 98.3 | 96.0 | 94.6 | 100 | 78.7 | 86.5 |
| ARTEMIS | 98.3 | 95.1 | 94.3 | 100 | 81.4 | 87.0 |
| [[sources/diffusiondrive.md]] | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| DIVER | 98.5 | 96.5 | 94.9 | 100 | 82.6 | 88.3 |
| [[sources/drivesuprim.md]] | 97.8 | 97.3 | 93.6 | 100 | 86.7 | 89.9 |
| GoalFlow *(second row, unlabelled)* | 98.4 | 98.3 | 94.6 | 100 | 85.0 | 90.3 |
| *VLA-based* | | | | | | |
| AutoVLA | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 |
| ReCogDrive | 98.2 | 97.8 | 95.2 | 99.8 | 83.5 | 89.6 |
| [[sources/drivevla-w0.md]] | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.2 |
| DriveWorld-VLA | **99.1** | 98.2 | **96.1** | 100 | **85.9** | **91.3** |
| *World-model-based* | | | | | | |
| LAW | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 |
| World4Drive | 97.4 | 94.3 | 92.8 | 100 | 79.9 | 85.1 |
| WorldRFT | 97.8 | 96.8 | 94.0 | 100 | 81.7 | 87.8 |
| WoTE | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| [[sources/drivelaw.md]] | 99.0 | 97.1 | 96.7 | 100 | 81.3 | 89.1 |
| **DriveFuture** | 98.8 | **99.1** | 95.4 | 100 | 84.2 | **90.7** |

**90.7 is mid-pack by wiki standards**, not a frontier result: CLEAR 93.7 = DA-WAM 93.7 > DriveSuprim 93.5 > Drive-JEPA 93.3 > WCog-VLA 92.9 > HybridDriveVLA 92.1 > LWDrive 92.0 > WA-JEPA 91.8 > DynVLA 91.7 > SimWAM 91.5. The abstract's "achieves SOTA performance on NAVSIM-v1 navtest" is scoped to this table, which **omits every one of those** and also carries DriveSuprim at the circulated **89.9** rather than its published 93.5 — the same substitution [[sources/geoworldad.md]] made, now a second instance. GoalFlow appears twice with different numbers and no distinguishing label.

**The world-model group is where the claim actually lands**: 90.7 against DriveLaW 89.1, WoTE 88.3, WorldRFT 87.8, World4Drive 85.1, LAW 84.6. Within its own family and its own table the result is clean.

---

## Ablations

### Table 4 — Future supervision and PFG guidance (navhard, no scorer)

| FF | Impl | MSE | KS | GT | EPDMS | NC | DAC | TTC | LK | HC | EC |
|:-:|:-:|:-:|:-:|:-:|---:|---:|---:|---:|---:|---:|---:|
| | | | ✓ | ✓ | 30.9 | 81.8 | 72.6 | 79.7 | 46.4 | 95.7 | 71.7 |
| ✓ | | ✓ | ✓ | ✓ | 32.1 | 81.7 | 77.0 | 78.4 | 49.5 | 96.7 | 69.9 |
| ✓ | ✓ | | | | 32.0 | 81.1 | 75.2 | 78.1 | 47.6 | 97.0 | 75.9 |
| ✓ | ✓ | | ✓ | ✓ | **34.6** | 82.3 | 78.8 | 79.6 | 47.6 | 97.0 | 75.9 |

FF = future frames used in training; Impl = the implicit future constraint (adapter + LatentAlign); MSE = direct feature regression on the future latent; KS = kinematic guidance source; GT = GT-trajectory guidance source.

Four comparisons are clean because only one factor moves between the rows:

| Comparison | Rows | Δ EPDMS | Reading |
|---|---|---:|---|
| Future-frame grounding vs. none | 30.9 → 34.6 | **+3.7** | What real future observations add on top of an already-present foresight latent |
| **Implicit conditioning vs. MSE regression** | 32.1 → 34.6 | **+2.5** | The paper's central claim, and it holds |
| Dual-source guidance vs. none | 32.0 → 34.6 | **+2.6** | PFG's sources matter at *training* time, not only inference |
| MSE regression vs. no future frames | 30.9 → 32.1 | +1.2 | Regression helps a little, against WA-JEPA |

**The baseline row is not "no world model."** FF is "whether future frames are considered during training", and the KS/GT columns are the world model's own conditioning-source modes — so row 1 still builds $\hat{\mathbf{Z}}_{t+T}$ and still feeds it to the DiT; it simply never sees a future observation. Appendix C.3 names `use_wm` and `use_wm_to_dit` as separate switches and **no reported table varies either of them**. So +3.7 prices the *grounding*, not the foresight-latent pathway, and the paper never measures what the pathway is worth on its own.

**This is the wiki's second controlled comparison of future-prediction objective form**, and it partly disagrees with the first. [[sources/wa-jepa.md]] measured deterministic regression on future scene latents as *worse than not predicting the future at all* (90.7 vs. 91.1) and flow matching as +0.6 over the null. DriveFuture measures regression as **+1.2 over the null**, not negative — but still 2.5 below its conditioning alternative. The two are reconcilable through the entropy-of-target argument already on [[concepts/world-model-for-ad.md]]: WA-JEPA regresses multi-view EMA ViT-L scene latents, a high-entropy target where the conditional mean is a blur; DriveFuture regresses a 16-token compression of a 64-token BEV bank, far lower entropy. **The ordering survives in both papers even though the sign of the regression term does not.**

The third arm is new and is the reason this page thinks the result generalises: DriveFuture's winning configuration has **no future-prediction loss at all**. The future latent is shaped only by the planning gradient plus an attention-based grounding against real future evidence. That is a different answer from "use a better prediction objective."

### Table 5 — Hyper-parameter sensitivity (navhard, no scorer)

| $t_f$ (s) | EPDMS | $q_s$ | EPDMS | $e_0$ | EPDMS |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 30.2 | 4 | 28.2 | 0.75 | 29.2 |
| 1.0 | 31.2 | 16 | **34.6** | 0.83 | **34.6** |
| 1.5 | **34.6** | 64 | 33.7 | 0.95 | 28.9 |

**Read against the 30.9 no-future baseline from Table 4, this table is the paper's most uncomfortable result and it goes unremarked.**

- $q_s=4$ → **28.2**, and $e_0=0.95$ → **28.9**, and $t_f=0.5$ → **30.2**. All three are *below* 30.9. **A misconfigured future condition is worse than no future condition**, and a third of the swept settings are misconfigured.
- The annealing inflection $e_0$ swings **5.7 EPDMS** across $0.75$–$0.95$ — larger than the +3.7 the entire mechanism is worth. The paper's contribution is smaller than its own schedule's sensitivity.
- **$e_0=0.95$ is the closest thing to a `force_alpha_one` result the paper reports.** Appendix C.3 names that switch — "tests the effect of keeping the oracle future condition throughout training" — and never reports it. The $e_0$ trend says what it would show: the longer the GT-future oracle is held, the worse the model, and by 95% of training it is net harmful. That is [[sources/drivelaw.md]]'s t=10 collapse (89.1 → 23.2 when conditioning on the finished future) arriving from the opposite direction, at much smaller magnitude, on a different representation. **Two independent papers now measure the same thing: a planner conditioned on a future it cannot produce at inference degrades.** DriveLaW hit it by reading a clean generated latent; DriveFuture hits it by keeping a real one too long.
- $t_f=1.5$ s is the **largest value swept**, so "best in group" is a boundary, not an optimum. The conclusion "effective planning requires a balanced future horizon" is not supported on the upper side — nothing beyond 1.5 s was tried, and the paper separately lists near-future-only conditioning as a limitation.

### Table 6 — Relative comparison on navhard

| Method | EPDMS | Gain | Rel. |
|---|---:|---:|---:|
| TransFuser | 23.1 | +32.4 | +140.3% |
| [[sources/diffusiondrive.md]] | 24.2 | +31.3 | +129.3% |
| GuideFlow | 27.1 | +28.4 | +104.8% |
| MindDrive | 30.9 | +24.6 | +79.6% |
| World4Drive | 34.9 | +20.6 | +59.0% |
| GTRS-E | 49.4 | +6.1 | +12.3% |
| SimScale | 53.2 | +2.3 | +4.3% |
| DrivoR | 54.6 | +0.9 | +1.6% |

The honest version of this table is its last row: **+0.9 over DrivoR**, a ViT-S model. Everything above it is a comparison against a method without dense proposal scoring.

### Table 7 — Stage-wise navhard, with and without the scorer {#decomposition}

| Model | Stage | NC | DAC | DDC | TL | EP | TTC | LK | HC | EC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DriveFuture | 1 | 96.3 | 87.8 | 98.2 | 99.6 | 83.1 | 96.9 | 94.9 | 97.6 | 76.9 |
| DriveFuture | 2 | 82.3 | 78.8 | 88.1 | 98.4 | 83.6 | 79.6 | 47.6 | 97.0 | 75.9 |
| DriveFuture + Score | 1 | **99.8** | **99.8** | **100.0** | 99.6 | 85.7 | **99.8** | **98.7** | 97.6 | 66.2 |
| DriveFuture + Score | 2 | 90.6 | 87.5 | 94.1 | 99.1 | 84.6 | 88.8 | 58.3 | 93.5 | 45.6 |

**The unscored model's Stage-2 row is Table 4's best ablation row, digit for digit** — NC 82.3, DAC 78.8, TTC 79.6, LK 47.6, HC 97.0, EC 75.9 against the identical six values at EPDMS 34.6. Every ablation in the paper is labelled "without GTRS-Dense scorer." So:

$$\underbrace{55.5}_{\text{headline}}\;=\;\underbrace{34.6}_{\text{full model}}\;+\;\underbrace{20.9}_{\text{GTRS-Dense scorer}},\qquad \underbrace{34.6}_{\text{full model}}\;=\;\underbrace{30.9}_{\text{no future frames}}\;+\;\underbrace{3.7}_{\text{future-frame grounding}}$$

**The scorer is worth 5.6× the mechanism the paper is about** — and 30.9 is not a world-model-free baseline, so 3.7 is an upper bound on what the grounding adds and says nothing about the foresight latent underneath it. The paper is not hiding this — Table 7 exists precisely to report it, and §D.1 discusses the scorer's effect frankly, including the EC trade-off (76.9 → 66.2 on Stage 1, 75.9 → 45.6 on Stage 2: safer, more rule-compliant proposals are less comfortable). But the abstract, the contributions list, and the leaderboard claim all attribute 55.5 to future-conditioned latent modelling, and 62% of the distance from the no-future baseline (30.9) to 55.5 is scoring.

This is the sharpest instance yet of the pattern [[concepts/selection-based-planning.md]] tracks — *nothing above 92.0 PDMS scores its own candidates* — and the first time the wiki can price the scorer against the world model **inside one paper, on the reactive split**.

---

## Qualitative Results

![[visv2.png|Visual comparison between World4Drive and DriveFuture across NAVSIM-v2 navhard scenarios]]

**Figure 3**: Comparison against World4Drive on navhard. The paper reports better behaviour in three named failure categories — collision, inefficiency, and braking — as evidence that "conditioning current decision representations on future states is more effective than using future states merely as prediction targets."

![[tu3.drawio.png|DriveFuture trajectories across six navhard scenarios: curved turning, straight lane following with traffic, complex intersection, dense traffic, narrow road with parked vehicles, roundabout]]

**Figure 4**: Six scenarios — curved turning, straight lane following with nearby traffic, turning through a complex intersection (top row); dense-traffic lane following, a narrow segment constrained by parked vehicles, roundabout navigation (bottom row). §D.3 claims smoother curvature transitions, fewer jitter/over-steering artifacts, and earlier adaptation to conflict-zone geometry.

Neither figure carries quantitative annotation, and the failure-category claim in §4.5 has no counts behind it.

---

## Limitations

1. **The headline is 85% scorer.** 30.9 (no future frames) → 34.6 (full model) → 55.5 (with GTRS-Dense). See [The Decomposition](#decomposition). Every ablation is run without the scorer, so the paper never measures whether future conditioning still helps *once a strong scorer is present* — which is the configuration it actually submits. If the scorer saturates Stage-1 safety (NC 99.8, DAC 99.8, DDC 100.0), a +3.7 proposal-quality gain may not survive it.
2. **The foresight-latent pathway is never ablated.** `use_wm` and `use_wm_to_dit` are named in Appendix C.3 and varied in no table. The ablation's baseline removes *future observations*, not the world model, so the paper's own headline mechanism — a predicted future latent conditioning a diffusion planner — has no measured value in the paper that proposes it. The one-line experiment is already implemented.
3. **A third of the hyper-parameter sweep is below the no-future-frame baseline** (28.2, 28.9, 30.2 against 30.9), and the annealing inflection is more sensitive than the grounding is large. Reported without comment.
4. **`force_alpha_one` is defined and never reported.** Appendix C.3 names the switch that would test whether the oracle condition is necessary or merely tolerable; no row uses it. $e_0=0.95$'s 28.9 is the only evidence and it is indirect.
5. **No latency, no parameter count, no FLOPs.** At inference the system runs 100 proposals, each needing $\hat{\boldsymbol{\epsilon}}_\varnothing$ plus up to two guidance branches per denoising step, plus a world-model forward per proposal per step wherever $w_{\mathrm{tw}}>0$, plus a GTRS-Dense scoring pass over 100 candidates. Against [[sources/adaptive-wam.md]]'s 170 ms and the latencies recorded for [[sources/simwam.md]] (518 ms) and ForeSight (900 ms), this is a conspicuous omission for a method whose whole argument is that latent world models are the cheap option. The number of denoising steps $S$ is never stated either.
6. **The navhard prose cites NAVSIM-v1 numbers.** "99.1 DAC and 95.4 TTC" in §4.3 are Table 3's values, not Table 1's.
7. **Baseline-table provenance.** DriveSuprim appears at 89.9 on v1 where its published headline is 93.5 — the circulated value [[sources/geoworldad.md]] also used; GoalFlow appears twice on v1 with different numbers and no distinguishing label; ARTEMIS's EC duplicates its own HC at 98.3. The v1 table omits every method above 91.3. Its v2 baselines are the shared block, which is a provenance finding rather than an error — see item 3 above.
8. **Cross-table contradiction with WA-JEPA on two cohort assignments** (ReCogDrive, DriveVLA-W0), unresolvable from either paper.
9. **The comparison against World4Drive is the only world-model head-to-head**, and World4Drive is a ResNet-34 model against DriveFuture's V2-99 at $2048\times512$ with two temporal frames. The +20.6 navhard gain in Table 6 is not backbone-matched. **No ablation anywhere varies the backbone.**
10. **Single runs throughout.** No seed variance, on a benchmark where [[sources/wa-jepa.md]] measured std 0.053 for a flow sampler and where several of this paper's ablation deltas are 1–2 points.
11. **Near-future only.** $t_f\le1.5$ s against a 4 s planning horizon, acknowledged as a limitation. The world model predicts a single future step, not a rollout, so "long-horizon planning" in the introduction's motivation for latent world models is not something this instantiation does.
12. **No generation evaluation of any kind**, which is consistent with the design (there is nothing to decode) but means the [generation-vs-planning decoupling](../concepts/world-model-for-ad.md#generation-planning-decoupling) question cannot be asked here.
13. **No code release, no closed-loop benchmark** (no Bench2Drive, no HUGSIM, no nuScenes), and navhard is the only reactive evaluation.
14. **Minor QC**: §4.3 refers to "Table 4.2" (a broken cross-reference to Table 3); Figure 3's caption is referenced as "Fig. 4" in §4.5; Figure 4 appears under a NAVSIM-v2 *navtest* subsection with a *navhard* caption; the abstract dates the leaderboard claim to April 2026 against a 2605 arXiv identifier.

---

## Key Cross-References

- [[concepts/world-model-for-ad.md]] — Pattern 33 (future latent as planning **condition** with a training-time oracle annealed away); the [objective-form](../concepts/world-model-for-ad.md#objective-form) discussion gains a third arm (no prediction loss at all); the [shared-future](../concepts/world-model-for-ad.md#shared-future-reopened) thread gains a per-candidate branch inside a generator.
- [[concepts/navhard-ood-evaluation.md]] — the combined leaderboard is rebuilt from this paper's Table 1; SpanVLA's 40.1 convention ambiguity is resolved; the "Stage-2 lane keeping collapses to ~48 for everyone" claim now has five counterexamples.
- [[concepts/navsim-benchmark.md]] — second paper reporting both EPDMS columns; verifies two of WA-JEPA's correction deltas and contradicts two of its cohort assignments.
- [[concepts/selection-based-planning.md]] — the cleanest in-paper price of a scorer in the wiki: **+20.9 EPDMS on navhard**.
- [[concepts/diffusion-planner.md]] — three-branch phase-scheduled CFG with a Tweedie self-conditioning branch; guidance applied to a *world model's* input rather than the denoiser's own condition.
- [[concepts/intent-conditioned-planning.md]] — the intent↔future circular dependency and the two surrogates that break it.
- [[sources/wa-jepa.md]] — the other controlled objective-form experiment; agrees on ordering, disagrees on the sign of the regression term.
- [[sources/drivelaw.md]] — the t=10 collapse, corroborated here from the training side by $e_0=0.95$.
- [[sources/da-wam.md]] — files DriveFuture under "(b) loosely coupled"; the Tweedie branch says otherwise.
- [[sources/geowam.md]] — the wiki's other navhard leaderboard. The two tables share no method, so together they extend the split's coverage from 10 entries to 22.
