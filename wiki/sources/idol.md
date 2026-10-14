---
title: "IDOL: Inverse-Dynamics-Guided Future Prediction for End-to-End Autonomous Driving"
type: source-summary
sources: ["raw/papers/IDOL_ Inverse-Dynamics-Guided Future Prediction for End-to-End Autonomous Driving.md"]
related: [sources/maneuverflow.md, concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/navhard-ood-evaluation.md, concepts/selection-based-planning.md, concepts/bench2drive.md, concepts/evaluation-variance.md, concepts/inference-latency.md, concepts/perception-for-planning.md, sources/resworld.md, sources/diffusiondrive.md, sources/diffusiondrive-v2.md, sources/pair-drive.md, sources/drive-jepa.md, sources/drivewam.md, sources/futuresightdrive.md, sources/da-wam.md, sources/mm-future.md, sources/world4scorer.md, sources/drivefuture.md, sources/simscale.md, sources/hydra-mdp-pp.md, sources/drivevla-w0.md, sources/sgdrive.md, sources/hyworldvla.md, sources/geoworldad.md, sources/lwdrive.md, research-directions.md]
created: 2026-10-12
updated: 2026-10-14
confidence: medium
---

# IDOL: Inverse-Dynamics-Guided Future Prediction for End-to-End Autonomous Driving

**Paper**: IDOL: Inverse-Dynamics-Guided Future Prediction for End-to-End Autonomous Driving
**Authors**: Chenghao Zhang, Timin Li, Dongmei Li
**Orgs**: Department of Electronic Engineering, Tsinghua University
**arXiv**: 2605.31476v2 (v1 May 2026)
**Code**: released (github.com/Waterloo-ins/IDOL)
**Source**: `raw/papers/IDOL_ Inverse-Dynamics-Guided Future Prediction for End-to-End Autonomous Driving.md`. The clipping has all text and all 14 tables. Figure 3 (an SVG in the arXiv HTML) is missing from it and Figure 4 keeps only its first row. Figure 3 was rendered from the HTML's `visual.svg` to `raw/assets/idol_fig3.png`; Figure 4's left- and right-turn rows were downloaded as `idol_left2.png` and `idol_right2.png`.

**Confidence: medium.** The ablations are internally consistent and include a parameter-matched control, but they are single runs on navtest only. The navhard and NAVSIM-v2 margins compare rows that the closed-form check places on different evaluator scales (§[Table audit](#table-audit)), and the ResNet-34 "state of the art" omits three published ResNet-34 entries above it.

---

## What It Is

**A WoTE-style BEV world-model planner whose per-candidate future rollouts feed back into the candidate's own query.** The components are those of WoTE (not ingested), whose reward model IDOL uses to rank: a TransFuser backbone (ResNet-34, stitched front cameras + LiDAR) gives an 8×8 latent BEV; each of 256 offline trajectory anchors gets a planning query; a transformer world model rolls the latent BEV forward with that query injected along the anchor. IDOL adds:

1. **An "inverse dynamics model" (IDM)**: a three-stage MLP over each *ordered pair* of consecutive predicted BEV states of one candidate's rollout. It outputs a spatial dynamics map (8×8×256) and a pooled global feature, averaged over the rollout.
2. **Query refinement from the IDM output**: cross-attention from the candidate query to the spatial map, then MLN modulation by the global feature.
3. **One closed-loop iteration**: the refined query is re-anchored to the initial query and used to roll out the world model again, and the refinement is repeated once before an offset head decodes each candidate.
4. **A motion-correction loss on the IDM**: the global feature of the expert-nearest anchor must predict, through a training-only head, the difference between the expert's and the anchor's speed, acceleration and heading-rate sequences (Smooth L1, weight 0.2).

**Results**

| Benchmark | IDOL | Closest matched comparison |
|---|---:|---|
| NAVSIM-v1 PDMS (navtest, ResNet-34, C+L) | **90.0** | WoTE 88.3; own no-IDM base 87.3 |
| NAVSIM-v2 EPDMS (navtest) | 89.6 (corrected-like, residual −0.60) | MeanFuser 89.5 (corrected-like) |
| navhard two-stage EPDMS (camera-only "IDOL-LTF" variant) | 38.0 | WoTE 27.9 in the same table |
| navhard stage-1-only EPDMS | 76.2 | Hydra-MDP (V2-99) 73.1 |
| Bench2Drive DS / SR | 63.22 / 34.09% | WoTE 61.71 / 31.36% |
| Parameters / speed | 69.36M / 17.65 FPS (≈57 ms) on one RTX 3090, batch 1 | – |

---

## Key Takeaways

- **The mechanism has a parameter-matched control, which is rare here.** With the world model, refinement path, losses and training fixed, the refinement reads (Table 12): future states alone 88.6 PDMS, future states + a parameter-matched MLP 88.9, latent differences of consecutive states 89.0, differences + MLP 89.1, a learned encoder of ordered state pairs **90.0**. So transition information as a fixed difference is worth +0.4, and the learned pairwise encoder another +0.9 to +1.1 at equal parameters. Single runs.
- **Feeding per-candidate futures back into the candidates is the larger effect.** No feedback 87.3 → IDM feedback 89.2 → one more rollout-and-refine iteration 90.0. More iterations are worse (89.4, 89.4). This is a navtest effect of +1.3 to +2.7 for reading a per-candidate future, larger than the per-candidate-future effects the wiki records for scorers (DA-WAM +0.15, MM-Future +0.4; World4Scorer's future target +0.95), though not compute-matched against the no-feedback base.
- **"Inverse dynamics" is a framing more than a function.** Each rollout is conditioned on its anchor, so the action that produced the transition is known by construction. A classical inverse-dynamics model would recover the anchor's motion. IDOL's only direct supervision of the module is the *expert's* deviation from the anchor, a quantity the rollout under the anchor does not contain. The module is better described as a learned residual-correction head reading pairs of imagined states. No probe shows it encodes the transition's own motion, and the correction loss itself is not ablated.
- **The NAVSIM-v2 and navhard margins cross evaluators.** On navtest v2 IDOL's row is corrected-like (−0.60) while the shared baseline block and the human agent (90.3) are pre-fix-like (+1.3 to +4.1). On navhard stage-1-only, every baseline's reported score sits 1.8–7.6 points below its own sub-scores' closed form and IDOL's sits 0.35 above; by closed form Hydra-MDP (V2-99) leads IDOL 80.4 to 75.9, the reverse of the claimed +3.1. On two-stage navhard, IDOL's 38.0 is 2.6 above the product of its stage closed forms while every learned baseline is 0.2–2.2 below. The "+10.1 over WoTE" probably includes several points of evaluator difference.
- **"Best among ResNet-34 methods" omits three published ResNet-34 entries above it**: iPad 91.1 (camera only; cited in IDOL's related work), DiffusionDriveV2 91.2 (in IDOL's own v2 table, absent from its v1 table) and PaIR-Drive 91.2. All three are in the wiki.
- **On the wiki's ladders**: navhard 38.0 sits with PhysWAM (38.1) and SUV (36.9), below every V2-99 scorer and below GTRS-Dense on ResNet-34 (38.3 real-only, 46.9 with SimScale data; [[sources/simscale.md]]), and above every unscored ResNet-34 entry (World4Drive 34.9). It is a scorer-cohort entry, since WoTE's reward model is trained on five cached PDM metrics. Bench2Drive 63.22 DS is below every entry on [[concepts/bench2drive.md]] except UniAD and VAD. Its table compares only against those two, SeerDrive and WoTE.
- **Efficient for a per-candidate world model**: 256 rollouts on 64 BEV tokens, two passes, 69.4M parameters, about 57 ms on an RTX 3090. The rollout length $U$ is not stated anywhere.

---

## Method

![[Figure1.png|Figure 1: Comparison of three planning paradigms. (a) Traditional methods map sensor inputs directly to a trajectory. (b) World-model-based methods predict future latent BEV states that are only loosely connected to motion generation. (c) IDOL applies an inverse dynamics model to adjacent future latent BEV states to extract transition-aware motion cues for trajectory refinement.]]

**Figure 1**: Three planning paradigms. (a) Sensor → planner. (b) A world model predicts future latent BEV states that are "only loosely connected" to motion generation. (c) IDOL applies an inverse dynamics model to adjacent future BEV states and feeds the resulting motion cues back to the planner.

![[Figure2.png|Figure 2: Overview of IDOL. (a) The closed-loop refinement module rolls out future latent BEV states with the BEV world model and refines the planning query through inverse-dynamics feedback. (b) The planning network fuses the refined query with trajectory anchors, predicts candidates and selects one by scoring and ranking. (c) The inverse-dynamics-guided refinement network decodes adjacent future BEV states into spatial and global dynamics cues that update the ego query.]]

**Figure 2**: Overview. (a) Closed-loop refinement: the BEV world model rolls out future latent BEV states; the inverse-dynamics-guided refinement network updates the ego query, ×ℓ. (b) Planning: the refined query is fused with the anchors, candidates are predicted, scored and ranked. (c) The refinement network: an IDM over (BEV state $t$, BEV state $t{+}1$) gives a spatial dynamics map and a global dynamics feature; a spatial branch (cross-attention) and a global branch (MLN) update the query, and query fusion re-anchors it for the next iteration. The figure labels the image input "multi-view images"; the text uses a stitched front-view image at 256×1024.

### Scene and queries (§3.2)

TransFuser-style image–LiDAR backbone (ResNet-34) → latent BEV $B$, flattened to 64 tokens (8×8, dimension 256). Offline anchor vocabulary $\mathcal A=\{\tau_k\}_{k=1}^{K}$, $K=256$, $\tau_k\in\mathbb R^{8\times3}$ (4 s at 0.5 s; position and heading). Each anchor feature $a_k$ is joined with the ego-state feature $z$:

$$e_{k,\mathrm{init}}=\phi([z,a_k])$$

### Candidate-conditioned latent rollout (§3.3)

For candidate $k$ and iteration $\ell$, with $q^{(0)}_{k,\ell}=e^{(\ell)}_{k,\mathrm{in}}$ and $B^{(0)}_{k,\ell}=B$:

$$B^{(u)}_{k,\ell,\mathrm{cond}}=\mathcal I\big(B^{(u)}_{k,\ell},q^{(u)}_{k,\ell};\tau_k\big),\qquad \big[q^{(u+1)}_{k,\ell},B^{(u+1)}_{k,\ell}\big]=\mathrm{BEVWorldModel}\big(\big[q^{(u)}_{k,\ell},B^{(u)}_{k,\ell,\mathrm{cond}}\big]\big)$$

$\mathcal I$ injects the ego query at the anchor's future location. The rollout $\mathcal R^{(\ell)}_k=\{(q^{(u)}_{k,\ell},B^{(u)}_{k,\ell,\mathrm{cond}})\}_{u=0}^{U}$; $U$ is not given.

### Inverse dynamics and refinement (§3.4)

$$\big(S^{(u)}_{k,\ell},g^{(u)}_{k,\ell}\big)=\mathrm{IDM}\big(B^{(u)}_{k,\ell,\mathrm{cond}},B^{(u+1)}_{k,\ell,\mathrm{cond}}\big),\qquad S^{(\ell)}_k=\tfrac1U\textstyle\sum_u S^{(u)}_{k,\ell},\quad g^{(\ell)}_k=\tfrac1U\sum_u g^{(u)}_{k,\ell}$$

The IDM concatenates the two states' 64 query features channel-wise, then applies a three-stage MLP (fusion, then residual MLPs) that keeps the 8×8 layout; the global feature pools it.

$$e^{(\ell)}_{k,\mathrm{spa}}=\mathrm{LN}\big(e^{(\ell)}_{k,\mathrm{in}}+\mathrm{CrossAttn}(e^{(\ell)}_{k,\mathrm{in}},S^{(\ell)}_k,S^{(\ell)}_k)\big),\qquad e^{(\ell)}_{k,\mathrm{ref}}=\mathrm{MLN}_\mathrm{idm}\big(e^{(\ell)}_{k,\mathrm{spa}},g^{(\ell)}_k\big)$$

$$e^{(\ell+1)}_{k,\mathrm{in}}=\mathrm{MLN}_\mathrm{cl}\big(e_{k,\mathrm{init}},e^{(\ell)}_{k,\mathrm{ref}}\big),\qquad \ell=0,\ldots,L-2$$

The anchor never moves; offsets are decoded from $e^{(L-1)}_{k,\mathrm{ref}}$ by a transformer decoder and MLP, and **candidates are ranked by WoTE's reward model**.

### Losses (§3.5, Appendix B)

$$\mathcal L=\mathcal L_\mathrm{off}+0.1\,\mathcal L_\mathrm{off\text-im}+\mathcal L_\mathrm{im}+\mathcal L_\mathrm{sim}+10\,\mathcal L^\mathrm{cur}_\mathrm{map}+0.1\,\mathcal L^\mathrm{fut}_\mathrm{map}+0.2\,\mathcal L_\mathrm{corr}$$

- $\mathcal L_\mathrm{off}$: L1 on the offset of the winner-take-all anchor.
- $\mathcal L_\mathrm{im}$, $\mathcal L_\mathrm{off\text-im}$: reward heads against a soft target from anchor-to-expert distances.
- $\mathcal L_\mathrm{sim}$: binary cross-entropy over **five cached PDM metrics**, so the ranking head is trained on the benchmark's sub-scores ([[concepts/selection-based-planning.md]]).
- $\mathcal L_\mathrm{map}$: focal loss on current (weight 10) and future (weight 0.1) BEV semantics. Which candidates' rollouts the future term supervises is not stated.
- $\mathcal L_\mathrm{corr}$:

$$\mathcal L_\mathrm{corr}=\ell_\mathrm{SL1}\Big(h_\mathrm{mot}\big(g^{(L-1)}_{k^\star}\big),\ \mathrm{sg}\big[E_\mathrm{mot}(\tau^\mathrm{GT})-E_\mathrm{mot}(\tau_{k^\star})\big]\Big),\qquad E_\mathrm{mot}(\tau)=\big[v_{1:S}/\sigma_v,\ a_{1:S-1}/\sigma_a,\ \omega_{1:S}/\sigma_\omega\big]$$

with $(\sigma_v,\sigma_a,\sigma_\omega)=(10\ \mathrm{m/s},3\ \mathrm{m/s^2},0.5\ \mathrm{rad/s})$, heading rates masked below 0.5 m/s, only the winning anchor supervised, and $h_\mathrm{mot}$ dropped at inference.

**Training**: 30 epochs, 4 RTX 3090s, batch 4 per GPU, about 24 h, AdamW at 2e-4, weight decay 1e-4, NAVSIM training split only.

### Appendix tables 6–8

**Table 6** (notation): $K,S$ anchors and poses; $U,L$ rollout steps and refinement iterations; $B$ current latent BEV; $\mathcal A$ anchors (positions and headings); $\tau^\mathrm{GT}$ expert trajectory; $a_k,z$ anchor and ego-state features; $e_{k,\mathrm{init}}$ initial query; $e^{(\ell)}_{k,\mathrm{in}},e^{(\ell)}_{k,\mathrm{ref}}$ query before and after refinement; $q^{(u)}_{k,\ell}$ rollout ego query; $B^{(u)}_{k,\ell,\mathrm{cond}}$ query-injected latent BEV; $S^{(\ell)}_k,g^{(\ell)}_k$ aggregated IDM features; $k^\star$ expert-nearest anchor; $E_\mathrm{mot}$ fixed motion representation; $h_\mathrm{mot}$ training-only head; $\mathrm{sg}$ stop-gradient.

**Table 7** (hyperparameters):

| Parameter | Value | Parameter | Value |
|---|---|---|---|
| Epochs | 30 | Planning horizon | 4 s |
| Image input | 256×1024 | Waypoint interval | 0.5 s |
| BEV latent resolution | 8×8 | Future waypoints | 8 |
| BEV queries | 64 | Anchor vocabulary | 256 |
| Latent dimension | 256 | Closed-loop iterations | 1 |
| Learning rate | 2e-4 | Weight decay | 1e-4 |

**Table 8** (loss coefficients): offset 1.0; auxiliary imitation reward 0.1; main imitation reward 1.0; simulator-metric reward 1.0; current BEV semantics 10.0; future BEV semantics 0.1; IDM motion correction 0.2.

---

## Results

### Table 1: NAVSIM v1 navtest (ResNet-34 image backbone)

Residual = closed form − reported (computed here).

| Method | Backbone | Input | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ | Residual |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| VADv2 | ResNet-34 | C & L | 97.2 | 89.1 | 91.6 | 100 | 76.0 | 80.9 | −6.0 |
| UniAD | ResNet-34 | Camera | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 | −4.1 |
| PARA-Drive | ResNet-34 | Camera | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 | −4.0 |
| TransFuser | ResNet-34 | C & L | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 | −3.9 |
| LAW | ResNet-34 | Camera | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 | −4.0 |
| DiffusionDrive | ResNet-34 | C & L | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 | −2.7 |
| WoTE | ResNet-34 | C & L | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 | −2.2 |
| SeerDrive | ResNet-34 | C & L | 98.4 | 97.0 | 94.9 | 99.9 | 83.2 | 88.9 | −2.2 |
| ResWorld\* (history frames) | ResNet-34 | C & L | 98.9 | 96.5 | 95.6 | 100 | 83.1 | 89.0 | −2.0 |
| MeanFuser | ResNet-34 | Camera | 98.6 | 97.0 | 95.0 | 100 | 82.8 | 89.0 | −2.2 |
| DiffE2E† (ResNet-34 variant) | ResNet-34 | C & L | 99.2 | 96.8 | 96.7 | 100 | 83.6 | 89.8 | −1.7 |
| **IDOL** | ResNet-34 | C & L | 98.7 | 97.7 | 95.5 | 100 | 84.0 | **90.0** | −1.8 |

**Not in the table, also ResNet-34**: iPad 91.1 (camera; [[sources/drive-jepa.md]]'s table), DiffusionDriveV2 91.2 ([[sources/diffusiondrive-v2.md]]; GRPO), PaIR-Drive 91.2 single-plan ([[sources/pair-drive.md]]; GRPO). The paper's v2 table marks RL methods with †, but its v1 table neither includes nor excludes them by rule.

### Table 2: NAVSIM v2 navhard, two-stage

Combined EPDMS as printed. The last two columns are computed here: the closed form of each stage's sub-scores, and their product (an approximation of the combined score; [[concepts/navhard-ood-evaluation.md]] bounds its error at roughly 0.5–3 points).

| Method | Backbone | Stage | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS | Stage closed form | Product − reported |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PDM-Closed (privileged) | – | 1 | 94.4 | 98.8 | 100.0 | 99.5 | 100.0 | 93.5 | 99.3 | 87.7 | 36.0 | | 82.0 | |
| | | 2 | 88.1 | 90.6 | 96.3 | 98.5 | 100.0 | 83.1 | 73.7 | 91.5 | 25.4 | 51.3 | 61.4 | −1.0 |
| LTF | ResNet-34 | 1 | 96.2 | 79.5 | 99.1 | 99.5 | 84.1 | 95.1 | 94.2 | 97.5 | 79.1 | | 67.8 | |
| | | 2 | 77.7 | 70.2 | 84.2 | 98.0 | 85.1 | 75.6 | 45.4 | 95.7 | 75.9 | 23.1 | 34.8 | +0.5 |
| GTRS-DP | ResNet-34 | 1 | 94.7 | 78.8 | 96.1 | 99.5 | 83.0 | 94.4 | 92.0 | 97.5 | 72.8 | | 63.0 | |
| | | 2 | 80.3 | 74.4 | 84.9 | 98.0 | 81.9 | 78.8 | 45.4 | 96.7 | 70.1 | 23.8 | 38.2 | +0.2 |
| DiffusionDrive | ResNet-34 | 1 | 96.0 | 79.7 | 97.4 | 99.5 | 81.3 | 93.1 | 90.8 | 96.8 | 73.8 | | 64.6 | |
| | | 2 | 82.1 | 72.2 | 88.5 | 98.7 | 85.1 | 78.8 | 49.2 | 89.3 | 71.2 | 24.2 | 40.1 | +1.7 |
| GuideFlow | ResNet-34 | 1 | 96.6 | 80.5 | 96.3 | 99.3 | 82.3 | 94.9 | 91.5 | 97.7 | 67.8 | | 65.1 | |
| | | 2 | 87.3 | 76.7 | 88.8 | 99.2 | 84.3 | 85.1 | 49.7 | 93.1 | 44.5 | 27.1 | 45.0 | +2.2 |
| WoTE | ResNet-34 | 1 | 97.4 | 88.2 | 97.7 | 99.3 | 82.7 | 96.4 | 90.8 | 97.3 | 68.0 | | 73.3 | |
| | | 2 | 81.2 | 77.7 | 84.8 | 98.1 | 85.9 | 78.5 | 46.2 | 96.6 | 63.3 | 27.9 | 40.5 | +1.8 |
| **IDOL** (camera-only LTF variant) | ResNet-34 | 1 | 97.2 | 89.6 | 98.0 | 99.6 | 82.3 | 96.9 | 95.3 | 97.6 | 72.9 | | 75.9 | |
| | | 2 | 84.9 | 81.7 | 87.7 | 98.5 | 84.7 | 81.1 | 50.0 | 95.4 | 64.0 | **38.0** | 46.7 | **−2.6** |

The five learned baselines' reported scores sit 0.2–2.2 *below* the product of their stage closed forms; IDOL's sits 2.6 *above*. The baseline rows are the DriveFuture / SpanVLA lineage (LTF 23.1, DiffusionDrive 24.2, GuideFlow 27.1). WoTE's navhard row is new to the wiki. For the two-stage run IDOL is retrained as a camera-only "IDOL-LTF" (a learnable latent replaces LiDAR, initialized from an LTF checkpoint) because "point-cloud inputs are unavailable in stage 2".

### Table 9 (Appendix): NAVSIM v2 navtest

| Method | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS | Residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Human Agent | 100.0 | 100.0 | 99.8 | 100.0 | 87.4 | 100.0 | 100.0 | 98.1 | 90.1 | 90.3 | +4.10 |
| Ego Status MLP | 93.1 | 77.9 | 92.7 | 99.6 | 86.0 | 91.5 | 89.4 | 98.3 | 85.4 | 64.0 | −4.00 |
| TransFuser | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 | +1.28 |
| Hydra-MDP++ | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 | +2.16 |
| DriveSuprim | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 | +2.28 |
| ReCogDrive \*,† | 98.3 | 95.2 | 99.5 | 99.8 | 87.1 | 97.5 | 96.6 | 98.3 | 86.5 | 83.6 | +2.69 |
| DiffusionDrive | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 | +2.48 |
| GTRS-Dense + SimScale ‡,† | 98.4 | 98.8 | 99.4 | 99.9 | 87.9 | 98.1 | 96.4 | 97.6 | 58.8 | 84.6 | +2.02 |
| World4Drive | 97.8 | 96.3 | 99.4 | 99.8 | 88.3 | 97.1 | 97.7 | 98.0 | 53.9 | 84.8 | −1.52 |
| Epona | 97.1 | 95.7 | 99.3 | 99.7 | 88.6 | 96.3 | 97.0 | 98.0 | 67.8 | 85.1 | −1.72 |
| DiffusionDriveV2 † | 97.7 | 96.6 | 99.2 | 99.8 | 88.9 | 97.2 | 96.0 | 97.8 | 91.0 | 85.5 | +2.10 |
| VADv2 ⚠ | 98.0 | 98.3 | 99.4 | 99.8 | 87.1 | 96.8 | 95.2 | 98.3 | 88.1 | 85.8 | +2.76 |
| DriveVLA-W0 \* | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 | −1.31 |
| SGDrive \*,‡ | 98.6 | 94.3 | 99.5 | 99.9 | 86.0 | 97.9 | 96.1 | 98.3 | 85.9 | 86.2 | −0.70 |
| DiffRefiner | 98.5 | 97.4 | 99.6 | 99.8 | 87.6 | 97.7 | 97.7 | 98.3 | 86.2 | 86.2 | +2.66 |
| WorldRFT † | 97.8 | 96.5 | 99.5 | 99.8 | 88.5 | 97.0 | 97.4 | 98.1 | 69.1 | 86.7 | −1.38 |
| SafeDrive | 99.5 | 99.0 | 99.7 | 99.9 | 88.6 | 98.9 | 97.5 | 98.2 | 81.9 | 87.5 | +4.03 |
| MeanFuser | 98.3 | 97.2 | 99.6 | 99.8 | 87.6 | 97.4 | 97.3 | 98.3 | 88.2 | 89.5 | −0.90 |
| **IDOL** | 98.8 | 97.6 | 99.5 | 99.8 | 87.1 | 98.3 | 96.3 | 98.3 | 85.5 | **89.6** | **−0.60** |

\* VLA/VLM, † RL or reward-guided training, ‡ extra synthetic, pseudo-expert or QA data (the paper's marks). Residual = closed form − reported. ⚠ The "VADv2" row matches no other record: VADv2 scores 80.9 PDMS with DAC 89.1 in IDOL's own Table 1, and 76.6 under the Hydra-MDP++ formula; a DAC of 98.3 here is not its profile. ReCogDrive's DDC and HC are swapped relative to every other wiki copy (98.3 / 99.5). *(2026-10-14: [[sources/maneuverflow.md]] prints an ICLR 2026 VADv2 at 89.3 PDMS v1 (98.3 / 97.4 / 95.7 / 100 / 82.3), whose sub-scores are compatible with this row; the two are plausibly the v1 and v2 results of an updated VADv2, distinct from the original 80.9.)*

### Table 10 (Appendix): navhard stage 1 only

| Method | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS | Closed form | Residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TransFuser | 96.3 | 74.6 | 98.4 | 99.3 | 82.9 | 93.7 | 92.7 | 97.5 | 78.2 | 60.5 | 62.3 | +1.8 |
| LTF | 96.2 | 79.5 | 99.1 | 99.5 | 84.1 | 95.1 | 94.2 | 97.5 | 79.1 | 62.3 | 67.8 | +5.5 |
| DiffusionDrive | 95.9 | 84.0 | 98.6 | 99.8 | 84.4 | 96.0 | 95.1 | 97.6 | 71.1 | 63.2 | 70.8 | +7.6 |
| WoTE | 97.4 | 88.2 | 97.8 | 99.3 | 82.7 | 96.4 | 90.9 | 97.3 | 68.0 | 66.7 | 73.4 | +6.7 |
| Hydra-MDP† (V2-99) | 97.6 | 96.4 | 99.2 | 99.3 | 80.2 | 96.9 | 94.9 | 97.8 | 58.7 | 73.1 | **80.4** | +7.3 |
| **IDOL** | 97.2 | 89.6 | 98.0 | 99.6 | 82.3 | 96.9 | 95.3 | 97.6 | 72.9 | **76.2** | 75.9 | **−0.4** |

IDOL's stage-1 sub-scores are identical to its Table 2 stage-1 row, so this is the camera-only variant. DiffusionDrive here has different stage-1 sub-scores from Table 2's (DAC 84.0 vs 79.7), so the paper carries two DiffusionDrive evaluations.

### Table 11 (Appendix): Bench2Drive

| Method | DS ↑ | SR ↑ |
|---|---:|---:|
| VAD | 42.35 | 15.00% |
| UniAD-Base | 45.81 | 16.36% |
| SeerDrive | 58.32 | 30.17% |
| WoTE | 61.71 | 31.36% |
| **IDOL** | **63.22** | **34.09%** |

Training data, sensor setup and closed-loop adaptation for CARLA are not described.

---

## Ablations (navtest, single runs)

### Table 3: IDM and closed-loop iterations

| IDM | CL iters | NC | DAC | TTC | Comf. | EP | PDMS |
|:-:|:-:|---:|---:|---:|---:|---:|---:|
| × | × | 98.5 | 95.0 | 95.2 | 100 | 81.5 | 87.3 |
| ✓ | × | 98.6 | 97.0 | 95.4 | 100 | 83.3 | 89.2 |
| ✓ | 1 | 98.7 | 97.7 | 95.5 | 100 | 84.0 | **90.0** |
| ✓ | 2 | 98.6 | 97.3 | 95.2 | 100 | 83.5 | 89.4 |
| ✓ | 3 | 98.8 | 97.2 | 95.7 | 100 | 83.1 | 89.4 |

The +2.7 is mostly DAC (+2.7) and EP (+2.5); NC and TTC barely move. The no-IDM base is 1.0 below WoTE's published 88.3; the paper does not say how the base differs from WoTE.

### Table 4: IDM temporal input

| Input | NC | DAC | TTC | Comf. | EP | PDMS |
|---|---:|---:|---:|---:|---:|---:|
| 4 frames | 98.7 | 97.3 | 95.6 | 100 | 83.7 | 89.6 |
| **2 frames** (adjacent) | 98.7 | 97.7 | 95.5 | 100 | 84.0 | **90.0** |

### Table 5: Spatial and global branches

| Branches | NC | DAC | TTC | Comf. | EP | PDMS |
|---|---:|---:|---:|---:|---:|---:|
| w/o spatial | 98.8 | 96.9 | 95.8 | 100 | 83.2 | 89.3 |
| w/o global | 98.7 | 97.2 | 95.3 | 100 | 83.9 | 89.5 |
| **Both** | 98.7 | 97.7 | 95.5 | 100 | 84.0 | **90.0** |

### Table 12: Transition modeling, parameter-matched

| Transition modeling | Params (M) | NC | DAC | TTC | Comf. | EP | PDMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Future state only | 68.2 | 98.4 | 96.6 | 95.2 | 100 | 82.6 | 88.6 |
| Future state + parameter-matched MLP | 69.4 | 98.5 | 97.0 | 94.7 | 100 | 83.3 | 88.9 |
| Latent difference | 68.2 | 98.6 | 96.8 | 95.7 | 100 | 82.8 | 89.0 |
| Latent difference + parameter-matched MLP | 69.4 | 98.7 | 97.1 | 95.8 | 100 | 82.6 | 89.1 |
| **IDM** (ordered pair, learned) | 69.4 | 98.7 | 97.7 | 95.5 | 100 | 84.0 | **90.0** |

### Table 13: Global-feature fusion

| Fusion | NC | DAC | TTC | Comf. | EP | PDMS |
|---|---:|---:|---:|---:|---:|---:|
| Additive | 98.7 | 97.2 | 95.7 | 100 | 83.5 | 89.5 |
| Concat-MLP | 98.7 | 97.6 | 95.5 | 100 | 83.8 | 89.8 |
| **MLN** | 98.7 | 97.7 | 95.5 | 100 | 84.0 | **90.0** |

### Table 14: Anchor vocabulary size

| $K$ | NC | DAC | TTC | Comf. | EP | PDMS |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 98.7 | 97.6 | 95.5 | 100 | 83.8 | 89.8 |
| **256** | 98.7 | 97.7 | 95.5 | 100 | 84.0 | **90.0** |
| 512 | 98.7 | 97.2 | 95.3 | 100 | 84.0 | 89.6 |

**Reading the ablations.** All ablation residuals lie between −1.78 and −2.81, so sub-scores and aggregates agree. Every IDM variant lands in 88.6–90.0 and comfort is 100 in every row. Two different configurations, Concat-MLP fusion (Table 13) and $K=128$ (Table 14), have identical numbers in all six columns; possible by chance at one decimal, but worth noting next to [[concepts/evaluation-variance.md#column-regularity]]. Not ablated: the motion-correction loss $\mathcal L_\mathrm{corr}$, the rollout length $U$, the future-BEV loss, and the reward-model ranking.

---

## Visualizations

![[idol_fig3.png|Figure 3: Qualitative visualization of inverse-dynamics-guided refinement on navtest. (a) Latent BEV change between adjacent imagined states. (b) IDM spatial response, with white contours marking high-change regions from (a). (c) Refinement-query attention over the IDM map. (d) Original, refined and expert trajectories.]]

**Figure 3**: Inverse-dynamics-guided refinement on one navtest left turn. (a) latent BEV change between adjacent imagined states; (b) IDM spatial response, white contours = high-change regions from (a); (c) the refinement query's attention over the IDM map; (d) original, refined and expert trajectories. *(Rendered from the arXiv HTML's SVG.)*

![[straight2.png|Figure 4, row 1: straight driving. Panels (a)-(d) as in Figure 3.]]

![[idol_left2.png|Figure 4, row 2: left turn. Panels (a)-(d) as in Figure 3.]]

![[idol_right2.png|Figure 4, row 3: right turn. Panels (a)-(d) as in Figure 3.]]

**Figure 4**: Straight, left turn and right turn (top to bottom); panels as in Figure 3.

*(Wiki note on Figures 3–4: the heat maps cover only the area ahead of the ego, consistent with an 8×8 forward BEV. In each case the IDM response and the query attention concentrate on one blob that overlaps or touches one of the high-change contours, while other contours get little response; the paper reads this as motion-selective. In every (d) panel the original and refined trajectories nearly overlap, so the per-scene refinement is small; the aggregate +0.8 for the iteration is the evidence, not the figures.)*

![[figure_failure_cases.png|Figure 5: Representative failure cases on navtest. (a) A turning area with ambiguous road geometry: the refined trajectory heads right toward a gated entrance while the expert turns left. (b) A roundabout: the refined trajectory captures part of the motion trend but stays offset, cutting across the entry.]]

**Figure 5**: Failure cases. (a) Ambiguous turning area: the refined trajectory takes the wrong direction (right, toward a gate) where the expert turns left. (b) A roundabout: the trajectory follows the trend but cuts across the entry instead of following the lane. The paper's reading: when the maneuver intention is wrong, refinement cannot recover it, and topology-sensitive scenes need lane-connectivity reasoning.

---

## Table Audit {#table-audit}

1. **Navhard stage-1-only (Table 10) is on two scales.** Every baseline's reported score is 1.8–7.6 below the closed form of its own sub-scores; IDOL's is 0.4 above. On the closed form Hydra-MDP (V2-99) scores 80.4 against IDOL's 75.9, so the claimed +3.1 lead reverses. The baselines' values most likely come from another source and evaluator; the paper names neither.
2. **Two-stage navhard (Table 2) shows the same split, less sharply**: learned baselines 0.2–2.2 below the product of their stage closed forms, IDOL 2.6 above (PDM-Closed 1.0 above). If the baselines were on IDOL's scale, WoTE would be roughly 32 rather than 27.9, and the +10.1 margin roughly halved. That is an estimate from an approximate check.
3. **NAVSIM-v2 navtest (Table 9) mixes evaluators**, like most v2 tables in the wiki. Corrected-like: IDOL (−0.60), MeanFuser (−0.90), SGDrive, DriveVLA-W0, WorldRFT, World4Drive, Epona. Pre-fix-like: the human agent (90.3, +4.10), the shared baseline block, GTRS-Dense + SimScale, DiffusionDriveV2 (85.5; WA-JEPA attributes 87.5 to the corrected evaluator), DiffRefiner, SafeDrive (+4.03, as in World4Scorer's table) and the "VADv2" row. "Ranking first" (89.6 vs MeanFuser 89.5) is a same-scale comparison; "competitive with the human agent" compares a corrected-like row with the pre-fix human (the corrected human agent is 94.5). The table carries five of the shared baseline block's six rows (all but ARTEMIS), the eleventh such table recorded in [[concepts/navsim-benchmark.md]].
4. **v1 scope.** "Best PDMS among ResNet-34 methods" omits iPad 91.1, DiffusionDriveV2 91.2 and PaIR-Drive 91.2.
5. **Row details**: the unidentified "VADv2 85.8" row; ReCogDrive DDC/HC swapped; TransFuser LK 92.7 against 92.4 elsewhere; GTRS-Dense + SimScale within 0.8 of SimScale's own ResNet-34 reward-label row on every sub-score but not identical (EC 58.8 vs 58.0). WoTE's v1 TTC is 94.9 here and in the other wiki copies; [[sources/hyworldvla.md]]'s 94.4 is the outlier.

---

## Limitations

1. **Single runs on navtest only** for every ablation; no seeds, no navhard ablations.
2. **The "inverse dynamics" interpretation is untested.** The module's direct target is the expert's correction to the anchor, not the transition's own motion; no probe, no ablation of $\mathcal L_\mathrm{corr}$.
3. **Evaluator mismatches** inflate the navhard and v2 margins (§[Table audit](#table-audit)).
4. **The rollout length $U$ is unreported**, as are which candidates the future-BEV loss supervises and how the no-IDM base differs from WoTE.
5. **A different model for navhard.** The two-stage and stage-1-only results use a camera-only variant whose navtest score is not reported, so navtest and navhard do not describe the same checkpoint.
6. **Bench2Drive without protocol**: training data, sensors and adaptation unstated; 63.22 DS is low on the wiki's ladder.
7. **Privileged supervision**: the ranking head is trained on five cached PDM metrics and the BEV on ground-truth semantics.
8. **Failure modes are maneuver-level** (wrong turn direction, roundabout topology), which per-candidate refinement around fixed anchors cannot correct.

---

## Relation to Other Wiki Pages

- WoTE (not ingested): the base. IDOL keeps its anchors, BEV world model and reward-model ranking. WoTE is now load-bearing in one ingested paper and a baseline in many.
- [[sources/da-wam.md]], [[sources/mm-future.md]], [[sources/world4scorer.md]]: the other per-candidate future designs. They use per-candidate futures to *score* candidates (+0.15, +0.4; World4Scorer's states are outcome-supervised). IDOL uses them to *refine* each candidate before scoring, and measures +1.3 to +2.7 for doing so.
- [[sources/drivewam.md]], [[sources/futuresightdrive.md]]: "inverse dynamics" in the wiki's earlier sense, an action read from a future generated *without* that action. IDOL's futures are generated *with* the anchor, which is why its module cannot be inverse dynamics in that sense.
- [[sources/maneuverflow.md]] *(2026-10-14)*: the nearest design on a similar ResNet-34 TransFuser backbone. One shared action-free BEV forecast sampled along each of 64 generated candidates, instead of a rollout per anchor; a Frenet refiner on the best 16, then a re-score. Future evidence is worth +1.58 / +0.55 there against +1.3 to +2.7 for feedback here; 90.58 / 90.14 against 90.0 / 89.6. It cites WoTE, not IDOL.
- [[sources/geoworldad.md]], [[sources/lwdrive.md]]: other iterative refiners (five stages, six stages). IDOL finds one refinement iteration best and two or three worse.
- [[sources/resworld.md]]: also an action-side reading of predicted BEV change (temporal residuals as input); IDOL prints its history variant at 89.0.
- [[sources/drivefuture.md]], [[sources/simscale.md]]: sources of IDOL's lineage-B navhard baselines and its GTRS-Dense + SimScale v2 row.
- [[sources/diffusiondrive-v2.md]], [[sources/pair-drive.md]], [[sources/drive-jepa.md]] (iPad): the omitted ResNet-34 entries above 90.0.

---

## Open Questions

- **What does a transition encoder over candidate-conditioned futures encode?** A probe on the IDM features (do they predict the anchor's own motion, the expert correction, or neither?) and an ablation of $\mathcal L_\mathrm{corr}$ would say whether the module does inverse dynamics or residual correction.
- **Does per-candidate refinement survive a matched evaluator on navhard?** Rescoring the baselines with IDOL's evaluator, or IDOL with theirs, would settle the size of the navhard margin.
- **Why does a second iteration hurt?** Two and three iterations score 89.4. Whether this is over-correction, as the paper suggests, or train–test mismatch from training with one iteration is not tested.
