---
title: "ReWorld: Representation Learning for World Action Models"
type: source-summary
sources: [raw/papers/ReWorld_ Representation Learning for World Action Models.md]
related: [concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/foundation-backbones-for-ad.md, concepts/selection-based-planning.md, concepts/diffusion-planner.md, concepts/nuscenes-waymo-evals.md, concepts/rl-for-ad.md, sources/drivelaw.md, sources/adaptive-wam.md, sources/brainwam.md, sources/foresight.md, sources/simwam.md, sources/da-wam.md, sources/wa-jepa.md, sources/drive-jepa.md, sources/auto-jepa.md, sources/geoworldad.md, sources/lwdrive.md, sources/epona.md, sources/uniugp.md, sources/drivevla-w0.md, sources/policy-world-model.md, sources/drivewam.md, sources/driveva.md, sources/diffusiondrive.md, sources/recogdrive.md, sources/drivesuprim.md]
created: 2026-09-04
updated: 2026-09-04
confidence: high
---

**Paper**: ReWorld: Representation Learning for World Action Models
**Authors**: Tianze Xia, Lijun Zhou (equal contribution; Lijun Zhou project lead), Kaixin Xiong, Jingfeng Yao, Zhenxin Zhu, Haiyang Sun, Bing Wang, Guang Chen, Wenyu Liu, Hangjun Ye, Xinggang Wang (corresponding)
**Orgs**: Huazhong University of Science and Technology + Xiaomi EV
**arXiv**: 2606.27504v2
**Code**: https://github.com/xiaomi-research/ReWorld

---

## Summary

ReWorld is **the direct sequel to [[sources/drivelaw.md]] from the same group** — same first author, same project lead, same 2B LTX-Video DiT + 133M Action DiT, same chained latent interface — and it asks the question DriveLaW's architecture left open. DriveLaW established that a planner can read a video generator's mid-denoising states instead of decoded pixels. ReWorld observes that those states are then supervised *only through the final generation output*, and names the gap the **representation bottleneck of WAMs**: the latents connecting world prediction to action generation are never explicitly optimized to be future-predictive, cross-modally grounded, or sensitive to closed-loop behaviour quality.

Three objectives, all derived from the model's own targets — no external encoder, no teacher branch, **1.003× per-step video training cost**:

| Stage | Objective | What it constrains | Measured effect |
|---|---|---|---|
| 1 | $\mathcal{L}_{\mathrm{Mid}}$ — auxiliary velocity head on Video DiT **block 8** predicting the same flow target as the final head | intermediate video states | FVD 81.3 → 78.9; ~2× convergence speedup; enables self-guided sampling → **61.9** |
| 2 | $\mathcal{L}_{\mathrm{align}}$ — cosine alignment of post-cross-attention Action DiT states to their **stop-gradient attended video readout**, layer 12 | action states retain what they retrieved | PDMS 89.1 → **89.5** |
| 3 | $\mathcal{L}_{\mathrm{RDE}}$ — repulsion from the **nearest low-scoring trajectory** (PDM score < 0.6) among 64 simulator-scored candidates | action space separates expert from nearby-unsafe | PDMS 89.1 → **89.8**; with align → **90.4** |

Headlines: **FVD 81.3 → 61.9 and FID 4.6 → 4.4 on nuScenes; PDMS 89.1 → 90.4 on NAVSIM-v1 with no RL and no inference-time scorer; UCF-101 frozen linear probe 68.3% → 80.2%.**

**The reading this page arrives at is that ReWorld is two good contributions marketed as one curriculum, and the connective claim in its title is the one thing it does not measure.**

- **The video half decomposes badly for the abstract.** Of the 19.4 FVD improvement, the representation objective is worth **2.4** (81.3 → 78.9) and the inference-time self-guidance is worth **17.0** (78.9 → 61.9). The paper reports the $\gamma=1.0$ row honestly; the abstract leads with 61.9. Self-guidance is a sampling trick — structurally identical in form to classifier-free guidance, adapted from a cited CVPR'26 precedent — and it does not touch the planner-facing interface at all, which the paper states outright.
- **The planning half never involves the video representation.** Table VII's rows are DriveLaW 89.1 → +align 89.5 → +RDE 89.8 → both 90.4. They account for the entire +1.3 between them. **There is no row measuring Stage 1's contribution to PDMS**, and the implementation section says Stage 2 *"initializes from the DriveLaW checkpoint"* — which either contradicts the sequential curriculum described in §III-F or means the planning result is built on the un-improved video model. Either way, the paper offers no evidence that a more future-predictive world representation produces a better plan.

Read against the wiki's running argument, that is not a small omission — it is the sharpest available evidence that **video-generation quality and planning quality are decoupled inside a single chained WAM.** See [below](#decoupling).

---

## Positioning

![[ReWorld_framework.drawio_compressed.png|ReWorld overview: a Video DiT whose mid-denoising states condition an Action DiT, with three representation objectives applied in three stages]]

**Figure 1**: A Video DiT learns a latent representation of future scene evolution, whose mid-denoising states $\mathcal{F}$ condition an Action DiT for trajectory generation. Stage 1 makes intermediate video states future-predictive through $\mathcal{L}_{\mathrm{Mid}}$ and enables self-guided video sampling. Stage 2 freezes the Video DiT and aligns post-cross-attention action states with their attended video readouts through $\mathcal{L}_{\mathrm{align}}$. Stage 3 jointly fine-tunes both branches with $\mathcal{L}_{\mathrm{RDE}}$.

The framing is a refinement of DriveLaW's own taxonomy rather than a replacement for it. DriveLaW argued that "unified" world models keep video and action as **parallel output streams** and that chaining fixes the representation disconnect. ReWorld accepts that and adds: *latent access alone does not ensure effective representations.*

> "With standard output-level objectives, intermediate Video DiT states are supervised only through the final denoising output, and Action DiT states only through the final trajectory prediction. Consequently, video states may lack explicit future-predictive structure, while action states may fail to retain the world information retrieved through cross-attention."

The related-work section is unusually useful for this wiki because it maps the **diffusion representation-learning** literature onto driving for the first time here: latent-space reshaping (RAE, VA-VAE, VFM-VAE, FAE), intermediate-feature objectives (REPA and its extensions; SRA's teacher-free self-alignment), and inference-time latent refinement (Latent Forcing). ReWorld's claim is that these were built for **image** generation and transfer poorly to multi-second driving video — a claim it then tests directly in Table IV, and largely wins.

---

## Method

### Preliminaries — the inherited interface

Unchanged from [[sources/drivelaw.md]]: a 2B Video DiT initialized from LTX-Video over a highly compressed causal VAE with hybrid late-stage pixel decoding, and a 133M Action DiT. Both branches are rectified-flow, with **independently sampled** flow times $t_v$ and $t_a$.

$$\mathcal{L}_{\mathrm{Gen}}=\mathbb{E}\left[\left\|v_{\theta}^{z}(z_{t_{v}},t_{v},c^{v})-(\epsilon_{z}-z_{0})\right\|_{2}^{2}\right],\qquad \mathcal{L}_{\mathrm{FM}}=\mathbb{E}\left[\left\|v_{\phi}^{a}(a_{t_{a}},t_{a},c^{a},\mathcal{F})-(\epsilon_{a}-a_{0})\right\|_{2}^{2}\right]$$

The planner conditions on $\mathcal{F}=\{f^{(b)}\}_{b=1}^{B}$, the block features retained **at the first discrete reverse-flow step** — the maximum-noise endpoint. This is DriveLaW's t=1 finding operationalised as an architecture. The paper is careful about a distinction the wiki has had to make repeatedly: *"'cached' means that these activations are reused across action flow-matching steps, rather than precomputed or detached offline."* No future video is ever decoded during planning.

### Stage 1 — Future-predictive world representations

A lightweight head $q_l$ on selected blocks predicts the **same** velocity target as the final generation head:

$$\hat{v}_{t_{v}}^{(l)}=q_{l}\!\left(h_{t_{v}}^{(l)}\right),\qquad \mathcal{L}_{\mathrm{Mid}}=\frac{1}{|\mathcal{S}|}\sum_{l\in\mathcal{S}}\mathbb{E}\left[\left\|\hat{v}_{t_{v}}^{(l)}-(\epsilon_{z}-z_{0})\right\|_{2}^{2}\right],\qquad \mathcal{S}=\{8\}$$

$$\mathcal{L}_{\mathrm{Video}}=\mathcal{L}_{\mathrm{Gen}}+\lambda_{\mathrm{Mid}}\mathcal{L}_{\mathrm{Mid}}$$

**This is the whole idea and it is genuinely cheap**: the target already exists, the activations already exist, so the only added cost is one head. Hence 1.003×, against ~1.4× for methods that need a second DiT forward pass and ~1.7× for external-encoder alignment.

**Self-guided sampling** is an *inference* mechanism the training objective enables, adapted from a cited CVPR'26 precedent on guiding a DiT with its own internal dynamics:

$$v_{w}=v_{i}+\gamma\left(v_{f}-v_{i}\right),\qquad \gamma=1.4$$

The intermediate head's prediction $v_i$ plays the role of the weak model and the final head's $v_f$ the strong one — **structurally the classifier-free-guidance form**, with the discrepancy between depths substituting for the conditional/unconditional gap. The paper is explicit that this changes nothing about the planner: *"self-guidance improves video sampling without changing the planner-facing latent interface."*

### Stage 2 — World-grounded action representations

The premise is that cross-attention *access* does not imply *retention*. For action token $i$ at cross-attention layer $k$, the output-projected readout is $r_{i}^{(k)}=\sum_{j}\alpha_{ij}^{(k)}v_{j}^{(k)}$, and

$$\mathcal{L}_{\mathrm{align}}=\frac{1}{|\mathcal{K}|N_{a}}\sum_{k\in\mathcal{K}}\sum_{i=1}^{N_{a}}\left[1-\cos\!\left(a_{i}^{(k)},\operatorname{sg}(r_{i}^{(k)})\right)\right],\qquad \mathcal{K}=\{12\},\ \lambda_{\mathrm{align}}=0.05$$

Two design points are worth stealing independently of the rest.

**The grounding target is internal, not external.** The readout is recomputed every forward pass from the model's own attention, so there is no teacher, no encoder, and no cached target. The stop-gradient is load-bearing: without it the readout would drift toward the action state rather than the reverse. The Video DiT is frozen in this stage precisely so the target space is stable.

**It is a self-distillation of an attention operation.** The objective says: *whatever you just retrieved, still be pointing at it after the residual stream has processed it.* That is a general-purpose fix for any cross-attention interface where the conditioning signal is suspected of being used transiently, and nothing about it is specific to driving or to video.

### Stage 3 — Behaviour-aware action representations

![[ReWorld_infer.drawio.png|Self-guided inference using the discrepancy between intermediate and final velocity predictions, and the roughly 2x convergence acceleration from intermediate supervision]]

**Figure 2**: (a) ReWorld uses the discrepancy between the intermediate prediction $v_i$ and the final prediction $v_f$ to construct the self-guided velocity $v_w$. (b) ReWorld reaches a comparable validation level using approximately half the optimization steps of vanilla flow matching. Neither training scheme uses an external representation encoder or teacher model.

**Hard-negative mining** (following BeyondDrive): for each training scene, a flow-matching generator produces **64 candidates** with CFG and noise-scale diversification; each is scored by the **NAVSIM PDM simulator**; candidates below $\delta=0.6$ form the low-scoring set; the negative is the one *closest to the expert* under the same normalization used for flow matching.

$$\mathcal{I}_{\mathrm{low}}=\{n\mid s(\tau^{(n)})<0.6\},\qquad n^{\star}=\arg\min_{n\in\mathcal{I}_{\mathrm{low}}}\frac{1}{L}\sum_{\ell}\left\|\tau_{\ell}^{(n)}-\tau_{\ell}^{\mathrm{exp}}\right\|_{2}^{2}$$

The rationale is the same one [[sources/drivesuprim.md]] gives for coarse-to-fine filtering: *"random negatives are often distinguishable by geometry alone."*

**Repulsive distance objective.** A clean trajectory estimate is recovered from the *same* forward pass and the *same* $t_a$ used by $\mathcal{L}_{\mathrm{FM}}$ — so RDE costs no extra compute:

$$\hat{a}_{0}=a_{t_{a}}-t_{a}v_{\phi}^{a}(\cdot),\qquad \Delta(\tau)_{\ell}=\left[\widetilde{\Delta x}_{\ell},\widetilde{\Delta y}_{\ell},\sin\psi_{\ell},\cos\psi_{\ell}\right]$$

$$\mathcal{L}_{\mathrm{RDE}}=-\frac{1}{|\mathcal{V}|}\sum_{b\in\mathcal{V}}\frac{1}{L}\sum_{\ell}\frac{1}{4}\sum_{d}\left|\Delta(\hat{\tau}_{b})_{\ell}^{(d)}-\Delta(\tau_{b}^{\mathrm{neg}})_{\ell}^{(d)}\right|,\qquad \lambda_{\mathrm{RDE}}=0.04$$

The **delta / sine-cosine parameterization** targets relative motion rather than absolute position and avoids the angular discontinuity. The paper is candid that the objective is unbounded below in isolation and works only as a weak regularizer against the quadratic $\mathcal{L}_{\mathrm{FM}}$ — a claim its own weight sweep confirms brutally ($\lambda=0.10$ collapses PDMS to 85.5).

In Stage 3 the Video DiT is **unfrozen and $\mathcal{F}$ is recomputed without detachment**, so behaviour-oriented gradients reach the planner-facing video states. $\mathcal{L}_{\mathrm{align}}$ is deliberately dropped here, because Stage 2 wanted a stable target and Stage 3 wants the target to move.

### Training configuration

| Stage | Init | Trains | Steps | Batch | Key hyperparameters |
|---|---|---|---|---|---|
| 1 | LTX-Video | Video DiT | 20k | 64 | AdamW, lr 1e-5, wd 5e-2, $\mathcal{S}=\{8\}$ |
| 2 | **"the DriveLaW checkpoint"** | Action DiT (Video frozen) | 6k | 128 | $\lambda_{\mathrm{align}}=0.05$, layer 12 |
| 3 | Stage 2 | **both jointly** | 10k | 160 | $\lambda_{\mathrm{RDE}}=0.04$, $\delta=0.6$, 64 candidates |

Inference: 30 video sampling steps with $\gamma=1.4$; **five** action flow-matching steps. Video training data is nuPlan + nuScenes at 8 Hz; trajectory supervision is NAVSIM navtrain at 2 Hz.

**Stage 2's initialization is the ambiguity that matters.** §III-F describes a sequential curriculum in which Stage 2 follows Stage 1, but §IV-A says Stage 2 "initializes from the DriveLaW checkpoint." If read literally, the planning pipeline never sees the Stage-1 video model and the curriculum is not progressive at all. If read loosely — "the chained-WAM checkpoint" — then Stage 1 is in the planning path but its contribution is still unmeasured, because no ablation isolates it. See [Limitations](#limitations).

---

## Results

### Table I — nuScenes video generation

| Method | FID ↓ | FVD ↓ |
|---|---:|---:|
| DriveGAN | 73.4 | 502.3 |
| DriveDreamer | 52.6 | 452.0 |
| DrivingGPT | 12.8 | 142.6 |
| DriveWorld | 7.4 | 90.9 |
| Vista | 6.9 | 89.4 |
| Epona | 7.5 | 82.8 |
| DriveLaW | 4.6 | 81.3 |
| **ReWorld** | **4.4** | **61.9** |

**ReWorld is the first entry in this wiki to lead both nuScenes generation metrics simultaneously.** [[concepts/world-model-for-ad.md]] previously recorded DriveLaW best on FID (4.6) and UniUGP best on FVD (75.9) with no single leader; 4.4 / 61.9 takes both, and 61.9 is a 23.9% relative improvement over its own predecessor.

**The decomposition matters more than the number.** From the paper's own ablation: intermediate supervision at $\gamma=1.0$ gives **78.9**; self-guidance at $\gamma=1.4$ gives **61.9**. So the representation objective is worth 2.4 FVD and the sampling mechanism is worth 17.0. FID moves 4.6 → 4.4, i.e. frame-level fidelity is essentially unchanged — consistent with the gain being temporal-coherence-shaped, and also consistent with a guidance-strength effect, which FVD is known to respond to. The paper does not report FID under $\gamma=1.0$, so the two cannot be fully separated.

### Table II — NAVSIM navtest (PDMS)

| Method | Ref | Img | LiDAR | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|---|---|:-:|:-:|---:|---:|---:|---:|---:|---:|
| *Traditional End-to-End* | | | | | | | | | |
| VADv2-$\mathcal{V}_{8192}$ | arXiv'24 | ✓ | | 97.2 | 89.1 | 91.6 | 100 | 76.0 | 80.9 |
| UniAD | CVPR'23 | ✓ | | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| TransFuser | TPAMI'23 | ✓ | ✓ | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| PARA-Drive | CVPR'24 | ✓ | | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| ReCogDrive-**IL** | ICLR'26 | ✓ | | 98.1 | 94.7 | 94.2 | 100 | 80.9 | 86.5 |
| DiffusionDrive | CVPR'25 | ✓ | ✓ | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| *World Model Methods* | | | | | | | | | |
| DrivingGPT | ICCV'25 | ✓ | | 98.9 | 90.7 | 94.9 | 95.6 | 79.7 | 82.4 |
| LAW | ICLR'25 | ✓ | | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 |
| Epona | ICCV'25 | ✓ | | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 |
| ReSim | NeurIPS'25 | ✓ | | – | – | – | – | – | 86.6 |
| WoTE | ICCV'25 | ✓ | ✓ | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| DriveVLA-W0 **†** | ICLR'26 | ✓ | | 98.4 | 95.3 | 95.2 | 100 | 80.9 | 87.2 |
| PWM | NeurIPS'25 | ✓ | | 98.6 | 95.9 | 95.4 | 100 | 81.8 | 88.1 |
| WorldDrive | arXiv'26 | ✓ | | 98.4 | 96.8 | 95.2 | 100 | 83.3 | 89.0 |
| DriveLaW | CVPR'26 | ✓ | | 99.0 | 97.1 | 96.7 | 100 | 81.3 | 89.1 |
| **ReWorld** | – | ✓ | | **99.1** | **98.2** | **97.7** | 99.8 | 82.0 | **90.4** |

† = trained with the same flow-matching objective.

**This is the cleanest NAVSIM-v1 baseline table ingested since [[sources/foresight.md]]**, and on one axis it is better. Every row matches this wiki's canonical values — VADv2 80.9, UniAD 83.4, TransFuser 84.0, PARA-Drive 84.0, DiffusionDrive 88.1, DrivingGPT 82.4, LAW 84.6, Epona 86.2, WoTE 88.3, PWM 88.1, WorldDrive 89.0, DriveLaW 89.1 — and **the two configuration-sensitive rows are both explicitly labelled**: ReCogDrive appears as "ReCogDrive-**IL**" (86.5, its imitation-only variant) and DriveVLA-W0 carries a **†** whose meaning the caption defines. [[sources/brainwam.md]] propagated both of those same numbers unmarked; [[sources/adaptive-wam.md]] propagated the DriveVLA-W0 one. ReWorld is the first ingested paper to label both.

**The claim is also correctly scoped**: "the best overall performance among the compared methods" and "the highest PDMS, NC, DAC, and TTC among the compared **world-model** methods." In the wider wiki, 90.4 sits below CLEAR/DA-WAM 93.7, DriveSuprim 93.5, Drive-JEPA 93.3, WCog-VLA 92.9, HybridDriveVLA 92.1, LWDrive 92.0, WA-JEPA 91.8, DynVLA 91.7, SimWAM 91.5, FLARE 91.4, DiffusionDriveV2 91.2, SGDrive 91.1, ELF-VLA/WAM-Diff 91.0 — none of which appear in the table.

**Where the gain lands.** DAC +1.1 (97.1 → 98.2) and TTC +1.0 (96.7 → 97.7) carry it, with NC +0.1 and **EP only +0.7 (81.3 → 82.0)**. ReWorld inherits DriveLaW's conservative-progress profile: 82.0 EP is far below iPad 88.0, LWDrive 87.3, ReCogDrive 87.3, and even the human driver's 87.5. This is exactly the shape the hard-negative objective predicts — repelling from unsafe neighbours buys compliance and time-to-collision, not progress. Comfort drops 100 → 99.8, the only regression.

### Table III — UCF-101 frozen linear probing

| Frozen Video DiT representation | Top-1 Acc. (%) ↑ |
|---|---:|
| LTX-Video | 66.8 |
| DriveLaW | 68.3 |
| ReWorld (Stage 1) | 71.7 |
| **ReWorld (full)** | **80.2** |

**This is the most interesting table in the paper and the least explained.** It is also a rare thing in this wiki: a *direct* measurement of representation quality rather than a task-level proxy. Same 33-frame 224×224 clips, same flow timestep, same normalization, same classifier schedule, features from the final block of the 28-layer Video DiT with global mean pooling.

Two readings, and the paper commits to neither:

- Generic video pretraining → driving adaptation is worth **+1.5**; Stage-1 future-predictive supervision is worth a further **+3.4**. Both are plausible and modest.
- **Stage 1 → full curriculum is worth +8.5**, and since the Video DiT is *frozen throughout Stage 2*, that entire jump must come from **Stage 3** — i.e. from planning-oriented hard-negative gradients propagating into the video branch. On its face, ReWorld is claiming that repelling a trajectory from an unsafe neighbour improves generic action-recognition transfer by 8.5 points. That is a striking claim, it is nowhere analysed, and it is the single result in the paper most in need of a control.

### Table IV — Unified representation-learning protocol

All methods trained from scratch on LTX-Video, 120k steps, nuPlan + nuScenes, 224×224×25 clips, no text conditioning, batch 32. FVD on the nuScenes test set.

| Model | FVD ↓ |
|---|---:|
| *Without external representations* | |
| Vanilla Flow | 304.1 |
| SRA | 296.9 |
| SRA2 | 295.2 |
| Self-Flow | 283.3 |
| **ReWorld** | **270.4** |
| *With external representations* | |
| REPA w/ DINOv2 | 295.9 |
| REPA w/ DepthAnything3 | 319.4 |
| REPA w/ VideoMAEv2 | 328.3 |
| REPA w/ **V-JEPA 2** | **331.6** |
| ReDi | 421.7 |

**Three of the four external teachers are worse than no teacher at all**, and the worst of them is **V-JEPA 2 at 331.6 against vanilla flow's 304.1** — a 27.5-point regression. The wiki's JEPA thread ([[sources/drive-jepa.md]], [[sources/auto-jepa.md]], [[sources/wa-jepa.md]], [[sources/da-wam.md]]) treats V-JEPA 2 as the strongest available video representation for driving, and this is the first ingested result where it is actively harmful.

**The scope of that finding is narrow and should stay narrow.** The task is *aligning a video generator's intermediate features to a frozen encoder's features*, not *encoding a scene for a planner*. V-JEPA 2's representation is trained to be predictive-in-latent and deliberately discards appearance detail — exactly the information a generator's intermediate states still need. So the result is better read as **"a representation optimized to throw pixels away is a bad alignment target for a model that must put pixels back"** than as anything about V-JEPA 2's planning value. It nonetheless belongs on the record, because it is the only controlled measurement here of a JEPA encoder underperforming a null baseline.

The paper's own explanation is the same in shape: external encoders "may supply semantic priors that are poorly matched to multi-second ego and agent dynamics," while ReWorld's target is the generator's **native future-flow target**, which is temporally structured by construction.

### Table V — Per-step training cost

| Method | Normalized cost |
|---|---:|
| Vanilla Flow | 1.0× |
| **ReWorld** | **1.003×** |
| Self-Flow | ~1.4× |
| SRA | ~1.4× |
| ReDi | ~1.6× |
| REPA | ~1.7× |

Best FVD in Table IV at essentially free cost. The mechanism is simply that the target and the activations both already exist; SRA and Self-Flow need a second DiT forward pass to construct their self-supervised targets, and REPA needs a separate encoder resident in memory. **This is the strongest efficiency argument in the paper and the least contestable claim in it.**

---

## Ablations

### Table VI(a) — Which block to supervise

| Supervised block | 2 | **8** | 12 | 16 | 20 |
|---|---:|---:|---:|---:|---:|
| FVD ↓ | 65.5 | **61.9** | 62.7 | 63.0 | 64.3 |

The best block is **8 of 28 — about 29% depth** — and the stated reason is "a favorable balance between representation maturity and subsequent refinement": earlier blocks have less developed future estimates, deeper blocks give less hierarchical separation from the final head.

**The spread is 3.6 FVD across five depths, which is small.** Set beside [[sources/adaptive-wam.md]]'s 4.80 PDMS across six *readout* depths on a video DiT and [[sources/lwdrive.md]]'s 0.2 PDMS across VLM depths, this is a different quantity — where to *inject supervision*, not where to *read features* — but it points the same way: **a mid-network block beats both ends**, and 29% here versus Adaptive-WAM's ~50% are the two datapoints available.

### Table VI(b) — Self-guidance scale

| $\gamma$ | 1.0 | 1.2 | **1.4** | 1.6 | 1.8 |
|---|---:|---:|---:|---:|---:|
| FVD ↓ | 78.9 | 72.0 | **61.9** | 69.7 | 68.2 |

Strongly non-monotonic, with a sharp optimum — the signature of a guidance-scale parameter rather than a representation property. The $\gamma=1.0$ row is the honest baseline for the representation objective alone.

### Table VII — The planning ablation

| Configuration | $\mathcal{L}_{\mathrm{align}}$ | $\mathcal{L}_{\mathrm{RDE}}$ | PDMS ↑ |
|---|:-:|:-:|---:|
| DriveLaW | | | 89.1 |
| + Align only | ✓ | | 89.5 |
| + RDE only | | ✓ | 89.8 |
| **ReWorld** | ✓ | ✓ | **90.4** |

+0.4 and +0.7 individually, +1.3 together — mildly super-additive, and the paper's account of why is reasonable: alignment establishes a world-grounded action representation and RDE then separates locally similar trajectories within it.

**What is not in this table is Stage 1.** Every row is an action-side objective, and between them they account for the full headline delta. The paper's title claim — that optimizing the *world* representation improves planning — has no row.

### Table VIII — Weight sensitivity

| $\lambda_{\mathrm{align}}$ (Stage 2 only) | 0.01 | 0.03 | **0.05** | 0.07 | 0.10 |
|---|---:|---:|---:|---:|---:|
| PDMS | 88.8 | 89.2 | **89.5** | 88.2 | 87.7 |

| $\lambda_{\mathrm{RDE}}$ (Stage 3 from best Stage 2) | 0.02 | 0.03 | **0.04** | 0.05 | 0.10 |
|---|---:|---:|---:|---:|---:|
| PDMS | 89.4 | 89.6 | **90.4** | 89.7 | 85.5 |

**Both objectives are sharply peaked and both go negative when over-weighted.** $\lambda_{\mathrm{align}}=0.01$ scores 88.8, *below* the 89.1 DriveLaW baseline; $\lambda_{\mathrm{align}}=0.10$ scores 87.7, 1.4 below it. On the RDE side, 0.04 → 0.05 costs 0.7 and 0.10 costs 4.9. These are auxiliary regularizers with narrow operating windows, tuned on the evaluation benchmark, and neither paper nor page should describe them as robust. The 90.4 headline is the peak of a 5-point sweep whose neighbours are 89.6 and 89.7.

---

## Two Papers in One Envelope {#decoupling}

The wiki's [test-time-imagination synthesis](../concepts/world-model-for-ad.md#test-time-imagination) has converged on: *future-prediction objectives are valuable; instantiated future world states at decision time are not.* ReWorld runs the natural next experiment without framing it as one, and the result is sharper than that statement.

| Half | Mechanism | Video quality | Planning quality |
|---|---|---|---|
| Stage 1 | $\mathcal{L}_{\mathrm{Mid}}$ + self-guidance | FVD 81.3 → **61.9**, UCF-101 68.3 → 71.7 | **never measured** |
| Stages 2–3 | $\mathcal{L}_{\mathrm{align}}$ + $\mathcal{L}_{\mathrm{RDE}}$ | not measured in isolation | 89.1 → **90.4** |

**Within one architecture, one codebase, and one paper, the mechanism that improves the world model and the mechanism that improves the plan are disjoint, and neither is shown to help the other's metric.** The +1.3 PDMS comes entirely from action-side objectives, one of which ($\mathcal{L}_{\mathrm{RDE}}$) uses no world information at all — it compares two trajectories in delta space. The +19.4 FVD comes from a video-side objective plus a sampling trick that the paper explicitly says leaves the planner-facing interface unchanged.

This is the strongest evidence yet for a claim the wiki has been assembling piecemeal: **video-generation quality and planning quality are decoupled in a chained WAM.** Prior support was inter-paper and confounded — [[sources/drivelaw.md]] leads on FID while planning at 89.1; [[sources/simwam.md]] plans at 91.5 with no inference-time generation at all; [[sources/foresight.md]] runs a 2.5B generator to a finished future for 89.3. ReWorld makes it intra-paper.

**Two honest counter-readings.** First, Stage 3's video branch *is* updated by planning gradients, so the pathway is not inert — the UCF-101 jump from 71.7 to 80.2 shows the video representation changes materially under decision-oriented supervision. The influence the paper demonstrates runs **action → world**, which is the opposite of its thesis. Second, the missing experiment is one run: Stages 2–3 initialized from the vanilla DriveLaW video model versus from the Stage-1 model, PDMS reported for both. Given that both checkpoints exist and the code is released, its absence is conspicuous.

---

## Qualitative Results

![[contrast.png|Side-by-side future video generation, DriveLaW versus ReWorld, over a 3-second horizon]]

**Figure 3**: Conditioning uses 1 s of history (8 frames at 8 Hz); columns $T{-}1$ and $T$ show its first and last frames. Generated futures (3 s, 24 frames) lie right of the dashed line. Each pair of consecutive rows is one scene for DriveLaW and ReWorld respectively. ReWorld better preserves lane markings, roadside geometry, distant objects, and temporal consistency.

![[qualitive_new.png|Six additional nuScenes generation scenes covering sunny and rainy weather, high-speed travel, and intersections]]

**Figure 4**: Additional generation results across sunny and rainy weather, high-speed travel, intersections, and other urban conditions.

![[navsimvis.png|NAVSIM navtest planning examples: straight, left turn, right turn, intersection]]

**Figure 5**: Planning on NAVSIM navtest — straight, turn left, turn right, intersection. Red = predicted ego trajectory, green = ground-truth expert.

Note that Figures 3–4 show **decoded** futures, which the planning path never produces: video decoding is an evaluation artifact here, not part of inference.

---

## Limitations

1. **The title claim is unmeasured.** Table VII contains no Stage-1 row, so nothing in the paper shows that a more future-predictive world representation yields a better plan. The whole +1.3 PDMS is attributed by its own ablation to two action-side objectives, one of which never touches world information.

2. **Stage 2's initialization contradicts the curriculum description.** §III-F describes Stage 2 as following Stage 1; §IV-A says Stage 2 "initializes from the DriveLaW checkpoint." Read literally, the planning results are built on the *un-improved* video model and the "progressive curriculum" is two independent fine-tunes. Read loosely, the ambiguity still leaves (1) intact.

3. **88% of the FVD headline is a sampling trick.** Representation supervision alone: 81.3 → 78.9. Self-guidance: 78.9 → 61.9. The abstract and conclusion lead with 61.9 throughout. FID at $\gamma=1.0$ is not reported, so the sampling and representation effects on frame fidelity cannot be separated.

4. **Self-guidance is a guidance-scale knob with a sharp optimum** (72.0 → 61.9 → 69.7 across $\gamma\in\{1.2,1.4,1.6\}$), tuned on the same nuScenes set the headline is reported on. Whether the baseline DriveLaW 81.3 uses a comparable text-CFG setting is not stated, so the two rows may not be sampling-matched.

5. **$\mathcal{L}_{\mathrm{RDE}}$ is simulator distillation, and the abstract's phrasing understates it.** "Without reinforcement learning or test-time scoring" is literally true — but hard negatives are mined by scoring 64 candidates per training scene with the **NAVSIM PDM simulator**, i.e. the benchmark's own function is used to construct the training signal. This is the same class of privileged supervision the wiki flags for [[sources/drivesuprim.md]], [[sources/drive-jepa.md]], [[sources/da-wam.md]], [[sources/geoworldad.md]], and [[sources/lwdrive.md]], differing only in that the simulator trains a *repulsion target* rather than a *scorer*. It is also the larger of the two planning contributions (+0.7 of +1.3).

6. **Both objectives are sharply peaked and can go negative.** $\lambda_{\mathrm{align}}=0.01$ (88.8) and $=0.10$ (87.7) are both *below* the DriveLaW baseline; $\lambda_{\mathrm{RDE}}=0.10$ costs 4.9 PDMS. The headline is the peak of a five-point sweep on the evaluation benchmark, with 89.6 and 89.7 on either side.

7. **The +8.5 UCF-101 jump from Stage 3 is unexplained.** Since the Video DiT is frozen in Stage 2, the entire Stage-1 → full gap must come from planning-oriented gradients. No control isolates it, no hypothesis is offered, and the direction of influence it implies (action → world) is the reverse of the paper's thesis.

8. **No NAVSIM-v2, no navhard, no Bench2Drive, no HUGSIM, no nuScenes planning.** For a paper whose contribution is representation robustness, the absence of any OOD or reactive protocol is the most valuable missing experiment after (1). Single runs, no seed variance, against ablation deltas of 0.4.

9. **No latency, FPS, or inference-cost numbers anywhere.** Training cost is reported meticulously (1.003×) and inference cost not at all — although the architecture is inherited from DriveLaW and unchanged, so DriveLaW's profile should carry over.

10. **EP 82.0 remains the weak metric**, 5+ points below the current frontier and below the human driver's 87.5. The safety-shaped objective plausibly *causes* this: repelling from low-scoring neighbours is a conservatism prior. Not discussed.

11. **The unified protocol in Table IV is 224×224×25 without text conditioning**, which is far from the deployed 1280×704×25 setting. Whether the ordering — and particularly V-JEPA 2's 331.6 — survives at full resolution and horizon is untested.

12. **Neither $\lambda_{\mathrm{Mid}}$ nor the number of supervised blocks beyond $|\mathcal{S}|=1$ is ablated.** Table VI(a) varies *which* single block; nothing varies *how many*, or how strongly.

---

## Key Cross-References

- **The predecessor, same group**: [[sources/drivelaw.md]] — same architecture, same latent interface, same authors. DriveLaW's t=1 sweep (89.1 at the earliest latent, 23.2 at the near-clean one) is the finding that makes ReWorld's premise sensible: if the planner reads a barely-formed latent, that latent had better be explicitly shaped. ReWorld then shapes it and does not test whether the planner benefits.
- **Generation and planning are decoupled**: [[concepts/world-model-for-ad.md]] — the [analysis above](#decoupling) is the wiki's first *intra-paper* evidence, complementing DriveLaW (best FID, mid PDMS), SimWAM (91.5 with no inference-time generation), and ForeSight (900 ms of generation for 89.3).
- **First to lead both nuScenes generation metrics**: 4.4 FID / 61.9 FVD supersedes the split leadership (DriveLaW FID 4.6, UniUGP FVD 75.9) recorded on the generation-quality tables.
- **A JEPA encoder measured below a null baseline**: Table IV puts REPA-with-V-JEPA-2 at 331.6 against vanilla flow's 304.1. Scoped to *generator feature alignment*, not planning — see [[concepts/foundation-backbones-for-ad.md]] for why a representation trained to discard appearance is a poor alignment target for a model that must reconstruct it.
- **Hard negatives, third mechanism**: [[concepts/selection-based-planning.md]] — [[sources/drivesuprim.md]] concentrates them by coarse-to-fine filtering, [[sources/da-wam.md]] retrieves them as scorer training data (+0.22), and ReWorld repels from them with a delta-space loss on a flow-matching planner (+0.7). All three mine with privileged supervision; only ReWorld's needs no scorer at inference.
- **Mid-network supervision, mid-network readout**: block 8 of 28 (~29%) is best for *injecting* supervision here; [[sources/adaptive-wam.md]] found block 15 of 30 (~50%) best for *reading* features. Different operations, same shape of answer — see [[concepts/foundation-backbones-for-ad.md]].
- **A teacher-free grounding objective worth stealing**: $\mathcal{L}_{\mathrm{align}}$'s stop-gradient self-distillation of a cross-attention readout is architecture-agnostic and applies to any interface where conditioning may be used transiently — including the VLM-to-planner interfaces on [[concepts/dual-system-vla.md]].
- **New methods for the gap list**: BeyondDrive (2605.19771, the hard-negative source), ReSim (2506.09981, 86.6 PDMS), SRA2 (CVPR'26), Self-Flow (2603.06507), ReDi, "Guiding a diffusion transformer with the internal dynamics of itself" (CVPR'26, the self-guidance precedent), plus the diffusion-representation family REPA / RAE / VA-VAE / VFM-VAE / AlignTok / FAE / Latent Forcing. WorldDrive 89.0 is corroborated against [[sources/geoworldad.md]]'s value.
