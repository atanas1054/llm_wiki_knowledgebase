---
title: "PhysWAM: Physically Consistent World Action Model for Autonomous Driving"
type: source-summary
sources: ["raw/papers/PhysWAM_ Physically Consistent World Action Model for Autonomous Driving.md"]
related: [sources/drivereferee.md, concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/navhard-ood-evaluation.md, concepts/hugsim-benchmark.md, concepts/best-of-n.md, concepts/selection-based-planning.md, concepts/evaluation-variance.md, concepts/inference-latency.md, concepts/teacher-pseudo-labels.md, concepts/foundation-backbones-for-ad.md, concepts/wam-attention-masks.md, sources/geowam.md, sources/geoworldad.md, sources/drivedreamer-policy.md, sources/suv.md, sources/wa-jepa.md, sources/simwam.md, sources/metis.md, sources/drivefuture.md, sources/coworld-vla.md, sources/reworld.md, sources/ad-e2e-jepa.md, sources/adaptive-wam.md, sources/spanvla.md, sources/latent-wam.md, sources/hydra-mdp-pp.md]
created: 2026-09-30
updated: 2026-09-30
confidence: medium
---

# PhysWAM

**Paper**: PhysWAM: Physically Consistent World Action Model for Autonomous Driving
**Authors**: Dhruv Parikh (corresponding), Fengcheng Yu, Quankai Gao, Jiawei Yang, Junjie Ye, Maulik Bhatt, Thang Vu, Charles Ochoa, Rowan McAllister, Igor Vasiljevic, Rajgopal Kannan, Viktor Prasanna, Vitor Guizilini, Yue Wang
**Orgs**: University of Southern California, Woven by Toyota, Toyota Research Institute, DEVCOM Army Research Office
**arXiv**: 2609.37970v1
**Code**: release promised ("code, training and evaluation scripts, and model checkpoints"); no link in the paper
**Source**: `raw/papers/PhysWAM_ Physically Consistent World Action Model for Autonomous Driving.md`

---

## What It Is

A world-action model that **co-denoises three modalities in one flow-matching transformer**: three-camera future video, per-view **metric depth**, and ego motion as SE(3) pose rows. It is a fine-tune of **Cosmos 3 Nano** (15.2B parameters, 7.0B trained), the largest model in the wiki.

The contribution is one loss, **Coupled Point Projection (CPP)**. Generated depth is unprojected to 3D points and moved by the *generated* ego motion. LiDAR points are moved by the *recorded* ego motion. CPP penalizes the distance between the two point sets in one frame. A depth error, a motion error, or both show up in the same residual, so each output is supervised through the other.

Inference has no scorer. Either one sample is used, or the **medoid of eight samples** (the trajectory closest to all the others) is returned.

Headline numbers, single sample: **90.3 EPDMS navtest, 38.1 EPDMS navhard, 48.9 RC / 35.5 HD-Score zero-shot HUGSIM**, with no RL and no learned scorer. The paper also prints **91.4 "PDMS"**, which this page argues is [not a NAVSIM-v1 measurement](#v1-column).

---

## Key Takeaways

- **CPP is worth about +1.9 everywhere it is measured**: navtest 89.6 → 91.4 PDMS and 88.4 → 90.3 EPDMS, navhard 36.2 → 38.1, HUGSIM HD-Score 33.4 → 35.5. Unlike the attention-mask mechanisms recorded in the wiki, its effect is the *same size* on navtest and navhard.
- **The coupling itself is not isolated, and the paper says so twice.** CPP adds metric supervision of depth, metric supervision of pose, and a cross term. No run compares it with the two decoupled losses.
- **Generating depth does nothing for planning without CPP.** Multi-view with no depth stream scores 87.5; adding the depth stream under flow matching alone gives 87.2.
- **The "PDMS" column is built from NAVSIM-v2 sub-scores.** All four PhysWAM rows have identical NC / DAC / TTC / EP in the v1 and v2 halves of Table 1, and their PDMS equals the closed form of those means to within 0.1. No other method's row in the table has either property.
- **Sampler noise is 0.24 PDMS / 0.30 EPDMS**, five to twenty times the two earlier measurements in the wiki (0.013–0.053).
- **A label-free selector recovers 8% of the oracle headroom.** Medoid-of-8 adds +0.3 PDMS; oracle-of-8 adds +3.9. On navhard the medoid adds +1.7.
- **One plan costs 9.4 GPU-seconds** at 30 steps, and 75 for the medoid of eight. The closed-loop result replans in simulation time.
- **World-model quality is measured against real references**: depth against LiDAR (not against the teacher that labelled it), FVD against a recorded-versus-recorded floor, and generated video against generated motion.
- **Training is fully instrumented**: LiDAR, annotated 3D boxes and HD-map drivable polygons all enter the loss. "Label-free" describes trajectory *selection* only.

---

## Method

### Problem and representation

Cameras $\mathcal V=\{\mathrm F,\mathrm L,\mathrm R\}$ (front 832×468, sides 416×234), frames $t=0,\dots,8$ at 2 Hz. The model learns

$$p_\theta\big(\{\mathbf I^{v}_{1:T},\mathbf D^{v}_{0:T}\}_{v\in\mathcal V},\ \mathbf a_{1:T}\ \big|\ c\big)$$

where $c$ holds the current images, calibration, a text template (city, command, speed) and the previous motion step. **Depth is generated at the current frame too**; current RGB and motion history stay clean.

- **Backbone**: Cosmos 3 Mixture-of-Transformers on Qwen3-VL. Text and generation pathways have separate weights and share attention. Text attends causally; visual and action tokens attend to text and **to each other bidirectionally**. Only the generation pathway, its projections and embeddings are trained.
- **RGB and depth share one frozen video VAE** (Wan). Depth is clipped to [0.5, 80] m, log-encoded, written as an 8-bit grey video and encoded like RGB, so a decoded value is metric with no per-scene scale:

$$\mathcal C(d)=2\,\frac{\ell_{\max}-\log(1+\operatorname{clip}(d,d_{\min},d_{\max}))}{\ell_{\max}-\ell_{\min}}-1$$

- **Motion** is the pretrained pose row $\mathbf a_t=[\mathbf t_t;\mathbf r^{(1)}_t;\mathbf r^{(2)}_t]\in\mathbb R^9$: translation in metres in the previous front-camera frame plus two rotation columns, orthonormalized by Gram–Schmidt. Relative transforms compose into $\mathbf T_t=\bm\Delta_1\cdots\bm\Delta_t$.
- **Camera geometry** enters as a Plücker ray embedding added to every visual token, $\mathbf z^{i}_{j,q}=\mathbf W_V[\mathcal P(\mathbf x_i^\sigma)]_{j,q}+\mathbf b_V+\mathbf W_R\mathbf r_v(q)$, shared by RGB and depth. Depth gets an extra learned embedding. Both new parameters are zero-initialized.
- **Positions** use mRoPE. RGB and depth tokens of the same patch share coordinates; action tokens sit at their endpoint time; the three views are laid out left–front–right.

### Flow matching and the clean estimate

$$\mathbf x^{\sigma}=(1-\sigma)\mathbf x+\sigma\bm\epsilon,\qquad \mathbf v^\star=\bm\epsilon-\mathbf x,\qquad \widehat{\mathbf x}=\mathbf x^\sigma-\sigma\widehat{\mathbf v}$$

One noise level is shared by all generated modalities of an example. The second identity matters: **every geometric loss acts on the one-step clean estimate $\widehat{\mathbf x}$ of the current forward pass**, so no denoising rollout is needed during training.

$$\mathcal L_{\mathrm{FM},V}=\mathbb E\Big[\sum_{v,m}\omega_{v,m}\operatorname{MSE}(\widehat{\mathbf v}_{v,m},\mathbf v^\star_{v,m})\Big],\qquad \mathcal L_{\mathrm{FM},A}=\mathbb E\big[\operatorname{MSE}(\widehat{\mathbf v}_A,\mathbf v^\star_A)\big]$$

### Coupled Point Projection

A small frozen depth head $\Pi_\phi$ (three 3×3 convolutions, fitted offline) maps clean depth latents to metric depth on the VAE grid, which avoids decoding through the VAE at every update. For each LiDAR-supported cell $u$ of view $v$ at anchor time $t$:

$$\widehat{\mathbf P}_{v,t}(u)=\widehat{\mathbf T}_t\,\mathbf E_v\,\mathcal U_v\big(\widehat d^{\,v}_{j(t)}(u),u\big),\qquad \mathbf P^\star_{v,t}(u)=\mathbf T^\star_t\,\mathbf E_v\,\mathcal U_v\big(d^{L,v}_t(u),u\big)$$

$$\mathcal L_{\mathrm{cpp}}=\frac{1}{|\Omega|}\sum_{(v,t,u)\in\Omega}\rho_\delta\Big(\big\|\widehat{\mathbf P}_{v,t}(u)-\mathbf P^\star_{v,t}(u)\big\|_2\Big)$$

- $\mathcal U_v$ is calibrated unprojection, $\mathbf E_v$ maps view $v$ into the front camera, $\rho_\delta$ is a Huber penalty with $\delta=0.5$ m.
- The LiDAR target is the geometric mean of the returns in a 16×16-pixel cell with at least 3 returns.
- Anchors are $t\in\{4,8\}$ (2 s and 4 s). $t=0$ is excluded because the transform is the identity there and the loss would supervise depth alone.
- Correspondence is by index $(v,t,u)$, so no point-cloud registration is needed. Both clouds describe the same instant, so moving objects are valid.

**The decomposition the paper gives in Appendix A.2.** With $\mathbf p(d)$ the point at depth $d$ on the cell's ray:

$$\mathbf r=\underbrace{\mathbf T^\star_t\mathbf p(\widehat d)-\mathbf T^\star_t\mathbf p(d^L)}_{\mathbf r_D\ \text{(depth only)}}+\underbrace{\widehat{\mathbf T}_t\mathbf p(d^L)-\mathbf T^\star_t\mathbf p(d^L)}_{\mathbf r_T\ \text{(motion only)}}+\underbrace{(\widehat{\mathbf T}_t-\mathbf T^\star_t)(\mathbf p(\widehat d)-\mathbf p(d^L))}_{\mathbf r_{DT}\ \text{(product of both errors)}}$$

A decoupled objective would penalize $\rho_\delta(\|\mathbf r_D\|)+\rho_\delta(\|\mathbf r_T\|)$. CPP applies $\rho_\delta$ to the sum, so the depth gradient contains the motion error along the ray and the pose gradient contains the depth error. The pose gradients are a net force on translation and a net torque about the camera centre:

$$\frac{\partial\rho_\delta}{\partial\widehat{\mathbf t}_t}=\mathbf g,\qquad \frac{\partial\rho_\delta}{\partial\bm\theta_t}=\big(\widehat{\mathbf P}_{v,t}(u)-\widehat{\mathbf t}_t\big)\times\mathbf g$$

### Two hinge losses on waypoints

Both act on the planar waypoint $\widehat{\mathbf p}_t=\pi_{xz}(\operatorname{trans}(\widehat{\mathbf T}_t))$ and ignore generated depth.

$$\mathcal L_{\mathrm{obs}}=\frac1T\sum_{t}\big[\operatorname{ReLU}(m_t-d_t(\widehat{\mathbf p}_t))\big]^2,\qquad \mathcal L_{\mathrm{drv}}=\frac1T\sum_{t}\big[\operatorname{ReLU}(m_{\mathrm{drv}}-S(\widehat{\mathbf p}_t))\big]^2$$

- $d_t$ is clearance from static LiDAR structure and from **annotated object boxes** at time $t$, using a signed distance to oriented rectangles. The margin is $m_t=\min(1.6\,\mathrm m,\ d^\star_t)$, capped at the clearance the recorded trajectory kept, so the expert path never incurs loss.
- $S$ is a signed distance field to the **nuPlan map's drivable polygons** (road blocks, intersections, car parks: "the same layers that the NAVSIM drivable-area metric treats as permissible"), rasterized at 0.5 m on a 128×128 grid; $m_{\mathrm{drv}}=1.5$ m.
- Both are annealed to zero between updates 9,000 and 15,000.

### Objective and gradient balancing

$$\mathcal L_{\mathrm{total}}=\lambda_V\mathcal L_{\mathrm{FM},V}+\lambda_A\big(\mathcal L_{\mathrm{FM},A}+s_{\mathrm{cpp}}\overline{\mathcal L}_{\mathrm{cpp}}+s_{\mathrm{obs}}\overline{\mathcal L}_{\mathrm{obs}}+s_{\mathrm{drv}}\overline{\mathcal L}_{\mathrm{drv}}\big)$$

with $\overline{\mathcal L}_k=\mathbb E[(1-\sigma)^2\mathcal L_k]$ so that heavily noised examples count less. The coefficients are not fixed. On every update each geometric term is rescaled so that its gradient **at the action velocity output** has a fraction $\eta_k=0.2$ of the action flow-matching gradient norm:

$$s_k=\operatorname{sg}\Big(\eta_k\frac{\|\nabla_{\widehat{\mathbf v}_A}\mathcal L_{\mathrm{FM},A}\|}{\|\nabla_{\widehat{\mathbf v}_A}\overline{\mathcal L}_k\|}\Big)$$

A second gain $\gamma$ on CPP's depth branch only, applied through $\mathcal G_\gamma(\widehat d)=\operatorname{sg}(\widehat d)+\gamma(\widehat d-\operatorname{sg}(\widehat d))$, imposes the same fraction against the depth flow gradient. The norms are measured with local backward passes to the velocity outputs, so the cost does not depend on the transformer.

### Inference

All generated tokens start from Gaussian noise and are denoised jointly with UniPC, 30 steps, no guidance. The VAE decodes RGB and depth; the pose rows compose into the trajectory. The depth head is not used. For $K$ samples the medoid is

$$k^*=\operatorname*{argmin}_{k}\sum_{l=1}^{K}\frac1T\sum_{t=1}^{T}\big\|\mathbf y^{(k)}_t-\mathbf y^{(l)}_t\big\|_2$$

over planar positions only.

### Data and labels

- **103,281 navtrain windows**, heavily overlapping. 30,000 updates × 44 windows is about 12.8 passes.
- **Dense depth labels** (for the flow loss): MapAnything run on the three views with calibrated poses and the frame's LiDAR as a sparse metric prior; sky from Depth Anything 3; holes (about a fifth of non-sky pixels) filled from DA3 monocular depth rescaled by a Laplace-interpolated log-ratio; a piecewise-linear bias correction fitted on 34.6M label–LiDAR pairs.
- **CPP targets** are the raw LiDAR, not the dense labels.
- **Hinge labels**: annotated boxes within 40 m, static LiDAR returns 0.3–3 m above ground, and the map raster.

---

## Figures

![[tile_a.png|Coupled Point Projection: generated depth unprojected and moved by generated motion (left, red trajectory) against LiDAR moved by recorded motion (right, black trajectory); a consistent example at CPP 1.02 m and an inconsistent one at 11.3 m]]

*Figure 1: CPP compares the scene implied by generated depth and motion with the measured scene. The caption describes four rows; the clipped asset contains two (one consistent at 1.02 m, one inconsistent at 11.3 m).*

![[physwam-method-v1-9-23.png|PhysWAM overview: multiview RGB and depth encoded by a shared frozen VAE, noised, tokenized with ray and mRoPE embeddings; action rows and text; the Cosmos 3 generator with shared multimodal attention and a frozen reasoner; velocity outputs supervised by flow matching; clean depth and action estimates feeding the CPP, drivable-area and obstacle losses]]

*Figure 2: Overview. Current RGB and motion history stay clean; everything else is noised and denoised jointly. The clean estimates of depth and action feed CPP and the two hinges.*

![[sec43_qualitative.png|Two qualitative comparisons: with versus without CPP at a left turn past a parked van, and multi-view versus single-view at a left turn with an oncoming car visible only in the left camera; generated views, depth maps and top-down rollouts]]

*Figure 3: Top: without CPP the model generates a parked van pressed against the ego and hits it at 3.4 s. Bottom: the single-view model cuts a corner into an oncoming car that only the left camera shows, and collides at 2.3 s.*

![[sec43_qualitative_appendix.png|Six further scenes: two with versus without CPP, two multi-view versus single-view, and two failures of the full model]]

*Figure 4: Additional scenes, including two failures of the full model: it renders a stopped minivan at constant size, plans as if it were pulling away and rear-ends it at 3.7 s (a world-model error); and it clips an inner curb by 0.4 m (a plan-precision error).*

---

## Tables

### Table 1: NAVSIM navtest, "v1" and v2

Other methods as reported in their papers. R34: ResNet-34. +L: LiDAR input. PhysWAM rows are one sample per scene unless stated; sample-to-sample sd 0.24 PDMS / 0.30 EPDMS.

| Method | Input | NC | DAC | TTC | C | EP | **PDMS** | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | **EPDMS** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Human | – | 100 | 100 | 100 | 99.9 | 87.5 | 94.8 | – | – | – | – | – | – | – | – | – | – |
| *End-to-end planners* | | | | | | | | | | | | | | | | | |
| TransFuser | 3×Cam+L | 97.7 | 92.8 | 92.8 | 100.0 | 79.2 | 84.0 | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| LTF | 3×Cam | – | – | – | – | – | – | 97.6 | 91.9 | 99.2 | 99.8 | 87.6 | 97.2 | 96.6 | 98.3 | 86.3 | 83.6 |
| Hydra-MDP++ (R34) | 3×Cam | 97.6 | 96.0 | 93.1 | 100 | 80.4 | 86.6 | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| ARTEMIS | 3×Cam+L | 98.3 | 95.1 | 94.3 | 100 | 81.4 | 87.0 | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | – | 98.3 | 83.1 |
| DiffusionDrive | 3×Cam+L | 98.2 | 96.2 | 94.7 | 100.0 | 82.2 | 88.1 | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| WoTE | 3×Cam+L | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 | – | – | – | – | – | – | – | – | – | – |
| BeyondDrive | 3×Cam | 98.4 | 97.9 | 95.0 | 100.0 | 83.7 | 89.7 | 98.4 | 97.9 | 99.5 | 99.8 | 87.8 | 98.0 | 97.3 | 98.3 | 88.5 | 90.1 |
| DriveSuprim (R34) | 3×Cam | 97.8 | 97.3 | 93.6 | 100 | 86.7 | 89.9 | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| *Vision–language–action models* | | | | | | | | | | | | | | | | | |
| AutoVLA | 3×Cam | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 | – | – | – | – | – | – | – | – | – | – |
| ReCogDrive | 3×Cam | 97.9 | 97.3 | 94.9 | 100 | 87.3 | 90.8 | – | – | – | – | – | – | – | – | – | – |
| *World models and world-action models* | | | | | | | | | | | | | | | | | |
| Epona | 1×Cam | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 | – | – | – | – | – | – | – | – | – | – |
| PWM | 1×Cam | 98.6 | 95.9 | 95.4 | 100.0 | 81.8 | 88.1 | – | – | – | – | – | – | – | – | – | – |
| DriveVLA-W0 | 1×Cam | 98.7 | 96.2 | 95.5 | 100.0 | 82.2 | 88.4 | – | – | – | – | – | – | – | – | – | – |
| DVGT-2 | 8×Cam | 97.8 | 97.2 | 93.9 | 100 | 83.4 | 88.6 | 97.8 | 97.2 | 99.6 | 99.9 | 88.4 | 97.3 | 98.1 | 98.2 | 83.2 | 88.9 |
| DriveDreamer-Policy | 3×Cam | 98.4 | 97.1 | 95.1 | 100.0 | 83.5 | 89.2 | 98.4 | 97.1 | 99.5 | 99.9 | 87.9 | 97.7 | 97.6 | 98.3 | 79.4 | 88.7 |
| DriveVLA-W0 (anchors) | 1×Cam | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.2 | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 |
| DVGT-2-NAVSIM | 8×Cam | 98.7 | 97.9 | 95.8 | 100 | 84.3 | 90.3 | 98.7 | 97.9 | 99.7 | 99.9 | 87.9 | 98.0 | 98.2 | 98.2 | 77.0 | 89.6 |
| GeoWAM | 8×Cam | – | – | – | – | – | – | 98.7 | 97.7 | 99.7 | 99.9 | 87.0 | 98.1 | 97.9 | 98.3 | 86.8 | 90.2 |
| **PhysWAM** | 3×Cam | 99.0 | 97.5 | 98.5 | 99.8 | 88.7 | **91.4** | 99.0 | 97.5 | 99.3 | 99.8 | 88.7 | 98.5 | 96.5 | 98.6 | 90.5 | **90.3** |
| **PhysWAM, medoid-of-8** | 3×Cam | 99.0 | 97.4 | 98.6 | 99.9 | 89.6 | **91.7** | 99.0 | 97.4 | 99.2 | 99.9 | 89.6 | 98.6 | 95.8 | 98.7 | 90.5 | **90.4** |
| PhysWAM, oracle-of-8 | 3×Cam | 99.7 | 99.3 | 99.2 | 99.9 | 91.9 | 95.3 | 99.7 | 99.3 | 99.0 | 99.8 | 91.9 | 99.2 | 95.8 | 98.7 | 90.0 | 94.1 |
| PhysWAM, without CPP | 3×Cam | 98.2 | 97.0 | 97.6 | 99.8 | 88.0 | 89.6 | 98.2 | 97.0 | 99.0 | 99.8 | 88.0 | 97.6 | 96.1 | 98.6 | 90.5 | 88.4 |

### Table 2a: NAVSIM-v2 navhard (combined two-stage EPDMS, as reported)

| Method | Input | EPDMS |
|---|---|---:|
| LTF | 3×Cam | 25.1 |
| DiffusionDrive | 3×Cam | 27.5 |
| DriveVLA-W0 (Flow) | 1×Cam | 24.4 |
| DVGT-2 | 8×Cam | 31.7 |
| DriveFuture | 6×Cam | 34.6 |
| 4D-WAM | 1×Cam | 35.9 |
| EponaV2 | 1×Cam | 36.1 |
| GeoWAM | 8×Cam | 36.6 |
| **PhysWAM** | 3×Cam | **38.1** |
| **PhysWAM, medoid-of-8** | 3×Cam | **39.8** |
| PhysWAM, without CPP | 3×Cam | 36.2 |

### Table 2b: Zero-shot closed-loop HUGSIM (RC / HD-Score)

PhysWAM is trained on NAVSIM only, one sample per planner call. BeyondDrive's overall is the unweighted mean of the four levels; the others weight levels by episode count.

| Method | Easy | Medium | Hard | Extreme | Overall |
|---|---:|---:|---:|---:|---:|
| UniAD | 58.6 / 48.7 | 41.2 / 29.5 | 40.4 / 27.3 | 26.0 / 14.3 | 40.6 / 28.9 |
| VAD | 38.7 / 24.3 | 27.0 / 9.9 | 25.5 / 10.4 | 23.0 / 8.2 | 27.9 / 12.3 |
| LTF | 68.4 / 52.8 | 40.7 / 24.6 | 36.9 / 19.8 | 25.5 / 8.1 | 41.4 / 24.8 |
| BeyondDrive | 76.8 / 65.6 | 43.0 / **31.4** | 35.5 / **26.3** | **29.6 / 16.2** | 46.2 / 34.8 |
| **PhysWAM** | **93.5 / 86.9** | **47.4** / 30.1 | **38.6** / 25.2 | 26.1 / 13.4 | **48.9 / 35.5** |
| PhysWAM, without CPP | 92.6 / 85.4 | 44.9 / 26.8 | 37.8 / 23.9 | 25.4 / 12.0 | 47.5 / 33.4 |

### Table 3: Ablations and future depth

**(a) CPP at the full recipe**

| | PDMS | EPDMS | Off-road | Collision | Heading error (median) |
|---|---:|---:|---:|---:|---:|
| 4k updates, with CPP | 80.3 | 78.8 | 9.7% | 3.8% | 2.7° |
| 4k updates, without CPP | 57.0 | 55.7 | 32.7% | 9.8% | 8.9° |

| | navtest PDMS / EPDMS | navhard | HUGSIM RC / HD |
|---|---:|---:|---:|
| Final, with CPP | 91.4 / 90.3 | 38.1 | 48.9 / 35.5 |
| Final, without CPP | 89.6 / 88.4 | 36.2 | 47.5 / 33.4 |

**(b) Objective terms** (LoRA setting, navtest, one sample)

| Objective | PDMS | EPDMS |
|---|---:|---:|
| flow | 85.7 | 84.3 |
| flow + hinge | 87.2 | 85.7 |
| flow + CPP | 87.8 | 86.2 |
| flow + hinge + CPP | 88.2 | 86.5 |

**(c) Future metric depth against LiDAR** (no scale alignment, four samples per scene)

| View / method | AbsRel +2 s ↓ | δ₁.₂₅ +2 s ↑ | AbsRel +4 s ↓ | δ₁.₂₅ +4 s ↑ |
|---|---:|---:|---:|---:|
| PhysWAM, front (NAVSIM) | 0.175 | 0.814 | 0.232 | 0.742 |
| PhysWAM, front, CPP cells (NAVSIM) | 0.144 | 0.856 | 0.194 | 0.795 |
| *Reported on nuScenes, front view* | | | | |
| Epona + DVGT | 0.263 | 0.677 | 0.310 | 0.589 |
| Cosmos 3 + DVGT | 0.376 | 0.513 | 0.422 | 0.447 |
| VGGT-World | 0.329 | 0.553 | 0.357 | 0.497 |
| GeoWAM | 0.245 | 0.769 | 0.297 | 0.703 |

**(d) Views** (PDMS by driving command)

| Views | All | Left | Straight | Right |
|---|---:|---:|---:|---:|
| Single-view (LoRA) | 87.1 | 81.2 | 90.0 | 82.0 |
| Multi-view (LoRA) | 88.2 | 83.9 | 90.3 | 84.3 |
| Full fine-tune | 91.4 | 87.8 | 92.7 | 90.4 |

### Table 4: Visual representation pathway for one stream

| Quantity | Shape | Operation |
|---|---|---|
| RGB or encoded depth | $(T+1)\times H_v\times W_v\times3$ | Input to the shared frozen VAE |
| $\mathbf x_i$ | $J\times h_v\times w_v\times c_\ell$ | VAE grid, indexed by $u$ |
| $\mathcal P(\mathbf x_i^\sigma)$ | $N_i\times p^2c_\ell$ | Flattened patches of the noisy grid, indexed by $q$ |
| $\mathbf z_i$ | $N_i\times d_h$ | Projected patches with ray, modality and noise embeddings |
| $\mathcal O_V(\mathbf z_i^{\mathrm{out}})$ | $N_i^{\mathrm{gen}}\times p^2c_\ell$ | Patch velocities at generated positions |
| $\widehat{\mathbf v}_i$ | $J\times h_v\times w_v\times c_\ell$ | Unpatchified velocity field; conditioning frames zero-filled |

### Table 5: Depth-label quality against LiDAR (1,000 training windows)

| View | AbsRel, cells | AbsRel, pixels ≤ 25 m | Sky error |
|---|---:|---:|---:|
| Front | 0.041 | 0.073 | 0.05% |
| Front-left | 0.051 | 0.113 | 0.22% |
| Front-right | 0.060 | 0.118 | 0.20% |

### Table 6: Hyperparameters

| | |
|---|---|
| Training windows | 103,281 (navtrain) |
| Frames per window | 1 current + 8 future, 2 Hz |
| Views and resolution | front 832×468; front-left and front-right 416×234 |
| Initialization | Cosmos 3 Nano (15.2B parameters) |
| Trained parameters | generation pathway, 7.0B |
| Noise-level shift | $\sigma=5u/(1+4u)$, $u$ uniform; training and sampling |
| Updates × batch | 30,000 × 44 windows |
| Optimizer | AdamW, β = (0.9, 0.99), no weight decay, clip 1.0 |
| Learning rate | peak 2e-5, 300 warm-up updates, linear decay to 0 |
| Precision | fp32 master weights, bf16 compute |
| Weight averaging | EMA, power-function profile, relative width 0.1 |
| $\lambda_V$, $\lambda_A$ | 10, 20 |
| View–modality weights | front RGB 1; side RGB 1/4; depth 1/6 per view |
| $\eta_{\mathrm{cpp}}$ | 0.2 throughout |
| $\eta_{\mathrm{obs}}$, $\eta_{\mathrm{drv}}$ | 0.2 to update 9,000, annealed to 0 by 15,000 |
| CPP anchors; Huber δ | $t\in\{4,8\}$ (2 s, 4 s); 0.5 m |
| Hinge margins | $m_{\mathrm{obs}}$ 1.6 m, $m_{\mathrm{drv}}$ 1.5 m |
| Sampler | UniPC, 30 steps, no guidance |
| Trajectory selection | one sample, or medoid of 8 |

Hardware is given only as "NVIDIA RTX PRO 6000 Blackwell GPUs"; no GPU count or training time.

### Table 7: navhard per-stage sub-scores

Stage 1: 450 recorded scenes. Stage 2: 5,462 synthetic scenes with reactive traffic. LTF and DriveVLA-W0 rows are as reported by EponaV2, which reprints the LTF score of the benchmark paper; the DVGT-2 row is as reported by GeoWAM.

**Stage 1**

| Method | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | Stage 1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LTF | 96.2 | 79.6 | 99.1 | 99.6 | 84.1 | 95.1 | 94.2 | 97.6 | 79.1 | – |
| DriveVLA-W0 | 96.8 | 83.3 | 99.0 | 99.6 | 84.6 | 95.3 | 96.4 | 97.6 | 78.2 | – |
| DVGT-2 | 97.2 | 91.3 | 98.4 | 99.8 | 84.8 | 95.5 | 95.5 | 97.5 | 71.4 | – |
| DriveFuture | 96.3 | 87.8 | 98.2 | 99.6 | 83.1 | 96.9 | 94.9 | 97.6 | 76.9 | – |
| 4D-WAM | 92.8 | 89.6 | 99.7 | 99.3 | 98.6 | 91.6 | 86.9 | 97.1 | 81.3 | – |
| EponaV2 | 97.3 | 90.7 | 99.4 | 100 | 83.3 | 97.3 | 97.3 | 97.6 | 60.9 | – |
| GeoWAM | 97.7 | 91.5 | 99.1 | 99.8 | 83.8 | 95.8 | 96.0 | 97.8 | 79.0 | – |
| PhysWAM | 96.9 | 90.9 | 99.0 | 99.6 | 83.8 | 95.6 | 95.8 | 97.8 | 67.6 | 77.0 |
| PhysWAM, medoid-of-8 | 96.4 | 91.1 | 99.3 | 100.0 | 83.6 | 96.4 | 96.0 | 97.8 | 72.4 | 78.3 |
| PhysWAM, without CPP | 97.0 | 90.2 | 99.0 | 99.8 | 83.9 | 95.6 | 95.1 | 97.8 | 67.6 | 76.6 |

**Stage 2**

| Method | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | Stage 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LTF | 77.8 | 70.2 | 84.3 | 98.1 | 85.1 | 75.7 | 45.4 | 95.7 | 76.0 | – |
| DriveVLA-W0 | 76.8 | 64.3 | 79.9 | 98.3 | 89.2 | 75.0 | 46.8 | 95.8 | 53.1 | – |
| DVGT-2 | 77.8 | 73.8 | 81.3 | 98.3 | 91.5 | 73.2 | 48.0 | 83.9 | 45.1 | – |
| DriveFuture | 82.3 | 78.8 | 88.1 | 98.4 | 83.6 | 79.6 | 47.6 | 97.0 | 75.9 | – |
| 4D-WAM | 82.6 | 71.8 | 86.9 | 98.2 | 97.6 | 78.8 | 49.0 | 95.8 | 71.6 | – |
| EponaV2 | 83.6 | 78.0 | 88.0 | 98.9 | 86.0 | 80.3 | 50.1 | 96.1 | 52.0 | – |
| GeoWAM | 80.4 | 76.3 | 87.3 | 98.7 | 88.9 | 76.2 | 49.9 | 94.0 | 56.0 | – |
| PhysWAM | 83.9 | 80.2 | 87.3 | 98.7 | 88.7 | 81.6 | 48.9 | 95.3 | 54.4 | 48.8 |
| PhysWAM, medoid-of-8 | 85.0 | 81.7 | 88.1 | 98.7 | 88.0 | 82.2 | 50.5 | 95.5 | 58.4 | 50.3 |
| PhysWAM, without CPP | 83.7 | 79.6 | 86.1 | 98.7 | 89.1 | 80.7 | 48.5 | 95.0 | 53.5 | 47.0 |

### Table 8: HUGSIM by difficulty and metric (×100)

Baselines as reported by the benchmark paper. Overall: per-dataset episode means, averaged with equal weight over the four source datasets within each difficulty, then weighted by episode counts (80, 157, 96, 103).

| Method | Difficulty | NC | DAC | TTC | COM | RC | HD-Score |
|---|---|---:|---:|---:|---:|---:|---:|
| UniAD | Easy | 77.4 | 88.5 | 70.8 | 82.8 | 58.6 | 48.7 |
| | Medium | 72.4 | 86.8 | 60.4 | 72.0 | 41.2 | 29.5 |
| | Hard | 66.5 | 86.0 | 55.0 | 67.2 | 40.4 | 27.3 |
| | Extreme | 54.4 | 89.6 | 42.9 | 58.5 | 26.0 | 14.3 |
| VAD | Easy | 66.1 | 73.9 | 58.2 | 100.0 | 38.7 | 24.3 |
| | Medium | 45.7 | 79.8 | 29.0 | 100.0 | 27.0 | 9.9 |
| | Hard | 44.3 | 81.0 | 28.8 | 100.0 | 25.5 | 10.4 |
| | Extreme | 36.1 | 83.7 | 25.9 | 100.0 | 23.0 | 8.2 |
| LTF | Easy | 74.1 | 81.7 | 70.6 | 99.6 | 68.4 | 52.8 |
| | Medium | 47.4 | 83.2 | 43.3 | 99.7 | 40.7 | 24.6 |
| | Hard | 40.7 | 83.9 | 37.0 | 99.9 | 36.9 | 19.8 |
| | Extreme | 28.3 | 85.1 | 21.8 | 99.8 | 25.5 | 8.1 |
| PhysWAM | Easy | 93.1 | 99.1 | 91.7 | 99.4 | 93.5 | 86.9 |
| | Medium | 51.8 | 97.5 | 39.9 | 95.6 | 47.4 | 30.1 |
| | Hard | 47.7 | 96.0 | 39.1 | 95.8 | 38.6 | 25.2 |
| | Extreme | 34.3 | 93.8 | 27.8 | 95.5 | 26.1 | 13.4 |
| | **Overall** | 54.4 | 96.6 | 46.4 | 96.3 | 48.9 | 35.5 |
| PhysWAM, without CPP | Easy | 92.7 | 98.8 | 91.0 | 99.3 | 92.6 | 85.4 |
| | Medium | 49.3 | 97.4 | 37.3 | 95.6 | 44.9 | 26.8 |
| | Hard | 47.0 | 96.0 | 37.8 | 95.2 | 37.8 | 23.9 |
| | Extreme | 32.6 | 93.9 | 25.7 | 95.6 | 25.4 | 12.0 |
| | **Overall** | 52.8 | 96.5 | 44.5 | 96.2 | 47.5 | 33.4 |

### Table 9: HUGSIM by source dataset, PhysWAM (RC / HD-Score)

| Dataset | Episodes | Easy | Medium | Hard | Extreme | All |
|---|---:|---:|---:|---:|---:|---:|
| nuScenes | 88 | 94.5 / 88.4 | 43.7 / 27.1 | 59.5 / 43.1 | 40.6 / 27.3 | 56.7 / 42.9 |
| Waymo | 108 | 94.6 / 91.6 | 51.5 / 34.1 | 57.3 / 44.8 | 23.9 / 8.7 | 55.8 / 42.8 |
| KITTI-360 | 113 | 87.4 / 76.5 | 51.6 / 39.3 | 14.9 / 6.1 | 8.0 / 0.7 | 41.1 / 31.0 |
| PandaSet | 127 | 97.4 / 91.1 | 42.8 / 20.1 | 22.5 / 6.6 | 32.0 / 16.9 | 41.3 / 24.3 |

### Table 10: The LoRA ablation setting

| | |
|---|---|
| Adaptation | LoRA, rank 128, α = 256, on the attention projections |
| Trained parameters | 122.7M |
| Updates × batch | 16,000 × 44 windows |
| Optimizer | AdamW, β = (0.9, 0.99), weight decay 0.05, clip 1.0 |
| Learning rate | peak 2e-4, 300 warm-up updates, linear decay to 0 |
| Weight averaging | none |

### Table 11: LoRA setting, all sub-scores (navtest, one sample)

| Row | NC | DAC | DDC | TLC | EP | TTC | LK | HC | PDMS | EPDMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flow | 95.9 | 96.4 | 98.6 | 99.7 | 86.0 | 94.9 | 95.2 | 98.4 | 85.7 | 84.3 |
| flow + hinge | 96.8 | 96.6 | 98.5 | 99.8 | 86.8 | 95.9 | 95.3 | 98.6 | 87.2 | 85.7 |
| flow + CPP | 96.8 | 97.0 | 98.5 | 99.7 | 87.0 | 95.7 | 95.1 | 98.6 | 87.8 | 86.2 |
| flow + hinge + CPP (multi-view) | 96.9 | 97.2 | 98.5 | 99.7 | 87.2 | 95.8 | 95.1 | 98.6 | 88.2 | 86.5 |
| single-view | 96.6 | 96.7 | 98.6 | 99.7 | 86.8 | 95.6 | 95.2 | 98.5 | 87.1 | 85.6 |
| multi-view, no depth | 96.9 | 96.8 | 98.5 | 99.7 | 87.0 | 95.9 | 95.2 | 98.6 | 87.5 | 85.9 |

### Table 12: Scores by driving command (2,501 left / 8,070 straight / 1,575 right)

| Row | PDMS L | PDMS S | PDMS R | EPDMS L | EPDMS S | EPDMS R |
|---|---:|---:|---:|---:|---:|---:|
| flow | 81.4 | 87.8 | 81.9 | 80.3 | 86.2 | 81.3 |
| flow + hinge | 83.4 | 89.0 | 83.8 | 82.4 | 87.2 | 82.8 |
| flow + CPP | 83.5 | 89.9 | 83.9 | 82.1 | 88.1 | 83.0 |
| flow + hinge + CPP (multi-view) | 83.9 | 90.3 | 84.3 | 82.4 | 88.4 | 83.3 |
| single-view | 81.2 | 90.0 | 82.0 | 80.1 | 88.2 | 81.4 |
| multi-view, no depth | 83.5 | 89.4 | 84.0 | 82.4 | 87.6 | 83.0 |
| PhysWAM (full fine-tune) | 87.8 | 92.7 | 90.4 | 87.3 | 91.4 | 89.4 |

### Table 13: Generated depth with and without CPP (final models, navtest, against LiDAR)

| View | AbsRel +2 s (with / without) | δ₁.₂₅ +2 s | AbsRel +4 s | δ₁.₂₅ +4 s |
|---|---:|---:|---:|---:|
| Front | 0.175 / 0.193 | 0.814 / 0.802 | 0.232 / 0.253 | 0.742 / 0.743 |
| Front-left | 0.292 / 0.315 | 0.725 / 0.706 | 0.374 / 0.404 | 0.654 / 0.649 |
| Front-right | 0.240 / 0.277 | 0.752 / 0.725 | 0.305 / 0.346 | 0.676 / 0.663 |
| Front, CPP cells | 0.144 / 0.154 | 0.856 / 0.844 | 0.194 / 0.205 | 0.795 / 0.795 |

### Table 14: Front-view video quality on NAVSIM

I3D FVD over one context and eight generated frames at 2 Hz. "600": generated clips of half the scenes against recorded clips of the other half, ten splits. "1,200": generated against recorded clips of the same scenes. Other rows use their own clip counts and lengths.

| | Clips | FVD ↓ | FID ↓ |
|---|---:|---:|---:|
| DrivingGPT | 512 | 142.6 | 12.8 |
| PWM † | – | 86.0 | – |
| DriveDreamer-Policy | – | 53.6 | – |
| CoWorld-VLA | – | 32.7 | – |
| **Recorded (floor)** | 600 | **91.5 ± 3.2** | 11.0 ± 0.3 |
| PhysWAM | 600 | 111.3 ± 6.8 | 15.4 ± 0.6 |
| PhysWAM | 1,200 | 42.2 | 6.8 |

### Table 15: Consistency of the generated future (navtest, medians)

| Quantity | Floor | PhysWAM | Excess |
|---|---:|---:|---|
| Yaw, video vs. generated motion, 4 s (°) | 0.29 | 0.80 | +0.44 [+0.39, +0.48] |
| Same, turns above 45° | 6.11 | – | +2.31 [+1.93, +2.86] |
| Cross-view $\lvert\Delta\log z\rvert$, +2 s, L / R | 0.032 / 0.030 | 0.045 / 0.049 | 1.5× |
| Temporal $\lvert\Delta\log z\rvert$, 0 to 4 s | 0.087 | 0.285 | 3.3× |

Yaw is recovered from the generated video by MapAnything. Floors come from recorded clips and from two LiDAR sweeps 0.5 s apart.

### Table 16: Sampler steps and cost (one RTX PRO 6000)

| UniPC steps | Δ PDMS vs. 30 steps | Samples averaged | GPU-s per scene |
|---:|---:|---:|---:|
| 4 | −0.6 | 1 | 3.5 |
| 8 | +0.7 | 3 | 4.4 |
| 15 | −0.4 | 1 | 6.0 |
| 30 | 0 | 4 | 9.4 |

### Table 17: How future prediction connects to planning (the paper's taxonomy)

| Representative works | Predicted world | Connection to action |
|---|---|---|
| LAW, SimWAM | Future features or video | Training supervision only; planning needs no explicit future |
| DriveWAM, GeoWAM | Future video or 3D geometry | The predicted future conditions trajectory generation |
| Epona, DriveVA | Future video | Video and action generation share representations |
| 4D-WAM | Future video | Joint learning supervised by matching geometry recovered from generated and recorded video |
| WoTE, DA-WAM | Future scene features | A learned scorer evaluates candidates from their predicted futures |
| **PhysWAM** | Multiview video and metric depth | Co-denoised with ego motion; CPP supervises depth and motion jointly against measured geometry; label-free consensus selection |

---

## Reading the Results

### 1. What the CPP ablation establishes

| Setting | Without CPP | With CPP | Δ |
|---|---:|---:|---:|
| LoRA, flow only → flow + CPP (PDMS) | 85.7 | 87.8 | +2.1 |
| LoRA, flow + hinge → + CPP (PDMS) | 87.2 | 88.2 | +1.0 |
| Full recipe, navtest PDMS / EPDMS | 89.6 / 88.4 | 91.4 / 90.3 | +1.8 / +1.9 |
| Full recipe, navhard | 36.2 | 38.1 | +1.9 |
| Full recipe, HUGSIM HD-Score | 33.4 | 35.5 | +2.1 |
| Final pose within 10° and 0.3 m of the recording | 70.9% | 81.4% | +10.5 pts |
| Front depth AbsRel at +2 s | 0.193 | 0.175 | −9% |

- **The effect is consistent in size across three benchmarks.** Against a sampler sd of 0.24–0.30 it is six or more standard deviations on navtest, from one training run per arm.
- **It is equal on navtest and navhard.** The mask-family mechanisms in [[concepts/wam-attention-masks.md]] are near zero on navtest and several points on navhard. CPP is +1.9 on both, which suggests it improves the trajectory itself and not only robustness under observation shift.
- **Video quality is unchanged** with or without CPP (FVD and FID within sample noise), while planning moves by 1.9. This is a second intra-paper case of generation quality and planning quality moving independently, after [[sources/reworld.md]].

### 2. What the ablation does not establish {#coupling-not-isolated}

The paper names the missing experiment itself: "distinguishing the effect of the coupling from that of metric-space supervision of each branch requires a matched decoupled comparison." The residual splits exactly into a depth-only part, a motion-only part and a cross term, and no run trains on $\rho_\delta(\|\mathbf r_D\|)+\rho_\delta(\|\mathbf r_T\|)$.

Two observations in the paper's own tables bear on which part carries the planning gain. This reading is the wiki's, not the paper's.

- **The motion-only term is a pose loss with a lever arm.** $\mathbf r_T$ applies the generated pose to measured points, so a rotation error is multiplied by each point's distance. That is a much stronger heading signal than a flow-matching loss on rotation columns.
- **The early-training signature is a heading signature.** At 4,000 updates the model without CPP has 8.9° median heading error against 2.7°, leaves the road in 32.7% of scenes against 9.7%, and "drifts laterally through a less accurate heading rather than taking wrong turns."

So a plausible decomposition is that most of the planning benefit is $\mathbf r_T$, a metric pose loss weighted by scene geometry, and that the coupling in the paper's title is the smaller part. The depth improvement (8–15%) could equally come from $\mathbf r_D$ alone, which is direct LiDAR supervision of depth. **One run with the decoupled objective would decide it.**

### 3. Depth generation alone is a null {#depth-null}

| LoRA row | Depth stream | CPP | PDMS |
|---|:-:|:-:|---:|
| multi-view, no depth | ✗ | ✗ | 87.5 |
| flow + hinge | ✓ | ✗ | 87.2 |
| flow + hinge + CPP | ✓ | ✓ | 88.2 |

Adding a jointly denoised metric-depth stream that the action tokens can attend to changes PDMS by −0.3, inside the sampler noise. The paper's conclusion is the right one: "generated depth therefore helps planning through CPP rather than on its own."

This cuts against three earlier results in the wiki, none of them large: [[sources/drivedreamer-policy.md]] (depth +0.5 alone, +0.3 on top of video), [[sources/suv.md]] (three structured streams +1.0 navtest with no access) and ExploreVLA (depth +1.6 alone). PhysWAM's depth target is the most physically meaningful of the four (metric, LiDAR-anchored) and the only one that gives nothing by itself.

**Multi-view follows the same pattern.** Three views instead of one add +1.1 PDMS (+2.3 to +2.7 on turns, +0.3 on straights). Without generated depth, and so without CPP on the side views, the three views add only +0.4.

### 4. The "v1" column is not a NAVSIM-v1 measurement {#v1-column}

Table 1 prints a PDMS block and an EPDMS block. For PhysWAM's rows, two things hold that hold for no other method's row. (The Human row also matches its closed form, trivially, because its gates are all 100.)

**(a) The shared sub-scores are identical in both blocks.**

| Row | v1 TTC → v2 TTC | v1 EP → v2 EP |
|---|---|---|
| TransFuser | 92.8 → 95.4 | 79.2 → 87.1 |
| DiffusionDrive | 94.7 → 97.3 | 82.2 → 87.5 |
| BeyondDrive | 95.0 → 98.0 | 83.7 → 87.8 |
| DriveDreamer-Policy | 95.1 → 97.7 | 83.5 → 87.9 |
| DVGT-2-NAVSIM | 95.8 → 98.0 | 84.3 → 87.9 |
| *(all ten dual-protocol baselines)* | +2.2 to +3.4 | +0.1 to +7.9 |
| **PhysWAM** | 98.5 → **98.5** | 88.7 → **88.7** |
| **PhysWAM, medoid** | 98.6 → **98.6** | 89.6 → **89.6** |
| **PhysWAM, oracle** | 99.2 → **99.2** | 91.9 → **91.9** |
| **PhysWAM, no CPP** | 97.6 → **97.6** | 88.0 → **88.0** |

The two evaluators define time-to-collision and ego progress differently, which is why every other method's values differ. PhysWAM's do not.

**(b) The PDMS equals the closed form of the printed means.** NAVSIM averages per-scene products, so a reported PDMS normally sits above $\mathrm{NC}\cdot\mathrm{DAC}\cdot(5\,\mathrm{EP}+5\,\mathrm{TTC}+2\,\mathrm C)/12$ computed from mean sub-scores.

| Row | Closed form | Reported | Reported − closed |
|---|---:|---:|---:|
| 16 other methods | | | **+1.2 to +3.9** |
| PhysWAM | 91.34 | 91.4 | **+0.06** |
| PhysWAM, medoid | 91.67 | 91.7 | **+0.03** |
| PhysWAM, oracle | 95.31 | 95.3 | **−0.01** |
| PhysWAM, no CPP | 89.51 | 89.6 | **+0.09** |

**Reading.** The PDMS column appears to be the v1 formula applied to NAVSIM-v2 mean sub-scores. This is an inference from the table; the paper says only that both scores are "from the official devkit". The two departures push in opposite directions (v2 sub-scores are higher; a closed form is lower), so the true v1 score cannot be recovered from the table.

**Consequences.**
- 91.4 / 91.7 should not be placed on the NAVSIM-v1 ladder in [[concepts/navsim-benchmark.md]].
- The paper's v1 claims rest on it: "ego progress (88.7) exceeds the compared methods by 1.4 under v1" compares a v2 EP against v1 EPs. Under v2 the margin over DVGT-2's 88.4 is 0.3.
- "Oracle-of-8 95.3, above the human 94.8" compares across the same gap.
- The relative results inside the paper (with and without CPP, the LoRA ablations) are unaffected, because every PhysWAM row uses the same construction.

### 5. The v2 result and its table

**90.3 EPDMS looks like a corrected-evaluator number.** The closed form of its sub-scores is 90.1, a residual of −0.2, which is the corrected side of the wiki's [residual-sign heuristic](../concepts/navsim-benchmark.md#residual-sign). EC 90.5 is the highest extended comfort of any 90+ entry in the wiki and is identical with and without CPP. *(Lint 2026-09-30: [[sources/momworld.md]], ingested later and held at low confidence, prints 90.6 at 90.1 EPDMS.)*

**The table mixes evaluators**, in one unlabelled column:

| Row | Reported | Residual (closed − reported) | Side |
|---|---:|---:|---|
| TransFuser | 76.7 | +1.3 | pre-fix |
| Hydra-MDP++ (R34) | 81.4 | +2.2 | pre-fix |
| DriveSuprim (R34) | 83.1 | +2.3 | pre-fix |
| DiffusionDrive | 84.5 | +2.5 | pre-fix-like (a known exception) |
| LTF | 83.6 | −1.1 | corrected-like |
| DriveVLA-W0 (anchors) | 86.1 | −1.3 | corrected-like (a known exception) |
| DriveDreamer-Policy | 88.7 | −0.9 | corrected-like |
| DVGT-2 / DVGT-2-NAVSIM | 88.9 / 89.6 | −1.0 / −0.8 | corrected-like |
| BeyondDrive | 90.1 | −0.6 | corrected-like |
| GeoWAM | 90.2 | −0.7 | corrected-like |
| PhysWAM | 90.3 | −0.2 | corrected-like |

- **LTF at 83.6 against TransFuser at 76.7** is the visible symptom: the camera-only variant scores 6.9 above the camera+LiDAR model it is derived from. About 4.5 of that is sub-scores; the rest is the evaluator.
- It is the **eighth user of the shared baseline block** (TransFuser 76.7, Hydra-MDP++ 81.4, DriveSuprim 83.1, ARTEMIS 83.1, DiffusionDrive 84.5, DriveVLA-W0 86.1), and it labels the two ResNet-34 rows as such.
- It prints **ARTEMIS with HC blank and EC 98.3**, a new variant of that row's known duplication.
- It separates **DVGT-2 (88.9)** from **DVGT-2-NAVSIM (89.6)**. The row the wiki has carried as "DVGT-2 89.6" is the second.

**What the comparison omits.** Among methods with no scorer and no RL: [[sources/wa-jepa.md]] 91.7, [[sources/suv.md]] 91.0, CoWorld-VLA 90.0 (cited, for its FVD only). With a scorer: [[sources/geoworldad.md]] 90.4 (cited, not tabled), SparseDriveV2 90.1 (cited, not tabled), DriveFuture 89.9 (cited for navhard only). The paper's claim is hedged ("among the highest reported values for methods trained without a learned trajectory scorer or reinforcement-learning post-training") and survives as worded. "The best … world-action models … lie below" does not.

### 6. navhard

- **38.1 single-sample is second among unscored methods** in the wiki, behind [[sources/spanvla.md]] at 40.1 (RL with negative-recovery samples) and ahead of SUV 36.9 and GeoWAM 36.6. The medoid's 39.8 is still below SpanVLA. The table omits SpanVLA, SUV and the whole scorer cohort (42–55.5).
- **DriveFuture is cited at 34.6, its unscored value**, which is the fair comparison for a no-scorer method and independently confirms the wiki's decomposition of DriveFuture's 55.5.
- **Stage 2 is where it leads**: NC 83.9, DAC 80.2 and TTC 81.6 are the best in its table. Stage-2 lane keeping is 48.9, inside the 45–50 band every unscored method falls into.
- **Extended comfort does not survive the split.** EC is 90.5 on navtest, 67.6 on navhard Stage 1 and 54.4 on Stage 2.
- **Split size**: 450 Stage-1 and 5,462 Stage-2 scenes, agreeing with SUV against Metis's 244 / 4,164.
- **The combined score is printed with both stage scores**: 77.0 and 48.8 give 38.1, where the product of the means is 37.6. The paper states why: "the combined score multiplies the two stages per scene before averaging."
- **It crosses two baseline lineages.** See [[concepts/navhard-ood-evaluation.md#two-lineages]].

### 7. HUGSIM

- **48.9 RC / 35.5 HD-Score, zero-shot from NAVSIM-only training.** The lead over BeyondDrive is printed conservatively: BeyondDrive's 34.8 is an unweighted mean of four levels, PhysWAM's 35.5 is episode-weighted. On one convention the gap is 38.9 vs 34.8 (unweighted) or 35.5 vs 33.0 (weighted).
- **The HD-Score lead is on Easy only** (86.9 against 65.6). On Medium, Hard and Extreme, BeyondDrive has the higher HD-Score, and UniAD keeps higher NC and TTC. The paper says so.
- **[[sources/wa-jepa.md]] is not cited** and reports 44.6 HD-Score on the same 436 scenarios under a pinned commit. By tier: WA-JEPA 79.8 / 55.6 / 30.6 / 13.6 against PhysWAM 86.9 / 30.1 / 25.2 / 13.4. PhysWAM is ahead on Easy and 25 points behind on Medium. The two protocols are probably close but not verified identical: PhysWAM's baselines are "as reported by the benchmark paper" (UniAD 28.9, LTF 24.8, VAD 12.3), where WA-JEPA's rescoring gives 31.2, 23.1 and 13.9. PhysWAM names no commit.
- **Comfort is 96.3**, against 66.2 for WA-JEPA and 58–83 by tier for UniAD. A sampled flow-matching planner that generates full video is not inherently uncomfortable in closed loop.
- **DAC is above 93.8 at every difficulty**; the score is lost on NC and TTC.
- **KITTI-360 Hard and Extreme collapse** (6.1 and 0.7 HD-Score).
- **Not real time.** "The simulator advances only after the planner returns." One plan is 9.4 GPU-seconds against a 0.5 s replanning interval.
- CPP is worth +2.1 HD-Score here. Keeping CPP active after update 20,000 changes nothing open-loop and is worth +1.2 HD-Score and +1.1 RC in closed loop.

### 8. Consensus selection against the oracle {#medoid}

| Selection among 8 samples | Needs | PDMS | EPDMS | navhard |
|---|---|---:|---:|---:|
| One sample | – | 91.4 | 90.3 | 38.1 |
| **Medoid** | Nothing | 91.7 | 90.4 | 39.8 |
| Oracle | The simulator | 95.3 | 94.1 | – |

- The medoid recovers **+0.3 of +3.9 PDMS (8%)** and **+0.1 of +3.8 EPDMS (3%)**. On navtest that is about one sampler standard deviation.
- It trades progress for lane keeping under v2 (EP +0.9, LK −0.7).
- **On navhard it is worth +1.7**, with gains in both stages and in EC (67.6 → 72.4 in Stage 1). That is small next to a learned scorer (+20.9 in DriveFuture) but it costs no labels.
- The oracle gap shows eight samples from this model are diverse enough to contain much better plans. Agreement among samples is a weak proxy for which one is good.

### 9. Variance

- **Sampler sd 0.24 PDMS / 0.30 EPDMS.** WA-JEPA measured 0.053 and CoWorld-VLA 0.013. The difference is plausibly that PhysWAM samples the whole multi-view future from noise, without anchors or a deterministic head.
- **The step sweep is inside that noise and non-monotone**: 4 steps −0.6 (one sample), 8 steps +0.7 (three), 15 steps −0.4 (one). The 30-step headline is not the best row.
- **One training run per configuration**, stated as a limitation. Run-to-run variance is not reported.

### 10. World-model quality, measured against real references

This is the most careful generation evaluation in the wiki.

- **Depth is scored against LiDAR**, with no scale alignment, and the teacher labels are separately validated against LiDAR (Table 5: AbsRel 0.041–0.060 on CPP cells). Compare [[concepts/teacher-pseudo-labels.md]], where two papers score depth against the teacher that labelled it.
- **FVD is reported with a floor.** Recorded clips against other recorded clips score **91.5 at 600 clips**. PhysWAM is 111.3 at that count and **42.2 at 1,200 clips**. "The recorded floor shows how much of any reported value is sample size." The NAVSIM FVD values in the wiki (32.7 to 142.6) carry no clip counts for three of four methods.
- **Generated video is checked against generated motion.** Yaw recovered from the video deviates from the generated motion by a median 0.80° over 4 s (floor 0.29°). Where generated and recorded motion differ by more than 2°, the video follows the generated motion in 67% of scenes. In sharp turns the video realizes about 96% of the turn.
- **Depth is less consistent over time than across views.** Front and side depth agree to 1.5× a LiDAR floor; warped over 4 s with the generated motion, disagreement reaches 3.3× the floor.
- **The cross-dataset depth comparison is labelled as such.** PhysWAM's 0.175 / 0.232 is on NAVSIM; GeoWAM's 0.245 / 0.297 is on nuScenes. The same base model with a DVGT head (Cosmos 3 + DVGT) is the worst row of that nuScenes table.

### 11. Cost

| | |
|---|---|
| Parameters | 15.2B total, 7.0B trained |
| One plan, 30 steps | 9.4 GPU-s |
| One plan, 8 steps | 4.4 GPU-s |
| Medoid of 8, 30 steps | about 75 GPU-s |
| All of navtest, one sample | about 32 GPU-hours |

An action-only inference path "was not tested". With bidirectional attention between action and future tokens there is no mask that would allow one without retraining.

### 12. LoRA against full fine-tuning

Rank-128 LoRA (122.7M parameters, 16,000 updates, no EMA) reaches 88.2 PDMS; the full fine-tune (7.0B, 30,000 updates, EMA) reaches 91.4. The +3.2 is larger than CPP's +1.8, and it is confounded by training length, learning rate and weight averaging. The right-turn gap is the largest (84.3 → 90.4).

---

## Relationships

- **[[sources/geowam.md]]**: the closest prior work and the main baseline on navhard (36.6) and future depth. GeoWAM forecasts metric point maps in the ego frame and **stops the trajectory loss from reaching the geometry**. PhysWAM does the opposite: one loss whose gradient reaches both. GeoWAM has no ablation; PhysWAM has one and it is worth +1.9.
- **[[sources/drivedreamer-policy.md]]**: depth → video → action with three separate generators, depth scored against its own teacher. PhysWAM puts all three in one denoiser and scores depth against LiDAR. It finds depth alone worth nothing where DriveDreamer-Policy found +0.5.
- **[[sources/suv.md]]**: not cited, and the nearest design on the generation side. Both write depth as a video and push it through the frozen Wan VAE so one video expert generates it. SUV's depth is relative and teacher-scored; PhysWAM's is metric and LiDAR-scored. SUV uses a one-way mask (action reads future); PhysWAM is bidirectional.
- **[[sources/wa-jepa.md]]**: not cited. It leads PhysWAM on corrected navtest EPDMS (91.7) and on HUGSIM (44.6), with a JEPA latent future, no depth and no LiDAR supervision. It is the comparison the paper most needs.
- **[[sources/simwam.md]] / [[sources/metis.md]]**: SimWAM is cited as the training-time-only alternative. PhysWAM uses the bidirectional mask that Metis found worst on navhard, runs no mask ablation, and reports the best unscored navhard result outside SpanVLA.
- **[[sources/drivefuture.md]]**: cited at its unscored navhard value, 34.6.
- **[[sources/coworld-vla.md]]**: cited for its FVD of 32.7 with the note that its clip count is unstated. Its 90.0 EPDMS is not in Table 1.
- **[[sources/reworld.md]]**: both papers improve planning through action-side objectives while generation quality is flat or moves separately.
- **[[sources/ad-e2e-jepa.md]]**: the other no-scorer selector ingested the same day. AD-E2E-JEPA selects by distance to an oracle future frame; PhysWAM selects by agreement among its own samples. Neither comes close to a learned scorer.
- **[[sources/adaptive-wam.md]]**: measured a full video rollout at 13.22 s and planned from an intermediate feature at 170 ms. PhysWAM pays the full-rollout price by design.
- **Un-ingested and load-bearing**: **BeyondDrive** (89.7 / 90.1 navtest, 34.8 HUGSIM; "learning from hard negatives"), **4D-WAM** (35.9 navhard; geometry recovered from generated video as supervision, the nearest competing idea), **Cosmos 3** (the backbone), **MapAnything** (the depth teacher and the consistency probe), **DVGT-2**, **EponaV2**.

---

## Limitations

**Attribution**

1. **The coupling is not isolated.** No decoupled control ($\mathbf r_D$ and $\mathbf r_T$ penalized separately) is run. The paper says so in §4.2 and Appendix A.2.
2. **One training run per configuration**, with a sampler sd of 0.24–0.30. The LoRA ablation deltas of 0.3–0.4 (hinges on top of CPP, depth without CPP) are at the noise level.
3. **The LoRA-to-full-fine-tune gap (+3.2) is confounded** and is larger than the named mechanism.
4. **No mask or inference-path ablation.** Whether the 9.4-second generated future is needed at inference is untested.

**Reporting**

5. **The v1 PDMS column is not a v1 measurement** (see [above](#v1-column)). v1 claims and the oracle-versus-human comparison depend on it.
6. **The v2 table mixes evaluators** in one column and omits WA-JEPA 91.7 and SUV 91.0.
7. **The navhard table omits SpanVLA 40.1 and SUV 36.9** and crosses two baseline lineages.
8. **HUGSIM omits WA-JEPA** (44.6) and names no commit; baselines are quoted, not rescored.
9. **GPU count and training time are not given.**

**Method**

10. **Training needs an instrumented vehicle.** LiDAR calibrated to cameras, accurate ego poses, annotated 3D boxes, an HD map, plus two geometry foundation models for dense depth labels. The paper lists this first among its limitations.
11. **The hinges are differentiable versions of two benchmark gates.** The drivable-area field is built from the same map layers as NAVSIM's DAC, and the obstacle term uses annotated boxes. "No learned scorer or simulator feedback" is true at inference; training is not label-free.
12. **Consistency is supervised only along the recorded future.** Counterfactual commands and horizons beyond 4 s are unsupervised and measured indirectly.
13. **Inference cost**: 9.4 GPU-seconds per plan, 75 for the medoid. The closed-loop evaluation does not establish real-time operation.
14. **Temporal depth consistency is 3.3× the LiDAR floor** over 4 s, and generated surfaces sit 0.2–0.3 m nearer than LiDAR in the median.
15. **Closed-loop strength is confined to Easy scenarios**; KITTI-360 Hard and Extreme are near zero.

**Source conversion**

16. Figure 1's asset holds two of the four rows its caption describes. One sentence of Appendix C.3 is scrambled in the clipping (the replanning interval). The author field in the front matter is empty but the author line is present in the body. All 17 tables are present.

**Disclosure.** The paper's AI-use statement says AI models "including Claude and ChatGPT" were used "for writing and polishing the paper, for implementing the method, and for running and monitoring the experiments."

---

## Key Cross-References

- [[concepts/world-model-for-ad.md]] — Pattern 39: depth and ego motion tied by one metric residual; depth generation as a null without it; video–motion consistency measured.
- [[concepts/navsim-benchmark.md]] — the 90.3 row; why 91.4 stays off the v1 table; an eighth user of the shared block.
- [[concepts/navhard-ood-evaluation.md]] — 38.1 / 39.8 in the unscored cohort; the 450 / 5,462 split; two baseline lineages and the origin of LTF 25.1.
- [[concepts/hugsim-benchmark.md]] — the second method on the 436-scenario set; aggregation conventions; the comfort counterexample.
- [[concepts/best-of-n.md]] and [[concepts/selection-based-planning.md]] — oracle-of-8 against the medoid.
- [[concepts/evaluation-variance.md]] — sampler sd 0.24–0.30; an FVD floor.
- [[concepts/inference-latency.md]] — 9.4 GPU-seconds per plan.
- [[concepts/teacher-pseudo-labels.md]] — a teacher label pipeline validated against LiDAR.
- [[concepts/foundation-backbones-for-ad.md]] — Cosmos 3 Nano as a policy core; LoRA against full fine-tuning.

## Follow-Up From the Same Group: DriveReferee {#drivereferee}

[[sources/drivereferee.md]] (2609.22762; all eight of its authors are authors of this paper) keeps the Cosmos 3 backbone and drops this paper's depth stream, side cameras and CPP loss. It bears on four items on this page:

- **The base model.** A one-camera, imitation-only fine-tune scores 91.08 PDMS / 90.69 EPDMS, 0.4 EPDMS above the full PhysWAM. PhysWAM is not in its comparison table.
- **The v1 column.** DriveReferee's v1 sub-scores have v1-scale ego progress (84–85) and its PDMS sits 0.9–1.5 above the closed form, as a measured v1 score should. The anomaly recorded above for PhysWAM's "PDMS" does not recur.
- **Selection.** The label-free medoid here is replaced by a geometric rule on a predicted map: +0.30 EPDMS with two samples, against +0.1 for the medoid of eight. Oracle headroom is not re-measured.
- **FVD.** DriveReferee reports 23.90 with no clip count and no recorded-against-recorded floor, the two pieces of context this paper introduced.
