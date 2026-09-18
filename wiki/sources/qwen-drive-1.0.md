---
title: "Qwen-Drive-1.0: An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving"
type: source-summary
sources: [raw/papers/Qwen-Drive-1.0_ An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving.md]
related: [concepts/vlm-domain-adaptation.md, concepts/general-capability-retention.md, concepts/alpasim-benchmark.md, concepts/foundation-backbones-for-ad.md, concepts/perception-for-planning.md, concepts/navsim-benchmark.md, concepts/rl-for-ad.md, concepts/diffusion-planner.md, concepts/chain-of-thought-for-ad.md, concepts/physicalai-av-benchmark.md, concepts/nuscenes-waymo-evals.md, concepts/best-of-n.md, concepts/world-model-for-ad.md, concepts/dual-system-vla.md, concepts/action-tokenization.md, concepts/mixture-of-experts.md, sources/alpamayo-r1.md, sources/unidrivevla.md, sources/automot.md, sources/percept-wam.md, sources/drivewam.md, sources/simwam.md, sources/spanvla.md, sources/autovla.md, sources/nord.md, sources/dial.md, sources/hermes.md, sources/recogdrive.md, sources/explorevla.md, sources/coworld-vla.md, sources/drivelaw.md, sources/drive-hwm.md, sources/epona.md, sources/diffusiondrive.md, sources/onedrive.md, sources/sgdrive.md, sources/adaptive-wam.md, sources/geoworldad.md, sources/latent-wam.md, sources/drivevla-w0.md, sources/da-wam.md, sources/foresight.md, sources/futuresightdrive.md, sources/wcog-vla.md]
created: 2026-09-18
updated: 2026-09-18
confidence: high
---

**Paper**: Qwen-Drive-1.0: An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving
**Authors**: Xin Zhou, Zongchuang Zhao, Zhibo Yang, Mingsheng Li, Humen Zhong, Shuai Bai, Dingkang Liang, Xiang Bai, Dayiheng Liu (core); Du Chu, Ruizhe Chen, Zhaohai Li, Jun Tang, Qiuyue Wang, Mingkun Yang, Jiazhao Zhang
**Orgs**: Qwen Team (Alibaba) + Huazhong University of Science and Technology
**arXiv**: 2609.00111v1
**Code/weights**: **released** — `github.com/QwenLM/Qwen-Drive-1.0`, `huggingface.co/Qwen/Qwen-Drive-1.0-4B`, ModelScope mirror

---

## Source Integrity Note

The clipping is complete from abstract through Appendix C. **Eleven of the thirteen numbered figures are present.** **Figures 11 and 12 — the reinforcement-learning reward/mixture ablation and the PAI-AV data-scaling curve — are captioned but have no image**, the same failure mode as [[sources/drive-hwm.md]]'s Fig. 4. Unlike that case, both are largely recoverable: the prose states Fig. 11's four PDMS values and both RFS/ADE endpoints, and Fig. 12's ADE and FDE endpoints. What is lost is the shape of the scaling curve between endpoints and any per-source RFS breakdown.

All eight tables are intact, though **Table 3 arrives flattened into prose** (the HTML column structure is gone). Its column mapping is recovered below and cross-checked two independent ways. Appendix B's six qualitative images are external arXiv URLs and were not downloaded; the prompts and full model responses are present as text and are reproduced where they carry evidence.

---

## Summary

Qwen-Drive-1.0 is the first entry in this wiki that is a **foundation-model release** rather than a planner: 4B weights and code are public, and the organizing question is not "how do we score higher on NAVSIM" but **"how much driving competence can be added to a pretrained VLM before the VLM stops being a VLM?"**

The architecture is deliberately conservative. Qwen3.5-4B is kept **exactly as pretrained** — no new tokens, no vocabulary expansion, no MoT split, no architectural surgery — and two external modules are attached:

| | **BEV perception head** | **Planning Expert** |
|---|---|---|
| Reads | vision-encoder features (lifted to a voxel volume) **and** post-VLM image-token features | cached post-RoPE keys/values from the VLM's 8 GQA-softmax layers |
| Produces | 3D boxes, 10-class semantic occupancy, 6-class BEV map raster | 50 waypoints $(x,y,\theta)$, 5 s at 10 Hz |
| Objective | focal + L1 + scene-class affinity + Lovász | flow matching with **$x$-prediction**, plus 1st/2nd-difference Huber |
| Size | not stated | 32-layer DiT, $d{=}1024$, ~1.1 B |
| Role the paper assigns it | **a probe** of what 3D information the shared representation exposes | the action head |

Trained in four stages — head-only → joint perception+VQA (encoder and VLM *unfrozen*) → Planning Expert on frozen representations → RL on the Planning Expert only.

**Headline numbers**: 43.95 mAP / 60.99 map mIoU on nuScenes and 43.45 / 71.27 on OpenScene; driving-VQA average 69.43 against the base model's 63.52; **general vision-language average 66.41 against the base model's 67.40**; **90.7 PDMS on NAVSIM** (91.4 with oracle best-of-6); 7.91 RFS on the WOD-E2E test split; and a closed-loop AlpaSim evaluation at 5.0 B parameters.

**Five readings this page arrives at:**

1. **The general-capability retention result is the contribution, and it is the first of its kind in this wiki.** Fifteen general VLM benchmarks, thirteen models, one decoding protocol. Every driving- or embodiment-specialized comparison model loses most of its general competence; Qwen-Drive loses 0.99 points on one group and *gains* on the other. See [Retention](#retention) and the new page [[concepts/general-capability-retention.md]].
2. **The BEV head answers a question this wiki has asked repeatedly: do vision-language-pretrained features contain driving 3D structure? Measurably, no.** A converged head on frozen SigLIP-Qwen features trails a dedicated detector *on the same features* by 6.34 mAP. Unfreezing recovers +10.46. See [The 3D probe](#probe).
3. **The paper's own ablation prices its entire driving adaptation at +0.08 RFS for planning** — 1.54 M curated samples, 3.09 M filtered VQA pairs, 24 public datasets and a 3D perception stack, measured against an unadapted Qwen3.5-4B under a matched Planning Expert. See [Stage 2 ablation](#stage2-ablation). The wiki's "mechanism is the smaller term" streak reaches six, and this is the largest gap between investment and measured effect it has recorded.
4. **Its AlpaSim table is the wiki's first head-to-head closed-loop reproduction of NAVSIM leaders — and the ordering nearly inverts.** [[sources/simwam.md]] (91.5 PDMS, the wiki's best world-action model) has the *worst* at-fault AlpaSim score in the table; [[sources/drivewam.md]] posts a flattering at-fault score by barely moving; Alpamayo-R1, which reports no NAVSIM result at all, leads. See [AlpaSim](#alpasim).
5. **It reverses the PhysicalAI-AV ordering that [[concepts/physicalai-av-benchmark.md]] flagged as unverified.** DriveWAM's own curated subset put DriveWAM at 0.47 ADE@3s and Alpamayo-1.5 at 0.80. On the standard split, reproduced here, DriveWAM is 0.67 and Alpamayo-1.5 is 0.35. See [PAI-AV](#paiav).

---

## Positioning

![[intro.png|Performance overview of Qwen-Drive-1.0 across driving VQA, general VQA, 3D perception, and motion planning]]

**Fig. 1**: Performance overview of Qwen-Drive-1.0 across driving VQA, general VQA, 3D perception, and motion planning.

§I makes two arguments against the prevailing recipe of adapting a general VLM through driving VQA alone:

> "Textual VQA targets do not directly constrain 3D layout, depth, or occupancy... A model adapted only through VQA can therefore produce fluent scene descriptions while remaining imprecise in 3D space."

> "Extensive domain adaptation can cause catastrophic forgetting of the general knowledge acquired during pretraining. No finite driving dataset can exhaustively represent the rare and unseen situations encountered in deployment."

The second then gets a **deployment** justification this wiki has not seen before, and it explains why the paper spends a third of its evaluation budget on MMMU and OCRBench:

> "Production vehicles are moving towards cockpit-driving integration, in which the intelligent cockpit and the driving system share a single compute platform rather than two separate domain controllers... A model that trades general capability for driving performance forfeits this benefit, because the cockpit functions would then require a separate model and additional compute."

This is a different motivation from the OOD-robustness argument that [[sources/automot.md]] and [[sources/unidrivevla.md]] use for the same concern. It is falsifiable in a way the robustness argument is not — one model or two, on one SoC — and it sets the retention target at *parity with the base model*, not "acceptable degradation."

**On the novelty claim.** "The first vision-language foundation model for autonomous driving that integrates 3D perception, driving VQA, and motion planning" is scoped by a fourth clause: *without changing the pretrained VLM architecture*. [[sources/unidrivevla.md]] unifies the same three capabilities with a 3-expert MoT; [[sources/percept-wam.md]] unifies 2D/3D perception and planning inside one VLM through world tokens. Both change the architecture. The unchanged-architecture constraint is the actual claim, and it is what makes the retention result interpretable.

---

## Method

![[qwendrive_overview.png|Unified architecture: shared vision encoder and VLM feeding text generation, an external BEV perception head, and a Planning Expert]]

**Fig. 2**: Unified architecture of Qwen-Drive-1.0 for 3D perception, visual question answering, and motion planning. A shared vision encoder and VLM support text generation, while the external BEV perception head and Planning Expert produce geometric predictions and future ego trajectories.

### Input serialization: tags, not tokens

Eight canonical view tags (`<FRONT VIEW>`, `<FRONT RIGHT VIEW>`, …) and a frame tag `frame: k` identify each image. **Both use ordinary vocabulary tokens** — no added special tokens, no embedding-table surgery. This is the cheapest possible answer to multi-view/multi-frame identification, and it stands against [[sources/futuresightdrive.md]]'s vocabulary expansion and [[sources/wcog-vla.md]]'s injected agent tokens.

The ordering is task-dependent and the paper is explicit about why:

- **Question answering — frame-major**: all views at $t{=}0$, then all views at $t{=}1$.
- **Planning — view-major**: all timesteps of the front view, then all timesteps of front-right, …

> "View-major serialization places consecutive observations from each view adjacent in the token sequence and exposes temporal variation within that view, which is important for control in dynamic environments."

A real finding hiding in a design note: **the same model wants different token orderings for perception-style and control-style tasks**, and the paper's own limitations section names the resulting format divergence as a barrier to cross-task transfer. Nothing measures the cost of either ordering.

### BEV perception head

![[head.png|Architectures of the external modules: BEV perception head fusing voxelized encoder features with a VLM-output feature pyramid, and the Planning Expert conditioning noisy trajectory tokens on cached VLM keys and values]]

**Fig. 3**: (a) The BEV perception head fuses voxelized vision encoder features with a feature pyramid of VLM outputs. (b) The Planning Expert conditions noisy trajectory tokens on cached VLM keys and values to recover a clean ego trajectory.

Two feature streams per view $i$:

- $\mathbf{F}^{v}_{i}$ — vision-encoder features, *before* the tokens enter the VLM. Low-level appearance.
- $\mathbf{F}^{m}_{i}$ — the same image tokens *after* traversing the full VLM. Scene context, and the semantic source for BEV construction.

A depth-based LSS-style view transform lifts $\mathbf{F}^{v}$ into a voxel volume with **no depth supervision** (a residual + atrous-pyramid network predicts a categorical distribution over $N_d$ bins):

$$\mathbf{V}(\mathbf{p})=\sum_{i\in\Omega(\mathbf{p})}\mathbf{D}_{i}(u_{i},v_{i},d_{i})\,\mathbf{F}^{v}_{i}(u_{i},v_{i})$$

$\mathbf{F}^{m}$ is expanded into a pyramid and aggregated onto the BEV plane by a query-based transformer **whose queries are initialized from the height-collapsed volume $\bar{\mathbf{V}}$** — geometry initializes the queries, semantics fills them. Occupancy decoding re-fuses the height-resolved $\mathbf{V}$ with the height-expanded BEV feature before a shallow 3D UNet.

The two-stream design has a consequence the paper states once and never returns to: **perception loss reaches the vision encoder twice** — directly through $\mathbf{F}^{v}$, and through the entire VLM via $\mathbf{F}^{m}$. During Stage 2 the detection, occupancy and map objectives are therefore gradient sources for the *language model*.

$$\mathcal{L}_{\mathrm{det}}=\sum_{l=1}^{L}\Big(2\,\mathcal{L}^{(l)}_{\mathrm{focal}}+0.75\,\mathcal{L}^{(l)}_{\ell_{1}}\Big),\qquad \mathcal{L}_{\mathrm{occ}}=100\,\mathcal{L}_{\mathrm{focal}}+\mathcal{L}_{\mathrm{geo}}+\mathcal{L}_{\mathrm{sem}}+\mathcal{L}_{\mathrm{lov}},\qquad \mathcal{L}_{\mathrm{map}}=100\,\mathcal{L}_{\mathrm{focal}}+\mathcal{L}_{\mathrm{lov}}$$

**No rig-specific camera embeddings.** One model trains and evaluates across the 6-camera nuScenes rig and the 8-camera OpenScene rig — the property that makes its own comparison methods un-runnable on OpenScene, and the one that enables the unseen-rig transfer in Fig. 13.

### Planning Expert

Conditional generation over a 5 s future:

$$\boldsymbol{\tau}\sim p\!\left(\boldsymbol{\tau}\mid\mathbf{s},\ell,\boldsymbol{\tau}_{\mathrm{hist}},\mathbf{n},\mathbf{e},\mathbf{r}\right),\qquad \boldsymbol{\tau}=\{(x_{k},y_{k},\theta_{k})\}_{k=1}^{50}$$

with sensor inputs $\mathbf{s}$, serialized layout $\ell$, history, navigation command, ego state, and an **optional** textual planning reason $\mathbf{r}$ (set to $\varnothing$ for 75.8 % of training samples). Normalization scales are fixed per channel: **165 m, 25 m, $\pi/2$ rad**.

The conditioning interface is the reusable part. The VLM alternates gated linear attention with grouped-query softmax attention; **the keys (post-RoPE) and values of all eight softmax layers are cached, and each cache conditions four consecutive Planning Expert layers.** Each expert layer concatenates cached KV with trajectory-token KV for joint attention. Flow time, navigation instruction and ego state enter through shared AdaLN.

This is the same family as [[sources/spanvla.md]]'s sparse-KV action bridge and [[sources/drive-hwm.md]]'s FiLM injection: **a narrow, non-token channel from a pretrained backbone into a continuous action head**, chosen so the backbone's sequence layout is never disturbed. Qwen-Drive's variant is the most direct of the three — it reads the backbone's attention memory rather than its hidden states or its outputs.

**$x$-prediction, and the reason given for it.** The expert predicts the clean endpoint $\hat{\boldsymbol{\tau}}_{1}$, not the velocity or the noise; the velocity field is induced as $(\hat{\boldsymbol{\tau}}_{1}-\boldsymbol{\tau}_{t})/(1-t)$:

> "This endpoint parameterization reduces sensitivity to sensor noise in trajectories recorded across heterogeneous datasets."

To keep the conversion well-conditioned, $\tilde t\sim\mathrm{Beta}(1.5,1.0)$ and $t=\min\{\tilde t,0.9\}$ so $1-t\ge 0.1$. This is a **multi-source training** argument for a parameterization choice, and the first in the wiki: four datasets with different ego-motion distributions and recording pipelines make a noisy endpoint less damaging than a noisy velocity.

$$\mathcal{L}_{\mathrm{plan}}=\mathcal{L}_{\mathrm{fm}}+2\times 10^{-4}\,\mathcal{L}_{\Delta^{1}}+2\times 10^{-5}\,\mathcal{L}_{\Delta^{2}}$$

Inference is a 10-step Euler integration from Gaussian noise.

### Four-stage training recipe

![[training_recipe.png|Four-stage training recipe with flames for trainable and snowflakes for frozen modules]]

**Fig. 4**: Stages 1 and 2 adapt the shared vision-language pathway; Stages 3 and 4 train the Planning Expert on top of these fixed representations.

| Stage | Trainable | Objective | Output |
|---|---|---|---|
| 1 | BEV head only | $\mathcal{L}_{\mathrm{perc}}$ | initialized head |
| 2 | **BEV head + vision encoder + VLM** | $\mathcal{L}_{\mathrm{perc}}$ on perception samples, $\mathcal{L}_{\mathrm{ntp}}$ on VL samples | the shared representation |
| 3 | Planning Expert only | $\mathcal{L}_{\mathrm{plan}}$ | **Qwen-Drive-1.0-SFT** |
| 4 | Planning Expert only | policy gradient on task rewards | **Qwen-Drive-1.0-RL** |

Two engineering details worth carrying:

- Each Stage-2 minibatch contains **both** sample types, and because they activate different pathways, **dummy inputs are fed to inactive branches** to keep the computation graph consistent across distributed workers, with the dummy outputs excluded from the loss. This is the practical cost of multi-task joint training under sharded data parallelism, and the first time a wiki source states it.
- **The BEV head runs at 20× the VLM's learning rate.** A newly initialized task module and a pretrained backbone are not co-trained at one rate.

### Stage 4: turning a deterministic flow into a policy {#rl-mechanism}

The inference sampler is deterministic once the initial noise is drawn, so it defines no transition probabilities to differentiate. The fix has three parts and is the most transferable piece of machinery in the paper.

**1. Stochasticity only at the output end.** Indexing the $K{=}10$ Euler steps from zero, noise is injected only over $\mathcal{W}=\{7,8,9\}$, with $\sigma=0.03$ there and $0$ elsewhere.

> "Under the endpoint parameterization, perturbations near $t=1$ affect the emitted trajectory more directly, while earlier perturbations are increasingly attenuated by subsequent integration steps."

**2. A restoring score so perturbed states stay near the pretrained flow.** Substituting the predicted endpoint for the unknown clean trajectory in the Gaussian conditional gives

$$s_{\theta}\left(\boldsymbol{\tau}^{(k)},t_{k}\right)=-\frac{\boldsymbol{\tau}^{(k)}-t_{k}\hat{\boldsymbol{\tau}}_{1}^{(k)}}{(1-t_{k})^{2}},\qquad \boldsymbol{\mu}^{(k)}=\boldsymbol{\tau}^{(k)}+v_{\theta}\Delta t+\frac{\sigma_{k}^{2}}{2}s_{\theta}$$

The paper is careful that this is "an approximate restoring correction rather than an exact marginal-preserving transformation," since the score is derived for isotropic diffusion while the perturbation is not isotropic — which is the next part.

**3. Exploration confined to a smooth low-frequency subspace.** This is the idea worth stealing:

> "Independent waypoint noise primarily introduces high-frequency jitter rather than meaningful maneuver diversity, making the resulting samples poorly suited to comparisons of driving quality."

With $\boldsymbol{\Phi}\in\mathbb{R}^{50\times 6}$ the first $M{=}6$ orthonormal cosine modes, $\boldsymbol{\tau}^{(k+1)}=\boldsymbol{\mu}^{(k)}+\sigma_{k}\boldsymbol{\Phi}\mathbf{Z}_{k}$. Per-waypoint perturbation averages $\sigma\sqrt{M/N}\approx 0.010$ in normalized units — **≈ 1.7 m longitudinally and 0.26 m laterally per stochastic step** under the 165 m / 25 m scales. The likelihood surrogate is evaluated in the 6-dimensional mode coordinates rather than as a full-rank density in 150-dimensional trajectory space.

**Group construction.** $G{=}8$ rollouts per scene. The **frozen VLM samples 8 independent reasoning traces**, each conditioning one rollout; all rollouts share the initial trajectory noise $\boldsymbol{\tau}^{(0)}$. Within-group diversity therefore comes from exactly two sources — sampled reasoning and the low-frequency perturbations — with everything else held identical. Advantage is group-relative (mean/std, $\epsilon_R{=}10^{-8}$), no value function.

$$\mathcal{L}_{\mathrm{rl}}=-\frac{1}{GW}\sum_{i=1}^{G}\sum_{w=0}^{W-1}\gamma^{\,W-1-w}A_{i}\log\pi_{\theta}\!\left(\boldsymbol{\tau}_{i}^{(k_{w}+1)}\mid\boldsymbol{\tau}_{i}^{(k_{w})}\right),\qquad \gamma=0.6$$

Each group is consumed by a single on-policy update, so no importance correction is needed.

**Rewards** (Appendix A), with $\Delta(n;\delta,\kappa)=(\delta-\mathrm{ADE}_n)/\kappa$:

| Source | Reward | Scenes |
|---|---|---|
| NAVSIM | $1\cdot\mathrm{PDMS}+2\,\Delta(50;2,10)$ | 15 K navtrain, balanced across navigation commands |
| WOD-E2E | $1\cdot\mathrm{RFS}+2\,\Delta(50;2,1)$ | 479 rater-annotated scenarios |
| PAI-AV | displacement only: $\Delta(50;2,1)+\sum_{h}w_h\Delta(10h;2,\kappa_h)$, effective coefficients **2.5 / 1 / 0.5 / 0.25** at 1–4 s against 1 at 5 s | 15 K |

Because the group-relative advantage standardizes rewards within a group, "the offsets $\delta$ therefore affect only the logged reward magnitude, while the ratios $w/\kappa$ determine the relative influence of the terms" — a clean statement of a fact most GRPO reward tables leave implicit.

**One mismatch to flag**: inside the NAVSIM reward the sub-score weights are **EP 6 / TTC 4 / Comf. 2**, while the reported evaluation uses the official protocol. The RL objective deliberately over-weights progress relative to the metric it is scored on.

---

## Data Recipe

### Perception: two kinds of unification

![[occ_process_vis.png|Cross-dataset label unification: task-specific alignment and completion strategies, and original vs. processed occupancy labels]]

**Fig. 5**: (a) Task-specific alignment and label-completion strategies. (b) Original and processed occupancy labels for nuScenes (top) and OpenScene (bottom).

Sources: nuScenes (28 K train / 6 K val keyframes, 6 cameras, occupancy from nuScenes-OccNet) and OpenScene (607 K train / 9 K val after holding out **16 logs balanced for city and time of day**, 8 cameras).

**Label unification** at "the coarsest mutually compatible granularity":

- **Detection** — 7 classes (`vehicle`, `bicycle`, `generic_object`, `pedestrian`, `traffic_cone`, `barrier`, `czone_sign`). nuScenes' five vehicle types merge into one; bicycle and motorcycle merge; `czone_sign` is supervised by OpenScene only.
- **Occupancy** — 10 classes (the 7 above plus `driveable`, `background`, `empty`).
- **Map** — 6 classes rasterized **online** from vector maps over $x\in[-30,30]$, $y\in[-15,15]$ at 0.15 m, giving a $400\times200$ target rather than a city-scale raster.
- **Offline completion** — OpenScene has no `driveable` class, so nuPlan's vector map is rasterized and a voxel is relabelled `driveable` only if it is a ground voxel inside a driveable region *and was already `background`*. nuScenes has no `generic_object` occupancy, so pseudo-labels come from 3D boxes of bicycle racks, debris and pushable objects, **changing only voxels that already carry a semantic label**. Class frequencies are recomputed afterwards for the balanced focal loss.

The honesty here is worth recording:

> "Label unification does not remove noise from the source annotations... Sensor and registration errors can therefore introduce artifacts, including floating voxels detached from physical surfaces. Offline completion adds missing semantic labels but retains these artifacts."

**Spatial unification** is the more reusable trick. Both sources store $200\times200\times16$ grids, but nuScenes spans $\pm40$ m with $z\in[-1.0,5.4]$ at 0.4 m and a **non-identity LiDAR→ego transform with a ~1.84 m vertical offset**, while OpenScene spans $\pm50$ m with $z\in[-4.0,4.0]$ at 0.5 m and an identity transform. Rather than resampling categorical labels (which distorts supervision) or sharing voxel indices (which misaligns), **a single differentiable trilinear sampling maps the predicted volume onto whichever native grid the sample came from**, with an expanded vertical source range of $[-5.0,5.4]$ m to cover the mounting offset. One head, two native grids, no dataset-specific branches.

The same care shows up in the metric: OpenScene stores occupancy in the rear-axle frame with an identity LiDAR transform, so RayIoU rays would originate at road level and terminate immediately on road voxels. **The ray origin is raised by 1.84 m** to match nuScenes' LiDAR height. This is a protocol correction other papers using OpenScene RayIoU should adopt.

### Vision-language data

![[data_analysis.png|Vision-language data composition: input-format distribution of the 3.09M filtered samples, Stage 2 mixture, and representative scenes]]

**Fig. 6**: (a) Input-format distribution of the 3.09 M filtered public driving samples. (b) Composition of the 1.54 M Stage 2 training set before repetition. (c) Representative scenes spanning diverse road environments, illumination and weather.

**24 public driving VQA datasets** are aggregated — CODA-LM, DRAMA, DriveAction, DriveGPT4, DriveLM, DrivingVQA, Impromptu VLA, LingoQA, MapLM, MM-AU, NAVSIM-ReCogDrive, NuInstruct, NuPlanQA, nuScenes-MQA, nuScenes-QA, the OOD-reasoning subset of PhysicalAI-AV, OmniDrive, ROADWork, Senna, STSBench, SURDS, SUTD-TrafficQA, Talk2Car, WaymoQA — training splits only.

The two-model pipeline is the part to reuse:

1. **Qwen3.5-Plus rewrites** every prompt and response into a common conversational schema, normalizes boxes to $[0,1000)$, inserts view/frame tags and rewrites textual view references to match. Most multiple-choice items become open-ended; a subset is retained as MCQ "to preserve this instruction type."
2. **Qwen3.5-Flash then checks** whether each rewritten response is semantically consistent with the *source annotation*, which the rewrite never validated. **5.53 M → 3.09 M, a 55.9 % retention rate.**

**Nearly half of the public driving VQA corpus fails a consistency check against its own source annotation.** That is the most quotable data-quality number in this wiki, and it is measured across 24 datasets rather than asserted.

Retained format mix: 61.6 % multi-view, 20.1 % single-view, 10.1 % single-view temporal, 4.4 % multi-view temporal, 3.7 % video.

**Self-constructed data**, three components:

1. **Chain-of-Causation planning reasoning**, explicitly "inspired by Alpamayo-R1" ([[sources/alpamayo-r1.md]]). Built from NAVSIM, Waymo and PAI-AV: a rule-based classifier derives longitudinal and lateral maneuver components from the recorded future to form a **motion prior**, then Qwen3.7-Plus writes the trace conditioned on images, history, motion prior and navigation command. The audit is the interesting part — **judges answer classification questions rather than emit scalar quality scores**, and decisions are aggregated programmatically: check the predicted maneuver against GT, classify the causal role of each cited factor, reject traces that reveal future information. Qwen3.5-Flash assigns a rarity score and rare scenes get sampling priority. Two response formats per accepted trace: trace only, and trace followed by a JSON trajectory.
2. **Camera ordering.** Surround views are shuffled and *all view tags removed*; the model must identify the front view from visual cues and recover the clockwise order. A self-supervised cross-view spatial task with no annotation cost — new to this wiki.
3. **30 K in-house China perception QA** for traffic-light grounding and 3D detection, with camera pose supplied as text and boxes returned in the global frame.

**Stage 2 mixture**: ~20 % stratified subsample of each public source + self-constructed + general-purpose VL = **1.54 M before repetition** (9.7 % perception / 26.0 % general VL / 64.3 % driving VL), becoming **12.7 % / 31.0 % / 56.3 %** after group-specific repetition factors that favour perception.

### Planning data

~2.83 M samples: NAVSIM + OpenScene 890 K (2.5 K clips), WOD-E2E 557 K (2 K clips), PAI-AV 1.38 M (156 K clips). **685 K (24.2 %) carry an accepted reasoning trace** — 78 K NAVSIM, 142 K WOD-E2E, 465 K PAI-AV.

NAVSIM and OpenScene are kept as *separate sources despite both deriving from nuPlan*, "because their ego-motion distributions differ" — a distinction most multi-source planners do not draw.

The WOD-E2E preprocessing is a small forensics exercise worth flagging for anyone using that dataset: positions at 4 Hz are spline-fitted and resampled to 10 Hz with derivatives giving velocity and acceleration; heading rate is $\dot\theta=(v_xa_y-v_ya_x)/\|\mathbf v\|^2$, zeroed below 0.3 m/s; samples are dropped if any acceleration exceeds $9.8\,\mathrm{m/s^2}$ or either heading-rate check exceeds 1.2 rad/s; and — **"we apply a factor-of-four scale correction to the raw `accel_x` and `accel_y` metadata."** A published dataset field is being silently corrected by 4×, with no citation or issue reference.

Inputs per planning example: front, front-left and front-right at 4 timesteps (current + 3 history at 0.5 s), **history at 320p and the current frame at 720p** — a deliberate token-budget allocation toward the present.

---

## Results

### Table 1 — Unified 3D perception

Comparison methods are reproduced by the authors on the remapped nuScenes annotations under a common 24-epoch schedule at $896\times512$. "SigLIP-Qwen" is the SigLIP-style encoder initialized from the same Qwen3.5-4B weights. Greyed values in the original mark cross-dataset evaluation of the nuScenes-only head.

| Method | Encoder | nuS mAP | nuS NDS | nuS Map | nuS Occ | nuS RayIoU | OS mAP | OS NDS | OS Map | OS Occ | OS RayIoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BEVFormerV2 | ResNet-50 | 33.04 | 33.02 | – | – | – | – | – | – | – | – |
| PETR | ResNet-50-DCN | 29.77 | 26.65 | – | – | – | – | – | – | – | – |
| PETRv2 | ResNet-50-DCN | 25.98 | 23.90 | 52.52 | – | – | – | – | – | – | – |
| BEVFormerV2* | ResNet-50 | 35.34 | 30.76 | 48.49 | 23.39 | 40.69 | – | – | – | – | – |
| BEVFormerV2 | SigLIP-Qwen | 40.78 | 39.78 | – | – | – | – | – | – | – | – |
| PETR | SigLIP-Qwen | 37.61 | 34.37 | – | – | – | – | – | – | – | – |
| PETRv2 | SigLIP-Qwen | 36.10 | 33.28 | 57.62 | – | – | – | – | – | – | – |
| **BEVFormerV2*** | SigLIP-Qwen | 41.94 | 36.46 | 47.76 | **25.72** | **43.89** | – | – | – | – | – |
| Head-only (nuScenes) | SigLIP-Qwen | 35.60 | 34.13 | 55.55 | 20.21 | 36.98 | 16.14 | 16.50 | 40.45 | 11.50 | 17.43 |
| Head-only (joint) | SigLIP-Qwen | 33.49 | 33.37 | 51.15 | 14.83 | 29.54 | 40.57 | 41.86 | 66.34 | 20.13 | 25.36 |
| **Qwen-Drive-1.0-SFT** | SigLIP-Qwen | **43.95** | **42.83** | **60.99** | 19.82 | 37.02 | **43.45** | **44.16** | **71.27** | 19.84 | 25.17 |

\* = the authors' unified multi-task BEVFormerV2 variant doing all three tasks. NDS is modified with mAAE set to zero for all methods because the unified labels carry no attributes.

Four readings:

**1. The encoder swap is worth 7–10 mAP to *every* dedicated detector.** BEVFormerV2 +7.74, PETR +7.84, PETRv2 +10.12, BEVFormerV2* +6.60. Vision-language pretraining is a strong visual initialization for 3D detection even when nothing else changes — the cleanest such measurement in the wiki, because the same architectures are run twice with only the encoder swapped.

**2. The probe result.** See [below](#probe).

**3. Occupancy is where Qwen-Drive does not win**, and this page says so plainly: 19.82 Occ mIoU against BEVFormerV2*'s 25.72, and RayIoU 37.02 against 43.89. The abstract's "strong 3D perception" holds for detection and map segmentation, not occupancy. The paper's explanation is source-label quality — joint training raises OpenScene occupancy but costs nuScenes occupancy 26.6 % (20.21 → 14.83), and while Stage 2 recovers nuScenes it moves **both OpenScene occupancy metrics by less than 0.3 points**, because OpenScene's machine-generated voxel labels "retain source-specific semantic and construction artifacts."

**4. Cross-dataset transfer fails badly without mixing.** The nuScenes-only head reaches **16.50 NDS on OpenScene, less than half its own 34.13**, despite nuScenes' higher-quality annotations. Mixed training lifts it to 41.86. The comparison methods cannot be evaluated on OpenScene at all — they learn camera embeddings tied to the 6-camera rig — so **the entire OpenScene half of this table has no external baseline**.

### The 3D probe: what vision-language pretraining does and does not contain {#probe}

This is the paper's most transferable perception finding and deserves separation from the leaderboard reading.

| Configuration | Encoder / VLM state | nuScenes mAP | RayIoU |
|---|---|---:|---:|
| BEVFormerV2* (dedicated detector) | SigLIP-Qwen, trained end-to-end for detection | 41.94 | **43.89** |
| Head-only, converged | **frozen** SigLIP-Qwen + frozen VLM | 35.60 | 36.98 |
| Qwen-Drive-1.0-SFT (Stage 2) | encoder **and** VLM updated by perception loss | **43.95** | 37.02 |

> "This contrast indicates that vision-language-pretrained features support visual-text alignment but do not directly expose the 3D structure required for driving perception."

> "The gains from Stage 2 cannot be explained by continued head optimization alone, since the head-only model had already converged."

Two numbers carry it: **−6.34 mAP for freezing**, and **+10.46 mAP for unfreezing** (against the head-only joint row). This is the VLM-side analogue of [[sources/adaptive-wam.md]]'s video-prior adaptation ladder (frozen Wan 84.20 → joint LoRA 90.62) and [[sources/latent-wam.md]]'s geometric-distillation collapse under LoRA. Three different backbone families, three papers, one answer: **a foundation model used off the shelf leaves a large amount on the table, and the fix is joint adaptation against the task objective.**

The negative half of the claim lands on [[concepts/perception-for-planning.md]]: a VLM that answers spatial questions fluently is not thereby holding a 3D scene representation. The wiki previously had this as an assertion in several papers' motivation sections; it is now measured.

![[perception_vis.png|Qualitative 3D detection, semantic occupancy, and BEV map segmentation on OpenScene and nuScenes validation splits]]

**Fig. 7**: Qualitative results on OpenScene (a, b) and nuScenes (c, d), each row showing detection, occupancy and map segmentation. Notably in (b), floating voxels above the road surface survive preprocessing in the machine-generated OpenScene label while **the prediction suppresses these artifacts** and recovers the road-surface semantics — weak evidence that cross-source supervision denoises the labels.

### Table 2 — Driving VQA

Judged by an LLM or scored as multiple choice; all comparison methods re-evaluated under one near-deterministic protocol (top-$k$ 1, top-$p$ 0.001, temperature 0.01). "–" means an invalid or unparsable response and counts as **zero** in the averages. IH is an in-house Chinese urban driving-decision benchmark.

| Method | LingoQA | Ego3D Acc | Ego3D RMSE ↓ | VLAD | SURDS | Waymo Safety | Waymo All | **Avg** | CoC Key | CoC Plan | CoC All | IH | **Avg** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| InternVL3.5-8B-Inst. | 46.40 | 47.38 | 23.01 | 54.47 | 32.80 | 54.47 | 58.09 | 48.94 | – | – | – | 47.50 | 11.88 |
| LLaVA-OV2-8B | 41.20 | 42.07 | 24.97 | 58.71 | 38.60 | 49.65 | 55.23 | 47.58 | 0.86 | 12.32 | 0.57 | 54.00 | 16.94 |
| Gemma4-12B | 50.00 | 56.31 | 27.30 | 57.90 | 52.25 | 63.59 | 68.82 | 58.15 | 12.89 | 10.32 | 4.01 | 61.00 | 22.05 |
| **Qwen3.5-4B (base)** | 70.40 | **62.82** | 13.17 | 65.38 | 52.95 | 62.46 | 67.10 | **63.52** | 8.88 | 9.17 | 2.58 | 59.00 | 19.91 |
| Cosmos-Reason1-7B | 45.20 | 44.17 | 26.71 | 33.64 | 8.49 | 39.53 | 43.90 | 35.82 | 14.61 | 11.75 | 3.15 | 30.50 | 15.00 |
| Cosmos-Reason2-2B | 38.60 | 44.42 | 12.03 | 53.90 | 29.46 | 63.71 | 59.65 | 48.29 | 9.17 | 5.44 | 2.01 | 42.50 | 14.78 |
| Cosmos-Reason2-8B | 59.60 | 48.35 | 12.62 | 56.37 | 19.54 | 57.68 | 57.93 | 49.91 | 7.74 | 5.44 | 1.72 | 56.00 | 17.73 |
| Cosmos-Reason2-32B | 58.80 | 47.31 | 20.32 | 57.13 | 19.52 | 48.56 | 48.40 | 46.62 | 18.34 | 15.47 | 5.73 | 29.50 | 17.26 |
| Cosmos3-nano | 65.00 | 44.02 | 22.41 | 57.73 | 39.72 | 56.93 | 58.36 | 53.63 | 12.89 | 10.60 | 4.01 | 2.00 | 7.38 |
| MiMo-Embodied-7B | 72.00 | 60.41 | 9.85 | 50.33 | 43.06 | 66.54 | 69.56 | 60.32 | – | – | – | 61.00 | 15.25 |
| UniDriveVLA-8B | 62.00 | 44.11 | 8.45 | 53.09 | 20.06 | 49.19 | 49.28 | 46.29 | 0.86 | 12.89 | 0.86 | 48.50 | 15.78 |
| Alpamayo-1.5-10B | 64.00 | 36.79 | 25.31 | 9.13 | 3.10 | 42.61 | 44.37 | 33.33 | 9.46 | 8.31 | 3.44 | 3.00 | 6.05 |
| **Qwen-Drive-1.0-SFT** | **77.80** | 60.98 | **7.78** | **66.52** | **66.13** | **70.70** | **74.47** | **69.43** | **65.33** | **55.59** | **41.26** | **71.00** | **58.30** |

**The result the wiki should carry from this table is not Qwen-Drive's row.** It is that **the unadapted, smallest model in the table — Qwen3.5-4B at 63.52 — beats every driving- and embodiment-specialized comparison method**, including MiMo-Embodied-7B (60.32), Cosmos-Reason2-32B (46.62), [[sources/unidrivevla.md]] (46.29) and Alpamayo-1.5-10B (33.33), as well as Gemma4-12B (58.15). If that holds up, a large share of published driving post-training is *net negative* on driving question answering measured outside its own training distribution.

Qwen-Drive then adds 5.91 over its own base, distributed rather than concentrated: LingoQA +7.40, SURDS +13.18 (24.9 % relative), and Ego3D distance RMSE **13.17 → 7.78 (−40.9 %)**, 7.9 % better than UniDriveVLA-8B's 8.45.

The RMSE mechanism the paper proposes is worth recording because it is testable and unusual:

> "We attribute this reduction not to geometrically precise representations learned by the VLM, but to a richer physical understanding of driving scenes together with the model's retained quantitative reasoning. Semantic cues of stable real-world scale, such as the regular spacing of parked cars along a street or of dashed lane markings and streetlights, provide implicit references for metric distance estimation."

**Causal reasoning is where the margin is extreme**: 58.30 average against 22.05 for the second-best model, and 41.26 CoC-overall against 5.73 for the 32 B Cosmos-Reason2 — 7.2×. Against its own base: +56.45 key-object, +46.42 decision, +38.68 overall.

Two caveats this page attaches to that margin:

- **PAI-AV-CoC is the authors' own benchmark**, built from the PAI-AV validation split and judged by **Qwen3.5-Plus** — the same model family as the system under test, and the same family that produced the training traces (Qwen3.7-Plus wrote them, Qwen3.5-Flash scored their rarity). Format alignment between training and benchmark is total. The margin is too large to be explained by that alone, but "7× the 32 B model" should not be read as a capability ratio.
- The paper itself scopes Alpamayo-1.5's low score — "may reflect instruction-following errors and differences in input format, and should not be interpreted as evidence of weak causal reasoning." That caveat generalizes to the whole table: a fixed prompt with no per-model tuning, near-deterministic decoding, and "–" scored as zero makes this partly a **format-compliance** measurement.

The in-house Chinese benchmark is the best generalization evidence in the table: 71.00 against the base model's 59.00, on a benchmark requiring **Chinese-language responses** with no comparable data in the training mixture.

![[drivevqa-vis1.png|Qualitative comparison of four driving VQA capabilities with correct content in green and incorrect in red]]

**Fig. 8**: (a) temporal scope — Qwen-Drive reports no *parked* vehicles in the final frame while Qwen3.5-4B and MiMo-Embodied-7B count vehicles merely stopped in traffic; (b) causal attribution to a stop sign where Cosmos-Reason2-32B and Alpamayo-1.5 predict continued straight driving; (c) cross-view distance decomposed into street width plus longitudinal offset using parked-vehicle spacing as an implicit scale, predicting 22 m against a 22.93 m ground truth; (d) ego-lane identification from a persistent left-turn arrow. Cases (c) and (d) also expose format failures in the baselines — UniDriveVLA-8B omits the required `\boxed{}` delimiter, Alpamayo-1.5 emits a DriveLM-style object reference outside the option set.

Appendix C.1 contains the most damning qualitative artifact in the paper: MiMo-Embodied-7B's answer to "how many parked vehicles can you see" is a `<think>` trace that **repeats a self-correction for roughly four thousand characters** before settling on the wrong number.

### Table 3 — General vision-language capability {#retention}

This table is the reason to ingest this paper. Fifteen public benchmarks, thirteen models, one protocol.

**(a) Knowledge, reasoning and recognition.** The clipping flattens the HTML header; the column order is recovered as MMBench / MMStar / MMMU / MMMU-Pro-Std / MMMU-Pro-Vis / CharXiv / OCRBench / RealWorldQA / SimpleVQA / CountQA, verified two independent ways — every row average reproduces exactly (Qwen3.5-4B: 674.01/10 = 67.40), and the prose claims "best scores on MMStar and RealWorldQA" plus "first or second on 6 of the 10 settings," both of which hold only under this mapping.

| Method | MMBench | MMStar | MMMU | MMMU-Pro Std | MMMU-Pro Vis | CharXiv | OCRBench | RealWorldQA | SimpleVQA | CountQA | **Avg** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| InternVL3.5-8B-Inst. | 80.03 | 64.13 | 62.00 | 46.42 | 42.25 | 41.70 | 83.20 | 66.93 | 40.77 | 20.94 | 54.84 |
| LLaVA-OV2-8B | 82.66 | 64.93 | 54.67 | 36.30 | 25.95 | 40.10 | 79.30 | 71.76 | 36.68 | 22.58 | 51.49 |
| Gemma4-12B | 85.53 | 72.33 | 69.56 | 59.94 | 49.25 | 64.10 | 77.80 | 68.89 | 40.82 | **39.46** | 62.77 |
| **Qwen3.5-4B (base)** | 87.07 | 75.33 | **73.44** | **64.86** | **61.27** | **65.10** | 86.90 | 76.34 | 47.84 | 35.86 | **67.40** |
| Cosmos-Reason1-7B | 79.95 | 63.53 | 54.22 | 38.38 | 35.78 | 39.70 | 85.20 | 67.45 | 44.98 | 18.52 | 52.77 |
| Cosmos-Reason2-2B | 75.00 | 53.13 | 51.56 | 35.09 | 30.35 | 28.50 | 79.60 | 60.52 | 36.94 | 18.00 | 46.87 |
| Cosmos-Reason2-8B | 82.82 | 65.27 | 59.11 | 36.07 | 43.53 | 42.50 | 87.00 | 67.45 | 45.25 | 22.32 | 55.13 |
| Cosmos-Reason2-32B | **88.70** | 72.47 | 61.67 | 41.45 | 52.77 | 53.20 | **88.20** | 75.69 | **48.44** | 26.70 | 60.93 |
| Cosmos3-nano | 79.57 | 66.67 | 60.89 | 46.36 | 40.75 | 42.10 | 85.20 | 69.67 | 44.99 | 23.63 | 55.98 |
| **MiMo-Embodied-7B** | **–** | 22.40 | **–** | 27.40 | 28.09 | 57.50 | 78.80 | 28.50 | **–** | 22.64 | **26.53** |
| UniDriveVLA-8B | 74.30 | 64.07 | 50.67 | 32.43 | 31.56 | 33.80 | 80.20 | 68.10 | 35.26 | 16.88 | 48.73 |
| **Alpamayo-1.5-10B** | **7.51** | 26.13 | 27.44 | 15.61 | 13.47 | 1.50 | **3.20** | 46.93 | **–** | 4.71 | **14.65** |
| **Qwen-Drive-1.0-SFT** | 85.53 | **75.87** | 72.67 | 62.72 | 59.71 | 64.40 | 86.40 | **78.95** | 46.12 | 31.74 | **66.41** |

**(b) Spatial understanding and grounding.**

| Method | EmbSpatial | ERQA | RefSpatial | Omni3D | ODinW13 | **Avg** |
|---|---:|---:|---:|---:|---:|---:|
| InternVL3.5-8B-Inst. | 74.20 | 42.00 | – | – | – | 23.24 |
| LLaVA-OV2-8B | 78.43 | 42.25 | – | – | – | 24.14 |
| Gemma4-12B | 73.16 | 42.00 | – | – | – | 23.03 |
| **Qwen3.5-4B (base)** | 75.99 | 46.25 | 54.51 | **47.40** | 40.78 | 52.99 |
| Cosmos-Reason1-7B | 68.76 | 38.50 | 0.36 | – | 4.77 | 22.48 |
| Cosmos-Reason2-2B | 66.40 | 38.75 | 32.49 | 31.41 | 33.04 | 40.42 |
| Cosmos-Reason2-8B | 77.61 | 43.25 | 51.81 | 32.85 | 40.19 | 49.14 |
| Cosmos-Reason2-32B | **79.26** | 45.25 | **57.76** | 31.70 | 28.47 | 48.49 |
| Cosmos3-nano | 77.88 | 41.25 | – | 32.26 | 35.87 | 37.45 |
| MiMo-Embodied-7B | 45.05 | 39.75 | 2.17 | – | – | 17.39 |
| UniDriveVLA-8B | 68.16 | 38.00 | 1.44 | 0.33 | – | 21.59 |
| Alpamayo-1.5-10B | 20.58 | 27.50 | – | – | – | 9.62 |
| **Qwen-Drive-1.0-SFT** | 78.85 | **48.50** | 50.78 | 45.79 | **45.87** | **53.96** |

**What it says.** Group (a): −0.99 against the base model, first or second on 6 of 10 settings. Group (b): **+0.97 above the base model**, best on ERQA and ODinW13. Across all 15 settings Qwen-Drive averages **62.26 against Cosmos-Reason2-32B's 56.78 (+5.48)** while matching or beating it on 10 of 15 — against a model eight times its size that was itself post-trained for physical AI.

**And what the other rows say.** Every specialized model in the table has lost most of its general capability:

| Model | Specialization | Group (a) avg | vs. the best general model |
|---|---|---:|---|
| Qwen3.5-4B | none | 67.40 | — |
| **Qwen-Drive-1.0-SFT** | **driving, three tasks** | **66.41** | **−0.99** |
| Cosmos-Reason2-32B | physical AI | 60.93 | −6.5 at 8× the size |
| UniDriveVLA-8B | driving | 48.73 | −18.7 |
| MiMo-Embodied-7B | embodied + driving | 26.53 | −40.9, three unparsable |
| Alpamayo-1.5-10B | driving CoC | 14.65 | −52.8, MMBench 7.51 |

**The honest caveat**: Alpamayo-1.5's 7.51 MMBench and 3.20 OCRBench are not plausible *capability* scores for a 10 B model — they are parse failures under a protocol with no per-model prompt adaptation, and "–" counts as zero. So this table conflates **retained knowledge** with **retained general instruction-following interface**. The paper's framing accepts that conflation on purpose: for the cockpit-integration argument, a model that knows the answer but cannot emit it in the requested format is equally unusable. Read it as a measurement of *deployable* general capability, not of knowledge in the weights.

### Table 4 — WOD-E2E open-loop (Rater Feedback Score)

**(a) Validation split** — the split whose rater annotations supply the RL reward.

| Method | RL | ADE 3 s ↓ | ADE 5 s ↓ | RFS ↑ |
|---|:-:|---:|---:|---:|
| Human Driver | – | – | – | 8.13 |
| VAD | – | 3.19 | 5.81 | 4.45 |
| UniAD | – | 6.50 | 10.81 | 5.78 |
| RAP-DINO | – | 0.97 | 2.20 | 7.91 |
| MindVLA-U1 | – | 0.89 | 2.11 | 7.92 |
| MindVLA-U1 | ✓ | 1.01 | 2.28 | 8.20 |
| Qwen-Drive-1.0-SFT w/o reasoning | – | 0.99 | 2.33 | 7.95 |
| Qwen-Drive-1.0-SFT w/ reasoning | – | 0.99 | 2.31 | 7.95 |
| **Qwen-Drive-1.0-RL** | ✓ | **0.62** | **1.27** | **8.45** |

**(b) Test split**

| Method | RL | ADE 3 s ↓ | ADE 5 s ↓ | RFS ↑ |
|---|:-:|---:|---:|---:|
| Swin-Trajectory | – | 1.21 | 2.81 | 7.54 |
| DiffusionLTF | – | 1.36 | 2.89 | 7.72 |
| UniPlan | – | 1.31 | 2.99 | 7.78 |
| LightEMMA | – | 1.71 | 3.74 | 6.52 |
| NaiveEMMA | – | 1.32 | 3.02 | 7.53 |
| dVLM-AD | – | 1.29 | 3.02 | 7.63 |
| HMVLM | – | 1.33 | 3.07 | 7.74 |
| MindVLA-U1 | – | 1.16 | 2.67 | 7.77 |
| AutoVLA | ✓ | 1.35 | 2.96 | 7.56 |
| NoRD | ✓ | 1.25 | – | 7.71 |
| MindVLA-U1 | ✓ | **1.09** | 2.66 | 7.87 |
| Qwen-Drive-1.0-SFT w/o reasoning | – | 1.20 | 2.66 | 7.76 |
| Qwen-Drive-1.0-SFT w/ reasoning | – | 1.19 | **2.65** | 7.78 |
| **Qwen-Drive-1.0-RL** | ✓ | 1.19 | 2.67 | **7.91** |

The paper's own reading is unusually disciplined and should be quoted:

> "Since these annotations supervise the reward, this in-sample result indicates effective optimization of preference alignment on the training scenarios rather than generalization beyond human driving."

So 8.45 > the human reference of 8.13 is explicitly *not* claimed as superhuman driving. The transferable number is **+0.13 on the test split (7.78 → 7.91)**, "without materially changing displacement from the recorded future" — RL moves preference alignment, not geometry. Reasoning is worth **+0.02** here, from 142 K of 557 K WOD-E2E samples.

For wiki context: [[sources/autovla.md]] 7.56 and [[sources/nord.md]] 7.71 appear here at values consistent with their own pages; [[sources/hermes.md]] (6.81) and [[sources/dial.md]] (8.211 held-out peak, 9.14 oracle BoN-128 ceiling) are absent. DIAL's protocol used 338 of **438** labelled validation sequences for RL; Qwen-Drive uses **479** — two papers give different counts for the annotated WOD-E2E validation set, which [[concepts/nuscenes-waymo-evals.md]] now tracks.

### Table 5 — PAI-AV open-loop, and a reversal {#paiav}

Six trajectories per scene; all comparison methods reproduced by the authors under one setting. The 644-example standard split "overlaps with publicly available training data by official construction," so a **leakage-free 700-frame subset curated from held-out test clips** is reported alongside — the first time any wiki source has published both.

| Method | 644 Avg ADE 3 s | 5 s | 644 minADE 3 s | 5 s | 700 Avg ADE 3 s | 5 s | 700 minADE 3 s | 5 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Alpamayo-R1-10B | 0.37 | 1.13 | **0.16** | **0.48** | 0.41 | 1.22 | **0.18** | **0.51** |
| Alpamayo-1.5-10B | **0.35** | **1.05** | **0.16** | 0.50 | **0.36** | **1.06** | 0.17 | 0.49 |
| DriveWAM | 0.67 | – | 0.37 | – | 0.69 | – | 0.38 | – |
| SimWAM | 0.41 | – | 0.38 | – | 0.43 | – | 0.40 | – |
| Qwen-Drive-1.0-SFT w/o reasoning | 0.38 | 1.07 | 0.34 | 0.96 | 0.43 | 1.24 | 0.39 | 1.11 |
| Qwen-Drive-1.0-SFT w/ reasoning | 0.37 | 1.07 | 0.34 | 0.97 | 0.42 | 1.23 | 0.39 | 1.11 |
| Qwen-Drive-1.0-RL | 0.42 | 1.11 | 0.38 | 1.00 | 0.47 | 1.27 | 0.43 | 1.15 |

**The reversal.** [[concepts/physicalai-av-benchmark.md]] currently carries [[sources/drivewam.md]]'s own table, on DriveWAM's own curated 1,000-clip subset, where **DriveWAM is 0.47 ADE@3s and Alpamayo-1.5 is 0.80**. Reproduced here on the standard split, **DriveWAM is 0.67 and Alpamayo-1.5 is 0.35** — the ordering inverts and the gap roughly doubles in the other direction. The splits differ, so this is not a direct contradiction of a single number; it is exactly the failure that page's "no standard public test protocol" caveat predicted, now realized.

**On candidate diversity**, the paper reads its own numbers against itself: avg-ADE 0.42 vs. minADE 0.39 on the leakage-free subset is a narrow gap, "suggesting that our candidates remain concentrated around similar motions," where Alpamayo-1.5 spans 0.36 → 0.17. Part of the gap is attributed to scale — **80,000 hours and 3 M CoC traces for Alpamayo-1.5 against ~900 raw hours of PAI-AV before sparse frame sampling**. DriveWAM shows broad coverage with weak typical accuracy (0.69 / 0.38); SimWAM shows the opposite (0.43 / 0.40).

RL costs 3–5 cm here, a trade the paper accepts explicitly against gains on the other three evaluations.

### Table 6 — NAVSIM v1.1 navtest (pseudo-closed-loop)

‡ = best-of-6 selection by true PDMS (oracle).

| Method | RL | NC | DAC | EP | TTC | Comf. | **PDMS** |
|---|:-:|---:|---:|---:|---:|---:|---:|
| TransFuser | – | 97.7 | 92.8 | 79.2 | 92.8 | 100.0 | 84.0 |
| DRAMA | – | 98.0 | 93.1 | 80.1 | 94.8 | 100.0 | 85.5 |
| Hydra-MDP | – | 98.3 | 96.0 | 78.7 | 94.6 | 100.0 | 86.5 |
| DiffusionDrive | – | 98.2 | 96.2 | 82.2 | 94.7 | 100.0 | 88.1 |
| Epona | – | 97.9 | 95.1 | 80.4 | 93.8 | 99.9 | 86.2 |
| ReCogDrive | – | 98.3 | 95.1 | 81.1 | 94.3 | 100.0 | 86.8 |
| AutoVLA | – | 96.9 | 92.4 | 75.8 | 88.1 | 99.9 | **80.5** |
| SpanVLA | – | 97.5 | 90.8 | 76.9 | 93.7 | 99.5 | **82.1** |
| Qwen-Drive-1.0-SFT w/o reasoning | – | 98.2 | 96.4 | 82.0 | 94.4 | 100.0 | 87.8 |
| Qwen-Drive-1.0-SFT w/ reasoning | – | 98.4 | 96.6 | 82.4 | 94.7 | 100.0 | **88.2** |
| Qwen-Drive-1.0-SFT‡ w/o reasoning | – | 98.6 | 97.1 | 82.9 | 95.1 | 100.0 | 88.9 |
| Qwen-Drive-1.0-SFT‡ w/ reasoning | – | 98.7 | 97.2 | 83.2 | 95.5 | 100.0 | 89.3 |
| ReCogDrive | ✓ | 98.2 | 97.8 | 83.5 | 95.2 | 99.8 | 89.6 |
| AutoVLA | ✓ | 98.4 | 95.6 | 81.9 | **98.0** | 99.9 | 89.1 |
| SpanVLA | ✓ | **99.1** | 97.1 | **86.3** | 95.2 | 100.0 | 90.3 |
| ExploreVLA | ✓ | 98.8 | 98.4 | 83.5 | 96.5 | 99.9 | 90.4 |
| EponaV2 | ✓ | 98.6 | 97.9 | 84.8 | 95.7 | 100.0 | 90.4 |
| **Qwen-Drive-1.0-RL** | ✓ | 98.6 | **98.2** | 84.8 | 95.9 | 100.0 | **90.7** |
| **Qwen-Drive-1.0-RL‡** | ✓ | 98.8 | **98.4** | 85.5 | 96.5 | 100.0 | **91.4** |

**This is a well-disciplined table by wiki standards**: RL and non-RL are separated into blocks, oracle selection is marked, and pre-RL numbers are given for methods this wiki has only ever carried post-RL — **AutoVLA at 80.5 and SpanVLA at 82.1 before RL**, against their published 89.1 and 90.3. Those two rows are new information for [[concepts/rl-for-ad.md]]: they imply RL is worth **+8.6 and +8.2** for those methods, far above the +2.5 Qwen-Drive gets, consistent with the wiki's existing observation that GRPO gains are largest where the SFT policy is weakest.

**Comparison-scope caveat, as always**: the table contains nothing above 90.4. Absent are DriveSuprim 93.5, CLEAR 93.7, DA-WAM 93.7, Drive-JEPA 93.3, WCog-VLA 92.9, LWDrive 92.0 and SimWAM 91.5 — the last of which this same paper evaluates in closed loop two tables later. 90.7 is a strong result, not a frontier one.

The paper's own interpretive note is one [[concepts/navsim-benchmark.md]] should adopt verbatim:

> "PDMS should not be treated as a direct proxy for interactive driving quality. Near the upper end of this benchmark, further gains may increasingly reflect adaptation to the scoring function, while the non-reactive protocol cannot reveal how errors accumulate during interaction."

Reasoning is worth **+0.4 PDMS** from 78 K NAVSIM samples; RL is worth **+2.5**; oracle best-of-6 is worth a further **+0.7** — and the BoN gap is read correctly, as headroom for a better inference-time selector rather than as a score.

### Table 7 — AlpaSim closed-loop {#alpasim}

916 scenarios on PAI-AV-NuRec v26.02, with novel-view synthesis as the ego deviates from the recorded log. Params excludes LLM token embeddings. All comparison methods reproduced by the authors.

| Method | Params | CE all ↓ | CE at-fault ↓ | Off-road ↓ | Progress ↑ | Score all ↑ | Score at-fault ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Alpamayo-R1** | 9.8 B | **19.0** | 6.0 | 17.0 | **67.0** | **0.36** | **0.58** |
| Alpamayo-1.5 | 9.8 B | 37.0 | 11.0 | 16.0 | 59.0 | 0.23 | 0.45 |
| DriveWAM | 15.0 B | 56.0 | **5.0** | **8.0** | 35.0 | 0.10 | 0.53 |
| SimWAM | 6.0 B | 35.0 | 22.0 | 19.0 | 62.0 | 0.22 | 0.30 |
| Qwen-Drive-1.0-SFT w/ reasoning | **5.0 B** | 38.0 | 12.0 | 24.0 | 54.0 | 0.16 | 0.27 |
| **Qwen-Drive-1.0-RL** | **5.0 B** | 41.0 | 11.0 | 12.0 | 48.0 | 0.16 | 0.37 |

**This is the most consequential table in the paper for this wiki, and Qwen-Drive loses it.**

Set the NAVSIM ordering beside the closed-loop ordering:

| | NAVSIM-v1 PDMS | AlpaSim at-fault score |
|---|---:|---:|
| [[sources/simwam.md]] | **91.5** (wiki's best WAM) | **0.30** (last) |
| Qwen-Drive-1.0-RL | 90.7 | 0.37 |
| [[sources/drivewam.md]] | 90.1 | 0.53 |
| Alpamayo-1.5 | not reported | 0.45 |
| [[sources/alpamayo-r1.md]] | not reported | **0.58** (first) |

The two rankings are close to inverted. The wiki's open question "do PDMS gains transfer to closed-loop?" now has its first multi-method answer under one reproduction, and it is **no**.

Three readings the paper supplies and this page endorses:

**1. An AlpaSim score cannot be read alone.** DriveWAM's 0.53 at-fault score comes with **35 % progress**: "it often remains stationary or advances only briefly," which suppresses ego-at-fault and off-road events while the metric divides distance travelled by event count. Its all-event close-encounter rate is **56 %**, the worst in the table, because a stationary vehicle is still struck from behind. SimWAM is the mirror image: "considerably more aggressive, frequently accelerating forward while failing to decelerate sufficiently," giving 62 % progress with 22 % at-fault encounters and 19 % off-road.

**2. RL bought safety and paid progress — the opposite direction from NAVSIM.** Off-road **24 → 12 %** (halved, and below both Alpamayo variants), at-fault score 0.27 → 0.37, at-fault CE 12 → 11 %, against progress **54 → 48 %** and all-event CE 38 → 41 %. On NAVSIM the same RL stage *raised* ego progress 82.4 → 84.8 and the paper says it is "particularly effective at reducing conservative, low-progress behavior." **The same policy update is progress-positive on the non-reactive benchmark and progress-negative in closed loop.** The paper does not remark on this; this page records it as the sharpest available instance of the two protocols disagreeing about one change.

**3. The paper's hypothesis for its own closed-loop deficit is observation cadence**, not capability: Alpamayo-1.5 sees a dense 0.4 s visual history while Qwen-Drive sees 4 frames at 0.5 s over 1.5 s — "this broader but sparser history may limit responsiveness over the short replanning horizon." Untested, and testable in this codebase.

![[planning_vis.png|Open-loop predictions on WOD-E2E and PAI-AV, and two closed-loop AlpaSim rollouts at selected timestamps]]

**Fig. 9**: (a) open-loop predictions from Qwen-Drive-1.0-SFT with reasoning on the WOD-E2E test split and PAI-AV — decelerating for a crossing animal, following a right-turn-only lane, adjusting laterally past a stopped vehicle. (b) two AlpaSim rollouts from Qwen-Drive-1.0-RL — following a lead vehicle through a green light then stopping on red; following a slower vehicle, turning right, and adjusting laterally as another vehicle overtakes.

![[navsim-rl.png|Qualitative effect of reinforcement learning on the same NAVSIM left-turn scene, before and after]]

**Fig. 10**: the same left-turn scene before (a) and after (b) reinforcement learning; prediction in red, recorded future in green. Both keep the same high-level maneuver and RL removes a small lateral deviation. **The qualitative signature of Stage 4 is fine-grained alignment with the scoring criteria, not a change of decision** — consistent with the +0.13 test-split RFS "without materially changing displacement."

---

## Ablations

### Table 8 — the Stage 2 mixture, and the +0.08 {#stage2-ablation}

Each row trains a Planning Expert for 15 epochs on WOD-E2E only, then reports validation RFS. The vision-language columns are the 6-metric driving aggregate and the 15-setting general aggregate defined above.

| ID | Stage 2 VL data | Stage 2 3D perception | Driving QA Avg ↑ | CoC Overall ↑ | General VQA Avg ↑ | WOD-E2E RFS ↑ |
|---|:-:|:-:|---:|---:|---:|---:|
| i | ✗ | ✗ | 63.52 | 2.58 | 62.60 | 7.88 |
| ii | ✓ | ✗ | **70.07** | 40.97 | **63.18** | 7.91 |
| iii | ✓ | ✓ | 69.43 | **41.26** | 62.26 | **7.96** |

Row i is the **unadapted Qwen3.5-4B**. Read down the last column:

- **The entire driving adaptation — 24 public datasets, 3.09 M filtered VQA pairs, 1.54 M curated Stage-2 samples, a full 3D perception stack, and an updated vision encoder and VLM — is worth +0.08 RFS to the planner.**
- Driving VQA supervision alone: **+0.03**. Adding 3D perception supervision: **+0.05**.
- For comparison, in the same paper: Stage-4 RL is worth **+0.50** on the same split, and **+0.13** on the test split.

The paper states the limit of the claim itself — "the planning result supports the compatibility of the Stage 2 mixture with subsequent Planning Expert training, but does not establish explicit 3D supervision as the source of the improvement" — but does not remark on the size.

**Why this matters beyond this paper.** Stage 3 freezes the VLM, so everything Stage 2 installed must reach the planner through cached keys and values. The wiki has been accumulating evidence that *representation* and *planner* are substitutes rather than complements ([[sources/coworld-vla.md]]'s 2×2, interaction −3.9) and that named mechanisms are consistently smaller than their framing ([[sources/da-wam.md]] +0.15, [[sources/foresight.md]] +0.3, [[sources/drive-hwm.md]] +0.3 to +0.8). **This is the largest investment-to-effect gap yet recorded** — and unlike those cases, the investment is not the paper's headline mechanism, so there is no incentive to overstate it. The retention result and the perception result stand on their own; the *unification* thesis — that one shared representation makes all three tasks better — is the part this table does not support.

Note also what the middle columns show: **3D perception supervision costs 0.64 driving-QA points and 0.92 general-VQA points** while adding +0.29 CoC and +0.05 RFS. Explicit 3D capability is not free, and the price is paid in language.

### Fig. 11 — RL mixture and reward design (image missing, values from prose)

| Setting | NAVSIM PDMS | WOD-E2E val RFS | WOD-E2E val 5 s ADE |
|---|---:|---:|---:|
| NAVSIM only, PDMS reward | 90.4 | – | – |
| NAVSIM only, PDMS + shared ADE | **90.8** | – | – |
| Joint 3-source, source-specific rewards | 90.6 | **8.68** | 2.24 |
| **Joint 3-source + shared ADE (final)** | 90.7 | 8.45 | **1.27** |

Two findings:

- **Cross-dataset interference is small when each source carries a task-aligned reward**: 90.8 single-source against 90.7 joint. That is a positive result for multi-source RL, and the wiki has no other data point on it.
- **The shared displacement term is a deliberate trade**: −0.23 RFS for −0.97 m of 5 s ADE. Without it, preference optimization drifts from the recorded motion. The final recipe takes the anchor.

Worth noting: the un-anchored 8.68 is the highest RFS anywhere in the paper, well above the human reference — the clearest available illustration that **RFS optimized in-sample measures reward-model agreement, not driving**.

### Fig. 12 — planning data scale (image missing, endpoints from prose)

Planning Expert trained on PAI-AV only, at 0.17 M → 0.35 M → 0.69 M → 1.04 M → 1.38 M samples. On the standard 644 split, 5 s Avg ADE **1.34 → 1.05** and Avg FDE **4.18 → 3.24**, monotonic, with the leakage-free subset following the same trend and **no sign of saturation at 1.38 M**.

This is the third unsaturated real-log data-scaling curve in the wiki, after [[sources/drivewam.md]]'s 4 k → 100 k clips and [[sources/drivevla-w0.md]]'s scaling study. None has found the knee.

### Fig. 13 — qualitative transfer to unseen camera rigs

![[ood_perception_vis.png|Perception outputs on unseen camera rigs: WOD-E2E eight-camera ring and PAI-AV six-camera, predictions only]]

**Fig. 13**: WOD-E2E's 8-camera ring rig (a, b) and PAI-AV's 6-camera rig (c, d), neither present in any perception training data. After correcting lens distortion to a pinhole model, the model runs directly with no dataset-specific adaptation and produces plausible projected boxes, coherent road surfaces, drivable areas, lane markings and road edges.

The paper scopes this correctly — "because neither dataset provides unified ground truth, they do not establish reliable 3D accuracy under the new camera settings" — but the *capability* is real and follows directly from having no rig-specific camera embeddings. It is the strongest argument in the paper for a calibration-driven view transform over learned per-rig positional encodings.

---

## Limitations

**The paper's own** (§Limitations and Future Work), all three worth carrying:

1. **Multi-timescale causal structure is unresolved.** "A red light 20 m ahead calls for early, gradual deceleration, whereas a child emerging 5 m ahead demands an immediate response. When such causes coexist, the model remains unstable in identifying the governing cause and its temporal scope. Even when the suggested trend is appropriate, the decision executed within the next 1 to 2 s may not reflect the stated immediate cause."
2. **The trajectory does not always follow the rationale** — and the paper draws the uncomfortable inference itself: "part of this gain may stem from the additional model-internal information that the self-generated trace contributes to the conditioning context." In other words the reasoning trace may be functioning as extra conditioning capacity rather than as a rationale. Given that reasoning is worth +0.4 PDMS and +0.02 RFS, that is entirely live, and it belongs on [[concepts/chain-of-thought-for-ad.md]] beside [[sources/nord.md]]'s reasoning-free result.
3. **Cross-task transfer is limited by format divergence** — the three tasks use different input serializations, temporal contexts and image resolutions.

**This page adds:**

4. **No latency, no throughput, no memory, no parameter breakdown beyond "5.0 B", no GPU-hours.** For a paper whose central deployment argument is *sharing one SoC between cockpit and driving*, the absence of any compute measurement is the largest gap in it. The inference path is a 4 B VLM over three 720p cameras plus nine 320p history frames, then 10 Euler steps of a 1.1 B DiT — and Stage-4-style operation additionally samples eight reasoning traces.
5. **The perception comparison is entirely self-reproduced, on self-defined labels.** Seven-class merged taxonomies, the authors' own reproductions of BEVFormerV2/PETR/PETRv2 under a 24-epoch schedule, a modified NDS with mAAE zeroed, and no contact with any published nuScenes leaderboard number. Internally consistent and clearly documented, but not externally anchored — and the OpenScene half has no external baseline at all.
6. **Occupancy is a loss, not a win** (19.82 / 37.02 against 25.72 / 43.89), and joint adaptation cannot move OpenScene occupancy at all (<0.3 points).
7. **Same-family judging.** Qwen3.5-Plus judges, Qwen3.5-Flash filters, Qwen3.7-Plus writes the CoC training traces, and the system under test is a Qwen3.5-4B derivative. This touches LingoQA, PAI-AV-CoC and the data pipeline simultaneously.
8. **The general-capability table is partly a format-compliance table** (see [Retention](#retention)); several comparison rows are parse failures scored as zero.
9. **`accel_x` / `accel_y` in WOD-E2E metadata are corrected by a factor of four**, unexplained and uncited. Anyone reusing that dataset needs to know whether this is a dataset bug or a unit convention the authors inferred.
10. **No NAVSIM-v2 / EPDMS, no navhard, no Bench2Drive, no HUGSIM, no nuScenes planning**, and the NAVSIM table omits everything above 90.4.
11. **The unification thesis is not demonstrated** — see [Stage 2 ablation](#stage2-ablation). Each of the three capabilities works; the claim that sharing a representation makes them better is measured once, at +0.08 RFS, with the paper declining to attribute even that.
12. **Everything reported is the SFT model except planning.** Perception and VQA are evaluated on Qwen-Drive-1.0-SFT only; no table shows whether Stage 4 damaged them. Since RL touches only the Planning Expert this should be exactly neutral, and one row would confirm it.

---

## Key Cross-References

- [[concepts/general-capability-retention.md]] — **new page**, seeded almost entirely by this paper's Table 3
- [[concepts/alpasim-benchmark.md]] — **new page**, seeded by Table 7
- [[concepts/vlm-domain-adaptation.md]] — the retention recipe (26–31 % general data, unchanged architecture, unfrozen backbone) against [[sources/automot.md]]'s frozen-VLM position and [[sources/unidrivevla.md]]'s MoT position
- [[concepts/foundation-backbones-for-ad.md]] — the adaptation ladder gains its VLM-side data point; Qwen3.5-4B enters as a backbone; SigLIP-Qwen vs. ResNet-50 is the wiki's cleanest encoder-swap measurement
- [[concepts/perception-for-planning.md]] — the 3D probe, and a perception head whose stated purpose is *diagnosis* rather than planning
- [[concepts/navsim-benchmark.md]] — 90.7 / 91.4‡; pre-RL AutoVLA 80.5 and SpanVLA 82.1; the paper's own scoring-function caveat
- [[concepts/rl-for-ad.md]] — subspace-constrained exploration on a deterministic flow; multi-source reward normalization
- [[concepts/diffusion-planner.md]] — cached-KV conditioning and $x$-prediction flow matching, beside [[sources/spanvla.md]]'s sparse-KV bridge
- [[concepts/chain-of-thought-for-ad.md]] — CoC traces as an *optional condition* on a separate action expert, priced at +0.4 PDMS / +0.02 RFS
- [[concepts/physicalai-av-benchmark.md]] — the leakage-free subset, and the DriveWAM/Alpamayo ordering reversal
- [[concepts/nuscenes-waymo-evals.md]] — WOD-E2E RFS entries and the 438-vs-479 annotated-sequence discrepancy
- [[concepts/best-of-n.md]] — 90.7 → 91.4 at N=6 on top of RL
- [[concepts/world-model-for-ad.md]] — the closed-loop reproduction of [[sources/simwam.md]] and [[sources/drivewam.md]]
