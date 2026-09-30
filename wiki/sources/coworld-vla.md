---
title: "CoWorld-VLA: Thinking in a Multi-Expert World Model for Autonomous Driving"
type: source-summary
sources: [raw/papers/CoWorld-VLA_ Thinking in a Multi-Expert World Model for Autonomous Driving.md]
related: [concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/foundation-backbones-for-ad.md, concepts/chain-of-thought-for-ad.md, concepts/diffusion-planner.md, concepts/selection-based-planning.md, concepts/adaptive-routing.md, concepts/perception-for-planning.md, sources/wa-jepa.md, sources/drivefuture.md, sources/reworld.md, sources/drivelaw.md, sources/latent-wam.md, sources/onevl.md, sources/dynvla.md, sources/geoworldad.md, sources/geowam.md, sources/unified-driving-tokens.md, sources/drive-jepa.md, sources/auto-jepa.md, sources/da-wam.md, sources/simwam.md, sources/brainwam.md, sources/adaptive-wam.md, sources/drivewam.md, sources/driveva.md, sources/recogdrive.md, sources/sgdrive.md, sources/drivevla-w0.md, sources/dreameraD.md, sources/policy-world-model.md, sources/diffusiondrive.md, sources/diffusiondrive-v2.md, sources/drivesuprim.md, sources/wcog-vla.md, sources/epona.md, sources/wam-flow.md, sources/clear.md, sources/drivedreamer-policy.md, sources/physwam.md]
created: 2026-09-11
updated: 2026-09-30
confidence: high
---

**Paper**: CoWorld-VLA: Thinking in a Multi-Expert World Model for Autonomous Driving
**Authors**: Minqing Huang, Yujiao Xiang, Zihan Liang, Jiajie Huang, Jingqi Wang (corresponding), Yuheng Zhou, Zhi Xu, Feiyang Tan, Hangning Zhou, Mu Yang, Gong Chen — "contributed equally and listed in no particular order"
**Orgs**: Southeast University · Tianjin University · Afari Intelligent Drive · UESTC · Shanghai Jiao Tong University · BUPT
**arXiv**: 2605.10426v3
**Code**: https://github.com/AFARI-Research/CoWorld-VLA (announced, "will be available")

---

## Source Integrity Note

The clipping is complete through Appendix A.2.4 but **Figures 5 and 7 are multi-panel and only panel (a) of each survives** — the sub-captions for panels (b) and (c) are described in prose but the images are not in the clipping. One extracted asset, `case1_baseline.png`, is never referenced by the markdown. All seven tables and Figures 1–4 and 6 are intact.

---

## Summary

CoWorld-VLA's premise is that "the world model" has been treated as a single object when planning needs several different things from the future at once. Its answer is to run **four future-prediction objectives in parallel, each against a different frozen teacher, all landing in one VLM's latent space as four typed expert tokens**:

| Expert token | Teacher / target | Objective | What it is supposed to carry |
|---|---|---|---|
| $H_{\mathrm{sem}}$ semantic interaction | Frozen **V-JEPA** features of the *future* frame, pooled | SmoothL1 + cosine | Interaction intent, object-level context |
| $H_{\mathrm{geo}}$ geometric structure | Frozen **VGGT** features of the *future* frame, pooled | MSE | Road layout, spatial constraints, 3D structure |
| $H_{\mathrm{dyn}}$ dynamic evolution | **Wan2.2-5B** video DiT, with $H_{\mathrm{dyn}}$ *replacing* the text condition | Flow matching on future latents | Future motion trend, temporal consistency |
| $H_{\mathrm{traj}}$ ego trajectory | GT waypoints via an MLP head | MSE | Behavioural goal |

Then a **Hierarchical Multi-Expert Fusion (HMEF)** planner denoises one trajectory per expert and averages them with learned global weights. Backbone is Qwen3-VL-**2B**, input is a **single front-camera frame**, and there is **no RL and no trajectory scorer**.

Headlines: **90.0 PDMS on NAVSIM-v1 navtest, 90.0 corrected EPDMS (86.2 pre-fix) on NAVSIM-v2 navtest, FVD 32.7 on NAVSIM.** The v2 number places it **fourth in the corrected cohort** and, critically, it is now verified at the primary source — CoWorld-VLA was one of the four un-ingested corrected-protocol methods [[concepts/navsim-benchmark.md]] has been flagging since the WA-JEPA ingest.

**Three readings this page arrives at:**

1. **The wiki's first complete representation × planner factorial, and the interaction is strongly negative.** Table 5 gives all four cells. Multi-expert representation is worth **+5.0** with a plain VLM readout and **+1.1** on top of HMEF; HMEF is worth **+5.2** on a plain trajectory token and **+1.3** on top of the full expert set. Expected additive total +10.2, actual **+6.3**. The two interventions are largely substitutes. See [The 2×2](#factorial).
2. **This is WA-JEPA's lab, and the two papers contradict each other on evaluator provenance.** Ten of CoWorld-VLA's eleven authors are on [[sources/wa-jepa.md]], both use the same `AFARI-Research` GitHub org, and each paper's corresponding author is an author of the other. They disagree about which evaluator produced DriveVLA-W0's 86.1 — WA-JEPA files it pre-fix, CoWorld-VLA prints it corrected. See [One Lab, Two Answers](#one-lab).
3. **The objective forms are assigned by target entropy, correctly, and the paper never says so.** Flow matching for the high-entropy future video; deterministic regression for the *pooled* (low-entropy) semantic and geometric targets and the single trajectory. That is exactly the rule [[sources/wa-jepa.md]] and [[sources/drivefuture.md]] arrived at from opposite directions, applied without being stated.

**Compute is disclosed in full and it is large**: 64×A800 for 74 h (Stage 1) + 32×A800 for 70 h (Stage 2) + 16×A800 for 20 h (Stage 3) = **≈7,300 A800-GPU-hours**, which the paper names as its primary limitation.

---

## Positioning

![[introduction.png|Four reasoning paradigms for VLA driving: direct action prediction, textual CoT, single-world latent reasoning, and CoWorld-VLA's multi-expert latent CoT with a fusion diffusion planner]]

**Figure 1**: (a) Direct action prediction maps multimodal inputs to actions without intermediate reasoning. (b) Textual CoT introduces language-based reasoning but may lose continuous spatio-temporal detail. (c) Single-world latent reasoning relies on one implicit world representation, "which may be incomplete or weakly coupled with actions." (d) CoWorld-VLA organizes Latent CoT experts and uses a fusion diffusion planner.

The framing targets two things at once, and the second is the interesting one:

> "planning depends on multiple forms of world knowledge, including semantic interactions, 3D structure, and dynamic evolution, which are difficult to capture with a single representation. Moreover, predicted world representations are usually used only as auxiliary supervision rather than explicit conditions for trajectory generation during inference."

The first half is a claim about **sufficiency** of a single world-model target — which is precisely the axis [[concepts/world-model-for-ad.md]] has been organizing patterns along (pixels, video latents, semantic features, occupancy, symbolic state, metric geometry, ego-trajectory latents, multi-agent trajectories). Every one of those papers picks *one*. CoWorld-VLA is the first here to run several in parallel and ablate their complementarity.

The second half is the same complaint [[sources/drivefuture.md]] made one ingest earlier, in almost the same words — the future latent should be a *condition* for action generation, not only an auxiliary loss. The two papers answer it differently: DriveFuture routes a single predicted BEV-derived latent into every denoising step of one planner; CoWorld-VLA routes four typed VLM hidden states into four parallel denoising branches and averages the results.

---

## Method

![[overview_new.png|CoWorld-VLA three-stage pipeline: Wan video-generator pretraining, multi-expert VLM representation learning against V-JEPA/VGGT/Wan/trajectory teachers, and HMEF diffusion planning]]

**Figure 2**: Three-stage training. Stage 1 learns future scene evolution from visual and textual conditions. Stage 2 aligns Qwen3-VL hidden states with semantic, geometric, visual-dynamic, and trajectory experts. Stage 3 fuses the expert representations to generate world-consistent ego trajectories.

### Formulation

Standard VLA is $p_\theta(\mathbf{A}_{t+1:t+T}\mid o_t,c_t)$. CoWorld-VLA inserts a structured latent:

$$p_\theta(\mathbf{A}_{t+1:t+T}\mid o_t,c_t,\mathcal{Z}),\qquad \mathcal{Z}=\{z_{\mathrm{sem}},z_{\mathrm{geo}},z_{\mathrm{dyn}},z_{\mathrm{traj}}\}$$

instantiated by the hidden states at four groups of learnable expert-token positions in the VLM input sequence. These are explicitly *not* decoded as text.

### Stage 1 — Action-conditioned predictive world model

A Wan2.2-5B DiT is trained in the frozen Wan VAE latent space on 8 Hz nuPlan video. Flow matching is applied **only to the future segment**, with history left noise-free as observed context:

$$\tilde{\mathbf{z}}_{f,\sigma}=(1-\sigma)\mathbf{z}_f+\sigma\boldsymbol{\epsilon},\quad \mathbf{v}_{\mathrm{target}}=\boldsymbol{\epsilon}-\mathbf{z}_f,\quad \mathcal{L}_{\mathrm{flow}}=\mathbb{E}\left\|\mathcal{F}_\theta(\tilde{\mathbf{z}}_\sigma,\mathbf{c},\sigma)_f-(\boldsymbol{\epsilon}-\mathbf{z}_f)\right\|_2^2$$

**"Action-conditioned" here means text.** Appendix A.1.1 is explicit: the condition is a serialized natural-language prompt $\mathcal{P}=[\mathrm{Scene}]\oplus[\mathrm{Speed}]\oplus[\mathrm{Navigation}]\oplus[\mathrm{Trajectory}]$ encoded by Wan's frozen UMT5 text encoder — ego speed as a coarse phrase ("nearly stopped", "driving at moderate speed"), navigation as one of four commands, and the future waypoints serialized as a polyline with the current position anchored at the origin. Headings are omitted on the grounds that polyline shape plus command already determines direction.

**This is a distinctive and slightly odd choice**: a continuous control signal is round-tripped through a text encoder to condition a video model. It buys graceful degradation — Appendix A.1.1 specifies fallbacks when the command or the future trajectory is missing — at the cost of quantizing the intent.

### Stage 2 — Multi-expert representation learning

Four groups of learnable expert tokens are appended after the image and text tokens; their output hidden states become $H_{\mathrm{sem}},H_{\mathrm{geo}},H_{\mathrm{dyn}},H_{\mathrm{traj}}$.

**Semantic and geometric branches align to the *future* frame**, not the current one:

$$Z_{\mathrm{sem}}=\operatorname{Pool}(E_{\mathrm{sem}}(o_{\mathrm{fut}})),\quad Z_{\mathrm{geo}}=\operatorname{Pool}(E_{\mathrm{geo}}(o_{\mathrm{fut}}))$$
$$\mathcal{L}_{\mathrm{sem}}=\lambda_{l1}\operatorname{SmoothL1}(\hat Z_{\mathrm{sem}},Z_{\mathrm{sem}})+\lambda_{\cos}(1-\cos(\hat Z_{\mathrm{sem}},Z_{\mathrm{sem}})),\qquad \mathcal{L}_{\mathrm{geo}}=\operatorname{MSE}(\hat Z_{\mathrm{geo}},Z_{\mathrm{geo}})$$

so both are genuine future-prediction objectives in the JEPA sense, with pooling to match token counts and lightweight projections into each teacher's feature space.

**The dynamic branch is the clever one.** $H_{\mathrm{dyn}}$ **replaces the Stage-1 text condition** as the conditioning input to the Wan world model, and the flow-matching loss then back-propagates into the VLM token:

$$\hat o_{\mathrm{fut}}=W_\psi(o_t,H_{\mathrm{dyn}}),\qquad \mathcal{L}_{\mathrm{dyn}}=\mathcal{L}_{\mathrm{flow}}(\hat o_{\mathrm{fut}},o_{\mathrm{fut}};o_t,H_{\mathrm{dyn}})$$

The paper is precise about why this matters: *"The VLM itself does not directly decode future images... This design allows the dynamic tokens to learn future motion trends and temporal consistency without requiring the VLM backbone to act as a pixel-level generator."* The video model is used as a **differentiable critic on a latent**, which is a cheaper way to impose a generative future objective than making the VLM generate.

Total: $\mathcal{L}=w_{\mathrm{dyn}}\mathcal{L}_{\mathrm{dyn}}+w_{\mathrm{sem}}\mathcal{L}_{\mathrm{sem}}+w_{\mathrm{geo}}\mathcal{L}_{\mathrm{geo}}+w_{\mathrm{traj}}\mathcal{L}_{\mathrm{traj}}$ with $(1.0,\,0.1,\,0.1,\,1.0)$.

### Stage 3 — Hierarchical Multi-Expert Fusion

Two streams enter a joint-self-attention denoiser. The **clean scene stream** is Perceiver-compressed from three sources — non-action VLM tokens, current-frame V-JEPA tokens, and current-frame VGGT tokens, each with its own compressor. The **noisy action stream** carries, per expert and per future step, the noisy action, timestep embedding, MLP-encoded ego history and status, and that expert's per-step feature (obtained by a bidirectional Transformer over the expert's token group, averaged within each horizon step).

Rectified-flow training with a logit-normal $\tau$ schedule, in a $[-1,1]$-normalized $(x,y,\psi)$ space using empirical NAVSIM coordinate ranges:

$$A_\tau=(1-\tau)\epsilon+\tau A^{\mathrm{norm}},\qquad \mathcal{L}_{\mathrm{diff}}=\frac{1}{N_e}\sum_{e=1}^{N_e}\left\|\hat A_e-A^{\mathrm{norm}}\right\|_2^2,\qquad \bar A=\sum_e \alpha_e \hat A_e$$

**Two details decide what HMEF actually is.** Every expert branch is supervised against the *same* target $A^{\mathrm{norm}}$, and the fusion weights $\alpha=\operatorname{softmax}(w)$ are **global scalars**, not scene-conditioned. And Appendix A.1.3 says the expert trajectories are **detached** before the weighted average, so $\mathcal{L}_{\mathrm{act}}$'s fusion term updates only $w$.

So HMEF is a **four-member ensemble of denoisers that differ only in conditioning, averaged with a fixed learned convex combination**. It is variance reduction over a shared target, not multimodal proposal generation — there is no candidate set, no scorer, and exactly one output trajectory. That is worth being clear about before reading it against [[concepts/selection-based-planning.md]] or [[concepts/adaptive-routing.md]], where it belongs to neither family.

### What runs at inference

Not the world model. The paper's limitations section states: *"the Wan model is not used in Stage 3 or during planning inference, while the Stage-2 VLM and the V-JEPA/VGGT encoders are frozen during Stage 3."* So deployment needs **Qwen3-VL-2B + frozen V-JEPA + frozen VGGT + a 10-step HMEF denoiser**, and Wan2.2-5B is discarded after Stage 2.

That places CoWorld-VLA in the **training-time-only** camp for its video world model, alongside [[sources/simwam.md]], [[sources/drivevla-w0.md]] and [[sources/latent-wam.md]] — but with a twist none of them has: the *token* the video model supervised survives into inference and conditions a denoising branch directly. The generator is discarded; its gradient's residue is not. **No latency is reported for any of this**, and two frozen foundation encoders in the inference input path is a real cost that nothing in the paper prices.

### Training configuration

| Stage | Data | Steps | Batch | GPUs | Hours |
|---|---|---:|---:|---|---:|
| 1 — video DiT | nuPlan 8 Hz | 48k | 192 | 64 × A800 | ~74 |
| 2 — VLM multi-expert | NAVSIM-v1 | 40k | — | 32 × A800 | ~70 |
| 3 — action expert (VLM frozen) | NAVSIM-v1 | 60k | 256 | 16 × A800 | ~20 |

Learning rates: VLM $2\times10^{-5}$, JEPA adaptor $1\times10^{-4}$, new modules $1\times10^{-5}$, Stage 3 $2\times10^{-5}$. Inference: 20 sampling steps for video, **10 for trajectory**. Total ≈ **7,296 A800-GPU-hours**.

---

## Results

### Table 1 — NAVSIM-v1 navtest (PDMS)

† = no reinforcement learning. ¶ = CoWorld-VLA with the ReCogDrive action expert substituted for HMEF.

| Method | Ref | Sensors | Frames | NC↑ | DAC↑ | TTC↑ | Comf.↑ | EP↑ | PDMS↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| *End-to-end* | | | | | | | | | |
| UniAD | CVPR'23 | C | N | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| Hydra-MDP | arXiv'24 | C & L | N | 98.3 | 96.0 | 94.6 | 100 | 78.7 | 86.5 |
| [[sources/diffusiondrive.md]] | CVPR'25 | C & L | N | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| TrajDiff | arXiv'25 | C & L | N | 98.1 | 97.0 | 94.3 | 100 | 82.7 | 88.5 |
| *World model* | | | | | | | | | |
| LAW | ICLR'25 | C | N | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 |
| FSDrive | NeurIPS'25 | C | N | 98.2 | 93.8 | 93.3 | 99.9 | 80.1 | 85.1 |
| [[sources/epona.md]] | ICCV'25 | C | N | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 |
| ReSim | NeurIPS'25 | C | N | – | – | – | – | – | 86.6 |
| [[sources/policy-world-model.md]] | NeurIPS'25 | C | N | 98.6 | 95.9 | 95.4 | 100 | 81.8 | 88.1 |
| WoTE | ICCV'25 | C & L | N | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| ResWorld | ICLR'26 | C & L | N | 98.9 | 96.5 | 95.6 | 100 | 83.1 | 89.0 |
| WorldDrive | arXiv'26 | C | N | 98.4 | 96.8 | 95.2 | 100 | 83.3 | 89.0 |
| [[sources/drivelaw.md]] | CVPR'26 | C | N | **99.0** | **97.1** | **96.7** | 100 | 81.3 | 89.1 |
| *Vision-language* | | | | | | | | | |
| [[sources/recogdrive.md]] † | ICLR'26 | C | 1 | 98.1 | 94.7 | 94.2 | 100 | 80.9 | 86.5 |
| [[sources/drivevla-w0.md]] | ICLR'26 | C | N | 98.4 | 95.3 | 95.4 | 100 | 80.9 | 87.2 |
| LaST-VLA † | arXiv'26 | C | 1 | 98.7 | 95.4 | 95.7 | 100 | 80.5 | 87.3 |
| [[sources/sgdrive.md]] † | CVPR'26 | C | N | 98.6 | 95.1 | 95.4 | 100 | 81.2 | 87.4 |
| Uni-World VLA | arXiv'26 | C | N | 98.7 | 96.7 | 96.1 | 100 | 83.2 | **89.4** |
| CoWorld-VLA ¶ | – | C | 1 | 98.5 | 96.9 | 95.4 | 100 | 83.2 | 89.1 |
| **CoWorld-VLA** | – | **C** | **1** | **99.1** | **97.0** | **96.5** | 100 | **84.0** | **90.0** |

**90.0 from a single front-camera frame and a 2B VLM, with no RL and no scorer, is the result worth crediting.** Every *other* entry at or above 89.0 in this table uses multiple frames (ResWorld 89.0, WorldDrive 89.0, DriveLaW 89.1, Uni-World VLA 89.4), and ResWorld additionally uses LiDAR. On the wiki's full v1 ladder 90.0 sits around 27th — behind CLEAR 93.7, DA-WAM 93.7, DriveSuprim 93.5, Drive-JEPA 93.3, WCog-VLA 92.9, LWDrive 92.0, WA-JEPA 91.8 and a dozen others — but almost all of those spend more input, more parameters, RL, or a scorer.

**Comparison-scope caveat, and it is an unusual one.** The table omits the entire wiki frontier, which is normal. What is not normal is that it omits **[[sources/wa-jepa.md]] at 91.8 PDMS — the same lab, the same GitHub organization, ten shared authors — and does not cite it anywhere.** The most likely explanation is chronology: CoWorld-VLA's arXiv identifier is 2605 (May 2026), WA-JEPA's is 2608 (August 2026), so the table was probably assembled before WA-JEPA existed and not refreshed for v3. It remains the case that a reader cannot see the group's stronger result from this paper.

**One baseline-provenance issue.** DriveVLA-W0 appears at **87.2**, which is [[sources/drivelaw.md]]'s † reimplementation value, not that method's published 90.2★ or 88.4 single-sample. It is carried here unmarked. Against that, three rows *are* correctly marked † for no-RL variants (ReCogDrive 86.5, LaST-VLA 87.3, SGDrive 87.4) — better labelling discipline than most tables this wiki has audited.

### Table 2 — NAVSIM-v2 navtest (EPDMS) {#two-columns}

EPDMS* = before the benchmark bug fix; EPDMS = after.

| Method | NC↑ | DAC↑ | DDC↑ | TL↑ | EP↑ | TTC↑ | LK↑ | HC↑ | EC↑ | EPDMS*↑ | EPDMS↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TransFuser | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 | – |
| [[sources/diffusiondrive.md]] | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | – | 84.5 |
| [[sources/drivesuprim.md]] | 97.8 | 97.9 | 99.5 | 99.9 | **90.6** | 97.1 | 96.6 | 98.3 | 77.9 | 86.0 | – |
| [[sources/diffusiondrive-v2.md]] | 97.7 | 96.6 | 99.2 | 99.8 | 88.9 | 97.2 | 96.0 | 97.8 | **91.0** | 85.5 | 87.5 |
| WoTE | 98.5 | 96.8 | 98.8 | 99.8 | 86.1 | 97.9 | 95.5 | 98.3 | 82.9 | – | 87.7 |
| [[sources/dreameraD.md]] | 98.0 | 97.2 | 99.5 | 99.8 | 87.8 | 97.4 | 97.5 | 98.3 | 72.4 | – | 87.7 |
| [[sources/policy-world-model.md]] | 98.8 | 95.9 | 99.4 | 99.9 | 86.4 | 98.4 | 97.6 | 98.3 | 85.3 | – | 88.2 |
| [[sources/drivelaw.md]] | 98.7 | 96.9 | 99.6 | 99.8 | 87.5 | 98.3 | 97.6 | **98.4** | 77.4 | – | 88.6 |
| [[sources/drive-jepa.md]] (R34) | 98.8 | 97.4 | 99.0 | 99.8 | 83.5 | 98.0 | 96.2 | 98.1 | 85.6 | 85.4 | – |
| [[sources/latent-wam.md]] | 98.1 | 97.3 | 99.6 | 99.8 | 87.7 | 97.3 | 97.6 | 98.1 | 87.3 | – | 89.3 |
| [[sources/wam-flow.md]] | 98.5 | 94.5 | 99.5 | 99.8 | 86.9 | 96.8 | 97.4 | 97.6 | 73.9 | 84.7 | – |
| [[sources/recogdrive.md]] † | 98.3 | 95.2 | 98.3 | 99.8 | 87.1 | 97.5 | 96.6 | **99.5** | 86.5 | 83.6 | – |
| [[sources/drivevla-w0.md]] | 98.5 | **99.1** | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | – | 86.1 |
| [[sources/sgdrive.md]] † | 98.6 | 94.3 | 99.5 | 99.9 | 86.0 | 97.9 | 96.1 | 98.3 | 85.9 | – | 86.2 |
| DriveWorld-VLA | 98.6 | **99.1** | 99.6 | 99.8 | 87.4 | 97.9 | 97.0 | 97.8 | 78.6 | – | 86.8 |
| **CoWorld-VLA** | **99.1** | 97.0 | 99.6 | **99.9** | 87.8 | **98.5** | **97.7** | 98.2 | 86.2 | 86.2 | **90.0** |

**The 86.2 / 90.0 pair matches exactly what [[sources/wa-jepa.md]] reported for CoWorld-VLA second-hand** — but see [below](#one-lab): that is not independent corroboration, because WA-JEPA is the same lab.

**This table adds four NAVSIM-v2 numbers the wiki did not have**, and their provenance is the problem:

| Method | v2 EPDMS here | Status in the wiki |
|---|---:|---|
| WoTE | 87.7 (corrected) | v1 only (88.3) — **no v2 in its own paper** |
| [[sources/policy-world-model.md]] | 88.2 (corrected) | v1 only (88.1) — **no v2 in its own paper** |
| [[sources/drivelaw.md]] | 88.6 (corrected) | **DriveLaW reports no NAVSIM-v2 at all** — an explicit limitation on its page |
| [[sources/drive-jepa.md]] (R34) | 85.4 (pre-fix) | Wiki carries 87.8 — a *different, explicitly labelled* variant |

Three numbers appear for methods whose own papers never published them, with **no statement of where they came from.** Either CoWorld-VLA re-evaluated released checkpoints — which would be the good practice [[concepts/navsim-benchmark.md]] keeps asking for, and worth far more than the numbers themselves — or they were drawn from an unnamed source. The paper does not distinguish, and it matters which. The Drive-JEPA row is the counter-example that shows the authors *can* be careful: labelling it "(R34)" correctly marks it as the ResNet-34 configuration rather than passing it off as the headline.

**And a third distinct DriveSuprim v2 value.** 86.0 pre-fix, with submetrics (EP 90.6, EC 77.9) matching neither WA-JEPA's 87.1 row nor the shared-block 83.1 row. That is now three values from three submetric sets for one method on one split.

### Table 3 — Future video generation (FVD ↓)

| Method | SVD | GenAD | DrivingGPT | [[sources/epona.md]] | [[sources/drivelaw.md]] | **CoWorld-VLA** |
|---|---:|---:|---:|---:|---:|---:|
| Dataset | NAVSIM | OpenDV | NAVSIM | nuPlan | nuPlan | **NAVSIM** |
| FVD ↓ | 227.5 | 184.0 | 142.6 | 61.3 | 55.6 | **32.7** |

**The paper's own caption disarms the comparison**: *"Dataset and evaluation settings may differ across methods; cross-setting results are provided for contextual reference."* That is more honest than most generation tables in this wiki, and it should be taken at face value — only DrivingGPT (142.6) and SVD (227.5) are on NAVSIM.

**It also omits the two NAVSIM FVD entries the wiki already has**: [[sources/policy-world-model.md]] at 85.95 and [[sources/drivedreamer-policy.md]] at 53.59. Against those, 32.7 is a genuine lead — see [[concepts/world-model-for-ad.md]] — though generation resolution is never stated, which FVD is sensitive to.

---

## Ablations

### Table 4 — Which experts, and are they complementary?

| EgoT. | Geo. | Sem. | Dyn. | NC↑ | DAC↑ | TTC↑ | Comf.↑ | EP↑ | PDMS↑ |
|:-:|:-:|:-:|:-:|---:|---:|---:|---:|---:|---:|
| ✓ | | | | 97.7 | 92.7 | 92.7 | 100 | 78.5 | 83.7 |
| ✓ | ✓ | | | 97.7 | 93.9 | 92.7 | 100 | 80.3 | 85.1 |
| ✓ | | ✓ | | 98.1 | 93.7 | 93.8 | 100 | 79.0 | 85.2 |
| ✓ | | ✓ | ✓ | 98.3 | 95.4 | 95.1 | 100 | 81.1 | 87.3 |
| ✓ | ✓ | ✓ | | 98.4 | 95.6 | 95.0 | 100 | 81.9 | 87.7 |
| ✓ | ✓ | ✓ | ✓ | 98.4 | 96.5 | 95.3 | 100 | 82.3 | **88.7** |

Marginal contributions, which are strongly context-dependent:

| Expert added | To EgoT. | To EgoT.+Sem. | To EgoT.+Geo.+Sem. |
|---|---:|---:|---:|
| Geometric (VGGT) | +1.4 | +2.5 | – |
| Semantic (V-JEPA) | +1.5 | – | – |
| Dynamic (Wan) | *not measured* | +2.1 | +1.0 |

**The complementarity claim holds** — no expert is redundant, and the full set beats every three-expert subset. But two things are worth recording against the paper's reading:

- **The dynamic expert is never measured alone**, and it is the one the paper's title is about. The two cells that do contain it (+2.1 and +1.0) are both *on top of* the semantic expert. The single missing row, EgoT.+Dyn., is the one that would price the world-model branch in isolation. (This is the same shape as [[sources/drivefuture.md]]'s missing `use_wm` ablation, one ingest earlier.)
- **The geometric expert beats the dynamic one as a third addition**: EgoT.+Geo.+Sem. reaches 87.7 against EgoT.+Sem.+Dyn.'s 87.3. That ordering matters because the learned fusion weights say the opposite — see [below](#weights).

**Where the gains land.** Across the whole sweep, DAC moves +3.8 (92.7 → 96.5) and EP +3.8 (78.5 → 82.3) while NC moves +0.7 and TTC +2.6. The multi-expert representation buys **drivable-area compliance and progress** far more than collision avoidance, which matches this wiki's running observation that geometric supervision buys progress rather than safety ([[concepts/world-model-for-ad.md]], GeoWorldAD / UDT / GeoWAM).

### Table 5 — Representation × planner: the wiki's first complete 2×2 {#factorial}

| Representation | Planner | NC↑ | DAC↑ | TTC↑ | Comf.↑ | EP↑ | PDMS↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| EgoT. only | VLM only | 97.7 | 92.7 | 92.7 | 100 | 78.5 | 83.7 |
| Full Experts | VLM only | 98.4 | 96.5 | 95.3 | 100 | 82.3 | 88.7 |
| Full Experts | VLM + AE (ReCogDrive) | 98.5 | 96.9 | 95.4 | 100 | 83.2 | 89.1 |
| EgoT. only | VLM + HMEF | 98.3 | 96.8 | 95.0 | 100 | 83.2 | 88.9 |
| Full Experts | VLM + HMEF | **99.1** | **97.0** | **96.5** | 100 | **84.0** | **90.0** |

Reading it as a factorial:

| | VLM readout | + HMEF | Effect of HMEF |
|---|---:|---:|---:|
| **EgoT. only** | 83.7 | 88.9 | **+5.2** |
| **Full Experts** | 88.7 | 90.0 | **+1.3** |
| **Effect of experts** | **+5.0** | **+1.1** | interaction **−3.9** |

**Each intervention is worth about five points alone and about one point on top of the other.** Additively one would expect 83.7 + 5.0 + 5.2 = 93.9; the observed joint value is 90.0. The paper describes this as "complementary benefits from the planner and multi-expert representations," which is true in the weak sense that neither is redundant — but the dominant story in its own table is **substitution**: a diffusion action expert on a plain trajectory token already reaches 88.9, above every no-RL VLA baseline in Table 1 (ReCogDrive† 86.5, DriveVLA-W0 87.2, LaST-VLA† 87.3, SGDrive† 87.4).

**How much weight this deserves.** PDMS is bounded and compresses near the top, so some sub-additivity is mechanical — but 90.0 is well short of the 94.8 human reference, so a ceiling artifact cannot account for a −3.9 interaction on its own. Single runs, one architecture, one benchmark. What makes it valuable is that it is **complete**: [[sources/drivelaw.md]] varied representation family under a fixed planner and [[sources/unified-driving-tokens.md]] varied tokenizer objectives under a fixed readout, but neither varied both, so neither could see an interaction. This one can, and it is large and negative.

**HMEF over a strong external action expert is +0.9** (90.0 vs. 89.1 with ReCogDrive's AE) — the honest size of the paper's planner contribution once a competent diffusion planner is already in place.

### Table 6 — Denoising steps

| Steps | NC↑ | DAC↑ | TTC↑ | Comf.↑ | EP↑ | PDMS↑ |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 99.0 | 96.8 | 95.9 | 100 | 83.6 | 89.5 |
| **10** | **99.1** | **97.0** | **96.5** | 100 | **84.0** | **90.0** |
| 20 | 99.1 | 96.7 | 96.4 | 100 | 83.5 | 89.7 |

**Non-monotonic, peaking in the middle, total spread 0.5 PDMS.** This is an *action* denoiser rather than a video one, so it does not bear directly on the [[sources/drivelaw.md]]-vs-[[sources/foresight.md]] dispute about video denoising depth — but it is another instance of the same shape, and it supports the general position that more sampling steps are not monotonically better for planning.

### Table 7 — Inference-seed stability

| Metric | Seed 42 | 1 | 2 | 3 | 4 | 5 | Mean | Std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PDMS | 89.964 | 89.975 | 89.986 | 89.971 | 89.982 | 90.000 | 89.980 | **0.013** |
| EPDMS | 89.979 | 89.989 | 89.961 | 89.983 | 90.000 | 89.997 | 89.985 | **0.014** |

**The second seed-variance measurement in this wiki, after [[sources/wa-jepa.md]]'s 0.053**, and the paper is careful about what it does and does not show:

> "These results indicate low sensitivity to the evaluated inference seeds. They do not replace variance estimates over independently trained models, which were not conducted because of the high training cost."

That caveat is the most valuable sentence in the appendix. **It is the first paper in this wiki to name the training-seed gap explicitly** — every variance number on [[concepts/navsim-benchmark.md]] is sampler noise, and nobody has measured the one that would actually license 0.2-point ablation claims.

**One oddity worth flagging.** PDMS (v1 protocol) and EPDMS (v2 protocol) are different metrics with different sub-metric sets, different weights, and different penalty products, yet both means land at 89.98 and both tables headline 90.0. That can happen; two independent aggregations of the same trajectories are not forbidden from coinciding. But the coincidence is close enough that a reader should confirm the two rows are what their labels say before quoting either. The paper offers no comment.

### Learnable expert weights {#weights}

Initialized uniform at 0.25; converged to **dynamic 0.35, trajectory 0.31, semantic 0.19, geometric 0.15**.

**This contradicts the ablation ordering.** Table 4 makes the geometric expert the *stronger* third addition (87.7 with Geo. vs. 87.3 with Dyn.) and rates Geo. and Sem. near-identically in isolation (+1.4, +1.5). The learned weights rank geometry last and dynamics first.

The two measure different things and the tension is resolvable in principle — the fusion weight says how much a branch's *output trajectory* is trusted in a convex average, while the ablation says how much a branch's *token* improves the shared representation. A branch can shape the latent usefully and still produce a worse trajectory from it. But the paper reads the weights as "importance" without noticing that its own ablation disagrees, and **nothing in the paper tests whether the learned weights beat a uniform 0.25 average** — which, given that all four branches regress the same target, is the obvious control.

---

## One Lab, Two Answers {#one-lab}

**CoWorld-VLA and [[sources/wa-jepa.md]] are the same group.** Ten of CoWorld-VLA's eleven authors appear on WA-JEPA (Minqing Huang, Yujiao Xiang, Jiajie Huang, Jingqi Wang, Yuheng Zhou, Zhi Xu, Feiyang Tan, Hangning Zhou, Mu Yang, Gong Chen — only Zihan Liang is new); both list Afari Intelligent Drive, UESTC, Southeast University, BUPT and Tianjin University; both release under **`github.com/AFARI-Research`**; and each paper's corresponding author is an author of the other. This wiki previously recorded the overlap as "four authors," which understated it substantially.

**Consequence 1 — WA-JEPA's CoWorld-VLA row is first-party, not third-party.** That makes it *more* reliable as a statement about CoWorld-VLA's own score and *useless* as independent corroboration of the EPDMS correction's magnitude. The +3.8 delta recorded on [[concepts/navsim-benchmark.md]] is one lab's number reported twice.

**Consequence 2 — and this is the important one — the two papers disagree with each other about evaluator provenance.**

| Method | [[sources/wa-jepa.md]] | **CoWorld-VLA** (same lab) | [[sources/drivefuture.md]] (independent) |
|---|---|---|---|
| ReCogDrive 83.6 | pre-fix | **pre-fix** | corrected |
| [[sources/drivevla-w0.md]] 86.1 | pre-fix | **corrected** | corrected |

On DriveVLA-W0, **two papers from one lab print the same number in different columns.** No copying error explains this: both tables were assembled by overlapping author sets with access to the same pipeline, and they still disagree about which evaluator produced a value they both publish.

This is the strongest evidence this wiki has that **the pre-fix/corrected attributions circulating in these tables are inferences, not knowledge.** [[concepts/navsim-benchmark.md]] had already concluded that v2 EPDMS is not comparable across papers; this narrows it further — the attributions are not reliable *within* a lab either. The practical rule tightens to: **treat a column label as trustworthy only for the paper's own row.**

On ReCogDrive, CoWorld-VLA agreeing with WA-JEPA is one group against one, not two against one. The question stays open.

---

## Qualitative Results

![[case1_main.png|Stage 1 vs Stage 2 future generation: Stage 2 preserves driving direction and lane-level scene evolution]]

**Figure 3**: Stage 1 generates stable futures but deviates from the ground-truth driving direction at intersections; Stage 2, with VLM-conditioned latent world modelling, preserves turning behaviour and road-layout evolution.

![[traj_best.png|Stage 2 vs Stage 3 trajectory planning across scenarios]]

**Figure 4**: Stage 2 predicts reasonable directions but drifts in lane keeping and turning; Stage 3 with HMEF aligns more closely with ground truth, particularly in lateral position and turning tendency.

![[case1.png|Left-turn navigation at a forked intersection: Stage 1 predicts a straight future, Stage 2 preserves the turn]]

**Figure 5(a)**: At a forked intersection, Stage 1 predicts a straight-driving future instead of the intended left turn; Stage 2 preserves it. Panels (b) straight cruising — Stage 1 drifts into the adjacent lane — and (c) close-proximity car-following are described in the text but are not in the clipping.

![[case4.png|Local fidelity: parked vehicles blur in Stage 1 predictions and stay sharp in Stage 2]]

**Figure 6**: Red boxes mark roadside parked vehicles that blur and distort in later Stage-1 frames while Stage 2 preserves boundaries. The paper's reading — that multi-expert supervision improves fine-grained visual consistency and not only high-level direction — is the one claim here that a single figure can support, and it is unquantified.

![[traj_2.png|Lane-keeping cruising: Stage 2 drifts laterally, Stage 3 stays centred]]

**Figure 7(a)**: Lane-keeping cruising. Stage 2 captures direction with lateral drift over the horizon; Stage 3 is more centred. Panels (b) intersection left turn and (c) detour around a leading vehicle are described but absent from the clipping.

**All qualitative claims are Stage-1-vs-Stage-2 or Stage-2-vs-Stage-3.** There is no qualitative comparison against any external method anywhere in the paper.

---

## Limitations

1. **The dynamic-evolution expert — the world model the paper is named for — is never ablated in isolation.** Table 4 measures it only on top of the semantic expert (+2.1 and +1.0). The missing EgoT.+Dyn. row is one run.
2. **The representation and the planner are largely substitutes, and the paper reads the result the other way.** +5.0 / +5.2 alone, +1.1 / +1.3 together. See [The 2×2](#factorial).
3. **No latency, no parameter count, no FLOPs, no throughput.** Inference requires Qwen3-VL-2B plus **two frozen foundation encoders in the input path** (V-JEPA and VGGT, both run per frame for the scene stream) plus a 10-step four-branch denoiser. Two frozen encoders at inference is the deployment cost [[concepts/foundation-backbones-for-ad.md]] flags for [[sources/unified-driving-tokens.md]], doubled, and priced by neither.
4. **Three NAVSIM-v2 numbers appear for methods that never published them** (WoTE 87.7, PWM 88.2, DriveLaW 88.6) with no statement of provenance. If these are recomputations they are the most valuable thing in the table and should be labelled as such; if they are not, they are a new failure mode for [[concepts/navsim-benchmark.md]]'s catalogue.
5. **A third distinct DriveSuprim v2 value (86.0)** from a third submetric set, and DriveVLA-W0 carried on v1 at DriveLaW's reimplementation value (87.2) without a mark.
6. **The learned fusion weights contradict the ablation ordering** on geometry vs. dynamics, and no uniform-weight control is run — even though all four branches regress an identical target, which makes uniform averaging the natural null.
7. **Fusion weights are global scalars**, so "hierarchical fusion" is a fixed convex combination at inference with no scene conditioning. Against [[sources/clear.md]]'s scene-conditioned routing and [[sources/adaptive-wam.md]]'s quality router, this is the non-adaptive end of that design space, and the paper does not test whether adaptivity would help.
8. **Single front camera, single frame** — stated as a limitation and a genuine constraint on the result's reach, though it also makes 90.0 more impressive than the raw number suggests.
9. **≈7,300 A800-GPU-hours across three stages**, named by the paper as its primary limitation. Stage 1 alone is 4,736 GPU-hours for a video model that is then discarded before inference.
10. **No RL, no scorer, no closed-loop or reactive evaluation** — no navhard, no Bench2Drive, no HUGSIM, no nuScenes. Given that navhard is where [[concepts/navhard-ood-evaluation.md]] now shows the scoring pipeline dominates, an unscored single-trajectory method is exactly the configuration whose navhard behaviour would be informative.
11. **WA-JEPA is neither cited nor compared**, despite being the same lab's stronger method on both splits. Chronology (2605 vs. 2608) probably explains it for v1 of the paper; the clipped version is v3.
12. **PDMS and EPDMS both average 89.98 across six seeds.** Possible, but close enough to warrant a check that Table 7's two rows are what they claim.
13. **Single training run throughout**; the seed table varies inference seeds only, as the paper says.
14. **Figures 5 and 7 are multi-panel and the clipping holds only panel (a) of each.**

---

## Key Cross-References

- [[concepts/world-model-for-ad.md]] — Pattern 34 (four parallel future targets in one latent space); the first complete representation × planner factorial; the objective-form assignment by target entropy; the training-time-only camp gains a member whose supervised *token* survives inference.
- [[concepts/navsim-benchmark.md]] — CoWorld-VLA 90.0 verified at the primary source; the intra-lab evaluator-provenance contradiction; four new v2 numbers of unstated provenance; the second seed-variance measurement and the first explicit statement of the training-seed gap.
- [[concepts/foundation-backbones-for-ad.md]] — V-JEPA as an alignment target that *works*, refining [[sources/reworld.md]]'s negative result by separating generative from discriminative alignment; VGGT as a second geometry target; a fifth Wan2.2-5B coupling strategy.
- [[concepts/chain-of-thought-for-ad.md]] — multi-expert Latent CoT as a fifth CoT substrate, after text, visual, dynamics, and single-latent.
- [[concepts/diffusion-planner.md]] — per-expert denoising branches with detached learned fusion weights; the non-monotonic denoising-step sweep.
- [[sources/drivefuture.md]] — the immediately preceding ingest, making the same complaint about future latents being auxiliary rather than conditioning, and answering it with one latent instead of four. Both papers' named mechanisms turn out to be the smaller term in their own ablations.
- [[sources/wa-jepa.md]] — same lab, same GitHub org, citation in one direction only (WA-JEPA's table lists CoWorld-VLA; CoWorld-VLA never cites WA-JEPA), and a direct disagreement about evaluator provenance.
- [[sources/physwam.md]] — cites the FVD of 32.7 here as the low end of reported NAVSIM values and notes that its clip count is unstated. PhysWAM measures a **recorded-versus-recorded FVD floor of 91.5 at 600 clips**, and its own model reads 111.3 at 600 clips and 42.2 at 1,200. Until the protocol behind 32.7 is known, it cannot be ranked against anything. PhysWAM does not list this paper's 90.0 EPDMS in its planning table.
