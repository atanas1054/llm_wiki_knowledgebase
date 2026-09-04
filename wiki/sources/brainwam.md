---
title: "BrainWAM: Action-Space Coordination of Semantic Priors and Predictive Dynamics for Autonomous Driving"
type: source-summary
sources: [raw/papers/BrainWAM_ Action-Space Coordination of Semantic Priors and Predictive Dynamics for Autonomous Driving.md]
related: [sources/adaptive-wam.md, concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/mixture-of-experts.md, concepts/dual-system-vla.md, concepts/foundation-backbones-for-ad.md, concepts/diffusion-planner.md, sources/foresight.md, sources/drivelaw.md, sources/simwam.md, sources/drivewam.md, sources/driveva.md, sources/hybriddriveVLA.md, sources/drivevla-w0.md, sources/dynvla.md, sources/drivedreamer-policy.md, sources/da-wam.md, sources/wa-jepa.md, sources/drivesuprim.md, sources/automot.md, sources/unidrivevla.md, sources/recogdrive.md, sources/autovla.md, sources/epona.md]
created: 2026-09-04
updated: 2026-09-04
confidence: high
---

**Paper**: BrainWAM: Action-Space Coordination of Semantic Priors and Predictive Dynamics for Autonomous Driving
**Authors**: Bing Zhan\*, Shuyao Shang\*, Shuo Lu, Yuan Xu, Zhao Wang, Yida Wang, Xueyang Zhang, Kun Zhan, Jiahao Gu (project lead) — \*equal contribution
**Orgs**: NLPR, Institute of Automation, Chinese Academy of Sciences (CASIA) + Li Auto Inc.
**arXiv**: 2608.12854v2

---

## Summary

BrainWAM asks how to combine a VLA (semantic priors) with a WAM (predictive dynamics), and its answer is less interesting than the problem it documents on the way there.

**The finding worth the ingest**: putting VLM tokens, video-generator tokens, and action tokens into one shared attention space — a design the paper calls **Tri-MoT** — scores **87.8 PDMS, *below* the WAM-only branch at 88.1**, despite having strictly more information. The diagnosis is **modality competition**: action tokens attend more strongly to VLM tokens than to VGM tokens across most layers, because VLM tokens are clean pretrained semantic abstractions while VGM tokens are mid-denoising and low-signal. The easier modality wins the shared attention budget and suppresses the one that carries the predictive dynamics.

The fix is to never mix raw tokens. Each branch first compresses itself into **8 action tokens**, and the two branches communicate only through those — bidirectional gated cross-attention (**CAB**) at two layers, then a 2-layer Transformer fusion (**CIF**). This reaches **89.5 PDMS on NAVSIM v1** and **89.6 EPDMS on NAVSIM v2**, at **475–644 ms** on an H20.

**The second result that matters to this wiki** is Table 5. BrainWAM decouples the video and action denoising timesteps, so the number of video denoising steps executed before the features are cached becomes a free parameter. **One step gets 89.3 PDMS; three steps get 89.4.** This is independent corroboration of [[sources/drivelaw.md]]'s t=1 finding from a completely different architecture, and it puts [[sources/foresight.md]]'s claim that 100 denoising steps beat 25 in a 2-against-1 minority.

The neuroscience framing (left hemisphere / right hemisphere / corpus callosum / cerebellum) is decorative and carries no load — CAB is Flamingo-style zero-init gated cross-attention and CIF is a small Transformer plus a mean.

---

## The Problem: Attention-Allocation Mismatch

![[teaser 1.png|Four paradigms: VLA, WAM, Tri-MoT joint token fusion, and BrainWAM action-space coordination]]

**Figure 1**: Comparison of paradigms. (a) VLA leverages vision-language priors for task-aware semantic grounding but lacks explicit predictive planning. (b) WAM captures future scene evolution but has limited semantic grounding. (c) Tri-MoT jointly fuses VLM, VGM, and action tokens in a shared raw-token space, which may cause attention interference. (d) BrainWAM separates semantic and predictive pathways and coordinates them in the action space.

![[tri-mot.png|Attention ratio of action tokens to VLM vs VGM tokens across Transformer layers in Tri-MoT]]

**Figure 2**: Attention allocation in Tri-MoT. Action tokens attend more strongly to VLM tokens than to VGM tokens across most Transformer layers, especially shallow ones — semantic dominance in the joint representation space.

The paper's Appendix A argument is careful, and it anticipates the two obvious objections:

1. **"Maybe the VGM tokens really are uninformative."** Disabling video denoising entirely drops PDMS from 89.3–89.5 to **79.3**. The predictive stream carries a great deal.
2. **"Maybe Tri-MoT just needs more capacity."** Tri-MoT has strictly more information than WAM-only and comparable parameters, and still scores 0.3 lower.

So the problem is not signal availability but **competition**: once a clean, stable, large-scale-pretrained VLM stream shares an attention pool with a stream still emerging from Gaussian noise, the optimizer takes the semantic shortcut.

**Scope this claim carefully.** BrainWAM's Tri-MoT has *three* modalities including a pretrained VLM. [[sources/simwam.md]]'s mask ablation varies joint attention with only *two* (video + action, no VLM) and finds bidirectional 90.2 vs. isolated 90.3 — no competition, because there is no clean third stream to lose to. The two results are consistent, and together they sharpen the claim: **the harm comes from adding a clean semantic stream to a denoising one, not from joint attention as such.** [[sources/driveva.md]] (90.9) and [[sources/drivewam.md]] (90.1) also use joint video-action attention successfully and likewise have no VLM in the attention pool.

---

## Method

![[framework 1.png|BrainWAM architecture: WAM pathway and VLA pathway, each producing action tokens, bridged by CAB and fused by CIF]]

**Figure 3**: The VLA pathway distills scene semantics, route instructions, and rule-aware priors into semantic-grounded action tokens; the WAM pathway distills future dynamics and physical priors into prediction-grounded action tokens. Instead of mixing raw VLM and VGM tokens in a shared attention space, CAB bridges the two action streams and CIF fuses the refined action intents for trajectory decoding.

### WAM branch

**Wan2.2-TI2V-5B** video backbone plus a lightweight rectified-flow action expert, coupled by **Dual-MoT** (shared self-attention, modality-specific FFNs). Video latents and the action trajectory are perturbed with **independent** rectified-flow timesteps $t_v, t_a$:

$$x^{v}_{t_{v}}=(1-t_{v})x^{v}_{0}+t_{v}\epsilon^{v},\qquad x^{a}_{t_{a}}=(1-t_{a})x^{a}_{0}+t_{a}\epsilon^{a}$$

$$\hat{u}^{v},\,\hat{u}^{a}_{\mathrm{pred}}=F_{\mathrm{WAM}}\!\left(x^{v}_{t_{v}},\,x^{a}_{t_{a}},\,t_{v},\,t_{a},\,c_{\mathrm{obs}}\right)$$

with velocity targets $u^{v}=\epsilon^{v}-x^{v}_{0}$, $u^{a}=\epsilon^{a}-x^{a}_{0}$ and loss $\mathcal{L}_{\mathrm{WAM}}=\mathcal{L}_{\mathrm{vid}}+\lambda_{\mathrm{pred}}^{\mathrm{a}}\mathcal{L}_{\mathrm{pred}}^{\mathrm{a}}$.

**The decoupled schedule is the architectural key to the efficiency result**: because $t_v$ is independent of $t_a$, the video stream can terminate early, cache its features, and let the action stream keep denoising against them.

### VLA branch

**Qwen3-VL-4B** encodes multi-view images and driving instructions into semantic tokens $U$, ego history into state tokens $E$. A lightweight action expert denoises $x^a_{t_a}$ into action tokens $A_{\mathrm{sem}}$, again with Dual-MoT coupling:

$$\hat{u}^{a}_{\mathrm{sem}}=F_{\mathrm{VLA}}\!\left(U,\,E,\,x^{a}_{t_{a}},\,t_{a}\right),\qquad \mathcal{L}_{\mathrm{sem}}^{\mathrm{a}}=\mathbb{E}\|\hat{u}^{a}_{\mathrm{sem}}-u^{a}\|_{2}^{2}$$

### Callosal Action Bridge (CAB)

Bidirectional cross-attention between the two action streams, with **zero-initialized tanh gates** (following Flamingo):

$$M_{\mathrm{pred}\leftarrow\mathrm{sem}}^{l}=\Psi_{\mathrm{cab}}^{l}(A_{\mathrm{pred}}^{l},A_{\mathrm{sem}}^{l}),\qquad M_{\mathrm{sem}\leftarrow\mathrm{pred}}^{l}=\Psi_{\mathrm{cab}}^{l}(A_{\mathrm{sem}}^{l},A_{\mathrm{pred}}^{l})$$

$$\widetilde{A}_{x}^{l}=A_{x}^{l}+\tanh(g_{x}^{l})\odot\operatorname{Attn}(A_{x}^{l},A_{y}^{l}),\qquad g_{x}^{l}\in\mathbb{R}^{1024}\ \text{zero-init}$$

Concrete dimensions, from Appendix B: **each stream carries $L=8$ action tokens at hidden dim 1024**; CAB is inserted at **layers 9 and 18** of the two action experts; 8 heads × 128 dim; no bias on Q/K/V/O projections; separate norms for query and context streams. **Two CAB blocks total ≈ 16.8M parameters.**

The compression ratio is the point. A Wan2.2 video latent stream and a Qwen3-VL token stream both get squeezed to 8 tokens before they are allowed to talk. There is no attention budget to compete over.

### Cerebellar Intent Fusion (CIF)

Both streams are projected to a shared 1024-dim space with a learnable source embedding, concatenated, passed through a **2-layer Transformer (8 heads) with action-timestep-conditioned AdaLN**, then averaged element-wise:

$$Z_{\mathrm{pred}},Z_{\mathrm{sem}}=\mathrm{CIF}(\tilde{A}_{\mathrm{pred}}^{L},\tilde{A}_{\mathrm{sem}}^{L}),\qquad Z=\mathcal{M}(Z_{\mathrm{pred}},Z_{\mathrm{sem}})$$

$$\hat{u}^{a}_{\mathrm{fuse}}=D_{\mathrm{fuse}}(Z,t_{a}),\qquad \mathcal{L}_{\mathrm{fuse}}=\mathbb{E}\|\hat{u}^{a}_{\mathrm{fuse}}-u^{a}\|_{2}^{2}$$

**≈49.3M parameters.** Only $\mathcal{L}_{\mathrm{fuse}}$ is supervised in Stage 3.

### Three-stage training

![[training_pipeline.png|Stage 1 trains the WAM branch, Stage 2 the VLA branch, Stage 3 freezes both and trains only CAB, CIF, and the action decoder]]

**Figure 4**: Stage 1 trains the WAM branch with video and action rectified-flow objectives. Stage 2 trains the VLA branch with visual and language inputs. **Stage 3 freezes both branches** and optimizes only CAB, CIF, and the action decoder.

Each stage: 100K steps, 8× NVIDIA H20, batch 6/GPU, AdamW, peak lr $5\times10^{-5}$, cosine schedule, 200 warmup steps, weight decay 0.01, bf16, DeepSpeed ZeRO-2, checkpoints every 3K steps. Inference uses **3-step rectified-flow action sampling**.

---

## Results

### Table 1 — NAVSIM v1 (PDMS)

| Method | Ref. | Img | LiDAR | NC ↑ | DAC ↑ | TTC ↑ | C ↑ | EP ↑ | PDMS ↑ |
|---|---|:-:|:-:|---:|---:|---:|---:|---:|---:|
| Human | – | – | – | 100.0 | 100.0 | 100.0 | 99.9 | 87.5 | 94.8 |
| *Traditional end-to-end* | | | | | | | | | |
| TransFuser | TPAMI'23 | ✓ | ✓ | 97.7 | 92.8 | 92.8 | 100.0 | 79.2 | 84.0 |
| UniAD | CVPR'23 | ✓ | | 97.8 | 91.9 | 92.9 | 100.0 | 78.8 | 83.4 |
| PARA-Drive | CVPR'24 | ✓ | | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| DiffusionDrive | CVPR'25 | ✓ | ✓ | 98.2 | 96.2 | 94.7 | 100.0 | 82.2 | 88.1 |
| *Vision-Language-Action* | | | | | | | | | |
| ReCogDrive | ICLR'26 | ✓ | | 98.1 | 94.7 | 94.2 | 100.0 | 80.9 | 86.5 |
| DynVLA | ICML'26 | ✓ | | 98.6 | 95.3 | 95.5 | 100.0 | 80.6 | 87.2 |
| AutoVLA | NeurIPS'25 | ✓ | | 98.4 | 95.6 | **98.0** | 99.9 | 81.9 | 89.1 |
| DriveVLA-W0 | ICLR'26 | ✓ | | 98.4 | 95.3 | 95.2 | 100.0 | 80.9 | 87.2 |
| *World-model-based* | | | | | | | | | |
| DrivingGPT | ICCV'25 | ✓ | | 98.9 | 90.7 | 94.9 | 95.6 | 79.7 | 82.4 |
| LAW | ICLR'25 | ✓ | | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 |
| Epona | ICCV'25 | ✓ | | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 |
| WoTE | ICCV'25 | ✓ | ✓ | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| DriveLaW | CVPR'26 | ✓ | | **99.0** | 97.1 | 96.7 | 100.0 | 81.3 | 89.1 |
| **BrainWAM (Ours)** | – | ✓ | | 98.1 | **97.5** | 94.9 | 100.0 | **83.8** | **89.5** |

Camera-only, no LiDAR. Gains are concentrated in **DAC (97.5, best in the table) and EP (83.8, best)** — drivable-area compliance and progress. NC 98.1 and TTC 94.9 are mid-table.

**Three baseline rows are weaker configurations than the published headline, none of them marked as such** — see Limitations. In wiki terms the corrected values are ReCogDrive 89.6, DynVLA 91.7, DriveVLA-W0 90.2★.

### Table 2 — NAVSIM v2 (EPDMS)

| Method | NC ↑ | DAC ↑ | DDC ↑ | TLC ↑ | EP ↑ | TTC ↑ | LK ↑ | HC ↑ | EC ↑ | EPDMS ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *Traditional end-to-end* | | | | | | | | | | |
| TransFuser | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| HydraMDP++ | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| DriveSuprim | 97.5 | 96.5 | 99.4 | 99.6 | **88.4** | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| ARTEMIS | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | **89.1** | 83.1 |
| *Vision-Language-Action* | | | | | | | | | | |
| DriveVLA-W0 | **98.5** | **99.1** | 98.0 | 99.7 | 86.4 | **98.1** | 93.2 | 97.9 | 58.9 | 86.1 |
| *World-model-based* | | | | | | | | | | |
| DriveDreamer-Policy | 98.4 | 97.1 | 99.5 | **99.9** | 87.9 | 97.7 | **97.6** | 98.3 | 79.4 | 88.7 |
| **BrainWAM (Ours)** | 98.1 | 97.5 | **99.6** | **99.9** | 88.2 | 97.4 | **97.6** | **98.4** | 85.8 | **89.6** |

**EC 85.8 is unusually strong** for a world-model method — compare DriveVLA-W0's 58.9 (the wiki's lowest) and DriveDreamer-Policy's 79.4. Extended comfort has been a systematic weakness of generative planners; BrainWAM's 3-step rectified-flow action sampling with a fused, averaged latent is a plausible reason, but the paper does not investigate it.

**Protocol placement is undeterminable** — see [Limitations](#protocol) and [[concepts/navsim-benchmark.md]].

---

## Ablations

### Table 3 — Branch and coordination strategy (NAVSIM v1) — the paper's central result

| Method | NC ↑ | DAC ↑ | TTC ↑ | C ↑ | EP ↑ | PDMS ↑ |
|---|---:|---:|---:|---:|---:|---:|
| VLA-only | 97.7 | 94.9 | 93.3 | 100.0 | 80.7 | 86.1 |
| WAM-only | 98.0 | 96.4 | 94.4 | 100.0 | 82.6 | 88.1 |
| **Tri-MoT** (raw-token fusion) | 98.3 | 96.2 | 94.7 | 100.0 | 81.7 | **87.8** |
| **BrainWAM** | 98.1 | **97.5** | **94.9** | 100.0 | **83.8** | **89.5** |

Two things to read out:

**Tri-MoT (87.8) < WAM-only (88.1).** Adding a 4B pretrained VLM to a working WAM, through shared attention, makes it *worse*. This is the wiki's first measurement of a VLM being actively harmful to a world-model planner, and it is the clean version of a suspicion several papers have gestured at.

**WAM-only (88.1) >> VLA-only (86.1).** On NAVSIM, the predictive branch is worth 2.0 PDMS more than the semantic branch on its own. Consistent with the wiki's broader pattern that video-prior methods do well on this benchmark, and it means BrainWAM's +1.4 over WAM-only is the honest attribution of the coordination mechanism.

### Table 4 — CAB and CIF

| CAB | CIF | NC ↑ | DAC ↑ | TTC ↑ | C ↑ | EP ↑ | PDMS ↑ |
|:-:|:-:|---:|---:|---:|---:|---:|---:|
| ✓ | | 98.1 | 96.8 | 94.8 | 100.0 | 83.0 | 88.7 |
| | ✓ | 98.1 | 96.7 | 94.7 | 100.0 | 82.9 | 88.5 |
| ✓ | ✓ | 98.1 | **97.5** | **94.9** | 100.0 | **83.8** | **89.5** |

Either alone lands at 88.5–88.7 (above WAM-only 88.1 but below the full model); together 89.5. Gains concentrate in DAC and EP, matching the main table.

### Table 5 — Asynchronous video denoising {#async}

| Video denoise steps | Latency ↓ | PDMS ↑ | EPDMS ↑ |
|---:|---:|---:|---:|
| 0 | **382 ms** | 79.3 | 75.8 |
| **1** | **475 ms** | 89.3 | 89.4 |
| 2 | 565 ms | **89.5** | **89.6** |
| 3 | 644 ms | 89.4 | **89.6** |

Measured on a single H20. **One video denoising step recovers 89.3 of the achievable 89.5**; steps 2 and 3 add 0.2 and then nothing, for 169 ms.

This is the wiki's third measurement of how much denoising a planner's conditioning latent needs, and it **agrees with DriveLaW and disagrees with ForeSight** — see [Key Cross-References](#crossrefs).

The 0-step row (79.3) is the paper's evidence that "video dynamics are essential to planning," and it is the weakest experiment in the paper. With zero denoising steps the video stream is **pure Gaussian noise**, not absent — the action expert was trained to attend to partially-denoised features and is now fed noise. That is a distribution-shift ablation, and a −10.2 PDMS collapse is what one would expect from feeding garbage into any trained pathway. The clean version of this experiment is SimWAM's isolated attention mask: retrain *without* the dependency and see what is lost. BrainWAM does not run it.

### Tables 6–8 — CAB count, CIF fusion type, CIF depth

Appendix ablations use **10-step joint denoising** rather than the asynchronous schedule, so their reference point is 89.3, not 89.5. The paper states this explicitly, which is better practice than most.

| # CAB blocks | PDMS ↑ | | CIF fusion | PDMS ↑ | | CIF layers | PDMS ↑ |
|---:|---:|---|---|---:|---|---:|---:|
| 1 | 88.9 | | MLP | 88.8 | | 1 | 89.0 |
| **2** | **89.3** | | Gate | 89.1 | | **2** | **89.3** |
| 3 | 89.2 | | **Transformer** | **89.3** | | 3 | 89.3 |
| 5 | 89.3 | | | | | | |
| 28 | 89.3 | | | | | | |

**Cross-stream communication saturates after two CAB blocks** — 28 blocks (every layer) scores the same 89.3 as 2. With 8 tokens per stream there is very little to exchange, and the paper's own numbers say most of it transfers immediately. Token-level interaction (Transformer) beats feature-wise gating by 0.2 and direct MLP projection by 0.5.

### Table 9 — Stage-3 update strategy

| Stage-3 update strategy | PDMS ↑ |
|---|---:|
| Full-model fine-tuning | 88.8 |
| **CAB, CIF, and action decoder only** | **89.5** |

**Freezing both branches beats end-to-end fine-tuning by 0.7.** The paper's explanation is a convergence-rate mismatch, supported by a genuinely useful datum: **VLA-only reaches 86.1 PDMS after 54K steps; WAM-only needs 81K steps to reach 88.1.** Unfreezing both means CAB and CIF receive representations that are still moving, at different rates. This is a concrete, transferable argument for freeze-then-coordinate in two-backbone systems, and it lines up with ForeSight's stability rationale for its own two-phase schedule.

---

## Qualitative Analysis

![[qualitative 1.png|VLA-only, WAM-only, and BrainWAM across four scenario types]]

**Figure 5**: Four representative cases. **VLA-only wins** on navigation following (taking the routed branch rather than a locally plausible one) and red-light understanding (jointly reading a lead vehicle's brake lights and a red light). **WAM-only wins** on interactive negotiation (coupled ego/pedestrian/agent behavior) and trajectory feasibility on curved roads. BrainWAM handles all four.

![[appendix.png|Additional VLA-only vs WAM-only vs BrainWAM comparisons in BEV and front view]]

**Figure 6**: Rows 1–2 show WAM-only succeeding where VLA-only fails; rows 3–5 the opposite; the last row shows both single-branch models failing while BrainWAM still produces a reasonable trajectory. The complementarity is qualitative — no failure-set overlap statistics are reported, unlike [[sources/hybriddriveVLA.md]]'s set-level complementarity analysis, which is the right instrument for this claim.

---

## Implementation Summary

| Component | Backbone / size |
|---|---|
| WAM video backbone | Wan2.2-TI2V-5B |
| VLA backbone | Qwen3-VL-4B |
| Action experts | Two lightweight rectified-flow experts (size unreported) |
| CAB | 2 blocks @ layers 9, 18; 8 tokens × 1024 dim; 8 heads × 128 — **16.8M** |
| CIF | 2-layer Transformer, 8 heads, AdaLN on $t_a$ — **49.3M** |
| Training | 3 × 100K steps, 8× H20, batch 6/GPU, AdamW lr 5e-5, cosine, bf16, ZeRO-2 |
| Inference | 3-step action sampling; 1–3 video steps; **475–644 ms on H20** |
| Benchmarks | NAVSIM v1 (89.5 PDMS), NAVSIM v2 (89.6 EPDMS) |

---

## Limitations

1. **89.5 PDMS is mid-frontier, and the v1 table omits everything above it** — CLEAR 93.7, DA-WAM 93.7, DriveSuprim 93.5, Drive-JEPA 93.3, HybridDriveVLA 92.1, WA-JEPA 91.8, DynVLA 91.7, SimWAM 91.5, FLARE 91.4, DiffusionDriveV2 91.2, SGDrive 91.1, DriveVA 90.9. The best entries it does include are AutoVLA and DriveLaW at 89.1.

2. **DriveSuprim appears in Table 2 but not Table 1.** This is worse than the usual comparison-scope gap: the authors evidently know the method, and included it in the v2 comparison, but left it out of the v1 table where its **93.5 PDMS would beat BrainWAM by 4.0**. No stated criterion excludes it — it is camera-based and NAVSIM-v1 is its headline benchmark.

3. **Three baseline rows are weaker configurations presented without qualification.** All three are verifiable against wiki records:
   - **DynVLA at 87.2** is DynVLA's *SFT-only* result (EMU3 + Dynamics CoT, no RFT). The submetrics match [[sources/dynvla.md]]'s SFT row exactly (98.6 / 95.3 / 95.5 / 100 / 80.6). Its published headline is **91.7** after RFT — a 4.5 PDMS understatement, and it would rank above BrainWAM.
   - **DriveVLA-W0 at 87.2** with submetrics 98.4 / 95.3 / 95.2 / 100 / 80.9 is the flow-matching **reimplementation** row from DriveLaW's table, which DriveLaW marks with a † and BrainWAM does not. Published values are 90.2★ (anchors) / 88.4 (single-sample).
   - **ReCogDrive at 86.5** is ReCogDrive-IL, the imitation-only variant; the RL version is 89.6 (90.8 in the camera-ready).

   None of these is a fabrication — each is a real number from a real configuration — but presenting all three unlabelled in a table where the paper's own result is 89.5 systematically flatters the comparison.

4. **The 0-step video ablation does not show what it is said to show.** Feeding pure Gaussian noise to a pathway trained on partially-denoised features is a distribution-shift test, not a test of whether predictive context helps. The −10.2 PDMS drop is uninformative about the counterfactual "a planner trained without the video stream." The informative rows are 1 vs. 2 vs. 3 steps, which show the marginal value of denoising is ~0.2 PDMS after the first step.

5. **Latency is 475–644 ms on an H20 and is acknowledged as not deployable.** Two large backbones (5B video + 4B VLM) stay resident at inference. This is better than [[sources/foresight.md]]'s 900 ms on the stronger H100 for a comparable score, but well behind [[sources/simwam.md]] (518 ms, 91.5) and everything non-generative.

6. **The video branch is never evaluated as a generator.** No FVD, no FID, no qualitative future frames. "Predictive dynamics" is established entirely by ablation, so there is no way to tell whether the WAM branch is forecasting anything or simply acting as a well-initialized visual encoder — the same ambiguity [[sources/drivelaw.md]] raised and [[sources/da-wam.md]] left open.

7. **Figure 2 is the paper's core evidence and is a single visualization.** No numeric attention ratios in the text, no variance across seeds or checkpoints, no layer indices on the claim "most layers," and no causal test — e.g., up-weighting VGM attention in Tri-MoT to see whether performance recovers. The mechanism story is plausible and well-argued from the literature, but the direct evidence is one figure and two indirect controls.

8. **Tri-MoT is the authors' own baseline, not a published method.** "Comparable parameter counts" is asserted, never tabulated. Since Tri-MoT is the foil the whole paper is built against, a weak instantiation would inflate every downstream claim, and there is no way to check it.

9. **The neuroscience framing does no work.** CAB is zero-init gated cross-attention (Flamingo, cited); CIF is a 2-layer Transformer plus element-wise mean. Nothing in the ablations tests a hemispheric prediction, and Tables 6–8 show both modules saturate at minimal depth — the mechanisms are generic and small. The analogy is presentation, not method.

10. **Single runs, no seed variance,** against deltas of +0.8 (CAB+CIF vs. CAB alone), +0.7 (freezing vs. full fine-tuning), +0.2 (Transformer vs. gated CIF), and +0.1 (2 vs. 3 CAB blocks).

11. **NAVSIM only.** No navhard, no Bench2Drive, no HUGSIM, no nuScenes, no reactive closed loop. For a paper whose qualitative claims center on *interactive negotiation*, the absence of any reactive benchmark is the same gap ForeSight has.

12. **No RL.** All three stages are supervised flow matching. Several methods above it in the wiki reach their scores partly through RFT.

13. **Pretraining-overlap is unaddressed** for both backbones. Wan2.2-TI2V-5B and Qwen3-VL-4B pretraining corpora versus OpenScene/nuPlan navtest is never discussed — the same systemic gap flagged for WA-JEPA, GeoWAM, and ForeSight.

14. **Protocol placement on v2 is undeterminable** {#protocol}. BrainWAM's table agrees with the wiki's **pre-fix** cohort on TransFuser (76.7), ARTEMIS (83.1), and DriveVLA-W0 (86.1), but disagrees on **HydraMDP++ (81.4 vs. 84.1)** and **DriveSuprim (83.1 vs. 87.1)** — and its DriveSuprim submetrics differ throughout, so that row is a different configuration rather than a recomputation. If the table is pre-fix, 89.6 would be the highest pre-fix EPDMS in the wiki, above WA-JEPA's 88.0. That reading cannot be confirmed. This is now the **third** v2 table provably mixing conventions, after DA-WAM's and GeoWAM's.

---

## Key Cross-References {#crossrefs}

- **The denoising-depth question, now 2–1**: three papers have measured how much denoising a planner's conditioning latent needs, and BrainWAM breaks the tie in DriveLaW's favor.

  | Paper | What it varies | Result |
  |---|---|---|
  | [[sources/drivelaw.md]] | Extraction point on a fixed schedule | **t=1 best** (89.1); t=10 collapses (23.2) |
  | [[sources/foresight.md]] | Total schedule length | **100 steps best** (89.3); 25 steps 88.0 |
  | **BrainWAM** | Video steps executed before caching | **1 step ≈ 3 steps** (89.3 / 89.4) |

  BrainWAM's axis is closest to DriveLaW's — how much denoising happens before the planner reads the features — and it lands in the same place: the first step carries essentially everything. ForeSight remains the outlier, and its ablation is the one that cannot be interpreted because it never reports its extraction step $t_{\rm d}$. See [[concepts/world-model-for-ad.md]].

- **The third vote, and a correction to the framing**: [[sources/adaptive-wam.md]] reaches the same practical conclusion from a third architecture — a single conditional forward is what the planner needs — but shows the variable Table 5 sweeps is not the one that matters. Holding depth fixed, the *noise index* is worth <=0.15 PDMS across five indices of a 40-step schedule; holding noise fixed, the *readout depth* is worth 4.80, with the mid-network block beating the full-depth exit. BrainWAM reads a fixed depth throughout and never reports which. Adaptive-WAM also plans in **170 ms on an A100** against BrainWAM's 475-644 ms on an H20, with a comparable score, largely by never running the denoiser more than once.
- **The modality-competition result**: [[concepts/mixture-of-experts.md]] — Tri-MoT is the wiki's first *negative* MoT result, and it bounds where joint attention works. [[sources/unidrivevla.md]]'s masked joint attention and [[sources/automot.md]]'s KV-cache coupling both succeed with asymmetric information flow; BrainWAM's symmetric three-way pool does not.

- **Complementarity done differently**: [[sources/hybriddriveVLA.md]] runs the same VLM-vs-visual-backbone complementarity question with CKA/CCA/SAE analysis and *set-level* failure statistics, then fuses by interpolation and scoring to reach 92.1 PDMS. BrainWAM has the better mechanism story and the weaker evidence; HybridDriveVLA has the better evidence and the higher score. Neither cites the other.

- **Freeze-then-coordinate**: [[concepts/dual-system-vla.md]] — Table 9 (frozen 89.5 vs. full fine-tuning 88.8) plus the convergence-rate data (VLA 54K steps, WAM 81K steps) is the wiki's clearest quantitative argument for freezing pretrained branches in a two-backbone planner, and it independently supports [[sources/foresight.md]]'s and [[sources/automot.md]]'s freezing choices.

- **Same video backbone, fourth design**: [[concepts/foundation-backbones-for-ad.md]] — Wan2.2-TI2V-5B now appears in [[sources/driveva.md]] (joint denoising, 90.9), [[sources/drivewam.md]] (chunked inverse dynamics, 90.1), [[sources/simwam.md]] (training-time only, 91.5), and BrainWAM (action-space coordination, 89.5). Four coupling strategies on one backbone is the closest thing the wiki has to a controlled comparison of how to attach a video prior to a planner.

- **Evaluator drift**: [[concepts/navsim-benchmark.md]] — BrainWAM supplies a published source for **ARTEMIS EC = 89.1**, a value the last lint flagged as unsourced and contradicted, and adds a third distinct DriveSuprim v2 EPDMS (83.1, after 87.1 and 86.0).
