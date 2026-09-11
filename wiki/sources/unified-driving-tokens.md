---
title: "Unified Driving Tokens: Representation- and Geometry-Guided Discrete Tokenizer for Driving World Models and Planning"
type: source-summary
sources: [raw/papers/Unified Driving Tokens_ Representation- and Geometry-Guided Discrete Tokenizer for Driving World Models and Planning.md]
related: [concepts/visual-tokenization.md, concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/foundation-backbones-for-ad.md, concepts/selection-based-planning.md, concepts/action-tokenization.md, concepts/perception-for-planning.md, sources/drivelaw.md, sources/reworld.md, sources/geoworldad.md, sources/geowam.md, sources/policy-world-model.md, sources/explorevla.md, sources/futuresightdrive.md, sources/drivevla-w0.md, sources/dynvla.md, sources/epona.md, sources/drivesuprim.md, sources/diffusiondrive.md, sources/deepsight.md, sources/latent-wam.md, sources/adaptive-wam.md, sources/lwdrive.md, sources/simwam.md]
created: 2026-09-04
updated: 2026-09-04
confidence: high
---

**Paper**: Unified Driving Tokens: Representation- and Geometry-Guided Discrete Tokenizer for Driving World Models and Planning
**Authors**: Ziyang Yao, Zeyu Zhu, YunCheng Jiang, Zibin Guo, Huijing Zhao
**Orgs**: Peking University + Xiaomi EV
**arXiv**: 2606.01935v2

---

## Summary

This is the wiki's **first paper about a tokenizer rather than a planner or a world model**, and it is worth reading for that reason alone. Its premise: in token-based driving pipelines the discrete tokenizer is not a compressor but *the discrete language of the driving world* — simultaneously the prediction target for an autoregressive world model and the input representation a planner decodes. Yet nearly every driving tokenizer is inherited from image generation and optimized for pixel reconstruction, leaving "a gap between what is easy to generate and what is useful to decode for driving decisions."

The fix is three-part supervision on a VQ-VAE bottleneck:

| Component | Mechanism | Measured effect (PDMS, fixed 20M readout) |
|---|---|---|
| **Rep** | Frozen **DINOv3-B** features as encoder input *and* as a decoding target, alongside RGB + LPIPS + GAN | 85.5 → **89.4** (+3.9) |
| **Geo** | Adjacent-frame **depth + relative-pose** supervision through a VGGT-style temporal aggregator; training-time only | 89.4 → **90.9** (+1.5, almost all EP) |
| **MCB** | **Multi-codebook** quantization (4 × 4096 with an attention split-and-merge) instead of one 16384 book | 90.9 → **91.8** (+0.9) |

**91.8 PDMS on NAVSIM-v1 from a 20M trajectory head on frozen discrete tokens, single front camera.** That ties [[sources/wa-jepa.md]] and DriveFine\* in this wiki, above [[sources/drivevla-w0.md]] 90.2, [[sources/drivelaw.md]] 89.1, [[sources/policy-world-model.md]] 88.1, and [[sources/epona.md]] 86.2 — the four other frozen-representation planners in its own table.

**The three things this page takes from it:**

1. **A +6.3 PDMS spread from tokenizer training objectives alone, with the planner held fixed and deliberately tiny.** This is the wiki's cleanest isolation of representation quality from planner capacity, and it is a larger spread than [[sources/drivelaw.md]]'s cross-family sweep (video latents 89.1 > VLM hidden states 86.5 > BEV 84.1) despite varying only the supervision on one architecture.

2. **A directly measured capacity conflict at a discrete bottleneck.** Adding geometry supervision to a single 16384-entry codebook costs **2.61 PSNR** (26.51 → 23.90) and more than doubles the decoded-feature discrepancy. Multi-codebook quantization restores appearance almost fully — but the semantic discrepancy stays **40% worse** than the representation-only variant. The paper's conclusion that "the main challenge is not whether geometry supervision is compatible with representation alignment, but whether the discrete bottleneck has sufficient capacity" is supported by the first half of that and overstated by the second.

3. **The "unified interface" is not demonstrated — it is forked.** §4.1 states that planning uses the **geometry-enhanced** tokenizer and world modeling uses the **representation-guided** one. Two different tokenizers for the two tasks the paper's thesis says share a single token language, and Table 1 shows why: Ours-Rep beats Ours-Rep+Geo on *every* reconstruction metric. No experiment shows one tokenizer doing both jobs well. See [The Fork](#the-fork).

---

## Positioning

![[pic2.png|Overall framework: RGB detail branch plus frozen DINO features, VQ bottleneck with multi-codebook quantization, RGB and DINO decoders, cross-frame geometry branch, and the two downstream consumers]]

**Figure 1**: Subfigures (a), (b), (c) show the discrete tokenizer architecture; (d) and (e) show the two downstream consumption tasks — a token-based planning readout and a GPT-style next-token world model.

The argument is a criticism of a division of labour rather than of any specific method:

> "Existing driving token pipelines often optimize tokenizers primarily for pixel reconstruction and generation, and assess planning performance only after the fact, rather than treating **planning-consumability as a first-class objective during token learning**."

And the requirement it sets out is genuinely three-way, which is what makes the capacity result later meaningful: *"tokens should preserve appearance for generation, encode semantics for scene understanding, and capture geometry/motion cues for planning."* It also anticipates the cost — "richer alignment also stresses VQ quantization — leading to information loss and codebook instability (collapse and low utilization)" — and then measures it.

**Lineage.** The tokenizer side draws on UniTok (multi-codebook, unified generation+understanding) and TokenFlow; the geometry branch on **VGGT**; the depth pseudo-labels on **DVGT**, the predecessor of the DVGT-2 backbone that [[sources/geowam.md]] and [[sources/geoworldad.md]] both build on. With [[sources/drivelaw.md]] and [[sources/reworld.md]], this is the fourth recently-ingested paper from a **Xiaomi EV** collaboration, and it cites DriveLaW directly.

---

## Method

### Tokenizer architecture

An RGB frame is split into $P\times P$ patches giving $L=H_pW_p$ locations. A frozen $\Phi$ (DINOv3-B) supplies normalized patch features $\mathbf{F}_t=\Phi(\mathbf{I}_t)$, never back-propagated through.

**The encoder takes both**, which is the first design point worth noting — most representation-guided tokenizers align to a foundation model at the *output*; this one also feeds it in at the *input*, with a lightweight RGB branch alongside to "recover textures and boundaries weakened by high-level features and quantization":

$$\mathbf{R}_{t}=E_{\text{rgb}}(\mathbf{I}_{t}),\qquad \mathbf{X}_{t}=W_{f}[\mathbf{R}_{t};\mathbf{F}_{t}],\qquad \mathbf{H}_{t}=E_{\text{g}}(\mathbf{X}_{t})$$

with a pre-norm Transformer and RoPE over the patch grid. Quantization is hard nearest-neighbour against codebook $\mathcal{C}$:

$$k_{t,l}=\arg\min_{k}\|\mathbf{z}_{t,l}-\mathbf{c}_{k}\|_{2}^{2},\qquad \mathbf{e}_{t,l}=\mathbf{c}_{k_{t,l}}$$

**The decoder is dual-headed** — the second design point. A shared post-Transformer feeds two decoders, one to pixels and one back to DINO feature space:

$$\hat{\mathbf{I}}_{t}=D_{\text{img}}(\mathbf{S}_{t}),\qquad \hat{\mathbf{F}}_{t}=D_{\text{dino}}(\mathbf{S}_{t})$$

So the discrete code must be sufficient to reconstruct *both* the image and a strong semantic representation of it. That is what makes $\Delta^{\cos}_{\mathrm{dec}}$ a meaningful diagnostic later — it measures how much of the frozen representation survived quantization.

### Objectives

$$\mathcal{L}_{\text{tok}}=\lambda_{\text{rec}}\mathcal{L}_{\text{rec}}+\lambda_{\text{sem}}\mathcal{L}_{\text{sem}}+\lambda_{\text{gan}}\mathcal{L}_{\text{gan}}+\lambda_{\text{vq}}\mathcal{L}_{\text{vq}}$$

with $\mathcal{L}_{\text{rec}}=\|\hat{\mathbf{I}}_{t}-\mathbf{I}_{t}\|_{2}^{2}+\lambda_{\text{lpips}}\mathrm{LPIPS}(\hat{\mathbf{I}}_{t},\mathbf{I}_{t})$, $\mathcal{L}_{\text{sem}}$ a cosine + MSE term on DINO features, and a standard GAN discriminator. **None of the $\lambda$ values is reported anywhere in the paper.**

**Quantization is EMA-based, not gradient-based**, and the stated reason is directly about the multi-objective setting: *"This choice decouples codeword updates from the competing multiple objectives and empirically improves stability under joint supervision."* Straight-through estimator, commitment term, plus dead-code reinitialization and a weak orthogonality regularizer over active codewords.

### Cross-frame geometric supervision (training only)

A VGGT-style temporal aggregator $A_\psi$ alternates frame-wise and cross-frame attention over the post-quantization tokens of **two adjacent frames**, each carrying an appended ego token:

$$\{\mathbf{U}_{t},\bar{\mathbf{g}}_{t},\mathbf{U}_{t+1},\bar{\mathbf{g}}_{t+1}\}=A_{\psi}\!\left(\{[\mathbf{g}_{t};\tilde{\mathbf{H}}_{t}],[\mathbf{g}_{t+1};\tilde{\mathbf{H}}_{t+1}]\}\right)$$

Depth comes from a DPT-style head with a confidence map; relative pose is regressed from the two updated ego tokens and supervised with $\ell_1$ translation plus a **sign-invariant** quaternion loss (accounting for the double cover):

$$\mathcal{L}_{\text{pose}}=\left\|\hat{\mathbf{t}}-\mathbf{t}\right\|_{1}+\min\!\left(\left\|\hat{\mathbf{q}}-\mathbf{q}\right\|_{1},\left\|\hat{\mathbf{q}}+\mathbf{q}\right\|_{1}\right)$$

$$\mathcal{L}=\mathcal{L}_{\text{tok}}+\lambda_{\text{geo}}\left(\lambda_{\text{depth}}\mathcal{L}_{\text{depth}}+\lambda_{\text{pose}}\mathcal{L}_{\text{pose}}\right)$$

**Note what is being asked of the tokens.** The aggregator sits on *post-quantization* features, so the depth and pose heads can only use what survived the codebook. This is a stricter test than a geometry branch attached to the encoder, and it is what makes the depth numbers in Table 2 informative about the tokens rather than about the encoder.

Depth targets are radar-aligned pseudo-labels from **DVGT**, used only at tokenizer training and not required at inference.

### Multi-codebook quantization

Adapted from UniTok, and the motivation is stated as capacity competition: "the same set of patch tokens must preserve appearance details while also carrying semantic representation and geometric cues."

$$\mathbf{V}_{t,l}=P_{\text{attn}}(\mathbf{z}_{t,l})\in\mathbb{R}^{Md_{q}},\qquad \mathbf{e}_{t,l}=P_{\text{merge}}\!\left([\mathbf{e}^{(1)}_{t,l};\dots;\mathbf{e}^{(M)}_{t,l}]\right)$$

An attention-based splitter produces $M$ head-specific vectors, each quantized against its own codebook, then merged back to a single embedding for the shared decoder. **Tokenization resolution — the $H_p\times W_p$ patch grid — is unchanged**, so capacity grows combinatorially ($4096^4$ against $16384$) at fixed $L$.

**One consequence the paper does not draw out**: MCB yields $M$ discrete indices per patch, so an autoregressive next-token model over these tokens faces a **4× longer sequence** (or needs four prediction heads). That is the most economical explanation for why the world-model experiments use the single-codebook variant — see [The Fork](#the-fork).

### The two consumers

**Planning readout** (§3.3): tokenizer frozen. Patch features → learnable registers + small transformer → $R$ scene tokens; an 11-dim ego status vector is embedded, conditioned on the scene tokens by shallow attention, and an MLP head emits **multiple** trajectories under an L1 loss where **only the closest to ground truth is supervised**. A **score head predicts PDM-style metric outcomes**, supervised by running the rule-based evaluator on the predicted trajectory with BCE; at inference the highest-scoring trajectory is the plan. **20M parameters, identical across tokenizers.**

**World model** (§3.4): tokenizer frozen. GPT-style next-token Transformer, **1B parameters**, over patch tokens linearized in a fixed scan order and concatenated across a temporal window. Conditioning on actions and ego state via **AdaLN**. Teacher forcing, cross-entropy.

### Setup

| | |
|---|---|
| Data | Tokenizer + world model on **OpenScene** train; planning decoder on **NAVSIM** navtrain; all evaluation on NAVSIM test |
| Resolution | 288 × 512 |
| Tokenizer variants | Naive (RGB only, 16384) / +Rep (DINOv3-B, 16384) / +Rep+Geo+MCB (4 × 4096) |
| Planning readout | 20M, frozen tokens, front camera only |
| World model | 1B GPT, 3 history frames → 8 generated frames at 2 Hz |
| Depth labels | DVGT radar-aligned pseudo-labels, training only |

---

## Results

### Table 1 — Tokenizer comparison (NAVSIM test)

| Tokenizer | Codebook | rFID ↓ | PSNR ↑ | SSIM ↑ | $\Delta^{\cos}_{\mathrm{img}}$ ↓ | $\Delta^{\mathrm{rms}}_{\mathrm{img}}$ ↓ |
|---|---|---:|---:|---:|---:|---:|
| LlamaGen *(the DrivingGPT tokenizer)* | 16384 | 5.67 | 23.09 | 0.652 | 0.0869 | 0.167 |
| Orbis | 2 × 16384 | 5.53 | 25.94 | **0.773** | 0.0595 | 0.139 |
| Ours-Rep+Geo | 4 × 4096 | 5.14 | 26.33 | 0.769 | 0.0563 | 0.136 |
| **Ours-Rep** | **16384** | **4.15** | **26.51** | **0.774** | **0.0453** | **0.122** |

$\Delta_{\mathrm{img}}$ measures the distance between *reconstructed* and ground-truth images in frozen DINO feature space — a diagnostic the paper introduces to complement PSNR/SSIM, on the grounds that it "emphasizes object-level semantics and spatial layout … and is less sensitive to low-level appearance variations." **That is a genuinely useful metric to have named**, and this wiki has no other source of it.

**The row ordering is the finding, and it is inconvenient for the paper.** The best tokenizer on every reconstruction column is **Ours-Rep**, the variant *without* geometry — beating Ours-Rep+Geo by 0.99 rFID, 0.18 PSNR, and 24% relative on $\Delta^{\cos}_{\mathrm{img}}$. Geometry supervision costs reconstruction even after multi-codebook compensation.

**LlamaGen at 23.09 PSNR is the reference point that matters**, because it is what [[sources/policy-world-model.md]]-era token pipelines and DrivingGPT actually used. A 3.4 PSNR gap between a generic image tokenizer and a driving-tuned one is large, and it is the strongest support for the paper's opening complaint.

### Table 2 — Tokenizer roadmap

| Variant | AbsRel ↓ | $\delta_1$ ↑ | Trans (m) ↓ | Rot (°) ↓ | $\Delta^{\cos}_{\mathrm{dec}}$ ↓ | $\Delta^{\mathrm{rms}}_{\mathrm{dec}}$ ↓ | PSNR ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Naive | – | – | – | – | – | – | 25.96 |
| +Rep | – | – | – | – | **0.0346** | **0.105** | **26.51** |
| +Geo | 0.0640 | 0.959 | 0.717 | 2.02 | 0.0717 | 0.153 | **23.90** |
| +MCB | **0.0556** | **0.965** | **0.647** | **2.01** | 0.0486 | 0.124 | 26.33 |

**This is the most valuable table in the paper**, and the +Geo row is why. Adding depth and pose supervision to a *fixed* 16384-entry bottleneck:

- costs **2.61 PSNR** (26.51 → 23.90), dropping *below* the naive RGB-only tokenizer (25.96);
- **doubles** the decoded-feature discrepancy (0.0346 → 0.0717) — i.e. the tokens stop carrying the DINO representation they were explicitly trained to carry.

Multi-codebook quantization then recovers appearance almost entirely (26.33) and improves depth and pose *as well* (AbsRel 0.0640 → 0.0556, $\delta_1$ 0.959 → 0.965, Trans 0.717 → 0.647). **Everything improves at once**, which is exactly the signature of a capacity bottleneck rather than an objective conflict, and is strong support for the paper's reading.

**But the residual should be stated.** $\Delta^{\cos}_{\mathrm{dec}}$ at MCB is 0.0486 against 0.0346 for Rep-only — still **40% worse**. So multi-codebook mitigates the conflict; it does not dissolve it. On the evidence here, three-way supervision at 4 × 4096 still costs semantic fidelity relative to two-way supervision at 1 × 16384, and no larger MCB configuration is tried.

### Table 3 — NAVSIM test planning (PDMS)

| Method | Sensor | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| VADv2 | C | 97.2 | 89.1 | 91.6 | 100.0 | 76.0 | 80.9 |
| UniAD | C | 97.8 | 91.9 | 92.9 | 100.0 | 78.8 | 83.4 |
| Para-Drive | C | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| TransFuser | C+L | 97.7 | 92.8 | 92.8 | 100.0 | 79.2 | 84.0 |
| DRAMA | C+L | 98.0 | 93.1 | 94.8 | 100.0 | 80.1 | 85.5 |
| DiffusionDrive | C+L | 98.2 | 96.2 | 94.7 | 100.0 | 82.2 | 88.1 |
| WoTE | C+L | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| ResWorld | C | **98.9** | 96.5 | 95.6 | 100.0 | 83.1 | 89.0 |
| Hydra-MDP++ | C+L | 98.6 | **98.6** | 95.1 | 100.0 | 85.7 | 91.0 |
| **DriveSuprim** | C | 98.6 | **98.6** | 95.5 | 100.0 | **91.3** | **93.5** |
| Epona † | \*C | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 |
| PWM † | \*C | 98.6 | 95.9 | 95.4 | 100.0 | 81.8 | 88.1 |
| DriveLaW † | \*C | **99.0** | 97.1 | **96.7** | 100.0 | 81.3 | 89.1 |
| DriveVLA-W0 † | \*C | 98.7 | **99.1** | 95.3 | 99.3 | 83.3 | 90.2 |
| **Ours †** | \*C | 98.7 | 98.2 | 95.9 | 100.0 | 87.3 | **91.8** |

† = plans from visual tokens produced by a frozen tokenizer. \*C = single front view.

**Baseline hygiene is very good, and on one row it is better than any other recent ingest.** Every value matches this wiki's canonical record — VADv2 80.9, UniAD 83.4, Para-Drive 84.0, TransFuser 84.0, DRAMA 85.5, DiffusionDrive 88.1, WoTE 88.3, Epona 86.2, PWM 88.1, DriveLaW 89.1, DriveVLA-W0 90.2 (the anchor headline, correctly) — and **DriveSuprim appears at 93.5 with submetrics identical to its own published row** (98.6 / 98.6 / 95.5 / 100 / 91.3). [[sources/geoworldad.md]] used the widely-circulated 89.9 instead. This is the first ingested paper to cite DriveSuprim's ViT-L headline.

**The claim is scoped honestly as a result of that.** DriveSuprim sits *above* Ours in the same table, and the paper says only "the highest PDMS among methods that plan from frozen visual tokens," conceding explicitly that "several end-to-end systems benefit from multi-camera or LiDAR inputs, which makes direct comparison less controlled."

**Where 91.8 sits in the wiki**: tied with [[sources/wa-jepa.md]] 91.8 and DriveFine\* 91.8, below CLEAR/DA-WAM 93.7, DriveSuprim 93.5, [[sources/drive-jepa.md]] 93.3, WCog-VLA 92.9, [[sources/adaptive-wam.md]] (aux) 92.6, HybridDriveVLA 92.1, and [[sources/lwdrive.md]] 92.0. **EP 87.3 is the standout sub-score** — the joint-best in the wiki alongside ReCogDrive and LWDrive, and 5.5 above DriveLaW's 81.3 on the same single-camera input.

**One classification quibble.** The † group mixes discrete-token conditioning (Ours, and DriveVLA-W0's AR variant) with *continuous* latent conditioning ([[sources/drivelaw.md]] caches Video DiT activations; [[sources/epona.md]] is AR+diffusion). "Frozen tokenizer" is doing loose work, and the comparison is really "planners that consume a frozen upstream generative representation."

### Table 4 — Tokenizer ablation for planning

Same 20M decoder, same training protocol, only the tokenizer changes.

| ID | Rep | Geo | MCB | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|---|:-:|:-:|:-:|---:|---:|---:|---:|---:|---:|
| 1 | ✗ | ✗ | ✗ | 98.2 | 94.8 | 94.6 | 100.0 | 77.8 | **85.5** |
| 2 | ✓ | ✗ | ✗ | 98.7 | 97.4 | **96.3** | 100.0 | 82.1 | **89.4** |
| 3 | ✓ | ✓ | ✗ | 98.6 | 97.6 | 95.7 | 100.0 | 86.3 | **90.9** |
| 4 | ✓ | ✓ | ✓ | 98.7 | **98.2** | 95.9 | 100.0 | **87.3** | **91.8** |

**A +6.3 PDMS spread from tokenizer training objectives alone, with the planner fixed at 20M parameters.** This is the paper's best result and the reason it belongs in this wiki. Each component's signature is distinct:

- **Rep (+3.9)** is the largest and it moves everything — DAC +2.6, TTC +1.7, EP +4.3. Semantic alignment to DINOv3 is what turns a reconstruction tokenizer into a planning-consumable one.
- **Geo (+1.5) is almost entirely ego progress** — EP 82.1 → 86.3 (+4.2), with TTC *down* 0.6. That is the same signature [[sources/geoworldad.md]] reports for its future-geometry module (+3.3 EP, safety flat) and [[sources/geowam.md]] argues for rhetorically: **geometric supervision buys progress, not safety.** Two papers, two mechanisms, one direction.
- **MCB (+0.9)** shows up in DAC (+0.6) and EP (+1.0) — consistent with Table 2, where MCB recovers the appearance and semantic fidelity that Geo had cost.

**Compare the spread to the wiki's other fixed-planner representation study.** [[sources/drivelaw.md]] swept representation *families* under a fixed planner and found video latents 89.1 > VLM hidden states 86.5 > BEV 84.1, a 5.0 spread. UDT sweeps only the *supervision* on a single tokenizer and gets **6.3**. On this evidence, how a representation is trained matters at least as much as which family it comes from.

**Two caveats on reading it as pure representation quality.** The score head is retrained per tokenizer, so part of each delta may be scorer quality rather than token quality — the protocol is fixed so the comparison stays internally valid, but "differences primarily reflect the quality of the token representation" is slightly stronger than the design supports. And rows 3–4 differ in codebook configuration (1 × 16384 vs 4 × 4096), which changes both $M$ and $K_m$ together.

### World-model generation (Figure 4)

![[pic5.png|World-model generation comparison against the baseline, and a qualitative rollout with decoded DINO features alongside generated frames]]

**Figure 4**: (a) generation performance versus the baseline; (b) qualitative rollout. Protocol follows DrivingGPT — condition on 3 past frames, generate 8 future frames at 2 Hz, 288 × 512, NAVSIM test split.

**No numbers appear in the body for either FID or FVD**, and the baseline is never named. The entire quantitative content of one of the paper's two headline claims is:

> "the world model trained on our tokens achieves better FID and FVD than the baseline under the matched setting."

The qualitative point is more interesting than the missing scalars: because the tokenizer has a DINO decoder, **the same predicted tokens can be decoded into both future images and future DINO features**, giving a structural-consistency view of a rollout that pixel metrics do not provide. That is a real capability and no other world model in this wiki has it.

---

## The Fork {#the-fork}

The paper's thesis is one token language for two consumers. Its experimental protocol uses two:

> "For planning, we freeze the tokenizer and train the same lightweight (20M parameters) trajectory readout head on token features, using the **geometry-enhanced** tokenizer by default in our main planning comparisons. For world modeling, we train a GPT-style next-token Transformer (1B parameters) …, using the **semantic representation-guided** tokenizer by default."

So the 91.8 PDMS comes from Rep+Geo+MCB (4 × 4096) and the generation result comes from Rep (1 × 16384). **No experiment anywhere shows a single tokenizer serving both tasks**, and the tables explain why nobody tried: Table 1 has Ours-Rep beating Ours-Rep+Geo on all five reconstruction metrics, and Table 2 has +Geo dropping PSNR below even the naive tokenizer before MCB rescues it.

**Two unstated reasons the fork is probably necessary, both inferable from the paper's own design:**

1. **Sequence length.** MCB emits $M=4$ indices per patch. An autoregressive world model over those tokens has a 4× longer stream at the same spatial resolution, or needs four heads and a factorized output distribution. Neither is discussed, and the 1B GPT is described as operating on $L$ tokens per frame.
2. **Residual semantic cost.** Even after MCB, $\Delta^{\cos}_{\mathrm{dec}}$ is 40% worse than Rep-only. Generation quality plausibly tracks that.

**What this costs the contribution, and what it does not.** It does not damage the planning result: +6.3 PDMS from tokenizer supervision under a fixed decoder stands on its own. It does damage the framing. "A shared token interface for both token-based world modeling and planning consumption" is the paper's stated contribution (1) and (4), and the honest version of what is shown is *narrower and still interesting*: **the same tokenizer architecture and training recipe serve both tasks, with the codebook configuration and geometry branch tuned per task.** The unified-language claim needs one more experiment — Rep+Geo+MCB tokens fed to the 1B world model — and the paper has both artifacts already.

---

## Qualitative Results

![[pic3.png|Tokenizer reconstruction comparison across variants, with DINO features visualized by PCA]]

**Figure 2**: (a) and (b) show the semantic representation-guided tokenizer; (c) adds geometry enhancement with multi-codebook design. DINO features are visualized via PCA.

![[pic4.png|Two planning cases where the naive tokenizer fails and the Rep+Geo tokenizer succeeds]]

**Figure 3**: Two cases where the naive tokenizer fails and the Rep+Geo tokenizer succeeds.

---

## Limitations

1. **The unified-interface claim is not tested.** Planning and world modeling use *different* tokenizers (§4.1), and Table 1 shows the two configurations trade off against each other. The single missing run — geometry-enhanced tokens into the 1B world model — is the one the thesis needs. See [The Fork](#the-fork).

2. **The generation result has no numbers.** "Better FID and FVD than the baseline" is the whole of it; values are in a figure, the baseline is unnamed, and no table exists. One of two headline claims is therefore unauditable and cannot be entered on this wiki's generation-quality tables.

3. **Codebook utilization is the central motivation and is never measured.** The introduction blames "codebook instability (collapse and low utilization)" for the problem; §3.2 adds dead-code reinitialization and an orthogonality regularizer "to improve utilization"; §3.2 says split-and-merge with EMA was "helpful for balancing codebook utilization." **No utilization figure appears anywhere**, so the mechanism that the whole MCB argument rests on is supported only by downstream metrics.

4. **No ablation of $M$ or $K_m$.** Only 1 × 16384 versus 4 × 4096 is shown, and those differ in both. Whether the gain comes from more books, smaller books, the attention splitter, or simply a larger product space is unresolved — and 2 × 8192 (Orbis's shape) is in the comparison table but not in the ablation.

5. **The residual semantic cost is not acknowledged.** MCB restores PSNR to 26.33 but leaves $\Delta^{\cos}_{\mathrm{dec}}$ at 0.0486 against Rep-only's 0.0346. The prose says MCB "largely restores reconstruction quality"; it does not note that semantic fidelity stays 40% worse, which is the quantity the paper's own thesis says matters most for planning.

6. **The planning readout selects with a PDM-score head.** Trajectories are generated multiply, scored by a head trained on rule-based evaluator outputs, and the argmax is returned — so **91.8 is a selection result, not a single-trajectory one**, and it belongs with the simulator-distilled cohort on [[concepts/selection-based-planning.md]]. The number of trajectories $R$ and the scene-register count are unreported.

7. **No hyperparameters.** Not one of $\lambda_{\text{rec}}$, $\lambda_{\text{sem}}$, $\lambda_{\text{gan}}$, $\lambda_{\text{vq}}$, $\lambda_{\text{lpips}}$, $\lambda_{\text{geo}}$, $\lambda_{\text{depth}}$, $\lambda_{\text{pose}}$, or $\lambda_c$ is given; the depth loss form is deferred to implementation details, and the supplementary is not part of this clipping. $d_q=64$, $K=16384$, and $M=4$ are the only quantization numbers stated.

8. **No latency, throughput, or parameter count for the tokenizer.** The two consumers are sized (20M, 1B) but the tokenizer — a Transformer encoder, post-encoder, two decoders, discriminator, and a frozen DINOv3-B in the input path — is not. A frozen DINOv3-B forward is required at *inference* to tokenize, which is a real deployment cost the paper never quantifies.

9. **Geometry supervision needs DVGT pseudo-depth**, radar-aligned and post-processed. Training-time only and clearly disclosed, but it makes the recipe dependent on an external geometry model, unlike a purely self-supervised tokenizer.

10. **Adjacent frames only.** The temporal window for geometry is two frames, which is the paper's own stated limitation ("extending geometric supervision beyond adjacent pairs"). Depth and pose are evaluated at that horizon; nothing tests whether the tokens carry longer-range motion structure.

11. **No NAVSIM-v2, no navhard, no reactive protocol, no nuScenes.** Single runs, no seed variance, against ablation deltas of 0.9 on the smallest component.

12. **The `Ours-Rep+Geo` row in Table 1 and the `+MCB` row in Table 2 are the same model** but reported under two names across tables, which makes the roadmap harder to follow than it needs to be.

---

## Key Cross-References

- **First tokenizer paper here, and the reason for a new page**: [[concepts/visual-tokenization.md]] collects the discrete visual tokenizers previously scattered across source pages — LlamaGen in DrivingGPT, MoVQGAN in [[sources/futuresightdrive.md]], MAGVIT-v2 in [[sources/explorevla.md]], Emu3 in [[sources/drivevla-w0.md]], the 28-token context-guided tokenizer in [[sources/policy-world-model.md]], the dual VQ dynamics codebooks in [[sources/dynvla.md]] — none of which had a home.
- **Representation quality isolated from planner capacity**: +6.3 PDMS under a fixed 20M head, against [[sources/drivelaw.md]]'s 5.0 spread across representation *families* under a fixed planner. Together they are the wiki's two controlled measurements of what a representation is worth, and they agree that it is worth several PDMS. See [[concepts/foundation-backbones-for-ad.md]].
- **Geometry buys progress, not safety — third instance**: Geo is worth +4.2 EP with TTC down 0.6. [[sources/geoworldad.md]]'s latent future geometry gave +3.3 EP with safety flat; [[sources/geowam.md]] argues the coordinate-frame case for geometry without ablating it. Three papers, one direction. See [[concepts/world-model-for-ad.md]].
- **A capacity conflict measured at a bottleneck**: the +Geo row (−2.61 PSNR, doubled semantic discrepancy) and its MCB repair is the wiki's only direct measurement of multi-objective competition inside a *discrete* representation. The analogous continuous-side result is [[sources/brainwam.md]]'s modality competition, which is fixed by narrowing the interface rather than widening the codebook — opposite remedies for related problems.
- **A new reconstruction diagnostic**: $\Delta^{\cos}_{\mathrm{img}}$ / $\Delta^{\mathrm{rms}}_{\mathrm{img}}$, the distance between reconstructed and ground-truth images in frozen DINO space. Worth adopting when comparing tokenizers, since PSNR/SSIM are insensitive to exactly the object-level structure planning needs.
- **Simulator-distilled selection again**: the 20M readout scores its own trajectories with a PDM-metric head and returns the argmax, putting 91.8 in the same class as [[sources/drivesuprim.md]], [[sources/drive-jepa.md]], [[sources/geoworldad.md]], and [[sources/lwdrive.md]]. See [[concepts/selection-based-planning.md]].
- **The Xiaomi EV cluster**: [[sources/drivelaw.md]], [[sources/reworld.md]], [[sources/geoworldad.md]], and this paper are all Xiaomi EV collaborations ingested within days of each other, and UDT cites DriveLaW. The DVGT lineage connects it to [[sources/geowam.md]] as well.
- **New methods for the gap list**: ResWorld (2602.10884, 89.0 PDMS), Orbis (2507.13162, a 2 × 16384 driving tokenizer), DVGT (2512.16919, the depth-pseudo-label source and DVGT-2's predecessor), UniTok, TokenFlow, LlamaGen, and DrivingGPT's tokenizer configuration.
