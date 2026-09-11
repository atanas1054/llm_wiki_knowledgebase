---
title: Discrete Visual Tokenization for Driving
type: concept
sources: [raw/papers/Unified Driving Tokens_ Representation- and Geometry-Guided Discrete Tokenizer for Driving World Models and Planning.md, raw/papers/From Forecasting to Planning_ Policy World Model for Collaborative State-Action Prediction.md, raw/papers/ExploreVLA_ Dense World Modeling and Exploration for End-to-End Autonomous Driving.md, raw/papers/FutureSightDrive_ Thinking Visually with Spatio-Temporal CoT for Autonomous Driving.md, raw/papers/DriveVLA-W0_ World Models Amplify Data Scaling Law in Autonomous Driving.md, raw/papers/DynVLA_ Learning World Dynamics for Action Reasoning in Autonomous Driving.md]
related: [sources/unified-driving-tokens.md, sources/policy-world-model.md, sources/explorevla.md, sources/futuresightdrive.md, sources/drivevla-w0.md, sources/dynvla.md, sources/drivelaw.md, sources/reworld.md, sources/geoworldad.md, sources/geowam.md, sources/epona.md, sources/uniugp.md, sources/brainwam.md, concepts/world-model-for-ad.md, concepts/action-tokenization.md, concepts/foundation-backbones-for-ad.md, concepts/selection-based-planning.md, concepts/navsim-benchmark.md]
created: 2026-09-04
updated: 2026-09-04
confidence: medium
---

## What It Is

**Discrete visual tokenization** converts an image or video frame into a finite sequence of codebook indices, so that scene evolution can be modelled with next-token prediction. It is the counterpart to [[concepts/action-tokenization.md]] on the perception side: that page covers how *trajectories* become tokens, this one covers how *pixels* do.

The distinction matters because the two serve different masters. An action codebook exists so a language model can emit a plan. A visual codebook exists so a sequence model can predict the world — and, increasingly, so a planner can read it. **A driving visual tokenizer therefore has two consumers with different requirements**, and until [[sources/unified-driving-tokens.md]] no paper here treated the tension between them as a first-class problem.

This page is medium-confidence: it is assembled largely from tokenizer details recorded incidentally on source pages, and only one ingested paper studies tokenization as its subject.

---

## Why It Is a Distinct Design Axis

Most driving world models in this wiki are **continuous** — video DiTs ([[sources/drivelaw.md]], [[sources/simwam.md]], [[sources/adaptive-wam.md]]), JEPA latents ([[sources/wa-jepa.md]], [[sources/da-wam.md]]), geometry tokens ([[sources/geoworldad.md]]). Discrete tokenization buys three things they do not:

1. **Sequence-model compatibility.** Next-token prediction with cross-entropy, teacher forcing, and a fixed vocabulary — the whole autoregressive toolchain, including LLM backbones that already speak in tokens ([[sources/futuresightdrive.md]] simply *appends* image codes to a text vocabulary).
2. **A shared interface.** The same index sequence can be a world model's prediction target and a planner's input, which is the premise [[sources/unified-driving-tokens.md]] is built on.
3. **Compression with an explicit budget.** [[sources/policy-world-model.md]] compresses a frame to **28 tokens** precisely so that rolling out a future is cheap enough to sit inside a planning loop.

What it costs is fidelity and stability: quantization discards information, and codebooks collapse or go underused when the objectives pulling on them multiply.

---

## Tokenizers Recorded in This Wiki

| Paper | Tokenizer | Codebook | Notes |
|---|---|---|---|
| DrivingGPT *(via [[sources/unified-driving-tokens.md]]'s Table 1)* | **LlamaGen** | 16384 | Generic image tokenizer; 23.09 PSNR on NAVSIM — **3.4 below** a driving-tuned one |
| Orbis *(not ingested)* | – | 2 × 16384 | 25.94 PSNR; the only prior multi-codebook driving tokenizer here |
| [[sources/futuresightdrive.md]] | **MoVQGAN** VQ-VAE | vocabulary appended to the MLLM's | 128 × 192 generation; the sole architectural change is extending the text vocabulary, which activates image generation with ~0.3% of prior data |
| [[sources/explorevla.md]] | **MAGVIT-v2** (inside Show-o) | 8192 | 16 × 16 patches; masked-token loss over the image codebook |
| [[sources/drivevla-w0.md]] | **Emu3-8B** discrete visual tokens | – | The AR world-model variant; the diffusion variant uses continuous latents instead — the same paper implements both |
| [[sources/policy-world-model.md]] | Context-guided two-branch | 8192 (both branches) | Frozen high-res first-frame branch guides a low-res branch at **28 tokens/frame**; compression is the point |
| [[sources/dynvla.md]] | Dual VQ over *dynamics*, not appearance | **64 ego + 64 environment** | Tokenizes the *change* between adjacent frames; embedding dim 32 |
| [[sources/unified-driving-tokens.md]] | DINOv3-guided + geometry, MCB | 16384 or **4 × 4096** | The only one trained with planning-consumability as an explicit objective |

**Two of these are not appearance tokenizers at all**, and the difference is instructive. [[sources/dynvla.md]] quantizes *dynamics* into 128 total code types — three orders of magnitude smaller than an image codebook — because it only needs to name how the scene changed, not what it looks like. [[sources/policy-world-model.md]] keeps appearance but caps the budget at 28 tokens per frame. Both are evidence that **the vocabulary size a driving task needs is set by what the token must support downstream, not by reconstruction fidelity.**

---

## The Three-Way Requirement, and the Capacity Conflict {#capacity}

[[sources/unified-driving-tokens.md]] states the requirement as three-way — tokens should "preserve appearance for generation, encode semantics for scene understanding, and capture geometry/motion cues for planning" — and then measures what happens when a fixed bottleneck is asked for all three.

| Supervision | Depth AbsRel ↓ | $\Delta^{\cos}_{\mathrm{dec}}$ ↓ | PSNR ↑ |
|---|---:|---:|---:|
| RGB only (naive) | – | – | 25.96 |
| + DINOv3 feature decoding | – | **0.0346** | **26.51** |
| + adjacent-frame depth & pose, **same 16384 codebook** | 0.0640 | 0.0717 | **23.90** |
| + **multi-codebook** (4 × 4096) | **0.0556** | 0.0486 | 26.33 |

**Adding geometry to a fixed codebook costs 2.61 PSNR and doubles the semantic discrepancy** — the tokens stop carrying the DINO representation they were trained to carry, and appearance falls below even the RGB-only tokenizer. Multi-codebook quantization then improves *every column at once*, which is the signature of a capacity limit rather than an objective conflict, and is the paper's central argument.

**The residual matters and the paper does not flag it.** Semantic discrepancy at 4 × 4096 is 0.0486 against 0.0346 for two-way supervision at 1 × 16384 — still **40% worse**. Multi-codebook mitigates the conflict; on the evidence available it does not remove it, and no larger configuration is tried.

**Multi-codebook quantization** (adapted from UniTok) is the mechanism: an attention-based splitter produces $M$ head-specific vectors from one patch embedding, each quantized against its own codebook, then merged for a shared decoder. Capacity grows combinatorially ($4096^4$ vs $16384$) while the patch grid — the "tokenization resolution" — is unchanged. The cost is $M$ indices per patch, which multiplies an autoregressive world model's sequence length by $M$.

**Compare the continuous-side analogue.** [[sources/brainwam.md]] found that mixing a clean semantic stream with a denoising one in a shared attention pool *hurts* (87.8 against 88.1 for the WAM branch alone) and fixed it by **narrowing** the interface to 8 action tokens. UDT finds that mixing appearance, semantics, and geometry in a shared codebook hurts and fixes it by **widening** capacity. Opposite remedies, and the difference is diagnostic: BrainWAM's problem was gradient competition between streams of unequal maturity, UDT's is an information budget. Distinguishing the two before reaching for a remedy is the transferable lesson.

---

## Does Tokenizer Quality Reach the Planner? {#planning-readout}

This is the question the axis exists for, and there is now one controlled answer. [[sources/unified-driving-tokens.md]] freezes each tokenizer and trains the **same 20M trajectory head** on top:

| Tokenizer supervision | NC | DAC | TTC | EP | PDMS |
|---|---:|---:|---:|---:|---:|
| RGB reconstruction only | 98.2 | 94.8 | 94.6 | 77.8 | **85.5** |
| + DINOv3 representation | 98.7 | 97.4 | 96.3 | 82.1 | **89.4** |
| + geometry | 98.6 | 97.6 | 95.7 | 86.3 | **90.9** |
| + multi-codebook | 98.7 | 98.2 | 95.9 | 87.3 | **91.8** |

**+6.3 PDMS from tokenizer training objectives alone**, with planner capacity and training protocol fixed. Three readings worth keeping:

1. **Reconstruction-only tokenizers are genuinely bad for planning.** 85.5 is below DRAMA and roughly at 2024-era end-to-end baselines, from a pipeline whose ceiling is 91.8. The paper's opening complaint — that tokenizers inherited from image generation leave "a gap between what is easy to generate and what is useful to decode for driving decisions" — is its own best-supported claim.
2. **Semantic alignment is the largest single component (+3.9)** and it moves every sub-metric.
3. **Geometry supervision buys ego progress specifically** (+4.2 EP, TTC −0.6). That is the same signature [[sources/geoworldad.md]] measures for latent future-depth tokens (+3.3 EP, safety flat) and that [[sources/geowam.md]] argues for without ablating. Three papers, three mechanisms, one direction — see [[concepts/world-model-for-ad.md]].

**Two caveats.** The readout includes a PDM-score head and returns the argmax over multiple trajectories, so these are selection results and the scorer is retrained per tokenizer — part of each delta may be scorer quality. And rows 3–4 change $M$ and $K_m$ together.

**Set beside the wiki's other fixed-planner study**, [[sources/drivelaw.md]] swept representation *families* (video latents 89.1 > VLM hidden states 86.5 > BEV 84.1, a 5.0 spread). UDT sweeps only *supervision* within one family and gets 6.3. Together they say the representation is worth several PDMS either way, and that **how it is trained matters at least as much as which family it comes from**.

---

## Evaluating a Tokenizer

Pixel metrics are the default and they are insensitive to what planning needs. [[sources/unified-driving-tokens.md]] introduces a diagnostic worth adopting: measure reconstruction error **in a frozen foundation model's feature space** rather than in pixels.

- $\Delta^{\cos}_{\mathrm{img}}$, $\Delta^{\mathrm{rms}}_{\mathrm{img}}$ — distance between the *reconstructed* and *ground-truth* image once both are passed through frozen DINO. Captures whether vehicles, road structure, and their arrangement survived, and is "less sensitive to low-level appearance variations."
- $\Delta^{\cos}_{\mathrm{dec}}$, $\Delta^{\mathrm{rms}}_{\mathrm{dec}}$ — distance between the tokenizer's *decoded* feature head and the frozen target, i.e. how much of the representation the discrete code itself retained.

The two come apart from PSNR in exactly the case that matters: the +Geo row above holds depth accuracy while PSNR collapses and $\Delta^{\cos}_{\mathrm{dec}}$ doubles.

**What is still missing across the board**: no ingested paper reports **codebook utilization**, despite collapse and under-use being the stated motivation for multi-codebook designs. UDT adds dead-code reinitialization and an orthogonality regularizer and reports no utilization figure at all. Until someone does, the capacity argument rests entirely on downstream metrics.

---

## Open Questions

- **Can one tokenizer actually serve both consumers?** [[sources/unified-driving-tokens.md]] argues yes and then uses **different tokenizers** for planning and world modelling — geometry-enhanced for the first, representation-only for the second. The experiment its thesis needs (geometry-enhanced tokens into the autoregressive world model) is not run, and its Table 1 suggests why: the geometry variant is worse on every reconstruction metric.
- **How does $M$ trade against $K_m$?** Only 1 × 16384 and 4 × 4096 are compared, and Orbis's 2 × 16384 appears in a comparison table but not in the ablation.
- **Does discrete tokenization survive the frontier?** The best token-based planner here is 91.8, against 93.7 for continuous-latent methods. Whether the ~2-point gap is quantization loss, tokenizer immaturity, or unrelated is untested.
- **Is a frozen foundation encoder affordable at inference?** UDT's tokenizer requires a DINOv3-B forward pass to tokenize, and reports no latency. That cost is invisible in every table on this page.
- **Does anything beyond adjacent frames help?** UDT's geometry window is two frames, which is its own stated limitation.
