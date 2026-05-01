---
title: "OneVL: One-Step Latent Reasoning and Planning with Vision-Language Explanation"
type: source-summary
sources: [raw/papers/OneVL_ One-Step Latent Reasoning and Planning with Vision-Language Explanation.md]
related: [concepts/chain-of-thought-for-ad.md, concepts/world-model-for-ad.md, concepts/vlm-domain-adaptation.md, concepts/navsim-benchmark.md, concepts/foundation-backbones-for-ad.md, sources/adathinkdrive.md, sources/futuresightdrive.md, sources/dynvla.md, sources/drivevla-w0.md]
created: 2026-05-01
updated: 2026-05-01
confidence: high
---

# OneVL

**Paper**: OneVL: One-Step Latent Reasoning and Planning with Vision-Language Explanation  
**arXiv**: https://arxiv.org/html/2604.18486v1  
**Team**: Xiaomi Embodied Intelligence Team  
**Project**: https://Xiaomi-Embodied-Intelligence.github.io/OneVL

## Core Idea

OneVL argues that prior latent-CoT methods fail in autonomous driving because they compress language, not world dynamics. It replaces explicit autoregressive Chain-of-Thought with compact latent tokens, but supervises those tokens through two training-only auxiliary decoders:

- **Language auxiliary decoder**: reconstructs human-readable CoT from language latent tokens.
- **Visual auxiliary decoder**: predicts future-frame visual tokens at +0.5s and +1.0s, acting as a world-model objective.

At inference, both auxiliary decoders are discarded. Visual and language latent tokens are prefilled into the prompt context in one parallel pass, so the model keeps latent reasoning benefits while matching answer-only autoregressive latency.

## Figures and Available Images

![[x1 28.png|Figure 1: Accuracy and efficiency comparison across NAVSIM, ROADWork, Impromptu, and APR1. OneVL is the only latent-CoT method shown to beat explicit CoT while matching answer-only latency.]]

![[x2 26.png|Figure 2: Explicit CoT vs. prior implicit CoT vs. OneVL. OneVL uses visual latent tokens and language latent tokens supervised by dual auxiliary decoders during training.]]

![[x3 26.png|Figure 3: OneVL architecture. Qwen3-VL receives image/text inputs, visual latent tokens, language latent tokens, and trajectory answer tokens; auxiliary decoders train on future visual tokens and text CoT.]]

![[x4 24.png|Figure 4: NAVSIM qualitative example. OneVL improves the predicted trajectory and provides decoded future frames plus text reasoning.]]

![[x5 22.png|Extracted local analysis image: activated VQ-code dynamics diagnostic. Caption was not preserved in the raw markdown.]]

![[x6 19.png|Extracted local qualitative image: dynamic/visual CoT decoded future examples and BEV planning outcomes. Caption was not preserved in the raw markdown.]]

![[x7 15.png|Extracted local qualitative image: decoded future image/BEV compared with ground-truth future image/BEV. Caption was not preserved in the raw markdown.]]

![[x8 11.png|Extracted local qualitative image: counterfactual/additional dynamics examples. Caption was not preserved in the raw markdown.]]

![[x9 5.png|Extracted local qualitative image: w/o dynamic CoT vs. w/ dynamic CoT, current observation, and decoded future. Caption was not preserved in the raw markdown.]]

![[x10 6.png|Extracted local qualitative image: APR1-style planning result, current observation, decoded future, and ground-truth future. Caption was not preserved in the raw markdown.]]

**Note on source completeness**: the local raw markdown jumps from Table 1 into references, so the table rows below were recovered from the arXiv HTML source. The local asset directory contains 10 OneVL-related images; the arXiv HTML also references additional appendix qualitative figures whose local captions/assets were not fully preserved by the converter.

## Architecture

Backbone: **Qwen3-VL-4B-Instruct**, fully trainable in Stage 0 and Stage 2.

Inputs include a front-view image, ego state, command, and historical trajectory. The assistant response contains:

- 4 visual latent tokens, represented in implementation as 35 original-vocabulary tokens.
- 2 language latent tokens, represented as 20 original-vocabulary tokens.
- Trajectory answer tokens.

The visual auxiliary decoder uses current ViT embeddings plus visual latent hidden states:

$$
Z_v = [W_v(V), W_v(H_v)]
$$

It predicts discrete future-frame tokens using an IBQ / Emu3.5 tokenizer with a 131,072-code visual vocabulary. The language auxiliary decoder similarly receives current ViT embeddings plus language latent hidden states:

$$
Z_l = [W_l(V), W_l(H_l)]
$$

The combined objective is:

$$
L = L_c + \lambda_l L_l + \lambda_v L_v
$$

with language loss weight 1.0 and visual loss weight 0.1.

## Training Recipe

OneVL uses a preliminary visual-decoder pretraining step plus three main stages:

| Phase | Trainable components | Purpose |
| --- | --- | --- |
| Preliminary | Visual auxiliary decoder | Learn a future-frame prior from current ViT embeddings before latent conditioning |
| Stage 0 | Main VLM | Warm up latent tokens on trajectory prediction |
| Stage 1 | Language and visual auxiliary decoders | Freeze main VLM; align decoders to stable latent hidden states |
| Stage 2 | All components | Jointly fine-tune main VLM plus auxiliary decoders |

The staged curriculum is essential: direct joint training collapses to 67.13 PDMS, versus 88.84 with the full recipe.

## Main Results

### Table 1: NAVSIM

| Method | Model size | PDM-score | Latency (s) | Interpretability |
| --- | ---: | ---: | ---: | --- |
| AdaThinkDrive | 8B | 86.20* | - | Language |
| LaST-VLA | 8B | 87.30* | - | - |
| AR Answer | 4B | 87.47 | 4.49 | - |
| AR CoT+Answer | 4B | 88.29 | 6.58 | Language |
| COCONUT | 4B | 84.84 | 5.93 | - |
| CODI | 4B | 83.92 | 8.62 | - |
| SIM-CoT | 4B | 84.21 | 10.86 | Language |
| **OneVL** | **4B** | **88.84** | **4.46** | **Vision + Language** |

OneVL beats explicit AR CoT on NAVSIM while running at answer-only latency. It is strong as a supervised latent-CoT result, but it is not a current NAVSIM frontier score in the broader wiki because many RL, selection, and diffusion systems report above 90 PDMS.

### Table 2: ROADWork

| Method | ADE (px) | FDE (px) | Latency (s) | Interpretability |
| --- | ---: | ---: | ---: | --- |
| YNet | 22.68* | 80.78* | - | - |
| AR Answer | 15.98 | 40.29 | 4.74 | - |
| AR CoT+Answer | 13.18 | 29.98 | 10.74 | Language |
| COCONUT | 15.44 | 38.60 | 6.06 | - |
| CODI | 16.45 | 44.28 | 6.73 | - |
| SIM-CoT | 16.49 | 44.32 | 6.19 | Language |
| **OneVL** | **12.49** | **28.80** | **4.71** | **Vision + Language** |

### Table 3: Impromptu ADE/FDE

| Method | Model size | ADE (m) | FDE (m) | Latency (s) | Interpretability |
| --- | ---: | ---: | ---: | ---: | --- |
| Impromptu VLA | 3B | 1.60 | 4.28 | 6.10 | - |
| AR Answer | 4B | 1.46 | 4.03 | 4.24 | - |
| AR CoT+Answer | 4B | 1.42 | 3.96 | 6.84 | Language |
| COCONUT | 4B | 1.49 | 4.07 | 5.27 | - |
| CODI | 4B | 1.86 | 5.18 | 5.24 | - |
| SIM-CoT | 4B | 2.43 | 6.10 | 5.09 | Language |
| **OneVL** | **4B** | **1.34** | **3.70** | **4.02** | **Vision + Language** |

### Table 4: Impromptu Trajectory L2

| Method | 1s | 2s | 3s | 4s | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: |
| Impromptu VLA | 0.14 | 0.60 | 1.45 | 2.67 | 1.22 |
| AR Answer | 0.13 | 0.51 | 1.29 | 2.46 | 1.11 |
| AR CoT+Answer | 0.13 | 0.51 | 1.27 | 2.44 | 1.09 |
| COCONUT | 0.15 | 0.54 | 1.32 | 2.50 | 1.13 |
| CODI | 0.17 | 0.63 | 1.61 | 3.13 | 1.39 |
| SIM-CoT | 0.41 | 1.10 | 2.25 | 3.94 | 1.93 |
| **OneVL** | **0.13** | **0.48** | **1.18** | **2.25** | **1.01** |

### Table 5: APR1

| Method | Model size | ADE (m) | FDE (m) | Latency (s) | Interpretability |
| --- | ---: | ---: | ---: | ---: | --- |
| Cosmos-Reason | 10B | 2.86 | 7.42 | - | Language |
| AR Answer | 4B | 3.27 | 9.59 | 3.06 | - |
| AR CoT+Answer | 4B | 2.99 | 8.54 | 3.51 | Language |
| COCONUT | 4B | 3.29 | 9.48 | 3.76 | - |
| CODI | 4B | 3.22 | 9.25 | 3.85 | - |
| SIM-CoT | 4B | 3.40 | 9.85 | 3.78 | Language |
| **OneVL** | **4B** | **2.62** | **7.53** | **3.23** | **Vision + Language** |

OneVL has the best ADE in Table 5 but slightly worse FDE than Cosmos-Reason (7.53 vs. 7.42), which the paper attributes to Cosmos-Reason using RL.

## Explanation Quality and Ablations

### Table 6: Text CoT Quality on NAVSIM

| Method | Meta Action Acc. | STS | LLM Judge | Avg. |
| --- | ---: | ---: | ---: | ---: |
| AR CoT+Answer | 73.20 | 79.75 | 81.86 | 78.27 |
| SIM-CoT | 67.20 | 76.25 | 78.73 | 74.06 |
| **OneVL (language aux.)** | **71.00** | **78.26** | **79.13** | **76.13** |

OneVL does not match explicit AR CoT text quality, but it substantially improves over SIM-CoT while avoiding sequential CoT decoding.

### Table 7: Component Ablation

| Model variant | Language aux. dec. | Visual aux. dec. | Staged train | PDM-score |
| --- | --- | --- | --- | ---: |
| OneVL w/o visual decoder | Yes | No | Yes | 87.97 |
| OneVL w/o language decoder | No | Yes | Yes | 88.53 |
| OneVL w/o staged train | Yes | Yes | No | 67.13 |
| **OneVL** | **Yes** | **Yes** | **Yes** | **88.84** |

The visual decoder contributes +0.87 PDMS over language-only latent supervision, while the language decoder adds +0.31. The staged curriculum is the dominant stability requirement.

### Table 8: Deployment Tradeoff

| Variant | PDM-score | Latency (s) |
| --- | ---: | ---: |
| OneVL (regression / MLP head) | 86.83 | 0.24 |
| **OneVL (AR)** | **88.84** | **4.46** |

The MLP-head variant keeps latent supervision but replaces autoregressive waypoint generation with a feed-forward regression head. It loses 2.01 PDMS but reduces latency to 5.4% of the AR variant.

### Table 9: Training Hyperparameters

| Hyperparameter | Pre-training | Stage 0 | Stage 1 | Stage 2 |
| --- | --- | --- | --- | --- |
| Steps | Not rendered | - | - | - |
| Epochs | - | 2 | 1 | 5 |
| Batch (global) | 64 | 64 | 64 | 64 |
| Learning rate | Not rendered in HTML | Not rendered in HTML | Not rendered in HTML | Not rendered in HTML |
| LR schedule | Cosine | Cosine | Cosine | Cosine |
| Optimizer | AdamW | AdamW | AdamW | AdamW |
| Precision | BF16 | BF16 | BF16 | BF16 |
| Parallelism | ZeRO-2 | ZeRO-2 | ZeRO-2 | ZeRO-2 |
| Trainable | Visual aux decoder | ViT, LLM, aligner | Language and visual aux decoders | All |
| Frozen | - | - | Main VLM | - |
| Language loss weight | - | - | 1.0 | 1.0 |
| Visual loss weight | 1.0 | - | 0.1 | 0.1 |

### Table 10: Impromptu Reproduction Check

| Method | 1s | 2s | 3s | 4s | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: |
| Impromptu VLA* | 0.90 | 2.80 | 3.75 | 5.89 | 3.16 |
| Impromptu VLA | 0.14 | 0.60 | 1.45 | 2.67 | 1.22 |
| **OneVL** | **0.13** | **0.48** | **1.18** | **2.25** | **1.01** |

The paper notes that the author-provided Impromptu checkpoint and official training-script reproduction did not match the originally claimed performance, so it uses the stronger reproduced baseline for the main comparison.

## Relationships

- **vs. explicit CoT methods**: OneVL keeps explanation supervision but avoids sequential reasoning-token generation at inference.
- **vs. COCONUT/CODI/SIM-CoT**: prior latent-CoT methods compress linguistic reasoning only. OneVL adds future-frame supervision so latents must encode spatial-temporal dynamics.
- **vs. FutureSightDrive**: both use visual foresight as reasoning, but FSDrive generates a visual CoT at inference, while OneVL uses visual prediction only for training-time latent supervision and optional post-hoc explanation.
- **vs. DynVLA**: both compress world dynamics into a non-prose reasoning substrate. DynVLA generates dynamics tokens at inference; OneVL pre-fills latent tokens and trains them with auxiliary decoders.
- **vs. DriveVLA-W0 / FLARE**: all use future visual supervision as dense training signal. OneVL differs by explicitly pairing visual supervision with recoverable text explanations through dual auxiliary decoders.

## Limitations

- **Training memory**: the paper states training requires roughly three full 4B model instances in memory because the main VLM and two auxiliary decoders coexist.
- **Latent-token count is empirical**: no systematic sweep over visual/language latent token budget is provided.
- **AR trajectory decoding remains slow**: prefill removes latent-CoT overhead, but the main AR trajectory output still takes 4.46s on NAVSIM.
- **MLP deployment variant trades accuracy for speed**: 0.24s latency is practical, but PDMS falls from 88.84 to 86.83.
- **Single/front-view orientation**: the paper proposes extending the visual world-model decoder to multi-camera 360-degree future prediction as future work.
- **No NAVSIM-v2/navhard/Bench2Drive**: the paper is not directly comparable to recent EPDMS/OOD/closed-loop simulator leaders in the wiki.
