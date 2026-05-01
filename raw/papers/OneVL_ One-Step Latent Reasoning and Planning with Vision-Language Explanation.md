---
title: "OneVL: One-Step Latent Reasoning and Planning with Vision-Language Explanation"
source: "https://arxiv.org/html/2604.18486v1"
author:
published:
created: 2026-05-01
description:
tags:
  - "clippings"
---
See the [Contributions and Acknowledgments](#S7 "7 Contributions and Acknowledgments ‣ Future Directions ‣ 6 Conclusion ‣ Why Prior Latent CoT Methods Fail on Autonomous Driving ‣ 5.7 In-Depth Analysis: Where Does the Benefit Come From? ‣ 5.6 Towards Real-World Deployment ‣ On the efficacy of three-stage training ‣ 5.5 Ablation Study ‣ Text CoT Quality ‣ 5.4 Explanation Quality ‣ Explicit CoT supervision helps: AR CoT+Answer vs. AR Answer ‣ Existing latent CoT methods fail on VLA-based autonomous driving ‣ Prefill inference matches answer-only prediction speed ‣ OneVL achieves best performance ‣ 5.3 Main Results ‣ 5 Experiments ‣ OneVL: One-Step Latent Reasoning and Planning with Vision-Language Explanation") section for a list of contributors.

Xiaomi Embodied Intelligence Team

###### Abstract

Chain-of-Thought (CoT) reasoning has become a powerful driver of trajectory prediction in VLA-based autonomous driving, yet its autoregressive nature imposes a latency cost that is prohibitive for real-time deployment. Latent CoT methods attempt to close this gap by compressing reasoning into continuous hidden states, but consistently fall short of their explicit counterparts. We suggest that this is due to purely linguistic latent representations compressing a symbolic abstraction of the world, rather than the causal dynamics that actually govern driving. Thus, we present OneVL (One-step latent reasoning and planning with Vision-Language explanations), a unified VLA and World Model framework that routes reasoning through compact latent tokens supervised by dual auxiliary decoders. Alongside a language decoder that reconstructs text CoT, we introduce a visual world model decoder that predicts future-frame tokens, forcing the latent space to internalize the causal dynamics of road geometry, agent motion, and environmental change. A three-stage training pipeline progressively aligns these latents with trajectory, language, and visual objectives, ensuring stable joint optimization. At inference, the auxiliary decoders are discarded and all latent tokens are prefilled in a single parallel pass, matching the speed of answer-only prediction. Across four benchmarks, OneVL becomes the first latent CoT method to surpass explicit CoT, delivering state-of-the-art accuracy at answer-only latency, and providing direct evidence that tighter compression, when guided in both language and world-model supervision, produces more generalizable representations than verbose token-by-token reasoning.  
Project Page: [https://Xiaomi-Embodied-Intelligence.github.io/OneVL](https://xiaomi-embodied-intelligence.github.io/OneVL)

![[x1 28.png|Refer to caption]]

Figure 1: Accuracy and efficiency comparison across four benchmarks. Existing latent CoT methods underperform explicit CoT. OneVL is the first to surpass it while matching answer-only prediction latency.

## 1 Introduction

Vision-Language Models (VLMs) [^49] [^4] [^6] [^5] [^104] [^66] [^67] [^1] [^34] [^36] [^37] [^97] [^79] [^15] [^10] [^100] [^48] have rapidly become a foundational building block for autonomous driving, unifying holistic scene understanding, natural language reasoning, and end-to-end trajectory planning within a single model [^44] [^56] [^89] [^113] [^103] [^83] [^64] [^114] [^42] [^46]. When further extended to produce action outputs, such as trajectory waypoints or control signals, these models are known as Vision-Language-Action models (VLAs) [^13] [^77] [^17] [^62] [^41] [^29] [^30] [^47] [^102].

A central driver of recent progress in VLA-based driving is Chain-of-Thought (CoT) reasoning [^78] [^115] [^77] [^17], where the model articulates intermediate reasoning steps before committing to a final trajectory, yielding substantial gains in prediction quality [^44] [^105]. By explicitly surfacing scene semantics, anticipated agent behaviors, and high-level driving intent, CoT supervision binds predictions into coherent causal chains and markedly reduces planning errors. This success echoes a broader body of LLM CoT research spanning mathematical reasoning [^65], document understanding [^75] [^72] [^74] [^26] [^73] [^27], code synthesis [^14], multimodal QA [^76] [^92], RL-based deep reasoning [^36] [^52], and test-time scaling [^51] [^90]. A unifying explanation for why CoT works comes from the *compression view of intelligence* [^58] [^20]: under next-token supervision, a model forced to articulate intermediate steps must compress its understanding into structured, generalizable representations rather than memorize shallow input–output mappings.

Yet deploying CoT in real driving systems exposes a sharp tension between interpretability and efficiency. Standard autoregressive (AR) CoT generation must emit every reasoning token before the trajectory can be produced. This yields inference latency proportional to the chain length, which is far above that of answer-only prediction. In safety-critical real-time settings, this gap is prohibitive. At the same time, explicit CoT chains are strikingly redundant; for example, much of the sequence merely restates context or follows formulaic patterns. This redundancy suggests that the essential reasoning content can be compressed into a far more compact form [^96] without sacrificing and even strengthening generalization, since tighter compression forces the model to retain only the causal structure that truly matters for prediction.

##### Latent CoT and Its Limitations

A growing line of work pursues exactly this direction, replacing explicit reasoning tokens with compact latent representations. COCONUT [^40] introduced curriculum learning over latent thought tokens, progressively replacing discrete reasoning steps with continuous vectors. CODI [^87] extended this with self-distillation, training a student to mimic a teacher’s CoT behavior in latent space. SIM-CoT [^106] attached a separate text-decoding auxiliary decoder to enable direct text supervision during latent training. However, adapting these methods to VLA-based driving reveals critical shortcomings. COCONUT, CODI, and SIM-CoT were designed for language-only reasoning and make no use of the rich visual structure that defines driving scenes. As a result, their purely linguistic latents prove insufficient for the multimodal reasoning demanded by trajectory prediction, and as Figure 1 shows, every existing latent CoT method underperforms explicit CoT across all benchmarks.

More fundamentally, natural language descriptions of driving scenes are inherently abstract. They encode semantic labels rather than the spatiotemporal causal dynamics that actually determine future outcomes. A latent vector that compresses language is therefore compressing a symbolic abstraction of the world, not its underlying causal structure. A further limitation is that in prior methods, latent hidden states are still produced autoregressively (one latent hidden state at a time), leaving inference sequential. We instead aim to generate all latent hidden states in a single step via prefill. Figure 2 contrasts these three paradigms, that is, explicit CoT, prior implicit latent CoT, OneVL, and motivates our design.

![[x2 26.png|Refer to caption]]

Figure 2: Comparison of three CoT paradigms. (a) Explicit CoT: the model generates a full chain of discrete reasoning tokens before the answer. (b) Implicit CoT: reasoning is compressed into a small number of opaque latent vectors 𝒵 \\mathcal{Z}. (c) OneVL (Ours): two types of latent tokens ( v \\mathcal{Z}\_{v}, red ) and language ( l \\mathcal{Z}\_{l} salmon ); during training, dual auxiliary decoders decode these into future-frame visual tokens and CoT text, respectively, providing rich text and world model supervision. During inference, the decoders are discarded, and the latent tokens are prefilled into the prompt context, matching the speed of answer-only prediction while keeping the interpretability of (a) in both vision and language.

##### OneVL: One-Step Latent Reasoning and Planning with Vision-Language Explanations

We present OneVL, a framework that overcomes the limitations of prior latent CoT methods through two key innovations. First, we introduce dual-modal auxiliary decoders: a *language auxiliary decoder* that reconstructs human-readable CoT reasoning from compact language latent tokens, and a *visual auxiliary decoder* that predicts anticipated future frames [^18] [^88] [^70] from visual latent representations. The visual decoder plays the role of a *world model* auxiliary. By forcing the compressed latents to anticipate what the scene will look like at future time steps, it ensures that the bottleneck encodes genuinely causal scene dynamics, such as agent trajectories, road geometry evolution, and emerging hazards, rather than abstract symbolic summaries. This is precisely the missing ingredient in language-only latent CoT. Future-frame prediction is a concrete compression target that directly reflects the causal structure of the physical world, satisfying the compression view of intelligence in a way that text descriptions alone cannot. The resulting framework simultaneously handles planning, language reasoning, and visual interpretation within a single model.

Beyond interpretability, the dual reconstruction objectives serve a deeper role: they ensure that the compressed latents encode genuinely generalizable structure rather than superficial correlations [^20] [^95]. If compact latent tokens can be decoded into both coherent language reasoning and plausible future frames, the model has necessarily discovered transferable representations of scene dynamics rather than memorized input-output mappings. Critically, the world model supervision (visual decoder) and the language supervision act as complementary forms of validation. Language grounds the latents in semantic intent, while visual prediction grounds them in physical scene dynamics. Together, they guarantee that the compressed representation satisfies both the semantic and causal requirements of robust trajectory planning.

Second, we design a *prefill inference* mechanism. At inference time, the latent tokens (both visual and language) are prefilled into the model’s context as fixed prompt inputs, enabling single-pass generation of all latent tokens. This eliminates the iterative latent token generation overhead and achieves inference speed essentially identical to answer-only AR prediction. The resulting model performs one-step latent reasoning (fast inference), vision-language explanation (interpretable reasoning), and finally planning in a unified sequence. Empirically, OneVL not only matches but surpasses explicit AR CoT in trajectory quality, demonstrating that compression, far from being a necessary compromise, is itself a driver of more effective reasoning [^58] [^20].

##### Contributions

The key contributions of this work are summarized as follows:

- OneVL Framework: We introduce a latent CoT framework built on the principle that compression drives generalization, and identify a critical gap in prior latent CoT work, that is, purely linguistic latent representations are too abstract to satisfy this principle for planning tasks. We address this with dual-modal auxiliary decoders, a language decoder, and a visual world model decoder that jointly supervise compact latent tokens to encode both linguistic reasoning and future scene dynamics. The world model decoder provides the concrete, causal compression target that language alone cannot supply. A principled three-stage training pipeline progressively aligns the latent bottleneck with trajectory prediction, ensuring that the compressed representations capture causal structure rather than memorized patterns.
- Superior performance across Four Benchmarks: OneVL achieves superior performance across many benchmarks. Notably, OneVL is the only latent CoT method that outperforms explicit autoregressive CoT, directly supporting our hypothesis that tighter compression encourages more generalizable reasoning. Ablation studies confirm each component’s contribution. Both the visual and language decoders yield consistent performance gains, and the staged training recipe is essential.
- Prefill Inference: At inference time, the auxiliary decoders are discarded, and all latent tokens are prefilled into the prompt, enabling single-pass latent CoT reasoning with no iterative overhead. For example, on NAVSIM, the latency matches AR answer-only prediction and is $1.5\times$ faster than explicit autoregressive CoT. On ROADWork, prefill latency is identical to answer-only and $2.3\times$ faster than its explicit counterpart. For real-world deployment, appending an MLP head for producing trajectory further reduces latency to $0.24$ s $(4.16$ Hz), just $5.4\%$ of the AR model’s latency, offering a practical deployment option.
- Interpretable Explanations: The language auxiliary decoder recovers high-quality CoT text from compressed latents, while the visual auxiliary decoder generates spatially coherent future-frame previews, providing both linguistic and visual interpretability.

##### Organization

The remainder of this paper is organized as follows. Section˜2 presents related work. Section˜3 describes the OneVL architecture in detail, including the main VLM, latent token design, and auxiliary decoders. Section˜4 elaborates on the three-stage training pipeline and the motivation for each stage. Section˜5 presents the experimental setup, main results, and ablation studies. Section˜6 concludes with a discussion of future directions.

## 2 Related Work

### 2.1 Implicit and Latent Chain-of-Thought

Since explicit CoT incurs inference overhead proportional to the length of the reasoning chain, a growing line of work internalizes reasoning into continuous latent representations [^21] [^16] [^60] [^91] [^112] [^59]. [^21] proposed a step-by-step internalization curriculum that progressively replaces each explicit reasoning token with implicit internal computation, training the model to absorb CoT one step at a time. COCONUT [^40] generalizes this idea to continuous latent thought tokens through a staged curriculum, enabling breadth-first-like exploration of solution paths entirely within the LLM’s hidden state space. Compressed Chain of Thought [^16] takes a complementary distillation approach, condensing an explicit CoT trace into a small set of dense summary vectors prepended to the input, achieving substantial length reduction at only modest accuracy cost. CODI [^87] adopts sequence-level self-distillation, training a student model to align its anchor latent hidden state, typically the final hidden representation before the answer, with the teacher model’s full chain-of-thought sequence, narrowing the performance gap while preserving efficiency. Token Assorted [^91] offers a flexible middle ground by interleaving discrete text tokens and continuous latent tokens within the same sequence, interpolating between fully explicit and fully implicit reasoning. SIM-CoT [^106] identifies a latent instability problem, where representations collapse as the number of latent tokens grows without per-step supervision, and addresses it with a plug-and-play auxiliary decoder that aligns each implicit token with its corresponding explicit reasoning step at training time only.

As discussed, all of these methods were developed for language-only tasks and do not transfer effectively to VLA-based autonomous driving.

### 2.2 VLM and VLA for Autonomous Driving

Beyond the foundational VLM works and CoT-augmented driving models discussed in Section˜1, a parallel line of research has focused on establishing richer evaluation and supervision signals for language-grounded driving [^108] [^44]. MapLM [^9] introduced a large-scale benchmark specifically targeting map and traffic scene understanding, probing whether VLMs can parse structured road topology from sensor data. [^22] augmented VLMs with bird’s-eye-view feature injection, enabling holistic scene understanding that fuses camera and top-down spatial context within a single multimodal model. Complementary efforts have explored corner-case evaluation [^12] and risk localization [^81] to stress-test VLM reasoning under rare or safety-critical scenarios.

Closer to trajectory prediction, recent VLA models pair language reasoning with waypoint or action outputs [^69] [^68] [^116]. DriveVLA-W0 [^61] employs world modeling to generate dense self-supervised signals that amplify data scaling laws in VLA-based driving. AdaThinkDrive [^78] introduces adaptive CoT for driving decisions, LaST-VLA [^77] trains a large vision-language-action model on driving data, and Alpamayo-R1 [^105] explicitly bridges reasoning traces with long-tail action prediction. OneVL builds on these foundations by addressing the latency cost of explicit CoT through dual-modal latent supervision and prefill inference, delivering competitive performance without sacrificing interpretability.

### 2.3 World Modeling for Autonomous Driving

The concept of the *world model* originates from model-based reinforcement learning, where it seeks to emulate human cognitive processes and predict the effects of actions on environmental evolution [^39] [^38] [^54], particularly in 3D and 4D spaces [^56] [^7] [^63] [^111] [^55] [^57]. To further enhance spatial reasoning, several approaches incorporate advanced perception frameworks [^101] [^117] [^118] [^109] [^110] to achieve a more robust understanding of 3D environments. With advances in video generation and the introduction of the Joint Embedding Predictive Architecture by [^2], the scope and applications of world models have broadened considerably [^43] [^84] [^93] [^24] [^45] [^23]. In autonomous driving, world models are typically applied to three ends: data generation, closed-loop evaluation, and representation learning [^28] [^35] [^56] [^64] [^94].

For data generation, Cosmos [^3] integrates multimodal inputs such as text, images, videos, and motion signals to synthesize consistent data for training robotic and autonomous driving systems. For closed-loop evaluation, DICC [^32] leverages generative world models to produce realistic driving images and performs adversarial evaluation on end-to-end driving systems to improve safety and robustness. Similarly, AD-R1 [^114] takes advantage of the high physical fidelity of world models, employing them as interactive simulators for reinforcement learning and thereby reducing safety violations in challenging scenarios. For representation learning, DriveVLA-W0 [^61] incorporates future temporal information from a world model to improve trajectory planning, and DynVLA [^86] reduces redundancy in generated images by modeling inter-frame similarity, achieving lower inference latency while maintaining competitive performance.

In contrast, OneVL uses short-horizon future visual token prediction as a training-only world model auxiliary paired with compressed latent CoT inside a single VLA. This auxiliary guides the bottleneck toward causal scene dynamics precisely where language-only implicit CoT is insufficient (Section˜1), and is then discarded at inference so that prefilled latents yield answer-only latency. Prior work emphasizes data generation, simulators, or separate representation stacks, rather than this joint certification of language and visual latent bottlenecks.

## 3 Model Architecture

OneVL augments a pretrained VLM with a compact latent token interface and dual auxiliary decoders for multimodal explanation. Figure 3 gives a complete overview. We describe each component in detail below.

![[x3 26.png|Refer to caption]]

Figure 3: OneVL architecture. An image and structured text prompt (ego state, command, historical trajectory) are fed into the VLM. The output hidden states shown below the VLM, contains image tokens ( 𝒯 v \\mathcal{T}\_{v} ), text tokens ( l \\mathcal{T}\_{l} ), visual latent tokens ( 𝒵 \\mathcal{Z}\_{v} ), language latent tokens ( \\mathcal{Z}\_{l} ), and trajectory answer tokens ( ^ y \\hat{\\mathcal{T}}\_{y} ). During training, hidden states ℋ \\mathcal{H}\_{v} and \\mathcal{H}\_{l} at the latent positions are routed to two auxiliary decoders: the Visual Aux. Decoder (left) directly predicts future-frame visual tokens at 0.5 s 0.5\\mathrm{s} 1.0 1.0\\mathrm{s} ( ℒ \\mathcal{L}\_{v} ), and the Language Aux. Decoder (right) predicts chain-of-thought reasoning ( \\mathcal{L}\_{l} ). During inference, both decoders are discarded; latent tokens are prefilled into the prompt, matching the answer-only AR prediction latency.

### 3.1 Main Vision-Language Model

The backbone of OneVL is Qwen3-VL-4B-Instruct [^5], a VLM that processes interleaved image and text inputs. The model consists of three standard components: Vision Encoder (ViT), Visual Projector (MLP Aligner), and Large Language Model (LLM). All three components are initialized from the Qwen3-VL-4B-Instruct checkpoint and remain fully trainable in Stages 0 (Section˜4.2) and 2 (Section˜4.4). The backbone is primarily optimized via a standard next-token prediction objective, applying a cross-entropy loss ($\mathcal{L}_{c}$) to both the trajectory answers and the latent reasoning tokens introduced below.

### 3.2 Latent Token Design

A critical design decision in OneVL is the introduction of specialized latent tokens that serve as compact carriers of implicit reasoning. We define two classes of latent tokens:

Language Latent Tokens ($\langle|$ latent $|\rangle$): A fixed-length sequence of $\mathcal{C}_{t}=2$ language latent tokens, framed by start and end delimiters ($\langle|$ start-latent $|\rangle$ and $\langle|$ end-latent $|\rangle$). These tokens are placed in the assistant response before the trajectory answer, occupying the position where explicit CoT reasoning would appear in standard AR models. The hidden states extracted at these token positions after LLM processing encode the model’s implicit language-grounded reasoning.

Visual Latent Tokens ($\langle|$ latent-vis $|\rangle$): A fixed-length sequence of $\mathcal{C}_{v}=4$ visual latent tokens, similarly delimited (i.e., $\langle|$ start-latent-vis $|\rangle$ and $\langle|$ end-latent-vis $|\rangle$), placed before the language latent tokens in the response. These tokens are designed to encode spatial and temporal visual reasoning about the future scene state.

Both sets of latent tokens serve as reasoning carriers whose hidden states at the VLM output layer are fed into auxiliary decoders. Note that during implementation, we found that adding dedicated special tokens (e.g., <|latent-vis|>) to the VLM vocabulary causes performance degradation. Instead, we represent latent tokens using the original vocabulary. Concretely, the $\mathcal{C}_{v}$ visual latent tokens are realized as 35 tokens, and the $\mathcal{C}_{t}$ language latent tokens are realized as 20 tokens.

### 3.3 Language Auxiliary Decoder

The language auxiliary decoder $\mathcal{D}_{l}$ aims to recover human-readable CoT reasoning text from the compact language latent hidden states.

Input Construction. For each training sample, the language latent tokens in the main model produce hidden states $\mathcal{H}_{l}\in\mathbb{R}^{\mathcal{C}_{t}\times d}$, where $d$ is the LLM hidden dimension. We additionally supply the current-frame ViT patch embeddings $\mathcal{V}\in\mathbb{R}^{N_{v}\times d}$ from the backbone, where $N_{v}$ denotes the length of vision tokens after ViT embedding. An MLP layer maps both branches into the auxiliary decoder’s embedding space; we then form the multimodal input by concatenation:

$$
\mathcal{Z}_{l}=\Big[\text{W}_{l}(\mathcal{V}),\;\text{W}_{l}(\mathcal{H}_{l})\Big],
$$

where $\text{W}_{l}$ is MLP (with dimensions chosen so that $\mathcal{Z}_{l}$ matches the LLM input dimension). The tensor $\mathcal{Z}_{l}$ is fed into $\mathcal{D}_{l}$.

Training Objective. The language auxiliary decoder is trained to predict the ground-truth CoT reasoning text $\mathcal{T}_{{y}_{{}_{t}}}$ given $\mathcal{Z}_{l}$, that is:

$$
\mathcal{L}_{l}=-\sum_{i=1}^{|\mathcal{T}_{{y}_{{}_{t}}}|}\log P_{\mathcal{D}_{l}}\left(\mathcal{T}_{{y}_{{}_{t}},i}\mid\mathcal{Z}_{l},\mathcal{T}_{{y}_{{}_{t}},<i}\right)\penalty 10000\ .
$$

This cross-entropy loss encourages the main model’s language latent tokens to encode semantically rich information about the driving scene that is decodable as natural language reasoning.

### 3.4 Visual Auxiliary Decoder

The visual auxiliary decoder $\mathcal{D}_{v}$ aims to predict anticipated future-frame visual tokens.

Motivation. Autonomous driving is inherently a spatial-temporal prediction task. Future frame visual tokens, which represent what the driving scene will look like at near-term horizons, are a natural target for learning the visual latent representations. This visual prediction objective serves as a world model auxiliary, supplementing language-only latent CoT. This task acts as a rigorous test of generalization, as predicting unseen configurations requires a robust causal model rather than pattern memorization. By combining visual and language decoders, the framework supervises latents in both physical dynamics and semantic intent, imposing a multi-modal constraint that captures the shared causal structure of the environment.

Input Construction. Let $\mathcal{V}\in\mathbb{R}^{N_{v}\times d}$ denote the ViT embeddings from the current frame (extracted from the main model’s visual encoder), and let $\mathcal{H}_{v}\in\mathbb{R}^{\mathcal{C}_{v}\times d}$ denote the visual latent token hidden states from the main model. Let $\text{W}_{v}\in\mathbb{R}^{d\times d}$ be the MLP layer. The visual auxiliary decoder receives the concatenation:

$$
\mathcal{Z}_{v}=\Big[\text{W}_{v}(\mathcal{V}),\;\text{W}_{v}(\mathcal{H}_{v})\Big]\penalty 10000\ .
$$

This conditioning on both the current visual context and the latent state allows the decoder to perform conditioned future-frame prediction.

Visual Tokenizer and Vocabulary Extension. To represent images as discrete token sequences, we adopt the IBQ (Index Backpropagation Quantization) visual tokenizer [^88]. We use the Emu3.5 tokenizer [^18] [^88] with a codebook of 131,072 discrete visual codes. The images are resized to a maximum resolution of 512x512 pixels. To integrate this visual vocabulary into OneVL, the Qwen3-VL-4B base vocabulary is extended by 131,072 additional visual token IDs. The visual token sequences for training are constructed offline by running the IBQ tokenizer over the ground-truth future frames from the dataset, requiring no additional forward passes during training.

Training Objective. Let $\mathcal{T}_{y_{v}}=\left[\mathcal{T}_{y_{v},1},\,\mathcal{T}_{y_{v},2}\right]$ be the concatenated discrete visual token sequence for the future frames at time steps $\mathcal{T}_{y_{v},1}$ ($+0.5\text{s}$) and $\mathcal{T}_{y_{v},2}$ ($+1.0\text{s}$). The visual loss is:

$$
\mathcal{L}_{v}=-\sum_{t=1}^{|\mathcal{T}_{y_{v}}|}\log P_{\mathcal{D}_{v}}\!\left(\mathcal{T}_{y_{v},t}\mid\mathcal{Z}_{v},\mathcal{T}_{y_{v},<t}\right)\penalty 10000\ .
$$

### 3.5 Combined Training Objective

The total training loss $\mathcal{L}$ is a weighted sum of three components:

$$
\mathcal{L}=\mathcal{L}_{c}+\lambda_{l}\mathcal{L}_{l}+\lambda_{v}\mathcal{L}_{v}\penalty 10000\ ,
$$

where $\mathcal{L}_{c}$ is the main model’s cross-entropy loss, $\lambda_{l}=1.0$ is the language explanation loss weight, and $\lambda_{v}=0.1$ is the visual explanation loss weight. The lower weight on $\mathcal{L}_{v}$ reflects that visual token reconstruction is a harder task, and a smaller weight prevents it from dominating the training signal.

### 3.6 Prefill Inference

At inference time, the auxiliary decoders are discarded. The key efficiency insight is that the latent tokens, both visual and language, can be prefilled into the prompt context as fixed token sequences, because their specific vocabulary identities have been seen by the model during training.

Concretely, the inference prompt is constructed as:

$$
\resizebox{393.61993pt}{}{$\displaystyle\footnotesize[\text{System},\,\text{User query},\,\langle|\texttt{start-latent-vis}|\rangle,\,\underbrace{\langle|\texttt{latent-vis}|\rangle\cdots}_{\mathcal{C}_{v}},\,\langle|\texttt{end-latent-vis}|\rangle,\,\langle|\texttt{start-latent}|\rangle,\,\underbrace{\langle|\texttt{latent}|\rangle\cdots}_{\mathcal{C}_{t}},\,\langle|\texttt{end-latent}|\rangle]$}\penalty 10000\ .
$$

All latent tokens are included in the *prefill* phase rather than the *decode* phase. Since modern transformers [^99] process the entire prefill in parallel, these additional tokens add negligible overhead compared to sequential autoregressive generation. The model then generates only the trajectory tokens autoregressively. This yields inference latency nearly identical to answer-only AR prediction, while the main model’s processing of the prefilled latent tokens still implicitly activates the reasoning pathways learned during training. To conclude, the model outputs:

- Trajectory prediction: The primary output—future waypoints for autonomous driving.
- Language explanation (optional, via aux decoder): Human-readable CoT reasoning describing the model’s interpretation of the scene and its driving decision rationale.
- Visual explanation (optional, via visual aux decoder): future frame visual tokens, providing a spatial preview of the predicted scene evolution.

Items 2 and 3 are available during post-hoc explanation generation (e.g., for human-in-the-loop [^71] debugging, safety auditing, or human-robot interaction), while item 1 is always generated during inference.

## 4 Three-Stage Training Pipeline

Training OneVL presents a unique optimization challenge. The main VLM, the language auxiliary decoder, and the visual auxiliary decoder must all be jointly optimized, yet they have fundamentally different learning objectives and start from different relative states of alignment.

We address this challenge through a training pipeline consisting of a preliminary self-supervised pretraining step followed by three main stages, each with a clear purpose. The configuration is summarized in Table 9.

### 4.1 Preliminary: Visual Auxiliary Decoder Self-Supervised Pretraining

##### Motivation

Before integrating the visual auxiliary decoder into the full OneVL pipeline, we first pretrain it independently as a future-frame generator. The intuition is straightforward. Asking the decoder to immediately predict future frames conditioned on latent tokens that carry no information yet (early in training) is an ill-posed task that impedes learning. Instead, we first train the decoder with a strong unconditional prior—given the current-frame ViT embeddings, predict what the scene will look like at next two timestamps—before introducing the latent conditioning signal. This preliminary stage is conceptually analogous to self-supervised video prediction where the decoder learns purely from visual observations without any reasoning supervision.

##### Training Objective

The visual auxiliary decoder $\mathcal{D}_{v}$ receives only the current-frame ViT embeddings $\mathcal{V}$ (projected via $\text{Proj}_{v}$) as input—the visual latent token hidden states $\mathcal{H}_{v}$ are absent at this stage (the main model is not yet connected). Using the same concatenated target $\mathcal{T}_{y_{v}}=\left[\mathcal{T}_{y_{v},1},\,\mathcal{T}_{y_{v},2}\right]$ defined in Section˜3.4, the pretraining loss is:

$$
\mathcal{L}_{p}=-\sum_{t=1}^{|\mathcal{T}_{y_{v}}|}\log P_{\mathcal{D}_{v}}\!\left(\mathcal{T}_{y_{v},t}\mid\mathcal{V},\,\mathcal{T}_{y_{v},<t}\right)\penalty 10000\ .
$$

##### From Unconditioned to Action-Conditioned Generation World Model

After pretraining, the decoder has learned a robust prior for visual dynamics, enabling it to predict plausible future frames from the current scene alone. This component functions as the model’s implicit world model, capturing the underlying rules of visual evolution. When it is subsequently connected to the main model, the visual latent tokens $\mathcal{H}^{v}$ are introduced as an additional conditioning signal alongside the ViT embeddings. Since these latent tokens encode the driving agent’s planned action—derived from the main model’s reasoning—the decoder effectively transitions from unconditioned next-frame generation to action-conditioned rollouts of the world model. This framing provides a principled interpretation where the visual latent tokens serve as a compact, actionable representation that steers the world model’s predictions.

### 4.2 Stage 0: Main Model Warmup

##### Motivation

The fundamental prerequisite for auxiliary decoders to provide meaningful supervision is that the main model’s latent tokens (i.e., <|latent-vis|> or <|latent|>) carry information that is semantically aligned with reasoning content. Without targeted warmup, these tokens would not produce meaningful hidden states that auxiliary decoders can decode.

Stage 0 addresses this by training the main VLM end-to-end on the trajectory prediction task, with latent tokens embedded in each training sample’s assistant response. The model learns to:

- Predict accurate trajectories: The main CE loss $\mathcal{L}_{\text{CE}}$ over the latent tokens and trajectory answer tokens ensures the model develops strong base prediction capability.
- Develop meaningful latent representations: By contextualizing the latent tokens within the prompt-response structure alongside trajectory targets, the model naturally learns to use the latent positions to encode intermediate representations useful for trajectory prediction. Besides, the attention mechanism allows trajectory tokens to attend to latent token positions, establishing the information routing pathways that the auxiliary decoders will later exploit.

### 4.3 Stage 1: Auxiliary Decoder Warmup

##### Motivation

With the main model producing stable, meaningful latent representations (as established in Stage 0), Stage 1 focuses exclusively on training the auxiliary decoders to align with these representations. Crucially, we freeze the main model during this stage, ensuring that the auxiliary decoders optimize against a consistent semantic distribution. By maintaining this stability, the decoders can more effectively internalize the mapping from fixed latent features to visual and language reasoning. To conclude, Stage 1 trains:

- Language auxiliary decoder $\mathcal{D}_{l}$: Trained to decode the language CoT reasoning text and fine-tuned with $\mathcal{L}_{l}$ against the ground-truth reasoning annotations.
- Visual auxiliary decoder $\mathcal{D}_{v}$: Trained to predict two future frames with $\mathcal{L}_{v}$.

### 4.4 Stage 2: Joint End-to-End Fine-tuning

##### Motivation

Finally, Stage 2 jointly fine-tunes all three model components with the combined loss $\mathcal{L}$ (Eq. 5). The gradients from $\mathcal{L}_{l}$ and $\mathcal{L}_{v}$ now flow back into the main model, directly shaping the latent representations to simultaneously serve trajectory prediction, language explanation, and visual prediction objectives. This creates a virtuous cycle:

- The richer latent representations enable the main model to make better trajectory predictions (as the latent tokens carry more useful intermediate representations).
- The auxiliary decoders adapt to the updated latent representations, improving their explanation quality.

This joint optimization is possible in Stage 2 precisely because both the main model and the auxiliary decoders are already well-initialized. Ablation studies (see Section˜5.5) confirm that skipping three-stage training leads to degraded performance.

## 5 Experiments

In this section, we present a comprehensive quantitative evaluation of OneVL across four benchmarks, followed by ablation studies and analyses that isolate the source of each performance gain.

### 5.1 Datasets

We evaluate OneVL on four complementary benchmarks: NAVSIM [^19], ROADWork [^33], Impromptu [^17], and APR1. These datasets are chosen because they have been shown to be effective in settings where CoT reasoning is employed, or because they provide sufficient labels to construct explicit reasoning traces [^78] [^77] [^17].

##### CoT Annotation Construction

A key challenge in training OneVL is obtaining high-quality chain-of-thought reasoning annotations paired with each driving scenario. On NAVSIM, we leverage the CoT annotations released by AdaThinkDrive [^78], the previous state-of-the-art method. These annotations provide natural-language reasoning traces that cover scene interpretation (such as lane boundaries), critical object analysis (including vehicles and pedestrians), and the final driving intent. They are synthesized by a VLM that converts raw detection labels (e.g., objects and lanes) into reasoning sequences, and serve as the supervision target for the language auxiliary decoder ($\mathcal{L}_{\text{l}}$). On ROADWork, we construct CoT annotations using a similar in-house pipeline. On Impromptu, we build CoT annotations from the original dataset’s Q&A pairs, augmenting the data with explicit decision and root-cause labels for corner-case trajectory prediction. On APR1, we use the released checkpoint to predict the CoC labels for all training examples. These annotations supervise the language auxiliary decoder, enabling the model to learn robust reasoning and decision-making logic in unstructured driving scenarios. Further details are provided in Appendix 8.3.

### 5.2 Experimental Setups

##### Evaluation Metrics

On NAVSIM, all methods are evaluated using the Predictive Driver Model (PDM) score, a composite metric that jointly assesses trajectory safety, comfort, and progress. On ROADWork, we report ADE (Average Displacement Error) and FDE (Final Displacement Error) to measure waypoint accuracy. On Impromptu, in addition to ADE and FDE, we also report the trajectory prediction L2 error over the first four seconds, following the protocol of the original paper. On APR1, we report ADE and FDE as well. Across all methods, we additionally report the average inference latency.

##### Baselines

We compare OneVL against two categories of baselines, all built on Qwen3-VL-4B-Instruct [^5], together with previous state-of-the-art methods as stronger reference points.

The *AR-based methods* that use standard autoregressive generation are:

- AR Answer: Direct autoregressive trajectory prediction without any reasoning. The model receives the front-view image and ego state and directly outputs trajectory waypoints. This is the fastest baseline and defines the latency lower bound.
- AR CoT+Answer: Standard CoT reasoning followed by trajectory prediction. The model first generates a full reasoning chain and then produces the trajectory. This represents the performance upper bound for explicit reasoning, at the cost of substantially higher latency.

The *Latent CoT methods* that use continuous latent representations for implicit reasoning are:

- COCONUT [^40]: Adapted for VLA-based autonomous driving. It uses curriculum learning to replace discrete reasoning tokens with continuous latent vectors.
- CODI [^87]: A COCONUT variant based on self-distillation, where a teacher model provides full textual CoT supervision and the student reasons in latent space.
- SIM-CoT [^106]: A CODI variant that adds a separate text-decoding auxiliary decoder for language interpretability.

We also compare against previous *state-of-the-art methods* reported in the literature. On NAVSIM (*supervised fine-tuning setting*), these methods are:

- AdaThinkDrive [^78]: An 8B-parameter model with adaptive CoT reasoning for autonomous driving.
- LaST-VLA [^77]: An 8B-parameter vision-language-action model for autonomous driving.

On ROADWork, we compare against YNet [^33] and on Impromptu, we compare against the Impromptu VLA [^17]. For Impromptu VLA, we report the result from our own replication. We also include a result obtained using the provided model checkpoint, which is shown in Table 10. On APR1, we compare against the Cosmos-Reason, a flow-matching based VLA model. AR-based baselines are trained for 2 epochs with a learning rate of $4\times 10^{-5}$ and batch size 64. Latent CoT baselines are trained for 6 epochs with a learning rate of $4\times 10^{-5}$ and batch size 64. The results of previous state-of-the-art methods, where not explicitly stated otherwise, are taken directly from the literature.

### 5.3 Main Results

![[x4 24.png|Refer to caption]]

Table 1: Performance comparisons on the NAVSIM benchmark. PDM-score (higher is better) and average inference latency (lower is better) for all methods. ∗ indicates the result is derived from the corresponding paper. For OneVL, we only count the parameters of the main VLM, as the auxiliary decoders are discarded during inference. The same applies to all subsequent models.

[^1]: Anthropic. Claude 3.7 Sonnet and Claude Code. [https://www.anthropic.com/news/claude-3-7-sonnet](https://www.anthropic.com/news/claude-3-7-sonnet), 2025.

[^2]: Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 15619–15629, 2023.

[^3]: Alisson Azzolini, Junjie Bai, Hannah Brandon, Jiaxin Cao, Prithvijit Chattopadhyay, Huayu Chen, Jinju Chu, Yin Cui, Jenna Diamond, Yifan Ding, Liang Feng, Francesco Ferroni, Rama Govindaraju, Jinwei Gu, Siddharth Gururani, Imad El Hanafi, Zekun Hao, Jacob Huffman, Jingyi Jin, Brendan Johnson, Rizwan Khan, George Kurian, Elena Lantz, Nayeon Lee, Zhaoshuo Li, Xuan Li, Maosheng Liao, Tsung-Yi Lin, Yen-Chen Lin, Ming-Yu Liu, Xiangyu Lu, Alice Luo, Andrew Mathau, Yun Ni, Lindsey Pavao, Wei Ping, David W. Romero, Misha Smelyanskiy, Shuran Song, Lyne Tchapmi, Andrew Z. Wang, Boxin Wang, Haoxiang Wang, Fangyin Wei, Jiashu Xu, Yao Xu, Dinghao Yang, Xiaodong Yang, Zhuolin Yang, Jingxu Zhang, Xiaohui Zeng, and Zhe Zhang. Cosmos-Reason1: From physical common sense to embodied reasoning. *arXiv preprint arXiv:2503.15558*, 2025.

[^4]: Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-VL: A versatile vision-language model for understanding, localization, text reading, and beyond. *arXiv preprint arXiv:2308.12966*, 2023.

[^5]: Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-VL technical report. *arXiv preprint arXiv:2511.21631*, 2025a.

[^6]: Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-VL technical report. *arXiv preprint arXiv:2502.13923*, 2025b.

[^7]: Hengwei Bian, Lingdong Kong, Haozhe Xie, Liang Pan, Yu Qiao, and Ziwei Liu. DynamicCity: Large-scale 4D occupancy generation from dynamic scenes. In *International Conference on Learning Representations*, 2025.

[^8]: Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuScenes: A multimodal dataset for autonomous driving. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 11621–11631, 2020.

[^9]: Xu Cao, Tong Zhou, Yunsheng Ma, Wenqian Ye, Can Cui, Kun Tang, Zhipeng Cao, Kaizhao Liang, Ziran Wang, James M. Rehg, and Chao Zheng. MAPLM: A real-world large-scale vision-language benchmark for map and traffic scene understanding. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 21819–21830, 2024.

[^10]: Jiahong Chen, Jing Wang, Long Chen, Chuwei Cai, and Jinghui Lu. NanoVLA: Routing decoupled vision-language understanding for nano-sized generalist robotic policies. *arXiv preprint arXiv:2510.25122*, 2025a.

[^11]: Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. M3-Embedding: Multi-linguality, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. *arXiv preprint arXiv:2402.03216*, 2024a.

[^12]: Kai Chen, Yanze Li, Wenhua Zhang, Yanxin Liu, Pengxiang Li, Ruiyuan Gao, Lanqing Hong, Meng Tian, Xinhai Zhao, Zhenguo Li, Dit-Yan Yeung, Huchuan Lu, and Xu Jia. Automated evaluation of large vision-language models on self-driving corner cases. In *IEEE/CVF Winter Conference on Applications of Computer Vision*, pages 7817–7826, 2025b.

[^13]: Long Chen, Oleg Sinavski, Jan Hünermann, Alice Karnsund, Andrew James Willmott, Danny Birch, Daniel Maund, and Jamie Shotton. Driving with llms: Fusing object-level vector modality for explainable autonomous driving. In *IEEE International Conference on Robotics and Automation*, 2024b.

[^14]: Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. *arXiv preprint arXiv:2107.03374*, 2021.

[^15]: Qimao Chen, Fang Li, Shaoqing Xu, Zhiyi Lai, Zixun Xie, Yuechen Luo, Shengyin Jiang, Hanbing Li, Long Chen, Bing Wang, Yi Zhang, and Zhi-Xin Yang. VILTA: A VLM-in-the-loop adversary for enhancing driving policy robustness. *arXiv preprint arXiv:2601.12672*, 2026.

[^16]: Jeffrey Cheng and Benjamin Van Durme. Compressed chain of thought: Efficient reasoning through dense representations. *arXiv preprint arXiv:2412.13171*, 2024.

[^17]: Haohan Chi, Huan ang Gao, Ziming Liu, Jianing Liu, Chenyu Liu, Jinwei Li, Kaisen Yang, Yangcheng Yu, Zeda Wang, Wenyi Li, Leichen Wang, Xingtao Hu, Hao Sun, Hang Zhao, and Hao Zhao. Impromptu VLA: Open weights and open data for driving vision-language-action models. In *Advances in Neural Information Processing Systems (Datasets and Benchmarks Track)*, volume 38, 2025.

[^18]: Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghao Li, Jirong Liu, Yang Liu, Zhuoyan Luo, Jinsheng Wang, Wenxuan Wang, Yueze Wang, Chengyuan Wang, Fan Zhang, Yingli Zhao, Ting Pan, Xianduo Li, Zecheng Hao, Wenxuan Ma, Zhuo Chen, Yulong Ao, Tiejun Huang, Zhongyuan Wang, and Xinlong Wang. Emu3.5: Native multimodal models are world learners. *arXiv preprint arXiv:2510.26583*, 2025.

[^19]: Daniel Dauner, Marcel Hallgarten, Tianyu Li, Xinshuo Weng, Zhiyu Huang, Zetong Yang, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, Andreas Geiger, and Kashyap Chitta. NAVSIM: Data-driven non-reactive autonomous vehicle simulation and benchmarking. In *Advances in Neural Information Processing Systems*, volume 37, pages 28706–28719, 2024.

[^20]: Grégoire Delétang, Anian Ruoss, Paul-Ambroise Duquenne, Elliot Catt, Tim Genewein, Christopher Mattern, Jordi Grau-Moya, Li Kevin Wenliang, Matthew Aitchison, Laurent Orseau, Marcus Hutter, and Joel Veness. Language modeling is compression. In *International Conference on Learning Representations*, 2024.

[^21]: Yuntian Deng, Yejin Choi, and Stuart Shieber. From explicit CoT to implicit CoT: Learning to internalize CoT step by step. *arXiv preprint arXiv:2405.14838*, 2024.

[^22]: Xinpeng Ding, Jianhua Han, Hang Xu, Xiaodan Liang, Wei Zhang, and Xiaomeng Li. Holistic autonomous driving understanding by bird’s-eye-view injected multi-modal large models. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 13668–13677, 2024.

[^23]: Yifei Dong, Fengyi Wu, Guangyu Chen, Lingdong Kong, Xu Zhu, Qiyu Hu, Yuxuan Zhou, Jingdong Sun, Jun-Yan He, Qi Dai, Alexander G. Hauptmann, and Zhi-Qi Cheng. Towards unified world models for visual navigation via memory-augmented planning and foresight. *arXiv preprint arXiv:2510.08713*, 2025.

[^24]: Yifei Dong, Fengyi Wu, Yilong Dai, Lingdong Kong, Guangyu Chen, Xu Zhu, Qiyu Hu, Tianyu Wang, Johnalbert Garnica, Feng Liu, Siyu Huang, Qi Dai, and Zhi-Qi Cheng. Language-conditioned world modeling for visual navigation. *arXiv preprint arXiv:2603.26741*, 2026.

[^25]: Scott Ettinger, Shuyang Cheng, Benjamin Caine, Chenxi Liu, Hang Zhao, Sabeek Pradhan, Yuning Chai, Ben Sapp, Charles R Qi, Yin Zhou, et al. Large scale interactive motion forecasting for autonomous driving: The Waymo open motion dataset. In *IEEE/CVF International Conference on Computer Vision*, pages 9710–9719, 2021.

[^26]: Xiang Fei, Jinghui Lu, Qi Sun, Hao Feng, Yanjie Wang, Wei Shi, An-Lan Wang, Jingqun Tang, and Can Huang. Advancing sequential numerical prediction in autoregressive models. In *Annual Meeting of the Association for Computational Linguistics*, pages 562–574, 2025.

[^27]: Hao Feng, Shu Wei, Xiang Fei, Wei Shi, Yingdong Han, Lei Liao, Jinghui Lu, Binghong Wu, Qi Liu, Chunhui Lin, et al. Dolphin: Document image parsing via heterogeneous anchor prompting. In *Annual Meeting of the Association for Computational Linguistics*, pages 21919–21936, 2025a.

[^28]: Tuo Feng, Wenguan Wang, and Yi Yang. A survey of world models for autonomous driving. *arXiv preprint arXiv:2501.11260*, 2025b.

[^29]: Haoyu Fu, Diankun Zhang, Zongchuang Zhao, Jianfeng Cui, Dingkang Liang, Chong Zhang, Dingyuan Zhang, Hongwei Xie, Bing Wang, and Xiang Bai. Orion: A holistic end-to-end autonomous driving framework by vision-language instructed action generation. In *IEEE/CVF International Conference on Computer Vision*, pages 24823–24834, 2025a.

[^30]: Haoyu Fu, Diankun Zhang, Zongchuang Zhao, Jianfeng Cui, Hongwei Xie, Bing Wang, Guang Chen, Dingkang Liang, and Xiang Bai. MindDrive: A vision-language-action model for autonomous driving via online reinforcement learning. *arXiv preprint arXiv:2512.13636*, 2025b.

[^31]: Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The KITTI dataset. *International Journal of Robotics Research*, 32(11):1231–1237, 2013.

[^32]: Jiaheng Geng, Jiatong Du, Xinyu Zhang, Ye Li, Panqu Wang, and Yanjun Huang. DICC: Driving in corner cases. *arXiv preprint arXiv:2512.16055*, 2025.

[^33]: Anurag Ghosh, Shen Zheng, Robert Tamburo, Khiem Vuong, Juan Alvarez-Padilla, Hailiang Zhu, Michael Cardei, Nicholas Dunn, Christoph Mertz, and Srinivasa G. Narasimhan. ROADWork: A dataset and benchmark for learning to recognize, observe, analyze and drive through work zones. In *IEEE/CVF International Conference on Computer Vision*, pages 6132–6142, 2025.

[^34]: Google. Gemini 2.5 Pro preview: even better coding performance. [https://developers.googleblog.com/en/gemini-2-5-pro-io-improved-coding-performance](https://developers.googleblog.com/en/gemini-2-5-pro-io-improved-coding-performance), 2025.

[^35]: Yanchen Guan, Haicheng Liao, Zhenning Li, Jia Hu, Runze Yuan, Guohui Zhang, and Chengzhong Xu. World models for autonomous driving: An initial survey. *IEEE Transactions on Intelligent Vehicles*, pages 1–17, 2024.

[^36]: Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z.F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, et al. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025a.

[^37]: Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, Jingji Chen, Jingjia Huang, Kang Lei, Liping Yuan, Lishu Luo, Pengfei Liu, Qinghao Ye, Rui Qian, Shen Yan, Shixiong Zhao, Shuai Peng, Shuangye Li, Sihang Yuan, Sijin Wu, Tianheng Cheng, Weiwei Liu, Wenqian Wang, Xianhan Zeng, Xiao Liu, Xiaobo Qin, Xiaohan Ding, Xiaojun Xiao, Xiaoying Zhang, Xuanwei Zhang, Xuehan Xiong, Yanghua Peng, Yangrui Chen, et al. Seed1.5-VL technical report. *arXiv preprint arXiv:2505.07062*, 2025b.

[^38]: David Ha and Jürgen Schmidhuber. World models. *arXiv preprint arXiv:1803.10122*, 2018.

[^39]: Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. *arXiv preprint arXiv:1912.01603*, 2019.

[^40]: Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space. *arXiv preprint arXiv:2412.06769*, 2024.

[^41]: Xiaoshuai Hao, Lei Zhou, Zhijian Huang, Zhiwen Hou, Yingbo Tang, Lingfeng Zhang, Guang Li, Zheng Lu, Shuhuai Ren, Xianhui Meng, Yuchen Zhang, Jing Wu, Jinghui Lu, Chenxu Dang, Jiayi Guan, Jianhua Wu, Zhiyi Hou, Hanbing Li, Shumeng Xia, Mingliang Zhou, Yinan Zheng, Zihao Yue, Shuhao Gu, Hao Tian, Yuannan Shen, Jianwei Cui, Wen Zhang, Shaoqing Xu, Bing Wang, Haiyang Sun, Zeyu Zhu, Yuncheng Jiang, Zibin Guo, Chuhong Gong, Chaofan Zhang, Wenbo Ding, Kun Ma, Guang Chen, Rui Cai, Diyun Xiang, Heng Qu, Fuli Luo, Hangjun Ye, and Long Chen. MiMo-Embodied: X-embodied foundation model technical report. *arXiv preprint arXiv:2511.16518*, 2025.

[^42]: Zhiyi Hou, Enhui Ma, Fang Li, Zhiyi Lai, Kalok Ho, Zhanqian Wu, Lijun Zhou, Long Chen, Chitian Sun, Haiyang Sun, et al. Drivemrp: Enhancing vision-language models with synthetic motion data for motion risk prediction. *arXiv preprint arXiv:2507.02948*, 2025.

[^43]: Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. GAIA-1: A generative world model for autonomous driving. *arXiv preprint arXiv:2309.17080*, 2023.

[^44]: Tianshuai Hu, Xiaolu Liu, Song Wang, Yiyao Zhu, Ao Liang, Lingdong Kong, Guoyang Zhao, Zeying Gong, Jun Cen, Zhiyu Huang, Xiaoshuai Hao, Linfeng Li, Hang Song, Xiangtai Li, Jun Ma, Shaojie Shen, Jianke Zhu, Dacheng Tao, Ziwei Liu, and Junwei Liang. Vision-language-action models for autonomous driving: Past, present, and future. *arXiv preprint arXiv:2512.16760*, 2025.

[^45]: Tianshuai Hu, Zeying Gong, Lingdong Kong, Xiaodong Mei, Yiyi Ding, Qi Zeng, Ao Liang, Rong Li, Yangyi Zhong, and Junwei Liang. NavThinker: Action-conditioned world models for coupled prediction and planning in social navigation. *arXiv preprint arXiv:2603.15359*, 2026.

[^46]: Zhijian Huang, Sihao Lin, Guiyu Liu, Mukun Luo, Chaoqiang Ye, Hang Xu, Xiaojun Chang, and Xiaodan Liang. Fuller: Unified multi-modality multi-task 3D perception via multi-level gradient calibration. In *IEEE/CVF International Conference on Computer Vision*, pages 3502–3511, 2023.

[^47]: Zhijian Huang, Tao Tang, Shaoxiang Chen, Sihao Lin, Zequn Jie, Lin Ma, Guangrun Wang, and Xiaodan Liang. Making large language models better planners with reasoning-decision alignment. In *European Conference on Computer Vision*, pages 73–90. Springer, 2024.

[^48]: Zhijian Huang, Chengjian Feng, Feng Yan, Baihui Xiao, Zequn Jie, Yujie Zhong, Xiaodan Liang, and Lin Ma. RoboTron-Drive: All-in-one large multimodal model for autonomous driving. In *IEEE/CVF International Conference on Computer Vision*, pages 8011–8021, 2025.

[^49]: Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. GPT-4o system card. *arXiv preprint arXiv:2410.21276*, 2024.

[^50]: Ayesha Ishaq, Jean Lahoud, Ketan More, Omkar Thawakar, Ritesh Thawkar, Dinura Dissanayake, Noor Ahsan, Yuhao Li, Fahad Shahbaz Khan, Hisham Cholakkal, Ivan Laptev, Rao Muhammad Anwer, and Salman Khan. DriveLMM-o1: A step-by-step reasoning dataset and large multimodal model for driving scenario understanding. *arXiv preprint arXiv:2503.10621*, 2025.

[^51]: Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, Alex Iftimie, Alex Karpenko, Alex Tachard Passos, Alexander Neitz, Alexander Prokofiev, Alexander Wei, Allison Tam, Ally Bennett, Ananya Kumar, Andre Saraiva, Andrea Vallone, Andrew Duberstein, Andrew Kondrich, Andrey Mishchenko, Andy Applebaum, Angela Jiang, et al. OpenAI o1 system card. *arXiv preprint arXiv:2412.16720*, 2024.

[^52]: Weitao Jia, Jinghui Lu, Haiyang Yu, Siqi Wang, Guozhi Tang, An-Lan Wang, Weijie Yin, Dingkang Yang, Yuxiang Nie, Bin Shan, et al. MEML-GRPO: Heterogeneous multi-expert mutual learning for RLVR advancement. In *AAAI Conference on Artificial Intelligence*, volume 40, pages 31283–31291, 2026.

[^53]: Napat Karnchanachari, Dimitris Geromichalos, Kok Seang Tan, Nanxiang Li, Christopher Eriksen, Shakiba Yaghoubi, Noushin Mehdipour, Gianmarco Bernasconi, Whye Kit Fong, Yiluan Guo, et al. Towards learning-based planning: The nuPlan benchmark for real-world autonomous driving. In *IEEE International Conference on Robotics and Automation*, pages 629–636, 2024.

[^54]: Lingdong Kong, Shaoyuan Xie, Hanjiang Hu, Yaru Niu, Wei Tsang Ooi, Benoit R. Cottereau, Lai Xing Ng, Yuexin Ma, Wenwei Zhang, Liang Pan, Kai Chen, Ziwei Liu, Weichao Qiu, Wei Zhang, Xu Cao, Hao Lu, Ying-Cong Chen, Caixin Kang, Xinning Zhou, Chengyang Ying, Wentao Shang, Xingxing Wei, Yinpeng Dong, Bo Yang, Shengyin Jiang, Zeliang Ma, Dengyi Ji, Haiwen Li, Xingliang Huang, Yu Tian, Genghua Kou, Fan Jia, Yingfei Liu, Tiancai Wang, Ying Li, Xiaoshuai Hao, Yifan Yang, Hui Zhang, Mengchuan Wei, Yi Zhou, Haimei Zhao, Jing Zhang, Jinke Li, Xiao He, Xiaoqiang Cheng, Bingyang Zhang, Lirong Zhao, Dianlei Ding, et al. The RoboDrive challenge: Drive anytime anywhere in any condition. *arXiv preprint arXiv:2405.08816*, 2024.

[^55]: Lingdong Kong, Xiang Xu, Jiawei Ren, Wenwei Zhang, Liang Pan, Kai Chen, Wei Tsang Ooi, and Ziwei Liu. Multi-modal data-efficient 3D scene understanding for autonomous driving. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 47(5):3748–3765, 2025a.

[^56]: Lingdong Kong, Wesley Yang, Jianbiao Mei, Youquan Liu, Ao Liang, Dekai Zhu, Dongyue Lu, Wei Yin, Xiaotao Hu, Mingkai Jia, Junyuan Deng, Kaiwen Zhang, Yang Wu, Tianyi Yan, Shenyuan Gao, Song Wang, Linfeng Li, Liang Pan, Yong Liu, Jianke Zhu, Wei Tsang Ooi, Steven C. H. Hoi, and Ziwei Liu. 3D and 4D world modeling: A survey. *arXiv preprint arXiv:2509.07996*, 2025b.

[^57]: Lingdong Kong, Xiang Xu, Youquan Liu, Jun Cen, Runnan Chen, Wenwei Zhang, Liang Pan, Kai Chen, and Ziwei Liu. LargeAD: Large-scale cross-sensor data pretraining for autonomous driving. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 48(2):1291–1308, 2026.

[^58]: Shane Legg and Marcus Hutter. Universal intelligence: A definition of machine intelligence. *Minds and Machines*, 17(4):391–444, 2007.

[^59]: Yingyan Li, Lue Fan, Jiawei He, Yuqi Wang, Yuntao Chen, Zhaoxiang Zhang, and Tieniu Tan. Enhancing end-to-end autonomous driving with latent world model. In *International Conference on Learning Representations*, 2025a.

[^60]: Yingyan Li, Yuqi Wang, Yang Liu, Jiawei He, Lue Fan, and Zhaoxiang Zhang. End-to-end driving with online trajectory evaluation via BEV world model. In *IEEE/CVF International Conference on Computer Vision*, pages 27137–27146, 2025b.

[^61]: Yingyan Li, Shuyao Shang, Weisong Liu, Bing Zhan, Haochen Wang, Yuqi Wang, Yuntao Chen, Xiaoman Wang, Yasong An, Chufeng Tang, Lu Hou, Lue Fan, and Zhaoxiang Zhang. DriveVLA-W0: World models amplify data scaling law in autonomous driving. In *International Conference on Learning Representations*, 2026.

[^62]: Yongkang Li, Kaixin Xiong, Xiangyu Guo, Fang Li, Sixu Yan, Gangwei Xu, Lijun Zhou, Long Chen, Haiyang Sun, Bing Wang, Kun Ma, Guang Chen, Hangjun Ye, Wenyu Liu, and Xinggang Wang. ReCogDrive: A reinforced cognitive framework for end-to-end autonomous driving. *arXiv preprint arXiv:2506.08052*, 2025c.

[^63]: Ao Liang, Youquan Liu, Yu Yang, Dongyue Lu, Linfeng Li, Lingdong Kong, Huaici Zhao, and Wei Tsang Ooi. LiDARCrafter: Dynamic 4D world modeling from LiDAR sequences. In *AAAI Conference on Artificial Intelligence*, volume 40, pages 18406–18414, 2025.

[^64]: Ao Liang, Lingdong Kong, Tianyi Yan, Hongsi Liu, Wesley Yang, Ziqi Huang, Wei Yin, Jialong Zuo, Yixuan Hu, Dekai Zhu, Dongyue Lu, Youquan Liu, Guangfeng Jiang, Linfeng Li, Xiangtai Li, Long Zhuo, Lai Xing Ng, Benoit R. Cottereau, Changxin Gao, Liang Pan, Wei Tsang Ooi, and Ziwei Liu. WorldLens: Full-spectrum evaluations of driving world models in real world. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2026.

[^65]: Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In *International Conference on Learning Representations*, 2024.

[^66]: Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In *Advances in Neural Information Processing Systems*, volume 36, pages 34892–34916, 2023.

[^67]: Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 26296–26306, 2024.

[^68]: Lin Liu, Caiyan Jia, Guanyi Yu, Ziying Song, Junqiao Li, Feiyang Jia, Peiliang Wu, Xiaoshuai Hao, and Yadan Luo. GuideFlow: Constraint-guided flow matching for planning in end-to-end autonomous driving. *arXiv preprint arXiv:2511.18729*, 2025a.

[^69]: Lin Liu, Ziying Song, Caiyan Jia, Hangjun Ye, Xiaoshuai Hao, and Long Chen. DriveWorld-VLA: Unified latent-space world modeling with vision-language-action for autonomous driving. *arXiv preprint arXiv:2602.06521*, 2026.

[^70]: Xueyi Liu, Zuodong Zhong, Junli Wang, Yuxin Guo, Zhiguo Su, Qichao Zhang, Yinfeng Gao, Yupeng Zheng, Donbin Zhao, et al. ReasonPlan: Unified scene prediction and decision reasoning for closed-loop autonomous driving. In *Conference on Robot Learning*, pages 3051–3068. PMLR, 2025b.

[^71]: Jinghui Lu, Linyi Yang, Brian Namee, and Yue Zhang. A rationale-centric framework for human-in-the-loop machine learning. In *Annual Meeting of the Association for Computational Linguistics*, pages 6986–6996, 2022a.

[^72]: Jinghui Lu, Rui Zhao, Brian Mac Namee, and Fei Tan. PUnifiedNER: A prompting-based unified ner system for diverse datasets. In *AAAI conference on artificial intelligence*, volume 37, pages 13327–13335, 2023a.

[^73]: Jinghui Lu, Dongsheng Zhu, Weidong Han, Rui Zhao, Brian Mac Namee, and Fei Tan. What makes pre-trained language models better zero-shot learners? In *Annual Meeting of the Association for Computational Linguistics*, pages 2288–2303, 2023b.

[^74]: Jinghui Lu, Ziwei Yang, Yanjie Wang, Xuejing Liu, Brian Mac Namee, and Can Huang. PaDeLLM-NER: Parallel decoding in large language models for named entity recognition. In *Advances in Neural Information Processing Systems*, volume 37, pages 117853–117880, 2024.

[^75]: Jinghui Lu, Haiyang Yu, Yanjie Wang, Yongjie Ye, Jingqun Tang, Ziwei Yang, Binghong Wu, Qi Liu, Hao Feng, Han Wang, Hao Liu, and Can Huang. A bounding box is worth one token - interleaving layout and text in a large language model for document understanding. In *Annual Meeting of the Association for Computational Linguistics*, pages 7252–7273, 2025.

[^76]: Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In *Advances in Neural Information Processing Systems*, volume 35, pages 2507–2521, 2022b.

[^77]: Yuechen Luo, Fang Li, Shaoqing Xu, Yang Ji, Zehan Zhang, Bing Wang, Yuannan Shen, Jianwei Cui, Long Chen, Guang Chen, Hangjun Ye, Zhi-Xin Yang, and Fuxi Wen. LaST-VLA: Thinking in latent spatio-temporal space for vision-language-action in autonomous driving. *arXiv preprint arXiv:2603.01928*, 2025a.

[^78]: Yuechen Luo, Fang Li, Shaoqing Xu, Zhiyi Lai, Lei Yang, Qimao Chen, Ziang Luo, Zixun Xie, Shengyin Jiang, Jiaxin Liu, Long Chen, Bing Wang, and Zhi-Xin Yang. AdaThinkDrive: Adaptive chain-of-thought reasoning for autonomous driving. *arXiv preprint arXiv:2509.13769*, 2025b.

[^79]: Yuechen Luo, Qimao Chen, Fang Li, Shaoqing Xu, Jaxin Liu, Ziying Song, Zhi-xin Yang, and Fuxi Wen. Unleashing VLA potentials in autonomous driving via explicit learning from failures. *arXiv preprint arXiv:2603.01063*, 2026.

[^80]: Ziang Luo, Kangan Qian, Jiahua Wang, Yuechen Luo, Jinyu Miao, Zheng Fu, Yunlong Wang, Sicong Jiang, Zilin Huang, Yifei Hu, Yuhao Yang, Hao Ye, Mengmeng Yang, Xiaojian Dong, Kun Jiang, and Diange Yang. MTRDrive: Memory-tool synergistic reasoning for robust autonomous driving in corner cases. *arXiv preprint arXiv:2509.20843*, 2025c.

[^81]: Srikanth Malla, Chiho Choi, Isht Dwivedi, Joon Hee Choi, and Jiachen Li. DRAMA: Joint risk localization and captioning in driving. In *IEEE/CVF Winter Conference on Applications of Computer Vision*, pages 1043–1052, 2023.

[^82]: Jiageng Mao, Minzhe Niu, Chenhan Jiang, Hanxue Liang, Jingheng Chen, Xiaodan Liang, Yamin Li, Chaoqiang Ye, Wei Zhang, Zhenguo Li, et al. One million scenes for autonomous driving: ONCE dataset. *arXiv preprint arXiv:2106.11037*, 2021.

[^83]: Ana-Maria Marcu, Long Chen, Jan Hünermann, Alice Karnsund, Benoit Hanotte, Prajwal Chidananda, Saurabh Nair, Vijay Badrinarayanan, Alex Kendall, Jamie Shotton, Elahe Arani, and Oleg Sinavski. LingoQA: Visual question answering for autonomous driving. In *European Conference on Computer Vision*, pages 252–269. Springer, 2024.

[^84]: Ishan Misra, Jean-Bastien Grill, Florent Altché, Sainbayar Sukhbaatar, Piotr Bojanowski, and Yann LeCun. Emerging properties in self-supervised vision transformers. In *International Conference on Learning Representations*, 2022.

[^85]: Gerhard Neuhold, Tobias Ollmann, Samuel Rota Bulo, and Peter Kontschieder. The Mapillary Vistas dataset for semantic understanding of street scenes. In *IEEE/CVF International Conference on Computer Vision*, pages 4990–4999, 2017.

[^86]: Shuyao Shang, Bing Zhan, Yunfei Yan, Yuqi Wang, Yingyan Li, Yasong An, Xiaoman Wang, Jierui Liu, Lu Hou, Lue Fan, et al. DynVLA: Learning world dynamics for action reasoning in autonomous driving. *arXiv preprint arXiv:2603.11041*, 2026.

[^87]: Zhenyi Shen, Hanqi Yan, Linhai Zhang, Zhanghao Hu, Yali Du, and Yulan He. CODI: Compressing chain-of-thought into continuous space via self-distillation. In *Conference on Empirical Methods in Natural Language Processing*, pages 677–693, 2025.

[^88]: Fengyuan Shi, Zhuoyan Luo, Yixiao Ge, Yujiu Yang, Ying Shan, and Limin Wang. Scalable image tokenization with index backpropagation quantization. *arXiv preprint arXiv:2412.02692*, 2024.

[^89]: Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beißwenger, Ping Luo, Andreas Geiger, and Hongyang Li. DriveLM: Driving with graph visual question answering. In *European Conference on Computer Vision*, pages 256–274. Springer, 2024.

[^90]: Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM test-time compute optimally can be more effective than scaling model parameters. *arXiv preprint arXiv:2408.03314*, 2024.

[^91]: Dijia Su, Hanlin Zhu, Yingchen Xu, Jiantao Jiao, Yuandong Tian, and Qinqing Zheng. Token assorted: Mixing latent and text tokens for improved language model reasoning. *arXiv preprint arXiv:2502.03275*, 2025.

[^92]: Jingqun Tang, Qi Liu, Yongjie Ye, Jinghui Lu, Shu Wei, An-Lan Wang, Chunhui Lin, Hao Feng, Zhen Zhao, Yanjie Wang, Yuliang Liu, Hao Liu, Xiang Bai, and Can Huang. MTVQA: Benchmarking multilingual text-centric visual question answering. In *Annual Meeting of the Association for Computational Linguistics*, pages 7748–7763, 2025.

[^93]: Basile Terver, Tsung-Yen Yang, Jean Ponce, Adrien Bardes, and Yann LeCun. What drives success in physical planning with joint-embedding predictive world models? *arXiv preprint arXiv:2512.24497*, 2025.

[^94]: Haochen Tian, Tianyu Li, Haochen Liu, Jiazhi Yang, Yihang Qiu, Guang Li, Junli Wang, Yinfeng Gao, Zhang Zhang, Liang Wang, Long Chen, Hongyang Li, et al. SimScale: Learning to drive via real-world simulation at scale. *arXiv preprint arXiv:2511.23369*, 2025.

[^95]: Naftali Tishby and Noga Zaslavsky. Deep learning and the information bottleneck principle. In *IEEE Information Theory Workshop*, pages 1–5. IEEE, 2015.

[^96]: Naftali Tishby, Fernando C. Pereira, and William Bialek. The information bottleneck method. In *Annual Allerton Conference on Communication, Control, and Computing*, pages 368–377, 1999.

[^97]: Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Ziteng Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal LLMs. In *Advances in Neural Information Processing Systems*, volume 37, pages 87310–87356, 2024.

[^98]: Girish Varma, Anbumani Subramanian, Anoop Namboodiri, Manmohan Chandraker, and CV Jawahar. IDD: A dataset for exploring problems of autonomous navigation in unconstrained environments. In *IEEE/CVF Winter Conference on Applications of Computer Vision*, pages 1743–1751, 2019.

[^99]: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In *Advances in Neural Information Processing Systems*, volume 30, pages 6000–6010, 2017.

[^100]: Han Wang, Yongjie Ye, Bingru Li, Yuxiang Nie, Jinghui Lu, Jingqun Tang, Yanjie Wang, and Can Huang. Vision as LoRA. *arXiv preprint arXiv:2503.20680*, 2025a.

[^101]: Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. VGGT: Visual geometry grounded transformer. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 5294–5306, 2025b.

[^102]: Jie Wang, Guang Li, Zhijian Huang, Chenxu Dang, Hangjun Ye, Yahong Han, and Long Chen. VGGDrive: Empowering vision-language models with cross-view geometric grounding for autonomous driving. *arXiv preprint arXiv:2602.20794*, 2026.

[^103]: Shihao Wang, Zhiding Yu, Xiaohui Jiang, Shiyi Lan, Min Shi, Nadine Chang, Jan Kautz, Ying Li, and Jose M Alvarez. OmniDrive: A holistic vision-language dataset for autonomous driving with counterfactual reasoning. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 22442–22452, 2025c.

[^104]: Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, Zhaokai Wang, Zhe Chen, Hongjie Zhang, Ganlin Yang, Haomin Wang, Qi Wei, Jinhui Yin, Wenhao Li, Erfei Cui, Guanzhou Chen, Zichen Ding, Changyao Tian, Zhenyu Wu, Jingjing Xie, Zehao Li, Bowen Yang, Yuchen Duan, Xuehui Wang, Zhi Hou, Haoran Hao, Tianyi Zhang, Songze Li, Xiangyu Zhao, Haodong Duan, Nianchen Deng, Bin Fu, Yinan He, Yi Wang, Conghui He, Botian Shi, Junjun He, Yingtong Xiong, Han Lv, Lijun Wu, Wenqi Shao, Kaipeng Zhang, Huipeng Deng, Biqing Qi, Jiaye Ge, Qipeng Guo, Wenwei Zhang, Songyang Zhang, Maosong Cao, Junyao Lin, Kexian Tang, Jianfei Gao, Haian Huang, Yuzhe Gu, Chengqi Lyu, Huanze Tang, Rui Wang, Haijun Lv, Wanli Ouyang, Limin Wang, Min Dou, Xizhou Zhu, Tong Lu, Dahua Lin, Jifeng Dai, Weijie Su, Bowen Zhou, Kai Chen, Yu Qiao, Wenhai Wang, and Gen Luo. InternVL3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. *arXiv preprint arXiv:2508.18265*, 2025d.

[^105]: Yan Wang, Wenjie Luo, Junjie Bai, Yulong Cao, Tong Che, Ke Chen, Yuxiao Chen, Jenna Diamond, Yifan Ding, Wenhao Ding, Liang Feng, Greg Heinrich, Jack Huang, Peter Karkus, Boyi Li, Pinyi Li, Tsung-Yi Lin, Dongran Liu, Ming-Yu Liu, Langechuan Liu, Zhijian Liu, Jason Lu, Yunxiang Mao, Pavlo Molchanov, Lindsey Pavao, Zhenghao Peng, Mike Ranzinger, Ed Schmerling, Shida Shen, Yunfei Shi, Sarah Tariq, Ran Tian, Tilman Wekel, Xinshuo Weng, Tianjun Xiao, Eric Yang, Xiaodong Yang, Yurong You, Xiaohui Zeng, Wenyuan Zhang, Boris Ivanovic, and Marco Pavone. Alpamayo-R1: Bridging reasoning and action prediction for generalizable autonomous driving in the long tail. *arXiv preprint arXiv:2511.00088*, 2025e.

[^106]: Xilin Wei, Xiaoran Liu, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Jiaqi Wang, Xipeng Qiu, and Dahua Lin. Sim-cot: Supervised implicit chain-of-thought. *arXiv preprint arXiv:2509.20317*, 2025.

[^107]: Benjamin Wilson, William Qi, Tanmay Agarwal, John Lambert, Jagjeet Singh, Siddhesh Khandelwal, Bowen Pan, Ratnesh Kumar, Andrew Hartnett, Jhony Kaesemodel Pontes, et al. Argoverse 2: Next generation datasets for self-driving perception and forecasting. *arXiv preprint arXiv:2301.00493*, 2023.

[^108]: Shaoyuan Xie, Lingdong Kong, Yuhao Dong, Chonghao Sima, Wenwei Zhang, Qi Alfred Chen, Ziwei Liu, and Liang Pan. Are VLMs ready for autonomous driving? An empirical study from the reliability, data, and metric perspectives. In *IEEE/CVF International Conference on Computer Vision*, pages 6585–6597, 2025a.

[^109]: Shaoyuan Xie, Lingdong Kong, Wenwei Zhang, Jiawei Ren, Liang Pan, Kai Chen, and Ziwei Liu. Benchmarking and improving bird’s eye view perception robustness in autonomous driving. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 47(5):3878–3894, 2025b.

[^110]: Shaoqing Xu, Fang Li, Shengyin Jiang, Ziying Song, Li Liu, and Zhi-xin Yang. GaussianPretrain: A simple unified 3D Gaussian representation for visual pre-training in autonomous driving. *arXiv preprint arXiv:2411.12452*, 2024a.

[^111]: Xiang Xu, Ao Liang, Youquan Liu, Linfeng Li, Lingdong Kong, Ziwei Liu, and Qingshan Liu. U4D: Uncertainty-aware 4D world modeling from LiDAR sequences. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2026.

[^112]: Yige Xu, Xu Guo, Zhiwei Zeng, and Chunyan Miao. SoftCoT: Soft chain-of-thought for efficient reasoning with LLMs. In *Annual Meeting of the Association for Computational Linguistics*, pages 23336–23351, 2025.

[^113]: Zhenhua Xu, Yujia Zhang, Enze Xie, Zhen Zhao, Yong Guo, Kwan-Yee K Wong, Zhenguo Li, and Hengshuang Zhao. DriveGPT4: Interpretable end-to-end autonomous driving via large language model. *IEEE Robotics and Automation Letters*, 9(10):8186–8193, 2024b.

[^114]: Tianyi Yan, Tao Tang, Xingtai Gui, Yongkang Li, Jiasen Zheng, Weiyao Huang, Lingdong Kong, Wencheng Han, Xia Zhou, Xueyang Zhang, et al. AD-R1: Closed-loop reinforcement learning for end-to-end autonomous driving with impartial world models. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2026.

[^115]: Shuang Zeng, Xinyuan Chang, Mengwei Xie, Xinran Liu, Yifan Bai, Zheng Pan, Mu Xu, Xing Wei, and Ning Guo. FutureSightDrive: Thinking visually with spatio-temporal CoT for autonomous driving. *arXiv preprint arXiv:2505.17685*, 2025.

[^116]: Lingfeng Zhang, Xiaoshuai Hao, Yingbo Tang, Haoxiang Fu, Xinyu Zheng, Pengwei Wang, Zhongyuan Wang, Wenbo Ding, and Shanghang Zhang. $nava^{3}$: Understanding any instruction, navigating anywhere, finding anything. *arXiv preprint arXiv:2508.04598*, 2025.

[^117]: Sicheng Zuo, Zixun Xie, Wenzhao Zheng, Shaoqing Xu, Fang Li, Shengyin Jiang, Long Chen, Zhi-Xin Yang, and Jiwen Lu. DVGT: Driving visual geometry transformer. *arXiv preprint arXiv:2512.16919*, 2025.

[^118]: Sicheng Zuo, Zixun Xie, Wenzhao Zheng, Shaoqing Xu, Fang Li, Hanbing Li, Long Chen, Zhi-Xin Yang, and Jiwen Lu. DVGT-2: Vision-geometry-action model for autonomous driving at scale. *arXiv preprint arXiv:2604.00813*, 2026.