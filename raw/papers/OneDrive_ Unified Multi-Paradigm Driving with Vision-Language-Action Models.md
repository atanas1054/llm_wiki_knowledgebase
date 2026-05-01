---
title: "OneDrive: Unified Multi-Paradigm Driving with Vision-Language-Action Models"
source: "https://arxiv.org/html/2604.17915v1"
author:
published:
created: 2026-05-01
description:
tags:
  - "clippings"
---
<sup>1</sup> <sup>2</sup> <sup>3</sup> <sup>4</sup> <sup>5</sup>

Yiwei Zhang    Xuesong Chen<sup>,†</sup>    Jin Gao<sup>,∗</sup>    Hanshi Wang    Fudong Ge    Weiming Hu    Shaoshuai Shi<sup>,∗</sup>    Zhipeng Zhang<sup>,∗</sup>  
<sup>∗</sup> Corresponding authors    <sup>†</sup> Project Leader

###### Abstract

Vision-Language Models (VLMs) excel at autoregressive text generation, yet end-to-end autonomous driving requires multi-task learning with structured outputs and heterogeneous decoding behaviors, such as autoregressive language generation, parallel object detection and trajectory regression. To accommodate these differences, existing systems typically introduce separate or cascaded decoders, resulting in architectural fragmentation and limited backbone reuse. In this work, we present a unified autonomous driving framework built upon a pretrained VLM, where heterogeneous decoding behaviors are reconciled within a single transformer decoder. We demonstrate that pretrained VLM attention exhibits strong transferability beyond pure language modeling. By organizing visual and structured query tokens within a single causal decoder, structured queries can naturally condition on visual context through the original attention mechanism. Textual and structured outputs share a common attention backbone, enabling stable joint optimization across heterogeneous tasks. Trajectory planning is realized within the same causal LLM decoder by introducing structured trajectory queries. This unified formulation enables planning to share the pretrained attention backbone with images and perception tokens. Extensive experiments on end-to-end autonomous driving benchmarks demonstrate state-of-the-art performance, including 0.28 L2 and 0.18 collision rate on nuScenes open-loop evaluation and competitive results (86.8 PDMS) on NAVSIM closed-loop evaluation. The full model preserves multi-modal generation capability, while an efficient inference mode achieves approximately 40% lower latency. Code and models are available at [https://github.com/Z1zyw/OneDrive](https://github.com/Z1zyw/OneDrive)

## 1 Introduction

Recent breakthroughs in Vision Language Models (VLMs) \[Qwen2-VL, Qwen2.5-VL, chen2024internvl, zhu2025internvl3\] highlight their extraordinary multimodal reasoning capabilities. This success naturally inspires the pursuit of Vision Language Action (VLA) models for autonomous driving \[zhou2025autovla, li2025recogdrive\]. Yet integrating these foundational models into driving systems typically demands intricate 3D structural modifications like Bird Eye View (BEV) modeling \[jiang2024senna, han2025percept\]. Such severe architectural deviations from native VLMs hinder the possibility of joint training with massive general domain data, fundamentally bottlenecking model scalability. Conversely, attempting to preserve the original architecture exposes a critical dilemma. When structural changes are kept minimal, the model falls significantly behind leading methods on diverse heterogeneous driving tasks. Although previous works manage to mask this deficiency by relying heavily on point cloud inputs, their performance degrades sharply under pure vision settings. This degradation is highly undesirable since pure vision settings naturally align with the original VLM architecture \[han2025percept\]. Furthermore, given that these models are inherently constrained to output information in textual form, bridging the massive gap between pure text generation and complex driving outputs remains a formidable challenge. Then,

> Can a single pretrained multimodal model simultaneously handle diverse E2E autonomous driving tasks with various forms of output, while maintaining coherent textual generation and generalizing to generative modeling paradigms?

Table 1: Diagnostic study on transferring pretrained LLM weights to a parallel decoder. ✗ means random initialize.

<table><thead><tr><th>VLM</th><th>Attn</th><th>FFN</th><th>NDS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th></tr></thead><tbody><tr><th rowspan="4">InternVL3-1B <cite>[zhu2025internvl3]</cite></th><td>✓</td><td>✓</td><td>31.95</td></tr><tr><td>✓</td><td>✗</td><td>32.05</td></tr><tr><td>✗</td><td>✓</td><td>29.90</td></tr><tr><td>✗</td><td>✗</td><td>31.48</td></tr><tr><th rowspan="4">Qwen2.5-VL-3B <cite>[Qwen2.5-VL]</cite></th><td>✓</td><td>✓</td><td>27.14</td></tr><tr><td>✓</td><td>✗</td><td>31.37</td></tr><tr><td>✗</td><td>✓</td><td>27.95</td></tr><tr><td>✗</td><td>✗</td><td>30.15</td></tr></tbody></table>

Tackling this question requires confronting the inherent complexity of autonomous driving, where interdependent tasks demand distinct input and output formats. Perception and planning typically employ parallel decoders with specific attention mechanisms for simultaneous predictions \[detr, zhu2020deformable, zhang2026integrating, bevformer, bevformerv2, MapTR, maptrv2, zhang2024vq, vad, chen2024vadv2, PARAdrive, chitta2022transfuser\]. Conversely, textual reasoning relies on sequential autoregressive decoding \[sima2023drivelm, chen2025asynchronous, wang2024omnidrive, jiang2024senna\] (see Fig. 2 (b)). This profound structural gap forces traditional approaches to use isolated \[PARAdrive\] or cascaded \[uniad, vad, jia2025drivetransformer, wang2024omnidrive, chen2025asynchronous, drivevlm\] decoders. Such fragmented designs strictly prevent the unified application of pretrained weights and restrict information flow across tasks. Consequently, engineering a singular architecture that harmonizes these diverse modalities while preserving the native VLM structure remains a formidable open challenge.

To determine whether native pretrained VLM models can actually overcome this challenge and support diverse driving tasks, we conduct a preliminary diagnostic study. We adapt pretrained VLM weights into a parallel decoder using the first six layers of the language model, inspired by DETR \[detr\] and StreamPETR \[streampetr\]. Specifically, we freeze the visual backbone and map each pretrained self attention module to the corresponding cross attention in the parallel decoder. For the feedforward networks, we compare pretrained initialization against random initialization, alongside a fully random baseline for all modules to ensure completeness. As Tab. 1 demonstrates, attention weights transfer effectively across tasks. Conversely, reusing feedforward weights provides limited benefits and occasionally causes severe performance degradation alongside training instability (31.37 $\to$ 27.14). These observations reveal that feedforward networks optimized strictly for text generation lack the flexibility required for heterogeneous downstream tasks. However, attention modules pretrained to align visual and textual tokens retain remarkable transferability. This success likely stems from their ability to capture fundamental correspondence patterns between queries and visual features. Inspired by this crucial insight, we investigate whether such transferable attention mechanisms can model a broader range of relationships, ultimately enabling the construction of a unified multitask decoder.

![[intro-arch-cmp-oneDrive.drawio.png|Refer to caption]]

Figure 1: (a) dual-system design with separate decoders; (b) Q-Former–style cascaded decoding; (c) our unified single-decoder framework handling both within one transformer.

OneDrive (see Fig. 3) is our definitive answer. It is a unified multitask framework elegantly leverages pretrained attention to support heterogeneous decoding behaviors within a single transformer decoder. Specifically, we organize visual tokens and structured query tokens into a unified sequence. This arrangement allows query tokens to deeply condition on visual context through the pretrained causal attention of the language model. To enable structured prediction, we augment the shallow layers with dedicated attention among perception query tokens. This modification enhances comprehensive perception capabilities while keeping the foundational backbone attention strictly unchanged. Furthermore, we insert task-specific feedforward networks to execute necessary feature transformations. This elegant design allows autoregressive textual generation alongside parallel perception and planning to share a common attention backbone. Consequently, our design enables stable joint optimization across diverse tasks without introducing isolated transformer branches. Moreover, the unified sequence and shared attention structure inherently facilitate seamless interaction across all driving tasks.

Extensive experiments on end-to-end autonomous driving benchmarks demonstrate that our framework achieves state-of-the-art performance, including 0.28 $L_{2}$ error and 0.18 collision rate on nuScenes open-loop evaluation and competitive results on NAVSIM closed-loop evaluation. The full model preserves multi-modal generation capability, while a truncated inference mode that forwards only the early layers reduces latency by approximately 40% (264→156 ms) for planning-focused deployment.

The main contributions of this work are as follows:

1. We show that pretrained VLM causal attention, originally trained to capture text–image or text–text relations, can be adapted through training to model relations between queries and visual, whereas FFNs pretrained for text generation are difficult to transfer.
2. We propose OneDrive, a unified archicture for multitask, enabling a single transformer decoder to perform both autoregressive text generation and parallel perception and planning tasks.
3. Extensive experiments on nuScenes and NAVSIM demonstrate the effectiveness of our approach, achieving state-of-the-art performance.

## 2 Related Work

#### 2.0.1 Unified Architectures for End-to-end Autonomous Driving.

![[ar-paral-cmp.drawio.png|Refer to caption]]

Figure 2: Two representative decoding paradigms: (a) an autoregressive decoder, (b) a parallel decoder. Existing end-to-end multi-task autonomous driving models typically organize heterogeneous decoders either in a cascaded manner or in parallel.

Autonomous driving requires the integration of multiple interdependent tasks, including perception, prediction, and planning, such as 3D object detection \[huang2021bevdet, bevformer, bevformerv2, bevfusion\], lane detection \[MapTR, maptrv2, maptracker\], BEV segmentation \[bevfusion, zhang2024vq\], occupancy prediction \[zheng2024monoocc, occworld\], and trajectory planning. Early systems decomposed these tasks into separate modules with predefined interfaces, which limited information flow and hindered global optimization. Recent unified frameworks \[uniad, vad, chen2024vadv2, PARAdrive, jia2025drivetransformer\], often referred to as conventional end-to-end driving models \[GE2EAD\], advocate for the joint optimization of perception, mapping, motion forecasting, and planning within a shared backbone. These approaches demonstrate that multi-task learning effectively enhances cross-task consistency and overall computational efficiency. However, these systems do not incorporate language modeling into the unified architecture. Notably, the autoregressive decoding paradigm of language models (Fig. 2(b)) differs fundamentally from the parallel structured decoders (Fig. 2(a)) commonly adopted in end-to-end driving frameworks.

#### 2.0.2 Vision-Language Models in Autonomous Driving.

Recent studies have explored integrating vision-language models into autonomous driving to enhance scene understanding, instruction following, and interpretability. Dual-system designs (see Fig. 1(a)), such as Senna \[jiang2024senna\] and DriveVLM \[drivevlm\], employ separate decoders for structured prediction and language generation. Other works, including OmniDrive \[wang2024omnidrive\], Orion \[fu2025orion\] and SOLVE \[chen2025asynchronous\], adopt cascaded architectures that combine structured decoders with autoregressive LLM decoders (see Fig. 1(b)). In contrast, some approaches \[hwang2024emma, xing2024openemma\] directly formulate all tasks as text generation using a single autoregressive decoder. Despite these efforts, existing methods typically organize heterogeneous decoders either in parallel or in cascaded forms. Conventional end-to-end driving models rely primarily on parallel decoding, while large language models operate in an autoregressive manner. The structural discrepancy between autoregressive token generation and parallel structured prediction remains largely unresolved.

In contrast to prior VLM-based driving works that treat language as an auxiliary modality and unified driving frameworks that integrate tasks through shared representations, our work focuses on unifying heterogeneous driving outputs at the decoding level. We adopt a token-centric formulation based on a pretrained VLM, enabling a single Transformer decoder to generate structured perception, planning trajectories, and textual responses within a unified causal attention framework.

![[unidrivevla-main.drawio.png|Refer to caption]]

Figure 3: Architecture of OneDrive. Surround-view images are encoded into image tokens by a ViT and concatenated with structured query tokens for detection, lane estimation, and planning, as well as text tokens. The unified token sequence is processed by mixed decoder layers built upon the pretrained LLM causal attention. Perception query tokens are augmented with additional self-attention and task-specific feed-forward networks, while the backbone attention remains unchanged. Task-specific MLP heads decode 3D bounding boxes, lanes, and trajectories in parallel, alongside autoregressive next-token prediction for textual generation, enabling unified multi-task learning within a single transformer decoder.

## 3 OneDrive

Experiments in Table 1 show that adapting pretrained VLMs to heterogeneous driving structures and tasks is nontrivial: attention modules transfer well, but feedforward layers often degrade performance. These findings highlight the need for careful architectural and training design for a unified multitask decoder.

Fig. 3 illustrates the overall architecture of OneDrive. We build upon a pretrained vision-language model and unify perception, planning, and language generation within a single Transformer decoder. Visual tokens, structured query tokens, and text tokens are concatenated into a shared sequence and processed by the pretrained causal attention. To accommodate heterogeneous prediction paradigms, we introduce task-specific adaptations, including additional self attention among structured perceptual queries and task-specific feed-forward networks, while preserving the original attention backbone.

### 3.1 Problem Formulation

We formulate end-to-end autonomous driving as a multi-task prediction problem over heterogeneous output spaces. Given multi-view images $\mathcal{I}$ and optional textual inputs, the system is required to jointly predict (i) structured perception outputs, such as 3D bounding boxes and lane structures; (ii) continuous trajectory predictions for planning; and (iii) optional textual descriptions. These tasks differ not only in output representation but also in decoding paradigms, including parallel query-based prediction and autoregressive token generation.

### 3.2 Unified Token Representation

Instead of introducing task-specific decoders, we organize heterogeneous outputs using a unified token-based formulation. Structured prediction tasks, such as 3D object detection and lane estimation, are represented by fixed sets of learnable query tokens. Planning is modeled with dedicated planning tokens, while textual outputs follow standard autoregressive token generation. All tokens are concatenated into a shared sequence processed by a single Transformer decoder:

$$
\mathbf{Z}=[\mathbf{X}_{img},\mathbf{Q}_{det},\mathbf{Q}_{lane},\mathbf{Q}_{plan},\mathbf{X}_{text}],
$$

where $\mathbf{X}_{img}$ denotes image tokens, $\mathbf{Q}_{\cdot}$ denote task-specific query tokens, and $\mathbf{X}_{text}$ represents text tokens. This unified token organization enables heterogeneous prediction paradigms to be handled within a shared decoder architecture.

Similar to SOLVE \[chen2025asynchronous\] and OmniDrive \[wang2024omnidrive\], we adopt StreamPETR \[streampetr\] to organize the detection and lane queries. Specifically, $\mathbf{Q}_{det}$ and $\mathbf{Q}_{lane}$ are constructed following the query formulation of StreamPETR. For planning, we allocate one planning query per future timestep. Each planning query is initialized with an anchor trajectory derived from VAD \[vad\]. In addition, an ego-status embedding is prepended to the planning queries, forming the complete $\mathbf{Q}_{plan}$.

### 3.3 Mixed Decoder Layers

Although all tokens share a unified representation space, heterogeneous tasks entail distinct interaction patterns and feature transformation requirements. To preserve the transferable relational modeling capacity of pretrained VLMs while enabling parallel structured prediction, mixed decoder layers are introduced in the shallow stages, retaining the original causal attention while incorporating minimal task-specific adaptations. The query outputs for structured prediction are directly taken from these shallow mixed layers, whereas the deeper layers remain unchanged from the pretrained decoder.

#### 3.3.1 Causal Attention as Conditional Interface

At the core of each decoder layer, we retain the pretrained LLM causal self-attention. All tokens, including image tokens, perception queries, planning queries, and text tokens, are processed under a shared causal mask.

To enhance spatial modeling for perception and planning, we follow StreamPETR and incorporate 3D positional embeddings \[liu2022petr\] for image tokens and structured query tokens, applied after RoPE \[su2024roformer\], while text tokens remain unchanged. Specifically, for image tokens $\mathbf{X}_{img}$ and structured query tokens $\mathbf{Q}_{task}$, the attention projections are computed as

$$
\mathbf{Q}=\text{RoPE}(\mathbf{X}W_{q})+e^{3D},\quad\mathbf{K}=\text{RoPE}(\mathbf{X}W_{k})+e^{3D},
$$

where $\mathbf{X}\in\{\mathbf{X}_{img},\mathbf{Q}_{det},\mathbf{Q}_{lane},\mathbf{Q}_{plan}\}$ and $e^{3D}$ denotes the corresponding 3D positional embedding. Text tokens $\mathbf{X}_{text}$ follow the original RoPE formulation without additional 3D positional encoding. For the query tokens, we remove the residual connection in the causal attention module, which stabilizes training when adapting from pretrained VLM weights.

This shared attention backbone serves as a unified conditional modeling interface across modalities and tasks. Since query tokens are placed after image tokens, they naturally condition on visual context via causal attention without requiring explicit cross-attention modules. Planning tokens are further appended after perception queries, enabling implicit conditioning on perception features through the same attention mechanism.

#### 3.3.2 Query Interaction and Task-Specific Transformation

Autoregressive causal attention enforces sequential dependencies, which is not ideal for parallel structured prediction. To support end-to-end perception \[detr, sun2021makes\], we introduce an additional self-attention block applied only among structured query tokens:

$$
{\mathbf{Q_{perception}}}=\text{SelfAttn}_{q}(\mathbf{Q_{perception}})=\text{SelfAttn}_{q}([\mathbf{Q_{det}},\mathbf{Q_{lane}}]),
$$

This module enables bidirectional interaction within each query group, facilitating parallel reasoning while leaving the backbone causal attention unchanged. Furthermore, we replace the pretrained language-modeling feed-forward networks (FFNs) for structured queries with task-specific FFNs:

$$
\mathbf{Q}^{\prime}=\text{FFN}_{t}(\tilde{\mathbf{Q}}),
$$

where $t\in\{\text{det},\text{lane},\text{plan}\}$. Text tokens continue to use the original pretrained FFNs. This design preserves transferable attention while allowing task-dependent feature transformations.

### 3.4 Multi-Stage Training

We adopt a three-stage training strategy for stable adaptation from pretrained VLMs to multi-task driving. In the first stage (Perception–language pretraining), we freeze the ViT encoder and train the mixed decoder with perception and text-generation objectives. The causal attention is fully fine-tuned, the LLM decoder is adapted via LoRA, and the perception-specific self-attention, perception FFNs, and MLP heads are randomly initialized and optimized. Gradients are driven by perception and language modeling losses, denoted as $\mathcal{L}_{\text{pretrain}}=\lambda_{perc}\mathcal{L}_{\text{perc}}+\mathcal{L}_{\text{text}}$. In the second stage (Planning adaptation), planning tokens are introduced. We optimize planning queries, the planning FFN, and the planning MLP head in an end-to-end manner, while continuing LoRA adaptation of the LLM decoder. Perception-specific modules remain fixed. Gradients are provided by planning and language modeling loss $\mathcal{L}_{\text{adaptation}}=\lambda_{plan}\mathcal{L}_{\text{plan}}+\mathcal{L}_{\text{text}}$. In the final stage (Joint finetuning), all modules, including the ViT encoder, are jointly fine-tuned under the combined perception, planning, and text objectives, with the overall training loss $\mathcal{L}_{\text{joint}}=\lambda_{perc}\mathcal{L}_{\text{perc}}+\lambda_{plan}\mathcal{L}_{\text{plan}}+\mathcal{L}_{\text{text}}$, enabling full end-to-end optimization of the framework. For object detection \[zhu2020deformable, zhang2026integrating, streampetr, bevformer\], deeply supervision \[lee2015deeply\] is often applied to propagate gradients at every layer, where each decoder layer computes its own loss. Similarly, we employ deeply supervision for the planning task.

## 4 Experiments

### 4.1 Implement Details

#### 4.1.1 Datasets and Evaluation.

We conduct experiments on two autonomous driving benchmarks: nuScenes and NAVSIM.

nuScenes \[nuscenes\] contains 1,000 urban driving scenes, each approximately 20 seconds long, captured with six surround-view cameras and a LiDAR sensor. The dataset provides 3D bounding box annotations for 10 object categories, along with high-definition semantic maps. We follow the standard split with 700 training, 150 validation, and 150 test scenes. For perception supervision, we use 3D detection annotations and lane annotations derived from OpenLaneV2 \[wang2023openlane\]. To enrich language and reasoning signals, we additionally leverage the OmniDrive extension \[wang2024omnidrive\], which augments nuScenes with QA-style annotations spanning perception, prediction, and planning. For planning in the open-loop setting, we measure trajectory accuracy using the L2 displacement error and assess safety using collision rate.

NAVSIM \[dauner2025navsim\] is a planning-oriented benchmark built upon OpenScene \[openscene2023\], a redistribution of nuPlan \[caesar2021nuplan\]. The dataset is divided into 1,192 training scenes (navtrain) and 136 evaluation scenes (navtest). Compared with nuScenes, NAVSIM focuses more heavily on interactive and safety-critical planning scenarios. Metrics include average displacement error, collision rate, and the official NAVSIM score PDMS, which jointly reflect safety, rule compliance, and driving efficiency.

#### 4.1.2 Training.

On the nuScenes benchmark, our model is built upon InternVL3-1B \[zhu2025internvl3\]. We follow the three-stage training strategy described in Sec. 3.4, where each stage is trained for 20 epochs with an initial learning rate of $1\times 10^{-4}$. Experiments are conducted with a batch size of 64 on $64\times$ NVIDIA H20 GPUs.

On NAVSIM, we initialize InternVL3-2B \[zhu2025internvl3\] from the ReCogDrive \[li2025recogdrive\] checkpoint, which has been fine-tuned for autonomous driving scenarios. In this setting, we use only planning queries without introducing additional perception queries, while keeping the same configuration as in Stage1. The model is trained with a learning rate of $1\times 10^{-4}$ and a batch size of 128.

### 4.2 Main Results

#### 4.2.1 Open-loop evaluation in nuScenes.

Table 2: Performance comparison of different methods on the nuScenes dataset for open-loop planning. Methods are categorized into text-based driving models (top) and action-based driving models (bottom). <sup>†</sup> indicates that the model does not use ego-vehicle status as input.

<table><tbody><tr><th rowspan="2">Method</th><th rowspan="2">Reference</th><td colspan="4">L2 (<math><semantics><mi>m</mi> <annotation>m</annotation></semantics></math>) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td colspan="4">Col. Rate (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>1 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></td><td>2 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></td><td>3 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></td><td>Avg.</td><td>1 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></td><td>2 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></td><td>3 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></td><td>Avg.</td></tr><tr><th colspan="10">Text-Based Driving Models</th></tr><tr><th>DriveVLM <cite>[tian2024drivevlm]</cite></th><th>CoRL 2024</th><td>0.18</td><td>0.34</td><td>0.68</td><td>0.40</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><th>DriveVLM-Dual <cite>[tian2024drivevlm]</cite></th><th>CoRL 2024</th><td>0.15</td><td>0.29</td><td>0.48</td><td>0.31</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><th>OmniDrive <cite>[wang2024omnidrive]</cite></th><th>CVPR 2025</th><td>0.14</td><td>0.29</td><td>0.55</td><td>0.33</td><td>0.00</td><td>0.13</td><td>0.78</td><td>0.30</td></tr><tr><th>EMMA <cite>[hwang2024emma]</cite></th><th>TMLR</th><td>0.14</td><td>0.29</td><td>0.54</td><td>0.32</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><th>EMMA+ <cite>[hwang2024emma]</cite></th><th>TMLR</th><td>0.13</td><td>0.27</td><td>0.48</td><td>0.29</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><th>ImpromptuVLA <cite>[chi2025impromptu]</cite></th><th>NeurIPS 2025</th><td>0.13</td><td>0.27</td><td>0.53</td><td>0.30</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><th>SOLVE-VLM <cite>[Chen_2025_CVPR]</cite></th><th>CVPR 2025</th><td>0.13</td><td>0.25</td><td>0.47</td><td>0.28</td><td>0.00</td><td>0.16</td><td>0.43</td><td>0.20</td></tr><tr><th>VGGDrive <cite>[wang2026vggdrive]</cite></th><th>CVPR 2026</th><td>0.14</td><td>0.28</td><td>0.51</td><td>0.31</td><td>0.02</td><td>0.10</td><td>0.55</td><td>0.22</td></tr><tr><th colspan="10">Action-Based Driving Models</th></tr><tr><th>UniAD <sup>†</sup> <cite>[hu2023planning]</cite></th><th>CVPR 2023</th><td>0.59</td><td>1.01</td><td>1.48</td><td>1.03</td><td>0.16</td><td>0.51</td><td>1.64</td><td>0.77</td></tr><tr><th>VAD-Base <sup>†</sup> <cite>[jiang2023vad]</cite></th><th>ICCV 2023</th><td>0.69</td><td>1.22</td><td>1.83</td><td>1.25</td><td>0.06</td><td>0.68</td><td>2.52</td><td>1.09</td></tr><tr><th>BEV-Planner <sup>†</sup> <cite>[li2024ego]</cite></th><th>CVPR 2024</th><td>0.30</td><td>0.52</td><td>0.83</td><td>0.55</td><td>0.10</td><td>0.37</td><td>1.30</td><td>0.59</td></tr><tr><th>UniAD <cite>[hu2023planning]</cite></th><th>CVPR 2023</th><td>0.20</td><td>0.42</td><td>0.75</td><td>0.46</td><td>0.02</td><td>0.25</td><td>0.84</td><td>0.37</td></tr><tr><th>VAD-Base <cite>[jiang2023vad]</cite></th><th>ICCV 2023</th><td>0.17</td><td>0.34</td><td>0.60</td><td>0.37</td><td>0.04</td><td>0.27</td><td>0.67</td><td>0.33</td></tr><tr><th>AD-MLP <cite>[admlp]</cite></th><th>ArXiv 2023</th><td>0.15</td><td>0.32</td><td>0.59</td><td>0.35</td><td>0.00</td><td>0.27</td><td>0.85</td><td>0.37</td></tr><tr><th>BEV-Planner <cite>[li2024ego]</cite></th><th>CVPR 2024</th><td>0.16</td><td>0.32</td><td>0.57</td><td>0.35</td><td>0.00</td><td>0.29</td><td>0.73</td><td>0.34</td></tr><tr><th>SOLVE-E2E <cite>[Chen_2025_CVPR]</cite></th><th>CVPR 2025</th><td>0.14</td><td>0.28</td><td>0.50</td><td>0.31</td><td>0.04</td><td>0.17</td><td>0.68</td><td>0.30</td></tr><tr><th>ColaVLA <cite>[peng2025colavla]</cite></th><th>CVPR 2026</th><td>0.14</td><td>0.27</td><td>0.50</td><td>0.30</td><td>0.04</td><td>0.17</td><td>0.47</td><td>0.23</td></tr><tr><th>gray!20 OneDrive</th><th>–</th><td>0.13</td><td>0.25</td><td>0.46</td><td>0.28</td><td>0.00</td><td>0.12</td><td>0.43</td><td>0.18</td></tr></tbody></table>

As shown in Table 2, among action-based planners, OneDrive achieves the best overall performance in terms of both accuracy and safety, with the lowest average L2 error (0.28 m) and the lowest average collision rate (0.18%). Compared with the strongest prior action-based baselines, SOLVE-E2E (0.31m L2; 0.30% collision) and ColaVLA (0.30m L2; 0.23% collision), our method reduces the L2 error to 0.28 m and lowers the collision rate by 23% relative to ColaVLA, indicating more precise and safer trajectory prediction.

Although ColaVLA also avoids autoregressive decoding, it requires forwarding the full LLM and relies on a customized attention mask, which prevents the use of optimized implementations such as FlashAttention \[dao2022flashattention\]. In contrast, OneDrive only forwards the shadow LLM layers without introducing custom attention masking, enabling a more lightweight and hardware-efficient integration(see Table 7). Moreover, OneDrive remains competitive with recent text-based VLM planners, while avoiding SOLVE-VLM autoregressive text generation with complex chain of thought.

#### 4.2.2 Closed-loop evaluation in NAVSIM.

Table 3: Closed-loop performance on NAVSIM navtest under supervised fine-tuning. Our unified modeling improves planning stability and interaction awareness over prior end-to-end and VLA methods, while reducing inference cost by approximately 40% through unified causal attention without forwarding the full LLM. <sup>†</sup> means using Lidar as extra input.

| Method | Reference | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ego Status MLP | \- | 93.0 | 77.3 | 83.6 | 100 | 62.8 | 65.6 |
| VADv2 \[chen2024vadv2\] | ArXiv 2024 | 97.2 | 89.1 | 91.6 | 100 | 76.0 | 65.6 |
| DrivingGPT \[chen2024drivinggpt\] | ICCV 2025 | 98.9 | 90.7 | 94.9 | 95.6 | 79.7 | 82.4 |
| UniAD \[uniad\] | CVPR 2023 | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| TransFuser <sup>†</sup> \[chitta2022transfuser\] | PAMI 2022 | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| PARA-Drive \[weng2024paradrive\] | CVPR 2024 | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| DRAMA <sup>†</sup> \[yuan2024drama\] | ArXiv 2024 | 98.0 | 93.1 | 94.8 | 100 | 80.1 | 85.5 |
| Hydra-MDP <sup>†</sup> \[hydraMDP\] | ArXiv 2024 | 98.3 | 96.0 | 94.6 | 100 | 78.7 | 86.5 |
| DiffusionDrive <sup>†</sup> \[liao2024diffusiondrive\] | CVPR 2025 | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| Qwen2.5-VL-8B \[Qwen2.5-VL\] | \- | 97.8 | 92.1 | 92.8 | 100 | 78.3 | 83.3 |
| InternVL3-8B \[zhu2025internvl3\] | \- | 97.0 | 92.1 | 91.8 | 100 | 78.9 | 83.3 |
| AutoVLA(SFT) \[zhou2025autovla\] | NeurIPS 2025 | 96.9 | 94.4 | 88.1 | 99.9 | 75.8 | 80.5 |
| ReCogDrive(SFT) \[li2025recogdrive\] | ICLR 2026 | 98.1 | 94.7 | 94.2 | 100 | 80.9 | 86.5 |
| gray!20 Query Decoder Baseliine | – | – | – | – | – | – | 85.0 |
| gray!20 OneDrive | – | 98.4 | 95.2 | 94.9 | 100 | 81.1 | 86.8 |

Table 3 shows the performance comparison on the NAVSIM navtest benchmark, evaluated using closed-loop metrics, including both state-of-the-art end-to-end driving approaches and existing VLA models under supervised fine-tuning. Rather than solely emphasizing absolute leaderboard ranking, this evaluation focuses on how our unified modeling strategy enhances the closed-loop trajectory planning capability of the base VLM. To provide a fair comparison, we additionally include a Query Decoder baseline, which adopts the query-based planning decoder used in ReCogDrive \[li2025recogdrive\]. As shown in the table, while this baseline already achieves competitive performance (PDMS 85.0), our method further improves the result to 86.8 PDMS, indicating that the proposed unified causal decoding design is more effective than conventional query-based planning heads. Notably, our method achieves these gains while reducing the inference cost by approximately 40% (see Table 7), as it avoids repeatedly forwarding the full LLM. This highlights that the proposed design improves not only closed-loop performance but also computational efficiency, making it more suitable for real-time autonomous driving scenarios.

### 4.3 Ablation Study

#### 4.3.1 Text Evaluation.

Table 4: Comparison with OmniDrive under the same training data.

<table><thead><tr><th rowspan="2">Method</th><th colspan="4">L2 (<math><semantics><mi>m</mi> <annotation>m</annotation></semantics></math>) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th></tr><tr><th>1 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></th><th>2 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></th><th>3 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></th><th>Avg.</th></tr></thead><tbody><tr><th>OmniDrive-7B <cite>[li2025recogdrive]</cite></th><td>0.14</td><td>0.29</td><td>0.55</td><td>0.33</td></tr><tr><th>gray!20 Ours-Text-1B</th><td>0.15</td><td>0.29</td><td>0.51</td><td>0.32</td></tr></tbody></table>

To verify that the proposed framework preserves the language generation capability of the underlying VLM, we compare it with OmniDrive under the same training data setting. Specifically, we train our model using the same dataset as OmniDrive and evaluate the text-conditioned trajectory prediction accuracy. As shown in Table 4, our method achieves comparable or better L2 errors across the prediction horizons, yielding a lower average error (0.32 vs. 0.33). These results indicate that integrating the planning decoder and perception modules does not degrade the language capability of the model, while maintaining competitive performance in text-conditioned driving tasks.

#### 4.3.2 Disentangling the Role of Text Supervision.

Table 5: Impact of Text Supervision on Perception and Planning.

<table><thead><tr><th rowspan="2">Task</th><th rowspan="2">Text</th><th rowspan="2">NDS/mAP</th><th colspan="4">L2 (<math><semantics><mi>m</mi> <annotation>m</annotation></semantics></math>) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th colspan="4">Col. Rate (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th></tr><tr><th>1 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></th><th>2 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></th><th>3 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></th><th>Avg.</th><th>1 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></th><th>2 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></th><th>3 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></th><th>Avg.</th></tr></thead><tbody><tr><td>3D Det</td><td></td><td>32.31/22.47</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><td>3D Det</td><td>✓</td><td>33.94/24.39</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><td>E2E</td><td></td><td>–</td><td>0.13</td><td>0.27</td><td>0.51</td><td>0.31</td><td>0.04</td><td>0.19</td><td>0.98</td><td>0.40</td></tr><tr><td>E2E</td><td>✓</td><td>–</td><td>0.13</td><td>0.27</td><td>0.51</td><td>0.31</td><td>0.02</td><td>0.20</td><td>0.85</td><td>0.36</td></tr></tbody></table>

As shown in Table 5, we analyze the effect of retaining text supervision when using a unified causal attention to jointly model perception (3D object detection) or planning. Enabling the text loss leads to small but consistent improvements across tasks. For 3D detection, the NDS/mAP increases slightly after introducing text supervision. For end-to-end planning, the L2 displacement error remains largely unchanged, while the collision rate is reduced (from 0.40 to 0.36), indicating improved driving safety. We attribute this behavior to the optimization characteristics of the pretrained causal attention. The causal attention module is originally pretrained under autoregressive text generation objectives, where its parameter space is shaped to capture sequential semantic dependencies. When adapting it to heterogeneous structured prediction tasks such as object detection and trajectory regression, the optimization process may gradually drift from the regime favored during pretraining. Maintaining a text loss during training helps preserve partial alignment with the original objective, which can provide a mild regularization effect on the shared attention module. This regularization stabilizes the unified causal attention to some extent, benefiting both perception and planning.

#### 4.3.3 Key Hyperparameter λplan\\lambda\_{plan}.

Table 6: Ablation on hyperparameter $\lambda_{plan}$.We vary the weight of the planning loss in the third stage (joint training) to study its effect on planning performance.

<table><thead><tr><th rowspan="2"><math><semantics><msub><mi>λ</mi> <mrow><mi>p</mi> <mo></mo><mi>l</mi> <mo></mo><mi>a</mi> <mo></mo><mi>n</mi></mrow></msub> <annotation>\lambda_{plan}</annotation></semantics></math></th><th rowspan="2">NDS/mAP</th><th colspan="4">L2 (<math><semantics><mi>m</mi> <annotation>m</annotation></semantics></math>) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th colspan="4">Col. Rate (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th></tr><tr><th>1 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></th><th>2 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></th><th>3 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></th><th>Avg.</th><th>1 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></th><th>2 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></th><th>3 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></th><th>Avg.</th></tr></thead><tbody><tr><th>0.25</th><th>44.82/35.10</th><td>0.127</td><td>0.258</td><td>0.473</td><td>0.287</td><td>0.000</td><td>0.078</td><td>0.449</td><td>0.176</td></tr><tr><th>0.5</th><th>45.45/35.58</th><td>0.126</td><td>0.255</td><td>0.467</td><td>0.283</td><td>0.000</td><td>0.039</td><td>0.488</td><td>0.176</td></tr><tr><th>1.0</th><th>45.75/35.95</th><td>0.126</td><td>0.252</td><td>0.461</td><td>0.280</td><td>0.000</td><td>0.117</td><td>0.430</td><td>0.182</td></tr><tr><th>2.0</th><th>44.72/34.01</th><td>0.127</td><td>0.258</td><td>0.472</td><td>0.286</td><td>0.000</td><td>0.098</td><td>0.762</td><td>0.287</td></tr></tbody></table>

We conduct an ablation study on the planning loss weight $\lambda_{plan}$, as shown in Table 6. Increasing $\lambda_{plan}$ from 0.25 to 1.0 consistently improves both perception and planning metrics, with the best overall performance achieved at $\lambda_{plan}=1.0$, yielding an NDS/mAP of 45.75/35.95, the lowest average L2 error of 0.280 m, and a competitive collision rate. Further increasing $\lambda_{plan}$ to 2.0 leads to a degradation in both NDS/mAP and collision rate, suggesting that over-emphasizing the planning objective can hurt model stability. These results indicate that a balanced weighting ($\lambda_{plan}=1.0$) effectively aligns perception and planning tasks for optimal closed-loop performance.

#### 4.3.4 Inference Latency.

Table 7: Inference latency comparison on a single NVIDIA H20 GPU (<sup>†</sup> results without FlashAttention, tested using the ColaVLA \[peng2025colavla\] implementation). We report end-to-end latency in milliseconds per frame under identical batch-size settings.

| Method | AR | Latency(ms) |
| --- | --- | --- |
| ReCogDrive \[li2025recogdrive\] |  | 263 |
| gray!20 Ours |  | 156 |
| OmniDrive <sup>†</sup> \[wang2024omnidrive\] | ✓ | 3727 |
| SOLVE-VLM <sup>†</sup> \[xia2020synthesize\] | ✓ | 3719 |
| ColaVLA <sup>†</sup> \[peng2025colavla\] |  | 727 |
| gray!20 Ours |  | 513 |

Table 7 reports the end-to-end inference latency of our framework compared with recent VLM-based autonomous driving systems, measured on a single NVIDIA H20 GPU under identical batch settings. On NAVSIM, our model reduces the per-frame latency from 263 ms (ReCogDrive) to 156 ms, demonstrating a more efficient decoding pipeline. On nuScenes, compared with ColaVLA, a strong non-autoregressive baseline, our method achieves lower latency (513 ms vs. 727 ms). Notably, our framework processes all camera views as input to the LLM, resulting in longer token sequences. Despite this, the unified decoder design avoids customized modules and fully leverages standard transformer acceleration, yielding a clear latency advantage in practice.

#### 4.3.5 Multi-Stage Training.

Table 8: Perception and Planning Performance at Each Stage. We report perception performance (NDS/mAP) and planning metrics at different training stages, including direct E2E adaptation, perception pretraining with planning adaptation, and final joint training.

<table><thead><tr><th rowspan="2">Stage</th><th rowspan="2">NDS/mAP</th><th colspan="4">L2 (<math><semantics><mi>m</mi> <annotation>m</annotation></semantics></math>) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th colspan="4">Col. Rate (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th></tr><tr><th>1 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></th><th>2 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></th><th>3 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></th><th>Avg.</th><th>1 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></th><th>2 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></th><th>3 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></th><th>Avg.</th></tr><tr><th>Only Planning Adapation</th><th>–</th><th>0.14</th><th>0.29</th><th>0.54</th><th>0.32</th><th>0.02</th><th>0.23</th><th>1.01</th><th>0.42</th></tr></thead><tbody><tr><th>Perception Pretrain</th><th>33.18/22.66</th><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><th>Planning Adaption</th><th>33.18/22.66</th><td>0.13</td><td>0.26</td><td>0.49</td><td>0.29</td><td>0.02</td><td>0.12</td><td>0.66</td><td>0.29</td></tr><tr><th>Joint Training</th><th>45.75/35.95</th><td>0.13</td><td>0.25</td><td>0.46</td><td>0.28</td><td>0.00</td><td>0.12</td><td>0.43</td><td>0.18</td></tr></tbody></table>

We conduct an ablation study on the proposed multi-stage training strategy, as summarized in Table 8. Directly adapting the model for end-to-end planning already produces reasonable planning performance. Introducing perception pretraining before planning adaptation further improves the planning metrics, reducing the average L2 error from 0.32 m to 0.29 m and the collision rate from 0.42% to 0.29%. Finally, performing joint training on both perception and planning tasks significantly improves overall performance, boosting the perception accuracy to 45.75/35.95 NDS/mAP while further reducing the planning error to 0.28 m and the collision rate to 0.18%. These results suggest that perception pretraining provides a strong initialization for planning, while joint optimization of perception and planning leads to the best overall performance.

#### 4.3.6 Tasks Sequences.

Table 9: Ablation on Token Sequence. We compare different ordering of perception tokens (lane and detection) before the E2E planning tokens, under both sequential adaptation and joint training.

<table><thead><tr><th rowspan="2">Token Sequence</th><th colspan="4">L2 (<math><semantics><mi>m</mi> <annotation>m</annotation></semantics></math>) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th colspan="4">Col. Rate (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th></tr><tr><th>1 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></th><th>2 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></th><th>3 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></th><th>Avg.</th><th>1 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></th><th>2 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></th><th>3 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></th><th>Avg.</th></tr></thead><tbody><tr><td>Lane <math><semantics><mo>→</mo> <annotation>\to</annotation></semantics></math> Det <math><semantics><mo>→</mo> <annotation>\to</annotation></semantics></math> Planning (Adaptation)</td><td>0.14</td><td>0.28</td><td>0.52</td><td>0.31</td><td>0.02</td><td>0.18</td><td>0.82</td><td>0.34</td></tr><tr><td>Lane <math><semantics><mo>→</mo> <annotation>\to</annotation></semantics></math> Det <math><semantics><mo>→</mo> <annotation>\to</annotation></semantics></math> Planning (Joint Training)</td><td>0.13</td><td>0.27</td><td>0.50</td><td>0.30</td><td>0.02</td><td>0.21</td><td>0.64</td><td>0.29</td></tr><tr><td>Det <math><semantics><mo>→</mo> <annotation>\to</annotation></semantics></math> Lane <math><semantics><mo>→</mo> <annotation>\to</annotation></semantics></math> Planning (Adaptation)</td><td>0.13</td><td>0.26</td><td>0.49</td><td>0.29</td><td>0.00</td><td>0.12</td><td>0.66</td><td>0.27</td></tr><tr><td>Det <math><semantics><mo>→</mo> <annotation>\to</annotation></semantics></math> Lane <math><semantics><mo>→</mo> <annotation>\to</annotation></semantics></math> Planning (Joint Training)</td><td>0.13</td><td>0.25</td><td>0.46</td><td>0.28</td><td>0.00</td><td>0.12</td><td>0.43</td><td>0.18</td></tr></tbody></table>

We ablate the token sequence order in Table 9. Det $\to$ Lane $\to$ Planning consistently outperforms Lane $\to$ Det $\to$ Planning in both L2 error and collision rate. Joint training further improves results, reducing average L2 from 0.31 m to 0.28 m and collision rate from 0.34% to 0.18%, indicating that placing detection tokens first and end-to-end optimization benefits planning accuracy and safety.

#### 4.3.7 Discussion.

Following StreamPETR, we analyze the 3D detection capability of the InternVL3 visual backbone. We re-implement StreamPETR using InternVL3-ViT and align the decoder dimension for fair comparison.

Table 10: 3D Detection performance with the InternVL3-ViT backbone on nuscenes.

| Method | NDS $\uparrow$ | mAP $\uparrow$ |
| --- | --- | --- |
| StreamPETR (frozen) | 31.48 | 20.26 |
| Ours (frozen) | 33.94 | 24.39 |
| StreamPETR | 45.31 | 36.11 |
| Ours | 47.73 | 40.03 |

As shown in Table 10, under the same InternVL3-ViT backbone, our method consistently outperforms the StreamPETR baseline in both frozen and fully fine-tuned settings (31.48/45.31 vs. 33.94/47.73). However, the overall detection performance with InternVL3-ViT is still noticeably lower than the original StreamPETR baseline reported with standard ViT-Large backbones \[streampetr\]. We attribute this gap to two core factors. First, the effective token resolution is reduced because VLM pipelines downsample image features before entering the decoder. Second, the language-centric pretraining objective of InternVL3 does not explicitly optimize for detection-oriented visual representations \[han2025percept, di2026revisiting\]. While Percept-WAM \[han2025percept\] achieves strong detection performance, this advantage is partly due to its use of additional point cloud inputs.

#### 4.3.8 Future work.

Based on the discussion above, future work could further enhance the capabilities of pure VLM-oriented ViTs in detection and other perception tasks. One promising avenue is to incorporate detection-aware objectives during multimodal pretraining, allowing the visual encoder to develop more object-centric representations. Another potential direction is to explore higher-resolution or multi-scale tokenization strategies to reduce the spatial detail loss caused by feature downsampling in VLM pipelines. Moreover, the model’s scaling behavior has yet to be fully assessed. Evaluating its performance with larger architectures and on substantially bigger datasets will be crucial to determine whether the current improvements generalize to more demanding settings and to identify potential limitations in efficiency and robustness.

## 5 Conclusion

In this paper, we presented OneDrive, a unified end-to-end autonomous driving framework that reconciles heterogeneous decoding behaviors within a single pretrained VLM decoder. By retaining the original causal attention as a shared backbone and organizing visual and structured query tokens within a unified sequence, the framework enables stable joint optimization across perception and planning while preserving multi-modal generation capability. Extensive experiments on nuScenes and NAVSIM demonstrate state-of-the-art performance and improved inference efficiency. We hope that OneDrive can contribute to the advancement of both vision–language action modeling and end-to-end autonomous driving research.