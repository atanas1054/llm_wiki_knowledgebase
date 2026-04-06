---
title: "HERMES: A Holistic End-to-End Risk-Aware Multimodal Embodied System with Vision–Language Models for Long-Tail Autonomous Driving"
source: "https://arxiv.org/html/2602.00993v1"
author:
published:
created: 2026-04-05
description:
tags:
  - "clippings"
---
Weizhe Tang, Junwei You\*, Jiaxi Liu, Zhaoyi Wang, Rui Gan, Zilin Huang, Feng Wei, and Bin Ran \*Corresponding author: Junwei You (jyou38@wisc.edu)Weizhe Tang, Junwei You, Jiaxi Liu, Zhaoyi Wang, Rui Gan, Zilin Huang, and Bin Ran are with the Department of Civil and Environmental Engineering, University of Wisconsin–Madison, Madison, WI 53711 USA.

###### Abstract

End-to-end autonomous driving models increasingly benefit from large vision–language models for semantic understanding, yet ensuring safe and accurate operation under long-tail conditions remains challenging. These challenges are particularly prominent in long-tail mixed-traffic scenarios, where autonomous vehicles must interact with heterogeneous road users, including human-driven vehicles and vulnerable road users, under complex and uncertain conditions. This paper proposes HERMES, a holistic risk-aware end-to-end multimodal driving framework designed to inject explicit long-tail risk cues into trajectory planning. HERMES employs a foundation-model-assisted annotation pipeline to produce structured Long-Tail Scene Context and Long-Tail Planning Context, capturing hazard-centric cues together with maneuver intent and safety preference, and uses these signals to guide end-to-end planning. HERMES further introduces a Tri-Modal Driving Module that fuses multi-view perception, historical motion cues, and semantic guidance, ensuring risk-aware accurate trajectory planning under long-tail scenarios. Experiments on the real-world long-tail dataset demonstrate that HERMES consistently outperforms representative end-to-end and VLM-driven baselines under long-tail mixed-traffic scenarios. Ablation studies verify the complementary contributions of key components.

## I Introduction

The rapid advancement of artificial intelligence (AI) has catalyzed transformative progress in autonomous driving, where ensuring safety through accurate perception, prediction, and decision-making remains the paramount challenge [^1] [^29] [^19]. In particular, these challenges are exacerbated in long-tail mixed-traffic environments, where autonomous vehicles must interact with human-driven vehicles and vulnerable road users under complex, uncertain, and socially coupled conditions.

Early autonomous driving systems predominantly adopted a modular architecture, decomposing the complex driving task into sequential components: perception, prediction, planning, and control [^43] [^33] [^35] [^40] [^12] [^23], where each module could be independently developed and optimized. For instance, perception systems detect objects and lane boundaries, while prediction modules forecast agent trajectories using probabilistic frameworks such as MultiPath [^5], graph-based representations like VectorNet [^13], and goal-driven approaches such as TNT [^45]. The modular approach enables clear failure attribution and facilitates leveraging domain expertise for targeted improvements. However, this decomposition introduces fundamental limitations: information loss may occur at module boundaries as rich sensor data is compressed into intermediate representations, errors propagate and compound through the pipeline, and hand-crafted interfaces prevent joint optimization toward the ultimate driving objective [^4] [^6].

![[x1 9.png|Refer to caption]]

Figure 1: Comparison of autonomous driving paradigms. (a) Traditional end-to-end models integrate perception, prediction, and planning in a unified pipeline. (b) VLM/VLA-based approaches leverage foundation models for reasoning and planning. (c) HERMES (ours) combines VLM-generated long-tail context embeddings with a risk-aware end-to-end driving model for safe trajectory planning.

To address these information bottlenecks, advances have explored planning-oriented end-to-end frameworks that enable joint optimization across all driving tasks through differentiable training. Specifically, InterFuser [^31] fuses multimodal multi-view sensor data through transformer-based attention mechanisms, generating interpretable semantic outputs such as waypoints and object density maps to constrain control predictions. UniAD [^18] integrates full-stack driving tasks including tracking, mapping, motion forecasting, and occupancy prediction within a unified query-based transformer architecture, enabling gradient flow from planning objectives back through all preceding modules. VAD [^22] employs a fully vectorized scene representation, encoding dynamic agents and static map elements as explicit instance-level planning constraints while achieving remarkable inference efficiency. These systems achieve state-of-the-art planning performance by preserving continuous information flow through differentiable architectures. However, a critical limitation emerges. While these models excel at pattern recognition from large-scale training data, they lack the high-level semantic scene understanding and common-sense reasoning capabilities necessary to interpret complex traffic scenarios, understand implicit social conventions, and generalize to rare corner cases beyond their training distribution.

Recognizing this semantic understanding gap, further studies have turned to large language models (LLMs) and vision-language models (VLMs) to leverage their powerful reasoning capabilities and extensive pre-trained knowledge. For example, GPT-Driver [^24] pioneers the reformulation of motion planning as a language modeling problem, demonstrating that LLMs can generate safe driving trajectories through natural language descriptions of coordinate positions while providing interpretable reasoning. DriveGPT4 [^39] extends this paradigm by processing multi-frame video inputs and textual queries, which enables vehicles to interpret actions and answer user questions. DriveLM [^32] introduces graph-structured visual question answering to connect perception, prediction, and planning tasks through human-written reasoning logic, facilitating more explainable decision-making. These approaches demonstrate promising semantic understanding and interpretability. Nevertheless, a fundamental gap remains: these methods primarily operate at the semantic reasoning and high-level planning abstraction, rather than delivering end-to-end planning or closed-loop control with continuous, high-frequency actuation commands.

To bridge this gap between semantic understanding and actionable planning, recent research has explored multimodal large language models (MLLMs)-powered autonomous driving systems that connect high-level reasoning to actionable trajectory planning and control within a unified framework. For example, LMDrive [^30] pioneered language-guided closed-loop driving by processing multi-view camera and LiDAR data alongside natural language instructions, enabling human-vehicle interaction through 64K instruction-following trajectories. EMMA [^21], developed by Waymo and built upon Google’s Gemini foundation model, demonstrates how billion-parameter MLLMs can be adapted by representing all inputs and outputs including 3D object locations, trajectories, and road graphs as natural language text within a unified language space, achieving state-of-the-art motion planning performance. DriveVLM [^34] proposes a dual-system architecture that combines VLM-based scene understanding with traditional planning pipelines to balance reasoning capability with spatial precision. While these systems successfully unify semantic reasoning with continuous control, they predominantly focus on nominal driving scenarios and overlook explicit safety modeling and risk-aware planning. They lack dedicated mechanisms to identify and reason about rare but safety-critical long-tail events, such as sudden pedestrian incursions, aggressive cut-ins, or sensor occlusions, that are statistically underrepresented in training data yet account for the majority of real-world accidents. These limitations are particularly problematic in long-tail mixed-traffic scenarios, where safety-critical decisions must account for heterogeneous agents, social interactions, and rare but high-impact events.

To address these critical safety and long-tail challenges, this study introduces HERMES, an MLLM-based embodied framework that explicitly integrates safety-critical reasoning and risk-aware trajectory planning. As shown in Figure 1, HERMES bridges the gap between end-to-end actionable planning and semantic understanding by leveraging foundation model intelligence to identify and reason particularly about rare corner cases in safety-critical scenarios. The key contributions are summarized as follows:

- We propose HERMES, the first framework that leverages large foundation models to systematically address safety-critical long-tail scenarios in end-to-end autonomous driving, where the foundation model is specifically applied to identify and reason about rare corner cases that are critical for safe trajectory planning and vehicle operation.
- We construct a specialized dataset for long-tail scenario reasoning based on WOD-E2E [^38], a curated dataset for long-tail end-to-end driving. On top of WOD-E2E, we add two types of fixed offline annotations: the Long-Tail Scene Context for multi-view hazard-centric descriptions, and the Long-Tail Planning Context including the risk level, driving intention, high-level directives, and planning rationale, which provides an explicit foundation for risk-informed planning and safety-critical decision-making in long-tail scenarios.
- We introduce a multimodal fusion architecture, termed the Tri-Modal Driving Module. It delivers risk-aware and accurate end-to-end trajectory planning through the fusion of multi-view camera features, historical ego-motion patterns, and long-tail instruction embeddings encoded from the long-tail contexts.
- Extensive experiments on the annotated WOD-E2E dataset demonstrate the effectiveness of our approach against existing methods, with ablation studies further validating the contribution of each key component.

## II Related Work

### II-A Foundation-Model-Driven End-to-End Driving

Recent years have witnessed rapid progress in end-to-end autonomous driving driven by large foundation models, where planning is increasingly cast as a unified sequence modeling problem over multimodal observations and structured outputs. A representative line of work reformulates motion planning into language modeling, prompting and fine-tuning large language models to generate discretized trajectory tokens while exposing interpretable reasoning traces [^24] [^11]. Beyond pure text generation, multimodal LLM and VLM systems have been developed to jointly process video or multi-view visual inputs and produce driving actions together with natural-language explanations, improving transparency and human interaction [^39] [^32] [^41] [^42] [^20]. Toward more deployable autonomy, LMDrive [^30] introduces a language-guided, closed-loop end-to-end driving framework that integrates multimodal sensor inputs with textual navigation instructions in interactive environments. In parallel, EMMA [^21] advances a generalist MLLM that unifies planning, perception, and map-related outputs within a language-centric interface, demonstrating the potential of foundation models for multi-task driving. More recent efforts further examine efficiency and systematization, such as OpenEMMA [^37] and LightEMMA [^26], which provide lightweight or open implementations to facilitate reproducible evaluation of VLM-based driving agents under practical constraints. Vision–Language–Action (VLA) models also emerge as a closely related paradigm that directly outputs actions or trajectory tokens from multimodal inputs, such as OpenDriveVLA [^46] and AutoVLA [^47].

Despite these advances, existing foundation-model-powered end-to-end driving methods still mainly focus on general semantic understanding and nominal planning, with limited dedicated design for safety-critical long-tail events and risk-aware trajectory generation in complex and mixed traffic environments.

### II-B Long-Tail Modeling in Autonomous Driving

Long-tail challenges in autonomous driving manifest in many forms, and prior work has studied a range of representative regimes rather than a fixed taxonomy. For low-illumination conditions, Dark Model Adaptation [^8] targets the day-to-night domain gap for driving-scene understanding, and GCMA [^27] further addresses nighttime ambiguity via a progressive day–twilight–night adaptation curriculum with uncertainty-aware evaluation. For adverse weather and degraded visibility, ACDC [^28] provides a systematic benchmark covering fog, rain, snow, and night with correspondences to support controlled robustness evaluation, while Seeing Through Fog Without Seeing Fog [^3] demonstrates that adaptive multimodal fusion can generalize to unseen harsh weather without explicit fog estimation. For work zones with temporary topology and rule changes, ROADWork [^14] highlights work zones as a distinct long-tail regime and shows that strong foundation models underperform without targeted adaptation. Beyond condition-specific robustness, long-tail safety is also approached through knowledge transfer and generation: DiMA [^16] distills multimodal large-model knowledge to improve planning robustness to rare events while enabling efficient inference, and generative world-model efforts such as GAIA-1 [^17] and DriveDreamer-2 [^44] enable controllable synthesis of rare hazards for scalable stress testing and data augmentation. Recent surveys further systematize world-model-based simulation as an emerging backbone for long-tail coverage and evaluation under distribution shifts [^10]. Latest industry-scale efforts further highlight reasoning-augmented VLA models, and for instance, Alpamayo-R1 [^36] couples interpretable chain-of-causation reasoning with action prediction and reports improved generalization under long-tail conditions.

Nevertheless, prior work often addresses scenario-specific robustness on dedicated driving functions or data synthesis in isolation, leaving a broader gap in systematically integrating long-tail signals into end-to-end trajectory planning with explicit and controllable safety preference, particularly in mixed-traffic environments involving heterogeneous road users.

## III Methodology

### III-A Overview of HERMES

HERMES addresses safety-critical long-tail scenarios in autonomous driving through a teacher–student framework that distills risk-aware semantic reasoning from large VLMs into an efficient end-to-end trajectory planning system. These long-tail scenarios primarily arise in mixed-traffic environments involving heterogeneous road users, where complex interactions are rare but high-impact events pose significant challenges for safe trajectory planning. Hence, the core idea of HERMES is to leverage structured semantic instructions generated by a foundation model as auxiliary guidance, which enables a lightweight student model to handle rare and hazardous situations while maintaining real-time inference capability.

As illustrated in Figure 2, HERMES consists of two main components: a Long-Tail Instruction Embedding module, which functions as a teacher to provide semantic guidance, and a Tri-Modal Driving Module, which serves as a student to perform real-time trajectory planning. The Long-Tail Instruction Embedding module employs a cloud-based VLM, to analyze multi-view surround images together with vehicle motion history, producing structured semantic instructions in the form of Long-Tail Scene Context and Long-Tail Planning Context. The Long-Tail Scene Context describes rare and safety-critical elements in the driving environment such as occlusions, abnormal interactions, and uncommon objects from multiple viewpoints, while the Long-Tail Planning Context provides interpretable risk-aware high-level planning guidance based on the identified hazards. In our experiments, these instructions are pre-computed and annotated as part of the training data to facilitate efficient training and reproducibility.

The Tri-Modal Driving Module performs real-time trajectory planning by jointly processing visual observations, historical vehicle states, high-level driving intent, and the pre-computed instructions. Specifically, multi-view visual observations are first encoded to obtain spatial representations of the surrounding environment. These visual features are then augmented with the encoded Long-Tail Scene Context to produce scene-aware representations that emphasize rare and safety-critical elements. In parallel, historical vehicle states are encoded to summarize temporal motion patterns. The resulting visual and temporal representations are subsequently fused to form a unified planning context, which is further modulated by high-level driving intent and refined using the Long-Tail Planning Context. Through this process, the final planning representation integrates rich information from local observations together with VLM-derived long-tail contexts, enabling accurate and safety-aware vehicle operation.

![[x2 7.png|Refer to caption]]

Figure 2: Overview of the HERMES framework. The Long-Tail Instruction Embedding module employs a cloud-based VLM (e.g., Qwen-VL-Flash) to analyze 8-camera images and generate structured annotations comprising Long-Tail Scene Context and Long-Tail Planning Context, which are encoded by BGE-M3 into text embeddings. The Tri-Modal Driving Module sequentially processes multimodal inputs: the Vision Encoder first extracts spatial features, which are fused with scene embeddings via Scene Fusion to produce scene context; subsequently, the State Encoder processes historical trajectories, which are integrated with scene context through Planning Context Fusion; finally, Intent Modulator, Risk Planning Cross-Attention, and Temporal Decoder generate risk-aware trajectories by balancing VLM guidance with learned trajectory patterns.

### III-B Prompt Crafting for Long-Tail Annotation

The quality of VLM-generated annotations critically depends on carefully designed prompts that elicit structured, consistent, and safety-focused reasoning. Unlike general-purpose vision-language tasks, safe and reliable autonomous driving requires the model to produce actionable planning directives grounded in precise spatial understanding and risk assessment. To address this, we develop a specialized prompt template tailored for annotating the WOD-E2E dataset [^38], which focuses on long-tail and safety-critical driving scenarios. The prompt explicitly guides the VLM to analyze such scenarios through a structured reasoning process, enabling the generation of high-quality long-tail annotations suitable for end-to-end driving research.

TABLE I: Structured VLM prompt design for long-tail scenario annotation

<table><thead><tr><th>Component</th><th>Prompt Content</th></tr></thead><tbody><tr><td colspan="2">Input Specification</td></tr><tr><td>System Role</td><td>You are an autonomous driving planner specializing in safety-critical long-tail scenario analysis. Your task is to identify rare hazardous elements and generate risk-aware planning strategies.</td></tr><tr><td>Camera Images</td><td>You are given four grouped image inputs with fixed ordering: Image 1 contains FRONT_LEFT, FRONT, and FRONT_RIGHT views; Image 2 contains REAR_LEFT, REAR, and REAR_RIGHT views; Image 3 contains the SIDE_LEFT view; Image 4 contains the SIDE_RIGHT view. The ego-centric coordinate frame follows the convention that the forward direction is positive and the left direction is positive.</td></tr><tr><td>Motion States</td><td>You are provided with historical ego vehicle motion states expressed in the same ego-centric coordinate frame as the camera inputs, including past positions [x, y], velocities [v_x, v_y], and accelerations [a_x, a_y] over multiple timesteps. A high-level driving intent is also provided as one of unknown, go straight, turn left, turn right. Use this information to infer motion trends and detect anomalous behaviors.</td></tr><tr><td colspan="2">Long-Tail Scene Context</td></tr><tr><td>Scene Description</td><td>Describe the driving scene using a multi-view structure organized by viewpoint. Report observations separately for the front, rear, side-left, and side-right views. Focus on objective, image-grounded descriptions of the scene.</td></tr><tr><td></td><td>Include basic traffic elements such as road layout, lanes, vehicles, pedestrians, cyclists, obstacles, buildings, and intersections.</td></tr><tr><td></td><td>Pay special attention to long-tail hazards, including occlusions, abnormal agent behaviors, rare objects, adverse environmental conditions, and compound edge cases involving multiple simultaneous risks.</td></tr><tr><td colspan="2">Long-Tail Planning Context</td></tr><tr><td>Risk Level</td><td>Assess the overall safety risk of the scenario and classify it as low, medium, or high based on scene complexity, vulnerable road users, collision likelihood, and reaction time.</td></tr><tr><td>Intention</td><td>Infer the high-level driving maneuver for the ego vehicle. Choose one action from go straight, turn left, turn right, or stop.</td></tr><tr><td>High-Level Planning</td><td>Provide a concise planning directive describing the intended driving behavior, such as maintaining lane, yielding, braking, or proceeding cautiously.</td></tr><tr><td>Planning Rationale</td><td>Briefly explain the reasoning behind the selected plan. The explanation should be grounded in observable scene elements or ego vehicle dynamics and should not introduce unobserved information.</td></tr><tr><td colspan="2">Output Constraint</td></tr><tr><td>Output Format</td><td>Produce exactly five output fields in the following order: [scene description], [risk level], [trajectory intention], [high-level plan], and [plan rationale]. No additional information should be included beyond these fields.</td></tr></tbody></table>

As summarized in Table I, our prompt design is structured to guide the VLM to act as an autonomous driving planner specialized in safety-critical long-tail scenario analysis. The prompt explicitly instructs the model to identify rare hazardous elements from multi-view observations and to generate risk-aware planning strategies grounded in both perception and motion context.

The prompt specification consists of a fixed input description and structured semantic outputs corresponding to long-tail scene understanding and planning guidance. For the input, the model is provided with multi-view surround images arranged into four grouped inputs with a fixed ordering, together with an ego-centric coordinate frame in which the forward and left directions are consistently defined. This explicit specification reduces viewpoint ambiguity and helps prevent incorrect spatial interpretation. In addition, historical ego vehicle motion states are supplied in the same coordinate frame, including past positions, velocities, and accelerations over multiple timesteps, along with a high-level driving intent, enabling the model to reason about temporal motion patterns and dynamic behaviors.

Based on these inputs, the prompt elicits two categories of semantic outputs. The first corresponds to the Long-Tail Scene Context, which requires the model to produce a multi-view description of the driving scene organized by viewpoint, including front, rear, side-left, and side-right observations. The scene description covers both basic traffic elements and long-tail hazards, such as occlusions, abnormal agent behaviors, rare objects, adverse environmental conditions, and compound edge cases involving multiple simultaneous risks. This structured representation ensures comprehensive coverage of spatially distributed hazards that may only be visible from specific perspectives.

The second category corresponds to the Long-Tail Planning Context, which captures decision-level semantic guidance. Specifically, the model is required to assess the overall safety risk of the scenario, infer a high-level maneuver intention, generate a concise planning directive, and provide a brief rationale grounded in observable scene elements or ego vehicle dynamics. Together, these outputs encode risk-aware planning knowledge that complements geometric trajectory supervision.

Finally, all outputs are constrained to follow a fixed structured format consisting of exactly five fields, ensuring consistency across annotations and reliable downstream integration. This prompt design forms the foundation of our long-tail instruction dataset, which enables the generation of high-quality annotations that capture both perceptual context and planning intent. These annotations are subsequently encoded into text embeddings and used to guide the student model during training and inference, effectively distilling VLM reasoning into an efficient end-to-end planning framework.

### III-C Long-Tail Instruction Embedding

The Long-Tail Instruction Embedding module provides semantic supervision that guides the student model in safety-critical and long-tail driving scenarios. As illustrated in Figure 2, using the crafted prompt template described in the above section, this module leverages a cloud-based VLM to analyze 8-camera surround-view images together with historical vehicle motion states and high-level intent, producing structured semantic instruction annotations that capture both environmental hazards and planning considerations.

The generated textual annotations, comprising the Long-Tail Scene Context and Long-Tail Planning Context, are subsequently encoded into fixed-dimensional vector representations using BGE-M3 [^7], a strong and efficient sentence embedding model that produces semantically discriminative representations. Specifically, the Long-Tail Scene Context is encoded as a scene embedding denoted as scene\_emb $\in\mathbb{R}^{B\times D_{\text{text}}}$, while the Long-Tail Planning Context is encoded as a planning embedding denoted as risk\_plan\_emb $\in\mathbb{R}^{B\times D_{\text{text}}}$, where $B$ is the batch size and $D_{\text{text}}$ is the embedding dimension of the text encoder.

Both text embeddings serve as compact semantic representations of long-tail knowledge and are treated as fixed inputs to the student model. As shown in Figure 2, scene\_emb and risk\_plan\_emb are passed to dedicated text projection modules in the Tri-Modal Driving Module, where they are mapped into the model’s latent space and fused with visual and temporal features for downstream trajectory planning.

### III-D Tri-Modal Driving Module

The Tri-Modal Driving Module serves as the student component in our framework and performs real-time trajectory planning by fusing visual, temporal, and linguistic modalities. As illustrated in Figure 2, this module processes multi-view camera images, historical vehicle states, high-level driving intent, and pre-computed long-tail instruction embeddings through a sequential pipeline of specialized components, ultimately generating safe and risk-informed trajectories. The module comprises the following key components.

Vision Encoder. The Vision Encoder adopts a Vision Transformer (ViT) architecture [^9] to extract spatial features from 8-camera surround-view images. Patch tokens produced by the transformer backbone, excluding the classification token, are projected to the model latent dimension $D_{\text{model}}$ and augmented with learnable camera-specific and viewpoint-specific embeddings to encode multi-view spatial priors. The resulting visual token representations are denoted by $\mathit{v_{\text{tokens}}}\in\mathbb{R}^{B\times N\times D_{\text{model}}}$, and a global visual feature $\mathit{v_{\text{global}}}\in\mathbb{R}^{B\times D_{\text{model}}}$ is obtained via mean pooling over all visual tokens.

State Encoder. The State Encoder summarizes recent vehicle motion dynamics using a lightweight transformer-based temporal encoder. Specifically, the most recent 16 frames of historical vehicle states, including position, velocity, and acceleration, are first projected into the latent space through a linear layer and then processed by a multi-layer transformer encoder to capture temporal dependencies. The resulting temporal features are aggregated via mean pooling across the time dimension, yielding a compact temporal representation $\mathit{s_{\text{global}}}\in\mathbb{R}^{B\times D_{\text{model}}}$.

Scene Fusion. The embedded Long-Tail Scene Context scene\_emb is first mapped into the model latent space through the Scene Text Projector, which is implemented as a single linear projection that maps scene\_emb to scene\_vec $\in\mathbb{R}^{B\times D_{\text{model}}}$. The Scene Fusion module then performs text-guided aggregation over visual tokens using cross-attention, where scene\_vec serves as the query and $\mathit{v_{\text{tokens}}}$ serve as keys and values. This operation produces a text-conditioned visual summary, as formulated below:

$$
\mathit{c}_{\text{scene}}=\text{Attn}(\text{scene\_vec},\mathit{v_{\text{tokens}}}),
$$

which captures scene-relevant visual information aligned with the semantic description. The attended feature is then concatenated with $\mathit{v_{\text{global}}}$ and processed by a two-layer MLP [^25] to obtain an intermediate scene feature:

$$
\tilde{\mathit{s}}=\text{MLP}\!\left(\left[\mathit{v_{\text{global}}}\,\|\,\mathit{c}_{\text{scene}}\right]\right).
$$

Finally, a residual connection toward the global visual feature is applied to produce the scene-aware context:

$$
\text{scene\_context}=\mathit{v_{\text{global}}}+\tilde{\mathit{s}},
$$

where $\text{scene\_context}\in\mathbb{R}^{B\times D_{\text{model}}}$ serves as a compact visual representation conditioned on long-tail scene semantics.

Planning Context Fusion. The scene-aware representation scene\_context is combined with the temporal state summary $\mathit{s_{\text{global}}}$ through the Planning Context Fusion module, implemented as a two-layer MLP, yielding the planning context:

$$
\text{plan\_context}=\text{MLP}\!\left(\left[\text{scene\_context}\,\|\,\mathit{s_{\text{global}}}\right]\right).
$$

The resulting planning context $\text{plan\_context}\in\mathbb{R}^{B\times D_{\text{model}}}$ provides a unified representation that integrates scene-level semantics with historical motion dynamics.

Intent Modulator. To incorporate high-level driving intent, the Intent Modulator applies a feature-wise adaptive transformation conditioned on a discrete intent signal. As shown in Figure 3, given the planning context plan\_context and an intent identifier $\mathit{intent}\in\{0,1,2,3\}$, the intent is embedded and transformed to produce scale and shift parameters. The intent-aware planning context is computed as follows:

$$
\text{plan\_context\_intent}=\text{plan\_context}\odot\sigma(\mathit{scale})+\mathit{shift},
$$

where $\sigma(\cdot)$ is the sigmoid function and $\odot$ denotes element-wise multiplication. This mechanism enables adaptive modulation of planning features according to intended maneuvers.

![[x3 7.png|Refer to caption]]

Figure 3: Architecture of the Intent Modulator. High-level driving intent is embedded and processed through a two-layer MLP to generate scale and shift parameters, which adaptively modulate the planning context.

Risk Planning Cross-Attention.

Integrating VLM-generated planning instructions into trajectory prediction requires careful control to avoid over-reliance on potentially noisy or overly conservative semantic guidance. To address this challenge, we design the Risk Planning Cross-Attention module, which conditions the planning representation on semantic risk information through a controlled cross-attention mechanism.

Figure 4 illustrates the detailed architecture of the Risk Planning Cross-Attention module. Given the intent-aware planning context plan\_context\_intent $\in\mathbb{R}^{B\times D_{\text{model}}}$ and the planning instruction representation projected by the Planning Text Projector from risk\_plan\_emb, a multi-head cross-attention operation is applied, where the planning context serves as the query and the projected instruction embedding $\text{risk\_vec}\in\mathbb{R}^{B\times D_{\text{model}}}$ serves as the key and value. This operation produces an attention output $\mathit{c}\in\mathbb{R}^{B\times D_{\text{model}}}$ that captures risk-relevant semantic cues aligned with the current planning state.

To ensure robustness, the attention output is not directly substituted into the planning representation. Instead, it is integrated through a residual connection scaled by a control parameter $\alpha$, followed by layer normalization. This design explicitly limits the influence of semantic guidance and stabilizes optimization. Formally, the risk-aware planning representation is computed as:

$$
\text{plan\_context\_ctrl}=\text{LN}\!\left(\text{plan\_context\_intent}+\alpha\,\mathit{c}\right),
$$

where $\alpha$ controls the strength of semantic conditioning. By modulating semantic guidance through scaled residual integration, the Risk Planning Cross-Attention module allows the model to selectively incorporate risk-aware planning information in safety-critical long-tail scenarios, while preventing semantic instructions from overwhelming learned motion priors in common driving conditions.

![[x4 6.png|Refer to caption]]

Figure 4: Architecture of the Risk Planning Cross-Attention module. The intent-aware planning context attends to the planning instruction embedding through multi-head cross-attention, and the resulting semantic adjustment is integrated via a scaled residual connection followed by layer normalization.

Temporal Decoder. Finally, the Temporal Decoder adopts a query-based transformer decoder to generate future trajectories. Learnable temporal queries attend to the combined memory formed by plan\_context\_ctrl, producing a sequence of relative displacements that are refined through an MLP and accumulated over time to yield the final trajectory $\text{traj}\in\mathbb{R}^{B\times T_{\text{future}}}$.

## IV Experiment

### IV-A Dataset

We build our long-tail scenario annotations on top of the WOD-E2E dataset [^38], a latest large-scale benchmark for learning-based autonomous driving under realistic and safety-critical conditions, and evaluate the proposed HERMES on the annotated benchmark. WOD-E2E contains 4,021 driving segments collected from diverse urban and suburban environments, covering approximately 12 hours of real-world driving. The official training, validation, and testing split consists of 2,037, 479, and 1,505 segments, respectively. WOD-E2E emphasizes rare and safety-critical long-tail scenarios such as vulnerable road users, occlusions, unusual agent behaviors, and other challenging conditions that are underrepresented in standard driving datasets. This focus makes it a strong foundation for long-tail analysis and modeling in end-to-end autonomous driving, especially under complex and mixed traffic conditions.

### IV-B Implementation Details

We implement HERMES in PyTorch, and conduct all experiments on a single NVIDIA RTX 4090 GPU with a batch size of 16. The model is trained for 3 epochs. The training objective is the Mean Squared Error (MSE) between the predicted and ground-truth future ego trajectories over a 5-second horizon, corresponding to 20 future timesteps. We use the Adam optimizer with an initial learning rate of $1\times 10^{-4}$. Training employs gradient clipping with an $\ell_{2}$ norm threshold of 5.0, along with a linear warmup schedule over the first epoch starting from $1\times 10^{-5}$, followed by cosine annealing to a minimum learning rate of $5\times 10^{-6}$. Within the Risk Planning Cross-Attention module, the control parameter $\alpha$ is set to 0.3 in all experiments.

Input images are resized to $224\times 224$ and normalized using ImageNet statistics. Qwen3-VL-Flash [^2] is applied for long-tail reasoning and annotation. Textual instruction embeddings are obtained through the BGE-M3 text encoder [^7]. All components of the Tri-Modal Driving Module, including the vision backbone and downstream fusion and decoding modules, are trained end-to-end.

### IV-C Baselines

We compare Hermes against three representative end-to-end autonomous driving models: UniAD [^18], VAD [^22], and LightEMMA [^26]. These baselines are selected to cover planning-oriented transformer-based models as well as recent VLM driven end-to-end frameworks.

LightEMMA provides a publicly available, modular framework with pre-configured inference pipelines for multiple VLMs. As a result, LightEMMA is evaluated in a zero-shot manner on the annotated WOD-E2E benchmark without task-specific fine-tuning. In contrast, UniAD and VAD were originally designed and evaluated on datasets with input and output specifications that differ from the WOD-E2E format. To enable meaningful comparison, we perform limited adaptation and training for both models following their original implementations as closely as possible. These adaptations are restricted to data interface alignment and do not alter the core model architectures or training objectives.

Due to access restrictions of the official test set, all baseline evaluations are conducted on the WOD-E2E validation split under identical data preprocessing and evaluation protocols.

### IV-D Metrics

We evaluate end-to-end driving performance using a combination of the Rater Feedback Score (RFS) [^38] and standard trajectory distance metrics, including Average Displacement Error (ADE) [^15] and Final Displacement Error (FDE) [^15]. Specifically, RFS is the primary evaluation metric for the WOD-E2E benchmark, especially designed to assess driving quality in safety-critical and multimodal long-tail scenarios. Unlike conventional open-loop metrics that measure distance to a single ground-truth trajectory, RFS evaluates how well a predicted trajectory aligns with multiple human-rated reference trajectories.

### IV-E Performance Evaluation

TABLE II: Overall performance comparison

| Method | RFS $\uparrow$ | ADE@3s $\downarrow$ | ADE@5s $\downarrow$ | FDE@3s $\downarrow$ | FDE@5s $\downarrow$ |
| --- | --- | --- | --- | --- | --- |
| UniAD [^18] | 5.78 | 6.50 | 10.81 | 12.14 | 21.41 |
| VAD [^22] | 4.45 | 3.19 | 5.81 | 5.85 | 12.30 |
| LightEMMA [^26] | 6.11 | 3.07 | 5.41 | 6.07 | 11.32 |
| HERMES (Ours) | 6.81 | 0.82 | 1.82 | 1.73 | 4.81 |

#### IV-E1 Overall Results

Table II summarizes the overall performance of HERMES. It shows that HERMES achieves the best performance among all baselines, attaining the highest RFS score while consistently reducing ADE and FDE at both the 3-second and 5-second horizons.

In particular, the improvement in RFS indicates that HERMES better aligns with human-preferred and safety-aware driving behaviors in long-tail scenarios. While ADE and FDE measure geometric trajectory accuracy, the consistent gains in RFS highlight the benefit of incorporating semantic long-tail guidance into end-to-end planning.

TABLE III: Category-wise RFS comparison. The ten scenario categories include Interaction, Construction, Cyclists, Pedestrian, Single-Lane Maneuvers, Multi-Lane Maneuvers, Special Vehicles, Cut-ins, Others, and Foreign Object Debris (FOD).

| Method | Global | Interaction | Construction | Cyclists | Pedestrian | Single-Lane | Multi-Lane | Special Vehicles | Cut-ins | Others | FOD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UniAD [^18] | 5.78 | 6.08 | 6.07 | 5.50 | 6.45 | 5.78 | 5.82 | 6.04 | 4.64 | 5.91 | 5.23 |
| VAD [^22] | 4.45 | 4.42 | 4.11 | 4.40 | 4.66 | 4.68 | 5.46 | 4.30 | 4.50 | 4.33 | 4.30 |
| LightEMMA [^26] | 6.11 | 6.32 | 6.55 | 5.83 | 6.12 | 5.24 | 6.38 | 6.18 | 6.00 | 6.44 | 6.16 |
| HERMES (Ours) | 6.81 | 6.96 | 7.26 | 6.78 | 7.11 | 7.28 | 6.71 | 6.72 | 6.54 | 6.41 | 6.37 |

TABLE IV: Results of Ablation study

| Method | RFS $\uparrow$ | ADE@3s $\downarrow$ | ADE@5s $\downarrow$ | FDE@3s $\downarrow$ | FDE@5s $\downarrow$ |
| --- | --- | --- | --- | --- | --- |
| No Instruction | 6.21 | 0.92 | 1.94 | 1.98 | 4.82 |
| No Intent | 6.56 | 0.91 | 1.94 | 2.00 | 4.82 |
| No State | 5.51 | 2.48 | 4.22 | 4.69 | 8.59 |
| Base (HERMES) | 6.81 | 0.82 | 1.82 | 1.73 | 4.81 |

#### IV-E2 Category-wise RFS Analysis

Table III reports category-wise RFS results, which covers ten scenario categories that reflect diverse long-tail driving conditions, including interaction-heavy scenes, vulnerable road users, rare obstacles, and complex lane-level maneuvers.

Again, HERMES achieves the highest global RFS and outperforms baseline methods in the majority of categories. Notably, clear improvements are observed in categories that require semantic reasoning beyond local motion patterns, such as construction zones, pedestrians, cyclists, and multi-lane maneuvers. These scenarios often involve ambiguous right-of-way, temporary road structures, or complex interactions, where long-tail semantic guidance plays a critical role.

Although HERMES does not dominate every individual category, it maintains consistently strong performance across all scenario types and achieves the best overall RFS. This result suggests that incorporating long-tail semantic instructions improves robustness across diverse driving conditions without overfitting to specific scenario categories.

### IV-F Ablation Study

We conduct an ablation study to analyze the contribution of textual instructions, high-level intent, and historical state information in the proposed tri-modal driving framework. All ablation models are evaluated on the same annotated validation set using identical training and evaluation protocols as the full model.

#### IV-F1 No Instruction

In this setting, we remove the entire Long-Tail Instruction Embedding module and its related downstream pathway, including the Scene Text Projector, Scene Fusion, Planning Text Projector, and Risk Planning Cross-Attention modules. The planning representation is obtained by directly fusing visual features from the Vision Encoder and temporal features from the State Encode, followed by intent modulation and temporal decoding.

As shown in Table IV, removing textual instructions leads to a noticeable degradation in performance, with RFS dropping from 6.81 to 6.21. Trajectory accuracy also degrades consistently across all ADE and FDE metrics. This result indicates that long-tail semantic instructions provide necessary and invaluable guidance for improving human-aligned planning behavior beyond purely visual and temporal cues.

#### IV-F2 No Intent

This variant removes the Intent Modulator while retaining visual, state, and textual instruction inputs. The planning context produced by Planning Context Fusion is directly passed to the Risk Planning Cross-Attention module without intent conditioning.

Removing intent information results in a moderate performance drop, with RFS decreasing to 6.56. While the degradation is smaller than that observed in the No Instruction setting, both ADE and FDE increase, suggesting that explicit intent conditioning helps align the planning representation with high-level maneuver objectives and improves trajectory consistency.

#### IV-F3 No State

In this ablation, the State Encoder is removed, and historical motion information is no longer available. Consequently, Planning Context Fusion is omitted, and the scene-aware representation produced by Scene Fusion is directly fed into the Intent Modulator and subsequent modules.

This setting leads to the most significant performance degradation, with RFS dropping to 5.51 and substantial increases in all distance-based metrics. The sharp decline highlights the critical role of historical state information in capturing ego dynamics and ensuring stable trajectory prediction, especially in complex long-tail scenarios.

#### IV-F4 Discussion

Overall, the ablation results demonstrate that all three components contribute to the final performance. Textual instructions primarily improve semantic alignment and human-preferred behavior, intent information refines maneuver-level planning, and historical state provides essential temporal context for stable trajectory generation. The full model achieves the best balance across all metrics by jointly leveraging these complementary signals.

![[x5 6.png|Refer to caption]]

Figure 5: Nighttime driving under heavy rain and poor visibility. Planned trajectory is shown in red.

![[x6 5.png|Refer to caption]]

Figure 6: Extremely low-visibility residential street with wet road conditions. Planned trajectory is shown in red.

### IV-G Qualitative Analysis

In this section, we present qualitative examples to illustrate how HERMES leverages long-tail semantic instructions to produce robust and interpretable planning behaviors under challenging driving conditions.

#### IV-G1 Case 1: Adverse Weather and Low Visibility

This scenario depicts nighttime driving under heavy rain, where reflections on wet road surfaces and reduced illumination significantly degrade visual cues. Guided by the Long-Tail Scene Context identifying low visibility and adverse weather conditions, HERMES generates a conservative and smooth turning trajectory with controlled speed, avoiding abrupt steering or acceleration. This example highlights the model’s ability to incorporate risk-aware semantic guidance when visual information alone is unreliable.

#### IV-G2 Case 2: Residential Street under Extremely Degraded Visual Conditions

In this example, the ego vehicle navigates a residential street with significantly limited visibility and wet pavement. Although no immediate obstacle is present, the scene requires cautious planning due to extremely dark scene, potential pedestrians and reduced friction. HERMES produces a stable and centered trajectory, reflecting a risk-aware driving strategy that balances progress and safety.

#### IV-G3 Case 3: Construction Zone with Lane Channelization

This scenario contains a work zone with barricades and temporary lane markings, resulting in an uncommon road layout. Such configurations are sparse in standard driving data and often challenge end-to-end planners. By leveraging the Long-Tail Planning Context, HERMES anticipates the upcoming lane change and generates a smooth lateral maneuver while maintaining a controlled speed.

##### Case 4: Complex Urban Intersection.

This example illustrates a wide urban intersection with crosswalks and potential pedestrian interactions. Although the immediate path is unobstructed, the scene demands cautious behavior due to latent risks. HERMES generates a stable straight trajectory while maintaining readiness to yield, demonstrating appropriate planning behavior in moderately risky urban environments.

![[x7 5.png|Refer to caption]]

Figure 7: Work-zone scenario with lane channelization caused by construction. Planned trajectory is shown in red.

![[x8 4.png|Refer to caption]]

Figure 8: Complex urban intersection with crosswalks and surrounding traffic. Planned trajectory is shown in red.

## V Conclusion

In this paper, we introduce HERMES, a holistic risk-aware end-to-end multimodal autonomoous driving framework designed to generate safer and more accurate trajectories in safety-critical long-tail scenarios. We introduce a foundation-model-assisted long-tail annotation pipeline that delivers structured scene and planning context, and leverage these signals to guide end-to-end trajectory generation. Building on this, HERMES employs a tri-modal planning architecture that fuses multi-view perception, historical motion cues, and semantic guidance through risk- and intent-aware conditioning, which improves safety alignment while preserving motion feasibility. Experiments on the long-tail WOD-E2E benchmark demonstrate that HERMES outperforms representative end-to-end and VLM-driven baselines on long-tail planning.

Future work will focus on improving annotation reliability under ambiguity, and on extending evaluation to closed-loop settings with explicit safety constraints. Moreover, integrating long-tail scenario generation and world-model-based simulation can further strengthen the model’s robustness to complex and rapidly changing conditions by synthesizing rare hazards and enabling scalable stress testing of risk-aware planning.

## References

|  | Weizhe Tang is currently pursuing the Ph.D. degree at the University of Wisconsin–Madison, under the supervision of Prof. Bin Ran. His research interests include autonomous driving, end-to-end trajectory planning, multimodal learning, and long-tail scenario modeling, with an emphasis on integrating semantic reasoning and vision–language models into safety-critical transportation systems. |
| --- | --- |

|  | Junwei You is currently a Ph.D. candidate in the Department of Civil and Environmental Engineering at the University of Wisconsin–Madison. He received the M.S. degree in civil and environmental engineering from Northwestern University in 2022. His research interests are end-to-end autonomous driving, V2X cooperative autonomous driving, multimodal foundation models, and intelligent transportation systems. |
| --- | --- |

|  | Jiaxi Liu earned both his B.E. and M.E. degrees in Mechanical Engineering from the School of Vehicle and Mobility at Tsinghua University. Currently, he is pursuing a Ph.D. in the Department of Civil and Environmental Engineering at the University of Wisconsin-Madison. His research interests focus on cooperative perception, vehicle-road-cloud integration systems, and LLM-assisted autonomous driving |
| --- | --- |

|  | Zhaoyi Wang is an incoming PhD student at the University of Wisconsin–Madison. He received his M.S. degree from the School of Automotive Studies, Tongji University in 2025 and his B.S. degree from the College of Automotive Engineering, Jilin University in 2021. His research interests include autonomous driving, vision-language models, diffusion models, and reinforcement learning. |
| --- | --- |

|  | Rui Gan is currently pursuing the Ph.D. degree in Transportation Engineering at the University of Wisconsin-Madison, Madison, WI, USA, under the supervision of Prof. Bin Ran. His research interests include autonomous driving safety, trajectory prediction, multi-agent systems, vision-language models for transportation applications, and connected vehicle technologies. |
| --- | --- |

|  | Zilin Huang is a Ph.D. candidate in the Department of Civil and Environmental Engineering at the University of Wisconsin–Madison, WI, USA. Prior to joining UW–Madison, he worked at the Center for Connected and Automated Transportation (CCAT) at Purdue University, IN, USA. He received his M.S. degree in Communication and Transportation Engineering from South China University of Technology in 2021, and his B.S. degree from the School of Electromechanical Engineering, Guangdong University of Technology in 2018. His research interests include human-centered AI, autonomous driving, human–AI collaboration, robotics, and intelligent transportation. More information is available at: www.huang-zilin.com |
| --- | --- |

|  | Feng Wei received his Ph.D. in Electronic Information from Chongqing University in 2025, and his M.S. degree in Software Engineering from the University of Science and Technology of China in 2010. His research interests include electronic information systems, artificial general intelligence, software engineering, and enterprise informatization. |
| --- | --- |

|  | Bin Ran is the Vilas Distinguished Achievement Professor and Director of ITS Program at the University of Wisconsin-Madison. Dr. Ran is an expert in dynamic transportation network models, traffic simulation and control, traffic information system, Internet of Mobility, and Connected Automated Vehicle Highway (CAVH) System. He has led the development and deployment of various traffic information systems and the demonstration of CAVH systems. Dr. Ran is the author of two leading textbooks on dynamic traffic networks. He has co-authored more than 240 journal papers and more than 260 referenced papers at national and international conferences. He holds more than 20 patents of CAVH in the U.S. and other countries. He is an associate editor of Journal of Intelligent Transportation Systems. |
| --- | --- |

[^1]: C. Badue, R. Guidolini, R. V. Carneiro, P. Azevedo, V. B. Cardoso, A. Forechi, L. Jesus, R. Berriel, T. M. Paixao, F. Mutz, et al. (2021) Self-driving cars: a survey. Expert Systems with Applications 165, pp. 113816. Cited by: §I.

[^2]: S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, W. Ge, Z. Guo, Q. Huang, J. Huang, F. Huang, B. Hui, S. Jiang, Z. Li, M. Li, M. Li, K. Li, Z. Lin, J. Lin, X. Liu, J. Liu, C. Liu, Y. Liu, D. Liu, S. Liu, D. Lu, R. Luo, C. Lv, R. Men, L. Meng, X. Ren, X. Ren, S. Song, Y. Sun, J. Tang, J. Tu, J. Wan, P. Wang, P. Wang, Q. Wang, Y. Wang, T. Xie, Y. Xu, H. Xu, J. Xu, Z. Yang, M. Yang, J. Yang, A. Yang, B. Yu, F. Zhang, H. Zhang, X. Zhang, B. Zheng, H. Zhong, J. Zhou, F. Zhou, J. Zhou, Y. Zhu, and K. Zhu (2025) Qwen3-vl technical report. arXiv preprint arXiv:2511.21631. Cited by: §IV-B.

[^3]: M. Bijelic, T. Gruber, F. Mannan, F. Kraus, W. Ritter, K. Dietmayer, and F. Heide (2020) Seeing through fog without seeing fog: deep multimodal sensor fusion in unseen adverse weather. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Cited by: §II-B.

[^4]: M. Bojarski, D. Del Testa, D. Dworakowski, B. Firner, B. Flepp, P. Goyal, L. D. Jackel, M. Monfort, U. Muller, J. Zhang, et al. (2016) End to end learning for self-driving cars. arXiv preprint arXiv:1604.07316. Cited by: §I.

[^5]: Y. Chai, B. Sapp, M. Bansal, and D. Anguelov (2019) MultiPath: multiple probabilistic anchor trajectory hypotheses for behavior prediction. In Conference on Robot Learning (CoRL), pp. 86–99. Cited by: §I.

[^6]: C. Chen, A. Seff, A. Kornhauser, and J. Xiao (2015) DeepDriving: learning affordance for direct perception in autonomous driving. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 2722–2730. Cited by: §I.

[^7]: J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian, and Z. Liu (2024) BGE m3-embedding: multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216. Cited by: §III-C, §IV-B.

[^8]: D. Dai and L. Van Gool (2018) Dark model adaptation: semantic image segmentation from daytime to nighttime. In 2018 IEEE Intelligent Transportation Systems Conference (ITSC), Cited by: §II-B.

[^9]: A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby (2021) An image is worth 16x16 words: transformers for image recognition at scale. In International Conference on Learning Representations (ICLR), Cited by: §III-D.

[^10]: T. Feng, W. Wang, and Y. Yang (2025) A survey of world models for autonomous driving. arXiv preprint arXiv:2501.11260. Cited by: §II-B.

[^11]: R. Gan, P. Li, K. Long, B. An, J. You, K. Wu, and B. Ran (2025) Planning safety trajectories with dual-phase, physics-informed, and transportation knowledge-driven large language models. arXiv preprint arXiv:2504.04562. Cited by: §II-A.

[^12]: R. Gan, H. Shi, P. Li, K. Wu, B. An, J. You, L. Li, J. Ma, C. Ma, and B. Ran (2025) Goal-based neural physics vehicle trajectory prediction model. Transportation Research Part C: Emerging Technologies 179, pp. 105283. Cited by: §I.

[^13]: J. Gao, C. Sun, H. Zhao, Y. Shen, D. Anguelov, C. Li, and C. Schmid (2020) VectorNet: encoding hd maps and agent dynamics from vectorized representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11525–11533. Cited by: §I.

[^14]: A. Ghosh, S. Zheng, R. Tamburo, et al. (2025) ROADWork: a dataset and benchmark for learning to recognize, observe, analyze and drive through work zones. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), Cited by: §II-B.

[^15]: A. Gupta, J. Johnson, L. Fei-Fei, S. Savarese, and A. Alahi (2018) Social gan: socially acceptable trajectories with generative adversarial networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Cited by: §IV-D.

[^16]: D. Hegde, R. Yasarla, H. Cai, S. Han, A. Bhattacharyya, S. Mahajan, L. Liu, R. Garrepalli, V. M. Patel, and F. Porikli (2025) Distilling multi-modal large language models for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Cited by: §II-B.

[^17]: A. Hu, L. Russell, H. Yeo, Z. Murez, G. Fedoseev, A. Kendall, J. Shotton, and G. Corrado (2023) GAIA-1: a generative world model for autonomous driving. arXiv preprint arXiv:2309.17080. Cited by: §II-B.

[^18]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, et al. (2023) Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 17853–17862. Cited by: §I, §IV-C, TABLE II, TABLE III.

[^19]: Z. Huang, Z. Sheng, C. Ma, and S. Chen (2024) Human as ai mentor: enhanced human-in-the-loop reinforcement learning for safe and efficient autonomous driving. Communications in Transportation Research 4, pp. 100127. Cited by: §I.

[^20]: Z. Huang, Z. Sheng, Y. Qu, J. You, and S. Chen (2025) Vlm-rl: a unified vision language models and reinforcement learning framework for safe autonomous driving. Transportation Research Part C: Emerging Technologies 180, pp. 105321. Cited by: §II-A.

[^21]: J. Hwang, R. Xu, H. Lin, W. Hung, J. Ji, K. Choi, D. Huang, T. He, P. Covington, B. Sapp, Y. Zhou, J. Guo, D. Anguelov, and M. Tan (2024) EMMA: end-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2410.23262. Cited by: §I, §II-A.

[^22]: B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang (2023) VAD: vectorized scene representation for efficient autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 8340–8350. Cited by: §I, §IV-C, TABLE II, TABLE III.

[^23]: C. Ma, H. Zhou, P. Zhang, K. Ma, H. Shi, and X. Li (2025) Safety assurance adaptive control for modular autonomous vehicles. Communications in Transportation Research 5, pp. 100204. Cited by: §I.

[^24]: J. Mao, Y. Qian, J. Ye, H. Zhao, and Y. Wang (2023) GPT-driver: learning to drive with gpt. In NeurIPS 2023 Workshop on Foundation Models for Decision Making, Cited by: §I, §II-A.

[^25]: M. Popescu, V. E. Balas, L. Perescu-Popescu, and N. Mastorakis (2009) Multilayer perceptron and neural networks. WSEAS Transactions on Circuits and Systems 8 (7), pp. 579–588. Cited by: §III-D.

[^26]: Z. Qiao, H. Li, Z. Cao, and H. X. Liu (2025) LightEMMA: lightweight end-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2505.00284. Cited by: §II-A, §IV-C, TABLE II, TABLE III.

[^27]: C. Sakaridis, D. Dai, and L. Van Gool (2019) Guided curriculum model adaptation and uncertainty-aware evaluation for semantic nighttime image segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), Cited by: §II-B.

[^28]: C. Sakaridis, D. Dai, and L. Van Gool (2021) ACDC: the adverse conditions dataset with correspondences for semantic driving scene understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), Cited by: §II-B.

[^29]: W. Schwarting, J. Alonso-Mora, and D. Rus (2018) Planning and decision-making for autonomous vehicles. Annual Review of Control, Robotics, and Autonomous Systems 1, pp. 187–210. Cited by: §I.

[^30]: H. Shao, Y. Hu, L. Wang, G. Song, S. L. Waslander, Y. Liu, and H. Li (2024) LMDrive: closed-loop end-to-end driving with large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15120–15130. Cited by: §I, §II-A.

[^31]: H. Shao, L. Wang, R. Chen, H. Li, and Y. Liu (2022) Safety-enhanced autonomous driving using interpretable sensor fusion transformer. In Conference on Robot Learning (CoRL), pp. 726–737. Cited by: §I.

[^32]: C. Sima, K. Renz, K. Chitta, L. Chen, H. Zhang, C. Xie, P. Luo, A. Geiger, and H. Li (2024) DriveLM: driving with graph visual question answering. In European Conference on Computer Vision (ECCV), Cited by: §I, §II-A.

[^33]: S. Thrun, M. Montemerlo, H. Dahlkamp, D. Stavens, A. Aron, J. Diebel, P. Fong, J. Gale, M. Halpenny, G. Hoffmann, et al. (2006) Stanley: the robot that won the darpa grand challenge. Journal of Field Robotics 23 (9), pp. 661–692. Cited by: §I.

[^34]: X. Tian, J. Gu, B. Li, Y. Liu, C. Hu, Y. Wang, K. Zhan, P. Jia, X. Lang, and H. Zhao (2024) DriveVLM: the convergence of autonomous driving and large vision-language models. arXiv preprint arXiv:2402.12289. Cited by: §I.

[^35]: C. Urmson, J. Anhalt, D. Bagnell, C. Baker, R. Bittner, M. Clark, J. Dolan, D. Duggins, T. Galatali, C. Geyer, et al. (2008) Autonomous driving in urban environments: boss and the urban challenge. Journal of Field Robotics 25 (8), pp. 425–466. Cited by: §I.

[^36]: Y. Wang, W. Luo, J. Bai, Y. Cao, T. Che, K. Chen, Y. Chen, J. Diamond, Y. Ding, W. Ding, L. Feng, G. Heinrich, J. Huang, P. Karkus, B. Li, P. Li, T. Lin, D. Liu, M. Liu, L. Liu, Z. Liu, J. Lu, Y. Mao, P. Molchanov, L. Pavao, Z. Peng, M. Ranzinger, E. Schmerling, S. Shen, Y. Shi, S. Tariq, R. Tian, T. Wekel, X. Weng, T. Xiao, E. Yang, X. Yang, Y. You, X. Zeng, W. Zhang, B. Ivanovic, and M. Pavone (2025) Alpamayo-r1: bridging reasoning and action prediction for generalizable autonomous driving in the long tail. arXiv preprint arXiv:2511.00088. Cited by: §II-B.

[^37]: S. Xing, C. Qian, Y. Wang, H. Hua, K. Tian, Y. Zhou, and Z. Tu (2025) Openemma: open-source multimodal model for end-to-end autonomous driving. In Proceedings of the Winter Conference on Applications of Computer Vision, pp. 1001–1009. Cited by: §II-A.

[^38]: R. Xu, H. Lin, W. Jeon, H. Feng, Y. Zou, L. Sun, J. Gorman, K. Tolstaya, S. Tang, B. White, B. Sapp, M. Tan, J. Hwang, and D. Anguelov (2025) WOD-e2e: waymo open dataset for end-to-end driving in challenging long-tail scenarios. arXiv preprint arXiv:2510.26125. External Links: [Link](https://arxiv.org/abs/2510.26125) Cited by: 2nd item, §III-B, §IV-A, §IV-D.

[^39]: Z. Xu, Y. Zhang, E. Xie, Z. Zhao, Y. Guo, K. K. Wong, Z. Li, and H. Zhao (2023) DriveGPT4: interpretable end-to-end autonomous driving via large language model. arXiv preprint arXiv:2310.01412. Cited by: §I, §II-A.

[^40]: J. You, R. Gan, W. Tang, Z. Huang, J. Liu, Z. Jiang, H. Shi, K. Wu, K. Long, S. Fu, et al. (2025) Followgen: a scaled noise conditional diffusion model for car-following trajectory prediction. Communications in Transportation Research 5, pp. 100215. Cited by: §I.

[^41]: J. You, Z. Jiang, Z. Huang, H. Shi, R. Gan, K. Wu, X. Cheng, X. Li, and B. Ran (2026) V2x-vlm: end-to-end v2x cooperative autonomous driving through large vision-language models. Transportation Research Part C: Emerging Technologies 183, pp. 105457. Cited by: §II-A.

[^42]: J. You, P. Li, Z. Jiang, Z. Huang, R. Gan, H. Shi, and B. Ran (2025) V2X-realm: vision-language model-based robust end-to-end cooperative autonomous driving with adaptive long-tail modeling. arXiv preprint arXiv:2506.21041. Cited by: §II-A.

[^43]: E. Yurtsever, J. Lambert, A. Carballo, and K. Takeda (2020) A survey of autonomous driving: common practices and emerging technologies. IEEE Access 8, pp. 58443–58469. Cited by: §I.

[^44]: G. Zhao, X. Wang, et al. (2024) DriveDreamer-2: llm-enhanced world models for diverse driving video generation. arXiv preprint arXiv:2403.06845. Cited by: §II-B.

[^45]: H. Zhao, J. Gao, T. Lan, C. Sun, B. Sapp, B. Varadarajan, Y. Shen, Y. Shen, Y. Chai, C. Schmid, et al. (2021) TNT: target-driven trajectory prediction. In Conference on Robot Learning (CoRL), pp. 895–904. Cited by: §I.

[^46]: X. Zhou, X. Han, F. Yang, Y. Ma, V. Tresp, and A. Knoll (2025) OpenDriveVLA: towards end-to-end autonomous driving with large vision language action model. arXiv preprint arXiv:2503.23463. Cited by: §II-A.

[^47]: Z. Zhou, T. Cai, S. Z. Zhao, Y. Zhang, Z. Huang, B. Zhou, and J. Ma (2025) AutoVLA: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. arXiv preprint arXiv:2506.13757. Cited by: §II-A.