---
title: "RAF-VLA: Representation Alignment with the Future for End-to-End Autonomous Driving"
source: "https://arxiv.org/html/2609.17728v1"
author:
published:
created: 2026-10-19
description:
tags:
  - "clippings"
---
Dogun Kim Affiliation: Graduate School of Mobility, Korea Advanced Institute of Science & Technology (KAIST), Daejeon, Korea (dogun.kim@kaist.ac.kr)    Yongjae Lee Affiliation: Graduate School of Mobility, Korea Advanced Institute of Science & Technology (KAIST), Daejeon, Korea (dogun.kim@kaist.ac.kr)    Joonhee Lim Affiliation: Robotics Program, Korea Advanced Institute of Science & Technology (KAIST), Daejeon, Korea ✉: Corresponding author.    Yeina Lee Affiliation: Robotics Program, Korea Advanced Institute of Science & Technology (KAIST), Daejeon, Korea ✉: Corresponding author.    Junhyeok Park Affiliation: Graduate School of Mobility, Korea Advanced Institute of Science & Technology (KAIST), Daejeon, Korea (dogun.kim@kaist.ac.kr)    Moogeun Park Affiliation: Robotics Program, Korea Advanced Institute of Science & Technology (KAIST), Daejeon, Korea ✉: Corresponding author.    Dongsuk Kum Affiliation: Graduate School of Mobility, Korea Advanced Institute of Science & Technology (KAIST), Daejeon, Korea (dogun.kim@kaist.ac.kr)

###### Abstract

Recent Vision-Language-Action (VLA) models for autonomous driving have incorporated world modeling by predicting future driving scenes alongside driving actions, demonstrating strong planning performance. Future driving scenes are utilized as dense supervision, encouraging the policy to learn rich internal representations useful for planning. However, these World-Modeling VLAs rely on explicit future generation to learn such representations, thereby introducing two key limitations: additional training burden and inference latency. To address these limitations, we propose RAF-VLA (Representation Alignment with the Future), a VLA-based autonomous driving framework that shapes planning-relevant internal representations through direct guidance from future-frame representations. RAF-VLA employs Future-Aligned Supervised Fine-Tuning, in which a straightforward regularization aligns the policy’s hidden states with future-frame representations obtained from a pretrained world encoder while learning driving actions. This simple alignment allows RAF-VLA to avoid the training burden and inference latency associated with future generation. Extensive experiments on the NAVSIM benchmark show that RAF-VLA achieves competitive planning performance against state-of-the-art VLA planners with substantially fewer training samples seen. Moreover, RAF-VLA incurs only 3.8% training overhead and a negligible 1 ms inference overhead.

![[Uncaptioned image]](https://arxiv.org/html/2609.17728v1/RAF-VLA_Main_figure1.png)

Fig. 1: Overview of RAF-VLA. (a) Overall architecture of RAF-VLA, where future-frame representations directly guide the policy’s hidden states to shape planning-relevant representations. (b) RAF-VLA achieves competitive planning performance with substantially fewer training samples seen than recent VLA planners that learn to generate future scenes. (c) Two-stage training strategy consisting of Future-Aligned SFT followed by GRPO-based RFT.

## I INTRODUCTION

Vision-Language-Action (VLA) models have emerged as a promising approach to end-to-end autonomous driving [^1] by fine-tuning Vision-Language Models (VLMs) pretrained on Internet-scale data as driving policies [^2] [^3]. These models leverage VLMs’ world knowledge and reasoning capabilities to map visual observations and navigation commands to driving actions. Early VLA approaches directly predicted actions from current observations [^4] [^5] [^6] [^7], whereas subsequent studies introduced textual reasoning before action prediction to harness VLMs’ reasoning capabilities [^8] [^9] [^10] [^11] [^12] [^13]. However, these approaches compress continuous visual information into discrete language, which can obscure fine-grained spatial and temporal cues important for planning. Recent VLA approaches have incorporated world modeling into policy learning, predicting future driving scenes as images, depth, or occupancy alongside action prediction [^14] [^15] [^16] [^17] [^18] [^19] [^20]. These World-Modeling VLAs utilize future driving scenes that reflect spatiotemporal structure and driving dynamics as dense supervision. Such future supervision encourages the policy to learn rich internal representations useful for planning, and World-Modeling VLAs have demonstrated strong planning performance.

However, explicit future generation in existing World-Modeling VLAs introduces two key limitations: additional training burden and inference latency. First, these methods jointly learn autoregressive generation, diffusion-based generation, or dense future-scene prediction alongside action prediction, which requires optimizing these additional generative objectives [^15] [^16] [^17] [^18] [^19]. Second, predicting future scenes at inference requires additional steps, including sequential decoding or iterative denoising, imposing considerable latency overhead on real-time driving systems [^15] [^16] [^18] [^19]. Some methods bypass future generation during inference to avoid this overhead; however, the additional training burden remains [^17].

To address these limitations, we reconsider the reliance on explicit future generation when leveraging future-scene supervision to learn planning-relevant representations. Prior studies on generative modeling show that while generative objectives can induce useful internal representations that transfer to downstream tasks [^21] [^22] [^23] [^24] [^25] [^26] [^27], such internal representations can be shaped more directly through guidance from target representations [^28] [^29] [^30] [^31]. In driving policy learning, where the ultimate goal is planning rather than future generation, these observations motivate exploiting future-scene representations to directly shape the policy’s internal representations without a generation objective. To this end, we propose RAF-VLA (Representation Alignment with the Future), a VLA-based autonomous driving framework that incorporates future-scene supervision at the representation level. RAF-VLA employs Future-Aligned Supervised Fine-Tuning, in which the policy learns driving actions while a straightforward regularization aligns its hidden states with future-frame representations. This simple alignment enables RAF-VLA to learn representations useful for planning without the training burden and inference latency of future generation.

We show that RAF-VLA achieves competitive planning performance against state-of-the-art VLA planners on the NAVSIM benchmark, with substantially fewer training samples seen than existing World-Modeling VLAs. Controlled comparisons and qualitative results further support the effectiveness of the proposed approach. Future-Aligned SFT incurs only a 3.8% training overhead, while latency measurements show that RAF-VLA introduces a negligible 1 ms inference overhead. The main contributions of this work are summarized as follows:

- We propose RAF-VLA, a VLA-based autonomous driving framework that leverages future-scene supervision at the representation level, enabling the driving policy to directly learn planning-relevant internal representations.
- RAF-VLA performs Future-Aligned Supervised Fine-Tuning with a straightforward regularization on action learning that aligns the policy’s hidden states with future-frame representations, thereby avoiding the training burden and inference latency of future generation.
- Experiments on the NAVSIM benchmark demonstrate competitive planning performance against state-of-the-art VLA planners, with substantially fewer training samples seen, low training overhead, and negligible inference overhead.

![[RAF-VLA_Figure_2.png|Refer to caption]]

Fig. 2: Mixture-of-Transformers architecture. (a) Joint attention between the two experts. (b) Attention mask across token groups.

## II RELATED WORK

### II-A VLA for Autonomous Driving

Vision-Language-Action (VLA) models fine-tune pretrained VLMs as action policies, leveraging their world knowledge and reasoning capabilities to improve generalization [^3] [^32] [^33] [^34]. This paradigm has also shown promise in end-to-end autonomous driving, mapping visual observations and navigation commands to driving actions to address long-tail and safety-critical scenarios [^2]. Early approaches focused on directly predicting driving actions from observations [^4] [^5] [^6] [^7], while subsequent studies introduced textual reasoning before action prediction to further harness VLMs’ reasoning capabilities, improving planning performance and generalization [^8] [^9] [^10] [^11] [^12] [^13]. Nevertheless, these approaches rely on autoregressive language-token decoding, increasing inference latency. Generating high-quality reasoning traces also incurs additional annotation costs [^35]. Moreover, symbolically compressing visual information into language can obscure spatiotemporal relationships and omit fine-grained visual cues important for planning [^15].

Recent studies have incorporated world modeling into VLA policy learning by predicting future driving scenes alongside actions [^14] [^15] [^16] [^17] [^18] [^19]. These World-Modeling VLAs utilize future scenes that reflect spatiotemporal structure and driving dynamics as dense supervision to shape useful internal representations for planning and have demonstrated strong performance [^17] [^18] [^19]. DriveVLA-W0 [^17] generates future images through autoregressive visual-token prediction or a diffusion-based generation module, while ExploreVLA [^19] jointly generates future RGB and depth images through masked-token prediction. SGDrive [^18] predicts future occupancy, agent states, and driving goals using task-specific prediction heads. However, exploiting future-scene supervision through future generation introduces additional training burden and inference overhead. This entails optimizing additional objectives for future generation, often also requiring dedicated training stages. Inference-time generation adds latency through autoregressive decoding or iterative denoising. Our work instead leverages future-scene supervision directly at the representation level, enabling the driving policy to learn planning-relevant internal representations without future generation.

### II-B Representation Learning in Generative Models

Prior studies have shown that generative objectives can induce useful internal representations for downstream tasks beyond generation [^21] [^22] [^24] [^25] [^26]. More recent studies demonstrate that such representations can be shaped through direct guidance from target representations. REPA [^28] [^29] aligns internal diffusion features with representations from pretrained visual encoders to improve training efficiency, while iREPA [^30] highlights the importance of spatial structure in target representations. ARRA [^31] extends this guidance to autoregressive models. Whereas these methods pair representation guidance with generative objectives, our work uses representation guidance alone to shape planning-relevant representations for driving policies. Specifically, we align the VLA’s hidden representations with future-frame representations alongside action prediction. This leverages future-scene supervision without future generation, thereby avoiding its associated training burden and inference overhead.

## III METHOD

RAF-VLA leverages future-scene supervision at the representation level to directly guide the shaping of planning-relevant representations within the driving policy. In this section, we first formulate the driving task in Sec. III-A and describe the overall architecture in Sec. III-B. We then introduce Future-Aligned Supervised Fine-Tuning (SFT) in Sec. III-C, which augments action learning with a straightforward regularization that aligns the policy’s hidden states with future-frame representations. Finally, we describe GRPO-based reinforcement fine-tuning (RFT) in Sec. III-D to further optimize the policy using planning rewards.

### III-A Problem Formulation

#### III-A1 Input

At time step $t$, the driving policy $\pi_{\theta}$ takes a front-view image sequence, the current navigation instruction, and the driving status as input. Let $\mathbf{I}_{t}=\{I_{t-T},\ldots,I_{t}\}$ denote the front-view image sequence including the current frame $I_{t}$, where $T$ is the observation-history length. The current navigation instruction is denoted by $\ell_{t}$. The driving status $\mathbf{s}_{t}$ consists of recent ego poses and a one-hot encoded current navigation command. We denote the driving input at time step $t$ by $q_{t}=(\mathbf{I}_{t},\ell_{t},\mathbf{s}_{t})$.

#### III-A2 Output

Given the input $q_{t}$, the driving policy $\pi_{\theta}$ predicts an ego trajectory consisting of future waypoints:

$$
\hat{\boldsymbol{\tau}}_{t}=\pi_{\theta}(q_{t})=\left\{\hat{\boldsymbol{p}}_{t+h}\right\}_{h=1}^{H}.
$$

Here, $\hat{\boldsymbol{\tau}}_{t}$ denotes the predicted ego trajectory, $H$ is the number of future steps in the planning horizon, and $h\in\{1,\ldots,H\}$ is the planning-step index. Each predicted ego pose is defined as $\hat{\boldsymbol{p}}_{t+h}=(\hat{x}_{t+h},\hat{y}_{t+h},\hat{\theta}_{t+h})$, where $\hat{x}_{t+h}$ and $\hat{y}_{t+h}$ denote the predicted ego position at the $h$ -th future step, and $\hat{\theta}_{t+h}$ denotes the corresponding heading.

### III-B Overall Architecture

As shown in Fig. (a), RAF-VLA adopts a Mixture-of-Transformers (MoT) architecture comprising a Vision-Language Expert and a lightweight Action Expert [^33] [^17]. The two experts interact through layer-wise joint attention, enabling the Action Expert to leverage the Vision-Language Expert’s multimodal context for planning.

#### III-B1 Vision-Language Expert

The Vision-Language Expert is built on a pretrained VLM to provide multimodal driving context. Its inputs, the front-view image sequence $\mathbf{I}_{t}$ and navigation instruction $\ell_{t}$, are encoded by the VLM’s vision encoder and tokenizer, respectively. We append learnable world queries after the image and language tokens as representation slots for future alignment (Sec. III-C). The resulting sequence is processed to form contextualized multimodal representations.

#### III-B2 Action Expert

The Action Expert uses a similar transformer block structure to the Vision-Language Expert, with a smaller hidden dimension. Its inputs comprise a set of learnable action queries and a status token obtained by projecting the driving status $\mathbf{s}_{t}$ through a multilayer perceptron (MLP). The expert leverages the Vision-Language Expert’s multimodal representations through joint attention to form action representations for trajectory prediction.

#### III-B3 Joint Attention

As illustrated in Fig. 2(a), the two experts compute their own query, key, and value matrices at each transformer layer $l$. These matrices share a common attention embedding dimension and are concatenated along the token sequence dimension to form the inputs to joint attention:

$$
\mathbf{Q}^{l}=[\mathbf{Q}_{\mathrm{VL}}^{l};\mathbf{Q}_{\mathrm{AE}}^{l}],\,\mathbf{K}^{l}=[\mathbf{K}_{\mathrm{VL}}^{l};\mathbf{K}_{\mathrm{AE}}^{l}],\,\mathbf{V}^{l}=[\mathbf{V}_{\mathrm{VL}}^{l};\mathbf{V}_{\mathrm{AE}}^{l}].
$$

Here, the subscripts $\mathrm{VL}$ and $\mathrm{AE}$ denote the Vision-Language Expert and Action Expert, respectively, and the semicolon denotes concatenation. As shown in Fig. 2(b), the Action Expert applies full bidirectional attention among its status token and action queries. Attention from the Action Expert to the Vision-Language Expert is permitted, while attention in the opposite direction is masked. The resulting attention output is then split and passed back to the corresponding expert. Following prior work [^17] [^36], we use a query-based action head with multiple trajectory anchors to decode the final action representations into an ego trajectory. The corresponding trajectory prediction loss is denoted by $\mathcal{L}_{\mathrm{act}}$ [^17].

### III-C Future-Aligned Supervised Fine-Tuning (SFT)

Future-Aligned SFT augments action learning with a straightforward regularization that aligns the policy’s hidden states with future-frame representations. This regularization directly guides the policy to learn planning-relevant representations without a separate generation objective.

#### III-C1 World Query

World queries serve as representation slots for future-frame supervision. Let $\mathcal{H}_{\mathrm{F}}\subseteq\{1,\ldots,H\}$ denote the future planning steps selected for alignment. We assign $K$ learnable world queries to each $h\in\mathcal{H}_{\mathrm{F}}$. We denote their concatenated hidden states after processing by the Vision-Language Expert as $\mathbf{Z}_{t,h}\in\mathbb{R}^{Kd}$, where each hidden state is $d$ -dimensional.

As shown in Fig. 2(b), world queries can attend to the image and language tokens. We apply blockwise causal attention to the world queries, with full bidirectional attention among the $K$ queries assigned to each horizon, while masking attention to later-horizon queries.

#### III-C2 Representation Alignment with the Future

As illustrated in Fig. (a), a frozen pretrained world encoder $f$ extracts target representations from future frames $I_{t+h}$. These representations are used solely as alignment targets during training; neither the future frames nor their extracted representations are provided as inputs to the policy. A trainable MLP projector $g_{\phi}$ maps the world-query hidden states $\mathbf{Z}_{t,h}$ to the target feature space. We align the projected hidden states with the corresponding future-frame representations by minimizing their mean squared error:

$$
\mathcal{L}_{\mathrm{align}}=\frac{1}{|\mathcal{H}_{\mathrm{F}}|}\sum_{h\in\mathcal{H}_{\mathrm{F}}}\left\|f(I_{t+h})-g_{\phi}(\mathbf{Z}_{t,h})\right\|_{2}^{2}.
$$

#### III-C3 Training Objective

We augment the trajectory prediction objective with the alignment regularization:

$$
\mathcal{L}_{\mathrm{SFT}}=\mathcal{L}_{\mathrm{act}}+\lambda_{\mathrm{align}}\mathcal{L}_{\mathrm{align}},
$$

where $\lambda_{\mathrm{align}}$ controls the regularization weight. The world encoder $f$ remains frozen during SFT.

The alignment regularization serves as training-time guidance to shape planning-relevant representations within the policy. The projector’s outputs are used solely to compute the alignment loss. After Future-Aligned SFT, the world encoder and alignment projector are removed, while the learned world queries remain within the Vision-Language Expert. The Action Expert attends directly to their contextualized hidden states through joint attention for trajectory prediction.

### III-D Reinforcement Fine-Tuning (RFT)

After Future-Aligned SFT, we perform reinforcement fine-tuning (RFT) using Group Relative Policy Optimization (GRPO) [^37] (Fig. (c)), following the standard training strategy of recent VLA-based driving methods. GRPO improves the policy by comparing multiple candidate trajectories under the same driving context. In this stage, future-frame supervision is no longer used, and the policy is optimized with planning rewards.

Given the driving input $q_{t}$, the old policy $\pi_{\theta_{\mathrm{old}}}$ samples a group of $G$ candidate outputs $\mathcal{O}=\{o_{1},\ldots,o_{G}\}$, where $o_{i}\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid q_{t})$ denotes a sampled ego trajectory defined as in Eq. (1). The current policy $\pi_{\theta}$ is then optimized using the normalized group-relative advantage $A_{i}$ by maximizing the following objective:

$$
\mathcal{J}(\theta)=\mathbb{E}_{q_{t},\mathcal{O}}\left[\frac{1}{G}\sum_{i=1}^{G}\left(\mathcal{J}_{i}-\beta D_{\mathrm{KL}}(\pi_{\theta}\,\|\,\pi_{\mathrm{ref}})\right)\right],
$$
 
$$
\mathcal{J}_{i}=\min\left(c_{i}A_{i},\,\operatorname{clip}(c_{i},1-\epsilon,1+\epsilon)A_{i}\right),
$$
 
$$
A_{i}=\frac{r_{i}-\operatorname{mean}(\{r_{j}\}_{j=1}^{G})}{\operatorname{std}(\{r_{j}\}_{j=1}^{G})}.
$$

Here, $\mathcal{J}_{i}$ denotes the clipped surrogate objective, and $c_{i}=\frac{\pi_{\theta}(o_{i}\mid q_{t})}{\pi_{\theta_{\mathrm{old}}}(o_{i}\mid q_{t})}$ is the probability ratio between the current and old policies. The group-relative advantage $A_{i}$ is computed by normalizing the planning reward $r_{i}$ using the mean and standard deviation of rewards within the sampled group. The hyperparameter $\epsilon$ controls clipping, while $\beta$ weights the KL penalty between the current policy and the SFT reference policy $\pi_{\mathrm{ref}}$.

The planning reward $r_{i}$ is computed using the NAVSIM Predictive Driver Model Score (PDMS) [^38], which evaluates trajectory quality across multiple driving criteria, including collision avoidance, drivable-area compliance, progress, time-to-collision, and comfort.

## IV EXPERIMENTS

TABLE I: Comparison on NAVSIM v1 with closed-loop metrics. Abbreviations: Sensors: 1x Cam: front-view camera only;  
Nx Cam: N surround-view cameras; L: LiDAR. <sup>†</sup>: best-of- $N$ ($N=6$) strategy following [^10] [^17] [^19].

<table><tbody><tr><td>Method</td><td>Venue</td><td>Sensors</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>C.<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>UniAD <sup><a href="#fn:39">39</a></sup></td><td>CVPR’23</td><td>6x Cam</td><td>97.8</td><td>91.9</td><td>78.8</td><td>92.9</td><td>100.0</td><td>83.4</td></tr><tr><td>TransFuser <sup><a href="#fn:40">40</a></sup></td><td>TPAMI’23</td><td>3x Cam + L</td><td>97.7</td><td>92.8</td><td>79.2</td><td>92.8</td><td>100.0</td><td>84.0</td></tr><tr><td>LAW <sup><a href="#fn:41">41</a></sup></td><td>ICLR’25</td><td>1x Cam</td><td>96.4</td><td>95.4</td><td>81.7</td><td>88.7</td><td>99.9</td><td>84.6</td></tr><tr><td>Epona <sup><a href="#fn:42">42</a></sup></td><td>ICCV’25</td><td>1x Cam</td><td>97.9</td><td>95.1</td><td>80.4</td><td>93.8</td><td>99.9</td><td>86.2</td></tr><tr><td>Hydra-MDP <sup><a href="#fn:36">36</a></sup></td><td>arXiv’24</td><td>3x Cam + L</td><td>98.3</td><td>96.0</td><td>78.7</td><td>94.6</td><td>100.0</td><td>86.5</td></tr><tr><td>DiffusionDrive <sup><a href="#fn:43">43</a></sup></td><td>CVPR’25</td><td>3x Cam + L</td><td>98.2</td><td>96.2</td><td>82.2</td><td>94.7</td><td>100.0</td><td>88.1</td></tr><tr><td>WoTE <sup><a href="#fn:44">44</a></sup></td><td>ICCV’25</td><td>3x Cam + L</td><td>98.5</td><td>96.8</td><td>81.9</td><td>94.9</td><td>99.9</td><td>88.3</td></tr><tr><td>DriveLaW <sup><a href="#fn:45">45</a></sup></td><td>CVPR’26</td><td>1x Cam</td><td>99.0</td><td>97.1</td><td>81.3</td><td>96.7</td><td>100.0</td><td>89.1</td></tr><tr><td colspan="9"><em>Textual-Reasoning VLA</em></td></tr><tr><td>AutoVLA <sup><a href="#fn:10">10</a></sup></td><td>NeurIPS’25</td><td>3x Cam</td><td>98.4</td><td>95.6</td><td>81.9</td><td>98.0</td><td>99.9</td><td>89.1</td></tr><tr><td>ReCogDrive <sup><a href="#fn:11">11</a></sup></td><td>ICLR’26</td><td>3x Cam</td><td>98.2</td><td>97.8</td><td>83.5</td><td>95.2</td><td>99.8</td><td>89.6</td></tr><tr><td>AdaThinkDrive <sup><a href="#fn:9">9</a></sup></td><td>ICRA’26</td><td>1x Cam</td><td>98.4</td><td>97.8</td><td>84.4</td><td>95.2</td><td>100.0</td><td>90.3</td></tr><tr><td>AutoVLA <sup>†</sup> <sup><a href="#fn:10">10</a></sup></td><td>NeurIPS’25</td><td>3x Cam</td><td>99.1</td><td>97.1</td><td>87.6</td><td>97.1</td><td>100.0</td><td>92.1</td></tr><tr><td>AdaThinkDrive <sup>†</sup> <sup><a href="#fn:9">9</a></sup></td><td>ICRA’26</td><td>1x Cam</td><td>99.1</td><td>98.8</td><td>87.9</td><td>97.2</td><td>100.0</td><td>93.0</td></tr><tr><td colspan="9"><em>World-Modeling VLA</em></td></tr><tr><td>DrivingGPT <sup><a href="#fn:14">14</a></sup></td><td>ICCV’25</td><td>1x Cam</td><td>98.9</td><td>90.7</td><td>79.7</td><td>94.9</td><td>95.6</td><td>82.4</td></tr><tr><td>FSDrive <sup><a href="#fn:15">15</a></sup></td><td>NeurIPS’25</td><td>3x Cam</td><td>98.2</td><td>93.8</td><td>80.1</td><td>93.3</td><td>99.9</td><td>85.1</td></tr><tr><td>PWM <sup><a href="#fn:16">16</a></sup></td><td>NeurIPS’25</td><td>1x Cam</td><td>98.6</td><td>95.9</td><td>81.8</td><td>95.4</td><td>100.0</td><td>88.1</td></tr><tr><td>DriveVLA-W0 <sup><a href="#fn:17">17</a></sup></td><td>ICLR’26</td><td>1x Cam</td><td>98.7</td><td>99.1</td><td>83.3</td><td>95.3</td><td>99.3</td><td>90.2</td></tr><tr><td>ExploreVLA <sup><a href="#fn:19">19</a></sup></td><td>ECCV’26</td><td>1x Cam</td><td>98.8</td><td>98.4</td><td>83.5</td><td>96.5</td><td>99.9</td><td>90.4</td></tr><tr><td>SGDrive <sup><a href="#fn:18">18</a></sup></td><td>CVPR’26</td><td>1x Cam</td><td>98.6</td><td>97.8</td><td>85.8</td><td>96.2</td><td>100.0</td><td>91.1</td></tr><tr><td>DriveVLA-W0 <sup>†</sup> <sup><a href="#fn:17">17</a></sup></td><td>ICLR’26</td><td>1x Cam</td><td>99.3</td><td>97.4</td><td>88.3</td><td>97.0</td><td>99.9</td><td>93.0</td></tr><tr><td>ExploreVLA <sup>†</sup> <sup><a href="#fn:19">19</a></sup></td><td>ECCV’26</td><td>1x Cam</td><td>99.4</td><td>98.9</td><td>88.3</td><td>98.3</td><td>99.7</td><td>93.7</td></tr><tr><td>RAF-VLA (Ours)</td><td>–</td><td>1x Cam</td><td>98.0</td><td>98.5</td><td>85.2</td><td>93.9</td><td>99.7</td><td>90.0</td></tr><tr><td>RAF-VLA <sup>†</sup> (Ours)</td><td>–</td><td>1x Cam</td><td>99.0</td><td>99.6</td><td>89.8</td><td>97.3</td><td>100.0</td><td>94.2</td></tr></tbody></table>

TABLE II: NAVSIM v2 results with extended closed-loop metrics using Future-Aligned SFT.

| Method | NC $\uparrow$ | DAC $\uparrow$ | DDC $\uparrow$ | TLC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | LK $\uparrow$ | HC $\uparrow$ | EC $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TransFuser [^40] | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| VADv2 [^46] | 97.3 | 91.7 | 98.2 | 99.9 | 77.6 | 92.7 | 66.0 | 100.0 | 97.4 | 76.6 |
| Hydra-MDP [^36] | 97.5 | 96.3 | 98.3 | 99.9 | 80.1 | 93.0 | 65.5 | 100.0 | 97.4 | 79.8 |
| Hydra-MDP++ [^47] | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| ReCogDrive [^11] | 98.3 | 95.2 | 99.5 | 99.8 | 87.1 | 97.5 | 96.6 | 98.3 | 86.5 | 83.6 |
| DiffusionDrive [^43] | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| DriveVLA-W0 [^17] | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 |
| SGDrive [^18] | 98.6 | 94.3 | 99.5 | 99.9 | 86.0 | 97.9 | 96.1 | 98.3 | 85.9 | 86.2 |
| RAF-VLA (Ours) | 97.5 | 98.2 | 99.4 | 99.8 | 86.9 | 97.0 | 92.7 | 98.0 | 65.7 | 86.2 |

Fig. 3: Qualitative comparison of trajectory predictions on the NAVSIM navtest split. From left to right: front-camera input, vanilla SFT without alignment, Future-Aligned SFT, and Future-Aligned SFT followed by GRPO-based RFT (black: GT, red/orange/green: prediction from each model).

TABLE III: Ablation of future-representation alignment under SFT and RFT. Bold values indicate the best result for each metric.

ID Alignment SFT RFT NC $\uparrow$ DAC $\uparrow$ EP $\uparrow$ TTC $\uparrow$ C.$\uparrow$ PDMS $\uparrow$ 1 $\checkmark$ 97.86 98.58 81.01 93.03 99.52 87.92 2 $\checkmark$ $\checkmark$ 97.42 98.39 84.31 92.38 99.51 88.85 3 $\checkmark$ $\checkmark$ 98.04 98.71 82.99 93.60 99.79 89.13 4 $\checkmark$ $\checkmark$ $\checkmark$ 97.98 98.46 85.21 93.86 99.73 90.03

TABLE IV: Inference latency with and without alignment. Inference time denotes the average time required to predict 4-second trajectories on the NAVSIM v1 navtest split.

| Alignment | Latency (s) $\downarrow$ | $\Delta$ |
| --- | --- | --- |
| $\times$ | 0.152 | – |
| $\checkmark$ | 0.153 | +0.001 |

### IV-A Dataset and Metrics

#### IV-A1 Dataset

We conduct extensive experiments on the NAVSIM v1 and NAVSIM v2 benchmarks [^38] [^48] to evaluate the performance of end-to-end autonomous driving models. NAVSIM is built on OpenScene [^49], which provides real-world sensor observations and annotated driving scenes. The benchmark filters out near-trivial driving cases to emphasize challenging scenarios with a higher proportion of non-trivial maneuvers. We train RAF-VLA on the *navtrain* split and evaluate on the *navtest* split.

#### IV-A2 Metrics

For NAVSIM v1, we evaluate planning performance using the Predictive Driver Model Score (PDMS):

$$
\mathrm{PDMS}=\mathrm{NC}\times\mathrm{DAC}\times\frac{5\mathrm{EP}+5\mathrm{TTC}+2\mathrm{C}}{12},
$$

where No At-Fault Collision (NC), Drivable Area Compliance (DAC), Ego Progress (EP), Time-to-Collision (TTC), and Comfort (C) constitute the evaluation metrics.

For NAVSIM v2, we use the Extended Predictive Driver Model Score (EPDMS):

$$
\displaystyle\mathrm{EPDMS}
$$
 
$$
\displaystyle=\mathrm{NC}\times\mathrm{DAC}\times\mathrm{DDC}\times\mathrm{TLC}
$$
 
$$
\displaystyle\times\frac{5\mathrm{EP}+5\mathrm{TTC}+2\mathrm{LK}+2\mathrm{HC}+2\mathrm{EC}}{16},
$$

which additionally includes Driving Direction Compliance (DDC), Traffic Light Compliance (TLC), Lane Keeping (LK), History Comfort (HC), and Extended Comfort (EC). Higher values indicate better performance for all metrics.

### IV-B Implementation Details

We use Qwen2.5-VL-7B-Instruct as the Vision-Language Expert [^50], a lightweight transformer with an embedding dimension of 1024 as the Action Expert, and a pretrained encoder from the Cosmos world model [^51] to extract target representations from future frames. The alignment projector is a two-layer MLP with a GELU activation [^52]. For Future-Aligned SFT, we train for 4 epochs with AdamW, a global batch size of 144, and a learning rate of $1\times 10^{-5}$ with cosine decay, while freezing the VLM vision encoder and world encoder. Unless otherwise specified, alignment uses the 1 s and 4 s horizons with $K=4$ world queries per horizon and $\lambda_{\mathrm{align}}=1$. For RFT, we perform GRPO for 1.5 epochs with a global batch size of 288, a learning rate of $2\times 10^{-6}$, group size $G=8$, and KL coefficient $\beta=0.04$; the world encoder and alignment projector are removed, while the VLM vision encoder, world queries, and action head remain frozen. Both stages are trained on 4 $\times$ H200 GPUs, taking approximately 11 hours for SFT and 7 hours for RFT.

### IV-C Main Results

#### IV-C1 Performance Comparison

Table I compares RAF-VLA with state-of-the-art (SOTA) methods on NAVSIM v1. Using only a single front-view camera, RAF-VLA achieves a PDMS of 90.0, demonstrating competitive planning performance against SOTA VLA-based planners. As shown in Fig. (b), this performance is achieved with substantially fewer training samples seen than existing World-Modeling VLAs. The total training samples seen for each method are estimated from its reported configurations, official implementations, and official dataset splits. With the best-of- $N$ strategy, the PDMS further increases to 94.2, achieving the highest overall performance, together with the best DAC and EP scores of 99.6 and 89.8, respectively. We further evaluate the Future-Aligned SFT model on NAVSIM v2 using the extended metric. As shown in Table II, it achieves an EPDMS of 86.2, matching the best reported SFT result.

#### IV-C2 Effectiveness of Future-Aligned SFT

Table III evaluates the effect of the regularization for representation alignment in Future-Aligned SFT, both before and after reinforcement fine-tuning. For the vanilla SFT baseline, we remove the entire future-alignment mechanism, including the associated world queries. Future-Aligned SFT improves the PDMS from 87.92 to 89.13 (+1.21) over SFT without alignment. This improvement incurs only a 3.8% increase in training time, from 10.56 h (42.2 H200 GPU-hours) for vanilla SFT to 10.96 h (43.8 H200 GPU-hours) for Future-Aligned SFT, including online extraction of future representations. Importantly, this gain is retained after reinforcement fine-tuning. The model initialized from Future-Aligned SFT achieves a PDMS of 90.03, compared with 88.85 for the model initialized from SFT without alignment (+1.18). These results show that the benefit introduced during Future-Aligned SFT persists through subsequent RFT.

Fig. 3 provides a qualitative comparison of trajectory predictions from SFT without alignment, Future-Aligned SFT, and Future-Aligned SFT followed by RFT. SFT without alignment produces trajectories that poorly follow the lane consistent with the navigation command and deviate from the ground-truth trajectory. They also fail to make sufficient progress toward the intended direction. Future-Aligned SFT corrects these failures, yielding trajectories that better follow the intended maneuver, while subsequent RFT further refines the predicted paths. The quantitative and qualitative results consistently support the effectiveness of Future-Aligned SFT.

#### IV-C3 Inference Overhead

As shown in Table IV, Future-Aligned SFT adds only 0.001 s (1 ms) of latency for predicting a 4-second trajectory, resulting in negligible inference overhead. Inference latency is measured on a single NVIDIA H200 GPU with a batch size of 1. This is because Future-Aligned SFT directly shapes policy representations through alignment during training, leaving only a few learned world queries to be processed during the prefill stage at inference. No additional scene prediction or generation is required.

TABLE V: Ablation of future-alignment horizons on NAVSIM v1.

| ID | 1 s | 2 s | 3 s | 4 s | NC $\uparrow$ | DAC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | C.$\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 |  |  |  |  | 97.86 | 98.58 | 81.01 | 93.03 | 99.52 | 87.92 |
| 2 | $\checkmark$ |  |  |  | 98.40 | 98.50 | 82.33 | 93.91 | 99.81 | 88.93 |
| 3 |  |  |  | $\checkmark$ | 98.30 | 98.67 | 82.39 | 93.78 | 99.76 | 88.97 |
| 4 | $\checkmark$ | $\checkmark$ |  |  | 97.80 | 98.48 | 82.32 | 93.47 | 99.53 | 88.60 |
| 5 | $\checkmark$ |  |  | $\checkmark$ | 98.04 | 98.71 | 82.99 | 93.60 | 99.79 | 89.13 |
| 6 | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | 97.88 | 98.53 | 82.52 | 93.13 | 99.79 | 88.62 |

TABLE VI: Ablation of the query count $K$ on NAVSIM v1.

| ID | $K$ | NC $\uparrow$ | DAC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | C.$\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 98.14 | 98.36 | 82.94 | 93.21 | 99.78 | 88.76 |
| 2 | 4 | 98.04 | 98.71 | 82.99 | 93.60 | 99.79 | 89.13 |
| 3 | 8 | 97.94 | 98.63 | 82.30 | 93.25 | 99.76 | 88.64 |
| 4 | 16 | 97.97 | 98.63 | 82.49 | 93.03 | 99.79 | 88.65 |

### IV-D Ablation Studies

Unless otherwise specified, all ablation studies follow the implementation details in Sec. IV-B and are conducted under the SFT setting.

#### IV-D1 Effect of Future Alignment Horizons

Table VI compares different future horizons used for representation alignment. Using either the 1 s or 4 s future representation improves the planning performance over the vanilla SFT baseline, with the 4 s horizon corresponding to the end of the planning horizon. Among the evaluated configurations, jointly using the 1 s and 4 s horizons achieves the highest PDMS of 89.13, while adding intermediate horizons does not provide further improvement. We therefore use the 1 s and 4 s horizons as the default configuration.

#### IV-D2 Effect of the Number of World Queries

Table VI ablates the number of world queries assigned to each future horizon. Among the evaluated settings, $K=4$ achieves the highest PDMS of 89.13, while increasing the number of world queries beyond four does not provide further improvement. We therefore use four world queries per horizon as the default configuration.

## V CONCLUSIONS

In this work, we introduced RAF-VLA, a VLA-based autonomous driving framework that leverages future-scene supervision at the representation level without future generation. RAF-VLA performs Future-Aligned Supervised Fine-Tuning, which aligns the policy’s hidden states with future-frame representations, thereby providing direct guidance for shaping planning-relevant internal representations during action learning. Experiments on NAVSIM v1 and v2 demonstrate competitive planning performance with substantially fewer training samples seen, low training overhead, and negligible inference overhead. Controlled comparisons and qualitative results provide further evidence for the effectiveness of our proposed approach. Future work will explore alternative target encoders and projection designs. We also plan to extend future representation alignment to multi-view inputs and richer scene modalities.

[^1]: L. Chen, P. Wu, K. Chitta, B. Jaeger, A. Geiger, and H. Li (2024) End-to-end autonomous driving: challenges and frontiers. IEEE Transactions on Pattern Analysis and Machine Intelligence 46 (12), pp. 10164–10183. Cited by: §I.

[^2]: S. Jiang, Z. Huang, K. Qian, Z. Luo, T. Zhu, Y. Zhong, Y. Tang, M. Kong, Y. Wang, S. Jiao, et al. (2025) A survey on vision-language-action models for autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4583–4595. Cited by: §I, §II-A.

[^3]: A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, et al. (2023) Rt-2: vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818. Cited by: §I, §II-A.

[^4]: H. Shao, Y. Hu, L. Wang, G. Song, S. L. Waslander, Y. Liu, and H. Li (2024) Lmdrive: closed-loop end-to-end driving with large language models. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15120–15130. Cited by: §I, §II-A.

[^5]: H. Fu, D. Zhang, Z. Zhao, J. Cui, D. Liang, C. Zhang, D. Zhang, H. Xie, B. Wang, and X. Bai (2025) Orion: a holistic end-to-end autonomous driving framework by vision-language instructed action generation. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 24823–24834. Cited by: §I, §II-A.

[^6]: K. Renz, L. Chen, E. Arani, and O. Sinavski (2025) Simlingo: vision-only closed-loop autonomous driving with language-action alignment. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11993–12003. Cited by: §I, §II-A.

[^7]: X. Zhou, X. Han, F. Yang, Y. Ma, V. Tresp, and A. Knoll (2026) Opendrivevla: towards end-to-end autonomous driving with large vision language action model. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, pp. 13782–13790. Cited by: §I, §II-A.

[^8]: J. Hwang, R. Xu, H. Lin, W. Hung, J. Ji, K. Choi, D. Huang, T. He, P. Covington, B. Sapp, et al. (2024) Emma: end-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2410.23262. Cited by: §I, §II-A.

[^9]: Y. Luo, F. Li, S. Xu, Z. Lai, L. Yang, Q. Chen, Z. Luo, Z. Xie, S. Jiang, J. Liu, et al. (2025) Adathinkdrive: adaptive thinking via reinforcement learning for autonomous driving. arXiv preprint arXiv:2509.13769. Cited by: §I, §II-A, TABLE I, TABLE I.

[^10]: Z. Zhou, T. Cai, S. Zhao, Y. Zhang, Z. Huang, B. Zhou, and J. Ma (2026) Autovla: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. Advances in Neural Information Processing Systems 38, pp. 27920–27956. Cited by: §I, §II-A, TABLE I, TABLE I, TABLE I.

[^11]: K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, K. Ma, et al. (2026) Recogdrive: a reinforced cognitive framework for end-to-end autonomous driving. In International Conference on Learning Representations, Vol. 2026, pp. 157518–157556. Cited by: §I, §II-A, TABLE I, TABLE II.

[^12]: Z. Yuan, C. Qian, J. Tang, R. Chen, Z. Song, L. Sun, X. Chu, Y. Cai, D. Zhang, and S. Li (2026) Autodrive-r ${}^{2}$: incentivizing reasoning and self-reflection capacity for vla model in autonomous driving. In International Conference on Learning Representations, Vol. 2026, pp. 97934–97955. Cited by: §I, §II-A.

[^13]: M. Zawalski, W. Chen, K. Pertsch, O. Mees, C. Finn, and S. Levine (2024) Robotic control via embodied chain-of-thought reasoning. arXiv preprint arXiv:2407.08693. Cited by: §I, §II-A.

[^14]: Y. Chen, Y. Wang, and Z. Zhang (2025) Drivinggpt: unifying driving world modeling and planning with multi-modal autoregressive transformers. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 26890–26900. Cited by: §I, §II-A, TABLE I.

[^15]: S. Zeng, X. Chang, M. Xie, X. Liu, Y. Bai, Z. Pan, M. Xu, and X. Wei (2026) Futuresightdrive: thinking visually with spatio-temporal cot for autonomous driving. Advances in Neural Information Processing Systems 38, pp. 67299–67318. Cited by: §I, §I, §II-A, §II-A, TABLE I.

[^16]: Z. Zhao, T. Fu, Y. Wang, L. Wang, and H. Lu (2026) From forecasting to planning: policy world model for collaborative state-action prediction. Advances in Neural Information Processing Systems 38, pp. 134585–134611. Cited by: §I, §I, §II-A, TABLE I.

[^17]: Y. Li, S. Shang, W. Liu, B. Zhan, H. Wang, Y. Wang, Y. Chen, X. Wang, Y. An, C. Tang, et al. (2026) Drivevla-w0: world models amplify data scaling law in autonomous driving. In International Conference on Learning Representations, Vol. 2026, pp. 7890–7911. Cited by: §I, §I, §II-A, §III-B3, §III-B, TABLE I, TABLE I, TABLE I, TABLE II.

[^18]: J. Li, J. Wu, D. Hu, X. Huang, B. Sun, Z. Hao, X. Lang, X. Zhu, and L. Zhang (2026) Sgdrive: scene-to-goal hierarchical world cognition for autonomous driving. arXiv preprint arXiv:2601.05640. Cited by: §I, §I, §II-A, TABLE I, TABLE II.

[^19]: Z. Sheng, X. Ye, J. Luo, S. Chen, and L. Ren (2026) Explorevla: dense world modeling and exploration for end-to-end autonomous driving. arXiv preprint arXiv:2604.02714. Cited by: §I, §I, §II-A, TABLE I, TABLE I, TABLE I.

[^20]: Q. Zhao, Y. Lu, M. J. Kim, Z. Fu, Z. Zhang, Y. Wu, Z. Li, Q. Ma, S. Han, C. Finn, et al. (2025) Cot-vla: visual chain-of-thought reasoning for vision-language-action models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1702–1713. Cited by: §I.

[^21]: A. C. Li, M. Prabhudesai, S. Duggal, E. Brown, and D. Pathak (2023) Your diffusion model is secretly a zero-shot classifier. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 2206–2217. Cited by: §I, §II-B.

[^22]: L. Tang, M. Jia, Q. Wang, C. P. Phoo, and B. Hariharan (2023) Emergent correspondence from image diffusion. Advances in neural information processing systems 36, pp. 1363–1389. Cited by: §I, §II-B.

[^23]: X. Chen, Z. Liu, S. Xie, and K. He (2025) Deconstructing denoising diffusion models for self-supervised learning. In International Conference on Learning Representations, Vol. 2025, pp. 55458–55472. Cited by: §I.

[^24]: H. Jeong, J. Shin, and D. Kum (2026) To view transform or not to view transform: nerf-based pre-training perspective. arXiv preprint arXiv:2603.28090. Cited by: §I, §II-B.

[^25]: M. Chen, A. Radford, R. Child, J. Wu, H. Jun, D. Luan, and I. Sutskever (2020) Generative pretraining from pixels. In International conference on machine learning, pp. 1691–1703. Cited by: §I, §II-B.

[^26]: R. Dong, Y. Peng, Z. Qi, Z. Ge, J. Yang, L. Zhao, J. Sun, H. Zhou, H. Wei, X. Kong, et al. (2024) Dreamllm: synergistic multimodal comprehension and creation. In International Conference on Learning Representations, Vol. 2024, pp. 6666–6702. Cited by: §I, §II-B.

[^27]: Z. Su, H. Wei, K. Cen, Y. Wang, G. Chen, C. Yuan, and X. Chu (2026) Generation enhances understanding in unified multimodal models via multi-representation generation. arXiv preprint arXiv:2601.21406. Cited by: §I.

[^28]: S. Yu, S. Kwak, H. Jang, J. Jeong, J. Huang, J. Shin, and S. Xie (2024) Representation alignment for generation: training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940. Cited by: §I, §II-B.

[^29]: X. Leng, J. Singh, Y. Hou, Z. Xing, S. Xie, and L. Zheng (2025) Repa-e: unlocking vae for end-to-end tuning with latent diffusion transformers. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 18262–18272. Cited by: §I, §II-B.

[^30]: J. Singh, X. Leng, Z. Wu, L. Zheng, R. Zhang, E. Shechtman, and S. Xie (2025) What matters for representation alignment: global information or spatial structure?. In The Fourteenth International Conference on Learning Representations, Cited by: §I, §II-B.

[^31]: X. Xie, J. Liu, Z. Lin, H. Fan, Z. Han, Y. Tang, and L. Qu (2026) Unleashing the potential of large language models for text-to-image generation through autoregressive representation alignment. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, pp. 11105–11113. Cited by: §I, §II-B.

[^32]: M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al. (2024) Openvla: an open-source vision-language-action model. arXiv preprint arXiv:2406.09246. Cited by: §II-A.

[^33]: K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. (2024) $\pi_{0}$: a vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164. Cited by: §II-A, §III-B.

[^34]: J. Bjorck, F. Castañeda, N. Cherniadev, X. Da, R. Ding, L. Fan, Y. Fang, D. Fox, F. Hu, S. Huang, et al. (2025) Gr00t n1: an open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734. Cited by: §II-A.

[^35]: I. Rawal, S. Gupta, Y. Hu, and W. Zhan (2026) Nord: a data-efficient vision-language-action model that drives without reasoning. arXiv preprint arXiv:2602.21172. Cited by: §II-A.

[^36]: Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y. Ji, Z. Li, Z. Zhu, J. Kautz, Z. Wu, et al. (2024) Hydra-mdp: end-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978. Cited by: §III-B3, TABLE I, TABLE II.

[^37]: Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. (2024) Deepseekmath: pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300. Cited by: §III-D.

[^38]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, et al. (2024) Navsim: data-driven non-reactive autonomous vehicle simulation and benchmarking. Advances in Neural Information Processing Systems 37, pp. 28706–28719. Cited by: §III-D, §IV-A1.

[^39]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, et al. (2023) Planning-oriented autonomous driving. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 17853–17862. Cited by: TABLE I.

[^40]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger (2022) Transfuser: imitation with transformer-based sensor fusion for autonomous driving. IEEE transactions on pattern analysis and machine intelligence 45 (11), pp. 12878–12895. Cited by: TABLE I, TABLE II.

[^41]: Y. Li, L. Fan, J. He, Y. Wang, Y. Chen, Z. Zhang, and T. Tan (2025) Enhancing end-to-end autonomous driving with latent world model. In International Conference on Learning Representations, Vol. 2025, pp. 42942–42959. Cited by: TABLE I.

[^42]: K. Zhang, Z. Tang, X. Hu, X. Pan, X. Guo, Y. Liu, J. Huang, L. Yuan, Q. Zhang, X. Long, et al. (2025) Epona: autoregressive diffusion world model for autonomous driving. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 27220–27230. Cited by: TABLE I.

[^43]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al. (2025) Diffusiondrive: truncated diffusion model for end-to-end autonomous driving. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12037–12047. Cited by: TABLE I, TABLE II.

[^44]: Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang (2025) End-to-end driving with online trajectory evaluation via bev world model. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 27137–27146. Cited by: TABLE I.

[^45]: T. Xia, Y. Li, L. Zhou, J. Yao, K. Xiong, H. Sun, B. Wang, K. Ma, G. Chen, H. Ye, et al. (2026) Drivelaw: unifying planning and video generation in a latent driving world. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 39701–39712. Cited by: TABLE I.

[^46]: S. Chen, B. Jiang, H. Gao, B. Liao, Q. Xu, Q. Zhang, C. Huang, W. Liu, and X. Wang (2024) Vadv2: end-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243. Cited by: TABLE II.

[^47]: K. Li, Z. Li, S. Lan, Y. Xie, Z. Zhang, J. Liu, Z. Wu, Z. Yu, and J. M. Alvarez (2025) Hydra-mdp++: advancing end-to-end driving via expert-guided hydra-distillation. arXiv preprint arXiv:2503.12820. Cited by: TABLE II.

[^48]: W. Cao, M. Hallgarten, T. Li, D. Dauner, X. Gu, C. Wang, Y. Miron, M. Aiello, H. Li, I. Gilitschenski, et al. (2025) Pseudo-simulation for autonomous driving. arXiv preprint arXiv:2506.04218. Cited by: §IV-A1.

[^49]: O. Contributors (2023) OpenScene: the largest up-to-date 3d occupancy prediction benchmark in autonomous driving. Note: https://github.com/OpenDriveLab/OpenScene Cited by: §IV-A1.

[^50]: S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, et al. (2025) Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923. Cited by: §IV-B.

[^51]: N. Agarwal, A. Ali, M. Bala, Y. Balaji, E. Barker, T. Cai, P. Chattopadhyay, Y. Chen, Y. Cui, Y. Ding, et al. (2025) Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575. Cited by: §IV-B.

[^52]: D. Hendrycks and K. Gimpel (2016) Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415. Cited by: §IV-B.