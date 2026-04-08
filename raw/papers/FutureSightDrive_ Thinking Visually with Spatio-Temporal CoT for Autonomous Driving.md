---
title: "FutureSightDrive: Thinking Visually with Spatio-Temporal CoT for Autonomous Driving"
source: "https://arxiv.org/html/2505.17685v3"
author:
published:
created: 2026-04-07
description:
tags:
  - "clippings"
---
<sup>†</sup> <sup>†</sup>

Shuang Zeng <sup>1, 2 *</sup>,   Xinyuan Chang <sup>2</sup>,   Mengwei Xie <sup>2</sup>,   Xinran Liu <sup>2</sup>,  
Yifan Bai <sup>1, 3</sup>,   Zheng Pan <sup>2</sup>,   Mu Xu <sup>2</sup>,   Xing Wei <sup>1†</sup>,  Ning Guo <sup>2</sup>  
<sup>1</sup> Xi’an Jiaotong University   <sup>2</sup> Amap, Alibaba Group   <sup>3</sup> DAMO Academy, Alibaba Group  
zengshuang@stu.xjtu.edu.cn, weixing@mail.xjtu.edu.cn,  
{changxinyuan.cxy, xiemengwei.xmw, tom.lxr}@alibaba-inc.com,  
{baiyifan.byf, panzheng.pan, xumu.xm,ning.guo}@alibaba-inc.com

###### Abstract

Vision-Language-Action (VLA) models offer significant potential for end-to-end driving, yet their reasoning is often constrained by textual Chains-of-Thought (CoT). This symbolic compression of visual information creates a modality gap between perception and planning by blurring spatio-temporal relations and discarding fine-grained cues. We introduce FSDrive, a framework that empowers VLAs to "think visually" using a novel visual spatio-temporal CoT. FSDrive first operates as a world model, generating a unified future frame that combines a predicted background with explicit, physically-plausible priors like future lane dividers and 3D object boxes. This imagined scene serves as the visual spatio-temporal CoT, capturing both spatial structure and temporal evolution in a single representation. The same VLA then functions as an inverse-dynamics model to plan trajectories conditioned on current observations and this visual CoT. We enable this with a unified pre-training paradigm that expands the model’s vocabulary with visual tokens and jointly optimizes for semantic understanding (VQA) and future-frame prediction. A progressive curriculum first generates structural priors to enforce physical laws before rendering the full scene. Evaluations on nuScenes and NAVSIM show FSDrive improves trajectory accuracy and reduces collisions, while also achieving competitive FID for video generation with a lightweight autoregressive model and advancing scene understanding on DriveLM. These results confirm that our visual spatio-temporal CoT bridges the perception-planning gap, enabling safer, more anticipatory autonomous driving. Code is available at [https://github.com/MIV-XJTU/FSDrive](https://github.com/MIV-XJTU/FSDrive).

## 1 Introduction

The advent of Multimodal Large Language Models (MLLMs) is reshaping autonomous driving, with Vision-Language-Action (VLA) models emerging as a promising end-to-end paradigm \[hu2025contextalignment, ma2023dolphins, zhang2024chatscene, li2024llada\]. Harnessing the superior capabilities of MLLMs in world knowledge, reasoning, and interpretability, these models directly map visual observations and language instructions to vehicle control commands (e.g., speed and trajectory). This approach not only simplifies the conventional modular architecture, thereby minimizing potential information loss across components, but also enables the system to leverage vast pre-trained knowledge for analyzing complex driving environments and reasoning about safe decisions.

To enhance their reasoning abilities, many such models have incorporated the Chain-of-Thought (CoT) strategy, which encourages step-by-step thinking \[Wei2022ChainOT, qu2025spatialvla, guo2025tom, sarkar2025reasoning\]. However, in existing autonomous driving applications \[Hwang2024EMMAEM, mao2023agentdriver, gao2025openfly\], this often involves generating discrete textual CoTs (e.g., language descriptions of the current scene or bounding box coordinates) as intermediate steps. This process forces a conversion of rich, continuous visual data into abstract, symbolic representations — a form of lossy compression that can obscure critical spatio-temporal relationships, discard fine-grained visual details, and introduce a "modality gap" between perception and planning \[motamed2025physics, song2025hume, wu2025teaching\], as illustrated in Figure 1. For autonomous vehicles requiring deep physical-world interaction, should their thinking process more closely resemble simulation and imagination of world, rather than merely relying on logical deduction of language?

![[compare.png|Refer to caption]]

Figure 1: Comparison of different CoT. Textual CoT expression provides insufficient information. The modalities between the image-text CoT are inconsistent. The proposed spatio-temporal CoT captures the temporal and spatial relationships in the future.

Inspired by the human driver’s cognitive mechanism of directly constructing visual representations of future scenarios in the mind, rather than converting them into language descriptions for reasoning, we propose a more intuitive spatio-temporal CoT method as shown in the bottom part of Figure 1. This method avoids information loss during text abstraction and enables the model to think visually about trajectory planning. First, the VLA serves as a world model to generate unified image frame for predicting future world states: Inspired by visual prompting engineering \[redcircle, yu2025forgetme\] that draws red circles on images to guide model attention and by VLIPP \[Yang2025VLIPPTP\] first predicts future bounding boxes to introduce physical priors when generating future frames, we represent future world spatial relationships through future red lane dividers and 3D detection boxes on the predicted unified frames \[yu2025physics\]. These coarse-grained visual cues direct the model’s attention toward drivable areas and critical objects in future scenes while enforcing physically plausible constraints. Meanwhile, the temporal relationships are represented by the ordinary future frame, where the dynamic evolution of visual content intuitively characterizes temporal progression and the inherent laws of scene development. Subsequently, the spatio-temporal CoT acts as an intermediate reasoning step, enabling the VLA to function as an inverse dynamics model for trajectory planning based on current observations and future predictions. Compared to traditional discrete text CoT, and even image-text CoT methods \[Hwang2024EMMAEM, Zhao2025CoTVLAVC, liu2025qfft\] as shown in the middle of the Figure 1, our method unifies both future scene representations and perception outputs in image format, which more effectively conveys the temporal and spatial relationships. This eliminates semantic gaps caused by cross-modal conversions (e.g., converting visual perceptions into textual descriptions for reasoning), establishing an end-to-end visual reasoning pipeline that enables direct visual causal inference by the model.

To endow VLAs with image generation capabilities, we propose a pre-training paradigm that simultaneously preserves the semantic understanding of existing MLLM and activates their visual generation capacity. Specifically, for the semantic understanding preservation part, we follow previous approaches \[Wang2024OmniDriveAH, Hwang2024EMMAEM, huang2025intelligent\] by incorporating visual question answering (VQA) tasks for current scene comprehension. For the activation of visual generation capabilities, we investigate the shared vocabulary space between image and text, directly unleashing the visual generation potential of existing MLLMs in the field of autonomous driving through minimal data (approximately 0.3% of previous methods \[wu2024janus, wu2024vila-u, huang2024intelligent, liang2025persistent\]) without requiring complex model architecture modifications or redesigns. However, directly generating complete detailed future scenes may fail to adhere to physical laws \[Yang2025VLIPPTP, zhang2025rearank\]. Thus, we propose a progressive, easy-to-hard generation method. We leverage the world knowledge of VLAs to first infer drivable regions and key object positions in future scenarios, generating coarse-grained future perception images (e.g., lane dividers and 3D detection) to constrain physical laws. Subsequently, full future frames are generated under this constraint to supplement fine-grained details, enabling the model to think visually about accurate future prediction.

Extensive experiments on trajectory planning, future frames generation, and scene understanding tasks demonstrate the effectiveness of pre-training paradigm and spatio-temporal CoT in FSDrive. FSDrive achieves road scene comprehension by establishing pixel-level embodied associations with the environment, rather than relying on human-designed abstract linguistic symbols, advancing autonomous driving towards visual reasoning. In summary, our main contributions are as follows:

- We propose a spatio-temporal CoT reasoning method that allows the model to enhance trajectory planning by thinking visually from future temporal and spatial dimensions.
- We propose a unified pre-training paradigm for visual generation and understanding. Meanwhile, we introduce a progressive generation approach that evolves from imposing physical constraints to supplementing details.
- We conduct comprehensive evaluations across trajectory planning, future frames generation, and scene understanding tasks, demonstrating the effectiveness of our FSDrive.

## 2 Related work

### 2.1 Unified multimodal understanding and generation

Recent research efforts \[lwm, wu2024janus, Siamese-Diffusion, wei2025copeft\] have increasingly focused on unifying multimodal understanding and visual generation within a single LLM. On one front, methods like Show-o \[xie2024showo\], and VILA-U \[wu2024vila-u\] employ VQ-VAE \[vqvae\] to transform images into discrete tokens while training LLMs to predict them. However, these methods suffer from insufficient semantic information preservation, often leading to performance degradation in downstream understanding tasks. Alternative methods \[emu, dong2024dreamllm, qian2025diffusion, dai2025caption, yuan2025unimapgen\] utilize ViT \[vit\] -based vision encoders (e.g., CLIP \[clip\]) to encode images into continuous feature maps. Nevertheless, such methods typically depend on external diffusion models for image generation or use different training objectives (i.e. diffusion and autoregression) for the two tasks, further complicates the infrastructure design with overall lower efficiency. Moreover, the aforementioned methods usually require massive billion-scale datasets for extensive training from scratch, which results in prohibitively high computational costs when disseminating explorations in this form. In this work, we demonstrate that the visual generative capabilities of existing MLLMs can be directly activated through minimal training costs (approximately 0.3% of previous methods \[wu2024janus, sun2025gdiffretro, lu2025dammfnd, dai2025securetugo\]) without requiring sophisticated architectural designs.

### 2.2 Vision-language models for autonomous driving

Given the superior capabilities of large language models (LLMs) in world knowledge, reasoning, and interpretability, recent researches \[Chang2024MapDR, yue2025real, liu2024fedbcgd, zeng2025janusvln\] increasingly integrate Vision-Language Models (VLMs)/LLMs with autonomous driving systems to address limitations in end-to-end approaches. DriveGPT4 \[xu2024drivegpt4\] employs LLMs through iterative question-answering interactions to explain vehicle behaviors and predict control signals. DriveVLM \[DriveVLM\] synergizes LLMs with end-to-end architectures, where LLMs predict low-frequency trajectories that are subsequently refined by the end-to-end model for final planning. Doe-1 \[doe\] reformulates autonomous driving as a next-token prediction task using Lumina-mGPT’s \[liu2024lumina-mgpt\] multimodal generation capabilities, executing diverse tasks through multimodal token processing. EMMA \[Hwang2024EMMAEM\] leverages Gemini’s multimodal foundation by encoding all non-sensor inputs (navigation instructions, vehicle status) and outputs (trajectories, 3D positions) as natural language text, fully exploiting pre-trained LLMs’ world knowledge. In this work, we propose a spatio-temporal chain of thought (CoT) reasoning method that unifies the form of images, allowing the model to think visually about trajectory planning.

### 2.3 World models for autonomous driving

World models \[wang2023driving, Min2024DriveWorld4P, zhang2025cchall, zhang2025vitcot\] aim to infer ego status and dynamic environments from past observations to enable accurate future prediction and planning. Current applications of world models in autonomous driving primarily focus on driving scenario generation \[Ni2025MaskGWMAG, Hassan2024GEMAG, li2025semi\], planning \[wang2023driving, liu2025qfft\], and representation learning \[Min2024DriveWorld4P, Yang\_2024\_CVPR, zeng2024driving\]. For driving scenario generation, most prior works are built upon diffusion models, with the exception of GAIA-1 \[Hu2023GAIA1AG\] which incorporates a progressive next-token predictor and an additional diffusion image decoder. Recent DrivingGPT \[Chen2024DrivingGPTUD\] leverages existing vision generation LLM LlamaGen \[sun2024llamagen\] while simultaneously outputting predictions for future states and actions. However, such VQ-VAE based visual tokens lack semantic information, often leading to performance degradation in downstream visual understanding tasks \[xie2024showo, liu2024rag, tan2025teaching\]. In this work, we propose to directly activate the visual generation capabilities of existing multimodal large language models, enabling VLMs to act as world models and predict future frames.

## 3 Proposed method: FSDrive

The proposed FSDrive is illustrated in Figure 2. Section 3.1 describes the preliminaries. Section 3.2 presents a unified visual generation and understanding pre-training paradigm and a progressive generation method. Section 3.3 proposes spatio-temporal chain-of-thought methods. Section 3.4 details the training strategy.

### 3.1 Preliminary

#### End-to-end trajectory planning.

End-to-end autonomous driving directly generates future trajectory from sensor data, convertible to vehicle control actions like acceleration and steering \[Hwang2024EMMAEM\]. given $N$ surround-view images $I_{t}=\{I_{t}^{1},I_{t}^{2},\ldots,I_{t}^{N}\}$ at timestep $t$, model $\mathcal{M}$ outputs a BEV trajectory $W_{t}=\{w_{t}^{1},w_{t}^{2},\ldots,w_{t}^{n}\}$, where each waypoint $w_{t}^{i}=(x_{t}^{i},y_{t}^{i})$. The process is formulated as:

$$
W_{t}=\mathcal{M}(I_{t},opt(T_{com},T_{ego})),
$$

$opt(T_{com},T_{ego})$ denotes optional navigation commands and ego status (e.g., velocity, acceleration).

#### Unified visual generation and understanding.

Recent works \[wu2024janus, huang2024magicfight\] unify multimodal understanding and vision generation in single LLM. While understanding aligns with standard LLMs, generation methods \[lwm, huang2025m4v\] typically use VQ-VAE \[vqvae\] to encode images into discrete tokens. First, the image tokenizer quantizes image pixels $x\in\mathbb{R}^{H\times W\times 3}$ into discrete tokens $q\in\mathcal{Q}^{h\times w}$, where $h=H/p$, $w=W/p$, $p$ is the downsampling factor, and $q(i,j)$ represents the index of the image codebook. These $h\cdot w$ tokens are arranged in raster order to train a Transformer \[Vaswani2017AttentionIA\] -based autoregressive model. During image generation, a general language modeling (LM) objective is adopted to autoregressively predict the next token, maximizing the likelihood of each image token:

$$
\mathcal{L}=-\sum_{i=1}\log P_{\theta}(q_{i}|q_{<i}),
$$

where $q_{i}$ denotes the visual token and $\theta$ represents the LLM parameters. Finally, the VQ-VAE’s detokenizer converts these image tokens back into image pixels.

### 3.2 Unified pre-training paradigm for visual generation and understanding

To enable unified pre-training, MLLMs require visual generation capabilities. As described in Section 3.1, existing methods (e.g. Lumina-mGPT \[liu2024lumina-mgpt\], the visual generation LLM used by Doe-1 \[doe\]) typically employ VQ-VAE to encode images into discrete tokens when extracting visual information. However, these tokens lack semantic information, which hurts downstream understanding performance \[xie2024showo, zhou2025opening\]. Moreover, current methods \[wu2024janus, zhou2025foodsky\] demand expensive training from scratch on massive billion-scale datasets without leveraging existing LLM knowledge.

Our method is directly built upon any existing MLLM that employs ViT-based encoders to convert images into continuous features. We preserve the original MLLM architecture without altering any structural components to maintain compatibility with pretrained weights. The sole modification involves expanding the MLLM’s vocabulary by incorporating image tokens of the VQ-VAE into the text codebook, thereby extending the vocabulary’s scope from language space to a multimodal space encompassing both visual and textual modalities. This enhancement enables the MLLM to predict image tokens, which can then be converted to image pixels through an VQ-VAE’s detokenizer.

#### Pre-training for visual understanding.

To effectively preserve the semantic understanding capabilities of the native MLLM during the pre-training stage, as shown in the left part of Figure 2, we follow previous methods \[Wang2024OmniDriveAH, Hwang2024EMMAEM\] by using a VQA task, which is crucial for autonomous vehicles to analyze complex driving scenarios. Given an image-text question-answer pair $(I,L)$, where $I$ represents the surround-view images of the current scene and $L$ denotes the instructional question, model $\mathcal{M}$ generates a corresponding answer $A$:

$$
A=\mathcal{M}(I,L).
$$

#### Pre-training for visual generation.

Inspired by the world models in autonomous driving \[kim2021drivegan, yang2024genad\] that generate future frames to learn physical laws, after activating the visual generation capability, we also enable the VLA to predict future frames, thereby capturing the dynamic evolution of the world. Specifically, given an image-instruction pair $(I,L)$, the model predicts the next visual token of the future front-view frame through autoregressive generation:

$$
P(q_{1},q_{2},\dots,q_{h\cdot w})=\Pi_{t=1}^{h\cdot w}P_{\theta}(q_{i}\mid q_{<i}).
$$

The predicted visual tokens are then converted back into image pixels by VQ-VAE’s detokenizer. Since future frames naturally exist in video datasets without requiring any labeled data, this approach unlocks the potential to harness abundant video data for improving generation quality.

![[method_final.png|Refer to caption]]

Figure 2: Overview of FSDrive. Taking the currently surround images and task instructions as input, MLLM is trained in the form of next token prediction. MLLM predicts the future spatio-temporal CoT, and then generates trajectory based on the current observation and predicted future.

#### Progressive image generation.

However, directly generating complete detailed future scenes may fail to adhere to physical laws \[Yang2025VLIPPTP\]. Therefore, during pre-training stage, we propose a progressive, easy-to-hard generation method, incorporating annotated data containing lane divider and 3D detection. Before generating visual tokens of future frames $Q_{f}$, we leverage the world knowledge of VLA to first reason about visual tokens of lane dividers $Q_{l}$, which serve as the skeleton of the road scene and define drivable areas to enforce static physical constraints. Subsequently, we reason about visual tokens of 3D bounding boxes $Q_{d}$, representing motion patterns of key objects to impose dynamic physical constraints. This progressive method sequence explicitly guides the model to infer structural layouts and geometric details of future scenes while enforcing physical laws. By leveraging these intermediate visual reasoning steps as context, the model learns to think visually about the dynamic evolution of scenes, ultimately enabling accurate future prediction:

$$
P(Q_{f}\mid Q_{l},Q_{d})=\Pi_{t=1}^{h\cdot w}P_{\theta}(q_{i}\mid q_{<i},Q_{l},Q_{d}).
$$

### 3.3 Think visually with spatio-temporal CoT

Autonomous driving planning requires not only understanding the current scene but also envisioning potential future developments to achieve forward-looking comprehension. This thinking process should resemble physical world simulation and imagination rather than purely text symbolic logical deduction. Since our model has already learned physical constraints through the progressive generation during pre-training, and considering efficiency, we no longer separately generate lane dividers, 3D detection, and future frames, but instead integrate all these results into a single unified frame. As shown in the right part of Figure 2, here, VLA serves as a world model to generate a unified image frame predicting the future world state: Inspired by visual prompting engineering \[redcircle\] that draws red circles on images to guide model attention and by VLIPP \[Yang2025VLIPPTP\] first predicts future bounding boxes to introduce physical priors when generating future frames, we represent future world spatial relationships through future red lane dividers and 3D detection boxes on the predicted unified frames. These coarse-grained visual cues direct the model’s attention toward drivable areas and critical objects in future scenes while enforcing physically plausible constraints. Meanwhile, the temporal relationships are represented by the ordinary future frame, where the dynamic evolution of visual content intuitively characterizes temporal progression and the inherent laws of scene development. Subsequently, spatio-temporal CoT $Q_{CoT}$ serves as an intermediate reasoning step, allowing the VLA to function as an inverse dynamics model that plans trajectory based on current observations and future predictions:

$$
P(W_{t}\mid I_{t},Q_{CoT},opt(T_{com},T_{ego}))=\Pi_{i=1}^{n}P_{\theta}(w_{i}\mid w_{<i},I_{t},Q_{CoT},opt(T_{com},T_{ego})).
$$

### 3.4 Training strategy

Our FSDrive can be initialized from any existing MLLM (e.g., Qwen2-VL, LLaVA), avoiding training from scratch and saving significant costs. During training, we fully fine-tune the LLM parameters while freezing all encoders. The training process is divided into two stages:

#### Stage 1: Unified pre-training.

Our objective is to preserve understanding capabilities of MLLMs through VQA tasks and activate their visual generation capabilities to predict future frames. VQA task data originates from OmniDrive-nuScenes \[Wang2024OmniDriveAH\]. We incorporate a large volume of unlabeled image data from nuScenes \[Caesar2019nuScenesAM\] for future frame prediction. To implement progressive easy-to-hard CoT, we integrate nuScenes annotated data to teach the model predicting image-formatted future lane dividers and 3D detection. Finally, we add future frame prediction with CoT datas containing intermediate reasoning steps. All the above understanding and generation tasks are trained together.

#### Stage 2: Supervised fine-tuning.

We focus on autonomous driving scene understanding and trajectory planning. Following OmniDrive \[Wang2024OmniDriveAH\], scene understanding utilizes DriveLM’s GVQA \[sima2023drivelm\] dataset. For trajectory planning, we follow VAD \[jiang2023vad, hu2023\_uniad\] using nuScenes, where our spatio-temporal CoT integrates the holistic future scene, explicit lane dividers, and 3D detection results into a single future frame as intermediate reasoning steps. We train these tasks simultaneously using a single model, enabling task-specific predictions during inference through different task prompts.

## 4 Experiments

### 4.1 Experimental settings

#### Datasets.

Following the previous methods \[jiang2023vad, gao2024vista, chen2025technical\], we evaluate trajectory planning and future frames generation on the nuScenes \[Caesar2019nuScenesAM\]. The nuScenes contains 1,000 scenes of approximately 20 seconds each captured by a 32-beam LiDAR and six cameras providing 360-degree field of view. Specifically, The dataset provides 28,130 (train), 6,019 (val), and 193,082 (unannotated) samples. Additionally, we conducted experiments on NAVSIM \[navsim\], a realistic scenario dataset designed for real-world planning. This dataset aims to highlight challenging driving scenarios involving dynamic changes in driving intent, while deliberately excluding simple situations such as static scenes or constant-speed driving.

Following the previous methods \[cube-llm, Wang2024OmniDriveAH\], we evaluate scene understanding on DriveLM \[sima2023drivelm\]. This dataset features keyframe descriptions paired with QA annotations covering full-stack autonomous driving (perception, prediction, planning), offering comprehensive language support for development.

#### Metrics.

We evaluate trajectory planning using L2 displacement error and collision rate following previous methods \[hu2023\_uniad, jiang2023vad, hu2022stp3\]. Notably, UniAD \[hu2023\_uniad\] computes L2 metrics and collision rate at each timestep, whereas ST-P3 \[hu2022stp3\] and VAD \[jiang2023vad\] considers the average of all previous time-steps. For a fair comparison, we adopted these two different calculation methods. Following existing methods \[wang2023drivedreamer, yang2024genad, 3DGAN\], we report Fréchet Inception Distance (FID) \[fid\] to measure the future frames generation quality. DriveLM GVQA \[sima2023drivelm\] metrics include language metrics like BLEU, ROUGE\_L, and CIDEr for text generation, the ChatGPT Score for open-ended Q&A and accuracy for multiple-choice questions. For NAVSIM \[navsim\], we adopt the official metrics for evaluation, especially PDMS.

#### Implementation details.

We initialize our model with Qwen2-VL-2B \[Qwen2-VL\] and pre-train it for 32 epochs to enable visual generation while preserving semantic understanding. During fine-tuning (12 epochs on 8 NVIDIA RTX A6000), we use $1\times 10^{-4}$ learning rate and batch size of 16. We expand the visual codebook of MoVQGAN \[Zheng2022MoVQgan\] to the vocabulary of the large language model and use its detokenizer to convert the visual tokens predicted by the large language model to the pixel space.

Table 1: End-to-end trajectory planning experiments on nuScenes \[Caesar2019nuScenesAM\]. We evaluated the L2 and collision metrics based on the distinct computational methodologies of ST-P3 \[hu2022stp3\] and UniAD \[hu2023\_uniad\], respectively. \* indicates that the ego status is additionally used. VAD \[jiang2023vad\] and UniAD \[hu2023\_uniad\] results are derived from BEV-Planner \[bev-planner\], while the remaining results are sourced from their respective papers.

<table><tbody><tr><td rowspan="3">Method</td><td colspan="8">ST-P3 metrics</td><td colspan="8">UniAD metrics</td><td rowspan="4">LLM</td></tr><tr><td colspan="4">L2 (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td colspan="4">Collision (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td colspan="4">L2 (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td colspan="4">Collision (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><td colspan="18">Non-Autoregressive methods</td></tr><tr><td>ST-P3* [ECCV22] <cite>[hu2022stp3]</cite></td><td>1.33</td><td>2.11</td><td>2.90</td><td>2.11</td><td>0.23</td><td>0.62</td><td>1.27</td><td>0.71</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>VAD [ICCV23] <cite>[jiang2023vad]</cite></td><td>0.69</td><td>1.22</td><td>1.83</td><td>1.25</td><td>0.06</td><td>0.68</td><td>2.52</td><td>1.09</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>VAD* [ICCV23] <cite>[jiang2023vad]</cite></td><td>0.17</td><td>0.34</td><td>0.60</td><td>0.37</td><td>0.04</td><td>0.27</td><td>0.67</td><td>0.33</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>UniAD [CVPR23] <cite>[hu2023_uniad]</cite></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.59</td><td>1.01</td><td>1.48</td><td>1.03</td><td>0.16</td><td>0.51</td><td>1.64</td><td>0.77</td><td>-</td></tr><tr><td>UniAD* [CVPR23] <cite>[hu2023_uniad]</cite></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.20</td><td>0.42</td><td>0.75</td><td>0.46</td><td>0.02</td><td>0.25</td><td>0.84</td><td>0.37</td><td>-</td></tr><tr><td>BEV-Planner [CVPR24] <cite>[bev-planner]</cite></td><td>0.30</td><td>0.52</td><td>0.83</td><td>0.55</td><td>0.10</td><td>0.37</td><td>1.30</td><td>0.59</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>BEV-Planner* [CVPR24] <cite>[bev-planner]</cite></td><td>0.16</td><td>0.32</td><td>0.57</td><td>0.35</td><td>0.00</td><td>0.29</td><td>0.73</td><td>0.34</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>PreWorld [ICLR25] <cite>[li2025semi]</cite></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.49</td><td>1.22</td><td>2.32</td><td>1.34</td><td>0.19</td><td>0.57</td><td>2.65</td><td>1.14</td><td>-</td></tr><tr><td colspan="18">Autoregressive methods</td></tr><tr><td>ELM [ECCV24] <cite>[zhou2024embodied]</cite></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.34</td><td>1.23</td><td>2.57</td><td>1.38</td><td>0.12</td><td>0.50</td><td>2.36</td><td>0.99</td><td>BLIP2-2.7B</td></tr><tr><td>FeD* [CVPR24] <cite>[zhang2023coaching]</cite></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.27</td><td>0.53</td><td>0.94</td><td>0.58</td><td>0.00</td><td>0.04</td><td>0.52</td><td>0.19</td><td>LLaVA-7B</td></tr><tr><td>OccWorld [ECCV24] <cite>[zheng2023occworld]</cite></td><td>0.39</td><td>0.73</td><td>1.18</td><td>0.77</td><td>0.11</td><td>0.19</td><td>0.67</td><td>0.32</td><td>0.52</td><td>1.27</td><td>2.41</td><td>1.40</td><td>0.12</td><td>0.40</td><td>2.08</td><td>0.87</td><td>GPT3-like</td></tr><tr><td>Doe-1 [arxiv24] <cite>[doe]</cite></td><td>0.37</td><td>0.67</td><td>1.07</td><td>0.70</td><td>0.02</td><td>0.14</td><td>0.47</td><td>0.21</td><td>0.50</td><td>1.18</td><td>2.11</td><td>1.26</td><td>0.04</td><td>0.37</td><td>1.19</td><td>0.53</td><td>Lumina-mGPT-7B</td></tr><tr><td>RDA-Driver* [ECCV24] <cite>[RDA-Driver]</cite></td><td>0.17</td><td>0.37</td><td>0.69</td><td>0.40</td><td>0.01</td><td>0.05</td><td>0.26</td><td>0.10</td><td>0.23</td><td>0.73</td><td>1.54</td><td>0.80</td><td>0.00</td><td>0.13</td><td>0.83</td><td>0.32</td><td>LLaVA-7B</td></tr><tr><td>EMMA* [arxiv24] <cite>[Hwang2024EMMAEM]</cite></td><td>0.14</td><td>0.29</td><td>0.54</td><td>0.32</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>Gemini 1-1.8B</td></tr><tr><td>OmniDrive [CVPR25] <cite>[Wang2024OmniDriveAH]</cite></td><td>0.40</td><td>0.80</td><td>1.32</td><td>0.84</td><td>0.04</td><td>0.46</td><td>2.32</td><td>0.94</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>LLaVA-7B</td></tr><tr><td>OmniDrive* [CVPR25] <cite>[Wang2024OmniDriveAH]</cite></td><td>0.14</td><td>0.29</td><td>0.55</td><td>0.33</td><td>0.00</td><td>0.13</td><td>0.78</td><td>0.30</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>LLaVA-7B</td></tr><tr><td>FSDrive (ours)</td><td>0.28</td><td>0.52</td><td>0.80</td><td>0.53</td><td>0.06</td><td>0.13</td><td>0.32</td><td>0.17</td><td>0.40</td><td>0.89</td><td>1.60</td><td>0.96</td><td>0.07</td><td>0.12</td><td>1.02</td><td>0.40</td><td>Qwen2-VL-2B</td></tr><tr><td>FSDrive* (ours)</td><td>0.14</td><td>0.25</td><td>0.46</td><td>0.28</td><td>0.03</td><td>0.06</td><td>0.21</td><td>0.10</td><td>0.18</td><td>0.39</td><td>0.77</td><td>0.45</td><td>0.00</td><td>0.06</td><td>0.42</td><td>0.16</td><td>Qwen2-VL-2B</td></tr><tr><td>FSDrive (ours)</td><td>0.29</td><td>0.57</td><td>0.94</td><td>0.60</td><td>0.04</td><td>0.14</td><td>0.38</td><td>0.19</td><td>0.36</td><td>1.01</td><td>1.90</td><td>1.09</td><td>0.08</td><td>0.34</td><td>1.11</td><td>0.51</td><td>LLaVA-7B</td></tr><tr><td>FSDrive* (ours)</td><td>0.13</td><td>0.28</td><td>0.52</td><td>0.31</td><td>0.03</td><td>0.07</td><td>0.24</td><td>0.12</td><td>0.22</td><td>0.51</td><td>0.94</td><td>0.56</td><td>0.02</td><td>0.07</td><td>0.53</td><td>0.21</td><td>LLaVA-7B</td></tr></tbody></table>

Table 2: Performance comparison on NAVSIM navtest using closed-loop metrics. All the methods only use images as input and do not use lidar.

| Method | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf. $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| VADv2 \[arXiv24\] \[chen2024vadv2\] | 97.2 | 89.1 | 91.6 | 100 | 76.0 | 80.9 |
| UniAD \[CVPR23\] \[hu2023\_uniad\] | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| DiffusionDrive-Cam \[CVPR25\] \[diffusiondrive\] | 97.8 | 92.2 | 92.6 | 99.9 | 78.9 | 83.6 |
| LTF \[TPAMI23\] \[ltf\] | 97.4 | 92.8 | 92.4 | 100 | 79.0 | 83.8 |
| PARA-Drive \[CVPR24\] \[Weng2024paradrive\] | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| LAW \[ICLR25\] \[law\] | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 |
| FSDrive (ours) | 98.2 | 93.8 | 93.3 | 99.9 | 80.1 | 85.1 |

### 4.2 Main results

#### End-to-End trajectory planning.

We present trajectory planning performance on nuScenes following previous methods \[jiang2023vad, hu2023\_uniad\] in Table 1. When using ego status, FSDrive surpasses previous SOTA methods using ego status in ST-P3 and UniAD metrics. However, following BEV-Planner \[bev-planner\] findings about ego-status’s performance boost, we prioritize non-ego-status evaluations. Compared to non-autoregressive (e.g., UniAD) and autoregressive methods (e.g., OmniDrive), FSDrive demonstrates superior effectiveness. Notably, FSDrive outperforms Doe-1 \[doe\] which also enables vision generation (L2: 0.53 vs. 0.70 and 0.96 vs. 1.26; collision: 0.19 vs. 0.21 and 0.40 vs. 0.53), indicating limitations in their VQ-VAE-based discrete visual features for understanding. For a fair comparison, we also used LLaVA like methods \[Wang2024OmniDriveAH, RDA-Driver, zhang2023coaching, xie2025seqgrowgraph\]. Under the corresponding settings, FSDrive still has excellent competitiveness, indicating that FSDrive can be widely applied to any existing MLLM.

Table 3: Future frames generation results on the nuScenes \[Caesar2019nuScenesAM\] dataset.

<table><thead><tr><th rowspan="2">Method</th><th>DriveGAN</th><th>DriveDreamer</th><th>Drive-WM</th><th>GenAD</th><th>GEM</th><th>Doe-1</th><th rowspan="2">FSDrive</th></tr><tr><th>[CVPR21 <cite>[kim2021drivegan]</cite>]</th><th>[ECCV24 <cite>[wang2023drivedreamer]</cite>]</th><th>[CVPR24 <cite>[wang2023driving]</cite>]</th><th>[CVPR24 <cite>[yang2024genad]</cite>]</th><th>[CVPR25 <cite>[Hassan2024GEMAG]</cite>]</th><th>[arxiv24 <cite>[doe]</cite>]</th></tr></thead><tbody><tr><th>Type</th><td>GAN</td><td>Diffusion</td><td>Diffusion</td><td>Diffusion</td><td>Diffusion</td><td>Autoregressive</td><td>Autoregressive</td></tr><tr><th>Resolution</th><td>256 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 256</td><td>128 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 192</td><td>192 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 384</td><td>256 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 448</td><td>576 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 1024</td><td>384 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 672</td><td>128 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 192</td></tr><tr><th>FID <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><td>73.4</td><td>52.6</td><td>15.8</td><td>15.4</td><td>10.5</td><td>15.9</td><td>10.1</td></tr></tbody></table>

#### Results on NAVSIM.

Table 2 shows the evaluation results for NAVSIM \[navsim\]. All approaches rely exclusively on camera input, with no lidar data being used. Achieving a PDMS score of 85.1, FSDrive outperforms prior camera-only methods like LAW \[law\] and DiffusionDrive-Cam \[diffusiondrive\], thus showcasing its efficacy in the pseudo closed-loop setting.

#### Evaluation of generation results.

Although we generate future frames as CoT for trajectory planning, we still validate visual quality via FID in Table 3. To enable rapid generation for real-time driving, we generate frames at 128 $\times$ 192 resolution. Our autoregressive FSDrive achieves competitive performance against specialized diffusion models. Compared to Doe-1 \[doe\] which employs the vision generation MLLM Lumina-mGPT 7B \[liu2024lumina-mgpt\], FSDrive 2B maintains superior advantages, indicating that the visual generation capabilities of MLLM can be effectively unlocked even with minimal data.

#### Results on DriveLM dataset.

FSDrive’s scene understanding was evaluated on DriveLM in Table 4, achieving 0.57 and outperforming recent methods like Cube-LLM \[cube-llm\] and OmniDrive \[Wang2024OmniDriveAH\]. This highlights the effectiveness of FSDrive pre-training paradigm for generation and understanding.

Table 4: Results on DriveLM \[sima2023drivelm\] GVQA benchmark.

| Method | Accuracy $\uparrow$ | ChatGPT $\uparrow$ | BLEU\_1 $\uparrow$ | ROUGE\_L $\uparrow$ | CIDEr $\uparrow$ | Match $\uparrow$ | Final Score $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DriveLM baseline \[sima2023drivelm\] | 0.00 | 0.65 | 0.05 | 0.08 | 0.10 | 0.28 | 0.32 |
| Cube-LLM \[cube-llm\] | 0.39 | 0.89 | 0.16 | 0.20 | 0.31 | 0.39 | 0.50 |
| TrackingMeetsLMM \[ishaq2025trackingmeets\] | 0.60 | 0.58 | 0.72 | 0.72 | 0.04 | 0.36 | 0.52 |
| SimpleLLM4AD \[Zheng2024SimpleLLM4ADAE\] | 0.66 | 0.57 | 0.76 | 0.73 | 0.15 | 0.35 | 0.53 |
| OmniDrive \[Wang2024OmniDriveAH\] | 0.70 | 0.65 | 0.52 | 0.73 | 0.13 | 0.37 | 0.56 |
| FSDrive (ours) | 0.72 | 0.63 | 0.76 | 0.74 | 0.17 | 0.39 | 0.57 |

![[vis.png|Refer to caption]]

Figure 3: Qualitative analysis of our CoT. The red trajectory is the prediction and the green is the GT.

Table 5: Ablation results of pre-training.

<table><thead><tr><th rowspan="2">VQA</th><th>Future</th><th>Future</th><th>Future</th><th colspan="4">L2 (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th colspan="4">Collision (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th></tr><tr><th>frames</th><th>3D detection</th><th>lane divider</th><th>1s</th><th>2s</th><th>3s</th><th>Avg.</th><th>1s</th><th>2s</th><th>3s</th><th>Avg.</th></tr></thead><tbody><tr><th><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></th><th><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></th><th><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></th><th><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></th><td>0.45</td><td>1.09</td><td>2.12</td><td>1.22</td><td>0.12</td><td>0.43</td><td>1.45</td><td>0.67</td></tr><tr><th><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></th><th><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></th><th><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></th><th><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></th><td>0.46</td><td>1.07</td><td>2.04</td><td>1.19</td><td>0.12</td><td>0.42</td><td>1.42</td><td>0.65</td></tr><tr><th><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></th><th><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></th><th><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></th><th><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></th><td>0.39</td><td>0.96</td><td>1.71</td><td>1.02</td><td>0.10</td><td>0.38</td><td>1.32</td><td>0.60</td></tr><tr><th><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></th><th><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></th><th><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></th><th><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></th><td>0.46</td><td>1.06</td><td>1.99</td><td>1.17</td><td>0.10</td><td>0.37</td><td>1.35</td><td>0.61</td></tr><tr><th><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></th><th><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></th><th><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></th><th><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></th><td>0.42</td><td>0.97</td><td>1.80</td><td>1.06</td><td>0.13</td><td>0.41</td><td>1.40</td><td>0.65</td></tr><tr><th><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></th><th><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></th><th><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></th><th><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></th><td>0.39</td><td>0.91</td><td>1.63</td><td>0.98</td><td>0.09</td><td>0.36</td><td>1.33</td><td>0.58</td></tr></tbody></table>

### 4.3 Ablation study

In this section, unless otherwise specified, we evaluate the computing metrics of UniAD \[hu2023\_uniad\] based on the Qwen2-VL-2B model \[Qwen2-VL\] and do not use the ego status.

#### Qualitative analysis.

We evaluate our CoT’s effectiveness in Figure 3. Without spatial-temporal CoT, erroneous navigation inputs caused significant trajectory deviations and potential collisions. Use correct instruction when reasoning our CoT, while still employing wrong instruction for planning. However, FSDrive mitigated instruction errors through observation-based trajectory planning and future prediction, demonstrating its inverse dynamics modeling capability.

Table 6: Ablation results of different CoT.

<table><thead><tr><th rowspan="2">Type</th><th colspan="4">L2 (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th colspan="4">Collision (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th></tr><tr><th>1s</th><th>2s</th><th>3s</th><th>Avg.</th><th>1s</th><th>2s</th><th>3s</th><th>Avg.</th></tr></thead><tbody><tr><th>None</th><td>0.39</td><td>0.91</td><td>1.63</td><td>0.98</td><td>0.09</td><td>0.36</td><td>1.33</td><td>0.58</td></tr><tr><th>Text CoT</th><td>0.39</td><td>0.92</td><td>1.61</td><td>0.97</td><td>0.10</td><td>0.29</td><td>1.21</td><td>0.53</td></tr><tr><th>Image-text CoT</th><td>0.38</td><td>0.90</td><td>1.65</td><td>0.98</td><td>0.09</td><td>0.25</td><td>1.15</td><td>0.50</td></tr><tr><th>Spatio-temporal CoT</th><td>0.40</td><td>0.89</td><td>1.60</td><td>0.96</td><td>0.07</td><td>0.12</td><td>1.02</td><td>0.40</td></tr></tbody></table>

Table 7: Ablation experiments of future frames generation.

| Pre-training Data | Progressive Method | FID $\downarrow$ |
| --- | --- | --- |
| None | $\times$ | 29.4 |
| $\sim$ 100k | $\times$ | 16.2 |
| $\sim$ 200k | $\times$ | 12.7 |
| $\sim$ 200k | $\checkmark$ | 10.1 |

#### Pre-training ablation study.

The impact of pre-training on trajectory planning is summarized in Table 5. Pure VQA tasks show negligible effects. Future frame generation pre-training improves L2 by 16.4% and collisions by 15.8%, validating world-model-based prediction’s effectiveness in capturing physical dynamics. 3D detection and lane divider pre-training yield moderate gains in L2/collision metrics respectively. The combined understanding and generation pre-training achieves better performance, demonstrating our unified paradigm’s capacity to enhance scene representation and effectively learn physical laws, thereby strengthening spatial understanding capabilities.

#### Results of different CoT.

Ablation studies on CoT variants in Table 6 show marginal L2 changes but notable collision rate improvements. Pure text CoT (8.6% improvement) exhibits limited representation capability due to unimodal perception. Compared to text CoT, image-text CoT (combining future frames with textual perception) shows insignificant gains due to the inconsistent modalities between CoTs. The proposed spatio-temporal CoT achieves 31% improvement, demonstrating that unified image-based reasoning effectively identifies future collision risks.

#### Ablation study on generation results.

We conduct ablation studies on future frames generation in Table 7. The upper part of Table 7 shows that larger pre-training datasets improve MLLM’s visual generation capability. Despite being much smaller (200K vs. 100M in previous work \[wu2024janus\]), our data achieves more robust visual generation. Scaling datasets may further enhance performance. The lower part of Table 7 confirms our progressive method improves autoregressive image generation.

## 5 Conclusion

This paper proposes FSDrive, an autonomous driving framework based on spatio-temporal CoT that enables VLAs to think visually. By unifying future scene generation and perception results through intermediate image-form reasoning steps, our FSDrive eliminates the semantic gap caused by cross-modal conversions and establishes an end-to-end visual reasoning pipeline. The VLA serves dual roles: as a world model that predicts future image frames with lane divider and 3D detection, and as an inverse dynamics model that plans trajectory based on both current observations and future predictions. To enable visual generation in VLAs, we present a pretraining paradigm that unifies visual generation and understanding, along with a progressive easy-to-hard visual CoT to enhance autoregressive image generation. Extensive experimental results demonstrate the effectiveness of the proposed FSDrive method, advancing autonomous driving towards visual reasoning.

#### Limitations and broader impacts.

Though autonomous driving requires surrounding environmental awareness, considering real-time efficiency, we currently only generate future frames for the front-view. Future work can attempt to generate Surround views to achieve safer autonomous driving. Moreover, more robust visual quality can be achieved in future work through the use of larger training datasets and a more advanced unified paradigm that integrates generation and understanding. In terms of impact, the ethical challenges posed by LLMs extend to autonomous driving. Advances in technology and regulation will drive development of safer, more efficient systems.

#### Acknowledgments.

This work was support by the National Natural Science Foundation of China No. 62572385, the Fundamental Research Funds for the Central Universities No. xxj032023020, and CAAI-CANN Open Fund, developed on OpenI Community.