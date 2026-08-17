---
title: "DriveLaW: Unifying Planning and Video Generation in a Latent Driving World"
source: "https://arxiv.org/html/2512.23421v3"
author:
published:
created: 2026-08-17
description:
tags:
  - "clippings"
---
Tianze Xia Affiliation: Huazhong University of Science and Technology    Yongkang Li Affiliation: Huazhong University of Science and Technology    Lijun Zhou    Jingfeng Yao Affiliation: Huazhong University of Science and Technology    Kaixin Xiong Affiliation: Xiaomi EV    Haiyang Sun    Bing Wang Affiliation: Xiaomi EV    Kun Ma Affiliation: Xiaomi EV    Guang Chen Affiliation: Xiaomi EV    Hangjun Ye Affiliation: Xiaomi EV    Wenyu Liu Affiliation: Huazhong University of Science and Technology    Xinggang Wang Affiliation: Huazhong University of Science and Technology

###### Abstract

World models have become crucial for autonomous driving, as they learn how scenarios evolve over time to address the long-tail challenges of the real world. However, current approaches relegate world models to limited roles: they operate within ostensibly unified architectures that still keep world prediction and motion planning as decoupled processes. To bridge this gap, we propose DriveLaW, a novel paradigm that unifies video generation and motion planning. By directly injecting the latent representation from its video generator into the planner, DriveLaW ensures inherent consistency between high-fidelity future generation and reliable trajectory planning. Specifically, DriveLaW consists of two core components: DriveLaW-Video, our powerful world model that generates high-fidelity forecasting with expressive latent representations, and DriveLaW-Act, a diffusion planner that generates consistent and reliable trajectories from the latent of DriveLaW-Video, with both components optimized by a three-stage progressive training strategy. New state-of-the-art results across both tasks demonstrate the power of our unified paradigm. DriveLaW not only significantly advances video prediction, surpassing the previous best-performing work by 33.3% in FID and 1.8% in FVD, but also sets a new record on the NAVSIM planning benchmark. Code available at [https://github.com/xiaomi-research/drivelaw](https://github.com/xiaomi-research/drivelaw).

<sup>†</sup>

## 1 Introduction

Autonomous driving has advanced rapidly in recent years, driven by significant progress in perception [^45] [^75] [^47] [^32] and planning [^31] [^34] [^48] [^44]. However, existing systems remain brittle in long-tail and rare scenarios, degrading closed-loop performance. Recent work has introduced world models that forecast the future evolution of driving scenes from past multi-view observations and ego-state to address long-tail brittleness. They can synthesize downstream task data [^20] [^29] [^18] [^68] [^40] for rare scenarios, enabling policy learning in simulation [^17] [^77] [^39], and provide auxiliary future supervision [^42] [^41] [^83] [^54] [^69], both of which contribute to improving generalization and robustness under distribution shift.

Despite the generalization gains from learning physical regularities through large-scale video generation, current world models’ contributions to autonomous driving planning remain either indirect or merely parallel to the planner, rather than tightly coupled with decision making. Specifically, in terms of their role in planning, the existing world-model approaches fall into three categories. First, *world-model simulators* [^38] [^22] [^86] [^77] [^20] [^29] synthesize downstream data or serve as closed-loop environments to guide policy learning, which is indirect and does not transmit the model’s physical understanding into the planner’s state. Second, *world-model supervision* [^42] [^41] [^84] [^67] predicts future visual or affordance signals to supervise future frames, occupancy, or trajectories, improving foresight but keeping planning externally specified. Third, *unified world-model* [^81] [^4] [^79] co-generates videos and trajectories, yielding tighter coupling between perception and control and improved temporal consistency. However, current instantiations of this paradigm still fall short of fully realizing the intended tight coupling. Methods such as Epona [^81] and DriveVLA-W0 [^42] decouple generation and planning, training the video generator and the policy head as separate modules. Despite evidence that video generators encode strong world understanding and perceptual priors, these approaches do not leverage the generator’s internal latents as the planning state, leaving a gap between visual imagination and action selection.

In this paper, we propose DriveLaW, a latent world model that unifies generation and planning through a shared latent-space representation. Instead of treating generation and planning as two parallel modules, we chain generation and planning inspired by Genie envisioner [^49] as shown in Fig. 1, leveraging the strong driving representations learned by the video generator from large-scale video generation to perform trajectory planning. Latent features from the video generator, learned on large-scale driving videos, encapsulate scene semantics, agent dynamics, and physical regularities in a compact representation. The Action DiT (Diffusion Transformer [^56]), conditioned on these latents, generates temporally distribution-robust trajectories and improves closed-loop stability. In comparison, our chained design offers three advantages over parallel baselines: (1) it fully exploits representations learned from large-scale video pretraining, unifying generation and planning in a shared latent space; (2) its training regime avoids gradient interference between the video generator and the planner; (3) cascading ensures consistency between generated visual detail and the planned trajectories. However, high-fidelity video synthesis and real-time stable planning are inherently in tension. To address this, we design DriveLaW-Video with a spatiotemporal VAE and an efficient Video DiT, and introduce a noise reinjection mechanism to balance aggressive compression with visual fidelity. On the control side, DriveLaW-Act uses a vanilla DiT trained with a flow-matching objective to produce smooth, reliable trajectories. To harmonize optimization between generation and planning, we adopt a three-stage progressive curriculum that first learns long-horizon motion, then refines spatial detail, and finally chains video latents into the planner for stable training.

We extensively evaluate DriveLaW on nuScenes [^9] for video generation and on NAVSIM [^16] for trajectory planning. On nuScenes, DriveLaW achieves state-of-the-art generation quality, outperforming both pure video generators and unified gen–plan baselines. On NAVSIM, our planner attains strong closed-loop metrics without any post-training (e.g., RL) or post-processing (e.g., scorers), highlighting the strength of using video-generator representations for control. Ablations and scaling studies further substantiate our design. The main contributions of this work are as follows:

- We propose DriveLaW, a unified world model that shifts from parallel to chained generation and planning. We demonstrate that the latent representations learned from large-scale video generation possess superior semantic coherence and spatial structure compared to traditional Bird’s Eye View (BEV) or Vision-Language Models(VLM) features. By injecting these rich generative priors into the planner, we effectively bridge the gap between visual forecasting and action decision-making.
- We design a specialized architecture comprising DriveLaW-Video and DriveLaW-Act. The former incorporates a novel noise reinjection mechanism to resolve structural inconsistencies and blurring in high-speed scenarios, while the latter employs a diffusion planner directly conditioned on video latents. A three-stage progressive training strategy is introduced to resolve the optimization tension between high-fidelity video synthesis and reliable trajectory generation.
- We validate DriveLaW on standard autonomous driving benchmarks. It achieves state-of-the-art FID and FVD scores in single-view video generation on nuScenes and sets a new record for closed-loop planning metrics (PDMS) on the NAVSIM benchmark, surpassing previous world-model approaches and confirming the effectiveness of our unified paradigm without relying on post-training or auxiliary scorers.

## 2 Related Work

### 2.1 World Models for Autonomous Driving

World models [^7] [^8] [^3] aim to internalize the physical structure and dynamics of the real world into a predictive latent representation for imagination, control, and planning. Recent studies introduce world models into autonomous driving for scene generation [^68] [^82] [^20] [^26] [^18] [^29] [^65] [^19] [^22] [^21] [^53], simulation evaluation [^55] [^39] [^77] [^74] [^86] and decision-making [^81] [^69] [^42] [^41] [^43]. Early works [^29] [^18] [^20] treat this task as conditional multi-view video generation, prioritizing multi-camera geometry consistency. To more fundamentally capture the 3D spatiotemporal structure, OccWorld [^84] and OccSora [^67] introduce an occupancy-centric representation. UniScenes [^38] and Genesis [^22] further achieve joint generation of multi-modal signals (RGB, LiDAR, occupancy), primarily serving as data-centric generators for downstream task training. Meanwhile, some works [^77] [^86] [^74] [^39] [^17] integrate world models as simulation engines directly into autonomous driving, focusing on closed-loop evaluation and scene reconstruction. For instance, HUGSIM [^86], ReconDreamer-R [^55], and RAD [^17] introduce 3D Gaussian Splatting to reconstruct scenes for closed-loop simulation. ReSim [^77] and OmniNWM [^39] focus on behavioral simulation, synthesize videos conditioned on candidate trajectories, and provide reward signals to guide trajectory filtering and reinforcement learning.

However, these approaches provide only indirect guidance to the planning module via data generation and simulation, whereas recent studies aim to integrate generative modeling and decision-making within a unified world model. DrivingGPT [^13] adopts a GPT-style autoregressive policy to generate future videos and trajectories. Epona [^81] uses an autoregressive diffusion scheme to produce videos and trajectories in parallel. VaViM/VaVAM [^4] and DriveVLA-W0 [^42] employ a $\pi$ 0-like [^5] mixture-of-transformers [^46] architecture that autoregressively generates both modalities within a single model. FSDrive [^79] unifies generation and planning via visual spatio-temporal Chain-of-Thought reasoning for end-to-end driving. While this paradigm represents a significant step forward, it typically treats video and trajectory generation as two independent output streams. This design can lead to a representation disconnect as the planning trajectory is not directly grounded in the internal features that govern video synthesis. In contrast, we argue that the rich, spatiotemporally grounded features learned by video generators constitute a powerful yet under-explored resource for planning. To bridge this gap, DriveLaW is the first to exploit mid-level latents from a video generator as planning representations, enabling more stable closed-loop driving.

### 2.2 Video Generation

Video generation has emerged as a core component of autonomous driving [^52] [^68] [^69] [^76], underpinning vision-centric world models [^29] [^30]. Beyond static scene prediction, it captures temporal continuity and rich dynamics [^64] [^51] [^6] [^25] [^28], enabling models to learn how agents and environments evolve over time. GAIA-1 [^29] innovatively adopts an autoregressive framework that outputs discrete image tokens to generate autonomous-driving scene videos. DriveDreamer [^82], Panacea [^70], DrivingDiffusion [^40] and MagicDrive [^18] condition diffusion models on geometric features such as Bird’s Eye View (BEV) maps and 3D boxes to synthesize controllable driving scenes. MagicDrive-V2 [^19], MiLA [^65], and GAIA-2 [^60] generate long-horizon, high-fidelity driving videos via latent diffusion with structured geometric and action conditioning. TeraSim-World [^66] and Cosmos-Drive [^59] build large-scale synthetic data pipelines for autonomous driving, producing diverse and controllable video samples for downstream perception training and evaluation. In summary, while existing methods have advanced the state of the art in visual synthesis, they predominantly treat the video generator as a renderer. We posit that the internal activations of these powerful generators encode a rich, temporally coherent understanding of scene dynamics and geometry. DriveLaW repurposes the video generator as a feature extractor and pairs it with a diffusion planner to enable end-to-end driving.

### 2.3 Diffusion Policies for Autonomous Driving

Recent work brings diffusion policies [^14] to autonomous driving, leveraging their strength in temporal action modeling. DiffusionDrive [^48] introduces truncated diffusion and anchor-initialized noise to achieve real-time, multimodal trajectory planning. Diffusion Planner [^85] jointly generates trajectories for the ego vehicle and surrounding agents via diffusion to model interactive driving scenes. ReCogDrive [^44] couples a VLM with a diffusion planner, injecting driving cognition priors into the diffusion process to enable efficient, continuous action generation. GoalFlow [^73] employs flow matching to generate goal-point guidance, enabling safe and stable driving. We innovatively chain a Video DiT with a diffusion planner, distill driving priors from large-scale driving videos, and inject them into a diffusion planner to enable stable closed-loop driving.

## 3 Method

![[CVPR-Fig2.png|Refer to caption]]

Figure 1: Overview of the overall architecture of DriveLaW. The model first encodes historical observations (images, actions) into a unified latent world representation through a powerful video diffusion model. In order to improve the generation quality, we introduced the Noise Reinjection mechanism to explore and select the optimal generation path in the early stage of denoising. The denoised video latents produced by the Video DiT are then passed as conditioning signals to the action planner. Leveraging these latents, the lightweight Action DiT predicts future trajectories that are aligned with the visual scene evolution. In this chained design, the Video Model and Action Model share the same latent-space representation.

In this section, we first investigate learning generalizable driving representations from a video generator. We then introduce DriveLaW, a world model that unifies generation and planning through a shared latent-space representation. Subsequently, we describe DriveLaW-Video, a spatiotemporal video generator, and DriveLaW-Act, a diffusion-based planner. Finally, we present a three-stage training framework that produces high-quality videos and stable trajectories.

### 3.1 Learning Generalizable Driving Representations from Video Generators

World models such as Genie [^8] and Cosmos [^2] learn real-world structure by training on large-scale video generation, and many studies [^72] [^23] indicate that video generators internalize physical regularities and act as strong zero-shot learners. In autonomous driving, real scenes provide virtually unlimited video while dense annotation is costly, so we propose to learn driving representations from large-scale driving video generation, akin to how humans acquire driving competence. Concretely, let $\mathcal{E}$ be the video encoder and $z$ the latent representation of a clip $x_{0}$ with $z=\mathcal{E}(x_{0})$. A generic denoiser produces a latent denoising trajectory $\{z_{t}\}_{t\in\mathcal{T}}$ under conditioning $c$ via a single-step update

$$
z_{t-1}=\Psi_{\theta}(z_{t},t,c),
$$

where $\Psi_{\theta}$ denotes the learned denoising operator and $t$ indexes the inference schedule. We extract mid-denoising features

$$
h_{t}=\phi_{\theta}(z_{t}),\quad t\in\mathcal{T},
$$

and select one or a small set of timesteps $t^{\star}$ to form the perception latent $h=h_{t^{\star}}$. This latent encodes driving cognition priors distilled from generation and is fed to the planning module for stable closed-loop driving.

### 3.2 DriveLaW

As illustrated in Fig. 1, DriveLaW is a unified framework composed of a *DriveLaW-Video* and a *DriveLaW-Act*. The video model, e.g., LTX-Video [^24], first encodes past driving frames with a spatiotemporal VAE and encodes textual prompts with a text encoder. A stack of Video DiT blocks then performs latent-space denoising, and the VAE decoder reconstructs the video. Concurrently, action noise, ego status, and high-level commands are encoded and fed into the action model. Video latents from the Video DiT serve as conditioning signals, guiding the Action DiT to output the final trajectory. The Video DiT and Action DiT are chained and trained to learn driving representations from large-scale video generation, providing a shared basis for planning.

### 3.3 DriveLaW-Video: Spatiotemporal World Generator

#### Spatiotemporal VAE.

We employ a high-compression spatiotemporal VAE [^24] to efficiently model long-horizon driving scenarios. The VAE encodes each video clip into a causal latent space with $32\times 32\times 8$ spatial-temporal resolution and 128 channels, achieving a compression ratio of $1{:}192$ (pixel-to-token ratio $1{:}8192$). This is significantly more compact than typical $1{:}48$ [^58] [^37] or $1{:}96$ [^36] [^78] compressions, enabling longer prediction horizons under the same computational budget—critical for modeling long-term dependencies like traffic light changes and vehicle dynamics. The encoder uses 3D causal convolutions to ensure each timestep depends only on past and current frames, preventing temporal information leakage.

Unlike conventional pipelines that complete all reverse diffusion steps in the latent space before a single decoding pass, we employ a hybrid approach. We decode at a late stage of the rectified-flow schedule ($t=t_{1}$) and perform a final refinement in pixel space:

$$
x_{0}=D(z_{t_{1}},t_{1}),
$$

where

$$
z_{t_{1}}=(1-t_{1})z_{0}+t_{1}\epsilon,\quad\epsilon\sim\mathcal{N}(0,\mathbf{I}).
$$

Here, $D$ denotes a time-conditioned denoising decoder trained with pixel-space losses. Since $D$ maps latents to pixels, it is used only for the final step. Performing the last step in pixel space recovers high-frequency details, e.g., highlights, dynamic shadows, fine road textures, without an extra super-resolution module and adds minimal overhead.

#### Video Transformer Architecture.

After compression to the high-dimensional latent space, the serialized spatiotemporal tokens are processed by a three-dimensional Transformer adapted from PixArt- $\alpha$ [^11] for full spatiotemporal modeling. Each block uses self-attention [^63] for global spatiotemporal modeling and cross-attention to integrate task-specific conditions (navigation commands, visual cues). We apply RMSNorm [^80] to queries and keys for attention stability. To enhance spatiotemporal consistency under different resolutions and frame rates, we use Rotary Positional Embedding [^61] with normalized fractional coordinates, reducing spatial drift in long-horizon predictions.

To align video generation with realistic driving dynamics, we introduce a motion-conditioned prompting mechanism that converts recent ego-vehicle kinematics into structured natural language instructions rather than using a dedicated motion encoder. This approach leverages pre-trained text-to-video architectures directly, provides interpretable control, unifies static and dynamic conditioning, and improves cross-dataset generalization by avoiding numeric encodings tied to data-specific scales.

#### Noise Reinjection.

![[simplerenoise_compressed.png|Refer to caption]]

Figure 2: Restoring Structural and Temporal Consistency via Noise Reinjection. This comparison highlights the impact of our method. The baseline generation shows significant degradation, including (a) blurring, (b) structural inconsistency, and (c) artifacts. By integrating noise reinjection, our model preserves sharp details, maintains object structures, and produces clean, artifact-free frames, demonstrating a crucial improvement in video quality.

In high-speed driving video generation, long-range motion and large displacements often cause perceptual degradation: boundaries are over-smoothed, fine textures fade, and blur and ghosting accumulate, undermining the structural consistency of vehicles, lane markings, and distant backgrounds. To mitigate this, we adapt the principle of iterative refinement found in works like DiffuseSlide [^33]. Unlike methods that apply noise globally, we introduce a more targeted strategy. Our approach selectively re-injects noise into high-frequency regions before each main denoising step, compelling the model to actively regenerate details rather than smooth them over.

At each denoising step $t$ with current latent $L_{t}$, we first identify regions likely to contain high-frequency details. To do this stably, we perform an initial prediction of the clean latent $\hat{L}_{0}=\Psi_{\theta}(L_{t},t)$. Crucially, we then decode this latent into a temporary pixel-space image using the VAE decoder, $\hat{I}_{0}=D(\hat{L}_{0})$, and convert each frame to its grayscale representation, $G_{f}$. This entire process of computing the high-frequency mask occurs in the pixel domain for maximum fidelity. We then apply a discrete Laplacian kernel $K_{L}$ to obtain a response map $H_{f}=|G_{f}*K_{L}|$, compute an adaptive threshold $au=\beta\cdot\mathrm{std}(H_{f})$, and define the high-frequency mask:

$$
M_{f}(x,y)=\begin{cases}1,&H_{f}(x,y)>au\\
0,&\text{otherwise}.\end{cases}
$$

To apply this mask back in the latent space, we downsample $M_{f,\text{pixel}}$ from the image resolution ($H\times W$) to the latent resolution ($h\times w$) using nearest-neighbor interpolation, resulting in the final latent-space mask $M$. Next, we create a selectively perturbed latent $L^{\prime}_{t}$ by injecting a small amount of controlled noise only in the masked regions:

$$
L^{\prime}_{t}=L_{t}+\sigma^{\prime}_{t}\cdot M\odot\varepsilon_{t},\quad\varepsilon_{t}\sim\mathcal{N}(0,\mathbf{I}),
$$

where $\sigma^{\prime}_{t}$ is a manually-tuned noise strength for the reinjection step, and $M$ is the high-frequency mask.

Finally, this perturbed latent $L^{\prime}_{t}$ is used as the input to the full Transformer-based denoising operator $\Psi_{\theta}$ to compute the latent for the next step, $L_{t-\Delta t}$. This forces the model to leverage its powerful generative prior to "inpaint" the noisy regions with plausible high-frequency details consistent with the rest of the scene. As shown in Fig. 2, this targeted approach restores sharpness and texture to dynamic objects and road markings while preserving the natural smoothness of areas like the sky, achieving a better balance between detail restoration and artifact suppression.

### 3.4 DriveLaW-Act: A Diffusion-Based Planner

Inspired by [^85] [^48] [^44], we adopt a diffusion-based planner that generates continuous, temporally smooth trajectories. Specifically, we encode a sampled noise action and the driving context as follows. Concretely, we encode a noised action $a_{t}$, ego status $s_{t}$, and high-level command $g_{t}$ using an action encoder and a context encoder respectively:

$$
\displaystyle h_{\mathrm{act}}
$$
 
$$
\displaystyle=E_{\mathrm{act}}(a_{t}),\qquad h_{\mathrm{ctx}}=E_{\mathrm{ctx}}([s_{t};g_{t}]),
$$

where $a_{t}=(1-t)a_{0}+t\epsilon$ and $\epsilon\sim\mathcal{N}(0,\mathbf{I})$. Meanwhile, during the first denoising step of the Video DiT, the latent features from each Transformer block are cached as $\{f_{1},f_{2},\dots,f_{B}\}$. For every step in the flow-matching process, the Action DiT (denoted as $f_{\theta}$) takes the encoded noise action $h_{\mathrm{act}}$ and the continuous time $t$ as input, conditioned on both the context embedding $h_{\mathrm{ctx}}$ and the cached video features $\{f_{i}\}$:

$$
f_{\theta}(a_{t},t)=\mathrm{DiT}_{\mathrm{act}}\big([\,h_{\mathrm{act}};\,t\,]\,\big|\,h_{\mathrm{ctx}},\{f_{i}\}_{i=1}^{B}\big),
$$

where $t$ denotes the continuous timestep.

We train the planner using a flow-matching [^50] objective that aligns the model’s predicted output $f_{\theta}$ with the target flow:

$$
\mathcal{L}_{\mathrm{FM}}=\mathbb{E}_{t,\,a_{0},\,\epsilon}\Big[\big\|f_{\theta}(a_{t},t)-(a_{0}-\epsilon)\big\|_{2}^{2}\Big].
$$

This encourages smooth, stable trajectory generation consistent with the learned driving dynamics.

### 3.5 Three-Stage Progressive Training

To produce high-quality, stable driving videos while furnishing the planner with strong representations, we adopt a three-stage progressive training scheme.

In the first stage, we focus on learning robust motion patterns by training on longer clips at a reduced spatial resolution, $740\times 352\times 121$ (width $\times$ height $\times$ frames). This configuration prioritizes temporal span over spatial detail, enabling the model to learn smooth, continuous driving behaviors such as lane keeping, turning, and speed variations. Because memory in video diffusion scales with both spatial and temporal extents, lowering the resolution allows for processing more frames, which is crucial for modeling long-horizon scenarios.

Subsequently, we switch to higher spatial resolution but shorter clips, $1280\times 704\times 25$, to further enhance the visual quality and fine-grained details, such as lane markings, surrounding vehicles, and environmental textures. In this phase, the larger spatial dimensions with fewer frames allocate capacity to spatial fidelity, while preserving the temporal coherence established in the first stage.

Finally, building on this strong video generator that learns physically grounded driving dynamics, we condition DriveLaW-Act on latent features from DriveLaW-Video and train it for trajectory planning. This third stage couples generation and planning by using video latents as compact perception for the planner. The three-stage curriculum equips DriveLaW with high-fidelity video synthesis and reliable, stable trajectory planning.

## 4 Experiment

Table 1: Quantitative evaluation of video generation on the NuScenes validation set. Our method outperforms prior single-view state-of-the-art methods in generation quality.

| Metric | DriveGAN [^35] | DriveDreamer [^68] | DrivingGPT [^13] | DriveWorld [^54] | Vista [^20] | Epona [^81] | DriveLaW (Ours) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FID $\downarrow$ | 73.4 | 52.6 | 12.8 | 7.4 | 6.9 | 7.5 | 4.6 |
| FVD $\downarrow$ | 502.3 | 452.0 | 142.6 | 90.9 | 89.4 | 82.8 | 81.3 |

Table 2: Performance comparison on NAVSIM Navtest using closed-loop metrics. Methods are grouped by whether they employ an explicit world model: Traditional End-to-End Methods and World Model Methods. <sup>†</sup> denotes methods trained with the same flow-matching objective.

<table><tbody><tr><td>Method</td><td>Ref</td><td>Image</td><td>Lidar</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Comf.<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>Constant Velocity</td><td>-</td><td></td><td></td><td>68.0</td><td>57.8</td><td>50.0</td><td>100</td><td>19.4</td><td>20.6</td></tr><tr><td>Ego Status MLP</td><td>arXiv’23</td><td></td><td></td><td>93.0</td><td>77.3</td><td>83.6</td><td>100</td><td>62.8</td><td>65.6</td></tr><tr><td colspan="8">Traditional End-to-End Methods</td><td></td><td></td></tr><tr><td>VADv2- <math><semantics><msub><mi>𝒱</mi> <mtext>8192</mtext></msub> <annotation>\mathcal{V}_{\text{8192}}</annotation></semantics></math> <sup><a href="#fn:12">12</a></sup></td><td>arXiv’24</td><td>✓</td><td></td><td>97.2</td><td>89.1</td><td>91.6</td><td>100</td><td>76.0</td><td>80.9</td></tr><tr><td>UniAD <sup><a href="#fn:31">31</a></sup></td><td>CVPR’23</td><td>✓</td><td></td><td>97.8</td><td>91.9</td><td>92.9</td><td>100</td><td>78.8</td><td>83.4</td></tr><tr><td>TransFuser <sup><a href="#fn:15">15</a></sup></td><td>TPAMI’23</td><td>✓</td><td>✓</td><td>97.7</td><td>92.8</td><td>92.8</td><td>100</td><td>79.2</td><td>84.0</td></tr><tr><td>PARA-Drive <sup><a href="#fn:71">71</a></sup></td><td>CVPR’24</td><td>✓</td><td></td><td>97.9</td><td>92.4</td><td>93.0</td><td>99.8</td><td>79.3</td><td>84.0</td></tr><tr><td>ReCogDrive-IL <sup><a href="#fn:44">44</a></sup></td><td>arXiv’25</td><td>✓</td><td></td><td>98.1</td><td>94.7</td><td>94.2</td><td>100</td><td>80.9</td><td>86.5</td></tr><tr><td>DiffusionDrive <sup><a href="#fn:48">48</a></sup></td><td>CVPR’25</td><td>✓</td><td>✓</td><td>98.2</td><td>96.2</td><td>94.7</td><td>100</td><td>82.2</td><td>88.1</td></tr><tr><td colspan="8">World Model Methods</td><td></td><td></td></tr><tr><td>DrivingGPT <sup><a href="#fn:13">13</a></sup></td><td>arXiv’24</td><td>✓</td><td></td><td>98.9</td><td>90.7</td><td>94.9</td><td>95.6</td><td>79.7</td><td>82.4</td></tr><tr><td>LAW <sup><a href="#fn:41">41</a></sup></td><td>ICLR’25</td><td>✓</td><td></td><td>96.4</td><td>95.4</td><td>88.7</td><td>99.9</td><td>81.7</td><td>84.6</td></tr><tr><td>Epona <sup><a href="#fn:81">81</a></sup></td><td>ICCV’25</td><td>✓</td><td></td><td>97.9</td><td>95.1</td><td>93.8</td><td>99.9</td><td>80.4</td><td>86.2</td></tr><tr><td>Resim <sup><a href="#fn:77">77</a></sup></td><td>NeurIPS’25</td><td>✓</td><td></td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>86.6</td></tr><tr><td>WoTE <sup><a href="#fn:43">43</a></sup></td><td>ICCV’25</td><td>✓</td><td>✓</td><td>98.5</td><td>96.8</td><td>94.9</td><td>99.9</td><td>81.9</td><td>88.3</td></tr><tr><td>DriveVLA-W0 <sup>†</sup> <sup><a href="#fn:42">42</a></sup></td><td>arXiv’25</td><td>✓</td><td></td><td>98.4</td><td>95.3</td><td>95.2</td><td>100</td><td>80.9</td><td>87.2</td></tr><tr><td>PWM <sup><a href="#fn:83">83</a></sup></td><td>NeurIPS’25</td><td>✓</td><td></td><td>98.6</td><td>95.9</td><td>95.4</td><td>100</td><td>81.8</td><td>88.1</td></tr><tr><td>DriveLaW(Ours)</td><td>-</td><td>✓</td><td></td><td>99.0</td><td>97.1</td><td>96.7</td><td>100</td><td>81.3</td><td>89.1</td></tr></tbody></table>

Table 3: Planning performance on NuScenes. We report L2 displacement error and collision rate at 1s, 2s, 3s, and averaged.

<table><tbody><tr><td>Method</td><td colspan="4">L2 (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td colspan="4">Collision (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td></td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><td>Epona <sup><a href="#fn:81">81</a></sup></td><td>0.61</td><td>1.17</td><td>1.98</td><td>1.25</td><td>0.01</td><td>0.22</td><td>0.85</td><td>0.36</td></tr><tr><td>DriveLaW</td><td>0.44</td><td>1.10</td><td>1.91</td><td>1.15</td><td>0.15</td><td>0.10</td><td>0.48</td><td>0.24</td></tr></tbody></table>

### 4.1 Experimental Setup

#### Implementation Details.

Our DriveLaW model consists of a 2B video DiT, initialized from LTX-Video [^24] pretrained weights, and a 133M diffusion planner for trajectory planning. We enable DriveLaW to acquire both video generation and planning capabilities through a three-stage training pipeline consisting of video pretraining followed by action fine-tuning. In the video pretraining stage, we train the Video DiT [^56] on $8\,\mathrm{Hz}$ frames from nuScenes [^9] and NuPlan [^10] with a two-stage curriculum. We first establish temporal coherence by training on low-resolution, long-duration clips ($740\times 352\times 121$ frames). Then we fine-tune the model on high-resolution, shorter clips ($1280\times 704\times 25$ frames). First two stages are trained for 30k iterations with a batch size of 4, a learning rate of $1\times 10^{-5}$, and a weight decay of $5\times 10^{-2}$. We use flow matching [^50] with token-wise uniform $\sigma\in[0,1]$, so each latent token interpolates independently between data and noise. In the trajectory fine-tuning stage, we feed the past four camera frames and supervise $2\,\mathrm{Hz}$ trajectory points over the next 4s, updating both the Video DiT and the Planning DiT. We use a batch size of 192 for 44k steps with a fixed learning rate of $3\times 10^{-5}$ and a weight decay of $1\times 10^{-5}$. At inference, we use 30 sampling steps for video generation and 5 steps for trajectory planning.

#### Dataset and Metrics.

Following the previous autonomous driving experimental protocol [^81] [^42] [^83], we train our model on nuPlan and nuScenes, and subsequently evaluate it on nuScenes for video generation and on NAVSIM [^16] for trajectory planning. The nuScenes dataset comprises 1,000 driving scenes collected in Boston and Singapore, featuring multi-sensor data from cameras and LiDAR, with 850 scenes used for training and validation and the remaining 150 reserved for testing. NuPlan, the first large-scale planning dataset for autonomous driving, provides 1,200 hours of human driving data from four cities. NAVSIM [^16] further builds on OpenScene [^57], a redistribution of nuPlan, offering a planning-oriented benchmark for trajectory prediction. It is divided into two subsets: Navtrain, containing 103k samples, and Navtest, containing 12k samples. We utilize 8 Hz driving camera videos from nuScenes and NuPlan for video training, and 2 Hz camera data from NAVSIM for trajectory prediction.

For video generation quality evaluation on the nuScenes validation set, we use FVD [^62] and FID [^27] for assessment. For planning evaluation on NAVSIM [^16], we use the Predictive Driver Model Score (PDMS), which combines penalties for no-at-fault collisions (NC) and drivable-area compliance (DAC) with weighted measures of ego progress (EP), time-to-collision (TTC), and comfort (Comf.) to evaluate overall safety, compliance, and efficiency.

### 4.2 Main Results

#### Quantitative experiments on video generation.

Tab. 1 reports the quantitative results of the generative evaluation on nuScenes. In both metrics, DriveLaW surpassing all previous single-view approaches with state-of-the-art performance of 4.6 FID and 81.3 FVD. Results demonstrate the effectiveness of our method for high-fidelity driving video generation.

#### Quantitative experiments on motion planning.

Tab. 2 reports NAVSIM results. DriveLaW attains a PDMS of 89.1, setting a new state of the art without any post-training such as reinforcement learning or post-processing such as learned scorers. It surpasses traditional end-to-end planners including DiffusionDrive [^48], which fuses camera and LiDAR, and ReCogDrive [^44], which relies on Vision-Language Models. Compared with world-model methods, DriveLaW improves over Epona [^81] by 2.9 PDMS, where Epona adopts a parallel generation–planning design, and over DriveVLA-W0 [^42] and PWM [^83] by 1.9 and 1.0 PDMS respectively, which use VLMs and world-model supervision, demonstrating the effectiveness of chaining generation with planning.

#### Additional performance on motion planning.

As an additional open-loop evaluation, we report planning performance on the NuScenes validation set in Tab. 3. DriveLaW achieves lower L2 errors and collision rates compared to Epona [^81], demonstrating robust generalization and safety on real-world data.

#### Qualitative Results.

Fig. 3 presents qualitative comparisons between our DriveLaW and the current state-of-the-art open-source driving world model Epona [^81]. As shown in the leftmost pair of images, vehicles in videos generated by Epona exhibit noticeable visual artifacts, whereas DriveLaW produces results with clearer vehicle details and more stable structural integrity. In Epona’s outputs, pedestrian figures nearly degrade to the point of being unrecognizable, while DriveLaW preserves complete shapes that remain easily identifiable. Additionally, in the case of the visually inconspicuous yellow van in the scene, Epona misclassifies it as a different object, whereas DriveLaW correctly recognizes and maintains its appearance and spatial position. These results demonstrate that DriveLaW excels in visual quality, subject preservation, and semantic world understanding.

![[contrast2_compressed.png|Refer to caption]]

Figure 3: Qualitative Comparison with state-of-the-art driving world model. We compare DriveLaW with Epona 81 on nuScenes validation set. DriveLaW generates videos with (1) clearer vehicle details and more stable structural integrity, (2) well-preserved pedestrian shapes that remain easily identifiable, and (3) correct recognition and maintenance of inconspicuous objects (e.g., the yellow van), demonstrating superior visual quality, subject preservation, and semantic understanding.

### 4.3 Ablation Study

#### Planning Gains from Scaling Video Generator Pretraining.

As shown in Tab. 4, increasing pretraining samples for the video generator consistently boosts DriveLaW’s closed-loop performance on NAVSIM. A fully pretrained generator delivers a +3.2 PDMS improvement over a generator without driving-domain pretraining, indicating that larger corpora deepen the model’s grasp of driving physics and translate into stronger planning, exhibiting a clear scaling law.

Table 4: Scaling video pretraining improves planning on NAVSIM Navtest. Rows vary the number of video pretraining samples used before fine-tuning the diffusion planner on NAVSIM.

| Video P.T. Size | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| 0 (scratch) | 98.2 | 93.8 | 94.1 | 99.9 | 80.8 | 85.9 |
| 76k | 98.7 | 94.7 | 95.3 | 99.9 | 80.8 | 87.0 |
| 3.8M | 98.6 | 95.8 | 94.8 | 100 | 82.2 | 87.8 |
| 7.6M | 99.0 | 97.1 | 96.7 | 100 | 81.3 | 89.1 |

#### Comparison of Driving Representations on NAVSIM Navtest.

Tab. 5 reports results with different driving representations. Under the same diffusion based planner, video generator latent features improve PDMS by 5.0 over BEV features and by 2.6 over VLM hidden states, demonstrating the effectiveness of this representation.

Table 5: Representation ablation on NAVSIM Navtest. We compare BEV features, VLM hidden states, and video latents as diffusion condition.

| Representation | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| BEV Features | 97.6 | 93.0 | 92.9 | 100 | 79.1 | 84.1 |
| VLM Hidden State | 98.1 | 94.7 | 94.2 | 100 | 80.9 | 86.5 |
| Video Latents | 99.0 | 97.1 | 96.7 | 100 | 81.3 | 89.1 |

#### Ablation on the Video Denoising Step for Action DiT Conditioning.

As shown in Tab. 6, we ablate which video denoising step provides the latent condition to the Action DiT. Conditioning on latents from early denoising steps yields stronger planning, while latents from later steps perform worse. This occurs because raw pixel-format videos frequently contain redundant, non-essential information, which can hinder the effectiveness of decision-making.

Table 6: Which video denoising step feeds the Action DiT. We evaluate planning when conditioning on video latents taken from different diffusion denoising steps.

| Video Denoise Step | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| $t=1$ | 99.0 | 97.1 | 96.7 | 100 | 81.3 | 89.1 |
| $t=5$ | 99.2 | 93.7 | 95.6 | 100 | 81.8 | 86.9 |
| $t=10$ | 81.7 | 63.4 | 67.6 | 0 | 15.4 | 23.2 |

#### Effect of Training Strategy.

Tab. 7 presents ablation results of different training strategies. Removing the first stage results in comparable FID but a much higher FVD, indicating a notable loss of temporal coherence due to the absence of long-horizon motion modeling. Omitting the second stage preserves temporal stability but slightly degrades spatial detail, as reflected in a moderate FVD increase. The complete multi-stage training strategy achieves the best balance, yielding both the lowest FID and FVD, confirming that each stage plays a complementary role in ensuring high-quality and temporally consistent driving video generation.

Table 7: Comparison of different training strategies with FID and FVD scores.

| Methods | FID $\downarrow$ | FVD $\downarrow$ |
| --- | --- | --- |
| w/o First Stage Training | 5.0 | 109.3 |
| w/o Second Stage Training | 5.0 | 93.2 |
| Ours | 4.6 | 81.3 |

## 5 Conclusion

In this work, we propose DriveLaW, a unified latent world model that addresses the long-standing disconnect between video generation and motion planning in autonomous driving. We first introduced DriveLaW-Video, a spatiotemporal generation module enhanced with a Noise Reinjection mechanism to ensure high-fidelity, temporally consistent video synthesis. Building on this foundation, we designed DriveLaW-Act, a diffusion-based planner that leverages video latents to generate smooth and reliable trajectories. To further harmonize optimization between generation and planning, we adopted a three-stage progressive training strategy. Extensive experiments on nuScenes and NAVSIM benchmarks demonstrate that DriveLaW achieves state-of-the-art performance, validating the effectiveness of unifying driving world generation and decision-making through a shared latent representation for next-generation end-to-end autonomous driving systems.

## References

## Appendix A More Implementation Details

In this section, we provide additional implementation specifics for the core components of DriveLaW, including the video generation backbone (DriveLaW-Video), the trajectory planning module (DriveLaW-Act), and the motion-conditioned prompting mechanism.

### A.1 DriveLaW-Video: Video Generation Backbone

DriveLaW-Video adopts a diffusion-based architecture optimized for high-compression spatiotemporal encoding and efficient chained generation with the downstream planner. This design balances computational efficiency and generation quality, enabling long-horizon driving scenario synthesis under practical hardware constraints.

#### Spatiotempora VAE and Compression Optimization.

The Video-VAE serves as the core spatiotemporal compression module, applying a $32\times 32\times 8$ spatial–temporal downsampling with 128 output channels. This configuration achieves a total compression ratio of 1:192 (pixels-to-tokens ratio of 1:8192), approximately twice the compression rate of common text-to-video pipelines. To enable such aggressive compression without compromising generation fidelity, we introduce the following architectural and training modifications:

- Causal 3D Encoder: Ensures each temporal step depends only on current and past frames, preserving the autoregressive consistency critical for driving prediction tasks.
- Hybrid Decoding Strategy: Instead of completing all denoising steps in the latent space, the final rectified-flow step ($t_{1}$) is executed by the VAE decoder directly in the pixel space. This design recovers high-frequency details (e.g., road texture, reflections, traffic signs) without requiring a separate super-resolution stage.
- Reconstruction GAN: The discriminator receives paired real–reconstructed samples and focuses on fine-grained detail differences. This improves training stability and perceptual quality under high compression.
- Multi-layer Noise Injection: Introduces per-channel learned stochasticity in the decoder, enhancing the diversity of synthesized textures.
- Uniform Log-variance Across Channels: Ensures balanced KL regularization and avoids underutilized latent dimensions, improving the efficiency of the latent space.
- Video-DWT Loss: Complements MSE and perceptual losses by explicitly penalizing high-frequency errors across eight 3D wavelet sub-bands, strengthening the preservation of structural details.

#### Video Transformer Backbone.

The diffusion backbone adopts a 3D Transformer architecture adapted from PixArt- $\alpha$, with $28$ self–cross attention blocks, a hidden size of $2048$, a feed-forward expansion factor of $\times 4$, and RMSNorm normalization in place of LayerNorm for better stability.

To maintain spatial–temporal consistency across varying resolutions and durations, we employ normalized fractional Rotary Positional Embeddings (RoPE) computed with exponential frequency spacing. Unlike patchifier-based designs (e.g., $2\times 2\times 1$ patch size), tokens are serialized directly from the VAE latents at a $1\times 1\times 1$ granularity, eliminating redundant patchification operations and preserving geometric consistency.

### A.2 DriveLaW-Act: Trajectory Planning Module

DriveLaW-Act is implemented as a lightweight diffusion planner (133M parameters) that is tightly integrated with the DriveLaW-Video backbone. Its key design details are as follows:

#### Input Conditioning.

The planner is directly conditioned on cached Video-DiT latents from the first denoising step. These latents encode rich scene information, including current geometry and agent dynamics, and serve as keys in the planner’s cross-attention mechanism, paired with the trajectory noise input. Additionally, the planner receives structured context embeddings, including: ego-vehicle kinematics and navigation commands.

#### Training and Inference.

The planner is trained with a flow-matching objective to generate smooth, physically consistent trajectories. It predicts continuous $(x,y,\theta)$ positions at a sampling rate of 2 Hz over a 4 s planning horizon. During inference, the planner operates purely in the latent space without requiring video decoding, significantly reducing computational overhead. Notably, gradient isolation between the video generation and planning modules is preserved during training, ensuring stable optimization of each component.

### A.3 Motion-Conditioned Prompting Mechanism

To align video generation with realistic driving dynamics, we design a structured motion-conditioned prompting mechanism that unifies dynamic ego-state information and static scene context into interpretable text guidance for the Video-DiT.

#### Prompt Construction Logic.

Ego-state numerical variables (speed, steering angle, displacement) are first discretized into semantic bins (e.g., "low speed", "steady motion", "turning left/right"). These semantic labels are integrated into a fixed prompt template, which also includes technical numerical grounding to ensure precise control. The template is defined as follows:

> A high-quality, photorealistic dashboard camera view of autonomous driving. Based on the past $T_{h}$ seconds, predict and generate the next $T_{p}$ seconds of realistic driving continuation, moving at \[speed bin\] with \[motion descriptor\], smoothly continue for the next $T_{p}$ seconds. Maintain temporal consistency, stable camera perspective, natural motion flow without jitter or artifacts, clear details, and realistic physics. \[Technical: forward $\Delta x$ m, lateral $\Delta y$ m, yaw $\Delta\theta^{\circ}$, speed $v$ m/s\]

The text prompt is encoded by the frozen T5-XXL encoder, and the resulting embeddings are injected via cross-attention into all layers of the Video-DiT. This allows semantic motion cues to modulate the generation process, ensuring alignment between the synthesized video and the ego-vehicle’s dynamic constraints.

## Appendix B Additional Experimental Results

### B.1 Qualitative analysis of latent representations.

To demonstrate that VGM (Video Generation Model) latent features can serve as more efficient and informative conditions for action learning, we conduct a systematic analysis. As shown in Fig. 4, we visualize and compare three types of latent representations. We apply PCA (Principal Component Analysis) [^1] to project each representation to 3 principal components mapped to RGB channels, all upsampled to 1280×704 (Note that for BEV features, limited by the single-view visual input, we extract intermediate backbone features before the BEV query transformation to ensure fair comparison). The visualization clearly shows that BEV and VLM features are diffuse, unstable, and exhibit irregular focus patterns. In contrast, VGM features are sharper, less noisy, and demonstrate superior semantic coherence with strong spatial structure awareness, even under challenging driving conditions. This suggests that VGM features provide a more suitable representation for action learning in autonomous driving.

![[feature_compressed.png|Refer to caption]]

Figure 4: Qualitative analysis of latent representations. We visualize the quality of latent representations from three different feature sources: perspective-view features extracted from BEVFormer 45 ’s ResNet-101 backbone, VLM features from the pretrained Qwen2.5-VL model in ReCogDrive 44, and VGM (Video Generation Model) features from our DriveLaW-Video. To enable visual comparison, we apply PCA to reduce each representation to its top 3 principal components and map them to RGB channels. From top to bottom, each row displays: (1) the original input frame, (2) BEV features, (3) VLM features, and (4) VGM features, all upsampled to 1280×704 for visualization. While the BEV and VLM features appear diffuse, unstable, and exhibit irregular focus shifts, our VGM features are notably sharper, contain significantly less noise, and demonstrate superior semantic coherence with strong spatial structure awareness, even under severe driving motion.

### B.2 Quantitative Evaluation of Video Generation

Tab. 8 presents extensive quantitative evaluation of video generation quality across multiple datasets and horizon lengths.

Table 8: Quantitative evaluation of video generation. We report FID and FVD on NuScenes and OpenDV, and FVD at varying horizon lengths on NuPlan.

<table><tbody><tr><td>Methods</td><td colspan="2">NuScenes</td><td colspan="2">OpenDV</td><td colspan="4">NuPlan</td></tr><tr><td></td><td>FID <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>FVD <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>FID <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>FVD <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>FVD <sub>24</sub> <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>FVD <sub>40</sub> <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>FVD <sub>80</sub> <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>FVD <sub>100</sub> <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>Epona</td><td>7.5</td><td>82.8</td><td>6.9</td><td>80.7</td><td>61.3</td><td>74.9</td><td>239.6</td><td>277.3</td></tr><tr><td>DriveLaW</td><td>4.6</td><td>81.3</td><td>4.6</td><td>72.9</td><td>55.6</td><td>71.2</td><td>230.2</td><td>296.1</td></tr></tbody></table>

Cross-Dataset Generalization. In Tab. 8, we evaluate zero-shot generalization on the OpenDV dataset following GEM (CVPR 2025). DriveLaW-Video outperforms Epona on OpenDV, indicating robust generalization beyond the training domains of NuScenes/NuPlan.

Long-Horizon Generation. Tab. 8 reports FVD at varying prediction horizons (24, 40, 80, and 100 frames) on NuPlan. DriveLaW consistently outperforms Epona up to 80 frames and Epona shows better performance at 100 frames. Moreover, Epona is substantially slower (Tab. 10), and this gap increases with horizon length. Considering both quality and efficiency, DriveLaW provides a more practical trade-off for realistic driving horizons.

### B.3 Inference Speed Analysis

To evaluate the efficiency of our video generation stage, we compare the per-frame speed of DriveLaW with the unified world-model baseline Epona under identical experimental settings: single NVIDIA 4090 GPU, 30 DiT sampling steps, and matching resolutions as listed in Tab. 9.

Table 9: Video generation speed per frame on a single NVIDIA 4090 GPU with 30 DiT sampling steps.

<table><tbody><tr><td>Method</td><td>Resolution</td><td>Params</td><td>Times</td></tr><tr><td>Epona</td><td><math><semantics><mrow><mn>1024</mn> <mo>×</mo> <mn>512</mn></mrow> <annotation>1024\times 512</annotation></semantics></math></td><td><math><semantics><mrow><mo>∼</mo> <mn>1.9</mn></mrow> <annotation>\sim 1.9</annotation></semantics></math> B</td><td>0.88 s</td></tr><tr><td rowspan="3">DriveLaW (Ours)</td><td><math><semantics><mrow><mn>768</mn> <mo>×</mo> <mn>512</mn></mrow> <annotation>768\times 512</annotation></semantics></math></td><td></td><td>0.12 s</td></tr><tr><td><math><semantics><mrow><mn>1024</mn> <mo>×</mo> <mn>512</mn></mrow> <annotation>1024\times 512</annotation></semantics></math></td><td><math><semantics><mrow><mo>∼</mo> <mn>2.0</mn></mrow> <annotation>\sim 2.0</annotation></semantics></math> B</td><td>0.18 s</td></tr><tr><td><math><semantics><mrow><mn>1280</mn> <mo>×</mo> <mn>704</mn></mrow> <annotation>1280\times 704</annotation></semantics></math></td><td></td><td>0.39 s</td></tr></tbody></table>

As shown in Tab. 9, DriveLaW achieves substantially faster generation at lower resolutions. For $768\times 512$, DriveLaW requires only 0.12 s per frame, while at $1024\times 512$ the speed remains modest at 0.18 s, despite the model size being slightly larger than Epona’s. At the highest resolution ($1280\times 704$), DriveLaW achieves 0.39 s per frame, which is more than twice as fast as Epona’s result 0.88 s, even though our output resolution is significantly higher.

These results indicate that the proposed architectural optimizations, which include a higher compression ratio and hybrid decoding, preserve runtime efficiency across resolutions. This allows DriveLaW to deliver competitive generation speed while maintaining high video fidelity.

### B.4 Runtime Performance on H20 GPU

For completeness, we also report inference speed on an NVIDIA H20 GPU in Table 10.

Table 10: Runtime performance on an NVIDIA H20 GPU. We report trajectory planning time, and per-frame video generation time.

| Method | Resolution | Params | Traj. (s) | Frame (s) |
| --- | --- | --- | --- | --- |
| Epona | $1024\times 512$ | $\sim 1.9$ B | 0.42 | 1.06 |
| DriveLaW (Ours) | $1024\times 512$ | $\sim 2.0$ B | 0.71 | 0.21 |

### B.5 Ablation on Noise Reinjection Usage

We conduct an ablation study to evaluate the effect of enabling or disabling the proposed noise reinjection mechanism on video generation quality. The experiments are performed on the nuScenes validation set, with FID and FVD as evaluation metrics.

Table 11: Effect of enabling noise reinjection on driving video generation quality.

| Setting | FID $\downarrow$ | FVD $\downarrow$ |
| --- | --- | --- |
| w/o Noise Reinjection | 6.1 | 102.1 |
| w/ Noise Reinjection (Ours) | 4.6 | 81.3 |

As shown in Tab. 11, removing noise reinjection results in a noticeable degradation in temporal coherence and a slight decline in spatial fidelity. By selectively perturbing high-frequency regions before each denoising step, our method compels the generator to actively regenerate fine details, thereby improving both sharpness and temporal stability while reducing artifacts.

## Appendix C More Qualitative Results

In this section, we present additional qualitative examples to further illustrate the capabilities of DriveLaW in diverse driving scenarios and planning tasks.

![[supp1_compressed.png|Refer to caption]]

Figure 5: Qualitative examples of DriveLaW video generation on the nuScenes dataset. (a) Conventional urban driving scenarios, showing stable lane keeping and interactions with surrounding traffic. (b) Complex urban driving scenarios involving dense multi-agent interactions, turning maneuvers, and occlusions. (c) Night driving scenarios, demonstrating the model’s robustness to low-light conditions while preserving temporal consistency and fine details.

### C.1 Video Generation on nuScenes

We evaluate DriveLaW on the nuScenes validation set across a wide variety of real-world driving scenarios. As shown in Fig. 5, our results demonstrate that the model maintains temporal coherence, fine-grained spatial detail, and robust performance across diverse visual conditions.

### C.2 Planning Results Visualization

As shown in Fig. 6, we present representative cases from the Navtest splits, highlighting DriveLaW’s ability to predict future trajectories while ensuring safety and smoothness.

### C.3 Supplementary Video Demonstrations

To facilitate clearer understanding, we provide 6 MP4 demo videos in the supplementary material, including 4 normal-driving scenarios and 2 rainy-weather scenarios. These examples help reviewers and readers visually assess temporal consistency, spatial detail, and the practical utility of planning outputs in diverse and challenging driving conditions.

![[supp2.png|Refer to caption]]

Figure 6: Qualitative results on the Navtest benchmark.

## Appendix D Limitations and Future Work

While DriveLaW demonstrates strong performance in video generation and trajectory planning, we acknowledge several limitations that present opportunities for future research.

### D.1 Motion Artifacts in High-Compression VAE.

To achieve efficient inference, DriveLaW employs a high-compression Video-VAE with a $32\times 32\times 8$ downsampling factor. Our experiments reveal that such aggressive compression introduces noticeable artifacts during reconstruction, particularly in high-motion scenarios. These artifacts propagate to the video generation stage, manifesting as visual distortions during rapid ego-motion or dynamic agent interactions. Although our proposed noise reinjection mechanism mitigates this issue to some extent (Tab. 11), it does not fundamentally resolve the underlying limitation. We plan to address this through architectural improvements and advanced training strategies in future work.

### D.2 Inference Latency.

Despite our optimizations (*e.g*., high-compression VAE, resolution scaling, and hybrid decoding), DriveLaW’s inference speed remains slower than end-to-end planning models that bypass explicit video generation. This gap stems from the inherent computational demands of diffusion-based video generation.

### D.3 Scalability and Future Outlook.

Notwithstanding these limitations, DriveLaW’s primary advantage lies in its *scalability* with rapid advances in video generation technology. As foundational video models continue to improve in quality, speed, and efficiency, DriveLaW’s performance will advance commensurately without requiring architectural redesign. Furthermore, the paradigm enables the research community to leverage powerful pretrained video generators to rapidly develop generalizable planners with minimal domain-specific training. With anticipated improvements in inference acceleration techniques (*e.g*., distillation, quantization, and dedicated hardware), we envision the DriveLaW paradigm becoming viable for onboard deployment in the near future.

[^1]: Hervé Abdi and Lynne J Williams. Principal component analysis. *Wiley interdisciplinary reviews: computational statistics*, 2(4):433–459, 2010.

[^2]: Arslan Ali, Junjie Bai, Maciej Bala, Yogesh Balaji, Aaron Blakeman, Tiffany Cai, Jiaxin Cao, Tianshi Cao, Elizabeth Cha, Yu-Wei Chao, et al. World simulation with video foundation models for physical ai. *arXiv preprint arXiv:2511.00062*, 2025.

[^3]: Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, et al. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. *arXiv preprint arXiv:2506.09985*, 2025.

[^4]: Florent Bartoccioni, Elias Ramzi, Victor Besnier, Shashanka Venkataramanan, Tuan-Hung Vu, Yihong Xu, Loick Chambon, Spyros Gidaris, Serkan Odabas, David Hurych, et al. Vavim and vavam: Autonomous driving through video generative modeling. *arXiv preprint arXiv:2502.15672*, 2025.

[^5]: Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. $\pi_{0}$: A vision-language-action flow model for general robot control. *arXiv preprint arXiv:2410.24164*, 2024.

[^6]: Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. *arXiv preprint arXiv:2311.15127*, 2023.

[^7]: Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. *OpenAI Blog*, 1(8):1, 2024.

[^8]: Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In *Forty-first International Conference on Machine Learning*, 2024.

[^9]: Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 11621–11631, 2020.

[^10]: Holger Caesar, Juraj Kabzan, Kok Seang Tan, Whye Kit Fong, Eric Wolff, Alex Lang, Luke Fletcher, Oscar Beijbom, and Sammy Omari. nuplan: A closed-loop ml-based planning benchmark for autonomous vehicles. *arXiv preprint arXiv:2106.11810*, 2021.

[^11]: Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart- $alpha$: Fast training of diffusion transformer for photorealistic text-to-image synthesis. *arXiv preprint arXiv:2310.00426*, 2023.

[^12]: Shaoyu Chen, Bo Jiang, Hao Gao, Bencheng Liao, Qing Xu, Qian Zhang, Chang Huang, Wenyu Liu, and Xinggang Wang. Vadv2: End-to-end vectorized autonomous driving via probabilistic planning. *arXiv preprint arXiv:2402.13243*, 2024.

[^13]: Yuntao Chen, Yuqi Wang, and Zhaoxiang Zhang. Drivinggpt: Unifying driving world modeling and planning with multi-modal autoregressive transformers. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 26890–26900, 2025.

[^14]: Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. *The International Journal of Robotics Research*, 44(10-11):1684–1704, 2025.

[^15]: Kashyap Chitta, Aditya Prakash, Bernhard Jaeger, Zehao Yu, Katrin Renz, and Andreas Geiger. Transfuser: Imitation with transformer-based sensor fusion for autonomous driving. *IEEE transactions on pattern analysis and machine intelligence*, 45(11):12878–12895, 2022.

[^16]: Daniel Dauner, Marcel Hallgarten, Tianyu Li, Xinshuo Weng, Zhiyu Huang, Zetong Yang, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, et al. Navsim: Data-driven non-reactive autonomous vehicle simulation and benchmarking. *Advances in Neural Information Processing Systems*, 37:28706–28719, 2024.

[^17]: Hao Gao, Shaoyu Chen, Bo Jiang, Bencheng Liao, Yiang Shi, Xiaoyang Guo, Yuechuan Pu, Haoran Yin, Xiangyu Li, Xinbang Zhang, et al. Rad: Training an end-to-end driving policy via large-scale 3dgs-based reinforcement learning. *arXiv preprint arXiv:2502.13144*, 2025a.

[^18]: Ruiyuan Gao, Kai Chen, Enze Xie, Lanqing Hong, Zhenguo Li, Dit-Yan Yeung, and Qiang Xu. Magicdrive: Street view generation with diverse 3d geometry control. *arXiv preprint arXiv:2310.02601*, 2023.

[^19]: Ruiyuan Gao, Kai Chen, Bo Xiao, Lanqing Hong, Zhenguo Li, and Qiang Xu. Magicdrive-v2: High-resolution long video generation for autonomous driving with adaptive control. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 28135–28144, 2025b.

[^20]: Shenyuan Gao, Jiazhi Yang, Li Chen, Kashyap Chitta, Yihang Qiu, Andreas Geiger, Jun Zhang, and Hongyang Li. Vista: A generalizable driving world model with high fidelity and versatile controllability. *Advances in Neural Information Processing Systems*, 37:91560–91596, 2024.

[^21]: Jiazhe Guo, Yikang Ding, Xiwu Chen, Shuo Chen, Bohan Li, Yingshuang Zou, Xiaoyang Lyu, Feiyang Tan, Xiaojuan Qi, Zhiheng Li, et al. Dist-4d: Disentangled spatiotemporal diffusion with metric depth for 4d driving scene generation. *arXiv preprint arXiv:2503.15208*, 2025a.

[^22]: Xiangyu Guo, Zhanqian Wu, Kaixin Xiong, Ziyang Xu, Lijun Zhou, Gangwei Xu, Shaoqing Xu, Haiyang Sun, Bing Wang, Guang Chen, et al. Genesis: Multimodal driving scene generation with spatio-temporal and cross-modal consistency. *arXiv preprint arXiv:2506.07497*, 2025b.

[^23]: Ziyu Guo, Xinyan Chen, Renrui Zhang, Ruichuan An, Yu Qi, Dongzhi Jiang, Xiangtai Li, Manyuan Zhang, Hongsheng Li, and Pheng-Ann Heng. Are video models ready as zero-shot reasoners? an empirical study with the mme-cof benchmark. *arXiv preprint arXiv:2510.26802*, 2025c.

[^24]: Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent diffusion. *arXiv preprint arXiv:2501.00103*, 2024.

[^25]: William Harvey, Saeid Naderiparizi, Vaden Masrani, Christian Weilbach, and Frank Wood. Flexible diffusion modeling of long videos. *Advances in neural information processing systems*, 35:27953–27965, 2022.

[^26]: Mariam Hassan, Sebastian Stapf, Ahmad Rahimi, Pedro Rezende, Yasaman Haghighi, David Brüggemann, Isinsu Katircioglu, Lin Zhang, Xiaoran Chen, Suman Saha, et al. Gem: A generalizable ego-vision multimodal world model for fine-grained ego-motion, object dynamics, and scene composition control. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pages 22404–22415, 2025.

[^27]: Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. *Advances in neural information processing systems*, 30, 2017.

[^28]: Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. *arXiv preprint arXiv:2210.02303*, 2022.

[^29]: Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. Gaia-1: A generative world model for autonomous driving. *arXiv preprint arXiv:2309.17080*, 2023a.

[^30]: Xiaotao Hu, Wei Yin, Mingkai Jia, Junyuan Deng, Xiaoyang Guo, Qian Zhang, Xiaoxiao Long, and Ping Tan. Drivingworld: Constructing world model for autonomous driving via video gpt. *arXiv preprint arXiv:2412.19505*, 2024.

[^31]: Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, et al. Planning-oriented autonomous driving. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 17853–17862, 2023b.

[^32]: Junjie Huang, Guan Huang, Zheng Zhu, Yun Ye, and Dalong Du. Bevdet: High-performance multi-camera 3d object detection in bird-eye-view. *arXiv preprint arXiv:2112.11790*, 2021.

[^33]: Geunmin Hwang, Hyun kyu Ko, Younghyun Kim, Seungryong Lee, and Eunbyung Park. Diffuseslide: Training-free high frame rate video generation diffusion, 2025.

[^34]: Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, and Xinggang Wang. Vad: Vectorized scene representation for efficient autonomous driving. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 8340–8350, 2023.

[^35]: Seung Wook Kim, Jonah Philion, Antonio Torralba, and Sanja Fidler. Drivegan: Towards a controllable high-quality neural simulation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 5820–5829, 2021.

[^36]: Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. *arXiv preprint arXiv:2412.03603*, 2024.

[^37]: Jiarui Lei, Xiaobo Hu, Yue Wang, and Dong Liu. Pyramidflow: High-resolution defect contrastive localization using pyramid normalizing flow. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 14143–14152, 2023.

[^38]: Bohan Li, Jiazhe Guo, Hongsi Liu, Yingshuang Zou, Yikang Ding, Xiwu Chen, Hu Zhu, Feiyang Tan, Chi Zhang, Tiancai Wang, et al. Uniscene: Unified occupancy-centric driving scene generation. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pages 11971–11981, 2025a.

[^39]: Bohan Li, Zhuang Ma, Dalong Du, Baorui Peng, Zhujin Liang, Zhenqiang Liu, Chao Ma, Yueming Jin, Hao Zhao, Wenjun Zeng, et al. Omninwm: Omniscient driving navigation world models. *arXiv preprint arXiv:2510.18313*, 2025b.

[^40]: Xiaofan Li, Yifu Zhang, and Xiaoqing Ye. Drivingdiffusion: layout-guided multi-view driving scenarios video generation with latent diffusion model. In *European Conference on Computer Vision*, pages 469–485. Springer, 2024a.

[^41]: Yingyan Li, Lue Fan, Jiawei He, Yuqi Wang, Yuntao Chen, Zhaoxiang Zhang, and Tieniu Tan. Enhancing end-to-end autonomous driving with latent world model. *arXiv preprint arXiv:2406.08481*, 2024b.

[^42]: Yingyan Li, Shuyao Shang, Weisong Liu, Bing Zhan, Haochen Wang, Yuqi Wang, Yuntao Chen, Xiaoman Wang, Yasong An, Chufeng Tang, et al. Drivevla-w0: World models amplify data scaling law in autonomous driving. *arXiv preprint arXiv:2510.12796*, 2025c.

[^43]: Yingyan Li, Yuqi Wang, Yang Liu, Jiawei He, Lue Fan, and Zhaoxiang Zhang. End-to-end driving with online trajectory evaluation via bev world model. *arXiv preprint arXiv:2504.01941*, 2025d.

[^44]: Yongkang Li, Kaixin Xiong, Xiangyu Guo, Fang Li, Sixu Yan, Gangwei Xu, Lijun Zhou, Long Chen, Haiyang Sun, Bing Wang, et al. Recogdrive: A reinforced cognitive framework for end-to-end autonomous driving. *arXiv preprint arXiv:2506.08052*, 2025e.

[^45]: Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Qiao Yu, and Jifeng Dai. Bevformer: learning bird’s-eye-view representation from lidar-camera via spatiotemporal transformers. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2024c.

[^46]: Weixin Liang, Lili Yu, Liang Luo, Srinivasan Iyer, Ning Dong, Chunting Zhou, Gargi Ghosh, Mike Lewis, Wen-tau Yih, Luke Zettlemoyer, et al. Mixture-of-transformers: A sparse and scalable architecture for multi-modal foundation models. *arXiv preprint arXiv:2411.04996*, 2024.

[^47]: Bencheng Liao, Shaoyu Chen, Xinggang Wang, Tianheng Cheng, Qian Zhang, Wenyu Liu, and Chang Huang. Maptr: Structured modeling and learning for online vectorized hd map construction. *arXiv preprint arXiv:2208.14437*, 2022.

[^48]: Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang, Qian Zhang, et al. Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pages 12037–12047, 2025a.

[^49]: Yue Liao, Pengfei Zhou, Siyuan Huang, Donglin Yang, Shengcong Chen, Yuxin Jiang, Yue Hu, Jingbin Cai, Si Liu, Jianlan Luo, et al. Genie envisioner: A unified world foundation platform for robotic manipulation. *arXiv preprint arXiv:2508.05635*, 2025b.

[^50]: Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. *arXiv preprint arXiv:2210.02747*, 2022.

[^51]: William Lotter, Gabriel Kreiman, and David Cox. Deep predictive coding networks for video prediction and unsupervised learning. *arXiv preprint arXiv:1605.08104*, 2016.

[^52]: Jiachen Lu, Ze Huang, Zeyu Yang, Jiahui Zhang, and Li Zhang. Wovogen: World volume-aware diffusion for controllable multi-camera driving scene generation. In *European Conference on Computer Vision*, pages 329–345. Springer, 2024.

[^53]: Enhui Ma, Lijun Zhou, Tao Tang, Zhan Zhang, Dong Han, Junpeng Jiang, Kun Zhan, Peng Jia, Xianpeng Lang, Haiyang Sun, et al. Unleashing generalization of end-to-end autonomous driving with controllable long video generation. *arXiv preprint arXiv:2406.01349*, 2024.

[^54]: Chen Min, Dawei Zhao, Liang Xiao, Jian Zhao, Xinli Xu, Zheng Zhu, Lei Jin, Jianshu Li, Yulan Guo, Junliang Xing, et al. Driveworld: 4d pre-trained scene understanding via world models for autonomous driving. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 15522–15533, 2024.

[^55]: Chaojun Ni, Guosheng Zhao, Xiaofeng Wang, Zheng Zhu, Wenkang Qin, Xinze Chen, Guanghong Jia, Guan Huang, and Wenjun Mei. Recondreamer-rl: Enhancing reinforcement learning via diffusion-based scene reconstruction. *arXiv preprint arXiv:2508.08170*, 2025.

[^56]: William Peebles and Saining Xie. Scalable diffusion models with transformers. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 4195–4205, 2023.

[^57]: Songyou Peng, Kyle Genova, Chiyu Jiang, Andrea Tagliasacchi, Marc Pollefeys, Thomas Funkhouser, et al. Openscene: 3d scene understanding with open vocabularies. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 815–824, 2023.

[^58]: Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. *arXiv preprint arXiv:2410.13720*, 2024.

[^59]: Xuanchi Ren, Yifan Lu, Tianshi Cao, Ruiyuan Gao, Shengyu Huang, Amirmojtaba Sabour, Tianchang Shen, Tobias Pfaff, Jay Zhangjie Wu, Runjian Chen, et al. Cosmos-drive-dreams: Scalable synthetic driving data generation with world foundation models. *arXiv preprint arXiv:2506.09042*, 2025.

[^60]: Lloyd Russell, Anthony Hu, Lorenzo Bertoni, George Fedoseev, Jamie Shotton, Elahe Arani, and Gianluca Corrado. Gaia-2: A controllable multi-view generative world model for autonomous driving. *arXiv preprint arXiv:2503.20523*, 2025.

[^61]: Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. *Neurocomputing*, 568:127063, 2024.

[^62]: Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. *arXiv preprint arXiv:1812.01717*, 2018.

[^63]: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. *Advances in neural information processing systems*, 30, 2017.

[^64]: Ruben Villegas, Jimei Yang, Seunghoon Hong, Xunyu Lin, and Honglak Lee. Decomposing motion and content for natural video sequence prediction. *arXiv preprint arXiv:1706.08033*, 2017.

[^65]: Haiguang Wang, Daqi Liu, Hongwei Xie, Haisong Liu, Enhui Ma, Kaicheng Yu, Limin Wang, and Bing Wang. Mila: Multi-view intensive-fidelity long-term video generation world model for autonomous driving. *arXiv preprint arXiv:2503.15875*, 2025a.

[^66]: Jiawei Wang, Haowei Sun, Xintao Yan, Shuo Feng, Jun Gao, and Henry X Liu. Terasim-world: Worldwide safety-critical data synthesis for end-to-end autonomous driving. *arXiv preprint arXiv:2509.13164*, 2025b.

[^67]: Lening Wang, Wenzhao Zheng, Yilong Ren, Han Jiang, Zhiyong Cui, Haiyang Yu, and Jiwen Lu. Occsora: 4d occupancy generation models as world simulators for autonomous driving. *arXiv preprint arXiv:2405.20337*, 2024a.

[^68]: Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jiagang Zhu, and Jiwen Lu. Drivedreamer: Towards real-world-drive world models for autonomous driving. In *European conference on computer vision*, pages 55–72. Springer, 2024b.

[^69]: Yuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen, and Zhaoxiang Zhang. Driving into the future: Multiview visual forecasting and planning with world model for autonomous driving. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 14749–14759, 2024c.

[^70]: Yuqing Wen, Yucheng Zhao, Yingfei Liu, Fan Jia, Yanhui Wang, Chong Luo, Chi Zhang, Tiancai Wang, Xiaoyan Sun, and Xiangyu Zhang. Panacea: Panoramic and controllable video generation for autonomous driving. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 6902–6912, 2024.

[^71]: Xinshuo Weng, Boris Ivanovic, Yan Wang, Yue Wang, and Marco Pavone. Para-drive: Parallelized architecture for real-time autonomous driving. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 15449–15458, 2024.

[^72]: Thaddäus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners. *arXiv preprint arXiv:2509.20328*, 2025.

[^73]: Zebin Xing, Xingyu Zhang, Yang Hu, Bo Jiang, Tong He, Qian Zhang, Xiaoxiao Long, and Wei Yin. Goalflow: Goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pages 1602–1611, 2025.

[^74]: Yunzhi Yan, Haotong Lin, Chenxu Zhou, Weijie Wang, Haiyang Sun, Kun Zhan, Xianpeng Lang, Xiaowei Zhou, and Sida Peng. Street gaussians: Modeling dynamic urban scenes with gaussian splatting. In *European Conference on Computer Vision*, pages 156–173. Springer, 2024.

[^75]: Chenyu Yang, Yuntao Chen, Hao Tian, Chenxin Tao, Xizhou Zhu, Zhaoxiang Zhang, Gao Huang, Hongyang Li, Yu Qiao, Lewei Lu, et al. Bevformer v2: Adapting modern image backbones to bird’s-eye-view recognition via perspective supervision. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 17830–17839, 2023.

[^76]: Jiazhi Yang, Shenyuan Gao, Yihang Qiu, Li Chen, Tianyu Li, Bo Dai, Kashyap Chitta, Penghao Wu, Jia Zeng, Ping Luo, et al. Generalized predictive model for autonomous driving. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 14662–14672, 2024a.

[^77]: Jiazhi Yang, Kashyap Chitta, Shenyuan Gao, Long Chen, Yuqian Shao, Xiaosong Jia, Hongyang Li, Andreas Geiger, Xiangyu Yue, and Li Chen. Resim: Reliable world simulation for autonomous driving. *arXiv preprint arXiv:2506.09981*, 2025.

[^78]: Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. *arXiv preprint arXiv:2408.06072*, 2024b.

[^79]: Shuang Zeng, Xinyuan Chang, Mengwei Xie, Xinran Liu, Yifan Bai, Zheng Pan, Mu Xu, Xing Wei, and Ning Guo. Futuresightdrive: Thinking visually with spatio-temporal cot for autonomous driving. *arXiv preprint arXiv:2505.17685*, 2025.

[^80]: Biao Zhang and Rico Sennrich. Root mean square layer normalization. *Advances in neural information processing systems*, 32, 2019.

[^81]: Kaiwen Zhang, Zhenyu Tang, Xiaotao Hu, Xingang Pan, Xiaoyang Guo, Yuan Liu, Jingwei Huang, Li Yuan, Qian Zhang, Xiao-Xiao Long, et al. Epona: Autoregressive diffusion world model for autonomous driving. *arXiv preprint arXiv:2506.24113*, 2025.

[^82]: Guosheng Zhao, Xiaofeng Wang, Zheng Zhu, Xinze Chen, Guan Huang, Xiaoyi Bao, and Xingang Wang. Drivedreamer-2: Llm-enhanced world models for diverse driving video generation. In *Proceedings of the AAAI Conference on Artificial Intelligence*, pages 10412–10420, 2025a.

[^83]: Zhida Zhao, Talas Fu, Yifan Wang, Lijun Wang, and Huchuan Lu. From forecasting to planning: Policy world model for collaborative state-action prediction. *arXiv preprint arXiv:2510.19654*, 2025b.

[^84]: Wenzhao Zheng, Weiliang Chen, Yuanhui Huang, Borui Zhang, Yueqi Duan, and Jiwen Lu. Occworld: Learning a 3d occupancy world model for autonomous driving. In *European conference on computer vision*, pages 55–72. Springer, 2024.

[^85]: Yinan Zheng, Ruiming Liang, Kexin Zheng, Jinliang Zheng, Liyuan Mao, Jianxiong Li, Weihao Gu, Rui Ai, Shengbo Eben Li, Xianyuan Zhan, et al. Diffusion-based planning for autonomous driving with flexible guidance. *arXiv preprint arXiv:2501.15564*, 2025.

[^86]: Hongyu Zhou, Longzhong Lin, Jiabao Wang, Yichong Lu, Dongfeng Bai, Bingbing Liu, Yue Wang, Andreas Geiger, and Yiyi Liao. Hugsim: A real-time, photo-realistic and closed-loop simulator for autonomous driving. *arXiv preprint arXiv:2412.01718*, 2024.