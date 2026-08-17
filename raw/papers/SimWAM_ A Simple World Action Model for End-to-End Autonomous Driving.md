---
title: "SimWAM: A Simple World Action Model for End-to-End Autonomous Driving"
source: "https://arxiv.org/html/2608.07468v2"
author:
published:
created: 2026-08-17
description:
tags:
  - "clippings"
---
Zongchuang Zhao <sup>1</sup>, Xin Zhou <sup>1</sup>, Tianyang Xu <sup>1</sup>, Zhengyang Sun <sup>1</sup>  
Kaixuan Zhou <sup>2</sup>, Honglin Li <sup>2</sup>, Dingkang Liang <sup>1†</sup>, Xiang Bai <sup>1</sup>  
  
<sup>1</sup> Huazhong University of Science & Technology, <sup>2</sup> Dongfeng Research & Development Institute.  
{zcuangzhao, xzhou03, dkliang, xbai}@hust.edu.cn

###### Abstract

World-Action Models (WAMs) improve end-to-end autonomous driving by transferring video dynamics priors to action prediction, but existing methods incur costly test-time future imagination. We present SimWAM, a simple yet effective WAM that leverages future-video prediction as a training-time supervision signal. It co-trains a pretrained video expert and a lightweight action expert with joint flow matching. An isolated attention mask keeps action prediction independent of future frames, allowing trajectory prediction without explicit future-frame generation at inference. Since the two experts share no parameters and interact only through a unified attention interface, the video backbone could be replaced and the action expert scaled independently without modifying the learning objective or inference pipeline. We further apply reinforcement learning to optimize a compositional driving reward beyond trajectory imitation. Our SimWAM achieves $91.5$ PDMS on NAVSIM, surpasses state-of-the-art WAM-based planners with substantially lower latency, and transfers zero-shot to nuScenes. These results position SimWAM as a simple yet solid baseline that could readily benefit from advances in video generation for efficient autonomous driving. The code and model weights are available at [https://github.com/H-EmbodVis/SimWAM/](https://github.com/H-EmbodVis/SimWAM/).

<sup>†</sup>

## 1 Introduction

End-to-end autonomous driving [^55] [^9] maps raw sensor observations directly to a planned trajectory with a unified network. Joint optimization removes hand-crafted interfaces and reduces error propagation in the classical perception, prediction, and planning pipeline [^39] [^5]. Although recent end-to-end planners [^18] [^19] [^28] have steadily improved planning accuracy, they remain primarily imitation policies. They reproduce behavior from logged trajectories while capturing traffic semantics, user intent, and scene dynamics only implicitly.

Vision-Language-Action (VLA) models [^20] [^4] [^27] [^22] [^12] address the semantic limitation by adapting pretrained vision-language models to driving. Their semantic knowledge and high-level reasoning improve scene understanding and connect trajectory generation with user intent. Many driving VLAs [^60] [^67] [^49] further produce an explicit rationale before predicting a trajectory, which improves interpretability in complex and instruction-conditioned scenarios. Recent methods [^56] [^44] [^38] introduce future-scene generation or latent reasoning to strengthen spatiotemporal understanding. However, these components remain loosely coupled with action prediction and often require additional training stages or sequential inference. Motion and temporal evolution therefore remain modeled only indirectly, which motivates a more explicit treatment of world dynamics.

World models meet this demand by furnishing an explicit prior over how the environment evolves under motion. Building on this principle, recent World-Action Models (WAMs) in embodied intelligence,

![[x1 40.png|Refer to caption]]

Figure 1: SimWAM achieves the best PDMS with substantially lower latency than world-model-based planners on NAVSIM.

such as DreamZero [^53] and LingBot-VA [^24], jointly predict future observations and actions through pretrained video-generation backbones. This world-action paradigm has recently been adopted in autonomous driving. DriveLaW [^50] and DriveWAM [^43] jointly train a video predictor and a planner, allowing anticipated scene dynamics to inform trajectory generation. Nevertheless, existing driving WAMs commonly follow an *imagine-then-act* pipeline in which the planner conditions its output on generated future frames. This design places costly video synthesis inside the real-time planning loop and substantially increases inference latency, see Fig. 1.

Crucially, explicit future synthesis is unnecessary for effective world-action learning. Fast-WAM [^54] shows that video co-training benefits action prediction primarily through *training-time* representation learning rather than *test-time* future imagination. Building on this insight, we introduce SimWAM, a plain yet effective World-Action Model that uses video generation as a training signal while retaining direct trajectory prediction at inference. SimWAM jointly trains a pretrained video expert and a lightweight action expert with flow matching. A simple isolated attention mask prevents the action expert from accessing future frames, which allows inference to bypass explicit future-frame prediction. The resulting action dit leverages the learned traffic-dynamics prior without auxiliary motion modules or explicit future-frame generation at deployment. This decoupling also makes the video expert replaceable, allowing more advanced video generators to improve the learned prior without changing the action expert or inference pipeline. Furthermore, we reformulate the deterministic flow ODE as a stochastic SDE and reinforce the action expert with GRPO [^14] [^30], enabling diverse maneuver exploration and direct optimization of a compositional driving reward. Rather than claiming algorithmic superiority, this work establishes a simple and solid WAM baseline for exploring the potential of generic video models in autonomous driving.

The advantages of SimWAM arise from three aspects: 1) SimWAM effectively transfers traffic dynamics priors from a pretrained video generator to the planner without auxiliary motion modules. 2) Thanks to the isolated attention mask, the action expert remains independent of future-frame representations, allowing efficient inference without explicitly generating future frames. 3) The decoupled architecture seamlessly accommodates more advanced video generators without modifying the action expert or inference pipeline.

Experiments on the NAVSIM benchmark [^10] validate the effectiveness of this simple design. SimWAM achieves $91.5$ PDMS with substantially lower inference latency than state-of-the-art planners based on world models, as shown in Fig. 1. Furthermore, our method supports different pretrained video generators and transfers zero-shot to nuScenes [^6] without fine-tuning, demonstrating architectural scalability and cross-domain generalization. We hope SimWAM will serve as a strong and practical baseline for efficient world-action modeling in autonomous driving.

## 2 Related Work

### 2.1 Vision-Language-Action Models for Autonomous Driving

End-to-end autonomous driving integrates perception, prediction, and planning within a unified framework. Methods such as UniAD [^18] and VAD [^19] reduce hand-crafted interfaces and mitigate error propagation in modular pipelines. Despite this integration, these methods are largely trained on driving observations with expert trajectory supervision, which provides limited support for explicit semantic reasoning about route intent and complex traffic interactions. Vision-Language-Action (VLA) models [^27] [^12] [^65] [^35] introduce pretrained vision-language representations to enhance driving policies with semantic knowledge and reasoning capabilities. AutoVLA [^67] unifies chain-of-thought reasoning and action generation within an autoregressive framework. ORION [^12] aggregates long-term visual context through a query-based temporal module and employs a large language model for scenario understanding and driving reasoning. Its generative planner further maps the resulting planning representation into multimodal trajectories. FutureSightDrive [^56] and ExploreVLA [^42] incorporate future image generation to model scene evolution and support trajectory planning. In contrast, our SimWAM directly transfers the motion prior of a pretrained video generator into a lightweight action expert for direct trajectory prediction.

### 2.2 World-Action Models for Autonomous Driving

World-Action Models [^3] [^47] [^1] have recently attracted growing interest in robotics by jointly learning action prediction and image generation to capture object motion, physical interactions, task progress, and future scene evolution. DreamZero [^53] adapts pretrained video generation models for generalizable robotic control. LingBot-VA [^24] unifies visual prediction and policy execution for closed-loop robotic control. In autonomous driving, earlier world models mainly focused on predicting and generating future driving scenes. DriveDreamer [^48] learns structured traffic constraints and future driving states for controllable video generation. HERMES [^65] extends this direction by unifying 3D scene understanding and future scene generation through a shared bird’s-eye-view representation. More recent studies [^25] [^43] have integrated visual world modeling with trajectory planning. Epona [^59] jointly predicts future videos and trajectories through autoregressive diffusion, while DriveLaW [^50] conditions a diffusion planner on latent representations produced by its video generator. These methods follow an imagine-then-act paradigm in which trajectory planning remains coupled with future visual generation during inference. In contrast, SimWAM uses future-video prediction to learn a motion prior during training and predicts trajectories directly without conditioning on generated future driving frames.

### 2.3 Reinforcement Learning for Autonomous Driving

Imitation learning trains autonomous driving policies to reproduce expert trajectories, but this objective confines learning to demonstrated behavior and only indirectly reflects overall driving quality. Reinforcement learning provides a complementary refinement stage that directly optimizes driving policies with task-level rewards. CarPlanner [^58] uses expert guided rewards to improve large scale trajectory planning. Raw2Drive [^52] refines driving policies with raw sensor inputs and privileged world models. Recent studies [^67] [^27] [^13] [^42] have further introduced reinforcement learning into Vision-Language-Action driving models. MindDrive [^13] improves online exploration by optimizing high level decisions and continuous action generation with separate LoRA parameterizations. CritiqueDriveVLM [^34] applies verifier guided reinforcement learning to improve driving reasoning and distills the learned capability into an efficient policy. These methods mainly reinforce language-mediated reasoning or high-level decisions in VLA planners. Our SimWAM instead reinforces the action expert for direct continuous trajectory prediction after video-action co-training.

## 3 Preliminary

Flow matching. We model both trajectories and future frames with rectified flow [^29] [^33]. Given a clean target $x$ and Gaussian noise $\epsilon\sim\mathcal{N}(0,I)$, the linear interpolation $x_{\tau}=(1{-}\tau)\,x+\tau\,\epsilon$ ($\tau\in[0,1]$) has constant velocity $\epsilon-x$, which a network $v_{\theta}$ learns to predict under conditioning $c$:

$$
\mathcal{L}_{\text{FM}}=\mathbb{E}_{x,\epsilon,\tau}\big[\,\|v_{\theta}(x_{\tau},\tau,c)-(\epsilon-x)\|_{2}^{2}\,\big].
$$

Sampling integrates the probability-flow ODE $\mathrm{d}x_{\tau}=v_{\theta}(x_{\tau},\tau,c)\,\mathrm{d}\tau$ from noise ($\tau{=}1$) to data ($\tau{=}0$).

From ODE to SDE. The deterministic ODE generates a single trajectory and lacks a tractable transition density. These limitations restrict exploration over alternative driving trajectories and preclude policy-gradient optimization. Following Flow-GRPO [^30], we therefore transform the ODE into an SDE that preserves the same marginal distributions $p_{\tau}(x_{\tau})$, defined as:

$$
\mathrm{d}x_{\tau}=\Big[v_{\theta}(x_{\tau},\tau)+\tfrac{\sigma_{\tau}^{2}}{2\tau}\big(x_{\tau}+(1{-}\tau)\,v_{\theta}(x_{\tau},\tau)\big)\Big]\mathrm{d}\tau+\sigma_{\tau}\,\mathrm{d}w,\qquad\sigma_{\tau}=a\sqrt{\tfrac{\tau}{1{-}\tau}},
$$

where $\mathrm{d}w$ is a Wiener increment and $a$ controls the noise scale. Each Euler-Maruyama step yields an isotropic Gaussian transition $\pi_{\theta}(x_{\tau-\Delta\tau}\mid x_{\tau})=\mathcal{N}\big(\mu_{\theta}(x_{\tau},\tau),\,\sigma_{\tau}^{2}\Delta\tau\,I\big)$ with tractable log-likelihoods for importance sampling.

## 4 Method

We present SimWAM as a plain yet solid world-action model for end-to-end autonomous driving, as illustrated in Fig. 2. A pretrained video expert transfers traffic dynamics knowledge to a lightweight action expert through joint flow matching. An isolated attention mask keeps action prediction independent of future frames, allowing trajectory prediction without future-frame rollout at inference. The action branch directly predicts trajectories and is further optimized via reinforcement learning.

![[x2 38.png|Refer to caption]]

Figure 2: Overview of SimWAM. During training, the video and action DiTs are jointly optimized for future-frame generation and trajectory prediction via shared attention, while the isolated mask prevents the action tokens from accessing future-frame tokens. During inference and reinforcement learning, the model directly predicts trajectories without explicitly predicting future frames.

### 4.1 Model Architecture

Problem formulation. We consider end-to-end trajectory planning from a front-camera observation $o_{t}$, the ego state $s_{t}$ containing velocity, acceleration, and yaw rate, and a navigation command $l$. The planner predicts an ego trajectory $a_{t+1:t+H}=(a_{t+1},\ldots,a_{t+H})$ in the ego-vehicle coordinate frame, where each waypoint $a_{i}=(x_{i},y_{i},\theta_{i})$ specifies the planned position and heading. Existing driving WAMs [^50] [^43] commonly adopt an imagine-then-act factorization, expressed as:

$$
p_{\theta}(a_{t+1:t+H}\mid o_{t},s_{t},l)=\int p_{\theta}(z_{t+1:t+N}\mid o_{t},s_{t},l)\,p_{\theta}(a_{t+1:t+H}\mid o_{t},s_{t},l,z_{t+1:t+N})\,\mathrm{d}z_{t+1:t+N},
$$

which first synthesizes the future driving-scene latents $z_{t+1:t+N}$ and then conditions trajectory generation on them. This factorization places costly video generation inside the real-time planning loop. SimWAM instead retains a simple and direct policy interface, expressed as:

$$
p_{\theta}(a_{t+1:t+H}\mid o_{t},s_{t},l)=p_{\theta}\big(a_{t+1:t+H}\mid z(o_{t}),s_{t},l\big),
$$

where $z(o_{t})$ is the representation produced from the current observation. The traffic-dynamics prior is learned through future-video supervision during training. Consequently, inference avoids future-scene generation and auxiliary motion modules while retaining direct trajectory prediction.

Video expert. The video expert is a video Diffusion Transformer [^37] initialized from Wan2.2-5B [^46], together with its video VAE [^21] and T5 [^40] text encoder. The VAE maps each driving frame into latent tokens, while the navigation command enters through T5 cross-attention. The current frame serves as a clean condition, and the $N$ future frames are noised and reconstructed with flow matching. This standard video-generation objective supplies the action expert with a traffic-aware motion prior without introducing a driving-specific prediction module.

Action expert. The action expert is a lightweight Diffusion Transformer with hidden size $d_{a}{=}1024$. Conditioned on $c=\{z(o_{t}),s_{t},l\}$, it predicts the trajectory velocity field $v_{\theta_{a}}(a^{\tau}_{t+1:t+H},\tau,c)$ via flow matching, where a small MLP embeds the ego state. Integrating the ODE maps noise to a planned trajectory. At inference, we omit explicit future-frame prediction and directly generate trajectories.

Co-training. The two experts interact only through shared attention [^54] and retain their original architectures. Joint flow matching over video and trajectory modalities allows future-scene prediction to shape the observation representation used for planning. The joint objective is defined as:

$$
\mathcal{L}=\mathcal{L}_{\text{FM}}^{\text{act}}+\lambda\,\mathcal{L}_{\text{FM}}^{\text{vid}},
$$

where $\mathcal{L}_{\text{FM}}^{\text{act}}$ and $\mathcal{L}_{\text{FM}}^{\text{vid}}$ instantiate Eq. 1 on the action trajectory $a_{t+1:t+H}$ and the future-frame latents $z_{t+1:t+N}$, and $\lambda$ balances the two terms.

Reinforcement. The preceding stage trains the action expert through imitation learning. However, imitation learning relies exclusively on expert trajectories, constraining the policy to the behavior and quality of the demonstrations. We therefore introduce reinforcement learning (RL) to optimize trajectory generation directly toward driving quality. The deterministic flow ODE lacks the stochasticity required to explore diverse maneuvers and provides no tractable transition likelihoods for policy optimization. Following Flow-GRPO [^30], we replace the ODE with the marginal-preserving SDE in Eq. 2 and sample a group of $G$ candidate trajectories for each scenario. Each candidate is evaluated using the compositional NAVSIM PDM reward [^10], from which group-relative advantages are derived for the clipped policy update [^41] [^14]. During this RL stage, we focus on the hard navtrain scenarios with the lowest PDMS after imitation learning. To preserve the distilled motion prior and maintain a simple planner, we update only the LoRA adapters [^16] of the action expert.

### 4.2 Isolated Attention Mask

SimWAM aims to exploit future-video generation during training while avoiding the computational overhead of explicit future-frame generation at inference. To this end, we introduce an isolated attention mask that decouples action prediction from explicit future driving scene generation. As shown in Fig. 2, the shared attention stream contains the current observation latents $z(o_{t})$, the future frame latents $z_{t+1:t+N}$, and the action tokens. Both future frame tokens and action tokens attend to $z(o_{t})$, while remaining mutually invisible. The action expert learns from the shared observation representation without depending on future frame tokens. This mask constitutes the only structural modification required to isolate the action tokens from future-frame information.

Thanks to this separation,the future-video prediction objective serves as a training-time supervision signal that enriches the observation representation with traffic dynamics. At inference, the action expert directly predicts trajectories from the current inputs. Consequently, the future-frame decoder could be discarded after training, avoiding explicit future scene generation and substantially reducing inference latency. The same property also allows reinforcement learning to optimize trajectory prediction without relying on future-frame generation (§4.1).

The structural simplicity of SimWAM naturally yields flexibility in both architecture and model scale. The two experts share no weights and exchange information only through the attention stream. Consequently, neither expert depends on the internal parameterization of the other, allowing their architectures and capacities to be adjusted separately within the unified attention interface.

Video generator flexibility. Thanks to this simple interface, SimWAM can seamlessly accommodate different pretrained video generators. The action expert operates on the shared representation of the current observation without depending on future-frame tokens or decoded future predictions. During co-training, the future-video objective provides dynamics supervision that enriches this representation, while inference directly predicts trajectories without explicitly generating future frames. Replacing the video backbone with a newer or more driving domain-relevant model therefore requires no redesign of the action expert or trajectory objective. In this sense, SimWAM can readily benefit from advances in video generation while preserving the same world-action interface.

Scale flexibility. The same simplicity also makes model capacity straightforward to scale. The video and action experts provide two complementary capacity controls. A stronger video expert can provide richer dynamics-aware representations, while SimWAM avoids the additional inference cost of explicitly predicting future frames. Conversely, we could adjust the width and depth of the action DiT to meet a target latency without changing the video expert or training objective. SimWAM thus balance the representation capacity of the video expert and the planning capacity of the action expert, naturally supporting different performance and computation budgets through one unified design.

## 5 Experiments

### 5.1 Experimental Setup

#### Dataset and benchmark.

We evaluate SimWAM on NAVSIM [^10], a non-reactive planning benchmark built from the OpenScene subset of nuPlan [^7]. NAVSIM removes trivial stationary and constant-velocity cases while retaining challenging intersections, merges, and turns. We train on navtrain with $103{,}288$ scenes and evaluate on the held-out navtest split with $12{,}146$ scenes. Although each scene provides multi-view cameras and LiDAR, SimWAM uses only the front camera as visual input. The primary metric is the Predictive Driver Model Score (PDMS), which combines five closed-loop submetrics according to:

$$
\text{PDMS}=\prod_{m\in\{\text{NC},\text{DAC}\}}r_{m}\times\frac{\sum_{m\in\{\text{EP},\text{TTC},\text{C}\}}w_{m}\cdot r_{m}}{\sum_{m\in\{\text{EP},\text{TTC},\text{C}\}}w_{m}},
$$

where NC and DAC denote No Collision and Drivable Area Compliance. They serve as binary penalty factors. EP, TTC, and C denote Ego Progress, Time-to-Collision, and Comfort and form the weighted quality term.

Table 1: Comparison with state-of-the-art planners on the NAVSIM navtest benchmark. C denotes camera and L denotes LiDAR. The best learned result in each column is shown in bold.

<table><tbody><tr><th>Method</th><th>Reference</th><td>Sensors</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>C <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><th>Human Agent</th><th>-</th><td>-</td><td>100.0</td><td>100.0</td><td>87.5</td><td>100.0</td><td>99.9</td><td>94.8</td></tr><tr><th colspan="9">Traditional E2E planners</th></tr><tr><th>UniAD <sup><a href="#fn:18">18</a></sup></th><th>CVPR’23</th><td>6 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C</td><td>97.8</td><td>91.9</td><td>78.8</td><td>92.9</td><td>100.0</td><td>83.4</td></tr><tr><th>TransFuser <sup><a href="#fn:8">8</a></sup></th><th>TPAMI’22</th><td>3 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C+L</td><td>97.7</td><td>92.8</td><td>79.2</td><td>92.8</td><td>100.0</td><td>84.0</td></tr><tr><th>ARTEMIS <sup><a href="#fn:11">11</a></sup></th><th>RA-L’25</th><td>3 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C+L</td><td>98.3</td><td>95.1</td><td>81.4</td><td>94.3</td><td>100.0</td><td>87.0</td></tr><tr><th>WorldRFT <sup><a href="#fn:51">51</a></sup></th><th>AAAI’26</th><td>3 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C</td><td>97.8</td><td>96.8</td><td>81.7</td><td>94.0</td><td>100.0</td><td>87.8</td></tr><tr><th>DiffusionDrive <sup><a href="#fn:28">28</a></sup></th><th>CVPR’25</th><td>3 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C+L</td><td>98.2</td><td>96.2</td><td>82.2</td><td>94.7</td><td>100.0</td><td>88.1</td></tr><tr><th>WoTE <sup><a href="#fn:26">26</a></sup></th><th>ICCV’25</th><td>3 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C+L</td><td>98.5</td><td>96.8</td><td>81.9</td><td>94.9</td><td>99.9</td><td>88.3</td></tr><tr><th>SeerDrive <sup><a href="#fn:57">57</a></sup></th><th>NeurIPS’25</th><td>3 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C+L</td><td>98.4</td><td>97.0</td><td>83.2</td><td>94.9</td><td>99.9</td><td>88.9</td></tr><tr><th colspan="9">VLM-based planners</th></tr><tr><th>UniWorldVLA <sup><a href="#fn:32">32</a></sup></th><th>arXiv’26</th><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C</td><td>98.7</td><td>96.7</td><td>83.2</td><td>96.1</td><td>100.0</td><td>89.4</td></tr><tr><th>DriveDreamer-Policy <sup><a href="#fn:66">66</a></sup></th><th>arXiv’26</th><td>3 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C</td><td>98.4</td><td>97.1</td><td>83.5</td><td>95.1</td><td>100.0</td><td>89.2</td></tr><tr><th>Vega <sup><a href="#fn:68">68</a></sup></th><th>arXiv’26</th><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C</td><td>98.9</td><td>95.3</td><td>81.6</td><td>96.1</td><td>100.0</td><td>87.9</td></tr><tr><th>ImagiDrive <sup><a href="#fn:23">23</a></sup></th><th>ICRA’26</th><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C</td><td>98.6</td><td>96.2</td><td>80.5</td><td>94.5</td><td>100.0</td><td>87.4</td></tr><tr><th>AutoVLA <sup><a href="#fn:67">67</a></sup></th><th>NeurIPS’25</th><td>3 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C</td><td>98.4</td><td>95.6</td><td>81.9</td><td>98.0</td><td>99.9</td><td>89.1</td></tr><tr><th>ReCogDrive <sup><a href="#fn:27">27</a></sup></th><th>ICLR’26</th><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C</td><td>97.9</td><td>97.3</td><td>87.3</td><td>94.9</td><td>100.0</td><td>90.8</td></tr><tr><th>ExploreVLA <sup><a href="#fn:42">42</a></sup></th><th>ECCV’26</th><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C</td><td>98.8</td><td>98.4</td><td>83.5</td><td>96.5</td><td>99.9</td><td>90.4</td></tr><tr><th>DriveVLA-W0 <sup><a href="#fn:25">25</a></sup></th><th>ICLR’26</th><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C</td><td>98.7</td><td>99.1</td><td>83.3</td><td>95.3</td><td>99.3</td><td>90.2</td></tr><tr><th>SGDrive <sup><a href="#fn:22">22</a></sup></th><th>CVPR’26</th><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C</td><td>98.6</td><td>97.8</td><td>85.8</td><td>96.2</td><td>100.0</td><td>91.1</td></tr><tr><th colspan="9">World-model-based planners</th></tr><tr><th>Epona <sup><a href="#fn:59">59</a></sup></th><th>ICCV’25</th><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C</td><td>97.9</td><td>95.1</td><td>80.4</td><td>93.8</td><td>99.9</td><td>86.2</td></tr><tr><th>PWM <sup><a href="#fn:61">61</a></sup></th><th>NeurIPS’25</th><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C</td><td>98.6</td><td>95.9</td><td>81.8</td><td>95.4</td><td>100.0</td><td>88.1</td></tr><tr><th>DriveLaW <sup><a href="#fn:50">50</a></sup></th><th>CVPR’26</th><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C</td><td>99.0</td><td>97.1</td><td>81.3</td><td>96.7</td><td>100.0</td><td>89.1</td></tr><tr><th>DriveWAM <sup><a href="#fn:43">43</a></sup></th><th>arXiv’26</th><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C</td><td>98.3</td><td>98.1</td><td>84.3</td><td>95.2</td><td>100.0</td><td>90.1</td></tr><tr><th>SimWAM (ours)</th><th>-</th><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C</td><td>98.4</td><td>98.7</td><td>86.4</td><td>95.5</td><td>100.0</td><td>91.5</td></tr></tbody></table>

#### Implementation details.

The video expert is initialized from Wan2.2-5B [^46], together with its VAE and T5 encoder. The action expert is a lightweight DiT with a hidden size of $1024$. Unless otherwise specified, all experiments use a single front camera at a resolution of $384{\times}672$. The action expert predicts $8$ waypoints over $4$  s at $2$  Hz, while the video expert predicts the corresponding $8$ future frames. In the joint training stage, we adopt AdamW [^36] and a cosine learning rate schedule with an initial learning rate of $10^{-4}$. We train the model for $100$ epochs and set $\lambda{=}1$. During reinforcement learning (RL), we optimize only rank- $32$ LoRA adapters [^16] with a scale of $\alpha{=}16$ on the attention projections of the action expert. We sample $G{=}8$ trajectories per scenario and use a learning rate of $5{\times}10^{-5}$. RL focuses on challenging navtrain scenes where the imitation policy obtains a PDMS below $90$, while evaluation always covers the full navtest split.

### 5.2 Main Results

As shown in Tab. 1, we compare SimWAM with recent state-of-the-art planners on NAVSIM navtest. Even with only a single front camera, our method achieves $91.5$ PDMS and establishes a new state of the art in end-to-end planning. Our SimWAM notably surpasses the strongest VLM-based planner, SGDrive [^22], by $0.4$ points. ExploreVLA [^42] explicitly incorporates future image prediction to enhance VLA planning, yet still trails our method by $1.1$ points. Compared with recent imagine-then-act WAM planners, SimWAM effectively internalizes video dynamics priors during training and directly generates trajectories without costly future prediction at inference. Under the same single-camera setting, SimWAM consistently outperforms DriveLaW [^50] and DriveWAM [^43] by $2.4$ and $1.4$ points, respectively. Among the world-model-based planners, our method achieves the best DAC and EP while maintaining competitive NC and TTC. Together with the latency results in Fig. 1, these results show that training-time world modeling could support superior planning performance with efficient inference.

### 5.3 Analysis

Table 2: Component analysis.

| Configuration | NC | DAC | EP | TTC | PDMS |
| --- | --- | --- | --- | --- | --- |
| Action-only | 97.6 | 95.7 | 81.7 | 92.6 | 86.6 |
| \+ Video | 98.7 | 98.0 | 83.9 | 95.9 | 90.3 |
| \+ RL | 98.4 | 98.7 | 86.4 | 95.5 | 91.5 |

Table 3: Attention mask analysis.

| Mask | NC | DAC | EP | TTC | PDMS |
| --- | --- | --- | --- | --- | --- |
| Bidirectional | 98.4 | 98.0 | 84.7 | 95.1 | 90.2 |
| Action $\to$ video | 98.5 | 97.8 | 84.3 | 95.5 | 90.1 |
| Isolated | 98.7 | 98.0 | 83.9 | 95.9 | 90.3 |

Table 4: Video backbone flexibility.

| Video model | NC | DAC | EP | TTC | PDMS |
| --- | --- | --- | --- | --- | --- |
| LTX-Video [^15] | 98.1 | 97.2 | 83.1 | 94.3 | 88.7 |
| Wan2.1-1.3B [^46] | 98.6 | 98.1 | 84.0 | 95.9 | 90.2 |
| Cosmos2.5 [^2] | 98.7 | 98.0 | 84.2 | 96.0 | 90.4 |
| Wan2.2-5B | 98.7 | 98.0 | 83.9 | 95.9 | 90.3 |

Table 5: Action expert scaling.

| Action DiT | NC | DAC | EP | TTC | PDMS |
| --- | --- | --- | --- | --- | --- |
| $0.21$  B | 98.6 | 97.8 | 84.0 | 95.4 | 89.9 |
| $0.45$  B | 98.6 | 97.9 | 83.8 | 95.9 | 90.1 |
| $1.02$  B | 98.7 | 98.0 | 83.9 | 95.9 | 90.3 |

Component analysis. We analyze the contributions of different training stages, as listed in Tab. 2. The action-only DiT establishes a solid baseline with $86.6$ PDMS. Joint training with the video expert consistently improves all metrics and substantially raises PDMS to $90.3$. These broad improvements demonstrate that future-video supervision effectively transfers traffic-dynamics priors into the shared observation representation, enabling the action expert to better understand scene evolution without auxiliary modules or future generation at inference. RL further improves PDMS to $91.5$ by directly optimizing driving quality beyond trajectory imitation. Although minor trade-offs occur in individual metrics, the improvement confirms that RL better balances safety, compliance, and progress. Video co-training and RL thus contribute complementary gains, improving PDMS by $4.9$ points while preserving direct and efficient trajectory inference without explicit future-frame generation.

Attention mask. The attention pattern determines how information flows between the two experts, and we compare three alternatives in Tab. 3. Both bidirectional and action $\to$ video attention tie action prediction to future video tokens, forcing the model to instantiate future-frame representations at inference. In contrast, our isolated mask cleanly decouples the action expert from future prediction while retaining the benefits of joint learning through the current observation. Despite its simpler dependency structure, the isolated mask achieves the best PDMS of $90.3$, along with the strongest NC and TTC. These results suggest that exposing the action branch to the future-video tokens provides no measurable benefit in our setting, while the isolated design enables efficient inference without explicit future-frame prediction.

Video backbone flexibility. SimWAM accommodates diverse pretrained video generators through a unified attention interface, as summarized in Tab. 4. Wan2.1-1.3B and Wan2.2-5B achieve comparable PDMS values of $90.2$ and $90.3$, confirming that our method is not tied to a particular video backbone. Notably, the newer Cosmos-Predict2.5 [^2] has been pretrained on driving videos and therefore provides stronger driving-relevant dynamics priors, achieving the best PDMS of $90.4$ together with the strongest EP and TTC. By comparison, the lightweight LTX-Video reaches $88.7$ PDMS, suggesting that the quality of the video prior remains important. These results highlight that SimWAM can seamlessly absorb stronger and more domain-relevant priors from advanced video generation models while preserving the action expert and inference pipeline.

Action expert scalability. The parameter-independent two-expert design further allows the action expert to scale independently, as reported in Tab. 5. Increasing the action DiT from $0.21$ B to $1.02$ B steadily improves PDMS from $89.9$ to $90.3$. Since the experts interact through a unified attention interface, their capacities can be adjusted separately. A stronger video expert can enrich the dynamics-aware representation, whereas the action expert can be resized according to the desired balance between planning quality and efficiency. This decoupling provides SimWAM with two complementary scaling dimensions. We adopt the $1.02$ B action expert for the remaining experiments.

Table 6: Zero-shot generalization on the nuScenes open-loop planning benchmark. $*$ represents only using the front camera as input.

<table><tbody><tr><th rowspan="2">Method</th><td rowspan="2">Finetune</td><td rowspan="2">Input</td><td rowspan="2">Auxiliary Supervision</td><td colspan="4">L2 (m)  <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td colspan="4">Collision Rate (%)  <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>1 s</td><td>2 s</td><td>3 s</td><td>Avg.</td><td>1 s</td><td>2 s</td><td>3 s</td><td>Avg.</td></tr><tr><th>ST-P3 <sup><a href="#fn:17">17</a></sup></th><td>✓</td><td>Camera</td><td>Map&Box&Depth</td><td>1.33</td><td>2.11</td><td>2.90</td><td>2.11</td><td>0.23</td><td>0.62</td><td>1.27</td><td>0.71</td></tr><tr><th>UniAD <sup><a href="#fn:18">18</a></sup></th><td>✓</td><td>Camera</td><td>Map&Box&Motion</td><td>0.48</td><td>0.96</td><td>1.65</td><td>1.03</td><td>0.05</td><td>0.17</td><td>0.71</td><td>0.31</td></tr><tr><th>OccNet <sup><a href="#fn:45">45</a></sup></th><td>✓</td><td>Camera</td><td>3D-Occ&Map&Box</td><td>1.29</td><td>2.13</td><td>2.99</td><td>2.14</td><td>0.21</td><td>0.59</td><td>1.37</td><td>0.72</td></tr><tr><th>OccWorld <sup><a href="#fn:62">62</a></sup></th><td>✓</td><td>Camera</td><td>3D-Occ</td><td>0.52</td><td>1.27</td><td>2.41</td><td>1.40</td><td>0.12</td><td>0.40</td><td>2.08</td><td>0.87</td></tr><tr><th>VAD-Tiny <sup><a href="#fn:19">19</a></sup></th><td>✓</td><td>Camera</td><td>Map&Box&Motion</td><td>0.60</td><td>1.23</td><td>2.06</td><td>1.30</td><td>0.31</td><td>0.53</td><td>1.33</td><td>0.72</td></tr><tr><th>VAD-Base <sup><a href="#fn:19">19</a></sup></th><td>✓</td><td>Camera</td><td>Map&Box&Motion</td><td>0.54</td><td>1.15</td><td>1.98</td><td>1.22</td><td>0.04</td><td>0.39</td><td>1.17</td><td>0.53</td></tr><tr><th>GenAD <sup><a href="#fn:63">63</a></sup></th><td>✓</td><td>Camera</td><td>Map&Box&Motion</td><td>0.36</td><td>0.83</td><td>1.55</td><td>0.91</td><td>0.06</td><td>0.23</td><td>1.00</td><td>0.43</td></tr><tr><th>Doe-1 <sup><a href="#fn:64">64</a></sup></th><td>✓</td><td>Camera <sup>∗</sup></td><td>QA</td><td>0.50</td><td>1.18</td><td>2.11</td><td>1.26</td><td>0.04</td><td>0.37</td><td>1.19</td><td>0.53</td></tr><tr><th>Epona <sup><a href="#fn:59">59</a></sup></th><td>✓</td><td>Camera <sup>∗</sup></td><td>None</td><td>0.61</td><td>1.17</td><td>1.98</td><td>1.25</td><td>0.01</td><td>0.22</td><td>0.85</td><td>0.36</td></tr><tr><th>DriveVA <sup><a href="#fn:31">31</a></sup></th><td>✗</td><td>Camera <sup>∗</sup></td><td>None</td><td>0.33</td><td>0.76</td><td>1.43</td><td>0.84</td><td>0.00</td><td>0.07</td><td>0.12</td><td>0.06</td></tr><tr><th>DriveWAM <sup><a href="#fn:43">43</a></sup></th><td>✗</td><td>Camera <sup>∗</sup></td><td>None</td><td>0.28</td><td>0.81</td><td>1.80</td><td>0.96</td><td>0.00</td><td>0.05</td><td>0.14</td><td>0.06</td></tr><tr><th>SimWAM (ours)</th><td>✗</td><td>Camera <sup>∗</sup></td><td>None</td><td>0.29</td><td>0.82</td><td>1.77</td><td>0.96</td><td>0.00</td><td>0.03</td><td>0.11</td><td>0.04</td></tr></tbody></table>

Cross-dataset generalization. We directly evaluate the NAVSIM-trained SimWAM on the nuScenes [^6] open-loop benchmark without fine-tuning. As shown in Tab. 6, SimWAM achieves the lowest average collision rate of $0.04\%$ without nuScenes supervision or auxiliary annotations. Its average L2 error of $0.96$ m remains competitive with the strongest zero-shot baselines. L2 emphasizes agreement with dataset-specific expert trajectories, whereas collision rate more directly measures safe interaction with traffic. The strong safety performance under this domain shift shows that the learned dynamics prior transfers beyond the training benchmark.

### 5.4 Ablation Studies

We ablate RL and other key choices. Unless otherwise noted, configuration ablations use the imitation-trained world-action model, and all latency is measured on a single NVIDIA A100 GPU.

Table 7: Exploration sampler analysis.

| Sampler | NC | DAC | EP | TTC | PDMS |
| --- | --- | --- | --- | --- | --- |
| Random noise | 97.7 | 98.4 | 88.0 | 94.1 | 91.3 |
| SDE | 98.4 | 98.7 | 86.4 | 95.5 | 91.5 |

Table 8: Future-video target analysis.

| Target | NC | DAC | EP | TTC | PDMS |
| --- | --- | --- | --- | --- | --- |
| 4 f, 2 s, 2 Hz | 98.6 | 97.7 | 83.9 | 95.5 | 89.9 |
| 4 f, 4 s, 1 Hz | 98.7 | 97.9 | 84.2 | 95.6 | 90.2 |
| 8 f, 4 s, 2 Hz | 98.7 | 98.0 | 83.9 | 95.9 | 90.3 |

Exploration sampler. RL requires diverse trajectory candidates, whereas the original flow ODE is deterministic. We therefore compare two stochastic sampling strategies in Tab. 7. Native random perturbations encourage exploration and improve EP, but noticeably degrade NC and TTC due to less structured maneuvers. In contrast, the marginal-preserving SDE explores diverse yet plausible trajectories while providing tractable transition likelihoods for policy optimization. It consequently achieves a better overall balance and $91.5$ PDMS. We therefore adopt the SDE throughout RL.

![[x3 34.png|Refer to caption]]

Figure 3: RL training dynamics. The star denotes the imitation checkpoint. Training on the hard subset consistently outperforms training on all navtrain scenes.

RL training dynamics. We then compare RL training on the full navtrain set and a challenging subset with imitation PDMS below $90$ in Fig. 3. Training on the challenging subset consistently outperforms training on all scenes and steadily improves PDMS to a peak of $91.5$ at $15$ k steps. These difficult scenarios expose clearer differences among sampled trajectories and consequently provide more informative reward signals for policy optimization. In contrast, many scenes in the full set are already well handled by imitation learning, contributing limited learning signals and diluting the benefit of RL. Both curves decline slightly beyond $15$ k steps, indicating diminishing returns from prolonged optimization.

Prediction horizon and frame density. We further examine the temporal configuration of future-video supervision in Tab. 8. Shortening the prediction horizon from $4$ s to $2$ s noticeably reduces PDMS, whereas maintaining the $4$ s horizon with half as many frames recovers most of the performance. This comparison indicates that broad temporal coverage is more important than dense frame sampling for learning traffic dynamics. The full $4$ s target at $2$ Hz achieves the strongest performance.

Table 9: The effect of input resolution.

<table><thead><tr><th rowspan="2">Resolution</th><th colspan="5">navtest metric</th><th rowspan="2">Latency (ms)</th></tr><tr><th>NC</th><th>DAC</th><th>EP</th><th>TTC</th><th>PDMS</th></tr></thead><tbody><tr><td><math><semantics><mrow><mn>192</mn> <mo>×</mo> <mn>352</mn></mrow> <annotation>192{\times}352</annotation></semantics></math></td><td>98.2</td><td>97.1</td><td>83.0</td><td>94.9</td><td>88.9</td><td>509</td></tr><tr><td><math><semantics><mrow><mn>384</mn> <mo>×</mo> <mn>672</mn></mrow> <annotation>384{\times}672</annotation></semantics></math></td><td>98.7</td><td>98.0</td><td>83.9</td><td>95.9</td><td>90.3</td><td>518</td></tr><tr><td><math><semantics><mrow><mn>768</mn> <mo>×</mo> <mn>1344</mn></mrow> <annotation>768{\times}1344</annotation></semantics></math></td><td>98.7</td><td>98.1</td><td>84.3</td><td>96.1</td><td>90.6</td><td>573</td></tr></tbody></table>

Table 10: The effect of sampling steps.

<table><thead><tr><th rowspan="2">Steps</th><th colspan="5">navtest metric</th><th rowspan="2">Latency (ms)</th></tr><tr><th>NC</th><th>DAC</th><th>EP</th><th>TTC</th><th>PDMS</th></tr></thead><tbody><tr><td>1</td><td>97.4</td><td>91.3</td><td>79.1</td><td>83.3</td><td>68.9</td><td>115</td></tr><tr><td>5</td><td>98.6</td><td>97.9</td><td>84.0</td><td>95.6</td><td>90.1</td><td>297</td></tr><tr><td>10</td><td>98.7</td><td>98.0</td><td>83.9</td><td>95.9</td><td>90.3</td><td>518</td></tr><tr><td>20</td><td>98.6</td><td>98.0</td><td>83.9</td><td>95.8</td><td>90.2</td><td>968</td></tr></tbody></table>

Input resolution. We next study the trade-off between visual detail and inference efficiency in Tab. 9. Increasing the resolution from $192{\times}352$ to $384{\times}672$ substantially improves PDMS by $1.4$ points with only $9$ ms of additional latency. Further increasing the resolution to $768{\times}1344$ yields merely a $0.3$ point gain while adding considerably more computation. These results identify $384{\times}672$ as the most favorable balance between planning accuracy and inference efficiency.

Number of sampling steps. Finally, we investigate the convergence of the action flow sampler in Tab. 10. A single sampling step is insufficient to produce well-refined trajectories, whereas five steps already recover most of the performance. Increasing the budget to ten steps achieves the highest PDMS of $90.3$. Using twenty steps provides no further improvement while nearly doubling the latency, indicating that the sampler has already converged.

### 5.5 Qualitative Results

As shown in Fig. 4, we compare the imitation-trained and reinforced models in two scenes. The imitation-trained model produces conservative trajectories and advances only a short distance at the intersection and along the narrow street. After reinforcement, the model follows the intended route more decisively and completes a larger portion of each maneuver. Meanwhile, the trajectories remain within the drivable area and maintain safe clearance from surrounding vehicles.

![[x4 32.png|Refer to caption]]

Figure 4: Qualitative comparison of Ours-IL and Ours-RL on two navtest scenarios. Red ellipses highlight regions where progresses farther while remaining within the drivable area.

## 6 Conclusion

In this paper, we present SimWAM, a simple yet effective and flexible world-action model for end-to-end autonomous driving. Through joint flow matching, it transfers traffic-dynamics priors from a pretrained video expert to a lightweight action expert. An isolated attention mask decouples action prediction from future frames, enabling direct trajectory planning without explicit future-frame prediction at inference. This design also makes the video backbone replaceable and the two experts independently scalable, allowing stronger video priors to be incorporated without redesigning the planner while adapting the action expert to different efficiency requirements. Reinforcement learning further aligns trajectory generation with driving quality beyond imitation. Using only a single front camera, SimWAM achieves $91.5$ PDMS on NAVSIM with efficient direct trajectory inference and transfers zero-shot to nuScenes. These results show that training-time future-video generation could provide effective dynamics supervision for superior planning performance without costly test-time future imagination.

[^1]: N. Agarwal, A. Ali, M. Bala, Y. Balaji, E. Barker, T. Cai, P. Chattopadhyay, Y. Chen, Y. Cui, Y. Ding, et al. (2025) Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575. Cited by: §2.2.

[^2]: A. Ali, J. Bai, M. Bala, Y. Balaji, A. Blakeman, T. Cai, J. Cao, T. Cao, E. Cha, Y. Chao, et al. (2025) World simulation with video foundation models for physical ai. arXiv preprint arXiv:2511.00062. Cited by: §5.3, Table 4.

[^3]: M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes, M. Muckley, A. Rizvi, C. Roberts, K. Sinha, A. Zholus, et al. (2025) V-jepa 2: self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985. Cited by: §2.2.

[^4]: K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. (2025) Pi0: a vision-language-action flow model for general robot control. In Proc. of Robotics: Science and Systems, Cited by: §1.

[^5]: M. Bojarski, D. Del Testa, D. Dworakowski, B. Firner, B. Flepp, P. Goyal, L. D. Jackel, M. Monfort, U. Muller, J. Zhang, et al. (2016) End to end learning for self-driving cars. arXiv preprint arXiv:1604.07316. Cited by: §1.

[^6]: H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom (2020) Nuscenes: a multimodal dataset for autonomous driving. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, Cited by: §1, §5.3.

[^7]: H. Caesar, J. Kabzan, K. S. Tan, W. K. Fong, E. Wolff, A. Lang, L. Fletcher, O. Beijbom, and S. Omari (2021) Nuplan: a closed-loop ml-based planning benchmark for autonomous vehicles. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, Cited by: §5.1.

[^8]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger (2022) Transfuser: imitation with transformer-based sensor fusion for autonomous driving. IEEE Transactions on Pattern Analysis and Machine Intelligence. Cited by: Table 1.

[^9]: F. Codevilla, M. Müller, A. López, V. Koltun, and A. Dosovitskiy (2018) End-to-end driving via conditional imitation learning. In Proc. of the IEEE Int. Conf. on Robotics and Automation, Cited by: §1.

[^10]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, et al. (2024) Navsim: data-driven non-reactive autonomous vehicle simulation and benchmarking. In Proc. of Advances in Neural Information Processing Systems, Cited by: §1, §4.1, §5.1.

[^11]: R. Feng, N. Xi, D. Chu, R. Wang, Z. Deng, A. Wang, L. Lu, J. Wang, and Y. Huang (2025) Artemis: autoregressive end-to-end trajectory planning with mixture of experts for autonomous driving. IEEE Robotics and Automation Letters. Cited by: Table 1.

[^12]: H. Fu, D. Zhang, Z. Zhao, J. Cui, D. Liang, C. Zhang, D. Zhang, H. Xie, B. Wang, and X. Bai (2025) Orion: a holistic end-to-end autonomous driving framework by vision-language instructed action generation. In Proc. of IEEE Intl. Conf. on Computer Vision, Cited by: §1, §2.1.

[^13]: H. Fu, D. Zhang, Z. Zhao, J. Cui, H. Xie, B. Wang, G. Chen, D. Liang, and X. Bai (2026) Minddrive: a vision-language-action model for autonomous driving via online reinforcement learning. In Proc. of European Conference on Computer Vision, Cited by: §2.3.

[^14]: D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhu, R. Xu, R. Zhang, S. Ma, X. Bi, et al. (2025) Deepseek-r1: incentivizing reasoning capability in llms via reinforcement learning. Nature. Cited by: §1, §4.1.

[^15]: Y. HaCohen, N. Chiprut, B. Brazowski, D. Shalem, D. Moshe, E. Richardson, E. Levin, G. Shiran, N. Zabari, O. Gordon, et al. (2024) Ltx-video: realtime video latent diffusion. arXiv preprint arXiv:2501.00103. Cited by: Table 4.

[^16]: E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, et al. (2022) Lora: low-rank adaptation of large language models.. In Proc. of Intl. Conf. on Learning Representations, Cited by: §4.1, §5.1.

[^17]: S. Hu, L. Chen, P. Wu, H. Li, J. Yan, and D. Tao (2022) St-p3: end-to-end vision-based autonomous driving via spatial-temporal feature learning. In Proc. of European Conference on Computer Vision, Cited by: Table 6.

[^18]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, et al. (2023) Planning-oriented autonomous driving. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, Cited by: §1, §2.1, Table 1, Table 6.

[^19]: B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang (2023) Vad: vectorized scene representation for efficient autonomous driving. In Proc. of IEEE Intl. Conf. on Computer Vision, Cited by: §1, §2.1, Table 6, Table 6.

[^20]: M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al. (2024) Openvla: an open-source vision-language-action model. In Proc. of Conference on Robot Learning, Cited by: §1.

[^21]: D. P. Kingma and M. Welling (2014) Auto-encoding variational bayes. In Proc. of Intl. Conf. on Learning Representations, Cited by: §4.1.

[^22]: J. Li, J. Wu, D. Hu, X. Huang, B. Sun, Z. Hao, X. Lang, X. Zhu, and L. Zhang (2026) Sgdrive: scene-to-goal hierarchical world cognition for autonomous driving. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, Cited by: §1, §5.2, Table 1.

[^23]: J. Li, B. Zhang, X. Jin, J. Deng, X. Zhu, and L. Zhang (2026) Imagidrive: a unified imagination-and-planning framework for autonomous driving. In Proc. of the IEEE Int. Conf. on Robotics and Automation, Cited by: Table 1.

[^24]: L. Li, Q. Zhang, Y. Luo, S. Yang, R. Wang, F. Han, M. Yu, Z. Gao, N. Xue, X. Zhu, et al. (2026) Causal world modeling for robot control. In Proc. of Robotics: Science and Systems, Cited by: §1, §2.2.

[^25]: Y. Li, S. Shang, W. Liu, B. Zhan, H. Wang, Y. Wang, Y. Chen, X. Wang, Y. An, C. Tang, et al. (2026) Drivevla-w0: world models amplify data scaling law in autonomous driving. In Proc. of Intl. Conf. on Learning Representations, Cited by: §2.2, Table 1.

[^26]: Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang (2025) End-to-end driving with online trajectory evaluation via bev world model. In Proc. of IEEE Intl. Conf. on Computer Vision, Cited by: Table 1.

[^27]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, et al. (2026) ReCogDrive: a reinforced cognitive framework for end-to-end autonomous driving. In Proc. of Intl. Conf. on Learning Representations, Cited by: §1, §2.1, §2.3, Table 1.

[^28]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al. (2025) Diffusiondrive: truncated diffusion model for end-to-end autonomous driving. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, Cited by: §1, Table 1.

[^29]: Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le (2023) Flow matching for generative modeling. In Proc. of Intl. Conf. on Learning Representations, Cited by: §3.

[^30]: J. Liu, G. Liu, J. Liang, Y. Li, J. Liu, X. Wang, P. Wan, D. Zhang, and W. Ouyang (2025) Flow-grpo: training flow matching models via online rl. In Proc. of Advances in Neural Information Processing Systems, Cited by: §1, §3, §4.1.

[^31]: M. Liu, D. Zhang, J. Liu, J. Cui, H. Xie, G. Chen, H. Ye, M. Y. Yang, F. Nex, and H. Cheng (2026) Driveva: video action models are zero-shot drivers. In Proc. of European Conference on Computer Vision, Cited by: Table 6.

[^32]: Q. Liu, H. Xu, J. Li, B. Sun, Z. Hao, D. She, X. Zhu, and L. Zhang (2026) Uni-world vla: interleaved world modeling and planning for autonomous driving. arXiv preprint arXiv:2603.27287. Cited by: Table 1.

[^33]: X. Liu, C. Gong, and Q. Liu (2023) Flow straight and fast: learning to generate and transfer data with rectified flow. In Proc. of Intl. Conf. on Learning Representations, Cited by: §3.

[^34]: Z. Liu, H. Ye, X. Zhang, and M. Qi (2026) CritiqueDriveVLM: from verifier-guided reinforcement learning to latent thought distillation for autonomous driving. arXiv preprint arXiv:2607.04179. Cited by: §2.3.

[^35]: Z. Liu, R. Huang, R. Yang, S. Yan, Z. Wang, L. Hou, D. Lin, X. Bai, and H. Zhao (2026) Drivepi: spatial-aware 4d mllm for unified autonomous driving understanding, perception, prediction and planning. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, Cited by: §2.1.

[^36]: I. Loshchilov and F. Hutter (2019) Decoupled weight decay regularization. In Proc. of Intl. Conf. on Learning Representations, Cited by: §5.1.

[^37]: W. Peebles and S. Xie (2023) Scalable diffusion models with transformers. In Proc. of IEEE Intl. Conf. on Computer Vision, Cited by: §4.1.

[^38]: Q. Peng, X. Chen, C. Yang, S. Shi, and H. Li (2026) Colavla: leveraging cognitive latent reasoning for hierarchical parallel trajectory planning in autonomous driving. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, Cited by: §1.

[^39]: D. A. Pomerleau (1988) Alvinn: an autonomous land vehicle in a neural network. In Proc. of Advances in Neural Information Processing Systems, Cited by: §1.

[^40]: C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu (2020) Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research. Cited by: §4.1.

[^41]: Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. (2024) Deepseekmath: pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300. Cited by: §4.1.

[^42]: Z. Sheng, X. Ye, J. Luo, S. Chen, and L. Ren (2026) Explorevla: dense world modeling and exploration for end-to-end autonomous driving. In Proc. of European Conference on Computer Vision, Cited by: §2.1, §2.3, §5.2, Table 1.

[^43]: C. Shi, J. Xu, S. Shi, K. Sheng, B. Zhang, and L. Jiang (2026) DriveWAM: video generative priors enable scalable world-action modeling for autonomous driving. arXiv preprint arXiv:2605.28544. Cited by: §1, §2.2, §4.1, §5.2, Table 1, Table 6.

[^44]: S. Tan, K. Chitta, Y. Chen, R. Tian, Y. You, Y. Wang, W. Luo, Y. Cao, P. Krähenbühl, M. Pavone, and B. Ivanovic (2026) Latent chain-of-thought world modeling for end-to-end autonomous driving. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, Cited by: §1.

[^45]: W. Tong, C. Sima, T. Wang, L. Chen, S. Wu, H. Deng, Y. Gu, L. Lu, P. Luo, D. Lin, et al. (2023) Scene as occupancy. In Proc. of IEEE Intl. Conf. on Computer Vision, Cited by: Table 6.

[^46]: T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C. Xie, D. Chen, F. Yu, H. Zhao, J. Yang, et al. (2025) Wan: open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314. Cited by: §4.1, §5.1, Table 4.

[^47]: H. Wang, X. Ye, F. Tao, C. Pan, A. Mallik, B. Yaman, L. Ren, and J. Zhang (2025) Adawm: adaptive world model based planning for autonomous driving. In Proc. of Intl. Conf. on Learning Representations, Cited by: §2.2.

[^48]: X. Wang, Z. Zhu, G. Huang, X. Chen, J. Zhu, and J. Lu (2024) Drivedreamer: towards real-world-drive world models for autonomous driving. In Proc. of European Conference on Computer Vision, Cited by: §2.2.

[^49]: Y. Wang, W. Luo, J. Bai, Y. Cao, T. Che, K. Chen, Y. Chen, J. Diamond, Y. Ding, W. Ding, et al. (2025) Alpamayo-r1: bridging reasoning and action prediction for generalizable autonomous driving in the long tail. arXiv preprint arXiv:2511.00088. Cited by: §1.

[^50]: T. Xia, Y. Li, L. Zhou, J. Yao, K. Xiong, H. Sun, B. Wang, K. Ma, G. Chen, H. Ye, et al. (2026) Drivelaw: unifying planning and video generation in a latent driving world. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, Cited by: §1, §2.2, §4.1, §5.2, Table 1.

[^51]: P. Yang, B. Lu, Z. Xia, C. Han, Y. Gao, T. Zhang, K. Zhan, X. Lang, Y. Zheng, and Q. Zhang (2026) Worldrft: latent world model planning with reinforcement fine-tuning for autonomous driving. In Proc. of the AAAI Conf. on Artificial Intelligence, Cited by: Table 1.

[^52]: Z. Yang, X. Jia, Q. Li, X. Yang, M. Yao, and J. Yan (2025) Raw2drive: reinforcement learning with aligned world models for end-to-end autonomous driving (in carla v2). In Proc. of Advances in Neural Information Processing Systems, Cited by: §2.3.

[^53]: S. Ye, Y. Ge, K. Zheng, S. Gao, S. Yu, G. Kurian, S. Indupuru, Y. L. Tan, C. Zhu, J. Xiang, et al. (2026) World action models are zero-shot policies. arXiv preprint arXiv:2602.15922. Cited by: §1, §2.2.

[^54]: T. Yuan, Z. Dong, Y. Liu, and H. Zhao (2026) Fast-wam: do world action models need test-time future imagination?. arXiv preprint arXiv:2603.16666. Cited by: §1, §4.1.

[^55]: E. Yurtsever, J. Lambert, A. Carballo, and K. Takeda (2020) A survey of autonomous driving: common practices and emerging technologies. IEEE access. Cited by: §1.

[^56]: S. Zeng, X. Chang, M. Xie, X. Liu, Y. Bai, Z. Pan, M. Xu, and X. Wei (2025) Futuresightdrive: thinking visually with spatio-temporal cot for autonomous driving. In Proc. of Advances in Neural Information Processing Systems, Cited by: §1, §2.1.

[^57]: B. Zhang, N. Song, X. Zhu, J. Deng, L. Zhang, et al. (2025) Future-aware end-to-end driving: bidirectional modeling of trajectory planning and scene evolution. In Proc. of Advances in Neural Information Processing Systems, Cited by: Table 1.

[^58]: D. Zhang, J. Liang, K. Guo, S. Lu, Q. Wang, R. Xiong, Z. Miao, and Y. Wang (2025) Carplanner: consistent auto-regressive trajectory planning for large-scale reinforcement learning in autonomous driving. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, Cited by: §2.3.

[^59]: K. Zhang, Z. Tang, X. Hu, X. Pan, X. Guo, Y. Liu, J. Huang, L. Yuan, Q. Zhang, X. Long, et al. (2025) Epona: autoregressive diffusion world model for autonomous driving. In Proc. of IEEE Intl. Conf. on Computer Vision, Cited by: §2.2, Table 1, Table 6.

[^60]: Q. Zhao, Y. Lu, M. J. Kim, Z. Fu, Z. Zhang, Y. Wu, Z. Li, Q. Ma, S. Han, C. Finn, et al. (2025) Cot-vla: visual chain-of-thought reasoning for vision-language-action models. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, Cited by: §1.

[^61]: Z. Zhao, T. Fu, Y. Wang, L. Wang, and H. Lu (2025) From forecasting to planning: policy world model for collaborative state-action prediction. In Proc. of Advances in Neural Information Processing Systems, Cited by: Table 1.

[^62]: W. Zheng, W. Chen, Y. Huang, B. Zhang, Y. Duan, and J. Lu (2024) Occworld: learning a 3d occupancy world model for autonomous driving. In Proc. of European Conference on Computer Vision, Cited by: Table 6.

[^63]: W. Zheng, R. Song, X. Guo, C. Zhang, and L. Chen (2024) Genad: generative end-to-end autonomous driving. In European Conference on Computer Vision, Cited by: Table 6.

[^64]: W. Zheng, Z. Xia, Y. Huang, S. Zuo, J. Zhou, and J. Lu (2024) Doe-1: closed-loop autonomous driving with large world model. arXiv preprint arXiv:2412.09627. Cited by: Table 6.

[^65]: X. Zhou, D. Liang, S. Tu, X. Chen, Y. Ding, D. Zhang, F. Tan, H. Zhao, and X. Bai (2025) Hermes: a unified self-driving world model for simultaneous 3d scene understanding and generation. In Proc. of IEEE Intl. Conf. on Computer Vision, Cited by: §2.1, §2.2.

[^66]: Y. Zhou, X. Wang, H. Shao, L. Wang, G. Zhao, J. Shao, J. Zhu, T. Yu, Z. Zhu, G. Huang, et al. (2026) Drivedreamer-policy: a geometry-grounded world-action model for unified generation and planning. arXiv preprint arXiv:2604.01765. Cited by: Table 1.

[^67]: Z. Zhou, T. Cai, S. Zhao, Y. Zhang, Z. Huang, B. Zhou, and J. Ma (2025) Autovla: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. In Proc. of Advances in Neural Information Processing Systems, Cited by: §1, §2.1, §2.3, Table 1.

[^68]: S. Zuo, Y. Li, W. Zheng, Z. Zhu, J. Zhou, and J. Lu (2026) Vega: learning to drive with natural language instructions. arXiv preprint arXiv:2603.25741. Cited by: Table 1.