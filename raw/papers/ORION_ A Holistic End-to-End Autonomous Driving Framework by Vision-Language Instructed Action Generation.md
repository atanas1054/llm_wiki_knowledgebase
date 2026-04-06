---
title: "ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation"
source: "https://arxiv.org/html/2503.19755v1"
author:
published:
created: 2026-04-05
description:
tags:
  - "clippings"
---
Haoyu Fu <sup>1∗</sup>, Diankun Zhang <sup>2∗</sup>, Zongchuang Zhao <sup>1∗</sup>, Jianfeng Cui <sup>2</sup>, Dingkang Liang <sup>1†</sup>,  
Chong Zhang <sup>2</sup>, Dingyuan Zhang <sup>1</sup>, Hongwei Xie <sup>2†</sup>, Bing Wang <sup>2</sup>, Xiang Bai <sup>1</sup>  
  
<sup>1</sup> Huazhong University of Science and Technology, <sup>2</sup> Xiaomi EV  
{hyfu, zcuangzhao, dkliang}@hust.edu.cn  
[https://xiaomi-mlab.github.io/Orion/](https://xiaomi-mlab.github.io/Orion/)

###### Abstract

End-to-end (E2E) autonomous driving methods still struggle to make correct decisions in interactive closed-loop evaluation due to limited causal reasoning capability. Current methods attempt to leverage the powerful understanding and reasoning abilities of Vision-Language Models (VLMs) to resolve this dilemma. However, the problem is still open that few VLMs for E2E methods perform well in the closed-loop evaluation due to the gap between the semantic reasoning space and the purely numerical trajectory output in the action space. To tackle this issue, we propose ORION, a hOlistic E2E autonomous dRiving framework by vIsion-language instructed actiON generation. ORION uniquely combines a QT-Former to aggregate long-term history context, a Large Language Model (LLM) for driving scenario reasoning, and a generative planner for precision trajectory prediction. ORION further aligns the reasoning space and the action space to implement a unified E2E optimization for both visual question-answering (VQA) and planning tasks. Our method achieves an impressive closed-loop performance of 77.74 Driving Score (DS) and 54.62% Success Rate (SR) on the challenge Bench2Drive datasets, which outperforms state-of-the-art (SOTA) methods by a large margin of 14.28 DS and 19.61% SR.

<sup>0</sup>![[x1 7.png|Refer to caption]]

Figure 1: The comparison of different E2E paradigms. Our ORION framework establishes the differentiable connection between reasoning and action space via the generative planner.

## 1 Introduction

End-to-end (E2E) autonomous driving has witnessed significant advancements in recent years. Classic E2E methods [^18] [^25] [^67] [^70] [^8] integrate perception [^43] [^27] [^66], prediction [^15] [^50] [^7], and planning [^17] [^44] modules through multi-task learning, as shown in Fig. 1(a). These methods optimize driving trajectories by imitating expert demonstrations, achieving promising performance in the open-loop evaluation [^6] [^53]. Nevertheless, these methods lack the common sense to complete complex causal reasoning. As a result, they struggle with comprehensive closed-loop benchmarks [^23] that require autonomous decision-making and dynamic environmental interactions. Recently, Vision-Language Models (VLMs) [^39] [^1] [^10] [^57] have accumulated rich world knowledge and aligned vision-language space through large-scale data training, providing new insight for achieving E2E autonomous driving.

Despite these advances, leveraging VLMs for E2E autonomous driving is not trivial, as the capabilities of VLMs focus on the semantic reasoning space, while E2E methods only need the numerical planning results in the action space. Some methods [^59] [^19] [^60] [^63] [^49] attempt to directly output text-based planning results using VLM, as shown in Fig. 1(b). Although this paradigm is convenient, VLM is not well-suited for handling mathematical calculations or numerical reasoning [^13] [^42]. Besides, limited by the intrinsic autoregressive mechanism of VLM, this framework only infers single results, which is inconsistent with the natural uncertainty of human planning [^8]. Therefore, directly using VLM for E2E autonomous driving may produce suboptimal solutions in complex scenes [^62]. Other methods endeavor to bridge the gap via utilizing VLM output meta-action (e.g., turn left) to assist classic E2E methods [^26] [^40], as shown in Fig. 1(c). They adopt a carefully crafted interface to transmit the reasoning space information into the action space. However, this paradigm decouples these two spaces, hindering collaborative optimization between the trajectory optimization and the VLM reasoning process. Thus, the capabilities of VLM for E2E planning are not fully leveraged by the above framework.

To tackle this problem, we propose a hOlistic E2E autonomous dRiving framework by vIsion-language instructed actiON generation, termed ORION. Inspired by the field of conditional generation [^28] [^38] [^47] [^48], where the semantic information controls the generation of detailed image features, we find that the generative model can construct a unified distribution of diverse types of data (e.g., image, text). Therefore, considering that the reasoning space of VLM and the action space of trajectory belong to different domains, we introduce a generative planner to establish a unified latent representation for aligning the two spaces. With the help of the introduced module, we take advantage of VLMs’ reasoning information to construct trajectory, facilitating the model to capture the causal relationship between scene information and driving behavior.

Furthermore, it is well-known that long-term memory is necessary for E2E autonomous driving since historical information often influences trajectory planning within the current scene. Existing VLMs for E2E methods [^19] [^62] typically concatenate multi-frame images for temporal modeling. They are constrained by the token length of VLM and incur significant computational overhead. Instead, motivated by OmniDrive [^59], which extracts features through Q-Former-styled architecture, we introduce QT-Former, a query-based temporal module. By leveraging a memory bank and a set of history queries, QT-Former effectively stores and extracts essential historical scene information to aggregate long-term visual context, further enhancing the temporal perception ability of reasoning and action space.

We evaluate the closed-loop driving ability of ORION on the Bench2Drive dataset, which builds interactive scenarios based on the CARLA [^11] simulator. ORION achieves 77.74 Driving Score (DS) and 54.62% Success Rate (SR), surpassing previous SOTA methods [^24] with 14.28 driving scores and 19.61% success rates, demonstrating the powerful superiority of ORION.

The benefits of ORION are from three aspects: 1) Thanks to the capability of the generative model to characterize the latent distribution of data, we bridge the gap between the reasoning space of VLM and the action space of trajectories through a generative planner, enabling the VLM to understand the scene and instruct trajectory generation. 2) The QT-former in ORION effectively captures long-term temporal dependencies, enabling the model to integrate temporal vision context into reasoning and action spaces. 3) Without bells and whistles, ORION achieves excellent performance in the Bench2Drive closed-loop benchmark. Experiments also show that ORION is compatible with diverse generative models, which further demonstrate the flexibility of our proposed framework.

## 2 Related work

### 2.1 End-to-End Autonomous Driving

End-to-end autonomous driving [^61] [^68] aims to directly process raw sensor data to predict motion trajectories or control signals, jointly optimizing the entire system to minimize error accumulation. Recent works like UniAD [^18] and VAD [^25] integrate perception, prediction, and planning into a unified planning framework, making the framework ultimately planning-oriented. VADv2 [^8] introduces probabilistic planning, outputting the probabilistic distribution of action and sampling one action to control the vehicle. GenAD [^70] and DiffusionDrive [^32] explore a new paradigm for end-to-end autonomous driving, employing the generative model to predict multi-modal trajectories. However, these methods mainly excel in open-loop evaluation, where the model could readily overfit to the ego status, as highlighted in Ego-MLP [^64] and BEV-Planner [^31]. Although some studies [^8] [^70] [^22] [^21] adopt closed-loop evaluation in CARLA [^11] to assess robust driving ability, their performance remains suboptimal, revealing a notable gap between their open-loop and closed-loop results. Thus, we aim to construct an E2E autonomous driving system that demonstrates excellent performance in both open-loop and closed-loop evaluations.

![[x2 6.png|Refer to caption]]

Figure 2: The pipeline of our ORION, a holistic E2E framework aligning vision-reasoning-action space. It consists of three key components: a QT-Former to extract long-term context and link the vision space of the vision encoder and the reasoning space of LLM; the LLM for performing reasoning tasks and predicting a planning token; and a generative planner that bridges reasoning and action space for generating a multi-modal trajectory conditioned by the planning token.

### 2.2 Vision-Language Models (VLMs)

VLMs [^1] [^3] [^35] [^57] [^10] [^30] introduce visual information to large language models (LLMs) [^55] [^39] through various vision encoders [^45] [^65], demonstrating powerful visual contextual understanding and reasoning. LLaVA series [^35] [^36] employ visual instruction tuning to perform image-text alignment. Monkey [^30] improves detail comprehension by dividing images. InternVL series [^10] [^9] further enhances the vision detail understanding via a dynamic resolution strategy. However, most methods map the numerous visual tokens into language space through MLP, incurring high computational costs. To alleviate this burden, QwenVL [^4] and Flamingo [^2] reduce token redundancy using cross-attention, while Qwen2VL [^57] enhances efficiency with dynamic resolution and multimodal rotary position embedding (M-RoPE) for simultaneously processing diverse modalities. Inspired by these, we introduce QT-Former, which leverages a set of queries and cross-attention operations to extract multi-view image features.

### 2.3 VLM for End-to-End Autonomous Driving

VLMs showcase excellent contextual understanding and comprehensive world knowledge, motivating their application in autonomous driving. Some methods [^59] [^19] [^62] directly employ VLMs for environment perception and explainable trajectory prediction in text form. For example, Omnidrive [^59] adopts StreamPETR [^58] as Q-Formar3D to compress current scene features and connect the vision-reasoning space and then performs textual trajectory prediction. EMMA [^19], trained on large-scale data, enables Gemini [^3] to predict discrete textual planning with strong open-loop performance. Other studies [^54] [^26] integrate VLMs with representative E2E models in a fast-slow dual system. DriveVLM [^54] leverages VLM to predict the low-frequency trajectory, which will be refined by an E2E model. Senna [^26] further replaces the low-frequency with the meta-action, guiding the VAD [^25] to predict motion. These methods only implement the open-loop evaluation. Although DriveMLM [^60] and LMDrive [^49] leverage the VLM to implement closed-loop evaluation, they struggle with processing complex scenarios limited by the simple CARLA Town05Long benchmark. In contrast, we propose a holistic E2E framework that employs a generative planner to bridge the reasoning space of VLM and the action space of trajectories, generating precise trajectories with interpretable action decisions in complex real-world driving scenarios of Bench2Drive.

## 3 Method

In this paper, we propose a hOlistic end-to-end autonomous dRiving framework by vIsion-language model instructed actiON generation, termed ORION. The pipeline of our ORION is shown in Fig. 2. Specifically, given the multi-view images of the current scene, the ORION first encodes the image tokens with a vision encoder. Then, QT-Former (Sec. 3.1) leverages diverse queries to aggregate long-term vision context, compress image tokens, and perceive traffic elements. The LLM (Sec. 3.2) subsequently combines the compressed scene features and historical vision information with user instructions, performing diverse understanding and reasoning tasks and generating a planning token. Finally, a generative planner (Sec. 3.3) bridges the reasoning space of LLM and the action space of trajectories, predicting multi-modal trajectory conditioned by the planning token. ORION effectively aligns the vision-reasoning-action space through these core components, achieving the collaborative optimization of scene understanding and trajectory generation in a unified space.

![[x3 5.png|Refer to caption]]

Figure 3: The detailed architecture of QT-Former. It accepts diverse queries and image features as inputs to detect traffic elements, predict motion, and aggregate long-term vision context.

### 3.1 QT-Former

To achieve long-term information modeling while compressing and extracting multi-view image features $F_{m}$ derived from the vision encoder, we introduce QT-Former, a query-based temporal module, as shown in Fig. 3. Specifically, following Q-Former3D [^59], we first set up two types of learnable queries, the scene queries $Q_{s}\in\mathbb{R}^{N_{s}\times C_{q}}$ and the perception queries $Q_{p}\in\mathbb{R}^{N_{p}\times C_{q}}$, where $N_{s}$ and $N_{p}$ are the number of scene and perception queries, respectively, and $C_{q}$ is the channel of queries. ${Q}_{s},{Q}_{p}$ are processed through self-attention (SA) to exchange their information. Then they interact with image features $F_{m}$ with 3D positional encoding [^37] $P_{m}$ in the cross-attention (CA) module. After that, the perception queries are fed into the multiple auxiliary heads for object detection (e.g., critical objects and lanes), traffic state, and motion prediction of dynamic agents. The scene queries serve as tokens representing the key information of the current scene.

Additionally, we employ a set of history queries $Q_{h}\in\mathbb{R}^{N_{h}\times C_{q}}$ and a long-term memory bank $M\in\mathbb{R}^{(N_{h}\times n)\times C_{q}}$ to efficiently retrieve and store essential historical information (e.g., previous road conditions and ego status), where $N_{h}$ is the number of history queries and $n$ is the maximum history frame length. We utilize the $Q_{h}$ to extract the former frame queries in $M$ with relative timestamp embedding $P_{t}$ through a CA block. Then $Q_{h}$ interacts with current scene features $Q_{s}$ in another CA block, enabling the extraction of relevant details about the current scenario. This process can be formulated as:

$$
\displaystyle Q_{h}
$$
 
$$
\displaystyle=\text{CA}(Q_{h},M+P_{t},M+P_{t}),
$$
$$
\displaystyle\hat{Q}_{h}
$$
 
$$
\displaystyle=\text{CA}(Q_{h},Q_{s},Q_{s}),
$$

where $P_{t}$ denotes the relative timestamp embedding.

Subsequently, the updated history queries $\hat{Q}_{h}$ are stored in the memory bank $M$ following the First-In-First-Out (FIFO) replacement policy, formulated as:

$$
\displaystyle M=[\hat{Q}^{t-n}_{h},\cdot\cdot\cdot,\hat{Q}^{t-1}_{h},\hat{Q}^{%
t}_{h}],
$$

where $t$ is the current frame time.

Although some methods [^58] [^51] also leverage the memory bank to store preceding information, they typically only store the compressed historical information without guiding for extracting the current scene information. Instead, we initialize a few numbers of the history queries to further extract the current scene features that are most closely related to historical information, enhancing the long-term memory ability of the model.

Finally, we utilize a two-layer MLP to convert the updated history queries $\hat{Q}_{h}$ and current scene features $Q_{s}$ to history tokens $x_{h}$ and scene tokens $x_{s}$ in the reasoning space of LLM.

### 3.2 Large Language Model

The LLM is pivotal in our framework because the high-quality reasoning of the current driving scenario is necessary to instruct the generative planner to generate a reasonable trajectory in action space.

As shown in Fig. 2, the user instruction is first encoded into language tokens $x_{q}\in\mathbb{R}^{L\times C}$ by the text tokenizer, where $L$ is the token length and $C$ is the dimension of LLM. Then, the scene tokens $x_{s}$ and history tokens $x_{h}$ are combined with the language tokens $x_{q}$ and fed into LLM.

Leveraging the abundant world knowledge and outstanding reasoning ability of LLM, ORION performs various text-based understanding and reasoning tasks in the driving scenario, including scene description, history information review, scene analysis, and action reasoning. Meanwhile, we design a planning QA template with a special planning token $s$ for LLM as the final QA to accumulate the understanding and reasoning context of the entire driving scenario to the $s$, formally written as:

$$
s\sim p(s|x_{s},x_{h},x_{q},x_{a}),
$$

where $x_{a}$ denotes the generation answer of LLM. The embedding of the planning token $s$ will serve as a condition to control the trajectory generation.

However, there is still a lack of high-quality VQA annotations within closed-loop simulation environments to train LLMs for comprehensively understanding driving scenarios. Thus, we extend the Bench2Drive dataset via a fully automatic VQA annotation pipeline powered by Qwen2-VL [^57] and propose our VQA dataset, Chat-B2D, expecting to further promote the research of VLM on closed-loop simulation. We provide detailed information on Chat-B2D and its annotation pipeline in the Appendix.

### 3.3 Generative Planner

Generative models [^28] [^48] [^14] can effectively capture intrinsic features within data by learning the distribution of the data. Recent researches [^47] [^5] [^38] have demonstrated semantic correlations between latent spaces of different modalities of data, where adjusting the distribution parameters of one modality space enables precise control over the generation process of another modality space.

Inspired by the generative domain, we introduce a generative planner to bridge the gap between the reasoning and action space. Specifically, we formulate the current trajectory $a$ in action space as a conditional probability distribution $p(a|s)$, where $s$ is the planning token. To construct $p(a|s)$, there are many excellent methods in the generation field (e.g., variational autoencoders (VAE) [^28] and diffusion model [^48]).

As there are essential differences in the distribution between the reasoning space of VLM and the action space of trajectory, we use the VAE [^28] model to align them in the Gaussian distribution. We employ two-layer MLPs to project both the state $s$ and the ground-truth trajectory $t$ into Gaussian variables $z$ in the latent space, denoted as:

$$
p(z_{s}|s)\sim N(\mathbf{\mu}_{s},\mathbf{\sigma}_{s}^{2}),p(z_{t}|t)\sim N(%
\mathbf{\mu}_{t},\mathbf{\sigma}_{t}^{2}),
$$

where $N(\mathbf{\mu},\mathbf{\sigma}^{2})$ denotes a Gaussian distribution with a mean of $\mathbf{\mu},$ and standard deviation of $\mathbf{\sigma}$. We then use Kullback-Leibler divergence loss to enforce distribution matching, represented as:

$$
\mathcal{L}_{vae}=D_{KL}(p(\mathbf{z}|\mathbf{s}),p(\mathbf{z}|\mathbf{t})).
$$

Finally, we use the GRU decoder in GenAD [^70] to decode the trajectory from the latent space $z$. Significantly, the functions of VAE in this paper are not the same as VAE of GenAD. We only use a single token encoded in the reasoning space from the perspective of the ego vehicle as input, aiming to bridge the gap between reasoning space and action space. In contrast, the latter leverages features of all agents encoded in the BEV space as input, designed to learn specific patterns of the highly structured trajectories of both the ego vehicle and other agents.

Additionally, we also attempt to replace the VAE with alternative generative models, such as the diffusion model for trajectory generation. Benefiting from the proposed method that bridges the gap between the reasoning and action space through distribution learning in latent space, our framework still demonstrates superior performance compared to other methods (detailed in Sec. 4.5).

### 3.4 Training Objectives

For the detection task of the proposed QT-Former, the detection loss is defined as $\mathcal{L}_{det}=\mathcal{L}_{cls}+\mathcal{L}_{reg}$, where $\mathcal{L}_{cls}$ is focal loss [^34] and $\mathcal{L}_{reg}$ is L1 loss. For the traffic state and motion prediction, the losses are defined as $\mathcal{L}_{tra}$ and $\mathcal{L}_{m}=\mathcal{L}_{mcls}+\mathcal{L}_{mreg}$, respectively, where $\mathcal{L}_{tra}$ and $\mathcal{L}_{mcls}$ are focal loss, and $\mathcal{L}_{mreg}$ is L1 loss. The total loss of QT-Former is:

$$
\mathcal{L}_{qt}=\mathcal{L}_{det}+\mathcal{L}_{tra}+\mathcal{L}_{m}.
$$

For the LLM, we leverage the auto-regressive cross-entropy loss $\mathcal{L}_{ce}$. For the generative planner in our framework, $\mathcal{L}_{vae}$ is the Kullback-Leibler divergence loss used to align the reasoning space and action space. Following VAD [^25], we adopt the collision loss $\mathcal{L}_{col}$, boundary loss $\mathcal{L}_{bd}$, and MSE loss $\mathcal{L}_{mse}$ for the planning prediction. The total loss of the generative planner is:

$$
\mathcal{L}_{gp}=\mathcal{L}_{vae}+\mathcal{L}_{mse}+\mathcal{L}_{col}+%
\mathcal{L}_{bd}.
$$

In summary, the total loss of the proposed ORION is:

$$
\mathcal{L}=\mathcal{L}_{qt}+\mathcal{L}_{ce}+\mathcal{L}_{gp}.
$$

## 4 Experiments

### 4.1 Dataset and Evaluation Metrics

Dataset. We train and evaluate ORION on the Bench2drive dataset [^23], a closed-loop evaluation protocol under CARLA V2 [^11] for E2E autonomous driving. It provides an official training set where we use the base set (1000 clips) for fair comparison with all the other baselines, which is divided into 950 clips for training and 50 clips for open-loop validation. Each clip captures approximately 150 meters of continuous driving within a specific traffic scene. For closed-loop evaluation, we evaluate the proposed method on the official set of 220 short routes designed by Bench2drive, spanning 44 interactive scenarios with 5 routes per scenario. Additionally, we compare our method with other SOTA baselines on nuScenes [^6] open-loop evaluation, which will be provided in the Appendix.

Evaluation Metrics. Bench2drive includes five metrics for closed-loop evaluation: Driving Score (DS), Success Rate (SR), Efficiency, Comfortness, and Multi-Ability. The Success Rate quantifies the proportion of routes successfully completed within the allotted time. The Driving Score follows CARLA [^11], incorporating both route completion status and violation penalties, where infractions reduce the score via discount factors. Efficiency and Comfortness are used to measure the speed performance and comfort of the autonomous driving system during the driving process, respectively. Multi-Ability measures 5 advanced skills independently for urban driving. For open-loop evaluation, we use the L2 distance error and the collision rate. Additionally, we use CIDEr [^56], BLEU [^41], and ROUGE-L [^33] to evaluate the performance of ORION on VQA tasks.

Table 1: Closed-loop and Open-loop Results of E2E-AD Methods in Bench2Drive under base set. C/L refers to camera/LiDAR. Avg. L2 is averaged over the predictions in 2 seconds under 2Hz, similar to UniAD. \* denote expert feature distillation. NC: navigation command, TP: target point, DS: Driving Score, SR: Success Rate.

<table><tbody><tr><th rowspan="2">Method</th><td rowspan="2">Reference</td><td rowspan="2">Condition</td><td rowspan="2">Modality</td><td colspan="4">Closed-loop Metric</td><td>Open-loop Metric</td></tr><tr><td>DS <math><semantics><mo>↑</mo> <annotation-xml><ci>↑</ci></annotation-xml> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>SR(%) <math><semantics><mo>↑</mo> <annotation-xml><ci>↑</ci></annotation-xml> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>Efficiency <math><semantics><mo>↑</mo> <annotation-xml><ci>↑</ci></annotation-xml> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>Comfortness <math><semantics><mo>↑</mo> <annotation-xml><ci>↑</ci></annotation-xml> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>Avg. L2 <math><semantics><mo>↓</mo> <annotation-xml><ci>↓</ci></annotation-xml> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></td></tr><tr><th>TCP* <sup><a href="#fn:61">61</a></sup></th><td>NeurIPS 22</td><td>TP</td><td>C</td><td>40.70</td><td>15.00</td><td>54.26</td><td>47.80</td><td>1.70</td></tr><tr><th>TCP-ctrl*</th><td>NeurIPS 22</td><td>TP</td><td>C</td><td>30.47</td><td>7.27</td><td>55.97</td><td>51.51</td><td>-</td></tr><tr><th>TCP-traj*</th><td>NeurIPS 22</td><td>TP</td><td>C</td><td>59.90</td><td>30.00</td><td>76.54</td><td>18.08</td><td>1.70</td></tr><tr><th>TCP-traj w/o distillation</th><td>NeurIPS 22</td><td>TP</td><td>C</td><td>49.30</td><td>20.45</td><td>78.78</td><td>22.96</td><td>1.96</td></tr><tr><th>ThinkTwice* <sup><a href="#fn:22">22</a></sup></th><td>CVPR 23</td><td>TP</td><td>C</td><td>62.44</td><td>31.23</td><td>69.33</td><td>16.22</td><td>0.95</td></tr><tr><th>DriveAdapter* <sup><a href="#fn:21">21</a></sup></th><td>ICCV 23</td><td>TP</td><td>C&L</td><td>64.22</td><td>33.08</td><td>70.22</td><td>16.01</td><td>1.01</td></tr><tr><th>AD-MLP <sup><a href="#fn:64">64</a></sup></th><td>arXiv 23</td><td>NC</td><td>C</td><td>18.05</td><td>0.00</td><td>48.45</td><td>22.63</td><td>3.64</td></tr><tr><th>UniAD-Tiny <sup><a href="#fn:18">18</a></sup></th><td>CVPR 23</td><td>NC</td><td>C</td><td>40.73</td><td>13.18</td><td>123.92</td><td>47.04</td><td>0.80</td></tr><tr><th>UniAD-Base <sup><a href="#fn:18">18</a></sup></th><td>CVPR 23</td><td>NC</td><td>C</td><td>45.81</td><td>16.36</td><td>129.21</td><td>43.58</td><td>0.73</td></tr><tr><th>VAD <sup><a href="#fn:25">25</a></sup></th><td>ICCV 23</td><td>NC</td><td>C</td><td>42.35</td><td>15.00</td><td>157.94</td><td>46.01</td><td>0.91</td></tr><tr><th>GenAD <sup><a href="#fn:70">70</a></sup></th><td>ECCV 24</td><td>NC</td><td>C</td><td>44.81</td><td>15.90</td><td>-</td><td>-</td><td>-</td></tr><tr><th>MomAD <sup><a href="#fn:52">52</a></sup></th><td>CVPR25</td><td>NC</td><td>C</td><td>44.54</td><td>16.71</td><td>170.21</td><td>48.63</td><td>0.87</td></tr><tr><th>DriveTransformer-Large <sup><a href="#fn:24">24</a></sup></th><td>ICLR 25</td><td>NC</td><td>C</td><td>63.46</td><td>35.01</td><td>100.64</td><td>20.78</td><td>0.62</td></tr><tr><th>ORION(Ours)</th><td>-</td><td>NC</td><td>C</td><td>77.74(+14.28)</td><td>54.62(+19.61)</td><td>151.48</td><td>17.38</td><td>0.68</td></tr></tbody></table>

Table 2: Multi-Ability Results of E2E-AD Methods under base set. \* denote expert feature distillation. C/L refers to camera/LiDAR. NC: navigation command, TP: target point.

<table><tbody><tr><th rowspan="2">Method</th><td rowspan="2">Reference</td><td rowspan="2">Condition</td><td rowspan="2">Modality</td><td colspan="5">Ability (%) <math><semantics><mo>↑</mo> <annotation-xml><ci>↑</ci></annotation-xml> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td></td></tr><tr><td>Merging</td><td>Overtaking</td><td>Emergency Brake</td><td>Give Way</td><td>Traffic Sign</td><td>Mean</td></tr><tr><th>TCP* <sup><a href="#fn:61">61</a></sup></th><td>NeurIPS 22</td><td>TP</td><td>C</td><td>16.18</td><td>20.00</td><td>20.00</td><td>10.00</td><td>6.99</td><td>14.63</td></tr><tr><th>TCP-ctrl*</th><td>NeurIPS 22</td><td>TP</td><td>C</td><td>10.29</td><td>4.44</td><td>10.00</td><td>10.00</td><td>6.45</td><td>8.23</td></tr><tr><th>TCP-traj*</th><td>NeurIPS 22</td><td>TP</td><td>C</td><td>8.89</td><td>24.29</td><td>51.67</td><td>40.00</td><td>46.28</td><td>34.22</td></tr><tr><th>TCP-traj w/o distillation</th><td>NeurIPS 22</td><td>TP</td><td>C</td><td>17.14</td><td>6.67</td><td>40.00</td><td>50.00</td><td>28.72</td><td>28.51</td></tr><tr><th>ThinkTwice* <sup><a href="#fn:22">22</a></sup></th><td>CVPR 23</td><td>TP</td><td>C</td><td>27.38</td><td>18.42</td><td>35.82</td><td>50.00</td><td>54.23</td><td>37.17</td></tr><tr><th>DriveAdapter* <sup><a href="#fn:21">21</a></sup></th><td>ICCV 23</td><td>TP</td><td>C&L</td><td>28.82</td><td>26.38</td><td>48.76</td><td>50.00</td><td>56.43</td><td>42.08</td></tr><tr><th>AD-MLP <sup><a href="#fn:64">64</a></sup></th><td>arXiv 23</td><td>NC</td><td>C</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>4.35</td><td>0.87</td></tr><tr><th>UniAD-Tiny <sup><a href="#fn:18">18</a></sup></th><td>CVPR 23</td><td>NC</td><td>C</td><td>8.89</td><td>9.33</td><td>20.00</td><td>20.00</td><td>15.43</td><td>14.73</td></tr><tr><th>UniAD-Base <sup><a href="#fn:18">18</a></sup></th><td>CVPR 23</td><td>NC</td><td>C</td><td>14.10</td><td>17.78</td><td>21.67</td><td>10.00</td><td>14.21</td><td>15.55</td></tr><tr><th>VAD <sup><a href="#fn:25">25</a></sup></th><td>ICCV 23</td><td>NC</td><td>C</td><td>8.11</td><td>24.44</td><td>18.64</td><td>20.00</td><td>19.15</td><td>18.07</td></tr><tr><th>DriveTransformer-Large <sup><a href="#fn:24">24</a></sup></th><td>ICLR 25</td><td>NC</td><td>C</td><td>17.57</td><td>35.00</td><td>48.36</td><td>40.00</td><td>52.10</td><td>38.60</td></tr><tr><th>ORION (Ours)</th><td>-</td><td>NC</td><td>C</td><td>25.00</td><td>71.11</td><td>78.33</td><td>30.00</td><td>69.15</td><td>54.72(+16.12)</td></tr></tbody></table>

### 4.2 Implementation Details

Model Setting. Consistent with classic E2E baselines [^18] [^25] [^70] on Bench2Drive, ORION is a fully HD map-free method that only uses the Navigation Command (NC) as an input condition for the trajectory predictions rather than locations of lane center (i.e., target point, TP). ORION is an anchor-free method that outputs 6 mode trajectory predictions corresponding to the 6 NC defined in Bench2Drive.

Training Process. All experiments are conducted on 32 NVIDIA A800 GPUs with 80 GB of memory. Following Omnidrive [^59], we adopt EVA-02-L [^12] as the vision encoder. Vicuna v1.5 [^69] is employed in ORION and fine-tuned using LoRA [^16], with the rank dimension and alpha set to 16. The default number of scene, perception, and historical queries is 512, 600, and 16, respectively. We set the Memory Bank’s stored frame number $n$ to 16. During training, data augmentations are applied to input images, which are first resized to a resolution of $640\times 640$. More training details are provided in the Appendix.

### 4.3 Main Results

As reported in Tab. 1, the performance of ORION significantly exceeds all E2E methods on Bench2Drive, even the method with expert feature distillation. Specifically, ORION surpasses the latest SOTA method DriveTransformer [^24] by +14.28 DS and +19.61% SR. It also achieves improvements of +13.52 DS and +21.54% SR over DriveAdapter [^21], even if DriveAdapter distills the expert feature from Think2Drive [^29] and leverages two modalities (i.e., camera and LiDAR) inputs. The above promising results effectively demonstrate the superiority of our ORION.

Additionally, the Multi-Ability results are also illustrated in Tab. 2. ORION achieves +16.12% and +12.64% performance improvements compared with DriveTransformer [^24] and DriveAdapter [^21] in the mean ability, respectively. Specifically, our model demonstrates outstanding performance in some scenarios, such as Overtaking (71.11%), Emergency Brake (78.33%), and Traffic Sign (69.15%), which shows that our model benefits from the powerful reasoning capability of VLM to understand the causal interaction between the ego vehicle, dynamic elements, and static elements (Traffic Signs) in driving scenarios. On the other hand, our model falls behind DriveAdapter in Merging and Give Way, which shows that ORION is not good at making lane-changing decisions. The phenomenon may be caused by the more diverse decision-making timing for lane-changing, making the model encounter difficulties in capturing the correct causal relationship [^21].

![[x4 4.png|Refer to caption]]

Figure 4: Qualitative results of ORION on the Bench2Drive closed-loop evaluation set. The brown, red, and green refer to the action decision, the objects that influence driving decisions, and the prediction trajectory, respectively.

![[x5 4.png|Refer to caption]]

Figure 5: 25

### 4.4 Qualitative Results

The qualitative results of ORION in two canonical closed-loop evaluation scenarios of Bench2Drive are shown in Fig. 4. It shows both the driving action reasoning and trajectory prediction outputted by our model, as well as the corresponding ego-vehicle states. We observe that ORION can capture the correct causal relationship in the scenario and make correct driving decisions, then predict the planning trajectory following the reasoning instruction, demonstrating the surprising interpretability of our method. More qualitative results can be found in the Appendix.

### 4.5 Ablation Study

Advantages of the vision-language instructed action generation. To validate the effectiveness of the planning generation paradigm proposed in this paper, extensive experiments are conducted to compare our paradigm with canonical trajectory prediction paradigms of VLM-based E2E autonomous driving methods, including (a) plain text outputs [^59] [^19], (b) dual-system paradigm which classic E2E methods(e.g., VAD [^25]) output trajectory guided by elaborated design VLM interface (e.g., meta-action) [^26], and (c) special token decode outputs by MLP [^46], as shown in the left part of Fig. 5. To ensure the fairness of the ablations, experiments of different paradigms use the same sensor inputs, vision encoder, QT-former, and VLM as our ORION and are trained by the same strategy. Only the output formats of VLMs are adjusted according to the requirements of different paradigms.

The results are illustrated in the right part of Fig. 5. The plain text paradigm performs the worst (42.23 DS, 13.14% SR, and 15.39% mean ability), indicating the limitations of plain text output in closed-loop driving scenarios, potentially due to its inadequate numerical reasoning capabilities [^13] [^42]. Compared with the plain text paradigm, the dual-system paradigm only obtains a slight performance improvement. Note that the reproduced results of the dual-system paradigm are very close to the official results of VAD in Tab. 1. This result may indicate that the performance of the dual-system paradigm may be bottlenecked by the insufficient capabilities of classic E2E methods. Although the effectiveness of the MLP decoder paradigm has been validated in CarLLaVA [^46], our paradigm still shows a performance gain of +7.01 DS, +9.5% SR, and +6.28% mean ability. The result may be caused by the fact that the MLP is the simplest way to align features between different spaces, which is consistent with the viewpoint presented in this paper. Additionally, the MLP decoder struggles with handling multi-modal trajectory [^20] [^8], making it still significantly lag behind ORION in closed-loop evaluation.

Table 3: Ablation on diverse generative planner. DS and SR denote Driving Score and Success Rate separately.

<table><thead><tr><th rowspan="2">Generative Planner</th><th colspan="2">Closed-loop</th><th colspan="2">Open-loop</th><th>Ability</th></tr><tr><th>DS <math><semantics><mo>↑</mo> <annotation-xml><ci>↑</ci></annotation-xml> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></th><th>SR(%) <math><semantics><mo>↑</mo> <annotation-xml><ci>↑</ci></annotation-xml> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></th><th>Avg. L2 (m) <math><semantics><mo>↓</mo> <annotation-xml><ci>↓</ci></annotation-xml> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></th><th>Avg. col (%) <math><semantics><mo>↓</mo> <annotation-xml><ci>↓</ci></annotation-xml> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></th><th>Avg.</th></tr></thead><tbody><tr><th>Diffusion</th><td>71.97</td><td>46.54</td><td>0.73</td><td>0.96</td><td>46.68</td></tr><tr><th>VAE (Ours)</th><td>77.74</td><td>54.62</td><td>0.68</td><td>0.47</td><td>54.72</td></tr></tbody></table>

Table 4: Ablation on QT-Former designs in different frameworks. DS and SR denote Driving Score and Success Rate separately. Traffic state means using explicit traffic state supervision. T: Plain Text, G: Instructed Generator

<table><tbody><tr><th rowspan="2">ID</th><td rowspan="2">Traffic State</td><td rowspan="2">Motion Pred.</td><td rowspan="2">Memory Bank</td><td colspan="2">Output type</td><td colspan="2">Closed-loop</td></tr><tr><td>T</td><td>G</td><td>DS <math><semantics><mo>↑</mo> <annotation-xml><ci>↑</ci></annotation-xml> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>SR(%) <math><semantics><mo>↑</mo> <annotation-xml><ci>↑</ci></annotation-xml> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td></tr><tr><th>1</th><td></td><td></td><td></td><td></td><td>✓</td><td>56.33</td><td>26.05</td></tr><tr><th>2</th><td>✓</td><td></td><td></td><td></td><td>✓</td><td>74.65</td><td>49.31</td></tr><tr><th>3</th><td>✓</td><td>✓</td><td></td><td></td><td>✓</td><td>74.07</td><td>49.77</td></tr><tr><th>4</th><td>✓</td><td>✓</td><td>✓</td><td></td><td>✓</td><td>77.74</td><td>54.62</td></tr><tr><th>5</th><td></td><td></td><td></td><td>✓</td><td></td><td>25.45</td><td>10.38</td></tr><tr><th>6</th><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td><td>42.23</td><td>13.14</td></tr></tbody></table>

Analysis on different generative planners. We then investigate the effect of employing different generative planners to bridge the reasoning-action space. Specifically, we implement the diffusion model by simply replacing the VAE, which uses K-means trajectory anchors as prior information and outputs 20 mode trajectory predictions. The results are listed in Tab. 3. Note that the VAE-based trajectory generator demonstrates a significant performance improvement over the diffusion-based. We argue the main reasons are as follows: 1) Compared with the conditional denoising process of diffusion, the latent space of VAE more directly and effectively aligns the reasoning information of VLM to the multi-modal action space. 2) The training process of VAE is inherently more stable, facilitating better alignment between the reasoning and action spaces. Surprisingly, even using diffusion, ORION still surpasses the DriveTransformer by +8.51 DS, +11.53% SR, and +8.08% mean ability. This impressive result emphasizes the effectiveness and flexibility of our framework.

Effectiveness of QT-Former designs. Tab. 4 shows the detailed ablations of each design in the introduced QT-Former. By leveraging explicit traffic state supervision (ID-2), ORION achieves 74.65 DS and 49.31% SR, which already outperforms DriveAdapter [^21] and DriveTransformer [^24] by a large margin and makes an improvement of +18.32 and +23.26% compared with the baseline (ID-1). This is because a better understanding of traffic signals helps ORION directly reduce infractions in closed-loop evaluation. It is worth noting that due to the causal confusion [^21], it’s not trivial for previous methods to fully understand the corresponding causal relationships by simply introducing traffic state supervision, especially when encountering mixed expert behaviors before traffic signs [^21] [^24] [^22] [^61]. This result also proves that ORION can better utilize the reasoning ability of VLM to capture the causal relationship between scene information and driving behavior by aligning reasoning space and action space. This conclusion can also be verified by the results in Tab. 2, where ORION shows a significant advantage in traffic sign ability (+17.05%) compared to previous E2E methods [^24].

Table 5: Ablation of history queries number. DS and SR denote Driving Score and Success Rate separately.

<table><thead><tr><th rowspan="2">Query Num. <math><semantics><msub><mi>N</mi> <mi>h</mi></msub> <annotation-xml><apply><csymbol>subscript</csymbol> <ci>𝑁</ci> <ci>ℎ</ci></apply></annotation-xml> <annotation>N_{h}</annotation> <annotation>italic_N start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT</annotation></semantics></math></th><th colspan="2">Closed-loop</th><th colspan="2">Open-loop</th></tr><tr><th>DS <math><semantics><mo>↑</mo> <annotation-xml><ci>↑</ci></annotation-xml> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></th><th>SR(%) <math><semantics><mo>↑</mo> <annotation-xml><ci>↑</ci></annotation-xml> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></th><th>Avg. L2 (m) <math><semantics><mo>↓</mo> <annotation-xml><ci>↓</ci></annotation-xml> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></th><th>Avg. col (%) <math><semantics><mo>↓</mo> <annotation-xml><ci>↓</ci></annotation-xml> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></th></tr></thead><tbody><tr><th>0</th><td>65.10</td><td>38.83</td><td>0.67</td><td>0.61</td></tr><tr><th>8</th><td>68.09</td><td>39.09</td><td>0.66</td><td>0.62</td></tr><tr><th>16</th><td>74.10</td><td>44.66</td><td>0.68</td><td>0.55</td></tr><tr><th>32</th><td>62.46</td><td>37.73</td><td>0.65</td><td>0.73</td></tr></tbody></table>

Then, we combine the motion prediction module in the QT-Former’s perception head, which gains a slight improvement of +0.4% SR and further reduces the collision rate. The slight degradation on DS may be caused by the trade-off between DS and SR in the CARLA benchmark protocol [^71]. Involving a memory bank into QT-Former and supervised by QA pairs about historical information leads to an increase of +3.67 DS and +4.85% SR and boosts the final performance to 77.74 DS and 54.62% SR, which demonstrates our model can benefit from the long-temporal memory of vision tokens.

We also apply QT-former to the plain text output type (ID-6). By leveraging it, we improve the model’s performance by +16.78 DS and +2.78% SR over the baseline (ID-5). Meanwhile, with the same QT-former designs, our ORION achieves further improvements of +35.51 DS and +41.48% SR compared with the plain text output mode, demonstrating the effectiveness of our framework.

Influence of the number of history queries. We conduct ablation experiments to further study the influence of different numbers of history queries. Here, to accelerate the training process, we only train the model using the planning trajectory and history QA pairs without other auxiliary VQA tasks. The results are detailed in Tab. 5. Increasing the history query number $N_{h}$ from 0 to 8 brings a significant performance boost of around 2.99 DS and 0.26% SR. Further increasing $N_{h}$ from 8 to 16 leads to the sweet point that achieves the best performance of 74.10 DS and 44.66% SR. However, enlarging $N_{h}$ from 16 to 32 shows a significant performance degradation. We argue that introducing more history queries hinders the VLM from capturing the current frame features, which are more essential than historical information in the driving scene.

Table 6: Effectiveness of auxiliary VQA task training. DS and SR denote Driving Score and Success Rate separately. C/B/R refers to CIDEr/BLEU/ROUGE-L. FT: Fine Tuning

<table><thead><tr><th rowspan="2">ID</th><th rowspan="2">VQA FT</th><th rowspan="2">Planning FT</th><th colspan="2">Closed-loop</th><th colspan="3">Language</th><th>Open-loop</th></tr><tr><th>DS <math><semantics><mo>↑</mo> <annotation-xml><ci>↑</ci></annotation-xml> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></th><th>SR(%) <math><semantics><mo>↑</mo> <annotation-xml><ci>↑</ci></annotation-xml> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></th><th>C <math><semantics><mo>↑</mo> <annotation-xml><ci>↑</ci></annotation-xml> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></th><th>B <math><semantics><mo>↑</mo> <annotation-xml><ci>↑</ci></annotation-xml> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></th><th>R <math><semantics><mo>↑</mo> <annotation-xml><ci>↑</ci></annotation-xml> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></th><th>Avg. L2 (m) <math><semantics><mo>↓</mo> <annotation-xml><ci>↓</ci></annotation-xml> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></th></tr></thead><tbody><tr><td>1</td><td>✓</td><td></td><td>-</td><td>-</td><td>65.65</td><td>50.82</td><td>77.65</td><td>-</td></tr><tr><td>2</td><td></td><td>✓</td><td>74.10</td><td>44.66</td><td>-</td><td>-</td><td>-</td><td>0.68</td></tr><tr><td>3</td><td>✓</td><td>✓</td><td>77.74</td><td>54.62</td><td>65.77</td><td>52.49</td><td>77.58</td><td>0.68</td></tr></tbody></table>

Influence between VQA task training and planning task training. As shown in Tab. 6. The model cannot obtain both reasoning and planning capabilities with single-task training. Surprisingly, when we perform two tasks simultaneously during training, ORION achieves better performance in both planning and language metrics compared to single-task training. Specifically, the multi-task training leads to improvements of +3.64 DS and +9.66% SR in the planning task, as well as a performance gain of +0.12 CIDEr, +1.67 BLEU and competitive performance of ROUGE-L in the VQA tasks. Furthermore, the results also validate the high quality and validity of the Chat-B2D dataset produced by our auto-pipeline.

## 5 Conclusion

In this paper, we mainly focus on the challenges faced by VLM methods for end-to-end autonomous driving in aligning the reasoning space of VLM with the pure numerical action space used for planning. This dilemma makes it not trivial for existing methods to simultaneously analyze the driving scenario and output high-quality multimodal prediction trajectories. To address this problem, we propose ORION, a holistic end-to-end autonomous driving framework by vision-language instructed action generation. By leveraging a generative planner and incorporating long-term visual context, we effectively bridge the vision-reasoning-action space. Extensive experiments validate the flexibility and superiority of our proposed framework, where ORION demonstrates significant improvements in closed-loop planning evaluation, surpassing SOTA methods.

Limitation. Although ORION performs well in the closed-loop simulation environment on Bench2Drive [^23], it is limited by the high computational complexity of the scalable VLM in real-time driving scenarios. In the future, we would like to reduce the complexity of ORION through techniques such as model compression and pruning, thereby enabling the model to achieve real-time autonomous driving.

## References

  

Supplementary Material  

Table A1: Comparison of the Open-loop planning in nuScene. †: The ego status and planning trajectory are both processed by LLM in textual modality. ${}\ddagger$: The high-level command is not used during the training and testing phases.

<table><tbody><tr><th rowspan="2">Method</th><td rowspan="2">VLM-Based</td><td colspan="2">Ego Status</td><td colspan="4">L2 (m) <math><semantics><mo>↓</mo> <annotation-xml><ci>↓</ci></annotation-xml> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></td><td colspan="4">Collision (%) <math><semantics><mo>↓</mo> <annotation-xml><ci>↓</ci></annotation-xml> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></td></tr><tr><td>BEV</td><td>Planner</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><th>ST-P3</th><td>-</td><td>-</td><td>-</td><td>1.33</td><td>2.11</td><td>2.90</td><td>2.11</td><td>0.23</td><td>0.62</td><td>1.27</td><td>0.71</td></tr><tr><th>UniAD <sup><a href="#fn:18">18</a></sup></th><td>-</td><td>-</td><td>-</td><td>0.48</td><td>0.96</td><td>1.65</td><td>1.03</td><td>0.05</td><td>0.17</td><td>0.71</td><td>0.31</td></tr><tr><th>UniAD</th><td>-</td><td>✓</td><td>✓</td><td>0.20</td><td>0.42</td><td>0.75</td><td>0.46</td><td>0.02</td><td>0.25</td><td>0.84</td><td>0.37</td></tr><tr><th>VAD-Base <sup><a href="#fn:25">25</a></sup></th><td>-</td><td>-</td><td>-</td><td>0.69</td><td>1.22</td><td>1.83</td><td>1.25</td><td>0.06</td><td>0.68</td><td>2.52</td><td>1.09</td></tr><tr><th>VAD-Base</th><td>-</td><td>✓</td><td>-</td><td>0.41</td><td>0.70</td><td>1.06</td><td>0.72</td><td>0.04</td><td>0.43</td><td>1.15</td><td>0.54</td></tr><tr><th>VAD-Base</th><td>-</td><td>✓</td><td>✓</td><td>0.17</td><td>0.34</td><td>0.60</td><td>0.37</td><td>0.04</td><td>0.27</td><td>0.67</td><td>0.33</td></tr><tr><th>Ego-MLP <sup><a href="#fn:64">64</a></sup></th><td>-</td><td>-</td><td>✓</td><td>0.15</td><td>0.32</td><td>0.59</td><td>0.35</td><td>0.00</td><td>0.27</td><td>0.85</td><td>0.37</td></tr><tr><th>BEV-Planner <sup><a href="#fn:31">31</a></sup></th><td>-</td><td>-</td><td>-</td><td>0.30</td><td>0.52</td><td>0.83</td><td>0.55</td><td>0.10</td><td>0.37</td><td>1.30</td><td>0.59</td></tr><tr><th>BEV-Planner++</th><td>-</td><td>✓</td><td>✓</td><td>0.16</td><td>0.32</td><td>0.57</td><td>0.35</td><td>0.00</td><td>0.29</td><td>0.73</td><td>0.34</td></tr><tr><th>DriveVLM† <sup><a href="#fn:54">54</a></sup></th><td>✓</td><td>-</td><td>-</td><td>0.18</td><td>0.34</td><td>0.68</td><td>0.40</td><td>0.10</td><td>0.22</td><td>0.45</td><td>0.27</td></tr><tr><th>DriveVLM-Dual <sup><a href="#fn:54">54</a></sup></th><td>✓</td><td>✓</td><td>-</td><td>0.15</td><td>0.29</td><td>0.48</td><td>0.31</td><td>0.05</td><td>0.08</td><td>0.17</td><td>0.10</td></tr><tr><th>OmniDrive <math><semantics><mo>‡</mo> <annotation-xml><ci>‡</ci></annotation-xml> <annotation>\ddagger</annotation> <annotation>‡</annotation></semantics></math> <sup><a href="#fn:59">59</a></sup></th><td>✓</td><td>-</td><td>-</td><td>1.15</td><td>1.96</td><td>2.84</td><td>1.98</td><td>0.80</td><td>3.12</td><td>7.46</td><td>3.79</td></tr><tr><th>OmniDrive</th><td>✓</td><td>-</td><td>-</td><td>0.40</td><td>0.80</td><td>1.32</td><td>0.84</td><td>0.04</td><td>0.46</td><td>2.32</td><td>0.94</td></tr><tr><th>OmniDrive++</th><td>✓</td><td>✓</td><td>✓</td><td>0.14</td><td>0.29</td><td>0.55</td><td>0.33</td><td>0.00</td><td>0.13</td><td>0.78</td><td>0.30</td></tr><tr><th>Senna <sup><a href="#fn:26">26</a></sup></th><td>✓</td><td>-</td><td>-</td><td>0.37</td><td>0.54</td><td>0.86</td><td>0.59</td><td>0.09</td><td>0.12</td><td>0.33</td><td>0.18</td></tr><tr><th>Senna</th><td>✓</td><td>✓</td><td>✓</td><td>0.11</td><td>0.21</td><td>0.35</td><td>0.22</td><td>0.04</td><td>0.08</td><td>0.13</td><td>0.08</td></tr><tr><th>EMMA† <sup><a href="#fn:19">19</a></sup></th><td>✓</td><td>-</td><td>-</td><td>0.14</td><td>0.29</td><td>0.54</td><td>0.32</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><th>ORION (Ours)</th><td>✓</td><td>✓</td><td>-</td><td>0.17</td><td>0.31</td><td>0.55</td><td>0.34</td><td>0.05</td><td>0.25</td><td>0.80</td><td>0.37</td></tr></tbody></table>

We provide supplementary material to complement the main paper, arranged as follows:

- Appendix A: Details on the Chat-B2D dataset.
- Appendix B: Traning Details.
- Appendix C: More results.

## Appendix A Details on the Chat-B2D dataset

To compensate for the absence of a high-quality scene text annotation dataset and promote the application of VLM in the closed-loop simulated driving scenario, we carefully design an automated annotation pipeline to extend the Bench2Drive dataset [^23] to support VQA pairs, named Chat-B2D, covering diverse tasks.

### A.1 Data Annotation Pipeline

As shown in Fig. A1, the automated annotation pipeline consists of three steps:

Critical object selection. Unlike mainline self-driving perception modules that process all detected objects equally, we emphasize identifying the crucial object that potentially affects the ego vehicle’s driving behavior, grounded in human driving strategies. Our selection criteria include: 1) Objects have potential collisions within three seconds. 2) Leading vehicles in current and adjacent lanes. 3) Active traffic signals. 4) The vulnerable road users (VRUs), such as pedestrians/cyclists.

Description generation. We extract video clips comprising the current and five preceding frames. Subsequently, these clips, along with the ego vehicle’s status and the ground truth information (e.g., 2D/3D coordinates and velocity, etc.) of selected crucial objects, serve as input to Qwen2VL-72B [^57] for multi-task generation: 1) the scene description; 2) attributes of key objects and their impact on the ego vehicle; 3) operational meta-commands and action reasoning for autonomous navigation.

History Information. During the generation process, we incorporate a queue mechanism to preserve essential historical information. The stored information comprises two principal components: 1) Environmental dynamics that capture spatiotemporal variations of critical scene elements, and 2) Ego-motion characteristics derived from comparative analysis between current speed/action and their historical counterparts across previous frames.

The generated description and collected historical information are combined with predefined question templates to create VQA pairs. Tab. A3 displays the detailed crafted prompt, and Tab. A4 shows the question templates.

![[x6 3.png|Refer to caption]]

Figure A1: The automated annotation pipeline for the Chat-B2D dataset.

### A.2 Chat-B2D Attribute

Through the carefully crafted prompts and the above generation pipeline, we have automatically conducted a large-scale, high-quality VQA dataset for the Bench2Drive [^23], creating Chat-B2D. This dataset, including a total of 2.11M VQA pairs for training and 0.12M for validation, supports four primary categories: 1) Scene description, which provides a comprehensive overview of the driving scenarios, including weather, time of day, traffic situations, and road characteristics. 2) Behavior description of critical objects detailing their current state and intentions. 3) Meta-driving decisions and action reasoning of the ego car, such as turning left and lane following. 4) Recall of essential historical information.

## Appendix B Training Details

To accelerate the alignment of the vision-reasoning-action space and gradually enhance the reasoning and planning capabilities of our ORION, we adopt a three-stage training strategy. In each stage, the model inherits the weights from the previous stage and continues training. Additionally, we train the model for six epochs per stage with a total batch size of 32. The three-stage training strategy is as follows:

3D Vision-Language Alignment: In this first stage, we primarily train the QT-Former and the VLM while freezing the generative planner. By training on VQA pairs from Chat-B2D, we focus on aligning the vision space with the reasoning space.

Language-Action Alignment: In this stage, we unfreeze the generative planner and train the entire model except for the LLM, which is trained by LoRA [^16], to predict planning trajectories without auxiliary VQA pairs. This stage primarily focuses on transmitting world knowledge from the reasoning space to the action space.

End-to-End Fine-tuning: We follow the training settings from the previous stage, with the only difference being the incorporation of joint training on VQA and planning tasks. This step further facilitates the alignment of the vision-reasoning-action space.

## Appendix C More Results

### C.1 Experiments on nuScenes dataset

nuScenes Dataset. nuScenes [^6] is a popular autonomous driving benchmark typically used for detection and open-loop planning evaluation. The dataset contains 1000 scenes from Singapore and Boston, with 700 scenes for training, 150 scenes for validation, and 150 scenes for testing. Each scene spans 20 seconds and is annotated at 2 Hz. nuScenes utilizes the L2 error and collision rate as planning metrics.

Results on nuScenes. We compare the ORION with previous SOTA end-to-end autonomous driving methods on the nuScenes dataset. Here, for a fair comparison with other VLM-Based methods, we modify ORION by replacing QT-Former with the Q-Former from OmniDrive [^59], and without the explicit ego status in the generative planner. As shown in Tab. A1, our ORION achieves comparable performance to classic SoTA methods [^18] [^25] [^31] without VLM. However, compared with other VLM-Based methods, our ORION is suboptimal. We argue that this is due to the latent space of VAE being more suitable for multimodal trajectory distributions of Bench2Drive [^23]. In contrast, the nuScene dataset follows a uni-modal Gaussian distribution (with straight trajectories accounting for about 70%).

Additionally, as highlighted in BEV-Planner [^31] and Ego-MLP [^64], even a simple MLP decoder with ego status can achieve strong open-loop planning performance on nuScenes. Thus, in the main paper, we primarily focus on evaluating ORION’s closed-loop performance on the Bench2Drive dataset.

![[x7 3.png|Refer to caption]]

Figure A2: Qualitative results of historical information memory and retrieval on Bench2Drive open-loop validation set.

### C.2 More Ablation Studies on Bench2Drive

Table A2: Ablation study of training strategy. V/L/A indicates vision/language/action space. DS and SR denote Driving Score and Success Rate separately. C/B/R refers to CIDEr/BLEU/ROUGE-L.

<table><thead><tr><th rowspan="2">ID</th><th rowspan="2">V <math><semantics><mo>→</mo> <annotation-xml><ci>→</ci></annotation-xml> <annotation>\to</annotation> <annotation>→</annotation></semantics></math> L</th><th rowspan="2">L <math><semantics><mo>→</mo> <annotation-xml><ci>→</ci></annotation-xml> <annotation>\to</annotation> <annotation>→</annotation></semantics></math> A</th><th rowspan="2">V <math><semantics><mo>→</mo> <annotation-xml><ci>→</ci></annotation-xml> <annotation>\to</annotation> <annotation>→</annotation></semantics></math> L <math><semantics><mo>→</mo> <annotation-xml><ci>→</ci></annotation-xml> <annotation>\to</annotation> <annotation>→</annotation></semantics></math> A</th><th colspan="2">Closed-loop</th></tr><tr><th>DS <math><semantics><mo>↑</mo> <annotation-xml><ci>↑</ci></annotation-xml> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></th><th>SR(%) <math><semantics><mo>↑</mo> <annotation-xml><ci>↑</ci></annotation-xml> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></th></tr></thead><tbody><tr><td>1</td><td></td><td>✓</td><td></td><td>57.96</td><td>26.32</td></tr><tr><td>2</td><td>✓</td><td>✓</td><td></td><td>65.10</td><td>38.83</td></tr><tr><td>3</td><td>✓</td><td>✓</td><td>✓</td><td>74.65</td><td>49.31</td></tr></tbody></table>

Ablation of training pipeline. To facilitate the vision-language-action space alignment of our model, we implement a progressive space alignment training strategy. We validate the effectiveness of the training pipeline, and the results are presented in Tab. A2. Here, the QT-Former of our model does not incorporate collision loss or long-term memory bank with history queries. Specifically, through our second-stage training(ID-2), ORION achieves a significant improvement by +7.14 DS and +12.51% SR compared to direct training planning without the first stage (ID-1). After completing the third-stage training (ID-3), our model further improved the performance and achieved optimal (74.65 DS and 49.32 SR), demonstrating the effectiveness of our training strategy.

### C.3 More Qualitative Results

Historical information memory and retrieval. Benefiting from the introduced long-term memory bank and history queries in QT-Former, our ORION could store and retrieve historical information, as illustrated in Fig. A2. Our model could perceive critical elements (e.g., traffic light) changes in previous and current times.

Scene understanding and action reasoning. Fig. A3 shows scene understanding and action reasoning results of ORION. It could be observed that ORION could not only accurately perceive detailed scene information but also identify key objects influencing the ego vehicle’s behavior and infer appropriate motion decisions. Even in extreme situations (e.g., a pedestrian suddenly crosses the road in Fig. A3(b)), our model maintains robust performance, highlighting its superior reasoning and decision-making ability.

![[x8 3.png|Refer to caption]]

Figure A3: Qualitative results for scene understanding and action reasoning on Bench2Drive open-loop validation. From top to bottom, each sub-figure displays the multi-view input and traffic conditions in Bird’s Eye View (BEV) of the current scene, the scene understanding, and the reasoning result. The red rectangles indicate the critical objects influencing the action of the ego vehicle, while the red text highlights our method’s correct scene comprehension.

Table A3: Prompts fed into Qwen2VL to generate corresponding response.

<svg class="ltx_picture ltx_centering" height="1055.22" id="A3.T3.pic1" overflow="visible" version="1.1" width="600"><g fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,1055.22) matrix(1 0 0 -1 0 0)"><g fill="#404040" fill-opacity="1.0"><path d="M 0 5.91 L 0 1049.31 C 0 1052.57 2.64 1055.22 5.91 1055.22 L 594.09 1055.22 C 597.36 1055.22 600 1052.57 600 1049.31 L 600 5.91 C 600 2.64 597.36 0 594.09 0 L 5.91 0 C 2.64 0 0 2.64 0 5.91 Z" style="stroke:none"></path></g><g fill="#F2F2F2" fill-opacity="1.0"><path d="M 1.97 5.91 L 1.97 1049.31 C 1.97 1051.49 3.73 1053.25 5.91 1053.25 L 594.09 1053.25 C 596.27 1053.25 598.03 1051.49 598.03 1049.31 L 598.03 5.91 C 598.03 3.73 596.27 1.97 594.09 1.97 L 5.91 1.97 C 3.73 1.97 1.97 3.73 1.97 5.91 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 13.78)"><foreignObject class="ltx_minipage" color="#000000" height="1027.66" overflow="visible" transform="matrix(1 0 0 -1 0 16.6)" width="402.3pt">Prompt 1: Scene Description Suppose you are driving, generate a description of the driving scene which includes the key factors for driving planning, including the traffic conditions, weather, time of day and road conditions, traffic signs, and traffic lights that affect the driving of the ego vehicle if it exists, indicating smooth surfaces or the presence of obstacles; The description should be concise, and accurate to facilitate informed decision-making. Please make sure the traffic light colors you provide are accurate; otherwise, give ‘unknown.’ Prompt 2: Critical Objects Analysis I will provide you with several critical objects that are most important to my short-term command in the last image of the video. I provide you with 2d coordinates, which are two points of the top-left and bottom-right coordinates, and the 3d position and velocity information of these critical objects: {objects_desc}. Please describe their action and explain why they are most important, including their speed, position, heading, and influence on ego vehicle. Please associate these objects with the objects in the image and please remember the ego vehicle is located at the **center of the bottom edge of the picture**. Prompt 3: Expert Meta-Decision Besides, I will provide you speed, historical trajectory and future driving behaviors of ego vehicle, which can be divided into SPEED decisions and COMMAND decisions, SPEED includes keep, accelerate, decelerate, while COMMAND includes left, right, straight, lane follow, change lane left, change lane right. Your current speed is {ego_vel} m/s, historical trajectory is {ego_his_trajs}. The next SPEED decision is {speed_decision}, the next COMMAND decision is {path_decision}. Please analyze the reasons for the future driving behaviors or the reason why ego vehicle can {path_decision} based on the driving environment, including the behavior of other traffic participants, especially the critical objects, road conditions, and traffic light status. Example: You should refer to the following example and format the results like {“description”: “xxx”,“critical_objects”: “xxx”, “action”: “{speed_decision} and {path_decision}”}}: {{ “description”: “The scene captures a moment of urban life framed by a red traffic light in mid-transition. To the right, a pedestrian crossing, …, waiting for the signal to change. Directly ahead, … On the left, the sidewalk bustles with people of all ages, … Behind this foreground of orderly traffic and pedestrian movement, …” “critical_objects: “[“Car at <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&lt;"><semantics><mo mathsize="80%">&lt;</mo> <annotation>&lt;</annotation> <annotation>&lt;</annotation></semantics></math> -0.24, 7.56 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&gt;"><semantics><mo mathsize="80%">&gt;</mo> <annotation>&gt;</annotation> <annotation>&gt;</annotation></semantics></math> directly in front of ego vehicle, …”, “Car at <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&lt;"><semantics><mo mathsize="80%">&lt;</mo> <annotation>&lt;</annotation> <annotation>&lt;</annotation></semantics></math> -2.64, 10.00 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&gt;"><semantics><mo mathsize="80%">&gt;</mo> <annotation>&gt;</annotation> <annotation>&gt;</annotation></semantics></math> …, moving at a slower speed, may influence left change.“]” “action”: “Slow down and right lane change. - The decision to change lanes is influenced by the need to overtake Car at <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&lt;"><semantics><mo mathsize="80%">&lt;</mo> <annotation>&lt;</annotation> <annotation>&lt;</annotation></semantics></math> -0.24, 7.56 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&gt;"><semantics><mo mathsize="80%">&gt;</mo> <annotation>&gt;</annotation> <annotation>&gt;</annotation></semantics></math> in front of the ego vehicle. - There are no traffic lights for the vehicle,… - Pedestrians are visible on the sidewalk to the right, …” }} If it has no critical_objects, you should refer to the following example and format the results like {{“description”: “xxx”, “critical_objects”: [], “action”: ”xxx”}}.</foreignObject></g></g></svg>

Table A4: A list of question templates for diverse VQA tasks.

<svg class="ltx_picture ltx_centering" height="1218.05" id="A3.T4.pic1" overflow="visible" version="1.1" width="600"><g fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,1218.05) matrix(1 0 0 -1 0 0)"><g fill="#404040" fill-opacity="1.0"><path d="M 0 5.91 L 0 1212.14 C 0 1215.4 2.64 1218.05 5.91 1218.05 L 594.09 1218.05 C 597.36 1218.05 600 1215.4 600 1212.14 L 600 5.91 C 600 2.64 597.36 0 594.09 0 L 5.91 0 C 2.64 0 0 2.64 0 5.91 Z" style="stroke:none"></path></g><g fill="#F2F2F2" fill-opacity="1.0"><path d="M 1.97 5.91 L 1.97 1212.14 C 1.97 1214.32 3.73 1216.08 5.91 1216.08 L 594.09 1216.08 C 596.27 1216.08 598.03 1214.32 598.03 1212.14 L 598.03 5.91 C 598.03 3.73 596.27 1.97 594.09 1.97 L 5.91 1.97 C 3.73 1.97 1.97 3.73 1.97 5.91 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 13.78)"><foreignObject class="ltx_minipage" color="#000000" height="1190.49" overflow="visible" transform="matrix(1 0 0 -1 0 16.6)" width="402.3pt">Type 1: Scene Description 1. What can you tell about the current driving conditions from the images? 2. What can be observed in the panoramic images provided? 3. Can you provide a summary of the current driving scenario based on the input images? 4. What can you observe from the provided images regarding the driving conditions? 5. Please describe the current driving conditions based on the images provided. 6. Can you describe the current weather conditions and the general environment depicted in the images? 7. Please describe the current driving conditions based on the input images. 8. Could you summarize the current driving conditions based on the input images? 9. Please provide an overview of the current driving conditions based on the images. 10. Can you summarize what the panoramic images show? 11. Can you describe the overall conditions and environment based on the images? 12. Could you describe the overall environment and objects captured in the images provided? Type 2: Critical Objects Analysis 1. Where are the critical objects in the scene and what impact do they have on the ego vehicle? 2. Identify the significant objects in the scene and their specific impacts on the ego vehicle. 3. Can you pinpoint the critical objects in the scene and describe their influence on the ego vehicle? 4. Which objects in the scene are critical, and what effects do they have on the ego vehicle’s movement? 5. Please describe the critical objects in the scene, their positions, and the influence they have on the ego vehicle. Type 3: Interpretable Action of Ego Vehicle 1. Please describe your driving behavior and explain the reasons. 2. What is the current behavior of the vehicle? Type 4: Historical Information 1. What are the differences between the current scene and the past scene in terms of critical objects? 2. How do the critical objects in the current scene differ from those in the past scene? 3. What changes have occurred in the critical objects between the current and past scenes? 4. What are the differences in critical objects between the present scene and the previous scene? 5. What distinctions exist between the critical objects of the current scene and those of the past scene? 6. In the past few frames, has a traffic light affected the driving strategy of the ego vehicle? 7. Within the recent frames, has the driving strategy of the ego vehicle been influenced by a traffic light? 8. In the last few frames, has the driving strategy of the ego vehicle been impacted by a traffic light? 9. Has the driving strategy of the ego vehicle been affected by a traffic light in the past few frames? 10. Has the traffic light influenced the driving strategy of the ego vehicle in the previous frames? 11. How has the current speed changed compared to the previous frames? 12. What was my driving behavior in the previous frame?</foreignObject></g></g></svg>

[^1]: Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023.

[^2]: Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In *Proc. of Advances in Neural Information Processing Systems*, 2022.

[^3]: Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: A family of highly capable multimodal models. *arXiv preprint arXiv:2312.11805*, 1, 2023.

[^4]: Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. *arXiv preprint arXiv:2308.12966*, 2023.

[^5]: James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. *Computer Science.*, page 8, 2023.

[^6]: Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In *Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition*, pages 11621–11631, 2020.

[^7]: Yuning Chai, Benjamin Sapp, Mayank Bansal, and Dragomir Anguelov. Multipath: Multiple probabilistic anchor trajectory hypotheses for behavior prediction. *arXiv preprint arXiv:1910.05449*, 2019.

[^8]: Shaoyu Chen, Bo Jiang, Hao Gao, Bencheng Liao, Qing Xu, Qian Zhang, Chang Huang, Wenyu Liu, and Xinggang Wang. Vadv2: End-to-end vectorized autonomous driving via probabilistic planning. *arXiv preprint arXiv:2402.13243*, 2024a.

[^9]: Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. *Science China Information Sciences*, page 220101, 2024b.

[^10]: Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In *Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition*, pages 24185–24198, 2024c.

[^11]: Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun. Carla: An open urban driving simulator. In *Conf. on Robot Learning*, pages 1–16, 2017.

[^12]: Yuxin Fang, Quan Sun, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva-02: A visual representation for neon genesis. *Image and Vision Computing*, page 105171, 2024.

[^13]: Simon Frieder, Luca Pinchetti, Ryan-Rhys Griffiths, Tommaso Salvatori, Thomas Lukasiewicz, Philipp Petersen, and Julius Berner. Mathematical capabilities of chatgpt. *Proc. of Advances in Neural Information Processing Systems*, 36, 2024.

[^14]: Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. *Communications of the ACM*, pages 139–144, 2020.

[^15]: Junru Gu, Chenxu Hu, Tianyuan Zhang, Xuanyao Chen, Yilun Wang, Yue Wang, and Hang Zhao. Vip3d: End-to-end visual trajectory prediction via 3d agent queries. In *Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition*, pages 5496–5506, 2023.

[^16]: Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In *Proc. of Intl. Conf. on Learning Representations*, 2022a.

[^17]: Shengchao Hu, Li Chen, Penghao Wu, Hongyang Li, Junchi Yan, and Dacheng Tao. St-p3: End-to-end vision-based autonomous driving via spatial-temporal feature learning. In *Proc. of European Conference on Computer Vision*, pages 533–549, 2022b.

[^18]: Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, et al. Planning-oriented autonomous driving. In *Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition*, pages 17853–17862, 2023.

[^19]: Jyh-Jing Hwang, Runsheng Xu, Hubert Lin, Wei-Chih Hung, Jingwei Ji, Kristy Choi, Di Huang, Tong He, Paul Covington, Benjamin Sapp, et al. Emma: End-to-end multimodal model for autonomous driving. *arXiv preprint arXiv:2410.23262*, 2024.

[^20]: Bernhard Jaeger, Kashyap Chitta, and Andreas Geiger. Hidden biases of end-to-end driving models. In *Porc. of IEEE Intl. Conf. on Computer Vision*, pages 8240–8249, 2023.

[^21]: Xiaosong Jia, Yulu Gao, Li Chen, Junchi Yan, Patrick Langechuan Liu, and Hongyang Li. Driveadapter: Breaking the coupling barrier of perception and planning in end-to-end autonomous driving. In *Porc. of IEEE Intl. Conf. on Computer Vision*, 2023a.

[^22]: Xiaosong Jia, Penghao Wu, Li Chen, Jiangwei Xie, Conghui He, Junchi Yan, and Hongyang Li. Think twice before driving: Towards scalable decoders for end-to-end autonomous driving. In *Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition*, 2023b.

[^23]: Xiaosong Jia, Zhenjie Yang, Qifeng Li, Zhiyuan Zhang, and Junchi Yan. Bench2drive: Towards multi-ability benchmarking of closed-loop end-to-end autonomous driving. *Proc. of Advances in Neural Information Processing Systems*, 2024.

[^24]: Xiaosong Jia, Junqi You, Zhiyuan Zhang, and Junchi Yan. Drivetransformer: Unified transformer for scalable end-to-end autonomous driving. In *Proc. of Intl. Conf. on Learning Representations*, 2025.

[^25]: Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, and Xinggang Wang. Vad: Vectorized scene representation for efficient autonomous driving. In *Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition*, pages 8340–8350, 2023.

[^26]: Bo Jiang, Shaoyu Chen, Bencheng Liao, Xingyu Zhang, Wei Yin, Qian Zhang, Chang Huang, Wenyu Liu, and Xinggang Wang. Senna: Bridging large vision-language models and end-to-end autonomous driving. *arXiv preprint arXiv:2410.22313*, 2024a.

[^27]: Xiaohui Jiang, Shuailin Li, Yingfei Liu, Shihao Wang, Fan Jia, Tiancai Wang, Lijin Han, and Xiangyu Zhang. Far3d: Expanding the horizon for surround-view 3d object detection. In *Proc. of the AAAI Conf. on Artificial Intelligence*, pages 2561–2569, 2024b.

[^28]: Diederik P Kingma. Auto-encoding variational bayes. *arXiv preprint arXiv:1312.6114*, 2013.

[^29]: Qifeng Li, Xiaosong Jia, Shaobo Wang, and Junchi Yan. Think2drive: Efficient reinforcement learning by thinking with latent world model for autonomous driving (in carla-v2). In *Proc. of European Conference on Computer Vision*, pages 142–158, 2024a.

[^30]: Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. In *Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition*, pages 26763–26773, 2024b.

[^31]: Zhiqi Li, Zhiding Yu, Shiyi Lan, Jiahan Li, Jan Kautz, Tong Lu, and Jose M Alvarez. Is ego status all you need for open-loop end-to-end autonomous driving? In *Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition*, pages 14864–14873, 2024c.

[^32]: Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang, Qian Zhang, et al. Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving. In *Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition*, 2025.

[^33]: Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In *Proc. Annual Meeting of the Association for Computational Linguistics Workshop*, pages 74–81, 2004.

[^34]: Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In *Porc. of IEEE Intl. Conf. on Computer Vision*, pages 2980–2988, 2017.

[^35]: Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In *Proc. of Advances in Neural Information Processing Systems*, 2023.

[^36]: Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024a.

[^37]: Yingfei Liu, Tiancai Wang, Xiangyu Zhang, and Jian Sun. Petr: Position embedding transformation for multi-view 3d object detection. In *Proc. of European Conference on Computer Vision*, pages 531–548, 2022.

[^38]: Yixin Liu, Kai Zhang, Yuan Li, Zhiling Yan, Chujie Gao, Ruoxi Chen, Zhengqing Yuan, Yue Huang, Hanchi Sun, Jianfeng Gao, et al. Sora: A review on background, technology, limitations, and opportunities of large vision models. *arXiv preprint arXiv:2402.17177*, 2024b.

[^39]: Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, et al. Deepseek-vl: towards real-world vision-language understanding. *arXiv preprint arXiv:2403.05525*, 2024.

[^40]: Jianbiao Mei, Yukai Ma, Xuemeng Yang, Licheng Wen, Xinyu Cai, Xin Li, Daocheng Fu, Bo Zhang, Pinlong Cai, Min Dou, et al. Continuously learning, adapting, and improving: A dual-process approach to autonomous driving. In *Proc. of Advances in Neural Information Processing Systems*, 2024.

[^41]: Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In *Proc. Annual Meeting of the Association for Computational Linguistics*, pages 311–318, 2002.

[^42]: Shuai Peng, Ke Yuan, Liangcai Gao, and Zhi Tang. Mathbert: A pre-trained model for mathematical formula understanding. *arXiv preprint arXiv:2105.00377*, 2021.

[^43]: Jonah Philion and Sanja Fidler. Lift, splat, shoot: Encoding images from arbitrary camera rigs by implicitly unprojecting to 3d. In *Proc. of European Conference on Computer Vision*, pages 194–210, 2020.

[^44]: Aditya Prakash, Kashyap Chitta, and Andreas Geiger. Multi-modal fusion transformer for end-to-end autonomous driving. In *Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition*, pages 7077–7087, 2021.

[^45]: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *Proc. of Intl. Conf. on Machine Learning*, pages 8748–8763, 2021.

[^46]: Katrin Renz, Long Chen, Ana-Maria Marcu, Jan Hünermann, Benoit Hanotte, Alice Karnsund, Jamie Shotton, Elahe Arani, and Oleg Sinavski. Carllava: Vision language models for camera-only closed-loop driving. *arXiv preprint arXiv:2406.10165*, 2024.

[^47]: Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In *Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition*, pages 10684–10695, 2022.

[^48]: Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In *Proc. of Intl. Conf. on Medical Image Computing and Computer Assisted Intervention*, pages 234–241, 2015.

[^49]: Hao Shao, Yuxuan Hu, Letian Wang, Guanglu Song, Steven L Waslander, Yu Liu, and Hongsheng Li. Lmdrive: Closed-loop end-to-end driving with large language models. In *Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition*, pages 15120–15130, 2024.

[^50]: Shaoshuai Shi, Li Jiang, Dengxin Dai, and Bernt Schiele. Motion transformer with global intention localization and local movement refinement. *Proc. of Advances in Neural Information Processing Systems*, 35:6531–6543, 2022.

[^51]: Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, et al. Moviechat: From dense token to sparse memory for long video understanding. In *Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition*, pages 18221–18232, 2024.

[^52]: Ziying Song, Caiyan Jia, Lin Liu, Hongyu Pan, Yongchang Zhang, Junming Wang, Xingyu Zhang, Shaoqing Xu, Lei Yang, and Yadan Luo. Don’t shake the wheel: Momentum-aware planning in end-to-end autonomous driving. In *Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition*, 2025.

[^53]: Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al. Scalability in perception for autonomous driving: Waymo open dataset. In *Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition*, pages 2446–2454, 2020.

[^54]: Xiaoyu Tian, Junru Gu, Bailin Li, Yicheng Liu, Yang Wang, Zhiyong Zhao, Kun Zhan, Peng Jia, Xianpeng Lang, and Hang Zhao. Drivevlm: The convergence of autonomous driving and large vision-language models. *arXiv preprint arXiv:2402.12289*, 2024.

[^55]: Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*, 2023.

[^56]: Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In *Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition*, pages 4566–4575, 2015.

[^57]: Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. *arXiv preprint arXiv:2409.12191*, 2024a.

[^58]: Shihao Wang, Yingfei Liu, Tiancai Wang, Ying Li, and Xiangyu Zhang. Exploring object-centric temporal modeling for efficient multi-view 3d object detection. In *Porc. of IEEE Intl. Conf. on Computer Vision*, pages 3621–3631, 2023a.

[^59]: Shihao Wang, Zhiding Yu, Xiaohui Jiang, Shiyi Lan, Min Shi, Nadine Chang, Jan Kautz, Ying Li, and Jose M Alvarez. Omnidrive: A holistic llm-agent framework for autonomous driving with 3d perception, reasoning and planning. In *Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition*, 2024b.

[^60]: Wenhai Wang, Jiangwei Xie, ChuanYang Hu, Haoming Zou, Jianan Fan, Wenwen Tong, Yang Wen, Silei Wu, Hanming Deng, Zhiqi Li, et al. Drivemlm: Aligning multi-modal large language models with behavioral planning states for autonomous driving. *arXiv preprint arXiv:2312.09245*, 2023b.

[^61]: Penghao Wu, Xiaosong Jia, Li Chen, Junchi Yan, Hongyang Li, and Yu Qiao. Trajectory-guided control prediction for end-to-end autonomous driving: A simple yet strong baseline. In *Proc. of Advances in Neural Information Processing Systems*, 2022.

[^62]: Shuo Xing, Chengyuan Qian, Yuping Wang, Hongyuan Hua, Kexin Tian, Yang Zhou, and Zhengzhong Tu. Openemma: Open-source multimodal model for end-to-end autonomous driving. In *Proc. of IEEE Winter Conf. on Applications of Computer Vision*, pages 1001–1009, 2025.

[^63]: Zhenhua Xu, Yujia Zhang, Enze Xie, Zhen Zhao, Yong Guo, Kwan-Yee K Wong, Zhenguo Li, and Hengshuang Zhao. Drivegpt4: Interpretable end-to-end autonomous driving via large language model. *IEEE Robotics and Automation Letters*, 2024.

[^64]: Jiang-Tian Zhai, Ze Feng, Jinhao Du, Yongqiang Mao, Jiang-Jiang Liu, Zichang Tan, Yifu Zhang, Xiaoqing Ye, and Jingdong Wang. Rethinking the open-loop evaluation of end-to-end autonomous driving in nuscenes. *arXiv preprint arXiv:2305.10430*, 2023a.

[^65]: Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In *Porc. of IEEE Intl. Conf. on Computer Vision*, pages 11975–11986, 2023b.

[^66]: Diankun Zhang, Zhijie Zheng, Haoyu Niu, Xueqing Wang, and Xiaojun Liu. Fully sparse transformer 3-d detector for lidar point cloud. *IEEE Transactions on Geoscience and Remote Sensing*, 61:1–12, 2023.

[^67]: Diankun Zhang, Guoan Wang, Runwen Zhu, Jianbo Zhao, Xiwu Chen, Siyu Zhang, Jiahao Gong, Qibin Zhou, Wenyuan Zhang, Ningzi Wang, et al. Sparsead: Sparse query-centric paradigm for efficient end-to-end autonomous driving. *arXiv preprint arXiv:2404.06892*, 2024.

[^68]: Zhejun Zhang, Alexander Liniger, Dengxin Dai, Fisher Yu, and Luc Van Gool. End-to-end urban driving by imitating a reinforcement learning coach. In *Porc. of IEEE Intl. Conf. on Computer Vision*, 2021.

[^69]: Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. *Proc. of Advances in Neural Information Processing Systems*, 36:46595–46623, 2023.

[^70]: Wenzhao Zheng, Ruiqi Song, Xianda Guo, Chenming Zhang, and Long Chen. Genad: Generative end-to-end autonomous driving. In *Proc. of European Conference on Computer Vision*, pages 87–104, 2024.

[^71]: Julian Zimmerlin, Jens Beißwenger, Bernhard Jaeger, Andreas Geiger, and Kashyap Chitta. Hidden biases of end-to-end driving datasets. *arXiv preprint arXiv:2412.09602*, 2024.