---
title: "LVDrive: Latent Visual Representation Enhanced Vision-Language-Action Autonomous Driving Model"
source: "https://arxiv.org/html/2605.22089v1"
author:
published:
created: 2026-10-19
description:
tags:
  - "clippings"
---
Xiaodong Mei       Hongwei Xie Affiliation: Xiaomi EV    Guang Chen Affiliation: Xiaomi EV    Hangjun Ye Affiliation: Xiaomi EV    Dan Xu Affiliation: The Hong Kong University of Science and Technology

###### Abstract

Vision-Language-Action (VLA) models have emerged as a promising framework for end-to-end autonomous driving. However, existing VLAs typically rely on sparse action supervision, which underutilizes their powerful scene understanding and reasoning capabilities. Recent attempts to incorporate dense visual supervision via world modeling often overemphasize pixel-level image reconstruction, neglecting semantically meaningful scene representation learning. In this work, we propose LVDrive, a Latent Visual representation enhanced VLA framework for autonomous driving. LVDrive introduces a future scene prediction task into the VLA paradigm, where future representations are learned entirely in a high-level latent space under auxiliary supervision from a pretrained vision backbone. Departing from inefficient autoregressive generation, we jointly model future scene and motion prediction within a unified embedding space, processed in a single forward pass to conduct the future-aware reasoning. We further design a two-stage trajectory decoding strategy that explicitly leverages the learned latent future representations to refine trajectory generation. Extensive experiments on the challenging Bench2Drive benchmark demonstrate that LVDrive achieves significant improvements in closed-loop driving performance, outperforming both action supervised methods and image-reconstruction-based world model approaches.

![[nips_intro_0.png|Refer to caption]]

Figure 1: The comparison of different VLA paradigms. Standard VLA approaches, as depicted in (a), rely on sparse action supervision. Our LVDrive, illustrated in (c), performs the future visual and action representation learning jointly. Unlike VLA with the world modeling paradigm in (b), LVDrive predicts future scenes entirely in latent space, capturing rich semantic features without pixel-level reconstruction. The trajectory is then generated and explicitly refined through direct interaction with the learned future scene representations.

## 1 Introduction

Leveraging the powerful scene understanding and reasoning abilities of multimodal foundation models, Vision-Language-Action (VLA) models [^19] [^54] [^21] [^41] [^32] [^28] hold immense potential for the end-to-end autonomous driving task, which can directly map raw sensor inputs and language instructions into future trajectories. Current VLA models, however, suffer from a critical supervision deficit [^31] [^22]: the exclusive reliance on sparse action labels leaves the structured spatial scene understanding and reasoning capacity of large foundation models substantially underutilized.

To enhance the spatial scene understanding capacity of large foundation models, recent frontiers have explored incorporating high-dimensional representations as auxiliary objectives, such as future visual feature prediction. Dense visual signals in the driving scenario exhibit a more natural alignment with ego-vehicle trajectories than language guidance alone, thereby grounding motion planning in the future-aware reasoning. Inspired by the world modeling paradigm, existing approaches [^58] [^52] [^31] [^22] [^65] learn to anticipate future visual observations for modeling environmental dynamics, thereby internalizing causal relationships that enhance planning performance.

Despite several efforts to exploit dense scene features, for end-to-end autonomous driving models with the ultimate objective of reliable motion planning, three persistent issues remain unaddressed in existing methods. First, for a planning-oriented autonomous driving model, scene understanding should prioritize semantically rich representation learning over pixel-perfect frame reconstruction. Overemphasizing texture-level reconstruction risks misleading the representation learning objectives, diverting the model’s capacity away from high-level semantic understanding [^3] [^2]. Second, formulating future frame and trajectory prediction as the next-token generation task with the discrete visual and action vocabulary is inherently time-consuming. The autoregressive formulation typically requires a large quantity of tokens to synthesize high-fidelity images and precise actions, which incurs prohibitive inference costs [^52] [^63] [^31] [^59]. Third, future visual features are consistently underutilized in the motion planning pipeline. Existing frameworks either confine future visual representations to implicit interactions with motion features [^52] [^58], or employ future frames as a rudimentary world model to generate reward signals for trajectory optimization [^22]. Neither design is an explicit mechanism that embeds the rich semantic features directly into the trajectory generation.

To mitigate these challenges, we propose LVDrive, a framework that enhances VLA-based autonomous driving models with Latent Visual representation learning, shown in Figure 1. By leveraging dense future image frames as an additional supervision signal, our approach introduces a future scene prediction task into the VLA architecture. Crucially, future scene representation is learned entirely within a latent space, without relying on texture-level reconstruction or hand-crafted annotations, so that the model is focused on capturing high-level semantic cues essential for motion planning. Instead of autoregressively generating the discrete tokens, future scene features are learned alongside the trajectory decoding within the shared continuous optimization space in a single forward process, which naturally enhances the alignment between vision and action space. With the future-aware reasoning, semantic scene features are gradually and implicitly injected into motion features, yielding a coarse predicted trajectory. To fully unlock the benefits of predictive visual features, we further design a lightweight trajectory refiner that explicitly conditions on the learned future semantics to perform a fine-grained optimization of the trajectory decoding. The two-stage design ensures that future scene information is not merely an auxiliary signal but an integral condition of the trajectory generation, leading to consistent improvements in closed-loop driving performance.

Our contributions are summarized as follows:

- We introduce the latent future representation learning into the VLA-based autonomous driving framework, by jointly predicting the visual and action features in a single forward pass within a unified continuous space, to conduct the future-aware scene understanding and reasoning. With the utilization of semantically rich visual features as auxiliary supervision, our approach strengthens the spatial scene understanding and reasoning capability of VLA model, which yields efficient and effective planning.
- We design the two-stage trajectory decoding strategy to explicitly injects the learned future scene features into the motion generation. Specifically, the coarse trajectory proposal is decoded after the future-aware reasoning, and further refined in a lightweight trajectory refiner module that conditions on the predicted semantic features, producing the more feasible and reasonable final trajectory.
- Extensive experiments on the challenging Bench2Drive benchmark demonstrate that our method significantly improves the closed-loop driving performance, validating the effectiveness of dense future visual supervision exploitation in motion planning.

## 2 Related Work

#### VLAs for Autonomous Driving.

Vision-Language-Action (VLA) models, built upon large multi-modal foundation models [^35] [^4] [^5], have rapidly gained prominence in autonomous driving. Early approaches [^54] [^7] [^28] [^47] [^49] primarily employed VLMs as high-level semantic interpreters, generating scene descriptions and textual guidance to inform downstream planning performance. More recent efforts have shifted toward end-to-end VLA frameworks [^11] [^41] [^67] [^66] [^42] [^27] [^21] [^56] [^58] [^63] [^32] [^40] [^38] [^37] [^33] that directly map raw sensor inputs and language instructions to driving actions, bypassing intermediate textual representations. In the unified framework, actions are typically represented either as meta-action tokens drawn from the discrete vocabulary [^67] [^12] [^46] or as continuous motion features [^11] [^56] [^32] [^40] that are subsequently decoded into the ego-trajectory through the action decoder. Furthermore, the integration of reinforcement learning has recently been explored to boost the closed-loop performance of VLA driving models [^12] [^32] [^40]. Previous works [^31] [^22] identify that the reliance of current VLAs on sparse action supervision and abstract language guidance underutilizes the spatial understanding capabilities of large multi-modal models. To this end, we propose a joint representation learning framework to incorporates auxiliary dense vision supervisions.

#### World Models for Autonomous Driving.

World models, which aim to understand and model environmental dynamics, have emerged as a powerful paradigm for autonomous driving. One line of research explores high-fidelity scene generation across various perceptual modalities, including images [^17] [^59] [^13] [^15] and 3D representations [^53] [^55] [^60], in order to construct the realistic simulator. Another line seeks to integrate world modeling into end-to-end driving frameworks as a multi-task objective [^31] [^52] [^63] [^36], aiming to jointly enhance scene understanding and motion planning. We argue that in this formulation, what matters for autonomous driving is not the fidelity of image synthesis, but rather the underlying world dynamics captured through predictive modeling, termed as latent world modeling [^14]. Several prior works [^30] [^64] [^34] have explored leveraging latent world models within traditional end-to-end planning architectures. Our work extends this idea to large VLA driving models, enabling the thorough and efficient exploitation of the reasoning capabilities inherent in large multi-modal foundation models.

## 3 Methodology

In this section, we present LVDrive, a VLA framework that jointly learns the future visual and action representations with the auxiliary semantic supervision signals from the pretrained vision backbone. The model architecture is presented in Figure 2. We incorporate the future scene prediction alongside the planning to generate the latent visual features in a single-forward process. The aligned vision and action representations initially enable the feasible trajectory generation. We further design the two-stage decoding strategy to produce the fine-grained trajectory conditioned on the latent vision feature explicitly.

### 3.1 Problem Formulation

Given a driving scenario at timestep $t$, we follow the recent end-to-end VLA model [^11], where structured and historical multi-view image features $x_{s},x_{h}$ are compressed and extracted through the dedicated vision encoder, and then projected with textual instructions $x_{q}$ into the unified token embedding space. To generate future continuous motion features in the time horizon $F$, the core LLM performs scene understanding and reasoning to produce the special planning token $s$, the embedding of which is subsequently decoded into the ego trajectory.

We extend the task to predict the latent future visual features $V_{t+1:t+F}$ jointly with the planning, to enforce the model to conduct the future-aware reasoning. Motivated by the fact that human drivers attend primarily to the forward perspective, front-view prediction is sufficient for providing adequate future awareness. Notably, the model relies solely on structured multi-view representations to predict future scene evolution. This design effectively eliminates the potential noise introduced by texture-level details, allowing it to focus on learning semantically meaningful scene dynamics. The whole task is formally formulated as

$$
\displaystyle V_{t+1:t+F},s\sim p(V_{t+1:t+F},s|x_{s},x_{h},x_{q}).
$$
![[nips_model_0.png|Refer to caption]]

Figure 2: Overview of LVDrive. LVDrive is a VLA framework that unifies latent future scene representation learning and motion planning, with dense auxiliary supervision provided by a pre-trained vision backbone. Given multi-view images, the model encodes current and historical scene features and performs future-aware reasoning to predict both latent visual representations and motion features in a single forward pass. The trajectory is then generated through a two-stage decoding process: a coarse proposal is first decoded from the planning embedding, and then refined by explicitly conditioning on the learned latent future scene features to produce the final trajectory.

### 3.2 Latent Future Visual Representation Learning

For the latent future visual feature prediction, we design a set of special latent visual placeholder tokens, termed <img\_i>, where $i\in\{0,1,...,N\}$ indicates the number of tokens for each future image frame. Special boundary markers are also designed as <img\_start> and <img\_end> to form the fixed output sequence of every future frame. The sequence is repeated $F$ times to get multiple future scene predictions.

During the future-aware reasoning process, the model first locates all occurrences of latent visual placeholders in the outcomes, and divides them into multiple future frames at different timesteps according to the boundary markers. Then the embedding of per-frame $H_{t+j}\in\mathbb{R}^{N\times D}$ is obtained in each placeholder span from the hidden states of the last LLM layer, where $D$ is the dimension of LLM and $j\in\{1,...,F\}$ denotes the number of future frames. A lightweight vision decoder $\textsc{ViS}_{\theta}$ is subsequently applied to extract the semantic future features $V_{t+j}$:

$$
\displaystyle V_{t+j}=\textsc{ViS}_{\theta}(H_{t+j}),V_{t+j}\in\mathbb{R}^{N\times C_{v}}
$$

where $C_{v}$ is the dimension of latent visual feature. The semantic supervision signals are derived from an independently pre-trained vision foundation model, which is maintained frozen in the training phase. Its well-established scene understanding ability supplies the model with semantically meaningful guidance beyond the limited action labels and texture-level image reconstruction, contributing to more effective representation learning.

The special planning token <waypoint\_ego> is positioned after the latent visual sequence, allowing future scene representations to be implicitly aggregated into motion planning embedding via the causal attention mechanism. The planning embedding $H_{p}\in\mathbb{R}^{1\times D}$ is further decoded into the ego-trajectory through the two-stage trajectory decoding strategy, which is explicitly conditioned on latent future features.

### 3.3 Two-stage Trajectory Decoding

The ego-trajectory generation process are divided into two stages: to predict the coarse trajectory proposal $\tau_{c}\in\mathbb{R}^{K\times F\times 2}$ and then to generate the fine-grained final trajectory $\tau_{f}\in\mathbb{R}^{K\times F\times 2}$, with K indicating the number of multi-modal trajectories.

In the first stage, we utilize the VAE-based generative planner to produce the coarse proposals, following the designs in VAD [^29], GenAD [^62] and ORION [^11]. Specifically, the planning embedding $H_{p}$ is mapped to the Gaussian distribution $p(z|H_{p})\sim N(\mu,\sigma^{2})$ to represent the future motion features with the distribution generator $\textsc{DiS}_{\theta}$, which is supervised by the ground-truth trajectory during training. The sampled latent variable $z$ from the distribution, together with the planning embedding which carries the current motion features, is decoded with the GRU-based state decoder $\textsc{State}_{\theta}$ to generate the future motion states $s_{ego}$ across all timesteps. This resulting ego states $s_{ego}$ are subsequently converted into $K$ coarse trajectory proposals with MLPs.

$$
\displaystyle z\sim p(\mu,\sigma^{2});(\mu,\sigma^{2})=\textsc{DiS}_{\theta}(H_{p}),z\in\mathbb{R}^{C_{z}}
$$
 
$$
\displaystyle s_{ego}=\textsc{StAtE}_{\theta}(z,H_{p}),s_{ego}\in\mathbb{R}^{F\times 2D}
$$

where $C_{z}$ is the latent embedding dimension of the Gaussian distribution.

We further project the future ego motion states to the ego motion queries $Q_{ego}$ in the second decoding stage. The trajectory refiner module $\textsc{TrajRefiner}_{\theta}$ is composed of stacked transformer blocks, where cross-attention is applied to extract future semantics from the predicted visual embeddings. We transform the high-dimensional future visual embeddings $H_{t+j}$ instead of the direct utilization of predicted latent features $V_{t+j}$ to preserve more reasoning information from the LLM.

$$
\displaystyle Q_{ego}=\textsc{ProJ}_{\theta}(s_{ego}),Q_{ego}\in\mathbb{R}^{F\times C_{r}}
$$
 
$$
\displaystyle K_{fut},V_{fut}=\textsc{ProJ}_{\theta}(H_{t+1:t+F}),K_{fut},V_{fut}\in\mathbb{R}^{F\times C_{r}}
$$
 
$$
\displaystyle s_{ego}^{\ast}=\textsc{TrajRefiner}_{\theta}(Q_{ego},K_{fut},V_{fut}),s_{ego}^{\ast}\in\mathbb{R}^{F\times C_{r}}
$$

where $C_{r}$ is the cross-attention dimension. The refined motion feature $s_{ego}^{\ast}$ is then decoded into $K$ base trajectories and $K$ offsets through two separate MLP blocks to enhance the generation flexibility and diversity. Finally, the combination of the base trajectory and the offset forms the fine-grained trajectory $\tau_{f}$.

### 3.4 Training Objectives

LVDrive is trained in an end-to-end manner with a combination of four kinds of loss terms, eliminating the need for a complex multi-stage training process.

For the proposed latent future visual feature prediction task, we employ the weighted sum of the frame-wise cosine similarity loss and the L1 loss. The future visual prediction loss is defined as $\mathcal{L}_{vis}=w_{c}\mathcal{L}_{cosine}+w_{r}\mathcal{L}_{reg}$. For the two-stage trajectory planning, we adopt the same regression loss $\mathcal{L}_{mse}$, boundary loss $\mathcal{L}_{bd}$ and collision loss $\mathcal{L}_{col}$ for both coarse trajectory proposal and final trajectory, termed as $\mathcal{L}_{plan}$ and $\mathcal{L}_{plan\_r}$. We empirically omit the KL-divergence loss in VAE objectives, as its regularization might overly constrain the multi-modal latent feature space and hinders the learning ability.

We also follow the previous work to leverage the training loss $\mathcal{L}_{qt}$ of the structured multi-view feature extraction. And we utilize the cross-entropy loss $\mathcal{L}_{ce}$ for the LLM to generate the fixed special placeholder tokens. So that the overall loss is summarized as follows:

$$
\displaystyle\mathcal{L}=\mathcal{L}_{vis}+\mathcal{L}_{plan}+\mathcal{L}_{plan\_r}+\mathcal{L}_{qt}+\mathcal{L}_{ce}.
$$

## 4 Experiments

Table 1: Closed-loop and Open-loop Results of E2E-AD Methods in Bench2Drive under base training set. C/L refers to camera/LiDAR. Avg. L2 is averaged over the predictions in 3 seconds under 2Hz, similar to UniAD. \* denote expert feature distillation. IL: Imitation Learning, RL: Reinforcement Learning, DS: Driving Score, SR: Success Rate.

<table><tbody><tr><th>Method</th><td rowspan="2">Scheme</td><td rowspan="2">Modality</td><td colspan="4">Closed-loop Metric</td><td>Open-loop Metric</td></tr><tr><th></th><td>DS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>SR(%) <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Efficiency <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Comfortness <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Avg. L2 <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><th colspan="8">Traditional End-to-End Paradigm</th></tr><tr><th>TCP* <sup><a href="#fn:51">51</a></sup></th><td>IL</td><td>C</td><td>40.70</td><td>15.00</td><td>54.26</td><td>47.80</td><td>1.70</td></tr><tr><th>VAD <sup><a href="#fn:29">29</a></sup></th><td>IL</td><td>C</td><td>42.35</td><td>15.00</td><td>157.94</td><td>46.01</td><td>0.91</td></tr><tr><th>MomAD <sup><a href="#fn:45">45</a></sup></th><td>IL</td><td>C</td><td>44.54</td><td>16.71</td><td>170.21</td><td>48.63</td><td>0.87</td></tr><tr><th>UniAD-Base <sup><a href="#fn:20">20</a></sup></th><td>IL</td><td>C</td><td>45.81</td><td>16.36</td><td>129.21</td><td>43.58</td><td>0.73</td></tr><tr><th>TCP-traj* <sup><a href="#fn:51">51</a></sup></th><td>IL</td><td>C</td><td>59.90</td><td>30.00</td><td>76.54</td><td>18.08</td><td>1.70</td></tr><tr><th>DriveDPO <sup><a href="#fn:43">43</a></sup></th><td>RL</td><td>C</td><td>62.02</td><td>30.62</td><td>166.80</td><td>26.79</td><td>-</td></tr><tr><th>ThinkTwice* <sup><a href="#fn:24">24</a></sup></th><td>IL</td><td>C</td><td>62.44</td><td>31.23</td><td>69.33</td><td>16.22</td><td>0.95</td></tr><tr><th>DriveTransformer-Large <sup><a href="#fn:26">26</a></sup></th><td>IL</td><td>C</td><td>63.46</td><td>35.01</td><td>100.64</td><td>20.78</td><td>0.62</td></tr><tr><th>DriveAdapter* <sup><a href="#fn:23">23</a></sup></th><td>IL</td><td>C&L</td><td>64.22</td><td>33.08</td><td>70.22</td><td>16.01</td><td>1.01</td></tr><tr><th>DiffAD <sup><a href="#fn:48">48</a></sup></th><td>IL</td><td>C</td><td>67.92</td><td>38.64</td><td>-</td><td>-</td><td>1.55</td></tr><tr><th>Raw2Drive <sup><a href="#fn:57">57</a></sup></th><td>RL</td><td>C</td><td>71.36</td><td>50.24</td><td>214.17</td><td>22.42</td><td>-</td></tr><tr><th colspan="8">Vision-Language-Action Paradigm</th></tr><tr><th>Drive <math><semantics><msub><mi>π</mi> <mn>0</mn></msub> <annotation>\pi_{0}</annotation></semantics></math> <sup><a href="#fn:56">56</a></sup></th><td>IL</td><td>C</td><td>60.45</td><td>30.00</td><td>168.41</td><td>14.88</td><td>0.56</td></tr><tr><th>ReCogDrive <sup><a href="#fn:32">32</a></sup></th><td>RL</td><td>C</td><td>71.36</td><td>45.45</td><td>138.18</td><td>17.45</td><td>-</td></tr><tr><th>DriveMoE <sup><a href="#fn:56">56</a></sup></th><td>IL</td><td>C</td><td>74.22</td><td>48.64</td><td>175.96</td><td>15.31</td><td>0.38</td></tr><tr><th>ORION <sup><a href="#fn:11">11</a></sup></th><td>IL</td><td>C</td><td>77.74</td><td>54.62</td><td>151.48</td><td>17.38</td><td>0.68</td></tr><tr><th>MindDrive <sup><a href="#fn:12">12</a></sup></th><td>RL</td><td>C</td><td>78.04</td><td>55.09</td><td>-</td><td>-</td><td>-</td></tr><tr><th>UniDrive-WM-AR <sup><a href="#fn:52">52</a></sup></th><td>IL</td><td>C</td><td>79.22</td><td>56.36</td><td>158.44</td><td>28.01</td><td>0.64</td></tr><tr><th>UniDrive-WM-AR+Diff <sup><a href="#fn:52">52</a></sup></th><td>IL</td><td>C</td><td>79.31</td><td>56.42</td><td>158.65</td><td>27.93</td><td>0.63</td></tr><tr><th>LVDrive (Ours)</th><td>IL</td><td>C</td><td>80.71</td><td>58.26</td><td>155.77</td><td>14.34</td><td>0.63</td></tr></tbody></table>

### 4.1 Dataset and Evaluation Metrics

#### Dataset.

We conduct experiments on Bench2Drive dataset [^25], a closed-loop evaluation benchmark built on CARLA v2 simulator [^8] for end-to-end autonomous driving. For the fair comparison, we adopt the official base training split, which contains 1000 clips uniformly distributed across 44 interactive scenes, 23 weathers and 12 towns. And we follow the standard split of 950 clips for training and 50 for open-loop validation. Closed-loop performance is evaluated on the official evaluation protocol, which encompasses 220 short routes. These routes cover all 44 scenarios with five distinct route instances per scenario under varying weather and town configurations, enabling a comprehensive assessment of a diverse set of challenging driving scenarios and skills. Additionally, for efficient ablation studies during the model development, we further utilize Dev10 [^26], a compact evaluation subset officially provided, which consists of 10 representative routes.

#### Evaluation metrics.

Bench2Drive officially employs five closed-loop evaluation metrics: Driving Score (DS), Success Rate (SR), Efficiency, Comfort and Multi-Ability. SR quantifies the proportion of routes successfully completed within designated time limit. DS incorporates both route completion status and violation penalties, where infractions proportionally reduce the score through discount factors. Efficiency and Comfort respectively assess the speed performance and driving smoothness, and Multi-Ability provides a disentangled assessment of five advanced urban driving skills. We adopt L2 trajectory error in open-loop evaluation, as used in motion planning [^33] [^6] [^39].

### 4.2 Implementation Details

#### Model Settings.

For the latent future prediction, we set N as 256, indicating that each future feature frame is represented by 256 tokens. Ground-truth latent visual features in 6 future timesteps are provided by the pre-trained VQGAN-ImageNet [^9] with the codebooksize as 16384. We utilize the quantized code embeddings instead of the discrete code indices as the visual supervision signal. As for the planning task, 6 multi-modal trajectories are generated corresponding to the 6 navigation commands defined in Bench2Drive. Additional implementation details are provided in the Appendix.

#### Training and Evaluation Process.

Our experiments are conducted on 32 NVIDIA H20 GPUs with 96 GB of memory. We employ EVA-02-L [^10] and QT-Former [^11] as the vision encoder. Vicuna v1.5 [^61] is utilized as the core LLM and fine-tuned with LoRA [^18]. We adopt the pre-trained weights in the first stage of ORION [^11], which is trained on VQA pairs from Chat-B2D to align the vision space with the reasoning space. Our LVdrive is trained in an end-to-end manner for 6 epochs, eliminating the complex multi-stage training process. In the evaluation, the fixed visual placeholder and planning placeholder token sequence is pre-filled, so that the predicted future scene and action are obtained in a single forward process with parallel decoding instead of the autoregressive decoding, improving the inference efficiency significantly.

### 4.3 Main Results

Table 1 summarizes the closed-loop evaluation results on the Bench2Drive benchmark. Our LVDrive consistently outperforms all existing approaches with 80.71 Driving Score and 58.26% Success Rate, including both conventional end-to-end planners and VLA-based methods, under both imitation learning and reinforcement learning paradigms. Notably, a particularly instructive comparison is with UniDrive-WM [^52], which employs autoregressive generation of texture-level future images as an auxiliary task and relies on a complex multi-stage training pipeline. The substantial improvement of LVDrive over UniDrive-WM highlights the advantage of our latent future representation learning. By learning in a latent space rather than pursuing high-fidelity pixel reconstruction, our method captures semantically richer scene features that are more directly beneficial for motion planning.

Table 2: Multi-Ability Results of E2E-AD Methods under base training set. \* denote expert feature distillation. C/L refers to camera/LiDAR. IL: Imitation Learning, RL: Reinforcement Learning.

<table><tbody><tr><th>Method</th><td rowspan="2">Scheme</td><td rowspan="2">Modality</td><td colspan="6">Ability (%) <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><th></th><td>Merging</td><td>Overtaking</td><td>Emergency Brake</td><td>Give Way</td><td>Traffic Sign</td><td>Mean</td></tr><tr><th colspan="8">Traditional End-to-End Paradigm</th><td></td></tr><tr><th>TCP* <sup><a href="#fn:51">51</a></sup></th><td>IL</td><td>C</td><td>16.18</td><td>20.00</td><td>20.00</td><td>10.00</td><td>6.99</td><td>14.63</td></tr><tr><th>UniAD-Base <sup><a href="#fn:20">20</a></sup></th><td>IL</td><td>C</td><td>14.10</td><td>17.78</td><td>21.67</td><td>10.00</td><td>14.21</td><td>15.55</td></tr><tr><th>VAD <sup><a href="#fn:29">29</a></sup></th><td>IL</td><td>C</td><td>8.11</td><td>24.44</td><td>18.64</td><td>20.00</td><td>19.15</td><td>18.07</td></tr><tr><th>TCP-traj* <sup><a href="#fn:51">51</a></sup></th><td>IL</td><td>C</td><td>8.89</td><td>24.29</td><td>51.67</td><td>40.00</td><td>46.28</td><td>34.22</td></tr><tr><th>ThinkTwice* <sup><a href="#fn:24">24</a></sup></th><td>IL</td><td>C</td><td>27.38</td><td>18.42</td><td>35.82</td><td>50.00</td><td>54.23</td><td>37.17</td></tr><tr><th>DriveTransformer-Large <sup><a href="#fn:26">26</a></sup></th><td>IL</td><td>C</td><td>17.57</td><td>35.00</td><td>48.36</td><td>40.00</td><td>52.10</td><td>38.60</td></tr><tr><th>DiffAD <sup><a href="#fn:48">48</a></sup></th><td>IL</td><td>C</td><td>30.00</td><td>35.55</td><td>46.66</td><td>40.00</td><td>46.32</td><td>38.79</td></tr><tr><th>DriveAdapter* <sup><a href="#fn:23">23</a></sup></th><td>IL</td><td>C&L</td><td>28.82</td><td>26.38</td><td>48.76</td><td>50.00</td><td>56.43</td><td>42.08</td></tr><tr><th>Raw2Drive <sup><a href="#fn:57">57</a></sup></th><td>RL</td><td>C</td><td>43.35</td><td>51.11</td><td>60.00</td><td>50.00</td><td>62.26</td><td>53.34</td></tr><tr><th colspan="8">Vision-Language-Action Paradigm</th><td></td></tr><tr><th>Drive <math><semantics><msub><mi>π</mi> <mn>0</mn></msub> <annotation>\pi_{0}</annotation></semantics></math> <sup><a href="#fn:56">56</a></sup></th><td>IL</td><td>C</td><td>29.35</td><td>36.58</td><td>48.83</td><td>40.00</td><td>54.45</td><td>41.84</td></tr><tr><th>DriveMoE <sup><a href="#fn:56">56</a></sup></th><td>IL</td><td>C</td><td>34.67</td><td>40.00</td><td>65.45</td><td>40.00</td><td>59.44</td><td>47.91</td></tr><tr><th>ReCogDrive <sup><a href="#fn:32">32</a></sup></th><td>RL</td><td>C</td><td>29.73</td><td>20.00</td><td>69.09</td><td>20.00</td><td>71.34</td><td>42.03</td></tr><tr><th>ORION <sup><a href="#fn:11">11</a></sup></th><td>IL</td><td>C</td><td>25.00</td><td>71.11</td><td>78.33</td><td>30.00</td><td>69.15</td><td>54.72</td></tr><tr><th>MindDrive <sup><a href="#fn:12">12</a></sup></th><td>RL</td><td>C</td><td>32.89</td><td>75.56</td><td>68.33</td><td>50.00</td><td>57.89</td><td>56.94</td></tr><tr><th>UniDrive-WM-AR <sup><a href="#fn:52">52</a></sup></th><td>IL</td><td>C</td><td>28.81</td><td>74.04</td><td>79.84</td><td>40.00</td><td>71.30</td><td>59.00</td></tr><tr><th>UniDrive-WM-AR+Diff <sup><a href="#fn:52">52</a></sup></th><td>IL</td><td>C</td><td>29.97</td><td>74.64</td><td>79.98</td><td>40.00</td><td>71.54</td><td>59.23</td></tr><tr><th>LVDrive (Ours)</th><td>IL</td><td>C</td><td>39.74</td><td>68.89</td><td>71.67</td><td>20.00</td><td>74.21</td><td>54.90</td></tr></tbody></table>

We further report the Multi-Ability evaluation results in Table 2. LVDrive achieves the highest Traffic Sign score of 74.21% and the second-best Merging score of 39.74%, together with competitive performance on Overtaking and Emergency Brake. These results demonstrate the strong scene understanding and reasoning capability enabled by latent scene representation learning. Since most evaluation scenarios correspond to Traffic Sign and Merging, the gains in these two skills largely account for the superior DS and SR. For the Give Way skill, where the ego vehicle is typically required to yield to an emergency vehicle approaching from behind, LVDrive often fails to complete the maneuver. This failure is likely due to the incorporation of the front-view future scene prediction task, which leads the model’s attention predominantly toward front-view visual cues and thereby limits its capacity to capture interactions occurring behind. More detailed analysis with qualitative results is provided in the Appendix A.4.

### 4.4 Ablation Study

For ablation studies, we utilize Dev10, a dedicated subset officially provided and recommended by Bench2Drive benchmark. It consists of 10 clips, each selected from a distinct representative scenario. The small-scale subset offers a diverse yet low-variance evaluation signal that prevents overfitting the full 220 testing routes during the model development.

Table 3: Ablation study on core components of LVDrive on Dev10 set. "Latent Vis." indicates the latent future frame prediction task. "One-stage Dec." and "Two-stage Dec." indicate the trajectory decoding strategy. DS: Driving Score; SR: Success Rate.

<table><thead><tr><th rowspan="2">Model Variants</th><th rowspan="2">Latent Vis.</th><th rowspan="2">One-stage Dec.</th><th rowspan="2">Two-stage Dec.</th><th colspan="2">Closed-loop</th></tr><tr><th>DS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>SR <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th></tr></thead><tbody><tr><td><math><semantics><msub><mi>ℳ</mi> <mrow><mi>b</mi> <mo></mo><mi>a</mi> <mo></mo><mi>s</mi> <mo></mo><mi>e</mi></mrow></msub> <annotation>\mathcal{M}_{base}</annotation></semantics></math></td><td></td><td></td><td></td><td>65.25</td><td>4/10</td></tr><tr><td><math><semantics><msub><mi>ℳ</mi> <mrow><mi>v</mi> <mo></mo><mi>i</mi> <mo></mo><mi>s</mi></mrow></msub> <annotation>\mathcal{M}_{vis}</annotation></semantics></math></td><td>✓</td><td></td><td></td><td>66.31</td><td>3/10</td></tr><tr><td><math><semantics><msub><mi>ℳ</mi> <mrow><mi>o</mi> <mo></mo><mi>n</mi> <mo></mo><mi>e</mi></mrow></msub> <annotation>\mathcal{M}_{one}</annotation></semantics></math></td><td>✓</td><td>✓</td><td></td><td>60.43</td><td>3/10</td></tr><tr><td>LVDrive (ours)</td><td>✓</td><td></td><td>✓</td><td>82.39</td><td>7/10</td></tr></tbody></table>

Table 4: Ablation study on different latent visual feature supervisions. "Vision Supervision" indicates the utilized pre-trained visual backbone. "Feature Dim." indicates the dimension of the ground-truth latent feature. DS: Driving Score; SR: Success Rate.

<table><thead><tr><th rowspan="2">Model Variants</th><th rowspan="2">Vision Supervision</th><th rowspan="2">Feature Dim.</th><th colspan="2">Closed-loop</th></tr><tr><th>DS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>SR <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th></tr></thead><tbody><tr><td><math><semantics><msub><mi>ℳ</mi> <mrow><mi>b</mi> <mo></mo><mi>a</mi> <mo></mo><mi>s</mi> <mo></mo><mi>e</mi></mrow></msub> <annotation>\mathcal{M}_{base}</annotation></semantics></math></td><td>-</td><td>-</td><td>65.25</td><td>4/10</td></tr><tr><td><math><semantics><msub><mi>ℳ</mi> <mn>1</mn></msub> <annotation>\mathcal{M}_{1}</annotation></semantics></math></td><td>Internal Vision Enc.</td><td>1024</td><td>65.42</td><td>4/10</td></tr><tr><td><math><semantics><msub><mi>ℳ</mi> <mn>2</mn></msub> <annotation>\mathcal{M}_{2}</annotation></semantics></math></td><td>MoVQGAN <sup><a href="#fn:50">50</a></sup></td><td>4</td><td>59.91</td><td>3/10</td></tr><tr><td><math><semantics><msub><mi>ℳ</mi> <mn>3</mn></msub> <annotation>\mathcal{M}_{3}</annotation></semantics></math></td><td>DINOv3-Large <sup><a href="#fn:44">44</a></sup></td><td>1024</td><td>71.72</td><td>5/10</td></tr><tr><td>LVDrive (ours)</td><td>VQGAN-ImageNet <sup><a href="#fn:9">9</a></sup></td><td>256</td><td>82.39</td><td>7/10</td></tr></tbody></table>

#### Effects of key components.

Table 4 reports the effectiveness of two key designs in LVDrive. The baseline model $\mathcal{M}_{base}$, presented in the first row, is trained to plan solely from action labels, analogous to the variants used in the second training phase of [^11]. We then directly incorporate the latent future scene prediction task as an auxiliary objective, denoted $\mathcal{M}_{vis}$. While the Driving Score improves, a decline in Success Rate is observed under this multi-task setting. We speculate that this degradation arises because motion planning and future scene prediction operate in distinct action and vision feature spaces that are not inherently aligned. Consequently, naively combining them introduces conflicting optimization signals. To address this misalignment, we first investigate a one-stage decoding variant $\mathcal{M}_{one}$. The planning embedding $H_{p}$ is directly used as the ego motion query to capture future features through the cross-attention mechanism and then fed into the VAE-base planner, constraining both tasks to optimize toward a shared objective. However, this direct integration negatively impacts motion feature learning, resulting in further performance decline. We therefore adopt a two-stage trajectory decoding strategy. This design first leverages action labels to stabilize the learned action features for decoding coarse proposals, and subsequently aggregates the learned future scene features to refine the final trajectory. With this design, LVDrive ultimately achieves significant and consistent performance gains.

#### Effects of various latent visual supervision signals.

The vision backbone responsible for supplying future scene supervision signals plays a crucial role in our framework. We conduct experiments with several other alternative backbones, as summarized in Table 4. As shown in $\mathcal{M}_{1}$, we first experiment with the internal vision encoder that processes multi-view image input. This yields a slight improvement in Driving Score but no gain in Success Rate. We attribute this to the fact that the encoder is jointly trained with multiple objectives during the training, limiting its visual representation capacity compared to the external large-scale pretrained backbones.

We then explore the pre-trained backbones. One candidate is the post-quantized code embedding from the MoVQGAN vision encoder of Emu3 [^50], the aim of which is to tokenize images into a discrete space. So the code embedding preserves the critical and highly compressed features to recover the original image. However, this variant leads to a performance decline shown in $\mathcal{M}_{2}$, likely due to the low feature dimension of the ground-truth future frame compared to the LLM’s hidden state, making it difficult to transfer sufficient information. In contrast, the features from DINOv3 [^44] bring clear gains in $\mathcal{M}_{3}$, confirming the effectiveness of latent future scene representation learning. Nevertheless, compared to VQGAN-ImageNet, the feature of DINOv3-Large is excessively rich, potentially introducing redundant information as noise into the shared vision-action feature space.

#### Effectiveness of two-stage trajectory decoding.

We further validate the effectiveness of the proposed two-stage trajectory decoding strategy in Table 6. Since both the coarse trajectory proposal and the final fine-grained trajectory are generated during inference, we evaluate the closed-loop performance of each to examine how future scene features are progressively aggregated through the two stages. The coarse trajectory in $\mathcal{M}_{coarse}$, which is directly decoded from the planning token, already achieves a substantial improvement over the baseline $\mathcal{M}_{base}$. This confirms that the planning embedding initially captures effective future scene representations through future-aware reasoning. The performance is further boosted when the motion feature is explicitly refined with latent visual embeddings in the second stage, leading to the final trajectory.

#### Efficiency of single forwarding process.

In contrast to autoregressive paradigms that generate visual and action tokens sequentially, LVDrive processes pre-filled placeholder tokens for latent visual features and actions in a single forward step, which in principle leads to higher inference efficiency. We report the measured inference speed in Table 6. Measurements are performed on the official Bench2Drive validation set using the NVIDIA H20 GPU with a batch size of one. LVDrive achieves an inference speed approximately twice that of $\mathcal{M}_{base}$, despite the overhead of the latent future scene prediction task. In comparison, a hypothetical autoregressive baseline $\mathcal{M}_{AR}$, which is configured to generate vision and action tokens of equivalent length, results in an order of magnitude slower, incurring prohibitive computational costs for the autonomous driving deployment.

### 4.5 Qualitative Results

![[nips_vis_2.png|Refer to caption]]

Figure 3: Qualitative results of our LVDrive and ℳ b a s e \\mathcal{M}\_{base} in an Overtaking scenario from Bench2Drive. The ego vehicle encounters an accident ahead that blocks its driving lane, while a steady stream of oncoming traffic occupies the adjacent lane. The l u \\color\[rgb\]{0,0,1}{blue} line denotes the generated trajectory. The ego vehicle controlled by becomes immobilized at the accident site. In contrast, our LVDrive successfully and smoothly overtakes the accident cars with the safe planning.

The qualitative results of LVDrive and $\mathcal{M}_{base}$ are presented in Figure 3. In this scene, the ego vehicle is tasked with navigating a challenging scenario, which demands highly precise planning. The ego vehicle encounters the accident ahead, with oncoming traffic continuously approaching from the opposite direction in the adjacent lane. $\mathcal{M}_{base}$ performs unsatisfactorily in this scenario, where the ego vehicle halts at the accident site and remains stranded there, unable to plan a drivable path. In contrast, enhanced by its future scene prediction capability, LVDrive successfully maneuvers past the accident cars and navigates the lane safely, undisturbed by the oncoming traffic. This comparison demonstrates that LVDrive possesses stronger scene understanding and reasoning abilities, yielding more robust and reliable planning performance. Additional results are provided in the Appendix A.4.

Table 5: Ablation study on two-stage decoding strategy. "Trajectory Type" indicates whether coarse proposal or fine-gained trajectory is utilized.

<table><thead><tr><th rowspan="2">Model Variants</th><th rowspan="2">Trajectory Type</th><th colspan="2">Closed-loop</th></tr><tr><th>DS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>SR <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th></tr></thead><tbody><tr><th><math><semantics><msub><mi>ℳ</mi> <mrow><mi>b</mi> <mo></mo><mi>a</mi> <mo></mo><mi>s</mi> <mo></mo><mi>e</mi></mrow></msub> <annotation>\mathcal{M}_{base}</annotation></semantics></math></th><td>-</td><td>65.25</td><td>4/10</td></tr><tr><th><math><semantics><msub><mi>ℳ</mi> <mrow><mi>c</mi> <mo></mo><mi>o</mi> <mo></mo><mi>a</mi> <mo></mo><mi>r</mi> <mo></mo><mi>s</mi> <mo></mo><mi>e</mi></mrow></msub> <annotation>\mathcal{M}_{coarse}</annotation></semantics></math></th><td>coarse proposal</td><td>73.22</td><td>5/10</td></tr><tr><th>LVDrive (ours)</th><td>fine-grained trajectory</td><td>82.39</td><td>7/10</td></tr></tbody></table>

Table 6: Comparison of inference speed with different model variants. The inference time is reported in seconds.

| Model Variants | Inference Time |
| --- | --- |
| $\mathcal{M}_{base}$ | 0.93s |
| $\mathcal{M}_{AR}$ | 36.62s |
| LVDrive (ours) | 2.03s |

## 5 Conclusion

In this work, we introduce LVDrive, a VLA framework enhanced with latent future visual representation learning, guided by auxiliary semantic supervision from a pretrained VQGAN model. LVDrive performs future-aware reasoning that jointly predicts future scene features and motion features in a single forward pass, efficiently using pre-filled placeholder tokens. The trajectory is generated through the dedicated two-stage decoding strategy, which is explicitly refined with the learned future visual representations. Extensive experiments on the challenging Bench2Drive benchmark demonstrate that LVDrive outperforms existing VLA frameworks, including those relying on sparse action supervision and those incorporating image-reconstruction-based world modeling.

#### Limitations.

One limitation of the current framework is that it primarily leverages the reasoning capability of the underlying language space, without explicitly investigating how language supervision could further enhance visual and action representation learning. Another limitation concerns the choice of the pre-trained vision backbone. Although LVDrive already achieves strong performance with general-purpose backbones, we believe that adopting vision foundation models specifically pre-trained on large-scale autonomous driving data could further boost planning performance.

## 6 Acknowledgment

#### Data usage statement.

This paper uses the external dataset resources. We utilize the dataset provided in the Bench2Drive project (source: https://github.com/Thinklab-SJTU/Bench2Drive), which is licensed under the CC BY-NC-ND 4.0. The authors confirm that the usage of the above data in this paper is strictly limited to academic research and has not involved any commercial activities.

## References

## Appendix A Technical appendices and supplementary material

We first provide more detailed implementations of our LVDrive in the following Section A.1 to Section A.3. Then we provide more qualitative results and comparisons in Section A.4.

### A.1 Latent Vision Decoding

#### Vision decoder ViSθ\\textsc{ViS}\_{\\theta}.

The visual decoder $\textsc{ViS}_{\theta}$ extracts latent semantic features $V_{t+j}\in\mathbb{R}^{N\times C_{v}}$ from the vision embedding $H_{t+j}\in\mathbb{R}^{N\times D}$ produced by LLM. Here, $D$ is set to 4096, matching the hidden dimension of the LLM, and $C_{v}$ is set to 256, corresponding to the post-quantized embedding dimension of VQGAN-ImageNet [^9]. The lightweight vision decoder $\textsc{ViS}_{\theta}$ consists of a single linear layer that projects from 4096 to 256 dimensions, with no activation function applied.

#### Ground-truth future scene feature generation.

To obtain the ground-truth supervision targets, we feed the future front-view images with the width and height of $256\times 256$ at the next six timesteps into VQGAN-ImageNet [^9], which produces latent visual features of shape $16\times 16\times 256$. Layer normalization is then applied to these ground-truth features to stabilize training.

### A.2 Two-stage Trajectory Decoding

#### VAE-based planner.

As described in Section 3.3, the planning embedding is first mapped to a Gaussian distribution by the distribution generator $\textsc{DiS}_{\theta}$. This generator employs three $1\times 1$ 1D convolutional layers with ReLU activations [^1], which compress the 4096-dimensional planning embedding to 2048 dimensions. An average pooling layer followed by an additional $1\times 1$ convolutional layer then produces the mean and log standard deviation of the distribution, each with a latent dimension $C_{z}$ as 32. During training, the ground-truth ego trajectory is concatenated with the planning embedding as the input to generate the distribution to stabilize distribution learning. A latent code of dimension 32 is then sampled and fed into the GRU-based state decoder $\textsc{State}_{\theta}$, which generates the future motion states.

The sampled latent code is first repeated six times to align with the six prediction timesteps. The state decoder $\textsc{State}_{\theta}$ consists of four GRU layers with a latent dimension of 1024, followed by three MLP layers with ReLU activations with the output dimension of 4096. To preserve the current motion information, the planning embedding is also repeated six times and concatenated with the output future motion states produced by the state decoder. The resulting ego state $s_{ego}$ thus has a shape of 8192.

This state is subsequently passed through three additional MLP layers with ReLU activations to generate the multi-modal coarse trajectory.

#### Trajectory refiner.

The core of the trajectory refinement module is a two-layer transformer block with a cross-attention mechanism, operating with a latent dimension $C_{r}$ as 1024. The ego state serves as the learnable query, and the future visual embedding acts as the key and value, both of which are first projected to match the latent dimension. The projector $\textsc{ProJ}_{\theta}$ consists of a single linear layer, followed by layer normalization and GELU activation [^16]. The updated ego motion state $s_{ego}^{\ast}$ is then fed into two separate MLP blocks with identical architectures, which generate the base trajectory and the offset to enhance generation diversity. Each MLP block contains two linear layers with layer normalization and GELU activation. The sum of the base trajectory and the offset yields the final trajectory.

### A.3 Vision Encoding with the Multi-View Image Input

We adopt the same multi-view visual feature extraction and compression procedure as [^11], to which we refer readers for more detailed explanations.

In brief, image tokens are first extracted by the vision encoder EVA-02-L [^10]. The QT-Former module then employs three types of learnable queries, termed as perception queries, scene queries, and history queries, to capture structured visual features, compressed current scene features, and historical scene context, respectively. The module which performs critical object detection, traffic state prediction, and motion prediction of surrounding agents, is supervised by ground-truth labels during the training, yielding the auxiliary loss $\mathcal{L}_{qt}$. The compressed current and historical scene features, rather than the heavy image tokens, are subsequently fed into the core LLM.

### A.4 Additional Qualitative Results

We provide more qualitative comparisons of our LVDrive and the baseline model $\mathcal{M}_{base}$. We visualize the multi-view image (left) and the structured BEV view (right) in each scenario. The blue line denotes the generated trajectory. In the BEV view, the ego vehicle is shown as the gray box. The detected surrounding vehicles are shown as green boxes with an orange heading indicator that marks their moving direction.

#### Multi-ability evaluation.

Bench2Drive defines five advanced driving skills for evaluation: Emergency Brake, Merging, Traffic Sign, Overtaking, and Give Way. Each evaluation scenarios is mapped to one or more of these skills, and the distribution across skills is not uniform, as shown in Table 7.

Table 7: Evaluation scenario distribution for five driving skills.

|  | Traffic Sign | Merging | Emergency Brake | Overtaking | Give Way |
| --- | --- | --- | --- | --- | --- |
| Numbers of Scenarios | 18 | 16 | 12 | 9 | 2 |

As shown in Table 2, LVDrive achieves strong performance on four of the five driving skills, with the exception of Give Way. We present qualitative results for the Emergency Brake, Merging, Traffic Sign, and Overtaking scenarios to illustrate this effectiveness. Additionally, we provide an analysis of a representative failure case in the Give Way scenario to understand the limitations of our approach.

#### Performance on Emergency Brake.

The qualitative results and comparisons in an Emergency Brake scenario are shown in Figure 4. This scenario presents a particularly challenging task that demands both precise spatial reasoning and timely planning. The ego vehicle is first required to execute an unprotected left turn at an intersection without a traffic light, which also requires careful judgment of gaps in the oncoming traffic. Immediately after completing the turn, the ego vehicle encounters a bicycle that unexpectedly crosses the path, forcing an abrupt transition from turning to braking. The ego vehicle must execute an emergency brake with minimal hesitation, wait for the bicycle to fully clear the road, and subsequently resume the driving. The ego vehicle controlled by $\mathcal{M}_{base}$ fails at the very first step. It cannot complete the unprotected left turn and collides with another vehicle at the intersection. In contrast, LVDrive successfully navigates the entire sequence by completing the unprotected turn, braking promptly for the crossing bicycle and safely resuming driving. The performance demonstrates how the future scene prediction capability enables LVDrive to anticipate sudden events and plan accordingly.

#### Performance on Merging.

The qualitative results and comparisons in a Merging scenario are shown in Figure 5. Navigating highway exits and merging onto narrow roads presents a demanding planning challenge, where the ego vehicle must simultaneously perceive the off-ramp geometry, identify the merging point, and execute a precise lateral maneuver into a confined space. The ego vehicle controlled by $\mathcal{M}_{base}$ fails to capture the exact road boundary and generates a trajectory with a lateral bias that ultimately leads to a collision with the guardrail. This failure suggests that without a powerful scene understanding ability, the baseline model struggles to reason about the spatial constraints imposed by the complex road layout. In contrast, LVDrive accurately perceives the available merging space, plans a safe and smooth trajectory, and successfully merges into the narrow road without incident. This result demonstrates that latent visual representation learning provides a stronger spatial understanding capability, which is critical for precise maneuvers in spatially constrained scenarios.

#### Performance on Traffic Sign.

The qualitative results and comparisons in a Traffic Sign scenario are shown in Figure 6. The ego vehicle is required to make a right turn at a non-signalized intersection without traffic lights or signs, and simultaneously avoid collisions with surrounding vehicles navigating through the junction. This scenario poses a dual challenge: the model must accurately perceive the unstructured road geometry, including the intersection layout and traffic sign placement, and track the behavior of nearby vehicles to identify safe turning gaps. The ego vehicle controlled by $\mathcal{M}_{base}$ plans an inaccurate trajectory that deviates from the intended route and ultimately collides with a traffic sign, suggesting a failure to correctly interpret the spatial layout of the intersection. In contrast, LVDrive successfully captures both the precise road topology and the motion of surrounding dynamic cars, and plans a carefully executed trajectory that navigates through the junction without incident. This highlights LVDrive’s enhanced spatial understanding and reasoning of road structures and dynamic agents in the highly-interactive scenario.

#### Performance on Overtaking.

The qualitative results and comparisons in another Overtaking scenario are shown in Figure 7. This scenario presents a subtle yet demanding perception challenge. While the ego vehicle is driving forward, a car parked in the adjacent lane suddenly opens its door, significantly deforming its original contour and partially intruding into the ego vehicle’s driving path. Unlike typical static obstacles, an open door represents a non-rigid shape variation of a familiar car object, which can easily be overlooked or misinterpreted by models that lack fine-grained semantic understanding of the scene. The ego vehicle controlled by $\mathcal{M}_{base}$ fails to recognize the shape variation. It comes to a halt and subsequently becomes immobilized, unable to plan a feasible path around the obstruction. LVDrive accurately perceives the altered vehicle shape and correctly interprets it as a navigational hazard, successfully planning a safe trajectory that bypasses the car with the open door and continues forward without interruption. This comparison highlights LVDrive’s ability to capture fine-grained semantic details and reason about their implications for safe motion planning.

#### Failure case on Give Way.

One failure case in Give Way scenario is shown in Figure 8. The ego vehicle is required to yield to an emergency vehicle that approaches from behind. However, LVDrive maintains its straight route and fails to yield to the ambulance. We attribute this failure to the incorporation of the front-view future scene prediction task, which inherently leads the model to prioritize front-view scene representations. As a result, LVDrive overlooks the rear interaction and fails to correctly identify the approaching vehicle as an emergency ambulance, leading to an unsuccessful Give Way task.

![[nips_vis_a3.png|Refer to caption]]

Figure 4: Qualitative results of our LVDrive and ℳ b a s e \\mathcal{M}\_{base} in an Emergency Brake scenario. The ego vehicle is required to first perform an unprotected left turn at an intersection without a traffic light, then yield to a bicycle crossing its path. Upon encountering the crossing bicycle, the ego vehicle should execute an emergency brake, wait for the bicycle to clear the road, and subsequently resume driving. The ego vehicle of M M\_{base} fails to turn, resulting in the collision with the car at the intersection. In contrast, LVDrive successfully navigates the entire sequence, with completing the turn, braking for the bicycle, and safely resuming driving, which demonstrates robust planning performance in this challenging scenario.

![[nips_vis_a2.png|Refer to caption]]

Figure 5: Qualitative results of our LVDrive and ℳ b a s e \\mathcal{M}\_{base} in a Merging scenario. Navigating highway exits and merging onto narrow roads requires precise perception of the road layout and fine-grained trajectory planning. The ego vehicle of fails to capture the exact road boundary, leading to a lateral deviation that results in a collision with the guardrail. In contrast, LVDrive accurately perceives the off-ramp geometry and plans a safe trajectory to merge into the narrow road without incident.

![[nips_vis_a4.png|Refer to caption]]

Figure 6: Qualitative results of our LVDrive and ℳ b a s e \\mathcal{M}\_{base} in a Traffic Sign scenario. The ego vehicle is required to make a right turn at a non-signalized intersection without traffic lights or signs, while avoiding collisions with surrounding vehicles. The ego vehicle controlled by plans an inaccurate trajectory that deviates from the intended route and ultimately collides with a traffic sign. In contrast, LVDrive successfully and carefully plans a safe trajectory, navigating through the junction without incident.

![[nips_vis_a1.png|Refer to caption]]

Figure 7: Qualitative results of our LVDrive and ℳ b a s e \\mathcal{M}\_{base} in an Overtaking scenario. The ego vehicle drives forward in its lane, and the front car in the adjacent lane stops and opens the door, which blocks the driving lane. The ego vehicle of \\mathcal{M}\_{b}ase stops and gets stuck in the place. In contrast, our LVDrive successfully plans the safe trajectory to bypass the front car with the open door and drives forward continuously.

![[nips_vis_a5.png|Refer to caption]]

Figure 8: Failure case of our LVDrive in a Give Way scenario. The ego vehicle is required to yield to the emergency vehicle that approaches from behind. LVDrive maintains the straight route and fails to yield to the ambulance.

[^1]: A. F. Agarap (2018) Deep learning using rectified linear units (relu). External Links: [Link](https://api.semanticscholar.org/CorpusID:287476751) Cited by: §A.2.

[^2]: M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes, M. Komeili, M. Muckley, A. Rizvi, C. Roberts, K. Sinha, A. Zholus, S. Arnaud, A. Gejji, A. Martin, F. Hogan, D. Dugas, P. Bojanowski, V. Khalidov, P. Labatut, F. Massa, M. Szafraniec, K. Krishnakumar, Y. Li, X. Ma, S. Chandar, F. Meier, Y. LeCun, M. Rabbat, N. Ballas, F. at Meta, M. -. Québec, A. Institute, and P. Montréal (2025) V-jepa 2: self-supervised video models enable understanding, prediction and planning. ArXiv abs/2506.09985. External Links: [Link](https://api.semanticscholar.org/CorpusID:279306055) Cited by: §1.

[^3]: M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. G. Rabbat, Y. LeCun, and N. Ballas (2023) Self-supervised learning from images with a joint-embedding predictive architecture. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15619–15629. External Links: [Link](https://api.semanticscholar.org/CorpusID:255999752) Cited by: §1.

[^4]: J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and J. Zhou (2023) Qwen-vl: a versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966. Cited by: §2.

[^5]: L. Beyer, A. Steiner, A. S. Pinto, A. Kolesnikov, X. Wang, D. M. Salz, M. Neumann, I. M. Alabdulmohsin, M. Tschannen, E. Bugliarello, T. Unterthiner, D. Keysers, S. Koppula, F. Liu, A. Grycner, A. A. Gritsenko, N. Houlsby, M. Kumar, K. Rong, J. M. Eisenschlos, R. Kabra, M. Bauer, M. Bovsnjak, X. Chen, M. Minderer, P. Voigtlaender, I. Bica, I. Balazevic, J. Puigcerver, P. Papalampidi, O. Hénaff, X. Xiong, R. Soricut, J. Harmsen, and X. Zhai (2024) PaliGemma: a versatile 3b vlm for transfer. ArXiv abs/2407.07726. External Links: [Link](https://api.semanticscholar.org/CorpusID:271088378) Cited by: §2.

[^6]: J. Cheng, Y. Chen, and Q. Chen (2024) PLUTO: pushing the limit of imitation learning-based planning for autonomous driving. ArXiv abs/2404.14327. External Links: [Link](https://api.semanticscholar.org/CorpusID:269293351) Cited by: §4.1.

[^7]: K. Ding, B. Chen, Y. Su, H. Gao, B. Jin, C. Sima, W. Zhang, X. Li, P. Barsch, H. Li, and H. Zhao (2024) Hint-ad: holistically aligned interpretability in end-to-end autonomous driving. In Conference on Robot Learning, External Links: [Link](https://api.semanticscholar.org/CorpusID:272550834) Cited by: §2.

[^8]: A. Dosovitskiy, G. Ros, F. Codevilla, A. M. López, and V. Koltun (2017) CARLA: an open urban driving simulator. In Conference on Robot Learning, External Links: [Link](https://api.semanticscholar.org/CorpusID:5550767) Cited by: §4.1.

[^9]: P. Esser, R. Rombach, and B. Ommer (2020) Taming transformers for high-resolution image synthesis. External Links: 2012.09841 Cited by: §A.1, §A.1, §4.2, Table 4.

[^10]: Y. Fang, Q. Sun, X. Wang, T. Huang, X. Wang, and Y. Cao (2023) EVA-02: a visual representation for neon genesis. Image Vis. Comput. 149, pp. 105171. External Links: [Link](https://api.semanticscholar.org/CorpusID:257631498) Cited by: §A.3, §4.2.

[^11]: H. Fu, D. Zhang, Z. Zhao, J. Cui, D. Liang, C. Zhang, D. Zhang, H. Xie, B. Wang, and X. Bai (2025) Orion: a holistic end-to-end autonomous driving framework by vision-language instructed action generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 24823–24834. Cited by: §A.3, §2, §3.1, §3.3, §4.2, §4.4, Table 1, Table 2.

[^12]: H. Fu, D. Zhang, Z. Zhao, J. Cui, H. Xie, B. Wang, G. Chen, D. Liang, and X. Bai (2025) MindDrive: a vision-language-action model for autonomous driving via online reinforcement learning. arXiv Preprint arXiv:2512.13636. Cited by: §2, Table 1, Table 2.

[^13]: S. Gao, J. Yang, L. Chen, K. Chitta, Y. Qiu, A. Geiger, J. Zhang, and H. Li (2024) Vista: a generalizable driving world model with high fidelity and versatile controllability. ArXiv abs/2405.17398. External Links: [Link](https://api.semanticscholar.org/CorpusID:270062607) Cited by: §2.

[^14]: D. R. Ha and J. Schmidhuber (2018) World models. ArXiv abs/1803.10122. External Links: [Link](https://api.semanticscholar.org/CorpusID:4807711) Cited by: §2.

[^15]: M. Hassan, S. Stapf, A. Rahimi, P. M. B. Rezende, Y. Haghighi, D. Brüggemann, I. Katircioglu, L. Zhang, X. Chen, S. Saha, M. Cannici, E. Aljalbout, B. Ye, X. Wang, A. Davtyan, M. Salzmann, D. Scaramuzza, M. Pollefeys, P. Favaro, and A. Alahi (2024) GEM: a generalizable ego-vision multimodal world model for fine-grained ego-motion, object dynamics, and scene composition control. 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 22404–22415. External Links: [Link](https://api.semanticscholar.org/CorpusID:274776855) Cited by: §2.

[^16]: D. Hendrycks and K. Gimpel (2016) Gaussian error linear units (gelus). arXiv: Learning. External Links: [Link](https://api.semanticscholar.org/CorpusID:125617073) Cited by: §A.2.

[^17]: A. Hu, L. Russell, H. Yeo, Z. Murez, G. Fedoseev, A. Kendall, J. Shotton, and G. Corrado (2023) GAIA-1: a generative world model for autonomous driving. ArXiv abs/2309.17080. External Links: [Link](https://api.semanticscholar.org/CorpusID:263310665) Cited by: §2.

[^18]: J. E. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, and W. Chen (2021) LoRA: low-rank adaptation of large language models. ArXiv abs/2106.09685. External Links: [Link](https://api.semanticscholar.org/CorpusID:235458009) Cited by: §4.2.

[^19]: T. Hu, X. Liu, S. Wang, Y. Zhu, A. Liang, L. Kong, G. Zhao, Z. Gong, J. Cen, Z. Huang, X. Hao, L. Li, H. Song, X. Li, J. Ma, S. Shen, J. Zhu, D. Tao, Z. Liu, and J. Liang (2025) Vision-language-action models for autonomous driving: past, present, and future. ArXiv abs/2512.16760. External Links: [Link](https://api.semanticscholar.org/CorpusID:283934451) Cited by: §1.

[^20]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, L. Lu, X. Jia, Q. Liu, J. Dai, Y. Qiao, and H. Li (2023) Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, Cited by: Table 1, Table 2.

[^21]: J. Hwang, R. Xu, H. Lin, W. Hung, J. Ji, K. Choi, D. Huang, T. He, P. Covington, B. Sapp, J. Guo, D. Anguelov, and M. Tan (2024) EMMA: end-to-end multimodal model for autonomous driving. Trans. Mach. Learn. Res. 2025. External Links: [Link](https://api.semanticscholar.org/CorpusID:273695673) Cited by: §1, §2.

[^22]: F. Jia, L. Liu, Z. Song, C. Jia, H. Ye, X. Hao, and L. Chen (2026) DriveWorld-vla: unified latent-space world modeling with vision-language-action for autonomous driving. ArXiv abs/2602.06521. External Links: [Link](https://api.semanticscholar.org/CorpusID:285401135) Cited by: §1, §1, §1, §2.

[^23]: X. Jia, Y. Gao, L. Chen, J. Yan, P. L. Liu, and H. Li (2023) DriveAdapter: breaking the coupling barrier of perception and planning in end-to-end autonomous driving. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 7919–7929. External Links: [Link](https://api.semanticscholar.org/CorpusID:260351212) Cited by: Table 1, Table 2.

[^24]: X. Jia, P. Wu, L. Chen, J. Xie, C. He, J. Yan, and H. Li (2023) Think twice before driving: towards scalable decoders for end-to-end autonomous driving. In CVPR, Cited by: Table 1, Table 2.

[^25]: X. Jia, Z. Yang, Q. Li, Z. Zhang, and J. Yan (2024) Bench2Drive: towards multi-ability benchmarking of closed-loop end-to-end autonomous driving. In NeurIPS 2024 Datasets and Benchmarks Track, Cited by: §4.1.

[^26]: X. Jia, J. You, Z. Zhang, and J. Yan (2025) DriveTransformer: unified transformer for scalable end-to-end autonomous driving. In The Thirteenth International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=M42KR4W9P5) Cited by: §4.1, Table 1, Table 2.

[^27]: A. Jiang, Y. Gao, Z. Sun, Y. Wang, J. Wang, J. Chai, Q. Cao, Y. Heng, H. Jiang, Z. Zhang, X. Guo, H. Sun, and H. Zhao (2025) DiffVLA: vision-language guided diffusion planning for autonomous driving. ArXiv abs/2505.19381. External Links: [Link](https://api.semanticscholar.org/CorpusID:278904380) Cited by: §2.

[^28]: B. Jiang, S. Chen, B. Liao, X. Zhang, W. Yin, Q. Zhang, C. Huang, W. Liu, and X. Wang (2024) Senna: bridging large vision-language models and end-to-end autonomous driving. ArXiv abs/2410.22313. External Links: [Link](https://api.semanticscholar.org/CorpusID:273661831) Cited by: §1, §2.

[^29]: B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang (2023) VAD: vectorized scene representation for efficient autonomous driving. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 8306–8316. External Links: [Link](https://api.semanticscholar.org/CorpusID:257636676) Cited by: §3.3, Table 1, Table 2.

[^30]: Y. Li, L. Fan, J. He, Y. Wang, Y. Chen, Z. Zhang, and T. Tan (2024) Enhancing end-to-end autonomous driving with latent world model. ArXiv abs/2406.08481. External Links: [Link](https://api.semanticscholar.org/CorpusID:270391188) Cited by: §2.

[^31]: Y. Li, S. Shang, W. Liu, B. Zhan, H. Wang, Y. Wang, Y. Chen, X. Wang, Y. An, C. Tang, L. Hou, L. Fan, and Z. Zhang (2025) DriveVLA-w0: world models amplify data scaling law in autonomous driving. ArXiv abs/2510.12796. External Links: [Link](https://api.semanticscholar.org/CorpusID:282064644) Cited by: §1, §1, §1, §2, §2.

[^32]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, et al. (2025) ReCogDrive: a reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052. Cited by: §1, §2, Table 1, Table 2.

[^33]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al. (2025) Diffusiondrive: truncated diffusion model for end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 12037–12047. Cited by: §2, §4.1.

[^34]: H. Lin, Y. Yang, Y. Zhang, C. Zheng, J. Feng, S. Wang, Z. Wang, S. Chen, B. Wang, Y. Zhang, X. Liu, S. Cui, and Z. Li (2025) FutureX: enhance end-to-end autonomous driving via latent chain-of-thought world model. ArXiv abs/2512.11226. External Links: [Link](https://api.semanticscholar.org/CorpusID:283883278) Cited by: §2.

[^35]: H. Liu, C. Li, Q. Wu, and Y. J. Lee (2023) Visual instruction tuning. NeurIPS. Cited by: §2.

[^36]: M. Liu, D. Zhang, J. Liu, J. Cui, H. Xie, G. Chen, H. Ye, M. Y. Yang, F. Nex, and H. Cheng (2026) DriveVA: video action models are zero-shot drivers. arXiv preprint arXiv:2604.04198. Cited by: §2.

[^37]: Y. Luo, Q. Chen, F. Li, S. Xu, J. Liu, Z. Song, Z. Yang, and F. Wen (2026) Unleashing vla potentials in autonomous driving via explicit learning from failures. arXiv preprint arXiv:2603.01063. Cited by: §2.

[^38]: Y. Luo, F. Li, S. Xu, Y. Ji, Z. Zhang, B. Wang, Y. Shen, J. Cui, L. Chen, G. Chen, et al. (2026) Last-vla: thinking in latent spatio-temporal space for vision-language-action in autonomous driving. arXiv preprint arXiv:2603.01928. Cited by: §2.

[^39]: X. Mei, S. Wang, J. Cheng, Y. Chen, and D. Xu (2025) HAMF: a hybrid attention-mamba framework for joint scene context understanding and future motion representation learning. 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 4107–4114. External Links: [Link](https://api.semanticscholar.org/CorpusID:278782237) Cited by: §4.1.

[^40]: NVIDIA, Y. Wang, W. Luo, J. Bai, Y. Cao, T. Che, K. Chen, Y. Chen, J. Diamond, Y. Ding, W. Ding, L. Feng, G. Heinrich, J. Huang, P. Karkus, B. Li, P. Li, T. Lin, D. Liu, M. Liu, L. Liu, Z. Liu, J. Lu, Y. Mao, P. Molchanov, L. Pavao, Z. Peng, M. Ranzinger, E. Schmerling, S. Shen, Y. Shi, S. Tariq, R. Tian, T. Wekel, X. Weng, T. Xiao, E. Yang, X. Yang, Y. You, X. Zeng, W. Zhang, B. Ivanovic, and M. Pavone (2025) Alpamayo-R1: bridging reasoning and action prediction for generalizable autonomous driving in the long tail. arXiv preprint arXiv:2511.00088. Cited by: §2.

[^41]: K. Renz, L. Chen, E. Arani, and O. Sinavski (2025) SimLingo: vision-only closed-loop autonomous driving with language-action alignment. 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11993–12003. External Links: [Link](https://api.semanticscholar.org/CorpusID:276937682) Cited by: §1, §2.

[^42]: K. Renz, L. Chen, A. Marcu, J. Hünermann, B. Hanotte, A. Karnsund, J. Shotton, E. Arani, and O. Sinavski (2024) CarLLaVA: vision language models for camera-only closed-loop driving. ArXiv abs/2406.10165. External Links: [Link](https://api.semanticscholar.org/CorpusID:270521573) Cited by: §2.

[^43]: S. Shang, Y. Chen, Y. Wang, Y. Li, and Z. Zhang (2025) DriveDPO: policy learning via safety dpo for end-to-end autonomous driving. ArXiv abs/2509.17940. External Links: [Link](https://api.semanticscholar.org/CorpusID:281420247) Cited by: Table 1.

[^44]: O. Sim’eoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose, V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa, F. Massa, D. Haziza, L. Wehrstedt, J. Wang, T. Darcet, T. Moutakanni, L. Sentana, C. Roberts, A. Vedaldi, J. Tolan, J. Brandt, C. Couprie, J. Mairal, H. J’egou, P. Labatut, and P. Bojanowski (2025) DINOv3. External Links: [Link](https://api.semanticscholar.org/CorpusID:280649654) Cited by: §4.4, Table 4.

[^45]: Z. Song, C. Jia, L. Liu, H. Pan, Y. Zhang, J. Wang, X. Zhang, S. Xu, L. Yang, and Y. Luo (2025) Don’t shake the wheel: momentum-aware planning in end-to-end autonomous driving. 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 22432–22441. External Links: [Link](https://api.semanticscholar.org/CorpusID:276782149) Cited by: Table 1.

[^46]: S. Tan, K. Chitta, Y. Chen, R. Tian, Y. You, Y. Wang, W. Luo, Y. Cao, P. Krahenbuhl, M. Pavone, et al. (2025) Latent chain-of-thought world modeling for end-to-end driving. arXiv preprint arXiv:2512.10226. Cited by: §2.

[^47]: X. Tian, J. Gu, B. Li, Y. Liu, C. Hu, Y. Wang, K. Zhan, P. Jia, X. Lang, and H. Zhao (2024) DriveVLM: the convergence of autonomous driving and large vision-language models. ArXiv abs/2402.12289. External Links: [Link](https://api.semanticscholar.org/CorpusID:267750682) Cited by: §2.

[^48]: T. Wang, C. Zhang, X. Qu, K. Li, W. Liu, and C. Huang (2025) DiffAD: a unified diffusion modeling approach for autonomous driving. External Links: 2503.12170, [Link](https://arxiv.org/abs/2503.12170) Cited by: Table 1, Table 2.

[^49]: W. Wang, J. Xie, C. Hu, H. Zou, J. Fan, W. Tong, Y. Wen, S. Wu, H. Deng, Z. Li, H. Tian, L. Lu, X. Zhu, X. Wang, Y. Qiao, and J. Dai (2023) DriveMLM: aligning multi-modal large language models with behavioral planning states for autonomous driving. Visual Intelligence 3. External Links: [Link](https://api.semanticscholar.org/CorpusID:266210476) Cited by: §2.

[^50]: X. Wang, X. Zhang, Z. Luo, Q. Sun, Y. Cui, J. Wang, F. Zhang, Y. Wang, Z. Li, Q. Yu, Y. Zhao, Y. Ao, X. Min, T. Li, B. Wu, B. Zhao, B. Zhang, L. Wang, G. Liu, Z. He, X. Yang, J. Liu, Y. Lin, T. Huang, and Z. Wang (2024) Emu3: next-token prediction is all you need. ArXiv abs/2409.18869. External Links: [Link](https://api.semanticscholar.org/CorpusID:272968818) Cited by: §4.4, Table 4.

[^51]: P. Wu, X. Jia, L. Chen, J. Yan, H. Li, and Y. Qiao (2022) Trajectory-guided control prediction for end-to-end autonomous driving: a simple yet strong baseline. In NeurIPS, Cited by: Table 1, Table 1, Table 2, Table 2.

[^52]: Z. Xiong, X. Ye, B. Yaman, S. Cheng, Y. Lu, J. Luo, N. Jacobs, and L. Ren (2026) UniDrive-wm: unified understanding, planning and generation world model for autonomous driving. ArXiv abs/2601.04453. External Links: [Link](https://api.semanticscholar.org/CorpusID:284544294) Cited by: §1, §1, §2, §4.3, Table 1, Table 1, Table 2, Table 2.

[^53]: T. Xu, H. Lu, X. Yan, Y. Cai, B. Liu, and Y. Chen (2025) Occ-llm: enhancing autonomous driving with occupancy-based large language models. 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 8434–8441. External Links: [Link](https://api.semanticscholar.org/CorpusID:276250108) Cited by: §2.

[^54]: Z. Xu, Y. Zhang, E. Xie, Z. Zhao, Y. Guo, K. K. Wong, Z. Li, and H. Zhao (2023) DriveGPT4: interpretable end-to-end autonomous driving via large language model. IEEE Robotics and Automation Letters 9, pp. 8186–8193. External Links: [Link](https://api.semanticscholar.org/CorpusID:263605524) Cited by: §1, §2.

[^55]: Z. Yang, L. Chen, Y. Sun, and H. Li (2023) Visual point cloud forecasting enables scalable autonomous driving. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14673–14684. External Links: [Link](https://api.semanticscholar.org/CorpusID:266690615) Cited by: §2.

[^56]: Z. Yang, Y. Chai, X. Jia, Q. Li, Y. Shao, X. Zhu, H. Su, and J. Yan (2025) DriveMoE: mixture-of-experts for vision-language-action model in end-to-end autonomous driving. ArXiv abs/2505.16278. External Links: [Link](https://api.semanticscholar.org/CorpusID:278789416) Cited by: §2, Table 1, Table 1, Table 2, Table 2.

[^57]: Z. Yang, X. Jia, Q. Li, X. Yang, M. Yao, and J. Yan (2026) Raw2Drive: reinforcement learning with aligned world models for end-to-end autonomous driving (in CARLA v2). In The Thirty-ninth Annual Conference on Neural Information Processing Systems, External Links: [Link](https://openreview.net/forum?id=CAz7UGRdLs) Cited by: Table 1, Table 2.

[^58]: S. Zeng, X. Chang, M. Xie, X. Liu, Y. Bai, Z. Pan, M. Xu, and X. Wei (2025) FutureSightDrive: thinking visually with spatio-temporal cot for autonomous driving. ArXiv abs/2505.17685. External Links: [Link](https://api.semanticscholar.org/CorpusID:278886333) Cited by: §1, §1, §2.

[^59]: K. Zhang, Z. Tang, X. Hu, X. Pan, X. Guo, Y. Liu, J. Huang, L. Yuan, Q. Zhang, X. Long, X. Cao, and W. Yin (2025) Epona: autoregressive diffusion world model for autonomous driving. ArXiv abs/2506.24113. External Links: [Link](https://api.semanticscholar.org/CorpusID:280010626) Cited by: §1, §2.

[^60]: L. Zhang, Y. Xiong, Z. Yang, S. Casas, R. Hu, and R. Urtasun (2023) Copilot4D: learning unsupervised world models for autonomous driving via discrete diffusion. In International Conference on Learning Representations, External Links: [Link](https://api.semanticscholar.org/CorpusID:264935456) Cited by: §2.

[^61]: L. Zheng, W. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica (2023) Judging llm-as-a-judge with mt-bench and chatbot arena. ArXiv abs/2306.05685. External Links: [Link](https://api.semanticscholar.org/CorpusID:259129398) Cited by: §4.2.

[^62]: W. Zheng, R. Song, X. Guo, and L. Chen (2024) GenAD: generative end-to-end autonomous driving. In European Conference on Computer Vision, External Links: [Link](https://api.semanticscholar.org/CorpusID:267750483) Cited by: §3.3.

[^63]: W. Zheng, Z. Xia, Y. Huang, S. Zuo, J. Zhou, and J. Lu (2024) Doe-1: closed-loop autonomous driving with large world model. ArXiv abs/2412.09627. External Links: [Link](https://api.semanticscholar.org/CorpusID:274656022) Cited by: §1, §2, §2.

[^64]: Y. Zheng, P. Yang, Z. Xing, Q. Zhang, Y. Zheng, Y. Gao, P. Li, T. Zhang, Z. Xia, P. Jia, and D. Zhao (2025) World4Drive: end-to-end autonomous driving via intention-aware physical latent world model. ArXiv abs/2507.00603. External Links: [Link](https://api.semanticscholar.org/CorpusID:280115120) Cited by: §2.

[^65]: X. Zhou, D. Liang, S. Tu, X. Chen, Y. Ding, D. Zhang, F. Tan, H. Zhao, and X. Bai (2025) HERMES: a unified self-driving world model for simultaneous 3d scene understanding and generation. ArXiv abs/2501.14729. External Links: [Link](https://api.semanticscholar.org/CorpusID:275906961) Cited by: §1.

[^66]: X. Zhou, X. Han, F. Yang, Y. Ma, and A. Knoll (2025) OpenDriveVLA: towards end-to-end autonomous driving with large vision language action model. ArXiv abs/2503.23463. External Links: [Link](https://api.semanticscholar.org/CorpusID:277452134) Cited by: §2.

[^67]: Z. Zhou, T. Cai, S. Z. Zhao, Y. Zhang, Z. Huang, B. Zhou, and J. Ma (2025) AutoVLA: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. ArXiv abs/2506.13757. External Links: [Link](https://api.semanticscholar.org/CorpusID:279410595) Cited by: §2.