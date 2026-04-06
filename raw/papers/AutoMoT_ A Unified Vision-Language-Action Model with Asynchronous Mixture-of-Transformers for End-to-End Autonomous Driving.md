---
title: "AutoMoT: A Unified Vision-Language-Action Model with Asynchronous Mixture-of-Transformers for End-to-End Autonomous Driving"
source: "https://arxiv.org/html/2603.14851v1"
author:
published:
created: 2026-04-05
description:
tags:
  - "clippings"
---
Wenhui Huang    Songyan Zhang    Qihang Huang    Zhidong Wang    Zhiqi Mao    Collister Chua    Zhan Chen    Long Chen    Chen Lv

###### Abstract

Integrating vision-language models (VLMs) into end-to-end (E2E) autonomous driving (AD) systems has shown promise in improving scene understanding. However, existing integration strategies suffer from several limitations: they either struggle to resolve distribution misalignment between reasoning and action spaces, underexploit the general reasoning capabilities of pretrained VLMs, or incur substantial inference latency during action policy generation, which degrades driving performance. To address these challenges, we propose AutoMoT in this work, an end-to-end AD framework that unifies reasoning and action generation within a single vision-language-action (VLA) model. Our approach leverages a mixture-of-transformer (MoT) architecture with joint attention sharing, which preserves the general reasoning capabilities of pre-trained VLMs while enabling efficient fast-slow inference through asynchronous execution at different task frequencies. Extensive experiments on multiple benchmarks, under both open- and closed-loop settings, demonstrate that AutoMoT achieves competitive performance compared to state-of-the-art methods. We further investigate the functional boundary of pre-trained VLMs in AD, examining when AD-tailored fine-tuning is necessary. Our results show that pre-trained VLMs can achieve competitive multi-task scene understanding performance through semantic prompting alone, while fine-tuning remains essential for action-level tasks such as decision-making and trajectory planning. We refer to [Project Page](https://automot-website.github.io/) for the demonstration videos and qualitative results.

Machine Learning, ICML

## 1 Introduction

The hierarchical modular pipeline, typically comprising perception, prediction, and planning, has been widely adopted in end-to-end (E2E) autonomous driving (AD) systems in recent years [^11] [^21] [^28] [^16] [^27]. Recent advances in vision–language models (VLMs) have further benefited AD by enhancing high-level scene understanding, a capability that is often insufficient in conventional data-driven E2E systems when deployed in complex open-world scenarios. By leveraging their strong generalization and reasoning capabilities, VLMs endow AD systems with the potential to handle complex interactions and provide semantic explanations, thereby improving the interpretability.

![[x1 13.png|Refer to caption]]

Refer to caption

The integration of vision–language models (VLMs) with end-to-end (E2E) autonomous driving systems is undergoing rapid development, giving rise to a diverse set of emerging design paradigms. A natural extension of the E2E framework incorporates VLMs into the upstream stages of the pipeline [^7] [^25], where pre-trained models provide rich scene understanding to support downstream planning, as illustrated in Fig. 1(a). Another line of work adopts a dual-system architecture (Fig. 1(b)), in which the VLM operates as an auxiliary module that assists conventional E2E pipelines by supplying high-level conditioning signals [^20] [^22] [^37]. However, these approaches suffer from inherent distributional misalignment between the reasoning space of VLMs and the action space of planners. Furthermore, fine-tuning VLMs to generate intermediate conditioning signals inevitably constrains them to task-specific roles, diminishing the general capabilities of pretrained models.

More recently, as illustrated in Fig. 1c, emerging vision–language–action (VLA) architectures integrate reasoning and planning within a single pre-trained VLM backbone via autoregressive modeling [^39] [^49] [^48]. While this unified design is compact and effectively leverages the strong reasoning capabilities of VLMs, tightly coupling action policy execution with high-level reasoning at a synchronized temporal frequency is impractical for real-world autonomous driving. This limitation becomes particularly severe in complex interactive environments, where low-latency control and rapid replanning are critical. Prior vision–language models that generate actions in textual form [^46] [^45] [^14] can also be viewed as instances of this paradigm. In addition to the aforementioned limitations, these approaches rely on textual token supervision, which is inherently weaker than direct supervision on numerical action representations. Taking all these limitations into consideration, we pose the following key question: How can VLA models effectively leverage the general intelligence of pre-trained VLMs while acquiring domain-specific capabilities and meeting real-time inference requirements?

In this work, we propose AutoMoT, an end-to-end autonomous driving framework that seamlessly unifies asynchronous reasoning and action within a single vision–language–action (VLA) model, while avoiding both the degradation of VLM capabilities and distributional discrepancies across task spaces. As illustrated in Fig. 1d, AutoMoT adopts a mixture-of-transformers (MoT) architecture that bridges high-level reasoning (scene understanding) and low-level action policies (decision-making and trajectory planning) through joint attention in a shared latent space. This design enables asynchronous execution of textual reasoning and action generation at different temporal frequencies, thereby facilitating fast–slow inference. We comprehensively evaluate AutoMoT on both simulation and real-world benchmarks under closed-loop and open-loop settings. Experimental results demonstrate competitive performance against state-of-the-art (SOTA) baselines, validating both the feasibility of the proposed framework and its effectiveness across diverse evaluation benchmarks. Moreover, through the comprehensive ablation studies, we found that pre-trained VLMs can achieve competitive multi-task scene understanding performance through semantic prompting alone, while fine-tuning remains essential for action-level tasks such as decision-making and trajectory planning.

The primary contributions of this work are as follows:

1. We propose AutoMoT, an end-to-end autonomous driving (AD) framework that seamlessly unifies scene understanding, decision-making, and planning within a single asynchronous VLA model via layer-wise joint attention sharing, while enabling fast–slow inference across tasks through different frequencies.
2. We investigate the functional boundaries of pretrained VLMs in autonomous driving, clarifying when and to what extent AD-specific fine-tuning is necessary across different tasks.
3. Extensive experiments demonstrate competitive performance against state-of-the-art baselines, validating both the feasibility of the proposed framework and its effectiveness across general knowledge, open-loop, and closed-loop evaluation benchmarks.

![[x2 10.png|Refer to caption]]

Refer to caption

## 2 Related Work

### 2.1 End-to-End Autonomous Driving

Planning-oriented methods have been widely adopted in end-to-end autonomous driving frameworks in recent years. For instance, UniAD [^11] proposes a hierarchical modular architecture that enables multiple tasks to be jointly learned in an end-to-end manner, mitigating error accumulation and consequently improving planning performance. The VAD series [^21] [^4] follows this design while introducing vectorized scene representations, which simplify the overall architecture and improve inference efficiency. Subsequently, Para-Drive [^40] extends the hierarchical paradigm to a fully parallel formulation by unifying multiple tasks within the bird’s-eye-view (BEV) space. More recently, diffusion-based policies [^5] have attracted increasing attention in autonomous driving. Existing approaches typically apply diffusion models either as the core planner [^27] [^29] or as a trajectory refiner [^47], leveraging their strong generative capabilities [^35] [^10] to improve driving performance. Nevertheless, these conventional end-to-end approaches still struggle with complex scene understanding, particularly when encountering long-tail and rare scenarios.

### 2.2 Vision-Language Models for Autonomous Driving

The strong scene understanding and semantic reasoning capabilities of VLMs have motivated their rapid integration into E2E AD systems, resulting in several emerging design paradigms. Representative works such as Orion [^7] and ReCogDrive [^25] introduce VLMs as upstream modules to enhance scene understanding and interpretability. Another line of work incorporates VLMs as secondary systems through intermediate representations, where DriveVLM [^37] generates initial trajectory proposals, while Senna [^20] and ReCogDrive [^25] provide high-level decisions to guide downstream planning. Vision–language–action (VLA) architectures, including AutoVLA [^49], Simlingo [^34], OpenREAD [^45], and Alpamayo-R1 [^39], further unify multiple tasks within a single pre-trained VLM backbone. However, the single-transformer design tightly couples reasoning and planning at a synchronized frequency, resulting in substantial inference latency, especially when chain-of-thought (CoT) reasoning is required for complex scene understanding. In contrast, our MoT-based VLA architecture systematically unifies scene understanding, decision-making, and planning in one single model through joint attention sharing [^6] [^12], while remaining functionally decomposed. This design enables fast–slow reasoning with decoupled asynchronous inference frequencies, thereby alleviating latency bottlenecks.

![[x3 11.png|Refer to caption]]

Refer to caption

## 3 AutoMoT

### 3.1 Network Architecture

The overall framework of AutoMoT is illustrated in Fig. 2. AutoMoT comprises two core components: a scene understanding expert and an action expert, all implemented using transformer-based architectures. In the following sections, we detail the design of each component as well as its corresponding training strategies.

#### Understanding Expert

The primary role of the understanding expert (UE) in AutoMoT is to perform scene understanding and generate chain-of-thought (CoT) reasoning for complex scenarios, particularly long-tail and rare cases, while transferring its general knowledge to facilitate action policy learning. The UE adopts Qwen3-VL-4B dense model as its vision–language backbone, which takes as input multi-view and multi-frame RGB images $I^{RGB}\in\mathbb{R}^{N\times H\times W\times C}$ captured by onboard cameras, together with textual prompts $\ell$ consisting of system prompts and user instructions, and outputs semantic reasoning results. To fully leverage the general knowledge of the pretrained Qwen3-VL model and avoid catastrophic degradation of reasoning performance, we freeze the understanding expert throughout the entire training process. The rationale behind this design is further investigated and discussed in Section 4.4.

#### Action Expert

The Action Expert (AE) in AutoMoT is responsible for decision-making and trajectory planning within the unified VLA framework. At each timestep $t$, the AE takes the current observation $o_{t}=\{\mathit{I}^{RGB}_{t},\mathit{I}^{BEV}_{t},Q(t)\}$ as input and produces action-side latent representations. Here, $\mathit{I}^{BEV}_{t}$ denotes the LiDAR BEV feature and $Q(t)$ represents the action queries. From these latent representations, the layer-wise query, key, and value embeddings $\{Q^{l}(t),\tilde{K}^{l}(t),\tilde{V}^{l}(t)\}$ are derived, where $l$ indexes the l-th attention layer. Based on these latent representations, the AE generates semantic decisions for the next three consecutive frames, along with temporal and spatial trajectory proposals over the same horizon. More specifically, given the current observation $o_{t}$ and a set of action queries $Q(t)$, the AE jointly produces latent representations for decision-making and trajectory planning. These representations are decoded into three outputs: (i) concrete meta-actions $\hat{Z}_{t}=\{\hat{z}_{t+h}\}_{h=1}^{H}$, (ii) future temporal waypoints $\hat{Y}_{t}=\{\hat{y}_{t+m}\}_{m=1}^{M}$, and (iii) spatial route points $\bar{Y}_{t}=\{\bar{y}_{t+n}\}_{n=1}^{N}$. Here, $H=3$ denotes a 3-second prediction horizon at 1s intervals for meta-actions, $M=6$ denotes temporal waypoints sampled at 0.5s intervals over the same horizon, and $N$ represents the number of spatial route nodes used to parameterize the reference path. Notably, language, cross-modal, and cross-task interactions are constrained to follow causal attention, while intra-task and self-modal interactions adopt bidirectional attention.

By operating in a shared attention space with the UE, the AE conditions the latent reasoning generated by the UE into the action generation process, thereby grounding decision-making and planning in high-level scene understanding and enabling knowledge transfer from the pretrained VLM to policy learning. The attention patterns are visualized in Fig. 3. As shown, understanding, decision-making, and planning are regulated through cross-task causal attention, where decision representations are conditioned on understanding, and planning is further conditioned on both understanding and decision in the latent space. Within each task, latent features follow bidirectional attention across modalities, while cross-task interactions are governed by causal attention. The AE is implemented as a task-specialized transformer with approximately 1.6B parameters and is trained from scratch to capture domain-specific knowledge for autonomous driving. Notably, the AE operates at a higher frequency than the UE, enabling efficient inference and supporting real-time autonomous driving in complex environments.

### 3.2 Training Strategy

#### Decision Making

We formulate decision-making as a token-level sequence modeling problem over meta-actions, conditioned on multi-frame driving observations. For real-world evaluation, we construct a multi-frame decision-making dataset based on nuScenes, termed NuSync.

Specifically, NuSync takes four consecutive historical RGB observations along with an additional RGB-BEV pair as input. In the synchronous setting, the RGB-BEV pair shares the same timestamp as the last historical frame, i.e., $I^{\text{sync}}_{t}=\{I^{RGB}_{t},I^{RGB}_{t+1},I^{RGB}_{t+2},I^{RGB}_{t+3},I^{RGB}_{t+3},I^{BEV}_{t+3}\}$. In addition, we construct temporally asynchronous samples in which the four historical frames remain consecutive, while the RGB-BEV pair is randomly selected from 1 to 2 frames ahead (corresponding to 0.5–1 s at 2 Hz). For example, $I^{\text{async}}_{t}=\{I^{RGB}_{t},I^{RGB}_{t+1},I^{RGB}_{t+2},I^{RGB}_{t+3},I^{RGB}_{t+k},I^{BEV}_{t+k}\}$, where $k\in\{4,5\}$. In the output space, NuSync annotates meta-actions over a 3-second horizon, providing up to twenty possible combinations of longitudinal and lateral actions at 1s, 2s, and 3s. After curation, NuSync contains 80.1K samples in total. More details about NuSync and the associated decision benchmark are provided in Appendix A.1.

Similarly, for CARLA simulation, we construct the PDM-Meta dataset based on PDM-Lite following the same protocol. Due to the ambiguous boundaries between lateral meta-actions in simulation, we only annotate longitudinal decisions. To the best of our knowledge, NuSync and PDM-Meta are the first open-source decision datasets that support asynchronous multi-frame meta-action inference.

Based on the constructed meta-action datasets, given an observation sequence $o_{t}$, the AE predicts a sequence of meta-action tokens $\hat{z}_{t}=\{\hat{z}_{t}^{j}\}_{j=1}^{J}$, where $j$ represents j-th token and M depicts the necessary amount of tokens to be encoded as a meta-action. Unlike the next-token prediction used by the UE, the AE adopts a token-wise prediction paradigm and optimizes the policy by minimizing the negative log-likelihood of the target decision tokens:

$$
\mathcal{L}_{\mathrm{DM}}=\mathbb{E}_{(o_{t},z_{t})\sim\mathcal{D}}\left[-\sum_{j=1}^{J}\log p_{\theta}\left(z_{t}^{j}\mid o_{t}\right)\right].
$$

where $\mathcal{D}$ denotes the corresponding dataset.

Table 1: Comparison of closed-loop planning performance on the CARLA Bench2Drive leaderboard. C/L denotes camera/LiDAR input. DS and SR represent Driving Score and Success Rate, respectively.

<table><tbody><tr><th rowspan="2">Method</th><td rowspan="2">Expert</td><td rowspan="2">Modality</td><td rowspan="2">VLM</td><td>Generative</td><td colspan="2">Closed-loop Metric</td></tr><tr><td>Planner</td><td>DS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>SR(%) <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><th>MomAD <sup><a href="#fn:36">36</a></sup></th><td>Think2Drive</td><td>C</td><td>-</td><td>-</td><td>44.54</td><td>16.71</td></tr><tr><th>UniAD-Base <sup><a href="#fn:11">11</a></sup></th><td>Think2Drive</td><td>C</td><td>-</td><td>-</td><td>45.81</td><td>16.36</td></tr><tr><th>TCP-traj <sup><a href="#fn:41">41</a></sup></th><td>Think2Drive</td><td>C</td><td>-</td><td>-</td><td>59.90</td><td>30.00</td></tr><tr><th>DriveTransformer-Large <sup><a href="#fn:19">19</a></sup></th><td>Think2Drive</td><td>C</td><td>-</td><td>-</td><td>63.46</td><td>35.01</td></tr><tr><th>DriveAdapter <sup><a href="#fn:17">17</a></sup></th><td>Think2Drive</td><td>C&L</td><td>-</td><td>-</td><td>64.22</td><td>33.08</td></tr><tr><th>Raw2Drive <sup><a href="#fn:44">44</a></sup></th><td>Think2Drive</td><td>C</td><td>-</td><td>-</td><td>71.36</td><td>50.24</td></tr><tr><th>DiffusionDrive <sup><a href="#fn:27">27</a></sup></th><td>PDM-Lite</td><td>C&L</td><td>-</td><td>✓</td><td>77.68</td><td>57.72</td></tr><tr><th>TransFuser++ <sup><a href="#fn:16">16</a></sup></th><td>PDM-Lite</td><td>C&L</td><td>-</td><td>-</td><td>84.21</td><td>67.27</td></tr><tr><th>ReasonPlan <sup><a href="#fn:30">30</a></sup></th><td>Think2Drive</td><td>C</td><td>✓</td><td>-</td><td>64.01</td><td>34.55</td></tr><tr><th>Recogdrive <sup><a href="#fn:25">25</a></sup></th><td>Think2Drive</td><td>C</td><td>✓</td><td>✓</td><td>71.36</td><td>45.45</td></tr><tr><th>DriveMoE <sup><a href="#fn:43">43</a></sup></th><td>Think2Drive</td><td>C</td><td>✓</td><td>-</td><td>74.22</td><td>48.64</td></tr><tr><th>ORION <sup><a href="#fn:7">7</a></sup></th><td>Think2Drive</td><td>C</td><td>✓</td><td>✓</td><td>77.74</td><td>54.62</td></tr><tr><th>SpaceDrive+ <sup><a href="#fn:24">24</a></sup></th><td>PDM-Lite</td><td>C</td><td>✓</td><td>-</td><td>78.02</td><td>55.11</td></tr><tr><th>MindDrive <sup><a href="#fn:8">8</a></sup></th><td>Think2Drive</td><td>C</td><td>✓</td><td>✓</td><td>78.04</td><td>55.09</td></tr><tr><th>AutoVLA <sup><a href="#fn:49">49</a></sup></th><td>PDM-Lite</td><td>C</td><td>✓</td><td>-</td><td>78.84</td><td>57.73</td></tr><tr><th>SimLingo <sup><a href="#fn:34">34</a></sup></th><td>PDM-Lite</td><td>C</td><td>✓</td><td>-</td><td>85.07</td><td>67.27</td></tr><tr><th>AutoMoT (Ours)</th><td>PDM-Lite</td><td>C&L</td><td>✓</td><td>-</td><td>87.34</td><td>70.00</td></tr></tbody></table>

#### Trajectory Planning

AutoMoT follows the original setting of nuScenes and PDM-Lite for both of AE and AR, with each sample consisting of four historical frames and predicting and refining of temporal and spatial trajectories over a 3-second horizon. For the AE, we optimize the trajectory planning with an $\ell_{1}$ loss:

$$
\displaystyle\mathcal{L}_{\text{traj}}^{\text{temp}}
$$
 
$$
\displaystyle=\mathbb{E}_{(o_{t},Y_{t}^{\text{temp}})\sim\mathcal{D}}\left[\frac{1}{M}\sum_{m=1}^{M}\left\|\hat{Y}_{t+m}-Y_{t+m}^{\text{temp}}\right\|_{1}\right],
$$
$$
\displaystyle\mathcal{L}_{\text{traj}}^{\text{spatial}}
$$
 
$$
\displaystyle=\mathbb{E}_{(o_{t},Y_{t}^{\text{spatial}})\sim\mathcal{D}}\left[\frac{1}{N}\sum_{n=1}^{N}\left\|\bar{Y}_{t+n}-Y_{t+n}^{\text{spatial}}\right\|_{1}\right],
$$

where $Y_{t}^{\text{temp}}$ and $Y_{t}^{\text{spatial}}$ denote ground-truth temporal and spatial trajectories. Notably, the decision-making and trajectory planning are jointly optimized within the AE, enabling AutoMoT to learn coherent action policies grounded in semantic representations from the UE.

### 3.3 Asynchronous Inference with Joint Attention

We formulate asynchronous inference as a multi-rate process in which reasoning and action inference evolve at different temporal resolutions, while both remain grounded in real-time visual observations. The interaction between these two processes is mediated by a shared key–value (KV) cache, as illustrated in Fig. 2. At an arbitrary timestep $t$, given the current observation $o_{t}$, the AE derives layer-wise queries, keys, and values $\{Q^{l}_{\text{act}}(t),K^{l}_{\text{act}}(t),V^{l}_{\text{act}}(t)\}$ for each attention layer. Correspondingly, $\tau(t)$ denotes the time index of the most recent scene representation update available at action step $t$, satisfying $\tau(t)\leq t$. At the update time $\tau(t)$, the UE produces a set of layer-wise KV representations, which are stored in a persistent KV cache:

$$
\mathcal{C}^{\tau(t)}=\{K^{l}_{scene}(\tau(t)),V^{l}_{scene}(\tau(t))\}_{l=1}^{L}\ .
$$

Therefore, the keys and values involved in the final attention computation are formed by combining the KV cache from the UE at time $\tau(t)$ with the KV representations derived from the AE at time $t$, which can be expressed as:

$$
\displaystyle\tilde{K}^{l}(t)
$$
 
$$
\displaystyle=[K^{l}_{scene}(\tau(t))\;\|\;K^{l}_{act}(t)],
$$
$$
\displaystyle\tilde{V}^{l}(t)
$$
 
$$
\displaystyle=[V^{l}_{scene}(\tau(t))\;\|\;V^{l}_{act}(t)],
$$

where $[\cdot\|\cdot]$ denotes concatenation along the sequence dimension, and all keys and values share the same embedding dimensionality $d$. The joint attention is then computed as

$$
\mathrm{Attn}^{l}(t)=\mathrm{softmax}\!\left(\frac{Q^{l}_{act}(t)\tilde{K}^{l}(t)^{\top}}{\sqrt{d}}\right)\tilde{V}^{l}(t)\ .
$$

The joint attention and asynchronous inference constitute the core characteristics of AutoMoT. By allowing action inference to reuse scene representations that are updated at a different temporal frequency, the proposed framework enables decision-making and trajectory planning to operate with a higher execution frequency than scene understanding, while remaining grounded in real-time perceptual inputs. This design aligns with the real-time requirements of real-world autonomous driving.

## 4 Experiments

### 4.1 Experimental Setup

#### Datasets.

For the reasoning tasks, we evaluate the general performance of all models on both autonomous driving benchmarks and general-domain datasets, including OmniDrive [^38], ScienceQA, and FigureQA. For action-level tasks, AutoMoT is primarily trained on three datasets: nuSync, which is annotated and curated in this work for decision-making, nuScenes [^2], and the CARLA-Garage dataset [^15] for trajectory planning. We follow the original training and evaluation protocols provided by the trajectory planning benchmarks. Additionally, as part of our ablation study, we fine-tune the understanding expert of AutoMoT exclusively on two autonomous driving VQA datasets, LingoQA [^32] and CODA-LM [^3].

#### Benchmarks and Metrics.

Scene understanding performance is evaluated on the LingoQA [^32] benchmark using its native metric, Lingo-Judge, as well as on other AD-tailored and general VQA datasets using GPT-based scores. We further evaluate the open-loop performance of AutoMoT on the nuScenes [^2] benchmark, using average accuracy (AA) for decision-making, as well as L2 distance and collision rate for trajectory planning. Closed-loop performance is assessed on the Bench2Drive [^18] benchmark following the officially provided evaluation metrics.

#### Implementation Details.

Each action token corresponds to 0.5 seconds of motion prediction. The AE predicts a sequence of action tokens to decode coarse future trajectories, which are further refined by a diffusion-based planner. For action policy learning, we adopt a learning rate ranging from $1\times 10^{-4}$ to $2\times 10^{-5}$ and employ the Fully Sharded Data Parallel (FSDP) training strategy. The action expert predicts 6 trajectory points and 20 route points, with $\lambda=0.5$. The model is trained using 8 NVIDIA A100 GPUs. Additional details are provided in the Supplementary Material.

Table 2: Comparison of the Open-loop planning in nuScenes. The ST-P3 evaluation protocol is used by default.

<table><tbody><tr><th rowspan="2">Method</th><td rowspan="2">Ego Status</td><td colspan="3">Finetuning</td><td colspan="4">L2 (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td colspan="4">Collision (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>Understanding</td><td>Decision</td><td>Planning</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><th>UniAD <sup><a href="#fn:11">11</a></sup></th><td>Vector</td><td>-</td><td>-</td><td>✓</td><td>0.44</td><td>0.67</td><td>0.96</td><td>0.69</td><td>0.04</td><td>0.08</td><td>0.23</td><td>0.12</td></tr><tr><th>VAD <sup><a href="#fn:21">21</a></sup></th><td>Vector</td><td>-</td><td>-</td><td>✓</td><td>0.17</td><td>0.34</td><td>0.60</td><td>0.37</td><td>0.07</td><td>0.10</td><td>0.24</td><td>0.14</td></tr><tr><th>Ego-MLP <sup><a href="#fn:26">26</a></sup></th><td>Vector</td><td>-</td><td>-</td><td>✓</td><td>0.15</td><td>0.32</td><td>0.59</td><td>0.35</td><td>0.00</td><td>0.27</td><td>0.85</td><td>0.37</td></tr><tr><th>DriveTransformer-Large <sup><a href="#fn:19">19</a></sup></th><td>Vector</td><td>-</td><td>-</td><td>✓</td><td>0.16</td><td>0.30</td><td>0.55</td><td>0.33</td><td>0.01</td><td>0.06</td><td>0.15</td><td>0.07</td></tr><tr><th>AutoVLA <sup><a href="#fn:49">49</a></sup></th><td>Text</td><td>✓</td><td>-</td><td>✓</td><td>0.21</td><td>0.38</td><td>0.60</td><td>0.40</td><td>0.13</td><td>0.18</td><td>0.28</td><td>0.20</td></tr><tr><th>ORION(Chat-B2D) <sup><a href="#fn:7">7</a></sup></th><td>-</td><td>✓</td><td>-</td><td>✓</td><td>0.17</td><td>0.31</td><td>0.55</td><td>0.34</td><td>0.05</td><td>0.25</td><td>0.80</td><td>0.37</td></tr><tr><th>RoboTron-Drive <sup><a href="#fn:13">13</a></sup></th><td>-</td><td>✓</td><td>-</td><td>✓</td><td>0.14</td><td>0.30</td><td>0.57</td><td>0.33</td><td>0.03</td><td>0.12</td><td>0.63</td><td>0.26</td></tr><tr><th>OpenDrive-VLA <sup><a href="#fn:48">48</a></sup></th><td>Text</td><td>✓</td><td>-</td><td>✓</td><td>0.15</td><td>0.31</td><td>0.55</td><td>0.33</td><td>0.01</td><td>0.08</td><td>0.21</td><td>0.10</td></tr><tr><th>OmniDrive <sup><a href="#fn:38">38</a></sup></th><td>Vector</td><td>✓</td><td>-</td><td>✓</td><td>0.14</td><td>0.29</td><td>0.55</td><td>0.33</td><td>0.00</td><td>0.13</td><td>0.78</td><td>0.30</td></tr><tr><th>EMMA <sup>†</sup> <sup><a href="#fn:14">14</a></sup></th><td>Text</td><td>✓</td><td>-</td><td>✓</td><td>0.14</td><td>0.29</td><td>0.54</td><td>0.32</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><th>SpaceDrive <sup><a href="#fn:49">49</a></sup></th><td>Vector</td><td>✓</td><td>-</td><td>✓</td><td>0.15</td><td>0.29</td><td>0.51</td><td>0.32</td><td>0.04</td><td>0.18</td><td>0.49</td><td>0.23</td></tr><tr><th>OpenREAD <sup><a href="#fn:45">45</a></sup></th><td>Vector</td><td>✓</td><td>-</td><td>✓</td><td>0.17</td><td>0.34</td><td>0.56</td><td>0.36</td><td>0.04</td><td>0.08</td><td>0.22</td><td>0.11</td></tr><tr><th>DriveVLM-Dual <sup><a href="#fn:37">37</a></sup></th><td>Vector</td><td>✓</td><td>-</td><td>✓</td><td>0.15</td><td>0.29</td><td>0.48</td><td>0.31</td><td>0.05</td><td>0.08</td><td>0.17</td><td>0.10</td></tr><tr><th>OpenEMMA <sup><a href="#fn:42">42</a></sup></th><td>Text</td><td>-</td><td>-</td><td>-</td><td>1.45</td><td>3.21</td><td>3.76</td><td>2.81</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><th>AutoMoT (Ours)</th><td>Vector</td><td>-</td><td>✓</td><td>✓</td><td>0.14</td><td>0.29</td><td>0.54</td><td>0.32</td><td>0.01</td><td>0.06</td><td>0.15</td><td>0.07</td></tr></tbody></table>

### 4.2 Main Results

In this section, we present detailed quantitative comparisons between AutoMoT and representative prior and SOTA methods across both reasoning and action-level tasks. Due to space limitations, detailed decision-making results are reported in Appendix A.1.

#### Closed-Loop Planning Benchmark Results.

We first evaluate AutoMoT on a closed-loop evaluation benchmark and report the quantitative results in Table 1. It is clear that AutoMoT outperforms all VLM-augmented baseline methods and achieves SOTA performance in terms of both driving score (DS) and the success rate (SR). It is worth noting that SimLingo employs action dreamer–based simulation for data augmentation to increase the amount of training data, while AutoMoT is trained solely on the original dataset, yet dominates SimLingo [^34] in terms of both main metrics. We further explored the closed-loop driving performance with diffusion policy [^5] head, and reported the detailed quantitative results in Appendix A.3.

Table 3: Comparison of reasoning capabilities across both general-domain and autonomous driving–specific datasets. $\dagger$: Results are reproduced using the official checkpoints and evaluation environments.

| Method | LingoQA | OmniDrive | CODA-LM | TallyQA | InfoVQA |
| --- | --- | --- | --- | --- | --- |
| ReCogDrive | 67.20 | 0.82 | 5.90 | 69.60 | 75.80 |
| Robotron-Drive <sup>†</sup> | 59.20 | 0.82 | 6.20 | 63.40 | 42.60 |
| OpenEMMA | 48.00 | 0.43 | 4.80 | 80.00 | 71.40 |
| AutoMoT | 67.00 | 0.89 | 6.07 | 81.40 | 89.30 |

#### Open-Loop Planning Benchmark Results.

The open-loop planning performance of AutoMoT compared with various baselines is reported in Table 2. AutoMoT achieves competitive performance in terms of L2 displacement and attains SOTA results on collision rate. Notably, most existing methods adapt scene understanding to the autonomous driving domain by fine-tuning the VLM backbone, whereas only OpenEMMA and AutoMoT refrain from such domain-specific adaptation, yet exhibit a clear performance distinction in terms of L2 displacement. These results indicate that policy learning in VLA models plays a critical role in action-level tasks, extending beyond the inherent expertise of pre-trained VLMs. In contrast, fine-tuning the VLM backbone on AD datasets yields only marginal improvements in planning metrics. More importantly, such minor gains (e.g., a few centimeters in L2 displacement) may come at the cost of severe degradation in scene understanding capability due to catastrophic forgetting. To examine this trade-off more comprehensively, we further evaluate AutoMoT together with other open-source methods on both AD-specific and general-domain VQA datasets to assess their scene understanding performance.

#### General VQA Benchmark Results.

Observations from the planning benchmarks further motivate a deeper investigation into whether additional domain-specific adaptation of scene understanding on top of pre-trained VLMs is indeed beneficial for overall autonomous driving performance. To clarify this question, we evaluate AutoMoT against other open-source baseline approaches on a diverse set of VQA benchmarks spanning both autonomous-driving–specific and general-domain tasks, as summarized in Table 3. Although both ReCogDrive and Robotron-Drive fine-tune their VLM backbones on LingoQA, OmniDrive, and CODA-LM, the resulting improvements are either marginal or even inferior to AutoMoT, which keeps the VLM backbone frozen. For example, ReCogDrive only marginally outperforms AutoMoT by 0.2 on the Lingo-Judge metric, while both ReCogDrive and Robotron-Drive underperform AutoMoT on the perception task of OmniDrive. More importantly, their performance on general-domain VQA benchmarks degrades substantially after fine-tuning, falling well below that of OpenEMMA and AutoMoT. Taken together with the analysis on planning benchmarks, these results suggest that fine-tuning the VLM backbone on AD-tailored scene understanding tasks provides only limited benefits for subsequent planning behaviors. Such gains are often marginal and accompanied by overfitting to specific benchmarks, leading to catastrophic forgetting and degraded generalization. In the following section, we further investigate the functional boundaries of pre-trained VLMs in autonomous driving, examining whether domain-specific fine-tuning is necessary across different AD tasks.

### 4.3 Performance Boundary of Pretrained Backbone

In this section, we aim to investigate when and to what extent AD-tailored fine-tuning is beneficial under a more controlled and fair setting, as direct comparisons with existing methods are often confounded by various factors. To this end, we fine-tune the VLM backbone of AutoMoT on two autonomous driving datasets: LingoQA [^32] and the counterfactual reasoning subset of OmniDrive [^38]. The former is widely used for scene understanding in the autonomous driving domain, while the latter is closely related to planning performance, as it shares the same visual inputs as the nuScenes [^2] benchmark and its question prompts explicitly contain trajectory-related information. We then evaluate the fine-tuned model on the test splits of these two datasets, together with five additional general-domain knowledge benchmarks, ScienceQA [^31], FigureQA [^23], TallyQA [^1], InfographicVQA [^33], and VizWiz [^9], as summarized in Table 4.

As shown by the quantitative results, fine-tuning the VLM backbone yields marginal improvements on scene understanding performance in LingoQA, but leads to substantial gains on the counterfactual planning task. This suggests that pre-trained VLMs can already support competitive multi-task scene understanding through semantic prompting alone, whereas fine-tuning remains essential for action-level tasks such as trajectory planning. Notably, the impact of fine-tuning on general-domain reasoning is highly task-dependent. On datasets with relatively simple answer spaces, such as ScienceQA and FigureQA, fine-tuning results in only minor performance changes, indicating that basic recognition and short-form reasoning capabilities are largely preserved. In contrast, for more complex VQA benchmarks that require compositional reasoning and multi-step inference, such as TallyQA, InfographicVQA, and VizWiz, performance degrades substantially after fine-tuning. For instance, accuracy on TallyQA drops from 81.40 to 52.40, and on InfographicVQA from 89.30 to 50.20, corresponding to an almost 50% reduction relative to the pre-trained baseline. These results demonstrate that domain-specific fine-tuning mainly degrades high-level reasoning ability, providing clear evidence of catastrophic forgetting when VLMs are directly adapted to the autonomous driving domain.

Beyond validating our design choice, these results prompt a rethinking of the role of pre-trained VLMs in autonomous driving systems. Rather than uniformly adapting VLMs across all task levels, our findings suggest a clearer functional boundary: pre-trained VLMs are best suited for high-level scene understanding and reasoning, while task-specific adaptation should primarily focus on action-level components that operate under domain-specific constraints. In this regard, our design preserves the general intelligence of the scene understanding expert and transfers its reasoning knowledge to the action expert and the action refiner through joint attention sharing, enabling effective action-level learning without sacrificing general reasoning ability.

Table 4: Ablation study results of investigating the performance boundary of the pre-trained backbone. $\dagger$: System prompt is provided; $\ddagger$: Fine-tuned on autonomous driving datasets; L: Lingo-Judge; G:GPT-Score; A: Token Accuracy.

| Benchmark | Task Category | AutoMoT <sup>†</sup> | AutoMoT <sup>‡</sup> |
| --- | --- | --- | --- |
| LingoQA (L) | Scene Understanding | 67.00 | 67.20 |
| OmniDrive (G) | Counterfactual Planning | 18.20 | 67.80 |
| ScienceQA (A) | General Knowledge | 88.60 | 87.80 |
| FigureQA (A) | General Knowledge | 97.60 | 91.20 |
| TallyQA (A) | General Knowledge | 81.40 | 52.40 |
| InfographicVQA (G) | General Knowledge | 89.30 | 50.20 |
| VizWiz (G) | General Knowledge | 75.60 | 50.20 |

Table 5: Trajectory planning performance under synchronized and asynchronous settings. AutoMoT refers to the proposed model with the KV cache enabled, while AutoMoT-S denotes the synchronized variant that runs the understanding expert (UE) and action expert (AE) at the same frequency, without introducing temporal misalignment between UE and AE.

| Setting | L2@1s $\downarrow$ | L2@2s $\downarrow$ | L2@3s $\downarrow$ | L2 ${}_{\text{avg}}$ $\downarrow$ | Lat. (s) $\downarrow$ |
| --- | --- | --- | --- | --- | --- |
| AutoMoT-S | 0.140 | 0.290 | 0.537 | 0.322 | 0.38 |
| AutoMoT | 0.141 | 0.293 | 0.544 | 0.326 | 0.05 |

### 4.4 Asynchronous versus Synchronous Inference

In this ablation study, we investigate whether decoupling reasoning and action inference at different temporal resolutions degrades driving performance due to stale visual observations. To quantify this effect, we construct two dedicated validation sets by introducing controlled temporal offsets between the visual inputs processed by the understanding expert (UE) and the action expert (AE). The *decoupled* set contains asynchronous samples with one-step asynchrony (0.5 s) and two-step asynchrony (1.0 s), while the *coupled* set contains synchronized samples with zero temporal offset. We then compare AutoMoT (decoupled inference with a persistent KV cache) against a coupled variant AtuoMoT-S that disables the KV cache and runs UE and AE at the same frequency, recomputing all outputs at every step.

As shown in Table 5, AutoMoT (asynchronous inference with KV caching) achieves planning accuracy comparable to the synchronized variant AutoMoT-S, while substantially reducing inference latency. The performance gap between the two settings is negligible across all prediction horizons, with only a 1.24% increase in L2 ${}_{\text{avg}}$ ($0.322\rightarrow 0.326$). In contrast, AutoMoT reduces inference latency from 0.38 s to 0.05 s, corresponding to an 86.8% speedup ($7.6\times$ faster). These results demonstrate that the proposed asynchronous fast–slow inference mechanism significantly improves efficiency without meaningfully degrading planning accuracy.

## 5 Conclusion

We propose AutoMoT, an end-to-end autonomous driving framework that unifies reasoning and action generation within a single VLA model via a MoT architecture with joint attention sharing. AutoMoT preserves the general reasoning capability of the VLM backbone during action policy learning, while improving driving performance through a VLA-oriented diffusion-based action refiner. By executing reasoning and action asynchronously, AutoMoT enables efficient, fast–slow inference across different task components. Extensive evaluations on simulation and real-world benchmarks under both open- and closed-loop settings show that AutoMoT achieves competitive performance against state-of-the-art baselines, despite not fine-tuning the VLM backbone on AD-specific datasets. Moreover, experiments on both general-domain and AD-specific VQA benchmarks show that pre-trained VLMs already provide strong multi-task scene understanding through semantic prompting, whereas fine-tuning remains essential for action-level tasks in end-to-end autonomous driving.

## References

## Appendix A Appendix for AutoMoT.

### A.1 Decision-Making Benchmark Results

#### Decision Benchmark on NuSync Dataset.

In this section, we report the quantitative result of decision-making over the dataset constructed by ourselves, as mentioned in Section 3.2. The decision space consists of lateral actions, including *turn left*, *slight left*, *go straight*, *slight right*, *turn right*, and longitudinal actions, including *accelerate*, *slow*, *keep*, and *stop*, which contains consecutive meta-actions for three future time frames: 1s, 2s, 3s. Moreover, on top of this multi-frame decision-making dataset, we additionally construct an asynchronous version by decoupling the temporal alignment between semantic reasoning and action prediction. Specifically, the decoupled set contains asynchronous samples with one-step asynchrony (0.5 s) and two-step asynchrony (1.0 s). We then evaluate both AutoMoT (decoupled inference with a persistent KV cache) and a coupled variant (AutoMoT-S) that disables the KV cache and runs UE and AE at the same frequency, recomputing all outputs at every step, and the results are shown in Table 6. We observe that the accuracy gap remains negligible, indicating that KV-cache reuse preserves semantic and temporal coherence across timesteps. Importantly, this negligible loss in accuracy is accompanied by up to a 7.6 $\times$ increase in inference frequency compared to the synchronized setting, demonstrating the efficiency advantage of the proposed asynchronous VLA design.

Table 6: Decision-making accuracy under synchronized and asynchronous settings at different time horizons.

<table><thead><tr><th></th><th colspan="3">Lateral Acc. <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th colspan="3">Longitudinal Acc. <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th colspan="3">Joint Acc. <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th></tr><tr><th>Method</th><th>1s</th><th>2s</th><th>Avg</th><th>1s</th><th>2s</th><th>Avg</th><th>1s</th><th>2s</th><th>Avg</th></tr></thead><tbody><tr><th>AutoMoT-S</th><td>95.00%</td><td>83.20%</td><td>84.50%</td><td>77.38%</td><td>56.81%</td><td>62.28%</td><td>73.84%</td><td>46.77%</td><td>53.49%</td></tr><tr><th>AutoMoT</th><td>94.06%</td><td>82.69%</td><td>83.79%</td><td>77.57%</td><td>56.85%</td><td>62.38%</td><td>73.40%</td><td>46.36%</td><td>53.10%</td></tr></tbody></table>

#### Decision Benchmark on Senna Dataset.

In order to confirm the superiority of AutoMoT, we further evaluate decision-making performance on the meta-action benchmark constructed by Senna [^20]. lAccording to the original setting, Senna defines discrete meta-action labels by analyzing future trajectories, where longitudinal actions are categorized into *stop*, *accelerate*, *decelerate*, and *constant-speed* based on velocity variations, while lateral actions are categorized into *left/right turn*, *left/right lane change*, and *straight* based on lateral displacement and heading changes.

We fine-tune both AutoMoT and Senna on the same dataset while keeping their original hyperparameter settings, and report the results in Table 7. AutoMoT achieves higher decision accuracy than Senna, indicating stronger action policy learning capability in our action expert.

Table 7: Decision-making performance comparison on the Senna nuScenes benchmark.

| Model | Fine-tuned | Accuracy (%) |
| --- | --- | --- |
| Senna | ✓ | 88.47 |
| AutoMoT (Ours) | ✓ | 90.92 |

### A.2 Impact of Scene Understanding and Decision-Making.

In this section, we conduct additional ablation studies to systematically analyze the contributions of individual components in AutoMoT, including the scene understanding and decision-making (meta-action planning), and the asynchronous inference mechanism, with respect to overall driving performance. For consistency, all experiments are performed on the nuScenes benchmark, using the official trajectory planning metrics.

Table 8: Ablation study on the understanding and decision-making components of AutoMoT. AutoMoT-R denotes the variant with a randomly initialized VLM backbone. AutoMoT-P denotes the planning-only variant, where the action expert is trained without the decision-making (meta-action planning) objective.

| Method | L2@1s $\downarrow$ | L2@2s $\downarrow$ | L2@3s $\downarrow$ | L2 <sub>avg</sub> $\downarrow$ |
| --- | --- | --- | --- | --- |
| AutoMoT (Ours) | 0.14 | 0.29 | 0.54 | 0.32 |
| AutoMoT-R (w/o Pre-trained UE) | 0.16 | 0.33 | 0.60 | 0.36 |
| AutoMoT-P (w/o decision-making) | 0.14 | 0.30 | 0.58 | 0.34 |

#### Impact of the Scene Understanding.

To assess the importance of preserving the general intelligence of the VLM backbone, we replace the pre-trained Qwen3-VL-4B in the understanding expert with a randomly initialized counterpart and train it E2E on the trajectory planning task. As shown in Table 8, this from-scratch variant exhibits substantial performance drops across all planning horizons, indicating that the general-purpose knowledge and reasoning capabilities provided by the pre-trained understanding expert are crucial for stable and accurate planning. Notably, the degradation becomes more pronounced at longer horizons, suggesting that long-horizon trajectory planning relies heavily on high-quality scene understanding.

#### Impact of the Decision-Making.

Additionally, we keep all components unchanged but remove the decision-making objective from the action expert (AE), training it solely on trajectory planning to quantify the contribution of decision-making to overall driving performance. Quantitative results are reported in Table 8. Removing decision-making consistently degrades performance across all prediction horizons, with the average L2 displacement increasing to 0.34.

### A.3 Discussion of Planning Head.

Recently, generative planners such as diffusion policies [^5] have demonstrated strong potential for autonomous driving. In our framework, we implement the policy module as a diffusion-based policy built on the Diffusion Transformer (DiT). Instead of starting the reverse process from clustered trajectories [^50] or pure white noise [^5], we use the coarse trajectories predicted by the AE as informative priors and perform truncated reverse denoising to generate the final policy trajectories. This design provides a more reliable initialization and significantly accelerates inference.

Concretely, the AE trajectory proposals are perturbed with multiplicative Gaussian noise, formulated as

$$
\tau^{\prime}=(1+\epsilon_{\text{mul}})\odot\tau,
$$

where the longitudinal and lateral perturbations follow DiffusionDriveV2 [^50]. Based on the noisy trajectory samples, we construct temporal trajectory queries $Q_{\text{temp}}\in\mathbb{R}^{B\times M\times 2}$, and the spatial queries $Q_{\text{spatial}}\in\mathbb{R}^{B\times N\times 2}$, and concatenate them as:

$$
X=[Q_{\text{temp}}\|Q_{\text{spatial}}],
$$

and processed by a stack of $N$ DiT decoder blocks. The conditioning signal $c$, which integrates the diffusion timestep, the current ego state, and lower-dimensional state history, is injected into each block through adaptive layer normalization (AdaLN).

To effectively exploit heterogeneous information during denoising, the diffusion policy leverages two complementary sources: the latent decision states $h_{\text{de}}$ from the AE for decision-aware trajectory generation, and the BEV feature $F_{\text{bev}}$ from the vision encoder for spatial guidance. Existing diffusion planners, such as encoder–decoder architectures [^25] and cascading cross-attention decoders [^27], usually rely on unstructured initialization and implicit attention balancing across heterogeneous modalities, which may weaken the structural guidance carried by trajectory priors. To address this issue, we introduce a Mixture-of-Attention (MoA) mechanism, as illustrated in Fig. 4, to enable more effective multi-source fusion while preserving the meaningful information provided by the anchor trajectories.

Specifically, MoA adopts a main–bypass fusion design. In the main pathway, joint attention is computed over three sources: self-attention among temporal and spatial queries, cross-attention to BEV features, and cross-attention to latent decision states. In addition, the contribution of latent decision states are modulated by a learnable factor $g=\tanh(\gamma)$, enabling adaptive control over multi-frame meta-actions.

To further stabilize information propagation across diffusion stages, we introduce residual bypass pathways that preserve global contextual cues from different modalities. Specifically, $R_{\text{bev}}$ is obtained by mean pooling over BEV features, while $R_{\text{reason}}$ is obtained by attention pooling over reasoning tokens. The final fused representation is computed as

$$
X^{\prime}=X+\alpha\cdot\big(O_{\text{main}}+\sigma(\beta_{b})R_{\text{bev}}+\sigma(\beta_{r})R_{\text{reason}}\big),
$$

where $O_{\text{main}}$ denotes the fused output of the main pathway, $\alpha$ is a scaling factor derived from AdaLN conditioned on $c$, and $\sigma(\beta_{b})$ and $\sigma(\beta_{r})$ are learnable gating coefficients.

The resulting decoder representations are used to predict both the future temporal and spatial trajectory, consistent with the formulations defined above. To train the diffusion policy, we minimize the $\ell_{1}$ prediction error between the generated trajectories and the expert ground truth:

$$
\displaystyle\mathcal{L}_{\text{traj}}^{\text{temp}}
$$
 
$$
\displaystyle=\mathbb{E}_{(o_{t},Y_{t}^{\text{temp}})\sim\mathcal{D}}\left[\frac{1}{M}\sum_{m=1}^{M}\left\|\hat{Y}_{t+m}-Y_{t+m}^{\text{temp}}\right\|_{1}\right],
$$
$$
\displaystyle\mathcal{L}_{\text{traj}}^{\text{spatial}}
$$
 
$$
\displaystyle=\mathbb{E}_{(o_{t},Y_{t}^{\text{spatial}})\sim\mathcal{D}}\left[\frac{1}{N}\sum_{n=1}^{N}\left\|\bar{Y}_{t+n}-Y_{t+n}^{\text{spatial}}\right\|_{1}\right].
$$
![[x4 10.png|Refer to caption]]

Refer to caption

Table 9: Comparison of different policy heads on the CARLA Bench2Drive leaderboard. DS and SR denote Driving Score and Success Rate, respectively. For the diffusion head, we report both the best run ($*$) and the average performance ($\dagger$) over multiple runs.

| Method | DS $\uparrow$ | SR(%) $\uparrow$ |
| --- | --- | --- |
| SimLingo [^34] | 85.07 | 67.27 |
| AutoMoT (MLP head) | 87.34 | 70.00 |
| AutoMoT <sup>∗</sup> (Diffusion head, Best) | 88.75 | 71.36 |
| AutoMoT <sup>†</sup> (Diffusion head, Avg.) | 85.84 | 66.21 |

We report a quantitative comparison of different policy heads on Bench2Drive in Table 9. When equipped with a diffusion head, AutoMoT achieves the highest peak performance, reaching a Driving Score of 88.75 and a Success Rate of 71.36% in the best run, outperforming both SimLingo and default setting of AutoMoT. However, its average performance across multiple runs drops to 85.84 DS and 66.21% SR, indicating substantially higher variance in closed-loop evaluation. We attribute this performance gap to the stochastic nature of diffusion-based action generation. While diffusion policies may achieve higher peak performance, their inherent randomness can be amplified in closed-loop driving, where small trajectory deviations accumulate over time and lead to markedly different outcomes. Consequently, the diffusion head demonstrates a higher performance ceiling but reduced stability. In contrast, the MLP head produces more consistent closed-loop behavior, achieving 87.34 DS and 70.00% SR with stronger robustness across runs. Therefore, despite the higher peak performance of the diffusion head, we adopt the MLP head as the default policy head in our final design to ensure stable reproducibility.

[^1]: Tallyqa: answering complex counting questions. In Proceedings of the AAAI conference on artificial intelligence, Vol. 33, pp. 8076–8084. Cited by: §4.3.

[^2]: Nuscenes: a multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11621–11631. Cited by: §4.1, §4.1, §4.3.

[^3]: Automated evaluation of large vision-language models on self-driving corner cases. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 7817–7826. Cited by: §4.1.

[^4]: Vadv2: end-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243. Cited by: §2.1.

[^5]: Diffusion policy: visuomotor policy learning via action diffusion. The International Journal of Robotics Research 44 (10-11), pp. 1684–1704. Cited by: §A.3, §2.1, §4.2.

[^6]: Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683. Cited by: §2.2.

[^7]: Orion: a holistic end-to-end autonomous driving framework by vision-language instructed action generation. arXiv preprint arXiv:2503.19755. Cited by: §1, §2.2, Table 1, Table 2.

[^8]: MindDrive: a vision-language-action model for autonomous driving via online reinforcement learning. arXiv preprint arXiv:2512.13636. Cited by: Table 1.

[^9]: VizWiz grand challenge: answering visual questions from blind people. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Cited by: §4.3.

[^10]: Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (Eds.), Vol. 33, pp. 6840–6851. Cited by: §2.1.

[^11]: Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 17853–17862. Cited by: §1, §2.1, Table 1, Table 2.

[^12]: MoTVLA: a vision-language-action model with unified fast-slow reasoning. arXiv preprint arXiv:2510.18337. Cited by: §2.2.

[^13]: RoboTron-drive: all-in-one large multimodal model for autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 8011–8021. Cited by: Table 2.

[^14]: Emma: end-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2410.23262. Cited by: §1, Table 2.

[^15]: Hidden biases of end-to-end driving models. In Proc. of the IEEE International Conf. on Computer Vision (ICCV), Cited by: §4.1.

[^16]: Hidden biases of end-to-end driving models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 8240–8249. Cited by: §1, Table 1.

[^17]: Driveadapter: breaking the coupling barrier of perception and planning in end-to-end autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7953–7963. Cited by: Table 1.

[^18]: Bench2drive: towards multi-ability benchmarking of closed-loop end-to-end autonomous driving. Advances in Neural Information Processing Systems 37, pp. 819–844. Cited by: §4.1.

[^19]: DriveTransformer: unified transformer for scalable end-to-end autonomous driving. In International Conference on Learning Representations (ICLR), Cited by: Table 1, Table 2.

[^20]: Senna: bridging large vision-language models and end-to-end autonomous driving. arXiv preprint arXiv:2410.22313. Cited by: §A.1, §1, §2.2.

[^21]: Vad: vectorized scene representation for efficient autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 8340–8350. Cited by: §1, §2.1, Table 2.

[^22]: Alphadrive: unleashing the power of vlms in autonomous driving via reinforcement learning and reasoning. arXiv preprint arXiv:2503.07608. Cited by: §1.

[^23]: Figureqa: an annotated figure dataset for visual reasoning. arXiv preprint arXiv:1710.07300. Cited by: §4.3.

[^24]: SpaceDrive: infusing spatial awareness into vlm-based autonomous driving. arXiv preprint arXiv:2512.10719 2. Cited by: Table 1.

[^25]: Recogdrive: a reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052. Cited by: §A.3, §1, §2.2, Table 1.

[^26]: Is ego status all you need for open-loop end-to-end autonomous driving?. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14864–14873. Cited by: Table 2.

[^27]: Diffusiondrive: truncated diffusion model for end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 12037–12047. Cited by: §A.3, §1, §2.1, Table 1.

[^28]: Hybrid-prediction integrated planning for autonomous driving. IEEE Transactions on Pattern Analysis and Machine Intelligence. Cited by: §1.

[^29]: BridgeDrive: diffusion bridge policy for closed-loop trajectory planning in autonomous driving. arXiv preprint arXiv:2509.23589. Cited by: §2.1.

[^30]: ReasonPlan: unified scene prediction and decision reasoning for closed-loop autonomous driving. arXiv preprint arXiv:2505.20024. Cited by: Table 1.

[^31]: Learn to explain: multimodal reasoning via thought chains for science question answering. In The 36th Conference on Neural Information Processing Systems (NeurIPS), Cited by: §4.3.

[^32]: Lingoqa: visual question answering for autonomous driving. In European Conference on Computer Vision, pp. 252–269. Cited by: §4.1, §4.1, §4.3.

[^33]: Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 1697–1706. Cited by: §4.3.

[^34]: Simlingo: vision-only closed-loop autonomous driving with language-action alignment. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 11993–12003. Cited by: Table 9, §2.2, Table 1, §4.2.

[^35]: J. Song, C. Meng, and S. Ermon Denoising diffusion implicit models. In International Conference on Learning Representations, Cited by: §2.1.

[^36]: Don’t shake the wheel: momentum-aware planning in end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 22432–22441. Cited by: Table 1.

[^37]: DriveVLM: the convergence of autonomous driving and large vision-language models. In Conference on Robot Learning, pp. 4698–4726. Cited by: §1, §2.2, Table 2.

[^38]: Omnidrive: a holistic llm-agent framework for autonomous driving with 3d perception, reasoning and planning. CoRR. Cited by: §4.1, §4.3, Table 2.

[^39]: Alpamayo-r1: bridging reasoning and action prediction for generalizable autonomous driving in the long tail. arXiv preprint arXiv:2511.00088. Cited by: §1, §2.2.

[^40]: Para-drive: parallelized architecture for real-time autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15449–15458. Cited by: §2.1.

[^41]: Trajectory-guided control prediction for end-to-end autonomous driving: a simple yet strong baseline. Advances in Neural Information Processing Systems 35, pp. 6119–6132. Cited by: Table 1.

[^42]: Openemma: open-source multimodal model for end-to-end autonomous driving. In Proceedings of the Winter Conference on Applications of Computer Vision, pp. 1001–1009. Cited by: Table 2.

[^43]: DriveMoE: mixture-of-experts for vision-language-action model in end-to-end autonomous driving. arXiv preprint arXiv:2505.16278. Cited by: Table 1.

[^44]: Raw2Drive: reinforcement learning with aligned world models for end-to-end autonomous driving (in carla v2). arXiv preprint arXiv:2505.16394. Cited by: Table 1.

[^45]: OpenREAD: reinforced open-ended reasoning for end-to-end autonomous driving with llm-as-critic. arXiv preprint arXiv:2512.01830. Cited by: §1, §2.2, Table 2.

[^46]: Wisead: knowledge augmented end-to-end autonomous driving with vision-language model. arXiv preprint arXiv:2412.09951. Cited by: §1.

[^47]: Diff-refiner: enhancing multi-agent trajectory prediction with a plug-and-play diffusion refiner. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 10779–10785. Cited by: §2.1.

[^48]: Opendrivevla: towards end-to-end autonomous driving with large vision language action model. arXiv preprint arXiv:2503.23463. Cited by: §1, Table 2.

[^49]: AutoVLA: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. arXiv preprint arXiv:2506.13757. Cited by: §1, §2.2, Table 1, Table 2, Table 2.

[^50]: DiffusionDriveV2: reinforcement learning-constrained truncated diffusion modeling in end-to-end autonomous driving. arXiv preprint arXiv:2512.07745. Cited by: §A.3, §A.3.