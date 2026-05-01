---
title: "SpanVLA: Efficient Action Bridging and Learning from Negative-Recovery Samples for Vision-Language-Action Model"
source: "https://arxiv.org/html/2604.19710v1"
author:
published:
created: 2026-04-28
description:
tags:
  - "clippings"
---
<sup>1</sup> <sup>2</sup> <sup>3</sup>

Zewei Zhou Equal contribution. Email: zeweizhou@ucla.edu, yang.ruini@northeastern.edu    Ruining Yang <sup>⋆</sup>    Xuewei (Tony) Qi Corresponding author. Email: qixuewei@gmail.com    Yiluan Guo    Sherry X. Chen    Tao Feng    Kateryna Pistunova    Yishan Shen    Lili Su    Jiaqi Ma

###### Abstract

Vision-Language-Action (VLA) models offer a promising autonomous driving paradigm for leveraging world knowledge and reasoning capabilities, especially in long-tail scenarios. However, existing VLA models often struggle with the high latency in action generation using an autoregressive generation framework and exhibit limited robustness. In this paper, we propose SpanVLA, a novel end-to-end autonomous driving framework, integrating an autoregressive reasoning and a flow-matching action expert. First, SpanVLA introduces an efficient bridge to leverage the vision and reasoning guidance of VLM to efficiently plan future trajectories using a flow-matching policy conditioned on historical trajectory initialization, which significantly reduces inference time. Second, to further improve the performance and robustness of the SpanVLA model, we propose a GRPO-based post-training method to enable the VLA model not only to learn from positive driving samples but also to learn how to avoid the typical negative behaviors and learn recovery behaviors. We further introduce mReasoning, a new real-world driving reasoning dataset, focusing on complex, reasoning-demanding scenarios and negative-recovery samples. Extensive experiments on the NAVSIM (v1 and v2) demonstrate the competitive performance of the SpanVLA model. Additionally, the qualitative results across diverse scenarios highlight the planning performance and robustness of our model.

## 1 Introduction

End-to-end autonomous driving systems, which directly map raw sensor input to the final driving actions within a unified framework, have emerged as the mainstream paradigm of autonomous driving \[hu2023planning, jiang2023vad, song2024collaborative, zhou2024v2xpnp, zhou2025turbotrain, jia2024bench2drive, xu2025wod\]. By eliminating modular design, the end-to-end paradigm mitigates error accumulation and enables joint optimization toward the final planning task \[liao2024diffusiondrive, lei2025risk, gao2025rad, kirby2026driving, zhao2026bridgesim\]. However, conventional end-to-end systems rely on imitation learning of expert trajectories, lacking understanding and reasoning about the surrounding environment \[peng2025counterfactual, jia2023think\], especially in long-tail scenarios. Recently, Vision-Language-Action (VLA) models have attracted significant attention \[zhou2025autovla, xie2026latentvla, fu2025orion, ma2025dvlmadenhancediffusionvisionlanguagemodel\], which leverages the reasoning capabilities and extensive world knowledge of Vision-Language Models (VLM) to generate driving actions, improving the adaptability and scalability of end-to-end systems across diverse driving scenarios.

![[x1 27.png|Refer to caption]]

Figure 1: SpanVLA is a novel end-to-end autonomous driving framework, integrating the autoregressive reasoning and flow-matching action expert. It leverages a vision-language model (VLM) with chain-of-thought reasoning as the backbone, and introduces an efficient bridge to extract the multi-granular features from the VLM. Moreover, a flow-matching action expert is introduced to efficiently generate a continuous trajectory from the historical initialization. The model is trained via supervised fine-tuning to jointly learn reasoning and planning, and reinforcement fine-tuning with real-world negative-recovery samples further enhances the planning and robustness.

However, the existing VLA models still face two challenges: 1) High action generation latency with autoregressive decoding. How to bridge the vision, reasoning, and action space is the core question of the VLA model. Although directly generating action tokens within VLM \[zhou2025autovla, kim2024openvla\] simplifies the model structure and unifies the reasoning and planning, it requires autoregressive decoding with the large model, leading to high latency, especially for high-frequency control in autonomous driving. Thus, some methods \[jiang2024senna, xie2026latentvla, liao2025cot, xu2024vlm, pan2024vlp\] decouple the VLM from the end-to-end driving pipeline, using the VLM to provide supervision or high-level guidance while delegating low-level planning to a separate end-to-end module. However, such designs break the end-to-end optimization paradigm, increasing system complexity and training difficulty. 2) Only learning from positive samples with limited robustness. Current VLA models only rely on imitation learning from positive/expert trajectories \[liu2025takead, yu2025survey, peng2025counterfactual\], leading to limited robustness, especially for unseen and long-tail scenarios. However, real-world negative and takeover data, which capture negative behaviors that must be avoided, as well as recovery behavior from challenge scenarios, are often overlooked in datasets and models \[wang2024learning\]. Such negative-recovery data can provide targeted refinement signals, improving both performance and robustness.

To this end, we propose SpanVLA, a VLA model equipped with an efficient action bridging and learned from real-world negative-recovery samples, as illustrated in Fig.˜1. To overcome the linearly increasing latency of autoregressive decoding with respect to action length, we introduce an efficient action bridging with a flow-matching action expert. First, unlike prior designs that rely solely on the final-layer \[fu2025orion, li2025recogdrive\] or dense full-layer features \[wang2025alpamayo, wang2025vla\], our efficient action bridging aggregates multi-granular features from multiple sparse layers of the VLM, capturing different levels of information from raw vision to final reasoning. Then, based on the extracted feature, we introduce a flow-matching-based action expert to generate high-frequency, multi-modal trajectories. Instead of learning the flow directly from random noise \[li2025recogdrive, wang2025alpamayo\], our formulation conditions on historical trajectory embeddings and learns the transformation from past actions to future actions, improving both generation quality and efficiency. Furthermore, we introduce the negative-recovery samples into the reinforcement fine-tuning (RFT) with Group Relative Policy Optimization (GRPO), and design the negative penalty and recovery reward to facilitate the oriented policy optimization with these long-tail samples.

Extensive evaluation on NAVSIM (v1 and v2) \[dauner2024navsim, cao2025pseudo\] benchmark demonstrate the state-of-the-art performance of SpanVLA model. Empirical results validate that our efficient action bridging can significantly accelerate the action generation, and the learning with real-world negative and recovery samples further improves the planning performance. The main contributions are as follows:

1. We propose SpanVLA, a novel end-to-end autonomous driving framework that integrates a VLM backbone with an action bridging, leveraging the vision and reasoning guidance of VLM to efficiently plan future trajectory using a flow-matching policy conditioned on historical initialization.
2. We introduce a GRPO-based post-training method to enable the model not only to learn from positive driving demonstrations, but also to learn how to avoid the typical negative behavior and learn recovery behaviors.
3. We introduce a mReasoning, a real-world driving reasoning dataset, focusing on reasoning-demanding scenarios and negative-recovery samples.
4. We demonstrate that SpanVLA achieves state-of-the-art performance across the NAVSIM v1, v2 benchmarks with a significant inference time reduction.

## 2 Related Work

VLA Model for Autonomous Driving. The recent success of VLMs \[comanici2025gemini, jaech2024openai, bai2025qwen2\], characterized by strong reasoning capabilities and extensive world knowledge, has spurred their application in embodied agents, including autonomous vehicles \[hwang2024emma, cai2024driving, liu2026driveworld, zhang2026minddriver\] and robotics \[kim2016fine, intelligence2025pi, liu2025robopilot\], to efficiently generate high-quality continuous trajectories based on visual observations and language instructions. First, several approaches incorporate an additional VLM module into conventional end-to-end autonomous driving systems to provide high-level meta-action guidance \[xie2026latentvla, jiang2024senna, tian2024drivevlm\] or supervision \[pan2024vlp, xu2024vlm\]. While straightforward to implement, such designs hinder full end-to-end optimization. Thus, some approaches formulate the driving task as a language problem \[rowe2025poutine, zhou2025opendrivevla, xing2025openemma, mao2023gpt\] and directly reuse the world knowledge in language space, and represent the trajectory with text. Furthermore, more different representations for action \[zhou2025autovla\] and world \[tan2025latent, zeng2025futuresightdrive\] with an autoregressive framework are introduced to improve the reasoning and planning; however, those one-by-one predictions with a large model suffer from high generation latency and limited robustness with accumulated error \[tan2025flow, wang2025alpamayo\]. In this paper, we introduce a flow matching action expert with an efficient action bridging to speed up the generation and further fine-tune the model with real-world negative-recovery samples to improve the robustness.

#### 2.0.1 Action Bridging for VLA model.

Bridging the vision, reasoning, and action space is a critical challenge for VLA models in end-to-end autonomous driving \[hu2025vision, wang2025vla\]. To mitigate the long latency and error accumulation of unifying reasoning and planning with an autoregressive framework, an additional action expert \[wang2025alpamayo, li2025drivevla, dang2026drivefine, li2025recogdrive\] has demonstrated the promise to bridge the VLM features and generate a continuous trajectory efficiently. Typically, ReCogDrive \[li2025recogdrive\] extracts the final-layer feature of the VLM model and adopts a diffusion planner based on the VLM priors to generate the trajectory. To further improve the generation efficiency, Alpamayo \[wang2025alpamayo\] introduces a flow-matching planner based on the KV-cache from fully dense VLM layers. We employ an efficient action bridge for multiple sparse VLM layers to reduce redundant features and merge it with a flow-matching policy based on historical initialization, instead of pure noise.

#### 2.0.2 Reinforcement Fine-tuning.

RFT provides a promising post-training method to improve the performance and has been demonstrated in DeepSeek-R1 \[guo2025deepseek\]. Recently, several models \[li2025finetuning, gao2025rad, wang2025alpamayo, zhou2025autovla, fu2025minddrive, rawal2026nord, jiang2025irl, zou2025diffusiondrivev2\] have leveraged the RFT to further improve the driving performance based on safety, comfort, and other driving constraints or preferences. However, existing approaches rely on positive training data, overlooking the value of negative-recovery samples with undesirable behaviors and how to recover from them. Some methods \[liu2025takead, fang2025corevla\] exploit simulation takeover data with Direct Preference Optimization (DPO) \[rafailov2023direct\] to refine driving policies, but DPO essentially optimizes by increasing the likelihood of expert actions or decreasing the likelihood of negative behaviors, resulting in imitation or conservative avoidance \[shang2025drivedpo\]. In contrast, we employ the GRPO-based RFT method to leverage negative-recovery samples for exploration-driven optimization, enabling more adaptive and robust policy optimization.

## 3 SpanVLA

Our SpanVLA framework includes two main components, as shown in Fig.˜1.

1) VLM Backbone: The backbone processes visual and language inputs, and jointly generates reasoning tokens and physical action tokens (used during training) in a unified autoregressive manner.

2) Efficient Action Bridging: This module conditions on the KV-cache of sparse VLM layers and leverages the historical trajectory as initialization to generate continuous future trajectories via flow matching.

Training of SpanVLA is performed in:

a) Supervised Fine-Tuning (SFT). This stage leverages ground-truth trajectory and high-quality reasoning traces to jointly supervise planning and reasoning.

b) Reinforcement Fine-Tuning (RFT). This stage optimizes planning performance using task-specific reward functions over positive, negative, and recovery samples, improving the robustness and reducing unnecessary reasoning steps.

![[x2 25.png|Refer to caption]]

Figure 2: Overview of the efficient action bridging of the SpanVLA model. The VLM backbone leverages the autoregressive decoding to generate the reasoning results, and we introduce an action bridging to utilize the sparse KV cache to efficiently generate the continuous trajectory with historical initialization based on flow-matching, avoiding the linearly increasing latency of the autoregressive decoding with the long action length.

### 3.1 VLM Backbone

#### 3.1.1 Model Inputs.

SpanVLA supports mixed vision and textual prompt inputs. First, it takes multi-frame and multi-view image data as the vision inputs $\mathcal{V}^{t}=\bigcup_{i}\{c_{i}^{\tau}\}_{\tau=t-T_{h}}^{t}$ where $c_{i}$ is each camera stream, which consists of four history frames per camera stream sampled at 2hz. The default configuration includes three camera views, i.e., front, front-left, and front-right, and can be easily extended to additional camera streams. For language inputs $\mathcal{T}^{t}=\{I^{t},\{S_{\text{ego}}^{\tau}\}^{t}_{\tau=t-T_{h}}\}$, the system prompt describes the role and reasoning and planning task, and the high-level instruction $I$ (e.g., go straight and turn right), and the ego states $S_{\text{ego}}$ are inputted and tokenized as text. The former provides the intended directions, and the latter consists of ego acceleration, history velocity, and trajectory.

#### 3.1.2 Reasoning with Autoregressive Decoding.

During the inference stage, the VLM backbone performs autoregressive decoding to generate structured reasoning $\mathcal{T}_{\text{Reason}}$, which analyzes key scene elements and their states and produces the corresponding ego action, as Fig.˜2 illustrated. However, for simple scenarios, such additional “slow thinking” reasoning is often redundant and computationally inefficient. Following the AutoVLA \[zhou2025autovla\], our model adopts an adaptive reasoning mechanism that dynamically switches between fast thinking (action only) and slow thinking (explicit reasoning with chain-of-thought (CoT)). Once a predefined special token indicating action generation is emitted, the VLM terminates reasoning and delegates trajectory generation to the action expert.

During training, to enable the model to learn how to reason for planning, we introduce an additional discrete action generation task following reasoning in the VLM, which unifies reasoning and planning within the SFT, as following:

$$
[\mathcal{T}_{\text{Reason}},(A_{\text{token}})]=\mathrm{VLM}(\mathcal{V}^{t},\mathcal{T}^{t});A_{\text{token}}=\{a_{\text{token}}^{\tau}\}^{t+T_{f}}_{\tau=t},
$$

where we leverage the action codebook of \[zhou2025autovla\] to discretize the trajectories into action tokens $A_{\text{token}}$ from current timestamp $t$ to future $t+T_{f}$.

![[x3 25.png|Refer to caption]]

Figure 3: mReasoning data distribution and typical negative-recovery samples.

### 3.2 Efficient Action Bridging

We use a lightweight stack of Transformer layers $f_{\theta}$ to process the KV cache of designated VLM layers (from the sequence $[\mathcal{V}^{t},\mathcal{T}^{t},\mathcal{T}_{\text{Reason}}]$) as the flow-matching conditioning $\mathbf{c}_{\text{vlm}}$, as shown in Fig. 2. These layers maintain the same number of attention heads as the VLM but adopt a smaller embedding dimension to improve efficiency. During each flow-matching step, the action expert progressively attends to the VLM KV cache and predicts the vector field $\mathbf{v}_{\tau}$ using a query formed by combining historical trajectory embeddings with the time embedding of the current flow-matching time $\tau\in[0,1]$.

Unlike the prior approaches that start from pure Gaussian noise $\mathcal{N}(0,\mathbf{I})$ and denoise in the action space $\mathbf{a}\sim\mathcal{X}$ \[wang2025alpamayo, li2025recogdrive\], our method initializes from the historical trajectory embedding $\mathbf{a}_{\text{his}}$ with MLP layers and directly learns the transition from the historical action space to the future action space: $\mathbf{a}_{\text{his}}\xrightarrow{\mathbf{v}_{\tau}}\mathbf{a}$, improving performance and sampling efficiency. Moreover, to enhance robustness, we inject some Gaussian noise into the historical trajectory embeddings during training. In practice, we employ the optimal transport displacement map \[lipman2022flow\]. For training, we sample $\mathbf{a}_{\tau}=\tau\mathbf{a}+(1-\tau)\mathbf{a}_{\text{his}}$. In inference, the flow matching denoising with in each $\Delta\tau$ step is formalized as:

$$
\mathbf{a}_{t+\Delta\tau}=\mathbf{a}_{t}+\Delta\tau\cdot f_{\theta}(\mathbf{a}_{t},\tau,\mathbf{c}_{\text{vlm}}),
$$

### 3.3 Reasoning and Negative-Recovery Data

#### 3.3.1 Reasoning Data.

These data provide the primary supervision signals for the reasoning capability of the VLA model in SFT \[arai2025covla, liu2025omnireason\]. However, most existing open-source datasets \[park2025nuplanqa, qian2024nuscenes, wang2024omnidrive, sima2024drivelm\] are confined to relatively simple scenarios due to the simple database, where rich CoT annotations are not so necessary. In contrast, reasoning datasets that involve complex interactions or long-tail scenarios remain limited in scale. Moreover, current CoT annotations are typically verbose with redundant information \[zhou2025autovla, fu2025orion\], resulting in increased reasoning latency.

To this end, we introduce mReasoning, a new reasoning dataset from in-housing data, together with an automated CoT annotation pipeline. The dataset comprises 30K samples, focusing on scenarios with strong interactions, curated from expert human driving logs. It covers diverse and safety-critical scenarios, e.g., ego lane changes, lane bias, vulnerable road users (VRU), construction zones, and stop signs, as shown in Fig.˜3. For the annotation pipeline, we adopt the Gemini-3-Pro model \[gemini3report2025\] as the backbone, enabling more compact target models to distill both knowledge and structured reasoning capabilities from the advanced model. During CoT generation, the pipeline first identifies and analyzes critical elements in each scenario based on a predefined element list and state analysis schema. Only elements that directly influence the ego action are retained, while irrelevant factors are filtered out to reduce redundancy. Conditioned on the structured analysis of all critical elements, the system then selects the optimal longitudinal and lateral ego actions according to a predefined action space. Finally, the reasoning is consolidated into a concise reasoning trace, which serves as the CoT supervision signal for VLA models.

#### 3.3.2 Negative-Recovery Data.

We further curate a negative-recovery subset (3K + 3K scenarios) from our in-house dataset for mReasoning, consisting of suboptimal real-world ego trajectories and their corresponding expert corrections. These samples are collected from early-stage exploratory real-world testing of the same scenario list as the 30K reasoning data. Additional dataset details and illustrative examples are provided in the Supplementary Material.

### 3.4 Supervised Fine-tuning

The SFT aims to enable both the VLM backbone and the action expert with the capability of reasoning and planning.

#### 3.4.1 VLM Backbone.

As illustrated in Eq.˜1, we introduce additional action tokens $A_{\text{token}}=\{a_{\text{token}}^{\tau}\}^{t+T_{f}}_{\tau=t}$ during training to better align reasoning and planning. The model is thus trained to generate reasoning tokens $\mathcal{T}_{\text{Reason}}=\{\text{text}^{i}_{\text{token}}\}_{i=1}^{L}$ followed by action tokens in a unified sequence. Following \[zhou2025autovla\], we further equip the model with adaptive reasoning ability. Specifically, slow-thinking samples consist of CoT reasoning concatenated with the corresponding action token sequence, while fast-thinking samples contain only the action token sequence. All training data are curated with ground-truth assistant responses to ensure high-quality supervision for the VLM. Thus, the loss function for the output sequence $o=[\mathcal{T}_{\text{Reason}},A_{\text{token}}]$ is defined as:

$$
\mathcal{L}_{\text{LM}}=-\frac{1}{N}\sum\limits_{i=1}^{N}\log p_{\theta}(o_{i}\mid o_{<i},\mathcal{V}^{t},\mathcal{T}^{t}),\mathcal{L}_{\text{Action}}=-\frac{1}{T}\sum\limits_{i=L+1}^{L+T}\log p_{\theta}(o_{i}\mid o_{<i},\mathcal{V}^{t},\mathcal{T}^{t})
$$

where $N=L+T$, and $p_{\theta}$ denotes the predicted distribution with policy $\theta$.

Action Bridging. We adopt a conditional flow matching loss for action bridging, which encourages the model to learn the optimal transport path from the historical action space to the future planning action space:

$$
\mathcal{L}_{\text{FM}}=\mathbb{E}_{\tau\in[0,1],\mathbf{c}_{\text{vlm}}\in[\mathcal{V}^{t},\mathcal{T}^{t},\mathcal{T}_{\text{Reason}}]}\|f_{\theta}(\mathbf{a}_{\tau},\tau,\mathbf{c}_{\text{vlm}})-\mathbf{v}_{\tau}(\mathbf{a}_{\tau})\|^{2},
$$

where the $\tau$ is sampled by shifted normal distribution during training, and the target vector filed is $\mathbf{v}_{\tau}(\mathbf{a}_{\tau})=\frac{d\mathbf{a}_{\tau}}{d\tau}=\mathbf{a}-\mathbf{a}_{\text{his}}$. Moreover, following Alpamayo \[wang2025alpamayo\], we stop gradients for both the VLM backbone and the bridging module to preserve their learned knowledge, preventing gradients from the action bridging from disrupting the VLM representations. In practice, we train the VLM backbone first, and then finetune the action bridging individually.

### 3.5 Reinforcement Fine-tuning with Negative-Recovery Samples

To further improve the robustness and performance of SpanVLA, we propose a reinforcement fine-tuning framework that jointly trains on positive, negative, and recovery samples, together with a negative-behavior penalty and recovery-behavior reward that provide direct learning signals on complex and long-tail cases. We follow the RFT strategy of \[wang2025alpamayo\] to just finetune the VLM backbone in the RFT stage. Moreover, we adopt group relative policy optimization (GRPO) \[shao2024deepseekmath\], which is stable in practice and naturally fits the multi-modality of planning \[jiang2025alphadrive\]. Given a scenario query $q=(\mathcal{V}^{t},\mathcal{T}^{t})$, we sample a group of $G$ candidate outputs $\mathcal{O}=\{o_{1},\dots,o_{G}\}$ from the old policy $\pi_{\theta_{\mathrm{old}}}$, and optimize the current policy $\pi_{\theta}$ using the group-relative advantage:

$$
{\scriptsize\mathcal{J}_{\text{GRPO}}(\theta)=\mathbb{E}_{q,\{o_{i}\}\sim\pi_{\theta_{\text{old}}}(\mathcal{O}\mid q)}\left[\frac{1}{G}\sum_{i=1}^{G}\left(\mathcal{J}^{R}_{i}-\beta\mathbb{D}_{\text{KL}}(\pi_{\theta}\|\pi_{\text{ref}})\right)\right],}
$$
 
$$
\mathcal{J}_{i}^{R}=\min\left(\frac{\pi_{\theta}(o_{i}|q)}{\pi_{\theta_{\text{old}}}(o_{i}|q)}\mathrm{Adv}_{i},\ \text{clip}\left(\frac{\pi_{\theta}(o_{i}|q)}{\pi_{\theta_{\text{old}}}(o_{i}|q)},1-\epsilon,1+\epsilon\right)\mathrm{Adv}_{i}\right),\mathrm{Adv}_{i}=\frac{r_{i}-\text{mean}(\{r_{j}\}_{j=1}^{G})}{\text{std}(\{r_{j}\}_{j=1}^{G})},
$$

where $r_{i}$ represents the reward corresponding to sample $o_{i}$; $\epsilon$ and $\beta$ are hyperparameters controlling the clipping range and the weight of the KL-divergence; and $\pi_{\text{ref}}$ denotes the reference policy obtained from the SFT stage.

#### 3.5.1 RFT Reward.

Each training sample is labeled as $s\in\{\text{pos},\text{neg},\text{rec}\}$ according to the type of its ground-truth trajectory.

1) Positive Samples: The ground truth corresponds to expert driving trajectories. We adopt the Predictive Driver Model Score (PDMS) series \[dauner2024navsim, cao2025pseudo\] as the driving reward $r_{\text{Driving}}$ which provides a comprehensive evaluation of driving performance, including safety, comfort, efficiency, and other metrics. For positive cases, the ground-truth trajectory primarily serves as a reference for ego progress, encouraging the policy to maintain efficient and goal-directed behavior.

2) Negative Samples: We also use PDMS as the primary driving reward $r_{\text{Driving}}$. However, ego progress is not treated as a reference objective, since the ground truth reflects suboptimal behavior. To discourage the policy from reproducing similar undesirable actions, we introduce an additional negative-behavior penalty $r_{\text{Negative}}$. This penalty is formulated as an L2-based shaping term that penalizes trajectories close to the negative ground-truth trajectory. To avoid unbounded penalties that could push the policy toward extreme deviated action, but not an optimal one, we restrict the penalty to a bounded activation region.

3) Recovery Samples: The ground truth represents expert corrective actions. Given the inherent multi-modality of recovery behaviors in complex scenarios, we introduce a recovery-behavior reward $r_{\text{Recovery}}$ analogous to the negative shaping term. Specifically, within a bounded L2 region, trajectories that are closer to the expert recovery trajectory receive higher rewards, with the reward magnitude determined by the degree of similarity. This design encourages corrective behaviors while preserving flexibility in multi-modal solution spaces.

Moreover, we incorporate a reasoning length penalty $r_{\text{CoT}}$ into the reward to facilitate the adaptive reasoning as in \[zhou2025autovla\]. And we introduce an action–reasoning alignment penalty on the $r_{\text{CoT}}$ using a rule-based detection method. When inconsistencies are detected between the reasoning and the action, the CoT reward is overridden with a large fixed penalty. The final reward function is defined as:

$$
\begin{gathered}r=r_{\text{Driving}}-w_{\text{N}}r_{\text{Negative}}+w_{\text{R}}r_{\text{Recovery}}-\lambda_{\text{C}}r_{\text{CoT}},\\
w_{\text{N}}:\{\lambda_{\text{N}}\text{ if negative, else }0\},\quad w_{\text{R}}:\{\lambda_{\text{R}}\text{ if recovery, else }0\},\end{gathered}
$$

where the $\lambda_{\text{N}}$, $\lambda_{\text{R}}$, and $\lambda_{\text{C}}$ denote the weight for the negative, recovery, and reasoning reward item. More details are provided in the Supplementary Material.

## 4 Experiments

### 4.1 Experimental Setup

#### 4.1.1 Dataset.

We train the SpanVLA model with the mixed dataset across navtrain split \[dauner2024navsim\] (100K scenarios) of the nuPlan (Open-Scene) dataset \[karnchanachari2024towards, openscene2023\] and our mReasoning dataset (30K scenarios). Both of these datasets collect driving logs from Las Vegas, Boston, Pittsburgh, and Singapore with eight streams of camera data, object annotations, and HD maps. Moreover, the original NAVSIM only supports the nuPlan dataset, and we curate the data and develop the PDMS evaluation pipeline for our mReasoning dataset, supporting the RFT with the PDMS-based reward function in our dataset.

#### 4.1.2 Benchmark.

We evaluate the SpanVLA model on the NAVSIM v1 (navtest) \[dauner2024navsim\] and NAVSIM v2 (navtest and navhard) benchmarks \[cao2025pseudo\]. The NAVSIM v1 benchmark employs PDMS to assess the driving performance across comfort, safety, and efficiency. The NAVSIM v2 adopts EPDMS (extended PDMS) to add more driving consideration to the original PDMS. Notably, the navhard benchmark of NAVSIM v2 focuses on the most challenging scenarios and leverages pseudo closed-loop simulation to evaluate the driving performance and robustness within the two-stage testing \[cao2025pseudo\].

#### 4.1.3 Implementation Details.

We choose the Qwen2.5VL-3B model as the VLM backbone. For SFT, we employ Fully Sharded Data Parallel (FSDP) training on 8 NVIDIA A100 GPUs. The per-GPU batch size is set to 1, with a gradient accumulation step of 4 for training the VLM backbone. For the action bridge, the batch size is set to 16, and we selected the feature from VLM with an interval of two. For RFT, we adopt LoRA adapters \[hu2022lora\] for efficient fine-tuning. The learning rate for RFT is set to $3\times 10^{-5}$ and the group sample size is $64$. We perform a single policy update per step, resulting in a simplified objective that eliminates the need for clipping or maintaining an old policy. Additional implementation details are provided in the Supplementary Material.

![[x4 23.png|Refer to caption]]

Table 1: Comparison with SOTA methods on the NAVSIM v1 ( navtest ). PDMS (Predictive Driver Model Score), NC (No Collision), DAC (Drivable Area Compliance), EP (Ego Process), TTC (Time-To-Collision), Comf. (Comfort),