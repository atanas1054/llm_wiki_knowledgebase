---
title: "Unifying Language-Action Understanding and Generation for Autonomous Driving"
source: "https://arxiv.org/html/2603.01441v1"
author:
published:
created: 2026-04-05
description:
tags:
  - "clippings"
---
Xinyang Wang <sup>1,2,∗</sup>  Qian Liu <sup>2,∗</sup>  Wenjie Ding <sup>2,∗</sup>  Zhao Yang <sup>2,†</sup>  Wei Li <sup>2</sup>  
Chang Liu <sup>2</sup>  Bailin Li <sup>2</sup>  Kun Zhan <sup>2</sup>  Xianpeng Lang <sup>2</sup>  Wei Chen <sup>1</sup>  
<sup>1</sup> State Key Lab of CAD&CG, Zhejiang University   <sup>2</sup> Li Auto

###### Abstract

Vision-Language-Action (VLA) models are emerging as a promising paradigm for end-to-end autonomous driving, valued for their potential to leverage world knowledge and reason about complex driving scenes. However, existing methods suffer from two critical limitations: a persistent misalignment between language instructions and action outputs, and the inherent inefficiency of typical auto-regressive action generation. In this paper, we introduce LinkVLA, a novel architecture that directly addresses these challenges to enhance both alignment and efficiency. First, we establish a structural link by unifying language and action tokens into a shared discrete codebook, processed within a single multi-modal model. This structurally enforces cross-modal consistency from the ground up. Second, to create a deep semantic link, we introduce an auxiliary action understanding objective that trains the model to generate descriptive captions from trajectories, fostering a bidirectional language–action mapping. Finally, we replace the slow, step-by-step generation with a two-step coarse-to-fine generation method ($C2F$) that efficiently decodes the action sequence, saving $86\%$ inference time. Experiments on closed-loop driving benchmarks show consistent gains in instruction following accuracy and driving performance, alongside reduced inference latency.

<sup>†</sup> <sup>†</sup>

## 1 Introduction

![[x1 8.png|Refer to caption]]

(a)

End-to-end learning [^14] [^23] [^3] [^42] [^31] [^55] [^51] [^5] [^30] [^13] [^60] [^61] emerged as a dominant paradigm for developing autonomous driving systems, learning direct sensorimotor policies from raw sensor inputs to vehicle planning. While conventional end-to-end models excel at reactive control in familiar scenarios, they often struggle with complex reasoning, long-tail events, and human interaction [^40] [^46]. Vision-language-models (VLMs) [^16] [^11] have attracted significant attention in recent years due to their capable of leveraging extensive world knowledge and sophisticated reasoning capabilities and have been introduced into driving scenarios to enhance their generalization [^50] [^2] [^27] [^49] [^32] [^52] [^45] [^33] [^25] [^35] [^6] [^34] [^24] [^22]. Vision-Language-Action (VLA) methods extend VLMs to action genaration and have been explored in robotics [^36] [^1] and autonomous driving [^64] [^29] [^8] [^37] [^28] [^58] [^21] [^56] [^38] [^63] [^44] to generate physical feasible action based on visual and language inputs. VLAs promise a paradigm shift from learning implicit, reactive policies to developing agents capable of explicit reasoning and long-horizon planning. These models unlock superior interactivity and flexible instruction following, allowing language to serve as a powerful medium for conveying explicit rules and compositional logic, representing a critical step towards achieving more generalist and trustworthy autonomous agents.

At the heart of the VLA paradigm lies the ability to follow natural language instructions. Instruction following is a critical function for real-world deployment, enabling dynamic re-tasking and enhanced user trust through transparent interaction. However, a fundamental challenge plagues current VLA models: a persistent misalignment between language understanding and physical action [^37] [^10] [^49]. A model might correctly generate the decision change lane to the left, yet output a lane keep trajectory. This failure in instruction following undermines the core promise of VLAs and poses a significant barrier to their safety and reliability.

Several lines of research have emerged to address the misalignment problem. Some approaches focus on improving data collection techniques [^37] [^10] [^49], a strategy that sidesteps the fundamental modeling challenge. Others rely on post-hoc policy refinement using reinforcement learning [^64], which treats alignment as a corrective afterthought. A third vein attempts to align modalities through implicit distribution matching in latent space [^8], which lacks a direct, verifiable supervision. In contrast to these methods, we argue that this semantic gap necessitates an explicit bidirectional link woven into the primary supervised learning phase. To this end, we introduce LinkVLA, a novel VLA model designed specifically to strengthen this connection.

First, we establish a structural link at the architectural level. By unifying language instructions and trajectories into a shared discrete codebook, we eliminate the modality gap by design, forcing both concepts into a common representational ground. Second, it forges a deep semantic link through a novel bidirectional learning objective. We introduce an explicit action-understanding objective that compels the model to translate the planned actions back into descriptive text. This synergy enforces bidirectional consistency, ensuring the link between language and action is formative and verifiable. While this tightly coupled representation is powerful for alignment, its auto-regressive nature creates an inference bottleneck. To make our framework practical, we replace conventional step-by-step decoding with a coarse-to-fine, two-step generation process. This mechanism first generates a high-level structural outline of the trajectory and then refines this outline into the full, fine-grained path. As visualized in Figure 1, LinkVLA not only achieves a dramatic reduction in inference latency but also achieves consistent gains in instruction following accuracy and driving performance. Here are the main contributions:

- A unified tokenized framework that learns a shared codebook for language and action, structurally bridging the modality gap to enhance alignment.
- An explicit action understanding objective that enforces bidirectional semantic consistency.
- A coarse-to-fine action generation schema that drastically reduces inference latency.
- State-of-the-art performance on challenging closed-loop driving benchmarks, demonstrating significant gains in both instruction-following accuracy and driving ability.

## 2 Related Work

### 2.1 End-to-end Autonomous Driving

End-to-end autonomous driving has achieved tremendous success in recent years. UniAD [^14] adopts an ultimately planning-oriented design philosophy that integrates perception, prediction, and planning into a single network. VAD [^23] models driving scenarios into fully vectorized representations, thereby eliminate computationally intensive rasterized representations and resulting in a much faster running speed. ParaDrive [^51] proposes a fully parallel end-to-end architecture and conducts a comprehensive exploration of the design space of modular stacks. VADv2 [^3] discretizes the planning action space into a large planning vocabulary and leverages extensive driving demonstrations to learn the probability distribution of planning actions. SparseDrive [^42] unifies multiple tasks through sparse instance representation and propose a symmetric sparse perception module and a parallel motion planner. GenAD [^61] casts autonomous driving into a generative modeling problem. DriveTransformer [^20] adopts task parallelism and sparse representation to improve efficiency by working in a streaming manner. DiffusionDrive [^31] uses anchor to truncate the diffusion process, achieving better performance, higher inference efficiency. GoalFlow [^55] constrains the generated trajectories by introducing a goal point and use it as the condition of flow matching to generate multimodal trajectories. While proficient at reactive control within known operational domains, these methods display critical limitations in complex logical reasoning, long-tail event processing, and human-interactive task execution.

![[x3 6.png|Refer to caption]]

Figure 2: 4

### 2.2 VLM and VLA for Autonomous Driving

Early-stage VLMs primarily follow a language-centric paradigm that use visual question answering to discribe the action [^25] [^35] [^34]. DriveGPT4 [^56] employs Large Language Models (LLMs) to generate scene explanations and control signals. DriveVLM [^46] uses VLM to enhance scene understanding and planning capabilities. Drivemlm [^50] standardizes decision states to bridge the gap between the language decisions and the vehicle control signals and uses LLM to make driving decisions and explanations. EMMA [^15] capitalizes on Gemini’s multimodal capabilities by representing all non-sensor data – including navigation instructions, vehicle status, trajectories, and 3D positions – as textual sequences, thereby transferring pre-trained LLMs’ world knowledge to autonomous driving tasks.

VLA direct outputs action from raw input. Opendrivevla [^63] projects visual tokens into a unified semantic space to bridge the modality gap. DriveMoE [^57] introduces vision MoE and action MoE to handle diverse scenarios. Recogdrive [^29] injects the VLM’s learned driving priors into a diffusion planner and uses reinforcement learning to bridge the gap between language and action. DiffVLA [^21] proposes a hybrid sparse-dense diffusion policy empowered by a Vision-Language Model (VLM), enabling efficient multimodal driving behavior generation. However, a modality gap still exists, resulting in misalignment between language and action.

### 2.3 Language and Action Alignment

AutoVLA [^64] unifies reasoning with action generation and employs GRPO [^39] to improve planning performance. The step-by-step autoregressive action generation in AutoVLA makes it inefficienct in practical deployment. ORION [^8] bridges reasoning and action spaces by merging generative planners with VLM architectures, jointly optimizing VQA and planning. SimLingo [^37] focuses on aligning linguistic comprehension with driving actions. However, a persistent misalignment between language instructions and action still exist in ORION and SimLingo due to the modal gap. CAST [^10] leverage vision language models to create counterfactual labels to augment existing robot datasets. OmniDrive [^49] proposes a holistic vision-language dataset for aligning agent models with 3D driving tasks using counterfactual reasoning. In unified multimodal understanding and generation model, understanding and generation are found to be mutually beneficial to each other [^47] [^54] [^62]. Inspired by this, we propose a novel architecture that improves action generation ability by introducing an action understanding objective without the need for additional data curation.

## 3 Method

Our proposed model, LinkVLA, is a Vision-Language-Action model designed to enhance language-action alignment and inference efficiency in autonomous driving. Our methodology introduces three key innovations, as illustrated in Figure 2. First, we establish a unified auto-regressive framework that models language and action tokens within a single discrete space (Sec. 3.1). Second, to enhance semantic alignment, we introduce a novel action understanding objective that fosters a bidirectional mapping between language and trajectories (Sec. 3.2). Third, we replace slow, sequential decoding with an efficient, coarse-to-fine generation mechanism that drastically reduces inference latency (Sec. 3.3).

### 3.1 Unified Tokenization Framework

We posit that the language-action misalignment in autonomous driving is a direct consequence of an architectural schism between modalities. To eliminate this, our method is founded on the principle of unification: modeling the entire process—from understanding an instruction to generating a trajectory—within a single unified framework. Our approach maps both the language instruction $L$ and the action trajectory $\mathcal{T}$ into a unified sequence of discrete tokens, which is then processed by a VLM backbone. For language, we leverage the VLM’s existing tokenizer. For actions, which are inherently continuous, we devise a spatial-aware action tokenization scheme. Instead of regressing continuous values, our model predicts a sequence of action tokens from a tokenized codebook.

#### Unified Token Space.

Our approach is founded on a unified token space for language and action. We achieve this by first quantizing continuous trajectories: the local Bird’s-Eye-View (BEV) space is partitioned into a grid of $K_{action}$ cells, each defining a unique action token. A trajectory $\mathcal{T}=\{w_{1},\dots,w_{T}\}$ is thus converted into a sequence of action tokens ${A}=\{a_{1},\dots,a_{T}\}$ by mapping each waypoint to its corresponding cell. This action codebook ($\mathcal{C}_{action}$) is then merged with the model’s text vocabulary (size $K_{text}$) to form a single, unified codebook $\mathcal{C}$ of size $K=K_{text}+K_{action}$. The action token embeddings are learned end-to-end, forcing the model to map linguistic and spatial concepts into a shared representation and enabling a single VLM model to process both. During inference, each predicted action token is simply decoded back to the center of its corresponding grid cell.

#### Action Tokenization.

Naive tokenization, which tokenizes waypoints onto a uniform BEV grid using one-hot labels, suffers from two key issues. First, a uniform grid allocates resolution evenly, failing to provide the fine-grained precision essential for near-field control. Second, the hard assignment of one-hot labels discards the grid’s inherent spatial topology, complicating the learning of spatial priors. To mitigate these issues, we introduce two key refinements: Log Coordinate Transformation and Spatial Soft-labeling.

1) Log Coordinate Transformation. We devise a non-uniform quantization scheme that prioritizes precision near the ego-vehicle. This is achieved by first applying a non-linear transformation to the waypoint coordinates $(x,y)$ independently along each axis. Specifically, each coordinate $z\in\{x,y\}$ is transformed using a signed logarithmic function:

$$
z^{\prime}=\operatorname{sign}(z)\cdot\log(1+k\cdot|z|).
$$

Here, $k$ is a positive scaling factor that controls the size of the linear region around the origin. The resulting transformed space $(x^{\prime},y^{\prime})$ is then uniformly quantized to produce the action tokens.

2) Spatial Soft-labeling. To embed a physical prior about the continuity of the action space into our learning objective, we employ a spatial soft-labeling strategy. A standard one-hot target provides a discrete supervisory signal, but does not explicitly account for the spatial topology of our action grid. Our approach refines this by defining a smooth target distribution that acknowledges spatial adjacency.

Specifically, for a ground-truth token $a_{gt}$, we construct the target distribution $q(a)$ over all action tokens $a\in\mathcal{C}_{action}$ as a normalized 2D Gaussian centered on the coordinates of $a_{gt}$ with radius $R$:

$$
q(a)=\frac{1}{Z}\exp\left(-\frac{\|\text{pos}(a)-\text{pos}(a_{gt})\|_{2}^{2}}{2\sigma^{2}}\right),
$$

where $\text{pos}(a)$ maps an action token to its 2D coordinates in the spatial grid, $\|\cdot\|_{2}^{2}$ is the squared Euclidean distance, $\sigma$ is a hyperparameter controlling the spread of the distribution, and $Z$ is a normalization constant ensuring $\sum_{a\in\mathcal{C}_{action}}q(a)=1$. For action generation, the model’s predicted distribution $p(a)$ is then optimized to match this soft target using the cross-entropy loss:

$$
\mathcal{L}_{\text{generation}}=-\sum_{a\in\mathcal{C}_{action}}q(a)\log p(a).
$$

This objective encourages the model to assign probability mass not only to the correct token but also to its spatial neighbors, as dictated by the Gaussian shape. This fosters a locally smooth action manifold, making the model more robust to minor groundtruth errors. When incorporating chain-of-thought, an additional standard cross-entropy loss for language generation is added to this objective.

### 3.2 Unified Language-Action Understanding and Generation

To strengthen the link between language and behavior, we introduce a bidirectional training objective inspired by the duality of image captioning and text-to-image generation. In generative vision-language modeling, the tasks of generating text $L$ from an image $V$ ($p(L|V)$) and an image from text ($p({V}|{L})$) are reciprocal [^54] [^62] [^47]. Training a model on both objectives has been shown to produce a more robust and aligned joint embedding space.

![[x4 5.png|Refer to caption]]

Figure 3: Illustration of the action understanding ( Left ) and the action generation ( Right ).

We posit that a similar duality exists for the language-action mapping within VLA models. The conventional task is action generation, where a language instruction guides the prediction of an action sequence $A$ ($p(A|L)$). This is analogous to text-to-image synthesis. We propose its reciprocal: action understanding, where an executed action sequence $A$ is used to infer the original language instruction ($p(L|A)$). This mirrors image captioning, as the model must produce a linguistic description explaining the observed behavior. Crucially, both of these mappings are grounded in a shared visual context $V$, which provides the necessary environmental awareness. Formally, besides $\mathcal{L}_{generation}$ (Eq. 3), we introduce a reciprocal objective designed for reconstructing the language instruction ${L}$ given the vision and action inputs:

$$
\mathcal{L}_{\text{understanding}}=-\sum_{j}\log p(l_{j}|V,A,l_{<j}),
$$

where $l_{<j}$ represents the preceding ground-truth tokens.

The final training objective is a weighted sum of these two losses: $\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{generation}}+\lambda\mathcal{L}_{\text{understanding}}$, where $\lambda$ is a balancing hyperparameter. By forcing the model to solve this inverse problem, we enforce a bidirectional consistency within the shared embedding space. This process enriches the semantic grounding of the action tokens, ensuring they are intrinsically linked to descriptive linguistic concepts, thereby leading to better instruction following. In practice, both tasks are handled by the same decoder by simply swapping the roles of $L$ and $A$ as the prediction target.

### 3.3 Coarse-to-Fine Action Generation

Auto-regressive generation of a long trajectory of $T$ waypoints requires $T$ sequential forward passes, which is computationally expensive and introduces significant inference latency. To address this, we collapse the $T$ -step sequential dependency into a two-stage process: (1) endpoint prediction and coarse trajectory initialization, and (2) parallel trajectory refinement. Following [^37], our action outputs include both temporal speed waypoints and geometric path waypoints. As both are treated as point sequences and processed symmetrically throughout our framework, we will hereafter refer to them collectively as a trajectory and its constituents as waypoints for clarity.

#### Training with a Coarse Trajectory Prior.

Our coarse-to-fine inference is enabled by a carefully designed training objective. To directly predict endpoints, we put special tokens at the beginning of the decoder’s input sequence. During training, the ground-truth target sequence is reordered, $\{w_{T},w_{1},w_{2},...,w_{T-1}\}$, teaching the model to associate the special token with the endpoint prediction. For the refinement stage, we simulate coarse trajectory via linear interpolation using the groundtruth endpoint, quantize it into coarse waypoint tokens as model inputs, and train the model to map these coarse tokens to the corresponding fine-grained trajectory.

#### Coarse Trajectory Initialization.

During inference, we first establish a strong structural prior for the trajectory, which serves to guide the subsequent generation steps. With our modified training sequence, the model performs a single forward pass to predict specifically the final waypoint, $\hat{w}_{T}$. While this initial step is inspired by goal-point prediction methods [^55], our approach is fundamentally distinct as we integrate endpoint prediction and trajectory refinement within a single, unified transformer architecture.

Given the start point $w_{0}$ (the ego-vehicle’s origin at (0, 0)) and the predicted end point $w_{T}$, we construct the coarse trajectory, $\mathcal{W}_{coarse}$, via linear interpolation:

$$
w_{i}^{coarse}=w_{0}+\frac{i}{T}(w_{T}-w_{0})\quad\text{for }i\in\{1,\dots,T\}
$$

Then these waypoints are tokenized into trajectory tokens, as an initial scaffold for the subsequent refinement stage.

#### Parallel Trajectory Refinement.

The second inference step refines the coarse, straight-line path into a dynamically feasible trajectory. We formulate this as a structure-preserving refinement, where each coarse waypoint $w_{i}^{coarse}$ is mapped to its corresponding refined waypoint $w^{fine}_{i}$. Given tokenized coarse waypoints as initial input, LinkVLA predicts $T$ refined points in parallel. Conditioned by the vision-language context via cross-attention, the refined path respects lane boundaries, avoids obstacles, and adheres to the language instruction.

## 4 Experiments

Table 1: Main results and multi-ability in Bench2Drive. \* denote expert feature distillation.

<table><tbody><tr><td>Method</td><td colspan="4">Closed-loop metrics <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td colspan="6">Multi-Ability (%) <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td></td><td>gray!17DS</td><td>gray!17SR (%)</td><td>Efficiency</td><td>Comfort.</td><td>Merging</td><td>Overtake</td><td>Brake</td><td>Give-Way</td><td>Traffic-Sign</td><td>gray!17Mean</td></tr><tr><td><sup><a href="#fn:53">53</a></sup></td><td>gray!1740.70</td><td>gray!1715.00</td><td>54.26</td><td>47.80</td><td>16.18</td><td>20.00</td><td>20.00</td><td>10.00</td><td>6.99</td><td>gray!1714.63</td></tr><tr><td>TCP-ctrl*</td><td>gray!1730.47</td><td>gray!177.27</td><td>55.97</td><td>51.51</td><td>10.29</td><td>4.44</td><td>10.00</td><td>10.00</td><td>6.45</td><td>gray!178.23</td></tr><tr><td>TCP-traj*</td><td>gray!1759.90</td><td>gray!1730.00</td><td>76.54</td><td>18.08</td><td>8.89</td><td>24.29</td><td>51.67</td><td>40.00</td><td>46.28</td><td>gray!1734.22</td></tr><tr><td><sup><a href="#fn:18">18</a></sup></td><td>gray!1762.44</td><td>gray!1731.23</td><td>69.33</td><td>16.22</td><td>27.38</td><td>18.42</td><td>35.82</td><td>50.00</td><td>54.23</td><td>gray!1737.17</td></tr><tr><td><sup><a href="#fn:17">17</a></sup></td><td>gray!1764.22</td><td>gray!1733.08</td><td>70.22</td><td>16.01</td><td>28.82</td><td>26.38</td><td>48.76</td><td>50.00</td><td>56.43</td><td>gray!1742.08</td></tr><tr><td><sup><a href="#fn:59">59</a></sup></td><td>gray!1718.05</td><td>gray!170.00</td><td>48.45</td><td>22.63</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>4.35</td><td>gray!170.87</td></tr><tr><td><sup><a href="#fn:14">14</a></sup></td><td>gray!1740.73</td><td>gray!1713.18</td><td>123.92</td><td>47.04</td><td>8.89</td><td>9.33</td><td>20.00</td><td>20.00</td><td>15.43</td><td>gray!1714.73</td></tr><tr><td><sup><a href="#fn:14">14</a></sup></td><td>gray!1745.81</td><td>gray!1716.36</td><td>129.21</td><td>43.58</td><td>14.10</td><td>17.78</td><td>21.67</td><td>10.00</td><td>14.21</td><td>gray!1715.55</td></tr><tr><td><sup><a href="#fn:23">23</a></sup></td><td>gray!1742.35</td><td>gray!1715.00</td><td>157.94</td><td>46.01</td><td>8.11</td><td>24.44</td><td>18.64</td><td>20.00</td><td>19.15</td><td>gray!1718.07</td></tr><tr><td><sup><a href="#fn:20">20</a></sup></td><td>gray!1763.46</td><td>gray!1735.01</td><td>100.64</td><td>20.78</td><td>17.57</td><td>35.00</td><td>48.36</td><td>40.00</td><td>52.10</td><td>gray!1738.60</td></tr><tr><td><sup><a href="#fn:8">8</a></sup></td><td>gray!1777.74</td><td>gray!1754.62</td><td>151.48</td><td>17.38</td><td>25.00</td><td>71.11</td><td>78.33</td><td>30.00</td><td>69.15</td><td>gray!1754.72</td></tr><tr><td><sup><a href="#fn:64">64</a></sup></td><td>gray!1778.84</td><td>gray!1757.73</td><td>146.93</td><td>39.33</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>gray!17-</td></tr><tr><td><sup><a href="#fn:37">37</a></sup></td><td>gray!1785.07</td><td>gray!1767.27</td><td>259.23</td><td>33.67</td><td>53.75</td><td>68.89</td><td>81.67</td><td>50.00</td><td>82.11</td><td>gray!1767.28</td></tr><tr><td>lavender LinkVLA (Ours)</td><td>gray!1791.01</td><td>gray!1774.55</td><td>255.84</td><td>34.62</td><td>60.00</td><td>80.00</td><td>93.33</td><td>50.00</td><td>83.68</td><td>gray!1773.40</td></tr></tbody></table>

We first introduce the benchmarks, evaluation metrics (Sec. 4.1), and implementation details (Sec. 4.2). Then we present and analyze the closed-loop results and the instruction-following ability (Sec. 4.3), followed by detailed ablation experiment and comparisons (Sec. 4.4).

### 4.1 Settings

Bench2Drive. We train and evaluate LinkVLA using the Bench2Drive benchmark [^19], which provides a set of interactive scenarios within the widely used CARLA simulator [^7]. We follow SimLingo [^37] and use an open-source expert PDM-lite [^40] to collect driving dataset in the CARLA simulator. Our evaluation follows the CARLA v2 closed-loop protocol for end-to-end autonomous driving, comprising 44 interactive scenarios with 5 routes each, for a total of 220 official routes across diverse weather conditions. We report performance using the benchmark’s official metrics: Driving Score (DS), Success Rate (SR), Efficiency, Comfortness, and Multi-Ability.

Instruction-following Evaluation. We evaluate the model’s instruction-following capabilities using the Action Dreaming dataset from SimLingo [^37]. This dataset is designed to assess a model’s ability not only to comprehend scene-specific knowledge from language but also to translate this understanding into the corresponding action space. Given a natural language instruction, the model is expected to generate a sequence of actions that corresponds to the command. The evaluation is conducted on the CARLA Town 13 dreamer dataset for validation. The instructions belong to one of six classes: Slow down, Speed up, Reach target speed, Lane change, Object centric. Performance is measured by the success rate.

DriveLM-hard (VQA) and Commentary. We evaluate VQA and commentary generation on the DriveLM-hard benchmark [^37]. This challenging validation set is derived from DriveLM [^40] and focuses on the CARLA Town 13 environment. To ensure a balanced test that includes rare cases, the benchmark was constructed by uniformly sampling 10 examples per answer type, rather than using simple random sampling. The final dataset contains 330 VQA answer types and 190 Commentary types. We report scores using the SPICE, BLEU, and ROUGE-L metrics.

Table 2: Performance and Latency Comparison. All metrics are evaluated on the CARLA benchmark. Latency is the average inference time per step, measured on H20 GPU.

| ID | Method | Type | Latency $\downarrow$ | Driving Score $\uparrow$ |
| --- | --- | --- | --- | --- |
| 1 | Orion [^8] | VAE | 65ms | 77.74 |
| 2 | Simlingo [^37] | MLP | 34ms | 85.07 |
| 3 | Ours | AR | 361ms | 90.66 |
| lavender 4 | Ours | C2F | 48ms | 91.01 |

### 4.2 Implementation details

Action Tokenize. The framework operates within a Bird’s-Eye-View (BEV) space spanning the coordinate ranges $x\in[0,50]\text{m}$ and $y\in[-30,30]\text{m}$. To create a discrete action space, these coordinates are first transformed using the logarithmic function detailed in Sec 3.1 (with hyperparameter $k=5$) and subsequently discretized into a uniform grid with a 0.1 step size. This process yields a $56\times 101$ grid, which constitutes a vocabulary of $K_{action}=$ 5,656 discrete action tokens. For the spatial soft-labeling procedure, a neighbor weighting radius of $R=10$ cells and a Gaussian standard deviation of $\sigma=1.2$ are employed. Furthermore, to enable hierarchical action generation for the coarse-to-fine (C2F) framework, two special tokens are introduced: the path goal token and the waypoint goal token.

Training Details. We use the InternVL2-1B from the Mini-InternVL family [^9] as our main architecture. The InternVL2-1B [^9] model consists of the vision encoder InternViT-300M-448px (ViT) [^4] and Qwen2-0.5B-Instruct [^43] as the language model (LLM). We train our model using the AdamW optimizer [^26] with a cosine learning rate schedule. The hyperparameters are set as follows: a base learning rate of 1e-4, weight decay of 0.1, $\beta_{1}=0.9$, $\beta_{2}=0.999$, and a dropout rate of 0.1. The model is trained for 30 epochs on 32 H20 GPUs with a batch size of 48. For model adaptation, we follow SimLingo [^37] and apply LoRA [^12] with a rank of 32 and $\alpha=64$.

During inference, we employ a Chain-of-Thought (CoT) approach. First, the model generates a textual rationale for the action. Conditioned on this commentary, it then predicts the final action sequence. This output consists of 20 geometric path tokens and 10 temporal waypoint tokens per frame.

Unified understanding and generation. We randomly concatenate a $(V,L,A)$ tuple to: 1) $[V,A,L]$ and supervise $L$ for action understanding, or 2) $[V,L,A]$ and supervise $A$ for action generation. Both are trained together with $\mathcal{L}_{total}$.

![[x5 5.png|Refer to caption]]

Figure 4: Visualization in challenging environment with various language instructions. The generated trajectory accurately adheres to the language instruction while remaining safe and feasible within the complex environment.

Table 3: Instruction-following evaluation on Action Dreaming dataset [^37]. Align. refers to alignment with unified training.

<table><tbody><tr><td>ID</td><td>Token</td><td>C2F</td><td>Align.</td><td colspan="7">Success Rate (%) <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td></td><td></td><td></td><td></td><td>Faster</td><td>Slower</td><td>Target Speed</td><td>Lane Change</td><td>Object</td><td>Stop</td><td>Mean</td></tr><tr><td>1</td><td>✗</td><td>✗</td><td>✗</td><td>81.42</td><td>61.83</td><td>66.27</td><td>75.53</td><td>74.69</td><td>60.93</td><td>70.11</td></tr><tr><td>2</td><td>✓</td><td>✗</td><td>✗</td><td>88.44</td><td>65.24</td><td>63.37</td><td>88.49</td><td>84.34</td><td>99.88</td><td>81.63</td></tr><tr><td>3</td><td>✓</td><td>✓</td><td>✗</td><td>93.16</td><td>55.86</td><td>69.24</td><td>95.45</td><td>85.38</td><td>92.14</td><td>81.87</td></tr><tr><td>lavender 4</td><td>✓</td><td>✓</td><td>✓</td><td>96.48</td><td>65.57</td><td>74.73</td><td>97.42</td><td>91.41</td><td>97.34</td><td>87.16</td></tr></tbody></table>

Table 4: Language Ability in DriveLM-VQA and commentary evaluation. S / B / R refers to SPICE / BLEU / ROUGE-L

<table><tbody><tr><td rowspan="2">ID</td><td rowspan="2">Token</td><td rowspan="2">C2F</td><td rowspan="2">Align.</td><td colspan="3">DriveLM-VQA <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td colspan="3">Commentary <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>S</td><td>B</td><td>R</td><td>S</td><td>B</td><td>R</td></tr><tr><td>1</td><td>✗</td><td>✗</td><td>✗</td><td>66.7</td><td>68.9</td><td>71.5</td><td>49.2</td><td>60.3</td><td>64.3</td></tr><tr><td>2</td><td>✓</td><td>✗</td><td>✗</td><td>69.7</td><td>70.5</td><td>73.1</td><td>53.3</td><td>63.7</td><td>68.0</td></tr><tr><td>3</td><td>✓</td><td>✓</td><td>✗</td><td>71.3</td><td>69.9</td><td>73.4</td><td>53.6</td><td>61.6</td><td>67.3</td></tr><tr><td>lavender 4</td><td>✓</td><td>✓</td><td>✓</td><td>73.0</td><td>74.7</td><td>77.0</td><td>57.4</td><td>65.7</td><td>70.8</td></tr></tbody></table>

Table 5: Closed-loop performance ablation study with different components we proposed.

<table><tbody><tr><td rowspan="2">ID</td><td rowspan="2">Token</td><td rowspan="2">C2F</td><td rowspan="2">Align.</td><td colspan="2">Closed-loop metrics <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>Driving Score</td><td>Success Rate(%)</td></tr><tr><td>1</td><td>✗</td><td>✗</td><td>✗</td><td>85.07</td><td>67.27</td></tr><tr><td>2</td><td>✓</td><td>✗</td><td>✗</td><td>89.57</td><td>73.18</td></tr><tr><td>3</td><td>✓</td><td>✓</td><td>✗</td><td>89.85</td><td>72.27</td></tr><tr><td>lavender 4</td><td>✓</td><td>✓</td><td>✓</td><td>91.01</td><td>74.55</td></tr></tbody></table>

### 4.3 Results

Bench2Drive Results. Table 1 reports closed-loop metrics and multi-ability evaluations on Bench2Drive. LinkVLA archives the highest driving score and success rate while maintaining comparable efficiency and comfortness. Concretely, LinkVLA achieves driving score of 91.01 and an success rate of 74.55, surpassing the previous state of the art, SimLingo (85.07 DS, 67.27 SR), by 5.94 and 7.28 points, respectively, corresponding to relative gains of 6.98% and 10.82%. For efficiency, LinkVLA reaches 255.84, markedly exceeding earlier methods such as Orion (151.48) and AutoVLA (146.93). For comfortness, LinkVLA (34.62) modestly surpasses SimLingo (33.67) and substantially outperforms Orion (17.38).

The multi-ability results show robust driving behavior across diverse interaction scenarios. LinkVLA achieves the best scores in Merging (60.00), Overtake (80.00), Brake (93.33), and Traffic-Sign (83.68), and matches the top performance on Give-Way (50.00). The gains are especially pronounced in interaction-heavy and hazard-responsive skills, with improvements over SimLingo of 6.25 points in Merging, 11.11 in Overtake, and 11.66 in Brake. The overall multi-ability perfomance reaches 73.40, outperforming SimLingo (67.28) by 6.12 points (9.09% relative) and Orion (54.72) by 18.68, indicating that LinkVLA is better for closed-loop driving performance.

Inference Latency. Table 2 presents a comparative analysis of the latency-performance on the Bench2Drive benchmark. Latency is quantified as the average inference time to generate one compact trajectory per frame. We omit the computational cost of the Chain-of-Thought (CoT) process, as its variable, query-dependent nature would introduce a confounding factor into the latency analysis.

Among the baselines, SimLingo is the fastest with a latency of 34 ms per step but achieves a Driving Score of 85.07. In contrast, Orion (VAE) is slower at 65 ms. Our auto-regressive (AR) variant achieves a high Driving Score of 90.66 but at a prohibitive latency of 361 ms.

In contrast, our proposed C2F method achieves a superior balance. It reduces latency from 361 ms to 48 ms, while simultaneously increasing the Driving Score to 91.01, the highest among all compared methods, achieving a 13.27-point higher Driving Score with 26% lower latency than Orion. Compared to the fastest baseline, SimLingo, our method provides a 5.94-point performance gain with only a modest 14 ms increase in latency.

### 4.4 Comparison Analysis

Instruction-following Evaluation. Our instruction-following evaluation, detailed in Table 3, shows that the progressive addition of action tokenization, coarse-to-fine (C2F) trajectory generation, and the action understanding objective for alignment (short for align. in table) substantially improves performance. Compared to the baseline (ID 1, 70.11% mean success), introducing tokenization alone (ID 2) boosts the overall success rate to 81.63%. This gain is driven by near-perfect performance on the Stop task (99.88%) and significant improvements in Lane Change (88.49%) and Object centric (84.34%). Subsequently adding C2F generation (ID 3) further enhances performance on tasks such as Faster (93.16%), Target Speed (69.24%), and Lane Change (95.45%). Finally, incorporating the alignment module (ID 4) yields the most optimal and balanced performance. This final configuration achieves the highest mean success rate of 87.16% and sets new peak scores for Faster (96.48%), Target Speed (74.73%), and Lane Change (97.42%), demonstrating the surprising instruction-following ability of our method and the effectiveness of our proposed action understanding objective.

Language Ability Evaluation. Table 4 indicates that across both benchmarks, enabling tokenization (ID 2) yields consistent gains over the baseline (ID 1) in SPICE, BLEU, and ROUGE-L. Adding C2F (ID 3) further strengthens semantic fidelity: SPICE rises on both DriveLM-VQA (to 71.3) and commentary (to 53.6). Incorporating alignment training in addition to tokenization and C2F (ID 4) yields the best language understand performance in both VQA and commentary benchmarks, with all metrics improved. Thanks to the unified token space design, our model has achieved more outstanding language ability.

Ablation Experiment. Table 5 shows that tokenization yields the large improvement, increasing the driving score from 85.07% to 89.57% and the success rate from 67.27% to 73.18%. Adding C2F without alignment produces a marginal gain in driving score (to 89.85%) and slightly reduces success rate (to 72.27%). Incorporating alignment in addition to tokenization and C2F yields the best performance (driving score 91.01%; success rate 74.55%), surpassing all prior configurations.

Table 6: Effect of soft label in close-loop evaluation.

| Method | Driving Score $\uparrow$ | Success Rate(%) $\uparrow$ |
| --- | --- | --- |
| Ours w/o. soft label | 90.85 | 72.73 |
| Ours w/. soft label | 91.01 | 74.55 |

Effect of Soft-labeling. Table S2 shows that adding soft-label supervision yields consistent gains in closed-loop performance. The driving score increases from 90.85 to 91.01 (by 0.16 points), and the success rate rises from 72.73% to 74.55% (by 1.82 points). These results suggest that soft-label supervision effectively leverages spatial prior and results in robust driving performance and task completion.

Table 7: Effect of different navigation forms.

| Method | Driving Score $\uparrow$ | Success Rate(%) $\uparrow$ |
| --- | --- | --- |
| GPS target point | 91.01 | 74.55 |
| Navigation command | 91.25 | 73.18 |

Navigation Modalities. Table 7 shows that the two navigation modalities deliver comparable closed-loop performance. It shows the ability of the LinkVLA to follow basic navigational commands and navigational GPS target points with the same model.

Additional ablation Discussions. Due to space limitation, we further provide more extensive ablation studies on action codebook size $K_{action}$, scaling factor $k$ in the log transformation, and the spread scale parameter $\sigma$ in the spatial soft-labeling in Supplementary Materials.

## 5 Conclusion

We present LinkVLA, a novel VLA model that improves language-action alignment and efficiency. We achieve deep cross-modal consistency by unifying language and action tokens in a shared codebook and training a novel bidirectional captioning objective. This core strategy, coupled with a coarse-to-fine decoder that cuts latency by 86%, delivers substantial gains in instruction following and driving performance on closed-loop benchmarks. Our work thus provides a practical path toward reliable and responsive language-guided agents for real-world deployment.

## References

Supplementary Material  

## Appendix

## Appendix A Action Tokenization

### A.1 Log-Coordinate Transformation

We visualize the log-transformed coordinate space to facilitate intuitive comparison. We devise a non-uniform quantization scheme that prioritizes precision near the ego-vehicle by first applying a non-linear transformation to the waypoint coordinates $(x,y)$ along each axis independently.

![[x6 4.png|Refer to caption]]

Figure S1: Comparison of uniform and log token grids, with the corresponding waypoint distributions under each grid.

### A.2 Number of Action Tokens

We evaluate the effect of the number of action tokens on driving performance on the Bench2Drive [^19] benchmark. To this end, we adopt a non-uniform quantization scheme that prioritizes precision near the ego-vehicle by first applying a non-linear transformation to the waypoint coordinates $(x,y)$ along each axis independently. Specifically, each coordinate $z\in\{x,y\}$ is transformed using a signed logarithmic function:

$$
z^{\prime}=\operatorname{sign}(z)\cdot\log\!\big(1+k\cdot|z|\big).
$$

We then vary the symmetric logarithmic (symlog) scaling factor $k$ from 5.0 to 10.0, which increases the number of action tokens from 5,656 to 7,245. The parameter $k$ controls the mapping from physical coordinates (both $x$ and $y$, in meters) to the transformed space, determining the degree of compression prior to binning and, in turn, the effective bin widths in the original coordinate space.

Table S1: Effect of the number of action tokens (controlled by $k$) in closed-loop evaluation [^7].

| Method | Driving Score $\uparrow$ | Success Rate (%) $\uparrow$ |
| --- | --- | --- |
| k = 5.0 (5,656 tokens) | 91.01 | 74.55 |
| k = 10.0 (7,245 tokens) | 89.85 | 70.45 |

### A.3 Spatial Soft-Labeling

We evaluate the spread scale in the spatial soft-labeling on driving performance on the Bench2Drive [^19] benchmark by varying the spread parameter $\sigma$ from 1.2 to 3.0. Larger $\sigma$ yields softer targets by broadening the Gaussian smoothing and distributing probability over a wider neighborhood. Specifically, for a ground-truth token $a_{gt}$, we construct the target distribution $q(a)$ over all action tokens $a\in\mathcal{C}_{\text{action}}$ as a normalized 2D Gaussian centered at the coordinates of $a_{gt}$:

$$
q(a)=\frac{1}{Z}\exp\left(-\frac{\|\text{pos}(a)-\text{pos}(a_{gt})\|_{2}^{2}}{2\sigma^{2}}\right),
$$

where $\text{pos}(a)$ maps an action token to its 2D coordinates in the spatial grid, $\|\cdot\|_{2}^{2}$ denotes the squared Euclidean distance, $\sigma$ is a hyperparameter controlling the spread of the distribution, and $Z$ is a normalization constant ensuring $\sum_{a\in\mathcal{C}_{\text{action}}}q(a)=1$.

Table S2: Effect of the spread scale parameter $\sigma$ in the spatial soft-labeling in closed-loop evaluation [^7].

| Method | Driving Score $\uparrow$ | Success Rate (%) $\uparrow$ |
| --- | --- | --- |
| $\sigma=1.2$ | 91.01 | 74.55 |
| $\sigma=3.0$ | 89.73 | 69.55 |

## Appendix B Dataset

### B.1 Action Dreaming

We conduct experiments on the Action Dreaming [^37] dataset and its offline, nonreactive simulator, which generates alternative ego-vehicle trajectories and assesses their feasibility with respect to collision avoidance and traffic-rule compliance. Ego-trajectory prediction is implemented with a kinematic bicycle model controlled by PID controllers (PDM-lite [^40]), driven either by perturbed ground-truth actions or by PID commands computed from predefined path waypoints and target speeds. The dataset provides simulator states and short-horizon forecasts for dynamic objects to enable collision checking. The simulator supports several modes (Objects/Collision, Faster, Slower, Target Speed, Lane Changes, Stop) to induce diverse behaviors and trajectories. we use Success Rate as the metric. Each category has its own definition of success, which we detail in the following:

- Objects (Collision): This describes the task of driving towards or crashing into specific objects. The path is evaluated first. If the path of the expert trajectory and the ground truth dreamer trajectory is different (Average Displacement Error $ADE>1.0$) it is counted as success if the predicted path is closer to the ground truth dreamer path than to the expert path ($ADE_{pred2expert}>ADE_{pred2dreamer}$). If the dreamer path is nearly identical to the expert path ($ADE<1.0$) the instruction is about correct speed predictions (e.g., if the instruction is “drive towards a dynamic object” it is important to get the speed right and not just the path). The success is then defined as $ADE_{pred2dreamer}<1.0$ and the average predicted speed is within 30% of the ground truth dreamer speed.
- Faster (Speed up): For each predicted speed waypoint, derive target speeds for future timesteps and fit a linear regression to obtain the slope $s$. Let $v$ denote the current ego speed at the start of the sequence. Success is defined as $s>0.05\,v$.
- Slower (Slow down): Same computation as Faster. Success is defined as $s<-0.05\,v$.
- Target Speed: Since the target speed may not be reached in the prediction horizon of the waypoints, predictions are compared with the ground truth actions instead of directly comparing to the target speed. Two rules define success: First, if the predicted target speed inferred from the last two waypoints is in a 20% range of the instructed target speed. Second, if the predicted target speed inferred from the last two waypoints is in the 20% range of the speed of the last two waypoints of the ground truth speed waypoints. This can be different from the instructed target speed due to limitations in the acceleration rates of the vehicle.
- Lane Changes: Compare the final waypoint of the predicted path with the final waypoints of the ground-truth dreamer path and the ground-truth expert path. Success is defined when the predicted final location is closer to the dreamer’s final location than to the expert’s final location.
- Stop: Success is defined when the minimum predicted speed over the sequence is below $0.1\,\mathrm{m/s}$.

### B.2 VQA-DriveLM

The VQA data from SimLingo [^37] are sourced from the DriveLM-Carla [^41] dataset and generated using its data-creation pipeline. Question–answer pairs are extracted from the adopted dataset rather than the original DriveLM release; the training split contains 28M QA pairs over 1M frames in Town 12. Evaluation follows DriveLM keyframe selection to focus on informative frames, and the validation split is balanced across answer types. Labels are heuristically auto-generated, and the dataset includes GPT-4–based paraphrase augmentation (up to 20 variants per QA) to mitigate phrase-level overfitting, with variants sampled at load time.

### B.3 Commentary

The commentary labels in SimLingo [^37] are automatically generated from a subset of saved simulator state using heuristic, template-based rules. Each label comprises: (1) a route action with justification—default “Follow the route,” replaced only for scenarios requiring lane deviation (e.g., obstacle, encroaching vehicle), with phase-specific templates for before/during/after the deviation; (2) a speed action categorized as remain stopped, stop now, maintain (or maintain reduced) speed, increase speed, or slow down, with a special “Wait for a gap before changing lanes” case when stationary prior to a deviation; and (3) a speed reason derived from IDM [^48] features that identify the leading object (vehicle/pedestrian/static control) and its attributes/state, from which concise rationales are composed (e.g., due to pedestrian crossing, behind a red SUV, because the light is red/green). When near a junction, an additional notice summarizes other vehicles’ positions and motions (e.g., junction clear, vehicle moving away, oncoming traffic).

## Appendix C Qualitative Results

Figure S2 provides further qualitative results from the closed-loop evaluation, illustrating our model’s performance in a variety of driving scenarios.

![[x7 4.png|Refer to caption]]

Figure S2: Qualitative results of our proposed model during closed-loop evaluation in the CARLA simulator. The figure showcases representative driving scenarios, such as navigating intersections and avoiding obstacles.

[^1]: K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. $\pi$ 0: A vision-language-action flow model for general robot control. corr, abs/2410.24164, 2024. doi: 10.48550. arXiv preprint ARXIV.2410.24164. Cited by: §1.

[^2]: K. Chen, Y. Li, W. Zhang, Y. Liu, P. Li, R. Gao, L. Hong, M. Tian, X. Zhao, Z. Li, et al. (2025) Automated evaluation of large vision-language models on self-driving corner cases. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 7817–7826. Cited by: §1.

[^3]: S. Chen, B. Jiang, H. Gao, B. Liao, Q. Xu, Q. Zhang, C. Huang, W. Liu, and X. Wang (2024) Vadv2: end-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243. Cited by: §1, §2.1.

[^4]: Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, M. Zhong, Q. Zhang, X. Zhu, L. Lu, et al. (2024) Internvl: scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 24185–24198. Cited by: Figure 2, Figure 2, §4.2.

[^5]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger (2022) Transfuser: imitation with transformer-based sensor fusion for autonomous driving. IEEE transactions on pattern analysis and machine intelligence 45 (11), pp. 12878–12895. Cited by: §1.

[^6]: K. Ding, B. Chen, Y. Su, H. Gao, B. Jin, C. Sima, W. Zhang, X. Li, P. Barsch, H. Li, et al. (2024) Hint-ad: holistically aligned interpretability in end-to-end autonomous driving. arXiv preprint arXiv:2409.06702. Cited by: §1.

[^7]: A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun (2017) CARLA: an open urban driving simulator. In Conference on robot learning, pp. 1–16. Cited by: Table S1, Table S2, §4.1.

[^8]: H. Fu, D. Zhang, Z. Zhao, J. Cui, D. Liang, C. Zhang, D. Zhang, H. Xie, B. Wang, and X. Bai (2025) ORION: a holistic end-to-end autonomous driving framework by vision-language instructed action generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, Cited by: §1, §1, §2.3, Table 1, Table 2.

[^9]: Z. Gao, Z. Chen, E. Cui, Y. Ren, W. Wang, J. Zhu, H. Tian, S. Ye, J. He, X. Zhu, et al. (2024) Mini-internvl: a flexible-transfer pocket multi-modal model with 5% parameters and 90% performance. Visual Intelligence 2 (1), pp. 32. Cited by: §4.2.

[^10]: C. Glossop, W. Chen, A. Bhorkar, D. Shah, and S. Levine (2025) CAST: counterfactual labels improve instruction following in vision-language-action models. arXiv preprint arXiv:2508.13446. Cited by: §1, §1, §2.3.

[^11]: D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi, et al. (2025) Deepseek-r1: incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948. Cited by: §1.

[^12]: E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, et al. (2022) Lora: low-rank adaptation of large language models.. ICLR 1 (2), pp. 3. Cited by: §4.2.

[^13]: S. Hu, L. Chen, P. Wu, H. Li, J. Yan, and D. Tao (2022) St-p3: end-to-end vision-based autonomous driving via spatial-temporal feature learning. In European Conference on Computer Vision, pp. 533–549. Cited by: §1.

[^14]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, et al. (2023) Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 17853–17862. Cited by: §1, §2.1, Table 1, Table 1.

[^15]: J. Hwang, R. Xu, H. Lin, W. Hung, J. Ji, K. Choi, D. Huang, T. He, P. Covington, B. Sapp, et al. (2024) Emma: end-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2410.23262. Cited by: §2.2.

[^16]: A. Jaech, A. Kalai, A. Lerer, A. Richardson, A. El-Kishky, A. Low, A. Helyar, A. Madry, A. Beutel, A. Carney, et al. (2024) Openai o1 system card. arXiv preprint arXiv:2412.16720. Cited by: §1.

[^17]: X. Jia, Y. Gao, L. Chen, J. Yan, P. L. Liu, and H. Li (2023) Driveadapter: breaking the coupling barrier of perception and planning in end-to-end autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7953–7963. Cited by: Table 1.

[^18]: X. Jia, P. Wu, L. Chen, J. Xie, C. He, J. Yan, and H. Li (2023) Think twice before driving: towards scalable decoders for end-to-end autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21983–21994. Cited by: Table 1.

[^19]: X. Jia, Z. Yang, Q. Li, Z. Zhang, and J. Yan (2024) Bench2Drive: towards multi-ability benchmarking of closed-loop end-to-end autonomous driving. In NeurIPS 2024 Datasets and Benchmarks Track, Cited by: §A.2, §A.3, §4.1.

[^20]: X. Jia, J. You, Z. Zhang, and J. Yan (2025) DriveTransformer: unified transformer for scalable end-to-end autonomous driving. In International Conference on Learning Representations (ICLR), Cited by: §2.1, Table 1.

[^21]: A. Jiang, Y. Gao, Z. Sun, Y. Wang, J. Wang, J. Chai, Q. Cao, Y. Heng, H. Jiang, Y. Dong, et al. (2025) Diffvla: vision-language guided diffusion planning for autonomous driving. arXiv preprint arXiv:2505.19381. Cited by: §1, §2.2.

[^22]: B. Jiang, S. Chen, B. Liao, X. Zhang, W. Yin, Q. Zhang, C. Huang, W. Liu, and X. Wang (2024) Senna: bridging large vision-language models and end-to-end autonomous driving. arXiv preprint arXiv:2410.22313. Cited by: §1.

[^23]: B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang (2023) Vad: vectorized scene representation for efficient autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 8340–8350. Cited by: §1, §2.1, Table 1.

[^24]: B. Jiang, S. Chen, Q. Zhang, W. Liu, and X. Wang (2025) Alphadrive: unleashing the power of vlms in autonomous driving via reinforcement learning and reasoning. arXiv preprint arXiv:2503.07608. Cited by: §1.

[^25]: B. Jin and H. Liu (2023) Adapt: action-aware driving caption transformer. In CAAI International Conference on Artificial Intelligence, pp. 473–477. Cited by: §1, §2.2.

[^26]: D. Kinga, J. B. Adam, et al. (2015) A method for stochastic optimization. In International conference on learning representations (ICLR), Vol. 5. Cited by: §4.2.

[^27]: B. Li, Y. Wang, J. Mao, B. Ivanovic, S. Veer, K. Leung, and M. Pavone (2024) Driving everywhere with large language model policy adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14948–14957. Cited by: §1.

[^28]: Y. Li, S. Shang, W. Liu, B. Zhan, H. Wang, Y. Wang, Y. Chen, X. Wang, Y. An, C. Tang, et al. (2025) DriveVLA-w0: world models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796. Cited by: §1.

[^29]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, et al. (2025) Recogdrive: a reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052. Cited by: §1, §2.2.

[^30]: Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y. Ji, Z. Li, Z. Zhu, J. Kautz, Z. Wu, et al. (2024) Hydra-mdp: end-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978. Cited by: §1.

[^31]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al. (2025) Diffusiondrive: truncated diffusion model for end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 12037–12047. Cited by: §1, §2.1.

[^32]: Y. Ma, Y. Cao, J. Sun, M. Pavone, and C. Xiao (2024) Dolphins: multimodal language model for driving. In European Conference on Computer Vision, pp. 403–420. Cited by: §1.

[^33]: J. Mao, Y. Qian, J. Ye, H. Zhao, and Y. Wang (2023) Gpt-driver: learning to drive with gpt. arXiv preprint arXiv:2310.01415. Cited by: §1.

[^34]: A. Marcu, L. Chen, J. Hünermann, A. Karnsund, B. Hanotte, P. Chidananda, S. Nair, V. Badrinarayanan, A. Kendall, J. Shotton, et al. (2024) Lingoqa: visual question answering for autonomous driving. In European Conference on Computer Vision, pp. 252–269. Cited by: §1, §2.2.

[^35]: S. Park, M. Lee, J. Kang, H. Choi, Y. Park, J. Cho, A. Lee, and D. Kim (2024) Vlaad: vision and language assistant for autonomous driving. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 980–987. Cited by: §1, §2.2.

[^36]: K. Pertsch, K. Stachowicz, B. Ichter, D. Driess, S. Nair, Q. Vuong, O. Mees, C. Finn, and S. Levine (2025) Fast: efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747. Cited by: §1.

[^37]: K. Renz, L. Chen, E. Arani, and O. Sinavski (2025) Simlingo: vision-only closed-loop autonomous driving with language-action alignment. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 11993–12003. Cited by: §B.1, §B.2, §B.3, §1, §1, §1, §2.3, §3.3, §4.1, §4.1, §4.1, §4.2, Table 1, Table 2, Table 3.

[^38]: H. Shao, Y. Hu, L. Wang, G. Song, S. L. Waslander, Y. Liu, and H. Li (2024) Lmdrive: closed-loop end-to-end driving with large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15120–15130. Cited by: §1.

[^39]: Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. (2024) Deepseekmath: pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300. Cited by: §2.3.

[^40]: C. Sima, K. Renz, K. Chitta, L. Chen, H. Zhang, C. Xie, J. Beißwenger, P. Luo, A. Geiger, and H. Li (2024) Drivelm: driving with graph visual question answering. In European conference on computer vision, pp. 256–274. Cited by: §B.1, §1, §4.1, §4.1.

[^41]: C. Sima, K. Renz, K. Chitta, L. Chen, H. Zhang, C. Xie, P. Luo, A. Geiger, and H. Li (2023) DriveLM: driving with graph visual question answering. arXiv preprint arXiv:2312.14150. Cited by: §B.2.

[^42]: W. Sun, X. Lin, Y. Shi, C. Zhang, H. Wu, and S. Zheng (2025) Sparsedrive: end-to-end autonomous driving via sparse scene representation. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 8795–8801. Cited by: §1, §2.1.

[^43]: Q. Team et al. (2024) Qwen2 technical report. arXiv preprint arXiv:2407.10671 2 (3). Cited by: Figure 2, Figure 2, §4.2.

[^44]: W. R. Team et al. (2024) LINGO-2: driving with natural language. Cited by: §1.

[^45]: K. Tian, J. Mao, Y. Zhang, J. Jiang, Y. Zhou, and Z. Tu (2025) Nuscenes-spatialqa: a spatial understanding and reasoning benchmark for vision-language models in autonomous driving. arXiv preprint arXiv:2504.03164. Cited by: §1.

[^46]: X. Tian, J. Gu, B. Li, Y. Liu, Y. Wang, Z. Zhao, K. Zhan, P. Jia, X. Lang, and H. Zhao (2024) Drivevlm: the convergence of autonomous driving and large vision-language models. arXiv preprint arXiv:2402.12289. Cited by: §1, §2.2.

[^47]: S. Tong, D. Fan, J. Li, Y. Xiong, X. Chen, K. Sinha, M. Rabbat, Y. LeCun, S. Xie, and Z. Liu (2025) Metamorph: multimodal understanding and generation via instruction tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 17001–17012. Cited by: §2.3, §3.2.

[^48]: M. Treiber, A. Hennecke, and D. Helbing (2000) Congested traffic states in empirical observations and microscopic simulations. Physical review E 62 (2), pp. 1805. Cited by: §B.3.

[^49]: S. Wang, Z. Yu, X. Jiang, S. Lan, M. Shi, N. Chang, J. Kautz, Y. Li, and J. M. Alvarez (2025) Omnidrive: a holistic vision-language dataset for autonomous driving with counterfactual reasoning. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 22442–22452. Cited by: §1, §1, §1, §2.3.

[^50]: W. Wang, J. Xie, C. Hu, H. Zou, J. Fan, W. Tong, Y. Wen, S. Wu, H. Deng, Z. Li, et al. (2023) Drivemlm: aligning multi-modal large language models with behavioral planning states for autonomous driving. arXiv preprint arXiv:2312.09245. Cited by: §1, §2.2.

[^51]: X. Weng, B. Ivanovic, Y. Wang, Y. Wang, and M. Pavone (2024) Para-drive: parallelized architecture for real-time autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15449–15458. Cited by: §1, §2.1.

[^52]: K. Winter, M. Azer, and F. B. Flohr (2025) BEVDriver: leveraging bev maps in llms for robust closed-loop driving. arXiv preprint arXiv:2503.03074. Cited by: §1.

[^53]: P. Wu, X. Jia, L. Chen, J. Yan, H. Li, and Y. Qiao (2022) Trajectory-guided control prediction for end-to-end autonomous driving: a simple yet strong baseline. Advances in Neural Information Processing Systems 35, pp. 6119–6132. Cited by: Table 1.

[^54]: J. Xie, W. Mao, Z. Bai, D. J. Zhang, W. Wang, K. Q. Lin, Y. Gu, Z. Chen, Z. Yang, and M. Z. Shou (2024) Show-o: one single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528. Cited by: §2.3, §3.2.

[^55]: Z. Xing, X. Zhang, Y. Hu, B. Jiang, T. He, Q. Zhang, X. Long, and W. Yin (2025) Goalflow: goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 1602–1611. Cited by: §1, §2.1, §3.3.

[^56]: Z. Xu, Y. Zhang, E. Xie, Z. Zhao, Y. Guo, K. K. Wong, Z. Li, and H. Zhao (2024) Drivegpt4: interpretable end-to-end autonomous driving via large language model. IEEE Robotics and Automation Letters. Cited by: §1, §2.2.

[^57]: Z. Yang, Y. Chai, X. Jia, Q. Li, Y. Shao, X. Zhu, H. Su, and J. Yan (2025) DriveMoE: mixture-of-experts for vision-language-action model in end-to-end autonomous driving. arXiv preprint arXiv:2505.16278. Cited by: §2.2.

[^58]: S. Zeng, X. Chang, M. Xie, X. Liu, Y. Bai, Z. Pan, M. Xu, and X. Wei (2025) FutureSightDrive: thinking visually with spatio-temporal cot for autonomous driving. arXiv preprint arXiv:2505.17685. Cited by: §1.

[^59]: J. Zhai, Z. Feng, J. Du, Y. Mao, J. Liu, Z. Tan, Y. Zhang, X. Ye, and J. Wang (2023) Rethinking the open-loop evaluation of end-to-end autonomous driving in nuscenes. arXiv preprint arXiv:2305.10430. Cited by: Table 1.

[^60]: W. Zheng, W. Chen, Y. Huang, B. Zhang, Y. Duan, and J. Lu (2024) Occworld: learning a 3d occupancy world model for autonomous driving. In European conference on computer vision, pp. 55–72. Cited by: §1.

[^61]: W. Zheng, R. Song, X. Guo, C. Zhang, and L. Chen (2024) Genad: generative end-to-end autonomous driving. In European Conference on Computer Vision, pp. 87–104. Cited by: §1, §2.1.

[^62]: C. Zhou, L. Yu, A. Babu, K. Tirumala, M. Yasunaga, L. Shamis, J. Kahn, X. Ma, L. Zettlemoyer, and O. Levy (2024) Transfusion: predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039. Cited by: §2.3, §3.2.

[^63]: X. Zhou, X. Han, F. Yang, Y. Ma, and A. C. Knoll (2025) Opendrivevla: towards end-to-end autonomous driving with large vision language action model. arXiv preprint arXiv:2503.23463. Cited by: §1, §2.2.

[^64]: Z. Zhou, T. Cai, S. Z. Zhao, Y. Zhang, Z. Huang, B. Zhou, and J. Ma (2025) AutoVLA: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, External Links: [Link](https://openreview.net/forum?id=28qUA2bSe5) Cited by: §1, §1, §2.3, Table 1.