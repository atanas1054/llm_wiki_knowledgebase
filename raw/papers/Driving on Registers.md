---
title: "Driving on Registers"
source: "https://arxiv.org/html/2601.05083v2"
author:
published:
created: 2026-10-14
description:
tags:
  - "clippings"
---
Ellington Kirby Affiliation: valeo.ai, Paris, France Email: [firstname.lastname@valeo.com](mailto:firstname.lastname@valeo.com)    Alexandre Boulch Affiliation: valeo.ai, Paris, France    Yihong Xu Affiliation: valeo.ai, Paris, France    Yuan Yin Affiliation: valeo.ai, Paris, France    Gilles Puy Affiliation: valeo.ai, Paris, France    Éloi Zablocki Affiliation: valeo.ai, Paris, France    Andrei Bursuc Affiliation: valeo.ai, Paris, France    Spyros Gidaris Affiliation: valeo.ai, Paris, France    Renaud Marlet Affiliation: valeo.ai, Paris, France Affiliation: LIGM, ENPC, IP Paris, UGE, CNRS, France    Florent Bartoccioni Affiliation: valeo.ai, Paris, France    Anh-Quan Cao Affiliation: valeo.ai, Paris, France    Nermin Samet Affiliation: valeo.ai, Paris, France    Tuan-Hung VU Affiliation: valeo.ai, Paris, France    Matthieu Cord Affiliation: valeo.ai, Paris, France Affiliation: Sorbonne Université, CNRS, ISIR, F-75005 Paris, France

###### Abstract

We present DrivoR, a simple and efficient transformer-based architecture for end-to-end autonomous driving. Our approach builds on pretrained Vision Transformers (ViTs) and introduces camera-aware register tokens that compress multi-camera features into a compact scene representation, significantly reducing downstream computation without sacrificing accuracy. These tokens drive two lightweight transformer decoders that generate and then score candidate trajectories. The scoring decoder learns to mimic an oracle and predicts interpretable sub-scores representing aspects such as safety, comfort, and efficiency, enabling behavior-conditioned driving at inference. Despite its minimal design, DrivoR outperforms or matches strong contemporary baselines across NAVSIM-v1, NAVSIM-v2, and the photorealistic closed-loop HUGSIM benchmark. Our results show that a pure-transformer architecture, combined with targeted token compression, is sufficient for accurate, efficient, and adaptive end-to-end driving. Code and checkpoints will be made available [via the project page](https://valeoai.github.io/driving-on-registers/).

<sup>*</sup>

## 1 Introduction

![[architecture_pdf.png|Refer to caption]]

Figure 1: DrivoR architecture. The proposed architecture is composed of three transformer blocks: one encoder (perception) and two decoders (trajectory and scoring). The perception encoder compresses perceptual information in camera-aware registers for lightweight subsequent processing in the trajectory and scoring decoders. The decoded trajectories are re-embedded and detached from the gradient computation graph to disentangle scoring and generation. The final trajectory is chosen from the proposal set via the max predicted score.

End-to-end (E2E) planning has emerged as a promising direction for autonomous driving (AD), offering a single pipeline that maps raw sensor data and ego state to driving decisions [^19] [^49]. Besides, by avoiding intermediate annotations such as 3D boxes, these methods reduce labeling cost. Among E2E approaches, trajectory-proposal methods, whether using a large pre-computed vocabulary [^6] [^33] [^29] [^35], or generating proposals on the fly [^36] [^52] [^16] [^15], have shown particularly strong performance.

Methods predicting multiple possible trajectories and selecting between them naturally capture the uncertainty within navigation. As in model-based RL [^38], the ability to score becomes central: the scorer must reliably choose the best candidate using context encoded in the sensor features.

The sensor processing backbone producing the features that capture this context are thus key in E2E planning methods. These backbones typically dominate the parameter and FLOP count of E2E methods, often leveraging convolutional architectures like VoV-Net [^27] or large pre-trained networks such as Vision Transformers [^13] like EVA [^14], or DINO [^39]. Such backbones output thousands of tokens per frame, which must be processed for hundreds of trajectories. This creates a major computational bottleneck that only worsens as resolution or sensor count increases.

The most common solution to reducing the bottleneck is to pool these features along the spatial dimensions. However, feature pooling enforces specific resolution requirements on sensor inputs, and treats all inputs as equally informative, performing the same averaging operation across all cameras. Inspired by works like [^54], we ask just how many tokens are needed to represent a driving scene?

We introduce DrivoR, a ViT-based E2E planning architecture that replaces uniform pooling with a fixed set of per-camera register tokens that serve as compact scene descriptors. These tokens preserve planning-relevant context while drastically reducing the visual representation length. Using this compressed representation, DrivoR generates and scores trajectory proposals using two disentangled modules. The final trajectory is selected using predicted sub-scores, allowing behavior modulation at inference.

Our method relies only on scoring annotations (rather than explicit 3D supervision) and achieves state-of-the-art results on NAVSIM-v1 [^12], NAVSIM-v2 [^5], and the closed-loop HUGSIM benchmark [^60]. Overall, our contributions are as follows:

- An intentionally simple transformer architecture, without any intermediate BEV representations, or any large trajectory dictionary.
- The first work to explore specific, structural, advantages of ViT-based image backbones for E2E planning in the usage of register-based token compression.
- A disentangled scoring module enabling stronger performance and controllable behavior.

## 2 Related Work

##### Token compression.

Token reduction is central to efficient ViTs, whose attention cost grows quadratically with sequence length. Simple solutions such as patch-group pooling are parameter-free but treat all tokens uniformly. Other training-free strategies include matching-based compression [^2]. Learned approaches range from Perceiver-IO’s latent queries [^21] to ViT register tokens, originally introduced to fix attention sinks [^10] and later used in compact generative models like TiTok [^54]. Recent driving-focused works [^20] [^47] highlight the growing need for token reduction in real-time systems. To our knowledge, we are the first to repurpose ViT register tokens specifically for reducing visual tokens in E2E planning, enabling compact scene representations while retaining planning-critical context.

##### End-to-end driving.

End-to-end learning has become popular in autonomous driving since the proof of concept of pioneering works [^19] [^24] like UniAD [^19]. However, they still heavily rely on modular designs with different sub modules, such as detection, tracking and mapping, making it hard to deploy. With the introduction of more efficient (pseudo) closed-loop evaluation metrics [^12] [^5] or simulation [^60], more recent methods [^16] [^36] [^8] [^32] [^6] predict the planned trajectory or actions directly from sensor inputs, a step forward toward the fully end-to-end paradigm.

These E2E methods [^16] [^36] [^32] mostly rely on off-the-shelf CNN-based (e.g., ResNet-34, ResNet-50, V2-99) image encoders without putting further attention on the design of the perception stack. Downstream in the planning stack, transformers are more robustly explored. Transfuser [^8] uses transformers for LiDAR–image fusion; DriveTransformer [^23] unifies multiple tasks under one transformer; and iPad [^16] exploits transformer residuals for iterative trajectory refinement. Innovation has concentrated on how to use mid-level sensor features rather than on stronger pretrained backbones. When ViTs [^13] are used, methods [^15] [^35] [^53] [^34] typically rely on large or huge variants without considering the computation costs. These models lean on attention to fuse heterogeneous inputs (ego state, images, LiDAR, commands), but often supplement it with costly 3D reasoning-BEV projections, deformable cross-attention, or LiDAR supervision, raising annotation and compute demands. In this context, we design a simple query-based transformer architecture, built on the much smaller ViT variants, avoiding the high cost of complex intermediate representation, while keeping superior E2E driving performance.

##### Trajectory scoring.

Trajectory-proposal based planning has become a compelling direction for E2E driving, introduced in Hydra-MDP [^29]. Producing many possible futures and selecting one forces the model to address multimodality head-on. This shifts importance toward the scoring module, which must reliably choose the best candidate [^36]. A recent state-of-the-art work on NAVSIM-v2, GTRS [^35] showed that learning a sufficiently strong scorer using a ViT backbone, paired with a large trajectory vocabulary, can solve complex scenarios. We introduce the importance of separate scoring and trajectory generation pathways in our E2E model.

## 3 Method

### 3.1 A Simple Design

We design a simple and efficient transformer based architecture for planning in autonomous driving. The overall pipeline is presented in Fig. 1. It is composed of three modules: an encoder for perception and two decoders for trajectory estimation and scoring. The model follows a classical transformer encoder-decoder architecture [^46] without any complex intermediate representation.

The *encoder for perception* (see Sec. 3.2) is a vision transformer [^13] applied per camera. In order to compress the feature map to small set of tokens, we introduce additional per-camera registers, finetuned with the backbone to encode the perceptual information. We group together all per-camera register tokens at the output of the ViT to form the scene tokens.

The *trajectories* are then estimated from learned queries using a transformer decoder attending to the scene tokens via cross-attention (see Sec. 3.3).

Finally, the trajectories are scored in the *scoring decoder*. Each trajectory becomes a query input of the decoder, which also attends to the scene tokens to produce scores (see Sec. 3.4).

During training, the trajectories are learned using a winner-takes-all regression loss (Sec. 3.3) and the scores are learned against an oracle scorer, e.g., provided along with the dataset (Sec. 3.4). At inference, we re-interpret our learned scoring function as a reward function, enabling driving according to several behavior conditioned policies with a single trained model.

### 3.2 Perception Encoder

![[architecture_encoder.png|Refer to caption]]

Figure 2: Encoder and decoder architectures follow standard transformer architectures, with introduction of sensor registers in the encoder, and using these registers as scene tokens in downstream decoders.

At perception level, we seek to fulfill three objectives: 1) leveraging recent ViT architectures; 2) benefiting from pretrained weights; and 3) limiting the output size to contain the computational complexity in the decoders. To this end, we efficiently compress the perceptual signals into a limited set of tokens using additional registers, and finetune the backbone with LoRA. as illustrated in Fig. 2a.

For each camera, we concatenate $R$ camera registers of size $D_{\text{ViT}}$, along with pre-existing registers, classification token, and patch tokens. $D_{\text{ViT}}$ denotes the dimension of the ViT features. All registers and tokens are fed to the ViT. We then retrieve the $R$ camera tokens at the final layer of the ViT. The final camera tokens for each of the $N$ cameras are finally grouped together to obtain $N\times R$ scene tokens of size $D_{\text{ViT}}$. Note that we use per-camera registers, that is we initialize $N\times R$ registers, where $N$ is the number of input cameras. This allows us to have camera-aware scene tokens: the model can differentiate if a given scene token is extracted from, e.g., the front, left or right camera.

This compression into a small set of camera tokens is close in spirit to Perceiver approaches [^22] [^21], with the noticeable difference that these approaches use cross-attentions for compression. Setting up such a mechanism in the encoder would require changes in the ViT architecture.

In our case, we can directly use a pretrained ViT as initialization, and perform LoRA finetuning of the ViT backbone to learn the vision-to-register compression, reducing parameter count and speeding training.

### 3.3 Trajectories

##### Trajectory decoder.

All decoders use the architecture depicted in Fig. 2b, which consists of a vanilla transformer decoder [^46]: a stack of $k$ transformer blocks, each made of a self-attention layer, followed by a cross-attention to the scene tokens, and a feed-forward network (FFN), all with residual connections.

The input of the trajectory decoder consists of a set of learnable trajectory queries $Q_{\text{traj}}$, each of dimension $D_{\text{traj}}$, which is also the inner dimension of the trajectory decoder. After the process, the queries are decoded into $|Q_{\text{traj}}|$ candidate trajectories. These queries are randomly initialized and learned during training. The ego status inputs, consisting of poses, velocities, accelerations, and driving command, are encoded and added to the trajectory queries before entering the decoder. The final trajectory tokens at the end of the transformer are decoded into trajectories with an MLP.

Each decoded candidate trajectory $\tau_{i}$ is a sequence of $n_{p}$ poses predicted from the current timestep $t$ (excluded) up to a future horizon at $t+T$, $T$ being the total prediction duration. Each pose is represented as $(x,y,\theta)\in\mathbb{R}^{3}$, and the full trajectory lies in $\mathbb{R}^{n_{p}\times 3}$. The variables $x$ and $y$ denote the longitudinal and lateral displacements, respectively, and $\theta$ is the heading. All quantities are expressed in the ego agent’s local reference frame at timestep $t$. The time interval between successive predicted poses is assumed to be uniform.

##### Trajectory loss.

The trajectories ${\tau_{i}}$ are learned using a Winner-Takes-All (WTA) approach [^17] or, equivalently, the minimum-over-n (MoN) loss [^16]. Given a reference human trajectory $\hat{\tau}$ of duration $T$ and consisting of $n_{p}$ poses, only the closest predicted trajectory is supervised, which allows the model to produce diverse candidate trajectories:

$$
\mathcal{L}_{\text{traj}}=\min_{i}\left\lVert\tau_{i}-\hat{\tau}\right\rVert_{1}
$$

This formulation encourages the model to consider multiple plausible pathways for a given scene.

An additional regression target $\hat{\tau}^{\prime}$ can be introduced to encourage the predicted trajectories to reach farther waypoints. To construct this target, a reference trajectory with duration $T^{\prime}>T$ is resampled to $T$ using cubic spline interpolation, producing an accelerated version that matches the number of predicted poses $n_{p}$. The resulting multi-target trajectory loss is written as

$$
\mathcal{L}_{\text{traj}}=\min_{i}(\left\lVert\tau_{i}-\hat{\tau}\right\rVert_{1}+\left\lVert\tau_{i}-\hat{\tau}^{\prime}\right\rVert_{1})\>.
$$

### 3.4 Scores

##### Scoring decoder.

The scoring decoder, which evaluates the quality of each candidate trajectory, uses an architecture mirroring that of trajectory generation.

The scoring decoder takes as input the decoded trajectories as well as perceptual information through the scene tokens. Each decoded trajectory is turned into a $D_{\text{score}}$ -dimensional query using an MLP. All the trajectory queries are fed to the scoring decoder.

Embedding the decoded trajectories into a new feature space rather than reusing the trajectory decoder’s output tokens is key in our architecture. This enforces a separation between the information used to *generate* trajectories and the information used to *score* them: the scorer sees only the decoded trajectory, not the additional latent details still present in the trajectory tokens.

Our scoring decoder then uses cross-attentions between scene tokens and score queries, allowing gradients to flow back to the perception encoder to learn scene tokens useful for both trajectory and scoring predictions. However, we prevent the gradient from the scoring decoder from flowing back to the trajectory decoder. This prevents the trajectory decoder from being influenced by the current quality of the scoring decoder during training.

Finally we predict the six score components [^12] using a dedicated MLP for each score.

##### Scoring for adjustable driving behavior.

A key feature of our model is its ability to adapt trajectory selection to different driving preferences. For instance, one user may prioritize safety and comfort, while another may favor faster progress at the cost of smoothness. To enable such flexibility, our scoring head predicts separate sub-scores corresponding to different aspects of driving quality (e.g., safety, comfort, efficiency). These sub-scores can then be combined post-hoc into a single meta-score at inference time, allowing the relative importance of each term to be adjusted without retraining the model. We adopt the sub-scores directly from the Predictive Driver Model Score (PDMS) scorer used in NAVSIM-v1.

##### Scoring loss.

We train our scoring network via binary cross entropy (BCE) to predict the individual sub-score components $c$ of the PDMS, each given a weight $\lambda_{c}$:

$$
\mathcal{L}_{\text{score}}=\sum_{c}\lambda_{c}\sum_{i}\operatorname{BCE}\left(\mathcal{G}_{\theta_{c}}(\tau_{i}),\mathcal{G}_{c}(\tau_{i})\right)
$$

where $\mathcal{G}_{c}$ is an oracle scorer for the sub-score $c$, and $\mathcal{G}_{\theta_{c}}(\tau_{i})$ is our learned scoring head for sub-score $c$.

##### Inference.

At inference, we interpret our scoring network as a reward function and our full pipeline as a driving policy which is conditioned on a specific behavior profile encapsulated in the set of weights applied to the scoring outputs. Borrowing from Offline-RL literature such as CtRL-Sim [^40], we can thus condition our final scoring output on a driving behavior by modifying the values of our $\lambda_{c}$, and selecting trajectories which maximize the score computed with this new set of weights. For example, this can encourage trajectories which make maximal estimated progress.

### 3.5 Final Training Loss.

The training loss is the combination the trajectory loss and the scoring loss:

$$
\mathcal{L}=\mathcal{L}_{\text{traj}}+\lambda_{s}\mathcal{L}_{\text{score}}.
$$

In practice during training, the weights of the losses are all set to 1 for ease of implementation, s.t., $\lambda_{c}=1$ for each sub-score $c$ and $\lambda_{s}=1$.

## 4 Experiments

##### Experimental setup.

As input, we use 4 cameras (front, front left, front right and back). The perception module is a DINOv2 ViT-S [^39] encoder, LoRA finetuned [^18] (rank 32) following [^1]. By default, we add 16 registers per camera. The decoders are 4-layer transformers with an inner dimension of 256. The feed- forward network has a dilation factor of 4. The camera registers as well as the initial trajectory tokens are randomly initialized with normal distribution $\mathcal{N}(0,10^{-6})$. The model is trained on the navtrain split for 10 epochs, with learning rate $2\times 10^{-4}$ and cosine annealing scheduling. For the ablation studies, if not described otherwise, we train our model with this common base architecture. All ablation model scores are computed on the navval split. All models were trained on 4 NVIDIA A100 GPUs. The DrivoR model is roughly 40M parameters, significantly below comparable works.

### 4.1 Benchmarks

##### NAVSIM-v1.

NAVSIM-v1 [^12] is a dataset built out of nuPlan [^4] as a subset of OpenScene [^9]. As opposed to previous benchmark such as nuScenes [^3], mostly formulating driving quality as a measure of similarity to the expert human trajectory, NAVSIM-v1 introduces metrics inspired from closed-loop simulation. The main metric, Predictive Driver Model Score (PDMS), is an aggregation of penalties, e.g., collisions with other, non-reactive, agents, staying on-road and in the correct direction, and avoiding near-misses with other agents. These scores are combined with quality-related scores such as the comfort and progress. Progress is measured as a comparison to the centerline-progress of a PDM agent [^11] given access to complete ground-truth information. For training, we use default PDMS weights defined in the benchmark [^12].

The results obtained on the NAVSIM-v1 benchmark are presented in Tab. 1. DrivoR outperforms all other methods on NAVSIM-v1, and nears human-level performance. We highlight the comparison to RAP [^15], which we consider an orthogonal work: the large quantity of rasterized data introduced in the paper may be used in any method.

<table><tbody><tr><td>Method</td><td></td><td></td><td>NC</td><td>DAC</td><td>TTC</td><td>Comf.</td><td>EP</td><td>PDMS</td></tr><tr><td>PDM‑Closed</td><td><sup><a href="#fn:11">11</a></sup></td><td>PMLR’23</td><td>94.6</td><td>99.8</td><td>89.9</td><td>86.9</td><td>99.9</td><td>89.1</td></tr><tr><td>Human driver</td><td><sup><a href="#fn:12">12</a></sup></td><td>NeurIPS’24</td><td>100</td><td>100</td><td>100</td><td>99.9</td><td>87.5</td><td>94.8</td></tr><tr><td>RAP-DINO <sup>†</sup></td><td><sup><a href="#fn:15">15</a></sup></td><td>arXiv’25</td><td>99.1</td><td>98.9</td><td>96.7</td><td>100</td><td>90.3</td><td>93.8</td></tr><tr><td>Ego‑stat. MLP</td><td><sup><a href="#fn:12">12</a></sup></td><td>NeurIPS’24</td><td>93.0</td><td>77.3</td><td>83.6</td><td>100</td><td>62.8</td><td>65.6</td></tr><tr><td>UniVLA</td><td><sup><a href="#fn:48">48</a></sup></td><td>arXiv’25</td><td>96.9</td><td>91.1</td><td>91.7</td><td>96.7</td><td>76.8</td><td>81.7</td></tr><tr><td>DrivingGPT</td><td><sup><a href="#fn:7">7</a></sup></td><td>ICCV’24</td><td>98.9</td><td>90.7</td><td>94.9</td><td>95.6</td><td>79.7</td><td>82.4</td></tr><tr><td>UniAD</td><td><sup><a href="#fn:19">19</a></sup></td><td>CVPR’23</td><td>97.8</td><td>91.9</td><td>92.9</td><td>100</td><td>78.8</td><td>83.4</td></tr><tr><td>LTF</td><td><sup><a href="#fn:8">8</a></sup></td><td>TPAMI’22</td><td>97.4</td><td>92.8</td><td>92.4</td><td>100</td><td>79.0</td><td>83.8</td></tr><tr><td>PARA‑Drive</td><td><sup><a href="#fn:49">49</a></sup></td><td>CVPR’24</td><td>97.9</td><td>92.4</td><td>93.0</td><td>99.8</td><td>79.3</td><td>84.0</td></tr><tr><td>DriveX-S</td><td><sup><a href="#fn:41">41</a></sup></td><td>ICCV’25</td><td>97.5</td><td>94.0</td><td>93.0</td><td>100</td><td>79.7</td><td>84.5</td></tr><tr><td>World4Drive</td><td><sup><a href="#fn:58">58</a></sup></td><td>ICCV’25</td><td>97.4</td><td>94.3</td><td>92.8</td><td>100</td><td>79.9</td><td>85.1</td></tr><tr><td>DRAMA</td><td><sup><a href="#fn:56">56</a></sup></td><td>ISRR’24</td><td>98.0</td><td>93.1</td><td>94.8</td><td>100</td><td>80.1</td><td>85.5</td></tr><tr><td>VAD-v2</td><td><sup><a href="#fn:6">6</a></sup></td><td>arXiv’24</td><td>98.1</td><td>94.8</td><td>94.3</td><td>100</td><td>80.6</td><td>86.2</td></tr><tr><td>PRIX</td><td><sup><a href="#fn:50">50</a></sup></td><td>arXiv’25</td><td>98.1</td><td>96.3</td><td>94.1</td><td>100</td><td>82.3</td><td>87.8</td></tr><tr><td>DiffusionDrive</td><td><sup><a href="#fn:36">36</a></sup></td><td>CVPR’25</td><td>98.2</td><td>96.2</td><td>94.7</td><td>100</td><td>82.2</td><td>88.1</td></tr><tr><td>DIVER</td><td><sup><a href="#fn:43">43</a></sup></td><td>arXiv’25</td><td>98.5</td><td>96.5</td><td>94.9</td><td>100</td><td>82.6</td><td>88.3</td></tr><tr><td>AutoVLA</td><td><sup><a href="#fn:61">61</a></sup></td><td>NeurIPS’25</td><td>98.4</td><td>95.6</td><td>98.0</td><td>99.9</td><td>81.9</td><td>89.1</td></tr><tr><td>DriveVLA-W0</td><td><sup><a href="#fn:30">30</a></sup></td><td>arXiv’25</td><td>98.7</td><td>99.1</td><td>95.3</td><td>99.3</td><td>83.3</td><td>90.2</td></tr><tr><td>ReCogDrive</td><td><sup><a href="#fn:32">32</a></sup></td><td>arXiv’25</td><td>97.9</td><td>97.3</td><td>94.9</td><td>100</td><td>87.3</td><td>90.8</td></tr><tr><td>Hydra-MDP++</td><td><sup><a href="#fn:29">29</a></sup></td><td>arXiv’25</td><td>98.6</td><td>98.6</td><td>95.1</td><td>100</td><td>85.7</td><td>91.0</td></tr><tr><td>iPad</td><td><sup><a href="#fn:16">16</a></sup></td><td>arXiv’25</td><td>98.6</td><td>98.3</td><td>94.9</td><td>100</td><td>88.0</td><td>91.7</td></tr><tr><td>Centaur</td><td><sup><a href="#fn:42">42</a></sup></td><td>arXiv’25</td><td>99.5</td><td>98.9</td><td>98.0</td><td>100</td><td>85.9</td><td>92.6</td></tr><tr><td>DriveSuprim</td><td><sup><a href="#fn:53">53</a></sup></td><td>arXiv’25</td><td>98.6</td><td>98.6</td><td>95.5</td><td>100</td><td>91.3</td><td>93.5</td></tr><tr><td>DrivoR (train)</td><td></td><td></td><td>98.9</td><td>98.3</td><td>96.2</td><td>100</td><td>89.1</td><td>93.1</td></tr><tr><td>DrivoR (trainval)</td><td></td><td></td><td>99.0</td><td>98.9</td><td>96.7</td><td>100</td><td>90.0</td><td>93.7</td></tr><tr><td>DrivoR (+65k SimScale data)</td><td></td><td></td><td>99.1</td><td>99.0</td><td>96.9</td><td>100</td><td>90.3</td><td>94.0</td></tr><tr><td>DrivoR (+134k SimScale data)</td><td></td><td></td><td>99.1</td><td>99.2</td><td>96.9</td><td>100</td><td>91.6</td><td>94.6</td></tr><tr><td colspan="9"><sup>†</sup>: RAP <sup><a href="#fn:15">15</a></sup> is trained on a dataset that is 10 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> larger than navtrain (the default training set).</td></tr></tbody></table>

Table 1: NAVSIM-v1. Comparison to existing camera-only methods on the NAVSIM-v1 benchmark on test set (navtest). Full definition of scores in supplementary material, higher is better.

<table><tbody><tr><th></th><th></th><td colspan="5">RC</td><td colspan="5">HD-Score</td></tr><tr><th>Method</th><th></th><td>E</td><td>M</td><td>H</td><td>X</td><td>Avg.</td><td>E</td><td>M</td><td>H</td><td>X</td><td>Avg.</td></tr><tr><th>VAD</th><th><sup><a href="#fn:24">24</a></sup></th><td>51.3</td><td>31.1</td><td>25.3</td><td>26.5</td><td>31.4</td><td>36.3</td><td>9.5</td><td>8.0</td><td>11.5</td><td>13.4</td></tr><tr><th>LTF</th><th><sup><a href="#fn:8">8</a></sup></th><td>67.8</td><td>35.1</td><td>26.2</td><td>40.5</td><td>38.9</td><td>58.9</td><td>18.0</td><td>9.8</td><td>25.9</td><td>23.7</td></tr><tr><th>UniAD</th><th><sup><a href="#fn:19">19</a></sup></th><td>78.4</td><td>60.5</td><td>33.6</td><td>17.8</td><td>45.9</td><td>64.9</td><td>45.8</td><td>20.6</td><td>6.6</td><td>32.7</td></tr><tr><th>DrivoR</th><th></th><td>80.9</td><td>50.5</td><td>33.8</td><td>47.1</td><td>49.8</td><td>73.3</td><td>34.6</td><td>18.8</td><td>32.5</td><td>35.7</td></tr></tbody><tfoot><tr><th colspan="12">E: Easy, M: Medium, H: Hard, X: Extreme</th></tr></tfoot></table>

Table 2: Photorealistic closed-loop evaluation on HUGSIM [^60]. Zero-shot generalization using the DrivoR model from the NAVSIM-v1 evaluation. Scores are per difficulty and overall average road completion (RC) and HD-Score, higher always better.

<table><tbody><tr><td></td><td></td><td colspan="9">Stage 1</td><td colspan="9">Stage 2</td><td></td></tr><tr><td>Method</td><td></td><td>NC</td><td>DAC</td><td>DDC</td><td>TLC</td><td>EP</td><td>TTC</td><td>LK</td><td>HC</td><td>EC</td><td>NC</td><td>DAC</td><td>DDC</td><td>TLC</td><td>EP</td><td>TTC</td><td>LK</td><td>HC</td><td>EC</td><td>EPDMS</td></tr><tr><td>RAP-DINO (ViT-H)</td><td><sup><a href="#fn:15">15</a></sup></td><td>97.1</td><td>94.4</td><td>98.8</td><td>99.8</td><td>83.9</td><td>96.9</td><td>94.7</td><td>96.4</td><td>66.2</td><td>83.2</td><td>83.9</td><td>87.4</td><td>98.0</td><td>86.9</td><td>80.4</td><td>52.3</td><td>95.2</td><td>52.4</td><td>39.6</td></tr><tr><td>GTRS-D (V2-99)</td><td><sup><a href="#fn:35">35</a></sup></td><td>98.9</td><td>96.2</td><td>99.4</td><td>99.3</td><td>72.9</td><td>98.9</td><td>95.1</td><td>96.9</td><td>39.1</td><td>91.2</td><td>89.4</td><td>94.4</td><td>98.8</td><td>69.5</td><td>90.0</td><td>54.3</td><td>94.0</td><td>48.7</td><td>45.0</td></tr><tr><td>GTRS-A (V2-99)</td><td><sup><a href="#fn:35">35</a></sup></td><td>98.9</td><td>95.1</td><td>99.1</td><td>99.6</td><td>76.2</td><td>99.1</td><td>94.9</td><td>97.6</td><td>54.2</td><td>88.1</td><td>88.8</td><td>89.3</td><td>98.9</td><td>98.9</td><td>85.9</td><td>53.7</td><td>96.8</td><td>56.9</td><td>45.4</td></tr><tr><td>GTRS-DrivoR (ViT-S) <sup>∗</sup></td><td></td><td>98.0</td><td>95.8</td><td>99.7</td><td>99.3</td><td>72.9</td><td>98.2</td><td>95.6</td><td>96.9</td><td>51.6</td><td>91.6</td><td>86.7</td><td>90.2</td><td>98.8</td><td>73.2</td><td>88.9</td><td>51.9</td><td>94.9</td><td>46.4</td><td>45.8</td></tr><tr><td>ZTRS (V2-99)</td><td><sup><a href="#fn:34">34</a></sup></td><td>98.9</td><td>97.6</td><td>100</td><td>100</td><td>66.7</td><td>98.9</td><td>96.2</td><td>96.7</td><td>44.0</td><td>91.1</td><td>90.4</td><td>95.8</td><td>99.0</td><td>63.6</td><td>89.8</td><td>60.4</td><td>97.6</td><td>66.1</td><td>48.1</td></tr><tr><td>DrivoR (ViT-S)</td><td></td><td>98.8</td><td>95.1</td><td>98.9</td><td>100</td><td>72.6</td><td>98.7</td><td>94.0</td><td>97.6</td><td>73.3</td><td>90.2</td><td>88.4</td><td>91.9</td><td>98.6</td><td>70.0</td><td>88.0</td><td>50.1</td><td>98.5</td><td>76.2</td><td>48.3</td></tr><tr><td>DrivoR (+65k SimScale data, ViT-S)</td><td></td><td>98.9</td><td>97.3</td><td>99.2</td><td>99.6</td><td>77.7</td><td>99.1</td><td>95.3</td><td>97.6</td><td>68.4</td><td>92.3</td><td>92.2</td><td>97.0</td><td>99.0</td><td>72.1</td><td>90.3</td><td>56.3</td><td>97.1</td><td>38.8</td><td>52.3</td></tr><tr><td>SimScale (+185k SimScale data, V2-99)</td><td><sup><a href="#fn:45">45</a></sup></td><td>99.6</td><td>99.16</td><td>99.9</td><td>100</td><td>69.6</td><td>99.6</td><td>95.8</td><td>95.6</td><td>28.4</td><td>94.5</td><td>94.2</td><td>95.8</td><td>99.2</td><td>75.8</td><td>92.8</td><td>60.1</td><td>96.1</td><td>43.2</td><td>53.2</td></tr><tr><td>DrivoR (+134k SimScale data, ViT-S)</td><td></td><td>99.1</td><td>98.2</td><td>99.3</td><td>99.8</td><td>75.4</td><td>98.7</td><td>94.9</td><td>97.6</td><td>70.2</td><td>92.3</td><td>91.6</td><td>97.3</td><td>99.1</td><td>75.7</td><td>90.6</td><td>56.1</td><td>98.4</td><td>44.7</td><td>54.6</td></tr><tr><td colspan="21"><sup>∗</sup>: uses the same ViT-S backbone with registers as DrivoR, the prediction and scoring heads remain the same as in GTRS.</td></tr></tbody></table>

Table 3: NAVSIM-v2 navhard-two-stage. Comparison to existing methods on the NAVSIM-v2 benchmark test set using the EPDMS. Full definition of all scores in supplementary material, larger always better. Note: other scores reported in the literature before the official benchmark bug fix of the metrics are reported in supplementary material.

##### NAVSIM-v2.

NAVSIM-v2 [^5] builds on NAVSIM-v1 with the objective of closing the gap with closed-loop, simulator driven, benchmarks. NAVSIM-v2 introduces a second stage of evaluation, where novel variations of scenes are generated via Gaussian Splatting. These novel scenes consist of perturbations of the ego vehicle status, i.e., shifts and rotations, forcing the model to generalize outside of the training distribution. NAVSIM-v2 is scored using an extended version of the score from NAVSIM-v1, termed the EPDMS. We present the results on the navhard-two-stage split of NAVSIM-v2 in table Tab. 3, where DrivoR outperforms all existing works. We note that the results in Tab. 3 were computed after an official bug fix, and thus do not include the largest GTRS-Dense model. Results on NAVSIM-2 before the fix are included in the supplementary material. We use warmup-two-stage <sup>1</sup> as validation set for NAVSIM-v2 containing 7 scenes.

We highlight GTRS-DrivoR-ViT-S (row 4), produced by replacing the GTRS backbone with our ViT-S + register-based compression while keeping their original vocabulary and scorer. This swap removes GTRS’s pooling in favor of our learned compression, improving performance over the similarly sized V2-99 backbone and approaching the EVA-ViT-L variants (see supplementary), while delivering over 3× higher throughput. Notably, our full SOTA model differs from this variant only in the scoring pipeline, demonstrating the gains from our scorer.

##### HUGSIM.

For the closed-loop evaluation, we use HUGSIM [^60], a benchmark with scenarios adapted from the original scenes of KITTI-360 [^37], nuScenes [^3], PandaSet [^51], and Waymo [^44]. These scenarios are reconstructed as photorealistic 3D environments in which an E2E model controls the ego agent to navigate within each scenario. The planner perceives the 3D environment through RGB cameras whose viewpoints are determined by the ego agent’s position in the scenario.

Tab. 2reports results on the pre-challenge HUGSIM test set (345 scenarios across four difficulty levels). We follow the zero-shot protocol and evaluate with Road Completion (RC) and the HUGSIM Driving Score (HD-Score), the latter combining RC with averaged NC, DAC, TTC, and comfort. DrivoR, trained only on NAVSIM-v1 with no finetuning, achieves an RC of 49.8 and an HD-Score of 35.7—the highest among the reported baselines.

Note that we identified several anomalies in the official evaluation and simulation code, including inconsistent acceleration bounds in the comfort metric computation and an incorrect heading computation for the planned trajectory provided to the controller. After applying the necessary fixes, we reproduced all scores using our corrected implementation, since the results produced by the original code are not directly comparable and do not ensure a reliable evaluation.

##### Efficiency.

We benchmark the runtime performance of DrivoR against a ViT-L baseline (GTRS) on a single element batch and an A100 GPU without quantization nor acceleration process. DrivoR obtains a more than 3x throughput improvement (from 400ms/forward to 110ms/forward), with 3x reduction in GFLOPS and peak memory usage, with improved scores, demonstrating a step towards real time usage of a ViT backbone.

![[nav1_scaling_v2.png|Refer to caption]]

(a) NAVSIM-v1 (PDMS)

##### Scaling DrivoR.

With its simple design using vanilla transformers and registers, we study the scalability of DrivoR. We leverage synthetic data from [^45] with pseudo annotations from PDM-Closed [^11]. Concretely, we extract 65k and 134k annotations from [^45] and jointly train with navtrain, respectively. We show in Fig. 3 that DrivoR consistently improves the NAVSIM performance with the increased data. Moreover, DrivoR sets a new state-of-the-art performance with less data, compared to RAP [^15] using 500k+ synthetic data and SimScale [^45] with 185k extra data. More detailed performance scores are provided in Tab. 1 and Tab. 3.

### 4.2 Ablation Studies

#### 4.2.1 Perception

| Init. | Random | ImageNet 21k | DINOv2 |
| --- | --- | --- | --- |
| PDMS | 70.1 | 87.5 | 90.0 |

(a) Pretaining.

<table><tbody><tr><td></td><td>Compres-</td><td>Cam.</td><td>Scene</td><td>Img. Enc.</td><td colspan="2">Parameters</td><td></td></tr><tr><td></td><td>sion</td><td>tokens</td><td>tokens</td><td>training</td><td>Optim</td><td>Total</td><td>PDMS</td></tr><tr><td>(a)</td><td></td><td></td><td></td><td>Frozen</td><td>18.2</td><td>41.2</td><td>88.2</td></tr><tr><td>(b)</td><td>No</td><td>4k</td><td>16k</td><td>LoRA</td><td>18.8</td><td>41.8</td><td>90.2</td></tr><tr><td>(c)</td><td>Pooling</td><td>16</td><td>64</td><td>LoRA</td><td>18.2</td><td>41.2</td><td>89.7</td></tr><tr><td>(d)</td><td></td><td>16</td><td>64</td><td>Frozen</td><td>19.3</td><td>42.3</td><td>86.9</td></tr><tr><td>(e)</td><td>Decoder</td><td>16</td><td>64</td><td>LoRA</td><td>19.9</td><td>42.9</td><td>89.3</td></tr><tr><td>(f)</td><td></td><td>16</td><td>64</td><td>Full ft.</td><td>41.2</td><td>41.2</td><td>88.4</td></tr><tr><td>(g)</td><td></td><td>16</td><td>64</td><td>Frozen</td><td>18.2</td><td>41.2</td><td>84.4</td></tr><tr><td>(h)</td><td>Registers</td><td>16</td><td>64</td><td>LoRA</td><td>18.8</td><td>41.8</td><td>90.0</td></tr></tbody></table>

(b) Compression and finetuning. Note the higher performance of register based compression, nearing the model using 250x as many tokens.

<table><tbody><tr><td colspan="2"># Tokens</td><td rowspan="2">Registers</td><td rowspan="2">PDMS</td></tr><tr><td>per cam.</td><td>per scene</td></tr><tr><td>5</td><td>20</td><td>DINOv2</td><td>88.1</td></tr><tr><td>5 + 16</td><td>84</td><td>DINOv2 + Rand. init.</td><td>89.8</td></tr><tr><td>5</td><td>20</td><td>Rand. init.</td><td>89.7</td></tr><tr><td>8</td><td>32</td><td>Rand. init.</td><td>89.7</td></tr><tr><td>16</td><td>64</td><td>Rand. init.</td><td>90.0</td></tr><tr><td>32</td><td>128</td><td>Rand. init.</td><td>89.8</td></tr></tbody></table>

(c) Scene token count influence.

Table 4: Ablations of the perception. All results are presented on navval using a ViT-S backbone.

##### Image backbone initialization.

Various previous methods [^15] [^16] use pretrained backbones. We first study the impact of such an initialization on the planning results in 4(a). First, a good initialization of the perception ViT is crucial: using a pretrained backbone improves scores by a large margin (more than 15 PDMS points). Second, using large-scale pretrained DINOv2 [^39] improves further over pretraining on ImageNet21k [^26]. In all subsequent experiments, we use a pretrained DINOv2 backbone.

##### Compression to low number of perceptual tokens.

As argued in Sec. 3, using few perceptual tokens is of interest as it makes the trajectory prediction and scoring lightweight and faster. In 4(b), first column, we study three compression approaches: using a pooling operation of the output feature map as in [^35] (c), using a transformer decoder with 16 queries per camera (e), and our proposed approach using 16 additional registers in the model (h). For comparison purpose, we also provide models using the full feature maps, i.e., 16k scene tokens in (b). We observe that our register-based approach outperforms the pooling operation while introducing a low-overhead (0.6M parameters) at training. It also nearly reaches the performances of the no-compression model, despite using 250 times fewer tokens for downstream processing. Additionally, we observe that using a transformer decoder (with roughly the same number of parameters) does not reach the same performance.

##### Register token correlation.

Fig. 4shows inter-register cosine similarity for each camera. Front-camera tokens are largely de-correlated, suggesting per-register specialization. Fig. 5 confirms this: different front camera registers attend to distinct regions (e.g., lead vehicle, traffic lights, sidewalk).

Moving toward the rear cameras, similarity increases sharply: most tokens collapse to the same representation, and within the back camera only one remains distinct. The attention maps (second row of Fig. 5) reflect this collapse.

This pattern aligns with driving intuition: most attention is devoted to the scene ahead, with only brief checks behind. We hypothesize that collapsing less informative views (side/back) reduces noise for downstream planning, an effect which is impossible to observe with uniform pooling.

![[correlation_matrices.png|Refer to caption]]

Figure 4: Cosine similarity between scene tokens. Darker indicates lower cosine similarity. Note the specialized tokens in the front cam, and collapsed tokens in the back cam, showing relative camera compression. Averaged on navval.

![[cam_f0_base.png|Refer to caption]]

Figure 5: Attention maps of scene tokens. From the final attention layer, front-camera tokens specialize to distinct regions (traffic light, lead vehicle, road edges), while back-camera tokens largely collapse to the same features, aside from a single distinct token.

##### Finetuning strategy.

Next, we study the finetuning strategy in 4(b), second column, with three different settings: full finetuning (d, f), frozen backbone (a, d, g) and LoRA finetuning [^18] (b, c, e, h). For frozen backbone with registers, only the registers are learnable. First, we observe that LoRA finetuning improves the results by a large margin, reaching similar conclusion as in [^1] over frozen backbone. It is also the case when comparing LoRA (h) and Full finetuning (f), but to our understanding, it should be possible to close this gap with careful learning rate scheduling specific to the backbone. LoRA being more robust to these meta parameters, we use LoRA finetuning as default.

##### Number of registers.

We present in 4(c) the evolution of the validation score depending on the number of camera tokens. We either use new registers randomly initialized (the 4 original registers of the DINOv2 backbone are discarded) or, we keep the DINOv2 registers along with our randomly initialized ones. In the later case, to discriminate the cameras, we add per-camera learnable positional encoding to the DINOv2 registers at the image backbone output.

First, using the DINOv2 registers does not help compared to random initialization. We observe that the already specialized registers could be a bad initialization for driving tasks. We hypothesize that similarly to full finetuning, a careful learning rate scheduling could mitigate this gap.

Second, we observe that using more registers improves the performances up to a plateau between 16 and 32 registers per camera, we thus select 16 registers.

#### 4.2.2 Trajectories

The number of trajectory queries strongly affects performance. As shown in Tab. 5, increasing queries from 1 (human-only regression) to 128 yields gains that plateau around 64, which we adopt as our default.

| Num traj. | 1 | 8 | 16 | 32 | 64 | 128 |
| --- | --- | --- | --- | --- | --- | --- |
| PDMS | 80.1 | 87.6 | 88.1 | 89.5 | 90.0 | 90.0 |

Table 5: Ablations of the trajectory prediction. Influence of the number of trajectories on navval.

<table><thead><tr><th></th><th>Separate</th><th>Disentan-</th><th>Score</th><th rowspan="2">PDMS</th><th>Behavior</th></tr><tr><th></th><th>decoders</th><th>glement</th><th>num.</th><th>control</th></tr></thead><tbody><tr><td>(a)</td><td>✗</td><td>-</td><td>6</td><td>84.7</td><td>✓</td></tr><tr><td>(b)</td><td>✓</td><td>✗</td><td>6</td><td>86.8</td><td>✓</td></tr><tr><td>(c)</td><td>✓</td><td>✓</td><td>6</td><td>90.0</td><td>✓</td></tr><tr><td>(d)</td><td>✓</td><td>✓</td><td>1</td><td>88.2</td><td>✗</td></tr></tbody></table>

Table 6: Ablations of the scoring on navval. *Disentanglement* refers to a stop-gradient followed by an embedding of decoded trajectories. *Behavior control* stands for predicting all 6 PDMS components.

#### 4.2.3 Scoring

Tab. 6 shows our scoring pipeline ablation results.

Single vs. dual branches: (a) Using one transformer branch for both trajectory generation and scoring (with separate MLP heads) harms performance. As shown in Fig. 6, the two tasks attend to different cameras: generation focuses on the front view even for left turns, while scoring draws on left- or rear-camera features depending on trajectory sharpness or collision risk. This motivates separate branches.

Disentanglement: We next examine whether the scoring branch should embed trajectories in a new space and block gradients to the generator. Variant (b) does not disentangle; (c, d) do. Increased separation consistently improves performance.

Sub-score prediction: Comparing (c) and (d), predicting multiple sub-scores not only enables behavior control but also improves accuracy. This is likely due to the fact that learning the final output of the scoring formula in Eq. 3 is more difficult than predicting its components.

![[dominant_cameras.png|Refer to caption]]

Figure 6: Scoring head disentanglement. Dominant cameras are identified via cross-attention between scene tokens and score or trajectory tokens. Trajectory prediction consistently relies on the front camera, while scoring shifts attention based on trajectory behavior—underscoring the need to separate the two pipelines.

#### 4.2.4 Training

##### Longer trajectory.

In Sec. 3, we introduced an augmentation that regresses to a second, more aggressive trajectory. Tab. 7 compares this double objective to standard regression on the human trajectory. It improves NAVSIM-v1, which rewards progress over comfort, but hurts performance on NAVSIM-v2 warmup-two-stage, where perturbed agent states require more cautious driving to avoid collisions and recover safely.

| Targets | navval (PDMS) | warmup (EPDMS) |
| --- | --- | --- |
| ($t+T$) | 90.0 | 39.4 |
| ($t+T$) & ($t+T^{\prime}$) | 90.6 | 37.8 |

Table 7: Ablations. Multiple targets for WTA regression, we report PDMS score for NAVSIM-v1 navval and EPDMS for NAVSIM-v2 warmup-two-stage.

##### Final training setup.

For the final model setting, we evaluated longer training lengths. We observe that PDMS on navval increase with the number of epochs, reaching a plateau at 25 epochs. Our final model for NAVSIM-v1 is then trained for 25 epochs, on the competition split (navtrain + navval). On NAVSIM-v2 (warmup-two-stage), we observe an opposite trend, with EPDMS decreasing with the number of training epochs as well as training using the competition split. We therefore use 10 epochs for NAVSIM-v2.

##### Safety-oriented agent.

We evaluate the behavior of our agent obtained after tuning the score weights on warmup-two-stage. Intuitively we expect safer driving to be required for NAVSIM-v2 due to out-of-distribution scenes. In Fig. 7 we represent this agent in dark blue (“Safety-Oriented Agent”), and indeed see improved performance on safety and comfort metrics, but decreased progress, representing more passive, safe, driving.

Figure 7: Safety-oriented agent. Dark blue was tuned on warmup-two-stage, light blue is our NAVSIM-v1 model. The result of our behavior tuning is a more cautious, but safer agent.

## 5 Conclusion

We present DrivoR, a novel E2E driving method using register-based compression and disentangled scoring representations. DrivoR highlights that full-transformer architectures without complex intermediate states nor large trajectory dictionaries can achieve state-of-the-art results. Future works may explore compression incorporating historical frames, additional sensors or map information.

## Acknowledgments

We thank Loick Chambon for constant support throughout the project and Lan Feng for helpful discussions. This work was granted access to the HPC resources of IDRIS under the allocations AD011016241, AD011016239R1 and AD011012883R4 made by GENCI. We acknowledge EuroHPC Joint Undertaking for awarding the project ID EHPC-REG-2024R02-210 access to Karolina, Czech Republic. Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Commission. Neither the European Union nor the European Commission can be held responsible for them. This work was supported by the European Union’s Horizon Europe research and innovation programme under grant agreement No 101214398 (ELLIOT).

## References

Supplementary Material

## Appendix A Description of PDMS and EPDMS metrics

The *Predictive Driver Model Score (PDMS)*, main metric of NAVSIM-v1, and the *Extended Predictive Driver Model Score (EPDMS)*, main metric of NAVSIM-v2, follow a similar definition, based on a group $G$ of sub metrics. $G$ can be divided into two kinds of metrics: a group $G_{p}$ of penalties, and a group $G_{b}$ of behavioral metrics. $G_{p}$ contains the metrics measuring the compliance to driving rules, such as drivable area compliance or collision occurrence. $G_{b}$ contains the metrics measuring more driver-related metrics, such as comfort or progress towards the goal. The PDMS and EPDMS metrics are defined as:

$$
\text{Score}(\tau_{i})=\prod_{c\in G_{p}}\mathcal{G}_{c}(\tau_{i})^{\kappa^{p}_{c}}\times\frac{1}{Z}\sum_{c\in G_{b}}{\kappa^{b}_{c}\;\mathcal{G}_{c}(\tau_{i})}
$$

for $\tau_{i}$ a candidate trajectory, $Z=\sum_{c\in G_{b}}{\kappa^{b}_{c}}$ and $\kappa$ as defined in Tab. 8. $G_{p}$ metrics are multiplicative, meaning one failure to comply to a targeted driving rule, i.e., $\mathcal{G}_{c}(\tau_{i})=0$ results in a score equal to zero. On the contrary, $G_{b}$ metrics are additive and allow compromise between the metrics, e.g., comfort and progress.

| Abbr. | Sub-score | Grp. | PDMS | EPDMS |
| --- | --- | --- | --- | --- |
|  | $c$ |  | $\kappa_{c}$ | $\kappa_{c}$ |
| NC | No-at-fault Collisions | $G_{p}$ | 1 | 1 |
| DAC | Drivable Area Compliance | $G_{p}$ | 1 | 1 |
| DDC | Driving direction Compliance | $G_{p}$ | 0 | 1 |
| TLC | Traffic-line compliance | $G_{p}$ | \- | 1 |
| TTC | Time to Collision | $G_{b}$ | 5 | 5 |
| EP | Ego Progress | $G_{b}$ | 5 | 5 |
| Comf. | Ego Comfort | $G_{b}$ | 2 | \- |
| LK | Lane Keeping | $G_{b}$ | \- | 2 |
| HC | History Comfort | $G_{b}$ | \- | 2 |
| EC | Extended Comfort | $G_{b}$ | \- | 2 |

Table 8: Weights of the sub-scores in the PDMS and EPDMS.

## Appendix B Training loss weights

We recall here the equation of the score loss defined in the main paper:

$$
\mathcal{L}_{\text{score}}=\sum_{c}\lambda_{c}\sum_{i}\operatorname{BCE}\left(\mathcal{G}_{\theta_{c}}(\tau_{i}),\mathcal{G}_{c}(\tau_{i})\right)
$$

which corresponds an individual loss for each sub-score of the PDMS. During training, the predicted sub-scores and their associated weight $\lambda_{c}$ are set as defined in Tab. 9, i.e., all $\lambda_{c}$ are set to 1.

| Sub-score | NC | DAC | DDC | TTC | EP | Comf. |
| --- | --- | --- | --- | --- | --- | --- |
| $\lambda_{c}$ | 1 | 1 | 1 | 1 | 1 | 1 |

Table 9: Training loss weights of the sub-scores.

Similarly, as defined in the main paper, the final loss is:

$$
\mathcal{L}=\mathcal{L}_{\text{traj}}+\lambda_{s}\mathcal{L}_{\text{score}}.
$$

where, again, $\lambda_{s}$ is set to 1.

## Appendix C Inference weights of the sub-scores

| Abbr. | Sub-score | Grp. | PDMS | EPDMS |
| --- | --- | --- | --- | --- |
|  | $c$ |  | $\kappa_{c}$ | $\kappa_{c}$ |
| NC | No-at-fault Collisions | $G_{p}$ | 1 | 10 |
| DAC | Drivable Area Compliance | $G_{p}$ | 1 | 13 |
| DDC | Driving direction Compliance | $G_{p}$ | 0 | 6 |
| TTC | Time to Collision | $G_{a}$ | 5 | 14 |
| EP | Ego Progress | $G_{a}$ | 5 | 15 |
| Comf. | Ego Comfort | $G_{a}$ | 2 | 2 |

Table 10: Inference weights of the sub-scores.

The subscore prediction design in DrivoR enables flexible weighting strategy. In Tab. 10, we provide the sub-score weights for both PDMS (NAVSIM-v1) and EPDMS (NAVSIM-v2). For simplicity, the model trained for NAVSIM-v1 keeps the same weights as in the standard PDMS metrics [^12]. For NAVSIM-v2, we adjust the weights with the exception of the Comfort. The adjustment is validated with NAVSIM-v2 warmup-two-stage, see Figure 6 in the original paper.

<table><tbody><tr><th></th><th></th><th></th><th></th><td></td><td></td><td>Peak</td><td>Throughput</td><td>NAVSIM-v2</td></tr><tr><th>Method</th><th>Img. size</th><th>Cams</th><th>Encoder</th><td>Parameters <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>GFLOPs <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>Memory <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>(ms) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><th>RAP-Dino <sup>†</sup></th><th>(448, 768)</th><th>4</th><th>ViT-H</th><td>888M</td><td>4760</td><td>4.2GB</td><td>690ms</td><td>39.6</td></tr><tr><th></th><th></th><th></th><th></th><td>857M <math><semantics><mo>∥</mo> <annotation>\|</annotation></semantics></math> 31M</td><td>4622 <math><semantics><mo>∥</mo> <annotation>\|</annotation></semantics></math> 138</td><td>-</td><td>653ms <math><semantics><mo>∥</mo> <annotation>\|</annotation></semantics></math> 37ms</td><td>-</td></tr><tr><th colspan="9"><em>GTRS uses extra trajectories from a diffusion-based model, introducing extra overheads.</em></th></tr><tr><th>+GTRS-DP</th><th>(512, 2048)</th><th>4</th><th>+V2-99</th><td>+116M</td><td>+1249</td><td>+1.15GB</td><td>+389ms</td><td>-</td></tr><tr><th></th><th></th><th></th><th>-</th><td>110M <math><semantics><mo>∥</mo> <annotation>\|</annotation></semantics></math> 6M</td><td>770 <math><semantics><mo>∥</mo> <annotation>\|</annotation></semantics></math> 479</td><td>-</td><td>60ms <math><semantics><mo>∥</mo> <annotation>\|</annotation></semantics></math> 329ms</td><td>-</td></tr><tr><th>GTRS-D</th><th>(512, 2048)</th><th>1</th><th>V2-99</th><td>81M</td><td>404</td><td>1.0GB</td><td>96ms</td><td>45.0</td></tr><tr><th>-</th><th>-</th><th>-</th><th>-</th><td>69.5M <math><semantics><mo>∥</mo> <annotation>\|</annotation></semantics></math> 11.5M</td><td>345 <math><semantics><mo>∥</mo> <annotation>\|</annotation></semantics></math> 59</td><td>-</td><td>22ms <math><semantics><mo>∥</mo> <annotation>\|</annotation></semantics></math> 74ms</td><td>-</td></tr><tr><th>GTRS-A</th><th>(512, 2048)</th><th>1</th><th>V2-99</th><td>171M</td><td>439</td><td>1.42GB</td><td>243ms</td><td>45.4</td></tr><tr><th>-</th><th>-</th><th>-</th><th>-</th><td>69.5M <math><semantics><mo>∥</mo> <annotation>\|</annotation></semantics></math> 101.5M</td><td>345 <math><semantics><mo>∥</mo> <annotation>\|</annotation></semantics></math> 94</td><td>-</td><td>22ms <math><semantics><mo>∥</mo> <annotation>\|</annotation></semantics></math> 221ms</td><td>-</td></tr><tr><th>GTRS-D <sup>‡</sup></th><th>(512, 2048)</th><th>1</th><th>ViT-L</th><td>321M</td><td>1730</td><td>1.6GB</td><td>400ms</td><td>47.0</td></tr><tr><th></th><th>-</th><th>-</th><th>-</th><td>313M <math><semantics><mo>∥</mo> <annotation>\|</annotation></semantics></math> 8M</td><td>1610 <math><semantics><mo>∥</mo> <annotation>\|</annotation></semantics></math> 120</td><td>-</td><td>352ms <math><semantics><mo>∥</mo> <annotation>\|</annotation></semantics></math> 48ms</td><td>-</td></tr><tr><th>GTRS-D <sup>‡</sup></th><th>(512, 2048)</th><th>1</th><th>ViT-S</th><td>32M</td><td>234</td><td>0.38GB</td><td>303ms</td><td>45.9</td></tr><tr><th></th><th></th><th></th><th>-</th><td>24M <math><semantics><mo>∥</mo> <annotation>\|</annotation></semantics></math> 8M</td><td>117 <math><semantics><mo>∥</mo> <annotation>\|</annotation></semantics></math> 117</td><td>-</td><td>46ms <math><semantics><mo>∥</mo> <annotation>\|</annotation></semantics></math> 257ms</td><td>-</td></tr><tr><th>ZTRS</th><th>(512, 2048)</th><th>2</th><th>V2-99</th><td>81M</td><td>840</td><td>1.0GB</td><td>193ms</td><td>48.1</td></tr><tr><th></th><th></th><th></th><th>-</th><td>69.5M <math><semantics><mo>∥</mo> <annotation>\|</annotation></semantics></math> 11.5M</td><td>690 <math><semantics><mo>∥</mo> <annotation>\|</annotation></semantics></math> 150</td><td>-</td><td>119ms <math><semantics><mo>∥</mo> <annotation>\|</annotation></semantics></math> 74ms</td><td>-</td></tr><tr><th>DrivoR</th><th>(672, 1148)</th><th>4</th><th>ViT-S</th><td>41M</td><td>351</td><td>0.5GB</td><td>110ms</td><td>48.3</td></tr><tr><th></th><th></th><th></th><th>-</th><td>24M <math><semantics><mo>∥</mo> <annotation>\|</annotation></semantics></math> 17M</td><td>350 <math><semantics><mo>∥</mo> <annotation>\|</annotation></semantics></math> 1</td><td>-</td><td>107ms <math><semantics><mo>∥</mo> <annotation>\|</annotation></semantics></math> 3ms</td><td>-</td></tr><tr><th colspan="6"><sup>†</sup>: RAP is trained on a dataset that is 10 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> larger than navtrain (the default training set).    <sup>‡</sup>: Reproduced.</th><td></td><td></td><td></td></tr></tbody></table>

Table 11: Efficiency. We compare the number of parameters, GFLOPs, peak memory consumption, and throughput w.r.t. the NAVSIM-v2 EPDMS performance. GTRS-A refers to GTRS-Aug and GTRS-D for GTRS-Dense, we decompose the parameters, GFLOPs and throughout in the image backbone and the rest of the network respectively: image backbone $\|$ rest.

## Appendix D Implementation details

We conduct our experiments on a node of 8  $\times$  A100 GPUs. We run each training on 4  $\times$  A100 GPUs with a base learning rate of 0.0002 and batch size of 16, using AdamW optimizer and a cosine annealing learning rate scheduler. The training lasts around 1.5 hours per epoch for the competition split (navtrain + navval) and around 1 hour per epoch for navtrain only. As described in the main paper, we train for 25 epochs for the NAVSIM-v1 (navtrain + navval) and 10 epochs for NAVSIM-v2 navtrain.

## Appendix E Efficiency analysis

Tab. 11 shows the full results of the throughput analysis. We benchmarked each model on a single batch on a single A100 GPU. FLOPs counts were conducted using the FVCore library. Peak memory was counted using Pytorch’s max\_memory\_allocated function. (Note: FVCore currently excludes Scaled Dot Product Attention and thus FLOPs counts do not reflect the real count for each model, though relative performance should remain consistent.) Throughput was counted on a single forward pass after three warm-up iterations, with the final number averaged over 10 iterations. DrivoR has much improved throughput and lower memory consumption, demonstrating the best tradeoff between performance and efficiency.

## Appendix F Single- vs multi- token per trajectory

Previous works such as [^16] have used a single token per trajectory pose, decoding each token into an $(x,y,\theta)$ tuple. We ablate the choice of decoding a single token to a full trajectory vs decoding a set of tokens. Tab. 12 shows that mapping a single token to a trajectory leads to a large jump in performance. Fig. 7 visualizes the same scene, with trajectories decoded either from multi- or single-token. The trajectories mapped from a single token are much smoother and contain less noise, showing that representing trajectories with a single token simplifies learning.

| Trajectory representation | Multi-token | Single-token |
| --- | --- | --- |
| PDMS | 83.9 | 90.0 |

Table 12: Single- vs multi-token trajectory. Quantitative comparison on navval.

![[multi_tok.png|Refer to caption]]

(a) Multi-token

<table><tbody><tr><td>Method</td><td></td><td></td><td>Mod.</td><td>NC</td><td>DAC</td><td>TTC</td><td>Comf.</td><td>EP</td><td>PDMS</td></tr><tr><td>PDM‑Closed</td><td><sup><a href="#fn:11">11</a></sup></td><td>PMLR’23</td><td>-</td><td>94.6</td><td>99.8</td><td>89.9</td><td>86.9</td><td>99.9</td><td>89.1</td></tr><tr><td>Human driver</td><td><sup><a href="#fn:12">12</a></sup></td><td>NeurIPS’24</td><td>-</td><td>100</td><td>100</td><td>100</td><td>99.9</td><td>87.5</td><td>94.8</td></tr><tr><td colspan="10"><em>Test-time training.</em></td></tr><tr><td>Centaur</td><td><sup><a href="#fn:42">42</a></sup></td><td>arXiv’25</td><td>C</td><td>99.5</td><td>98.9</td><td>98.0</td><td>100</td><td>85.9</td><td>92.6</td></tr><tr><td colspan="10"><em>Best-of-N scores.</em></td></tr><tr><td>AutoVLA</td><td><sup><a href="#fn:61">61</a></sup></td><td>NeurIPS’25</td><td>C</td><td>99.1</td><td>97.1</td><td>97.1</td><td>100.0</td><td>87.6</td><td>92.1</td></tr><tr><td>DriveVLA-W0</td><td><sup><a href="#fn:30">30</a></sup></td><td>arXiv’25</td><td>C</td><td>99.3</td><td>97.4</td><td>97.0</td><td>99.9</td><td>88.3</td><td>93.0</td></tr><tr><td>TransDiffuser</td><td><sup><a href="#fn:25">25</a></sup></td><td>arXiv’25</td><td>C+L</td><td>99.4</td><td>96.5</td><td>97.8</td><td>99.4</td><td>94.1</td><td>94.9</td></tr><tr><td colspan="10"><em>Ensemble methods.</em></td></tr><tr><td>Hydra-MDP-C (V2-99)</td><td><sup><a href="#fn:33">33</a></sup></td><td>arXiv’25</td><td>C+L</td><td>98.7</td><td>98.2</td><td>95.0</td><td>100</td><td>86.5</td><td>91.0</td></tr><tr><td colspan="10"><em>Methods using 10× more training data.</em></td></tr><tr><td>RAP-DINO <sup>†</sup></td><td><sup><a href="#fn:15">15</a></sup></td><td>arXiv’25</td><td>C</td><td>99.1</td><td>98.9</td><td>96.7</td><td>100</td><td>90.3</td><td>93.8</td></tr><tr><td colspan="10"><em>Multi-modal methods.</em></td></tr><tr><td>TransFuser</td><td><sup><a href="#fn:8">8</a></sup></td><td>TPAMI’22</td><td>C+L</td><td>97.7</td><td>92.8</td><td>92.8</td><td>100</td><td>79.2</td><td>84.0</td></tr><tr><td>DistillDrive</td><td><sup><a href="#fn:55">55</a></sup></td><td>arXiv’25</td><td>C+L</td><td>98.1</td><td>94.6</td><td>93.6</td><td>100</td><td>81.0</td><td>86.2</td></tr><tr><td>TrajHF (EM)</td><td><sup><a href="#fn:28">28</a></sup></td><td>arXiv’25</td><td>C+L</td><td>96.6</td><td>96.6</td><td>92.1</td><td>100</td><td>84.5</td><td>87.6</td></tr><tr><td>DiffusionDrive</td><td><sup><a href="#fn:36">36</a></sup></td><td>CVPR’25</td><td>C+L</td><td>98.2</td><td>96.2</td><td>94.7</td><td>100</td><td>82.2</td><td>88.1</td></tr><tr><td>WOTE</td><td><sup><a href="#fn:31">31</a></sup></td><td>ICCV’25</td><td>C+L</td><td>98.5</td><td>96.8</td><td>94.9</td><td>99.9</td><td>81.9</td><td>88.3</td></tr><tr><td>Hydra-MDP (V2-99)</td><td><sup><a href="#fn:33">33</a></sup></td><td>arXiv’24</td><td>C+L</td><td>98.0</td><td>97.8</td><td>93.9</td><td>100</td><td>86.5</td><td>90.3</td></tr><tr><td>GoalFlow (V2-99)</td><td><sup><a href="#fn:52">52</a></sup></td><td>CVPR’25</td><td>C+L</td><td>98.4</td><td>98.3</td><td>94.6</td><td>100</td><td>85.0</td><td>90.3</td></tr><tr><td>ResAD (V2-99)</td><td><sup><a href="#fn:59">59</a></sup></td><td>arXiv’25</td><td>C+L</td><td>98.9</td><td>97.8</td><td>94.9</td><td>100</td><td>87.0</td><td>90.6</td></tr><tr><td>SeerDrive (V2-99)</td><td><sup><a href="#fn:57">57</a></sup></td><td>NeurIPS’25</td><td>C+L</td><td>98.8</td><td>98.6</td><td>95.8</td><td>100</td><td>84.2</td><td>90.7</td></tr><tr><td colspan="10"><em>Camera-only methods.</em></td></tr><tr><td>Ego‑stat. MLP</td><td><sup><a href="#fn:12">12</a></sup></td><td>NeurIPS’24</td><td>C</td><td>93.0</td><td>77.3</td><td>83.6</td><td>100</td><td>62.8</td><td>65.6</td></tr><tr><td>UniVLA</td><td><sup><a href="#fn:48">48</a></sup></td><td>arXiv’25</td><td>C</td><td>96.9</td><td>91.1</td><td>91.7</td><td>96.7</td><td>76.8</td><td>81.7</td></tr><tr><td>DrivingGPT</td><td><sup><a href="#fn:7">7</a></sup></td><td>ICCV’24</td><td>C</td><td>98.9</td><td>90.7</td><td>94.9</td><td>95.6</td><td>79.7</td><td>82.4</td></tr><tr><td>UniAD</td><td><sup><a href="#fn:19">19</a></sup></td><td>CVPR’23</td><td>C</td><td>97.8</td><td>91.9</td><td>92.9</td><td>100</td><td>78.8</td><td>83.4</td></tr><tr><td>LTF</td><td><sup><a href="#fn:8">8</a></sup></td><td>TPAMI’22</td><td>C</td><td>97.4</td><td>92.8</td><td>92.4</td><td>100</td><td>79.0</td><td>83.8</td></tr><tr><td>PARA‑Drive</td><td><sup><a href="#fn:49">49</a></sup></td><td>CVPR’24</td><td>C</td><td>97.9</td><td>92.4</td><td>93.0</td><td>99.8</td><td>79.3</td><td>84.0</td></tr><tr><td>DriveX-S</td><td><sup><a href="#fn:41">41</a></sup></td><td>ICCV’25</td><td>C</td><td>97.5</td><td>94.0</td><td>93.0</td><td>100</td><td>79.7</td><td>84.5</td></tr><tr><td>World4Drive</td><td><sup><a href="#fn:58">58</a></sup></td><td>ICCV’25</td><td>C</td><td>97.4</td><td>94.3</td><td>92.8</td><td>100</td><td>79.9</td><td>85.1</td></tr><tr><td>DRAMA</td><td><sup><a href="#fn:56">56</a></sup></td><td>ISRR’24</td><td>C</td><td>98.0</td><td>93.1</td><td>94.8</td><td>100</td><td>80.1</td><td>85.5</td></tr><tr><td>VAD-v2</td><td><sup><a href="#fn:6">6</a></sup></td><td>arXiv’24</td><td>C</td><td>98.1</td><td>94.8</td><td>94.3</td><td>100</td><td>80.6</td><td>86.2</td></tr><tr><td>PRIX</td><td><sup><a href="#fn:50">50</a></sup></td><td>arXiv’25</td><td>C</td><td>98.1</td><td>96.3</td><td>94.1</td><td>100</td><td>82.3</td><td>87.8</td></tr><tr><td>DiffusionDrive</td><td><sup><a href="#fn:36">36</a></sup></td><td>CVPR’25</td><td>C</td><td>98.2</td><td>96.2</td><td>94.7</td><td>100</td><td>82.2</td><td>88.1</td></tr><tr><td>DIVER</td><td><sup><a href="#fn:43">43</a></sup></td><td>arXiv’25</td><td>C</td><td>98.5</td><td>96.5</td><td>94.9</td><td>100</td><td>82.6</td><td>88.3</td></tr><tr><td>AutoVLA</td><td><sup><a href="#fn:61">61</a></sup></td><td>NeurIPS’25</td><td>C</td><td>98.4</td><td>95.6</td><td>98.0</td><td>99.9</td><td>81.9</td><td>89.1</td></tr><tr><td>DriveVLA-W0</td><td><sup><a href="#fn:30">30</a></sup></td><td>arXiv’25</td><td>C</td><td>98.7</td><td>99.1</td><td>95.3</td><td>99.3</td><td>83.3</td><td>90.2</td></tr><tr><td>ReCogDrive</td><td><sup><a href="#fn:32">32</a></sup></td><td>arXiv’25</td><td>C</td><td>97.9</td><td>97.3</td><td>94.9</td><td>100</td><td>87.3</td><td>90.8</td></tr><tr><td>Hydra-MDP++</td><td><sup><a href="#fn:29">29</a></sup></td><td>arXiv’25</td><td>C</td><td>98.6</td><td>98.6</td><td>95.1</td><td>100</td><td>85.7</td><td>91.0</td></tr><tr><td>iPad</td><td><sup><a href="#fn:16">16</a></sup></td><td>arXiv’25</td><td>C</td><td>98.6</td><td>98.3</td><td>94.9</td><td>100</td><td>88.0</td><td>91.7</td></tr><tr><td>DriveSuprim</td><td><sup><a href="#fn:53">53</a></sup></td><td>arXiv’25</td><td>C</td><td>98.6</td><td>98.6</td><td>95.5</td><td>100</td><td>91.3</td><td>93.5</td></tr><tr><td>DrivoR (train)</td><td></td><td></td><td>C</td><td>98.9</td><td>98.3</td><td>96.2</td><td>100</td><td>89.1</td><td>93.1</td></tr><tr><td>DrivoR (trainval)</td><td></td><td></td><td>C</td><td>99.0</td><td>98.9</td><td>96.7</td><td>100</td><td>90.0</td><td>93.7</td></tr><tr><td>DrivoR (+65k SimScale data)</td><td></td><td></td><td>C</td><td>99.1</td><td>99.0</td><td>96.9</td><td>100</td><td>90.3</td><td>94.0</td></tr><tr><td>DrivoR (+134k SimScale data)</td><td></td><td></td><td>C</td><td>99.1</td><td>99.2</td><td>96.9</td><td>100</td><td>91.6</td><td>94.6</td></tr><tr><td colspan="9"><sup>†</sup>: RAP <sup><a href="#fn:15">15</a></sup> is trained on a dataset that is 10 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> larger than navtrain (the default training set).</td><td></td></tr></tbody></table>

Table 13: NAVSIM-v1 scores. Full comparison to existing methods, possibly with different modalities (Mod.), on the NAVSIM-v1 benchmark on test set (navtest). ‘C’ refers to camera, and ‘L’ to LiDAR.

## Appendix G Expanded NAVSIM-v1 results

We provide in Tab. 13 a full comparison to state-of-the-art methods, including methods with LiDAR inputs and with post-processing such test-time training, Best-of-N scoring and ensembling. We note that the Best-of-N practice aims to score all predicted trajectories with ground truth and select the best trajectory based on the scores from the evaluation, which is not aligned with the evaluation protocol defined in NAVSIM [^12] [^5].

We note that the goal of DrivoR is to provide a simple and efficient baseline model with registers for end-to-end driving. Without bells and whistles, DrivoR still achieves competitive performance compared to methods with post-processing techniques or additional sensor modality.

## Appendix H NAVSIM-v2 results before benchmark fix

We provide in Tab. 14 the full NAVSIM-v2 state-of-the-art comparison, including the results of methods before the benchmark bug fix (Issue #151 in NAVSIM official GitHub), which is related to the failure of filtering out human driver errors.

From the table, we see that DrivoR ranks among the best-performing methods, with a much lighter model design (see more detail in App. E) and without extra training data. Also, DrivoR ranks first among state-of-the-art methods after the bug fix. This indicates that the errors made by DrivoR are mostly due to human errors, which are not penalized after the fix according to the NAVSIM-v2 benchmark [^5].

<table><tbody><tr><td></td><td></td><td colspan="9">Stage 1</td><td colspan="9">Stage 2</td><td></td></tr><tr><td>Method</td><td></td><td>NC</td><td>DAC</td><td>DDC</td><td>TLC</td><td>EP</td><td>TTC</td><td>LK</td><td>HC</td><td>EC</td><td>NC</td><td>DAC</td><td>DDC</td><td>TLC</td><td>EP</td><td>TTC</td><td>LK</td><td>HC</td><td>EC</td><td>EPDMS</td></tr><tr><td colspan="21"><em>Results before official metric bug fixing.</em></td></tr><tr><td>PDM-Closed</td><td><sup><a href="#fn:11">11</a></sup></td><td>94.4</td><td>98.8</td><td>100</td><td>99.5</td><td>100</td><td>93.5</td><td>99.3</td><td>97.7</td><td>36.0</td><td>88.1</td><td>90.6</td><td>96.3</td><td>98.5</td><td>100</td><td>83.1</td><td>73.7</td><td>91.5</td><td>25.4</td><td>51.3</td></tr><tr><td>Const. Vel.</td><td><sup><a href="#fn:12">12</a></sup></td><td>88.8</td><td>42.8</td><td>70.6</td><td>99.3</td><td>77.5</td><td>87.3</td><td>78.6</td><td>97.1</td><td>60.4</td><td>83.2</td><td>59.1</td><td>76.5</td><td>98.0</td><td>71.3</td><td>81.1</td><td>47.9</td><td>97.1</td><td>61.9</td><td>10.9</td></tr><tr><td>Ego Hist. MLP</td><td><sup><a href="#fn:12">12</a></sup></td><td>93.2</td><td>55.7</td><td>86.6</td><td>99.3</td><td>81.2</td><td>92.2</td><td>83.5</td><td>97.5</td><td>77.7</td><td>77.2</td><td>51.9</td><td>74.4</td><td>98.2</td><td>77.1</td><td>75.0</td><td>40.8</td><td>97.8</td><td>79.8</td><td>12.7</td></tr><tr><td>LTF</td><td><sup><a href="#fn:8">8</a></sup></td><td>96.2</td><td>79.6</td><td>99.1</td><td>99.6</td><td>84.1</td><td>95.1</td><td>94.2</td><td>97.6</td><td>79.1</td><td>77.8</td><td>70.2</td><td>84.3</td><td>98.1</td><td>85.1</td><td>75.7</td><td>45.4</td><td>95.8</td><td>76.0</td><td>23.1</td></tr><tr><td>RAP-DINO (ViT-H) <sup>†</sup></td><td><sup><a href="#fn:15">15</a></sup></td><td>97.1</td><td>94.4</td><td>98.8</td><td>99.8</td><td>83.9</td><td>96.9</td><td>94.7</td><td>96.4</td><td>66.2</td><td>83.2</td><td>83.9</td><td>87.4</td><td>98.0</td><td>86.9</td><td>80.4</td><td>52.3</td><td>95.2</td><td>52.4</td><td>36.9</td></tr><tr><td>GTRS-D (V2-99)</td><td><sup><a href="#fn:35">35</a></sup></td><td>98.7</td><td>91.4</td><td>95.8</td><td>89.2</td><td>99.4</td><td>94.4</td><td>99.3</td><td>98.8</td><td>72.8</td><td>69.5</td><td>98.7</td><td>90.1</td><td>95.1</td><td>54.6</td><td>96.9</td><td>94.1</td><td>40.4</td><td>49.7</td><td>41.7</td></tr><tr><td>GTRS-A (V2-99)</td><td><sup><a href="#fn:35">35</a></sup></td><td>98.9</td><td>87.9</td><td>95.1</td><td>88.8</td><td>99.2</td><td>89.6</td><td>99.6</td><td>98.8</td><td>76.1</td><td>80.3</td><td>99.1</td><td>86.0</td><td>94.7</td><td>53.5</td><td>97.6</td><td>97.1</td><td>54.2</td><td>56.1</td><td>42.1</td></tr><tr><td>GTRS-DrivoR (ViT-S) <sup>∗</sup></td><td></td><td>98.0</td><td>95.8</td><td>99.7</td><td>99.3</td><td>72.9</td><td>98.2</td><td>95.6</td><td>96.9</td><td>51.6</td><td>91.6</td><td>86.7</td><td>90.2</td><td>98.8</td><td>73.2</td><td>88.9</td><td>51.9</td><td>94.9</td><td>46.4</td><td>42.3</td></tr><tr><td>GTRS-D EVA-ViT-L</td><td><sup><a href="#fn:35">35</a></sup></td><td>97.6</td><td>95.8</td><td>99.8</td><td>99.0</td><td>77.2</td><td>97.8</td><td>95.3</td><td>97.3</td><td>46.7</td><td>91.9</td><td>91.3</td><td>92.7</td><td>99.0</td><td>72.7</td><td>90.4</td><td>53.8</td><td>94.1</td><td>41.6</td><td>43.4</td></tr><tr><td>GTRS-A (ViT-L)</td><td><sup><a href="#fn:35">35</a></sup></td><td>98.7</td><td>98.0</td><td>99.1</td><td>99.8</td><td>75.9</td><td>98.7</td><td>94.7</td><td>97.6</td><td>49.8</td><td>89.5</td><td>89.6</td><td>92.9</td><td>98.5</td><td>78.9</td><td>86.4</td><td>55.3</td><td>96.5</td><td>52.7</td><td>44.7</td></tr><tr><td>DriveSuprim (EVA-ViT-L)</td><td><sup><a href="#fn:53">53</a></sup></td><td>98.7</td><td>98.0</td><td>99.1</td><td>99.8</td><td>75.9</td><td>98.7</td><td>94.7</td><td>97.6</td><td>49.8</td><td>89.5</td><td>89.6</td><td>92.9</td><td>98.5</td><td>78.9</td><td>86.4</td><td>55.3</td><td>96.5</td><td>52.7</td><td>44.7</td></tr><tr><td>GTRS-D (ViT-L)</td><td><sup><a href="#fn:35">35</a></sup></td><td>98.9</td><td>98.2</td><td>99.8</td><td>99.6</td><td>73.9</td><td>98.9</td><td>95.3</td><td>97.3</td><td>40.0</td><td>91.5</td><td>90.8</td><td>94.7</td><td>98.5</td><td>70.8</td><td>90.1</td><td>55.4</td><td>97.2</td><td>54.2</td><td>45.3</td></tr><tr><td>DrivoR (ViT-S)</td><td></td><td>98.8</td><td>95.1</td><td>98.9</td><td>100</td><td>72.6</td><td>98.7</td><td>94.0</td><td>97.6</td><td>73.3</td><td>90.2</td><td>88.4</td><td>91.9</td><td>98.6</td><td>69.8</td><td>88.0</td><td>50.1</td><td>98.5</td><td>76.2</td><td>45.3</td></tr><tr><td>ZTRS (V2-99)</td><td><sup><a href="#fn:34">34</a></sup></td><td>98.9</td><td>97.6</td><td>100.0</td><td>100.0</td><td>66.7</td><td>98.9</td><td>96.2</td><td>96.7</td><td>44.0</td><td>91.1</td><td>90.4</td><td>95.8</td><td>99.0</td><td>63.6</td><td>89.8</td><td>60.4</td><td>97.6</td><td>66.1</td><td>45.5</td></tr><tr><td colspan="21"><em>Results after official metric bug fixing.</em></td></tr><tr><td>RAP-DINO (ViT-H) <sup>†</sup></td><td><sup><a href="#fn:15">15</a></sup></td><td>97.1</td><td>94.4</td><td>98.8</td><td>99.8</td><td>83.9</td><td>96.9</td><td>94.7</td><td>96.4</td><td>66.2</td><td>83.2</td><td>83.9</td><td>87.4</td><td>98.0</td><td>86.9</td><td>80.4</td><td>52.3</td><td>95.2</td><td>52.4</td><td>39.6</td></tr><tr><td>GTRS-D (V2-99)</td><td><sup><a href="#fn:35">35</a></sup></td><td>98.9</td><td>96.2</td><td>99.4</td><td>99.3</td><td>72.9</td><td>98.9</td><td>95.1</td><td>96.9</td><td>39.1</td><td>91.2</td><td>89.4</td><td>94.4</td><td>98.8</td><td>69.5</td><td>90.0</td><td>54.3</td><td>94.0</td><td>48.7</td><td>45.0</td></tr><tr><td>GTRS-A (V2-99)</td><td><sup><a href="#fn:35">35</a></sup></td><td>98.9</td><td>95.1</td><td>99.1</td><td>99.6</td><td>76.2</td><td>99.1</td><td>94.9</td><td>97.6</td><td>54.2</td><td>88.1</td><td>88.8</td><td>89.3</td><td>98.9</td><td>98.9</td><td>85.9</td><td>53.7</td><td>96.8</td><td>56.9</td><td>45.4</td></tr><tr><td>GTRS-DrivoR (ViT-S) <sup>∗</sup></td><td></td><td>98.0</td><td>95.8</td><td>99.7</td><td>99.3</td><td>72.9</td><td>98.2</td><td>95.6</td><td>96.9</td><td>51.6</td><td>91.6</td><td>86.7</td><td>90.2</td><td>98.8</td><td>73.2</td><td>88.9</td><td>51.9</td><td>94.9</td><td>46.4</td><td>45.8</td></tr><tr><td>ZTRS (V2-99)</td><td><sup><a href="#fn:34">34</a></sup></td><td>98.9</td><td>97.6</td><td>100</td><td>100</td><td>66.7</td><td>98.9</td><td>96.2</td><td>96.7</td><td>44.0</td><td>91.1</td><td>90.4</td><td>95.8</td><td>99.0</td><td>63.6</td><td>89.8</td><td>60.4</td><td>97.6</td><td>66.1</td><td>48.1</td></tr><tr><td>DrivoR (ViT-S)</td><td></td><td>98.8</td><td>95.1</td><td>98.9</td><td>100</td><td>72.6</td><td>98.7</td><td>94.0</td><td>97.6</td><td>73.3</td><td>90.2</td><td>88.4</td><td>91.9</td><td>98.6</td><td>70.0</td><td>88.0</td><td>50.1</td><td>98.5</td><td>76.2</td><td>48.3</td></tr><tr><td>DrivoR (+65k SimScale data, ViT-S)</td><td></td><td>98.9</td><td>97.3</td><td>99.2</td><td>99.6</td><td>77.7</td><td>99.1</td><td>95.3</td><td>97.6</td><td>68.4</td><td>92.3</td><td>92.2</td><td>97.0</td><td>99.0</td><td>72.1</td><td>90.3</td><td>56.3</td><td>97.1</td><td>38.8</td><td>52.3</td></tr><tr><td>SimScale (+185k SimScale data, V2-99)</td><td><sup><a href="#fn:45">45</a></sup></td><td>99.6</td><td>99.16</td><td>99.9</td><td>100</td><td>69.6</td><td>99.6</td><td>95.8</td><td>95.6</td><td>28.4</td><td>94.5</td><td>94.2</td><td>95.8</td><td>99.2</td><td>75.8</td><td>92.8</td><td>60.1</td><td>96.1</td><td>43.2</td><td>53.2</td></tr><tr><td>DrivoR (+134k SimScale data, ViT-S)</td><td></td><td>99.1</td><td>98.2</td><td>99.3</td><td>99.8</td><td>75.4</td><td>98.7</td><td>94.9</td><td>97.6</td><td>70.2</td><td>92.3</td><td>91.6</td><td>97.3</td><td>99.1</td><td>75.7</td><td>90.6</td><td>56.1</td><td>98.4</td><td>44.7</td><td>54.6</td></tr><tr><td colspan="19"><sup>†</sup>: RAP <sup><a href="#fn:15">15</a></sup> is trained on a dataset that is 10 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> larger than navtrain (the default training set).</td><td></td><td></td></tr><tr><td colspan="19"><sup>∗</sup>: same ViT-S backbone with DrivoR registers, the prediction and scoring heads remain the same as in GTRS.</td><td></td><td></td></tr></tbody></table>

Table 14: NAVSIM-v2 navhard-two-stage. Full comparison to other methods on the NAVSIM-v2 benchmark test set using the EPDMS. GTRS-A refers to GTRS-Aug and GTRS-D for GTRS-Dense.

## Appendix I Limitations and additional visualizations

Fig. 8, Fig. 9, and Fig. 10 show DrivoR navigating diverse scenarios, and show the camera focus between the scoring and prediction pathways.

Fig. 9shows a surprising case where the scoring attention is mainly focused on the back camera, despite the visible traffic light. We hypothesize that adding traffic light compliance as a score component, as in the EPDMS, could force the scoring modules attention to focus more on the traffic light as it crosses the intersection. However we highlight that despite no active scoring on this component, DrivoR still displays very high TLC score.

Fig. 10shows a failure case, highlighting the challenge of predicting viable trajectories without historical (past) frames. DrivoR operates only on the current timestep, which makes examples like Fig. 10 challenging, due to the ambiguous nature of the vehicle’s position in the scene. The front camera does not contain easily interpretable objects, and we see the model focusing entirely on the right camera. The resulting trajectories undercut the turn, and could result in the model driving in the wrong way. Future work could include the use of historical frames for planning, potentially alleviating these ambiguous failure cases.

Finally, Fig. 11 compares the predicted and selected trajectories of the two models on the same scene from the NAVSIM-v1 validation set. We can see that the trajectories of our default trained NAVSIM-v1 agent are more aggressive, traveling faster and with less spread than the trajectories from our agent tuned to have more passive behavior. The “passive” agent is however better able to navigate the out-of-distribution (OOD) scenes in NAVSIM-v2.

![[right_turn_bev_with_images.png|Refer to caption]]

Figure 8: Visualization of generated trajectories (right turn) in a NAVSIM-v1 val set scene. Right turn scenario with right camera as dominant camera used in trajectory scoring and generation.

![[no_tlc_bev_with_images.png|Refer to caption]]

Figure 9: Visualization of generated trajectories (intersection crossing) in a NAVSIM-v1 val set scene. The ego-vehicle crosses a signalized intersection, but the scoring attention focuses on the rear camera. We hypothesize that the inclusion of the Traffic Light Compliance (TLC) scoring could help in such a case. As a matter of fact, our TLC score on NAVSIM-v2 navhard-two-stage are high ( Tab. 14 ).

![[failure_mode_bev_with_images.png|Refer to caption]]

Figure 10: Visualization of generated trajectories (failure case) in a NAVSIM-v1 val set scene. Failure case highlighting difficulty of navigation without historical (past) frames. The front camera image is very ambiguous, resulting in full focus on the right camera and trajectories which could result in wrong-way driving.

![[aggressive_agent_trajs.png|Refer to caption]]

Figure 11: Visualization of generated trajectories (agent behavior re-weighting) in a NAVSIM-v1 val set scene.. Left: trajectories are generated from an agent using default PDMS weights on score components. Right: the agent uses our tuned weights. Note the much shorter and less aggressive trajectories of the agent in the second (right) setting, which are better suited for navigating the out-of-distribution (OOD) scenes in the NAVSIM-v2 evaluation.

[^1]: M. R. Barın, G. Aydemir, and F. Güney (2024) Robust bird’s eye view segmentation by adapting dinov2. In ECCV Workshops, Cited by: §4, §4.2.1.

[^2]: D. Bolya, C. Fu, X. Dai, P. Zhang, C. Feichtenhofer, and J. Hoffman (2023) Token merging: your vit but faster. In ICLR, Cited by: §2.

[^3]: H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom (2020) NuScenes: A multimodal dataset for autonomous driving. In CVPR, Cited by: §4.1, §4.1.

[^4]: H. Caesar, J. Kabzan, K. S. Tan, W. K. Fong, E. Wolff, A. Lang, L. Fletcher, O. Beijbom, and S. Omari (2021) Nuplan: a closed-loop ml-based planning benchmark for autonomous vehicles. In CVPR Workshops, Cited by: §4.1.

[^5]: W. Cao, M. Hallgarten, T. Li, D. Dauner, X. Gu, C. Wang, Y. Miron, M. Aiello, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta (2025) Pseudo-simulation for autonomous driving. In CoRL, Cited by: Appendix G, Appendix H, §1, §2, §4.1.

[^6]: S. Chen, B. Jiang, H. Gao, B. Liao, Q. Xu, Q. Zhang, C. Huang, W. Liu, and X. Wang (2024) Vadv2: end-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243. Cited by: Table 13, §1, §2, Table 1.

[^7]: Y. Chen, Y. Wang, and Z. Zhang (2025) Drivinggpt: unifying driving world modeling and planning with multi-modal autoregressive transformers. In ICCV, Cited by: Table 13, Table 1.

[^8]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger (2022) Transfuser: imitation with transformer-based sensor fusion for autonomous driving. TPAMI. Cited by: Table 13, Table 13, Table 14, §2, §2, Table 1, Table 2.

[^9]: O. Contributors (2023) Openscene: the largest up-to-date 3d occupancy prediction benchmark in autonomous driving. In CVPR, Cited by: §4.1.

[^10]: T. Darcet, M. Oquab, J. Mairal, and P. Bojanowski (2024) Vision transformers need registers. In ICLR, Cited by: §2.

[^11]: D. Dauner, M. Hallgarten, A. Geiger, and K. Chitta (2023) Parting with misconceptions about learning-based vehicle motion planning. In CoRL, Cited by: Table 13, Table 14, §4.1, §4.1, Table 1.

[^12]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, et al. (2024) Navsim: data-driven non-reactive autonomous vehicle simulation and benchmarking. In NeurIPS, Cited by: Appendix C, Table 13, Table 13, Appendix G, Table 14, Table 14, §1, §2, §3.4, §4.1, Table 1, Table 1.

[^13]: A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby (2021) An image is worth 16x16 words: transformers for image recognition at scale. In ICLR, Cited by: §1, §2, §3.1.

[^14]: Y. Fang, W. Wang, B. Xie, Q. Sun, L. Wu, X. Wang, T. Huang, X. Wang, and Y. Cao (2023) Eva: exploring the limits of masked visual representation learning at scale. In CVPR, Cited by: §1.

[^15]: L. Feng, Y. Gao, E. Zablocki, Q. Li, W. Li, S. Liu, M. Cord, and A. Alahi (2025) RAP: 3d rasterization augmented end-to-end planning. arXiv preprint arXiv:2510.04333. Cited by: Table 13, Table 13, Table 14, Table 14, Table 14, §1, §2, Figure 3, Figure 3, §4.1, §4.1, §4.2.1, Table 1, Table 1, Table 3.

[^16]: K. Guo, H. Liu, X. Wu, J. Pan, and C. Lv (2025) IPad: iterative proposal-centric end-to-end autonomous driving. arXiv preprint arXiv:2505.15111. Cited by: Table 13, Appendix F, §1, §2, §2, §3.3, §4.2.1, Table 1.

[^17]: A. Guzman-Rivera, D. Batra, and P. Kohli (2012) Multiple choice learning: learning to produce multiple structured outputs. In NeurIPS, Cited by: §3.3.

[^18]: E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, et al. (2022) Lora: low-rank adaptation of large language models.. In ICLR, Cited by: §4, §4.2.1.

[^19]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, L. Lu, X. Jia, Q. Liu, J. Dai, Y. Qiao, and H. Li (2023) Planning-oriented autonomous driving. In CVPR, Cited by: Table 13, §1, §2, Table 1, Table 2.

[^20]: B. Ivanovic, C. Saltori, Y. You, Y. Wang, W. Luo, and M. Pavone (2025) Efficient multi-camera tokenization with triplanes for end-to-end driving. arXiv preprint arXiv:2506.12251. Cited by: §2.

[^21]: A. Jaegle, S. Borgeaud, J. Alayrac, C. Doersch, C. Ionescu, D. Ding, S. Koppula, D. Zoran, A. Brock, E. Shelhamer, O. J. Hénaff, M. M. Botvinick, A. Zisserman, O. Vinyals, and J. Carreira (2022) Perceiver IO: A general architecture for structured inputs & outputs. In ICLR, Cited by: §2, §3.2.

[^22]: A. Jaegle, F. Gimeno, A. Brock, O. Vinyals, A. Zisserman, and J. Carreira (2021) Perceiver: general perception with iterative attention. In ICML, M. Meila and T. Zhang (Eds.), Cited by: §3.2.

[^23]: X. Jia, J. You, Z. Zhang, and J. Yan (2025) Drivetransformer: unified transformer for scalable end-to-end autonomous driving. arXiv preprint arXiv:2503.07656. Cited by: §2.

[^24]: B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang (2023) Vad: vectorized scene representation for efficient autonomous driving. In ICCV, Cited by: §2, Table 2.

[^25]: X. Jiang, Y. Ma, P. Li, L. Xu, X. Wen, K. Zhan, Z. Xia, P. Jia, X. Lang, and S. Sun (2025) TransDiffuser: end-to-end trajectory generation with decorrelated multi-modal representation for autonomous driving. arXiv preprint arXiv:2505.09315. Cited by: Table 13.

[^26]: A. Kolesnikov, L. Beyer, X. Zhai, J. Puigcerver, J. Yung, S. Gelly, and N. Houlsby (2020) Big transfer (bit): general visual representation learning. In ECCV, Cited by: §4.2.1.

[^27]: Y. Lee, J. Hwang, S. Lee, Y. Bae, and J. Park (2019) An energy and gpu-computation efficient backbone network for real-time object detection. In CVPR Workshops, Cited by: §1.

[^28]: D. Li, C. Li, Y. Wang, J. Ren, X. Wen, P. Li, L. Xu, K. Zhan, P. Jia, X. Lang, N. Xu, and H. Zhao (2025) Learning personalized driving styles via reinforcement learning from human feedback. External Links: 2503.10434, [Link](https://arxiv.org/abs/2503.10434) Cited by: Table 13.

[^29]: K. Li, Z. Li, S. Lan, Y. Xie, Z. Zhang, J. Liu, Z. Wu, Z. Yu, and J. M. Alvarez (2025) Hydra-mdp++: advancing end-to-end driving via expert-guided hydra-distillation. arXiv e-prints, pp. arXiv–2503. Cited by: Table 13, §1, §2, Table 1.

[^30]: Y. Li, S. Shang, W. Liu, B. Zhan, H. Wang, Y. Wang, Y. Chen, X. Wang, Y. An, C. Tang, et al. (2025) DriveVLA-w0: world models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796. Cited by: Table 13, Table 13, Table 1.

[^31]: Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang (2025) End-to-end driving with online trajectory evaluation via bev world model. ICCV. Cited by: Table 13.

[^32]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, et al. (2025) ReCogDrive: a reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052. Cited by: Table 13, §2, §2, Table 1.

[^33]: Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y. Ji, Z. Li, Z. Zhu, J. Kautz, Z. Wu, et al. (2024) Hydra-mdp: end-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978. Cited by: Table 13, Table 13, §1.

[^34]: Z. Li, W. Yao, Z. Wang, X. Sun, J. Chen, N. Chang, M. Shen, J. Song, Z. Wu, S. Lan, et al. (2025) ZTRS: zero-imitation end-to-end autonomous driving with trajectory scoring. arXiv preprint arXiv:2510.24108. Cited by: Table 14, Table 14, §2, Table 3.

[^35]: Z. Li, W. Yao, Z. Wang, X. Sun, J. Chen, N. Chang, M. Shen, Z. Wu, S. Lan, and J. M. Alvarez (2025) Generalized trajectory scoring for end-to-end multimodal planning. arXiv preprint arXiv:2506.06664. Cited by: Table 14, Table 14, Table 14, Table 14, Table 14, Table 14, Table 14, §1, §2, §2, §4.2.1, Table 3, Table 3.

[^36]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al. (2025) Diffusiondrive: truncated diffusion model for end-to-end autonomous driving. In CVPR, pp. 12037–12047. Cited by: Table 13, Table 13, §1, §2, §2, §2, Table 1.

[^37]: Y. Liao, J. Xie, and A. Geiger (2023) KITTI-360: A novel dataset and benchmarks for urban scene understanding in 2d and 3d. TPAMI. Cited by: §4.1.

[^38]: T. M. Moerland, J. Broekens, A. Plaat, C. M. Jonker, et al. (2023) Model-based reinforcement learning: a survey. Foundations and Trends in ML. Cited by: §1.

[^39]: M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, M. Assran, N. Ballas, W. Galuba, R. Howes, P. Huang, S. Li, I. Misra, M. Rabbat, V. Sharma, G. Synnaeve, H. Xu, H. Jégou, J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski (2024) DINOv2: learning robust visual features without supervision. TMLR. Cited by: §1, §4, §4.2.1.

[^40]: L. Rowe, R. Girgis, A. Gosselin, B. Carrez, F. Golemo, F. Heide, L. Paull, and C. Pal (2024) CtRL-sim: reactive and controllable driving agents with offline reinforcement learning. In CoRL, Cited by: §3.4.

[^41]: C. Shi, S. Shi, K. Sheng, B. Zhang, and L. Jiang (2025) DriveX: omni scene modeling for learning generalizable world knowledge in autonomous driving. In ICCV, Cited by: Table 13, Table 1.

[^42]: C. Sima, K. Chitta, Z. Yu, S. Lan, P. Luo, A. Geiger, H. Li, and J. M. Alvarez (2025) Centaur: robust end-to-end autonomous driving with test-time training. arXiv preprint arXiv:2503.11650. Cited by: Table 13, Table 1.

[^43]: Z. Song, L. Liu, H. Pan, B. Liao, M. Guo, L. Yang, Y. Zhang, S. Xu, C. Jia, and Y. Luo (2025) Breaking imitation bottlenecks: reinforced diffusion powers diverse trajectory generation. arXiv preprint arXiv:2507.04049. Cited by: Table 13, Table 1.

[^44]: P. Sun, H. Kretzschmar, X. Dotiwalla, A. Chouard, V. Patnaik, P. Tsui, J. Guo, Y. Zhou, Y. Chai, B. Caine, V. Vasudevan, W. Han, J. Ngiam, H. Zhao, A. Timofeev, S. Ettinger, M. Krivokon, A. Gao, A. Joshi, Y. Zhang, J. Shlens, Z. Chen, and D. Anguelov (2020) Scalability in perception for autonomous driving: waymo open dataset. In CVPR, Cited by: §4.1.

[^45]: H. Tian, T. Li, H. Liu, J. Yang, Y. Qiu, G. Li, J. Wang, Y. Gao, Z. Zhang, L. Wang, H. Ye, T. Tan, L. Chen, and H. Li (2025) SimScale: learning to drive via real-world simulation at scale. arXiv preprint arXiv:2511.23369. Cited by: Table 14, §4.1, Table 3.

[^46]: A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin (2017) Attention is all you need. In NeurIPS, Cited by: §3.1, §3.3.

[^47]: Y. Wang, W. Luo, J. Bai, Y. Cao, T. Che, K. Chen, Y. Chen, J. Diamond, Y. Ding, W. Ding, et al. (2025) Alpamayo-r1: bridging reasoning and action prediction for generalizable autonomous driving in the long tail. arXiv preprint arXiv:2511.00088. Cited by: §2.

[^48]: Y. Wang, X. Li, W. Wang, J. Zhang, Y. Li, Y. Chen, X. Wang, and Z. Zhang (2025) Unified vision-language-action model. arXiv preprint arXiv:2506.19850. Cited by: Table 13, Table 1.

[^49]: X. Weng, B. Ivanovic, Y. Wang, Y. Wang, and M. Pavone (2024) PARA-drive: parallelized architecture for real-time autonomous driving. In CVPR, Cited by: Table 13, §1, Table 1.

[^50]: M. K. Wozniak, L. Liu, Y. Cai, and P. Jensfelt (2025) PRIX: learning to plan from raw pixels for end-to-end autonomous driving. arXiv preprint arXiv:2507.17596. Cited by: Table 13, Table 1.

[^51]: P. Xiao, Z. Shao, S. Hao, Z. Zhang, X. Chai, J. Jiao, Z. Li, J. Wu, K. Sun, K. Jiang, Y. Wang, and D. Yang (2021) PandaSet: advanced sensor suite dataset for autonomous driving. In ITSC, Cited by: §4.1.

[^52]: Z. Xing, X. Zhang, Y. Hu, B. Jiang, T. He, Q. Zhang, X. Long, and W. Yin (2025) Goalflow: goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In CVPR, Cited by: Table 13, §1.

[^53]: W. Yao, Z. Li, S. Lan, Z. Wang, X. Sun, J. M. Alvarez, and Z. Wu (2025) DriveSuprim: towards precise trajectory selection for end-to-end planning. arXiv preprint arXiv:2506.06659. Cited by: Table 13, Table 14, §2, Table 1.

[^54]: Q. Yu, M. Weber, X. Deng, X. Shen, D. Cremers, and L. Chen (2024) An image is worth 32 tokens for reconstruction and generation. In NeurIPS, Cited by: §1, §2.

[^55]: R. Yu, X. Zhang, R. Zhao, H. Yan, and M. Wang (2025) DistillDrive: end-to-end multi-mode autonomous driving distillation by isomorphic hetero-source planning model. arXiv preprint arXiv:2508.05402. Cited by: Table 13.

[^56]: C. Yuan, Z. Zhang, J. Sun, S. Sun, Z. Huang, C. D. W. Lee, D. Li, Y. Han, A. Wong, K. P. Tee, et al. (2024) Drama: an efficient end-to-end motion planner for autonomous driving with mamba. In ISRR, Cited by: Table 13, Table 1.

[^57]: B. Zhang, N. Song, J. Li, X. Zhu, J. Deng, and L. Zhang (2025) Future-aware end-to-end driving: bidirectional modeling of trajectory planning and scene evolution. NeurIPS. Cited by: Table 13.

[^58]: Y. Zheng, P. Yang, Z. Xing, Q. Zhang, Y. Zheng, Y. Gao, P. Li, T. Zhang, Z. Xia, P. Jia, et al. (2025) World4Drive: end-to-end autonomous driving via intention-aware physical latent world model. In ICCV, Cited by: Table 13, Table 1.

[^59]: Z. Zheng, S. Chen, H. Yin, X. Zhang, J. Zou, X. Wang, Q. Zhang, and L. Zhang (2025) ResAD: normalized residual trajectory modeling for end-to-end autonomous driving. arXiv preprint arXiv:2510.08562. Cited by: Table 13.

[^60]: H. Zhou, L. Lin, J. Wang, Y. Lu, D. Bai, B. Liu, Y. Wang, A. Geiger, and Y. Liao (2024) HUGSIM: A real-time, photo-realistic and closed-loop simulator for autonomous driving. CoRR abs/2412.01718. External Links: [Link](https://doi.org/10.48550/arXiv.2412.01718), [Document](https://dx.doi.org/10.48550/ARXIV.2412.01718), 2412.01718 Cited by: §1, §2, §4.1, Table 2, Table 2.

[^61]: Z. Zhou, T. Cai, S. Z. Zhao, Y. Zhang, Z. Huang, B. Zhou, and J. Ma (2025) AutoVLA: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. NeurIPS. Cited by: Table 13, Table 13, Table 1.