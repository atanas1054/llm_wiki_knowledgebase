---
title: "Drive-JEPA: Video JEPA Meets Multimodal Trajectory Distillation for End-to-End Driving"
source: "https://arxiv.org/html/2601.22032v1"
author:
published:
created: 2026-05-01
description:
tags:
  - "clippings"
---
Linhan Wang    Zichong Yang    Chen Bai    Guoxiang Zhang    Xiaotong Liu    Xiaoyin Zheng    Xiao-Xiao Long    Chang-Tien Lu    Cheng Lu

###### Abstract

End-to-end autonomous driving increasingly leverages self-supervised video pretraining to learn transferable planning representations. However, pretraining video world models for scene understanding has so far brought only limited improvements. This limitation is compounded by the inherent ambiguity of driving: each scene typically provides only a single human trajectory, making it difficult to learn multimodal behaviors. In this work, we propose Drive-JEPA, a framework that integrates Video Joint-Embedding Predictive Architecture (V-JEPA) with multimodal trajectory distillation for end-to-end driving. First, we adapt V-JEPA for end-to-end driving, pretraining a ViT encoder on large-scale driving videos to produce predictive representations aligned with trajectory planning. Second, we introduce a proposal-centric planner that distills diverse simulator-generated trajectories alongside human trajectories, with a momentum-aware selection mechanism to promote stable and safe behavior. When evaluated on NAVSIM, the V-JEPA representation combined with a simple transformer-based decoder outperforms prior methods by 3 PDMS in the perception-free setting. The complete Drive-JEPA framework achieves 93.3 PDMS on v1 and 87.8 EPDMS on v2, setting a new state-of-the-art. The code is available at [https://github.com/linhanwang/Drive-JEPA](https://github.com/linhanwang/Drive-JEPA).

Computer vision, End-to-end autonomous driving, V-JEPA

## 1 Introduction

End-to-end autonomous driving [^34] [^11] [^22] has emerged as a promising paradigm that directly maps raw sensor observations to driving actions using a unified neural model. By eliminating hand-designed intermediate representations used in traditional modular pipelines, end-to-end approaches aim to reduce information loss and improve scalability by learning directly from large collections of human driving data.

![[teaser_v2.png|Refer to caption]]

Figure 1: Comparison between end-to-end planners on both perception-free and perception-based settings.

Recently, end-to-end autonomous driving increasingly seeks to leverage self-supervised video pretraining to learn transferable representations for planning. However, pretraining video world models for scene understanding has so far brought only limited improvements. Existing approaches in this direction largely fall into two categories. First, video-generative methods, such as VaVAM [^5] and Epona [^45], learn representations by reconstructing or generating videos and then transfer them to planning, but this pixel-level objective incurs heavy computation and may over-emphasize visual details that are irrelevant to decision making. Second, to reduce cost, latent world models predict compact feature dynamics (e.g., LAW [^28] predicts feature $T{+}1$ from feature $T$, and World4Drive [^46] further introduces pretrained foundation models to enrich latent targets. However, these latent approaches are typically used as auxiliary objectives and have not demonstrated clear benefits from scaling up pretraining.

Orthogonally, end-to-end driving faces a supervision bottleneck: each scene typically provides only a single human trajectory, despite inherently multimodal futures. Prior works address this by generating multimodal trajectories through either discrete or continuous formulations. Discrete approaches such as VAD v2 [^10] and Hydra-MDP [^29] cluster trajectories into a fixed vocabulary and predict scores reflecting safety and comfort; however, their expressiveness is fundamentally limited by the coverage and quality of anchor trajectories, leading to poor generalization in out-of-vocabulary scenarios [^31]. Alternatively, diffusion-based methods, including DiffusionDrive [^31] and GoalFlow [^41], model multimodal trajectory distributions via iterative sampling, which has shown strong generative capability. Nevertheless, these approaches remain constrained by supervision from single human trajectories per scene, inherently limiting the diversity of learned behaviors.

In this work, we propose Drive-JEPA, an end-to-end autonomous driving framework that addresses the above two bottlenecks in a unified way. First, we adapt V-JEPA [^2] [^3] to the driving domain to learn planning-aligned predictive representations from large-scale raw videos, improving transfer beyond prior world-model pretraining. Second, we introduce multimodal trajectory distillation that distills knowledge from simulators into a proposal-centric planner, providing diverse supervision beyond single human trajectories and enabling safer multimodal decision making.

Specifically, our framework consists of three components: Driving Video Pretraining, Multimodal Trajectory Distillation, and Momentum-aware Trajectory Selection. In the first module, we curate a large-scale driving video dataset and pretrain a ViT-based vision encoder using V-JEPA [^2] [^3], which learns predictive representation by predicting future latent with effective mode collapse prevention. In the second module, the waypoint-anchored proposals generation leverages deformable attention [^40] [^18] to aggregate BEV features [^30] at trajectory waypoints and refine proposals iteratively. To increase diversity, proposals are supervised using both human trajectories and simulator-generated multimodal trajectories that satisfy safety and comfort constraints, enabling effective knowledge distillation from the simulator. Finally, the selection module assigns scores to all candidates by predicting collision risk, traffic-rule compliance, and comfort, and further incorporates a momentum-aware penalty to reduce frame-to-frame trajectory distortion.

We validate Drive-JEPA on NAVSIM v1 [^12], NAVSIM v2 [^8], and Bench2Drive [^23]. Drive-JEPA achieves 93.3 PDMS on NAVSIM v1 and 87.8 EPDMS on NAVSIM v2, setting a new state of the art. Notably, with only a single front-view camera and a lightweight transformer planner, our V-JEPA-pretrained model outperforms prior work by 3 PDMS in the perception-free setting, highlighting the effectiveness of V-JEPA pretraining for planning. On Bench2Drive, the Multimodal Trajectory Distillation consistently improves driving quality, demonstrating the benefit of diverse supervision for generating safe, multimodal trajectories.

Our contributions can be summarized as follows:

- We introduce V-JEPA pretraining to end-to-end autonomous driving, boosting performance in both perception-based and perception-free settings.
- We propose a novel multimodal trajectories supervision to distill simulator knowledge to a proposal-centric framework, generating diverse multimodal trajectories.
- We design a momentum-aware trajectory selection module, enhancing driving comfort.
- Our method achieves a new state of the art on NAVSIM v1 and NAVSIM v2. In addition, our method achieves strong performance on NAVSIM even without relying on perception annotations.

## 2 Related Work

### 2.1 End-to-end autonomous driving

Early works such as ALVINN [^34] and PilotNet [^6] leverage large-scale human driving data to learn policies that map sensor observations directly to control actions. However, these models often lack interpretability and can degrade due to issues such as causal confusion. To mitigate this, recent studies incorporate intermediate representations and auxiliary supervision to improve robustness. Transfuser [^11] fuses LiDAR and camera features in the BEV space and strengthens BEV features with BEV segmentation and 3D detection supervision. Going further, UniAD [^22] unifies the full stack of driving tasks—including tracking, mapping, and motion prediction—within a single framework jointly optimized with planning. VAD [^25] explores compact vectorized scene representations for efficiency, while SparseDrive [^37] proposes a query-centric sparse structure as a BEV-free alternative. DriveTransformer [^24] further improves efficiency by using a small set of learned queries to aggregate multi-view image features. Moreover, DiffusionDrive [^31] and GoalFlow [^41] investigate diffusion-based end-to-end trajectory generation. Despite these advances, end-to-end driving remains challenging due to two fundamental requirements: capturing the spatiotemporal structure of complex scenes and modeling the inherently multimodal nature of driving behaviors.

### 2.2 World Models for End-to-end driving

Video world models in autonomous driving predict how scenes evolve under ego actions. Recent progress in controllable video generation [^17] [^21] has enabled action-conditioned world models, suggesting their potential as learned simulators. Motivated by the idea that realistic generation implies strong dynamics understanding, several works transfer world-model knowledge to end-to-end driving. VaVIM [^5] trains a causal auto-regressive video model and extends it to generate ego trajectories, while Epona [^45] proposes a hybrid diffusion–auto-regressive predictor with a dual-stream diffusion decoder for joint video and trajectory synthesis. However, both approaches remain computationally heavy due to pixel-level reconstruction. To improve efficiency, latent world models predict future features instead of pixels. LAW [^28] integrates latent dynamics learning into end-to-end driving and achieves strong perception-free performance, and World4Drive [^46] further leverages multimodal video foundation models as richer latent targets. Nonetheless, these latent approaches have not clearly demonstrated benefits from scaling to large-scale video pretraining and may suffer from representation collapse. To overcome this, we adopt V-JEPA [^3] as an efficient latent world model with built-in collapse prevention, enabling scalable video pretraining for end-to-end driving.

### 2.3 MultiModal Trajectories Generation

In planning tasks such as manipulation and autonomous driving, a given scenario often offers multiple action options, requiring effective multimodal modeling. Recently, VADv2 [^10] and Hydra-MDP [^29] introduce a large fixed vocabulary of trajectories by discretizing and clustering the continuous action space. For each scene, this vocabulary provides a diverse multimodal choice space for driving behaviors, and planning is performed by selecting trajectories based on predicted scores. However, this paradigm is fundamentally limited by the coverage of the vocabulary and cannot generalize to out-of-vocabulary scenes. Other works introduce diffusion models to generate multimodal trajectories. DiffusionDrive [^31] guides the diffusion process using a small fixed vocabulary, while GoalFlow [^41] uses goal points as guidance during diffusion-based generation. Compared with computationally expensive diffusion-based methods, proposal-centric approaches such as iPad [^18] iteratively refine a set of trajectory proposals using efficient deformable attention. Compared with Hydra-MDP, iPad’s proposals can be viewed as an online-generated continuous vocabulary; however, its candidates are supervised purely by a single human trajectory per scene, which limits diversity. In contrast, our Multimodal Trajectory Distillation breaks this limitation by distilling knowledge from simulator-generated trajectories.

## 3 Method

### 3.1 Preliminary

#### End-to-end Autonomous Driving

In the task of end-to-end autonomous driving, the objective is to estimate the future trajectory of the ego vehicle in the form of waypoints. Formally, let $I_{t}=\{I_{t}^{1},I_{t}^{2},\ldots,I_{t}^{N}\}$ denote the set of $N$ surrounding multi-view images captured at time step $t$. The model is expected to predict a sequence of waypoints $W_{t}=\{w_{t}^{1},w_{t}^{2},\ldots,w_{t}^{M}\}$, where each waypoint $w_{t}^{i}=(x_{t}^{i},y_{t}^{i},\psi_{t}^{i})$ represents the predicted BEV position and heading angle of the ego vehicle at time step $t+i$. Here, $M$ denotes the number of future positions to be predicted. In addition, the model also takes as input the ego status of the vehicle, which includes the driving command (e.g., left, forward, right), speed, and acceleration.

#### V-JEPA

V-JEPA [^4] learns predictive video representations by estimating the latent representation of a target view $y$ from a masked view $x$, where a subset of spatiotemporal patches are randomly dropped. The method adopts a meta-architecture with an encoder $E_{\theta}(\cdot)$ that extracts video features and a predictor $P_{\phi}(\cdot)$ that predicts the representations at masked locations. The encoder and predictor are optimized jointly with

$$
\min_{\theta,\phi,\Delta_{y}}\;\left\|P_{\phi}(\Delta_{y},E_{\theta}(x))-\mathrm{sg}(E_{\bar{\theta}}(y))\right\|_{1},
$$

where $\Delta_{y}$ is a learnable mask token indicating the dropped patch locations. The target branch uses a stop-gradient operator $\mathrm{sg}(\cdot)$ and an exponential moving average encoder $E_{\bar{\theta}}$ (with parameters $\bar{\theta}$) to stabilize training and avoid representation collapse. The loss is computed only on masked positions. Both $E_{\theta}(\cdot)$ and $P_{\phi}(\cdot)$ are instantiated as Vision Transformers (ViTs) [^14].

![[x1 31.png|Refer to caption]]

Figure 2: Overview of the Drive-JEPA architecture. Driving Video Pretraining learns a ViT encoder from large-scale driving videos using the self-supervised V-JEPA objective. Given the pretrained features, Waypoint-anchored Proposal Generation efficiently produces multiple trajectory proposals, whose distribution is guided by Multimodal Trajectory Distillation. Finally, Momentum-aware Trajectory Selection picks the final trajectory by accounting for cross-frame comfort.

### 3.2 Driving Video Pretraining

To enhance the representation for planning through self-supervised video pretraining, prior works explored pixel-space driving world models and latent world models. While the former faces expensive computation and the latter fails to scale, we propose to leverage V-JEPA in large-scale driving video pretraining.

Table 1: Comparison with previous perception-free planners.

|  | LAW | World4Drive | Epona | Ours |
| --- | --- | --- | --- | --- |
| Encoder size | 21M | 21M | 1.1B | 307M |
| Data scale | ~20h | ~20h | 128h | 208h |
| PDMS | 83.8 | 85.1 | 86.1 | 89.0 |

#### Driving Video Dataset Curation and Scaling

We initialize the ViT encoder with parameters released by V-JEPA 2 [^3]. To bridge the domain gap, we curate a large-scale driving video dataset from three publicly available datasets: CoVLA [^1], DrivingDojo [^39], and OpenScene [^36]. All videos are captured from a front-view camera and processed into 8-frame clips at a resolution of $512\times 256$ and 2 Hz. We adopt the V-JEPA objective to train the ViT encoder in a self-supervised manner on this curated dataset. As shown in Table 1, thanks to the efficiency of the latent prediction task and effective mode-collapse prevention, we successfully scale pretraining to 208 hours with lower computational cost than prior methods.

#### Perception-free End-to-End Autonomous Driving

Following prior world-model-based end-to-end driving works, we adopt a perception-free setting for evaluation, where the model is supervised solely by human trajectories without relying on perception annotations [^28]. Given spatiotemporal features extracted by the ViT encoder from the front-view image, future waypoints are predicted using a transformer decoder with learnable queries. Given the front-view inputs $I_{t}^{1}$ and $I_{t-1}^{1}$, we extract spatiotemporal features using the pretrained ViT encoder and denote them as $\mathbf{F}_{t}\in\mathbb{R}^{N_{f}\times D}$. We introduce $M$ learnable query embeddings $\mathbf{Q}\in\mathbb{R}^{M\times D}$, each corresponding to one future waypoint. The transformer-based decoder [^38] attends to $\mathbf{F}_{t}$ via cross-attention to produce

$$
\mathbf{H}=\mathrm{TransformerDecoder}(\mathbf{Q},\mathbf{F}_{t}),
$$

which is then mapped to predicted waypoints

$$
\hat{W}_{t}=\mathrm{MLP}(\mathbf{H}),
$$

where $\hat{W}_{t}=\{\hat{w}_{t}^{1},\ldots,\hat{w}_{t}^{M}\}$ and each $\hat{w}_{t}^{i}=(\hat{x}_{t}^{i},\hat{y}_{t}^{i},\hat{\psi}_{t}^{i})$ represents the BEV position and heading at time step $t+i$. The network is trained end-to-end using a MSE loss between $\hat{W}_{t}$ and the ground-truth trajectory $W_{t}$. Despite its simplicity, this setup significantly outperforms prior methods (Table 1), highlighting the effectiveness of V-JEPA-based driving video pretraining.

### 3.3 Waypoint-anchored Proposals Generation

Building upon the strong representations from Driving Video Pretraining, we design a planner that follows a proposal-selection paradigm. As mentioned before, a fixed vocabulary can be seen as proposals but suffers from discretization error. Inspired by iPad [^18], we instead generate proposals online.

Given the visual features $\mathbf{F}_{t}\in\mathbb{R}^{N_{f}\times D}$ and ego status at time $t$, we project the ego status by a linear layer into an ego feature $\mathbf{e}_{t}\in\mathbb{R}^{1\times D}$. The proposal queries are initialized as $\mathbf{Q}_{0}\in\mathbb{R}^{N_{p}\times M\times D}$ by adding $\mathbf{e}_{t}$ to learnable positional embeddings, where $N_{p}$ is the number of waypoint-trajectory proposals and $M$ is the number of future waypoints. We iteratively refine the proposal queries $\mathbf{Q}_{\ell}$ for $L$ iterations. At iteration $\ell$, an MLP decodes $\mathbf{Q}_{\ell}$ into waypoint-trajectory proposals $\tilde{W}_{\ell}=\{\tilde{W}_{\ell}^{(n)}\}_{n=1}^{N_{p}}$, with $\tilde{W}_{\ell}\in\mathbb{R}^{N_{p}\times M\times 3}$ and each waypoint $(x,y,\psi)$. Using these explicit waypoint locations as anchors, we refine the queries by exchanging information among proposals and aggregating features from $\mathbf{F}_{t}$ around each predicted waypoint via lift-splat BEV feature sampling [^33]. We then update the queries with a lightweight MLP:

$$
\mathbf{Q}_{\ell+1}=\mathrm{MLP}\!\left(\mathrm{WADA}(\mathbf{Q}_{\ell},\tilde{W}_{\ell},\mathbf{F}_{t})\right),
$$

where WADA denotes Waypoint-anchored Deformable Attention [^40].

Because the final trajectory for planning is selected from $\tilde{W}_{L}$, its distribution is critical. Given a human trajectory $W_{t}$ and the intermediate proposals $\{\tilde{W}_{\ell}\}_{\ell=0}^{L-1}$, a naive way to guide $\tilde{W}_{\ell}$ is using the minimum-over- $N$ loss [^19] with discounted supervision across iterations:

$$
\mathcal{L}_{traj}=\sum_{\ell=0}^{L-1}\lambda^{L-\ell-1}\min_{n\in\{1,\dots,N_{p}\}}\left\lVert W_{t}-\tilde{W}_{\ell}^{(n)}\right\rVert_{2},
$$

where $\lambda=0.1$ down-weights earlier iterations to encourage coarse-to-fine refinement.

However, in autonomous driving, there are often multiple valid choices beyond the single human trajectory for a scene. This naive guidance method limits the multimodality of the proposals. We present our solution in the next section.

### 3.4 Multimodal Trajectories Distillation

To alleviate sparse supervision from a single human trajectory per scene, we distill knowledge from rule-based simulators. HydraMDP [^29] performs hydra-distillation by learning scores over a fixed vocabulary. Instead, we let the simulator provide multimodal trajectory targets to guide the proposal distribution.

Concretely, we start by building a trajectory vocabulary following VADv2 and HydraMDP, but use the vocabulary for a different purpose. We gather all trajectories in the training dataset, which includes more than 100k trajectories. Then we use a clustering method, k-means [^16], to select trajectory centers. We select 8192 centers as the trajectory vocabulary, balancing coverage and computational cost. For each scene in the training dataset, we select high quality multimodal trajectories from the vocabulary using rule-based simulators. Following NAVSIM v2 [^9] [^7], we calculate the EPDM score for all trajectories. We refer to the Appendix A for the detailed definition. Specifically, we first run a PID controller to convert 8 waypoints into a denser 41-point trajectory. Then, at each timestep, we replay other road agents, traffic lights, etc., and compute collisions and other metrics. The rule-based simulator in NAVSIM v2 is designed only for evaluation. We further improve the vectorized computation efficiency to meet large-scale offline scoring needs. After obtaining scores for all trajectories in the vocabulary across all scenes in the training dataset, we select a group of multimodal trajectories $\mathcal{P}_{t}=\{P_{t}^{1},\ldots,P_{t}^{N_{pseudo}}\}$ for each scene by ranking and thresholding. During training, we use these multimodal trajectories as pseudo-teachers to guide the proposals, instead of a single human trajectory. The final $\mathcal{L}_{traj}$ is defined as:

$$
\begin{split}\sum_{\ell=1}^{L}\lambda^{L-\ell}\Big(\min\lVert W_{t}-\tilde{W}_{\ell}^{(n)}\rVert_{2}+\sum_{P\in\mathcal{P}_{t}}\min\lVert P-\tilde{W}_{\ell}^{(n)}\rVert_{2}\Big).\end{split}
$$

Here, the $\min$ operator takes the minimum over the proposal index $n\in\{1,\ldots,N_{p}\}$ at iteration $\ell$.

![[demo_proposals.png|Refer to caption]]

Figure 3: Bird’s eye view of proposals.

As shown in Figure 3, without Multimodal Trajectories Distillation (MTD), the proposals exhibit clear mode collapse; with MTD, they become multimodal.

### 3.5 Momentum-aware Trajectory Selection

To select the best trajectory among the proposal set for planning, we train a neural scorer to evaluate the final proposals $\{\tilde{W}_{L}^{(n)}\}_{n=1}^{N_{p}}$. Concretely, we apply max pooling over the waypoint dimension to the proposal queries $\mathbf{Q}_{L}\in\mathbb{R}^{N_{p}\times M\times D}$ to obtain pooled features $\bar{\mathbf{Q}}_{L}\in\mathbb{R}^{N_{p}\times D}$, which are then fed into a multi-layer perceptron (MLP) to produce proposal scores $S\in\mathbb{R}^{N_{p}\times 1}$. The scorer is trained with a binary cross-entropy (BCE) loss:

$$
\mathcal{L}_{\text{score}}=\mathrm{BCE}(S,\hat{S}),
$$

where $\mathrm{BCE}(x,y)=-y\log x-(1-y)\log(1-x)$. The supervision $\hat{S}$ is derived from simulator-based EPDMS evaluation and is also used to define candidate pseudo-targets $\mathcal{P}_{t}=\{P_{t}^{1},\ldots,P_{t}^{N_{pseudo}}\}$.

While Multimodal Trajectory Distillation improves proposal diversity, it can amplify temporal inconsistency, increasing discomfort due to larger variation across adjacent time frames. To mitigate this, we make the score momentum-aware by incorporating a comfort term. Let $\hat{W}_{t-1}$ denote the selected trajectory at the previous time frame. We compute a distortion-based comfort score $S_{c}\in\mathbb{R}^{N_{p}\times 1}$ by comparing $\hat{W}_{t-1}$ with each current proposal in $\{\tilde{W}_{L}^{(n)}\}_{n=1}^{N_{p}}$, and recalibrate the learned score via

$$
S\leftarrow\frac{7\,S+\,S_{c}}{8}.
$$

where the weights are following NAVSIM v2. Finally, the selected trajectory is formalized as

$$
\hat{W}_{t}=\tilde{W}_{L}^{(n^{*})},\quad n^{*}=\arg\max_{n\in\{1,\ldots,N_{p}\}}S_{n},
$$

where $S_{n}$ denotes the recalibrated score of proposal $\tilde{W}_{L}^{(n)}$.

### 3.6 Losses

In end-to-end driving tasks, it is important to add auxiliary tasks to enhance the model’s environment understanding capability, e.g., BEV map segmentation, 3D object detection, and tracking. However, these traditional dense understanding tasks are computationally intensive. Here we use the lightweight auxiliary tasks instead [^18], which contain rich spatiotemporal signals and are compatible with the proposal-centric design.

We use two auxiliary tasks: proposal-centric mapping and collision prediction. For the first task, our model predicts the on-road and on-route probabilities of the proposed waypoints in $\tilde{W}_{\ell}$, denoted as $R\in\mathbb{R}^{N_{p}\times M\times 2}$. The proposal-centric mapping loss is $\mathcal{L}_{map}=\mathrm{BCE}(R,\hat{R})$. For proposal-centric collision prediction, we estimate the collision probability $A_{v}$ of waypoints in $\tilde{W}_{\ell}$ using log-replay simulation. This task not only requires the model to detect surrounding objects, but also to understand their moving pattern. The proposal-centric collision loss is $\mathcal{L}_{colli}=\lVert A_{v}-\hat{A}_{v}\rVert+0.1\,\mathrm{BCE}(A_{v},\hat{A}_{v})$.

Our Drive-JEPA is end-to-end differentiable. The training loss is defined as:

$$
\mathcal{L}=\mathcal{L}_{traj}+w_{score}\mathcal{L}_{score}+w_{map}\mathcal{L}_{map}+w_{colli}\mathcal{L}_{colli},
$$

where $w_{score}=1$, $w_{map}=2$ and $w_{colli}=1$.

## 4 Experiments

### 4.1 Dataset and Metrics

We evaluate our method on three benchmarks, including NAVSIM v1, NAVSIM v2 and Bench2Drive.

#### NAVSIM v1

NAVSIM is a real-world dataset based on OpenScene [^36] and NuPlan [^7]. It contains 103k and 12k diverse and challenging driving scenarios for model training (Navtrain) and evaluation (Navtest), and introduces simulation-based metrics to better review closed-loop planning capability through open-loop evaluation. During evaluation, the output trajectory is evaluated by a simulator to get rule-based simulation metric scores, including No at-fault Collisions(NC), Drivable Area Compliance(DAC), Time to Collision with bounds(TTC), Ego Progress(EP) and Comfort(C). The final PDM Score(PDMS) is derived by aggregating these metrics:

$$
PDMS=NC\times DAC\times\frac{5\times(EP+TTC)+2\times C}{12}
$$

#### NAVSIM v2

Compared with NAVSIM v1, NAVSIM v2 strengthens driving-quality evaluation by extending PDMS to EPDMS with richer rule-compliance and comfort assessment. It adds Driving Direction Compliance (DDC), Traffic Light Compliance (TLC), and Lane Keeping (LK) to better capture traffic-rule adherence, and replaces the original comfort term with History Comfort (HC) and Extended Comfort (EC) to evaluate both short- and longer-horizon smoothness.

#### Bench2Drive

Bench2Drive [^23] is a closed-loop evaluation benchmark based on CARLA [^15], designed to assess end-to-end autonomous driving systems in interactive urban scenarios. The evaluation includes 220 routes spanning 44 diverse, interactive scenarios. Official metrics include Driving Score (DS), Success Rate (SR), Efficiency and Comfortness, which collectively measure navigation performance, safety, and rule adherence. For detailed metric definitions, see Appendix A.

### 4.2 Implementation Details

In the Driving Video Pretraining stage, we use 8 H800 GPUs and train for 50 epochs, which takes about 3 days. The Drive-JEPA planners are trained on two NVIDIA A30 GPUs for 20 epochs with a total batch size of 64, using the Adam [^26] optimizer with a learning rate of $1\times 10^{-4}$, while the ViT encoder uses a learning rate of $1\times 10^{-5}$. We set $N_{p}=32$ proposals, which is efficient while achieving strong performance in our ablation studies. We use only the front-view camera, resized to $512\times 256$. Despite using fewer input images than prior methods and no LiDAR, we outperform them. See Appendix B for details.

Table 2: Quantitative comparisons on NAVSIM v1. We indicate the best and second best with bold and underlined respectively. The first block shows results for perception-free setting.

| Method | Backbone | Inputs | NC  $\uparrow$ | DAC  $\uparrow$ | EP  $\uparrow$ | C  $\uparrow$ | TTC  $\uparrow$ | \[gray\]0.9PDMS  $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LAW [^28] | resnet34 | C & L | 97.4 | 93.3 | 78.8 | 100 | 91.9 | \[gray\]0.983.8 |
| World4Drive [^46] | resnet34 | C & L | 97.4 | 94.3 | 79.9 | 100 | 92.8 | \[gray\]0.985.1 |
| Epona [^45] | ViT/G | Camera | 97.9 | 95.1 | 80.4 | 99.9 | 93.8 | \[gray\]0.986.2 |
| Ours | ViT/L | Camera | 98.7 | 96.2 | 82.9 | 100 | 95.5 | \[gray\]0.989.0 |
| Transfuser [^11] | ResNet34 | C & L | 97.7 | 92.8 | 79.2 | 100 | 92.8 | \[gray\]0.984.0 |
| HydraMDP [^29] | ResNet34 | C & L | 98.3 | 96.0 | 78.7 | 100 | 94.6 | \[gray\]0.986.5 |
| HydraMDP++ [^27] | ResNet34 | C & L | 97.6 | 96.0 | 80.4 | 100 | 93.1 | \[gray\]0.986.6 |
| DiffusionDrive [^31] | ResNet34 | C & L | 98.2 | 96.2 | 82.2 | 100 | 94.7 | \[gray\]0.988.1 |
| GoalFLow [^41] | ResNet34 | C & L | 98.4 | 98.3 | 85.0 | 100 | 94.6 | \[gray\]0.990.3 |
| DriveDPO [^35] | ResNet34 | C & L | 98.5 | 98.1 | 84.3 | 100 | 94.8 | \[gray\]0.990.0 |
| DriveSuprim [^43] | ResNet34 | Camera | 97.8 | 97.3 | 86.7 | 100 | 93.6 | \[gray\]0.989.9 |
| iPad [^18] | ResNet34 | Camera | 98.4 | 97.9 | 87.4 | 99.9 | 94.9 | \[gray\]0.991.1 |
| Drive-JEPA(Ours) | ResNet34 | Camera | 98.2 | 98.0 | 88.8 | 99.9 | 94.2 | \[gray\]0.991.5 |
| Hydra-MDP [^29] | ViT/L | C & L | 98.4 | 97.7 | 85.0 | 100 | 94.5 | \[gray\]0.989.9 |
| iPad [^18] | ViT/L | Camera | 99.2 | 97.4 | 87.8 | 99.7 | 96.3 | \[gray\]0.991.7 |
| DriveSuprim [^43] | ViT/L | Camera | 98.6 | 98.6 | 91.3 | 100 | 95.5 | \[gray\]0.993.5 |
| Drive-JEPA(Ours) | ViT/L | Camera | 99.1 | 98.2 | 90.8 | 99.9 | 95.9 | \[gray\]0.993.3 |

Table 3: Quantitative comparisons on NAVSIM v2.

| Method | Backbone | NC $\uparrow$ | DAC $\uparrow$ | DDC $\uparrow$ | TL $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | LK $\uparrow$ | HC $\uparrow$ | EC $\uparrow$ | \[gray\]0.9EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Transfuser | ResNet34 | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | \[gray\]0.976.7 |
| HydraMDP++ | ResNet34 | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | \[gray\]0.981.4 |
| DriveSuprim | ResNet34 | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | \[gray\]0.983.1 |
| iPad | ResNet34 | 98.7 | 97.8 | 99.1 | 99.8 | 84.0 | 98.0 | 96.0 | 98.0 | 68.2 | \[gray\]0.984.1 |
| Drive-JEPA(Ours) | ResNet34 | 98.8 | 97.4 | 99.0 | 99.8 | 83.5 | 98.0 | 96.2 | 98.1 | 85.6 | \[gray\]0.985.4 |
| HydraMDP++ | ViT/L | 98.5 | 98.5 | 99.5 | 99.7 | 87.4 | 97.9 | 95.8 | 98.2 | 75.7 | \[gray\]0.985.6 |
| iPad | ViT/L | 98.7 | 98.0 | 98.9 | 99.8 | 86.6 | 98.3 | 97.2 | 98.3 | 74.6 | \[gray\]0.985.8 |
| DriveSuprim | ViT/L | 98.4 | 98.6 | 99.6 | 99.8 | 90.5 | 97.8 | 97.0 | 98.3 | 78.6 | \[gray\]0.987.1 |
| Drive-JEPA(Ours) | ViT/L | 98.4 | 98.6 | 99.1 | 99.8 | 88.4 | 97.8 | 97.6 | 97.9 | 84.8 | \[gray\]0.987.8 |

### 4.3 Main Results

Results on NAVSIM v1 As shown in Table 2, compared with previous methods, Drive-JEPA achieves the best PDMS with ResNet34. When using ViT/L, Drive-JEPA is only second to DriveSuprim [^43], which uses advanced data augmentations. Notably, while maintaining high safe metrics, such as NC, DC and TTC, our method achieves the best Ego Progress, resulting in an assertive driving style.

Perception-free End-to-end Autonomous Driving We also evaluated our method in a perception-free setting, where we use a simple decoder with the pretrained ViT encoder as described in 3.2. As shown in Table 2, our method surpasses previous methods by a large margin, regardless of backbone size. The PDMS is even close to SOTA methods that rely on perception annotations, demonstrating the strength of V-JEPA pretraining.

Results on NAVSIM v2 NAVSIM v2 has more sophisticated metrics than NAVSIM v1. Our method still outperformances all prior methods. While prior methods struggle with EC, our method performs quite well while achieving good results on safety metrics, traffic rule compliance and Ego Progress.

Table 4: Quantitative comparisons on Bench2Drive.

| Method | Effi. $\uparrow$ | Comf. $\uparrow$ | SR  $\uparrow$ | DS  $\uparrow$ |
| --- | --- | --- | --- | --- |
| AD-MLP | 48.45 | 22.63 | 0.00 | 18.05 |
| UniAD | 129.21 | 43.58 | 16.36 | 45.81 |
| VAD | 157.94 | 46.01 | 15.00 | 42.35 |
| TCP | 76.54 | 18.08 | 30.00 | 59.90 |
| DriveDPO | 166.80 | 26.79 | 30.62 | 62.02 |
| iPad | 153.83 | 35.51 | 33.18 | 60.52 |
| DriveTransformer | 100.64 | 20.78 | 35.01 | 63.46 |
| Ours | 157.85 | 30.24 | 36.82 | 64.52 |

Results on Bench2Drive Bench2Drive evaluates autonomous agents in close-loop simulation. Our method achieves the best Driving Score with very competitive Efficiency. Our method surpasses another proposal-centric method iPad by 4 in Driving Score, identifying the effectiveness of Multimodal Trajectories Distillation.

Table 5: Ablation study of the proposed modules on NAVSIM v2.

| $\mathcal{M}_{1}$ | $\mathcal{M}_{2}$ | $\mathcal{M}_{3}$ | $\mathcal{M}_{4}$ | NC $\uparrow$ | DAC $\uparrow$ | DDC $\uparrow$ | TL $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | LK $\uparrow$ | HC $\uparrow$ | EC $\uparrow$ | $\mathcal{D}$ $\uparrow$ | \[gray\]0.9EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ✗ | ✗ | ✗ | ✗ | 98.7 | 97.8 | 99.1 | 99.8 | 84.0 | 98.0 | 96.0 | 98.0 | 68.2 | 25% | \[gray\]0.984.1 |
| ✓ | ✗ | ✗ | ✗ | 98.7 | 98.0 | 98.9 | 99.8 | 86.6 | 98.3 | 97.2 | 98.3 | 74.6 | 21% | \[gray\]0.9 $85.8_{\scriptsize{\color[rgb]{0,0.5,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.5,0}+1.7}}$ |
| ✗ | ✓ | ✗ | ✗ | 98.3 | 98.1 | 99.1 | 99.9 | 89.1 | 97.7 | 97.7 | 98.1 | 69.7 | 24% | \[gray\]0.9 $86.1_{\scriptsize{\color[rgb]{0,0.5,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.5,0}+2.0}}$ |
| ✗ | ✓ | ✓ | ✗ | 98.5 | 98.6 | 99.1 | 99.8 | 89.1 | 97.9 | 97.6 | 97.8 | 47.9 | 40% | \[gray\]0.9 $84.5_{\scriptsize{\color[rgb]{0,0.5,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.5,0}+0.4}}$ |
| ✗ | ✓ | ✓ | ✓ | 98.4 | 98.6 | 99.1 | 99.8 | 88.4 | 97.8 | 97.6 | 97.9 | 84.8 | 40% | \[gray\]0.9 $87.8_{\scriptsize{\color[rgb]{0,0.5,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.5,0}+3.7}}$ |

![[x2 29.png|Refer to caption]]

Figure 4: Qualitative comparison of trajectories by different models in front-facing camera and bird’s eye view on different driving scenarios. Trajectories are shown for: Human Trajectory, Drive-JEPA, iPad, Transfuser.

### 4.4 Ablation Studies

Ablation studies on proposed modules We first conducted ablation studies on the proposed modules: $\mathcal{M}_{1}$: V-JEPA 2 checkpoints, $\mathcal{M}_{2}$: Driving Video Pretraining, $\mathcal{M}_{3}$: Multimodal Trajectories Distillation, and $\mathcal{M}_{4}$: Momentum-aware Trajectory Selection. As shown in Table 5, replacing ResNet34 with the ViT released by V-JEPA 2 ($\mathcal{M}_{1}$) improves EPDMS. Our Driving Video Pretraining further boosts performance by reducing the domain gap ($\mathcal{M}_{2}$). After adding $\mathcal{M}_{3}$, the framework achieves better $\mathcal{D}$ (Diversity) [^31] and overall metrics, which is also supported by the validation score curve in Figure 5. However, the increased diversity results in worse EC. Finally, adding $\mathcal{M}_{4}$ not only largely boosts EC to 84.8, but also sets a new best record on EPDMS.

Ablation study on the number of Pseudo Teacher Trajectories. As shown in Table 6, we tried $N_{pseudo}=0,1,2,4,8$ pseudo-teacher trajectories. While the correlation between $N_{pseudo}$ and EPDMS is not very strong, using pseudo-teacher trajectories consistently performs better than using none ($N_{pseudo}=0$).

Table 6: Ablation on number of pseudo-teacher trajectories.

| $N_{pseudo}$ | 0 | 1 | 2 | 4 | 8 |
| --- | --- | --- | --- | --- | --- |
| EPDMS | 87.2 | 87.8 | 87.7 | 87.8 | 87.5 |

Table 7: Comparison with mainstream vision pretraining methods.

| Vision Encoder | Size | \[gray\]0.9PDMS  $\uparrow$ |
| --- | --- | --- |
| Epona [^45] | ViT/G | \[gray\]0.986.2 |
| ImageNet [^13] | ResNet34 | \[gray\]0.976.0 |
| DepthAnything [^42] | ViT/L | \[gray\]0.9- |
| MAE [^20] | ViT/L | \[gray\]0.9- |
| Dinov2 [^32] | ViT/L | \[gray\]0.976.1 |
| Sigclip [^44] | ViT/L | \[gray\]0.983.4 |
| V-JEPA 2 [^3] | ViT/L | \[gray\]0.986.1 |
| Ours | ViT/L | \[gray\]0.989.0 |

Ablation on Driving Video Pretraining. We use the same simple decoder with encoders pretrained by mainstream pretraining methods. As shown in Table 7, V-JEPA 2 performs the best among them. MAE and DepthAnything could not converge. This highlights the strength of the V-JEPA objective for video pretraining. In this work, we curated a large driving video dataset. The ViT/L encoder trained on this dataset with the V-JEPA objective further boosts performance, surpassing the SOTA Epona by 3 PDMS.

![[val_score_epoch_groups.png|Refer to caption]]

Figure 5: Multimodal Trajectory Distillation improves PDM score.

## 5 Conclusion

We proposed Drive-JEPA, an end-to-end driving framework that combines V-JEPA video pretraining with multimodal trajectory distillation to mitigate imitation-learning modal collapse. Pretraining a ViT encoder on large-scale driving videos yields strong planning representations, enabling a simple decoder to achieve competitive perception-free performance. Distilling simulator-guided pseudo-teacher trajectories improves proposal diversity, and momentum-aware selection further enhances temporal stability and comfort. Drive-JEPA achieves state-of-the-art results on NAVSIM v1/v2 and improves closed-loop performance on Bench2Drive.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

## Appendix A Metrics

### A.1 Extended Predictive Driver Model Score (EPDMS)

NAVSIM v2 [^9] extends the PDMS metric from NAVSIM v1 to EPDMS, which is defined as:

$$
EPDMS=NC\times DAC\times DDC\times TLC\times\frac{5\times(EP+TTC)+2\times(LK+HC+EC)}{16}
$$

The subscores include: No at-fault Collision (NC), Drivable Area Compliance (DAC), Driving Direction Compliance (DDC), Traffic Light Compliance (TLC), Ego Progress (EP), Time to Collision (TTC), Lane Keeping (LK), History Comfort (HC), and Extended Comfort (EC).

Among these, DDC, TLC, LK, HC, and EC are newly introduced in NAVSIM v2. We summarize them below; for full computation details, we refer readers to NAVSIM v2 [^9].

#### Driving Direction Compliance (DDC).

The ego vehicle must follow the legal direction of travel within lanes and avoid driving in oncoming lanes outside intersections.

#### Traffic Light Compliance (TLC).

This score evaluates whether the ego vehicle obeys traffic-light phases and enters intersections only under a valid green signal.

#### Lane Keeping (LK).

This score measures whether the ego vehicle stays near the centerline of the current lane and avoids lingering between adjacent lanes, while discouraging hesitant “half-commit” lane-change probes.

#### History Comfort (HC).

To better assess ride comfort, we prepend the predicted trajectory with a short segment of the human driver’s recent motion using a fixed padding length of 1.5 seconds. The resulting continuous trajectory is then evaluated using the same comfort metric adopted in the nuPlan framework [^7].

#### Extended Comfort (EC).

This score checks that the predicted motion remains smooth across consecutive time steps.

### A.2 Diversity (𝒟\\mathcal{D})

We report this metric in Table 5. Following DiffusionDrive [^31], it is defined as:

$$
\mathcal{D}=1-\frac{1}{N_{p}}\sum_{i=1}^{N_{p}}\frac{\mathrm{Area}\!\left(\tilde{W}_{t_{i}}\,\cap\,\bigcup_{j=1}^{N_{p}}\tilde{W}_{t_{j}}\right)}{\mathrm{Area}\!\left(\tilde{W}_{t_{i}}\,\cup\,\bigcup_{j=1}^{N_{p}}\tilde{W}_{t_{j}}\right)}\,.
$$

To compute the IoU between trajectories, we rasterize each trajectory polyline into a 2D occupancy mask by buffering the polyline with a 2-meter width (i.e., a 2 m-thick corridor) and projecting it onto a grid.

### A.3 Metrics used in Bench2Drive

We briefly describe the four metrics used in Bench2Drive; for full computation details, we refer readers to Bench2Drive [^23].

#### Success Rate (SR).

The proportion of routes completed successfully within the allotted time and without traffic violations.

#### Driving Score (DS).

This score follows the official CARLA [^15] metric, combining route completion with penalties for infractions.

#### Efficiency.

CARLA includes a check for excessively low speed by comparing the ego vehicle’s speed with nearby traffic.

#### Comfort.

This metric follows nuPlan’s [^7] smoothness (comfort) protocol, which evaluates longitudinal acceleration (min/max), the maximum absolute lateral acceleration, yaw rate, yaw acceleration, longitudinal jerk, and the maximum magnitude of the jerk vector.

## Appendix B More details

#### Threshold.

In Section 3.4, we use a threshold on simulated EPDMS to select trajectories from the vocabulary. The threshold is set to 0.95. In many scenes, more than $N_{\text{pseudo}}$ trajectories exceed this threshold; during training, we uniformly sample $N_{\text{pseudo}}$ trajectories at random from this high-quality subset.

Table 8: Input image resolution.

| Method | Transfuser | HydraMDP++ | DriveSuprim | GoalFlow | iPad | Ours |
| --- | --- | --- | --- | --- | --- | --- |
| Input image resolution | 1024 $\times$ 256 | 1024 $\times$ 256 | 1024 $\times$ 256 | 1024 $\times$ 256 | 4 $\times$ 768 $\times$ 432 | 2 $\times$ 512 $\times$ 256 |

#### Resolution.

As shown in Table 8, HydraMDP++, DriveSuprim, and GoalFlow follow Transfuser in using an input resolution of $1024\times 256$, formed by stacking the front, left, and right camera images. iPad uses a higher resolution ($768\times 432$) and four camera views (front, left, right, and back). Our setting uses only the front camera at $512\times 256$. We include both $I_{t}$ and $I_{t-1}$, resulting in an input tensor of $2\times 512\times 256$.

## Appendix C More visualization

![[more_mtd_demo.png|Refer to caption]]

Figure 6: Bird’s eye view of proposals. Without Mutimodal Trajectory Distillation (MTD), the proposals collapse into one mode. With MTD, the proposals show multimodal distribution.

[^1]: Covla: comprehensive vision-language-action dataset for autonomous driving. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 1933–1943. Cited by: §3.2.

[^2]: Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15619–15629. Cited by: §1, §1.

[^3]: V-jepa 2: self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985. Cited by: §1, §1, §2.2, §3.2, Table 7.

[^4]: Revisiting feature prediction for learning visual representations from video. arXiv preprint arXiv:2404.08471. Cited by: §3.1.

[^5]: Vavim and vavam: autonomous driving through video generative modeling. arXiv preprint arXiv:2502.15672. Cited by: §1, §2.2.

[^6]: End to end learning for self-driving cars. arXiv preprint arXiv:1604.07316. Cited by: §2.1.

[^7]: Nuplan: a closed-loop ml-based planning benchmark for autonomous vehicles. arXiv preprint arXiv:2106.11810. Cited by: §A.1, §A.3, §3.4, §4.1.

[^8]: Pseudo-simulation for autonomous driving. In Conference on Robot Learning (CoRL), Cited by: §1.

[^9]: Pseudo-simulation for autonomous driving. arXiv preprint arXiv:2506.04218. Cited by: §A.1, §A.1, §3.4.

[^10]: Vadv2: end-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243. Cited by: §1, §2.3.

[^11]: Transfuser: imitation with transformer-based sensor fusion for autonomous driving. IEEE transactions on pattern analysis and machine intelligence 45 (11), pp. 12878–12895. Cited by: §1, §2.1, Table 2.

[^12]: Navsim: data-driven non-reactive autonomous vehicle simulation and benchmarking. Advances in Neural Information Processing Systems 37, pp. 28706–28719. Cited by: §1.

[^13]: Imagenet: a large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Cited by: Table 7.

[^14]: An image is worth 16x16 words: transformers for image recognition at scale. In International Conference on Learning Representations, Cited by: §3.1.

[^15]: CARLA: an open urban driving simulator. In Conference on robot learning, pp. 1–16. Cited by: §A.3, §4.1.

[^16]: The faiss library. External Links: 2401.08281 Cited by: §3.4.

[^17]: Vista: a generalizable driving world model with high fidelity and versatile controllability. Advances in Neural Information Processing Systems 37, pp. 91560–91596. Cited by: §2.2.

[^18]: IPad: iterative proposal-centric end-to-end autonomous driving. arXiv preprint arXiv:2505.15111. Cited by: §1, §2.3, §3.3, §3.6, Table 2, Table 2.

[^19]: Social gan: socially acceptable trajectories with generative adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2255–2264. Cited by: §3.3.

[^20]: Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 16000–16009. Cited by: Table 7.

[^21]: Gaia-1: a generative world model for autonomous driving. arXiv preprint arXiv:2309.17080. Cited by: §2.2.

[^22]: Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 17853–17862. Cited by: §1, §2.1.

[^23]: Bench2Drive: towards multi-ability benchmarking of closed-loop end-to-end autonomous driving. In NeurIPS 2024 Datasets and Benchmarks Track, Cited by: §A.3, §1, §4.1.

[^24]: X. Jia, J. You, Z. Zhang, and J. Yan DriveTransformer: unified transformer for scalable end-to-end autonomous driving. In The Thirteenth International Conference on Learning Representations, Cited by: §2.1.

[^25]: Vad: vectorized scene representation for efficient autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 8340–8350. Cited by: §2.1.

[^26]: Adam: a method for stochastic optimization. arXiv preprint arXiv:1412.6980. Cited by: §4.2.

[^27]: Hydra-mdp++: advancing end-to-end driving via expert-guided hydra-distillation. arXiv preprint arXiv:2503.12820. Cited by: Table 2.

[^28]: Y. Li, L. Fan, J. He, Y. Wang, Y. Chen, Z. Zhang, and T. Tan Enhancing end-to-end autonomous driving with latent world model. In The Thirteenth International Conference on Learning Representations, Cited by: §1, §2.2, §3.2, Table 2.

[^29]: Hydra-mdp: end-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978. Cited by: §1, §2.3, §3.4, Table 2, Table 2.

[^30]: Bevformer: learning bird’s-eye-view representation from lidar-camera via spatiotemporal transformers. IEEE Transactions on Pattern Analysis and Machine Intelligence. Cited by: §1.

[^31]: Diffusiondrive: truncated diffusion model for end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 12037–12047. Cited by: §A.2, §1, §2.1, §2.3, §4.4, Table 2.

[^32]: Dinov2: learning robust visual features without supervision. arXiv preprint arXiv:2304.07193. Cited by: Table 7.

[^33]: Lift, splat, shoot: encoding images from arbitrary camera rigs by implicitly unprojecting to 3d. In European conference on computer vision, pp. 194–210. Cited by: §3.3.

[^34]: Alvinn: an autonomous land vehicle in a neural network. Advances in neural information processing systems 1. Cited by: §1, §2.1.

[^35]: Drivedpo: policy learning via safety dpo for end-to-end autonomous driving. arXiv preprint arXiv:2509.17940. Cited by: Table 2.

[^36]: Scene as occupancy. External Links: 2306.02851 Cited by: §3.2, §4.1.

[^37]: Sparsedrive: end-to-end autonomous driving via sparse scene representation. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 8795–8801. Cited by: §2.1.

[^38]: Attention is all you need. Advances in neural information processing systems 30. Cited by: §3.2.

[^39]: Drivingdojo dataset: advancing interactive and knowledge-enriched driving world model. Advances in Neural Information Processing Systems 37, pp. 13020–13034. Cited by: §3.2.

[^40]: Vision transformer with deformable attention. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4794–4803. Cited by: §1, §3.3.

[^41]: Goalflow: goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 1602–1611. Cited by: §1, §2.1, §2.3, Table 2.

[^42]: Depth anything: unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10371–10381. Cited by: Table 7.

[^43]: DriveSuprim: towards precise trajectory selection for end-to-end planning. arXiv preprint arXiv:2506.06659. Cited by: §4.3, Table 2, Table 2.

[^44]: Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 11975–11986. Cited by: Table 7.

[^45]: Epona: autoregressive diffusion world model for autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), Cited by: §1, §2.2, Table 2, Table 7.

[^46]: World4drive: end-to-end autonomous driving via intention-aware physical latent world model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 28632–28642. Cited by: §1, §2.2, Table 2.