---
title: "AD-E2E-JEPA: A Joint-Embedding Predictive Architecture For End-to-End Autonomous Driving"
source: "https://arxiv.org/html/2609.34085v1"
author:
published:
created: 2026-09-30
description:
tags:
  - "clippings"
---
###### Abstract

Autonomous driving requires world models that can understand the physical world, reason and plan, and operate safely. In this paper, we first systematically evaluate existing action-conditioned joint-embedding predictive architecture (JEPA) world models, including LeWM, DINO-WM, and JEPA-WM for end-to-end autonomous driving (E2EAD). To isolate world-model quality from policy learning, we employ a goal-conditioned zero-shot planning setting that evaluates these models using ground-truth future observations as goals, without training any driving policy. We find that existing JEPA-based world models are either accurate for driving but computationally expensive, or computationally efficient but insufficient for planning. To address this trade-off, we propose AD-E2E-JEPA, which introduces a SIGReg-regularized learnable projector applied to projected patch embeddings. The projector reduces the number of planning patches by $16\times$ and the embedding dimension by $4\times$, achieving a $100\times$ inference speedup while retaining planning performance, with a 0.8-second runtime for an 8-frame rollout over 256 candidate trajectories. Without training any driving policy, the world model itself reaches the goals located 20 meters away on average within the displacement of respectively 4.0/2.8 meters, using world-model rollouts over trajectory vocabularies of respectively 256/8,192 candidates. On the NAVSIMv2 benchmark, it achieves 67.3/72.9 EPDMS with multiplicative safety metrics and 84.1/86.5 EPDMS <sup>†</sup> without them in goal-conditioned zero-shot planning. Experiments further show that the self-supervised pretrained projector improves downstream imitation learning performance from 80.2 to 85.4 EPDMS. The source code is available at [https://github.com/HaoranZhuExplorer/AD-E2E-JEPA](https://github.com/HaoranZhuExplorer/AD-E2E-JEPA).

![[overview.png|Refer to caption]]

Figure 1: AD-E2E-JEPA is an action-conditioned, JEPA-based world model for end-to-end autonomous driving. Without training any driving policy, it enables driving via goal-conditioned zero-shot planning, where future sensor observations are specified as the goal and a trajectory is selected from the trajectory vocabulary solely via world-model rollouts. AD-E2E-JEPA demonstrates strong zero-shot driving performance, efficient planning, low displacement errors, high trajectory hit rates, and representations that transfer effectively to downstream imitation learning. Zero-shot planning metrics are evaluated on 100 subsampled test scenes for a fair comparison with the time-consuming DINO-WM/JEPA-WM. See Table 2 for AD-E2E-JEPA’s results on the full set of 12,146 test scenes.

## 1 Introduction

Among intelligent systems operating in the physical world, autonomous vehicles, traditionally operated by humans, lie in the core interest of governments and the industry (Waymo, Tesla, NVIDIA, Wayve, XPENG, Uber, etc.). Efficient, robust, and safe autonomy defines the future of transportation leading to improved mobility of the population, reduced commute burdens, and increased road use. End-to-end learning [^42] [^6] [^22] has emerged as a popular paradigm for autonomous driving. Instead of relying on modular pipelines, an end-to-end autonomous driving (E2EAD) system [^9] takes raw sensor observations as input and directly predicts a driving trajectory. However, existing E2EAD methods still rely heavily on imitation learning [^22] [^35], where a reactive policy is trained to imitate human driving trajectories based on observed sensory context, without explicitly modeling the underlying dynamics of the driving environment. Moreover, human driving demonstrations can be suboptimal and noisy, which may limit the quality of supervision provided by imitation learning.

A promising direction beyond purely reactive imitation is to equip autonomous driving agents with a world model [^30] based on a joint-embedding predictive architecture (JEPA) that captures the underlying dynamics of the environment. Through self-supervised prediction in a latent embedding space, a driving agent can learn representations of the physical world by anticipating the future consequences of candidate driving trajectories or inferring environmental states that are not directly observable from sensor inputs. Such predictive objectives can yield representations useful for downstream tasks. More importantly, an action-conditioned world model can be rolled out to predict possible future states under different candidate actions, enabling the agent to evaluate alternative trajectories and select actions that best achieve a desired goal. This provides a principled foundation for planning and decision-making beyond direct imitation of human demonstrations.

In this paper, we present AD-E2E-JEPA (Autonomous Driving with an End-to-End Joint-Embedding Predictive Architecture), an action-conditioned JEPA-based world model for autonomous driving. Unlike imitation-learning-based reactive systems, AD-E2E-JEPA uses human driving trajectories only as action conditioning for world-model learning, rather than as supervision for training a driving policy. Without training an explicit driving policy, the learned world model enables goal-conditioned zero-shot planning by selecting among candidate driving trajectories to navigate toward a goal specified by an image. Compared with existing JEPA world models, AD-E2E-JEPA introduces a learnable projector with SIGReg regularization, which substantially reduces planning latency while preserving planning performance. Moreover, the self-supervised pretrained projector transfers effectively to downstream imitation-learning-based E2E driving.

Our contributions can be summarized as follows:

- We are the first to adapt JEPA to E2EAD for goal-conditioned zero-shot planning. Beyond the success rate commonly used for JEPA-based world models, we also report driving performance metrics such as planning efficiency, geodesic accuracy, and reliability.
- We show that existing JEPA-based world models face a substantial trade-off between efficiency and performance for planning: they are either computationally efficient but inaccurate at planning, or accurate but require over a minute of planning per scene.
- To address this trade-off, we propose AD-E2E-JEPA, which builds on [^52] by reducing embedding dimensionality and further reducing the number of patch embeddings while leveraging SIGReg regularization [^3] to preserve planning performance. This design substantially improves planning efficiency and may be applied to other JEPA-based world-model planning domains beyond E2EAD.
- For goal-conditioned zero-shot planning on the NAVSIM test set, AD-E2E-JEPA significantly improves planning performance over LeWM, a gain of 23.7 EPDMS averaged over all 12,146 testing scenes, while requiring $100\times$ less planning time over DINO-WM/JEPA-WM. Our best variant achieves a 72.9 EPDMS, 2.8-meter average displacement error, and a 2.0-degree mean absolute heading error when selecting from 8192 candidate trajectories.
- We further show that the self-supervised pretrained projector transfers effectively to downstream imitation learning. Integrated into a simple ViT-based E2E architecture, it improves EPDMS from 80.2 to 85.4 compared with a random projector, demonstrating that world-model representations can also benefit imitation-learning-based E2E driving.

## 2 Related Work

### 2.1 World Models

World models are internal representations that enable an agent to predict what is likely, plausible, or impossible, thereby providing a foundation for what is often referred to as common sense [^30]. The idea of world models can be traced back to a long history of planning and control [^7] [^46]. Early world models performed predictions directly in pixel space for robotic planning [^17] or learned latent dynamics using pixel reconstruction objectives [^20]. More recent work suggests that learning world models purely in latent space, without reconstructing pixels, can lead to improved planning performance [^59]. Generative world models have also recently received considerable attention. For example, current generative world models employ diffusion transformers [^40] to generate high-fidelity videos. However, despite their visual realism, whether such generative models reliably capture physical laws and can consequently benefit planning remains unclear [^25].

Joint-Embedding Predictive Architecture (JEPA) [^30] enables world models to be learned directly in latent space by predicting the representation of target data $y$ from context data $x$, with regularization-based methods [^5] [^3] [^28] [^54] maximizing information to avoid representation collapse, in which all representations converge to a constant vector and become useless. Recent studies suggest that intuitive physics can emerge from JEPA [^18]. JEPA-based approaches have been explored across multiple modalities, including images [^1], videos [^4] [^2] [^37], LiDAR [^61], and audio [^16] [^53]. These latent representations have further been shown to support zero-shot planning [^59] [^44] [^36] with action-conditioned world models, in which the model predicts future latents conditioned on actions. Recent work has investigated additional structural properties of the latent space that can facilitate planning. Temporal straightening [^52], for example, regularizes latent trajectories toward straighter paths, and prior work shows that projecting embeddings into a lower-dimensional space can improve planning performance. Other recent studies have demonstrated the potential of hierarchical planning [^57], the benefits of sparse representations for planning [^27], and the benefits of adaptively updating world models during planning [^51].

### 2.2 End-to-End Autonomous Driving

End-to-end autonomous driving (E2EAD) directly maps raw sensor inputs to planning actions using a fully differentiable model [^9], reducing error accumulation across modules and enabling joint optimization for planning [^22]. Most existing methods rely heavily on imitation learning to mimic human driving policies [^12] [^22] [^24] [^10]. More recently, scoring-based methods rank candidate trajectories using annotated driving scores [^34] [^31] [^26], but still need imitation learning and dense trajectory-level supervision. World models have also gained attention in E2EAD, mainly for aligning predicted and ground-truth future representations [^33] [^32] or generating future driving videos [^56] [^21].

JEPA has recently been explored for autonomous driving. AD-L-JEPA [^61] firstly introduces JEPA pre-training for LiDAR perception, then Drive-JEPA [^49] [^38] applies JEPA-based representations to end-to-end driving. AD-LiST-JEPA [^60] extends JEPA to temporal LiDAR world modeling, while Auto-JEPA [^55] and DA-WAM [^58] predict future representations for scoring-based planning. WA-JEPA [^50] combines representation learning, future prediction, and imitation learning. In contrast, our work directly investigates the zero-shot planning capability of JEPA without task-specific imitation learning.

## 3 Method

We propose AD-E2E-JEPA, illustrated in Figure 2, a joint-embedding predictive architecture (JEPA) for end-to-end autonomous driving (E2EAD). It enables efficient zero-shot goal-conditioned planning, while its self-supervised pretrained projector also benefits downstream imitation learning.

![[architecture 1.png|Refer to caption]]

Figure 2: AD-E2E-JEPA architecture: (a) A self-supervised, action-conditioned JEPA world model for E2EAD. The DINOv3 encoder is frozen, while (a.1) a learnable projector is shared by the context-history and target-future branches, compressing the representation by 16 × 16\\times and its dimensionality by 4 4\\times, with SIGReg maximizing information while avoiding collapse and improving planning performance. This enables a 100 100\\times speedup in zero-shot goal-conditioned planning in (b), where trajectories are selected via world-model rollouts over a driving vocabulary. (c) The projector pretrained through self-supervision also improves downstream imitation learning.

### 3.1 Preliminary: Task formulation of E2EAD

For simplicity, we consider a front-camera-only setting. At each time step $t$, the driving agent observes a context window of historical image-pose pairs and predicts a sequence of future ego poses that form a driving trajectory:

$$
\displaystyle(\mathbf{I}_{t-K:t},\,\mathbf{P}_{t-K:t})
$$
 
$$
\displaystyle\hat{\mathbf{P}}_{t+1:t+F}=f_{\phi}(\mathbf{I}_{t-K:t},\,\mathbf{P}_{t-K:t})
$$

Here, $\mathbf{I}_{j}\in\mathbb{R}^{3\times H\times W}$ denotes the front-camera image at time step $j$, and $\mathbf{P}_{j}=[x_{j},y_{j},\theta_{j}]^{\top}\in\mathbb{R}^{3}$ denotes the corresponding ego pose, consisting of the planar position $(x_{j},y_{j})$ and heading angle $\theta_{j}$ in radians. All poses are expressed relative to the current ego pose at time step $t$, such that $\mathbf{P}_{t}=[0,0,0]^{\top}$. The objective of E2EAD is to predict future ego poses that safely make progress.

### 3.2 World Model Architecture

#### 3.2.1 JEPA-WM Adaptation for E2EAD

We initialize our world model baseline using the best-performing configuration identified in JEPA-WM [^47], which conducts extensive ablations on the design choices of JEPA-based world models and finds that the optimal configuration uses DINOv3 [^43] ViT-L as the backbone encoder, together with an AdaLN-style [^41] [^40] predictor equipped with RoPE [^45], optionally with rollout training. DINOv3 is preferred over DINOv2 [^39] and V-JEPA 2 [^2], potentially due to its dense semantic representations, which are also well suited to autonomous driving in our setting.

To adapt the model to E2EAD, we define the action at each time step as the relative pose change between consecutive frames, following [^56], as shown in Eq. (3):

$$
\mathbf{a}_{t}=\operatorname{Relative}(\mathbf{P}_{t},\mathbf{P}_{t+1})=\left[\Delta x_{t\rightarrow t+1},\Delta y_{t\rightarrow t+1},\Delta\theta_{t\rightarrow t+1}\right]^{\top}.
$$

The frozen DINOv3 backbone independently encodes the history frames and the subsequent frame, as shown in Eq. (4). Given the history embeddings and action embeddings produced by a linear layer $E_{a}$, the AdaLN predictor predicts the embedding sequence one time step ahead, as shown in Eq. (5). We supervise the predictions using only an MSE loss, as defined in Eq. (6), without an additional anti-collapse objective, since the prediction targets are from the frozen DINOv3 encoder:

$$
\displaystyle s_{t-K:t+1}
$$
 
$$
\displaystyle=\operatorname{Enc}(\mathbf{I}_{t-K:t+1}),
$$
$$
\displaystyle\hat{s}_{t-K+1:t+1}
$$
 
$$
\displaystyle=\operatorname{Pred}\!\left(s_{t-K:t},E_{a}(\mathbf{a}_{t-K:t})\right),
$$
$$
\displaystyle\mathcal{L}_{\mathrm{pred}}
$$
 
$$
\displaystyle=\operatorname{MSE}\!\left(\hat{s}_{t-K+1:t+1},s_{t-K+1:t+1}\right).
$$

#### 3.2.2 AD-E2E-JEPA

We propose AD-E2E-JEPA for efficient and reliable planning. Existing JEPA-based world model baselines operate on dense patch embeddings produced by the encoder, making planning computationally expensive and potentially requiring several minutes [^2]. This is impractical for the E2EAD setting and motivates embedding compression. Experiments in [^52] show that adding a projector with two convolutional layers with stride $1\times 1$ can reduce embedding dimensionality while improving planning performance. Inspired by this, AD-E2E-JEPA further introduces a learnable projector $\operatorname{Proj}(\cdot)$ consisting of two convolutional layers with stride $2\times 2$. The projector is applied independently to the encoder embeddings of the history and future frames, as shown in Eq. (7), reducing their spatial size by $16\times$ while also reducing the ViT-L embedding dimensionality by $4\times$, from $D_{\text{enc}}=1024$ to $D_{\text{proj}}=256$.

$$
z_{t-K:t+1}=\operatorname{Proj}(s_{t-K:t+1})
$$

The predictor operates on the projected embeddings, as shown in Eq. (8).

$$
\hat{z}_{t-K+1:t+1}=\operatorname{Pred}^{\prime}\!\left(z_{t-K:t},E_{a}(\mathbf{a}_{t-K:t})\right)
$$

The prediction loss is then defined over the projected embeddings in Eq. (9). However, in this case, the projected embeddings may suffer from representation collapse, where distinct DINOv3 embeddings are mapped to nearly constant vectors and thus become uninformative. We apply stop gradient [^11] [^19] to the projected target embedding in the MSE loss.

$$
\mathcal{L}^{\mathrm{proj}}_{\mathrm{pred}}=\operatorname{MSE}\!\left(\hat{z}_{t-K+1:t+1},\operatorname{sg}\left(z_{t-K+1:t+1}\right)\right)
$$

Furthermore, we apply SIGReg [^3] to encourage the embeddings to follow an isotropic Gaussian distribution and prevent representation collapse. Unlike the SIGReg formulation in [^36], which is applied to global CLS embeddings across the batch independently at each time step and then averaged over time, we apply SIGReg to patch embeddings independently at each patch location and time step across the batch, and average the resulting regularization over all time steps and patch locations, as shown in Eq. (10):

$$
\mathcal{L}^{t-K:t+1}_{\mathrm{SIGReg}}=\frac{1}{NH^{\prime}W^{\prime}M}\sum_{l=1}^{NH^{\prime}W^{\prime}}\sum_{m=1}^{M}T\left(\left\{\left\langle z_{l,b},\bm{u}^{(m)}\right\rangle\right\}_{b=1}^{B}\right).
$$

For the single-step setting in Eq. (10), $l$ indexes the projected patch locations across the temporal window of length $N=K+2$ and spatial dimensions $H^{\prime}\times W^{\prime}$. $T(\cdot)$ denotes the univariate Epps–Pulley test [^15], which regularizes the $D$ -dimensional embeddings across the batch of size $B$ using $M$ random projection directions $\bm{u}^{(m)}\in\mathbb{S}^{D-1}$. By the Cramér–Wold theorem [^13], matching all one-dimensional projected distributions is equivalent to matching the full joint distribution; SIGReg approximates this objective using a finite number of random projections. A detailed configuration of SIGReg is provided in Appendix A.1.

The overall training loss for AD-E2E-JEPA is given by Eq. (11):

$$
\mathcal{L}=\mathcal{L}^{\mathrm{proj}}_{\mathrm{pred}}+\lambda\mathcal{L}^{t-K:t+1}_{\mathrm{SIGReg}}.
$$

We optionally add rollout training, as in JEPA-WM [^47], over the entire future horizon $F$ for E2EAD. The training objective consists of a teacher-forcing prediction loss, $\overline{\mathcal{L}}_{\mathrm{TF}}$, over the full future horizon from the next frame through frame $t+F$. We also apply rollout losses, $\overline{\mathcal{L}}_{2}+\cdots+\overline{\mathcal{L}}_{F}$, based on autoregressive predictions from frame $t+2$ through frame $t+F$. These rollout losses encourage consistency. In addition, we apply SIGReg from frame $t-K$ through frame $t+F$. The overall loss is given by Eq. (12), with further details provided in Appendix A.2.

$$
\mathcal{L}^{\mathrm{multi}}=\frac{\overline{\mathcal{L}}_{\mathrm{TF}}+\overline{\mathcal{L}}_{2}+\cdots+\overline{\mathcal{L}}_{F}}{F}+\lambda\mathcal{L}^{t-K:t+F}_{\mathrm{SIGReg}}.
$$

### 3.3 Zero-Shot Goal-Conditioned Planning

World models enable zero-shot goal-conditioned planning without training a policy, allowing generalization to unseen scenes. This paradigm has been widely adopted in JEPA-based world models. We are the first to introduce it to E2EAD, significantly improve planning efficiency, and report several quantitative metrics to evaluate world-model beyond the success-rate metric used in prior work.

#### 3.3.1 Planning with AD-E2E-JEPA

The cross-entropy method (CEM) is widely used for JEPA-based world-model planning [^59] [^47] [^44] but is too costly for autonomous driving due to its iterative evaluation of many candidate action sequences. We instead search over the clustered driving trajectory vocabulary as anchors from [^10], $\mathcal{V}=\{P^{i}_{t:t+F}\}_{i=1}^{8192}$. To balance efficiency and trajectory granularity, we sort trajectories by angular coordinate and subsample them at evenly spaced intervals to obtain e.g., $|V_{\text{sampled}}|=256$ candidates.

We define the goal as the future image $I_{t+F}$, $F$ frames ahead. Given our action-conditioned world model, we perform zero-shot planning by selecting the candidate whose predicted future latent embedding is closest to that of the goal. For the $i$ -th candidate trajectory, the planning cost is

$$
\mathcal{C}^{i}=\left\|z_{t+F}-\hat{z}_{t+F}^{\,i}\right\|_{2}^{2},
$$

where $\hat{z}_{t+F}^{\,i}$ is obtained by autoregressively rolling out the world model from the initial projected observations $z_{t-K:t}$ under the $i$ -th candidate trajectory using Eq. (8). We select the trajectory as

$$
i^{*}=\operatorname*{argmin}_{i\in\{1,\ldots,|V_{\text{sampled}}|\}}\mathcal{C}^{i}=\operatorname*{argmin}_{i\in\{1,\ldots,|V_{\text{sampled}}|\}}\left\|z_{t+F}-\hat{z}_{t+F}^{\,i}\right\|_{2}^{2},\qquad P^{*}_{t:t+F}=P^{i^{*}}_{t:t+F}.
$$

#### 3.3.2 Metrics

To quantitatively evaluate the reliability and efficiency of world-model zero-shot goal-conditioned planning for autonomous driving, we report metrics for driving performance (EPDMS, EPDMS <sup>†</sup>), planning efficiency, geodesic accuracy. (FDE, $\Delta x$, $\Delta y$, $\Delta\theta$), and reliability (hit rate). Detailed definitions are provided in Appendix A.3.

EPDMS: NAVSIMv2 [^8] uses EPDMS as a pseudo-simulation-based planning metric that combines multiplicative safety terms with a weighted measure of driving quality.

EPDMS <sup>†</sup>: Since zero-shot planning does not explicitly optimize for safety, we report only the weighted component of EPDMS, excluding the multiplicative safety terms.

Planning time: We report planning time to assess E2EAD planning efficiency.

Final-pose displacement (FDE, $\Delta x$, $\Delta y$, $\Delta\theta$): We measure the displacement between the final pose of the selected trajectory and the ground-truth pose. Specifically, we report the final displacement error (FDE) and absolute errors in longitudinal position ($\Delta x$), lateral position ($\Delta y$), and heading ($\Delta\theta$), which measure the geodesic accuracy. of the world model.

Hit rate: Inspired by the diagnostic metric released in the DrivoR [^26] codebase, we use hit rate to measure whether the world model ranks the ground-truth trajectory among the top- $k$ lowest-cost candidates. Specifically, we report Top-1 and Top-5 hit rates based on the rollout latent distance to the goal, which assess the reliability of the world-model rollout.

### 3.4 Downstream Transfer for Imitation Learning

Beyond zero-shot goal-conditioned planning, we evaluate whether the self-supervised projector pretrained during world modeling provides useful representations for imitation learning. Existing works [^49] [^38] show that self-supervised pretrained representations improve E2EAD, but have not explored projectors that compress dense patch embeddings.

We discard the patch predictor from AD-E2E-JEPA while retaining the DINOv3 encoder and projector, attach a simple trajectory decoder, and fully fine-tune the model for E2EAD. We compare against variants with a randomly initialized projector.

## 4 Experiments

### 4.1 Datasets

We use the NAVSIM [^8] dataset and evaluate on the latest NAVSIMv2 benchmark. We use the navtrain split, which contains 10 hours of driving video sampled at 2 Hz, for a fair comparison among LeWM, DINO-WM, JEPA-WM, and AD-E2E-JEPA, while scaling AD-E2E-JEPA to 70 hours of driving video from the training portion of the trainval split to further improve its performance.

### 4.2 Implementation Details

We use DINOv3 ViT-L backbone for DINO, JEPA-WM and AD-E2E-JEPA. LeWM’s backbone is ViT-L trained from scratch. We use $K+1=4$ frames for a 2-s history context and $F=8$ frames for a 4-s future horizon. Without the optional rollout loss, AD-E2E-JEPA uses only the first future frame in Eq. (11); the rollout-loss variant uses all 8 future frames in Eq. (12). We use AdamW with 30 training epochs for all settings and scale the learning rate with the square root of the batch size. We use one warmup epoch followed by a cosine annealing schedule. The default SIGReg weight is $\lambda=0.09$; for larger batch sizes, we tune $\lambda$ heuristically based on early training loss curves. For LeWM, DINO-WM, and JEPA-WM, we use a learning rate of $1\times 10^{-4}$ across all settings with the same learning-rate scheduler. AD-E2E-JEPA requires fewer GPU resources hours than the other methods when training settings are the same. Table 1 summarizes the training configurations. For transferring self-supervised pretrained projectors to imitation learning, see Appendix A.5.

Table 1: Training configurations.

<table><tbody><tr><td>Split</td><td>Variant</td><td>GPUs</td><td>Batch size</td><td>Learning rate</td><td><math><semantics><mi>λ</mi> <annotation>\lambda</annotation></semantics></math></td><td>Training time</td></tr><tr><td rowspan="5">navtrain</td><td>LeWM</td><td><math><semantics><mrow><mn>4</mn> <mo>×</mo></mrow> <annotation>4\times</annotation></semantics></math> A100</td><td>8</td><td><math><semantics><mrow><mn>1</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>4</mn></mrow></msup></mrow> <annotation>1\times 10^{-4}</annotation></semantics></math></td><td>0.09</td><td>1 d</td></tr><tr><td>DINO-WM</td><td><math><semantics><mrow><mn>4</mn> <mo>×</mo></mrow> <annotation>4\times</annotation></semantics></math> A100</td><td>64</td><td><math><semantics><mrow><mn>1</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>4</mn></mrow></msup></mrow> <annotation>1\times 10^{-4}</annotation></semantics></math></td><td>–</td><td>11 h</td></tr><tr><td>JEPA-WM</td><td><math><semantics><mrow><mn>4</mn> <mo>×</mo></mrow> <annotation>4\times</annotation></semantics></math> A100</td><td>64</td><td><math><semantics><mrow><mn>1</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>4</mn></mrow></msup></mrow> <annotation>1\times 10^{-4}</annotation></semantics></math></td><td>–</td><td>13 h</td></tr><tr><td>AD-E2E-JEPA</td><td><math><semantics><mrow><mn>1</mn> <mo>×</mo></mrow> <annotation>1\times</annotation></semantics></math> A100</td><td>128</td><td><math><semantics><mrow><mn>1</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>4</mn></mrow></msup></mrow> <annotation>1\times 10^{-4}</annotation></semantics></math></td><td>0.09</td><td>20 h</td></tr><tr><td>    + rollout</td><td><math><semantics><mrow><mn>1</mn> <mo>×</mo></mrow> <annotation>1\times</annotation></semantics></math> A100</td><td>128</td><td><math><semantics><mrow><mn>1</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>4</mn></mrow></msup></mrow> <annotation>1\times 10^{-4}</annotation></semantics></math></td><td>0.09</td><td>1 d 22 h</td></tr><tr><td rowspan="2">trainval</td><td>AD-E2E-JEPA</td><td><math><semantics><mrow><mn>4</mn> <mo>×</mo></mrow> <annotation>4\times</annotation></semantics></math> A100</td><td>512</td><td><math><semantics><mrow><mn>2</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>4</mn></mrow></msup></mrow> <annotation>2\times 10^{-4}</annotation></semantics></math></td><td>0.025</td><td>2 d 5 h</td></tr><tr><td>    + rollout</td><td><math><semantics><mrow><mn>4</mn> <mo>×</mo></mrow> <annotation>4\times</annotation></semantics></math> A100</td><td>256</td><td><math><semantics><mrow><mn>1.4</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>4</mn></mrow></msup></mrow> <annotation>1.4\times 10^{-4}</annotation></semantics></math></td><td>0.09</td><td>4 d 2 h</td></tr></tbody></table>

### 4.3 Zero-Shot Planning Performance

We set the number of subsampled trajectories to 256 unless specified. We evaluate zero-shot goal-conditioned planning on 100 sampled scenes from the NAVSIM test split for all methods. We additionally evaluate the computationally efficient models on the full test set. DINO-WM and JEPA-WM operate on dense patch embeddings after the encoder, making full-set evaluation prohibitively expensive (over 10 days). AD-E2E-JEPA performs planning following Section 3.3.1. LeWM, DINO-WM, and JEPA-WM follow the same planning procedure, except that DINO-WM and JEPA-WM operate on patch embeddings, whereas LeWM operates on the global CLS token. We omit these formulations for brevity. The qualitative planning visualization across these methods is in Figure 3.

![[visualization 1.png|Refer to caption]]

Figure 3: Zero-shot goal-conditioned planning qualitative comparison. LeWM is efficient but insufficient for accurate planning, resulting in a shifted selected trajectory. DINO-WM and JEPA-WM better match the ground-truth trajectory. AD-E2E-JEPA achieves comparable planning quality to DINO-WM and JEPA-WM while being 100 × \\times faster.

Table 2: Zero-shot goal-conditioned planning on the NAVSIM dataset, evaluated on 100 subsampled test scenes and the full set of 12,146 test scenes, with the ground-truth future frame used as the target. Unless specified, the world model selects from 256 candidate trajectories. The NAVSIM-v2 navtest-stage-1 metrics EPDMS $\uparrow$ and EPDMS ${}^{\dagger}\uparrow$ measure driving performance, where EPDMS includes safety metrics and EPDMS <sup>†</sup> excludes them. Per-scene planning time (s) measures efficiency on an A100 GPU. FDE (m), $\Delta x$ (m), $\Delta y$ (m), and $\Delta\theta$ (degrees) measure geodesic accuracy. Top-1/top-5 hit rate (%) measures world-model reliability. $\uparrow$: higher is better; $\downarrow$: lower is better. For the 100 subsampled-test scene setting, the extended comfort (EC) metrics for EPDMS and EPDMS <sup>†</sup> are excluded because the subsampled set does not contain temporally adjacent scenes. See Appendix A.4 for details of each submetric of EPDMS.

<table><tbody><tr><th></th><td></td><td colspan="2">Driving performance</td><td>Efficiency</td><td colspan="4">Geodesic accuracy</td><td>Reliability</td></tr><tr><th>Method</th><td>Split</td><td>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EPDMS <math><semantics><mrow><msup><mo>†</mo></msup> <mo>↑</mo></mrow> <annotation>{}^{\dagger}\uparrow</annotation></semantics></math></td><td>Time <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>FDE <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td><math><semantics><mrow><mi>Δ</mi> <mo></mo><mi>x</mi></mrow> <annotation>\Delta x</annotation></semantics></math> <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td><math><semantics><mrow><mi>Δ</mi> <mo></mo><mi>y</mi></mrow> <annotation>\Delta y</annotation></semantics></math> <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td><math><semantics><mrow><mi>Δ</mi> <mo></mo><mi>θ</mi></mrow> <annotation>\Delta\theta</annotation></semantics></math> <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>Hit rate <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><th colspan="10">100 subsampled test scenes</th></tr><tr><th>LeWM</th><td rowspan="5">navtrain</td><td>48.3</td><td>73.9</td><td>0.7</td><td>12.4</td><td>11.3</td><td>2.7</td><td>13.6</td><td>6/18</td></tr><tr><th>DINO-WM</th><td>68.3</td><td>91.4</td><td>91.8</td><td>3.9</td><td>3.4</td><td>1.1</td><td>5.9</td><td>40/73</td></tr><tr><th>JEPA-WM</th><td>74.2</td><td>90.9</td><td>101.0</td><td>4.0</td><td>3.4</td><td>1.2</td><td>5.2</td><td>45/75</td></tr><tr><th>AD-E2E-JEPA</th><td>76.6</td><td>92.4</td><td>0.8</td><td>4.2</td><td>3.9</td><td>1.0</td><td>4.7</td><td>27/59</td></tr><tr><th>+ rollout</th><td>70.4</td><td>92.2</td><td>0.8</td><td>3.5</td><td>3.2</td><td>1.0</td><td>4.1</td><td>34/67</td></tr><tr><th>AD-E2E-JEPA</th><td rowspan="2">trainval</td><td>72.1</td><td>89.8</td><td>0.8</td><td>4.7</td><td>4.4</td><td>0.9</td><td>3.4</td><td>33/55</td></tr><tr><th>+ rollout</th><td>72.3</td><td>91.9</td><td>0.8</td><td>3.2</td><td>3.0</td><td>0.7</td><td>3.6</td><td>47/78</td></tr><tr><th colspan="10">Full 12,146 test scenes</th></tr><tr><th>LeWM</th><td rowspan="3">navtrain</td><td>39.8</td><td>66.7</td><td>0.7</td><td>14.6</td><td>13</td><td>3.5</td><td>17.2</td><td>5.7/11.3</td></tr><tr><th>AD-E2E-JEPA</th><td>63.5</td><td>80.1</td><td>0.8</td><td>6.3</td><td>5.9</td><td>1.2</td><td>6.3</td><td>32.9/65.3</td></tr><tr><th>+ rollout</th><td>64.9</td><td>83.1</td><td>0.8</td><td>4.5</td><td>4.0</td><td>1.2</td><td>4.1</td><td>42.6/71.0</td></tr><tr><th>AD-E2E-JEPA</th><td rowspan="7">trainval</td><td>63.2</td><td>80.1</td><td>0.8</td><td>6.2</td><td>5.8</td><td>1.1</td><td>4.5</td><td>31.0/64.8</td></tr><tr><th>+ rollout</th><td>67.3</td><td>84.1</td><td>0.8</td><td>4.0</td><td>3.6</td><td>1.1</td><td>3.5</td><td>53.8/82.7</td></tr><tr><th>+ rollout, 512 traj.</th><td>69.2</td><td>85.0</td><td>1.4</td><td>3.6</td><td>3.3</td><td>0.9</td><td>2.9</td><td>45.2/73.9</td></tr><tr><th>+ rollout, 1024 traj.</th><td>70.5</td><td>85.5</td><td>2.5</td><td>3.2</td><td>2.9</td><td>0.8</td><td>2.5</td><td>36.3/64.3</td></tr><tr><th>+ rollout, 2048 traj.</th><td>71.5</td><td>86.0</td><td>4.7</td><td>3.0</td><td>2.7</td><td>0.8</td><td>2.2</td><td>27.4/53.2</td></tr><tr><th>+ rollout, 4096 traj.</th><td>72.1</td><td>86.3</td><td>9.3</td><td>2.9</td><td>2.6</td><td>0.8</td><td>2.1</td><td>20.7/43.3</td></tr><tr><th>+ rollout, 8192 traj.</th><td>72.9</td><td>86.5</td><td>18.2</td><td>2.8</td><td>2.5</td><td>0.8</td><td>2.0</td><td>15.3/33.8</td></tr></tbody></table>

As shown in Table 2, LeWM is computationally efficient but achieves substantially lower driving performance, geodesic accuracy, and hit rates than the other methods. DINO-WM and JEPA-WM are considerably more reliable, achieving higher hit rates and EPDMS scores of 68.3 and 74.2, respectively, on the 100-scene subset. However, their per-scene planning times are 91.8 and 101.0 seconds, respectively, making full-test-set evaluation prohibitively expensive. In contrast, AD-E2E-JEPA requires only 0.8 seconds per scene. When trained on navtrain, AD-E2E-JEPA achieves the highest driving performance on the 100-scene subset, although its geodesic accuracy and hit rate are lower than those of DINO-WM and JEPA-WM.

Adding the rollout loss and training on the larger trainval split substantially improves geodesic accuracy and reliability. On the 100-scene subset, this configuration achieves an FDE of 3.2 m and a top-1/top-5 hit rate of 47%/78%, comparable to or better than DINO-WM and JEPA-WM on these metrics. Its EPDMS is lower on this subset, which may partly reflect variance from evaluating only 100 scenes. This gap is reduced when evaluating on the full test set. Another possible explanation is that the world model is trained using a latent-distance objective, which is more directly aligned with geodesic distance and hit rate than with the NAVSIM-v2 driving metrics.

On the full 12,146-scene test set, AD-E2E-JEPA trained on trainval with rollout loss achieves an EPDMS of 67.3 and an EPDMS <sup>†</sup> of 84.1. It also achieves an FDE of 4.0 m and a top-1/top-5 hit rate of 54%/83%, providing the strongest overall results on the full test set and significantly outperforms LeWM in the same setting. When we further scales the subsampled candidates to 512, 1024, 2048, 4096, 8192, it further boosts the performance and 8192 variant reaches an EPDMS of 72.9 and an EPDMS <sup>†</sup> of 86.5 with FDE of 2.8 m. Hit rate decreases accordingly as the number of candidates increases.

### 4.4 Transfer Learning Performance

Recent high-performing E2E autonomous driving methods can require substantial training compute. For example, WA-JEPA [^50], which reports state-of-the-art NAVSIM-v2 performance at submission time, uses 64 A800 GPUs for Stage 1 pretraining and 32 A800 GPUs for Stage 2 training. Rather than pursuing gains through large-scale training, we study a complementary question: whether the lightweight projector learned by self-supervised AD-E2E-JEPA world modeling transfers useful predictive structure to downstream imitation learning. We isolate the effect of projector pretraining without additional self-supervised pretraining of the encoder backbone. While prior JEPA-based E2E driving methods mainly study pretrained encoder representations or jointly pretrained world-action models, we are not aware of prior work that specifically isolates the downstream transferability of a pretrained projector. We include Transfuser [^12], Latent-WAM [^48], Drive-JEPA [^49], and WA-JEPA [^50] as reference methods rather than direct baselines.

Table 3: NAVSIMv2 navtest-stage-1 benchmark. V: single-view (SV) or multi-view (MV); Type: perception-free (PF) or perception-based (PB); Fr.: number of input frames. PB methods use perception annotations or trajectory-score labels. <sup>∗</sup> denotes results obtained before the human-filter bug fix in commit 359c7f7 in NAVSIM’s codebase.

<table><tbody><tr><td></td><td></td><td></td><td></td><td colspan="11">NAVSIMv2 stage 1 driving metrics</td></tr><tr><td>Method</td><td>V</td><td>Type</td><td>Fr.</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DDC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TLC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>LK <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>HC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EPDMS <math><semantics><mrow><msup><mo>∗</mo></msup> <mo>↑</mo></mrow> <annotation>{}^{*}\uparrow</annotation></semantics></math></td><td>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>Transfuser</td><td>MV</td><td>PF</td><td>1</td><td>96.9</td><td>89.9</td><td>97.8</td><td>99.7</td><td>87.1</td><td>95.4</td><td>92.7</td><td>98.3</td><td>87.2</td><td>76.7</td><td>–</td></tr><tr><td>Latent-WAM</td><td>MV</td><td>PF</td><td>4</td><td>98.1</td><td>97.3</td><td>99.6</td><td>99.8</td><td>87.7</td><td>97.3</td><td>97.6</td><td>98.1</td><td>72.4</td><td>–</td><td>89.3</td></tr><tr><td>WA-JEPA</td><td>MV</td><td>PF</td><td>4</td><td>99.4</td><td>98.2</td><td>99.7</td><td>99.9</td><td>87.8</td><td>98.9</td><td>98.3</td><td>98.3</td><td>88.1</td><td>88.0</td><td>91.7</td></tr><tr><td>Drive-JEPA</td><td>SV</td><td>PB</td><td>2</td><td>98.4</td><td>98.6</td><td>99.1</td><td>99.8</td><td>88.4</td><td>97.8</td><td>97.6</td><td>97.9</td><td>84.8</td><td>87.8</td><td>–</td></tr><tr><td colspan="15">DINOv3</td></tr><tr><td>+ rand. proj.</td><td>SV</td><td>PF</td><td>4</td><td>96.8</td><td>89.8</td><td>98.3</td><td>99.7</td><td>87.1</td><td>95.7</td><td>94.8</td><td>98.3</td><td>84.1</td><td>–</td><td>80.2</td></tr><tr><td>+ AD-E2E-JEPA proj.</td><td>SV</td><td>PF</td><td>4</td><td>97.7</td><td>93.7</td><td>99.2</td><td>99.8</td><td>87.3</td><td>96.8</td><td>97.0</td><td>98.4</td><td>88.7</td><td>–</td><td>85.4</td></tr></tbody></table>

## 5 Conclusion

We propose AD-E2E-JEPA, a joint-embedding predictive architecture for end-to-end autonomous driving. To isolate the effect of world modeling from driving policy learning, we perform zero-shot goal-conditioned planning and quantitatively evaluate the JEPA-based world model in terms of driving performance, planning efficiency, geodesic accuracy, and reliability. We further introduce a learnable projector with SIGReg to compress latent representations for planning while preserving their information content. Without any policy training, AD-E2E-JEPA improves planning efficiency by $100\times$ compared with DINO-WM/JEPA-WM under the same setting of 256 candidate trajectories. The best AD-E2E-JEPA variant, using 8192 candidate trajectories, achieves an EPDMS of 72.9, reaches goal with a displacement error of 2.8 meters and a mean heading error of 2.0 degrees. It also demonstrates the projector pretrained during world modeling benefits imitation learning.

## References

## Appendix A Appendix

### A.1 SIGReg Configuration Details

$T(\cdot)$ denotes the univariate Epps–Pulley test [^15]. We use the default SIGReg settings [^3] used in LeWorld [^36] [^29], where the integral is approximated with 17 knots over $[0,3]$ and $M=1024$ directions.

### A.2 Rollout Setting

For the AD-E2E-JEPA variant with rollout training in Eq. (12), we adapt the rollout implementation of [^47] to the full future horizon of $F$ frames. Each prediction window contains $K+2$ consecutive frames: the first $K+1$ frames are used as input, and the predictor produces the corresponding one-step-shifted predictions. In the NAVSIM setting, $F=8$ and $K+1=4$.

#### A.2.1 Teacher-Forcing Loss

The first component is the teacher-forcing prediction loss $\overline{\mathcal{L}}_{\mathrm{TF}}$. For each time window, we use the ground-truth context embeddings as input, without autoregressive rollout. In our current implementation: for $k=0$, we supervise the entire prediction window; For each subsequent teacher-forced prediction ($k\geq 1$), we supervise only the final time step of the prediction window. Specifically,

$$
\hat{z}^{\mathrm{TF}}_{t+k-K+1:t+k+1}=\operatorname{Pred}^{\prime}\!\left(z_{t+k-K:t+k},E_{a}(\mathbf{a}_{t+k-K:t+k})\right),\qquad k=0,\ldots,F-1.
$$

For the first prediction window,

$$
\mathcal{L}^{\mathrm{proj}}_{\mathrm{pred}}[0]=\operatorname{MSE}\!\left(\hat{z}^{\mathrm{TF}}_{t-K+1:t+1},\operatorname{sg}\!\left(z_{t-K+1:t+1}\right)\right).
$$

For the subsequent prediction windows,

$$
\mathcal{L}^{\mathrm{proj}}_{\mathrm{pred}}[k]=\operatorname{MSE}\!\left(\hat{z}^{\mathrm{TF}}_{t+k+1},\operatorname{sg}\!\left(z_{t+k+1}\right)\right),\qquad k=1,\ldots,F-1.
$$

The teacher-forcing loss is then

$$
\overline{\mathcal{L}}_{\mathrm{TF}}=\frac{(K+1)\,\mathcal{L}^{\mathrm{proj}}_{\mathrm{pred}}[0]+\sum_{k=1}^{F-1}\mathcal{L}^{\mathrm{proj}}_{\mathrm{pred}}[k]}{K+F}.
$$

#### A.2.2 Rollout Consistency Loss

The second component consists of the rollout consistency losses $\overline{\mathcal{L}}_{k}$ for future horizons $k=2,\ldots,F$. We perform a single autoregressive rollout of $F$ steps starting from the initial context window $z_{t-K:t}$. Throughout the rollout, we maintain a context window of $K+1$ embeddings. After each prediction, we remove the oldest context embedding and append only the final predicted embedding.

Specifically, we initialize the autoregressive context as

$$
\tilde{z}^{\mathrm{AR},(0)}_{t-K:t}=z_{t-K:t}.
$$

We reuse the initial teacher-forcing prediction as the first autoregressive prediction, which uses the initial context without stop-gradient:

$$
\hat{z}^{\mathrm{AR},(1)}_{t-K+1:t+1}=\operatorname{Pred}^{\prime}\!\left(\tilde{z}^{\mathrm{AR},(0)}_{t-K:t},E_{a}(\mathbf{a}_{t-K:t})\right).
$$

Following [^47], we apply truncated backpropagation through time (TBPTT) [^23] by stopping gradients through the autoregressive context before each subsequent predictor call. For rollout steps $j=2,\ldots,F$, the predictor produces

$$
\hat{z}^{\mathrm{AR},(j)}_{t-K+j:t+j}=\operatorname{Pred}^{\prime}\!\left(\operatorname{sg}\!\left(\tilde{z}^{\mathrm{AR},(j-1)}_{t-K+j-1:t+j-1}\right),E_{a}(\mathbf{a}_{t-K+j-1:t+j-1})\right).
$$

After each prediction, the autoregressive context is updated by removing its earliest embedding and appending the final predicted embedding:

$$
\tilde{z}^{\mathrm{AR},(j)}_{t-K+j:t+j}=\left[\tilde{z}^{\mathrm{AR},(j-1)}_{t-K+j:t+j-1},\hat{z}^{\mathrm{AR},(j)}_{t+j}\right],\qquad j=1,\ldots,F.
$$

The rollout consistency loss at horizon $k$ supervises only the final prediction in the corresponding window against the ground-truth embedding:

$$
\overline{\mathcal{L}}_{k}=\operatorname{MSE}\!\left(\hat{z}^{\mathrm{AR},(k)}_{t+k},\operatorname{sg}\!\left(z_{t+k}\right)\right),\qquad k=2,\ldots,F.
$$

The first-step prediction is supervised by the teacher-forcing loss and is therefore excluded from the additional rollout loss terms. Each horizon loss remains differentiable through its current predictor call, while stop-gradient prevents backpropagation through earlier rollout steps.

#### A.2.3 SIGReg

The third component is SIGReg, which is applied over all time steps from $t-K$ through $t+F$, yielding $\mathcal{L}^{t-K:t+F}_{\mathrm{SIGReg}}$.

Finally, the overall loss for the rollout setting is given by Eq. (12).

### A.3 Metrics

EPDMS. EPDMS is a pseudo-simulation-based metric for evaluating end-to-end autonomous driving planning, introduced in the NAVSIM benchmark [^8] [^14]. It evaluates both safety and driving quality. Specifically, EPDMS consists of a multiplicative term capturing safety-related metrics, including no at-fault collision (NC), drivable area compliance (DAC), driving direction compliance (DDC), and traffic light compliance (TLC), and a weighted-average term capturing driving quality, including ego progress (EP), time-to-collision (TTC), lane keeping (LK), history comfort (HC), and extended comfort (EC). EC is included only for samples for which a valid neighboring scene is available.

We define an indicator $b_{i}$ denoting the availability of a valid neighboring scene:

$$
b_{i}=\begin{cases}1,&\text{if a valid neighboring scene is available for computing EC},\\
0,&\text{otherwise}.\end{cases}
$$

The EPDMS for sample $i$ is then computed as

$$
\mathrm{EPDMS}_{i}=\mathrm{NC}_{i}\cdot\mathrm{DAC}_{i}\cdot\mathrm{DDC}_{i}\cdot\mathrm{TLC}_{i}\cdot\frac{5\,\mathrm{EP}_{i}+5\,\mathrm{TTC}_{i}+2\,\mathrm{LK}_{i}+2\,\mathrm{HC}_{i}+2\,b_{i}\,\mathrm{EC}_{i}}{14+2\,b_{i}}.
$$

EPDMS <sup>†</sup>. Since zero-shot goal-conditioned planning with a world model is not directly optimized for the safety-related components of EPDMS, we additionally report EPDMS <sup>†</sup>, which excludes the multiplicative safety terms and evaluates driving quality using only the weighted-average component:

$$
\mathrm{EPDMS}^{\dagger}_{i}=\frac{5\,\mathrm{EP}_{i}+5\,\mathrm{TTC}_{i}+2\,\mathrm{LK}_{i}+2\,\mathrm{HC}_{i}+2\,b_{i}\,\mathrm{EC}_{i}}{14+2\,b_{i}}.
$$

Planning time. We report the average per-scene planning time to assess the computational efficiency of end-to-end autonomous driving planning using a single NVIDIA A100 80 GB GPU, with all candidate trajectories evaluated in parallel on the GPU.

Final-pose error (FDE, $\Delta x$, $\Delta y$, $\Delta\theta$). We evaluate the final-pose error between the ground-truth pose corresponding to the goal image and the final pose of the selected trajectory. FDE denotes the mean Euclidean distance between the ground-truth and selected final positions. We additionally report the mean absolute errors along the $x$ - and $y$ -axes, denoted by $\Delta x$ and $\Delta y$, respectively, as well as the mean absolute heading error $\Delta\theta$.

Hit rate. Hit rate is a diagnostic metric introduced in the DrivoR [^26] codebase, where it was originally used to evaluate whether the proposed scorer can identify high-scoring trajectories. Inspired by this idea, we adapt hit rate to evaluate the reliability of the world model. Given the subsampled candidate trajectory set $V_{\text{sampled}}$, we append the ground-truth trajectory to the candidate set and perform world-model rollouts for all trajectories. We then compute the planning cost for each trajectory according to Eq. (13) and determine whether the ground-truth trajectory ranks among the top- $k$ trajectories with the lowest planning costs. Specifically, we report Top-1 and Top-5 hit rates, which measure the percentage of evaluation scenes in which the ground-truth trajectory is ranked among the top 1 and top 5 lowest-cost trajectories, respectively.

For each evaluation scene $n$, we define

$$
h_{n}^{@k}=\mathbb{I}\left[\operatorname{rank}\!\left(C_{n}^{\mathrm{gt}}\right)\leq k\right],
$$

where $C_{n}^{\mathrm{gt}}$ denotes the planning cost of the ground-truth trajectory, and the rank is computed over the planning costs of all trajectories in $V_{\text{sampled}}\cup\{P_{n}^{\mathrm{gt}}\}$, with lower planning costs corresponding to higher ranks. The Top- $k$ hit rate is then

$$
\mathrm{HitRate}@k=\frac{1}{Q}\sum_{n=1}^{Q}h_{n}^{@k},\qquad k\in\{1,5\},
$$

where $Q$ denotes the number of evaluation scenes.

### A.4 EPDMS Details for Zero-Shot Goal-Conditioned Planning

Detailed EPDMS results are reported in Table 4 for 100 subsampled test scenes and in Table 5 for the full set of 12,146 test scenes.

Table 4: EPDMS details for 100 subsampled test scenes.

<table><tbody><tr><th>Method</th><td>Split</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DDC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TLC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>LK <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>HC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EPDMS <sup>†</sup> <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><th>LeWM</th><td rowspan="5">navtrain</td><td>78.5</td><td>77.0</td><td>84.0</td><td>97.0</td><td>68.6</td><td>76.0</td><td>86.0</td><td>70.0</td><td>–</td><td>48.3</td><td>73.9</td></tr><tr><th>DINO-WM</th><td>96.0</td><td>79.0</td><td>96.5</td><td>99.0</td><td>87.8</td><td>93.0</td><td>92.0</td><td>96.0</td><td>–</td><td>68.3</td><td>91.4</td></tr><tr><th>JEPA-WM</th><td>94.5</td><td>87.0</td><td>96.5</td><td>99.0</td><td>87.6</td><td>92.0</td><td>94.0</td><td>93.0</td><td>–</td><td>74.2</td><td>90.9</td></tr><tr><th>AD-E2E-JEPA</th><td>96.0</td><td>87.0</td><td>95.5</td><td>99.0</td><td>88.0</td><td>95.0</td><td>94.0</td><td>95.0</td><td>–</td><td>76.6</td><td>92.4</td></tr><tr><th>+ rollout</th><td>96.0</td><td>77.0</td><td>95.5</td><td>100.0</td><td>87.9</td><td>95.0</td><td>93.0</td><td>95.0</td><td>–</td><td>70.4</td><td>92.2</td></tr><tr><th>AD-E2E-JEPA</th><td rowspan="2">trainval</td><td>94.0</td><td>82.0</td><td>96.5</td><td>100.0</td><td>87.2</td><td>92.0</td><td>91.0</td><td>90.0</td><td>–</td><td>72.1</td><td>89.8</td></tr><tr><th>+ rollout</th><td>95.5</td><td>82.0</td><td>98.0</td><td>99.0</td><td>87.3</td><td>93.0</td><td>97.0</td><td>96.0</td><td>–</td><td>72.3</td><td>91.9</td></tr></tbody></table>

Table 5: EPDMS details for the full set of 12,146 test scenes.

<table><tbody><tr><th>Method</th><td>Split</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DDC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TLC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>LK <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>HC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EPDMS <sup>†</sup> <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><th>LeWM</th><td rowspan="3">navtrain</td><td>82.0</td><td>67.5</td><td>81.4</td><td>98.5</td><td>66.1</td><td>79.5</td><td>79.0</td><td>66.7</td><td>13.7</td><td>39.8</td><td>66.7</td></tr><tr><th>AD-E2E-JEPA</th><td>93.1</td><td>82.4</td><td>95.6</td><td>99.5</td><td>80.6</td><td>90.9</td><td>89.4</td><td>89.6</td><td>21.7</td><td>63.5</td><td>80.1</td></tr><tr><th>+ rollout</th><td>94.7</td><td>80.9</td><td>93.8</td><td>99.7</td><td>84.0</td><td>92.5</td><td>87.9</td><td>91.8</td><td>34.6</td><td>64.9</td><td>83.1</td></tr><tr><th>AD-E2E-JEPA</th><td rowspan="7">trainval</td><td>92.8</td><td>82.8</td><td>95.3</td><td>99.4</td><td>82.0</td><td>90.6</td><td>89.8</td><td>88.5</td><td>19.5</td><td>63.2</td><td>80.1</td></tr><tr><th>+ rollout</th><td>95.2</td><td>82.0</td><td>94.4</td><td>99.6</td><td>85.2</td><td>93.3</td><td>89.3</td><td>92.3</td><td>35.5</td><td>67.3</td><td>84.1</td></tr><tr><th>+ rollout, 512 traj.</th><td>95.9</td><td>83.6</td><td>95.5</td><td>99.6</td><td>85.6</td><td>94.4</td><td>90.4</td><td>93.5</td><td>36.9</td><td>69.2</td><td>85.0</td></tr><tr><th>+ rollout, 1024 traj.</th><td>96.4</td><td>84.2</td><td>96.1</td><td>99.6</td><td>85.6</td><td>95.0</td><td>91.0</td><td>94.1</td><td>38.5</td><td>70.5</td><td>85.5</td></tr><tr><th>+ rollout, 2048 traj.</th><td>96.7</td><td>84.8</td><td>96.4</td><td>99.7</td><td>85.6</td><td>95.4</td><td>91.5</td><td>94.5</td><td>40.9</td><td>71.5</td><td>86.0</td></tr><tr><th>+ rollout, 4096 traj.</th><td>96.7</td><td>85.4</td><td>96.4</td><td>99.7</td><td>85.5</td><td>95.6</td><td>91.5</td><td>95.0</td><td>42.5</td><td>72.1</td><td>86.3</td></tr><tr><th>+ rollout, 8192 traj.</th><td>96.9</td><td>86.0</td><td>96.7</td><td>99.7</td><td>85.4</td><td>95.7</td><td>91.8</td><td>95.1</td><td>43.8</td><td>72.9</td><td>86.5</td></tr></tbody></table>

### A.5 Downstream Imitation Learning

![[imitation_learning_architecture.png|Refer to caption]]

Figure 4: Downstream imitation learning architecture.

For downstream imitation learning, we adapt the simple ViT-based architecture used in Drive-JEPA [^49]. We replace the encoder with a pretrained DINOv3 encoder and append the proposed patch projector. We additionally incorporate temporal embeddings and patch positional embeddings, with the patch positional embeddings shared across frames. The encoded driving command is concatenated with the projected patch embeddings. The future trajectory is encoded as a query and processed by a cross-attention layer, followed by an MLP that predicts the ground-truth future trajectory. We train the network using an MSE loss. We compare the performance of a randomly initialized projector with that of the pretrained projector obtained during AD-E2E-JEPA world modeling. The input context consists of four front-camera frames, each represented as a $3\times 256\times 512$ tensor. The architecture is illustrated in Figure 4.

[^1]: M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat, Y. LeCun, and N. Ballas Self-supervised learning from images with a joint-embedding predictive architecture. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15619–15629. Cited by: §2.1.

[^2]: M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes, M. Muckley, A. Rizvi, C. Roberts, K. Sinha, A. Zholus, et al. V-jepa 2: self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985. Cited by: §2.1, §3.2.1, §3.2.2.

[^3]: R. Balestriero and Y. LeCun Lejepa: provable and scalable self-supervised learning without the heuristics. arXiv preprint arXiv:2511.08544. Cited by: §A.1, 3rd item, §2.1, §3.2.2.

[^4]: A. Bardes, Q. Garrido, J. Ponce, X. Chen, M. Rabbat, Y. LeCun, M. Assran, and N. Ballas Revisiting feature prediction for learning visual representations from video. arXiv preprint arXiv:2404.08471. Cited by: §2.1.

[^5]: A. Bardes, J. Ponce, and Y. LeCun Vicreg: variance-invariance-covariance regularization for self-supervised learning. arXiv preprint arXiv:2105.04906. Cited by: §2.1.

[^6]: M. Bojarski, D. Del Testa, D. Dworakowski, B. Firner, B. Flepp, P. Goyal, L. D. Jackel, M. Monfort, U. Muller, J. Zhang, et al. End to end learning for self-driving cars. arXiv preprint arXiv:1604.07316. Cited by: §1.

[^7]: A. E. Bryson Applied optimal control: optimization, estimation and control. CRC press. Cited by: §2.1.

[^8]: W. Cao, M. Hallgarten, T. Li, D. Dauner, X. Gu, C. Wang, Y. Miron, M. Aiello, H. Li, I. Gilitschenski, et al. Pseudo-simulation for autonomous driving. arXiv preprint arXiv:2506.04218. Cited by: §A.3, §3.3.2, §4.1.

[^9]: L. Chen, P. Wu, K. Chitta, B. Jaeger, A. Geiger, and H. Li End-to-end autonomous driving: challenges and frontiers. IEEE Transactions on Pattern Analysis and Machine Intelligence. Cited by: §1, §2.2.

[^10]: S. Chen, B. Jiang, H. Gao, B. Liao, Q. Xu, Q. Zhang, C. Huang, W. Liu, and X. Wang Vadv2: end-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243. Cited by: §2.2, §3.3.1.

[^11]: X. Chen and K. He Exploring simple siamese representation learning. In 2021 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp. 15745–15753. Cited by: §3.2.2.

[^12]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger Transfuser: imitation with transformer-based sensor fusion for autonomous driving. IEEE transactions on pattern analysis and machine intelligence 45 (11), pp. 12878–12895. Cited by: §2.2, §4.4.

[^13]: H. Cramér and H. Wold Some theorems on distribution functions. Journal of the London Mathematical Society 1 (4), pp. 290–294. Cited by: §3.2.2.

[^14]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, et al. Navsim: data-driven non-reactive autonomous vehicle simulation and benchmarking. Advances in Neural Information Processing Systems 37, pp. 28706–28719. Cited by: §A.3.

[^15]: T. W. Epps and L. B. Pulley A test for normality based on the empirical characteristic function. Biometrika 70 (3), pp. 723–726. Cited by: §A.1, §3.2.2.

[^16]: Z. Fei, M. Fan, and J. Huang A-jepa: joint-embedding predictive architecture can listen. arXiv preprint arXiv:2311.15830. Cited by: §2.1.

[^17]: C. Finn and S. Levine Deep visual foresight for planning robot motion. In 2017 IEEE international conference on robotics and automation (ICRA), pp. 2786–2793. Cited by: §2.1.

[^18]: Q. Garrido, N. Ballas, M. Assran, A. Bardes, L. Najman, M. Rabbat, E. Dupoux, and Y. LeCun Intuitive physics understanding emerges from self-supervised pretraining on natural videos. arXiv preprint arXiv:2502.11831. Cited by: §2.1.

[^19]: J. Grill, F. Strub, F. Altché, C. Tallec, P. Richemond, E. Buchatskaya, C. Doersch, B. Avila Pires, Z. Guo, M. Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. Advances in neural information processing systems 33, pp. 21271–21284. Cited by: §3.2.2.

[^20]: D. Ha and J. Schmidhuber World models. arXiv preprint arXiv:1803.10122 2 (3), pp. 440. Cited by: §2.1.

[^21]: A. Hu, L. Russell, H. Yeo, Z. Murez, G. Fedoseev, A. Kendall, J. Shotton, and G. Corrado Gaia-1: a generative world model for autonomous driving. arXiv preprint arXiv:2309.17080. Cited by: §2.2.

[^22]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, et al. Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 17853–17862. Cited by: §1, §2.2.

[^23]: H. Jaeger Tutorial on training recurrent neural networks, covering bppt, rtrl, ekf and the echo state network approach. GMD-Forschungszentrum Informationstechnik Bonn. Cited by: §A.2.2.

[^24]: B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang Vad: vectorized scene representation for efficient autonomous driving. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 8306–8316. Cited by: §2.2.

[^25]: B. Kang, Y. Yue, R. Lu, Z. Lin, Y. Zhao, K. Wang, G. Huang, and J. Feng How far is video generation from world model: a physical law perspective. arXiv preprint arXiv:2411.02385. Cited by: §2.1.

[^26]: E. Kirby, A. Boulch, Y. Xu, Y. Yin, G. Puy, É. Zablocki, A. Bursuc, S. Gidaris, R. Marlet, F. Bartoccioni, et al. Driving on registers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 32058–32069. Cited by: §A.3, §2.2, §3.3.2.

[^27]: Y. Kuang, Y. Dagade, Q. L. Lidec, L. Maes, R. Balestriero, and Y. LeCun LpWM: a case for sparse representations in world models. arXiv preprint arXiv:2608.22764. Cited by: §2.1.

[^28]: Y. Kuang, Y. Dagade, T. G. Rudner, R. Balestriero, and Y. LeCun Rectified lpjepa: joint-embedding predictive architectures with sparse and maximum-entropy representations. arXiv preprint arXiv:2602.01456. Cited by: §2.1.

[^29]: L. Kuhn, L. Maes, G. Serra, Q. L. Lidec, Y. LeCun, R. Balestriero, and F. Buettner LeVJEPA: efficient & scalable video pretraining without the heuristics. arXiv preprint arXiv:2608.27395. Cited by: §A.1.

[^30]: Y. LeCun et al. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. Open Review 62 (1), pp. 1–62. Cited by: §1, §2.1, §2.1.

[^31]: K. Li, Z. Li, S. Lan, Y. Xie, Z. Zhang, J. Liu, Z. Wu, Z. Yu, and J. M. Alvarez Hydra-mdp++: advancing end-to-end driving via expert-guided hydra-distillation. arXiv preprint arXiv:2503.12820. Cited by: §2.2.

[^32]: P. Li and D. Cui Navigation-guided sparse scene representation for end-to-end autonomous driving. arXiv preprint arXiv:2409.18341. Cited by: §2.2.

[^33]: Y. Li, L. Fan, J. He, Y. Wang, Y. Chen, Z. Zhang, and T. Tan Enhancing end-to-end autonomous driving with latent world model. arXiv preprint arXiv:2406.08481. Cited by: §2.2.

[^34]: Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y. Ji, Z. Li, Z. Zhu, J. Kautz, Z. Wu, et al. Hydra-mdp: end-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978. Cited by: §2.2.

[^35]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al. Diffusiondrive: truncated diffusion model for end-to-end autonomous driving. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12037–12047. Cited by: §1.

[^36]: L. Maes, Q. L. Lidec, D. Scieur, Y. LeCun, and R. Balestriero Leworldmodel: stable end-to-end joint-embedding predictive architecture from pixels. arXiv preprint arXiv:2603.19312. Cited by: §A.1, §2.1, §3.2.2.

[^37]: L. Mur-Labadia, M. Muckley, A. Bar, M. Assran, K. Sinha, M. Rabbat, Y. LeCun, and N. Ballas V-jepa 2.1: unlocking dense features in video self-supervised learning. In European Conference on Computer Vision, pp. 671–689. Cited by: §2.1.

[^38]: F. Naeinian, A. Hamza, H. Zhu, and A. Choromanska Zero-shot cross-city generalization in end-to-end autonomous driving: self-supervised versus supervised representations. arXiv preprint arXiv:2603.11417. Cited by: §2.2, §3.4.

[^39]: M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al. Dinov2: learning robust visual features without supervision. arXiv preprint arXiv:2304.07193. Cited by: §3.2.1.

[^40]: W. Peebles and S. Xie Scalable diffusion models with transformers. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4172–4182. Cited by: §2.1, §3.2.1.

[^41]: E. Perez, F. Strub, H. De Vries, V. Dumoulin, and A. Courville Film: visual reasoning with a general conditioning layer. In Proceedings of the AAAI conference on artificial intelligence, Cited by: §3.2.1.

[^42]: D. A. Pomerleau Alvinn: an autonomous land vehicle in a neural network. Advances in neural information processing systems 1. Cited by: §1.

[^43]: O. Siméoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose, V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104. Cited by: §3.2.1.

[^44]: U. Sobal, W. Zhang, K. Cho, R. Balestriero, T. G. Rudner, and Y. LeCun Learning from reward-free offline data: a case for planning with latent dynamics models. Advances in Neural Information Processing Systems 38, pp. 43905–43941. Cited by: §2.1, §3.3.1.

[^45]: J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu Roformer: enhanced transformer with rotary position embedding. Neurocomputing 568, pp. 127063. Cited by: §3.2.1.

[^46]: R. S. Sutton Dyna, an integrated architecture for learning, planning, and reacting. ACM Sigart Bulletin 2 (4), pp. 160–163. Cited by: §2.1.

[^47]: B. Terver, T. Yang, J. Ponce, A. Bardes, and Y. LeCun What drives success in physical planning with joint-embedding predictive world models?. arXiv preprint arXiv:2512.24497. Cited by: §A.2.2, §A.2, §3.2.1, §3.2.2, §3.3.1.

[^48]: L. Wang, Y. Zheng, Q. Chen, S. Li, Y. Zhang, Z. Xing, Q. Zhang, X. Li, D. Qian, P. Yang, et al. Latent-wam: latent world action modeling for end-to-end autonomous driving. arXiv preprint arXiv:2603.24581. Cited by: §4.4.

[^49]: L. Wang, Z. Yang, C. Bai, G. Zhang, X. Liu, X. Zheng, X. Long, C. Lu, and C. Lu Drive-jepa: video jepa meets multimodal trajectory distillation for end-to-end driving. arXiv preprint arXiv:2601.22032. Cited by: §A.5, §2.2, §3.4, §4.4.

[^50]: X. Wang, Y. Xiang, Y. Zhou, J. Wang, M. Huang, J. Huang, D. Wei, T. Zhou, X. Wang, G. Chen, et al. WA-jepa: rethinking the video jepa paradigm for world-action modeling in autonomous driving. arXiv preprint arXiv:2608.20974. Cited by: §2.2, §4.4.

[^51]: Y. Wang, O. Bounou, Y. LeCun, and M. Ren AdaJEPA: an adaptive latent world model. arXiv preprint arXiv:2606.32026. Cited by: §2.1.

[^52]: Y. Wang, O. Bounou, G. Zhou, R. Balestriero, T. G. J. Rudner, Y. LeCun, and M. Ren Temporal straightening for latent planning. In Forty-third International Conference on Machine Learning, External Links: [Link](https://openreview.net/forum?id=Ik1mKtUYlZ) Cited by: 3rd item, §2.1, §3.2.2.

[^53]: Z. Wang, K. Fang, and Y. LeCun Music-jepa: learning a world model of sound from action. arXiv preprint arXiv:2607.22000. Cited by: §2.1.

[^54]: H. Wu, R. Balestriero, and M. Levine VISReg: variance-invariance-sketching regularization for jepa training. arXiv preprint arXiv:2606.02572. Cited by: §2.1.

[^55]: J. Yang, Z. Chen, C. Huang, and J. Li Auto-jepa: a latent world model of continuous intent for end-to-end autonomous driving. arXiv preprint arXiv:2607.29031. Cited by: §2.2.

[^56]: K. Zhang, Z. Tang, X. Hu, X. Pan, X. Guo, Y. Liu, J. Huang, L. Yuan, Q. Zhang, X. Long, et al. Epona: autoregressive diffusion world model for autonomous driving. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 27220–27230. Cited by: §2.2, §3.2.1.

[^57]: W. Zhang, B. Terver, A. Zholus, S. Chitnis, H. Sutaria, M. Assran, R. Balestriero, A. Bar, A. Bardes, Y. LeCun, et al. Hierarchical planning with latent world models. arXiv preprint arXiv:2604.03208. Cited by: §2.1.

[^58]: R. Zhong, B. Ma, X. Chen, L. Zhang, M. Feng, Y. Wang, P. Liu, and J. Ma DA-wam: decision-aligned future latents for driving world models. arXiv preprint arXiv:2608.19085. Cited by: §2.2.

[^59]: G. Zhou, H. Pan, Y. LeCun, and L. Pinto Dino-wm: world models on pre-trained visual features enable zero-shot planning. arXiv preprint arXiv:2411.04983. Cited by: §2.1, §2.1, §3.3.1.

[^60]: H. Zhu and A. Choromanska Self-supervised jepa-based world models for lidar occupancy completion and forecasting. arXiv preprint arXiv:2602.12540. Cited by: §2.2.

[^61]: H. Zhu, Z. Dong, K. Topollai, B. Sha, and A. E. Choromanska Self-supervised representation learning with joint embedding predictive architecture for automotive lidar object detection. In Proceedings of the AAAI Conference on Artificial Intelligence, Cited by: §2.1, §2.2.