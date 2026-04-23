---
title: "DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning"
source: "https://arxiv.org/html/2506.06659v1"
author:
published:
created: 2026-04-23
description:
tags:
  - "clippings"
---
Wenhao Yao <sup>1</sup>   Zhenxin Li <sup>1, 2</sup>   Shiyi Lan <sup>2</sup>   Zi Wang <sup>2</sup>   Xinglong Sun <sup>2</sup>   Jose M. Alvarez <sup>2</sup>   Zuxuan Wu <sup>1</sup>  
<sup>1</sup> Fudan University, <sup>2</sup> NVIDIA Corresponding author.

###### Abstract

In complex driving environments, autonomous vehicles must navigate safely. Relying on a single predicted path, as in regression-based approaches, usually does not explicitly assess the safety of the predicted trajectory. Selection-based methods address this by generating and scoring multiple trajectory candidates and predicting the safety score for each, but face optimization challenges in precisely selecting the best option from thousands of possibilities and distinguishing subtle but safety-critical differences, especially in rare or underrepresented scenarios. We propose DriveSuprim to overcome these challenges and advance the selection-based paradigm through a coarse-to-fine paradigm for progressive candidate filtering, a rotation-based augmentation method to improve robustness in out-of-distribution scenarios, and a self-distillation framework to stabilize training. DriveSuprim achieves state-of-the-art performance, reaching 93.5% PDMS in NAVSIM v1 and 87.1% EPDMS in NAVSIM v2 without extra data, demonstrating superior safety-critical capabilities, including collision avoidance and compliance with rules, while maintaining high trajectory quality in various driving scenarios.

## 1 Introduction

End-to-end autonomous driving has traditionally relied on regression-based approaches that predict a single trajectory to mimic expert behavior [^38] [^6] [^3] [^23] [^18] [^34] [^16] [^54]. While regression is a common approach, it fundamentally lacks the ability to evaluate multiple alternatives in safety-critical scenarios where subtle trajectory differences can significantly impact outcomes.

In recent years, selection-based methods [^5] [^30] [^27] [^55] have clearly outperformed regression approaches. The key is their capability to generate and explicitly evaluate diverse trajectory candidates using comprehensive safety metrics such as collision risk and rule compliance [^9]. This explicit comparison enables the system to select the safest and most appropriate trajectory from multiple alternatives, addressing safety-critical issues that regression-based methods cannot address.

Our oracle study demonstrates the substantial potential of selection-based methods: when making optimal selection, these approaches can even surpass human demonstrations in safety-critical metrics (see Tab 1). This performance ceiling highlights why selection-based planning has become the preferred paradigm for autonomous driving systems requiring robust safety guarantees.

![[x1 23.png|Refer to caption]]

Figure 1: The overall pipeline of our method. Selection-based methods struggle to distinguish suboptimal “hard negative” trajectories, perform poorly in turning, and utilize hard binary labels for scoring training. Our proposed DriveSuprim introduces a coarse-to-fine refinement framework and a rotation-based data augmentation method with self-distillation to address these weaknesses. The green trajectory in the image is the ground-truth trajectory, the red and orange trajectories are obviously unsafe and seemingly correct trajectories, and the blue trajectory denotes the refined trajectory predicted by our model.

However, selection-based methods still face three critical limitations: First, selection-based methods struggle to differentiate between optimal trajectories and similar but suboptimal alternatives. This weakness makes the ideal perfect trajectory selection in Tab 1 challenging to implement and limits the model performance. During training, these models encounter thousands of trajectory candidates where the vast majority are unsafe or impractical (“easy negatives”, the red trajectory in Fig 1). These easy-to-reject options dominate the training process, causing the model to primarily learn to avoid clearly bad choices. As a result, the model receives inadequate training signals for distinguishing between reasonable-looking trajectories with subtle but important differences (“hard negatives”, the orange trajectory in Fig 1). The overwhelming number of obvious negative examples hinders the model’s ability to develop fine-grained discrimination capability, which is crucial for selecting optimal trajectories when presented with multiple plausible options. Such discrimination capability is essential for safely navigating complex driving scenarios.

Second, selection-based methods suffer from directional bias in trajectory distribution. This bias manifests as an imbalance in training data that, while reflecting real-world driving patterns where straight driving predominates, leads to models that perform relatively poorly in turning scenarios. Training with such imbalanced data naturally results in models that excel at straight-line driving but struggle with turns and complex maneuvers. Even advanced autonomous driving datasets like NAVSIM [^9] exhibit this limitation. Our analysis reveals that only 8% of ground-truth trajectories in NAVSIM involve turns exceeding 30 degrees. While this distribution may reflect typical driving patterns, it creates a significant challenge for learning models, which require sufficient examples of all maneuver types to develop robust capabilities. This directional bias significantly impairs the model’s ability to learn from and correctly execute large-angle turning trajectories, particularly in navigation-critical scenarios where turns are essential.

Table 1: PDM score of the best trajectory in the top-K candidates on ranked predicted scores.

| Top-K | NC $\uparrow$ | DAC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | C $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 99.0 | 98.7 | 86.5 | 96.2 | 100 | 91.9 |
| 4 | 99.4 | 99.6 | 89.6 | 98.0 | 100 | 94.5 |
| 16 | 99.7 | 99.8 | 92.0 | 99.1 | 100 | 96.1 |
| 256 | 100 | 100 | 97.1 | 99.9 | 100 | 98.7 |
| Human | 100 | 100 | 87.5 | 100 | 99.9 | 94.8 |

compliance. This binary approach creates hard decision boundaries where trajectories just above or below a safety threshold could be treated entirely differently. As a result, models become overly sensitive to minor changes in trajectory features, which causes inconsistent behavior when slight variations in collision risk or rule adherence could suddenly thoroughly flip a trajectory from being selected to rejected.

We present DriveSuprim, a novel method that tackles these three critical challenges in selection-based trajectory prediction and makes more precise trajectory selection. Our solution replaces rigid binary classifications with probabilistic soft-label distillation, enabling nuanced decision-making in complex driving scenarios. Our contributions include:

- We propose a coarse-to-fine refinement method that addresses the challenge of distinguishing between similar trajectories. Our method filters promising candidates and applies fine-grained scoring to the most challenging options, significantly improving discrimination between similar but subtly different trajectories.
- We propose an integrated pipeline combining rotation-based data augmentation with self-distillation to address directional bias and hard decision boundaries. Our approach synthesizes challenging turning scenarios while leveraging teacher-generated soft pseudo-labels, effectively balancing trajectory distributions as shown in Fig 4 and realizing better optimization for training a more robust model.
- DriveSuprim achieves state-of-the-art performance on the NAVSIM benchmark purely based on the public dataset, demonstrating the effectiveness of our model in handling challenging driving scenarios. Our approach achieves state-of-the-art results on NAVSIMv1 and NAVSIMv2, significantly outperforming the previous methods by 3.6% and 1.5%.

## 2 Related Works

### 2.1 End-to-end planning

Autonomous driving has traditionally relied on modular pipelines that separate perception from planning. However, UniAD [^18] highlights several limitations of this approach, including information loss and error propagation. To address these challenges, end-to-end driving methods [^4] [^6] [^38] [^3] [^57] [^43] [^44] [^7] [^22] [^21] [^64] [^17] [^18] [^23] [^31] [^56] [^61] [^5] [^30] [^52] [^65] [^55] unify the perception-to-planning pipeline within a single optimizable network. Many of these models process raw sensor inputs and directly output driving trajectories. While some methods [^3] [^52] [^65] use reinforcement learning (RL) to learn through interaction with simulated environments, the majority adopt imitation learning (IL), training from expert demonstrations without environment interaction. Most IL-based approaches [^7] [^18] [^23] [^26] [^33] generate a single trajectory using regression or diffusion-based methods to mimic expert behavior.

More recently, selection-based methods [^23] [^5] [^30] [^27] [^55] have emerged. These models evaluate a diverse set of candidate trajectories by scoring them against safety-focused metrics (e.g., PDM scores [^9]). A prominent example is Hydra-MDP [^30], which won the recent NAVSIM challenge. It employs multiple rule-based teachers and distills them into the planner to create diverse trajectory candidates tailored to different evaluation metrics.

Our proposed model also falls under the selection-based paradigm. However, unlike prior methods that perform a single-shot selection from a fixed candidate set—often leading to suboptimal decisions—we introduce a coarse-to-fine selection and refinement strategy. This approach significantly improves selection precision by progressively narrowing down the trajectory set to the most optimal candidates.

### 2.2 Iterative & Multi-stage Refinement

Iterative refinement has been widely adopted to improve results in optical flow [^20] [^39] [^45] [^59] [^19] [^51] [^58], motion estimation [^47] [^13] [^66], and related tasks [^32] [^67] [^50] [^28]. A common strategy in these works is to iteratively propagate features or trajectory estimates through a shared module to progressively refine the predictions. Inspired by this, iterative refinement has also been applied in object detection [^68] [^2] to improve detection performance. For instance, in Deformable DETR, each decoder layer refines bounding boxes based on predictions from the previous layer.

In our work, we similarly adopt a multi-stage refinement strategy to improve trajectory selection accuracy. However, a key distinction lies in the mechanism of refinement: rather than repeatedly updating fixed-dimensional features, we implement selection from a fixed vocabulary of candidate trajectories. After selection, the search space is progressively narrowed to a more precise subset, improving the final prediction quality.

### 2.3 Augmentation for Enhanced Robustness

Robustness has long been a critical focus in computer vision research [^8] [^11] [^46] [^49] [^63] [^40]. Early studies demonstrated that image models are highly sensitive to minor domain shifts [^1], adversarial perturbations [^48] [^12], and common real-world corruptions such as brightness variations and fog [^37] [^15]. For example, MNIST-C [^36] introduced 15 distinct corruption types to benchmark model performance against diverse failure modes. Motivated by these insights, several methods [^41] [^35] [^42] [^24] utilize corruption-based augmentations, such as adding Gaussian and speckle noise, to enhance robustness. Inspired by this established research in image robustness, our study is the first to explore the use of similar corruption-based augmentation techniques specifically for end-to-end driving models. We introduce targeted perturbations and corruptions tailored to autonomous driving scenarios, addressing critical domain shifts—particularly the overrepresentation of straightforward driving trajectories—which pose challenges for scenarios involving complex maneuvers such as turns.

## 3 Methods

We introduce our method in this section. Firstly, we introduce some preliminaries, including the end-to-end planning and the selection-based planning method. Next, we introduce our proposed coarse-to-fine selection paradigm, rotation-based data augmentation method and self-distillation.

### 3.1 Preliminaries

#### End-to-End planning

In the autonomous driving domain, the end-to-end planning requires the planning system to output a future trajectory $T$ based on input sensor data, like RGB image or Lidar point cloud:

$$
T=\mathrm{Planner}\left(Img,Lidar\right),
$$

where the trajectory $T$ can be represented as a sequence of vehicle locations $(u_{1},u_{2},...,u_{l})$ or a sequence of controller actions $(a_{1},a_{2},...,a_{l})$, and $l$ denotes the sequence length.

#### Selection-based planning

The selection-based planning paradigm predefines a trajectory vocabulary $\{\tau_{i}\}_{i=1}^{N}$, which covers $N$ planning trajectories. Given a specific driving scenario, the quality of each trajectory can be measured by several metrics, like the $l_{2}$ distance to the human teacher trajectory, or metrics considering driving safety and traffic rule adherence. The selection-based planning paradigm learns a scorer that generates trajectory scores $\{s_{i}\}_{i=1}^{N}$ revealing trajectory quality. The trajectory with the highest score is chosen as the prediction result in inference:

$$
T=\tau_{k},\quad\text{where }k=\arg\max_{i}s_{i}
$$
![[x2 21.png|Refer to caption]]

Figure 2: Model architecture. DriveSuprim adopts a coarse-to-fine paradigm to better distinguish hard negatives. According to the scoring distribution, the Trajectory Decoder filters potential candidates, and the Refinement Decoder further outputs fine-grained trajectory scores. The model introduces rotation-based augmented data to ease the directional bias and applies a self-distillation framework for stable training. The teacher outputs serve as soft labels for auxiliary supervision for the student.

### 3.2 Coarse-to-Fine Trajectory selection

DriveSuprim proposes a coarse-to-fine trajectory selection paradigm comprising coarse filtering and fine-grained scoring, improving model capability on distinguishing hard negative trajectories, as shown in Fig 2. In the coarse filtering stage, the model selects a series of trajectory candidates based on the predicted scores, similar to classic selection-based approaches. The fine-grained scoring stage then produces more accurate scores for the filtered trajectories.

#### Coarse filtering

The coarse filtering stage scores all trajectories in the vocabulary and filters a smaller set of candidates for the next stage. Here we apply the same strategy as the previous selection-based method [^30]. Each trajectory is encoded into a vector by a lightweight MLP encoder, and then the trajectory feature cross-attends with the input image feature to extract the planning-related information. To measure the quality of each trajectory in a specific scenario, several prediction heads are applied on the refined trajectory feature to regress the $l_{2}$ distance to the ground-truth human trajectory and the rule-based metric scores:

$$
\displaystyle\mathcal{E}_{\mathrm{img}}=\mathrm{Enc_{i}}(I),\quad f_{j}=%
\mathrm{Enc_{t}}(\tau_{j}),\quad g_{j}=\mathrm{TransDec}(\mathcal{E}_{\mathrm{%
img}},f_{j})
$$
 
$$
\displaystyle s_{j}^{(m)}=\mathrm{Sigmoid}\left(\mathrm{head^{(m)}}(g_{j})%
\right),
$$

where $\mathrm{Enc_{i}}$ and $\mathrm{Enc_{t}}$ are image encoder and trajectory encoder, $I$ and $\mathcal{E}_{\mathrm{img}}$ are the input image and image feature, $\tau_{j}$ and $f_{j}$ denote the trajectory and the encoded trajectory feature, $\mathrm{TransDec}$ denotes the Trajectory Decoder, which is a Transformer decoder [^53], $g_{j}$ denotes the refined trajectory feature, $\mathrm{head^{(m)}}$ denotes the prediction head of metric $m$, and $s_{j}^{(m)}$ denotes the score of trajectory $\tau_{j}$ on the evalutation metric $m$.

At the end of the coarse filtering stage, each trajectory $\tau_{j}$ corresponds to a score $s_{j}$ revealing its quality on end-to-end planning. We select the trajectories with top-k scores as the filtered trajectories $T_{\mathrm{filter}}=\{\tau_{j}\mid j\in\mathcal{I}_{\text{top}k}\}$, where $\mathcal{I}_{\text{top}k}=\operatorname{argsort}_{k}(\{s_{j}\}_{j=1}^{N})$, and the refined features $G_{\mathrm{filter}}=\{g_{j}\mid j\in\mathcal{I}_{\text{top}k}\}$ are utilized for fine-grained fitting.

#### Fine-grained scoring

In this stage, a Transformer decoder similar to the first stage is applied to make fine-grained scoring and further distinguish the trajectories in the last-stage filtered candidates.

Specifically, we obtain the refined score from the $l$ -th decoder layer output feature, and optimize with the ground truth trajectory score:

$$
\displaystyle\left\{h_{j,l}\right\}_{l=1}^{n_{\mathrm{ref}}}
$$
 
$$
\displaystyle=\mathrm{RefineDec}\left(\mathcal{E}_{\mathrm{img}},g_{j}\right)
$$
 
$$
\displaystyle s_{j,l}^{(m)}
$$
 
$$
\displaystyle=\mathrm{Sigmoid}\left(\mathrm{head^{(m)}}\left(h_{j,l}\right)\right)
$$

where $\mathrm{RefineDec}$ is a Transformer decoder with $n_{\mathrm{ref}}$ layers, $h_{j,l}$ is the refined feature output from the $l$ -th layer of the Refinement Transformer, $s_{j,l}^{(m)}$ denotes the score of $\tau_{j}$ on metric $m$ output by the $l$ -th decoder layer. The output trajectory $T$ is selected based on the last-layer output score $s_{j,n_{\mathrm{ref}}}^{(m)}$. This method can ease the difficulties of selecting the best trajectory from the top-k trajectories containing a large proportion of hard negative candidates.

The loss for our coarse-to-fine module on the original dataset is represented as:

$$
L_{\mathrm{ori}}=L_{\mathrm{coarse}}+L_{\mathrm{refine}},
$$

where $L_{\mathrm{coarse}}$ and $L_{\mathrm{refine}}$ respectively train the coarse filtering stage and the fine-grained scoring stage. $L_{\mathrm{coarse}}$ consists of an imitation loss $L_{\mathrm{imi}}$ [^30] [^27] and a binary cross-entropy between the predicted trajectory score and the ground truth metric score:

$$
L_{\mathrm{coarse}}=L_{\mathrm{imi}}+\sum_{m,i}\mathrm{BCE}(s_{i}^{(m)},y_{i}^%
{(m)}),
$$

where $y_{i}^{(m)}$ is the ground-truth metric score of each trajectory $\tau_{i}$ on metric $m$, and $s_{i}^{(m)}$ is its corresponding predicted score. The fine-grained scoring stage loss $L_{\mathrm{refine}}$ is similar to $L_{\mathrm{coarse}}$, while it only considers the filtered trajectories $T_{\mathrm{filter}}$ rather than the entire vocabulary, and is applied on each decoder layer output.

![[x3 21.png|Refer to caption]]

Figure 3: Rotation demonstration for input image in a 1-camera FOV setting. Our rotation-based data augmentation method generates the rotated image through horizontal shifting and cropping.

### 3.3 Rotation-based data augmentation

To mitigate the data imbalance, our model introduces an end-to-end rotation-based augmentation pipeline, where a 2D horizontal view transformation is applied to the sensor input data to simulate the ego-vehicle rotation in the 3D space. This approach conveniently synthesizes more challenging scenarios and diversifies the driving scenarios, enabling the model to learn to output accurate trajectories regardless of the vehicle orientation.

As shown in Fig 3, for each scenario, we sample a random angle $\theta$ from a uniform distribution $U[-\Theta,\Theta]$, where $\Theta$ is the angle boundary. The positive angle indicates the leftward rotation of the ego vehicle. To prepare the input RGB image, camera images corresponding to the original field of view (FOV) along with images from two extended views are concatenated to simulate a “pseudo panoramic view”, then the input image is cropped from the concatenated image according to a shifting window based on $\theta$. For example, for a 1-camera FOV input setting, we choose these three cameras as our input: $f$ (front camera), $l_{0}$ (front-left camera), and $r_{0}$ (front-right camera), where $f$ corresponds to the original field of view, and $l_{0}$ and $r_{0}$ correspond to the extended view.

The ground truth human trajectory $(u_{1},u_{2},\dots,u_{l})$ in the rotated scenario is generated by a trivial rotation approach. Under a specific rotation angle $\theta$, each location $u_{j}$ of the human trajectory $T_{h}$ applies a 2D-rotation transformation surrounding the initial vehicle position $u_{0}$, and the rotation angle is $-\theta$. Given augmented camera views and the prediction of our model, we calculate a loss $L_{\mathrm{aug}}$ for training, which has the same formulation as $L_{\mathrm{ori}}$.

### 3.4 Self-distillation with Soft-labeling

Instead of training the selection-based model to fit a hard decision boundary given limited environmental contexts (e.g., insufficient fields of view), which can impair training stability, we propose a self-distillation framework with teacher-generated soft labels to stabilize training. The self-distillation framework consists of a teacher and a student model, both sharing the identical architecture. The student parameters are updated via standard gradient descent, whereas the teacher parameters are updated using an exponential moving average (EMA) of the student’s parameters.

During training, the student receives both original and augmented data to calculate $L_{\mathrm{ori}}$ and $L_{\mathrm{aug}}$, as shown in Fig 2. The teacher only receives original data to generate scores served as soft labels, where a threshold $\delta_{m}$ is introduced to control the difference between the teacher output and the ground-truth:

$$
\displaystyle\hat{y}_{i}^{(m)}=y_{i}^{(m)}+\mathrm{clip}\left(s_{i,\mathrm{%
teacher}}^{(m)}-y_{i}^{(m)},-\delta_{m},\delta_{m}\right)
$$
 
$$
\displaystyle L_{\mathrm{soft}}=L_{\mathrm{imi-soft}}+\sum_{m,i}{\mathrm{BCE}(%
s_{i}^{(m)},\hat{y}_{i}^{(m)})},
$$

where $s_{i,\mathrm{teacher}}^{(m)}$ is the score of trajectory $i$ on metric $m$ predicted by the teacher, $y_{i}^{(m)}$ is the ground-truth score of trajectory $i$ on metric $m$, $\mathrm{clip}\left(\cdot,\alpha,\beta\right)$ denotes clipping the input value to the interval $[\alpha,\beta]$, and $\hat{y}_{i}^{(m)}$ is the soft label, and $L_{\mathrm{imi-soft}}$ is a soft version of $L_{\mathrm{imi}}$. We shift the human trajectory toward the teacher model’s output trajectory for at most 1 meter, and adopt this new trajectory to calculate the imitation loss $L_{\mathrm{imi-soft}}$. The overall training loss of the student model is represented as:

$$
L=L_{\mathrm{ori}}+L_{\mathrm{aug}}+L_{\mathrm{soft}}
$$

During inference, the teacher model is utilized to output results.

## 4 Experiments

In this section, we first introduce our implementation details. Next, we show the superior performance of DriveSuprim on NAVSIM v1 and NAVSIM v2. Results of ablation studies are listed to validate the effectiveness of the proposed modules. Finally, we produce some visualization results to intuitively show the advantages of our method.

### 4.1 Implementation Details

#### Dataset and Metrics

We conduct experiments mainly on the NAVSIM [^9] benchmark, which is a driving dataset containing challenging driving scenarios.

There are two different evaluation metrics on NAVSIM, leading to NAVSIM v1 and NAVSIM v2. The evaluation metric of NAVSIM v1 is the PDM Score (PDMS). Each predicted trajectory is sent to a simulator to collect different rule-based subscores, which are aggregated to get the final PDMS:

$$
\mathrm{PDMS}=\left(\prod_{m\in S_{P}}{\mathrm{score}_{m}}\right)\times\left(%
\frac{\sum_{w\in S_{A}}{\mathrm{weight}_{w}\times\mathrm{score}_{w}}}{\sum_{w%
\in S_{A}}{\mathrm{weight}_{w}}}\right),
$$

where $S_{P}$ and $S_{A}$ denote the penalty subscore set and the weighted average subscore set. In NAVSIM v1, $S_{P}$ comprises two subscores, including no collisions (NC) and drivable area compliance (DAC), and $S_{A}$ comprises ego progress (EP), time-to-collision (TTC), and comfort (C). NAVSIM v2 introduces 4 more subscores, including driving direction compliance (DDC) and traffic light compliance (TLC) in $S_{P}$, and lane keeping (LK) and extended comfort (EC) in $S_{A}$. The comfort subscore is also revised to become the history comfort (HC). Aggregating all these subscores leads to the $\mathrm{EPDMS}$ metric in NAVSIM v2.

#### Model Details

We conduct our methods on three different backbones as the image encoder $\mathrm{Enc}_{i}$, including ResNet34 [^14], VoVNet [^25], and ViT-Large [^10]. The input images are resized to a resolution of $2048\times 512$ for ResNet34 and VoVNet, and $1024\times 256$ for Vit-Large. The ViT-Large backbone is pre-trained through Depth Anything [^60]. A 2-layer MLP is leveraged as $\mathrm{Enc}_{t}$ to encode each trajectory in the vocabulary. The trajectory decoder $\mathrm{TransDec}$ and refinement decoder $\mathrm{RefineDec}$ are both 3-layer Transformer Decoders with 256 hidden dimensions. The prediction head $\mathrm{head}^{(m)}$ predicts the normalized $l_{2}$ distance to the human ground-truth trajectory, and each subscore in NAVSIM except for EC. We adopt a 2-layer MLP for the distance prediction, and a linear layer for predicting other metrics. The number of trajectories in the vocabulary and filtered trajectories of the coarse filtering stage are set to 8192 and 256. We adopt a 3-camera FOV setting, with images from $l_{0}$, $f$, and $r_{0}$ cameras as the input. The rotation angle boundary $\Theta$ is set to $\pi/6$ in the augmentation pipeline. The threshold $\delta_{m}$ in soft-labeling is 0.15.

Table 2: Evaluation on NAVSIM v1. Results are grouped by backbone types.

| Method | Backbone | NC $\uparrow$ | DAC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | C $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Human Agent | — | 100 | 100 | 87.5 | 100 | 99.9 | 94.8 |
| Transfuser [^7] | ResNet34 | 97.7 | 92.8 | 79.2 | 92.8 | 100 | 84.0 |
| UniAD [^18] | ResNet34 | 97.8 | 91.9 | 78.8 | 92.9 | 100 | 83.4 |
| PARA-Drive [^56] | ResNet34 | 97.9 | 92.4 | 79.3 | 93.0 | 99.8 | 84.0 |
| VADv2 [^5] | ResNet34 | 97.9 | 91.7 | 77.6 | 92.9 | 100 | 83.0 |
| LAW [^29] | ResNet34 | 96.4 | 95.4 | 81.7 | 88.7 | 99.9 | 84.6 |
| DRAMA [^62] | ResNet34 | 98.0 | 93.1 | 80.1 | 94.8 | 100 | 85.5 |
| Hydra-MDP [^30] | ResNet34 | 98.3 | 96.0 | 78.7 | 94.6 | 100 | 86.5 |
| DiffusionDrive [^34] | ResNet34 | 98.2 | 96.2 | 82.2 | 94.7 | 100 | 88.1 |
| DriveSuprim | ResNet34 | 97.8 | 97.3 | 86.7 | 93.6 | 100 | 89.9 (+1.8) |
| Hydra-MDP [^30] | V2-99 | 98.4 | 97.8 | 86.5 | 93.9 | 100 | 90.3 |
| DriveSuprim | V2-99 | 98.0 | 98.2 | 90.0 | 94.2 | 100 | 92.1 (+1.8) |
| Hydra-MDP [^30] | ViT-L | 98.4 | 97.7 | 85.0 | 94.5 | 100 | 89.9 |
| DriveSuprim | ViT-L | 98.6 | 98.6 | 91.3 | 95.5 | 100 | 93.5 (+3.6) |

#### Training Details

We train our model on 8 NVIDIA A100. We adopt the Adam optimizer for model training, the batch size on a GPU is 8, and the learning rate is set to $7.5\times 10^{-5}$. More training details about the training pipeline are shown in appendix A.

### 4.2 Main Results

#### Result on NAVSIM v1

We evaluate the performance of DriveSuprim on the NAVSIM v1 benchmark. As shown in Tab 2, our method reaches 89.9% PDMS with the ResNet34 backbone, surpassing DiffusionDrive by 1.8%. Moreover, DriveSuprim with the stronger Vit-Large backbone can reach 93.5% PDMS, which is close to the human agent.

#### Result on NAVSIM v2

Table 3 shows that our model can also reach the SOTA result on the more challenging NAVSIM v2 benchmark. On the EPDMS metric, DriveSuprim surpasses previous SOTA methods by 1.7%, 0.9%, and 1.5%, respectively.

Table 3: Evaluation on NAVSIM v2. Results are grouped by backbone types.

| Method | Backbone | NC $\uparrow$ | DAC $\uparrow$ | DDC $\uparrow$ | TL $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | LK $\uparrow$ | HC $\uparrow$ | EC $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Human Agent | — | 100 | 100 | 99.8 | 100 | 87.4 | 100 | 100 | 98.1 | 90.1 | 90.3 |
| Ego Status MLP | — | 93.1 | 77.9 | 92.7 | 99.6 | 86.0 | 91.5 | 89.4 | 98.3 | 85.4 | 64.0 |
| Transfuser [^7] | ResNet34 | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| HydraMDP++ [^27] | ResNet34 | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| DriveSuprim | ResNet34 | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 (+1.7) |
| HydraMDP++ [^27] | V2-99 | 98.4 | 98.0 | 99.4 | 99.8 | 87.5 | 97.7 | 95.3 | 98.3 | 77.4 | 85.1 |
| DriveSuprim | V2-99 | 97.8 | 97.9 | 99.5 | 99.9 | 90.6 | 97.1 | 96.6 | 98.3 | 77.9 | 86.0 (+0.9) |
| HydraMDP++ [^27] | ViT-L | 98.5 | 98.5 | 99.5 | 99.7 | 87.4 | 97.9 | 95.8 | 98.2 | 75.7 | 85.6 |
| DriveSuprim | ViT-L | 98.4 | 98.6 | 99.6 | 99.8 | 90.5 | 97.8 | 97.0 | 98.3 | 78.6 | 87.1 (+1.5) |

Table 4: Ablation study on different proposed modules. “Multi-stage” denotes using coarse-to-fine selection, “Aug Data” denotes introducing rotation-based augmentation data, and “Self-distill” denotes adopting self-distillation.

| Multi- stage | Aug Data | Self- distill | NC $\uparrow$ | DAC $\uparrow$ | DDC $\uparrow$ | TL $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | LK $\uparrow$ | HC $\uparrow$ | EC $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ✓ | ✓ | ✓ | 98.4 | 98.6 | 99.6 | 99.8 | 90.5 | 97.8 | 97.0 | 98.3 | 78.6 | 87.1 |
| ✗ | ✓ | ✓ | 98.7 | 98.5 | 99.5 | 99.8 | 87.5 | 98.3 | 96.0 | 98.3 | 71.1 | 85.6 (-1.5) |
| ✓ | ✗ | ✓ | 98.2 | 98.3 | 99.5 | 99.8 | 90.3 | 97.6 | 96.4 | 98.3 | 77.9 | 86.4 (-0.7) |
| ✓ | ✓ | ✗ | 97.5 | 98.3 | 99.5 | 99.8 | 90.7 | 96.8 | 96.4 | 98.3 | 75.4 | 85.6 (-1.5) |

Table 5: Ablation study of the evolution from single-stage selection to coarse-to-fine selection.

|  | NC $\uparrow$ | DAC $\uparrow$ | DDC $\uparrow$ | TL $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | LK $\uparrow$ | HC $\uparrow$ | EC $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Single-stage | 98.5 | 98.5 | 99.5 | 99.7 | 87.4 | 97.9 | 95.8 | 98.2 | 75.7 | 85.6 |
| \+ 6 layer decoder | 98.3 | 98.3 | 99.4 | 99.8 | 87.5 | 97.8 | 95.2 | 98.3 | 75.9 | 85.3 (-0.3) |
| \+ Layer-wise scoring | 98.7 | 98.4 | 99.4 | 99.9 | 87.2 | 98.1 | 95.5 | 98.2 | 75.6 | 85.6 (+0.3) |
| \+ Trajectory filtering | 98.6 | 98.2 | 99.5 | 99.9 | 89.3 | 98.0 | 97.0 | 98.3 | 76.2 | 86.4 (+0.8) |

Table 6: Ablation on refinement setting. “Stage Layer” is the layer number of $\mathrm{RefineDec}$, and “Top-K” denotes the number of trajectories selected by the coarse filtering stage.

| Stage Layer | Top-K | EPDMS $\uparrow$ |
| --- | --- | --- |
| 1 | 256 | 86.6 |
| 3 | 256 | 87.1 |
| 6 | 256 | 86.6 |
| 3 | 64 | 86.5 |
| 3 | 512 | 86.9 |
| 3 | 1024 | 86.4 |

Table 7: Results with different soft label thresholds. $\delta_{m}$ denotes the soft label threshold in Equ 9.

| $\delta_{m}$ | EPDMS $\uparrow$ |
| --- | --- |
| 0.00 | 86.8 |
| 0.15 | 87.1 |
| 0.30 | 86.8 |
| 0.70 | 86.5 |
| 1.00 | 86.6 |

### 4.3 Ablation Studies

#### Ablation on different modules

We conduct ablation studies on the modules and approaches we propose, the result is shown in Tab 4. Adopting multi-stage refinement leads to a remarkable 1.5% performance gain on PDMS, and the augmented data improves the EPDMS by 0.7%. Applying the self-distillation framework is also beneficial to the training, with a 1.5% EPDMS improvement.

#### Effectiveness of coarse-to-fine selection

To thoroughly validate the effectiveness of our coarse-to-fine trajectory selection paradigm, we devise a series of model evolutions from single-stage selection to coarse-to-fine selection, to eliminate the effect of parameter numbers in the model decoder. The results are listed in Tab 5. Firstly, we expand the number of trajectory decoder layers from 3 to 6, aligned with our coarse-to-fine setting, then we adopt the layer-wise scoring to the decoder. Nevertheless, these two revisions don’t lead to performance improvement. Only after we decrease the number of trajectories in the refinement stage does the performance increase from 85.6% to 86.4%.

#### Ablation on refinement settings and soft label threshold

We further conduct comprehensive ablation studies on the refinement approach and soft-labeling. the results are shown in Tab 7 and Tab 7. Utilizing a 3-layer refinement Transformer decoder and filtering 256 trajectories for the refinement stage leads to the best result. For the teacher soft label, choosing 0.15 as the threshold leads to the best EPDM score.

#### Comparison of different FOVs

Our model performance on different FOV settings is shown in appendix C.1. Utilizing images from 3 camera sensors leads to the best result.

### 4.4 Visualization and Analysis

#### Superior performance on turning scenarios

Tab. 8 demonstrates the superior performance of DriveSuprim in turning scenarios. We divide the test dataset into three subsets based on the turning angle of the ground-truth trajectories: $\mathrm{NAVTEST}_{\mathrm{left}}$ (left turns exceeding 30 degrees), $\mathrm{NAVTEST}_{\mathrm{forward}}$ (near-straight trajectories), and $\mathrm{NAVTEST}_{\mathrm{right}}$ (right turns exceeding 30 degrees). The performance improvements are more pronounced in turning scenarios than in near-straight ones, highlighting DriveSuprim ’s enhanced ability to handle turning maneuvers.

Table 8: EPDMS on three dataset splits. $\mathrm{NAVTEST}_{\mathrm{left}}$, $\mathrm{NAVTEST}_{\mathrm{forward}}$, and $\mathrm{NAVTEST}_{\mathrm{right}}$ involve left-turning scenarios, near-forward scenarios, and right-turning scenarios.

| Method | $\mathrm{NAVTEST}_{\mathrm{left}}$ | $\mathrm{NAVTEST}_{\mathrm{forward}}$ | $\mathrm{NAVTEST}_{\mathrm{right}}$ |
| --- | --- | --- | --- |
| Hydra-MDP++ [^27] | 68.7 | 87.2 | 77.7 |
| DriveSuprim | 71.6 (+2.9) | 88.1 (+0.9) | 79.7 (+2.0) |

![[trajectories_ori_vs_rotated.png|Refer to caption]]

Figure 4: Comparison of dataset high-score trajectory distribution. Our augmentation mitigates directional bias, significantly increasing the frequency of previously underrepresented turning trajectories. The color bar represents the normalized frequency of different trajectories in the dataset.

#### Trajectory distribution comparison

Fig. 4 illustrates the trajectory frequency distribution in both the original dataset and the dataset enhanced with our rotation-based augmentation method. Using the ground-truth PDM score, we count the trajectory that either scores above 0.99 or ranks among the top-three-highest scores in each scenario. We then normalize the frequency by setting the most frequent trajectory to 1. Results show that in the original dataset, trajectories are predominantly concentrated in the forward or near-forward direction, whereas in the augmented dataset, trajectories across all directions appear with similar frequency.

#### Visulization results

The visualization results are shown in appendix B. This qualitative result shows that DriveSuprim not only performs well in complex challenging scenarios, but also handles sharp turn scenarios perfectly.

## 5 Conclusion

In this work, we present DriveSuprim, a novel framework for end-to-end autonomous driving that addresses three key challenges of selection-based methods. By introducing a coarse-to-fine selection strategy and an integrated training pipeline including rotation-based data augmentation and soft-label self-distillation, DriveSuprim significantly enhances the model’s ability to distinguish hard negatives to make more nuanced decisions to precisely select trajectories, and performs well in scenarios involving sharp turns. Extensive experiments on the NAVSIM benchmark demonstrate that our approach outperforms prior methods by a substantial margin.

## References

## Appendix A Training Details and Inference Pipeline

We adopt two different training pipelines for different backbones. For the models loading the pre-trained backbones, including VoVNet and Vit-Large, we train our model for 6 epochs. The EMA momentum $m$ is increased linearly from 0.992 to 0.996 in the first 3 epochs, and then fixed to 0.998. The model with the ResNet-34 backbone trained from scratch is trained for 10 epochs. The EMA update is applied after the training for 3 epochs, which means that $m$ is set to 0 in the first 3 epochs, then increased from 0.992 to 0.996 in the subsequent 3 epochs, then fixed to 0.998.

During inference, we compute a final score for each trajectory by linearly combining the predicted scores across different metrics. We select the filtered and final trajectory based on the final score. Specifically, for the imitation learning metric and the multiplier metrics introduced in NAVSIM [^9], we apply a coefficient to the logarithm of the predicted score. For the weighted average metric in NAVSIM, the coefficient is applied to the predicted score, followed by an additional logarithmic process of the sum. The overall inference score can be formulated as:

$$
s_{i}=\sum_{m}{\lambda^{(m)}\log{s_{i}^{(m)}}}+\lambda_{\mathrm{avg}}\log\left%
({\sum_{n}{\lambda^{(n)}s_{i}^{(n)}}}\right),
$$

where $m$ denotes the imitation metric and the multiplier metric, $n$ denotes the weighted average metric, $\lambda^{(m)}$ and $\lambda^{(n)}$ denote the coefficient, and $s_{i}$ denotes the final combined prediction score. $\lambda_{\mathrm{avg}}$ is set to 8.0 and 6.0 in NAVSIM v1 and v2, respectively. The detailed coefficients used during inference on NAVSIM v1 and v2 are shown in Tab 9 and Tab 10.

Table 9: The inference coefficients on each metric of NAVSIM v1. “Imi” denotes the imitation metric, “Mul” denotes the multiplied penalties, and “Avg” denotes the weighted averages.

<table><tbody><tr><td></td><td rowspan="2">Imi</td><td colspan="2">Mul</td><td colspan="3">Avg</td></tr><tr><td></td><td>NC</td><td>DAC</td><td>EP</td><td>TTC</td><td>C</td></tr><tr><td>coefficient</td><td>0.05</td><td>0.5</td><td>0.5</td><td>5.0</td><td>5.0</td><td>2.0</td></tr></tbody></table>

Table 10: The inference coefficients on each metric of NAVSIM v2. “Imi” denotes the imitation metric, “Mul” denotes the multiplied penalties, and “Avg” denotes the weighted averages.

<table><tbody><tr><td></td><td rowspan="2">Imi</td><td colspan="4">Mul</td><td colspan="4">Avg</td></tr><tr><td></td><td>NC</td><td>DAC</td><td>DDC</td><td>TL</td><td>EP</td><td>TTC</td><td>LK</td><td>HC</td></tr><tr><td>coefficient</td><td>0.02</td><td>0.5</td><td>0.5</td><td>0.3</td><td>0.1</td><td>5.0</td><td>5.0</td><td>2.0</td><td>1.0</td></tr></tbody></table>

## Appendix B Visualization Results

Fig 5 shows qualitative visualization results comparing our method with the selection-based approach Hydra-MDP++ [^27]. The subimages in the first row are the results of Hydra-MDP++, and the second row is our result.

The subimages in the first column show a challenging driving scenario where the ego vehicle is preparing to overtake a leading vehicle just before a crossroad. DriveSuprim successfully executes the overtaking maneuver, whereas the classic selection-based method incorrectly does a left turn, potentially leading to a collision. The subimages in the second column demonstrate DriveSuprim’s ability to distinguish between trajectories that appear right at first glance, i.e., the “hard negatives”. Although both models generate similar trajectories, the trajectory produced by Hydra-MDP++ deviates from the road centerline at the endpoint, while our method generates a near-perfect trajectory that closely matches the human expert’s ground truth. The third and fourth columns highlight the superior performance of our model in sharp-turning scenarios. DriveSuprim consistently generates smooth and accurate turning trajectories even under large turning angles.

These qualitative results demonstrate that DriveSuprim not only performs well in complex and challenging scenarios by generating precise trajectories, but also excels in handling sharp turns with high accuracy.

![[x4 20.png|Refer to caption]]

Figure 5: Visualization results across various challenging scenarios. The first row shows the results of Hydra-MDP++, while the second row presents our method. In each example, the green trajectory represents the ground truth from the human expert, the red trajectory is generated by the classic selection-based method, and the blue trajectory is produced by DriveSuprim. Zoom in for a better view.

## Appendix C Supplementary Experiment Result

### C.1 Model Performance with Different FOV Settings

We show the model performance with different FOV settings in Tab 11. The FOV is revealed by the number of cameras. As shown in Fig 6, five cameras are involved in these settings: $f$ (front camera), $l_{0}$ (front-left camera), $l_{1}$ (left camera), $r_{0}$ (front-right camera), and $r_{1}$ (right camera). The 1-camera setting uses $f_{0}$ as input, the 3-camera setting uses $l_{0}$, $f_{0}$ and $r_{0}$ as input, while the 5-camera setting uses $l_{1}$, $l_{0}$, $f_{0}$, $r_{0}$ and $r_{1}$ as input. Experimental results show that the 3-camera setting leads to the best performance.

![[x5 20.png|Refer to caption]]

Figure 6: Images from the five cameras.

Table 11: Results with different FOVs (number of cameras).

| Camera Number | NC $\uparrow$ | DAC $\uparrow$ | DDC $\uparrow$ | TL $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | LK $\uparrow$ | HC $\uparrow$ | EC $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 Camera | 97.5 | 97.8 | 99.4 | 99.8 | 91.2 | 96.7 | 96.9 | 98.3 | 76.2 | 85.4 |
| 3 Cameras | 98.4 | 98.6 | 99.6 | 99.8 | 90.5 | 97.8 | 97.0 | 98.3 | 78.6 | 87.1 |
| 5 Cameras | 98.4 | 98.4 | 99.6 | 99.8 | 90.5 | 97.8 | 97.0 | 98.3 | 78.4 | 86.9 |

## Appendix D Limitations and Future Work

In this paper, we evaluate the performance of DriveSuprim on the challenging NAVSIM benchmark. Our two-stage coarse-to-fine trajectory filtering approach proves effective in enhancing model performance. However, extending this filtering strategy to a multi-stage setting does not yield additional improvements, and the inference speed remains suboptimal.

In future work, we will further investigate multi-stage refinement techniques and develop more efficient planning methods.

[^1]: Aharon Azulay and Yair Weiss. Why do deep convolutional networks generalize so poorly to small image transformations? Journal of Machine Learning Research, 20(184):1–25, 2019.

[^2]: Zhaowei Cai and Nuno Vasconcelos. Cascade r-cnn: Delving into high quality object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6154–6162, 2018.

[^3]: Dian Chen, Vladlen Koltun, and Philipp Krähenbühl. Learning to drive from a world on rails. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15590–15599, 2021.

[^4]: Dian Chen, Brady Zhou, Vladlen Koltun, and Philipp Krähenbühl. Learning by cheating. In Conference on Robot Learning, pages 66–75. PMLR, 2020.

[^5]: Shaoyu Chen, Bo Jiang, Hao Gao, Bencheng Liao, Qing Xu, Qian Zhang, Chang Huang, Wenyu Liu, and Xinggang Wang. Vadv2: End-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243, 2024.

[^6]: Kashyap Chitta, Aditya Prakash, and Andreas Geiger. Neat: Neural attention fields for end-to-end autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15793–15803, 2021.

[^7]: Kashyap Chitta, Aditya Prakash, Bernhard Jaeger, Zehao Yu, Katrin Renz, and Andreas Geiger. Transfuser: Imitation with transformer-based sensor fusion for autonomous driving. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(11):12878–12895, 2022.

[^8]: Francesco Croce, Maksym Andriushchenko, Vikash Sehwag, Edoardo Debenedetti, Nicolas Flammarion, Mung Chiang, Prateek Mittal, and Matthias Hein. Robustbench: a standardized adversarial robustness benchmark. arXiv preprint arXiv:2010.09670, 2020.

[^9]: Daniel Dauner, Marcel Hallgarten, Tianyu Li, Xinshuo Weng, Zhiyu Huang, Zetong Yang, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, et al. Navsim: Data-driven non-reactive autonomous vehicle simulation and benchmarking. arXiv preprint arXiv:2406.15349, 2024.

[^10]: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.

[^11]: Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario March, and Victor Lempitsky. Domain-adversarial training of neural networks. Journal of machine learning research, 17(59):1–35, 2016.

[^12]: Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.

[^13]: Adam W Harley, Zhaoyuan Fang, and Katerina Fragkiadaki. Particle video revisited: Tracking through occlusions using point trajectories. In European Conference on Computer Vision, pages 59–75. Springer, 2022.

[^14]: Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016.

[^15]: Dan Hendrycks and Thomas G Dietterich. Benchmarking neural network robustness to common corruptions and surface variations. arXiv preprint arXiv:1807.01697, 2018.

[^16]: Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. Gaia-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023.

[^17]: Shengchao Hu, Li Chen, Penghao Wu, Hongyang Li, Junchi Yan, and Dacheng Tao. St-p3: End-to-end vision-based autonomous driving via spatial-temporal feature learning. In European Conference on Computer Vision, pages 533–549. Springer, 2022.

[^18]: Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, et al. Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17853–17862, 2023.

[^19]: Tak-Wai Hui, Xiaoou Tang, and Chen Change Loy. Liteflownet: A lightweight convolutional neural network for optical flow estimation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 8981–8989, 2018.

[^20]: Eddy Ilg, Nikolaus Mayer, Tonmoy Saikia, Margret Keuper, Alexey Dosovitskiy, and Thomas Brox. Flownet 2.0: Evolution of optical flow estimation with deep networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2462–2470, 2017.

[^21]: Bernhard Jaeger, Kashyap Chitta, and Andreas Geiger. Hidden biases of end-to-end driving models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8240–8249, 2023.

[^22]: Xiaosong Jia, Penghao Wu, Li Chen, Jiangwei Xie, Conghui He, Junchi Yan, and Hongyang Li. Think twice before driving: Towards scalable decoders for end-to-end autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21983–21994, 2023.

[^23]: Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, and Xinggang Wang. Vad: Vectorized scene representation for efficient autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8340–8350, 2023.

[^24]: Oğuzhan Fatih Kar, Teresa Yeo, Andrei Atanov, and Amir Zamir. 3d common corruptions and data augmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18963–18974, 2022.

[^25]: Youngwan Lee, Joong-won Hwang, Sangrok Lee, Yuseok Bae, and Jongyoul Park. An energy and gpu-computation efficient backbone network for real-time object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2019.

[^26]: Derun Li, Jianwei Ren, Yue Wang, Xin Wen, Pengxiang Li, Leimeng Xu, Kun Zhan, Zhongpu Xia, Peng Jia, Xianpeng Lang, et al. Finetuning generative trajectory model with reinforcement learning from human feedback. arXiv preprint arXiv:2503.10434, 2025.

[^27]: Kailin Li, Zhenxin Li, Shiyi Lan, Jiayi Liu, Yuan Xie, zhizhong zhang, Zuxuan Wu, Zhiding Yu, and Jose M. Alvarez. Hydra-MDP++: Advancing end-to-end driving via hydra-distillation with expert-guided decision analysis, 2024.

[^28]: Xia Li, Jianlong Wu, Zhouchen Lin, Hong Liu, and Hongbin Zha. Recurrent squeeze-and-excitation context aggregation net for single image deraining. In Proceedings of the European conference on computer vision (ECCV), pages 254–269, 2018.

[^29]: Yingyan Li, Lue Fan, Jiawei He, Yuqi Wang, Yuntao Chen, Zhaoxiang Zhang, and Tieniu Tan. Enhancing end-to-end autonomous driving with latent world model. arXiv preprint arXiv:2406.08481, 2024.

[^30]: Zhenxin Li, Kailin Li, Shihao Wang, Shiyi Lan, Zhiding Yu, Yishen Ji, Zhiqi Li, Ziyue Zhu, Jan Kautz, Zuxuan Wu, et al. Hydra-mdp: End-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978, 2024.

[^31]: Zhiqi Li, Zhiding Yu, Shiyi Lan, Jiahan Li, Jan Kautz, Tong Lu, and Jose M Alvarez. Is ego status all you need for open-loop end-to-end autonomous driving? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14864–14873, 2024.

[^32]: Zhengfa Liang, Yiliu Feng, Yulan Guo, Hengzhu Liu, Wei Chen, Linbo Qiao, Li Zhou, and Jianfeng Zhang. Learning for disparity estimation through feature constancy. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2811–2820, 2018.

[^33]: Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang, Qian Zhang, et al. Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving. arXiv preprint arXiv:2411.15139, 2024.

[^34]: Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang, Qian Zhang, and Xinggang Wang. Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving. arXiv preprint arXiv:2411.15139, 2024.

[^35]: Eric Mintun, Alexander Kirillov, and Saining Xie. On interaction between augmentations and corruptions in natural corruption robustness. Advances in Neural Information Processing Systems, 34:3571–3583, 2021.

[^36]: Norman Mu and Justin Gilmer. Mnist-c: A robustness benchmark for computer vision. arXiv preprint arXiv:1906.02337, 2019.

[^37]: Kexin Pei, Linjie Zhu, Yinzhi Cao, Junfeng Yang, Carl Vondrick, and Suman Jana. Towards practical verification of machine learning: The case of computer vision systems. arXiv preprint arXiv:1712.01785, 2017.

[^38]: Aditya Prakash, Kashyap Chitta, and Andreas Geiger. Multi-modal fusion transformer for end-to-end autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7077–7087, 2021.

[^39]: Anurag Ranjan and Michael J Black. Optical flow estimation using a spatial pyramid network. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4161–4170, 2017.

[^40]: Sylvestre-Alvise Rebuffi, Sven Gowal, Dan Andrei Calian, Florian Stimberg, Olivia Wiles, and Timothy A Mann. Data augmentation can improve robustness. Advances in Neural Information Processing Systems, 34:29935–29948, 2021.

[^41]: Evgenia Rusak, Lukas Schott, Roland S Zimmermann, Julian Bitterwolf, Oliver Bringmann, Matthias Bethge, and Wieland Brendel. A simple way to make neural networks robust against diverse image corruptions. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part III 16, pages 53–69. Springer, 2020.

[^42]: Tonmoy Saikia, Cordelia Schmid, and Thomas Brox. Improving robustness against common corruptions with frequency biased models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10211–10220, 2021.

[^43]: Hao Shao, Letian Wang, Ruobing Chen, Hongsheng Li, and Yu Liu. Safety-enhanced autonomous driving using interpretable sensor fusion transformer. In Conference on Robot Learning, pages 726–737. PMLR, 2023.

[^44]: Hao Shao, Letian Wang, Ruobing Chen, Steven L Waslander, Hongsheng Li, and Yu Liu. Reasonnet: End-to-end driving with temporal and global reasoning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13723–13733, 2023.

[^45]: Deqing Sun, Xiaodong Yang, Ming-Yu Liu, and Jan Kautz. Models matter, so does training: An empirical study of cnns for optical flow estimation. IEEE transactions on pattern analysis and machine intelligence, 42(6):1408–1423, 2019.

[^46]: Jiachen Sun, Qingzhao Zhang, Bhavya Kailkhura, Zhiding Yu, Chaowei Xiao, and Z Morley Mao. Benchmarking robustness of 3d point cloud recognition against common corruptions. arXiv preprint arXiv:2201.12296, 2022.

[^47]: Xinglong Sun, Adam W Harley, and Leonidas J Guibas. Refining pre-trained motion models. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 4932–4938. IEEE, 2024.

[^48]: C Szegedy. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199, 2013.

[^49]: Shiyu Tang, Ruihao Gong, Yan Wang, Aishan Liu, Jiakai Wang, Xinyun Chen, Fengwei Yu, Xianglong Liu, Dawn Song, Alan Yuille, et al. Robustart: Benchmarking robustness on architecture design and training techniques. arXiv preprint arXiv:2109.05211, 2021.

[^50]: Zachary Teed and Jia Deng. Deepv2d: Video to depth with differentiable structure from motion. arXiv preprint arXiv:1812.04605, 2018.

[^51]: Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, pages 402–419. Springer, 2020.

[^52]: Marin Toromanoff, Emilie Wirbel, and Fabien Moutarde. End-to-end model-free reinforcement learning for urban driving using implicit affordances. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7153–7162, 2020.

[^53]: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, 2017.

[^54]: Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jiagang Zhu, and Jiwen Lu. Drivedreamer: Towards real-world-driven world models for autonomous driving. arXiv preprint arXiv:2309.09777, 2023.

[^55]: Zi Wang, Shiyi Lan, Xinglong Sun, Nadine Chang, Zhenxin Li, Zhiding Yu, and Jose M Alvarez. Enhancing autonomous driving safety with collision scenario integration. arXiv preprint arXiv:2503.03957, 2025.

[^56]: Xinshuo Weng, Boris Ivanovic, Yan Wang, Yue Wang, and Marco Pavone. Para-drive: Parallelized architecture for real-time autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15449–15458, 2024.

[^57]: Penghao Wu, Xiaosong Jia, Li Chen, Junchi Yan, Hongyang Li, and Yu Qiao. Trajectory-guided control prediction for end-to-end autonomous driving: A simple yet strong baseline. Advances in Neural Information Processing Systems, 35:6119–6132, 2022.

[^58]: Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, and Dacheng Tao. Gmflow: Learning optical flow via global matching. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8121–8130, 2022.

[^59]: Gengshan Yang and Deva Ramanan. Volumetric correspondence networks for optical flow. Advances in neural information processing systems, 32, 2019.

[^60]: Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In CVPR, 2024.

[^61]: Tengju Ye, Wei Jing, Chunyong Hu, Shikun Huang, Lingping Gao, Fangzhen Li, Jingke Wang, Ke Guo, Wencong Xiao, Weibo Mao, et al. Fusionad: Multi-modality fusion for prediction and planning tasks of autonomous driving. arXiv preprint arXiv:2308.01006, 2023.

[^62]: Chengran Yuan, Zhanqi Zhang, Jiawei Sun, Shuo Sun, Zefan Huang, Christina Dao Wen Lee, Dongen Li, Yuhang Han, Anthony Wong, Keng Peng Tee, and Marcelo H. Ang Jr au2. Drama: An efficient end-to-end motion planner for autonomous driving with mamba, 2024.

[^63]: Longhui Yuan, Binhui Xie, and Shuang Li. Robust test-time adaptation in dynamic scenarios. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15922–15932, 2023.

[^64]: Jimuyang Zhang, Zanming Huang, and Eshed Ohn-Bar. Coaching a teachable student. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7805–7815, 2023.

[^65]: Zhejun Zhang, Alexander Liniger, Dengxin Dai, Fisher Yu, and Luc Van Gool. End-to-end urban driving by imitating a reinforcement learning coach. In Proceedings of the IEEE/CVF international conference on computer vision, pages 15222–15232, 2021.

[^66]: Yang Zheng, Adam W Harley, Bokui Shen, Gordon Wetzstein, and Leonidas J Guibas. Pointodyssey: A large-scale synthetic dataset for long-term point tracking. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19855–19865, 2023.

[^67]: Huizhong Zhou, Benjamin Ummenhofer, and Thomas Brox. Deeptam: Deep tracking and mapping. In Proceedings of the European conference on computer vision (ECCV), pages 822–838, 2018.

[^68]: Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable detr: Deformable transformers for end-to-end object detection. arXiv preprint arXiv:2010.04159, 2020.