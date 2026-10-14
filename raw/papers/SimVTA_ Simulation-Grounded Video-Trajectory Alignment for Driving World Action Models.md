---
title: "SimVTA: Simulation-Grounded Video-Trajectory Alignment for Driving World Action Models"
source: "https://arxiv.org/html/2610.13781v1"
author:
published:
created: 2026-10-14
description:
tags:
  - "clippings"
---
Changhui Jing <sup>1,2,4</sup>   Haochen Tian <sup>2 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\dagger"><semantics><mo>†</mo> <annotation>\dagger</annotation></semantics></math></sup>   Hongchen Li <sup>1,2,4</sup>   Tianyu Li <sup>1</sup>   Caojun Wang <sup>1</sup>     
Zihao Zhang <sup>2</sup>   Yanhao Liu <sup>1</sup>   Chengen Xie <sup>1</sup>   Bailin Li <sup>3</sup>   Hao Zhao <sup>5</sup>   Zhenjie Yang <sup>2 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\ddagger"><semantics><mo>‡</mo> <annotation>\ddagger</annotation></semantics></math></sup>  
<sup>1</sup> Shanghai Innovation Institute   <sup>2</sup> OpenDriveLab at The University of Hong Kong <sup>3</sup> Li Auto Inc.  <sup>4</sup> Tongji University   <sup>5</sup> Tsinghua University <sup>†</sup> Project Lead. <sup>‡</sup> Correspondence Author. [https://opendrivelab.com/SimVTA/](https://opendrivelab.com/SimVTA/)

###### Abstract

Driving world action models (WAMs) learn from paired trajectories and future observations. Post-training with reinforcement learning allows the policy to explore alternative trajectories and improve planning using driving rewards. However, driving logs contain future videos only for the recorded trajectories, leaving these alternatives without corresponding visual supervision. Thus, we introduce SimVTA, a simulation-grounded post-training framework that combines trajectory optimization with video supervision. First, we sample candidate trajectories for each observation and use a real-world simulator (e.g., World Engine) to render future videos for high-reward candidates in real-world scenes reconstructed with 3D Gaussian Splatting (3DGS). This provides trajectory-matched supervision independent of the model’s own predictions. Under a limited rendering budget, we prioritize video supervision for higher-error futures among these rendered candidates. Finally, we optimize trajectory planning with Group Relative Policy Optimization (GRPO) and use a flow-matching loss on the matched videos to update shared representations. At inference time, the model predicts trajectories directly, without simulator calls. On NAVSIM v1, applying SimVTA to a WAM initialization improves PDMS from 86.2 to 91.3 (+5.1). It achieves 88.2 EPDMS on NAVSIM v2 and an average HD-Score of 38.8 in closed-loop evaluation on HUGSIM. Ablations show that updating shared representations with video supervision improves planning, while prioritizing higher-error futures among high-reward trajectories further improves performance. The results and demo are available at [https://github.com/OpenDriveLab/SimVTA](https://github.com/OpenDriveLab/SimVTA).

![[1 1.png|Refer to caption]]

Figure 1: From logged futures to candidate-matched supervision. (1) WAM pretraining uses the ground-truth (GT) trajectory and its logged future video. (2) Trajectory GRPO explores candidate trajectories using driving rewards, without matched video supervision. (3) World Engine renders reconstructed scenes along selected candidate paths, providing matched supervision independent of the model’s predictions.

## 1 Introduction

Driving WAMs combine future-scene prediction with trajectory planning for the ego vehicle. Predicting future frames provides dense visual supervision that complements sparse trajectory targets and helps models learn representations for planning [^19]. Recent driving models retain video prediction during training but predict trajectories directly at inference time, avoiding the computational cost of future-video generation [^54].

Reinforcement learning can refine a planner trained on demonstrations to explore alternative trajectories and learn from driving rewards [^57]. However, driving logs contain future videos only for recorded trajectories. For example, when a candidate turns while the recorded trajectory continues straight, the logged video does not show the visual consequences. Reusing such videos creates incorrect trajectory-video pairs, while omitting video training leaves these candidates without visual supervision. This creates a supervision gap during post-training: current-policy trajectories lack matched future-video targets (Figure 1).

To address this gap, we introduce SimVTA, a simulation-grounded framework for video-trajectory alignment. A WAM’s own video predictions provide no external reference for checking whether they follow a candidate trajectory. We use the 3DGS-based simulator in World Engine [^17] to render reconstructed real-world scenes from camera poses along executed candidate paths (Figure 1). These renderings provide prediction-independent targets that match the scene geometry and motion of each selected trajectory, establishing *video-trajectory alignment*.

We jointly train video prediction with flow matching and trajectory planning with GRPO, allowing the video loss to update shared representations. The former teaches the model to predict the visual consequences of candidate trajectories; the latter compares driving rewards among candidates and favors higher-reward trajectories.

Rendering videos for every sampled trajectory is costly. We first select high-reward candidates for rendering, then prioritize video supervision for higher-error futures among the rendered candidates. This separates trajectory exploration from video-supervision target selection. Our ablations examine shared representation updates, target selection, trajectory-video correspondence, and online versus offline rendering. At inference, the planner predicts trajectories directly, without simulator calls or future-video generation.

Our main contributions are summarized as follows:

- We introduce SimVTA, a post-training framework that combines trajectory GRPO with simulator-based video supervision. It connects policy sampling, trajectory execution, rendering, and joint updates to learn the visual consequences of actions beyond those recorded in driving logs.
- We introduce a video-target selection strategy based on trajectory reward and video prediction error. It prioritizes higher-error futures of high-reward trajectories for video supervision without restricting trajectory exploration.
- Applying SimVTA to a WAM initialization improves NAVSIM v1 PDMS by 5.1 points, from 86.2 to 91.3, and achieves 88.2 EPDMS on NAVSIM v2 and an average HD-Score of 38.8 on HUGSIM.

![[2.png|Refer to caption]]

Figure 2: Simulation-grounded video–trajectory alignment. (a) Training overview. (b) Trajectory learning combines GRPO with a behavior-cloning (BC) regularizer. (c) Video supervision selects high-reward, higher-error futures. (d) World Engine renders matched futures along executed paths.

## 2 Related Work

#### World Action Models.

World models predict future observations or latent states to support driving simulation and planning [^41] [^5] [^19] [^47] [^40] [^42] [^8] [^29]. Video-based approaches jointly model future frames and trajectories [^52] [^26] [^35] [^25]. Video prediction can also be retained during training and omitted at inference, as explored in robotics by Fast-WAM [^50] and in driving by SimWAM [^54]. SimVTA focuses on trajectory-matched video supervision during driving WAM post-training.

#### Simulation for Driving.

Driving simulators support policy training and evaluation through controlled rollouts [^4] [^16] [^56] [^2] [^31] [^37] [^17] [^6]. 3DGS [^11] renders reconstructed scenes from specified camera poses, providing visual observations for closed-loop simulation [^56] [^31]. World Engine [^17] renders these observations along simulated ego motion for policy post-training. CausalDrive [^43] instead uses a learned renderer. SimVTA uses World Engine to render video targets along selected candidate trajectories.

#### Post-Training for Driving.

Driving policies can be refined after initial training through reinforcement learning or model adaptation [^38] [^51] [^46] [^36] [^21] [^57] [^32] [^14]. GRPO [^33] has been applied to both diffusion and autoregressive planners [^21] [^36]. PlannerRFT [^14] uses closed-loop reinforcement fine-tuning with adaptive exploration to improve diffusion planners. Our trajectory optimization uses the stochastic differential equation (SDE) sampling formulation from Flow-GRPO [^24]. ReSim [^44] trains a video simulator on simulated non-expert trajectories to improve action controllability. SimVTA jointly updates shared representations through video prediction and trajectory GRPO.

![[3.png|Refer to caption]]

Figure 3: Planning results and video-supervision comparisons. (a) NAVSIM v1. (b) NAVSIM v2 and HUGSIM. (c) Video-supervision configurations and rendering budget on NAVSIM v1.

## 3 Method

SimVTA’s post-training framework is illustrated in Figure 2 and comprises trajectory sampling, video rendering, and video supervision. We sample candidate trajectories from the current policy and execute them to obtain paths and driving rewards. World Engine [^17] renders videos along selected paths during training. We select video targets based on trajectory reward and video prediction error, then update the model with trajectory GRPO and video supervision. The next sampling round uses the updated policy.

### 3.1 Trajectory Sampling

Given observation history $o$, we compute a shared representation $F=h_{\theta}(o)$ from visual and ego-motion history for trajectory and video prediction. The trajectory policy $\pi_{\psi}$ uses SDE sampling to generate $G$ candidate trajectories:

$$
F=h_{\theta}(o),\qquad\tau_{i}\sim\pi_{\psi}(\cdot\mid F),\qquad i=1,\ldots,G.
$$

NAVSIM’s PDM executor $\mathcal{E}$ converts each candidate into an executed ego path $\tilde{\tau}_{i}=\mathcal{E}(o,\tau_{i})$. We compute its reward $R_{i}$ from the PDM component scores, using the progress-weighted aggregation in Eq. 6. All $G$ candidates can contribute to trajectory learning.

### 3.2 Video Rendering

We first check candidate execution validity, motion-conditioning support, and safety, then rank eligible candidates by $R_{i}$. World Engine renders the highest-reward candidates within the rendering budget $K$. These checks determine eligibility for video supervision and do not remove candidates from trajectory learning. For each selected candidate, the simulator $\mathcal{S}$ renders a target video along its executed path, and the video model $W_{\phi}$ predicts the future along the same path:

$$
V_{i}^{\rm sim}=\mathcal{S}(o,\tilde{\tau}_{i}),\qquad V_{i}^{\rm pred}=W_{\phi}(h_{\theta}(o),\tilde{\tau}_{i}).
$$

Here $W_{\phi}$ denotes the video prediction process. We render the reconstructed scene from camera poses along the executed path, as shown in Figure 2(d). The same path conditions video prediction, so both the rendered target and the prediction correspond to the same ego motion.

Rendering during training allows visual supervision to follow changes in the policy. A fixed offline cache contains videos only for trajectories sampled before optimization, whereas policy updates continuously change the trajectory distribution explored during post-training. The stored videos remain valid for their original paths, but do not provide matching targets for new paths with different motions or viewpoints. In each sampling round, SimVTA renders videos for selected candidates from the current policy, maintaining trajectory-video correspondence as training proceeds.

### 3.3 Video Supervision

#### Video Loss.

Let $z^{\rm sim}_{i,r}$ be the image latent of future frame $r$ in $V_{i}^{\rm sim}$, distinct from the shared representation $F$. Flow matching [^23] [^28] interpolates between Gaussian noise and the target latent:

$$
x_{i,r,t}=tz^{\rm sim}_{i,r}+(1-t)\epsilon,\qquad u_{i,r}^{\star}=z^{\rm sim}_{i,r}-\epsilon,\quad t\sim\mathcal{U}(0,1),\;\epsilon\sim\mathcal{N}(0,I).
$$

The velocity network $D_{\phi}$ conditions on history features $c_{\theta,i,r}$ and an ego-motion embedding $p_{\theta,i,r}$. During training, preceding rendered frames supply the history context; the clean current target is excluded. This teacher-forcing construction is given in Appendix B. The video loss is

$$
\mathcal{L}_{\mathrm{video}}(i)=\mathbb{E}_{t,\epsilon,r}\!\left[\frac{1}{d_{z}}\left\|D_{\phi}(x_{i,r,t},t;c_{\theta,i,r},p_{\theta,i,r})-\operatorname{sg}(u_{i,r}^{\star})\right\|_{2}^{2}\right],
$$

where $d_{z}$ is the number of elements in a frame latent and $\operatorname{sg}$ blocks gradients. The expectation averages over frames, flow times, and noise samples. The loss trains video prediction and the shared representations used for planning.

#### Target Selection.

We first select candidates for rendering by trajectory reward, then select rendered videos for supervision by prediction error (Figure 2(c)). For each valid rendered candidate, we compute the teacher-forced latent prediction error $E_{i}^{\rm TF}$ in Eq. 9, which combines frame reconstruction and temporal-change errors. Within each scene, we retain candidates whose error is at or above the mean over valid rendered candidates. This subset, denoted Q2, contains high-reward, high-error candidates. Their matched videos serve as targets for the flow-matching loss in Eq. 4, focusing supervision on visual consequences that the model predicts less accurately.

Table 1: Planning results on NAVSIM v1 NavTest. UniAD results are from [^22]; DriveVA also uses CARLA training data. Scores for the VLM-based baselines are taken from [^54]. Epona is adapted to NAVSIM following the official recipe and re-evaluated using the same protocol as SimVTA.

Method NC $\uparrow$ DAC $\uparrow$ TTC $\uparrow$ Comfort $\uparrow$ EP $\uparrow$ PDMS $\uparrow$ Traditional E2E planners UniAD [^7] 97.8 91.9 92.9 100.0 78.8 83.4 DiffusionDrive [^22] 98.2 96.2 94.7 100.0 82.2 88.1 WoTE [^20] 98.5 96.8 94.9 99.9 81.9 88.3 MeanFuser + BeyondDrive [^39] 98.5 98.4 95.1 100.0 84.2 90.3 ResAD [^55] 98.9 97.8 94.9 100.0 87.0 90.6 VLM-based planners AutoVLA [^57] 98.4 95.6 98.0 99.9 81.9 89.1 Uni-World VLA [^27] 98.7 96.7 96.1 100.0 83.2 89.4 ExploreVLA [^34] 98.8 98.4 96.5 99.9 83.5 90.4 ReCogDrive [^21] 97.9 97.3 94.9 100.0 87.3 90.8 SGDrive [^15] 98.6 97.8 96.2 100.0 85.8 91.1 World-model-based planners LAW [^18] 96.4 95.4 88.7 99.9 81.7 84.6 Epona [^52] 97.8 95.0 93.9 99.9 80.3 86.2 PWM [^53] 98.6 95.9 95.4 100.0 81.8 88.1 DreamerAD [^45] 98.0 97.2 94.3 100.0 83.1 88.7 DriveWAM [^35] 98.3 98.1 95.2 100.0 84.3 90.1 DriveVA [^26] 99.2 97.5 98.7 100.0 83.5 90.9 SimVTA 98.0 97.2 97.0 100.0 86.7 91.3

### 3.4 Training and Inference

We use SDE trajectory sampling following Flow-GRPO [^24] and group-relative advantages following GRPO [^33]. The trajectory loss $\mathcal{L}_{\mathrm{traj}}$ weights clipped transition scores by these advantages and applies a discount across denoising steps. Video prediction error does not enter the trajectory advantages. Our overall training objective combines the trajectory loss, the video loss on selected targets $\mathcal{L}_{\mathrm{video}}^{\rm sel}$, and the BC regularizer, as defined in Eq. 5:

$$
\mathcal{L}=\mathcal{L}_{\mathrm{traj}}+\lambda_{v}\mathcal{L}_{\mathrm{video}}^{\rm sel}+\lambda_{\rm BC}\mathcal{L}_{\mathrm{anchor}},
$$

where $\mathcal{L}_{\mathrm{anchor}}$ is the BC regularizer. The trajectory loss favors candidates with higher relative driving rewards, while the BC regularizer trains the model to follow denoising chains generated by a teacher planner (Figure 2(b)). We use the frozen initial trajectory network (TrajDiT) as the teacher. The teacher and student use the current shared representation, with gradients stopped on the teacher side. The weights $\lambda_{v}$ and $\lambda_{\rm BC}$ control video supervision and BC regularization. Training repeats sampling, execution, rendering, and model updates. At inference, the model predicts trajectories directly from observation history, without simulator calls, video generation, or video-based reranking.

Table 2: Planning results on NAVSIM v2 NavTest. Epona is re-evaluated using the same protocol as SimVTA. SGDrive’s scores are taken from [^54].

Method NC $\uparrow$ DAC $\uparrow$ DDC $\uparrow$ TLC $\uparrow$ EP $\uparrow$ TTC $\uparrow$ LK $\uparrow$ HC $\uparrow$ EC $\uparrow$ EPDMS $\uparrow$ Traditional E2E planners ResAD [^55] 97.8 97.2 99.5 99.8 88.2 96.9 97.0 98.4 88.2 85.5 DriveSuprim [^48] 98.4 98.6 99.6 99.8 90.5 97.8 97.0 98.3 78.6 87.1 DiffRefiner [^49] 98.6 98.5 99.7 99.9 87.6 98.1 98.1 98.3 87.9 87.4 VLM-based planners SGDrive [^15] 98.6 94.3 99.5 99.9 86.0 97.9 96.1 98.3 85.9 86.2 DriveWorld-VLA [^8] 98.6 99.1 99.6 99.8 87.4 97.9 97.0 97.8 78.6 86.8 ELF-VLA-8B [^30] 98.9 98.1 99.4 99.8 88.5 98.4 96.9 98.3 87.2 87.1 World-model-based planners Epona [^52] 97.2 95.6 99.3 99.7 88.4 96.4 96.9 97.9 67.6 85.0 SafeDrive [^12] 99.5 99.0 99.7 99.9 88.6 98.9 97.5 98.2 81.9 87.5 DreamerAD [^45] 98.0 97.2 99.5 99.8 87.8 97.4 97.5 98.3 72.4 87.7 SimVTA 98.5 98.5 99.8 99.0 90.8 98.5 97.5 99.0 60.0 88.2

## 4 Experiments

### 4.1 Experimental Setup

#### Implementation Details.

We initialize SimVTA from an Epona checkpoint adapted to NAVSIM as described in Appendix A, and perform post-training on NavTrain samples with available World Engine scene reconstructions. We use the 3DGS-based simulator in World Engine [^17] to render matched future videos. In SimVTA (Exp2), the video flow-matching loss uses Q2 targets and updates both the video network (VideoDiT) and the shared encoder. All ablations retain trajectory GRPO and the BC regularizer, with trajectory GRPO updating the shared encoder throughout. The mixed high/low-reward variants, Exp4 and Exp6, additionally apply the safety-specific advantage treatment described in Appendix D.

#### Benchmarks and Metrics.

We evaluate SimVTA on the NavTest splits of NAVSIM v1 and v2 [^3] [^1], using PDMS and EPDMS, respectively, to assess safety, progress, and comfort. On HUGSIM [^56], we report route completion (RC) and HD-Score, which combines RC with safety and comfort. Training uses the reward in Eq. 6; Benchmark scores follow their respective evaluation protocols. We report results averaged over three independent runs with different random seeds. Metric details are provided in Appendix F.

Table 3: Planning results on HUGSIM. E/M/H/X denote easy, medium, hard, and extreme scenarios. Baseline results are taken from Driving on Registers, Table 2. Higher values are better for all metrics.

RC $\uparrow$ HD-Score $\uparrow$ Method E M H X Avg. E M H X Avg. VAD 51.3 31.1 25.3 26.5 31.4 36.3 9.5 8.0 11.5 13.4 LTF 67.8 35.1 26.2 40.5 38.9 58.9 18.0 9.8 25.9 23.7 UniAD 78.4 60.5 33.6 17.8 45.9 64.9 45.8 20.6 6.6 32.7 DrivoR 80.9 50.5 33.8 47.1 49.8 73.3 34.6 18.8 32.5 35.7 SimVTA 78.8 57.6 38.3 40.1 51.5 69.9 45.0 23.3 27.2 38.8

### 4.2 Main Results

#### NAVSIM.

Starting from the NAVSIM-adapted Epona initialization at 86.2 PDMS, SimVTA reaches 91.3 (+5.1) on NAVSIM v1 (Figure 3a; Table 1). Ego progress rises from 80.3 to 86.7, time-to-collision compliance from 93.9 to 97.0, and drivable-area compliance from 95.0 to 97.2. On NAVSIM v2, SimVTA improves EPDMS from 85.0 to 88.2 (+3.2), and achieves 90.8 ego progress and 98.5 for both drivable-area and time-to-collision compliance (Table 2).

SimVTA also improves planning in scenes where the Epona initialization fails (Figure 4). In these examples, Epona’s trajectories lead to a collision or leave the drivable area, whereas SimVTA avoids the collision and stays within the drivable area.

![[4.png|Refer to caption]]

Figure 4: Qualitative planning comparison. Left: planned trajectories of SimVTA (A, blue) and Epona (B, orange). Right: World Engine renderings along the corresponding paths at 1–4 s. Red boxes highlight Epona’s collision and road-departure regions.

#### HUGSIM.

We further evaluate the final planner in closed-loop driving on the same 345 pre-challenge HUGSIM scenarios, using the same corrected evaluator and controller as [^13]. SimVTA achieves the highest average RC (51.5) and HD-Score (38.8) among the compared methods, exceeding the reported DrivoR results by 1.7 and 3.1 points, respectively (Table 3). It also leads on both metrics in hard scenarios, reaching 38.3 RC and 23.3 HD-Score.

#### Video-Trajectory Alignment.

Beyond the planning gains, we compare SimVTA with Epona to examine whether post-training improves the consistency of predicted videos with the scene along a trajectory (Figure 6). Epona generates an additional vehicle on the right (highlighted by red boxes) that is absent from both the bird’s-eye view (BEV) and the corresponding World Engine renderings. SimVTA avoids this inconsistency in the illustrated scene.

Table 4: Video quality.

<table><tbody><tr><th>Source</th><td>FID <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>FVD <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><th colspan="3">3DGS renderings</th></tr><tr><th>World Engine</th><td>7.6</td><td>21.8</td></tr><tr><th colspan="3">WAM predictions</th></tr><tr><th>Trajectory GRPO only (Exp0)</th><td>20.4</td><td>105.4</td></tr><tr><th>SimVTA (Exp2)</th><td>13.1</td><td>46.3</td></tr></tbody></table>

#### Video Quality.

We evaluate both World Engine renderings and autoregressive model predictions against real videos on NAVSIM v1 (Table 4). Lower Fréchet Inception Distance (FID) and Fréchet Video Distance (FVD) indicate better video quality. Along logged GT trajectories, World Engine achieves an FID of 7.6 and an FVD of 21.8. Compared with trajectory GRPO alone (Exp0), SimVTA reduces FID from 20.4 to 13.1 and FVD from 105.4 to 46.3, improving video prediction quality.

![[5.png|Refer to caption]]

Figure 5: Matched futures for alternative actions. Left: shared observation history and executed paths for going straight (A), changing lanes to the left (B), and changing lanes to the right (C). Right: corresponding future frames rendered by the simulator. Colors match the paths and image rows.

### 4.3 Ablation Studies

#### Updating the Shared Representations.

Video supervision improves planning when its gradients reach the shared encoder. Restricting video gradients to VideoDiT (Exp1) yields 89.1 PDMS, matching trajectory GRPO alone (Exp0; Figure 3a). Allowing video gradients to also update the shared encoder (Exp2) raises PDMS to 91.3 (+2.2) (Table 6).

#### Selecting Informative Futures.

We compare video supervision across four candidate groups: Q1 (high reward, low error), Q2 (high reward, high error), Q3 (low reward, low error), and Q4 (low reward, high error). High- and low-reward candidates come from the top and bottom of the trajectory reward ranking, respectively. Within each scene, errors below the mean over valid rendered candidates are low; those at or above it are high.

Q2 supervision yields the highest PDMS among the four settings. Exp2 uses Q2 supervision and achieves 91.3 PDMS. Exp5 uses the same high-reward rendering pool but expands video supervision to Q1+Q2, yielding 91.0 PDMS. Under the same high-reward rendering pool, focusing supervision on the higher-error Q2 subset performs better than supervising Q1+Q2. Exp4 and Exp6 use mixed high/low-reward pools with Q2+Q3 and Q1–Q4 supervision, respectively, both achieving 90.9 PDMS (Table 6). These two variants also apply the safety-specific advantage treatment described in Appendix D.

Table 5: Ablation on shared encoder updates.

| Setting | Video loss updates | NC $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- |
| Exp0 | None | 97.1 | 89.1 |
| Exp1 | VideoDiT | 97.6 | 89.1 |
| Exp2 | VideoDiT + shared encoder | 98.0 | 91.3 |

Table 6: Ablation on video target selection.

| Setting | Targets | NC $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- |
| Exp4 | Q2 + Q3 | 97.4 | 90.9 |
| Exp6 | Q1 + Q2 + Q3 + Q4 | 97.7 | 90.9 |
| Exp5 | Q1 + Q2 | 98.0 | 91.0 |
| Exp2 | Q2 | 98.0 | 91.3 |

Table 7: Video-training variants.

| Setting | Video supervision | NC $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- |
| Exp8 | All valid rendered + video GRPO | 97.9 | 91.0 |
| Exp2 | Q2 + flow matching | 98.0 | 91.3 |

Table 8: Online versus offline rendering.

| Setting | Rendering strategy | PDMS $\uparrow$ |
| --- | --- | --- |
| Exp0 | No video supervision | 89.1 |
| Exp9 | Fixed offline cache | 90.4 |
| Exp2 | Online rendering | 91.3 |

#### Comparing Video-Training Variants.

We further compare our selective video supervision with directly applying video GRPO to the rendered candidates. Under the same rendering budget of four high-reward candidates, Exp8 applies video GRPO to all valid rendered candidates and obtains 91.0 PDMS. In contrast, Exp2 first selects the higher-error Q2 subset according to $E_{i}^{\rm TF}$ and then optimizes the selected targets with the flow-matching loss, reaching 91.3 PDMS (Table 8). Both variants allow video gradients to update the shared encoder. This comparison tests whether selecting informative video targets before supervision is more effective than directly applying video GRPO to all valid rendered candidates.

#### Comparing Online and Offline Rendering.

We compare online rendering with a fixed offline cache (Table 8). Exp9 follows the same settings as Exp2, except that trajectories are sampled offline from the initial policy and their rendered videos are cached and reused throughout post-training. Exp2 instead refreshes video targets for current-policy trajectories. Exp9 achieves 90.4 PDMS, compared with 89.1 for trajectory-only Exp0 and 91.3 for Exp2, showing a 0.9-point advantage for refreshing video targets from the current policy over using a fixed initial-policy cache.

#### Preserving Trajectory-Video Correspondence.

To test whether the benefit depends on correct trajectory-video pairing rather than video supervision alone, Exp3 shuffles rendered targets across trajectory conditions within the same scene after selection. Breaking this alignment reduces PDMS from 91.3 (Exp2) to 83.1 (Exp3; Table 10), demonstrating the importance of preserving trajectory-video correspondence. Figure 5 shows the matched setting, where each target video is rendered along its corresponding path.

#### Increasing the Rendering Budget.

Doubling the rendering budget provides no PDMS gain: Exp7 achieves 91.2 versus 91.3 for Exp2 with the same target selection (Figure 3c; Table 10).

Table 9: Trajectory–video correspondence.

| Setting | Video targets | NC $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- |
| Exp3 | Shuffled | 94.1 | 83.1 |
| Exp2 | Matched | 98.0 | 91.3 |

Table 10: Rendering budget.

| Setting | Targets | Budget | NC $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- |
| Exp7 | Q2 | $2\times$ | 97.8 | 91.2 |
| Exp2 | Q2 | $1\times$ | 98.0 | 91.3 |

![[6.png|Refer to caption]]

Figure 6: Video–trajectory alignment before and after post-training. Left: observation history and trajectory in BEV. Right: World Engine renderings and predictions from SimVTA and Epona.

## 5 Conclusion

We presented SimVTA, a simulation-grounded post-training framework that combines trajectory GRPO with video supervision. Rendering futures along current-policy trajectories provides visual targets beyond logged action-future pairs and jointly trains shared representations through video prediction and trajectory optimization. The full framework improves NAVSIM v1 PDMS from 86.2 to 91.3 (+5.1) and achieves 88.2 EPDMS on NAVSIM v2 and 38.8 average HD-Score on HUGSIM. Ablations show that updating shared representations with video supervision improves planning, while prioritizing higher-error futures among high-reward trajectories further improves performance.

Limitations and directions for future work are discussed in Appendix G.

## Acknowledgments

We would like to thank other members of OpenDriveLab for their valuable discussions and technical support.

## References

Appendix

## Appendix A Training Procedure

Before applying SimVTA, we fine-tune the nuPlan-pretrained Epona checkpoint on NAVSIM following the official adaptation recipe to adapt it from 10 Hz to 2 Hz. The adapted checkpoint initializes both Exp0 and Exp2. We evaluate it using the same NAVSIM v1 protocol as the post-trained models and obtain 86.2 PDMS, matching the overall score reported for Epona. Unless stated otherwise, the implementation details below describe Exp2.

#### Training Loop.

1. Encode observation histories and sample $G$ trajectories, retaining their normalized denoising states.
2. Execute each candidate with PDM, compute its training reward, and normalize rewards within each observation group.
3. Check execution validity, pose support, and safety. Rank eligible candidates by reward and select the rendering pool.
4. Render the selected paths and encode their future frames, retaining the candidate-to-video correspondence.
5. Compute detached teacher-forced errors and classify the valid rendered candidates into quadrants.
6. Select the video-training subset and future frame for this update. Generate the detached BC teacher chain from the current shared representation.
7. Combine the trajectory loss, selected video loss, and BC regularizer. Update the trainable parameters and repeat with the updated policy.

#### Architecture and Temporal Settings.

Epona’s spatiotemporal Transformer is the shared encoder, paired with TrajDiT and VideoDiT. Four observed frames condition eight trajectory waypoints spanning four seconds. PDM evaluates the path at 10 Hz; video pose conditions are sampled at 2 Hz. Exp2 computes teacher-forced prediction errors over future frames within the planning window and trains one sampled future frame per update.

#### Trajectory Reward.

Each candidate is scored together with the cached PDM reference trajectory. The training reward is

$$
R_{i}=\mathrm{NC}_{i}\,\mathrm{DAC}_{i}\frac{10\,\mathrm{EP}_{i}+5\,\mathrm{TTC}_{i}+2\,\mathrm{Comfort}_{i}}{17}.
$$

Following ReCogDrive [^21], we use weights of 10, 5, and 2 for EP, TTC, and Comfort, respectively, with the higher EP weight encouraging efficient driving. The component scores lie in $[0,1]$. Progress follows the PDM scorer’s normalization over the candidate and reference trajectories. Driving-direction compliance is used for candidate screening but has zero weight in this reward. Exp2 excludes video prediction error from trajectory rewards.

#### Candidate Validity and Safety.

An eligible video candidate must execute successfully and have finite pose changes within VideoDiT’s support at every 2 Hz step: $0\leq\Delta x\leq 16$ m, $|\Delta y|\leq 1.5$ m, and $|\Delta\mathrm{yaw}|\leq 32^{\circ}$. Its maximum reverse speed must not exceed $10^{-3}$ m/s. For Exp2, the NC, DAC, TTC, and DDC scores must each be at least 0.999. We rank eligible candidates by $R_{i}$, break ties by sampling index, and select the first $K$. If fewer than $K$ qualify, only those candidates are rendered. Failed renders are excluded from video supervision. A scene skips video supervision if all candidates fail the safety checks or its maximum reward is at most $10^{-6}$. These filters do not remove candidates from the Exp2 trajectory loss.

#### Optimization Configuration.

Table 11 lists the video-supervision settings for Exp2 and the optimization defaults. Training uses AdamW, implemented by DeepSpeedCPUAdam in AdamW mode when optimizer offloading is enabled. Bias and normalization parameters are excluded from weight decay. The learning rate is halved at steps 300 and 1500. The shared encoder uses the same learning rate as the heads. When no video target participates across all devices, VideoDiT parameters remain unchanged for that update.

Table 11: Exp2 video-supervision settings and default optimization parameters.

| Parameter | Value |
| --- | --- |
| Sampled trajectories / rendering budget | $G=32$ / $K=4$ |
| Batch size per GPU / default random seed | 4 / 1234 |
| Training hardware | 16 NVIDIA H20 GPUs |
| Initial learning rate / AdamW betas | $10^{-5}$ / $(0.9,0.95)$ |
| Weight decay / LR decay factor | 0.01 / 0.5 |
| LR decay steps | 300, 1500 |
| Trajectory denoising steps / noise scale | 10 / 0.3 |
| Diffusion-coefficient floor / denoising discount | 0.01 / 0.6 |
| Video loss / BC weights | $\lambda_{v}=0.1$ / $\lambda_{\rm BC}=0.1$ |
| Video gradient scale into shared parameters | $\eta=0.2$ |
| Video-error horizon / temporal discount | $H=8$ / $\gamma_{\rm v}=0.85$ |
| Video-training candidates per scene per update | All valid Q2 |
| Trained future frames per candidate per update | 1 |
| Mixed precision / distributed optimizer | bfloat16 / ZeRO-2, CPU offload |

## Appendix B Conditional Video Prediction

The next-frame prediction head VideoDiT is named VisDiT in Epona. The process $W_{\phi}$ applies the velocity network $D_{\phi}$ and decodes the resulting image latents. Flow time $t$ in Eq. 3 runs from noise at zero to the target at one and is distinct from frame index $r$. The factor $1/d_{z}$ in Eq. 4 averages the squared error over the elements of one frame latent.

#### Internal Motion Conditioning.

For each selected candidate, World Engine renders the supervision target along its executed path. The WAM uses the same executed pose sequence for its internal video conditioning:

$$
V_{i}^{\rm sim}=\mathcal{S}(o,\tilde{\tau}_{i}),\qquad V_{i}^{\rm pred}=W_{\phi}(h_{\theta}(o),\tilde{\tau}_{i}).
$$

This pairing is used during post-training; the motion conditioning is internal to the WAM.

To predict frame $r$, we construct conditions from a rolling history window containing observed or preceding rendered frames and pose changes:

$$
c_{\theta,i,r}=h_{\theta}(o,z^{\rm sim}_{i,<r},\operatorname{sg}(\Delta\tilde{\tau}_{i,<r})),\qquad p_{\theta,i,r}=e_{\theta}(\operatorname{sg}(\Delta\tilde{\tau}_{i,r})).
$$

Here $<r$ denotes preceding frames, $\Delta\tilde{\tau}_{i,r}$ is the executed pose change, and $e_{\theta}$ embeds that motion. The expression for $h_{\theta}$ denotes the rolling context rather than an ever-growing sequence. The clean current target is excluded from this context. Rendered latents and pose inputs are fixed; the shared encoder and motion embedding remain trainable.

#### Teacher-forced Prediction Error.

We compute the prediction error at $t_{\star}=0.5$ using one Gaussian tensor $\epsilon_{o}$ shared by all candidates and future frames of an observation:

$$
\displaystyle x^{\star}_{i,r}
$$
 
$$
\displaystyle=t_{\star}z^{\rm sim}_{i,r}+(1-t_{\star})\epsilon_{o},
$$
$$
\displaystyle\hat{z}_{i,r}
$$
 
$$
\displaystyle=x^{\star}_{i,r}+(1-t_{\star})D_{\phi}(x^{\star}_{i,r},t_{\star};c_{\theta,i,r},p_{\theta,i,r}),
$$
$$
\displaystyle e_{i,r}
$$
 
$$
\displaystyle=\hat{z}_{i,r}-z^{\rm sim}_{i,r}.
$$

With $\alpha_{r}=\gamma_{\rm v}^{r-1}/\sum_{j=1}^{H}\gamma_{\rm v}^{j-1}$, we define

$$
\displaystyle E_{i}^{\rm TF}
$$
 
$$
\displaystyle=0.7\sum_{r=1}^{H}\alpha_{r}\frac{\|e_{i,r}\|_{2}^{2}}{d_{z}}
$$
 
$$
\displaystyle+0.3\sum_{r=2}^{H}\frac{\alpha_{r}}{\sum_{j=2}^{H}\alpha_{j}}\frac{\|\sqrt{\omega_{i,r}}\odot(e_{i,r}-e_{i,r-1})\|_{2}^{2}}{d_{z}}.
$$

For token $p$ and channel $c$, the motion weight is

$$
\displaystyle m_{i,r,p}
$$
 
$$
\displaystyle=\operatorname{mean}_{c}|z^{\rm sim}_{i,r,p,c}-z^{\rm sim}_{i,r-1,p,c}|,
$$
$$
\displaystyle\omega_{i,r,p}
$$
 
$$
\displaystyle=\operatorname{clip}\left(\frac{m_{i,r,p}}{\max(\operatorname{mean}_{p^{\prime}}m_{i,r,p^{\prime}},10^{-6})},0.25,4\right).
$$

The weight is broadcast over channels. We use hyperparameters $H$ and $\gamma_{\rm v}=0.85$. The complete error computation is detached. Unlike the video-training loss, this score uses a fixed flow time, shared evaluation noise, and a temporal-change term.

## Appendix C Video Supervision and Policy Optimization

#### Quadrant Assignment and Training Subset.

Reward labels follow the ranked rendering pool: Exp2 uses the high-reward pool, while mixed-pool variants retain separate high- and low-reward labels. No reward-mean threshold is applied. The error threshold is the mean of $E_{i}^{\rm TF}$ over valid rendered candidates within each scene. An error below the mean is low; an error equal to or above it is high.

For each scene, Exp2 trains on all valid Q2 candidates at every update. At training step $s$, the trained future frame for each candidate is $1+(s\bmod H)$. Preceding target frames still provide context when only a later frame is trained. Flow time and training noise are sampled independently for each selected latent. Candidate selection is detached.

#### Video-gradient Masks.

Let $m_{i}^{\rm head}$ include candidate $i$ in video training, and let $m_{i}^{\rm shared}\leq m_{i}^{\rm head}$ additionally allow gradients into the shared parameters. We use $g_{\eta}(x)=\operatorname{sg}(x)+\eta(x-\operatorname{sg}(x))$, which preserves the forward value and scales its backward gradient by $\eta$. For the current update, we estimate Eq. 4 at the cyclic frame $r=1+(s\bmod H)$ using newly sampled flow time and noise. Write $\mathcal{L}_{\mathrm{video}}^{\rm joint}(i)$ for this estimate with $g_{\eta}(c_{\theta,i,r})$ and $g_{\eta}(p_{\theta,i,r})$ as conditions, and $\mathcal{L}_{\mathrm{video}}^{\rm head}(i)$ for the estimate with both conditions detached. Then

$$
\mathcal{L}_{\mathrm{video}}^{\rm sel}=\frac{1}{Z}\sum_{i}\left[m_{i}^{\rm shared}\mathcal{L}_{\mathrm{video}}^{\rm joint}(i)+(m_{i}^{\rm head}-m_{i}^{\rm shared})\mathcal{L}_{\mathrm{video}}^{\rm head}(i)\right],
$$

where $Z=\max(1,\sum_{i}m_{i}^{\rm head})$ and the sum covers participating candidates across all devices. In Exp2, both masks select Q2 candidates and $\eta=0.2$. This scale affects only video gradients entering the shared encoder and motion embedding; trajectory, BC, and VideoDiT gradients retain their own loss weights. The video loss is zero when no target participates.

#### Trajectory GRPO.

We use group-relative advantages with a clipped transition-score objective. Let $a_{i}^{(k)}$ be the normalized trajectory state at step $k$, with $N$ transitions and $d_{\tau}$ scalar coordinates. For a stored transition $(a,a^{\prime})$, the sampler specifies a mean $\mu_{\theta,\psi,k}(a,o)$ and standard deviation $\sigma_{k}$. The implemented score is

$$
\ell_{\theta,\psi,k}(a,a^{\prime};o)=\frac{1}{d_{\tau}}\sum_{d=1}^{d_{\tau}}\operatorname{clip}\left(-\frac{(a^{\prime}_{d}-\mu_{\theta,\psi,k,d}(a,o))^{2}}{2\sigma_{k}^{2}},-5,2\right).
$$

This score omits the Gaussian normalizing term and is clipped before averaging over coordinates. For each observation, we compute

$$
\displaystyle\bar{R}
$$
 
$$
\displaystyle=\frac{1}{G}\sum_{j=1}^{G}R_{j},
$$
$$
\displaystyle s_{R}^{2}
$$
 
$$
\displaystyle=\frac{1}{G-1}\sum_{j=1}^{G}(R_{j}-\bar{R})^{2},
$$
$$
\displaystyle A_{i}
$$
 
$$
\displaystyle=\frac{R_{i}-\bar{R}}{s_{R}+\varepsilon_{\rm adv}},
$$
$$
\displaystyle\mathcal{L}_{\mathrm{traj}}
$$
 
$$
\displaystyle=-\mathbb{E}_{o}\left[\frac{1}{GN}\sum_{i=1}^{G}\sum_{k=0}^{N-1}\gamma_{\rm d}^{N-1-k}\operatorname{sg}(A_{i})\ell_{\theta,\psi,k}(a_{i}^{(k)},a_{i}^{(k+1)};o)\right].
$$

We use $\varepsilon_{\rm adv}=10^{-8}$ and $\gamma_{\rm d}=0.6$. Sampled states and advantages are detached; gradients through the transition mean update the shared encoder and TrajDiT. The trajectory objective uses no current-to-sampling-policy probability ratio.

#### SDE Transitions.

Let $s_{0}=1$ and $s_{N}=0$, with a decreasing grid between them. For transitions $k=0,\ldots,N-1$, define $\Delta s_{k}=s_{k+1}-s_{k}<0$. The trajectory velocity network $v_{\psi}$ uses ascending flow time $1-s_{k}$. Define

$$
\displaystyle\bar{s}_{k}
$$
 
$$
\displaystyle=\begin{cases}s_{1},&k=0,\\
s_{k},&k>0,\end{cases}
$$
 
$$
\displaystyle\beta_{k}
$$
 
$$
\displaystyle=\max\left(\beta_{\min},\nu\sqrt{\frac{s_{k}}{1-\bar{s}_{k}}}\right),
$$
$$
\displaystyle\mu_{\theta,\psi,k}(a,o)
$$
 
$$
\displaystyle=a\left(1+\frac{\beta_{k}^{2}}{2s_{k}}\Delta s_{k}\right)
$$
 
$$
\displaystyle-v_{\psi}(a,1-s_{k};h_{\theta}(o))\left(1+\frac{\beta_{k}^{2}(1-s_{k})}{2s_{k}}\right)\Delta s_{k},
$$
$$
\displaystyle\sigma_{k}
$$
 
$$
\displaystyle=\beta_{k}\sqrt{-\Delta s_{k}}.
$$

Starting from Gaussian noise, we sample $a^{(k+1)}=\mu_{\theta,\psi,k}+\sigma_{k}\operatorname{clip}(\xi_{k},-5,5)$ with independent $\xi_{k}\sim\mathcal{N}(0,I)$. We use $\nu=0.3$, $\beta_{\min}=0.01$, and $N=10$. The shifted time grid is $s_{k}=e^{m}u_{k}/(e^{m}u_{k}+1-u_{k})$, where $u_{k}=1-k/N$ and $m=0.5+(1.15-0.5)(8-256)/(4096-256)$. Before denormalization and reward evaluation, final longitudinal, lateral, and yaw action coordinates are clipped to $[-1.7,1.7]$, $[-2.6,2.6]$, and $[-10,10]$, respectively. These bounds are in normalized coordinates. The inverse mapping is $x=25(a_{x}+1)$ m, $y=10a_{y}$ m, and $\mathrm{yaw}=12a_{\mathrm{yaw}}$ degrees, with waypoints expressed relative to the current ego frame. Stored policy transitions retain the unclipped final state.

#### BC Regularizer.

Let $\psi_{0}$ denote the frozen initial TrajDiT parameters. For each observation, one Gaussian initial state generates an ODE chain $b^{(0)},\ldots,b^{(N)}$ using $\psi_{0}$ and $\operatorname{sg}(h_{\theta}(o))$. The teacher follows $b^{(k+1)}=b^{(k)}-\Delta s_{k}v_{\psi_{0}}(b^{(k)},1-s_{k};\operatorname{sg}(h_{\theta}(o)))$. Its final state uses the action-clipping bounds above. We optimize

$$
\mathcal{L}_{\mathrm{anchor}}=-\mathbb{E}_{o,b^{(0)}}\left[\frac{1}{N}\sum_{k=0}^{N-1}\ell_{\theta,\psi,k}(\operatorname{sg}(b^{(k)}),\operatorname{sg}(b^{(k+1)});o)\right].
$$

The teacher parameters remain fixed, while teacher chains are regenerated from the current shared representation. Student-side gradients update the shared encoder and TrajDiT. The BC objective averages over steps without the trajectory loss’s denoising discount.

## Appendix D Experimental Variants and Budget Accounting

All ablations use Exp2 as the reference setting. Paper experiment numbers differ from the implementation entry names: Exp5, Exp6, and Exp8 correspond to exp8\_high4\_all, exp11\_high2\_low2\_all, and exp10\_high4\_video\_grpo, respectively. Table 12 summarizes the supervision configurations, and Table 13 reports their NAVSIM v1 results. Both tables use paper experiment numbers and include Exp0–Exp9.

#### Gradient Paths and Candidate Pools.

Exp0 uses trajectory GRPO and BC without video supervision. In Exp1, video gradients update only VideoDiT; trajectory and BC gradients still update the shared encoder. Exp2 also allows video gradients to update shared parameters. Both use Q2 targets. Exp3 shuffles rendered targets across trajectory conditions within a scene after selection. Scenes without a valid target permutation contribute no shuffled video supervision.

Exp4 selects the two highest- and two lowest-reward candidates from the execution- and pose-valid pool. If this pool contains $n<4$ candidates, the high and low subsets contain $\lceil n/2\rceil$ and $\lfloor n/2\rfloor$ candidates, respectively. The subsets are disjoint, with reward ties broken by sampling index. Exp4 applies the four safety checks before training on Q2 and Q3. Exp5 renders four safe high-reward candidates and trains on all valid Q1 and Q2 candidates. Exp6 uses the same high/low pool as Exp4 but trains on all valid rendered candidates, including individually unsafe candidates. In Exp4 and Exp6, an unsafe candidate’s positive trajectory advantage is additionally truncated to zero: $A_{i}\leftarrow\min(A_{i},0)$. Exp2 does not apply this projection. Exp7 increases the rendering budget from four to eight while retaining Exp2’s Q2 selection and video-training rule. Both settings train on all valid Q2 candidates. Table 12 distinguishes the rendered pool from the subset used for video training.

#### Fixed Offline Cache (Exp9).

Exp9 follows the same experimental settings as Exp2, except that trajectories are sampled offline from the initial model and their rendered videos are stored in a fixed cache. The cached trajectory-video pairs are reused throughout post-training. All other settings, including candidate selection, rendering budget, training objectives, and optimization hyperparameters, are identical to Exp2. Exp9 achieves 90.4 PDMS, compared with 89.1 for Exp0 and 91.3 for Exp2 (Table 8).

Table 12: Video-supervision configurations. $K$ is the maximum number of rendered candidates per observation in each sampling round. For Exp9, this budget applies to offline cache construction. FM denotes flow matching.

| Setting | $K$ | Rendered pool | Video-training set | Video objective | PDMS |
| --- | --- | --- | --- | --- | --- |
| Exp0 | 0 | None | None | None | 89.1 |
| Exp1 | 4 | High reward | Q2 | FM | 89.1 |
| Exp2 | 4 | High reward | Q2 | FM | 91.3 |
| Exp3 | 4 | High reward | Q2, shuffled | FM | 83.1 |
| Exp4 | 4 | 2 high + 2 low | Q2 + Q3 | FM | 90.9 |
| Exp5 | 4 | High reward | Q1 + Q2 | FM | 91.0 |
| Exp6 | 4 | 2 high + 2 low | Q1–Q4 | FM | 90.9 |
| Exp7 | 8 | High reward | Q2 | FM | 91.2 |
| Exp8 | 4 | High reward | Valid rendered candidates | Video GRPO | 91.0 |
| Exp9 | 4 | Initial-policy cache | Q2 from cache | FM | 90.4 |

Table 13: NAVSIM v1 ablation results. Configurations are listed in Table 12.

| Setting | NC | DAC | TTC | Comfort | EP | PDMS |
| --- | --- | --- | --- | --- | --- | --- |
| Exp0 | 97.1 | 96.7 | 96.0 | 99.8 | 83.6 | 89.1 |
| Exp1 | 97.6 | 96.7 | 96.4 | 99.7 | 83.3 | 89.1 |
| Exp2 | 98.0 | 97.2 | 97.0 | 100.0 | 86.7 | 91.3 |
| Exp3 | 94.1 | 91.3 | 91.8 | 99.6 | 81.2 | 83.1 |
| Exp4 | 97.4 | 97.3 | 96.4 | 99.9 | 86.4 | 90.9 |
| Exp5 | 98.0 | 97.1 | 97.0 | 99.9 | 86.1 | 91.0 |
| Exp6 | 97.7 | 97.2 | 96.6 | 99.9 | 86.5 | 90.9 |
| Exp7 | 97.8 | 97.3 | 96.9 | 99.9 | 86.6 | 91.2 |
| Exp8 | 97.9 | 97.2 | 96.9 | 99.9 | 86.2 | 91.0 |
| Exp9 | 97.1 | 97.2 | 95.9 | 99.9 | 85.9 | 90.4 |

#### Video GRPO Control.

Exp8 evaluates video GRPO under the same four-candidate rendering budget used by Exp2. Exp8 applies video GRPO to all valid rendered high-reward candidates, whereas Exp2 first selects the Q2 subset and optimizes the selected targets with flow matching. For each candidate, it generates future latents with five stochastic Euler steps and noise scale 0.3, using preceding target frames as context. The reward is the negative framewise latent MSE, aggregated with normalized temporal weights proportional to $0.85^{r-1}$. It does not include the motion-weighted temporal-change term in Eq. 9. Different trajectory conditions within the same scene form the video comparison group. The video policy objective uses group-normalized advantages and a clipped ratio computed from dimension-normalized Gaussian log densities, with clipping parameter 0.2. One future frame is replayed for optimization per update, cycling through the video horizon. Groups with fewer than two valid candidates or reward standard deviation below $10^{-6}$ contribute no video policy update. Video GRPO replaces video flow matching and updates VideoDiT and shared parameters; its reward does not enter trajectory advantages.

#### Rendering and Optimization Budgets.

$K$ specifies the number of candidates budgeted for rendering; Table 12 separately lists those used for video training. Error-based selection follows rendering and scoring, so these costs apply to every rendered candidate. Total training cost also includes trajectory sampling, PDM evaluation, video optimization, and optimizer updates. Comparisons at equal $K$ match rendering caps, while video optimization costs depend on the training subset and objective. Exp9 renders its cache once; Exp2 renders candidates at each sampling round.

#### Interpreting the Ablations.

Table 12 summarizes the supervision configurations underlying the main-text ablations. For Exp3, scenes without a valid within-scene target permutation contribute no shuffled video supervision. Exp2 and Exp5 compare Q2 versus Q1+Q2 supervision from the same rendering pool, although the number of video-training candidates differs. Exp4 and Exp6 additionally modify the safety-specific treatment of trajectory advantages and are therefore interpreted as broader configuration comparisons.

#### Results across Random Seeds.

Table 14 reports NAVSIM v1 PDMS for Exp0, Exp2, Exp5, and Exp7 across three random seeds. Exp2 outperforms the other three variants on every seed.

Table 14: NAVSIM v1 PDMS across random seeds. The final column reports the mean and sample standard deviation over three runs.

<table><thead><tr><th>Setting</th><th colspan="3">Random seed</th><th>Mean <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> SD</th></tr></thead><tbody><tr><td></td><th>1234</th><th>20260901</th><th>20260910</th><td></td></tr><tr><td>Exp0</td><td>89.1</td><td>89.2</td><td>89.0</td><td><math><semantics><mrow><mn>89.1</mn> <mo>±</mo> <mn>0.1</mn></mrow> <annotation>89.1\pm 0.1</annotation></semantics></math></td></tr><tr><td>Exp2</td><td>91.3</td><td>91.4</td><td>91.3</td><td><math><semantics><mrow><mn>91.3</mn> <mo>±</mo> <mn>0.1</mn></mrow> <annotation>91.3\pm 0.1</annotation></semantics></math></td></tr><tr><td>Exp5</td><td>91.0</td><td>90.9</td><td>91.0</td><td><math><semantics><mrow><mn>91.0</mn> <mo>±</mo> <mn>0.1</mn></mrow> <annotation>91.0\pm 0.1</annotation></semantics></math></td></tr><tr><td>Exp7</td><td>91.1</td><td>91.2</td><td>91.2</td><td><math><semantics><mrow><mn>91.2</mn> <mo>±</mo> <mn>0.1</mn></mrow> <annotation>91.2\pm 0.1</annotation></semantics></math></td></tr></tbody></table>

## Appendix E Video Supervision and Evaluation

The teacher-forced error quantifies model prediction consistency against rendered simulation targets given preceding reference frames. We separately evaluate renderer fidelity, autoregressive video generation quality, and action-conditioning accuracy.

#### Renderer Fidelity.

We compare front-view World Engine renderings along logged GT trajectories with real observations on NAVSIM v1 NavTest. Each clip contains eight future frames over four seconds at 2 Hz. FID is computed on frames and FVD on clips using the same preprocessing and feature extractors for the two domains. These measurements quantify rendering fidelity at logged viewpoints; novel-view coverage is discussed in Appendix G.

#### Model Video Quality.

For each evaluation sample, Exp0 and Exp2 generate autoregressive video predictions along the same logged GT trajectory used for the World Engine rendering. Both models use the same observation history. We use the same dataset, clip length, preprocessing, and feature extractors as the renderer evaluation. FID and FVD compare the predictions with the corresponding real observations.

#### Qualitative Examples.

Figure 5 shows a shared observation history, a bird’s-eye view of three alternative executed paths (A–C), and World Engine renderings along each path at $+1$, $+2$, $+3$, and $+4$ seconds. Figure 6 compares World Engine renderings with predictions from SimVTA and the Epona initialization at the same displayed future times.

## Appendix F Cross-Benchmark Results

#### Evaluation Metrics and Scope.

PDMS combines collision and drivable-area terms with progress, TTC, and comfort. EPDMS additionally evaluates driving direction, traffic lights, lane keeping, and further comfort measures. On HUGSIM, HD-Score combines safety and comfort over the rollout and scales the result by RC. Higher scores are better. We evaluate SimVTA and the NAVSIM-adapted Epona initialization on the full NAVSIM v2 NavTest set using the NAVSIM v2.2 devkit with single-stage EPDMS evaluation, LQR tracking, and non-reactive traffic agents.

#### Result Sources and Comparison Scope.

Figure 7 summarizes the NAVSIM and HUGSIM results. On NAVSIM v1, the full framework improves PDMS by 5.1 points over the initialization (86.2 to 91.3). The comparison with Exp0 (89.1 to 91.3) measures the additional gain when video supervision is added within the framework. NAVSIM v2 and HUGSIM extend the evaluation to additional planning criteria and closed-loop driving. On NAVSIM v2, the NAVSIM-adapted Epona initialization obtains 85.0 EPDMS under the same evaluation protocol as SimVTA, which reaches 88.2 (+3.2). This aggregate improvement coexists with lower traffic-light compliance (99.7 to 99.0) and extended comfort (67.6 to 60.0) (Table 2). Baseline HUGSIM results in Table 3 are taken from [^13]. SimVTA achieves the highest reported averages among these methods and leads on hard scenarios, while DrivoR has higher extreme-scenario RC and HD-Score (47.1/32.5 versus 40.1/27.2). These closed-loop results assess the final planner; isolating the post-training gain on HUGSIM additionally requires the initialization and Exp0 under the same routes, controller, and rollout settings.

#### Recent WAM Results.

Table 15 includes results from recent WAMs discussed in the related work, including methods whose reports also supply baseline values in the main tables. These published results complement the within-backbone ablations; the table retains each source’s checkpoint and evaluation convention.

Table 15: Recent WAM results reported by their respective papers. Training data, initialization, and evaluators differ across methods.

| Method | NAVSIM v1 PDMS | NAVSIM v2 EPDMS |
| --- | --- | --- |
| SimWAM [^54] | 91.5 | 90.2 |
| Latent-WAM [^40] | – | 89.3 |
| WA-JEPA [^42] | 91.8 | 91.7 |
| SimVTA | 91.3 | 88.2 |

Sources: SimWAM, arXiv:2608.07468v4, Tables 1–2; Latent-WAM, arXiv:2603.24581v1, Table 1; WA-JEPA, arXiv:2608.20974v2, Tables 1 and 3. SimWAM’s v1 and v2 results are from its post-RL and pre-RL checkpoints, respectively. A dash indicates no result in the cited source.

![[7.png|Refer to caption]]

Figure 7: Reported planning results across benchmarks. (a) NAVSIM v1 PDMS and v2 EPDMS, with NAVSIM v2 component-score differences ( SimVTA minus the NAVSIM-adapted Epona initialization). (b) HUGSIM RC and HD-Score using benchmark aggregation. (c) RC and (d) HD-Score for DrivoR and by scenario difficulty. Values correspond to Tables 1, 2, and 3.

## Appendix G Limitations

#### Reconstruction Quality and Coverage.

Occlusions and limited pose coverage can degrade renderings away from logged paths, while renderer metrics assess only logged viewpoints. Even with safety and pose-support checks, rendering artifacts can still appear. A high prediction error can stem from either model limitations or poor-quality simulation targets. Quality-aware filtering and evaluation at alternative poses remain necessary.

#### Distribution Shifts.

Current-policy trajectories broaden action coverage, but observation histories remain log-based and rendered targets can differ from real images. Reconstruction coverage and candidate filtering further limit supervision. Teacher-forced selection cannot capture autoregressive error accumulation. We leave robustness evaluation under shifted observation histories and reactive traffic [^10] [^9] for future work.

#### Architecture and Training Cost.

We validate our framework based on the Epona backbone. Generalization to alternative WAM architectures is left for future work. Online rendering, error scoring, and video optimization increase training cost, while inference uses direct trajectory prediction.

[^1]: W. Cao, M. Hallgarten, T. Li, D. Dauner, X. Gu, C. Wang, Y. Miron, M. Aiello, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta Pseudo-simulation for autonomous driving. In CoRL, Cited by: §4.1.

[^2]: M. Cusumano-Towner, D. Hafner, A. Hertzberg, B. Huval, A. Petrenko, E. Vinitsky, E. Wijmans, T. Killian, S. Bowers, O. Sener, et al. Robust autonomy emerges from self-play. In ICML, Cited by: §2.

[^3]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta NAVSIM: data-driven non-reactive autonomous vehicle simulation and benchmarking. In NeurIPS, Cited by: §4.1.

[^4]: A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun CARLA: An open urban driving simulator. In CoRL, Cited by: §2.

[^5]: S. Gao, J. Yang, L. Chen, K. Chitta, Y. Qiu, A. Geiger, J. Zhang, and H. Li Vista: a generalizable driving world model with high fidelity and versatile controllability. In NeurIPS, Cited by: §2.

[^6]: H. He, C. Hu, Z. Su, H. Zhang, H. Liu, J. Li, H. Tian, Z. Shen, H. Li, Z. Li, Y. Yang, B. Huang, S. Zhang, K. Chen, X. Zhang, W. Dai, H. Dai, S. Liu, Z. Huang, and N. Wang DriveZero: end-to-end driving beyond human demonstrations. arXiv preprint arXiv:2609.06055. Cited by: §2.

[^7]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, et al. Planning-oriented autonomous driving. In CVPR, Cited by: Table 1.

[^8]: F. Jia, L. Liu, Z. Song, C. Jia, H. Ye, X. Hao, and L. Chen DriveWorld-VLA: unified latent-space world modeling with vision-language-action for autonomous driving. In ICML, Cited by: §2, Table 2.

[^9]: X. Jia, Y. Shao, Z. Yang, Q. Li, Z. Zhang, and J. Yan Bench2Drive-vl: benchmarks for closed-loop autonomous driving with vision-language models. arXiv preprint arXiv:2604.01259. Cited by: Appendix G.

[^10]: X. Jia, Z. Yang, Q. Li, Z. Zhang, and J. Yan Bench2Drive: towards multi-ability benchmarking of closed-loop end-to-end autonomous driving. arXiv preprint arXiv:2406.03877. Cited by: Appendix G.

[^11]: B. Kerbl, G. Kopanas, T. Leimkühler, and G. Drettakis 3D gaussian splatting for real-time radiance field rendering. TOG. Cited by: §2.

[^12]: J. Kim, J. Oh, S. Yu, H. Shin, D. Kwak, and J. W. Choi SafeDrive: fine-grained safety reasoning for end-to-end driving in a sparse world. In CVPR, Cited by: Table 2.

[^13]: E. Kirby, A. Boulch, Y. Xu, Y. Yin, G. Puy, É. Zablocki, A. Bursuc, S. Gidaris, R. Marlet, F. Bartoccioni, A. Cao, N. Samet, T. Vu, and M. Cord Driving on registers. In CVPR, Cited by: Appendix F, §4.2.

[^14]: H. Li, T. Li, J. Yang, H. Tian, C. Wang, L. Shi, M. Shang, Z. Lin, G. Wu, Z. Hao, et al. PlannerRFT: reinforcing diffusion planners through closed-loop and sample-efficient fine-tuning. arXiv preprint arXiv:2601.12901. Cited by: §2.

[^15]: J. Li, J. Wu, D. Hu, X. Huang, B. Sun, Z. Hao, X. Lang, X. Zhu, and L. Zhang SGDrive: scene-to-goal hierarchical world cognition for autonomous driving. In CVPR, Cited by: Table 1, Table 2.

[^16]: Q. Li, Z. Peng, L. Feng, Q. Zhang, Z. Xue, and B. Zhou Metadrive: composing diverse driving scenarios for generalizable reinforcement learning. TPAMI. Cited by: §2.

[^17]: T. Li, L. Chen, C. Wang, H. Liu, K. Chitta, Z. Yang, Y. Lu, N. Ye, Y. Qiu, H. Tian, Y. Wang, et al. World engine: towards the era of post-training for autonomous driving. arXiv preprint arXiv:2606.19836. Cited by: §1, §2, §3, §4.1.

[^18]: Y. Li, L. Fan, J. He, Y. Wang, Y. Chen, Z. Zhang, and T. Tan Enhancing end-to-end autonomous driving with latent world model. In ICLR, Cited by: Table 1.

[^19]: Y. Li, S. Shang, W. Liu, B. Zhan, H. Wang, Y. Wang, Y. Chen, X. Wang, Y. An, C. Tang, et al. DriveVLA-w0: world models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796. Cited by: §1, §2.

[^20]: Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang End-to-end driving with online trajectory evaluation via bev world model. arXiv preprint arXiv:2504.01941. Cited by: Table 1.

[^21]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, et al. ReCogDrive: a reinforced cognitive framework for end-to-end autonomous driving. In ICLR, Cited by: Appendix A, §2, Table 1.

[^22]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al. DiffusionDrive: truncated diffusion model for end-to-end autonomous driving. In CVPR, Cited by: Table 1, Table 1.

[^23]: Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le Flow Matching for generative modeling. In ICLR, Cited by: §3.3.

[^24]: J. Liu, G. Liu, J. Liang, Y. Li, J. Liu, X. Wang, P. Wan, D. Zhang, and W. Ouyang Flow-GRPO: training Flow Matching models via online RL. In NeurIPS, Cited by: §2, §3.4.

[^25]: M. Liu, D. Zhang, J. Liu, J. Cui, H. Xie, G. Chen, H. Ye, F. Nex, H. Cheng, and M. Y. Yang UNIVERSE: unified video action models for autonomous driving with flexible mask-modulated modality generation. arXiv preprint arXiv:2607.05133. Cited by: §2.

[^26]: M. Liu, D. Zhang, J. Liu, J. Cui, H. Xie, G. Chen, H. Ye, M. Y. Yang, F. Nex, and H. Cheng DriveVA: video action models are zero-shot drivers. arXiv preprint arXiv:2604.04198. Cited by: §2, Table 1.

[^27]: Q. Liu, H. Xu, J. Li, B. Sun, Z. Hao, D. She, X. Zhu, and L. Zhang Uni-World VLA: interleaved world modeling and planning for autonomous driving. arXiv preprint arXiv:2603.27287. Cited by: Table 1.

[^28]: X. Liu, C. Gong, and Q. Liu Flow straight and fast: learning to generate and transfer data with Rectified Flow. In ICLR, Cited by: §3.3.

[^29]: J. Lu, J. Guan, Z. Huang, J. Li, G. Li, L. Kong, Y. Li, H. Wang, S. Xu, Y. Luo, et al. Xiaomi onevl: one-step latent reasoning and planning with vision-language explanation. arXiv preprint arXiv:2604.18486. Cited by: §2.

[^30]: Y. Luo, F. Li, Q. Chen, S. Xu, J. Liu, Z. Song, Z. Yang, and F. Wen Unleashing VLA potentials in autonomous driving via explicit learning from failures. In CVPR, Cited by: Table 2.

[^31]: NVIDIA, Y. Cao, R. de Lutio, S. Fidler, G. G. Cobo, Z. Gojcic, M. Igl, B. Ivanovic, P. Karkus, J. M. Esturo, M. Pavone, A. Smith, E. Tanimura, M. Tyszkiewicz, M. Watson, Q. Wu, and L. Zhang Alpasim: a modular, lightweight, and data-driven research simulator for autonomous driving. Cited by: §2.

[^32]: NVIDIA, Y. Wang, W. Luo, J. Bai, Y. Cao, T. Che, et al. Alpamayo-R1: bridging reasoning and action prediction for generalizable autonomous driving in the long tail. arXiv preprint arXiv:2511.00088. Cited by: §2.

[^33]: Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li, Y. Wu, and D. Guo DeepSeekMath: pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300. Cited by: §2, §3.4.

[^34]: Z. Sheng, X. Ye, J. Luo, S. Chen, and L. Ren ExploreVLA: dense world modeling and exploration for end-to-end autonomous driving. In ECCV, Cited by: Table 1.

[^35]: C. Shi, J. Xu, S. Shi, K. Sheng, B. Zhang, and L. Jiang DriveWAM: video generative priors enable scalable world-action modeling for autonomous driving. arXiv preprint arXiv:2605.28544. Cited by: §2, Table 1.

[^36]: X. Tang, M. Kan, S. Shan, and X. Chen Plan-R1: safe and feasible trajectory planning as language modeling. In ICLR, Cited by: §2.

[^37]: H. Tian, T. Li, H. Liu, J. Yang, Y. Qiu, G. Li, J. Wang, Y. Gao, Z. Zhang, L. Wang, H. Ye, L. Chen, and H. Li SimScale: learning to drive via real-world simulation at scale. In CVPR, Cited by: §2.

[^38]: H. Wang, X. Ye, F. Tao, C. Pan, A. Mallik, B. Yaman, L. Ren, and J. Zhang AdaWM: adaptive world model based planning for autonomous driving. In ICLR, Cited by: §2.

[^39]: J. Wang, Z. Hua, X. Liu, Z. Xing, H. Tian, K. Ma, H. Ye, G. Chen, L. Chen, and Q. Zhang Beyond imitation: learning safe end-to-end autonomous driving from hard negatives. In ECCV, Cited by: Table 1.

[^40]: L. Wang, Y. Zheng, Q. Chen, S. Li, Y. Zhang, Z. Xing, Q. Zhang, X. Li, D. Qian, P. Yang, Y. Dong, C. Hao, X. Ye, J. Han, Y. Pan, and D. Zhao Latent-WAM: latent world action modeling for end-to-end autonomous driving. arXiv preprint arXiv:2603.24581. Cited by: Table 15, §2.

[^41]: X. Wang, Z. Zhu, G. Huang, X. Chen, J. Zhu, and J. Lu Drivedreamer: towards real-world-drive world models for autonomous driving. In ECCV, Cited by: §2.

[^42]: X. Wang, Y. Xiang, Y. Zhou, J. Wang, M. Huang, J. Huang, D. Wei, T. Zhou, X. Wang, G. Chen, Z. Xu, F. Tan, H. Zhou, and M. Yang WA-JEPA: rethinking the video JEPA paradigm for world-action modeling in autonomous driving. arXiv preprint arXiv:2608.20974. Cited by: Table 15, §2.

[^43]: T. Yan, H. Zheng, D. Chen, M. Qu, Y. Shen, L. Zhou, M. Tu, B. Wang, G. Chen, H. Ye, H. Sun, C. Xu, and J. Shen CausalDrive: real-time causal world models for autonomous driving. In ECCV, Cited by: §2.

[^44]: J. Yang, K. Chitta, S. Gao, L. Chen, Y. Shao, X. Jia, H. Li, A. Geiger, X. Yue, and L. Chen ReSim: reliable world simulation for autonomous driving. In NeurIPS, Cited by: §2.

[^45]: P. Yang, Y. Zheng, D. Qian, Z. Xing, Q. Zhang, L. Wang, Y. Zhang, S. Guo, Z. Xia, Q. Chen, J. Han, L. Xu, Y. Pan, and D. Zhao DreamerAD: efficient reinforcement learning via latent world model for autonomous driving. In ECCV, Cited by: Table 1, Table 2.

[^46]: Z. Yang, Y. Chai, X. Jia, Q. Li, Y. Shao, X. Zhu, H. Su, and J. Yan DriveMoE: mixture-of-experts for vision-language-action model in end-to-end autonomous driving. arXiv preprint arXiv:2505.16278. Cited by: §2.

[^47]: Z. Yang, X. Jia, Q. Li, X. Yang, M. Yao, and J. Yan Raw2Drive: reinforcement learning with aligned world models for end-to-end autonomous driving (in carla v2). In NeurIPS, Cited by: §2.

[^48]: W. Yao, Z. Li, S. Lan, Z. Wang, X. Sun, J. M. Alvarez, and Z. Wu Drivesuprim: towards precise trajectory selection for end-to-end planning. In AAAI, Cited by: Table 2.

[^49]: L. Yin, R. Ju, G. Guo, and E. Cheng DiffRefiner: coarse to fine trajectory planning via diffusion refinement with semantic interaction for end to end autonomous driving. In AAAI, Cited by: Table 2.

[^50]: T. Yuan, Z. Dong, Y. Liu, and H. Zhao Fast-WAM: do world action models need test-time future imagination?. arXiv preprint arXiv:2603.16666. Cited by: §2.

[^51]: D. Zhang, J. Liang, K. Guo, S. Lu, Q. Wang, R. Xiong, Z. Miao, and Y. Wang CarPlanner: consistent auto-regressive trajectory planning for large-scale reinforcement learning in autonomous driving. In CVPR, Cited by: §2.

[^52]: K. Zhang, Z. Tang, X. Hu, X. Pan, X. Guo, Y. Liu, J. Huang, L. Yuan, Q. Zhang, X. Long, et al. Epona: autoregressive diffusion world model for autonomous driving. In ICCV, Cited by: §2, Table 1, Table 2.

[^53]: Z. Zhao, T. Fu, Y. Wang, L. Wang, and H. Lu From forecasting to planning: policy world model for collaborative state-action prediction. In NeurIPS, Cited by: Table 1.

[^54]: Z. Zhao, X. Zhou, T. Xu, Z. Sun, K. Zhou, Y. Wu, H. Li, D. Liang, and X. Bai SimWAM: a simple world action model for end-to-end autonomous driving. arXiv preprint arXiv:2608.07468. Cited by: Table 15, §1, §2, Table 1, Table 2.

[^55]: Z. Zheng, S. Chen, H. Yin, X. Zhang, J. Zou, X. Wang, Q. Zhang, and L. Zhang ResAD: normalized residual trajectory modeling for end-to-end autonomous driving. In CVPR, Cited by: Table 1, Table 2.

[^56]: H. Zhou, L. Lin, J. Wang, Y. Lu, D. Bai, B. Liu, Y. Wang, A. Geiger, and Y. Liao HUGSIM: a real-time, photo-realistic and closed-loop simulator for autonomous driving. arXiv preprint arXiv:2412.01718. Cited by: §2, §4.1.

[^57]: Z. Zhou, T. Cai, S. Z. Zhao, Y. Zhang, Z. Huang, B. Zhou, and J. Ma AutoVLA: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. In NeurIPS, Cited by: §1, §2, Table 1.