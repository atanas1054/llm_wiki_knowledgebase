---
title: "RoG-DAgger: Rollout-Guided Post-Training for End-to-End Driving"
source: "https://arxiv.org/html/2608.24525v1"
author:
published:
created: 2026-10-28
description:
tags:
  - "clippings"
---
Liangyu Zhong    Joachim Sicking    Fabian Hueger    Hanno Gottschalk

###### Abstract

Recent end-to-end driving systems demonstrate strong performance on closed-loop benchmarks, yet are still predominantly trained on fixed expert-collected data using open-loop imitation learning. This training-inference mismatch leaves the policy vulnerable in policy-induced states, where accumulated errors can lead to safety-critical failures. A promising post-training approach to overcome this issue is Dataset Aggregation (DAgger), which gathers expert demonstrations in policy-induced states and subsequently fine-tunes the policy on the resulting aggregated dataset. Existing driving DAgger pipelines, however, face three challenges: i) the expert is restricted to a limited trajectory-and-speed solution space, ii) takeover may occur too early or too late relative to impending failures, and iii) privileged expert decisions may rely on information unavailable to the student. To address this, we introduce RoG-DAgger, a post-training framework that uses short-horizon kinematic rollouts to construct high-quality expert demonstrations in safety-critical states. Specifically, RoG-DAgger expands the expert’s trajectory-and-speed solution space and evaluates candidate plans through rollout to construct preventive supervision. Moreover, it uses rollout solvability to time the takeover near the estimated point of no return. Lastly, it aligns the expert’s field of view with that of the student to provide student-compatible supervision. Across in-distribution (including long-horizon) and out-of-distribution evaluations, RoG-DAgger improves the end-to-end model SimLingo by 5.3 driving-score points and 6.2 percentage points in success rate on Bench2Drive, doubles its driving score from 22 to 44 on Longest6 v2, and improves out-of-distribution success rate from 55% to 66% on Fail2Drive.

<sup>1</sup> CARIAD SE, Volkswagen Group

<sup>2</sup> Institute of Mathematics, Technical University of Berlin

## 1 Introduction

Recent end-to-end (E2E) driving systems [^20] [^6] [^16] have achieved strong results on closed-loop benchmarks, where predicted actions are executed in the simulator and influence subsequent observations. However, these models are still predominantly trained on fixed expert-collected data using open-loop imitation learning. They therefore remain vulnerable to the distribution shift between open-loop training and closed-loop execution [^14]. We study this issue using SimLingo [^20], a widely used camera-based E2E driving model. Evaluating SimLingo on the closed-loop Bench2Drive benchmark (B2D; see [^12]), we find that collisions account for roughly 80% of its failed routes (see App. A for details). These failures likely arise when small imitation errors propagate and compound over time, causing the policy to deviate from expert trajectories and visit states underrepresented in the training data, often referred to as compounding errors [^22]. To illustrate the resulting train–inference mismatch, we compare how frequently the model encounters safety-critical interactions, measured by the time-to-collision (TTC; see [^13]) distributions of SimLingo’s training data and its inference on B2D. As shown in Fig. 1, the SimLingo driving model encounters substantially lower TTC values during closed-loop inference than observed during training. This indicates that the model may learn neither how to prevent such near-collision events nor how to recover from collisions once they occur.

Figure 1: SimLingo pre-training data (orange) lacks safety-critical interactions as measured by time-to-collision (TTC), limiting the model’s ability to learn how to prevent or recover from such situations. As interactions, however, occur frequently in closed-loop Bench2Drive inference (blue), this constitutes a substantial distribution shift. Samples with TTC > 2s are omitted for clarity.

![[teaser 5.png|Refer to caption]]

Figure 2: RoG-DAgger leverages kinematic rollout and introduces three improvements to existing DAgger methods. Panel A shows how rollouts are used to construct trajectory-and-speed supervision and to evaluate candidate expert plans. Panels B1-B3 highlight three modifications: (1) an expanded trajectory-and-speed solution space (red: collision-prone nominal plan, green: improved plan), (2) solvability-aware takeover near the estimated point of no return (yellow star: PONR, red region: no feasible corrective plan remains), and (3) expert-student field-of-view alignment (yellow triangle) for student-learnable supervision.

Dataset Aggregation (DAgger; see [^23]) provides a promising way to mitigate this issue by running an expert model alongside a pretrained E2E policy in closed-loop simulation and fine-tuning the policy using the expert’s demonstrations. Here, the expert is a generally stronger privileged driving policy with access to scene information, such as surrounding-actor states and lane geometry. In driving DAgger, expert supervision is typically collected around *takeover events*, where control is transferred from the student policy to the expert. Recent methods such as TakeAD [^14] primarily collect post-takeover expert demonstrations, providing supervision for recovery from unsafe states. TakeVLA [^8] additionally retains pre-takeover frames and assigns warning-style language labels (e.g., “decelerate”) to encourage proactive avoidance, while further refining low-level behavior through reinforcement learning in reconstructed scenarios. However, TakeVLA’s pre-takeover labels remain event-level rather than state-specific, even though different states associated with the same warning may require substantially different trajectories and speed profiles. We therefore seek to provide explicit trajectory-and-speed supervision for each pre-takeover state by constructing counterfactual rollouts of expert-proposed candidate plans. Generating reliable preventive supervision, however, remains challenging due to three limitations of existing DAgger pipelines: i) a restricted expert trajectory-and-speed solution space, ii) takeover timing that may be premature or too late to prevent the impending failure, and iii) a misalignment between the fields of view of expert and student.

To address these limitations, we propose RoG-DAgger, a Rollout-Guided post-training framework for E2E driving (Fig. 3). RoG-DAgger uses kinematic rollouts (Fig. 2, A) to guide both expert supervision and takeover. First, we expand the expert’s trajectory-and-speed solution space with multiple candidate geometric trajectories and target speeds, and retain only rollout-solvable candidates that satisfy driving constraints (Fig. 2, B1). Second, we introduce a solvability-aware trigger that takes over near the boundary beyond which the impending failure can no longer be prevented (Fig. 2, B2). Finally, we align the fields of view of expert and student to reduce supervision that depends on information unavailable to the student, similar to [^16], see Fig. 2, B3.

![[overall.png|Refer to caption]]

Figure 3: Overview of the RoG-DAgger training pipeline. Starting from a pretrained E2E policy, we collect post-training data in two independent closed-loop runs (see zoom-in panel) and fine-tune the policy using a mixture of pre- and post-training data. The infraction-based run captures policy-induced failures and expert recovery demonstrations, while the solvability-aware run intervenes before failures become unavoidable to collect corrective behavior (gray hatched bar indicates avoided infraction). A restart trigger is used in both runs but omitted for clarity.

We evaluate RoG-DAgger by post-training SimLingo on the public CARLA Leaderboard 2.0 training routes [^2] and testing on three complementary closed-loop benchmarks: B2D for in-distribution evaluation, Longest6 v2 for long-horizon driving, and Fail2Drive for out-of-distribution generalization. On B2D, RoG-DAgger improves SimLingo by 5.3 driving score (DS) and 6.2 p.p. in success rate (SR), outperforming TakeAD and performing on par with TakeVLA. It further doubles SimLingo’s DS from 22 to 44 on Longest6 v2 and improves OOD SR from 55% to 66% on Fail2Drive.

Overall, our contributions are three-fold:

- We introduce kinematic ego and actor rollouts for constructing and assessing explicit pre-takeover trajectory-and-speed supervision for DAgger methods.
- We propose RoG-DAgger, a rollout-guided post-training framework that improves both expert supervision and takeover by expanding the expert solution space, identifying when preventive interventions remain feasible, and aligning the fields of view of expert and student.
- We demonstrate that RoG-DAgger improves closed-loop driving across complementary in-distribution (including long-horizon) and out-of-distribution settings, showing that rollout-guided supervision generalizes beyond the scenarios used for post-training.

## 2 Related Work

We first review representative end-to-end driving systems in Sec. 2.1. Online post-training approaches that enable these systems to perform better in closed-loop settings are discussed in Sec. 2.2.

### 2.1 End-to-End Driving Systems

End-to-end driving systems directly optimize the mapping from sensory observations to planned trajectories, enabling a fully differentiable pipeline that reduces reliance on hand-crafted interfaces between different driving modules. Earlier (non-language-based) approaches include TransFuser [^5] and UniAD [^11], which jointly optimize perception, prediction, and planning tasks within a planning-oriented framework.

Recently, vision-language models have increasingly been adapted to end-to-end driving, due to their improved scene understanding and broad world knowledge [^25] [^30] [^29] [^28], giving rise to vision-language-action models. Representative methods include SimLingo [^20], ORION [^6], and AutoVLA [^31] which have primarily focused on improving the alignment between visual observations, language instructions, and driving actions. However, despite the growing emphasis on closed-loop simulation in the community, these models are still primarily trained on fixed offline datasets and do not explicitly address the distribution shift between open-loop and closed loop (see introduction and [^14]). This motivates closed-loop post-training methods that adapt pretrained driving policies using feedback or supervision derived from their behavior during deployment.

### 2.2 Online Post-Training for E2E Systems

Unlike fixed offline training, online post-training refines a pretrained driving policy using data collected through closed-loop interaction, where the policy’s actions influence subsequent states, observations, and the responses of surrounding actors. This enables the policy to learn from its own closed-loop errors. Existing approaches mainly differ in their supervision signal. Reward-based methods, such as MindDrive [^7], optimize the policy using environment-derived feedback, e.g, traffic-infraction penalties. Expert-supervised methods commonly build on Dataset Aggregation (DAgger; see [^23]), which queries a privileged expert at policy-induced states and uses expert demonstrations for further policy optimization.

Driving DAgger methods can be further characterized along two axes: *which type of* expert supervision is collected and *when* expert takeover is triggered. TakeAD [^14] collects expert demonstrations around infraction- or discrepancy-triggered takeovers, primarily providing recovery supervision. TakeVLA [^8] additionally retains pre-takeover observations and assigns warning-style language labels to encourage preventive behavior. Their takeovers are determined by infraction events, expert-student discrepancies, or heuristic risk conditions, without explicitly assessing whether a feasible corrective plan remains at the current state. RoG-DAgger instead constructs state-specific trajectory-and-speed supervision for pre-takeover states and triggers takeover when rollout evaluation indicates that an impending collision can no longer be avoided by nominal emergency braking and therefore requires trajectory-and-speed replanning.

## 3 Rollout-Guided DAgger for E2E Driving

We first formulate the learning setup and overall RoG-DAgger framework in Sec. 3.1 and Sec. 3.2, respectively. We then introduce the kinematic rollout model used to evaluate candidate trajectory-and-speed plans in Sec. 3.3, followed by the rollout-guided expert supervision and takeover mechanisms in Sec. 3.4.

### 3.1 Problem Formulation

Driving policies in E2E driving map the current sensory observation, ego state, and navigational condition at time step $t$ to a future ego plan

$$
\hat{\mathbf{a}}_{t}=\Pi_{\theta}\left(\mathbf{C}_{t},\mathbf{S}_{t}^{\mathrm{ego}},\mathbf{O}_{t}\right),
$$

where $\Pi_{\theta}$ denotes the learned policy, $\mathbf{C}_{t}$ the navigational condition (e.g., GPS target points or language instructions), $\mathbf{S}_{t}^{\mathrm{ego}}$ the ego state (i.e., position and speed), and $\mathbf{O}_{t}$ the sensory observation, such as RGB images. Following SimLingo, we represent the predicted ego plan as

$$
\hat{\mathbf{a}}_{t}=\left(\hat{\mathbf{a}}^{\mathrm{geo}}_{t},\hat{\mathbf{a}}^{\mathrm{temp}}_{t}\right),
$$

where $\hat{\mathbf{a}}^{\mathrm{geo}}_{t}$ denotes the geometric trajectory, represented by a sequence of equidistantly sampled waypoints describing the intended path, and $\hat{\mathbf{a}}^{\mathrm{temp}}_{t}$ specifies the ego vehicle’s temporal progress along this path. During execution, the predicted trajectories are converted into steering, throttle, and braking commands using lateral and longitudinal PID controllers [^1].

Driving policies are commonly trained on expert demonstrations using an L1 imitation loss (IL)

$$
\mathcal{L}_{\mathrm{IL}}=\lambda^{\mathrm{geo}}\left\|\hat{\mathbf{a}}^{\mathrm{geo}}-\mathbf{a}^{\mathrm{geo,}}{{}^{*}}\right\|_{1}+\lambda^{\mathrm{temp}}\left\|\hat{\mathbf{a}}^{\mathrm{temp}}-\mathbf{a}^{\mathrm{temp},}{{}^{*}}\right\|_{1},
$$

where $\mathbf{a}^{*}$ denotes the expert supervision. During pre-training, this supervision is collected at expert-induced states, whereas DAgger [^23] queries the expert at states visited by the student policy, providing corrective supervision for policy-induced states. Following recent driving DAgger methods [^14] [^8], we selectively collect data around takeover events rather than at every policy step. We refer to states before and after takeover as *pre-* and *post-takeover* samples, respectively. Post-takeover trajectory supervision is directly constructed from the expert-controlled rollout, while our goal is to additionally construct state-specific trajectory-and-speed supervision for pre-takeover states.

### 3.2 RoG-DAgger Framework

We now provide an overview of our RoG-DAgger framework (Fig. 3). Starting from a pretrained student policy, we perform two independent closed-loop data-collection runs using *infraction-based* and *solvability-aware* takeover triggers, respectively. Running the two protocols separately is necessary because solvability-aware takeover changes the subsequent state distribution and may suppress failures that the student would otherwise encounter.

In the infraction-based run, takeover is triggered by collision, lane deviation, red-light or stop-sign violation. We retain the pre-takeover states leading to the failure together with post-takeover expert recovery demonstrations. In the solvability-aware run, in contrast, takeover occurs before an impending failure according to a rollout-based criterion (details in Sec. 3.4) and only post-takeover expert demonstrations are retained. In both runs, moreover, a restart trigger, similar in purpose to that used in TakeVLA [^8], is enabled when the ego vehicle fails to make expected progress for a prolonged period, with only post-takeover recovery samples being retained. Detailed trigger definitions are provided in App. B.2. The collected samples are subsequently used for fine-tuning with the imitation objective mentioned above, which yields the RoG-DAgger post-trained model.

In the following, we describe DoG-DAgger in greater detail, beginning with the kinematic rollout model used to construct preventive supervision and to assess candidate solvability.

### 3.3 Kinematic Ego and Actor Rollout

RoG-DAgger uses short-horizon rollouts of the ego vehicle and surrounding actors to evaluate candidate trajectory-and-speed plans. These rollouts support both preventive supervision for pre-takeover states and the solvability-aware takeover mechanism. We first describe the ego rollout, followed by the surrounding-actor rollout. Given the current ego state $\mathbf{s}_{t}^{\mathrm{ego}}$ (including speed $v_{0}$), a candidate geometric trajectory $\mathbf{a}_{t}^{\mathrm{geo}}$, and target speed $v_{t}^{\mathrm{target}}$, we define the ego rollout as $\mathbf{a}_{t}^{\mathrm{rollout}}=\mathcal{R}_{\mathrm{ego}}\left(\mathbf{s}_{t}^{\mathrm{ego}},\mathbf{a}_{t}^{\mathrm{geo}},v_{t}^{\mathrm{target}}\right),$ which represents the resulting counterfactual ego trajectory. Our rollout closely reflects the execution of candidate trajectory-and-speed plans. At each step, we apply the same lateral and longitudinal control logic used during deployment and propagate the resulting controls using the kinematic bicycle model and dynamics parameters from World on Rails [^3]. Crucially, the rollout reproduces the deployed braking condition, enabling realistic transitions between throttle and full braking. Unlike PDM-Lite [^32], which models throttle control without this braking behavior, our rollout more faithfully reflects the deployed longitudinal controller. Alg. 1 summarizes the procedure.

For surrounding actors, we follow the non-reactive rollout setting of PDM-Lite, propagating each actor over the same horizon while assuming unchanged low-level control. Together, the ego and actor rollouts provide the counterfactual future trajectories that are used to evaluate candidate trajectory-and-speed plans, as described in the following sections.

Algorithm 1 Controller-in-the-Loop Kinematic Ego Rollout

Inputs: ego state $\mathbf{s}_{0}^{\mathrm{ego}}$, geometric trajectory $\mathbf{a}^{\mathrm{geo}}$, target speed $v_{\mathrm{target}}$, planning horizon $H$, braking ratio $\rho$  
Output: ego rollout $\mathbf{a}_{\mathrm{rollout}}$

  $\mathbf{s}_{0}\leftarrow\mathbf{s}_{0}^{\mathrm{ego}},\;\mathbf{a}_{\mathrm{rollout}}\leftarrow\emptyset$

 for $k=0$ to $H-1$ do

   $\phi_{k}\leftarrow\textsc{LateralCtrl}(\mathbf{s}_{k},\mathbf{a}^{\mathrm{geo}})$

  if $v_{\mathrm{target}}\approx 0~\textbf{or}{}~v_{k}>\rho\ v_{\mathrm{target}}$ then

    $(u_{k}^{\mathrm{thr}},u_{k}^{\mathrm{brk}})\leftarrow(0,1)$

  else

    $(u_{k}^{\mathrm{thr}},u_{k}^{\mathrm{brk}})\leftarrow(\textsc{ThrottleCtrl}(v_{k},v_{\mathrm{target}}),0)$

  end if

   $\mathbf{s}_{k+1},v_{k+1}\leftarrow\textsc{BicycleModel}(\mathbf{s}_{k},\phi_{k},u_{k}^{\mathrm{thr}},u_{k}^{\mathrm{brk}})$    $\mathbf{a}_{\mathrm{rollout}}[k+1]\leftarrow\mathbf{s}_{k+1}$

 end for

 return $\mathbf{a}_{\mathrm{rollout}}$

### 3.4 Rollout-Guided Expert Supervision

We now use the rollout model from Sec. 3.3 and examine whether expert supervision assigned to pre-takeover states is actually preventive. In a diagnostic run on the CARLA LB2 training routes, we use CARLA collision infractions as takeover triggers and PDM-Lite [^32], a commonly used privileged expert with near-perfect performance on B2D and employed by prior driving DAgger work [^14] [^8]. For each pre-takeover state, we roll out the corresponding expert trajectory-and-speed plan and assess whether it prevents the impending collision.

Our analysis shows that roughly 30% of pre-takeover samples remain collision-prone under rollout (additional details are provided in App. B.1). This shows that expert supervision at pre-takeover states is not necessarily sufficient to prevent the impending collision. We identify three underlying limitations: the expert’s restricted trajectory-and-speed solution space, takeover timing that is not aligned with the impending failures, and a mismatch between privileged expert inputs and the student’s field of view. RoG-DAgger addresses these limitations by expanding the expert’s solution space, introducing solvability-aware takeover, and aligning the expert’s field of view with that of the student. We elaborate on each of these aspects in the following.

#### Expanding the Trajectory-and-Speed Solution Space

![[expert_comparison_main_text.png|Refer to caption]]

Figure 4: Comparison between experts with fixed and expanded sets of geometric trajectories. In (a), the expert follows a fixed pre-computed route and attempts to avoid the predicted collision through emergency braking. However, due to vehicle dynamics, the ego vehicle cannot decelerate sufficiently before reaching the collision point. In (b), the expert additionally considers alternative geometric trajectories, which are evaluated through rollouts for collision-freeness before selection. Geometric trajectories (red dots in the images) correspond to the routes executed in simulation. TTC and distance are estimated using our rollout model.

PDM-Lite follows a precomputed route and responds to predicted collisions with dynamic actors primarily by setting the target speed to zero, resulting in full braking along the nominal path. This is effective when longitudinal control suffices, but does not verify that braking prevents the collision and may fail when geometric adaptation is required (see representative examples of such failure cases in Fig. 4 and App. B.4). We therefore expand the expert’s trajectory-and-speed solution space and validate candidate plans through the rollout model.

When PDM-Lite predicts a collision requiring braking, we generate geometric candidates by laterally shifting the nominal trajectory $\mathbf{a}_{t}^{\mathrm{geo}}$. We first identify the waypoints corresponding to the predicted overlap between the ego and actor rollouts. We then shift these waypoints laterally, away from the actor’s direction of motion. We increase the offset in steps of $\Delta d$ up to $d_{\max}$, using cosine ease-in/ease-out transitions, and discard trajectories leaving the admissible lanes. For each geometric candidate $\mathbf{a}_{t,i}^{\mathrm{geo}}$, indexed by $i$, we evaluate a set of target speeds indexed by $j$. Outside junctions, we retain zero target speed. Inside junctions, where stopping in the conflict region may be undesirable, we additionally consider nonzero evasive speeds

$$
\mathcal{V}_{\mathrm{target}}=\begin{cases}\{0\},&\text{outside junctions},\\
\{0,\;0.25\;v_{t},\;0.5\;v_{t}\},&\text{inside junctions},\end{cases}
$$

and compute the corresponding ego rollout $\mathbf{a}_{t,i,j}^{\mathrm{rollout}}=\mathcal{R}_{\mathrm{ego}}\left(\mathbf{s}_{t}^{\mathrm{ego}},\mathbf{a}_{t,i}^{\mathrm{geo}},v_{t,j}^{\mathrm{target}}\right).$ We retain only collision-free, lane-compliant candidates, prioritizing higher target speed and then smaller lateral offset. If no valid candidate exists, we fall back to the nominal trajectory with full braking. Additional details regarding solution space expansion is provided in App. B.3.

#### Solvability-Aware Takeover Trigger

A richer expert solution space alone cannot prevent collision if takeover occurs too late. As the ego approaches a conflict, collision-free candidates may eventually disappear. We define the latest state with at least one collision-free candidate as the estimated *point of no return* (PONR). Because this boundary is difficult to derive analytically, we approximate it using rollout solvability.

Existing takeover triggers do not explicitly account for this solvability margin. TakeAD [^14] uses infraction- or expert-student discrepancy-based triggers, which may intervene after failure or for benign behavioral differences. TakeVLA [^8] additionally uses heuristic predictive triggers, such as constant-speed collision prediction, which may trigger prematurely without accounting for braking or alternative corrective plans. Premature takeover reduces exposure to policy-induced states, whereas overly late takeover may leave insufficient room for recovery.

We therefore aim to intervene near the estimated PONR while recovery remains possible. When PDM-Lite predicts a collision and initiates emergency braking, we first test with our kinematic rollout model whether nominal braking avoids it. If so, the student remains in control, otherwise, trajectory-and-speed replanning is required. Takeover is triggered when a collision-free replanning solution is available and either an imminent collision is detected (Replan Success) or the need for replanning persists over multiple timestamps within an interval for a less imminent collision (Repeated Replan Success). These triggers allow the expert to take over at highly critical yet still recoverable states, enabling the student to learn recovery behavior near the estimated PONR. For a limited number of samples with an imminent collision, no collision-free candidate exists (Replan Failure). In these cases, takeover with emergency braking without replanning is triggered as a fallback. For detailed definitions of trigger conditions and solvability-aware takeover examples, see Tab. 7 and App. B.5, respectively.

#### Field-of-View-Aligned Planning

Even a rollout-solvable expert plan may provide ineffective supervision if it depends on information unavailable to the student. Following [^16], we therefore restrict expert planning to surrounding actors within the student’s field of view.

DAgger introduces an additional issue when an infraction is caused by an actor outside the student’s field of view. In such cases, the observation-aligned expert also cannot condition its action on the responsible actor, so its pre-takeover trajectory does not provide meaningful supervision for preventing that event. For example, during a lane change, the vehicle responsible for a later collision may still be outside the student’s field of view in early pre-takeover states. We therefore discard pre-takeover samples whose responsible actor is not visible to the student, while retaining the associated post-takeover samples for recovery supervision. We refer to the resulting expert as RoG-Expert.

## 4 Experiments

We first introduce our experimental setup and implementation details in Sec. 4.1. We then present the performance of RoG-DAgger on in-distribution and out-of-distribution benchmarks in Secs. 4.2 and 4.3, respectively. Finally, we validate our design choices through ablation studies in Sec. 4.4.

### 4.1 Implementation Details

<table><tbody><tr><td rowspan="2">Method</td><td colspan="2">Bench2Drive</td><td colspan="2">Longest6 v2</td></tr><tr><td>DS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>SR <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>RC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td colspan="5">Privileged methods</td></tr><tr><td>PDM-Lite <sup><a href="#fn:32">32</a></sup></td><td>97.02</td><td>92.27</td><td>73</td><td>100</td></tr><tr><td>RoG-Expert (ours)</td><td>93.41</td><td>81.81</td><td>55</td><td>100</td></tr><tr><td colspan="5">Pre-trained-only methods</td></tr><tr><td>UniAD <sup><a href="#fn:11">11</a></sup></td><td>45.81</td><td>16.36</td><td>–</td><td>–</td></tr><tr><td>TF <math><semantics><mrow><mo>+</mo> <mo>⁣</mo> <mo>+</mo></mrow> <annotation>++</annotation></semantics></math> <sup><a href="#fn:32">32</a></sup></td><td>84.21</td><td>67.27</td><td>23</td><td>70</td></tr><tr><td>ORION <sup><a href="#fn:6">6</a></sup></td><td>77.74</td><td>54.62</td><td>–</td><td>–</td></tr><tr><td>AutoVLA <sup><a href="#fn:31">31</a></sup></td><td>78.84</td><td>57.73</td><td>–</td><td>–</td></tr><tr><td>HiP-AD <sup><a href="#fn:26">26</a></sup></td><td>86.77</td><td>69.09</td><td>7</td><td>56</td></tr><tr><td>SimLingo <sup><a href="#fn:20">20</a></sup></td><td>85.07</td><td>67.27</td><td>22</td><td>70</td></tr><tr><td colspan="5">Post-trained methods</td></tr><tr><td>MindDrive <sup><a href="#fn:7">7</a></sup></td><td>78.04</td><td>55.09</td><td>–</td><td>–</td></tr><tr><td>TakeAD <sup><a href="#fn:14">14</a></sup></td><td>71.39</td><td>40.83</td><td>–</td><td>–</td></tr><tr><td>TakeVLA <sup><a href="#fn:8">8</a></sup></td><td>89.72</td><td>73.73</td><td>–</td><td>–</td></tr><tr><td>RoG-DAgger (ours)</td><td>90.34</td><td>73.51</td><td>44</td><td>88</td></tr></tbody></table>

Table 1: Closed-loop performance on Bench2Drive and Longest6 v2. DS, SR, and RS denote Driving Score, Success Rate, and Route Completion, respectively. RoG-DAgger achieves strong performance across both benchmarks. Dash indicates an unreported result.

<table><tbody><tr><td rowspan="2">Method</td><td colspan="3">Base</td><td colspan="3">Generalization</td></tr><tr><td>DS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>SR <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>HM <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>SR <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>HM <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>UniAD</td><td>47.5</td><td>36.3</td><td>41.2</td><td>44.0</td><td>27.6</td><td>33.9</td></tr><tr><td>TF <math><semantics><mrow><mo>+</mo> <mo>⁣</mo> <mo>+</mo></mrow> <annotation>++</annotation></semantics></math></td><td>83.3</td><td>78.5</td><td>80.8</td><td>75.4</td><td>61.1</td><td>67.5</td></tr><tr><td>ORION</td><td>53.0</td><td>52.0</td><td>52.5</td><td>51.2</td><td>46.0</td><td>48.5</td></tr><tr><td>HiP-AD</td><td>74.1</td><td>70.7</td><td>72.4</td><td>67.1</td><td>56.7</td><td>61.5</td></tr><tr><td>SimLingo</td><td>82.6</td><td>79.3</td><td>80.9</td><td>71.7</td><td>55.0</td><td>62.2</td></tr><tr><td>RoG-DAgger (ours)</td><td>80.2</td><td>80.0</td><td>80.1</td><td>74.7</td><td>66.0</td><td>70.1</td></tr></tbody></table>

Table 2: Closed-loop performance on Fail2Drive. Base denotes in-distribution evaluation, while Generalization refers to distribution-shifted scenarios. DS, SR, and HM denote Driving Score, Success Rate, and their harmonic mean, respectively. RoG-DAgger remains competitive on base scenarios while showing improved OOD generalization over SimLingo.

<table><tbody><tr><td colspan="4">Ablation</td><td colspan="2">Bench2Drive</td></tr><tr><td></td><td>Traj.- speed</td><td>Solv.- aware</td><td>FoV alignment</td><td>DS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>SR <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>Component</td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td><math><semantics><mo>+</mo> <annotation>+</annotation></semantics></math></td><td><math><semantics><mo>+</mo> <annotation>+</annotation></semantics></math></td><td>88.71</td><td>68.94</td></tr><tr><td></td><td><math><semantics><mo>+</mo> <annotation>+</annotation></semantics></math></td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td><math><semantics><mo>+</mo> <annotation>+</annotation></semantics></math></td><td>88.65</td><td>69.86</td></tr><tr><td></td><td><math><semantics><mo>+</mo> <annotation>+</annotation></semantics></math></td><td><math><semantics><mo>+</mo> <annotation>+</annotation></semantics></math></td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td>88.96</td><td>69.86</td></tr><tr><td>Takeover data</td><td colspan="3">Post-takeover only</td><td>88.05</td><td>68.04</td></tr><tr><td colspan="4">PDM-Lite DAgger baseline</td><td>87.85</td><td>66.66</td></tr><tr><td colspan="4">RoG-DAgger full</td><td>90.34</td><td>73.51</td></tr></tbody></table>

Table 3: Ablation study on Bench2Drive. We evaluate the effects of trajectory-and-speed space expansion, solvability-aware triggering, FoV alignment, takeover data selection and expert model.

![[qua_example_final.png|Refer to caption]]

Figure 5: Qualitative comparison of RoG-DAgger with SimLingo on a Bench2Drive scene. At t 1 t\_{1}, both policies approach a junction while a cross-traffic fire truck is already visible. Both policies continue to accelerate, likely because the pre-training data contains primarily successful junction traversals and lacks corrective supervision for such hazardous interactions. As a result, SimLingo keeps accelerating at 2 t\_{2} and collides with the fire truck at 3 t\_{3}. In contrast, after RoG-DAgger post-training, the policy slows down at and shifts to the left, avoiding the collision at. Red dots denote the predicted geometric trajectory, while green dots denote the predicted temporal trajectory. “thr”: throttle, “brk”: brake.

Following prior work [^8], we use SimLingo as the pretrained E2E driving policy. In our implementation we replace the original InternVL-2 backbone [^4] with Qwen3-VL-2B [^19], which yielded comparable performance in our preliminary evaluation. Further details on the model are provided in App. A.

We then deploy the pretrained policy on the CARLA LB2 training routes [^2] and perform online post-training using RoG-DAgger as described in Sec. 3. Following [^14], we reduce data-collection and fine-tuning costs by focusing on routes where the pretrained policy performs poorly and applying rule-based filtering (see App. B.6) to remove uninformative samples, such as cases where the ego vehicle remains stuck in traffic. For infraction-triggered events, we retain a 40-frame (2s) pre-takeover window and a 40-frame (2s) post-takeover window. For solvability-aware and restart triggers, we retain only post-takeover observations: 40 frames (2s) for solvability-aware interventions and 80 frames (4s) for restart events to capture the full recovery process. Overall, we collect approximately 51K post-training samples and mix them with the original SimLingo pretraining data at a 3:1 post-to-pre ratio to mitigate catastrophic forgetting [^21]. Details on the distribution of the collected post-training data are reported in App. C.

We fine-tune the policy for five epochs using AdamW [^15], with a batch size of 24 and a learning rate of $10^{-5}$. We evaluate on three closed-loop benchmarks: Bench2Drive (B2D; see [^12]), CARLA Longest6 v2 [^2], and Fail2Drive (F2D; see [^9]). B2D evaluates in-distribution (ID) driving across diverse short routes, while Longest6 v2 stresses long-horizon closed-loop robustness with routes approximately $10\times$ longer and an additional penalty for driving substantially slower than local traffic. F2D finally evaluates out-of-distribution (OOD) generalization under unseen assets and novel scenario configurations. Following each benchmark’s practices, we report Driving Score (DS) and Success Rate (SR) on B2D and F2D, and DS as well as Route Completion (RC) on Longest6 v2, where failure-free completion is substantially harder. Details of metrics are provided in App. D.1.

### 4.2 ID and Long-Horizon Benchmarks

We report results on B2D and Longest6 v2 in Tab. 1. Despite its richer solution space, RoG-Expert is outperformed by PDM-Lite on both benchmarks. This is partly expected because RoG-Expert restricts privileged hazard information to the student’s field of view, sacrificing some standalone expert performance in favor of student-compatible supervision. Nevertheless, post-training substantially improves the base policy. On B2D, RoG-DAgger improves over SimLingo by 5.3 p.p. in DS and 6.2 p.p. in SR, reaching 90.34 DS and 73.51 SR. This demonstrates that our post-training procedure effectively narrows the open-loop-closed-loop distribution gap. Compared to other post-training methods, RoG-DAgger clearly outperforms TakeAD and MindDrive and performs comparably to TakeVLA.

The gains are more pronounced on Longest6 v2, where RoG-DAgger doubles SimLingo’s DS from 22 to 44 and improves RC from 70 to 88, substantially outperforming the other non-privileged methods. Notably, on Longest6 v2, pretrained SimLingo achieves 22 DS compared with 73 DS for its privileged data-generation expert, PDM-Lite, leaving a 51-point student-expert gap. In contrast, RoG-DAgger reaches 44 DS despite being post-trained with the substantially weaker RoG-Expert, which achieves 55 DS, leaving only an 11-point gap to its corresponding supervision expert. This observation motivates our analysis in Sec. 4.4 of why standalone expert performance does not necessarily translate into effective post-training supervision.

In Fig. 5, we provide a qualitative comparison of RoG-DAgger with SimLingo on a B2D scene where RoG-DAgger decelerates and shifts left to safely avoid the cross-traffic, whereas SimLingo continues accelerating and ultimately collides. Further qualitative examples and more detailed B2D metrics can be found in App. D.2.

### 4.3 Out-of-Distribution Benchmarks

In addition to in-distribution evaluation, we assess OOD generalization on Fail2Drive (F2D). As shown in Tab. 2, RoG-DAgger performs comparably to SimLingo on the base scenarios, while TF++ achieves the highest DS on both base and generalization scenarios, potentially benefiting from its LiDAR input.

On the generalization counterpart, RoG-DAgger outperforms SimLingo by about 3 p.p. in DS and 11 p.p. in SR. In challenging scenarios such as FullyBlocked, RoG-DAgger slows down and stops when the road is obstructed, whereas SimLingo often maintains speed and collides with the blocking vehicle. Although this exact scenario is absent in the CARLA LB2, we hypothesize that by learning from preventive pre-takover samples encountered under its own policy, the model acquires more general behaviors, such as slowing down or stopping when an obstacle blocks the drivable path, rather than memorizing responses to specific training scenarios. Additional qualitative examples and more detailed F2D metrics are provided in App. D.3.

### 4.4 Ablation Studies

In this section, we validate our design choices on B2D and summarize the respective results in Tab. 3. We use the hyperparameters from Sec. 4.1 unless stated otherwise.

#### RoG-Expert Component Analysis

We ablate the three main components of RoG-DAgger individually in the upper part of Tab. 3. Removing any component degrades both DS and SR relative to the full method (90.34 DS / 73.51 SR), demonstrating that the three design choices provide complementary benefits. The three ablated variants achieve broadly similar DS, suggesting that no single component dominates the overall improvement. Removing trajectory-speed solution space expansion produces the largest SR drop to 68.94, highlighting the importance of geometric adaptation when collision avoidance cannot be achieved by speed adjustment alone along the precomputed route. Disabling solvability-aware takeover and FoV alignment reduces SR to 69.86.

#### Takeover Data

Next, we evaluate the contribution of preventive pre-takeover supervision by training with post-takeover samples only. Removing pre-takeover samples reduces the B2D SR from 73.51 to 68.04. The larger degradation in SR suggests that pre-takeover supervision is particularly important for preventing failures, rather than only learning to recover after an unsafe state has already been reached. This supports explicitly supervising the policy on the states leading up to safety-critical events.

#### PDM-Lite DAgger Baseline

We further compare against a conventional DAgger configuration using PDM-Lite as the takeover expert together with the infraction- and restart-based triggers. This reduces B2D DS from 90.34 to 87.85 and SR from 73.51 to 66.66. Interestingly, although PDM-Lite achieves higher standalone closed-loop performance than RoG-Expert, its corresponding DAgger pipeline produces a weaker student. This suggests that privileged expert performance alone is not a sufficient proxy for supervision quality: effective post-training additionally requires corrective plans that are feasible, timely, and compatible with the student’s observations. Since this baseline also removes the solvability-aware intervention used by RoG-DAgger, the comparison reflects the combined effect of expert design and takeover strategy rather than expert choice alone.

## 5 Discussion

In this work, we introduced RoG-DAgger, a post-training framework that improves pre-trained driving policies through closed-loop expert supervision. To mitigate the open-loop-closed-loop distribution shift, RoG-DAgger extends conventional DAgger with trajectory-and-speed solution-space expansion, solvability-aware triggering, and expert-student field-of-view alignment, while validating expert demonstrations through kinematic rollouts. Applied to SimLingo, RoG-DAgger consistently improves closed-loop performance and is competitive on B2D, while being substantially stronger under long-horizon and OOD evaluation. These results demonstrate that learning from policy-induced states with reliable expert supervision can substantially improve closed-loop driving performance, particularly in challenging states that are underrepresented during pre-training, highlighting a promising direction for post-training of autonomous agents, e.g., for general robotics.

Although RoG-DAgger demonstrates strong performance across benchmarks, its rollout-based solvability estimation remains approximate. Following PDM-Lite, our actor rollout assumes that surrounding actors remain non-reactive over a short rollout time horizon, which can lead to inaccurate forecasts when they change their driving intentions in response to the ego vehicle or the scene. In addition, although RoG-DAgger expands the expert’s trajectory-and-speed solution space, the considered candidate set may not contain all feasible collision-avoidance solutions. Consequently, the estimated point of no return should not be interpreted as a physical feasibility boundary: a candidate trajectory classified as solvable may lead to a different outcome when surrounding actors react, while an unsuccessful search does not rule out feasible solutions outside the considered candidate space. Extending RoG-DAgger with interaction-aware actor prediction, including robust uncertainty estimates [^24] [^18] and richer trajectory generation is an important direction for future work.

A further constraint of our method is its reliance on CARLA closed-loop simulation to explore policy-induced states together with their corresponding observations, which limits direct application to real-world data. Extending rollout-guided post-training to high-fidelity data-driven simulators (e.g., AlpaSim [^17]) is a natural direction for future research.

## 6 Acknowledgments

Hanno Gottschalk acknowledges financial support by the German Federal Ministry of Economic Affairs and Energy under grant no. 19A23014Q within the nxtAIM consortium.  

## References

Supplementary Material for  
RoG-DAgger: Rollout-Guided Post-Training for End-to-End Driving Written by AAAI Press Staff <sup>1</sup> With help from the AAAI Publications Committee.  
AAAI Style Contributions by Peter Patel Schneider, Sunil Issar,  
J. Scott Penberthy, George Ferguson, Hans Guesgen, Francisco Cruz, Marc Pujol-Gonzalez

## Appendix A Model Architecture and Pre-Training

SimLingo [^20] operates on a front-facing camera view with a horizontal field of view (FoV) of $110^{\circ}$, which defines the visual context available to the policy for scene understanding and planning. For motion prediction, the policy outputs 10 trajectory waypoints together with 10 corresponding speed waypoints, jointly representing the spatial path and desired speed profile over the prediction horizon. These planning outputs are decoded from the learned visual-language representation using lightweight MLP heads for trajectory and speed prediction.

With the model architecture established, we next describe the training procedure and optimization settings used for fine-tuning. We pre-train the model following the SimLingo training recipe, while keeping all other unspecified settings identical to those in SimLingo. Training is performed for 5 epochs with a learning rate of $1\times 10^{-4}$ and an effective batch size of 48 across 4 GPUs, corresponding to 12 samples per GPU. We apply LoRA [^10] with rank $r=32$ and scaling factor $\alpha=64$. Although SimLingo enables chain-of-thought [^27] output by default, the original paper reports no notable performance degradation when it is disabled [^20]. We observe the same behavior in our implementation and thus disable the chain-of-thought output in all experiments to reduce inference latency. Moreover, we replace the InternVL backbone used in (Renz et al. 2025) with a more recent Qwen3-VL-2B-Instruct. As shown in Tab. 4, however, we observe no significant performance difference between the original SimLingo with an InternVL backbone and our Qwen3-VL-based version on B2D. Throughout the paper, we use the latter version and refer to it simply as SimLingo. We analyze the B2D failure cases in Tab. 5 and find that collisions account for roughly 80% of the analyzed failures. We exclude the YieldToEmergencyVehicle scenario from this analysis because SimLingo lacks a rear-facing camera and therefore cannot reliably detect and proactively yield to an emergency vehicle approaching from behind.

| Model | DS | SR (%) |
| --- | --- | --- |
| SimLingo [^20] | 85.07 | 67.27 |
| SimLingo-Qwen3-VL-2B (ours) | 86.59 | 66.36 |

Table 4: Comparison between the original SimLingo and our Qwen3-VL-2B-based version. Both variants achieve comparable driving performance despite the change in VLM backbone. DS and SR denote driving score and success rate, respectively.

| Failure Category | Count | Proportion (%) |
| --- | --- | --- |
| Total recorded failures | 73 | – |
| Ignored cases | 5 | – |
| Analyzed failures | 68 | 100.0 |
| Collision-related failures | 53 | 77.9 |
| Collision with vehicle | 49 | 72.1 |
| Collision with static | 4 | 5.9 |
| Collision with pedestrian | 1 | 1.5 |
| Non-collision failures | 15 | 22.1 |

Table 5: Breakdown of the recorded failures. After excluding five Yield To Emergency Vehicle cases, we analyze 68 failures, of which 53 are collision-related. The collision subcategories are not mutually exclusive, as one failure involves both a vehicle and a static object.

## Appendix B Implementation Details of RoG-DAgger

This section provides additional implementation details of RoG-DAgger. We first present a preliminary DAgger experiment with PDM-Lite that motivates our design choices in App. B.1, then describe the data-collection trigger conditions and the trajectory-and-speed solution-space expansion used by RoG-Expert in App. B.2 and B.3, respectively. We subsequently provide qualitative examples of geometric trajectories in App. B.4 and solvability-aware takeovers in App. B.5, followed by details of the rule-based data-filtering procedure in App. B.6 and other implementation settings in App. B.7.

### B.1 Preliminary DAgger Experiment with PDM-Lite

As described in Sec. 3.4, we conduct a diagnostic DAgger run using PDM-Lite as the expert policy and CARLA collision events as the takeover trigger. For each pre-takeover sample, we query PDM-Lite for a counterfactual trajectory-and-speed plan from the corresponding policy-induced state. We then execute this plan with our controller-in-the-loop kinematic rollout model and determine whether the resulting ego temporal trajectory overlaps with the predicted temporal trajectory of any surrounding actor. Such an overlap is counted as a predicted collision. We evaluate pre-takeover windows of 20 frames, following TakeVLA [^8], and 40 frames, corresponding to our default setting, while fixing the post-takeover window to 40 frames. As shown in Tab. 6, the counterfactual PDM-Lite plans produce predicted collision rates of 43.2% and 29.6% for the 20-frame and 40-frame pre-takeover windows, respectively, compared to only 0.47% for the post-takeover samples. These results indicate that expert actions generated before takeover are frequently unsafe in policy-induced states. As discussed in Sec. 3.4, this discrepancy can likely be attributed to the expert’s restricted solution space, delayed intervention timing, and misalignment between the expert and student policies.

| Split | Window | Samples | Collision | Rate |
| --- | --- | --- | --- | --- |
| Pre-takeover | 20 | 5568 | 2404 | 43.18% |
| Pre-takeover | 40 | 9565 | 2829 | 29.58% |
| Post-takeover | 40 | 20845 | 99 | 0.47% |

Table 6: Collision statistics for pre- and post-takeover samples. For pre-takeover samples, we compare temporal windows of 20 and 40 frames preceding the trigger, while the post-takeover window is 40. Collision rate denotes the fraction of retained samples for which our rollout model predicts a collision within the corresponding window when executing the trajectory-and-speed plan provided by PDM-Lite.

Algorithm 2 Trajectory-and-speed solution-space expansion. PDM-Collision denotes PDM-Lite’s native collision prediction used to trigger emergency braking. It returns the collision-relevant actor $o$ and the index set $\mathcal{I}$ of nominal-trajectory waypoints involved in the predicted ego-actor overlap. $\textsc{Local-Shift}(\mathbf{a},\mathcal{I},o,d)$ shifts only the waypoints of ego trajectory $\mathbf{a}$ indexed by $\mathcal{I}$ by distance $d$ along the direction perpendicular to the local trajectory, choosing the side away from actor $o$ ’s direction of motion, with smooth ease-in/ease-out transitions to the unchanged trajectory. $\mathcal{R}_{\mathrm{ego}}$ denotes the controller-in-the-loop ego rollout model (see Sec. 3.3). Replanning is targeted to avoid $o$, while candidate plans are validated against all predicted actors $\mathbf{A}_{t}$.

Inputs: ego state $\mathbf{s}_{t}^{\mathrm{ego}}$, nominal trajectory $\mathbf{a}_{t}^{\mathrm{geo}}$, current speed $v_{t}$, predicted actors $\mathbf{A}_{t}$, shift unit size $\Delta d$, maximum shift $d_{\max}$  
Output: expert plan $(\mathbf{a}_{\mathrm{geo}}^{*},\mathbf{a}_{\mathrm{temp}}^{*})$ or $\varnothing$

  $(o,\mathcal{I})\leftarrow\textsc{PDM-Collision}(\mathbf{s}_{t}^{\mathrm{ego}},\mathbf{a}_{t}^{\mathrm{geo}},\mathbf{A}_{t})$

 if $o=\varnothing$ then

  return $\varnothing$

 end if

  $\mathbf{a}_{\mathrm{temp}}^{\mathrm{brk}}\leftarrow\mathcal{R}_{\mathrm{ego}}(\mathbf{s}_{t}^{\mathrm{ego}},\mathbf{a}_{t}^{\mathrm{geo}},0)$

 if $\textsc{Collision-Free}(\mathbf{a}_{\mathrm{temp}}^{\mathrm{brk}},\mathbf{A}_{t})$ then

  return $(\mathbf{a}_{t}^{\mathrm{geo}},\mathbf{a}_{\mathrm{temp}}^{\mathrm{brk}})$

 end if

 if $\mathbf{s}_{t}^{\mathrm{ego}}$ is inside a junction then

   $\mathcal{V}_{\mathrm{target}}\leftarrow\{0.5\,v_{t},\,0.25\,v_{t},\,0\}$

 else

   $\mathcal{V}_{\mathrm{target}}\leftarrow\{0\}$

 end if

 for $v_{\mathrm{target}}\in\mathcal{V}_{\mathrm{target}}$ do

  for $d=\Delta d,\,2\Delta d,\ldots,d_{\max}$ do

    $\mathbf{a}_{\mathrm{geo}}^{\prime}\leftarrow\textsc{Local-Shift}(\mathbf{a}_{t}^{\mathrm{geo}},\mathcal{I},o,d)$

   if $\textsc{Lane-Compliant}(\mathbf{a}_{\mathrm{geo}}^{\prime})$ then

     $\mathbf{a}_{\mathrm{temp}}^{\prime}\leftarrow\mathcal{R}_{\mathrm{ego}}(\mathbf{s}_{t}^{\mathrm{ego}},\mathbf{a}_{\mathrm{geo}}^{\prime},v_{\mathrm{target}})$

    if $\textsc{Collision-Free}(\mathbf{a}_{\mathrm{temp}}^{\prime},\mathbf{A}_{t})$ then

     return $(\mathbf{a}_{\mathrm{geo}}^{\prime},\mathbf{a}_{\mathrm{temp}}^{\prime})$

    end if

   end if

  end for

 end for

 return $(\mathbf{a}_{t}^{\mathrm{geo}},\mathbf{a}_{\mathrm{temp}}^{\mathrm{brk}})$

### B.2 Trigger Conditions

As described in Sec. 3.2, we conduct two independent data-collection runs using three types of triggers. The corresponding trigger conditions are detailed in Tab. 7.

| Trigger | Trigger Condition |
| --- | --- |
| Infraction | A standard CARLA infraction, including a collision, red-light violation, or out-of-lane violation. |
| Solvability-Aware Takeover | Replan Success:   $\mathrm{R\text{-}TTC}<1\,\mathrm{s}$ $\mathtt{AND}$ replanning succeeds. Repeated Replan Success:   $1\,\mathrm{s}<\mathrm{R\text{-}TTC}<1.5\,\mathrm{s}$ $\mathtt{AND}$   replanning succeeds in two of four consecutive frames. Replan Failure:   $\mathrm{R\text{-}TTC}<1\,\mathrm{s}$ $\mathtt{AND}$ replanning fails. |
| Restart | A persistent discrepancy is detected between the ego vehicle and the expert model: the ego speed remains below $0.1\,\mathrm{m/s}$ for all of the previous 200 frames, while both the mean expert target speed over these frames and the current expert target speed exceed $0.4\,\mathrm{m/s}$. |

Table 7: Details on the three types of trigger conditions used for post-training data collection. R-TTC denotes the time-to-collision estimated by our rollout model.

### B.3 Trajectory-and-Speed Solution-Space Expansion

When full braking alone is insufficient, we expand the solution space by generating geometric candidates around the predicted ego-actor overlap, as summarized in Alg. 2. Replanning is triggered when the nominal PDM-Lite collision detector requests an emergency stop, which is the case if the ego vehicle is moving faster than $0.4\mathrm{m/s}$, while the target speed falls below $10^{-3}\mathrm{m/s}$. Collision prediction is then performed assuming a target speed of zero.

Given the resulting predicted collision interval, we locally modify the affected portion of the nominal trajectory rather than shifting the entire route. In particular, the modified segment starts approximately in a distance of

$$
5\,\mathrm{m}+l_{\mathrm{half\_ego}}+l_{\mathrm{half\_actor}}
$$

before the first predicted ego-actor overlap and extends approximately

$$
2\,\mathrm{m}+l_{\mathrm{half\_ego}}+l_{\mathrm{half\_actor}}
$$

beyond the final overlap, where $l_{\mathrm{half\_ego}}$ and $l_{\mathrm{half\_actor}}$ denote the longitudinal half-extents of the ego vehicle and collision-relevant actor, respectively. The $5\,\mathrm{m}$ pre-collision margin allows the avoidance maneuver to begin before entering the predicted conflict region, while the $2\,\mathrm{m}$ post-collision extension avoids a sudden change in trajectory curvature and steering.

Within this segment, we shift the nominal trajectory laterally, i.e., perpendicular to the local ego-route direction and away from the collision-relevant actor. To avoid abrupt geometric changes, the lateral displacement is introduced and removed using a spatial cosine ease-in/ease-out transition over 2 $\mathrm{m}$. Specifically, we define the transition weight as

$$
E(u)=\frac{1-\cos(\pi u)}{2},\qquad u\in[0,1],
$$

where $u$ denotes the normalized progress through the transition interval. During the ease-in transition, the weight increases smoothly from $E(0)=0$ to $E(1)=1$, and the interpolated trajectory is given by

$$
\bigl(1-E(u)\bigr)~\mathbf{a}_{\mathrm{geo}}+E(u)~\mathbf{a}^{\prime}_{\mathrm{geo}}\ ,
$$

where $\mathbf{a}^{\prime}_{\mathrm{geo}}$ denotes the shifted trajectory. For the ease-out transition, the same weighting profile is applied in reverse, smoothly returning the shifted trajectory to the nominal trajectory. As a result, the trajectory gradually approaches the selected lateral displacement, remains shifted through the central collision-avoidance region, and smoothly merges back into the nominal trajectory after the predicted conflict.

As in Alg. 2, we search over increasing lateral offsets and retain the first candidate whose rollout is both collision-free and lane-compliant. In the implementation, offsets $\Delta d$ are evaluated in $0.2~\mathrm{m}$ increments until $d_{\max}=3~m$, thereby favoring the smallest geometric deviation that resolves the predicted collision. A candidate is considered lane-valid only if its lateral deviation remains within approximately

$$
\frac{w_{\mathrm{lane}}}{2}+0.2\,\mathrm{m},
$$

where $w_{\mathrm{lane}}$ is the local lane width. Candidates outside this admissible region are discarded. We first consider target-speed options in descending order and, for each target speed, evaluate geometric candidates with progressively increasing lateral offsets using the controller-in-the-loop rollout model $\mathcal{R}_{\mathrm{ego}}$, following Alg. 1. Candidates are retained only when their temporal rollout remains collision-free with respect to all predicted surrounding actors. We prioritize higher target speeds and then smaller lateral deviations. If no valid shifted candidate exists, we fall back to the nominal geometric trajectory with full braking.

### B.4 Examples of Geometric Trajectory Shifting

In the main paper, we discuss why a restricted trajectory-and-speed solution space can lead an expert model to produce suboptimal actions. Here, we provide a concrete example in Figs. 7 and 7. We evaluate two privileged experts in the MergeIntoSlowTraffic scenario. The expert in Fig. 7 follows the default PDM-Lite strategy: it tracks a precomputed geometric trajectory and attempts to avoid collisions solely through full braking, i.e., by setting the target speed to zero. In contrast, the RoG-Expert in Fig. 7 uses an expanded trajectory-and-speed solution space. While the default expert fails to complete the scenario safely, the RoG-Expert with the expanded solution space resolves the traffic conflict by combining adapted geometric trajectory and full braking. This comparison shows that an expert with a restricted trajectory-and-speed solution space can itself become the performance bottleneck, as it may fail to provide a safe corrective action even with privileged scene information.

![[no_replan.png|Refer to caption]]

Figure 6: Expert without expanded trajectory-and-speed solution space fails to prevent collision. An expert such as PDM-Lite follows a precomputed route and primarily responds to predicted collisions with full braking along the nominal path. However, in safety-critical states, especially those induced by the policy, braking alone does not necessarily prevent an impending collision. At t 1 t\_{1}, the ego vehicle controlled by expert follows the nominal route and prepares for lane change. At 2 t\_{2}, the expert predicts a collision and initiates full braking, but the intervention is insufficient due to the vehicle dynamics, and a collision occurs at 4 t\_{4}. The geometric trajectories shown as red dots in the lower row correspond to the trajectories illustrated in the upper row.

### B.5 Solvability-Aware Takeover Examples

We illustrate the solvability-aware takeover mechanism in Fig. 8. As the ego vehicle approaches the junction, even a small deviation in the turning angle can lead to a collision. The solvability-aware trigger activates RoG-Expert at a highly critical yet still recoverable state, close to the estimated point of no return (PONR). RoG-Expert then corrects the trajectory in time to avoid the impending accident. By collecting supervision near this boundary, the student is exposed to recovery behavior in states where intervention is necessary but successful correction remains feasible.

![[lb2_et.png|Refer to caption]]

Figure 8: SimLingo with solvability-aware takeover. At t 1 t\_{1}, the policy controls the ego vehicle while preparing for a left turn. As oncoming traffic approaches, our kinematic rollout model predicts at 2 t\_{2} a time-to-collision (TTC) of 0.05 s 0.05\\,\\mathrm{s}, indicating an imminent collision. This activates the solvability-aware takeover, which shifts the geometric trajectory 0.5 m 0.5\\,\\mathrm{m} to the right while setting the target speed to 50% of the current speed. At 3 t\_{3}, the RoG-Expert applies a second, larger correction of 0.9 0.9\\,\\mathrm{m} to the right, again setting the target speed to 50% of the current speed, successfully avoiding the collision. At 4 t\_{4}, the ego vehicle continues safely under expert control during the post-takeover period.

### B.6 Rule-Based Training Data Filtering

Following the filtering principle used in TakeAD [^14], we retain collected samples that exhibit a meaningful discrepancy between the policy and the expert model, while removing uninformative low-motion cases. A sample is considered relevant when the target-speed disagreement exceeds 0.5 m/s, the current speed exceeds the target speed by more than 10%, or the absolute steering deviation is at least 0.2 rad. Speed and steering criteria are combined with OR logic, such that a disagreement in either control dimension is sufficient to retain the sample. Similarly, we filter stuck or near-stationary behavior: events with an average speed below 0.5 m/s are removed, and post-takeover segments are discarded when both their average and final speeds remain below 0.3 m/s.

In addition to these TakeAD-inspired filters, we apply an observability-based filter specific to our recovery setting. As discussed in Sec. 3.4, pre-takeover collision frames are retained only when the collision-relevant actor is visible in the camera view. This prevents samples from being included when the relevant actor lies outside the model’s observable field of view.

### B.7 Further Implementation Details

For the expert target speed, we follow LEAD [^16], i.e., the target speed is capped by the minimum of the posted speed limit and the typical flow speed of nearby vehicles. In addition, we use PDM-Lite’s default bounding-box inflation to account for uncertainty in both the predicted actor rollouts and the estimated execution of the ego action.

## Appendix C Distribution of Post-Training Data

Tab. 8 summarizes the composition of the RoG-DAgger post-training dataset from two complementary perspectives. Panel (a) reports the proportion of samples collected before and after takeover, with the majority originating from the post-takeover phase. Panel (b) groups samples by their high-level collection trigger and further breaks them down into the corresponding event types (including collisions, red-light and out-of-lane infractions), different outcomes of the solvability-aware takeover, and restart events.

<table><tbody><tr><td colspan="2">(a) Takeover Phase</td></tr><tr><td>Pre-takeover</td><td>27.3%</td></tr><tr><td>Post-takeover</td><td>72.7%</td></tr><tr><td colspan="2">(b) Collection Trigger / Event Type</td></tr><tr><td>Infraction</td><td>55.3%</td></tr><tr><td>   Collision</td><td>47.7%</td></tr><tr><td>   Red-Light Infraction</td><td>2.9%</td></tr><tr><td>   Out-of-Lane Infraction</td><td>4.7%</td></tr><tr><td>Solvability-Aware Takeover</td><td>23.1%</td></tr><tr><td>   Repeated Replan Success</td><td>1.7%</td></tr><tr><td>   Replan Failed</td><td>8.7%</td></tr><tr><td>   Replan Success</td><td>12.7%</td></tr><tr><td>Restart</td><td>21.6%</td></tr></tbody></table>

Table 8: Composition of the post-training dataset from two different perspectives: (a) takeover phase and (b) collection trigger and event type. Percentages are normalized separately within each panel. Bold rows in (b) denote high-level trigger classes, while indented rows show their event-type breakdown.

## Appendix D Additional RoG-DAgger Results

This section provides further details on the evaluation metrics and additional results. We first elaborate on the primary metrics used throughout the paper in App. D.1. Then, we report complementary in-distribution results on Bench2Drive, including efficiency, comfort, multi-capability performance, and additional qualitative examples, in App. D.2. Finally, we present out-of-distribution results on Fail2Drive at both the scenario-class and individual-scenario level, together with visual examples, in App. D.3.

| Scenario Class | SimLingo | RoG-DAgger | $\Delta$ |
| --- | --- | --- | --- |
| Visual-longitudinal | 71.1 | 81.2 | +10.1 |
| Visual-lateral | 45.9 | 51.3 | +5.4 |
| Behavior | 31.2 | 62.0 | +30.8 |
| Robustness | 86.8 | 80.7 | \-6.1 |
| Mean | 58.8 | 68.8 | +10.0 |

Table 9: Per-class harmonic mean of DS and SR on generalization scenarios of Fail2Drive. RoG-DAgger improves over SimLingo on three out of four scenario classes, with the largest gain on behavior generalization (+30.8 HM), while SimLingo performs better on robustness scenarios.

### D.1 Details of Evaluation Metrics

The primary evaluation metrics reported in the main paper are Route Completion (RC), Driving Score (DS), and Success Rate (SR). Route Completion measures the percentage of the prescribed route traversed by the autonomous driving agent and therefore quantifies its progress toward the destination. Driving Score jointly evaluates route progress and driving compliance by weighting the Route Completion of each route with a multiplicative infraction penalty. The penalty accounts for unsafe or invalid driving behavior, such as collisions, red-light violations, lane departures, and scenario timeouts. Success Rate measures the proportion of evaluation routes that are completed successfully while satisfying the requirements of the evaluation protocol.

### D.2 In-Distribution Evaluation

In Tab. 11, we report additional B2D evaluation metrics covering driving efficiency, comfort, and multi-ability performance. RoG-DAgger generally outperforms SimLingo across these metrics and achieves performance comparable to that of TakeVLA. Further qualitative RoG-DAgger examples on B2D are presented in Fig. 12.

### D.3 Out-of-Distribution Evaluation

We report additional F2D results in Tabs. 9 and 11. Compared to SimLingo, RoG-DAgger improves harmonic mean by 30 p.p. on Behavior scenarios, including Fully Blocked and Construction Pedestrian. This gain is likely attributable to the inclusion of a broader range of safety-critical scenarios during post-training. In contrast, RoG-DAgger performs 6 p.p. worse on Robustness scenarios. We hypothesize that the model exhibits overly conservative behavior near construction sites with irregular object orientations and placements, which can impede route progress and lead to scenario-timeout penalties. We provide additional qualitative head-to-head comparisons of RoG-DAgger and SimLingo in Figs. 10 and 10. Further qualitative examples for F2D are presented in Fig. 12.

## Appendix E Technical Details

### E.1 Hardware and Software

We conduct our experiments on Microsoft Azure Standard\_NC96ads\_A100\_v4 instances, equipped with 96 AMD EPYC 7V13 (Milan) CPU cores, 880 GB of system memory, and four NVIDIA A100 PCIe GPUs with 80 GB of memory each. Our software environment uses Python 3.10, PyTorch 2.8 with CUDA 12.6 and cuDNN 9.10, and CARLA 0.9.15 for closed-loop simulation.

### E.2 Compute Time

Using four NVIDIA A100 GPUs, pre-training SimLingo-Qwen3VL-2B takes approximately 35 hours with our training schedule. Data collection on CARLA Leaderboard 2.0 requires approximately 15 hours with six parallel CARLA instances per GPU, and post-training takes an additional 3-4 hours.

<table><tbody><tr><td rowspan="2">Method</td><td rowspan="2">Efficiency</td><td rowspan="2">Comfort</td><td colspan="6">Multi-Capability (%)</td></tr><tr><td>Merging</td><td>Overtaking</td><td>Emergency Brake</td><td>Give Way</td><td>Traffic Sign</td><td>Mean</td></tr><tr><td colspan="9">Privileged methods</td></tr><tr><td>PDM-Lite <sup><a href="#fn:32">32</a></sup></td><td>213.47</td><td>22.33</td><td>82.05</td><td>91.11</td><td>81.67</td><td>80.00</td><td>70.97</td><td>81.16</td></tr><tr><td>RoG-Expert (ours)</td><td>196.89</td><td>14.13</td><td>75.00</td><td>84.44</td><td>93.33</td><td>40.00</td><td>88.95</td><td>76.35</td></tr><tr><td colspan="9">Pre-trained-only methods</td></tr><tr><td>UniAD <sup><a href="#fn:11">11</a></sup></td><td>129.21</td><td>43.58</td><td>12.16</td><td>20.00</td><td>23.64</td><td>10.00</td><td>13.89</td><td>15.89</td></tr><tr><td>TF <math><semantics><mrow><mo>+</mo> <mo>⁣</mo> <mo>+</mo></mrow> <annotation>++</annotation></semantics></math> <sup><a href="#fn:32">32</a></sup></td><td>–</td><td>–</td><td>58.75</td><td>57.77</td><td>83.33</td><td>40.00</td><td>82.11</td><td>64.39</td></tr><tr><td>ORION <sup><a href="#fn:6">6</a></sup></td><td>151.48</td><td>17.38</td><td>25.00</td><td>71.11</td><td>78.33</td><td>30.00</td><td>69.15</td><td>54.72</td></tr><tr><td>HiP-AD <sup><a href="#fn:26">26</a></sup></td><td>203.12</td><td>19.36</td><td>50.00</td><td>84.44</td><td>83.33</td><td>40.00</td><td>72.10</td><td>65.98</td></tr><tr><td>AutoVLA <sup><a href="#fn:31">31</a></sup></td><td>146.93</td><td>39.33</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><td>SimLingo <sup><a href="#fn:20">20</a></sup></td><td>259.23</td><td>33.67</td><td>54.01</td><td>57.04</td><td>88.33</td><td>53.33</td><td>82.45</td><td>67.03</td></tr><tr><td colspan="9">Post-trained methods</td></tr><tr><td>MindDrive <sup><a href="#fn:7">7</a></sup></td><td>–</td><td>–</td><td>32.89</td><td>75.56</td><td>68.33</td><td>50.00</td><td>57.89</td><td>56.94</td></tr><tr><td>TakeAD <sup><a href="#fn:14">14</a></sup></td><td>193.30</td><td>22.89</td><td>30.77</td><td>35.56</td><td>56.67</td><td>50.00</td><td>42.02</td><td>43.00</td></tr><tr><td>TakeVLA <sup><a href="#fn:8">8</a></sup></td><td>249.01</td><td>30.27</td><td>63.64</td><td>64.44</td><td>91.67</td><td>50.00</td><td>85.48</td><td>71.05</td></tr><tr><td>RoG-DAgger (ours)</td><td>242.84</td><td>35.10</td><td>67.09</td><td>64.44</td><td>85.00</td><td>50.00</td><td>84.74</td><td>70.25</td></tr></tbody></table>

Table 10: Detailed Bench2Drive evaluation of efficiency, comfort, and multi-ability performance. RoG-DAgger improves over SimLingo in merging, overtaking, and traffic-sign handling, and achieves a higher overall mean. It performs comparably to TakeVLA overall, with stronger merging but lower emergency-braking performance. Beyond these scenario-specific capabilities, we further assess whether the observed gains preserve desirable overall driving behavior in terms of efficiency and comfort. RoG-DAgger maintains competitive efficiency while improving comfort over both SimLingo and TakeVLA, suggesting that its closed-loop gains do not come at the cost of less comfortable driving.

<table><tbody><tr><td rowspan="2">Class</td><td rowspan="2">Scenario</td><td colspan="3">Driving Score (DS) <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td colspan="3">Success Rate (SR) <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>Base</td><td>Gen.</td><td>Rel.</td><td>Base</td><td>Gen.</td><td>Rel.</td></tr><tr><td>Behavior</td><td>Wall</td><td>98.6</td><td>68.5</td><td>-30.5</td><td>100.0</td><td>20.0</td><td>-80.0</td></tr><tr><td>Behavior</td><td>Fully Blocked</td><td>94.1</td><td>83.8</td><td>-11.0</td><td>100.0</td><td>80.0</td><td>-20.0</td></tr><tr><td>Behavior</td><td>Pedestrians On Road</td><td>82.6</td><td>64.8</td><td>-21.5</td><td>80.0</td><td>60.0</td><td>-25.0</td></tr><tr><td>Behavior</td><td>Construction Pedestrian</td><td>54.1</td><td>67.4</td><td>+24.6</td><td>40.0</td><td>60.0</td><td>+50.0</td></tr><tr><td>Visual-lat</td><td>Bad Parking</td><td>79.3</td><td>71.0</td><td>-10.5</td><td>70.0</td><td>60.0</td><td>-14.3</td></tr><tr><td>Visual-lat</td><td>Construction Permutations</td><td>76.4</td><td>54.3</td><td>-28.9</td><td>100.0</td><td>40.0</td><td>-60.0</td></tr><tr><td>Visual-lat</td><td>Custom Obstacles</td><td>51.8</td><td>55.6</td><td>+7.2</td><td>30.0</td><td>30.0</td><td>0.0</td></tr><tr><td>Visual-lon</td><td>Obscured Stop</td><td>95.4</td><td>99.4</td><td>+4.2</td><td>100.0</td><td>100.0</td><td>0.0</td></tr><tr><td>Visual-lon</td><td>Hard Brake</td><td>83.3</td><td>73.8</td><td>-11.4</td><td>80.0</td><td>60.0</td><td>-25.0</td></tr><tr><td>Visual-lon</td><td>Right Of Way</td><td>90.9</td><td>83.7</td><td>-8.0</td><td>100.0</td><td>80.0</td><td>-20.0</td></tr><tr><td>Visual-lon</td><td>Animals</td><td>78.6</td><td>82.1</td><td>+4.6</td><td>70.0</td><td>80.0</td><td>+14.3</td></tr><tr><td>Visual-lon</td><td>Pedestrian Other Blocker</td><td>73.8</td><td>74.1</td><td>+0.4</td><td>80.0</td><td>80.0</td><td>0.0</td></tr><tr><td>Robustness</td><td>Right Construction</td><td>93.7</td><td>72.5</td><td>-22.7</td><td>100.0</td><td>60.0</td><td>-40.0</td></tr><tr><td>Robustness</td><td>Opposite Construction</td><td>98.5</td><td>98.6</td><td>+0.2</td><td>100.0</td><td>100.0</td><td>0.0</td></tr><tr><td>Robustness</td><td>Image On Object</td><td>79.9</td><td>73.5</td><td>-8.0</td><td>100.0</td><td>80.0</td><td>-20.0</td></tr><tr><td>Robustness</td><td>Passable Obstacles</td><td>92.5</td><td>97.5</td><td>+5.3</td><td>100.0</td><td>100.0</td><td>0.0</td></tr><tr><td>Robustness</td><td>Pedestrian Crowd</td><td>71.8</td><td>64.5</td><td>-10.2</td><td>80.0</td><td>60.0</td><td>-25.0</td></tr></tbody></table>

Table 11: Per-scenario performance of RoG-DAgger on Fail2Drive. We report Driving Score (DS) and Success Rate (SR) on the Base and Generalization splits. SR and relative changes between the two splits (Rel.) are reported as percentages. RoG-DAgger performs particularly well on several novel scenario configurations, such as Construction Pedestrian and Fully Blocked. However, it still fails on entirely novel scenarios involving unseen assets, such as Wall.

![[h2h_pede.png|Refer to caption]]

Figure 9: Head-to-head comparison of RoG-DAgger (bottom) and SimLingo (top) in the Construction Pedestrian scenario of Fail2Drive. SimLingo tends to steer left despite oncoming traffic approaching from the left while the right side of the construction site is completely free. This behavior is consistent with a training bias in which construction zones predominantly appear on the right side of the road. As a result, SimLingo may resort to a late evasive maneuver and collide with the traffic warning sign while attempting to avoid the oncoming vehicles. In contrast, RoG-DAgger learns to steer right in the same situation. This behavior can be likely attributed to the RoG-Expert, whose expanded trajectory-and-speed solution space exposes the student to multiple feasible recovery strategies and thereby encourages greater trajectory multimodality.

![[b2d.png|Refer to caption]]

Figure 11: Qualitative examples of RoG-DAgger on Bench2Drive. In row (A), RoG-DAgger successfully merges into traffic while executing a left turn, using emergency braking to avoid a collision with the lead vehicle. In row (B), the predicted geometric trajectory adapts laterally from left to right as the pedestrian moves across the route. This contrasts with PDM-Lite/SimLingo-style behavior, where the geometric path remains largely fixed and collision avoidance is handled primarily through speed adjustment. Row (C) shows a failure case: RoG-DAgger slows down and attempts a lane change to pass through the construction area, however, due to the limited field of view, a vehicle approaching from the rear-left remains unobserved and causes a side collision.

[^1]: K. J. Åström and T. Hägglund Advanced pid control. ISA—The Instrumentation, Systems, and Automation Society. External Links: ISBN 978-1-55617-942-6, [Document](https://dx.doi.org/10.1002/9781394442102) Cited by: §3.1.

[^2]: Autonomous Vision CARLA Garage: a starter kit for the CARLA leaderboard 2.0. Note: https://github.com/autonomousvision/carla\_garage Cited by: §1, §4.1, §4.1.

[^3]: D. Chen, V. Koltun, and P. Krähenbühl Learning to drive from a world on rails. In 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021, pp. 15570–15579. External Links: [Link](https://doi.org/10.1109/ICCV48922.2021.01530), [Document](https://dx.doi.org/10.1109/ICCV48922.2021.01530) Cited by: §3.3.

[^4]: Z. Chen, W. Wang, Y. Cao, Y. Liu, Z. Gao, E. Cui, J. Zhu, S. Ye, H. Tian, Z. Liu, L. Gu, X. Wang, Q. Li, Y. Ren, Z. Chen, J. Luo, J. Wang, T. Jiang, B. Wang, C. He, B. Shi, X. Zhang, H. Lv, Y. Wang, W. Shao, P. Chu, Z. Tu, T. He, Z. Wu, H. Deng, J. Ge, K. Chen, M. Dou, L. Lu, X. Zhu, T. Lu, D. Lin, Y. Qiao, J. Dai, and W. Wang Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. Vol. abs/2412.05271. External Links: [Link](https://doi.org/10.48550/arXiv.2412.05271), [Document](https://dx.doi.org/10.48550/ARXIV.2412.05271), 2412.05271 Cited by: §4.1.

[^5]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger TransFuser: imitation with transformer-based sensor fusion for autonomous driving. IEEE Trans. Pattern Anal. Mach. Intell. 45 (11), pp. 12878–12895. External Links: [Link](https://doi.org/10.1109/TPAMI.2022.3200245), [Document](https://dx.doi.org/10.1109/TPAMI.2022.3200245) Cited by: §2.1.

[^6]: H. Fu, D. Zhang, Z. Zhao, J. Cui, D. Liang, C. Zhang, D. Zhang, H. Xie, B. Wang, and X. Bai Orion: A holistic end-to-end autonomous driving framework by vision-language instructed action generation. In IEEE/CVF International Conference on Computer Vision, ICCV 2025, Honolulu, HI, USA, October 19-25, 2025, pp. 24823–24834. External Links: [Link](https://doi.org/10.1109/ICCV51701.2025.02302), [Document](https://dx.doi.org/10.1109/ICCV51701.2025.02302) Cited by: Table 11, §1, §2.1, Table 1.

[^7]: H. Fu, D. Zhang, Z. Zhao, J. Cui, H. Xie, B. Wang, G. Chen, D. Liang, and X. Bai MindDrive: a vision-language-action model for autonomous driving via online reinforcement learning. In Computer Vision - ECCV 2026 - 19th European Conference, Malmö, Sweden, September 8-12, 2026, Proceedings, External Links: [Link](https://arxiv.org/abs/2512.13636) Cited by: Table 11, §2.2, Table 1.

[^8]: Y. Gao, D. Liu, Q. Zhang, Y. Zheng, H. Tian, G. Li, H. Ye, L. Chen, D. Ding, and D. Zhao Learning from mistakes: post-training for driving VLA with takeover data. Vol. abs/2603.14972. External Links: [Link](https://doi.org/10.48550/arXiv.2603.14972), [Document](https://dx.doi.org/10.48550/ARXIV.2603.14972), 2603.14972 Cited by: §B.1, Table 11, §1, §2.2, §3.1, §3.2, §3.4, §3.4, §4.1, Table 1.

[^9]: S. Gerstenecker, A. Geiger, and K. Renz Fail2Drive: benchmarking closed-loop driving generalization. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), External Links: [Link](https://arxiv.org/abs/2604.08535) Cited by: §4.1.

[^10]: E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen LoRA: low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022, External Links: [Link](https://openreview.net/forum?id=nZeVKeeFYf9) Cited by: Appendix A.

[^11]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, L. Lu, X. Jia, Q. Liu, J. Dai, Y. Qiao, and H. Li Planning-oriented autonomous driving. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pp. 17853–17862. External Links: [Link](https://doi.org/10.1109/CVPR52729.2023.01712), [Document](https://dx.doi.org/10.1109/CVPR52729.2023.01712) Cited by: Table 11, §2.1, Table 1.

[^12]: X. Jia, Z. Yang, Q. Li, Z. Zhang, and J. Yan Bench2Drive: towards multi-ability benchmarking of closed-loop end-to-end autonomous driving. In Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, A. Globersons, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. M. Tomczak, and C. Zhang (Eds.), External Links: [Link](http://papers.nips.cc/paper/_files/paper/2024/hash/017761f94a1cd66d01c041aff85492c4-Abstract-Datasets/_and/_Benchmarks/_Track.html) Cited by: §1, §4.1.

[^13]: N. Karnchanachari, D. Geromichalos, K. S. Tan, N. Li, C. Eriksen, S. Yaghoubi, N. Mehdipour, G. Bernasconi, W. K. Fong, Y. Guo, and H. Caesar Towards learning-based planning: the nuplan benchmark for real-world autonomous driving. In IEEE International Conference on Robotics and Automation, ICRA 2024, Yokohama, Japan, May 13-17, 2024, pp. 629–636. External Links: [Link](https://doi.org/10.1109/ICRA57147.2024.10610077), [Document](https://dx.doi.org/10.1109/ICRA57147.2024.10610077) Cited by: §1.

[^14]: D. Liu, Y. Gao, D. Qian, Q. Zhang, X. Ye, J. Han, Y. Zheng, X. Liu, Z. Xia, D. Ding, Y. Pan, and D. Zhao TakeAD: preference-based post-optimization for end-to-end autonomous driving with expert takeover data. IEEE Robotics Autom. Lett. 11 (2), pp. 1738–1745. External Links: [Link](https://doi.org/10.1109/LRA.2025.3643264), [Document](https://dx.doi.org/10.1109/LRA.2025.3643264) Cited by: §B.6, Table 11, §1, §1, §2.1, §2.2, §3.1, §3.4, §3.4, §4.1, Table 1.

[^15]: I. Loshchilov and F. Hutter Decoupled weight decay regularization. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019, External Links: [Link](https://openreview.net/forum?id=Bkg6RiCqY7) Cited by: §4.1.

[^16]: L. Nguyen, M. Fauth, B. Jaeger, D. Dauner, M. Igl, A. Geiger, and K. Chitta LEAD: minimizing learner-expert asymmetry in end-to-end driving. In Conference on Computer Vision and Pattern Recognition (CVPR), Cited by: §B.7, §1, §1, §3.4.

[^17]: NVIDIA, Y. Cao, R. de Lutio, S. Fidler, G. Garcia Cobo, Z. Gojcic, M. Igl, B. Ivanovic, P. Karkus, J. Martinez Esturo, M. Pavone, A. Smith, E. Tanimura, M. Tyszkiewicz, M. Watson, Q. Wu, and L. Zhang AlpaSim: a modular, lightweight, and data-driven research simulator for autonomous driving. Note: https://github.com/NVlabs/alpasimAccessed: 2026-07-28 Cited by: §5.

[^18]: M. Pintz, J. Sicking, M. Poretschkin, and M. Akila A survey on uncertainty toolkits for deep learning. ICLR 2022 Workshop on ML Evaluation Standards. Cited by: §5.

[^19]: Qwen Qwen3-vl technical report. Vol. abs/2511.21631. External Links: [Link](https://doi.org/10.48550/arXiv.2511.21631), [Document](https://dx.doi.org/10.48550/ARXIV.2511.21631), 2511.21631 Cited by: §4.1.

[^20]: K. Renz, L. Chen, E. Arani, and O. Sinavski SimLingo: vision-only closed-loop autonomous driving with language-action alignment. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pp. 11993–12003. External Links: [Link](https://openaccess.thecvf.com/content/CVPR2025/html/Renz/_SimLingo/_Vision-Only/_Closed-Loop/_Autonomous/_Driving/_with/_Language-Action/_Alignment/_CVPR/_2025/_paper.html), [Document](https://dx.doi.org/10.1109/CVPR52734.2025.01120) Cited by: Table 4, Appendix A, Appendix A, Table 11, §1, §2.1, Table 1.

[^21]: D. Rolnick, A. Ahuja, J. Schwarz, T. P. Lillicrap, and G. Wayne Experience replay for continual learning. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, H. M. Wallach, H. Larochelle, A. Beygelzimer, F. d’Alché-Buc, E. B. Fox, and R. Garnett (Eds.), pp. 348–358. External Links: [Link](https://proceedings.neurips.cc/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html) Cited by: §4.1.

[^22]: S. Ross and D. Bagnell Efficient reductions for imitation learning. In Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, AISTATS 2010, Chia Laguna Resort, Sardinia, Italy, May 13-15, 2010, Y. W. Teh and D. M. Titterington (Eds.), JMLR Proceedings, Vol. 9, pp. 661–668. External Links: [Link](http://proceedings.mlr.press/v9/ross10a.html) Cited by: §1.

[^23]: S. Ross, G. J. Gordon, and D. Bagnell A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, AISTATS 2011, Fort Lauderdale, USA, April 11-13, 2011, G. J. Gordon, D. B. Dunson, and M. Dudík (Eds.), JMLR Proceedings, Vol. 15, pp. 627–635. External Links: [Link](http://proceedings.mlr.press/v15/ross11a/ross11a.pdf) Cited by: §1, §2.2, §3.1.

[^24]: J. Sicking, M. Akila, M. Pintz, T. Wirtz, A. Fischer, and S. Wrobel A novel regression loss for non-parametric uncertainty optimization. 3rd Symposium on Advances in Approximate Bayesian Inference (AABI). Cited by: §5.

[^25]: C. Sima, K. Renz, K. Chitta, L. Chen, H. Zhang, C. Xie, J. Beißwenger, P. Luo, A. Geiger, and H. Li DriveLM: driving with graph visual question answering. In Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part LII, A. Leonardis, E. Ricci, S. Roth, O. Russakovsky, T. Sattler, and G. Varol (Eds.), Lecture Notes in Computer Science, Vol. 15110, pp. 256–274. External Links: [Link](https://doi.org/10.1007/978-3-031-72943-0/_15), [Document](https://dx.doi.org/10.1007/978-3-031-72943-0%5F15) Cited by: §2.1.

[^26]: Y. Tang, Z. Xu, Z. Meng, and E. Cheng HiP-ad: hierarchical and multi-granularity planning with deformable attention for autonomous driving in a single decoder. In IEEE/CVF International Conference on Computer Vision, ICCV 2025, Honolulu, HI, USA, October 19-25, 2025, pp. 25605–25615. External Links: [Link](https://doi.org/10.1109/ICCV51701.2025.02375), [Document](https://dx.doi.org/10.1109/ICCV51701.2025.02375) Cited by: Table 11, Table 1.

[^27]: J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. H. Chi, Q. V. Le, and D. Zhou Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (Eds.), External Links: [Link](http://papers.nips.cc/paper/_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html) Cited by: Appendix A.

[^28]: Y. Wu, D. Slieter, V. Subramanian, A. Abouelazm, R. Bohn, and J. M. Zöllner Why braking? scenario extraction and reasoning utilizing LLM. In IEEE International Conference on Vehicular Electronics and Safety, ICVES 2025, Coventry, United Kingdom, October 27-28, 2025, pp. 248–255. External Links: [Link](https://doi.org/10.1109/ICVES65691.2025.11376535), [Document](https://dx.doi.org/10.1109/ICVES65691.2025.11376535) Cited by: §2.1.

[^29]: L. Zhong, F. Rosenthal, J. Sicking, F. Hüger, T. Bagdonat, H. Gottschalk, and L. Schwinn FOCUS: internal MLLM representations for efficient fine-grained visual question answering. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, External Links: [Link](https://openreview.net/forum?id=8I1XNt70lj) Cited by: §2.1.

[^30]: L. Zhong, J. Sicking, F. Hüger, and H. Gottschalk VL4AD: vision-language models improve pixel-wise anomaly detection. In 2nd Workshop on Vision-Centric Autonomous Driving (VCAD) at the European Conference on Computer Vision (ECCV), External Links: 2409.17330, [Link](https://arxiv.org/abs/2409.17330) Cited by: §2.1.

[^31]: Z. Zhou, T. Cai, S. Z. Zhao, Y. Zhang, Z. Huang, B. Zhou, and J. Ma AutoVLA: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, External Links: [Link](https://openreview.net/forum?id=28qUA2bSe5) Cited by: Table 11, §2.1, Table 1.

[^32]: J. Zimmerlin, J. Beißwenger, B. Jaeger, A. Geiger, and K. Chitta Hidden biases of end-to-end driving datasets. Vol. abs/2412.09602. External Links: [Link](https://doi.org/10.48550/arXiv.2412.09602), [Document](https://dx.doi.org/10.48550/ARXIV.2412.09602), 2412.09602 Cited by: Table 11, Table 11, §3.3, §3.4, Table 1, Table 1.