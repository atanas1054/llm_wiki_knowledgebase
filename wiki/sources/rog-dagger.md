---
title: "RoG-DAgger: Rollout-Guided Post-Training for End-to-End Driving"
type: source-summary
sources: ["raw/papers/RoG-DAgger_ Rollout-Guided Post-Training for End-to-End Driving.md"]
related: [concepts/bench2drive.md, concepts/pdm-lite.md, concepts/rl-for-ad.md, concepts/teacher-pseudo-labels.md, concepts/evaluation-variance.md, concepts/data-driven-simulators.md, concepts/alpasim-benchmark.md, sources/think2drive.md, sources/raw2drive.md, sources/learning-to-drive-from-a-world-model.md, sources/simscale.md, sources/rad.md, sources/clover.md, sources/toad.md, sources/linkvla.md, sources/sparsedrivev2.md, sources/deepsight.md, sources/automot.md, sources/orion.md, sources/autovla.md, sources/lvdrive.md, sources/dynvla.md, research-directions.md]
created: 2026-10-28
updated: 2026-10-28
confidence: medium
---

# RoG-DAgger: Rollout-Guided Post-Training for End-to-End Driving

**Paper**: RoG-DAgger: Rollout-Guided Post-Training for End-to-End Driving
**Authors**: Liangyu Zhong, Joachim Sicking, Fabian Hueger, Hanno Gottschalk (per-author affiliations are not marked in the clipping)
**Orgs**: CARIAD SE, Volkswagen Group; Institute of Mathematics, Technical University of Berlin
**arXiv**: 2608.24525v1 (August 2026)
**Code**: not stated
**Source**: `raw/papers/RoG-DAgger_ Rollout-Guided Post-Training for End-to-End Driving.md`. The clipping has the full text, Tables 1–11, Algorithms 1–2 and eight figures. Four figures were missing and were recovered from the arXiv HTML into `raw/assets/` (new files, nothing overwritten): Figure 1 (`ttc_compare.svg`), Figure 7 (`replan.png`), Figure 10 (`h2h_blocked.png`) and Figure 12 (`f2d.png`).

**Confidence: medium.** The method is fully specified (both algorithms, every trigger threshold, the shift geometry and the filters), and the paper reports three closed-loop benchmarks and five ablations. Against that:
- The headline "+5.3 DS / +6.2 SR" is measured against the published InternVL SimLingo. The checkpoint actually post-trained is the authors' Qwen3-VL-2B re-implementation (86.59 / 66.36), and against it the gain is **+3.75 DS / +7.15 SR**.
- Every number is a single run. The ablation deltas are 1.4–2.5 DS, and no full Bench2Drive repeat exists in the wiki to size them against.
- The rate at which the new expert's own pre-takeover labels still collide under the rollout is not reported. That rate is the paper's motivating measurement (29.6% for PDM-Lite).
- No code.

---

## What It Is

**One round of DAgger on a pretrained camera VLA (SimLingo) in CARLA, with a privileged expert (PDM-Lite) changed in three ways and every expert plan checked by a kinematic rollout before it becomes a label.**

1. **Rollout model.** The candidate ego plan (path + target speed) is run through the deployed lateral and longitudinal controllers and a kinematic bicycle model (World on Rails parameters), including the deployed switch to full braking. Surrounding actors are propagated non-reactively, as in PDM-Lite.
2. **Expanded expert.** When PDM-Lite wants to emergency-brake and the rollout says braking still collides, the expert laterally shifts the colliding segment of its route (0.2 m steps to 3 m, cosine ease-in/out, lane-bounded) and, inside junctions, also tries 25% and 50% of the current speed. It keeps the first collision-free, lane-compliant candidate, preferring higher speed, then smaller shift.
3. **Solvability-aware takeover.** The expert takes over when the rollout-estimated time to collision (R-TTC) is under 1 s and a collision-free replan exists, or 1–1.5 s with a successful replan in two of four frames. This is meant to take over near the **point of no return** (PONR): the last state from which at least one candidate avoids the collision.
4. **Field-of-view alignment.** The expert plans only around actors inside the student's 110° front camera view, and pre-takeover samples are dropped when the actor that later causes the infraction was not visible.
5. **Two separate collection runs.** An infraction-triggered run keeps 2 s before takeover (with the expert's counterfactual plans as labels) and 2 s of expert recovery after it. A solvability-triggered run keeps only the 2 s after takeover. Both also restart the expert when the ego is stuck (4 s of recovery kept).
6. **Fine-tune** on about 51K post-training samples mixed 3:1 with SimLingo's pre-training data, L1 imitation, five epochs.

No reward and no RL. The deployed policy is unchanged: front camera only, MLP heads for 10 path and 10 speed waypoints, PID control.

**Results**

| Benchmark | SimLingo (published) | SimLingo-Qwen3 (actual base) | **RoG-DAgger** | Notes |
|---|---|---|---|---|
| Bench2Drive DS / SR | 85.07 / 67.27 | 86.59 / 66.36 | **90.34 / 73.51** | TakeVLA (same base model, plus RL) 89.72 / 73.73; LinkVLA 91.01 / 74.55 is not in the table |
| Longest6 v2 DS / RC | 22 / 70 | – | **44 / 88** | Which SimLingo the 22 is, is not stated |
| Fail2Drive base DS / SR | 82.6 / 79.3 | – | 80.2 / 80.0 | DS −2.4 |
| Fail2Drive generalization DS / SR | 71.7 / 55.0 | – | **74.7 / 66.0** | TF++ (LiDAR) has the best DS on both splits |
| Cost (4× A100) | | 35 h pre-training | +15 h data collection, +3–4 h fine-tuning | |

---

## Key Takeaways

- **The wiki's first DAgger result, and second on its Bench2Drive ladder.** 90.34 DS sits 0.67 below LinkVLA (91.01) and above SparseDriveV2 (89.15). It is a PDM-Lite-protocol student: SimLingo was pre-trained on PDM-Lite data and the post-training expert is built on PDM-Lite.
  - Against the checkpoint it starts from, the gain is +3.75 DS / +7.15 SR. The paper's "+5.3 / +6.2" uses the published InternVL SimLingo row.
  - The success-rate gain is the larger and more robust one: every ablation keeps DS within 2.5 of the full method, while SR moves by up to 6.9.
- **A weaker expert makes a better teacher, in this pipeline.** RoG-Expert drives worse than PDM-Lite on its own (Bench2Drive 93.41 vs 97.02 DS; Longest6 v2 55 vs 73), mostly because it is blinded to actors outside the student's view. Yet the student DAgger'd with it scores 90.34, against 87.85 for conventional DAgger with PDM-Lite.
  - The authors note the comparison changes the expert and the takeover trigger together.
  - The PDM-Lite DAgger baseline is only +1.26 DS / +0.3 SR over the base. So conventional DAgger with the stronger expert buys almost nothing on Bench2Drive.
- **Expert labels before takeover are often unsafe.** Rolled out from the states the student actually reached, PDM-Lite's own plans collide in **29.6%** of 40-frame pre-takeover samples and **43.2%** of 20-frame ones, against 0.47% after takeover (Table 6). PDM-Lite responds to a predicted collision only by braking along its fixed route, and braking is too late from many policy-induced states (Figures 4, 6).
- **Supervising the states before failure matters most for success rate.** Training on post-takeover samples alone costs −2.29 DS / −5.47 SR. Pre-takeover samples are 27.3% of the data.
- **The largest gain is long-horizon.** Longest6 v2 doubles (22 → 44 DS) and route completion rises 70 → 88. The student ends 11 DS below its post-training expert (55), where pretrained SimLingo was 51 below PDM-Lite (73). Longest6's routes are about 10× longer than Bench2Drive's, so compounding errors have more room, which is the case DAgger is meant for.
- **OOD success rises, OOD driving score barely.** On Fail2Drive's generalization split SR goes 55 → 66 and DS 71.7 → 74.7. On the base split DS falls 2.4. Behaviour scenarios gain +30.8 harmonic mean; robustness scenarios lose 6.1, which the paper attributes to over-caution near construction sites (timeouts).
- **The expert's solution space is the single largest component for SR.** Without lateral shifts and junction speeds, SR is 68.94 (−4.57). Without solvability-aware takeover or FoV alignment, 69.86 each. The three removals cost 1.38–1.69 DS each. Their sum (4.70) is almost twice the 2.49 DS between the full method and the PDM-Lite DAgger baseline that lacks all three, which in single runs cannot be told apart from noise.
- **Cheap compared with pre-training.** Collection is about 15 h (six CARLA instances per GPU) and fine-tuning 3–4 h on four A100s, against 35 h of pre-training.

---

## Method

### Motivation: the student meets states its data never had

![[ttc_compare.svg|Figure 1: SimLingo pre-training data (orange) lacks safety-critical interactions as measured by time-to-collision (TTC), limiting the model's ability to learn how to prevent or recover from such situations. As interactions, however, occur frequently in closed-loop Bench2Drive inference (blue), this constitutes a substantial distribution shift. Samples with TTC > 2s are omitted for clarity. Recovered from the arXiv HTML.]]

**Figure 1** (read from the SVG). Histogram of TTC in 0.1 s bins up to 2 s, each distribution normalized to 100% over the bins shown. **About 17.5% of the Bench2Drive inference samples fall in the lowest bin (TTC < 0.1 s), against about 0.1% of the pre-training samples.** Pre-training mass rises steadily toward 2 s (about 8–9% per bin above 1.1 s). Because both curves are normalized within TTC ≤ 2 s, the figure shows the shape of near-interactions, not how often they occur. The lowest bin probably includes frames in contact.

The authors' failure count for pretrained SimLingo on Bench2Drive (Table 5, below): 53 of 68 analysed failures (77.9%) are collisions, 49 of them with vehicles.

### Overview

![[teaser 5.png|Figure 2: RoG-DAgger leverages kinematic rollout and introduces three improvements to existing DAgger methods. Panel A shows how rollouts are used to construct trajectory-and-speed supervision and to evaluate candidate expert plans. Panels B1-B3 highlight three modifications: (1) an expanded trajectory-and-speed solution space (red: collision-prone nominal plan, green: improved plan), (2) solvability-aware takeover near the estimated point of no return (yellow star: PONR, red region: no feasible corrective plan remains), and (3) expert-student field-of-view alignment (yellow triangle) for student-learnable supervision.]]

**Figure 2**: (A) ego and actor rollouts overlap in a predicted collision. (B1) the nominal plan (red, straight) collides with a crossing actor; a shifted plan (green) avoids it. (B2) states split into solvable and unsolvable regions; the latest solvable state is the PONR. (B3) the expert sees only the student's field of view.

![[overall.png|Figure 3: Overview of the RoG-DAgger training pipeline. Starting from a pretrained E2E policy, we collect post-training data in two independent closed-loop runs (see zoom-in panel) and fine-tune the policy using a mixture of pre- and post-training data. The infraction-based run captures policy-induced failures and expert recovery demonstrations, while the solvability-aware run intervenes before failures become unavoidable to collect corrective behavior (gray hatched bar indicates avoided infraction). A restart trigger is used in both runs but omitted for clarity.]]

**Figure 3**: pre-training data → pretrained E2E policy → closed-loop simulation with the expert, run twice (infraction-based takeover, with pre- and post-takeover frames kept; solvability-aware takeover, with post-takeover frames only) → post-training data mixed with pre-training data → fine-tuned policy.

**Why two runs.** A solvability-aware takeover prevents the failure, so the infraction it would have caused never happens. Running both triggers in one rollout would suppress the infraction-run data.

### Problem formulation (§3.1)

$$\hat{\mathbf a}_t=\Pi_\theta(\mathbf C_t,\mathbf S^\text{ego}_t,\mathbf O_t),\qquad \hat{\mathbf a}_t=(\hat{\mathbf a}^\text{geo}_t,\hat{\mathbf a}^\text{temp}_t)$$

$\mathbf C_t$ is the navigation condition (GPS target points or language), $\mathbf S^\text{ego}_t$ position and speed, $\mathbf O_t$ the camera image. The plan is a geometric path of equidistant waypoints plus a temporal profile along it, executed by PID controllers.

$$\mathcal L_\text{IL}=\lambda^\text{geo}\lVert\hat{\mathbf a}^\text{geo}-\mathbf a^{\text{geo},*}\rVert_1+\lambda^\text{temp}\lVert\hat{\mathbf a}^\text{temp}-\mathbf a^{\text{temp},*}\rVert_1$$

DAgger queries the expert $\mathbf a^*$ at states the student visits. As in TakeAD and TakeVLA, only states around takeover events are kept, not every step.

### Kinematic ego and actor rollout (§3.3, Algorithm 1)

```
Algorithm 1  Controller-in-the-loop kinematic ego rollout
Inputs: ego state s0, geometric trajectory a_geo, target speed v_target,
        horizon H, braking ratio rho
s_0 <- s0 ; a_rollout <- {}
for k = 0 .. H-1:
    phi_k <- LateralCtrl(s_k, a_geo)
    if v_target ≈ 0  or  v_k > rho * v_target:
        (u_thr, u_brk) <- (0, 1)                       # full braking, as deployed
    else:
        (u_thr, u_brk) <- (ThrottleCtrl(v_k, v_target), 0)
    s_{k+1}, v_{k+1} <- BicycleModel(s_k, phi_k, u_thr, u_brk)
    a_rollout[k+1] <- s_{k+1}
return a_rollout
```

The paper's point against PDM-Lite's own rollout: PDM-Lite models throttle without the deployed braking condition, so it does not reproduce the throttle-to-full-brake transition. Actors are propagated over the same horizon "assuming unchanged low-level control" (non-reactive).

### Expanding the trajectory-and-speed solution space (§3.4, Algorithm 2, App. B.3)

![[expert_comparison_main_text.png|Figure 4: Comparison between experts with fixed and expanded sets of geometric trajectories. In (a), the expert follows a fixed pre-computed route and attempts to avoid the predicted collision through emergency braking. However, due to vehicle dynamics, the ego vehicle cannot decelerate sufficiently before reaching the collision point. In (b), the expert additionally considers alternative geometric trajectories, which are evaluated through rollouts for collision-freeness before selection. Geometric trajectories (red dots in the images) correspond to the routes executed in simulation. TTC and distance are estimated using our rollout model.]]

**Figure 4** (a lane change next to a green car). (a) Fixed route: distance 2.7 m, TTC 0.05 s at 6.40 m/s with target speed 0; it collides at 1.16 m/s. (b) Expanded: with a 0.70 m left shift the rollout gives distance 6.9 m and TTC 1.00 s at 7.52 m/s; the ego stops safely (0.01 m/s).

$$\mathcal V_\text{target}=\begin{cases}\{0\}&\text{outside junctions}\\\{0,\ 0.25\,v_t,\ 0.5\,v_t\}&\text{inside junctions}\end{cases}$$

```
Algorithm 2  Trajectory-and-speed solution-space expansion
Inputs: ego state s, nominal trajectory a_geo, current speed v, predicted actors A,
        shift unit Δd, max shift d_max
(o, I) <- PDM-Collision(s, a_geo, A)        # collision actor and overlapping waypoint indices
if o = ∅: return ∅
a_brk <- R_ego(s, a_geo, 0)
if Collision-Free(a_brk, A): return (a_geo, a_brk)       # braking suffices: no replanning
V <- {0.5 v, 0.25 v, 0} if inside a junction else {0}
for v_target in V:                                        # higher speed first
    for d = Δd, 2Δd, ..., d_max:                          # smaller shift first
        a' <- Local-Shift(a_geo, I, o, d)                 # away from o's direction of motion
        if Lane-Compliant(a'):
            a'_temp <- R_ego(s, a', v_target)
            if Collision-Free(a'_temp, A): return (a', a'_temp)
return (a_geo, a_brk)                                     # fallback: nominal path, full braking
```

Shift geometry (App. B.3):
- Replanning starts when PDM-Lite's detector requests an emergency stop (ego > 0.4 m/s, target speed < 10⁻³ m/s).
- The shifted segment starts $5\,\text{m}+l_\text{half,ego}+l_\text{half,actor}$ before the first predicted overlap and ends $2\,\text{m}+l_\text{half,ego}+l_\text{half,actor}$ after the last.
- Ease-in and ease-out over 2 m with $E(u)=\tfrac{1-\cos(\pi u)}{2}$, blending $(1-E)\,\mathbf a_\text{geo}+E\,\mathbf a'_\text{geo}$.
- $\Delta d=0.2$ m, $d_\max=3$ m. A candidate is lane-valid if its lateral deviation stays within $w_\text{lane}/2+0.2$ m.
- Candidates are checked against all predicted actors, though the shift targets only the collision actor. PDM-Lite's default bounding-box inflation is kept.
- Expert target speed follows LEAD: the minimum of the speed limit and nearby traffic's flow speed.

![[no_replan.png|Figure 6: Expert without expanded trajectory-and-speed solution space fails to prevent collision. An expert such as PDM-Lite follows a precomputed route and primarily responds to predicted collisions with full braking along the nominal path. However, in safety-critical states, especially those induced by the policy, braking alone does not necessarily prevent an impending collision. At t1, the ego vehicle controlled by expert follows the nominal route and prepares for lane change. At t2, the expert predicts a collision and initiates full braking, but the intervention is insufficient due to the vehicle dynamics, and a collision occurs at t4.]]

**Figure 6** (MergeIntoSlowTraffic, PDM-Lite-style expert): t1 = 0 s at 8.61 m/s; t2 = 0.25 s, collision predicted (2.8 m, TTC 0.25 s), target speed 0; t3 = 0.5 s at 6.40 m/s (2.7 m, TTC 0.05 s); t4 = 0.75 s, collision at 1.16 m/s.

![[replan.png|Figure 7: Expert with expanded trajectory-and-speed solution space succeeds in preventing impending collision. RoG-Expert searches for an alternative geometric trajectory when the rollout model predicts that emergency braking along the nominal trajectory is insufficient. At t2, the expert proposes emergency braking combined with a first local trajectory shift; at t3 it performs a second, larger shift as the previous plan remains insufficient. The ego then brakes safely to zero speed at t4. Across the two replanning steps the predicted TTC increases from 0.4 s to 1.0 s and no collision occurs. Bounding boxes are inflated following PDM-Lite. Recovered from the arXiv HTML.]]

**Figure 7** (same scene, RoG-Expert): t1 = 0 s at 8.92 m/s; t2 = 0.15 s, shift left 0.30 m (2.7 m, TTC 0.40 s); t3 = 0.25 s, shift left 0.70 m (6.9 m, TTC 1.00 s); t4 = 0.9 s, stopped (0.01 m/s), safe.

### Solvability-aware takeover (§3.4, App. B.2, B.5)

When PDM-Lite predicts a collision and starts emergency braking, the rollout first checks whether braking alone avoids it. If so, the student keeps control. Otherwise replanning is needed, and:

| Trigger | Condition (Table 7) |
|---|---|
| Infraction | A standard CARLA infraction: collision, red-light violation or out-of-lane violation (the text also names stop-sign violations) |
| Solvability-aware: Replan Success | R-TTC < 1 s **and** replanning succeeds |
| Solvability-aware: Repeated Replan Success | 1 s < R-TTC < 1.5 s **and** replanning succeeds in two of four consecutive frames |
| Solvability-aware: Replan Failure | R-TTC < 1 s **and** replanning fails (fallback: emergency braking, no replanning) |
| Restart | Ego speed < 0.1 m/s for all of the previous 200 frames, while the mean expert target speed over those frames and the current one both exceed 0.4 m/s |

R-TTC is the time to collision estimated by the rollout model. The paper argues that TakeAD's infraction and discrepancy triggers intervene after a failure or for benign differences, and that TakeVLA's constant-speed collision prediction triggers prematurely because it ignores braking and alternative plans.

![[lb2_et.png|Figure 8: SimLingo with solvability-aware takeover. At t1, the policy controls the ego vehicle while preparing for a left turn. As oncoming traffic approaches, our kinematic rollout model predicts at t2 a time-to-collision (TTC) of 0.05 s, indicating an imminent collision. This activates the solvability-aware takeover, which shifts the geometric trajectory 0.5 m to the right while setting the target speed to 50% of the current speed. At t3, the RoG-Expert applies a second, larger correction of 0.9 m to the right, again setting the target speed to 50% of the current speed, successfully avoiding the collision. At t4, the ego vehicle continues safely under expert control during the post-takeover period.]]

**Figure 8**: t1 = 0 s, policy in control at 9.03 m/s; t2 = 0.25 s, R-TTC 0.05 s at 2.3 m, takeover, shift right 0.50 m, target 4.64 m/s; t3 = 0.3 s, shift right 0.90 m (2.7 m), target 4.77 m/s; t4 = 0.5 s, expert driving at 8.75 m/s.

### Field-of-view-aligned planning (§3.4, App. B.6)

Following LEAD, the expert plans only around actors in the student's field of view. Pre-takeover samples whose responsible actor is not visible are discarded, while their post-takeover recovery samples are kept. The paper's example: in an early lane-change state the car that later collides may still be behind.

### Data and training (§4.1, App. A, B.6, C)

- **Student**: SimLingo with Qwen3-VL-2B-Instruct in place of InternVL-2, front camera 110° horizontal FoV, chain-of-thought output disabled. Pre-trained on SimLingo's recipe: 5 epochs, lr 1×10⁻⁴, effective batch 48 on 4 GPUs, LoRA r = 32, α = 64.
- **Collection**: CARLA Leaderboard 2.0 training routes, focusing on routes where the pretrained policy performs poorly (following TakeAD).
- **Windows**: infraction events keep 40 frames (2 s) before and 40 after takeover; solvability-aware events 40 frames after; restarts 80 frames (4 s) after.
- **Filtering** (TakeAD's rules plus one):
  - A sample is kept if target-speed disagreement exceeds 0.5 m/s, the current speed exceeds the target by more than 10%, or steering deviates by at least 0.2 rad.
  - Events averaging under 0.5 m/s are removed. Post-takeover segments are removed if both average and final speed are below 0.3 m/s.
  - Pre-takeover collision frames are kept only if the collision actor is visible.
- **Mix**: about 51K post-training samples, 3:1 with pre-training data (against forgetting).
- **Fine-tuning**: five epochs, AdamW, batch 24, lr 10⁻⁵.
- **Hardware**: Azure NC96ads A100 v4 (4× A100 80 GB, 96 CPU cores); CARLA 0.9.15, PyTorch 2.8.

**Table 8: composition of the post-training data**

| (a) Takeover phase | Share |
|---|---:|
| Pre-takeover | 27.3% |
| Post-takeover | 72.7% |

| (b) Collection trigger / event type | Share |
|---|---:|
| **Infraction** | **55.3%** |
| Collision | 47.7% |
| Red-light infraction | 2.9% |
| Out-of-lane infraction | 4.7% |
| **Solvability-aware takeover** | **23.1%** |
| Repeated Replan Success | 1.7% |
| Replan Failed | 8.7% |
| Replan Success | 12.7% |
| **Restart** | **21.6%** |

The sub-rows add up to their totals. More than a third of the solvability-aware data (8.7 of 23.1 points) comes from Replan Failure, the fallback in which no collision-free candidate was found.

---

## Results

### Table 1: Bench2Drive and Longest6 v2

| Method | B2D DS ↑ | B2D SR ↑ | Longest6 v2 DS ↑ | Longest6 v2 RC ↑ |
|---|---:|---:|---:|---:|
| *Privileged* | | | | |
| PDM-Lite | 97.02 | 92.27 | 73 | 100 |
| RoG-Expert (ours) | 93.41 | 81.81 | 55 | 100 |
| *Pre-trained only* | | | | |
| UniAD | 45.81 | 16.36 | – | – |
| TF++ | 84.21 | 67.27 | 23 | 70 |
| ORION | 77.74 | 54.62 | – | – |
| AutoVLA | 78.84 | 57.73 | – | – |
| HiP-AD | 86.77 | 69.09 | 7 | 56 |
| SimLingo | 85.07 | 67.27 | 22 | 70 |
| *Post-trained* | | | | |
| MindDrive | 78.04 | 55.09 | – | – |
| TakeAD | 71.39 | 40.83 | – | – |
| TakeVLA | 89.72 | 73.73 | – | – |
| **RoG-DAgger (ours)** | **90.34** | 73.51 | **44** | **88** |

The caption's "RS" means route completion (RC). TF++ and SimLingo share SR 67.27.

### Table 4: the actual base model

| Model | DS | SR (%) |
|---|---:|---:|
| SimLingo (published, InternVL-2) | 85.07 | 67.27 |
| SimLingo-Qwen3-VL-2B (ours; "SimLingo" in the rest of the paper) | 86.59 | 66.36 |

### Table 5: pretrained SimLingo's Bench2Drive failures

| Failure category | Count | Proportion (%) |
|---|---:|---:|
| Total recorded failures | 73 | – |
| Ignored (YieldToEmergencyVehicle; no rear camera) | 5 | – |
| Analysed failures | 68 | 100.0 |
| Collision-related | 53 | 77.9 |
| — with vehicle | 49 | 72.1 |
| — with static object | 4 | 5.9 |
| — with pedestrian | 1 | 1.5 |
| Non-collision | 15 | 22.1 |

One failure involves both a vehicle and a static object.

### Table 6: PDM-Lite's counterfactual plans from policy-induced states (diagnostic run)

Collision-triggered DAgger with PDM-Lite; each sample's PDM-Lite plan is executed with the rollout model and counted as a collision if it overlaps any actor's predicted trajectory.

| Split | Window (frames) | Samples | Predicted collisions | Rate |
|---|---:|---:|---:|---:|
| Pre-takeover | 20 | 5,568 | 2,404 | 43.18% |
| Pre-takeover | 40 | 9,565 | 2,829 | 29.58% |
| Post-takeover | 40 | 20,845 | 99 | 0.47% |

The 20-frame window follows TakeVLA. Both the measurement and RoG-Expert's acceptance test use the same rollout model.

### Table 2: Fail2Drive (closed loop, out of distribution)

HM is the harmonic mean of DS and SR. Every printed HM reproduces from its DS and SR to ±0.05 (checked here).

| Method | Base DS | Base SR | Base HM | Gen. DS | Gen. SR | Gen. HM |
|---|---:|---:|---:|---:|---:|---:|
| UniAD | 47.5 | 36.3 | 41.2 | 44.0 | 27.6 | 33.9 |
| TF++ | **83.3** | 78.5 | **80.8** | **75.4** | 61.1 | 67.5 |
| ORION | 53.0 | 52.0 | 52.5 | 51.2 | 46.0 | 48.5 |
| HiP-AD | 74.1 | 70.7 | 72.4 | 67.1 | 56.7 | 61.5 |
| SimLingo | 82.6 | 79.3 | **80.9** | 71.7 | 55.0 | 62.2 |
| **RoG-DAgger** | 80.2 | **80.0** | 80.1 | 74.7 | **66.0** | **70.1** |

### Table 9: Fail2Drive generalization, harmonic mean by scenario class

| Scenario class | SimLingo | RoG-DAgger | Δ |
|---|---:|---:|---:|
| Visual-longitudinal | 71.1 | 81.2 | +10.1 |
| Visual-lateral | 45.9 | 51.3 | +5.4 |
| Behavior | 31.2 | 62.0 | **+30.8** |
| Robustness | 86.8 | 80.7 | −6.1 |
| Mean | 58.8 | 68.8 | +10.0 |

### Table 11: RoG-DAgger on Fail2Drive, per scenario

Rel. = relative change from base to generalization, in percent.

| Class | Scenario | Base DS | Gen. DS | Rel. | Base SR | Gen. SR | Rel. |
|---|---|---:|---:|---:|---:|---:|---:|
| Behavior | Wall | 98.6 | 68.5 | −30.5 | 100.0 | 20.0 | −80.0 |
| Behavior | Fully Blocked | 94.1 | 83.8 | −11.0 | 100.0 | 80.0 | −20.0 |
| Behavior | Pedestrians On Road | 82.6 | 64.8 | −21.5 | 80.0 | 60.0 | −25.0 |
| Behavior | Construction Pedestrian | 54.1 | 67.4 | +24.6 | 40.0 | 60.0 | +50.0 |
| Visual-lat | Bad Parking | 79.3 | 71.0 | −10.5 | 70.0 | 60.0 | −14.3 |
| Visual-lat | Construction Permutations | 76.4 | 54.3 | −28.9 | 100.0 | 40.0 | −60.0 |
| Visual-lat | Custom Obstacles | 51.8 | 55.6 | +7.2 | 30.0 | 30.0 | 0.0 |
| Visual-lon | Obscured Stop | 95.4 | 99.4 | +4.2 | 100.0 | 100.0 | 0.0 |
| Visual-lon | Hard Brake | 83.3 | 73.8 | −11.4 | 80.0 | 60.0 | −25.0 |
| Visual-lon | Right Of Way | 90.9 | 83.7 | −8.0 | 100.0 | 80.0 | −20.0 |
| Visual-lon | Animals | 78.6 | 82.1 | +4.6 | 70.0 | 80.0 | +14.3 |
| Visual-lon | Pedestrian Other Blocker | 73.8 | 74.1 | +0.4 | 80.0 | 80.0 | 0.0 |
| Robustness | Right Construction | 93.7 | 72.5 | −22.7 | 100.0 | 60.0 | −40.0 |
| Robustness | Opposite Construction | 98.5 | 98.6 | +0.2 | 100.0 | 100.0 | 0.0 |
| Robustness | Image On Object | 79.9 | 73.5 | −8.0 | 100.0 | 80.0 | −20.0 |
| Robustness | Passable Obstacles | 92.5 | 97.5 | +5.3 | 100.0 | 100.0 | 0.0 |
| Robustness | Pedestrian Crowd | 71.8 | 64.5 | −10.2 | 80.0 | 60.0 | −25.0 |

No per-scenario SimLingo table is given, so the per-scenario effect of post-training is not visible; only the class-level Table 9 compares the two.

### Table 10: Bench2Drive efficiency, comfort and multi-ability

| Method | Efficiency | Comfort | Merging | Overtaking | Emergency Brake | Give Way | Traffic Sign | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| *Privileged* | | | | | | | | |
| PDM-Lite | 213.47 | 22.33 | 82.05 | 91.11 | 81.67 | 80.00 | 70.97 | 81.16 |
| RoG-Expert | 196.89 | 14.13 | 75.00 | 84.44 | 93.33 | 40.00 | 88.95 | 76.35 |
| *Pre-trained only* | | | | | | | | |
| UniAD | 129.21 | 43.58 | 12.16 | 20.00 | 23.64 | 10.00 | 13.89 | 15.89 |
| TF++ | – | – | 58.75 | 57.77 | 83.33 | 40.00 | 82.11 | 64.39 |
| ORION | 151.48 | 17.38 | 25.00 | 71.11 | 78.33 | 30.00 | 69.15 | 54.72 |
| HiP-AD | 203.12 | 19.36 | 50.00 | 84.44 | 83.33 | 40.00 | 72.10 | 65.98 |
| AutoVLA | 146.93 | 39.33 | – | – | – | – | – | – |
| SimLingo | 259.23 | 33.67 | 54.01 | 57.04 | 88.33 | 53.33 | 82.45 | 67.03 |
| *Post-trained* | | | | | | | | |
| MindDrive | – | – | 32.89 | 75.56 | 68.33 | 50.00 | 57.89 | 56.94 |
| TakeAD | 193.30 | 22.89 | 30.77 | 35.56 | 56.67 | 50.00 | 42.02 | 43.00 |
| TakeVLA | 249.01 | 30.27 | 63.64 | 64.44 | 91.67 | 50.00 | 85.48 | 71.05 |
| **RoG-DAgger** | 242.84 | 35.10 | 67.09 | 64.44 | 85.00 | 50.00 | 84.74 | 70.25 |

Against SimLingo, RoG-DAgger gains on merging (+13.1), overtaking (+7.4) and traffic signs (+2.3), and loses on emergency braking (−3.3) and give way (−3.3). Against TakeVLA it is stronger on merging and weaker on emergency braking. Comfort 35.10 is the highest among the post-trained rows. The RoG-Expert row explains part of the student's give-way score: the FoV-blinded expert scores 40 there, against PDM-Lite's 80. The text calls this "Tab. 11"; the caption numbers it Table 10.

### Table 3: ablations on Bench2Drive

Δ columns computed here, against the full method and against the actual base (86.59 / 66.36).

| Variant | Traj.-speed | Solv.-aware | FoV align. | DS | SR | Δ full (DS / SR) | Δ base (DS / SR) |
|---|:---:|:---:|:---:|---:|---:|---:|---:|
| No solution-space expansion | − | + | + | 88.71 | 68.94 | −1.63 / −4.57 | +2.12 / +2.58 |
| No solvability-aware takeover | + | − | + | 88.65 | 69.86 | −1.69 / −3.65 | +2.06 / +3.50 |
| No FoV alignment | + | + | − | 88.96 | 69.86 | −1.38 / −3.65 | +2.37 / +3.50 |
| Post-takeover samples only | | | | 88.05 | 68.04 | −2.29 / −5.47 | +1.46 / +1.68 |
| PDM-Lite DAgger baseline (PDM-Lite expert, infraction + restart triggers) | | | | 87.85 | 66.66 | −2.49 / −6.85 | +1.26 / +0.30 |
| **RoG-DAgger (full)** | + | + | + | **90.34** | **73.51** | – | +3.75 / +7.15 |

---

## Visualizations

![[qua_example_final.png|Figure 5: Qualitative comparison of RoG-DAgger with SimLingo on a Bench2Drive scene. At t1, both policies approach a junction while a cross-traffic fire truck is already visible. Both policies continue to accelerate, likely because the pre-training data contains primarily successful junction traversals and lacks corrective supervision for such hazardous interactions. As a result, SimLingo keeps accelerating at t2 and collides with the fire truck at t3. In contrast, after RoG-DAgger post-training, the policy slows down and shifts to the left, avoiding the collision. Red dots denote the predicted geometric trajectory, while green dots denote the predicted temporal trajectory. "thr": throttle, "brk": brake.]]

**Figure 5**: SimLingo 5.6 → 6.8 → 9.0 m/s at full throttle and collides with the cross-traffic fire truck. RoG-DAgger 6.0 → 7.3 → 6.0 m/s, braking fully from t2 and shifting left; the junction clears.

![[h2h_pede.png|Figure 9: Head-to-head comparison of RoG-DAgger (bottom) and SimLingo (top) in the Construction Pedestrian scenario of Fail2Drive. SimLingo tends to steer left despite oncoming traffic approaching from the left while the right side of the construction site is completely free. This behavior is consistent with a training bias in which construction zones predominantly appear on the right side of the road. As a result, SimLingo may resort to a late evasive maneuver and collide with the traffic warning sign. In contrast, RoG-DAgger learns to steer right in the same situation, likely attributable to the RoG-Expert, whose expanded solution space exposes the student to multiple feasible recovery strategies.]]

**Figure 9**: SimLingo accelerates 3.2 → 8.9 m/s, swerves late and hits the warning sign at t4 = 1.5 s. RoG-DAgger passes on the right at about 15 m/s.

![[h2h_blocked.png|Figure 10: Head-to-head comparison of RoG-DAgger (bottom) and SimLingo (top) in the Fully Blocked scenario of Fail2Drive. SimLingo continues to accelerate from t1 to t3 toward the fire truck blocking the road, likely reflecting a lack of corrective experience for such safety-critical situations, ultimately resulting in a collision at t4. In contrast, RoG-DAgger is post-trained on safety-critical scenarios induced by the policy and paired with expert demonstrations; this preventive supervision enables the model to decelerate as it approaches the fire truck and avoid the collision. Recovered from the arXiv HTML.]]

**Figure 10**: SimLingo 10.5 → 11.8 → 12.4 m/s, collision at 0.75 s. RoG-DAgger 6.2 → 2.9 → 0.7 → 0.0 m/s.

![[b2d.png|Figure 11: Qualitative examples of RoG-DAgger on Bench2Drive. In row (A), RoG-DAgger successfully merges into traffic while executing a left turn, using emergency braking to avoid a collision with the lead vehicle. In row (B), the predicted geometric trajectory adapts laterally from left to right as the pedestrian moves across the route. This contrasts with PDM-Lite/SimLingo-style behavior, where the geometric path remains largely fixed and collision avoidance is handled primarily through speed adjustment. Row (C) shows a failure case: RoG-DAgger slows down and attempts a lane change to pass through the construction area, however, due to the limited field of view, a vehicle approaching from the rear-left remains unobserved and causes a side collision.]]

**Figure 11**: (A) merge on a left turn, 15.1 → 0.4 m/s behind the lead car. (B) the path bends around a crossing pedestrian at 1.3–2.5 m/s. (C) failure: a lane change at a construction site, side collision with a car from the rear-left, which the front camera cannot see.

![[f2d.png|Figure 12: Qualitative examples of RoG-DAgger on Fail2Drive. In row (A), although RoG-DAgger has never been trained on scenes involving animals, the model appears to recognize that the road ahead is no longer safely drivable and begins braking in time to avoid the animal. A similar behavior is observed in row (B): despite not encountering children's slides during either pre- or post-training, the post-trained policy successfully navigates the constrained space, performs a lane change, and passes around the slide obstructing the road. Row (C) shows a failure case: although the model appears to recognize the wall blocking the road once it is at close range, it fails to brake sufficiently and ultimately collides with the obstacle. Recovered from the arXiv HTML.]]

**Figure 12**: (A) brakes for an animal, 9.6 → 0.2 m/s. (B) passes a children's slide with a lane change, 2.6 → 10.0 m/s. (C) failure: an image-textured wall; braking from 10.4 m/s starts too late and it collides (Wall: generalization SR 20%).

---

## Table Audit {#table-audit}

1. **The headline is measured against a different model.** Table 1's SimLingo row is the published InternVL-2 result (85.07 / 67.27, cited to SimLingo). The model post-trained is the authors' Qwen3-VL-2B version (86.59 / 66.36, Table 4), which the paper says it calls "SimLingo" throughout. Against it RoG-DAgger is +3.75 DS / +7.15 SR, not +5.3 / +6.2. The backbone swap the paper calls "no significant performance difference" is itself 1.52 DS, larger than three of its five ablation effects.
2. **The Longest6 v2 and Fail2Drive SimLingo rows have no stated source.** Whether "22 → 44" and "55 → 66" compare against the published model or the Qwen3 base is not said. The TF++, HiP-AD, UniAD and ORION rows are presumably copied from LEAD and Fail2Drive, which the wiki has not ingested.
3. **SimLingo's ability row fits averaged runs.** Table 10's SimLingo row differs from the one [[sources/linkvla.md]] prints in every ability column (merging 54.01 vs 53.75, overtaking 57.04 vs 68.89, emergency brake 88.33 vs 81.67, give way 53.33 vs 50.00, traffic sign 82.45 vs 82.11), with DS, SR, efficiency and comfort identical. Overtaking 57.04 is not a count out of 45 routes and give way 53.33 not one out of 10, but both fit three-run averages (77 / 135 and 16 / 30). So in this row, one ability score is not one run ([[concepts/bench2drive.md#rog-dagger]]). The other post-trained rows fit single runs (overtaking 64.44 = 29 / 45; give way 50.00).
4. **Table 11 does not reproduce Table 2 unweighted.** The plain means of the 17 scenarios are 82.1 DS / 84.1 SR (base) and 75.6 / 67.6 (generalization), against Table 2's 80.2 / 80.0 and 74.7 / 66.0. The per-scenario route counts are not given, so a route weighting may explain it. Table 9's class harmonic means do reproduce from Table 11's class means for three classes (within 0.1); visual-lateral gives 50.4 against the printed 51.3.
5. **Bench2Drive comparison scope.** The table omits [[sources/linkvla.md]] (91.01 / 74.55, March 2026), above RoG-DAgger, and [[sources/sparsedrivev2.md]] (89.15), [[sources/dynvla.md]] (88.34), [[sources/automot.md]] (87.34) and [[sources/deepsight.md]] (86.23), which are below it. "Strong performance across both benchmarks" holds within the table. Against TakeVLA, the one comparable post-training method, the difference is +0.62 DS and −0.22 SR in single runs.
6. **The ablation sum exceeds the total.** The three component removals cost 1.38–1.69 DS each (sum 4.70), while the PDM-Lite DAgger baseline, which lacks all three and the solvability trigger, is only 2.49 below the full method. Strong interactions or run-to-run noise; single runs cannot separate them.
7. **Small slips.** "DoG-DAgger" (§3.2); "Figs. 7 and 7" for Figures 6 and 7, and "Figs. 10 and 10" for Figures 9 and 10; the Bench2Drive detail table is cited as Tab. 11 and captioned Table 10; Table 1's caption uses "RS" for route completion; Figure 5's caption lost its time stamps in the clipping; the HTML places Figure 11's caption under the Figure 9–10 block as well.

---

## Limitations

The paper's own:
- Actor rollouts are non-reactive, so a candidate judged solvable may fail when actors react, and the PONR is not a physical feasibility boundary.
- The candidate set (lateral shifts, three speeds) may miss feasible avoidance plans, so a failed search does not prove that none exists.
- Data collection depends on CARLA. Extending it to a data-driven simulator such as AlpaSim is left for future work.

The wiki adds:

1. **Baseline choice** inflates the headline by 1.5 DS (audit 1), and the Longest6 and Fail2Drive baselines are of unstated origin (audit 2).
2. **Single runs throughout.** The five ablations differ from the full method by 1.4–2.5 DS. The wiki has no repeated full Bench2Drive run to set against them. Think2Drive's CARLA DS standard deviations were 1 to 12.2 over three runs ([[concepts/evaluation-variance.md]]). SR shifts of 3.7–6.9 points are larger, but are on 220 routes per run.
3. **"Weaker expert, better student" is confounded.** The PDM-Lite DAgger baseline also lacks the solvability-aware trigger and the FoV filter, as the authors say. PDM-Lite with the new trigger and FoV alignment, but without the expanded solution space, was not run, and neither was the expanded expert without FoV blinding.
4. **The new expert's label quality is not reported.** Table 6 shows PDM-Lite's pre-takeover plans colliding 29.6% of the time under the rollout. The same measurement for RoG-Expert, whose fallback is still full braking on the nominal path, is not given. That would show how many unsafe pre-takeover labels remain.
5. **One aggregation round.** Classical DAgger iterates: collect with the current policy, aggregate, retrain, repeat. RoG-DAgger collects once from the pretrained policy. Whether gains compound over rounds, or whether the 3:1 mix or the window lengths matter, is not tested.
6. **A front camera bounds both student and expert.** FoV alignment makes the expert as blind as the student. Give way falls to 40 for the expert and 50 for the student, and Figure 11(C) fails on a car from the rear-left. YieldToEmergencyVehicle is excluded from the failure analysis for the same reason. A multi-camera student would need a different alignment.
7. **Trade-offs on Fail2Drive.** Base-split DS falls 2.4, robustness scenarios lose 6.1 HM and the Wall scenario still fails (generalization SR 20%). The OOD gain is in SR (+11) more than DS (+3).
8. **Figure 1 shows shape, not frequency.** Both histograms are normalized within TTC ≤ 2 s, so they do not show how often safety-critical states occur in either setting.
9. **Two data sources are unablated.** Restarts make up 21.6% of the data and Replan Failure fallbacks 8.7%. Neither is ablated, and the latter labels states where the rollout found no collision-free plan.
10. **The privileged loop needs a simulator with ground truth.** Expert planning, R-TTC and the FoV filter all read CARLA's actor states and lane geometry. The same applies to any reconstruction-based simulator that wants to host it ([[concepts/data-driven-simulators.md]]).

---

## Relation to Other Wiki Pages

- [[concepts/bench2drive.md]]: second on the wiki's ladder (90.34), the first Longest6 v2 and Fail2Drive numbers in the wiki, and a SimLingo ability row that fits averaged runs.
- [[concepts/pdm-lite.md]]: the expert being modified. The paper documents how PDM-Lite plans: a precomputed route, non-reactive actor forecasts with unchanged control, and a target speed of zero as its only collision response. It also gives PDM-Lite's Longest6 v2 score (73 DS / 100 RC).
- [[sources/learning-to-drive-from-a-world-model.md]]: the other reward-free on-policy imitation result in the wiki. openpilot relabels every state the policy reaches with a learned future-conditioned plan model in a data-driven simulator. RoG-DAgger relabels only states around takeovers with a privileged rule-based expert in CARLA ([[concepts/rl-for-ad.md#rog-dagger]]).
- [[sources/simscale.md]]: off-log states labelled by a privileged rule-based planner (PDM-Closed on ground-truth state) in 3DGS, but sampled by perturbation rather than by the policy. Both papers find the teacher's properties shape the student: SimScale's comfort, RoG-DAgger's feasibility, timing and visibility.
- [[sources/clover.md]]: the closest analogue to the expanded solution space. CLOVER builds rule-based candidates (lateral shifts, speed profiles) from NAVSIM's metric cache and keeps those the evaluator accepts. RoG-Expert builds lateral shifts and junction speeds and keeps those its kinematic rollout accepts.
- [[sources/toad.md]]: also runs plans through a kinematic bicycle model, at test time as the search space. RoG-DAgger uses one at training time to validate labels.
- [[sources/think2drive.md]], [[sources/raw2drive.md]]: CARLA policies trained with a reward (privileged model-based RL; camera RL bootstrapped from it, 71.36 DS). RoG-DAgger reaches 90.34 with labels and no reward, from a different base. It is not a matched comparison.
- [[sources/rad.md]]: post-training in 3DGS with PPO plus interleaved imitation of the logged expert. RoG-DAgger's labels are on-policy expert labels instead of log labels.
- [[sources/linkvla.md]] (91.01, absent from the table), [[sources/sparsedrivev2.md]], [[sources/dynvla.md]], [[sources/automot.md]], [[sources/deepsight.md]], [[sources/orion.md]], [[sources/autovla.md]]: Bench2Drive neighbours. [[sources/lvdrive.md]] and [[sources/dynvla.md]] also carry the MindDrive row (78.04 / 55.09).
- [[concepts/alpasim-benchmark.md]]: the data-driven simulator named as the next host.
- Not ingested: **SimLingo** (the base model; a row in eight ingested source tables, at 85.07, 85.94 or 86.02 DS), **TakeVLA** and **TakeAD** (the DAgger predecessors; TakeVLA also post-trains SimLingo, with RL in reconstructed scenarios added), **MindDrive** (online RL), **LEAD** (FoV alignment and the target-speed rule), **Fail2Drive**, **"Hidden biases of end-to-end driving datasets"** (PDM-Lite and TF++), **HiP-AD**, **World on Rails** (the bicycle-model parameters), **DAgger** (Ross et al., 2011).

---

## Open Questions

- **How often are RoG-Expert's pre-takeover labels still collision-prone under the rollout?** Table 6 for the new expert would show how much of the 29.6% the expansion removes.
- **Is it the expert or the trigger?** Run PDM-Lite DAgger with solvability-aware takeover and FoV alignment but no expanded solution space, and the expanded expert without FoV blinding.
- **Does a second or third DAgger round add more?** Ross's DAgger iterates; one round is reported.
- **What is the gain against the actual base on Longest6 v2 and Fail2Drive?** And what are the run-to-run spreads of the full method and the ablations?
- **Does a reactive actor model change where the PONR falls, and does that change the student?** The paper names interaction-aware prediction with uncertainty as future work.
- **Does the loop transfer to a data-driven simulator?** It needs ground-truth actor states for the rollout and the trigger, which 3DGS reconstructions of logs have ([[sources/simscale.md]] runs PDM-Closed on them), and an observation of every policy-induced state.

---

**Figure note**: the clipping links eight figure files (`teaser 5.png` = Figure 2, `overall.png`, `expert_comparison_main_text.png`, `qua_example_final.png`, `no_replan.png`, `lb2_et.png`, `h2h_pede.png`, `b2d.png`). Figures 1, 7, 10 and 12 were missing and were recovered from the arXiv HTML (`ttc_compare.svg`, `replan.png`, `h2h_blocked.png`, `f2d.png`); Figure 1's values were read from a rendering of the SVG. All twelve figures are embedded, both algorithms and all eleven tables are reproduced, and computed Δ columns are added to Table 3.
