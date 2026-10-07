---
title: "Think2Drive: Efficient Reinforcement Learning by Thinking with Latent World Model for Autonomous Driving (in CARLA-v2)"
type: source-summary
sources: ["raw/papers/Think2Drive_ Efficient Reinforcement Learning by Thinking with Latent World Model for Autonomous Driving (in CARLA-v2).md"]
related: [sources/raw2drive.md, concepts/bench2drive.md, concepts/pdm-lite.md, concepts/rl-for-ad.md, concepts/world-model-for-ad.md, concepts/data-driven-simulators.md, concepts/teacher-pseudo-labels.md, concepts/evaluation-variance.md, sources/dreameraD.md, sources/learning-to-drive-from-a-world-model.md, sources/deepsight.md, sources/dynvla.md, sources/unidrivevla.md, sources/orion.md, sources/automot.md, sources/simscale.md, sources/senna2.md, research-directions.md]
created: 2026-10-06
updated: 2026-10-06
confidence: medium
---

# Think2Drive: Efficient Reinforcement Learning by Thinking with Latent World Model for Autonomous Driving (in CARLA-v2)

**Paper**: Think2Drive: Efficient Reinforcement Learning by Thinking with Latent World Model for Autonomous Driving (in CARLA-v2)
**Authors**: Qifeng Li, Xiaosong Jia (equal contribution), Shaobo Wang, Junchi Yan
**Orgs**: Shanghai Jiao Tong University
**arXiv**: 2402.16720v2 (first version February 2024; older than most of the wiki)
**Code**: none stated; a demo page (thinklab-sjtu.github.io/CornerCaseRepo)
**Source**: `raw/papers/Think2Drive_ Efficient Reinforcement Learning by Thinking with Latent World Model for Autonomous Driving (in CARLA-v2).md`

---

## What It Is

A **privileged planner trained by model-based reinforcement learning in CARLA Leaderboard v2**, and the expert that later generated the training data for [[concepts/bench2drive.md]] ("the Think2Drive protocol"). It is not an end-to-end driving model: its input is a 128×128 bird's-eye-view (BEV) rasterization built from simulator ground truth, and its output is one of 30 discrete throttle/steer/brake actions.

The method is DreamerV3 applied to driving:
1. The planner drives in CARLA and its experience goes into a replay buffer.
2. A recurrent state-space model (RSSM) learns the transitions, rewards and episode terminations of that experience in a compact latent space.
3. The planner (an actor-critic) is trained on 15-step **imagined rollouts in that latent space** ("thinking"), not on CARLA steps.

The paper's contribution is making this work on CARLA v2's 39 "corner-case" scenarios, which at the time no published method had solved. It lists seven problems and a fix ("brick") for each.

**Results** (privileged input throughout):

| Evaluation | Think2Drive | Baseline |
|---|---|---|
| CornerCaseRepo (390 short single-scenario routes) | **83.8 ± 1 DS**, 99.6 RC | Roach-style PPO with the same bricks: 57.5 ± 9 DS |
| CARLA v2 official test routes (7–10 km) | 56.8 DS, **98.6 RC**, 91.7 Weighted DS | PPO: 0.7 DS, 1.0 RC |
| CARLA v1 official test routes | 90.2 DS, 99.7 RC | Roach: 84.0 DS, 95.0 RC |
| 39 CARLA v2 scenarios, success rate | 0.61–1.0 (mean 0.84, computed here) | – |
| Bench2Drive (from [[sources/raw2drive.md]]'s and [[sources/dynvla.md]]'s tables, not this paper) | 91.85 DS / 85.41 SR; multi-ability mean 86.26 | PDM-Lite 97.02 / 92.27 |

Training takes **3 days on one A6000 GPU** (with a 128-core CPU).

---

## Key Takeaways

- **The wiki's earliest world model used as an RL environment, and its only one in a fully reactive simulator.** The planner learns entirely from imagined latent rollouts with predicted rewards and terminations. [[sources/dreameraD.md]] (latent RL on NAVSIM) and [[sources/learning-to-drive-from-a-world-model.md]] (imitation in a learned simulator of real video) are later variations on real-world logs.
- **Model-based beats model-free by a wide margin under the same bricks**: 83.8 vs 57.5 DS on CornerCaseRepo, and 56.8 vs 0.7 DS on the long CARLA v2 routes, where PPO stalls at 1% route completion. The baseline is the authors' own reimplementation.
- **Most of the contribution is engineering against RL failure modes.** Re-initializing the planner mid-training (keeping the world model) to escape local optima; a scenario generator and termination-priority replay against the long tail; a steering cost against wobble; a lane-following warm-up curriculum; a 4:1 planner-to-world-model update ratio; an asynchronous, parallel CARLA wrapper. The ablation figure is not in the clipping; the text says bricks 1 (reset) and 5 (warm-up) matter most.
- **An imitation student loses most of the teacher's skill.** TCP trained on 200K frames of Think2Drive driving, from cameras, scores 36.4 DS against the teacher's 83.8 on CornerCaseRepo. Later Bench2Drive students trained on Think2Drive data reach 86.23 DS ([[sources/deepsight.md]]) against the teacher's 91.85 there.
- **CARLA's Driving Score punishes long routes.** Infraction penalties multiply, so the same per-kilometre infraction rate gives a lower score on a longer route. The paper's Weighted Driving Score divides infraction counts by scenario density: 56.8 DS becomes 91.7 WDS on the v2 routes.
- **The world model is shown to work only by pictures.** Decoded BEV predictions stay plausible for 50 steps (Figure 2); no prediction or reward-error metric is reported.

---

## Method

### Problem formulation

Input $x_t$: privileged information (boxes of agents and obstacles, HD map, traffic-light states), rasterized to BEV masks plus an ego vector (speed, previous control, relative height). Output: throttle, steer, brake, discretized to 30 actions (Table 6). The "think" loop:

$$s_t \leftarrow F^{Enc}_\theta(x_t),\qquad a_t \leftarrow \pi(s_t),\qquad s_{t+1} \leftarrow F^{Pre}_\theta(s_t, a_t)$$

### World model (DreamerV3 RSSM)

| Component | Form |
|---|---|
| Sequence model | $h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1})$ |
| Encoder | $z_t \sim q_\theta(z_t \mid h_t, x_t)$ |
| Dynamics predictor | $\hat z_t \sim p_\theta(\hat z_t \mid h_t)$ |
| Reward predictor | $\hat r_t \sim p_\theta(\hat r_t \mid h_t, z_t)$ |
| Termination predictor | $\hat c_t \sim p_\theta(\hat c_t \mid h_t, z_t)$ |
| Decoder | $\hat x_t \sim p_\theta(\hat x_t \mid h_t, z_t)$ |

Loss $\mathcal L = \mathbb E\sum_t(\beta_\text{pred}\mathcal L_\text{pred} + \beta_\text{dyn}\mathcal L_\text{dyn} + \beta_\text{rep}\mathcal L_\text{rep})$, where $\mathcal L_\text{pred}$ is the negative log-likelihood of the input, reward (symlog) and termination (binary cross-entropy), and $\mathcal L_\text{dyn}$ / $\mathcal L_\text{rep}$ are free-bits KL terms between posterior and prior with the stop-gradient on opposite sides.

### Planner (actor-critic in imagination)

From a recorded start state, the world model and planner roll out $T = 15$ steps; the return beyond 15 is bootstrapped from the critic, $R_{T:} = v(s_T)$. The critic is trained with cross-entropy on two-hot symlog $\lambda$-returns; the actor maximizes normalized returns (divided by $\max(1, S)$, $S$ the decaying 5th–95th percentile range) plus an entropy bonus. All as in DreamerV3; the paper's novelty is "the first successful adoption of latent world model to AD".

### The seven challenges and bricks

| # | Challenge | Brick |
|---|---|---|
| 1 | Policy degradation: scenarios want contradictory behaviour (keep distance vs. merge assertively), so training settles in the easy local optimum | **Re-initialize all planner weights mid-training** (primacy-bias reset), keeping the world model, which supplies dense rewards to the fresh planner |
| 2 | The 39 scenarios are sparse in the released routes and tied to fixed waypoints | **Automated scenario generator** that splits routes and places scenarios by road layout; the CornerCaseRepo benchmark |
| 3 | Valuable transitions are rare (a 10 s red light gives 100 near-identical frames) | **Termination-priority replay**: half the samples end at an episode termination (the $K$ frames before it) |
| 4 | Stochastic agents wobble on straight roads | **Steering cost** (difference between consecutive steers) |
| 5 | Hard scenarios trap training in over-conservative behaviour | **Warm-up curriculum** on lane following and simple turns |
| 6 | The planner learns from delayed world-model rewards and converges slowly | **Incremental planner train ratio**, ending at 4× the world model's |
| 7 | A CARLA reset takes over 40 s | **Asynchronous reloading and parallel environments** (Figure 4; about one day saved per 100K steps) |

### Reward

$$r = r_\text{speed} + \alpha_\text{tr}\,r_\text{travel} + \alpha_\text{de}\,p_\text{deviation} + \alpha_\text{st}\,c_\text{steer}$$

Speed reward depends on distance to and type of nearby objects; travel reward is route progress per tick; deviation penalty is distance to the lane centre normalized by $D_\max$; steering cost is the change in steer. The $\alpha$ values, $D_\max$ and the termination conditions are not given (Figure 2's caption implies running a red light terminates an episode).

### Weighted Driving Score

$$\text{WDS} = \text{RC}\cdot\prod_{i=1}^{m}\text{penalty}_i^{\,n_i},\qquad n_i = \frac{\text{number of infractions of type } i}{\text{scenario density}}$$

with $n_i$ equal to the raw count when a route has no scenarios. The motivating example: at 0.2 infractions per km and penalty 0.8, a perfect-completion run scores 0.8 on a 5 km route and 0.64 on a 10 km route. How scenario density is measured (per km, per route) is not defined.

---

## Figures

*Figure 1 (bar charts of driving performance and infractions per kilometre on CornerCaseRepo) and Figure 3 (ablation learning curves over 500K steps) are not in the clipping. The 37 scenario illustrations in Appendix A are linked to arXiv and were not saved to `raw/assets/`.*

![[bev.png|Grid of decoded bird's-eye-view masks. Left two columns are context inputs at T=0 and T=5; the remaining nine columns are open-loop world-model predictions at T=10 to 50. Four success rows show multi-lane roads and an intersection evolving plausibly, with lanes in magenta, other vehicles in blue and traffic lights in red, yellow and green. A failure row at an intersection degrades into scattered noise from T=35 onward]]

*Figure 2: World-model predictions from 5 context frames, decoded to BEV masks out to step 50. In the failure row the planner runs a red light, the world model predicts termination, and later frames are noise.*

![[resetting.png|Diagram of the wrapped RL environment. A queue of resetting environments feeds a ring of running environments (env1 to env4) that run in parallel with the agent, which sends actions and receives observations and rewards. Finished or failed environments go back to the reset queue. On the right, the reset workflow: load a task and the town, generate a route, instantiate scenarios]]

*Figure 4: Asynchronous reset and parallel execution. A full reset takes about a minute in the large Town12 and Town13 maps.*

---

## Tables

### Table 1: CornerCaseRepo (privileged input unless noted)

Mean ± standard deviation over three runs. The infraction columns are per kilometre (Figure 1(b)'s caption).

| Input | Method | DS | Weighted DS | RC | Infraction penalty | Coll. pedestrians | Coll. vehicles | Coll. layout | Red light | Stop sign | Agent blocked |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Privileged | Roach (PPO + all bricks) | 57.5 ± 9 | 54.8 ± 0.5 | 96.4 ± 1.1 | 0.59 ± 0.28 | 0.85 ± 0.56 | 8.42 ± 4.65 | 0.85 ± 0.51 | 0.56 ± 0.45 | 0.49 ± 0.44 | 0.78 ± 0.31 |
| Privileged | **Think2Drive** | **83.8 ± 1** | **89.0 ± 0.2** | **99.6 ± 0.1** | **0.84 ± 0.01** | **0.16 ± 0.01** | **1.2 ± 0.5** | **0.29 ± 0.02** | **0.14 ± 0.01** | **0.03 ± 0.01** | **0.08 ± 0.01** |
| Raw sensors | Think2Drive + TCP (TCP trained on 200K Think2Drive frames) | 36.40 ± 12.23 | 29.6 ± 0.2 | 85.88 ± 8.26 | 0.41 ± 0.32 | 0.46 ± 0.32 | 9.92 ± 5.12 | 6.75 ± 3.08 | 3.27 ± 1.64 | 5.03 ± 3.82 | 6.18 ± 4.62 |

DS is close to RC × infraction penalty in every row (83.7, 56.9, 35.2 computed here). **The WDS standard deviations are 18–60 times smaller than the DS ones for the same runs** (0.5 against 9; 0.2 against 12.23), which the definition does not explain; see [Reading the Results §3](#variance).

### Table 2: Success rate on the 39 CARLA v2 scenarios

Success = 100% route completion with no infraction. A high success rate does not imply a high score (e.g. YieldToEmergencyVehicle can be completed without yielding).

| Scenario | SR | Scenario | SR | Scenario | SR | Scenario | SR |
|---|---:|---|---:|---|---:|---|---:|
| ParkingExit | 0.89 | HazardAtSideLane | 0.75 | VanillaTurn | 0.99 | InvadingTurn | 0.90 |
| SignalizedLeftTurn | 0.95 | SignalizedRightTurn | 0.76 | OppositeVehicleTakingPriority | 0.89 | OppositeVehicleRunningRedLight | 0.85 |
| Accident | 0.81 | AccidentTwoWays | **0.61** | CrossingBicycleFlow | 0.83 | HighwayCutIn | 1.0 |
| Construction | 0.84 | ConstructionTwoWays | 0.72 | InterurbanActorFlow | 0.83 | InterurbanAdvancedActorFlow | 0.8 |
| BlockedIntersection | 0.80 | EnterActorFlow | **0.65** | NonSignalizedRightTurn | 0.75 | NonSignalizedJunctionLeftTurnEnterFlow | **0.67** |
| MergerIntoSlowTraffic | **0.67** | MergerIntoSlowTrafficV2 | 0.87 | HighwayExit | 0.83 | NonSignalizedJunctionLeftTurn | 0.79 |
| SignalizedJunctionLeftTurnEnterFlow | 0.86 | VehicleTurningRoute | 0.78 | VehicleTurningRoutePedestrian | 0.75 | PedestrianCrossing | 0.91 |
| YieldToEmergencyVehicle | 0.92 | HardBrake | 1.0 | ParkingCrossingPedestrian | 0.98 | DynamicObjectCrossing | 0.94 |
| VehiclesDooropenTwoWays | 0.78 | HazardAtSideLaneTwoWays | 0.92 | ParkedObstacle | 0.90 | ParkedObstacleTwoWays | 0.91 |
| StaticCutIn | 0.85 | ParkingCutIn | 0.90 | ControlLoss | 0.78 | | |

Computed here: mean 0.84; 13 of 39 below 0.80. The four weakest are an opposite-lane bypass with a short window (AccidentTwoWays) and three merges into or across traffic flows. The number of trials behind each rate is not stated; values such as 0.61 and 0.89 are not multiples of 1/10 or 1/30, so they do not come from CornerCaseRepo's 10 evaluation routes per scenario (× 3 runs) alone.

### Table 3: Official test routes

| Method | Benchmark | DS | Weighted DS | RC % |
|---|---|---:|---:|---:|
| Roach (expert) | CARLA Leaderboard v1 | 84.0 | – | 95.0 |
| Think2Drive | CARLA Leaderboard v1 | 90.2 | 90.2 | 99.7 |
| PPO (expert) | CARLA Leaderboard v2 | 0.7 | 0.6 | 1.0 |
| Think2Drive | CARLA Leaderboard v2 | 56.8 | 91.7 | 98.6 |

Which v2 routes are "official test routes" is not stated here. *(2026-10-06: [[sources/raw2drive.md]], from the same group, reports Think2Drive at 56.8 DS / 98.6 RC on the Town13 **devtest** route and 43.8 / 78.2 on the **validation** route, so this row is devtest.)*

### Table 4: BEV channels ($C = 34$)

| Static (1 channel each) | Dynamic (4 history channels each) |
|---|---|
| road, route, ego, lane, yellow line, white line | vehicle, walker, emergency car, obstacle, green traffic light, yellow & red traffic light, stop sign |

6 static + 7 dynamic × 4 = 34.

### Table 5: Input and output dimensions

| H | W | C | T (history steps) | M (actions) |
|---:|---:|---:|---:|---:|
| 128 | 128 | 34 | 4, at steps −16, −11, −6, −1 | 30 |

### Table 6: The 30 discrete actions

| Throttle | Brake | Steer values |
|---:|---:|---|
| 0 | 1 | 0 (full brake) |
| 0.7 | 0 | −0.5, −0.3, −0.2, −0.1, 0, 0.1, 0.2, 0.3, 0.5 |
| 0.3 | 0 | −0.7, −0.5, −0.3, −0.2, −0.1, 0, 0.1, 0.2, 0.3, 0.5, 0.7 |
| 0 | 0 | −1, −0.6, −0.3, −0.1, 0, 0.1, 0.3, 0.6, 1 (coasting) |

Regrouped from the paper's three-column layout: 1 + 9 + 11 + 9 = 30.

### Table 7: Network

| GRU recurrent units | CNN multiplier | Dense hidden units | MLP layers | Parameters |
|---:|---:|---:|---:|---:|
| 512 | 96 | 512 | 5 | 104M |

A CNN maps the 128×128 BEV to a 4×4 feature map (4×4 kernels, stride 2); flattened features plus an MLP encoding of the ego vector feed the RSSM; the decoder mirrors the encoder.

### Results given only in the text

| Quantity | Value |
|---|---|
| CornerCaseRepo | 1,600 training routes (40 per scenario + 40 with none), 390 evaluation routes (10 per scenario), each < 300 m with one scenario type |
| CARLA v2 routes | 7–10 km, about 60 scenarios each; 90 official training routes |
| CARLA version | 0.9.14 |
| Imagination horizon | 15 steps |
| Decision rate | 10 FPS (from the red-light example) |
| CARLA reset time | > 40 s; about 1 minute in Town12 / Town13 |
| TCP student data | 200K frames, varied weather |

---

## Reading the Results

### 1. What the model-based vs model-free comparison controls {#mbrl-vs-ppo}

- **Matched where it matters**: the "Roach" row replaces only the RL algorithm (PPO instead of the world-model actor-critic) and keeps the bricks, routes and inputs. That makes 83.8 vs 57.5 a fair comparison of the two learners in this setup.
- **Not matched in tuning.** Every brick was designed for the model-based learner, and the paper's own explanation for PPO's failure on long routes (relearning from a noisy replay buffer after each reset) is about brick 1. A PPO setup tuned on its own terms is not tried. The v1 Roach row (84.0) appears to be the original Roach's number, not a rerun.
- **CornerCaseRepo is the authors' benchmark**, built by the same scenario generator that produces the training routes.

### 2. The student gap {#student-gap}

| Policy | Input | CornerCaseRepo DS | Bench2Drive DS |
|---|---|---:|---:|
| Think2Drive | Privileged BEV | 83.8 | 91.85 ([[sources/dynvla.md]]'s table) |
| TCP trained on Think2Drive driving | Cameras | 36.4 (200K frames) | 40.70 (Bench2Drive base set, also Think2Drive-generated; [[sources/dynvla.md]]'s table) |
| Best Think2Drive-protocol student in the wiki | Cameras | – | 86.23 ([[sources/deepsight.md]]) |

- **On the authors' own benchmark the student keeps 43% of the teacher's score.** Its infraction rates rise 8× for vehicle collisions, about 23× for layout collisions and red lights, and 77× and 170× for agent blocking and stop signs. The paper puts it down to perception and imitation.
- **Later Bench2Drive students closed most of that gap.** The best Think2Drive-protocol student in the wiki is 5.6 DS below its teacher. That bears on whether the expert bounds the benchmark; see [[concepts/bench2drive.md#expert]].

### 3. Three runs, and two very different spreads {#variance}

- Table 1 reports mean ± sd over three runs. Whether these are three trainings or three evaluations of one model is not stated ("evaluated on other 390 routes ... for 3 runs" suggests evaluations).
- **Closed-loop CARLA scores move a lot between runs**: DS sd 9 for PPO and 12.2 for TCP, against 1 for Think2Drive.
- **WDS barely moves on the same runs**: 0.5, 0.2 and 0.2. WDS multiplies RC by penalties raised to infraction counts divided by scenario density; with RC sd 1.1–8.3 and infraction sds of 0.3–5, an sd of 0.2 is hard to reproduce. The paper does not explain it.

### 4. Driving Score and route length {#ds-length}

- **The argument is correct as arithmetic.** CARLA's per-route DS multiplies a penalty per infraction, so a fixed infraction rate scores lower on longer routes. On the 7–10 km v2 routes, 98.6% completion becomes 56.8 DS.
- **WDS changes what is compared.** Dividing counts by scenario density makes infractions cheaper on routes with many scenarios; how density is computed is not given. It is the paper's own metric, and WDS 91.7 is not comparable with any published DS.
- **Bench2Drive's short routes are the benchmark-side answer to the same problem**: Think2Drive scores 91.85 DS there.

### 5. What "first" means here

- The paper calls itself the first model-based RL method for driving while citing two earlier ones (model-predictive policy learning in dense traffic; UMBRELLA). Its claim holds in the narrower form it also uses: the first to handle CARLA v2's scenarios.
- The abstract says "no reported success (100% route completion) on CARLA v2"; Think2Drive's own v2 route completion is 98.6%.
- The paper notes that PDM-Lite, a rule-based planner released after the first arXiv version, also solves all v2 scenarios, "with different hyper-parameters for different scenarios". On Bench2Drive PDM-Lite scores 97.02 against Think2Drive's 91.85 ([[concepts/pdm-lite.md]]).

### 6. Where it sits among world models used for training {#position}

| | Think2Drive | [[sources/dreameraD.md]] | [[sources/learning-to-drive-from-a-world-model.md]] |
|---|---|---|---|
| Environment the world model learns | CARLA, privileged BEV masks | Real NAVSIM video (Epona latents) | Real openpilot video |
| World model | DreamerV3 RSSM, 104M | Shortcut-forced video DiT | 500M DiT on VAE latents |
| Learning signal | Predicted reward + termination, actor-critic | Learned latent reward (AD-RM), GRPO | Future-anchored Plan Model labels, no reward |
| Real environment in the loop | Yes, CARLA collects new experience throughout | No, the logs are fixed | No, the logs are fixed |
| Reactive agents | Yes (CARLA) | No | No |
| Policy at test time | Privileged planner | Camera planner | Camera steering policy, deployed |

Think2Drive is the only one in which the world model keeps being refit on the current policy's own experience, which is standard Dreamer and the reason it can use a small, inaccurate-at-length model (15-step horizon, critic bootstrap). The other two learn from fixed logs and cannot see what their policy would do outside them.

---

## Relationships

- **[[concepts/bench2drive.md]]**: Think2Drive generated Bench2Drive's official training data ("officially generated using the Think2Drive paradigm", [[sources/unidrivevla.md]]); papers tag rows by "Think2Drive" or "PDM-Lite" expert protocol ([[sources/automot.md]], [[sources/deepsight.md]]). DriveAdapter distills Think2Drive's features ([[sources/orion.md]]).
- **[[sources/dreameraD.md]]**: the same "RL in a latent world model" idea, three years of world-model progress later, on real logs and NAVSIM; reward from a learned head in both.
- **[[sources/learning-to-drive-from-a-world-model.md]]**: a world model as the training environment, but with imitation labels and real video.
- **[[sources/senna2.md]]**: RL in a reactive environment built from real logs (3DGS); Think2Drive is RL in a reactive game engine through a learned model of it.
- **[[sources/simscale.md]]**: also uses a privileged rule-based planner, PDM-Closed, as a reference and teacher; Think2Drive is the learned counterpart in CARLA.
- **[[concepts/pdm-lite.md]]**: the other CARLA v2 expert.
- **[[sources/raw2drive.md]]**: the camera-input sequel from the same group. Think2Drive is its Stage I (reusable as is); a camera world model is trained into Think2Drive's latent space and the policy is fine-tuned by RL there. 71.36 DS on Bench2Drive, 20.5 below Think2Drive. Its tables are the source of Think2Drive's Bench2Drive multi-ability profile and its validation-route score.

---

## Limitations

**Evidence**

1. **Ablations have no numbers in the clipping** (Figure 3 is missing); the ranking of bricks is the text's.
2. **The baseline is the authors' reimplementation** with bricks designed for their own learner.
3. **CornerCaseRepo is the authors' benchmark**, generated by the same tool that generates the training routes.
4. **The v2 "official test routes" are not identified in this paper** (they match the Town13 devtest numbers in [[sources/raw2drive.md]]; the validation route gives 43.8 DS / 78.2 RC), and the v2 result relies on the paper's own metric (WDS) for its "expert-level" reading; DS is 56.8.
5. **Three runs of unclear kind**, and WDS standard deviations that look too small for the DS spread beside them.
6. **No world-model accuracy metric**: prediction, reward and termination errors are not reported; Figure 2 is qualitative.
7. Table 2's per-scenario trial counts are not given.

**Method**

8. **Privileged input only.** The planner sees ground-truth boxes, map and light states. Its value to sensor-based driving is as a teacher, and the paper's own camera student loses 47 DS.
9. **Discrete control** (30 actions) and a hand-shaped reward with unstated weights.
10. **Simulator-specific engineering** (scenario generator, reset wrapper) carries much of the result; the world model learns CARLA's simplified dynamics and behaviour models, not real traffic.
11. **No latency or inference cost** for the planner; training cost is given (3 days, one A6000).

**Source conversion**

12. Two of four figures are local (Figures 2 and 4); Figures 1 and 3 and the 37 scenario illustrations are missing. All seven tables are reproduced; Table 6 is regrouped by throttle level. The clipping's front-matter author field is empty; authors are taken from the body.

---

## Key Cross-References

- [[concepts/bench2drive.md]] — the expert behind the benchmark's base set; teacher vs. student scores; Driving Score and route length.
- [[concepts/world-model-for-ad.md]] — a latent world model as the RL environment for a privileged planner (Pattern 46).
- [[concepts/rl-for-ad.md]] — model-based vs. model-free RL in CARLA and the seven bricks.
- [[concepts/data-driven-simulators.md]] — the game-engine contrast: a learned model of CARLA rather than of real logs.
- [[concepts/pdm-lite.md]] — the two CARLA v2 experts.
- [[concepts/teacher-pseudo-labels.md]] — a privileged RL teacher and the size of the imitation gap.
- [[concepts/evaluation-variance.md]] — closed-loop run-to-run spread in CARLA.
