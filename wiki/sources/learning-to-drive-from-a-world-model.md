---
title: "Learning to Drive from a World Model"
type: source-summary
sources: ["raw/papers/Learning to Drive from a World Model.md"]
related: [concepts/data-driven-simulators.md, concepts/world-model-for-ad.md, concepts/rl-for-ad.md, concepts/counterfactual-prediction.md, concepts/teacher-pseudo-labels.md, concepts/evaluation-variance.md, concepts/navhard-ood-evaluation.md, concepts/wam-attention-masks.md, concepts/perception-for-planning.md, concepts/foundation-backbones-for-ad.md, concepts/inference-latency.md, concepts/alpasim-benchmark.md, sources/dreameraD.md, sources/senna2.md, sources/driving-wm-counterfactuals.md, sources/ad-e2e-jepa.md, sources/physwam.md, sources/qwen-drive-1.0.md, sources/latent-wam.md, sources/simwam.md, sources/drivevla-w0.md]
created: 2026-10-02
updated: 2026-10-02
confidence: medium
---

# Learning to Drive from a World Model

**Paper**: Learning to Drive from a World Model
**Authors**: Mitchell Goff, Greg Hogan, George Hotz, Armand du Parc Locmaria, Kacper Raczy, Harald Schäfer, Adeeb Shihadeh, Weixing Zhang
**Orgs**: not captured by the clipping (the affiliation line is empty). The policies are deployed in **openpilot**, and the paper cites the authors' own comma2k19 dataset [26] and "Learning a driving simulator" [25].
**arXiv**: 2504.19077v1 (April 2025, older than most papers in the wiki)
**Code**: none stated for the method; the deployment target, openpilot, is described as open source
**Source**: `raw/papers/Learning to Drive from a World Model.md`

---

## What It Is

The wiki's first paper in which a world model is **the training environment, not part of the policy**. A small vision policy is trained **on-policy**: it drives inside a simulator built from real driving logs, sees the frames its own actions produce, and is supervised at every step by a model that knows where the human driver ended up. The trained policy then steers real cars in openpilot, a production Level 2 driver-assistance system.

Two simulators are compared, both data-driven:

| Simulator | How a new frame is made | Main limitation |
|---|---|---|
| **Reprojective** | Reproject the recorded image through a depth map to the new pose, inpaint the holes | Static scene, artifacts that grow with the pose offset, kept under 4 m |
| **World model** | A diffusion transformer (DiT) generates the next frame, conditioned on past frames, past poses, the commanded pose and **future frames from the log** | Under-executes commanded deviations; image quality |

Both use the same source of labels: a **Plan Model**, the world model's own trajectory head, which is conditioned on the recorded future and so always plans a route back toward where the human went ("recovery pressure").

**Results** (lateral control only):

| Policy | MetaDrive lane-centre tests | MetaDrive lane-change tests | Off-policy trajectory MAE | Field: engaged time / distance |
|---|---:|---:|---:|---|
| Off-policy (behaviour cloning) | 5/24 | 8/20 | **0.361** | not deployed |
| On-policy, reprojective simulator | **24/24** | **20/20** | 0.369 | 27.63% / 48.10% |
| On-policy, world-model simulator | **24/24** | 19/20 | 0.394 | **29.92% / 52.49%** |

---

## Key Takeaways

- **The open-loop metric ranks the three policies in reverse order of the closed-loop tests.** The imitation-only policy has the best trajectory error on held-out logs (0.361) and fails most closed-loop tests (5/24 lane-centre, 8/20 lane-change). Both on-policy policies pass almost everything while scoring worse open-loop. It is the wiki's cleanest single-paper demonstration of compounding error in behaviour cloning.
- **A world model can stand in for the simulator for on-policy training, and the result ships.** Both on-policy policies were deployed to openpilot and used over about two months by a cohort of 500 users (87,073 trips in total). The paper claims this is the first world-model-trained policy deployed in the real world.
- **The two simulators train equally good policies on the evidence given.** 24/24 for both, 20/20 against 19/20, and 0.369 against 0.394 in the reprojective policy's favour open-loop. The world-model policy is ahead in the field (+2.29 points of engaged time, +4.39 of engaged distance), but those cohorts are not matched and carry no error bars.
- **The supervision trick is a teacher that sees the future.** The Plan Model is conditioned on 1 s of recorded frames and poses that starts up to 7 s after the 2 s context. Offline, that future is available. The student policy never sees it. This is how a dataset of human drives yields labels for states the human never visited.
- **The reprojective simulator's failure modes are listed precisely**: static scene ("the counterfactual problem"), noisy depth, occlusions, lighting at night, a range of under 4 m, and **artifacts that correlate with the pose offset**, which a policy learns to read. The paper's remedy for the last one is a roughly 700-bit information bottleneck on the feature extractor.
- **The world model is checked against the commanded motion**, with a Pose Net whose own error floor is reported (Table 1). Under a commanded ±0.5 m lateral deviation, the generated video moves "but not to its full extent". The numbers are in a figure that is not in the clipping.
- **Almost nothing is ablated.** Future anchoring and noise-level augmentation are each called "essential", and the information bottleneck and vehicle-model randomization are motivated, but no table removes any of them.

---

## Method

### Problem setup

- **Policy** $\pi: h^\pi_T \mapsto p(a_{T+1}\mid h^\pi_T)$ with $h^\pi_T=((o_1,a_1),\dots,(o_T,a_T))$.
- **Action**: desired turning curvature and desired longitudinal acceleration.
- **Observation**: camera images only. Two cameras in practice (wide and narrow field of view); the paper shows the narrow one.
- **State** used by the simulator: the images plus a global 6-DoF pose $p_t=(x,y,z,\phi,\theta,\psi)$ from a tightly coupled GPS/vision Multi-State Constraint Kalman Filter.
- **Driving simulator** = a **driving state generator** (reprojection or a world model) + an **action ground-truth source** (the Plan Model).

### Vehicle Model

A function from actions to poses that models "vehicle dynamics, delayed steering response, wind, and more". Its parameters are randomized during training for sim-to-real transfer. Inverted, it turns a trajectory of poses into the actions that would produce it.

The world model is conditioned on **poses, not actions**. The paper's reason: the Vehicle Model can then be changed or randomized without retraining the world model.

### World model and Future Anchored World Model

$$w: h^w_T \mapsto p(o_T\mid h^w_T),\qquad h^w_T=\big((p_1,o_1),\dots,(p_{T-1},o_{T-1}),(p_T,\cdot)\big)$$

The **Future Anchored** version (after VPT [2]) also conditions on recorded frames and poses over a future window $F=(f_s,f_e)$ with $f_s>T$:

$$h^w_{T,F}=\big((p_{f_s},o_{f_s}),\dots,(p_{f_e},o_{f_e}),\ (p_1,o_1),\dots,(p_T,\cdot)\big)$$

It can only be used offline. Its rollouts produce "human-like driving video sequences and trajectories that converge to a goal state at $F$", which the paper calls **recovery pressure**.

### Plan Model: the action ground truth

The same network, given $h_{T,F}$, also predicts a trajectory $\mathcal T$ (positions, speeds, accelerations, orientations and orientation rates out to 10 s). The Vehicle Model maps it to actions. "Without [future anchoring], the Plan Model does not exhibit recovery pressure when in a bad state." The Plan Model can supervise a policy in either simulator.

### Reprojective simulation

Given a dense depth map $d_T$, pose $p_T$ and image $o_T$, render $o'_T$ at a new pose $p'_T$ by reprojecting the 3D points; in practice a history of images and depth maps is used and missing regions are inpainted. How the depth is obtained and which inpainting method is used are not stated.

![[reprojective_simulation_depth_.png|Reprojective simulation on a highway scene. Left: the recorded image at time T and its depth map. Right: four reprojected images at pose offsets of plus and minus 0.75 m lateral and plus and minus 4 m longitudinal, with stretched and duplicated regions visible near the image edges]]

*Figure 2: Left, the image at $T$ (top) and its depth map (bottom). Right, reprojections at $p_T+(0,\pm0.75,0)$ and $p_T+(\pm4,0,0)$. The offsets are read from the figure labels; the caption's symbols were lost in the clipping.*

**Limitations the paper lists** (Section 3.1):

| Limitation | What goes wrong |
|---|---|
| Static scene ("the counterfactual problem") | Other road users do not react; "swerving towards a neighboring car might cause the driver of the neighboring car to react" |
| Depth inaccuracies | Artifacts in the reprojected image |
| Occlusions | Regions hidden at $T$ must be inpainted |
| Reflections and lighting | Reprojection ignores light transport; worst at night (Figure 3) |
| Limited range | Artifacts grow with $p'_T-p_T$; translation kept "typically less than 4m", "especially limiting for longitudinal motion" |
| **Artifacts correlated with $p'_T-p_T$** | The policy uses them to predict the corrective action: "cheating or shortcut learning" |

![[night.png|Reprojection of a night highway scene. Left: the recorded frame at time T with headlights and reflective lane markings. Right: reprojections at plus and minus 0.5 m lateral offset, where the light reflections on the road are smeared into streaks]]

*Figure 3: Left, the image at $T$; right, reprojections at $p_T+(0,\pm0.5,0)$. Lighting artifacts.*

### World-model simulator architecture

- **Tokenizer**: the pretrained Stable Diffusion image VAE (`vae-ft-mse-840000-ema-pruned`), 8×8 spatial compression, 4 latent channels. Frames are downscaled to 128×256 before encoding (a 16×32×4 latent) and all data are at 5 Hz.
- **Backbone**: a DiT with a 3D patch table, flattened before the Transformer blocks. Sizes follow GPT-2 configurations: **250M, 500M and 1B** parameters.
- **Conditioning**: vehicle poses, world timesteps and diffusion noise timesteps are embedded, summed and fed to AdaLN, modified to take a different conditioning vector per frame (as in Navigation World Models [3]).
- **Mask**: frame-wise block-causal. Queries attend to keys in the same or earlier frames. Because the future anchor frames are **prepended**, every generated frame can read them, so the model is not physically causal. The mask exists for KV caching during sampling.
- **Plan Head**: a stack of residual feed-forward blocks predicting $\mathcal T$.

**Training objective.** Rectified flow with $\tau\sim\text{Logit-Normal}(0,1)$ and $o_\tau=\tau\epsilon+(1-\tau)o$. The Plan Head uses a multi-hypothesis (MHP) loss with 5 hypotheses, each a heteroscedastic Laplace NLL:

$$\mathcal L=\mathcal L_{\mathrm{RF}}+\alpha\,\mathcal L_{\mathcal T},\qquad \mathcal L_{\mathrm{RF}}=\|w(o_\tau,p,\tau)-(o-\epsilon)\|^2,\qquad \mathcal L_{\mathcal T}=\mathrm{MHP}\big(w(o_\tau,p,\tau),\mathcal T\big)$$

The value of $\alpha$ is missing from the clipping.

**Sampling.** Euler with 15 steps per frame ($\Delta\tau=1/15$); context latents are clean and given $\tau=0$. The pose $p_T$ can come from the Plan Head, a policy, the logged trajectory, or a hand-made perturbation. After each frame the context shifts by one and the new latent is appended, until $f_e$.

**Noise-level augmentation** against autoregressive drift, on 30% of training samples:
- context latents at $1,\dots,T-1$ are noised with $\tau\sim\text{Logit-Normal}(0,0.25)$;
- the future anchor latents are not noised;
- the model is still told $\tau=0$ for all context frames, so it is never told how corrupted its context is;
- the diffusion loss is computed only at frame $T$.

The paper calls this "essential" and notes that, unlike GameNGen [29], it did not need to discretize noise levels. No ablation is shown.

**Training samples.** A 2 s context; $f_s$ drawn between $T$ and 9 s; $f_e-f_s=1$ s. At 5 Hz that is 10 context frames, a 5-frame anchor and a gap of up to 35 generated frames. Data: 100k, 200k and 400k one-minute segments (about 1,700, 3,300 and 6,700 hours). Default model: 500M parameters on 400k segments.

### The policy

1. **Feature extractor**: FastViT, trained supervised to predict lane lines, road edges, lead-car information and the ego future trajectory. "Lane lines and road edges outputs are used for visualization, and never used as part of a steering policy."
2. **Temporal model**: a small Transformer over the frozen FastViT features of the last 2 s, predicting the same outputs plus the next action. Trajectory output: MHP with 5 hypotheses and a Laplace prior; other outputs: Laplace NLL.

Only the temporal model differs between the three policies compared.

**Information bottleneck.** The feature extractor's output is limited to "roughly 700bits" by adding white Gaussian noise during training: a Gaussian channel with per-sample capacity $\tfrac12\log(1+\mathrm{SNR})$, similar to Gaussian dropout but additive. Its stated purpose is to stop the policy exploiting simulator artifacts.

### On-policy learning

An IMPALA-style setup: parallel actors run rollouts with the latest policy from a parameter server; a central learner updates the policy. Each rollout is

$$h^{\pi,wp}=\big((o_1,a_1,\hat a^{wp}_1),\dots,(o_{f_s},a_{f_s},\hat a^{wp}_{f_s})\big)$$

where $a_t$ is the policy's action, $o_t$ is produced by the simulator in response, and $\hat a^{wp}_t$ is the Plan Model's action for the same history. The simulator "can be $wp$ itself, a different World Model $w$, or any driving simulator". Rollouts end at $f_s$. The learner trains $\pi: h^\pi_T\mapsto p(\hat a^{w}_T\mid h^\pi_T)$: **the policy imitates what the future-anchored model would do from where the policy actually is.** There is no reward.

---

## Figures

Figure 1 (the block diagram of one simulation step) is not in `raw/assets/`. Its caption: gray shapes are inputs to the world model only, black shapes are inputs to both the policy and the world model ("the Policy Model can be the World Model itself"), circles are actions (positions and orientations) and rectangles are observations.

Figures 2 and 3 are in the Method section.

![[0cf857_narrow_imgs_9.png|A single generated or context frame from the narrow front camera with a blue border: a suburban two-lane road with two cars ahead]]

*Figure 4 (one frame of five examples): blue-bordered frames are the last frames of the past context, red-bordered frames are the first frames of the future anchor, green-bordered frames are simulated. Per the caption, the simulated frames "comply with the future anchoring by executing lanes changes, or turning the traffic light to green." Only this blue-bordered frame was saved with the clipping.*

**Figure 5** (not in the clipping): LPIPS against model size (250M / 500M / 1B, on 400k segments) and against data size (100k / 200k / 400k, at 500M), in the action teacher-forced sequential-rollout setting. The paper draws its scaling claim from this figure; no values are available here.

**Figure 6** (not in the clipping): LPIPS for next-frame prediction with observations and actions teacher-forced (image quality) and for sequential rollout with actions teacher-forced (video quality). The caption and the text assign these to opposite panels.

**Figure 7** (not in the clipping): Pose Net errors of action teacher-forced rollouts.

![[deviation_RIGHT.png|One generated highway frame with a red curve and a blue line overlaid, taken from a rollout in which the world model is forced to deviate laterally from the lane]]

*Figure 8 (one of two panels): action noise-forced sequential rollout. A smooth lateral deviation of ±0.5 m is forced over the first 25 steps; the saved panel is the deviation to the right at step 25. The text does not say what the red and blue overlays are.*

**Figure 9** (not in the clipping): commanded lateral deviation (solid) against the deviation measured by the Pose Net on the generated video (dashed), averaged over 1,500 rollouts. The text's summary: "The World Model simulates the commanded deviation, but not to its full extent."

![[hugging.png|A MetaDrive render of a multi-lane road with cars close by on both sides of the ego vehicle]]

*Figure 10 (one example): a MetaDrive unit-test scenario.*

**Figure 11** (not in the clipping): per-scenario results of the lane-centre convergence and lane-change completion tests. Only the pass counts in Table 2 are available.

---

## Tables

### Table 1: Pose Net MAE on real segments, before and after VAE compression

| MAE | Not VAE-compressed | VAE-compressed | Increase |
|---|---:|---:|---:|
| x speed | 0.46366 m/s | 0.59390 m/s | +28% |
| y speed | 0.04216 m/s | 0.04393 m/s | +4% |
| z speed | 0.04424 m/s | 0.04548 m/s | +3% |
| roll rate | 0.00468 rad/s | 0.00524 rad/s | +12% |
| pitch rate | 0.00433 rad/s | 0.00453 rad/s | +5% |
| yaw rate | 0.00211 rad/s | 0.00254 rad/s | +20% |
| y lane lines | 0.15852 m | 0.15995 m | +1% |

The last column is computed here. The Pose Net is "a supervised model trained to predict a variety of outputs, such as pose, lane lines, road edges, lead car position". This table is the floor for any pose error measured on generated video: the VAE alone costs 28% on forward speed and 20% on yaw rate.

### Table 2: Policy evaluation

|  | Off-policy | Reprojective | World Model |
|---|---:|---:|---:|
| MetaDrive lane centre | 5/24 | 24/24 | 24/24 |
| MetaDrive lane change | 8/20 | 20/20 | 19/20 |
| Off-policy trajectory MAE | **0.361** | 0.369 | 0.394 |

The MAE's unit and horizon are not stated.

### Table 3: Field performance in openpilot

|  | Reprojective | World Model |
|---|---:|---:|
| Number of trips | 47,047 | 40,026 |
| Engaged % (time) | 27.63% | 29.92% |
| Engaged % (distance) | 48.10% | 52.49% |

"Approximately two months of driving from a cohort of 500 users." Steering is the end-to-end policy; longitudinal control is a classical adaptive cruise control using lead detection and radar.

### Results given only in the text

| Quantity | Value |
|---|---|
| LPIPS floor from VAE compression alone (test set) | **0.148** |
| World-model evaluation set | 1,500 rollouts from test segments |
| Noise model for the deviation test | ±0.5 m smooth lateral offset over the first 25 steps (5 s at 5 Hz), then released |
| Off-policy evaluation set | 1,500 held-out segments |
| Reprojection range | "typically less than 4m" in translation |
| Information bottleneck | about 700 bits per sample |

---

## Reading the Results

### 1. What Table 2 shows, and what it does not {#table-2}

- **It isolates the temporal model's training regime.** The frozen FastViT extractor is shared, so the 5/24 → 24/24 difference comes from training the temporal model on its own rollouts instead of on logs.
- **The off-policy failure is the textbook covariate-shift result**, and the open-loop column inverts it: 0.361 < 0.369 < 0.394 against 5/24, 24/24, 24/24. The world-model policy imitates a model rather than the human, so a larger gap to the human trajectory is expected.
- **The baseline is pure behaviour cloning.** No off-policy baseline with perturbation augmentation (shifted views with corrective labels) is included, so the table prices on-policy training against the weakest alternative.
- **The two simulators are not separated.** One lane-change scenario out of twenty is the only difference in the closed-loop tests.
- **The tests are in a third simulator.** Neither on-policy policy is tested in the simulator it trained in, which is good practice. But MetaDrive is a game-engine render, and how a policy trained on real camera images is fed MetaDrive frames (and from which of its two cameras) is not described. Pass criteria are not quantified.
- **The tests are lateral only.** Longitudinal motion is exactly where the reprojective simulator is weakest (range under 4 m) and where the world model's advantage would show, and it is not tested. The field system also uses classical longitudinal control.

### 2. What Table 3 can support

- The world-model policy was engaged for **29.92% of time against 27.63%** and **52.49% of distance against 48.10%**.
- The paper does not say whether the two policies were used by the same users, in the same period, on the same vehicles or routes, or whether assignment was random. The trip counts differ (47,047 against 40,026). Engagement in a Level 2 system depends on the driver, the road and trust. There are no intervals.
- The supported statement is the paper's own: both policies "are capable of delivering meaningful driver assistance in real-world conditions". It is not evidence that one simulator trains a better policy. See [[concepts/evaluation-variance.md#outside-navsim]].

### 3. The Plan Model is a privileged teacher {#privileged-teacher}

- It is conditioned on 1 s of recorded frames and poses that starts up to 7 s after the 2 s context. The student sees only the past 2 s and its own actions.
- **This is the same move as [[sources/ad-e2e-jepa.md]]'s oracle-goal search**, which picks the trajectory whose rollout lands nearest the recorded future frame. There, using the future disqualifies the result as a planner. Here it is legitimate, because the future is used only to label training data offline. Goal-conditioned inference that would be an oracle at test time is a valid teacher at training time.
- **Lane changes need a side channel.** The anchor decides whether a lane change happens (Figure 4's caption). A student that cannot see the anchor cannot know. The paper adds "a conditioning impulse ... prior to lane changes" in training and supplies the same impulse at inference to trigger one. That impulse is how the anchor's information reaches the student.
- Whether the reprojective policy is also labelled by the Plan Model is implied by Section 2.6 but not stated.
- See [[concepts/teacher-pseudo-labels.md#privileged-teachers]].

### 4. Future anchoring and counterfactuals {#future-anchoring}

- The Future Anchored World Model conditions on the history and on the **recorded continuation** of the episode, and then generates frames under a different ego motion. That is the conditioning structure [[sources/driving-wm-counterfactuals.md]] says a counterfactual query needs ($F^+$ as evidence) and that no other ingested model had.
- **It uses $F^+$ as a hard endpoint, not as evidence.** Whatever the policy does in the gap, the rollout must arrive at the recorded anchor. Figure 4's traffic light "turning ... to green" to agree with the anchor shows the world being bent to match the log. That is the static-scene assumption the paper criticizes in reprojection, moved from every frame to the anchor frames.
- **The two simulators are the two halves of the abduction design principle** on [[concepts/counterfactual-prediction.md#what-abduction-can-and-cannot-recover]]: reprojection is the geometric handler for surfaces that were observed, the world model is the generative prior for what was not.
- The paper does not evaluate counterfactual fidelity.

### 5. Does the world model follow the commanded pose? {#pose-following}

- It is measured with the right instrument: a Pose Net run on generated video, with its floor on real video, both raw and VAE-compressed (Table 1). This is the second action-following check in the wiki after [[sources/physwam.md]] (0.80° median yaw disagreement, floor 0.29°), and the first on an action-*conditioned* model rather than a joint one.
- **The answer is "partly".** Under a commanded ±0.5 m lateral deviation the generated video deviates less than commanded. The size is in Figure 9, which is not in the clipping.
- A possible cause, not tested by the paper: the anchor frames show the car where the human drove, which pulls the rollout back.
- **Why it matters for training.** If the frames show less deviation than the poses record, the policy's input and the Plan Model's input disagree about where the car is. The policy learns corrections against an attenuated world.

### 6. The shortcut problem {#shortcut}

- Any simulator whose rendering error grows with the deviation from the log hands the policy a feature that encodes the deviation, and the deviation is exactly what the corrective label depends on. The paper names this and caps the feature extractor at about 700 bits.
- **It is asserted, not shown.** No experiment demonstrates the exploitation, and no ablation removes the bottleneck. Whether the world-model simulator has the same problem (its errors may also correlate with the commanded offset; see section 5) is not discussed.
- The same mechanism is a candidate explanation for evaluation-side effects in rendered benchmarks; see [[concepts/navhard-ood-evaluation.md]] and [[concepts/data-driven-simulators.md#shortcut]].

### 7. What is new relative to the rest of the wiki

| | World-model papers elsewhere in the wiki | This paper |
|---|---|---|
| Role of the world model | Part of the policy, an auxiliary loss, a scorer, or a reward model | **The environment and the teacher** |
| Policy at inference | Often the world model itself (video WAMs) | FastViT + small Transformer; no generation |
| Supervision | Logged trajectory; RL reward from a simulator or learned reward model | Future-anchored Plan Model labels on the policy's own states |
| Evaluation | NAVSIM, navhard, HUGSIM, Bench2Drive, nuScenes | MetaDrive pass counts, held-out MAE, **real-world deployment** |

- The nearest neighbour is [[sources/dreameraD.md]], which also trains a policy inside a world model, but with GRPO against a learned latent reward and evaluated on NAVSIM.
- Every world-model cost is paid at training time, which makes this the limiting case of the training-time-only designs ([[sources/drivevla-w0.md]], [[sources/simwam.md]]): the world model shares no parameters with the deployed policy.

---

## Relationships

- **[[sources/dreameraD.md]]**: the other paper here that trains a policy in a world model's imagination. DreamerAD uses a learned reward and GRPO in latent space; this paper uses supervised labels from a future-conditioned head in VAE-latent video. Neither compares against the other's supervision.
- **[[sources/senna2.md]]**: closed-loop training in 3DGS reconstructions of 1,300 clips, with reacting agents, and longitudinal penalties instead of a policy gradient. The reconstruction-based counterpart to both simulators here.
- **[[sources/driving-wm-counterfactuals.md]]**: its evidence transport (monocular depth, lift, splat) is reprojective simulation; its abduction argument is what future anchoring half-implements. See [above](#future-anchoring).
- **[[sources/ad-e2e-jepa.md]]**: the recorded future frame as a goal. An oracle there, a teacher here. See [above](#privileged-teacher).
- **[[sources/physwam.md]]**: the other measured check that generated video follows an action, also with a measured floor.
- **[[sources/qwen-drive-1.0.md]]**: its AlpaSim reproduction shows NAVSIM's open-loop ranking inverting in a reactive closed loop. Table 2 here is the same inversion inside one paper, with three policies.
- **[[sources/latent-wam.md]]**: also a frame-wise block-causal mask; see [[concepts/wam-attention-masks.md]].

---

## Limitations

**Evidence**

1. **Lateral control only**, in tests and in the field. The longitudinal regime, where reprojection is weakest and the world model's advantage would show, is untested.
2. **No ablations** of future anchoring, noise-level augmentation, the information bottleneck or vehicle-model randomization, all of which the paper calls necessary.
3. **The simulators are not separated.** 24/24 against 24/24 and 20/20 against 19/20. The field difference comes from unmatched cohorts with no intervals.
4. **The off-policy baseline is pure behaviour cloning**, with no perturbation-augmented alternative.
5. **MetaDrive tests are small** (44 scenarios) and pass criteria are not quantified; the sim-to-sim transfer to game-engine graphics is not discussed.
6. **No safety metrics from the field**: engagement only, with no disengagement reasons, interventions or incidents.
7. **The world-model evaluation is mostly in figures** (5, 6, 7, 9) that the clipping does not contain, including the scaling result behind the paper's main forward-looking claim.

**Method**

8. **The world model under-executes commanded deviations**, so the policy is trained against attenuated consequences of its own actions.
9. **The anchor forces the world back to the log.** Reactions in the gap are possible in principle but must end at the recorded state; reactive behaviour is not evaluated.
10. **The shortcut problem is asserted, not measured**, and is not checked for the world-model simulator.
11. **Low fidelity**: 128×256 frames at 5 Hz; whether both cameras are generated is not stated.
12. **No cost figures**: world-model sampling (15 Euler steps per frame at 500M parameters), actor count, rollout volume and training compute are not given. Policy latency on the device is not given.

**Source conversion**

13. Five figure files are in `raw/assets/`. Figure 1, 5, 6, 7, 9 and 11 are missing; Figures 4, 8 and 10 have one panel or example each. All three tables are reproduced. The loss weight $\alpha$ and some symbols in the Figure 2 and 3 captions were lost. The Figure 6 caption and the text disagree on which panel is which. The affiliation line is empty.

---

## Key Cross-References

- [[concepts/data-driven-simulators.md]] — reprojection, reconstruction and learned world models as training environments; the failure-mode list this paper supplies.
- [[concepts/world-model-for-ad.md]] — the world model as the environment and teacher ([Pattern 44](../concepts/world-model-for-ad.md#world-model-as-environment)).
- [[concepts/rl-for-ad.md]] — on-policy imitation without a reward.
- [[concepts/counterfactual-prediction.md]] — the first model here conditioned on the factual continuation, used as an endpoint.
- [[concepts/teacher-pseudo-labels.md]] — a future-conditioned teacher for a past-only student.
- [[concepts/evaluation-variance.md]] — measurement floors (LPIPS 0.148, Table 1) and unmatched field cohorts.
- [[concepts/navhard-ood-evaluation.md]] — rendering artifacts that grow with the ego offset.
- [[concepts/wam-attention-masks.md]] — a block-causal mask with the future prepended.
- [[concepts/perception-for-planning.md]] — supervised perception targets for a frozen extractor, behind an information bottleneck.
