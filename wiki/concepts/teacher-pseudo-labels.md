---
title: Teacher-Derived Supervision and Pseudo-Labels
type: concept
sources: ["raw/papers/RoG-DAgger_ Rollout-Guided Post-Training for End-to-End Driving.md", "raw/papers/Enhancing End-to-End Autonomous Driving with Latent World Model.md", "raw/papers/WorldRFT_ Latent World Model Planning with Reinforcement Fine-Tuning for Autonomous Driving.md", "raw/papers/SparseDriveV2_ Scoring is All You Need for End-to-End Autonomous Driving.md", "raw/papers/CLOVER_ Closed-Loop Value Estimation & Ranking for End-to-End Autonomous Driving Planning.md", "raw/papers/DVGT-2_ Vision-Geometry-Action Model for Autonomous Driving at Scale.md", "raw/papers/HyWorldVLA_ A Vision-Language-Action Model with Hybrid World Modeling for Autonomous Driving.md", "raw/papers/Outcome-Grounded World Modeling for Autonomous Driving.md", "raw/papers/Raw2Drive_ Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving(in CARLA v2).md", "raw/papers/Think2Drive_ Efficient Reinforcement Learning by Thinking with Latent World Model for Autonomous Driving (in CARLA-v2).md", "raw/papers/SimScale_ Learning to Drive via Real-World Simulation at Scale.md", "raw/papers/Learning to Drive from a World Model.md", "raw/papers/PhysWAM_ Physically Consistent World Action Model for Autonomous Driving.md", "raw/papers/SUV_ Future Scene Understanding as Video Generation for End-to-End Driving.md", raw/papers/DriveDreamer-Policy_ A Geometry-Grounded World–Action Model for Unified Generation and Planning.md, raw/papers/CoWorld-VLA_ Thinking in a Multi-Expert World Model for Autonomous Driving.md, raw/papers/Latent-WAM_ Latent World Action Modeling for End-to-End Autonomous Driving.md, raw/papers/GeoWAM_ Visual Geometry World Action Models for Autonomous Driving.md, raw/papers/GeoWorldAD_ Geometry World Action Model for Autonomous Driving.md, raw/papers/FLARE_ Learning Future-Aware Latent Representations from Vision-Language Models for Autonomous Driving.md, raw/papers/Drive-HWM_ Hierarchical World Models for Dynamic-Latent Guided Autonomous Driving.md, "raw/papers/[-0.5mm] GRAVA GRAVA_ Grounded Reasoning-to-Action Representation and Learning for Autonomous Driving.md", raw/papers/Qwen-Drive-1.0_ An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving.md, raw/papers/ReCogDrive_ A Reinforced Cognitive Framework for End-to-End Autonomous Driving.md, raw/papers/HERMES_ A Holistic End-to-End Risk-Aware Multimodal Embodied System with Vision–Language Models for Long-Tail Autonomous Driving.md, raw/papers/Alpamayo-R1_ Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail.md, raw/papers/SGDrive_ Scene-to-Goal Hierarchical World Cognition for Autonomous Driving.md, raw/papers/How Can Driving World Models Do Counterfactual Prediction_.md, raw/papers/DA-WAM_ Decision-Aligned Future Latents for Driving World Models.md, raw/papers/Auto-JEPA_ A Latent World Model of Continuous Intent for End-to-End Autonomous Driving.md]
related: [sources/rog-dagger.md, concepts/pdm-lite.md, sources/law.md, sources/worldrft.md, sources/sparsedrivev2.md, sources/clover.md, sources/dvgt-2.md, sources/hyworldvla.md, sources/world4scorer.md, sources/raw2drive.md, sources/think2drive.md, sources/simscale.md, sources/learning-to-drive-from-a-world-model.md, concepts/data-driven-simulators.md, sources/physwam.md, sources/hydra-mdp-pp.md, sources/suv.md, sources/drivedreamer-policy.md, sources/coworld-vla.md, sources/latent-wam.md, sources/geowam.md, sources/geoworldad.md, sources/flare.md, sources/drive-hwm.md, sources/grava.md, sources/qwen-drive-1.0.md, sources/recogdrive.md, sources/hermes.md, sources/alpamayo-r1.md, sources/sgdrive.md, sources/driving-wm-counterfactuals.md, sources/da-wam.md, sources/auto-jepa.md, concepts/perception-for-planning.md, concepts/world-model-for-ad.md, concepts/reasoning-faithfulness.md, concepts/foundation-backbones-for-ad.md, concepts/vlm-domain-adaptation.md, research-directions.md]
created: 2026-09-27
updated: 2026-10-28
confidence: medium
---

# Teacher-Derived Supervision and Pseudo-Labels

## What It Is

Much of the supervision in recent driving models is not human annotation or sensor ground truth. It is the **output of a frozen foundation model** (a teacher) applied offline. Teachers now supply depth, segmentation, tracks, geometry, visual features, reasoning traces and QA pairs. This page records who uses which teacher, and the two recurring risks:
1. **Evaluation against the teacher.** The model is scored on agreement with the same teacher that labelled its training data.
2. **Teacher errors inherited silently.** Label noise is rarely measured.

---

## Teachers in the Wiki

| Teacher | Supplies | Used by | Evaluated against |
|---|---|---|---|
| **SAM 3** | Future segmentation and instance tracks, rendered as video | [[sources/suv.md]] | **The same teacher** (mIoU, AssA@50) |
| **Depth Anything 3** | Relative depth targets | [[sources/suv.md]], [[sources/drivedreamer-policy.md]] | **The same teacher** in both |
| **MapAnything with a LiDAR prior, + Depth Anything 3 for sky and hole filling** | Dense **metric** depth video for three views | [[sources/physwam.md]] | **LiDAR**, for both the labels (AbsRel 0.041–0.060 on supervised cells) and the generated depth |
| **Depth Anything V2** | Relative depth for abduction | [[sources/driving-wm-counterfactuals.md]] | Matched counterfactual GT |
| **VGGT / StreamVGGT / WorldMirror** | Geometry features, point maps | [[sources/coworld-vla.md]], [[sources/latent-wam.md]], [[sources/geowam.md]], [[sources/geoworldad.md]], [[sources/worldrft.md]] (frozen VGGT tokens fused into the encoder at inference, not a teacher) | Planning score only |
| **Grounded-SAM** (high-confidence masks only) | Semantic segmentation pseudo-labels for the image features, following World4Drive | [[sources/worldrft.md]] | Planning score only: alone 0.52 m / 0.08% nuScenes against VGGT alone 0.49 / 0.06; 0.01 m and 0.01 points on top of VGGT |
| **MoGe-2** (threshold-filtered) | Dense metric depth for point-map supervision, five datasets | [[sources/dvgt-2.md]] (and DVGT before it), and through DVGT-2 [[sources/geowam.md]]'s initialization | **Not stated**: the geometry tables give no evaluation ground truth, and the zero-shot baselines are LiDAR-aligned |
| **V-JEPA** | Future semantic features | [[sources/coworld-vla.md]] | Planning score only |
| **DINOv2** | Future feature targets | [[sources/flare.md]] | Planning score only |
| **Optical-flow estimator (unnamed)** | Slow-branch flow targets | [[sources/drive-hwm.md]] | Not evaluated; the estimator is not even identified |
| **LLM agents + LiDAR-assisted perception tool** | Grounded QA, reasoning graphs, metric states | [[sources/grava.md]] | An LLM judge against its own references |
| **Qwen3.7-Plus / Qwen3.5-Flash** | Reasoning traces; consistency filtering | [[sources/qwen-drive-1.0.md]] | Filter: **55.9% of public driving VQA survives** |
| **Qwen2.5-VL** | 775K auto-annotated NAVSIM QA | [[sources/recogdrive.md]] | Downstream planning |
| **Qwen3.6-plus** captions, embedded by Flan-T5 | Text guidance inside a video VAE whose latent is the world-model target | [[sources/hyworldvla.md]] | Planning score only (+0.24 PDMS); no reconstruction metric |
| **Frontier VLM** | Risk-aware plan rationales, baked into embeddings | [[sources/hermes.md]] | Downstream planning |
| **Future-anchored Plan Model** (the world model's own trajectory head, conditioned on 1 s of recorded frames starting up to 7 s after its 2 s context) | Action labels on the student's own on-policy states | [[sources/learning-to-drive-from-a-world-model.md]] | Closed-loop MetaDrive tests and real-world engagement; never against the teacher |
| **PDM-Closed** (rule-based planner on ground-truth state), filtered to trajectories with every EPDMS sub-metric at 1 | Expert trajectories on 237K simulated off-log states | [[sources/simscale.md]] | navhard and navtest; the teacher itself scores 51.3 on navhard, above every student (48.0 single, 50.9 ensemble) |
| **Log retrieval** (the logged human trajectory that best joins the perturbed state to the recorded state 4 s later) | Recovery trajectories on 147K simulated states | [[sources/simscale.md]] | navhard and navtest |
| **Think2Drive** (privileged BEV planner trained by model-based RL in CARLA v2) | Expert demonstrations: Bench2Drive's official base set; features distilled by DriveAdapter | [[sources/think2drive.md]]; Bench2Drive students such as [[sources/deepsight.md]] and [[sources/orion.md]] | Closed loop on Bench2Drive, where the teacher itself scores 91.85 DS and the best Think2Drive-protocol student 86.23 |
| **Think2Drive's world model and policy** (privileged, ground-truth BEV) | Target latents for a camera encoder, the dynamics, reward and continuation labels for imagined rollouts, policy initialization and action distillation | [[sources/raw2drive.md]] | Bench2Drive closed loop: 71.36 DS for the student against 91.85 for the teacher |
| **PDM-Lite as an on-policy DAgger expert**, and **RoG-Expert** (PDM-Lite with rollout-checked lateral-shift replanning and junction speeds, blinded to actors outside the student's front camera) | Trajectory-and-speed labels on the student's own closed-loop states: 2 s before an infraction, 2 s after a takeover, 4 s after a restart | [[sources/rog-dagger.md]] (SimLingo) | Bench2Drive: RoG-Expert alone 93.41 DS against PDM-Lite's 97.02, yet its student scores 90.34 against 87.85 for PDM-Lite's. PDM-Lite's own labels collide under a kinematic rollout in 29.6% of pre-takeover samples |
| **Roach** (privileged RL expert, CARLA 0.9.10) | 189K frames of expert driving, the TCP data protocol | [[sources/law.md]] (TCP head), and TCP-era CARLA v1 students | Town05 Long closed loop; the student's spread (sd 2.1–2.6 DS) is wider than its world-model effect |
| **NAVSIM PDM simulator** | Scorer targets, hard negatives, RL reward | [[sources/hydra-mdp-pp.md]] (origin: offline simulation of a vocabulary with GT perception), [[sources/da-wam.md]], [[sources/auto-jepa.md]], [[sources/world4scorer.md]] (online labels for all 64 generated candidates every step, plus a pre-labelled CLOVER bank; the labels also stand in for the unobserved futures of unexecuted plans), [[sources/clover.md]] (pseudo-expert trajectories built by rule from the metric cache, including future obstacle occupancy, and filtered by the evaluator; the scorer refit to the evaluator's labels on its own proposals every cycle), [[sources/sparsedrivev2.md]] (labels for the 400 composed trajectories its imitation-trained filter keeps per scene, computed during training; none at all for its Bench2Drive model), most RL papers | The benchmark's own metric; see [[concepts/selection-based-planning.md]] |

---

## Five Recurring Problems

### 1. Scoring the student against its own teacher

- [[sources/suv.md]]: all segmentation, depth and track metrics measure agreement with SAM 3 and DA3. The paper says so explicitly.
- [[sources/drivedreamer-policy.md]]: depth is both trained and evaluated against DA3 pseudo-labels.

Agreement with the teacher bounds quality from above only up to the teacher's own error, and it rewards reproducing the teacher's mistakes. **NAVSIM has LiDAR and annotated boxes, so real ground truth was available in both cases.**

A useful internal control from SUV: **generating the RGB future and running the teachers on it** beats native generation on segmentation and tracking, and loses on depth. At least one comparison is teacher-fair.

**[[sources/physwam.md]] is the counterexample that shows it can be done.** Its depth stream is trained on teacher labels (the flow-matching loss needs a dense target) and **evaluated against the LiDAR sweep of the same future frame**, with no scale alignment. Its second depth loss, Coupled Point Projection, bypasses the teacher and compares against raw LiDAR. So the teacher supplies density and the sensor supplies the truth, in both training and evaluation. The reported future-depth error (AbsRel 0.175 at +2 s) is an error against a measurement.

### 2. Label noise is almost never measured

- [[sources/qwen-drive-1.0.md]] is the exception. A consistency filter rejects **44%** of 24 public driving-VQA datasets against their own source annotations.
- [[sources/grava.md]]'s printed showcase transcript has a vehicle/pedestrian type flip and a 15–20 m vs 24.0 m distance conflict.

- [[sources/physwam.md]] is the second exception, and the first for a geometric teacher. It validates its finished depth labels against independently projected LiDAR on 1,000 windows covering all sixteen vehicles:

| View | AbsRel on LiDAR-supported cells | AbsRel, pixels ≤ 25 m | Sky mislabelled |
|---|---:|---:|---:|
| Front | 0.041 | 0.073 | 0.05% |
| Front-left | 0.051 | 0.113 | 0.22% |
| Front-right | 0.060 | 0.118 | 0.20% |

  It also reports what the raw teacher gets wrong before correction: MapAnything reads about 3% far at 10–25 m and 5% near at 60–80 m, and leaves about a fifth of non-sky pixels invalid. Both are fixed by an explicit bias fit and a hole-filling step. **The label error is the floor under the model's error**: in the front view, labels are at 0.041 (cells) and 0.073 (pixels within 25 m), and generated depth at +2 s is at 0.144 (cells) and 0.175 (pixels).

If public, human-curated corpora fail at 44%, agent-generated labels should not be assumed clean.

### 3. Privileged teachers leak into "annotation-free" claims {#privileged-teachers}

"No perception annotations" is often true while "no privileged supervision" is not:
- [[sources/da-wam.md]] and [[sources/auto-jepa.md]] use simulator-derived metrics.
- [[sources/grava.md]] uses LiDAR 3D states behind every number its front-camera model is trained to say.
- [[sources/sgdrive.md]] is camera-only at inference but needs occupancy and boxes to train.
- [[sources/physwam.md]] advertises "label-free" trajectory *selection* with "no learned scorer or simulator feedback", which is accurate for inference. Its training uses LiDAR, recorded poses, annotated 3D boxes and the HD map's drivable polygons, the last being "the same layers that the NAVSIM drivable-area metric treats as permissible". Its two hinge losses are differentiable stand-ins for the no-collision and drivable-area gates. The paper states the dependence plainly in its limitations.

Traces written with the future in view are a special case of the same leak; see [[concepts/reasoning-faithfulness.md#hindsight]].
- **A teacher at every level of training** *(2026-10-06)*. [[sources/raw2drive.md]] uses its privileged teacher as latent target, dynamics model, reward and termination source, policy initialization and action distiller, and calls the result reinforcement learning. Its gap to the teacher (20.5 DS on Bench2Drive) is larger than that of imitation students of the same teacher's data (5.6).
- **Every Bench2Drive student is taught by a privileged teacher.** [[sources/think2drive.md]] plans on ground-truth BEV; a camera student (TCP) trained on 200K of its frames keeps 36.4 of its 83.8 DS on the teacher's own benchmark. Later students on Bench2Drive come within 5.6 DS of it ([[concepts/bench2drive.md#expert]]).
- [[sources/worldrft.md]] *(2026-10-15)* calls its pretraining "perception annotation-free" (Grounded-SAM pseudo-labels, a frozen VGGT), which is accurate for that stage. Its RL reward is overlap between the ego box and every agent's annotated 3D box in future frames, and on nuScenes that overlap is exactly what the collision metric counts.
- [[sources/simscale.md]] labels simulated states with PDM-Closed run on ground-truth state, and its recovery teacher uses the recorded state 4 s ahead. Both are training-only and stated.

**A teacher that sees the future, used on purpose** *(2026-10-02)*. [[sources/learning-to-drive-from-a-world-model.md]] labels a student policy with a Plan Model that is conditioned on 1 s of recorded frames and poses that starts up to 7 s after the 2 s context. The student sees only the past 2 s. This is privileged supervision by design, and it is what makes on-policy training possible: the teacher can label states the human never visited because it knows where the human ended up, and so always plans a way back.
- **It is legitimate where hindsight traces are not.** The future is used only to produce training labels; nothing at deployment depends on it.
- **The cost is information the student cannot have.** Whether to change lanes is decided by the recorded future. The paper supplies it to the student as an explicit lane-change impulse, in training and at inference.
- **The teacher's quality is the student's ceiling** and is never evaluated on its own. The student's open-loop error against the human (0.394) is the worst of the three policies compared, which is expected when it imitates a model rather than the human. See [[concepts/data-driven-simulators.md#supervision]].

---

### 4. The student inherits the teacher's style {#inherited-style}

*(2026-10-06)* [[sources/simscale.md]] compares two teachers for the same off-log states and the students differ the way the teachers do.
- **A conservative teacher saturates early.** Retrieved human trajectories that steer back to the log help at small data sizes and then stop helping, or hurt, as more simulated states from the same scenes are added.
- **An exploratory teacher keeps paying, and passes on its weakness.** PDM-Closed scores EC 36.0 / 25.4 on navhard. Students trained on its labels lose Stage-2 extended comfort: LTF 70.6 → 42.9, DiffusionDrive 72.8 → 31.9. The paper names the cause.
- **A filter on the teacher is a filter by the benchmark.** Only teacher trajectories with every EPDMS sub-metric at 1 are kept, so the student is trained on the teacher's successes as the benchmark defines them.

### 5. A teacher's own score does not rank it as a teacher {#teacher-score}

*(2026-10-28)* [[sources/rog-dagger.md]] runs two privileged experts as DAgger teachers for the same camera student in CARLA.

| Teacher | Its own Bench2Drive DS / SR | Its own Longest6 v2 DS | Student after DAgger (DS / SR) |
|---|---|---:|---|
| PDM-Lite (conventional triggers) | 97.02 / 92.27 | 73 | 87.85 / 66.66 |
| RoG-Expert (new triggers) | 93.41 / 81.81 | 55 | **90.34 / 73.51** |

The student starts at 86.59 / 66.36. Three properties of the labels, none of which the teacher's own score measures, account for the difference in the paper's reading:
- **Feasibility from the student's state.** Re-executed from the states the student reached, PDM-Lite's labels collide in 29.6% of the 2 s before a takeover (43.2% of the last 1 s), against 0.47% after it. Its collision response is primarily braking along a fixed route.
- **Timing.** Takeover near the last state from which some plan still avoids the collision gives recovery labels where recovery is possible.
- **Observability.** The new expert plans only around actors the student can see, and pre-takeover labels caused by unseen actors are dropped. The new expert drives 18 DS worse than PDM-Lite on Longest6 and has half its give-way score (40 vs 80), which the paper attributes partly to this restriction.

The two runs also differ in trigger, so the table does not isolate the teacher. It is the opposite of the [[sources/simscale.md]] and Think2Drive cases on this page, where the open question is whether the student can reach the teacher at all. Here a teacher made worse as a driver made a better student. LEAD (not ingested) is cited for the same principle, minimizing learner–expert asymmetry.

## Why Teachers Still Win

The rise of teacher supervision is not an accident:
- **Annotation-free scaling.** SUV builds three extra target streams for all of navtrain at zero labelling cost.
- **Dense targets.** Point maps, features and flow give far more signal per frame than sparse boxes.
- **Measurable planning benefit.**
  - SUV: structured teacher streams give +1.0 navtest / +2.3 navhard even without inference access.
  - Latent-WAM: geometric distillation.
  - FLARE: a DINOv2 future-feature objective is the core of an 86.9 SFT / 91.4 after-RL pipeline, though the RL stage carries +4.5 of it.

---

## Open Questions

*Wiki-wide open questions are collected in [[research-directions.md]].*

- **Teacher vs. real ground truth, same planner.** Swap DA3 depth for LiDAR-projected depth on NAVSIM and compare both the future-depth metric and planning score. It is cheap and has never been done. *(2026-09-30: [[sources/physwam.md]] does half of it. Adding a raw-LiDAR point loss on top of teacher-labelled depth improves generated depth by 8–15% AbsRel and planning by +1.9 EPDMS. The loss also involves the generated motion, so the depth-only effect of LiDAR is not separated, and relative-versus-metric teachers are not compared.)*
- **Does teacher quality propagate to planning?** For example Depth Anything V2 vs V3, or SAM 2 vs SAM 3, under a fixed recipe.
- **What makes a privileged expert a good on-policy teacher?** [[sources/rog-dagger.md]] finds a field-of-view-blinded, worse-driving expert gives the better student, but changes the takeover trigger at the same time ([above](#teacher-score)). PDM-Lite with the new trigger and FoV alignment but no replanning, and the replanning expert without blinding, would separate the three properties.
- **Should agent-generated reasoning corpora carry a measured noise rate** in the style of Qwen-Drive's 55.9%, before any planning claim is made on them?
