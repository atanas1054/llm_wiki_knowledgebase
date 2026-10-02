---
title: Teacher-Derived Supervision and Pseudo-Labels
type: concept
sources: ["raw/papers/Learning to Drive from a World Model.md", "raw/papers/PhysWAM_ Physically Consistent World Action Model for Autonomous Driving.md", "raw/papers/SUV_ Future Scene Understanding as Video Generation for End-to-End Driving.md", raw/papers/DriveDreamer-Policy_ A Geometry-Grounded World–Action Model for Unified Generation and Planning.md, raw/papers/CoWorld-VLA_ Thinking in a Multi-Expert World Model for Autonomous Driving.md, raw/papers/Latent-WAM_ Latent World Action Modeling for End-to-End Autonomous Driving.md, raw/papers/GeoWAM_ Visual Geometry World Action Models for Autonomous Driving.md, raw/papers/GeoWorldAD_ Geometry World Action Model for Autonomous Driving.md, raw/papers/FLARE_ Learning Future-Aware Latent Representations from Vision-Language Models for Autonomous Driving.md, raw/papers/Drive-HWM_ Hierarchical World Models for Dynamic-Latent Guided Autonomous Driving.md, "raw/papers/[-0.5mm] GRAVA GRAVA_ Grounded Reasoning-to-Action Representation and Learning for Autonomous Driving.md", raw/papers/Qwen-Drive-1.0_ An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving.md, raw/papers/ReCogDrive_ A Reinforced Cognitive Framework for End-to-End Autonomous Driving.md, raw/papers/HERMES_ A Holistic End-to-End Risk-Aware Multimodal Embodied System with Vision–Language Models for Long-Tail Autonomous Driving.md, raw/papers/Alpamayo-R1_ Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail.md, raw/papers/SGDrive_ Scene-to-Goal Hierarchical World Cognition for Autonomous Driving.md, raw/papers/How Can Driving World Models Do Counterfactual Prediction_.md, raw/papers/DA-WAM_ Decision-Aligned Future Latents for Driving World Models.md, raw/papers/Auto-JEPA_ A Latent World Model of Continuous Intent for End-to-End Autonomous Driving.md]
related: [sources/learning-to-drive-from-a-world-model.md, concepts/data-driven-simulators.md, sources/physwam.md, sources/hydra-mdp-pp.md, sources/suv.md, sources/drivedreamer-policy.md, sources/coworld-vla.md, sources/latent-wam.md, sources/geowam.md, sources/geoworldad.md, sources/flare.md, sources/drive-hwm.md, sources/grava.md, sources/qwen-drive-1.0.md, sources/recogdrive.md, sources/hermes.md, sources/alpamayo-r1.md, sources/sgdrive.md, sources/driving-wm-counterfactuals.md, sources/da-wam.md, sources/auto-jepa.md, concepts/perception-for-planning.md, concepts/world-model-for-ad.md, concepts/reasoning-faithfulness.md, concepts/foundation-backbones-for-ad.md, concepts/vlm-domain-adaptation.md, research-directions.md]
created: 2026-09-27
updated: 2026-10-02
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
| **VGGT / StreamVGGT / WorldMirror** | Geometry features, point maps | [[sources/coworld-vla.md]], [[sources/latent-wam.md]], [[sources/geowam.md]], [[sources/geoworldad.md]] | Planning score only |
| **V-JEPA** | Future semantic features | [[sources/coworld-vla.md]] | Planning score only |
| **DINOv2** | Future feature targets | [[sources/flare.md]] | Planning score only |
| **Optical-flow estimator (unnamed)** | Slow-branch flow targets | [[sources/drive-hwm.md]] | Not evaluated; the estimator is not even identified |
| **LLM agents + LiDAR-assisted perception tool** | Grounded QA, reasoning graphs, metric states | [[sources/grava.md]] | An LLM judge against its own references |
| **Qwen3.7-Plus / Qwen3.5-Flash** | Reasoning traces; consistency filtering | [[sources/qwen-drive-1.0.md]] | Filter: **55.9% of public driving VQA survives** |
| **Qwen2.5-VL** | 775K auto-annotated NAVSIM QA | [[sources/recogdrive.md]] | Downstream planning |
| **Frontier VLM** | Risk-aware plan rationales, baked into embeddings | [[sources/hermes.md]] | Downstream planning |
| **Future-anchored Plan Model** (the world model's own trajectory head, conditioned on 1 s of recorded frames starting up to 7 s after its 2 s context) | Action labels on the student's own on-policy states | [[sources/learning-to-drive-from-a-world-model.md]] | Closed-loop MetaDrive tests and real-world engagement; never against the teacher |
| **NAVSIM PDM simulator** | Scorer targets, hard negatives, RL reward | [[sources/hydra-mdp-pp.md]] (origin: offline simulation of a vocabulary with GT perception), [[sources/da-wam.md]], [[sources/auto-jepa.md]], most RL papers | The benchmark's own metric; see [[concepts/selection-based-planning.md]] |

---

## Three Recurring Problems

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

**A teacher that sees the future, used on purpose** *(2026-10-02)*. [[sources/learning-to-drive-from-a-world-model.md]] labels a student policy with a Plan Model that is conditioned on 1 s of recorded frames and poses that starts up to 7 s after the 2 s context. The student sees only the past 2 s. This is privileged supervision by design, and it is what makes on-policy training possible: the teacher can label states the human never visited because it knows where the human ended up, and so always plans a way back.
- **It is legitimate where hindsight traces are not.** The future is used only to produce training labels; nothing at deployment depends on it.
- **The cost is information the student cannot have.** Whether to change lanes is decided by the recorded future. The paper supplies it to the student as an explicit lane-change impulse, in training and at inference.
- **The teacher's quality is the student's ceiling** and is never evaluated on its own. The student's open-loop error against the human (0.394) is the worst of the three policies compared, which is expected when it imitates a model rather than the human. See [[concepts/data-driven-simulators.md#supervision]].

---

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
- **Should agent-generated reasoning corpora carry a measured noise rate** in the style of Qwen-Drive's 55.9%, before any planning claim is made on them?
