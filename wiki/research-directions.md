---
title: Research Directions
type: directions
sources: []
related: [concepts/evaluation-variance.md, concepts/navsim-benchmark.md, concepts/alpasim-benchmark.md, concepts/hugsim-benchmark.md, concepts/bench2drive.md, concepts/navhard-ood-evaluation.md, concepts/physicalai-av-benchmark.md, concepts/world-model-for-ad.md, concepts/wam-attention-masks.md, concepts/counterfactual-prediction.md, concepts/perception-for-planning.md, concepts/teacher-pseudo-labels.md, concepts/visual-tokenization.md, concepts/chain-of-thought-for-ad.md, concepts/reasoning-faithfulness.md, concepts/general-capability-retention.md, concepts/dual-system-vla.md, concepts/adaptive-routing.md, concepts/selection-based-planning.md, concepts/inference-time-safety.md, concepts/divergent-thinking-in-vlms.md, concepts/r1-zero-like-training.md, concepts/mixture-of-experts.md, concepts/inference-latency.md, concepts/best-of-n.md, concepts/foundation-backbones-for-ad.md]
created: 2026-09-30
updated: 2026-09-30
confidence: medium
---

# Research Directions

The wiki-wide list of open research questions, grouped by theme. Each entry states a question, what the ingested papers say about it so far, and where the detail lives. The "Open Questions" section of each concept page remains the full local record; this page collects the questions that matter across pages.

Every claim traces to a wiki page. The file is checked and updated on every ingest and every lint. Answered questions are moved to [Answered](#answered), not deleted.

Entry format:

```
### The question, as a heading
- **Status**: open | partially answered
- **Known so far**: the evidence to date, with numbers and [[page]] citations
- **See**: [[pages]] that discuss it
- **Updated**: YYYY-MM-DD
```

---

## 1. Evaluation and benchmark validity

### How large are the error bars on NAVSIM and navhard comparisons?
- **Status**: partially answered
- **Known so far**: Training-seed variance has never been measured. The three reported figures are sampler-seed noise: 0.053 over 10 seeds ([[sources/wa-jepa.md]]), 0.013 over 6 ([[sources/coworld-vla.md]]) and 0.24–0.30 ([[sources/physwam.md]]). Paired scene-sampling noise is measured once, within one model family: a 95% half-width of about 0.21 EPDMS on navtest ([[sources/drivereferee.md]]). Across two different published methods it is unmeasured, and no paper reports per-scene navhard scores, so no navhard effect in the wiki has an error bar.
- **See**: [[concepts/evaluation-variance.md]]
- **Updated**: 2026-09-30

### Can NAVSIM-v2 EPDMS be compared across papers at all?
- **Status**: open
- **Known so far**: There are three EPDMS lineages (the Hydra-MDP++ formula, the pre-fix evaluator and the corrected one), and at least eleven ingested v2 tables mix conventions. Two papers score Transfuser at 76.7 and 84.0 from identical sub-metrics. The residual-sign heuristic (+1 to +4 for pre-fix rows, slightly negative for corrected ones) identifies the protocol of most rows but has four exceptions. The current rule is to trust a column label only for the paper's own row.
- **See**: [[concepts/navsim-benchmark.md]], [[sources/hydra-mdp-pp.md]]
- **Updated**: 2026-09-30

### Is NAVSIM-v1 saturated?
- **Status**: open
- **Known so far**: Best-of-6 reaches 94.8 PDMS, the human ground-truth score (Curious-VLA). Single-pass scores now reach 93.7 (CLEAR, DA-WAM) and 94.0 with the val split added to training ([[sources/mm-future.md]]), and almost every entry above 92.0 trains a scorer or an RL reward against the benchmark's own metric. If oracle selection already matches the logged human, gains above about 93 may measure selection quality and not driving quality.
- **See**: [[concepts/best-of-n.md]], [[concepts/navsim-benchmark.md]], [[concepts/selection-based-planning.md]]
- **Updated**: 2026-09-30

### Do NAVSIM rankings survive a reactive closed loop?
- **Status**: partially answered
- **Known so far**: The only closed-loop reproduction of NAVSIM leaders inverts the order: in AlpaSim, SimWAM (91.5 PDMS) is last on at-fault score (0.30) and Alpamayo-R1, which reports no NAVSIM number, is first (0.58) ([[sources/qwen-drive-1.0.md]]). That is one group's reproduction of two methods not designed for the simulator. Of the entries above 92 PDMS, only [[sources/mm-future.md]] (93.4) has a closed-loop number: 32.3 HD-Score on HUGSIM, below WA-JEPA's 44.6 (91.8 PDMS). None has been run in AlpaSim or Bench2Drive, and no paper has run AlpaSim and HUGSIM side by side, so it is unknown whether the two closed-loop benchmarks agree with each other.
- **See**: [[concepts/alpasim-benchmark.md]], [[concepts/hugsim-benchmark.md]], [[concepts/bench2drive.md]]
- **Updated**: 2026-09-30

### How much of a navhard score is the generator and how much is the selector?
- **Status**: partially answered
- **Known so far**: [[sources/drivefuture.md]] is the only paper to report one checkpoint with and without a scorer: 30.9 → 34.6 (its world model) → 55.5 (GTRS-Dense over 100 proposals). Every navhard entry above 42 scores its own candidates and none of the others reports an unscored ablation. [[sources/momworld.md]] puts a predicted future inside a strong scorer and is +1.1 over GTRS-Dense's published score, but that base is not a matched rerun and the page is held at low confidence. Whether future conditioning survives a strong scorer is still open. On navtest there are two partial measurements: [[sources/physwam.md]] reports first sample, a medoid-of-8 selector and the oracle on one checkpoint (90.3 → 90.4 EPDMS for the selector), and [[sources/mm-future.md]] prices a learned scorer at +8.2 PDMS between a one-trajectory and a 32-trajectory model.
- **See**: [[concepts/navhard-ood-evaluation.md]], [[concepts/selection-based-planning.md]]
- **Updated**: 2026-09-30

### Is the navhard Stage-2 lane-keeping collapse a planner failure or a rendering artifact?
- **Status**: open
- **Known so far**: Every method, including constant velocity, loses 35–45 points of lane keeping in Stage 2. If 3DGS renderings degrade as the ego pose leaves the recorded trajectory, part of the drop measures the benchmark. A candidate scorer recovers 10 points of Stage-2 LK on a fixed checkpoint, which fits either explanation. No paper has separated them.
- **See**: [[concepts/navhard-ood-evaluation.md]]
- **Updated**: 2026-09-30

### Can HUGSIM results be put on one table, and does anything move its Extreme tier?
- **Status**: open
- **Known so far**: HUGSIM numbers fall into two incompatible eras (345 and 436 scenarios) and the controller commit is unstated for [[sources/physwam.md]], [[sources/mm-future.md]] and Latent-WAM. Every ingested method scores 0.06–0.14 on the Extreme tier. NAVSIM-only models were thought to collapse after the Easy tier (PhysWAM 86.9 → 30.1, Latent-WAM 72.5 → 24.0, against WA-JEPA 79.8 → 55.6), but MM-Future, also NAVSIM-only, has the opposite shape (Easy 53.8, Medium 40.0). Pretraining data, camera count and objective are confounded.
- **See**: [[concepts/hugsim-benchmark.md]]
- **Updated**: 2026-09-30

### Will PhysicalAI-AV converge on one protocol, and where does data scaling stop helping?
- **Status**: open
- **Known so far**: Two papers use two protocols and report opposite orderings; DriveWAM's ordering against Alpamayo-1.5 inverts between subsets by a factor of about 3. The leakage-free subset costs 0.04–0.06 m uniformly. Both scaling studies on the benchmark stop while still improving, so the knee on real logs has not been found.
- **See**: [[concepts/physicalai-av-benchmark.md]]
- **Updated**: 2026-09-30

---

## 2. World models: what the predicted future is for

### Does world modeling buy open-loop accuracy or closed-loop robustness?
- **Status**: partially answered
- **Known so far**: Four effects on navtest / navhard point the same way: [[sources/geowam.md]] future geometry +0.6 / +4.9, [[sources/suv.md]] future access +0.3 / +4.1, [[sources/metis.md]] asymmetric mask +0.5 / +2.2 and video-prior scale ≈0 / +2.4. One mechanism breaks the pattern: [[sources/physwam.md]]'s geometric loss is worth +1.9 on both. Most world-model papers optimize and report navtest only, which may be the wrong benchmark for the question.
- **See**: [[concepts/world-model-for-ad.md]], [[concepts/navhard-ood-evaluation.md]]
- **Updated**: 2026-09-30

### Does generating the future at test time ever pay off?
- **Status**: partially answered
- **Known so far**: On navtest, no: SimWAM finds no benefit over isolation, and [[sources/foresight.md]] pays 870 ms for +0.3 PDMS. On navhard, one-way access is worth +4.1 EPDMS ([[sources/suv.md]]) while the bidirectional version is worst ([[sources/metis.md]]). Whether it matters for long-horizon rollout or reactive closed-loop interaction, which NAVSIM cannot measure, is open. WA-JEPA's ablations never separate its training objective from its inference-time scene denoising.
- **See**: [[concepts/world-model-for-ad.md]], [[concepts/wam-attention-masks.md]]
- **Updated**: 2026-09-30

### Who should attend to whom among observation, future and action tokens?
- **Status**: open
- **Known so far**: SimWAM, Metis and SUV share Wan2.2-5B and differ mainly in the mask. On navtest every non-bidirectional mask lands within 0.5 EPDMS; on navhard, future-reads-action is +2.2 and action-reads-future is +4.1, yet the bidirectional mask is worst. No single model has run the full set of cells. WA-JEPA blocks the gradient from action to future while Metis relies on it. Outside that backbone, [[sources/mm-future.md]] finds bidirectional best on navtest, but only with modality-specific branches (+0.4; −0.1 in a single DiT). Every number is a single run.
- **See**: [[concepts/wam-attention-masks.md]]
- **Updated**: 2026-09-30

### Are per-candidate futures really futures, and do they help at a realistic horizon?
- **Status**: open
- **Known so far**: Two per-candidate-future scorers agree in sign and size: +0.15 PDMS ([[sources/da-wam.md]], 0.5 s futures) and +0.4 for letting the scorer read the future ([[sources/mm-future.md]], 4 s futures). In both, only the expert-matched future is supervised (31 of 32 and 63 of 64 per scene are not). MM-Future shows action sensitivity (futures of extreme trajectories are 17% further apart in RMS), not correctness. [[sources/ad-e2e-jepa.md]] offers a hit-rate instrument (53.8% top-1 among 256) and shows 4 s rollouts cost about 3 ms per candidate, but neither DA-WAM nor MM-Future has been measured with it.
- **See**: [[concepts/world-model-for-ad.md]], [[concepts/selection-based-planning.md]]
- **Updated**: 2026-09-30

### Do the two world-model supervision rules compose?
- **Status**: open
- **Known so far**: Objective form follows target entropy: regression on multi-view scene latents is worse than no future prediction (90.7 vs 91.1 EPDMS, [[sources/wa-jepa.md]]). Target content follows horizon: optical flow is best far out and RGB one step ahead ([[sources/drive-hwm.md]]). Nobody has crossed them. Two results complicate the first rule: L1 regression on future latents helps in [[sources/redrive.md]] (+0.7), and [[sources/geowam.md]] pairs cosine alignment with dense point-map regression without collapsing. It is also unknown which other deterministic predictors (DeepSight, FLARE, Latent-WAM, AD-E2E-JEPA) are losing performance to the same effect.
- **See**: [[concepts/world-model-for-ad.md]]
- **Updated**: 2026-09-30

### Geometry, pixels or flow: which future representation helps a planner most?
- **Status**: open
- **Known so far**: [[sources/drivelaw.md]] holds the planner fixed and spans 84.1 → 89.1 PDMS across representation families, but geometry is not in its comparison. GeoWAM argues geometry beats pixels using other papers' numbers. [[sources/geoworldad.md]] prices the coordinate frame alone at +2.5 PDMS, while [[sources/drive-hwm.md]]'s linear probe puts an image-space flow latent first for future ego motion (83.7) and an ego-frame BEV latent last (71.2). No paper compares the three under one planner.
- **See**: [[concepts/world-model-for-ad.md]]
- **Updated**: 2026-09-30

### How much of a "world-model-shaped" representation is world modeling?
- **Status**: open
- **Known so far**: [[sources/redrive.md]]'s encoder keeps its gain under a frozen probe (86.7 → 90.2), but it was fine-tuned by both the planning loss and the future loss, and the paper's own split is +3.5 for unfreezing against +0.7 for future prediction. The same question applies to every training-time-only world model that reports a with/without-future-loss delta only end to end. Related: [[sources/physwam.md]] does not isolate whether its +1.9 EPDMS needs the depth–motion cross term or only a pose loss.
- **See**: [[concepts/world-model-for-ad.md]]
- **Updated**: 2026-09-30

### Does the generated video follow the generated action?
- **Status**: open
- **Known so far**: [[sources/physwam.md]] is the only joint model that checks: a 0.80° median yaw disagreement over 4 s (floor 0.29°) and 67% agreement where the plan departs from the log. DriveVA, DriveWAM, SUV and WA-JEPA make the same joint-generation claim without reporting it.
- **See**: [[concepts/world-model-for-ad.md]]
- **Updated**: 2026-09-30

### Can a world model replace the simulator as the source of reward or cost?
- **Status**: partially answered
- **Known so far**: [[sources/dreameraD.md]] shows a latent reward model can replace simulator calls during RL rollout (87.7 EPDMS) but still needs the simulator to annotate its vocabulary. [[sources/ad-e2e-jepa.md]] has cheap 4 s rollouts over 256–8,192 candidates but selects with the ground-truth future frame; that oracle-goal search reaches only 67.3–72.9 EPDMS against 85.4 for the same paper's imitation model. Nobody has attached a learned reward or goal predictor to a compressed rollout model and reported a real planning score.
- **See**: [[concepts/world-model-for-ad.md]]
- **Updated**: 2026-09-30

### Can a driving world model predict counterfactuals?
- **Status**: partially answered
- **Known so far**: For retrospective counterfactuals over recorded episodes, no: action-conditioned generation recovers 0.38 (Vista) and 0.31 (DrivingWorld), below the 0.5 no-preference point ([[sources/driving-wm-counterfactuals.md]]). Open parts: whether a model trained to condition on the factual continuation (abduction) would beat the paper's depth-plus-splatting transport, whether the failure is causal or a domain shift to CARLA renders, whether anything carries over to decision time, and whether any counterfactual metric predicts planning quality.
- **See**: [[concepts/counterfactual-prediction.md]], [[concepts/world-model-for-ad.md]]
- **Updated**: 2026-09-30

### Where and how should a planner read a video prior?
- **Status**: open
- **Known so far**: [[sources/adaptive-wam.md]] shows readout depth matters about 40 times more than noise level (4.80 vs 0.15 PDMS) and that a mid-network exit beats the final block, on one backbone family with one head. Three papers agree a single forward pass suffices and ForeSight disagrees (100 steps). DriveLaW's collapse at ten denoising iterations (89.1 → 23.2) is undiagnosed. No other paper reports which layer it reads.
- **See**: [[concepts/world-model-for-ad.md]]
- **Updated**: 2026-09-30

### Does the scale of the video prior matter?
- **Status**: partially answered
- **Known so far**: On navtest, barely: SimWAM holds the planner fixed and gets 90.2 from Wan2.1-1.3B against 90.3 from Wan2.2-5B, while a weak prior (LTX-Video, 88.7) costs and a driving-pretrained one (Cosmos-Predict2.5, 90.4) helps most. On navhard, [[sources/metis.md]] finds the same 1.3B-versus-5B swap worth 2.4 EPDMS. Scale may matter exactly where navtest cannot see it. Zero-shot transfer, DriveVA's distinguishing claim, is untested.
- **See**: [[concepts/world-model-for-ad.md]], [[concepts/foundation-backbones-for-ad.md]]
- **Updated**: 2026-09-30

### Does generation quality matter for planning?
- **Status**: open
- **Known so far**: [[sources/reworld.md]] improves nuScenes FVD 81.3 → 61.9 and NAVSIM PDMS 89.1 → 90.4 in one paper through disjoint mechanisms, and never measures whether the video-side objective helps planning. FVD itself is hard to rank on: [[sources/physwam.md]] measures a recorded-versus-recorded FVD floor of 91.5 at 600 clips.
- **See**: [[concepts/world-model-for-ad.md]], [[concepts/navsim-benchmark.md]]
- **Updated**: 2026-09-30

### Does a clean VLM stream suppress an iteratively refined stream in shared attention?
- **Status**: open
- **Known so far**: [[sources/brainwam.md]] measures it for VLM plus video denoising (Tri-MoT 87.8 below WAM-only 88.1); SimWAM's two-modality ablation shows no such effect without a VLM. Untested for occupancy diffusion, flow-matched latents or JEPA predictors, and untested whether one-way masking (UniDriveVLA, AutoMoT) avoids it.
- **See**: [[concepts/world-model-for-ad.md]], [[concepts/mixture-of-experts.md]]
- **Updated**: 2026-09-30

---

## 3. Perception and representation

### Is explicit perception supervision necessary for planning?
- **Status**: open
- **Known so far**: Both routes work. [[sources/auto-jepa.md]] reaches 91.3 PDMS with a frozen encoder and no perception labels; [[sources/wcog-vla.md]] reaches 92.9 with 3D boxes and per-agent futures. Removing WCog-VLA's 3D perception costs 3.3 PDMS (89.3 → 86.0), yet its 3D pretraining is worth only +1.1 while the planner still emits text, which suggests explicit structure needs a continuous action head to be used. [[sources/foresight.md]] loses 1.1 PDMS (89.3 → 88.2) when its whole perception branch is deleted. No paper compares the supervised and unsupervised routes at matched capacity and data.
- **See**: [[concepts/perception-for-planning.md]]
- **Updated**: 2026-09-30

### Does agent selectivity predict driving quality?
- **Status**: open
- **Known so far**: [[sources/auto-jepa.md]] measures 2.97× occlusion selectivity in latent space (mean 1−cos 0.080 vs 0.027) and shows behavioural consequences on three hand-picked scenes. The occlusion protocol has been run on one model. No paper correlates an occlusion-sensitivity statistic with PDMS, collision rate or interaction-scenario performance, and Auto-JEPA has no ablation against plain waypoint regression through the same trajectory encoder.
- **See**: [[concepts/perception-for-planning.md]], [[concepts/world-model-for-ad.md]]
- **Updated**: 2026-09-30

### Do teacher pseudo-labels cap what the student can learn?
- **Status**: partially answered
- **Known so far**: Swapping a teacher's depth for LiDAR depth under one planner has not been run. [[sources/physwam.md]] does half of it: generating depth alone adds nothing to planning (87.2 vs 87.5 PDMS without it), while a raw-LiDAR point loss that also involves the generated motion improves depth by 8–15% AbsRel and planning by +1.9 EPDMS. Whether teacher quality (for example Depth Anything V2 vs V3, SAM 2 vs SAM 3) propagates to planning is untested, and agent-generated reasoning corpora rarely carry a measured noise rate (Qwen-Drive reports 55.9% for its own).
- **See**: [[concepts/teacher-pseudo-labels.md]], [[concepts/world-model-for-ad.md]]
- **Updated**: 2026-09-30

### Is it video pretraining or the JEPA objective that transfers to planning?
- **Status**: open
- **Known so far**: V-JEPA 2 beats image-pretrained encoders in every sweep that includes it ([[sources/redrive.md]]: +5.5 PDMS over DINOv2, and fine-tuning DINOv2 under a predictive loss does not close the gap, 84.4 vs 89.9). But V-JEPA 2 is the only video-pretrained entry in those tables, and no paper includes a video-pretrained non-JEPA control. Video-generation priors (SimWAM, DriveLaW) are also effective without a JEPA objective. [[sources/ad-e2e-jepa.md]] inherits the opposite default (DINOv3 over V-JEPA 2) from a robotics ablation the wiki can cite only second-hand.
- **See**: [[concepts/foundation-backbones-for-ad.md]]
- **Updated**: 2026-09-30

### When should a foundation backbone be frozen, LoRA-adapted or fully fine-tuned?
- **Status**: open
- **Known so far**: Off-the-shelf use is priced three times: a BEV head on frozen Qwen-initialized features trails a dedicated detector by 6.34 mAP and unfreezing recovers +10.46; a frozen video DiT costs 6.42 PDMS; LoRA collapses under DINOv2 geometric distillation. [[sources/redrive.md]] prices unfreezing its encoder at +3.5 PDMS. The page's rule of thumb (LoRA when the pretrained objective already matches the downstream one, full fine-tuning otherwise) is fitted to two papers, neither of which tests the other's setting.
- **See**: [[concepts/foundation-backbones-for-ad.md]]
- **Updated**: 2026-09-30

### Does discrete visual tokenization survive at the frontier?
- **Status**: open
- **Known so far**: The best token-based planner scores 91.8 PDMS against 93.7 for continuous-latent methods; whether the gap is quantization loss or tokenizer immaturity is untested. [[sources/unified-driving-tokens.md]] argues one tokenizer can serve planning and world modeling and then uses different tokenizers for each; its geometry-enhanced variant is worse on every reconstruction metric. Its tokenizer needs a DINOv3-B forward pass and reports no latency.
- **See**: [[concepts/visual-tokenization.md]]
- **Updated**: 2026-09-30

---

## 4. Reasoning and the VLM

### Does grounded reasoning help on a public benchmark, and is the trace faithful?
- **Status**: open
- **Known so far**: [[sources/grava.md]] gives the most graded ablation: action-only 74.8 → coarse reasoning 77.9 → box-grounded objects 85.5 → full grounded reasoning 90.1, all RL-trained, on an internal long-tail benchmark. Without RL, full reasoning scores 73.1. Editing the trace edits the action (0.44 → 0.82 normalized PDMS), but the trace's final maneuver is copied from the expert trajectory at annotation time. Unknown: whether the ladder holds on a public benchmark, whether any trace stays predictive of the action on navhard, and whether a verbalized action is enough for auditing.
- **See**: [[concepts/chain-of-thought-for-ad.md]], [[concepts/reasoning-faithfulness.md]]
- **Updated**: 2026-09-30

### What does driving adaptation cost the VLM, and is retained general capability worth anything for driving?
- **Status**: open
- **Known so far**: Under one 15-benchmark protocol, Qwen-Drive scores 66.41 against its base model's 67.40 while Alpamayo-1.5 scores 14.65, MiMo-Embodied 26.53 and UniDriveVLA 48.73 ([[sources/qwen-drive-1.0.md]]). The distinguishing variable looks like the data mixture (26% general-purpose); nothing between 0% and 26% has been run. Qwen-Drive freezes the VLM during RL, so retention after GRPO through the VLM is unmeasured for every method that does it. The only evidence on usefulness points the other way: the whole knowledge-adaptation stage is worth +0.08 RFS for planning.
- **See**: [[concepts/general-capability-retention.md]]
- **Updated**: 2026-09-30

### When should slow reasoning run, and does it need to emit a decision?
- **Status**: open
- **Known so far**: [[sources/drive-hwm.md]] runs its slow branch every 8 steps unconditionally (25.6 ms) and has it emit a forecast, not a plan; CLEAR, AdaThinkDrive and AutoVLA route slow compute by difficulty. The two policies have never been compared at matched average cost, and no paper compares a forecast-only slow branch against a meta-action one under a fixed fast policy. Drive-HWM's rate hierarchy is worth +0.3 to +0.8 PDMS on NAVSIM, where the slow model runs once per scenario, so the design's purpose is untested in a closed loop. Adaptive CoT's robustness when its complexity classifier fails on OOD scenes is also unmeasured.
- **See**: [[concepts/dual-system-vla.md]], [[concepts/adaptive-routing.md]], [[concepts/chain-of-thought-for-ad.md]]
- **Updated**: 2026-09-30

### Advisory VLM or VLM in the action path?
- **Status**: open
- **Known so far**: [[sources/drivewam.md]]'s frozen Qwen3-VL-8B only emits text guidance and helps at every data scale, but the paper reports no VLM–action agreement rate, and Senna-2's evidence is that unaligned guidance under-delivers. No paper compares the advisory arrangement against fine-tuning the VLM into the action path at matched backbone and data, or distils the VLM into the fast policy after alignment.
- **See**: [[concepts/dual-system-vla.md]], [[concepts/world-model-for-ad.md]]
- **Updated**: 2026-09-30

---

## 5. RL, selection and diversity

### Do learned scorers overfit the benchmark's own metric?
- **Status**: open
- **Known so far**: Almost every non-BoN NAVSIM-v1 entry above 92.0 PDMS trains a scorer or an RL reward against the benchmark metric. [[sources/mm-future.md]] shows what that looks like on v2: ego progress 92.2 (human agent 87.4) with driving direction, lane keeping and history comfort the lowest in its table, because its scorer learned the v1 components. Whether PDMS-supervised routing and scoring stay reliable in interactive, reactive settings is untested.
- **See**: [[concepts/selection-based-planning.md]], [[concepts/adaptive-routing.md]]
- **Updated**: 2026-09-30

### Does RL buy out-of-distribution robustness?
- **Status**: open
- **Known so far**: Within [[sources/geowam.md]]'s navhard table, which contains no candidate-scoring method, three of the four methods above 31 use RL or PDMS-score supervision, but GeoWAM (36.6) tops them without it. Margins are 0.5–2.5 points on single runs, so the wiki cannot say. Whether GRPO-trained NAVSIM methods (FLARE, DriveFine) are competitive on Bench2Drive's interactive scenarios is also unreported.
- **See**: [[concepts/navhard-ood-evaluation.md]], [[concepts/bench2drive.md]]
- **Updated**: 2026-09-30

### Does an inference-time safety check add anything to a policy already trained on the rule?
- **Status**: partially answered
- **Known so far**: The nearest measurement says no. [[sources/drivereferee.md]] applies one geometric rule at inference: +0.30 EPDMS on the untrained base, and +0.06 PDMS / −0.04 EPDMS on a policy already distilled on it. Whether reflective inference stacks with a GRPO-trained base in other architectures (masked diffusion, discrete flow matching) is still open.
- **See**: [[concepts/inference-time-safety.md]]
- **Updated**: 2026-09-30

### How should diversity be rewarded and measured when the output is text plus a trajectory?
- **Status**: open
- **Known so far**: Best-of-6 on NAVSIM-v1 matches the human trajectory's score (Curious-VLA, 94.8 PDMS). Untested: MUPO-style multi-group advantages with groups clustered by trajectory geometry, separate diversity rewards for reasoning and action, and whether learned scorers benefit from diversity-trained candidates.
- **See**: [[concepts/divergent-thinking-in-vlms.md]], [[concepts/best-of-n.md]]
- **Updated**: 2026-09-30

### Which RL recipe suits dense, safety-gated driving rewards?
- **Status**: open
- **Known so far**: Whether Dr. GRPO remains preferable when rewards are dense, continuous and safety-gated rather than binary is untested. It is also unknown at what data scale reasoning-free Dr. GRPO training (NoRD) catches up with CoT-supervised training on 212K+ samples (AutoVLA), and whether a single sequence-level objective can serve both expert types in a mixture-of-experts planner.
- **See**: [[concepts/r1-zero-like-training.md]], [[concepts/chain-of-thought-for-ad.md]], [[concepts/mixture-of-experts.md]]
- **Updated**: 2026-09-30

---

## 6. Efficiency

### What does each method actually cost at inference?
- **Status**: open
- **Known so far**: Recorded latencies span 22 ms to 1.36 s on different hardware, and only the Wan2.2-5B family is close to comparable (Metis and SUV both use an RTX 4090). Missing figures matter most where the cost is likely highest: GRAVA's grounded reasoning (8B, 4,096-token limit) reports none, SUV reports only its with-access cost, and [[sources/physwam.md]] pays 9.4 GPU-seconds per plan with no action-only path tested. Per-candidate futures are cheap once the state is a few hundred tokens (about 3.2 ms per hypothesis in [[sources/mm-future.md]]).
- **See**: [[concepts/inference-latency.md]], [[concepts/wam-attention-masks.md]]
- **Updated**: 2026-09-30

---

## Answered

Questions closed by an ingested paper, each with the paper and its result.

- **Is the comfort deficit inherent to sampled planners?** No. [[sources/physwam.md]] samples video, depth and motion from noise and scores 96.3 closed-loop comfort and 90.5 navtest EC. What it does differently from WA-JEPA, which scores far lower on closed-loop comfort, is not isolated. See [[concepts/hugsim-benchmark.md]]. (2026-09-30)
- **Does performance on curated rare-event clips predict closed-loop behaviour?** No. The AlpaSim reproduction in [[sources/qwen-drive-1.0.md]] inverts the ordering. The remaining open part is tracked under "Do NAVSIM rankings survive a reactive closed loop?". See [[concepts/physicalai-av-benchmark.md]]. (2026-09-30)
