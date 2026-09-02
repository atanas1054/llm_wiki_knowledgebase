---
title: Perception-Enhanced Planning in VLA Models
type: concept
sources: [raw/papers/GeoWAM_ Visual Geometry World Action Models for Autonomous Driving.md, raw/papers/Auto-JEPA_ A Latent World Model of Continuous Intent for End-to-End Autonomous Driving.md, raw/papers/Percept-WAM_ Perception-Enhanced World-Awareness-Action Model for Robust End-to-End Autonomous Driving.md, raw/papers/UniDriveVLA_ Unifying Understanding, Perception, and Action Planning for Autonomous Driving.md, raw/papers/OneDrive_ Unified Multi-Paradigm Driving with Vision-Language-Action Models.md, raw/papers/Latent-WAM_ Latent World Action Modeling for End-to-End Autonomous Driving.md, raw/papers/SGDrive_ Scene-to-Goal Hierarchical World Cognition for Autonomous Driving.md]
related: [sources/geowam.md, sources/auto-jepa.md, sources/percept-wam.md, sources/unidrivevla.md, sources/onedrive.md, sources/latent-wam.md, sources/sgdrive.md, concepts/diffusion-planner.md, concepts/vlm-domain-adaptation.md, concepts/navsim-benchmark.md, concepts/world-model-for-ad.md, concepts/dual-system-vla.md, concepts/intent-conditioned-planning.md]
created: 2026-04-05
updated: 2026-09-02
confidence: high
---

## The Core Tension

End-to-end VLA planners face a structural choice:

**Option A — QA-style spatial reasoning** (EMMA, DriveVLM): spatial understanding encoded as question-answer supervision. Indirect localization signals; no persistent world state; duplicate detections with poorly calibrated confidence in crowded scenes.

**Option B — Encoder-decoder pipelines** (DiffusionDrive, WAM-Flow): skip VLM entirely for spatial tasks; use specialized BEV encoders + diffusion/flow decoders. Sacrifice reasoning capacity for geometric precision. Brittle in long-tail scenarios.

**Option C — Embedded world states** (Percept-WAM, UniAD philosophy in VLM): encode persistent, metrically grounded world states as specialized tokens inside the VLM backbone. Perception and planning jointly optimized with shared representations.

## Why Perception Helps Planning

The motivation is borrowed from UniAD (planning-oriented multi-task learning): explicit perception supervision forces the model to maintain accurate, spatially precise representations of scene geometry, which in turn grounds trajectory generation.

Two mechanisms:

**1. Shared representation grounding**: World-PV and World-BEV tokens must simultaneously support detection/segmentation and trajectory decoding. This forces the backbone to maintain metric accuracy that pure trajectory-from-images training does not require.

**2. Modality-aligned trajectory decoding**: when the trajectory decoder explicitly attends to World-BEV tokens (3D spatial context), World-PV tokens (semantic context), and ego-state (kinematic context) via separate attention heads, it avoids over-reliance on any single representation.

Evidence from Percept-WAM: joint PV 2D+3D training improves 2D detection by +3.2 mAP (synergy between tasks). The four-query trajectory decoder (Q_ego, Q_pv, Q_bev, Q_full) achieves best L2 accuracy and best inference speed among all decoding variants.

## World Tokens: Architecture Pattern

### World-PV Tokens (Perspective-View)

- Source: VLM ViT image features, patchified into H×W spatial grid
- Each grid cell = **localized query** tied to an image-plane coordinate
- Prediction: one bounding box per grid cell (2D or mono-3D), via parallel AR decoding
- Grid tokens across cells are **mutually masked** → independent parallel prediction, not sequential

**Format** (2D): `cls, <box> x,y,w,h </box>, <conf> s </conf>`  
**Format** (3D): `cls, <box> x,y,z,w,h,ℓ,θ,vx,vy </box>, <conf> s </conf>`

### World-BEV Tokens (Bird's-Eye-View)

- Learnable query tokens forming H×W BEV grid centered on ego vehicle
- Attend to World-PV tokens via cross-attention → implicit PV→BEV view lifting (no explicit depth supervision)
- Optional initialization from LiDAR (PointPillars → PixelUnshuffle → MLP) for metrically grounded 3D priors
- BEV grid resolution: 40×40 for detection, 10×10 for segmentation

**Camera-only BEV** is significantly weaker than LiDAR-initialized (25.0 vs. 58.9 mAP on nuScenes). The view-lifting problem remains an open challenge.

### Token Reuse for Action

The same World-PV and World-BEV tokens produced during the perception prefill stage are directly reused by the trajectory decoder — no additional forward pass required. This is computationally free: perception and planning share the prefill computation.

## IoU-Aware Confidence: Calibration for Dense Scenes

A critical practical contribution. Standard VLM confidence (softmax class logits) is systematically overconfident, producing many false positives in crowded scenes. Percept-WAM addresses this with a dedicated IoU prediction token per box.

**Key insight on training data**: the distribution of the confidence-tuning dataset matters enormously:
- Random-perturb of GT → near-uniform IoU distribution → hurts performance (−1.2 AP)
- Uniform sampling of model predictions → still misaligned → hurts (−1.9 AP)
- **Real model-prediction distribution (skewed toward low IoU)** → aligned with actual inference → +1.5 AP, +2.3 AP₇₅

This finding generalizes: calibration training should match the realistic distribution of model errors, not a synthetic distribution designed for coverage.

**Final confidence** = class_conf × predicted_IoU — provides a unified, localization-sensitive reliability measure suitable for NMS post-processing.

## Grid-Conditioned Parallel AR Decoding

Standard sequence-based detectors (Pix2Seq, LLM-based detectors) generate all boxes in a left-to-right sequence — attention from later boxes to earlier boxes creates implicit position biases and sequential coupling.

Percept-WAM's grid-conditioned approach:
1. Interpolate World-PV/BEV tokens at each grid position → localized grid token
2. Each grid token independently predicts one object at that location
3. Grid tokens from different locations are **mutually masked** → fully parallel generation
4. 16× inference speedup over sequential AR for BEV detection (Table 6) with no accuracy loss

This is structurally similar to WAM-Flow's parallel DFM decoding, but applied to object detection rather than trajectory generation.

## Sparse Query-Based Perception: UniDriveVLA

**UniDriveVLA** ([[sources/unidrivevla.md]]) introduces a different paradigm: instead of dense BEV tokens inserted into a shared-weight VLM (as in OmniDrive, OpenDriveVLA, Percept-WAM), it uses **sparse query-based perception** inside a dedicated MoT Perception expert.

### The Perception–Reasoning Conflict

UniDriveVLA provides the strongest empirical evidence yet for why naive perception injection hurts VLMs. In a shared-weight decoder, cosine similarity between LLM tokens and perception tokens progressively increases toward 1 across layers — indicating **feature collapse** where spatial and semantic representations become indistinguishable. The consequence is that improving spatial perception directly degrades semantic reasoning and vice versa.

MoT solves this by decoupling parameters: the Understanding expert (und) never sees perception tokens, so its representations cannot collapse. The Perception expert (per) attends to und tokens but not vice versa — a one-way semantic enrichment channel. Quantitative evidence:

| Architecture | General VQA↑ | DriveBench↑ | L2(m)↓ | CR(%)↓ |
|---|---|---|---|---|
| Shared-Weight | 31.1% | 50.8% | 0.641 | 0.175 |
| MoT | 45.5% | 54.9% | 0.533 | 0.140 |
| Δ | **+14.4pp** | **+4.1pp** | **−0.108m** | **−0.035** |

### Sparse Perception Design

- **Task-specific queries**: K-Means instance bank initialization (dataset-level clustering) for 3D detection, HD map, ego-status, motion forecasting, and occupancy (5 tasks in one decoder)
- **Two-pass enrichment**: first decoder pass → projection into VLM hidden space → Masked Joint Attention (per attends und) → project back → second refinement pass. Perception is not a one-shot extractor; it benefits from the VLM's semantic understanding
- **No dense BEV**: no PointPillars, no BEV grid construction, no view-lifting from multi-camera — spatial geometry is extracted directly from multi-scale 2D visual features via deformable attention

### Perception vs. Planning Gains (Table 5 ablation)

| Added component | ΔL2 | ΔCR |
|----------------|-----|-----|
| Ego-state | −0.14m | −0.06 |
| Detection | −0.03m | **−0.11** |
| Map | 0.00m | +0.04 |
| Occupancy | **−0.05m** | 0.00 |
| Motion | +0.01m | +0.03 |

Detection is most critical for safety (CR halved: 0.21→0.10). Occupancy gives the best trajectory accuracy (0.58→0.53). Map and motion add no consistent gains in the current nuScenes open-loop regime — possibly because map supervision overlaps with what detection already provides, and motion prediction is still far behind specialized models.

## Single-Decoder Structured Queries: OneDrive

**OneDrive** ([[sources/onedrive.md]]) introduces a third perception-integration path. It does not create dense World-PV/BEV tokens like Percept-WAM and does not use decoupled MoT perception experts like UniDriveVLA. Instead, it puts detection queries, lane queries, planning queries, image tokens, and optional text tokens into one causal VLM decoder.

The planning order matters: detection tokens before lane tokens before planning tokens gives the best nuScenes result (0.28 L2 / 0.18 collision), while lane-before-detection is worse. This supports the same high-level claim as UniAD/Percept-WAM: perception supervision helps planning, but OneDrive shows it can be mediated by causal attention rather than a separate BEV or expert stream.

Key tradeoff: OneDrive preserves text generation and uses one attention backbone, but it still needs query-only self-attention and task-specific FFNs because raw autoregressive causal attention is not sufficient for parallel structured prediction.

## Geometric Distillation Without Perception Heads: Latent-WAM

**Latent-WAM** ([[sources/latent-wam.md]]) adds a training-time spatial-supervision path that is not explicit detection, BEV segmentation, or occupancy prediction. It distills patch-level features from a frozen geometric foundation model, WorldMirror, into a DINOv2-Base encoder, then compresses the result into scene tokens for latent world modeling.

This is a useful middle ground:

| Approach | Spatial signal | Runtime cost |
| --- | --- | --- |
| Percept-WAM | Explicit PV/BEV perception tokens and heads | Perception tokens run at inference |
| UniDriveVLA | Sparse perception expert with detection/map/occupancy tasks | Perception expert runs at inference |
| OneDrive | Detection/lane/planning queries in one causal decoder | Structured queries run at inference |
| **Latent-WAM** | **WorldMirror feature distillation into DINOv2 patches** | **Teacher removed at inference** |

The ablation is sharp: no geometric feature gives 88.3 EPDMS, direct feature concatenation gives 88.0, and distillation gives 89.3. The lesson is that spatial foundation features help only when aligned into the trainable planning representation, not when appended as frozen key-value inputs.

## Ego-Relevance Filtering: SGDrive

[[sources/sgdrive.md]] adds a selection criterion the other entries here do not have. Rather than detecting every visible object, it restricts detection targets to **safety-critical agents** — vehicles, pedestrians, and cyclists chosen by proximity to the ego trajectory and visibility in the front-camera frustum. The stated rationale is capacity allocation: forcing a finite set of queries onto the agents that can actually influence the ego decision, "rather than exhaustively perceiving all objects in the scene."

This is a different lever from the sparsity in [[sources/unidrivevla.md]] and [[sources/percept-wam.md]]. Those reduce *representational* cost (fewer queries, sparser tokens) while still aiming at the full scene; SGDrive reduces the *task* itself, changing what counts as a positive detection. The supervision is otherwise conventional — DETR bipartite matching with $\lambda_{\text{cls}}=10$ and $L_1$ regression — applied at both the current time and a future step.

Two further distinctions. Agents are predicted at $t$ **and** $t{+}n$, so the detection head doubles as a motion-forecasting head. And nothing is decoded at inference: the agent subquery's hidden states pass straight to the DiT planner, so the detection head exists purely to shape the representation.

The ablation isolates its contribution cleanly. Adding the agent subquery on top of scene geometry moves PDMS 86.0 → 86.3, with the gain concentrated in NC and DAC — exactly the collision-and-compliance metrics an ego-relevance filter should affect. That is a small absolute number, but the functional signature is the right one, and it comes on top of an already-strong geometric representation.

**Cost**: 3D box annotations at training time. Along with occupancy labels for the scene head, this makes SGDrive markedly more annotation-hungry than the perception-free world models ([[sources/latent-wam.md]], Drive-JEPA) tracked on this page.

## Emergent Ego-Relevance Without Perception: Auto-JEPA

SGDrive above buys agent relevance by *supervising* it. [[sources/auto-jepa.md]] gets the same behavior with no perception supervision of any kind, and — more usefully for this page — supplies a protocol for **measuring** it.

The model never sees a box, an agent identity, an interaction label, or surrounding-agent motion. Its only target is the latent encoding of the future ego trajectory. Yet the predicted latent responds selectively to traffic participants, and in the per-vehicle case to *the right* traffic participants: occluding an interacting lead vehicle shifts both the intent and the selected trajectory, while occluding a non-interacting adjacent vehicle leaves both essentially unchanged.

The mechanism is a claim about supervision sufficiency: **if the prediction target depends on the agents, the model must attend to the agents, whether or not you name them.** This is the strongest counterweight in the wiki to the assumption running through the rest of this page — that planning-relevant spatial understanding has to be *installed* via detection, occupancy, or BEV heads.

### The Semantic Occlusion Protocol

This is worth adopting independently of Auto-JEPA, because it addresses something attention maps and CKA plots do not: whether a planner's representation is *causally* dependent on the objects we think matter.

```
For each validation scene:
  1. Build a dynamic-agent mask from projected regions of visible traffic participants
  2. Apply it identically to all four input frames
  3. Build a control mask of equal total image area from randomly sampled regions
  4. Hold ego-motion history and navigation command fixed in both arms
  5. Measure Δ_intent = 1 − cos(Ẑ, Ẑ_m) over the flattened latent
```

| Intervention | Mean $\Delta_\mathrm{intent}$ ($n=15{,}364$) |
|---|---:|
| Dynamic-agent masking | 0.080 |
| Equal-area random masking | 0.027 |
| **Ratio** | **2.97×** |
| Larger on dynamic-agent arm | **71.1% of scenes** |

Three design choices do the work. It is **paired** — the same scene under both interventions, so scene difficulty cancels. It is **area-matched**, removing the standard confound where the salient region is simply the large one. And it **holds the non-visual inputs fixed**, so the measured change is attributable to visual evidence rather than to ego history or command.

### What It Does Not Yet Control

Two gaps keep this from being a finished methodology, and both are cheap to close:

- **Shape, contiguity, and placement are uncontrolled.** Agent masks are object-shaped, road-level, and clustered near the vanishing point; the random control is described only as "independently sampled equal-area random regions," which can land on sky or periphery. Part of the 2.97× is plausibly a *road-region* effect rather than an *agent-semantics* effect. The missing arm is equal-area masks placed on the drivable surface or on static road furniture.
- **Latent change is not shown to be behavioral change.** The dataset-level statistic is measured in embedding space; the link to a different *selected trajectory* is shown on three hand-picked scenes. No correlation between $\Delta_\mathrm{intent}$ and PDMS, collision rate, or interaction-scenario performance is reported.

Also worth keeping in proportion: $1-\cos = 0.080$ means cosine similarity 0.92. Removing **every visible dynamic agent from all four frames** still leaves the intent 92% aligned with the unoccluded one. The *ratio* is the finding; the absolute dependence on agents is modest.

### How This Compares to the Page's Other Evidence

| Evidence type | Method | What it shows | Limitation |
|---|---|---|---|
| Cosine collapse across layers | [[sources/unidrivevla.md]] | Perception and semantic tokens become indistinguishable in a shared decoder | Diagnostic of interference, not of relevance |
| Detection-head ablation | [[sources/sgdrive.md]] | Ego-relevant boxes lift NC/DAC by +0.3 PDMS | Confounded with added capacity and supervision |
| Attention visualization | UniUGP and others | Attention moves to distant causal objects with a world model | Qualitative; no area or placement control |
| **Paired semantic occlusion** | **Auto-JEPA** | **Intent depends 2.97× more on agent regions than on matched random regions** | **Shape/placement uncontrolled; latent-space only** |

The interventional design is the advance. Everything above it in that table either observes the model's internals or compares two trained models; occlusion perturbs the input and measures the response, which is the only one of the four that supports a dependence claim. Whether the dependence is on *agents* specifically or on *road-region content* is the question the missing control arm would settle.

## Metric Geometry Without Annotation: GeoWAM

This page's running cost question is what spatial understanding *costs in labels*. [[sources/sgdrive.md]] buys ego-relevant 3D structure with occupancy and box annotation; [[sources/latent-wam.md]] avoids labels by distilling a frozen geometry foundation model; [[sources/auto-jepa.md]] gets agent selectivity from an ego-motion target with no spatial supervision at all. [[sources/geowam.md]] adds a fourth position: **dense metric 3D structure, supervised, but with pseudo-labels rather than human annotation**.

Its targets are dense point maps — one 3D point per pixel in the ego frame — derived from off-the-shelf geometry foundation models. Training needs only RGB. The paper contrasts this explicitly with occupancy world models (OccWorld, Drive-OccWorld), which need voxelized ground truth to construct their prediction space.

| Approach | Spatial supervision | Label cost | Available at inference? |
|---|---|---|---|
| [[sources/sgdrive.md]] | Occupancy voxels + 3D boxes + goal pose | **Occupancy + box annotation** | No — hidden states condition the DiT |
| [[sources/latent-wam.md]] | WorldMirror/VGGT feature distillation | None (frozen teacher, discarded) | No — teacher removed |
| [[sources/auto-jepa.md]] | None | None | No spatial output exists |
| **[[sources/geowam.md]]** | **Dense metric point maps + surface normals** | **None (geometry-model pseudo-labels)** | **Yes — point maps are decoded** |

Two things follow. GeoWAM is the only entry here that **produces inspectable 3D output at inference** without having paid for 3D annotation — Figure 3 of the paper shows forecast trees, poles, and road markings, and a following vehicle tracked through a left turn. And it is supervising *future* geometry, so the same head doubles as a motion-forecasting mechanism, much as SGDrive's agent head predicts boxes at $t$ and $t{+}n$.

**The cost is a dependency the paper does not discuss.** Pseudo-labels bound the supervision ceiling at whatever the geometry foundation model gets right, and GeoWAM's encoder is *initialized from DVGT-2* while its targets come from that same family of models. "Requires only RGB" is presented as a pure advantage; part of what is being learned is another model's biases. Compare Latent-WAM, which has the same dependency but discards the teacher at inference, and whose ablation showed the distillation target demanded full backbone updates — LoRA collapsed it from 89.3 to 68.5 EPDMS.

**On the page's central tension**, GeoWAM sits with Latent-WAM on the side that says explicit perception heads are not required: there is no detection, no segmentation, no occupancy classification. What it shares with the perception camp is the belief that *metric spatial structure* must be represented explicitly rather than left implicit in features — it just gets that structure from geometry rather than from semantics.

## Comparison: Perception Integration Approaches in AD VLMs

| Approach | Spatial supervision | World state type | Shared params? | Planning benefit |
|---|---|---|---|---|
| QA-style (EMMA, DriveVLM) | Indirect (language) | None — ephemeral text | ✓ | Indirect reasoning only |
| Multi-task E2E (UniAD) | Direct (detection heads) | BEV occupancy, motion | ✓ | Strong (planning-oriented) |
| **World tokens (Percept-WAM)** | **Direct (token-level)** | **World-PV + World-BEV tokens** | **✓** | **Shared representation + modality-aligned decoding** |
| Dense spatial injection (OmniDrive, OpenDriveVLA) | Direct (3D Q-Former / BEV) | BEV features | ✓ | Improved spatial precision, impaired reasoning |
| **Sparse MoT (UniDriveVLA)** | **Direct (sparse queries)** | **K-Means instance banks (5 tasks)** | **✗ (decoupled)** | **Spatial precision + preserved reasoning** |
| **Single causal decoder (OneDrive)** | **Direct (det/lane/planning queries)** | **Structured query tokens in VLM sequence** | **Yes, shared attention with task FFNs** | **Unified text/perception/planning; 0.28 L2 / 0.18 collision on nuScenes** |
| Latent-WAM | Geometry foundation distillation | Compact scene/world-status tokens | No VLM | Spatial grounding without inference-time perception heads |
| **Auto-JEPA** | **None — ego-trajectory latent target only** | **No scene state is represented at all** | No VLM; frozen V-JEPA 2 | **Agent selectivity emerges (2.97× occlusion ratio); no spatial output available** |
| **GeoWAM** | **Dense metric point maps (pseudo-labelled)** | **Future 3D point clouds in the ego frame** | No VLM; DVGT-2 encoder | **Explicit metric structure with no human annotation; decoded and inspectable at inference** |
| World model (UniUGP) | Video generation | Future frame prediction | ✓ | Causal feature grounding |

UniAD's planning-oriented multi-task learning is the closest philosophical predecessor to both Percept-WAM and UniDriveVLA. The key architectural divergence: Percept-WAM uses shared-weight tokens with a four-query decoder; UniDriveVLA uses decoupled MoT experts with sparse queries. Both share the insight that explicit spatial supervision improves planning, but they resolve the perception–reasoning conflict differently.

## Four-Query Trajectory Decoder

Rather than a single action head, Percept-WAM uses four parallel MLP decoders with modality-specific attention masking:

| Query | Attends to | Purpose |
|---|---|---|
| Q_ego | Ego-state | Kinematic grounding |
| Q_pv | World-PV tokens | Semantic/appearance context |
| Q_bev | World-BEV tokens | 3D geometric context |
| Q_full | All tokens | Final trajectory (inference output) |

All four trained simultaneously with Smooth-L1 loss. The separate decoders prevent the model from ignoring any modality — each must independently learn to produce reasonable trajectories from its limited view.

This contrasts with Reasoning-VLA's single set of learnable action queries (which attend to all VLM hidden states via cross-attention without modality partitioning).

## Limitations and Open Questions

1. **LiDAR dependence**: camera-only BEV is 25.0 mAP vs. 58.9 with LiDAR init — a 2.4× gap. The view-lifting problem from monocular cameras to BEV remains hard without depth supervision.

2. **No RL**: all methods in this wiki that approach or exceed 90 PDMS use GRPO. Percept-WAM uses purely SFT — RL is explicitly listed as future work in the paper's conclusion.

3. **Comfort issue on base model**: the plain query-based decoder produces Comf=92.8 vs. 99-100 for all other methods. This suggests the trajectory output has uncomfortable acceleration profiles that require post-processing (trajectory scoring) to fix. The root cause is not analyzed in the paper.

4. **Grid resolution trade-offs**: higher BEV grid resolution gives +9.1% mAP but the relationship between grid granularity and planning accuracy is not ablated.

5. **Task interference**: the model trains on 7+ tasks simultaneously. While joint training generally helps, negative transfer is possible — PV semantic segmentation training could harm trajectory prediction quality. No systematic analysis provided.

6. **Is explicit perception supervision necessary at all?** Everything on this page above the Auto-JEPA section assumes planning-relevant spatial understanding must be installed via detection, occupancy, or BEV heads. [[sources/auto-jepa.md]] reaches 91.3 PDMS with a frozen encoder, no perception labels, and demonstrable agent selectivity — but also with no spatial output, no interpretability, and no reusable scene representation. The honest reading is that supervision buys *inspectable* perception, not necessarily *better* planning, and no paper has compared the two routes at matched capacity and data.

7. **Does the occlusion protocol generalize?** It has been run on exactly one model. Applying it to SGDrive (supervised relevance), Latent-WAM (geometric distillation), and a plain imitation baseline would turn a single-paper observation into a comparable diagnostic — and would show whether explicit agent supervision produces *more* selectivity than an ego-motion target, or merely more legible selectivity.
