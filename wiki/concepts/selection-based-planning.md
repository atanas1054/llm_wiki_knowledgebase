---
title: Selection-Based Trajectory Planning
type: concept
sources: [raw/papers/DA-WAM_ Decision-Aligned Future Latents for Driving World Models.md, raw/papers/Auto-JEPA_ A Latent World Model of Continuous Intent for End-to-End Autonomous Driving.md, raw/papers/DriveSuprim_ Towards Precise Trajectory Selection for End-to-End Planning.md, raw/papers/DiffusionDriveV2_ Reinforcement Learning-Constrained Truncated Diffusion Modeling in End-to-End Autonomous Driving.md, raw/papers/From Representational Complementarity to Dual Systems_ Synergizing VLM and Vision-Only Backbones for End-to-End Driving.md, raw/papers/Drive-JEPA_ Video JEPA Meets Multimodal Trajectory Distillation for End-to-End Driving.md, raw/papers/HAD_ Combining Hierarchical Diffusion with Metric-Decoupled RL for End-to-End Driving.md, raw/papers/CLEAR_ Cognition and Latent Evaluation for Adaptive Routing in End-to-End Autonomous Driving.md, raw/papers/Fine-tuning is Not Enough_ A Parallel Framework for Collaborative Imitation and Reinforcement Learning in End-to-end Autonomous Driving.md]
related: [sources/da-wam.md, sources/auto-jepa.md, sources/drivesuprim.md, sources/diffusiondrive-v2.md, sources/hybriddriveVLA.md, sources/dreameraD.md, sources/drive-jepa.md, sources/had.md, sources/clear.md, sources/pair-drive.md, concepts/navsim-benchmark.md, concepts/best-of-n.md, concepts/diffusion-planner.md, concepts/rl-for-ad.md, concepts/adaptive-routing.md, concepts/parallel-il-rl.md]
created: 2026-04-23
updated: 2026-09-02
confidence: high
---

## What It Is

Selection-based planning is a trajectory prediction paradigm for end-to-end autonomous driving. Rather than regressing a single trajectory or sampling stochastically, the model selects the best option from a **fixed pre-defined vocabulary** of candidate trajectories.

---

## Core Paradigm

```
Vocabulary: {τ₁, τ₂, ..., τ_N}   (N ≈ 8192 candidates, pre-computed)
                     ↓
Scorer: estimates quality s_i^(m) per trajectory per metric m
                     ↓
Selection: T = τ_k  where k = argmax_i s_i
```

The vocabulary is generated offline (e.g., K-Means over expert trajectories) and covers diverse maneuver types. At inference, the model does not generate trajectories — it scores all candidates and returns the top-ranked one.

**Key distinction from other paradigms**:
| Paradigm | At inference | Trajectory source |
|---|---|---|
| Regression | Outputs one trajectory | Learned regressor |
| Diffusion/FM | Samples from noise | Denoising process |
| Best-of-N sampling | Runs N forward passes | Same model, N times |
| Selection-based | Scores N fixed candidates | Pre-defined vocabulary |
| Selection-based + BoN | Oracle over multiple runs | Vocabulary + stochastic |
| **Latent retrieval** | **Nearest-neighbor lookup, then score the top-K** | **Recorded trajectory memory, indexed by a learned latent** |

---

## Theoretical Ceiling (Oracle Study)

DriveSuprim's oracle study (Table 1) quantifies how much selection-based methods can achieve with perfect scoring:

| Top-K oracle selection | PDMS |
|---|---|
| Top-1 (best current model) | 91.9 |
| Top-4 | 94.5 |
| Top-16 | 96.1 |
| Top-256 | 98.7 |
| Human ground truth | 94.8 |

**Key insight**: with oracle selection from just **4 candidates**, you nearly match human GT (94.5 vs. 94.8). With 256 candidates, you reach 98.7 PDMS — near-perfect on NAVSIM. The bottleneck is entirely in the **selector quality**, not candidate coverage.

This ceiling is higher than stochastic BoN-N results (e.g., Curious-VLA BoN-6 = 94.8 PDMS at N=6) because the vocabulary is purpose-built for coverage, whereas stochastic sampling from a single model produces correlated outputs. See [[concepts/best-of-n.md]].

---

## Three Failure Modes

Selection-based methods share three structural weaknesses identified by DriveSuprim:

### 1. Hard Negatives

The vocabulary contains thousands of obviously bad trajectories ("easy negatives"). During training, BCE loss forces the model to score these correctly — but this dominates the gradient signal. The model rarely encounters two plausible-looking trajectories where one is subtly unsafe ("hard negatives"). As a result, fine-grained discrimination remains weak.

**Fix (DriveSuprim)**: coarse-to-fine filtering — first pass selects top-256 (mostly hard negatives once obvious ones are removed), second pass scores only those 256 at higher precision.

### 2. Directional Bias

Real driving is dominated by straight-ahead motion. In NAVSIM, only 8% of ground-truth trajectories involve turns >30°. Training on this distribution naturally produces a model that underperforms on turns.

**Fix (DriveSuprim)**: rotation-based data augmentation — simulate ego rotation by shifting camera FOV, proportionally rotating GT trajectory.

### 3. Hard Binary Labels

Safety scores are {0,1} per metric. BCE against binary labels creates sharp training boundaries — a trajectory just below the collision threshold is treated identically to a catastrophically bad one. This causes training instability and oversensitivity to minor trajectory variations.

**Fix (DriveSuprim)**: EMA self-distillation with clipped soft labels ($\delta_m = 0.15$).

---

## Methods in the Wiki Using Selection-Based Planning

| Method | Vocabulary Size | Scoring | Notes |
|---|---|---|---|
| **Hydra-MDP** | 8192 | Single-stage multi-head | Multi-teacher distillation; won NAVSIM challenge |
| **HydraMDP++** | 8192 | Single-stage multi-head | Added DDC, TLC, EC metrics for NAVSIM-v2 |
| **DriveSuprim** | 8192 (→ 256) | Two-stage coarse-to-fine | Rotation aug + EMA self-distill; **93.5 PDMS** |
| **DreamerAD** | 8192 (→ 256) | Learned latent AD-RM | Gaussian vocab sampling; reward from latent WM |
| **HybridDriveVLA** | 2 + 9 interp. | Trajectory scorer | Cross-model (VLM + ViT) with linear interpolations |
| **HAD** | 8192 reward cache + 20 coarse anchors -> 50 local candidates | Hierarchical diffusion + metric heads | Uses selection vocabulary for offline reward retrieval and coarse-to-fine local generation; 88.6 EPDMS |
| **Drive-JEPA** | 8192 pseudo-teacher vocabulary + 32 online proposals | Proposal scoring + momentum-aware selection | Uses vocabulary for simulator-distilled supervision, not direct fixed selection; 93.3 PDMS NAVSIM-v1 |
| **Auto-JEPA** | 110,335 recorded GT trajectories (→ top-300 by latent cosine) | CLOVER-initialized scene scorer + DAC gate | Retrieval, not classification: the candidate set is scene-dependent; 91.3 PDMS NAVSIM-v1 |
| **DA-WAM** | 32 generated proposals + retrieved hard negatives | Factorized NC/DAC/EP/TTC/Comfort heads → utility head, conditioned on **each candidate's own predicted future latent** | First scorer conditioned on per-candidate futures rather than scene geometry alone; 93.7 PDMS NAVSIM-v1 |

### DreamerAD as a deployable selection variant

DreamerAD generates 256 trajectories via Mahalanobis-ranked Gaussian sampling over the 8192 vocabulary, then selects via a learned reward model (AD-RM) trained on latent video features — no PDM simulator needed at inference. This is the closest approach in the wiki to a deployable selection system: the selection quality is approx. but fast. +2.6 EPDMS from base to selected. See [[sources/dreameraD.md]].

---

### HAD: Selection as Reward Cache + Local Refinement

HAD ([[sources/had.md]]) is not a pure fixed-vocabulary selector like DriveSuprim. It uses an 8192-trajectory vocabulary primarily as an offline reward-retrieval cache: nearest-neighbor matching maps generated trajectories to precomputed metric rewards, avoiding online simulator calls during RL. The deployed policy still generates and refines trajectories with hierarchical diffusion.

The selection connection is the coarse-to-fine structure. HAD first narrows the global plan to top-K coarse intentions, then expands local candidates around those intentions and learns metric-specific scores. This gives some of the hard-negative concentration benefits of selection-based methods without constraining the final trajectory to a fixed library entry.

### Drive-JEPA: Simulator-Distilled Online Proposals

Drive-JEPA ([[sources/drive-jepa.md]]) is adjacent to selection-based planning but should not be classified as a pure fixed-vocabulary selector. It clusters the training set into an 8192-trajectory vocabulary and uses a NAVSIM-v2-style simulator to choose high-scoring pseudo-teacher trajectories above an EPDMS threshold of 0.95. Those trajectories supervise the distribution of 32 continuous online proposals during training.

The deployed planner still generates and refines proposals with Waypoint-anchored Deformable Attention. The vocabulary is therefore a training-time distillation device, not the inference-time trajectory source. The key failure mode is comfort: MTD increases diversity from 24% to 40%, but EC drops to 47.9 unless the momentum-aware selector compares proposals with the previous selected trajectory.

### Auto-JEPA: Latent Retrieval Instead of Fixed-Vocabulary Scoring

[[sources/auto-jepa.md]] is the wiki's first planner whose candidate set is **retrieved rather than fixed**. Every other method on this page presents the scorer with the same $N$ trajectories in every scene — 8192 K-Means clusters, 20 diffusion anchors, 32 online proposals. Auto-JEPA predicts a continuous 8×1024 "intent" latent, uses it as a query into a memory of 110,335 recorded ground-truth trajectories under flat cosine similarity, and hands the top-300 to a scorer and a drivable-area gate.

**Why this is architecturally different, not just a bigger vocabulary:**

| | Fixed vocabulary (DriveSuprim) | Latent retrieval (Auto-JEPA) |
|---|---|---|
| Candidate set | Identical in every scene | Scene-dependent, chosen by the query |
| First-stage narrowing | Learned coarse scorer over all 8192 | Cosine nearest-neighbor in latent space |
| Candidate geometry | Cluster centroids | Real recorded trajectories, unclustered |
| What the scorer sees | Hard negatives *plus* whatever survived coarse scoring | Only intent-compatible geometry |
| Failure mode | Bad ranking | Bad *recall* — a correct maneuver never reaches the scorer |

The consequence for DriveSuprim's hard-negative analysis is worth spelling out. Coarse-to-fine filtering works because Stage 2 faces a concentrated set of plausible-looking trajectories. Auto-JEPA gets the same concentration for free — retrieval by intent similarity returns 300 trajectories that are all *maneuver-appropriate* by construction — but it inherits a failure mode fixed vocabularies do not have. A fixed vocabulary always contains the right maneuver somewhere; only the scorer can lose it. In retrieval, the query can simply miss, and the paper acknowledges that "if no feasible maneuver is represented in the retrieved candidate pool, neither the scene scorer nor the feasibility gate can synthesize one." **The oracle ceiling analysis on this page therefore does not transfer**: DriveSuprim's 98.7 PDMS at top-256 assumes the 256 came from a set that covers the space. No retrieval-recall study exists for Auto-JEPA.

**What the ablation actually attributes.** $K=1$ — pure retrieval, no selection — scores 87.6 PDMS. Going to $K=200$ buys +3.5, and 200 → 300 only +0.2. So the selection stage is worth roughly what it is worth in fixed-vocabulary methods, and Auto-JEPA's evidence for "candidate selection matters" is the same shape as DriveSuprim's. The difference is where the remaining headroom sits: DriveSuprim's is in the scorer (oracle 98.7 vs. achieved 93.5), Auto-JEPA's is split between scorer quality and memory coverage, and the paper cannot separate them.

**Two caveats specific to this design.** The scorer is *initialized from the released CLOVER checkpoint* and contributes +3.7 of the 91.3, so the deployed selector is largely inherited rather than novel. And retrieval offers no frame-to-frame continuity: consecutive frames can land on different memory entries with nothing penalizing the jump. Drive-JEPA hit exactly this and needed a momentum-aware selector (EC 47.9 → 84.8); Auto-JEPA's EC of 75.2 on NAVSIM-v2 is near the bottom of the wiki, and the paper does not discuss it.

### DA-WAM: Scoring Candidates Against Their Own Predicted Futures

Every scorer on this page evaluates candidates against the **current** scene — geometry, BEV features, VLM hidden states, or a learned reward model over one latent world state. [[sources/da-wam.md]] adds a conditioning input none of them have: **a distinct predicted future latent for each candidate**, produced by a shared predictor that uses the candidate's action encoding as the attention query.

The diagnosis motivating it is the same one DriveSuprim makes, arrived at independently: a scorer trained on geometrically diverse candidates "may rely primarily on geometric cues rather than the scene-conditioned future content that distinguishes safe from unsafe outcomes." Both papers then attack it from opposite ends — DriveSuprim by *concentrating* the candidate set so geometry stops being discriminative, DA-WAM by *adding* a signal geometry cannot supply.

**Two contributions, and the sizes are the opposite of what the framing suggests:**

| Component | PDMS gain |
|---|---:|
| Per-candidate future conditioning (vs. no future prediction) | **+0.15** |
| Safety-critical hard negatives | **+0.22** |
| *(for scale)* Representation choices: LoRA + V-JEPA 2.1 dense + EMA target | +2.42 |

**The hard-negative construction is the more transferable half.** Negatives are retrieved from an offline trajectory bank under two simultaneous constraints — geometrically close to the expert ($d_\mathrm{traj}<\epsilon_\mathrm{geo}$) but substantially worse in safety ($\Delta_\mathrm{safety}>\epsilon_\mathrm{safety}$) — then appended to the candidate set, given their own future latent, and passed through the same shared scorer with upweighted ranking pairs. They are excluded from expert matching and dense future supervision because their visual futures are unobserved.

This is a **retrieval-based** answer to the hard-negative problem, where DriveSuprim's is *filtering*-based and HAD's is a reward cache. Retrieval has an advantage the wiki should note: DriveSuprim's Stage 1 can only surface hard negatives that its coarse scorer already ranks highly, whereas DA-WAM's constraints target the region of trajectory space that is *geometrically indistinguishable from the expert but unsafe* — precisely the region a scorer relying on curvature and speed will get wrong. The cost is that both $\epsilon$ thresholds and the bank's construction go unreported.

**The scorer architecture also matters and is easy to miss.** $S_\psi^\mathrm{enc}$ cross-attends scene tokens, action representation, and future latent while "preserving fine-grained token-level interactions rather than pooling futures into a coarse proposal-invariant vector." Pooling is what DA-WAM's Figure 1(c) identifies as the standard mistake, and its ablation measures a pooled/shared future at **0.50 PDMS worse than no future at all** — so the anti-pooling design is load-bearing in the negative direction even where the positive gain is small.

**Candidate-count behaviour** differs sharply from the retrieval planners on this page: 1 → 87.11, 8 → 90.76, 16 → 91.89, 32 → 93.68, 64 → 93.68. Saturation at 32 generated candidates, against Auto-JEPA needing 300 retrieved ones to reach 91.3. Generated proposals conditioned on the scene cover the useful space far more efficiently than nearest neighbours in a fixed memory.

### PaIR-Drive: Residual Tree plus Reward World Model

[[sources/pair-drive.md]] turns an IL trajectory into the root of a recurrent proposal tree. Intention tokens generate residual branches; a learned reward world model scores their predicted reward and confidence and chooses the final plan. This is a hybrid of generative refinement and selection rather than fixed-vocabulary classification.

The RWM ablation reports 88.1/84.3 PDMS/EPDMS for vanilla DiffusionDrive, 90.2/87.0 for IL + RWM, and 94.0/89.6 for PaIR-Drive + RWM under the paper's selected setting. The comparison supports the value of the tree generator, but does not provide PaIR-Drive without RWM. Selector calibration, architecture, and latency are also missing, so the deployment mechanism is less reproducible than the GRPO sampler.

## Coarse-to-Fine Selection (DriveSuprim)

The critical DriveSuprim ablation (Table 5):

| Modification | EPDMS | Change |
|---|---|---|
| Single-stage (Hydra-MDP, ViT-L) | 85.6 | baseline |
| + 6-layer decoder (more parameters) | 85.3 | −0.3 |
| + Layer-wise scoring (aux loss per layer) | 85.6 | +0.3 |
| + Trajectory filtering to 256 | **86.4** | **+0.8** |

Only trajectory filtering helps. Adding decoder depth or auxiliary supervision without filtering does nothing. The model must be presented with a concentrated hard-negative set.

**Why this works**: once easy negatives are removed in Stage 1, Stage 2 faces a set of trajectories that all look plausible. The refinement decoder must develop genuine fine-grained discrimination. The gradient signal from easy negatives no longer dominates.

This is analogous to Cascade R-CNN for object detection (two-stage cascade with progressively tightening IoU thresholds), applied to trajectory scoring.

---

## Rotation-Based Augmentation

The augmentation pipeline:

1. Sample rotation angle $\theta \sim U[-\pi/6, +\pi/6]$
2. Concatenate three cameras into pseudo-panoramic view: $[l_0 | f | r_0]$
3. Crop the standard-FOV window from the panorama, shifted by $\theta$
4. Rotate GT trajectory waypoints $(u_1, \ldots, u_l)$ by $-\theta$ around origin $u_0$
5. Compute loss $L_{\text{aug}}$ identically to $L_{\text{ori}}$

**Effect**: the original NAVSIM dataset has a forward-heavy trajectory distribution. Post-augmentation, all directions appear at similar frequency (Figure 4 in [[sources/drivesuprim.md]]).

**Performance impact by scenario type**:
| Scenario | Gain vs. no augmentation |
|---|---|
| Turning scenarios | +2–3% EPDMS |
| Near-straight scenarios | +0.9% EPDMS |

This is first application in the AD wiki of camera-shift-based rotation augmentation for trajectory planning.

---

## Relationship to Best-of-N Sampling

Selection-based planning and BoN sampling are often confused but are architecturally different:

| Property | Selection-based | Stochastic BoN |
|---|---|---|
| Trajectory source | Pre-defined fixed vocabulary | N model forward passes |
| Selection at inference | Learned scorer | Oracle (PDM simulator) |
| Deployable? | Yes (scorer replaces oracle) | No (oracle unavailable) |
| Ceiling | High: 98.7 PDMS (256-oracle) | Medium: 94.8 PDMS (N=6, Curious-VLA) |
| Diversity source | Vocabulary design | Stochastic decoding |

The fixed-vocabulary oracle ceiling (98.7 PDMS at top-256) is substantially higher than stochastic BoN (94.8 at N=6). This is because the vocabulary is curated to cover diverse maneuver types systematically, while stochastic decoding from a single model produces correlated near-optimal outputs.

The practical convergence point is in deployable selectors: DreamerAD (latent reward model over 256 vocabulary candidates) and HybridDriveVLA (cross-model scorer) both convert oracle selection into feasible inference, with partial but real gains. See [[concepts/best-of-n.md]].

---

## NAVSIM Performance Overview

Selection-based methods' trajectory on the NAVSIM-v1 leaderboard:

| Method | PDMS (ViT-L) | Year |
|---|---|---|
| Hydra-MDP | 89.9 | 2024 |
| HydraMDP++ | 85.6* (EPDMS) | 2024 |
| DreamerAD | 88.7 (no ViT-L) | 2025 |
| **DriveSuprim** | **93.5** | 2025 |

*HydraMDP++ is evaluated primarily on NAVSIM-v2 (EPDMS).

DriveSuprim (93.5) remains the strongest fixed-vocabulary selection result in the wiki, surpassing DiffusionDriveV2 (91.2 with Camera+LiDAR) and HybridDriveVLA (92.1 dual-model ensemble). CLEAR later reports 93.7 with online candidate generation plus learned adaptive routing, so it is adjacent to selection but not a fixed-vocabulary selector. Auto-JEPA (91.3) is adjacent in the other direction — retrieval rather than classification — and is the cheapest of the three to train, since its visual encoder is frozen and only small task modules are optimized. See [[concepts/navsim-benchmark.md]] and [[concepts/adaptive-routing.md]].
