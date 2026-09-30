---
title: Selection-Based Trajectory Planning
type: concept
sources: ["raw/papers/MM-Future_ Multi-Mode Joint World–Action Modeling for Autonomous Driving.md", "raw/papers/DriveReferee_ Geometric Safety Verdicts Need Not Be Learned for Driving World-Action Models.md", "raw/papers/MomWorld_ Momentum-Aware Latent World Model for Long-Horizon Autonomous Driving.md", "raw/papers/PhysWAM_ Physically Consistent World Action Model for Autonomous Driving.md", "raw/papers/AD-E2E-JEPA_ A Joint-Embedding Predictive Architecture For End-to-End Autonomous Driving.md", "raw/papers/Hydra-MDP++_ Advancing End-to-End Driving via Expert-Guided Hydra-Distillation.md", raw/papers/DriveFuture_ Future-Aware Latent World Models for Autonomous Driving.md, raw/papers/Unified Driving Tokens_ Representation- and Geometry-Guided Discrete Tokenizer for Driving World Models and Planning.md, raw/papers/ReWorld_ Representation Learning for World Action Models.md, raw/papers/LWDrive_ Layer-Wise World-Model-Guided Vision-Language ModelPlanning for Autonomous Driving.md, raw/papers/GeoWorldAD_ Geometry World Action Model for Autonomous Driving.md, raw/papers/Adaptive-WAM_ Quality-Guided Early-Exit Planningfrom Intermediate Video-Diffusion Features.md, raw/papers/DA-WAM_ Decision-Aligned Future Latents for Driving World Models.md, raw/papers/Auto-JEPA_ A Latent World Model of Continuous Intent for End-to-End Autonomous Driving.md, raw/papers/DriveSuprim_ Towards Precise Trajectory Selection for End-to-End Planning.md, raw/papers/DiffusionDriveV2_ Reinforcement Learning-Constrained Truncated Diffusion Modeling in End-to-End Autonomous Driving.md, raw/papers/From Representational Complementarity to Dual Systems_ Synergizing VLM and Vision-Only Backbones for End-to-End Driving.md, raw/papers/Drive-JEPA_ Video JEPA Meets Multimodal Trajectory Distillation for End-to-End Driving.md, raw/papers/HAD_ Combining Hierarchical Diffusion with Metric-Decoupled RL for End-to-End Driving.md, raw/papers/CLEAR_ Cognition and Latent Evaluation for Adaptive Routing in End-to-End Autonomous Driving.md, raw/papers/Fine-tuning is Not Enough_ A Parallel Framework for Collaborative Imitation and Reinforcement Learning in End-to-end Autonomous Driving.md]
related: [sources/mm-future.md, sources/drivereferee.md, sources/momworld.md, sources/physwam.md, sources/ad-e2e-jepa.md, concepts/world-model-for-ad.md, concepts/inference-latency.md, sources/hydra-mdp-pp.md, sources/drivefuture.md, concepts/navhard-ood-evaluation.md, sources/unified-driving-tokens.md, concepts/visual-tokenization.md, sources/reworld.md, sources/lwdrive.md, sources/geoworldad.md, sources/adaptive-wam.md, sources/da-wam.md, sources/auto-jepa.md, sources/drivesuprim.md, sources/diffusiondrive-v2.md, sources/hybriddriveVLA.md, sources/dreameraD.md, sources/drive-jepa.md, sources/had.md, sources/clear.md, sources/pair-drive.md, concepts/navsim-benchmark.md, concepts/best-of-n.md, concepts/diffusion-planner.md, concepts/rl-for-ad.md, concepts/adaptive-routing.md, concepts/parallel-il-rl.md]
created: 2026-04-23
updated: 2026-09-30
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

**Three mechanisms now exist for the same problem, and only one needs no scorer at inference.**

| Method | How negatives are obtained | How they are used | Worth |
|---|---|---|---:|
| [[sources/drivesuprim.md]] | Coarse pass discards easy negatives from an 8192 vocabulary | Stage 2 scores the surviving 256 at higher precision | part of 93.5 |
| [[sources/da-wam.md]] | Retrieved safety-critical trajectories | Extra training rows for the candidate scorer | **+0.22** |
| [[sources/reworld.md]] | 64 generated candidates per scene scored by the **NAVSIM PDM simulator**; take the *nearest* one below 0.6 | **Repulsive loss on a flow-matching planner's own output** — no candidate set, no scorer | **+0.7** |

ReWorld's variant is the odd one out and the most transferable. It converts hard negatives from *scorer training data* into a **regularizer on a generative policy**: recover $\hat a_0$ from the same forward pass and the same $t_a$ already used by the flow-matching loss, map both trajectories to a delta representation $[\widetilde{\Delta x},\widetilde{\Delta y},\sin\psi,\cos\psi]$, and maximize L1 distance while $\mathcal{L}_{\mathrm{FM}}$ anchors to the expert. Cost is zero extra forward passes, and nothing about a candidate pool survives to inference.

The selection rule is the load-bearing part, and it is DriveSuprim's insight restated: *"random negatives are often distinguishable by geometry alone"*, so the negative must be the **closest** low-scoring trajectory rather than a sampled one. All three mechanisms mine with privileged supervision — the simulator, or a scorer trained on it — so none is annotation-free. And ReWorld's is unbounded below in isolation and collapses PDMS to 85.5 at $\lambda=0.10$ against 90.4 at 0.04; it works only as a weak counterweight to the quadratic imitation term.

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
| **HydraMDP++** ([[sources/hydra-mdp-pp.md]]) | 8192 | Single-stage multi-head | Added TL, DDC, LK, EC as metrics *and* extra distillation teachers; weighted-cost selection (+1.5 PDMS); its own EPDMS formula differs from NAVSIM-v2's |
| **DriveSuprim** | 8192 (→ 256) | Two-stage coarse-to-fine | Rotation aug + EMA self-distill; **93.5 PDMS** |
| **DreamerAD** | 8192 (→ 256) | Learned latent AD-RM | Gaussian vocab sampling; reward from latent WM |
| **HybridDriveVLA** | 2 + 9 interp. | Trajectory scorer | Cross-model (VLM + ViT) with linear interpolations |
| **HAD** | 8192 reward cache + 20 coarse anchors -> 50 local candidates | Hierarchical diffusion + metric heads | Uses selection vocabulary for offline reward retrieval and coarse-to-fine local generation; 88.6 EPDMS |
| **Drive-JEPA** | 8192 pseudo-teacher vocabulary + 32 online proposals | Proposal scoring + momentum-aware selection | Uses vocabulary for simulator-distilled supervision, not direct fixed selection; 93.3 PDMS NAVSIM-v1 |
| **Auto-JEPA** | 110,335 recorded GT trajectories (→ top-300 by latent cosine) | CLOVER-initialized scene scorer + DAC gate | Retrieval, not classification: the candidate set is scene-dependent; 91.3 PDMS NAVSIM-v1 |
| **DA-WAM** | 32 generated proposals + retrieved hard negatives | Factorized NC/DAC/EP/TTC/Comfort heads → utility head, conditioned on **each candidate's own predicted future latent** | First scorer conditioned on per-candidate futures rather than scene geometry alone; 93.7 PDMS NAVSIM-v1 |
| **Adaptive-WAM** (aux model) | 64 proposals at a fixed block-22 exit | Six-component DINOv2-Small verifier (soft-label BCE, **no rank loss**) | CLOVER pseudo-expert targets scored by the true NAVSIM evaluator at training time; 92.6 PDMS; see [the tie problem](#tie-problem) |
| **GeoWorldAD** | 64 learned proposals, refined over 5 stages | MLP head trained with BCE against the NAVSIM simulator's own PDMS composition | Min-over-proposals supervision at every refinement stage; simulator-distilled scoring like Hydra-MDP; 91.0 PDMS |
| **Unified Driving Tokens** | Multiple trajectories from a 20M readout on frozen visual tokens (**count unreported**) | MLP score head predicting PDM-style metric outcomes, BCE against a rule-based evaluator run on the *predicted* trajectory | Min-over-N regression (only the closest trajectory is supervised); the scorer is retrained per tokenizer, so representation ablations partly measure scorer quality; 91.8 PDMS |
| **LWDrive** | $N_\mathrm{p}$ proposals (**size never stated**), refined over 6 stages | MLP head, BCE against per-candidate PDMS from **non-reactive log simulation of every candidate** | Pool initialised from the VLM's pooled action-query latent; Bridge Attention gives proposals a self-memory alongside the VLM foresight memory; min-over-$N$ at every stage with an exponential discount on earlier ones; 92.0 PDMS |
| **MomWorld** ([[sources/momworld.md]]) | 16,384 (GTRS-Dense base; half dropped per training step) | Hydra-style multi-head scorer (imitation, NC, DAC, TTC, EP, DDC, LK, TLC) whose candidate queries also attend to a **40-token latent future memory shared by all candidates** | The selected plan is then moved by a residual flow and **not re-scored**; 90.2 PDMS, 42.8 navhard (+1.1 over GTRS-Dense); see [below](#future-memory-scorer) |
| **DriveReferee** ([[sources/drivereferee.md]]) | 1 default sample, plus 1 more on an alarm (34% of scenes) | **No learned scorer.** The evaluator's collision and drivable-area geometry, executed on a predicted BEV map; ranks by violation count, then clearance | Selection is worth +0.30 EPDMS on the base policy and nothing after the same rule is distilled into the policy; 92.02 PDMS; see [below](#computed-verdict) |
| **MM-Future** ([[sources/mm-future.md]]) | 64 jointly generated trajectory–future pairs (32 in its ablation) | Heads for the PDMS components trained on simulator scores of the sampled trajectories; each proposal reads the history and **its own generated future** (block-diagonal attention) | Many modes plus a history-only scorer: +8.2 over one trajectory; reading the paired future: +0.4; 93.4 PDMS on navtrain, 94.0 on trainval; see [below](#mm-future) |

| **AD-E2E-JEPA** (zero-shot mode) | 8192, angularly subsampled to 256 (also 512 … 8192) | **No scorer.** Squared latent distance between each candidate's 4 s world-model rollout and the **ground-truth future frame** | Oracle-goal diagnostic, not a deployable selector; 67.3 EPDMS at 256 / 72.9 at 8192; see [below](#no-scorer) |

| **PhysWAM** (medoid mode) | 8 joint samples of the full future | **No scorer and no labels.** The sample with the smallest total planar distance to the other seven | +0.3 "PDMS" / +0.1 EPDMS on navtest, +1.7 on navhard; oracle over the same eight is +3.8 EPDMS; see [below](#consensus) |

**[[sources/drivefuture.md]]** is a boundary case worth listing separately: it is a *generator* (a future-conditioned diffusion planner producing 100 proposals) that submits through a GTRS-Dense scorer it did not train and does not describe. It therefore belongs in this family only at inference — and it is the one paper here that reports what that membership is worth. See [below](#scorer-price).

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

### AD-E2E-JEPA: Selection With No Scorer, and an Oracle in Its Place {#no-scorer}

[[sources/ad-e2e-jepa.md]] uses the Hydra vocabulary with nothing learned on top of it. A JEPA world model rolls each candidate out 4 s in a 32-token latent space, and the candidate whose final latent is nearest the **real future frame's** latent wins. No simulator labels, no BCE heads, no weights to grid-search. The price is that the future frame has to be supplied, so this is a probe of the world model and not a planner.

It is still informative for this page, because it is the only entry that scales the candidate count while holding everything else fixed and reports cost at each size:

| Candidates | EPDMS | NC | DAC | EC | FDE (m) | Time (s, A100) |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 67.3 | 95.2 | 82.0 | 35.5 | 4.0 | 0.8 |
| 512 | 69.2 | 95.9 | 83.6 | 36.9 | 3.6 | 1.4 |
| 1,024 | 70.5 | 96.4 | 84.2 | 38.5 | 3.2 | 2.5 |
| 2,048 | 71.5 | 96.7 | 84.8 | 40.9 | 3.0 | 4.7 |
| 4,096 | 72.1 | 96.7 | 85.4 | 42.5 | 2.9 | 9.3 |
| 8,192 | 72.9 | 96.9 | 86.0 | 43.8 | 2.8 | 18.2 |

**Three readings.**

1. **Coverage matters here in a way DriveSuprim's oracle says it should not.** The [oracle study](#theoretical-ceiling-oracle-study) puts 98.7 PDMS within reach of 256 candidates, so the vocabulary is not the bottleneck for a *score* oracle. For an *endpoint-matching* criterion it is: 32× more candidates still buys +5.6 EPDMS and 1.2 m of FDE. The two criteria want different things from a vocabulary. A score oracle needs one good trajectory per scene. Endpoint matching needs a candidate near one specific trajectory.
2. **The subsample is even in angle, not in length.** Final-pose error at 256 is 3.6 m longitudinal against 1.1 m lateral. How much of that is the vocabulary and how much the world model is unknown, because the paper reports no nearest-endpoint oracle.
3. **Per-frame selection without a continuity term destroys extended comfort.** EC is 35.5–43.8, the lowest in the wiki, even though every frame is aimed at the human's own endpoint. [[sources/drive-jepa.md]] needed a momentum-aware selector for the same reason (EC 47.9 → 84.8), and [[sources/auto-jepa.md]] sits at 75.2. **Three selection designs have now hit this, and only one fixed it.**

**Relation to the rest of the page.** Every other scorer here is distilled from the benchmark's simulator and carries the [scorer-cohort caveat](#top-of-leaderboard). This one is the opposite extreme: no privileged *labels*, a privileged *input*. Neither is a deployable, label-free selector. The combination nobody has built is this rollout model with a learned reward head in place of the goal, which is [[sources/dreameraD.md]]'s recipe on a 3 ms-per-candidate latent model. See [[concepts/world-model-for-ad.md#world-model-as-planner]].

### PhysWAM: Consensus Selection, Priced Against Its Oracle {#consensus}

[[sources/physwam.md]] selects among $K=8$ samples with the medoid rule,

$$k^*=\operatorname*{argmin}_{k}\sum_{l=1}^{K}\frac1T\sum_{t=1}^{T}\big\|\mathbf y^{(k)}_t-\mathbf y^{(l)}_t\big\|_2$$

over planar positions only. No scorer is trained, no simulator is called and no label is used. It is the second label-free selector on this page in two ingests, after [the oracle-goal search above](#no-scorer), and the first that needs no privileged input at inference.

| Selector | Trained on | navtest gain | navhard gain |
|---|---|---:|---:|
| Medoid of 8 ([[sources/physwam.md]]) | Nothing | +0.1 EPDMS | **+1.7** |
| Oracle over the same 8 | – (needs the simulator) | +3.8 EPDMS | – |
| Analytic NC/DAC rule on a predicted map, 2 samples ([[sources/drivereferee.md]]) | Map and box labels for a 21M readout; **no verdict labels** | +0.30 EPDMS (0 after the rule is distilled into the policy) | – |
| Learned verifier on the same predicted map, 2 samples ([[sources/drivereferee.md]]) | The same, plus 66,385 evaluator-labelled candidates | −0.01 against the analytic rule (CI ±0.15) | – |
| GTRS-Dense over 100 proposals ([[sources/drivefuture.md]]) | Simulator-derived scores | – | **+20.9** |

**Three readings.**

1. **Consensus is nearly worthless on navtest and recovers 3% of what an oracle finds in the same samples.** The mode of a generator's own distribution is not where the good plans are. Whatever a scorer contributes, it is information the samples do not carry about themselves.
2. **It is worth something on navhard**, where both stage scores and extended comfort improve (Stage-1 EC 67.6 → 72.4). A plausible reason is that the central sample excludes outlier plans, which the two-stage protocol punishes twice.
3. **It puts a floor under the scorer cohort.** The gap between +1.7 and +20.9 is what privileged supervision buys on this split. A label-free rule does not get a method across the [42-point line](../concepts/navhard-ood-evaluation.md#scorer-cohort).

The cost is eight full samples per decision, which for this model is about 75 GPU-seconds.

### MomWorld: a Forecast Inside the Scorer, and a Refinement After It {#future-memory-scorer}

[[sources/momworld.md]] changes a GTRS-Dense scorer in two places.

| Stage | Base (GTRS-Dense) | MomWorld |
|---|---|---|
| What a candidate query attends to | Current scene features | Current scene queries **and** a 40-token memory from an action-free latent rollout |
| What is output | The top-scoring vocabulary trajectory | That trajectory plus a clipped, horizon-weighted correction from a 4-step flow |

**1. The future is shared, not per-candidate.** One rollout serves all 16,384 candidates. [[sources/da-wam.md]] measured that arrangement at −0.50 PDMS and built per-candidate futures to avoid it. MomWorld's argument is that distinct candidate queries attend to the shared memory differently. A per-candidate rollout at this vocabulary size would be 16,384 rollouts, so the shared design is also the only affordable one ([[concepts/inference-latency.md#per-candidate]]).

**2. The output leaves the vocabulary and is not checked again.** After selection the plan is shifted by up to δ = 2.0 per component at the last waypoint (scaled by a learned global gate that starts at 0.018). The paper states MoFlow runs "without reconstructing the memory or re-ranking the candidate vocabulary". Every other refine-after-select design on this page (DriveSuprim's second stage, HAD's local candidates, LWDrive's staged pool) scores what it finally outputs.

**3. What it is worth.** navhard 41.7 → 42.8. The [sub-score comparison](../concepts/navhard-ood-evaluation.md#scorer-cohort) shows progress and comfort up, NC and TTC down. That is the same order of magnitude [Hydra-MDP++](#origin) found for a temporal module (+0.1) against the selection rule (+1.5), and far below the +20.9 a scorer is worth over no scorer.

The paper's component ablations are not cited here; see [[sources/momworld.md#regularities]].

### DriveReferee: a Computed Verdict Against Learned Ones {#computed-verdict}

Every scorer on this page learns a mapping from sensor features and a candidate to a verdict. [[sources/drivereferee.md]] tests whether the verdict half needs learning, for the two sub-scores that are pure geometry (NC and DAC).

**The matched-budget comparison** (same base WAM, same two pre-sampled candidates per scene, all operating points calibrated to about 1.35× samples):

| Verdict source | Δ EPDMS | Replaced plans |
|---|---:|---:|
| Random / smoother / more conservative | +0.02 / −0.01 / −0.02 | 2,093 / 1,966 / 1,514 |
| Learned, score gating (Hydra-MDP style) | +0.23 | 1,246 |
| Learned, confidence gating (DriveVer style) | +0.26 | 1,540 |
| Learned, per-metric distillation | +0.06 | 2,174 |
| Learned, argmax (SparseDriveV2 style) | −0.04 | 2,161 |
| Learned, pairwise ranking | 0.00 | 2,017 |
| **Analytic rule on a predicted map** | **+0.30** | 1,339 |

**How to read it.**

1. **It is a non-inferiority result.** The interval on +0.30 is ±0.12 and the learned same-map verifier is within ±0.15. A computed verdict is not shown to be better; a learned one is not shown to be needed.
2. **The decision protocol dominates the verdict source.** The same kind of learned score is +0.23 when it gates and −0.04 when it always picks the top candidate. On a 90+ policy most swaps are between two acceptable plans, and an ungated selector mostly adds noise. Compare [the tie problem](#tie-problem): 1,339 replacements change 62 hard-gate outcomes.
3. **It is a two-candidate experiment.** The [+20.9 on navhard](#scorer-price) comes from 100 proposals on a split where the base fails often. Nothing here says a geometric rule would match a learned scorer there. The test is cheap to run and has not been.
4. **The learned side is under-resourced** by this page's standards: 4.15M parameters and 66,385 labels, against vocabulary-scale simulation in the [Hydra-MDP++ recipe](#origin).
5. **The rule's best use is in training.** Used to build 462 winner/loser pairs on ground-truth maps, it is worth +0.92 EPDMS, after which selecting with it adds nothing. That is this page's "a published number is a pipeline number" run in reverse: here the selector can be folded into the generator.

The privileged-supervision caveat still applies. The rule is the benchmark's own gate logic and its training-time inputs are annotated maps.

### MM-Future: What Many Modes, a Paired Future and a Future-Aware Scorer Are Each Worth {#mm-future}

[[sources/mm-future.md]] is a world–action model whose ablation doubles as a price list for this page. All rows are NAVSIM v1 navtest, trained on navtrain.

| Configuration | Hypotheses | Scorer input | PDMS | Latency |
|---|---:|---|---:|---:|
| Action-only flow | 1 | – | 84.1 | 52 ms |
| Action-only flow | 16 | History | 91.1 | 51 ms |
| Action-only flow | 32 | History | 92.3 | 65 ms |
| Joint trajectory–future flow | 32 | History | 92.9 | 132 ms |
| Joint trajectory–future flow | 32 | History + each proposal's own future | 93.3 | 131 ms |
| Joint trajectory–future flow (reported model) | 64 | History + own future | 93.4 | 233 ms |

1. **Proposal count with a simulator-trained scorer is the dominant term**: +7.0 at 16 and +8.2 at 32, from a weak single-trajectory baseline (84.1).
2. **A future to score against is worth +0.4**, the second such measurement after [DA-WAM's +0.15](#da-wam-scoring-candidates-against-their-own-predicted-futures).
3. **Doubling proposals from 32 to 64 is worth +0.1** and 100 ms.
4. **Generation and selection are not separated.** There is no many-mode row without a scorer and no oracle over the hypotheses, so this table cannot say how far the scorer is from the ceiling of its own candidates. [[sources/physwam.md]] is still the only entry that reports first-sample, selector and oracle on one set.
5. **The scorer's targets are the v1 components**, and it shows on v2: the highest ego progress of any ingested method (92.2) beside the lowest DDC, LK and HC in its own table.

Its encoder design comes from DrivoR (register tokens; not ingested), which scores 93.1 on navtrain with no world model. MM-Future is +0.3 above it.

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

*HydraMDP++'s paper evaluates on navtest with **its own** EPDMS formula (80.6 / 84.1). 85.6 is the ViT-L official-v2 value reported by later papers. See [[sources/hydra-mdp-pp.md#two-epdms]].

DriveSuprim (93.5) remains the strongest fixed-vocabulary selection result in the wiki, surpassing DiffusionDriveV2 (91.2 with Camera+LiDAR) and HybridDriveVLA (92.1 dual-model ensemble). CLEAR later reports 93.7 with online candidate generation plus learned adaptive routing, so it is adjacent to selection but not a fixed-vocabulary selector. Auto-JEPA (91.3) is adjacent in the other direction — retrieval rather than classification — and is the cheapest of the three to train, since its visual encoder is frozen and only small task modules are optimized. See [[concepts/navsim-benchmark.md]] and [[concepts/adaptive-routing.md]].

### Nothing Above 92 PDMS Scores Its Own Candidates {#top-of-leaderboard}

With [[sources/lwdrive.md]] ingested, this is now a complete statement about the top of the wiki's NAVSIM-v1 table rather than a tendency:

| Rank | Method | PDMS | What the scorer is trained on |
|---:|---|---:|---|
| 1= | [[sources/clear.md]] | 93.7 | Pairwise hinge + MSE against per-candidate PDMS |
| 1= | [[sources/da-wam.md]] | 93.7 | Factorized NC/DAC/EP/TTC/Comfort heads → utility, from simulator labels |
| 3 | [[sources/drivesuprim.md]] | 93.5 | Hydra-MDP multi-teacher distillation of simulator metrics |
| 4 | [[sources/mm-future.md]] (navtrain; 94.0 when trained on trainval) | 93.4 | BCE heads on simulator sub-scores of its own sampled trajectories, each read with its own generated future |
| 5 | [[sources/drive-jepa.md]] | 93.3 | 8192-entry pseudo-teacher vocabulary scored by the simulator |
| 6 | [[sources/wcog-vla.md]] | 92.9 | DiffGRPO whose reward *is* PDMS |
| 7 | [[sources/adaptive-wam.md]] (aux) | 92.6 | Six NAVSIM components predicted with soft-label BCE |
| 8 | [[sources/hybriddriveVLA.md]] | 92.1 | Component-wise BCE/regression on PDMS sub-scores |
| 9= | [[sources/lwdrive.md]] | 92.0 | BCE against per-candidate PDMS from log simulation |
| 9= | [[sources/drivereferee.md]] | 92.0 | No scorer needed at inference; the generator is trained on winner/loser pairs labelled by the evaluator's collision and drivable-area gates |

**Every entry above 92.0 in this wiki trains against the benchmark's own scoring function**, whether as a ranking head, as an RL reward (WCog-VLA) or as preference labels for the generator (DriveReferee). *(Table extended on 2026-09-30 with MM-Future and DriveReferee.)* The highest-scoring method that does *not* is [[sources/wa-jepa.md]] at 91.8.

This is not an accusation of cheating — simulator-distilled scoring is a legitimate and widely-declared design — but it does bound what the leaderboard measures. It says the last ~2 PDMS on NAVSIM-v1 has been bought by learning the evaluator rather than by improving the policy, and it predicts that the ordering among these ten would not survive a benchmark whose scoring function was withheld. The [oracle ceiling analysis](#theoretical-ceiling-oracle-study) and [the tie problem](#tie-problem) below both bear on how much headroom is actually left in that mechanism.

### What a Scorer Is Actually Worth: +20.9 EPDMS on navhard {#scorer-price}

The section above establishes *that* the top of NAVSIM-v1 scores its candidates. It could not say *how much* the scoring is worth, because no paper reported the same checkpoint both ways. [[sources/drivefuture.md]] does, on the harder split:

| Configuration | navhard combined EPDMS |
|---|---:|
| No future frames in training | 30.9 |
| + the paper's world-model mechanism | 34.6 |
| + **GTRS-Dense scorer over 100 diffusion proposals** | **55.5** |

The decomposition is recoverable because every ablation in the paper is labelled "without GTRS-Dense scorer" and the best ablation row's six submetrics match the unscored stage-wise row of its Table 7 digit-for-digit. **The scorer is worth 5.6x the architectural contribution of the paper it appears in**, and **85% of the distance** from the weakest configuration (30.9) to the headline (55.5).

**Three things this does and does not license.**

- It **does** establish the order of magnitude on navhard: selection is a first-order effect there, architecture a second-order one. The [navhard cohort table](../concepts/navhard-ood-evaluation.md#scorer-cohort) shows the same split across fourteen methods — every entry above 42 scores, every entry below 35 does not.
- It **does not** transfer directly to NAVSIM-v1, where the margins are 1-2 PDMS and the scored/unscored gap has never been measured on one checkpoint. *(Lint 2026-09-30: two partial measurements now exist on navtest. [[sources/physwam.md]] reports first sample, a medoid-of-8 selector and the oracle on one checkpoint (90.3 → 90.4 EPDMS for the selector), and [[sources/mm-future.md]] prices a learned scorer at +8.2 PDMS, though between a one-trajectory and a 32-trajectory model.)* The +20.9 is large partly because navhard multiplies two stage scores, so a selector that improves both stages compounds.
- It **does** come with a visible cost. EC falls 76.9 -> 66.2 on Stage 1 and 75.9 -> 45.6 on Stage 2: the selected proposals are safer, more rule-compliant, and less comfortable. EPDMS's multiplicative safety penalties make that trade profitable under *this* metric. A deployment objective that weighted comfort differently would score the same two checkpoints in the opposite order.

**The generalization worth carrying**: a published planning number is a *pipeline* number, and the selector is usually the larger term. Papers that report only the scored configuration — which is most of them — are reporting the pipeline while describing the generator.

## The Tie Problem, Finally Named {#tie-problem}

Every method in the table above trains a scorer to order candidates, and every one of them reports oracle ceilings and selection accuracies that cluster suspiciously tightly. [[sources/adaptive-wam.md]] gives the reason, measured on a fixed offline pool of **12,146 scenes**:

> More than **95% of scenes contain candidate groups that are jointly perfect, jointly zero, or tied at the top.**

If that is right — and it follows directly from PDMS being a product of near-saturated multiplicative terms — then **a rank loss is fitting noise on 95% of the training signal**, and rank correlation is close to meaningless as a scorer diagnostic. It also explains why oracle Best-of-N ceilings saturate so fast (see [[concepts/best-of-n.md]]): with most groups tied at the top, the oracle has little left to pick out.

**Adaptive-WAM's response is a design change worth copying.** Its verifier predicts the six evaluator components (NC, DAC, DDC, TTC, EP, Comf) with **equal-weight soft-label BCE on un-binarized targets, and no rank loss at all**, then composes them through the PDMS formula. It is explicitly framed as an *exit-quality verifier* rather than a total-order ranker. Evaluation follows the same logic — instead of Spearman correlation, it reports tie-aware selection rates and consequential-failure counts:

| Diagnostic (12,146 scenes) | Rate |
|---|---:|
| Exact top-score selection | 91.2% |
| Selection within 5 points of the true top | 94.4% |
| Failure with ≥ 20-point gap | 0.57% |
| Failure with ≥ 50-point gap | **0.42%** (51 scenes) |

The last row is the one that matters for deployment, and the paper says so: *"passing the quality threshold is not a formal safety certificate."*

**This retro-explains several entries above.** [[sources/da-wam.md]]'s scorer is deliberately built to avoid pooling futures into a proposal-invariant vector — a symptom of exactly this problem, since a pooled representation cannot break ties. [[sources/drivesuprim.md]]'s coarse-to-fine filtering works precisely because it discards the jointly-zero mass before scoring, leaving a harder but more informative 256. And [[sources/hybriddriveVLA.md]]'s finding that representation-only gating fails is consistent with a signal where most of the ordering is unlearnable.

**One caution about the 64-proposal result.** Adaptive-WAM's auxiliary 92.6 PDMS model belongs in this table rather than on the single-trajectory ladder: fixed block-22 exit, 64 proposals, no adaptive routing, and **CLOVER-derived pseudo-expert targets scored with the true NAVSIM evaluator using training-time map and future occupancy** — the same privileged-supervision caveat that applies to Hydra-MDP distillation, Auto-JEPA, and DA-WAM. Against DriveSuprim's 93.5 and CLEAR's 93.7 it does not lead.

**And one measurement that complicates encoder comparisons made through selection.** In the same paper, Wan intermediate features beat ViT-Large by **1.74 PDMS** for a single trajectory but by only **0.28** with 64 proposals. Multi-proposal scoring compensates for a weaker representation, so a selection-based leaderboard is a poor instrument for comparing encoders.

---

## Origin of the Pattern: Hydra-MDP++ {#origin}

[[sources/hydra-mdp-pp.md]] states the recipe that most entries on this page reuse:
1. **Sample** a fixed vocabulary (k-means over 700K nuPlan trajectories).
2. **Run the PDM simulator offline** on every vocabulary trajectory in every training scene, with ground-truth perception.
3. **Distil** each sub-score into a BCE head.
4. **Select** by a weighted log-cost whose weights are grid-searched.

Its ablation already shows the pattern this page later documents at scale. **The selection rule is the biggest lever** (weighted cost +1.5 PDMS), bigger than the temporal module (+0.1). Auxiliary perception supervision *hurts* (−0.5). The criticism attached to the scorer cohort (fitting the benchmark's own metric with privileged supervision) applies from the first paper.

