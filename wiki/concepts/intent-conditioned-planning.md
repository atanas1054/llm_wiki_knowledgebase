---
title: Intent-Conditioned Trajectory Planning
type: concept
sources: [raw/papers/DriveFuture_ Future-Aware Latent World Models for Autonomous Driving.md, raw/papers/Auto-JEPA_ A Latent World Model of Continuous Intent for End-to-End Autonomous Driving.md, raw/papers/Driving Intents Amplify Planning-Oriented Reinforcement Learning.md, raw/papers/Fine-tuning is Not Enough_ A Parallel Framework for Collaborative Imitation and Reinforcement Learning in End-to-end Autonomous Driving.md, raw/papers/SGDrive_ Scene-to-Goal Hierarchical World Cognition for Autonomous Driving.md]
related: [sources/drivefuture.md, concepts/world-model-for-ad.md, sources/auto-jepa.md, sources/dial.md, sources/pair-drive.md, sources/sgdrive.md, concepts/rl-for-ad.md, concepts/gspo-vs-grpo.md, concepts/best-of-n.md, concepts/diffusion-planner.md, concepts/parallel-il-rl.md, concepts/nuscenes-waymo-evals.md, concepts/perception-for-planning.md]
created: 2026-06-23
updated: 2026-09-11
confidence: high
---

# Intent-Conditioned Trajectory Planning

Intent-conditioned planning introduces a discrete semantic variable—such as cruise, turn, lane change, accelerate, or decelerate—between scene understanding and continuous trajectory generation. The intent is not necessarily the final driving output. It can act as a control variable that forces a generative planner to expose multiple maneuver basins for the same scene.

## Why Intent Helps

One logged scene usually supplies one demonstrated trajectory even when several actions were feasible. Continuous diffusion or flow policies trained on that single target may respond to different noise seeds with geometrically similar trajectories. This is behavioral mode collapse: the network is stochastic in coordinates but unimodal in maneuver semantics.

Intent conditioning supplies an explicit axis along which proposals can differ. It changes the learning question from “reproduce this path with noise” to “produce a plausible path under this maneuver hypothesis.”

## Four Uses in the Wiki

| Method | Intent mechanism | Training role | Inference role |
| --- | --- | --- | --- |
| [[sources/pair-drive.md]] | Learned intention tokens in a recurrent residual tree | Structure GRPO candidates around a human reference | Expand alternatives around an IL trajectory; RWM selects |
| [[sources/dial.md]] | Eight rule-derived labels with classifier-free guidance | Expand SFT support and balance every GRPO group across intents | Intent classifier selects one mode; conditioned flow generates |
| [[sources/sgdrive.md]] | Continuous goal pose ~4 s ahead, predicted by an MLP head on a dedicated ⟨world⟩ subquery | Auxiliary $L_1$ supervision that shapes the VLM representation | Goal subquery hidden state conditions the DiT; never decoded |
| [[sources/auto-jepa.md]] | Continuous 8×1024 latent encoding the *entire* 4 s future trajectory, predicted by a JEPA predictor | The only training objective — aligned with a frozen trajectory encoder's output | The retrieval key into a memory of 110,335 recorded trajectories |
| [[sources/drivefuture.md]] | Tokenized trajectory (8 pose-difference tokens) fed to a **world model**, not to the planner | Drawn from expert / kinematic rollout / null token at 0.4/0.4/0.2 | Two surrogates on a phase schedule: a constant-acceleration rollout, then a Tweedie estimate of the proposal being denoised |

PaIR-Drive treats intent as tree-branch structure in a separate RL refiner. DIAL treats intent as a condition inside one continuous generative policy and explicitly preserves all modes during preference fine-tuning.

## Continuous Goal as Intent: SGDrive

SGDrive is the odd one out here, and worth keeping in view because it shows the intent axis is not inherently discrete. Its "intent" is a **single continuous ego pose roughly 4 seconds into the future**, supervised by an $L_1$ loss and decoded from its own subquery. The paper's framing is that this "disentangles high-level decision-making from low-level trajectory planning" — the goal says *where* the maneuver ends, the diffusion planner works out *how* to get there.

Three consequences follow from continuity:

- **No ontology problem.** The design questions this page raises for DIAL — class frequency, coverage, boundary noise, label derivation — simply do not arise. A pose is derived unambiguously from the demonstration.
- **No multimodality either.** A regressed goal is a point estimate, so it cannot expand proposal support the way intent-CFG does. Where DIAL uses intent to *create* distinct maneuver basins, SGDrive uses the goal to *commit* to one. It offers nothing for Best-of-N or for GRPO group diversity.
- **The measured effect is efficiency, not safety.** In SGDrive's Table 4 the goal subquery moves PDMS 86.3 → 87.0, and the gain is concentrated in Ego Progress (80.4 → 81.2) — the largest single jump in that ablation. The paper's motivation matches: without goal prediction the ego "may exhibit incomplete or suboptimal maneuvers, such as covering only part of the planned path."

This is a useful contrast for the ontology discussion below. A continuous goal buys progress and sidesteps every labeling difficulty, at the cost of the mode-spanning property that makes discrete intent valuable for preference optimization. The two uses are complementary rather than competing, and no ingested paper has combined them.

## Full-Trajectory Latent as Intent: Auto-JEPA

[[sources/auto-jepa.md]] uses the word "intent" for something further along the same continuous axis, and the distinction is worth pinning down because the shared vocabulary hides three different objects:

| | DIAL / PaIR-Drive | SGDrive | Auto-JEPA |
|---|---|---|---|
| Intent is | A discrete class label | A continuous terminal pose | A continuous 8×1024 latent of the whole 4 s path |
| Cardinality | 8 classes | $\mathbf{R}^2$ (one point) | $\mathbf{R}^{8\times1024}$ (eight time-aligned tokens) |
| Where it comes from | Rule-derived labels | GT waypoint at $t{+}4\,\mathrm{s}$ | Frozen trajectory autoencoder |
| What it under-determines | The path within a maneuver class | Everything except the endpoint | Nothing — it *is* the trajectory, in latent form |
| Multimodality | Created by conditioning on each class | None (point estimate) | None (one realization) |

Auto-JEPA's intent tokens are explicitly *not* maneuver classes — the paper states the eight tokens "jointly describe one continuous future realization rather than eight maneuver classes." So it sits at the extreme of the continuous branch: the richest intent representation on this page and, for exactly that reason, the one with the least mode-spanning capacity. A latent that specifies the whole path cannot serve as a conditioning variable that opens alternative maneuver basins, because there are no alternatives left to open.

**What it gets in exchange is a shared retrieval space.** SGDrive's goal pose conditions a generator; Auto-JEPA's latent is compared directly against encodings of real trajectories under cosine similarity, which only works because target and memory pass through the same frozen encoder. That is a genuine payoff no discrete ontology can offer — and it depends on the trajectory encoder being frozen *before* the intent predictor is trained, which is the paper's key sequencing decision.

**Where this leaves the multimodality argument.** Auto-JEPA recovers alternatives downstream rather than through intent: the top-300 retrieved trajectories *are* the proposal set, and they are diverse because the memory is. Note the difference from intent-CFG, though — retrieved neighbors are diverse in *geometry* but all maneuver-compatible with a single predicted intent, so they cannot span the "should I yield or go?" split that DIAL's eight classes are designed to expose. The distinction between spatial and semantic diversity in [Diversity Is Multidimensional](#diversity-is-multidimensional) below applies directly: Auto-JEPA has the first kind and, by construction, not the second. No ablation measures whether that costs anything.

## Intent as the World Model's Input: The Circular Dependency (DriveFuture) {#circular-intent}

Every use above puts the intent between the scene and the **planner**. [[sources/drivefuture.md]] puts it between the scene and the **world model**, and that one change creates a problem none of the others has.

Its future latent is *action-conditional* by construction — $\hat{\mathbf{Z}}_{t+T}=f_\psi(\mathbf{Z}_t;\mathbf{E}_	au)$ describes the scene at horizon $T$ *under the hypothesis that the ego enacts $oldsymbol{	au}$*. That is the right thing for a world model to be. But the planner is conditioned on $\hat{\mathbf{Z}}_{t+T}$, so at inference:

$$\hat{\mathbf{Z}}_{t+T}=f_\psi(\mathbf{Z}_t;\mathbf{E}_	au),\qquad oldsymbol{	au}=\operatorname{Decode}(\mathbf{Z}_t,\hat{\mathbf{Z}}_{t+T})$$

**Intent is required to predict the future, and the future is required to produce the intent.** Training hides this completely — the expert trajectory is available — which is why the problem only shows up as a deployment question.

**Three ways the wiki's papers avoid or solve it:**

| Design | How the circularity is avoided |
|---|---|
| DIAL, PaIR-Drive | The intent is *predicted by a classifier* before generation, so nothing downstream is needed to produce it |
| SGDrive, Auto-JEPA | The intent is predicted directly from the scene by a dedicated head; no world model consumes it |
| [[sources/da-wam.md]] | Candidates are generated **first**, then each conditions its own future — the future is used for *scoring*, never for generation |
| **DriveFuture** | **Two surrogates on a noise-phase schedule**: a constant-acceleration rollout while the sample is too noisy to trust, then a Tweedie estimate of the sample itself |

DA-WAM's answer is the structurally simplest — move the action-conditional future to the scoring stage, where the action already exists — and it is why that design has no analogue of PFG. DriveFuture's answer keeps the future inside the generation loop and pays for it with a recursive estimate whose error it has to schedule around. The mechanics are on [[concepts/diffusion-planner.md]].

**What this adds to the page's ontology discussion.** DriveFuture's "intent" is a full 8-pose trajectory, so by the taxonomy above it sits with Auto-JEPA at the fully-specified end and has **no mode-spanning capacity at all** — conditioning on it cannot open alternative maneuver basins. But it is used for something no other entry here uses intent for: as the *action argument of a dynamics model*, which is the sense the term carries in model-based RL rather than in behaviour prediction. **The wiki now has three unrelated objects sharing the word "intent"** — a maneuver class, a goal or path specification, and a dynamics-model action — and only the first supports the multimodality argument this page is built on.

## Intent-CFG

Classifier-free guidance trains one generator with both conditional and unconditional intent inputs. At sampling time, the conditional and unconditional flow fields are combined to strengthen the selected maneuver mode.

This is different from using CFG only to improve fidelity. In DIAL, CFG's purpose is proposal-support expansion: different intent labels should reach different rater-meaningful behaviors.

## Intent-Balanced GRPO

For $C$ intents and $S$ noise samples per intent, the group size is $K=CS$. DIAL uses $C=8$, $S=2$, $K=16$. Advantages are normalized across the pooled group.

The comparison to $C=1$, $S=16$ isolates the key effect: total sample count is fixed, but semantic coverage changes. DIAL reaches held-out RFS 8.211 versus 7.992 for the strongest single-intent alternative. Therefore, candidate count alone is not the operative variable; the group must span reward-relevant modes.

## Diversity Is Multidimensional

Trajectory separation and preference diversity are not interchangeable:

- **Spatial diversity:** pairwise ADE between proposals.
- **Reward diversity:** standard deviation of preference scores.
- **Diversity dividend:** Best-of-N minus Best-of-1 score.
- **Deployment quality:** score after selecting/predicting one intent.

DIAL's top-rater single-intent variant has greater spatial separation than multi-intent DIAL but worse reward spread and lower final RFS. A useful proposal set must be both different and meaningfully rankable.

## Intent Ontology Design

A practical ontology should be:

- small enough that every class is represented frequently;
- broad enough to cover genuinely different maneuver modes;
- derivable consistently from demonstrations or labels;
- predictable from scene context at inference;
- compositional enough to avoid forcing ambiguous behavior into one class.

DIAL's eight classes satisfy simplicity but not full coverage. For example, “yield while changing lanes,” “creep,” and obstacle-specific avoidance are not explicit classes. Hard rule thresholds can also make semantically similar trajectories receive different labels.

## Evaluation Requirements

Report all of the following:

- intent-label derivation and class frequency;
- intent-classifier accuracy/confusion;
- per-intent feasibility and reward;
- fixed-budget comparison against extra noise samples;
- spatial and reward diversity;
- Best-of-N support ceiling separately from deployable single-intent performance;
- latency of intent prediction and conditional generation;
- closed-loop safety and interaction metrics, not preference score alone;
- sensitivity to ontology size and label noise.

## Core Insight

An optimizer can only amplify distinctions present in its sample group. Intent conditioning is valuable when it creates semantically distinct, feasible alternatives that the reward can rank; arbitrary labels or spatial scatter alone do not solve the problem.
