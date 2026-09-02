---
title: Counterfactual Prediction for Driving World Models
type: concept
sources: [raw/papers/How Can Driving World Models Do Counterfactual Prediction_.md, raw/papers/DA-WAM_ Decision-Aligned Future Latents for Driving World Models.md, raw/papers/Auto-JEPA_ A Latent World Model of Continuous Intent for End-to-End Autonomous Driving.md]
related: [sources/da-wam.md, sources/auto-jepa.md, sources/driving-wm-counterfactuals.md, concepts/world-model-for-ad.md, concepts/vlm-domain-adaptation.md, concepts/bench2drive.md, concepts/hugsim-benchmark.md, concepts/nuscenes-waymo-evals.md, concepts/best-of-n.md, sources/simwam.md, sources/drivelaw.md, sources/dreameraD.md, sources/vega.md, sources/policy-world-model.md]
created: 2026-08-24
updated: 2026-09-02
confidence: high
---

# Counterfactual Prediction for Driving World Models

"Counterfactual" is one of the most frequently claimed and least precisely used capabilities in the driving world-model literature. This page separates the senses in which the word is used, states what the causal definition actually requires, and records the first quantitative evidence in this wiki about whether driving world models meet it.

The primary source is [[sources/driving-wm-counterfactuals.md]] (Purdue + Bosch CAI, arXiv 2608.11601).

---

## Pearl's Ladder, Applied to Driving

| Rung | Question | Driving instance | What it conditions on |
|---|---|---|---|
| **1. Association** | What do I see given what I've seen? | Next-frame prediction from history | $H$ |
| **2. Intervention** | What happens *in general* if I do $a'$? | Action-conditioned video generation for a candidate manoeuvre | $H$, $\mathrm{do}(a')$ |
| **3. Counterfactual** | What *would have* happened in **this** episode had I done $a'$ instead? | Replaying a recorded incident under early braking | $H$, $F^{+}$, $a'$ |

The hierarchy is strict: higher rungs are in general not identifiable from lower-rung information alone. Rung 3 differs from rung 2 by one thing — it additionally conditions on the **factual outcome** $F^{+}$, the continuation that was actually observed in the episode being asked about.

Pearl's recipe for rung 3 is three steps:

$$
\underbrace{w\sim p(w\mid H,F^{+})}_{\text{abduction}}\;\longrightarrow\;\underbrace{a^{\prime}}_{\text{action}}\;\longrightarrow\;\underbrace{Y_{a^{\prime}}=G(w,a^{\prime})}_{\text{prediction}}
$$

**Abduction is the step the field skips.** A driving world model queried with $(H, a')$ performs action and prediction but never infers the realized world from what was observed after the history.

---

## Four Different Things Called "Counterfactual" in AD

Keeping these apart resolves most apparent disagreements between papers:

| Sense | Example | Actual rung | Is the label right? |
|---|---|---|---|
| **A. Action-conditioned generation** — feed an alternative/abnormal trajectory, generate video | Vista, Drive-WM, Genie 3 promptable events | Rung 2 at best | **No.** This is intervention, not counterfactual |
| **B. Retrospective log replay** — re-simulate a recorded drive under a different route | Waymo World Model blog | Rung 3 *if* the recorded episode's state is preserved; rung 2 if only the history is | Depends on whether the realized state is carried over |
| **C. Counterfactual VQA** — ask a VLM, in language, what would happen under a hypothetical | OmniDrive counterfactual planning (see [[concepts/vlm-domain-adaptation.md]]: frozen VLM 18.20 → fine-tuned 67.80) | Language-space reasoning, not a prediction task | Different object entirely; don't compare scores across senses |
| **D. Candidate-manoeuvre rollout for planning** — imagine futures for several proposals and score them | [[sources/dreameraD.md]] latent rollouts + reward model, [[sources/da-wam.md]] per-candidate future latents, world-model-as-scorer designs | **Rung 2, correctly** | The label is often loose but the *computation* is the right one for planning |

[[sources/da-wam.md]] is the wiki's purest instance of sense D and shows how entrenched the loose label is: it describes its per-candidate predicted latents as "counterfactual latent futures" and "candidate-specific counterfactual evidence" throughout, and titles a section "Action-Conditioned Counterfactual World Modeling." There is no abduction step anywhere — the alternative action is specified by the candidate set rather than observed, and nothing conditions on a factual continuation, which is the strict rung-2 situation. Its *engineering* is sound and appropriate; the terminology is exactly what this page exists to disambiguate. Worth noting in DA-WAM's favour that it is unusually careful about the underlying data problem even while using the wrong word: it refuses to apply the observed future as a target for unexecuted candidates, restricting dense supervision to the expert-matched one, which is precisely the recognition that the other 31 outcomes were never realized.

Sense D deserves emphasis because it is where most of this wiki's world-model planners live. At decision time there **is no factual continuation** — the future has not happened yet. Rung 2 is therefore the correct and only available target for a planner. The critique below applies to systems that claim rung 3 for *already recorded* episodes; it does not say action-conditioned generation is the wrong tool for planning.

---

## The Formal Gap

Let $Y$ be the video following the history under a given action, $Y_{a'}$ its value under $a'$, $H$ the shared history, and $F^{+}$ the factual continuation:

$$
\underbrace{p\big(Y_{a^{\prime}}\mid H,\,F^{+}\big)}_{\text{counterfactual}}\quad\text{vs.}\quad\underbrace{p\big(Y\mid H,\,a^{\prime}\big)}_{\text{direct action-conditioned prediction}}
$$

Expanded as mixtures over the world $w$:

$$
p(Y\mid H,a^{\prime})=\int p(Y\mid w,a^{\prime})\,p(w\mid H)\,dw,\qquad
p(Y_{a^{\prime}}\mid H,F^{+})=\int p(Y\mid w,a^{\prime})\,p(w\mid H,F^{+})\,dw
$$

Both integrate the **same mechanism**. They differ only in the posterior over the world. So:

- The gap is a **conditioning gap**, not a capacity, fidelity, or controllability gap. Scaling the generator does not close it.
- The gap is exactly zero when $F^{+}$ carries no information beyond $H$ — i.e. when nothing interesting happened. It is largest precisely for the events that matter: an agent emerging from an occlusion, a cut-in, anything first revealed after the history.
- Since $a'$ is *specified by the query* rather than observed, conditioning on it does not update the posterior over $w$ beyond $H$; that is what makes the direct prediction interventional rather than merely observational.

### What abduction can and cannot recover

Given a monocular camera and a viewpoint-only ego edit, the counterfactual view splits cleanly:

| Region | Posterior | Correct handler |
|---|---|---|
| Surfaces also observed in the factual video | Concentrated on observed surfaces; determined up to depth-recovery error | **Geometry** — reprojection / novel-view synthesis |
| Occluded regions, beyond-FOV regions | Genuinely uncertain; every consistent completion admissible | **A generative prior** — i.e. the world model |

This is the useful design principle to carry forward: a generative prior is entitled to invent only what the evidence does not determine. Any architecture claiming counterfactual capability should be able to say which of its output pixels are which.

---

## Measuring It

### The obstacle

Real driving records exactly one outcome per episode. The future under any alternative action is never recorded, so **no real dataset can supply counterfactual ground truth**. Simulation is not a convenience here; it is the only source of a matched reference.

### The three-arm construction

[[sources/driving-wm-counterfactuals.md]] builds each CARLA case as three replays of one placement, varying only the ego action and whether the event fires:

| Arm | Ego action | Event | Role |
|---|---|---|---|
| $F$ | executed | occurs | the factual log — the model's input |
| $P$ | target $a'$ | occurs | **counterfactual ground truth** |
| $U$ | target $a'$ | never triggered | **null reference** (what "missing the event" looks like) |

The null arm $U$ is the construction's clever part. Without it, a similarity-to-$P$ score cannot separate "recovered the event" from "produced a generically plausible video that happens to share road layout and ego viewpoint with $P$."

### Recovered fraction

With $\Delta(\hat{Y}) = s(\hat{Y},P) - s(\hat{Y},U)$ the preference for the counterfactual over the null (cosine similarity of frame embeddings, averaged over the prediction window):

$$\mathrm{Rec}(\hat{Y})=\frac{\Delta(\hat{Y})-\Delta(U)}{\Delta(P)-\Delta(U)}$$

so $\mathrm{Rec}(U)=0$, $\mathrm{Rec}(P)=1$, and $0.5$ means equally similar to both references. Reported under DINOv2 ViT-B/14 and CLIP ViT-L/14. Paired with **LPIPS against $P$** for perceptual fidelity — possible only because the matched ground truth exists, and a meaningfully stronger evaluation than the distributional FID/FVD used everywhere else in [[concepts/world-model-for-ad.md]].

**Two properties to remember before reusing this metric:**

1. **It is category-sensitive, not identity-sensitive.** Transporting evidence from a *different episode* with the same scenario type still scores 0.62–0.64 versus 0.66–0.67 for the matching episode, while LPIPS more than doubles (0.556 vs. 0.261). $\mathrm{Rec}$ largely asks "is there an event of this kind," not "is it this event." It must be reported alongside a spatial metric.
2. **Its denominator shrinks when $P$ and $U$ are similar.** For scenario types where the edit weakens the visible difference (the lead-brake control), small encoder fluctuations get magnified.

---

## What the Evidence Says

On 186 CARLA cases across three towns, with two publicly released backbones — **Vista** (latent diffusion, anchor frame + trajectory) and **DrivingWorld** (autoregressive VQ, frame history + pose + heading):

| | Direct prediction $B$ | Evidence-transport pipeline |
|---|---:|---:|
| Vista $\mathrm{Rec}_{\mathrm{D}}$ | 0.38 | **0.70** |
| Vista LPIPS ↓ | 0.423 | **0.169** |
| DrivingWorld $\mathrm{Rec}_{\mathrm{D}}$ | 0.31 | **0.67** |
| DrivingWorld LPIPS ↓ | 0.291 | **0.211** |

**Direct predictions land below 0.5** — closer to the event-free replay than to the matched counterfactual — in every scenario/encoder/backbone cell but one. The internal gradient is the strongest part of the evidence:

| Scenario type | What the model can infer from $H$ alone | Vista $B$ | DW $B$ |
|---|---|---:|---:|
| **side street** (event first revealed in $F^{+}$) | nothing | 0.29 | 0.25 |
| **lead cuts in** | little | 0.45 | 0.39 |
| **lead brake** (confounded control: already-visible lead looms under acceleration) | much | 0.50 | 0.37 |

Performance tracks *how much of the event is inferable from the shared history* — which is what the conditioning-gap analysis predicts and what a generic "these models are bad at CARLA renders" explanation does not.

**Evidence from the wrong time is worthless.** Transporting the final history frame $F_{14}$ — information the direct prediction already has — scores 0.35/0.36, indistinguishable from $B$'s 0.38/0.31. Pasting pixels does not help; pasting the right pixels from the right moment does.

**The world model contributes almost none of the recovery.** Geometric transport alone reaches 0.68/0.67; the frozen generator's Complete and Combine stages add ≤0.02 recovered fraction and buy fidelity instead (LPIPS 0.195 → 0.169). The demonstration is that *evidence* closes the gap, not that world models can be coaxed into abduction.

---

## Consequences for the Rest of the Wiki

**1. The standing claim that world models "enable counterfactual reasoning" is now qualified.** [[concepts/world-model-for-ad.md]] listed counterfactual reasoning — "if I turn left the future looks like X, if I go straight like Y" — among the reasons future prediction helps planning. That framing is rung 2, and it is a fine reason to train a world model; it is not counterfactual prediction, and the page now says so.

**2. It narrows the escape hatch in the test-time-imagination debate.** [[sources/simwam.md]]'s mask ablation and [[sources/drivelaw.md]]'s denoising-step sweep both found no planning benefit from conditioning on generated futures on NAVSIM, leaving "but imagination must matter for counterfactual manoeuvre evaluation" as the last untested defence of imagine-then-act designs. This paper tests counterfactual evaluation directly and finds the standard procedure does not deliver it either. The defence is not dead — sense D above (comparing *candidate* manoeuvres before acting) is rung 2 and remains untested by this benchmark — but the specific "world models are counterfactual simulators" version of it is now contradicted by measurement.

**3. It does not condemn action-conditioned generation for planning.** A planner has no $F^{+}$. Rung 2 is the right target at decision time, and every action-conditioned generator in this wiki is doing an appropriate computation for that purpose. What the paper removes is the *retrospective* claim: that the same machinery can answer "what would have happened in that recorded incident."

**5. Some world models opt out of the ladder entirely, and say so.** [[sources/auto-jepa.md]] predicts only the latent of the future ego trajectory, and its limitations section states plainly that the learned representation "does not provide the scene-level forecasts required by applications such as interactive simulation or counterfactual environment generation." This is worth recording as the honest boundary case: it is a *world model* in the sense of predicting the future, and it is on **no rung of this ladder for the environment**, because it never represents an environment state that could be intervened on. The trade is explicit — planning-relevant selectivity (masking dynamic agents changes its intent 2.97× more than equal-area random masks) without any queryable model of what those agents will do. Papers that want both must pay for both; Auto-JEPA is the demonstration that planning alone does not require the second.

**4. Retrospective analysis needs a different architecture, not a bigger one.** For incident analysis, safety auditing, and liability assessment — where the full log exists by definition — the missing component is an abduction path that ingests the observed continuation. No ingested method in this wiki has one.

---

## Open Questions

- **Can abduction be learned rather than hand-built?** The paper's transport stage is monocular depth + splatting. A world model conditioned on both $H$ and $F^{+}$ (i.e. trained for the retrospective task) has never been tried in this wiki. Would it beat geometry, or only match it?
- **What happens under reactive agents?** Transported evidence *preserves* behaviour the counterfactual action would have changed — a pedestrian who would have stopped keeps walking. Beyond ~1 s the method's central assumption breaks, and it fails confidently rather than by omission. Detecting these cases (the paper suggests posterior predictive checks on the abduced world) is unsolved.
- **Is the failure causal or domain shift?** Both backbones are real-world-trained and evaluated on CARLA renders. The scenario-type gradient argues against pure domain shift, but a world model trained in the benchmark's render domain would settle it.
- **Do the industrial claims hold?** Waymo's world model and Genie 3 make the strongest counterfactual claims and are the least testable. Drive-WM, whose claim is quoted directly, is not evaluated.
- **Is there a decision-time analogue?** All of this is retrospective. Whether anything from the abduction framing helps when the outcome is still unobserved is genuinely open — and it is the version that would matter for planning.
- **Does counterfactual fidelity predict anything downstream?** No paper links $\mathrm{Rec}$ (or any counterfactual metric) to planning quality, RL reward accuracy, or scenario-generation usefulness. Until it does, this remains an evaluation of a claimed capability rather than of a useful one.
