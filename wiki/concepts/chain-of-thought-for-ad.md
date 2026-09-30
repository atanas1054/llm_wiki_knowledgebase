---
title: Chain-of-Thought Reasoning for Autonomous Driving
type: concept
sources: ["raw/papers/[-0.5mm] GRAVA GRAVA_ Grounded Reasoning-to-Action Representation and Learning for Autonomous Driving.md", raw/papers/CoWorld-VLA_ Thinking in a Multi-Expert World Model for Autonomous Driving.md, raw/papers/WCog-VLA_ A Dual-Level World-Cognitive Vision-Language-Action Model for End-to-End Autonomous Driving.md, raw/papers/ReCogDrive_ A Reinforced Cognitive Framework for End-to-End Autonomous Driving.md, raw/papers/UniUGP_ Unifying Understanding, Generation, and Planing For End-to-end Autonomous Driving.md, raw/papers/AutoVLA_ A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning.md, raw/papers/AdaThinkDrive_ Adaptive Thinking via Reinforcement Learning for Autonomous Driving.md, raw/papers/AutoDrive-R²_ Incentivizing Reasoning and Self-Reflection Capacity for VLA Model in Autonomous Driving.md, raw/papers/Alpamayo-R1_ Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail.md, raw/papers/FutureSightDrive_ Thinking Visually with Spatio-Temporal CoT for Autonomous Driving.md, raw/papers/HERMES_ A Holistic End-to-End Risk-Aware Multimodal Embodied System with Vision–Language Models for Long-Tail Autonomous Driving.md, raw/papers/NoRD_ A Data-Efficient Vision-Language-Action Model that Drives without Reasoning.md, raw/papers/Reasoning-VLA_ A Fast and General Vision-Language-Action Reasoning Model for Autonomous Driving.md, raw/papers/Unleashing VLA Potentials in Autonomous Driving via Explicit Learning from Failures.md, raw/papers/DynVLA_ Learning World Dynamics for Action Reasoning in Autonomous Driving.md, raw/papers/SpanVLA_ Efficient Action Bridging and Learning from Negative-Recovery Samples for Vision-Language-Action Model.md, raw/papers/OneVL_ One-Step Latent Reasoning and Planning with Vision-Language Explanation.md, raw/papers/DeepSight_ Long-Horizon World Modeling via Latent States Prediction for End-to-End Autonomous Driving.md, raw/papers/Understanding R1-Zero-Like Training_ A Critical Perspective.md, raw/papers/All Roads Lead to Rome_ Incentivizing Divergent Thinking in Vision-Language Models.md]
related: [concepts/reasoning-faithfulness.md, sources/grava.md, sources/qwen-drive-1.0.md, sources/coworld-vla.md, sources/wcog-vla.md, sources/recogdrive.md, sources/uniugp.md, sources/autovla.md, sources/adathinkdrive.md, sources/autodrive-r2.md, sources/alpamayo-r1.md, sources/futuresightdrive.md, sources/hermes.md, sources/nord.md, sources/reasoning-vla.md, sources/elf-vla.md, sources/dynvla.md, sources/spanvla.md, sources/onevl.md, sources/deepsight.md, sources/understanding-r1-zero-like-training.md, sources/all-roads-lead-to-rome.md, concepts/vlm-domain-adaptation.md, concepts/rl-for-ad.md, concepts/world-model-for-ad.md, concepts/r1-zero-like-training.md, concepts/divergent-thinking-in-vlms.md, research-directions.md]
created: 2026-04-15
updated: 2026-09-30
confidence: high
---

## Why CoT for Autonomous Driving

Chain-of-Thought prompting forces a VLM to produce intermediate reasoning steps before outputting an action. In the AD context, this serves three purposes:

1. **Grounding**: scene description → key object identification → intention inference → decision forces the model to explicitly account for causally relevant objects before planning, reducing shortcut learning
2. **Interpretability**: the reasoning trace is human-readable, making the model's decision auditable
3. **Sample efficiency**: high-quality reasoning annotations can teach a model *why* to take an action, not just *what* action to take — potentially reducing the amount of trajectory-labeled data needed

The critical question addressed by the literature: **is CoT actually necessary**, or does the reasoning trace add cost without commensurate benefit? NoRD provides the strongest negative evidence; AdaThinkDrive provides the most nuanced answer (CoT helps in complex scenes but hurts in simple ones).

---

## CoT Content: What Gets Reasoned About

Most text-based CoT pipelines in the wiki use a 3–4 stage structure, with variations:

### Standard 4-Stage Text CoT (AutoVLA, UniUGP, ReCogDrive)

| Stage | Content |
|-------|---------|
| 1. Scene description | Time, weather, road type, traffic density |
| 2. Object identification | Traffic signals, dynamic agents, rare/anomalous objects, bounding boxes |
| 3. Intention inference | How detected objects will affect ego's future path; predicted agent trajectories |
| 4. Action decision | Chosen maneuver, grounded in stages 1–3 |

**UniUGP** adds trajectory-grounding: CoT is constructed by prompting a frontier VLM with the *known future trajectory* as context, ensuring the reasoning is causally consistent with the planned path. Without trajectory grounding, the model may generate plausible-sounding but action-inconsistent reasoning.

### Driving-Specific Enrichments (AdaThinkDrive)

AdaThinkDrive's Think mode adds structured spatial annotation:
- **Road boundary estimation** from HD map (topology, critical boundary features along ego's future path)
- **CIPO agent classification** (Closest In-Path Object-1 in ego lane; CIPO-2 likely to merge; Motion Interaction predicted to cross ego path)
- Traffic light states and weather (auto-annotated by Qwen2.5-VL-72B)

This structured CoT is more spatial and agent-focused than scene-description-first pipelines.

### Self-Reflection CoT (AutoDrive-R²)

AutoDrive-R² adds a **backward-check** stage after the action decision: the model re-reads its own trajectory prediction and verifies that the reasoning chain is consistent with the output. If the action appears inconsistent, the model revises. This 4-step + self-reflection structure achieves strong zero-shot generalization (0.19m avg L2 on nuScenes, 0.20m on Waymo) from only 6K training samples — the backward-check provides free data augmentation by catching self-contradictions at training time.

[[sources/understanding-r1-zero-like-training.md]] adds an important caution from math R1-Zero analysis: self-reflection-like language can appear in base models before RL, including DeepSeek-V3-Base, and does not necessarily imply higher inference accuracy. For AD, backward-check or self-reflection CoT should be judged by downstream trajectory/safety metrics and consistency audits, not by reflective phrasing alone.

[[sources/all-roads-lead-to-rome.md]] adds a second caution: RL can make a VLM better at one reasoning path while reducing its ability to sample alternative paths. For AD CoT, this separates **sequential depth** from **parallel breadth**: longer or more polished CoT is not enough if all samples follow the same flawed strategy.

### Failure-Diagnostic CoT (ELF-VLA)

**ELF-VLA** ([[sources/elf-vla.md]]) uses CoT as a repair target rather than only a planning trace. When a rollout scores below threshold $s=0.8$, Qwen3-VL-32B receives the wrong trajectory, GT trajectory, NAVSIM metric scores, and task context, then produces structured feedback covering meta-action analysis, think-process analysis, safety failure, efficiency failure, and actionable lateral/longitudinal correction.

The student is SFT-trained to consume feedback inputs before RL, then during GRPO it re-rolls from teacher feedback and injects corrected refinements into the policy update. This makes CoT supervision active in the RL loop: the teacher can identify hallucinated obstacle positions or wrong high-level maneuvers and force the policy to practice the corrected reasoning path.

### Visual CoT (FutureSightDrive / FSDrive)

FSDrive replaces text reasoning with a **generated future video frame** as the CoT intermediate. The model autoregressively generates a unified future image (lane dividers + 3D agent bounding boxes overlaid) before planning, using the visual scene prediction as its reasoning step.

**Contrast with text CoT**: text CoT encodes reasoning as natural language (interpretable, compact); visual CoT encodes reasoning as a predicted image (spatially grounded, but expensive to generate and not human-readable as a reasoning trace). FSDrive's visual CoT primarily reduces *collision rate* (31% improvement) rather than L2 accuracy — the spatial structure of the image carries information that text descriptions lose.

See [[concepts/world-model-for-ad.md]] Pattern 5 for the full treatment.

### Dynamics CoT (DynVLA)

**DynVLA** ([[sources/dynvla.md]]) introduces a third CoT substrate: compact **world dynamics tokens**. The model generates a bounded sequence:

$$[\langle BOD\rangle,\mathcal{D}_{t:t+K-1},\langle EOD\rangle,\langle BOA\rangle,\mathcal{A}_{t:t+N-1},\langle EOA\rangle]$$

where $\mathcal{D}$ contains VQ-coded ego-centric and environment-centric dynamics extracted by a Dynamics Tokenizer. The default setting uses 16 dynamics tokens over a 2s horizon, with 4 ego and 4 environment tokens per step.

This is a middle ground between text and image CoT: it preserves explicit "think-then-act" generation, but the thought is a compact transition representation rather than prose or pixels. Table 4 in DynVLA shows Dynamics CoT is both faster and stronger than alternatives in its controlled setting: 0.37s / 87.2 PDMS vs. scene-description CoT 3.04s / 85.3 PDMS and future-image CoT 2.29s / 86.3 PDMS.

### Latent Vision-Language CoT (OneVL)

**OneVL** ([[sources/onevl.md]]) introduces a fourth CoT substrate: compact latent tokens whose meaning is forced by training-only decoders. It uses visual latent tokens supervised to predict future-frame visual tokens and language latent tokens supervised to reconstruct CoT text. At inference, both decoders are removed and the latent tokens are prefilled into the context, avoiding sequential CoT decoding.

The key distinction from prior latent-CoT methods is the compression target. COCONUT, CODI, and SIM-CoT compress language-level reasoning; OneVL also compresses short-horizon visual dynamics. This makes it closer to a training-time world model than a pure text-compression method. In its controlled Qwen3-VL-4B setup, OneVL reaches 88.84 PDMS on NAVSIM vs. 88.29 for explicit AR CoT+Answer and 87.47 for AR Answer, while matching answer-only latency.

### Multi-Expert Latent CoT (CoWorld-VLA)

**CoWorld-VLA** ([[sources/coworld-vla.md]]) introduces a fifth substrate, and the novelty is not the latency saving — it is that the latent is **typed**. Four groups of learnable expert tokens are appended to the VLM input sequence, and each group's hidden state is pulled toward a different frozen teacher:

| Token | Supervised by | What it is meant to mean |
|---|---|---|
| Semantic interaction | Frozen V-JEPA on the **future** frame (pooled) | Interaction intent, object-level context |
| Geometric structure | Frozen VGGT on the **future** frame (pooled) | Road layout, spatial constraints |
| Dynamic evolution | A Wan2.2-5B video DiT that **this token conditions** | Future motion trend, temporal consistency |
| Ego trajectory | GT waypoints via an MLP head | Behavioural goal |

Every prior latent-CoT method here compresses *one* thing: OneVL compresses CoT text plus short-horizon visual dynamics, DynVLA compresses ego and environment transitions, FSDrive keeps a whole future image. CoWorld-VLA instead **assigns each latent a supervisor and lets them specialize**, then hands all four to a planner as separate conditioning streams.

**The complementarity is measured and it is real but modest.** Ego-trajectory token alone gives 83.7 PDMS; adding geometry +1.4, semantics +1.5, and the full four-token set reaches 88.7. No token is redundant. But the dynamic token — the world-model one — is never ablated alone, and the whole +5.0 shrinks to **+1.1 once a diffusion action expert is in place** (see [[concepts/foundation-backbones-for-ad.md]]). A typed latent CoT is worth a lot against a weak readout and much less against a strong planner.

**What it says about the substrate question.** The wiki's running comparison has been text-vs-visual-vs-latent, on a latency axis. This adds a second axis: **whether the latent's content is specified by an external supervisor or left to emerge.** OneVL and DynVLA specify with decoders and tokenizers built for the purpose; CoWorld-VLA specifies with off-the-shelf frozen foundation models, which is cheaper to assemble and ties the reasoning trace's semantics to whatever those teachers happen to encode.

---

## CoT Annotation Methods

Generating high-quality CoT training data is expensive. Three strategies are used in the wiki:

### 1. Frontier-VLM Annotation (AutoVLA, AdaThinkDrive, HERMES)

Prompt a large frozen VLM (Qwen2.5-VL-72B) with the scene + GT meta-action as a hint. The VLM generates the reasoning trace; the hint steers the reasoning toward the actual decision.

- AutoVLA: 88.8% annotation accuracy (human-verified on 3K samples); includes nuPlan (45.6K) and Waymo E2E (7.2K) CoT
- AdaThinkDrive: NAVSIM multi-turn CoT (CIPO classification + road boundary + traffic state)
- HERMES: offline annotation only — no CoT at inference time; reasoning baked into embeddings

**Limitation**: the VLM annotator is bounded by its own driving knowledge. Edge-case scenarios (construction zones, unusual maneuvers) may produce generic or incorrect reasoning traces.

### 2. GT-Trajectory-Grounded CoT (UniUGP, Alpamayo-R1)

Use the ground-truth future trajectory as a conditioning signal when generating CoT. The reasoning must be consistent with what the ego vehicle actually did.

- UniUGP: trajectory-grounded CoT constructed by prompting with future trajectory context; manually calibrated for physical consistency
- Alpamayo-R1: CoC (Chain-of-Thought Corpus, 700K samples) with hybrid labeling — combines rule-based labels (lane position, speed constraints) with VLM-annotated high-level reasoning

**Advantage**: GT grounding prevents hallucinated justifications that contradict the actual behavior. **Limitation**: only available for training; at inference, the GT trajectory is unknown.

### 3. GRPO-Optimized CoT (AutoDrive-R², Alpamayo-R1)

RL shapes not just the trajectory but the CoT quality:

- **AutoDrive-R²**: physics-based GRPO rewards — position accuracy, steering constraint satisfaction, velocity smoothness, temporal consistency. Gradients flow through the reasoning tokens, implicitly improving reasoning quality when it leads to better trajectories
- **Alpamayo-R1**: LRM-as-critic reward — a separate language reasoning model scores the generated CoT for logical coherence; combined with consistency reward (CoT must match final action) and safety reward

LRM-as-critic is the most principled approach: a dedicated critic evaluates whether the reasoning is sound, not just whether the trajectory satisfies physical constraints.

---

## When CoT Helps vs. Hurts

### AdaThinkDrive: The Case for Adaptive CoT

Controlled comparison on InternVL3-8B across 3 scene complexity levels:

| Scene Level | Non-Think PDMS | Think PDMS | Winner |
|-------------|---------------|-----------|--------|
| Level 1 (Simple) | 88.5 | Worse | Non-Think |
| Level 2 (Moderate) | — | Better | Think |
| Level 3 (Challenging) | 87.8 | **89.8** | Think |

CoT adds overhead (0.86s vs. 0.68s for non-think) with no benefit in simple scenes. AdaThinkDrive's adaptive policy achieves 90.3 PDMS — +1.4 over always-Think (88.9), +2.0 over always-Non-Think (88.3) — at a 14% latency savings vs. always-Think.

### AutoVLA: CoT Needs Scale

CoT data scaling analysis shows CoT underperforms action-only training at <50K samples. At ≥50K samples CoT surpasses action-only. On nuScenes (simple urban driving), action-only outperforms CoT throughout — CoT complexity is domain-appropriate only for structurally complex scenarios (intersections, multi-agent interactions).

### SpanVLA: Adaptive CoT with Continuous Action Expert

**SpanVLA** ([[sources/spanvla.md]]) follows AutoVLA's adaptive fast/slow reasoning idea but separates the final action decoder. The VLM generates compact text reasoning only until it emits an action-generation token; then a flow-matching action expert reads sparse KV-cache and produces the continuous trajectory. This keeps CoT available for complex scenes while avoiding long autoregressive waypoint decoding.

The mReasoning annotation pipeline uses Gemini-3-Pro to produce compact reasoning traces for 30K complex samples. It filters to causally relevant elements before selecting longitudinal and lateral actions, then uses human quality checks over 250 samples. During RFT, SpanVLA also penalizes excessive CoT length and rule-detected action-reasoning inconsistency.

### DeepSight: Adaptive CoT Gated by a World-Model State

**DeepSight** ([[sources/deepsight.md]]) follows AdaThinkDrive's "reason only when needed" philosophy, but the gating decision is made **after** the model has predicted its future world state $\mathbf{F}$ (five-frame DINOv3 BEV latents): $T_\text{cot}=M_\text{uni}(\dots\mid\mathbf{F})$. If the scene is complex the model emits structured reasoning; otherwise it outputs a placeholder token $T_\text{cot}^{\emptyset}$. Across 220 Bench2Drive routes the CoT fired in **<30% of frames**, adding only **+4.12%** average latency (Table 6).

Annotation uses a **Qwen3-VL-235B** pipeline producing ~1.3M structured labels, with a three-part format (infer current action from history → decide if complex decision-making is needed → summarize) and a filter that discards mismatches (judged simple but reasoned, or judged complex without reasoning). Only the summary reasoning of "complex" scenes is distilled into DeepSight.

**Key finding on CoT's ceiling** (Table 5): unlike world modeling, adaptive CoT alone is weak — CoT-only reaches 69.87 DS vs. world-model-only 84.52 DS; adding CoT on top of the world model adds just +1.71 DS / +5.45 SR. DeepSight explicitly frames this as evidence of the "inherent limitations of adaptive CoT alone in autonomous driving," where spatial foresight (the world model) does the heavy lifting and text reasoning helps mainly on long-tail cases (emergency vehicles, construction zones, traffic signs). This complements AutoVLA/AdaThinkDrive (CoT helps complex scenes) and NoRD (CoT is not the bottleneck) by locating the benefit specifically in the long tail.

### NoRD: CoT Is Not the Bottleneck

NoRD achieves 85.6 PDMS with zero reasoning annotations and only 80K training samples (vs. AutoVLA's 212K+ with CoT). The bottleneck was the RL optimizer, not the reasoning format. Dr. GRPO enables +11.68% improvement from the same reasoning-free base that standard GRPO could only improve by +0.67%. See [[sources/nord.md]] and [[concepts/rl-for-ad.md]].

**Key implication**: reasoning annotations provide sample efficiency for strong SFT initialization. Whether they provide an irreplaceable signal depends on whether the RL optimizer can recover that signal from weaker SFT bases. NoRD suggests that with the right RL optimizer (Dr. GRPO), a reasoning-free policy can approach CoT-trained policies with 60% less data.

---

## Game-Theoretic CoT (WCog-VLA)

[[sources/wcog-vla.md]] adds a CoT *content* type that none of the entries above cover: the reasoning is about **what other agents will do in response to the ego**, framed as a Stackelberg game with the ego as leader and surrounding agents as followers.

Four sequential steps, generated automatically by Qwen3-VL-Plus over NAVSIM:

1. **Scene description**
2. **Critical object analysis**
3. **Game-theoretic reasoning** — enumerate candidate ego actions and infer each follower's reaction ("if-what" imagination)
4. **Payoff evaluation** — score each hypothetical for safety and efficiency, then select

85k annotations. Steps 1–2 are the standard opening of the [4-stage text CoT](#standard-4-stage-text-cot-autovla-uniugp-recogdrive); steps 3–4 are new. The distinction from AdaThinkDrive's driving-specific enrichments is that this reasons over *counterfactual ego actions and their social consequences* rather than over scene attributes.

**The annotations are post-hoc rationalizations.** Ground-truth actions are supplied to the annotator as hints, so that the model "reconstruct[s] explicit causal chains linking observed scene contexts to final GT actions." The paper is candid, and the motivation (hallucination control) is sound, but the traces justify a known answer rather than deriving one. That is the same construction as [GT-trajectory-grounded CoT](#2-gt-trajectory-grounded-cot-uniugp-alpamayo-r1) and inherits the same caveat: it is supervision, not evidence of reasoning ability.

### The Price of Inference-Time Text CoT, Measured

WCog-VLA's Table 5 contains the sharpest efficiency datum this page has on textual CoT, on the same model and hardware:

| Path | PDMS | Inference time |
|---|---:|---:|
| VLM text output, **no reasoning** | 85.0 | **1.131 s** |
| VLM text output, **with Game-CoT reasoning** | 85.5 | **9.896 s** |
| ADDT diffusion head, 5 steps | **89.3** | 0.106 s |

**Generating the reasoning costs 8.8 seconds and buys 0.5 PDMS.** Meanwhile a 5-step diffusion action head, conditioned on the same VLM's hidden states, scores 3.8 points higher than the reasoning path at roughly 1/93rd the latency.

**And the deployed system discards the text path entirely.** The Game-CoT data is retained as *training* supervision — it is worth +0.8 PDMS alone and +1.1 on top of open driving VQA (its Table 6) — while the reasoning is never generated at inference.

This is a distinct position on this page's central question. The entries above ask *when* to reason (AdaThinkDrive, SpanVLA, DeepSight) or *whether* reasoning is needed at all (NoRD). WCog-VLA answers: **keep the CoT corpus, drop the CoT computation.** The reasoning shapes the representation during fine-tuning and is then thrown away — which makes CoT a data-curation technique rather than an inference-time mechanism, and sidesteps the adaptive-routing machinery entirely.

Two caveats before generalizing it. The 0.5-point measurement is on *this* model's text head, which may simply be a weak trajectory decoder — [[sources/nord.md]] argues text-token trajectory output is the bottleneck rather than the reasoning. And WCog-VLA never tests whether an SFT run *without* Game-CoT but with the same total token budget would do as well, so the +1.1 could be data volume rather than reasoning structure.

## Reasoning as an Optional Condition on a Separate Action Head (Qwen-Drive-1.0) {#optional-condition}

[[sources/qwen-drive-1.0.md]] places a Chain-of-Causation trace in a position no other entry on this page uses: the trace is a **conditioning variable $\mathbf{r}$ that may be $\varnothing$**, consumed by a flow-matching Planning Expert that reads the VLM's cached attention KV. Three consequences:

- **Reasoning is trained as an ablatable input, not a required prefix.** Only **685 K of 2.83 M planning samples (24.2 %)** carry an accepted trace; the remaining 75.8 % train with $\mathbf{r}=\varnothing$. The same checkpoint therefore has a with- and without-reasoning mode at inference by construction, with no dual-mode SFT ([[sources/autovla.md]], [[sources/adathinkdrive.md]]) and no learned router.
- **The price is measured on three benchmarks, and it is small.** NAVSIM **+0.4 PDMS** (87.8 → 88.2, from 78 K reasoning-conditioned samples); WOD-E2E test **+0.02 RFS** (7.76 → 7.78, from 142 K); PAI-AV **−0.01 m** ADE. This sits at the low end of the range this page records, below [[sources/wcog-vla.md]]'s +0.5 PDMS for 9.9 s of Game-CoT and consistent with [[sources/nord.md]]'s position that reasoning annotations are not the bottleneck.
- **The trace is also the RL exploration source.** In Stage 4 the frozen VLM samples **eight independent reasoning traces per scene**, each conditioning one of eight rollouts. Reasoning diversity is thus repurposed as policy diversity — a use of CoT nothing else here makes.

### The paper's own doubt about what the trace is doing {#trace-doubt}

*See [[concepts/reasoning-faithfulness.md]] for the four readings of a trace (rationale, verbalized action, capacity, ignored), the evidence table, and the [hindsight-annotation problem](reasoning-faithfulness.md#hindsight).*

The limitations section contains the most direct statement of the rationale-adherence problem in this wiki:

> "The generated trajectory does not always adhere to the textual rationale. Although reasoning improves downstream planning performance, **part of this gain may stem from the additional model-internal information that the self-generated trace contributes to the conditioning context.**"

That is the authors naming the alternative hypothesis: the trace may be acting as extra conditioning capacity — more tokens, more computation, a richer KV cache — rather than as a *reason*. At +0.4 PDMS and +0.02 RFS, the effect is small enough that either explanation fits.

**This is the sharpest form of a question this page has circled repeatedly.** [[sources/alpamayo-r1.md]] addressed it with a binary CoC–action consistency reward; [[sources/spanvla.md]] with an action-alignment penalty; [[sources/orion.md]] with latent alignment between reasoning and action. Qwen-Drive uses **none of these** — nothing in $\mathcal{L}_{\mathrm{plan}}$ or in the RL objective requires the trajectory to match the text — and the paper correctly identifies "explicit consistency supervision between the rationale and the generated trajectory" as the missing piece.

**The decisive experiment is cheap and nobody has run it**: condition the planner on a *shuffled* trace from a different scene. If PDMS holds, the gain is conditioning capacity; if it collapses, the trace carries scene-specific content.

### Multi-timescale causation as an unsolved CoT content problem {#multi-timescale}

The other limitation is a content taxonomy gap that applies to every CoC-style dataset on this page:

> "Driving mixes causes that act at different time scales. A red light 20 m ahead calls for early, gradual deceleration, whereas a child emerging 5 m ahead demands an immediate response. When such causes coexist, the model remains unstable in identifying the governing cause and its temporal scope. Even when the suggested trend is appropriate, the decision executed within the next 1 to 2 s may not reflect the stated immediate cause."

Neither Alpamayo-R1's 14-type closed decision set nor Qwen-Drive's rule-derived motion prior encodes *when* a cited cause takes effect. A CoC trace names the causes; it does not order them in time. That is a schema change, not a scale problem.

### Annotation method: classification-question audits {#classification-audits}

Worth adding to this page's annotation-methods section as a variant of GT-grounded CoT. Qwen-Drive's traces are written by Qwen3.7-Plus conditioned on a **rule-based motion prior** derived from the recorded future, then audited — and the audit design is the reusable part:

> "Rather than requesting scalar quality scores, judge models answer classification questions, and their decisions are aggregated programmatically."

The audit checks the predicted maneuver against the ground-truth trajectory, classifies the causal role of each cited factor, and **rejects traces that reveal future information** — the leakage control [[sources/alpamayo-r1.md]] enforces with a 0–2 s human observation window, here enforced by a judge. A separate rarity score prioritizes rare scenes. Against LRM-as-critic scalar grading, classification-plus-aggregation is cheaper to calibrate and produces an auditable reject reason.

## Grounded Reasoning with Box Tokens (GRAVA) {#grounded-reasoning}

[[sources/grava.md]] makes the CoT itself the grounding channel. Every action-relevant object appears in the trace as a `<|box_start|>…<|box_end|>` token together with its ego-frame distance, speed and heading. The trace then goes per object: interaction → object decision (Follow / Yield / Caution …). The object decisions merge into an ego decision, and a primitive action (STOP / CRAWL / CURVE / CRUISE plus parameters) follows in the same stream. Targets are serializations of a trajectory-anchored DAG that is built offline by agents and uses privileged LiDAR states.

**The reasoning-structure ladder** (internal long-tail benchmark, all variants RL-trained, CDS):

| Reasoning condition | KOC | CDS |
|---|---:|---:|
| Action only | 77.4 | 74.8 |
| Coarse reasoning | 80.6 | 77.9 |
| Grounded objects only | 87.9 | 85.5 |
| Full GRA (objects + interactions + decisions) | **92.3** | **90.1** |

This is the most graded CoT ablation in the wiki. It separates *having* reasoning (+3.1), *grounding* it (+7.6) and *structuring* it (+4.6). Caveats: it is on a non-public benchmark, it is single-run, and it has no NAVSIM counterpart.

**Trace intervention.** Hold the scene and the decoder fixed and swap low-reward reasoning for high-reward reasoning. Normalized PDMS goes from 0.438 to 0.823, and win rate from 3% to 55%. This answers [Qwen-Drive's doubt](#trace-doubt) for this architecture: **the action head reads the trace**. The caveat is that the trace's last statement is the ego maneuver (e.g. keep_lane + decelerate), and that statement was **copied from the expert trajectory's intent** at annotation time. So the experiment shows the planner obeys the stated maneuver, not that the grounded evidence produced it.

**Numerical density.** Traces that mention more object distances and speeds score higher (0.804 → 0.899 normalized PDMS across bins). The result is correlational and has no length control.

**Reconciling with NoRD.** GRAVA's evidence comes from a benchmark built from obstruction, lane-borrow and hazard scenes. That is the regime where [[sources/adathinkdrive.md]] finds CoT helps, and the opposite of NAVSIM's easy-scene majority, where [[sources/nord.md]] finds it unnecessary. **All three results are consistent with "CoT pays in interaction-heavy scenes, and only after RL."**

## CoT Design Space

| Paper | CoT type | Generation method | RL optimization | Inference CoT? | Notes |
|-------|---------|------------------|----------------|---------------|-------|
| ReCogDrive | 3-part text (traj + reason + desc) | Frontier VLM | GRPO (NAVSIM) | Yes | CoT consumed by diffusion planner |
| UniUGP | 4-stage text | GT-grounded + VLM | None | Yes | Trajectory-grounded; ensures causal consistency |
| AutoVLA | 4-stage text | Frontier VLM (GT hint) | GRPO + length penalty | Adaptive | Adaptive fast/slow within single model |
| AdaThinkDrive | Spatial text (CIPO + boundary) | Rule-based + Frontier VLM | GRPO + Adaptive Think Reward | Adaptive | Learns *when* to reason per scene |
| DeepSight | Long-tail text CoT, gated on world state | Qwen3-VL-235B pipeline (~1.3M labels) | None (SFT only) | Adaptive (<30% frames) | CoT decided after 5-frame BEV latent prediction; +4.12% latency; CoT alone weak vs. world model |
| AutoDrive-R² | 4-stage + backward-check | GT-grounded | Physics GRPO | Yes | Self-reflection adds zero-shot generalization |
| Alpamayo-R1 | Chain-of-thought corpus (CoC) | Hybrid (rule + VLM) | GRPO + LRM-as-critic | Yes | Most principled critic; internal evals only |
| ELF-VLA | Failure-diagnostic feedback | Teacher model from wrong traj + GT + metrics | GRPO with corrected refinement injection | Yes | CoT repair loop for persistent failures |
| FSDrive | Visual CoT (future frame) | AR VQ-VAE generation | None | Yes (mandatory) | Spatial/temporal; not text-based |
| DynVLA | Dynamics CoT (ego/env VQ tokens) | Dynamics Tokenizer from adjacent frames | GRPO with PDMS + format reward | Yes | Compact 16-token dynamics trace; 0.37s controlled latency |
| SpanVLA | Compact text CoT + action-token training | Gemini-3-Pro on mReasoning with critical-element filtering | GRPO with CoT length and action-reasoning alignment penalties | Adaptive | VLM reasons until action token, then FM expert decodes continuous trajectory |
| OneVL | Latent vision-language CoT | Visual and language auxiliary decoders; future-frame tokens + CoT text | None | Prefilled latent tokens; optional post-hoc decoders | 88.84 PDMS at answer-only AR latency; MLP head reaches 0.24s |
| CoWorld-VLA | Multi-expert latent CoT (4 typed tokens) | Frozen V-JEPA / VGGT on future frames + a Wan video DiT the token conditions + GT trajectory | None | Yes — tokens are produced by the VLM and condition four denoising branches | +5.0 PDMS against a plain readout, **+1.1 against a diffusion planner**; dynamic token never ablated alone |
| HERMES | Risk-aware reasoning | Offline Frontier VLM | None | No (baked into embeddings) | CoT at annotation time only |
| GRAVA | Grounded text: box tokens + metric states → per-object interaction/decision → ego decision | Trajectory-anchored agentic DAG (forward grounding + backward trajectory anchoring; privileged LiDAR) | DAPO-style GRPO on re-screened recoverable scenes; one advantage for reasoning + action | Yes (always) | Action is an in-stream primitive with parameters; ego decision in targets copied from GT trajectory intent |
| NoRD | None | — | Dr. GRPO | No | Reasoning-free; competitive with CoT at 60% data |

---

## Efficiency Tradeoff

Text CoT adds latency proportional to reasoning token count. Reported latencies:

| Method | Latency | CoT? | PDMS |
|--------|---------|------|------|
| NoRD | Fast (3× fewer tokens vs. CoT VLAs) | No | 85.6 |
| AdaThinkDrive (Non-Think) | 0.68s | No | 88.3 |
| **AdaThinkDrive (Adaptive)** | **0.74s** | **Adaptive** | **90.3** |
| AdaThinkDrive (Always-Think) | 0.86s | Yes | 88.9 |
| AutoVLA (post-RFT) | ~1 Hz (varies) | Adaptive | 89.11 |
| **SpanVLA (FM)** | **0.67s** | **Adaptive** | **90.3** |
| OneVL (prefill AR) | 4.46s | Latent prefill | 88.84 |
| OneVL (MLP head) | 0.24s | Latent-trained, no AR waypoint decode | 86.83 |
| LinkVLA (C2F, CoT excluded) | 48ms | CoT excluded from timing | 91.01 DS |

Note: LinkVLA's 48ms excludes CoT text generation; real latency is higher.

**Adaptive CoT (AdaThinkDrive)** achieves the best efficiency-performance tradeoff currently in the wiki: 14% faster than always-Think while outperforming both fixed modes.

---

## Open Questions

*Wiki-wide open questions are collected in [[research-directions.md]].*

- At what dataset scale does CoT stop providing marginal benefit over data-efficient reasoning-free training (NoRD)? Is there a data regime where NoRD-style Dr. GRPO training catches up to 212K+ CoT-supervised AutoVLA?
- Does LRM-as-critic (Alpamayo-R1) provide measurably better CoT quality than frontier VLM annotation, or is the quality difference lost in downstream trajectory generation noise?
- Can the backward-check mechanism (AutoDrive-R²) be combined with NAVSIM simulator feedback (GRPO) for dual verification — both logical consistency and closed-loop safety?
- Does visual CoT (FSDrive) complement text CoT — e.g., use text CoT for high-level decisions and visual CoT for spatial collision prediction?
- Does GRAVA's reasoning ladder (action-only 74.8 → full grounded 90.1 CDS) hold on a public benchmark? Does it survive if the ego-decision target is inferred from the object branches instead of copied from the expert trajectory's intent?
- Is adaptive CoT (AdaThinkDrive) robust to distribution shift? If the scene complexity classifier fails on OOD scenarios (construction zones, rare events), the model may default to Non-Think in precisely the situations that need CoT most.
