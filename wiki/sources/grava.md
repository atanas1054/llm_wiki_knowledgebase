---
title: "GRAVA: Grounded Reasoning-to-Action Representation and Learning for Autonomous Driving"
type: source-summary
sources: ["raw/papers/[-0.5mm] GRAVA GRAVA_ Grounded Reasoning-to-Action Representation and Learning for Autonomous Driving.md"]
related: [concepts/chain-of-thought-for-ad.md, concepts/rl-for-ad.md, concepts/navsim-benchmark.md, concepts/action-tokenization.md, concepts/perception-for-planning.md, concepts/vlm-domain-adaptation.md, concepts/best-of-n.md, sources/qwen-drive-1.0.md, sources/curious-vla.md, sources/adathinkdrive.md, sources/autovla.md, sources/spanvla.md, sources/dynvla.md, sources/recogdrive.md, sources/drivevla-w0.md, sources/explorevla.md, sources/alpamayo-r1.md, sources/linkvla.md, sources/dapo.md, sources/nord.md, sources/elf-vla.md, sources/sgdrive.md, sources/diffusiondrive.md]
created: 2026-09-27
updated: 2026-09-27
confidence: medium
---

**Paper**: GRAVA: Grounded Reasoning-to-Action Representation and Learning for Autonomous Driving
**Authors**: Xiao Liu, Haoyu Li, Jianghao Leng, Lin Wang, Chao Sun
**Orgs**: not stated in the clipping (compute is NVIDIA H20, so the lab is most likely in China; asset filenames use the internal project name "nous")
**arXiv**: 2609.15169v1
**Code / data**: none announced. GR-NavSim (the dataset) is not stated to be released. The internal long-tail benchmark is not released.

---

## Source Integrity Note

The clipping runs from the abstract through Appendix F, and all six tables (I–VI) and both appendix tables (S1, S2) are present. **Eight images are present. Five captioned figures have no image**, and one caption sits on the wrong image:

| Figure | Caption | Image in clipping? |
|---|---|---|
| Fig. 1 | Grounding and reasoning-to-action gaps | ✅ `nous_v2_fig1_gap.png` |
| Fig. 2 | GRAVA overview | ✅ `nous_v2_fig1_main.png` |
| Fig. 3 | GRA graph and serialization | ✅ `nous_v2_fig2_main.png` |
| Fig. 4 | Training loss and Active RL PDMS curves | ❌ **missing**. Its numbers appear only in the prose. |
| Fig. 5 | GR-NavSim scale compared with public QA datasets | ❌ missing |
| Fig. 6 | Reasoning-intervention PDMS and win rate | ❌ missing. The prose gives the headline numbers (0.438→0.823; 3%→55%). |
| Fig. 7 | "Numerical grounding density and planning quality" | ⚠️ **The attached image (`nous_qualitative_compact_draft.png`) is the four-panel qualitative figure, not a density plot.** The density figure is missing. The prose gives the key numbers. |
| Fig. 8 | Qualitative GRA examples (referenced in §V-G, never captioned) | This is the image attached under the Fig. 7 caption. |
| Fig. S1, S2 | Expanded qualitative examples | ✅ |
| Fig. S3 | Active RL infrastructure | ❌ missing |
| Fig. S4, S5 | Alias grounding; QA overlay | ✅ |

The **cross-references in the text are also off by one or two**. The prose calls the NAVSIM main table (Table III) "Table V" and the NAVSIM component ablation (Table IV) "Table V". It calls the density figure "Figure 8". The table numbers used on this page are those in the table captions.

The Appendix E 30-QA transcript survives only as SVG text. It was recovered and is summarized [below](#qa-transcript).

---

## Summary

GRAVA is a **purely autoregressive driving VLA** built on Qwen3-VL-8B-Instruct. It uses a single front camera. It generates, in one stream:

1. **Grounded reasoning $r$.** This is text in which every action-relevant object appears as a `<|box_start|>(x1,y1),(x2,y2)<|box_end|>` token (0–1000 normalized), together with its ego-frame distance, speed and heading. The objects are linked to object-level interactions, then object-level decisions, then an ego decision.
2. **A compact "Executable Planner" action** $a=(p,g,\phi)$. This is a motion primitive (STOP / CRAWL / CURVE / CRUISE), a gear (D/R) and primitive-specific parameters. A fixed, parameter-free geometric decoder turns it into 8 waypoints over 4 s.

The reasoning targets are serializations of a **trajectory-anchored typed DAG (the "GRA graph")**, built offline by a tool-augmented multi-agent annotation pipeline. The pipeline has two passes. *Forward scene grounding* finds candidate objects and their states. *Backward trajectory anchoring* starts from the expert trajectory and keeps only the objects, interactions and decisions that explain it. The same graph supplies the targets for **2.2M cognition QA pairs** (pre-training) and **70K reasoning-to-action traces** (planning). Together these form **GR-NavSim**: 107K nuPlan/NAVSIM scenes, which the authors call the largest open-vocabulary, non-template driving QA set.

Training has four stages: **GRA QA pre-training → planner warm-up (SFT on $[r^*;a^*]$) → verified self-distillation (reward-selected own rollouts) → "Active RL"**. Active RL is DAPO-style GRPO on complete $[r;a]$ sequences, with PDMS as the reward. It is restricted to a re-screened set of **"recoverable" scenarios**: greedy score is below threshold, best-of-8 reaches a high score, and the rollout group has enough variance.

**Headline results**:
- **NAVSIM-v1 navtest: 90.48 PDMS**, single greedy sample, one camera.
- Pre-RL checkpoint: 82.10, so Active RL is worth **+8.38**.
- Internal long-tail benchmark: full GRA vs. action-only gives **KOC 92.3 vs. 77.4 and CDS 90.1 vs. 74.8**, i.e. +19.3% / +20.5% relative (arithmetic checks).
- Held-out GR-NavSim QA: 6.86 overall vs. GPT-5.4 at 5.39.

---

## Positioning

- **Where it sits on the wiki's AR-VLA ladder.** 90.48 is +0.18 over [[sources/curious-vla.md]] and [[sources/adathinkdrive.md]] (both 90.3). The paper's "best purely autoregressive" claim holds **only within its own table**. [[sources/dynvla.md]] (91.7) is also purely autoregressive: dynamics tokens, then action tokens. It is absent from GRAVA's table. [[sources/elf-vla.md]] (91.0) is also absent.
- **The direct-action comparison group is the right one.** GRAVA groups methods by design into non-VLA, VLA + trajectory head/world model, and direct-action VLA. This is a better way to frame a single-number comparison than most tables in the wiki use. Its AutoVLA one-shot row (80.5) matches the pre-RL AutoVLA number [[sources/qwen-drive-1.0.md]] reported, which confirms the source.
- **Nearest relatives.**
  - [[sources/alpamayo-r1.md]] also builds trajectory-anchored causal traces (Chain-of-Causation), but decodes with a flow-matching expert.
  - [[sources/recogdrive.md]] uses the same "cognition pre-training → planner imitation → RL" staging, with a diffusion planner.
  - [[sources/linkvla.md]] also targets language–action consistency, through a shared codebook.
  - [[sources/curious-vla.md]] is the closest RL relative. Its sample refresh after policy narrowing anticipates Active RL's re-screening.
  - GRAVA's specific move is to put **box tokens and metric states inside the reasoning trace itself**, not in an auxiliary QA head.

---

## Method

### Problem framing: two gaps

![[nous_v2_fig1_gap.png|Three paradigms: (A) reasoning without explicit visual grounding; (B) grounding separated from action through auxiliary tasks or modules; (C) GRAVA's single stream from grounded evidence through interaction reasoning and decisions to executable action]]

*Fig. 1: (A) Reasoning without explicit visual grounding. (B) Grounding separated from action generation through auxiliary tasks or modules. (C) GRAVA connects grounded evidence, interaction reasoning, decisions, and executable actions within one autoregressive stream.*

- **G1, the grounding gap.** The reasoning names entities in free language and never resolves them to image regions or metric states.
- **G2, reasoning-to-action fragmentation.** Grounding is learned through separate QA or perception tasks, and the objects it establishes are not carried into the reasoning that produces the action.

### Overview

![[nous_v2_fig1_main.png|GRAVA overview: (A) agentic data pipeline with a tool box (camera, 2D det, 3D state, BEV, ego state, trajectory plot, sign/OCR, case recall) and four agents (scene grounding, interaction, decision/planning, trajectory backtrace); (B) VLM emitting thinking tokens then planner-parameter tokens into the Executable Planner; (C1) GRA QA data; (C2) planning/decision data; (D1) GRA pre-training; (D2) post-training: light SFT warm-up, self-distillation, Active RL (GRPO) with K rollouts, planner run, driving reward, active-set update]]

*Fig. 2: (A) Agentic GRA graph construction through forward grounding and backward trajectory anchoring. (B) Autoregressive grounded reasoning and Executable Planner action generation, followed by fixed trajectory decoding. (C1–C2) Cognition and planning supervision derived from the same GRA graph. (D1–D2) Progressive training from grounded pre-training to Active RL.*

The input is $x=(V_{\mathrm{env}}, s_{\mathrm{ego}}, c)$, where $c$ is the navigation command. Per the appendix, the input also includes the native 1920×1080 front image and the motion history. The policy factorizes as

$$\pi_\theta(y\mid x)=\pi_\theta(r\mid x)\,\pi_\theta(a\mid x,r),\qquad \hat\tau=D_p(g,\phi;s_{\mathrm{ego}}).$$

The decoder $D_p$ has no learnable parameters. Offline, the target graph is $G^*=\mathcal A(x,\tau^*,z)$, where $\tau^*$ is the expert trajectory, $z$ is privileged evidence (surround cameras, LiDAR-assisted 3D states) and $\mathcal A$ is the agentic pipeline. At inference only $x$ is used.

### The GRA graph

![[nous_v2_fig2_main.png|GRA graph structure: (A) structured node chain — scene context, trajectory analysis (from GT trajectory), object grounding, object decision, ego decision, planning, with forward and backward traces; (B1) action-relevant subgraph for the running example, lead vehicle G (same lane, stopped 15.0 m → decel + follow → Follow) and adjacent vehicle L (right adjacent lane, stopped → lateral hazard → Caution) merging at ego_decision keep_lane, decelerate, with a dashed edge from trajectory.intent (keep_lane, decelerate) into ego_decision; (B2) serialized grounded reasoning with box tokens followed by a planning line]]

*Fig. 3: GRA graph structure and serialization. (A) Semantic node types and their organization. (B1) Object-specific reasoning paths converge at the ego decision and connect to the action anchor. (B2) The action-relevant subgraph is serialized into grounded reasoning with visual references and physical states, followed by the planning target.*

The graph is a typed DAG, $G=(V,E,\nu,\eta)$, with nodes $V=V^S\,\dot\cup\,V^O\,\dot\cup\,V^\Psi\,\dot\cup\,V^D\,\dot\cup\,\{v^\delta,v^a\}$:
- $V^S$: scene and route context.
- $V^O$: grounded objects.
- $V^\Psi$: interactions.
- $V^D$: object-level decisions.
- $v^\delta$: the ego decision.
- $v^a$: the terminal action anchor.

**Object node**: $o_i=(m_i,b_i,\sigma_i)$. Here $m_i$ is the linguistic reference, $b_i\in[0,1000)^4$ is the 2D box and $\sigma_i$ is the ego-frame state (category, relative position, distance, speed, heading).

**Branch-and-merge topology**: $(V^S,o_i)\to\{\psi_i^\ell\}\to d_i$ for each critical object $i\in\mathcal C$, then $(V^S,\{d_i\})\to v^\delta\to v^a$.

**Structural grounding constraints**:
- Every interaction or decision node has a grounded-object ancestor: $\mathrm{Anc}_G(v)\cap V^O\neq\emptyset$.
- Every $d_i$ reaches $v^\delta$.
- $v^\delta$ reaches $v^a$.

**Serialization**: $G_a=G[\mathrm{Anc}_G(v^a)]$ and $r=\mathrm{Ser}(G_a)$, in topological order. Objects are introduced with their box and state; later statements reuse the box token.

**Shared supervision** comes from the same $G^*$:

$$\mathcal L_{\mathrm{QA}}=-\mathbb E\log\pi_\theta(u\mid x,q),\qquad \mathcal L_{\mathrm{plan}}=-\mathbb E\log\pi_\theta([r^*;a^*]\mid x).$$

⚠️ **Fig. 3 shows something the text never discusses.** Panel B1 has a **`trajectory.intent: keep_lane, decelerate` node, orange (derived from the GT trajectory), with a dashed edge straight into `ego_decision`**. Panel B2's planning line says "longitudinal step spacing decreases from 1.06 m to 0.24 m", which is a description of $\tau^*$. The ego decision is therefore **read off the expert trajectory**, not inferred from the object branches. The Appendix E transcript confirms this: the ego-decision chain ends `trajectory_intent(turn_right, accelerate) -> ego_decision(turn_right, accelerate)`. See [What the reasoning trace is](#what-the-trace-is).

### Executable Planner

**Table I — action schema** (ego frame):

| Primitive $p$ | Parameters $\phi\in\Phi_p$ | Shape | Motion regime |
|---|---|---|---|
| STOP | endpoint | $\mathbb R^2$ | near-stationary or stopping motion |
| CRAWL | points | $\mathbb R^{8\times2}$ | low-speed stop-and-go |
| CURVE | $p_1,p_2,p_3$ | $\mathbb R^{3\times2}$ | turning and high-curvature motion |
| CRUISE | progress_end, $v_{\mathrm{end}}$, lat_end | $\mathbb R^3$ | lane keeping and car following |

- The action is serialized as `{planner: p, gear: g, params: φ}` and continues the reasoning directly.
- Gear is D or R.
- The decoder (Algorithm S1) works per primitive:
  - **STOP** expands the endpoint with a speed-dependent stop profile.
  - **CRAWL** decodes its 8 points directly.
  - **CURVE** evaluates a cubic Bézier from the ego origin at $t=k/8$.
  - **CRUISE** uses a monotone cubic Hermite longitudinal profile, conditioned on current and terminal speed, plus a linear lateral ramp to lat_end.
- **Why CRAWL keeps 8 waypoints.** Its coordinate range is narrow, so direct regression is easy. A low-dimensional curve would smooth out small stop-and-go changes.
- **Supervised targets.** $a^*$ is derived deterministically from $\tau^*$: pick the primitive from the motion profile and command, pick the gear from the motion direction, fit only $\Phi_{p^*}$, then quantize.

This is a hand-designed, **mode-conditioned parametric action vocabulary**. In the terms of [[concepts/action-tokenization.md]], it sits between raw text waypoints (Curious-VLA) and a learned codebook (AutoVLA's K-disk). The decoder carries no policy, so every action choice stays in $\pi_\theta(a\mid x,r)$.

### Training stages

**Table S1 — training configuration** (Qwen3-VL-8B-Instruct, DeepSpeed ZeRO-2, full fine-tune):

| Stage | Epochs | LR | Batch / GA | Max length | Compute |
|---|---|---|---|---|---|
| GRA pre-training | 1 | 1×10⁻⁵ | 4 / 1 | 4,096 | 4×8 H20 |
| Planner warm-up | 3 | 1×10⁻⁵ | 4 / 1 | 4,096 | 2×8 H20 |
| Self-distillation | 1 | 5×10⁻⁶ | 4 / 1 | 4,096 | 2×8 H20 |
| Active RL | — | 1×10⁻⁶ | 4 / 1 | 4,096 | 1 rollout + 3 training + 1 sim node |

1. **GRA pre-training** on $\mathcal L_{\mathrm{QA}}$ (about 30+ QAs per scene).
2. **Planner warm-up** on $\mathcal L_{\mathrm{plan}}$. It uses **62,111 of 102,861** cases after grounding-consistency filtering; this is the source of the "about 60% of demonstrations" claim.
3. **Verified self-distillation.** Sample $[r_i;a_i]$, keep rollouts selected by closed-loop reward, and SFT on the kept $[r^+_i;a^+_i]$ pairs. The result is the frozen reference $\pi_{\mathrm{ref}}$. This is STaR-style rejection-sampling fine-tuning. The reward threshold is not given.
4. **Active RL.**
   - Candidate pool: $\mathcal D_g=\{x: s_{\mathrm{greedy}}<\tau_g\}$.
   - Active set: $\mathcal D_{\mathrm{act}}=\{x\in\mathcal D_g:\max_i s_i\ge\tau_{hi},\ \min_i s_i<\tau_{lo},\ \mathrm{std}_i s_i\ge\tau_\sigma\}$.
   - Reward: PDMS of the decoded trajectory; 0 for unparsable or schema-violating outputs.
   - Advantage: group-normalized $\hat A_i$.
   - Clipping: DAPO asymmetric clip $(1-0.20,\,1+0.28)$.
   - KL to $\pi_{\mathrm{ref}}$ with $\beta=0.01$.
   - Sampling: $K=8$, $T=0.8$, top-p 0.95.
   - Loop: after each optimization window, re-evaluate greedily and rebuild $\mathcal D_{\mathrm{act}}$.
   - **The same sequence-level advantage updates both reasoning tokens and action tokens.**

**Table S2 — recoverable-scenario selection** (thresholds: greedy < 0.9, max ≥ 0.9, min < 0.8, std ≥ 0.10):

| Selection step | Cases | Meaning |
|---|---:|---|
| Current checkpoint evaluation | 102,861 | One greedy completion per case |
| Greedy below 0.9 | 26,151 | Cases with potential for improvement |
| Best-of-8 reaches at least 0.9 | 14,398 | Recoverable under best-of-8 sampling |
| All reward-variation criteria | 4,676 | Cases used for the RL update |

Consecutive active sets (4,676 and 6,464) share only 1,995 cases, a **Jaccard overlap of 21.8%**. This is the paper's argument for re-screening after every outer loop. The RL update set is **4.5% of the pool**.

**Infrastructure (App. C; figure missing)**:
- A vLLM rollout server on 8×H20 with tensor parallelism and prefix caching.
- Trainer: 3×8 H20, DDP plus ZeRO-2, all parameters updated.
- A separate sim-engine node holds scene caches and returns scalar reward plus metadata.
- Zero-variance groups are resampled and overlong completions filtered, with up to 3 attempts.
- This is DAPO's dynamic sampling. Active selection is an outer-loop, per-scene version of the same idea.

### Annotation-time aliases (App. D)

![[grava_alias_grounding.png|Alias resolution: annotation-time short aliases (G, L) linking annotation LLM and perception tools are replaced at training time by bounding-box tokens referring to the same image regions, with interaction content unchanged]]

*Fig. S4: Short aliases keep object identity across annotation steps. Training replaces them with bounding-box tokens for the same image regions, preserving the interaction content.*

Objects are annotated using surround cameras plus a LiDAR-assisted 3D tool, and kept if they project into some annotation camera. At training time they are reprojected into the *training* cameras (front only) and dropped if not visible. Aliases become box tokens. **So privileged 3D states from LiDAR become the metric numbers the front-camera model is trained to emit.**

---

## Data and Evaluation

**Table II — GR-NavSim composition**:

| Corpus component | Count | Role |
|---|---:|---|
| Driving scenes with grounded state labels | 107K | Grounded objects, physical states, interactions, and decisions |
| Grounded question-answer pairs | 2.2M | Grounded scene understanding and interaction reasoning |
| Front-camera Executable Planner examples | 415K | Planner warm-up and self-distillation |
| Grounded trajectory traces | 70K | GRA paths paired with planner actions |

⚠️ **These counts do not reconcile with the appendix.** Table II says 415K planner examples feed warm-up and self-distillation. Appendix A says warm-up uses 62,111 cases, and 70K reasoning traces exist. The 415K may include self-distillation samples or frames without traces, but the paper does not say. This bears on the "60% of demonstrations" claim.

**Fig. 5** (the scale comparison with public QA datasets) is missing.

### Internal long-tail benchmark

- **Content**: 50K clips, 700K frames covering route obstructions, lane borrowing and road hazards.
- **Key objects**: $o_j=(t_j,\Omega_j,p_j)$, with a manually annotated BEV region and a passable / non-passable label.
- **Scoring**: predicted trajectories are rolled out, then scored with

$$\mathrm{CDS}=G_{\mathrm{col}}G_{\mathrm{nds}}\,(0.45\,\mathrm{KOC}+0.35\,\mathrm{Progress}+0.20\,\mathrm{Comfort}).$$

- **KOC** turns pass/hold labels into route-projected geometric constraints:
  - Pass means the maximum front-edge progress exceeds $s_j^-+d_{\mathrm{trigger}}$.
  - Hold means it does not exceed $U=\max(s_{j_0}^- - d_{\mathrm{follow}},U_{\min})$.
  - Only pass-required objects before the nearest hold boundary count.
  - Progress is zeroed if the ego crosses a hold boundary.
- **Why it is designed this way.** Several geometrically different trajectories can satisfy a pass/hold constraint, so the metric does not require agreement with the logged future. This is a sound design choice, well argued.
- **Undisclosed**: the margins, whether rollouts are reactive, the simulator, and whether the benchmark's clips overlap with any training data.

---

## Results

### Table III — NAVSIM navtest v1, single sample

| Method | Backbone | Views | NC | DAC | EP | TTC | Comf | PDMS ↑ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| *Non-VLA methods* | | | | | | | | |
| UniAD | — | Cam | 97.8 | 91.9 | 78.8 | 92.9 | 100.0 | 83.4 |
| PARA-Drive | — | Cam | 97.9 | 92.4 | 79.3 | 93.0 | 99.8 | 84.0 |
| TransFuser | — | C+L | 97.7 | 92.8 | 79.2 | 92.8 | 100.0 | 84.0 |
| DiffusionDrive | — | C+L | 98.2 | 96.2 | 82.2 | 94.7 | 100.0 | 88.1 |
| *VLAs with additional trajectory head or world model* | | | | | | | | |
| DriveVLA-W0 | Emu-3-8B | Cam | 98.7 | 99.1 | 87.6 | 97.1 | 100.0 | 90.2 |
| ExploreVLA | — | Cam | 98.8 | 98.4 | 83.5 | 96.5 | 99.9 | 90.4 |
| ReCogDrive | InternVL2-8B | Cam | 98.2 | 97.8 | 83.5 | 95.2 | 100.0 | 89.6 |
| *VLAs with direct action generation* | | | | | | | | |
| AutoVLA one-shot | Qwen2.5-VL-3B | Cam | 96.9 | 92.4 | 75.8 | 88.1 | 99.9 | 80.5 |
| AutoVLA Post-RFT | Qwen2.5-VL-3B | Cam | 98.4 | 95.6 | 81.9 | 98.0 | 99.9 | 89.1 |
| Curious-VLA | Qwen2.5-VL-3B | Cam | 98.4 | 96.9 | 88.5 | 97.9 | 98.1 | 90.3 |
| AdaThinkDrive | InternVL3-8B | Cam | 98.4 | 97.8 | 84.4 | 95.2 | 100.0 | 90.3 |
| GRAVA before Active RL | Qwen3-VL-8B | Cam | 96.1 | 91.2 | 78.8 | 91.0 | 99.7 | 82.1 |
| **GRAVA after Active RL** | Qwen3-VL-8B | Cam | **98.8** | 97.6 | 83.5 | 97.1 | **100.0** | **90.5** |

- **Baseline hygiene is good.** Every non-GRAVA row matches the wiki's canonical value. ReCogDrive appears at 89.6, its RL figure. AutoVLA's one-shot and post-RFT rows are both shown.
- **What RL buys is safety, not progress.** The pre-RL → post-RL deltas are NC +2.7, **DAC +6.4, TTC +6.1**, EP +4.7 and Comf +0.3. The wiki has now recorded four PDMS-reward RL runs on already-safe policies that mostly bought progress. GRAVA is the reverse: RL mainly repairs a pre-RL policy with **DAC 91.2, below TransFuser's 92.8**.

### Table IV — NAVSIM component ablation (greedy, single sample)

| Pre-training | Reasoning | Action representation | SD | Active RL | NC | DAC | EP | TTC | Comf | PDMS ↑ |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| None | Full GRA | Executable Planner | ✓ | ✓ | 95.3 | 90.0 | 75.4 | 90.9 | 99.8 | 79.80 |
| GRA | Full GRA | Direct waypoints | ✓ | ✓ | 97.7 | 95.0 | 82.4 | 94.2 | 99.9 | 87.23 |
| GRA | Full GRA | Executable Planner | — | — | 96.0 | 91.3 | 77.3 | 91.7 | 99.9 | 81.74 |
| GRA | Full GRA | Executable Planner | ✓ | — | 96.1 | 91.2 | 78.8 | 91.0 | 99.7 | 82.10 |
| GRA | Full GRA | Executable Planner | ✓ | ✓ | 98.8 | 97.6 | 83.5 | 97.1 | 100.0 | **90.48** |

- Removing GRA pre-training costs **−10.68**, and RL does not recover it. Fig. 4a (missing) reportedly shows lower initial planning loss with pre-training.
- Replacing the Executable Planner with direct waypoints costs **−3.25**.
- SD is worth +0.36 and Active RL +8.38.
- Fig. 4b (missing) reportedly shows gains across outer loops, and a *decline* when training continues past one on-policy window without re-screening.
- **No NAVSIM row varies the reasoning.** There is no action-only or coarse-reasoning row on NAVSIM. The reasoning-structure ablation exists only on the internal benchmark. Active selection is never compared against plain GRPO on the full pool.

### Table V — Internal long-tail benchmark ablation

All variants use the Executable Planner and verified SD.

| Pre-training | Reasoning condition | Active RL | Col. | NDS | KOC | Progress | Comfort | CDS ↑ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| None | Full GRA | ✓ | 95.8 | 94.7 | 81.7 | 89.6 | 98.8 | 79.7 |
| Ungrounded | Full GRA | ✓ | 96.3 | 95.4 | 84.2 | 90.7 | 99.1 | 82.2 |
| GRA | Action only | ✓ | 94.6 | 93.2 | 77.4 | 86.9 | 97.9 | 74.8 |
| GRA | Coarse reasoning | ✓ | 95.2 | 94.1 | 80.6 | 88.4 | 98.6 | 77.9 |
| GRA | Grounded objects only | ✓ | 97.1 | 96.4 | 87.9 | 91.3 | 99.3 | 85.5 |
| GRA | Full GRA | — | 93.1 | 94.9 | 69.8 | 90.1 | 98.9 | 73.1 |
| GRA | Full GRA | ✓ | 98.2 | 97.6 | 92.3 | 92.9 | 99.6 | **90.1** |

Col. and NDS are gate pass rates.

The ladder is clean and monotone:
- action-only 74.8
- coarse reasoning 77.9
- grounded objects only 85.5
- full GRA 90.1

**Grounding the objects is worth +7.6 over coarse reasoning, and the interaction/decision paths another +4.6.**

On pre-training: ungrounded pre-training is worth +2.5 over none, and GRA pre-training another +7.9. What "ungrounded" pre-training contains (the same QAs with boxes and numbers stripped?) is not defined.

**This is the strongest reasoning-helps evidence in the wiki**, and the only evidence of it on the benchmark built to show it. Note, however, that **full GRA without RL (73.1) is below action-only with RL (74.8)**. Reasoning helps only once RL has been applied.

### Grounded reasoning analyses (§V-E)

**Reasoning intervention (Fig. 6, figure missing).**
- Setup: fix the scene and the decoder. Replace low-reward reasoning with the high-reward reasoning from the same scene, and regenerate the action.
- Result: normalized PDMS goes from **0.438 to 0.823**, and win rate from **3% to 55%**.
- What it shows: **the action follows the text**. This is a faithfulness result in the "the head is not ignoring the CoT" sense. See [What the trace is](#what-the-trace-is) for what it does *not* show.

**Numerical grounding density (Fig. 7 density plot missing).**
- Density is the count of distance and speed mentions for action-relevant objects.
- Normalized PDMS rises from **0.804 in the lowest-density bin to 0.899 in the highest**.
- Within a scene, the densest completion scores **0.862 with a 7.8% zero-score rate**, vs. **0.818 and 12.7%** for the sparsest.
- The authors say this measures content, not trace length, but they do not show a length control. **The result is correlational.**

**Table VI — held-out GR-NavSim QA** (0–10; decisions scored by label accuracy ×10):

| Subtype | Qwen3-VL-8B | Kimi-K2.5 | GPT-5.4 | GRAVA |
|---|---:|---:|---:|---:|
| *Scene* | | | | |
| Summary | 6.09 | 6.03 | 7.08 | 7.47 |
| Traffic sign | 4.74 | 5.58 | 5.87 | 7.63 |
| Navigation | 6.21 | 6.44 | 6.33 | 6.67 |
| Congestion | 5.41 | 6.79 | 6.59 | 7.92 |
| Average | 5.61 | 6.21 | 6.47 | 7.42 |
| *Object perception* | | | | |
| Critical object description | 2.01 | 2.19 | 2.79 | 5.71 |
| Critical object identification | 1.32 | 2.80 | 3.71 | 5.69 |
| Average | 1.65 | 2.51 | 3.27 | 5.70 |
| *Object interaction* | | | | |
| Intention | 4.69 | 5.04 | **6.37** | 6.13 |
| Navigation interaction | 4.20 | 4.60 | 6.17 | 6.49 |
| Object interaction | 4.00 | 3.31 | 5.06 | 5.23 |
| Map interaction | 4.68 | 4.65 | 5.62 | 6.73 |
| Ego interaction | 4.27 | 4.03 | 4.94 | 6.00 |
| Average | 4.37 | 4.32 | 5.63 | 6.12 |
| *Decision (label accuracy)* | | | | |
| Object decision | 3.16 | 3.68 | 4.21 | 8.34 |
| Ego decision | 6.05 | 6.32 | 5.13 | 8.66 |
| **Overall** | 4.40 | 4.77 | 5.39 | **6.86** |

**How it is judged (App. F).**
- Open-ended answers are scored by a text-only **MiniMax-M3** judge at T=0. The judge sees the reference and the candidate, not the image.
- The judge scores grounding 0–2, semantic correctness 0–4, reasoning alignment 0–2 and factual consistency 0–2. It is told explicitly to reward precision over recall.
- **Perception localization**: $\mathrm{loc}=2\exp(-\lVert\hat p-p^*\rVert/5\,\mathrm m)$.
- **Decisions** use exact match over GRAVA's own label sets:
  - Object decisions: Follow / Yield / Stop / Nudge L/R / Overtake / Caution.
  - Ego decisions: a lateral label and a longitudinal label.

**How to read the comparison.** GRAVA is evaluated in-distribution on its own annotation format. The frontier models are zero-shot. The paper does not state whether the prompts listed the label set, so the decision rows (8.34 vs. 4.21) mostly measure knowing the taxonomy. The perception rows are the more meaningful: metric 3D localization from one front image, where every zero-shot model is below 3.8.

### Executable Planner analysis (§V-F)

Take low-score scenes that have a large reward gap between sampled actions. In most of them, **the high- and low-reward samples use the same primitive**. What differs is the endpoints, lateral offsets, speed profiles and CRAWL waypoints. So most of the RL signal is about *continuous parameters within a mode*, not about choosing the mode. No count is given for "most cases".

---

## Qualitative Results

![[nous_qualitative_compact_draft.png|Four GRA examples: (a) NAVSIM queue and crossing rider → CRAWL; (b) NAVSIM open-door roadside hazard → CURVE; (c) internal red signal and stopped lead vehicle → STOP; (d) internal construction-side clearance → CRUISE; each with grounding, interaction, decision and action panels]]

*Main-text qualitative figure (referred to as Fig. 8 in the text; the clipping attaches the Fig. 7 caption to it). NAVSIM: (a) queue and crossing rider → CRAWL; (b) open-door roadside hazard → CURVE. Internal: (c) red signal and stopped lead → STOP; (d) construction-side clearance → CRUISE.*

![[nous_v2_fig7_qualitative.png|Expanded NAVSIM examples, all PDMS 1.00: (a) signal-controlled STOP behind a lead pickup at 9.7 m; (b) CRAWL for a queue and crossing rider; (c) CRUISE as the queue resumes; (d) CURVE around an open roadside door, each with full THINK text and planner parameters]]

*Fig. S1: Expanded qualitative GRA planning examples on NAVSIM: (a) signal-controlled STOP, (b) CRAWL for a queue and crossing rider, (c) CRUISE as the queue resumes, and (d) CURVE around an open roadside door.*

![[nous_v2_fig8_internal_qualitative.png|Internal-benchmark examples: (a) STOP behind a lead vehicle at a red signal; (b) CRUISE with lateral clearance from a construction zone; (c) CRAWL in a wet-road queue; (d) CURVE for a left turn constrained by a pedestrian]]

*Fig. S2: Qualitative GRA planning on the internal benchmark: (a) STOP behind a lead vehicle at a red signal, (b) CRUISE with lateral clearance from a construction zone, (c) CRAWL in a wet-road queue, and (d) CURVE for a left turn constrained by a pedestrian.*

**Every Fig. S1 example is PDMS 1.00, and none shows a failure.** In the trace, the metric content arrives in exactly the form the density analysis rewards, e.g. "a black pickup truck at [0.3, 15.7] and 15.7 m ahead, is stationary at 0.0 m/s".

### The Appendix E transcript {#qa-transcript}

![[gra_qa_front_overlay_letters.png|Front-view NAVSIM frame with letter overlays R, F, D, U marking the four boxed objects used in the 30-QA transcript]]

*Fig. S5: Front-view observation shared by all 30 QA pairs in the complete GRA transcript. Labels map to normalized boxes: R (331,502),(418,629); F (482,494),(532,622); D (645,481),(684,600); U (187,523),(213,606).*

The 30 QAs are grouped by node type: scene context (4) → grounded object perception (4) → per-object interactions (6 per object) → object decisions (3) → ego decision (1). Each decision answer has `Think:` / `Chain:` / `Answer:` fields. The chain for the lead car is `scene_context.navigation | scene_context.summary -> R.map_interaction | R.motion_state -> R.ego_interaction -> Follow`.

**The showcase transcript contradicts itself in three places:**
1. **F and D are called "vehicle" in all four scene-context answers, and "PEDESTRIAN" in perception and everywhere after.**
2. **R is "approximately 15–20 meters ahead" in the navigation answer and "24.0m" in the perception answer**, within the same scene.
3. The scene-context answer says **"The future trajectory shows a smooth rightward curve consistent with a right turn"**. The ego decision says **"The trajectory intent explicitly specifies turn_right laterally and accelerate longitudinally."** A front-camera model at inference has no future trajectory. These are cognition targets that teach the model to *assert* it has seen $\tau^*$.

This is the example the authors chose to print. It is direct evidence that the agentic pipeline's cross-step consistency, its central claim for the data, is not guaranteed even on a curated case.

---

## What the Reasoning Trace Is {#what-the-trace-is}

GRAVA's intervention result is the cleanest in the wiki that the action head *reads* the CoT. [[sources/qwen-drive-1.0.md]] openly doubted this about its own trace. Three facts limit what the result means:

1. **The trace ends with the decision.** The serialized $r$ concludes with the ego decision (keep_lane + decelerate, turn_right + accelerate). That decision is a discretized description of the action's lateral and longitudinal mode, and in training it was **copied from `trajectory.intent`, i.e. from $\tau^*$** (Fig. 3 B1, Appendix E). Swapping low-reward reasoning for high-reward reasoning therefore swaps the stated maneuver. A 0.44 → 0.82 jump shows that the planner obeys the stated maneuver. It does not show that the grounded evidence *caused* the maneuver.
2. **The branch-and-merge structure is a target format, not an enforced computation.** Nothing at inference checks that the ego decision follows from the object-level decisions. The graph constraints are imposed on $G^*$ offline.
3. **The evidence that grounding itself matters is Table V's ladder** (coarse → objects → full). That is the right experiment, and it is on an internal benchmark. It has no NAVSIM counterpart, and it is single-run.

What GRAVA does establish, beyond the wiki's prior evidence:
- **Text-with-boxes reasoning beats coarse text reasoning by a large margin** once RL has been applied (+12.2 CDS on internal).
- **The action conditions on the trace strongly enough that editing the trace edits the behavior.**

Compare [[sources/nord.md]], where reasoning is unnecessary at 85.6 PDMS, and [[sources/adathinkdrive.md]], where CoT helps only in complex scenes. GRAVA's evidence comes from a benchmark *selected for* long-tail interaction. That setting is exactly where AdaThinkDrive predicts CoT should help, so the three results are compatible.

---

## Limitations

1. **The "best purely autoregressive" claim is a +0.18 margin, from one greedy run, against a table that omits DynVLA** (91.7, also purely AR, with dynamics tokens then action tokens) and ELF-VLA (91.0). In the wiki, 90.48 is mid-pack: below CLEAR/DA-WAM 93.7, DriveSuprim 93.5, FLARE 91.4, DiffusionDriveV2 91.2 and others.
2. **The reasoning ablation never touches NAVSIM.** Action-only, coarse and objects-only variants are reported only on a non-public 50K-clip benchmark with a self-defined score (CDS weights 0.45/0.35/0.20, undisclosed margins, undisclosed simulator). The public-benchmark claim (90.48) and the reasoning claim (+20.5% CDS) rest on disjoint evidence.
3. **Reasoning targets leak the expert trajectory.** The ego decision is wired from `trajectory.intent` (Fig. 3), the planning line describes $\tau^*$'s step spacing, and cognition QAs say "the future trajectory shows…". The model is trained to state facts it cannot observe. This also means the [intervention result](#what-the-trace-is) mainly shows maneuver-following.
4. **"About 60% of demonstrations" is narrow.** It counts planner warm-up cases (62,111 / 102,861). Active RL then screens *all* 102,861 scenes under PDMS reward, self-distillation samples the pool, and privileged LiDAR/surround-view annotation supplies every metric number in the traces. Table II's "415K planner examples" is not reconciled with the 62,111 figure.
5. **Most of the NAVSIM score is RL.** Pre-RL is 82.10, below DiffusionDrive and near UniAD, with DAC 91.2. RL adds +8.38 (DAC +6.4, TTC +6.1). The same 8-point jump appears in AutoVLA (+8.6) and SpanVLA (+8.2) from comparable pre-RL levels. GRAVA's specific contribution to the NAVSIM number is therefore hard to separate from the generic effect of GRPO on a weak SFT policy.
6. **"Active" selection is never ablated against plain GRPO or DAPO dynamic sampling.** The only evidence is the missing Fig. 4b curve and the Jaccard statistic. DAPO already drops zero-variance groups, so the added value of the $\tau_{hi}/\tau_{lo}/\tau_\sigma$ filter, and of re-screening, is unmeasured.
7. **There is a train/test gap in the screening numbers.** Screening the training pool gives a mean normalized greedy PDMS of **0.9101**, while the pre-RL checkpoint scores **82.1** on navtest. Which checkpoint was screened is not stated, and "normalized PDMS" is never defined. Either the screened checkpoint is post-RL, or there is roughly a 9-point train/test gap.
8. **The QA evaluation favors GRAVA by construction.** It is in-distribution on the model's own annotation format and label taxonomy, compared against zero-shot models, and scored by a text-only judge that never sees the image. The decision rows mostly measure taxonomy knowledge.
9. **Annotation noise appears in the showcase example.** The 30-QA transcript has a vehicle/pedestrian type flip and a 15–20 m vs. 24.0 m distance conflict. [See above](#qa-transcript).
10. **No latency, token counts or throughput are reported.** The trace is long (box tokens, numbers, per-object chains, 4,096-token limit), and the method needs 8B autoregressive decoding per frame. The conclusion names "more efficient reasoning" as future work.
11. **Only NAVSIM-v1 is used publicly.** There is no NAVSIM-v2/EPDMS, navhard, Bench2Drive, nuScenes or closed-loop public benchmark. Evaluation uses a single front camera and a single frame of observation, with motion history as text.
12. **The numerical-grounding result is correlational.** Densest vs. sparsest completions within a scene may differ in which objects they mention, not only in how many numbers they give. There is no length-matched control.
13. **Five figures are missing from the clipping**, including the Active RL curve, the intervention plot and the density plot. One caption is attached to the wrong image, and several in-text table and figure references are misnumbered.
14. **All results are single runs** with no seeds or variance, while the headline margin is 0.18 PDMS.
15. **No code, no data release, and no organizations are named.**

---

## Key Cross-References

- [[concepts/chain-of-thought-for-ad.md]] — grounded reasoning with box tokens. This is the wiki's cleanest reasoning-structure ladder (74.8 → 77.9 → 85.5 → 90.1 CDS) and its cleanest evidence that editing the trace edits the action. Caveat: the trace's decision is copied from $\tau^*$ at annotation time.
- [[concepts/rl-for-ad.md]] — Active RL, which filters recoverable scenarios with outer-loop re-screening (21.8% Jaccard between loops). It is a third pre-RL/post-RL data point at about +8 (82.1 → 90.5), and here RL buys **safety**, the reverse of the already-safe-policy pattern.
- [[concepts/navsim-benchmark.md]] — 90.48 on the v1 ladder, a clean baseline table, and the pre-RL row.
- [[concepts/action-tokenization.md]] — the Executable Planner, a mode-conditioned parametric vocabulary with a fixed decoder, worth +3.25 over direct waypoints.
- [[concepts/perception-for-planning.md]] — perception pushed *into the reasoning text* (box tokens plus metric states) rather than into a head. Grounded vs. ungrounded pre-training is +7.9 CDS.
- [[sources/curious-vla.md]] — the closest AR-VLA competitor (90.3). Its sample refresh against policy narrowing is the precursor of Active RL's re-screening.
- [[sources/dapo.md]] — the source of the clip-higher and dynamic-sampling machinery.
- [[sources/alpamayo-r1.md]] — trajectory-anchored causal traces with a flow-matching head, the counterpart to GRAVA's in-stream primitive action.
- [[sources/qwen-drive-1.0.md]] — doubted whether its trace was a rationale. GRAVA's intervention is the experiment that doubt called for, though it is confounded as described above.
- [[sources/dynvla.md]] — the purely AR VLA at 91.7 that GRAVA's claim omits.
