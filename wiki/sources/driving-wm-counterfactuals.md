---
title: "How Can Driving World Models Do Counterfactual Prediction?"
type: source-summary
sources: [raw/papers/How Can Driving World Models Do Counterfactual Prediction_.md]
related: [concepts/counterfactual-prediction.md, concepts/world-model-for-ad.md, concepts/bench2drive.md, concepts/hugsim-benchmark.md, concepts/nuscenes-waymo-evals.md, sources/simwam.md, sources/drivelaw.md, sources/epona.md, sources/policy-world-model.md, sources/vega.md, sources/drivewam.md, sources/driveva.md]
created: 2026-08-24
updated: 2026-08-24
confidence: high
---

**Paper**: How Can Driving World Models Do Counterfactual Prediction?
**Authors**: Jiaru Zhang (corresponding), Can Cui, Yi Xu, Xin Ye, Ruqi Zhang, Ziran Wang
**Orgs**: Purdue University + Bosch Center for Artificial Intelligence (the affiliation block in the raw clipping is garbled; per-author mapping is unreliable)
**arXiv**: 2608.11601v1
**Code / benchmark release**: not stated in the paper

---

## Summary

This is not a planning paper and it proposes no model. It is an argument that a capability the driving world-model literature routinely claims — counterfactual prediction — is **not what action-conditioned generation computes**, plus a benchmark that makes the gap measurable and a deliberately unglamorous pipeline that closes much of it.

The claim being audited is explicit in the field: Vista advertises "counterfactual reasoning ability" to "predict the counterfactual consequences caused by abnormal actions"; Drive-WM states "our Drive-WM can generate counterfactual events"; Waymo's world model blog demonstrates replaying a recorded drive under an alternative route; Genie 3 advertises promptable world events. In every case the procedure is the same: feed the model the shared history $H$ and an alternative action $a'$, generate, and call the output a counterfactual.

The paper's observation is that this procedure **never looks at what actually happened**. A counterfactual question is asked *after* an episode is recorded — "given that a car emerged from the side street, what would the camera have seen had I accelerated?" — and the factual continuation $F^{+}$ is therefore available evidence at query time. Direct action-conditioned prediction discards it, and so returns a fluent video that is consistent with the action but need not contain the event that defines the episode.

![[teaser-new-wide.png|Counterfactual prediction for one observed episode]]

**Figure 1**: The shared history shows the ego approaching an intersection. In the factual continuation, a red car emerges from a side street as the ego follows its recorded trajectory. The query asks what the camera would have recorded had the ego instead accelerated along target trajectory $a'$. The counterfactual ground truth shows the car would still have emerged. The direct prediction — using only the shared history and $a'$ — fails to preserve this event; the paper's method, which additionally uses the factual continuation, recovers the car near its counterfactual location.

Measured on 186 controlled CARLA cases, direct predictions from **Vista** (diffusion) and **DrivingWorld** (autoregressive) score a *recovered fraction* of **0.38 and 0.31** — i.e. closer to a replay in which the event never happened than to the matched counterfactual replay in which it did. A training-free evidence-transport pipeline over the same frozen backbones raises this to **0.70 and 0.67** and cuts LPIPS-to-ground-truth from 0.423 → 0.169 (Vista) and 0.291 → 0.211 (DrivingWorld).

---

## The Gap, Stated Causally

### Setup

A factual driving log $(F, a_{\mathrm{obs}})$ contains the full front-camera RGB video and its synchronized executed ego trajectory (position + heading per frame). The **shared history** $H$ is the synchronized prefix common to both the executed and target trajectories. $F^{+}$ denotes the RGB frames of $F$ after that prefix — the **factual continuation**. Although these frames sit later on the episode timeline, they are *observed evidence* when the query is posed. The query adds an alternative action $a'$ (target position and heading per frame).

**Scope**: a short horizon in which the alternative ego action changes the camera viewpoint and its direct consequences, while surrounding agents follow predetermined behaviors. The paper justifies this open-loop restriction on the grounds that driver perception–reaction times are on the order of a second, so within ~1 s of the ego action changing, surrounding agents have little time to respond. This is also what makes the question empirically checkable — a world whose other agents are scripted can be replayed.

### Rung 2 vs. rung 3

Letting $Y$ be the video following the history under a given action and $Y_{a'}$ its value under $a'$:

$$
\underbrace{p\big(Y_{a^{\prime}}\mid H,\,F^{+}\big)}_{\text{counterfactual prediction}}\quad\text{vs.}\quad\underbrace{p\big(Y\mid H,\,a^{\prime}\big)}_{\text{direct prediction}}
$$

Pearl's recipe for the left-hand side is abduction → action → prediction. With $G$ the mechanism mapping a world $w$ and an action to an outcome, the observation itself arose as $F^{+}=G(w,a_{\mathrm{obs}})$, and:

$$
\underbrace{w\sim p(w\mid H,F^{+})}_{\text{abduction}}\;\longrightarrow\;\underbrace{a^{\prime}}_{\text{action}}\;\longrightarrow\;\underbrace{Y_{a^{\prime}}=G(w,a^{\prime})}_{\text{prediction}}
$$

World models are trained on (history, action, future) triples to model $p(Y\mid H,a)$. Because $a'$ is *specified by the query* rather than observed as evidence, conditioning on it does not update the posterior over $w$ beyond $H$ — so at best the direct prediction equals the **interventional** $p(Y\mid H,\mathrm{do}(a'))$, which is rung 2. Written as mixtures over the world:

$$
p(Y\mid H,a^{\prime})=\int p\big(Y\mid w,a^{\prime}\big)\,p(w\mid H)\,dw
$$

$$
p(Y_{a^{\prime}}\mid H,F^{+})=\int p\big(Y\mid w,a^{\prime}\big)\,p(w\mid H,F^{+})\,dw
$$

The two share the mechanism $p(Y\mid w,a')$ and differ **only in the posterior over the world**. Whenever $F^{+}$ carries outcome information absent from $H$ — which is exactly the case for events revealed after the history, such as a car emerging from an occluded side street — the two distributions differ. The gap is therefore not a capacity problem, a fidelity problem, or a controllability problem. It is a **conditioning problem**, and no amount of scaling the generator fixes it.

---

## Method: Abduce → Transport → Complete → Combine

![[overview-replot.png|Overview of the four-stage evidence-transport pipeline]]

**Figure 2**: The factual driving log comprises RGB video $F$ and executed ego trajectory $a_{\mathrm{obs}}$; the target trajectory $a'$ specifies the counterfactual action. The four stages instantiate abduction, action, and prediction. (1) **Abduce** recovers the observed part of the realized world from the factual log. (2) **Transport** applies the target action by moving the camera while holding the world fixed, yielding evidence $E_t$ and support mask $M_t$ in the counterfactual view for each frame of the prediction window. (3–4) **Complete** and **Combine** implement prediction: the frozen world model generates unsupported regions, and Combine restores the transported evidence. The counterfactual ground truth $P$ is the replay of the same world under $a'$, used only for evaluation.

### The decomposition that assigns the work

The counterfactual view splits in two:

| Region | Posterior status | Who handles it |
|---|---|---|
| Surfaces also observed in $F$ | $p(w\mid H,F^{+})$ concentrates on the observed surfaces; determined up to monocular-depth error | **Geometry** (transport) |
| Regions occluded in $F$ or outside its field of view | Posterior retains genuine uncertainty; every consistent completion is admissible | **The frozen world model** (a prior over $p(w\mid H)$) |

This is the paper's cleanest conceptual move: it identifies precisely which part of the counterfactual a generative prior is *entitled* to invent, and hands the rest to geometry.

### Stages

**Abduce.** Depth Anything V2 Small estimates relative per-pixel depth on every frame of $F^{+}$. A road patch near the bottom centre of the image plus a camera-height constant of 1.8 m above the road converts relative depth to metres. The fixed camera model lifts each pixel to a coloured 3D point.

**Transport.** Because the camera is rigidly mounted, $a_{\mathrm{obs}}$ and $a'$ determine the relative pose per timestep between the camera that *did* film the scene and the one that *would have*. Forward splatting with a depth buffer reprojects the lifted points:

$$(E_{t},M_{t})=\mathrm{splat}\big(\mathrm{lift}(F^{+};a_{\mathrm{obs}}),\,\mathrm{cam}_{t}(a^{\prime})\big)$$

$M_t$ marks the **supported region** — pixels whose corresponding 3D point is visible in $F^{+}$. The time-aligned factual frame is the primary donor because it shows moving agents at the correct time; residual holes admit projections from other frames of $F^{+}$, favouring pixels whose projections agree across frames (**MF**, filling from multiple frames).

**Complete.** The frozen world model fills unsupported regions under $a'$. An input video is built from $E_t$ where $M_t=1$ and from the direct prediction $B$ elsewhere. For the diffusion backbone, sampling starts midway through denoising from a noisy encoding of that video (SDEdit-style), and after every step $i$ the evidence region is restored at the current noise level (RePaint-style):

$$x\leftarrow M\odot\big(z_{E}+\sigma_{i}\,\varepsilon\big)+(1-M)\odot x$$

where each cell of $M$ stores the *fraction* of its pixels covered by evidence, so partly covered latent cells are only partly pinned. For the autoregressive VQ backbone, tokens covered by transported evidence are held fixed and the rest are generated normally.

**Combine.** Encoding/decoding through the world model blurs transported pixels, so the reliable ones are restored with a feathered boundary:

$$\hat{Y}_{t}=\alpha_{t}\odot E_{t}+(1-\alpha_{t})\odot\mathrm{cc}(C_{t})$$

$\alpha_t = 1$ inside the reliable part of the support mask and decays to 0 near its boundary; $\mathrm{cc}$ applies a small, temporally smoothed colour adjustment matching the completed frame $C_t$ to the transported evidence.

Everything runs at inference time with all networks frozen. Case-specific inputs are $(F, a_{\mathrm{obs}})$ and $a'$; the camera setup is shared across cases.

---

## Benchmark

### Why simulation is unavoidable

Real driving records exactly one outcome per episode; the future under any alternative action is never recorded. CARLA can replay the same world, so the paper builds each case as **three runs of one placement** (ego + one event agent, both on predefined open-loop motion scripts), varying only the ego action and whether the event fires:

| Arm | Ego action | Event | Role |
|---|---|---|---|
| $F$ (with $a_{\mathrm{obs}}$) | executed trajectory | occurs | the factual log — the model's input |
| $P$ | target trajectory $a'$ | occurs | **counterfactual ground truth** — scoring only |
| $U$ | target trajectory $a'$ | never triggered | **null reference** (event-free) — scoring only |

Each arm is 25 frames at 10 fps, 576×320. All three share the 15-frame history $H$ (frames 0–14) and diverge only over the 10-frame prediction window (frames 15–24). Counterfactual edits **retime the ego along its factual path** — from the first prediction frame, the displacement between consecutive factual positions is scaled by 1.6 (accelerate), 0.4 (brake), or 0 (full stop). Only ego motion changes.

### Composition (Table 3)

| Scenario type | Total | Accel. | Brake | Full stop |
|---|---:|---:|---:|---:|
| side street | 60 | 26 | 17 | 17 |
| lead cuts in | 45 | 19 | 13 | 13 |
| lead brake | 81 | 27 | 27 | 27 |
| **Total** | **186** | **72** | **57** | **57** |

186 cases from **72 placements** (27 lead brake, 26 side street, 19 lead cuts in) across three towns — Town01 (60), Town03 (72), Town10HD (54). Every placement contributes an acceleration case; braking and full-stop cases cover subsets. Ten pedestrian-crossing cases were collected but excluded as too small a sample.

The three types are not equally informative, and the paper says so:

- **side street** — the headline type and the cleanest test: the event is *first revealed in $F^{+}$*, exactly over the prediction window, so nothing in $H$ predicts it.
- **lead cuts in** — secondary clean type.
- **lead brake** — a deliberate **confounded control**: the lead vehicle is already visible, so an accelerating ego makes it loom larger, mimicking the event signal through geometry alone.

### Capture and checks

CARLA 0.9.15, synchronous mode, fixed 0.05 s simulation step, two steps between stored frames. Single front RGB camera (576×320, 70° horizontal FOV) mounted 1.5 m forward of and 1.5 m above the ego actor origin of a `vehicle.tesla.model3`. In $U$ the event vehicle is present with the same starting state but its scripted manoeuvre is pushed beyond the captured window (the lead keeps speed, the side-street vehicle stays waiting, the cutting-in vehicle keeps its lane). Each case ships a `meta.json` with identifiers, map, action edit, spawn poses, scripted-motion parameters, ego positions/headings for $F$/$P$/$U$, a visibility summary, and minimum approach distance.

Two checks: a **geometric check** (forward 70° view, 60 m max range, event vehicle visible in the factual view for ≥2 frames) that gates inclusion — all 186 pass; and an **image check** comparing $P$ and $U$ for agreement at the end of the history and a localized difference during the prediction window — this passes 167 and **flags 19**, concentrated where the action edit weakens the visible event (e.g. a braking edit letting the lead recede) or where differences approach rendering noise. The flagged cases are retained; the paper states that scoring without them leaves the $B$-vs-Ours comparison unchanged, but does not report those numbers.

---

## Metrics

Two axes: does the prediction depict the **right world**, and does it depict it **well**.

### Recovered fraction (semantic, headline)

With $s(\cdot,\cdot)$ the cosine similarity of frame embeddings averaged over the prediction window, and $\Delta(\hat{Y})=s(\hat{Y},P)-s(\hat{Y},U)$ the preference for the counterfactual over the null:

$$\mathrm{Rec}(\hat{Y})=\frac{\Delta(\hat{Y})-\Delta(U)}{\Delta(P)-\Delta(U)}$$

Concretely, per case, with $\phi$ a frozen $L_2$-normalized encoder:

$$\Delta(\hat{Y})=\frac{1}{10}\sum_{t=15}^{24}\big[\cos(\phi(\hat{Y}_{t}),\phi(P_{t}))-\cos(\phi(\hat{Y}_{t}),\phi(U_{t}))\big],\qquad d=\frac{1}{10}\sum_{t=15}^{24}\big[1-\cos(\phi(P_{t}),\phi(U_{t}))\big]$$

so $\Delta(P)=d$, $\Delta(U)=-d$, and $\mathrm{Rec}=(\Delta+d)/(2d)$. Across a set of cases, $\Delta$ and $d$ are averaged **separately** and reported as $(\overline{\Delta}+\bar{d})/(2\bar{d})$. By construction $\mathrm{Rec}(U)=0$ (complete event omission), $\mathrm{Rec}(P)=1$, and $0.5$ means equally similar to both references. Scores outside $[0,1]$ are retained, not clipped. Encoders: DINOv2 ViT-B/14 ($\mathrm{Rec}_{\mathrm{D}}$) and CLIP ViT-L/14 ($\mathrm{Rec}_{\mathrm{C}}$).

### Perceptual fidelity

LPIPS (AlexNet backbone) between each predicted frame and the corresponding frame of $P$, averaged over the ten prediction frames. Because the true counterfactual exists, quality is measured as distance to *it* rather than distributionally (contrast FID/FVD in [[concepts/world-model-for-ad.md]]). LPIPS compares deep features spatially, so locally wrong content, seams, and blur all accrue distance. For DrivingWorld the references are resized to the model's 512×256 output before comparison.

---

## Results

### Qualitative (Figure 3)

![[exp.png|Qualitative comparison at a representative late frame of the prediction window]]

**Figure 3**: Upper block — a vehicle emerges from a side street under ego acceleration; lower block — a lead vehicle cuts in while the ego brakes to a full stop. One row per backbone (Vista, DrivingWorld). Columns: factual continuation $F^{+}$, the evaluation-only references $U$ (event-free null) and $P$ (counterfactual ground truth), then $B$ (direct prediction) and Ours. Ellipses mark the event location — solid where the event vehicle is present, dashed where absent. Within each row, $B$ and Ours share the same frozen model, history $H$, and target trajectory $a'$.

Despite fluent video, $B$ resembles $U$. Where a vehicle emerges from a side street, neither backbone shows it. Where the lead vehicle cuts in, Vista *removes* the vehicle and DrivingWorld leaves it in its original lane. Ours places the vehicle at approximately the location and pose shown in $P$ on both backbones; transport largely determines this geometry, and the remaining seams and mild warp artifacts show up in LPIPS.

![[qual_time.png|Temporal comparison across the prediction window for a side-street case]]

**Figure 4** (Appendix C.1): A side-street case in Town03 with Vista, at frames 15, 18, 21, and 24. Rows: $F^{+}$, $U$, $P$, $B$, Ours. The vehicle becomes visible in Ours by frame 18 and advances across the junction as it does in $P$; the $B$ frames grow blurrier over the window while Ours keeps the background sharp. The recovered event develops over time rather than appearing at a single frame.

### Quantitative (Table 1)

Means over five seeds; the $\pm$ value is the **maximum deviation** from the mean, not a standard deviation. Better value in bold.

| Scenario type | $\mathrm{Rec}_{\mathrm{D}}\uparrow$ $B$ | $\mathrm{Rec}_{\mathrm{D}}\uparrow$ Ours | $\mathrm{Rec}_{\mathrm{C}}\uparrow$ $B$ | $\mathrm{Rec}_{\mathrm{C}}\uparrow$ Ours | LPIPS $\downarrow$ $B$ | LPIPS $\downarrow$ Ours |
|---|---:|---:|---:|---:|---:|---:|
| *Vista (diffusion)* | | | | | | |
| side street | 0.29 ±.005 | **0.75** ±.003 | 0.25 ±.007 | **0.72** ±.001 | 0.415 ±.0075 | **0.172** ±.0003 |
| lead cuts in | 0.45 ±.011 | **0.73** ±.002 | 0.38 ±.056 | **0.73** ±.005 | 0.465 ±.0046 | **0.167** ±.0007 |
| lead brake | 0.50 ±.022 | **0.59** ±.005 | 0.41 ±.036 | **0.48** ±.002 | 0.407 ±.0029 | **0.167** ±.0005 |
| **Overall** | 0.38 ±.006 | **0.70** ±.002 | 0.33 ±.015 | **0.65** ±.002 | 0.423 ±.0043 | **0.169** ±.0003 |
| *DrivingWorld (autoregressive)* | | | | | | |
| side street | 0.25 ±.009 | **0.74** ±.004 | 0.23 ±.010 | **0.67** ±.003 | 0.288 ±.0014 | **0.212** ±.0002 |
| lead cuts in | 0.39 ±.027 | **0.67** ±.003 | 0.28 ±.012 | **0.71** ±.004 | 0.309 ±.0048 | **0.214** ±.0003 |
| lead brake | 0.37 ±.018 | **0.55** ±.003 | 0.23 ±.016 | **0.51** ±.006 | 0.284 ±.0021 | **0.208** ±.0003 |
| **Overall** | 0.31 ±.008 | **0.67** ±.001 | 0.24 ±.007 | **0.64** ±.001 | 0.291 ±.0013 | **0.211** ±.0002 |

**Reading the table.** $B$ sits below 0.5 in every cell except Vista's lead-brake $\mathrm{Rec}_{\mathrm{D}}$ (0.50) — meaning the direct prediction is generally *closer to the event-free replay than to the matched counterfactual replay*, consistently across two architectures and two encoders. The pattern tracks the scenario taxonomy exactly as the causal analysis predicts: $B$ is worst on **side street** (0.29 / 0.25), the type whose event is revealed only in $F^{+}$, and best on **lead brake** (0.50 / 0.37), the confounded control whose "event signal" is partly reproducible from geometry the model can already see. That internal gradient is stronger evidence than the headline averages, because it rules out "the models are simply bad at CARLA" as the sole explanation.

Ours is also weakest on lead brake (0.59 / 0.55) — Appendix C.2 explains why: $P$ and $U$ are most similar for lead-brake cases, so the denominator $d$ is small and small encoder fluctuations are magnified.

**Seed stability** (Appendix C.2): across five seeds, $B$'s overall $\mathrm{Rec}_{\mathrm{D}}$ spans 0.377–0.388 (Vista) and 0.305–0.321 (DrivingWorld), and overall LPIPS spans 0.419–0.426 and 0.290–0.292. Vista uses a deterministic per-case seed, DrivingWorld a global seed. The deviations for Ours are smaller still. The low direct-prediction scores are not a sampling accident.

---

## Ablations

### Component ablation (Table 2)

Tr = transport, MF = filling from multiple frames, Cm = completion, Cb = Combine. First row is the direct prediction $B$; last row is the full method.

| Tr | MF | Cm | Cb | Vista $\mathrm{Rec}_{\mathrm{D}}\uparrow$ | Vista $\mathrm{Rec}_{\mathrm{C}}\uparrow$ | Vista LPIPS $\downarrow$ | DW $\mathrm{Rec}_{\mathrm{D}}\uparrow$ | DW $\mathrm{Rec}_{\mathrm{C}}\uparrow$ | DW LPIPS $\downarrow$ |
|:-:|:-:|:-:|:-:|---:|---:|---:|---:|---:|---:|
| – | – | – | – | 0.38 ±.006 | 0.33 ±.015 | 0.423 ±.0043 | 0.31 ±.008 | 0.24 ±.007 | 0.291 ±.0013 |
| ✓ | ✓ | – | – | 0.68 ±.003 | 0.65 ±.002 | 0.195 ±.0002 | 0.67 ±.001 | 0.66 ±.001 | 0.238 ±.0002 |
| ✓ | – | ✓ | ✓ | 0.69 ±.005 | 0.64 ±.003 | 0.187 ±.0004 | 0.65 ±.003 | 0.63 ±.003 | 0.223 ±.0005 |
| ✓ | ✓ | ✓ | ✓ | **0.70** ±.002 | **0.65** ±.002 | **0.169** ±.0003 | **0.67** ±.001 | **0.64** ±.001 | **0.211** ±.0002 |

**Transport carries essentially all of the event signal.** Tr+MF alone — pixels warped from the factual continuation, with $B$ filling the rest and no world-model involvement beyond that — already reaches 0.68/0.65 (Vista) and 0.67/0.66 (DrivingWorld). Complete and Combine add at most +0.02 $\mathrm{Rec}_{\mathrm{D}}$ and, on DrivingWorld's $\mathrm{Rec}_{\mathrm{C}}$, *subtract* 0.02. What the last two stages buy is **fidelity**: LPIPS 0.195 → 0.169 (Vista) and 0.238 → 0.211 (DrivingWorld).

This is worth stating plainly, and the paper does not dodge it: the frozen world model contributes almost nothing to recovering the counterfactual event. A depth-warped copy-paste does that. The generator's job is confined to the regions where the evidence genuinely does not determine the answer — which is precisely the division of labour the causal analysis prescribed, but it also means the headline improvement is an indictment of the backbones rather than a demonstration of their latent counterfactual ability.

### Stage checkpoints (Table 4, Appendix C.3)

Means over five seeds; the intermediate stage deviates by at most 0.003 in the recovered fractions and 0.001 in LPIPS.

| Stage | Vista $\mathrm{Rec}_{\mathrm{D}}$ | Vista $\mathrm{Rec}_{\mathrm{C}}$ | Vista LPIPS | DW $\mathrm{Rec}_{\mathrm{D}}$ | DW $\mathrm{Rec}_{\mathrm{C}}$ | DW LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| Tr+MF | 0.68 | 0.65 | 0.195 | 0.67 | 0.66 | 0.238 |
| + Complete | 0.69 | 0.62 | 0.180 | **0.52** | **0.45** | 0.261 |
| + Combine (full) | 0.70 | 0.65 | 0.169 | 0.67 | 0.64 | 0.211 |

The DrivingWorld dip is the informative row: passing the full image through a VQ token encoder/decoder **destroys a third of the recovered signal** (0.67 → 0.52), and Combine recovers it by pasting the transported pixels back. Vista, whose completion runs in a continuous latent space, barely dips. This is a concrete measurement of discrete tokenization loss on evidence-bearing content — the same tokenizer-fidelity concern that shows up in [[concepts/action-tokenization.md]] for actions, here for pixels.

### Evidence-source controls (Table 5, Appendix C.4) — the paper's sharpest diagnostic

One seed, transport from a single frame, with the corresponding $B$ supplying remaining pixels. The question: does *any* extra pixel content help, or specifically evidence from the correct episode at the correct time?

| Evidence | Vista $\mathrm{Rec}_{\mathrm{D}}$ | Vista $\mathrm{Rec}_{\mathrm{C}}$ | Vista LPIPS | DW $\mathrm{Rec}_{\mathrm{D}}$ | DW $\mathrm{Rec}_{\mathrm{C}}$ | DW LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| matching evidence | **0.66** | **0.65** | 0.225 | **0.67** | **0.64** | 0.261 |
| five frames earlier | 0.40 | 0.44 | 0.276 | 0.41 | 0.47 | 0.323 |
| final history frame $F_{14}$ | 0.35 | 0.39 | 0.229 | 0.36 | 0.43 | 0.290 |
| different case (same scenario + action edit) | 0.62 | 0.58 | **0.564** | 0.64 | 0.59 | **0.556** |
| direct prediction $B$ | 0.38 | 0.32 | 0.419 | 0.31 | 0.24 | 0.291 |

Evidence from an earlier time — including the last frame of the shared history, which is exactly the information a direct prediction already has — lands at 0.35–0.41, statistically indistinguishable from $B$'s 0.31–0.38. Pasting pixels does not help; pasting *the right pixels from the right moment* does. That is the cleanest confirmation that the missing ingredient is abduction over $F^{+}$ and not extra visual context.

The **different-case** row is the one worth flagging as a metric caveat rather than a result. A donor frame from a different episode with the same scenario type and action edit still contains a similar vehicle event, and $\mathrm{Rec}_{\mathrm{D}}$ barely notices (0.62–0.64 vs. 0.66–0.67) — while LPIPS more than doubles (0.556–0.564). The recovered fraction is a *category*-sensitive metric, not an identity-sensitive one: it asks "is there an event of roughly this kind," not "is it *this* event, here, now." The paper uses the two metrics jointly and is explicit that both are needed, but anyone reusing $\mathrm{Rec}$ on its own should know it can be nearly satisfied by the wrong episode.

---

## Cost (Section 6.5, Appendix B)

Single A100, batch size 1, timings after model loading, averaged over three cases (one per scenario type).

| Component | Vista | DrivingWorld |
|---|---:|---:|
| Direct prediction $B$ (also consumed by Ours) | 47 s | 45 s |
| Depth estimation | ~3 s | ~3 s |
| Transport (CPU) | ~18 s | ~18 s |
| Completion | ~21 s | ~40 s |
| Combine | ~2 s | ~2 s |
| **Total, Ours** | **~90 s** | **~108 s** |
| Peak GPU memory ($B$ / Ours) | 39 / 49 GB | ~12 / ~12 GB |

Ours computes $B$ once because $B$ fills the unsupported regions of the Complete stage's input video — so the pipeline is a strict superset of the baseline, not an alternative to it. Roughly 2× the runtime, which the paper argues is acceptable for offline retrospective analysis. Environments: Ubuntu 24.04; PyTorch 2.0.1 / CUDA 11.8 (Vista) and PyTorch 2.5.1 / CUDA 12.1 (DrivingWorld).

---

## Implementation Details (Appendix B)

- **Resolution**: Vista uses native 576×320. DrivingWorld frames are resized to 512×284 (bicubic) and cropped 14 px top and bottom → 512×256, preserving aspect ratio.
- **Depth**: Depth Anything V2 Small. Camera height constant 1.8 m above the road surface (the 1.5 m mount height is measured from the ego actor origin, not the road). Pixels at sharp depth changes are removed via a relative depth-gradient threshold of 0.15. The source image is sampled at 2× resolution, keeping the nearest projected 3D point per target pixel; three rounds of neighbour averaging close 1–3 px holes.
- **MF**: a pixel with one available projection is kept; with several projections, if the average spread across channels is below 28 intensity levels the median RGB is kept, otherwise the pixel stays unsupported. Donor brightness/contrast is matched to accepted evidence near the boundary before filling.
- **Complete (Vista)**: native EDM sampler, 25 steps, starting at schedule index 14 ($\sigma\approx6.4$); the 15 history frames stay fixed; the evidence mask is resized to the latent grid where each 8×8 cell stores its covered fraction.
- **Complete (DrivingWorld)**: an evidence token is fixed when transported evidence covers ≥60% of its 16×16 image patch.
- **Combine**: the support mask is shrunk by 2 px to form the $\alpha_t=1$ region; the transition is blended over 12 px (Vista) / 24 px (DrivingWorld — wider, to hide 16×16 token boundaries). The colour map applies a per-channel contrast scale in $[0.8,1.25]$ and brightness shift within ±25 intensity levels, averaged with the previous frame's adjustment at weight 0.5 for temporal stability.
- All constants were fixed during initial development and kept for the reported runs.

---

## Limitations

The paper's own (Appendix D) come first; the rest are this wiki's reading.

1. **Scripted agents are load-bearing, and the method inherits their assumption.** Scripting is what makes $P$ obtainable at all, but transported evidence *preserves behaviour that the counterfactual action would have changed*. The paper's own example: a pedestrian who would have stopped had the ego slowed keeps walking in the transported evidence. Over any horizon where agents react, evidence transport is not merely incomplete — it is **wrong in a specific, confidently-rendered way**, which is arguably more dangerous than the direct prediction's omission. Proposed remedy is posterior predictive checks on the abduced world plus extension to reactive suites ([[concepts/bench2drive.md]], nuPlan).
2. **Both backbones are evaluated outside their training render domain.** Vista and DrivingWorld are trained on real driving video and scored on CARLA renders. The paper argues the causal analysis is domain-independent and that each model runs under its authors' released protocol, and calls for replication with a world model trained on the benchmark's render domain. This is the single largest threat to the headline numbers: a domain-shifted generator may under-use its conditioning for reasons unrelated to the causal gap. The internal gradient across scenario types (side street ≪ lead brake) is the main evidence that domain shift is not the whole story, since all three types share the render domain.
3. **The method is retrospective by construction.** It reads $F^{+}$, which exists only after the episode is recorded. It serves incident analysis, safety auditing, and liability assessment — not decision-time planning. Extending to the decision-time case, where the outcome is still unobserved, is left open. This bounds how much the result transfers to the planning literature: it does **not** show that a planner could do better counterfactual rollout, because a planner has no $F^{+}$.
4. **The world model contributes little of the measured gain.** Table 2's Tr+MF row gets 0.68/0.67 with no completion at all. Honest framing by the authors ("a deliberately simple, training-free pipeline… a constructive check"), but it means the paper demonstrates *that evidence closes the gap*, not that world models can be made to do abduction.
5. **$\mathrm{Rec}$ is category-sensitive, not identity-sensitive** (Table 5, different-case row: 0.62–0.64 vs. 0.66–0.67 matching, while LPIPS doubles). Reported alone it would nearly certify a wrong-episode paste.
6. **The lead-brake type is confounded by design and also has the smallest metric denominator**, so a third of the benchmark (81 of 186 cases) is the least informative third and simultaneously the noisiest.
7. **19 of 186 cases fail the post-hoc image check and are retained.** The claim that excluding them leaves the comparison unchanged is asserted without numbers.
8. **The ceiling is 0.70, not 1.0**, and the paper offers no decomposition of the residual between monocular-depth error, splatting artifacts, completion quality in unsupported regions, and encoder noise.
9. **Two open backbones only.** The industrial claims that motivate the paper — Waymo's world model, Genie 3 — are cited but untestable, and Drive-WM, whose counterfactual claim is quoted directly, is not evaluated. Vista and DrivingWorld are the accessible proxies, not the strongest current systems.
10. **Ten pedestrian cases were collected and dropped.** Vulnerable-road-user counterfactuals are exactly the case where the scripted-agent assumption is least defensible and the application (liability) most consequential, so their absence is more than a sample-size footnote.
11. **No stated release** of benchmark, code, or generated data, which limits reuse of the 186 cases as a shared protocol — the same problem [[concepts/physicalai-av-benchmark.md]] notes for other recent evaluation sets.

---

## Key Cross-References

- **Concept page**: [[concepts/counterfactual-prediction.md]] — the causal ladder, the abduction requirement, the recovered-fraction metric, and which wiki claims this paper constrains.
- **World-model claims**: [[concepts/world-model-for-ad.md]] — the page's long-standing statement that world models "enable counterfactual reasoning: if I turn left, the future should look like X" is exactly the conflation audited here, and has been corrected.
- **The wiki's open thread on test-time imagination**: [[sources/simwam.md]] and [[sources/drivelaw.md]] independently found no planning benefit from conditioning on generated futures on NAVSIM, leaving "counterfactual maneuver evaluation" as the untested escape hatch for imagine-then-act designs. This paper attacks that escape hatch from the other side: on a benchmark built specifically for counterfactual evaluation, action-conditioned generation does not produce counterfactuals at all. See the [test-time imagination](../concepts/world-model-for-ad.md#test-time-imagination) synthesis.
- **Evaluation metrics**: [[concepts/nuscenes-waymo-evals.md]] and the FID/FVD tables in [[concepts/world-model-for-ad.md]] — this paper's protocol is the wiki's first example of scoring a generated video against a *matched ground-truth counterfactual* rather than against a distribution.
- **Backbones evaluated**: Vista and DrivingWorld are both un-ingested (Vista is the wiki's 7th most-cited un-ingested method); they appear here only as frozen evaluation subjects.
- **Counterfactual in the other sense**: [[concepts/vlm-domain-adaptation.md]] records OmniDrive's "counterfactual planning" VQA task (AutoMoT: frozen VLM 18.20 → fine-tuned 67.80). That is a language-space reasoning benchmark, not a rung-3 prediction task — see the disambiguation in [[concepts/counterfactual-prediction.md]].
