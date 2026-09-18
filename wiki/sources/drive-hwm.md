---
title: "Drive-HWM: Hierarchical World Models for Dynamic-Latent Guided Autonomous Driving"
type: source-summary
sources: [raw/papers/Drive-HWM_ Hierarchical World Models for Dynamic-Latent Guided Autonomous Driving.md]
related: [concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/foundation-backbones-for-ad.md, concepts/dual-system-vla.md, concepts/best-of-n.md, concepts/visual-tokenization.md, concepts/action-tokenization.md, concepts/selection-based-planning.md, concepts/counterfactual-prediction.md, concepts/hugsim-benchmark.md, sources/drivevla-w0.md, sources/coworld-vla.md, sources/wa-jepa.md, sources/drivefuture.md, sources/drivelaw.md, sources/geowam.md, sources/geoworldad.md, sources/da-wam.md, sources/auto-jepa.md, sources/drive-jepa.md, sources/clear.md, sources/simwam.md, sources/adaptive-wam.md, sources/brainwam.md, sources/foresight.md, sources/unified-driving-tokens.md, sources/latent-wam.md, sources/deepsight.md, sources/drivesuprim.md, sources/diffusiondrive.md, sources/autovla.md, sources/adathinkdrive.md, sources/epona.md, sources/lwdrive.md, sources/wcog-vla.md, sources/drivewam.md, sources/hybriddriveVLA.md, sources/recogdrive.md, sources/sgdrive.md, sources/driveva.md, sources/pair-drive.md, sources/wam-diff.md, sources/explorevla.md, sources/had.md, sources/dreameraD.md, sources/dynvla.md]
created: 2026-09-14
updated: 2026-09-14
confidence: medium
---

**Paper**: Drive-HWM: Hierarchical World Models for Dynamic-Latent Guided Autonomous Driving
**Authors**: Zhaoxin Fan, Tianbao Zhang, Wenjun Wu, Xiaofeng Wang, Yeying Jin, Jian Zhao, Zheng Zhu, Shuicheng Yan
**Orgs**: not stated in the clipping — funding acknowledges Beihang University (Beijing Advanced Innovation Center for Future Blockchain and Privacy Computing), a National S&T Major Project, and Beijing NSF grants
**arXiv**: 2609.03572v1
**Code**: none announced

---

## Source Integrity Note

The clipping is complete from abstract through acknowledgments. **Six of the seven figures are present** (Figs. 1, 2, 3, 5, 6, 7). **Figure 4 — the bar chart comparing BEV / depth / RGB / optical-flow prediction targets for the slow world model — is referenced twice in the text and captioned, but no image exists in the clipping**, so the only record of that sweep is the prose claim that "optical flow achieves the best results." That is the paper's most important ablation and its numbers are not recoverable from this source. All eight tables are intact. Author photograph links are dead external URLs and are not reproduced.

---

## The Headline Problem: This Paper Contains Two Different Result Sets {#two-result-sets}

**Every table reports one set of numbers for Drive-HWM. Every sentence of prose reports a different, uniformly lower set.** This is systematic, not a stray typo — it holds across five separate passages and both benchmarks, and the prose set is internally arithmetically consistent with itself.

| Quantity | **Every table** (I, III, IV, V, VII, VIII) | **Every sentence** (§IV-B, §IV-D) | Δ |
|---|---:|---:|---:|
| NAVSIM-v1 NC | 99.6 | 99.5 | 0.1 |
| NAVSIM-v1 DAC | 99.0 | 98.4 | 0.6 |
| NAVSIM-v1 TTC | 98.5 | 98.0 | 0.5 |
| NAVSIM-v1 Comf. | 100.0 | 100.0 | 0 |
| NAVSIM-v1 EP | 89.0 | 88.5 | 0.5 |
| **NAVSIM-v1 PDMS** | **93.8** | **93.3** | **0.5** |
| NAVSIM-v2 DDC | 99.4 | 99.4 | 0 |
| NAVSIM-v2 EP | 88.5 | 88.1 | 0.4 |
| NAVSIM-v2 LK | 96.5 | 94.7 | 1.8 |
| NAVSIM-v2 EC | 87.5 | 86.4 | 1.1 |
| **NAVSIM-v2 EPDMS** | **86.4** | **86.2** | **0.2** |

**How the prose set is recovered.** §IV-B never prints Drive-HWM's v2 submetrics directly; it prints *deltas* against DriveVLA-W0, and those deltas reconstruct a complete row:

- v1: "improves NC, DAC, TTC, comfort, EP, and PDMS by 0.2, 1.0, 1.0, 0.1, 0.2, and 0.3" over DriveVLA-W0‡ (99.3 / 97.4 / 97.0 / 99.9 / 88.3 / 93.0) → **99.5 / 98.4 / 98.0 / 100.0 / 88.5 / 93.3**. The same passage then says "our DAC of 98.4," confirming it.
- v2: "gains of 1.4 points in DDC, 1.7 points in EP, 1.5 points in LK, and 27.5 points in EC" over DriveVLA-W0 (98.0 / 86.4 / 93.2 / 58.9) → **99.4 / 88.1 / 94.7 / 86.4**. The same passage then says "a comparable EP of 88.1, only 0.3 points below the best result" — and note that under the *table* value of 88.5, Drive-HWM's EP would be the best in Table II rather than third.

Three further passages use the same set: "improving PDMS from 93.0 to 93.3" (§IV-B, efficiency), "decreases NC, DAC, and PDMS from 99.5, 98.4, and 93.3" (§IV-D, Table IV), and "FiLM ... reaching 99.5 NC, 98.4 DAC, and 93.3 PDMS" (§IV-D, Table VII). **The prose set never appears in a table and the table set never appears in a sentence.**

### Why this is not cosmetic

**It changes the size of the paper's central claim by a factor of 2.7.** Table IV's `Fast only` row — remove the slow world model — scores 93.0. So the entire measured contribution of hierarchical slow–fast world modelling is:

| Reading | Full model | Fast only | Value of the slow world model |
|---|---:|---:|---:|
| Prose | 93.3 | 93.0 | **+0.3** |
| Tables | 93.8 | 93.0 | **+0.8** |

The paper's own text picks the smaller one and describes it as validating the design. +0.3 sits inside the range this wiki has repeatedly found for named world-model mechanisms — [[sources/da-wam.md]]'s per-candidate future is +0.15, [[sources/foresight.md]]'s world model under vanilla attention is +0.3 — and close to the ~0.2 floor that [[concepts/navsim-benchmark.md]] treats as meaningful given measured sampler noise.

**It also decides a leaderboard position.** 93.8 would be the highest non-BoN NAVSIM-v1 score in this wiki, 0.1 above [[sources/clear.md]] and [[sources/da-wam.md]] at 93.7. 93.3 exactly ties [[sources/drive-jepa.md]] and sits below three other methods. **A reader cannot determine from this paper which of those two statements is true**, so this page records both. Confidence is `medium` for that reason alone.

The most likely mechanical explanation is that the tables were refreshed after a final run and the prose was not — the deltas all point the same way, and the largest (LK +1.8) is on a metric the prose treats as a weakness. That is a guess. The paper offers no comment, and its abstract makes no numerical claim at all.

---

## Summary

Drive-HWM asks a scheduling question rather than a representation question: **at what rate should a driving world model predict the future, and at what rate should it decide?** Its answer is two world models with different update periods and, more interestingly, **different prediction targets**:

| | **Slow world model** | **Fast world model** |
|---|---|---|
| Runs | once every $N=8$ steps | every step |
| Backbone | V-JEPA (see [naming](#vjepa-naming)) | **Emu3-8B** |
| Predicts | $K=8$ future **optical-flow** fields, in parallel from one history | the **next RGB frame** (as visual tokens) + the immediate action |
| Reaches the planner as | the pre-decoder hidden state (**Dynamic-Aware Latent**), via FiLM | action tokens |
| Best target, measured | **optical flow** (Fig. 4, numbers missing) | **RGB** (Table VIII: 93.8 vs. flow 93.5, depth 93.6, none 93.1) |
| Latency | 25.6 ms | 81.6 ms |

The interface between them is deliberately *not* the decoded flow field. Flow is a training signal for the decoder; what crosses into the fast model is $q_t^s = \hat d^s_{t+1\mid\tau(t)}$, the latent that had to be sufficient to reconstruct flow, injected as FiLM scale-and-shift on Emu3's hidden states rather than as extra tokens.

**Headline: 93.8 PDMS / 86.4 EPDMS in the tables, 93.3 / 86.2 in the prose** ([above](#two-result-sets)), from **a single front camera at 256×144**, no RL, no scorer, no trajectory anchors, no best-of-$N$.

**Four readings this page arrives at:**

1. **The target-role split is the paper's real contribution, and it is new to this wiki.** Every world-model planner ingested so far picks *one* prediction target. Drive-HWM runs two objectives at two horizons and ablates the target for each independently — finding the long-horizon branch wants motion (flow) and the short-horizon branch wants appearance (RGB). [[sources/coworld-vla.md]] ran four targets in parallel, but all at one horizon. See [Target by Role](#target-by-role).
2. **The paper's central mechanism — high-frequency, observation-grounded re-decision — cannot be exercised on its only benchmark.** NAVSIM is non-reactive and single-shot. With $N=K=8$ at NAVSIM's 2 Hz / 8-pose convention, the slow model runs **exactly once per evaluated scenario**, and no new observation ever arrives for the fast model to be grounded in. See [What "Every Timestep" Means Here](#every-timestep).
3. **What one action *is* is never defined, and the two possible answers differ by 8× in latency.** $L_a$ is never given, and each reading contradicts a different claim the paper makes.
4. **Two of Table IV's four ablation rows have PDMS identical to DriveVLA-W0's two published rows**, one of them across all three reported metrics. See [Ablation Baseline Provenance](#ablation-provenance).

---

## Positioning

![[firgure_teaser.png|Three world-modelling paradigms: action-only prediction, joint sparse-action plus dense-visual prediction, and Drive-HWM's separated slow and fast world models]]

**Fig. 1**: (a) Action-only models rely on sparse action prediction. (b) Joint-prediction world models perform sparse action prediction and dense visual prediction at the same temporal scale. (c) Drive-HWM separates slow and fast world modelling.

The argument in §I is a temporal-mismatch argument, stated cleanly:

> "Future representation prediction aims to capture how a driving scene evolves over an extended horizon... Action prediction, by contrast, is local and high-frequency: each action should be grounded in the latest observation to respond promptly to changing conditions. Predicting a long action sequence at once may accumulate execution errors and gradually deviate from the intended behavior."

**The related-work section draws the right boundary against the existing fast–slow literature**, and this is the sentence that earns the paper a place on [[concepts/dual-system-vla.md]]:

> "Several driving VLAs further adopt fast–slow designs to balance decision quality and computational cost: routine scenarios use direct action generation, whereas challenging situations invoke more expensive semantic or chain-of-thought reasoning. These methods separate reasoning modes and allocate computation according to scenario complexity. In contrast, our hierarchy separates explicit future representation prediction from action generation according to their temporal roles."

That is a genuine distinction. [[sources/autovla.md]], [[sources/adathinkdrive.md]] and [[sources/deepsight.md]] route by *scene difficulty*; [[sources/drivewam.md]] and DualDriveVLA ([[sources/hybriddriveVLA.md]]) split by *module cost*. Drive-HWM splits by **horizon**, an axis none of them vary, and it is the first entry on that page whose slow branch emits no decision of any kind.

---

## Method

![[framework 2.png|Drive-HWM slow-fast framework: a V-JEPA slow world model predicting multi-horizon optical flow into Dynamic-Aware Latents, FiLM-injected into an Emu3-8B fast model whose autoregressive expert predicts the next action and next-frame visual tokens]]

**Fig. 2**: The slow world model, updated every $N$ steps, employs VL-JEPA to predict multi-horizon optical flow under dynamic supervision, producing Dynamic-Aware Latents $\mathbf{Z}^{d}_{(t+1)\rightarrow(t+k)}$. At every step the temporally aligned dynamic latent is injected via FiLM into the fast world model built on Emu3-8B. The autoregressive expert integrates the current instruction, observation, and action history to jointly predict the next action and next-frame visual tokens.

### Hierarchical factorization

With $\tau(t)=N\lfloor t/N\rfloor$ the most recent slow update and $\mathcal{R}=\{0,N,2N,\ldots\}$:

$$p_{\Theta}\!\left(\{a_{t},z^{f}_{t+1}\}_{t=0}^{T-1},\{Z^{s}_{\tau}\}_{\tau\in\mathcal{R}}\,\middle|\,o_{\leq T}\right)=\prod_{\tau\in\mathcal{R}}p_{\theta_{s}}\!\left(Z^{s}_{\tau}\mid\mathcal{H}_{\tau}\right)\prod_{t=0}^{T-1}p_{\theta_{f}}\!\left(a_{t},z^{f}_{t+1}\mid\mathcal{H}_{t},q_{t}^{s}\right)$$

The slow prediction is *reused* across the $N$ fast steps in its interval; the fast model is the only component that sees new evidence.

### Slow world model — Dynamic-Aware Latents

The motivation is an information-density argument against RGB feature prediction:

> "such features are typically dominated by appearance semantics and spatial content, while the information most relevant to driving — including ego-motion, object displacement, and their temporal evolution — may occupy only a small portion of the representation."

$K$ flow fields are predicted **in parallel from one history**, not recursively:

$$\left\{\hat{d}^{s}_{\tau+k\mid\tau},\hat{\mathcal{F}}_{\tau+k\mid\tau}\right\}_{k=1}^{K}=S_{\theta_{s}}(h_{\tau}),\qquad \hat{\mathcal{F}}_{\tau+k\mid\tau}=D_{\mathrm{flow}}\left(\hat{d}^{s}_{\tau+k\mid\tau}\right)$$

$$p_{\theta_{s}}\left(\mathcal{F}_{\tau+1:\tau+K}\mid\mathcal{H}_{\tau}\right)=\prod_{k=1}^{K}p_{\theta_{s}}\left(\mathcal{F}_{\tau+k}\mid\mathcal{H}_{\tau},k\right)$$

with targets $\mathcal{F}_{\tau+k}=\operatorname{Flow}(o_{\tau+k-1},o_{\tau+k})$ from an **unnamed** off-the-shelf estimator, and a masked robust loss

$$\ell_{\mathrm{flow}}(\hat{\mathcal{F}},\mathcal{F})=\frac{\sum_{u}M(u)\,\rho(\hat{\mathcal{F}}(u)-\mathcal{F}(u))}{\sum_{u}M(u)+\epsilon},\qquad \mathcal{L}_{\mathrm{flow}}=\frac{1}{K}\sum_{k=1}^{K}w_{k}\,\ell_{\mathrm{flow}}(\cdot)$$

**The parallel multi-offset formulation is the right call and is under-argued.** It is the direct structural answer to the accumulation problem [[sources/epona.md]] attacks with chain-of-forward training and [[sources/drivewam.md]] attacks with chunked rollout: if every offset is predicted from the same clean history, there is no recursive warp for early errors to ride. The cost is that the $K$ predictions are conditionally independent given $h_\tau$ and therefore need not describe one *coherent* future — which is precisely the property §I claims for them ("a coherent future evolution, such as the progression of a complete left turn"). The paper does not notice the tension and nothing measures cross-offset consistency.

**The interface choice is the reusable part.** Raw flow is withheld from the fast model on the grounds that displacement is uninterpretable without context — "similar image displacement may correspond to different driving implications depending on whether it originates from a vehicle, a pedestrian, the road surface, or camera motion" — so the pre-decoder hidden state $D^s_\tau=\{\hat d^s_{\tau+1\mid\tau},\ldots,\hat d^s_{\tau+K\mid\tau}\}$ crosses instead. **Flow is the objective; the latent is the interface.** That is the same shape as [[sources/geowam.md]]'s point-map supervision feeding a latent planner and [[sources/coworld-vla.md]]'s Wan-as-differentiable-critic-on-one-token, and it generalizes: any dense pseudo-labelled target can be used this way without ever putting the dense field in the inference path.

### Fast world model — FiLM conditioning on Emu3-8B

$$H_{t}=B_{\theta_{b}}(V_{\leq t},A_{<t}),\qquad (\gamma_{t}^{(l)},\beta_{t}^{(l)})=G_{\mathrm{FiLM}}^{(l)}(q_{t}^{s}),\qquad \widetilde{H}_{t}^{(l)}=(1+\gamma_{t}^{(l)})\odot\operatorname{LN}(H_{t}^{(l)})+\beta_{t}^{(l)}$$

The stated reason for FiLM over concatenation is that it "allows the predicted dynamics to adaptively rescale and shift the feature channels used for action prediction **without changing the token organization of the pretrained backbone**" — a real consideration for a discrete-token backbone whose sequence layout is load-bearing, and one that concatenation disturbs. Which layers $l$ are modulated is never stated.

A driving-specific autoregressive expert $E^{\mathrm{AR}}_{\theta_e}$ then produces $R_t$, from which action tokens and next-frame visual tokens are decoded:

$$\mathcal{L}_{\mathrm{act}}=-\frac{1}{L_{a}}\sum_{j}\log p_{\theta_{f}}(a_{t,j}\mid a_{t,<j},R_{t}),\qquad \mathcal{L}_{\mathrm{img}}=-\frac{1}{L_{v}}\sum_{i}\log p_{\theta_{f}}(v_{t+1,i}\mid v_{t+1,<i},A_{t},R_{t})$$

**Note the conditioning order**: $V_{t+1}$ is predicted *given* $A_t$. The next-frame objective is therefore action-conditioned — interventional in the [[concepts/counterfactual-prediction.md]] sense, rung 2 — a forward model of the consequence of the chosen action rather than a passive forecast. The paper gives three reasons for it, and the third is the architecturally important one: "because the future visual prediction is conditioned on the DAL, it encourages the fast model to make effective use of the dynamic context instead of ignoring it during action training." The auxiliary loss is partly there to stop FiLM from being tuned toward zero.

### §III-D contradicts §III-C on what the fast loss is {#loss-contradiction}

§III-C defines the fast objective as a token-level cross-entropy over next-frame visual tokens:

$$\mathcal{L}_{\mathrm{fast}}=\mathcal{L}_{\mathrm{act}}+\lambda_{\mathrm{img}}\mathcal{L}_{\mathrm{img}}$$

§III-D, one page later, defines the same symbol as a **latent distance**:

$$\mathcal{L}_{\mathrm{fast}}=\mathcal{L}_{\mathrm{act}}(\hat{a}_{t},a_{t})+\lambda_{f}\,d(\hat{z}^{f}_{t+1},y_{t+1}),\qquad \mathcal{L}=\lambda_{s}\mathcal{L}_{\mathrm{slow}}+\mathcal{L}_{\mathrm{act}}+\lambda_{f}\mathcal{L}_{\mathrm{vis}}$$

and opens by describing the training signal as "long-horizon latent prediction, short-horizon latent prediction, and action generation" — **no mention of optical flow or visual tokens anywhere**. $\mathcal{L}_{\mathrm{slow}}$ is never related to $\mathcal{L}_{\mathrm{flow}}$, and $d(\cdot,\cdot)$ is never specified.

These are not two views of one loss. A cross-entropy over a discrete codebook and a regression onto a continuous latent are the two arms of the comparison [[concepts/world-model-for-ad.md]]'s objective-form section is built around, where [[sources/wa-jepa.md]] measures deterministic regression on scene latents as **worse than predicting nothing** (90.7 vs. 91.1). The most economical reading is that §III-D is residue from an earlier latent-regression version of the method. **None of $\lambda_{\mathrm{img}}$, $\lambda_s$, $\lambda_f$, $w_k$, $\rho$, $L_a$, $L_v$, or $d$ is ever given a value or a form.**

### What runs at inference

| Component | Trained on | Present at inference |
|---|---|---|
| Slow V-JEPA predictor | nuPlan (10K steps) | **Yes** — every $N$ steps |
| Flow decoder $D_{\mathrm{flow}}$ | nuPlan | **No** — only $\hat d^s$ is used |
| Emu3-8B + FiLM + AR expert | nuPlan, then NAVSIM (6K steps) | Yes |
| Next-frame visual decoding | nuPlan | **No** — "future image generation can be omitted" |

**Both world-model objectives are training-time-only, and both are confined to nuPlan pretraining.** §IV-A states the NAVSIM stage uses "the action prediction loss" alone, so neither the flow loss nor the next-frame loss ever sees NAVSIM data. This places Drive-HWM in a position no existing entry occupies: the *decoders* are discarded like [[sources/simwam.md]] and [[sources/drivevla-w0.md]], but the **slow predictor keeps running at inference** to produce a latent whose only supervision came from a different dataset. It is neither the imagine-then-act camp ([[sources/foresight.md]]) nor the pure training-time camp — and unlike [[sources/coworld-vla.md]], where a supervised *token* survives a discarded generator, here a whole *predictor* survives its discarded decoder.

### Training configuration

| Item | Value |
|---|---|
| Pretrain | nuPlan, 10K steps, full objective $\mathcal{L}_{\mathrm{total}}$ |
| Fine-tune | NAVSIM, 6K steps, **action loss only** |
| $N=K$ | 8 |
| Input resolution | **256 × 144** |
| Hardware | 8 × A100, global batch 48 |
| Optimizer | AdamW, lr $2\times10^{-4}$, cosine |

No parameter counts for the slow model, flow decoder, AR expert, or FiLM modules. No GPU-hours. No code.

---

## What "Every Timestep" Means Here {#every-timestep}

This is the largest unexamined gap in the paper, and it is structural rather than a matter of degree.

**NAVSIM is non-reactive and single-shot.** The agent receives one observation (with history), emits one 4-second trajectory, and a PDM simulator scores it. No future observation is ever delivered. NAVSIM's convention is 2 Hz over 4 seconds — **eight poses** — and Drive-HWM sets $N=K=8$. So on its only benchmark:

- the slow world model runs **exactly once per evaluated scenario**, covering precisely the planning horizon;
- the fast model's eight steps consume **no new observations**, because none exist.

The abstract's claim — "one-step action generation allows decisions to be continuously updated as new observations arrive" — is a closed-loop property. **No experiment in this paper measures it.** Neither does any experiment measure the error accumulation the paper attributes to long autoregressive action rollouts, since that failure mode also manifests only under sequential execution.

**And the definition of one action is load-bearing but absent.** $A_t=(a_{t,1},\ldots,a_{t,L_a})$ is said to hold "tokens [that] encode the executable driving output," and $L_a$ is never given. Two readings exist and each breaks something:

| Reading | What $A_t$ is | Consequence |
|---|---|---|
| **(a)** $A_t$ is the whole 4 s trajectory | one forward pass, 81.6 ms | "Immediate action," "one-step action generation," and the critique of long AR rollouts all become inaccurate descriptions of the model |
| **(b)** $A_t$ is one pose; eight are rolled out | 8 × 84.8 ms ≈ **678 ms** per trajectory | Steps 2–8 have no observation, so the AR expert must consume **its own predicted next frame** — meaning next-frame generation *cannot* be omitted at inference, contradicting §III-C, and Table III's latency comparison is off by 8× |

Table III's arithmetic presumes (b) — amortizing $T_s/N$ across $N$ fast steps only makes sense if there are $N$ of them — while its headline comparison presumes (a), since DriveVLA-W0's 117.8 ms is a per-trajectory figure. Under (b), Drive-HWM is roughly **5.8× slower** than the baseline it reports beating by 28%.

**A charitable resolution exists and the paper does not state it**: the fast model may emit all eight poses autoregressively within one forward pass over a single observation, which would make the latency honest and the "high-frequency" framing aspirational — a property of the architecture in deployment, not of anything measured. That reading is consistent with every number in the paper. It is also the reading under which the paper's stated contribution is untested.

**The constructive version of this criticism is cheap to run**: [[concepts/hugsim-benchmark.md]] and Bench2Drive are closed-loop, deliver new observations, and would exercise the mechanism directly. [[sources/latent-wam.md]], [[sources/had.md]] and [[sources/wa-jepa.md]] all report HUGSIM. Drive-HWM reports neither, nor navhard, nor nuScenes.

---

## Results

### Table I — NAVSIM-v1 navtest (PDMS)

† = query-based action expert with multiple trajectory anchors. ‡ = **autoregressive action expert with best-of-$N$ ($N=6$)** — i.e. an oracle selection score, correctly footnoted by the paper.

| Method | Ref | Sensors | NC↑ | DAC↑ | TTC↑ | C.↑ | EP↑ | PDMS↑ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Human | – | – | 100.0 | 100.0 | 100.0 | 99.9 | 87.5 | 94.8 |
| *BEV-based* | | | | | | | | |
| UniAD | CVPR'23 | 6×Cam | 97.8 | 91.9 | 92.9 | 100.0 | 78.8 | 83.4 |
| TransFuser | TPAMI'23 | 3×Cam + L | 97.7 | 92.8 | 92.8 | 100.0 | 79.2 | 84.0 |
| PARA-Drive | CVPR'24 | 6×Cam | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| LAW | ICLR'25 | 1×Cam | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 |
| Hydra-MDP | arXiv'24 | 3×Cam + L | 98.3 | 96.0 | 94.6 | 100.0 | 78.7 | 86.5 |
| [[sources/diffusiondrive.md]] | CVPR'25 | 3×Cam + L | 98.2 | 96.2 | 94.7 | 100.0 | 82.2 | 88.1 |
| WoTE | ICCV'25 | 3×Cam + L | 98.5 | 96.8 | 94.4 | 99.9 | 81.9 | 88.3 |
| *Normal view* | | | | | | | | |
| [[sources/autovla.md]] | NeurIPS'25 | 3×Cam | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 |
| [[sources/recogdrive.md]] | arXiv'25 | 3×Cam | 98.2 | 97.8 | 95.2 | 99.8 | 83.5 | 89.6 |
| [[sources/drivevla-w0.md]] † | arXiv'25 | 1×Cam | 98.7 | **99.1** | 95.3 | 99.3 | 83.3 | 90.2 |
| [[sources/autovla.md]] ‡ *(BoN-6)* | NeurIPS'25 | 3×Cam | 99.1 | 97.1 | 97.1 | 100.0 | 87.6 | 92.1 |
| [[sources/drivevla-w0.md]] ‡ *(BoN-6)* | arXiv'25 | 1×Cam | 99.3 | 97.4 | 97.0 | 99.9 | 88.3 | 93.0 |
| **Drive-HWM** | – | **1×Cam** | **99.6** | 99.0 | **98.5** | **100.0** | **89.0** | **93.8** |

**Baseline-labelling discipline here is good, and better than most tables this wiki has audited.** Both best-of-6 oracle rows are explicitly footnoted as such — exactly the practice missing from [[sources/wcog-vla.md]], which lists AutoVLA at its 92.12 BoN in a single-sample table. Drive-HWM's own row carries no BoN mark, so **93.8 is a single-sample score standing above two oracle-6 scores**, which is a stronger claim than the paper makes for it. See [[concepts/best-of-n.md]].

**The comparison set is conspicuously narrow at the top, though.** Everything in this wiki above 90.2 is absent: [[sources/clear.md]] 93.7, [[sources/da-wam.md]] 93.7, [[sources/drivesuprim.md]] 93.5, [[sources/drive-jepa.md]] 93.3, [[sources/wcog-vla.md]] 92.9, [[sources/hybriddriveVLA.md]] 92.1, [[sources/lwdrive.md]] 92.0, [[sources/wa-jepa.md]] 91.8, [[sources/dynvla.md]] 91.7, [[sources/simwam.md]] 91.5. The table's strongest non-oracle entry is DriveVLA-W0† at 90.2. So "the highest overall PDMS" is true within this table and uncheckable against the frontier from the paper.

**Sub-scores are the strongest part, and they survive both result sets.** **NC 99.6 would be the highest v1 No-at-fault-Collision score in this wiki**, above [[sources/wa-jepa.md]]'s 99.5; even the prose reading (99.5) ties it. TTC 98.5 ties [[sources/wcog-vla.md]] for **second**, behind [[sources/driveva.md]]'s 98.7. DAC 99.0 is second, behind [[sources/drivevla-w0.md]]'s 99.1 — which is the one sub-score the paper concedes, and correctly. EP 89.0 is above the human reference of 87.5 in its own table but below [[sources/drivesuprim.md]]'s 91.3. **A single front camera at 256×144 producing the wiki's best NC is the result worth crediting regardless of which aggregate is right.**

**On the EP claim.** Text: "the gain in EP indicates that such long-horizon reasoning allows the agent to make safer yet less overly conservative progress." EP 89.0 against a human reference of 87.5 is above the recorded human trajectory, which [[sources/pair-drive.md]] has shown is not an oracle ceiling under PDM scoring — so this is possible. It is the same pattern [[sources/lwdrive.md]] shows (EP 90.3 vs. human 87.4), where progress above human comes at EC 73.3. Drive-HWM's v1 comfort is 100.0 and its v2 EC is 87.5, so it does not pay that cost. That combination is unusual and unremarked.

### Table II — NAVSIM-v2 navtest (EPDMS)

| Method | NC↑ | DAC↑ | DDC↑ | TLC↑ | EP↑ | TTC↑ | LK↑ | HC↑ | EC↑ | EPDMS↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ego Status | 93.1 | 77.9 | 92.7 | 99.6 | 86.0 | 91.5 | 89.4 | 98.3 | 85.4 | 64.0 |
| TransFuser | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| HydraMDP++ | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| "DriveSupervisor" *(= [[sources/drivesuprim.md]])* | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| ARTEMIS | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | – | 83.1 |
| [[sources/diffusiondrive.md]] | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | **96.8** | 98.3 | **87.7** | 84.5 |
| [[sources/drivevla-w0.md]] | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 |
| **Drive-HWM** | **98.9** | **99.5** | **99.4** | **99.9** | **88.5** | **98.7** | 96.5 | **98.3** | 87.5 | **86.4** |

**This is the seventh table in the wiki drawing on the shared baseline block, and it carries all six rows digit-identically.** TransFuser (76.7), HydraMDP++ (81.4), DriveSuprim (83.1), ARTEMIS (83.1), DiffusionDrive (84.5) and DriveVLA-W0 (86.1) match [[sources/geoworldad.md]], [[sources/drivefuture.md]], [[sources/lwdrive.md]], [[sources/brainwam.md]] and [[sources/da-wam.md]] across every submetric. Drive-HWM prints them in **one unlabelled column**, so its own 86.4 is filed alongside a block that [[sources/drivefuture.md]] has shown is internally split between pre-fix and corrected conventions. See [[concepts/navsim-benchmark.md]].

**One detail makes the block's propagation mechanism visible.** The paper renames DriveSuprim to **"DriveSupervisor"** while citing DriveSuprim's own arXiv entry (2506.06659) — a row copied without the authors recognizing the method it describes. This wiki has inferred block-copying from digit-identity before; this is the first direct evidence of it.

ARTEMIS's EC is given as "–", so Drive-HWM abstains from the 98.3-vs-89.1 tally and does not change it.

**The v2 claim is much weaker than the v1 claim, and the paper does not say so.** Its strongest v2 baseline is DriveVLA-W0 at 86.1. Absent from the table: [[sources/wam-diff.md]] 89.7, [[sources/lwdrive.md]] 89.6, [[sources/latent-wam.md]] 89.3, [[sources/explorevla.md]] 88.8, [[sources/had.md]] 88.6, [[sources/dreameraD.md]] 87.7, [[sources/drivesuprim.md]] 87.1 — all of which exceed 86.4 under any reading. **93.8 on v1 would lead the wiki; 86.4 on v2 sits roughly fifteenth.** That asymmetry is not addressed.

### The aggregate does not track the submetrics on Drive-HWM's own row {#epdms-residual}

Applying the paper's own stated formula, $\mathrm{EPDMS}=\mathrm{NC}\times\mathrm{DAC}\times\mathrm{DDC}\times\mathrm{TLC}\times(5\mathrm{EP}+5\mathrm{TTC}+2\mathrm{LK}+2\mathrm{HC}+2\mathrm{EC})/16$, to every row of its own Table II:

| Method | Computed | Reported | Residual |
|---|---:|---:|---:|
| Ego Status | 60.00 | 64.0 | −4.00 |
| TransFuser | 77.98 | 76.7 | +1.28 |
| HydraMDP++ | 83.56 | 81.4 | +2.16 |
| DriveSuprim | 85.38 | 83.1 | +2.28 |
| DiffusionDrive | 86.98 | 84.5 | +2.48 |
| DriveVLA-W0 | 84.79 | 86.1 | −1.31 |
| **Drive-HWM** | **91.65** | **86.4** | **+5.25** |

**The closed form is not expected to reproduce exactly** — real EPDMS aggregates per scenario, so a formula applied to averaged submetrics picks up a Jensen gap, which is why every row is off. The signal is in the *structure* of the residuals: peers cluster at +1.3 to +2.5 and Drive-HWM's own row is **more than double the largest peer residual**.

The same anomaly without the formula: Drive-HWM's submetrics **beat DriveVLA-W0's on all nine**, including **EC by 28.6 points** (87.5 vs. 58.9). EC carries weight 2/16 in the weighted average, so that gap alone is worth roughly +3.4 EPDMS before counting DAC, DDC, EP, TTC or LK — yet the reported aggregate moves **+0.3**, from 86.1 to 86.4.

This does not show an error and certainly not inflation — if anything the reported aggregate is *conservative* relative to what its own submetrics imply, and the same check on Table I gives Drive-HWM the **smallest** PDMS residual in that table (−0.33 against a −2 to −4 field), so there is no over-crediting there either. What it shows is that **a reader cannot reconcile Table II's row with Table II's own column**, which is a second reason for `medium` confidence. It is also consistent with the [two-result-set problem](#two-result-sets): submetrics and aggregate may simply come from different runs.

### Table III — Computational efficiency (H200, batch 1)

| Method | $N$ | $T_s$ | $T_f$ | $T_{\mathrm{peak}}$ | $T_{\mathrm{avg}}$ | PDMS↑ |
|---|---:|---:|---:|---:|---:|---:|
| [[sources/drivevla-w0.md]] | – | – | 117.8 | 117.8 | 117.8 | 93.0 |
| Fast Model Only | – | – | 81.6 | 81.6 | 81.6 | 93.0 |
| **Drive-HWM** | 8 | 25.6 | 81.6 | 107.2 | **84.8** | **93.8** |

$T_{\mathrm{avg}}=T_f+T_s/N=81.6+3.2=84.8$ ✓, and 84.8/117.8 = 0.72, so the "≈28% reduction" is arithmetically sound **on its own terms**. Whether those are the right terms is [the open question above](#every-timestep). Note also that the *peak* step is 107.2 ms, so a real-time budget must be sized for $T_{\mathrm{peak}}$, not $T_{\mathrm{avg}}$ — the amortization is a throughput argument, not a latency guarantee. No jitter analysis is given; the paper's own limitations section names asynchronous inference as future work, which is the right fix.

**The slow branch is cheap in the right way.** 25.6 ms for a $K=8$ parallel latent prediction, amortized to 3.2 ms, sits at the efficient extreme of the world-model cost spectrum this wiki tracks: [[sources/adaptive-wam.md]] 170 ms total, [[sources/simwam.md]] 518 ms, [[sources/foresight.md]] 900 ms with 870 ms of world model. **Predicting flow latents in parallel is roughly 270× cheaper than running a video generator to a finished future**, and if the +0.3-to-+0.8 PDMS gain is real it is the best cost-per-point datapoint in the test-time-imagination comparison.

---

## Ablations

### Table IV — Hierarchy and prediction horizon

| Configuration | Slow | Fast | $K$ | NC↑ | DAC↑ | PDMS↑ |
|---|:-:|:-:|---:|---:|---:|---:|
| Fast only | – | ✓ | – | 99.3 | 97.4 | 93.0 |
| Slow only | ✓ | – | 8 | 98.2 | 97.1 | 90.2 |
| Drive-HWM | ✓ | ✓ | 4 | 99.0 | 98.0 | 93.0 |
| **Drive-HWM** | ✓ | ✓ | **8** | **99.6** | **99.0** | **93.8** |
| Drive-HWM | ✓ | ✓ | 12 | 99.2 | 98.4 | 93.2 |

**The horizon sweep is the clean result here.** 93.0 → 93.8 → 93.2 across $K=4,8,12$ is non-monotonic with an interior optimum, and the paper's explanation is the right one: "A short horizon provides insufficient future context, whereas an excessively long horizon introduces greater prediction uncertainty." A 0.8-point spread over a 3× horizon range is also a useful bound on how much this axis is worth. Note that $K=8$ coincides exactly with NAVSIM's scoring horizon, so the sweep may be measuring horizon-matching rather than an intrinsic property of driving dynamics — a confound the paper does not consider and that a second benchmark would settle.

**The `Slow only` row should not exist.** §III-B states plainly: *"The slow branch therefore does not output actions or an explicit trajectory."* Table IV nonetheless reports it at 90.2 PDMS. Some action head must have been attached; the paper never says what, and the row is uninterpretable as printed.

### Ablation baseline provenance {#ablation-provenance}

**Two of Table IV's four ablation PDMS values equal DriveVLA-W0's two published Table I rows**, and one matches on every reported submetric:

| Table IV row | NC | DAC | PDMS | Table I row | NC | DAC | PDMS |
|---|---:|---:|---:|---|---:|---:|---:|
| Fast only | 99.3 | 97.4 | 93.0 | DriveVLA-W0 ‡ *(BoN-6)* | 99.3 | 97.4 | 93.0 |
| Slow only | 98.2 | 97.1 | 90.2 | DriveVLA-W0 † *(anchors)* | 98.7 | 99.1 | 90.2 |

Table III presents `Fast Model Only` as a *different model* from DriveVLA-W0 — same 93.0 PDMS, but 81.6 ms against 117.8 ms — so the paper's position is that this is a retrained variant that happens to land on three identical digits. That is possible. If it is instead DriveVLA-W0's published row, then **the ablation's no-slow-model baseline is an oracle best-of-6 score**, and the +0.8 (or +0.3) attributed to the slow world model is measured against a *stronger* baseline than a like-for-like single-sample ablation would give — which would make the paper understate its own mechanism rather than overstate it.

Either way, **no ablation row in this paper is stated to be a retrained model**, and the `Slow only` row's PDMS-only match to an external row with different submetrics is the harder one to explain as coincidence. This is a new variant for [[concepts/navsim-benchmark.md]]'s provenance catalogue: not a mislabelled *baseline* row, but an **ablation row indistinguishable from an external published row**.

### Table V — Backbone swaps (two controlled sweeps)

| Backbone | NC↑ | DAC↑ | PDMS↑ |
|---|---:|---:|---:|
| **Slow world model** *(fast fixed = Emu3)* | | | |
| CogVideo | 98.8 | 98.1 | 93.0 |
| WAN | 99.0 | 98.4 | 93.2 |
| **V-JEPA (Ours)** | **99.6** | **99.0** | **93.8** |
| **Fast action model** *(slow fixed = V-JEPA)* | | | |
| LLaVA-OneVision | 99.0 | 98.4 | 93.5 |
| "Qwen2.5-VL" *(ref. is the Qwen3-VL report)* | 99.1 | 98.6 | 93.3 |
| **Emu3 (Ours)** | **99.6** | **99.0** | **93.8** |

**This is the wiki's first controlled comparison of a JEPA predictor against generative video models in the same slow slot**, and it is the more valuable of the two sweeps. [[sources/simwam.md]] swapped Wan 1.3B for Wan 5B (scale barely mattered); this swaps *paradigms* with everything downstream fixed and finds **+0.6 for latent-space prediction over WAN and +0.8 over CogVideo**. The paper's reading is the one this wiki has been converging on from other directions:

> "its latent-space predictive objective focuses more directly on temporally meaningful scene dynamics while avoiding the unnecessary complexity of reconstructing low-level visual details."

Set against [[sources/foresight.md]] (a frozen 2.5B Epona as the primary encoder, 870 ms, +0.3 under vanilla attention) and [[sources/adaptive-wam.md]] (exit the video DiT early; readout depth worth 4.80, noise index ≤0.15), the picture is consistent: **the pixel-generation capability of a video world model is mostly not what the planner is buying.** Drive-HWM adds the cheapest version of that claim — skip the generator entirely, predict a motion latent, and beat both video priors at 25.6 ms.

**Caveat: the comparison is not compute- or capacity-matched.** No parameter counts are given for any of the three, and CogVideo/WAN are large generative models being asked to produce a conditioning latent. The supportable statement is that V-JEPA is the better *choice at this budget*, not that latent prediction dominates generation.

**Two citation errors sit in this table.** The fast-model row is labelled **Qwen2.5-VL** but cites reference [57], the **Qwen3-VL technical report** (2511.21631) — so the reader cannot tell which model was run. And the slow-model row is labelled **V-JEPA** citing reference [55], which is **VL-JEPA** (2512.10942, a vision-*language* JEPA), while Fig. 2's caption says "employs VL-JEPA" and §IV-D's prose says "we compare V-JEPA with the generative video models." See [below](#vjepa-naming).

### Fig. 4 / Table VI — The slow model's prediction target {#slow-target}

**Fig. 4's image is missing from the clipping**, so the sweep's numbers are unavailable. The prose records the ordering and the reasoning:

> "RGB prediction yields the lowest performance because reconstructing appearance details may distract the model from learning planning-relevant motion patterns. BEV and depth supervision provide stronger geometric and spatial cues, but only implicitly characterize temporal changes and object movements. In contrast, optical flow achieves the best results."

So for the **slow** branch: **flow > depth ≈ BEV > RGB**, magnitudes unknown.

![[vis1_new.png|Predicted future representations compared: BEV, depth, and optical flow, with flow giving denser and more motion-discriminative cues]]

**Fig. 5**: Compared with BEV and depth representations, optical flow provides denser and more motion-discriminative cues, explicitly capturing scene dynamics and object displacements.

**Table VI — linear probing of the frozen slow-model latents:**

| Probed Latent | FEM↑ *(future ego motion)* | MC↑ *(motion consistency)* |
|---|---:|---:|
| BEV | 71.2 | 63.8 |
| Depth | 74.5 | 66.9 |
| RGB | 76.8 | 69.4 |
| **Optical Flow (Ours)** | **83.7** | **76.1** |

**The probe is the right instrument and the wrong task.** Freezing the slow model and training a linear head is exactly how to ask what a representation encodes, and this wiki has almost no such measurements — [[sources/hybriddriveVLA.md]]'s CKA/CCA/SAE analysis and [[sources/unified-driving-tokens.md]]'s reconstruction diagnostics are the closest. But both probe tasks are **motion** tasks, and the flow latent was trained on **motion**. A representation trained to reconstruct displacement fields outperforming ones trained on appearance and geometry at predicting displacement is close to definitional. The informative probe is an *off-objective* one — semantics, occupancy, traffic-light state, agent identity — where the appearance-trained latents should win and the question becomes how much the flow latent gives up. The paper's own framing invites it: §III-B argues the DAL keeps "the contextual information required to explain that displacement," and Table VI cannot test that claim.

**Units are never stated.** FEM and MC are bare numbers with no metric definition, no probe architecture, and no train/test split.

**One unremarked tension with the geometry papers.** Optical flow is 2D image-space displacement — it does *not* share a coordinate frame with the output trajectory, the exact property [[sources/geowam.md]] and [[sources/geoworldad.md]] argue is decisive ("coordinate frame alone is worth +2.5"). Drive-HWM measures its image-space flow latent as the *best* encoder of future ego motion, beating a BEV latent that does share the ego frame. Both cannot be the general rule, and no paper compares them. See [[concepts/world-model-for-ad.md]].

### Table VII — Slow-to-fast conditioning

| Conditioning | NC↑ | DAC↑ | PDMS↑ |
|---|---:|---:|---:|
| Concatenation | 98.2 | 97.7 | 92.5 |
| Cross-Attention | 98.7 | 98.0 | 93.0 |
| Gated Cross-Attention | 98.9 | 98.5 | 93.1 |
| AdaLN | 99.2 | 98.6 | 93.3 |
| **FiLM (Ours)** | **99.6** | **99.0** | **93.8** |

**A clean, monotone, five-way sweep with everything else fixed, and the most directly reusable table in the paper.** The 1.3-point span from concatenation to FiLM is larger than the paper's own slow-vs-fast hierarchy effect under either result set — so **how the future latent is injected matters more here than whether the hierarchy exists.** The ordering (affine modulation > attention > concatenation) is worth recording against [[sources/brainwam.md]], which reached a compatible conclusion by a different route: gated cross-attention through a narrow 8-token bottleneck, chosen because raw-token concatenation in a shared attention pool *actively hurt* (87.8 against its own WAM-only 88.1). Both papers find concatenation worst; both preserve the backbone's token layout. FiLM is the cheaper mechanism and here it wins outright.

The caveat is that this sweep, like every other in the paper, is a single run with no seed variance, and AdaLN at 93.3 is only 0.5 behind.

### Table VIII — The fast model's next-frame target

| Auxiliary Target | NC↑ | DAC↑ | PDMS↑ |
|---|---:|---:|---:|
| None (action only) | 99.2 | 98.5 | 93.1 |
| Next optical flow | 99.5 | 98.7 | 93.5 |
| Next-frame depth | 99.4 | 98.8 | 93.6 |
| **Next-frame RGB (Ours)** | **99.6** | **99.0** | **93.8** |

**+0.7 for any dense auxiliary target over action-only supervision**, with RGB ahead of the alternatives by 0.2–0.3. The reasoning is the inverse of the slow-model argument, and stated explicitly:

> "Unlike the slow world model, whose primary role is to anticipate long-horizon scene dynamics, the fast model focuses on making observation-grounded decisions at the current moment. Its representation must therefore retain comprehensive local evidence, including object semantics, lane markings, traffic signals, spatial layouts, and the states of nearby agents. RGB prediction provides richer and more complete scene information than depth or optical flow."

Note this also gives Drive-HWM a second, independent dense-supervision measurement — and **+0.7 for next-frame RGB is larger than the +0.3–0.8 for the entire slow hierarchy**. The auxiliary objective the paper treats as a supporting detail outperforms the contribution in its title.

### Target by Role: the paper's most transferable result {#target-by-role}

Putting [Fig. 4](#slow-target) and Table VIII side by side gives the finding, which the paper states in one sentence without developing:

| Branch | Horizon | Best target | Worst target |
|---|---|---|---|
| **Slow** | 8 steps ahead | **optical flow** | **RGB** |
| **Fast** | 1 step ahead | **RGB** (93.8) | **optical flow** (93.5) |

> "These results suggest that optical flow is more suitable for learning dynamics-oriented representations in the slow world model, whereas next-frame RGB supervision better supports the fast model in generating accurate and responsive driving actions."

**The ordering inverts completely between the two branches**, and both sweeps run with everything else fixed in one codebase — which is what makes this worth extracting. Every world-model planner this wiki has ingested picks one target and defends it: pixels ([[sources/simwam.md]], [[sources/driveva.md]]), video latents ([[sources/drivelaw.md]]), semantic features ([[sources/wa-jepa.md]], [[sources/deepsight.md]]), metric geometry ([[sources/geowam.md]], [[sources/geoworldad.md]]), symbolic state ([[sources/sgdrive.md]]), ego-trajectory latents ([[sources/auto-jepa.md]]), multi-agent trajectories ([[sources/wcog-vla.md]]). [[sources/coworld-vla.md]] ran four in parallel — but all at one horizon, and its ablation never separates them by role.

**The generalization Drive-HWM supports is that the target should match the horizon**: far futures are dominated by motion and appearance is unpredictable, so predict motion; near futures are dominated by what is currently visible, so predict appearance. That is a scheduling principle for world-model supervision, and it is orthogonal to the objective-form principle this wiki assembled from WA-JEPA and DriveFuture (match the *loss* to the target's entropy). **Target content by horizon; objective form by entropy.** The two compose, and nobody has run them together.

**The evidence is thinner than the claim.** The slow-side sweep's numbers are missing from the clipping; the fast-side spread is 0.3 across the top three; both are single runs; and the two branches also differ in backbone, horizon, loss form, and dataset stage, so "role" is confounded with several other things. Treat it as the best-supported hypothesis on this axis rather than a result.

---

## Naming: V-JEPA or VL-JEPA? {#vjepa-naming}

The slow backbone's identity is ambiguous across four places:

| Location | Says |
|---|---|
| Fig. 2 caption | "employs **VL-JEPA** to predict multi-horizon optical flow" |
| Table V | "**V-JEPA** [55] (Ours)" |
| §IV-D prose | "we compare **V-JEPA** with the generative video models CogVideo and WAN" |
| Reference [55] | Chen et al., "**VL-JEPA**: joint embedding predictive architecture for vision-language", arXiv 2512.10942 |

These are different models. V-JEPA / V-JEPA 2 is Meta's video JEPA and is what [[sources/drive-jepa.md]], [[sources/wa-jepa.md]], [[sources/auto-jepa.md]], [[sources/da-wam.md]] and [[sources/coworld-vla.md]] all use; VL-JEPA is a vision-language variant from a different group. Two of three mentions say V-JEPA and the citation says VL-JEPA. **This matters for [[concepts/foundation-backbones-for-ad.md]]**, where V-JEPA-family encoders now appear in six ingested papers and cross-paper comparison assumes a common backbone. This page records the slow backbone as **V-JEPA-family, exact variant undetermined**.

---

## Qualitative Results

![[vis_new.png|Front-camera and bird's-eye-view trajectory comparison across four scenarios: human, Drive-HWM, DriveVLA-W0, and TransFuser]]

**Fig. 3**: Trajectories from Human, Drive-HWM, [[sources/drivevla-w0.md]], and TransFuser in front-camera and BEV views across four scenarios — pedestrian avoidance, intersection turning, stop-line approach, and a traffic-light-controlled intersection.

The four cases exercise route-level intent rather than local control, and the paper is explicit about why the last is the interesting one: "the current observation alone provides limited evidence about the complete maneuver, requiring the planner to reason jointly about traffic signals, road topology, and future route evolution." That is the correct scenario class for the thesis. All comparisons are against two baselines, both among the weaker entries in the paper's own Table I, and no quantitative per-scenario breakdown accompanies them.

![[failed.png|Failure case at an unsignalized Y-intersection: under the predicted trajectory an approaching vehicle progressively enters the ego path, ending in a potential collision]]

**Fig. 6**: Failure at an unsignalized Y-intersection. Rows are observed inputs, ground-truth future observations, futures conditioned on the ground-truth trajectory, and futures conditioned on the predicted trajectory. In the failed rollout the approaching vehicle enters the ego path from $V_{t+1}$ to $V_{t+3}$ while the ego continues turning without yielding, ending in a potential collision at $V_{t+4}$.

**The failure analysis is the best-argued page in the paper, and it quietly contradicts §III-C.** Producing this figure requires rolling the next-frame predictor forward four steps under two different action conditionings — a real action-conditioned world-model rollout, the capability §III-C says can be omitted at inference. It is also the only place the fast model's generative head is used for anything, and the paper reports no FID, FVD, or any other generation metric anywhere. The diagnosis — "Drive-HWM may still struggle in highly interactive scenarios where accurate anticipation of other agents' future motion is critical" — is exactly the gap [[sources/wcog-vla.md]] addresses by making other agents' trajectories the prediction target, and is the failure mode a non-reactive benchmark is least able to surface.

![[user.png|User study: Drive-HWM preferred in 68% of 50 paired scenarios against DriveVLA-W0, with higher mean ratings on safety, comfort, human-likeness, and route correctness]]

**Fig. 7**: 20 raters, 50 paired scenarios, 1–5 Likert. Drive-HWM preferred in 68% of comparisons vs. 21% for [[sources/drivevla-w0.md]] (11% no preference); means 4.58/4.21 safety, 4.29/4.06 comfort, 4.44/4.05 human-likeness, 4.51/4.17 route correctness.

**The first user study in this wiki**, which is worth noting on its own. The caveats are the standard ones and none is addressed: authors' own study, no rater recruitment or expertise details, no inter-rater agreement, no statistical test on either the 68/21/11 split or the mean differences, no blinding protocol stated, and DriveVLA-W0 is presumably a local reimplementation. The four rated axes duplicate PDMS submetrics (safety ≈ NC/TTC, comfort ≈ C/EC, route correctness ≈ DAC/DDC), so the study is closest to an independent confirmation that the metric gaps are *perceptible* — a modest but real claim, and more than most papers attempt.

---

## Limitations

1. **Two complete result sets, tables against prose, across both benchmarks** — 93.8/86.4 versus 93.3/86.2, changing the paper's headline contribution from +0.8 to +0.3 and its NAVSIM-v1 rank in this wiki from first to fourth. See [above](#two-result-sets). This is the dominant limitation and the reason for `medium` confidence.
2. **The central claim — high-frequency, observation-grounded re-decision — is never exercised.** NAVSIM is non-reactive and single-shot; with $N=K=8$ at NAVSIM's 8-pose convention the slow model runs once per scenario and no new observation ever arrives. No closed-loop benchmark is reported.
3. **What one action is is never defined.** $L_a$ is unspecified, and the two readings differ by 8× in per-trajectory latency while each contradicts a different claim in the paper. See [above](#every-timestep).
4. **Figure 4 is missing from the clipping**, so the slow-model prediction-target sweep — the evidence for the paper's most transferable claim — has no recoverable numbers.
5. **§III-D contradicts §III-C on the fast loss**, defining $\mathcal{L}_{\mathrm{fast}}$ once as a token-level NLL and once as a latent regression, and describing the whole objective without mentioning optical flow. See [above](#loss-contradiction).
6. **§III-B says the slow branch outputs no actions; Table IV reports a `Slow only` row at 90.2 PDMS.** Whatever action head was attached is undescribed.
7. **Two of four Table IV ablation values equal DriveVLA-W0's two published rows**, one on all three reported metrics. No ablation is stated to be a retrained model. See [above](#ablation-provenance).
8. **Drive-HWM's own Table II row cannot be reconciled with its own submetrics** — a +5.25 residual under the paper's stated EPDMS formula against a +1.3-to-+2.5 peer field, and a +0.3 aggregate gain over DriveVLA-W0 despite dominating it on all nine submetrics including EC by 28.6. See [above](#epdms-residual).
9. **Almost nothing is specified.** No parameter counts for any component; no value for $\lambda_{\mathrm{img}}$, $\lambda_s$, $\lambda_f$, $w_k$, $\rho$, $L_a$, $L_v$, or $d$; no identification of the optical-flow estimator that produces every slow-model target; no list of which layers FiLM modulates; no DAL token count; no GPU-hours; no code.
10. **Citation errors in Table V**: the row labelled "Qwen2.5-VL" cites the Qwen3-VL report, and the row labelled "V-JEPA" cites VL-JEPA while the framework figure says VL-JEPA and the prose says V-JEPA. See [above](#vjepa-naming).
11. **Seventh user of the shared NAVSIM-v2 baseline block, in one unlabelled column**, with DriveSuprim renamed "DriveSupervisor" — direct evidence that the block is copied rather than recomputed.
12. **Both comparison sets stop well below the frontier.** Table I's strongest non-oracle row is DriveVLA-W0† 90.2; Table II's is DriveVLA-W0 86.1. Roughly ten wiki methods exceed 90.2 on v1 and seven exceed 86.4 on v2.
13. **Single runs throughout, no seed variance**, in a paper whose headline ablation effect is 0.3–0.8 and whose largest sweep spread is 1.3.
14. **No generation metrics of any kind** (no FID, FVD, flow EPE, reconstruction error) despite two generative objectives and a failure analysis built on multi-step rollouts.
15. **Calling Emu3-8B a "lightweight multimodal backbone"** (abstract, §I) is not a defensible description; the fast model is the expensive component, 81.6 ms of the 84.8 ms budget.
16. **256 × 144 input, single front camera**, neither flagged as a limitation. That is the lowest resolution recorded in this wiki and a real constraint on traffic-light and distant-agent perception — although v2 TLC of 99.9 is the table's best, which sits oddly with it.
17. **No navhard, no HUGSIM, no Bench2Drive, no nuScenes, no reactive evaluation.** The paper's own stated limitations are only computational overhead and the absence of multimodal-future modelling.
18. **The parallel multi-offset flow formulation cannot guarantee a coherent future**, which is the property §I claims for it; nothing measures cross-offset consistency.

---

## Key Cross-References

- [[concepts/world-model-for-ad.md]] — Pattern 35 (two world models at two rates with two targets); dense optical flow as a new annotation-free target family; [Target by Role](#target-by-role) as a horizon-based complement to the entropy-based objective-form rule; a third position in the training-time/inference-time taxonomy (predictor survives, decoders discarded); the image-space-flow versus ego-frame-geometry tension.
- [[concepts/navsim-benchmark.md]] — the two-result-set problem and its effect on the v1 ladder; the seventh user of the shared baseline block and the "DriveSupervisor" rename as direct evidence of copying; a new provenance failure mode (ablation rows matching external published rows); the [EPDMS residual audit](#epdms-residual).
- [[concepts/dual-system-vla.md]] — fast–slow by **temporal role** rather than scene difficulty or module cost; the first entry whose slow branch emits no decision; FiLM against [[sources/brainwam.md]]'s gated cross-attention as two answers to one interface question.
- [[concepts/foundation-backbones-for-ad.md]] — the second controlled video-prior swap and the first JEPA-versus-generative-video comparison in a fixed slow slot (V-JEPA 93.8 > WAN 93.2 > CogVideo 93.0 at 25.6 ms); Emu3 vs. LLaVA-OneVision vs. Qwen (93.8 / 93.5 / 93.3); the V-JEPA/VL-JEPA ambiguity.
- [[sources/drivevla-w0.md]] — the same Emu3-8B backbone and the paper's primary comparison target; supplies both oracle BoN-6 rows in Table I and, apparently, two of four ablation values in Table IV.
- [[sources/coworld-vla.md]] — the closest relative: four future targets in one latent space at one horizon, against Drive-HWM's two targets at two horizons. Both use a frozen foundation predictor's latent as the planner's conditioning interface and discard the generator.
- [[sources/adaptive-wam.md]], [[sources/foresight.md]], [[sources/simwam.md]] — the cost-per-point comparison for test-time future prediction; 25.6 ms amortized to 3.2 ms is the cheapest slow branch recorded here.
- [[sources/wcog-vla.md]] — the failure mode Drive-HWM's Fig. 6 diagnoses (multi-agent interaction anticipation) is exactly what WCog-VLA makes its prediction target.
