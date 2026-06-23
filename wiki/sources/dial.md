---
title: "Driving Intents Amplify Planning-Oriented Reinforcement Learning"
type: source-summary
sources: [raw/papers/Driving Intents Amplify Planning-Oriented Reinforcement Learning.md]
related: [concepts/intent-conditioned-planning.md, concepts/rl-for-ad.md, concepts/gspo-vs-grpo.md, concepts/best-of-n.md, concepts/diffusion-planner.md, concepts/nuscenes-waymo-evals.md]
created: 2026-06-23
updated: 2026-06-23
confidence: high
---

# DIAL

DIAL (Driving-Intent-Amplified reinforcement Learning) is a two-stage method for preference-aligning a continuous flow-matching driving policy. It argues that ordinary single-demonstration SFT collapses the policy's samples into one maneuver basin. If a GRPO group contains only near-duplicate trajectories, their preference scores are too similar to provide useful relative advantages.

Stage 1 expands proposal support by conditioning the action head on eight discrete driving intents with classifier-free guidance (CFG). Stage 2 preserves that support by constructing every GRPO group across all eight intents rather than sampling many noise seeds under one intent. At deployment, a learned intent classifier selects one intent and the same conditioned generator produces a single trajectory.

## Key Takeaways

- **Preference RL is support-limited.** An optimizer cannot reward a maneuver the policy never samples.
- **Single-demonstration SFT hides mode collapse.** Distance-to-log metrics reward concentration near the recorded path; multi-rater RFS reveals that other trajectories may be preferred.
- **Semantic conditioning beats extra random samples.** Eight intent modes expose maneuver-level alternatives that ordinary stochastic flow sampling misses.
- **Best-of-N measures the support ceiling.** Intent-CFG reaches RFS 9.14 at Best-of-128, above the logged trajectory's 8.13 and the cited RAP Best-of-64 result of 8.5.
- **Multi-intent groups preserve preference contrast.** DIAL reaches held-out RFS 8.211; all four single-intent groups remain below 8.0 and decline more sharply after their peak.
- **More samples per intent are not monotonically better.** Two samples × eight intents gives the highest held-out peak; larger groups peak earlier and lower.
- **Reward design matters.** Dense temporal anchors and label-weighted rater aggregation reduce two evaluator-specific reward-hacking paths.
- **RL improves deployment score while slightly reducing oracle diversity.** DIAL's Best-of-16 diversity ceiling is 6.540 versus 6.617 at SFT initialization, but its deployable held-out RFS rises from 7.696 to 8.211.

![Figure 1: Ordinary samples occupy one maneuver basin and have little RFS contrast; intent-conditioned samples span distinct basins and create useful group-relative advantages.](<../../raw/assets/x1 36.png>)

## Method

### Stage 1: Intent-Conditioned CFG

The action head is conditioned on one of eight rule-derived intents:

1. cruise;
2. lane change left;
3. lane change right;
4. turn left;
5. turn right;
6. U-turn;
7. accelerate;
8. decelerate.

Labels are inferred from demonstrated trajectory geometry using displacement, heading change, lateral shift, and speed change. Classifier-free dropout teaches conditional and unconditional flow fields. CFG combines them at generation time so intent changes steer the same scene toward distinct semantic modes rather than merely perturbing coordinates.

### Stage 2: Multi-Intent GRPO

Each scene contributes $C=8$ intents and $S=2$ stochastic samples per intent, so $K=16$. RFS scores all candidates, and advantages are normalized across the complete per-scene group:

$$
A_i=\frac{R_i-\frac{1}{K}\sum_{j=1}^{K}R_j}{\operatorname{std}_{j=1}^{K}(R_j)+\epsilon}.
$$

The clipped policy update replays each candidate's cumulative SDE path log-probability under the intent that generated it. A $k_3$ reference-path penalty anchors the policy to the starting SFT checkpoint with coefficient $\beta=0.002$.

![Figure 2: DIAL pipeline: intent-CFG imitation training followed by 16-rollout multi-intent GRPO groups scored with RFS.](<../../raw/assets/x2 35.png>)

### Reward-Hacking-Aware RFS

Canonical RFS uses only 3 s and 5 s anchors and takes a hard maximum over up to three rater trajectories. DIAL identifies two exploit paths:

- intermediate waypoints can drift while the two scored anchors remain acceptable;
- the policy can overfit to the easiest high-label rater geometry.

Training therefore scores 1–5 s anchors and replaces the hard maximum with label-softmax aggregation. The weights depend only on human labels, not model-controlled geometry. Evaluation still uses standard WOD-E2E RFS.

## Experimental Protocol

- Backbone: MindVLA-U1 with a continuous flow action head.
- SFT data: Waymo training split.
- Preference pool: 438 RFS-labeled validation sequences.
- RL split: deterministic hash, 338 training / 100 held-out, `split_seed=43`.
- Main group: eight intents × two samples = 16 candidates per scene.
- Batch size: 4.
- Learning rate: constant $5\times10^{-7}$.
- SDE sampler noise: 0.5.
- GRPO clipping: 0.2.
- Reference-path coefficient: 0.002.
- Primary reported model selection: highest RFS on the 100-sequence held-out split.

## Results

### Table 1: Controlled Waymo-Only Preference RL

`TR` is reported by the paper as a percentage but is not defined in the extracted text. Peak rows are selected by the same 100-sequence held-out RFS column.

| Model | Action representation | Stage | Held RFS | RL gain | TR | Full RFS |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| WAM-Flow | Discrete flow | SFT init | 5.547 | - | 24.0% | 5.757 |
| WAM-Flow | Discrete flow | RL peak | 5.634 | +0.087 | 19.0% | 6.111 |
| Curious-VLA | Action token | SFT init | 5.808 | - | 30.0% | 5.827 |
| Curious-VLA | Action token | RL peak | 5.954 | +0.146 | 31.0% | 7.157 |
| AutoVLA | Action token | SFT init | 6.744 | - | 46.0% | 6.809 |
| AutoVLA | Action token | RL peak | 6.787 | +0.043 | 47.0% | 6.780 |
| ReCogDrive | DiT diffusion | SFT init | 7.399 | - | 58.0% | 7.244 |
| ReCogDrive | DiT diffusion | RL peak | 7.714 | +0.315 | 65.4% | 7.543 |
| DIAL | Continuous flow | SFT init | 7.696 | - | 54.7% | 7.369 |
| DIAL | Continuous flow | RL peak | **8.211** | **+0.515** | **68.0%** | **8.631** |

The baselines are retrained under the authors' common Waymo-only SFT and RFS-RL protocol. These are controlled reimplementations, not necessarily their published headline configurations.

### Pre-RL Proposal Ceiling

Ordinary SFT samples from WAM-Flow, Curious-VLA, AutoVLA, and ReCogDrive saturate below the logged-trajectory RFS of 8.13 even at $K=128$. Intent-conditioned strategies cross 8.13 at roughly $K=8$. Equal-budget pooling across all eight intents reaches 9.14 at $K=128$.

This is an oracle support test, not a deployable result: it assumes RFS can select the best candidate after generation.

![Figure 3: Best-of-K support ceiling. Ordinary SFT baselines remain below logged GT; intent-conditioned proposals cross it and eight-intent pooling reaches RFS 9.14 at K=128.](<../../raw/assets/x3 32.png>)

### Table 2: Multi-Intent vs. Single-Intent GRPO

Every row uses $K=16$ candidates per scene.

| Group composition | Intents $C$ | Samples/intent $S$ | Group $K$ | Held peak | TR | Full RFS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GT geometric intent | 1 | 16 | 16 | 7.783 | 65.0% | 7.733 |
| Predicted intent | 1 | 16 | 16 | 7.864 | 57.0% | 8.331 |
| Top-rater intent (label leakage) | 1 | 16 | 16 | 7.728 | 61.0% | 7.545 |
| Random single intent | 1 | 16 | 16 | 7.992 | 64.0% | 8.035 |
| DIAL multi-intent | 8 | 2 | 16 | **8.211** | **68.0%** | **8.631** |

The predicted-intent baseline exhibits the clearest overfitting signature: training RFS continues rising while held-out RFS declines after its peak.

### Table 3: Diversity Preservation at Iteration 4800

`D1` is mean pairwise ADE across eight intent-conditioned trajectories; `D2` is their per-scene RFS standard deviation. `Gap` is Best-of-16 minus Best-of-1 RFS.

| Sampling | D1 | D2 | D3@1 | D3@16 | Gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT init | 6.43 m | 0.52 | 4.387 | **6.617** | **+2.23** |
| DIAL multi-intent | 4.17 m | **0.75** | **4.500** | 6.540 | +2.04 |
| Single random | 2.40 m | 0.43 | 4.381 | 6.231 | +1.85 |
| Single predicted | 4.82 m | 0.64 | 4.372 | 6.172 | +1.80 |
| Single top-rater | **7.08 m** | 0.65 | 4.240 | 6.020 | +1.78 |
| Single GT intent | 6.02 m | 0.59 | 4.229 | 5.889 | +1.66 |

Spatial spread alone is not enough: top-rater has the largest trajectory separation but worse preference differentiation and deployment quality than DIAL.

### Samples per Intent (Referenced Figure 4)

The raw markdown references a chart that is not embedded. Its numeric results are preserved here from the surrounding text:

| Intents $C$ | Samples/intent $S$ | Group $K$ | Held peak | Reported behavior |
| ---: | ---: | ---: | ---: | --- |
| 8 | 1 | 8 | 7.962 | Cheapest and stable throughout training |
| 8 | 2 | 16 | **8.211** | Highest peak; main setting |
| 8 | 3 | 24 | 8.094 | Earlier, slightly lower peak |
| 8 | 4 | 32 | 8.033 | Earlier, lower peak |

### Table 4: Training Reward Ablation

All variants use $C=8$, $S=2$, and $K=16$.

| Reward | Aggregation | Anchors | Temperature | Held peak | TR | Full peak |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Vanilla | Max | Sparse | - | 7.990 | 68.0% | 8.328 |
| Dense anchors only | Max | Dense | - | 7.965 | 67.0% | 8.430 |
| Softmax only | Softmax | Sparse | 1.0 | 8.147 | **72.0%** | 8.485 |
| Softmax + dense, deployment-friendly | Softmax | Dense | 1.0 | 8.130 | 62.0% | **8.728** |
| Softmax + dense | Softmax | Dense | 0.5 | 8.097 | 62.0% | 8.634 |
| DIAL main | Softmax | Dense | 0.3 | **8.211** | 68.0% | 8.631 |
| Mean + dense | Mean | Dense | $\to0$ | 7.834 | 60.0% | 8.098 |

The main configuration optimizes held-out peak height, while temperature 1.0 gives the best full-pool score. The difference reinforces that checkpoint/metric selection materially affects the headline.

## Training Dynamics

Multi-intent DIAL peaks highest and declines least. All single-intent runs peak lower and then deteriorate more sharply; predicted intent continues improving on the RL-training subset after held-out performance falls.

![Figure 5: Held-out RFS dynamics. DIAL peaks at 8.211 and declines less; single-intent policies peak lower and predicted intent shows train/held-out divergence.](<../../raw/assets/x5 27.png>)

## Limitations

- **Hand-coded ontology.** Eight geometry-derived intents omit long-tail and composite behaviors; ambiguous demonstrations can be mislabeled.
- **Small preference dataset.** RL uses 338 sequences, with only 100 sequences in the reported held-out partition.
- **Held-out selection leakage.** The 100-sequence “held-out” RFS is explicitly used for checkpoint and reward-hyperparameter selection, so it is a validation set rather than an untouched test set.
- **Full-pool contamination.** Full RFS includes the 338 sequences used for RL and should not be interpreted as independent generalization.
- **Open-loop preference metric only.** No NAVSIM, nuPlan, Bench2Drive, collision-rate, closed-loop interaction, or real-vehicle validation is provided.
- **Preference is not safety.** RFS encodes human trajectory ratings at selected anchors; surpassing the logged RFS does not prove safer closed-loop behavior.
- **Oracle ceiling is non-deployable.** Best-of-128 assumes oracle RFS selection and is evidence about proposal support, not inference performance.
- **Intent classifier is under-evaluated.** Accuracy, confusion by maneuver, robustness, and its contribution to final RFS are not reported.
- **Inference cost is missing.** The paper does not report latency for conditional/unconditional CFG passes or the MindVLA-U1 deployment stack.
- **Peak reporting.** All methods are compared at their best held-out checkpoint even though several later collapse; final-checkpoint and multi-seed uncertainty are absent.
- **Initialization-number inconsistency.** The abstract reports held-out RFS improving from 7.681, while Table 1 and the conclusion use 7.696.
- **Training/evaluation reward mismatch.** Dense label-softmax RFS is optimized, but canonical sparse max-RFS is reported; this is reasonable shaping but adds sensitivity to aggregation and temperature.
- **Undefined metric in extraction.** `TR` is tabulated but not defined in the available raw text.
- **Missing figure asset.** The source references a samples-per-intent Figure 4 but the raw markdown contains no corresponding image link; its reported values are reconstructed above.

## Wiki Relevance

- [[concepts/intent-conditioned-planning.md]] — intent as a control variable for proposal support rather than only a classification target.
- [[concepts/rl-for-ad.md]] — preference RL succeeds when per-scene groups contain maneuver-level contrast.
- [[concepts/gspo-vs-grpo.md]] — group composition can matter more than changing GRPO normalization.
- [[concepts/best-of-n.md]] — distinguishes support ceilings from deployable intent-classified performance.
- [[concepts/diffusion-planner.md]] — intent CFG structures a continuous flow action distribution before RL.
- [[concepts/nuscenes-waymo-evals.md]] — adds WOD-E2E RFS protocol, data-split, and evaluation-leakage caveats.
