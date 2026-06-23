---
title: "DisCO: Reinforcing Large Reasoning Models with Discriminative Constrained Optimization"
type: source-summary
sources: [raw/papers/DisCO_ Reinforcing Large Reasoning Models with Discriminative Constrained Optimization.md]
related: [concepts/discriminative-policy-optimization.md, concepts/gspo-vs-grpo.md, concepts/r1-zero-like-training.md, concepts/rl-for-ad.md]
created: 2026-06-23
updated: 2026-06-23
confidence: medium
---

# DisCO

DisCO redesigns binary-reward reasoning RL as discriminative learning rather than modifying GRPO's clipping or normalization heuristically. For each question, it separates generated responses into positive and negative sets, increases scores assigned to positive responses, decreases scores assigned to negatives, emphasizes the highest-scoring hard negatives, and constrains the policy update through a KL trust region.

The local markdown is truncated after Proposition 1. Method, experimental, appendix, table, and figure details below were recovered from the official arXiv v5 PDF. Because only Figure 1 is present under `raw/assets`, confidence is medium rather than high despite the PDF recovery.

## Key Takeaways

- **GRPO is implicitly discriminative.** With binary rewards, its expected objective increases likelihood-ratio scores for positive answers and decreases them for negative answers.
- **Difficulty bias comes from a question weight.** Standard GRPO weights the per-question discriminative objective by $\sqrt{p(q)(1-p(q))}$, which approaches zero for very easy and very hard questions.
- **Dr. GRPO does not remove the bias completely.** Removing reward-standard-deviation normalization changes the weight to $p(q)(1-p(q))$, which still vanishes at both extremes.
- **DisCO removes question-difficulty weighting.** It directly optimizes positive-versus-negative score separation for each mixed group.
- **Clipping is replaced, not retuned.** DisCO uses unclipped log-likelihood or likelihood-ratio scoring to avoid vanishing gradients and entropy pathologies associated with clipping.
- **KL is enforced as a threshold.** A squared-hinge penalty is inactive inside the trust region and grows dynamically only after the estimated old-to-new KL exceeds $\delta$.
- **Hard-negative imbalance gets a dedicated objective.** Full DisCO applies a DRO/partial-AUC-style log-sum-exp over negative responses; DisCO-b is the unweighted pairwise base version.
- **The gains are consistent across 1.5B, 7B, and 8B distilled reasoning models.** DisCO variants lead every same-base 8k comparison in the main tables.

## GRPO Difficulty-Bias Derivation

Let $p(q)$ be the probability that the old policy answers question $q$ correctly. Under binary reward and normalized group advantage, the expected GRPO objective can be rewritten as:

$$
\mathbb E_q\sqrt{p(q)(1-p(q))}\,
\mathbb E_{o^+\sim\pi^+_{old},o^-\sim\pi^-_{old}}
[s^+_\theta(o^+,q)-s^-_\theta(o^-,q)].
$$

The term in brackets is a positive-versus-negative discriminative objective. The preceding weight is largest near $p=0.5$ and tends to zero near $p=0$ or $p=1$. Thus GRPO prioritizes medium-difficulty questions even when a rare correct answer to a hard question is especially informative.

For Dr. GRPO, the same analysis gives weight $p(q)(1-p(q))$. This is flatter in one sense but still suppresses both extremes.

![Figure 1: Difficulty weights, per-question accuracy distribution, and the effect of removing question weights on all-correct/all-wrong ratios.](<../../raw/assets/x1 37.png>)

## DisCO Objectives

### DisCO-b: Basic Pairwise Discrimination

For a scoring function $s_\theta(o,q)$ and proper surrogate $\ell$, DisCO-b maximizes:

$$
J_1(\theta)=\mathbb E_q\mathbb E_{o^+\sim\pi^+_{old},o^-\sim\pi^-_{old}}
\ell(s_\theta(o^+,q)-s_\theta(o^-,q)).
$$

The experiments use the identity surrogate with two score choices:

- **Log-L:** mean token log-likelihood under the current policy.
- **L-ratio:** mean token likelihood ratio $\pi_\theta/\pi_{old}$.

Both are unclipped and use one score definition for positives and negatives.

### DisCO: DRO Hard-Negative Weighting

When a group contains one positive and many negatives, ordinary AUC can be high even if one negative outranks the positive. DisCO replaces the uniform negative average with a log-sum-exp/DRO objective:

$$
J_2(\theta)=-\mathbb E_q\mathbb E_{o^+}\tau\log
\mathbb E_{o^-}\exp\left(\frac{s_\theta(o^-,q)-s_\theta(o^+,q)}{\tau}\right).
$$

Low temperature concentrates weight on the highest-scoring hard negatives. DisCO-b omits this hard-negative weighting.

### KL-Constrained Optimization

The desired constraint is:

$$
D_{KL}(\pi_{old}\Vert\pi_\theta)\le\delta.
$$

DisCO optimizes the squared-hinge penalized objective:

$$
J(\theta)-\beta[D_{KL}(\pi_{old}\Vert\pi_\theta)-\delta]^2_+.
$$

Unlike constant KL regularization, the penalty contributes no gradient while the constraint is satisfied. When violated, its effective weight increases with the amount of violation.

## Algorithm

For each training step:

1. Copy the current policy to $\pi_{old}$.
2. Sample a question batch and eight responses per question.
3. Partition each question's responses into positive and negative sets using binary verification.
4. Estimate old-to-current KL over sampled token paths.
5. Compute positive-score gradients and DRO-weighted negative-score gradients.
6. Add the squared-hinge KL-constraint gradient.
7. Update with AdamW over mini-batches.

Questions whose sampled group is entirely positive or entirely negative cannot form a discriminative pair and provide no DisCO objective for that iteration.

## Table 1: Method Comparison

| Method | Difficulty bias | Clipping | KL handling | Score | Handles imbalanced rollouts |
| --- | --- | --- | --- | --- | --- |
| GRPO | Yes | Yes | Regularization to reference | Clipped likelihood ratio | No |
| Dr. GRPO | Yes | Yes | None | Clipped likelihood ratio | No |
| DAPO | Yes | Yes | None | Clipped likelihood ratio | No |
| GPG | Yes | No | None | Log-likelihood | No |
| TRPA | No | No | Regularization to old policy | Log likelihood ratio | No |
| DisCO | **No** | **No** | **Constraint to old policy** | **Proper score** | **Yes** |

## Experimental Setup

- Training data: DeepScaleR Preview, about 40.3k math problems.
- Additional generalization data: DAPO-Math-17K.
- Models: DeepSeek-R1-Distill-Qwen 1.5B/7B and DeepSeek-R1-Distill-Llama 8B.
- Benchmarks: AIME 2024, AIME 2025, MATH500, AMC 2023, Minerva, OlympiadBench.
- Evaluation: pass@1 averaged over 16 sampled responses per question.
- Response length: 8k train/test for comparable method rows.
- Sampling: eight responses per question, temperature 0.6.
- Batch/mini-batch: 128/32.
- Steps: 1,400 for 1.5B; 1,000 for 7B/8B; evaluation every 200 steps; best checkpoint reported.
- DisCO constraint: $\delta=10^{-4}$, $\beta=10^3$.
- DRO temperature: $\tau=1$ for L-ratio, $10$ for log-L.
- Compute: eight 40GB A100 GPUs for 1.5B at about six minutes/step; eight 80GB H100 GPUs for 7B at about 6.5 minutes/step.

## Table 2: Qwen 1.5B on DeepScaleR

| Method | Train/Test MRL | AIME24 | AIME25 | MATH500 | AMC23 | Minerva | O-Bench | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenAI-o1-Preview | - | 0.400 | - | 0.814 | - | - | - | - |
| DS-Distill-Qwen-1.5B | 32k+/32k | 0.288 | 0.263 | 0.828 | 0.629 | 0.265 | 0.433 | 0.451 |
| DS-Distill-Qwen-1.5B | 32k+/8k | 0.181 | 0.215 | 0.758 | 0.515 | 0.237 | 0.353 | 0.376 |
| STILL-3-1.5B-preview | 29k/32k | 0.325 | 0.248 | 0.844 | 0.667 | 0.290 | 0.454 | 0.471 |
| DSR-1.5B-Preview | 24k/32k | 0.431 | 0.304 | 0.878 | 0.736 | 0.302 | 0.500 | 0.525 |
| DSR-1.5B-Preview | 24k/8k | 0.358 | 0.258 | 0.860 | 0.679 | 0.297 | 0.473 | 0.488 |
| GRPO | 8k/8k | 0.277 | 0.242 | 0.838 | 0.647 | 0.276 | 0.462 | 0.457 |
| GRPO-ER | 8k/8k | 0.298 | 0.242 | 0.839 | 0.649 | 0.279 | 0.452 | 0.460 |
| Dr. GRPO | 8k/8k | 0.252 | 0.238 | 0.831 | 0.631 | 0.268 | 0.440 | 0.443 |
| DAPO | 8k/8k | 0.310 | 0.252 | 0.848 | 0.675 | 0.296 | 0.456 | 0.473 |
| TRPA | 8k/8k | 0.354 | 0.235 | 0.835 | 0.653 | 0.283 | 0.458 | 0.470 |
| DisCO L-ratio | 8k/8k | 0.381 | 0.306 | **0.878** | 0.746 | 0.319 | **0.512** | 0.524 |
| DisCO log-L | 8k/8k | **0.404** | **0.317** | 0.876 | **0.758** | **0.333** | 0.509 | **0.533** |

## Table 3: Qwen 7B on DeepScaleR

| Method | Train/Test MRL | AIME24 | AIME25 | MATH500 | AMC23 | Minerva | O-Bench | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DS-Distill-Qwen-7B | 32k+/32k | 0.560 | 0.396 | 0.923 | 0.825 | 0.380 | 0.568 | 0.609 |
| DS-Distill-Qwen-7B | 32k+/8k | 0.402 | 0.292 | 0.873 | 0.688 | 0.355 | 0.471 | 0.513 |
| GRPO-LEAD-7B | 8k/8k | 0.470 | 0.345 | 0.893 | 0.748 | 0.372 | 0.500 | 0.555 |
| TRPA (published reference row) | 8k/8k | 0.570 | - | 0.870 | 0.780 | 0.360 | 0.550 | - |
| GRPO | 8k/8k | 0.498 | 0.394 | 0.916 | 0.807 | 0.381 | 0.555 | 0.592 |
| GRPO-ER | 8k/8k | 0.515 | 0.381 | 0.916 | 0.825 | 0.376 | 0.544 | 0.593 |
| Dr. GRPO | 8k/8k | 0.488 | 0.346 | 0.910 | 0.792 | 0.368 | 0.546 | 0.575 |
| DAPO | 8k/8k | 0.454 | 0.335 | 0.907 | 0.799 | 0.388 | 0.535 | 0.570 |
| TRPA | 8k/8k | 0.510 | 0.367 | 0.898 | 0.779 | 0.379 | 0.534 | 0.578 |
| DisCO L-ratio | 8k/8k | **0.583** | **0.421** | 0.923 | 0.852 | 0.399 | 0.585 | **0.627** |
| DisCO log-L | 8k/8k | 0.558 | 0.410 | **0.927** | **0.854** | **0.410** | **0.592** | 0.625 |

The table contains both an incomplete published TRPA reference row and the authors' complete same-run TRPA row.

## Table 4: Llama 8B on DeepScaleR

| Method | Train/Test MRL | AIME24 | AIME25 | MATH500 | AMC23 | Minerva | O-Bench | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DS-Distill-Llama-8B | 32k+/32k | 0.506 | 0.346 | 0.896 | 0.815 | 0.295 | 0.541 | 0.566 |
| DS-Distill-Llama-8B | 32k+/8k | 0.348 | 0.238 | 0.825 | 0.652 | 0.267 | 0.440 | 0.462 |
| GRPO | 8k/8k | 0.410 | 0.240 | 0.873 | 0.759 | 0.307 | 0.506 | 0.516 |
| GRPO-ER | 8k/8k | 0.408 | 0.277 | 0.882 | 0.785 | 0.311 | 0.511 | 0.529 |
| Dr. GRPO | 8k/8k | 0.423 | 0.285 | 0.867 | 0.786 | 0.300 | 0.497 | 0.526 |
| DAPO | 8k/8k | 0.333 | 0.308 | 0.879 | 0.794 | 0.325 | 0.522 | 0.527 |
| TRPA | 8k/8k | 0.454 | 0.279 | 0.864 | 0.756 | 0.289 | 0.518 | 0.527 |
| DisCO L-ratio | 8k/8k | 0.506 | **0.356** | **0.900** | 0.831 | 0.326 | 0.553 | 0.579 |
| DisCO log-L | 8k/8k | **0.523** | 0.354 | 0.896 | **0.843** | **0.331** | **0.560** | **0.584** |

## Table 5: Qwen 1.5B on DAPO-Math-17K

| Method | Train/Test MRL | AIME24 | AIME25 | MATH500 | AMC23 | Minerva | O-Bench | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DS-Distill-Qwen-1.5B | 32k+/32k | 0.288 | 0.263 | 0.828 | 0.629 | 0.265 | 0.433 | 0.451 |
| DS-Distill-Qwen-1.5B | 32k+/8k | 0.181 | 0.215 | 0.758 | 0.515 | 0.237 | 0.353 | 0.376 |
| GRPO | 8k/8k | 0.342 | 0.256 | 0.842 | 0.672 | 0.267 | 0.458 | 0.473 |
| GRPO-ER | 8k/8k | 0.290 | 0.260 | 0.852 | 0.681 | 0.287 | 0.463 | 0.472 |
| Dr. GRPO | 8k/8k | 0.300 | 0.250 | 0.849 | 0.705 | 0.292 | 0.464 | 0.477 |
| DAPO | 8k/8k | 0.275 | 0.229 | 0.812 | 0.653 | 0.256 | 0.441 | 0.444 |
| TRPA | 8k/8k | 0.346 | 0.279 | 0.836 | 0.683 | 0.281 | 0.450 | 0.479 |
| DisCO L-ratio | 8k/8k | 0.413 | 0.310 | **0.874** | **0.775** | 0.307 | 0.495 | 0.529 |
| DisCO log-L | 8k/8k | **0.460** | **0.317** | 0.873 | **0.775** | **0.320** | **0.502** | **0.541** |

## Table 6: Discriminative Reformulation of GRPO Variants

| Method | Question weight $\omega(q)$ | Surrogate | Positive/negative score form |
| --- | --- | --- | --- |
| GRPO | $\sqrt{p(1-p)}$ | Identity | Length-normalized clipped likelihood ratios, asymmetric for reward sign |
| Dr. GRPO | $p(1-p)$ | Identity | Token-summed clipped likelihood ratios |
| DAPO | $\sqrt{p(1-p)}$ | Identity | Token-normalized likelihood ratios with asymmetric clipping |
| GPG | Proportional to $p(1-p)$ | Identity | Token-normalized log-likelihood |
| TRPA | 1 | Log-sigmoid | Sequence log likelihood ratio to a reference model |
| DisCO | 1 | Proper discriminative surrogate | Shared unclipped L-ratio or log-L score; optional DRO hard-negative weighting |

## Training Dynamics and Ablations

The official PDF contains five additional figures not available in the local raw assets:

- **Figure 2:** reward and entropy curves for Qwen 1.5B/7B. GRPO, GRPO-ER, and Dr. GRPO collapse entropy; DAPO grows entropy excessively; TRPA becomes unstable; DisCO holds entropy near 0.22 while reward continues rising.
- **Figure 3:** DisCO beats DisCO-b; unclipped scores outperform clipped likelihood ratios at upper clip 0.2 and 0.28.
- **Figure 4:** constrained optimization beats constant KL regularization; performance is insensitive across tested $\tau$; removing hard-negative weighting, restoring question bias, using constant KL, or clipping each hurts.
- **Figure 5:** DisCO beats DisCO-b for both scores on Qwen 1.5B/7B.
- **Figure 6:** Llama 8B training dynamics repeat the entropy stability pattern.

These findings are described but not embedded because `raw/assets` contains only Figure 1 and `raw/` is immutable.

## Limitations

- **Binary rewards only.** The derivation and implemented pair construction assume correct/incorrect labels; continuous multi-objective driving rewards require a ranking or regression extension.
- **Mixed groups only.** DisCO needs at least one positive and one negative rollout. All-wrong hard questions and all-correct easy questions provide no pairwise signal and are removed/ignored, so “completely eliminates difficulty bias” does not solve missing-class groups.
- **Math-only evidence.** Results cover verifiable mathematical reasoning on distilled DeepSeek models, not multimodal reasoning, language-action models, or autonomous driving.
- **Best-checkpoint reporting.** Models are evaluated every 200 steps and the best result is reported; final-checkpoint robustness and selection variance are not given.
- **No seed uncertainty.** Tables report point estimates without repeated-run variance.
- **Baseline scope caveat.** DAPO dynamic sampling is intentionally omitted because it costs about 3× more sampling; the comparison isolates objectives but is not full DAPO.
- **High compute.** A 1.5B run uses eight 40GB A100s for roughly 140 hours at reported step time; 7B uses eight 80GB H100s for roughly 108 hours.
- **Hyperparameter dependence.** The trust threshold and large hinge coefficient are motivated empirically; broader sensitivity is limited.
- **Length normalization remains.** Both score definitions average over tokens, so DisCO may still encode response-length preferences even though it removes the GRPO question weight.
- **Local extraction is incomplete.** The raw markdown ends after Figure 1; Tables 1–6, Figures 2–6, the algorithm, experiments, and conclusion were recovered from the official PDF, but missing images cannot be embedded from `raw/assets`.

## Relevance to Autonomous Driving

The transfer is methodological, not empirical:

- [[concepts/gspo-vs-grpo.md]] — DisCO disputes the claim that Dr. GRPO fully removes question-level difficulty bias and introduces objective redesign rather than another normalization fix.
- [[concepts/r1-zero-like-training.md]] — separates support/data difficulty from optimizer-induced weighting under binary verification.
- [[concepts/discriminative-policy-optimization.md]] — formalizes positive/negative trajectory scoring, hard-negative emphasis, and trust-region constraints.
- [[concepts/rl-for-ad.md]] — suggests a candidate approach for binary safety-feasibility rewards, but continuous PDMS/EPDMS/RFS needs a ranking generalization.
