---
title: "All Roads Lead to Rome: Incentivizing Divergent Thinking in Vision-Language Models"
type: source-summary
sources: [raw/papers/All Roads Lead to Rome_ Incentivizing Divergent Thinking in Vision-Language Models.md]
related: [concepts/divergent-thinking-in-vlms.md, concepts/gspo-vs-grpo.md, concepts/best-of-n.md, concepts/chain-of-thought-for-ad.md, concepts/r1-zero-like-training.md]
created: 2026-06-18
updated: 2026-06-18
confidence: high
---

## Overview

This paper argues that GRPO-trained VLM reasoning models become **deeper but narrower**: they improve single-path accuracy, but lose the base model's ability to sample diverse alternative strategies. The authors call this **diversity collapse**. Under multiple attempts (`acc@k`), base models can sometimes solve more problems than RL models because they explore a wider reasoning space.

The proposed fix is **Multi-Group Policy Optimization (MUPO)**, a drop-in GRPO replacement that clusters sampled responses into multiple reasoning groups, computes localized advantages per group, and adds an accuracy-gated diversity reward between groups.

**Project**: https://xytian1008.github.io/MUPO/  
**arXiv**: https://arxiv.org/html/2604.00479v1  
**Organizations**: Australian National University, Shanghai AI Lab, GE Research

---

## Figures

![[raw/assets/x10 8.png]]

**Figure 1**: Failure cases for RL VLMs. Vision-R1 repeatedly follows a narrow strategy, while Qwen2.5-VL-7B samples alternative reasoning routes that can succeed.

![[raw/assets/x11 5.png]]

**Figure 2**: `acc@k` and reasoning-diversity correlation. RL models are stronger at `k=1`, but base models gain more with multiple samples; higher diversity correlates with higher `acc@4`.

![[raw/assets/x12 4.png]]

**Figure 3**: GRPO diversity collapse. Reasoning diversity drops sharply early in training.

![[raw/assets/x13 7.png]]

**Figure 4**: t-SNE reasoning embeddings. RL models cluster tightly; base models cover broader strategy regions.

![[raw/assets/x14 3.png]]

**Figure 5**: MUPO overview. Responses are partitioned into multiple groups, each optimized with a GRPO-like objective plus diversity reward.

![[raw/assets/x15 3.png]]

**Figure 6**: MUPO reasoning embeddings. MUPO produces broader, multimodal reasoning clusters than GRPO.

![[raw/assets/x16 1.png]]

**Figure 7**: Qualitative comparison on MMStar. MUPO uses reference objects for spatial estimation, while Vision-R1 follows a brittle layer-by-layer estimate.

![[raw/assets/x17 1.png]]

**Figure 8**: Learning curves for accuracy and diversity reward.

![[raw/assets/x18 1.png]]

**Figure 9**: Ablation over initial/final diversity-reward weights.

---

## Core Finding

The paper distinguishes **sequential depth** from **parallel breadth**:

- RL models: better `acc@1`, deeper reasoning, but often repeat a dominant strategy.
- Base models: less refined per path, but sample broader reasoning strategies.
- Parallel test-time scaling depends on breadth; if every sample follows the same wrong strategy, extra samples add little.

The authors measure diversity by extracting reasoning segments, embedding them with Qwen3-Embedding-0.6B, and computing average pairwise cosine distance.

---

## GRPO Diversity Collapse

The paper trains Qwen2.5-VL-3B and Qwen2.5-VL-7B with GRPO on ViRL39K. For each validation step, it samples 10 responses and tracks reasoning diversity.

Observed behavior:
- Diversity falls sharply in the first 20 steps.
- The model has seen little data when this happens, so the collapse is not just mature convergence.
- Training then refines a narrow subset of strategies for most of the run.
- This causes local optima and weak parallel test-time scaling.

This is a distinct GRPO failure mode from [[sources/understanding-r1-zero-like-training.md]]:
- Dr. GRPO addresses length and per-question reward-std bias.
- MUPO addresses strategy-diversity collapse across sampled responses.

---

## MUPO Method

MUPO samples `N` responses and partitions them into `K` groups in reasoning-embedding space. It uses constrained clustering with a minimum group size `G_min`, so each group approximates a reasoning mode.

The objective is a weighted sum of per-group GRPO-style objectives. Larger groups are prevented from dominating through a load-balance weight:

$$
w_k = \left(\frac{N}{K|G_k|}\right)^\beta
$$

The reward combines accuracy, format, and an accuracy-gated diversity term:

$$
R_i^k = R_{acc} + R_{fmt} + \lambda \cdot 1[R_{acc}=1] \cdot R_{div}
$$

The accuracy gate is important: it prevents the model from chasing diverse but wrong outputs.

The diversity weight follows a cosine annealing schedule from `lambda_max` to `lambda_min`, encouraging exploration early and exploitation later.

Default training settings:
- Base models: Qwen2.5-VL-3B and Qwen2.5-VL-7B
- Dataset: ViRL39K
- Training: 2 epochs, learning rate 1e-6
- Responses per example: `N=15`
- Groups: `K=3`
- Minimum group size: `G_min=3`
- Load-balance exponent: `beta=1`
- Diversity weight: `lambda_max=0.4`, `lambda_min=0.1`
- Sampling temperature: 1.0

---

## Table 1: Mathematical Benchmarks

Scores are `acc@1` / `acc@4`.

| Model | MathVerse | MathVista | MathVision | LogicVista | WeMath | Geometry3K | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-5-Thinking* | 81.2 / 85.5 | 81.9 / 86.1 | 72.0 / 79.2 | 70.0 / 81.5 | 71.1 / 78.4 | 79.9 / 84.3 | 76.1 / 82.5 |
| Gemini-2.5-Pro* | 76.9 / 79.3 | 80.9 / 85.2 | 69.1 / 72.5 | 73.8 / 76.4 | 78.0 / 82.7 | 77.2 / 80.1 | 76.0 / 79.4 |
| Qwen2.5-VL-7B | 40.7 / 55.2 | 62.3 / 78.5 | 23.2 / 41.6 | 42.6 / 59.3 | 33.1 / 50.1 | 38.5 / 54.4 | 40.1 / 56.5 |
| InternVL2.5-8B* | 34.5 / 42.3 | 68.2 / 72.8 | 25.6 / 29.4 | 38.3 / 44.7 | 38.6 / 43.5 | 44.8 / 48.0 | 41.7 / 46.8 |
| R1-OneVision-7B | 46.4 / 49.9 | 64.1 / 69.4 | 29.9 / 34.1 | 45.6 / 52.5 | 44.6 / 47.9 | 46.1 / 50.4 | 46.1 / 50.7 |
| VLAA-Thinker-7B | 48.2 / 51.6 | 68.0 / 70.8 | 26.4 / 30.3 | 48.5 / 54.5 | 41.5 / 46.8 | 50.6 / 55.2 | 47.2 / 51.5 |
| Vision-R1-7B | 52.4 / 56.1 | 73.5 / 75.6 | 28.2 / 32.9 | 49.7 / 53.8 | 41.6 / 44.3 | 49.0 / 54.1 | 49.1 / 52.8 |
| MUPO-Thinker-3B | 41.3 / 50.8 | 64.3 / 72.5 | 27.8 / 35.4 | 42.8 / 50.3 | 36.5 / 46.8 | 45.1 / 52.9 | 43.0 / 51.5 |
| **MUPO-Thinker-7B** | **53.9 / 61.7** | **77.9 / 82.4** | **31.3 / 39.7** | **50.6 / 61.5** | **44.1 / 48.6** | **52.1 / 59.1** | **51.6 / 58.8** |

`*` indicates results sourced from OpenCompass.

---

## Table 2: General-Purpose Benchmarks

All models are 7B-scale. Scores are `acc@1` / `acc@4`.

| Model | MMStar | HallBench | MMVet | Avg. |
| --- | --- | --- | --- | --- |
| QwenVL | 59.2 / 74.8 | 50.0 / 65.6 | 64.8 / 74.1 | 58.0 / 71.5 |
| InternVL | 63.2 / 70.6 | 49.0 / 58.3 | 62.8 / 69.0 | 58.3 / 66.0 |
| R1-OV | 64.7 / 67.5 | 52.5 / 57.6 | 65.2 / 67.3 | 60.8 / 64.1 |
| VLAA | 66.1 / 69.1 | 54.7 / 56.9 | 70.0 / 72.7 | 63.6 / 66.2 |
| V-R1 | 66.3 / 68.9 | 55.4 / 59.0 | 68.2 / 70.5 | 63.3 / 66.1 |
| **MUPO** | **68.7 / 75.4** | **57.5 / 63.6** | **70.6 / 78.2** | **65.6 / 72.4** |

---

## Table 3: 3B-Scale Comparison

| Model | Params | Math acc@1 | Math acc@4 | General acc@1 | General acc@4 | Avg. acc@1 | Avg. acc@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| QwenVL | 3B | 33.4 | 47.0 | 51.3 | 62.3 | 40.0 | 52.1 |
| InternVL | 2B | 26.5 | 34.4 | 53.1 | 58.9 | 35.3 | 42.6 |
| VLM-R1 | 4B | 37.8 | 42.3 | 53.9 | 57.7 | 43.2 | 47.4 |
| VLAA | 3B | 39.5 | 43.3 | 55.1 | 59.3 | 44.7 | 48.6 |
| LMM-R1 | 4B | 41.5 | 45.5 | 55.4 | 59.2 | 46.1 | 50.1 |
| **MUPO** | **3B** | **43.5** | **51.5** | **57.8** | **65.2** | **48.3** | **56.0** |

---

## Table 4: Number of Groups

| K | MathVerse | MathVista | MathVision | MMStar | HallBench | Average |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 46.9 | 69.1 | 24.1 | 64.8 | 54.7 | 51.9 |
| 2 | 49.4 | 72.3 | 28.0 | 65.2 | 56.7 | 54.3 |
| 3 | 51.2 | 74.1 | 29.3 | 65.8 | 56.5 | 55.4 |
| 4 | 50.9 | 74.6 | 29.4 | 65.1 | 56.3 | 55.3 |
| 5 | 50.6 | 74.8 | 29.1 | 64.5 | 55.9 | 55.0 |

`K=1` reduces to GRPO. Accuracy peaks at `K=3`; math tasks prefer more diversity than general tasks.

---

## Results

MUPO-Thinker-7B improves over the previous best open-source RL VLM:
- Mathematical average `acc@1`: 49.1 -> 51.6 (+2.5)
- Mathematical average `acc@4`: 52.8 -> 58.8 (+6.0)
- General average `acc@1`: 63.3 -> 65.6 (+2.3)
- General average `acc@4`: 66.2 -> 72.4 (+6.2)

The larger `acc@4` gains are the key result: MUPO improves not just single-answer quality but parallel test-time scaling.

---

## Relationship to AD Wiki

This is not an autonomous-driving paper. Its relevance is methodological:

- It reinforces the wiki's existing concern that GRPO can collapse exploration, but at the **reasoning-strategy** level rather than trajectory-output level.
- It explains why stochastic Best-of-N can fail when samples are correlated around one local optimum.
- It is conceptually close to Curious-VLA's narrow-policy diagnosis, but operates on VLM reasoning paths rather than driving trajectories.
- It is complementary to Dr. GRPO: Dr. GRPO fixes normalization bias; MUPO tries to preserve multi-modal solution search.

---

## Limitations

1. **No AD validation**: results are VLM reasoning benchmarks, not NAVSIM, Bench2Drive, or driving trajectory tasks.
2. **Embedding-dependent diversity metric**: reasoning diversity is measured through Qwen3-Embedding-0.6B cosine distance, so clustering quality depends on that representation.
3. **Higher rollout cost**: MUPO samples `N=15` responses per example, more expensive than common GRPO group sizes.
4. **Reward hacking still possible**: the accuracy gate reduces but does not eliminate the risk that diversity rewards optimize superficial phrasing differences.
5. **Table provenance mixed**: some baselines are taken from papers or OpenCompass rather than fully reproduced under one evaluation harness.
6. **Limited training data disclosure in raw markdown**: ViRL39K is named, but the local markdown does not provide detailed data composition beyond the citation.

---

## Key Takeaways

1. GRPO can trade breadth for depth: better greedy answers, worse strategy diversity.
2. Base models can be stronger under parallel sampling because they retain alternative solution paths.
3. MUPO preserves multiple reasoning modes by clustering responses and optimizing each group locally.
4. Parallel test-time scaling depends on sample diversity, not just sample count.
5. For AD, this suggests future GRPO/RFT work should track whether samples represent genuinely different driving/reasoning modes or just local variants of the same policy.
